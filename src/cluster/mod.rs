//! `--head` / `--node` cluster orchestration for 2-node TP=2 (Stage 3).
//!
//! Goal (user): launch one binary per box, zero manual config or model copy. The **head** owns the
//! model; it auto-discovers **node**s, checks binary compatibility, and ships whatever artifacts the
//! node is missing. A **content-addressed cache** means a node that already has the files (or a re-run)
//! transfers nothing.
//!
//! Control plane = normal network (UDP discovery + TCP sync). RDMA (`net.rs`) is reserved for the
//! inference data plane only — bootstrap never depends on verbs, so recovery stays simple.
//!
//! MVP scope: whole-model distribution (per-rank shard distribution is a later optimization once the
//! G-D weight sharding exists). After sync the node has an assembled model dir ready for the TP run.
//!
//! File map (the former single `cluster.rs`; every item keeps its `crate::cluster::` path via `pub use self::<sub>::*`;
//! this file: the protocol, messages, space check, content-addressed cache, manifests, control-socket hardening, node watch):
//! - `discovery.rs`: UDP discovery, RoCE interfaces, rank assignment and topology
//! - `node.rs`: blob streaming, model-dir assembly and the node side (`node_handle`, `run_node`)
//! - `head.rs`: the head side: discover-and-push (`run_head*`, `head_sync_one`, `head_ship`)
//! - `models.rs`: blob-cache management CLI (list / remove cached models)
//! - one file per former inline `#[cfg(test)]` module, named after it (`tests`, `ipv4_tests`, `pack_manifest_tests`, `build_id_tests`, `tpship_tests`).


use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket, SocketAddr, IpAddr, Ipv4Addr};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use crate::boot_report::{Progress, Unit};

const PROTOCOL_VERSION: u32 = 6;   // v6 (S-B9-CF-AGREE16/16B): every pairwise lockstep word changed (the
                                     //     verify width is a 5-bit field of its own, the digest folded to 27
                                     //     bits, the step-go fence an FNV of (step, events)) and the hub RND
                                     //     frame carries the real accept / width. No frame changed size, so
                                     //     this number is what tells a v5 pair apart from a v6 one by NAME
                                     //     (the source stamp differs too, but only says "another build").
                                     // v5: + DraftManifest/DraftReady — the DFlash2 drafter rides the sync
                                     //     into the node's blob cache (v4: per-epoch payload lengths)
const DISCOVERY_PORT: u16 = 29499;          // UDP; TCP control plane defaults to 29500 (--port)
const DISCOVERY_MAGIC: &str = "GB10-TP-DISCOVER";
/// Binary-compat token: same compiled kernels + same Rust-side sources + protocol => same wire
/// behavior. Cheaper than hashing the 15 MB executable each launch, and it is exactly what must
/// match across boxes. `-k` covers the kernels (KERNEL_BUILD_ID), `-r` the Rust sources + C shim
/// (SOURCE_BUILD_ID): the sharders/protocol/scheduler change behavior without touching a .cu.
///
/// It is a SOURCE stamp on purpose, not the executable's sha256: two machines that build the same
/// commit must pair, and their binaries need not be byte-identical. What `-r` covers is the file
/// list `build.rs` bakes in (`SOURCE_BUILD_FILES`): every `.rs` under src/ at any depth except
/// src/bin and src/ptx, the included data files, native/net_shim.c, Cargo.lock, Cargo.toml and
/// build.rs. `build_id_tests` walks the tree and fails on an engine file outside both stamps.
fn binary_version() -> String {
    format!("v{}-k{}-r{}", PROTOCOL_VERSION, env!("KERNEL_BUILD_ID"), env!("SOURCE_BUILD_ID"))
}

/// The one comparison both ends of the Hello make. `Err` carries the message a refused pair prints
/// (and the node sends to the head): it names both tokens, so the differing part (`v` protocol,
/// `k` kernels, `r` sources) can be read off the log.
fn version_match(head: &str, node: &str) -> std::result::Result<(), String> {
    if head == node { Ok(()) } else { Err(format!("binary mismatch: head {head} vs node {node}")) }
}

// ---------------------------------------------------------------------------------------------------
// Wire protocol
// ---------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Artifact {
    pub logical: String,   // path relative to the model dir, e.g. "config.json"
    pub hash: String,      // sha256 hex
    pub size: u64,
}

#[derive(Serialize, Deserialize, Debug)]
enum Msg {
    Hello { version: String, role: String, hostname: String },
    /// `draft` (owner 2026-11-02, the drive-space check): the drafter files this session will offer after `Config`,
    /// so the node's space report covers everything the session would store on it. Empty = no drafter.
    Manifest { model_id: String, artifacts: Vec<Artifact>, #[serde(default)] draft: Vec<Artifact> },
    /// `space`: the node's drive-space report, sent with the MODEL round's reply (None on the drafter round).
    Missing { hashes: Vec<String>, #[serde(default)] space: Option<SpaceReport> },
    /// Velo-Fabric bulk RDMA negotiation (sent once before the blob loop whenever `missing` is non-empty).
    /// `enabled == false` (or `BulkReply { ok: false, .. }`) falls back cleanly to TCP `send_blob`/`recv_blob`.
    BulkOffer { enabled: bool, ep: crate::net::bulk::BulkEndpoint },
    BulkReply { ok: bool, ep: crate::net::bulk::BulkEndpoint },
    BlobHeader { hash: String, size: u64 },
    Ready { model_dir: String },
    Config(crate::tp::TpConfig),
    /// v5 — the DFlash2 draft artifact round, ALWAYS sent right after `Config` (empty artifacts
    /// = "no draft this session") so both sides stay in strict lockstep without the node having
    /// to guess from its config. The node answers `Missing` (shared with the model round), the
    /// head streams `BlobHeader`+bytes, and the node answers `DraftReady` with the assembled
    /// cache path. Nodes therefore NEVER need a local draft copy (owner gap fix 2026-08-23).
    DraftManifest { model_id: String, artifacts: Vec<Artifact> },
    DraftReady { model_dir: String },
    Error { msg: String },
}

/// Length-prefixed JSON framing, shared by the sync protocol (`Msg`) and, once a session is
/// retained, by the TP serving control plane (`tp_serve::ServingMsg`).
pub(crate) fn send_json<T: Serialize>(w: &mut impl Write, m: &T) -> Result<()> {
    let b = serde_json::to_vec(m)?;
    w.write_all(&(b.len() as u32).to_be_bytes())?;
    w.write_all(&b)?;
    w.flush()?;
    Ok(())
}
pub(crate) fn recv_json<T: serde::de::DeserializeOwned>(r: &mut impl Read) -> Result<T> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len)?;
    let n = u32::from_be_bytes(len) as usize;
    if n > 64 * 1024 * 1024 { bail!("control message too large ({n} B)"); }
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    Ok(serde_json::from_slice(&b)?)
}

fn send_msg(w: &mut impl Write, m: &Msg) -> Result<()> { send_json(w, m) }
fn recv_msg(r: &mut impl Read) -> Result<Msg> { recv_json(r) }

// ---------------------------------------------------------------------------------------------------
// Drive space (owner 2026-11-02): the head checks every node BEFORE the first byte ships
// ---------------------------------------------------------------------------------------------------

/// What a node reports with its model-round `Missing`: the bytes this session would store in its blob cache (the
/// model shards it does not have yet + the drafter files announced in `Manifest::draft` it does not have yet), the
/// free bytes of the drive that holds that cache, the cache dir and the node's hostname.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub(crate) struct SpaceReport { pub need: u64, pub free: u64, pub dir: String, pub host: String }

/// Room left on a node's drive on top of what a session stores there: a sync never fills a drive to the last byte.
pub(crate) const SHIP_HEADROOM: u64 = 1_000_000_000;

/// Free bytes for an unprivileged writer on the drive that holds `p` (its nearest existing ancestor), None when the
/// drive cannot be asked.
fn free_bytes(p: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let mut q = p.to_path_buf();
    while !q.exists() {
        if !q.pop() { return None; }
    }
    let c = std::ffi::CString::new(q.as_os_str().as_bytes()).ok()?;
    // SAFETY: a valid NUL-terminated path and a zeroed out-struct the call fills.
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 { return None; }
    Some(st.f_bavail as u64 * st.f_frsize as u64)
}

fn gb(b: u64) -> String { format!("{:.1} GB", b as f64 / 1e9) }

/// One line per node whose drive cannot hold what this session would store there plus `SHIP_HEADROOM` (empty = every
/// node has room). `rows` = (rank, the head's name for the node, its report); a node that sent no report (its drive
/// could not be asked) is not judged here: shipping then fails the way it always did if the drive fills.
fn space_shortfalls(rows: &[(i32, String, Option<SpaceReport>)]) -> Vec<String> {
    rows.iter().filter_map(|(rank, name, r)| {
        let r = r.as_ref()?;
        let need = r.need.saturating_add(SHIP_HEADROOM);
        if r.need == 0 || r.free >= need { return None; }
        let who = if r.host.is_empty() || r.host == *name { name.clone() } else { format!("{} ({name})", r.host) };
        Some(format!("rank {rank}, node {who}: needs ~{} ({} of model files it does not have yet + {} headroom), has {} free \
on the drive of {} — short by ~{}", gb(need), gb(r.need), gb(SHIP_HEADROOM), gb(r.free), r.dir, gb(need - r.free)))
    }).collect()
}

/// The head's refusal: why it stops, per short node how much space is needed, and what to do.
fn space_refusal(short: &[String], nodes: usize) -> String {
    format!("not enough drive space on {} of {nodes} TP node(s) for the model files this start would ship to them. Nothing \
was shipped; the head stops here.\n  {}\nTo start: free that much space on each node listed (its TP model cache: run \
`gb10_inference --cached-models-list` on the node, then `--cached-models-remove <ID>` for models you no longer use), or \
start the node with --tp-cache <a directory on a drive with room>.", short.len(), short.join("\n  "))
}

// ---------------------------------------------------------------------------------------------------
// Content-addressed cache
// ---------------------------------------------------------------------------------------------------

fn cache_root() -> PathBuf {
    crate::opts::var(crate::opt!("tp-cache")).map(PathBuf::from).unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(home).join(".cache/gb10_tp")
    })
}
fn blob_path(hash: &str) -> PathBuf { cache_root().join("blobs").join(hash) }
fn have_blob(hash: &str) -> bool { blob_path(hash).exists() }

/// Atomically publish `tmp` (already hash-verified) into the content store as `blobs/<hash>`.
fn publish_blob(hash: &str, tmp: &Path) -> Result<()> {
    let dst = blob_path(hash);
    std::fs::create_dir_all(dst.parent().unwrap())?;
    std::fs::rename(tmp, &dst).with_context(|| format!("publish blob {hash}"))?;
    Ok(())
}

/// A file's sha256 and size; `prog` counts every chunk into a boot progress line (BASICS, owner 2026-11-02).
fn sha256_file_with(path: &Path, prog: Option<&Progress>) -> Result<(String, u64)> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut h = crate::net::bulk::FastSha256::new();
    let mut buf = vec![0u8; 4 << 20];
    let mut total = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        h.update(&buf[..n]);
        total += n as u64;
        if let Some(p) = prog { p.add(n as u64); }
    }
    Ok((h.finalize_hex(), total))
}

/// BASICS: the bytes `files` would hash now (the cache answers the rest), for the hashing percentage.
fn uncached_bytes(files: &[PathBuf], cache: &HashMap<String, String>) -> Result<u64> {
    let mut n = 0u64;
    for p in files {
        if !cache.contains_key(&cached_key(p)?) { n += std::fs::metadata(p)?.len(); }
    }
    Ok(n)
}

/// BASICS: a stream that counts the bytes it carries into a boot progress line (model shipping, both ends).
struct Counted<'a, S> { inner: &'a mut S, prog: &'a Progress }
impl<S: Write> Write for Counted<'_, S> {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(b)?;
        self.prog.add(n as u64);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> { self.inner.flush() }
}
impl<S: Read> Read for Counted<'_, S> {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(b)?;
        self.prog.add(n as u64);
        Ok(n)
    }
}
fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{x:02x}")).collect() }

/// Per-file (path, mtime, size) -> hash cache so re-launches don't re-hash a 15 GB model.
fn hash_cache_load() -> HashMap<String, String> {
    let p = cache_root().join("hashcache.json");
    std::fs::read(&p).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}
fn hash_cache_save(m: &HashMap<String, String>) {
    let p = cache_root().join("hashcache.json");
    let _ = std::fs::create_dir_all(p.parent().unwrap());
    if let Ok(b) = serde_json::to_vec(m) { let _ = std::fs::write(&p, b); }
}
fn cached_key(path: &Path) -> Result<String> {
    let md = std::fs::metadata(path)?;
    let mtime = md.modified()?.duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    Ok(format!("{}|{}|{}", path.display(), mtime, md.len()))
}

// ---------------------------------------------------------------------------------------------------
// Manifest (head side)
// ---------------------------------------------------------------------------------------------------

/// Files a node needs to serve a model. Follows a symlinked model dir to the real files.
fn model_files(dir: &Path) -> Result<Vec<PathBuf>> {
    // Recursive walk: a bundle may carry required subdirs the loader reads (DSV4's
    // `inference/config.json`, `encoding/`). Top-level-only shipping starves the node of those and
    // it crashes at load. Symlinks followed; editor/OS cruft + our sidecars skipped at every level.
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).with_context(|| format!("read_dir {}", d.display()))? {
            let e = entry?;
            let p = e.path();
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') || name.ends_with(".tmp") { continue; }
            let meta = std::fs::metadata(&p)?; // follows symlinks
            if meta.is_file() {
                out.push(p);
            } else if meta.is_dir() {
                stack.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

pub fn build_manifest(model_dir: &Path, world: u32) -> Result<(String, Vec<Vec<Artifact>>)> {
    let world = world.max(1) as usize;
    let mut cache = hash_cache_load();
    let mut dirty = false;
    // Per-rank artifact lists indexed by rank. rank 0 is the head's own shard and is never shipped;
    // ranks 1..world-1 are each node's manifest. When `rank{r}/` exists the head ships ONLY that
    // node's shard (+ the root inference/config.json + root config.json, exactly the world==2 set);
    // when a rank dir does NOT exist, that rank receives the whole model (replicated, the P2
    // replicate-if-not-divisible rule). For world==2 this is byte-identical to the pre-P4 single
    // `rank1/` manifest.
    let mut per_rank: Vec<Vec<Artifact>> = vec![Vec::new(); world];
    let sharded: Vec<bool> = (1..world)
        .map(|r| model_dir.join(format!("rank{r}")).exists())
        .collect();
    let files = model_files(model_dir)?;
    let prog = Progress::new("hashing the model files (first use; cached afterwards)", uncached_bytes(&files, &cache)?, Unit::Bytes);
    for path in files {
        let rel = path.strip_prefix(model_dir).unwrap_or(&path).to_string_lossy().to_string();
        let key = cached_key(&path)?;
        let (hash, size) = if let Some(h) = cache.get(&key) {
            (h.clone(), std::fs::metadata(&path)?.len())
        } else {
            let (h, sz) = sha256_file_with(&path, Some(&prog))?;
            cache.insert(key, h.clone());
            dirty = true;
            (h, sz)
        };
        let artifact = Artifact { logical: rel.clone(), hash, size };
        for r in 1..world {
            if include_for_rank(&rel, r, sharded[r - 1]) {
                per_rank[r].push(artifact.clone());
            }
        }
    }
    prog.finish();
    if dirty { hash_cache_save(&cache); }
    let model_id = model_dir.file_name().map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".into());
    Ok((model_id, per_rank))
}

// ---------------------------------------------------------------------------------------------------
// B32: per-rank shard shipping (always on for the models the plan covers; PLAN/B32_SHARD_SHIPPING_DESIGN.md)
// ---------------------------------------------------------------------------------------------------

/// One node rank's shipment: its cache model id, its artifacts and where each artifact's bytes come from.
pub struct RankShip {
    pub model_id: String,
    pub artifacts: Vec<Artifact>,
    pub srcs: HashMap<String, crate::shard_plan::PlanSrc>,
}

/// A `Write` that only hashes (and counts) — a synthesized segment's sha256 without a copy on disk.
struct HashWriter<'a> { h: crate::net::bulk::FastSha256, n: u64, prog: Option<&'a Progress> }
impl Write for HashWriter<'_> {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.h.update(b);
        self.n += b.len() as u64;
        if let Some(p) = self.prog { p.add(b.len() as u64); }
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

fn hash_plan_src(src: &crate::shard_plan::PlanSrc, prog: Option<&Progress>) -> Result<String> {
    use crate::shard_plan::PlanSrc;
    Ok(match src {
        PlanSrc::Whole(p) => sha256_file_with(p, prog)?.0,
        PlanSrc::Segment(ps) => {
            let mut w = HashWriter { h: crate::net::bulk::FastSha256::new(), n: 0, prog };
            crate::shard_plan::write_segment(ps, &mut w)?;
            w.h.finalize_hex()
        }
        PlanSrc::Bytes(b) => {
            if let Some(p) = prog { p.add(b.len() as u64); }
            let mut h = Sha256::new(); h.update(b); hex(&h.finalize())
        }
        PlanSrc::Generated(g) => {
            let h = match g.known_hash() { Some(h) => h, None => g.hash()? };
            if let Some(p) = prog { p.add(g.size()); }
            h
        }
    })
}

/// The hash-cache key of a planned file: whole files by (canonical path, mtime, size) — the same key
/// `pack_manifest` uses — and segments by `synth|<sha256 of the recipe>` (sources' keys + byte ranges + plan
/// version), so only the FIRST ship hashes; generated bytes are tiny and hashed every time (None).
fn plan_src_key(src: &crate::shard_plan::PlanSrc) -> Result<Option<String>> {
    use crate::shard_plan::PlanSrc;
    Ok(match src {
        PlanSrc::Whole(p) => Some(canonical_key(p)?),
        PlanSrc::Segment(ps) => {
            let mut h = Sha256::new();
            h.update(crate::shard_plan::segment_recipe(ps)?.as_bytes());
            Some(format!("synth|{}", hex(&h.finalize())))
        }
        PlanSrc::Bytes(_) => None,
        PlanSrc::Generated(g) => Some(g.key()),
    })
}

/// Optional pack-bundled `shard_hashes.json` (format `"velogb10-shard-hashes"`, v1): lets a published
/// model pack ship precomputed file and per-rank segment SHA-256s so first-time cluster boots skip the
/// multi-hundred-GB local hashing pass. Verified against `model.safetensors.index.json`'s SHA-256 and
/// every listed shard's byte size; receiving nodes still verify every received blob's SHA-256 on wire.
/// Packs without `shard_hashes.json` (or with mismatched metadata) fall back silently to local hashing.
#[derive(serde::Deserialize)]
struct PackHashEntry {
    size: u64,
    sha256: String,
}

#[derive(serde::Deserialize)]
struct PackShardHashes {
    format: String,
    version: u32,
    plan_version: u32,
    index_sha256: String,
    #[serde(default)]
    files: HashMap<String, PackHashEntry>,
    #[serde(default)]
    segments: HashMap<String, PackHashEntry>,
}

fn load_pack_shard_hashes(model_dir: &Path) -> Option<PackShardHashes> {
    let raw = std::fs::read(model_dir.join("shard_hashes.json")).ok()?;
    let h: PackShardHashes = serde_json::from_slice(&raw).ok()?;
    if h.format != "velogb10-shard-hashes" || h.version != 1 || h.plan_version != crate::shard_plan::PLAN_VERSION {
        return None;
    }
    let idx_bytes = std::fs::read(model_dir.join("model.safetensors.index.json")).ok()?;
    let got_idx = hex(&Sha256::digest(&idx_bytes));
    if got_idx != h.index_sha256 {
        return None;
    }
    for (rel, ent) in &h.files {
        let p = model_dir.join(rel);
        if p.exists() {
            let Ok(md) = std::fs::metadata(&p) else { return None };
            if md.len() != ent.size {
                return None;
            }
        }
    }
    Some(h)
}

/// B32: the shipments of ranks 1..world (index 0 = the head, empty) from `shard_plan`, hashed through the
/// shared hash cache (uncached files in parallel, largest first). Returns per-rank artifacts and sources.
pub fn ship_plans(model_dir: &Path, world: u32) -> Result<Vec<RankShip>> {
    ship_plans_for(model_dir, world, &(1..world as usize).collect::<Vec<_>>())
}

/// `ship_plans` for a chosen set of node ranks (DSV4.1 `--dsv41-ship` syncs one node): the other ranks' entries are
/// empty placeholders, so `ships[rank]` indexes the same way.
pub fn ship_plans_for(model_dir: &Path, world: u32, ranks: &[usize]) -> Result<Vec<RankShip>> {
    use crate::shard_plan::{detect_format, shard_plan, RankPlan, ShardFormat};
    let world = world as usize;
    let fmt = detect_format(model_dir).with_context(|| format!(
        "shard shipping: {} is neither an EXL3 pack nor a qwen4_exp NVFP4 model — no shard plan", model_dir.display()))?;
    let deal = match fmt { ShardFormat::Exl3 | ShardFormat::Fp4l => crate::exl3_forward::xtp::ep_deal_kind(), ShardFormat::Nvfp4 => "contig".to_string(),
                           ShardFormat::Naive => crate::naive::shipplan::deal(world), ShardFormat::Dsv41 => crate::dsv41::load::ship::deal() };
    let base = model_dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "model".into());
    let mut plans = Vec::new();
    for &r in ranks { plans.push(shard_plan(model_dir, fmt, world, r, &deal)?); }
    let pack_hashes = load_pack_shard_hashes(model_dir);
    let canon_dir = std::fs::canonicalize(model_dir).unwrap_or_else(|_| model_dir.to_path_buf());
    // distinct cacheable sources, hashed once
    let mut cache = hash_cache_load();
    let mut todo: Vec<(String, String, crate::shard_plan::PlanSrc, u64)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut pack_hits = 0usize;
    for p in &plans {
        for f in &p.files {
            if let Some(k) = plan_src_key(&f.src)? {
                if !cache.contains_key(&k) {
                    let sz = RankPlan::file_size(f)?;
                    if let Some(ref ph) = pack_hashes {
                        let pre = match &f.src {
                            crate::shard_plan::PlanSrc::Segment(_) => {
                                let sk = format!("{}|w{}|r{}|{}", deal, p.world, p.rank, f.logical);
                                ph.segments.get(&sk).filter(|e| e.size == sz && e.sha256.len() == 64)
                            }
                            crate::shard_plan::PlanSrc::Whole(path) => {
                                let cp = std::fs::canonicalize(path).unwrap_or_else(|_| path.clone());
                                cp.strip_prefix(&canon_dir).ok()
                                    .and_then(|rel| ph.files.get(rel.to_string_lossy().as_ref()))
                                    .filter(|e| e.size == sz && e.sha256.len() == 64)
                            }
                            _ => None,
                        };
                        if let Some(ent) = pre {
                            cache.insert(k.clone(), ent.sha256.clone());
                            pack_hits += 1;
                            continue;
                        }
                    }
                    if seen.insert(k.clone()) {
                        todo.push((f.logical.clone(), k, f.src.clone(), sz));
                    }
                }
            }
        }
    }
    if pack_hits > 0 {
        hash_cache_save(&cache);
    }
    if !todo.is_empty() {
        // Resolve pre-known hashes (e.g. Engram manifest sha256s) first so they don't skew the hashing progress bar.
        let mut work_todo: Vec<(String, String, crate::shard_plan::PlanSrc, u64)> = Vec::with_capacity(todo.len());
        let mut known_count = 0usize;
        for item in todo {
            if let crate::shard_plan::PlanSrc::Generated(g) = &item.2 {
                if let Some(h) = g.known_hash() {
                    cache.insert(item.1, h);
                    known_count += 1;
                    continue;
                }
            }
            work_todo.push(item);
        }
        let hashed: u64 = work_todo.iter().map(|t| t.3).sum();
        if !work_todo.is_empty() {
            crate::deprintln!("[head] B32: hashing {} new shard file(s), {:.2} GB (first ship only; cached afterwards) ...",
                      work_todo.len(), hashed as f64 / 1e9);
            let t0 = std::time::Instant::now();

            // Group co-located segments across ranks by logical name (e.g. `seg/L00.safetensors` for ranks 1..W-1)
            // so each layer's source tensors are read ONCE sequentially from NVMe instead of W-1 times.
            enum HashJob {
                SegBatch {
                    sort_key: (PathBuf, u64),
                    items: Vec<(String, Vec<crate::shard_plan::Piece>)>,
                },
                Single {
                    sort_key: (PathBuf, u64),
                    key: String,
                    src: crate::shard_plan::PlanSrc,
                },
            }
            let mut seg_groups: std::collections::BTreeMap<String, Vec<(String, Vec<crate::shard_plan::Piece>)>> =
                std::collections::BTreeMap::new();
            let mut jobs: Vec<HashJob> = Vec::new();
            for (logical, key, src, _sz) in work_todo {
                match src {
                    crate::shard_plan::PlanSrc::Segment(pieces) => {
                        seg_groups.entry(logical).or_default().push((key, pieces));
                    }
                    other => {
                        let sk = match &other {
                            crate::shard_plan::PlanSrc::Whole(p) => (p.clone(), 0),
                            _ => (PathBuf::new(), u64::MAX),
                        };
                        jobs.push(HashJob::Single { sort_key: sk, key, src: other });
                    }
                }
            }
            for (_logical, items) in seg_groups {
                let mut sk: Option<(PathBuf, u64)> = None;
                for (_, ps) in &items {
                    if let Some(p0) = ps.first() {
                        let cand = (p0.src.clone(), p0.off);
                        if sk.as_ref().map_or(true, |cur| &cand < cur) { sk = Some(cand); }
                    }
                }
                jobs.push(HashJob::SegBatch { sort_key: sk.unwrap_or_else(|| (PathBuf::new(), u64::MAX)), items });
            }
            // Walk source files strictly sequentially (file 00001 -> N) to maximize NVMe sequential bandwidth
            // and avoid random cross-file page-cache thrashing on packs larger than RAM.
            jobs.sort_by(|a, b| {
                let ka = match a { HashJob::SegBatch { sort_key, .. } | HashJob::Single { sort_key, .. } => sort_key };
                let kb = match b { HashJob::SegBatch { sort_key, .. } | HashJob::Single { sort_key, .. } => sort_key };
                ka.cmp(kb)
            });

            use std::sync::atomic::{AtomicUsize, Ordering};
            let next = AtomicUsize::new(0);
            let out: std::sync::Mutex<Vec<Result<Vec<(String, String)>>>> = std::sync::Mutex::new(Vec::new());
            // 3 workers = ~7.3 GB/s ARMv8 HW SHA-256 capacity, saturating NVMe without 8-thread seek thrashing.
            let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(3).min(3);
            let prog = Progress::new("hashing the shard files (first ship; cached afterwards)", hashed, Unit::Bytes);
            std::thread::scope(|sc| {
                for _ in 0..threads.min(jobs.len()) {
                    sc.spawn(|| loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= jobs.len() { break; }
                        let res: Result<Vec<(String, String)>> = match &jobs[i] {
                            HashJob::SegBatch { items, .. } => {
                                let refs: Vec<&[crate::shard_plan::Piece]> = items.iter().map(|(_, ps)| ps.as_slice()).collect();
                                crate::shard_plan::hash_segments_single_pass(&refs, |n| prog.add(n)).map(|hs| {
                                    items.iter().zip(hs).map(|((k, _), h)| (k.clone(), h)).collect()
                                })
                            }
                            HashJob::Single { key, src, .. } => {
                                hash_plan_src(src, Some(&prog)).map(|h| vec![(key.clone(), h)])
                            }
                        };
                        out.lock().unwrap().push(res);
                    });
                }
            });
            prog.finish();
            for r in out.into_inner().unwrap() {
                for (k, h) in r? {
                    cache.insert(k, h);
                }
            }
            crate::deprintln!("[head] B32: hashed in {:.1} s", t0.elapsed().as_secs_f64());
        }
        if known_count > 0 || hashed > 0 {
            hash_cache_save(&cache);
        }
    }
    let mut ships: Vec<RankShip> = (0..world.max(1)).map(|_| RankShip { model_id: base.clone(), artifacts: Vec::new(), srcs: HashMap::new() }).collect();
    for p in plans {
        let mut arts = Vec::new();
        let mut srcs = HashMap::new();
        for f in p.files {
            let size = RankPlan::file_size(&f)?;
            let hash = match plan_src_key(&f.src)? {
                Some(k) => cache.get(&k).cloned().context("B32: a planned file was not hashed")?,
                None => hash_plan_src(&f.src, None)?,
            };
            arts.push(Artifact { logical: f.logical.clone(), hash, size });
            srcs.insert(f.logical, f.src);
        }
        let total: u64 = arts.iter().map(|a| a.size).sum();
        crate::deprintln!("[head] B32 plan rank {}/{} ({}, deal {}): {} files, {:.2} GiB",
                  p.rank, world, fmt.name(), p.deal, arts.len(), total as f64 / (1u64 << 30) as f64);
        let rank = p.rank;
        ships[rank] = RankShip { model_id: format!("{base}@{}-w{world}-r{}-{}", fmt.name(), p.rank, p.deal), artifacts: arts, srcs };
    }
    Ok(ships)
}

/// Send one artifact's bytes: from its B32 plan source when there is one, else the model dir's file.
fn send_artifact(s: &mut impl Write, model_dir: &Path, a: &Artifact,
                 srcs: Option<&HashMap<String, crate::shard_plan::PlanSrc>>) -> Result<()> {
    use crate::shard_plan::PlanSrc;
    match srcs.and_then(|m| m.get(&a.logical)) {
        Some(PlanSrc::Whole(p)) => send_blob(s, p),
        Some(PlanSrc::Segment(ps)) => {
            let mut w = std::io::BufWriter::with_capacity(1 << 20, &mut *s);
            crate::shard_plan::write_segment(ps, &mut w)?;
            w.flush()?;
            Ok(())
        }
        Some(PlanSrc::Bytes(b)) => { s.write_all(b)?; s.flush()?; Ok(()) }
        Some(PlanSrc::Generated(g)) => {
            let mut w = std::io::BufWriter::with_capacity(1 << 20, &mut *s);
            g.write_to(&mut w)?;
            w.flush()?;
            Ok(())
        }
        None => send_blob(s, &model_dir.join(&a.logical)),
    }
}

// ---------------------------------------------------------------------------------------------------
// EXL3 pack manifest (TP-A D-T0-6; PACK-FIX 2026-09-30: the files the engine READS, nothing else)
// ---------------------------------------------------------------------------------------------------

/// PACK-FIX: the small (non-tensor) files the EXL3 TP path reads from the pack dir — the loader,
/// the tokenizer and the serve boot. Code evidence (PLAN/notes_2026-09-28/PACK-FIX.md):
///   config.json                  exl3.rs Exl3Pack::open (qwen::Config), exl3_serve.rs cfg_eos,
///                                tokenizer.rs stop_token_ids, exl3_tune.rs config_sha256
///   quantization_config.json     exl3.rs is_exl3_dir + Exl3Pack::open (quant_method/codebook),
///                                exl3_serve.rs is_exl3_pack, exl3_tune.rs quant_sha256
///   model.safetensors.index.json exl3.rs Exl3Pack::open (weight_map -> the shard set),
///                                exl3_forward.rs PLE ngram loader, exl3_serve.rs is_exl3_pack
///   tokenizer.json               tokenizer.rs QwenTokenizer::from_file (+ special_ids)
///   tokenizer_config.json        tokenizer.rs stop_token_ids (eos_token) + load_chat_env fallback
///   generation_config.json       tokenizer.rs stop_token_ids (eos_token_id int|list)
///   chat_template.jinja          tokenizer.rs load_chat_env (the served chat template)
/// The three optional ones (tokenizer_config / generation_config / chat_template) are hashed when
/// present; presence itself is part of the manifest (one side having a template the other lacks
/// changes served bytes, so it is a mismatch). Everything else in the dir — README*, LICENSE,
/// *.md, qbench_prompts.*, pack_scan.json, *.native, vocab.json/merges.txt (the tokenizer reads
/// tokenizer.json only), preprocessor configs (the EXL3 serve path has no vision tower), dot-files,
/// unknown files — is NOT read by the engine and is ignored.
pub const PACK_READ_FILES: [&str; 7] = [
    "config.json", "quantization_config.json", "model.safetensors.index.json", "tokenizer.json",
    "tokenizer_config.json", "generation_config.json", "chat_template.jinja",
];
/// The sidecars `Exl3Pack::open` opens BY NAME (header read) whether or not the index lists them
/// (exl3.rs step 7). On the Flash-Next pack the index lists both as well.
pub const PACK_SIDECARS: [&str; 2] =
    ["ngram_embedding.safetensors", "mtp_hyper_connection_mixer_patch.safetensors"];

/// One hashed pack file: (path relative to the pack dir, sha256 hex, size in bytes). A file the
/// index references that does not exist is recorded as ("…", "MISSING", 0) so a node's diff names it.
pub type PackFile = (String, String, u64);

/// The EXL3 pack manifest: the per-file list (sorted by path), the manifest hash over it, the
/// hashed byte total, the top-level entries NOT hashed (for the boot log), and how many bytes were
/// actually hashed this call (0 on a warm hash cache).
pub struct PackManifest {
    pub hash: String,
    pub files: Vec<PackFile>,
    pub bytes: u64,
    pub ignored: Vec<String>,
    pub hashed_now: u64,
}

const PACK_MISSING: &str = "MISSING";

/// The loader-read file set of an EXL3 pack (relative names, sorted, deduplicated): every
/// safetensors file the index's weight_map references (= `Exl3Pack::shards`, whose headers and
/// tensors `FwdModel::load` reads) + the by-name sidecars that exist + the `PACK_READ_FILES` that
/// exist, and the sorted top-level names NOT in that set. The index is mandatory — the loader cannot
/// run without it.
fn pack_file_set(dir: &Path) -> Result<(Vec<String>, Vec<String>)> {
    let idx_path = dir.join("model.safetensors.index.json");
    let raw = std::fs::read_to_string(&idx_path)
        .with_context(|| format!("pack manifest: {} missing (not an EXL3 pack?)", idx_path.display()))?;
    let idx: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("pack manifest: {} unparsable", idx_path.display()))?;
    let wmap = idx["weight_map"].as_object()
        .with_context(|| format!("pack manifest: {} has no weight_map", idx_path.display()))?;
    let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for v in wmap.values() {
        if let Some(s) = v.as_str() { set.insert(s.to_string()); }
    }
    for f in PACK_SIDECARS.iter().chain(PACK_READ_FILES.iter()) {
        if dir.join(f).is_file() { set.insert((*f).to_string()); }
    }
    // top-level components of the hashed set (an index may reference a shard in a subdir)
    let tops: std::collections::BTreeSet<&str> = set.iter().filter_map(|s| s.split('/').next()).collect();
    let mut ignored = Vec::new();
    for e in std::fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
        let name = e?.file_name().to_string_lossy().to_string();
        if !tops.contains(name.as_str()) { ignored.push(name); }
    }
    ignored.sort();
    Ok((set.into_iter().collect(), ignored))
}

/// PACK-FIX hash-cache key: the CANONICAL (symlink-resolved) path + mtime + size. A symlinked view
/// of a pack (HF snapshot links, test views) therefore reuses the real files' cached hashes. As
/// sound as the path-as-given key: the bytes are a function of the real file, and its mtime/size
/// are what the key already trusted. (build_manifest / draft_manifest keep `cached_key`.)
fn canonical_key(path: &Path) -> Result<String> {
    let real = std::fs::canonicalize(path).with_context(|| format!("canonicalize {}", path.display()))?;
    cached_key(&real)
}

/// Hash `jobs` (path, size) with up to `threads` workers, largest file first (the largest file
/// bounds the wall time). Each file's sha256 is independent, so the result is bitwise the
/// sequential one. Returns the hashes in `jobs` order.
fn sha256_files_parallel(jobs: &[(PathBuf, u64)], threads: usize) -> Result<Vec<String>> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let mut order: Vec<usize> = (0..jobs.len()).collect();
    order.sort_by(|&a, &b| jobs[b].1.cmp(&jobs[a].1));
    let next = AtomicUsize::new(0);
    let out: std::sync::Mutex<Vec<Option<Result<String>>>> =
        std::sync::Mutex::new((0..jobs.len()).map(|_| None).collect());
    let prog = Progress::new("hashing the model files (first use; cached afterwards)", jobs.iter().map(|j| j.1).sum(), Unit::Bytes);
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1).min(jobs.len().max(1)) {
            sc.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= order.len() { break; }
                let i = order[k];
                let r = sha256_file_with(&jobs[i].0, Some(&prog)).map(|(h, _)| h);
                out.lock().unwrap()[i] = Some(r);
            });
        }
    });
    prog.finish();
    out.into_inner().unwrap().into_iter()
        .map(|r| r.unwrap_or_else(|| Err(anyhow::anyhow!("hash worker did not run"))))
        .collect()
}

/// TP-A (EXL3 TP, D-T0-6) + PACK-FIX: the pack manifest — sha256 over the sorted
/// `relative-path \0 sha256 \0 size \n` lines of the files the engine READS (`pack_file_set`: the
/// index-referenced safetensors + by-name sidecars + `PACK_READ_FILES`), content-hashed through the
/// per-file (canonical path, mtime, size) hash cache; uncached files are hashed in parallel. The head
/// ships the hash AND the per-file list in TpConfig; the node recomputes over ITS local copy and, on
/// a mismatch, refuses naming each differing file. Files the engine never reads (README.md, LICENSE,
/// ...) are not part of it: a republished model card must not block a TP boot (owner 2026-09-30).
pub fn pack_manifest(dir: &Path) -> Result<PackManifest> {
    let (names, ignored) = pack_file_set(dir)?;
    let mut cache = hash_cache_load();
    let mut entries: Vec<(String, Option<String>, u64)> = Vec::with_capacity(names.len());
    let mut todo: Vec<(PathBuf, u64)> = Vec::new();
    let mut todo_idx: Vec<(usize, String)> = Vec::new();
    for rel in &names {
        let path = dir.join(rel);
        match std::fs::metadata(&path) {
            Ok(md) if md.is_file() => {
                let size = md.len();
                let key = canonical_key(&path)?;
                match cache.get(&key) {
                    Some(h) => entries.push((rel.clone(), Some(h.clone()), size)),
                    None => {
                        todo_idx.push((entries.len(), key));
                        todo.push((path, size));
                        entries.push((rel.clone(), None, size));
                    }
                }
            }
            // index-referenced but absent: recorded, never silently skipped (the node names it)
            _ => entries.push((rel.clone(), Some(PACK_MISSING.to_string()), 0)),
        }
    }
    let hashed_now: u64 = todo.iter().map(|(_, s)| *s).sum();
    if !todo.is_empty() {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
        let hashes = sha256_files_parallel(&todo, threads)?;
        for ((i, key), h) in todo_idx.into_iter().zip(hashes) {
            cache.insert(key, h.clone());
            entries[i].1 = Some(h);
        }
        hash_cache_save(&cache);
    }
    let files: Vec<PackFile> = entries.into_iter().map(|(r, h, s)| (r, h.unwrap(), s)).collect();
    let bytes = files.iter().map(|f| f.2).sum();
    let mut lines: Vec<String> = files.iter().map(|(r, h, s)| format!("{r}\0{h}\0{s}\n")).collect();
    lines.sort();
    let mut h = Sha256::new();
    for l in &lines { h.update(l.as_bytes()); }
    Ok(PackManifest { hash: hex(&h.finalize()), files, bytes, ignored, hashed_now })
}

/// PACK-FIX: human-readable per-file differences between the head's manifest list and a node's
/// (one line per differing file; empty = the lists agree).
pub fn pack_manifest_diff(head: &[PackFile], node: &[PackFile]) -> Vec<String> {
    let hm: HashMap<&str, (&str, u64)> = head.iter().map(|(r, h, s)| (r.as_str(), (h.as_str(), *s))).collect();
    let nm: HashMap<&str, (&str, u64)> = node.iter().map(|(r, h, s)| (r.as_str(), (h.as_str(), *s))).collect();
    let mut names: Vec<&str> = hm.keys().chain(nm.keys()).copied().collect();
    names.sort();
    names.dedup();
    let mut out = Vec::new();
    for n in names {
        match (hm.get(n), nm.get(n)) {
            (Some(a), Some(b)) if a == b => {}
            (Some((ha, sa)), Some((hb, sb))) =>
                out.push(format!("  {n}: head sha256 {ha} ({sa} B)  !=  node sha256 {hb} ({sb} B)")),
            (Some((ha, sa)), None) => out.push(format!("  {n}: on the head (sha256 {ha}, {sa} B), ABSENT on the node")),
            (None, Some((hb, sb))) => out.push(format!("  {n}: ABSENT on the head, on the node (sha256 {hb}, {sb} B)")),
            (None, None) => {}
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------------
// TP-4D (audit C3 / C4 / C9): liveness of the RETAINED control stream, world-general
//
// The stream is the ONLY thing that ends a node's session: a node idle between requests (or between prefill
// chunks) blocks in a plain read, and a head HOST that loses power / a cable or switch port that drops delivers
// no FIN and no RST, so the node's read never returned, the node child never exited and its supervisor never
// re-armed (and a stray TCP client on the control port consumed the node's single accept and wedged it the same
// way). At world 4 that is three streams per head and three supervisors to re-arm. Socket options only: they add
// no application byte to any stream (TCP keepalive probes carry no data), so the wire protocol — including the
// world-2 one — is unchanged. `harden_control_stream` is applied to BOTH ends of every retained stream.
// ---------------------------------------------------------------------------------------------------

/// Seconds of idleness before the first keepalive probe / between probes / unanswered probes. With
/// `TCP_USER_TIMEOUT` set (below) the kernel's keepalive timer drops the connection once the idle time reaches the
/// user timeout with probes outstanding, so `IDLE + INTVL * CNT` = 25 s is NOT the detection time: a silent host
/// is detected in ~`CTL_USER_TIMEOUT_MS` (60 s, up to one probe interval more).
const CTL_KEEPIDLE_S: i32 = 10;
const CTL_KEEPINTVL_S: i32 = 5;
const CTL_KEEPCNT: i32 = 3;
/// Data in flight but unacknowledged for this long (ms) aborts the connection (the kernel default is ~15 min).
/// Generous against the sync phase: a receiver that does not read for 60 s while the head has data queued would
/// also trip it, and the node reads every blob as it arrives.
const CTL_USER_TIMEOUT_MS: i32 = 60_000;
/// How long a node waits for a fresh connection to send the first 4 bytes of its `Hello` frame before it drops
/// the connection as not-a-head (a health check that connects and stays silent, `telnet`).
const NODE_FIRST_FRAME_WAIT: Duration = Duration::from_secs(10);
/// How long the head waits for the node's `Hello` answer (the node's accept + first-frame check precede it).
const HEAD_HELLO_WAIT: Duration = Duration::from_secs(60);

fn set_sock_i32(fd: i32, level: i32, name: i32, val: i32) -> std::io::Result<()> {
    let r = unsafe {
        libc::setsockopt(fd, level, name, &val as *const i32 as *const libc::c_void, std::mem::size_of::<i32>() as libc::socklen_t)
    };
    if r == 0 { Ok(()) } else { Err(std::io::Error::last_os_error()) }
}

/// TCP keepalive + `TCP_USER_TIMEOUT` on a retained control stream (see the section comment). A read on the
/// stream then fails with `ETIMEDOUT` within ~60 s of a silent peer host instead of blocking forever, which every
/// reader already treats as "session over" (the node mirror returns, the child exits, the supervisor re-arms).
pub(crate) fn harden_control_stream(s: &TcpStream) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    let fd = s.as_raw_fd();
    set_sock_i32(fd, libc::SOL_SOCKET, libc::SO_KEEPALIVE, 1)?;
    set_sock_i32(fd, libc::IPPROTO_TCP, libc::TCP_KEEPIDLE, CTL_KEEPIDLE_S)?;
    set_sock_i32(fd, libc::IPPROTO_TCP, libc::TCP_KEEPINTVL, CTL_KEEPINTVL_S)?;
    set_sock_i32(fd, libc::IPPROTO_TCP, libc::TCP_KEEPCNT, CTL_KEEPCNT)?;
    set_sock_i32(fd, libc::IPPROTO_TCP, libc::TCP_USER_TIMEOUT, CTL_USER_TIMEOUT_MS)?;
    Ok(())
}

fn harden_or_warn(s: &TcpStream, who: &str) {
    if let Err(e) = harden_control_stream(s) {
        eprintln!("[{who}] WARNING: control-stream keepalive / user-timeout not set ({e}) — a silently dead peer host will not be detected");
    }
}

/// Wait until `s` has delivered the 4 length bytes of a first frame (peeked, not consumed).
fn wait_first_frame(s: &TcpStream, wait: Duration) -> std::result::Result<(), String> {
    let t0 = std::time::Instant::now();
    let mut b = [0u8; 4];
    loop {
        let left = match wait.checked_sub(t0.elapsed()) {
            Some(d) if !d.is_zero() => d,
            _ => return Err(format!("no complete frame header within {:.1} s", wait.as_secs_f64())),
        };
        s.set_read_timeout(Some(left.max(Duration::from_millis(1)))).map_err(|e| e.to_string())?;
        match s.peek(&mut b) {
            Ok(0) => return Err("closed without sending anything".into()),
            Ok(n) if n >= 4 => return Ok(()),
            Ok(_) => std::thread::sleep(Duration::from_millis(5)),
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {}
            Err(e) => return Err(e.to_string()),
        }
    }
}

/// Node side: accept the head's sync connection. A connection that does not send the first 4 bytes of a frame
/// within `first_frame_wait` (or closes without sending) is DROPPED and the node keeps listening — one stray
/// client on the control port no longer consumes the node's single accept and wedges it (audit C4). The accepted
/// stream is hardened (`harden_control_stream`) and returned with its read timeout cleared.
pub(crate) fn accept_head(listener: &TcpListener, first_frame_wait: Duration) -> Result<(TcpStream, SocketAddr)> {
    loop {
        let (s, from) = listener.accept()?;
        s.set_nodelay(true).ok();
        harden_or_warn(&s, "node");
        match wait_first_frame(&s, first_frame_wait) {
            Ok(()) => {
                s.set_read_timeout(None).ok();
                return Ok((s, from));
            }
            Err(why) => crate::deprintln!("[node] connection from {from} dropped: {why} (not a head sync; still listening)"),
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// PACK-FIX: node refusal -> the head fails loudly (never block in the RDMA bring-up for a node
// that already quit)
// ---------------------------------------------------------------------------------------------------

/// Cap on a reported node failure (the head peeks a whole frame; a per-file diff is ~200 B/file).
const NODE_FAIL_MAX: usize = 32 * 1024;

/// Node side: report a failure during the session BOOT (pack mismatch, option install, program
/// parse, its RDMA link bring-up; on the served path also its load / attach / boot agree) to the head
/// over the retained control stream as one `Msg::Error` frame, so the head — blocked in its link
/// bring-up or loading its own shard — prints the reason and exits instead of hanging. Best effort:
/// a dead socket is not an error here (the node is about to exit, which closes the stream; the head's
/// watch reports that too).
pub fn node_report_failure(s: &mut TcpStream, reason: &str) {
    let mut msg = format!("node {}: {reason}", hostname());
    if msg.len() > NODE_FAIL_MAX {
        let mut cut = NODE_FAIL_MAX;
        while !msg.is_char_boundary(cut) { cut -= 1; }
        msg.truncate(cut);
        msg.push_str(" …(truncated)");
    }
    let _ = send_msg(s, &Msg::Error { msg });
    let _ = s.flush();
}

enum PeekFrame { Incomplete, Error(String), Other }

fn peek_frame(b: &[u8]) -> PeekFrame {
    if b.len() < 4 { return PeekFrame::Incomplete; }
    let n = u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize;
    if n > NODE_FAIL_MAX + 1024 { return PeekFrame::Other; }
    if b.len() < 4 + n { return PeekFrame::Incomplete; }
    match serde_json::from_slice::<Msg>(&b[4..4 + n]) {
        Ok(Msg::Error { msg }) => PeekFrame::Error(msg),
        _ => PeekFrame::Other,
    }
}

/// Head side: while the head is blocked in the RDMA data-plane bring-up (the world==2 QP handshake
/// is a plain accept() with no deadline, before any liveness probe exists) and — on the served path —
/// while it loads its own shard, watch every node's retained control stream WITHOUT consuming it
/// (MSG_PEEK). A node that reports a failure (`node_report_failure`) or closes the stream (its session
/// process exited) has abandoned the session: print the node's reason and exit the head non-zero
/// within ~0.1 s. Legitimate traffic (the node's `Ready`) ends the watch for that stream, untouched.
/// `disarm` stops the watch and restores the streams' blocking reads BEFORE the head reads them.
pub struct NodeWatch {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

pub fn watch_nodes_during_bring_up(streams: &[TcpStream]) -> Result<NodeWatch> {
    use std::sync::atomic::{AtomicBool, Ordering};
    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let mut watched: Vec<(usize, String, TcpStream)> = Vec::with_capacity(streams.len());
    for (i, s) in streams.iter().enumerate() {
        let c = s.try_clone().context("node watch: clone control stream")?;
        c.set_read_timeout(Some(Duration::from_millis(100))).context("node watch: read timeout")?;
        let addr = c.peer_addr().map(|a| a.to_string()).unwrap_or_else(|_| "?".into());
        watched.push((i + 1, addr, c));
    }
    crate::deprintln!("[head] watching {} node control stream(s) through the RDMA bring-up + boot (a node failure is reported here, never hung on)",
              watched.len());
    let stop2 = stop.clone();
    let handle = std::thread::Builder::new().name("tp-node-watch".into()).spawn(move || {
        let mut buf = vec![0u8; NODE_FAIL_MAX + 2048];
        let mut live = vec![true; watched.len()];
        while !stop2.load(Ordering::Acquire) && live.iter().any(|&l| l) {
            let mut partial = false;
            for (k, (rank, addr, c)) in watched.iter().enumerate() {
                if !live[k] || stop2.load(Ordering::Acquire) { continue; }
                match c.peek(&mut buf) {
                    Ok(0) => node_watch_fatal(*rank, addr, None),
                    Ok(n) => match peek_frame(&buf[..n]) {
                        PeekFrame::Error(msg) => node_watch_fatal(*rank, addr, Some(msg)),
                        PeekFrame::Incomplete => partial = true,
                        PeekFrame::Other => live[k] = false,
                    },
                    Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {}
                    Err(e) => node_watch_fatal(*rank, addr, Some(format!("control connection error: {e}"))),
                }
            }
            if partial { std::thread::sleep(Duration::from_millis(20)); }
        }
    }).context("spawn node watch")?;
    Ok(NodeWatch { stop, handle: Some(handle) })
}

impl NodeWatch {
    /// Stop watching and restore blocking reads on `streams` (the socket's read timeout is shared
    /// with the watch's clones).
    pub fn disarm(mut self, streams: &[TcpStream]) {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        if let Some(h) = self.handle.take() { let _ = h.join(); }
        for s in streams { let _ = s.set_read_timeout(None); }
    }
}

fn node_watch_fatal(rank: usize, addr: &str, reason: Option<String>) -> ! {
    match reason {
        Some(r) => eprintln!("[head] FATAL: NODE FAILED — rank {rank} ({addr}) abandoned the session during its boot:\n{r}"),
        None => eprintln!("[head] FATAL: NODE FAILED — rank {rank} ({addr}) closed the control connection during the session \
                           boot (its session process exited; the reason is in the node's log)"),
    }
    eprintln!("[head] exiting (code 1): the head does not wait in the RDMA bring-up / boot for a node that already quit");
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    std::process::exit(1)
}

/// Whether a logical path (relative to the model dir) belongs in node rank `r`'s manifest. When
/// `rank{r}/` exists (`sharded`), only files under `rank{r}/` plus the two always-shipped root
/// files are kept; otherwise the whole model is replicated (P2 replicate-if-not-divisible).
fn include_for_rank(rel: &str, rank: usize, sharded: bool) -> bool {
    if !sharded {
        return true;
    }
    let rank_dir = format!("rank{rank}");
    rel.starts_with(&format!("{rank_dir}/")) || rel == "inference/config.json" || rel == "config.json"
}

/// Manifest for a REPLICATED auxiliary artifact dir — the DFlash2 drafter (gap fix 2026-08-23).
/// Unlike `build_manifest` there is no rank partitioning: every file goes to every node (the
/// drafter is never sharded over the sync; `Df2Round::load_tp` shards it in memory after load).
/// Shares the per-path (path,mtime,size)→hash cache with the trunk manifest, so a re-launch
/// re-hashes nothing.
fn draft_manifest(dir: &Path) -> Result<(String, Vec<Artifact>)> {
    let mut cache = hash_cache_load();
    let mut dirty = false;
    let mut artifacts = Vec::new();
    let files = model_files(dir)?;
    let prog = Progress::new("hashing the drafter files (first use; cached afterwards)", uncached_bytes(&files, &cache)?, Unit::Bytes);
    for path in files {
        let rel = path.strip_prefix(dir).unwrap_or(&path).to_string_lossy().to_string();
        let key = cached_key(&path)?;
        let (hash, size) = if let Some(h) = cache.get(&key) {
            (h.clone(), std::fs::metadata(&path)?.len())
        } else {
            let (h, sz) = sha256_file_with(&path, Some(&prog))?;
            cache.insert(key, h.clone());
            dirty = true;
            (h, sz)
        };
        artifacts.push(Artifact { logical: rel, hash, size });
    }
    prog.finish();
    if dirty { hash_cache_save(&cache); }
    let model_id = dir.file_name().map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "draft".into());
    Ok((model_id, artifacts))
}

mod discovery;
mod head;
mod models;
mod node;
pub use self::discovery::*;
pub use self::head::*;
pub use self::models::*;
pub use self::node::*;

/// Tests that point the process-global `--tp-cache` at a temp dir hold this lock (they would race otherwise).
#[cfg(test)]
pub(crate) static CACHE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod ipv4_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod pack_manifest_tests;
/// S-B9-CF-AGREE16B (W1, W2): what the head/node build guard covers, and that both ends apply it.
#[cfg(test)]
mod build_id_tests;
#[cfg(test)]
mod tpship_tests;
