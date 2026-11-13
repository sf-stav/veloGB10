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
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        h.update(&buf[..n]);
        total += n as u64;
        if let Some(p) = prog { p.add(n as u64); }
    }
    Ok((hex(&h.finalize()), total))
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
struct HashWriter<'a> { h: Sha256, n: u64, prog: Option<&'a Progress> }
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
            let mut w = HashWriter { h: Sha256::new(), n: 0, prog };
            crate::shard_plan::write_segment(ps, &mut w)?;
            hex(&w.h.finalize())
        }
        PlanSrc::Bytes(b) => {
            if let Some(p) = prog { p.add(b.len() as u64); }
            let mut h = Sha256::new(); h.update(b); hex(&h.finalize())
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
    })
}

/// B32: the shipments of ranks 1..world (index 0 = the head, empty) from `shard_plan`, hashed through the
/// shared hash cache (uncached files in parallel, largest first). Returns per-rank artifacts and sources.
pub fn ship_plans(model_dir: &Path, world: u32) -> Result<Vec<RankShip>> {
    use crate::shard_plan::{detect_format, shard_plan, RankPlan, ShardFormat};
    let world = world as usize;
    let fmt = detect_format(model_dir).with_context(|| format!(
        "shard shipping: {} is neither an EXL3 pack nor a qwen4_exp NVFP4 model — no shard plan", model_dir.display()))?;
    let deal = match fmt { ShardFormat::Exl3 | ShardFormat::Fp4l => crate::exl3_forward::xtp::ep_deal_kind(), ShardFormat::Nvfp4 => "contig".to_string(),
                           ShardFormat::Naive => crate::naive::shipplan::deal(world) };
    let base = model_dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "model".into());
    let mut plans = Vec::new();
    for r in 1..world { plans.push(shard_plan(model_dir, fmt, world, r, &deal)?); }
    // distinct cacheable sources, hashed once
    let mut cache = hash_cache_load();
    let mut todo: Vec<(String, crate::shard_plan::PlanSrc, u64)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for p in &plans {
        for f in &p.files {
            if let Some(k) = plan_src_key(&f.src)? {
                if !cache.contains_key(&k) && seen.insert(k.clone()) {
                    todo.push((k, f.src.clone(), RankPlan::file_size(f)?));
                }
            }
        }
    }
    let hashed: u64 = todo.iter().map(|t| t.2).sum();
    if !todo.is_empty() {
        crate::deprintln!("[head] B32: hashing {} new shard file(s), {:.2} GB (first ship only; cached afterwards) ...",
                  todo.len(), hashed as f64 / 1e9);
        let t0 = std::time::Instant::now();
        todo.sort_by(|a, b| b.2.cmp(&a.2));
        use std::sync::atomic::{AtomicUsize, Ordering};
        let next = AtomicUsize::new(0);
        let out: std::sync::Mutex<Vec<(usize, Result<String>)>> = std::sync::Mutex::new(Vec::new());
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
        let prog = Progress::new("hashing the shard files (first ship; cached afterwards)", hashed, Unit::Bytes);
        std::thread::scope(|sc| {
            for _ in 0..threads.min(todo.len()) {
                sc.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= todo.len() { break; }
                    let r = hash_plan_src(&todo[i].1, Some(&prog));
                    out.lock().unwrap().push((i, r));
                });
            }
        });
        prog.finish();
        for (i, r) in out.into_inner().unwrap() {
            cache.insert(todo[i].0.clone(), r?);
        }
        hash_cache_save(&cache);
        crate::deprintln!("[head] B32: hashed in {:.1} s", t0.elapsed().as_secs_f64());
    }
    let mut ships = vec![RankShip { model_id: base.clone(), artifacts: Vec::new(), srcs: HashMap::new() }];
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
        ships.push(RankShip { model_id: format!("{base}@{}-w{world}-r{}-{}", fmt.name(), p.rank, p.deal), artifacts: arts, srcs });
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

// ---------------------------------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct NodeInfo { pub hostname: String, pub addr: SocketAddr }

#[derive(Serialize, Deserialize)]
struct DiscoverProbe { magic: String, version: String }
#[derive(Serialize, Deserialize)]
struct DiscoverReply { hostname: String, tcp_port: u16, version: String }

/// Node side: answer discovery probes with our on-path IP + TCP port. Runs until the process exits.
pub fn spawn_discovery_responder(tcp_port: u16, udp_port: u16) -> Result<()> {
    let sock = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, udp_port))
        .context("bind UDP discovery port")?;
    let hostname = hostname();
    std::thread::spawn(move || {
        let mut buf = [0u8; 2048];
        loop {
            let (n, from) = match sock.recv_from(&mut buf) { Ok(x) => x, Err(_) => continue };
            let probe: DiscoverProbe = match serde_json::from_slice(&buf[..n]) { Ok(p) => p, Err(_) => continue };
            if probe.magic != DISCOVERY_MAGIC { continue; }
            // Reply via the same socket: the OS picks the source IP by the route back to the head, so
            // the head sees our on-path (RoCE) IP as the datagram source — exactly the TCP address to use.
            let reply = DiscoverReply { hostname: hostname.clone(), tcp_port, version: binary_version() };
            if let Ok(b) = serde_json::to_vec(&reply) { let _ = sock.send_to(&b, from); }
        }
    });
    Ok(())
}

/// Head side: broadcast a probe on the RoCE subnets (+ global broadcast) and collect responders.
pub fn discover_nodes(wait: Duration) -> Result<Vec<NodeInfo>> {
    let sock = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))?;
    sock.set_broadcast(true)?;
    sock.set_read_timeout(Some(Duration::from_millis(200)))?;
    let probe = serde_json::to_vec(&DiscoverProbe {
        magic: DISCOVERY_MAGIC.into(), version: binary_version() })?;
    // GB10 boxes always expose the same RoCE interface NAMES; resolve their live IP/broadcast at
    // runtime (IPs are config, names/types are fixed) and broadcast there + a global catch-all.
    let ifaces = roce_interfaces();
    for f in &ifaces {
        crate::deprintln!("  [discover] RoCE rail {} {} ({}) ip {} bcast {}", f.rail, f.ib, f.netdev, f.ip, f.bcast);
    }
    if ifaces.is_empty() {
        eprintln!("  [discover] no RoCE interface resolved — broadcasting globally only");
    }
    let mut targets: Vec<Ipv4Addr> = ifaces.iter().map(|f| f.bcast).collect();
    targets.push(Ipv4Addr::BROADCAST);
    for t in &targets { let _ = sock.send_to(&probe, (*t, DISCOVERY_PORT)); }
    let deadline = std::time::Instant::now() + wait;
    // A node replies once per subnet the probe reached it on (RoCE rail 1/2 + mgmt), all same hostname.
    // Keep the RoCE-preferred source IP: that is exactly the address the RDMA data plane must use.
    let mut nodes: HashMap<String, (u8, NodeInfo)> = HashMap::new();
    let mut buf = [0u8; 2048];
    while std::time::Instant::now() < deadline {
        match sock.recv_from(&mut buf) {
            Ok((n, from)) => {
                if let Ok(r) = serde_json::from_slice::<DiscoverReply>(&buf[..n]) {
                    if r.version != binary_version() {
                        eprintln!("  [discover] {} at {} has MISMATCHED binary ({} vs {}) — skipping",
                                  r.hostname, from.ip(), r.version, binary_version());
                        continue;
                    }
                    let rank = ip_rank(from.ip(), &ifaces);
                    let ni = NodeInfo { hostname: r.hostname.clone(), addr: SocketAddr::new(from.ip(), r.tcp_port) };
                    match nodes.get(&r.hostname) {
                        Some((existing, _)) if *existing >= rank => {}
                        _ => { nodes.insert(r.hostname, (rank, ni)); }
                    }
                }
            }
            Err(_) => {}   // read timeout; keep polling until the deadline
        }
    }
    Ok(nodes.into_values().map(|(_, ni)| ni).collect())
}

/// A live ConnectX-7 RoCE interface, resolved from its (fixed) IB device name to its current IPv4.
struct Roce { ib: String, netdev: String, ip: Ipv4Addr, bcast: Ipv4Addr, mask: u32, rail: u8 }

/// Resolve the GB10 RoCE rails by their fixed IB device names → netdev (via /sys) → IPv4 (via `ip`).
/// Names/types are constant across GB10 boxes (per the hardware); only the IPs are configuration.
fn roce_interfaces() -> Vec<Roce> {
    // The rails in order: explicit --rdma-dev=dev1[,dev2], else auto-detected (tp::rdma_devs: the only
    // active RoCE port pair; the ASUS GX10s here cable f1, the DGX Sparks f0).
    let devs: Vec<(String, u8)> = crate::tp::rdma_devs().iter().enumerate()
        .map(|(i, d)| (d.clone(), (i + 1) as u8)).collect();
    let mut out = Vec::new();
    for (ib, rail) in &devs {
        let netdir = format!("/sys/class/infiniband/{ib}/device/net");
        let netdev = std::fs::read_dir(&netdir).ok()
            .and_then(|mut d| d.next()).and_then(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string());
        let Some(netdev) = netdev else {
            eprintln!("  [discover] RoCE device '{ib}' not found — override with --rdma-dev / --rdma-dev, \
                       or use --nodes <ip> to skip discovery");
            continue;
        };
        if let Some((ip, prefix, bcast)) = ipv4_of(&netdev) {
            let mask = if prefix == 0 { 0 } else { u32::MAX << (32 - prefix) };
            out.push(Roce { ib: ib.clone(), netdev, ip, bcast, mask, rail: *rail });
        }
    }
    out
}

/// Parse `ip -o -4 addr show dev <netdev>` output → (addr, prefix_len, broadcast).
fn ipv4_of_str(s: &str) -> Option<(Ipv4Addr, u8, Ipv4Addr)> {
    let toks: Vec<&str> = s.split_whitespace().collect();
    let (mut ip, mut prefix, mut brd) = (None, None, None);
    let mut i = 0;
    while i + 1 < toks.len() {
        match toks[i] {
            "inet" => { let mut it = toks[i + 1].split('/');
                ip = it.next().and_then(|x| x.parse().ok());
                prefix = it.next().and_then(|x| x.parse().ok()); }
            "brd" => brd = toks[i + 1].parse().ok(),
            _ => {}
        }
        i += 1;
    }
    // Ported from sf-stav/veloGB10 PR#5 (Morxi): `brd` is absent for point-to-point prefixes
    // (/31, /30, ...) on some kernels: the runtime only needs the address itself, while the
    // bcast/mask serve discovery broadcast + rail ranking. So do NOT hard-require `brd`;
    // synthesize it from ip | ~mask when the kernel omits it.
    let ip = ip?;
    let prefix = prefix?;
    let mask = if prefix == 0 { 0 } else { u32::MAX << (32 - prefix) };
    let bcast = brd.unwrap_or_else(|| Ipv4Addr::from(u32::from(ip) | !mask));
    Some((ip, prefix, bcast))
}

/// Run `ip -o -4 addr show dev <netdev>` and parse it (the pure parser above is unit-tested).
fn ipv4_of(netdev: &str) -> Option<(Ipv4Addr, u8, Ipv4Addr)> {
    let out = std::process::Command::new("ip")
        .args(["-o", "-4", "addr", "show", "dev", netdev]).output().ok()?;
    ipv4_of_str(&String::from_utf8_lossy(&out.stdout))
}

#[cfg(test)]
mod ipv4_tests {
    use super::*;

    #[test]
    fn slash24_with_brd_uses_kernel_value() {
        let s = "2: eth0    inet 192.0.2.13/24 brd 192.0.2.255 scope global eth0";
        let (ip, prefix, bcast) = ipv4_of_str(s).expect("parse");
        assert_eq!(ip, Ipv4Addr::new(192, 0, 2, 13));
        assert_eq!(prefix, 24);
        assert_eq!(bcast, Ipv4Addr::new(192, 0, 2, 255));
    }

    #[test]
    fn slash31_no_brd_synthesizes() {
        // Peer link /31: kernel prints no `brd`; ip|~mask gives the link's other address.
        let s = "3: p2p    inet 10.0.0.0/31 scope link p2p";
        let (ip, prefix, bcast) = ipv4_of_str(s).expect("parse");
        assert_eq!(ip, Ipv4Addr::new(10, 0, 0, 0));
        assert_eq!(prefix, 31);
        assert_eq!(bcast, Ipv4Addr::new(10, 0, 0, 1));
    }

    #[test]
    fn slash30_no_brd_synthesizes() {
        let s = "4: p2p    inet 10.0.0.2/30 scope link p2p";
        let (ip, prefix, bcast) = ipv4_of_str(s).expect("parse");
        assert_eq!(ip, Ipv4Addr::new(10, 0, 0, 2));
        assert_eq!(prefix, 30);
        assert_eq!(bcast, Ipv4Addr::new(10, 0, 0, 3));
    }

    #[test]
    fn prefix_zero_no_panic_bcast_is_all_ones() {
        let s = "5: p2p    inet 10.0.0.7/0 scope link p2p";
        let (ip, prefix, bcast) = ipv4_of_str(s).expect("parse");
        assert_eq!(ip, Ipv4Addr::new(10, 0, 0, 7));
        assert_eq!(prefix, 0);
        assert_eq!(bcast, Ipv4Addr::new(255, 255, 255, 255));
    }

    #[test]
    fn missing_addr_is_none() {
        assert!(ipv4_of_str("2: eth0    <BROADCAST,MULTICAST,UP> mtu 1500").is_none());
    }
}

/// Rank a reply's source IP: on RoCE rail 1 (3) > rail 2 (2) > any other link (1).
fn ip_rank(ip: std::net::IpAddr, ifaces: &[Roce]) -> u8 {
    if let std::net::IpAddr::V4(v4) = ip {
        let x = u32::from(v4);
        for f in ifaces {
            if x & f.mask == u32::from(f.ip) & f.mask {
                return if f.rail == 1 { 3 } else { 2 };
            }
        }
    }
    1
}

/// Deterministic total order over a node's `SocketAddr`. Higher RoCE-rail preference sorts FIRST,
/// then ascending IPv4, then ascending port. The IPv4 tie-break is what makes the order a *total*
/// order (hostnames may collide across boxes or be unset in explicit `--nodes` mode).
fn node_order_key(addr: &SocketAddr, ifaces: &[Roce]) -> (u8, u128, u16) {
    let rail = ip_rank(addr.ip(), ifaces);
    let ip_key = match addr.ip() {
        IpAddr::V4(v4) => u32::from(v4) as u128,
        IpAddr::V6(v6) => u128::from(v6),
    };
    // rail desc => invert; ip/port asc.
    (255u8.wrapping_sub(rail), ip_key, addr.port())
}

/// Assign node ranks 1..N-1 in a stable, reproducible order. The head is ALWAYS rank 0; the
/// discovered nodes (sorted by `node_order_key`) become ranks 1,2,... in that order. The sort is
/// decoupled from `discover_nodes` (which only picks the best source IP per hostname) so explicit
/// `--nodes` and discovery take the same path.
fn assign_ranks(nodes: &mut Vec<NodeInfo>, ifaces: &[Roce]) {
    nodes.sort_by(|a, b| node_order_key(&a.addr, ifaces).cmp(&node_order_key(&b.addr, ifaces)));
}

/// Build the full rank→RoCE-IP topology (`Vec<String>` indexed by rank, size `world`). The head is
/// rank 0 at its own RoCE rail-1 IP (or the first resolved RoCE IP); `ranked_nodes` must already be
/// sorted by `assign_ranks` (ranks 1..N-1 in order). `topology[self_rank]` is the rank's own IP and
/// is unused by the N-way transport.
fn build_topology(ranked_nodes: &[NodeInfo], ifaces: &[Roce], world: u32) -> Result<Vec<String>> {
    anyhow::ensure!(
        ranked_nodes.len() as u32 == world.saturating_sub(1),
        "build_topology: got {} nodes for world {world} (need {})",
        ranked_nodes.len(),
        world.saturating_sub(1)
    );
    let mut topo: Vec<String> = vec![String::new(); world as usize];
    // rank 0 = this process; use its own RoCE rail-1 IP (fall back to any resolved RoCE IP). This
    // entry is the address every node uses to dial the control QP, so it MUST be non-empty.
    let head_ip = ifaces.iter().find(|f| f.rail == 1).map(|f| f.ip)
        .or_else(|| ifaces.first().map(|f| f.ip))
        .context("no RoCE interface resolved — cannot build the rank->IP topology")?;
    topo[0] = head_ip.to_string();
    for (i, node) in ranked_nodes.iter().enumerate() {
        let rank = (i + 1) as usize;
        topo[rank] = node.addr.ip().to_string();
    }
    Ok(topo)
}

fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname").map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "node".into())
}

// ---------------------------------------------------------------------------------------------------
// Blob streaming
// ---------------------------------------------------------------------------------------------------

fn send_blob(w: &mut impl Write, path: &Path) -> Result<()> {
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        w.write_all(&buf[..n])?;
    }
    w.flush()?;
    Ok(())
}

/// Receive `size` bytes into a temp file, hashing as we go; verify against `hash`; return the temp path.
fn recv_blob(r: &mut impl Read, hash: &str, size: u64) -> Result<PathBuf> {
    let tmp = cache_root().join("blobs").join(format!("tmp.{}.{}", std::process::id(), hash));
    std::fs::create_dir_all(tmp.parent().unwrap())?;
    let mut f = std::fs::File::create(&tmp)?;
    let mut hasher = Sha256::new();
    let mut left = size;
    let mut buf = vec![0u8; 1 << 20];
    while left > 0 {
        let want = left.min(buf.len() as u64) as usize;
        r.read_exact(&mut buf[..want])?;
        hasher.update(&buf[..want]);
        f.write_all(&buf[..want])?;
        left -= want as u64;
    }
    f.flush()?;
    let got = hex(&hasher.finalize());
    if got != hash {
        let _ = std::fs::remove_file(&tmp);
        bail!("blob hash mismatch: expected {hash}, got {got}");
    }
    Ok(tmp)
}

// ---------------------------------------------------------------------------------------------------
// Node: receive-and-assemble
// ---------------------------------------------------------------------------------------------------

/// Assemble a model dir under the cache: each logical name -> symlink to blobs/<hash>.
fn assemble_model_dir(model_id: &str, artifacts: &[Artifact]) -> Result<PathBuf> {
    let dir = cache_root().join("models").join(model_id);
    // B32: an empty manifest (the legacy EXL3 path ships nothing) caches nothing — no empty placeholder dir
    // (`--cached-models-list` used to show it as "0.00 GiB 0 blob(s)").
    if artifacts.is_empty() { return Ok(dir); }
    std::fs::create_dir_all(&dir)?;
    for a in artifacts {
        let link = dir.join(&a.logical);
        if let Some(parent) = link.parent() { std::fs::create_dir_all(parent)?; }
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(blob_path(&a.hash), &link)
            .with_context(|| format!("symlink {}", a.logical))?;
    }
    // B32: the dir mirrors THIS manifest — a link left from an earlier sync of the same id (another plan
    // version, a re-quantized pack) is removed, so the loader and the cache ops see exactly what shipped.
    let want: std::collections::HashSet<PathBuf> = artifacts.iter().map(|a| dir.join(&a.logical)).collect();
    let mut stack = vec![dir.clone()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for ent in rd.flatten() {
            let p = ent.path();
            match ent.file_type() {
                Ok(ft) if ft.is_dir() => stack.push(p),
                Ok(ft) if ft.is_symlink() && !want.contains(&p) => { let _ = std::fs::remove_file(&p); }
                _ => {}
            }
        }
    }
    Ok(dir)
}

/// Handle one head connection: Hello -> Manifest -> Missing -> blobs -> assemble -> Ready ->
/// Config -> DraftManifest -> (draft blobs) -> DraftReady. Returns the assembled model dir +
/// the assembled DRAFT cache dir (None when the head shipped no drafter) + the head's TP config
/// (so the node needs ZERO GB10_TP_* env, ZERO model paths, ZERO draft paths) + the RETAINED
/// control stream (the TP serving session runs over it; the bench path just drops it).
fn node_handle(mut s: TcpStream) -> Result<(PathBuf, Option<PathBuf>, crate::tp::TpConfig, TcpStream)> {
    match recv_msg(&mut s)? {
        Msg::Hello { version, hostname, .. } => {
            if let Err(why) = version_match(&version, &binary_version()) {
                send_msg(&mut s, &Msg::Error { msg: why.clone() })?;
                bail!("head/node {why}");
            }
            crate::deprintln!("  [node] head '{hostname}' connected (binary {version})");
        }
        _ => bail!("expected Hello"),
    }
    send_msg(&mut s, &Msg::Hello { version: binary_version(), role: "node".into(), hostname: hostname() })?;

    let (model_id, artifacts, draft_offer) = match recv_msg(&mut s)? {
        Msg::Manifest { model_id, artifacts, draft } => (model_id, artifacts, draft),
        _ => bail!("expected Manifest"),
    };
    let missing: Vec<String> = artifacts.iter().map(|a| a.hash.clone())
        .filter(|h| !have_blob(h)).collect();
    let have = artifacts.len() - missing.len();
    crate::deprintln!("  [node] manifest '{model_id}': {} artifacts, {have} cached, {} to fetch",
              artifacts.len(), missing.len());
    let fetch_bytes: u64 = artifacts.iter().filter(|a| missing.contains(&a.hash)).map(|a| a.size).sum();
    // owner 2026-11-02: report what this session would store here and the drive's free space; the head checks EVERY
    // node before it ships to any (and stops the session, telling us why, when a drive is short)
    let draft_need: u64 = draft_offer.iter().filter(|a| !have_blob(&a.hash) && !missing.contains(&a.hash)).map(|a| a.size).sum();
    let root = cache_root();
    let space = free_bytes(&root).map(|free| SpaceReport { need: fetch_bytes + draft_need, free,
                                                           dir: root.display().to_string(), host: hostname() });
    let t_fetch = std::time::Instant::now();
    send_msg(&mut s, &Msg::Missing { hashes: missing.clone(), space })?;

    let prog = Progress::new("receiving the model from the head", fetch_bytes, Unit::Bytes);
    for _ in 0..missing.len() {
        match recv_msg(&mut s)? {
            Msg::BlobHeader { hash, size } => {
                let tmp = recv_blob(&mut Counted { inner: &mut s, prog: &prog }, &hash, size)?;
                publish_blob(&hash, &tmp)?;
                crate::deprintln!("  [node] cached {} ({:.1} MB)", &hash[..12], size as f64 / 1e6);
            }
            Msg::Error { msg } => bail!("the head stopped the sync: {msg}"),
            _ => bail!("expected BlobHeader"),
        }
    }
    prog.finish();
    let dir = assemble_model_dir(&model_id, &artifacts)?;
    send_msg(&mut s, &Msg::Ready { model_dir: dir.to_string_lossy().to_string() })?;
    let cfg = match recv_msg(&mut s)? {
        Msg::Config(c) => c,
        Msg::Error { msg } => bail!("the head stopped the sync: {msg}"),
        _ => bail!("expected Config"),
    };
    crate::deprintln!("  [node] config from head: shard_mixers={} graph={} fp32_partials={} mtp={} depth={:?} mode_serve={}",
              cfg.shard_mixers, cfg.graph, cfg.fp32_partials, cfg.mtp, cfg.mtp_depth, cfg.mode_serve);


    // v5 — the DFlash2 draft round (always exactly one DraftManifest after Config). The bytes
    // land in the SAME content-addressed blob store as the model shards; the assembled dir lives
    // under models/<draft-id>/ exactly like a trunk model. A node therefore NEVER needs a local
    // draft copy — the caller rewrites the config's df2_draft_dir to the returned cache path.
    let draft_dir: Option<PathBuf> = match recv_msg(&mut s)? {
        Msg::DraftManifest { model_id, artifacts } if artifacts.is_empty() => {
            send_msg(&mut s, &Msg::DraftReady { model_dir: String::new() })?;
            crate::deprintln!("  [node] no draft artifact this session (head sent an empty manifest)");
            None
        }
        Msg::DraftManifest { model_id, artifacts } => {
            let missing: Vec<String> = artifacts.iter().map(|a| a.hash.clone())
                .filter(|h| !have_blob(h)).collect();
            let have = artifacts.len() - missing.len();
            crate::deprintln!("  [node] draft manifest '{model_id}': {} artifacts, {have} cached, {} to fetch",
                      artifacts.len(), missing.len());
            send_msg(&mut s, &Msg::Missing { hashes: missing.clone(), space: None })?;
            let dbytes: u64 = artifacts.iter().filter(|a| missing.contains(&a.hash)).map(|a| a.size).sum();
            let dprog = Progress::new("receiving the drafter from the head", dbytes, Unit::Bytes);
            for _ in 0..missing.len() {
                match recv_msg(&mut s)? {
                    Msg::BlobHeader { hash, size } => {
                        let tmp = recv_blob(&mut Counted { inner: &mut s, prog: &dprog }, &hash, size)?;
                        publish_blob(&hash, &tmp)?;
                        crate::deprintln!("  [node] cached draft blob {} ({:.1} MB)", &hash[..12], size as f64 / 1e6);
                    }
                    _ => bail!("expected draft BlobHeader"),
                }
            }
            dprog.finish();
            let ddir = assemble_model_dir(&model_id, &artifacts)?;
            send_msg(&mut s, &Msg::DraftReady { model_dir: ddir.to_string_lossy().to_string() })?;
            crate::deprintln!("  [node] DRAFT READY — drafter assembled at {} (loads from the cache, no local copy)",
                      ddir.display());
            Some(ddir)
        }
        _ => bail!("expected DraftManifest after Config"),
    };
    if cfg.sync_only {
        // B32 `--tp-sync-only`: report and stop here (the head stops after its sync too; the supervisor re-arms)
        let total: u64 = artifacts.iter().map(|a| a.size).sum();
        let mut uniq: Vec<&str> = artifacts.iter().map(|a| a.hash.as_str()).collect();
        uniq.sort(); uniq.dedup();
        eprintln!("SYNC_ONLY node '{}': model '{model_id}' {} artifacts ({} blobs) = {:.2} GiB; fetched {:.2} GiB in {:.1} s \
                   ({} already cached); assembled at {}",
                  hostname(), artifacts.len(), uniq.len(), total as f64 / (1u64 << 30) as f64,
                  fetch_bytes as f64 / (1u64 << 30) as f64, t_fetch.elapsed().as_secs_f64(), have, dir.display());
        std::process::exit(0);
    }
    Ok((dir, draft_dir, cfg, s))
}

/// Run as a node: answer discovery, accept ONE head sync, return the assembled model dir + the head's
/// IP (its RoCE address, used to bring up the RDMA data-plane link back to it) + the head's TP config
/// + the retained control stream (dropped by bench sessions, kept by serving ones).
pub fn run_node(tcp_port: u16, udp_port: u16)
    -> Result<(PathBuf, Option<PathBuf>, IpAddr, crate::tp::TpConfig, TcpStream)> {
    let _ = crate::tp::rdma_devs();   // resolve + print the RDMA rails now (an ambiguous box exits here, not mid-boot)
    spawn_discovery_responder(tcp_port, udp_port)?;
    let listener = TcpListener::bind((Ipv4Addr::UNSPECIFIED, tcp_port))
        .with_context(|| format!("bind TCP {tcp_port}"))?;
    eprintln!("[node] {} ready: discovery on UDP {udp_port}, control on TCP {tcp_port}, cache {}",
              hostname(), cache_root().display());
    // TP-4D (C4/C9): hardened stream + stray connections dropped (see `accept_head`)
    let (s, from) = accept_head(&listener, NODE_FIRST_FRAME_WAIT)?;
    crate::deprintln!("[node] head connected from {from}");
    let (dir, draft_dir, cfg, s) = node_handle(s)?;
    eprintln!("[node] SYNCED — model ready at {}{}",
              dir.display(),
              draft_dir.as_ref().map(|d| format!(", drafter at {}", d.display())).unwrap_or_default());
    Ok((dir, draft_dir, from.ip(), cfg, s))
}

// ---------------------------------------------------------------------------------------------------
// Head: discover-and-push
// ---------------------------------------------------------------------------------------------------

/// The head's half of the Hello: send ours (`mine` = `binary_version()`), read the node's, and REFUSE a
/// node whose version token differs. S-B9-CF-AGREE16B: the head used to accept any `Msg::Hello` here and
/// relied on the node's own comparison (`node_handle`) and, without `--nodes`, on the discovery filter.
/// A node built from this source always refuses first (the comparison is the same string inequality, so
/// its `Msg::Error` arrives instead of a Hello); this check is for a node that does not, and it keeps the
/// guarantee on the head, the side the operator reads. Nothing of the model is shipped before it.
fn head_hello(s: &mut TcpStream, node: &NodeInfo, mine: &str) -> Result<()> {
    send_msg(s, &Msg::Hello { version: mine.to_string(), role: "head".into(), hostname: hostname() })?;
    // TP-4D (C4): a node whose session process is wedged (accepted the connection, never answers) must not hang
    // the head forever; the bound covers only the Hello exchange, the later replies wait for pack hashing.
    s.set_read_timeout(Some(HEAD_HELLO_WAIT)).ok();
    let hello = recv_msg(s).with_context(|| format!(
        "node {} ({}) did not answer the Hello within {} s — a wedged `--node` session process there? (kill its child; the supervisor re-arms)",
        node.hostname, node.addr, HEAD_HELLO_WAIT.as_secs()))?;
    s.set_read_timeout(None).ok();
    match hello {
        Msg::Hello { version, .. } => {
            if let Err(why) = version_match(mine, &version) {
                bail!("node {} ({}) refused by the head: {why}", node.hostname, node.addr);
            }
            Ok(())
        }
        Msg::Error { msg } => bail!("node rejected: {msg}"),
        _ => bail!("expected node Hello"),
    }
}

/// Push the model to one node: Hello -> Manifest -> receive Missing -> stream blobs -> Ready, then
/// ship our TP config (`Msg::Config`) so the node runs with ZERO GB10_TP_* env vars. The shipped
/// config is a per-node clone: `node_rank` identifies this node and `topology` is the full
/// rank→RoCE-IP map (the node reads both back from its own `Msg::Config`). Returns the RETAINED
/// control stream — the TP serving session keeps it as its control plane; the bench path drops it.
/// One node's answer to the manifest, before anything ships: the open stream, the hashes it lacks and its
/// drive-space report.
struct Offer { s: TcpStream, missing: Vec<String>, space: Option<SpaceReport> }

/// Phase 1 of a node's sync: connect, check the version, offer the manifest (and the drafter files that will follow)
/// and read the node's answer. Nothing ships here.
fn head_sync_one(node: &NodeInfo, model_id: &str, artifacts: &[Artifact],
                 draft: Option<&(String, Vec<Artifact>)>) -> Result<Offer> {
    let mut s = TcpStream::connect_timeout(&node.addr, Duration::from_secs(10))
        .with_context(|| format!("connect {}", node.addr))?;
    s.set_nodelay(true).ok();
    harden_or_warn(&s, "head"); // TP-4D (C3/C9): both ends of the retained stream
    head_hello(&mut s, node, &binary_version())?;
    send_msg(&mut s, &Msg::Manifest { model_id: model_id.into(), artifacts: artifacts.to_vec(),
                                      draft: draft.map(|d| d.1.clone()).unwrap_or_default() })?;
    match recv_msg(&mut s)? {
        Msg::Missing { hashes, space } => Ok(Offer { s, missing: hashes, space }),
        Msg::Error { msg } => bail!("node error: {msg}"),
        _ => bail!("expected Missing"),
    }
}

/// Owner 2026-11-02: every node's drive must hold what this session would store on it; checked after every node
/// answered the manifest and BEFORE the first byte ships. A short drive stops the session: every node is told why
/// and the head returns the refusal (the callers print it and exit).
fn space_check(nodes: &[NodeInfo], offers: &mut [Offer]) -> Result<()> {
    let rows: Vec<(i32, String, Option<SpaceReport>)> = nodes.iter().zip(offers.iter()).enumerate()
        .map(|(i, (n, o))| ((i + 1) as i32, n.addr.ip().to_string(), o.space.clone())).collect();
    let short = space_shortfalls(&rows);
    if short.is_empty() { return Ok(()); }
    let msg = space_refusal(&short, nodes.len());
    for o in offers.iter_mut() {
        let _ = send_msg(&mut o.s, &Msg::Error { msg: msg.clone() });
    }
    bail!(msg)
}

/// Phase 2 of a node's sync: ship what it lacks, then Ready / Config / the drafter round (unchanged).
fn head_ship(node: &NodeInfo, node_rank: i32, offer: Offer, model_dir: &Path,
             artifacts: &[Artifact], topology: &[String],
             cfg: &crate::tp::TpConfig,
             draft: Option<&(String, Vec<Artifact>)>,
             srcs: Option<&HashMap<String, crate::shard_plan::PlanSrc>>) -> Result<TcpStream> {
    let Offer { mut s, missing, .. } = offer;
    let by_hash: HashMap<&str, &Artifact> = artifacts.iter().map(|a| (a.hash.as_str(), a)).collect();
    let total: u64 = missing.iter().filter_map(|h| by_hash.get(h.as_str())).map(|a| a.size).sum();
    crate::deprintln!("[head] {} (rank {node_rank}) needs {} / {} artifacts ({:.2} GB)", node.hostname,
              missing.len(), artifacts.len(), total as f64 / 1e9);
    let t0 = std::time::Instant::now();
    let prog = Progress::new(format!("shipping the model to {} (rank {node_rank})", node.hostname), total, Unit::Bytes);
    for h in &missing {
        let a = by_hash.get(h.as_str()).context("node asked for unknown hash")?;
        send_msg(&mut s, &Msg::BlobHeader { hash: a.hash.clone(), size: a.size })?;
        send_artifact(&mut Counted { inner: &mut s, prog: &prog }, model_dir, a, srcs)?;
    }
    prog.finish();
    match recv_msg(&mut s)? {
        Msg::Ready { model_dir } => {
            let secs = t0.elapsed().as_secs_f64();
            crate::deprintln!("[head] {} (rank {node_rank}) READY — model at {} ({:.2} GB in {:.1}s = {:.2} GB/s)",
                      node.hostname, model_dir, total as f64/1e9, secs,
                      if secs > 0.0 { total as f64/1e9/secs } else { 0.0 });
            let mut node_cfg = cfg.clone();
            node_cfg.node_rank = node_rank;
            // B32: this node got its rank's shard dir (load it, not the head's path). TPSHIP (2026-11-07): an EXL3 pack the
            // shard plan does not cover (the 27B dense / 35B / 122B MoE EXL3 packs, ...) now ships WHOLE through the same
            // blob cache — the node loads that copy too, never its own disk at the head's path.
            node_cfg.ship_shards = srcs.is_some() || (cfg.exl3 && !artifacts.is_empty());
            node_cfg.topology = topology.to_vec();
            // CLI-1: the head's resolved option registry rides the config (v23), taken at ship time
            // so every option the head resolved before its sync reaches the node.
            node_cfg.opts = crate::opts::snapshot();
            send_msg(&mut s, &Msg::Config(node_cfg.clone()))?;
            crate::deprintln!("[head] shipped config to {} (rank {node_rank}/{})", node.hostname, node_cfg.world);

            // v5 — the DFlash2 draft round: ALWAYS exactly one DraftManifest after Config (empty
            // artifacts = none), so the node's receive state machine is unconditional and can
            // never block waiting for a round the head decided not to send.
            match draft {
                Some((draft_id, draft_arts)) if !draft_arts.is_empty() => {
                    send_msg(&mut s, &Msg::DraftManifest { model_id: draft_id.clone(),
                                                          artifacts: draft_arts.clone() })?;
                    let src_dir = Path::new(&cfg.df2_draft_dir);
                    let dmissing = match recv_msg(&mut s)? {
                        Msg::Missing { hashes, .. } => hashes,
                        _ => bail!("expected draft Missing from {}", node.hostname),
                    };
                    let dby_hash: HashMap<&str, &Artifact> =
                        draft_arts.iter().map(|a| (a.hash.as_str(), a)).collect();
                    let dtotal: u64 = dmissing.iter().filter_map(|h| dby_hash.get(h.as_str())).map(|a| a.size).sum();
                    crate::deprintln!("[head] {} drafter: {} / {} artifacts ({:.2} GB)",
                              node.hostname, dmissing.len(), draft_arts.len(), dtotal as f64 / 1e9);
                    let dt0 = std::time::Instant::now();
                    let dprog = Progress::new(format!("shipping the drafter to {}", node.hostname), dtotal, Unit::Bytes);
                    for h in &dmissing {
                        let a = dby_hash.get(h.as_str()).context("node asked for unknown draft hash")?;
                        send_msg(&mut s, &Msg::BlobHeader { hash: a.hash.clone(), size: a.size })?;
                        send_blob(&mut Counted { inner: &mut s, prog: &dprog }, &src_dir.join(&a.logical))?;
                    }
                    dprog.finish();
                    match recv_msg(&mut s)? {
                        Msg::DraftReady { model_dir } => {
                            let secs = dt0.elapsed().as_secs_f64();
                            crate::deprintln!("[head] {} drafter READY at {} ({:.2} GB in {:.1}s)",
                                      node.hostname, model_dir, dtotal as f64/1e9, secs);
                        }
                        Msg::Error { msg } => bail!("node draft error: {msg}"),
                        _ => bail!("expected DraftReady"),
                    }
                }
                _ => {
                    send_msg(&mut s, &Msg::DraftManifest { model_id: String::new(), artifacts: Vec::new() })?;
                    match recv_msg(&mut s)? {
                        Msg::DraftReady { .. } => {}
                        Msg::Error { msg } => bail!("node draft error: {msg}"),
                        _ => bail!("expected DraftReady (none)"),
                    }
                }
            }
            Ok(s)
        }
        Msg::Error { msg } => bail!("node error: {msg}"),
        _ => bail!("expected Ready"),
    }
}

/// Run as the head: discover nodes (or use explicit addrs), assign ranks 1..N-1 deterministically,
/// then sync the model to each node's OWN shard. Sets the process-global topology before returning
/// (so `bring_up_head` can resolve its N-way partners).
pub fn run_head(model_dir: &Path, explicit: Option<Vec<SocketAddr>>, discover_wait: Duration,
                cfg: &crate::tp::TpConfig)
    -> Result<Vec<NodeInfo>>
{
    run_head_retain(model_dir, explicit, discover_wait, cfg).map(|(n, _)| n)
}

/// TP-D: `run_head` that RETAINS the per-node control streams (rank-indexed: `streams[i]` goes to
/// rank i + 1) — the EXL3 TP serve runs its Step / HeadFlag / Ready / Shutdown control plane over
/// them. TPSHIP (2026-11-07): an EXL3 pack ships through the blob cache like every other model — per-rank shards
/// when the shard plan covers it (Qwen3.8-Flash-Next, B32), otherwise the whole pack replicated to every node.
pub fn run_head_retain(model_dir: &Path, explicit: Option<Vec<SocketAddr>>, discover_wait: Duration,
                       cfg: &crate::tp::TpConfig)
    -> Result<(Vec<NodeInfo>, Vec<TcpStream>)>
{
    let world = cfg.world.max(1);
    // 2026-11-07: refuse a pack the engine cannot serve BEFORE shipping it (a ModelOpt pack whose formats the FP4-Local
    // reader does not serve used to ship ~98 GB to every node before a node refused it at open).
    if crate::fp4l::pack::is_fp4l_dir(model_dir) { crate::fp4l::pack::preflight(model_dir)?; }
    crate::deprintln!("[head] {} — building manifest for {} (world {world}) ...", hostname(), model_dir.display());
    // TPSHIP (2026-11-07): the TP-A (EXL3, D-T0-6) legacy path shipped NOTHING — every node loaded its OWN local
    // pack copy at the head's path, unchecked; with the models on .11 only (owner 2026-10-26) a node without that
    // copy failed its boot ("EXL3 pack <head path>: config.json missing"). An EXL3 pack the shard plan does not
    // cover now ships whole (build_manifest: replicated to every node, content-addressed, a re-run sends nothing).
    // B32: each node gets ONLY its rank's shard dir through the blob cache — always, when the shard plan covers
    // the model (no switch, owner 2026-10-02).
    let ships = if crate::shard_plan::applies(model_dir, world as usize).is_some() { Some(ship_plans(model_dir, world)?) } else { None };
    let (model_id, per_rank) = if let Some(sh) = &ships {
        (sh[0].model_id.clone(), sh.iter().map(|r| r.artifacts.clone()).collect::<Vec<_>>())
    } else {
        build_manifest(model_dir, world)?
    };
    let total: u64 = per_rank.iter().flatten().map(|a| a.size).sum();
    crate::deprintln!("[head] manifest '{model_id}': {} artifacts, {:.2} GB{}", total_artifacts(&per_rank), total as f64/1e9,
              if ships.is_some() { " (B32: per-rank shards through the blob cache)" }
              else if cfg.exl3 { " (EXL3: the whole pack, replicated to every node through the blob cache)" } else { "" });

    let _ = crate::tp::rdma_devs();   // resolve + print the RDMA rails before shipping anything
    let ifaces = roce_interfaces();
    let mut nodes: Vec<NodeInfo> = match explicit {
        Some(addrs) => addrs.into_iter()
            .map(|a| NodeInfo { hostname: a.ip().to_string(), addr: a }).collect(),
        None => {
            crate::deprintln!("[head] discovering nodes (UDP broadcast, {}s) ...", discover_wait.as_secs());
            let n = discover_nodes(discover_wait)?;
            if n.is_empty() { bail!("no nodes discovered — is a --node running? (or pass --nodes <ip:port>)"); }
            for x in &n { crate::deprintln!("  [head] found node '{}' at {}", x.hostname, x.addr); }
            n
        }
    };
    let need = (world - 1) as usize;
    if nodes.len() != need {
        bail!("--tp {world} needs exactly {need} node(s), got {} — start exactly {need} --node (or pass exactly {need} --nodes <ip:port>)",
              nodes.len());
    }
    assign_ranks(&mut nodes, &ifaces);
    let topology = build_topology(&nodes, &ifaces, world)?;
    crate::tp::set_topology(topology.clone());
    // FLASHNEXT-DF2-C (ISSUES 116): the EXL3 serve ships the drafter too (this entry passed `None`).
    let draft = draft_for_cfg(cfg);
    let df2_draft = draft; // master's DF2-class decision (ISSUES 116), kept for every non-Naive session
    // ISSUES 98 (2026-10-20): a Naive SERVE session ships its drafter through the same draft round the DFlash2 /
    // DSpark sessions use (`run_head_session`): the sharded drafter loads its shard on EVERY rank, and a node must
    // never depend on a hand-copied local dir. The head set `df2_draft_dir` only when the drafter is on its disk and
    // speculation is on (`naive::serve::head_draft_dir`). Every other session that comes here ships no drafter.
    let draft: Option<(String, Vec<Artifact>)> = if cfg.exl3_mode.starts_with("naive-serve:") && !cfg.df2_draft_dir.is_empty() {
        match draft_manifest(Path::new(&cfg.df2_draft_dir)) {
            Ok((id, arts)) => {
                let dtotal: u64 = arts.iter().map(|a| a.size).sum();
                println!("[head] Naive drafter manifest '{id}': {} artifacts, {:.2} GB — ships to every node", arts.len(), dtotal as f64 / 1e9);
                Some((id, arts))
            }
            Err(e) => bail!("the Naive drafter at {} could not be listed for the nodes ({e:#}); start with --spec-source none to serve without it", cfg.df2_draft_dir),
        }
    } else { None };
    // the port resolution: a naive-serve session ships the Naive drafter decision above; every other session
    // keeps master's DF2-class decision (FLASHNEXT-DF2-C) instead of the None the block above yields for it.
    let draft = if cfg.exl3_mode.starts_with("naive-serve:") { draft } else { df2_draft };
    // owner 2026-11-02: every node answers the manifest first (with its drive-space report); the head ships to none
    // of them until all have room
    let mut offers = Vec::with_capacity(nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        let rank = i + 1;
        let mid = ships.as_ref().map_or(model_id.as_str(), |sh| sh[rank].model_id.as_str());
        offers.push(head_sync_one(node, mid, &per_rank[rank], draft.as_ref())?);
    }
    space_check(&nodes, &mut offers)?;
    let mut streams = Vec::with_capacity(nodes.len());
    for (i, (node, offer)) in nodes.iter().zip(offers).enumerate() {
        let rank = (i + 1) as i32;
        // Bench sessions ship no drafter (bench nodes never load a round; the bench wire adds
        // only the empty-DraftManifest round to the pre-v5 behavior).
        let srcs = ships.as_ref().map(|sh| &sh[rank as usize].srcs);
        streams.push(head_ship(node, rank, offer, model_dir, &per_rank[rank as usize], &topology, cfg, draft.as_ref(), srcs)?);
    }
    crate::deprintln!("[head] all {} node(s) synced.", nodes.len());
    sync_only_exit(cfg, nodes.len());
    Ok((nodes, streams))
}

/// B32 `--tp-sync-only`: the head stops right after every node synced (no link, no model load).
fn sync_only_exit(cfg: &crate::tp::TpConfig, n: usize) {
    if cfg.sync_only {
        eprintln!("SYNC_ONLY head: {n} node(s) synced (bytes per node above); exiting without a model load");
        std::process::exit(0);
    }
}

fn total_artifacts(per_rank: &[Vec<Artifact>]) -> usize {
    per_rank.iter().flatten().count()
}

/// Run as the head for a TP SERVING session (TP item A): identical to `run_head` (same manifest,
/// discovery, and blob push), but the sync connections are RETAINED and returned — the whole serving
/// control plane (calibration table, per-step events, shutdown) then runs over them as
/// `tp_serve::ServingMsg`. The node count requirement is world-aware: exactly `world - 1` nodes.
/// Returns the retained streams RANK-INDEXED (`streams[i]` is the stream to rank `i + 1`), so the
/// head can fan out `CalibTable` / `Step` / `Shutdown` to every node (world > 2) and wait for
/// `Ready` from every node. The node side is `run_node`'s retained stream.
/// The drafter the head ships to every node this session, decided from the shipped config alone:
/// a serve session whose spec source is a DFlash2 / DSpark / DFlash source and whose
/// `df2_draft_dir` is resolved. ONE decision for BOTH head entries — FLASHNEXT-DF2-C
/// (ISSUES 116, 2026-10-23): `run_head_retain` (the EXL3 serve) passed `None` here while
/// `run_head_session` (nvfp4) computed this block inline, so an EXL3 node never received the
/// drafter, re-parsed the head's argv, opened the head's filesystem path on its own disk and —
/// without a local copy — fell to the 27B default geometry while the head ENGAGED.
/// A manifest failure is loud but not fatal: the head's own round load fails the same way and
/// the boot agree keeps the ranks consistent.
/// P14: `dflash` (the v1 BLOCK drafter) ships through the same content-addressed slot.
pub(crate) fn draft_for_cfg(cfg: &crate::tp::TpConfig) -> Option<(String, Vec<Artifact>)> {
    let src_parsed = crate::batch::SpecSource::from_cli(&cfg.spec_source)
        .unwrap_or(crate::batch::SpecSource::Mtp);
    if !(cfg.mode_serve
        && (crate::batch::is_df2_src(src_parsed)
            || matches!(src_parsed, crate::batch::SpecSource::Dspark)
            || matches!(src_parsed, crate::batch::SpecSource::DFlash))
        && !cfg.df2_draft_dir.is_empty()) {
        return None;
    }
    match draft_manifest(Path::new(&cfg.df2_draft_dir)) {
        Ok((id, arts)) => {
            let dtotal: u64 = arts.iter().map(|a| a.size).sum();
            crate::deprintln!("[head] draft manifest '{id}': {} artifacts, {:.2} GB — ships to every node",
                      arts.len(), dtotal as f64 / 1e9);
            Some((id, arts))
        }
        Err(e) => {
            eprintln!("[head] WARN: draft manifest build FAILED ({e:#}) — nodes get NO drafter; \
                       the head's round load fails the same way and the boot agree refuses a split outcome");
            None
        }
    }
}

pub fn run_head_session(model_dir: &Path, explicit: Option<Vec<SocketAddr>>, discover_wait: Duration,
                        cfg: &crate::tp::TpConfig)
    -> Result<(Vec<NodeInfo>, Vec<TcpStream>)>
{
    let world = cfg.world.max(1);
    crate::deprintln!("[head] {} — building manifest for {} (world {world}) ...", hostname(), model_dir.display());
    // B32: per-rank shard dirs instead of the whole-model replica — always, when the shard plan covers the model
    let ships = if crate::shard_plan::applies(model_dir, world as usize).is_some() { Some(ship_plans(model_dir, world)?) } else { None };
    let (model_id, per_rank) = match &ships {
        Some(sh) => (sh[0].model_id.clone(), sh.iter().map(|r| r.artifacts.clone()).collect::<Vec<_>>()),
        None => build_manifest(model_dir, world)?,
    };
    let total: u64 = per_rank.iter().flatten().map(|a| a.size).sum();
    crate::deprintln!("[head] manifest '{model_id}': {} artifacts, {:.2} GB{}", total_artifacts(&per_rank), total as f64/1e9,
              if ships.is_some() { " (B32: per-rank shards through the blob cache)" } else { "" });

    let ifaces = roce_interfaces();
    let mut nodes: Vec<NodeInfo> = match explicit {
        Some(addrs) => addrs.into_iter()
            .map(|a| NodeInfo { hostname: a.ip().to_string(), addr: a }).collect(),
        None => {
            crate::deprintln!("[head] discovering nodes (UDP broadcast, {}s) ...", discover_wait.as_secs());
            let n = discover_nodes(discover_wait)?;
            if n.is_empty() { bail!("no nodes discovered — is a --node running? (or pass --nodes <ip:port>)"); }
            for x in &n { crate::deprintln!("  [head] found node '{}' at {}", x.hostname, x.addr); }
            n
        }
    };
    let need = (world - 1) as usize;
    if nodes.len() != need {
        bail!("--tp {world} serving needs exactly {need} node(s), got {} — start exactly {need} --node (or pass exactly {need} --nodes <ip:port>)",
              nodes.len());
    }
    assign_ranks(&mut nodes, &ifaces);
    let topology = build_topology(&nodes, &ifaces, world)?;
    crate::tp::set_topology(topology.clone());
    // v5 — the DFlash2 drafter rides the same sync (gap fix 2026-08-23): on serve sessions with
    // a DF2 spec source and a resolved --draft-dir, ship the WHOLE artifact (replicated, content-
    // addressed) so every node loads it from its blob cache instead of a hand-copied local dir.
    // A manifest failure is loud but not fatal here — the head's own round load fails the same
    // way and CalibTable's df2_round=false keeps all ranks consistently on MTP.
    // WI1: the DSpark source ships its artifact through the same content-addressed sync.
    let draft = draft_for_cfg(cfg);

    // Sync every node and RETAIN every node's sync stream, RANK-INDEXED (streams[i] == rank i+1).
    // The serving control plane (CalibTable, per-step Step, Shutdown) must fan out to ALL world-1
    // nodes; dropping rank 2..N-1's streams here is exactly what deadlocked world>2 bring-up (those
    // nodes reached node_serve_tp and blocked forever on a CalibTable the head never sent them).
    // owner 2026-11-02: every node answers the manifest first (with its drive-space report); nothing ships to any
    // node until all have room
    let mut offers = Vec::with_capacity(need);
    for (i, node) in nodes.iter().enumerate() {
        let rank = i + 1;
        let mid = ships.as_ref().map_or(model_id.as_str(), |sh| sh[rank].model_id.as_str());
        offers.push(head_sync_one(node, mid, &per_rank[rank], draft.as_ref())?);
    }
    space_check(&nodes, &mut offers)?;
    let mut streams: Vec<TcpStream> = Vec::with_capacity(need);
    for (i, (node, offer)) in nodes.iter().zip(offers).enumerate() {
        let rank = (i + 1) as i32;
        let srcs = ships.as_ref().map(|sh| &sh[rank as usize].srcs);
        let stream = head_ship(node, rank, offer, model_dir, &per_rank[rank as usize],
                               &topology, cfg, draft.as_ref(), srcs)?;
        streams.push(stream);
    }
    anyhow::ensure!(streams.len() == need,
        "expected {need} retained serving control stream(s), got {}", streams.len());
    crate::deprintln!("[head] {} node(s) synced; all control streams RETAINED for the serving session", streams.len());
    sync_only_exit(cfg, streams.len());
    Ok((nodes, streams))
}

// ---------------------------------------------------------------------------------------------------
// Blob cache management (ops CLI: --cached-models-list / --cached-models-remove /
// --cached-models-remove-all — MODEL-centric; the old blob-centric names are deprecated aliases)
// ---------------------------------------------------------------------------------------------------

/// Walk every symlink under `cache/models/<model_id>/`, calling `f(model_id, link_path, target)`.
fn walk_model_links(mut f: impl FnMut(&str, &Path, &Path)) {
    let mdir = cache_root().join("models");
    let Ok(models) = std::fs::read_dir(&mdir) else { return };
    for e in models.flatten() {
        let mid = e.file_name().to_string_lossy().to_string();
        let mut stack = vec![e.path()];
        while let Some(d) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for ent in rd.flatten() {
                let p = ent.path();
                let Ok(ft) = ent.file_type() else { continue };
                if ft.is_dir() { stack.push(p); continue; }
                if ft.is_symlink() {
                    if let Ok(t) = std::fs::read_link(&p) { f(&mid, &p, &t); }
                }
            }
        }
    }
}

fn fmt_gib(b: u64) -> String { format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64) }

/// The blobs (by hash) referenced by the assembled model dir `model_id` (empty if absent).
fn model_blob_hashes(model_id: &str) -> Vec<String> {
    let mut v = Vec::new();
    walk_model_links(|mid, _p, t| {
        if mid == model_id {
            if let Some(h) = t.file_name() { v.push(h.to_string_lossy().to_string()); }
        }
    });
    v.sort(); v.dedup();
    v
}

/// Blob file size in `blobs/<hash>` (0 if missing).
fn blob_size(hash: &str) -> u64 {
    std::fs::metadata(blob_path(hash)).map(|m| m.len()).unwrap_or(0)
}

/// `--cached-models-list`: ONE line per assembled MODEL — name, total size (sum of its blob
/// files), blob count — plus a cache summary and any orphan-blob / interrupted-fetch tail.
pub fn list_cached_models() -> Result<()> {
    let mdir = cache_root().join("models");
    let mut names: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&mdir) {
        for e in rd.flatten() {
            let mid = e.file_name().to_string_lossy().to_string();
            if e.file_type().map(|ft| ft.is_dir()).unwrap_or(false) { names.push(mid); }
        }
    }
    names.sort();
    // B32: per model — its total (every blob it links), its blob count, and how much of it is SHARED with another
    // model (those blobs stay when this model is removed).
    let per_model: Vec<(String, Vec<String>)> = names.iter().map(|m| (m.clone(), model_blob_hashes(m))).collect();
    let mut refcount: HashMap<String, usize> = HashMap::new();
    for (_, hs) in &per_model { for h in hs { *refcount.entry(h.clone()).or_default() += 1; } }
    let mut rows: Vec<(String, u64, usize, usize, u64)> = Vec::new();
    let mut all_blobs: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (mid, hashes) in &per_model {
        let total: u64 = hashes.iter().map(|h| blob_size(h)).sum();
        let shared: Vec<&String> = hashes.iter().filter(|h| refcount[*h] > 1).collect();
        let shared_b: u64 = shared.iter().map(|h| blob_size(h)).sum();
        all_blobs.extend(hashes.iter().cloned());
        rows.push((mid.clone(), total, hashes.len(), shared.len(), shared_b));
    }
    let blob_dir = cache_root().join("blobs");
    let mut blob_total = 0u64;
    let mut n_blobs = 0usize;
    let mut tmp: Vec<(String, u64)> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&blob_dir) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let size = e.metadata().map(|m| m.len()).unwrap_or(0);
            if name.starts_with("tmp.") { tmp.push((name, size)); }
            else { blob_total += size; n_blobs += 1; }
        }
    }
    let orphans: Vec<String> = std::fs::read_dir(&blob_dir).into_iter().flatten()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| !n.starts_with("tmp.") && !all_blobs.contains(n))
        .collect();
    println!("cached models ({}):", rows.len());
    for (mid, total, n, ns, sb) in &rows {
        if *n == 0 {
            println!("  {mid}  (empty: nothing cached — a placeholder left by the legacy EXL3 sync; --cached-models-remove clears it)");
        } else if *ns > 0 {
            println!("  {mid}  {:>12}  {n} blob(s)  ({ns} blob(s), {} shared with other models)", fmt_gib(*total), fmt_gib(*sb));
        } else {
            println!("  {mid}  {:>12}  {n} blob(s)", fmt_gib(*total));
        }
    }
    println!("cache {} — {} blob(s), {} total", blob_dir.display(), n_blobs, fmt_gib(blob_total));
    if !orphans.is_empty() {
        let osz: u64 = orphans.iter().map(|h| blob_size(h)).sum();
        println!("-- {} orphan blob(s), {} (referenced by no model; --cached-models-remove-all reclaims)", orphans.len(), fmt_gib(osz));
    }
    if !tmp.is_empty() {
        let t: u64 = tmp.iter().map(|x| x.1).sum();
        println!("-- {} interrupted-fetch partial(s), {} reclaimable (tmp.*)", tmp.len(), fmt_gib(t));
    }
    Ok(())
}

/// Match a model id by exact name or unique prefix (a 4-char floor, like the old blob op).
fn resolve_model(id: &str) -> Result<String> {
    let mdir = cache_root().join("models");
    let mut matches: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&mdir) {
        for e in rd.flatten() {
            let mid = e.file_name().to_string_lossy().to_string();
            if e.file_type().map(|ft| ft.is_dir()).unwrap_or(false)
                && (mid == id || mid.starts_with(id)) { matches.push(mid); }
        }
    }
    match matches.len() {
        1 => Ok(matches.pop().unwrap()),
        0 => bail!("no cached model matching '{id}' in {}", mdir.display()),
        n => bail!("'{id}' matches {n} models — give a longer prefix"),
    }
}

/// `--cached-models-remove <id>`: remove ONE cached MODEL — its assembled dir plus the blobs
/// referenced by NO other model (shared blobs stay). Interrupted-fetch partials are never
/// referenced and are reclaimed here too. Other models are untouched.
pub fn remove_cached_model(id: &str) -> Result<()> {
    if id.len() < 4 { bail!("refusing to match an id shorter than 4 characters"); }
    let mid = resolve_model(id)?;
    let hashes = model_blob_hashes(&mid);
    let mdir = cache_root().join("models").join(&mid);
    std::fs::remove_dir_all(&mdir).with_context(|| format!("remove model dir {mdir:?}"))?;
    // Delete THIS model's blobs that no remaining model references (B32: only its own — other models'
    // orphans and in-flight tmp.* partials of a running fetch are left to --cached-models-remove-all).
    let mut refs: std::collections::HashSet<String> = std::collections::HashSet::new();
    walk_model_links(|_m, _p, t| {
        if let Some(h) = t.file_name() { refs.insert(h.to_string_lossy().to_string()); }
    });
    let mut removed = 0u64;
    let mut n_removed = 0usize;
    for h in &hashes {
        if refs.contains(h) { continue; }
        let p = blob_path(h);
        if let Ok(md) = std::fs::metadata(&p) {
            let _ = std::fs::remove_file(&p);
            removed += md.len(); n_removed += 1;
        }
    }
    let kept = hashes.iter().filter(|h| blob_path(h).exists()).count();
    println!("removed model {mid} — {n_removed} unreferenced blob(s) reclaimed, {}",
             fmt_gib(removed));
    println!("  {kept} of its blob(s) still referenced by other models were kept");
    Ok(())
}

/// `--cached-models-remove-all`: clear the whole cache (blobs incl. tmp.* partials + assembled
/// model dirs). The next head run re-syncs from scratch.
pub fn remove_all_cached_models() -> Result<()> {
    let dir = cache_root().join("blobs");
    let mut total = 0u64;
    let mut n = 0usize;
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            total += e.metadata().map(|m| m.len()).unwrap_or(0);
            let _ = std::fs::remove_file(e.path());
            n += 1;
        }
    }
    let mdir = cache_root().join("models");
    if mdir.exists() { std::fs::remove_dir_all(&mdir).context("remove assembled model dirs")?; }
    println!("cleared {n} blob(s), {} — assembled model dirs removed; next head run re-syncs from scratch",
             fmt_gib(total));
    Ok(())
}

// -- deprecated aliases (the old blob-centric names; warn once, route to the model-centric ops) --

/// DEPRECATED alias for `--cached-models-list` (was: one line per BLOB). Warns once.
pub fn list_model_blobs() -> Result<()> {
    eprintln!("warning: --list-model-blobs is deprecated — use --cached-models-list (lists MODELS, not blobs)");
    list_cached_models()
}

/// DEPRECATED alias for `--cached-models-remove <model-id>` (was: remove one BLOB by hash).
/// The argument is now a MODEL name/prefix.
pub fn remove_model_blob(id: &str) -> Result<()> {
    eprintln!("warning: --remove-model-blob is deprecated — use --cached-models-remove <model> (removes a MODEL, not a blob)");
    remove_cached_model(id)
}

/// DEPRECATED alias for `--cached-models-remove-all`. Warns once.
pub fn clear_model_blobs() -> Result<()> {
    eprintln!("warning: --clear-model-blobs is deprecated — use --cached-models-remove-all");
    remove_all_cached_models()
}

/// Tests that point the process-global `--tp-cache` at a temp dir hold this lock (they would race otherwise).
#[cfg(test)]
pub(crate) static CACHE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    /// B32: the cache ops on per-rank shard models in a THROWAWAY cache (--tp-cache): truthful per-model totals
    /// with shared blobs, remove deletes only the model's unshared blobs (a shared blob, another model's orphan
    /// and an in-flight tmp.* partial stay), an empty manifest leaves no placeholder dir, remove-all clears.
    #[test]
    fn b32_cache_ops_on_shard_models() {
        let tmp = std::env::temp_dir().join(format!("b32_cache_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let _g = CACHE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner()); // tp-cache is process-global
        crate::opts::set(crate::opt!("tp-cache"), &tmp.to_string_lossy());
        let mk = |name: &str, bytes: &[u8]| -> Artifact {
            let mut h = Sha256::new(); h.update(bytes);
            let hash = hex(&h.finalize());
            let p = blob_path(&hash);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, bytes).unwrap();
            Artifact { logical: name.into(), hash, size: bytes.len() as u64 }
        };
        let shared = mk("ngram_embedding.safetensors", &[7u8; 4096]);
        let a1 = mk("seg/experts-L00.safetensors", &[1u8; 1000]);
        let b1 = mk("seg/experts-L00.safetensors", &[2u8; 2000]);
        let orphan = mk("x", &[9u8; 10]);
        std::fs::write(cache_root().join("blobs").join("tmp.123.abc"), b"partial").unwrap();
        let da = assemble_model_dir("M@exl3-w2-r1-interleave", &[shared.clone(), a1.clone()]).unwrap();
        assemble_model_dir("M@exl3-w4-r1-interleave", &[shared.clone(), b1.clone()]).unwrap();
        assert!(da.join("seg/experts-L00.safetensors").exists());
        // empty manifest (legacy EXL3): no placeholder dir
        let pe = assemble_model_dir("Legacy-exl3", &[]).unwrap();
        assert!(!pe.exists(), "an empty manifest must not leave a placeholder model dir");
        assert_eq!(model_blob_hashes("M@exl3-w2-r1-interleave").len(), 2);
        list_cached_models().unwrap();
        // re-assembling the same id with another manifest drops the stale link
        assemble_model_dir("M@exl3-w2-r1-interleave", &[shared.clone()]).unwrap();
        assert!(!da.join("seg/experts-L00.safetensors").exists(), "stale link survived a re-assembly");
        assemble_model_dir("M@exl3-w2-r1-interleave", &[shared.clone(), a1.clone()]).unwrap();
        remove_cached_model("M@exl3-w2-r1-interleave").unwrap();
        assert!(!blob_path(&a1.hash).exists(), "the removed model's unshared blob must go");
        assert!(blob_path(&shared.hash).exists(), "a blob shared with another model must stay");
        assert!(blob_path(&b1.hash).exists(), "another model's blob must stay");
        assert!(blob_path(&orphan.hash).exists(), "another model's orphan is not this model's to delete");
        assert!(cache_root().join("blobs/tmp.123.abc").exists(), "an in-flight partial must stay");
        remove_all_cached_models().unwrap();
        assert!(!blob_path(&shared.hash).exists() && !cache_root().join("models").exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }


    fn node(hostname: &str, ip: &str, port: u16) -> NodeInfo {
        NodeInfo {
            hostname: hostname.to_string(),
            addr: SocketAddr::new(ip.parse().unwrap(), port),
        }
    }

    // Synthetic RoCE rails: rail 1 = 192.0.2.0/24, rail 2 = 198.51.100.0/24.
    fn rails() -> Vec<Roce> {
        vec![
            Roce {
                ib: "rocep1s0f1".into(),
                netdev: "r1".into(),
                ip: "192.0.2.1".parse().unwrap(),
                bcast: "192.0.2.255".parse().unwrap(),
                mask: 0xffffff00,
                rail: 1,
            },
            Roce {
                ib: "roceP2p1s0f1".into(),
                netdev: "r2".into(),
                ip: "198.51.100.1".parse().unwrap(),
                bcast: "198.51.100.255".parse().unwrap(),
                mask: 0xffffff00,
                rail: 2,
            },
        ]
    }

    #[test]
    fn deterministic_rank_order_prefers_rail_then_ip() {
        let ifaces = rails();
        // rail-1 .13, rail-1 .12, rail-2 .99, non-RoCE .1 -> expected order.
        let mut nodes = vec![
            node("a", "10.0.0.1", 29500),
            node("b", "198.51.100.99", 29500),
            node("c", "192.0.2.13", 29500),
            node("d", "192.0.2.12", 29500),
        ];
        assign_ranks(&mut nodes, &ifaces);
        let order: Vec<String> = nodes.iter().map(|n| n.hostname.clone()).collect();
        assert_eq!(order, vec!["d", "c", "b", "a"], "rail desc, then ip asc");

        // Reassign with the same set in a different input order -> identical result.
        let mut shuffled = vec![
            node("d", "192.0.2.12", 29500),
            node("b", "198.51.100.99", 29500),
            node("a", "10.0.0.1", 29500),
            node("c", "192.0.2.13", 29500),
        ];
        assign_ranks(&mut shuffled, &ifaces);
        let order2: Vec<String> = shuffled.iter().map(|n| n.hostname.clone()).collect();
        assert_eq!(order, order2, "order must be input-order independent");
    }

    #[test]
    fn topology_is_full_and_indexed_by_rank() {
        let ifaces = rails();
        let mut nodes = vec![
            node("b", "198.51.100.99", 29500),
            node("c", "192.0.2.13", 29500),
            node("d", "192.0.2.12", 29500),
        ];
        assign_ranks(&mut nodes, &ifaces);
        let topo = build_topology(&nodes, &ifaces, 4).unwrap();
        assert_eq!(topo.len(), 4);
        assert_eq!(topo[0], "192.0.2.1"); // head = rail 1
        assert_eq!(topo[1], "192.0.2.12"); // rail 1, lowest ip
        assert_eq!(topo[2], "192.0.2.13"); // rail 1, next ip
        assert_eq!(topo[3], "198.51.100.99"); // rail 2 last
    }

    #[test]
    fn manifest_filtering_per_rank_and_world2_identity() {
        // Logical paths exactly as model_files() would emit relative to the model dir.
        let logicals = vec![
            "config.json",
            "inference/config.json",
            "rank1/weights.safetensors",
            "rank1/dspark_stage0.safetensors",
            "rank2/weights.safetensors",
            "rank3/weights.safetensors",
            "rank0/weights.safetensors",
        ];

        // world==2 with rank1/ present: rank1 manifest == the pre-P4 set (rank1/* + 2 root files).
        let got = |rank: usize, sharded: bool| -> Vec<&str> {
            logicals.iter().copied().filter(|rel| include_for_rank(rel, rank, sharded)).collect()
        };
        assert_eq!(got(1, true), vec![
            "config.json",
            "inference/config.json",
            "rank1/weights.safetensors",
            "rank1/dspark_stage0.safetensors",
        ]);

        // world==4 with rank1/..rank3/ present: each node gets only its own shard.
        assert_eq!(got(2, true), vec![
            "config.json",
            "inference/config.json",
            "rank2/weights.safetensors",
        ]);
        assert_eq!(got(3, true), vec![
            "config.json",
            "inference/config.json",
            "rank3/weights.safetensors",
        ]);

        // rank0's own shard must never leak into any node manifest.
        assert!(!got(1, true).contains(&"rank0/weights.safetensors"));
        assert!(!got(2, true).contains(&"rank0/weights.safetensors"));
        assert!(!got(3, true).contains(&"rank0/weights.safetensors"));

        // Replicate-if-not-divisible: a rank with no dir gets the whole model.
        let all: Vec<&str> = logicals.iter().copied().filter(|rel| include_for_rank(rel, 2, false)).collect();
        assert_eq!(all.len(), logicals.len());
    }
}

#[cfg(test)]
mod pack_manifest_tests {
    use super::*;

    /// A synthetic EXL3-shaped pack: an index referencing two shards + a sidecar, the loader-read
    /// small files, and files the engine never reads (README, LICENSE, pack_scan.json, *.native).
    fn make_pack(root: &Path, name: &str, readme: &str, shard_b: &[u8], gen_cfg: Option<&str>) -> PathBuf {
        let d = root.join(name);
        std::fs::create_dir_all(&d).unwrap();
        let idx = serde_json::json!({"metadata": {}, "weight_map": {
            "w.a.trellis": "model-00001-of-00002.safetensors",
            "w.b.trellis": "model-00002-of-00002.safetensors",
            "mtp.x": "mtp_hyper_connection_mixer_patch.safetensors"}});
        std::fs::write(d.join("model.safetensors.index.json"), idx.to_string()).unwrap();
        std::fs::write(d.join("model-00001-of-00002.safetensors"), b"shard-a").unwrap();
        std::fs::write(d.join("model-00002-of-00002.safetensors"), shard_b).unwrap();
        std::fs::write(d.join("mtp_hyper_connection_mixer_patch.safetensors"), b"patch").unwrap();
        std::fs::write(d.join("ngram_embedding.safetensors"), b"ngram").unwrap();   // sidecar, not in the index
        std::fs::write(d.join("config.json"), b"{\"model_type\":\"qwen4_exp\"}").unwrap();
        std::fs::write(d.join("quantization_config.json"), b"{\"quant_method\":\"exl3\"}").unwrap();
        std::fs::write(d.join("tokenizer.json"), b"{}").unwrap();
        if let Some(g) = gen_cfg { std::fs::write(d.join("generation_config.json"), g).unwrap(); }
        std::fs::write(d.join("README.md"), readme).unwrap();
        std::fs::write(d.join("LICENSE"), b"license").unwrap();
        std::fs::write(d.join("pack_scan.json"), b"{}").unwrap();
        std::fs::write(d.join("config.json.native"), b"{}").unwrap();
        std::fs::write(d.join("qbench_prompts.md"), b"x").unwrap();
        d
    }

    #[test]
    fn manifest_is_the_loader_read_set_and_ignores_readme() {
        let root = std::env::temp_dir().join(format!("packfix_ut_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        // private hash cache for the test process (never the user's ~/.cache/gb10_tp)
        let _g = CACHE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner()); // tp-cache is process-global
        crate::opts::set(crate::opt!("tp-cache"), root.join("cache").to_string_lossy());
        let a = make_pack(&root, "a", "upstream card", b"shard-b", Some("{\"eos_token_id\": 1}"));
        let b = make_pack(&root, "b", "the owner's model card, different length", b"shard-b", Some("{\"eos_token_id\": 1}"));
        let ma = pack_manifest(&a).unwrap();
        let mb = pack_manifest(&b).unwrap();
        let names: Vec<&str> = ma.files.iter().map(|f| f.0.as_str()).collect();
        assert_eq!(names, vec!["config.json", "generation_config.json", "model-00001-of-00002.safetensors",
                               "model-00002-of-00002.safetensors", "model.safetensors.index.json",
                               "mtp_hyper_connection_mixer_patch.safetensors", "ngram_embedding.safetensors",
                               "quantization_config.json", "tokenizer.json"]);
        for ig in ["LICENSE", "README.md", "config.json.native", "pack_scan.json", "qbench_prompts.md"] {
            assert!(ma.ignored.iter().any(|x| x == ig), "{ig} must be ignored: {:?}", ma.ignored);
        }
        assert_eq!(ma.hash, mb.hash, "a README difference must not change the manifest");
        assert!(pack_manifest_diff(&ma.files, &mb.files).is_empty());
        // warm cache: nothing re-hashed
        assert_eq!(pack_manifest(&a).unwrap().hashed_now, 0);

        // one flipped byte in a weight shard (same size) -> mismatch naming exactly that file
        let c = make_pack(&root, "c", "upstream card", b"shard-B", Some("{\"eos_token_id\": 1}"));
        let mc = pack_manifest(&c).unwrap();
        assert_ne!(ma.hash, mc.hash);
        let d = pack_manifest_diff(&ma.files, &mc.files);
        assert_eq!(d.len(), 1, "{d:?}");
        assert!(d[0].contains("model-00002-of-00002.safetensors") && d[0].contains("!="), "{d:?}");

        // a whitespace change in a loader-read JSON -> mismatch naming it
        let e = make_pack(&root, "e", "upstream card", b"shard-b", Some("{\"eos_token_id\":  1}"));
        let me = pack_manifest(&e).unwrap();
        assert_ne!(ma.hash, me.hash);
        let d = pack_manifest_diff(&ma.files, &me.files);
        assert!(d.len() == 1 && d[0].contains("generation_config.json"), "{d:?}");

        // an optional read file present on one side only -> mismatch naming it
        let f = make_pack(&root, "f", "upstream card", b"shard-b", None);
        let mf = pack_manifest(&f).unwrap();
        let d = pack_manifest_diff(&ma.files, &mf.files);
        assert!(d.len() == 1 && d[0].contains("generation_config.json") && d[0].contains("ABSENT on the node"), "{d:?}");

        // an index-referenced shard missing on the node -> recorded, named
        let g = make_pack(&root, "g", "upstream card", b"shard-b", Some("{\"eos_token_id\": 1}"));
        std::fs::remove_file(g.join("model-00001-of-00002.safetensors")).unwrap();
        let mg = pack_manifest(&g).unwrap();
        let d = pack_manifest_diff(&ma.files, &mg.files);
        assert!(d.len() == 1 && d[0].contains("model-00001-of-00002.safetensors") && d[0].contains(PACK_MISSING), "{d:?}");

        // a symlinked view of a pack hashes identically and reuses the cache (canonical key)
        let v = root.join("view");
        std::fs::create_dir_all(&v).unwrap();
        for ent in std::fs::read_dir(&a).unwrap() {
            let ent = ent.unwrap();
            std::os::unix::fs::symlink(ent.path(), v.join(ent.file_name())).unwrap();
        }
        let mv = pack_manifest(&v).unwrap();
        assert_eq!(mv.hash, ma.hash);
        assert_eq!(mv.hashed_now, 0, "a symlink view must hit the real files' cached hashes");

        // the parallel hasher is bitwise the sequential one
        let jobs: Vec<(PathBuf, u64)> = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "LICENSE"]
            .iter().map(|n| (a.join(n), std::fs::metadata(a.join(n)).unwrap().len())).collect();
        let par = sha256_files_parallel(&jobs, 3).unwrap();
        let seq: Vec<String> = jobs.iter().map(|(p, _)| sha256_file_with(p, None).unwrap().0).collect();
        assert_eq!(par, seq);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn peeked_frames_classify() {
        let mut err = Vec::new();
        send_msg(&mut err, &Msg::Error { msg: "PACK MANIFEST MISMATCH: x".into() }).unwrap();
        assert!(matches!(peek_frame(&err), PeekFrame::Error(m) if m.contains("MISMATCH")));
        assert!(matches!(peek_frame(&err[..err.len() - 1]), PeekFrame::Incomplete));
        assert!(matches!(peek_frame(&err[..3]), PeekFrame::Incomplete));
        // the node's legitimate first frame on the served path (its mirror Ready) is not a failure
        let mut ready = Vec::new();
        send_json(&mut ready, &crate::tp_serve::ServingMsg::Ready).unwrap();
        assert!(matches!(peek_frame(&ready), PeekFrame::Other));
    }

    fn getsockopt_i32(s: &TcpStream, level: i32, name: i32) -> i32 {
        use std::os::fd::AsRawFd;
        let mut v: i32 = -1;
        let mut len = std::mem::size_of::<i32>() as libc::socklen_t;
        let r = unsafe { libc::getsockopt(s.as_raw_fd(), level, name, &mut v as *mut i32 as *mut libc::c_void, &mut len) };
        assert_eq!(r, 0, "getsockopt({level},{name}): {}", std::io::Error::last_os_error());
        v
    }

    /// Audit C3/C9: the retained control stream gets keepalive + user timeout (read back from the kernel). What it
    /// does NOT prove: that a silent peer host is then detected — that needs a dropped link, a hardware drill.
    #[test]
    fn harden_control_stream_sets_keepalive_and_user_timeout() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let c = TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let (s, _) = l.accept().unwrap();
        for st in [&c, &s] {
            assert_eq!(getsockopt_i32(st, libc::SOL_SOCKET, libc::SO_KEEPALIVE), 0, "default: keepalive off");
            harden_control_stream(st).unwrap();
            assert_ne!(getsockopt_i32(st, libc::SOL_SOCKET, libc::SO_KEEPALIVE), 0);
            assert_eq!(getsockopt_i32(st, libc::IPPROTO_TCP, libc::TCP_KEEPIDLE), CTL_KEEPIDLE_S);
            assert_eq!(getsockopt_i32(st, libc::IPPROTO_TCP, libc::TCP_KEEPINTVL), CTL_KEEPINTVL_S);
            assert_eq!(getsockopt_i32(st, libc::IPPROTO_TCP, libc::TCP_KEEPCNT), CTL_KEEPCNT);
            assert_eq!(getsockopt_i32(st, libc::IPPROTO_TCP, libc::TCP_USER_TIMEOUT), CTL_USER_TIMEOUT_MS);
        }
    }

    /// Audit C4: a silent stray connection (health check, telnet) and a close-without-data connection are dropped
    /// and the node keeps listening; the real head's connection is then accepted, hardened, with reads blocking again.
    #[test]
    fn accept_head_drops_silent_strays_and_serves_the_real_head() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let t0 = std::time::Instant::now();
        let stray_silent = TcpStream::connect(addr).unwrap(); // held open, sends nothing
        let stray_closed = TcpStream::connect(addr).unwrap();
        drop(stray_closed);
        let partial = {
            let mut p = TcpStream::connect(addr).unwrap();
            p.write_all(&[0, 0]).unwrap(); // 2 of the 4 length bytes, then silence
            p
        };
        let head = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            let mut h = TcpStream::connect(addr).unwrap();
            send_msg(&mut h, &Msg::Hello { version: "1".into(), role: "head".into(), hostname: "t".into() }).unwrap();
            h
        });
        let (mut s, _) = accept_head(&l, Duration::from_millis(300)).unwrap();
        assert!(t0.elapsed() >= Duration::from_millis(600), "the two silent strays cost one first-frame wait each: {:?}", t0.elapsed());
        assert!(t0.elapsed() < Duration::from_secs(5));
        assert_ne!(getsockopt_i32(&s, libc::SOL_SOCKET, libc::SO_KEEPALIVE), 0, "the accepted head stream is hardened");
        assert_eq!(s.read_timeout().unwrap(), None, "reads block again after the first-frame check");
        match recv_msg(&mut s).unwrap() {
            Msg::Hello { role, .. } => assert_eq!(role, "head", "the first frame on the accepted stream is the real head's Hello (peek consumed nothing)"),
            other => panic!("expected the head's Hello, got {other:?}"),
        }
        let _keep = (stray_silent, partial, head.join().unwrap());
    }

    #[test]
    fn node_watch_disarm_restores_blocking_reads() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let _peer = TcpStream::connect(addr).unwrap();
        let (s, _) = l.accept().unwrap();
        let streams = vec![s];
        let w = watch_nodes_during_bring_up(&streams).unwrap();
        std::thread::sleep(Duration::from_millis(250));
        w.disarm(&streams);
        assert_eq!(streams[0].read_timeout().unwrap(), None);
    }

    fn space_report(need: u64, free: u64) -> Option<SpaceReport> {
        Some(SpaceReport { need, free, dir: "/c".into(), host: "box".into() })
    }

    /// Owner 2026-11-02, the rule itself: a node is short when its drive cannot hold what it would store plus
    /// `SHIP_HEADROOM`; a node with nothing to store or no report is not judged; the line says how much is needed.
    #[test]
    fn a_node_is_short_when_its_drive_cannot_hold_the_shipment_plus_headroom() {
        let rows = vec![
            (1, "10.0.0.1".to_string(), space_report(40_000_000_000, 40_000_000_000 + SHIP_HEADROOM)), // exactly fits
            (2, "10.0.0.2".to_string(), space_report(40_000_000_000, 8_000_000_000)),                  // short
            (3, "10.0.0.3".to_string(), space_report(0, 0)),                                            // stores nothing
            (4, "10.0.0.4".to_string(), None),                                                          // no report
            (5, "10.0.0.5".to_string(), space_report(40_000_000_000, 40_000_000_000)),                 // the headroom is missing
        ];
        let short = space_shortfalls(&rows);
        assert_eq!(short.len(), 2, "{short:?}");
        assert_eq!(short[0], "rank 2, node box (10.0.0.2): needs ~41.0 GB (40.0 GB of model files it does not have yet + \
1.0 GB headroom), has 8.0 GB free on the drive of /c — short by ~33.0 GB");
        assert!(short[1].starts_with("rank 5, node box (10.0.0.5): needs ~41.0 GB") && short[1].ends_with("short by ~1.0 GB"), "{}", short[1]);
        let msg = space_refusal(&short, 5);
        assert!(msg.starts_with("not enough drive space on 2 of 5 TP node(s)") && msg.contains("Nothing was shipped")
                && msg.contains("--cached-models-remove") && msg.contains("--tp-cache"), "{msg}");
    }

    fn blob_artifact(dir: &Path, logical: &str, bytes: &[u8]) -> Artifact {
        std::fs::write(dir.join(logical), bytes).unwrap();
        let mut h = Sha256::new(); h.update(bytes);
        Artifact { logical: logical.into(), hash: hex(&h.finalize()), size: bytes.len() as u64 }
    }
    /// One real `node_handle` on loopback, returning its model dir and its rank on success.
    fn loopback_node() -> (NodeInfo, std::thread::JoinHandle<Result<(PathBuf, i32)>>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let t = std::thread::spawn(move || {
            let (s, _) = l.accept().unwrap();
            node_handle(s).map(|(dir, _, cfg, _)| (dir, cfg.node_rank))
        });
        (NodeInfo { hostname: addr.ip().to_string(), addr }, t)
    }

    /// Owner 2026-11-02, over loopback with the real node and head code: one node's drive cannot hold its share, so
    /// the head stops after the manifest round, before any byte ships. Each node reported what it would store
    /// (the drafter files announced with the manifest included, a file it gets in both counted once), each node is
    /// told why and stops, and nothing was written to the cache.
    #[test]
    fn a_short_node_drive_stops_the_head_before_anything_ships() {
        let tmp = std::env::temp_dir().join(format!("shipspace_short_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let _g = CACHE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner()); // tp-cache is process-global
        let cache = tmp.join("cache"); // not created: the free space comes from its nearest existing ancestor
        crate::opts::set(crate::opt!("tp-cache"), &cache.to_string_lossy());
        let md = tmp.join("model");
        std::fs::create_dir_all(&md).unwrap();
        let small = blob_artifact(&md, "a.bin", &[1u8; 1000]);
        let dfile = blob_artifact(&md, "d.bin", &[2u8; 5000]);
        let huge = Artifact { logical: "b.bin".into(), hash: "f".repeat(64), size: 1 << 60 }; // no drive holds 1.15 EB
        let draft = ("drafter".to_string(), vec![dfile, small.clone()]);
        let (n1, t1) = loopback_node();
        let (n2, t2) = loopback_node();
        let nodes = vec![n1, n2];
        let mut offers = vec![head_sync_one(&nodes[0], "m", &[small.clone()], Some(&draft)).unwrap(),
                              head_sync_one(&nodes[1], "m", &[small.clone(), huge], None).unwrap()];
        let r1 = offers[0].space.clone().expect("a node reports its drive");
        assert_eq!((r1.need, r1.dir.clone()), (1000 + 5000, cache.display().to_string()), "{r1:?}");
        assert!(r1.free > 0 && r1.host == hostname(), "{r1:?}");
        assert_eq!(offers[1].space.as_ref().unwrap().need, 1000 + (1 << 60));
        let e = format!("{:#}", space_check(&nodes, &mut offers).expect_err("the head must stop"));
        eprintln!("head error: {e}"); // the exact text the head prints (receipt with --nocapture)
        assert!(e.starts_with("not enough drive space on 1 of 2 TP node(s)") && e.contains("Nothing was shipped")
                && e.contains(&format!("rank 2, node {} (127.0.0.1): needs ~", hostname())) && !e.contains("rank 1,")
                && e.contains("short by ~"), "{e}");
        drop(offers);
        for t in [t1, t2] {
            let ne = format!("{:#}", t.join().unwrap().expect_err("each node is told and stops"));
            assert!(ne.starts_with("the head stopped the sync: not enough drive space"), "{ne}");
        }
        assert!(!cache.join("blobs").exists(), "nothing was stored on any node");
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// The two-phase sync still syncs: a node with room passes the check and gets its files, config and rank.
    #[test]
    fn a_node_with_room_is_synced_after_the_check() {
        let tmp = std::env::temp_dir().join(format!("shipspace_room_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let _g = CACHE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::opts::set(crate::opt!("tp-cache"), &tmp.join("cache").to_string_lossy());
        let md = tmp.join("model");
        std::fs::create_dir_all(&md).unwrap();
        let small = blob_artifact(&md, "a.bin", &[3u8; 1000]);
        let (n1, t1) = loopback_node();
        let nodes = vec![n1];
        let mut offers = vec![head_sync_one(&nodes[0], "m", &[small.clone()], None).unwrap()];
        assert_eq!(offers[0].space.as_ref().unwrap().need, 1000);
        space_check(&nodes, &mut offers).expect("room for 1 KB");
        let topo = vec!["127.0.0.1".to_string(); 2];
        let cfg = crate::tp::TpConfig::from_opts();
        let _s = head_ship(&nodes[0], 1, offers.pop().unwrap(), &md, &[small.clone()], &topo, &cfg, None, None).unwrap();
        let (dir, rank) = t1.join().unwrap().expect("the node synced");
        assert_eq!(rank, 1);
        assert_eq!(std::fs::read(dir.join("a.bin")).unwrap(), vec![3u8; 1000]);
        std::fs::remove_dir_all(&tmp).ok();
    }
}

/// S-B9-CF-AGREE16B (W1, W2): what the head/node build guard covers, and that both ends apply it.
#[cfg(test)]
mod build_id_tests {
    use super::*;
    use std::collections::BTreeSet;

    /// `(including file, include target)` pairs that are test-only: the include sits inside `#[cfg(test)]`,
    /// the target is a lab fixture under `tests/` (absent from a source tree shipped without `tests/`),
    /// and it is deliberately in NO stamp (see build.rs). Adding a pair here is a review decision.
    const CFG_TEST_ONLY_INCLUDES: &[(&str, &str)] =
        &[("src/loop_detect.rs", "tests/fixtures/loop_detect_vectors.txt")];

    fn root() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")) }
    fn hashed() -> BTreeSet<&'static str> { env!("SOURCE_BUILD_FILES").split(';').collect() }
    fn kernel() -> BTreeSet<&'static str> { env!("KERNEL_BUILD_FILES").split(';').collect() }

    /// Every `.rs` under `src/`, at any depth, as a path relative to the crate root. The walk is this
    /// test's own (build.rs has another): `src/bin` (separate executables) and `src/ptx` (build output)
    /// are the only directories left out, and they are named here, so a new one is walked.
    fn engine_sources() -> Vec<String> {
        let (root, mut out, mut stack) = (root(), Vec::new(), vec![root().join("src")]);
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap().flatten() {
                let p = e.path();
                let rel = p.strip_prefix(&root).unwrap().to_string_lossy().to_string();
                if p.is_dir() {
                    if rel != "src/bin" && rel != "src/ptx" { stack.push(p); }
                } else if p.is_file() && p.extension().map_or(false, |x| x == "rs") {
                    out.push(rel);
                }
            }
        }
        out.sort();
        out
    }

    /// The W1 hole: `SOURCE_BUILD_ID` read the `.rs` files of `src/` only, so a head and a node that
    /// differed only in `src/exl3_forward/xtp.rs` (or any of 19 more files) passed the handshake.
    #[test]
    fn the_source_stamp_covers_every_engine_source_file_at_any_depth() {
        let (hashed, on_disk) = (hashed(), engine_sources());
        let missing: Vec<&String> = on_disk.iter().filter(|f| !hashed.contains(f.as_str())).collect();
        assert!(missing.is_empty(), "engine sources OUTSIDE the head/node build stamp (a pair that differs only there would \
                 not be refused). If the file is new and build.rs has not rerun since, touch build.rs; otherwise fix the walk \
                 in build.rs:\n{missing:#?}");
        // the files the old stamp missed, by name, and the count: 20 in subdirectories at the time of the fix
        for f in ["src/exl3_forward/xtp.rs", "src/exl3_forward/xtp/ep_deal4.rs", "src/exl3_forward/dense.rs",
                  "src/exl3_forward/dhead.rs", "src/exl3_forward/w4s.rs", "src/exl3_forward/wp16.rs",
                  "src/dflash2/round.rs", "src/dflash2/mod.rs", "src/dspark/round.rs", "src/dspark/mod.rs",
                  "src/cluster.rs", "src/net.rs", "src/main.rs", "src/opts_table.rs"] {
            assert!(hashed.contains(f), "{f} is not in the build stamp");
        }
        let nested = on_disk.iter().filter(|f| f.matches('/').count() >= 2).count();
        assert!(nested >= 20, "the walk found only {nested} engine sources below src/<dir>/ (20 when this test was written): \
                 is the test still walking subdirectories?");
        // not the engine: separate executables and the build's own output
        assert!(!hashed.iter().any(|f| f.starts_with("src/bin/") || f.starts_with("src/ptx/")), "{hashed:?}");
        // every listed file exists (a stale list would mean the stamp was computed from another tree)
        for f in &hashed { assert!(root().join(f).is_file(), "{f} is in the build stamp but not on disk"); }
    }

    /// The other inputs of the binary: the manifests, the build script, the C shim, and every file an
    /// engine source pulls in with `include_str!` / `include_bytes!` / `include!`. Each include target
    /// must be in the source stamp or the kernel stamp (the two halves of `binary_version`).
    #[test]
    fn the_stamps_cover_the_manifests_and_every_included_file() {
        let (hashed, kernel, root) = (hashed(), kernel(), root());
        for f in ["Cargo.lock", "Cargo.toml", "build.rs", "native/net_shim.c", "data/ep_deal/qwen38_flash_next_tp2.json",
                  "data/ep_deal/qwen38_flash_next_tp4.json", "src/tq_consts/Pi.bin", "src/tq_consts/Pit.bin",
                  "src/tq_consts/S.bin", "src/tq_consts/codebooks.bin", "src/tq_consts/scale.bin"] {
            assert!(hashed.contains(f), "{f} is not in the source stamp");
        }
        assert!(kernel.contains("native/tp_doorbell.h") && kernel.contains("kernels/gpu_batch.cu")
                && kernel.contains("kernels/exl3_bench.cu"), "{kernel:?}");
        let (mut seen, mut bad) = (0usize, Vec::new());
        for src in engine_sources() {
            let text = std::fs::read_to_string(root.join(&src)).unwrap();
            // (this module spells the three macro names as needles: scan this file up to it)
            let text = if src == "src/cluster.rs" { &text[..text.find("mod build_id_tests").unwrap()] } else { &text[..] };
            let dir = Path::new(&src).parent().unwrap().to_path_buf();
            for mac in ["include_str!(", "include_bytes!(", "include!("] {
                let mut from = 0;
                while let Some(k) = text[from..].find(mac) {
                    let at = from + k + mac.len();
                    from = at;
                    // a literal path is the only spelling in the tree; anything else (concat!, a macro
                    // variable) is reported, because this test could not say what it pulls in
                    let rest = text[at..].trim_start();
                    let Some(lit) = rest.strip_prefix('"').and_then(|r| r.split('"').next()) else {
                        bad.push(format!("{src}: a non-literal {mac}..) the test cannot resolve: {:?}", rest.chars().take(40).collect::<String>()));
                        continue;
                    };
                    // normalize dir/lit (".." pops) into a path relative to the crate root
                    let mut parts: Vec<String> = dir.iter().map(|c| c.to_string_lossy().to_string()).collect();
                    for c in lit.split('/') {
                        match c { ".." => { parts.pop(); } "." | "" => {} c => parts.push(c.to_string()) }
                    }
                    let target = parts.join("/");
                    seen += 1;
                    // The one stated exception: a TEST fixture included from inside `#[cfg(test)]` is not an
                    // input of the engine binary, so it is in no stamp (a stamp that hashed it "when present"
                    // gave one tree two tokens). The exception is narrow: this exact (file, target) pair, and
                    // only an include that sits after a `#[cfg(test)]` line of that file.
                    if CFG_TEST_ONLY_INCLUDES.contains(&(src.as_str(), target.as_str()))
                        && text[..at].contains("#[cfg(test)]") {
                        continue;
                    }
                    // EMB1: build.rs-EMITTED PTX embeds are derived artifacts. Their bytes are a
                    // pure function of inputs that ARE stamped: the paired kernels/*.cu in the
                    // kernel stamp, and build.rs (owner of the nvcc flags and of this manifest)
                    // in the source stamp. cargo dep-info pins the crate rebuild on the emitted
                    // bytes; the boot-time KERNEL_BUILD_ID assert pins module-binary pairing.
                    // The manifest comes from build.rs itself: the exemption cannot grow.
                    if let Some((cu, _)) = env!("EMBEDDED_PTX_MANIFEST")
                        .split(';').filter_map(|p| p.split_once(':'))
                        .find(|&(_, ptx)| format!("src/ptx/{ptx}") == target)
                    {
                        assert!(kernel.contains(format!("kernels/{cu}").as_str()),
                                "{target} is emitted from kernels/{cu}, not in the kernel stamp");
                        continue;
                    }
                    if !hashed.contains(target.as_str()) && !kernel.contains(target.as_str()) {
                        bad.push(format!("{src} includes {target}: in neither the source stamp nor the kernel stamp (add it in build.rs)"));
                    }
                }
            }
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
        assert!(seen >= 30, "the scan resolved only {seen} include targets (40 when written): did it stop matching?");
    }

    /// W2: the lockstep words of AGREE16 are a protocol change, named as one.
    #[test]
    fn the_version_token_names_protocol_kernels_and_sources() {
        assert_eq!(PROTOCOL_VERSION, 6, "AGREE16 changed every pairwise lockstep word: protocol 6");
        let v = binary_version();
        assert_eq!(v, format!("v6-k{}-r{}", env!("KERNEL_BUILD_ID"), env!("SOURCE_BUILD_ID")));
        assert!(env!("SOURCE_BUILD_ID").len() == 16 && env!("KERNEL_BUILD_ID").len() == 16);
        assert_eq!(version_match(&v, &v), Ok(()));
        // a difference in any one part refuses, and the message names both tokens
        for other in [v.replacen("v6-", "v5-", 1), format!("{v}0"), v.replacen("-r", "-r0", 1), String::new()] {
            let e = version_match(&v, &other).unwrap_err();
            assert_eq!(e, format!("binary mismatch: head {v} vs node {other}"));
            assert!(version_match(&other, &v).is_err(), "the comparison is symmetric");
        }
    }

    /// FLASHNEXT-DF2-C (ISSUES 116): the drafter decision is ONE helper for both head entries, and an
    /// EXL3 serve config with a DF2 source + a resolved draft dir ships the artifact (config.json
    /// included — the node's geometry install reads it); a non-DF2 source or an empty dir ships nothing.
    #[test]
    fn exl3_serve_config_ships_the_dflash2_drafter_like_the_nvfp4_one() {
        let d = std::env::temp_dir().join(format!("df2c_manifest_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("config.json"), "{\"hidden_size\": 2560}").unwrap();
        std::fs::write(d.join("model.safetensors"), vec![7u8; 4096]).unwrap();
        let mut cfg = crate::tp::TpConfig::from_opts();
        cfg.mode_serve = true; cfg.exl3 = true; cfg.world = 2;
        cfg.spec_source = "dflash2".into();
        cfg.df2_draft_dir = d.to_string_lossy().to_string();
        let (id, arts) = draft_for_cfg(&cfg).expect("an EXL3 serve with a DF2 source ships its drafter");
        assert_eq!(id, d.file_name().unwrap().to_string_lossy());
        let mut names: Vec<&str> = arts.iter().map(|a| a.logical.as_str()).collect();
        names.sort();
        assert_eq!(names, ["config.json", "model.safetensors"]);
        cfg.spec_source = "mtp".into();
        assert!(draft_for_cfg(&cfg).is_none(), "a non-DF2 source ships no drafter");
        cfg.spec_source = "dflash2-auto".into(); cfg.df2_draft_dir.clear();
        assert!(draft_for_cfg(&cfg).is_none(), "no resolved draft dir: nothing to ship");
        // both head entries hand the SAME decision to head_sync_one (the retain entry passed None)
        let src = include_str!("cluster.rs");
        let retain = &src[src.find("pub fn run_head_retain(").unwrap()..src.find("fn sync_only_exit(").unwrap()];
        assert!(retain.contains("let draft = draft_for_cfg(cfg);") && retain.contains("cfg, draft.as_ref(), srcs)")
                && retain.contains("head_sync_one(node, mid, &per_rank[rank], draft.as_ref())"),
                "run_head_retain must ship draft_for_cfg's decision");
        assert!(!retain.contains("cfg, None, srcs)"), "the EXL3 head may no longer pass None as the drafter");
        std::fs::remove_dir_all(&d).ok();
    }
    /// The node's half, the real `node_handle` over loopback: a head with another version token is told
    /// `binary mismatch` and nothing follows (no manifest is read).
    #[test]
    fn a_node_refuses_a_head_built_from_other_sources() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let node = std::thread::spawn(move || { let (s, _) = l.accept().unwrap(); node_handle(s).map(|_| ()) });
        let mut h = TcpStream::connect(addr).unwrap();
        let theirs = binary_version().replacen("-r", "-rdead", 1);
        send_msg(&mut h, &Msg::Hello { version: theirs.clone(), role: "head".into(), hostname: "t".into() }).unwrap();
        match recv_msg(&mut h).unwrap() {
            Msg::Error { msg } => assert_eq!(msg, format!("binary mismatch: head {theirs} vs node {}", binary_version())),
            other => panic!("the node must refuse, got {other:?}"),
        }
        let e = node.join().unwrap().expect_err("node_handle must fail");
        assert!(format!("{e:#}").contains("binary mismatch"), "{e:#}");
    }

    /// The head's half, the real `head_hello` over loopback. Before AGREE16B the head accepted any Hello:
    /// a node that does not compare (here: a fake one that answers Hello whatever it was sent) got the
    /// manifest and the model. Now the head compares the node's token itself.
    #[test]
    fn a_head_refuses_a_node_built_from_other_sources_even_if_the_node_does_not_check() {
        let run = |node_version: String| -> Result<()> {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = l.local_addr().unwrap();
            let fake = std::thread::spawn(move || {
                let (mut s, _) = l.accept().unwrap();
                let got = recv_msg(&mut s).unwrap();
                send_msg(&mut s, &Msg::Hello { version: node_version, role: "node".into(), hostname: "fake".into() }).unwrap();
                // what the head sends next, if anything (None = it hung up: nothing was shipped)
                (got, recv_msg(&mut s).ok())
            });
            let mut s = TcpStream::connect(addr).unwrap();
            let ni = NodeInfo { hostname: "fake".into(), addr };
            let r = head_hello(&mut s, &ni, &binary_version());
            if r.is_ok() { send_msg(&mut s, &Msg::Manifest { model_id: "m".into(), artifacts: Vec::new(), draft: Vec::new() }).unwrap(); }
            drop(s);
            let (first, next) = fake.join().unwrap();
            assert!(matches!(first, Msg::Hello { ref role, .. } if role == "head"));
            assert_eq!(next.is_some(), r.is_ok(), "a refused node is sent nothing after the Hello");
            r
        };
        assert!(run(binary_version()).is_ok(), "the same build pairs");
        let other = binary_version().replacen("-r", "-rdead", 1);
        let e = format!("{:#}", run(other.clone()).expect_err("the head must refuse a node with another version token"));
        assert!(e.contains(&format!("binary mismatch: head {} vs node {other}", binary_version())), "{e}");
        let old = binary_version().replacen("v6-", "v5-", 1);
        assert!(run(old).is_err(), "a v5 node is refused by protocol number");
    }

    /// The two call sites, pinned: `head_sync_one` goes through `head_hello` before it ships the manifest,
    /// and discovery still filters on the same token.
    #[test]
    fn the_head_checks_the_version_before_the_manifest() {
        let src = include_str!("cluster.rs");
        let body = &src[src.find("fn head_sync_one(").unwrap()..];
        let body = &body[..body.find("\nfn ").unwrap()];
        let (hello, manifest) = (body.find("head_hello(&mut s, node, &binary_version())?;").expect("head_sync_one calls head_hello"),
                                 body.find("Msg::Manifest").expect("head_sync_one ships the manifest"));
        assert!(hello < manifest, "the version check comes before the manifest");
        assert!(src.contains("if r.version != binary_version() {"), "discovery filters on the version token");
    }

    /// EXL3-SHARDSHIP — the ISSUES row's contract, exercised end to end on the REAL code path:
    /// a minimal qwen4_exp EXL3 pack is planned by the real `ship_plans` (B32), rank 1's blobs
    /// are materialized exactly as the head's shipper produces the wire bytes (Whole = the
    /// file's bytes, Segment = `write_segment`, Bytes = inline), published and assembled by the
    /// real blob-cache path (`publish_blob` + `assemble_model_dir`), and the REAL node loader
    /// (`Exl3Pack::open`) must open the assembled rank dir: the rank's own expert modules +
    /// every common module and every sidecar the serve boot reads, no foreign expert, the
    /// regenerated index and gb10_shard.json in place — and strictly less than the whole pack.
    /// The wiring pins close the row's two ends: the head ships whenever the plan covers the
    /// model and stamps `ship_shards`; the node's EXL3 boot swaps the head's local pack dir for
    /// the shipped one (never the head's path).
    #[test]
    fn exl3_rank_ship_round_trip_through_the_blob_cache_and_the_node_loader() {
        use std::collections::BTreeMap;
        let deal = crate::exl3_forward::xtp::ep_deal_kind();
        assert_eq!(deal, "interleave", "the fixture pins the default expert deal");

        // ---- the fixture pack: 2 trunk layers x 4 experts (gate/up/down), a common attn
        // module per layer, lm_head, and the sidecars the node's serve boot reads by name.
        let root = std::env::temp_dir().join(format!("shardship_exl3_{}", std::process::id()));
        let pack = root.join("pack");
        std::fs::create_dir_all(&pack).unwrap();
        let (nl, ne) = (2usize, 4usize);
        let mut names: Vec<(String, Vec<usize>)> = Vec::new();
        let mut module = |v: &mut Vec<(String, Vec<usize>)>, name: String| {
            v.push((format!("{name}.trellis"), vec![1, 1, 32])); // words 32 -> k2 4 (kernel-supported)
            for s in ["suh", "svh", "mul1"] { v.push((format!("{name}.{s}"), vec![1])); }
        };
        for l in 0..nl {
            module(&mut names, format!("model.language_model.layers.{l}.self_attn.q_proj"));
        }
        module(&mut names, "lm_head".to_string());
        for l in 0..nl { for e in 0..ne { for proj in ["gate_proj", "up_proj", "down_proj"] {
            module(&mut names, format!("model.language_model.layers.{l}.mlp.experts.{e}.{proj}"));
        } } }
        let mut header: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        let mut data: Vec<u8> = Vec::new();
        for (name, shape) in &names {
            let bytes = shape.iter().product::<usize>() * 4; // F32
            let start = data.len();
            data.extend(std::iter::repeat(0u8).take(bytes));
            header.insert(name.clone(), serde_json::json!({
                "dtype": "F32", "shape": shape, "data_offsets": [start, start + bytes]
            }));
        }
        let hj = serde_json::to_string(&header).unwrap();
        let mut st = (hj.len() as u64).to_le_bytes().to_vec();
        st.extend_from_slice(hj.as_bytes());
        st.extend_from_slice(&data);
        let st_name = "model-00001-of-00001.safetensors";
        std::fs::write(pack.join(st_name), &st).unwrap();
        let mut wmap = serde_json::Map::new();
        for (name, _) in &names { wmap.insert(name.clone(), serde_json::json!(st_name)); }
        std::fs::write(pack.join("model.safetensors.index.json"),
                       serde_json::json!({ "weight_map": wmap }).to_string()).unwrap();
        std::fs::write(pack.join("config.json"), serde_json::json!({
            "model_type": "qwen4_exp",
            "text_config": { "hidden_size": 256, "num_hidden_layers": nl, "num_experts": ne,
                             "num_experts_per_tok": 2, "moe_intermediate_size": 128 }
        }).to_string()).unwrap();
        std::fs::write(pack.join("quantization_config.json"),
                       serde_json::json!({ "quant_method": "exl3", "codebook": "mul1", "bits": 2.0 }).to_string()).unwrap();
        std::fs::write(pack.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(pack.join("generation_config.json"), "{}").unwrap();

        // ---- the row's front door: the plan covers this pack at world 2, never at world 1
        assert_eq!(crate::shard_plan::detect_format(&pack), Some(crate::shard_plan::ShardFormat::Exl3));
        assert_eq!(crate::shard_plan::applies(&pack, 2), Some(crate::shard_plan::ShardFormat::Exl3));
        assert_eq!(crate::shard_plan::applies(&pack, 1), None, "world < 2 never ships shards");

        // ---- the head: the real B32 planning + hashing
        let ships = ship_plans(&pack, 2).expect("the covered pack must plan");
        assert_eq!(ships.len(), 2, "ships[0] is the head (nothing shipped), ships[1] is rank 1");
        assert!(ships[0].artifacts.is_empty() && ships[0].srcs.is_empty());
        let ship = &ships[1];
        let src_total = std::fs::metadata(pack.join(st_name)).unwrap().len()
            + ["config.json", "quantization_config.json", "tokenizer.json", "generation_config.json"]
                .iter().map(|f| std::fs::metadata(pack.join(f)).unwrap().len()).sum::<u64>();
        let ship_total: u64 = ship.artifacts.iter().map(|a| a.size).sum();
        assert!(ship_total < src_total,
                "rank 1 must receive only its shards, not the whole pack ({ship_total} vs {src_total} B)");

        // ---- the wire: every artifact's bytes exactly as the shipper produces them, published
        // into the real blob cache and assembled by the real node-side assembly
        let mut blobs = Vec::new();
        std::fs::create_dir_all(cache_root().join("blobs")).unwrap(); // first publish on a fresh box creates it
        for a in &ship.artifacts {
            let src = ship.srcs.get(&a.logical).expect("every artifact has its plan source");
            let bytes: Vec<u8> = match src {
                crate::shard_plan::PlanSrc::Whole(p) => std::fs::read(p).unwrap(),
                crate::shard_plan::PlanSrc::Bytes(b) => b.clone(),
                crate::shard_plan::PlanSrc::Segment(ps) => {
                    let mut v = Vec::new();
                    crate::shard_plan::write_segment(ps, &mut v).unwrap();
                    v
                }
            };
            assert_eq!(bytes.len() as u64, a.size, "{}: wire bytes must match the manifest size", a.logical);
            let tmp = cache_root().join("blobs").join(format!("tmp.shardship.{}", a.hash));
            std::fs::write(&tmp, &bytes).unwrap();
            publish_blob(&a.hash, &tmp).unwrap();
            blobs.push(blob_path(&a.hash));
        }
        let dir = assemble_model_dir(&ship.model_id, &ship.artifacts).unwrap();

        // ---- the node: the REAL loader opens the assembled rank dir
        let p = crate::exl3::Exl3Pack::open(&dir.to_string_lossy())
            .expect("the assembled rank shard dir must open as an EXL3 pack");
        assert_eq!(p.cfg.num_experts, ne, "the shipped config.json parses");
        let mut expert_modules = 0usize;
        for m in &p.modules {
            if let Some(rest) = m.name.strip_prefix("model.language_model.layers.") {
                if let Some((l, tail)) = rest.split_once('.') {
                    if let Some(experts) = tail.strip_prefix("mlp.experts.") {
                        let (e, _) = experts.split_once('.').expect("expert tensor name");
                        let own = crate::exl3_forward::xtp::ep_layer_owners_kind(&deal, ne, 2, l.parse().unwrap()).unwrap();
                        assert_eq!(own[e.parse::<usize>().unwrap()], 1,
                                   "rank 1's dir must hold no foreign expert: {}", m.name);
                        expert_modules += 1;
                    }
                }
            }
        }
        assert_eq!(expert_modules, nl * (ne / 2) * 3, "own gate/up/down modules only, every layer");
        for f in ["config.json", "quantization_config.json", "tokenizer.json",
                  "generation_config.json", "model.safetensors.index.json", "gb10_shard.json"] {
            assert!(dir.join(f).is_file(), "{f} must ship to the rank dir");
        }
        let info: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("gb10_shard.json")).unwrap()).unwrap();
        assert_eq!(info["rank"].as_u64(), Some(1), "the shard info names this rank");
        assert!(dir.join("config.json").symlink_metadata().unwrap().file_type().is_symlink(),
                "the assembled files are blob-cache links: the node never needed the head's path");

        // ---- the row's two wiring ends, pinned against regression
        let src = include_str!("cluster.rs");
        let retain = &src[src.find("pub fn run_head_retain(").unwrap()..src.find("fn sync_only_exit(").unwrap()];
        assert!(retain.contains("applies(model_dir, world"),
                "the head must ship whenever the shard plan covers the model");
        let one = &src[src.find("fn head_sync_one(").unwrap()..];
        assert!(one.contains("node_cfg.ship_shards = srcs.is_some()"),
                "the head must stamp ship_shards when the rank's shards shipped");
        let msrc = include_str!("main.rs");
        assert!(msrc.contains("if tpc.ship_shards {") &&
                msrc.contains("tpc.exl3_pack_dir = dir.to_string_lossy().to_string();"),
                "the node's EXL3 boot must load the shipped shard dir, not the head's local path");

        std::fs::remove_dir_all(&dir).ok();
        for b in blobs { std::fs::remove_file(b).ok(); }
        std::fs::remove_dir_all(&root).ok();
    }
}

#[cfg(test)]
mod tpship_tests {
    /// TPSHIP (2026-11-07): an EXL3 pack the shard plan does not cover ships whole (no empty-manifest branch is left
    /// in the EXL3 head sync) and the node is told to load that copy (ship_shards), never the head's path.
    #[test]
    fn exl3_packs_outside_the_shard_plan_ship_whole_and_load_the_shipped_copy() {
        let src = include_str!("cluster.rs");
        let retain = &src[src.find("pub fn run_head_retain(").unwrap()..];
        let retain = &retain[..retain.find("\n}\n").unwrap()];
        assert!(!retain.contains("vec![Vec::new(); world as usize]"), "the EXL3 empty-manifest (load at the head's path) branch is gone");
        assert!(retain.contains("build_manifest(model_dir, world)?"), "a pack outside the plan ships whole");
        // ship-space (2026-11-02) split the per-node sync: the shipment and its ship_shards stamp live in head_ship
        let sync = &src[src.find("fn head_ship(").unwrap()..];
        let sync = &sync[..sync.find("\nfn ").unwrap()];
        assert!(sync.contains("node_cfg.ship_shards = srcs.is_some() || (cfg.exl3 && !artifacts.is_empty());"),
                "a node that received an EXL3 pack loads the shipped copy");
    }
}
