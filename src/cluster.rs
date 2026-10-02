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

const PROTOCOL_VERSION: u32 = 5;   // v5: + DraftManifest/DraftReady — the DFlash2 drafter rides the sync
                                     //     into the node's blob cache (v4: per-epoch payload lengths)
const DISCOVERY_PORT: u16 = 29499;          // UDP; TCP control plane defaults to 29500 (--port)
const DISCOVERY_MAGIC: &str = "GB10-TP-DISCOVER";
/// Binary-compat token: same compiled kernels + same Rust-side sources + protocol => same wire
/// behavior. Cheaper than hashing the 15 MB executable each launch, and it is exactly what must
/// match across boxes. `-k` covers the kernels (KERNEL_BUILD_ID), `-r` the Rust sources + C shim
/// (SOURCE_BUILD_ID): the sharders/protocol/scheduler change behavior without touching a .cu.
fn binary_version() -> String {
    format!("v{}-k{}-r{}", PROTOCOL_VERSION, env!("KERNEL_BUILD_ID"), env!("SOURCE_BUILD_ID"))
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
    Manifest { model_id: String, artifacts: Vec<Artifact> },
    Missing { hashes: Vec<String> },
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

fn sha256_file(path: &Path) -> Result<(String, u64)> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        h.update(&buf[..n]);
        total += n as u64;
    }
    Ok((hex(&h.finalize()), total))
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
    for path in model_files(model_dir)? {
        let rel = path.strip_prefix(model_dir).unwrap_or(&path).to_string_lossy().to_string();
        let key = cached_key(&path)?;
        let (hash, size) = if let Some(h) = cache.get(&key) {
            (h.clone(), std::fs::metadata(&path)?.len())
        } else {
            let (h, sz) = sha256_file(&path)?;
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
struct HashWriter { h: Sha256, n: u64 }
impl Write for HashWriter {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> { self.h.update(b); self.n += b.len() as u64; Ok(b.len()) }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

fn hash_plan_src(src: &crate::shard_plan::PlanSrc) -> Result<String> {
    use crate::shard_plan::PlanSrc;
    Ok(match src {
        PlanSrc::Whole(p) => sha256_file(p)?.0,
        PlanSrc::Segment(ps) => {
            let mut w = HashWriter { h: Sha256::new(), n: 0 };
            crate::shard_plan::write_segment(ps, &mut w)?;
            hex(&w.h.finalize())
        }
        PlanSrc::Bytes(b) => { let mut h = Sha256::new(); h.update(b); hex(&h.finalize()) }
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
    let deal = match fmt { ShardFormat::Exl3 => crate::exl3_forward::xtp::ep_deal_kind(), ShardFormat::Nvfp4 => "contig".to_string() };
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
        eprintln!("[head] B32: hashing {} new shard file(s), {:.2} GB (first ship only; cached afterwards) ...",
                  todo.len(), hashed as f64 / 1e9);
        let t0 = std::time::Instant::now();
        todo.sort_by(|a, b| b.2.cmp(&a.2));
        use std::sync::atomic::{AtomicUsize, Ordering};
        let next = AtomicUsize::new(0);
        let out: std::sync::Mutex<Vec<(usize, Result<String>)>> = std::sync::Mutex::new(Vec::new());
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
        std::thread::scope(|sc| {
            for _ in 0..threads.min(todo.len()) {
                sc.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= todo.len() { break; }
                    let r = hash_plan_src(&todo[i].1);
                    out.lock().unwrap().push((i, r));
                });
            }
        });
        for (i, r) in out.into_inner().unwrap() {
            cache.insert(todo[i].0.clone(), r?);
        }
        hash_cache_save(&cache);
        eprintln!("[head] B32: hashed in {:.1} s", t0.elapsed().as_secs_f64());
    }
    let mut ships = vec![RankShip { model_id: base.clone(), artifacts: Vec::new(), srcs: HashMap::new() }];
    for p in plans {
        let mut arts = Vec::new();
        let mut srcs = HashMap::new();
        for f in p.files {
            let size = RankPlan::file_size(&f)?;
            let hash = match plan_src_key(&f.src)? {
                Some(k) => cache.get(&k).cloned().context("B32: a planned file was not hashed")?,
                None => hash_plan_src(&f.src)?,
            };
            arts.push(Artifact { logical: f.logical.clone(), hash, size });
            srcs.insert(f.logical, f.src);
        }
        let total: u64 = arts.iter().map(|a| a.size).sum();
        eprintln!("[head] B32 plan rank {}/{} ({}, deal {}): {} files, {:.2} GiB",
                  p.rank, world, fmt.name(), p.deal, arts.len(), total as f64 / (1u64 << 30) as f64);
        ships.push(RankShip { model_id: format!("{base}@{}-w{world}-r{}-{}", fmt.name(), p.rank, p.deal), artifacts: arts, srcs });
    }
    Ok(ships)
}

/// Send one artifact's bytes: from its B32 plan source when there is one, else the model dir's file.
fn send_artifact(s: &mut TcpStream, model_dir: &Path, a: &Artifact,
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
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1).min(jobs.len().max(1)) {
            sc.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= order.len() { break; }
                let i = order[k];
                let r = sha256_file(&jobs[i].0).map(|(h, _)| h);
                out.lock().unwrap()[i] = Some(r);
            });
        }
    });
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
            Err(why) => eprintln!("[node] connection from {from} dropped: {why} (not a head sync; still listening)"),
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
    eprintln!("[head] watching {} node control stream(s) through the RDMA bring-up + boot (a node failure is reported here, never hung on)",
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
    for path in model_files(dir)? {
        let rel = path.strip_prefix(dir).unwrap_or(&path).to_string_lossy().to_string();
        let key = cached_key(&path)?;
        let (hash, size) = if let Some(h) = cache.get(&key) {
            (h.clone(), std::fs::metadata(&path)?.len())
        } else {
            let (h, sz) = sha256_file(&path)?;
            cache.insert(key, h.clone());
            dirty = true;
            (h, sz)
        };
        artifacts.push(Artifact { logical: rel, hash, size });
    }
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
pub fn spawn_discovery_responder(tcp_port: u16) -> Result<()> {
    let sock = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, DISCOVERY_PORT))
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
        eprintln!("  [discover] RoCE rail {} {} ({}) ip {} bcast {}", f.rail, f.ib, f.netdev, f.ip, f.bcast);
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
    // Default = the fixed GB10 rail names (identical across DGX Spark + every OEM clone: same SoC,
    // hard-wired PCIe topology, systemd predictable naming). Manual fallback for any platform that
    // breaks that: --rdma-dev=dev1[,dev2] (rail order), set via --rdma-dev.
    let devs: Vec<(String, u8)> = match crate::opts::var(crate::opt!("rdma-dev")) {
        Ok(s) if !s.trim().is_empty() =>
            s.split(',').enumerate().map(|(i, d)| (d.trim().to_string(), (i + 1) as u8)).collect(),
        _ => vec![("rocep1s0f1".into(), 1), ("roceP2p1s0f1".into(), 2)],
    };
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
        let s = "2: eth0    inet 192.168.177.13/24 brd 192.168.177.255 scope global eth0";
        let (ip, prefix, bcast) = ipv4_of_str(s).expect("parse");
        assert_eq!(ip, Ipv4Addr::new(192, 168, 177, 13));
        assert_eq!(prefix, 24);
        assert_eq!(bcast, Ipv4Addr::new(192, 168, 177, 255));
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
            if version != binary_version() {
                send_msg(&mut s, &Msg::Error { msg: format!("binary mismatch: head {version} vs node {}", binary_version()) })?;
                bail!("head/node binary mismatch");
            }
            eprintln!("  [node] head '{hostname}' connected (binary {version})");
        }
        _ => bail!("expected Hello"),
    }
    send_msg(&mut s, &Msg::Hello { version: binary_version(), role: "node".into(), hostname: hostname() })?;

    let (model_id, artifacts) = match recv_msg(&mut s)? {
        Msg::Manifest { model_id, artifacts } => (model_id, artifacts),
        _ => bail!("expected Manifest"),
    };
    let missing: Vec<String> = artifacts.iter().map(|a| a.hash.clone())
        .filter(|h| !have_blob(h)).collect();
    let have = artifacts.len() - missing.len();
    eprintln!("  [node] manifest '{model_id}': {} artifacts, {have} cached, {} to fetch",
              artifacts.len(), missing.len());
    let fetch_bytes: u64 = artifacts.iter().filter(|a| missing.contains(&a.hash)).map(|a| a.size).sum();
    let t_fetch = std::time::Instant::now();
    send_msg(&mut s, &Msg::Missing { hashes: missing.clone() })?;

    for _ in 0..missing.len() {
        match recv_msg(&mut s)? {
            Msg::BlobHeader { hash, size } => {
                let tmp = recv_blob(&mut s, &hash, size)?;
                publish_blob(&hash, &tmp)?;
                eprintln!("  [node] cached {} ({:.1} MB)", &hash[..12], size as f64 / 1e6);
            }
            _ => bail!("expected BlobHeader"),
        }
    }
    let dir = assemble_model_dir(&model_id, &artifacts)?;
    send_msg(&mut s, &Msg::Ready { model_dir: dir.to_string_lossy().to_string() })?;
    let cfg = match recv_msg(&mut s)? {
        Msg::Config(c) => c,
        _ => bail!("expected Config"),
    };
    eprintln!("  [node] config from head: shard_mixers={} graph={} fp32_partials={} mtp={} depth={:?} mode_serve={}",
              cfg.shard_mixers, cfg.graph, cfg.fp32_partials, cfg.mtp, cfg.mtp_depth, cfg.mode_serve);


    // v5 — the DFlash2 draft round (always exactly one DraftManifest after Config). The bytes
    // land in the SAME content-addressed blob store as the model shards; the assembled dir lives
    // under models/<draft-id>/ exactly like a trunk model. A node therefore NEVER needs a local
    // draft copy — the caller rewrites the config's df2_draft_dir to the returned cache path.
    let draft_dir: Option<PathBuf> = match recv_msg(&mut s)? {
        Msg::DraftManifest { model_id, artifacts } if artifacts.is_empty() => {
            send_msg(&mut s, &Msg::DraftReady { model_dir: String::new() })?;
            eprintln!("  [node] no draft artifact this session (head sent an empty manifest)");
            None
        }
        Msg::DraftManifest { model_id, artifacts } => {
            let missing: Vec<String> = artifacts.iter().map(|a| a.hash.clone())
                .filter(|h| !have_blob(h)).collect();
            let have = artifacts.len() - missing.len();
            eprintln!("  [node] draft manifest '{model_id}': {} artifacts, {have} cached, {} to fetch",
                      artifacts.len(), missing.len());
            send_msg(&mut s, &Msg::Missing { hashes: missing.clone() })?;
            for _ in 0..missing.len() {
                match recv_msg(&mut s)? {
                    Msg::BlobHeader { hash, size } => {
                        let tmp = recv_blob(&mut s, &hash, size)?;
                        publish_blob(&hash, &tmp)?;
                        eprintln!("  [node] cached draft blob {} ({:.1} MB)", &hash[..12], size as f64 / 1e6);
                    }
                    _ => bail!("expected draft BlobHeader"),
                }
            }
            let ddir = assemble_model_dir(&model_id, &artifacts)?;
            send_msg(&mut s, &Msg::DraftReady { model_dir: ddir.to_string_lossy().to_string() })?;
            eprintln!("  [node] DRAFT READY — drafter assembled at {} (loads from the cache, no local copy)",
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
pub fn run_node(tcp_port: u16)
    -> Result<(PathBuf, Option<PathBuf>, IpAddr, crate::tp::TpConfig, TcpStream)> {
    spawn_discovery_responder(tcp_port)?;
    let listener = TcpListener::bind((Ipv4Addr::UNSPECIFIED, tcp_port))
        .with_context(|| format!("bind TCP {tcp_port}"))?;
    eprintln!("[node] {} ready: discovery on UDP {DISCOVERY_PORT}, control on TCP {tcp_port}, cache {}",
              hostname(), cache_root().display());
    // TP-4D (C4/C9): hardened stream + stray connections dropped (see `accept_head`)
    let (s, from) = accept_head(&listener, NODE_FIRST_FRAME_WAIT)?;
    eprintln!("[node] head connected from {from}");
    let (dir, draft_dir, cfg, s) = node_handle(s)?;
    eprintln!("[node] SYNCED — model ready at {}{}",
              dir.display(),
              draft_dir.as_ref().map(|d| format!(", drafter at {}", d.display())).unwrap_or_default());
    Ok((dir, draft_dir, from.ip(), cfg, s))
}

// ---------------------------------------------------------------------------------------------------
// Head: discover-and-push
// ---------------------------------------------------------------------------------------------------

/// Push the model to one node: Hello -> Manifest -> receive Missing -> stream blobs -> Ready, then
/// ship our TP config (`Msg::Config`) so the node runs with ZERO GB10_TP_* env vars. The shipped
/// config is a per-node clone: `node_rank` identifies this node and `topology` is the full
/// rank→RoCE-IP map (the node reads both back from its own `Msg::Config`). Returns the RETAINED
/// control stream — the TP serving session keeps it as its control plane; the bench path drops it.
fn head_sync_one(node: &NodeInfo, node_rank: i32, model_dir: &Path, model_id: &str,
                 artifacts: &[Artifact], topology: &[String],
                 cfg: &crate::tp::TpConfig,
                 draft: Option<&(String, Vec<Artifact>)>,
                 srcs: Option<&HashMap<String, crate::shard_plan::PlanSrc>>) -> Result<TcpStream> {
    let mut s = TcpStream::connect_timeout(&node.addr, Duration::from_secs(10))
        .with_context(|| format!("connect {}", node.addr))?;
    s.set_nodelay(true).ok();
    harden_or_warn(&s, "head"); // TP-4D (C3/C9): both ends of the retained stream
    send_msg(&mut s, &Msg::Hello { version: binary_version(), role: "head".into(), hostname: hostname() })?;
    // TP-4D (C4): a node whose session process is wedged (accepted the connection, never answers) must not hang
    // the head forever; the bound covers only the Hello exchange, the later replies wait for pack hashing.
    s.set_read_timeout(Some(HEAD_HELLO_WAIT)).ok();
    let hello = recv_msg(&mut s).with_context(|| format!(
        "node {} ({}) did not answer the Hello within {} s — a wedged `--node` session process there? (kill its child; the supervisor re-arms)",
        node.hostname, node.addr, HEAD_HELLO_WAIT.as_secs()))?;
    s.set_read_timeout(None).ok();
    match hello {
        Msg::Hello { .. } => {}
        Msg::Error { msg } => bail!("node rejected: {msg}"),
        _ => bail!("expected node Hello"),
    }
    send_msg(&mut s, &Msg::Manifest { model_id: model_id.into(), artifacts: artifacts.to_vec() })?;
    let missing = match recv_msg(&mut s)? {
        Msg::Missing { hashes } => hashes,
        _ => bail!("expected Missing"),
    };
    let by_hash: HashMap<&str, &Artifact> = artifacts.iter().map(|a| (a.hash.as_str(), a)).collect();
    let total: u64 = missing.iter().filter_map(|h| by_hash.get(h.as_str())).map(|a| a.size).sum();
    eprintln!("[head] {} (rank {node_rank}) needs {} / {} artifacts ({:.2} GB)", node.hostname,
              missing.len(), artifacts.len(), total as f64 / 1e9);
    let t0 = std::time::Instant::now();
    for h in &missing {
        let a = by_hash.get(h.as_str()).context("node asked for unknown hash")?;
        send_msg(&mut s, &Msg::BlobHeader { hash: a.hash.clone(), size: a.size })?;
        send_artifact(&mut s, model_dir, a, srcs)?;
    }
    match recv_msg(&mut s)? {
        Msg::Ready { model_dir } => {
            let secs = t0.elapsed().as_secs_f64();
            eprintln!("[head] {} (rank {node_rank}) READY — model at {} ({:.2} GB in {:.1}s = {:.2} GB/s)",
                      node.hostname, model_dir, total as f64/1e9, secs,
                      if secs > 0.0 { total as f64/1e9/secs } else { 0.0 });
            let mut node_cfg = cfg.clone();
            node_cfg.node_rank = node_rank;
            node_cfg.ship_shards = srcs.is_some(); // B32: this node got its rank's shard dir (load it, not the head's path)
            node_cfg.topology = topology.to_vec();
            // CLI-1: the head's resolved option registry rides the config (v23), taken at ship time
            // so every option the head resolved before its sync reaches the node.
            node_cfg.opts = crate::opts::snapshot();
            send_msg(&mut s, &Msg::Config(node_cfg.clone()))?;
            eprintln!("[head] shipped config to {} (rank {node_rank}/{})", node.hostname, node_cfg.world);

            // v5 — the DFlash2 draft round: ALWAYS exactly one DraftManifest after Config (empty
            // artifacts = none), so the node's receive state machine is unconditional and can
            // never block waiting for a round the head decided not to send.
            match draft {
                Some((draft_id, draft_arts)) if !draft_arts.is_empty() => {
                    send_msg(&mut s, &Msg::DraftManifest { model_id: draft_id.clone(),
                                                          artifacts: draft_arts.clone() })?;
                    let src_dir = Path::new(&cfg.df2_draft_dir);
                    let dmissing = match recv_msg(&mut s)? {
                        Msg::Missing { hashes } => hashes,
                        _ => bail!("expected draft Missing from {}", node.hostname),
                    };
                    let dby_hash: HashMap<&str, &Artifact> =
                        draft_arts.iter().map(|a| (a.hash.as_str(), a)).collect();
                    let dtotal: u64 = dmissing.iter().filter_map(|h| dby_hash.get(h.as_str())).map(|a| a.size).sum();
                    eprintln!("[head] {} drafter: {} / {} artifacts ({:.2} GB)",
                              node.hostname, dmissing.len(), draft_arts.len(), dtotal as f64 / 1e9);
                    let dt0 = std::time::Instant::now();
                    for h in &dmissing {
                        let a = dby_hash.get(h.as_str()).context("node asked for unknown draft hash")?;
                        send_msg(&mut s, &Msg::BlobHeader { hash: a.hash.clone(), size: a.size })?;
                        send_blob(&mut s, &src_dir.join(&a.logical))?;
                    }
                    match recv_msg(&mut s)? {
                        Msg::DraftReady { model_dir } => {
                            let secs = dt0.elapsed().as_secs_f64();
                            eprintln!("[head] {} drafter READY at {} ({:.2} GB in {:.1}s)",
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
/// them (EXL3 ships no artifacts, so `run_head_session`'s manifest/drafter path does not apply).
pub fn run_head_retain(model_dir: &Path, explicit: Option<Vec<SocketAddr>>, discover_wait: Duration,
                       cfg: &crate::tp::TpConfig)
    -> Result<(Vec<NodeInfo>, Vec<TcpStream>)>
{
    let world = cfg.world.max(1);
    eprintln!("[head] {} — building manifest for {} (world {world}) ...", hostname(), model_dir.display());
    // TP-A (EXL3, D-T0-6) legacy path: nothing ships — every node loads its OWN local pack copy at the
    // head's path (`TpConfig.exl3_pack_dir`). Nothing checks that copy against the head's: the manifest
    // check was removed (owner, 074c669, 2026-09-30). The node still runs the unchanged
    // Hello/Manifest/Ready/Config handshake, with an empty artifact list.
    // B32: each node gets ONLY its rank's shard dir through the blob cache — always, when the shard plan covers
    // the model (no switch, owner 2026-10-02).
    let ships = if crate::shard_plan::applies(model_dir, world as usize).is_some() { Some(ship_plans(model_dir, world)?) } else { None };
    let (model_id, per_rank) = if let Some(sh) = &ships {
        (sh[0].model_id.clone(), sh.iter().map(|r| r.artifacts.clone()).collect::<Vec<_>>())
    } else if cfg.exl3 {
        (model_dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "exl3".into()),
         vec![Vec::new(); world as usize])
    } else {
        build_manifest(model_dir, world)?
    };
    let total: u64 = per_rank.iter().flatten().map(|a| a.size).sum();
    eprintln!("[head] manifest '{model_id}': {} artifacts, {:.2} GB{}", total_artifacts(&per_rank), total as f64/1e9,
              if ships.is_some() { " (B32: per-rank shards through the blob cache)" }
              else if cfg.exl3 { " (EXL3 legacy: nodes load their local pack at the head's path, unchecked)" } else { "" });

    let ifaces = roce_interfaces();
    let mut nodes: Vec<NodeInfo> = match explicit {
        Some(addrs) => addrs.into_iter()
            .map(|a| NodeInfo { hostname: a.ip().to_string(), addr: a }).collect(),
        None => {
            eprintln!("[head] discovering nodes (UDP broadcast, {}s) ...", discover_wait.as_secs());
            let n = discover_nodes(discover_wait)?;
            if n.is_empty() { bail!("no nodes discovered — is a --node running? (or pass --nodes <ip:port>)"); }
            for x in &n { eprintln!("  [head] found node '{}' at {}", x.hostname, x.addr); }
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
    let mut streams = Vec::with_capacity(nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        let rank = (i + 1) as i32;
        // Bench sessions ship no drafter (bench nodes never load a round; the bench wire adds
        // only the empty-DraftManifest round to the pre-v5 behavior).
        let (mid, srcs) = match &ships {
            Some(sh) => (sh[rank as usize].model_id.as_str(), Some(&sh[rank as usize].srcs)),
            None => (model_id.as_str(), None),
        };
        streams.push(head_sync_one(node, rank, model_dir, mid, &per_rank[rank as usize], &topology, cfg, None, srcs)?);
    }
    eprintln!("[head] all {} node(s) synced.", nodes.len());
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
pub fn run_head_session(model_dir: &Path, explicit: Option<Vec<SocketAddr>>, discover_wait: Duration,
                        cfg: &crate::tp::TpConfig)
    -> Result<(Vec<NodeInfo>, Vec<TcpStream>)>
{
    let world = cfg.world.max(1);
    eprintln!("[head] {} — building manifest for {} (world {world}) ...", hostname(), model_dir.display());
    // B32: per-rank shard dirs instead of the whole-model replica — always, when the shard plan covers the model
    let ships = if crate::shard_plan::applies(model_dir, world as usize).is_some() { Some(ship_plans(model_dir, world)?) } else { None };
    let (model_id, per_rank) = match &ships {
        Some(sh) => (sh[0].model_id.clone(), sh.iter().map(|r| r.artifacts.clone()).collect::<Vec<_>>()),
        None => build_manifest(model_dir, world)?,
    };
    let total: u64 = per_rank.iter().flatten().map(|a| a.size).sum();
    eprintln!("[head] manifest '{model_id}': {} artifacts, {:.2} GB{}", total_artifacts(&per_rank), total as f64/1e9,
              if ships.is_some() { " (B32: per-rank shards through the blob cache)" } else { "" });

    let ifaces = roce_interfaces();
    let mut nodes: Vec<NodeInfo> = match explicit {
        Some(addrs) => addrs.into_iter()
            .map(|a| NodeInfo { hostname: a.ip().to_string(), addr: a }).collect(),
        None => {
            eprintln!("[head] discovering nodes (UDP broadcast, {}s) ...", discover_wait.as_secs());
            let n = discover_nodes(discover_wait)?;
            if n.is_empty() { bail!("no nodes discovered — is a --node running? (or pass --nodes <ip:port>)"); }
            for x in &n { eprintln!("  [head] found node '{}' at {}", x.hostname, x.addr); }
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
    let src_parsed = crate::batch::SpecSource::from_cli(&cfg.spec_source)
        .unwrap_or(crate::batch::SpecSource::Mtp);
    // P14: `dflash` (the v1 BLOCK drafter) ships through the same content-addressed slot: its lane
    // is part of the mirrored `decode_step`, so every rank needs the artifact, and the node must
    // never read the head's filesystem path.
    let draft: Option<(String, Vec<Artifact>)> = if cfg.mode_serve
        && (crate::batch::is_df2_src(src_parsed)
            || matches!(src_parsed, crate::batch::SpecSource::Dspark)
            || matches!(src_parsed, crate::batch::SpecSource::DFlash))
        && !cfg.df2_draft_dir.is_empty()
    {
        match draft_manifest(Path::new(&cfg.df2_draft_dir)) {
            Ok((id, arts)) => {
                let dtotal: u64 = arts.iter().map(|a| a.size).sum();
                eprintln!("[head] draft manifest '{id}': {} artifacts, {:.2} GB — ships to every node",
                          arts.len(), dtotal as f64 / 1e9);
                Some((id, arts))
            }
            Err(e) => {
                eprintln!("[head] WARN: draft manifest build FAILED ({e:#}) — nodes get NO drafter; \
                           the head's round load fails the same way and all ranks fall back to MTP");
                None
            }
        }
    } else { None };

    // Sync every node and RETAIN every node's sync stream, RANK-INDEXED (streams[i] == rank i+1).
    // The serving control plane (CalibTable, per-step Step, Shutdown) must fan out to ALL world-1
    // nodes; dropping rank 2..N-1's streams here is exactly what deadlocked world>2 bring-up (those
    // nodes reached node_serve_tp and blocked forever on a CalibTable the head never sent them).
    let mut streams: Vec<TcpStream> = Vec::with_capacity(need);
    for (i, node) in nodes.iter().enumerate() {
        let rank = (i + 1) as i32;
        let (mid, srcs) = match &ships {
            Some(sh) => (sh[rank as usize].model_id.as_str(), Some(&sh[rank as usize].srcs)),
            None => (model_id.as_str(), None),
        };
        let stream = head_sync_one(node, rank, model_dir, mid, &per_rank[rank as usize],
                                   &topology, cfg, draft.as_ref(), srcs)?;
        streams.push(stream);
    }
    anyhow::ensure!(streams.len() == need,
        "expected {need} retained serving control stream(s), got {}", streams.len());
    eprintln!("[head] {} node(s) synced; all control streams RETAINED for the serving session", streams.len());
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

    // Synthetic RoCE rails: rail 1 = 192.168.177.0/24, rail 2 = 192.168.178.0/24.
    fn rails() -> Vec<Roce> {
        vec![
            Roce {
                ib: "rocep1s0f1".into(),
                netdev: "r1".into(),
                ip: "192.168.177.1".parse().unwrap(),
                bcast: "192.168.177.255".parse().unwrap(),
                mask: 0xffffff00,
                rail: 1,
            },
            Roce {
                ib: "roceP2p1s0f1".into(),
                netdev: "r2".into(),
                ip: "192.168.178.1".parse().unwrap(),
                bcast: "192.168.178.255".parse().unwrap(),
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
            node("b", "192.168.178.99", 29500),
            node("c", "192.168.177.13", 29500),
            node("d", "192.168.177.12", 29500),
        ];
        assign_ranks(&mut nodes, &ifaces);
        let order: Vec<String> = nodes.iter().map(|n| n.hostname.clone()).collect();
        assert_eq!(order, vec!["d", "c", "b", "a"], "rail desc, then ip asc");

        // Reassign with the same set in a different input order -> identical result.
        let mut shuffled = vec![
            node("d", "192.168.177.12", 29500),
            node("b", "192.168.178.99", 29500),
            node("a", "10.0.0.1", 29500),
            node("c", "192.168.177.13", 29500),
        ];
        assign_ranks(&mut shuffled, &ifaces);
        let order2: Vec<String> = shuffled.iter().map(|n| n.hostname.clone()).collect();
        assert_eq!(order, order2, "order must be input-order independent");
    }

    #[test]
    fn topology_is_full_and_indexed_by_rank() {
        let ifaces = rails();
        let mut nodes = vec![
            node("b", "192.168.178.99", 29500),
            node("c", "192.168.177.13", 29500),
            node("d", "192.168.177.12", 29500),
        ];
        assign_ranks(&mut nodes, &ifaces);
        let topo = build_topology(&nodes, &ifaces, 4).unwrap();
        assert_eq!(topo.len(), 4);
        assert_eq!(topo[0], "192.168.177.1"); // head = rail 1
        assert_eq!(topo[1], "192.168.177.12"); // rail 1, lowest ip
        assert_eq!(topo[2], "192.168.177.13"); // rail 1, next ip
        assert_eq!(topo[3], "192.168.178.99"); // rail 2 last
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
        let seq: Vec<String> = jobs.iter().map(|(p, _)| sha256_file(p).unwrap().0).collect();
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
}
