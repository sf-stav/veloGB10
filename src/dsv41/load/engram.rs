//! DSV4.1 loader (leg LS): AUTOMATIC ENGRAM (UX row 3). A rank needs the packed head files of its
//! Engram heads (ENGb: `engram_head<h>.bin` + `.manifest.json`, 25.3 GB each, the FP8 tables of native
//! shards 47 / 48 for EVERY pack). The operator does nothing:
//!
//! 1. the head finds the packed files in `--engram-dir` (default: the one `~/dsv41/engram/fp8-*` that
//!    exists, else `~/dsv41/engram/fp8-<tag>`; `<tag>` = the first 8 hex of the native index's sha256);
//! 2. a head whose files are missing or do not match their manifest (size) is packed from the native
//!    checkpoint (`--engram-source`, else found next to the pack) WHILE IT IS SENT and never written on the
//!    head's disk (LS2: `engram_gen::GenHead`, one notice line, `[boot] dsv41 shipping Engram heads: NN%`); its
//!    sha256 comes from a manifest recorded in `--engram-dir` (a few KB, kept without the file) or one hashing
//!    pass that records it. A node that already holds the head (pre-staged; manifest sha256 and size match)
//!    adopts it into its cache and is sent nothing (`cluster::node::adopt_staged`). `ensure_heads` (pack to the
//!    head's own disk) remains for operators who want the file here;
//! 3. each rank is shipped ONLY its heads (TP8: head r; TP4: 2r, 2r+1; TP2: 4r..4r+3) as two planned
//!    files, `engram/engram_head<h>.bin` and `.manifest.json`, through the same content-addressed blob
//!    cache as the shard (a second start ships nothing). The node's assembled dir carries them under
//!    `engram/`, which is where the runtime opens them (`PackedHead::open(<shard dir>/engram, h)`).
//!
//! `--dsv41-ship-engram off` ships no Engram (an operator who pre-staged every box's NVMe).

use crate::dsv41::config::Dsv41Config;
use crate::dsv41::engram::manifest::{file_name, manifest_name, PackedHead, RecordLayout};
use crate::dsv41::engram::pack;
use crate::dsv41::rankmap::engram_row_bytes;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The heads rank `rank` of `world` owns (TP8 head r; TP4 {2r, 2r+1}; TP2 {4r..4r+3}).
pub fn heads_of_rank(n_heads: usize, world: usize, rank: usize) -> std::ops::Range<usize> {
    (rank * n_heads / world.max(1))..((rank + 1) * n_heads / world.max(1))
}

/// A planned Engram file: where it is and the name it ships under.
#[derive(Clone, Debug)]
pub struct EngramFile { pub head: usize, pub logical: String, pub path: PathBuf, pub bytes: u64 }

/// The shard-dir subdirectory the runtime opens the packed heads from.
pub const SUBDIR: &str = "engram";

fn home() -> PathBuf { PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| "/root".into())) }

/// The one command-line value of `--name` (these are shared HAND_FLAGS, `src/opts/mod.rs`).
fn cli_value(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    crate::arg_value(&args, name).map(str::to_string)
}

/// The directory the packed heads live in on this box.
pub fn engram_dir(source: Option<&Path>) -> PathBuf {
    if let Some(d) = cli_value("--engram-dir") { return PathBuf::from(d); }
    let root = home().join("dsv41").join("engram");
    let mut existing: Vec<PathBuf> = std::fs::read_dir(&root).map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.file_name().map_or(false, |n| n.to_string_lossy().starts_with("fp8-")) && p.is_dir()).collect()).unwrap_or_default();
    existing.sort();
    if let Some(p) = existing.into_iter().next() { return p; }
    let tag = source.and_then(|s| index_tag(s).ok()).unwrap_or_else(|| "unknown".into());
    root.join(format!("fp8-{tag}"))
}

/// First 8 hex of the native index's sha256 (a stable name for "these tables").
pub(super) fn index_tag(source: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let b = std::fs::read(source.join("model.safetensors.index.json"))?;
    Ok(Sha256::digest(&b).iter().take(4).map(|x| format!("{x:02x}")).collect())
}

/// The native checkpoint to pack from: `--engram-source`, the pack dir itself if it carries the tables, or a
/// DeepSeek-V4.1 dir that does, next to the pack (the pack's parent, its grandparent, one level down each).
pub fn find_source(model_dir: &Path) -> Result<PathBuf> {
    let carries = |d: &Path| d.join("config.json").is_file() && crate::dsv41::engram::source::load(d).is_ok();
    if let Some(s) = cli_value("--engram-source") {
        let p = PathBuf::from(s);
        if carries(&p) { return Ok(p); }
        bail!("--engram-source {} does not hold the Engram tables (layers.N.engram.embed.* in the native shards)", p.display());
    }
    if carries(model_dir) { return Ok(model_dir.to_path_buf()); }
    let bundled = model_dir.join("engram_source");
    if carries(&bundled) { return Ok(bundled); }
    let mut roots = Vec::new();
    for up in [model_dir.parent(), model_dir.parent().and_then(|p| p.parent())].into_iter().flatten() { roots.push(up.to_path_buf()); }
    let mut tried = Vec::new();
    for r in roots {
        let mut cands: Vec<PathBuf> = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&r) {
            for e in rd.filter_map(|e| e.ok()) {
                let p = e.path();
                if !p.is_dir() { continue; }
                cands.push(p.clone());
                if let Ok(rd2) = std::fs::read_dir(&p) { cands.extend(rd2.filter_map(|e| e.ok()).map(|e| e.path()).filter(|q| q.is_dir())); }
            }
        }
        cands.sort();
        for c in cands {
            let n = c.file_name().map(|s| s.to_string_lossy().to_lowercase()).unwrap_or_default();
            if !n.contains("deepseek-v4.1") || n.contains("exl3") || n == model_dir.file_name().map(|s| s.to_string_lossy().to_lowercase()).unwrap_or_default() { continue; }
            tried.push(c.display().to_string());
            if carries(&c) { return Ok(c); }
        }
    }
    bail!("the Engram tables are not in {} (an EXL3 / converted pack carries none: the tables come from DeepSeek's native FP8 checkpoint, shards 47 and 48). \
Point --engram-source at the native DeepSeek-V4.1-Flash dir, or pre-pack with --dsv41-engram-pack{}", model_dir.display(),
          if tried.is_empty() { String::new() } else { format!(" (looked at: {})", tried.join(", ")) })
}

/// Is head `h` complete in `dir` (manifest parses, file size equals the manifest's)?
pub fn head_ok(dir: &Path, h: usize) -> bool { PackedHead::open(dir, h).is_ok() }

/// What a pack of `n` heads costs: (bytes, a rough minutes figure at the measured ~0.2 GB/s per pack run, 4 in parallel).
fn pack_cost(cfg: &Dsv41Config, n: usize) -> (u64, f64) {
    let lay = crate::dsv41::rankmap::engram_layout(cfg);
    let per = crate::dsv41::rankmap::engram_bytes_for(cfg, &lay, 0..1);
    let bytes = per * n as u64;
    (bytes, (n as f64 / 4.0).ceil() * 125.0 / 60.0)
}

/// Make sure every head in `heads` is complete in `dir`, packing the missing ones from the native checkpoint
/// (found next to `model_dir`). Returns the heads that were packed now. Idempotent: complete heads cost one
/// header read.
pub fn ensure_heads(cfg: &Dsv41Config, model_dir: &Path, dir: &Path, heads: &[usize]) -> Result<Vec<usize>> {
    let missing: Vec<usize> = heads.iter().copied().filter(|&h| !head_ok(dir, h)).collect();
    if missing.is_empty() { return Ok(missing); }
    let source = find_source(model_dir)?;
    let (bytes, mins) = pack_cost(cfg, missing.len());
    crate::boot_report::notice(&format!(
        "Engram head file(s) {missing:?} missing in {} → packing them from {} ({:.1} GB, about {mins:.0} min; later starts find them; pre-stage with --dsv41-engram-pack)",
        dir.display(), source.display(), bytes as f64 / 1e9));
    for h in &missing { // a half-written head (manifest without a matching file) must be rebuilt, not skipped
        if dir.join(manifest_name(*h)).exists() { let _ = std::fs::remove_file(dir.join(manifest_name(*h))); }
    }
    let threads = cli_value("--engram-pack-threads").and_then(|s| s.parse().ok()).unwrap_or(4usize);
    pack::pack_heads(&source, &missing, dir, RecordLayout::Packed, threads, false)
        .with_context(|| format!("packing Engram heads {missing:?} into {} (fix: free the space, or --engram-dir <a drive with room>)", dir.display()))?;
    super::metrics_rows::engram_heads_packed(missing.len() as u64);
    Ok(missing)
}

/// The files to ship rank `rank`: its heads' `.bin` and `.manifest.json` (heads complete in `dir`).
pub fn files_for_rank(cfg: &Dsv41Config, world: usize, rank: usize, dir: &Path) -> Result<Vec<EngramFile>> {
    let mut out = Vec::new();
    for h in heads_of_rank(cfg.engram_heads, world, rank) {
        let ph = PackedHead::open(dir, h).map_err(|e| anyhow::anyhow!("Engram head {h} in {}: {e}", dir.display()))?;
        out.push(EngramFile { head: h, logical: format!("{SUBDIR}/{}", file_name(h)), path: dir.join(file_name(h)), bytes: ph.manifest.file_bytes });
        let mp = dir.join(manifest_name(h));
        out.push(EngramFile { head: h, logical: format!("{SUBDIR}/{}", manifest_name(h)), bytes: std::fs::metadata(&mp)?.len(), path: mp });
    }
    Ok(out)
}

/// The planned Engram files of rank `rank` WITHOUT holding them on this disk: a head complete in `dir` ships as the
/// file; a head that is not (the head box cannot keep all eight) is packed from the native checkpoint while it is
/// sent (`engram_gen::GenHead`), its manifest built from the recorded sha256 (or from one hashing pass, recorded).
/// Returns (logical name, source) pairs, the `.bin` and the `.manifest.json` of every head of the rank.
pub fn plan_sources_for_rank(cfg: &Dsv41Config, model_dir: &Path, world: usize, rank: usize, dir: &Path) -> Result<Vec<(String, crate::shard_plan::PlanSrc)>> {
    use crate::shard_plan::PlanSrc;
    let heads: Vec<usize> = heads_of_rank(cfg.engram_heads, world, rank).collect();
    let missing: Vec<usize> = heads.iter().copied().filter(|&h| !head_ok(dir, h)).collect();
    let mut generated: std::collections::BTreeMap<usize, (super::engram_gen::GenHead, crate::dsv41::engram::manifest::HeadManifest)> = Default::default();
    if !missing.is_empty() {
        let source = find_source(model_dir)?;
        let src = super::engram_gen::load_source(&source)?;
        crate::boot_report::notice(&format!(
            "Engram head file(s) {missing:?} are not in {} on this box: they are packed from {} and streamed to rank {rank} while they are packed (nothing is written here; {:.1} GB each)",
            dir.display(), source.display(), pack_cost(cfg, 1).0 as f64 / 1e9));
        let mut gens = Vec::new();
        for &h in &missing { gens.push(super::engram_gen::GenHead::new(&source, src.clone(), h, dir)?); }
        let out: Vec<Result<(usize, super::engram_gen::GenHead, crate::dsv41::engram::manifest::HeadManifest)>> = std::thread::scope(|sc| {
            let hs: Vec<_> = gens.into_iter().map(|mut g| { let h = g.head(); sc.spawn(move || { let m = g.manifest(dir)?; Ok((h, g, m)) }) }).collect();
            hs.into_iter().map(|j| j.join().unwrap_or_else(|_| Err(anyhow::anyhow!("a hashing thread panicked")))).collect()
        });
        for r in out { let (h, g, m) = r?; generated.insert(h, (g, m)); }
    }
    let mut files = Vec::new();
    for &h in &heads {
        let (bin, man) = super::engram_gen::names(h);
        if let Some((g, m)) = generated.remove(&h) {
            files.push((bin, PlanSrc::Generated(Arc::new(g))));
            files.push((man, PlanSrc::Bytes(serde_json::to_vec_pretty(&m)?)));
        } else {
            files.push((bin, PlanSrc::Whole(dir.join(file_name(h)))));
            files.push((man, PlanSrc::Whole(dir.join(manifest_name(h)))));
        }
    }
    Ok(files)
}

/// NODE side: a head file this box already holds, pre-staged in an Engram dir (`~/dsv41/engram/fp8-*`, or
/// `--engram-dir`) with a manifest whose sha256 and size are the shipped artifact's. `logical` is the artifact's
/// name (`engram/engram_head<h>.bin`). Returns the file's path, so the cache can adopt it as the blob instead of
/// receiving 25 GB. (The match is the manifest's recorded sha256 plus the file's size, not a re-hash; verify a
/// staged file with `--dsv41-engram-verify --engram-verify-sha-only`.)
pub fn find_staged(logical: &str, hash: &str, size: u64) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(d) = cli_value("--engram-dir") { dirs.push(PathBuf::from(d)); }
    if let Ok(rd) = std::fs::read_dir(home().join("dsv41").join("engram")) {
        dirs.extend(rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_dir()));
    }
    staged_in(&dirs, logical, hash, size)
}

/// `find_staged` over an explicit list of directories.
pub fn staged_in(dirs: &[PathBuf], logical: &str, hash: &str, size: u64) -> Option<PathBuf> {
    let name = logical.strip_prefix("engram/")?;
    let h: usize = name.strip_prefix("engram_head")?.strip_suffix(".bin")?.parse().ok()?;
    for d in dirs {
        let Ok(txt) = std::fs::read(d.join(manifest_name(h))) else { continue };
        let Ok(m) = serde_json::from_slice::<crate::dsv41::engram::manifest::HeadManifest>(&txt) else { continue };
        let bin = d.join(file_name(h));
        if m.head == h && m.sha256 == hash && m.file_bytes == size && std::fs::metadata(&bin).map(|md| md.len()).ok() == Some(size) { return Some(bin); }
    }
    None
}

/// Bytes a rank's Engram heads occupy on its drive (the disk figure of the fit check and the dry run).
pub fn bytes_for_rank(cfg: &Dsv41Config, world: usize, rank: usize) -> u64 {
    let lay = crate::dsv41::rankmap::engram_layout(cfg);
    let h = heads_of_rank(cfg.engram_heads, world, rank);
    // the records plus the 4 KiB header page per head
    crate::dsv41::rankmap::engram_bytes_for(cfg, &lay, h.clone()) + 4096 * h.len() as u64
}

pub fn row_bytes(cfg: &Dsv41Config) -> u64 { engram_row_bytes(cfg) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heads_follow_the_owner_table() {
        assert_eq!(heads_of_rank(8, 8, 5), 5..6);
        assert_eq!(heads_of_rank(8, 4, 2), 4..6);
        assert_eq!(heads_of_rank(8, 2, 1), 4..8);
        assert_eq!(heads_of_rank(8, 1, 0), 0..8);
        // the same table as rankmap's
        let c = crate::dsv41::config::Dsv41Config::from_value(&crate::dsv41::config::test_config_json(), None).unwrap();
        for p in crate::dsv41::rankmap::Posture::ALL {
            for s in crate::dsv41::rankmap::rank_map(&c, p) { assert_eq!(heads_of_rank(c.engram_heads, p.world(), s.rank), s.engram_heads); }
        }
    }

    #[test]
    fn a_staged_head_is_recognised_by_manifest_sha_and_size_only() {
        use crate::dsv41::engram::manifest::*;
        let d = std::env::temp_dir().join(format!("ls2_staged_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let geo = Geometry::compute(&[1, 14], 4, 8, 16_000_000, &[384006168, 384016682]);
        let m = HeadManifest { format: "dsv41-engram-pack".into(), version: FORMAT_VERSION, head: 5, file: file_name(5), record_bytes: RECORD_BYTES,
            value_bytes: VALUE_BYTES, scale_bytes: SCALE_BYTES, layout: RecordLayout::Packed, records_per_page: RECORDS_PER_PAGE, page_bytes: PAGE_BYTES,
            header_bytes: HEADER_BYTES, total_records: 3, file_bytes: 8192, sha256: "ab".repeat(32), source_model: "x".into(), source_layers: vec![], geometry: geo, sections: vec![] };
        std::fs::write(d.join(manifest_name(5)), serde_json::to_vec(&m).unwrap()).unwrap();
        let dirs = vec![d.clone()];
        let l = "engram/engram_head5.bin";
        assert!(staged_in(&dirs, l, &m.sha256, 8192).is_none(), "manifest without its file is not a staged head");
        std::fs::write(d.join(file_name(5)), vec![0u8; 8192]).unwrap();
        assert_eq!(staged_in(&dirs, l, &m.sha256, 8192), Some(d.join(file_name(5))));
        assert!(staged_in(&dirs, l, &"cd".repeat(32), 8192).is_none(), "another sha256");
        assert!(staged_in(&dirs, l, &m.sha256, 8193).is_none(), "another size");
        assert!(staged_in(&dirs, "engram/engram_head6.bin", &m.sha256, 8192).is_none(), "another head");
        assert!(staged_in(&dirs, "seg/L00.safetensors", &m.sha256, 8192).is_none(), "not an Engram artifact");
        std::fs::write(d.join(file_name(5)), vec![0u8; 100]).unwrap();
        assert!(staged_in(&dirs, l, &m.sha256, 8192).is_none(), "a truncated file");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn a_missing_source_names_the_fix() {
        let d = std::env::temp_dir().join(format!("ls_engsrc_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let e = find_source(&d).unwrap_err().to_string();
        assert!(e.contains("--engram-source") && e.contains("native FP8 checkpoint"), "{e}");
        std::fs::remove_dir_all(&d).ok();
    }
}
