//! DSV4.1 loader (leg LS): the per-rank SHARD DIR the head ships through the content-addressed blob cache
//! (UX row 2) — the planner `shard_plan::shard_plan` dispatches to for `ShardFormat::Dsv41`.
//!
//! The shard dir of rank r holds, all as ordinary planned files (whole files, safetensors SEGMENTS
//! synthesized from byte ranges of the pack's shards — no second copy on the head's disk — and small
//! generated files):
//!   * `seg/L<NN>.safetensors` — layer NN's tensors as the rank holds them (a slice keeps the tensor's name
//!     and carries the held shape), `seg/global.safetensors` — embedding, head, final norm, vision (rank 0);
//!   * `model.safetensors.index.json` (the segments), the pack's small files (config, quantization config,
//!     tokenizer, chat template, generation config), `gb10_shard.json` (the shared shard manifest) and
//!     `dsv41_shard.json` (this leg's: the plan options a loader needs to re-derive every cut);
//!   * `engram/engram_head<h>.{bin,manifest.json}` for the rank's Engram heads (`engram.rs`).
//!
//! Segments are per layer so the content addressing works at layer grain: a plan change touches only the
//! layers it moves, and a node that already holds a layer's blob is sent nothing.

use super::engram;
use super::plan::{self, ExpertSplit, PlanOpts, RankSelection};
use crate::dsv41::census::Census;
use crate::dsv41::config::Dsv41Config;
use crate::shard_plan::{Piece, PlanFile, PlanSrc, RankPlan, ShardFormat, ShardInfo, PLAN_VERSION, SHARD_FILE};
use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

/// The generated file a loader reads to re-derive the cuts (`plan::placement` is pure in these).
pub const MANIFEST_FILE: &str = "dsv41_shard.json";

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq)]
pub struct ShardManifest {
    pub version: u32,
    pub world: usize,
    pub rank: usize,
    pub expert_split: String,
    pub attn_whole: bool,
    pub engram_wkv_whole: bool,
    pub embed_rank0_only: bool,
    /// leg PA1 (absent in older manifests: false)
    #[serde(default)]
    pub hc_whole: bool,
    pub source_model: String,
    pub engram_heads: Vec<usize>,
    /// vocabulary rows of the head / Markov head this rank holds (whole 128-blocks)
    pub vocab: [usize; 2],
    pub tensors: usize,
    pub tensor_bytes: u64,
}

impl ShardManifest {
    pub fn opts(&self) -> Result<PlanOpts> {
        Ok(PlanOpts { world: self.world, expert_split: ExpertSplit::parse(&self.expert_split).context("expert_split")?,
                      attn_whole: self.attn_whole, engram_wkv_whole: self.engram_wkv_whole, embed_rank0_only: self.embed_rank0_only, hc_whole: self.hc_whole })
    }
}

fn census_cache() -> &'static Mutex<BTreeMap<std::path::PathBuf, Arc<Census>>> {
    static C: OnceLock<Mutex<BTreeMap<std::path::PathBuf, Arc<Census>>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// The census of `dir`, built once per process (a ship plans W-1 ranks from the same one).
pub fn census_of(dir: &Path) -> Result<Arc<Census>> {
    let key = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    if let Some(c) = census_cache().lock().unwrap().get(&key) { return Ok(c.clone()); }
    let mut prog: Option<crate::boot_report::Progress> = None;
    let c = Census::build_with(dir, &mut |done, total| { prog.get_or_insert_with(|| super::progress(super::Phase::Headers, None, total)).set(done); })?;
    if let Some(p) = prog { p.finish(); }
    c.check().with_context(|| format!("{}: the pack does not census cleanly; no shard plan", dir.display()))?;
    let a = Arc::new(c);
    census_cache().lock().unwrap().insert(key, a.clone());
    Ok(a)
}

fn opt_str(flag: &str) -> Option<String> { crate::opts::lookup(flag).and_then(crate::opts::get).filter(|s| s != "unset") }

/// The deal name a ship is filed under (the cache id and `RankPlan::deal`): the expert split.
pub fn deal() -> String { opt_str("dsv41-expert-split").unwrap_or_else(|| "inner".into()) }

/// The plan options of this start (flags `--dsv41-expert-split`, `--dsv41-attn-whole`).
pub fn plan_opts(c: &Census, world: usize, deal: &str) -> Result<PlanOpts> {
    let native = !c.modules.iter().any(|m| matches!(m.format, crate::dsv41::census::LinearFormat::Exl3 { .. }));
    let split = ExpertSplit::parse(deal).with_context(|| format!("--dsv41-expert-split takes inner|ep, not {deal:?}"))?;
    Ok(PlanOpts::resolve(world, split, &opt_str("dsv41-attn-whole").unwrap_or_else(|| "auto".into()), native)?
        .with_hc_whole(crate::dsv41::tp::pa1::hc_whole_for(world)))
}

fn piece_of(c: &Census, dir: &Path, h: &plan::Hold) -> Piece {
    let e = &c.entries[h.entry as usize];
    let src = dir.join(&c.files[e.file as usize]);
    match &h.slice {
        None => Piece { name: e.name.clone(), dtype: e.dtype.clone(), shape: e.shape.iter().map(|&x| x as usize).collect(), src, off: e.offset, len: e.bytes, rows: 1, stride: 0 },
        Some(s) => {
            let (off, rows, stride, len) = s.geometry();
            Piece { name: e.name.clone(), dtype: e.dtype.clone(), shape: h.shape.iter().map(|&x| x as usize).collect(), src, off: e.offset + off, len, rows, stride }
        }
    }
}

/// The files of one rank, WITHOUT the Engram heads (the dry run and the stage cache use it as is).
pub fn rank_files(c: &Census, dir: &Path, o: &PlanOpts, sel: &RankSelection) -> Result<(Vec<PlanFile>, ShardManifest)> {
    let mut by_seg: BTreeMap<String, Vec<Piece>> = BTreeMap::new();
    let mut weight_map: BTreeMap<String, String> = BTreeMap::new();
    let mut holds: Vec<&plan::Hold> = sel.holds.iter().collect();
    holds.sort_by_key(|h| { let e = &c.entries[h.entry as usize]; (e.layer.map_or(u32::MAX, |l| l as u32), e.file, e.offset) });
    for h in holds {
        let e = &c.entries[h.entry as usize];
        let seg = match e.layer { Some(l) => format!("seg/L{l:02}.safetensors"), None => "seg/global.safetensors".to_string() };
        weight_map.insert(e.name.clone(), seg.clone());
        by_seg.entry(seg).or_default().push(piece_of(c, dir, h));
    }
    let mut files: Vec<PlanFile> = by_seg.into_iter().map(|(logical, ps)| PlanFile { logical, src: PlanSrc::Segment(ps) }).collect();
    let tensor_bytes: u64 = sel.bytes();
    let heads = engram::heads_of_rank(c.cfg.engram_heads, o.world, sel.rank);
    let v = crate::dsv41::load::splits::vocab_range(c.cfg.vocab, o.world, sel.rank);
    let man = ShardManifest {
        version: 1, world: o.world, rank: sel.rank, expert_split: o.expert_split.name().into(), attn_whole: o.attn_whole,
        engram_wkv_whole: o.engram_wkv_whole, embed_rank0_only: o.embed_rank0_only, hc_whole: o.hc_whole,
        source_model: dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(),
        engram_heads: heads.collect(), vocab: [v.start, v.end], tensors: sel.holds.len(), tensor_bytes,
    };
    let seg_total: u64 = files.iter().map(|f| RankPlan::file_size(f)).sum::<Result<u64>>()?;
    let index = serde_json::to_vec_pretty(&serde_json::json!({"metadata": {"total_size": seg_total, "gb10_shard": PLAN_VERSION}, "weight_map": weight_map}))?;
    files.push(PlanFile { logical: "model.safetensors.index.json".into(), src: PlanSrc::Bytes(index) });
    for f in crate::cluster::PACK_READ_FILES {
        if f != "model.safetensors.index.json" && dir.join(f).is_file() { files.push(PlanFile { logical: f.to_string(), src: PlanSrc::Whole(dir.join(f)) }); }
    }
    let info = ShardInfo { plan_version: PLAN_VERSION, format: ShardFormat::Dsv41.name().into(), world: o.world, rank: sel.rank, deal: o.expert_split.name().into(),
                           source_model: man.source_model.clone(), presliced: Vec::new() };
    files.push(PlanFile { logical: SHARD_FILE.into(), src: PlanSrc::Bytes(serde_json::to_vec_pretty(&info)?) });
    files.push(PlanFile { logical: MANIFEST_FILE.into(), src: PlanSrc::Bytes(serde_json::to_vec_pretty(&man)?) });
    files.sort_by(|a, b| a.logical.cmp(&b.logical));
    Ok((files, man))
}

/// `shard_plan::shard_plan` for a V4.1 pack: rank `rank` of `world`, the Engram heads included (packed first
/// when missing, unless `--dsv41-ship-engram off`).
pub fn plan_rank(dir: &Path, world: usize, rank: usize, deal: &str) -> Result<RankPlan> {
    if world < 1 || rank >= world { bail!("dsv41 shard plan: rank {rank} of world {world}"); }
    let c = census_of(dir)?;
    let o = plan_opts(&c, world, deal)?;
    let sel = plan::select(&c, &o, rank)?;
    let (mut files, _man) = rank_files(&c, dir, &o, &sel)?;
    if opt_str("dsv41-ship-engram").as_deref() != Some("off") && rank != 0 {
        let edir = engram::engram_dir(engram::find_source(dir).ok().as_deref());
        for (logical, src) in engram::plan_sources_for_rank(&c.cfg, dir, world, rank, &edir)? { files.push(PlanFile { logical, src }); }
        files.sort_by(|a, b| a.logical.cmp(&b.logical));
    }
    super::metrics_rows::planned(world, rank, files.iter().filter_map(|f| RankPlan::file_size(f).ok()).sum());
    Ok(RankPlan { format: ShardFormat::Dsv41, world, rank, deal: o.expert_split.name().into(), files, presliced: Vec::new() })
}

/// Does `dir` hold a V4.1 pack the shard plan covers at `world`? (`shard_plan::applies`)
pub fn applies(dir: &Path, world: usize) -> bool {
    world >= 2 && crate::dsv41::config::is_dsv41_dir(&dir.to_string_lossy()) && Dsv41Config::load(dir).map_or(false, |c| {
        crate::dsv41::rankmap::Posture::from_world(world).map_or(false, |p| crate::dsv41::rankmap::posture_divides(&c, p).is_empty())
    })
}

/// Read a shipped / staged shard dir's manifest.
pub fn read_manifest(shard_dir: &Path) -> Result<ShardManifest> {
    let p = shard_dir.join(MANIFEST_FILE);
    serde_json::from_slice(&std::fs::read(&p).with_context(|| format!("{}: not a dsv41 shard dir", shard_dir.display()))?).context(MANIFEST_FILE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard_plan::write_segment;

    const SLICE: &str = "/mnt/models/dsv41/slices/exl3-eaq-v1-2.9-L0-1-2-14-20-21-24-40-41-42";

    /// Real bytes: the rank slices of an expert's gate trellis, reassembled column-block by column-block, are the
    /// source tensor; the down trellis slices concatenate to it; every rank's plan lists the same tensors the census has.
    #[test]
    fn real_slice_pack_rank_slices_reassemble_the_source_bytes() {
        let dir = Path::new(SLICE);
        if !dir.join("config.json").is_file() { eprintln!("SKIP: slice pack absent"); return; }
        let c = census_of(dir).unwrap();
        for world in [2usize, 4, 8] {
            let o = plan_opts(&c, world, "inner").unwrap();
            let sels: Vec<RankSelection> = (0..world).map(|r| plan::select(&c, &o, r).unwrap()).collect();
            for name in ["layers.2.ffn.experts.5.w1.trellis", "layers.2.ffn.experts.5.w2.trellis", "layers.2.ffn.experts.5.w1.svh", "layers.2.ffn.experts.5.w2.suh"] {
                let ei = c.entries.iter().position(|e| e.name == name).unwrap();
                let e = &c.entries[ei];
                let full = { let mut v = vec![0u8; e.bytes as usize]; use std::os::unix::fs::FileExt; std::fs::File::open(dir.join(&c.files[e.file as usize])).unwrap().read_exact_at(&mut v, e.offset).unwrap(); v };
                let mut parts = Vec::new();
                for s in &sels {
                    let h = s.holds.iter().find(|h| h.entry as usize == ei).expect("held");
                    let mut b = Vec::new();
                    write_segment(&[piece_of(&c, dir, h)], &mut b).unwrap();
                    let hl = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize + 8;
                    parts.push((h.clone(), b[hl..].to_vec()));
                }
                assert_eq!(parts.iter().map(|p| p.1.len()).sum::<usize>(), full.len(), "{name} w{world}");
                let spec0 = parts[0].0.slice.clone().unwrap();
                // rebuild: place each rank's runs back at (outer, start) of the full tensor
                let mut back = vec![0u8; full.len()];
                for (h, bytes) in &parts {
                    let sp = h.slice.as_ref().unwrap();
                    let run = (sp.len * sp.inner) as usize;
                    for r in 0..sp.outer as usize {
                        let dst = (r as u64 * sp.axis * sp.inner + sp.start * sp.inner) as usize;
                        back[dst..dst + run].copy_from_slice(&bytes[r * run..(r + 1) * run]);
                    }
                }
                assert!(back == full, "{name} w{world}: reassembled bytes differ (spec {spec0:?})");
            }
            // every rank plans the same set of non-expert tensors it should: nothing the census lacks
            assert!(sels.iter().all(|s| s.holds.iter().all(|h| (h.entry as usize) < c.entries.len())));
        }
    }

    #[test]
    fn a_rank_plan_has_segments_index_manifest_and_config() {
        let dir = Path::new(SLICE);
        if !dir.join("config.json").is_file() { return; }
        let c = census_of(dir).unwrap();
        let o = plan_opts(&c, 2, "inner").unwrap();
        let sel = plan::select(&c, &o, 1).unwrap();
        let (files, man) = rank_files(&c, dir, &o, &sel).unwrap();
        let names: Vec<&str> = files.iter().map(|f| f.logical.as_str()).collect();
        for want in ["seg/L02.safetensors", "seg/global.safetensors", "model.safetensors.index.json", "gb10_shard.json", MANIFEST_FILE, "config.json", "tokenizer.json"] {
            assert!(names.contains(&want), "{want} in {names:?}");
        }
        assert_eq!((man.world, man.rank, man.engram_heads.clone()), (2, 1, vec![4, 5, 6, 7]));
        assert!(!names.iter().any(|n| n.starts_with("engram/")), "engram files are added by plan_rank, not rank_files");
        // rank 1 holds no embedding at TP2
        assert!(!sel.holds.iter().any(|h| c.entries[h.entry as usize].name == "embed.weight"));
    }
}
