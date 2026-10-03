//! B32 (owner brief 2026-10-02): the per-rank SHARD PLAN of a Qwen3.8-Flash-Next model for TP shard shipping
//! (PLAN/B32_SHARD_SHIPPING_DESIGN.md). `shard_plan(dir, format, world, rank, deal)` returns the files of rank
//! `rank`'s shard dir; the head's shipper (`cluster.rs`) turns them into content-addressed blobs, the node
//! assembles them, and the loaders read the assembled dir. The plan decides "which tensors / which bytes a rank
//! needs" with the LOADERS' OWN functions — `xtp::ep_layer_owners_kind` (EXL3 expert deal) and
//! `gpu::qwen_load_shard_op` + `gpu::col_band` (NVFP4 stacked-expert bands) — so shipper and loader cannot drift.
//!
//! A planned file is a whole original file, a SEGMENT (a valid safetensors file synthesized from byte ranges of
//! the source files: header + the pieces' bytes, streamed — no second copy on the head's disk), or a small
//! generated file (the rank's `model.safetensors.index.json`, `gb10_shard.json`).

use anyhow::{bail, Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Bumped whenever a plan's layout changes (it is part of every synthesized segment's hash-cache key).
pub const PLAN_VERSION: u32 = 1;
/// The per-rank manifest file every shard dir carries (format, world, rank, deal, pre-sliced tensors).
pub const SHARD_FILE: &str = "gb10_shard.json";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShardFormat { Exl3, Nvfp4 }

impl ShardFormat {
    pub fn name(self) -> &'static str { match self { ShardFormat::Exl3 => "exl3", ShardFormat::Nvfp4 => "nvfp4" } }
}

/// One tensor of a segment: `len` bytes at absolute offset `off` of `src`, stored as `dtype`/`shape`.
#[derive(Clone, Debug)]
pub struct Piece {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub src: PathBuf,
    pub off: u64,
    pub len: u64,
}

#[derive(Clone, Debug)]
pub enum PlanSrc {
    Whole(PathBuf),
    Segment(Vec<Piece>),
    Bytes(Vec<u8>),
}

#[derive(Clone, Debug)]
pub struct PlanFile {
    pub logical: String,
    pub src: PlanSrc,
}

#[derive(Clone, Debug)]
pub struct RankPlan {
    pub format: ShardFormat,
    pub world: usize,
    pub rank: usize,
    pub deal: String,
    pub files: Vec<PlanFile>,
    /// NVFP4: tensors shipped already sliced to this rank's band (the loader skips their slice op).
    pub presliced: Vec<String>,
}

/// The contents of `gb10_shard.json`.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq)]
pub struct ShardInfo {
    pub plan_version: u32,
    pub format: String,
    pub world: usize,
    pub rank: usize,
    pub deal: String,
    pub source_model: String,
    #[serde(default)]
    pub presliced: Vec<String>,
}

impl RankPlan {
    /// Every tensor name the plan ships (segment tensors + the tensors inside shipped whole safetensors files).
    pub fn tensors(&self) -> Result<BTreeSet<String>> {
        let mut out = BTreeSet::new();
        for f in &self.files {
            match &f.src {
                PlanSrc::Segment(ps) => out.extend(ps.iter().map(|p| p.name.clone())),
                PlanSrc::Whole(p) if f.logical.ends_with(".safetensors") => {
                    out.extend(crate::exl3::shard_header(&p.to_string_lossy())?.into_keys());
                }
                _ => {}
            }
        }
        Ok(out)
    }
    /// The byte size of a planned file.
    pub fn file_size(f: &PlanFile) -> Result<u64> {
        Ok(match &f.src {
            PlanSrc::Whole(p) => std::fs::metadata(p).with_context(|| format!("stat {}", p.display()))?.len(),
            PlanSrc::Segment(ps) => segment_header(ps)?.len() as u64 + ps.iter().map(|p| p.len).sum::<u64>(),
            PlanSrc::Bytes(b) => b.len() as u64,
        })
    }
    pub fn total_bytes(&self) -> Result<u64> {
        self.files.iter().map(Self::file_size).sum()
    }
}

/// The safetensors header (8-byte LE length + JSON, space-padded to a multiple of 8) of a segment whose
/// tensors are laid out back to back in `pieces` order.
pub fn segment_header(pieces: &[Piece]) -> Result<Vec<u8>> {
    let mut m = serde_json::Map::new();
    let mut at = 0u64;
    for p in pieces {
        m.insert(p.name.clone(), serde_json::json!({"dtype": p.dtype, "shape": p.shape, "data_offsets": [at, at + p.len]}));
        at += p.len;
    }
    m.insert("__metadata__".into(), serde_json::json!({"format": "pt", "gb10_segment": PLAN_VERSION.to_string()}));
    let mut j = serde_json::to_vec(&serde_json::Value::Object(m))?;
    while j.len() % 8 != 0 { j.push(b' '); }
    let mut out = (j.len() as u64).to_le_bytes().to_vec();
    out.extend(j);
    Ok(out)
}

/// Stream a segment (header + every piece's bytes, read straight from its source file) into `w`.
pub fn write_segment(pieces: &[Piece], w: &mut impl Write) -> Result<()> {
    w.write_all(&segment_header(pieces)?)?;
    let mut buf = vec![0u8; 1 << 20];
    let mut cur: Option<(PathBuf, std::fs::File)> = None;
    for p in pieces {
        if cur.as_ref().map_or(true, |(cp, _)| cp != &p.src) {
            cur = Some((p.src.clone(), std::fs::File::open(&p.src).with_context(|| format!("open {}", p.src.display()))?));
        }
        let f = &mut cur.as_mut().unwrap().1;
        f.seek(SeekFrom::Start(p.off))?;
        let mut left = p.len;
        while left > 0 {
            let n = left.min(buf.len() as u64) as usize;
            f.read_exact(&mut buf[..n]).with_context(|| format!("read {} @{} of {}", p.name, p.off, p.src.display()))?;
            w.write_all(&buf[..n])?;
            left -= n as u64;
        }
    }
    Ok(())
}

/// A stable description of a segment's inputs: the plan version, every source file's (canonical path, mtime,
/// size) and every piece. Equal recipes produce byte-identical segments — the hash-cache key of a segment.
pub fn segment_recipe(pieces: &[Piece]) -> Result<String> {
    let mut srcs: BTreeMap<String, String> = BTreeMap::new();
    for p in pieces {
        let k = p.src.to_string_lossy().to_string();
        if !srcs.contains_key(&k) {
            let real = std::fs::canonicalize(&p.src).with_context(|| format!("canonicalize {}", p.src.display()))?;
            let md = std::fs::metadata(&real)?;
            let mt = md.modified()?.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
            srcs.insert(k, format!("{}|{}|{}", real.display(), mt, md.len()));
        }
    }
    let mut s = format!("plan{PLAN_VERSION}\n");
    for p in pieces {
        s += &format!("{}\t{}\t{:?}\t{}\t{}\t{}\n", p.name, p.dtype, p.shape, srcs[&p.src.to_string_lossy().to_string()], p.off, p.len);
    }
    Ok(s)
}

/// Which format a model dir is (None = neither shard-plannable format).
pub fn detect_format(dir: &Path) -> Option<ShardFormat> {
    if crate::exl3::is_exl3_dir(&dir.to_string_lossy()) { return Some(ShardFormat::Exl3); }
    let raw = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let mt = v.get("text_config").and_then(|t| t.get("model_type")).or_else(|| v.get("model_type"))?.as_str()?;
    if mt.starts_with("qwen4_exp") && dir.join("model.safetensors.index.json").is_file() { Some(ShardFormat::Nvfp4) } else { None }
}

/// B32: the format whose per-rank shards the head ALWAYS ships for this model at `world` (owner 2026-10-02: no
/// switch) — a Qwen3.8-Flash-Next (qwen4_exp) EXL3 pack or NVFP4 dir whose routed experts split by world. None = the
/// plan does not cover the model (every other Qwen model, the 27B / 35B EXL3 packs, DSV4, ...): its path is unchanged.
pub fn applies(dir: &Path, world: usize) -> Option<ShardFormat> {
    let fmt = detect_format(dir)?;
    // Qwen3.8-Flash-Next (qwen4_exp) only: the brief's scope — every other model keeps its existing path
    let raw = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let mt = v.get("text_config").and_then(|t| t.get("model_type")).or_else(|| v.get("model_type"))?.as_str()?;
    if !mt.starts_with("qwen4_exp") { return None; }
    let cfg = crate::qwen::Config::from_config_json(&dir.join("config.json").to_string_lossy()).ok()?;
    (world >= 2 && cfg.num_experts > 0 && cfg.num_experts % world == 0).then_some(fmt)
}

fn index_files(dir: &Path) -> Result<(BTreeMap<String, String>, BTreeSet<String>)> {
    let raw = std::fs::read_to_string(dir.join("model.safetensors.index.json"))
        .with_context(|| format!("shard plan: {}/model.safetensors.index.json missing", dir.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw).context("shard plan: index unparsable")?;
    let wm = v["weight_map"].as_object().context("shard plan: index has no weight_map")?;
    let mut map = BTreeMap::new();
    let mut files = BTreeSet::new();
    for (k, f) in wm {
        let f = f.as_str().context("shard plan: weight_map value not a string")?.to_string();
        files.insert(f.clone());
        map.insert(k.clone(), f);
    }
    Ok((map, files))
}

fn stem(f: &str) -> String { f.trim_end_matches(".safetensors").replace('/', "_") }

/// `model.language_model.layers.L.mlp.experts.E.<...>` -> Some((L, E)) (EXL3 per-expert trunk tensors only).
fn exl3_trunk_expert(name: &str) -> Option<(usize, usize)> {
    let rest = name.strip_prefix("model.language_model.layers.")?;
    let (l, rest) = rest.split_once('.')?;
    let rest = rest.strip_prefix("mlp.experts.")?;
    let (e, _) = rest.split_once('.')?;
    Some((l.parse().ok()?, e.parse().ok()?))
}

fn shard_info_bytes(fmt: ShardFormat, world: usize, rank: usize, deal: &str, dir: &Path, presliced: &[String]) -> Result<Vec<u8>> {
    let info = ShardInfo {
        plan_version: PLAN_VERSION, format: fmt.name().into(), world, rank, deal: deal.into(),
        source_model: dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(),
        presliced: presliced.to_vec(),
    };
    Ok(serde_json::to_vec_pretty(&info)?)
}

fn index_bytes(weight_map: &BTreeMap<String, String>, total: u64) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec_pretty(&serde_json::json!({"metadata": {"total_size": total, "gb10_shard": PLAN_VERSION},
                                                     "weight_map": weight_map}))?)
}

/// The plan of rank `rank` of `world` (deal = the EXL3 expert deal; NVFP4 bands are contiguous and ignore it).
pub fn shard_plan(dir: &Path, fmt: ShardFormat, world: usize, rank: usize, deal: &str) -> Result<RankPlan> {
    anyhow::ensure!(world >= 2 && rank < world, "shard plan: rank {rank} of world {world}");
    match fmt {
        ShardFormat::Exl3 => plan_exl3(dir, world, rank, deal),
        ShardFormat::Nvfp4 => plan_nvfp4(dir, world, rank),
    }
}

fn plan_exl3(dir: &Path, world: usize, rank: usize, deal: &str) -> Result<RankPlan> {
    let cfg = crate::qwen::Config::from_config_json(&dir.join("config.json").to_string_lossy())?;
    let (ne, nl) = (cfg.num_experts, cfg.num_layers);
    anyhow::ensure!(ne > 0 && ne % world == 0, "shard plan (exl3): {ne} experts do not divide by world {world}");
    let mut local: Vec<BTreeSet<usize>> = Vec::with_capacity(nl);
    for l in 0..nl {
        let own = crate::exl3_forward::xtp::ep_layer_owners_kind(deal, ne, world, l)?;
        local.push((0..ne).filter(|&e| own[e] == rank).collect());
    }
    let (wmap, files) = index_files(dir)?;
    let whole_st: BTreeSet<&str> = crate::cluster::PACK_SIDECARS.iter().copied().collect();
    let mut out = Vec::new();
    let mut new_map: BTreeMap<String, String> = BTreeMap::new();
    let mut experts: BTreeMap<usize, Vec<Piece>> = BTreeMap::new();
    for f in &files {
        let path = dir.join(f);
        if whole_st.contains(f.as_str()) {
            for (k, v) in &wmap { if v == f { new_map.insert(k.clone(), f.clone()); } }
            out.push(PlanFile { logical: f.clone(), src: PlanSrc::Whole(path) });
            continue;
        }
        let hdr = crate::exl3::shard_header(&path.to_string_lossy())?;
        let mut ents: Vec<(&String, &crate::exl3::HdrEntry)> = hdr.iter().collect();
        ents.sort_by_key(|(_, e)| e.offset);
        let mut common = Vec::new();
        for (name, e) in ents {
            if name.contains(".visual.") || name.starts_with("model.visual") { continue; } // the tower runs on the head
            let piece = Piece { name: name.clone(), dtype: e.dtype.clone(), shape: e.shape.clone(), src: path.clone(),
                                off: e.offset as u64, len: e.n_bytes as u64 };
            match exl3_trunk_expert(name) {
                Some((l, e_id)) => {
                    anyhow::ensure!(l < nl && e_id < ne, "shard plan (exl3): {name} outside the config ({nl} layers, {ne} experts)");
                    if local[l].contains(&e_id) { experts.entry(l).or_default().push(piece); }
                }
                None => common.push(piece),
            }
        }
        if !common.is_empty() {
            let logical = format!("seg/common-{}.safetensors", stem(f));
            for p in &common { new_map.insert(p.name.clone(), logical.clone()); }
            out.push(PlanFile { logical, src: PlanSrc::Segment(common) });
        }
    }
    for (l, ps) in experts {
        let logical = format!("seg/experts-L{l:02}.safetensors");
        for p in &ps { new_map.insert(p.name.clone(), logical.clone()); }
        out.push(PlanFile { logical, src: PlanSrc::Segment(ps) });
    }
    // CF-P1g: a by-name sidecar the index does NOT list (the 4.05 pack's index omits the 39 GB n-gram table and its aux tensors)
    // still goes to every rank whole — each rank computes its own PLE rows; the loader finds it by name.
    for f in crate::cluster::PACK_SIDECARS {
        if !files.contains(f) && dir.join(f).is_file() {
            out.push(PlanFile { logical: f.to_string(), src: PlanSrc::Whole(dir.join(f)) });
        }
    }
    for f in crate::cluster::PACK_READ_FILES {
        if f == "model.safetensors.index.json" { continue; }
        if dir.join(f).is_file() { out.push(PlanFile { logical: f.to_string(), src: PlanSrc::Whole(dir.join(f)) }); }
    }
    let mut plan = RankPlan { format: ShardFormat::Exl3, world, rank, deal: deal.into(), files: out, presliced: Vec::new() };
    let total = plan.total_bytes()?;
    plan.files.push(PlanFile { logical: "model.safetensors.index.json".into(), src: PlanSrc::Bytes(index_bytes(&new_map, total)?) });
    plan.files.push(PlanFile { logical: SHARD_FILE.into(), src: PlanSrc::Bytes(shard_info_bytes(ShardFormat::Exl3, world, rank, deal, dir, &[])?) });
    plan.files.sort_by(|a, b| a.logical.cmp(&b.logical));
    Ok(plan)
}

/// NVFP4: a stacked routed-expert raw tensor (`...mlp.experts.{gate_up,down}_proj.weight_{packed,scale}`) whose
/// load-time op is `Col` — shipped as the rank's contiguous row band (`col_band`, the loader's own).
fn nvfp4_band(name: &str, cfg: &crate::qwen::Config, world: usize) -> bool {
    let base = if let Some(b) = name.strip_suffix(".weight_packed") { b } else if let Some(b) = name.strip_suffix(".weight_scale") { b } else { return false };
    if !base.contains(".mlp.experts.") { return false; }
    matches!(crate::gpu::qwen_load_shard_op(&[format!("{base}.weight")], cfg, true, world), crate::gpu::LoadShardOp::Col)
}

fn plan_nvfp4(dir: &Path, world: usize, rank: usize) -> Result<RankPlan> {
    let cfg = crate::qwen::Config::from_config_json(&dir.join("config.json").to_string_lossy())?;
    let (_wmap, files) = index_files(dir)?;
    let mut out = Vec::new();
    let mut new_map: BTreeMap<String, String> = BTreeMap::new();
    let mut presliced = Vec::new();
    for f in &files {
        let path = dir.join(f);
        let hdr = crate::exl3::shard_header(&path.to_string_lossy())?;
        let mut ents: Vec<(&String, &crate::exl3::HdrEntry)> = hdr.iter().collect();
        ents.sort_by_key(|(_, e)| e.offset);
        let (mut common, mut band) = (Vec::new(), Vec::new());
        // a banded tensor's SIBLINGS (its weight_global_scale, ...) travel in the same segment: the NVFP4 loader
        // assembles a quantized tensor's parts from ONE shard file
        let banded_bases: BTreeSet<String> = ents.iter().filter(|(n, _)| nvfp4_band(n, &cfg, world))
            .filter_map(|(n, _)| n.rsplit_once(".weight_").map(|(b, _)| b.to_string())).collect();
        for (name, e) in ents {
            if name.contains(".visual.") || name.starts_with("model.visual") { continue; } // the tower runs on the head
            let sibling = name.rsplit_once(".weight_").map_or(false, |(b, _)| banded_bases.contains(b));
            let mut piece = Piece { name: name.clone(), dtype: e.dtype.clone(), shape: e.shape.clone(), src: path.clone(),
                                    off: e.offset as u64, len: e.n_bytes as u64 };
            if nvfp4_band(name, &cfg, world) {
                anyhow::ensure!(e.shape.len() == 2, "shard plan (nvfp4): {name} is not 2-D ({:?})", e.shape);
                let (m, cols) = (e.shape[0], e.shape[1]);
                anyhow::ensure!(m % world == 0 && (m / world) % 16 == 0, "shard plan (nvfp4): {name} rows {m} do not band by {world}");
                let row = (e.n_bytes / m) as u64;
                anyhow::ensure!(row * m as u64 == e.n_bytes as u64 && row == cols as u64,
                                "shard plan (nvfp4): {name} is not a 1-byte-per-element [m, k] tensor");
                let (r0, take) = crate::gpu::col_band(m, rank, world);
                piece.off += r0 as u64 * row;
                piece.len = take as u64 * row;
                piece.shape = vec![take, cols];
                presliced.push(name.clone());
                band.push(piece);
            } else if sibling {
                band.push(piece); // whole (e.g. the [1] global scale), next to its banded parts
            } else {
                common.push(piece);
            }
        }
        for (kind, ps) in [("common", common), ("experts", band)] {
            if ps.is_empty() { continue; }
            let logical = format!("seg/{kind}-{}.safetensors", stem(f));
            for p in &ps { new_map.insert(p.name.clone(), logical.clone()); }
            out.push(PlanFile { logical, src: PlanSrc::Segment(ps) });
        }
    }
    // every other top-level file (config, tokenizer, chat template, the PLE table + its json, ...) ships whole
    for e in std::fs::read_dir(dir)? {
        let e = e?;
        let n = e.file_name().to_string_lossy().to_string();
        if n.starts_with('.') || n.ends_with(".safetensors") || n == "model.safetensors.index.json" || !e.path().is_file() { continue; }
        out.push(PlanFile { logical: n, src: PlanSrc::Whole(e.path()) });
    }
    let mut plan = RankPlan { format: ShardFormat::Nvfp4, world, rank, deal: "contig".into(), files: out, presliced: presliced.clone() };
    let total = plan.total_bytes()?;
    plan.files.push(PlanFile { logical: "model.safetensors.index.json".into(), src: PlanSrc::Bytes(index_bytes(&new_map, total)?) });
    plan.files.push(PlanFile { logical: SHARD_FILE.into(), src: PlanSrc::Bytes(shard_info_bytes(ShardFormat::Nvfp4, world, rank, "contig", dir, &presliced)?) });
    plan.files.sort_by(|a, b| a.logical.cmp(&b.logical));
    Ok(plan)
}

/// A loader's view of a shard dir: `Some(info)` when `dir` is a B32 shard dir (it carries `gb10_shard.json`).
/// The loader must then hold the same (format, world, rank) or refuse.
pub fn read_shard_info(dir: &Path) -> Result<Option<ShardInfo>> {
    let p = dir.join(SHARD_FILE);
    if !p.is_file() { return Ok(None); }
    let info: ShardInfo = serde_json::from_slice(&std::fs::read(&p)?).with_context(|| format!("{} unparsable", p.display()))?;
    if info.plan_version != PLAN_VERSION {
        bail!("{}: plan version {} != this binary's {PLAN_VERSION} — re-ship the shards", p.display(), info.plan_version);
    }
    Ok(Some(info))
}

/// A loader's guard on a shard dir: the attach's (format, world, rank) must be the dir's.
pub fn check_shard_dir(dir: &Path, fmt: ShardFormat, world: usize, rank: usize) -> Result<Option<ShardInfo>> {
    let Some(info) = read_shard_info(dir)? else { return Ok(None) };
    anyhow::ensure!(info.format == fmt.name() && info.world == world && info.rank == rank,
        "{} is a {} shard for rank {} of world {}, but this {} load is rank {rank} of world {world} — wrong shard dir",
        dir.display(), info.format, info.rank, info.world, fmt.name());
    Ok(Some(info))
}

/// Is `name` a vision-tower tensor (never shipped: only the head runs the tower)?
pub fn is_visual(name: &str) -> bool { name.contains(".visual.") || name.starts_with("model.visual") }

/// B32 `--probe-shard-plan` (no GPU): for every (world, deal) asked, plan every rank and check
///   1. coverage: every non-visual source tensor is shipped to some rank; every routed-expert tensor (EXL3
///      per-expert, NVFP4 band) to exactly the rank(s) the loader's own map gives it; the shared rest to every rank;
///   2. the shard dir a node would assemble, built SPARSE in `tmp` (segments = their real header + a hole), opens:
///      EXL3 — `Exl3Pack::open` succeeds and holds every module the rank's loader asks for (all non-expert
///      modules + its experts' gate/up/down quadruples); NVFP4 — every pre-sliced tensor has the [m/W, k] shape
///      and shares ONE file with every part of its quantized tensor (the loader's grouping rule);
///   3. bytes: the two smallest segments of rank 1 are synthesized for real and each tensor's bytes compared with
///      its source range.
pub fn probe(dir: &Path, worlds: &[usize], deals: &[String], tmp: &Path) -> Result<()> {
    let fmt = detect_format(dir).context("probe-shard-plan: not an EXL3 pack nor a qwen4_exp NVFP4 model")?;
    let cfg = crate::qwen::Config::from_config_json(&dir.join("config.json").to_string_lossy())?;
    let (wmap, files) = index_files(dir)?;
    let mut src_names: BTreeSet<String> = BTreeSet::new();
    for f in &files {
        src_names.extend(crate::exl3::shard_header(&dir.join(f).to_string_lossy())?.into_keys().filter(|n| !is_visual(n)));
    }
    let _ = wmap;
    let deals: Vec<String> = if fmt == ShardFormat::Nvfp4 { vec!["contig".into()] } else { deals.to_vec() };
    let mut fails = 0usize;
    for &world in worlds {
        for deal in &deals {
            let t0 = std::time::Instant::now();
            let plans: Vec<RankPlan> = match (0..world).map(|r| shard_plan(dir, fmt, world, r, deal)).collect() {
                Ok(p) => p,
                Err(e) => { println!("PLAN {} w{world} {deal}: REFUSED — {e:#}", fmt.name()); continue; }
            };
            // 1. coverage
            let sets: Vec<BTreeSet<String>> = plans.iter().map(|p| p.tensors()).collect::<Result<_>>()?;
            let union: BTreeSet<String> = sets.iter().flatten().cloned().collect();
            let missing: Vec<&String> = src_names.iter().filter(|n| !union.contains(*n)).collect();
            let mut bad_owner = 0usize;
            for n in &src_names {
                let holders: Vec<usize> = (0..world).filter(|&r| sets[r].contains(n)).collect();
                let want: Vec<usize> = match fmt {
                    ShardFormat::Exl3 => match exl3_trunk_expert(n) {
                        Some((l, e)) => vec![crate::exl3_forward::xtp::ep_layer_owners_kind(deal, cfg.num_experts, world, l)?[e]],
                        None => (0..world).collect(),
                    },
                    ShardFormat::Nvfp4 => (0..world).collect(), // bands: every rank holds its own slice
                };
                if holders != want { bad_owner += 1; if bad_owner <= 3 { println!("  owner mismatch {n}: holders {holders:?} want {want:?}"); } }
            }
            // 2. sparse shard dirs
            let mut open_ok = 0usize;
            for p in &plans {
                let d = tmp.join(format!("{}-w{world}-r{}-{deal}", fmt.name(), p.rank));
                let _ = std::fs::remove_dir_all(&d);
                for f in &p.files {
                    let path = d.join(&f.logical);
                    std::fs::create_dir_all(path.parent().unwrap())?;
                    match &f.src {
                        PlanSrc::Whole(src) => std::os::unix::fs::symlink(src, &path)?,
                        PlanSrc::Bytes(b) => std::fs::write(&path, b)?,
                        PlanSrc::Segment(ps) => {
                            let mut fh = std::fs::File::create(&path)?;
                            fh.write_all(&segment_header(ps)?)?;
                            fh.set_len(RankPlan::file_size(f)?)?; // sparse: the bytes are never read here
                        }
                    }
                }
                match fmt {
                    ShardFormat::Exl3 => {
                        let pack = crate::exl3::Exl3Pack::open(&d.to_string_lossy())?;
                        let have: BTreeSet<&str> = pack.modules.iter().map(|m| m.name.as_str()).collect();
                        let own: Vec<Vec<usize>> = (0..cfg.num_layers).map(|l| crate::exl3_forward::xtp::ep_layer_owners_kind(deal, cfg.num_experts, world, l)).collect::<Result<_>>()?;
                        let mut need_missing = 0usize;
                        for l in 0..cfg.num_layers {
                            for e in (0..cfg.num_experts).filter(|&e| own[l][e] == p.rank) {
                                for proj in ["gate_proj", "up_proj", "down_proj"] {
                                    let m = format!("model.language_model.layers.{l}.mlp.experts.{e}.{proj}");
                                    if pack.modules.iter().all(|x| x.name != m) && !have.contains(m.as_str()) { need_missing += 1; }
                                }
                            }
                        }
                        let foreign = pack.modules.iter().filter(|m| exl3_trunk_expert(&format!("{}.trellis", m.name))
                            .map_or(false, |(l, e)| own[l][e] != p.rank)).count();
                        let info = check_shard_dir(&d, fmt, world, p.rank)?.context("no gb10_shard.json")?;
                        let ok = need_missing == 0 && foreign == 0 && info.rank == p.rank;
                        if ok { open_ok += 1; } else { println!("  rank {}: {need_missing} needed modules missing, {foreign} foreign expert modules", p.rank); }
                    }
                    ShardFormat::Nvfp4 => {
                        let mut bad = 0usize;
                        for f in p.files.iter().filter(|f| matches!(f.src, PlanSrc::Segment(_))) {
                            for (n, e) in crate::exl3::shard_header(&d.join(&f.logical).to_string_lossy())? {
                                if p.presliced.contains(&n) {
                                    let full = crate::exl3::shard_header(&dir.join(&wmap_file(dir, &n)?).to_string_lossy())?[&n].shape.clone();
                                    if e.shape != vec![full[0] / world, full[1]] { bad += 1; }
                                }
                            }
                        }
                        // the loader assembles a quantized tensor's parts (packed / scale / global scale) from ONE file
                        let mut file_of: BTreeMap<String, &str> = BTreeMap::new();
                        for f in &p.files {
                            if let PlanSrc::Segment(ps) = &f.src { for pc in ps { file_of.insert(pc.name.clone(), f.logical.as_str()); } }
                        }
                        for n in &p.presliced {
                            let base = n.rsplit_once(".weight_").map(|(b, _)| b).unwrap_or(n);
                            let home = file_of.get(n).copied();
                            for (m, fl) in &file_of {
                                if m.rsplit_once(".weight_").map_or(false, |(b, _)| b == base) && Some(*fl) != home { bad += 1; }
                            }
                        }
                        let info = check_shard_dir(&d, fmt, world, p.rank)?.context("no gb10_shard.json")?;
                        if bad == 0 && info.presliced.len() == p.presliced.len() && !p.presliced.is_empty() { open_ok += 1; }
                        else { println!("  rank {}: {bad} pre-sliced shapes wrong ({} pre-sliced)", p.rank, p.presliced.len()); }
                    }
                }
            }
            let sizes: Vec<String> = plans.iter().map(|p| p.total_bytes().map(|b| format!("{:.2}", b as f64 / (1u64 << 30) as f64))).collect::<Result<_>>()?;
            let ok = missing.is_empty() && bad_owner == 0 && open_ok == world;
            if !ok { fails += 1; }
            println!("PLAN {} w{world} deal={deal}: {} — {} source tensors, missing {}, owner mismatches {bad_owner}, shard dirs open {open_ok}/{world}; GiB per rank [{}] ({:.1} s)",
                     fmt.name(), if ok { "PASS" } else { "FAIL" }, src_names.len(), missing.len(), sizes.join(", "), t0.elapsed().as_secs_f64());
        }
    }
    // 3. bytes: synthesize rank 1's two smallest segments of the first (world, deal) and compare
    let p = shard_plan(dir, fmt, worlds[0], 1, &deals[0])?;
    let mut segs: Vec<&PlanFile> = p.files.iter().filter(|f| matches!(f.src, PlanSrc::Segment(_))).collect();
    segs.sort_by_key(|f| RankPlan::file_size(f).unwrap_or(u64::MAX));
    for f in segs.iter().take(2) {
        let PlanSrc::Segment(ps) = &f.src else { continue };
        let path = tmp.join("roundtrip.safetensors");
        write_segment(ps, &mut std::io::BufWriter::new(std::fs::File::create(&path)?))?;
        let hdr = crate::exl3::shard_header(&path.to_string_lossy())?;
        let mut bad = 0usize;
        for pc in ps {
            let got = &hdr[&pc.name];
            let mut a = vec![0u8; pc.len as usize];
            let mut fa = std::fs::File::open(&path)?; fa.seek(SeekFrom::Start(got.offset as u64))?; fa.read_exact(&mut a)?;
            let mut b = vec![0u8; pc.len as usize];
            let mut fb = std::fs::File::open(&pc.src)?; fb.seek(SeekFrom::Start(pc.off))?; fb.read_exact(&mut b)?;
            if a != b || got.shape != pc.shape || got.dtype != pc.dtype { bad += 1; }
        }
        println!("BYTES {}: {} tensors, {} B — {}", f.logical, ps.len(), RankPlan::file_size(f)?, if bad == 0 { "IDENTICAL to the source ranges".to_string() } else { format!("{bad} DIFFER"); fails += 1; String::new() });
        let _ = std::fs::remove_file(&path);
    }
    if fails > 0 { bail!("probe-shard-plan: {fails} check(s) FAILED"); }
    println!("PROBE_SHARD_PLAN_PASS");
    Ok(())
}

fn wmap_file(dir: &Path, name: &str) -> Result<String> {
    let (m, _) = index_files(dir)?;
    m.get(name).cloned().with_context(|| format!("{name} not in the source index"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shard shipping is unconditional but scoped: Qwen3.8-Flash-Next ships shards, every other model keeps its path.
    /// (Reads the lab's model dirs; skipped where they are absent.)
    #[test]
    fn applies_only_to_flash_next() {
        let m = |d: &str| Path::new(&std::env::var("HOME").unwrap_or_default()).join("models").join(d);
        if !m("Qwen3.8-Flash-Next-exl3-3.05bpw").is_dir() { return; }
        assert_eq!(applies(&m("Qwen3.8-Flash-Next-exl3-3.05bpw"), 2), Some(ShardFormat::Exl3));
        assert_eq!(applies(&m("Qwen3.8-Flash-Next-exl3-3.05bpw"), 4), Some(ShardFormat::Exl3));
        for other in ["Qwen3.6-35B-A3B-exl3-3.5bpw", "Qwen3.8-27B-exl3-3.05bpw", "3.8-27b-nvfp4-full-all"] {
            if m(other).is_dir() { assert_eq!(applies(&m(other), 2), None, "{other} must keep its own path"); }
        }
    }
}
