//! S5b: the Naive-N0.5 per-rank SHARD PLAN for the engine's B32 shard shipping (`shard_plan::shard_plan`
//! dispatches here for `ShardFormat::Naive`). The head ships rank r of W:
//!   * its experts only (e % W == r, the `interleave` deal — the same rule as `naive::shard::Shard`): the
//!     per-expert quadruples whole, one segment per layer;
//!   * the row-sliced quads PRE-SLICED (o_proj of every layer, the dense down_proj): trellis rows
//!     [r*k/W, (r+1)*k/W) and the matching suh range are contiguous byte ranges of the source tensors
//!     (a trellis is [k/16][n/16][words]); svh / mul1 whole. Listed in `presliced`, the loader sees k = k/W;
//!   * everything else whole (the column-sliced quads q/k/v/gate/up/lm_head are strided in the file and
//!     are sliced at load as before; norms, router, indexer, embeddings, sinks are replicated).
//! Per rank at W = 2: ~69 GB of the 137 GB pack. The node's assembled dir opens with `NaivePack::open`
//! (shard-aware through `gb10_shard.json`).
use anyhow::{ensure, Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::shard_plan::{Piece, PlanFile, PlanSrc, RankPlan, ShardFormat, SHARD_FILE};
use super::config::NaiveConfig;

/// The expert-parallel deal (the default): expert e lives on rank e % world.
pub const DEAL: &str = "interleave";
/// Expert TENSOR parallelism (`--naive-moe-shard tp`, 2026-10-20): every rank holds 1/W of EVERY expert — gate/up
/// columns [r*mi/W, (r+1)*mi/W) (a strided gather of the trellis tile rows + the svh range; suh whole) and the
/// matching down_proj rows (contiguous, like o_proj). Same bytes per rank as the deal above, zero per-layer
/// expert-count skew (the 2026-10-20 TP4 census: the MoE reduce waited 81.5 us mean vs 17.4 us after attention).
pub const DEAL_TP: &str = "tp";

/// `--naive-moe-shard auto|ep|tp` -> the deal name the head ships under (and the shard dir's cache id carries).
pub fn deal(world: usize) -> String {
    if moe_tp_resolve(crate::opts::var(crate::opt!("naive-moe-shard")).ok().as_deref(), world) { DEAL_TP.into() } else { DEAL.into() }
}

/// Pure: `tp` selects expert tensor parallelism, `ep` the interleaved expert deal; `auto` (the default, owner
/// decision 2026-10-21) = tp at TP=4, ep at TP=2 (expert TP never ran at TP=2: its ~68 GB shard did not fit .14's disk).
pub fn moe_tp_resolve(raw: Option<&str>, world: usize) -> bool {
    match raw.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        Some("tp") => true,
        Some("ep") => false,
        _ => world >= 4,
    }
}

pub fn plan(dir: &Path, world: usize, rank: usize) -> Result<RankPlan> { plan_deal(dir, world, rank, DEAL) }

pub fn plan_deal(dir: &Path, world: usize, rank: usize, deal: &str) -> Result<RankPlan> {
    ensure!(world >= 2 && rank < world, "naive shard plan: rank {rank} of world {world}");
    ensure!(deal == DEAL || deal == DEAL_TP, "naive shard plan: unknown deal {deal:?} ({DEAL} | {DEAL_TP})");
    let etp = deal == DEAL_TP;
    let cfg = NaiveConfig::from_dir(&dir.to_string_lossy())?;
    ensure!(cfg.num_experts % world == 0, "naive shard plan: {} experts do not divide by world {world}", cfg.num_experts);
    let (nh, vd) = (cfg.num_heads, cfg.v_head_dim);
    ensure!(nh % world == 0 && (nh / world * vd) % 128 == 0, "naive shard plan: o_proj rows do not split 128-aligned at world {world}");
    ensure!(cfg.intermediate_size % world == 0 && (cfg.intermediate_size / world) % 128 == 0, "naive shard plan: dense intermediate does not split at world {world}");
    ensure!(!etp || (cfg.moe_intermediate_size % world == 0 && (cfg.moe_intermediate_size / world) % 128 == 0),
            "naive shard plan: the expert intermediate {} does not split 128-aligned at world {world} (expert TP)", cfg.moe_intermediate_size);
    let raw = std::fs::read_to_string(dir.join("model.safetensors.index.json")).context("naive shard plan: index missing")?;
    let v: serde_json::Value = serde_json::from_str(&raw).context("index unparsable")?;
    let wm = v["weight_map"].as_object().context("index has no weight_map")?;
    let mut files: BTreeSet<String> = BTreeSet::new();
    for f in wm.values() { files.insert(f.as_str().context("weight_map value")?.to_string()); }
    let mut out = Vec::new();
    let mut new_map: BTreeMap<String, String> = BTreeMap::new();
    let mut presliced: Vec<String> = Vec::new();
    let mut experts: BTreeMap<usize, Vec<Piece>> = BTreeMap::new();
    for f in &files {
        let path = dir.join(f);
        let hdr = crate::exl3::shard_header(&path.to_string_lossy())?;
        let mut ents: Vec<(&String, &crate::exl3::HdrEntry)> = hdr.iter().collect();
        ents.sort_by_key(|(_, e)| e.offset);
        let mut common = Vec::new();
        let stem = f.trim_end_matches(".safetensors").replace('/', "_");
        for (name, e) in ents {
            let mut piece = Piece { rows: 1, stride: 0, name: name.clone(), dtype: e.dtype.clone(), shape: e.shape.clone(), src: path.clone(), off: e.offset as u64, len: e.n_bytes as u64 };
            if let Some((l, ex)) = expert_of(name) {
                ensure!(l < cfg.num_layers && ex < cfg.num_experts, "naive shard plan: {name} outside the config");
                if etp {
                    slice_expert_piece(&mut piece, name, cfg.moe_intermediate_size, world, rank)?;
                    experts.entry(l).or_default().push(piece);
                } else if ex % world == rank { experts.entry(l).or_default().push(piece); }
                continue;
            }
            // the row-sliced quads: o_proj (rows = this rank's heads x vd), dense down_proj (rows = its intermediate slice)
            if let Some((module, suffix)) = row_sliced(name, &cfg) {
                let k_full = match suffix { _ if module.ends_with("self_attn.o_proj") => nh * vd, _ => cfg.intermediate_size };
                let kl = k_full / world;
                let (a, b) = (rank * kl, (rank + 1) * kl);
                match suffix {
                    "trellis" => {
                        ensure!(e.shape.len() == 3 && e.shape[0] * 16 == k_full, "{name}: trellis shape {:?} vs k {k_full}", e.shape);
                        let row_bytes = (e.shape[1] * e.shape[2] * 2) as u64;   // one k-tile row: nb x words x u16
                        piece.off = e.offset as u64 + (a / 16) as u64 * row_bytes;
                        piece.len = ((b - a) / 16) as u64 * row_bytes;
                        piece.shape = vec![kl / 16, e.shape[1], e.shape[2]];
                    }
                    "suh" => {
                        ensure!(e.shape == vec![k_full], "{name}: suh shape {:?} vs k {k_full}", e.shape);
                        piece.off = e.offset as u64 + (a * 2) as u64;
                        piece.len = (kl * 2) as u64;
                        piece.shape = vec![kl];
                    }
                    _ => {}
                }
                if suffix == "trellis" { presliced.push(module.to_string()); }
            }
            common.push(piece);
        }
        if !common.is_empty() {
            let logical = format!("seg/common-{stem}.safetensors");
            for p in &common { new_map.insert(p.name.clone(), logical.clone()); }
            out.push(PlanFile { logical, src: PlanSrc::Segment(common) });
        }
    }
    for (l, ps) in experts {
        let logical = format!("seg/experts-L{l:02}.safetensors");
        for p in &ps { new_map.insert(p.name.clone(), logical.clone()); }
        out.push(PlanFile { logical, src: PlanSrc::Segment(ps) });
    }
    for f in crate::cluster::PACK_READ_FILES {
        if f == "model.safetensors.index.json" { continue; }
        if dir.join(f).is_file() { out.push(PlanFile { logical: f.to_string(), src: PlanSrc::Whole(dir.join(f)) }); }
    }
    let mut plan = RankPlan { format: ShardFormat::Naive, world, rank, deal: deal.into(), files: out, presliced: presliced.clone() };
    let total = plan.total_bytes()?;
    let index = serde_json::to_vec_pretty(&serde_json::json!({"metadata": {"total_size": total, "gb10_shard": crate::shard_plan::PLAN_VERSION}, "weight_map": new_map}))?;
    plan.files.push(PlanFile { logical: "model.safetensors.index.json".into(), src: PlanSrc::Bytes(index) });
    let info = crate::shard_plan::ShardInfo {
        plan_version: crate::shard_plan::PLAN_VERSION, format: ShardFormat::Naive.name().into(), world, rank, deal: deal.into(),
        source_model: dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(), presliced,
    };
    plan.files.push(PlanFile { logical: SHARD_FILE.into(), src: PlanSrc::Bytes(serde_json::to_vec_pretty(&info)?) });
    plan.files.sort_by(|a, b| a.logical.cmp(&b.logical));
    Ok(plan)
}

/// Expert TP: narrow one expert tensor's piece to rank `rank`'s slice. gate/up (k = h, n = mi): the trellis
/// `[h/16][mi/16][words]` keeps tile columns [r*nb/W, (r+1)*nb/W) of every tile row (a strided gather), svh the
/// matching output range, suh whole. down (k = mi, n = h): trellis tile rows [r*kb/W, (r+1)*kb/W) (contiguous) and
/// the matching suh range, svh whole. mul1 is whole everywhere.
fn slice_expert_piece(piece: &mut Piece, name: &str, mi: usize, world: usize, rank: usize) -> Result<()> {
    let (module, part) = name.rsplit_once('.').context("expert tensor name")?;
    let ml = mi / world;
    let (a, b) = (rank * ml, (rank + 1) * ml);
    let gate_up = module.ends_with(".gate_proj") || module.ends_with(".up_proj");
    let down = module.ends_with(".down_proj");
    ensure!(gate_up || down, "{name}: not an expert projection");
    match (gate_up, part) {
        (true, "trellis") => {
            ensure!(piece.shape.len() == 3 && piece.shape[1] * 16 == mi, "{name}: trellis shape {:?} vs n {mi}", piece.shape);
            let (kb, nb, words) = (piece.shape[0], piece.shape[1], piece.shape[2]);
            let tile = (words * 2) as u64;
            let row = nb as u64 * tile;                       // one tile row, every column
            let cols = (ml / 16) as u64;
            piece.off += (a / 16) as u64 * tile;
            piece.rows = kb as u64;
            piece.stride = row;
            piece.len = kb as u64 * cols * tile;
            piece.shape = vec![kb, ml / 16, words];
        }
        (true, "svh") => {
            ensure!(piece.shape == vec![mi], "{name}: svh shape {:?} vs n {mi}", piece.shape);
            piece.off += (a * 2) as u64; piece.len = (ml * 2) as u64; piece.shape = vec![ml];
        }
        (false, "trellis") => {
            ensure!(piece.shape.len() == 3 && piece.shape[0] * 16 == mi, "{name}: trellis shape {:?} vs k {mi}", piece.shape);
            let row = (piece.shape[1] * piece.shape[2] * 2) as u64;
            piece.off += (a / 16) as u64 * row;
            piece.len = ((b - a) / 16) as u64 * row;
            piece.shape = vec![ml / 16, piece.shape[1], piece.shape[2]];
        }
        (false, "suh") => {
            ensure!(piece.shape == vec![mi], "{name}: suh shape {:?} vs k {mi}", piece.shape);
            piece.off += (a * 2) as u64; piece.len = (ml * 2) as u64; piece.shape = vec![ml];
        }
        _ => {}   // gate/up suh, down svh, every mul1: whole
    }
    Ok(())
}

/// `model.layers.L.mlp.experts.E.<proj>.<part>` -> (L, E)
pub fn expert_of(name: &str) -> Option<(usize, usize)> {
    let rest = name.strip_prefix("model.layers.")?;
    let (l, rest) = rest.split_once('.')?;
    let rest = rest.strip_prefix("mlp.experts.")?;
    let (e, _) = rest.split_once('.')?;
    Some((l.parse().ok()?, e.parse().ok()?))
}

/// A tensor of a row-sliced quad: (module, part) for `model.layers.L.self_attn.o_proj.<part>` and the
/// dense layers' `model.layers.L.mlp.down_proj.<part>`.
fn row_sliced<'a>(name: &'a str, cfg: &NaiveConfig) -> Option<(&'a str, &'a str)> {
    let (module, part) = name.rsplit_once('.')?;
    if !matches!(part, "trellis" | "suh" | "svh" | "mul1") { return None; }
    if module.ends_with(".self_attn.o_proj") { return Some((module, part)); }
    if module.ends_with(".mlp.down_proj") {
        let l: usize = module.strip_prefix("model.layers.")?.split_once('.')?.0.parse().ok()?;
        if !cfg.moe[l] { return Some((module, part)); }
    }
    None
}

#[cfg(test)]
mod tests {
    /// Expert TP: a gate/up trellis slice is the strided gather of tile columns [r*nb/W, (r+1)*nb/W) of every tile
    /// row, down the contiguous tile rows; the W slices of a tensor tile its bytes exactly once.
    #[test]
    fn expert_tp_slices_tile_each_tensor_once() {
        use crate::shard_plan::Piece;
        // a synthetic geometry with the served shapes: h 4096, mi 2048, 56 words per tile
        let (h, mi, words, w) = (4096usize, 2048usize, 56usize, 4usize);
        let mk = |shape: Vec<usize>, len: u64| Piece { name: String::new(), dtype: "I16".into(), shape, src: "/dev/null".into(), off: 1000, len, rows: 1, stride: 0 };
        let tile = (words * 2) as u64;
        let gate_len = (h / 16 * mi / 16) as u64 * tile;
        let mut covered = vec![0u8; gate_len as usize / tile as usize];
        for r in 0..w {
            let mut p = mk(vec![h / 16, mi / 16, words], gate_len);
            p.name = "model.layers.3.mlp.experts.7.gate_proj.trellis".into();
            let nm = p.name.clone(); super::slice_expert_piece(&mut p, &nm, mi, w, r).unwrap();
            assert_eq!(p.rows as usize, h / 16);
            assert_eq!(p.shape, vec![h / 16, mi / 16 / w, words]);
            let row_len = p.len / p.rows;
            for row in 0..p.rows {
                let start = (p.off - 1000 + row * p.stride) / tile;
                for t in 0..row_len / tile { covered[(start + t) as usize] += 1; }
            }
            let mut d = mk(vec![mi / 16, h / 16, words], gate_len);
            d.name = "model.layers.3.mlp.experts.7.down_proj.trellis".into();
            let nm = d.name.clone(); super::slice_expert_piece(&mut d, &nm, mi, w, r).unwrap();
            assert_eq!(d.rows, 1);
            assert_eq!(d.len, gate_len / w as u64);
            assert_eq!(d.off, 1000 + r as u64 * gate_len / w as u64);
        }
        assert!(covered.iter().all(|c| *c == 1), "every gate tile exactly once over the {w} ranks");
    }

    #[test]
    fn plan_covers_the_pack_once() {
        let dir = std::path::Path::new("/mnt/models/NaiveAI/Naive-N0.5-Flash-exl3-3.5bpw-vllm");
        if !dir.join("config.json").is_file() { return; }
        let p0 = super::plan(dir, 2, 0).unwrap();
        let p1 = super::plan(dir, 2, 1).unwrap();
        let (t0, t1) = (p0.tensors().unwrap(), p1.tensors().unwrap());
        // every expert tensor to exactly one rank; the rest to both
        let experts0 = t0.iter().filter(|n| super::expert_of(n).is_some()).count();
        let experts1 = t1.iter().filter(|n| super::expert_of(n).is_some()).count();
        assert_eq!(experts0, experts1);
        assert!(t0.intersection(&t1).all(|n| super::expert_of(n).is_none()));
        // 2026-10-31: the TRANSPLANT_A6 re-export physically duplicates the row-sliced
        // trellis tensors across shard files, so the planner sees each module twice;
        // the invariant is one DISTINCT module per preslice kind (48 o_proj + 1 dense down).
        let mut distinct: Vec<&String> = p0.presliced.iter().collect();
        distinct.sort();
        distinct.dedup();
        assert_eq!(distinct.len(), 48 + 1);
        let b0 = p0.total_bytes().unwrap() as f64 / 1e9;
        assert!(b0 > 60.0 && b0 < 80.0, "rank bytes {b0} GB");
    }
}
