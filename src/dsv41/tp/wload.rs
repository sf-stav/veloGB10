//! DSV4.1 TP (leg TP): a rank's weights on its device, layer by layer, from what the rank holds.
//!
//! * A node holds its SHIPPED shard dir (leg LS: `seg/L<NN>.safetensors` with every tensor already cut to
//!   the rank's slice, `dsv41_shard.json` with the plan options). The head (rank 0) holds the same dir as its
//!   fast-reboot STAGE (`load::stage`, built once under the tp cache); without a stage it reads its slices
//!   straight out of the pack (the same plan, strided reads).
//! * Every module's cut is re-derived with `load::plan::placement` from the manifest's options (it is pure),
//!   and becomes the `Shard` of its `DenseWeight` (`Out` / `In` over the full logical shape; the EXL3
//!   backend detects the slice residency from the tensor shapes). A grouped input cut (Engram `wkv` at TP2:
//!   the rank's heads of every n-gram order) is held as one whole linear over the held K.
//! * What the TP forward needs that the placement leaves whole (attention at TP4, the `hc_*_fn` K slices)
//!   is cut at call time (`fwd.rs`), never here.
//! * DSpark stages (40-42) load on request (leg FW: `TpModel::load_stages`, split like the trunk layers); the vision
//!   tower is NOT loaded.

use super::super::census::classify::{Role, TensorClass as C};
use super::super::census::{Census, LinearFormat};
use super::super::config::Dsv41Config;
use super::super::fwd::dev::{Buf, Gpu};
use super::super::fwd::weights::{LayerW, LinKey};
use super::super::load::plan::{self, Hold, PlanOpts, Placement};
use super::super::weights::{BackendKind, DenseWeight, DevPtr, DevTensor, MoeWeight, Shard, WeightPlan};
use anyhow::{ensure, Context, Result};
use std::collections::HashMap;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

/// Where a rank's bytes come from.
pub struct RankSrc {
    pub dir: PathBuf,
    pub c: Census,
    pub cfg: Dsv41Config,
    pub opts: PlanOpts,
    pub rank: usize,
    /// `Some`: `dir` is the whole pack and these are the rank's holds by census entry (head without a stage)
    holds: Option<HashMap<u32, Hold>>,
    by_key: HashMap<String, usize>,
    plan: WeightPlan,
    /// how the bytes were found (the boot receipt)
    pub how: String,
}

fn align(x: u64) -> u64 { (x + 255) / 256 * 256 }

impl RankSrc {
    /// A shard dir (shipped to a node, or the head's stage): every tensor is already the rank's slice.
    pub fn shard_dir(dir: &Path) -> Result<RankSrc> {
        let m = super::super::load::ship::read_manifest(dir)?;
        let c = Census::build(dir).with_context(|| format!("census of the shard dir {}", dir.display()))?;
        let cfg = c.cfg.clone();
        let opts = m.opts()?;
        let plan = WeightPlan::from_census(&c)?;
        let by_key = plan.linears.iter().enumerate().map(|(i, p)| (p.key.clone(), i)).collect();
        let how = format!("shard dir {} (world {}, rank {}, split {}, attention whole {}, Engram wkv whole {})", dir.display(), m.world, m.rank,
                          m.expert_split, m.attn_whole, m.engram_wkv_whole);
        Ok(RankSrc { dir: dir.to_path_buf(), c, cfg, opts, rank: m.rank, holds: None, by_key, plan, how })
    }

    /// The whole pack, read through the rank's plan (strided reads; the head's fallback when no stage exists).
    pub fn pack(dir: &Path, world: usize, rank: usize, deal: &str) -> Result<RankSrc> {
        let c = Census::build(dir).with_context(|| format!("census of {}", dir.display()))?;
        c.check()?;
        let opts = super::super::load::ship::plan_opts(&c, world, deal)?;
        let sel = plan::select(&c, &opts, rank)?;
        let holds = sel.holds.into_iter().map(|h| (h.entry, h)).collect();
        let cfg = c.cfg.clone();
        let plan = WeightPlan::from_census(&c)?;
        let by_key = plan.linears.iter().enumerate().map(|(i, p)| (p.key.clone(), i)).collect();
        Ok(RankSrc { dir: dir.to_path_buf(), c, cfg, opts, rank, holds: Some(holds), by_key, plan,
                     how: format!("the pack {} read through the rank's plan (no local stage)", dir.display()) })
    }

    pub fn world(&self) -> usize { self.opts.world }

    /// The bytes this rank reads for trunk layer `l` (the load progress total).
    pub fn layer_bytes(&self, l: usize) -> u64 {
        self.c.entries.iter().enumerate().filter(|(_, e)| e.layer == Some(l as u16) && e.class != C::EngramTable)
            .map(|(i, e)| match &self.holds { None => e.bytes, Some(h) => h.get(&(i as u32)).map_or(0, |h| h.bytes) }).sum()
    }

    /// Does this source hold trunk layer `l` (a slice pack holds some)?
    pub fn has_layer(&self, l: usize) -> bool { self.c.entries.iter().any(|e| e.layer == Some(l as u16)) }

    /// The rank's size and shape of entry `i` (None = not held).
    fn held(&self, i: usize) -> Option<(u64, Vec<u64>)> {
        let e = &self.c.entries[i];
        match &self.holds { None => Some((e.bytes, e.shape.clone())), Some(h) => h.get(&(i as u32)).map(|h| (h.bytes, h.shape.clone())) }
    }

    /// The rank's bytes of entry `i` (held: `held(i)` is Some) into `v`, resized to them.
    fn read_into(&self, files: &mut HashMap<u32, std::fs::File>, i: usize, v: &mut Vec<u8>) -> Result<()> {
        let e = &self.c.entries[i];
        if !files.contains_key(&e.file) {
            let p = self.c.dir.join(&self.c.files[e.file as usize]);
            files.insert(e.file, std::fs::File::open(&p).with_context(|| format!("open {}", p.display()))?);
        }
        let f = &files[&e.file];
        match &self.holds {
            None => {
                v.resize(e.bytes as usize, 0);
                f.read_exact_at(v, e.offset).with_context(|| format!("read {}", e.name))?;
            }
            Some(h) => {
                let h = h.get(&(i as u32)).with_context(|| format!("{}: not held by this rank", e.name))?;
                v.resize(h.bytes as usize, 0);
                match &h.slice {
                    None => f.read_exact_at(v, e.offset).with_context(|| format!("read {}", e.name))?,
                    Some(s) => {
                        let (off, rows, stride, len) = s.geometry();
                        let run = (len / rows.max(1)) as usize;
                        for r in 0..rows.max(1) as usize {
                            f.read_exact_at(&mut v[r * run..(r + 1) * run], e.offset + off + r as u64 * stride).with_context(|| format!("read {}", e.name))?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn shard_of(&self, m: &super::super::census::Module) -> Result<(Shard, Option<u64>)> {
        Ok(match plan::placement(&self.cfg, m, &self.opts, self.rank) {
            Placement::Whole | Placement::Absent => (Shard::Whole, None),
            Placement::Cut(cut) if cut.groups > 1 => (Shard::Whole, Some(cut.held())),
            Placement::Cut(cut) => match cut.axis {
                super::super::load::splits::Axis::Out => (Shard::Out(cut.start..cut.start + cut.len), None),
                super::super::load::splits::Axis::In => (Shard::In(cut.start..cut.start + cut.len), None),
            },
        })
    }

    /// Upload the entries `ents` into one arena; returns (entry -> tensor, arena, bytes). Leg MEM1: the sizes come from the
    /// plan first, then one entry at a time is read into ONE reused host buffer and copied (the old path held a whole layer,
    /// ~2.5 GB at TP2, in host memory at once: the freed pieces stayed in the process's heap, ~3 GB of it never returned).
    fn upload(&self, g: &Gpu, ents: &[usize]) -> Result<(HashMap<usize, DevTensor>, Vec<Buf>, u64)> {
        let held: Vec<(usize, u64, Vec<u64>)> = ents.iter().filter_map(|&i| self.held(i).map(|(b, s)| (i, b, s))).collect();
        let total: u64 = held.iter().map(|(_, b, _)| align(*b)).sum();
        let arena = g.buf(total.max(256) as usize)?;
        let mut files = HashMap::new();
        let mut v: Vec<u8> = Vec::new();
        let mut off = 0u64;
        let mut dev = HashMap::new();
        for (i, b, shape) in held {
            let e = &self.c.entries[i];
            self.read_into(&mut files, i, &mut v)?;
            ensure!(v.len() as u64 == b, "{}: read {} B, the plan holds {b} B", e.name, v.len());
            g.h2d(arena.ptr() + off, &v)?;
            dev.insert(i, DevTensor { role: e.role, ptr: DevPtr(arena.ptr() + off), dtype: e.dtype.clone(), shape, bytes: b });
            off += align(b);
        }
        Ok((dev, vec![arena], total))
    }

    /// Load trunk layer `l` as this rank holds it.
    pub fn load_layer(&self, g: &Gpu, l: usize) -> Result<LayerW> {
        ensure!(l < self.cfg.total_layers(), "layer {l}: the pack has {} trunk layers and {} DSpark stages", self.cfg.layers, self.cfg.dspark_stages);
        let c = &self.c;
        let ents: Vec<usize> = (0..c.entries.len()).filter(|&i| c.entries[i].layer == Some(l as u16) && c.entries[i].class != C::EngramTable).collect();
        let (dev_t, arena, bytes) = self.upload(g, &ents)?;
        let mut mods: Vec<u32> = ents.iter().filter(|&&i| dev_t.contains_key(&i)).map(|&i| c.entries[i].module).collect();
        mods.sort_unstable();
        mods.dedup();
        let mut lin: HashMap<LinKey, (DenseWeight, BackendKind)> = HashMap::new();
        let mut params = HashMap::new();
        let mut experts: HashMap<u16, [Option<DenseWeight>; 3]> = HashMap::new();
        let mut moe_backend = None;
        for mi in mods {
            let m = &c.modules[mi as usize];
            let tensors: Vec<DevTensor> = m.tensors.iter().filter_map(|&ti| dev_t.get(&(ti as usize)).cloned()).collect();
            if !m.class.is_linear() {
                let t = tensors.first().with_context(|| format!("{}: parameter not uploaded", m.key))?;
                params.insert(m.class, t.clone());
                continue;
            }
            ensure!(tensors.len() == m.tensors.len(), "{}: {} of {} tensors held (a partial module)", m.key, tensors.len(), m.tensors.len());
            let pl = &self.plan.linears[*self.by_key.get(&m.key).with_context(|| format!("{}: not in the weight plan", m.key))?];
            let (shard, held_k) = self.shard_of(m)?;
            let inn = held_k.unwrap_or(pl.inn);
            let w = DenseWeight { key: m.key.clone(), class: m.class, layer: m.layer, sub: m.sub, format: m.format.clone(), out: pl.out, inn, shard, tensors };
            if m.class.is_routed_expert() {
                moe_backend = Some(pl.backend);
                let slot = match m.class { C::ExpertW1 => 0, C::ExpertW2 => 1, _ => 2 };
                experts.entry(m.sub.context("expert without an id")?).or_insert([None, None, None])[slot] = Some(w);
                continue;
            }
            lin.insert((m.class, m.sub), (w, pl.backend));
        }
        let (n_routed, top_k) = self.cfg.moe_of(l);
        let moe = if experts.is_empty() { None } else {
            ensure!(experts.len() == n_routed, "layer {l}: {} of {n_routed} experts held (the TP forward runs the inner-width split; EP is not wired)", experts.len());
            let mut ex = Vec::with_capacity(n_routed);
            for e in 0..n_routed as u16 {
                let [a, b, d] = experts.remove(&e).with_context(|| format!("layer {l}: expert {e} missing"))?;
                ex.push([a.context("w1")?, b.context("w2")?, d.context("w3")?]);
            }
            Some((MoeWeight { layer: l as u16, n_experts: n_routed, local: 0..n_routed, top_k, dim: self.cfg.hidden, inter: self.cfg.moe_inter,
                              swiglu_limit: self.cfg.swiglu_limit as f32, experts: ex }, moe_backend.context("experts without a backend")?))
        };
        Ok(LayerW::from_parts(l, lin, moe, params, bytes, arena))
    }

    /// The globals the trunk needs: embedding (when this rank holds it), final norm, head (the rank's vocabulary rows).
    pub fn load_global(&self, g: &Gpu) -> Result<Global> {
        let c = &self.c;
        let want = |cl: C| matches!(cl, C::Embed | C::Head | C::FinalNorm);
        let ents: Vec<usize> = (0..c.entries.len()).filter(|&i| c.entries[i].layer.is_none() && want(c.entries[i].class)).collect();
        let (dev_t, arena, bytes) = self.upload(g, &ents)?;
        let mut embed = None;
        let mut head = None;
        let mut norm = None;
        let mut mods: Vec<u32> = ents.iter().filter(|&&i| dev_t.contains_key(&i)).map(|&i| c.entries[i].module).collect();
        mods.sort_unstable();
        mods.dedup();
        for mi in mods {
            let m = &c.modules[mi as usize];
            let tensors: Vec<DevTensor> = m.tensors.iter().filter_map(|&ti| dev_t.get(&(ti as usize)).cloned()).collect();
            match m.class {
                C::FinalNorm => norm = tensors.first().cloned(),
                C::Embed | C::Head => {
                    let pl = &self.plan.linears[*self.by_key.get(&m.key).with_context(|| format!("{}: not in the weight plan", m.key))?];
                    let (shard, _) = self.shard_of(m)?;
                    let w = DenseWeight { key: m.key.clone(), class: m.class, layer: None, sub: None, format: m.format.clone(), out: pl.out, inn: pl.inn, shard, tensors };
                    if m.class == C::Embed { embed = Some((w, pl.backend)) } else { head = Some((w, pl.backend)) }
                }
                _ => {}
            }
        }
        Ok(Global { embed, head: head.context("the rank holds no head rows")?, norm: norm.context("no final norm")?, bytes, _arena: arena })
    }

    /// Rank 0's vision tower, 3x3 aligner and image delimiter embeddings (`None` on Ranks `1..N`, or when `--dsv41-vision off`).
    pub fn load_vision(&self, g: &Gpu) -> Result<Option<crate::dsv41::vision::tower::VisionWeights>> {
        use crate::dsv41::vision::{is_vision_global_class, tower::{vision_inv_freq, VisBlock, VisLinear, VisionWeights}, vision_enabled, VisionMode};
        let mode = VisionMode::from_registry();
        if mode == VisionMode::Off { return Ok(None); }
        let Some(vis) = self.cfg.vision.clone().filter(|v| v.layers > 0) else {
            ensure!(mode != VisionMode::On, "--dsv41-vision on requested, but this pack's config.json has no vision_config");
            return Ok(None);
        };
        if self.rank != 0 { return Ok(None); }
        let c = &self.c;
        let ents: Vec<usize> = (0..c.entries.len())
            .filter(|&i| c.entries[i].layer.is_none() && is_vision_global_class(c.entries[i].class) && self.held(i).is_some())
            .collect();
        if ents.is_empty() {
            ensure!(mode != VisionMode::On, "--dsv41-vision on requested, but rank 0 holds no vision tensors (a slice pack without --dsv41-slice-vision?)");
            return Ok(None);
        }
        let (dev_t, arena, bytes) = self.upload(g, &ents)?;
        let mut mods: Vec<u32> = ents.iter().filter(|&&i| dev_t.contains_key(&i)).map(|&i| c.entries[i].module).collect();
        mods.sort_unstable();
        mods.dedup();
        let mut lin_map: HashMap<String, VisLinear> = HashMap::new();
        let mut param_map: HashMap<String, DevTensor> = HashMap::new();
        for mi in mods {
            let m = &c.modules[mi as usize];
            let tensors: Vec<DevTensor> = m.tensors.iter().filter_map(|&ti| dev_t.get(&(ti as usize)).cloned()).collect();
            if !m.class.is_linear() {
                let t = tensors.first().with_context(|| format!("{}: parameter not uploaded", m.key))?;
                param_map.insert(m.key.clone(), t.clone());
                continue;
            }
            ensure!(tensors.len() == m.tensors.len(), "{}: {} of {} tensors held", m.key, tensors.len(), m.tensors.len());
            let pl = &self.plan.linears[*self.by_key.get(&m.key).with_context(|| format!("{}: not in the weight plan", m.key))?];
            let (shard, held_k) = self.shard_of(m)?;
            let w = DenseWeight {
                key: m.key.clone(), class: m.class, layer: m.layer, sub: m.sub, format: m.format.clone(),
                out: pl.out, inn: held_k.unwrap_or(pl.inn), shard, tensors,
            };
            lin_map.insert(m.key.clone(), VisLinear::from_dense((w, pl.backend))?);
        }
        let take_lin = |m: &mut HashMap<String, VisLinear>, k: &str| m.remove(k).with_context(|| format!("vision linear '{k}' missing on rank 0"));
        let take_prm = |m: &mut HashMap<String, DevTensor>, k: &str| m.remove(k).with_context(|| format!("vision parameter '{k}' missing on rank 0"));
        let patch_embed = take_lin(&mut lin_map, "vision.patch_embed.proj")?;
        let mut blocks = Vec::with_capacity(vis.layers);
        for b in 0..vis.layers {
            blocks.push(VisBlock {
                norm1: take_prm(&mut param_map, &format!("vision.blocks.{b}.norm1"))?,
                wqkv: take_lin(&mut lin_map, &format!("vision.blocks.{b}.attn.wqkv"))?,
                wo: take_lin(&mut lin_map, &format!("vision.blocks.{b}.attn.wo"))?,
                norm2: take_prm(&mut param_map, &format!("vision.blocks.{b}.norm2"))?,
                w1: take_lin(&mut lin_map, &format!("vision.blocks.{b}.mlp.w1"))?,
                w2: take_lin(&mut lin_map, &format!("vision.blocks.{b}.mlp.w2"))?,
            });
        }
        let norm = take_prm(&mut param_map, "vision.norm")?;
        let aligner_w1 = take_lin(&mut lin_map, "aligner.w1")?;
        let aligner_w2 = take_lin(&mut lin_map, "aligner.w2")?;
        let image_start = take_prm(&mut param_map, "image_start")?;
        let image_newline = take_prm(&mut param_map, "image_newline")?;
        let image_end = take_prm(&mut param_map, "image_end")?;
        let inv_freq = vision_inv_freq(vis.hidden / vis.heads / 2, vis.rope_theta)?;
        let _ = vision_enabled;
        Ok(Some(VisionWeights {
            vis, hidden: self.cfg.hidden, inv_freq, patch_embed, blocks, norm, aligner_w1, aligner_w2,
            image_start, image_newline, image_end, bytes, _arena: arena,
        }))
    }

    /// Bytes of every tensor this rank holds (the memory table's "planned" column), and of the parts not loaded.
    pub fn held_bytes(&self) -> (u64, u64) {
        let mut all = 0u64;
        let mut skipped = 0u64;
        let vis_on = self.rank == 0 && crate::dsv41::vision::vision_enabled(&self.cfg);
        for (i, e) in self.c.entries.iter().enumerate() {
            let b = match &self.holds { None => e.bytes, Some(h) => match h.get(&(i as u32)) { Some(h) => h.bytes, None => continue } };
            all += b;
            let loaded = match e.layer {
                Some(l) => (l as usize) < self.cfg.layers && e.class != C::EngramTable,
                None => matches!(e.class, C::Embed | C::Head | C::FinalNorm) || (vis_on && crate::dsv41::vision::is_vision_global_class(e.class)),
            };
            if !loaded { skipped += b; }
        }
        (all, skipped)
    }

    /// Test-only: remap the 10-layer slice pack `[0, 1, 2, 14, 20, 21, 24, 40, 41, 42]` into a compact 7-trunk + 3-DSpark
    /// `RankSrc` so `TpModel` + `TpSpec` + `TpServe` + `TpForward` run unmodified on a single GPU for bit-exactness verification.
    #[cfg(test)]
    pub fn compact_slice_for_test(mut self, with_engram: bool) -> Result<Self> {
        let map: [(u16, u16); 10] = [(0, 0), (1, 1), (2, 2), (14, 3), (20, 4), (21, 5), (24, 6), (40, 7), (41, 8), (42, 9)];
        let remap = |l: u16| map.iter().find(|&&(a, _)| a == l).map(|&(_, b)| b);
        for &(orig, _) in &map {
            ensure!(self.has_layer(orig as usize), "slice pack missing layer {orig}");
        }
        for e in &mut self.c.entries {
            if let Some(l) = e.layer { e.layer = remap(l); }
        }
        for m in &mut self.c.modules {
            if let Some(l) = m.layer { m.layer = remap(l); }
        }
        self.cfg.layers = 7;
        self.cfg.dspark_stages = 3;
        self.cfg.compress_ratios = vec![0, 0, 2, 2, 1, 1, 1, 0, 0, 0];
        self.cfg.kv_sources = vec![2, 3, 4];
        self.cfg.index_sources = vec![2, 3, 4, 6];
        self.cfg.candidate_source = Some(4);
        self.cfg.dspark_targets = vec![4, 5, 6];
        self.cfg.engram_layers = if with_engram { vec![1, 3] } else { Vec::new() };
        self.c.cfg = self.cfg.clone();
        Ok(self)
    }
}

/// The trunk's globals on a rank.
pub struct Global {
    /// the embedding table (plain BF16) when this rank holds it (TP2: rank 0 only)
    pub embed: Option<(DenseWeight, BackendKind)>,
    pub head: (DenseWeight, BackendKind),
    pub norm: DevTensor,
    pub bytes: u64,
    _arena: Vec<Buf>,
}

/// The held K of an EXL3 linear's resident trellis (rows of 16 x tiles).
pub fn held_k(w: &DenseWeight) -> Option<u64> {
    w.tensors.iter().find(|t| t.role == Role::Trellis).map(|t| t.shape[0] * 16)
}

pub fn is_exl3(w: &DenseWeight) -> bool { matches!(w.format, LinearFormat::Exl3 { .. }) }
