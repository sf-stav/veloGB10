//! SL: one layer's weights on the device, from ANY pack (a slice or a full pack; EXL3, native or mixed),
//! through the census: every linear becomes a `DenseWeight` held by the backend `WeightPlan` names for
//! it (per linear, never per pack), the routed experts one `MoeWeight` (EXL3: trellis tensors per expert;
//! native: KN2's mx1 packed buffers), everything else a parameter tensor (norms, hc params, sinks, gate
//! bias, Engram q / k). One device arena per layer; the layer is dropped as a whole.
//!
//! Reading: positioned reads (`FileExt::read_at`) at the census' absolute data offsets, so a full
//! 500 GB checkpoint is read only where the layer lives (`.12` reads the native layers straight out of
//! its full checkpoint; no native slice copy is needed).

use super::dev::{Buf, Gpu};
use crate::dsv41::census::classify::{Role, TensorClass};
use crate::dsv41::census::{Census, Entry, LinearFormat};
use crate::dsv41::weights::native::moe as nmoe;
use crate::dsv41::weights::{BackendKind, DenseWeight, DevPtr, DevTensor, MoeWeight, PlannedLinear, Shard, WeightPlan};
use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;
use std::os::unix::fs::FileExt;

pub type LinKey = (TensorClass, Option<u16>);

pub struct LayerW {
    pub layer: usize,
    pub lin: HashMap<LinKey, (DenseWeight, BackendKind)>,
    pub moe: Option<(MoeWeight, BackendKind)>,
    pub params: HashMap<TensorClass, DevTensor>,
    pub bytes: u64,
    _arena: Vec<Buf>,
}

impl LayerW {
    /// A layer assembled by another loader (leg TP: a rank's slices from its shard dir); the arena buffers own the bytes.
    pub fn from_parts(layer: usize, lin: HashMap<LinKey, (DenseWeight, BackendKind)>, moe: Option<(MoeWeight, BackendKind)>,
                      params: HashMap<TensorClass, DevTensor>, bytes: u64, arena: Vec<Buf>) -> LayerW {
        LayerW { layer, lin, moe, params, bytes, _arena: arena }
    }
    pub fn lin(&self, c: TensorClass) -> Result<&(DenseWeight, BackendKind)> {
        self.lin.get(&(c, None)).with_context(|| format!("layer {}: no {} linear", self.layer, c.name()))
    }
    pub fn lin_sub(&self, c: TensorClass, sub: u16) -> Option<&(DenseWeight, BackendKind)> { self.lin.get(&(c, Some(sub))) }
    pub fn has(&self, c: TensorClass) -> bool { self.lin.contains_key(&(c, None)) || self.params.contains_key(&c) }
    pub fn param(&self, c: TensorClass) -> Result<&DevTensor> {
        self.params.get(&c).with_context(|| format!("layer {}: no {} parameter", self.layer, c.name()))
    }
    /// The raw weight tensor of a plain linear (router gate) or a parameter.
    pub fn weight_tensor(&self, c: TensorClass) -> Result<&DevTensor> {
        if let Some((w, _)) = self.lin.get(&(c, None)) {
            return w.tensors.iter().find(|t| t.role == Role::Weight).with_context(|| format!("{}: no weight tensor", w.key));
        }
        self.param(c)
    }
}

/// The census and plan of a pack, built once.
pub struct Pack { pub c: Census, pub plan: WeightPlan, by_key: HashMap<String, usize> }

impl Pack {
    pub fn open(dir: &std::path::Path) -> Result<Pack> {
        let c = Census::build(dir).with_context(|| format!("census of {}", dir.display()))?;
        let plan = WeightPlan::from_census(&c)?;
        let by_key = plan.linears.iter().enumerate().map(|(i, p)| (p.key.clone(), i)).collect();
        Ok(Pack { c, plan, by_key })
    }
    pub fn planned(&self, key: &str) -> Option<&PlannedLinear> { self.by_key.get(key).map(|&i| &self.plan.linears[i]) }
    pub fn is_exl3(&self) -> bool { self.c.modules.iter().any(|m| matches!(m.format, LinearFormat::Exl3 { .. })) }
    pub fn has_layer(&self, l: usize) -> bool { self.c.entries.iter().any(|e| e.layer == Some(l as u16)) }
}

fn align(x: u64) -> u64 { (x + 255) / 256 * 256 }

/// Read one tensor's bytes from its file.
fn read_entry(c: &Census, files: &mut HashMap<u32, std::fs::File>, e: &Entry) -> Result<Vec<u8>> {
    if !files.contains_key(&e.file) {
        let p = c.dir.join(&c.files[e.file as usize]);
        files.insert(e.file, std::fs::File::open(&p).with_context(|| format!("open {}", p.display()))?);
    }
    let f = &files[&e.file];
    let mut v = vec![0u8; e.bytes as usize];
    f.read_exact_at(&mut v, e.offset).with_context(|| format!("read {}", e.name))?;
    Ok(v)
}

/// Load layer `l` (a trunk layer or a DSpark stage slot) onto the device.
pub fn load_layer(g: &Gpu, p: &Pack, l: usize) -> Result<LayerW> {
    let c = &p.c;
    // every tensor of the layer except the Engram tables (~100 GB per Engram layer in a full native checkpoint:
    // the EIO runtime reads their rows from the packed head files, never from device memory)
    let ents: Vec<usize> = (0..c.entries.len()).filter(|&i| c.entries[i].layer == Some(l as u16) && c.entries[i].class != TensorClass::EngramTable).collect();
    ensure!(!ents.is_empty(), "the pack at {} does not hold layer {l}", c.dir.display());
    let expert = |e: &Entry| e.class.is_routed_expert();
    let native_moe = ents.iter().any(|&i| expert(&c.entries[i]) && matches!(c.module_of(&c.entries[i]).format, LinearFormat::Fp4 { .. }));
    // one arena for every tensor except native experts (those are packed on the host first)
    let arena_bytes: u64 = ents.iter().map(|&i| &c.entries[i]).filter(|e| !(native_moe && expert(e))).map(|e| align(e.bytes)).sum();
    let arena = g.buf(arena_bytes as usize)?;
    let mut files = HashMap::new();
    let mut off = 0u64;
    let mut dev_t: HashMap<usize, DevTensor> = HashMap::new();
    for &i in &ents {
        let e = &c.entries[i];
        if native_moe && expert(e) { continue; }
        let v = read_entry(c, &mut files, e)?;
        g.h2d(arena.ptr() + off, &v)?;
        dev_t.insert(i, DevTensor { role: e.role, ptr: DevPtr(arena.ptr() + off), dtype: e.dtype.clone(), shape: e.shape.clone(), bytes: e.bytes });
        off += align(e.bytes);
    }
    let mut lw = LayerW { layer: l, lin: HashMap::new(), moe: None, params: HashMap::new(), bytes: arena_bytes, _arena: vec![arena] };
    // modules of the layer
    let mut mods: Vec<u32> = ents.iter().map(|&i| c.entries[i].module).collect();
    mods.sort_unstable();
    mods.dedup();
    let mut experts: HashMap<u16, [Option<DenseWeight>; 3]> = HashMap::new();
    let mut moe_backend = None;
    for mi in mods {
        let m = &c.modules[mi as usize];
        if !m.class.is_linear() {
            let t = m.tensors.first().and_then(|&ti| dev_t.get(&(ti as usize))).with_context(|| format!("{}: parameter not uploaded", m.key))?;
            lw.params.insert(m.class, t.clone());
            continue;
        }
        let pl = p.planned(&m.key).with_context(|| format!("{}: not in the weight plan", m.key))?;
        if m.class.is_routed_expert() {
            moe_backend = Some(pl.backend);
            if native_moe { continue; }
            let w = DenseWeight { key: m.key.clone(), class: m.class, layer: m.layer, sub: m.sub, format: m.format.clone(), out: pl.out, inn: pl.inn,
                                  shard: Shard::Whole, tensors: m.tensors.iter().filter_map(|&ti| dev_t.get(&(ti as usize)).cloned()).collect() };
            let slot = match m.class { TensorClass::ExpertW1 => 0, TensorClass::ExpertW2 => 1, _ => 2 };
            experts.entry(m.sub.context("expert without an id")?).or_insert([None, None, None])[slot] = Some(w);
            continue;
        }
        let tensors: Vec<DevTensor> = m.tensors.iter().filter_map(|&ti| dev_t.get(&(ti as usize)).cloned()).collect();
        ensure!(tensors.len() == m.tensors.len(), "{}: {} of {} tensors uploaded", m.key, tensors.len(), m.tensors.len());
        let w = DenseWeight { key: m.key.clone(), class: m.class, layer: m.layer, sub: m.sub, format: m.format.clone(), out: pl.out, inn: pl.inn, shard: Shard::Whole, tensors };
        lw.lin.insert((m.class, m.sub), (w, pl.backend));
    }
    // native wo_a: one FP8 [groups * o_lora, in / groups] matrix -> one group view per o_group (the forward calls wo_a per group)
    if let Some((w, b)) = lw.lin.get(&(TensorClass::AttnWoA, None)).cloned() {
        if b == BackendKind::Native {
            let groups = c.cfg.o_groups;
            let rank = c.cfg.o_lora_rank as u64;
            let wt = w.tensors.iter().find(|t| t.role == Role::Weight).context("wo_a weight")?;
            let st = w.tensors.iter().find(|t| t.role == Role::Scale).context("wo_a scale")?;
            let k = wt.shape[1];
            ensure!(wt.shape[0] == groups as u64 * rank && rank % 32 == 0, "wo_a shape {:?}", wt.shape);
            for gi in 0..groups as u64 {
                let wv = DevTensor { role: Role::Weight, ptr: DevPtr(wt.ptr.0 + gi * rank * k), dtype: wt.dtype.clone(), shape: vec![rank, k], bytes: rank * k };
                let sv = DevTensor { role: Role::Scale, ptr: DevPtr(st.ptr.0 + gi * (rank / 32) * st.shape[1]), dtype: st.dtype.clone(), shape: vec![rank / 32, st.shape[1]], bytes: (rank / 32) * st.shape[1] };
                let gw = DenseWeight { key: format!("{}.group{gi}", w.key), sub: Some(gi as u16), out: rank, inn: k, tensors: vec![wv, sv], ..w.clone() };
                lw.lin.insert((TensorClass::AttnWoA, Some(gi as u16)), (gw, b));
            }
        }
    }
    let (n_routed, top_k) = c.cfg.moe_of(l);
    if native_moe {
        let (mw, buf) = load_native_moe(g, c, &ents, l, n_routed, top_k, &mut files)?;
        lw.bytes += buf.iter().map(|b| b.bytes as u64).sum::<u64>();
        lw._arena.extend(buf);
        lw.moe = Some((mw, moe_backend.context("native experts without a backend")?));
    } else if !experts.is_empty() {
        let mut ex = Vec::with_capacity(n_routed);
        for e in 0..n_routed as u16 {
            let [a, b2, c3] = experts.remove(&e).with_context(|| format!("layer {l}: expert {e} missing"))?;
            ex.push([a.context("w1")?, b2.context("w2")?, c3.context("w3")?]);
        }
        let mw = MoeWeight { layer: l as u16, n_experts: n_routed, local: 0..n_routed, top_k, dim: c.cfg.hidden, inter: c.cfg.moe_inter,
                             swiglu_limit: c.cfg.swiglu_limit as f32, experts: ex };
        lw.moe = Some((mw, moe_backend.context("experts without a backend")?));
    }
    Ok(lw)
}

/// The pack's global tensors of the given classes on the device as a `LayerW` (`layer` = the trunk's layer count):
/// leg CE's score head (`FinalNorm` + `Head`). Linears through the weight plan, everything else a parameter.
pub fn load_globals(g: &Gpu, p: &Pack, classes: &[TensorClass]) -> Result<LayerW> {
    let c = &p.c;
    let ents: Vec<usize> = (0..c.entries.len()).filter(|&i| c.entries[i].layer.is_none() && classes.contains(&c.entries[i].class)).collect();
    ensure!(!ents.is_empty(), "the pack at {} holds none of {:?}", c.dir.display(), classes.iter().map(|x| x.name()).collect::<Vec<_>>());
    let arena_bytes: u64 = ents.iter().map(|&i| align(c.entries[i].bytes)).sum();
    let arena = g.buf(arena_bytes as usize)?;
    let mut files = HashMap::new();
    let mut off = 0u64;
    let mut dev_t: HashMap<usize, DevTensor> = HashMap::new();
    for &i in &ents {
        let e = &c.entries[i];
        let v = read_entry(c, &mut files, e)?;
        g.h2d(arena.ptr() + off, &v)?;
        dev_t.insert(i, DevTensor { role: e.role, ptr: DevPtr(arena.ptr() + off), dtype: e.dtype.clone(), shape: e.shape.clone(), bytes: e.bytes });
        off += align(e.bytes);
    }
    let mut lw = LayerW { layer: c.cfg.layers, lin: HashMap::new(), moe: None, params: HashMap::new(), bytes: arena_bytes, _arena: vec![arena] };
    let mut mods: Vec<u32> = ents.iter().map(|&i| c.entries[i].module).collect();
    mods.sort_unstable();
    mods.dedup();
    for mi in mods {
        let m = &c.modules[mi as usize];
        let tensors: Vec<DevTensor> = m.tensors.iter().filter_map(|&ti| dev_t.get(&(ti as usize)).cloned()).collect();
        if !m.class.is_linear() {
            lw.params.insert(m.class, tensors.first().cloned().with_context(|| format!("{}: parameter not uploaded", m.key))?);
            continue;
        }
        let pl = p.planned(&m.key).with_context(|| format!("{}: not in the weight plan", m.key))?;
        ensure!(tensors.len() == m.tensors.len(), "{}: {} of {} tensors uploaded", m.key, tensors.len(), m.tensors.len());
        let w = DenseWeight { key: m.key.clone(), class: m.class, layer: m.layer, sub: m.sub, format: m.format.clone(), out: pl.out, inn: pl.inn, shard: Shard::Whole, tensors };
        lw.lin.insert((m.class, m.sub), (w, pl.backend));
    }
    Ok(lw)
}

/// Native MXFP4 experts -> KN2's mx1 layout (two device buffers, uniform stride), as `native::moe`'s contract says.
fn load_native_moe(g: &Gpu, c: &Census, ents: &[usize], l: usize, n: usize, top_k: usize, files: &mut HashMap<u32, std::fs::File>)
                   -> Result<(MoeWeight, Vec<Buf>)> {
    // raw (w, s) of (expert, matrix)
    let mut raw: HashMap<(u16, usize, bool), Vec<u8>> = HashMap::new();
    for &i in ents {
        let e = &c.entries[i];
        if !e.class.is_routed_expert() { continue; }
        let mat = match e.class { TensorClass::ExpertW1 => 0, TensorClass::ExpertW2 => 1, _ => 2 };
        let is_scale = match e.role { Role::Weight => false, Role::Scale => true, r => bail!("{}: unexpected expert tensor role {r:?}", e.name) };
        raw.insert((e.sub.context("expert id")?, mat, is_scale), read_entry(c, files, e)?);
    }
    let get = |e: u16, m: usize, s: bool| raw.get(&(e, m, s)).map(|v| v.as_slice()).with_context(|| format!("layer {l}: expert {e} matrix {m} {}", if s { "scale" } else { "weight" }));
    let mut rx = Vec::with_capacity(n);
    for e in 0..n as u16 {
        rx.push(crate::dsv41::weights::native::moe::reference::RawExpert { w1: get(e, 0, false)?, s1: get(e, 0, true)?, w3: get(e, 2, false)?, s3: get(e, 2, true)?,
                                                                            w2: get(e, 1, false)?, s2: get(e, 1, true)? });
    }
    let geom = nmoe::pack::ExpertGeom::whole(c.cfg.hidden, c.cfg.moe_inter);
    let (b13, b2) = nmoe::pack_layer(&rx, &geom)?;
    drop(rx);
    let (d13, d2) = (g.buf(b13.len())?, g.buf(b2.len())?);
    g.h2d(d13.ptr(), &b13)?;
    g.h2d(d2.ptr(), &b2)?;
    let (e13, e2) = (geom.w13_bytes() as u64, geom.w2_bytes() as u64);
    let fmt = c.modules.iter().find(|m| m.layer == Some(l as u16) && m.class == TensorClass::ExpertW1).map(|m| m.format.clone()).context("expert format")?;
    let dw = |key: String, cl: TensorClass, out: u64, inn: u64, t: Vec<DevTensor>| DenseWeight { key, class: cl, layer: Some(l as u16), sub: None, format: fmt.clone(), out, inn, shard: Shard::Whole, tensors: t };
    let (dim, inter) = (c.cfg.hidden as u64, c.cfg.moe_inter as u64);
    let experts = (0..n as u64).map(|e| {
        let t13 = DevTensor { role: Role::Weight, ptr: DevPtr(d13.ptr() + e * e13), dtype: nmoe::DT_W13.into(), shape: vec![e13], bytes: e13 };
        let t2 = DevTensor { role: Role::Weight, ptr: DevPtr(d2.ptr() + e * e2), dtype: nmoe::DT_W2.into(), shape: vec![e2], bytes: e2 };
        [dw(format!("layers.{l}.ffn.experts.{e}.w13"), TensorClass::ExpertW1, inter, dim, vec![t13]),
         dw(format!("layers.{l}.ffn.experts.{e}.w2"), TensorClass::ExpertW2, dim, inter, vec![t2]),
         dw(format!("layers.{l}.ffn.experts.{e}.w3"), TensorClass::ExpertW3, inter, dim, vec![])]
    }).collect();
    let mw = MoeWeight { layer: l as u16, n_experts: n, local: 0..n, top_k, dim: c.cfg.hidden, inter: c.cfg.moe_inter, swiglu_limit: c.cfg.swiglu_limit as f32, experts };
    Ok((mw, vec![d13, d2]))
}
