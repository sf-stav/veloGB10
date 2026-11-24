//! KE / KE2: the MoE entry over a per-expert inner-width slice table. `prepare_moe` builds and uploads the device table
//! once per layer (call it at load / warm-up, before any graph capture); `moe` then needs no host work.
//! Paths (`moe_ex`): rows <= 32 -> the decode kernels (moe_dec.rs, 15_*.cuh); rows > 32 -> the prefill kernels
//! (moe_wide.rs, 16_*.cuh); a width the fast kernels do not cover (fractional K) -> the generic six-launch path
//! (13_*.cuh; > 32 rows as 32-row chunks), returned as `RowClass::Generic`. All three give the same bits for a row.

use super::dense::tensor;
use super::launch::*;
use crate::dsv41::census::classify::Role;
use crate::dsv41::weights::{Act, ActDtype, DenseWeight, Launch, MoeWeight, Route, RowClass, Scratch, Shard};
use anyhow::{ensure, Context, Result};
use cudarc::driver::sys;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

pub const MAX_ROWS: usize = 32;

/// One layer's device table. `cnt` = the decode kernels' arrival counters (zeroed here, re-armed by the kernels):
/// MAXS x `ncb` gate/up counters then dim / 128 down counters; one MoE call at a time per table (one compute stream).
/// `k13` / `k2` (leg SX1): the (smallest, largest) width code over the held experts' w1 and w3 matrices, and over their w2;
/// every kernel reads each matrix at its own entry's code (`ExpertEntry::k1` / `k3` / `k2`), these ranges only size the
/// rings / stages and select which entries a call launches.
pub struct MoeTable { pub dev: u64, pub n_experts: usize, pub nblk_max: usize, pub k13: (i32, i32), pub k2: (i32, i32), pub dim: usize, pub inter: usize, pub cnt: u64, pub ncb: usize }

impl MoeTable {
    /// The largest width code of any matrix in the table (ring / stage sizing).
    pub fn kmax(&self) -> i32 { self.k13.1.max(self.k2.1) }
    /// The smallest width code of any matrix in the table.
    pub fn kmin(&self) -> i32 { self.k13.0.min(self.k2.0) }
    /// Every matrix at an integer K 2..8 (the decode / prefill kernels' widths; fractional codes run the generic path).
    pub fn int_k(&self) -> bool { (2..=8).contains(&self.kmin()) && (2..=8).contains(&self.kmax()) }
}

pub(super) fn sm_count() -> i32 {
    static N: OnceLock<i32> = OnceLock::new();
    *N.get_or_init(|| {
        let mut v = 0i32;
        let r = unsafe { sys::cuDeviceGetAttribute(&mut v, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0) };
        if r == sys::CUresult::CUDA_SUCCESS && v > 0 { v } else { 48 }
    })
}

fn slice_of(sh: &Shard, full: usize) -> Result<(usize, usize)> {
    let (lo, hi) = match sh { Shard::Whole => (0, full), Shard::Out(r) | Shard::In(r) => (r.start as usize, r.end as usize) };
    ensure!(lo % 128 == 0 && (hi % 128 == 0 || hi == full) && lo < hi && hi <= full, "expert slice {lo}..{hi} of {full} is not whole 128-blocks");
    Ok((lo, hi))
}

/// Leg SX1: the width codes of one expert's three matrices as the kernels read them: (w1, w3, w2). Every MoE path reads
/// each matrix at its own code, so any combination of widths is served; nothing here is refused.
pub fn expert_widths(w1: &DenseWeight, w3: &DenseWeight, w2: &DenseWeight) -> Result<(i32, i32, i32)> {
    let code = |d: &DenseWeight| -> Result<i32> { bits_code(tensor(d, Role::Trellis)?.shape[2]) };
    Ok((code(w1)?, code(w3)?, code(w2)?))
}

/// Leg SX1, the load-time guard (prepare_moe, before upload): every held expert's entry must name, for EACH of its three
/// matrices, that matrix's own trellis width. A table whose entry would make a kernel read a matrix at another matrix's
/// width is refused here (serving wrong output is the one case where refusing is right): this is the check that would
/// have caught EAQ v1 2.9 bpw layer 29's shared w2 (K5) being read at its w1 / w3 width (K4).
pub fn check_widths(w: &MoeWeight, ents: &[ExpertEntry]) -> Result<()> {
    for (i, dw) in w.experts.iter().enumerate() {
        let e = w.local.start + i;
        let ent = ents.get(e).with_context(|| format!("layer {}: no table entry for expert {e}", w.layer))?;
        for (name, d, k) in [("w1", &dw[0], ent.k1), ("w3", &dw[2], ent.k3), ("w2", &dw[1], ent.k2)] {
            let own = bits_code(tensor(d, Role::Trellis)?.shape[2])?;
            ensure!(own == k, "layer {} expert {e}: {name} ({}) is stored at {} but the MoE table would read it at {}; refused at load \
                     instead of serving wrong output", w.layer, d.key, width_name(own), width_name(k));
        }
    }
    Ok(())
}

/// A width code as the pack names it (K 4, K 2.5).
pub fn width_name(code: i32) -> String { if code >= 17 { format!("K {}.5", code - 16) } else { format!("K {code}") } }

/// Build the host table (entries for every global expert id; not-held ones are all zero). Leg SX1: each entry carries the
/// width code of each of its three matrices, and each matrix's slice offset is computed at its own code.
pub fn build_entries(w: &MoeWeight) -> Result<(Vec<ExpertEntry>, usize, (i32, i32), (i32, i32))> {
    let (dim, inter) = (w.dim, w.inter);
    ensure!(dim % 128 == 0 && inter % 128 == 0, "dim {dim} / inter {inter} must be 128-multiples");
    let mut ents = vec![ExpertEntry::default(); w.n_experts];
    let mut nblk_max = 0usize;
    let (mut k13r, mut k2r): (Option<(i32, i32)>, Option<(i32, i32)>) = (None, None);
    let widen = |r: &mut Option<(i32, i32)>, k: i32| { *r = Some(r.map_or((k, k), |(a, b)| (a.min(k), b.max(k)))); };
    ensure!(w.experts.len() == w.local.len(), "layer {}: {} experts for local range {:?}", w.layer, w.experts.len(), w.local);
    for (i, dw) in w.experts.iter().enumerate() {
        let e = w.local.start + i;
        let (w1, w2, w3): (&DenseWeight, &DenseWeight, &DenseWeight) = (&dw[0], &dw[1], &dw[2]);
        let (lo, hi) = slice_of(&w1.shard, inter)?;
        ensure!(slice_of(&w3.shard, inter)? == (lo, hi) && slice_of(&w2.shard, inter)? == (lo, hi), "expert {e}: w1 / w2 / w3 slices differ");
        let nblk = (hi - lo) / 128;
        nblk_max = nblk_max.max(nblk);
        let (k1, k3, k2) = expert_widths(w1, w3, w2)?;
        widen(&mut k13r, k1);
        widen(&mut k13r, k3);
        widen(&mut k2r, k2);
        let mut ent = ExpertEntry { nblk: nblk as i32, k1, k3, k2, ..Default::default() };
        for (m, d) in [(0usize, w1), (1, w3)] {
            let (tr, suh, svh) = (tensor(d, Role::Trellis)?, tensor(d, Role::Suh)?, tensor(d, Role::Svh)?);
            let b = bits_code(tr.shape[2])?;   // this matrix's own code (= k1 for w1, k3 for w3)
            ensure!(tr.shape[0] as usize * 16 == dim, "expert {e}: w1/w3 trellis {:?}", tr.shape);
            let whole = tr.shape[1] as usize * 16 == inter;
            ensure!(whole || tr.shape[1] as usize * 16 == hi - lo, "expert {e}: w1/w3 trellis {:?} neither whole nor slice", tr.shape);
            ent.tr[m] = tr.ptr.0 + if whole { (lo / 16) as u64 * tile_words(b) * 2 } else { 0 };
            ent.nbs13 = tr.shape[1] as i32;
            ent.suh[m] = suh.ptr.0;
            ent.svh[m] = svh.ptr.0 + if svh.shape[0] as usize == inter { 2 * lo as u64 } else { 0 };
        }
        let (tr, suh, svh) = (tensor(w2, Role::Trellis)?, tensor(w2, Role::Suh)?, tensor(w2, Role::Svh)?);
        let b = bits_code(tr.shape[2])?;   // w2's own code (= k2)
        ensure!(tr.shape[1] as usize * 16 == dim, "expert {e}: w2 trellis {:?}", tr.shape);
        let whole = tr.shape[0] as usize * 16 == inter;
        ensure!(whole || tr.shape[0] as usize * 16 == hi - lo, "expert {e}: w2 trellis {:?} neither whole nor slice", tr.shape);
        ent.tr[2] = tr.ptr.0 + if whole { (lo / 16) as u64 * (dim as u64 / 16) * tile_words(b) * 2 } else { 0 };
        ent.suh[2] = suh.ptr.0 + if suh.shape[0] as usize == inter { 2 * lo as u64 } else { 0 };
        ent.svh[2] = svh.ptr.0;
        ents[e] = ent;
    }
    ensure!(nblk_max > 0, "layer {}: this rank holds no expert slice", w.layer);
    Ok((ents, nblk_max, k13r.context("no experts")?, k2r.context("no experts")?))
}

fn tables() -> &'static Mutex<HashMap<(usize, u16, u64, usize, u64, u64), std::sync::Arc<MoeTable>>> {
    static T: OnceLock<Mutex<HashMap<(usize, u16, u64, usize, u64, u64), std::sync::Arc<MoeTable>>>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Forget every cached table (call when MoeWeights are freed / rebuilt; tests call it between configurations).
pub fn clear_tables() {
    let mut t = tables().lock().unwrap();
    for (_, v) in t.drain() { unsafe { sys::cuMemFree_v2(v.dev as sys::CUdeviceptr); sys::cuMemFree_v2(v.cnt as sys::CUdeviceptr); } }
}

/// Upload the layer's table (idempotent). Needs a current CUDA context; never call inside a graph capture.
pub fn prepare_moe(w: &MoeWeight) -> Result<std::sync::Arc<MoeTable>> {
    let first = w.experts.first().map(|e| tensor(&e[0], Role::Trellis).map(|t| t.ptr.0)).transpose()?.unwrap_or(0);
    let sh = |d: Option<&[DenseWeight; 3]>| match d.map(|d| &d[0].shard) { Some(Shard::Out(r)) | Some(Shard::In(r)) => r.start * 65536 + r.end, _ => 0 };
    // key: the weight's address + a cheap fingerprint (first tensor pointer, first / last slice). A loader that frees and
    // re-creates a MoeWeight must call `clear_tables()` (stale-pointer class); the loader owns the lifetime.
    let key = (w as *const MoeWeight as usize, w.layer, first, w.n_experts, sh(w.experts.first()), sh(w.experts.last()));
    if let Some(t) = tables().lock().unwrap().get(&key) { return Ok(t.clone()); }
    let (ents, nblk_max, k13, k2) = build_entries(w)?;
    check_widths(w, &ents)?;
    let bytes = ents.len() * std::mem::size_of::<ExpertEntry>();
    let mut dptr: sys::CUdeviceptr = 0;
    unsafe {
        ensure!(sys::cuMemAlloc_v2(&mut dptr, bytes) == sys::CUresult::CUDA_SUCCESS, "cuMemAlloc for the expert table");
        ensure!(sys::cuMemcpyHtoD_v2(dptr, ents.as_ptr() as *const c_void, bytes) == sys::CUresult::CUDA_SUCCESS, "cuMemcpyHtoD for the expert table");
    }
    let ncb = w.inter / 128;
    // gate/up arrivals [MAXS][ncb], down arrivals [dim / 128], then (leg PF1) the per-slot gate/up block counts [MAXS], the
    // per-slot ready epochs [MAXS] and the call epoch (15_*: `d41x_md_rdy`)
    let cbytes = (super::moe_dec::MAXS * ncb + w.dim / 128 + 2 * super::moe_dec::MAXS + 4) * 4;
    let mut cptr: sys::CUdeviceptr = 0;
    unsafe {
        ensure!(sys::cuMemAlloc_v2(&mut cptr, cbytes) == sys::CUresult::CUDA_SUCCESS, "cuMemAlloc for the MoE counters");
        ensure!(sys::cuMemsetD8_v2(cptr, 0, cbytes) == sys::CUresult::CUDA_SUCCESS, "cuMemsetD8 for the MoE counters");
    }
    super::moe_dec::geometry()?;
    super::moe_wide::check_geometry()?;
    let t = std::sync::Arc::new(MoeTable { dev: dptr as u64, n_experts: w.n_experts, nblk_max, k13, k2, dim: w.dim, inter: w.inter, cnt: cptr as u64, ncb });
    tables().lock().unwrap().insert(key, t.clone());
    Ok(t)
}

struct Lay { ctl: usize, slot: usize, pair: usize, xh1: usize, y1: usize, xh2: usize, y2: usize, total: usize }
fn layout(rows: usize, topk: usize, dim: usize, inter_ld: usize) -> Lay {
    let a = |b: usize| (b + 255) & !255;
    let p = rows * topk;
    let ctl = 0; let slot = ctl + 256;
    let pair = slot + a(3 * p * 4);
    let xh1 = pair + a(5 * p * 4);
    let y1 = xh1 + a(p * 2 * dim * 2);
    let xh2 = y1 + a(p * 2 * inter_ld * 4);
    let y2 = xh2 + a(p * inter_ld * 2);
    Lay { ctl, slot, pair, xh1, y1, xh2, y2, total: y2 + a(p * dim * 4) }
}

/// Scratch for one call (<= 32 rows per chunk). `nblk_max` = the widest slice of the layer's table.
pub fn scratch_bytes(rows: usize, topk: usize, dim: usize, nblk_max: usize) -> usize {
    layout(rows.min(MAX_ROWS), topk, dim, nblk_max * 128).total
}

/// Which implementation serves a call (the probe compares them; the trait entry uses `Auto`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoePath {
    /// the fastest path for the call's shape (decode kernels <= 32 rows, the prefill kernels above)
    Auto,
    /// the generic six-launch path (any width code; > 32 rows as 32-row chunks)
    Generic,
}

/// Scratch for a call of `rows` rows on `path`, with an optional shared expert (a one-expert table) in the same call.
pub fn scratch_for(w: &MoeWeight, shared: Option<&MoeWeight>, rows: usize, path: MoePath) -> Result<usize> {
    let t = prepare_moe(w)?;
    let ts = shared.map(prepare_moe).transpose()?;
    if path == MoePath::Auto && rows <= MAX_ROWS && super::moe_dec::eligible(&t, ts.as_deref(), w, shared) {
        return Ok(super::moe_dec::scratch_bytes(&t, ts.as_deref(), rows, w.top_k));
    }
    if path == MoePath::Auto && rows > MAX_ROWS && super::moe_wide::eligible(&t, ts.as_deref(), w, shared) {
        return Ok(super::moe_wide::scratch_bytes(&t, ts.as_deref(), rows, w.top_k));
    }
    let mut n = scratch_bytes(rows, w.top_k, w.dim, t.nblk_max);
    if let Some(ts) = &ts { n = n.max(scratch_bytes(rows, 1, ts.dim, ts.nblk_max)); }
    Ok(n)
}

/// Routed experts (+ the shared expert when given: a one-expert table, ids 0, weight 1.0) ADDED into y f32 [rows, dim]:
/// y = (y + routed) + shared, the routed sum in ascending expert id.
pub fn moe_ex(l: &Launch, w: &MoeWeight, shared: Option<&MoeWeight>, x: &Act, route: &Route, y: &Act, s: Scratch, path: MoePath) -> Result<RowClass> {
    moe_ex_ow(l, w, shared, x, route, y, s, path, false)
}

/// `moe_ex` with `overwrite`: y = the MoE output instead of y += it (leg PF1). The decode kernels write `0 + sum` in their
/// combine (the same bits as a zeroed y plus the sum, so no memset node precedes the call); every other path zeroes y on
/// the stream first and accumulates as before.
#[allow(clippy::too_many_arguments)]
pub fn moe_ex_ow(l: &Launch, w: &MoeWeight, shared: Option<&MoeWeight>, x: &Act, route: &Route, y: &Act, s: Scratch, path: MoePath, overwrite: bool) -> Result<RowClass> {
    if path == MoePath::Auto && x.rows <= MAX_ROWS {
        let t = prepare_moe(w)?;
        let ts = shared.map(prepare_moe).transpose()?;
        if super::moe_dec::eligible(&t, ts.as_deref(), w, shared) && (!overwrite || shared.is_some()) {
            super::moe_dec::run_ow(l, &t, ts.as_deref(), w, shared, x, route, y, s, overwrite)?;
            return Ok(RowClass::of(x.rows));
        }
    }
    if overwrite {
        ensure!(y.dtype == ActDtype::F32 && y.ld == y.cols, "moe {}: overwrite needs a dense f32 y", w.layer);
        let r = unsafe { sys::cuMemsetD8Async(y.ptr.0, 0, y.rows * y.ld * 4, l.stream as sys::CUstream) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "moe {}: zero y: {r:?}", w.layer);
    }
    if path == MoePath::Auto && x.rows > MAX_ROWS {
        let t = prepare_moe(w)?;
        let ts = shared.map(prepare_moe).transpose()?;
        if super::moe_wide::eligible(&t, ts.as_deref(), w, shared) {
            super::moe_wide::run(l, &t, ts.as_deref(), w, shared, x, route, y, s)?;
            return Ok(RowClass::Wide);
        }
    }
    let c = moe_generic(l, w, x, route, y, s)?;
    if let Some(sh) = shared {
        ensure!(sh.n_experts == 1 && sh.top_k == 1, "shared expert table must hold one expert");
        let ones = shared_route(x.rows)?;
        moe_generic(l, sh, x, &ones, y, s)?;
    }
    // the served (Auto) call fell back: counted as Generic; an explicit Generic request reports the row class it ran
    Ok(if path == MoePath::Auto { RowClass::Generic } else { c })
}

/// ids 0 / weights 1.0 for `rows` rows (device constants, grown on demand; uploaded once, never inside a capture).
fn shared_route(rows: usize) -> Result<Route> {
    static R: OnceLock<Mutex<(u64, u64, usize)>> = OnceLock::new();
    let m = R.get_or_init(|| Mutex::new((0, 0, 0)));
    let mut g = m.lock().unwrap();
    if g.2 < rows {
        let n = rows.max(64).next_power_of_two();
        let (ids, wts) = (vec![0i32; n], vec![1.0f32; n]);
        let (mut a, mut b): (sys::CUdeviceptr, sys::CUdeviceptr) = (0, 0);
        unsafe {
            ensure!(sys::cuMemAlloc_v2(&mut a, n * 4) == sys::CUresult::CUDA_SUCCESS && sys::cuMemAlloc_v2(&mut b, n * 4) == sys::CUresult::CUDA_SUCCESS, "cuMemAlloc (shared route)");
            ensure!(sys::cuMemcpyHtoD_v2(a, ids.as_ptr() as *const c_void, n * 4) == sys::CUresult::CUDA_SUCCESS
                 && sys::cuMemcpyHtoD_v2(b, wts.as_ptr() as *const c_void, n * 4) == sys::CUresult::CUDA_SUCCESS, "upload (shared route)");
        }
        *g = (a as u64, b as u64, n);
    }
    Ok(Route { ids: crate::dsv41::weights::DevPtr(g.0), weights: crate::dsv41::weights::DevPtr(g.1), top_k: 1 })
}

pub fn moe(l: &Launch, w: &MoeWeight, x: &Act, route: &Route, y: &Act, s: Scratch) -> Result<RowClass> {
    moe_ex(l, w, None, x, route, y, s, MoePath::Auto)
}

/// The generic path: six launches per <= 32-row chunk, any width code.
pub fn moe_generic(l: &Launch, w: &MoeWeight, x: &Act, route: &Route, y: &Act, s: Scratch) -> Result<RowClass> {
    let t = prepare_moe(w)?;
    ensure!(y.dtype == ActDtype::F32 && x.cols == w.dim && y.cols == w.dim && x.rows == y.rows, "moe {}: y must be f32 [rows, dim]", w.layer);
    ensure!(route.top_k == w.top_k && route.top_k <= 8, "moe {}: top_k {}", w.layer, route.top_k);
    let rows = x.rows;
    let esz = |d: ActDtype| match d { ActDtype::F32 => 4u64, _ => 2 };
    let mut done = 0usize;
    while done < rows {
        let n = (rows - done).min(MAX_ROWS);
        let need = scratch_bytes(n, w.top_k, w.dim, t.nblk_max);
        ensure!(s.bytes >= need, "moe {}: scratch {} < {need}", w.layer, s.bytes);
        let inter_ld = t.nblk_max * 128;
        let lay = layout(n, w.top_k, w.dim, inter_ld);
        let b = s.ptr.0;
        let at = |o: usize| b + o as u64;
        let p = n * w.top_k;
        let mut a = MoeArgs {
            ex: t.dev, ids: route.ids.0 + (done * w.top_k * 4) as u64, wts: route.weights.0 + (done * w.top_k * 4) as u64,
            x: x.ptr.0 + done as u64 * x.ld as u64 * esz(x.dtype), xcode: act_code(x.dtype)?, ldx: x.ld as i32,
            ctl: at(lay.ctl), slot_expert: at(lay.slot), slot_off: at(lay.slot + 4 * p), slot_cnt: at(lay.slot + 8 * p),
            pair_expert: at(lay.pair), pair_row: at(lay.pair + 4 * p), pair_w: at(lay.pair + 8 * p), pair_of: at(lay.pair + 12 * p), row_order: at(lay.pair + 16 * p),
            xh1: at(lay.xh1), y1: at(lay.y1), xh2: at(lay.xh2), y2: at(lay.y2),
            out: y.ptr.0 + done as u64 * y.ld as u64 * 4, ldo: y.ld as i32, accum: 1, limit: w.swiglu_limit,
            n_experts: t.n_experts as i32, rows: n as i32, topk: w.top_k as i32, dim: w.dim as i32, inter_ld: inter_ld as i32,
            nblk_max: t.nblk_max as i32, kmax: t.kmax(), sms: sm_count(),
        };
        let pp = |a: &mut MoeArgs| [a as *mut MoeArgs as *mut c_void];
        launch(&kernel("d41x_moe_route")?, (1, 1, 1), 512, l.stream, &mut pp(&mut a))?;
        launch(&kernel("d41x_moe_prep1")?, ((w.dim / 128) as u32, p as u32, 1), 32, l.stream, &mut pp(&mut a))?;
        launch(&kernel("d41x_moe_gemm1")?, ((2 * sm_count()) as u32, 1, 1), 256, l.stream, &mut pp(&mut a))?;
        launch(&kernel("d41x_moe_act")?, (t.nblk_max as u32, p as u32, 1), 32, l.stream, &mut pp(&mut a))?;
        launch(&kernel("d41x_moe_gemm2")?, ((2 * sm_count()) as u32, 1, 1), 256, l.stream, &mut pp(&mut a))?;
        launch(&kernel("d41x_moe_post2")?, ((w.dim / 128) as u32, n as u32, 1), 32, l.stream, &mut pp(&mut a))?;
        done += n;
    }
    Ok(if rows <= MAX_ROWS { RowClass::of(rows) } else { RowClass::Generic })
}

#[cfg(test)]
mod tests {
    //! Leg SX1: host-side tests of the expert table (no device: the pointers are fake addresses, never dereferenced).
    use super::*;
    use crate::dsv41::census::classify::TensorClass;
    use crate::dsv41::census::LinearFormat;
    use crate::dsv41::weights::{DevPtr, DevTensor};

    /// A fake EXL3 linear [out, in] at K (integer) with trellis base address `base`.
    fn lin(key: &str, out: u64, inn: u64, k: u64, base: u64, shard: Shard) -> DenseWeight {
        let t = |role, ptr: u64, dt: &str, shape: Vec<u64>| DevTensor { role, ptr: DevPtr(ptr), dtype: dt.into(), shape, bytes: 0 };
        DenseWeight { key: key.into(), class: TensorClass::SharedW1, layer: Some(29), sub: None,
            format: LinearFormat::Exl3 { k: k as u8, mul1: true, suh: true, svh: true }, out, inn, shard,
            tensors: vec![t(Role::Trellis, base, "I16", vec![inn / 16, out / 16, 16 * k]), t(Role::Suh, base + 0x100_0000, "F16", vec![inn]),
                          t(Role::Svh, base + 0x200_0000, "F16", vec![out])] }
    }
    /// One expert (dim 512, inter 256) with widths (w1, w3, w2), sliced to the inner range lo..hi.
    fn expert(k1: u64, k3: u64, k2: u64, lo: u64, hi: u64) -> [DenseWeight; 3] {
        let (dim, inter) = (512u64, 256u64);
        let (s13, s2) = if lo == 0 && hi == inter { (Shard::Whole, Shard::Whole) } else { (Shard::Out(lo..hi), Shard::In(lo..hi)) };
        [lin("layers.29.ffn.shared_experts.w1", inter, dim, k1, 0x1000_0000, s13.clone()),
         lin("layers.29.ffn.shared_experts.w2", dim, inter, k2, 0x2000_0000, s2),
         lin("layers.29.ffn.shared_experts.w3", inter, dim, k3, 0x3000_0000, s13)]
    }
    fn table(ex: Vec<[DenseWeight; 3]>) -> MoeWeight {
        MoeWeight { layer: 29, n_experts: ex.len(), local: 0..ex.len(), top_k: 1, dim: 512, inter: 256, swiglu_limit: 10.0, experts: ex }
    }

    #[test]
    fn dsv41_sx1_mixed_width_shared_expert_entry_carries_each_matrix_width_and_offset() {
        // EAQ v1 2.9 bpw layer 29's shared expert: w1 / w3 K4, w2 K5, held as the inner slice 128..256 (TP2 rank 1)
        let w = table(vec![expert(4, 4, 5, 128, 256)]);
        let (ents, nblk, k13, k2) = build_entries(&w).unwrap();
        assert_eq!((ents[0].k1, ents[0].k3, ents[0].k2), (4, 4, 5));
        assert_eq!((nblk, k13, k2), (1, (4, 4), (5, 5)));
        // w1 / w3 are cut along their output n-tiles at THEIR width; w2 along its input k16 steps at ITS width (K5)
        assert_eq!(ents[0].tr[0], 0x1000_0000 + (128 / 16) * tile_words(4) * 2);
        assert_eq!(ents[0].tr[1], 0x3000_0000 + (128 / 16) * tile_words(4) * 2);
        assert_eq!(ents[0].tr[2], 0x2000_0000 + (128 / 16) * (512 / 16) * tile_words(5) * 2);
        check_widths(&w, &ents).unwrap();
        let t = MoeTable { dev: 0, n_experts: 1, nblk_max: nblk, k13, k2, dim: 512, inter: 256, cnt: 0, ncb: 2 };
        assert_eq!((t.kmin(), t.kmax(), t.int_k()), (4, 5, true));
    }

    #[test]
    fn dsv41_sx1_every_matrix_of_every_expert_at_its_own_width() {
        // the SAGE 3.30 bpw pattern: w1 != w3 != w2, and widths that differ across the routed experts of one layer
        let w = table(vec![expert(4, 5, 4, 0, 256), expert(5, 7, 6, 0, 256), expert(2, 3, 2, 0, 256), expert(8, 8, 8, 0, 256)]);
        let (ents, _, k13, k2) = build_entries(&w).unwrap();
        let got: Vec<(i32, i32, i32)> = ents.iter().map(|e| (e.k1, e.k3, e.k2)).collect();
        assert_eq!(got, vec![(4, 5, 4), (5, 7, 6), (2, 3, 2), (8, 8, 8)]);
        assert_eq!((k13, k2), ((2, 8), (2, 8)));
        check_widths(&w, &ents).unwrap();
        // uniform modules keep the pre-SX1 table bytes apart from the three width fields (every offset unchanged)
        let u = table(vec![expert(3, 3, 3, 0, 256), expert(3, 3, 3, 0, 256)]);
        let (ue, _, uk13, uk2) = build_entries(&u).unwrap();
        assert!(ue.iter().all(|e| (e.k1, e.k3, e.k2) == (3, 3, 3) && e.tr[0] == 0x1000_0000 && e.tr[2] == 0x2000_0000));
        assert_eq!((uk13, uk2), ((3, 3), (3, 3)));
    }

    #[test]
    fn dsv41_sx1_a_table_that_would_read_a_matrix_at_another_width_is_refused_at_load() {
        let w = table(vec![expert(4, 4, 5, 0, 256)]);
        let (mut ents, ..) = build_entries(&w).unwrap();
        // the pre-SX1 table: one width for the whole shared expert (w2 read at w1's K4)
        ents[0].k2 = ents[0].k1;
        let err = format!("{:#}", check_widths(&w, &ents).unwrap_err());
        assert!(err.contains("w2") && err.contains("stored at K 5") && err.contains("read it at K 4") && err.contains("refused at load"), "{err}");
        // an unsupported width is refused when the table is built, with the matrix named
        let bad = table(vec![[lin("x.w1", 256, 512, 4, 0x1000, Shard::Whole), lin("x.w2", 512, 256, 9, 0x2000, Shard::Whole), lin("x.w3", 256, 512, 4, 0x3000, Shard::Whole)]]);
        assert!(build_entries(&bad).is_err());
    }

    #[test]
    fn dsv41_sx1_width_names() {
        assert_eq!(width_name(5), "K 5");
        assert_eq!(width_name(18), "K 2.5");
    }
}
