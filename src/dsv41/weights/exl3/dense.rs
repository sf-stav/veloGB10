//! KE: the EXL3 dense path (attention, shared / DSpark / compressor / indexer linears, Engram wkv, main_proj, the head):
//! prep (suh + H128) -> trellis GEMM (GEMV / coop / wide by rows) -> post (split sum + H128 + svh), with the rank's
//! slice on either side. Residency: the trellis / suh / svh tensors may be the WHOLE tensor (the shard picks the range
//! inside it) or already the rank's SLICE (shape = slice; a block-aligned Out slice or any In slice); both are detected
//! from the tensor shapes. An un-aligned Out slice (a vocabulary range) needs the whole tensor resident.

use super::launch::*;
use crate::dsv41::census::classify::Role;
use crate::dsv41::weights::{Accum, Act, ActDtype, DenseWeight, DevTensor, Launch, RowClass, Scratch, Shard};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

pub fn tensor<'a>(w: &'a DenseWeight, role: Role) -> Result<&'a DevTensor> {
    w.tensors.iter().find(|t| t.role == role).with_context(|| format!("{}: no {} tensor on the device", w.key, role.name()))
}

/// The resolved geometry of one dense call.
pub struct Resolved {
    pub plan: DensePlan, pub bits: i32,
    pub tr: u64, pub suh: u64, pub svh: u64,
    pub nb_stride: i32, pub nb0: i32, pub kb0: i32, pub nkb: i32, pub nbn: i32,
}

pub fn resolve(w: &DenseWeight, rows: usize) -> Result<Resolved> {
    let (out, inn) = (w.out as usize, w.inn as usize);
    let (n_lo, n_hi, k_lo, k_hi) = match &w.shard {
        Shard::Whole => (0, out, 0, inn),
        Shard::Out(r) => (r.start as usize, r.end as usize, 0, inn),
        Shard::In(r) => (0, out, r.start as usize, r.end as usize),
    };
    ensure!(n_lo < n_hi && n_hi <= out && k_lo < k_hi && k_hi <= inn, "{}: bad shard {:?}", w.key, w.shard);
    ensure!(k_lo % 128 == 0 && (k_hi - k_lo) % 128 == 0, "{}: an EXL3 input slice is whole 128-blocks (got {k_lo}..{k_hi})", w.key);
    let (tr, suh, svh) = (tensor(w, Role::Trellis)?, tensor(w, Role::Suh)?, tensor(w, Role::Svh)?);
    ensure!(tr.shape.len() == 3, "{}: trellis shape {:?}", w.key, tr.shape);
    let bits = bits_code(tr.shape[2])?;
    let kloc = k_hi - k_lo;
    let plan = DensePlan::new(rows, kloc, n_lo, n_hi, 0);
    let nblk_lo = n_lo / 128;
    let (kb_res, nb_res) = (tr.shape[0] as usize, tr.shape[1] as usize);
    let whole = kb_res * 16 == inn && nb_res * 16 == out;
    let (kb0, nb0) = if whole { (k_lo / 16, nblk_lo * 8) } else {
        ensure!(kb_res * 16 == kloc && nb_res * 16 == plan.nblk * 128 && n_lo % 128 == 0,
                "{}: resident trellis {:?} is neither the whole [{}, {}] nor the slice [{}, {}]", w.key, tr.shape, inn / 16, out / 16, kloc / 16, plan.nblk * 8);
        (0, 0)
    };
    let suh_p = if suh.shape[0] as usize == inn { suh.ptr.0 + 2 * k_lo as u64 } else { ensure!(suh.shape[0] as usize == kloc, "{}: suh {:?}", w.key, suh.shape); suh.ptr.0 };
    let svh_p = if svh.shape[0] as usize == out { svh.ptr.0 + 2 * (nblk_lo * 128) as u64 } else { ensure!(svh.shape[0] as usize == plan.nblk * 128, "{}: svh {:?}", w.key, svh.shape); svh.ptr.0 };
    Ok(Resolved { plan, bits, tr: tr.ptr.0, suh: suh_p, svh: svh_p, nb_stride: nb_res as i32, nb0: nb0 as i32, kb0: kb0 as i32, nkb: (kloc / 16) as i32, nbn: (plan.nblk * 8) as i32 })
}

/// Rows above this go to the wide (tensor-core tile) path, 17_exl3_dense_wide.cuh.
pub const WIDE_MIN_ROWS: usize = 33;

pub fn dense(l: &Launch, w: &DenseWeight, x: &Act, y: &Act, acc: Accum, s: Scratch) -> Result<RowClass> {
    if x.rows >= WIDE_MIN_ROWS { return super::dense_wide::dense_wide(l, w, x, y, acc, s, super::dense_wide::configured_src()); }
    // PK4: rows <= 32 at an integer K take the stream-K decode class (18_*); fractional K keeps KE's split-K class
    if super::dense_dec::eligible(w, x.rows) {
        super::dense_dec::run(l, &[super::dense_dec::Call::new(w, x, y, acc)], s)?;
        return Ok(RowClass::of(x.rows));
    }
    dense_split(l, w, x, y, acc, s)
}

/// The kernel family `dense` runs for this weight at `rows` rows (for receipts: the forward's per-entry counters).
pub fn entry_of(w: &DenseWeight, rows: usize) -> String {
    if rows >= WIDE_MIN_ROWS { return "d41x_dw_* (wide tensor-core tiles)".into(); }
    match resolve(w, rows) {
        Ok(r) if (2..=8).contains(&r.bits) => format!("d41x_dd_prep+{}", super::dense_dec::entry(rows, r.bits)),
        _ => format!("d41x_prep+d41x_gemm_m{}+d41x_post", if rows <= 16 { 1 } else { 2 }),
    }
}

/// KE's split-K path (rows <= 32 at a fractional K; also callable at any row count, e.g. by the probe's "old" timing):
/// prep -> split-K GEMM -> post.
pub fn dense_split(l: &Launch, w: &DenseWeight, x: &Act, y: &Act, acc: Accum, s: Scratch) -> Result<RowClass> {
    let rows = x.rows;
    ensure!(rows > 0 && y.rows == rows, "{}: rows x {} y {}", w.key, rows, y.rows);
    let r = resolve(w, rows)?;
    let p = &r.plan;
    ensure!(x.cols == p.kloc && y.cols == p.ncols, "{}: x cols {} (want {}), y cols {} (want {})", w.key, x.cols, p.kloc, y.cols, p.ncols);
    ensure!(s.bytes >= p.scratch_bytes(), "{}: scratch {} < {}", w.key, s.bytes, p.scratch_bytes());
    let xh = s.ptr.0; let yraw = s.ptr.0 + p.xh_bytes as u64;
    let xcode = act_code(x.dtype)?; let ycode = act_code(y.dtype)?;
    ensure!(y.dtype != ActDtype::Fp8E4m3, "FP8 output");
    // prep
    let k = kernel("d41x_prep")?;
    let (mut xp, mut xc, mut ldx, mut suh, mut rws, mut xhp, mut lda) = (x.ptr.0, xcode, x.ld as i32, r.suh, rows as i32, xh, p.kloc as i32);
    launch(&k, ((p.kloc / 128) as u32, rows as u32, 1), 32, l.stream, &mut [&mut xp as *mut _ as *mut c_void, &mut xc as *mut _ as *mut c_void, &mut ldx as *mut _ as *mut c_void,
        &mut suh as *mut _ as *mut c_void, &mut rws as *mut _ as *mut c_void, &mut xhp as *mut _ as *mut c_void, &mut lda as *mut _ as *mut c_void])?;
    // gemm
    let ldy = (p.nblk * 128) as i32;
    let mut a = GemmArgs { tr: r.tr, xh, y: yraw, gstride: rows as i64 * ldy as i64, nb_stride: r.nb_stride, nb0: r.nb0, nbn: r.nbn, kb0: r.kb0, nkb: r.nkb,
                           lda: p.kloc as i32, rows: rows as i32, ldy, ks: p.ks, g: p.g, bits: r.bits };
    let k = kernel(match p.mt { 1 => "d41x_gemm_m1", 2 => "d41x_gemm_m2", 4 => "d41x_gemm_m4", _ => "d41x_gemm_m8" })?;
    launch(&k, p.grid, 256, l.stream, &mut [&mut a as *mut _ as *mut c_void])?;
    // post
    let k = kernel("d41x_post")?;
    let (mut yp, mut gs, mut g, mut ldy2, mut svh, mut rws, mut skip, mut nc, mut op, mut oc, mut ldo, mut ac) =
        (yraw, a.gstride, p.g, ldy, r.svh, rows as i32, p.skip as i32, p.ncols as i32, y.ptr.0, ycode, y.ld as i32, (acc == Accum::Add) as i32);
    launch(&k, (p.nblk as u32, rows as u32, 1), 32, l.stream, &mut [&mut yp as *mut _ as *mut c_void, &mut gs as *mut _ as *mut c_void, &mut g as *mut _ as *mut c_void,
        &mut ldy2 as *mut _ as *mut c_void, &mut svh as *mut _ as *mut c_void, &mut rws as *mut _ as *mut c_void, &mut skip as *mut _ as *mut c_void, &mut nc as *mut _ as *mut c_void,
        &mut op as *mut _ as *mut c_void, &mut oc as *mut _ as *mut c_void, &mut ldo as *mut _ as *mut c_void, &mut ac as *mut _ as *mut c_void])?;
    Ok(RowClass::of(rows))
}

pub fn scratch_bytes(w: &DenseWeight, rows: usize) -> usize {
    if rows >= WIDE_MIN_ROWS {
        // the served wide path (`configured_src()`: `Auto` by default, or `--dsv41-pf-wide`; `Auto` includes reconstruction
        // only at shapes/widths where it wins, and `dense_wide` falls back gracefully to `d41x_dw_tr` if scratch is tighter)
        return super::dense_wide::scratch_bytes(w, rows, super::dense_wide::configured_src()).unwrap_or(0);
    }
    if super::dense_dec::eligible(w, rows) { return super::dense_dec::scratch_bytes(w, rows); }
    resolve(w, rows).map(|r| r.plan.scratch_bytes()).unwrap_or(0)
}

