//! PK4: host side of the dense DECODE class (kernels/gpu_dsv41/18_exl3_dense_dec.cuh): rows 1..=32, integer K 2..=8,
//! one or more linears ("segments") per call. Two launches: `d41x_dd_prep` (x . suh -> H128 -> fp16, zeroes the block
//! counters) and `d41x_dd_a{8,16,24,32}_k{K}` (persistent stream-K trellis GEMM, output transform fused). Both honour
//! `Launch::pdl`; the prep triggers its dependents before it waits, and the GEMM streams its first weights before its
//! own wait, so consecutive linears keep the weight stream full under PDL / graph replay.
//!
//! Numerics (the rows <= 32 class): fixed by the slice shape and `PARTS` only, so every row of an N-row call is
//! bit-identical to its 1-row call (N = 1..32). Not bit-equal to KE's split-K class (`dense::dense_split`, still served
//! for fractional K) nor to the wide class (>= 33 rows): equal accuracy against the f64 chain (the probe gates both).
//!
//! Leg PD1 (the external reference recipe's dense decode path, PLAN/DSV41_RX4_REPORT.md A): (1) GROUPED launches: each
//! segment of a call is cut into its own `min(PARTS, units)` partitions exactly as its 1-segment call cuts them, so a
//! grouped call's bits per segment equal the segment's own call (`run_group` puts every same-width linear of a group in
//! one launch); (2) PRODUCER-SIDE ROTATION: a segment's rotated input may be written by the kernel that produces x
//! (`Call::pre`), and a segment may rotate its own finished output into the next linear's input (`Call::rot`, the
//! `rot_out` epilogue), so the next call has no `d41x_dd_prep`; `normrot` is the RMSNorm that rotates for wq_b; (3) the
//! block arrival counters live in one persistent zeroed buffer (re-armed by each block's last arrival), so a call
//! without a prep needs no zeroing; (4) `l2_paced`, the paced L2 prefetch kernel's launcher.

use super::dense::{resolve, Resolved};
use super::launch::{act_code, kernel, tile_words};
use super::moe_dec::{grid_for, launch_ex};
use crate::dsv41::weights::{Accum, Act, ActDtype, DenseWeight, Launch, Scratch};
use anyhow::{ensure, Context, Result};
use cudarc::driver::sys;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

/// D41X_DD_P: partitions of a call's unit line (P = min(PARTS, units)).
pub const PARTS: usize = 96;
/// D41X_DD_MAXSEG: linears per call.
pub const MAXSEG: usize = 8;
const KPS: usize = 4;
const PREP_ROWS: usize = 4;
const THREADS: u32 = 256;
/// Dynamic smem per CTA that keeps two GEMM CTAs per SM alongside `d41x_dd_prep` co-residency + L1 cache (44 KB),
/// the hard 2-CTA cap (GB10: 100 KB per SM, 1 KB reserved per CTA, 128 B static), and the 1-CTA cap.
const SMEM_2CTA_DEEP: usize = 44 * 1024;
const SMEM_2CTA: usize = 48 * 1024 - 256;
const SMEM_1CTA: usize = 101_376 - 256;
const NS_MAX: usize = 16;

/// kernels/gpu_dsv41/18_exl3_dense_dec.cuh `D41xDdSeg`.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct DdSeg {
    pub tr: u64, pub x: u64, pub suh: u64, pub svh: u64, pub out: u64, pub xh: u64, pub rsuh: u64, pub rxh: u64,
    pub kstride: i64,
    pub nkb: i32, pub ncb: i32, pub skip: i32, pub ncols: i32,
    pub ldx: i32, pub xcode: i32, pub ldo: i32, pub ocode: i32, pub accum: i32,
    pub u0: i32, pub jb0: i32, pub p0: i32, pub np: i32,
    pub xld: i32, pub pre: i32, pub rld: i32, pub pad_: i32,
}
/// `D41xDdArgs`.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct DdArgs {
    pub seg: [DdSeg; MAXSEG],
    pub part: u64, pub cnt: u64,
    pub nseg: i32, pub rows: i32, pub u: i32, pub ncb_all: i32,
    pub bits: i32, pub slot: i32, pub ns: i32, pub wreg: i32, pub maxkb: i32, pub p: i32,
}

/// `D41xNormRotArgs` (d41x_dd_normrot).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct NormRotArgs {
    pub x: u64, pub x_ld: i64, pub w: u64, pub y: u64, pub y_ld: i64,
    pub w_fmt: u32, pub dim: i32, pub rows: i32, pub eps: f32, pub inv_dim: f32, pub nt: i32,
    pub suh: [u64; 2], pub xh: [u64; 2], pub xld: [i32; 2],
}
/// `D41xL2PacedArgs` (09_l2_paced.cuh d41x_l2_paced).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct L2PacedArgs {
    pub ptr: [u64; PF_MAXR], pub run: [u64; PF_MAXR], pub stride: [u64; PF_MAXR], pub count: [u64; PF_MAXR],
    pub n: i32, pub pad_: i32, pub delay_ns: u64, pub wave_ns: u64,
}
/// Ranges one paced prefetch launch takes (D41X_PF_MAXR).
pub const PF_MAXR: usize = 16;

/// Block arrival counters of every dense decode call (leg PD1): one persistent zeroed buffer; each block's last arrival
/// re-arms its counter, so the buffer is all zero between calls (calls run one at a time on the forward's stream).
pub const CNT_LEN: usize = 1 << 16;
static GEOM: OnceLock<u64> = OnceLock::new();
fn counters() -> Result<u64> { GEOM.get().copied().context("dense decode: geometry() was not called (the counters are allocated there, before any capture)") }

/// Read the kernels' argument layout once and allocate the persistent counters (a tiny kernel + sync: call before any
/// graph capture; `exl3::prepare` does).
pub fn geometry() -> Result<()> {
    if GEOM.get().is_some() { return Ok(()); }
    let k = kernel("d41x_dd_sizes")?;
    let mut d: sys::CUdeviceptr = 0;
    let mut h = [0i32; 16];
    unsafe {
        ensure!(sys::cuMemAlloc_v2(&mut d, 64) == sys::CUresult::CUDA_SUCCESS, "cuMemAlloc (dd sizes)");
        let mut p = d as u64;
        let mut params = [&mut p as *mut u64 as *mut c_void];
        let mut st: sys::CUstream = std::ptr::null_mut();
        ensure!(sys::cuStreamCreate(&mut st, 0) == sys::CUresult::CUDA_SUCCESS, "cuStreamCreate (dd sizes)");
        let r = sys::cuLaunchKernel(k.0, 1, 1, 1, 32, 1, 1, 0, st, params.as_mut_ptr(), std::ptr::null_mut());
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "launch d41x_dd_sizes: {r:?}");
        ensure!(sys::cuStreamSynchronize(st) == sys::CUresult::CUDA_SUCCESS, "sync (dd sizes)");
        sys::cuStreamDestroy_v2(st);
        ensure!(sys::cuMemcpyDtoH_v2(h.as_mut_ptr() as *mut c_void, d, 64) == sys::CUresult::CUDA_SUCCESS, "read dd sizes");
        sys::cuMemFree_v2(d);
    }
    ensure!(h[0] as usize == std::mem::size_of::<DdArgs>() && h[1] as usize == std::mem::size_of::<DdSeg>(),
            "D41xDdArgs / D41xDdSeg are {} / {} B on the device, {} / {} B in Rust (stale PTX or a layout change)", h[0], h[1],
            std::mem::size_of::<DdArgs>(), std::mem::size_of::<DdSeg>());
    ensure!(h[2] as usize == PARTS && h[3] as usize == KPS && h[4] as usize == MAXSEG && h[5] as usize == PREP_ROWS,
            "dense decode constants differ between the device and the host: {:?}", &h[2..6]);
    ensure!(h[6] as usize == std::mem::size_of::<NormRotArgs>() && h[7] as usize == std::mem::size_of::<L2PacedArgs>(),
            "D41xNormRotArgs / D41xL2PacedArgs are {} / {} B on the device, {} / {} B in Rust", h[6], h[7],
            std::mem::size_of::<NormRotArgs>(), std::mem::size_of::<L2PacedArgs>());
    let mut c: sys::CUdeviceptr = 0;
    unsafe {
        ensure!(sys::cuMemAlloc_v2(&mut c, CNT_LEN * 4) == sys::CUresult::CUDA_SUCCESS, "cuMemAlloc (dense decode counters)");
        ensure!(sys::cuMemsetD8_v2(c, 0, CNT_LEN * 4) == sys::CUresult::CUDA_SUCCESS, "memset (dense decode counters)");
        ensure!(sys::cuCtxSynchronize() == sys::CUresult::CUDA_SUCCESS, "sync (dense decode counters)");
    }
    if GEOM.set(c as u64).is_err() { unsafe { sys::cuMemFree_v2(c); } }
    Ok(())
}

/// One linear of a call: x [rows, K slice] -> y [rows, N slice] (`acc`: overwrite or add).
/// `pre`: x's rotated rows fp16(H128(x . suh)) were already written by x's producer (no prep for this linear);
/// `rot`: also write the NEXT linear's rotated input from this linear's stored output (the rot_out epilogue).
pub struct Call<'a> { pub w: &'a DenseWeight, pub x: &'a Act, pub y: &'a Act, pub acc: Accum, pub pre: Option<Pre>, pub rot: Option<RotTo> }
impl<'a> Call<'a> {
    pub fn new(w: &'a DenseWeight, x: &'a Act, y: &'a Act, acc: Accum) -> Call<'a> { Call { w, x, y, acc, pre: None, rot: None } }
}
/// fp16 rotated input rows already on the device: base (the linear's input column 0) and row stride (elements).
#[derive(Clone, Copy, Debug)]
pub struct Pre { pub xh: u64, pub ld: usize }
/// A rot_out target: the next linear's suh at THIS linear's output column 0, and where its rotated rows go.
#[derive(Clone, Copy, Debug)]
pub struct RotTo { pub suh: u64, pub xh: u64, pub ld: usize }

/// The suh a producer rotates with for this linear (at the linear's input column 0) and the linear's input width.
pub fn suh_of(w: &DenseWeight) -> Result<(u64, usize)> { let r = resolve(w, 1)?; Ok((r.suh, r.plan.kloc)) }
/// The width code of a linear (2..=8 integer K serve the decode class).
pub fn bits_of(w: &DenseWeight) -> Result<i32> { Ok(resolve(w, 1)?.bits) }

/// Can this class serve the call? (rows 1..=32, an integer width K in 2..=8)
pub fn eligible(w: &DenseWeight, rows: usize) -> bool {
    if !(1..=32).contains(&rows) { return false; }
    match resolve(w, rows) { Ok(r) => (2..=8).contains(&r.bits), Err(_) => false }
}

fn a256(b: usize) -> usize { (b + 255) & !255 }
fn ar_of(rows: usize) -> usize { if rows <= 8 { 8 } else if rows <= 16 { 16 } else if rows <= 24 { 24 } else { 32 } }

/// The GEMM entry serving `rows` rows at width K (`bits`).
pub fn entry(rows: usize, bits: i32) -> String { format!("d41x_dd_a{}_k{}", ar_of(rows), bits) }

/// Partitions of one segment of `units` units (the segment's own cut: PD1).
pub fn seg_parts(units: usize) -> usize { PARTS.min(units.max(1)) }

struct Geo { _u: usize, p: usize, xh: Vec<usize>, part: usize, ncb_all: usize, total: usize }
/// `pre[i]`: segment i's rotated rows come from its producer (no prep output in the scratch).
fn geo(rs: &[Resolved], rows: usize, pre: &[bool]) -> Geo {
    let (mut u, mut p, mut ncb_all, mut off, mut xh) = (0usize, 0usize, 0usize, 0usize, vec![]);
    for (i, r) in rs.iter().enumerate() {
        xh.push(off);
        if !pre.get(i).copied().unwrap_or(false) { off += a256(rows * r.nkb as usize * 16 * 2); }
        let us = r.plan.nblk * (r.nkb as usize / KPS);
        u += us;
        p += seg_parts(us);
        ncb_all += r.plan.nblk;
    }
    let part = off;
    off += a256((p + ncb_all) * rows * 128 * 4);
    Geo { _u: u, p, xh, part, ncb_all, total: off }
}

/// Byte offsets of each linear's prep output (fp16 [rows][K slice]) in the scratch of one call (the probe reads them).
pub fn xh_offsets(ws: &[&DenseWeight], rows: usize) -> Vec<usize> {
    let rs: Vec<Resolved> = ws.iter().filter_map(|w| resolve(w, rows).ok()).collect();
    geo(&rs, rows, &[]).xh
}
/// Scratch of one call over these linears (`rows` rows; every linear prepped: the upper bound).
pub fn scratch_bytes_group(ws: &[&DenseWeight], rows: usize) -> Result<usize> {
    let rs: Vec<Resolved> = ws.iter().map(|w| resolve(w, rows)).collect::<Result<_>>()?;
    Ok(geo(&rs, rows, &[]).total)
}
pub fn scratch_bytes(w: &DenseWeight, rows: usize) -> usize { scratch_bytes_group(&[w], rows).unwrap_or(0) }

/// Ring geometry: (slot stride, slots, trellis bytes per slot, dynamic smem).
fn ring(rows: usize, bits: i32) -> Result<(usize, usize, usize, usize)> {
    let wreg = KPS * 8 * 32 * bits as usize;
    let slot = (wreg + KPS * rows * 32 + 127) / 128 * 128;
    let stage_rows = if rows <= 16 { 16 } else if rows <= 24 { 24 } else { 32 };
    let stage = stage_rows * 132 * 4;
    let mut ns = if rows > 16 { SMEM_2CTA_DEEP.saturating_sub(stage) / slot } else { SMEM_2CTA.saturating_sub(stage) / slot };
    if ns < 3 { ns = SMEM_2CTA.saturating_sub(stage) / slot; }
    if ns < 3 { ns = SMEM_1CTA.saturating_sub(stage) / slot; }
    ensure!(ns >= 3, "dense decode ring: slot {slot} B leaves fewer than 3 slots");
    let ns = ns.min(NS_MAX);
    Ok((slot, ns, wreg, ns * slot + stage))
}

static CALLS: AtomicU64 = AtomicU64::new(0);
static LINEARS: AtomicU64 = AtomicU64::new(0);
static ROWS: AtomicU64 = AtomicU64::new(0);
static SAID: [AtomicBool; 4] = [AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false)];
/// Dense decode-class launches since the start (the engagement counter), the linears they carried, and their rows.
pub fn calls() -> u64 { CALLS.load(Ordering::Relaxed) }
pub fn linears() -> u64 { LINEARS.load(Ordering::Relaxed) }
pub fn rows() -> u64 { ROWS.load(Ordering::Relaxed) }

static PREPS: AtomicU64 = AtomicU64::new(0);
static PRE_SEGS: AtomicU64 = AtomicU64::new(0);
static ROT_SEGS: AtomicU64 = AtomicU64::new(0);
/// PD1 receipts: prep launches, linears whose rotated input came from its producer, linears that rotated their output.
pub fn preps() -> u64 { PREPS.load(Ordering::Relaxed) }
pub fn pre_linears() -> u64 { PRE_SEGS.load(Ordering::Relaxed) }
pub fn rot_linears() -> u64 { ROT_SEGS.load(Ordering::Relaxed) }

/// One call over 1..=MAXSEG linears with the same row count and width K. y is overwritten or added per linear. Each
/// linear's output bits equal its own 1-linear call (PD1: per-segment partitions).
pub fn run(l: &Launch, calls: &[Call], s: Scratch) -> Result<()> {
    ensure!(!calls.is_empty() && calls.len() <= MAXSEG, "dense decode: {} linears in one call (1..={MAXSEG})", calls.len());
    let rows = calls[0].x.rows;
    ensure!((1..=32).contains(&rows), "dense decode: {rows} rows (1..=32)");
    geometry()?;
    let cnt = counters()?;
    let rs: Vec<Resolved> = calls.iter().map(|c| resolve(c.w, rows)).collect::<Result<_>>()?;
    let bits = rs[0].bits;
    ensure!((2..=8).contains(&bits), "{}: dense decode needs an integer K (width code {bits})", calls[0].w.key);
    let pre: Vec<bool> = calls.iter().map(|c| c.pre.is_some()).collect();
    let g = geo(&rs, rows, &pre);
    ensure!(s.bytes >= g.total, "dense decode: scratch {} < {}", s.bytes, g.total);
    ensure!(g.ncb_all <= CNT_LEN, "dense decode: {} output blocks > the {CNT_LEN} persistent counters", g.ncb_all);
    let mut a = DdArgs::default();
    let (mut u, mut jb, mut p0, mut maxkb) = (0i32, 0i32, 0i32, 0i32);
    for (i, (c, r)) in calls.iter().zip(&rs).enumerate() {
        let p = &r.plan;
        ensure!(r.bits == bits, "dense decode: one call mixes widths ({} has code {}, {} has {bits})", c.w.key, r.bits, calls[0].w.key);
        ensure!(c.x.rows == rows && c.y.rows == rows, "{}: rows x {} y {} (call rows {rows})", c.w.key, c.x.rows, c.y.rows);
        ensure!(c.x.cols == p.kloc && c.y.cols == p.ncols, "{}: x cols {} (want {}), y cols {} (want {})", c.w.key, c.x.cols, p.kloc, c.y.cols, p.ncols);
        ensure!(c.y.dtype != ActDtype::Fp8E4m3, "{}: FP8 output", c.w.key);
        let us = p.nblk * (r.nkb as usize / KPS);
        ensure!(us as u64 * PARTS as u64 <= i32::MAX as u64, "dense decode: {us} units overflow the kernel's 32-bit partition math");
        let np = seg_parts(us) as i32;
        let (xh, xld) = match c.pre {
            Some(pr) => {
                ensure!(pr.xh != 0 && pr.xh % 16 == 0 && pr.ld % 8 == 0 && pr.ld >= p.kloc, "{}: pre-rotated rows {:#x} ld {} (16-B aligned, ld a multiple of 8 >= {})", c.w.key, pr.xh, pr.ld, p.kloc);
                (pr.xh, pr.ld as i32)
            }
            None => (s.ptr.0 + g.xh[i] as u64, p.kloc as i32),
        };
        let (rsuh, rxh, rld) = match c.rot {
            Some(rt) => {
                ensure!(c.acc == Accum::Overwrite && p.skip == 0 && p.ncols == p.nblk * 128, "{}: rot_out needs whole 128-blocks, no skip, no accumulate", c.w.key);
                ensure!(rt.suh != 0 && rt.suh % 8 == 0 && rt.xh != 0 && rt.xh % 8 == 0 && rt.ld % 4 == 0 && rt.ld >= p.ncols, "{}: rot_out target {:?}", c.w.key, rt);
                (rt.suh, rt.xh, rt.ld as i32)
            }
            None => (0, 0, 0),
        };
        let tb = 2 * tile_words(bits) as i64;   // tile bytes (u16 words)
        a.seg[i] = DdSeg {
            tr: r.tr + ((r.kb0 as i64 * r.nb_stride as i64 + r.nb0 as i64) * tb) as u64,
            x: c.x.ptr.0, suh: r.suh, svh: r.svh, out: c.y.ptr.0, xh, rsuh, rxh,
            kstride: r.nb_stride as i64 * tb,
            nkb: r.nkb, ncb: p.nblk as i32, skip: p.skip as i32, ncols: p.ncols as i32,
            ldx: c.x.ld as i32, xcode: act_code(c.x.dtype)?, ldo: c.y.ld as i32, ocode: act_code(c.y.dtype)?, accum: (c.acc == Accum::Add) as i32,
            u0: u, jb0: jb, p0, np, xld, pre: c.pre.is_some() as i32, rld, pad_: 0,
        };
        u += us as i32;
        jb += p.nblk as i32;
        p0 += np;
        if c.pre.is_none() { maxkb = maxkb.max(r.nkb / 8); }
    }
    ensure!(p0 as usize == g.p, "dense decode: partition count {p0} != {}", g.p);
    let (slot, ns, wreg, smem) = ring(rows, bits)?;
    a.part = s.ptr.0 + g.part as u64; a.cnt = cnt;
    a.nseg = calls.len() as i32; a.rows = rows as i32; a.u = u; a.ncb_all = jb; a.bits = bits;
    a.slot = slot as i32; a.ns = ns as i32; a.wreg = wreg as i32; a.maxkb = maxkb.max(1); a.p = g.p as i32;
    let mut pp = [&mut a as *mut DdArgs as *mut c_void];
    if maxkb > 0 {
        let kp = kernel("d41x_dd_prep")?;
        launch_ex(&kp, (maxkb as u32, rows.div_ceil(PREP_ROWS) as u32, calls.len() as u32), 32, 0, l, &mut pp).context("dense decode prep launch")?;
        PREPS.fetch_add(1, Ordering::Relaxed);
    }
    let name = entry(rows, bits);
    let (kg, gmax) = grid_for(&name, smem as u32)?;
    let grid = gmax.min(g.p as u32);
    launch_ex(&kg, (grid, 1, 1), THREADS, smem as u32, l, &mut pp).with_context(|| format!("dense decode GEMM launch ({name})"))?;
    CALLS.fetch_add(1, Ordering::Relaxed);
    LINEARS.fetch_add(calls.len() as u64, Ordering::Relaxed);
    ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    PRE_SEGS.fetch_add(pre.iter().filter(|&&b| b).count() as u64, Ordering::Relaxed);
    ROT_SEGS.fetch_add(calls.iter().filter(|c| c.rot.is_some()).count() as u64, Ordering::Relaxed);
    let ai = match ar_of(rows) { 8 => 0, 16 => 1, 24 => 2, _ => 3 };
    if crate::boot_report::debug() && !SAID[ai].swap(true, Ordering::Relaxed) {
        crate::logq::send(false, format!("[dsv41] EXL3 dense decode: {name} (stream-K, {} partitions, grid {grid}, ring {ns} x {slot} B) serves rows {}..={} at K = {bits}",
            g.p, [1, 9, 17, 25][ai], [8, 16, 24, 32][ai]));
    }
    Ok(())
}

/// Several linears with the same rows in as few launches as their widths allow (PD1: one launch per width, up to MAXSEG
/// linears each, every linear's bits those of its own call: a rank's per-group `wo_a`, `wq_a` + `wkv` (+ the compressor)
/// on the same input). A linear this class cannot serve (fractional K, >= 33 rows) runs alone through `dense`
/// (it must not carry `pre` / `rot`).
pub fn run_group(l: &Launch, calls: &[Call], s: Scratch) -> Result<()> {
    if calls.is_empty() { return Ok(()); }
    let rows = calls[0].x.rows;
    let mut by_bits: Vec<(i32, Vec<usize>)> = Vec::new();
    for (i, c) in calls.iter().enumerate() {
        if c.x.rows != rows || !eligible(c.w, c.x.rows) {
            ensure!(c.pre.is_none() && c.rot.is_none(), "{}: a producer-side rotation on a linear the decode class does not serve", c.w.key);
            super::dense::dense(l, c.w, c.x, c.y, c.acc, s)?;
            continue;
        }
        let b = resolve(c.w, c.x.rows)?.bits;
        match by_bits.iter_mut().find(|(k, _)| *k == b) { Some((_, v)) => v.push(i), None => by_bits.push((b, vec![i])) }
    }
    for (_, idx) in &by_bits {
        for ch in idx.chunks(MAXSEG) {
            let cs: Vec<Call> = ch.iter().map(|&i| { let c = &calls[i]; Call { w: c.w, x: c.x, y: c.y, acc: c.acc, pre: c.pre, rot: c.rot } }).collect();
            run(l, &cs, s)?;
        }
    }
    GROUPS.fetch_add(1, Ordering::Relaxed);
    Ok(())
}
static GROUPS: AtomicU64 = AtomicU64::new(0);
/// `run_group` calls since the start.
pub fn groups() -> u64 { GROUPS.load(Ordering::Relaxed) }

/// PD1: y = rmsnorm(x) . w (d41_rmsnorm's bits) and, per target, the target linear's rotated input rows from the stored
/// y (`targets` = (suh at y's column 0, destination, row stride)); one launch.
#[allow(clippy::too_many_arguments)]
pub fn normrot(l: &Launch, x: u64, x_ld: usize, w: u64, w_fmt: u32, y: u64, y_ld: usize, dim: usize, rows: usize, eps: f32, targets: &[(u64, u64, usize)]) -> Result<()> {
    let mut a = normrot_args(x, x_ld, w, w_fmt, y, y_ld, dim, rows, eps, targets)?;
    let k = kernel("d41x_dd_normrot")?;
    let mut pp = [&mut a as *mut NormRotArgs as *mut c_void];
    launch_ex(&k, (rows as u32, 1, 1), 256, 0, l, &mut pp).context("normrot launch")
}

/// `normrot`'s argument block (leg PA1's fused q / kv norm launch carries the same one).
#[allow(clippy::too_many_arguments)]
pub fn normrot_args(x: u64, x_ld: usize, w: u64, w_fmt: u32, y: u64, y_ld: usize, dim: usize, rows: usize, eps: f32, targets: &[(u64, u64, usize)]) -> Result<NormRotArgs> {
    ensure!(targets.len() <= 2 && dim % 128 == 0 && dim <= 6144 && rows > 0, "normrot: {} targets, dim {dim}", targets.len());
    let mut a = NormRotArgs { x, x_ld: x_ld as i64, w, y, y_ld: y_ld as i64, w_fmt, dim: dim as i32, rows: rows as i32, eps, inv_dim: 1.0 / dim as f32,
                              nt: targets.len() as i32, ..Default::default() };
    for (i, &(su, xh, ld)) in targets.iter().enumerate() {
        ensure!(su % 8 == 0 && xh % 8 == 0 && ld % 4 == 0 && ld >= dim, "normrot target {i}: suh {su:#x} xh {xh:#x} ld {ld}");
        a.suh[i] = su; a.xh[i] = xh; a.xld[i] = ld as i32;
    }
    Ok(a)
}

/// A weight's bytes as the paced prefetch walks them: (base, run, stride, count) = `count` runs of `run` bytes
/// `stride` apart in the order the call reads them (k order). A slice-resident (or In-sliced) trellis is one run.
pub fn weight_range(w: &DenseWeight) -> Result<(u64, u64, u64, u64)> {
    let r = resolve(w, 1)?;
    let tb = 2 * tile_words(r.bits) as u64;
    let run = r.nbn as u64 * tb;
    let stride = r.nb_stride as u64 * tb;
    let base = r.tr + (r.kb0 as u64 * r.nb_stride as u64 + r.nb0 as u64) * tb;
    let count = r.nkb as u64;
    Ok(if stride == run { (base, run * count, run * count, 1) } else { (base, run, stride, count) })
}

/// Cut ranges (base, run, stride, count) to `budget` bytes in order (whole runs, then one partial run).
pub fn cut_ranges(ranges: &[(u64, u64, u64, u64)], budget: u64) -> Vec<(u64, u64, u64, u64)> {
    let mut out = Vec::new();
    let mut left = budget;
    for &(p, run, stride, count) in ranges {
        if left < 16 || out.len() == PF_MAXR { break; }
        let run16 = run & !15;
        if run16 == 0 || count == 0 { continue; }
        let whole = (left / run16).min(count);
        if whole > 0 { out.push((p, run16, stride, whole)); left -= whole * run16; }
        if whole < count && left >= 16 && out.len() < PF_MAXR { out.push((p + whole * stride, left & !15, stride, 1)); left = 0; }
    }
    out
}

static PF_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static PF_BYTES: AtomicU64 = AtomicU64::new(0);
/// Paced prefetch launches the host built (a graph replay re-runs captured ones without counting) and their bytes.
pub fn pf_launches() -> u64 { PF_LAUNCHES.load(Ordering::Relaxed) }
pub fn pf_bytes() -> u64 { PF_BYTES.load(Ordering::Relaxed) }

/// PD1: one paced L2 prefetch launch (1 CTA x 128 threads, plain launch on `stream`) over `ranges` cut to `budget` bytes:
/// waves of 128 x 8 KiB every `wave_ns` after `delay_ns`. Returns the bytes it asks for (0 = nothing launched).
pub fn l2_paced(stream: u64, ranges: &[(u64, u64, u64, u64)], budget: u64, delay_ns: u64, wave_ns: u64) -> Result<u64> {
    ensure!(stream != 0, "l2_paced: the legacy default stream is not allowed (AGENTS 2.1)");
    let rs = cut_ranges(ranges, budget);
    if rs.is_empty() { return Ok(0); }
    let mut a = L2PacedArgs { ptr: [0; PF_MAXR], run: [0; PF_MAXR], stride: [0; PF_MAXR], count: [0; PF_MAXR], n: rs.len() as i32, pad_: 0, delay_ns, wave_ns };
    let mut bytes = 0u64;
    for (i, &(p, run, stride, count)) in rs.iter().enumerate() {
        ensure!(p % 16 == 0 && (count == 1 || stride % 16 == 0), "l2_paced: range {i} base {p:#x} stride {stride} (16-B aligned)");
        a.ptr[i] = p; a.run[i] = run; a.stride[i] = stride; a.count[i] = count;
        bytes += run * count;
    }
    let k = kernel("d41x_l2_paced")?;
    let mut pp = [&mut a as *mut L2PacedArgs as *mut c_void];
    let r = unsafe { sys::cuLaunchKernel(k.0, 1, 1, 1, 128, 1, 1, 0, stream as sys::CUstream, pp.as_mut_ptr(), std::ptr::null_mut()) };
    ensure!(r == sys::CUresult::CUDA_SUCCESS, "launch d41x_l2_paced: {r:?}");
    PF_LAUNCHES.fetch_add(1, Ordering::Relaxed);
    PF_BYTES.fetch_add(bytes, Ordering::Relaxed);
    Ok(bytes)
}

/// Scratch of `run_group` over these linears (the largest of its calls).
pub fn scratch_bytes_for_group(ws: &[&DenseWeight], rows: usize) -> usize {
    if rows == 0 || rows >= super::dense::WIDE_MIN_ROWS || ws.iter().any(|w| !eligible(w, rows)) {
        return ws.iter().map(|w| super::dense::scratch_bytes(w, rows)).max().unwrap_or(0);
    }
    ws.chunks(MAXSEG).map(|c| scratch_bytes_group(c, rows).unwrap_or(0)).max().unwrap_or(0)
        .max(ws.iter().map(|w| super::dense::scratch_bytes(w, rows)).max().unwrap_or(0))
}

/// T0 lever 3, for the forward owner: prefetch the trellis slice this rank's call of `w` will read (the first `max_bytes`
/// of it in k order) into L2 (evict_last), with one small launch (`d41x_l2_prefetch`). Under PDL it triggers its successor
/// first, issues the prefetches, then waits for its predecessor (so the successor still waits for the real producer).
/// Place it right before a kernel that leaves DRAM idle (attention core, a collective's wait) whose successor reads `w`.
/// Measured (`tools/dsv41_pk4/dd_bench.cu --prefetch`, `.13`, eager): a 4.1 MB `wq_a` call 31 -> 23 us, 6.6 MB 40 -> 30 us,
/// 13.1 MB 65 -> 55 us; the L2-resident call runs at ~240 GB/s (the kernel's own consume rate), so the gain is the ramp,
/// not the stream.
pub fn prefetch_weight(l: &Launch, w: &DenseWeight, max_bytes: usize) -> Result<()> {
    let r = resolve(w, 1)?;
    let tb = 2 * tile_words(r.bits) as u64;
    let run = r.nbn as u64 * tb;                                   // the slice's tiles at one k-step (contiguous)
    let stride = r.nb_stride as u64 * tb;
    let count = ((max_bytes as u64).div_ceil(run.max(1))).min(r.nkb as u64).max(1);
    let base = r.tr + (r.kb0 as u64 * r.nb_stride as u64 + r.nb0 as u64) * tb;
    let (run, stride, count) = if stride == run { (run * count, run * count, 1) } else { (run, stride, count) };
    let k = kernel("d41x_l2_prefetch")?;
    let (mut p, mut ru, mut sd, mut ct) = (base, run as i64, stride as i64, count as i32);
    let mut pp = [&mut p as *mut u64 as *mut c_void, &mut ru as *mut i64 as *mut c_void, &mut sd as *mut i64 as *mut c_void, &mut ct as *mut i32 as *mut c_void];
    let lines = (run.div_ceil(128) * count) as u32;
    launch_ex(&k, (lines.div_ceil(256).clamp(1, 96), 1, 1), 256, 0, l, &mut pp).context("L2 prefetch launch")
}

/// d41x_had128's scale: the fp32 rounding of the C literal 0.08838834764831845f (a shorter literal rounds to another f32).
#[allow(clippy::excessive_precision)]
pub const HAD128_SCALE: f32 = 0.088_388_347_648_318_45;
/// Host emulation of d41x_prep4 (+ the fp16 store) over one 128-block, bit for bit: x . s, the lane butterflies of
/// d41x_had128 in its order, x fp32(1/sqrt(128)), fp16 round-to-nearest-even (the probes check producer-side rotations
/// against it).
pub fn prep128_emul(x: &[f32], s: &[f32]) -> Vec<u16> {
    let mut v = [[0f32; 4]; 32];
    for l in 0..32 { for r in 0..4 { v[l][r] = x[l * 4 + r] * s[l * 4 + r]; } }
    for st in 0..2 {
        let len = 1usize << st;
        for l in 0..32 { for r in 0..4 { if r & len == 0 { let (a, b) = (v[l][r], v[l][r + len]); v[l][r] = a + b; v[l][r + len] = a - b; } } }
    }
    for st in 2..7 {
        let mask = 1usize << (st - 2);
        let old = v;
        for l in 0..32 { for r in 0..4 { let t = old[l ^ mask][r]; v[l][r] = if l & mask != 0 { t - old[l][r] } else { old[l][r] + t }; } }
    }
    let mut o = vec![0u16; 128];
    for l in 0..32 { for r in 0..4 { o[l * 4 + r] = half::f16::from_f32(v[l][r] * HAD128_SCALE).to_bits(); } }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dsv41_exl3_dd_layout_and_ring() {
        assert_eq!(std::mem::size_of::<DdSeg>(), 144);
        assert_eq!(std::mem::size_of::<DdArgs>(), 144 * MAXSEG + 16 + 40);
        assert_eq!(std::mem::size_of::<NormRotArgs>(), 104);
        assert_eq!(std::mem::size_of::<L2PacedArgs>(), 4 * 8 * PF_MAXR + 8 + 16);
        // two CTAs per SM for K5 at every row count, K6 / K8 at 1 row
        for rows in [1usize, 6, 12, 24, 32] {
            let (slot, ns, wreg, smem) = ring(rows, 5).unwrap();
            assert!(ns >= 3 && smem <= SMEM_2CTA && slot >= wreg, "rows {rows}: {slot} {ns} {smem}");
        }
        assert!(ring(1, 8).unwrap().3 <= SMEM_2CTA);
        assert_eq!(entry(1, 5), "d41x_dd_a8_k5");
        // the rot_out / normrot host emulation: a delta at column 0 with unit scale is the constant row 1/sqrt(128)
        let mut x = vec![0f32; 128]; x[0] = 1.0;
        let o = prep128_emul(&x, &vec![1f32; 128]);
        assert!(o.iter().all(|&b| b == half::f16::from_f32(HAD128_SCALE).to_bits()), "{o:?}");
        // and H128 is its own inverse up to the scale: two applications return x * 1/128 * 128 = x for exact inputs
        let x: Vec<f32> = (0..128).map(|i| ((i * 7) % 13) as f32 - 6.0).collect();
        let a: Vec<f32> = prep128_emul(&x, &vec![1f32; 128]).iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
        let b: Vec<f32> = prep128_emul(&a, &vec![1f32; 128]).iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
        assert!(x.iter().zip(&b).all(|(p, q)| (p - q).abs() < 2e-2), "{b:?}");
        assert_eq!(entry(17, 5), "d41x_dd_a24_k5");
        assert_eq!(entry(24, 6), "d41x_dd_a24_k6");
        assert_eq!(entry(25, 6), "d41x_dd_a32_k6");
        // P7.5: at rows 1..=24 every width bits = 2..=8 stays at 2 CTAs/SM (ns >= 3 inside SMEM_2CTA)
        for rows in 1..=24usize {
            for bits in 2..=8i32 {
                let (slot, ns, wreg, smem) = ring(rows, bits).unwrap();
                assert!(ns >= 3 && smem <= SMEM_2CTA && slot >= wreg, "rows {rows} K{bits}: slot {slot} ns {ns} smem {smem}");
            }
        }
    }

    /// PD1: a grouped call cuts each segment into its own partitions (the 1-segment cut), so the bits per segment are the
    /// segment's own call's; the call's partition count is the sum.
    #[test]
    fn dsv41_pd1_segment_partitions() {
        // TP2 wo_a group: 4096 -> 1024 (8 blocks x 64 k-steps of 4 = 128 units each); wq_a 5120 -> 1024 (8 x 80 = 640 units)
        assert_eq!(seg_parts(8 * (256 / KPS)), 96);
        assert_eq!(seg_parts(3), 3);
        assert_eq!(seg_parts(0), 1);
        // the device's start() with the segment's own (U, P) equals a 1-segment call's
        let start = |p: usize, u: usize, pp: usize| p * u / pp;
        for &us in &[128usize, 640, 80, 5] {
            let np = seg_parts(us);
            let v: Vec<usize> = (0..=np).map(|q| start(q, us, np)).collect();
            assert!(v.windows(2).all(|w| w[1] > w[0]), "an empty partition at {us} units");
            assert_eq!(*v.last().unwrap(), us);
        }
    }

    /// PD1: the paced prefetch's ranges are cut to the budget in order, whole runs first.
    #[test]
    fn dsv41_pd1_prefetch_cut() {
        let mib = 1u64 << 20;
        // contiguous: 10.5 MB of wo_a then 13.1 MB of wo_b, budget 16 MiB -> all of wo_a, the head of wo_b
        let r = cut_ranges(&[(0x1000, 10_485_760, 10_485_760, 1), (0x10_0000_0000, 13_107_200, 13_107_200, 1)], 16 * mib);
        assert_eq!(r.len(), 2, "{r:?}");
        assert_eq!(r[0], (0x1000, 10_485_760, 10_485_760, 1));
        assert_eq!(r[1], (0x10_0000_0000, 16 * mib - 10_485_760, 13_107_200, 1));
        // strided (a column slice: 320 runs of 4 KiB, 8 KiB apart), budget 1 MiB -> 256 whole runs
        let r = cut_ranges(&[(0x2000, 4096, 8192, 320)], mib);
        assert_eq!(r, vec![(0x2000, 4096, 8192, 256)]);
        // 16-byte floors, at most PF_MAXR ranges, nothing below 16 bytes
        let many: Vec<(u64, u64, u64, u64)> = (0..40).map(|i| (0x10000 * i as u64, 1000, 1000, 1)).collect();
        let r = cut_ranges(&many, 1 << 30);
        assert_eq!(r.len(), PF_MAXR);
        assert!(r.iter().all(|x| x.1 == 992));
        assert!(cut_ranges(&many, 15).is_empty());
    }
}
