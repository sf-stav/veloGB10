//! DSV4.1 native MXFP4 experts (leg KN2): the launches of band 30 (`kernels/gpu_dsv41/3x_mxfp4_*.cuh`)
//! over raw device pointers, on the caller's stream, optionally with programmatic dependent launch.
//! The kernel argument structs mirror the CUDA ones byte for byte (`D41MxFc1Args` 128 B,
//! `D41MxFc2Args` 112 B; static-asserted on both sides).

use crate::dsv41::weights::DevPtr;
use anyhow::{ensure, Context, Result};
use cudarc::driver::sys;
use std::ffi::c_void;

/// Rows served by one grouped launch (the grouped kernels take <= 32 rows; more rows run as
/// consecutive 32-row pieces, which give the same bits because every class computes the same way).
pub const GROUPED_MAX_ROWS: usize = 32;
/// Routes a grouped launch can plan (32 rows x top-6).
pub const MAX_ROUTES: usize = 192;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Fc1Args {
    pub w: u64, pub e_stride: u64, pub mt_stride: u64,
    pub xq: u64, pub xs: u64, pub ids: u64, pub rw: u64, pub hq: u64, pub hs: u64, pub dbg: u64,
    pub rows: i32, pub top_k: i32, pub e_first: i32, pub n_local: i32,
    pub nkb: i32, pub slice: i32, pub ldx: i32, pub ldxs: i32, pub ldh: i32, pub ldhs: i32,
    pub limit: f32, pub pad_: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Fc2Args {
    pub w: u64, pub e_stride: u64, pub mt_stride: u64,
    pub hq: u64, pub hs: u64, pub ids: u64, pub y: u64, pub route_out: u64,
    pub rows: i32, pub top_k: i32, pub e_first: i32, pub n_local: i32,
    pub nkb: i32, pub dim: i32, pub ldh: i32, pub ldhs: i32, pub ldy: i32,
    pub round_bf16: i32, pub accumulate: i32, pub pad_: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PlanWArgs {
    pub ids: u64, pub cnt: u64, pub off: u64, pub cur: u64, pub list: u64, pub tile_e: u64, pub tile_s: u64, pub n_tiles: u64,
    pub rows: i32, pub top_k: i32, pub e_first: i32, pub n_local: i32, pub max_tiles: i32, pub tt: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct WideArgs {
    pub w: u64, pub e_stride: u64, pub mt_stride: u64,
    pub aq: u64, pub as_: u64, pub list: u64, pub off: u64, pub tile_e: u64, pub tile_s: u64, pub n_tiles: u64,
    pub rw: u64, pub hq: u64, pub hs: u64, pub out2: u64,
    pub top_k: i32, pub nkb: i32, pub slice: i32, pub dim: i32,
    pub lda: i32, pub ldas: i32, pub ldh: i32, pub ldhs: i32,
    pub fc1: i32, pub round_bf16: i32, pub limit: f32, pub pad_: i32,
}
const _: () = assert!(std::mem::size_of::<Fc1Args>() == 128 && std::mem::size_of::<Fc2Args>() == 112
                      && std::mem::size_of::<PlanWArgs>() == 88 && std::mem::size_of::<WideArgs>() == 160);
/// Routes per token tile of the wide kernels (32_mxfp4_wide.cuh D41_MX_TT; PK2: 128).
pub const WIDE_TT: usize = 128;
/// PK3: 64-route tiles (entries `d41_mx_fc1_w64` / `d41_mx_fc2_w64`) while the mean routes per expert of a sub-chunk
/// (rows x top-k / experts) is at most this (32_mxfp4_wide.cuh D41_MX_TT64_MAX_RPE): short prompts and agent turns
/// (384 x top-6: up to 3,072 rows). Same bits for either tile (every MMA column is one token).
pub const WIDE_TT64_MAX_RPE: usize = 48;
/// The token tile of a wide sub-chunk of `rows` rows.
pub fn wide_tt(rows: usize, top_k: usize, n_experts: usize) -> usize {
    if rows * top_k <= WIDE_TT64_MAX_RPE * n_experts.max(1) { 64 } else { WIDE_TT }
}
/// Wide FC1: a CTA of 12 warps (one m-tile each) covers 96 inner values; wide FC2: 8 warps cover 128 output columns.
pub const WIDE_FC1_INNER: usize = 96;
pub const WIDE_FC1_THREADS: u32 = 384;
pub const WIDE_FC2_COLS: usize = 128;
pub const WIDE_FC2_THREADS: u32 = 256;
/// Rows of one wide sub-chunk at most (the per-route outputs are f32 [routes][dim]: 4,096 rows of
/// top-6 at dim 5120 = 503 MB); a bigger prefill chunk runs as consecutive sub-chunks.
pub const WIDE_CHUNK_MAX: usize = 16_384;

/// The entry kernels of band 30.
pub const FNS: &[&str] = &["d41_mx_act_quant", "d41_mx_act_quant_p", "d41_mx_fc1_g1", "d41_mx_fc1_g2", "d41_mx_fc1_g3", "d41_mx_fc1_g4",
                           "d41_mx_fc2_g1", "d41_mx_fc2_g2", "d41_mx_fc2_g3", "d41_mx_fc2_g4",
                           "d41_mx_plan_w", "d41_mx_fc1_w", "d41_mx_fc2_w", "d41_mx_fc1_w64", "d41_mx_fc2_w64", "d41_mx_combine_w"];

/// One MoE layer as this rank holds it, in the mx1 layout (`mxfp4_pack`): local expert e's w13 at
/// `w13 + e * e13`, its w2 at `w2 + e * e2`.
#[derive(Clone, Copy, Debug)]
pub struct PackedView {
    pub w13: DevPtr, pub e13: u64, pub mt13: u64,
    pub w2: DevPtr, pub e2: u64, pub mt2: u64,
    pub n_local: usize, pub e_first: usize,
    /// the layer's expert count (all ranks; the token-tile rule's denominator)
    pub n_experts: usize,
    pub dim: usize, pub i0: usize, pub ns: usize,
    /// whole experts on this rank (TP1 or EP): each expert's output is rounded to bf16 like the
    /// reference; an inner slice leaves an FP32 partial (the reduce sums the ranks)
    pub round_bf16: bool,
    /// the per-route outputs of the WIDE (prefill) path in bf16: `round_bf16`, or an inner slice with
    /// `--dsv41-prefill-bf16-partials on` (PK3, opt-in, changes bits). The grouped path never reads it.
    pub wide_bf16: bool,
    pub limit: f32,
}

/// Where the call's intermediates live (device byte offsets inside the caller's scratch).
#[derive(Clone, Copy, Debug, Default)]
pub struct ScratchPlan { pub xq: usize, pub xs: usize, pub hq: usize, pub hs: usize, pub bytes: usize, pub ldxs: usize, pub ldhs: usize }

fn al(x: usize) -> usize { x.div_ceil(256) * 256 }

/// The scratch of one wide sub-chunk of `rc` rows (> 32).
#[derive(Clone, Copy, Debug, Default)]
pub struct WidePlan {
    pub rc: usize, pub ldxs: usize, pub ldhs: usize, pub max_tiles: usize,
    pub xq: usize, pub xs: usize, pub hq: usize, pub hs: usize, pub out2: usize,
    pub cnt: usize, pub off: usize, pub cur: usize, pub list: usize, pub tile_e: usize, pub tile_s: usize, pub n_tiles: usize,
    pub bytes: usize,
}
impl WidePlan {
    pub fn of(rc: usize, top_k: usize, dim: usize, ns: usize, n_local: usize) -> WidePlan {
        let nr = rc * top_k;
        let ldxs = (dim / 32).div_ceil(4) * 4;
        let ldhs = (ns / 32).div_ceil(4) * 4;
        let max_tiles = nr.div_ceil(64) + n_local;   // PK3: the 64-route tile's bound (a call plans with its own tile)
        let mut o = 0usize;
        let mut take = |b: usize| { let at = o; o += al(b); at };
        let (xq, xs, hq, hs) = (take(rc * dim), take(rc * ldxs), take(nr * ns), take(nr * ldhs));
        let out2 = take(nr * dim * 4);
        let (cnt, off, cur, list) = (take(n_local * 4), take((n_local + 1) * 4), take(n_local * 4), take(nr * 4));
        let (tile_e, tile_s, n_tiles) = (take(max_tiles * 4), take(max_tiles * 4), take(4));
        WidePlan { rc, ldxs, ldhs, max_tiles, xq, xs, hq, hs, out2, cnt, off, cur, list, tile_e, tile_s, n_tiles, bytes: o }
    }
    /// The largest sub-chunk (<= rows, <= WIDE_CHUNK_MAX) whose scratch fits in `bytes`.
    pub fn fit(rows: usize, top_k: usize, dim: usize, ns: usize, n_local: usize, bytes: usize) -> Option<WidePlan> {
        let mut rc = rows.min(WIDE_CHUNK_MAX);
        while rc > GROUPED_MAX_ROWS {
            let p = WidePlan::of(rc, top_k, dim, ns, n_local);
            if p.bytes <= bytes { return Some(p); }
            rc = (rc / 2).max(GROUPED_MAX_ROWS);
        }
        None
    }
}

impl ScratchPlan {
    /// The scratch one call of `rows` rows needs (the grouped kernels run <= 32 rows at a time and
    /// reuse the same scratch for every piece).
    pub fn of(rows: usize, top_k: usize, dim: usize, ns: usize) -> ScratchPlan {
        let r = rows.clamp(1, GROUPED_MAX_ROWS);
        let ldxs = (dim / 32).div_ceil(4) * 4;
        let ldhs = (ns / 32).div_ceil(4) * 4;
        let xq = 0;
        let xs = xq + al(r * dim);
        let hq = xs + al(r * ldxs);
        let hs = hq + al(r * top_k * ns);
        let bytes = hs + al(r * top_k * ldhs);
        ScratchPlan { xq, xs, hq, hs, bytes, ldxs, ldhs }
    }
}

/// Launch on `stream` (raw CUstream as an integer), with programmatic dependent launch if asked;
/// the handle comes from the raw module copy (`gpu::raw_fn`), as every dsv41 PDL launch does.
pub unsafe fn launch(name: &str, stream: u64, pdl: bool, grid: u32, block: u32, params: &mut [*mut c_void]) -> Result<()> {
    let f = crate::dsv41::gpu::raw_fn(name).with_context(|| format!("gpu_dsv41 kernel {name} missing"))?;
    let mut attr = sys::CUlaunchAttribute {
        id: sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
        pad: [0; 4],
        value: sys::CUlaunchAttributeValue { programmaticStreamSerializationAllowed: 1 },
    };
    let mut cfg = sys::CUlaunchConfig {
        gridDimX: grid, gridDimY: 1, gridDimZ: 1, blockDimX: block, blockDimY: 1, blockDimZ: 1,
        sharedMemBytes: 0, hStream: stream as sys::CUstream, attrs: &mut attr, numAttrs: if pdl { 1 } else { 0 },
    };
    let r = sys::cuLaunchKernelEx(&mut cfg, f, params.as_mut_ptr(), std::ptr::null_mut());
    ensure!(r == sys::CUresult::CUDA_SUCCESS, "launch {name} (grid {grid}): {r:?}");
    Ok(())
}

/// Optional per-route outputs for the exactness gates (decode only).
#[derive(Clone, Copy, Debug, Default)]
pub struct Debug {
    /// bf16 [route][2][ns]: gate / up after fp4_gemm
    pub gate_up: u64,
    /// f32 [route][dim]: each route's w2 output (after its bf16 rounding when whole)
    pub route_out: u64,
}

/// One grouped call (rows <= 32): act_quant(x) -> FC1 -> FC2 + ordered combine into y.
/// `x` bf16 [rows][ldx]; `ids` i32 / `rw` f32 [rows * top_k]; `y` f32 [rows][ldy].
#[allow(clippy::too_many_arguments)]
pub fn grouped(stream: u64, pdl: bool, v: &PackedView, x: u64, ldx: usize, rows: usize, top_k: usize, ids: u64, rw: u64,
               y: u64, ldy: usize, accumulate: bool, scratch: u64, sp: &ScratchPlan, dbg: Debug) -> Result<()> {
    ensure!((1..=GROUPED_MAX_ROWS).contains(&rows) && rows * top_k <= MAX_ROUTES, "grouped MXFP4 call: {rows} rows x top-{top_k}");
    let nt = rows.div_ceil(8);
    let nkb1 = (v.dim / 32) as i32;
    // act_quant: one warp per (row, 32-group)
    {
        let (mut px, mut pldx, mut prows, mut pk) = (x, ldx as i32, rows as i32, v.dim as i32);
        let (mut pq, mut pldq, mut ps, mut plds) = (scratch + sp.xq as u64, v.dim as i32, scratch + sp.xs as u64, sp.ldxs as i32);
        let mut params: [*mut c_void; 8] = [&mut px as *mut _ as _, &mut pldx as *mut _ as _, &mut prows as *mut _ as _, &mut pk as *mut _ as _,
                                            &mut pq as *mut _ as _, &mut pldq as *mut _ as _, &mut ps as *mut _ as _, &mut plds as *mut _ as _];
        let warps = rows * v.dim / 32;
        unsafe { launch("d41_mx_act_quant", stream, pdl, (warps * 32).div_ceil(256) as u32, 256, &mut params)? };
    }
    let mut a = Fc1Args {
        w: v.w13.0, e_stride: v.e13, mt_stride: v.mt13, xq: scratch + sp.xq as u64, xs: scratch + sp.xs as u64, ids, rw,
        hq: scratch + sp.hq as u64, hs: scratch + sp.hs as u64, dbg: dbg.gate_up,
        rows: rows as i32, top_k: top_k as i32, e_first: v.e_first as i32, n_local: v.n_local as i32, nkb: nkb1, slice: v.ns as i32,
        ldx: v.dim as i32, ldxs: sp.ldxs as i32, ldh: v.ns as i32, ldhs: sp.ldhs as i32, limit: v.limit, pad_: 0,
    };
    let max_items = (rows * top_k).min(v.n_local) * (v.ns / 32);
    let mut pa: [*mut c_void; 1] = [&mut a as *mut _ as _];
    const FC1: [&str; 4] = ["d41_mx_fc1_g1", "d41_mx_fc1_g2", "d41_mx_fc1_g3", "d41_mx_fc1_g4"];
    const FC2: [&str; 4] = ["d41_mx_fc2_g1", "d41_mx_fc2_g2", "d41_mx_fc2_g3", "d41_mx_fc2_g4"];
    unsafe { launch(FC1[nt - 1], stream, pdl, max_items.max(1) as u32, 128, &mut pa)? };
    let mut b = Fc2Args {
        w: v.w2.0, e_stride: v.e2, mt_stride: v.mt2, hq: scratch + sp.hq as u64, hs: scratch + sp.hs as u64, ids, y, route_out: dbg.route_out,
        rows: rows as i32, top_k: top_k as i32, e_first: v.e_first as i32, n_local: v.n_local as i32, nkb: (v.ns / 32) as i32, dim: v.dim as i32,
        ldh: v.ns as i32, ldhs: sp.ldhs as i32, ldy: ldy as i32, round_bf16: v.round_bf16 as i32, accumulate: accumulate as i32, pad_: 0,
    };
    let mut pb: [*mut c_void; 1] = [&mut b as *mut _ as _];
    unsafe { launch(FC2[nt - 1], stream, pdl, (v.dim / 64) as u32, 128, &mut pb)? };
    Ok(())
}

/// One wide sub-chunk (rows > 32, <= wp.rc): act_quant -> route plan -> FC1 tiles -> FC2 tiles ->
/// ordered combine. Bit-identical per row to `grouped` (the same per-output arithmetic).
#[allow(clippy::too_many_arguments)]
pub fn wide(stream: u64, pdl: bool, v: &PackedView, x: u64, ldx: usize, rows: usize, top_k: usize, ids: u64, rw: u64,
            y: u64, ldy: usize, accumulate: bool, scratch: u64, wp: &WidePlan) -> Result<()> {
    ensure!(rows > 0 && rows <= wp.rc && top_k <= 8 && v.n_local <= 512, "wide MXFP4 call: {rows} rows (plan {}), top-{top_k}, {} local experts", wp.rc, v.n_local);
    let at = |o: usize| scratch + o as u64;
    {
        let (mut px, mut pldx, mut prows, mut pk) = (x, ldx as i32, rows as i32, v.dim as i32);
        let (mut pq, mut pldq, mut ps, mut plds) = (at(wp.xq), v.dim as i32, at(wp.xs), wp.ldxs as i32);
        let mut params: [*mut c_void; 8] = [&mut px as *mut _ as _, &mut pldx as *mut _ as _, &mut prows as *mut _ as _, &mut pk as *mut _ as _,
                                            &mut pq as *mut _ as _, &mut pldq as *mut _ as _, &mut ps as *mut _ as _, &mut plds as *mut _ as _];
        unsafe { launch("d41_mx_act_quant_p", stream, pdl, (rows * v.dim / 32 * 32).div_ceil(256) as u32, 256, &mut params)? };
    }
    let tt = wide_tt(rows, top_k, v.n_experts);
    let max_tiles = (rows * top_k).div_ceil(tt) + v.n_local;
    ensure!(max_tiles <= wp.max_tiles, "wide MXFP4 call: {max_tiles} tiles > plan {}", wp.max_tiles);
    let mut pa = PlanWArgs { ids, cnt: at(wp.cnt), off: at(wp.off), cur: at(wp.cur), list: at(wp.list), tile_e: at(wp.tile_e), tile_s: at(wp.tile_s),
                             n_tiles: at(wp.n_tiles), rows: rows as i32, top_k: top_k as i32, e_first: v.e_first as i32, n_local: v.n_local as i32,
                             max_tiles: max_tiles as i32, tt: tt as i32 };
    let mut pp: [*mut c_void; 1] = [&mut pa as *mut _ as _];
    unsafe { launch("d41_mx_plan_w", stream, pdl, 1, 1024, &mut pp)? };
    let mut f1 = WideArgs {
        w: v.w13.0, e_stride: v.e13, mt_stride: v.mt13, aq: at(wp.xq), as_: at(wp.xs), list: at(wp.list), off: at(wp.off),
        tile_e: at(wp.tile_e), tile_s: at(wp.tile_s), n_tiles: at(wp.n_tiles), rw, hq: at(wp.hq), hs: at(wp.hs), out2: at(wp.out2),
        top_k: top_k as i32, nkb: (v.dim / 32) as i32, slice: v.ns as i32, dim: v.dim as i32, lda: v.dim as i32, ldas: wp.ldxs as i32,
        ldh: v.ns as i32, ldhs: wp.ldhs as i32, fc1: 1, round_bf16: v.wide_bf16 as i32, limit: v.limit, pad_: 0,
    };
    let mut p1: [*mut c_void; 1] = [&mut f1 as *mut _ as _];
    let (fc1, fc2) = if tt == 64 { ("d41_mx_fc1_w64", "d41_mx_fc2_w64") } else { ("d41_mx_fc1_w", "d41_mx_fc2_w") };
    unsafe { launch(fc1, stream, pdl, (max_tiles * (v.ns / WIDE_FC1_INNER)) as u32, WIDE_FC1_THREADS, &mut p1)? };
    let mut f2 = WideArgs { w: v.w2.0, e_stride: v.e2, mt_stride: v.mt2, aq: at(wp.hq), as_: at(wp.hs), nkb: (v.ns / 32) as i32,
                            lda: v.ns as i32, ldas: wp.ldhs as i32, fc1: 0, ..f1 };
    let mut p2: [*mut c_void; 1] = [&mut f2 as *mut _ as _];
    unsafe { launch(fc2, stream, pdl, (max_tiles * (v.dim / WIDE_FC2_COLS)) as u32, WIDE_FC2_THREADS, &mut p2)? };
    {
        let (mut o2, mut pids, mut py, mut pldy) = (at(wp.out2), ids, y, ldy as i32);
        let (mut prows, mut ptk, mut pdim, mut pef, mut pnl, mut pacc) = (rows as i32, top_k as i32, v.dim as i32, v.e_first as i32, v.n_local as i32, accumulate as i32);
        let mut pbf = v.wide_bf16 as i32;
        let mut params: [*mut c_void; 11] = [&mut o2 as *mut _ as _, &mut pids as *mut _ as _, &mut py as *mut _ as _, &mut pldy as *mut _ as _,
                                             &mut prows as *mut _ as _, &mut ptk as *mut _ as _, &mut pdim as *mut _ as _, &mut pef as *mut _ as _,
                                             &mut pnl as *mut _ as _, &mut pacc as *mut _ as _, &mut pbf as *mut _ as _];
        let f = crate::dsv41::gpu::raw_fn("d41_mx_combine_w").context("d41_mx_combine_w missing")?;
        let mut attr = sys::CUlaunchAttribute {
            id: sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
            pad: [0; 4],
            value: sys::CUlaunchAttributeValue { programmaticStreamSerializationAllowed: 1 },
        };
        let mut cfg = sys::CUlaunchConfig {
            gridDimX: v.dim.div_ceil(256) as u32, gridDimY: rows as u32, gridDimZ: 1, blockDimX: 256, blockDimY: 1, blockDimZ: 1,
            sharedMemBytes: 0, hStream: stream as sys::CUstream, attrs: &mut attr, numAttrs: if pdl { 1 } else { 0 },
        };
        let r = unsafe { sys::cuLaunchKernelEx(&mut cfg, f, params.as_mut_ptr(), std::ptr::null_mut()) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "launch d41_mx_combine_w: {r:?}");
    }
    Ok(())
}
