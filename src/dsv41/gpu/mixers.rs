//! DSV4.1 GPU host side, leg KM: mHC mixers (hc 4, Sinkhorn 20, fp32), RMS norms (eps 1e-20), the router (fp32 gate GEMV,
//! sqrtsoftplus, noaux_tc top-k, bias / bias_vl, renorm + 1e-20, x route_scale) and the Engram gate.
//! Kernels: `kernels/gpu_dsv41/70_mixers.cuh` and `71`..`74` (band 70-79). Reference semantics: PLAN/DSV41_REF_MAP.md 2.7,
//! 3.2, 3.6, 3.7. Every entry takes ONE by-value argument struct, mirrored here (`#[repr(C)]`, same field order).
//!
//! # How the TP forward composes these (PLAN/DSV41_TP_DESIGN.md 3.5)
//! 1. `mix_partial` over the rank's K slice of `hc_fn` (+ the slice's sum of squares): 25 f32 per row per 512-k chunk into
//!    `part`; the TP leg copies/ships them as the rider of the sublayer reduce (any slot order, the SAME order on every rank).
//! 2. `mix_finish` sums the slots in slot order, applies `rsqrt(mean + 1e-20)`, runs Sinkhorn and writes one 24-f32 record
//!    per row (pre 0..3, post 4..7, comb 8..23). One rank's full mix equals the sum of N slices up to FP32 order.
//! 3. `hc_pre_norm` consumes a DELAYED pre (the previous sublayer's record; layer 0 a constant one-hot with stride 0).
//! 4. `hc_post` consumes the sublayer's own record (`post` at +4, `comb` at +8).

use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, DeviceRepr, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// Entry kernels of this band; the gpu_dsv41 loader (`super::load_module`) collects them.
pub const FNS: &[&str] = &["d41_mix_partial", "d41_mix_partial2", "d41_mix_partial4", "d41_mix_partial8", "d41_mix_finish", "d41_hc_pre_norm", "d41_hc_pre_norm_pw", "d41_q_kv_norm", "d41_hc_post", "d41_hc_post_f32", "d41_rmsnorm",
                           "d41_router_scores_r8", "d41_router_scores_r12", "d41_router_scores_r24", "d41_router_scores_r32", "d41_router_select", "d41_engram_gate",
                           "d41_router_tc_r8", "d41_router_tc_r16", "d41_router_tc_r32",
                           "d41_router_scores_wide", "d41_router_scores_wide4", "d41_router_scores_wide8"];

pub const HC: usize = 4;
pub const HC_MIX: usize = 24;
pub const HC_PART: usize = 25;
/// k per `mix_partial` block (one partial slot per chunk).
pub const PART_CHUNK: usize = 512;
/// rows per `mix_partial` row group (smem bound).
pub const PART_RG_MAX: usize = 24;
/// rows per `router_scores` launch.
pub const ROUTER_ROWS_MAX: usize = 32;

macro_rules! args { ($($n:ident),*) => { $(unsafe impl DeviceRepr for $n {})* } }

#[repr(C)] #[derive(Clone, Copy, Debug, Default)]
pub struct MixPartialArgs { pub x: u64, pub x_ld: i64, pub fn_: u64, pub fn_ld: i64, pub part: u64, pub chunk_stride: i64, pub row_stride: i64,
                            pub rank_part: u64, pub rp_row_stride: i64, pub counters: u64,
                            pub k0: i32, pub klen: i32, pub rows: i32, pub rg: i32 }
#[repr(C)] #[derive(Clone, Copy, Debug, Default)]
pub struct MixFinishArgs { pub part: u64, pub slot_stride: i64, pub row_stride: i64, pub scale: u64, pub base: u64, pub out: u64, pub out_ld: i64,
                           pub nslots: i32, pub rows: i32, pub inv_k: f32, pub eps_norm: f32, pub eps_hc: f32, pub sinkhorn_iters: i32 }
#[repr(C)] #[derive(Clone, Copy, Debug, Default)]
pub struct HcPreNormArgs { pub x: u64, pub x_ld: i64, pub pre: u64, pub pre_stride: i64, pub w: u64, pub y_norm: u64, pub yn_ld: i64, pub y_raw: u64, pub yr_ld: i64,
                           pub w_fmt: u32, pub dim: i32, pub rows: i32, pub eps: f32, pub inv_dim: f32 }
#[repr(C)] #[derive(Clone, Copy, Debug, Default)]
pub struct HcPostArgs { pub xs: u64, pub xs_ld: i64, pub resid: u64, pub r_ld: i64, pub post: u64, pub post_stride: i64, pub comb: u64, pub comb_stride: i64,
                        pub out: u64, pub o_ld: i64, pub dim: i32, pub rows: i32 }
/// leg PF1: `d41_hc_post32_args` (hc_post reading the reduced f32 frame, rounding it to bf16 itself)
#[repr(C)] #[derive(Clone, Copy, Debug, Default)]
pub struct HcPost32Args { pub p: HcPostArgs, pub xs32: u64, pub x32_ld: i64 }
#[repr(C)] #[derive(Clone, Copy, Debug, Default)]
pub struct RmsNormArgs { pub x: u64, pub x_ld: i64, pub w: u64, pub y: u64, pub y_ld: i64, pub w_fmt: u32, pub dim: i32, pub rows: i32, pub eps: f32, pub inv_dim: f32 }
#[repr(C)] #[derive(Clone, Copy, Debug, Default)]
pub struct RouterScoresArgs { pub x: u64, pub x_ld: i64, pub w: u64, pub w_ld: i64, pub scores: u64, pub s_ld: i64, pub w_fmt: u32, pub dim: i32, pub e: i32, pub rows: i32 }
#[repr(C)] #[derive(Clone, Copy, Debug, Default)]
pub struct RouterSelectArgs { pub scores: u64, pub s_ld: i64, pub bias: u64, pub bias_vl: u64, pub image_mask: u64, pub out_w: u64, pub out_i: u64, pub o_ld: i64,
                              pub bias_fmt: u32, pub e: i32, pub topk: i32, pub rows: i32, pub route_scale: f32, pub temp: f32, pub norm_topk: i32 }
#[repr(C)] #[derive(Clone, Copy, Debug, Default)]
pub struct EngramGateArgs { pub h: u64, pub h_ld: i64, pub kv: u64, pub kv_ld: i64, pub wq: u64, pub wk: u64, pub gate_on: u64, pub out: u64, pub o_ld: i64,
                            pub wq_fmt: u32, pub wk_fmt: u32, pub dim: i32, pub rows: i32, pub eps: f32, pub inv_dim: f32, pub inv_sqrt_dim: f32, pub clamp_min: f32 }
args!(MixPartialArgs, MixFinishArgs, HcPreNormArgs, HcPostArgs, HcPost32Args, RmsNormArgs, RouterScoresArgs, RouterSelectArgs, EngramGateArgs);

/// The model constants (config.json: `rms_norm_eps` 1e-20, `hc_eps` 1e-6, `hc_sinkhorn_iters` 20, `dim` 5120).
#[derive(Clone, Copy, Debug)]
pub struct MixConsts { pub dim: usize, pub norm_eps: f32, pub hc_eps: f32, pub sinkhorn_iters: i32, pub engram_clamp: f32 }
impl MixConsts {
    pub fn v41() -> Self { Self { dim: 5120, norm_eps: 1e-20, hc_eps: 1e-6, sinkhorn_iters: 20, engram_clamp: 1e-6 } }
    pub fn inv_stream(&self) -> f32 { 1.0 / (HC * self.dim) as f32 }
    pub fn inv_dim(&self) -> f32 { 1.0 / self.dim as f32 }
    pub fn inv_sqrt_dim(&self) -> f32 { (self.dim as f64).powf(-0.5) as f32 }
}

fn go<A: DeviceRepr>(dev: &Arc<CudaDevice>, name: &str, grid: (u32, u32), block: u32, a: A) -> Result<()> {
    let f = dev.get_func(super::MODULE, name).with_context(|| format!("{name} not loaded (call gpu::load_module first)"))?;
    unsafe { f.launch(LaunchConfig { grid_dim: (grid.0, grid.1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 }, (a,)) }.with_context(|| format!("launch {name}"))
}

/// Row-group size for `mix_partial` by row count. The result does not depend on it (a row's partial is the same in any
/// group, bit for bit); it only sets the grid. PK1's sweep (graph-timed, cold hc_fn, `.12`; TP4 slice / full K):
/// rows <= 2 one group; 3..8 rows groups of 2 (6 rows: 5.5 -> 3.7 us); above 8 groups of 4 (12 rows: 5.8 -> 4.4 us,
/// 24 rows: 6.2 -> 5.8 us).
pub fn pick_rg(rows: usize) -> usize { match rows { 0..=2 => rows.max(1), 3..=8 => 2, _ => 4 } }

/// Number of partial slots (chunks) a slice of `klen` produces.
pub fn partial_chunks(klen: usize) -> usize { klen.div_ceil(PART_CHUNK) }

/// hc_fn K-slice partials: `part[chunk * chunk_stride + row * row_stride + m]`. klen % 512 == 0, k0 % 4 == 0.
pub fn mix_partial(dev: &Arc<CudaDevice>, a: &MixPartialArgs) -> Result<()> {
    ensure!(a.klen as usize <= 40 * PART_CHUNK || a.counters == 0, "mix_partial: the rank reduction supports up to 40 chunks (klen <= 20480)");
    ensure!(a.klen as usize % PART_CHUNK == 0 && a.k0 % 4 == 0 && a.rg >= 1 && a.rg as usize <= PART_RG_MAX, "mix_partial: klen % 512, k0 % 4, 1 <= rg <= 24");
    go(dev, "d41_mix_partial", (partial_chunks(a.klen as usize) as u32, (a.rows as u32).div_ceil(a.rg as u32)), 128, *a)
}
pub fn mix_finish(dev: &Arc<CudaDevice>, a: &MixFinishArgs) -> Result<()> { go(dev, "d41_mix_finish", ((a.rows as u32).div_ceil(8), 1), 128, *a) }
pub fn hc_pre_norm(dev: &Arc<CudaDevice>, a: &HcPreNormArgs) -> Result<()> {
    ensure!(a.dim % 4 == 0 && a.dim <= 6144, "hc_pre_norm: dim % 4 == 0 and <= 6144");
    go(dev, "d41_hc_pre_norm", (a.rows as u32, 1), 256, *a)
}
pub fn hc_post(dev: &Arc<CudaDevice>, a: &HcPostArgs) -> Result<()> {
    ensure!(a.dim % 4 == 0, "hc_post: dim % 4 == 0");
    go(dev, "d41_hc_post", ((a.dim as u32).div_ceil(1024), a.rows as u32), 256, *a)
}
pub fn rmsnorm(dev: &Arc<CudaDevice>, a: &RmsNormArgs) -> Result<()> {
    ensure!(a.dim % 4 == 0 && a.dim <= 6144, "rmsnorm: dim % 4 == 0 and <= 6144");
    go(dev, "d41_rmsnorm", (a.rows as u32, 1), 256, *a)
}
/// `--dsv41-router-tf32 on`: the opt-in tf32 tensor-core router (default off = the fp32 CUDA-core router).
pub fn router_tf32_on() -> bool { crate::opts::on(crate::opt!("dsv41-router-tf32")) }

/// Whether these router weights CAN take the tf32 tensor-core path (bf16 / f16 gates, E % 8 == 0, dim % 1024 == 0: every
/// DSV4.1 router); it is taken only when asked (`--dsv41-router-tf32 on`). Its scores are the same bits whichever tc entry or
/// launch shape serves them, but NOT the fp32 path's bits.
pub fn router_tc(a: &RouterScoresArgs) -> bool { a.w_fmt != 3 && a.e % 8 == 0 && a.dim % 1024 == 0 }

/// The router-score launch for `n` <= 32 rows: (entry, grid, block). Default (and f32 gates): the fp32 r* entries (grid E,
/// E / 2 for r12, 128 threads), bit-identical to KM's kernels; `tf32` with tc-capable gates: `d41_router_tc_r*` (E / 8 CTAs
/// of 512 threads, 256 for tc_r32).
pub fn router_launch_mode(a: &RouterScoresArgs, n: usize, tf32: bool) -> (&'static str, u32, u32) {
    if tf32 && router_tc(a) {
        return match n { 0..=8 => ("d41_router_tc_r8", (a.e as u32) / 8, 512), 9..=16 => ("d41_router_tc_r16", (a.e as u32) / 8, 512),
                         _ => ("d41_router_tc_r32", (a.e as u32) / 8, 256) };
    }
    match n { 0..=8 => ("d41_router_scores_r8", a.e as u32, 128), 9..=12 => ("d41_router_scores_r12", (a.e as u32).div_ceil(2), 128),
              13..=24 => ("d41_router_scores_r24", a.e as u32, 128), _ => ("d41_router_scores_r32", a.e as u32, 128) }
}
/// Leg PP1: the wide fp32 router entries, (entry, experts per block, rows per block, threads). Every one computes the r*
/// entries' bits (same per-lane fmaf chain, same d41_rs32 tree, same quarter order); they differ only in how many experts and
/// rows a block shares its loads over. `WIDE` is the one the forward uses for calls of more than 32 rows.
pub const ROUTER_WIDE: [(&str, u32, u32, u32); 3] = [("d41_router_scores_wide", 8, 16, 256), ("d41_router_scores_wide4", 16, 16, 512), ("d41_router_scores_wide8", 16, 8, 256)];
pub const WIDE: usize = 1;
/// The launch of wide entry `v` (index into `ROUTER_WIDE`) for `rows` rows: (entry, grid, block). fp32 path only (f32 / bf16 /
/// f16 gates, dim % 1024 == 0).
pub fn router_wide_launch(a: &RouterScoresArgs, rows: usize, v: usize) -> (&'static str, (u32, u32), u32) {
    let (name, ep, rt, th) = ROUTER_WIDE[v];
    (name, ((a.e as u32).div_ceil(ep), (rows as u32).div_ceil(rt)), th)
}
/// Router scores of any row count through wide entry `v` (one launch).
pub fn router_scores_wide(dev: &Arc<CudaDevice>, a: &RouterScoresArgs, v: usize) -> Result<()> {
    ensure!(a.dim % 1024 == 0 && matches!(a.w_fmt, 1 | 2 | 3) && a.rows >= 1, "router_scores_wide: dim % 1024 == 0, weights bf16 / f16 / f32, rows >= 1");
    let (name, (gx, gy), block) = router_wide_launch(a, a.rows as usize, v);
    let f = dev.get_func(super::MODULE, name).with_context(|| format!("{name} not loaded (call gpu::load_module first)"))?;
    unsafe { f.launch(LaunchConfig { grid_dim: (gx, gy, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 }, (*a,)) }.with_context(|| format!("launch {name}"))
}
/// `router_launch_mode` with the process's `--dsv41-router-tf32` setting.
pub fn router_launch(a: &RouterScoresArgs, n: usize) -> (&'static str, u32, u32) { router_launch_mode(a, n, router_tf32_on()) }

/// Router scores (the process's `--dsv41-router-tf32` setting); more than 32 rows are split into launches of 32 (a row's
/// score never depends on the split or on the entry within a path).
pub fn router_scores(dev: &Arc<CudaDevice>, a: &RouterScoresArgs) -> Result<()> { router_scores_mode(dev, a, router_tf32_on()) }
/// Router scores on the fp32 path (`tf32 = false`) or the opt-in tf32 tensor-core path.
pub fn router_scores_mode(dev: &Arc<CudaDevice>, a: &RouterScoresArgs, tf32: bool) -> Result<()> {
    ensure!(a.dim % 1024 == 0 && matches!(a.w_fmt, 1 | 2 | 3), "router_scores: dim % 1024 == 0, weights bf16 / f16 / f32");
    let mut r0 = 0i32;
    while r0 < a.rows {
        let n = (a.rows - r0).min(ROUTER_ROWS_MAX as i32);
        let mut b = *a;
        b.rows = n;
        b.x += (r0 as i64 * a.x_ld * 2) as u64;
        b.scores += (r0 as i64 * a.s_ld * 4) as u64;
        let (name, grid, block) = router_launch_mode(a, n as usize, tf32);
        go(dev, name, (grid, 1), block, b)?;
        r0 += n;
    }
    Ok(())
}
pub fn router_select(dev: &Arc<CudaDevice>, a: &RouterSelectArgs) -> Result<()> {
    ensure!(a.e >= 1 && a.e <= 512 && a.topk >= 1 && a.topk <= 8 && matches!(a.bias_fmt, 2 | 3), "router_select: E <= 512, topk <= 8, bias f32 / f16");
    go(dev, "d41_router_select", (a.rows as u32, 1), 128, *a)
}
pub fn engram_gate(dev: &Arc<CudaDevice>, a: &EngramGateArgs) -> Result<()> {
    ensure!(a.dim % 4 == 0 && a.dim <= 6144, "engram_gate: dim % 4 == 0 and <= 6144");
    go(dev, "d41_engram_gate", (HC as u32, a.rows as u32), 256, *a)
}

#[cfg(test)]
#[path = "mix_ref.rs"]
mod mix_ref;
#[cfg(test)]
#[path = "mix_tests.rs"]
mod mix_tests;
#[cfg(test)]
#[path = "pp1_tests.rs"]
mod pp1_tests;
#[cfg(test)]
#[path = "pa1_tests.rs"]
mod pa1_tests;
