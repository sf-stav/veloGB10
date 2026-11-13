//! FP4L — the FP4-Local weight kinds inside the EXL3 trunk (PLAN/FP4L_DESIGN.md).
//!
//! `nvidia/Qwen3.8-Flash-Next-FP4-Local` is the same qwen4_exp network the EXL3 trunk serves; only
//! its weight storage differs. This module adds that storage to the trunk without touching any
//! trellis path:
//!
//! * `FxW` rides on a `Quad` (`Quad::fx`): `Mx8` (MXFP8: E4M3 codes + UE8M0 per-32 scales, the
//!   checkpoint's own bytes) for every dense projection, `F16` for the checkpoint's bf16 linears
//!   (lm_head, the MTP layer) and the opt-in quantized draft head. The trellis fields of such a
//!   Quad are 1-element placeholders; every chain dispatcher (`exl3_chain_impl`, `exl3_chain_pair`,
//!   `chain_wide_raw`, `xtp::chain_rowpar`, `wp22r1_chain`) branches here FIRST, so no trellis
//!   kernel ever sees an FP4L Quad.
//! * `FxMoe` rides on a `MoeLayer` (`MoeLayer::fx`): the NVFP4 routed experts, stacked per layer
//!   (EP: this rank's local experts), csf scales decoded at load. The router, top-k, route tables,
//!   shared expert and combine are the trunk's own (unchanged); only the expert GEMMs branch here.
//! * `PleFx` rides on `PleDev` (`PleDev::fx`): the NVFP4 n-gram table rows (80 B nibbles + 10 B E4M3
//!   scales per 160-value row, one global multiplier), fetched/decoded on the host like the trellis rows.
//!
//! Numerics (receipts: `--probe-fp4l`): weights are served AS STORED. MXFP8 decode/verify GEMMs scale
//! each 32-k group's fp32 MMA partial by its power of two (exact); NVFP4 e2m1*e4m3 is exact in f16 and
//! the global multiplier is applied to the fp32 accumulator. Prefill (m > 16) dequantizes MXFP8 to f16
//! for cuBLASLt (56,179 sub-2^-14 values of the pack round to the nearest f16 subnormal there).
use super::*;
use crate::fp4l::pack::{Fp4lPack, LinKind};

/// The kernel module (kernels/gpu_fp4l.cu, embedded as ptx_embed::GPU_FP4L).
pub(super) const XF_MODULE: &str = "gpu_fp4l";
pub(super) const XF_FNS: &[&str] = &[
    "kernel_build_id", "xf_mx8_gemm", "xf_f16w_gemm", "xf_nv4_moe", "xf_nv4_moe_i",
    "xf_nv4_tiles_m1", "xf_nv4_tiles_m2", "xf_nv4_tiles_m4",
    "xf_nv4_tiles_m1_i", "xf_nv4_tiles_m2_i", "xf_nv4_tiles_m4_i",
    "xf_mx8_dequant", "xf_nv4_dequant", "xf_e2m1_selftest", "xf_mx8_wide", "xf_f16_to_bf16_scaled",
    // precision levers (§3a): stored-bf16 GEMMs + the trunk-kernel twins
    "xf_bf16w_gemm", "xf_bf16_wide", "xf_hc_norm_f", "xf_hc_inj_norm_f", "xf_pf_hc_inj_norm_f", "xf_hc_fuse_bf",
    "xf_hc_mix_bf", "xf_pf_hc_mix_bf", "xf_embed_resid_bf", "xf_sg_gate", "xf_ple_norm_f16_f", "xf_ple_norm_f32_f",
    "xf_ple_norm_hh_f", "xf_ple_conv_f", "xf_ple_conv_chunk_f", "xf_ple_conv_chunk_seg_f",
    "xf_mx8_dequant_bf16", "xf_copy_u16", "xf_b32_to_rm", "xf_hc_fuse_b32", "xf_mx8_tc", "xf_bf16_tc", "xf_nv4_tc", "xf_nv4_tc_i",
    "xf_dhs_screen", "xf_dhs_select", "xf_dhs_rescore", "xf_tr_gemm", "xf_tr_combine", "xf_router_topk",
    "xf_mx8_gemm_x2",
];

/// Launch an entry of the gpu_fp4l module on the trunk's stream (the xqlaunch! contract minus PDL:
/// every xf_* entry opens with griddepcontrol.wait/launch_dependents, so a PDL-launched neighbour
/// still orders correctly against it).
macro_rules! xflaunch {
    ($l:expr, $name:expr, $grid:expr, $block:expr, $smem:expr, ($($arg:expr),* $(,)?)) => {{
        let f = $l.dev.get_func(XF_MODULE, $name)
            .ok_or_else(|| anyhow::anyhow!("kernel {} missing from {} (FP4L module not loaded?)", $name, XF_MODULE))?;
        unsafe {
            f.clone().launch_on_stream($l.stream, LaunchConfig { grid_dim: $grid, block_dim: $block, shared_mem_bytes: $smem },
                                       ($($arg),*))
                .with_context(|| format!("launch {}", $name))
        }
    }};
}

/// Load the gpu_fp4l module (idempotent) and run its KERNEL_BUILD_ID handshake.
pub(super) fn load_module(dev: &Arc<CudaDevice>) -> Result<()> {
    if dev.get_func(XF_MODULE, "xf_mx8_gemm").is_none() {
        dev.load_ptx(Ptx::from_src(crate::ptx_embed::GPU_FP4L), XF_MODULE, XF_FNS)
            .context("load gpu_fp4l module")?;
    }
    crate::gpu::GpuModel::assert_kernel_build_id(dev, XF_MODULE).context("FP4L kernel handshake (gpu_fp4l)")
}

/// `--fp4l-e2m1 cvt|int` (default cvt): the E2M1 decode in the NVFP4 expert kernels — the SM12x
/// `cvt.rn.f16x2.e2m1x2` (one instruction per byte) or the exact integer twin (A/B, fallback).
pub(super) fn e2m1_int() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::var(crate::opt!("fp4l-e2m1")).as_deref() == Ok("int"))
}

/// One FP4L linear's device weight (see the module doc).
pub(crate) enum FxW {
    /// MXFP8: E4M3 codes + UE8M0 per-32 scales in the fragment-native F64 layout (`mx8_frag`: a byte
    /// permutation of the checkpoint's [n][k] / [n][k/32] arrays, n padded to a multiple of 16 with zero rows).
    Mx8 { w: CudaSlice<u8>, s: CudaSlice<u8> },
    /// Plain f16 [n][k] holding the checkpoint's bf16 values: each tensor is stored as f16(w * 2^k)
    /// with one power-of-two shift k chosen at load (bf16 has 8 significant bits, f16 11: a value is
    /// exact when its scaled magnitude lies in [2^-17, 2^16)), and the GEMMs scale their fp32 result by
    /// `osc` = 2^-k (exact). Values outside the shifted range round and are counted at load.
    /// (--fp4l-lm-head f16 / --fp4l-plain f16.)
    F16 { w: CudaSlice<u16>, osc: f32 },
    /// The checkpoint's bf16 bits AS STORED [n][k] (--fp4l-lm-head bf16 / --fp4l-plain exact, the
    /// defaults): xf_bf16w_gemm / xf_bf16_wide split each f16 activation into two exact bf16 halves,
    /// so every product is exact in fp32. Same bytes as F16; two MMAs per f16 one.
    Bf16 { w: CudaSlice<u16> },
    /// --fp4l-mtp-sidecar (draft-only): an EXL3 trellis linear of an external MTP-head sidecar, served next to the
    /// FP4L trunk. Decode / verify (m <= 16): exl3_had_suh -> xf_tr_gemm (FP4L-owned trellis decode, `ks` K-slabs,
    /// fp32 partials in `scr`) -> xf_tr_combine -> exl3_had_svh. Prefill widths: cuBLASLt on `weff`, the effective
    /// f16 weight [n][k] = (diag(suh) H Wq H diag(svh))^T built on the host at load (one rounding of the fp32 product).
    Trellis { tr: CudaSlice<u16>, suh: CudaSlice<u16>, svh: CudaSlice<u16>, bits: usize, ks: usize, scr: CudaSlice<f32>, weff: CudaSlice<u16> },
}

impl FxW {
    pub(super) fn kind(&self) -> &'static str {
        match self { FxW::Mx8 { .. } => "mxfp8", FxW::F16 { .. } => "f16", FxW::Bf16 { .. } => "bf16", FxW::Trellis { .. } => "trellis" }
    }
    pub(super) fn bytes(&self) -> usize {
        match self {
            FxW::Mx8 { w, s } => w.len() + s.len(),
            FxW::F16 { w, .. } | FxW::Bf16 { w } => 2 * w.len(),
            // the bytes a decode / verify pass reads (the prefill copy `weff` is read only at prefill widths)
            FxW::Trellis { tr, suh, svh, .. } => 2 * (tr.len() + suh.len() + svh.len()),
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Precision levers (PLAN/FP4L_DESIGN.md §3a). Each resolves once per process; all ride TpConfig, so
// every rank of a TP session agrees. The full-precision end is the provisional default of each
// (owner ruling 2026-10-29: "no conversions"; final defaults after the A/B).
// ------------------------------------------------------------------------------------------------

/// `--fp4l-plain exact|f16` (default exact): the bf16 tensors the trunk computes with itself.
pub(crate) fn plain_exact() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::var(crate::opt!("fp4l-plain")).as_deref() != Ok("f16"))
}

/// `--fp4l-norms f32|f16` (default f32): the pack's fp32 norm weights.
pub(crate) fn norms_f32() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::var(crate::opt!("fp4l-norms")).as_deref() != Ok("f16"))
}

/// `--fp4l-lm-head bf16|f16|mxfp8` (default bf16).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeadPrec { Bf16, F16, Mx8 }
pub(crate) fn head_prec() -> HeadPrec {
    static V: std::sync::OnceLock<HeadPrec> = std::sync::OnceLock::new();
    *V.get_or_init(|| match crate::opts::var(crate::opt!("fp4l-lm-head")).as_deref() {
        Ok("f16") => HeadPrec::F16,
        Ok("mxfp8") => HeadPrec::Mx8,
        _ => HeadPrec::Bf16,
    })
}

/// Process-wide: an FP4-Local pack with --fp4l-plain exact is loaded, so every Scratch built from here on
/// carries the stored-bf16 GEMM buffers (Scratch::new has no model handle).
static FX_SCRATCH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub(super) fn set_fx_scratch() { FX_SCRATCH.store(true, std::sync::atomic::Ordering::SeqCst); }
pub(super) fn fx_scratch_on() -> bool { FX_SCRATCH.load(std::sync::atomic::Ordering::SeqCst) }

/// The model-level precision context of an FP4-Local pack.
pub(crate) struct FxCtx {
    /// --fp4l-plain exact
    pub plain: bool,
    /// --fp4l-norms f32
    pub norms: bool,
    /// whole-device co-residency of xf_hc_fuse_bf (its grid barrier needs the whole grid resident); 0 = the
    /// fused twin is unavailable and the exact mixer runs as three launches (down GEMM, silu, up GEMM)
    pub hc_cap: u32,
}

/// The per-Scratch buffers of the stored-bf16 plain GEMMs at m <= 16 (see Scratch::fxs).
pub(crate) struct FxScratch {
    pub ws_r: CudaSlice<f32>,
    pub cnt_r: CudaSlice<u32>,
    pub ws_p: CudaSlice<f32>,
    pub cnt_p: CudaSlice<u32>,
    pub ws_h: CudaSlice<f32>,
    pub cnt_h: CudaSlice<u32>,
    /// the shared-gate channels of the m rows [w][h]; columns >= 6 stay zero (xf_sg_gate)
    pub xg: CudaSlice<u16>,
}

impl FxScratch {
    pub(crate) fn new(dev: &Arc<CudaDevice>, cfg: &crate::qwen::Config, w: usize) -> Result<Self> {
        Self::with_dims(dev, cfg.hidden_size, cfg.hc_count.max(1), cfg.hc_lowrank, cfg.num_experts.max(2), w)
    }
    pub(crate) fn with_dims(dev: &Arc<CudaDevice>, h: usize, hcn: usize, lr: usize, ne: usize, w: usize) -> Result<Self> {
        let rw = h * hcn;
        let rows = w.max(1);
        let ws = |k: usize, n: usize| if k % 32 == 0 && k > 0 { bf_ks(k, n) as usize * rows * n } else { 1 };
        let tiles = |n: usize| n.div_ceil(128).max(1);
        // explicit zeros (AGENTS §2.2): the fixup counters must start at 0; xg's columns >= 6 must stay 0
        let z32 = |n: usize| -> Result<CudaSlice<u32>> { Ok(dev.htod_sync_copy(&vec![0u32; n.max(1)])?) };
        Ok(FxScratch {
            ws_r: dev.alloc_zeros::<f32>(ws(h, ne).max(1))?,
            cnt_r: z32(tiles(ne))?,
            ws_p: dev.alloc_zeros::<f32>(ws(2560, 10240).max(ws(2560, 2560)).max(1))?,
            cnt_p: z32(tiles(10240))?,
            ws_h: dev.alloc_zeros::<f32>(if lr > 0 { ws(rw, lr).max(ws(lr, rw)).max(HCF_WS_PER_LR * lr) } else { 1 })?,
            cnt_h: z32(tiles(rw.max(lr)))?,
            xg: dev.htod_sync_copy(&vec![0u16; rows * h])?,
        })
    }
}

/// Split-K factor of a stored-bf16 plain GEMM at m <= 16: a pure function of (K, N) (batch invariance).
/// The smallest divisor d of K/32 that leaves >= 2 groups per slab and puts >= 192 CTAs on the device,
/// else the largest divisor leaving >= 2 groups (1 when K/32 < 2); N >= 192 tiles runs d = 1.
pub(crate) fn bf_ks(k: usize, n: usize) -> u32 {
    let tiles = n.div_ceil(128).max(1);
    let g = (k / 32).max(1);
    if tiles >= 192 || g < 2 { return 1; }
    let divs: Vec<usize> = (1..=g).filter(|d| g % d == 0 && g / d >= 2).collect();
    for &d in &divs { if tiles * d >= 192 { return d as u32; } }
    *divs.last().unwrap_or(&1) as u32
}

/// y[m, n] = x[m, k] . W[n, k]^T on the STORED bf16 bits of a plain trunk tensor (`w` = bf16 [n][k]).
/// m <= 16: xf_bf16w_gemm (split-K fixup on `cnt`, slabs in `ws`; batch-invariant); m > 16: xf_bf16_tc.
/// Output f16 at `y` or fp32 at `yf` (exactly one of them non-zero).
#[allow(clippy::too_many_arguments)]
pub(super) fn bf_gemm(l: &Launcher, w: &CudaSlice<u16>, x: u64, y: u64, yf: u64, m: usize, k: usize, n: usize,
                      ws: &CudaSlice<f32>, cnt: &CudaSlice<u32>) -> Result<()> {
    anyhow::ensure!(k % 32 == 0 && n % 2 == 0 && m >= 1 && w.len() >= n * k,
                    "FP4L bf16 GEMM: K {k} must be a multiple of 32, N {n} even, m {m} >= 1, weight {} >= N*K", w.len());
    anyhow::ensure!((y == 0) != (yf == 0), "FP4L bf16 GEMM: exactly one output (f16 y or fp32 yf)");
    crate::metrics::fp4l_gemm("bf16", m);
    if m > 16 {
        return xflaunch!(l, "xf_bf16_tc", ((n.div_ceil(128) * m.div_ceil(128)) as u32, 1, 1), (256, 1, 1), 0,
                         (w, x, y, yf, m as i32, k as i32, n as i32));
    }
    let ks = bf_ks(k, n) as usize;
    let tiles = n.div_ceil(128);
    let cntp = if ks > 1 {
        anyhow::ensure!(ks * m * n <= ws.len() && tiles <= cnt.len(),
                        "FP4L bf16 GEMM scratch too small: ws {} < {} or counters {} < {tiles}", ws.len(), ks * m * n, cnt.len());
        *cnt.device_ptr() as u64
    } else { 0 };
    xflaunch!(l, "xf_bf16w_gemm", (tiles as u32, ks as u32, 1), (256, 1, 1), 0,
              (w, x, y, yf, ws, cntp, m as i32, k as i32, n as i32, ks as i32))
}

/// The stored-bf16 GEMM at prefill row counts (any m >= 1; xf_bf16_tc, no split-K scratch): every prefill
/// site of a trunk-held bf16 weight (hc mixers, router, n-gram projections) — each output row is its own MMA
/// row, so a row block computed alone (TP sequence-parallel) carries the full chunk's bits.
#[allow(clippy::too_many_arguments)]
pub(super) fn bf_wide(l: &Launcher, w: &CudaSlice<u16>, x: u64, y: u64, yf: u64, m: usize, k: usize, n: usize) -> Result<()> {
    anyhow::ensure!(k % 32 == 0 && n % 2 == 0 && m >= 1 && w.len() >= n * k,
                    "FP4L bf16 wide GEMM: K {k} must be a multiple of 32, N {n} even, m {m} >= 1, weight {} >= N*K", w.len());
    anyhow::ensure!((y == 0) != (yf == 0), "FP4L bf16 wide GEMM: exactly one output (f16 y or fp32 yf)");
    crate::metrics::fp4l_gemm("bf16", m.max(17));
    xflaunch!(l, "xf_bf16_tc", ((n.div_ceil(128) * m.div_ceil(128)) as u32, 1, 1), (256, 1, 1), 0,
              (w, x, y, yf, m as i32, k as i32, n as i32))
}

/// Whole-device co-residency of a gpu_fp4l entry at block 256 (the grid-barrier twin's guard): the PTX is
/// loaded a second time through the raw driver API (cudarc keeps CudaFunction's handle private — the
/// hc_fuse_co_residency precedent). Any failure returns 0 (the caller then never launches the barrier kernel).
pub(super) fn xf_co_residency(name: &str) -> u32 {
    use cudarc::driver::sys;
    let Ok(ptx_c) = std::ffi::CString::new(crate::ptx_embed::GPU_FP4L) else { return 0; };
    let mut module: sys::CUmodule = std::ptr::null_mut();
    let r = unsafe { sys::cuModuleLoadDataEx(&mut module, ptx_c.as_ptr() as *const std::ffi::c_void, 0,
                                             std::ptr::null_mut(), std::ptr::null_mut()) };
    if r != sys::CUresult::CUDA_SUCCESS { return 0; }
    let Ok(cname) = std::ffi::CString::new(name) else { return 0; };
    let mut f: sys::CUfunction = std::ptr::null_mut();
    if unsafe { sys::cuModuleGetFunction(&mut f, module, cname.as_ptr()) } != sys::CUresult::CUDA_SUCCESS { return 0; }
    let mut per_sm: std::ffi::c_int = 0;
    if unsafe { sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut per_sm, f, 256, 0) } != sys::CUresult::CUDA_SUCCESS { return 0; }
    let mut dh: sys::CUdevice = 0;
    if unsafe { sys::cuDeviceGet(&mut dh, 0) } != sys::CUresult::CUDA_SUCCESS { return 0; }
    let mut sms: std::ffi::c_int = 0;
    if unsafe { sys::cuDeviceGetAttribute(&mut sms, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dh) }
        != sys::CUresult::CUDA_SUCCESS { return 0; }
    (per_sm.max(0) as u32).saturating_mul(sms.max(0) as u32)
}

// ---- trunk-kernel twins (launch helpers; raw device pointers, the trunk sites own the buffers) ----

/// The exact mixer on the B32 down / up (xf_hc_fuse_b32): ONE launch of `g` co-resident CTAs (the caller sized g
/// within the kernel's measured co-residency); `ws` >= HCF_WS_PER_LR * lr fp32; `bar` TWO zeroed ints (count,
/// generation).
#[allow(clippy::too_many_arguments)]
pub(super) fn hc_fuse_b32(l: &Launcher, dd: u64, uu: u64, hn: u64, down: &CudaSlice<u16>, up: &CudaSlice<u16>, ws: &CudaSlice<f32>,
                          bar: u64, hcn: f32, lr: usize, rw: usize, g: u32, m: usize) -> Result<()> {
    anyhow::ensure!((1..=16).contains(&m) && rw % 1024 == 0 && lr % 32 == 0 && lr <= 480 && ws.len() >= HCF_WS_PER_LR * lr && g >= 1,
                    "FP4L xf_hc_fuse_b32: m {m}, rw {rw}, lr {lr}, ws {} outside the contract", ws.len());
    crate::metrics::fp4l_twin("hc_mixer");
    xflaunch!(l, "xf_hc_fuse_b32", (g, 1, 1), (256, 1, 1), 0,
              (dd, uu, hn, down, up, ws, bar, hcn, lr as i32, rw as i32, m as i32))
}
/// fp32 scratch per mixer rank of xf_hc_fuse_b32: KS = 32 slabs x 16 rows of partials, then 16 rows of s.
pub(crate) const HCF_WS_PER_LR: usize = 33 * 16;
/// The CTA count of xf_hc_fuse_b32: 80 (one item per warp in both phases at rw 10240 / lr 320) within the margined
/// co-residency `cap` (0 = the kernel is not co-resident enough: the caller runs the unfused GEMMs).
pub(super) fn hc_fuse_grid(cap: u32) -> u32 { if cap < 16 { 0 } else { cap.min(80) } }

/// The FP4L router's softmax / top-k / renorm (xf_router_topk: xq_router_topk's math, a faster exact selection) over
/// `b` rows of fp32 logits at `logits` -> ids / wts (device pointers).
pub(super) fn router_topk(l: &Launcher, ids: u64, wts: u64, logits: u64, ne: usize, k: usize, b: usize) -> Result<()> {
    anyhow::ensure!(k * 8 <= 256 && ne >= k, "FP4L router top-k: k {k} / ne {ne} outside the kernel's contract");
    xflaunch!(l, "xf_router_topk", (b as u32, 1, 1), (256, 1, 1), ((256 + ne + 512) * 4) as u32,
              (ids, wts, logits, ne as i32, k as i32, b as i32))
}

/// xq_hc_norm with fp32 norm weights. grid hc*b, block 1024, 4 KB smem.
#[allow(clippy::too_many_arguments)]
pub(super) fn hc_norm_f(l: &Launcher, out: u64, x: u64, w: &CudaSlice<f32>, h: usize, hcn: usize, b: usize, eps: f32) -> Result<()> {
    crate::metrics::fp4l_twin("hc_norm");
    xflaunch!(l, "xf_hc_norm_f", ((hcn * b) as u32, 1, 1), (1024, 1, 1), 4096,
              (out, x, w, h as i32, hcn as i32, b as i32, eps))
}
/// xq_hc_inj_norm (decode fused inject + norm) with fp32 norm weights. grid hc*b, block 1024.
#[allow(clippy::too_many_arguments)]
pub(super) fn hc_inj_norm_f(l: &Launcher, out: u64, x: u64, y: u64, inj: u64, w: &CudaSlice<f32>, h: usize, hcn: usize,
                            b: usize, eps: f32) -> Result<()> {
    crate::metrics::fp4l_twin("hc_norm");
    xflaunch!(l, "xf_hc_inj_norm_f", ((hcn * b) as u32, 1, 1), (1024, 1, 1), 0,
              (out, x, y, inj, w, h as i32, hcn as i32, b as i32, eps))
}
/// xq_pf_hc_inj_norm (prefill fused inject + norm; y == 0 skips the inject) with fp32 norm weights. block 128.
#[allow(clippy::too_many_arguments)]
pub(super) fn pf_hc_inj_norm_f(l: &Launcher, out: u64, x: u64, y: u64, inj: u64, w: u64, h: usize, hcn: usize,
                               b: usize, eps: f32) -> Result<()> {
    crate::metrics::fp4l_twin("hc_norm");
    xflaunch!(l, "xf_pf_hc_inj_norm_f", ((hcn * b) as u32, 1, 1), (128, 1, 1), 0,
              (out, x, y, inj, w, h as i32, hcn as i32, b as i32, eps))
}
/// xq_hc_fuse with the stored bf16 mixers: grid (bx, m) — the caller sized bx within this kernel's co-residency.
#[allow(clippy::too_many_arguments)]
pub(super) fn hc_fuse_bf(l: &Launcher, dd: u64, uu: u64, hn: u64, down: &CudaSlice<u16>, up: &CudaSlice<u16>, bar: u64,
                         hcn: f32, lr: usize, rw: usize, bx: u32, m: usize) -> Result<()> {
    crate::metrics::fp4l_twin("hc_mixer");
    xflaunch!(l, "xf_hc_fuse_bf", (bx, m as u32, 1), (256, 1, 1), 0,
              (dd, uu, hn, down, up, bar, hcn, lr as i32, rw as i32))
}
/// xq_hc_mix_rg with the stored bf16 inject weights (any b): grid (ninj + ceil(h/1024), b), block 1024.
#[allow(clippy::too_many_arguments)]
pub(super) fn hc_mix_bf(l: &Launcher, x: u64, inj: u64, hn: u64, uu: u64, winj: u64, h: usize, hcn: usize, b: usize) -> Result<()> {
    crate::metrics::fp4l_twin("hc_mix");
    let ninj = if winj != 0 { hcn } else { 0 };
    let gx = (ninj + h.div_ceil(1024)) as u32;
    xflaunch!(l, "xf_hc_mix_bf", (gx, b as u32, 1), (1024, 1, 1), 0,
              (x, inj, hn, uu, winj, h as i32, hcn as i32, b as i32))
}
/// xq_pf_hc_mix_2560x4 with the stored bf16 inject weights (h == 2560, hc == 4 only). grid b, block 128.
pub(super) fn pf_hc_mix_bf(l: &Launcher, x: u64, inj: u64, hn: u64, uu: u64, winj: u64, b: usize) -> Result<()> {
    crate::metrics::fp4l_twin("hc_mix");
    xflaunch!(l, "xf_pf_hc_mix_bf", (b as u32, 1, 1), (128, 1, 1), 0, (x, inj, hn, uu, winj, b as i32))
}
/// xq_embed_resid with the stored bf16 table.
pub(super) fn embed_resid_bf(l: &Launcher, resid: u64, emb: &CudaSlice<u16>, toks: u64, h: usize, hcn: usize, b: usize) -> Result<()> {
    crate::metrics::fp4l_twin("embed");
    let n = (h * hcn * b) as u32;
    xflaunch!(l, "xf_embed_resid_bf", (n / 256 + 1, 1, 1), (256, 1, 1), 0, (resid, emb, toks, h as i32, hcn as i32, b as i32))
}
/// The shared-expert gate channels of m rows of x (see xf_sg_gate); `xg` [m][h], columns >= 6 zero.
pub(super) fn sg_gate(l: &Launcher, xg: u64, sg_bf: &CudaSlice<u16>, x: u64, h: usize, m: usize) -> Result<()> {
    crate::metrics::fp4l_twin("shared_gate");
    xflaunch!(l, "xf_sg_gate", (m as u32, 1, 1), (256, 1, 1), 0, (xg, sg_bf, x, h as i32, m as i32))
}

/// The n-gram norm twins with the stored fp32 weights: `which` 0 = key (xq_ple_norm_f16: f16 in, f32 out),
/// 1 = query (xq_ple_norm_f32: f32 in, f32 out), 2 = conv (xq_ple_norm_hh: f16 in, f16 out). grid rows*4, block 256.
pub(super) fn ple_norm_f(l: &Launcher, which: u8, x: u64, w: &CudaSlice<f32>, out: u64, rows: usize, eps: f32) -> Result<()> {
    crate::metrics::fp4l_twin("ple");
    let f = match which { 0 => "xf_ple_norm_f16_f", 1 => "xf_ple_norm_f32_f", _ => "xf_ple_norm_hh_f" };
    xflaunch!(l, f, ((rows * 4) as u32, 1, 1), (256, 1, 1), 0, (x, w, out, eps))
}
/// xq_ple_conv (decode / verify step) with the conv taps in fp32 (the stored bf16, widened exactly).
#[allow(clippy::too_many_arguments)]
pub(super) fn ple_conv_f(l: &Launcher, resid: u64, gated: u64, normed: u64, state: u64, conv: &CudaSlice<f32>, m: usize,
                         slots: u64) -> Result<()> {
    crate::metrics::fp4l_twin("ple");
    xflaunch!(l, "xf_ple_conv_f", (((m * 10240) as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
              (resid, gated, normed, state, conv, m as i32, slots))
}
/// xq_ple_conv_chunk (prefill chunk / verify ring) with fp32 conv taps.
#[allow(clippy::too_many_arguments)]
pub(super) fn ple_conv_chunk_f(l: &Launcher, resid: u64, gated: u64, normed: u64, state: u64, conv: &CudaSlice<f32>,
                               slot: usize, c: usize) -> Result<()> {
    crate::metrics::fp4l_twin("ple");
    xflaunch!(l, "xf_ple_conv_chunk_f", ((10240u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
              (resid, gated, normed, state, conv, slot as i32, c as i32))
}
/// xq_ple_conv_chunk_seg (lane-pack segments) with fp32 conv taps.
#[allow(clippy::too_many_arguments)]
pub(super) fn ple_conv_chunk_seg_f(l: &Launcher, resid: u64, gated: u64, normed: u64, state: u64, conv: &CudaSlice<f32>,
                                   seg: u64, seg_k: usize) -> Result<()> {
    crate::metrics::fp4l_twin("ple");
    xflaunch!(l, "xf_ple_conv_chunk_seg_f", ((10240u32 + 255) / 256, seg_k as u32, 1), (256, 1, 1), 0,
              (resid, gated, normed, state, conv, seg))
}

/// The f16 channel constants xf_sg_gate's channels are paired with: (1, 2^-11, 2^-22, 2^-24, 2^-24, 2^-24, 0, ...) [h].
pub(crate) fn sg_channel_constants(h: usize) -> Vec<u16> {
    let mut v = vec![0u16; h];
    for (i, b) in [0x3C00u16, 0x1000, 0x0004, 0x0001, 0x0001, 0x0001].into_iter().enumerate() { v[i] = b; }
    v
}

/// Host model of xf_sg_gate's encoding + a combine's fp32 sum over the channels (unit tests): returns the
/// six f16 channels of `l` and the value a combine rebuilds from them in the given pairing order.
pub(crate) fn sg_encode(l: f32) -> [u16; 6] {
    let mut a = l.abs().min(65504.0);
    let lo = [5i32, -6, -17, -28, -39, -48];
    let up = [0i32, 11, 22, 24, 24, 24];
    let mut out = [0u16; 6];
    for i in 0..6 {
        let v = (a * 2f32.powi(-lo[i])).floor() * 2f32.powi(lo[i]);
        a -= v;
        let c = v * 2f32.powi(up[i]);
        out[i] = half::f16::from_f32(if l < 0.0 { -c } else { c }).to_bits();
    }
    out
}

/// NVFP4 routed experts of one layer (or the MTP head), stacked in LOCAL order, each expert matrix in the
/// fragment-native F128 layout (`nv4_frag`; same byte counts as the checkpoint's row-major arrays).
pub(crate) struct FxMoe {
    /// gate|up fused per expert: F128 of nibbles [2*mi][h/2] + E4M3 scales [2*mi][h/16], [gate; up] rows
    pub gu_w: CudaSlice<u8>,
    pub gu_s: CudaSlice<u8>,
    /// down: F128 of nibbles [h][mi/2] + scales [h][mi/16]
    pub dn_w: CudaSlice<u8>,
    pub dn_s: CudaSlice<u8>,
    /// [E][3] fp32 multipliers (gate, up, down) = the checkpoint's weight_scale_2
    pub gs: CudaSlice<f32>,
    pub n_local: usize,
}

/// The NVFP4 PLE table (`ngram_embedding.shard_{s}.weight` [rows, 80] + `.weight_scale` [rows, 10]).
pub(crate) struct PleFx {
    /// per shard: absolute offset of the nibble data and of the scale data in `PleDev::shards[s]`
    pub w_off: Vec<u64>,
    pub s_off: Vec<u64>,
    /// the table's single weight_scale_2 (dequant multiplies by it)
    pub gs: f32,
}

/// FP4L PLE row: 80 B nibbles then 10 B E4M3 scales (the RAM table interleaves them this way too).
pub(super) const PLE_FX_ROW: usize = 90;

// ------------------------------------------------------------------------------------------------
// Shapes / split-K
// ------------------------------------------------------------------------------------------------

/// Split-K factor of an m <= 16 dense FP4L GEMM: a pure function of (K, N) (batch invariance).
/// The smallest divisor of K/128 that puts >= 192 CTAs (4 waves of 48 SMs) on the device, else the
/// largest one; lm_head-class N (>= 192 tiles) runs ks = 1. (FP4L-SPEED1 tried a one-wave rule from a cold-L2
/// microbench that favoured it per shape: in the real decode step it was not faster — plain 26.9 -> 26.7 tok/s —
/// so this rule stands.)
pub(crate) fn fx_ks(k: usize, n: usize) -> u32 {
    let tiles = n.div_ceil(128).max(1);
    let kc = (k / 128).max(1);
    if tiles >= 192 { return 1; }
    let divs: Vec<usize> = (1..=kc).filter(|d| kc % d == 0).collect();
    for &d in &divs { if tiles * d >= 192 { return d as u32; } }
    *divs.last().unwrap() as u32
}

fn check_dense_shape(k: usize, n: usize, what: &str) -> Result<()> {
    anyhow::ensure!(k % 128 == 0 && n % 2 == 0 && k > 0 && n > 0,
                    "FP4L {what}: K {k} must be a multiple of 128 and N {n} even (the xf_mx8 / xf_f16w contract)");
    Ok(())
}

// ------------------------------------------------------------------------------------------------
// Load
// ------------------------------------------------------------------------------------------------

/// `Some(pack)` when `dir` is an FP4-Local directory (the loader then reads every weight here).
pub(super) fn open_if_fp4l(dir: &str) -> Result<Option<Arc<Fp4lPack>>> {
    let p = std::path::Path::new(dir);
    if !crate::fp4l::pack::is_fp4l_dir(p) { return Ok(None); }
    crate::metrics::fp4l_pack_loaded();
    let t0 = std::time::Instant::now();
    let pk = Fp4lPack::open(p).with_context(|| format!("FP4L: open {dir}"))?;
    crate::rprintln!("[fp4l] {dir}: ModelOpt MIXED_PRECISION pack, {} tensors in {} files ({:.1}s); MXFP8 dense + NVFP4 csf experts \
                      + bf16 head/MTP served as stored (W8A16 / W4A16, f16 activations)", pk.t.len(), pk.files.len(), t0.elapsed().as_secs_f32());
    Ok(Some(Arc::new(pk)))
}

/// bf16 bytes -> (f16 bits of v * 2^k, osc = 2^-k), exact for every value in this pack (receipt:
/// --probe-fp4l FP4L_PLAIN_F16_SHIFTED); any value that still does not fit is counted and printed.
fn bf16_bytes_to_f16(raw: &[u8], what: &str) -> (Vec<u16>, f32) {
    let vals: Vec<f32> = raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
    let k = crate::fp4l::pack::exact_f16_shift(&vals);
    let sc = 2f32.powi(k);
    let mut n_round = 0u64;
    let out: Vec<u16> = vals.iter().map(|&v| {
        let h = half::f16::from_f32(v * sc);
        if h.to_f32() != v * sc && v.is_finite() { n_round += 1; }
        h.to_bits()
    }).collect();
    if n_round > 0 {
        crate::rprintln!("[fp4l] {what}: {n_round} bf16 values do not fit f16 even with the 2^{k} shift (rounded; range wider than f16's)");
    }
    (out, 2f32.powi(-k))
}

/// Row-major MXFP8 ([n][k] E4M3 codes, [n][k/32] UE8M0 scales) -> the fragment-native F64 layout the
/// xf_mx8_* kernels read (gpu_fp4l.cu header). A pure byte permutation: every code and scale byte is kept,
/// rows are padded to a multiple of 16 with zero codes / scale 127 (2^0; the padded rows are never stored).
/// Codes W'[t][b][h][lane][16]: lane = gid*4 + tig holds row 16t+8h+gid, k = 64b + 32gs + 8tig + j at byte
/// gs*8 + j. Scales S'[t][b][ts][8]: byte h*4 + r*2 + gs holds row 16t+8h+2ts+r, group 2b+gs.
pub(crate) fn mx8_frag(codes: &[u8], scales: &[u8], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    assert!(k % 64 == 0 && codes.len() == n * k && scales.len() == n * k / 32, "mx8_frag: [{n}][{k}] shape");
    let nt = n.div_ceil(16);
    let kb = k / 64;
    let mut w = vec![0u8; nt * 16 * k];
    let mut sc = vec![127u8; nt * k / 2];
    let th = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).clamp(1, 16);
    let per = nt.div_ceil(th).max(1);
    std::thread::scope(|sco| {
        for (ci, (wc, scc)) in w.chunks_mut(per * 16 * k).zip(sc.chunks_mut(per * k / 2)).enumerate() {
            sco.spawn(move || {
                for tl in 0..wc.len() / (16 * k) {
                    let t = ci * per + tl;
                    for rr in 0..16 {
                        let row = 16 * t + rr;
                        if row >= n { break; }
                        let (h, gid) = (rr >> 3, rr & 7);
                        let src = &codes[row * k..(row + 1) * k];
                        let ssrc = &scales[row * (k / 32)..(row + 1) * (k / 32)];
                        for b in 0..kb {
                            for gs in 0..2 {
                                for tig in 0..4 {
                                    let o = ((tl * kb + b) * 2 + h) * 512 + (gid * 4 + tig) * 16 + gs * 8;
                                    let i = 64 * b + 32 * gs + 8 * tig;
                                    wc[o..o + 8].copy_from_slice(&src[i..i + 8]);
                                }
                                scc[(tl * kb + b) * 32 + (gid >> 1) * 8 + h * 4 + (gid & 1) * 2 + gs] = ssrc[2 * b + gs];
                            }
                        }
                    }
                }
            });
        }
    });
    (w, sc)
}

/// Row-major stored bf16 [n][k] -> the fragment-native B32 layout the stored-bf16 GEMMs read (gpu_fp4l.cu): a pure
/// u16 permutation (no value changes), rows padded to a multiple of 16 with zeros. W'[t][g][h][lane][8]: lane =
/// gid*4 + tig holds row 16t+8h+gid, k = 32g + 8tig + j at u16 j.
pub(crate) fn bf16_frag(bits: &[u16], n: usize, k: usize) -> Vec<u16> {
    assert!(k % 32 == 0 && bits.len() == n * k, "bf16_frag: [{n}][{k}] shape ({} values)", bits.len());
    let (nt, g) = (n.div_ceil(16), k / 32);
    let mut w = vec![0u16; nt * 16 * k];
    let th = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).clamp(1, 16);
    let per = nt.div_ceil(th).max(1);
    std::thread::scope(|sco| {
        for (ci, wc) in w.chunks_mut(per * 16 * k).enumerate() {
            sco.spawn(move || {
                for tl in 0..wc.len() / (16 * k) {
                    for rr in 0..16 {
                        let row = 16 * (ci * per + tl) + rr;
                        if row >= n { break; }
                        let (h, gid) = (rr >> 3, rr & 7);
                        let src = &bits[row * k..(row + 1) * k];
                        for gg in 0..g {
                            for tig in 0..4 {
                                let o = ((tl * g + gg) * 2 + h) * 256 + (gid * 4 + tig) * 8;
                                let i = 32 * gg + 8 * tig;
                                wc[o..o + 8].copy_from_slice(&src[i..i + 8]);
                            }
                        }
                    }
                }
            });
        }
    });
    w
}

/// The inverse of `bf16_frag` (the first n rows, row-major) — host readers of a B32 device copy (--hc-int8).
pub(crate) fn bf16_unfrag(w: &[u16], n: usize, k: usize) -> Vec<u16> {
    assert!(k % 32 == 0 && w.len() >= n * k, "bf16_unfrag: [{n}][{k}] shape ({} values)", w.len());
    let g = k / 32;
    let mut out = vec![0u16; n * k];
    for row in 0..n {
        let (t, rr) = (row / 16, row % 16);
        let (h, gid) = (rr >> 3, rr & 7);
        for gg in 0..g {
            for tig in 0..4 {
                let o = ((t * g + gg) * 2 + h) * 256 + (gid * 4 + tig) * 8;
                let i = row * k + 32 * gg + 8 * tig;
                out[i..i + 8].copy_from_slice(&w[o..o + 8]);
            }
        }
    }
    out
}

/// Upload a row-major MXFP8 matrix in the F64 layout.
fn mx8_dev(dev: &Arc<CudaDevice>, codes: &[u8], scales: &[u8], n: usize, k: usize) -> Result<FxW> {
    let (w, s) = mx8_frag(codes, scales, n, k);
    Ok(FxW::Mx8 { w: dev.htod_sync_copy(&w)?, s: dev.htod_sync_copy(&s)? })
}

/// Concatenate the [r0, r1) element ranges of `v`, each range scaled by `unit` elements.
fn take_ranges<T: Copy>(v: &[T], ranges: &[(usize, usize)], unit: usize) -> Vec<T> {
    ranges.iter().flat_map(|&(a, b)| v[a * unit..b * unit].iter().copied()).collect()
}

/// An FP4L dense linear as a Quad. `cols` = output (N) ranges, `rows` = input (K) ranges, in
/// ELEMENTS (the EXL3 get_quad_sl convention; 128-aligned by the TP planner). `rowpar` marks the
/// K-sliced partial class (xtp::chain_rowpar).
pub(super) fn fx_quad(dev: &Arc<CudaDevice>, fp: &Fp4lPack, stem: &str, rows: Option<&[(usize, usize)]>,
                      cols: Option<&[(usize, usize)]>, rowpar: bool) -> Result<Quad> {
    let kind = fp.lin_kind(stem)?;
    let (n_full, k_full) = match kind {
        LinKind::Mxfp8 | LinKind::Bf16 => fp.shape2(&format!("{stem}.weight"))?,
        other => bail!("FP4L {stem}: {other:?} is not a dense linear kind"),
    };
    let nr: Vec<(usize, usize)> = cols.map(|c| c.to_vec()).unwrap_or_else(|| vec![(0, n_full)]);
    let kr: Vec<(usize, usize)> = rows.map(|r| r.to_vec()).unwrap_or_else(|| vec![(0, k_full)]);
    for &(a, b) in &nr { anyhow::ensure!(a <= b && b <= n_full, "FP4L {stem}: output range [{a}, {b}) outside N {n_full}"); }
    for &(a, b) in &kr {
        anyhow::ensure!(a <= b && b <= k_full && a % 32 == 0 && b % 32 == 0,
                        "FP4L {stem}: input range [{a}, {b}) outside K {k_full} or not 32-aligned");
    }
    let n: usize = nr.iter().map(|(a, b)| b - a).sum();
    let k: usize = kr.iter().map(|(a, b)| b - a).sum();
    check_dense_shape(k, n, stem)?;
    let whole_k = kr.len() == 1 && kr[0] == (0, k_full);
    let fx = match kind {
        LinKind::Mxfp8 => {
            let m = fp.mxfp8(stem)?;
            let mut codes = Vec::with_capacity(n * k);
            let mut scales = Vec::with_capacity(n * k / 32);
            for &(a, b) in &nr {
                for r in a..b {
                    let row = &m.codes[r * k_full..(r + 1) * k_full];
                    let srow = &m.scales[r * (k_full / 32)..(r + 1) * (k_full / 32)];
                    if whole_k { codes.extend_from_slice(row); scales.extend_from_slice(srow); }
                    else {
                        codes.extend(take_ranges(row, &kr, 1));
                        scales.extend(kr.iter().flat_map(|&(x, y)| srow[x / 32..y / 32].iter().copied()));
                    }
                }
            }
            anyhow::ensure!(!scales.contains(&0xFF), "FP4L {stem}: a UE8M0 scale is NaN (0xFF) — checkpoint corruption");
            mx8_dev(dev, &codes, &scales, n, k)?
        }
        LinKind::Bf16 => {
            // the lm_head follows --fp4l-lm-head; every other bf16 linear (the MTP layer) --fp4l-plain
            let mode = if stem == "lm_head" { head_prec() } else if plain_exact() { HeadPrec::Bf16 } else { HeadPrec::F16 };
            let raw = fp.raw(&format!("{stem}.weight"))?;
            let bits: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let gather = |full: &[u16]| -> Vec<u16> {
                let mut w = Vec::with_capacity(n * k);
                for &(a, b) in &nr {
                    for r in a..b {
                        let row = &full[r * k_full..(r + 1) * k_full];
                        if whole_k { w.extend_from_slice(row); } else { w.extend(take_ranges(row, &kr, 1)); }
                    }
                }
                w
            };
            match mode {
                HeadPrec::Bf16 => FxW::Bf16 { w: dev.htod_sync_copy(&bf16_frag(&gather(&bits), n, k))? },
                HeadPrec::F16 => {
                    let (full, osc) = bf16_bytes_to_f16(&raw, stem);
                    FxW::F16 { w: dev.htod_sync_copy(&gather(&full))?, osc }
                }
                HeadPrec::Mx8 => {
                    // opt-in speed lever: an MXFP8 RTN copy of the stored bf16 rows (changes the logits)
                    let vals: Vec<f32> = gather(&bits).iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
                    let (codes, scales) = mx8_rtn(&vals, n, k);
                    crate::rprintln!("[fp4l] {stem}: served as an MXFP8 RTN copy (--fp4l-lm-head mxfp8: {:.0} MB instead of {:.0} MB; \
                                      changes the logits)", (codes.len() + scales.len()) as f64 / 1e6, (2 * n * k) as f64 / 1e6);
                    mx8_dev(dev, &codes, &scales, n, k)?
                }
            }
        }
        _ => unreachable!(),
    };
    fx_quad_wrap(dev, fx, k, n, rowpar)
}

/// Wrap a device FxW into a Quad (placeholder trellis fields, zeroed fixup counters).
pub(super) fn fx_quad_wrap(dev: &Arc<CudaDevice>, fx: FxW, k: usize, n: usize, rowpar: bool) -> Result<Quad> {
    let one = || -> Result<CudaSlice<u16>> { Ok(dev.htod_sync_copy(&[0u16])?) };
    Ok(Quad {
        tr: one()?, suh: one()?, svh: one()?, bits: 0, k2: 0, k: k as i32, n: n as i32,
        ks: fx_ks(k, n), lmh: false,
        // one u32 per 128-row N-tile, explicitly zeroed (AGENTS §2.2); the fixup epilogue re-arms it
        cnt: dev.htod_sync_copy(&vec![0u32; n.div_ceil(128).max(1)])?,
        rowpar,
        fx: Some(Box::new(fx)),
    })
}

/// Rows [r0, r1) of an FP4L f16 Quad as a new Quad (the pruned draft head prefix, the vocab shard).
pub(super) fn fx_rows_of(dev: &Arc<CudaDevice>, stream: &CudaStream, q: &Quad, r0: usize, r1: usize) -> Result<Quad> {
    let Some(fx) = q.fx.as_deref() else { bail!("fx_rows_of on a trellis Quad") };
    let k = q.k as usize;
    anyhow::ensure!(r0 < r1 && r1 <= q.n as usize, "fx_rows_of [{r0}, {r1}) outside N {}", q.n);
    let _ = stream;
    let fx2 = match fx {
        FxW::Trellis { .. } => bail!("fx_rows_of on a sidecar trellis linear (only heads are row-sliced)"),
        FxW::F16 { w, osc } => {
            let mut d = dev.alloc_zeros::<u16>((r1 - r0) * k)?;
            dev.dtod_copy(&w.slice(r0 * k..r1 * k), &mut d)?;
            FxW::F16 { w: d, osc: *osc }
        }
        FxW::Bf16 { w } => {
            // B32 is tiled by 16 rows (as F64): whole tiles
            anyhow::ensure!(r0 % 16 == 0 && (r1 % 16 == 0 || r1 == q.n as usize),
                            "fx_rows_of [{r0}, {r1}) on a bf16 Quad must be 16-row aligned (the B32 tile)");
            let (t0, t1) = (r0 / 16, r1.div_ceil(16));
            let mut d = dev.alloc_zeros::<u16>((t1 - t0) * 16 * k)?;
            dev.dtod_copy(&w.slice(t0 * 16 * k..t1 * 16 * k), &mut d)?;
            FxW::Bf16 { w: d }
        }
        FxW::Mx8 { w, s } => {
            // F64 is tiled by 16 rows: slice whole tiles (a tail past r1 inside the last tile is zero-free
            // padding the epilogue never writes, since the new Quad's N is r1 - r0)
            anyhow::ensure!(r0 % 16 == 0 && (r1 % 16 == 0 || r1 == q.n as usize),
                            "fx_rows_of [{r0}, {r1}) on an MXFP8 Quad must be 16-row aligned (the F64 tile)");
            let (t0, t1) = (r0 / 16, r1.div_ceil(16));
            let mut dw = dev.alloc_zeros::<u8>((t1 - t0) * 16 * k)?;
            let mut ds = dev.alloc_zeros::<u8>((t1 - t0) * k / 2)?;
            dev.dtod_copy(&w.slice(t0 * 16 * k..t1 * 16 * k), &mut dw)?;
            dev.dtod_copy(&s.slice(t0 * k / 2..t1 * k / 2), &mut ds)?;
            FxW::Mx8 { w: dw, s: ds }
        }
    };
    dev.synchronize()?;
    let mut out = fx_quad_wrap(dev, fx2, k, r1 - r0, false)?;
    out.lmh = q.lmh;
    Ok(out)
}

/// A plain f16 view of an MXFP8 tensor (the tiny GDN a/b projections the trunk keeps as plain f16
/// [n][k] weights). Rounding of sub-2^-14 values is counted and printed (never silent).
pub(super) fn mx8_plain_f16(fp: &Fp4lPack, weight_name: &str) -> Result<Vec<u16>> {
    let stem = weight_name.strip_suffix(".weight").unwrap_or(weight_name);
    let m = fp.mxfp8(stem)?;
    let mut row = vec![0f32; m.k];
    let mut out = Vec::with_capacity(m.n * m.k);
    let mut n_round = 0u64;
    for r in 0..m.n {
        m.row_f32(r, &mut row);
        for &v in &row {
            let h = half::f16::from_f32(v);
            if h.to_f32() != v { n_round += 1; }
            out.push(h.to_bits());
        }
    }
    if n_round > 0 {
        crate::rprintln!("[fp4l] {stem}: {n_round} MXFP8 values below f16's exact range rounded to the nearest f16 (plain f16 projection)");
    }
    Ok(out)
}

/// FP4L: the GDN in_proj_a / in_proj_b as exact MXFP8 Quads (rows = value heads [h0, h1) of this rank).
pub(super) fn fx_gdn_ab(dev: &Arc<CudaDevice>, fp: &Fp4lPack, p: &str, heads: Option<(usize, usize)>) -> Result<(Quad, Quad)> {
    let r: Option<Vec<(usize, usize)>> = heads.map(|h| vec![h]);
    Ok((fx_quad(dev, fp, &format!("{p}.linear_attn.in_proj_a"), None, r.as_deref(), false)?,
        fx_quad(dev, fp, &format!("{p}.linear_attn.in_proj_b"), None, r.as_deref(), false)?))
}

/// The a|b pair's private split-K workspace: two halves of fp32 slabs (one per member — xf_mx8_gemm_x2 runs them at
/// the same time; the two-launch path and the prefill tails use the first half).
pub(super) fn ab_ws_len(a: &Quad, b: &Quad) -> usize {
    2 * ab_ws_half(a, b)
}
fn ab_ws_half(a: &Quad, b: &Quad) -> usize {
    16 * (a.ks.max(1) as usize * a.n as usize).max(b.ks.max(1) as usize * b.n as usize).max(1)
}

/// FP4L GDN a|b at m <= 16 (decode / verify): the exact MXFP8 GEMM, private workspace.
pub(super) fn fx_ab(l: &Launcher, ab: &(Quad, Quad, CudaSlice<f32>), x: &CudaSlice<u16>, a_out: &mut CudaSlice<u16>,
                    b_out: &mut CudaSlice<u16>, m: usize) -> Result<()> {
    let (qa, qb, ws) = ab;
    let half = ab_ws_half(qa, qb);
    anyhow::ensure!(m * qa.k as usize <= x.len() && m * qa.n as usize <= a_out.len() && m * qb.n as usize <= b_out.len(),
                    "FP4L GDN a|b m={m}: buffers too small");
    let wsp = *ws.device_ptr() as u64;
    if fx_pair(l, qa, qb, *x.device_ptr() as u64, *a_out.device_ptr() as u64, *b_out.device_ptr() as u64,
               (wsp, half.min(ws.len())), (wsp + (half * 4) as u64, ws.len().saturating_sub(half)), m)? {
        return Ok(());
    }
    fx_chain(l, qa, qa.fx.as_deref().unwrap(), x, a_out, m, ws, None)?;
    fx_chain(l, qb, qb.fx.as_deref().unwrap(), x, b_out, m, ws, None)
}

/// FP4L GDN a|b at a prefill chunk's c rows (the exact wide kernel; c <= 16 = the small one).
pub(super) fn fx_ab_rows(l: &Launcher, ab: &(Quad, Quad, CudaSlice<f32>), x: &CudaSlice<u16>, a_out: &mut CudaSlice<u16>,
                         b_out: &mut CudaSlice<u16>, c: usize, rb: &ReconBufs) -> Result<()> {
    let (qa, qb, ws) = ab;
    fx_chain(l, qa, qa.fx.as_deref().unwrap(), x, a_out, c, ws, Some(rb))?;
    fx_chain(l, qb, qb.fx.as_deref().unwrap(), x, b_out, c, ws, Some(rb))
}

/// The routed experts of `prefix` (`...layers.L.mlp` or `mtp.layers.0.mlp`) for the experts in
/// `local` (stacking order = local index). gate and up rows are fused per expert: [gate; up].
pub(super) fn fx_experts(dev: &Arc<CudaDevice>, fp: &Fp4lPack, prefix: &str, local: &[usize], h: usize, mi: usize) -> Result<FxMoe> {
    let ne = local.len();
    let (gu_w_e, gu_s_e) = (2 * mi * h / 2, 2 * mi * h / 16);
    let (dn_w_e, dn_s_e) = (h * mi / 2, h * mi / 16);
    let mut gu_w = vec![0u8; ne * gu_w_e];
    let mut gu_s = vec![0u8; ne * gu_s_e];
    let mut dn_w = vec![0u8; ne * dn_w_e];
    let mut dn_s = vec![0u8; ne * dn_s_e];
    let mut gs = vec![0f32; ne * 3];
    // parallel over experts: csf decode + reads (the load is I/O + integer adds)
    let nt = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).clamp(1, 16);
    let chunk = ne.div_ceil(nt).max(1);
    let res: Vec<Result<()>> = std::thread::scope(|s| {
        let mut hs = Vec::new();
        for (ci, ((((gw, gsc), dw), dsc), gsv)) in gu_w.chunks_mut(chunk * gu_w_e).zip(gu_s.chunks_mut(chunk * gu_s_e))
            .zip(dn_w.chunks_mut(chunk * dn_w_e)).zip(dn_s.chunks_mut(chunk * dn_s_e)).zip(gs.chunks_mut(chunk * 3)).enumerate() {
            let local = &local[ci * chunk..(ci * chunk + gw.len() / gu_w_e)];
            hs.push(s.spawn(move || -> Result<()> {
                for (j, &e) in local.iter().enumerate() {
                    let g = fp.nvfp4(&format!("{prefix}.experts.{e}.gate_proj"))?;
                    let u = fp.nvfp4(&format!("{prefix}.experts.{e}.up_proj"))?;
                    let d = fp.nvfp4(&format!("{prefix}.experts.{e}.down_proj"))?;
                    anyhow::ensure!(g.n == mi && g.k == h && u.n == mi && u.k == h && d.n == h && d.k == mi,
                                    "FP4L {prefix} expert {e}: shapes gate {}x{} up {}x{} down {}x{} vs h {h} mi {mi}",
                                    g.n, g.k, u.n, u.k, d.n, d.k);
                    let gu_rm: Vec<u8> = g.packed.iter().chain(&u.packed).copied().collect();
                    let gus_rm: Vec<u8> = g.scales.iter().chain(&u.scales).copied().collect();
                    nv4_frag_into(&gu_rm, &gus_rm, 2 * mi, h, &mut gw[j * gu_w_e..(j + 1) * gu_w_e],
                                  &mut gsc[j * gu_s_e..(j + 1) * gu_s_e]);
                    nv4_frag_into(&d.packed, &d.scales, h, mi, &mut dw[j * dn_w_e..(j + 1) * dn_w_e],
                                  &mut dsc[j * dn_s_e..(j + 1) * dn_s_e]);
                    gsv[3 * j] = g.gs; gsv[3 * j + 1] = u.gs; gsv[3 * j + 2] = d.gs;
                }
                Ok(())
            }));
        }
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("FP4L expert load thread panicked")))).collect()
    });
    for r in res { r?; }
    let bad = gu_s.iter().chain(dn_s.iter()).filter(|&&b| b & 0x80 != 0 || b & 0x7F == 0x7F).count();
    anyhow::ensure!(bad == 0, "FP4L {prefix}: {bad} expert block scales are negative or NaN — refusing to serve them");
    Ok(FxMoe {
        gu_w: dev.htod_sync_copy(&gu_w)?, gu_s: dev.htod_sync_copy(&gu_s)?,
        dn_w: dev.htod_sync_copy(&dn_w)?, dn_s: dev.htod_sync_copy(&dn_s)?,
        gs: dev.htod_sync_copy(&gs)?, n_local: ne,
    })
}

/// Row-major NVFP4 ([n][k/2] nibbles, [n][k/16] E4M3 scales; n % 16 == 0, k % 128 == 0) -> the fragment-native
/// F128 layout the xf_nv4_* expert kernels read (gpu_fp4l.cu). A pure byte permutation (same sizes).
/// Codes W'[t][b][h][lane][16]: lane = gid*4 + tig holds row 16t+8h+gid, bytes s*8 + j = row byte 64b + 32s + 8tig + j.
/// Scales S'[t][b][lane][4]: byte 2h + s = row 16t+8h+gid's scale 8b + 4s + tig.
pub(crate) fn nv4_frag_into(codes: &[u8], scales: &[u8], n: usize, k: usize, w: &mut [u8], sc: &mut [u8]) {
    assert!(n % 16 == 0 && k % 128 == 0 && codes.len() == n * k / 2 && scales.len() == n * k / 16
            && w.len() == codes.len() && sc.len() == scales.len(), "nv4_frag: [{n}][{k}] shape");
    let (kb, k2, k16) = (k / 128, k / 2, k / 16);
    for t in 0..n / 16 {
        for h in 0..2 {
            for gid in 0..8 {
                let row = 16 * t + 8 * h + gid;
                for b in 0..kb {
                    for tig in 0..4 {
                        let lane = gid * 4 + tig;
                        for s2 in 0..2 {
                            let o = ((t * kb + b) * 2 + h) * 512 + lane * 16 + s2 * 8;
                            let i = row * k2 + 64 * b + 32 * s2 + 8 * tig;
                            w[o..o + 8].copy_from_slice(&codes[i..i + 8]);
                            sc[(t * kb + b) * 128 + lane * 4 + 2 * h + s2] = scales[row * k16 + 8 * b + 4 * s2 + tig];
                        }
                    }
                }
            }
        }
    }
}

/// Bytes one decode token streams through the FP4L routed experts (10 experts): the byte model.
pub(super) fn fx_expert_bytes_per_token(h: usize, mi: usize, topk: usize) -> u64 {
    let per = 3 * h * mi / 2 + 3 * h * mi / 16;
    (topk * per) as u64
}

/// FP4L byte model of one plain decode step (W = 1): every dense FP4L weight once, the hyper-connection
/// mixers as loaded (f16, or int8 with --hc-int8), the router, top-k experts' NVFP4 bytes per layer, the head.
pub(super) fn byte_model(layers: &[LayerDev], hc_mixer: &HcDev, lm_head: &Quad, topk: usize, h: usize, mi: usize) -> u64 {
    let qb = |q: &Quad| q.fx.as_deref().map_or(0, |f| f.bytes()) as u64;
    let hcb = |hc: &HcDev| match hc.q8.as_ref() {
        Some(q) => (q.down_q.len() + q.up_q.len()) as u64,
        None => 2 * (hc.down.len() + hc.up.len()) as u64,
    };
    let mut b = qb(lm_head) + hcb(hc_mixer);
    for l in layers {
        b += hcb(&l.hc_attn) + hcb(&l.hc_mlp);
        b += match &l.mixer {
            Mixer::Gdn(g) => qb(&g.in_qkv) + qb(&g.in_z) + qb(&g.out_proj)
                + g.ab_fx.as_deref().map_or(2 * (g.a_w.len() + g.b_w.len()) as u64, |ab| qb(&ab.0) + qb(&ab.1)),
            Mixer::Attn(a) => qb(&a.q_proj) + qb(&a.k_proj) + qb(&a.v_proj) + qb(&a.o_proj)
                + a.idx.as_ref().map_or(0, |i| qb(&i.qk)),
        };
        b += qb(&l.moe.sh_gate) + qb(&l.moe.sh_up) + qb(&l.moe.sh_down) + 2 * l.moe.router.len() as u64;
        if l.moe.fx.is_some() { b += fx_expert_bytes_per_token(h, mi, topk); }
    }
    b
}

/// The NVFP4 PLE table's shard geometry: (files, nibble offsets, scale offsets, rows/shard, gs).
pub(super) fn fx_ple_table(fp: &Fp4lPack, ple_key: &str) -> Result<(Vec<std::fs::File>, PleFx, i64)> {
    let pre = format!("{ple_key}.ple_embedding.ngram_embedding");
    let gs = fp.scalar_f32(&format!("{pre}.weight_scale_2"))?;
    anyhow::ensure!(gs.is_finite() && gs > 0.0, "FP4L PLE weight_scale_2 {gs}");
    let mut files = Vec::new();
    let (mut w_off, mut s_off) = (Vec::new(), Vec::new());
    let mut rows = 0i64;
    for s in 0..128usize {
        let w = fp.get(&format!("{pre}.shard_{s}.weight"))?;
        let sc = fp.get(&format!("{pre}.shard_{s}.weight_scale"))?;
        anyhow::ensure!(w.shape.len() == 2 && w.shape[1] == 80 && sc.shape.len() == 2 && sc.shape[1] == 10 && w.shape[0] == sc.shape[0],
                        "FP4L PLE shard {s}: shapes {:?} / {:?} (want [rows, 80] / [rows, 10])", w.shape, sc.shape);
        anyhow::ensure!(w.file == sc.file, "FP4L PLE shard {s}: nibbles and scales live in different files (unsupported)");
        if s == 0 { rows = w.shape[0] as i64; }
        anyhow::ensure!(w.shape[0] as i64 == rows, "FP4L PLE shard {s}: {} rows, shard 0 has {rows}", w.shape[0]);
        files.push(fp.files[w.file as usize].try_clone()?);
        w_off.push(w.off);
        s_off.push(sc.off);
    }
    Ok((files, PleFx { w_off, s_off, gs }, rows))
}

/// Decode one FP4L PLE row (90 B: nibbles then scales) into 160 f16 bits: e2m1 * e4m3 * gs, one RN.
pub(super) fn ple_fx_decode(raw: &[u8], gs: f32, out: &mut [u16]) {
    use crate::fp4l::pack::{e4m3, E2M1};
    for b in 0..10 {
        let s = e4m3(raw[80 + b]) * gs;
        for j in 0..8 {
            let byte = raw[b * 8 + j];
            out[b * 16 + 2 * j] = half::f16::from_f32(E2M1[(byte & 0xF) as usize] * s).to_bits();
            out[b * 16 + 2 * j + 1] = half::f16::from_f32(E2M1[(byte >> 4) as usize] * s).to_bits();
        }
    }
}

/// `--ple-ram ram` for the FP4L table: every shard's rows interleaved as 90-B records in RAM.
pub(super) fn ple_fx_ram(files: &[std::fs::File], fx: &PleFx, rows: i64) -> Result<Vec<Vec<u8>>> {
    use std::os::unix::fs::FileExt;
    let rows = rows as usize;
    // BASICS (owner 2026-11-02): the table read as a percentage at the normal level
    let prog = crate::boot_report::Progress::new("loading the n-gram table into RAM", (files.len() * rows * PLE_FX_ROW) as u64,
                                                 crate::boot_report::Unit::Bytes);
    let prog = &prog;
    let res: Vec<Result<Vec<u8>>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..files.len()).map(|si| {
            let f = &files[si];
            let (wo, so) = (fx.w_off[si], fx.s_off[si]);
            s.spawn(move || -> Result<Vec<u8>> {
                let mut w = vec![0u8; rows * 80];
                let mut sc = vec![0u8; rows * 10];
                f.read_exact_at(&mut w, wo)?;
                f.read_exact_at(&mut sc, so)?;
                let mut out = vec![0u8; rows * PLE_FX_ROW];
                for r in 0..rows {
                    out[r * PLE_FX_ROW..r * PLE_FX_ROW + 80].copy_from_slice(&w[r * 80..(r + 1) * 80]);
                    out[r * PLE_FX_ROW + 80..(r + 1) * PLE_FX_ROW].copy_from_slice(&sc[r * 10..(r + 1) * 10]);
                }
                prog.add((rows * PLE_FX_ROW) as u64);
                Ok(out)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("PLE RAM thread panicked")))).collect()
    });
    prog.finish();
    res.into_iter().collect()
}

// ------------------------------------------------------------------------------------------------
// Dense dispatch
// ------------------------------------------------------------------------------------------------

/// The FP4L chain: y[m, n] = x[m, k] · W[n, k]^T (f16 out). m <= 16: the batch-invariant MMA GEMM
/// (split-K fixup in-kernel). m > 16: dequant (MXFP8) + cuBLASLt (opA = T on the [n, k] weight), or —
/// without recon buffers — the m <= 16 kernel over 16-row blocks.
pub(super) fn fx_chain(l: &Launcher, q: &Quad, fx: &FxW, x: &CudaSlice<u16>, y: &mut CudaSlice<u16>, m: usize,
                       ws: &CudaSlice<f32>, rb: Option<&ReconBufs>) -> Result<()> {
    let (k, n) = (q.k as usize, q.n as usize);
    anyhow::ensure!(m * k <= x.len() && m * n <= y.len(), "FP4L chain m={m} K={k} N={n}: x {} / y {} too small", x.len(), y.len());
    crate::metrics::fp4l_gemm(fx.kind(), m);
    if m <= 16 {
        return fx_small(l, q, fx, *x.device_ptr() as u64, *y.device_ptr() as u64, 0, ws, m);
    }
    match rb {
        Some(r) => fx_wide(l, q, fx, *x.device_ptr() as u64, *y.device_ptr() as u64, m, r),
        None => {
            // no recon staging (an m > 16 chain outside prefill): 16-row blocks of the small kernel
            let mut r0 = 0;
            while r0 < m {
                let mb = (m - r0).min(16);
                fx_small(l, q, fx, *x.device_ptr() as u64 + (r0 * k * 2) as u64, *y.device_ptr() as u64 + (r0 * n * 2) as u64,
                         0, ws, mb)?;
                r0 += mb;
            }
            Ok(())
        }
    }
}

/// m <= 16: one launch (fixup) — `y` f16, or, when `yf != 0`, the fp32 result (rowpar partial).
/// `yf` mode never uses the fixup: ks > 1 writes the fp32 slabs to `ws` for the caller's combine.
#[allow(clippy::too_many_arguments)]
pub(super) fn fx_small(l: &Launcher, q: &Quad, fx: &FxW, x: u64, y: u64, yf: u64, ws: &CudaSlice<f32>, m: usize) -> Result<()> {
    let (k, n, ks) = (q.k as usize, q.n as usize, q.ks.max(1));
    anyhow::ensure!((1..=16).contains(&m), "fx_small: m {m} outside 1..=16");
    if let FxW::Trellis { tr, suh, svh, bits, ks: tks, scr, .. } = fx {
        // --fp4l-mtp-sidecar: H128(x * suh) -> trellis GEMM (fp32 K-slabs) -> ascending combine -> H128(.) * svh
        anyhow::ensure!(yf == 0, "FP4L MTP sidecar: a trellis linear has no fp32 partial-output (row-parallel) form");
        let (part, xh, yraw) = tr_scratch(scr, k, n, *tks)?;
        xqlaunch!(l, "exl3_had_suh", ((m * (k / 128)) as u32, 1, 1), (32, 1, 1), 0, (x, suh, xh, k as i32))?;
        xflaunch!(l, "xf_tr_gemm", ((n / 128) as u32, *tks as u32, 1), (256, 1, 1), 0,
                  (tr, xh, part, m as i32, k as i32, n as i32, *bits as i32, *tks as i32))?;
        xflaunch!(l, "xf_tr_combine", (((m * n) as u32).div_ceil(256), 1, 1), (256, 1, 1), 0, (part, yraw, (m * n) as i32, *tks as i32))?;
        return xqlaunch!(l, "exl3_had_svh", ((m * (n / 128)) as u32, 1, 1), (32, 1, 1), 0, (yraw, svh, y, n as i32));
    }
    anyhow::ensure!(k % (128 * ks as usize) == 0, "FP4L chain: ks {ks} does not split K {k} into 128-multiples");
    if ks > 1 { anyhow::ensure!(ks as usize * m * n <= ws.len(), "FP4L chain_ws too small: need {} have {}", ks as usize * m * n, ws.len()); }
    let tiles = n.div_ceil(128) as u32;
    // fixup when producing f16 with ks > 1; slab-only when the caller combines (yf mode)
    let cnt: u64 = if ks > 1 && yf == 0 { *q.cnt.device_ptr() as u64 } else { 0 };
    let (yp, yfp) = if yf != 0 { (0u64, if ks > 1 { 0u64 } else { yf }) } else { (y, 0u64) };
    let wsp = *ws.device_ptr() as u64;
    match fx {
        FxW::Mx8 { w, s } => xflaunch!(l, "xf_mx8_gemm", (tiles, ks, 1), (256, 1, 1), 0,
            (w, s, x, yp, yfp, wsp, cnt, m as i32, k as i32, n as i32, ks as i32)),
        FxW::F16 { w, osc } => xflaunch!(l, "xf_f16w_gemm", (tiles, ks, 1), (256, 1, 1), 0,
            (w, x, yp, yfp, wsp, cnt, m as i32, k as i32, n as i32, ks as i32, *osc)),
        FxW::Bf16 { w } => xflaunch!(l, "xf_bf16w_gemm", (tiles, ks, 1), (256, 1, 1), 0,
            (w, x, yp, yfp, wsp, cnt, m as i32, k as i32, n as i32, ks as i32)),
        FxW::Trellis { .. } => unreachable!(),
    }
}

/// xf_mx8_gemm_x2's per-member pointers, passed by value (kernel twin: `struct XfPair` in gpu_fp4l.cu).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct XfPair { w: [u64; 2], s: [u64; 2], y: [u64; 2], ws: [u64; 2], cnt: [u64; 2] }
unsafe impl cudarc::driver::DeviceRepr for XfPair {}

/// Two m <= 16 FP4L GEMMs that read the same `x` (the GDN in_proj_a | in_proj_b pair, the shared expert's
/// gate | up) as ONE xf_mx8_gemm_x2 launch: taken when both are MXFP8 with the same (K, N, ks) and each member's
/// fp32 slab space (`ws0` / `ws1`, raw pointer + length in floats: the members run at the same time) holds its
/// ks x m x N slabs. Each output is bitwise its own xf_mx8_gemm result (same body, its own slabs and fixup
/// counters). Ok(false) = not taken, nothing launched: the caller runs the two chains as before.
#[allow(clippy::too_many_arguments)]
pub(super) fn fx_pair(l: &Launcher, q0: &Quad, q1: &Quad, x: u64, y0: u64, y1: u64, ws0: (u64, usize), ws1: (u64, usize),
                      m: usize) -> Result<bool> {
    let (Some(FxW::Mx8 { w: w0, s: s0 }), Some(FxW::Mx8 { w: w1, s: s1 })) = (q0.fx.as_deref(), q1.fx.as_deref()) else {
        return Ok(false);
    };
    let (k, n, ks) = (q0.k as usize, q0.n as usize, q0.ks.max(1));
    if !(1..=16).contains(&m) || q1.k as usize != k || q1.n as usize != n || q1.ks.max(1) != ks
        || k % (128 * ks as usize) != 0 || (ks > 1 && (ks as usize * m * n > ws0.1 || ks as usize * m * n > ws1.1)) {
        return Ok(false);
    }
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        crate::drprintln!("[fp4l] paired MXFP8 GEMMs (GDN a|b, shared gate|up) run as ONE xf_mx8_gemm_x2 launch (bitwise per member)");
    }
    crate::metrics::fp4l_gemm("mxfp8", m);
    crate::metrics::fp4l_gemm("mxfp8", m);
    let cnt = |q: &Quad| if ks > 1 { *q.cnt.device_ptr() as u64 } else { 0 };
    let p = XfPair {
        w: [*w0.device_ptr() as u64, *w1.device_ptr() as u64],
        s: [*s0.device_ptr() as u64, *s1.device_ptr() as u64],
        y: [y0, y1],
        ws: [ws0.0, ws1.0],
        cnt: [cnt(q0), cnt(q1)],
    };
    xflaunch!(l, "xf_mx8_gemm_x2", (n.div_ceil(128) as u32, ks, 2), (256, 1, 1), 0,
              (p, x, m as i32, k as i32, n as i32, ks as i32))?;
    Ok(true)
}

/// The trellis linear's decode-path scratch (fp32 units): [ks][16][n] partials, then xh [16][k] f16, then y_raw
/// [16][n] f16. Returns (partials, xh, y_raw) device pointers.
fn tr_scratch(scr: &CudaSlice<f32>, k: usize, n: usize, ks: usize) -> Result<(u64, u64, u64)> {
    let need = tr_scratch_len(k, n, ks);
    anyhow::ensure!(scr.len() >= need, "FP4L MTP sidecar scratch {} < {need} floats", scr.len());
    let base = *scr.device_ptr() as u64;
    let part = base;
    let xh = base + (ks * 16 * n * 4) as u64;
    let yraw = xh + (16 * k * 2) as u64;
    Ok((part, xh, yraw))
}
fn tr_scratch_len(k: usize, n: usize, ks: usize) -> usize { ks * 16 * n + 8 * k + 8 * n }

/// K-slabs of a sidecar trellis linear at m <= 16: a pure shape function — the smallest divisor d of K/16 putting
/// >= 96 CTAs on the device (2 waves of 48 SMs), else the largest d leaving >= 4 k16 steps per slab.
fn tr_ks(k: usize, n: usize) -> usize {
    let (kb, tiles) = (k / 16, n / 128);
    let divs: Vec<usize> = (1..=kb).filter(|d| kb % d == 0 && kb / d >= 4).collect();
    for &d in &divs { if tiles * d >= 96 { return d; } }
    *divs.last().unwrap_or(&1)
}

/// The pack directory's EXL3 MTP-head sidecar (`--fp4l-mtp-sidecar auto`, draft-only): the file's header, by tensor name.
pub(crate) struct MtpSidecar {
    pub path: String,
    pub hdr: std::collections::BTreeMap<String, crate::exl3::HdrEntry>,
}

/// `--fp4l-mtp-sidecar auto|off` resolved against the model directory `dir`: auto (default, owner 2026-10-30) = the
/// `mtp-*.safetensors` in `dir` (the lexicographically last of several), if any — under TP the head ships that file to every node (shard
/// plan), so every rank resolves the same sidecar in its own model directory; off = none. No external-path form (owner 2026-11-01:
/// the pack decides — a sidecar is used exactly when the pack directory ships one).
pub(super) fn mtp_sidecar_path(dir: &str) -> Option<String> {
    match crate::opts::var(crate::opt!("fp4l-mtp-sidecar")).ok().as_deref().map(str::trim) {
        Some("off") | Some("none") | Some("0") => None,
        _ => crate::fp4l::mtp_sidecar_in_dir(std::path::Path::new(dir)).map(|p| p.to_string_lossy().into_owned()),
    }
}

pub(super) fn mtp_sidecar_open(path: &str) -> Result<MtpSidecar> {
    let hdr = crate::exl3::shard_header(path).with_context(|| format!("--fp4l-mtp-sidecar {path}"))?;
    anyhow::ensure!(hdr.keys().any(|k| k.starts_with("mtp.") && k.ends_with(".trellis")),
                    "--fp4l-mtp-sidecar {path}: no mtp.*.trellis tensors (not an EXL3 MTP-head sidecar)");
    Ok(MtpSidecar { path: path.to_string(), hdr })
}

/// The sidecar's trellis linear `stem` as an FP4L Quad (FxW::Trellis), or None when the sidecar has no such stem.
/// Integer K = 2..8 only (the K = 1.5 / 2.5 / 3.5 half widths are refused); mul1 codebook only.
pub(super) fn mtp_sidecar_quad(dev: &Arc<CudaDevice>, sc: &MtpSidecar, stem: &str) -> Result<Option<Quad>> {
    let get = |suffix: &str| sc.hdr.get(&format!("{stem}.{suffix}"));
    let (Some(tm), Some(su), Some(sv), Some(m1)) = (get("trellis"), get("suh"), get("svh"), get("mul1")) else { return Ok(None) };
    use std::os::unix::fs::FileExt;
    let f = std::fs::File::open(&sc.path).with_context(|| format!("open {}", sc.path))?;
    let read = |e: &crate::exl3::HdrEntry| -> Result<Vec<u8>> {
        let mut b = vec![0u8; e.n_bytes];
        f.read_exact_at(&mut b, e.offset as u64).with_context(|| format!("read {stem} from {}", sc.path))?;
        Ok(b)
    };
    let mul1 = u32::from_le_bytes(read(m1)?[..4].try_into().unwrap());
    anyhow::ensure!(mul1 == 0x83DC_D12D, "{stem}: codebook flag {mul1:#x} is not mul1 (0x83dcd12d)");
    anyhow::ensure!(tm.dtype == "I16" && tm.shape.len() == 3 && su.dtype == "F16" && sv.dtype == "F16", "{stem}: tensor types");
    let (kb, nb, w) = (tm.shape[0], tm.shape[1], tm.shape[2]);
    let k2 = crate::exl3::k2_from_words(w).map_err(|e| anyhow::anyhow!("{stem}: {e}"))?;
    anyhow::ensure!(k2 % 2 == 0 && (4..=16).contains(&k2), "{stem}: K = {} — the sidecar path serves integer K = 2..8", k2 as f64 / 2.0);
    let (k, n, bits) = (kb * 16, nb * 16, k2 / 2);
    anyhow::ensure!(k % 128 == 0 && n % 128 == 0 && su.shape == [k] && sv.shape == [n], "{stem}: [{k}, {n}] not 128-aligned or suh/svh shapes");
    let trb = read(tm)?;
    let tr16: Vec<i16> = trb.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
    let suh: Vec<u16> = read(su)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let svh: Vec<u16> = read(sv)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let weff = tr_effective_weight(&tr16, &suh, &svh, kb, nb, k2);
    let ks = tr_ks(k, n);
    let fx = FxW::Trellis {
        tr: dev.htod_sync_copy(&tr16.iter().map(|&v| v as u16).collect::<Vec<u16>>())?,
        suh: dev.htod_sync_copy(&suh)?, svh: dev.htod_sync_copy(&svh)?, bits, ks,
        scr: dev.alloc_zeros::<f32>(tr_scratch_len(k, n, ks))?,
        weff: dev.htod_sync_copy(&weff)?,
    };
    Ok(Some(fx_quad_wrap(dev, fx, k, n, false)?))
}

/// In-place orthonormal Walsh-Hadamard transform of a 128-vector (Sylvester order: exl3_had128's matrix).
fn fwht128(v: &mut [f32]) {
    let mut len = 1;
    while len < 128 {
        for i in (0..128).step_by(2 * len) {
            for j in i..i + len {
                let (a, b) = (v[j], v[j + len]);
                v[j] = a + b;
                v[j + len] = a - b;
            }
        }
        len <<= 1;
    }
    let s = 0.088_388_35_f32; // 1/sqrt(128)
    for x in v.iter_mut() { *x *= s; }
}

/// The effective weight of an EXL3 linear, transposed for the prefill cuBLASLt arm: f16 [n][k] of
/// diag(suh) H Wq H diag(svh) (Wq = the decoded trellis [k][n]; H = block-diagonal orthonormal H128), fp32 math,
/// one RN to f16. y = x W_eff equals had_svh(had_suh(x) Wq) up to the f16 roundings of the decode path.
fn tr_effective_weight(tr: &[i16], suh: &[u16], svh: &[u16], kb: usize, nb: usize, k2: usize) -> Vec<u16> {
    let (k, n) = (kb * 16, nb * 16);
    let wq = crate::exl3::decode_trellis_host(tr, kb, nb, k2); // [k][n] f16 bits
    let mut w: Vec<f32> = wq.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
    let suf: Vec<f32> = suh.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
    let svf: Vec<f32> = svh.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
    // along N (each row k): H over every 128-block, then * svh
    for row in w.chunks_exact_mut(n) {
        for blk in row.chunks_exact_mut(128) { fwht128(blk); }
        for (x, s) in row.iter_mut().zip(&svf) { *x *= *s; }
    }
    // along K (each column n): H over every 128-block of rows, then * suh (row scale)
    let mut col = vec![0f32; 128];
    for c in 0..n {
        for b in 0..k / 128 {
            for i in 0..128 { col[i] = w[(b * 128 + i) * n + c]; }
            fwht128(&mut col);
            for i in 0..128 { w[(b * 128 + i) * n + c] = col[i] * suf[b * 128 + i]; }
        }
    }
    // transpose to [n][k] f16
    let mut out = vec![0u16; n * k];
    for kk in 0..k {
        for c in 0..n { out[c * k + kk] = half::f16::from_f32(w[kk * n + c]).to_bits(); }
    }
    out
}

/// `--fp4l-prefill-dense exact|lt` (default exact): the m > 16 MXFP8 GEMMs as the exact W8A16 kernel
/// (xf_mx8_tc: per-group fp32 scaling, the decode kernels' numerics) or as an f16 dequant into the recon
/// staging + cuBLASLt (faster; rounds the 56,179 MXFP8 weights of this pack that lie below f16's exact range).
pub(super) fn prefill_dense_lt() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::var(crate::opt!("fp4l-prefill-dense")).as_deref() == Ok("lt"))
}

/// m > 16 (prefill, head fill). MXFP8: the exact smem-tiled xf_mx8_tc (default) or dequant + cuBLASLt (opt-in).
/// F16: cuBLASLt straight on the stored (pre-scaled) weight with alpha = osc (exact power of two).
/// y = x · W^T, opA = T on the row-major [n, k] weight (the xq_gemm_f16 convention). `y` is f16.
pub(super) fn fx_wide(l: &Launcher, q: &Quad, fx: &FxW, x: u64, y: u64, m: usize, r: &ReconBufs) -> Result<()> {
    let (k, n) = (q.k as usize, q.n as usize);
    match fx {
        FxW::Mx8 { w, s } if !prefill_dense_lt() => {
            // xf_mx8_tc: the smem-tiled exact GEMM (bit-identical to xf_mx8_wide), 1-D grouped grid
            let ctas = n.div_ceil(128) * m.div_ceil(128);
            xflaunch!(l, "xf_mx8_tc", (ctas as u32, 1, 1), (256, 1, 1), 0,
                      (w, s, x, y, 0u64, m as i32, k as i32, n as i32))
        }
        FxW::Mx8 { w, s } => {
            anyhow::ensure!(n * k <= r.ws_len, "FP4L recon staging too small: need {} have {}", n * k, r.ws_len);
            let ne = (n * k) as i64;
            xflaunch!(l, "xf_mx8_dequant", (((ne / 8) as u32).div_ceil(256), 1, 1), (256, 1, 1), 0,
                      (w, s, r.ws, ne, k as i32))?;
            exl3_lt_matmul(l, r.ws, x, y, k, n, m, true, r.ltws, r.ltws_len)
        }
        FxW::F16 { w, osc } => exl3_lt_matmul_alpha(l, *w.device_ptr() as u64, x, y, k, n, m, true, r.ltws, r.ltws_len, *osc),
        // --fp4l-mtp-sidecar: the load-time effective f16 weight through cuBLASLt (the MTP prime over a prompt chunk)
        FxW::Trellis { weff, .. } => exl3_lt_matmul_alpha(l, *weff.device_ptr() as u64, x, y, k, n, m, true, r.ltws, r.ltws_len, 1.0),
        FxW::Bf16 { w } => {
            let _ = r;
            xflaunch!(l, "xf_bf16_tc", ((n.div_ceil(128) * m.div_ceil(128)) as u32, 1, 1), (256, 1, 1), 0,
                      (w, x, y, 0u64, m as i32, k as i32, n as i32))
        }
    }
}

/// The FP4L DFlash2 head: the pre-scaled f16 lm_head back to its stored bf16 values (exact).
pub(super) fn f16_to_bf16_scaled(l: &Launcher, out: u64, w: &CudaSlice<u16>, scale: f32) -> Result<()> {
    let n = w.len() as i64;
    xflaunch!(l, "xf_f16_to_bf16_scaled", (((n + 255) / 256).min(65535) as u32, 1, 1), (256, 1, 1), 0, (out, w, n, scale))
}

/// A u16 bit copy of `n` elements on the compute stream (src / dst raw device pointers).
pub(super) fn copy_u16(l: &Launcher, dst: u64, src: u64, n: usize) -> Result<()> {
    let n = n as i64;
    xflaunch!(l, "xf_copy_u16", (((n + 255) / 256).min(65535) as u32, 1, 1), (256, 1, 1), 0, (dst, src, n))
}

/// The verify head as bf16 [n][k] at `out` (the DFlash2 drafter's head view): the stored bits (Bf16), the exact
/// unshifted cast (F16), or the exact bf16 dequant of the MXFP8 copy (Mx8).
pub(super) fn head_to_bf16(l: &Launcher, out: u64, fx: &FxW, n: usize, k: usize) -> Result<()> {
    // the REAL rows (F64 / B32 pad N to a multiple of 16; the padding is never copied out)
    let ne = (n * k) as i64;
    anyhow::ensure!(ne % 8 == 0 && k % 32 == 0, "FP4L head_to_bf16: {n} x {k}");
    match fx {
        FxW::Trellis { .. } => bail!("FP4L head_to_bf16: the head is never a sidecar trellis linear"),
        FxW::F16 { w, osc } => f16_to_bf16_scaled(l, out, w, *osc),
        FxW::Bf16 { w } => {
            anyhow::ensure!(w.len() >= n * k, "FP4L head_to_bf16: B32 head {} < {n} x {k}", w.len());
            xflaunch!(l, "xf_b32_to_rm", (((ne / 8) as u32).div_ceil(256), 1, 1), (256, 1, 1), 0, (out, w, ne, k as i32))
        }
        FxW::Mx8 { w, s } => {
            anyhow::ensure!(w.len() >= n * k, "FP4L head_to_bf16: F64 head {} < {n} x {k}", w.len());
            xflaunch!(l, "xf_mx8_dequant_bf16", (((ne / 8) as u32).div_ceil(256), 1, 1), (256, 1, 1), 0, (w, s, out, ne, k as i32))
        }
    }
}

/// TP row-parallel partial (m <= 16): this rank's fp32 result over its K-slice as ks fp32 slabs in `ws`
/// (`[ks][m][n]`; ks == 1 = one slab), for the caller's ascending xq_ks_combine_f32* into the TP
/// partial buffer — the exl3_hmma_gemm_ks + combine contract of the trellis row-parallel chain.
pub(super) fn fx_rowpar_small(l: &Launcher, q: &Quad, fx: &FxW, x: &CudaSlice<u16>, ws: &CudaSlice<f32>, m: usize) -> Result<()> {
    crate::metrics::fp4l_gemm(fx.kind(), m);
    anyhow::ensure!(q.ks.max(1) as usize * m * q.n as usize <= ws.len(), "FP4L row-parallel chain_ws too small");
    fx_small(l, q, fx, *x.device_ptr() as u64, 0, *ws.device_ptr() as u64, ws, m)
}

// ------------------------------------------------------------------------------------------------
// MoE dispatch
// ------------------------------------------------------------------------------------------------

fn nv4_moe_fn() -> &'static str { if e2m1_int() { "xf_nv4_moe_i" } else { "xf_nv4_moe" } }

/// The tile-list classes the smem-tiled xf_nv4_tc serves (mt 4 and 8: 64-row sub-blocks); bit-identical to
/// xf_nv4_tiles_m4 there. Classes 1 and 2 keep xf_nv4_tiles_m1 / m2 (a 64-row CTA would idle most of its rows).
fn nv4_tc_fn(mt: usize) -> Option<&'static str> {
    if mt >= 4 && mt % 4 == 0 { Some(if e2m1_int() { "xf_nv4_tc_i" } else { "xf_nv4_tc" }) } else { None }
}

fn nv4_tiles_fn(mt: usize) -> (&'static str, usize) {
    let int = e2m1_int();
    // the kernel's register tile: MT 1, 2 or 4 token tiles; a list built for mt = 8 is walked in
    // two MT-4 halves (the kernel loops over sub-blocks of the list's 16*mt rows).
    match (mt.min(4), int) {
        (1, false) => ("xf_nv4_tiles_m1", 1), (2, false) => ("xf_nv4_tiles_m2", 2), (_, false) => ("xf_nv4_tiles_m4", 4),
        (1, true) => ("xf_nv4_tiles_m1_i", 1), (2, true) => ("xf_nv4_tiles_m2_i", 2), (_, true) => ("xf_nv4_tiles_m4_i", 4),
    }
}

/// Decode / verify (m <= 16) routed experts after the trunk's routing: gate|up -> sc.ygu
/// ([slot][m][2*mi]), silu(g)*u -> sc.din (xq_moe_gate_mul), down -> sc.yd ([slot][m][h]); the
/// trunk's shared expert and combine follow unchanged.
pub(super) fn moe_experts_decode(l: &Launcher, sc: &mut Scratch, fx: &FxMoe, h: usize, mi: usize, m: usize, emax: i32) -> Result<()> {
    anyhow::ensure!((1..=16).contains(&m) && (2 * mi) % 64 == 0 && h % 64 == 0 && h % 64 == 0 && mi % 64 == 0,
                    "FP4L MoE decode: m {m}, h {h}, mi {mi} outside the xf_nv4_moe contract");
    crate::metrics::fp4l_moe(m);
    let f = nv4_moe_fn();
    xflaunch!(l, f, (((2 * mi) / 64) as u32, emax as u32, 1), (128, 1, 1), 0,
              (&fx.gu_w, &fx.gu_s, &fx.gs, &sc.x, 0i64, &mut sc.ygu, &sc.idxmap, &sc.esel,
               m as i32, h as i32, (2 * mi) as i32, mi as i32))?;
    xqlaunch!(l, "xq_moe_gate_mul", (((emax as usize * m * mi) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
              (&mut sc.din, &sc.ygu, (m * mi) as i64, &sc.esel))?;
    xflaunch!(l, f, ((h / 64) as u32, emax as u32, 1), (128, 1, 1), 0,
              (&fx.dn_w, &fx.dn_s, &fx.gs, &sc.din, (m * mi) as i64, &mut sc.yd, &sc.idxmap, &sc.esel,
               m as i32, mi as i32, h as i32, h as i32))
}

/// The prefill MoE layer of an FP4-Local pack, FP4L-owned: a copy of the device-routed arm of the trunk's
/// `FwdModel::moe_prefill` (base d4615ea9) with the NVFP4 expert tiles in place of the trellis kernels, so the trunk
/// function carries ONE dispatch line and its later changes neither reach this path nor conflict with it. The
/// launches are the ones the trunk function made for this pack before the split (output bit-identical): router
/// (the stored-bf16 twin by default, the trunk's f16 kernels under --fp4l-plain f16; the own rows + one all-gather
/// under --tp-seq-parallel router), device routing (count / place / EP fill; the per-expert MT classes when
/// prefill.moe_mt_pe is on), the expert tiles, then the trunk's tail (shared expert + combine). The host-routed
/// trellis arm (--exl3-moe-recon-min / --exl3-moe-devroute=0) cannot serve NVFP4 experts: those EXL3 levers are
/// not applied here (one line at the first prefill that asks for them).
pub(super) fn moe_prefill(fm: &FwdModel, l: &Launcher, psc: &mut PrefillScratch, moe: &MoeLayer, c: usize,
                          rb: &ReconBufs) -> Result<()> {
    let fx = moe.fx.as_ref().context("FP4L prefill MoE: a stored-bf16 router on a layer without NVFP4 experts")?;
    let (h, ne, topk, mi) = (fm.cfg.hidden_size, fm.cfg.num_experts, fm.cfg.num_experts_per_tok, fm.cfg.moe_intermediate_size);
    let mt: usize = match tune::get(&T_WIDE_MT, &l.cx()) { v @ (1 | 2 | 4 | 8) => v as usize, _ => pfx1_moe_mt(c) };
    let rows_on = tune::get(&T_PF_ROUTER_ROWS, &l.cx()) != 0;
    let (lp, xp) = (*psc.logits_r.device_ptr() as u64, *psc.x.device_ptr() as u64);
    let (ip, wp) = (*psc.ids.device_ptr() as u64, *psc.wts.device_ptr() as u64);
    // router logits + softmax/top-k of rows [lo, lo + n) (every row depends only on its own x row)
    let route = |lo: usize, n: usize| -> Result<()> {
        let (lpo, xpo) = (lp + (lo * ne * 4) as u64, xp + (lo * h * 2) as u64);
        let (ipo, wpo) = (ip + (lo * topk * 4) as u64, wp + (lo * topk * 4) as u64);
        if moe.router_bf {
            crate::metrics::fp4l_twin("router");
            bf_wide(l, &moe.router, xpo, 0, lpo, n, h, ne)?;
            router_topk(l, ipo, wpo, lpo, ne, topk, n)
        } else {
            router_rows_launch(l, lpo, *moe.router.device_ptr() as u64, xpo, n, c, ne, h, rows_on)?;
            xqlaunch!(l, "xq_router_topk", (n as u32, 1, 1), (256, 1, 1), ((256 + ne + 512) * 4) as u32,
                      (ipo, wpo, lpo, ne as i32, topk as i32, n as i32))
        }
    };
    let sp_router = psc.sp_rows.filter(|_| xtp::seq_parallel_router() && (topk * 4) % 16 == 0);
    if let (Some(s), Some(tp)) = (sp_router, fm.tp.as_ref()) {
        anyhow::ensure!(s.c == c, "TP-SP router: chunk {} vs MoE c {c}", s.c);
        route(s.own.0, s.own.1 - s.own.0)?;
        tp.sp_all_gather_multi(l, &[(ip, topk * 4), (wp, topk * 4)], &s)?;
    } else {
        route(0, c)?;
    }
    let host_routing = crate::opts::var(crate::opt!("exl3-moe-recon-min")).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0) > 0
        || crate::opts::var(crate::opt!("exl3-moe-devroute")).as_deref() == Ok("0");
    if host_routing {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            crate::rprintln!("[fp4l] --exl3-moe-recon-min / --exl3-moe-devroute=0 select the EXL3 host-routed trellis arm, which has no \
NVFP4 form: the FP4-Local experts keep the device-routed tile lists");
        }
    }
    anyhow::ensure!(psc.pf_tiles.len() > 1, "FP4L prefill MoE: the prefill scratch has no expert tile lists");
    let ep = moe.ep.as_ref();
    let r = c * topk;
    // TP-B (expert parallel): count/place over THIS rank's local expert space (ids_ep: local stacked id or -1)
    let ne_eff = ep.map_or(ne, |e| e.n_local);
    if let Some(e) = ep {
        xqlaunch!(l, "xq_moe_ids_ep", (r.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0,
                  (&psc.ids, &mut psc.ids_ep, &mut psc.cand, &e.lut, r as i32, ne as i32))?;
    }
    let ids_route = if ep.is_some() { *psc.ids_ep.device_ptr() as u64 } else { ip };
    let ne = ne_eff;
    let max_tiles = (r / 16 + ne) as u32;
    // ONE launch group at the chunk-rule MT (the launches FP4L made before its tile lists moved to the pf_same line;
    // the trunk's per-expert MT classes (prefill.moe_mt_per_expert) measured 2-4 % slower TTFT on the NVFP4 tiles)
    let pe = false;
    // (mt, tiles ptr, ntiles ptr, grid-y cap) per launch group
    let tl_base = *psc.pf_tiles4.device_ptr() as u64;
    let nt_base = *psc.pf_ntiles4.device_ptr() as u64;
    let groups: Vec<(usize, u64, u64, u32)> = if pe {
        let caps = moe_pf_class_caps(r, ne);
        (0..4).filter(|&k| caps[k] > 0)
            .map(|k| (1usize << k, tl_base + (k * max_tiles as usize * 2 * 4) as u64, nt_base + (k * 4) as u64, caps[k] as u32))
            .collect()
    } else {
        vec![(mt, *psc.pf_tiles.device_ptr() as u64, *psc.pf_ntiles.device_ptr() as u64, max_tiles)]
    };
    if pe {
        xqlaunch!(l, "xq_moe_pf_count4", (1, 1, 1), (1024, 1, 1), 0,
                  (ids_route, r as i32, ne as i32, &mut psc.cnt, &mut psc.offs_row, &mut psc.pf_tiles4,
                   &mut psc.pf_ntiles4, max_tiles as i32))?;
    } else {
        xqlaunch!(l, "xq_moe_pf_count", (1, 1, 1), (1024, 1, 1), 0,
                  (ids_route, r as i32, ne as i32, &mut psc.cnt, &mut psc.offs_row, &mut psc.pf_tiles,
                   &mut psc.pf_ntiles, (16 * mt) as i32))?;
    }
    xqlaunch!(l, "xq_moe_pf_place", (ne as u32, 1, 1), (256, 1, 1), 0,
              (ids_route, r as i32, topk as i32, &psc.offs_row, &mut psc.row_tok, &mut psc.row_eidx, &mut psc.cand))?;
    if ep.is_some() {
        // rows [offs_row[ne_l], r) hold no local pick: give them a valid index (see kernel)
        xqlaunch!(l, "xq_moe_pf_ep_fill", (r.div_ceil(256).min(1024) as u32, 1, 1), (256, 1, 1), 0,
                  (&psc.offs_row, ne as i32, r as i32, &mut psc.row_tok, &mut psc.row_eidx))?;
    }
    moe_experts_prefill(l, psc, fx, h, mi, r, &groups)?;
    fm.moe_prefill_tail(l, psc, moe, c, rb)
}

/// Prefill routed experts over the trunk's device tile lists (xq_moe_pf_count/place already ran):
/// gate|up tiles read x through row_tok -> psc.ygu ([r][2*mi], compact rows), gate_mul -> psc.din,
/// down tiles -> psc.yd ([r][h]); moe_prefill_tail combines.
#[allow(clippy::too_many_arguments)]
pub(super) fn moe_experts_prefill(l: &Launcher, psc: &mut PrefillScratch, fx: &FxMoe, h: usize, mi: usize, r: usize,
                                  groups: &[(usize, u64, u64, u32)]) -> Result<()> {
    anyhow::ensure!(h < (1 << 15) && 2 * mi < (1 << 15) && (2 * mi) % 128 == 0 && h % 128 == 0,
                    "FP4L prefill MoE: h {h} / mi {mi} outside the xf_nv4_tiles contract");
    crate::metrics::fp4l_moe(17);
    let kn = |k: usize, n: usize| (k | (n << 16)) as i32;
    for &(gmt, tp, np, cap) in groups {
        let (f, kmt) = nv4_tiles_fn(gmt);
        anyhow::ensure!(gmt % kmt == 0, "FP4L prefill MoE: tile list mt {gmt} not a multiple of the kernel MT {kmt}");
        let f = nv4_tc_fn(gmt).unwrap_or(f);
        xflaunch!(l, f, (((2 * mi) / 128) as u32, cap, 1), (256, 1, 1), 0,
                  (&fx.gu_w, &fx.gu_s, &fx.gs, &psc.x, &psc.row_tok, &mut psc.ygu, tp, np,
                   &psc.offs_row, &psc.cnt, kn(h, 2 * mi), (mi | (gmt << 16)) as i32))?;
    }
    xqlaunch!(l, "xq_moe_gate_mul", (((r * mi) as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
              (&mut psc.din, &psc.ygu, mi as i64, &psc.moerows))?;
    for &(gmt, tp, np, cap) in groups {
        let (f, _) = nv4_tiles_fn(gmt);
        let f = nv4_tc_fn(gmt).unwrap_or(f);
        xflaunch!(l, f, ((h / 128) as u32, cap, 1), (256, 1, 1), 0,
                  (&fx.dn_w, &fx.dn_s, &fx.gs, &psc.din, 0u64, &mut psc.yd, tp, np,
                   &psc.offs_row, &psc.cnt, kn(mi, h), (h | (gmt << 16)) as i32))?;
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------
// Draft head (draft passes only; the verify keeps the stored bf16 head)
// ------------------------------------------------------------------------------------------------

/// `--fp4l-draft-head head|mxfp8` (default head = the verify head's rows in its own format, --fp4l-lm-head):
/// the pruned draft lm_head. mxfp8 (opt-in) = an MXFP8 RTN copy of the stored bf16 rows (E4M3 with a UE8M0
/// scale per 32 — the format's own rule), about half the bytes of a bf16 draft pass. Draft-only either way:
/// the verify scores with the verify head, so the served tokens (greedy) and distribution (sampled) are
/// unchanged; only acceptance can move.
pub(super) fn draft_head_mode() -> &'static str {
    static V: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    // default screen (owner 2026-10-30: the draft-only levers are defaults; draft-only = served output unchanged)
    *V.get_or_init(|| match crate::opts::var(crate::opt!("fp4l-draft-head")).as_deref() {
        Ok("mxfp8") => "mxfp8",
        Ok("head") => "head",
        _ => "screen",
    })
}

/// The candidate blocks of the draft-head screen (== XF_DHS_T in gpu_fp4l.cu).
pub(super) const DHS_T: usize = 16;

/// `--fp4l-draft-head screen` (DRAFT-ONLY): an int4 per-row-scaled copy of the draft head's rows picks the DHS_T best
/// 128-row blocks; those rows are rescored exactly (the full GEMV's bits) and every other row reads -inf. See
/// xf_dhs_* in gpu_fp4l.cu.
pub(crate) struct FxDhs {
    /// int4 [n2][k/2] (low nibble = even k), per-row fp32 scale
    pub q: CudaSlice<u8>,
    pub s: CudaSlice<f32>,
    pub bmax: CudaSlice<f32>,
    pub cand: CudaSlice<i32>,
    pub n2: usize,
    pub k: usize,
}

/// Build the screen from the stored bf16 rows [0, n2) of the lm_head (host quantize: s = max|v| / 7, q = RN(v / s)
/// clamped to [-7, 7]; parallel over rows).
pub(super) fn dhs_build(dev: &Arc<CudaDevice>, fp: &Fp4lPack, n2: usize) -> Result<FxDhs> {
    let (n, k) = fp.shape2("lm_head.weight")?;
    anyhow::ensure!(n2 <= n && n2 % 128 == 0 && n2 / 128 <= 512 && k % 64 == 0 && k <= 4096,
                    "FP4L draft-head screen: {n2} rows of [{n}, {k}] outside the kernels' contract");
    let raw = fp.raw_range("lm_head.weight", 0, (n2 * k * 2) as u64)?;
    let mut q = vec![0u8; n2 * k / 2];
    let mut sc = vec![0f32; n2];
    let th = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).clamp(1, 16);
    let per = n2.div_ceil(th);
    std::thread::scope(|sco| {
        for ((rc, qc), scc) in raw.chunks(per * k * 2).zip(q.chunks_mut(per * k / 2)).zip(sc.chunks_mut(per)) {
            sco.spawn(move || {
                for (r, (row, qrow)) in rc.chunks_exact(k * 2).zip(qc.chunks_exact_mut(k / 2)).enumerate() {
                    let v: Vec<f32> = row.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
                    let amax = v.iter().fold(0f32, |a, &x| a.max(x.abs()));
                    let s = if amax > 0.0 { amax / 7.0 } else { 1.0 };
                    scc[r] = s;
                    let qv = |x: f32| -> u8 { ((x / s).round().clamp(-7.0, 7.0) as i8 as u8) & 0xF };
                    for (j, b) in qrow.iter_mut().enumerate() { *b = qv(v[2 * j]) | (qv(v[2 * j + 1]) << 4); }
                }
            });
        }
    });
    Ok(FxDhs {
        q: dev.htod_sync_copy(&q)?, s: dev.htod_sync_copy(&sc)?,
        bmax: dev.alloc_zeros::<f32>(n2 / 128)?, cand: dev.htod_sync_copy(&vec![0i32; DHS_T])?, n2, k,
    })
}

/// One screened draft-head pass (m = 1, graph-capturable): `x` the normed hidden [k] f16, `head` the draft head
/// (B32 rows, FxW::Bf16), `logits` [n2] f16 out.
pub(super) fn dhs_logits(l: &Launcher, dhs: &FxDhs, head: &Quad, x: u64, logits: u64) -> Result<()> {
    let Some(FxW::Bf16 { w }) = head.fx.as_deref() else { bail!("FP4L draft-head screen: the draft head is not stored bf16") };
    let nblk = dhs.n2 / 128;
    crate::metrics::fp4l_gemm("bf16", 1);
    xflaunch!(l, "xf_dhs_screen", (nblk as u32, 1, 1), (256, 1, 1), 0, (&dhs.q, &dhs.s, x, &dhs.bmax, logits, dhs.k as i32))?;
    xflaunch!(l, "xf_dhs_select", (1, 1, 1), (512, 1, 1), 0, (&dhs.bmax, &dhs.cand, nblk as i32))?;
    xflaunch!(l, "xf_dhs_rescore", (DHS_T as u32, 1, 1), (256, 1, 1), 0, (w, x, logits, &dhs.cand, dhs.k as i32))
}

/// OCP MX RTN of one 32-value group: shared exponent = floor(log2(amax)) - 8 (E4M3 emax), codes RNE.
pub(crate) fn mx8_quant_group(v: &[f32], codes: &mut [u8]) -> u8 {
    let amax = v.iter().fold(0f32, |a, &x| a.max(x.abs()));
    if amax == 0.0 { codes.fill(0); return 127; }
    let e = amax.log2().floor() as i32 - 8;
    let e = e.clamp(-127, 127);
    let s = 2f32.powi(e);
    for (c, &x) in codes.iter_mut().zip(v) { *c = f32_to_e4m3_rne(x / s); }
    (e + 127) as u8
}

/// f32 -> OCP E4M3 (fn), round to nearest even, saturating at 448.
pub(crate) fn f32_to_e4m3_rne(x: f32) -> u8 {
    if x.is_nan() { return 0x7F; }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs().min(448.0);
    // enumerate the 127 non-negative finite codes: exact search of the nearest (ties to even code)
    let mut best = 0u8;
    let mut bd = f32::INFINITY;
    for c in 0u8..=0x7E {
        let v = crate::fp4l::pack::e4m3(c);
        let d = (v - a).abs();
        if d < bd || (d == bd && c % 2 == 0) { bd = d; best = c; }
    }
    sign | best
}

/// Build the MXFP8 draft copy of rows [0, n2) of the stored bf16 lm_head (host RTN, then upload).
pub(super) fn draft_head_mx8(dev: &Arc<CudaDevice>, fp: &Fp4lPack, n2: usize) -> Result<Quad> {
    let (n, k) = fp.shape2("lm_head.weight")?;
    anyhow::ensure!(n2 <= n && k % 32 == 0, "FP4L draft head: {n2} rows of [{n}, {k}]");
    let raw = fp.raw_range("lm_head.weight", 0, (n2 * k * 2) as u64)?;
    let vals: Vec<f32> = raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
    let (codes, scales) = mx8_rtn(&vals, n2, k);
    fx_quad_wrap(dev, mx8_dev(dev, &codes, &scales, n2, k)?, k, n2, false)
}

/// OCP MX RTN of a row-major [n][k] f32 matrix (k % 32 == 0): E4M3 codes + one UE8M0 scale per 32 (shared
/// exponent floor(log2 amax) - 8, codes RNE, saturating at 448). Host, parallel over row blocks.
pub(crate) fn mx8_rtn(vals: &[f32], n2: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let mut codes = vec![0u8; n2 * k];
    let mut scales = vec![0u8; n2 * k / 32];
    // precomputed e4m3 grid for a fast RNE search (127 values, monotone)
    let grid: Vec<f32> = (0u8..=0x7E).map(crate::fp4l::pack::e4m3).collect();
    let q = |x: f32| -> u8 {
        let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
        let a = x.abs().min(448.0);
        let i = grid.partition_point(|&g| g < a);
        let c = if i == 0 { 0 } else if i >= grid.len() { grid.len() - 1 } else {
            let (lo, hi) = (grid[i - 1], grid[i]);
            if a - lo < hi - a { i - 1 } else if hi - a < a - lo { i } else if (i - 1) % 2 == 0 { i - 1 } else { i }
        };
        sign | c as u8
    };
    let nt = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).clamp(1, 16);
    let rows_per = n2.div_ceil(nt);
    std::thread::scope(|s| {
        for ((vc, cc), sc) in vals.chunks(rows_per * k).zip(codes.chunks_mut(rows_per * k)).zip(scales.chunks_mut(rows_per * k / 32)) {
            let q = &q;
            s.spawn(move || {
                for (g, (v, c)) in vc.chunks(32).zip(cc.chunks_mut(32)).enumerate() {
                    let amax = v.iter().fold(0f32, |a, &x| a.max(x.abs()));
                    if amax == 0.0 { c.fill(0); sc[g] = 127; continue; }
                    let e = (amax.log2().floor() as i32 - 8).clamp(-127, 127);
                    let s = 2f32.powi(e);
                    for (ci, &x) in c.iter_mut().zip(v) { *ci = q(x / s); }
                    sc[g] = (e + 127) as u8;
                }
            });
        }
    });
    (codes, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every entry this file launches is in XF_FNS (the module load list): a launched name missing there fails at
    /// the first dispatch on the GPU (`kernel ... missing from gpu_fp4l`), which no host test would otherwise see.
    #[test]
    fn every_launched_entry_is_loaded() {
        let src = include_str!("fp4l_fwd.rs");
        let mut missing = Vec::new();
        let mut n = 0;
        for (i, pat) in src.match_indices("xflaunch!(") {
            // the second macro argument, when it is a string literal (a name expression such as nv4_tiles_fn's
            // result is a row of XF_FNS by construction)
            let rest = &src[i + pat.len()..];
            let Some(c) = rest.find(',') else { continue };
            let arg = rest[c + 1..].trim_start();
            let Some(lit) = arg.strip_prefix('"') else { continue };
            let Some(e) = lit.find('"') else { continue };
            let name = &lit[..e];
            n += 1;
            if !XF_FNS.contains(&name) && !missing.contains(&name) { missing.push(name); }
        }
        assert!(n > 20, "found only {n} literal launches: the scan is broken");
        assert!(missing.is_empty(), "launched but not in XF_FNS: {missing:?}");
    }

    /// xf_mx8_gemm_x2 takes `struct XfPair` by value: five 2-pointer arrays, 80 bytes, member z at index z.
    #[test]
    fn xf_pair_matches_the_kernel_struct() {
        assert_eq!(std::mem::size_of::<XfPair>(), 80);
        assert_eq!(std::mem::align_of::<XfPair>(), 8);
        let p = XfPair { w: [1, 2], s: [3, 4], y: [5, 6], ws: [7, 8], cnt: [9, 10] };
        let words: [u64; 10] = unsafe { std::mem::transmute(p) };
        assert_eq!(words, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "field order = the kernel's w, s, y, ws, cnt");
    }

    /// Host emulation of the kernel's integer E2M1 path (PRMT LUT + sign bits) against the table.
    fn prmt(a: u32, b: u32, sel: u32) -> u32 {
        let bytes = [(a & 0xFF) as u8, (a >> 8) as u8, (a >> 16) as u8, (a >> 24) as u8,
                     (b & 0xFF) as u8, (b >> 8) as u8, (b >> 16) as u8, (b >> 24) as u8];
        let mut r = 0u32;
        for i in 0..4 {
            let s = (sel >> (4 * i)) & 0xF;
            let mut v = bytes[(s & 7) as usize];
            if s & 8 != 0 { v = if v & 0x80 != 0 { 0xFF } else { 0 }; }
            r |= (v as u32) << (8 * i);
        }
        r
    }
    fn e2m1x8_int(w: u32) -> [u32; 4] {
        let (lut_lo, lut_hi) = (0x3E3C3800u32, 0x46444240u32);
        let mlo = prmt(lut_lo, lut_hi, w & 0x7777);
        let mhi = prmt(lut_lo, lut_hi, (w >> 16) & 0x7777);
        let s = w & 0x88888888;
        let (r0, r1, r2, r3) = (prmt(mlo, 0, 0x1404), prmt(mlo, 0, 0x3424), prmt(mhi, 0, 0x1404), prmt(mhi, 0, 0x3424));
        [r0 | ((s << 12) & 0x8000) | ((s << 24) & 0x8000_0000),
         r1 | ((s << 4) & 0x8000) | ((s << 16) & 0x8000_0000),
         r2 | ((s >> 4) & 0x8000) | ((s << 8) & 0x8000_0000),
         r3 | ((s >> 12) & 0x8000) | (s & 0x8000_0000)]
    }

    #[test]
    fn e2m1_int_path_matches_the_table_for_every_nibble_pair() {
        use crate::fp4l::pack::E2M1;
        for lo in 0..16u32 {
            for hi in 0..16u32 {
                for pos in 0..4 {
                    let w = (lo | (hi << 4)) << (8 * pos);
                    let o = e2m1x8_int(w);
                    let h = o[pos];
                    let (a, b) = (half::f16::from_bits((h & 0xFFFF) as u16).to_f32(), half::f16::from_bits((h >> 16) as u16).to_f32());
                    assert_eq!(a.to_bits(), E2M1[lo as usize].to_bits(), "lo {lo} pos {pos}");
                    assert_eq!(b.to_bits(), E2M1[hi as usize].to_bits(), "hi {hi} pos {pos}");
                }
            }
        }
    }

    #[test]
    fn ks_is_a_function_of_shape_and_divides_k() {
        for &(k, n) in &[(2560usize, 10240usize), (2560, 6144), (6144, 2560), (2560, 12288), (2560, 512), (6144, 2560),
                         (2560, 640), (640, 2560), (1536, 2560), (2560, 3072), (2560, 248320), (2560, 65536), (384, 2560), (128, 2560)] {
            let ks = fx_ks(k, n) as usize;
            assert!(ks >= 1 && (k / 128) % ks == 0, "k {k} n {n} ks {ks}");
            assert_eq!(fx_ks(k, n), fx_ks(k, n));
        }
        assert_eq!(fx_ks(2560, 248320), 1);
        assert_eq!(fx_ks(2560, 10240), 4);
    }

    /// The trunk combine's gate tree, on the host: per-lane partials over c = t, t + bs, ... (`part += sg * x`, an
    /// fma), then the halving tree of `bs` leaves (xq_moe_combine / _rows / _ep / _rows_ep: bs = 1024).
    fn tree_halving(sgc: &[u16], xg: &[u16], bs: usize) -> f32 {
        let h = sgc.len();
        let f = |b: u16| half::f16::from_bits(b).to_f32();
        let mut sm = vec![0f32; bs];
        for (t, v) in sm.iter_mut().enumerate() {
            let mut part = 0f32;
            let mut c = t;
            while c < h { part = f(sgc[c]).mul_add(f(xg[c]), part); c += bs; }
            *v = part;
        }
        let mut s2 = bs / 2;
        while s2 > 0 { for t in 0..s2 { sm[t] += sm[t + s2]; } s2 /= 2; }
        sm[0]
    }
    /// xq_moe_combine_rows_ep2's tree: 128 lanes x 8 columns (v = 8 tid + u over c0 = 8 tid + 1024 j), leaves
    /// re-paired in one warp (levels 512..32 in registers, 16..1 by shuffle) — the same pairs as the halving tree.
    fn tree_ep2(sgc: &[u16], xg: &[u16]) -> f32 {
        let h = sgc.len();
        let f = |b: u16| half::f16::from_bits(b).to_f32();
        let mut sm = vec![0f32; 1024];
        for tid in 0..128 {
            let mut part = [0f32; 8];
            let mut c0 = tid * 8;
            while c0 < h { for u in 0..8 { part[u] = f(sgc[c0 + u]).mul_add(f(xg[c0 + u]), part[u]); } c0 += 1024; }
            for u in 0..8 { sm[tid * 8 + u] = part[u]; }
        }
        let mut lanes = vec![0f32; 32];
        for (l, lv) in lanes.iter_mut().enumerate() {
            let mut a: Vec<f32> = (0..32).map(|j| sm[l + 32 * j]).collect();
            for (n, _) in [(16, 512), (8, 256), (4, 128), (2, 64)] { for j in 0..n { a[j] = a[j] + a[j + n]; } }
            *lv = a[0] + a[1];
        }
        let mut s2 = 16;
        while s2 > 0 { for t in 0..s2 { lanes[t] += lanes[t + s2]; } s2 /= 2; }
        lanes[0]
    }

    #[test]
    fn shared_gate_channels_rebuild_the_logit_in_every_combine_tree() {
        let h = 2560;
        let sgc = sg_channel_constants(h);
        // the constants are what the kernel's up-scales assume
        let want = [1.0f32, 2f32.powi(-11), 2f32.powi(-22), 2f32.powi(-24), 2f32.powi(-24), 2f32.powi(-24)];
        for i in 0..6 { assert_eq!(half::f16::from_bits(sgc[i]).to_f32(), want[i], "sgc[{i}]"); }
        assert!(sgc[6..].iter().all(|&b| b == 0));
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
        let mut checked = 0;
        for i in 0..200_000 {
            // log-uniform magnitudes 2^-25 .. 2^16, random sign and mantissa bits, plus edge cases
            let r = next();
            let mut l = f32::from_bits(((((r >> 32) as u32) % 41 + 102) << 23) | ((r as u32) & 0x7F_FFFF));
            if r >> 63 == 1 { l = -l; }
            if i < 8 { l = [0.0f32, -0.0, 65504.0, -65504.0, 2f32.powi(-25), 1.0, -0.5, 65503.0][i]; }
            if l.abs() > 65504.0 { continue; }
            let ch = sg_encode(l);
            let mut xg = vec![0u16; h];
            xg[..6].copy_from_slice(&ch);
            for c in &ch { assert!(half::f16::from_bits(*c).to_f32().is_finite()); }
            // bit for bit, except that -0 rebuilds as +0 (sigmoid(-0) == sigmoid(+0) == 0.5)
            let want = if l == 0.0 { 0u32 } else { l.to_bits() };
            assert_eq!(tree_halving(&sgc, &xg, 1024).to_bits(), want, "halving tree, L = {l:e}");
            assert_eq!(tree_ep2(&sgc, &xg).to_bits(), want, "ep2 tree, L = {l:e}");
            checked += 1;
        }
        assert!(checked > 190_000);
        // |L| > 65504 clamps to +-65504 (sigmoid is exactly 0 or 1 there either way)
        let ch = sg_encode(65519.0);
        let mut xg = vec![0u16; h];
        xg[..6].copy_from_slice(&ch);
        assert_eq!(tree_halving(&sgc, &xg, 1024), 65504.0);
        let ch = sg_encode(1e9);
        let mut xg = vec![0u16; h];
        xg[..6].copy_from_slice(&ch);
        assert_eq!(tree_halving(&sgc, &xg, 1024), 65504.0);
        // below 2^-25 only bits under 2^-48 are dropped: the rebuilt value is within 2^-48
        let tiny = 1.2345e-9f32;
        let ch = sg_encode(tiny);
        let mut xg = vec![0u16; h];
        xg[..6].copy_from_slice(&ch);
        assert!((tree_halving(&sgc, &xg, 1024) - tiny).abs() <= 2f32.powi(-48));
    }

    #[test]
    fn bf_ks_is_a_shape_function_that_divides_k_into_32_groups() {
        for &(k, n) in &[(2560usize, 512usize), (2560, 10240), (2560, 2560), (10240, 320), (320, 10240), (2560, 248320),
                         (6144, 2560), (640, 2560), (2560, 640), (2560, 12288), (32, 64), (64, 2)] {
            let ks = bf_ks(k, n) as usize;
            let g = k / 32;
            assert!(ks >= 1 && g % ks == 0, "k {k} n {n} ks {ks}");
            assert!(ks == 1 || g / ks >= 2, "k {k} n {n}: a slab holds >= 2 groups");
            assert_eq!(bf_ks(k, n), bf_ks(k, n));
        }
        assert_eq!(bf_ks(2560, 512), 40);     // the router: 160 CTAs
        assert_eq!(bf_ks(2560, 10240), 4);    // the n-gram key projection: 320 CTAs
        assert_eq!(bf_ks(320, 10240), 5);     // the hc up projection (K = 320 is not a multiple of 128)
        assert_eq!(bf_ks(2560, 248320), 1);   // lm_head class
    }

    #[test]
    fn fx_scratch_sizes_cover_every_decode_stored_bf16_gemm() {
        // the Flash-Next geometry at the widest Scratch (16 rows): ws holds ks * 16 * n and the counters every N tile
        let (h, hcn, lr, ne, w) = (2560usize, 4usize, 320usize, 512usize, 16usize);
        let ws = |k: usize, n: usize| bf_ks(k, n) as usize * w * n;
        let rw = h * hcn;
        let need_r = ws(h, ne);
        let need_p = ws(2560, 10240).max(ws(2560, 2560));
        let need_h = ws(rw, lr).max(ws(lr, rw));
        // FxScratch::with_dims's own arithmetic (kept in step with it by this test)
        assert_eq!(need_r, 40 * 16 * 512);
        assert_eq!(need_p, 4 * 16 * 10240);
        assert!(need_h >= 5 * 16 * 10240 && need_h >= bf_ks(rw, lr) as usize * 16 * lr);
        assert!(10240usize.div_ceil(128) <= rw.max(lr).div_ceil(128));
    }

    #[test]
    fn e4m3_rne_round_trips_every_code_and_rounds_midpoints_to_even() {
        for c in 0u8..=0x7E {
            let v = crate::fp4l::pack::e4m3(c);
            assert_eq!(f32_to_e4m3_rne(v), c);
            assert_eq!(f32_to_e4m3_rne(-v) & 0x7F, c);
        }
        assert_eq!(f32_to_e4m3_rne(1000.0), 0x7E);
        // midpoint between 1.0 (0x38) and 1.125 (0x39) -> even 0x38
        assert_eq!(f32_to_e4m3_rne(1.0625), 0x38);
    }

    #[test]
    fn mx8_frag_is_the_kernels_index_map_and_a_permutation() {
        // n = 37 (two full tiles + a padded one), k = 192 (three 64-k blocks): every element and scale lands at the
        // offset the kernels' xf_mx8_f64_at computes, and nothing else is written
        let (n, k) = (37usize, 192usize);
        let codes: Vec<u8> = (0..n * k).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8 | 1).collect();
        let scales: Vec<u8> = (0..n * k / 32).map(|i| (i % 250) as u8).collect();
        let (w, sc) = mx8_frag(&codes, &scales, n, k);
        assert_eq!((w.len(), sc.len()), (48 * k, 48 * k / 32));
        let kb = k / 64;
        let mut hit = vec![false; w.len()];
        for row in 0..n {
            for kk in 0..k {
                let (t, rr) = (row / 16, row % 16);
                let (h, gid) = (rr >> 3, rr & 7);
                let (b, gs, tig, j) = (kk >> 6, (kk >> 5) & 1, (kk >> 3) & 3, kk & 7);
                let o = ((t * kb + b) * 2 + h) * 512 + (gid * 4 + tig) * 16 + gs * 8 + j;
                assert_eq!(w[o], codes[row * k + kk], "code ({row}, {kk})");
                hit[o] = true;
                let so = (t * kb + b) * 32 + (gid >> 1) * 8 + h * 4 + (gid & 1) * 2 + gs;
                assert_eq!(sc[so], scales[row * (k / 32) + kk / 32], "scale ({row}, {kk})");
            }
        }
        assert!(w.iter().zip(&hit).all(|(&v, &h)| h || v == 0), "padding rows are zero codes");
    }

    #[test]
    fn nv4_frag_is_the_kernels_index_map_and_a_permutation() {
        // n = 32 (two tiles), k = 384 (three 128-k blocks): a permutation of distinct-position markers lands every
        // nibble byte and scale where the xf_nv4_* kernels read it
        let (n, k) = (32usize, 384usize);
        let codes: Vec<u8> = (0..n * k / 2).map(|i| (i.wrapping_mul(2654435761) >> 9) as u8).collect();
        let scales: Vec<u8> = (0..n * k / 16).map(|i| (i * 7 % 251) as u8).collect();
        let (mut w, mut sc) = (vec![0u8; codes.len()], vec![0u8; scales.len()]);
        nv4_frag_into(&codes, &scales, n, k, &mut w, &mut sc);
        let mut ws = w.clone(); ws.sort_unstable();
        let mut cs = codes.clone(); cs.sort_unstable();
        assert_eq!(ws, cs, "codes are a permutation");
        let kb = k / 128;
        for row in 0..n {
            let (t, h, gid) = (row / 16, (row % 16) / 8, row % 8);
            for kk in (0..k).step_by(2) {
                // kernel view: block b, step s2 (64 k), lane tig owns k 16tig..16tig+15 of the step
                let (b, s2, tig, jb) = (kk / 128, (kk / 64) % 2, (kk % 64) / 16, (kk % 16) / 2);
                let o = ((t * kb + b) * 2 + h) * 512 + (gid * 4 + tig) * 16 + s2 * 8 + jb;
                assert_eq!(w[o], codes[row * k / 2 + kk / 2], "nibble byte ({row}, {kk})");
                let so = (t * kb + b) * 128 + (gid * 4 + tig) * 4 + 2 * h + s2;
                assert_eq!(sc[so], scales[row * k / 16 + kk / 16], "scale ({row}, {kk})");
            }
        }
    }

    #[test]
    fn bf16_frag_is_the_kernels_index_map_and_unfrag_inverts_it() {
        // n = 37 (padded to 48), k = 96 (three 32-k groups): every value lands where xf_bf16w_gemm / xf_b32_to_rm read it
        let (n, k) = (37usize, 96usize);
        let bits: Vec<u16> = (0..n * k).map(|i| (i as u32).wrapping_mul(2654435761).rotate_right(7) as u16 | 1).collect();
        let w = bf16_frag(&bits, n, k);
        assert_eq!(w.len(), 48 * k);
        let g = k / 32;
        for row in 0..n {
            for kk in 0..k {
                let (t, h, gid) = (row / 16, (row % 16) / 8, row % 8);
                let (gg, tig, j) = (kk / 32, (kk % 32) / 8, kk % 8);
                assert_eq!(w[((t * g + gg) * 2 + h) * 256 + (gid * 4 + tig) * 8 + j], bits[row * k + kk], "({row}, {kk})");
            }
        }
        assert_eq!(w.iter().filter(|&&v| v == 0).count(), 11 * k, "only the 11 padding rows are zero");
        assert_eq!(bf16_unfrag(&w, n, k), bits);
    }

    #[test]
    fn mx8_group_quant_is_within_half_an_ulp() {
        let v: Vec<f32> = (0..32).map(|i| ((i as f32) - 15.5) * 0.013).collect();
        let mut c = [0u8; 32];
        let s = mx8_quant_group(&v, &mut c);
        let scale = crate::fp4l::pack::ue8m0(s);
        for (i, &x) in v.iter().enumerate() {
            let d = crate::fp4l::pack::e4m3(c[i]) * scale;
            assert!((d - x).abs() <= x.abs() * 0.0625 + scale * 2f32.powi(-9), "{x} -> {d}");
        }
    }

    #[test]
    fn ple_row_decode_matches_the_pack_reference() {
        let mut raw = [0u8; 90];
        for (i, b) in raw.iter_mut().enumerate() { *b = if i < 80 { (i * 37 % 256) as u8 } else { 0x30 + (i as u8 % 8) }; }
        let gs = 3.3242e-5f32;
        let mut out = [0u16; 160];
        ple_fx_decode(&raw, gs, &mut out);
        let h = crate::fp4l::pack::Nv4Host { packed: raw[..80].to_vec(), scales: raw[80..].to_vec(), gs, n: 1, k: 160 };
        let mut r = vec![0f32; 160];
        h.row_f32(0, &mut r);
        for i in 0..160 { assert_eq!(out[i], half::f16::from_f32(r[i]).to_bits(), "dim {i}"); }
    }
}

// ================================================================================================
// --probe-fp4l-gpu: the FIRST GPU check of the FP4L kernels (PLAN/FP4L_GPU_RUNBOOK.md rung 1).
// Real pack tensors, deterministic activations, an exact host reference (f64 sums of the exactly
// dequantized weights times the f16 activations), every row count 1..=16:
//   FP4L_GPU_E2M1   cvt vs int E2M1 decode (every byte pair, both nibble orders) bitwise + vs the table
//   FP4L_GPU_MX8    xf_mx8_gemm: max rel err vs host per m; row 0 BITWISE equal at every m (batch invariance)
//   FP4L_GPU_F16W   xf_f16w_gemm (lm_head rows, MTP fc): same
//   FP4L_GPU_MOE    xf_nv4_moe (gate|up with per-projection gs, down): vs host, row 0 bitwise across m
//   FP4L_GPU_TILES  xf_nv4_tiles_m{1,2,4} (prefill): vs host on a synthetic tile list
//   FP4L_GPU_DEQ    xf_mx8_dequant / xf_nv4_dequant vs host bits
//   FP4L_GPU_TIME   per-kernel µs and achieved GB/s at m = 1 (cold-ish: 200 launches, rotating nothing)
// Exit status 1 on any FAIL line. No model load; ~3 GB of device memory.
// ================================================================================================

fn lcg_f16(seed: &mut u64, n: usize, scale: f32) -> Vec<u16> {
    (0..n).map(|_| {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((*seed >> 40) as f32) / ((1u64 << 24) as f32) - 0.5;
        half::f16::from_f32(u * 2.0 * scale).to_bits()
    }).collect()
}

/// host y[t][r] = sum_k w(r, k) * x[t][k] in f64 over exactly-dequantized weights.
fn host_gemm(wrow: &dyn Fn(usize, &mut [f32]), x: &[u16], m: usize, n: usize, k: usize) -> Vec<f64> {
    let xf: Vec<f32> = x.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
    let mut out = vec![0f64; m * n];
    let mut row = vec![0f32; k];
    for r in 0..n {
        wrow(r, &mut row);
        for t in 0..m {
            let xs = &xf[t * k..(t + 1) * k];
            let mut s = 0f64;
            for c in 0..k { s += row[c] as f64 * xs[c] as f64; }
            out[t * n + r] = s;
        }
    }
    out
}

fn rel_err(got: &[u16], want: &[f64]) -> (f64, f64) {
    let (mut d2, mut w2, mut mx) = (0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let g = half::f16::from_bits(*g).to_f32() as f64;
        d2 += (g - w) * (g - w); w2 += w * w;
        mx = mx.max((g - w).abs() / w.abs().max(1e-3));
    }
    ((d2 / w2.max(1e-300)).sqrt(), mx)
}

/// Probe only: rows [0, n) of a bf16 linear in the given representation (the lm_head lever's three ends).
fn probe_quad_bf(dev: &Arc<CudaDevice>, fp: &Fp4lPack, stem: &str, n: usize, mode: HeadPrec) -> Result<Quad> {
    let (n_full, k) = fp.shape2(&format!("{stem}.weight"))?;
    let n = n.min(n_full);
    let raw = fp.raw_range(&format!("{stem}.weight"), 0, (n * k * 2) as u64)?;
    let bits: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let fx = match mode {
        HeadPrec::Bf16 => FxW::Bf16 { w: dev.htod_sync_copy(&bf16_frag(&bits, n, k))? },
        HeadPrec::F16 => { let (f, osc) = bf16_bytes_to_f16(&raw, stem); FxW::F16 { w: dev.htod_sync_copy(&f)?, osc } }
        HeadPrec::Mx8 => {
            let vals: Vec<f32> = bits.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
            let (c, s) = mx8_rtn(&vals, n, k);
            mx8_dev(dev, &c, &s, n, k)?
        }
    };
    fx_quad_wrap(dev, fx, k, n, false)
}

/// The stored bf16 bits of `name` with every value below 2^-17 in magnitude set to +-0: such a tensor is
/// exactly representable in f16 too, so the trunk kernel (its f16 copy) and the twin (bf16) see the same values.
fn f16_safe_bf16(fp: &Fp4lPack, name: &str) -> Result<(Vec<u16>, Vec<u16>)> {
    let bits: Vec<u16> = fp.raw(name)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let bf: Vec<u16> = bits.iter().map(|&b| { let v = f32::from_bits((b as u32) << 16); if v.abs() < 2f32.powi(-17) { b & 0x8000 } else { b } }).collect();
    let f16: Vec<u16> = bf.iter().map(|&b| half::f16::from_f32(f32::from_bits((b as u32) << 16)).to_bits()).collect();
    for (a, b) in bf.iter().zip(&f16) {
        anyhow::ensure!(f32::from_bits((*a as u32) << 16) == half::f16::from_bits(*b).to_f32(), "{name}: a bf16 value is not f16-exact");
    }
    Ok((bf, f16))
}

/// An fp32 norm rounded to f16 and widened back: (f32 copy, f16 copy) of the same values.
fn f16_safe_norm(fp: &Fp4lPack, name: &str) -> Result<(Vec<f32>, Vec<u16>)> {
    let v = fp.plain_f32(name)?;
    let h: Vec<u16> = v.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
    Ok((h.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect(), h))
}

fn count_diff<T: PartialEq>(a: &[T], b: &[T]) -> usize { a.iter().zip(b).filter(|(x, y)| x != y).count() }

/// FP4L_GPU_TWIN: each precision twin against the trunk kernel it copies, bit for bit, on f16-exact inputs.
fn probe_twins(dev: &Arc<CudaDevice>, l: &Launcher, fp: &Fp4lPack, seed: &mut u64, fails: &mut usize) -> Result<()> {
    let (h, hcn, lr) = (2560usize, 4usize, 320usize);
    let rw = h * hcn;
    let mut line = |name: &str, bad: usize, of: usize| {
        if bad != 0 { *fails += 1; }
        println!("FP4L_GPU_TWIN {} {name}: {bad} bit mismatches vs the trunk kernel (of {of})", if bad == 0 { "OK" } else { "FAIL" });
    };
    let lay = "model.language_model.layers.0.attn_hyper_connection";
    let (nf32, n16) = f16_safe_norm(fp, &format!("{lay}.hc_norm.weight"))?;
    let (dn_bf, dn_16) = f16_safe_bf16(fp, &format!("{lay}.input_mix_weight_down.weight"))?;
    let (up_bf, up_16) = f16_safe_bf16(fp, &format!("{lay}.input_mix_weight_up.weight"))?;
    let (wi_bf, wi_16) = f16_safe_bf16(fp, &format!("{lay}.block_inject_weight.weight"))?;
    let (nf, nh) = (dev.htod_sync_copy(&nf32)?, dev.htod_sync_copy(&n16)?);
    let (dnb, dnh) = (dev.htod_sync_copy(&dn_bf)?, dev.htod_sync_copy(&dn_16)?);
    let (upb, uph) = (dev.htod_sync_copy(&up_bf)?, dev.htod_sync_copy(&up_16)?);
    let (wib, wih) = (dev.htod_sync_copy(&wi_bf)?, dev.htod_sync_copy(&wi_16)?);
    let bar = dev.htod_sync_copy(&[0i32; 2])?;
    let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
    // the served mixer's B32 copies of the same weights (xf_hc_fuse_b32)
    let (dnb32, upb32) = (dev.htod_sync_copy(&bf16_frag(&dn_bf, lr, rw))?, dev.htod_sync_copy(&bf16_frag(&up_bf, rw, lr))?);
    let hcf_cap = xf_co_residency("xf_hc_fuse_b32");
    let wsb = dev.alloc_zeros::<f32>(HCF_WS_PER_LR * lr)?;
    let mut hcf_fails = 0usize;
    let mut hcf_row0: Option<Vec<u16>> = None;
    for m in [1usize, 5, 16] {
        // f32 stream state + an f16 sublayer output + gates
        let resid0: Vec<f32> = lcg_f16(seed, m * rw, 4.0).iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
        let y: Vec<u16> = lcg_f16(seed, m * h, 2.0);
        let inj: Vec<f32> = (0..m * hcn).map(|i| 0.5 + 0.1 * i as f32).collect();
        let (yd, injd) = (dev.htod_sync_copy(&y)?, dev.htod_sync_copy(&inj)?);
        // norm (+ fused inject) : trunk vs twin, each on its own copy of the state
        for fused in [false, true] {
            for pf in [false, true] {
                let (mut ra, mut rb) = (dev.htod_sync_copy(&resid0)?, dev.htod_sync_copy(&resid0)?);
                let (mut ha, mut hb) = (dev.alloc_zeros::<u16>(m * rw)?, dev.alloc_zeros::<u16>(m * rw)?);
                let yp = if fused { dp(yd.device_ptr()) } else { 0 };
                if pf {
                    xqlaunch!(l, "xq_pf_hc_inj_norm", ((hcn * m) as u32, 1, 1), (128, 1, 1), 0,
                              (&mut ha, &mut ra, yp, &injd, &nh, h as i32, hcn as i32, m as i32, 1e-6f32))?;
                    pf_hc_inj_norm_f(l, dp(hb.device_ptr()), dp(rb.device_ptr()), yp, dp(injd.device_ptr()), dp(nf.device_ptr()), h, hcn, m, 1e-6)?;
                } else if fused {
                    xqlaunch!(l, "xq_hc_inj_norm", ((hcn * m) as u32, 1, 1), (1024, 1, 1), 0,
                              (&mut ha, &mut ra, yp, &injd, &nh, h as i32, hcn as i32, m as i32, 1e-6f32))?;
                    hc_inj_norm_f(l, dp(hb.device_ptr()), dp(rb.device_ptr()), yp, dp(injd.device_ptr()), &nf, h, hcn, m, 1e-6)?;
                } else {
                    xqlaunch!(l, "xq_hc_norm", ((hcn * m) as u32, 1, 1), (1024, 1, 1), 4096,
                              (&mut ha, &ra, &nh, h as i32, hcn as i32, m as i32, 1e-6f32))?;
                    hc_norm_f(l, dp(hb.device_ptr()), dp(rb.device_ptr()), &nf, h, hcn, m, 1e-6)?;
                }
                dev.synchronize()?;
                let (a, b) = (dev.dtoh_sync_copy(&ha)?, dev.dtoh_sync_copy(&hb)?);
                let (sa, sb) = (dev.dtoh_sync_copy(&ra)?, dev.dtoh_sync_copy(&rb)?);
                let bad = count_diff(&a, &b) + sa.iter().zip(&sb).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
                line(&format!("{} m={m}", match (pf, fused) { (true, _) => "xf_pf_hc_inj_norm_f", (false, true) => "xf_hc_inj_norm_f", _ => "xf_hc_norm_f" }),
                     bad, 2 * m * rw);
            }
        }
        // mixer (fused) + mix: trunk xq_hc_fuse + xq_hc_mix_rg vs the twins, same hn
        let hn: Vec<u16> = lcg_f16(seed, m * rw, 1.0);
        let hnd = dev.htod_sync_copy(&hn)?;
        let bx = 8u32;
        let (mut dda, mut uua) = (dev.alloc_zeros::<u16>(m * lr)?, dev.alloc_zeros::<u16>(m * rw)?);
        let (ddb, uub) = (dev.alloc_zeros::<u16>(m * lr)?, dev.alloc_zeros::<u16>(m * rw)?);
        xqlaunch!(l, "xq_hc_fuse", (bx, m as u32, 1), (256, 1, 1), 0,
                  (&mut dda, &mut uua, &hnd, &dnh, &uph, &bar, hcn as f32, lr as i32, rw as i32))?;
        hc_fuse_bf(l, dp(ddb.device_ptr()), dp(uub.device_ptr()), dp(hnd.device_ptr()), &dnb, &upb, dp(bar.device_ptr()), hcn as f32, lr, rw, bx, m)?;
        dev.synchronize()?;
        let bad = count_diff(&dev.dtoh_sync_copy(&dda)?, &dev.dtoh_sync_copy(&ddb)?) + count_diff(&dev.dtoh_sync_copy(&uua)?, &dev.dtoh_sync_copy(&uub)?);
        line(&format!("xf_hc_fuse_bf m={m}"), bad, m * (lr + rw));
        // the SERVED mixer xf_hc_fuse_b32 (tensor cores, B32): a different summation order than the scalar twin, so
        // rel-L2 vs the twin; bitwise across grid sizes (80 / 24 CTAs) and, on row 0, across m (batch invariance)
        {
            let g = hc_fuse_grid(hcf_cap * 5 / 6);
            anyhow::ensure!(g > 0, "xf_hc_fuse_b32 co-residency {hcf_cap} too small for the probe");
            let mut outs = Vec::new();
            for gg in [g, 24u32.min(g)] {
                let (ddc, uuc) = (dev.alloc_zeros::<u16>(m * lr)?, dev.alloc_zeros::<u16>(m * rw)?);
                hc_fuse_b32(l, dp(ddc.device_ptr()), dp(uuc.device_ptr()), dp(hnd.device_ptr()), &dnb32, &upb32, &wsb,
                            dp(bar.device_ptr()), hcn as f32, lr, rw, gg, m)?;
                dev.synchronize()?;
                outs.push(dev.dtoh_sync_copy(&uuc)?);
            }
            let twin: Vec<f64> = dev.dtoh_sync_copy(&uub)?.iter().map(|&b| half::f16::from_bits(b).to_f32() as f64).collect();
            let (rel, _) = rel_err(&outs[0], &twin);
            let gbad = count_diff(&outs[0], &outs[1]);
            // row 0 of every m sees a different hn (fresh per m): batch invariance is checked by re-running row 0 alone
            let (ddc, uuc) = (dev.alloc_zeros::<u16>(lr)?, dev.alloc_zeros::<u16>(rw)?);
            hc_fuse_b32(l, dp(ddc.device_ptr()), dp(uuc.device_ptr()), dp(hnd.device_ptr()), &dnb32, &upb32, &wsb,
                        dp(bar.device_ptr()), hcn as f32, lr, rw, g, 1)?;
            dev.synchronize()?;
            let bi = count_diff(&dev.dtoh_sync_copy(&uuc)?, &outs[0][..rw]);
            let ok = rel < 2e-3 && gbad == 0 && bi == 0;
            if !ok { hcf_fails += 1; }
            if m == 1 { hcf_row0 = Some(outs[0].clone()); }
            println!("FP4L_GPU_HCF {} xf_hc_fuse_b32 m={m} grid {g}: rel-L2 vs the scalar twin {rel:.2e}; grid {g} vs 24 bit-diffs {gbad}; row 0 alone vs in m={m} bit-diffs {bi}", if ok { "OK" } else { "FAIL" });
        }
        for pf in [false, true] {
            let (mut xa, xb) = (dev.alloc_zeros::<u16>(m * h)?, dev.alloc_zeros::<u16>(m * h)?);
            let (mut ia, ib) = (dev.alloc_zeros::<f32>(m * hcn)?, dev.alloc_zeros::<f32>(m * hcn)?);
            if pf {
                xqlaunch!(l, "xq_pf_hc_mix_2560x4", (m as u32, 1, 1), (128, 1, 1), 0, (&mut xa, &mut ia, &hnd, &uua, &wih, m as i32))?;
                pf_hc_mix_bf(l, dp(xb.device_ptr()), dp(ib.device_ptr()), dp(hnd.device_ptr()), dp(uua.device_ptr()), dp(wib.device_ptr()), m)?;
            } else {
                let gx = (hcn + h.div_ceil(1024)) as u32;
                xqlaunch!(l, "xq_hc_mix_rg", (gx, m as u32, 1), (1024, 1, 1), 0, (&mut xa, &mut ia, &hnd, &uua, &wih, h as i32, hcn as i32, m as i32))?;
                hc_mix_bf(l, dp(xb.device_ptr()), dp(ib.device_ptr()), dp(hnd.device_ptr()), dp(uua.device_ptr()), dp(wib.device_ptr()), h, hcn, m)?;
            }
            dev.synchronize()?;
            let (fa, fb) = (dev.dtoh_sync_copy(&ia)?, dev.dtoh_sync_copy(&ib)?);
            let bad = count_diff(&dev.dtoh_sync_copy(&xa)?, &dev.dtoh_sync_copy(&xb)?) + fa.iter().zip(&fb).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
            line(&format!("{} m={m}", if pf { "xf_pf_hc_mix_bf" } else { "xf_hc_mix_bf" }), bad, m * (h + hcn));
        }
    }
    // embedding gather: 16 token rows of the real table
    {
        let name = "model.language_model.embed_tokens.weight";
        let (n_full, _) = fp.shape2(name)?;
        let toks: Vec<i32> = (0..16).map(|i| ((i as u64 * 15_485_863) % n_full as u64) as i32).collect();
        let rows: Vec<u8> = toks.iter().flat_map(|&t| fp.raw_range(name, (t as usize * h * 2) as u64, ((t as usize + 1) * h * 2) as u64).unwrap_or_default()).collect();
        anyhow::ensure!(rows.len() == 16 * h * 2, "embedding rows read");
        let bf: Vec<u16> = rows.chunks_exact(2).map(|c| { let b = u16::from_le_bytes([c[0], c[1]]); if f32::from_bits((b as u32) << 16).abs() < 2f32.powi(-17) { b & 0x8000 } else { b } }).collect();
        let f16: Vec<u16> = bf.iter().map(|&b| half::f16::from_f32(f32::from_bits((b as u32) << 16)).to_bits()).collect();
        // a 16-row table (row i = token i's row) and tokens 0..16
        let (tb, th) = (dev.htod_sync_copy(&bf)?, dev.htod_sync_copy(&f16)?);
        let td = dev.htod_sync_copy(&(0..16).collect::<Vec<i32>>())?;
        let (mut ra, rb) = (dev.alloc_zeros::<f32>(16 * rw)?, dev.alloc_zeros::<f32>(16 * rw)?);
        xqlaunch!(l, "xq_embed_resid", (((rw * 16) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0, (&mut ra, &th, &td, h as i32, hcn as i32, 16i32))?;
        embed_resid_bf(l, dp(rb.device_ptr()), &tb, dp(td.device_ptr()), h, hcn, 16)?;
        dev.synchronize()?;
        let (a, b) = (dev.dtoh_sync_copy(&ra)?, dev.dtoh_sync_copy(&rb)?);
        line("xf_embed_resid_bf m=16", a.iter().zip(&b).filter(|(p, q)| p.to_bits() != q.to_bits()).count(), 16 * rw);
    }
    // n-gram norms (fp32) and conv (fp32 taps) — the trunk's layer-1 PLE tensors
    {
        let pk = "model.language_model.layers.1.ple";
        let m = 5usize;
        for (which, nm) in [(0u8, "norm_key"), (1, "norm_query"), (2, "norm_conv")] {
            let (wf, wh) = f16_safe_norm(fp, &format!("{pk}.{nm}.weight"))?;
            let (wfd, whd) = (dev.htod_sync_copy(&wf)?, dev.htod_sync_copy(&wh)?);
            let x16 = dev.htod_sync_copy(&lcg_f16(seed, m * 4 * 2560, 2.0))?;
            let x32: Vec<f32> = lcg_f16(seed, m * 4 * 2560, 2.0).iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
            let x32d = dev.htod_sync_copy(&x32)?;
            let bad = match which {
                0 | 1 => {
                    let (mut oa, ob) = (dev.alloc_zeros::<f32>(m * 4 * 2560)?, dev.alloc_zeros::<f32>(m * 4 * 2560)?);
                    if which == 0 {
                        xqlaunch!(l, "xq_ple_norm_f16", ((m * 4) as u32, 1, 1), (256, 1, 1), 0, (&x16, &whd, &mut oa, 1e-6f32))?;
                        ple_norm_f(l, 0, dp(x16.device_ptr()), &wfd, dp(ob.device_ptr()), m, 1e-6)?;
                    } else {
                        xqlaunch!(l, "xq_ple_norm_f32", ((m * 4) as u32, 1, 1), (256, 1, 1), 0, (&x32d, &whd, &mut oa, 1e-6f32))?;
                        ple_norm_f(l, 1, dp(x32d.device_ptr()), &wfd, dp(ob.device_ptr()), m, 1e-6)?;
                    }
                    dev.synchronize()?;
                    let (a, b) = (dev.dtoh_sync_copy(&oa)?, dev.dtoh_sync_copy(&ob)?);
                    a.iter().zip(&b).filter(|(p, q)| p.to_bits() != q.to_bits()).count()
                }
                _ => {
                    let (mut oa, ob) = (dev.alloc_zeros::<u16>(m * 4 * 2560)?, dev.alloc_zeros::<u16>(m * 4 * 2560)?);
                    xqlaunch!(l, "xq_ple_norm_hh", ((m * 4) as u32, 1, 1), (256, 1, 1), 0, (&x16, &whd, &mut oa, 1e-6f32))?;
                    ple_norm_f(l, 2, dp(x16.device_ptr()), &wfd, dp(ob.device_ptr()), m, 1e-6)?;
                    dev.synchronize()?;
                    count_diff(&dev.dtoh_sync_copy(&oa)?, &dev.dtoh_sync_copy(&ob)?)
                }
            };
            line(&format!("xf_ple_{nm}_f m={m}"), bad, m * 4 * 2560);
        }
        // conv: decode step and chunk (f16-safe taps as f16 / as f32)
        let (cb, c16) = f16_safe_bf16(fp, &format!("{pk}.conv1d.weight"))?;
        let cf: Vec<f32> = cb.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
        let (cfd, chd) = (dev.htod_sync_copy(&cf)?, dev.htod_sync_copy(&c16)?);
        let gated = dev.htod_sync_copy(&lcg_f16(seed, m * 10240, 1.0))?;
        let normed = dev.htod_sync_copy(&lcg_f16(seed, m * 10240, 1.0))?;
        let state0 = lcg_f16(seed, 2 * 10240 * 9, 1.0);
        let slots = dev.htod_sync_copy(&(0..m).flat_map(|q| [(q % 2) as i32, 0]).collect::<Vec<i32>>())?;
        let r0: Vec<f32> = lcg_f16(seed, m * 10240, 3.0).iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
        for chunk in [false, true] {
            let (mut ra, rb) = (dev.htod_sync_copy(&r0)?, dev.htod_sync_copy(&r0)?);
            let (mut sa, sb) = (dev.htod_sync_copy(&state0)?, dev.htod_sync_copy(&state0)?);
            if chunk {
                xqlaunch!(l, "xq_ple_conv_chunk", ((10240u32 + 255) / 256, 1, 1), (256, 1, 1), 0, (&mut ra, &gated, &normed, &mut sa, &chd, 1i32, m as i32))?;
                ple_conv_chunk_f(l, dp(rb.device_ptr()), dp(gated.device_ptr()), dp(normed.device_ptr()), dp(sb.device_ptr()), &cfd, 1, m)?;
            } else {
                xqlaunch!(l, "xq_ple_conv", (((m * 10240) as u32 + 255) / 256, 1, 1), (256, 1, 1), 0, (&mut ra, &gated, &normed, &sa, &chd, m as i32, &slots))?;
                ple_conv_f(l, dp(rb.device_ptr()), dp(gated.device_ptr()), dp(normed.device_ptr()), dp(sb.device_ptr()), &cfd, m, dp(slots.device_ptr()))?;
            }
            dev.synchronize()?;
            let (a, b) = (dev.dtoh_sync_copy(&ra)?, dev.dtoh_sync_copy(&rb)?);
            let bad = a.iter().zip(&b).filter(|(p, q)| p.to_bits() != q.to_bits()).count()
                + count_diff(&dev.dtoh_sync_copy(&sa)?, &dev.dtoh_sync_copy(&sb)?);
            line(&format!("{} m={m}", if chunk { "xf_ple_conv_chunk_f" } else { "xf_ple_conv_f" }), bad, m * 10240);
        }
    }
    let _ = hcf_row0;
    *fails += hcf_fails;
    Ok(())
}

/// FP4L_GPU_SG: xf_sg_gate's channels through the trunk's combine kernels (decode xq_moe_combine, prefill
/// xq_moe_combine_rows, the TP vectorised tree xq_moe_combine_rows_ep2_k10) against the same combines fed an
/// f16 gate whose single product IS the logit (sg = s * e0 with s a power of two, so L = s * x0 is exact in
/// both paths): the outputs must agree bit for bit (the encoding + every tree rebuild L exactly).
fn probe_sg_gate(dev: &Arc<CudaDevice>, l: &Launcher, seed: &mut u64, fails: &mut usize) -> Result<()> {
    let h = 2560usize;
    let m = 16usize;
    let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
    let sgc = dev.htod_sync_copy(&sg_channel_constants(h))?;
    let mut x: Vec<u16> = lcg_f16(seed, m * h, 30.0);
    // row 0: zero logit; row 1: a huge one (clamped path); row 2: tiny
    x[0] = 0; x[h] = half::f16::from_f32(65504.0).to_bits(); x[2 * h] = half::f16::from_f32(3e-7).to_bits();
    let xd = dev.htod_sync_copy(&x)?;
    let ysh = dev.htod_sync_copy(&vec![half::f16::from_f32(1.0).to_bits(); m * h])?;
    let ids = dev.htod_sync_copy(&vec![0i32; m])?;            // k = 1, expert 0 ...
    let slotmap = dev.htod_sync_copy(&vec![-1i32; 512])?;     // ... absent: only the shared term
    let wts = dev.htod_sync_copy(&vec![0f32; m])?;
    let cand = dev.htod_sync_copy(&vec![-1i32; m])?;
    let yd = dev.alloc_zeros::<u16>(m * h)?;
    let mut total_bad = 0usize;
    for s_exp in [0i32, -6, -12, -20, 1] {
        let s = 2f32.powi(s_exp);
        let mut sg_bf = vec![0u16; h];
        sg_bf[0] = (s.to_bits() >> 16) as u16;
        let mut sg16 = vec![0u16; h];
        sg16[0] = half::f16::from_f32(s).to_bits();
        let (sgb, sgh) = (dev.htod_sync_copy(&sg_bf)?, dev.htod_sync_copy(&sg16)?);
        let xg = dev.htod_sync_copy(&vec![0u16; m * h])?;
        sg_gate(l, dp(xg.device_ptr()), &sgb, dp(xd.device_ptr()), h, m)?;
        let run = |name: &str, sgp: u64, xp: u64| -> Result<Vec<u32>> {
            match name {
                "xq_moe_combine" => {
                    let mut o = dev.alloc_zeros::<u16>(m * h)?;
                    xqlaunch!(l, "xq_moe_combine", (m as u32, 1, 1), (1024, 1, 1), 4096, (&mut o, &yd, &ysh, &ids, &wts, &slotmap, sgp, xp, 1i32, h as i32, m as i32))?;
                    dev.synchronize()?;
                    Ok(dev.dtoh_sync_copy(&o)?.iter().map(|&v| v as u32).collect())
                }
                "xq_moe_combine_rows" => {
                    let mut o = dev.alloc_zeros::<u16>(m * h)?;
                    xqlaunch!(l, "xq_moe_combine_rows", (m as u32, 1, 1), (1024, 1, 1), 4096, (&mut o, &yd, &ysh, &cand, &wts, sgp, xp, 1i32, h as i32, m as i32))?;
                    dev.synchronize()?;
                    Ok(dev.dtoh_sync_copy(&o)?.iter().map(|&v| v as u32).collect())
                }
                _ => {
                    let mut o = dev.alloc_zeros::<f32>(m * h)?;
                    xqlaunch!(l, "xq_moe_combine_rows_ep2_k10", (m as u32, 1, 1), (320, 1, 1), 0, (&mut o, &yd, &ysh, &cand, &wts, sgp, xp, 1i32, h as i32, m as i32, 1i32))?;
                    dev.synchronize()?;
                    Ok(dev.dtoh_sync_copy(&o)?.iter().map(|v| v.to_bits()).collect())
                }
            }
        };
        for name in ["xq_moe_combine", "xq_moe_combine_rows", "xq_moe_combine_rows_ep2_k10"] {
            let a = run(name, dp(sgh.device_ptr()), dp(xd.device_ptr()))?;
            let b = run(name, dp(sgc.device_ptr()), dp(xg.device_ptr()))?;
            total_bad += count_diff(&a, &b);
        }
        // the channels themselves vs the host model of the encoding (L = s * x0 exactly)
        let got = dev.dtoh_sync_copy(&xg)?;
        for r in 0..m {
            let want = sg_encode(s * half::f16::from_bits(x[r * h]).to_f32());
            total_bad += count_diff(&got[r * h..r * h + 6], &want);
            total_bad += got[r * h + 6..(r + 1) * h].iter().filter(|&&v| v != 0).count();
        }
    }
    if total_bad != 0 { *fails += 1; }
    println!("FP4L_GPU_SG {} shared-gate channels: {total_bad} mismatches over 5 gate scales x 16 rows x 3 combine kernels + the \
              encoding vs its host model (0 = every combine rebuilds the logit bit for bit)", if total_bad == 0 { "OK" } else { "FAIL" });
    Ok(())
}

pub fn probe_gpu(dir: &str) -> Result<()> {
    let fp = Fp4lPack::open(std::path::Path::new(dir))?;
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    crate::exl3_bench::load_module_pub(&dev)?;
    load_module(&dev)?;
    let s = crate::gpu::fork_blocking_stream(&dev);
    let l = Launcher { dev: &dev, stream: &s };
    let mut fails = 0usize;
    let mut seed = 0x5EED_F4F4u64;

    // ---- E2M1: every 16-bit word pattern through both decoders (8 nibbles per u32 word)
    {
        let n = 1usize << 16;
        let words: Vec<u32> = (0..n as u32).map(|v| v | (v.rotate_left(7) << 16)).collect();
        let w = dev.htod_sync_copy(&words)?;
        let mut oc = dev.alloc_zeros::<u32>(4 * n)?;
        let mut oi = dev.alloc_zeros::<u32>(4 * n)?;
        xflaunch!(l, "xf_e2m1_selftest", ((n as u32).div_ceil(256), 1, 1), (256, 1, 1), 0, (&w, &mut oc, &mut oi, n as i32))?;
        dev.synchronize()?;
        let (hc, hi) = (dev.dtoh_sync_copy(&oc)?, dev.dtoh_sync_copy(&oi)?);
        let mut bad_ci = 0usize;
        let mut bad_tab = 0usize;
        for i in 0..n {
            for j in 0..4 {
                if hc[4 * i + j] != hi[4 * i + j] { bad_ci += 1; }
                let byte = (words[i] >> (8 * j)) & 0xFF;
                let want_lo = half::f16::from_f32(crate::fp4l::pack::E2M1[(byte & 0xF) as usize]).to_bits() as u32;
                let want_hi = half::f16::from_f32(crate::fp4l::pack::E2M1[(byte >> 4) as usize]).to_bits() as u32;
                if hc[4 * i + j] != (want_lo | (want_hi << 16)) { bad_tab += 1; }
            }
        }
        let ok = bad_ci == 0 && bad_tab == 0;
        if !ok { fails += 1; }
        println!("FP4L_GPU_E2M1 {} cvt-vs-int mismatches {bad_ci}, cvt-vs-table mismatches {bad_tab} (of {})",
                 if ok { "OK" } else { "FAIL" }, 4 * n);
    }

    // ---- dense MXFP8 / f16 GEMMs on real tensors
    let ws = dev.alloc_zeros::<f32>(2_000_000)?; // == Scratch::chain_ws
    // `mode` (bf16 linears only): build the stated representation (the lm_head lever's three ends) instead of the
    // loader's default; Mx8 is lossy by design, so its line reports the error without a pass bound.
    let mut dense = |stem: &str, label: &str, rows_cap: Option<usize>, mode: Option<HeadPrec>| -> Result<()> {
        let kind = fp.lin_kind(stem)?;
        let (n_full, kk) = fp.shape2(&format!("{stem}.weight"))?;
        let n = rows_cap.map_or(n_full, |c| c.min(n_full));
        let q = match mode { Some(md) => probe_quad_bf(&dev, &fp, stem, n, md)?, None => fx_quad(&dev, &fp, stem, None, Some(&[(0, n)]), false)? };
        let lossy = mode == Some(HeadPrec::Mx8);
        let k = q.k as usize;
        let mx = if kind == LinKind::Mxfp8 { Some(fp.mxfp8_rows(stem, 0, n)?) } else { None };
        let bf = if kind == LinKind::Bf16 { Some(fp.plain_f32_range(&format!("{stem}.weight"), 0, n * kk)?) } else { None };
        let wrow = |r: usize, out: &mut [f32]| {
            if let Some(m) = mx.as_ref() { m.row_f32(r, out); }
            else if let Some(b) = bf.as_ref() {
                out[..k].copy_from_slice(&b[r * k..(r + 1) * k]); // the STORED bf16 values: the f16 copy must be exact
            }
        };
        let x_host = lcg_f16(&mut seed, 16 * k, 1.0);
        let want = host_gemm(&wrow, &x_host, 16, n, k);
        let x = dev.htod_sync_copy(&x_host)?;
        let mut row0: Option<Vec<u16>> = None;
        let (mut worst_rel, mut worst_mx, mut bitdiff) = (0f64, 0f64, 0usize);
        for m in 1..=16usize {
            let mut y = dev.alloc_zeros::<u16>(m * n)?;
            fx_chain(&l, &q, q.fx.as_deref().unwrap(), &x, &mut y, m, &ws, None)?;
            dev.synchronize()?;
            let got = dev.dtoh_sync_copy(&y)?;
            let (r, mxe) = rel_err(&got, &want[..m * n]);
            worst_rel = worst_rel.max(r); worst_mx = worst_mx.max(mxe);
            match &row0 { None => row0 = Some(got[..n].to_vec()), Some(r0) => bitdiff += r0.iter().zip(&got[..n]).filter(|(a, b)| a != b).count() }
        }
        // prefill row counts: the exact wide kernel (MXFP8) / cuBLASLt with alpha (f16)
        let (mut wide_rel, wide_ms) = (0f64, [17usize, 48, 100]);
        let mut tc_vs_wide: Option<usize> = None;
        {
            let xw_host = lcg_f16(&mut seed, 100 * k, 1.0);
            let want_w = host_gemm(&wrow, &xw_host, 100, n, k);
            let xw = dev.htod_sync_copy(&xw_host)?;
            let rws = dev.alloc_zeros::<u16>(n * k)?;
            let ltws = dev.alloc_zeros::<u8>(32 << 20)?;
            let rc = dev.alloc_zeros::<u8>(256)?;
            let rb = ReconBufs { ws: *rws.device_ptr() as u64, ws_len: rws.len(), rc: *rc.device_ptr() as u64,
                                 ltws: *ltws.device_ptr() as u64, ltws_len: ltws.len() };
            for &m in &wide_ms {
                let mut y = dev.alloc_zeros::<u16>(m * n)?;
                fx_chain(&l, &q, q.fx.as_deref().unwrap(), &xw, &mut y, m, &ws, Some(&rb))?;
                dev.synchronize()?;
                let got = dev.dtoh_sync_copy(&y)?;
                wide_rel = wide_rel.max(rel_err(&got, &want_w[..m * n]).0);
                // MXFP8: the served tiled kernel (xf_mx8_tc) must equal the simple wide kernel bit for bit
                if let (Some(FxW::Mx8 { w, s }), true) = (q.fx.as_deref(), !prefill_dense_lt()) {
                    let mut yw = dev.alloc_zeros::<u16>(m * n)?;
                    xflaunch!(l, "xf_mx8_wide", (n.div_ceil(128) as u32, m.div_ceil(64) as u32, 1), (256, 1, 1), 0,
                              (w, s, &xw, &mut yw, 0u64, m as i32, k as i32, n as i32))?;
                    dev.synchronize()?;
                    *tc_vs_wide.get_or_insert(0) += count_diff(&got, &dev.dtoh_sync_copy(&yw)?);
                }
            }
        }
        // timing at m = 1
        let mut y = dev.alloc_zeros::<u16>(n)?;
        dev.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..200 { fx_chain(&l, &q, q.fx.as_deref().unwrap(), &x, &mut y, 1, &ws, None)?; }
        dev.synchronize()?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / 200.0;
        let bytes = q.fx.as_deref().unwrap().bytes() as f64;
        let ok = lossy || (worst_rel < 2e-3 && bitdiff == 0 && wide_rel < 2e-3 && tc_vs_wide.unwrap_or(0) == 0);
        if !ok { fails += 1; }
        println!("{label} {} {stem} as {} [{n}x{k}] ks {}: rel-L2 vs host (stored values) <= {worst_rel:.2e} (max elem {worst_mx:.2e}); \
                  row 0 bit-diffs across m=1..16: {bitdiff}; prefill m {wide_ms:?} rel-L2 <= {wide_rel:.2e}{}; m=1 {us:.1} us = {:.0} GB/s{}",
                 if ok { "OK" } else { "FAIL" }, q.fx.as_deref().unwrap().kind(), q.ks,
                 tc_vs_wide.map_or(String::new(), |d| format!(" (xf_mx8_tc vs xf_mx8_wide bit-diffs {d})")), bytes / us / 1e3,
                 if lossy { " (lossy by design: no pass bound)" } else { "" });
        Ok(())
    };
    dense("model.language_model.layers.0.linear_attn.in_proj_qkv", "FP4L_GPU_MX8", None, None)?;
    dense("model.language_model.layers.0.linear_attn.in_proj_z", "FP4L_GPU_MX8", None, None)?;
    dense("model.language_model.layers.0.linear_attn.out_proj", "FP4L_GPU_MX8", None, None)?;
    dense("model.language_model.layers.3.self_attn.q_proj", "FP4L_GPU_MX8", None, None)?;
    dense("model.language_model.layers.3.self_attn.k_proj", "FP4L_GPU_MX8", None, None)?;
    dense("model.language_model.layers.3.self_attn.o_proj", "FP4L_GPU_MX8", None, None)?;
    dense("model.language_model.layers.0.mlp.shared_expert.down_proj", "FP4L_GPU_MX8", None, None)?;
    // the lm_head lever's three ends on the first 16384 rows (bf16 = stored, f16 = the shift, mxfp8 = RTN), MTP linears
    dense("lm_head", "FP4L_GPU_HEAD", Some(16384), Some(HeadPrec::Bf16))?;
    dense("lm_head", "FP4L_GPU_HEAD", Some(16384), Some(HeadPrec::F16))?;
    dense("lm_head", "FP4L_GPU_HEAD", Some(16384), Some(HeadPrec::Mx8))?;
    dense("mtp.fc_hidden", "FP4L_GPU_HEAD", None, Some(HeadPrec::Bf16))?;
    dense("mtp.fc_hidden", "FP4L_GPU_HEAD", None, Some(HeadPrec::F16))?;

    // ---- paired MXFP8 GEMMs (xf_mx8_gemm_x2): each member bitwise its own xf_mx8_gemm, m = 1..16
    for (p0, p1) in [("model.language_model.layers.0.linear_attn.in_proj_a", "model.language_model.layers.0.linear_attn.in_proj_b"),
                     ("model.language_model.layers.0.mlp.shared_expert.gate_proj", "model.language_model.layers.0.mlp.shared_expert.up_proj"),
                     ("model.language_model.layers.3.self_attn.k_proj", "model.language_model.layers.3.self_attn.v_proj")] {
        let q0 = fx_quad(&dev, &fp, p0, None, None, false)?;
        let q1 = fx_quad(&dev, &fp, p1, None, None, false)?;
        let (k, n) = (q0.k as usize, q0.n as usize);
        let x = dev.htod_sync_copy(&lcg_f16(&mut seed, 16 * k, 1.0))?;
        let half = 16 * q0.ks.max(1) as usize * n;
        let pws = dev.alloc_zeros::<f32>(2 * half)?;
        let (mut bad, mut taken) = (0usize, 0usize);
        for m in 1..=16usize {
            let (mut a0, mut a1) = (dev.alloc_zeros::<u16>(m * n)?, dev.alloc_zeros::<u16>(m * q1.n as usize)?);
            let (mut b0, mut b1) = (dev.alloc_zeros::<u16>(m * n)?, dev.alloc_zeros::<u16>(m * q1.n as usize)?);
            fx_chain(&l, &q0, q0.fx.as_deref().unwrap(), &x, &mut a0, m, &ws, None)?;
            fx_chain(&l, &q1, q1.fx.as_deref().unwrap(), &x, &mut a1, m, &ws, None)?;
            let wsp = *pws.device_ptr() as u64;
            if fx_pair(&l, &q0, &q1, *x.device_ptr() as u64, *b0.device_ptr() as u64, *b1.device_ptr() as u64,
                       (wsp, half), (wsp + (half * 4) as u64, half), m)? { taken += 1; }
            dev.synchronize()?;
            bad += count_diff(&dev.dtoh_sync_copy(&a0)?, &dev.dtoh_sync_copy(&b0)?);
            bad += count_diff(&dev.dtoh_sync_copy(&a1)?, &dev.dtoh_sync_copy(&b1)?);
        }
        let ok = bad == 0 && taken == 16;
        if !ok { fails += 1; }
        println!("FP4L_GPU_PAIR {} {p0} | {p1} [{n}x{k}] ks {}: xf_mx8_gemm_x2 taken at {taken}/16 widths; bit-diffs vs two \
                  xf_mx8_gemm launches over m = 1..16: {bad}", if ok { "OK" } else { "FAIL" }, q0.ks);
    }

    // ---- the stored-bf16 plain GEMMs (router fp32, n-gram projections, hc mixers) vs the host; batch invariance
    {
        let fxs = FxScratch::with_dims(&dev, 2560, 4, 320, 512, 16)?;   // the Flash-Next geometry
        let mut plain = |name: &str, k: usize, n: usize, f32out: bool| -> Result<()> {
            let bits: Vec<u16> = fp.raw(name)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            anyhow::ensure!(bits.len() == n * k, "{name}: {} values != {n} x {k}", bits.len());
            let w = dev.htod_sync_copy(&bf16_frag(&bits, n, k))?;
            let vals: Vec<f32> = bits.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
            let wrow = |r: usize, o: &mut [f32]| o[..k].copy_from_slice(&vals[r * k..(r + 1) * k]);
            let x_host = lcg_f16(&mut seed, 100 * k, 1.0);
            let want = host_gemm(&wrow, &x_host, 100, n, k);
            let x = dev.htod_sync_copy(&x_host)?;
            let (ws, cnt) = if n == 512 { (&fxs.ws_r, &fxs.cnt_r) } else if k == 2560 { (&fxs.ws_p, &fxs.cnt_p) } else { (&fxs.ws_h, &fxs.cnt_h) };
            let (mut worst, mut bitdiff, mut tcdiff) = (0f64, 0usize, 0usize);
            let mut row0: Option<Vec<u32>> = None;
            for m in (1..=16usize).chain([17, 48, 100]) {
                let (y16, yf) = (dev.alloc_zeros::<u16>(m * n)?, dev.alloc_zeros::<f32>(m * n)?);
                let (yp, yfp) = if f32out { (0u64, *yf.device_ptr() as u64) } else { (*y16.device_ptr() as u64, 0u64) };
                bf_gemm(&l, &w, *x.device_ptr() as u64, yp, yfp, m, k, n, ws, cnt)?;
                dev.synchronize()?;
                let got: Vec<f32> = if f32out { dev.dtoh_sync_copy(&yf)? }
                                    else { dev.dtoh_sync_copy(&y16)?.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect() };
                let (mut d2, mut w2) = (0f64, 0f64);
                for (g, w) in got.iter().zip(&want[..m * n]) { d2 += (*g as f64 - w).powi(2); w2 += w * w; }
                worst = worst.max((d2 / w2.max(1e-300)).sqrt());
                if m <= 16 {
                    let r0: Vec<u32> = got[..n].iter().map(|v| v.to_bits()).collect();
                    match &row0 { None => row0 = Some(r0), Some(a) => bitdiff += a.iter().zip(&r0).filter(|(p, q)| p != q).count() }
                } else {
                    // m > 16: the served tiled xf_bf16_tc must equal the simple xf_bf16_wide bit for bit
                    let (y16b, yfb) = (dev.alloc_zeros::<u16>(m * n)?, dev.alloc_zeros::<f32>(m * n)?);
                    let (yp2, yfp2) = if f32out { (0u64, *yfb.device_ptr() as u64) } else { (*y16b.device_ptr() as u64, 0u64) };
                    xflaunch!(l, "xf_bf16_wide", (n.div_ceil(128) as u32, m.div_ceil(64) as u32, 1), (256, 1, 1), 0,
                              (&w, *x.device_ptr() as u64, yp2, yfp2, m as i32, k as i32, n as i32))?;
                    dev.synchronize()?;
                    let other: Vec<f32> = if f32out { dev.dtoh_sync_copy(&yfb)? }
                                          else { dev.dtoh_sync_copy(&y16b)?.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect() };
                    tcdiff += got.iter().zip(&other).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
                }
            }
            // a pure-f32 output keeps fp32 accumulation error only (~1e-6); f16 output adds its own rounding (~3e-4)
            let bound = if f32out { 1e-5 } else { 2e-3 };
            let ok = worst < bound && bitdiff == 0 && tcdiff == 0;
            if !ok { fails += 1; }
            println!("FP4L_GPU_BF16 {} {name} [{n}x{k}] ks {} ({} out): rel-L2 vs host (stored bf16) <= {worst:.2e} at m = 1..16, 17, 48, 100; \
                      row 0 bit-diffs across m = 1..16: {bitdiff}; xf_bf16_tc vs xf_bf16_wide bit-diffs {tcdiff}",
                     if ok { "OK" } else { "FAIL" }, bf_ks(k, n), if f32out { "fp32" } else { "f16" });
            Ok(())
        };
        plain("model.language_model.layers.0.mlp.gate.weight", 2560, 512, true)?;
        plain("model.language_model.layers.1.ple.key_proj.weight", 2560, 10240, false)?;
        plain("model.language_model.layers.1.ple.value_proj.weight", 2560, 2560, false)?;
        plain("model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight", 10240, 320, false)?;
        plain("model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight", 320, 10240, false)?;
    }

    // ---- the TWINS: on weights exactly representable in f16 (the stored values with |v| < 2^-17 set to 0, and
    // the fp32 norms rounded to f16 and widened back) every twin must return the trunk kernel's bits
    probe_twins(&dev, &l, &fp, &mut seed, &mut fails)?;
    // ---- the shared-gate channels through the trunk's own combine kernels (bitwise vs an f16-expressible gate)
    probe_sg_gate(&dev, &l, &mut seed, &mut fails)?;

    // ---- NVFP4 experts (decode / verify kernel)
    {
        let (h, mi) = (2560usize, 640usize);
        let local: Vec<usize> = (0..512).collect();
        let fx = fx_experts(&dev, &fp, "model.language_model.layers.0.mlp", &local, h, mi)?;
        let picks: Vec<usize> = vec![0, 7, 255, 511, 100, 300, 42, 9, 77, 400];
        let hosts: Vec<(crate::fp4l::pack::Nv4Host, crate::fp4l::pack::Nv4Host, crate::fp4l::pack::Nv4Host)> = picks.iter().map(|&e| -> Result<_> {
            let p = format!("model.language_model.layers.0.mlp.experts.{e}");
            Ok((fp.nvfp4(&format!("{p}.gate_proj"))?, fp.nvfp4(&format!("{p}.up_proj"))?, fp.nvfp4(&format!("{p}.down_proj"))?))
        }).collect::<Result<_>>()?;
        let x_host = lcg_f16(&mut seed, 16 * h, 1.0);
        let din_host = lcg_f16(&mut seed, picks.len() * 16 * mi, 0.5);
        let x = dev.htod_sync_copy(&x_host)?;
        let idx: Vec<i32> = picks.iter().map(|&e| e as i32).collect();
        let idxmap = dev.htod_sync_copy(&idx)?;
        let esel = dev.htod_sync_copy(&[picks.len() as i32])?;
        let f = nv4_moe_fn();
        let (mut worst, mut bitdiff) = (0f64, 0usize);
        let mut row0: Option<Vec<u16>> = None;
        for m in 1..=16usize {
            let mut y = dev.alloc_zeros::<u16>(picks.len() * m * 2 * mi)?;
            xflaunch!(l, f, (((2 * mi) / 64) as u32, picks.len() as u32, 1), (128, 1, 1), 0,
                      (&fx.gu_w, &fx.gu_s, &fx.gs, &x, 0i64, &mut y, &idxmap, &esel, m as i32, h as i32, (2 * mi) as i32, mi as i32))?;
            // down on a fixed din [slot][m][mi] (rows of the 16-row host tensor)
            let din_m: Vec<u16> = (0..picks.len()).flat_map(|s| din_host[s * 16 * mi..s * 16 * mi + m * mi].to_vec()).collect();
            let din = dev.htod_sync_copy(&din_m)?;
            let mut yd = dev.alloc_zeros::<u16>(picks.len() * m * h)?;
            xflaunch!(l, f, ((h / 64) as u32, picks.len() as u32, 1), (128, 1, 1), 0,
                      (&fx.dn_w, &fx.dn_s, &fx.gs, &din, (m * mi) as i64, &mut yd, &idxmap, &esel, m as i32, mi as i32, h as i32, h as i32))?;
            dev.synchronize()?;
            let (gy, gd) = (dev.dtoh_sync_copy(&y)?, dev.dtoh_sync_copy(&yd)?);
            for (si, (g, u, d)) in hosts.iter().enumerate() {
                let wg = |r: usize, o: &mut [f32]| if r < mi { g.row_f32(r, o) } else { u.row_f32(r - mi, o) };
                let want = host_gemm(&wg, &x_host[..m * h], m, 2 * mi, h);
                let (r1, _) = rel_err(&gy[si * m * 2 * mi..(si + 1) * m * 2 * mi], &want);
                let wd = |r: usize, o: &mut [f32]| d.row_f32(r, o);
                let want_d = host_gemm(&wd, &din_m[si * m * mi..(si + 1) * m * mi], m, h, mi);
                let (r2, _) = rel_err(&gd[si * m * h..(si + 1) * m * h], &want_d);
                worst = worst.max(r1).max(r2);
            }
            match &row0 { None => row0 = Some(gy[..2 * mi].to_vec()), Some(r0) => bitdiff += r0.iter().zip(&gy[..2 * mi]).filter(|(a, b)| a != b).count() }
        }
        let ok = worst < 2e-3 && bitdiff == 0;
        if !ok { fails += 1; }
        println!("FP4L_GPU_MOE {} layer 0, {} expert slots, gate|up + down: rel-L2 vs host <= {worst:.2e}; slot 0 row 0 bit-diffs across m: {bitdiff}",
                 if ok { "OK" } else { "FAIL" }, picks.len());
        // timing: m = 1, 10 slots (a decode token's experts)
        let mut y = dev.alloc_zeros::<u16>(picks.len() * 2 * mi)?;
        dev.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..200 {
            xflaunch!(l, f, (((2 * mi) / 64) as u32, picks.len() as u32, 1), (128, 1, 1), 0,
                      (&fx.gu_w, &fx.gu_s, &fx.gs, &x, 0i64, &mut y, &idxmap, &esel, 1i32, h as i32, (2 * mi) as i32, mi as i32))?;
        }
        dev.synchronize()?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / 200.0;
        let bytes = (picks.len() * 2 * mi * (h / 2 + h / 16)) as f64;
        println!("FP4L_GPU_TIME xf_nv4_moe gate|up m=1 x 10 experts: {us:.1} us = {:.0} GB/s", bytes / us / 1e3);

        // ---- prefill tiles: 3 experts with 5, 37, 130 compact rows; every MT
        let counts = [5usize, 37, 130];
        let pe: [usize; 3] = [3, 400, 511];
        let rows: usize = counts.iter().sum();
        let row_tok: Vec<i32> = (0..rows).map(|i| ((i * 7) % 16) as i32).collect();
        let xs = lcg_f16(&mut seed, 16 * h, 1.0);
        let xdev = dev.htod_sync_copy(&xs)?;
        let rt = dev.htod_sync_copy(&row_tok)?;
        let mut offs = vec![0i32; 512];
        let mut cnt = vec![0i32; 512];
        let mut acc = 0i32;
        for (j, &e) in pe.iter().enumerate() { offs[e] = acc; cnt[e] = counts[j] as i32; acc += counts[j] as i32; }
        let (offs_d, cnt_d) = (dev.htod_sync_copy(&offs)?, dev.htod_sync_copy(&cnt)?);
        let ph: Vec<(crate::fp4l::pack::Nv4Host, crate::fp4l::pack::Nv4Host)> = pe.iter().map(|&e| -> Result<_> {
            let p = format!("model.language_model.layers.0.mlp.experts.{e}");
            Ok((fp.nvfp4(&format!("{p}.gate_proj"))?, fp.nvfp4(&format!("{p}.up_proj"))?))
        }).collect::<Result<_>>()?;
        for mt in [1usize, 2, 4, 8] {
            let mut tl: Vec<i32> = Vec::new();
            for (j, &e) in pe.iter().enumerate() { for sup in 0..counts[j].div_ceil(16 * mt) { tl.push(e as i32); tl.push(sup as i32); } }
            let ntl = (tl.len() / 2) as i32;
            let (tld, ntd) = (dev.htod_sync_copy(&tl)?, dev.htod_sync_copy(&[ntl])?);
            let mut y = dev.alloc_zeros::<u16>(rows * 2 * mi)?;
            let (fname, _) = nv4_tiles_fn(mt);
            xflaunch!(l, fname, (((2 * mi) / 128) as u32, ntl as u32, 1), (256, 1, 1), 0,
                      (&fx.gu_w, &fx.gu_s, &fx.gs, &xdev, &rt, &mut y, &tld, &ntd, &offs_d, &cnt_d,
                       (h | (2 * mi) << 16) as i32, (mi | (mt << 16)) as i32))?;
            dev.synchronize()?;
            let gy = dev.dtoh_sync_copy(&y)?;
            let mut worst = 0f64;
            for (j, &e) in pe.iter().enumerate() {
                let (g, u) = &ph[j];
                let base = offs[e] as usize;
                let xin: Vec<u16> = (0..counts[j]).flat_map(|i| { let t = row_tok[base + i] as usize; xs[t * h..(t + 1) * h].to_vec() }).collect();
                let wg = |r: usize, o: &mut [f32]| if r < mi { g.row_f32(r, o) } else { u.row_f32(r - mi, o) };
                let want = host_gemm(&wg, &xin, counts[j], 2 * mi, h);
                let (r1, _) = rel_err(&gy[base * 2 * mi..(base + counts[j]) * 2 * mi], &want);
                worst = worst.max(r1);
            }
            // mt 4 / 8: the served tiled kernel must equal xf_nv4_tiles_m4 bit for bit
            let tc_diff = if let Some(tf) = nv4_tc_fn(mt) {
                let mut y2 = dev.alloc_zeros::<u16>(rows * 2 * mi)?;
                xflaunch!(l, tf, (((2 * mi) / 128) as u32, ntl as u32, 1), (256, 1, 1), 0,
                          (&fx.gu_w, &fx.gu_s, &fx.gs, &xdev, &rt, &mut y2, &tld, &ntd, &offs_d, &cnt_d,
                           (h | (2 * mi) << 16) as i32, (mi | (mt << 16)) as i32))?;
                dev.synchronize()?;
                Some(count_diff(&gy, &dev.dtoh_sync_copy(&y2)?))
            } else { None };
            let ok = worst < 2e-3 && tc_diff.unwrap_or(0) == 0;
            if !ok { fails += 1; }
            println!("FP4L_GPU_TILES {} {fname} (list mt {mt}): rel-L2 vs host <= {worst:.2e} over {rows} compact rows{}", if ok { "OK" } else { "FAIL" },
                     tc_diff.map_or(String::new(), |d| format!("; xf_nv4_tc vs it bit-diffs {d}")));
        }
    }

    // ---- --fp4l-mtp-sidecar: every dense sidecar linear through the decode path (m = 1, 4, 16) and the prefill arm (m = 48)
    //      vs the host chain on the effective weight; row 0 bitwise across m (batch invariance of the decode path)
    if let Some(sp) = mtp_sidecar_path(dir) {
        let sc = mtp_sidecar_open(&sp)?;
        let stems: Vec<String> = sc.hdr.keys().filter_map(|k| k.strip_suffix(".trellis")).filter(|s| !s.contains(".experts.")).map(str::to_string).collect();
        let ws = dev.alloc_zeros::<f32>(2_000_000)?;
        let rws = dev.alloc_zeros::<u16>(1)?;
        let ltws = dev.alloc_zeros::<u8>(32 << 20)?;
        let rc = dev.alloc_zeros::<u8>(256)?;
        let rb = ReconBufs { ws: *rws.device_ptr() as u64, ws_len: 0, rc: *rc.device_ptr() as u64,
                             ltws: *ltws.device_ptr() as u64, ltws_len: ltws.len() };
        for stem in &stems {
            let Some(q) = mtp_sidecar_quad(&dev, &sc, stem)? else { continue };
            let (k, n) = (q.k as usize, q.n as usize);
            let Some(FxW::Trellis { weff, bits, ks, .. }) = q.fx.as_deref() else { unreachable!() };
            let wh = dev.dtoh_sync_copy(weff)?; // [n][k] f16
            let wrow = |r: usize, o: &mut [f32]| for (c, v) in o[..k].iter_mut().enumerate() { *v = half::f16::from_bits(wh[r * k + c]).to_f32(); };
            let x_host = lcg_f16(&mut seed, 48 * k, 1.0);
            let want = host_gemm(&wrow, &x_host, 48, n, k);
            let x = dev.htod_sync_copy(&x_host)?;
            let (mut worst, mut bitdiff) = (0f64, 0usize);
            let mut row0: Option<Vec<u16>> = None;
            for m in [1usize, 4, 16, 48] {
                let mut y = dev.alloc_zeros::<u16>(m * n)?;
                fx_chain(&l, &q, q.fx.as_deref().unwrap(), &x, &mut y, m, &ws, Some(&rb))?;
                dev.synchronize()?;
                let got = dev.dtoh_sync_copy(&y)?;
                worst = worst.max(rel_err(&got, &want[..m * n]).0);
                if m <= 16 {
                    match &row0 { None => row0 = Some(got[..n].to_vec()), Some(r0) => bitdiff += count_diff(r0, &got[..n]) }
                }
            }
            let ok = worst < 3e-3 && bitdiff == 0;
            if !ok { fails += 1; }
            println!("FP4L_GPU_SIDECAR {} {stem} K={bits} [{k}x{n}] ks {ks}: rel-L2 vs host (effective weight) <= {worst:.2e} at m = 1, 4, 16 \
                      (trellis decode path) and 48 (cuBLASLt arm); row 0 bit-diffs across m = 1..16: {bitdiff}", if ok { "OK" } else { "FAIL" });
        }
    }

    // ---- dequantizers
    {
        let stem = "model.language_model.layers.0.linear_attn.in_proj_z";
        let m = fp.mxfp8_rows(stem, 0, 256)?;
        let (fc, fs) = mx8_frag(&m.codes, &m.scales, m.n, m.k);
        let (w, sc) = (dev.htod_sync_copy(&fc)?, dev.htod_sync_copy(&fs)?);
        let mut out = dev.alloc_zeros::<u16>(m.n * m.k)?;
        let ne = (m.n * m.k) as i64;
        xflaunch!(l, "xf_mx8_dequant", (((ne / 8) as u32).div_ceil(256), 1, 1), (256, 1, 1), 0, (&w, &sc, &mut out, ne, m.k as i32))?;
        dev.synchronize()?;
        let got = dev.dtoh_sync_copy(&out)?;
        let mut row = vec![0f32; m.k];
        let mut bad = 0usize;
        for r in 0..m.n { m.row_f32(r, &mut row); for c in 0..m.k { if got[r * m.k + c] != half::f16::from_f32(row[c]).to_bits() { bad += 1; } } }
        if bad != 0 { fails += 1; }
        println!("FP4L_GPU_DEQ {} xf_mx8_dequant {stem} rows 0..256: {bad} f16 bit mismatches vs host RN", if bad == 0 { "OK" } else { "FAIL" });
    }
    println!("FP4L_GPU {} ({fails} failing checks)", if fails == 0 { "PASS" } else { "FAIL" });
    if fails > 0 { bail!("{fails} FP4L GPU checks failed"); }
    Ok(())
}
