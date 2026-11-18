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
//!
//! # File map (this was the single `fp4l_fwd.rs`; every item keeps its `exl3_forward::fp4l_fwd::` path via `pub(crate) use self::<file>::*`)
//! - `mod.rs` (this file): the kernel module list, `xflaunch!`, `FxW`, the FP4L switches, `FxCtx` / `FxScratch` and the bf16 GEMM helpers.
//! - `launch.rs`: launch helpers for the trunk-kernel twins (hc / router / PLE / gate), `FxMoe`, `PleFx` and the split-K shape rules.
//! - `weights.rs`: loading the FP4-Local pack: MXFP8 / bf16 fragments, `fx_quad`, experts, the PLE table and the byte model.
//! - `chains.rs`: the dense dispatch (`fx_chain`, `fx_small`, `fx_pair`, `fx_wide`), the MTP sidecar and the wide-prefill helpers.
//! - `moe.rs`: the NVFP4 expert dispatch (decode, prefill tiles). `draft_head.rs`: the pruned / MXFP8 draft head.
//! - `probe_gpu.rs`: `--probe-fp4l-gpu`, the first GPU check of the FP4L kernels. `tests.rs`: the inline unit tests.
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

mod chains;
mod draft_head;
mod launch;
mod moe;
mod probe_gpu;
mod weights;
pub(crate) use self::chains::*;
pub(crate) use self::draft_head::*;
pub(crate) use self::launch::*;
pub(crate) use self::moe::*;
pub(crate) use self::probe_gpu::*;
pub(crate) use self::weights::*;

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

/// xf_mx8_gemm_x2's per-member pointers, passed by value (kernel twin: `struct XfPair` in gpu_fp4l.cu).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct XfPair { w: [u64; 2], s: [u64; 2], y: [u64; 2], ws: [u64; 2], cnt: [u64; 2] }
unsafe impl cudarc::driver::DeviceRepr for XfPair {}
#[cfg(test)]
mod tests;
