//! Naive-N0.5-Flash GPU side (S3+): the `gpu_naive` kernel module (loader, build-id handshake)
//! and the kernel gates that compare every kernel against the CPU reference on REAL pack tensors.
//!
//! GPU use needs the owner's go-ahead per box (memory: feedback_ask_before_using_boxes); every
//! probe here is a short burst (seconds to a few minutes on one box).

use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use super::dequant::dequant_quad;
use super::model::a1_smem;   // T0-c: the ONE smem arithmetic (window-sized); gpu.rs's old private
                           // copy is gone so host and kernel cannot drift.
use super::pack::{NaivePack, Quad};

pub const MODULE: &str = "gpu_naive";
pub const FNS: &[&str] = &[
    "nv_kernel_build_id",
    "nv_decode_dump",
    "nv_had_suh",
    "nv_had_svh",
    "nv_gemm",
    "nv_gemm_ks",
    "nv_ks_combine",
    "nv_gemm_grouped_xh",
    // S3b layer glue
    "nv_embed", "nv_rmsnorm", "nv_add_f16", "nv_rope_rows", "nv_kv_write", "nv_attn_swa",
    "nv_idx_q", "nv_idx_k", "nv_idx_score", "nv_idx_score_wide", "nv_topk", "nv_attn_dsa_split", "nv_attn_combine",
    "nv_router_logits", "nv_router_topk", "nv_moe_route", "nv_had_suh_multi", "nv_had_svh_multi",
    "nv_silu_mul", "nv_moe_combine",
    // S5 TP glue
    "nv_cvt_f16_f32", "nv_add_f32", "nv_cols_copy",
    // S6 GEMM variants
    "nv_gemm_nostage", "nv_gemm_grouped_ks", "nv_ks_combine_grouped", "nv_gemm_a1ks", "nv_gemm_grouped_a1ks",
    "nv_ks_combine_had", "nv_ks_combine_grouped_had", "nv_rope_kv",
    // S6b wide prefill GEMM
    "nv_gemm_wide", "nv_gemm_wide_grouped", "nv_attn_swa_grp", "nv_attn_dsa_split_grp",
    "nv_zero_i32", "nv_moe_hist", "nv_moe_offsets", "nv_moe_gather_had", "nv_had_svh_rows", "nv_had_suh_rows_silu", "nv_moe_gather_combine",
    // S6 F8 fast sampler
    "nv_smp_prep", "nv_smp_stats", "nv_smp_z", "nv_smp_z2", "nv_smp_gumbel", "nv_smp_pick", "nv_smp_mass", "nv_smp_mass2", "nv_smp_accept",
    "nv_smp_cands_out", "nv_smp_sample_cands", "nv_smp_note", "nv_smp_count", "nv_spec_accept_rows", "nv_rows_pack", "nv_rng_rows",
];

/// Load the embedded module (idempotent) and refuse a binary/PTX build-id mismatch.
pub fn load_module(dev: &Arc<CudaDevice>) -> Result<()> {
    if dev.get_func(MODULE, "nv_kernel_build_id").is_none() {
        dev.load_ptx(Ptx::from_src(crate::ptx_embed::GPU_NAIVE), MODULE, FNS).context("load gpu_naive module")?;
    }
    let f = dev.get_func(MODULE, "nv_kernel_build_id").context("nv_kernel_build_id missing")?;
    let mut out = dev.alloc_zeros::<u64>(1)?;
    unsafe { f.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 }, (&mut out,)) }?;
    let got = dev.dtoh_sync_copy(&out)?[0];
    let want = u64::from_str_radix(env!("KERNEL_BUILD_ID"), 16).unwrap_or(0);
    ensure!(got == want, "gpu_naive kernel build id {got:016x} != binary {want:016x} (stale PTX)");
    Ok(())
}

/// PDL (programmatic dependent launch) for the naive family: `--naive-pdl=1` on the head's argv
/// (it rides to the nodes with the rest of the argv). Every `nv_*` entry kernel already opens with
/// `NV_PDL_ENTRY()` (griddepcontrol.wait / launch_dependents); without the launch attribute those are
/// no-ops and every graph node waits for the previous node's full completion. With it, a kernel's
/// launch and prologue overlap the predecessor's tail (graph capture records programmatic edges).
/// Ordering only: the numerics are untouched (the sibling family's A5-D14 precedent).
static NAIVE_PDL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn set_pdl(on: bool) { NAIVE_PDL.store(on, std::sync::atomic::Ordering::Relaxed); }
pub fn pdl_on() -> bool { NAIVE_PDL.load(std::sync::atomic::Ordering::Relaxed) }

/// The RAW module handle for PDL launches: cudarc's `get_func` copy hides its `CUfunction`, so the
/// same embedded PTX is loaded a second time through the driver and launched with
/// `cuLaunchKernelEx` + CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION. The two copies are
/// byte-identical code; a function attribute set on one copy never reaches the other (the
/// 2026-10-03 lesson), which is why `nv_raw_fn` is the ONLY place a PDL launch gets its handle.
pub fn nv_raw_fn(name: &str) -> Option<cudarc::driver::sys::CUfunction> {
    use cudarc::driver::sys;
    #[derive(Clone, Copy)]
    struct H(sys::CUmodule);
    unsafe impl Send for H {}
    unsafe impl Sync for H {}
    #[derive(Clone, Copy)]
    struct FnH(sys::CUfunction);
    unsafe impl Send for FnH {}
    unsafe impl Sync for FnH {}
    static RAW: std::sync::OnceLock<Option<H>> = std::sync::OnceLock::new();
    static CACHE: std::sync::Mutex<Vec<(String, FnH)>> = std::sync::Mutex::new(Vec::new());
    if let Some((_, f)) = CACHE.lock().ok()?.iter().find(|(n, _)| n == name) { return Some(f.0); }
    let module = RAW.get_or_init(|| {
        let ptx_c = std::ffi::CString::new(crate::ptx_embed::GPU_NAIVE).ok()?;
        let mut m: sys::CUmodule = std::ptr::null_mut();
        let r = unsafe { sys::cuModuleLoadDataEx(&mut m, ptx_c.as_ptr() as *const std::ffi::c_void, 0, std::ptr::null_mut(), std::ptr::null_mut()) };
        if r != sys::CUresult::CUDA_SUCCESS { eprintln!("[naive-pdl] raw module load failed ({r:?})"); return None; }
        Some(H(m))
    }).as_ref()?.0;
    let cname = std::ffi::CString::new(name).ok()?;
    let mut f: sys::CUfunction = std::ptr::null_mut();
    let r = unsafe { sys::cuModuleGetFunction(&mut f, module, cname.as_ptr()) };
    if r != sys::CUresult::CUDA_SUCCESS { eprintln!("[naive-pdl] raw function {name} missing ({r:?})"); return None; }
    CACHE.lock().ok()?.push((name.to_string(), FnH(f)));
    Some(f)
}

/// `cuLaunchKernelEx` with programmatic stream serialization on `stream`; `smem` > 48 KB opts the raw
/// handle in (the attribute lives on the handle that launches, see `nv_raw_fn`).
pub unsafe fn launch_pdl_raw(stream: cudarc::driver::sys::CUstream, f: cudarc::driver::sys::CUfunction,
                             grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32,
                             params: &mut [*mut std::ffi::c_void]) -> Result<()> {
    use cudarc::driver::sys::*;
    if smem > 48 * 1024 {
        let r = cuFuncSetAttribute(f, CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32);
        ensure!(r == CUresult::CUDA_SUCCESS, "cuFuncSetAttribute(smem {smem}): {r:?}");
    }
    let mut attr = CUlaunchAttribute {
        id: CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
        pad: [0; 4],
        value: CUlaunchAttributeValue { programmaticStreamSerializationAllowed: 1 },
    };
    let mut cfg = CUlaunchConfig {
        gridDimX: grid.0, gridDimY: grid.1, gridDimZ: grid.2,
        blockDimX: block.0, blockDimY: block.1, blockDimZ: block.2,
        sharedMemBytes: smem, hStream: stream, attrs: &mut attr, numAttrs: 1,
    };
    let r = cuLaunchKernelEx(&mut cfg, f, params.as_mut_ptr(), std::ptr::null_mut());
    ensure!(r == CUresult::CUDA_SUCCESS, "PDL launch failed: {r:?}");
    Ok(())
}

/// Kernel-parameter packing for the PDL launch path: the same `DeviceRepr::as_kernel_param` cudarc's
/// typed `launch` uses, over the tuple arities the naive launches take (1..=13).
pub trait NvParams { fn nv_params(&self) -> Vec<*mut std::ffi::c_void>; }
macro_rules! nv_params_impl {
    ($($T:ident $i:tt),*) => {
        impl<$($T: cudarc::driver::DeviceRepr),*> NvParams for ($($T,)*) {
            fn nv_params(&self) -> Vec<*mut std::ffi::c_void> { vec![$(self.$i.as_kernel_param(),)*] }
        }
    };
}
nv_params_impl!(A 0);
nv_params_impl!(A 0, B 1);
nv_params_impl!(A 0, B 1, C 2);
nv_params_impl!(A 0, B 1, C 2, D 3);
nv_params_impl!(A 0, B 1, C 2, D 3, E 4);
nv_params_impl!(A 0, B 1, C 2, D 3, E 4, F 5);
nv_params_impl!(A 0, B 1, C 2, D 3, E 4, F 5, G 6);
nv_params_impl!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);
nv_params_impl!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8);
nv_params_impl!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9);
nv_params_impl!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10);
nv_params_impl!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11);
nv_params_impl!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12);

/// F13: the split-K combine folded into the split-K GEMM (`--naive-fuse-combine=0` turns it off; default on).
/// Kernel-side struct `NvFuse { svh, y, out32, cnt }` passed by value; `cnt == 0` = no fusion.
static NAIVE_FUSE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
pub fn set_fuse(on: bool) { NAIVE_FUSE.store(on, std::sync::atomic::Ordering::Relaxed); }
pub fn fuse_on() -> bool { NAIVE_FUSE.load(std::sync::atomic::Ordering::Relaxed) }
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NvFuse { pub svh: u64, pub y: u64, pub out32: u64, pub cnt: u64 }
unsafe impl cudarc::driver::DeviceRepr for NvFuse {}
/// Fused-combine counters per model: one i32 per (expert, n-tile) of the widest grouped launch.
pub const FUSE_COUNTERS: usize = 8192;

/// One quad on the device: trellis words, suh/svh f16 bits, geometry.
pub struct GpuQuad {
    pub name: String,
    pub tr: CudaSlice<u16>,
    pub suh: CudaSlice<u16>,
    pub svh: CudaSlice<u16>,
    pub k: usize,
    pub n: usize,
    /// bits per 16 weights = u16 words per tile (56 for 3.5 bpw, 96 for the 6-bit head)
    pub bpb: usize,
    /// split-K factor for the M <= 16 chain (1 = single pass)
    pub ks: usize,
    /// split-K with the staged body (nv_gemm_a1ks) rather than the per-step one (nv_gemm_ks)
    pub a1: bool,
}

/// S6 (bench 2026-10-16, `--probe-naive-gemm-bench`, m = 1): the decode GEMM wants ~512 resident
/// CTAs with the STAGED body (A in smem once + L2 prefetch); split K to get there (chunks of >= 16
/// k-tiles). Very wide outputs (>= 512 column tiles: lm_head) run the per-step body at ks 2
/// (227 GB/s vs 193 staged — the box's sustained ceiling is ~225 GB/s).
pub fn chain_plan(k: usize, n: usize) -> (usize, bool) {
    let tiles = n / 128;
    let kb = k / 16;
    if tiles == 0 || kb == 0 { return (1, true); }
    if tiles >= 512 { return (if kb % 2 == 0 { 2 } else { 1 }, false); }
    let mut ks = 1usize;
    while tiles * ks * 2 <= 512 && kb % (ks * 2) == 0 && kb / (ks * 2) >= 16 { ks *= 2; }
    (ks, true)
}

/// The engine's chain_ks rule (exl3_forward::chain_ks_target with the 200-block target), copied:
/// narrow outputs split K so ~200 CTAs are resident; >= 128 column tiles run ks = 1. (Kept for the
/// S3 gates; the forward uses `chain_plan`.)
pub fn chain_ks(k: usize, n: usize) -> usize {
    let tiles = n / 128;
    let kb = k / 16;
    if tiles == 0 || kb == 0 || tiles >= 128 { return 1; }
    let target = ((200 + tiles - 1) / tiles).min(kb).max(1);
    let mut ks = target;
    while ks > 1 && kb % ks != 0 { ks -= 1; }
    ks
}

fn le_u16(raw: &[u8]) -> Vec<u16> { raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() }

pub fn upload_quad(dev: &Arc<CudaDevice>, pack: &NaivePack, q: &Quad) -> Result<GpuQuad> {
    let tr = le_u16(&q.trellis.read_bytes(&pack.dir)?);
    let suh = le_u16(&q.suh.read_bytes(&pack.dir)?);
    let svh = le_u16(&q.svh.read_bytes(&pack.dir)?);
    ensure!(tr.len() == (q.k / 16) * (q.n / 16) * q.bits.words(), "{}: trellis length", q.name);
    Ok(GpuQuad {
        name: q.name.clone(),
        tr: dev.htod_sync_copy(&tr)?,
        suh: dev.htod_sync_copy(&suh)?,
        svh: dev.htod_sync_copy(&svh)?,
        k: q.k, n: q.n, bpb: q.bits.words(), ks: chain_plan(q.k, q.n).0, a1: chain_plan(q.k, q.n).1,
    })
}


/// The dense chain for M <= 16 rows: xh = H(x*suh); y_raw = xh W (one pass or split-K + combine);
/// y = H(y_raw) * svh. x/y are f16 bits `[m][k]` / `[m][n]`.
pub fn chain(dev: &Arc<CudaDevice>, q: &GpuQuad, x: &CudaSlice<u16>, m: usize) -> Result<CudaSlice<u16>> {
    ensure!((1..=16).contains(&m), "chain: m {m} out of 1..16");
    let (k, n) = (q.k, q.n);
    ensure!(x.len() >= m * k, "chain {}: input has {} elements, the quad needs m {m} x k {k} = {}", q.name, x.len(), m * k);
    let f = |name: &str| dev.get_func(MODULE, name).with_context(|| format!("{name} missing"));
    let mut xh = dev.alloc_zeros::<u16>(m * k)?;
    let mut yraw = dev.alloc_zeros::<u16>(m * n)?;
    let mut y = dev.alloc_zeros::<u16>(m * n)?;
    let b32 = |g: u32| LaunchConfig { grid_dim: (g, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
    unsafe { f("nv_had_suh")?.launch(b32((m * k / 128) as u32), (x, &q.suh, &mut xh, k as i32)) }?;
    if q.ks > 1 {
        let kb = k / 16;
        ensure!(kb % q.ks == 0, "ks {} does not divide kb {kb}", q.ks);
        let mut ws = dev.alloc_zeros::<f32>(q.ks * m * n)?;
        unsafe { f("nv_gemm_ks")?.launch(LaunchConfig { grid_dim: ((n / 128) as u32, q.ks as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                         (&q.tr, &xh, &mut ws, m as i32, k as i32, n as i32, q.bpb as i32)) }?;
        let mn = (m * n) as u32;
        unsafe { f("nv_ks_combine")?.launch(LaunchConfig { grid_dim: ((mn + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                            (&ws, &mut yraw, m as i32, n as i32, q.ks as i32)) }?;
    } else {
        unsafe { f("nv_gemm")?.launch(LaunchConfig { grid_dim: ((n / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: a1_smem(m, k, 1) },
                                      (&q.tr, &xh, &mut yraw, m as i32, k as i32, n as i32, q.bpb as i32)) }?;
    }
    unsafe { f("nv_had_svh")?.launch(b32((m * n / 128) as u32), (&yraw, &q.svh, &mut y, n as i32)) }?;
    Ok(y)
}

/// GPU decode of a whole quad to the rotated-basis fp16 bits `[k, n]` (the decode gate's subject).
pub fn decode_dump(dev: &Arc<CudaDevice>, q: &GpuQuad) -> Result<Vec<u16>> {
    let rc = crate::exl3_bench::tmap_inv_table();
    let rc_dev = dev.htod_sync_copy(&rc)?;
    let mut out = dev.alloc_zeros::<u16>(q.k * q.n)?;
    let (kb, nb) = (q.k / 16, q.n / 16);
    let f = dev.get_func(MODULE, "nv_decode_dump").context("nv_decode_dump missing")?;
    unsafe { f.launch(LaunchConfig { grid_dim: ((kb * nb) as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 },
                      (&q.tr, &rc_dev, &mut out, nb as i32, q.bpb as i32)) }?;
    Ok(dev.dtoh_sync_copy(&out)?)
}

fn synth_x(rows: usize, k: usize, seed: u64) -> Vec<u16> {
    // deterministic, O(1)-scale values with a varied spectrum
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    (0..rows * k).map(|_| {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        let u = (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32;
        half::f16::from_f32((u - 0.5) * 2.0).to_bits()
    }).collect()
}

fn rel_l2_f16(y: &[u16], want: &[f32]) -> f32 {
    let a: Vec<f32> = y.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect();
    super::cpu::rel_l2(&a, want)
}

/// T0-c: `chain` with the GEMM body FORCED, for the bitwise A/B. Identical inputs, identical
/// `nv_had_suh` / `nv_ks_combine` / `nv_had_svh` passes; only `nv_gemm_a1ks` (staged) and
/// `nv_gemm_ks` (per-step) differ, so any output difference is the body's and nothing else's.
///
/// `body`: `true` = staged (`nv_gemm_a1ks`, requires the window to fit), `false` = per-step
/// (`nv_gemm_ks`). The caller decides both so the gate can refuse to compare what did not run.
fn chain_forced_body(
    dev: &Arc<CudaDevice>,
    q: &GpuQuad,
    x: &CudaSlice<u16>,
    m: usize,
    body_staged: bool,
) -> Result<CudaSlice<u16>> {
    ensure!((1..=16).contains(&m), "chain_forced_body: m {m} out of 1..16");
    let (k, n) = (q.k, q.n);
    ensure!(x.len() >= m * k, "input has {} elements, quad needs m {m} x k {k} = {}", x.len(), m * k);
    let f = |name: &str| dev.get_func(MODULE, name).with_context(|| format!("{name} missing"));
    let mut xh = dev.alloc_zeros::<u16>(m * k)?;
    let mut yraw = dev.alloc_zeros::<u16>(m * n)?;
    let mut y = dev.alloc_zeros::<u16>(m * n)?;
    let b32 = |g: u32| LaunchConfig { grid_dim: (g, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
    unsafe { f("nv_had_suh")?.launch(b32((m * k / 128) as u32), (x, &q.suh, &mut xh, k as i32)) }?;
    ensure!(q.ks > 1, "quad {} has ks {}: the A/B is defined for the split-K shapes", q.name, q.ks);
    let kb = k / 16;
    ensure!(kb % q.ks == 0, "ks {} does not divide kb {kb}", q.ks);
    let mut w = dev.alloc_zeros::<f32>(q.ks * m * n)?;
    if body_staged {
        let smem = a1_smem(m, k, q.ks);
        ensure!(smem > 0, "staged body unavailable for {} at m {m}, ks {} (smem {smem})", q.name, q.ks);
        unsafe { f("nv_gemm_a1ks")?.launch(
            LaunchConfig { grid_dim: ((n / 128) as u32, q.ks as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem },
            (&q.tr, &xh, &mut w, m as i32, k as i32, n as i32, q.bpb as i32, 0u64, 0u64, NvFuse::default())) }?;
    } else {
        unsafe { f("nv_gemm_ks")?.launch(
            LaunchConfig { grid_dim: ((n / 128) as u32, q.ks as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
            (&q.tr, &xh, &mut w, m as i32, k as i32, n as i32, q.bpb as i32)) }?;
    }
    let mn = (m * n) as u32;
    unsafe { f("nv_ks_combine")?.launch(
        LaunchConfig { grid_dim: ((mn + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
        (&w, &mut yraw, m as i32, n as i32, q.ks as i32)) }?;
    unsafe { f("nv_had_svh")?.launch(b32((m * n / 128) as u32), (&yraw, &q.svh, &mut y, n as i32)) }?;
    Ok(y)
}

/// `--probe-naive-a1-ab --model-dir <pack>`: T0-c's equality gate, on GPU, on REAL quads.
///
/// The claim under test is that the staged body (`nv_gemm_a1ks`) is a drop-in replacement for the
/// per-step body (`nv_gemm_ks`) at the same `ks`. Engagement was already proven by an nsys census of
/// a live TP=4 run (306,299 `nv_gemm_a1ks` launches vs 191 `nv_gemm_ks`); this is the OTHER half —
/// that what it computes is the same. Same inputs, same surrounding passes, only the body differs,
/// so a difference is attributable to the body alone.
///
/// m = 8 is the shipped spec block (and the width T0-c exists for); m = 1..=8 shows whether the two
/// bodies agree at every width or only at the shipped one.
pub fn probe_a1_ab(dir: &str) -> Result<()> {
    let pack = NaivePack::open(dir)?;
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    load_module(&dev)?;
    println!("NAIVE A1 A/B {} (gpu_naive build id ok)", pack.dir);
    let mut fails = 0usize;
    let mut compared = 0usize;

    // Real quads spanning the shapes the forward routes to the staged body (T0-c's own shape list).
    let names = [
        "model.layers.0.mlp.down_proj",                    // 16384 -> 4096: the deepest split
        "model.layers.0.mlp.gate_proj",                    // 4096 -> 16384: widest n
        "model.layers.1.self_attn.q_proj",                 // 4096 -> 12288
        "model.layers.1.self_attn.o_proj",                 // 8192 -> 4096
        "model.layers.0.self_attn.k_proj",                 // 4096 -> 768: many CTAs
        "model.layers.0.self_attn.indexer.wq",             // 4096 -> 2048
    ];
    for name in names {
        let q = pack.quad(name)?;
        let t0 = std::time::Instant::now();
        let gq = upload_quad(&dev, &pack, q)?;
        if gq.ks <= 1 {
            println!("  {name:50} ks {}: per-step by design, A/B not applicable (skip)", gq.ks);
            continue;
        }
        let mut line = format!("  {:50} k {:5} n {:6} bpb {:2} ks {:2}:", name, gq.k, gq.n, gq.bpb, gq.ks);
        for &m in &[1usize, 2, 4, 8] {
            let x16 = synth_x(m, gq.k, 0xA1_5EED ^ m as u64);
            let x_dev = dev.htod_sync_copy(&x16)?;
            let staged = chain_forced_body(&dev, &gq, &x_dev, m, true)?;
            let ys = dev.dtoh_sync_copy(&staged)?;
            let per_step = chain_forced_body(&dev, &gq, &x_dev, m, false)?;
            let yp = dev.dtoh_sync_copy(&per_step)?;
            let diff = ys.iter().zip(&yp).filter(|(a, b)| **a != **b).count();
            let of = ys.len();
            // Also report the magnitude, so a non-zero count is actionable rather than a bare flag.
            let max_abs = ys.iter().zip(&yp)
                .filter(|(a, b)| a != b)
                .map(|(a, b)| (half::f16::from_bits(*a).to_f32() - half::f16::from_bits(*b).to_f32()).abs())
                .fold(0f32, f32::max);
            compared += 1;
            let ok = diff == 0;
            if !ok { fails += 1; }
            line.push_str(&format!(" m{m}:{} {diff}/{of} d{max_abs:.1e} |", if ok { "BITWISE" } else { "DIFFER" }));
        }
        println!("{line} [{:.1} s]", t0.elapsed().as_secs_f64());
    }
    ensure!(compared > 0, "A/B compared NOTHING - every quad was ks=1; the gate proved nothing");
    println!(
        "NAIVE_A1_AB_{} ({} shape/width pairs, {fails} differing)",
        if fails == 0 { "OK" } else { "FAIL" },
        compared
    );
    Ok(())
}

/// `--probe-naive-gemm --model-dir <pack>`: on real quads of the pack, (1) the GPU decode is
/// bitwise the CPU decode (half-K and 6-bit), (2) the GPU chain y = x W matches the CPU
/// `x @ W_orig` from the dequantized weights (f16 chain rounding: rel-L2 ~1e-2 class), (3) batch
/// invariance: row 0 of the m-row chain is bitwise the m = 1 chain for every m in 1..16,
/// (4) the grouped per-expert GEMM equals the dense chain on each member.
pub fn probe_gemm(dir: &str) -> Result<()> {
    let pack = NaivePack::open(dir)?;
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    load_module(&dev)?;
    println!("NAIVE GEMM PROBE {} (gpu_naive build id ok)", pack.dir);
    let mut fails = 0usize;

    // (1)+(2)+(3) on a spread of real quads: half-K attention / indexer / expert, 6-bit head
    let names = [
        "model.layers.0.self_attn.k_proj",            // 4096 -> 768, ks > 1
        "model.layers.0.self_attn.indexer.wq",        // 4096 -> 2048
        "model.layers.1.self_attn.q_proj",            // 4096 -> 12288
        "model.layers.1.self_attn.o_proj",            // 8192 -> 4096
        "model.layers.1.mlp.experts.3.down_proj",     // 2048 -> 4096
        "model.layers.0.mlp.down_proj",               // 16384 -> 4096 (a1 fits m = 1 only)
        "lm_head",                                    // 4096 -> 152576, 6-bit
    ];
    for name in names {
        let q = pack.quad(name)?;
        let t0 = std::time::Instant::now();
        let gq = upload_quad(&dev, &pack, q)?;
        let w_cpu_bits = super::frac::decode_trellis_host(
            &le_u16(&q.trellis.read_bytes(&pack.dir)?).iter().map(|x| *x as i16).collect::<Vec<i16>>(), q.k / 16, q.n / 16, q.bits);
        let w_gpu_bits = decode_dump(&dev, &gq)?;
        let bad = w_cpu_bits.iter().zip(&w_gpu_bits).filter(|(a, b)| a != b).count();
        let decode_ok = bad == 0;
        if !decode_ok { fails += 1; }
        // chain vs CPU x @ W_orig
        let w_orig = dequant_quad(&pack, q)?; // [k][n]
        let x16 = synth_x(16, q.k, 7);
        let x_dev = dev.htod_sync_copy(&x16)?;
        let xf: Vec<f32> = x16.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect();
        let mut want = vec![0f32; 16 * q.n];
        for r in 0..16 {
            for kk in 0..q.k {
                let xv = xf[r * q.k + kk];
                if xv == 0.0 { continue; }
                let wr = &w_orig[kk * q.n..(kk + 1) * q.n];
                let yr = &mut want[r * q.n..(r + 1) * q.n];
                for c in 0..q.n { yr[c] += xv * wr[c]; }
            }
        }
        let ms: Vec<usize> = if q.n >= 100_000 { vec![1, 8] } else { (1..=16).collect() };
        let mut worst = 0f32;
        let mut row0: Option<Vec<u16>> = None;
        let mut binv_bad = 0usize;
        for &m in &ms {
            let xm = dev.htod_sync_copy(&x16[..m * q.k])?;
            let y = chain(&dev, &gq, &xm, m)?;
            let yh = dev.dtoh_sync_copy(&y)?;
            worst = worst.max(rel_l2_f16(&yh, &want[..m * q.n]));
            match &row0 {
                None => row0 = Some(yh[..q.n].to_vec()),
                Some(r0) => if r0[..] != yh[..q.n] { binv_bad += 1; },
            }
        }
        let chain_ok = worst < 3e-2 && binv_bad == 0;
        if !chain_ok { fails += 1; }
        println!("  {:50} k {:5} n {:6} bpb {:2} ks {}: decode {} ({bad} bad of {}); chain worst rel-L2 {:.3e} over m {:?}, binv_bad {} -> {} [{:.1} s]",
                 name, q.k, q.n, gq.bpb, gq.ks, if decode_ok { "BITWISE" } else { "MISMATCH" }, w_cpu_bits.len(), worst, ms, binv_bad,
                 if decode_ok && chain_ok { "PASS" } else { "FAIL" }, t0.elapsed().as_secs_f64());
        let _ = x_dev;
    }

    // (4) grouped per-expert GEMM: two real experts stacked, esel = 2, each member == its dense chain
    {
        let e3 = pack.quad("model.layers.1.mlp.experts.3.gate_proj")?;
        let e7 = pack.quad("model.layers.1.mlp.experts.7.gate_proj")?;
        let (k, n) = (e3.k, e3.n);
        let g3 = upload_quad(&dev, &pack, e3)?;
        let g7 = upload_quad(&dev, &pack, e7)?;
        let words = (k / 16) * (n / 16) * g3.bpb;
        let mut stacked = le_u16(&e3.trellis.read_bytes(&pack.dir)?);
        stacked.extend(le_u16(&e7.trellis.read_bytes(&pack.dir)?));
        let base = dev.htod_sync_copy(&stacked)?;
        let offs = dev.htod_sync_copy(&[0u64, words as u64])?;
        let esel = dev.htod_sync_copy(&[2i32])?;
        let m = 4usize;
        let x16 = synth_x(m, k, 11);
        let x_dev = dev.htod_sync_copy(&x16)?;
        // per-expert xh (each expert's own suh), [2][m][k]
        let f = |name: &str| dev.get_func(MODULE, name).with_context(|| format!("{name} missing"));
        let mut xh = dev.alloc_zeros::<u16>(2 * m * k)?;
        let mut y = dev.alloc_zeros::<u16>(2 * m * n)?;
        for (e, gq) in [(0usize, &g3), (1usize, &g7)] {
            let mut xh_e = dev.alloc_zeros::<u16>(m * k)?;
            unsafe { f("nv_had_suh")?.launch(LaunchConfig { grid_dim: ((m * k / 128) as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 },
                                             (&x_dev, &gq.suh, &mut xh_e, k as i32)) }?;
            let h = dev.dtoh_sync_copy(&xh_e)?;
            let mut all = dev.dtoh_sync_copy(&xh)?;
            all[e * m * k..(e + 1) * m * k].copy_from_slice(&h);
            xh = dev.htod_sync_copy(&all)?;
        }
        unsafe { f("nv_gemm_grouped_xh")?.launch(LaunchConfig { grid_dim: ((2 * n / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: a1_smem(m, k, 1) },
                                                 (&base, &offs, &xh, &mut y, m as i32, k as i32, n as i32, g3.bpb as i32, &esel)) }?;
        let yg = dev.dtoh_sync_copy(&y)?;
        let mut grouped_ok = true;
        for (e, gq) in [(0usize, &g3), (1usize, &g7)] {
            // dense reference: the single-pass raw GEMM on this expert (ks = 1 for n = 2048 -> 16 tiles => ks > 1!)
            // so compare y_raw via the full chain instead: run the dense chain and the grouped output through svh.
            let mut yraw_e = dev.alloc_zeros::<u16>(m * n)?;
            let xh_e: Vec<u16> = dev.dtoh_sync_copy(&xh)?[e * m * k..(e + 1) * m * k].to_vec();
            let xh_e_dev = dev.htod_sync_copy(&xh_e)?;
            unsafe { f("nv_gemm")?.launch(LaunchConfig { grid_dim: ((n / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: a1_smem(m, k, 1) },
                                          (&gq.tr, &xh_e_dev, &mut yraw_e, m as i32, k as i32, n as i32, gq.bpb as i32)) }?;
            let yr = dev.dtoh_sync_copy(&yraw_e)?;
            let bad = yr.iter().zip(&yg[e * m * n..(e + 1) * m * n]).filter(|(a, b)| a != b).count();
            if bad != 0 { grouped_ok = false; }
            println!("  grouped expert slot {e} ({}): {} of {} y_raw words differ from the dense single-pass GEMM", gq.name, bad, m * n);
        }
        if !grouped_ok { fails += 1; }
        println!("  grouped GEMM -> {}", if grouped_ok { "PASS (bitwise)" } else { "FAIL" });
    }

    if fails > 0 { bail!("NAIVE_GEMM_FAIL: {fails} gate(s) failed"); }
    println!("NAIVE_GEMM_OK");
    Ok(())
}

/// `--probe-naive-idx-prefill --model-dir <pack>`: the tensor-core prefill scorer `nv_idx_score_wide` vs the scalar
/// `nv_idx_score` on synthetic fp8-rounded planes (M = 64..1024 rows, positions up to 8192): scores rel-L2 and
/// the exact top-k selections (differing indices per row), plus kernel timing at M=1024, L=8192.
pub fn probe_idx_prefill(dir: &str) -> Result<()> {
    let pack = NaivePack::open(dir)?;
    let c = &pack.cfg;
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    load_module(&dev)?;
    let (ih, ihd, kk) = (c.index_n_heads, c.index_head_dim, c.index_top_k);
    ensure!(ih == 16 && ihd == 128, "wide scorer needs 16 heads x 128 dims");
    let lmax = 8192usize;
    let mut seed = 0xBADC0DEu64;
    // the probes run a single request: every row selects cache slot 0
    let sids_d: CudaSlice<i32> = dev.htod_sync_copy(&vec![0i32; super::model::MAX_PREFILL_ROWS])?;   // sized for the WIDE probe cells (rows up to 300), not MAX_ROWS: slot_ids[m] is read per row
    let mut rnd = || { seed ^= seed >> 12; seed ^= seed << 25; seed ^= seed >> 27; (seed.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32 - 0.5 };
    let mut kq = vec![0u8; lmax * ihd];
    let mut ks = vec![0f32; lmax];
    for p in 0..lmax {
        ks[p] = (0.5 + 0.5 * (rnd() + 0.5)) / 448.0;
        for d in 0..ihd { kq[p * ihd + d] = crate::dsv4_cpu::f32_to_e4m3_rne(rnd() * 400.0); }
    }
    let mmax = 1024usize;
    let mut iq = vec![0f32; mmax * ih * ihd];
    for v in iq.iter_mut() { *v = crate::quant::e4m3_to_f32(crate::dsv4_cpu::f32_to_e4m3_rne(rnd() * 300.0)) * 0.01; }
    let mut wgt16 = vec![0u16; mmax * 128];
    for r in 0..mmax { for h in 0..ih { wgt16[r * 128 + h] = half::f16::from_f32(rnd()).to_bits(); } }
    let f = |name: &str| dev.get_func(MODULE, name).with_context(|| format!("{name} missing"));
    let kq_d = dev.htod_sync_copy(&kq)?; let ks_d = dev.htod_sync_copy(&ks)?;
    let iq_d = dev.htod_sync_copy(&iq)?; let wgt_d = dev.htod_sync_copy(&wgt16)?;
    let mut ok = true;
    // (rows, keys L, first position): rows sit at pos0..pos0+m-1 (a prefill chunk), L = pos0 + m
    for &(m, l) in &[(64usize, 2048usize), (64, 8192), (256, 4096), (512, 8192), (1024, 1024), (1024, 8192)] {
        let pos: Vec<i32> = (0..m).map(|r| (l - m + r) as i32).collect();
        let pos_d = dev.htod_sync_copy(&pos)?;
        let mut sc_a = dev.alloc_zeros::<f32>(m * l)?;
        let mut sc_b = dev.alloc_zeros::<f32>(m * l)?;
        let mut sel_a = dev.alloc_zeros::<i32>(m * kk)?; let mut nsel_a = dev.alloc_zeros::<i32>(m)?;
        let mut sel_b = dev.alloc_zeros::<i32>(m * kk)?; let mut nsel_b = dev.alloc_zeros::<i32>(m)?;
        let cfg_s = LaunchConfig { grid_dim: (512, m as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let cfg_w = LaunchConfig { grid_dim: (((l + 127) / 128) as u32, ((m + 31) / 32) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let run_s = |sc: &mut CudaSlice<f32>| -> Result<()> { unsafe { f("nv_idx_score")?.launch(cfg_s, (sc, &iq_d, &wgt_d, &kq_d, &ks_d, &pos_d, &sids_d, ih as i32, ihd as i32, 128i32, l as i64, l as i32)) }?; Ok(()) };
        let run_w = |sc: &mut CudaSlice<f32>| -> Result<()> { unsafe { f("nv_idx_score_wide")?.launch(cfg_w, (sc, &iq_d, &wgt_d, &kq_d, &ks_d, &pos_d, &sids_d, m as i32, l as i32, 128i32, l as i64)) }?; Ok(()) };
        run_s(&mut sc_a)?; run_w(&mut sc_b)?; dev.synchronize()?;
        let time = |g: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
            dev.synchronize()?; let t = std::time::Instant::now();
            for _ in 0..5 { g()?; }
            dev.synchronize()?; Ok(t.elapsed().as_secs_f64() * 1e3 / 5.0)
        };
        let ts = time(&mut || run_s(&mut sc_a))?;
        let tw = time(&mut || run_w(&mut sc_b))?;
        for (sc, sel, nsel) in [(&sc_a, &mut sel_a, &mut nsel_a), (&sc_b, &mut sel_b, &mut nsel_b)] {
            unsafe { f("nv_topk")?.launch(LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 }, (sel, nsel, sc, &pos_d, kk as i32, l as i64, 0u64, 0u64)) }?;
        }
        dev.synchronize()?;
        let (ha, hb) = (dev.dtoh_sync_copy(&sc_a)?, dev.dtoh_sync_copy(&sc_b)?);
        let (sa, sb) = (dev.dtoh_sync_copy(&sel_a)?, dev.dtoh_sync_copy(&sel_b)?);
        let (na, nb) = (dev.dtoh_sync_copy(&nsel_a)?, dev.dtoh_sync_copy(&nsel_b)?);
        let (mut num, mut den) = (0f64, 0f64);
        let mut ndiff = 0usize; let mut maxdiff = 0usize;
        for r in 0..m {
            let n = pos[r] as usize + 1;
            for t in 0..n { let (a, b) = (ha[r * l + t] as f64, hb[r * l + t] as f64); num += (a - b) * (a - b); den += a * a; }
            ensure!(na[r] == nb[r], "nsel differs at row {r}: {} vs {}", na[r], nb[r]);
            let ga: std::collections::HashSet<i32> = sa[r * kk..r * kk + na[r] as usize].iter().copied().collect();
            let d = sb[r * kk..r * kk + nb[r] as usize].iter().filter(|x| !ga.contains(x)).count();
            ndiff += d; maxdiff = maxdiff.max(d);
        }
        let rel = (num / den).sqrt();
        let avg = ndiff as f64 / m as f64;
        println!("  M {m:4} L {l:5}: scores rel-L2 {rel:.3e}; top-{kk} differing/row avg {avg:.2} max {maxdiff}; scalar {ts:.3} ms, wide {tw:.3} ms ({:.1}x)", ts / tw);
        if !(rel < 2e-3) || avg > 4.0 { ok = false; }
    }
    if !ok { bail!("NAIVE_IDX_PREFILL_FAIL"); }
    println!("NAIVE_IDX_PREFILL_OK");
    Ok(())
}

/// `--probe-naive-topk --model-dir <pack>`: the indexer scorer + exact top-k at LONG positions, GPU vs
/// CPU, on synthetic data (random fp8-rounded keys and queries, mixed-sign weights): positions
/// 0..4095 with M = 4 query rows at pos 4095, 3000, 2048 and 1500, K = 2048 — the selection sets must
/// be identical (stable descending order, ties -> lower index) and the scores within f32 noise.
pub fn probe_topk(dir: &str) -> Result<()> {
    let pack = NaivePack::open(dir)?;
    let c = &pack.cfg;
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    load_module(&dev)?;
    let (ih, ihd, kk) = (c.index_n_heads, c.index_head_dim, c.index_top_k);
    let npos = 4096usize;
    let m = 4usize;
    let pos: Vec<usize> = vec![4095, 3000, 2048, 1500];
    // synthetic key plane: e4m3 bytes + scales (what nv_idx_k would have written), queries already rounded
    let mut seed = 0xC0FFEEu64;
    let mut rnd = || { seed ^= seed >> 12; seed ^= seed << 25; seed ^= seed >> 27; (seed.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32 - 0.5 };
    let mut kq = vec![0u8; npos * ihd];
    let mut ks = vec![0f32; npos];
    let mut kf = vec![0f32; npos * ihd]; // dequantized keys for the CPU
    for p in 0..npos {
        let scale = 0.5 + 0.5 * (rnd() + 0.5);
        ks[p] = scale / 448.0 * 448.0 / 448.0; // any positive scale
        for d in 0..ihd {
            let v = rnd() * 400.0;
            let code = crate::dsv4_cpu::f32_to_e4m3_rne(v);
            kq[p * ihd + d] = code;
            kf[p * ihd + d] = crate::quant::e4m3_to_f32(code) * ks[p];
        }
    }
    let mut iq = vec![0f32; m * ih * ihd];
    for v in iq.iter_mut() { *v = crate::quant::e4m3_to_f32(crate::dsv4_cpu::f32_to_e4m3_rne(rnd() * 300.0)) * 0.01; }
    let mut wgt16 = vec![0u16; m * 128];
    let mut wgt = vec![0f32; m * ih];
    for r in 0..m { for h in 0..ih { let w = rnd(); wgt[r * ih + h] = half::f16::from_f32(w).to_f32(); wgt16[r * 128 + h] = half::f16::from_f32(w).to_bits(); } }
    // CPU reference
    let wscale = (ih as f32).powf(-0.5);
    let mut sc_cpu = vec![f32::NEG_INFINITY; m * npos];
    for r in 0..m { for t in 0..=pos[r] {
        let mut s = 0f32;
        for h in 0..ih { let mut dot = 0f32; for d in 0..ihd { dot += iq[(r * ih + h) * ihd + d] * kf[t * ihd + d]; } s += dot.max(0.0) * (wgt[r * ih + h] * wscale); }
        sc_cpu[r * npos + t] = s;
    } }
    let sel_cpu: Vec<Vec<usize>> = (0..m).map(|r| { let mut v = super::cpu::topk_stable_desc(&sc_cpu[r * npos..r * npos + pos[r] + 1], kk); v.sort(); v }).collect();
    // GPU
    let f = |name: &str| dev.get_func(MODULE, name).with_context(|| format!("{name} missing"));
    let kq_d = dev.htod_sync_copy(&kq)?; let ks_d = dev.htod_sync_copy(&ks)?;
    let iq_d = dev.htod_sync_copy(&iq)?; let wgt_d = dev.htod_sync_copy(&wgt16)?;
    let pos_d = dev.htod_sync_copy(&pos.iter().map(|p| *p as i32).collect::<Vec<i32>>())?;
    let mut scores = dev.alloc_zeros::<f32>(m * npos)?;
    unsafe { f("nv_idx_score")?.launch(LaunchConfig { grid_dim: (((npos + 7) / 8) as u32, m as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                       (&mut scores, &iq_d, &wgt_d, &kq_d, &ks_d, &pos_d, ih as i32, ihd as i32, 128i32, npos as i64)) }?;
    let mut sel = dev.alloc_zeros::<i32>(m * kk)?;
    let mut nsel = dev.alloc_zeros::<i32>(m)?;
    unsafe { f("nv_topk")?.launch(LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 }, (&mut sel, &mut nsel, &scores, &pos_d, kk as i32, npos as i64, 0u64, 0u64)) }?;
    dev.synchronize()?;
    let sc_gpu = dev.dtoh_sync_copy(&scores)?;
    let sel_g = dev.dtoh_sync_copy(&sel)?; let nsel_g = dev.dtoh_sync_copy(&nsel)?;
    let mut ok = true;
    for r in 0..m {
        let n = pos[r] + 1;
        let rel = super::cpu::rel_l2(&sc_gpu[r * npos..r * npos + n], &sc_cpu[r * npos..r * npos + n]);
        let ns = nsel_g[r] as usize;
        let mut g: Vec<usize> = sel_g[r * kk..r * kk + ns].iter().map(|x| *x as usize).collect();
        let ascending = g.windows(2).all(|w| w[0] < w[1]);
        g.sort();
        let same = g == sel_cpu[r];
        let diff = g.iter().filter(|x| !sel_cpu[r].contains(x)).count();
        println!("  row {r} pos {:5}: scores rel-L2 {rel:.3e}; nsel {ns} (want {}); emitted ascending {ascending}; selection {} ({diff} differ)",
                 pos[r], sel_cpu[r].len(), if same { "IDENTICAL" } else { "DIFFERS" });
        if rel > 1e-4 || !same || !ascending { ok = false; }
    }
    if !ok { bail!("NAIVE_TOPK_FAIL"); }
    println!("NAIVE_TOPK_OK");
    Ok(())
}


/// `--probe-naive-gemm-bench --model-dir <pack> [--gemm-m M]`: time the decode-path GEMM variants on the
/// real shapes (µs per call and effective weight bandwidth), the S6 tuning loop's instrument.
pub fn probe_gemm_bench(dir: &str, m: usize) -> Result<()> {
    use cudarc::driver::DevicePtr;
    let pack = NaivePack::open(dir)?;
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    load_module(&dev)?;
    let stream = dev.fork_default_stream()?;
    let f = |name: &str| dev.get_func(MODULE, name).with_context(|| format!("{name} missing"));
    let time = |iters: usize, body: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
        body()?; unsafe { cudarc::driver::result::stream::synchronize(stream.stream) }?;
        let t0 = std::time::Instant::now();
        for _ in 0..iters { body()?; }
        unsafe { cudarc::driver::result::stream::synchronize(stream.stream) }?;
        Ok(t0.elapsed().as_secs_f64() / iters as f64)
    };
    println!("NAIVE GEMM BENCH m {m}: µs per call (weight GB/s)");
    // S6b: the wide GEMM at prefill widths — µs and the row rate (the token rate a layer of this quad would sustain)
    for (name, label) in [("model.layers.1.self_attn.q_proj", "q_proj"), ("model.layers.1.mlp.experts.3.gate_proj", "expert gate"), ("model.layers.0.mlp.gate_proj", "dense gate"), ("model.layers.1.self_attn.o_proj", "o_proj")] {
        let q = upload_quad(&dev, &pack, pack.quad(name)?)?;
        let (k, n, bpb) = (q.k, q.n, q.bpb);
        let mut line = format!("  WIDE {label:12} k {k:6} n {n:6}:");
        // the gate: the wide output must be BITWISE the small-M (ks = 1, staged) kernel's, row by row
        {
            let mw = 48usize;
            let xs: Vec<u16> = (0..mw * k).map(|i| half::f16::from_f32((((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0) * 0.05).to_bits()).collect();
            let xh = dev.htod_sync_copy(&xs)?;
            let mut yw = dev.alloc_zeros::<u16>(mw * n)?;
            let smem = 4 * 16 * (8 * 16 + 8) * 2;
            unsafe { f("nv_gemm_wide")?.launch_on_stream(&stream, LaunchConfig { grid_dim: ((mw as u32 + 63) / 64, (n / 128) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem as u32 }, (&q.tr, &xh, &mut yw, mw as i32, k as i32, n as i32, bpb as i32, 4i32)) }?;
            let mut bad = 0usize;
            for g in 0..mw / 16 {
                let xg = dev.htod_sync_copy(&xs[g * 16 * k..(g + 1) * 16 * k])?;
                let mut yg = dev.alloc_zeros::<u16>(16 * n)?;
                unsafe { f("nv_gemm")?.launch_on_stream(&stream, LaunchConfig { grid_dim: ((n / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: a1_smem(16, k, 1) }, (&q.tr, &xg, &mut yg, 16i32, k as i32, n as i32, bpb as i32)) }?;
                unsafe { cudarc::driver::result::stream::synchronize(stream.stream) }?;
                let a = dev.dtoh_sync_copy(&yg)?;
                let b = dev.dtoh_sync_copy(&yw)?;
                bad += a.iter().zip(&b[g * 16 * n..(g + 1) * 16 * n]).filter(|(x, y)| x != y).count();
            }
            line += &format!(" gate(M48 vs 3x m16): {bad} words differ |");
        }
        for mw in [64usize, 256, 1024] {
            let xh = dev.alloc_zeros::<u16>(mw * k)?;
            let mut y = dev.alloc_zeros::<u16>(mw * n)?;
            for mt in [2u32, 4, 8] {
                let smem = (mt as usize) * 16 * (8 * 16 + 8) * 2;
                let grid = ((mw as u32 + mt * 16 - 1) / (mt * 16), (n / 128) as u32, 1);
                let t = time(5, &mut || unsafe { f("nv_gemm_wide")?.launch_on_stream(&stream, LaunchConfig { grid_dim: grid, block_dim: (256, 1, 1), shared_mem_bytes: smem as u32 }, (&q.tr, &xh, &mut y, mw as i32, k as i32, n as i32, bpb as i32, mt as i32)) }.map_err(Into::into))?;
                line += &format!(" M{mw}/mt{mt} {:7.0}us ({:6.0} rows/s)", t * 1e6, mw as f64 / t);
            }
        }
        println!("{line}");
    }
    let shapes = [("model.layers.1.mlp.experts.3.gate_proj", "expert gate/up"), ("model.layers.1.mlp.experts.3.down_proj", "expert down"),
                  ("model.layers.0.mlp.gate_proj", "dense gate/up"), ("model.layers.0.mlp.down_proj", "dense down"),
                  ("model.layers.1.self_attn.q_proj", "q_proj"), ("model.layers.1.self_attn.o_proj", "o_proj"), ("lm_head", "lm_head")];
    for (name, label) in shapes {
        let q = upload_quad(&dev, &pack, pack.quad(name)?)?;
        let (k, n, bpb) = (q.k, q.n, q.bpb);
        let bytes = (k / 16) * (n / 16) * bpb * 2;
        let gbs = |t: f64| bytes as f64 / t / 1e9;
        let xh = dev.alloc_zeros::<u16>(m * k)?;
        let mut y = dev.alloc_zeros::<u16>(m * n)?;
        let mut line = format!("  {label:14} k {k:6} n {n:6} ({:.1} MB):", bytes as f64 / 1e6);
        let a1 = a1_smem(m, k, 1);
        let t = time(20, &mut || unsafe { f("nv_gemm")?.launch_on_stream(&stream, LaunchConfig { grid_dim: ((n / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: a1 }, (&q.tr, &xh, &mut y, m as i32, k as i32, n as i32, bpb as i32)) }.map_err(Into::into))?;
        line += &format!(" a1 {:6.1} ({:3.0})", t * 1e6, gbs(t));
        let t = time(20, &mut || unsafe { f("nv_gemm_nostage")?.launch_on_stream(&stream, LaunchConfig { grid_dim: ((n / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, (&q.tr, &xh, &mut y, m as i32, k as i32, n as i32, bpb as i32)) }.map_err(Into::into))?;
        line += &format!(" step {:6.1} ({:3.0})", t * 1e6, gbs(t));
        for ks in [2usize, 4, 8, 16] {
            if (k / 16) % ks != 0 { continue; }
            let mut w = dev.alloc_zeros::<f32>(ks * m * n)?;
            let t = time(20, &mut || {
                unsafe { f("nv_gemm_ks")?.launch_on_stream(&stream, LaunchConfig { grid_dim: ((n / 128) as u32, ks as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, (&q.tr, &xh, &mut w, m as i32, k as i32, n as i32, bpb as i32)) }?;
                unsafe { f("nv_ks_combine")?.launch_on_stream(&stream, LaunchConfig { grid_dim: (((m * n) as u32 + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, (&w, &mut y, m as i32, n as i32, ks as i32)) }?;
                Ok(())
            })?;
            line += &format!(" ks{ks} {:6.1} ({:3.0})", t * 1e6, gbs(t));
            if a1 > 0 {
                let t = time(20, &mut || {
                    unsafe { f("nv_gemm_a1ks")?.launch_on_stream(&stream, LaunchConfig { grid_dim: ((n / 128) as u32, ks as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: a1 }, (&q.tr, &xh, &mut w, m as i32, k as i32, n as i32, bpb as i32, 0u64, 0u64, NvFuse::default())) }?;
                    unsafe { f("nv_ks_combine")?.launch_on_stream(&stream, LaunchConfig { grid_dim: (((m * n) as u32 + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, (&w, &mut y, m as i32, n as i32, ks as i32)) }?;
                    Ok(())
                })?;
                line += &format!(" a1ks{ks} {:6.1} ({:3.0})", t * 1e6, gbs(t));
            }
        }
        println!("{line}");
        if label.starts_with("expert") {
            // the grouped form at 8 experts (the same trellis 8 times: bandwidth-equivalent to 8 distinct ones? no —
            // L2 would hit; use 8 DISTINCT experts of the layer)
            let e = 8usize;
            let mut trs = Vec::new(); let mut offs = Vec::new(); let mut off = 0u64;
            for ex in 0..e {
                let qe = pack.quad(&format!("model.layers.1.mlp.experts.{ex}.{}", if label.contains("gate") { "gate_proj" } else { "down_proj" }))?;
                let w = super::shard::read_quad_host(&pack, qe)?.0;
                offs.push(off); off += w.len() as u64; trs.extend(w);
            }
            let base = dev.htod_sync_copy(&trs)?; let offs_d = dev.htod_sync_copy(&offs)?;
            let esel = dev.htod_sync_copy(&[e as i32])?;
            let suh_e = dev.htod_sync_copy(&vec![0x3c00u16; e * k])?; // ones
            let idx_e = dev.htod_sync_copy(&(0..e as i32).collect::<Vec<i32>>())?;
            let xe = dev.alloc_zeros::<u16>(e * m * k)?;
            let mut ye = dev.alloc_zeros::<u16>(e * m * n)?;
            let gb8 = |t: f64| (bytes * e) as f64 / t / 1e9;
            let mut line = format!("  {:14} x{e} experts:", label);
            let t = time(20, &mut || unsafe { f("nv_gemm_grouped_xh")?.launch_on_stream(&stream, LaunchConfig { grid_dim: ((e * n / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: a1 }, (&base, &offs_d, &xe, &mut ye, m as i32, k as i32, n as i32, bpb as i32, &esel)) }.map_err(Into::into))?;
            line += &format!(" grouped {:6.1} ({:3.0})", t * 1e6, gb8(t));
            for ks in [2usize, 4, 8, 16] {
                if (k / 16) % ks != 0 { continue; }
                let mut w = dev.alloc_zeros::<f32>(e * ks * m * n)?;
                let t = time(20, &mut || {
                    unsafe { f("nv_gemm_grouped_ks")?.launch_on_stream(&stream, LaunchConfig { grid_dim: ((e * n / 128) as u32, ks as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, (&base, &offs_d, &xe, &mut w, m as i32, k as i32, n as i32, bpb as i32, &esel)) }?;
                    unsafe { f("nv_ks_combine_grouped")?.launch_on_stream(&stream, LaunchConfig { grid_dim: (((e * m * n) as u32 + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, (&w, &mut ye, m as i32, n as i32, ks as i32, &esel)) }?;
                    Ok(())
                })?;
                line += &format!(" gks{ks} {:6.1} ({:3.0})", t * 1e6, gb8(t));
                if a1 > 0 {
                    let t = time(20, &mut || {
                        unsafe { f("nv_gemm_grouped_a1ks")?.launch_on_stream(&stream, LaunchConfig { grid_dim: ((e * n / 128) as u32, ks as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: a1 }, (&base, &offs_d, &xe, &mut w, (m | (bpb << 8)) as i32, (k | (n << 16)) as i32, &esel, &suh_e, &idx_e, 0u64, 1i32, NvFuse::default())) }?;
                        unsafe { f("nv_ks_combine_grouped")?.launch_on_stream(&stream, LaunchConfig { grid_dim: (((e * m * n) as u32 + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, (&w, &mut ye, m as i32, n as i32, ks as i32, &esel)) }?;
                        Ok(())
                    })?;
                    line += &format!(" ga1ks{ks} {:6.1} ({:3.0})", t * 1e6, gb8(t));
                }
            }
            println!("{line}");
        }
    }
    Ok(())
}

/// `--probe-naive-attn-prefill`: the group-shared prefill attention kernels (nv_attn_swa_grp, nv_attn_dsa_split_grp)
/// against the per-head decode kernels on the same random inputs: the outputs must be BITWISE identical.
pub fn probe_attn_prefill() -> Result<()> {
    use half::f16;
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    load_module(&dev)?;
    let mut seed = 0x1234_5678_9abc_def1u64;
    // the probes run a single request: every row selects cache slot 0
    let sids_d: CudaSlice<i32> = dev.htod_sync_copy(&vec![0i32; super::model::MAX_PREFILL_ROWS])?;   // sized for the WIDE probe cells (rows up to 300), not MAX_ROWS: slot_ids[m] is read per row
    let mut rnd = move || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((seed >> 40) as f32 / 16777216.0) * 2.0 - 1.0 };
    let cfg1 = |g: u32, b: u32| LaunchConfig { grid_dim: (g, 1, 1), block_dim: (b, 1, 1), shared_mem_bytes: 0 };
    let _ = cfg1;
    // ---- SWA: (nh, kv, hd, vd, W, has_sink)
    for &(nh, kv, hd, vd, w, has_sink, m) in &[(64usize, 8usize, 128usize, 128usize, 128usize, 1usize, 300usize), (64, 8, 128, 128, 128, 0, 300), (32, 8, 128, 128, 100, 1, 200), (8, 8, 128, 128, 128, 1, 40), (64, 4, 192, 128, 128, 1, 100)] {
        if nh / kv > 8 { continue; }
        let s_ring = 256usize;
        let hv = |n: usize, rnd: &mut dyn FnMut() -> f32, sc: f32| -> Vec<u16> { (0..n).map(|_| f16::from_f32(rnd() * sc).to_bits()).collect() };
        let q = dev.htod_sync_copy(&hv(m * nh * hd, &mut rnd, 2.0))?;
        let kc = dev.htod_sync_copy(&hv(kv * s_ring * hd, &mut rnd, 2.0))?;
        let vc = dev.htod_sync_copy(&hv(kv * s_ring * vd, &mut rnd, 1.0))?;
        let sink = dev.htod_sync_copy(&hv(nh, &mut rnd, 1.0))?;
        let pos = dev.htod_sync_copy(&(0..m as i32).map(|i| i * 3 / 2).collect::<Vec<i32>>())?;
        let mut o1 = dev.alloc_zeros::<u16>(m * nh * vd)?;
        let mut o2 = dev.alloc_zeros::<u16>(m * nh * vd)?;
        let nhkv = (nh | (kv << 16)) as i32; let hdvd = (hd | (vd << 16)) as i32; let ws = (w | (has_sink << 16)) as i32;
        let f1 = dev.get_func(MODULE, "nv_attn_swa").context("nv_attn_swa")?;
        unsafe { f1.launch(LaunchConfig { grid_dim: (nh as u32, m as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, (&mut o1, &q, &kc, &vc, &sink, &pos, &sids_d, nhkv, hdvd, s_ring as i32, ws)) }?;
        let f2 = dev.get_func(MODULE, "nv_attn_swa_grp").context("nv_attn_swa_grp")?;
        unsafe { f2.launch(LaunchConfig { grid_dim: (kv as u32, m as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, (&mut o2, &q, &kc, &vc, &sink, &pos, &sids_d, nhkv, hdvd, s_ring as i32, ws)) }?;
        let (a, b) = (dev.dtoh_sync_copy(&o1)?, dev.dtoh_sync_copy(&o2)?);
        let bad = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        println!("  swa nh {nh} kv {kv} hd {hd} W {w} sink {has_sink} m {m}: {} of {} outputs differ", bad, a.len());
        ensure!(bad == 0, "nv_attn_swa_grp differs from nv_attn_swa");
    }
    // ---- DSA split: nh 64, kv 4 (gqa 16)
    for &(nh, kv, m) in &[(64usize, 4usize, 120usize), (64, 8, 70)] {
        let (hd, vd, s_len, kk, nsplits) = (192usize, 128usize, 512usize, 256usize, 8usize);
        let split_len = (kk + nsplits - 1) / nsplits;
        let hv = |n: usize, rnd: &mut dyn FnMut() -> f32, sc: f32| -> Vec<u16> { (0..n).map(|_| f16::from_f32(rnd() * sc).to_bits()).collect() };
        let q = dev.htod_sync_copy(&hv(m * nh * hd, &mut rnd, 2.0))?;
        let kc = dev.htod_sync_copy(&hv(kv * s_len * hd, &mut rnd, 2.0))?;
        let vc = dev.htod_sync_copy(&hv(kv * s_len * vd, &mut rnd, 1.0))?;
        let mut sel = vec![0i32; m * kk];
        let mut nsel = vec![0i32; m];
        for i in 0..m { let n = (kk as i32).min(1 + (i as i32 * 5) % (kk as i32 + 40)); nsel[i] = n; for r in 0..n as usize { sel[i * kk + r] = ((r * 2 + i) % s_len) as i32; } }
        let (sel, nsel) = (dev.htod_sync_copy(&sel)?, dev.htod_sync_copy(&nsel)?);
        let n = m * nh * nsplits;
        // Same layout as the nv_attn_dsa_split launch in model.rs: K [0:16) | nsplits [16:24) | split_len [24:40) | S [40:64).
        // Keep the two packs in lockstep -- they feed the same kernel and are not type-checked against it.
        let geom = super::model::pack_geom(kk, nsplits, split_len, s_len);
        let mut outs = Vec::new();
        for name in ["nv_attn_dsa_split", "nv_attn_dsa_split_grp"] {
            let mut pm = dev.alloc_zeros::<f32>(n)?; let mut pl = dev.alloc_zeros::<f32>(n)?; let mut pa = dev.alloc_zeros::<f32>(n * vd)?;
            let f = dev.get_func(MODULE, name).with_context(|| name.to_string())?;
            unsafe { f.launch(LaunchConfig { grid_dim: ((m * kv * nsplits) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                              (&mut pm, &mut pl, &mut pa, &q, &kc, &vc, &sel, &nsel, &sids_d, (nh | (kv << 16)) as i32, (hd | (vd << 16)) as i32, geom)) }?;
            outs.push((dev.dtoh_sync_copy(&pm)?, dev.dtoh_sync_copy(&pl)?, dev.dtoh_sync_copy(&pa)?));
        }
        let bits = |v: &Vec<f32>| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        let bad = [(0, 0), (1, 1), (2, 2)].iter().map(|_| 0).sum::<usize>()
            + bits(&outs[0].0).iter().zip(bits(&outs[1].0)).filter(|(a, b)| **a != *b).count()
            + bits(&outs[0].1).iter().zip(bits(&outs[1].1)).filter(|(a, b)| **a != *b).count()
            + bits(&outs[0].2).iter().zip(bits(&outs[1].2)).filter(|(a, b)| **a != *b).count();
        println!("  dsa nh {nh} kv {kv} m {m}: {bad} partial values differ");
        ensure!(bad == 0, "nv_attn_dsa_split_grp differs from nv_attn_dsa_split");
        // The decode geometry (64 splits of 32 keys) against the 8-split one, COMBINED: a different split
        // association of the same online softmax (fp32 partials merged ascending), so the gate is a
        // tolerance on the f16 outputs, not bit identity. Every row here selects >= 1 key; rows with
        // nsel up to kk exercise every split.
        let mut comb = Vec::new();
        for ns in [8usize, 64usize] {
            let sl = (kk + ns - 1) / ns;
            let n2 = m * nh * ns;
            let geom2 = super::model::pack_geom(kk, ns, sl, s_len);
            let mut pm = dev.alloc_zeros::<f32>(n2)?; let mut pl = dev.alloc_zeros::<f32>(n2)?; let mut pa = dev.alloc_zeros::<f32>(n2 * vd)?;
            let f = dev.get_func(MODULE, "nv_attn_dsa_split").context("nv_attn_dsa_split")?;
            unsafe { f.launch(LaunchConfig { grid_dim: ((m * kv * ns) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                              (&mut pm, &mut pl, &mut pa, &q, &kc, &vc, &sel, &nsel, &sids_d, (nh | (kv << 16)) as i32, (hd | (vd << 16)) as i32, geom2)) }?;
            let mut out = dev.alloc_zeros::<u16>(m * nh * vd)?;
            let fc = dev.get_func(MODULE, "nv_attn_combine").context("nv_attn_combine")?;
            unsafe { fc.launch(LaunchConfig { grid_dim: ((m * nh) as u32, 1, 1), block_dim: (vd as u32, 1, 1), shared_mem_bytes: 0 },
                               (&mut out, &pm, &pl, &pa, ns as i32, vd as i32)) }?;
            comb.push(dev.dtoh_sync_copy(&out)?.iter().map(|&b| f16::from_bits(b).to_f32()).collect::<Vec<f32>>());
        }
        let (mut num, mut den, mut maxd) = (0f64, 0f64, 0f32);
        for (a, b) in comb[0].iter().zip(&comb[1]) { num += ((a - b) * (a - b)) as f64; den += (a * a) as f64; maxd = maxd.max((a - b).abs()); }
        let rel = (num / den.max(1e-30)).sqrt();
        println!("  dsa nh {nh} kv {kv} m {m}: 64-split vs 8-split combined rel-L2 {rel:.3e}, max |diff| {maxd:.3e}");
        ensure!(rel < 2e-3, "64-split decode geometry drifts from the 8-split one: rel-L2 {rel:.3e}");
    }
    println!("NAIVE_ATTN_PREFILL_OK");
    Ok(())
}

#[cfg(test)]
mod pdl_entry_tests {
    /// Every naive kernel is launched with programmatic stream serialization under `--naive-pdl=1`, so every
    /// entry kernel must begin with `NV_PDL_ENTRY()` (griddepcontrol.wait before it reads a predecessor's output).
    /// 2026-10-20: `nv_idx_score_wide` lacked it and scored stale keys on long prompts.
    #[test]
    fn every_naive_entry_kernel_waits_for_its_predecessor() {
        let src = src_text!("kernels/gpu_naive");
        let mut missing = Vec::new();
        let mut n = 0;
        for (i, _) in src.match_indices("extern \"C\" __global__") {
            let rest = &src[i..];
            let name = rest.split('(').next().unwrap_or("").split_whitespace().last().unwrap_or("").to_string();
            let body = &rest[rest.find('{').unwrap_or(0)..];
            let end = body.find("\n}").unwrap_or(body.len());
            n += 1;
            if !body[..end].contains("NV_PDL_ENTRY()") { missing.push(name); }
        }
        assert!(n > 50, "found only {n} kernels: the scan is broken");
        assert!(missing.is_empty(), "naive kernels without NV_PDL_ENTRY (a PDL race under --naive-pdl=1): {missing:?}");
    }
}
