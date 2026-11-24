//! DSV4.1 GPU host side: the `gpu_dsv41` kernel module (embedded PTX, kernels/gpu_dsv41.cu) — its
//! loader with the KERNEL_BUILD_ID handshake, the raw-module handles for PDL / smem opt-in
//! launches, and the per-leg host files. I0 owns this file and `abi.rs`; each kernel leg owns its
//! own file below and lists its entry kernels in that file's `FNS` (this loader collects them, so
//! no leg edits this file to add a kernel).
//!
//! | file | leg | kernels (band in kernels/gpu_dsv41/) |
//! |---|---|---|
//! | `abi.rs` | I0 | 01 ABI probe |
//! | `crate::dsv41::weights::exl3` | KE | 10 trellis / plain linears |
//! | `crate::dsv41::weights::native::dense` | KN1 | 20 FP8 dense |
//! | `crate::dsv41::weights::native::moe` | KN2 | 30 MXFP4 experts |
//! | `kv.rs` | KC | 40 compressor + cache writes |
//! | `indexer.rs` | KI | 50 indexer / top-k / candidates |
//! | `attn.rs` | KA | 60 sparse attention |
//! | `mixers.rs` | KM | 70 mHC / norms / router / Engram gate |
//! | `engram.rs` | EIO | 80 Engram row decode |
//! | `dspark.rs` | DS | 90 DSpark round glue |
//! | `crate::dsv41::fwd::glue` | SL | 05 per-layer forward glue |

pub mod abi;
pub mod attn;
pub mod dspark;
pub mod engram;
pub mod indexer;
pub mod kv;
pub mod mixers;

use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

pub const MODULE: &str = "gpu_dsv41";

/// Every entry kernel of the module (I0's plus each leg's `FNS`).
pub fn all_fns() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = abi::FNS.to_vec();
    for leg in [crate::dsv41::weights::exl3::FNS, crate::dsv41::weights::native::dense::FNS, crate::dsv41::weights::native::moe::FNS,
                kv::FNS, indexer::FNS, attn::FNS, mixers::FNS, engram::FNS, dspark::FNS, crate::dsv41::fwd::FNS,
                crate::dsv41::vision::tower::FNS] {
        v.extend_from_slice(leg);
    }
    v
}

/// Load the embedded module once per device (idempotent) and refuse a binary / PTX build-id mismatch.
pub fn load_module(dev: &Arc<CudaDevice>) -> Result<()> {
    if dev.get_func(MODULE, "d41_kernel_build_id").is_none() {
        let fns = all_fns();
        let mut seen = std::collections::HashSet::new();
        for f in &fns { ensure!(seen.insert(*f), "gpu_dsv41: kernel {f} listed twice in the legs' FNS"); }
        dev.load_ptx(Ptx::from_src(crate::ptx_embed::GPU_DSV41), MODULE, &fns).context("load the gpu_dsv41 module")?;
    }
    let f = dev.get_func(MODULE, "d41_kernel_build_id").context("d41_kernel_build_id missing")?;
    let mut out = dev.alloc_zeros::<u64>(1)?;
    unsafe { f.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 }, (&mut out,)) }?;
    let got = dev.dtoh_sync_copy(&out)?[0];
    let want = u64::from_str_radix(env!("KERNEL_BUILD_ID"), 16).unwrap_or(0);
    ensure!(got == want, "gpu_dsv41 kernel build id {got:016x} != binary {want:016x} (stale PTX)");
    Ok(())
}

/// `--dsv41-pdl on|off` (default on, leg PF1): every dsv41 forward launch carries the programmatic-stream-serialization
/// attribute, so a kernel's prologue (and its constant-weight prefetch) overlaps its predecessor's tail. Safe because every
/// gpu_dsv41 entry executes `griddepcontrol.wait` before it reads a predecessor's output or writes memory a predecessor
/// reads (00_abi.cuh rule 4). Off = plain stream serialization (the pre-PF1 launches; an A/B and escape switch).
/// Leg PP1's prefill switches (`--dsv41-pf-router-wide`, `--dsv41-pf-gemv-rows`, `--dsv41-pf-engram-ahead`; default on, same bits).
pub fn pf_on(flag: &'static str) -> bool {
    crate::opts::lookup(flag).and_then(|id| crate::opts::get(id)).map_or(true, |v| crate::opts::is_on_value(&v))
}
/// Leg PP1's engagement receipts (`/metrics`): prefill router calls served by the wide entry, plain GEMV calls split along the rows.
pub static PF_ROUTER_WIDE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Leg P4's engagement receipt: plain GEMV calls of narrow weights (<= 256 columns) at 1..=32 rows served by `d41x_plain_gemv_w`.
pub static PF_GEMV_SMALL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PF_GEMV_ROWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn pdl_on() -> bool { crate::opts::get(crate::opt!("dsv41-pdl")).map_or(true, |v| crate::opts::is_on_value(&v)) }

/// Launches built with the PDL attribute (engagement receipt; a graph replay re-runs captured launches without counting).
static PDL_LAUNCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn pdl_launches() -> u64 { PDL_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed) }

/// Launch entry `name` of the raw module on `stream` (`cuLaunchKernelEx`), with the PDL attribute when `pdl`. `params`
/// = the kernel's parameter slots (pointers to each argument, as `cuLaunchKernel` takes them).
pub fn launch_raw(name: &str, grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32, stream: u64, pdl: bool,
                  params: &mut [*mut std::ffi::c_void]) -> Result<()> {
    use cudarc::driver::sys;
    ensure!(stream != 0, "dsv41 kernels never run on the legacy default stream (AGENTS 2.1): {name}");
    ensure!(grid.0 > 0 && grid.1 > 0 && grid.2 > 0, "{name}: empty grid {grid:?}");
    let f = raw_fn(name).with_context(|| format!("gpu_dsv41 entry {name} is not in the module"))?;
    let mut attr = sys::CUlaunchAttribute {
        id: sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
        pad: [0; 4],
        value: sys::CUlaunchAttributeValue { programmaticStreamSerializationAllowed: 1 },
    };
    let mut cfg = sys::CUlaunchConfig {
        gridDimX: grid.0, gridDimY: grid.1, gridDimZ: grid.2, blockDimX: block.0, blockDimY: block.1, blockDimZ: block.2,
        sharedMemBytes: smem, hStream: stream as sys::CUstream, attrs: &mut attr, numAttrs: u32::from(pdl),
    };
    let r = unsafe { sys::cuLaunchKernelEx(&mut cfg, f, params.as_mut_ptr(), std::ptr::null_mut()) };
    ensure!(r == sys::CUresult::CUDA_SUCCESS, "launch {name} {grid:?}: {r:?}");
    if pdl { PDL_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
    Ok(())
}

/// `launch_raw` for an entry taking ONE by-value argument struct (the KC / KM / SL / TP ABI), PDL per `pdl_on()`.
pub fn launch_args<A>(name: &str, grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32, stream: u64, a: &A) -> Result<()> {
    let mut p = [a as *const A as *mut std::ffi::c_void];
    launch_raw(name, grid, block, smem, stream, pdl_on(), &mut p)
}

/// The RAW CUfunction of an entry, from a second load of the same embedded PTX through the driver
/// (cudarc's copy hides its handle). PDL launches (`cuLaunchKernelEx`) and the > 48 KB dynamic-smem
/// opt-in must use THIS handle: a function attribute set on one module copy never reaches the
/// other (AGENTS §4, the 2026-10-03 lesson).
pub fn raw_fn(name: &str) -> Option<cudarc::driver::sys::CUfunction> {
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
        let ptx_c = std::ffi::CString::new(crate::ptx_embed::GPU_DSV41).ok()?;
        let mut m: sys::CUmodule = std::ptr::null_mut();
        let r = unsafe { sys::cuModuleLoadDataEx(&mut m, ptx_c.as_ptr() as *const std::ffi::c_void, 0, std::ptr::null_mut(), std::ptr::null_mut()) };
        if r != sys::CUresult::CUDA_SUCCESS { crate::boot_report::warn(&format!("gpu_dsv41: raw module load failed ({r:?})")); return None; }
        Some(H(m))
    }).as_ref()?.0;
    let cname = std::ffi::CString::new(name).ok()?;
    let mut f: sys::CUfunction = std::ptr::null_mut();
    let r = unsafe { sys::cuModuleGetFunction(&mut f, module, cname.as_ptr()) };
    if r != sys::CUresult::CUDA_SUCCESS { return None; }
    CACHE.lock().ok()?.push((name.to_string(), FnH(f)));
    Some(f)
}
