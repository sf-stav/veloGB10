//! DSV4 Phase-3 spine: production GPU launch infrastructure for `kernels/gpu_dsv4.cu`.
//!
//! # Why a separate launcher
//!
//! cudarc 0.9.15 keeps `CudaFunction.cu_function` private (`pub(crate)`), so
//! `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)` — the opt-in that
//! gather-attention and any future >48 KB-smem kernel needs — CANNOT be applied to a
//! cudarc-loaded function. This module loads `src/ptx/gpu_dsv4.ptx` via the raw driver API
//! (cudarc's own `result::module::load_data`) and holds the raw `CUfunction` handles, so the
//! smem opt-in and every launch are fully under our control.
//!
//! # Stream invariant (AGENTS.md §2.1)
//!
//! Every launch runs on the caller's BLOCKING compute stream — never the NULL/default stream.
//! The caller passes `stream.stream` (a `cudarc::driver::CudaStream`); we never fork or default
//! it. Pool / `alloc_zeros` buffers are NULL-stream-ordered, and the blocking compute stream
//! synchronizes with those in both directions — the load-bearing correctness property this
//! engine's MTP path depends on (see `gpu::fork_blocking_stream`).
//!
//! # Build-ID handshake (AGENTS.md §1)
//!
//! `gpu_dsv4.cu` exposes `dsv4_kernel_build_id` (returns the `-DKERNEL_BUILD_ID` that build.rs
//! bakes in). `Dsv4Kernels::load` asserts the loaded PTX's stamp == `env!("KERNEL_BUILD_ID")`, so
//! a deploy that ships a fresh binary with a stale `gpu_dsv4.ptx` fails loudly — the same
//! protection `gpu_batch` has had since the mixed-deploy crash.

use anyhow::{anyhow, Result};
use cudarc::driver::result;
use cudarc::driver::sys;
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchAsync, LaunchConfig};
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Arc;

/// bf16 device buffer (mirrors `gpu::B`).
pub type B = CudaSlice<half::bf16>;
/// f32 device buffer (mirrors `gpu::S`).
pub type S = CudaSlice<f32>;

/// Raw-loaded `gpu_dsv4.ptx` functions, launched on the caller's blocking compute stream.
///
/// One of these lives for the model's lifetime (load once at construction). The JIT `CUmodule`
/// is owned and unloaded on drop.
pub struct Dsv4Kernels {
    module: sys::CUmodule,
    funcs: HashMap<&'static str, sys::CUfunction>,
    /// The primary context these raw handles belong to. Raw driver calls (unlike cudarc's safe
    /// API) do NOT re-bind the context per call, so `dsv4_launch!` re-binds it — mandatory under
    /// the per-thread test harness, and a cheap no-op on the engine's single-thread serving path.
    ctx: sys::CUcontext,
}

impl Dsv4Kernels {
    /// Load `src/ptx/gpu_dsv4.ptx`, assert the build-id stamp, resolve `names` to raw handles.
    /// `dsv4_kernel_build_id` is resolved implicitly for the handshake (not required in `names`).
    pub fn load(dev: &Arc<CudaDevice>, names: &[&'static str]) -> Result<Self> {
        Self::load_module(dev, "src/ptx/gpu_dsv4.ptx", names)
    }

    /// Load an arbitrary DSV4 PTX module (e.g. the Phase-3 lane modules
    /// `src/ptx/gpu_dsv4_{attn,comp}.ptx`), assert the build-id stamp, resolve `names`.
    pub fn load_module(dev: &Arc<CudaDevice>, ptx_path: &str, names: &[&'static str]) -> Result<Self> {
        let ctx = *dev.cu_primary_ctx();
        // Bind the primary context to THIS thread before any raw driver op (the constructor bound
        // it on the device-creating thread, which may not be this one under the test harness).
        unsafe { result::ctx::set_current(ctx) }.map_err(|e| anyhow!("ctx bind in load: {e}"))?;

        let ptx = std::fs::read_to_string(ptx_path)
            .map_err(|e| anyhow!("read {ptx_path} (run `cargo build` first): {e}"))?;
        let ptx_c = CString::new(ptx).unwrap();
        let module = unsafe { result::module::load_data(ptx_c.as_ptr() as *const _) }
            .map_err(|e| anyhow!("cuModuleLoadData {ptx_path}: {e}"))?;

        let mut funcs: HashMap<&'static str, sys::CUfunction> = HashMap::new();
        // The stamp kernel is always resolved — the loader asserts it below.
        for n in names.iter().copied().chain(std::iter::once("dsv4_kernel_build_id")) {
            let cname = CString::new(n).unwrap();
            let f = unsafe { result::module::get_function(module, cname) }
                .map_err(|e| anyhow!("cuModuleGetFunction {ptx_path}::{n}: {e}"))?;
            funcs.insert(n, f);
        }
        let me = Self { module, funcs, ctx };
        me.assert_build_id(dev)?;
        Ok(me)
    }

    /// Bind this module's primary context to the calling thread. `dsv4_launch!` calls this before
    /// every launch; call it yourself if you use the raw `func()` handle directly off-thread.
    pub fn bind_ctx(&self) -> Result<()> {
        unsafe { result::ctx::set_current(self.ctx) }.map_err(|e| anyhow!("dsv4 ctx bind: {e}"))
    }

    /// Raw `CUfunction` for `name` (copyable — a raw handle). None if not requested at load.
    pub fn func(&self, name: &str) -> Option<sys::CUfunction> {
        self.funcs.get(name).copied()
    }

    /// All loaded (name, CUfunction) pairs — the CUDA-graph node classifier matches
    /// `CUDA_KERNEL_NODE_PARAMS.func` against these to name each captured kernel node.
    pub fn func_handles(&self) -> impl Iterator<Item = (&str, sys::CUfunction)> + '_ {
        self.funcs.iter().map(|(k, v)| (k.as_ref() as &str, *v))
    }

    /// One-time opt-in for dynamic shared memory beyond the 48 KB default. Idempotent; call once
    /// per big-smem kernel right after `load`. The GB10 opt-in cap is ~99 KB
    /// (`gb10-smem-optin-cap-99kb`); `bytes` must be ≤ that.
    pub fn set_dynamic_smem(&self, name: &str, bytes: u32) -> Result<()> {
        let f = self
            .func(name)
            .ok_or_else(|| anyhow!("set_dynamic_smem: kernel {name} not loaded"))?;
        let r = unsafe {
            sys::cuFuncSetAttribute(
                f,
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes as i32,
            )
        };
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuFuncSetAttribute {name} smem={bytes}: {r:?}"));
        }
        Ok(())
    }

    /// Refuse a stale `gpu_dsv4.ptx`. Launches `dsv4_kernel_build_id` on the NULL stream and
    /// reads the stamped value. This is a one-shot load-time check, not a hot-path launch, and
    /// never touches pooled buffers.
    fn assert_build_id(&self, dev: &Arc<CudaDevice>) -> Result<()> {
        let expect = u64::from_str_radix(env!("KERNEL_BUILD_ID"), 16)
            .map_err(|_| anyhow!("KERNEL_BUILD_ID not baked into binary (build.rs)"))?;
        let f = self
            .func("dsv4_kernel_build_id")
            .ok_or_else(|| anyhow!("gpu_dsv4.ptx has no dsv4_kernel_build_id — run `cargo build`"))?;
        let out = dev
            .alloc_zeros::<u64>(1)
            .map_err(|e| anyhow!("build-id probe alloc: {e}"))?;
        // kernel_params[0] must point to the argument value (the device address) — address OF the
        // CUdeviceptr field, not the device address itself.
        let mut params: [*mut std::ffi::c_void; 1] =
            [out.device_ptr() as *const sys::CUdeviceptr as *mut _];
        unsafe {
            result::launch_kernel(f, (1, 1, 1), (1, 1, 1), 0, result::stream::null(), &mut params)
                .map_err(|e| anyhow!("launch dsv4_kernel_build_id: {e}"))?;
        }
        dev.synchronize()
            .map_err(|e| anyhow!("build-id probe sync: {e}"))?;
        let got = dev
            .dtoh_sync_copy(&out)
            .map_err(|e| anyhow!("build-id probe dtoh: {e}"))?[0];
        if got != expect {
            return Err(anyhow!(
                "STALE DSV4 KERNELS: src/ptx/gpu_dsv4.ptx (stamp {got:016x}) != binary \
                 ({expect:016x}). A deploy is THREE files — the binary AND both src/ptx/*.ptx. \
                 Run `cargo build --release` and redeploy src/ptx/gpu_dsv4.ptx."
            ));
        }
        Ok(())
    }
}

impl Drop for Dsv4Kernels {
    fn drop(&mut self) {
        // Best-effort unload; launch errors here are non-fatal at process teardown.
        let _ = unsafe { result::module::unload(self.module) };
    }
}

/// Create the engine's compute stream as a BLOCKING stream (CU_STREAM_DEFAULT). Mirrors
/// `gpu::fork_blocking_stream`; kept here so the spine is self-contained for tests and a future
/// standalone DSV4 model. Keep the two in sync — both encode the AGENTS.md §2.1 invariant.
pub fn blocking_compute_stream(dev: &Arc<CudaDevice>) -> CudaStream {
    use cudarc::driver::result::stream::{create, destroy, StreamKind};
    let mut s = dev.fork_default_stream().expect("fork default stream");
    unsafe {
        // Discard the NonBlocking fork and re-create as CU_STREAM_DEFAULT (blocking): it
        // synchronizes with the NULL stream both ways, which is the whole point.
        let _ = destroy(s.stream);
        s.stream = create(StreamKind::Default).expect("create blocking stream");
    }
    s
}

// -----------------------------------------------------------------------------------------------
// gemm_dsv4_fp8_bsb wrapper — the §C.3-exact FP8 block-scale GEMM (G2-proven kernel), launched on
// the caller's blocking compute stream via cudarc. This is the attention (wq_a/wq_b/wkv/wo_b) and
// indexer (wq_b) projection GEMM for every DSV4 layer. The kernel lives in gpu_batch.ptx (loaded
// into the cudarc `bk` map alongside the MoE kernels); unlike the gpu_dsv4 kernels it needs no
// dynamic-smem opt-in, so it goes through the ergonomic cudarc launch path.
//
// Layout contract (kernel.py / gpu_batch.cu:3305):
//   C   [N, M] bf16 out, column-major (C[n*M + m])
//   Wt  MMA-repacked FP8 weight codes [M, K] (256 B per 16x16 tile)
//   Sb  per-(128-row, 128-K) weight block scales, UE8M0 [M/128, K/128]
//   X   FP8-E4M3 activation codes [N, K] (caller pre-quantizes via dsv4_act_quant_g128)
//   Sa  per-128-K activation scales, UE8M0 [N, K/128]
//   M % 128 == 0, K % 128 == 0, 1 <= N <= 16 (host contract). Cf = optional fp32 accumulator
//   (the TP partial-sum path); null => round to bf16.
// -----------------------------------------------------------------------------------------------

/// Launch `gemm_dsv4_fp8_bsb` at activation width `n` (1..=16). `f` is the cudarc handle from the
/// gpu_batch module. `cf` = `Some` to keep the fp32 accumulator (TP row-parallel partials), `None`
/// to round the output to bf16. Grid: one block per 16-row weight tile.
#[allow(clippy::too_many_arguments)]
pub fn launch_fp8_bsb(
    f: &CudaFunction,
    stream: &CudaStream,
    c: &mut B,
    wt: &CudaSlice<u8>,
    sb: &CudaSlice<u8>,
    x: &CudaSlice<u8>,
    sa: &CudaSlice<u8>,
    m: usize,
    k: usize,
    n: usize,
    cf: Option<&mut S>,
) -> Result<()> {
    debug_assert!(m % 128 == 0 && k % 128 == 0 && (1..=16).contains(&n), "fp8_bsb geometry");
    let cfg = LaunchConfig { grid_dim: ((m / 16) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let mi = m as i32;
    let ki = k as i32;
    let ni = n as i32;
    unsafe {
        if let Some(cf) = cf {
            f.clone().launch_on_stream(stream, cfg, (c, wt, sb, x, sa, mi, ki, ni, cf))?;
        } else {
            f.clone().launch_on_stream(stream, cfg, (c, wt, sb, x, sa, mi, ki, ni, 0u64))?;
        }
    }
    Ok(())
}

/// Pair-tile twin of [`launch_fp8_bsb`] for `gemm_dsv4_fp8_bsb2` (R3A.1 E1b): identical
/// per-element chains, two 16-row tiles per CTA — grid is (m+31)/32.
#[allow(clippy::too_many_arguments)]
pub fn launch_fp8_bsb2(
    f: &CudaFunction,
    stream: &CudaStream,
    c: &mut B,
    wt: &CudaSlice<u8>,
    sb: &CudaSlice<u8>,
    x: &CudaSlice<u8>,
    sa: &CudaSlice<u8>,
    m: usize,
    k: usize,
    n: usize,
    cf: Option<&mut S>,
) -> Result<()> {
    debug_assert!(m % 128 == 0 && k % 128 == 0 && (1..=16).contains(&n), "fp8_bsb geometry");
    let cfg = LaunchConfig { grid_dim: (((m + 31) / 32) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let mi = m as i32;
    let ki = k as i32;
    let ni = n as i32;
    unsafe {
        if let Some(cf) = cf {
            f.clone().launch_on_stream(stream, cfg, (c, wt, sb, x, sa, mi, ki, ni, cf))?;
        } else {
            f.clone().launch_on_stream(stream, cfg, (c, wt, sb, x, sa, mi, ki, ni, 0u64))?;
        }
    }
    Ok(())
}

/// R3A.4 P3: device-side arithmetic positions — `out[i] = start + (i*mul)/div` (i32
/// truncating division, exactly the old host expressions `start+i/nh`, `start+i`,
/// `start+b*ratio`). Replaces the host Vec + htod_sync_copy uploads (each a full
/// cuCtxSynchronize — ~167 syncs per prefill chunk).
pub fn iota_positions<I: Dsv4Buf<i32>>(
    dev: &Arc<CudaDevice>,
    ks: &Dsv4Kernels,
    stream: &CudaStream,
    start: i32,
    mul: i32,
    div: i32,
    n: usize,
) -> Result<I> {
    let out = I::alloc_zeros(dev, stream.stream, n)?;
    let n_i = n as i32;
    crate::dsv4_launch!(ks, "dsv4_iota_b", stream.stream,
        (((n + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
        (&out, &start, &mul, &div, &n_i))?;
    Ok(out)
}

/// One-time option check (cached): true iff the option is set to a non-empty, non-"0" value.
/// Used for debug/A-B arms (e.g. --verify-seq forces the sequential verify path).
pub fn env_flag_once(o: crate::opts::OptId) -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<std::sync::Mutex<std::collections::HashMap<crate::opts::OptId, bool>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut cache = cache.lock().unwrap();
    *cache.entry(o).or_insert_with(|| {
        crate::opts::var(o).map(|v| !v.is_empty() && v != "0").unwrap_or(false)
    })
}

/// Pointer-marshalling for [`dsv4_launch!`]. Each call-site passes its args as REFERENCES to
/// locals (e.g. `&slice`, `&topk_i`, `&scale`) that outlive the launch — `cuLaunchKernel` copies
/// the parameter values synchronously before returning (the same contract cudarc's `DeviceRepr`
/// relies on), so the addresses only need to be valid for the duration of the call.
pub trait Dsv4Arg {
    fn ptr(&self) -> *mut std::ffi::c_void;
}

impl<T> Dsv4Arg for CudaSlice<T> {
    fn ptr(&self) -> *mut std::ffi::c_void {
        // cuLaunchKernel reads kernel_params[i] as a pointer to argument i's VALUE. For a device
        // pointer arg the value is the device address, stored in the slice's CUdeviceptr field —
        // so we hand the address OF that field (a host pointer), not the device address itself.
        self.device_ptr() as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}
/// Blanket over references: lets call sites pass `&slice` / `&i32` uniformly.
impl<T: Dsv4Arg + ?Sized> Dsv4Arg for &T {
    fn ptr(&self) -> *mut std::ffi::c_void {
        (**self).ptr()
    }
}
/// Mutable-reference blanket so in-place kernels can take `&mut slice` for intent.
impl<T: Dsv4Arg + ?Sized> Dsv4Arg for &mut T {
    fn ptr(&self) -> *mut std::ffi::c_void {
        (**self).ptr()
    }
}
impl Dsv4Arg for i32 {
    fn ptr(&self) -> *mut std::ffi::c_void {
        self as *const i32 as *mut _
    }
}
impl Dsv4Arg for u32 {
    fn ptr(&self) -> *mut std::ffi::c_void {
        self as *const u32 as *mut _
    }
}
impl Dsv4Arg for f32 {
    fn ptr(&self) -> *mut std::ffi::c_void {
        self as *const f32 as *mut _
    }
}
impl Dsv4Arg for i64 {
    fn ptr(&self) -> *mut std::ffi::c_void {
        self as *const i64 as *mut _
    }
}
impl Dsv4Arg for u64 {
    fn ptr(&self) -> *mut std::ffi::c_void {
        self as *const u64 as *mut _
    }
}
impl Dsv4Arg for usize {
    fn ptr(&self) -> *mut std::ffi::c_void {
        self as *const usize as *mut _
    }
}

/// A raw device pointer wrapper for `dsv4_launch!` — used to alias a region of another
/// buffer (e.g. the compressor cache aliasing the kv_cache tail, Item 3). The stored
/// `dptr` is a device address; `ptr()` returns the address OF the stored field (matching
/// the CudaSlice contract: cuLaunchKernel reads the device address from that host slot).
pub struct DevPtr {
    pub dptr: sys::CUdeviceptr,
}
impl Dsv4Arg for DevPtr {
    fn ptr(&self) -> *mut std::ffi::c_void {
        &self.dptr as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

// ============================================================================
// GSlice — the CUDA-graph workspace buffer (DSV4_R3A.md §8). cudarc hard-binds every
// alloc/free/memset to the device LEGACY stream (forbidden during graph capture —
// measured: INVALIDATED / STREAM_CAPTURE_IMPLICIT). GSlice issues all three on the
// runtime compute stream (capture-legal): allocs/memsets become graph nodes; DROPS DURING
// CAPTURE intentionally LEAK to the graph pool (freeing graph-owned memory outside the
// graph is the corruption the driver guards against). The shared decode-path functions
// are generic over [`Dsv4Buf`]: prefill/verify instantiate with cudarc `CudaSlice`
// (unchanged legacy behavior, bitwise-identical eager), the graphed decode instantiates
// with `GSlice`. `GB`/`GS` are the bf16/f32 aliases mirroring `B`/`S`.
// ============================================================================

/// True while a graph capture is open on this process (set by the capture driver in
/// dsv4_model.rs). GSlice::drop skips the free while set (the graph pool owns the memory).
pub static GRAPH_CAPTURE_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The DEDICATED mempool every GSlice allocation draws from (lazily created, release
/// threshold = MAX). Why dedicated (measured 2026-07-30): CUDA 13 removed the per-stream
/// pool attribute, and graph-capture alloc nodes re-execute at every launch under
/// AUTO_FREE_ON_LAUNCH — if graph memory came from the device default pool, any eager
/// cudarc allocation between launches could take a freed graph block and the next launch's
/// re-allocs would land elsewhere, leaving the baked kernel-arg pointers stale (the
/// all-129280-mismatch tok-1 failure). A private pool makes eager/default-pool allocs
/// structurally unable to collide with graph blocks. The launch records the pool handle,
/// so replay-time re-allocs use it automatically.
///
/// NOTE (2026-07-30, deeper finding): even with a private pool, AUTO_FREE re-allocations
/// are NOT address-stable for multi-alloc graphs (the pool's free-list order differs from
/// the alloc order) — the baked kernel-arg pointers go stale and the output is garbage
/// (tok-1 all-mismatch with correct launches). THE SOUND PATTERN is the workspace below:
/// NO driver allocations inside capture at all — every transient buffer is a bump slice
/// of a persistent slab (addresses fixed by construction, zero alloc nodes in the graph).
pub fn graph_mempool(dev: &Arc<CudaDevice>) -> sys::CUmemoryPool {
    static POOL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let pool = POOL.get_or_init(|| unsafe {
        let props = sys::CUmemPoolProps {
            allocType: sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
            handleTypes: sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
            location: sys::CUmemLocation {
                type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                id: *dev.cu_device(),
            },
            win32SecurityAttributes: std::ptr::null_mut(),
            reserved: [0u8; 64],
        };
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        let r = sys::cuMemPoolCreate(&mut pool, &props);
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemPoolCreate: {r:?}");
        let threshold: u64 = u64::MAX;
        let r = sys::cuMemPoolSetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            &threshold as *const u64 as *mut std::ffi::c_void,
        );
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "pool threshold: {r:?}");
        pool as usize
    });
    *pool as sys::CUmemoryPool
}

pub struct GSlice<T> {
    dptr: sys::CUdeviceptr,
    len: usize,
    dev: Arc<CudaDevice>,
    stream: sys::CUstream,
    /// false for workspace bump slices (the slab owns the memory — drop is a no-op).
    owned: bool,
    _pd: std::marker::PhantomData<T>,
}

impl<T> GSlice<T> {
    pub fn len(&self) -> usize {
        self.len
    }
    /// Reference to the stored device address — the launch-marshalling contract
    /// (address OF the dptr field, same as CudaSlice's).
    pub fn device_ptr(&self) -> &sys::CUdeviceptr {
        &self.dptr
    }
    /// Raw device address of the first element (memcpy/dtod call sites).
    pub fn dptr(&self) -> sys::CUdeviceptr {
        self.dptr
    }
    /// Sub-range as a launch arg (the cudarc `.slice()` equivalent).
    pub fn view(&self, start: usize, len: usize) -> DevPtr {
        assert!(start + len <= self.len, "GSlice view OOB: {start}+{len} > {}", self.len);
        DevPtr { dptr: self.dptr + (start * std::mem::size_of::<T>()) as u64 }
    }
    /// The GraphAlloc constructor. Workspace-armed (the capture path): a bump slice of the
    /// persistent slab — NO driver allocation (no alloc node in the graph, addresses fixed
    /// by construction), plus a zeroing memset on the stream (a memset node — re-runs per
    /// replay, matching eager alloc_zeros semantics). Fallback: malloc from the dedicated
    /// graph pool (eager GSlice use outside graphs).
    pub fn alloc_on(dev: &Arc<CudaDevice>, stream: sys::CUstream, len: usize) -> Result<Self>
    where
        T: cudarc::driver::ValidAsZeroBits + cudarc::driver::DeviceRepr,
    {
        dev.bind_to_thread()?;
        let bytes = len * std::mem::size_of::<T>();
        if let Some(dptr) = graph_ws_alloc(bytes) {
            // no per-alloc memset: the slab-wide memset at capture start covers every bump
            // region (one node instead of ~150 — value-identical, regions are per-site).
            return Ok(GSlice { dptr, len, dev: dev.clone(), stream, owned: false, _pd: std::marker::PhantomData });
        }
        let pool = graph_mempool(dev);
        let mut dptr: sys::CUdeviceptr = 0;
        let r = unsafe { sys::cuMemAllocFromPoolAsync(&mut dptr, bytes, pool, stream) };
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("GSlice pool-alloc {bytes}: {r:?}"));
        }
        unsafe { result::memset_d8_async(dptr, 0, bytes, stream).map_err(|e| anyhow!("GSlice memset: {e}"))? };
        Ok(GSlice { dptr, len, dev: dev.clone(), stream, owned: true, _pd: std::marker::PhantomData })
    }
    /// dtoh for the rare host readout (graph logits copy, gates) — stream-ordered + sync.
    pub fn dtoh_sync(&self) -> Result<Vec<T>>
    where
        T: cudarc::driver::DeviceRepr + Clone,
    {
        self.dev.bind_to_thread()?;
        let mut out = vec![unsafe { std::mem::zeroed() }; self.len];
        unsafe {
            result::memcpy_dtoh_sync(&mut out, self.dptr).map_err(|e| anyhow!("GSlice dtoh: {e}"))?;
        }
        Ok(out)
    }
}

impl<T> Drop for GSlice<T> {
    fn drop(&mut self) {
        if !self.owned {
            return; // workspace bump slice — the slab owns the memory
        }
        unsafe {
            let _ = result::free_async(self.dptr, self.stream);
        }
    }
}

impl<T> Dsv4Arg for GSlice<T> {
    fn ptr(&self) -> *mut std::ffi::c_void {
        self.device_ptr() as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

// cudarc safe-launch marshalling (the `launch_on_stream` path used by the fp8/MoE kernels) —
// mirrors cudarc's own `DeviceRepr for &CudaSlice` (address OF the dptr field).
unsafe impl<T: cudarc::driver::DeviceRepr> cudarc::driver::DeviceRepr for &GSlice<T> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        &self.dptr as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}
unsafe impl<T: cudarc::driver::DeviceRepr> cudarc::driver::DeviceRepr for &mut GSlice<T> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        &self.dptr as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}
unsafe impl cudarc::driver::DeviceRepr for &DevPtr {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        &self.dptr as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

pub type GB = GSlice<half::bf16>;
pub type GS = GSlice<f32>;

// ============================================================================
// The decode-graph WORKSPACE (the sound capture pattern — see graph_mempool's note).
// A persistent slab allocated eagerly; during capture, GSlice::alloc_on hands out bump
// slices of it (NO driver allocation → no alloc nodes in the graph → nothing to
// re-execute or collide at replay; addresses fixed by construction). The zeroing memsets
// stay as stream ops (memset nodes re-run per replay, re-zeroing the slab regions — same
// as eager). GSlice drops in workspace mode are no-ops (the slab outlives everything).
// ============================================================================

struct GraphWs {
    dptr: sys::CUdeviceptr,
    len: usize,
    offset: usize,
    high_water: usize,
}

static GRAPH_WS: std::sync::Mutex<Option<GraphWs>> = std::sync::Mutex::new(None);

/// Allocate the slab (eager, at DecodeGraphs init). Idempotent: a second init (e.g. the
/// DSpark drafter graph after the trunk's) keeps the FIRST slab — re-allocating would
/// orphan baked kernel-arg pointers of already-captured graphs.
pub fn graph_ws_init(dev: &Arc<CudaDevice>, bytes: usize) -> Result<()> {
    {
        let g = GRAPH_WS.lock().unwrap();
        if let Some(ws) = g.as_ref() {
            anyhow::ensure!(ws.len >= bytes, "graph ws re-init requests {bytes} > slab {}", ws.len);
            return Ok(());
        }
    }
    let slab = unsafe { result::malloc_async(std::ptr::null_mut(), bytes).map_err(|e| anyhow!("graph ws slab alloc: {e}"))? };
    // NOTE: the slab is allocated on the device LEGACY stream once, eagerly, BEFORE any
    // capture — never freed (process lifetime), so no legacy-stream ops at replay.
    let mut g = GRAPH_WS.lock().unwrap();
    *g = Some(GraphWs { dptr: slab, len: bytes, offset: 0, high_water: 0 });
    let _ = dev;
    Ok(())
}

/// Reset the bump to 0 (called at the START of each variant's capture — every variant's
/// forward replays the same alloc sequence, so variants share slab regions).
pub fn graph_ws_begin_capture() {
    let mut g = GRAPH_WS.lock().unwrap();
    let ws = g.as_mut().expect("graph_ws_begin_capture before init");
    ws.high_water = ws.high_water.max(ws.offset);
    ws.offset = 0;
}

/// High-water bytes across all captures so far (for the sizing report).
pub fn graph_ws_high_water() -> usize {
    let g = GRAPH_WS.lock().unwrap();
    g.as_ref().map(|ws| ws.high_water.max(ws.offset)).unwrap_or(0)
}

/// The slab's (base, high-water) — for the single capture-start memset.
pub fn graph_ws_span() -> (sys::CUdeviceptr, usize) {
    let g = GRAPH_WS.lock().unwrap();
    let ws = g.as_ref().expect("graph_ws_span before init");
    (ws.dptr, ws.high_water.max(ws.len))
}

fn graph_ws_alloc(bytes: usize) -> Option<sys::CUdeviceptr> {
    let mut g = GRAPH_WS.lock().unwrap();
    let ws = g.as_mut()?;
    let aligned = (bytes + 255) & !255;
    if ws.offset + aligned > ws.len {
        panic!("graph workspace overflow: need {} > slab {} (bump the slab size)",
               ws.offset + aligned, ws.len);
    }
    let d = ws.dptr + ws.offset as u64;
    ws.offset += aligned;
    Some(d)
}

/// The buffer abstraction the decode-path shared functions are generic over (see GSlice).
/// Type inference keeps every pre-GSlice call site unchanged: prefill/verify/eager locals
/// are `CudaSlice` (the `Dsv4Buf` impl preserves the legacy-stream cudarc alloc EXACTLY);
/// the graphed decode's locals are `GSlice` (compute-stream allocs — capture-legal).
pub trait Dsv4Buf<T>: Dsv4Arg + Sized {
    fn alloc_zeros(dev: &Arc<CudaDevice>, stream: sys::CUstream, len: usize) -> Result<Self>;
    fn len(&self) -> usize;
    fn dptr(&self) -> sys::CUdeviceptr;
    fn view(&self, start: usize, len: usize) -> DevPtr;
}

impl<T: cudarc::driver::ValidAsZeroBits + cudarc::driver::DeviceRepr> Dsv4Buf<T> for CudaSlice<T> {
    fn alloc_zeros(dev: &Arc<CudaDevice>, _stream: sys::CUstream, len: usize) -> Result<Self> {
        // LEGACY-stream cudarc alloc — the pre-GSlice behavior, preserved exactly for the
        // prefill/verify/eager instantiations (the stream arg is deliberately unused).
        dev.alloc_zeros::<T>(len).map_err(|e| anyhow!("alloc_zeros: {e:?}"))
    }
    fn len(&self) -> usize {
        // cudarc's len() is the DeviceSlice trait method (no inherent one) — disambiguate
        // fully or this resolves to Dsv4Buf::len itself (unconditional recursion).
        cudarc::driver::DeviceSlice::len(self)
    }
    fn dptr(&self) -> sys::CUdeviceptr {
        *self.device_ptr()
    }
    fn view(&self, start: usize, len: usize) -> DevPtr {
        assert!(start + len <= self.len(), "CudaSlice view OOB: {start}+{len} > {}", self.len());
        DevPtr { dptr: *self.device_ptr() + (start * std::mem::size_of::<T>()) as u64 }
    }
}

impl<T: cudarc::driver::ValidAsZeroBits + cudarc::driver::DeviceRepr> Dsv4Buf<T> for GSlice<T> {
    fn alloc_zeros(dev: &Arc<CudaDevice>, stream: sys::CUstream, len: usize) -> Result<Self> {
        GSlice::alloc_on(dev, stream, len)
    }
    fn len(&self) -> usize {
        self.len
    }
    fn dptr(&self) -> sys::CUdeviceptr {
        self.dptr
    }
    fn view(&self, start: usize, len: usize) -> DevPtr {
        self.view(start, len)
    }
}

/// Launch a `gpu_dsv4` kernel by name on a BLOCKING compute stream.
///
/// Call sites pass references to their locals — every arg is `&device_slice` or `&scalar`. The
/// macro collects raw pointers and hands them to `cuLaunchKernel`; CUDA copies the parameter
/// values synchronously before the call returns (so the locals only need to outlive the macro
/// invocation, which they do by construction).
///
/// ```ignore
/// // topk: small-smem kernel, launched on the compute stream.
/// let (rows_i, t_i, k_i) = (rows as i32, t as i32, k as i32);
/// dsv4_launch!(ks, "dsv4_topk", stream.stream, (rows as u32,1,1), (256,1,1), 0,
///     (&scores_dev, &mut out_dev, &rows_i, &t_i, &k_i))?;
/// // gather_attn: big-smem (set once via ks.set_dynamic_smem("dsv4_gather_attn", 88320)? first).
/// dsv4_launch!(ks, "dsv4_gather_attn", stream.stream, (m as u32,b as u32,4), (256,1,1), 88320,
///     (&q_dev, &kv_dev, &mut o_dev, &sink_dev, &idx_dev, &topk_i, &n_i, &scale))?;
/// ```
#[macro_export]
macro_rules! dsv4_launch {
    ($ks:expr, $name:expr, $stream:expr, $grid:expr, $block:expr, $smem:expr, ($($arg:expr),+ $(,)?)) => {{
        let __ks = &$ks;
        // Raw cuLaunchKernel needs the context current on THIS thread (cudarc's safe launch
        // re-binds per call; we use the raw path, so re-bind here). Cheap no-op when already bound.
        __ks.bind_ctx().unwrap_or_else(|e| panic!("dsv4 ctx bind {}: {}", $name, e));
        let __f = __ks
            .func($name)
            .unwrap_or_else(|| panic!("dsv4 kernel not loaded: {}", $name));
        let __stream: cudarc::driver::sys::CUstream = $stream;
        let (gx, gy, gz) = $grid;
        let (bx, by, bz) = $block;
        // Every $arg is a reference to a caller local; collect their addresses.
        let mut __params: Vec<*mut std::ffi::c_void> = Vec::new();
        $( __params.push($crate::dsv4_gpu::Dsv4Arg::ptr(&$arg)); )+
        unsafe {
            cudarc::driver::result::launch_kernel(
                __f,
                (gx, gy, gz),
                (bx, by, bz),
                $smem,
                __stream,
                &mut __params,
            )
            .map_err(|e| anyhow::anyhow!("dsv4 launch {}: {}", $name, e))
        }
    }};
}
