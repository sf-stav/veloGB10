//! E9 (TP barrier-latency overlap): raw driver handles and the lazily loaded second module used by the `blaunch_e9!` launches,
//! plus the E11 raw kernels. Numerics are untouched (read-only prefetch + ordering). Map: `mod.rs`.

// -----------------------------------------------------------------------------------------------
// E9 — barrier-latency overlap (HY3_AS_FAST_AS_POSSIBLE.md E9 / HY3_MAX_SPEED_CONTEXT_PLAN.md):
// hide the TP all-reduce barrier's cpu_done wait behind the CONSUMER GEMM's weight streaming.
//
// Decode posts ~160 barriers/token at ~31 us each vs the ~8.6 us doorbell floor — the excess is
// GPU time parked in K2 (`tp_wait_add`) spinning for the peer's partial while 47/48 SMs sit idle.
// E9 turns that idle time into weight prefetch: K2 publishes a launch-completion edge BEFORE its
// spin (`griddepcontrol.launch_dependents` in the kernel); the elementwise kernels between K2 and
// the next layer's qkv/GDN projection GEMM are launched with the programmatic stream-serialization
// attribute (`blaunch_e9!`) so their grids become resident while K2 still spins; and the GEMM
// itself launches with the attribute + `pdl=1` (`launch_gemm_fp4_pdl`), streaming its weight bytes
// into L2 BEFORE `griddepcontrol.wait` gates the reduced activation. Numerics are untouched
// (read-only prefetch + ordering); the barrier protocol (epochs, payloads, rank0+rank1 add,
// agree()) is untouched; SPMD is untouched — both ranks resolve the same e9_fold from
// env → shipped TpConfig → default (AGENTS.md §5a).
//
// Escape: --e9-no-fold=1 (or the head shipping `e9_fold=false`) — no launch carries the
// attribute and `pdl=0` skips the GEMM preamble; the kernels' two programmatic instructions are
// then no-ops (the implicit stream serialization already satisfied the dependency), so the plain
// barrier path is restored behaviorally byte-for-byte.
//
// cudarc's LaunchConfig has no launch-attribute support and CudaFunction keeps its raw handle
// private, so the E9 launches go through cuLaunchKernelEx on a SECOND module loaded from the same
// PTX (the plain path stays on cudarc's module). Loaded once per process, lazily on the first E9
// launch (TP=2 + fold only; single-node and the escape never load it).
// -----------------------------------------------------------------------------------------------

/// E9 flag resolution — env-first, then the shipped TpConfig, then ON (the default).
pub(super) fn e9_fold_enabled() -> bool {
    if crate::opts::var(crate::opt!("e9-no-fold")).is_ok() { return false; }
    crate::tp::tp_config().map(|c| c.e9_fold).unwrap_or(true)
}

/// Raw driver handles for the E9 programmatic-launch sites (see the block comment above).
pub(super) struct E9Kernels {
    _module: cudarc::driver::sys::CUmodule,
    gemm_mma_fp4_b: cudarc::driver::sys::CUfunction,
    fused_res_rmsnorm_b: cudarc::driver::sys::CUfunction,
    rmsnorm_b: cudarc::driver::sys::CUfunction,
    add_residual_b: cudarc::driver::sys::CUfunction,
}

// Raw driver handles are context-bound to the one process-wide device and safe to launch from any
// thread (same reasoning as `unsafe impl Send for TpLink`).
unsafe impl Send for E9Kernels {}
unsafe impl Sync for E9Kernels {}

/// E11: raw driver handles for ANY batch-module kernel by name (used by `launch_raw` — the
/// 13-arg fold-pdl launches that exceed the `launch_on_stream` tuple cap — and the probe's pss
/// launches on single-node). Same module-load discipline as the E9 loader.
pub(super) struct E11RawKernels {
    _module: cudarc::driver::sys::CUmodule,
    pub(super) by_name: std::collections::HashMap<String, cudarc::driver::sys::CUfunction>,
}
unsafe impl Send for E11RawKernels {}
unsafe impl Sync for E11RawKernels {}

static E11_KERNELS: std::sync::OnceLock<Option<E11RawKernels>> = std::sync::OnceLock::new();

pub(super) fn e11_raw_kernels() -> Option<&'static E11RawKernels> {
    E11_KERNELS.get_or_init(|| {
        use cudarc::driver::sys;
        let ptx = crate::ptx_embed::GPU_BATCH;
        let ptx_c = std::ffi::CString::new(ptx).ok()?;
        let mut module: sys::CUmodule = std::ptr::null_mut();
        let r = unsafe {
            sys::cuModuleLoadDataEx(&mut module, ptx_c.as_ptr() as *const std::ffi::c_void,
                                    0, std::ptr::null_mut(), std::ptr::null_mut())
        };
        if r != sys::CUresult::CUDA_SUCCESS {
            eprintln!("[e11] raw module load failed ({r:?}) — pdl/13-arg launches unavailable");
            return None;
        }
        // Fetch the whole kernel list (mirrors the bfnames registration; unknown names → None).
        let mut by_name = std::collections::HashMap::new();
        for name in ["gemm_moe_mma_fp4_fold_pdl", "gemm_moe_mma_fp4_fold", "gemm_moe_mma_fp4_fold_u4",
                     "gemm_moe_mma_fp4_fold_x2", "gemm_moe_mma_fp4_fold_rast",
                     "gemm_moe_mma_fp4_fold_lb5", "gemm_moe_mma_fp4_fold_lb4"] {
            let cname = std::ffi::CString::new(name).unwrap();
            let mut f: sys::CUfunction = std::ptr::null_mut();
            let r = unsafe { sys::cuModuleGetFunction(&mut f, module, cname.as_ptr()) };
            if r == sys::CUresult::CUDA_SUCCESS { by_name.insert(name.to_string(), f); }
        }
        Some(E11RawKernels { _module: module, by_name })
    }).as_ref()
}

impl E9Kernels {
    pub(super) fn get(&self, name: &str) -> Option<cudarc::driver::sys::CUfunction> {
        match name {
            "gemm_mma_fp4_b" => Some(self.gemm_mma_fp4_b),
            "fused_res_rmsnorm_b" => Some(self.fused_res_rmsnorm_b),
            "rmsnorm_b" => Some(self.rmsnorm_b),
            "add_residual_b" => Some(self.add_residual_b),
            _ => None,
        }
    }
}

pub(super) static E9_KERNELS: std::sync::OnceLock<Option<E9Kernels>> = std::sync::OnceLock::new();

pub(super) fn e9_load_kernels() -> Option<E9Kernels> {
    use cudarc::driver::sys;
    let ptx = crate::ptx_embed::GPU_BATCH;
    // cuModuleLoadDataEx requires a NULL-TERMINATED PTX string. A Rust String's buffer is NOT
    // null-terminated (the driver reads past it into garbage → CUDA_ERROR_INVALID_PTX, found
    // 2026-08-11: the raw as_ptr() call failed where cudarc's own load_ptx — which wraps the
    // source in CString — succeeded on the SAME bytes). Same fix as cudarc's PtxKind::Src path.
    let ptx_c = std::ffi::CString::new(ptx).ok()?;
    let mut module: sys::CUmodule = std::ptr::null_mut();
    let r = unsafe {
        sys::cuModuleLoadDataEx(&mut module, ptx_c.as_ptr() as *const std::ffi::c_void,
                                0, std::ptr::null_mut(), std::ptr::null_mut())
    };
    if r != sys::CUresult::CUDA_SUCCESS {
        eprintln!("[e9] E9 module load failed ({r:?}) — plain barrier path (no overlap)");
        return None;
    }
    let getf = |name: &str| {
        let cname = std::ffi::CString::new(name).unwrap();
        let mut f: sys::CUfunction = std::ptr::null_mut();
        let r = unsafe { sys::cuModuleGetFunction(&mut f, module, cname.as_ptr()) };
        if r == sys::CUresult::CUDA_SUCCESS { Some(f) } else { None }
    };
    let ks = E9Kernels {
        _module: module,
        gemm_mma_fp4_b: getf("gemm_mma_fp4_b")?,
        fused_res_rmsnorm_b: getf("fused_res_rmsnorm_b")?,
        rmsnorm_b: getf("rmsnorm_b")?,
        add_residual_b: getf("add_residual_b")?,
    };
    eprintln!("[e9] programmatic dependent launch ON — K2 cpu_done spin overlapped with the \
               qkv/GDN weight prefetch (--e9-no-fold=1 to restore the plain path)");
    Some(ks)
}
