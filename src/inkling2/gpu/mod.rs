//! Inkling-2 GPU side: the `gpu_inkling2` kernel module (kernels/gpu_inkling2.cu, sm_121a), loaded raw
//! (one `CUmodule`, raw `CUfunction` handles: the dynamic-smem opt-in lives on the handle that launches),
//! the build-id handshake, and a small launch-argument builder. Kernel bands: see the .cu header.
//!
//!   * [`k1`]       S2a (leg K1) — projection GEMM, attention, short convs, AttnRes, MessageGate, norms:
//!                  launch wrappers + host-side weight layouts.
//!   * [`k1_probe`] `--probe-inkling2-kernels`: every K1 kernel vs the cpu.rs oracle on real tensors.
//!   * [`layout`]   S2b (leg K2) — load-time weight permutations (NVFP4 F128, MXFP8 F64, bf16 B32, gate|up tiles).
//!   * [`moe`]      S2b — router, routed experts (NVFP4 / MXFP8, decode M <= 16 and device-routed prefill), combine.
//!   * [`head`]     S2b — the LM head (bf16 unembed, fp32 logits, softcap 30, fused argmax partials).
//!   * [`probe`]    `--probe-inkling2-moe`: every K2 kernel vs the CPU reference on real checkpoint tensors.
//!
//! ONE module load per process ([`Ink2Kernels::global`]); the K2 band launches through the same handles via
//! [`ink2_launch!`](crate::ink2_launch) (every pointer argument a raw device address `u64`).

pub mod k1;
pub mod k1_probe;
pub mod layout;
pub mod moe;
pub mod head;
pub mod probe;

use anyhow::{anyhow, ensure, Result};
use cudarc::driver::{result, sys, CudaDevice, DevicePtr};
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::sync::{Arc, OnceLock};

static GLOBAL: OnceLock<Ink2Kernels> = OnceLock::new();

/// Every entry the module exports (all bands); a missing one fails the load.
pub const FNS: &[&str] = &[
    "ink2_kernel_build_id",
    // K1
    "ink_proj_dec_ks1", "ink_proj_dec_ks2", "ink_proj_dec_ks4", "ink_proj_dec_ks8", "ink_proj_wide",
    "ink_attn_prep", "ink_attn", "ink_attn_tc", "ink_attn_combine",
    "ink_sconv", "ink_ring_store",
    "ink_attnres", "ink_attnres_cl", "ink_attnres_cl_norm",
    "ink_rms2_cat_bf16", "ink_msggate_out",
    "ink_rmsnorm", "ink_rmsnorm_x3", "ink_embed_norm",
    // K2
    "ink2_router_logits", "ink2_router_topk",
    "ink2_nv4_gu_dec", "ink2_nv4_dn_dec", "ink2_mx8_gu_dec", "ink2_mx8_dn_dec",
    "ink2_moe_pf_route", "ink2_nv4_gu_pf", "ink2_nv4_dn_pf", "ink2_mx8_gu_pf", "ink2_mx8_dn_pf",
    "ink2_moe_combine", "ink2_lm_head", "ink2_head_argmax",
    // S3 glue
    "ink3_gumbel_part", "ink3_gumbel_pick",
    // S4 lane-batched decode (device-side positions)
    "ink_attn_prep_lanes", "ink_attn_lanes", "ink_attn_combine_lanes", "ink_sconv_lanes", "ink_proj_tile", "ink_proj_tile2", "ink_proj_tile3", "ink_msggate_out_m1", "ink_msggate_out_m4", "ink_attn_tc2",
    // S7 verify widths
    "ink2_nv4_gu_vt", "ink2_nv4_dn_vt", "ink2_mx8_gu_vt", "ink2_mx8_dn_vt", "ink2_lm_head_g",
    "ink_attn_tc2_sp", "ink_attn_lanes_mr", "ink_rms2_cat_bf16_x3", "ink_msg_down_x3", "ink_msggate_out_x3", "ink_msggate_out_m1_x3", "ink_msggate_out_m4_x3",
    "ink_proj_dec3_ks1", "ink_proj_dec3_ks2", "ink_proj_dec3_ks4", "ink_proj_dec3_ks8",
    // S6 drafter + accept
    "ink6_tap_store", "ink6_mix_norm", "ink6_ctx_kv", "ink6_block_init", "ink6_qk_rope", "ink6_dattn", "ink6_dattn_comb",
    "ink6_add_norm", "ink6_silu_mul", "ink6_rnn_in", "ink6_rnn_gates", "ink6_fold", "ink6_lse", "ink6_lse_fin",
    "ink6_accept", "ink6_final_part", "ink6_fold_pack", "ink6_head32", "ink6_fold4",
    // Q1 (actq): native expert activations (TP1 + TP2/TP4/TP8 inner widths)
    "ink2q_nv4_quant_f2304", "ink2q_nv4_quant_b2304", "ink2q_nv4_quant_b1152",
    "ink2q_nv4_quant_b640", "ink2q_nv4_quant_b576", "ink2q_nv4_quant_b512", "ink2q_nv4_quant_b384", "ink2q_nv4_quant_b320", "ink2q_nv4_quant_b256", "ink2q_nv4_quant_b192", "ink2q_nv4_quant_b128",
    "ink2q_mx8_quant_f2304", "ink2q_mx8_quant_b2304", "ink2q_mx8_quant_b1152",
    "ink2q_mx8_quant_b640", "ink2q_mx8_quant_b576", "ink2q_mx8_quant_b512", "ink2q_mx8_quant_b384", "ink2q_mx8_quant_b320", "ink2q_mx8_quant_b256", "ink2q_mx8_quant_b192", "ink2q_mx8_quant_b128",
    "ink2q_gu_epi_test", "ink2q_gamma", "ink2q_gamma_pf", "ink2q_nv4_gu_dec", "ink2q_nv4_dn_dec",
    "ink2q_mx8_gu_dec", "ink2q_mx8_dn_dec", "ink2q_nv4_gu_pf", "ink2q_nv4_dn_pf", "ink2q_mx8_gu_pf", "ink2q_mx8_dn_pf",
    "ink2q_nv4_gu_pf128", "ink2q_nv4_dn_pf128", "ink2q_mx8_gu_pf128", "ink2q_mx8_dn_pf128",
];

/// The loaded module. Launches go to the device's cudarc stream (blocking, AGENTS §2.1).
pub struct Ink2Kernels {
    pub dev: Arc<CudaDevice>,
    funcs: HashMap<&'static str, sys::CUfunction>,
    ctx: sys::CUcontext,
    /// launch stream (default: the device's NULL stream, blocking); swapped by `with_stream` (graph timing)
    pub stream: std::cell::Cell<sys::CUstream>,
    /// S7: launch with programmatic dependent launch (PDL, the programmatic stream-serialization attribute): every
    /// entry of the module opens with INK2_PDL_ENTRY (griddepcontrol.wait before its first global access), so a
    /// kernel's CTAs may be scheduled while its predecessor drains; outputs are unchanged (same arithmetic, same
    /// order). Default on; the probe switches it off for its before / after timing.
    pub pdl: std::cell::Cell<bool>,
}
unsafe impl Send for Ink2Kernels {}
unsafe impl Sync for Ink2Kernels {}

/// Kernel parameter list: each value is copied into its own buffer; `ptrs()` gives the driver its
/// `void**` (the driver copies the bytes at launch, so alignment of the host copy does not matter).
#[derive(Default)]
pub struct KArgs { bufs: Vec<Box<[u64]>> }
impl KArgs {
    pub fn new() -> Self { Self::default() }
    pub fn v<T: Copy>(&mut self, v: T) -> &mut Self {
        let n = std::mem::size_of::<T>().div_ceil(8).max(1);
        let mut b = vec![0u64; n].into_boxed_slice();
        unsafe { std::ptr::copy_nonoverlapping(&v as *const T as *const u8, b.as_mut_ptr() as *mut u8, std::mem::size_of::<T>()) };
        self.bufs.push(b);
        self
    }
    fn ptrs(&mut self) -> Vec<*mut c_void> { self.bufs.iter_mut().map(|b| b.as_mut_ptr() as *mut c_void).collect() }
}

/// Device address of a cudarc slice (for kernel arguments and argument structs).
pub fn dp<T>(s: &cudarc::driver::CudaSlice<T>) -> u64 { *s.device_ptr() }

impl Ink2Kernels {
    pub fn load(dev: &Arc<CudaDevice>) -> Result<Self> {
        let ctx = *dev.cu_primary_ctx();
        unsafe { result::ctx::set_current(ctx) }.map_err(|e| anyhow!("ctx bind: {e}"))?;
        let ptx = CString::new(crate::ptx_embed::GPU_INKLING2).unwrap();
        let module = unsafe { result::module::load_data(ptx.as_ptr() as *const _) }
            .map_err(|e| anyhow!("cuModuleLoadData gpu_inkling2: {e}"))?;
        let mut funcs = HashMap::new();
        for &n in FNS.iter().chain(crate::inkling2::vision::gpu::FNS) {
            let f = unsafe { result::module::get_function(module, CString::new(n).unwrap()) }
                .map_err(|e| anyhow!("gpu_inkling2::{n} missing: {e}"))?;
            funcs.insert(n, f);
        }
        // S-INK2-TP2: the tensor-parallel copies (bands d0-df)
        for n in crate::inkling2::tp::kern::fns() {
            let f = unsafe { result::module::get_function(module, CString::new(n.as_str()).unwrap()) }
                .map_err(|e| anyhow!("gpu_inkling2::{n} missing: {e}"))?;
            funcs.insert(Box::leak(n.into_boxed_str()), f);
        }
        let me = Ink2Kernels { dev: dev.clone(), funcs, ctx, stream: std::cell::Cell::new(*dev.cu_stream()), pdl: std::cell::Cell::new(true) };
        // big-smem entries opt in once (GB10 cap 101,376 B per block)
        me.smem_optin("ink_attn", k1::ATTN_SMEM_MAX as i32)?;
        me.smem_optin("ink_attn_tc", k1::ATTN_TC_SMEM as i32)?;
        me.smem_optin("ink_attn_tc2", k1::ATTN_TC2_SMEM as i32)?;
        me.smem_optin("ink_attn_tc2_sp", k1::ATTN_TC2_SP_SMEM as i32)?;
        me.smem_optin("ink_attn_lanes_mr", k1::ATTN_SMEM_MAX as i32)?;
        me.smem_optin("ink_proj_tile", k1::proj_tile_smem(4) as i32)?;
        me.smem_optin("ink_proj_tile3", k1::proj_tile_smem(3) as i32)?;
        // ink_proj_tile2 (the default, 2 stages): 36 KB, under the 48 KB default
        for (n, b) in crate::inkling2::tp::kern::smem_optins() { me.smem_optin(&n, b)?; }
        let out = dev.htod_sync_copy(&[0u64])?;
        me.launch("ink2_kernel_build_id", (1, 1, 1), (1, 1, 1), 0, KArgs::new().v(dp(&out)))?;
        let got = dev.dtoh_sync_copy(&out)?[0];
        let want = u64::from_str_radix(env!("KERNEL_BUILD_ID"), 16).unwrap_or(0);
        ensure!(got == want, "gpu_inkling2 kernel build id {got:016x} != binary {want:016x} (stale PTX)");
        Ok(me)
    }

    fn smem_optin(&self, name: &str, bytes: i32) -> Result<()> {
        let f = *self.funcs.get(name).ok_or_else(|| anyhow!("{name} not loaded"))?;
        let r = unsafe { sys::cuFuncSetAttribute(f, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuFuncSetAttribute({name}, smem {bytes}): {r:?}");
        Ok(())
    }

    pub fn launch(&self, name: &str, grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32, args: &mut KArgs) -> Result<()> {
        let f = *self.funcs.get(name).ok_or_else(|| anyhow!("{name} not loaded"))?;
        unsafe { result::ctx::set_current(self.ctx) }.map_err(|e| anyhow!("ctx bind: {e}"))?;
        let mut p = args.ptrs();
        if self.pdl.get() {
            let mut attr = sys::CUlaunchAttribute {
                id: sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
                pad: [0; 4],
                value: sys::CUlaunchAttributeValue { programmaticStreamSerializationAllowed: 1 },
            };
            let mut cfg = sys::CUlaunchConfig {
                gridDimX: grid.0, gridDimY: grid.1, gridDimZ: grid.2,
                blockDimX: block.0, blockDimY: block.1, blockDimZ: block.2,
                sharedMemBytes: smem, hStream: self.stream.get(), attrs: &mut attr, numAttrs: 1,
            };
            let r = unsafe { sys::cuLaunchKernelEx(&mut cfg, f, p.as_mut_ptr(), std::ptr::null_mut()) };
            ensure!(r == sys::CUresult::CUDA_SUCCESS, "launch (PDL) {name} grid {grid:?} block {block:?} smem {smem}: {r:?}");
            return Ok(());
        }
        unsafe { result::launch_kernel(f, grid, block, smem, self.stream.get(), &mut p) }
            .map_err(|e| anyhow!("launch {name} grid {grid:?} block {block:?} smem {smem}: {e}"))
    }

    pub fn sync(&self) -> Result<()> { self.dev.synchronize().map_err(|e| anyhow!("sync: {e}")) }

    /// GPU time per call (µs) of `f`'s launches: `reps` calls captured into one CUDA graph on a private stream
    /// and replayed (no host launch cost in the number: kernel time + graph node gaps).
    pub fn graph_time(&self, reps: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
        unsafe {
            let mut s: sys::CUstream = std::ptr::null_mut();
            ensure!(sys::cuStreamCreate(&mut s, 0) == sys::CUresult::CUDA_SUCCESS, "cuStreamCreate");
            let old = self.stream.replace(s);
            let res = (|| -> Result<f64> {
                for _ in 0..2 { f()?; }
                ensure!(sys::cuStreamSynchronize(s) == sys::CUresult::CUDA_SUCCESS, "warm-up sync");
                ensure!(sys::cuStreamBeginCapture_v2(s, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) == sys::CUresult::CUDA_SUCCESS, "begin capture");
                let r = (0..reps).try_for_each(|_| f());
                let mut g: sys::CUgraph = std::ptr::null_mut();
                let e = sys::cuStreamEndCapture(s, &mut g);
                r?;
                ensure!(e == sys::CUresult::CUDA_SUCCESS, "end capture: {e:?}");
                let mut x: sys::CUgraphExec = std::ptr::null_mut();
                ensure!(sys::cuGraphInstantiateWithFlags(&mut x, g, 0) == sys::CUresult::CUDA_SUCCESS, "instantiate");
                let (e0, e1) = (result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT)?, result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT)?);
                sys::cuGraphLaunch(x, s);
                result::event::record(e0, s)?;
                sys::cuGraphLaunch(x, s);
                result::event::record(e1, s)?;
                ensure!(sys::cuStreamSynchronize(s) == sys::CUresult::CUDA_SUCCESS, "graph sync");
                let ms = result::event::elapsed(e0, e1)?;
                let _ = result::event::destroy(e0);
                let _ = result::event::destroy(e1);
                sys::cuGraphExecDestroy(x);
                sys::cuGraphDestroy(g);
                Ok(ms as f64 * 1e3 / reps as f64)
            })();
            self.stream.set(old);
            sys::cuStreamSynchronize(s);
            sys::cuStreamDestroy_v2(s);
            res
        }
    }
}

impl Ink2Kernels {
    /// The process-wide module (loaded once; every later call returns the same handles).
    pub fn global(dev: &Arc<CudaDevice>) -> Result<&'static Ink2Kernels> {
        if let Some(k) = GLOBAL.get() { return Ok(k); }
        let k = Ink2Kernels::load(dev)?;
        let _ = GLOBAL.set(k);
        Ok(GLOBAL.get().unwrap())
    }
    /// The loaded module, if any (launch helpers call [`Ink2Kernels::global`] first).
    pub fn get() -> Result<&'static Ink2Kernels> { GLOBAL.get().ok_or_else(|| anyhow!("gpu_inkling2 module not loaded")) }
}

/// Load the embedded module (idempotent) and refuse a binary/PTX build-id mismatch.
pub fn load_module(dev: &Arc<CudaDevice>) -> Result<()> { Ink2Kernels::global(dev).map(|_| ()) }

/// `ink2_launch!(dev, "name", grid, block, smem, (args...))` — launches through the one loaded module on its
/// current stream (`Ink2Kernels::stream`, the device's blocking stream unless a capture swapped it).
#[macro_export]
macro_rules! ink2_launch {
    ($dev:expr, $name:expr, $grid:expr, $block:expr, $smem:expr, ($($a:expr),* $(,)?)) => {{
        let k = $crate::inkling2::gpu::Ink2Kernels::global($dev)?;
        let mut ka = $crate::inkling2::gpu::KArgs::new();
        $( ka.v($a); )*
        k.launch($name, $grid, $block, $smem, &mut ka)
    }};
}

/// Raw device address of a buffer (+ `off` elements).
pub fn dptr<T, S: DevicePtr<T>>(s: &S, off: usize) -> u64 { *s.device_ptr() as u64 + (off * std::mem::size_of::<T>()) as u64 }

/// CUDA-event timing on the module's stream: median of `reps` timings of `f`, after `warm` warm-up calls (ms).
pub fn time_ms(dev: &Arc<CudaDevice>, warm: usize, reps: usize, f: &mut dyn FnMut() -> Result<()>) -> Result<f64> {
    let mut ev: [sys::CUevent; 2] = [std::ptr::null_mut(); 2];
    for e in ev.iter_mut() {
        let r = unsafe { sys::cuEventCreate(e, 0) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuEventCreate {r:?}");
    }
    for _ in 0..warm { f()?; }
    dev.synchronize()?;
    let stream = Ink2Kernels::global(dev)?.stream.get();
    let mut v = Vec::with_capacity(reps);
    for _ in 0..reps {
        unsafe { sys::cuEventRecord(ev[0], stream); }
        f()?;
        unsafe { sys::cuEventRecord(ev[1], stream); }
        let r = unsafe { sys::cuEventSynchronize(ev[1]) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuEventSynchronize {r:?}");
        let mut ms = 0f32;
        unsafe { sys::cuEventElapsedTime(&mut ms, ev[0], ev[1]); }
        v.push(ms as f64);
    }
    for e in ev { unsafe { sys::cuEventDestroy_v2(e); } }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(v[reps / 2])
}
