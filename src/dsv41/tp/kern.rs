//! DSV4.1 TP (leg TPC): host side of band 05 (`kernels/gpu_dsv41/05_tp.cuh`). The entries load from the embedded
//! `gpu_dsv41` PTX under this leg's own module name (`MODULE`), so the I0 loader's entry list is untouched; the
//! KERNEL_BUILD_ID handshake runs on this copy too.

use super::comm::{TpOut, TpRow};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaFunction, CudaStream, DeviceRepr, LaunchAsync, LaunchConfig};
use std::ffi::c_void;
use std::sync::Arc;

pub const MODULE: &str = "gpu_dsv41_tp";
pub const FNS: &[&str] = &["d41_kernel_build_id", "d41tp_head_local", "d41tp_head_merge", "d41tp_head_combine",
                           "d41tp_synth_frame", "d41tp_synth_check", "d41tp_synth_logits", "d41tp_spin"];
/// `D41TP_HEAD_THREADS`
pub const HEAD_THREADS: u32 = 256;
/// Shard elements per block of the head kernels (stage 1a); the row's chunks merge in chunk order (stage 1b).
pub const CHUNK: usize = 2048;
/// Chunks of a `vlen`-element shard.
pub fn chunks(vlen: usize) -> usize { vlen.div_ceil(CHUNK).max(1) }

/// The by-value argument of the shard kernels (`d41tp_local_args`, same field order).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LocalArgs {
    pub logits: u64, pub ld: i64, pub dlogits: u64, pub dld: i64, pub rows_p: u64, pub tgt: u64, pub dout: u64, pub rec: u64,
    /// stage 1a -> 1b scratch: [rows][chunks(vlen)][16] floats
    pub part: u64,
    pub vlen: i32, pub voff: i32, pub rank: i32, pub world: i32, pub zero_others: i32,
    /// shard elements per block (`CHUNK`; set by the launchers)
    pub chunk: i32,
}
unsafe impl DeviceRepr for LocalArgs {}
unsafe impl DeviceRepr for TpRow {}
unsafe impl DeviceRepr for TpOut {}

#[allow(dead_code)]
pub struct TpKernels {
    pub dev: Arc<CudaDevice>,
    local: CudaFunction,
    merge: CudaFunction,
    combine: CudaFunction,
    synth_frame: CudaFunction,
    synth_check: CudaFunction,
    synth_logits: CudaFunction,
    spin: CudaFunction,
}

impl TpKernels {
    pub fn load(dev: &Arc<CudaDevice>) -> Result<Self> {
        if dev.get_func(MODULE, "d41tp_head_local").is_none() {
            dev.load_ptx(cudarc::nvrtc::Ptx::from_src(crate::ptx_embed::GPU_DSV41), MODULE, FNS)
                .context("load the gpu_dsv41 module (TP band 05)")?;
        }
        let f = |n: &str| dev.get_func(MODULE, n).with_context(|| format!("gpu_dsv41 kernel {n} missing (band 05)"));
        // the stale-PTX handshake on this copy of the module
        let id = f("d41_kernel_build_id")?;
        let mut out = dev.alloc_zeros::<u64>(1)?;
        unsafe { id.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 }, (&mut out,)) }?;
        let got = dev.dtoh_sync_copy(&out)?[0];
        let want = u64::from_str_radix(env!("KERNEL_BUILD_ID"), 16).unwrap_or(0);
        ensure!(got == want, "gpu_dsv41 (TP band) kernel build id {got:016x} != binary {want:016x} (stale PTX)");
        Ok(TpKernels { dev: dev.clone(), local: f("d41tp_head_local")?, merge: f("d41tp_head_merge")?, combine: f("d41tp_head_combine")?,
                       synth_frame: f("d41tp_synth_frame")?, synth_check: f("d41tp_synth_check")?, synth_logits: f("d41tp_synth_logits")?, spin: f("d41tp_spin")? })
    }

    fn chunk_cfg(rows: usize, nb: usize) -> LaunchConfig { LaunchConfig { grid_dim: (rows as u32, nb as u32, 1), block_dim: (HEAD_THREADS, 1, 1), shared_mem_bytes: 0 } }
    fn merge_cfg(rows: usize) -> LaunchConfig { LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 } }
    fn per_row_cfg(rows: usize) -> LaunchConfig { LaunchConfig { grid_dim: (rows.div_ceil(64) as u32, 1, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 } }
    fn flat_cfg(n: usize) -> LaunchConfig { LaunchConfig { grid_dim: (n.div_ceil(256).clamp(1, 1024) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 } }

    /// Stage 1 of the head exchange: this rank's records (`a.rec`, `a.rows_p`, `a.logits` / `ld` / `vlen` / `voff`, scratch
    /// `a.part`): one block per (row, chunk), then the chunk-order merge.
    pub fn head_local(&self, s: &CudaStream, rows: usize, mut a: LocalArgs) -> Result<()> {
        a.chunk = CHUNK as i32;
        let nb = chunks(a.vlen as usize);
        // leg PF1: the raw module with the PDL attribute per `--dsv41-pdl` (both entries wait first: 05_tp.cuh)
        let pdl = crate::dsv41::gpu::pdl_on();
        let st = s.stream as u64;
        let c = Self::chunk_cfg(rows, nb);
        crate::dsv41::gpu::launch_raw("d41tp_head_local", c.grid_dim, c.block_dim, 0, st, pdl, &mut [&a as *const LocalArgs as *mut c_void])?;
        let (c, nbi) = (Self::merge_cfg(rows), nb as i32);
        crate::dsv41::gpu::launch_raw("d41tp_head_merge", c.grid_dim, c.block_dim, 0, st, pdl,
                                      &mut [&a as *const LocalArgs as *mut c_void, &nbi as *const i32 as *mut c_void])
    }
    pub fn head_combine(&self, s: &CudaStream, rec: u64, rows: usize, world: usize, rows_p: u64, out: u64) -> Result<()> {
        let (r, w) = (rows as i32, world as i32);
        let c = Self::per_row_cfg(rows);
        let mut p = [&rec as *const u64 as *mut c_void, &r as *const i32 as *mut c_void, &w as *const i32 as *mut c_void,
                     &rows_p as *const u64 as *mut c_void, &out as *const u64 as *mut c_void];
        crate::dsv41::gpu::launch_raw("d41tp_head_combine", c.grid_dim, c.block_dim, 0, s.stream as u64, crate::dsv41::gpu::pdl_on(), &mut p)
    }
    /// The smoke's synthetic partial of `rank` over a sum frame (`n` floats, sections per `frame::FrameSpec`).
    #[allow(clippy::too_many_arguments)]
    pub fn synth_frame(&self, s: &CudaStream, f: u64, n: usize, rows: usize, hidden: usize, mix_used: usize, key: u64, rank: usize) -> Result<()> {
        unsafe { self.synth_frame.clone().launch_on_stream(s, Self::flat_cfg(n),
            (f, n as i64, rows as i32, hidden as i32, super::frame::MIX_STRIDE as i32, mix_used as i32, key, rank as i32)) }.context("d41tp_synth_frame")
    }
    /// Recompute every rank's partial and compare the reduced frame bit for bit (bad[0] += mismatches, bad[1] = min index).
    #[allow(clippy::too_many_arguments)]
    pub fn synth_check(&self, s: &CudaStream, f: u64, n: usize, rows: usize, hidden: usize, mix_used: usize, key: u64, world: usize,
                       first: usize, bad: u64) -> Result<()> {
        unsafe { self.synth_check.clone().launch_on_stream(s, Self::flat_cfg(n),
            (f, n as i64, rows as i32, hidden as i32, super::frame::MIX_STRIDE as i32, mix_used as i32, key, world as i32, first as i32, bad)) }
            .context("d41tp_synth_check")
    }
    /// Synthetic logits over the whole vocabulary ([rows][vocab], row stride `ld`).
    pub fn synth_logits(&self, s: &CudaStream, l: u64, ld: usize, rows: usize, vocab: usize, key: u64) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (64, rows as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { self.synth_logits.clone().launch_on_stream(s, cfg, (l, ld as i64, rows as i32, vocab as i32, key)) }.context("d41tp_synth_logits")
    }

    /// Hold the stream for `us` microseconds (the timing gate).
    pub fn spin(&self, s: &CudaStream, us: u64) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        unsafe { self.spin.clone().launch_on_stream(s, cfg, (us * 1000,)) }.context("d41tp_spin")
    }
}
