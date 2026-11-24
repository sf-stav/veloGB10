//! DSV4.1 GPU host side of the Engram row runtime (leg EIO). Kernels: `kernels/gpu_dsv41/80_engram.cuh`.
//!
//! * [`MappedHost`]: page-locked, device-mapped host memory (`cuMemHostAlloc(PORTABLE | DEVICEMAP)`). The
//!   row cache, the descriptor rings and the control blocks live here: the host reader writes records and
//!   slot tables with plain stores and publishes with one release store; the GPU reads them in place (GB10's
//!   memory is one coherent pool: no H2D copy step exists in this design, the decode kernel IS the transfer).
//! * [`Ctrl`] / [`DescHead`]: the `#[repr(C)]` mirrors of `d41_engram_ctrl` / `d41_engram_desc`.
//! * Launch wrappers for `d41_engram_step_decode` (the bounded in-stream wait + decode),
//!   `d41_engram_records_decode` (contiguous records, the gather plug-in) and the bench spin kernel.
//! * [`decode_record_cpu`]: the CPU twin of the row decode (the reference's lookup, bit for bit).

use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// Entry kernels of this band; the gpu_dsv41 loader (`super::load_module`) collects them.
pub const FNS: &[&str] = &["d41_engram_step_decode", "d41_engram_records_decode", "d41_engram_bench_spin"];

/// Values per Engram row / bytes per packed record (256 E4M3 + 8 E8M0).
pub const ROW_VALUES: usize = 256;
pub const REC_BYTES: usize = 264;
/// Cache slot stride: 264 B rounded up to 16 B (the decode kernel's vector loads need 8-B / 16-B alignment).
/// The kernel's `D41_ENG_SLOT_STRIDE` is the same constant (the identity probe fails on a mismatch).
pub const SLOT_STRIDE: usize = 272;
/// Slot index meaning "write a zero row, read nothing" (DEAD positions: the model's gate is 0 there).
pub const ZERO_SLOT: u32 = u32::MAX;
/// Kernel error codes (`D41_ENG_ERR_*`).
pub const ERR_TIMEOUT: u32 = 1;
pub const ERR_HOST: u32 = 2;
pub const ERR_SEQ: u32 = 3;
pub const ERR_SHAPE: u32 = 4;
/// `mode` bits of `d41_engram_step_decode`.
pub const MODE_DECODE: u32 = 1;
pub const MODE_RECORDS: u32 = 2;

pub fn err_name(e: u32) -> &'static str {
    match e {
        0 => "none",
        ERR_TIMEOUT => "timeout (rows not published within the bounded wait)",
        ERR_HOST => "host error status (I/O error on a row read)",
        ERR_SEQ => "descriptor sequence mismatch (submit / launch out of step)",
        ERR_SHAPE => "descriptor shape exceeds the launch geometry",
        _ => "unknown",
    }
}

/// Page-locked host memory mapped into the device address space.
pub struct MappedHost {
    ptr: *mut u8,
    dptr: u64,
    len: usize,
}
// Plain host memory; every concurrent access pattern on it is documented at the user (release / acquire on
// the sequence words, disjoint slots per reader).
unsafe impl Send for MappedHost {}
unsafe impl Sync for MappedHost {}

impl MappedHost {
    pub fn new(dev: &Arc<CudaDevice>, len: usize) -> Result<Self> {
        use cudarc::driver::sys;
        dev.bind_to_thread()?;
        let len = len.max(64).div_ceil(4096) * 4096;
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        let r = unsafe { sys::cuMemHostAlloc(&mut p, len, sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS && !p.is_null(), "cuMemHostAlloc({len} B, mapped) failed ({r:?})");
        let mut d: sys::CUdeviceptr = 0;
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut d, p, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            unsafe { sys::cuMemFreeHost(p) };
            anyhow::bail!("cuMemHostGetDevicePointer failed ({r:?})");
        }
        unsafe { std::ptr::write_bytes(p as *mut u8, 0, len) };
        Ok(MappedHost { ptr: p as *mut u8, dptr: d, len })
    }
    pub fn host_ptr(&self) -> *mut u8 { self.ptr }
    pub fn dev_ptr(&self) -> u64 { self.dptr }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
}
impl Drop for MappedHost {
    fn drop(&mut self) {
        unsafe { cudarc::driver::sys::cuMemFreeHost(self.ptr as *mut std::ffi::c_void) };
    }
}

/// Mirror of `d41_engram_ctrl` (256 B; host-device words on separate 64-B lines).
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct Ctrl {
    pub pub_seq: u64,
    pub pad0: [u64; 7],
    pub done_seq: u64,
    pub pad1: [u64; 7],
    pub stalls: u64,
    pub stall_ns: u64,
    pub max_stall_ns: u64,
    pub last_wait_ns: u64,
    pub timeouts: u64,
    pub host_errs: u64,
    pub err: u32,
    pub err_seq_lo: u32,
    /// block 0 saw the publish -> the last block finished (the decode itself), latest / sum / max
    pub last_decode_ns: u64,
    pub sum_decode_ns: u64,
    pub max_decode_ns: u64,
    /// steps consumed (decoded or failed)
    pub steps: u64,
    pub pad2: [u64; 5],
}
pub const CTRL_BYTES: usize = 256;
pub const CTRL_OFF_PUB: usize = 0;
pub const CTRL_OFF_DONE: usize = 64;

/// Mirror of `d41_engram_desc` (32 B), followed in the ring slot by the u32 slot table.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct DescHead {
    pub seq: u64,
    pub n_tok: u32,
    pub n_own: u32,
    pub n_layers: u32,
    pub status: u32,
    pub pad: [u32; 2],
}
pub const DESC_HEAD_BYTES: usize = 32;

/// Bytes of one descriptor ring slot for up to `max_rows` table entries (64-B multiple).
pub fn desc_stride(max_rows: usize) -> usize {
    (DESC_HEAD_BYTES + 4 * max_rows).div_ceil(64) * 64
}

/// The device-side state of one channel: u64 words in device memory ([0] next step to consume,
/// [1] block arrivals, [2] the step's wait ns, [3] the step's error, [4] the time block 0 saw the publish),
/// zeroed at creation.
pub struct ChannelDev {
    pub words: cudarc::driver::CudaSlice<u64>,
}
impl ChannelDev {
    pub fn new(dev: &Arc<CudaDevice>) -> Result<Self> {
        let mut words = dev.alloc_zeros::<u64>(8)?;
        dev.memset_zeros(&mut words)?;
        dev.synchronize()?;
        Ok(ChannelDev { words })
    }
    /// Set the consumption counter (a channel restart after an aborted round; the stream must be idle).
    pub fn reset_to(&mut self, dev: &Arc<CudaDevice>, next: u64) -> Result<()> {
        dev.htod_sync_copy_into(&[next, 0, 0, 0, 0, 0, 0, 0], &mut self.words)?;
        Ok(())
    }
}

/// Geometry and addresses of one `d41_engram_step_decode` launch (Copy: a round's launch code keeps it
/// without borrowing the runtime; every address stays valid while the runtime lives).
#[derive(Clone, Copy, Debug)]
pub struct StepLaunch {
    pub ctrl_dptr: u64,
    /// the channel's device words (`ChannelDev::words`)
    pub dev_words: u64,
    pub desc_dptr: u64,
    pub desc_stride: u32,
    pub ring: u32,
    pub cache_dptr: u64,
    pub out_bf16: u64,
    pub out_rec: u64,
    pub max_tok: u32,
    pub max_own: u32,
    /// Engram layers the channel carries (grid sizing only; the kernel reads the step's count)
    pub layers: u32,
    pub mode: u32,
    pub timeout_ns: u64,
    /// grid blocks (256 threads each); 0 = derive from the geometry
    pub blocks: u32,
}

/// Enqueue the bounded wait + decode of the channel's next step on the device's (blocking) stream.
pub fn launch_step_decode(dev: &Arc<CudaDevice>, l: &StepLaunch) -> Result<()> {
    let f = dev.get_func(super::MODULE, "d41_engram_step_decode").context("d41_engram_step_decode not loaded")?;
    let rows = l.layers.max(1) as usize * l.max_tok as usize * l.max_own as usize;
    let blocks = if l.blocks > 0 { l.blocks } else { (rows.div_ceil(8) as u32).clamp(1, 96) };
    let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        f.launch(cfg, (l.ctrl_dptr, l.dev_words, l.desc_dptr, l.desc_stride, l.ring, l.cache_dptr, l.out_bf16, l.out_rec,
                       l.max_tok, l.max_own, l.mode, l.timeout_ns))
    }?;
    Ok(())
}

/// Leg PH1: [`launch_step_decode`] on `stream` (the round's compute stream: a capture records it as a graph node; the
/// kernel consumes its channel's device counter, so every replay decodes the next published step).
pub fn launch_step_decode_on(dev: &Arc<CudaDevice>, stream: &cudarc::driver::CudaStream, l: &StepLaunch) -> Result<()> {
    let f = dev.get_func(super::MODULE, "d41_engram_step_decode").context("d41_engram_step_decode not loaded")?;
    let rows = l.layers.max(1) as usize * l.max_tok as usize * l.max_own as usize;
    let blocks = if l.blocks > 0 { l.blocks } else { (rows.div_ceil(8) as u32).clamp(1, 96) };
    let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        f.launch_on_stream(stream, cfg, (l.ctrl_dptr, l.dev_words, l.desc_dptr, l.desc_stride, l.ring, l.cache_dptr, l.out_bf16, l.out_rec,
                                         l.max_tok, l.max_own, l.mode, l.timeout_ns))
    }?;
    Ok(())
}

/// Decode `n` contiguous records (`rec_dptr + i * rec_stride`) into `out_dptr + i * 512` bytes (bf16).
pub fn launch_records_decode(dev: &Arc<CudaDevice>, rec_dptr: u64, rec_stride: u32, n: u32, out_dptr: u64) -> Result<()> {
    let f = dev.get_func(super::MODULE, "d41_engram_records_decode").context("d41_engram_records_decode not loaded")?;
    let blocks = (n.div_ceil(8)).clamp(1, 1024);
    let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    unsafe { f.launch(cfg, (rec_dptr, rec_stride, n, out_dptr)) }?;
    Ok(())
}

/// Bench: occupy the stream for `ns`, then (if `flag_dptr != 0`) release-store `value` there.
pub fn launch_bench_spin(dev: &Arc<CudaDevice>, ns: u64, flag_dptr: u64, value: u64) -> Result<()> {
    let f = dev.get_func(super::MODULE, "d41_engram_bench_spin").context("d41_engram_bench_spin not loaded")?;
    let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
    unsafe { f.launch(cfg, (ns, flag_dptr, value)) }?;
    Ok(())
}

/// The CPU twin of the row decode: `bf16(f32(e4m3[i]) * f32(e8m0[i / 32]))`, round to nearest even
/// (torch's `.to(torch.bfloat16)`), as bf16 bit patterns. Independent of the kernel's code path: the f32
/// decode from `abi::{e4m3, e8m0}` and the `half` crate's conversion.
pub fn decode_record_cpu(rec: &[u8], out: &mut [u16]) {
    assert!(rec.len() >= REC_BYTES && out.len() >= ROW_VALUES);
    for i in 0..ROW_VALUES {
        let v = super::abi::e4m3(rec[i]) * super::abi::e8m0(rec[ROW_VALUES + i / 32]);
        out[i] = half::bf16::from_f32(v).to_bits();
    }
}

/// Two bf16 bit patterns are the same value (every NaN equals every NaN; +0 and -0 differ: the reference
/// keeps the sign of a zero E4M3 code).
#[inline]
pub fn bf16_same(a: u16, b: u16) -> bool {
    let nan = |x: u16| (x & 0x7F80) == 0x7F80 && (x & 0x007F) != 0;
    a == b || (nan(a) && nan(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirrors_match_the_cuh_layout() {
        assert_eq!(std::mem::size_of::<Ctrl>(), CTRL_BYTES);
        assert_eq!(std::mem::offset_of!(Ctrl, pub_seq), CTRL_OFF_PUB);
        assert_eq!(std::mem::offset_of!(Ctrl, done_seq), CTRL_OFF_DONE);
        assert_eq!(std::mem::offset_of!(Ctrl, stalls), 128);
        assert_eq!(std::mem::offset_of!(Ctrl, err), 176);
        assert_eq!(std::mem::offset_of!(Ctrl, last_decode_ns), 184);
        assert_eq!(std::mem::offset_of!(Ctrl, steps), 208);
        assert_eq!(std::mem::size_of::<DescHead>(), DESC_HEAD_BYTES);
        assert_eq!(std::mem::offset_of!(DescHead, status), 20);
        assert_eq!(SLOT_STRIDE % 16, 0);
        assert!(SLOT_STRIDE >= REC_BYTES);
        assert_eq!(desc_stride(1152) % 64, 0);
    }

    #[test]
    fn cpu_decode_matches_the_reference_formula() {
        // value code 0x38 = 1.0, 0x40 = 2.0, 0xB8 = -1.0, 0x01 = 2^-9 (subnormal), 0x7E = 448; scales 2^0, 2^-3, 2^-127
        let mut rec = vec![0u8; REC_BYTES];
        rec[0] = 0x38;
        rec[1] = 0x40;
        rec[2] = 0xB8;
        rec[32] = 0x01;
        rec[64] = 0x7E;
        rec[256] = 127;
        rec[257] = 124;
        rec[258] = 0;
        let mut out = vec![0u16; ROW_VALUES];
        decode_record_cpu(&rec, &mut out);
        let f = |b: u16| half::bf16::from_bits(b).to_f32();
        assert_eq!((f(out[0]), f(out[1]), f(out[2])), (1.0, 2.0, -1.0));
        assert_eq!(f(out[32]), 2f32.powi(-9) * 0.125);
        assert_eq!(out[64], half::bf16::from_f32(448.0 * 2f32.powi(-127)).to_bits());
        assert_eq!(out[3], 0); // +0
        assert!(bf16_same(0x7FC0, 0x7FFF) && !bf16_same(0x0000, 0x8000));
    }
}
