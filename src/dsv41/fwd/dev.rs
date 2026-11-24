//! SL: the device context of the per-layer forward — one CUDA device, ONE blocking compute stream
//! (AGENTS §2.1: a blocking stream orders itself with cudarc's NULL-stream copies and with the bands that
//! launch through cudarc), raw by-value-struct launches of any `gpu_dsv41` entry on that stream (KC / KM
//! entries are launched here directly so every kernel of a layer lands on the same stream), device
//! buffers, downloads and event timing.

use crate::dsv41::weights::{Act, ActDtype, DevPtr, Launch};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{sys, CudaDevice, CudaSlice, DevicePtr, DeviceRepr, ValidAsZeroBits};
use std::ffi::c_void;
use std::sync::Arc;

pub struct Gpu { pub dev: Arc<CudaDevice>, pub stream: u64 }

/// A device buffer of raw bytes.
pub struct Buf { pub mem: CudaSlice<u8>, pub bytes: usize }
impl Buf {
    pub fn ptr(&self) -> u64 { *self.mem.device_ptr() }
    pub fn p(&self) -> DevPtr { DevPtr(self.ptr()) }
}

pub fn dp<T>(s: &CudaSlice<T>) -> u64 { *s.device_ptr() }

impl Gpu {
    pub fn new() -> Result<Gpu> {
        let dev = CudaDevice::new(0).context("CUDA device 0")?;
        crate::dsv41::gpu::load_module(&dev)?;
        let mut st: sys::CUstream = std::ptr::null_mut();
        // flags 0 = CU_STREAM_DEFAULT: a BLOCKING stream (never NonBlocking: AGENTS §2.1)
        ensure!(unsafe { sys::cuStreamCreate(&mut st, 0) } == sys::CUresult::CUDA_SUCCESS, "cuStreamCreate");
        Ok(Gpu { dev, stream: st as u64 })
    }
    /// The launch of every backend call on this stream: PDL per `--dsv41-pdl` (leg PF1; every gpu_dsv41 entry waits before
    /// it touches a predecessor's data, 00_abi.cuh rule 4).
    pub fn l(&self) -> Launch { Launch { stream: self.stream, pdl: crate::dsv41::gpu::pdl_on() } }
    /// Wait for the stream. Leg HT1: `--dsv41-round-wait hybrid` (opt-in) waits through `host_wait::stream_wait` (per call site:
    /// sleep until just before the predicted end, then poll); the default is the driver's `cuStreamSynchronize`, unchanged.
    #[track_caller]
    pub fn sync(&self) -> Result<()> {
        if round_wait_hybrid() {
            return crate::host_wait::stream_wait(self.stream as sys::CUstream).map_err(|e| anyhow::anyhow!("stream sync: {e:?} (a kernel failed)"));
        }
        let r = unsafe { sys::cuStreamSynchronize(self.stream as sys::CUstream) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "stream sync: {r:?} (a kernel failed)");
        Ok(())
    }
    /// `bytes` of device memory, zeroed on the stream.
    pub fn buf(&self, bytes: usize) -> Result<Buf> {
        let n = bytes.max(256);
        let mem = unsafe { self.dev.alloc::<u8>(n) }.context("device alloc")?;
        let b = Buf { mem, bytes: n };
        self.memset(b.ptr(), n, 0)?;
        Ok(b)
    }
    pub fn up<T: DeviceRepr + Unpin>(&self, v: &[T]) -> Result<CudaSlice<T>> { Ok(self.dev.htod_sync_copy(v)?) }
    pub fn zeros<T: DeviceRepr + ValidAsZeroBits + Unpin>(&self, n: usize) -> Result<CudaSlice<T>> {
        let s = self.dev.alloc_zeros::<T>(n.max(1))?;
        self.memset(dp(&s), n.max(1) * std::mem::size_of::<T>(), 0)?;
        Ok(s)
    }
    pub fn memset(&self, ptr: u64, bytes: usize, v: u8) -> Result<()> {
        let r = unsafe { sys::cuMemsetD8Async(ptr, v, bytes, self.stream as sys::CUstream) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "memset: {r:?}");
        Ok(())
    }
    /// Copy host bytes to a device address on the stream (synchronous for the host buffer's lifetime).
    pub fn h2d(&self, dst: u64, src: &[u8]) -> Result<()> {
        if src.is_empty() { return Ok(()); }
        self.sync()?;
        let r = unsafe { sys::cuMemcpyHtoD_v2(dst, src.as_ptr() as *const c_void, src.len()) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "htod: {r:?}");
        Ok(())
    }
    /// A small host array to a device address, staged through pinned-free memory on the stream (the copy
    /// is complete when this returns: cuMemcpyHtoDAsync from pageable memory is synchronous for the host).
    pub fn h2d_async<T: Copy>(&self, dst: u64, src: &[T]) -> Result<()> {
        if src.is_empty() { return Ok(()); }
        let r = unsafe { sys::cuMemcpyHtoDAsync_v2(dst, src.as_ptr() as *const c_void, std::mem::size_of_val(src), self.stream as sys::CUstream) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "htod async: {r:?}");
        Ok(())
    }
    pub fn d2d(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
        if bytes == 0 { return Ok(()); }
        let r = unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, self.stream as sys::CUstream) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "dtod: {r:?}");
        Ok(())
    }
    /// Download `n` elements of T from a raw device address on `self.stream` (avoids NULL-stream cross-stream stalls).
    #[track_caller]
    pub fn down<T: Copy + Default>(&self, ptr: u64, n: usize) -> Result<Vec<T>> {
        let mut v = vec![T::default(); n];
        if n == 0 { self.sync()?; return Ok(v); }
        let r = unsafe { sys::cuMemcpyDtoHAsync_v2(v.as_mut_ptr() as *mut c_void, ptr, n * std::mem::size_of::<T>(), self.stream as sys::CUstream) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "dtoh async: {r:?}");
        self.sync()?;
        Ok(v)
    }
    /// Download two device slices `(p0, n0)` and `(p1, n1)` on `self.stream` with a SINGLE stream sync.
    #[track_caller]
    pub fn down2<A: Copy + Default, B: Copy + Default>(&self, p0: u64, n0: usize, p1: u64, n1: usize) -> Result<(Vec<A>, Vec<B>)> {
        let mut a = vec![A::default(); n0];
        let mut b = vec![B::default(); n1];
        if n0 > 0 {
            let r = unsafe { sys::cuMemcpyDtoHAsync_v2(a.as_mut_ptr() as *mut c_void, p0, n0 * std::mem::size_of::<A>(), self.stream as sys::CUstream) };
            ensure!(r == sys::CUresult::CUDA_SUCCESS, "dtoh2 a: {r:?}");
        }
        if n1 > 0 {
            let r = unsafe { sys::cuMemcpyDtoHAsync_v2(b.as_mut_ptr() as *mut c_void, p1, n1 * std::mem::size_of::<B>(), self.stream as sys::CUstream) };
            ensure!(r == sys::CUresult::CUDA_SUCCESS, "dtoh2 b: {r:?}");
        }
        self.sync()?;
        Ok((a, b))
    }
    /// Download three device slices `(p0, n0)`, `(p1, n1)`, `(p2, n2)` on `self.stream` with a SINGLE stream sync.
    #[track_caller]
    pub fn down3<A: Copy + Default, B: Copy + Default, C: Copy + Default>(&self, p0: u64, n0: usize, p1: u64, n1: usize, p2: u64, n2: usize) -> Result<(Vec<A>, Vec<B>, Vec<C>)> {
        let mut a = vec![A::default(); n0];
        let mut b = vec![B::default(); n1];
        let mut c = vec![C::default(); n2];
        if n0 > 0 {
            let r = unsafe { sys::cuMemcpyDtoHAsync_v2(a.as_mut_ptr() as *mut c_void, p0, n0 * std::mem::size_of::<A>(), self.stream as sys::CUstream) };
            ensure!(r == sys::CUresult::CUDA_SUCCESS, "dtoh3 a: {r:?}");
        }
        if n1 > 0 {
            let r = unsafe { sys::cuMemcpyDtoHAsync_v2(b.as_mut_ptr() as *mut c_void, p1, n1 * std::mem::size_of::<B>(), self.stream as sys::CUstream) };
            ensure!(r == sys::CUresult::CUDA_SUCCESS, "dtoh3 b: {r:?}");
        }
        if n2 > 0 {
            let r = unsafe { sys::cuMemcpyDtoHAsync_v2(c.as_mut_ptr() as *mut c_void, p2, n2 * std::mem::size_of::<C>(), self.stream as sys::CUstream) };
            ensure!(r == sys::CUresult::CUDA_SUCCESS, "dtoh3 c: {r:?}");
        }
        self.sync()?;
        Ok((a, b, c))
    }
    #[track_caller]
    pub fn down_bf16(&self, ptr: u64, n: usize) -> Result<Vec<f32>> {
        Ok(self.down::<u16>(ptr, n)?.into_iter().map(|b| f32::from_bits((b as u32) << 16)).collect())
    }
    /// Launch entry `name` with ONE by-value argument struct on the compute stream (the KC / KM / SL ABI).
    pub fn launch1<A>(&self, name: &str, grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32, a: &A) -> Result<()> {
        // leg PF1: through the raw launcher, with the PDL attribute per `--dsv41-pdl`
        crate::dsv41::gpu::launch_args(name, grid, block, smem, self.stream, a)
    }
}

/// A row-major bf16 activation view.
pub fn bf16(ptr: u64, rows: usize, cols: usize, ld: usize) -> Act { Act { ptr: DevPtr(ptr), rows, cols, ld, dtype: ActDtype::Bf16 } }
pub fn f32a(ptr: u64, rows: usize, cols: usize, ld: usize) -> Act { Act { ptr: DevPtr(ptr), rows, cols, ld, dtype: ActDtype::F32 } }

/// CUDA events on the compute stream (step timing).
pub struct Ev(sys::CUevent);
impl Ev {
    pub fn new() -> Result<Ev> {
        let mut e: sys::CUevent = std::ptr::null_mut();
        ensure!(unsafe { sys::cuEventCreate(&mut e, 0) } == sys::CUresult::CUDA_SUCCESS, "cuEventCreate");
        Ok(Ev(e))
    }
    pub fn rec(&self, g: &Gpu) -> Result<()> {
        ensure!(unsafe { sys::cuEventRecord(self.0, g.stream as sys::CUstream) } == sys::CUresult::CUDA_SUCCESS, "cuEventRecord");
        Ok(())
    }
    /// ms from `start` to this event (both recorded, this one completed).
    pub fn ms_since(&self, start: &Ev) -> Result<f32> {
        ensure!(unsafe { sys::cuEventSynchronize(self.0) } == sys::CUresult::CUDA_SUCCESS, "cuEventSynchronize");
        let mut ms = 0f32;
        ensure!(unsafe { sys::cuEventElapsedTime(&mut ms, start.0, self.0) } == sys::CUresult::CUDA_SUCCESS, "cuEventElapsedTime");
        Ok(ms)
    }
}
impl Drop for Ev { fn drop(&mut self) { unsafe { sys::cuEventDestroy_v2(self.0); } } }

/// Leg HT1: `--dsv41-round-wait hybrid` (resolved once; default spin = the driver's own wait, as before).
pub fn round_wait_hybrid() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::lookup("dsv41-round-wait").and_then(crate::opts::get).as_deref() == Some("hybrid"))
}
