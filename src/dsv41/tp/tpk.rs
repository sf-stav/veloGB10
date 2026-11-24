//! DSV4.1 TP (leg TP): host side of the TP forward glue (`kernels/gpu_dsv41/05_tp_fwd.cuh`, included from band
//! 05). Loaded from the embedded `gpu_dsv41` PTX under this leg's module name, with the KERNEL_BUILD_ID handshake.

use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaFunction, CudaStream, DeviceRepr, LaunchAsync, LaunchConfig};
use std::ffi::c_void;
use std::sync::Arc;

pub const MODULE: &str = "gpu_dsv41_tpf";
pub const FNS: &[&str] = &["d41_kernel_build_id", "d41tpf_bf16_f32", "d41tpf_expand_hc", "d41tpf_fill_pre", "d41tpf_ring_lin", "d41fw_tap", "d41fw_xenc", "d41fw_xdec",
                           "d41fw_swa_rows", "d41fw_compress_seg", "d41fw_stash_seg", "d41fw_index_k_seg", "d41se_pen",
                           "d41fw_compress_tab", "d41fw_stash_tab", "d41fw_index_k_tab"];

#[repr(C)] #[derive(Clone, Copy, Debug, Default)]
pub struct CvtArgs { pub inp: u64, pub out: u64, pub n: i64 }
#[repr(C)] #[derive(Clone, Copy, Debug, Default)]
pub struct ExpandArgs { pub inp: u64, pub out: u64, pub in_ld: i64, pub out_ld: i64, pub rows: i32, pub dim: i32, pub hc: i32, pub pad: i32 }
#[repr(C)] #[derive(Clone, Copy, Debug, Default)]
pub struct PreArgs { pub out: u64, pub stride: i64, pub rows: i32, pub hc: i32 }
#[repr(C)] #[derive(Clone, Copy, Debug, Default)]
pub struct RingArgs { pub ring_q: u64, pub ring_s: u64, pub lin_q: u64, pub lin_s: u64, pub p_lo: i64, pub n: i32, pub slots: i32, pub q_bytes: i32, pub s_bytes: i32 }
unsafe impl DeviceRepr for CvtArgs {}
unsafe impl DeviceRepr for ExpandArgs {}
unsafe impl DeviceRepr for PreArgs {}
unsafe impl DeviceRepr for RingArgs {}
#[repr(C)] #[derive(Clone, Copy, Debug, Default)]
pub struct TapArgs { pub x: u64, pub dst: u64, pub dst_ld: i64, pub hc: i32, pub dim: i32, pub row0: i32, pub n: i32, pub dst0: i32, pub col: i32 }
#[repr(C)] #[derive(Clone, Copy, Debug, Default)]
pub struct XencArgs { pub inp: u64, pub out: u64, pub nwords: i32, pub rank: i32, pub world: i32, pub pad: i32 }
#[repr(C)] #[derive(Clone, Copy, Debug, Default)]
pub struct XdecArgs { pub inp: u64, pub out: u64, pub nwords: i32, pub world: i32 }
unsafe impl DeviceRepr for TapArgs {}
unsafe impl DeviceRepr for XencArgs {}
unsafe impl DeviceRepr for XdecArgs {}
/// `d41fw_swa_rows_args` (kernels/gpu_dsv41/44_kv_graph.cuh)
#[repr(C)] #[derive(Clone, Copy, Debug, Default)]
pub struct SwaRowsArgs { pub kv_raw: u64, pub norm_w: u64, pub inv_freq: u64, pub pos: u64, pub ring_q: u64, pub ring_s: u64, pub rows: i32, pub norm_fmt: u32, pub eps: f32, pub pad: i32 }
unsafe impl DeviceRepr for SwaRowsArgs {}

/// leg SE: `d41se_pen_args`
#[repr(C)] #[derive(Clone, Copy, Debug, Default)]
pub struct PenArgs { pub logits: u64, pub counts: u64, pub v0: i64, pub vlocal: i32, pub n_extra: i32, pub e: [i32; 5], pub pad0: i32, pub pres: f32, pub freq: f32, pub rep: f32, pub pad1: f32 }
unsafe impl DeviceRepr for PenArgs {}

#[allow(dead_code)]
pub struct TpfKernels { cvt: CudaFunction, expand: CudaFunction, pre: CudaFunction, ring: CudaFunction, tap: CudaFunction, xenc: CudaFunction, xdec: CudaFunction,
                        swa: CudaFunction, comp: CudaFunction, stash: CudaFunction, idxk: CudaFunction, pen: CudaFunction }

fn cfg(grid: (u32, u32), threads: u32) -> LaunchConfig { LaunchConfig { grid_dim: (grid.0.max(1), grid.1.max(1), 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 } }

/// leg PF1: every TP glue launch goes through the raw module with the PDL attribute per `--dsv41-pdl` (each entry here
/// starts with D41_PDL_ENTRY: it waits before touching its predecessor's data).
fn go<A>(s: &CudaStream, name: &str, c: LaunchConfig, a: &A) -> Result<()> {
    crate::dsv41::gpu::launch_args(name, c.grid_dim, c.block_dim, c.shared_mem_bytes, s.stream as u64, a)
}
fn go2<A, B>(s: &CudaStream, name: &str, c: LaunchConfig, a: &A, b: &B) -> Result<()> {
    let mut p = [a as *const A as *mut c_void, b as *const B as *mut c_void];
    crate::dsv41::gpu::launch_raw(name, c.grid_dim, c.block_dim, c.shared_mem_bytes, s.stream as u64, crate::dsv41::gpu::pdl_on(), &mut p)
}

impl TpfKernels {
    pub fn load(dev: &Arc<CudaDevice>) -> Result<Self> {
        if dev.get_func(MODULE, "d41tpf_bf16_f32").is_none() {
            dev.load_ptx(cudarc::nvrtc::Ptx::from_src(crate::ptx_embed::GPU_DSV41), MODULE, FNS).context("load the gpu_dsv41 module (TP forward glue)")?;
        }
        let f = |n: &str| dev.get_func(MODULE, n).with_context(|| format!("gpu_dsv41 kernel {n} missing (TP forward glue)"));
        let id = f("d41_kernel_build_id")?;
        let mut out = dev.alloc_zeros::<u64>(1)?;
        unsafe { id.launch(cfg((1, 1), 1), (&mut out,)) }?;
        let got = dev.dtoh_sync_copy(&out)?[0];
        let want = u64::from_str_radix(env!("KERNEL_BUILD_ID"), 16).unwrap_or(0);
        ensure!(got == want, "gpu_dsv41 (TP forward glue) kernel build id {got:016x} != binary {want:016x} (stale PTX)");
        Ok(TpfKernels { cvt: f("d41tpf_bf16_f32")?, expand: f("d41tpf_expand_hc")?, pre: f("d41tpf_fill_pre")?, ring: f("d41tpf_ring_lin")?,
                        tap: f("d41fw_tap")?, xenc: f("d41fw_xenc")?, xdec: f("d41fw_xdec")?,
                        swa: f("d41fw_swa_rows")?, comp: f("d41fw_compress_seg")?, stash: f("d41fw_stash_seg")?, idxk: f("d41fw_index_k_seg")?, pen: f("d41se_pen")? })
    }
    /// bf16 [n] -> f32 [n]
    pub fn bf16_f32(&self, s: &CudaStream, inp: u64, out: u64, n: usize) -> Result<()> {
        let a = CvtArgs { inp, out, n: n as i64 };
        go(s, "d41tpf_bf16_f32", cfg(((n as u32).div_ceil(256).min(4096), 1), 256), &a)
    }
    /// f32 [rows][in_ld] -> bf16 [rows][hc][dim] (the hc copies of the embedding)
    pub fn expand_hc(&self, s: &CudaStream, inp: u64, in_ld: usize, out: u64, rows: usize, dim: usize, hc: usize) -> Result<()> {
        let a = ExpandArgs { inp, out, in_ld: in_ld as i64, out_ld: (hc * dim) as i64, rows: rows as i32, dim: dim as i32, hc: hc as i32, pad: 0 };
        go(s, "d41tpf_expand_hc", cfg(((dim as u32).div_ceil(256), rows as u32), 256), &a)
    }
    /// the identity pre-mix (1, 0, ..) [rows][stride]
    pub fn fill_pre(&self, s: &CudaStream, out: u64, stride: usize, rows: usize, hc: usize) -> Result<()> {
        let a = PreArgs { out, stride: stride as i64, rows: rows as i32, hc: hc as i32 };
        go(s, "d41tpf_fill_pre", cfg((((rows * hc) as u32).div_ceil(256), 1), 256), &a)
    }
    /// linear window rows [0, n) <- ring slots of positions [p_lo, p_lo + n)
    #[allow(clippy::too_many_arguments)]
    pub fn ring_lin(&self, s: &CudaStream, ring_q: u64, ring_s: u64, lin_q: u64, lin_s: u64, p_lo: i64, n: usize, slots: usize, q_bytes: usize, s_bytes: usize) -> Result<()> {
        if n == 0 { return Ok(()); }
        let a = RingArgs { ring_q, ring_s, lin_q, lin_s, p_lo, n: n as i32, slots: slots as i32, q_bytes: q_bytes as i32, s_bytes: s_bytes as i32 };
        go(s, "d41tpf_ring_lin", cfg((n as u32, 1), 256), &a)
    }
    /// leg FW: the DSpark tap of stream rows [row0, row0 + n) (bf16 [rows][hc][dim]) -> dst rows dst0.. at column `col`
    #[allow(clippy::too_many_arguments)]
    pub fn tap(&self, s: &CudaStream, x: u64, hc: usize, dim: usize, row0: usize, n: usize, dst: u64, dst0: usize, dst_ld: usize, col: usize) -> Result<()> {
        if n == 0 { return Ok(()); }
        let a = TapArgs { x, dst, dst_ld: dst_ld as i64, hc: hc as i32, dim: dim as i32, row0: row0 as i32, n: n as i32, dst0: dst0 as i32, col: col as i32 };
        go(s, "d41fw_tap", cfg(((dim as u32).div_ceil(256), n as u32), 256), &a)
    }
    /// leg SE: penalties on one row of this rank's logits (see `d41se_pen`)
    pub fn penalty(&self, s: &CudaStream, a: &PenArgs) -> Result<()> {
        go(s, "d41se_pen", cfg(((a.vlocal as u32).div_ceil(256), 1), 256), a)
    }
    /// leg FW: `nwords` 32-bit words -> this rank's slot of the [world][2 * nwords] f32 gather frame (others zero)
    pub fn xenc(&self, s: &CudaStream, inp: u64, out: u64, nwords: usize, rank: usize, world: usize) -> Result<()> {
        let a = XencArgs { inp, out, nwords: nwords as i32, rank: rank as i32, world: world as i32, pad: 0 };
        let n = (2 * nwords * world) as u32;
        go(s, "d41fw_xenc", cfg((n.div_ceil(256), 1), 256), &a)
    }
    /// leg FW (graph mode): the window K/V of every row into its lane's ring at the device position
    pub fn swa_rows(&self, s: &CudaStream, a: &SwaRowsArgs) -> Result<()> {
        go(s, "d41fw_swa_rows", cfg((a.rows as u32, 1), 128), a)
    }
    /// leg FW (graph mode): one lane segment's compressor with p0 = *pos0 (grid = the most groups the segment completes)
    pub fn compress_seg(&self, s: &CudaStream, a: &crate::dsv41::gpu::kv::CompressArgs, pos0: u64, gmax: usize) -> Result<()> {
        if gmax == 0 { return Ok(()); }
        go2(s, "d41fw_compress_seg", cfg((gmax as u32, 1), 128), a, &pos0)
    }
    /// leg FW (graph mode): the frontier stash with p0 = *pos0 (`n` = min(rows, ring rows) blocks)
    pub fn stash_seg(&self, s: &CudaStream, a: &crate::dsv41::gpu::kv::StashArgs, pos0: u64, n: usize) -> Result<()> {
        if n == 0 { return Ok(()); }
        go2(s, "d41fw_stash_seg", cfg((n as u32, 1), 128), a, &pos0)
    }
    /// leg FW (graph mode): the index-K planes of the groups a segment of `rows` rows at p0 = *pos0 completes
    pub fn index_k_seg(&self, s: &CudaStream, a: &crate::dsv41::gpu::kv::IndexArgs, pos0: u64, rows: usize, gmax: usize) -> Result<()> {
        if gmax == 0 { return Ok(()); }
        let c = LaunchConfig { grid_dim: ((gmax as u32).div_ceil(4), 1, 1), block_dim: (32, 4, 1), shared_mem_bytes: 0 };
        { let r = rows as i32; let mut p = [a as *const _ as *mut c_void, &pos0 as *const u64 as *mut c_void, &r as *const i32 as *mut c_void];
          crate::dsv41::gpu::launch_raw("d41fw_index_k_seg", c.grid_dim, c.block_dim, 0, s.stream as u64, crate::dsv41::gpu::pdl_on(), &mut p) }
    }
    /// leg PG1 (graph keys by total rows): lane `slot`'s compressor with its (row0, rows) read from `seg` (i32 [lanes][2])
    /// and its positions from `pos` + row0; `gmax` = the most groups any segment of at most `graph_seg_max` rows completes.
    pub fn compress_tab(&self, s: &CudaStream, a: &crate::dsv41::gpu::kv::CompressArgs, pos: u64, seg: u64, slot: usize, gmax: usize) -> Result<()> {
        if gmax == 0 { return Ok(()); }
        let sl = slot as i32;
        let mut p = [a as *const _ as *mut c_void, &pos as *const u64 as *mut c_void, &seg as *const u64 as *mut c_void, &sl as *const i32 as *mut c_void];
        crate::dsv41::gpu::launch_raw("d41fw_compress_tab", (gmax as u32, 1, 1), (128, 1, 1), 0, s.stream as u64, crate::dsv41::gpu::pdl_on(), &mut p)
    }
    /// leg PG1: lane `slot`'s frontier stash (grid = the ring rows; blocks past the segment's rows return)
    pub fn stash_tab(&self, s: &CudaStream, a: &crate::dsv41::gpu::kv::StashArgs, pos: u64, seg: u64, slot: usize, n: usize) -> Result<()> {
        if n == 0 { return Ok(()); }
        let sl = slot as i32;
        let mut p = [a as *const _ as *mut c_void, &pos as *const u64 as *mut c_void, &seg as *const u64 as *mut c_void, &sl as *const i32 as *mut c_void];
        crate::dsv41::gpu::launch_raw("d41fw_stash_tab", (n as u32, 1, 1), (128, 1, 1), 0, s.stream as u64, crate::dsv41::gpu::pdl_on(), &mut p)
    }
    /// leg PG1: lane `slot`'s index-K planes (grid = ceil(gmax / 4) blocks of (32, 4))
    pub fn index_k_tab(&self, s: &CudaStream, a: &crate::dsv41::gpu::kv::IndexArgs, pos: u64, seg: u64, slot: usize, gmax: usize) -> Result<()> {
        if gmax == 0 { return Ok(()); }
        let sl = slot as i32;
        let mut p = [a as *const _ as *mut c_void, &pos as *const u64 as *mut c_void, &seg as *const u64 as *mut c_void, &sl as *const i32 as *mut c_void];
        crate::dsv41::gpu::launch_raw("d41fw_index_k_tab", ((gmax as u32).div_ceil(4), 1, 1), (32, 4, 1), 0, s.stream as u64, crate::dsv41::gpu::pdl_on(), &mut p)
    }
    /// leg FW: the reduced [world][2 * nwords] frame -> [world][nwords] words
    pub fn xdec(&self, s: &CudaStream, inp: u64, out: u64, nwords: usize, world: usize) -> Result<()> {
        let a = XdecArgs { inp, out, nwords: nwords as i32, world: world as i32 };
        let n = (nwords * world) as u32;
        go(s, "d41fw_xdec", cfg((n.div_ceil(256), 1), 256), &a)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn dsv41_tpf_arg_layouts() {
        assert_eq!(std::mem::size_of::<super::CvtArgs>(), 24);
        assert_eq!(std::mem::size_of::<super::ExpandArgs>(), 48);
        assert_eq!(std::mem::size_of::<super::PreArgs>(), 24);
        assert_eq!(std::mem::size_of::<super::RingArgs>(), 56);
        assert_eq!(std::mem::size_of::<super::TapArgs>(), 48);
        assert_eq!(std::mem::size_of::<super::XencArgs>(), 32);
        assert_eq!(std::mem::size_of::<super::XdecArgs>(), 24);
        assert_eq!(std::mem::size_of::<super::SwaRowsArgs>(), 64);
    }
}
