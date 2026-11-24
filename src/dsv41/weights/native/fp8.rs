//! DSV4.1 native FP8 dense kernels, host side (leg KN1): launch plumbing and the dispatch plan for
//! `kernels/gpu_dsv41/2x_fp8_*.cuh`. Weight = E4M3 [N][K] + one E8M0 per 32 x 32 block; activation = the
//! reference's `act_quant` (E4M3 + one E8M0 per 32 of a row, scales [rows][K/32]).
//!
//! Dispatch (`plan`, a pure function of the shape, unit-tested without a GPU):
//! * rows 1..=32 -> the split-K GEMV family (`d41_fp8_gemv_*`): tokens ride the MMA's N axis in n8 tiles
//!   (NT = ceil(rows / 8)), a warp-private cp.async ring streams the weights; the K split (KS warps, ascending
//!   inside a warp, fixed-order combine) depends ONLY on (n_count, nkb), so every row count 1..=32 produces
//!   bit-identical rows (the GEMV <= 2 and cooperative classes, incl. the 4-lane x 6-row verify width);
//! * rows >= 33 -> the tensor-core GEMM (`d41_fp8_gemm_*`, 128 x 64 tiles, grouped raster).
//! The GEMM sums K blocks in one ascending chain, so its rows differ from the split-K family in the last
//! bits (both are within fp32 rounding of the reference's chunked scheme).

use super::super::{Accum, Act, ActDtype, DevPtr, Launch, RowClass};
use crate::dsv41::gpu;
use anyhow::{bail, ensure, Result};
use cudarc::driver::sys;

/// Every entry kernel of bands 20-29.
pub const FNS: &[&str] = &[
    "d41_fp8_act_quant_bf16", "d41_fp8_act_quant_f32", "d41_fp8_act_quant_inplace_bf16",
    "d41_fp8_gemm_bf16", "d41_fp8_gemm_f32",
    "d41_fp8_gemv_m1n1k8_bf16", "d41_fp8_gemv_m1n2k8_bf16", "d41_fp8_gemv_m1n3k8_bf16", "d41_fp8_gemv_m1n4k8_bf16",
    "d41_fp8_gemv_m2n1k4_bf16", "d41_fp8_gemv_m2n2k4_bf16", "d41_fp8_gemv_m2n3k4_bf16", "d41_fp8_gemv_m2n4k4_bf16",
    "d41_fp8_gemv_m1n1k8_f32", "d41_fp8_gemv_m1n2k8_f32", "d41_fp8_gemv_m1n3k8_f32", "d41_fp8_gemv_m1n4k8_f32",
    "d41_fp8_gemv_m2n1k4_f32", "d41_fp8_gemv_m2n2k4_f32", "d41_fp8_gemv_m2n3k4_f32", "d41_fp8_gemv_m2n4k4_f32",
];

/// Largest row count served by the split-K GEMV family.
pub const GEMV_FAMILY_MAX_ROWS: usize = 32;
const GEMM_BN: usize = 128;
const GEMM_BM: usize = 64;
const GEMM_SMEM: u32 = 3 * (128 + 64) * 128;

/// `d41_fp8_args` (kernels/gpu_dsv41/20_fp8_dense.cuh), field for field.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Fp8Args {
    pub w: u64, pub ws: u64, pub xq: u64, pub xs: u64, pub y: u64,
    pub ldw: i64, pub ldws: i64, pub ldx: i64, pub ldxs: i64, pub ldy: i64,
    pub n0: i32, pub n_count: i32, pub k0b: i32, pub nkb: i32, pub rows: i32, pub accum: i32, pub gn: i32, pub _pad: i32,
}

/// What `plan` decides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Plan {
    Gemv { name: String, grid: u32, block: u32, smem: u32 },
    Gemm { name: String, grid: u32, gn: i32 },
}

/// The dispatch plan of one call (`f32_out`: f32 vs bf16 output).
pub fn plan(rows: usize, n_count: usize, nkb: usize, f32_out: bool) -> Plan {
    let sfx = if f32_out { "f32" } else { "bf16" };
    if rows <= GEMV_FAMILY_MAX_ROWS {
        let nt = rows.max(1).div_ceil(8);
        let units = nkb.div_ceil(4);
        // (MT, KS) depends on the weight shape only: batch invariance across row counts
        let (mt, ks) = if units < 8 && n_count >= 4096 { (2, 4) } else { (1, 8) };
        let r = mt * 16;
        Plan::Gemv { name: format!("d41_fp8_gemv_m{mt}n{nt}k{ks}_{sfx}"), grid: n_count.div_ceil(r) as u32, block: (ks * 32) as u32, smem: (ks * 3 * r * 128) as u32 }
    } else {
        let tn = n_count.div_ceil(GEMM_BN);
        let tm = rows.div_ceil(GEMM_BM);
        let ng = tn.div_ceil(8);
        Plan::Gemm { name: format!("d41_fp8_gemm_{sfx}"), grid: (tn * tm) as u32, gn: tn.div_ceil(ng) as i32 }
    }
}

/// The class a call of `rows` is served by (kernel family, not the ABI's nominal `RowClass::of`).
pub fn served_class(rows: usize) -> RowClass {
    if rows <= 2 { RowClass::Gemv } else if rows <= GEMV_FAMILY_MAX_ROWS { RowClass::Coop } else { RowClass::Wide }
}

fn align(n: usize, a: usize) -> usize { n.div_ceil(a) * a }
/// Bytes `quantize` needs for `rows` x `k`: E4M3 values then the E8M0 scales (each 256 B aligned).
pub fn quant_bytes(rows: usize, k: usize) -> usize { align(rows * k, 256) + align(rows * (k / 32), 256) }

unsafe fn launch_raw(l: &Launch, name: &str, grid: u32, block: u32, smem: u32, params: &mut [*mut std::ffi::c_void]) -> Result<()> {
    let Some(f) = gpu::raw_fn(name) else { bail!("gpu_dsv41 kernel {name} is not in the loaded module") };
    if smem > 48 * 1024 {
        let r = sys::cuFuncSetAttribute(f, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32);
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuFuncSetAttribute({name}, smem {smem}): {r:?}");
    }
    let mut attr = sys::CUlaunchAttribute {
        id: if l.pdl { sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION } else { sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_IGNORE },
        pad: [0; 4],
        value: if l.pdl { sys::CUlaunchAttributeValue { programmaticStreamSerializationAllowed: 1 } } else { sys::CUlaunchAttributeValue_union { pad: [0; 64] } },
    };
    let mut cfg = sys::CUlaunchConfig {
        gridDimX: grid, gridDimY: 1, gridDimZ: 1, blockDimX: block, blockDimY: 1, blockDimZ: 1,
        sharedMemBytes: smem, hStream: l.stream as sys::CUstream, attrs: &mut attr, numAttrs: if l.pdl { 1 } else { 0 },
    };
    let r = sys::cuLaunchKernelEx(&mut cfg, f, params.as_mut_ptr(), std::ptr::null_mut());
    ensure!(r == sys::CUresult::CUDA_SUCCESS, "launch {name}: {r:?}");
    Ok(())
}

/// `act_quant` of `x` (bf16 or f32 [rows, cols]) into `q` (E4M3, row stride `cols`) and `sc` (E8M0, row stride
/// `cols / 32`), the reference's exact formula. `cols % 32 == 0`.
pub fn quantize(l: &Launch, x: &Act, q: DevPtr, sc: DevPtr) -> Result<()> {
    ensure!(x.cols % 32 == 0, "act_quant: cols {} is not a multiple of 32", x.cols);
    let name = match x.dtype { ActDtype::Bf16 => "d41_fp8_act_quant_bf16", ActDtype::F32 => "d41_fp8_act_quant_f32",
        d => bail!("act_quant: unsupported input dtype {d:?}") };
    if x.rows == 0 { return Ok(()); }
    let (mut xp, mut ldx, mut rows, mut k, mut qp, mut ldq, mut sp, mut lds) =
        (x.ptr.0, x.ld as i64, x.rows as i32, x.cols as i32, q.0, x.cols as i64, sc.0, (x.cols / 32) as i64);
    let mut params = [&mut xp as *mut _ as *mut std::ffi::c_void, &mut ldx as *mut _ as _, &mut rows as *mut _ as _, &mut k as *mut _ as _,
                      &mut qp as *mut _ as _, &mut ldq as *mut _ as _, &mut sp as *mut _ as _, &mut lds as *mut _ as _];
    let groups = x.rows * (x.cols / 32);
    unsafe { launch_raw(l, name, groups.div_ceil(8) as u32, 256, 0, &mut params) }
}

/// The reference's inplace `act_quant` (bf16 -> bf16 of the dequantised value), for the window KV and the WO-A input.
pub fn quantize_inplace(l: &Launch, x: &Act, y: &Act) -> Result<()> {
    ensure!(x.dtype == ActDtype::Bf16 && y.dtype == ActDtype::Bf16 && x.cols % 32 == 0 && y.cols >= x.cols && y.rows >= x.rows, "act_quant inplace: bf16 [rows, cols % 32 == 0] in and out");
    if x.rows == 0 { return Ok(()); }
    let (mut xp, mut ldx, mut rows, mut k, mut yp, mut ldy) = (x.ptr.0, x.ld as i64, x.rows as i32, x.cols as i32, y.ptr.0, y.ld as i64);
    let mut params = [&mut xp as *mut _ as *mut std::ffi::c_void, &mut ldx as *mut _ as _, &mut rows as *mut _ as _, &mut k as *mut _ as _, &mut yp as *mut _ as _, &mut ldy as *mut _ as _];
    unsafe { launch_raw(l, "d41_fp8_act_quant_inplace_bf16", (x.rows * (x.cols / 32)).div_ceil(8) as u32, 256, 0, &mut params) }
}

/// One FP8 GEMM on a QUANTISED activation: output rows [n0, n0 + n_count) and K blocks [k0b, k0b + nkb) of the
/// full weight (`ldw` = the full row length in bytes, `ldws` = full K / 32), y [rows, ldy] with column 0 = n0.
/// `xq` / `xs` hold the rank's local K slice (row strides `nkb * 32` / `nkb`). n0 % 16 == 0. Returns the class.
#[allow(clippy::too_many_arguments)]
pub fn gemm_quantised(l: &Launch, w: DevPtr, ws: DevPtr, ldw: usize, ldws: usize, xq: DevPtr, xs: DevPtr, rows: usize,
                      n0: usize, n_count: usize, k0b: usize, nkb: usize, y: &Act, acc: Accum) -> Result<RowClass> {
    ensure!(n0 % 16 == 0 && n_count % 16 == 0, "fp8 gemm: n0 ({n0}) and n_count ({n_count}) must be multiples of 16");
    ensure!(matches!(y.dtype, ActDtype::Bf16 | ActDtype::F32), "fp8 gemm: output dtype {:?} (bf16 or f32)", y.dtype);
    ensure!(y.rows >= rows && y.cols >= n_count, "fp8 gemm: output [{}, {}] too small for [{rows}, {n_count}]", y.rows, y.cols);
    if rows == 0 || n_count == 0 || nkb == 0 { return Ok(served_class(rows)); }
    let p = plan(rows, n_count, nkb, y.dtype == ActDtype::F32);
    let mut a = Fp8Args { w: w.0, ws: ws.0, xq: xq.0, xs: xs.0, y: y.ptr.0, ldw: ldw as i64, ldws: ldws as i64, ldx: (nkb * 32) as i64, ldxs: nkb as i64,
        ldy: y.ld as i64, n0: n0 as i32, n_count: n_count as i32, k0b: k0b as i32, nkb: nkb as i32, rows: rows as i32,
        accum: (acc == Accum::Add) as i32, gn: 1, _pad: 0 };
    match &p {
        Plan::Gemv { name, grid, block, smem } => { let mut params = [&mut a as *mut _ as *mut std::ffi::c_void]; unsafe { launch_raw(l, name, *grid, *block, *smem, &mut params)?; } }
        Plan::Gemm { name, grid, gn } => { a.gn = *gn; let mut params = [&mut a as *mut _ as *mut std::ffi::c_void]; unsafe { launch_raw(l, name, *grid, 256, GEMM_SMEM, &mut params)?; } }
    }
    Ok(served_class(rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv41_fp8_args_layout_matches_the_kernel() {
        // 5 pointers + 5 i64 + 8 i32, no hidden padding (the .cuh struct has the same fields; the GPU test would
        // read garbage otherwise)
        assert_eq!(std::mem::size_of::<Fp8Args>(), 5 * 8 + 5 * 8 + 8 * 4);
        assert_eq!(std::mem::align_of::<Fp8Args>(), 8);
    }

    #[test]
    fn dsv41_fp8_plan_is_row_invariant_inside_the_gemv_family() {
        // the (MT, KS) K-split must not depend on rows, only the token tiles (NT) may
        for (n, nkb) in [(1280usize, 160usize), (32768, 40), (512, 160), (5120, 256), (32768, 5), (25600, 24), (48, 155)] {
            let mut split = None;
            for rows in 1..=GEMV_FAMILY_MAX_ROWS {
                let Plan::Gemv { name, smem, block, .. } = plan(rows, n, nkb, false) else { panic!("rows {rows} must be GEMV") };
                let core = name.trim_start_matches("d41_fp8_gemv_").trim_end_matches("_bf16").to_string(); // m1n2k8
                let (n_at, k_at) = (core.find('n').unwrap(), core.find('k').unwrap());
                let (mt_ks, nt) = (format!("{}{}", &core[..n_at], &core[k_at..]), rows.div_ceil(8));
                assert_eq!(&core[n_at + 1..k_at], nt.to_string(), "{name} rows {rows}");
                assert!(smem <= 48 * 1024 && block <= 256);
                match &split { None => split = Some(mt_ks), Some(s) => assert_eq!(s, &mt_ks, "K split changed with rows at n={n} nkb={nkb}") }
            }
        }
        assert!(matches!(plan(33, 1280, 160, true), Plan::Gemm { .. }));
        assert_eq!(served_class(1), RowClass::Gemv);
        assert_eq!(served_class(6), RowClass::Coop);
        assert_eq!(served_class(24), RowClass::Coop);
        assert_eq!(served_class(33), RowClass::Wide);
    }

    #[test]
    fn dsv41_fp8_every_planned_kernel_is_listed() {
        for rows in [1usize, 2, 6, 12, 24, 32, 33, 512] {
            for (n, nkb) in [(1280usize, 160usize), (32768, 5)] {
                for f32o in [false, true] {
                    let name = match plan(rows, n, nkb, f32o) { Plan::Gemv { name, .. } | Plan::Gemm { name, .. } => name };
                    assert!(FNS.contains(&name.as_str()), "{name} missing from FNS");
                }
            }
        }
    }
}
