//! KE: raw driver launches of the `gpu_dsv41` band-10 kernels (handles from `gpu::raw_fn`), the `#[repr(C)]`
//! mirrors of the kernels' by-value argument structs, and the shape-only launch planner.
//! Every launch goes on the caller's stream (`Launch::stream`, a raw CUstream; never 0).

use crate::dsv41::weights::{Act, ActDtype, DevPtr};
use anyhow::{bail, ensure, Result};
use cudarc::driver::sys;
use std::ffi::c_void;

/// Entry kernels of band 10 (`kernels/gpu_dsv41/1x_*.cuh`); the gpu_dsv41 loader collects them.
pub const FNS: &[&str] = &["d41x_gemm_m1", "d41x_gemm_m2", "d41x_gemm_m4", "d41x_gemm_m8", "d41x_prep", "d41x_post",
    "d41x_moe_route", "d41x_moe_prep1", "d41x_moe_gemm1", "d41x_moe_act", "d41x_moe_gemm2", "d41x_moe_post2",
    "d41x_plain_gemv", "d41x_plain_gemv_rg", "d41x_plain_gemv_w", "d41x_gather",
    // KE2 decode MoE (15_exl3_moe_dec.cuh, host side moe_dec.rs)
    "d41x_md_prep", "d41x_md_sizes",
    "d41x_md_gu_a8_k5", "d41x_md_gu_a16_k5", "d41x_md_gu_a32_k5", "d41x_md_gu_a8_k8", "d41x_md_gu_a16_k8", "d41x_md_gu_a32_k8",
    "d41x_md_dn_a8_k5", "d41x_md_dn_a16_k5", "d41x_md_dn_a32_k5", "d41x_md_dn_a8_k8", "d41x_md_dn_a16_k8", "d41x_md_dn_a32_k8",
    // PM1 short-CTA decode MoE (19_exl3_moe_cp.cuh, host side moe_cp.rs)
    "d41x_mc_prep", "d41x_mc_sizes", "d41x_mc_gu_a8", "d41x_mc_gu_a16", "d41x_mc_gu_a32", "d41x_mc_dn_a8", "d41x_mc_dn_a16", "d41x_mc_dn_a32",
    // KE2 prefill MoE (16_exl3_moe_wide.cuh, host side moe_wide.rs)
    "d41x_mp_plan1", "d41x_mp_plan2", "d41x_mp_plan3", "d41x_mp_gu_x2", "d41x_mp_gu_x2h", "d41x_mp_gu_x4", "d41x_mp_dn", "d41x_mp_combine", "d41x_mp_sizes",
    // KE2 / P7PF dense wide (17_exl3_dense_wide.cuh, host side dense_wide.rs)
    "d41x_dw_tr", "d41x_dw_f16", "d41x_dw_recon", "d41x_dw_prep8", "d41x_dw_post", "d41x_dw_sizes",
    // PK4 dense decode (18_exl3_dense_dec.cuh, host side dense_dec.rs): one GEMM entry per (rows class, K)
    "d41x_dd_prep", "d41x_dd_sizes", "d41x_l2_prefetch", "d41x_dd_normrot", "d41x_l2_paced",
    "d41x_dd_a8_k2", "d41x_dd_a16_k2", "d41x_dd_a24_k2", "d41x_dd_a32_k2", "d41x_dd_a8_k3", "d41x_dd_a16_k3", "d41x_dd_a24_k3", "d41x_dd_a32_k3",
    "d41x_dd_a8_k4", "d41x_dd_a16_k4", "d41x_dd_a24_k4", "d41x_dd_a32_k4", "d41x_dd_a8_k5", "d41x_dd_a16_k5", "d41x_dd_a24_k5", "d41x_dd_a32_k5",
    "d41x_dd_a8_k6", "d41x_dd_a16_k6", "d41x_dd_a24_k6", "d41x_dd_a32_k6", "d41x_dd_a8_k7", "d41x_dd_a16_k7", "d41x_dd_a24_k7", "d41x_dd_a32_k7",
    "d41x_dd_a8_k8", "d41x_dd_a16_k8", "d41x_dd_a24_k8", "d41x_dd_a32_k8"];

/// kernels/gpu_dsv41/12_exl3_dense.cuh `D41xGemmArgs`.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct GemmArgs {
    pub tr: u64, pub xh: u64, pub y: u64, pub gstride: i64,
    pub nb_stride: i32, pub nb0: i32, pub nbn: i32, pub kb0: i32, pub nkb: i32,
    pub lda: i32, pub rows: i32, pub ldy: i32, pub ks: i32, pub g: i32, pub bits: i32,
}

/// 13_exl3_moe.cuh `D41xExpert`.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
/// leg SX1: `k1` / `k3` / `k2` = the width codes of w1 / w3 / w2; every MoE kernel reads each matrix at its own code (a pack
/// may mix them: EAQ v1 2.9 bpw layer 29's shared expert has w2 at K5 under K4 w1 / w3; the SAGE 3.30 bpw pack mixes all three).
pub struct ExpertEntry { pub tr: [u64; 3], pub suh: [u64; 3], pub svh: [u64; 3], pub nbs13: i32, pub nblk: i32, pub k1: i32, pub k3: i32, pub k2: i32, pub kpad: i32 }

/// 13_exl3_moe.cuh `D41xMoeArgs`.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct MoeArgs {
    pub ex: u64, pub ids: u64, pub wts: u64, pub x: u64, pub xcode: i32, pub ldx: i32,
    pub ctl: u64, pub slot_expert: u64, pub slot_off: u64, pub slot_cnt: u64,
    pub pair_expert: u64, pub pair_row: u64, pub pair_w: u64, pub pair_of: u64, pub row_order: u64,
    pub xh1: u64, pub y1: u64, pub xh2: u64, pub y2: u64,
    pub out: u64, pub ldo: i32, pub accum: i32, pub limit: f32,
    pub n_experts: i32, pub rows: i32, pub topk: i32, pub dim: i32, pub inter_ld: i32, pub nblk_max: i32, pub kmax: i32, pub sms: i32,
}

pub fn act_code(d: ActDtype) -> Result<i32> {
    Ok(match d { ActDtype::Bf16 => 0, ActDtype::F16 => 1, ActDtype::F32 => 2, ActDtype::Fp8E4m3 => bail!("an FP8 activation is not an EXL3 / plain input") })
}

/// Width code of a trellis last dim (16 * K words): 2..8 integer K, 17 / 18 / 19 = K 1.5 / 2.5 / 3.5.
pub fn bits_code(words: u64) -> Result<i32> {
    ensure!(words % 8 == 0, "trellis last dim {words} is not 8 * k2");
    let k2 = (words / 8) as i32;
    Ok(if k2 % 2 == 0 { ensure!((2..=8).contains(&(k2 / 2)), "EXL3 K {} unsupported (2..8)", k2 / 2); k2 / 2 }
       else { ensure!((3..=7).contains(&k2), "EXL3 K {}.5 unsupported (1.5, 2.5, 3.5)", k2 / 2); 16 + k2 / 2 })
}
/// u16 words per 16x16 tile for a width code.
pub fn tile_words(bits: i32) -> u64 { if bits >= 17 { 8 * (2 * (bits as u64 - 16) + 1) } else { 16 * bits as u64 } }

pub struct Kernel(pub sys::CUfunction);
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}
pub fn kernel(name: &str) -> Result<Kernel> {
    match crate::dsv41::gpu::raw_fn(name) { Some(f) => Ok(Kernel(f)), None => bail!("gpu_dsv41 kernel {name} is not in the module (stale PTX or not loaded)") }
}

/// Launch `k` with the given by-value / pointer parameter slots.
pub fn launch(k: &Kernel, grid: (u32, u32, u32), block: u32, stream: u64, params: &mut [*mut c_void]) -> Result<()> {
    ensure!(stream != 0, "dsv41 kernels never run on the legacy default stream (AGENTS 2.1)");
    ensure!(grid.0 > 0 && grid.1 > 0 && grid.2 > 0, "empty grid {grid:?}");
    // leg PF1: the PDL attribute per `--dsv41-pdl` (every gpu_dsv41 entry waits before it touches a predecessor's data)
    let pdl = crate::dsv41::gpu::pdl_on();
    let mut attr = sys::CUlaunchAttribute {
        id: sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
        pad: [0; 4],
        value: sys::CUlaunchAttributeValue { programmaticStreamSerializationAllowed: 1 },
    };
    let mut cfg = sys::CUlaunchConfig {
        gridDimX: grid.0, gridDimY: grid.1, gridDimZ: grid.2, blockDimX: block, blockDimY: 1, blockDimZ: 1,
        sharedMemBytes: 0, hStream: stream as sys::CUstream, attrs: &mut attr, numAttrs: u32::from(pdl),
    };
    let r = unsafe { sys::cuLaunchKernelEx(&mut cfg, k.0, params.as_mut_ptr(), std::ptr::null_mut()) };
    ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuLaunchKernelEx failed: {r:?}");
    Ok(())
}
/// The (ks, g) partition of a GEMM, from the SHAPE only (never the row count), so rows <= 32 share one summation order.
pub fn plan_split(nbn: u64, nkb: u64) -> (i32, i32) {
    const TARGET_WARPS: u64 = 1536;
    let f = ((TARGET_WARPS + nbn - 1) / nbn).min((nkb / 4).max(1));
    if f <= 1 { return (1, 1); }
    let ks = [8u64, 4, 2, 1].into_iter().find(|&k| k <= f).unwrap_or(1);
    (ks as i32, ((f / ks).max(1)) as i32)
}

/// What a dense call looks like on the device (shared by `scratch_bytes` and `dense`).
#[derive(Clone, Copy, Debug)]
pub struct DensePlan {
    pub rows: usize, pub kloc: usize, pub nblk: usize, pub skip: usize, pub ncols: usize,
    pub ks: i32, pub g: i32, pub mt: i32, pub grid: (u32, u32, u32),
    pub xh_bytes: usize, pub y_bytes: usize,
}
impl DensePlan {
    pub fn new(rows: usize, kloc: usize, n_lo: usize, n_hi: usize, tile_nbn: usize) -> DensePlan {
        let nblk_lo = n_lo / 128;
        let nblk = (n_hi + 127) / 128 - nblk_lo;
        let nbn = nblk * 8;
        let _ = tile_nbn;
        let nkb = kloc / 16;
        let (mut ks, mut g) = plan_split(nbn as u64, nkb as u64);
        let mt = if rows <= 16 { 1 } else if rows <= 32 { 2 } else if rows <= 64 { 4 } else { 8 };
        if rows > 32 { ks = 1; g = 1; }
        let ntw = (8 / ks) as usize;
        let z = (rows + 16 * mt as usize - 1) / (16 * mt as usize);
        let grid = (((nbn + ntw - 1) / ntw) as u32, g as u32, z as u32);
        let a256 = |b: usize| (b + 255) & !255;
        DensePlan { rows, kloc, nblk, skip: n_lo - nblk_lo * 128, ncols: n_hi - n_lo, ks, g, mt, grid,
                    xh_bytes: a256(rows * kloc * 2), y_bytes: a256(g as usize * rows * nblk * 128 * 4) }
    }
    pub fn scratch_bytes(&self) -> usize { self.xh_bytes + self.y_bytes }
}

pub fn act_ptr(a: &Act) -> DevPtr { a.ptr }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dsv41_exl3_width_codes_and_tile_words() {
        assert_eq!(bits_code(32).unwrap(), 2);
        assert_eq!(bits_code(128).unwrap(), 8);
        assert_eq!(bits_code(24).unwrap(), 17);   // K = 1.5
        assert_eq!(bits_code(56).unwrap(), 19);   // K = 3.5
        assert!(bits_code(16).is_err() && bits_code(136).is_err() && bits_code(60).is_err());
        assert_eq!(tile_words(3), 48);
        assert_eq!(tile_words(17), 24);
    }
    #[test]
    fn dsv41_exl3_plan_depends_on_the_shape_only_for_up_to_32_rows() {
        for &(nbn, nkb) in &[(80usize, 320usize), (512, 80), (8080, 320), (16, 144)] {
            let base = DensePlan::new(1, nkb * 16, 0, nbn * 16, 0);
            for rows in 2..=32 { let p = DensePlan::new(rows, nkb * 16, 0, nbn * 16, 0); assert_eq!((p.ks, p.g), (base.ks, base.g), "rows {rows}"); }
            let w = DensePlan::new(100, nkb * 16, 0, nbn * 16, 0);
            assert_eq!((w.ks, w.g, w.mt), (1, 1, 8));
        }
    }
}
