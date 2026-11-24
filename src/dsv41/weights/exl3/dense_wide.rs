//! KE2 / P7PF: host side of the dense wide path (`kernels/gpu_dsv41/17_exl3_dense_wide.cuh`): rows > 32.
//! `d41x_dw_prep8` -> tensor-core 2-CTA/SM 64x128 tile GEMM with trellis decoded per warp (`d41x_dw_tr`, integer K) or,
//! the reconstruct arm, from the slice decoded once into fp16 (`d41x_dw_recon` + `d41x_dw_f16`, 100% bit-identical to
//! `d41x_dw_tr` for integer K, and required for fractional K), or `--dsv41-pf-wide cublas` (`d41x_dw_recon` + cached
//! `cublasLtMatmul` FP16->FP32 + `d41x_dw_post`). Epilogue H128 + svh + trim + dtype + accumulate.

use super::dense::{resolve, Resolved};
use super::launch::{act_code, kernel, Kernel};
use super::moe_dec::launch_ex;
use crate::dsv41::weights::{Accum, Act, ActDtype, DenseWeight, Launch, RowClass, Scratch};
use anyhow::{bail, ensure, Context, Result};
use cudarc::cublaslt::result as lt;
use cudarc::cublaslt::sys as ltsys;
use cudarc::driver::sys;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

/// Which B / GEMM source serves a wide call (`--dsv41-pf-wide auto|trellis|recon|cublas`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WideSrc { Auto, Trellis, Recon, Cublas }

/// Parse a `--dsv41-pf-wide` value (`auto`, `trellis`, `recon`, `cublas`).
pub fn parse_wide_src(v: Option<&str>) -> WideSrc {
    match v {
        Some("trellis") => WideSrc::Trellis,
        Some("recon") => WideSrc::Recon,
        Some("cublas") => WideSrc::Cublas,
        _ => WideSrc::Auto,
    }
}

/// Process-wide resolved `--dsv41-pf-wide` (`Scope::Spmd`: rides `TpConfig` to every TP rank).
pub fn configured_src() -> WideSrc {
    parse_wide_src(crate::opts::get(crate::opt!("dsv41-pf-wide")).as_deref())
}

/// `kernels/gpu_dsv41/17_exl3_dense_wide.cuh` `D41xDwArgs`.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct DwArgs {
    pub tr: u64, pub xh: u64, pub svh: u64,
    pub out: u64, pub ocode: i32, pub ldo: i32, pub accum: i32,
    pub rows: i32, pub lda: i32, pub kb0: i32, pub nkb: i32, pub nb_stride: i32, pub nb0: i32, pub nblk: i32, pub skip: i32, pub ncols: i32, pub bits: i32,
    pub w16_ld: i32,
}

const ASTG: usize = 128 * 128;
const STAGE_EPI: usize = 128 * 132 * 4;
const ASTG64: usize = 64 * 128;
const EPI64: usize = 64 * 132 * 4;
const LT_WS_BYTES: usize = 32 << 20;

fn a256(b: usize) -> usize { (b + 255) & !255 }

/// Measured `.16` crossover between 2-CTA/SM `d41x_dw_tr` and 2-CTA/SM `d41x_dw_recon` + `d41x_dw_f16` (both 100%
/// bit-identical): fractional K (`bits >= 17`) always reconstructs; integer K >= 5 (`EAQ v1` attention / head widths)
/// wins with reconstruct at `rows >= 4096` and `nblk >= 8` (+8%..+20% at 8K rows), and K >= 3 at `rows >= 12288` with
/// `nblk >= 32` (TP=4 16K prefill chunks). Below that, `d41x_dw_tr` (2 CTAs/SM, L2-swizzled) wins with 0 W16T scratch.
pub fn auto_uses_recon(bits: i32, rows: usize, nblk: usize) -> bool {
    bits >= 17 || (nblk >= 8 && ((bits >= 5 && rows >= 4096) || (bits >= 3 && rows >= 12288 && nblk >= 32)))
}

fn wants_recon(bits: i32, rows: usize, nblk: usize, src: WideSrc) -> bool {
    match src {
        WideSrc::Auto => auto_uses_recon(bits, rows, nblk),
        WideSrc::Trellis => bits >= 17,
        WideSrc::Recon | WideSrc::Cublas => true,
    }
}

/// (xh bytes, extra W16T [+ yraw + cuBLASLt workspace] bytes) of a wide call.
pub fn scratch_parts(r: &Resolved, rows: usize, src: WideSrc) -> (usize, usize) {
    let kloc = r.plan.kloc;
    let xb = a256(rows * kloc * 2);
    let wb = a256(r.plan.nblk * 128 * kloc * 2);
    let extra = match src {
        WideSrc::Cublas => wb + a256(rows * r.plan.nblk * 128 * 4) + LT_WS_BYTES,
        _ if wants_recon(r.bits, rows, r.plan.nblk, src) => wb,
        _ => 0,
    };
    (xb, extra)
}

pub fn scratch_bytes(w: &DenseWeight, rows: usize, src: WideSrc) -> Result<usize> {
    let r = resolve(w, rows)?;
    let (a, b) = scratch_parts(&r, rows, src);
    Ok(a + b)
}

pub fn check_geometry() -> Result<()> {
    static OK: OnceLock<()> = OnceLock::new();
    if OK.get().is_some() { return Ok(()); }
    let k = kernel("d41x_dw_sizes")?;
    let mut h = [0i32; 4];
    unsafe {
        let mut d: sys::CUdeviceptr = 0;
        ensure!(sys::cuMemAlloc_v2(&mut d, 16) == sys::CUresult::CUDA_SUCCESS, "cuMemAlloc (dw sizes)");
        let mut st: sys::CUstream = std::ptr::null_mut();
        ensure!(sys::cuStreamCreate(&mut st, 0) == sys::CUresult::CUDA_SUCCESS, "cuStreamCreate (dw sizes)");
        let mut p = d as u64;
        let mut params = [&mut p as *mut u64 as *mut c_void];
        ensure!(sys::cuLaunchKernel(k.0, 1, 1, 1, 32, 1, 1, 0, st, params.as_mut_ptr(), std::ptr::null_mut()) == sys::CUresult::CUDA_SUCCESS, "launch d41x_dw_sizes");
        ensure!(sys::cuStreamSynchronize(st) == sys::CUresult::CUDA_SUCCESS, "sync (dw sizes)");
        ensure!(sys::cuMemcpyDtoH_v2(h.as_mut_ptr() as *mut c_void, d, 16) == sys::CUresult::CUDA_SUCCESS, "read dw sizes");
        sys::cuStreamDestroy_v2(st);
        sys::cuMemFree_v2(d);
    }
    ensure!(h[0] as usize == std::mem::size_of::<DwArgs>() && h[1] as usize == ASTG && h[2] as usize == STAGE_EPI, "dense wide geometry mismatch {:?} (Rust args {} B)", h, std::mem::size_of::<DwArgs>());
    let _ = OK.set(());
    Ok(())
}

/// Dynamic smem opt-in. Monotonic per function so a smaller request never lowers an earlier larger limit.
fn opt_in(name: &str, smem: u32) -> Result<Kernel> {
    static SET: OnceLock<Mutex<HashMap<String, u32>>> = OnceLock::new();
    let k = kernel(name)?;
    let m = SET.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = m.lock().unwrap();
    let cur = g.get(name).copied().unwrap_or(0);
    if smem > cur {
        let r = unsafe { sys::cuFuncSetAttribute(k.0, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuFuncSetAttribute({name}, smem {smem}): {r:?}");
        g.insert(name.to_string(), smem);
    }
    Ok(k)
}

struct DwLtPlan {
    algo: ltsys::cublasLtMatmulAlgo_t,
    desc: ltsys::cublasLtMatmulDesc_t,
    la: ltsys::cublasLtMatrixLayout_t,
    lb: ltsys::cublasLtMatrixLayout_t,
    lc: ltsys::cublasLtMatrixLayout_t,
}
unsafe impl Send for DwLtPlan {}

fn lt_handle() -> Result<ltsys::cublasLtHandle_t> {
    static HANDLE: OnceLock<usize> = OnceLock::new();
    if let Some(&h) = HANDLE.get() { return Ok(h as ltsys::cublasLtHandle_t); }
    let h = lt::create_handle().map_err(|e| anyhow::anyhow!("cublasLtCreate: {e:?}"))?;
    let _ = HANDLE.set(h as usize);
    Ok(*HANDLE.get().unwrap() as ltsys::cublasLtHandle_t)
}

fn build_lt_plan(handle: ltsys::cublasLtHandle_t, k: usize, n: usize, m: usize, ws_bytes: usize) -> Result<DwLtPlan> {
    let desc = lt::create_matmul_desc(ltsys::cublasComputeType_t::CUBLAS_COMPUTE_32F, ltsys::cudaDataType_t::CUDA_R_32F)
        .map_err(|e| anyhow::anyhow!("cublasLt create_matmul_desc ({k},{n},{m}): {e:?}"))?;
    let (op_t, op_n): (u32, u32) = (1, 0);   // col-major C[n,m] = W16T_col[k,n]^T * xh_col[k,m]
    unsafe {
        lt::set_matmul_desc_attribute(desc, ltsys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, &op_t as *const _ as *const _, 4)
            .map_err(|e| anyhow::anyhow!("cublasLt set TRANSA: {e:?}"))?;
        lt::set_matmul_desc_attribute(desc, ltsys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB, &op_n as *const _ as *const _, 4)
            .map_err(|e| anyhow::anyhow!("cublasLt set TRANSB: {e:?}"))?;
    }
    let la = lt::create_matrix_layout(ltsys::cudaDataType_t::CUDA_R_16F, k as u64, n as u64, k as i64)
        .map_err(|e| anyhow::anyhow!("cublasLt layout A: {e:?}"))?;
    let lb = lt::create_matrix_layout(ltsys::cudaDataType_t::CUDA_R_16F, k as u64, m as u64, k as i64)
        .map_err(|e| anyhow::anyhow!("cublasLt layout B: {e:?}"))?;
    let lc = lt::create_matrix_layout(ltsys::cudaDataType_t::CUDA_R_32F, n as u64, m as u64, n as i64)
        .map_err(|e| anyhow::anyhow!("cublasLt layout C: {e:?}"))?;
    let pref = lt::create_matmul_pref().map_err(|e| anyhow::anyhow!("cublasLt create_matmul_pref: {e:?}"))?;
    unsafe {
        lt::set_matmul_pref_attribute(pref, ltsys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes as *const _ as *const _, 8)
            .map_err(|e| anyhow::anyhow!("cublasLt set MAX_WORKSPACE: {e:?}"))?;
    }
    let heur = lt::get_matmul_algo_heuristic(handle, desc, la, lb, lc, lc, pref);
    unsafe { let _ = lt::destroy_matmul_pref(pref); }
    match heur {
        Ok(h) => Ok(DwLtPlan { algo: h.algo, desc, la, lb, lc }),
        Err(e) => {
            unsafe {
                let _ = lt::destroy_matrix_layout(lc);
                let _ = lt::destroy_matrix_layout(lb);
                let _ = lt::destroy_matrix_layout(la);
                let _ = lt::destroy_matmul_desc(desc);
            }
            bail!("cublasLt heuristic ({k},{n},{m}): {e:?}");
        }
    }
}

fn run_lt_matmul(stream: u64, w16: u64, xh: u64, yraw: u64, k: usize, n: usize, m: usize, ws: u64, ws_bytes: usize) -> Result<()> {
    ensure!(stream != 0, "cublasLt never runs on the legacy default stream");
    static PLANS: OnceLock<Mutex<HashMap<(usize, usize, usize, usize), DwLtPlan>>> = OnceLock::new();
    let handle = lt_handle()?;
    let cache = PLANS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = cache.lock().unwrap();
    let key = (k, n, m, ws_bytes);
    if !g.contains_key(&key) {
        let plan = build_lt_plan(handle, k, n, m, ws_bytes)?;
        g.insert(key, plan);
    }
    let plan = g.get(&key).unwrap();
    let (alpha, beta): (f32, f32) = (1.0, 0.0);
    unsafe {
        lt::matmul(handle, plan.desc,
            &alpha as *const _ as *const c_void,
            &beta as *const _ as *const c_void,
            w16 as *const c_void, plan.la,
            xh as *const c_void, plan.lb,
            yraw as *mut c_void, plan.lc,
            yraw as *mut c_void, plan.lc,
            &plan.algo, ws as *mut c_void, ws_bytes,
            stream as usize as ltsys::cudaStream_t)
    }.with_context(|| format!("cublasLt matmul k={k} n={n} m={m}"))
}

fn launch_recon(l: &Launch, ra: &mut DwArgs, nblk: usize, nkb: usize, bits: i32) -> Result<()> {
    let grid = if bits >= 17 {
        ((nkb * nblk * 8).div_ceil(8) as u32, 1, 1)
    } else {
        (nblk as u32, nkb.div_ceil(4) as u32, 1)
    };
    launch_ex(&kernel("d41x_dw_recon")?, grid, 256, 0, l, &mut [ra as *mut DwArgs as *mut c_void]).context("d41x_dw_recon")
}

/// Resolve effective arm given available `s.bytes`: `Auto` gracefully falls back to `Trellis` when `s.bytes < xb + wb`
/// (for integer widths) so tight TP=2 Rank-0 scratch never overflows even if a caller sized for `xh` alone.
pub fn resolve_arm(bits: i32, rows: usize, nblk: usize, xb: usize, wb: usize, yraw_b: usize, scratch_avail: usize, src: WideSrc) -> Result<WideSrc> {
    match src {
        WideSrc::Cublas => {
            ensure!(scratch_avail >= xb + wb + yraw_b, "cublas wide scratch {scratch_avail} < {}", xb + wb + yraw_b);
            Ok(WideSrc::Cublas)
        }
        WideSrc::Recon => {
            ensure!(scratch_avail >= xb + wb, "recon wide scratch {scratch_avail} < {}", xb + wb);
            Ok(WideSrc::Recon)
        }
        WideSrc::Trellis => {
            if bits >= 17 {
                ensure!(scratch_avail >= xb + wb, "fractional wide scratch {scratch_avail} < {}", xb + wb);
                Ok(WideSrc::Recon)
            } else {
                ensure!(scratch_avail >= xb, "trellis wide scratch {scratch_avail} < {xb}");
                Ok(WideSrc::Trellis)
            }
        }
        WideSrc::Auto => {
            if bits >= 17 {
                ensure!(scratch_avail >= xb + wb, "fractional wide scratch {scratch_avail} < {}", xb + wb);
                Ok(WideSrc::Recon)
            } else if auto_uses_recon(bits, rows, nblk) && scratch_avail >= xb + wb {
                Ok(WideSrc::Recon)
            } else {
                ensure!(scratch_avail >= xb, "wide scratch {scratch_avail} < {xb}");
                Ok(WideSrc::Trellis)
            }
        }
    }
}

pub fn dense_wide(l: &Launch, w: &DenseWeight, x: &Act, y: &Act, acc: Accum, s: Scratch, src: WideSrc) -> Result<RowClass> {
    let rows = x.rows;
    ensure!(rows >= 1 && rows <= 65535 && y.rows == rows, "{}: wide rows {rows}", w.key);
    let r = resolve(w, rows)?;
    let p = &r.plan;
    ensure!(x.cols == p.kloc && y.cols == p.ncols, "{}: x cols {} (want {}), y cols {} (want {})", w.key, x.cols, p.kloc, y.cols, p.ncols);
    ensure!(y.dtype != ActDtype::Fp8E4m3, "FP8 output");
    check_geometry()?;
    let xb = a256(rows * p.kloc * 2);
    let wb = a256(p.nblk * 128 * p.kloc * 2);
    let yraw_b = a256(rows * p.nblk * 128 * 4);
    let arm = resolve_arm(r.bits, rows, p.nblk, xb, wb, yraw_b, s.bytes, src).with_context(|| w.key.clone())?;
    let xh = s.ptr.0;
    let w16 = s.ptr.0 + xb as u64;
    // prep: xh = fp16(H128(x * suh)), 8 warps x 4 rows per CTA
    let k = kernel("d41x_dw_prep8")?;
    let (mut xp, mut xc, mut ldx, mut suh, mut rws, mut xhp, mut lda) = (x.ptr.0, act_code(x.dtype)?, x.ld as i32, r.suh, rows as i32, xh, p.kloc as i32);
    launch_ex(&k, ((p.kloc / 128) as u32, rows.div_ceil(32) as u32, 1), 256, 0, l, &mut [&mut xp as *mut _ as *mut c_void, &mut xc as *mut _ as *mut c_void, &mut ldx as *mut _ as *mut c_void,
        &mut suh as *mut _ as *mut c_void, &mut rws as *mut _ as *mut c_void, &mut xhp as *mut _ as *mut c_void, &mut lda as *mut _ as *mut c_void]).context("d41x_dw_prep8")?;
    let mut a = DwArgs {
        tr: r.tr, xh, svh: r.svh, out: y.ptr.0, ocode: act_code(y.dtype)?, ldo: y.ld as i32, accum: (acc == Accum::Add) as i32,
        rows: rows as i32, lda: p.kloc as i32, kb0: r.kb0, nkb: r.nkb, nb_stride: r.nb_stride, nb0: r.nb0, nblk: p.nblk as i32,
        skip: p.skip as i32, ncols: p.ncols as i32, bits: r.bits, w16_ld: p.kloc as i32,
    };
    match arm {
        WideSrc::Cublas => {
            let mut ra = DwArgs { out: w16, ..a };
            launch_recon(l, &mut ra, p.nblk, r.nkb as usize, r.bits)?;
            let yraw = w16 + wb as u64;
            let ws_off = xb + wb + yraw_b;
            let ws_len = s.bytes.saturating_sub(ws_off).min(LT_WS_BYTES) & !255;
            let ws_ptr = if ws_len > 0 { s.ptr.0 + ws_off as u64 } else { 0 };
            run_lt_matmul(l.stream, w16, xh, yraw, p.kloc, p.nblk * 128, rows, ws_ptr, ws_len)?;
            let mut yp = yraw;
            launch_ex(&kernel("d41x_dw_post")?, (p.nblk as u32, rows.div_ceil(32) as u32, 1), 256, 0, l,
                      &mut [&mut yp as *mut u64 as *mut c_void, &mut a as *mut DwArgs as *mut c_void]).context("d41x_dw_post")?;
        }
        WideSrc::Recon => {
            let mut ra = DwArgs { out: w16, ..a };
            launch_recon(l, &mut ra, p.nblk, r.nkb as usize, r.bits)?;
            a.tr = w16;
            let smem = (2 * (ASTG64 + 128 * 128)) as u32;
            let k = opt_in("d41x_dw_f16", smem)?;
            launch_ex(&k, (p.nblk as u32, rows.div_ceil(64) as u32, 1), 256, smem, l, &mut [&mut a as *mut DwArgs as *mut c_void]).context("d41x_dw_f16")?;
        }
        WideSrc::Trellis | WideSrc::Auto => {
            let smem = (2 * (ASTG64 + 4 * 8 * 32 * r.bits as usize)).max(EPI64) as u32;
            let k = opt_in("d41x_dw_tr", smem)?;
            launch_ex(&k, (p.nblk as u32, rows.div_ceil(64) as u32, 1), 256, smem, l, &mut [&mut a as *mut DwArgs as *mut c_void]).context("d41x_dw_tr")?;
        }
    }
    Ok(RowClass::Wide)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv41_p7pf_wide_src_parse_and_registry_spmd() {
        assert_eq!(parse_wide_src(None), WideSrc::Auto);
        assert_eq!(parse_wide_src(Some("auto")), WideSrc::Auto);
        assert_eq!(parse_wide_src(Some("trellis")), WideSrc::Trellis);
        assert_eq!(parse_wide_src(Some("recon")), WideSrc::Recon);
        assert_eq!(parse_wide_src(Some("cublas")), WideSrc::Cublas);
        let id = crate::opts::lookup("dsv41-pf-wide").expect("--dsv41-pf-wide registered");
        let o = id.def();
        assert_eq!((o.env, o.def, o.scope, o.cat), ("", "auto", crate::opts::Scope::Spmd, crate::opts::Cat::Dsv4));
    }

    #[test]
    fn dsv41_p7pf_auto_crossover_and_graceful_scratch_fallback() {
        // Fractional K always reconstructs; integer K < 5 stays trellis at 8K; K >= 5 reconstructs at >= 4096 rows when nblk >= 8
        assert!(auto_uses_recon(17, 33, 4));
        assert!(!auto_uses_recon(3, 2048, 128));
        assert!(!auto_uses_recon(3, 8192, 128));
        assert!(auto_uses_recon(3, 16384, 64));
        assert!(!auto_uses_recon(5, 2048, 128));
        assert!(auto_uses_recon(5, 8192, 128));
        assert!(!auto_uses_recon(5, 8192, 4)); // wkv(512): nblk=4 stays trellis
        // Graceful fallback in Auto when scratch holds xh but not xh + w16
        let (xb, wb, yb) = (256 * 1024, 1024 * 1024, 2 * 1024 * 1024);
        assert_eq!(resolve_arm(5, 8192, 128, xb, wb, yb, xb + wb, WideSrc::Auto).unwrap(), WideSrc::Recon);
        assert_eq!(resolve_arm(5, 8192, 128, xb, wb, yb, xb, WideSrc::Auto).unwrap(), WideSrc::Trellis);
        assert!(resolve_arm(5, 8192, 128, xb, wb, yb, xb - 256, WideSrc::Auto).is_err());
        assert!(resolve_arm(5, 8192, 128, xb, wb, yb, xb, WideSrc::Recon).is_err());
        assert_eq!(resolve_arm(5, 8192, 128, xb, wb, yb, xb + wb + yb, WideSrc::Cublas).unwrap(), WideSrc::Cublas);
    }
}
