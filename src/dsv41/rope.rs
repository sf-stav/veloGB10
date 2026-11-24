//! DSV4.1 RoPE, ONE builder for every consumer (leg SL, from KI's finding; was three builders: KC
//! `gpu/kv/rope.rs`, KA `gpu/attn.rs`, KI `gpu/idx_ref.rs`). REFb, and DeepSeek's own engine, build
//! `freqs_cis` with torch ON CUDA, so the bits to reproduce are torch-CUDA's (measured on GB10 against
//! `tools/dsv41_sl/rope_truth.py`, PLAN/DSV41_SL_REPORT.md):
//!
//! * the table: `1 / base ** (arange(0, 64, 2) / 64)` with CUDA `powf` and an IEEE reciprocal, then YaRN
//!   for the compressed class: `ramp = (i - low) * fp32(1 / den)` (torch divides a tensor by a Python
//!   scalar as a multiply by the fp32 reciprocal), `f / factor * (1 - smooth) + f * smooth` as separate
//!   ops (no FMA). CUDA's `powf` is not correctly rounded: at theta 10000 it differs from a correctly
//!   rounded pow in entries 11 and 25, so [`table`] computes the table ON THE GPU (`d41f_rope_table`, band
//!   05) and checks it against the torch-CUDA tables of DeepSeek-V4.1-Flash-1017 recorded here;
//!   [`table_host`] is the CPU emulation for code with no GPU (it is exact for the compressed class);
//! * the angle `fp32(pos) * freq` (one rounding), sin / cos = `sincosf` of it (torch.polar on CUDA; the
//!   kernels use `d41_sincos`, the local-memory-free twin, equal on 1M samples to 1M positions);
//! * the rotation as torch's CUDA build contracts the complex multiply: forward `re = fma(a, c, -(b s))`,
//!   `im = fma(b, c, a s)`; inverse (conj) `re = fma(a, c, b s)`, `im = fma(b, c, -(a s))`
//!   (`d41_rope_fwd` / `d41_rope_inv` in kernels/gpu_dsv41/00_abi.cuh; separately rounded products differ
//!   from torch on ~3e-5 of the bf16 results).

use crate::dsv41::config::Dsv41Config;
use anyhow::{ensure, Context, Result};

/// The two RoPE classes (model.py 680-696): compressed layers (and their caches) rotate at
/// compress_rope_theta with YaRN; SWA-only layers and the DSpark stages at rope_theta without.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class { Compressed, Plain }

pub fn class_of(cfg: &Dsv41Config, layer: usize) -> Class {
    if !cfg.is_dspark(layer) && cfg.ratio(layer) > 0 { Class::Compressed } else { Class::Plain }
}

/// The parameters of a table: theta, rotary dims, and YaRN (factor, original context, beta_fast, beta_slow).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params { pub theta: f64, pub dim: usize, pub yarn: Option<(f64, f64, f64, f64)> }

pub fn params(cfg: &Dsv41Config, c: Class) -> Params {
    let r = &cfg.rope;
    match c {
        Class::Compressed => Params { theta: r.compress_theta, dim: cfg.rope_dim, yarn: Some((r.factor, r.original_max_position_embeddings as f64, r.beta_fast, r.beta_slow)) },
        Class::Plain => Params { theta: r.theta, dim: cfg.rope_dim, yarn: None },
    }
}

/// YaRN's band in Python doubles (model.py `corrected_dim`, floor / ceil, clamped), then fp32(1 / den).
fn yarn_band(p: &Params) -> Option<(f32, f32, f32)> {
    let (factor, orig, beta_fast, beta_slow) = p.yarn?;
    if orig <= 0.0 { return None; }
    let dim = p.dim as f64;
    let corrected = |rot: f64| dim * (orig / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * p.theta.ln());
    let low = corrected(beta_fast).floor().max(0.0);
    let high = corrected(beta_slow).ceil().min(dim - 1.0);
    let den = (high - low).max(1e-3) as f32;
    Some((low as f32, 1.0f32 / den, 1.0f32 / factor as f32))
}

/// The table on the host (CPU-only callers). Exact for the compressed class; the plain class's pow is
/// correctly rounded here, CUDA's is not (theta 10000: entries 11 and 25 differ by 1 ulp). GPU code uses [`table`].
pub fn table_host(p: &Params) -> [f32; 32] {
    let mut f = [0f32; 32];
    let inv_dim = 1.0f32 / p.dim as f32;
    for (i, v) in f.iter_mut().enumerate() {
        let e = (2 * i) as f32 * inv_dim;
        *v = 1.0f32 / ((p.theta as f32 as f64).powf(e as f64) as f32);
    }
    if let Some((low, inv_den, inv_factor)) = yarn_band(p) {
        for (i, v) in f.iter_mut().enumerate() {
            let ramp = ((i as f32 - low) * inv_den).clamp(0.0, 1.0);
            let smooth = 1.0f32 - ramp;
            let a = *v * inv_factor;
            let b = 1.0f32 - smooth;
            *v = a * b + *v * smooth;   // separate torch ops: Rust does not contract
        }
    }
    f
}

/// The device kernel's argument block (02_fwd.cuh `d41f_rope_table_args`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct TableArgs { out: u64, base: f32, inv_dim: f32, low: f32, inv_den: f32, inv_factor: f32, yarn: i32, half: i32, pad: i32 }

/// torch-on-CUDA tables of DeepSeek-V4.1-Flash-1017's config (theta 160000 + YaRN 16 / 65536 / 32 / 1, and
/// theta 10000), recorded by `tools/dsv41_sl/rope_truth.py` on GB10 (torch 2.11.0+cu130): the self-check of [`table`].
pub const TRUTH_COMPRESSED: [u32; 32] = [0x3f800000, 0x3f300a3a, 0x3ef21c1f, 0x3ea67d01, 0x3e64f92e, 0x3e1d7475, 0x3dd88cb4, 0x3d94e963,
    0x3d4ccccd, 0x3d0cd4fb, 0x3cc1b019, 0x3c8530ce, 0x3c372dbf, 0x3bfbed88, 0x3bad3d5e, 0x3b6e4237, 0x3b147ae1, 0x3ab714e0, 0x3a5ebdb6,
    0x3a0530ce, 0x399bb3af, 0x39305978, 0x38be904d, 0x383e9b5f, 0x37a3d708, 0x36b443d0, 0x3677eba6, 0x362a7be8, 0x35ea77ff, 0x35a13bdc,
    0x355dbf30, 0x35187c4d];
pub const TRUTH_PLAIN: [u32; 32] = [0x3f800000, 0x3f3ff911, 0x3f0ff59a, 0x3ed7e89b, 0x3ea1e89b, 0x3e72d423, 0x3e361887, 0x3e088d77,
    0x3dcccccd, 0x3d99940d, 0x3d6655c2, 0x3d2cba16, 0x3d0186e3, 0x3cc2434f, 0x3c91ad39, 0x3c5a7bf2, 0x3c23d70a, 0x3bf5b9b0, 0x3bb8449c,
    0x3b8a2e77, 0x3b4f3e38, 0x3b1b690d, 0x3ae91528, 0x3aaec98e, 0x3a83126f, 0x3a44948d, 0x3a136a16, 0x39dd1725, 0x39a5cb60, 0x3978a815,
    0x393a7753, 0x390bd472];
fn is_1017(p: &Params) -> Option<&'static [u32; 32]> {
    match p.yarn {
        Some((f, o, bf, bs)) if p.theta == 160000.0 && p.dim == 64 && f == 16.0 && o == 65536.0 && bf == 32.0 && bs == 1.0 => Some(&TRUTH_COMPRESSED),
        None if p.theta == 10000.0 && p.dim == 64 => Some(&TRUTH_PLAIN),
        _ => None,
    }
}

/// The table as torch builds it on CUDA, computed by the GPU (`d41f_rope_table`) once per parameter set and
/// cached. Needs a current CUDA context with the gpu_dsv41 module available (every dsv41 GPU path has one).
/// For the 1017 config the result is checked against the recorded torch-CUDA tables (a mismatch is an error).
pub fn table(p: &Params) -> Result<[f32; 32]> {
    use cudarc::driver::sys;
    static CACHE: std::sync::Mutex<Vec<(u64, [f32; 32])>> = std::sync::Mutex::new(Vec::new());
    ensure!(p.dim == 64, "RoPE tables are built for 64 rotary dims (config {})", p.dim);
    let key = p.theta.to_bits() ^ p.yarn.map_or(0, |(a, b, c, d)| a.to_bits().rotate_left(7) ^ b.to_bits().rotate_left(19) ^ c.to_bits().rotate_left(31) ^ d.to_bits().rotate_left(43) ^ 1);
    if let Some((_, t)) = CACHE.lock().ok().and_then(|c| c.iter().find(|(k, _)| *k == key).copied()) { return Ok(t); }
    // a caller with no current CUDA context (a probe that builds its tables before its device): the primary
    // context of device 0, kept for the process (the same context every dsv41 CudaDevice handle retains)
    let mut cur: sys::CUcontext = std::ptr::null_mut();
    if unsafe { sys::cuCtxGetCurrent(&mut cur) } != sys::CUresult::CUDA_SUCCESS || cur.is_null() {
        static DEV: std::sync::OnceLock<std::sync::Arc<cudarc::driver::CudaDevice>> = std::sync::OnceLock::new();
        if DEV.get().is_none() { let d = cudarc::driver::CudaDevice::new(0).context("rope table: CUDA device 0")?; let _ = DEV.set(d); }
        let d = DEV.get().context("rope table: device")?;
        unsafe { sys::cuCtxSetCurrent(*d.cu_primary_ctx()) };
    }
    let f = crate::dsv41::gpu::raw_fn("d41f_rope_table").context("d41f_rope_table missing from the gpu_dsv41 module (stale PTX?)")?;
    let (low, inv_den, inv_factor) = yarn_band(p).unwrap_or((0.0, 0.0, 1.0));
    let mut out: sys::CUdeviceptr = 0;
    ensure!(unsafe { sys::cuMemAlloc_v2(&mut out, 128) } == sys::CUresult::CUDA_SUCCESS, "rope table: cuMemAlloc");
    let a = TableArgs { out, base: p.theta as f32, inv_dim: 1.0f32 / p.dim as f32, low, inv_den, inv_factor, yarn: yarn_band(p).is_some() as i32, half: 32, pad: 0 };
    let mut params = [&a as *const TableArgs as *mut std::ffi::c_void];
    let mut t = [0f32; 32];
    let r = unsafe {
        let r = sys::cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, std::ptr::null_mut(), params.as_mut_ptr(), std::ptr::null_mut());
        if r != sys::CUresult::CUDA_SUCCESS { r } else { sys::cuMemcpyDtoH_v2(t.as_mut_ptr() as *mut std::ffi::c_void, out, 128) }
    };
    unsafe { sys::cuMemFree_v2(out) };
    ensure!(r == sys::CUresult::CUDA_SUCCESS, "rope table kernel: {r:?}");
    if let Some(truth) = is_1017(p) {
        let bad: Vec<usize> = (0..32).filter(|&i| t[i].to_bits() != truth[i]).collect();
        ensure!(bad.is_empty(), "RoPE table {p:?}: entries {bad:?} differ from torch on CUDA (recorded truth)");
    }
    if let Ok(mut c) = CACHE.lock() { c.push((key, t)); }
    Ok(t)
}

/// The table of a layer's class (GPU-built).
pub fn table_for_layer(cfg: &Dsv41Config, layer: usize) -> Result<[f32; 32]> { table(&params(cfg, class_of(cfg, layer))) }
pub fn table_for(cfg: &Dsv41Config, c: Class) -> Result<[f32; 32]> { table(&params(cfg, c)) }

#[cfg(test)]
mod tests {
    use super::*;

    fn p(theta: f64, yarn: bool) -> Params { Params { theta, dim: 64, yarn: yarn.then_some((16.0, 65536.0, 32.0, 1.0)) } }

    #[test]
    fn dsv41_rope_host_table_against_torch_cuda() {
        // compressed class: exact (incl. KI's entry 24, the reciprocal-multiply ramp)
        let c = table_host(&p(160000.0, true));
        let bad: Vec<usize> = (0..32).filter(|&i| c[i].to_bits() != TRUTH_COMPRESSED[i]).collect();
        assert!(bad.is_empty(), "compressed entries {bad:?}");
        // plain class: CUDA powf is not correctly rounded at entries 11 and 25 (1 ulp); everything else exact
        let q = table_host(&p(10000.0, false));
        let bad: Vec<usize> = (0..32).filter(|&i| q[i].to_bits() != TRUTH_PLAIN[i]).collect();
        assert_eq!(bad, vec![11, 25]);
        for i in bad { assert_eq!((q[i].to_bits() as i64 - TRUTH_PLAIN[i] as i64).abs(), 1); }
    }

    #[test]
    fn dsv41_rope_truth_file_matches_the_constants() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/dsv41_sl/rope_truth_1017.json");
        let Ok(txt) = std::fs::read_to_string(&path) else { return };
        let v: serde_json::Value = serde_json::from_str(&txt).unwrap();
        for (k, t) in [("compressed", &TRUTH_COMPRESSED), ("plain", &TRUTH_PLAIN)] {
            let got: Vec<u32> = v["inv_freq"][k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
            assert_eq!(&got[..], &t[..], "{k}");
        }
    }
}
