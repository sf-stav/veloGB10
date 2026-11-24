//! DSV4.1 native dense FP8 (32 x 32 ue8m0 blocks) — leg KN1. Kernels: `kernels/gpu_dsv41/2x_fp8_*.cuh`;
//! host plumbing and the dispatch plan: `fp8.rs`; the GPU probe: `fp8_probe.rs`.
//!
//! `dense` = the reference's `linear()` for an FP8 weight: `act_quant` of the input (FP8 E4M3, ue8m0 per
//! 32, into the caller's scratch) followed by the block-scaled GEMM, output bf16 or f32, overwrite or add.
//! The device tensors of a `DenseWeight` hold THIS RANK'S slice (the loader packs it: weight
//! `[out_local, in_local]` E4M3, scale `[ceil(out_local / 32), in_local / 32]` E8M0); the dense call
//! reads the local shapes from the tensors. A caller that holds the FULL tensors and wants a slice calls
//! `fp8::gemm_quantised` with (n0, n_count, k0b, nkb) directly. To share one quantisation between linears
//! with the same input (wq_a + wkv, w1 + w3), call `fp8::quantize` once and `fp8::gemm_quantised` per linear.

use super::fp8;
use crate::dsv41::census::classify::Role;
use crate::dsv41::weights::{Accum, Act, DenseWeight, DevPtr, Launch, RowClass, Scratch};
use anyhow::{bail, ensure, Result};

/// Entry kernels of band 20; the gpu_dsv41 loader collects them.
pub const FNS: &[&str] = fp8::FNS;

fn local_dims(w: &DenseWeight) -> Result<(usize, usize, DevPtr, DevPtr)> {
    let wt = w.tensors.iter().find(|t| t.role == Role::Weight);
    let st = w.tensors.iter().find(|t| t.role == Role::Scale);
    let (Some(wt), Some(st)) = (wt, st) else { bail!("native FP8 dense {}: needs a weight and a scale tensor", w.key) };
    ensure!(wt.shape.len() == 2 && st.shape.len() == 2, "native FP8 dense {}: weight / scale must be 2-D", w.key);
    let (n, k) = (wt.shape[0] as usize, wt.shape[1] as usize);
    ensure!(k % 32 == 0 && n % 16 == 0, "native FP8 dense {}: local shape [{n}, {k}] needs n % 16 == 0 and k % 32 == 0", w.key);
    ensure!(st.shape[0] as usize == n.div_ceil(32) && st.shape[1] as usize == k / 32, "native FP8 dense {}: scale shape {:?} does not match weight [{n}, {k}]", w.key, st.shape);
    Ok((n, k, wt.ptr, st.ptr))
}

pub fn scratch_bytes(w: &DenseWeight, rows: usize) -> usize {
    match local_dims(w) { Ok((_, k, _, _)) => fp8::quant_bytes(rows, k), Err(_) => 0 }
}

pub fn dense(l: &Launch, w: &DenseWeight, x: &Act, y: &Act, a: Accum, s: Scratch) -> Result<RowClass> {
    let (n, k, wp, sp) = local_dims(w)?;
    ensure!(x.cols == k, "native FP8 dense {}: input has {} columns, the local weight has K = {k}", w.key, x.cols);
    ensure!(y.rows >= x.rows && y.cols >= n, "native FP8 dense {}: output [{}, {}] is smaller than [{}, {n}]", w.key, y.rows, y.cols, x.rows);
    let need = fp8::quant_bytes(x.rows, k);
    ensure!(s.bytes >= need, "native FP8 dense {}: scratch {} B < {need} B", w.key, s.bytes);
    let q = s.ptr;
    let sc = DevPtr(s.ptr.0 + ((x.rows * k + 255) / 256 * 256) as u64);
    fp8::quantize(l, x, q, sc)?;
    fp8::gemm_quantised(l, wp, sp, k, k / 32, q, sc, x.rows, 0, n, 0, k / 32, y, a)
}
