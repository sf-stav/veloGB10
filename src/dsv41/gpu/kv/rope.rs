//! Rotary frequency tables of the compressed-KV / SWA writers: the ONE dsv41 builder (`crate::dsv41::rope`, the
//! SL RoPE unification; this file had its own f64-pow builder, 1 ulp off torch-on-CUDA in two plain-class
//! entries). The kernels form `cos / sin(pos * freq)` themselves (`d41_rope_cs`, 00_abi.cuh).
//!
//! Two classes (model.py 680-696): layers with a compress ratio (and their SWA / compressed-KV writes)
//! use theta = `compress_rope_theta` with YaRN; the pure sliding-window layers (0, 1) and the DSpark
//! stages use `rope_theta` with no YaRN.

use crate::dsv41::config::Dsv41Config;
use crate::dsv41::rope;
use anyhow::Result;

/// The 32 fp32 inverse frequencies of a 64-wide rotary tail, as torch builds them on CUDA (GPU-built).
pub fn inv_freq(dim: usize, original_seq_len: usize, base: f64, factor: f64, beta_fast: f64, beta_slow: f64) -> Result<Vec<f32>> {
    let p = rope::Params { theta: base, dim, yarn: (original_seq_len > 0).then_some((factor, original_seq_len as f64, beta_fast, beta_slow)) };
    Ok(rope::table(&p)?.to_vec())
}

/// The table of a layer class: `compressed` = layers with a compress ratio (theta 160000 + YaRN), else the plain one.
pub fn inv_freq_for(cfg: &Dsv41Config, compressed: bool) -> Result<Vec<f32>> {
    Ok(rope::table_for(cfg, if compressed { rope::Class::Compressed } else { rope::Class::Plain })?.to_vec())
}
