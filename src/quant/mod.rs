//! NVFP4 quantization: the codec, plus a load-time *simulated* quantization used to test which
//! tensors can actually survive 4 bits.
//!
//! # The format (NVFP4 / `nvfp4-pack-quantized`, matching compressed-tensors on HF)
//!
//! Two-level scaling around a 4-bit float:
//!   * **element**: E2M1 — 4 bits, magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}, sign bit. Max 6.
//!   * **block scale**: one **FP8 E4M3** per **16 consecutive elements along K** (the reduction dim).
//!   * **tensor scale**: one f32 per tensor, so the block scales land inside E4M3's range.
//!
//! `w ≈ e2m1(q) * e4m3(s_block) * s_tensor`, costing 4 + 8/16 = **4.5 bits/weight** (3.55x smaller
//! than bf16). The 16-element block is the whole trick: an outlier can only poison its 15 neighbours,
//! which is why plain round-to-nearest gets close to calibrated INT4 methods.
//!
//! **E4M3 is a FLOAT.** Decode it from its bit pattern; never integer-cast it. (That is the core bug
//! in the abandoned `kernels/gemm_nvfp4.cu` prototype, and it is the single easiest way to produce a
//! model that loads, runs, and is quietly wrong.)
//!
//! # Simulated quantization (why it exists)
//!
//! Everyone ships NVFP4 checkpoints with the LM head and the recurrent/GDN projections left in bf16,
//! on the folklore that those layers "need" high precision. We have not seen that proven. Encoding a
//! weight to NVFP4 and immediately decoding it back to bf16 leaves the *bytes* bf16 — so the engine
//! runs unmodified — while the *values* carry exactly the error the real 4-bit kernel would produce.
//! So we can measure the damage per tensor group, in the real engine, before writing a single kernel.
//!
//! Driven by `--fake-quant` (an experiment knob, not a serving setting):
//! ```text
//!   --fake-quant=all                 # quantize everything we intend to quantize
//!   --fake-quant=all,-gdn            # ...but keep the GDN projections in bf16
//!   --fake-quant=all,-lmhead,-embed  # ...keep the LM head and embedding in bf16
//!   --fake-quant=mlp,attn            # what the HF checkpoints actually do
//! ```
//!
//! Map of this directory (every public item keeps its `crate::quant::` path via `pub use self::<sub>::*`):
//! - `mod.rs` (this file): the E2M1/E4M3 codec, NVFP4 / FP8 tensors, PLE row records, Q2 fake-quant.
//! - `stq.rs`: STQ1_0 / Ternary2 / Ls3 simulated block formats.
//! - `recipe.rs`: weight format enum, tensor groups, `--fake-quant` recipe and layer-rule parsing.
//! - `repack.rs`: the MMA (Marlin-permutation) weight repack and weight fusion / row subsets.
//! - `draft_vocab.rs`: `--draft-vocab` (FR-Spec) row subset.
//! - `tests.rs`, `q2_tests.rs`: the inline unit-test modules, one file each.

use half::bf16;

mod stq;
mod recipe;
mod repack;
mod draft_vocab;

pub use self::stq::*;
pub use self::recipe::*;
pub use self::repack::*;
pub use self::draft_vocab::*;


/// E2M1 decode table, indexed by the 3 magnitude bits. Sign is bit 3.
pub const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
pub const E2M1_MAX: f32 = 6.0;
pub const E4M3_MAX: f32 = 448.0;

/// Decode one FP8 E4M3 byte (1 sign, 4 exp bias-7, 3 mantissa; no inf, 0xFF/0x7F are NaN).
#[inline]
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let man = (b & 0x07) as f32;
    if exp == 0 {
        // subnormal: 2^-6 * (man/8)
        sign * (man / 8.0) * 2.0f32.powi(-6)
    } else {
        sign * (1.0 + man / 8.0) * 2.0f32.powi(exp - 7)
    }
}

/// The 127 finite non-negative E4M3 values, in code order. They are **monotonically increasing**,
/// which is what lets the encoder binary-search instead of scanning all 127.
static E4M3_TABLE: std::sync::OnceLock<[f32; 127]> = std::sync::OnceLock::new();
fn e4m3_table() -> &'static [f32; 127] {
    E4M3_TABLE.get_or_init(|| {
        let mut t = [0.0f32; 127];
        for (code, slot) in t.iter_mut().enumerate() {
            *slot = e4m3_to_f32(code as u8); // 0x00..=0x7E; 0x7F is NaN
        }
        t
    })
}

/// Encode f32 -> E4M3 (round-to-nearest). Built against the decode table so the two cannot drift.
pub fn f32_to_e4m3(x: f32) -> u8 {
    if !x.is_finite() || x == 0.0 { return 0; }
    let sign_bit = if x < 0.0 { 0x80u8 } else { 0x00u8 };
    let a = x.abs().min(E4M3_MAX);
    let t = e4m3_table();
    // First code whose value is >= a; the answer is that one or its predecessor.
    let hi = t.partition_point(|&v| v < a);
    let code = if hi == 0 {
        0
    } else if hi >= 127 {
        126
    } else if (t[hi] - a) < (a - t[hi - 1]) {
        hi
    } else {
        hi - 1
    };
    sign_bit | code as u8
}

/// Encode a value already normalized into E2M1's range -> 4-bit code (sign in bit 3).
/// Round-to-nearest; exact ties go to the even code, matching the usual RTN convention.
#[inline]
pub fn f32_to_e2m1(x: f32) -> u8 {
    let sign_bit = if x < 0.0 { 0x8u8 } else { 0x0u8 };
    let a = x.abs().min(E2M1_MAX);
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, &v) in E2M1.iter().enumerate() {
        let err = (v - a).abs();
        if err < best_err - 1e-9 || ((err - best_err).abs() <= 1e-9 && (i as u8) % 2 == 0) {
            best_err = err;
            best = i as u8;
        }
    }
    sign_bit | best
}

#[inline]
pub fn e2m1_to_f32(code: u8) -> f32 {
    let v = E2M1[(code & 0x7) as usize];
    if code & 0x8 != 0 { -v } else { v }
}

/// Block size along K. Fixed by the format.
pub const BLOCK: usize = 16;

/// One quantized tensor in `nvfp4-pack-quantized` layout — byte-compatible with HF's
/// compressed-tensors, so our artifacts and theirs are mutually loadable.
///
/// **`global_scale` is stored in the RECIPROCAL convention**, matching llm-compressor:
/// `global_scale = (E2M1_MAX * E4M3_MAX) / amax(W)`, and dequant DIVIDES by it:
///
/// ```text
///   w ≈ e2m1(q) * e4m3(s_block) / global_scale
/// ```
///
/// This is not a guess. Dequantizing a real HF NVFP4 tensor
/// (`ig1/Qwen3.5-9B-NVFP4`, layer 0 `mlp.gate_proj`) with this convention recovers the original bf16
/// weights to 9.5% relative L2 — i.e. ordinary 4-bit quantization noise. With the scale applied the
/// other way the error is 8.8e7. The convention, the nibble order (low nibble = even index) and the
/// float decode of E4M3 were all confirmed against that checkpoint.
#[derive(Debug, Clone)]
pub struct Nvfp4Tensor {
    pub qweight: Vec<u8>,   // [M, K/2]  two nibbles per byte; low nibble = even index
    pub scales: Vec<u8>,    // [M, K/16] E4M3 block scales
    pub global_scale: f32,  // (6*448)/amax  — DIVIDE by this on dequant
    pub m: usize,
    pub k: usize,
}

/// Quantize a row-major [M, K] bf16 weight to NVFP4. K must be a multiple of 16 (true for every
/// reduction dim in this model family — assert rather than silently pad).
pub fn quantize_nvfp4(w: &[bf16], m: usize, k: usize) -> Nvfp4Tensor {
    assert_eq!(w.len(), m * k, "shape mismatch");
    assert_eq!(k % BLOCK, 0, "K={} is not a multiple of {}", k, BLOCK);

    // Reciprocal convention (matches llm-compressor / HF): global_scale = (6*448) / amax(W).
    // `s_tensor` below is its inverse, which is what the math actually multiplies by.
    let amax = w.iter().fold(0.0f32, |acc, x| acc.max(x.to_f32().abs()));
    let global_scale = if amax > 0.0 { (E2M1_MAX * E4M3_MAX) / amax } else { 1.0 };
    let s_tensor = 1.0 / global_scale;

    let nblk = k / BLOCK;
    let mut qweight = vec![0u8; m * k / 2];
    let mut scales = vec![0u8; m * nblk];

    // Rows are independent. 27B has ~14e9 weights to encode, so this is parallelized across rows;
    // single-threaded it takes minutes per model.
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).max(1);
    let rows_per = m.div_ceil(nthreads).max(1);
    std::thread::scope(|sc| {
        let qparts = qweight.chunks_mut(rows_per * (k / 2));
        let sparts = scales.chunks_mut(rows_per * nblk);
        for (t, (qp, sp)) in qparts.zip(sparts).enumerate() {
            sc.spawn(move || {
                let r0 = t * rows_per;
                let nrows = qp.len() / (k / 2);
                for r in 0..nrows {
                    let row = r0 + r;
                    for b in 0..nblk {
                        let blk = &w[row * k + b * BLOCK..][..BLOCK];
                        let bmax = blk.iter().fold(0.0f32, |a, x| a.max(x.to_f32().abs()));
                        // s_block = e4m3(amax(block) / 6 / s_tensor)
                        let s_raw = if bmax > 0.0 { bmax / E2M1_MAX / s_tensor } else { 0.0 };
                        let s_code = f32_to_e4m3(s_raw);
                        sp[r * nblk + b] = s_code;

                        let s = e4m3_to_f32(s_code) * s_tensor;
                        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
                        for i in 0..BLOCK {
                            let q = f32_to_e2m1(blk[i].to_f32() * inv);
                            let idx = r * k + b * BLOCK + i;   // index WITHIN this row-chunk
                            let byte = idx / 2;
                            if idx % 2 == 0 {
                                qp[byte] = (qp[byte] & 0xF0) | q;      // low nibble = even index
                            } else {
                                qp[byte] = (qp[byte] & 0x0F) | (q << 4);
                            }
                        }
                    }
                }
            });
        }
    });
    Nvfp4Tensor { qweight, scales, global_scale, m, k }
}

/// Dequantize to FP32 (no bf16 intermediate) — the device kernels' exact semantics:
/// `e2m1(code) * e4m3(scale) / global_scale`. Used by the vision-tower loader, which keeps its
/// weights in FP32 end-to-end (the CPU/GPU vision path is a plain BF16→FP32 class, AGENTS §2.4).
pub fn dequantize_nvfp4_f32(q: &Nvfp4Tensor) -> Vec<f32> {
    let nblk = q.k / BLOCK;
    let s_tensor = 1.0 / q.global_scale;   // reciprocal convention — see Nvfp4Tensor
    let mut out = vec![0.0f32; q.m * q.k];
    for row in 0..q.m {
        for b in 0..nblk {
            let s = e4m3_to_f32(q.scales[row * nblk + b]) * s_tensor;
            for i in 0..BLOCK {
                let idx = row * q.k + b * BLOCK + i;
                let byte = q.qweight[idx / 2];
                let code = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                out[idx] = e2m1_to_f32(code) * s;
            }
        }
    }
    out
}

/// Dequantize back to bf16. This is the host reference the device kernel must match bit-for-bit.
pub fn dequantize_nvfp4(q: &Nvfp4Tensor) -> Vec<bf16> {
    let nblk = q.k / BLOCK;
    let s_tensor = 1.0 / q.global_scale;   // reciprocal convention — see Nvfp4Tensor
    let mut out = vec![bf16::ZERO; q.m * q.k];
    for row in 0..q.m {
        for b in 0..nblk {
            let s = e4m3_to_f32(q.scales[row * nblk + b]) * s_tensor;
            for i in 0..BLOCK {
                let idx = row * q.k + b * BLOCK + i;
                let byte = q.qweight[idx / 2];
                let code = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                out[idx] = bf16::from_f32(e2m1_to_f32(code) * s);
            }
        }
    }
    out
}

// ============================ PLE n-gram table row records (qwen4_exp) ============================
//
// Qwen3.8-Flash-Next's PLE n-gram embedding is a 320M-row x 160 table (51 GB in FP8, 102 GB bf16)
// read by ROW — 16 rows per token — never as a GEMM operand. So it gets its own on-disk codec:
// one fixed-size record per row, NVFP4 inside:
//
//   [ 80 B  E2M1 nibbles (160 values, low nibble = even index) |
//     10 B  E4M3 block scales (one per 16 values along the row) |
//      6 B  zero pad ]                                  = 96 B per row (32-B aligned)
//
// plus ONE reciprocal-convention global scale per source shard (`w ≈ e2m1 * e4m3 / gs`, exactly the
// `Nvfp4Tensor` convention). Fixed-size records are the point: a row is ONE contiguous
// `pread(row * 96)` from NVMe when the table is offloaded to SSD, and one coalesced 96-B load when it
// is resident on the GPU. 30.7 GB for the whole table (vs 51 GB FP8 / 102 GB bf16).
pub const PLE_REC_BYTES: usize = 96;
pub const PLE_DIM: usize = 160;
pub const PLE_QW_BYTES: usize = PLE_DIM / 2;          // 80
pub const PLE_SC_BYTES: usize = PLE_DIM / BLOCK;      // 10

/// Quantize a [rows, PLE_DIM] bf16 table shard into 96-B row records. Returns (records, global_scale).
pub fn quantize_ple_rows(w: &[bf16], rows: usize) -> (Vec<u8>, f32) {
    assert_eq!(w.len(), rows * PLE_DIM, "PLE shard shape");
    let q = quantize_nvfp4(w, rows, PLE_DIM);
    let mut rec = vec![0u8; rows * PLE_REC_BYTES];
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).max(1);
    let rows_per = rows.div_ceil(nthreads).max(1);
    std::thread::scope(|sc| {
        for (t, chunk) in rec.chunks_mut(rows_per * PLE_REC_BYTES).enumerate() {
            let q = &q;
            sc.spawn(move || {
                let r0 = t * rows_per;
                for (i, r) in chunk.chunks_mut(PLE_REC_BYTES).enumerate() {
                    let row = r0 + i;
                    r[..PLE_QW_BYTES].copy_from_slice(&q.qweight[row * PLE_QW_BYTES..][..PLE_QW_BYTES]);
                    r[PLE_QW_BYTES..PLE_QW_BYTES + PLE_SC_BYTES]
                        .copy_from_slice(&q.scales[row * PLE_SC_BYTES..][..PLE_SC_BYTES]);
                }
            });
        }
    });
    (rec, q.global_scale)
}

/// Host reference decode of one 96-B record (the device kernel must match this bit-for-bit).
pub fn dequant_ple_row(rec: &[u8], global_scale: f32) -> [f32; PLE_DIM] {
    let s_tensor = 1.0 / global_scale;
    let mut out = [0f32; PLE_DIM];
    for b in 0..PLE_SC_BYTES {
        let s = e4m3_to_f32(rec[PLE_QW_BYTES + b]) * s_tensor;
        for i in 0..BLOCK {
            let idx = b * BLOCK + i;
            let byte = rec[idx / 2];
            let code = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            out[idx] = e2m1_to_f32(code) * s;
        }
    }
    out
}

/// Simulated quantization: NVFP4 round-trip in place. Bytes stay bf16, values carry the 4-bit error.
pub fn fake_quant_nvfp4(w: &mut [bf16], m: usize, k: usize) {
    let q = quantize_nvfp4(w, m, k);
    w.copy_from_slice(&dequantize_nvfp4(&q));
}

// ---------------------------------------------------------------------------------------------
// FP8-E4M3 with NVFP4-style per-16 block scales ("fp8-block") — PLAN/25 §1a-fp8b.
//
// The untested corner of the 2x2: nvfp4 = fine scale + coarse payload (lost TP=4 on deep-chain
// noise), fp8-per-row = fine payload + coarse scale (lost TP=4 on outlier rows). fp8-block is
// fine x fine: E4M3 payload (~3% per-weight steps) with a per-16 E4M3 scale (16x-local outlier
// adaptivity) at 8.5 bits/weight. Same reciprocal global-scale convention as NVFP4:
//
//   w ≈ e4m3(q) * e4m3(s_block) / global_scale,   global_scale = 448 / amax(W)
//
// so stored block scales land in (0, 1] — inside E4M3's normal range for any tensor whose
// per-block amax is at least ~1/448 of the tensor amax (a near-zero block dequantizes to
// near-zero weights regardless — the residual error is bounded by the subnormal scale, which
// is tiny in real units).
pub struct Fp8BlockTensor {
    pub qweight: Vec<u8>,   // [M, K] E4M3
    pub scales: Vec<u8>,    // [M, K/16] E4M3
    pub global_scale: f32,  // 448/amax — DIVIDE by this on dequant
    pub m: usize,
    pub k: usize,
}

pub fn quantize_fp8_block16(w: &[bf16], m: usize, k: usize) -> Fp8BlockTensor {
    assert_eq!(w.len(), m * k, "shape mismatch");
    assert_eq!(k % BLOCK, 0, "K={} is not a multiple of {}", k, BLOCK);

    let amax = w.iter().fold(0.0f32, |acc, x| acc.max(x.to_f32().abs()));
    let global_scale = if amax > 0.0 { E4M3_MAX / amax } else { 1.0 };
    let s_tensor = 1.0 / global_scale;

    let nblk = k / BLOCK;
    let mut qweight = vec![0u8; m * k];
    let mut scales = vec![0u8; m * nblk];
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).max(1);
    let rows_per = m.div_ceil(nthreads).max(1);
    std::thread::scope(|sc| {
        let qparts = qweight.chunks_mut(rows_per * k);
        let sparts = scales.chunks_mut(rows_per * nblk);
        for (t, (qp, sp)) in qparts.zip(sparts).enumerate() {
            sc.spawn(move || {
                let r0 = t * rows_per;
                let nrows = qp.len() / k;
                for r in 0..nrows {
                    let row = r0 + r;
                    for b in 0..nblk {
                        let blk = &w[row * k + b * BLOCK..][..BLOCK];
                        let bmax = blk.iter().fold(0.0f32, |a, x| a.max(x.to_f32().abs()));
                        // s_block = e4m3(amax(block) / 448 / s_tensor) — mirrors the nvfp4 recipe
                        // with the payload max (448) in E2M1_MAX's role.
                        let s_raw = if bmax > 0.0 { bmax / E4M3_MAX / s_tensor } else { 0.0 };
                        let s_code = f32_to_e4m3(s_raw);
                        sp[r * nblk + b] = s_code;

                        let s = e4m3_to_f32(s_code) * s_tensor;
                        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
                        for i in 0..BLOCK {
                            // local row r within this thread's chunk (mirrors quantize_nvfp4)
                            qp[r * k + b * BLOCK + i] = f32_to_e4m3(blk[i].to_f32() * inv);
                        }
                    }
                }
            });
        }
    });
    Fp8BlockTensor { qweight, scales, global_scale, m, k }
}

/// Dequantize to FP32 — the reference semantics the kernel must reproduce:
/// `e4m3(code) * e4m3(scale) / global_scale`.
pub fn dequantize_fp8_block16_f32(q: &Fp8BlockTensor) -> Vec<f32> {
    let nblk = q.k / BLOCK;
    let mut out = vec![0.0f32; q.m * q.k];
    for r in 0..q.m {
        for b in 0..nblk {
            let s = e4m3_to_f32(q.scales[r * nblk + b]) / q.global_scale;
            for i in 0..BLOCK {
                out[r * q.k + b * BLOCK + i] = e4m3_to_f32(q.qweight[r * q.k + b * BLOCK + i]) * s;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Q2 (2-bit, E26) — the 2-bit codec. Per-16 block along K (the NVFP4 granularity, so the future
// kernel reuses the tile machinery), Lloyd-Max 2-bit levels for a standard normal (the published
// optimum for Gaussian sources): {-1.5104, -0.4528, +0.4528, +1.5104} — RTN midpoints ±0.9816, 0.
// Block scale E4M3, tensor global scale in the reciprocal convention (as NVFP4).
pub const Q2_LEVELS: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];
pub const Q2_MAX: f32 = 1.5104;

#[inline]
fn q2_index(x: f32) -> usize {
    if x < -0.9816 { 0 } else if x < 0.0 { 1 } else if x < 0.9816 { 2 } else { 3 }
}

/// Simulated 2-bit quantization in place (bytes stay bf16, values carry exactly the error a real
/// 2-bit kernel would produce): per-16 block, E4M3 block scale round-trip, RTN to Q2_LEVELS.
/// Row-parallel like quantize_nvfp4; single pass (the q2 amax is derived, not materialized).
pub fn fake_quant_q2(w: &mut [bf16], m: usize, k: usize) {
    assert_eq!(w.len(), m * k, "shape mismatch");
    assert_eq!(k % BLOCK, 0, "K={} is not a multiple of {}", k, BLOCK);
    let amax = w.iter().fold(0.0f32, |a, x| a.max(x.to_f32().abs()));
    let gs = if amax > 0.0 { (Q2_MAX * E4M3_MAX) / amax } else { 1.0 };
    let s_tensor = 1.0 / gs;
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).max(1);
    let rows_per = m.div_ceil(nthreads).max(1);
    std::thread::scope(|sc| {
        for (t, chunk) in w.chunks_mut(rows_per * k).enumerate() {
            sc.spawn(move || {
                for row in 0..chunk.len() / k {
                    for b in 0..k / BLOCK {
                        let blk = &mut chunk[row * k + b * BLOCK..][..BLOCK];
                        let bmax = blk.iter().fold(0.0f32, |a, x| a.max(x.to_f32().abs()));
                        let s_code = f32_to_e4m3(if bmax > 0.0 { bmax / Q2_MAX / s_tensor } else { 0.0 });
                        let s = e4m3_to_f32(s_code) * s_tensor;
                        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
                        for x in blk.iter_mut() {
                            let q = Q2_LEVELS[q2_index(x.to_f32() * inv)];
                            *x = bf16::from_f32(q * s);
                        }
                    }
                }
            });
        }
    });
}

// ---------------------------------------------------------------------------------------------
// FP8 E4M3 weight-only — 8 bits + one f32 scale per output row.
//
// The quality fallback for tensors 4 bits hurts. At 8 bits the 16-element blocks are unnecessary;
// a per-row scale suffices. Kernel-side it is `gemm_binv_b` with a byte load and one multiply.
// Measured on 9B: the GDN projections are ~3x more perplexity-sensitive per parameter than anything
// else, which makes them exactly the tensors worth spending the extra 4 bits on.
// ---------------------------------------------------------------------------------------------

/// One FP8-E4M3 tensor: [M, K] bytes + one f32 scale per row.
pub struct Fp8Tensor {
    pub qweight: Vec<u8>,      // [M, K]
    pub row_scale: Vec<f32>,   // [M]
    pub m: usize,
    pub k: usize,
}

pub fn quantize_fp8(w: &[bf16], m: usize, k: usize) -> Fp8Tensor {
    assert_eq!(w.len(), m * k, "shape mismatch");
    let mut qweight = vec![0u8; m * k];
    let mut row_scale = vec![0.0f32; m];
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).max(1);
    let rows_per = m.div_ceil(nthreads).max(1);
    std::thread::scope(|sc| {
        for (t, (qp, sp)) in qweight.chunks_mut(rows_per * k)
            .zip(row_scale.chunks_mut(rows_per)).enumerate()
        {
            sc.spawn(move || {
                let r0 = t * rows_per;
                for r in 0..(qp.len() / k) {
                    let row = &w[(r0 + r) * k..][..k];
                    let amax = row.iter().fold(0.0f32, |a, x| a.max(x.to_f32().abs()));
                    let s = if amax > 0.0 { amax / E4M3_MAX } else { 1.0 };
                    sp[r] = s;
                    let inv = 1.0 / s;
                    for i in 0..k {
                        qp[r * k + i] = f32_to_e4m3(row[i].to_f32() * inv);
                    }
                }
            });
        }
    });
    Fp8Tensor { qweight, row_scale, m, k }
}

pub fn dequantize_fp8(q: &Fp8Tensor) -> Vec<bf16> {
    let mut out = vec![bf16::ZERO; q.m * q.k];
    for r in 0..q.m {
        let s = q.row_scale[r];
        for i in 0..q.k {
            out[r * q.k + i] = bf16::from_f32(e4m3_to_f32(q.qweight[r * q.k + i]) * s);
        }
    }
    out
}

/// Simulated FP8 quantization, in place.
pub fn fake_quant_fp8(w: &mut [bf16], m: usize, k: usize) {
    let q = quantize_fp8(w, m, k);
    w.copy_from_slice(&dequantize_fp8(&q));
}

/// Weight format, per tensor. Mixed precision is where the win is: the evidence says spend the extra
/// bits on the GDN projections, NOT on the LM head (which is the least sensitive tensor in the model,
/// despite everyone keeping it in bf16).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fmt { Bf16, Fp8, Nvfp4 }

/// Simulated quantization in the given format.
pub fn fake_quant(w: &mut [bf16], m: usize, k: usize, fmt: Fmt) {
    match fmt {
        Fmt::Bf16 => {}
        Fmt::Fp8 => fake_quant_fp8(w, m, k),
        Fmt::Nvfp4 => fake_quant_nvfp4(w, m, k),
    }
}

/// Bits per weight, including amortized scales — what actually sets decode speed.
pub fn bits_per_weight(fmt: Fmt, k: usize) -> f32 {
    match fmt {
        Fmt::Bf16 => 16.0,
        Fmt::Fp8 => 8.0 + 32.0 / k as f32,        // one f32 per row of K
        Fmt::Nvfp4 => 4.0 + 8.0 / BLOCK as f32,   // one E4M3 per 16 elements
    }
}

/// Relative error of a round-trip, as `||w' - w|| / ||w||` (and the max absolute deviation).
pub fn roundtrip_error(orig: &[bf16], deq: &[bf16]) -> (f32, f32) {
    let mut se = 0.0f64;
    let mut sn = 0.0f64;
    let mut mx = 0.0f32;
    for (a, b) in orig.iter().zip(deq.iter()) {
        let (a, b) = (a.to_f32(), b.to_f32());
        let d = (a - b) as f64;
        se += d * d;
        sn += (a as f64) * (a as f64);
        mx = mx.max((a - b).abs());
    }
    (((se / sn.max(1e-30)).sqrt()) as f32, mx)
}


#[cfg(test)]
mod tests;

#[cfg(test)]
mod q2_tests;
