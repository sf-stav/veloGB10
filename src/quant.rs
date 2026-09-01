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
//! Driven by `RUST_INFER_FAKE_QUANT` (an experiment knob, not a serving setting):
//! ```text
//!   RUST_INFER_FAKE_QUANT=all                 # quantize everything we intend to quantize
//!   RUST_INFER_FAKE_QUANT=all,-gdn            # ...but keep the GDN projections in bf16
//!   RUST_INFER_FAKE_QUANT=all,-lmhead,-embed  # ...keep the LM head and embedding in bf16
//!   RUST_INFER_FAKE_QUANT=mlp,attn            # what the HF checkpoints actually do
//! ```

use half::bf16;

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
// SQ campaign — STQ1_0 / ternary-2bit / 3-bit-LS quality-simulation quantizers.
//
// Values-only round-trips over 256-weight blocks (STQ1_0's QK_K), after AngelSlim's PTQ encoder
// for the Hy4 STQ1_0 build (`docs/sq_refs/angelslim_stq1_0_quant_cuda.patch`): imatrix-weighted
// least-squares scale `d = Σ(w·q·x)/Σ(w·q²)`, zero placed at the lane of minimum incremental cost
// `w·(x² − (|x|−d)²)`, 3 alternating rounds, `w[j] = qw[j]·sqrt(σ² + x[j]²)` with `σ² = 2Σx²/256`.
// The formats these simulate (the probe bakes the round-trip VALUES into NVFP4 tensors via
// `--stq-bake`; the engine then serves them unmodified — only the error is real):
//   * Stq1_0   1.3125 bpw — fp16 scale/256 + 4-bit slot + 1-bit sign over stride-16 groups of
//              4 lanes with exactly one forced zero per group (3:4 structure).
//   * Ternary2 2.0625 bpw — {−d, 0, +d} with FREE zero placement, fp16 scale/256 (TQ2_0-class).
//   * Ls3Bit   3.0625 bpw — 8 uniform levels ±(2i+1)/2 · d, fp16 scale/256 (IQ3-XXS-class).
// ---------------------------------------------------------------------------------------------

pub const STQ_BLOCK: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StqKind {
    Stq1_0,
    Ternary2,
    Ls3Bit,
}

impl StqKind {
    pub fn bpw(self) -> f32 {
        match self {
            StqKind::Stq1_0 => 1.3125,
            StqKind::Ternary2 => 2.0625,
            StqKind::Ls3Bit => 3.0625,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            StqKind::Stq1_0 => "stq1_0",
            StqKind::Ternary2 => "ternary2",
            StqKind::Ls3Bit => "ls3bit",
        }
    }
}

/// fp16 round-trip: the real formats store the block scale as fp16 — simulate that loss too.
#[inline]
fn round_f16(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

/// The AngelSlim per-block weights: `qw·sqrt(σ² + x²)` (σ² = 2·mean(x²)); unweighted when
/// `qw` is absent.
#[inline]
fn stq_weight(x: f32, sigma2: f32, qw: Option<f32>) -> f32 {
    let base = (sigma2 + x * x).sqrt();
    match qw {
        Some(q) => q * base,
        None => base,
    }
}

/// STQ1_0 block encode→decode. `x`/`out` are 256-long; `qw` (optional) is the matching slice of
/// per-input-channel importance. Verbatim port of `quantize_row_stq1_0_impl` + its decode.
pub fn stq1_0_block(x: &[f32], qw: Option<&[f32]>, out: &mut [f32]) {
    debug_assert_eq!(x.len(), STQ_BLOCK);
    let (mut sumx2, mut amax) = (0.0f32, 0.0f32);
    for &v in x {
        sumx2 += v * v;
        amax = amax.max(v.abs());
    }
    if !(amax > 0.0) {
        out.fill(0.0);
        return;
    }
    let sigma2 = 2.0 * sumx2 / STQ_BLOCK as f32;
    let mut weight = [0.0f32; STQ_BLOCK];
    for j in 0..STQ_BLOCK {
        weight[j] = stq_weight(x[j], sigma2, qw.map(|q| q[j]));
    }
    let mut sel = [0i8; STQ_BLOCK];
    let mut d = amax;
    for _ in 0..3 {
        // Zero placement over the stride-16 groups: group g owns lanes chunk*64 + gloc + p*16.
        for g in 0..STQ_BLOCK / 4 {
            let chunk = g / 16;
            let gloc = g % 16;
            let base = chunk * 64 + gloc;
            let mut zero_pos = 0usize;
            let mut best = f32::INFINITY;
            for p in 0..4 {
                let j = base + p * 16;
                let ax = x[j].abs();
                let cost = weight[j] * (x[j] * x[j] - (ax - d) * (ax - d));
                if cost < best {
                    best = cost;
                    zero_pos = p;
                }
            }
            for p in 0..4 {
                let j = base + p * 16;
                sel[j] = if p == zero_pos { 0 } else if x[j] < 0.0 { -1 } else { 1 };
            }
        }
        let (mut sumqx, mut sumq2) = (0.0f32, 0.0f32);
        for j in 0..STQ_BLOCK {
            let q = sel[j] as f32;
            sumqx += weight[j] * q * x[j];
            sumq2 += weight[j] * q * q;
        }
        if !(sumq2 > 0.0) {
            break;
        }
        let dnew = sumqx / sumq2;
        if !(dnew > 0.0) {
            break;
        }
        let converged = (dnew - d).abs() <= 1e-6 * d;
        d = dnew;
        if converged {
            break;
        }
    }
    let d16 = round_f16(d);
    for j in 0..STQ_BLOCK {
        out[j] = sel[j] as f32 * d16;
    }
}

/// Ternary-2bit block: {−d, 0, +d}, free placement (zero iff |x| < d/2 — the argmin, independent
/// of w since w scales both candidate costs equally), LS scale from the weighted objective.
pub fn ternary2_block(x: &[f32], qw: Option<&[f32]>, out: &mut [f32]) {
    debug_assert_eq!(x.len(), STQ_BLOCK);
    let (mut sumx2, mut amax, mut sumabs) = (0.0f32, 0.0f32, 0.0f32);
    for &v in x {
        sumx2 += v * v;
        sumabs += v.abs();
        amax = amax.max(v.abs());
    }
    if !(amax > 0.0) {
        out.fill(0.0);
        return;
    }
    let sigma2 = 2.0 * sumx2 / STQ_BLOCK as f32;
    let mut weight = [0.0f32; STQ_BLOCK];
    for j in 0..STQ_BLOCK {
        weight[j] = stq_weight(x[j], sigma2, qw.map(|q| q[j]));
    }
    let mut sel = [0.0f32; STQ_BLOCK];
    let mut d = sumabs / STQ_BLOCK as f32; // amax is a poor ternary init; mean|x| is not
    for _ in 0..3 {
        let half_d = 0.5 * d;
        for j in 0..STQ_BLOCK {
            sel[j] = if x[j].abs() < half_d {
                0.0
            } else if x[j] < 0.0 {
                -1.0
            } else {
                1.0
            };
        }
        let (mut sumqx, mut sumq2) = (0.0f32, 0.0f32);
        for j in 0..STQ_BLOCK {
            sumqx += weight[j] * sel[j] * x[j];
            sumq2 += weight[j] * sel[j] * sel[j];
        }
        if !(sumq2 > 0.0) {
            break;
        }
        let dnew = sumqx / sumq2;
        if !(dnew > 0.0) {
            break;
        }
        let converged = (dnew - d).abs() <= 1e-6 * d;
        d = dnew;
        if converged {
            break;
        }
    }
    let d16 = round_f16(d);
    for j in 0..STQ_BLOCK {
        out[j] = sel[j] * d16;
    }
}

/// 3-bit block: 8 uniform levels ±(2i+1)/2 · d (IQ3-XXS-class granularity without its
/// nonuniform codebook). Nearest-level placement is weight-independent; the LS scale is not.
pub fn ls3_block(x: &[f32], qw: Option<&[f32]>, out: &mut [f32]) {
    debug_assert_eq!(x.len(), STQ_BLOCK);
    let (mut sumx2, mut amax) = (0.0f32, 0.0f32);
    for &v in x {
        sumx2 += v * v;
        amax = amax.max(v.abs());
    }
    if !(amax > 0.0) {
        out.fill(0.0);
        return;
    }
    let sigma2 = 2.0 * sumx2 / STQ_BLOCK as f32;
    let mut weight = [0.0f32; STQ_BLOCK];
    for j in 0..STQ_BLOCK {
        weight[j] = stq_weight(x[j], sigma2, qw.map(|q| q[j]));
    }
    let mut sel = [0.0f32; STQ_BLOCK];
    let mut d = amax / 3.5;
    for _ in 0..3 {
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        for j in 0..STQ_BLOCK {
            let t = x[j] * inv;
            let idx = (t.abs() - 0.5).round().clamp(0.0, 3.0);
            let level = (2.0 * idx + 1.0) * 0.5;
            sel[j] = if t < 0.0 { -level } else { level };
        }
        let (mut sumqx, mut sumq2) = (0.0f32, 0.0f32);
        for j in 0..STQ_BLOCK {
            sumqx += weight[j] * sel[j] * x[j];
            sumq2 += weight[j] * sel[j] * sel[j];
        }
        if !(sumq2 > 0.0) {
            break;
        }
        let dnew = sumqx / sumq2;
        if !(dnew > 0.0) {
            break;
        }
        let converged = (dnew - d).abs() <= 1e-6 * d;
        d = dnew;
        if converged {
            break;
        }
    }
    let d16 = round_f16(d);
    for j in 0..STQ_BLOCK {
        out[j] = sel[j] * d16;
    }
}

fn stq_block_dispatch(kind: StqKind, x: &[f32], qw: Option<&[f32]>, out: &mut [f32]) {
    match kind {
        StqKind::Stq1_0 => stq1_0_block(x, qw, out),
        StqKind::Ternary2 => ternary2_block(x, qw, out),
        StqKind::Ls3Bit => ls3_block(x, qw, out),
    }
}

/// In-place values-only round-trip of an [M, K] f32 row-major tensor through one SQ format.
/// K must be a multiple of 256. `qw` = optional per-input-channel importance (len K).
/// Row-parallel; the tool is expected to run under a capped+niced taskset (SQ politeness rule).
pub fn fake_quant_stq(w: &mut [f32], m: usize, k: usize, qw: Option<&[f32]>, kind: StqKind) {
    assert_eq!(w.len(), m * k, "shape mismatch");
    assert_eq!(k % STQ_BLOCK, 0, "K={k} is not a multiple of STQ_BLOCK={STQ_BLOCK}");
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).max(1);
    let rows_per = m.div_ceil(nthreads).max(1);
    std::thread::scope(|sc| {
        for chunk in w.chunks_mut(rows_per * k) {
            let qw = qw.map(|q| &q[..k]);
            sc.spawn(move || {
                let mut blk = [0.0f32; STQ_BLOCK];
                for row in chunk.chunks_mut(k) {
                    for (b, out) in row.chunks_mut(STQ_BLOCK).enumerate() {
                        blk.copy_from_slice(&out[..STQ_BLOCK]);
                        let qws = qw.map(|q| &q[b * STQ_BLOCK..][..STQ_BLOCK]);
                        stq_block_dispatch(kind, &blk, qws, out);
                    }
                }
            });
        }
    });
}

/// Weighted SSD of a round-trip vs `x` under the AngelSlim objective (for `--stq-check`).
pub fn stq_weighted_ssd(x: &[f32], y: &[f32], qw: Option<&[f32]>) -> f64 {
    let n = x.len();
    let sumx2: f32 = x.iter().map(|v| v * v).sum();
    let sigma2 = 2.0 * sumx2 / n as f32;
    let mut ssd = 0.0f64;
    for j in 0..n {
        let w = stq_weight(x[j], sigma2, qw.map(|q| q[j]));
        let e = x[j] - y[j];
        ssd += (w * e * e) as f64;
    }
    ssd
}

/// The REFERENCE STQ1_0 encoder (upstream: d = amax, zero = argmin|x| per stride-16 group) —
/// the baseline the LS+imatrix encoder is measured against in `--stq-check`.
pub fn stq1_0_block_reference(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), STQ_BLOCK);
    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }
    if !(amax > 0.0) {
        out.fill(0.0);
        return;
    }
    let d16 = round_f16(amax);
    for g in 0..STQ_BLOCK / 4 {
        let chunk = g / 16;
        let gloc = g % 16;
        let base = chunk * 64 + gloc;
        let mut zero_pos = 0usize;
        let mut smallest = f32::INFINITY;
        for p in 0..4 {
            let j = base + p * 16;
            let ax = x[j].abs();
            if ax < smallest {
                smallest = ax;
                zero_pos = p;
            }
        }
        for p in 0..4 {
            let j = base + p * 16;
            out[j] = if p == zero_pos { 0.0 } else if x[j] < 0.0 { -d16 } else { d16 };
        }
    }
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

// ---------------------------------------------------------------------------------------------
// Tensor grouping + the RUST_INFER_FAKE_QUANT spec
// ---------------------------------------------------------------------------------------------

/// Which family a tensor belongs to, for the "can this survive 4 bits?" experiment.
///
/// NOTE for the 122B MoE (Phase G): the same "must stay high precision" folklore is told about MoE
/// **router** layers. It does not apply to this dense family — there is no router — but when the MoE
/// lands, add a `Router` group here and put the claim through the same test rather than inheriting it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Group { Mlp, Attn, Gdn, LmHead, Embed, Mtp, Router, Expert, Hc, Ple, PleTable, Other }

pub fn group_of(name: &str) -> Group {
    // Routing gates FIRST (before the mtp/embed/mlp catch-alls) so `-router` holds BOTH the main AND the
    // MTP-head routers/shared-gates bf16. A quantized MTP router mis-routes the DRAFTER → tanks
    // speculative acceptance while output quality stays perfect (verify corrects it) — the invisible bug.
    if name.contains(".mlp.gate.weight") || name.contains(".shared_expert_gate.") { return Group::Router; }
    if name.contains(".router.") { return Group::Router; }   // hy_v3: mlp.router.gate.weight
    if name.starts_with("mtp.") { return Group::Mtp; }
    if name.contains(".eh_proj") { return Group::Mtp; }   // hy_v3: layer-80 MTP embed→hidden projection
    if name.contains("lm_head") { return Group::LmHead; }
    if name.contains("embed_tokens") { return Group::Embed; }
    // qwen4_exp (Qwen3.8-Flash-Next). The PLE n-gram table is NOT a GEMM weight (an embedding
    // gathered by row) — its own group so the quantizer can route it to the row-record codec. The
    // PLE projections (key_proj [hc*h, ple_dim], value_proj [h, ple_dim]) and the hyper-connection
    // mixers (input_mix_weight_down [lowrank, hc*h] / _up [hc*h, lowrank]) are ordinary GEMMs.
    // Checked BEFORE `.mlp.`: `mlp_hyper_connection` must land in Hc, not Mlp.
    if name.contains(".ngram_embedding.") { return Group::PleTable; }
    if name.contains(".ple.") { return Group::Ple; }
    if name.contains("hyper_connection") { return Group::Hc; }
    // MoE — test BEFORE the generic `.mlp.`: the stacked routed experts are their own group (the
    // sparse-quant risk lands here).
    if name.contains(".mlp.experts.") { return Group::Expert; }
    if name.contains(".mlp.") { return Group::Mlp; }   // dense MLP + the shared_expert MLP
    if name.contains(".self_attn.") { return Group::Attn; }
    if name.contains(".linear_attn.") {
        // Only the projections. conv1d / A_log / dt_bias / norm take the f32 path and are tiny.
        if name.contains("in_proj") || name.contains("out_proj") { return Group::Gdn; }
    }
    Group::Other
}

pub fn group_name(g: Group) -> &'static str {
    match g {
        Group::Mlp => "mlp", Group::Attn => "attn", Group::Gdn => "gdn",
        Group::LmHead => "lmhead", Group::Embed => "embed", Group::Mtp => "mtp",
        Group::Router => "router", Group::Expert => "expert", Group::Other => "other",
        Group::Hc => "hc", Group::Ple => "ple", Group::PleTable => "pletable",
    }
}

/// Parse `RUST_INFER_FAKE_QUANT` into a per-group format map.
///
/// Tokens are `group[:fmt]`, or `-group` to drop one. `fmt` is `nvfp4` (default) or `fp8`.
/// ```text
///   all                 every group at NVFP4
///   all,gdn:fp8         NVFP4 everywhere, but FP8 for the sensitive GDN projections  <-- the recipe
///   all,-gdn            NVFP4 everywhere, GDN left in bf16
///   mlp,attn            what the HF checkpoints actually do
/// ```
pub fn fake_quant_spec() -> Option<Vec<(Group, Fmt)>> {
    let spec = std::env::var("RUST_INFER_FAKE_QUANT").ok()?;
    parse_recipe(&spec)
}

/// The per-LAYER override half of a recipe spec (the same env var / `--recipe` string): tokens
/// `layers:<sel>:<fmt>` where `<sel>` is `lo-hi` (inclusive range) or a comma list of indices.
/// Layer rules WIN over the group map for the trunk layers they cover (last rule wins on
/// overlap). Used by the S5F2 clean-tap ladder (L1a: `layers:5,19,33,47,61:fp8` on top of
/// `all`; L1b: `layers:0-61:bf16` on top of `all` — the bf16 override = no fake-quant on the
/// clean prefix). `fmt:bf16` in a layer rule means "keep this layer at the raw bf16 weights".
pub fn fake_quant_layer_rules() -> Vec<LayerRule> {
    match std::env::var("RUST_INFER_FAKE_QUANT") {
        Ok(spec) => parse_layer_rules(&spec),
        Err(_) => Vec::new(),
    }
}

/// One per-layer precision override: trunk layers `[lo, hi]` (inclusive) at `fmt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerRule { pub lo: usize, pub hi: usize, pub fmt: Fmt }

/// Parse the `layers:` tokens out of a recipe spec string, in spec order (later wins).
/// `layers:0-61:bf16` (range) or `layers:5,19,33,47,61:fp8` (comma list).
pub fn parse_layer_rules(spec: &str) -> Vec<LayerRule> {
    let mut rules = Vec::new();
    for tok in spec.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        let tok = tok.strip_prefix('-').unwrap_or(tok);
        let Some(rest) = tok.strip_prefix("layers:") else { continue };
        let (sel, fmt) = match rest.rsplit_once(':') {
            Some((s, f)) => (s, f),
            None => continue,
        };
        let fmt = match fmt {
            "fp8" => Fmt::Fp8,
            "nvfp4" => Fmt::Nvfp4,
            "bf16" => Fmt::Bf16,
            _ => { eprintln!("RUST_INFER_FAKE_QUANT: unknown layer format {:?}", fmt); std::process::exit(1); }
        };
        // The selector is `lo-hi` (range) or a list separated by `,` or `+` (the `+` form
        // keeps the whole token comma-free so the recipe spec tokenizes cleanly).
        for part in sel.split([',', '+']) {
            let part = part.trim();
            if let Some((lo, hi)) = part.split_once('-') {
                let (lo, hi): (usize, usize) = match (lo.parse(), hi.parse()) {
                    (Ok(a), Ok(b)) if a <= b => (a, b),
                    _ => { eprintln!("RUST_INFER_FAKE_QUANT: bad layer range {:?}", part); std::process::exit(1); }
                };
                rules.push(LayerRule { lo, hi, fmt });
            } else if let Ok(li) = part.parse::<usize>() {
                rules.push(LayerRule { lo: li, hi: li, fmt });
            } else {
                eprintln!("RUST_INFER_FAKE_QUANT: bad layer selector {:?}", part);
                std::process::exit(1);
            }
        }
    }
    rules
}

/// The trunk layer index in a weight name (`...layers.<N>.<rest>` — both the qwen3.5
/// `model.language_model.layers.N.*` and the hy_v3 `model.layers.N.*` roots), or None.
pub fn layer_index_of(name: &str) -> Option<usize> {
    let stem = name.strip_suffix(".weight").unwrap_or(name);
    let idx = stem.rfind("layers.")?;
    let rest = &stem[idx + "layers.".len()..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() { return None; }
    digits.parse().ok()
}

/// Layer-aware format selection: a matching per-layer rule wins over the group map; otherwise
/// the group map decides (`fmt_for`). `fmt:bf16` from a rule returns Bf16 = "no fake-quant".
pub fn fmt_for_layer(map: &[(Group, Fmt)], rules: &[LayerRule], name: &str) -> Fmt {
    if let Some(li) = layer_index_of(name) {
        for r in rules.iter().rev() {
            if li >= r.lo && li <= r.hi { return r.fmt; }
        }
    }
    fmt_for(map, name)
}

/// Parse a recipe string into a per-group format map. `None` means "no quantization".
pub fn parse_recipe(spec: &str) -> Option<Vec<(Group, Fmt)>> {
    let spec = spec.trim().to_string();
    if spec.is_empty() || spec == "off" || spec == "none" { return None; }
    let all = [Group::Mlp, Group::Attn, Group::Gdn, Group::LmHead, Group::Embed, Group::Mtp,
               Group::Router, Group::Expert, Group::Hc, Group::Ple, Group::PleTable];
    let mut map: Vec<(Group, Fmt)> = Vec::new();
    for tok in spec.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        if tok.strip_prefix('-').unwrap_or(tok).starts_with("layers:") { continue; } // per-layer overrides
        let (neg, tok) = match tok.strip_prefix('-') { Some(r) => (true, r), None => (false, tok) };
        let (name, fmt) = match tok.split_once(':') {
            Some((n, "fp8")) => (n, Fmt::Fp8),
            Some((n, "nvfp4")) => (n, Fmt::Nvfp4),
            Some((n, "bf16")) => (n, Fmt::Bf16),
            Some((_, f)) => { eprintln!("RUST_INFER_FAKE_QUANT: unknown format {:?}", f); std::process::exit(1); }
            None => (tok, Fmt::Nvfp4),
        };
        let groups: Vec<Group> = match name {
            "all" => all.to_vec(),
            "mlp" => vec![Group::Mlp],
            "attn" => vec![Group::Attn],
            "gdn" => vec![Group::Gdn],
            "lmhead" => vec![Group::LmHead],
            "embed" => vec![Group::Embed],
            "mtp" => vec![Group::Mtp],
            "router" => vec![Group::Router],
            "expert" => vec![Group::Expert],
            "hc" => vec![Group::Hc],
            "ple" => vec![Group::Ple],
            "pletable" => vec![Group::PleTable],
            other => { eprintln!("RUST_INFER_FAKE_QUANT: unknown group {:?}", other); std::process::exit(1); }
        };
        for g in groups {
            map.retain(|&(x, _)| x != g);
            if !neg && fmt != Fmt::Bf16 { map.push((g, fmt)); }
        }
    }
    Some(map)
}

/// The format chosen for a tensor, or Bf16 if it is not in scope.
pub fn fmt_for(map: &[(Group, Fmt)], name: &str) -> Fmt {
    let g = group_of(name);
    map.iter().find(|&&(x, _)| x == g).map(|&(_, f)| f).unwrap_or(Fmt::Bf16)
}

pub fn fmt_name(f: Fmt) -> &'static str {
    match f { Fmt::Bf16 => "bf16", Fmt::Fp8 => "fp8", Fmt::Nvfp4 => "nvfp4" }
}

// ============================ MMA weight repack (the Marlin permutation) ============================
//
// The decode/verify GEMM runs on `mma.sync.m16n8k16` tensor cores, which demand that each lane of a
// warp already hold *specific* elements of the 16x16 A-tile in *specific* registers. Storing weights
// row-major forces every lane to gather its 4 elements from 2 rows K/2 bytes apart — 8 scattered
// sectors per load instruction. So we permute ONCE, at load, into exactly the order the fragment
// wants: lane L's whole A-fragment becomes ONE contiguous aligned load, and a warp's tile is one
// contiguous run of bytes. This is the single change that turns the load from a gather into a stream.
//
// The mma A-fragment for m16n8k16 (PTX ISA), with g = lane>>2 and t = lane&3, is 8 elements in 4
// 32-bit registers holding bf16 pairs:
//
//     ra[0] = { A[g  ][2t  ], A[g  ][2t+1] }        ra[2] = { A[g  ][2t+8], A[g  ][2t+9] }
//     ra[1] = { A[g+8][2t  ], A[g+8][2t+1] }        ra[3] = { A[g+8][2t+8], A[g+8][2t+9] }
//
// so lane L wants two rows (g, g+8) x two column-pairs (2t, 2t+8) of the tile — and nothing else.
//
// The K-tile is 16 wide, which is EXACTLY the NVFP4 scale-block size. One mma step therefore consumes
// exactly one scale block per row, so the block scale is a constant over the step and can be folded
// into the A-fragment before the mma — a per-*weight* cost that does not scale with N. That is the
// whole reason the kernel is flat in N.

/// M rows and K columns per MMA tile. Both fixed by the `m16n8k16` fragment shape.
pub const MMA_M: usize = 16;
pub const MMA_K: usize = 16;
/// Bytes per repacked tile: 16x16 nibbles = 128 B for NVFP4, 16x16 bytes = 256 B for FP8.
pub const MMA_TILE_FP4: usize = MMA_M * MMA_K / 2;
pub const MMA_TILE_FP8: usize = MMA_M * MMA_K;

/// Where element (row r, col c) of a tile lives inside a repacked NVFP4 tile.
/// Returns (byte offset within the 128-byte tile, true if the value is in the HIGH nibble).
#[inline]
fn fp4_tile_slot(r: usize, c: usize) -> (usize, bool) {
    let (g, hi_row) = (r & 7, r >> 3);         // lane group, and which of the fragment's two rows
    let (t, hi_col) = ((c & 7) >> 1, c >> 3);  // lane within group, and which column-pair
    let lane = g * 4 + t;
    let j = hi_row | (hi_col << 1);            // ra[j]: bit0 = row+8, bit1 = col+8
    (lane * 4 + j, (c & 1) == 1)               // odd column = high nibble (packing convention)
}

/// Where element (row r, col c) of a tile lives inside a repacked FP8 tile (byte offset).
#[inline]
fn fp8_tile_slot(r: usize, c: usize) -> usize {
    let (g, hi_row) = (r & 7, r >> 3);
    let (t, hi_col) = ((c & 7) >> 1, c >> 3);
    let lane = g * 4 + t;
    let j = (c & 1) | (hi_row << 1) | (hi_col << 2);
    lane * 8 + j
}

/// Permute a row-major NVFP4 tensor into MMA tile order.
///
/// In:  `qw` [M, K/2] packed nibbles (even col = low nibble), `sc` [M, K/16] E4M3 block scales.
/// Out: `wt` [M/16 * K/16 * 128] tiles, `st` [M/16 * K/16 * 16] scales (one per row of the tile).
///
/// Panics on M%16 or K%16 — every reduction and output dim in this model family is a multiple of 16,
/// and silently falling back to a slow path is how a "fast" engine ends up not being one.
pub fn repack_nvfp4_mma(qw: &[u8], sc: &[u8], m: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    assert!(m % MMA_M == 0 && k % MMA_K == 0, "MMA repack needs M,K % 16 == 0 (got {}x{})", m, k);
    // The GEMM walks k-blocks in ADJACENT PAIRS (so a warp's two scale reads land in one 32-byte DRAM
    // sector instead of wasting half of two). An odd k-block count would make it silently skip the last
    // block -- a wrong answer that still looks like a model. Every K in this family is a multiple of 32.
    assert!(k % 32 == 0, "the paired-k GEMM needs K % 32 == 0 (got K={})", k);
    let (ntm, nblk) = (m / MMA_M, k / MMA_K);
    let mut wt = vec![0u8; ntm * nblk * MMA_TILE_FP4];
    let mut st = vec![0u8; ntm * nblk * MMA_M];
    repack_driver(ntm * nblk, |t, wtc, stc| {
        let (mt, kb) = (t / nblk, t % nblk);
        repack_fp4_tile(qw, sc, k, nblk, mt, kb, wtc, stc);
    }, &mut wt, &mut st);
    (wt, st)
}

/// One (mt, kb) tile of the NVFP4 repack: 128 B of packed weights + 16 B of scales. Factored out so
/// the sequential and threaded drivers emit the same bytes through the SAME code path.
#[inline]
fn repack_fp4_tile(qw: &[u8], sc: &[u8], k: usize, nblk: usize, mt: usize, kb: usize,
                   wt_tile: &mut [u8], st_tile: &mut [u8]) {
    for r in 0..MMA_M {
        let row = mt * MMA_M + r;
        st_tile[r] = sc[row * nblk + kb];
        // Copy a byte at a time: source and destination pack the same (even,odd) column pair
        // into the same nibble positions, so whole bytes move without re-nibbling.
        for cp in 0..(MMA_K / 2) {
            let c = cp * 2;
            let (off, _) = fp4_tile_slot(r, c);
            wt_tile[off] = qw[row * (k / 2) + (kb * MMA_K + c) / 2];
        }
    }
}

/// Permute a row-major FP8 tensor [M, K] into MMA tile order. Row scales are unchanged: FP8 scales
/// are per output row, constant over K, so they fold into the f32 accumulator once at the end.
pub fn repack_fp8_mma(qw: &[u8], m: usize, k: usize) -> Vec<u8> {    assert!(m % MMA_M == 0 && k % MMA_K == 0, "MMA repack needs M,K % 16 == 0 (got {}x{})", m, k);
    let (ntm, nblk) = (m / MMA_M, k / MMA_K);
    let mut wt = vec![0u8; ntm * nblk * MMA_TILE_FP8];
    repack_driver(ntm * nblk, |t, wtc, _| {
        let (mt, kb) = (t / nblk, t % nblk);
        repack_fp8_tile(qw, k, mt, kb, wtc);
    }, &mut wt, &mut []);
    wt
}

/// One (mt, kb) tile of the FP8 repack: 256 B of weights (no per-tile scales — see above).
#[inline]
fn repack_fp8_tile(qw: &[u8], k: usize, mt: usize, kb: usize, wt_tile: &mut [u8]) {
    for r in 0..MMA_M {
        let row = mt * MMA_M + r;
        for c in 0..MMA_K {
            wt_tile[fp8_tile_slot(r, c)] = qw[row * k + kb * MMA_K + c];
        }
    }
}

/// Quantize a bf16 weight [M, K] into the `gemm_dsv4_fp8_bsb` device format (the engine's
/// `Fp8Weight`): MMA-repacked e4m3 codes + UE8M0 block scales [M/128, K/128] (row-major,
/// matching the kernel's `Sb[(mt>>3)*nkb + kb]`). Per 128-row × 128-K block: amax (floored
/// at 1e-4) → s = 2^ceil(log2(amax/448)) (UE8M0, always up — the §C.1 convention) →
/// codes = e4m3(x/s). One-time load-path quantizer (the DSpark fp8 draft heads behind
/// GB10_DSPARK_FP8_LOGITS) — NOT the serving GEMM's weights (those ship quantized).
/// Returns (repacked codes, sb bytes).
pub fn quantize_fp8_bsb(w: &[bf16], m: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(w.len(), m * k, "shape mismatch");
    assert!(m % 128 == 0 && k % 128 == 0, "fp8_bsb quantize needs M,K % 128 (got {m}x{k})");
    let nkb = k / 128;
    let nbm = m / 128;
    let mut codes = vec![0u8; m * k];
    let mut sb = vec![0u8; nbm * nkb];
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).max(1);
    let bms_per = nbm.div_ceil(nthreads).max(1);
    std::thread::scope(|sc| {
        for (t, (cp, sp)) in codes
            .chunks_mut(bms_per * 128 * k)
            .zip(sb.chunks_mut(bms_per * nkb))
            .enumerate()
        {
            sc.spawn(move || {
                let bm0 = t * bms_per;
                for bm in bm0..(bm0 + bms_per).min(nbm) {
                    let cl = &mut cp[(bm - bm0) * 128 * k..][..128 * k];
                    let sl = &mut sp[(bm - bm0) * nkb..][..nkb];
                    for bk in 0..nkb {
                        let mut amax = 0.0f32;
                        for r in 0..128 {
                            let row = &w[(bm * 128 + r) * k + bk * 128..][..128];
                            for x in row {
                                amax = amax.max(x.to_f32().abs());
                            }
                        }
                        amax = amax.max(1e-4);
                        let s = crate::dsv4_cpu::fast_round_scale(amax, 1.0 / 448.0);
                        sl[bk] = (s.to_bits() >> 23) as u8; // UE8M0 byte (s is a normal pow2)
                        let inv = 1.0 / s;
                        for r in 0..128 {
                            let src = &w[(bm * 128 + r) * k + bk * 128..][..128];
                            let dst = &mut cl[r * k + bk * 128..][..128];
                            for (d, x) in dst.iter_mut().zip(src) {
                                *d = f32_to_e4m3(x.to_f32() * inv);
                            }
                        }
                    }
                }
            });
        }
    });
    (repack_fp8_mma(&codes, m, k), sb)
}


/// Don't spawn a thread for less work than this: a thread costs tens of µs, 4096 tiles (~0.5 MB)
/// costs ~2-3 ms single-core — below that the spawn overhead is the optimization.
const REPACK_MIN_TILES_PER_THREAD: usize = 4096;

thread_local! {
    static REPACK_THREADING: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Kill switch for the per-call threading, per thread. The load pipeline's WORKERS set this off:
/// the worker pool IS the parallelism there — 8 workers each spawning 16 inner threads is a
/// 6x-oversubscribed thrash on 20 cores (measured: pipeline crawls). The shard-loop inline repack
/// and any serial caller keep the default (threaded, GPU idle during load).
pub fn set_repack_threading(on: bool) { REPACK_THREADING.with(|c| c.set(on)); }

/// Drive a per-tile repack body over `total_tiles` tiles. Small inputs run sequentially on the
/// caller; big inputs partition the tiles into disjoint contiguous ranges across `std::thread::scope`
/// workers. Every output byte is a pure function of the input bytes and the tile index, so the
/// threaded result is BYTE-IDENTICAL to the sequential one — that is the load-correctness contract.
fn repack_driver<F: Fn(usize, &mut [u8], &mut [u8]) + Sync>(
    total_tiles: usize, body: F, wt: &mut [u8], st: &mut [u8],
) {
    let wt_tile = wt.len() / total_tiles.max(1);
    let st_tile = st.len() / total_tiles.max(1);
    let max_t = if REPACK_THREADING.with(|c| c.get()) {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(16)
    } else { 1 };
    let nthreads = max_t.min(total_tiles.div_ceil(REPACK_MIN_TILES_PER_THREAD)).max(1);
    if nthreads <= 1 {
        for t in 0..total_tiles {
            body(t, &mut wt[t * wt_tile..(t + 1) * wt_tile],
                 if st_tile > 0 { &mut st[t * st_tile..(t + 1) * st_tile] } else { &mut [] });
        }
        return;
    }
    let per = total_tiles.div_ceil(nthreads);
    std::thread::scope(|s| {
        let mut wt_rest: &mut [u8] = wt;
        let mut st_rest: &mut [u8] = st;
        let mut begin = 0usize;
        while begin < total_tiles {
            let n = per.min(total_tiles - begin);
            let (wtc, wtl) = wt_rest.split_at_mut(n * wt_tile);
            let (stc, stl) = st_rest.split_at_mut(n * st_tile);
            wt_rest = wtl;
            st_rest = stl;
            let body = &body;
            s.spawn(move || {
                for t in begin..begin + n {
                    let i = t - begin;
                    body(t, &mut wtc[i * wt_tile..(i + 1) * wt_tile],
                         if st_tile > 0 { &mut stc[i * st_tile..(i + 1) * st_tile] } else { &mut [] });
                }
            });
            begin += n;
        }
    });
}

// ================================ Weight FUSION ================================
//
// Several projections in this architecture read the SAME activation and are separate tensors only
// because the checkpoint stores them that way:
//
//   GDN:       in_proj_qkv + in_proj_z + in_proj_b + in_proj_a   (all read the normed hidden)
//   Attention: q_proj + k_proj + v_proj                          (likewise)
//
// Running them as separate GEMMs is a disaster at the small end. `in_proj_b`/`in_proj_a` have
// M = num_value_heads (32 on 9B), so `grid = M/16 = 2` — TWO blocks on a 48-SM GPU. Measured: 26 us
// to move 74 KB (2.8 GB/s against a 234 GB/s machine), and across 24 GDN layers that is **4.7% of all
// GEMM time to move 0.03% of the bytes**. Concatenating the four along M turns four launches into one,
// lets the tiny tensors ride inside a big efficient kernel, and lengthens the surviving kernel (which
// also helps the ramp/tail problem). vLLM and SGLang fuse QKV and gate/up at load for the same reason.
//
// The one real constraint is quantization metadata. NVFP4's `global_scale` is PER TENSOR, so a fused
// weight has several. But every segment boundary here is a multiple of 16 (conv_dim, value_dim and
// num_heads all are), so each 16-row MMA tile lies entirely within ONE source tensor — and the scale
// can be a per-TILE lookup the kernel reads once per block. No requantization, no loss of precision.

/// Concatenate NVFP4 tensors along M (rows). All must share K.
///
/// Returns the row-major concatenation plus `gs_tile[M/16]`: the reciprocal global scale that applies
/// to each 16-row tile. Feed the first two through `repack_nvfp4_mma`.
pub fn fuse_nvfp4(parts: &[(&[u8], &[u8], f32, usize)], k: usize) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    assert!(k % BLOCK == 0);
    let (mut qw, mut sc, mut gs) = (Vec::new(), Vec::new(), Vec::new());
    for &(pq, ps, inv_gs, m) in parts {
        assert!(m % MMA_M == 0, "fused segment M={} must be a multiple of {}", m, MMA_M);
        assert_eq!(pq.len(), m * k / 2, "fused segment qweight size");
        assert_eq!(ps.len(), m * k / BLOCK, "fused segment scales size");
        qw.extend_from_slice(pq);
        sc.extend_from_slice(ps);
        gs.extend(std::iter::repeat(inv_gs).take(m / MMA_M));   // one entry per 16-row tile
    }
    (qw, sc, gs)
}

/// Concatenate FP8 tensors along M. FP8 scales are already per output row, so they just concatenate.
pub fn fuse_fp8(parts: &[(&[u8], &[f32], usize)], k: usize) -> (Vec<u8>, Vec<f32>) {
    let (mut qw, mut rs) = (Vec::new(), Vec::new());
    for &(pq, prs, m) in parts {
        assert!(m % MMA_M == 0, "fused segment M={} must be a multiple of {}", m, MMA_M);
        assert_eq!(pq.len(), m * k, "fused segment fp8 size");
        assert_eq!(prs.len(), m, "fused segment row_scale size");
        qw.extend_from_slice(pq);
        rs.extend_from_slice(prs);
    }
    (qw, rs)
}

// ================================ DRAFT-VOCAB row subset ================================
//
// Picking a draft token needs an argmax over the vocabulary, i.e. a second full read of the LM head.
// On 9B that is 572 MB -- **2.75 ms, 11% of a decode step -- and it is paid (depth-1) times per
// speculative step**, which makes it essentially the entire slope of r(d). It is the single biggest
// cost left in speculation.
//
// The fix (FR-Spec): give the DRAFTER a smaller LM head. Rank the vocabulary by corpus frequency,
// keep the top slice, and let the drafter propose only from that. The VERIFY keeps the full head, so:
//
//   * **Greedy stays exactly lossless.** The proposal mechanism only affects how OFTEN a draft
//     matches; every emitted token is still the full model's argmax. A token outside the subset simply
//     never gets proposed, so that position is always rejected -- costing acceptance, never correctness.
//   * **Stochastic stays distribution-exact.** If the drafter samples from the RENORMALIZED restricted
//     softmax, then that restricted distribution *is* `q`, and `min(1, p/q)` is the standard
//     Leviathan/Chen scheme with a perfectly valid proposal. Tokens outside the subset have q=0 and can
//     only enter through the residual `(p-q)+` resample -- which is over the FULL vocab on the verify
//     side, which we already pay for. No approximation anywhere.
//
// Rows are independent in both codecs (NVFP4 has per-row block scales + one tensor scale; FP8 has one
// scale per row), so a row subset is EXACT -- no requantization, no loss.

/// Take a subset of rows from a row-major NVFP4 tensor. Exact: rows do not interact.
pub fn subset_rows_nvfp4(qw: &[u8], sc: &[u8], k: usize, rows: &[u32]) -> (Vec<u8>, Vec<u8>) {
    let nblk = k / BLOCK;
    let (rb, sb) = (k / 2, nblk);
    let mut oq = Vec::with_capacity(rows.len() * rb);
    let mut os = Vec::with_capacity(rows.len() * sb);
    for &r in rows {
        let r = r as usize;
        oq.extend_from_slice(&qw[r * rb..(r + 1) * rb]);
        os.extend_from_slice(&sc[r * sb..(r + 1) * sb]);
    }
    (oq, os)
}

/// Take a subset of rows from a row-major FP8 tensor (row scales come along).
pub fn subset_rows_fp8(qw: &[u8], rs: &[f32], k: usize, rows: &[u32]) -> (Vec<u8>, Vec<f32>) {
    let mut oq = Vec::with_capacity(rows.len() * k);
    let mut os = Vec::with_capacity(rows.len());
    for &r in rows {
        let r = r as usize;
        oq.extend_from_slice(&qw[r * k..(r + 1) * k]);
        os.push(rs[r]);
    }
    (oq, os)
}

/// The rows a draft head should keep: the `top` most-frequent tokens, PLUS the tail of the vocabulary.
///
/// The tail matters more than it looks. Special/added tokens (`<|im_end|>`, `<|endoftext|>`, the tool
/// and think markers) live at the TOP of the id range by convention, and the model must be able to
/// emit them -- `<|im_end|>` is how a chat turn STOPS. A drafter that can never propose it would fail
/// to draft the single most predictable token in the whole conversation.
///
/// Qwen's BPE ids are ordered by merge rank, which tracks training-corpus frequency (`Ġwould` is id
/// 1000; Thai and Arabic start around 150k). Measured on prose+code, the top 65536 ids (26% of the
/// vocabulary) cover **97.5%** of emitted tokens.
///
/// Result length is padded up to a multiple of 16 (the MMA tile height) with real rows, never dummies:
/// a zero row would have logit 0 and could WIN an argmax against all-negative logits.
pub fn draft_vocab_rows(top: usize, vocab: usize) -> Vec<u32> {
    const TAIL: usize = 512;                       // covers every special/added token, with margin
    // hy_v3's specials start at 120000 (`<｜hy_eos…｜>`, think/tool markers) — the generic
    // 512-row tail (120320+) would miss them all, and a drafter that cannot propose hy_eos
    // can never draft the end of a chat turn. Cover the whole special block for that vocab.
    let tail_start = if vocab == 120832 { 120000 } else { vocab.saturating_sub(TAIL) };
    let mut top = top.min(tail_start);
    // grow `top` (with real, more-frequent tokens) until the total is a multiple of 16
    while (top + (vocab - tail_start)) % MMA_M != 0 { top += 1; }
    let mut rows: Vec<u32> = (0..top as u32).collect();
    rows.extend(tail_start as u32..vocab as u32);
    rows
}

/// FR-Spec subset from an explicit id file (RUST_INFER_DRAFT_VOCAB_FILE) — the corpus-ranked
/// variant of `draft_vocab_rows` (the offline artifact step produces this from real frequencies).
/// Format: whitespace/newline-separated decimal token ids. The special-token tail is appended
/// (same rule as `draft_vocab_rows`, so a corpus list that omits specials stays chat-safe), then
/// id-order tokens fill the last partial 16-row tile. Rows are independent in every codec, so ANY
/// subset is exact — the file only chooses WHICH tokens are proposable.
pub fn draft_vocab_rows_file(path: &str, vocab: usize) -> std::io::Result<Vec<u32>> {
    let txt = std::fs::read_to_string(path)?;
    let mut rows: Vec<u32> = txt.split_whitespace()
        .filter_map(|t| t.parse::<u32>().ok())
        .filter(|&i| (i as usize) < vocab)
        .collect();
    rows.sort_unstable();
    rows.dedup();
    const TAIL: usize = 512;
    let tail_start = if vocab == 120832 { 120000 } else { vocab.saturating_sub(TAIL) };
    let mut next: std::collections::HashSet<u32> = rows.iter().copied().collect();
    for i in tail_start as u32..vocab as u32 {
        if next.insert(i) { rows.push(i); }
    }
    let mut fill = 0u32;
    while rows.len() % MMA_M != 0 {
        while !next.insert(fill) { fill += 1; }
        rows.push(fill);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_roundtrip_is_a_float_not_an_int() {
        // The bug that motivates this test: treating the E4M3 byte as an integer.
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert!((e4m3_to_f32(f32_to_e4m3(1.0)) - 1.0).abs() < 1e-6);
        assert!((e4m3_to_f32(f32_to_e4m3(448.0)) - 448.0).abs() < 1e-3);
        // A code whose integer value bears no relation to its float value.
        let v = e4m3_to_f32(0x38); // exp=7, man=0 -> 1.0
        assert!((v - 1.0).abs() < 1e-6, "0x38 should decode to 1.0, got {}", v);
    }

    #[test]
    fn e2m1_grid_is_exact() {
        for (i, &v) in E2M1.iter().enumerate() {
            assert_eq!(e2m1_to_f32(i as u8), v);
            assert!((e2m1_to_f32(f32_to_e2m1(v)) - v).abs() < 1e-6);
            assert!((e2m1_to_f32(f32_to_e2m1(-v)) + v).abs() < 1e-6);
        }
        // Clamps at the top of the grid.
        assert_eq!(e2m1_to_f32(f32_to_e2m1(100.0)), 6.0);
    }

    #[test]
    fn exactly_representable_block_survives_roundtrip() {
        // A block whose values all sit on the E2M1 grid times one scale must round-trip exactly.
        let k = 16;
        let vals: Vec<bf16> = E2M1.iter().chain(E2M1.iter())
            .map(|&v| bf16::from_f32(v * 2.0)).collect();
        assert_eq!(vals.len(), k);
        let q = quantize_nvfp4(&vals, 1, k);
        let d = dequantize_nvfp4(&q);
        for (a, b) in vals.iter().zip(d.iter()) {
            assert!((a.to_f32() - b.to_f32()).abs() < 1e-3, "{} vs {}", a, b);
        }
    }

    /// The tile permutation must be a BIJECTION. If two (r,c) map to one slot, weights are silently
    /// overwritten and the model is quietly wrong — it still loads, still generates, just worse.
    #[test]
    fn mma_tile_slots_are_bijective() {
        let mut seen4 = [false; MMA_TILE_FP4 * 2];   // 128 bytes x 2 nibbles
        let mut seen8 = [false; MMA_TILE_FP8];
        for r in 0..MMA_M {
            for c in 0..MMA_K {
                let (off, hi) = fp4_tile_slot(r, c);
                let idx = off * 2 + hi as usize;
                assert!(!seen4[idx], "fp4 slot collision at ({},{})", r, c);
                seen4[idx] = true;
                let o8 = fp8_tile_slot(r, c);
                assert!(!seen8[o8], "fp8 slot collision at ({},{})", r, c);
                seen8[o8] = true;
            }
        }
        assert!(seen4.iter().all(|&b| b) && seen8.iter().all(|&b| b), "tile not fully covered");
    }

    /// Repack, then walk the inverse map the CUDA `*_tiled` kernels use, and demand the original back.
    #[test]
    fn mma_repack_roundtrips() {
        let (m, k) = (32usize, 64usize);
        let mut s = 99u64;
        let w: Vec<bf16> = (0..m * k).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            bf16::from_f32(((s >> 33) as f32 / 2f32.powi(31) - 0.5) * 0.2)
        }).collect();

        let q4 = quantize_nvfp4(&w, m, k);
        let (wt, st) = repack_nvfp4_mma(&q4.qweight, &q4.scales, m, k);
        let nblk = k / MMA_K;
        for row in 0..m {
            for c in 0..k {
                let (mt, kb) = (row / MMA_M, c / MMA_K);
                let (off, hi) = fp4_tile_slot(row % MMA_M, c % MMA_K);
                let byte = wt[(mt * nblk + kb) * MMA_TILE_FP4 + off];
                let got = if hi { byte >> 4 } else { byte & 0x0F };
                let want = { let b = q4.qweight[row * (k / 2) + c / 2];
                             if c % 2 == 1 { b >> 4 } else { b & 0x0F } };
                assert_eq!(got, want, "fp4 nibble ({},{})", row, c);
                assert_eq!(st[(mt * nblk + kb) * MMA_M + row % MMA_M],
                           q4.scales[row * nblk + kb], "fp4 scale ({},{})", row, c);
            }
        }

        let q8 = quantize_fp8(&w, m, k);
        let wt8 = repack_fp8_mma(&q8.qweight, m, k);
        for row in 0..m {
            for c in 0..k {
                let (mt, kb) = (row / MMA_M, c / MMA_K);
                let off = fp8_tile_slot(row % MMA_M, c % MMA_K);
                assert_eq!(wt8[(mt * nblk + kb) * MMA_TILE_FP8 + off], q8.qweight[row * k + c],
                           "fp8 byte ({},{})", row, c);
            }
        }
    }

    #[test]
    fn roundtrip_error_is_bounded() {
        // Gaussian weights: relative error should land in the few-percent range, not blow up.
        let (m, k) = (8usize, 256usize);
        let mut s = 12345u64;
        let w: Vec<bf16> = (0..m * k).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / (2f32.powi(31));
            bf16::from_f32((u - 0.5) * 0.1)
        }).collect();
        let q = quantize_nvfp4(&w, m, k);
        let d = dequantize_nvfp4(&q);
        let (rel, _) = roundtrip_error(&w, &d);
        assert!(rel < 0.15, "relative error {} too large", rel);
        assert_eq!(q.qweight.len(), m * k / 2);
        assert_eq!(q.scales.len(), m * k / BLOCK);
    }
}

#[cfg(test)]
mod q2_tests {
    use super::*;
    #[test]
    fn layer_rules_parse_and_select() {
        // S5F2 L1 ladder spec: `layers:<sel>:<fmt>` tokens (list via `+`/`,`, range via `-`).
        let rules = parse_layer_rules("all,layers:5+19+33+47+61:fp8");
        assert_eq!(rules.len(), 5);
        assert!(rules.iter().all(|r| r.fmt == Fmt::Fp8));
        let rules2 = parse_layer_rules("all,layers:0-61:bf16");
        assert_eq!(rules2.len(), 1);
        assert_eq!((rules2[0].lo, rules2[0].hi, rules2[0].fmt), (0, 61, Fmt::Bf16));
        // layer_index_of: both qwen3.5 (`model.language_model.layers.N.*`) and hy_v3 roots.
        assert_eq!(layer_index_of("model.language_model.layers.5.self_attn.q_proj.weight"), Some(5));
        assert_eq!(layer_index_of("model.layers.61.mlp.down_proj.weight"), Some(61));
        assert_eq!(layer_index_of("lm_head.weight"), None);
        assert_eq!(layer_index_of("mtp.layers.0.mlp.gate_up_proj.weight"), Some(0));
        // Layer rules WIN over the group map; non-layered tensors fall back to the group map.
        let map = parse_recipe("all").unwrap();
        let f1 = fmt_for_layer(&map, &rules, "model.language_model.layers.19.self_attn.q_proj.weight");
        let f2 = fmt_for_layer(&map, &rules, "model.language_model.layers.20.self_attn.q_proj.weight");
        let f3 = fmt_for_layer(&map, &rules, "lm_head.weight");
        assert!(matches!(f1, Fmt::Fp8), "tap-layer override must win: {f1:?}");
        assert!(matches!(f2, Fmt::Nvfp4), "non-tap layer falls back to the group map: {f2:?}");
        assert!(matches!(f3, Fmt::Nvfp4), "non-layered tensor uses the group map: {f3:?}");
        // bf16 override = "no fake-quant" on the covered layers.
        let f4 = fmt_for_layer(&map, &rules2, "model.language_model.layers.5.self_attn.q_proj.weight");
        assert!(matches!(f4, Fmt::Bf16));
    }
    #[test]
    fn q2_levels_assigned() {
        // one block with a known spread: expect all four levels to appear
        let mut w: Vec<bf16> = vec![-0.249, -0.994, 0.497, -0.994, -1.491, 0.746, -0.249, 0.746,
                                    -0.994, 0.994, -0.994, -0.994, 1.491, 0.0, -0.249, 1.491]
            .iter().map(|&x| bf16::from_f32(x)).collect();
        // scale so the block amax is exactly Q2_MAX (s = e4m3(1)*st ≈ 1.0 path)
        let k = 16;
        let orig: Vec<f32> = w.iter().map(|x| x.to_f32()).collect();
        fake_quant_q2(&mut w, 1, k);
        let got: Vec<f32> = w.iter().map(|x| x.to_f32()).collect();
        eprintln!("orig: {:?}", orig);
        eprintln!("got:  {:?}", got);
        let has_small = got.iter().any(|v| v.abs() < 1.0);
        assert!(has_small, "q2 collapsed everything to the extreme level: {:?}", got);
    }
}
