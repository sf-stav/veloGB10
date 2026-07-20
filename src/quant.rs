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

/// Simulated quantization: NVFP4 round-trip in place. Bytes stay bf16, values carry the 4-bit error.
pub fn fake_quant_nvfp4(w: &mut [bf16], m: usize, k: usize) {
    let q = quantize_nvfp4(w, m, k);
    w.copy_from_slice(&dequantize_nvfp4(&q));
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
pub enum Group { Mlp, Attn, Gdn, LmHead, Embed, Mtp, Router, Expert, Other }

pub fn group_of(name: &str) -> Group {
    // Routing gates FIRST (before the mtp/embed/mlp catch-alls) so `-router` holds BOTH the main AND the
    // MTP-head routers/shared-gates bf16. A quantized MTP router mis-routes the DRAFTER → tanks
    // speculative acceptance while output quality stays perfect (verify corrects it) — the invisible bug.
    if name.contains(".mlp.gate.weight") || name.contains(".shared_expert_gate.") { return Group::Router; }
    if name.starts_with("mtp.") { return Group::Mtp; }
    if name.contains("lm_head") { return Group::LmHead; }
    if name.contains("embed_tokens") { return Group::Embed; }
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

/// Parse a recipe string into a per-group format map. `None` means "no quantization".
pub fn parse_recipe(spec: &str) -> Option<Vec<(Group, Fmt)>> {
    let spec = spec.trim().to_string();
    if spec.is_empty() || spec == "off" || spec == "none" { return None; }
    let all = [Group::Mlp, Group::Attn, Group::Gdn, Group::LmHead, Group::Embed, Group::Mtp,
               Group::Router, Group::Expert];
    let mut map: Vec<(Group, Fmt)> = Vec::new();
    for tok in spec.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
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
    for mt in 0..ntm {
        for kb in 0..nblk {
            let base = (mt * nblk + kb) * MMA_TILE_FP4;
            for r in 0..MMA_M {
                let row = mt * MMA_M + r;
                st[(mt * nblk + kb) * MMA_M + r] = sc[row * nblk + kb];
                // Copy a byte at a time: source and destination pack the same (even,odd) column pair
                // into the same nibble positions, so whole bytes move without re-nibbling.
                for cp in 0..(MMA_K / 2) {
                    let c = cp * 2;
                    let (off, _) = fp4_tile_slot(r, c);
                    wt[base + off] = qw[row * (k / 2) + (kb * MMA_K + c) / 2];
                }
            }
        }
    }
    (wt, st)
}

/// Permute a row-major FP8 tensor [M, K] into MMA tile order. Row scales are unchanged: FP8 scales
/// are per output row, constant over K, so they fold into the f32 accumulator once at the end.
pub fn repack_fp8_mma(qw: &[u8], m: usize, k: usize) -> Vec<u8> {
    assert!(m % MMA_M == 0 && k % MMA_K == 0, "MMA repack needs M,K % 16 == 0 (got {}x{})", m, k);
    let (ntm, nblk) = (m / MMA_M, k / MMA_K);
    let mut wt = vec![0u8; ntm * nblk * MMA_TILE_FP8];
    for mt in 0..ntm {
        for kb in 0..nblk {
            let base = (mt * nblk + kb) * MMA_TILE_FP8;
            for r in 0..MMA_M {
                let row = mt * MMA_M + r;
                for c in 0..MMA_K {
                    wt[base + fp8_tile_slot(r, c)] = qw[row * k + kb * MMA_K + c];
                }
            }
        }
    }
    wt
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
    let tail_start = vocab.saturating_sub(TAIL);
    let mut top = top.min(tail_start);
    // grow `top` (with real, more-frequent tokens) until the total is a multiple of 16
    while (top + (vocab - tail_start)) % MMA_M != 0 { top += 1; }
    let mut rows: Vec<u32> = (0..top as u32).collect();
    rows.extend(tail_start as u32..vocab as u32);
    rows
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
