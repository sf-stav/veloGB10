//! Numerics primitives of the CPU reference: bf16 rounding, E4M3 / E2M1 RNE casts, UE8M0 scale tricks,
//! QAT-sim round-trips, deterministic parallel row helpers and the GEMMs. Bit-faithful by construction (see `mod.rs` policy).
//! Siblings: `ops.rs` builds on these; `mod.rs` has the map.

use super::*;

// ---------------------------------------------------------------------------
// bf16 rounding helpers
// ---------------------------------------------------------------------------

/// Round f32 -> bf16 (RNE) -> f32. Every point the reference materializes a
/// bf16 tensor goes through this.
#[inline]
pub fn bf(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

#[inline]
pub fn round_bf16(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = bf(*x);
    }
}

// ---------------------------------------------------------------------------
// RNE cast encoders (§C.1–2: `cvt.rn` semantics — ties-to-even, sign-of-zero)
// ---------------------------------------------------------------------------

/// f32 -> FP8-E4M3 byte, round-to-nearest-even on the target grid, satfinite.
/// Input is expected pre-clamped to ±448 by the caller (the quant kernels clamp
/// before casting); we still saturate for safety. `quant::f32_to_e4m3` agrees
/// on every value except exact ties (it rounds ties down) and -0.0 (it returns
/// +0); this version is the exact `cvt.rn.satfinite.e4m3x2.f32` behavior the
/// tilelang kernels get from hardware. Proven equivalent in tests.
pub fn f32_to_e4m3_rne(x: f32) -> u8 {
    const MAX: f32 = 448.0;
    if x.is_nan() {
        return 0x7F; // e4m3 NaN
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0u8 };
    let a = x.abs().min(MAX);
    if a == 0.0 {
        return sign; // preserve -0.0 like cvt.rn
    }
    // The 127 finite non-negative codes are monotonic; locate the bracketing pair.
    let t = e4m3_pos_table();
    let hi = t.partition_point(|&v| v < a);
    let code = if hi == 0 {
        0usize
    } else if hi >= 127 {
        126usize
    } else {
        let d_hi = t[hi] - a;
        let d_lo = a - t[hi - 1];
        if d_hi < d_lo {
            hi
        } else if d_hi > d_lo {
            hi - 1
        } else if hi % 2 == 0 {
            hi // exact tie -> even code (RNE)
        } else {
            hi - 1
        }
    };
    sign | code as u8
}

fn e4m3_pos_table() -> &'static [f32; 127] {
    static T: OnceLock<[f32; 127]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0.0f32; 127];
        for (c, slot) in t.iter_mut().enumerate() {
            *slot = e4m3_to_f32(c as u8);
        }
        t
    })
}

/// f32 -> FP4-E2M1 nibble (0..=15), round-to-nearest-even, sign-of-zero kept.
/// Caller clamps to ±6 (we saturate regardless). quant::f32_to_e2m1 is also
/// ties-to-even but flattens -0.0; proven equivalent on non-zero values in tests.
pub fn f32_to_e2m1_rne(x: f32) -> u8 {
    const T: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    if x.is_nan() {
        return 0x7; // e2m1fn NaN (S.111)
    }
    let sign = if x.is_sign_negative() { 0x8u8 } else { 0u8 };
    let a = x.abs().min(6.0);
    if a == 0.0 {
        return sign;
    }
    let hi = T.partition_point(|&v| v < a);
    let code = if hi == 0 {
        0usize
    } else if hi >= 8 {
        7usize
    } else {
        let d_hi = T[hi] - a;
        let d_lo = a - T[hi - 1];
        if d_hi < d_lo {
            hi
        } else if d_hi > d_lo {
            hi - 1
        } else if hi % 2 == 0 {
            hi
        } else {
            hi - 1
        }
    };
    sign | code as u8
}

/// Pack two E2M1 nibbles into one byte: low nibble = even K index (§A.2, §C.2).
/// The QAT-sim path never packs (it round-trips values), but the convention is
/// load-bearing for Lane A's repack and pinned down by a unit test here.
#[inline]
pub fn pack_e2m1_pair(even_k: u8, odd_k: u8) -> u8 {
    (even_k & 0xF) | ((odd_k & 0xF) << 4)
}

// ---------------------------------------------------------------------------
// UE8M0 scale bit tricks (kernel.py:22-37, replicated operation-for-operation)
// ---------------------------------------------------------------------------

/// `fast_log2_ceil`: ceil(log2(x)) for x > 0 via IEEE 754 bits — including the
/// kernel's exact subnormal behavior (exp field 0 -> -127, +1 if mantissa != 0).
#[inline]
pub fn fast_log2_ceil(x: f32) -> i32 {
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x7F_FFFF;
    exp - 127 + if man != 0 { 1 } else { 0 }
}

/// `fast_pow2`: 2^x for integer x via IEEE bits (valid for x in [-126, 127];
/// the quant callers' amax floors keep the argument in range).
#[inline]
pub fn fast_pow2(x: i32) -> f32 {
    f32::from_bits(((x + 127) as u32) << 23)
}

/// `fast_round_scale(amax, 1/max)`: 2^ceil(log2(amax/max)) — always rounds UP.
#[inline]
pub fn fast_round_scale(amax: f32, max_inv: f32) -> f32 {
    fast_pow2(fast_log2_ceil(amax * max_inv))
}

// ---------------------------------------------------------------------------
// QAT-sim round-trips (§C.1 act_quant inplace, §C.2 fp4_act_quant inplace)
// ---------------------------------------------------------------------------

/// FP8-E4M3 QAT-sim, in place, over bf16-valued f32 data: per row per group,
/// `amax = max|x|` floored at 1e-4; `s = 2^ceil(log2(amax/448))` (UE8M0, always
/// up); `x <- bf16(f32(fp8_RNE(clamp(x/s, ±448))) * s)`. The product fp8·s is
/// exact in f32 (≤4 significand bits × pow2), so the bf16 store is exact — the
/// round-trip value is exactly `code·s`.
pub fn act_quant_sim(x: &mut [f32], rows: usize, n: usize, group: usize) {
    assert_eq!(n % group, 0, "act_quant_sim: n={} not divisible by group={}", n, group);
    const FP8_MAX_INV: f32 = 1.0 / 448.0;
    for r in 0..rows {
        let row = &mut x[r * n..(r + 1) * n];
        for g in 0..n / group {
            let blk = &mut row[g * group..(g + 1) * group];
            let mut amax = 0.0f32;
            for &v in blk.iter() {
                amax = amax.max(v.abs());
            }
            amax = amax.max(1e-4);
            let s = fast_round_scale(amax, FP8_MAX_INV);
            for v in blk.iter_mut() {
                let q = f32_to_e4m3_rne((*v / s).clamp(-448.0, 448.0));
                *v = bf(e4m3_to_f32(q) * s);
            }
        }
    }
}

/// FP4-E2M1 QAT-sim, in place: group 32; `amax` floored at `6·2^-126`;
/// `s = 2^ceil(log2(amax/6))`; `x <- bf16(f32(fp4_RNE(clamp(x/s, ±6))) * s)`.
pub fn fp4_act_quant_sim(x: &mut [f32], rows: usize, n: usize, group: usize) {
    assert_eq!(n % group, 0, "fp4_act_quant_sim: n={} not divisible by group={}", n, group);
    const FP4_MAX_INV: f32 = 1.0 / 6.0;
    let floor = 6.0f32 * 2f32.powi(-126);
    for r in 0..rows {
        let row = &mut x[r * n..(r + 1) * n];
        for g in 0..n / group {
            let blk = &mut row[g * group..(g + 1) * group];
            let mut amax = 0.0f32;
            for &v in blk.iter() {
                amax = amax.max(v.abs());
            }
            amax = amax.max(floor);
            let s = fast_round_scale(amax, FP4_MAX_INV);
            for v in blk.iter_mut() {
                let q = f32_to_e2m1_rne((*v / s).clamp(-6.0, 6.0));
                *v = bf(e2m1_to_f32(q) * s);
            }
        }
    }
}

/// Non-inplace FP8 activation quant for the quant GEMMs (§C.1 non-inplace path,
/// group always 128): returns (code values as f32, ue8m0 scales as f32), one
/// scale per 128-wide group per row.
pub fn act_quant_codes(x: &[f32], rows: usize, n: usize, group: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(n % group, 0);
    const FP8_MAX_INV: f32 = 1.0 / 448.0;
    let ng = n / group;
    let mut codes = vec![0.0f32; rows * n];
    let mut scales = vec![0.0f32; rows * ng];
    for r in 0..rows {
        for g in 0..ng {
            let base = r * n + g * group;
            let mut amax = 0.0f32;
            for j in 0..group {
                amax = amax.max(x[base + j].abs());
            }
            amax = amax.max(1e-4);
            let s = fast_round_scale(amax, FP8_MAX_INV);
            scales[r * ng + g] = s;
            for j in 0..group {
                let q = f32_to_e4m3_rne((x[base + j] / s).clamp(-448.0, 448.0));
                codes[base + j] = e4m3_to_f32(q);
            }
        }
    }
    (codes, scales)
}

// ---------------------------------------------------------------------------
// Parallel row helper (deterministic: rows are independent)
// ---------------------------------------------------------------------------

fn n_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Apply `f` to each disjoint `chunk`-sized row of `out` in parallel.
/// `f(row_index, &mut [T])` receives row `i`'s mutable slice. Static assignment
/// — deterministic (rows are independent).
pub fn par_chunks<T: Send>(out: &mut [T], chunk: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let n = out.len() / chunk;
    let nt = n_threads().min(n.max(1));
    if nt <= 1 {
        for (i, c) in out.chunks_mut(chunk).enumerate() {
            f(i, c);
        }
        return;
    }
    std::thread::scope(|sc| {
        let per = n.div_ceil(nt);
        let mut rest: &mut [T] = out;
        let mut start = 0usize;
        for _ in 0..nt {
            let take_chunks = per.min(rest.len() / chunk);
            if take_chunks == 0 {
                break;
            }
            let (head, tail) = rest.split_at_mut(take_chunks * chunk);
            rest = tail;
            let f = &f;
            let s0 = start;
            sc.spawn(move || {
                for (j, c) in head.chunks_mut(chunk).enumerate() {
                    f(s0 + j, c);
                }
            });
            start += take_chunks;
        }
    });
}

/// Apply `f` to each disjoint row-chunk of `out` in parallel.
/// `f(row_index, &mut T)`; T: Send. Chunk assignment is static — deterministic.
pub fn par_rows<T: Send>(out: &mut [T], f: impl Fn(usize, &mut T) + Sync) {
    let n = out.len();
    let nt = n_threads().min(n.max(1));
    if nt <= 1 {
        for (i, slot) in out.iter_mut().enumerate() {
            f(i, slot);
        }
        return;
    }
    let chunk = n.div_ceil(nt);
    std::thread::scope(|sc| {
        let mut rest: &mut [T] = out;
        for t in 0..nt {
            let take = chunk.min(rest.len());
            let (head, tail) = rest.split_at_mut(take);
            rest = tail;
            if head.is_empty() {
                break;
            }
            let f = &f;
            let start = t * chunk;
            sc.spawn(move || {
                for (i, slot) in head.iter_mut().enumerate() {
                    f(start + i, slot);
                }
            });
        }
    });
}

/// 8-accumulator f32 dot product (lets LLVM vectorize; summation order is our
/// own — GEMM-order is tolerance-level vs the oracle by design, see module docs).
/// Used where the products are EXACT in f32 (quant-GEMM code·pow2-dequant and
/// sparse-attn bf16·bf16 dots) — there mul+add equals FMA and this order was
/// measured bit-identical to torch's block matmul on the real shapes.
#[inline]
pub fn dot8(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut acc = [0.0f32; 8];
    let chunks = n / 8;
    for c in 0..chunks {
        let i = c * 8;
        for l in 0..8 {
            acc[l] += a[i + l] * b[i + l];
        }
    }
    for l in chunks * 8..n {
        acc[l % 8] += a[l] * b[l];
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}

/// Pairwise-tree reduction (in place). Measured against torch's vectorized
/// reduce on the real layer-0 tensors: ~1 ulp off torch's mean-of-squares,
/// where an 8-accumulator chain is ~18 ulps off. Used for the long-K f32
/// reductions (RMSNorm/hc_pre mean-of-squares, f32 GEMV dots) — the paths whose
/// noise snaps act_quant codes at midpoints (the G1 kv_cache diagnostic).
pub fn pairwise_sum(v: &mut [f32]) -> f32 {
    let mut n = v.len();
    if n == 0 {
        return 0.0;
    }
    while n > 1 {
        let mut w = 0;
        let mut r = 0;
        while r + 1 < n {
            v[w] = v[r] + v[r + 1];
            w += 1;
            r += 2;
        }
        if r < n {
            v[w] = v[r];
            w += 1;
        }
        n = w;
    }
    v[0]
}

/// Per-element square (f32, rounded like torch's materialized `.square()`)
/// then pairwise-tree sum.
pub fn sumsq_tree(x: &[f32]) -> f32 {
    let mut buf: Vec<f32> = x.iter().map(|v| v * v).collect();
    pairwise_sum(&mut buf)
}

/// f32 dot for long-K f32 GEMV/GEMM paths (hc_fn, router, compressor, LM head):
/// per-element product (f32-rounded) then pairwise-tree sum.
pub fn dot_tree(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut buf: Vec<f32> = a.iter().zip(b).map(|(x, y)| x * y).collect();
    pairwise_sum(&mut buf)
}

// ---------------------------------------------------------------------------
// GEMMs
// ---------------------------------------------------------------------------

/// §C.3/§C.4 blocked quant GEMM, output bf16-valued f32 [t, n].
///
/// `x` [t,k] bf16-valued f32 activations; `w` [n,k] f32 **exactly dequantized**
/// weights (fp8: e4m3·ue8m0; fp4: e2m1·e8m0 — both exact in f32, so the weight
/// block scale sb is a power of two already folded into `w`). The activation is
/// quantized per 128-group to FP8 codes + ue8m0 scales (§C.1 non-inplace).
/// Structure reproduces the dual-accumulator kernels: per inner K-block
/// (`inner_block` = 128 for fp8 weights / 32 for fp4), a raw f32 dot of
/// code·w_deq, then `acc += raw · sa[m, kb·B/128]`.
///
/// Exactness note: code·w_deq products scale the reference's code·code products
/// by the pow2 sb, and pow2 scaling commutes exactly with f32 addition, so each
/// block dot equals sb·(reference raw block dot) up to summation order (which
/// is tolerance-level anyway). `acc` and the final bf16 cast mirror the kernels.
pub fn quant_gemm(x: &[f32], t: usize, k: usize, w: &[f32], n: usize, inner_block: usize) -> Vec<f32> {
    assert!(k % 128 == 0 && (inner_block == 128 || inner_block == 32));
    assert_eq!(x.len(), t * k);
    assert_eq!(w.len(), n * k);
    let nkb = k / inner_block;
    let mut out = vec![0.0f32; t * n];
    par_chunks(&mut out, n, |m, row_out| {
        let (codes, sa) = act_quant_codes(&x[m * k..(m + 1) * k], 1, k, 128);
        for n0 in 0..n {
            let wrow = &w[n0 * k..(n0 + 1) * k];
            let mut acc = 0.0f32;
            for kb in 0..nkb {
                let base = kb * inner_block;
                let raw = dot8(&codes[base..base + inner_block], &wrow[base..base + inner_block]);
                acc += raw * sa[kb * inner_block / 128]; // act scale: per-128 on K (§C.4)
            }
            row_out[n0] = acc;
        }
    });
    round_bf16(&mut out);
    out
}

/// Plain f32 GEMM, out [t,n] f32 (router / mHC GEMV / compressor / LM head).
pub fn gemm_f32(x: &[f32], t: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(x.len(), t * k);
    assert_eq!(w.len(), n * k);
    let mut out = vec![0.0f32; t * n];
    par_chunks(&mut out, n, |i, row_out| {
        for n0 in 0..n {
            row_out[n0] = dot_tree(&x[i * k..(i + 1) * k], &w[n0 * k..(n0 + 1) * k]);
        }
    });
    out
}

/// bf16-in/out GEMM with f32 accumulate (weights_proj, wo_a einsum, markov_w2
/// is f32 — use gemm_f32 there). x and w hold bf16-valued f32; out bf16-rounded.
pub fn gemm_bf16(x: &[f32], t: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    let mut out = gemm_f32(x, t, k, w, n);
    round_bf16(&mut out);
    out
}

