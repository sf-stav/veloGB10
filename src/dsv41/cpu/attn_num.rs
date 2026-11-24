//! DSV4.1 attention oracle (leg OA): number formats, quantisers, linear-layer emulations and a
//! small thread pool. Everything here reproduces DeepSeek's `inference/kernel.py` / `model.py`
//! numerics op by op (REF map §2.2-2.8), in f32 with f64 where the reference sums exactly.
//!
//! Values travel as `f32` holding the exact value of the reference tensor's dtype: a "bf16 value"
//! is an f32 whose low 16 bits are zero (`bf16()` rounds to it), an "e4m3 value" is an f32 on the
//! E4M3FN grid. Conventions mirrored from the reference:
//! * `act_quant` (kernel.py 22-124): per group of 32 along the last dim, amax floored at
//!   fp32(1e-4), scale = the smallest power of two >= amax * fp32(1/448) taken from the exponent
//!   bits, fp32 divide, clamp +-448, round-to-nearest-even to E4M3.
//! * `fp4_act_quant` (kernel.py 128-204): E8M0 variant (indexer, groups of 32): scale
//!   pow2ceil(amax * fp32(1/6)), amax floored at 6 * 2^-126; E4M3 variant (compressed KV, groups of
//!   16): scale = fp32(e4m3_satfinite(amax / 6)), amax floored at 6 * 2^-9. Element = RNE onto
//!   the E2M1 grid of clamp(x / s, -6, 6) (REFb proved the GB10 cast is RNE), times s, to bf16.
//! * `fp8_gemm` (kernel.py 208-307): every 32-wide K block is one tensor-core product (the exact
//!   sum, formed in f64, rounded toward zero to fp32 as GB10's MMA does), scaled by the two
//!   power-of-two scales and accumulated in fp32 in K order, one bf16 RNE.
//! * bf16 x bf16 linears and einsums: fp32 accumulation, one bf16 rounding (REFb's `fp32acc`
//!   fixture semantics; the sum is formed in f64 and rounded once to f32).

use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

// ------------------------------------------------------------------------------------------ formats

/// f32 -> bf16 bits, round to nearest even (torch's `.to(bfloat16)`); NaN stays NaN.
#[inline]
pub fn bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    if x.is_nan() {
        return ((u >> 16) | 0x40) as u16;
    }
    let lsb = (u >> 16) & 1;
    (u.wrapping_add(0x7FFF + lsb) >> 16) as u16
}

/// Round to the nearest bf16 value (RNE), returned as f32.
#[inline]
pub fn bf16(x: f32) -> f32 { f32::from_bits((bf16_bits(x) as u32) << 16) }

/// A bf16 bit pattern as f32.
#[inline]
pub fn bf16_val(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }

pub use crate::dsv41::gpu::abi::{e2m1 as e2m1_decode, e4m3 as e4m3_decode, e8m0 as e8m0_decode};

/// Round onto the E4M3FN grid, nearest even (torch's `.to(float8_e4m3fn)` for |v| <= 448, which
/// every caller guarantees by clamping first). Subnormals: multiples of 2^-9 below 2^-6.
#[inline]
pub fn e4m3_round(v: f32) -> f32 {
    if v == 0.0 || !v.is_finite() {
        return v;
    }
    let a = v.abs();
    let e = (((a.to_bits() >> 23) & 0xFF) as i32 - 127).max(-6);
    let step = f32::from_bits(((e - 3 + 127) as u32) << 23); // 2^(e-3), e >= -6 so e-3 >= -9: normal f32
    let q = (a / step).round_ties_even() * step;
    q.copysign(v)
}

/// RNE onto the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6} of a value with |v| <= 6 (REFb's
/// `round_e2m1`, proven bitwise against the GB10 TileLang cast): ties to the even code,
/// .25->0 .75->1 1.25->1 1.75->2 2.5->2 3.5->4 5->4. Keeps the sign (copysign), so -0.1 -> -0.
#[inline]
pub fn e2m1_round(v: f32) -> f32 {
    let a = v.abs();
    let q: f32 = if a <= 0.25 {
        0.0
    } else if a < 0.75 {
        0.5
    } else if a <= 1.25 {
        1.0
    } else if a < 1.75 {
        1.5
    } else if a <= 2.5 {
        2.0
    } else if a < 3.5 {
        3.0
    } else if a <= 5.0 {
        4.0
    } else {
        6.0
    };
    q.copysign(v)
}

/// kernel.py `fast_round_scale`: 2^(exponent field - 127 + (mantissa != 0)) of a positive normal f32,
/// i.e. the smallest power of two >= v, read from the bits (exact at powers of two).
#[inline]
pub fn pow2_ceil(v: f32) -> f32 {
    let b = v.to_bits();
    let e = ((b >> 23) & 0xFF) as i32 - 127 + ((b & 0x7F_FFFF) != 0) as i32;
    f32::from_bits(((e + 127) as u32) << 23)
}

/// f64 -> f32 rounded toward zero. GB10's FP8 / FP4 block-scaled MMA (TileLang's `fp8_gemm` /
/// `fp4_gemm`, `kind::f8f6f4`) returns a 32-wide K block's dot as RZ(exact sum) when the exact sum
/// needs more than 24 bits (leg KN2, 512 K dots, every one equal); an RNE oracle differs on ~1 in 4
/// of those.
#[inline]
pub fn f32_rz(x: f64) -> f32 {
    let r = x as f32;
    if (r as f64).abs() > x.abs() {
        f32::from_bits(r.to_bits() - 1)
    } else {
        r
    }
}

/// The fp32 immediates of the reference kernels (a Python float rounded once to fp32).
#[inline]
pub fn f32_of(v: f64) -> f32 { v as f32 }

pub const FP8_MAX: f32 = 448.0;
pub const FP4_MAX: f32 = 6.0;

// ------------------------------------------------------------------------------------------ quantisers

/// `act_quant` of one group (bf16 values): writes the E4M3 values into `q`, returns the scale.
#[inline]
pub fn act_quant_group(x: &[f32], q: &mut [f32]) -> f32 {
    let mut amax = 0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }
    let amax = amax.max(f32_of(1e-4));
    let s = pow2_ceil(amax * f32_of(1.0 / 448.0));
    for (o, &v) in q.iter_mut().zip(x) {
        *o = e4m3_round((v / s).clamp(-FP8_MAX, FP8_MAX));
    }
    s
}

/// `act_quant(x, block, "ue8m0")` non-inplace: E4M3 values into `q`, one scale per block into `s`.
pub fn act_quant(x: &[f32], block: usize, q: &mut [f32], s: &mut [f32]) {
    assert!(x.len() % block == 0 && q.len() == x.len() && s.len() == x.len() / block);
    for (g, sc) in s.iter_mut().enumerate() {
        let r = g * block..(g + 1) * block;
        *sc = act_quant_group(&x[r.clone()], &mut q[r]);
    }
}

/// `act_quant(x, block, "ue8m0", inplace=True)`: x <- bf16(e4m3(x / s) * s) per block.
pub fn act_quant_inplace(x: &mut [f32], block: usize) {
    assert!(x.len() % block == 0);
    let mut q = vec![0f32; block];
    for g in x.chunks_mut(block) {
        let s = act_quant_group(g, &mut q);
        for (v, &qq) in g.iter_mut().zip(&q) {
            *v = bf16(qq * s);
        }
    }
}

/// `fp4_act_quant(x, block, inplace=True)` with E8M0 (power-of-two) scales: the indexer Q / K.
pub fn fp4_quant_e8m0_inplace(x: &mut [f32], block: usize) {
    assert!(x.len() % block == 0);
    for g in x.chunks_mut(block) {
        let mut amax = 0f32;
        for &v in g.iter() {
            amax = amax.max(v.abs());
        }
        let amax = amax.max(f32_of(6.0 * 2f64.powi(-126)));
        let s = pow2_ceil(amax * f32_of(1.0 / 6.0));
        for v in g.iter_mut() {
            *v = bf16(e2m1_round((*v / s).clamp(-FP4_MAX, FP4_MAX)) * s);
        }
    }
}

/// The E4M3-rounded scale of the compressed-KV FP4 variant: fp32(e4m3_satfinite(amax / 6)).
#[inline]
pub fn fp4_e4m3_scale(amax: f32) -> f32 {
    let amax = amax.max(f32_of(6.0 * 2f64.powi(-9)));
    if variant::is(variant::FP4_POW2_SCALE) {
        return pow2_ceil(amax * f32_of(1.0 / 6.0));
    }
    e4m3_round((amax / FP4_MAX).min(FP8_MAX))
}

/// `fp4_act_quant(x, block, True, scale_dtype=float8_e4m3fn)`: the compressed KV latent (block 16).
pub fn fp4_quant_e4m3_inplace(x: &mut [f32], block: usize) {
    assert!(x.len() % block == 0);
    for g in x.chunks_mut(block) {
        let mut amax = 0f32;
        for &v in g.iter() {
            amax = amax.max(v.abs());
        }
        let s = fp4_e4m3_scale(amax);
        for v in g.iter_mut() {
            *v = bf16(e2m1_round((*v / s).clamp(-FP4_MAX, FP4_MAX)) * s);
        }
    }
}

/// `RMSNorm.forward` (model.py 281-293) on one row of bf16 values: fp32 mean of squares (formed in
/// f64, rounded once), rsqrt(var + eps), x * r in fp32, weight * that in fp32, one bf16 rounding.
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), w.len());
    let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let var = (ss / x.len() as f64) as f32;
    let r = (1.0 / ((var + eps) as f64).sqrt()) as f32;
    for ((o, &v), &wv) in out.iter_mut().zip(x).zip(w) {
        *o = bf16(wv * (v * r));
    }
}

// ------------------------------------------------------------------------------------------ dots

/// Dot product formed in f64 (four partial sums; the inputs here are bf16 / fp8-grid values whose
/// products are exact, so the f64 sum is the exact sum to ~1e-16 relative).
#[inline]
pub fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    if variant::is(variant::F32_ACC) {
        let mut t = 0f32;
        for (x, y) in a.iter().zip(b) {
            t += x * y;
        }
        return t as f64;
    }
    let mut s = [0f64; 4];
    let n4 = a.len() / 4 * 4;
    let (a4, ar) = a.split_at(n4);
    let (b4, br) = b.split_at(n4);
    for (x, y) in a4.chunks_exact(4).zip(b4.chunks_exact(4)) {
        s[0] += x[0] as f64 * y[0] as f64;
        s[1] += x[1] as f64 * y[1] as f64;
        s[2] += x[2] as f64 * y[2] as f64;
        s[3] += x[3] as f64 * y[3] as f64;
    }
    let mut t = (s[0] + s[2]) + (s[1] + s[3]);
    for (x, y) in ar.iter().zip(br) {
        t += *x as f64 * *y as f64;
    }
    t
}

// ------------------------------------------------------------------------------------------ threads

// ------------------------------------------------------------------------------------------ variants

/// Oracle variants for the tolerance study (report §2): the stated tolerances must ACCEPT a valid
/// alternative summation order and REJECT each semantic mutation of a REF-map subtle point. Set
/// only by the `#[ignore]` study test; the probe and every other caller run `REFERENCE`.
pub mod variant {
    use std::sync::atomic::{AtomicU32, Ordering};
    static V: AtomicU32 = AtomicU32::new(0);
    pub const REFERENCE: u32 = 0;
    /// valid: every dot product summed sequentially in fp32 (a naive SGEMM order)
    pub const F32_ACC: u32 = 1;
    /// mutation: P kept fp32 for P.V (REF map §4 item 8)
    pub const NO_BF16_P: u32 = 2;
    /// mutation: the attention sink left out of the denominator (item 9)
    pub const NO_SINK: u32 = 3;
    /// mutation: the output's inverse RoPE skipped (item 17)
    pub const NO_INV_ROPE: u32 = 4;
    /// mutation: no act_quant on WO-A's input (item 6)
    pub const NO_WOA_QUANT: u32 = 5;
    /// mutation: no YaRN on the compressed layers (item 17)
    pub const NO_YARN: u32 = 6;
    /// mutation: compressed-KV FP4 with power-of-two scales instead of E4M3 (item 5)
    pub const FP4_POW2_SCALE: u32 = 7;
    /// mutation: fp32 index scores instead of the bf16 chain (item 7)
    pub const FP32_INDEX_SCORE: u32 = 8;
    pub const ALL: [(u32, &str); 9] = [
        (REFERENCE, "reference recipe"),
        (F32_ACC, "valid: fp32 sequential sums"),
        (NO_BF16_P, "mutation: fp32 P in P.V"),
        (NO_SINK, "mutation: no attention sink"),
        (NO_INV_ROPE, "mutation: no inverse RoPE on o"),
        (NO_WOA_QUANT, "mutation: no act_quant on WO-A input"),
        (NO_YARN, "mutation: no YaRN"),
        (FP4_POW2_SCALE, "mutation: pow2 FP4 scale for compressed KV"),
        (FP32_INDEX_SCORE, "mutation: fp32 index scores"),
    ];
    pub fn set(v: u32) { V.store(v, Ordering::Relaxed); }
    #[inline]
    pub fn is(v: u32) -> bool { V.load(Ordering::Relaxed) == v }
}

static THREADS: AtomicUsize = AtomicUsize::new(0);

/// Worker threads for the oracle (default: half the cores, at most 16: `.11` is shared).
pub fn threads() -> usize {
    match THREADS.load(Ordering::Relaxed) {
        0 => std::thread::available_parallelism().map(|n| (n.get() / 2).clamp(1, 16)).unwrap_or(4),
        n => n,
    }
}

/// Override the worker count (0 = default).
pub fn set_threads(n: usize) { THREADS.store(n, Ordering::Relaxed); }

/// Run `f` over `0..n` in chunks of `chunk` on the worker threads; the outputs are concatenated in
/// index order (each call of `f` returns the results of its range, in order). Deterministic: the
/// split never changes a result, only who computes it.
pub fn par_map<T: Send>(n: usize, chunk: usize, f: impl Fn(Range<usize>) -> Vec<T> + Sync) -> Vec<T> {
    let chunk = chunk.max(1);
    let nchunks = n.div_ceil(chunk);
    let nt = threads().min(nchunks).max(1);
    if nt <= 1 {
        let mut out = Vec::with_capacity(n);
        for c in 0..nchunks {
            out.extend(f(c * chunk..((c + 1) * chunk).min(n)));
        }
        return out;
    }
    let next = AtomicUsize::new(0);
    let mut parts: Vec<(usize, Vec<T>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..nt)
            .map(|_| {
                sc.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let c = next.fetch_add(1, Ordering::Relaxed);
                        if c >= nchunks {
                            break;
                        }
                        mine.push((c, f(c * chunk..((c + 1) * chunk).min(n))));
                    }
                    mine
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().expect("oracle worker panicked")).collect()
    });
    parts.sort_by_key(|p| p.0);
    let mut out = Vec::with_capacity(n);
    for (_, v) in parts {
        out.extend(v);
    }
    out
}

/// Transpose a column-major block result `[n][rows]` into row-major `[rows][n]`.
fn to_rows(colmajor: &[f32], n: usize, rows: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * n];
    for j in 0..n {
        for r in 0..rows {
            out[r * n + j] = colmajor[j * rows + r];
        }
    }
    out
}

// ------------------------------------------------------------------------------------------ linears

/// An FP8 linear (`Linear` with float8_e4m3fn weight and 32 x 32 E8M0 block scales).
#[derive(Clone)]
pub struct Fp8Lin {
    pub n: usize,
    pub k: usize,
    /// E4M3 values, unscaled, [n][k]
    pub w: Vec<f32>,
    /// block scales [ceil(n/32)][k/32]
    pub s: Vec<f32>,
}

impl Fp8Lin {
    pub const BLOCK: usize = 32;

    /// Build from the raw checkpoint bytes (e4m3 [n, k], e8m0 [ceil(n/32), ceil(k/32)]).
    pub fn from_raw(n: usize, k: usize, w: &[u8], s: &[u8]) -> Self {
        assert_eq!(w.len(), n * k);
        assert!(k % Self::BLOCK == 0);
        assert_eq!(s.len(), n.div_ceil(Self::BLOCK) * (k / Self::BLOCK));
        Fp8Lin { n, k, w: w.iter().map(|&b| e4m3_decode(b)).collect(), s: s.iter().map(|&b| e8m0_decode(b)).collect() }
    }

    /// Exact dequantised weight row `n` (e4m3 x block scale; exact in f32).
    pub fn row_dequant(&self, n: usize) -> Vec<f32> {
        let kb = self.k / Self::BLOCK;
        (0..self.k).map(|k| self.w[n * self.k + k] * self.s[(n / Self::BLOCK) * kb + k / Self::BLOCK]).collect()
    }

    /// `linear(x, w)` for `rows` rows of bf16 values: act_quant (per 32) + fp8_gemm, bf16 out [rows][n].
    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let acc = self.forward_f32(x, rows, 0..self.k / Self::BLOCK);
        acc.into_iter().map(bf16).collect()
    }

    /// The fp32 accumulator over the K blocks `kblocks` (before the bf16 rounding), [rows][n]. The
    /// full range is the reference; a sub-range is one rank's partial when the input dim is split
    /// (RowParallelLinear: `wo_b` by o_group under TP).
    pub fn forward_f32(&self, x: &[f32], rows: usize, kblocks: Range<usize>) -> Vec<f32> {
        let (n, k, b) = (self.n, self.k, Self::BLOCK);
        assert_eq!(x.len(), rows * k);
        let kb = k / b;
        let mut qa = vec![0f32; rows * k];
        let mut sa = vec![0f32; rows * kb];
        for r in 0..rows {
            act_quant(&x[r * k..(r + 1) * k], b, &mut qa[r * k..(r + 1) * k], &mut sa[r * kb..(r + 1) * kb]);
        }
        let cm = par_map(n, b, |range| {
            let mut out = Vec::with_capacity(range.len() * rows);
            for j in range {
                let wrow = &self.w[j * k..(j + 1) * k];
                let srow = &self.s[(j / b) * kb..(j / b + 1) * kb];
                for r in 0..rows {
                    let arow = &qa[r * k..(r + 1) * k];
                    let mut acc = 0f32;
                    for kk in kblocks.clone() {
                        let d = f32_rz(dot_f64(&arow[kk * b..(kk + 1) * b], &wrow[kk * b..(kk + 1) * b]));
                        acc += d * sa[r * kb + kk] * srow[kk];
                    }
                    out.push(acc);
                }
            }
            out
        });
        to_rows(&cm, n, rows)
    }
}

/// A dense linear with bf16 (or fp32) weights held as f32 values, [n][k].
#[derive(Clone)]
pub struct DenseLin {
    pub n: usize,
    pub k: usize,
    pub w: Vec<f32>,
}

impl DenseLin {
    /// fp32 accumulation, rows x [n] f32 (no output rounding: the fp32 linears of the reference,
    /// and the raw sum the bf16 variant rounds).
    pub fn forward_f32(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let (n, k) = (self.n, self.k);
        assert_eq!(x.len(), rows * k);
        let cm = par_map(n, 16, |range| {
            let mut out = Vec::with_capacity(range.len() * rows);
            for j in range {
                let wrow = &self.w[j * k..(j + 1) * k];
                for r in 0..rows {
                    out.push(dot_f64(&x[r * k..(r + 1) * k], wrow) as f32);
                }
            }
            out
        });
        to_rows(&cm, n, rows)
    }

    /// bf16 x bf16 linear with fp32 accumulation and one bf16 rounding (REFb `fp32acc`).
    pub fn forward_bf16(&self, x: &[f32], rows: usize) -> Vec<f32> {
        self.forward_f32(x, rows).into_iter().map(bf16).collect()
    }
}
