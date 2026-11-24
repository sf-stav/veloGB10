//! DSV4.1 leg OD: the numerics of the DSpark CPU oracle (`cpu/dspark.rs`).
//!
//! Every op reproduces DeepSeek's reference (`inference/model.py` + `kernel.py`, mapped in
//! `PLAN/DSV41_REF_MAP.md` §2 and pinned bitwise against the real TileLang kernels on GB10 by REFb's
//! `tools/dsv41_ref/kernel_emu.py`) at the points where the reference ROUNDS: every bf16 tensor
//! boundary, the e4m3 activation quantisation (`act_quant`), the fp32 block accumulators of the fp8 /
//! fp4 GEMMs. Sums whose order the reference leaves to a vendor library (cuBLAS SGEMM, torch
//! reductions) are taken in f64 and rounded once (the correctly rounded value). Those can differ from
//! the GPU by one fp32 ulp, which only becomes visible when a later bf16 / e4m3 rounding sits on a
//! boundary. Transcendentals (exp, sigmoid, softplus, sin / cos, pow) are evaluated in f64 and rounded
//! to f32 (CUDA's `expf` / `sinf` are within 2 ulp of that).
//!
//! Reference-grade code: plain loops, f64 accumulation, threads over output columns. The GPU legs
//! (KN1 / KN2 / KA / KM / DS) must not copy its structure, only its rounding points.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

// ------------------------------------------------------------------------------------------------ formats

/// Round an f32 to the nearest bf16 (ties to even); the result is returned as an f32.
#[inline]
pub fn bf16r(x: f32) -> f32 {
    if x.is_nan() { return f32::NAN; }
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    f32::from_bits(b.wrapping_add(0x7FFF + lsb) & 0xFFFF_0000)
}

/// A bf16 bit pattern as f32 (exact).
#[inline]
pub fn bf16_bits(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }

/// e4m3fn byte -> value (S.EEEE.MMM, bias 7, no infinity, S.1111.111 = NaN, subnormals m * 2^-9).
pub fn e4m3_decode(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as f32;
    if e == 0xF && b & 7 == 7 { return f32::NAN; }
    let v = if e == 0 { m * (2.0f32).powi(-9) } else { (1.0 + m / 8.0) * (2.0f32).powi(e - 7) };
    sign * v
}

/// The 256 e4m3fn values (lookup table for the weight decode).
pub fn e4m3_lut() -> &'static [f32; 256] {
    static T: OnceLock<[f32; 256]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0f32; 256];
        for (i, v) in t.iter_mut().enumerate() { *v = e4m3_decode(i as u8); }
        t
    })
}

/// e8m0 (ue8m0) byte -> 2^(byte - 127); 255 = NaN.
#[inline]
pub fn e8m0_decode(b: u8) -> f32 {
    match b {
        255 => f32::NAN,
        0 => f32::from_bits(0x0040_0000), // 2^-127 (an f32 subnormal)
        _ => f32::from_bits((b as u32) << 23),
    }
}

/// e2m1 (fp4) code -> value; code 8 is "-0" (read as -0.0, i.e. zero). Low nibble = earlier element.
pub const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// Round-to-nearest-even cast of an f32 already clamped to [-448, 448] onto the e4m3fn grid
/// (torch `.to(float8_e4m3fn)`, the TileLang cast; subnormal quantum 2^-9). Returns the value.
pub fn e4m3_rne(x: f32) -> f32 {
    if x == 0.0 || x.is_nan() { return x; }
    let a = x.abs();
    let quantum = if a < (2.0f32).powi(-6) {
        (2.0f64).powi(-9)
    } else {
        let e = ((a.to_bits() >> 23) & 0xFF) as i32 - 127;
        (2.0f64).powi(e - 3)
    };
    let n = (a as f64 / quantum).round_ties_even();
    let v = ((n * quantum) as f32).min(448.0);
    v.copysign(x)
}

/// kernel.py `fast_round_scale`: the smallest power of two >= v, read from the exponent bits of the
/// positive normal f32 `v` (one more when any mantissa bit is set).
#[inline]
pub fn pow2_ceil(v: f32) -> f32 {
    let bits = v.to_bits();
    let e = ((bits >> 23) & 0xFF) as i32 - 127 + if bits & 0x7F_FFFF != 0 { 1 } else { 0 };
    f32::from_bits(((e + 127) as u32) << 23)
}

/// fp32(1/448), the TileLang literal (`act_quant`).
pub fn inv448() -> f32 { (1.0f64 / 448.0) as f32 }

// ------------------------------------------------------------------------------------------------ act_quant

/// `act_quant(x, 32, "ue8m0")` (kernel.py 22-124, REF_MAP §2.2) on ONE row: per group of `block`,
/// amax floored at fp32(1e-4), scale = pow2ceil(amax * fp32(1/448)), q = e4m3(clamp(x / s, +-448)).
/// Returns the e4m3 VALUES and the per-group scales.
pub fn act_quant(x: &[f32], block: usize) -> (Vec<f32>, Vec<f32>) {
    assert!(x.len() % block == 0, "act_quant: {} not a multiple of {block}", x.len());
    let floor = 1e-4f32;
    let mut q = vec![0f32; x.len()];
    let mut s = Vec::with_capacity(x.len() / block);
    for (g, xs) in x.chunks(block).enumerate() {
        let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs())).max(floor);
        let sc = pow2_ceil(amax * inv448());
        for (j, v) in xs.iter().enumerate() { q[g * block + j] = e4m3_rne((v / sc).clamp(-448.0, 448.0)); }
        s.push(sc);
    }
    (q, s)
}

/// `act_quant(x, 32, "ue8m0", inplace=True)`: the fused quant-dequant, `bf16(fp32(q) * s)` written back.
pub fn act_quant_inplace(x: &mut [f32], block: usize) {
    let (q, s) = act_quant(x, block);
    for (i, v) in x.iter_mut().enumerate() { *v = bf16r(q[i] * s[i / block]); }
}

// ------------------------------------------------------------------------------------------------ threads

static THREADS: AtomicUsize = AtomicUsize::new(0);

/// Worker threads for the column-parallel loops (0 = min(8, available)). Results do not depend on it:
/// every output element is computed by one thread in a fixed order.
pub fn set_threads(n: usize) { THREADS.store(n, Ordering::Relaxed); }

fn threads() -> usize {
    match THREADS.load(Ordering::Relaxed) {
        0 => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(8),
        n => n,
    }
}

/// Compute `n` output columns for `rows` input rows: `f(col, out)` fills `out[row]` for column `col`.
/// Returns `[rows][n]`. Columns are split over threads in contiguous chunks.
pub fn par_cols<F: Fn(usize, &mut [f32]) + Sync>(n: usize, rows: usize, f: F) -> Vec<Vec<f32>> {
    let mut buf = vec![0f32; n * rows];
    if n == 0 || rows == 0 { return vec![Vec::new(); rows]; }
    let t = threads().max(1).min(n);
    let chunk = n.div_ceil(t);
    std::thread::scope(|sc| {
        for (ci, part) in buf.chunks_mut(chunk * rows).enumerate() {
            let f = &f;
            sc.spawn(move || {
                for (j, out) in part.chunks_mut(rows).enumerate() { f(ci * chunk + j, out); }
            });
        }
    });
    (0..rows).map(|r| (0..n).map(|c| buf[c * rows + r]).collect()).collect()
}

/// Map `n` independent jobs over threads (fixed order inside each job), results in job order.
pub fn par_map<T: Send, F: Fn(usize) -> T + Sync>(n: usize, f: F) -> Vec<T> {
    let t = threads().max(1).min(n.max(1));
    let chunk = n.div_ceil(t).max(1);
    let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
    std::thread::scope(|sc| {
        for (ci, part) in out.chunks_mut(chunk).enumerate() {
            let f = &f;
            sc.spawn(move || {
                for (j, o) in part.iter_mut().enumerate() { *o = Some(f(ci * chunk + j)); }
            });
        }
    });
    out.into_iter().map(|o| o.expect("par_map job")).collect()
}

// ------------------------------------------------------------------------------------------------ linears

/// A quantised linear weight as the native checkpoint stores it.
pub enum QLin {
    /// e4m3 `[n, k]` with e8m0 scales per 32 x 32 block `[ceil(n/32), k/32]` (model.py `Linear`, fp8).
    Fp8 { n: usize, k: usize, w: Vec<u8>, s: Vec<u8> },
    /// e2m1 packed `[n, k/2]` (low nibble = earlier k) with e8m0 scales `[n, k/32]` (routed experts).
    Fp4 { n: usize, k: usize, w: Vec<u8>, s: Vec<u8> },
}

impl QLin {
    pub fn n(&self) -> usize { match self { QLin::Fp8 { n, .. } | QLin::Fp4 { n, .. } => *n } }
    pub fn k(&self) -> usize { match self { QLin::Fp8 { k, .. } | QLin::Fp4 { k, .. } => *k } }
}

/// model.py `linear()` for an fp8 / fp4 weight: `act_quant(x, 32)` per row, then `fp8_gemm` /
/// `fp4_gemm` (kernel.py 208-307 / 478-591): for each 32-wide K block the exact block dot (an MMA of
/// e4m3 x e4m3 / e2m1 values, summed exactly in f64 here), rounded to f32, times the two power-of-two
/// scales, accumulated sequentially in f32 over the blocks; one RNE to bf16 at the end.
/// `rows` hold bf16 values; the outputs are bf16 values `[rows][n]`.
pub fn qlinear(rows: &[Vec<f32>], w: &QLin) -> Vec<Vec<f32>> {
    let k = w.k();
    let kb = k / 32;
    let qa: Vec<(Vec<f32>, Vec<f32>)> = rows.iter().map(|r| { assert_eq!(r.len(), k, "qlinear input width"); act_quant(r, 32) }).collect();
    let lut = e4m3_lut();
    match w {
        QLin::Fp8 { n, w, s, .. } => par_cols(*n, rows.len(), |c, out| {
            let wr = &w[c * k..(c + 1) * k];
            let sr = &s[(c / 32) * kb..(c / 32 + 1) * kb];
            for (ri, (q, sa)) in qa.iter().enumerate() {
                let mut acc = 0f32;
                for b in 0..kb {
                    let mut d = 0f64;
                    for j in b * 32..(b + 1) * 32 { d += q[j] as f64 * lut[wr[j] as usize] as f64; }
                    acc += (d as f32) * sa[b] * e8m0_decode(sr[b]);
                }
                out[ri] = bf16r(acc);
            }
        }),
        QLin::Fp4 { n, w, s, .. } => par_cols(*n, rows.len(), |c, out| {
            let wr = &w[c * k / 2..(c + 1) * k / 2];
            let sr = &s[c * kb..(c + 1) * kb];
            for (ri, (q, sa)) in qa.iter().enumerate() {
                let mut acc = 0f32;
                for b in 0..kb {
                    let mut d = 0f64;
                    for j in 0..16 {
                        let byte = wr[b * 16 + j];
                        d += q[b * 32 + 2 * j] as f64 * E2M1[(byte & 0xF) as usize] as f64;
                        d += q[b * 32 + 2 * j + 1] as f64 * E2M1[(byte >> 4) as usize] as f64;
                    }
                    acc += (d as f32) * sa[b] * e8m0_decode(sr[b]);
                }
                out[ri] = bf16r(acc);
            }
        }),
    }
}

/// A plain matrix `[n, k]` stored as bf16 bits (bf16 checkpoint tensors; fp32 parameters widened
/// from bf16 are exact in this form too).
pub struct Bf16Mat { pub n: usize, pub k: usize, pub w: Vec<u16> }

/// An fp32 matrix `[n, k]` (parameters the checkpoint stores in F32, e.g. `hc_*_fn`).
pub struct F32Mat { pub n: usize, pub k: usize, pub w: Vec<f32> }

/// `F.linear(x.float(), W.float())` with an fp32 result (cuBLAS SGEMM, TF32 off): f64 sums, one
/// rounding to f32. Used for the gate, the heads, the confidence head (bf16 weights widened).
pub fn linear_f32_bf16w(rows: &[Vec<f32>], w: &Bf16Mat) -> Vec<Vec<f32>> {
    let k = w.k;
    par_cols(w.n, rows.len(), |c, out| {
        let wr = &w.w[c * k..(c + 1) * k];
        for (ri, x) in rows.iter().enumerate() {
            let mut d = 0f64;
            for j in 0..k { d += x[j] as f64 * bf16_bits(wr[j]) as f64; }
            out[ri] = d as f32;
        }
    })
}

/// The same with fp32 weights (`hc_*_fn`).
pub fn linear_f32(rows: &[Vec<f32>], w: &F32Mat) -> Vec<Vec<f32>> {
    let k = w.k;
    par_cols(w.n, rows.len(), |c, out| {
        let wr = &w.w[c * k..(c + 1) * k];
        for (ri, x) in rows.iter().enumerate() {
            let mut d = 0f64;
            for j in 0..k { d += x[j] as f64 * wr[j] as f64; }
            out[ri] = d as f32;
        }
    })
}

/// A bf16 x bf16 GEMM with fp32 accumulation and one bf16 rounding (the `wo_a` einsum, model.py 789:
/// torch's bf16 einsum; its cuBLAS summation order is not reproduced, see the module doc).
pub fn linear_bf16(rows: &[Vec<f32>], w: &Bf16Mat) -> Vec<Vec<f32>> {
    linear_f32_bf16w(rows, w).into_iter().map(|r| r.into_iter().map(bf16r).collect()).collect()
}

// ------------------------------------------------------------------------------------------------ elementwise

#[inline] pub fn exp32(x: f32) -> f32 { (x as f64).exp() as f32 }
#[inline] pub fn sigmoid32(x: f32) -> f32 { (1.0 / (1.0 + (-(x as f64)).exp())) as f32 }
/// torch `F.silu` in fp32.
#[inline] pub fn silu32(x: f32) -> f32 { ((x as f64) / (1.0 + (-(x as f64)).exp())) as f32 }
/// torch `F.softplus` (beta 1, threshold 20) in fp32.
#[inline] pub fn softplus32(x: f32) -> f32 { if x > 20.0 { x } else { (x as f64).exp().ln_1p() as f32 } }

/// The f32 mean of the squares of bf16-valued `x` (each square is exact in fp32; f64 sum, one rounding).
pub fn mean_sq(x: &[f32]) -> f32 {
    let s: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    (s / x.len() as f64) as f32
}

/// `torch.rsqrt(v)` in fp32.
#[inline] pub fn rsqrt32(v: f32) -> f32 { (1.0 / (v as f64).sqrt()) as f32 }

/// model.py `RMSNorm` (281-293): `bf16( w * (x * rsqrt(mean(x^2) + eps)) )`, w a bf16 parameter.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), w.len(), "rms_norm width");
    let r = rsqrt32(mean_sq(x) + eps);
    x.iter().zip(w).map(|(v, g)| bf16r(g * (v * r))).collect()
}

// ------------------------------------------------------------------------------------------------ RoPE

/// `precompute_freqs_cis` without YaRN (layers 0, 1 and the DSpark stages; model.py 368-389):
/// freq_i = 1 / base^(2i / dim) in fp32, the angle at position p is fp32(p) * freq_i rounded once.
pub struct Rope { pub freqs: Vec<f32> }

impl Rope {
    pub fn plain(dim: usize, base: f64) -> Self {
        let freqs = (0..dim / 2).map(|i| {
            let e = (2 * i) as f32 / dim as f32; // fp32 arange / dim: exact
            let p = (base.powf(e as f64)) as f32;
            1.0f32 / p
        }).collect();
        Rope { freqs }
    }

    /// (cos, sin) at position `pos` for pair `i`.
    pub fn cs(&self, pos: usize, i: usize) -> (f32, f32) {
        let ang = (pos as f32) * self.freqs[i];
        let a = ang as f64;
        (a.cos() as f32, a.sin() as f32)
    }

    /// `apply_rotary_emb` (392-406) on the LAST `2 * freqs.len()` dims of `x` (adjacent pairs as complex
    /// numbers, fp32 product, written back into the bf16 tensor). `inverse` conjugates the rotation.
    pub fn apply(&self, x: &mut [f32], pos: usize, inverse: bool) {
        let rd = 2 * self.freqs.len();
        let off = x.len() - rd;
        for i in 0..self.freqs.len() {
            let (c, s0) = self.cs(pos, i);
            let s = if inverse { -s0 } else { s0 };
            let (a, b) = (x[off + 2 * i] as f64, x[off + 2 * i + 1] as f64);
            let re = (a * c as f64 - b * s as f64) as f32;
            let im = (a * s as f64 + b * c as f64) as f32;
            x[off + 2 * i] = bf16r(re);
            x[off + 2 * i + 1] = bf16r(im);
        }
    }
}

// ------------------------------------------------------------------------------------------------ sparse attention

/// kernel.py `sparse_attn` (311-403) as REFb's emulation runs it (REF_MAP §2.6): MQA, one kv row
/// serves as K and V; per query the index list is walked in blocks of 64 IN THE GIVEN ORDER, running
/// max initialised to fp32(-1e30), score = fp32(q . kv) * scale, p = exp(s - m), the denominator sums
/// the fp32 p (`l = l * corr + sum(p)` rounded once), the P.V product uses bf16(p), `acc = acc * corr + pv`
/// (two roundings), the sink enters the denominator only, `-1` indices are masked.
/// `q[qi]` is `[heads * d]`, `kv[r]` is `[d]` (all bf16 values); returns `[nq][heads * d]` bf16 values.
pub fn sparse_attn(q: &[Vec<f32>], heads: usize, d: usize, kv: &[Vec<f32>], idx: &[Vec<i32>], sink: &[f32], scale: f32) -> Vec<Vec<f32>> {
    const BLK: usize = 64;
    let nq = q.len();
    let res = par_map(nq * heads, |job| {
        let (qi, hh) = (job / heads, job % heads);
        let qv = &q[qi][hh * d..(hh + 1) * d];
        let list = &idx[qi];
        let nb = list.len().div_ceil(BLK);
        let mut m = -1e30f32;
        let mut l = 0f32;
        let mut acc = vec![0f32; d];
        let mut sblk = [f32::NEG_INFINITY; BLK];
        let mut pv = vec![0f64; d];
        for t in 0..nb {
            for (j, sv) in sblk.iter_mut().enumerate() {
                *sv = f32::NEG_INFINITY;
                let at = t * BLK + j;
                if at < list.len() && list[at] >= 0 {
                    let kr = &kv[list[at] as usize];
                    let mut dot = 0f64;
                    for e in 0..d { dot += qv[e] as f64 * kr[e] as f64; }
                    *sv = (dot as f32) * scale;
                }
            }
            let bmax = sblk.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let m_new = m.max(bmax);
            let corr = exp32(m - m_new);
            let mut psum = 0f64;
            pv.iter_mut().for_each(|v| *v = 0.0);
            for (j, &sv) in sblk.iter().enumerate() {
                if sv == f32::NEG_INFINITY { continue; }
                let p = exp32(sv - m_new);
                psum += p as f64;
                let pb = bf16r(p) as f64;
                if pb != 0.0 {
                    let kr = &kv[list[t * BLK + j] as usize];
                    for e in 0..d { pv[e] += pb * kr[e] as f64; }
                }
            }
            l = ((l as f64) * (corr as f64) + (psum as f32) as f64) as f32;
            for e in 0..d { acc[e] = acc[e] * corr + pv[e] as f32; }
            m = m_new;
        }
        l += exp32(sink[hh] - m);
        acc.iter().map(|a| bf16r(a / l)).collect::<Vec<f32>>()
    });
    (0..nq).map(|qi| res[qi * heads..(qi + 1) * heads].concat()).collect()
}

// ------------------------------------------------------------------------------------------------ mHC

/// `hc_split_sinkhorn` (kernel.py 407-474) for ONE token as REFb pinned it bitwise on sm_121:
/// pre = sigmoid(fma(m, s0, b)) + eps, post = 2 sigmoid(fma(m, s1, b)), comb = fma(m, s2, b); row
/// softmax (max-subtracted) + eps; column normalise; then (iters - 1) x (row, column) normalisations,
/// eps in every denominator; 4-lane sums in TileLang's xor-butterfly order (c0 + c2) + (c1 + c3).
/// Returns (pre[hc], post[hc], comb[hc][hc]) with comb[j][k] weighting residual copy j into output k.
pub fn sinkhorn(mixes: &[f32], scale: &[f32], base: &[f32], hc: usize, iters: usize, eps: f32) -> (Vec<f32>, Vec<f32>, Vec<Vec<f32>>) {
    assert_eq!(hc, 4, "the butterfly order is written for hc 4");
    assert_eq!(mixes.len(), (2 + hc) * hc);
    let fma = |a: f32, k: f32, c: f32| ((a as f64) * (k as f64) + c as f64) as f32;
    let pre: Vec<f32> = (0..hc).map(|j| sigmoid32(fma(mixes[j], scale[0], base[j])) + eps).collect();
    let post: Vec<f32> = (0..hc).map(|j| 2.0 * sigmoid32(fma(mixes[hc + j], scale[1], base[hc + j]))).collect();
    let mut c = [[0f32; 4]; 4];
    for j in 0..4 { for k in 0..4 { c[j][k] = fma(mixes[2 * hc + 4 * j + k], scale[2], base[2 * hc + 4 * j + k]); } }
    let rowsum = |c: &[[f32; 4]; 4], j: usize| (c[j][0] + c[j][2]) + (c[j][1] + c[j][3]);
    let colsum = |c: &[[f32; 4]; 4], k: usize| (c[0][k] + c[2][k]) + (c[1][k] + c[3][k]);
    for row in c.iter_mut() {
        let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        for v in row.iter_mut() { *v = exp32(*v - mx); }
    }
    let rs: Vec<f32> = (0..4).map(|j| rowsum(&c, j)).collect();
    for j in 0..4 { for k in 0..4 { c[j][k] = c[j][k] / rs[j] + eps; } }
    let cs: Vec<f32> = (0..4).map(|k| colsum(&c, k) + eps).collect();
    for j in 0..4 { for k in 0..4 { c[j][k] /= cs[k]; } }
    for _ in 1..iters {
        let rs: Vec<f32> = (0..4).map(|j| rowsum(&c, j) + eps).collect();
        for j in 0..4 { for k in 0..4 { c[j][k] /= rs[j]; } }
        let cs: Vec<f32> = (0..4).map(|k| colsum(&c, k) + eps).collect();
        for j in 0..4 { for k in 0..4 { c[j][k] /= cs[k]; } }
    }
    (pre, post, c.iter().map(|r| r.to_vec()).collect())
}

// ------------------------------------------------------------------------------------------------ selection

/// Indices of the `k` largest values, descending; ties go to the LOWER index (torch's CUDA topk on a
/// row with exact ties is implementation-defined; the oracle reports near-ties separately).
pub fn topk_desc(v: &[f32], k: usize) -> Vec<usize> {
    let mut ix: Vec<usize> = (0..v.len()).collect();
    ix.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    ix.truncate(k);
    ix
}

/// argmax with the first maximum (torch `argmax`).
pub fn argmax_first(v: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..v.len() { if v[i] > v[best] { best = i; } }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv41_od_bf16_rounding_is_rne() {
        assert_eq!(bf16r(1.0), 1.0);
        // 1 + 2^-8 is a tie between 1 and 1 + 2^-7: even mantissa (1.0) wins
        assert_eq!(bf16r(1.0 + 2f32.powi(-8)), 1.0);
        // 1 + 3 * 2^-8 is a tie between 1 + 2^-7 (odd) and 1 + 2^-6 (even)
        assert_eq!(bf16r(1.0 + 3.0 * 2f32.powi(-8)), 1.0 + 2f32.powi(-6));
        assert_eq!(bf16r(1.0 + 2f32.powi(-8) + 2f32.powi(-20)), 1.0 + 2f32.powi(-7));
        assert_eq!(bf16r(f32::INFINITY), f32::INFINITY);
        assert_eq!(bf16r(-3.0), -3.0);
    }

    #[test]
    fn dsv41_od_e4m3_decode_and_rne_cast_agree_on_the_grid() {
        let lut = e4m3_lut();
        assert_eq!(lut[0x7E], 448.0);
        assert_eq!(lut[0x01], 2f32.powi(-9));
        assert_eq!(lut[0x08], 2f32.powi(-6));
        assert!(lut[0x7F].is_nan() && lut[0xFF].is_nan());
        // every finite grid value is a fixed point of the cast
        for (i, &v) in lut.iter().enumerate() {
            if v.is_nan() { continue; }
            assert_eq!(e4m3_rne(v).to_bits() & 0x7FFF_FFFF, v.to_bits() & 0x7FFF_FFFF, "code {i:#x}");
        }
        // midpoints round to the even code
        assert_eq!(e4m3_rne(1.0625), 1.0); // between 1.0 (m=0) and 1.125 (m=1)
        assert_eq!(e4m3_rne(1.1875), 1.25); // between 1.125 (m=1) and 1.25 (m=2)
        assert_eq!(e4m3_rne(432.0), 448.0); // between 416 (m=5) and 448 (m=6): even is 448
        assert_eq!(e4m3_rne(1.5 * 2f32.powi(-9)), 2.0 * 2f32.powi(-9)); // subnormal tie -> even (2)
        assert_eq!(e4m3_rne(0.5 * 2f32.powi(-9)), 0.0);
        assert_eq!(e4m3_rne(-1.1), -1.125);
    }

    #[test]
    fn dsv41_od_pow2_ceil_and_act_quant_rules() {
        assert_eq!(pow2_ceil(1.0), 1.0);
        assert_eq!(pow2_ceil(1.0000001), 2.0);
        assert_eq!(pow2_ceil(0.75), 1.0);
        // a group at amax 448 has scale 1 exactly; values pass through the e4m3 grid
        let mut x = vec![0f32; 32];
        x[0] = 448.0;
        x[1] = 1.0625;
        let (q, s) = act_quant(&x, 32);
        assert_eq!(s, vec![pow2_ceil(448.0 * inv448())]);
        assert_eq!(q[0] * s[0], 448.0);
        // the 1e-4 floor: an all-zero group still gets a nonzero power-of-two scale
        let (q0, s0) = act_quant(&[0f32; 32], 32);
        assert!(s0[0] > 0.0 && s0[0] == pow2_ceil(1e-4f32 * inv448()));
        assert!(q0.iter().all(|v| *v == 0.0));
        // |x / s| <= 448 always and the quant-dequant is idempotent
        let mut y: Vec<f32> = (0..64).map(|i| bf16r(((i as f32) * 0.37).sin() * 3.0e3)).collect();
        act_quant_inplace(&mut y, 32);
        let z = y.clone();
        act_quant_inplace(&mut y, 32);
        assert_eq!(y, z);
    }

    #[test]
    fn dsv41_od_e8m0_and_e2m1() {
        assert_eq!(e8m0_decode(127), 1.0);
        assert_eq!(e8m0_decode(130), 8.0);
        assert_eq!(e8m0_decode(120), 2f32.powi(-7));
        assert!(e8m0_decode(255).is_nan());
        assert_eq!(E2M1[7], 6.0);
        assert_eq!(E2M1[15], -6.0);
    }

    #[test]
    fn dsv41_od_qlinear_matches_dequantised_f64_within_block_rounding() {
        // fp8: random e4m3 weights / scales against an explicit dequantised product
        let (n, k): (usize, usize) = (40, 64);
        let mut st = 12345u64;
        let mut rnd = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
        let w: Vec<u8> = (0..n * k).map(|_| { let b = (rnd() % 256) as u8; if b & 0x7F == 0x7F { 0x30 } else { b } }).collect();
        let s: Vec<u8> = (0..n.div_ceil(32) * (k / 32)).map(|_| 120 + (rnd() % 10) as u8).collect();
        let x: Vec<f32> = (0..k).map(|_| bf16r(((rnd() % 2000) as f32 - 1000.0) / 37.0)).collect();
        let lin = QLin::Fp8 { n, k, w: w.clone(), s: s.clone() };
        let y = qlinear(&[x.clone()], &lin);
        let (q, sa) = act_quant(&x, 32);
        for c in 0..n {
            let mut exact = 0f64;
            for j in 0..k { exact += (q[j] * sa[j / 32]) as f64 * (e4m3_decode(w[c * k + j]) * e8m0_decode(s[(c / 32) * (k / 32) + j / 32])) as f64; }
            let got = y[0][c] as f64;
            assert!((got - exact).abs() <= exact.abs() * 2f64.powi(-7) + 1e-30, "col {c}: {got} vs {exact}");
        }
        // fp4: low nibble first
        let w4: Vec<u8> = (0..n * k / 2).map(|_| (rnd() % 256) as u8).collect();
        let s4: Vec<u8> = (0..n * (k / 32)).map(|_| 124 + (rnd() % 6) as u8).collect();
        let lin4 = QLin::Fp4 { n, k, w: w4.clone(), s: s4.clone() };
        let y4 = qlinear(&[x], &lin4);
        for c in 0..n {
            let mut exact = 0f64;
            for j in 0..k {
                let byte = w4[c * k / 2 + j / 2];
                let code = if j % 2 == 0 { byte & 0xF } else { byte >> 4 };
                exact += (q[j] * sa[j / 32]) as f64 * (E2M1[code as usize] * e8m0_decode(s4[c * (k / 32) + j / 32])) as f64;
            }
            assert!((y4[0][c] as f64 - exact).abs() <= exact.abs() * 2f64.powi(-7) + 1e-30, "fp4 col {c}");
        }
    }

    #[test]
    fn dsv41_od_sinkhorn_is_doubly_stochastic() {
        let mixes: Vec<f32> = (0..24).map(|i| ((i as f32) * 0.7).sin()).collect();
        let scale = [0.5f32, 0.25, 2.0];
        let base: Vec<f32> = (0..24).map(|i| ((i as f32) * 1.3).cos() * 0.1).collect();
        let (pre, post, comb) = sinkhorn(&mixes, &scale, &base, 4, 20, 1e-6);
        assert!(pre.iter().all(|p| *p > 0.0 && *p < 1.0 + 1e-5));
        assert!(post.iter().all(|p| *p > 0.0 && *p < 2.0));
        for k in 0..4 { let cs: f32 = (0..4).map(|j| comb[j][k]).sum(); assert!((cs - 1.0).abs() < 1e-5, "col {k} sums to {cs}"); }
        for j in 0..4 { let rs: f32 = comb[j].iter().sum(); assert!((rs - 1.0).abs() < 1e-3, "row {j} sums to {rs}"); }
    }

    #[test]
    fn dsv41_od_sparse_attn_matches_plain_softmax_and_masks() {
        let (h, d) = (2, 8);
        let kv: Vec<Vec<f32>> = (0..70).map(|r| (0..d).map(|e| bf16r(((r * 7 + e) as f32 * 0.13).sin())).collect()).collect();
        let q = vec![(0..h * d).map(|e| bf16r((e as f32 * 0.29).cos())).collect::<Vec<f32>>()];
        let idx = vec![(0..70).map(|i| if i == 5 { -1 } else { i as i32 }).collect::<Vec<i32>>()];
        let sink = [0.3f32, -1.0];
        let out = sparse_attn(&q, h, d, &kv, &idx, &sink, 0.35);
        for hh in 0..h {
            let qv = &q[0][hh * d..(hh + 1) * d];
            let sc: Vec<f64> = (0..70).filter(|&r| r != 5).map(|r| (0..d).map(|e| qv[e] as f64 * kv[r][e] as f64).sum::<f64>() * 0.35).collect();
            let m = sc.iter().cloned().fold(f64::MIN, f64::max);
            let den: f64 = sc.iter().map(|s| (s - m).exp()).sum::<f64>() + (sink[hh] as f64 - m).exp();
            let rows: Vec<usize> = (0..70).filter(|&r| r != 5).collect();
            for e in 0..d {
                let num: f64 = rows.iter().zip(&sc).map(|(&r, s)| (s - m).exp() * kv[r][e] as f64).sum();
                let want = num / den;
                let got = out[0][hh * d + e] as f64;
                assert!((got - want).abs() < 2e-2 * want.abs().max(0.05), "h{hh} e{e}: {got} vs {want}");
            }
        }
        // an all-masked row gives zeros
        let z = sparse_attn(&q, h, d, &kv, &[vec![-1; 3]], &sink, 0.35);
        assert!(z[0].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn dsv41_od_rope_inverse_undoes_rotation_up_to_bf16() {
        let r = Rope::plain(64, 10000.0);
        assert_eq!(r.freqs[0], 1.0);
        let mut x: Vec<f32> = (0..512).map(|i| bf16r((i as f32 * 0.011).sin())).collect();
        let x0 = x.clone();
        r.apply(&mut x, 1234, false);
        assert_eq!(&x[..448], &x0[..448], "the nope dims are untouched");
        assert_ne!(&x[448..], &x0[448..]);
        r.apply(&mut x, 1234, true);
        for i in 448..512 { assert!((x[i] - x0[i]).abs() <= 2e-2 * x0[i].abs().max(0.1), "dim {i}"); }
    }

    #[test]
    fn dsv41_od_topk_and_argmax_ties() {
        let v = [1.0f32, 3.0, 3.0, 2.0];
        assert_eq!(topk_desc(&v, 3), vec![1, 2, 3]);
        assert_eq!(argmax_first(&v), 1);
    }
}
