//! DSV4.1 CPU oracle, leg OM: mHC (Hyper-Connections, hc = 4 streams), the norms and the router.
//!
//! Reference: DeepSeek's `inference/model.py` (`Block` 909-996, `hc_split_sinkhorn` kernel.py
//! 407-474, `RMSNorm` 280-292, `Gate` 794-829, `Transformer.forward` 1244-1275); REF map §2.7,
//! §3.2, §3.6, §4 items 1-3, 15-17. Everything here is row-local, so one row at a time.
//!
//! The residual stream of a row is `hc` copies of `dim` values (`[hc, dim]`, bf16). A block is
//!
//! ```text
//! residual = x
//! attn_pre, attn_post, attn_comb = hc_mixes(x)        # from the block INPUT
//! x = hc_pre(x, pre_mix)                              # pre_mix = the PREVIOUS block's ffn_pre (one-hot at layer 0)
//! x = hc_post(attn(attn_norm(x)), residual, attn_post, attn_comb)
//! residual = x
//! ffn_pre, ffn_post, ffn_comb = hc_mixes(x)           # from the stream after attention
//! x = hc_pre(x, attn_pre)                             # attn_pre, NOT ffn_pre: the delayed pre-mix
//! x = hc_post(ffn(ffn_norm(x)), residual, ffn_post, ffn_comb)
//! return x, ffn_pre                                   # ffn_pre feeds the next block (or the head's hc_pre)
//! ```
//!
//! Exactness (measured against REFb's dumps, `check.rs`, report §4): `hc_pre` and `hc_post`
//! reproduce the reference bit for bit (fp32 products, sequential sums over the copies, one bf16
//! rounding); the Sinkhorn reproduces DeepSeek's TileLang kernel's operation order (FMA-contracted
//! `m * scale + base`, xor-butterfly 4-lane sums, IEEE divides); `hc_mixes` sums its 20,480-long dot
//! products in f64 (the reference's cuBLAS fp32 order is not reproducible), so the coefficients
//! agree to ~1e-7 relative, not bitwise.
//!
//! Under TP (`PLAN/DSV41_TP_DESIGN.md` §3.5) each rank computes the 24 dot products over its K slice of
//! `hc_fn` plus its part of the sum of squares (`hc_mix_partial`), and the 25 fp32 partials are summed
//! in rank order after the sublayer's reduce (`hc_mixes_from_partials`).
//!
//! Sub-modules: `num` (bf16, ulps, comparisons), `norm` (RMSNorm), `router` (Gate), `params` (load
//! these parameters from any pack through the census), `refdump` (REFb's dump layout), `check` (the
//! `--probe-dsv41-oracle --dsv41-oracle-part mhc` validation against REFb's dumps).

pub mod check;
pub mod norm;
pub mod num;
pub mod params;
pub mod refdump;
pub mod router;
#[cfg(test)]
mod tests;

use crate::dsv41::config::Dsv41Config;
use anyhow::{ensure, Result};
use num::{bf16_round, dot_f64, rsqrt_f32, sum_sq_f32};

/// The mHC shape and constants of a model (from its config, never literals).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HcShape {
    /// streams (`hc_mult`, 4)
    pub hc: usize,
    /// model width (5,120)
    pub dim: usize,
    /// Sinkhorn iterations (20)
    pub iters: usize,
    /// Sinkhorn eps (1e-6)
    pub hc_eps: f32,
    /// RMS eps of the mixes' normalisation and of every RMSNorm (`norm_eps`, 1e-20)
    pub norm_eps: f32,
}

impl HcShape {
    pub fn of(cfg: &Dsv41Config) -> HcShape {
        HcShape { hc: cfg.hc, dim: cfg.hidden, iters: cfg.sinkhorn_iters, hc_eps: cfg.hc_eps as f32, norm_eps: cfg.rms_eps as f32 }
    }
    /// Rows of `hc_fn`: (2 + hc) x hc (24).
    pub fn mix(&self) -> usize { (2 + self.hc) * self.hc }
    /// Flattened stream width: hc x dim (20,480).
    pub fn width(&self) -> usize { self.hc * self.dim }
}

/// One mixer's parameters (`hc_attn_*` or `hc_ffn_*`), fp32.
#[derive(Clone, Debug)]
pub struct HcMixParams {
    /// `hc_*_fn`: `[mix, hc * dim]` row-major
    pub fn_: Vec<f32>,
    /// `hc_*_scale`: pre, post, comb
    pub scale: [f32; 3],
    /// `hc_*_base`: `[mix]`
    pub base: Vec<f32>,
}

/// The coefficients one mixer produces for one row.
#[derive(Clone, Debug, PartialEq)]
pub struct Coeffs {
    /// `[hc]`: weights that collapse the copies into one sublayer input (sigmoid + eps)
    pub pre: Vec<f32>,
    /// `[hc]`: how much of the sublayer output each copy receives (2 x sigmoid)
    pub post: Vec<f32>,
    /// `[hc, hc]` row-major: `comb[j * hc + k]` weights residual copy `j` into output copy `k`
    /// (columns sum to 1 after Sinkhorn)
    pub comb: Vec<f32>,
}

/// The one-hot pre-mix the first block's attention (and the first DSpark stage) uses.
pub fn identity_pre_mix(hc: usize) -> Vec<f32> {
    let mut v = vec![0f32; hc];
    v[0] = 1.0;
    v
}

/// `h.unsqueeze(2).repeat(1, 1, hc, 1)`: the embedding row copied into the hc streams.
pub fn expand(s: &HcShape, h: &[f32]) -> Vec<f32> {
    assert_eq!(h.len(), s.dim);
    let mut v = Vec::with_capacity(s.width());
    for _ in 0..s.hc {
        v.extend_from_slice(h);
    }
    v
}

/// `hc_mixes` up to the Sinkhorn (model.py 950-956): `F.linear(x_flat, hc_fn) * rsqrt(mean(x_flat^2) + eps)`
/// with one statistic over the whole flattened `hc * dim` stream, applied AFTER the fp32 matmul.
pub fn hc_mixes(s: &HcShape, p: &HcMixParams, x: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), s.width(), "hc_mixes: stream length");
    assert_eq!(p.fn_.len(), s.mix() * s.width(), "hc_mixes: hc_fn size");
    let r = rsqrt_f32(sum_sq_f32(x) * (1.0f32 / s.width() as f32) + s.norm_eps);
    p.fn_.chunks(s.width()).map(|w| (dot_f64(w, x) as f32) * r).collect()
}

/// One rank's share of a mix under the K split of `hc_fn` (TP design §3.5): the `mix` dot products
/// over the K range `[k0, k1)` of the flattened stream and that range's sum of squares, fp32.
#[derive(Clone, Debug, PartialEq)]
pub struct MixPartial {
    pub dots: Vec<f32>,
    pub sum_sq: f32,
}

/// The partial of the K range `[k0, k1)` (a rank's slice). `x` is the full flattened stream.
pub fn hc_mix_partial(s: &HcShape, p: &HcMixParams, x: &[f32], k0: usize, k1: usize) -> MixPartial {
    assert!(k0 <= k1 && k1 <= s.width() && x.len() == s.width());
    let w = s.width();
    let dots = (0..s.mix()).map(|m| dot_f64(&p.fn_[m * w + k0..m * w + k1], &x[k0..k1]) as f32).collect();
    MixPartial { dots, sum_sq: sum_sq_f32(&x[k0..k1]) }
}

/// The mixes from the ranks' partials, summed in fp32 in rank order (what the post-reduce kernel does).
pub fn hc_mixes_from_partials(s: &HcShape, parts: &[MixPartial]) -> Vec<f32> {
    assert!(!parts.is_empty());
    let mut dots = parts[0].dots.clone();
    let mut ss = parts[0].sum_sq;
    for p in &parts[1..] {
        for (d, v) in dots.iter_mut().zip(&p.dots) {
            *d += *v;
        }
        ss += p.sum_sq;
    }
    let r = rsqrt_f32(ss * (1.0f32 / s.width() as f32) + s.norm_eps);
    dots.iter().map(|d| d * r).collect()
}

/// The equal K ranges of an `n`-way split of the flattened stream (rank order).
pub fn k_ranges(s: &HcShape, n: usize) -> Vec<(usize, usize)> {
    let w = s.width();
    (0..n).map(|r| (r * w / n, (r + 1) * w / n)).collect()
}

/// Sum of a power-of-two number of lanes in TileLang's xor-butterfly order (offset n/2, ..., 1):
/// for 4 lanes `(c0 + c2) + (c1 + c3)`.
fn butterfly_sum(lanes: &[f32]) -> f32 {
    let n = lanes.len();
    let mut v = lanes.to_vec();
    let mut step = n / 2;
    while step >= 1 {
        let prev = v.clone();
        for i in 0..n {
            v[i] = prev[i] + prev[i ^ step];
        }
        step /= 2;
    }
    v[0]
}

fn sigmoid_f32(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

/// `hc_split_sinkhorn` (kernel.py 407-474) for one row, in the TileLang kernel's operation order:
///
/// ```text
/// pre[j]     = sigmoid(m[j] * s0 + b[j]) + eps                    (a*b+c FMA-contracted by nvcc)
/// post[j]    = 2 * sigmoid(m[hc + j] * s1 + b[hc + j])
/// comb[j][k] = m[2hc + j*hc + k] * s2 + b[...]
/// comb = exp(comb - rowmax) / rowsum + eps; comb /= colsum + eps
/// repeat iters - 1:  comb /= rowsum + eps; comb /= colsum + eps
/// ```
///
/// 4-lane sums in the xor-butterfly order. `hc` must be a power of two (the config's 4 is).
pub fn hc_split_sinkhorn(s: &HcShape, mixes: &[f32], scale: &[f32; 3], base: &[f32]) -> Coeffs {
    let hc = s.hc;
    assert!(hc.is_power_of_two(), "hc_split_sinkhorn: hc {hc} is not a power of two (the butterfly order needs one)");
    assert_eq!(mixes.len(), s.mix());
    assert_eq!(base.len(), s.mix());
    let eps = s.hc_eps;
    let pre: Vec<f32> = (0..hc).map(|j| sigmoid_f32(mixes[j].mul_add(scale[0], base[j])) + eps).collect();
    let post: Vec<f32> = (0..hc).map(|j| 2.0 * sigmoid_f32(mixes[hc + j].mul_add(scale[1], base[hc + j]))).collect();
    let mut c: Vec<f32> = (0..hc * hc).map(|i| mixes[2 * hc + i].mul_add(scale[2], base[2 * hc + i])).collect();
    let row_sums = |c: &[f32]| -> Vec<f32> { (0..hc).map(|j| butterfly_sum(&c[j * hc..(j + 1) * hc])).collect() };
    let col_sums = |c: &[f32]| -> Vec<f32> { (0..hc).map(|k| butterfly_sum(&(0..hc).map(|j| c[j * hc + k]).collect::<Vec<f32>>())).collect() };
    // softmax over k (max-subtracted), then + eps
    for j in 0..hc {
        let m = c[j * hc..(j + 1) * hc].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for k in 0..hc {
            c[j * hc + k] = (c[j * hc + k] - m).exp();
        }
    }
    let rs = row_sums(&c);
    for j in 0..hc {
        for k in 0..hc {
            c[j * hc + k] = c[j * hc + k] / rs[j] + eps;
        }
    }
    let cs = col_sums(&c);
    for j in 0..hc {
        for k in 0..hc {
            c[j * hc + k] /= cs[k] + eps;
        }
    }
    for _ in 1..s.iters {
        let rs = row_sums(&c);
        for j in 0..hc {
            for k in 0..hc {
                c[j * hc + k] /= rs[j] + eps;
            }
        }
        let cs = col_sums(&c);
        for j in 0..hc {
            for k in 0..hc {
                c[j * hc + k] /= cs[k] + eps;
            }
        }
    }
    Coeffs { pre, post, comb: c }
}

/// `hc_mixes` + `hc_split_sinkhorn`: one mixer's coefficients for one row.
pub fn hc_coeffs(s: &HcShape, p: &HcMixParams, x: &[f32]) -> Coeffs { hc_split_sinkhorn(s, &hc_mixes(s, p, x), &p.scale, &p.base) }

/// `hc_pre` (model.py 959-962): `bf16(sum_j pre[j] * x[j])`, fp32 products, sequential sum over j.
pub fn hc_pre(s: &HcShape, x: &[f32], pre: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), s.width());
    assert_eq!(pre.len(), s.hc);
    let d = s.dim;
    (0..d)
        .map(|i| {
            let mut acc = pre[0] * x[i];
            for j in 1..s.hc {
                acc += pre[j] * x[j * d + i];
            }
            bf16_round(acc)
        })
        .collect()
}

/// `hc_post` (model.py 964-968): `out[k] = bf16(post[k] * y + sum_j comb[j][k] * residual[j])`,
/// fp32 products, sequential sum over j, then the add, one bf16 rounding.
pub fn hc_post(s: &HcShape, y: &[f32], residual: &[f32], post: &[f32], comb: &[f32]) -> Vec<f32> {
    let (hc, d) = (s.hc, s.dim);
    assert_eq!(y.len(), d);
    assert_eq!(residual.len(), s.width());
    assert_eq!(post.len(), hc);
    assert_eq!(comb.len(), hc * hc);
    let mut out = vec![0f32; s.width()];
    for k in 0..hc {
        for i in 0..d {
            let mut acc = comb[k] * residual[i];
            for j in 1..hc {
                acc += comb[j * hc + k] * residual[j * d + i];
            }
            out[k * d + i] = bf16_round(post[k] * y[i] + acc);
        }
    }
    out
}

/// `h.mean(dim=2)` of a bf16 stream (the DSpark tap of the target layers' inputs, model.py 1268):
/// fp32 sequential sum over the copies, times 1/hc, one bf16 rounding.
pub fn stream_mean(s: &HcShape, x: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), s.width());
    let d = s.dim;
    let f = 1.0f32 / s.hc as f32;
    (0..d)
        .map(|i| {
            let mut acc = x[i];
            for j in 1..s.hc {
                acc += x[j * d + i];
            }
            bf16_round(acc * f)
        })
        .collect()
}

/// One block's mHC + norm parameters.
#[derive(Clone, Debug)]
pub struct BlockParams {
    pub attn_mix: HcMixParams,
    pub ffn_mix: HcMixParams,
    /// `attn_norm.weight` `[dim]`
    pub attn_norm: Vec<f32>,
    /// `ffn_norm.weight` `[dim]`
    pub ffn_norm: Vec<f32>,
}

/// Everything one block computes around its two sublayers, for one row.
#[derive(Clone, Debug)]
pub struct BlockTrace {
    /// the attention mixer's coefficients (from the block input)
    pub attn: Coeffs,
    /// `attn_norm(hc_pre(x_in, pre_mix_in))`: the attention's input
    pub attn_in: Vec<f32>,
    /// the stream after the attention's `hc_post`
    pub x_mid: Vec<f32>,
    /// the FFN mixer's coefficients (from `x_mid`); `ffn.pre` is the next block's pre-mix
    pub ffn: Coeffs,
    /// `ffn_norm(hc_pre(x_mid, attn.pre))`: the FFN's input
    pub ffn_in: Vec<f32>,
    /// the block's output stream
    pub x_out: Vec<f32>,
}

/// `Block.forward` (model.py 970-996) for one row, with the delayed pre-mix. `attn` maps the
/// attention input (`[dim]`, bf16) to the attention output (`[dim]`, bf16); `ffn` likewise for the
/// MoE. Returns the trace; `trace.ffn.pre` is the pre-mix the next block's attention uses.
pub fn block_forward(
    s: &HcShape,
    b: &BlockParams,
    x_in: &[f32],
    pre_mix_in: &[f32],
    attn: &mut dyn FnMut(&[f32]) -> Result<Vec<f32>>,
    ffn: &mut dyn FnMut(&[f32]) -> Result<Vec<f32>>,
) -> Result<BlockTrace> {
    ensure!(x_in.len() == s.width() && pre_mix_in.len() == s.hc, "block_forward: shapes");
    let ac = hc_coeffs(s, &b.attn_mix, x_in);
    let attn_in = norm::rms_norm(&hc_pre(s, x_in, pre_mix_in), &b.attn_norm, s.norm_eps);
    let ao = attn(&attn_in)?;
    ensure!(ao.len() == s.dim, "block_forward: attention output length {}", ao.len());
    let x_mid = hc_post(s, &ao, x_in, &ac.post, &ac.comb);
    let fc = hc_coeffs(s, &b.ffn_mix, &x_mid);
    let ffn_in = norm::rms_norm(&hc_pre(s, &x_mid, &ac.pre), &b.ffn_norm, s.norm_eps);
    let fo = ffn(&ffn_in)?;
    ensure!(fo.len() == s.dim, "block_forward: FFN output length {}", fo.len());
    let x_out = hc_post(s, &fo, &x_mid, &fc.post, &fc.comb);
    Ok(BlockTrace { attn: ac, attn_in, x_mid, ffn: fc, ffn_in, x_out })
}

/// The head's input after the last block (model.py 1271-1272): `norm(hc_pre(x, last ffn_pre))`.
/// Returns `(hc_pre output, normed)`; the reference's `h_final` dump is the first.
pub fn head_input(s: &HcShape, x_last: &[f32], ffn_pre_last: &[f32], final_norm: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let h = hc_pre(s, x_last, ffn_pre_last);
    let n = norm::rms_norm(&h, final_norm, s.norm_eps);
    (h, n)
}
