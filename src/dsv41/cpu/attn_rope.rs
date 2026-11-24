//! DSV4.1 attention oracle (leg OA): rotary embedding, as `model.py` `precompute_freqs_cis` (368-389)
//! and `apply_rotary_emb` (392-406) compute it.
//!
//! * Applied to the LAST `rope_dim` (64) dims of q, the window KV, the compressed KV, the index Q / K
//!   and (inverted) the attention output; adjacent element pairs are one complex number.
//! * Layers with `compress_ratio > 0` (2..39) use `compress_rope_theta` (160,000) WITH YaRN
//!   (`original_seq_len` 65,536, factor 16, beta 32 / 1: ramp over dims 15..25); layers with ratio 0
//!   (0, 1, the DSpark stages) use `rope_theta` (10,000) without YaRN. No mscale anywhere.
//! * Every quantity is fp32 as torch forms it ON CUDA (REFb built `freqs_cis` on the GPU; leg KI
//!   measured the rules): `1 / base^(arange(0,d,2)/d)` with the power rounded once; the YaRN ramp
//!   `(i - low) / (high - low)` as torch-CUDA divides a tensor by a Python scalar, i.e. a multiply by
//!   the fp32 reciprocal (entry 24 of the table differs from torch-CPU's true division); the blend
//!   `f / factor * (1 - smooth) + f * smooth` op by op; the angle `fp32(pos) * freq` rounded once;
//!   `polar` = (cos, sin) of that angle. The rotation is torch-CUDA's complex multiply,
//!   `re = fma(a, c, -(b s))`, `im = fma(b, c, a s)`, then the value is rounded to bf16.
//! * cos / sin here are correctly rounded (formed in f64); CUDA's `sincosf` can differ by one fp32
//!   ulp, which this CPU oracle cannot reproduce (visible only in rare bf16 flips).

/// One layer class's rotary frequencies.
#[derive(Clone, Debug)]
pub struct Rope {
    pub dim: usize,
    /// dim / 2 frequencies (fp32)
    pub freqs: Vec<f32>,
}

/// YaRN parameters (`original_seq_len` > 0 turns it on).
#[derive(Clone, Copy, Debug)]
pub struct Yarn {
    pub original_seq_len: f64,
    pub factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
}

impl Rope {
    /// `precompute_freqs_cis(dim, .., original_seq_len, base, factor, beta_fast, beta_slow)` frequencies.
    pub fn new(dim: usize, base: f64, yarn: Option<Yarn>) -> Rope {
        let half = dim / 2;
        let mut freqs: Vec<f32> = (0..half)
            .map(|i| {
                let t = (2 * i) as f32 / dim as f32;
                1.0f32 / (base.powf(t as f64) as f32)
            })
            .collect();
        if let Some(y) = yarn.filter(|y| y.original_seq_len > 0.0) {
            let d = dim as f64;
            let corrected = |rot: f64| d * (y.original_seq_len / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
            let low = corrected(y.beta_fast).floor().max(0.0);
            let high = corrected(y.beta_slow).ceil().min(d - 1.0);
            let inv_span = 1.0f32 / (high - low).max(1e-3) as f32;
            let factor = y.factor as f32;
            for (i, f) in freqs.iter_mut().enumerate() {
                let ramp = ((i as f32 - low as f32) * inv_span).clamp(0.0, 1.0);
                let smooth = 1.0f32 - ramp;
                let a = *f / factor;
                let b = 1.0f32 - smooth;
                *f = a * b + *f * smooth;
            }
        }
        Rope { dim, freqs }
    }

    /// (cos, sin) of `fp32(pos) * freq` for every frequency.
    pub fn cis(&self, pos: usize) -> Vec<(f32, f32)> {
        let p = pos as f32;
        self.freqs
            .iter()
            .map(|&f| {
                let a = (p * f) as f64;
                (a.cos() as f32, a.sin() as f32)
            })
            .collect()
    }

    /// Rotate `x` (exactly `dim` bf16 values: the rope tail) in place at `pos`; `inverse` conjugates.
    pub fn apply(&self, x: &mut [f32], pos: usize, inverse: bool) {
        let cis = self.cis(pos);
        self.apply_cis(x, &cis, inverse);
    }

    /// As `apply`, with the (cos, sin) table of the position precomputed.
    pub fn apply_cis(&self, x: &mut [f32], cis: &[(f32, f32)], inverse: bool) {
        debug_assert_eq!(x.len(), self.dim);
        for (i, &(c, s)) in cis.iter().enumerate() {
            let (a, b) = (x[2 * i], x[2 * i + 1]);
            let s = if inverse { -s } else { s };
            let re = a.mul_add(c, -(b * s));
            let im = b.mul_add(c, a * s);
            x[2 * i] = super::num::bf16(re);
            x[2 * i + 1] = super::num::bf16(im);
        }
    }
}
