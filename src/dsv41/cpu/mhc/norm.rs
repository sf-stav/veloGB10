//! DSV4.1 OM oracle: RMSNorm (leg OM). `model.py` 280-292:
//!
//! ```text
//! x   = x.float()
//! var = x.square().mean(-1)
//! x   = x * rsqrt(var + eps)                 # fp32
//! return (weight * x).to(bf16)               # bf16 weight promoted to fp32, one rounding
//! ```
//!
//! Every RMSNorm of this model (`attn_norm`, `ffn_norm`, the final `norm`, `q_norm`, `kv_norm`, the
//! compressor / indexer norms, the DSpark `main_norm` / `norm`) uses `eps = norm_eps = 1e-20`, which
//! is effectively zero: a row of zeros gives `rsqrt(1e-20) = 1e10` times zero, i.e. zeros, never NaN.
//! The weight is NOT `1 + w` (that is the Qwen convention); it multiplies directly.
//!
//! Exactness against the reference: the product order is reproduced exactly; the 5,120-element mean
//! and `rsqrtf` are not (GPU reduction order, 2-ulp `rsqrtf`), so about 5 in 10^6 outputs differ by
//! one bf16 ulp (measured on REFb's dumps, report §4).

use super::num::{bf16_round, mean_sq_f32, rsqrt_f32};

/// RMSNorm of one row: `bf16(w * (x * rsqrt(mean(x^2) + eps)))`. `x` and `w` are the row and the
/// weight as f32 (bf16 values); returns bf16 values.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), w.len(), "rms_norm: row {} vs weight {}", x.len(), w.len());
    let r = rsqrt_f32(mean_sq_f32(x) + eps);
    x.iter().zip(w).map(|(&v, &g)| bf16_round(g * (v * r))).collect()
}

/// The fp32 statistic alone (`rsqrt(mean(x^2) + eps)`), for kernels that fold it into a prologue.
pub fn rms_scale(x: &[f32], eps: f32) -> f32 { rsqrt_f32(mean_sq_f32(x) + eps) }

/// RMSNorm over several rows laid out back to back (`rows x w.len()`).
pub fn rms_norm_rows(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let d = w.len();
    assert!(d > 0 && x.len() % d == 0, "rms_norm_rows: {} values are not rows of {d}", x.len());
    x.chunks(d).flat_map(|r| rms_norm(r, w, eps)).collect()
}
