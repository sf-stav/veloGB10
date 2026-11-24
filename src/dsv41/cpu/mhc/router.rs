//! DSV4.1 OM oracle: the MoE router (`Gate`, `model.py` 794-829), leg OM.
//!
//! ```text
//! scores  = F.linear(x.float(), weight.float()) / gate_temp        # fp32 GEMM, TF32 off; gate_temp = 1
//! scores  = sqrt(softplus(scores))                                  # "sqrtsoftplus" (softmax / sigmoid also exist)
//! bias    = bias_vl if the row is inside an image span else bias    # noaux_tc: the bias SELECTS ...
//! indices = (scores + bias).topk(k)                                 # ... but never weights
//! weights = scores.gather(indices)
//! weights /= weights.sum(-1) + 1e-20                                # if norm_topk_prob and k > 1 (1e-20 is a literal, not norm_eps)
//! weights *= route_scale                                            # 1.5
//! ```
//!
//! Backbone layers: 384 experts, top-6. DSpark stages: 128 experts, top-3 (same formula; the
//! drafts are text, so they never use `bias_vl`). Selection ties: the reference uses `torch.topk`,
//! whose order among equal values is implementation-defined; this oracle takes the lower expert id
//! first and REPORTS every row whose k-th and (k+1)-th selection keys are closer than the fp32 noise
//! of the GEMM (`Route::margin`), so a flip there is told apart from a semantic error.

use super::num::dot_f64;
use anyhow::{bail, Result};

/// The literal in `weights /= weights.sum(-1) + 1e-20` (model.py 827: "not norm_eps, matches training").
pub const RENORM_EPS: f32 = 1e-20;

/// `ModelArgs.score_func`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreFunc { Softmax, Sigmoid, SqrtSoftplus }

impl ScoreFunc {
    pub fn parse(s: &str) -> Result<ScoreFunc> {
        Ok(match s {
            "softmax" => ScoreFunc::Softmax,
            "sigmoid" => ScoreFunc::Sigmoid,
            "sqrtsoftplus" => ScoreFunc::SqrtSoftplus,
            x => bail!("unknown router scoring function {x:?}"),
        })
    }
}

/// One router (`Gate`) with its parameters as f32 (whatever the pack stores: BF16 native, F16 in
/// some EXL3 packs; the conversion to f32 is exact either way).
#[derive(Clone, Debug)]
pub struct GateParams {
    /// `[n_experts, dim]` row-major
    pub weight: Vec<f32>,
    /// `[n_experts]` selection bias (`noaux_tc` correction bias)
    pub bias: Vec<f32>,
    /// `[n_experts]` selection bias for image-span rows (absent when the pack has no vision)
    pub bias_vl: Option<Vec<f32>>,
    pub n_experts: usize,
    pub dim: usize,
    pub top_k: usize,
    pub route_scale: f32,
    pub score: ScoreFunc,
    pub norm_topk_prob: bool,
    pub gate_temp: f32,
}

/// The router's decision for one row.
#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    /// selected experts, in `topk` order (descending selection key, lower id first on equal keys)
    pub ids: Vec<u32>,
    /// routing weights (renormalised x route_scale), aligned with `ids`
    pub weights: Vec<f32>,
    /// k-th minus (k+1)-th selection key (`score + bias`); 0 = an exact tie at the selection boundary
    pub margin: f32,
    /// the k-th selection key (to put `margin` on a relative scale)
    pub kth_key: f32,
}

/// `F.softplus(x)` (beta 1, threshold 20): `x` above the threshold, else `log1p(exp(x))`.
/// Computed in f64 and rounded once (the GPU's `log1pf(expf(x))` is within a few ulp of it).
pub fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 { x } else { ((x as f64).exp().ln_1p()) as f32 }
}

fn sigmoid_f32(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

/// The router scores of one row (`[n_experts]`, fp32): GEMM in f64 rounded once to fp32, then the
/// score function in fp32.
pub fn gate_scores(g: &GateParams, x: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), g.dim, "gate: row length {} vs dim {}", x.len(), g.dim);
    let logits: Vec<f32> = g.weight.chunks(g.dim).map(|w| dot_f64(w, x) as f32 / g.gate_temp).collect();
    match g.score {
        ScoreFunc::SqrtSoftplus => logits.iter().map(|&v| softplus_f32(v).sqrt()).collect(),
        ScoreFunc::Sigmoid => logits.iter().map(|&v| sigmoid_f32(v)).collect(),
        ScoreFunc::Softmax => {
            let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f64> = logits.iter().map(|&v| ((v - m) as f64).exp()).collect();
            let s: f64 = e.iter().sum();
            e.iter().map(|v| (v / s) as f32).collect()
        }
    }
}

/// Indices of the `k` largest keys, in descending key order, lower index first on equal keys
/// (NaN keys never win). Also returns the k-th and (k+1)-th keys.
pub fn topk_desc(keys: &[f32], k: usize) -> (Vec<u32>, f32, f32) {
    let mut idx: Vec<u32> = (0..keys.len() as u32).collect();
    let key = |i: u32| { let v = keys[i as usize]; if v.is_nan() { f32::NEG_INFINITY } else { v } };
    idx.sort_by(|&a, &b| key(b).partial_cmp(&key(a)).unwrap().then(a.cmp(&b)));
    let kth = idx.get(k.saturating_sub(1)).map_or(f32::NEG_INFINITY, |&i| key(i));
    let next = idx.get(k).map_or(f32::NEG_INFINITY, |&i| key(i));
    idx.truncate(k);
    (idx, kth, next)
}

/// Route one row given its scores (split out so a GPU leg can feed its own scores).
pub fn route_from_scores(g: &GateParams, scores: &[f32], image: bool) -> Route {
    let bias = match (&g.bias_vl, image) { (Some(b), true) => b, _ => &g.bias };
    let keys: Vec<f32> = scores.iter().zip(bias).map(|(s, b)| s + b).collect();
    let (ids, kth, next) = topk_desc(&keys, g.top_k);
    let mut w: Vec<f32> = ids.iter().map(|&i| scores[i as usize]).collect();
    if g.norm_topk_prob && g.top_k > 1 {
        // torch: weights.sum(dim=-1) (sequential over 6 values) + 1e-20, then an fp32 divide
        let mut s = 0f32;
        for &v in &w {
            s += v;
        }
        let den = s + RENORM_EPS;
        for v in w.iter_mut() {
            *v /= den;
        }
    }
    for v in w.iter_mut() {
        *v *= g.route_scale;
    }
    Route { ids, weights: w, margin: kth - next, kth_key: kth }
}

/// The full router for one row. `image` = the row is inside an image span (uses `bias_vl`).
pub fn route(g: &GateParams, x: &[f32], image: bool) -> Route { route_from_scores(g, &gate_scores(g, x), image) }
