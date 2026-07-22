//! Hand-computed gate for the hy_v3 router formula:
//!   sigmoid(logits) fp32 → top-k SELECTED by sigmoid+expert_bias → weights = the UN-biased
//!   sigmoid scores at the selected indices → renormalized (route_norm, +1e-20) × router_scaling.
//!
//! This pins the FORMULA the kernel must implement (transformers HYV3TopKRouter:
//! `routing_weights.gather(top_k_index) / (sum + 1e-20) * router_scaling_factor`). The GPU kernel
//! itself (moe_router_topk_sigmoid_b) is gated end-to-end by scripts/verify_hy3_moe.py over the
//! --probe-moe dump of the quantized fixture.

/// EXACT mirror of kernels/gpu_batch.cu moe_router_topk_sigmoid_b for E <= 32 (one expert per
/// lane: the warp argmax then keeps the LOWEST index on ties, as does this scan).
fn route_hy3(logits: &[f32], bias: &[f32], k: usize, route_norm: bool, scaling: f32) -> (Vec<usize>, Vec<f32>) {
    let e = logits.len();
    let sig: Vec<f32> = logits.iter().map(|&l| 1.0 / (1.0 + (-l).exp())).collect();
    let mut sel: Vec<f32> = (0..e).map(|i| sig[i] + bias[i]).collect();
    let mut ids = Vec::with_capacity(k);
    for _ in 0..k {
        let mut best = 0usize;
        for i in 1..e {
            if sel[i] > sel[best] { best = i; }       // strictly greater -> lowest index on ties
        }
        ids.push(best);
        sel[best] = -1e30;
    }
    let mut wts: Vec<f32> = ids.iter().map(|&i| sig[i]).collect();   // UN-biased scores
    if route_norm {
        let sum: f32 = wts.iter().sum();
        for w in wts.iter_mut() { *w = *w / (sum + 1e-20) * scaling; }
    } else {
        for w in wts.iter_mut() { *w *= scaling; }
    }
    (ids, wts)
}

const SC: f32 = 2.826;

#[test]
fn bias_flips_selection_but_never_the_weights() {
    // sig = [0.8808, 0.7311, 0.5, 0.2689]; bias lifts expert 2 (0.5 + 0.5 = 1.0) to rank 1.
    let logits = [2.0f32, 1.0, 0.0, -1.0];
    let bias = [0.0f32, 0.0, 0.5, 0.0];
    let (ids, wts) = route_hy3(&logits, &bias, 2, true, SC);
    assert_eq!(ids, vec![2, 0], "bias must drive the selection (no-bias top-2 would be [0, 1])");
    // Weights are the UN-biased sigmoid scores at the selected indices, renormalized, x 2.826.
    let s0 = 1.0 / (1.0 + (-2.0f32).exp());      // 0.880797
    let s2 = 0.5f32;
    let sum = s0 + s2;
    let w0 = s2 / (sum + 1e-20) * SC;
    let w1 = s0 / (sum + 1e-20) * SC;
    assert!((wts[0] - w0).abs() < 1e-6, "{} vs {}", wts[0], w0);
    assert!((wts[1] - w1).abs() < 1e-6, "{} vs {}", wts[1], w1);
    // route_norm: the renormalized weights sum to the scaling factor, not to 1.
    let tot: f32 = wts.iter().sum();
    assert!((tot - SC).abs() < 1e-5, "sum(wts) {} != router_scaling {}", tot, SC);
}

#[test]
fn route_norm_false_leaves_scores_unnormalized() {
    let logits = [2.0f32, 1.0, 0.0, -1.0];
    let bias = [0.0f32, 0.0, 0.5, 0.0];
    let (ids, wts) = route_hy3(&logits, &bias, 2, false, SC);
    assert_eq!(ids, vec![2, 0]);
    let s0 = 1.0 / (1.0 + (-2.0f32).exp());
    assert!((wts[0] - 0.5 * SC).abs() < 1e-6);
    assert!((wts[1] - s0 * SC).abs() < 1e-6);
}

#[test]
fn zero_bias_is_plain_sigmoid_topk() {
    let logits = [0.3f32, -1.2, 2.2, 0.7];
    let bias = [0.0f32; 4];
    let (ids, wts) = route_hy3(&logits, &bias, 2, true, SC);
    assert_eq!(ids, vec![2, 3], "monotonic sigmoid -> top-k of logits");
    let tot: f32 = wts.iter().sum();
    assert!((tot - SC).abs() < 1e-5);
}

#[test]
fn ties_break_toward_the_lower_index() {
    // Equal sigmoid scores (and equal bias): the kernel's scan keeps the lowest index.
    let logits = [1.0f32, 1.0, 0.5];
    let bias = [0.0f32; 3];
    let (ids, _) = route_hy3(&logits, &bias, 1, true, SC);
    assert_eq!(ids, vec![0]);
}
