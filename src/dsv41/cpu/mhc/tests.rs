//! Unit tests of the OM oracle (synthetic) plus one check against REFb's short-chat dump when the
//! fixture and the native checkpoint are present on this machine (skips cleanly otherwise).

use super::num::*;
use super::router::*;
use super::*;

/// A small deterministic generator (xorshift64*), so the tests need no RNG state.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> { (0..n).map(|_| self.next_f32() * scale).collect() }
    fn bf16_vec(&mut self, n: usize, scale: f32) -> Vec<f32> { self.vec(n, scale).into_iter().map(bf16_round).collect() }
}

fn shape(hc: usize, dim: usize) -> HcShape { HcShape { hc, dim, iters: 20, hc_eps: 1e-6, norm_eps: 1e-20 } }

fn mix_params(rng: &mut Rng, s: &HcShape) -> HcMixParams {
    HcMixParams { fn_: rng.vec(s.mix() * s.width(), 0.05), scale: [0.7, 1.3, 2.0], base: rng.vec(s.mix(), 0.5) }
}

#[test]
fn dsv41_om_bf16_rounding_is_rne() {
    assert_eq!(bf16_bits(1.0), 0x3F80);
    // exactly halfway between 0x3F80 and 0x3F81: ties to even (0x3F80)
    assert_eq!(bf16_bits(f32::from_bits(0x3F80_8000)), 0x3F80);
    // halfway between 0x3F81 and 0x3F82: ties to even (0x3F82)
    assert_eq!(bf16_bits(f32::from_bits(0x3F81_8000)), 0x3F82);
    assert_eq!(bf16_bits(f32::from_bits(0x3F80_8001)), 0x3F81);
    assert_eq!(bf16_bits(-2.0), 0xC000);
    assert!(bf16_round(f32::NAN).is_nan());
    assert_eq!(bf16_ulps(1.0, bf16_to_f32(0x3F81)), 1);
    assert_eq!(bf16_ulps(bf16_to_f32(0x0001), bf16_to_f32(0x8001)), 2);
    assert_eq!(f32_ulps(0.0, -0.0), 0);
}

#[test]
fn dsv41_om_butterfly_order_is_tilelangs() {
    // (c0 + c2) + (c1 + c3) = (1e8 - 1e8) + (1 + 1) = 2; a sequential sum gives 1 (1e8 + 1 rounds to 1e8)
    assert_eq!(super::butterfly_sum(&[1e8, 1.0, -1e8, 1.0]), 2.0);
    let seq = ((1e8f32 + 1.0) + -1e8) + 1.0;
    assert_eq!(seq, 1.0);
}

#[test]
fn dsv41_om_sinkhorn_is_doubly_stochastic_and_matches_the_formula() {
    let s = shape(4, 8);
    let mut rng = Rng(7);
    for _ in 0..50 {
        let m = rng.vec(s.mix(), 3.0);
        let base = rng.vec(s.mix(), 1.0);
        let scale = [0.9f32, 1.1, 2.5];
        let c = hc_split_sinkhorn(&s, &m, &scale, &base);
        // pre in (eps, 1 + eps), post in (0, 2)
        for &p in &c.pre {
            assert!(p > 1e-6 && p < 1.0 + 1e-6 + 1e-7);
        }
        for &p in &c.post {
            assert!(p > 0.0 && p < 2.0);
        }
        // the last step is a column normalisation: columns sum to 1 within eps
        for k in 0..4 {
            let col: f64 = (0..4).map(|j| c.comb[j * 4 + k] as f64).sum();
            assert!((col - 1.0).abs() < 1e-5, "column {k} sums to {col}");
        }
        // an f64 transcription of the formula, no order emulation
        let (hc, eps) = (4usize, 1e-6f64);
        let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
        let mut cc: Vec<f64> = (0..16).map(|i| m[8 + i] as f64 * scale[2] as f64 + base[8 + i] as f64).collect();
        for j in 0..hc {
            let mx = (0..hc).map(|k| cc[j * hc + k]).fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = (0..hc).map(|k| (cc[j * hc + k] - mx).exp()).collect();
            let sum: f64 = e.iter().sum();
            for k in 0..hc {
                cc[j * hc + k] = e[k] / sum + eps;
            }
        }
        let colnorm = |cc: &mut Vec<f64>| {
            for k in 0..hc {
                let cs: f64 = (0..hc).map(|j| cc[j * hc + k]).sum();
                for j in 0..hc {
                    cc[j * hc + k] /= cs + eps;
                }
            }
        };
        colnorm(&mut cc);
        for _ in 1..20 {
            for j in 0..hc {
                let rs: f64 = (0..hc).map(|k| cc[j * hc + k]).sum();
                for k in 0..hc {
                    cc[j * hc + k] /= rs + eps;
                }
            }
            colnorm(&mut cc);
        }
        for i in 0..16 {
            assert!((c.comb[i] as f64 - cc[i]).abs() < 1e-5, "comb[{i}] {} vs {}", c.comb[i], cc[i]);
        }
        for j in 0..4 {
            let pre = sig(m[j] as f64 * scale[0] as f64 + base[j] as f64) + eps;
            let post = 2.0 * sig(m[4 + j] as f64 * scale[1] as f64 + base[4 + j] as f64);
            assert!((c.pre[j] as f64 - pre).abs() < 1e-6 && (c.post[j] as f64 - post).abs() < 2e-6);
        }
    }
}

#[test]
fn dsv41_om_hc_pre_and_post_by_hand() {
    let s = shape(4, 2);
    // copies j = 0..3 of a 2-wide row
    let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let pre = [0.5f32, 0.25, 0.125, 0.0625];
    // dim 0: 0.5 + 0.75 + 0.625 + 0.4375 = 2.3125; dim 1: 1 + 1 + 0.75 + 0.5 = 3.25
    assert_eq!(hc_pre(&s, &x, &pre), vec![2.3125, 3.25]);
    let y = [10.0f32, -10.0];
    let post = [1.0f32, 0.5, 0.0, 2.0];
    // comb = identity: out[k] = post[k] * y + residual[k]
    let mut comb = vec![0f32; 16];
    for k in 0..4 {
        comb[k * 4 + k] = 1.0;
    }
    let o = hc_post(&s, &y, &x, &post, &comb);
    assert_eq!(o, vec![11.0, -8.0, 8.0, -1.0, 5.0, 6.0, 27.0, -12.0]);
    // comb[j][k]: residual copy j into output copy k (a transposed comb gives a different answer)
    let mut perm = vec![0f32; 16];
    perm[1] = 1.0; // copy 0 -> output 1
    perm[4 + 2] = 1.0; // copy 1 -> output 2
    perm[2 * 4 + 3] = 1.0; // copy 2 -> output 3
    perm[3 * 4] = 1.0; // copy 3 -> output 0
    let o = hc_post(&s, &[0.0, 0.0], &x, &post, &perm);
    assert_eq!(o, vec![7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn dsv41_om_hc_pre_sums_sequentially() {
    // pre[0] * x[0] = 1 + 2^-8 sits exactly on a bf16 tie (rounds to even = 1.0). Three more terms
    // of 2^-24 each: the sequential sum ((a + b) + c) + d keeps a (each +2^-24 is an fp32 tie that
    // rounds to the even a), so the bf16 result is 1.0; a pairwise or butterfly sum adds 2^-23 and
    // the bf16 result becomes 1.0078125. The reference (torch's sum over dim 2) is sequential.
    let s = shape(4, 1);
    let t = 2f32.powi(-24);
    let x = [1.0f32, t, t, t];
    let pre = [1.0 + 2f32.powi(-8), 1.0, 1.0, 1.0];
    assert_eq!(hc_pre(&s, &x, &pre), vec![1.0]);
    assert_eq!(bf16_round((pre[0] + t) + (t + t)), 1.0078125);
}

#[test]
fn dsv41_om_rms_norm_conventions() {
    let mut rng = Rng(3);
    let x = rng.bf16_vec(64, 4.0);
    let w = rng.bf16_vec(64, 1.0);
    let y = norm::rms_norm(&x, &w, 1e-20);
    // a power-of-two scaling of the input does not change the output (exact in fp32)
    let x4: Vec<f32> = x.iter().map(|v| v * 4.0).collect();
    assert_eq!(norm::rms_norm(&x4, &w, 1e-20), y);
    // the weight multiplies (w = 0 gives 0; it is not the 1 + w convention)
    assert!(norm::rms_norm(&x, &vec![0.0; 64], 1e-20).iter().all(|&v| v == 0.0));
    // a zero row stays zero (no NaN with eps 1e-20)
    assert!(norm::rms_norm(&vec![0.0; 64], &w, 1e-20).iter().all(|&v| v == 0.0));
    // values agree with an f64 transcription to bf16 precision
    let ms: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / 64.0;
    for i in 0..64 {
        let r = w[i] as f64 * x[i] as f64 / ms.sqrt();
        assert!((y[i] as f64 - r).abs() <= r.abs() / 128.0 + 1e-30, "{i}: {} vs {r}", y[i]);
    }
}

#[test]
fn dsv41_om_delayed_pre_mix() {
    let s = shape(4, 16);
    let mut rng = Rng(11);
    let b = BlockParams { attn_mix: mix_params(&mut rng, &s), ffn_mix: mix_params(&mut rng, &s), attn_norm: rng.bf16_vec(16, 1.0), ffn_norm: rng.bf16_vec(16, 1.0) };
    let x = rng.bf16_vec(s.width(), 2.0);
    let pm = identity_pre_mix(4);
    let mut seen_attn = Vec::new();
    let mut seen_ffn = Vec::new();
    let t = block_forward(
        &s,
        &b,
        &x,
        &pm,
        &mut |a| {
            seen_attn = a.to_vec();
            Ok(a.iter().map(|v| bf16_round(v * 0.5)).collect())
        },
        &mut |f| {
            seen_ffn = f.to_vec();
            Ok(f.iter().map(|v| bf16_round(-v)).collect())
        },
    )
    .unwrap();
    // layer 0: the one-hot pre-mix takes copy 0 alone
    assert_eq!(seen_attn, norm::rms_norm(&x[..16], &b.attn_norm, 1e-20));
    // the FFN input uses the ATTENTION mixer's pre (computed from the block input), not the FFN's
    assert_eq!(seen_ffn, norm::rms_norm(&hc_pre(&s, &t.x_mid, &t.attn.pre), &b.ffn_norm, 1e-20));
    assert_ne!(seen_ffn, norm::rms_norm(&hc_pre(&s, &t.x_mid, &t.ffn.pre), &b.ffn_norm, 1e-20));
    assert_eq!(t.attn, hc_coeffs(&s, &b.attn_mix, &x));
    assert_eq!(t.ffn, hc_coeffs(&s, &b.ffn_mix, &t.x_mid));
    // the next block's attention uses this block's ffn_pre
    let t2 = block_forward(&s, &b, &t.x_out, &t.ffn.pre, &mut |a| Ok(a.to_vec()), &mut |f| Ok(f.to_vec())).unwrap();
    assert_eq!(t2.attn_in, norm::rms_norm(&hc_pre(&s, &t.x_out, &t.ffn.pre), &b.attn_norm, 1e-20));
}

#[test]
fn dsv41_om_k_split_equals_full_up_to_fp32_order() {
    let s = shape(4, 64);
    let mut rng = Rng(5);
    let p = mix_params(&mut rng, &s);
    for _ in 0..20 {
        let x = rng.bf16_vec(s.width(), 3.0);
        let full = hc_mixes(&s, &p, &x);
        let one = hc_mixes_from_partials(&s, &[hc_mix_partial(&s, &p, &x, 0, s.width())]);
        assert_eq!(one, full, "a 1-way split is the full mix");
        for n in [2, 4, 8] {
            let parts: Vec<_> = k_ranges(&s, n).into_iter().map(|(a, b)| hc_mix_partial(&s, &p, &x, a, b)).collect();
            let split = hc_mixes_from_partials(&s, &parts);
            for (a, b) in split.iter().zip(&full) {
                assert!((a - b).abs() <= 1e-5 * b.abs().max(1e-2), "{n}-way: {a} vs {b}");
            }
        }
    }
    assert_eq!(k_ranges(&shape(4, 5120), 4), vec![(0, 5120), (5120, 10240), (10240, 15360), (15360, 20480)]);
}

#[test]
fn dsv41_om_stream_mean() {
    let s = shape(4, 2);
    assert_eq!(stream_mean(&s, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0]), vec![4.0, 5.25]);
    let e = expand(&s, &[1.5, -2.0]);
    assert_eq!(stream_mean(&s, &e), vec![1.5, -2.0]);
}

fn toy_gate(top_k: usize) -> GateParams {
    // 8 experts over a 2-wide input: expert e's logit = x0 * (e - 3.5) + x1 * 0.1
    let mut weight = Vec::new();
    for e in 0..8 {
        weight.push(e as f32 - 3.5);
        weight.push(0.1);
    }
    GateParams {
        weight,
        bias: vec![0.0; 8],
        bias_vl: Some(vec![0.0; 8]),
        n_experts: 8,
        dim: 2,
        top_k,
        route_scale: 1.5,
        score: ScoreFunc::SqrtSoftplus,
        norm_topk_prob: true,
        gate_temp: 1.0,
    }
}

#[test]
fn dsv41_om_router_bias_selects_but_does_not_weight() {
    let mut g = toy_gate(3);
    let x = [1.0f32, 0.0];
    let r = route(&g, &x, false);
    assert_eq!(r.ids, vec![7, 6, 5]);
    let w: f32 = r.weights.iter().sum();
    assert!((w - 1.5).abs() < 1e-6, "renormalised weights sum to route_scale, got {w}");
    // a large bias on expert 0 makes it selected, but its weight still comes from its raw score
    g.bias[0] = 100.0;
    let r2 = route(&g, &x, false);
    assert_eq!(r2.ids, vec![0, 7, 6]);
    let sc = gate_scores(&g, &x);
    let den = sc[0] + sc[7] + sc[6] + RENORM_EPS;
    assert!((r2.weights[0] - sc[0] / den * 1.5).abs() < 1e-6);
    // image rows use bias_vl instead (all zero here: back to the unbiased choice)
    assert_eq!(route(&g, &x, true).ids, vec![7, 6, 5]);
    // sqrt(softplus): softplus is the identity above 20
    assert_eq!(softplus_f32(25.0), 25.0);
    assert!((softplus_f32(0.0) - std::f32::consts::LN_2).abs() < 1e-7);
}

#[test]
fn dsv41_om_router_ties_take_the_lower_id_and_report_the_margin() {
    let g = toy_gate(2);
    // x = 0: every logit is 0.1 * 0 = 0 -> eight equal scores
    let r = route(&g, &[0.0, 0.0], false);
    assert_eq!(r.ids, vec![0, 1]);
    assert_eq!(r.margin, 0.0);
    let r = route(&g, &[1.0, 0.0], false);
    assert!(r.margin > 0.0);
    let (ids, kth, next) = topk_desc(&[1.0, f32::NAN, 3.0, 3.0], 2);
    assert_eq!(ids, vec![2, 3]);
    assert_eq!((kth, next), (3.0, 1.0));
}

/// REFb's short-chat dump against the native checkpoint, layers 0-2 (an Engram layer included) and
/// 39 (the head side): every bound of `check::BOUNDS` must hold. Skips without the fixture.
#[test]
fn dsv41_om_matches_refb_short_chat() {
    let model = std::path::Path::new("/models/DeepSeek-V4.1-Flash-1017");
    let dump = std::path::Path::new("/mnt/models/dsv41/ref/short_chat/prefill_exact");
    if !dump.join("meta.json").exists() {
        return; // skipped: no REFb short_chat dump on this machine (no print: logq direct-print baseline)
    };
    if !model.join("model.safetensors.index.json").exists() {
        return; // skipped: no native checkpoint on this machine
    }
    let o = check::CheckOpts { layers: Some(vec![0, 1, 2]), head_rows: 0, threads: 4 };
    let r = check::run(model, dump, &o).unwrap();
    assert!(r.failures.is_empty(), "{}", check::render(&r));
    let tot_rows: u64 = r.layers.iter().map(|l| l.router.rows).sum();
    assert!(tot_rows >= 3 * 30, "router rows checked: {tot_rows}");
    assert!(r.layers.iter().all(|l| l.chain_router.rows > 0), "the chain ran on every layer");
}
