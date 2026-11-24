//! Tests of the attention oracle (leg OA).
//!
//! * Unit tests (always run): the number formats against brute force, the reference constants
//!   (YaRN frequencies = torch's bits), the window / top-k / candidate rules, and a synthetic tiny
//!   model (random weights, every layer class) proving the state handling: a chunked prefill at odd
//!   boundaries and a step-by-step decode both reproduce the single-shot prefill BITWISE.
//! * Fixture tests (`#[ignore]`, one per layer class; need REFb's short_chat fixture and the native
//!   checkpoint on this box, skip cleanly otherwise):
//!   `cargo test --release --lib dsv41::cpu::attn::tests::fixture -- --ignored --test-threads 1`

use super::fixture::{probe, Cmp};
use super::layer::*;
use super::num::*;
use super::rope::{Rope, Yarn};
use super::weights::{AttnWeights, CompressorW, IndexerW};

// ------------------------------------------------------------------------------------------ formats

/// Every finite non-negative E4M3FN value, ascending, with its code.
fn e4m3_table() -> Vec<(f32, u8)> {
    let mut v: Vec<(f32, u8)> = (0u8..=0x7E).map(|c| (e4m3_decode(c), c)).collect();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    v
}

#[test]
fn dsv41_oa_e4m3_round_matches_brute_force_rne() {
    let t = e4m3_table();
    for (v, _) in &t {
        assert_eq!(e4m3_round(*v), *v);
        assert_eq!(e4m3_round(-*v), -*v);
    }
    for w in t.windows(2) {
        let (a, ca) = w[0];
        let (b, cb) = w[1];
        let mid = (a + b) / 2.0;
        // ties go to the even code (mantissa lsb 0)
        let even = if ca & 1 == 0 { a } else { b };
        assert_eq!(e4m3_round(mid), even, "tie between {a} and {b}");
        assert_eq!(e4m3_round(f32::from_bits(mid.to_bits() - 1)), a);
        assert_eq!(e4m3_round(f32::from_bits(mid.to_bits() + 1)), b);
        let _ = cb;
    }
    assert_eq!(e4m3_round(448.0), 448.0);
    assert_eq!(e4m3_round(2f32.powi(-10)), 0.0); // half the smallest subnormal: tie to zero (even)
}

#[test]
fn dsv41_oa_e2m1_ties_and_bf16_rne() {
    let cases = [(0.25, 0.0), (0.26, 0.5), (0.75, 1.0), (1.25, 1.0), (1.75, 2.0), (2.5, 2.0), (3.5, 4.0), (5.0, 4.0), (5.01, 6.0), (6.0, 6.0)];
    for (x, y) in cases {
        assert_eq!(e2m1_round(x), y, "{x}");
        assert_eq!(e2m1_round(-x), -y, "-{x}");
    }
    assert!(e2m1_round(-0.1).is_sign_negative());
    // bf16 RNE: 1 + 2^-8 is a tie between 1 and 1 + 2^-7 -> even (1); 1 + 3 * 2^-8 -> 1 + 2^-6
    assert_eq!(bf16(1.0 + 2f32.powi(-8)), 1.0);
    assert_eq!(bf16(1.0 + 3.0 * 2f32.powi(-8)), 1.0 + 2f32.powi(-6));
    assert_eq!(pow2_ceil(1.0), 1.0);
    assert_eq!(pow2_ceil(1.000_000_1), 2.0);
    assert_eq!(pow2_ceil(0.75), 1.0);
}

#[test]
fn dsv41_oa_act_quant_scale_and_floor() {
    // a group whose amax is 448 * 2^-3 gets scale 2^-3 exactly and keeps its max element
    let mut x = vec![0.0f32; 32];
    x[0] = 56.0;
    x[1] = 0.3;
    let mut q = vec![0f32; 32];
    let s = act_quant_group(&x, &mut q);
    assert_eq!(s, 0.125);
    assert_eq!(q[0], 448.0);
    // a zero group gets the floor's scale, not zero
    let z = vec![0.0f32; 32];
    let s0 = act_quant_group(&z, &mut q);
    assert_eq!(s0, pow2_ceil(f32_of(1e-4) * f32_of(1.0 / 448.0)));
    assert!(s0 > 0.0 && q.iter().all(|&v| v == 0.0));
    // the inplace variant writes bf16(e4m3 * s)
    let mut y = x.clone();
    act_quant_inplace(&mut y, 32);
    assert_eq!(y[0], 56.0);
    assert_eq!(y[1], bf16(e4m3_round(0.3 / 0.125) * 0.125));
    // compressed-KV FP4: the scale is amax / 6 rounded to E4M3
    let mut g = vec![0.0f32; 16];
    g[0] = 7.0;
    g[5] = -1.1;
    let s = fp4_e4m3_scale(7.0);
    assert_eq!(s, e4m3_round(7.0 / 6.0));
    fp4_quant_e4m3_inplace(&mut g, 16);
    assert_eq!(g[0], bf16(e2m1_round((7.0f32 / s).min(6.0)) * s));
    assert_eq!(g[5], bf16(e2m1_round(-1.1f32 / s) * s));
}

#[test]
fn dsv41_oa_rope_frequencies_equal_torch_bits() {
    // model.py precompute_freqs_cis(64, .., 65536, 160000, 16, 32, 1) and (.., 0, 10000, ..): torch 2.11
    // CPU bits, except YaRN entry 24, where torch on CUDA (REFb's device) multiplies by the fp32
    // reciprocal of the ramp span: 0x37a3d708 (CPU 0x37a3d70c; leg KI's measurement)
    let yarn_bits: [u32; 32] = [
        1065353216, 1060112954, 1056054303, 1051098369, 1046804782, 1042117749, 1037601972, 1033169251, 1028443341, 1024251131, 1019326489,
        1015361742, 1010249151, 1006366088, 1001209182, 997081655, 991197921, 985076960, 979287478, 973418702, 966505391, 959469944,
        952012877, 943627103, 0x37a3d708, 917783504, 913828774, 908753896, 904558591, 899759068, 895336240, 890797133,
    ];
    let plain_bits: [u32; 32] = [
        1065353216, 1061157137, 1058010522, 1054337179, 1050798235, 1047712803, 1043732615, 1040747895, 1036831949, 1033475085, 1030116802,
        1026341397, 1023510243, 1019364175, 1016180025, 1012562930, 1008981770, 1005959600, 1001931932, 998911607, 995049016, 991652109,
        988353832, 984533390, 981668463, 977573004, 974350870, 970790693, 967166816, 964208661, 960132947, 957076594,
    ];
    let y = Rope::new(64, 160000.0, Some(Yarn { original_seq_len: 65536.0, factor: 16.0, beta_fast: 32.0, beta_slow: 1.0 }));
    let p = Rope::new(64, 10000.0, None);
    let yb: Vec<u32> = y.freqs.iter().map(|f| f.to_bits()).collect();
    let pb: Vec<u32> = p.freqs.iter().map(|f| f.to_bits()).collect();
    assert_eq!(yb, yarn_bits.to_vec(), "the YaRN table equals torch-on-CUDA's");
    assert_eq!(pb, plain_bits.to_vec(), "the plain table equals torch's");
    // a rotation and its inverse return the bf16 input (to bf16 rounding)
    let mut x: Vec<f32> = (0..64).map(|i| bf16(((i * 37 % 11) as f32 - 5.0) * 0.37)).collect();
    let x0 = x.clone();
    y.apply(&mut x, 4096, false);
    y.apply(&mut x, 4096, true);
    let c = Cmp::of(&x, &x0);
    assert!(c.rel_l2 < 1e-2, "{}", c.line());
}

#[test]
fn dsv41_oa_block_dot_rounds_toward_zero_and_rotation_is_two_fmas() {
    // 1 + 2^-24 + 2^-30 needs 31 bits: RNE gives 1 + 2^-23, the MMA's RZ gives 1
    let x = 1.0 + 2f64.powi(-24) + 2f64.powi(-30);
    assert_eq!(x as f32, 1.0 + 2f32.powi(-23));
    assert_eq!(f32_rz(x), 1.0);
    assert_eq!(f32_rz(-x), -1.0);
    assert_eq!(f32_rz(1.5), 1.5);
    // torch-CUDA complex multiply: re = fma(a, c, -(b s)) (one rounding of a c), not a c - b s
    let r = Rope { dim: 2, freqs: vec![0.0] };
    let (a, b, c, s) = (1.0f32 + 2f32.powi(-12), 1.0f32, 1.0f32 - 2f32.powi(-12), 2f32.powi(-24));
    let mut v = [a, b];
    r.apply_cis(&mut v, &[(c, s)], false);
    assert_eq!(v[0], bf16(a.mul_add(c, -(b * s))));
    assert_eq!(v[1], bf16(b.mul_add(c, a * s)));
}

#[test]
fn dsv41_oa_window_lists_follow_the_reference() {
    // prefill: t = 3, P = 30, window 8 -> [0..3, -1 x 4] (width min(P, 8))
    assert_eq!(window_positions(8, 3, Order::Prefill { p_total: 30 }), vec![Some(0), Some(1), Some(2), Some(3), None, None, None, None]);
    assert_eq!(window_positions(8, 3, Order::Prefill { p_total: 5 }), vec![Some(0), Some(1), Some(2), Some(3), None]);
    assert_eq!(window_positions(8, 20, Order::Prefill { p_total: 30 }), (13..=20).map(Some).collect::<Vec<_>>());
    // decode: start_pos 3 -> slots 4..7 (unwritten, -1) then 0..3
    assert_eq!(window_positions(8, 3, Order::Decode), vec![None, None, None, None, Some(0), Some(1), Some(2), Some(3)]);
    // decode with a full ring: oldest first = ascending positions
    assert_eq!(window_positions(8, 20, Order::Decode), (13..=20).map(Some).collect::<Vec<_>>());
}

#[test]
fn dsv41_oa_topk_and_candidate_rules() {
    // ties go to the lower position; output sorted; -1 padding when fewer are reachable
    let s = [1.0, 3.0, 2.0, 3.0, 2.0, 0.5];
    assert_eq!(select_topk(&s, 3), vec![1, 2, 3]);
    assert_eq!(select_topk(&s, 4), vec![1, 2, 3, 4]);
    assert_eq!(select_topk(&s[..2], 4), vec![0, 1, -1, -1]);
    // candidate blocks of 2, keep 2: block maxima [3, 3, 2(+inf pinned: newest)] -> keep blocks 0 and 2?
    // reachable 6 -> 3 blocks; the newest block (2) is pinned to +inf, then the best of {0: 3, 1: 3} by
    // the lower index -> blocks {0, 2}
    assert_eq!(select_candidate_blocks(&s, 6, 2, 2), vec![true, false, true]);
    // fewer blocks than the budget: every reachable block kept
    assert_eq!(select_candidate_blocks(&s[..3], 3, 4, 2), vec![true, true]);
}

#[test]
fn dsv41_oa_sparse_attention_basic_cases() {
    let kv: Vec<f32> = (0..8).map(|i| bf16(i as f32 * 0.25 - 1.0)).collect();
    let q = vec![0.0f32; 8];
    // one valid key, sink -inf: the output is that key's value
    let out = sparse_attn_head(&q, &[Some(&kv[..]), None], f32::NEG_INFINITY, 1.0);
    assert_eq!(out, kv);
    // no valid key: zeros (the sink makes the denominator infinite)
    let out = sparse_attn_head(&q, &[None, None], 0.0, 1.0);
    assert!(out.iter().all(|&v| v == 0.0));
    // two equal-score keys and a sink of equal weight: (a + b) / 3
    let a = vec![1.0f32; 8];
    let b = vec![2.0f32; 8];
    let out = sparse_attn_head(&q, &[Some(&a[..]), Some(&b[..])], 0.0, 1.0);
    assert!(out.iter().all(|&v| v == bf16(3.0 / 3.0)));
}

// ------------------------------------------------------------------------------------------ tiny model

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unif(&mut self) -> f32 { (self.next() as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 }
    fn bf(&mut self, scale: f32) -> f32 { bf16(self.unif() * scale) }
}

fn tiny_cfg() -> OracleCfg {
    // 6 layers: SWA-only, ratio-2 source + index source, ratio-2 reuser, ratio-1 source + candidate
    // source + index source, ratio-1 reuser, ratio-1 index source using candidates
    let yarn = Yarn { original_seq_len: 64.0, factor: 4.0, beta_fast: 32.0, beta_slow: 1.0 };
    OracleCfg {
        layers: 6,
        dim: 64,
        heads: 2,
        head_dim: 64,
        rope_dim: 16,
        q_lora: 64,
        o_groups: 2,
        o_lora: 32,
        window: 8,
        index_heads: 2,
        index_head_dim: 32,
        index_topk: 4,
        cand_source: Some(3),
        cand_topk_blocks: 2,
        cand_block: 2,
        eps: 1e-20,
        ratios: vec![0, 2, 2, 1, 1, 1],
        kv_sources: vec![1, 3],
        index_sources: vec![1, 3, 5],
        rope_plain: Rope::new(16, 10000.0, None),
        rope_yarn: Rope::new(16, 160000.0, Some(yarn)),
    }
}

fn fp8_rand(r: &mut Lcg, n: usize, k: usize) -> Fp8Lin {
    let w: Vec<u8> = (0..n * k).map(|_| { let c = (r.next() % 0x6F) as u8; c | ((r.next() & 1) as u8) << 7 }).collect();
    let s: Vec<u8> = (0..n.div_ceil(32) * (k / 32)).map(|_| 120 + (r.next() % 3) as u8).collect();
    Fp8Lin::from_raw(n, k, &w, &s)
}

fn dense_rand(r: &mut Lcg, n: usize, k: usize, scale: f32) -> DenseLin { DenseLin { n, k, w: (0..n * k).map(|_| r.bf(scale)).collect() } }

fn tiny_weights(c: &OracleCfg, l: usize, r: &mut Lcg) -> AttnWeights {
    let (d, hd) = (c.dim, c.head_dim);
    let ratio = c.ratio(l);
    let norm = |r: &mut Lcg, n: usize| -> Vec<f32> { (0..n).map(|_| bf16(1.0 + 0.2 * r.unif())).collect() };
    AttnWeights {
        layer: l,
        wq_a: fp8_rand(r, c.q_lora, d),
        q_norm: norm(r, c.q_lora),
        wq_b: fp8_rand(r, c.heads * hd, c.q_lora),
        wkv: fp8_rand(r, hd, d),
        kv_norm: norm(r, hd),
        wo_a: (0..c.o_groups * c.o_lora * (c.heads * hd / c.o_groups)).map(|_| r.bf(0.1)).collect(),
        wo_b: fp8_rand(r, d, c.o_groups * c.o_lora),
        sink: (0..c.heads).map(|_| r.unif()).collect(),
        comp: if c.is_kv_source(l) {
            Some(CompressorW {
                ratio,
                wkv: dense_rand(r, hd, d, 0.2),
                wgate: if ratio > 1 { Some(dense_rand(r, hd, d, 0.2)) } else { None },
                norm: norm(r, hd),
            })
        } else {
            None
        },
        idx: if c.is_index_source(l) {
            let owner = c.is_kv_source(l);
            Some(IndexerW {
                wq_b: fp8_rand(r, c.index_heads * c.index_head_dim, c.q_lora),
                weights_proj: dense_rand(r, c.index_heads, d, 0.3),
                wk: if owner { Some(dense_rand(r, c.index_head_dim, hd, 0.3)) } else { None },
                k_norm: if owner { Some(norm(r, c.index_head_dim)) } else { None },
            })
        } else {
            None
        },
    }
}

/// The tiny model's per-layer attention inputs (independent random rows per layer: the oracle takes
/// `attn_in` as given, exactly as it takes REFb's dumped rows).
fn tiny_inputs(c: &OracleCfg, p: usize) -> (Vec<AttnWeights>, Vec<Vec<f32>>) {
    let mut r = Lcg(0x5EED_0A);
    let w: Vec<AttnWeights> = (0..c.layers).map(|l| tiny_weights(c, l, &mut r)).collect();
    let x: Vec<Vec<f32>> = (0..c.layers).map(|_| (0..p * c.dim).map(|_| r.bf(2.0)).collect()).collect();
    (w, x)
}

/// Single-shot prefill of `p` rows, every row a query; returns attn_out per layer and the oracle.
fn single_shot(c: &OracleCfg, w: &[AttnWeights], x: &[Vec<f32>], p: usize) -> (Oracle, Vec<Vec<f32>>) {
    let mut o = Oracle::new(c.clone());
    let q: Vec<usize> = (0..p).collect();
    let outs = (0..c.layers)
        .map(|l| o.forward_layer(&w[l], LayerIn { layer: l, s0: 0, x: &x[l], queries: &q, order: Order::Prefill { p_total: p }, topk_override: None }).unwrap().attn_out)
        .collect();
    (o, outs)
}

#[test]
fn dsv41_oa_tiny_chunked_prefill_is_bitwise_single_shot() {
    let _g = STUDY.lock().unwrap_or_else(|e| e.into_inner());
    let c = tiny_cfg();
    let p = 41;
    let (w, x) = tiny_inputs(&c, p);
    let (o1, ref_out) = single_shot(&c, &w, &x, p);
    // the selections bind: top-4 of up to 20 (ratio 2) / 40 (ratio 1) positions, candidates at layer 5
    let t = &o1.topk[&1][&40];
    assert_eq!(t.pos.len(), 4);
    assert!(o1.cand[&40].iter().filter(|&&k| k).count() == 2, "the candidate budget binds at row 40");
    for cuts in [vec![0, 13, 27, 41], vec![0, 1, 2, 7, 8, 39, 40, 41], vec![0, 20, 41]] {
        let mut o = Oracle::new(c.clone());
        let mut outs = vec![Vec::new(); c.layers];
        for seg in cuts.windows(2) {
            let (s0, s1) = (seg[0], seg[1]);
            let q: Vec<usize> = (s0..s1).collect();
            for l in 0..c.layers {
                let y = o
                    .forward_layer(&w[l], LayerIn { layer: l, s0, x: &x[l][s0 * c.dim..s1 * c.dim], queries: &q, order: Order::Prefill { p_total: p }, topk_override: None })
                    .unwrap();
                outs[l].extend(y.attn_out);
            }
        }
        for l in 0..c.layers {
            assert_eq!(outs[l], ref_out[l], "layer {l}, cuts {cuts:?}: chunked prefill differs from single-shot");
            assert_eq!(o.window_ring(l), o1.window_ring(l), "layer {l}: ring");
        }
        for (s, st) in &o1.src {
            assert!(st.same_live_state(&o.src[s], c.head_dim), "source {s}, cuts {cuts:?}");
        }
    }
}

#[test]
fn dsv41_oa_tiny_decode_steps_are_bitwise_prefill_rows() {
    let _g = STUDY.lock().unwrap_or_else(|e| e.into_inner());
    // every decode step's list holds the same valid entries in the same order inside one softmax
    // block (window 8 + top-4 < 64), so a decode chain reproduces the single-shot rows bitwise
    let c = tiny_cfg();
    let p = 41;
    let (w, x) = tiny_inputs(&c, p);
    let (o1, ref_out) = single_shot(&c, &w, &x, p);
    let mut o = Oracle::new(c.clone());
    for l in 0..c.layers {
        o.forward_layer(&w[l], LayerIn { layer: l, s0: 0, x: &x[l][..c.dim], queries: &[0], order: Order::Prefill { p_total: 1 }, topk_override: None }).unwrap();
    }
    for t in 1..p {
        for l in 0..c.layers {
            let y = o
                .forward_layer(&w[l], LayerIn { layer: l, s0: t, x: &x[l][t * c.dim..(t + 1) * c.dim], queries: &[t], order: Order::Decode, topk_override: None })
                .unwrap();
            assert_eq!(y.attn_out, ref_out[l][t * c.dim..(t + 1) * c.dim], "layer {l}, decode step {t}");
        }
    }
    for (s, st) in &o1.src {
        assert!(st.same_live_state(&o.src[s], c.head_dim), "source {s}");
    }
    // the open group (position 40) is held in slot 0 by both
    assert!(o.src[&1].kv_state[..c.head_dim].iter().any(|&v| v != 0.0));
}

#[test]
fn dsv41_oa_topk_override_and_wo_b_split() {
    let _g = STUDY.lock().unwrap_or_else(|e| e.into_inner());
    let c = tiny_cfg();
    let p = 24;
    let (w, x) = tiny_inputs(&c, p);
    let (_, ref_out) = single_shot(&c, &w, &x, p);
    // overriding layer 1's selection with its own selection changes nothing
    let mut o = Oracle::new(c.clone());
    let q: Vec<usize> = (0..p).collect();
    o.forward_layer(&w[0], LayerIn { layer: 0, s0: 0, x: &x[0], queries: &q, order: Order::Prefill { p_total: p }, topk_override: None }).unwrap();
    let own = o.forward_layer(&w[1], LayerIn { layer: 1, s0: 0, x: &x[1], queries: &q, order: Order::Prefill { p_total: p }, topk_override: None }).unwrap();
    let ov: std::collections::BTreeMap<usize, Vec<i32>> = q.iter().zip(&own.topk).map(|(&r, t)| (r, t.pos.clone())).collect();
    let mut o2 = Oracle::new(c.clone());
    o2.forward_layer(&w[0], LayerIn { layer: 0, s0: 0, x: &x[0], queries: &q, order: Order::Prefill { p_total: p }, topk_override: None }).unwrap();
    let y = o2.forward_layer(&w[1], LayerIn { layer: 1, s0: 0, x: &x[1], queries: &q, order: Order::Prefill { p_total: p }, topk_override: Some(&ov) }).unwrap();
    assert_eq!(y.attn_out, ref_out[1]);
    // wo_b with its input split over 2 ranks: fp32 partials in rank order, within fp32 summation noise
    let s = wo_b_split(&w[1].wo_b, &y.wo_b_in, p, 2);
    let cmp = Cmp::of(&s, &y.attn_out);
    assert!(cmp.rel_l2 < 1e-2 && cmp.equal_frac() > 0.9, "{}", cmp.line());
}

// ------------------------------------------------------------------------------------------ fixtures

/// Test output to stderr (shown with --nocapture).
fn note(s: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "{s}");
}

const NATIVE: &str = "/models/DeepSeek-V4.1-Flash-1017";
const FIXTURES: [&str; 1] = ["/mnt/models/dsv41/ref/short_chat"];

/// Run the probe on one layer set of the short_chat fixture; skip (pass) when this box lacks the
/// fixture or the checkpoint.
fn fixture_layers(layers: &[usize]) {
    let Some(f) = FIXTURES.iter().find(|f| std::path::Path::new(f).join("prefill_exact/meta.json").exists()) else {
        note("skip: no REFb short_chat fixture on this box");
        return;
    };
    if !std::path::Path::new(NATIVE).join("model.safetensors.index.json").exists() {
        note(&format!("skip: no native checkpoint at {NATIVE}"));
        return;
    }
    let _g = STUDY.lock().unwrap_or_else(|e| e.into_inner());
    let r = probe(std::path::Path::new(NATIVE), std::path::Path::new(f), Some(layers.to_vec())).unwrap();
    for l in &r.lines {
        note(&format!("{l}"));
    }
    assert!(r.ok, "layers {layers:?} out of tolerance against REFb");
}

/// The variant switch is process-global: the oracle tests that run it hold this lock.
static STUDY: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The tolerance study (report §2): the stated tolerances accept a valid alternative summation
/// order and reject each semantic mutation. Fixtures: every layer of short_chat, and the given
/// layers of agent_4k when it is on this box. Prints a table; asserts the two properties.
#[test]
#[ignore]
fn fixture_tolerance_study_accepts_valid_order_rejects_mutations() {
    let _g = STUDY.lock().unwrap_or_else(|e| e.into_inner());
    let n = std::path::Path::new(NATIVE);
    if !n.join("model.safetensors.index.json").exists() {
        note("skip: no native checkpoint");
        return;
    }
    let sets: Vec<(&str, Option<Vec<usize>>)> = vec![
        ("/mnt/models/dsv41/ref/short_chat", None),
        ("/mnt/models/dsv41/ref/agent_4k", Some(vec![0, 2, 3, 20, 21, 24])),
    ];
    let mut bad = Vec::new();
    for (fx, layers) in &sets {
        if !std::path::Path::new(fx).join("prefill_exact/meta.json").exists() {
            note(&format!("skip fixture {fx}: not on this box"));
            continue;
        }
        for (v, name) in variant::ALL {
            variant::set(v);
            let r = probe(n, std::path::Path::new(fx), layers.clone());
            variant::set(variant::REFERENCE);
            let r = r.unwrap();
            let fails = r.reports.iter().filter(|x| !x.ok).count();
            let w = |p: &str| r.worst(p).map_or("-".to_string(), |(a, b)| format!("{a:.2e} / {:.1}%", 100.0 * b));
            let tk: usize = r.reports.iter().filter_map(|x| x.topk.as_ref()).map(|t| t.beyond).sum();
            note(&format!("STUDY {fx} | {name:45} | layers out of tolerance {fails:2} of {:2} | attn_out {} | window_ring {} | compress_kv {} | index_k {} | top-k rows beyond ties {tk}",
                      r.reports.len(), w("attn_out"), w("window_ring"), w("compress_kv"), w("index_k")));
            for rep in r.reports.iter().filter(|x| !x.ok).take(3) {
                for l in r.lines.iter().skip_while(|l| !l.starts_with(&format!("L{:02} ", rep.layer))).take(1 + rep.points.len() + rep.topk.is_some() as usize) {
                    note(&format!("    {l}"));
                }
            }
            // fp32 index scores only reorder near ties, which the selection gate accepts: the dumps
            // hold no scores, so this mutation is invisible to them (BLIND, reported, not asserted)
            let expect = match v {
                variant::REFERENCE | variant::F32_ACC => Some(true),
                variant::FP32_INDEX_SCORE => None,
                _ => Some(false),
            };
            if let Some(e) = expect {
                if e != r.ok {
                    bad.push(format!("{fx}: {name}: verdict {} (expected {e})", r.ok));
                }
            }
        }
    }
    assert!(bad.is_empty(), "the tolerances do not separate valid from mutated: {bad:?}");
}

#[test]
#[ignore]
fn fixture_swa_only_layers_0_1() { fixture_layers(&[0, 1]); }

#[test]
#[ignore]
fn fixture_ratio2_kv_and_index_source_layer_2() { fixture_layers(&[2]); }

#[test]
#[ignore]
fn fixture_ratio2_reuser_layer_3_from_dumped_source() { fixture_layers(&[3]); }

#[test]
#[ignore]
fn fixture_ratio2_source_and_reusers_2_to_7() { fixture_layers(&[2, 3, 4, 5, 6, 7]); }

#[test]
#[ignore]
fn fixture_ratio1_kv_index_candidate_source_layer_20() { fixture_layers(&[20]); }

#[test]
#[ignore]
fn fixture_ratio1_reuser_layer_21_from_dumped_source() { fixture_layers(&[21]); }

#[test]
#[ignore]
fn fixture_ratio1_index_source_with_candidates_layer_24() { fixture_layers(&[24]); }
