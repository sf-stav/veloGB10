//! S2 unit tests — the P8 reference oracle, on a REDUCED-shape config so CI-style reruns are cheap
//! (the exact same code paths as the full-size `--probe-dspark-synth`). Covers workdoc §7.2–7.5:
//! determinism, wiring (anti-empty-compare), incremental==batch tap projection, structure, and
//! piecewise availability. The full-shape end-to-end lives in the probe.

use gb10_inference::dspark::oracle::{
    truncate, DsparkConfig, DsparkOracle, DsparkWeights, LayerWeights, RoundCtx,
};
use gb10_inference::dspark::synth::{SynthRng, SyntheticTables};

/// A reduced but structurally faithful config: GQA 4Q/2KV × hd32, hidden 128 (= 4×32), 2 layers,
/// vocab 1024, rank 16 — everything else (block 7, MASK id, YaRN) identical to the P8 anatomy.
fn reduced_cfg() -> DsparkConfig {
    DsparkConfig {
        hidden: 128,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 32,
        inter: 256,
        vocab: 4096,
        n_layers: 2,
        block: 7,
        mask_token_id: 248_077,
        rope_theta: 1e7,
        rope_factor: 32.0,
        beta_fast: 32,
        beta_slow: 1,
        orig_ctx: 8192,
        max_positions: 262_144,
        rms_eps: 1e-6,
        markov_rank: 16,
        confidence_threshold: 0.5,
    }
}

fn reduced_weights(cfg: &DsparkConfig) -> DsparkWeights {
    let mut rng = SynthRng::new(0x1234_5678_9ABC_DEF0);
    let norm = |rng: &mut SynthRng, n: usize| -> Vec<f32> {
        (0..n).map(|_| 0.01 + 0.01 * rng.uniform()).collect()
    };
    let lin = |rng: &mut SynthRng, outn: usize, inn: usize| -> Vec<f32> {
        let scale = 1.0 / (inn as f32).sqrt();
        (0..outn * inn).map(|_| rng.normal() * scale).collect()
    };
    let (h, nkv, hd, inter, rank, vocab) = (
        cfg.hidden, cfg.num_kv_heads, cfg.head_dim, cfg.inter, cfg.markov_rank, cfg.vocab,
    );
    let mut layers = Vec::new();
    for _ in 0..cfg.n_layers {
        layers.push(LayerWeights {
            q_proj: lin(&mut rng, h, h),
            k_proj: lin(&mut rng, nkv * hd, h),
            v_proj: lin(&mut rng, nkv * hd, h),
            o_proj: lin(&mut rng, h, h),
            q_norm: norm(&mut rng, hd),
            k_norm: norm(&mut rng, hd),
            input_ln: norm(&mut rng, h),
            post_ln: norm(&mut rng, h),
            gate_proj: lin(&mut rng, inter, h),
            up_proj: lin(&mut rng, inter, h),
            down_proj: lin(&mut rng, h, inter),
        });
    }
    let markov_scale = 1.0 / (rank as f32).sqrt();
    DsparkWeights {
        layers,
        fc: lin(&mut rng, h, 5 * h),
        hidden_norm: norm(&mut rng, h),
        norm: norm(&mut rng, h),
        w1: (0..vocab * rank).map(|_| rng.normal() * markov_scale).collect(),
        w2: (0..vocab * rank).map(|_| rng.normal() * markov_scale).collect(),
        confidence_w: lin(&mut rng, 1, h + rank),
        confidence_b: 0.0,
    }
}

fn reduced_oracle() -> DsparkOracle {
    let cfg = reduced_cfg();
    let w = reduced_weights(&cfg);
    DsparkOracle::from_weights(cfg, w).expect("reduced oracle")
}

fn synthetic_ctx(cfg: &DsparkConfig, l: usize) -> RoundCtx {
    let synth = SyntheticTables::new(0xAAAA_BBBB_CCCC_DDDD);
    let tap_dim = 5 * cfg.hidden;
    let scale = 1.0 / (tap_dim as f32).sqrt();
    let mut taps = Vec::with_capacity(l * tap_dim);
    for i in 0..l {
        taps.extend_from_slice(&synth.row(2, i as u32, tap_dim, scale));
    }
    RoundCtx { tap_hiddens: taps, anchor: 0, confidence_threshold: 0.5 }
}

// ---------------------------------------------------------------------------
// The reconciliation line (workdoc §3): 62 tensors, exactly 1,359,284,737 params.
// ---------------------------------------------------------------------------

#[test]
fn inventory_is_62_tensors() {
    assert_eq!(gb10_inference::dspark::inventory().len(), 62);
}

#[test]
fn params_reconcile_to_published_count() {
    assert_eq!(gb10_inference::dspark::reconcile_params(), 1_359_284_737);
}

// ---------------------------------------------------------------------------
// §7.2 determinism — two run_rounds bit-identical.
// ---------------------------------------------------------------------------

#[test]
fn determinism_two_rounds_bit_identical() {
    let o = reduced_oracle();
    let ctx = synthetic_ctx(&o.cfg, 8);
    let a = o.run_round(&ctx);
    let b = o.run_round(&ctx);
    assert_eq!(a.logits0, b.logits0);
    assert_eq!(a.h, b.h);
    assert_eq!(a.tokens, b.tokens);
    assert_eq!(a.latents, b.latents);
    assert_eq!(a.p, b.p);
    assert_eq!(a.survival, b.survival);
    assert_eq!(a.k_verify, b.k_verify);
}

// ---------------------------------------------------------------------------
// §7.3 wiring — flip one weight → outputs MUST change (anti-empty-compare).
// (The workdoc's own parenthetical "flip one weight" alternative: a single 1-ulp perturb is below
// f32 rounding once it passes through the per-head RMSNorm — recorded in the S2 report R4. A sign
// flip is a single-weight perturbation that provably propagates, so it is the hard gate.)
// ---------------------------------------------------------------------------

#[test]
fn wiring_flip_one_weight_changes_outputs() {
    let cfg = reduced_cfg();
    let w = reduced_weights(&cfg);
    let o = DsparkOracle::from_weights(cfg.clone(), w.clone()).unwrap();
    let ctx = synthetic_ctx(&cfg, 8);
    let a = o.run_round(&ctx);

    let mut wp = w;
    // flip the sign of the first NONZERO q_proj weight of layer 0 (a sign flip of 0 is a no-op).
    let q0 = wp.layers[0].q_proj.as_mut_slice();
    let idx = q0.iter().position(|&x| x != 0.0).expect("a nonzero q_proj weight exists");
    q0[idx] = -q0[idx];
    let op = DsparkOracle::from_weights(cfg, wp).unwrap();
    let b = op.run_round(&ctx);

    assert!(
        a.logits0 != b.logits0 || a.h != b.h || a.tokens != b.tokens,
        "flipping one q_proj weight left outputs bit-identical — the oracle is not reading its weights"
    );
}

// ---------------------------------------------------------------------------
// §7.4 (DECISION B) incremental == batch tap projection.
// ---------------------------------------------------------------------------

#[test]
fn incremental_equals_batch_tap_projection() {
    let o = reduced_oracle();
    let cfg = o.cfg.clone();
    let ctx = synthetic_ctx(&cfg, 8);
    let tap_dim = 5 * cfg.hidden;
    let batch = o.tap_project(&ctx.tap_hiddens, 8);
    let mut inc = Vec::with_capacity(8 * cfg.hidden);
    for i in 0..8 {
        inc.extend_from_slice(&o.tap_project(&ctx.tap_hiddens[i * tap_dim..(i + 1) * tap_dim], 1));
    }
    assert_eq!(batch, inc);
}

// ---------------------------------------------------------------------------
// §7.4 structure: finite logits, distinct latents, confidences in (0,1),
// survival monotone, k_verify in [1..8] and non-degenerate over a threshold sweep.
// ---------------------------------------------------------------------------

#[test]
fn structure_checks() {
    let o = reduced_oracle();
    let ctx = synthetic_ctx(&o.cfg, 8);
    let a = o.run_round(&ctx);

    assert!(a.logits0.iter().all(|x| x.is_finite()), "logits must be finite");

    // Not all identical (anti-degenerate-chain). Pairwise distinctness of the chain is asserted
    // directly in `markov_chain_latents_distinct_when_logits_distinct` — a random synthetic
    // backbone collapses the 6 MASK positions (see the S2 report R4), so full pairwise distinctness
    // is not a property of the synthetic full-round output.
    let rank = o.cfg.markov_rank;
    let mut set: std::collections::BTreeSet<Vec<u32>> = std::collections::BTreeSet::new();
    for k in 0..6 {
        set.insert(a.latents[k * rank..(k + 1) * rank].iter().map(|x| x.to_bits()).collect());
    }
    assert!(set.len() >= 2, "Markov latents must not all be identical");

    assert!(a.p.iter().all(|&x| x > 0.0 && x < 1.0), "confidences in (0,1)");
    assert!(a.survival.windows(2).all(|w| w[0] >= w[1]), "survival monotone non-increasing");

    let mut kvs = std::collections::BTreeSet::new();
    let mut thr = 0.1f32;
    while thr < 0.91 {
        let kv = truncate(&a.survival, thr);
        assert!((1..=8).contains(&kv), "k_verify {kv} not in [1..8] at τ={thr}");
        kvs.insert(kv);
        thr += 0.1;
    }
    assert!(kvs.len() > 1, "k_verify sweep must be non-degenerate, got {kvs:?}");
}

// ---------------------------------------------------------------------------
// The Markov chain produces pairwise-distinct latents when given distinct per-row logits0
// (the actual invariant behind the workdoc's "distinct Markov latents" — a degenerate random
// backbone collapses the 6 MASK positions, so the chain's distinctness is proven directly here).
// ---------------------------------------------------------------------------

#[test]
fn markov_chain_latents_distinct_when_logits_distinct() {
    let o = reduced_oracle();
    let vocab = o.cfg.vocab;
    // Hand-crafted logits0: row k peaks hard at token k (gap 100 ≫ the O(1) Markov bias), so the
    // chain returns d[k] = k, all pairwise distinct.
    let mut logits0 = vec![0.0f32; 7 * vocab];
    for k in 0..7 {
        logits0[k * vocab + k] = 100.0;
    }
    // S10 semantics: the chain seeds `prev` from the ANCHOR TOKEN (argmax of row 0 = 0 here);
    // the latents W1[d[k-1]] stay anchor-independent for this hand-crafted logits0.
    let mo = o.markov_chain(&logits0, 0);
    let rank = o.cfg.markov_rank;
    for k1 in 0..6 {
        for k2 in (k1 + 1)..6 {
            assert_ne!(
                mo.latents[k1 * rank..(k1 + 1) * rank],
                mo.latents[k2 * rank..(k2 + 1) * rank],
                "Markov latents must be pairwise distinct when logits0 rows are distinct"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// §7.5 piecewise — every piece callable standalone; composition == run_round.
// ---------------------------------------------------------------------------

#[test]
fn piecewise_composition_equals_run_round() {
    let o = reduced_oracle();
    let cfg = o.cfg.clone();
    let ctx = synthetic_ctx(&cfg, 8);
    let a = o.run_round(&ctx);

    let th = o.tap_project(&ctx.tap_hiddens, 8);
    let kv = o.draft_kv_write(&th, 8);
    let mut blk = vec![0u32];
    for _ in 1..7 {
        blk.push(cfg.mask_token_id);
    }
    let synth = SyntheticTables::new(gb10_inference::dspark::SYNTH_EMBED_HEAD_SEED);
    let escale = 1.0f32 / (cfg.hidden as f32).sqrt();
    let mut emb = Vec::with_capacity(7 * cfg.hidden);
    for &t in &blk {
        emb.extend_from_slice(&synth.row(SyntheticTables::TABLE_EMBED, t, cfg.hidden, escale));
    }
    let h = o.block_forward(&emb, &kv, 8);
    let logits0 = o.lm_head(&h, 7);
    // S10 semantics: anchor = the input id at query offset 0 (reference chain).
    let mo = o.markov_chain(&logits0, blk[0]);
    let co = o.confidence(&h, &mo.latents, 0.5);

    assert_eq!(h, a.h);
    assert_eq!(logits0, a.logits0);
    assert_eq!(mo.tokens, a.tokens);
    assert_eq!(mo.latents, a.latents);
    assert_eq!(co.p, a.p);
    assert_eq!(co.survival, a.survival);
    assert_eq!(co.k_verify, a.k_verify);
}
