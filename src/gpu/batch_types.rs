//! Public batch / decode state types: `AcceptSample`, `TreeTopo`, `KvMirror`, `BatchGpuState`, `DecodeBuffers`, penalty and
//! sample-stat structs with their goodness-of-fit helpers. Map: `mod.rs`.

use super::*;

/// Per-layer per-sequence inference state for batched decode.
/// KV caches [B, nkv, stride, hd]; conv state [B, conv_dim, k]; recurrent [B, nh, kd, vd].
/// One draft position from `bench_accept`: what the TARGET thought there, and whether the draft head
/// matched it. Only positions still on the correct prefix are recorded — see `bench_accept`.
#[derive(Clone, Copy, Debug)]
pub struct AcceptSample {
    /// 1-based position within the draft chain (1 = the first draft after the committed token).
    pub depth_idx: usize,
    /// The TARGET model's softmax probability for its own argmax. Low = the target is near-tied
    /// between tokens, i.e. the text is intrinsically unpredictable at this position.
    pub target_top1_p: f32,
    /// The target's top1-top2 LOGIT gap. A tiny margin means the argmax is a coin flip between two
    /// tokens, and no draft head can be expected to call it.
    pub target_margin: f32,
    /// Did the draft head propose the target's argmax? (target argmax ∈ head top-1)
    pub accepted: bool,
    /// Is the target's argmax within the head's top-2 candidates? A fork offering top-2 at this
    /// position would rescue it. (top-2 coverage ⊇ top-1 acceptance.)
    pub covered_top2: bool,
    /// ... within the head's top-3.
    pub covered_top3: bool,
}

/// A planted tree topology for the verify (Step-2.9 byte gates). All vectors are length-n (n columns),
/// except `path` which is n*MAX_VERIFY and `winsrc` which is n*conv_kernel.
pub struct TreeTopo {
    pub rope: Vec<i32>,      // logical position (tree depth) per column — governs pc, pos_start
    pub kv_pos: Vec<i32>,    // KV-cache write slot per column (distinct; DFS index + pos_start)
    pub parent: Vec<i32>,    // GDN scan parent (DFS parent; parent[0] = -1)
    pub path: Vec<u8>,       // rank->slot: path[b*MAX_VERIFY + d] = KV-slot offset of b's d-th ancestor
    pub winsrc: Vec<i32>,    // conv window sources per column (ancestor-path window)
    pub slot: Option<Vec<i32>>, // FOREST: per-column lane slot. None => uniform `slot` arg (a single
                                // tree/chain). Some(..) packs multiple lanes' chains into one verify.
    pub col_pos_start: Option<Vec<i32>>, // FOREST: per-column prefix boundary (lane committed length).
                                // None => attention uses pos[0] (single lane / tree, byte-identical).
}

/// E2 Fix 2: per-(full-attn layer, slot) bf16 dequant mirror of the packed KV cache (q4, TQ, and
/// k8v4 — under k8v4 the mirror pair is dequant_kv_k8v4 (K) + dequant_kv_q4 (V) under one
/// watermark). Lazily allocated on the slot's SECOND prefill window (single-window prompts pay
/// nothing); rows [0, up_to) are valid bf16 dequants of the packed cache (head-stride = the mirror
/// budget), advanced incrementally per window instead of the old full-prefix re-dequant per window
/// (the O(L²) TTFT cost).
pub struct KvMirror {
    pub k: B,            // [nkv][budget][hd] bf16
    pub v: B,
    pub up_to: usize,    // watermark: rows [0, up_to) are valid
}

pub struct BatchGpuState {
    pub k_cache: Vec<Option<B>>,
    pub v_cache: Vec<Option<B>>,
    pub conv_state: Vec<Option<S>>,
    pub s_state: Vec<Option<S>>,
    /// [layer][slot] packed-KV dequant mirrors (q4/TQ/k8v4); empty Vec for GDN layers, None per
    /// slot until that slot's second prefill window. Freed on slot reuse (zero_slot_state) and
    /// full zero_state.
    pub kv_mirror: Vec<Vec<Option<KvMirror>>>,
    /// Process-global generation of THIS state's device buffers. Folds into the verify-graph key so
    /// a captured graph is never replayed against a different (dropped) state's pointers (B8 blocker B).
    pub graph_epoch: u64,
    /// Vision-embed splice context (V3): when set, `prefill_batch` overwrites the `vision_spans`
    /// rows of the embedded hidden with the merged image embeddings (`vision_embeds`, bf16, width ==
    /// text hidden 5120). Set per serving request by the scheduler before prefill; cleared after.
    pub vision_embeds: Option<B>,
    pub vision_spans: Vec<crate::vision_encoder::ImageSpan>,
    /// GDN/conv state-slot count this state was ALLOCATED with (the `state_slots` argument of
    /// `new_batch_state`). Recorded so the verify can ASSERT that its per-column checkpoint band
    /// `[ckpt_slot, ckpt_slot + n - 1]` fits: `delta_step_prefill` writes a checkpoint per column
    /// `t < n-1` at slot `ckpt_slot + t` with no bounds check, so an undersized band is a silent
    /// out-of-bounds device write into whatever allocation follows (P14: the 16-column DFlash v1
    /// verify ran on a band sized for mtp-depth 8 → 8 OOB slots per GDN layer → garbage logits,
    /// all-zero argmax, and `!!!!` output while every gate stayed green). Keep this field in sync
    /// with the allocation; the assert is the only thing standing between a sizing bug and silent
    /// heap corruption.
    pub state_slots: usize,
    /// qwen4_exp PLE recurrent state, per state slot (+1 scratch slot at index `ple_scratch_slot`):
    /// the dilated conv's last L inputs `[slot][L][hc*h]` f32, and the n-gram token ring
    /// `[slot][ngram-1]` i32 (EOS-filled = empty history). Checkpointed / rolled back with the GDN
    /// state (`copy_gdn_slot`). None on every other family.
    pub ple_conv_state: Option<S>,
    pub ple_ring: Option<cudarc::driver::CudaSlice<i32>>,
    pub ple_scratch_slot: usize,
    /// qwen4_exp QSA raw-key cache per full-attention layer, `[kv_slots][stride][hd_idx]` bf16,
    /// addressed exactly like the KV cache (None on GDN layers / when the indexer is off).
    pub qsa_keys: Vec<Option<B>>,
}

/// LM-head logits handed from `forward_decode_logits` to `forward_decode_select`. The two arms
/// differ in dtype (hy_v3's fp32-logits contract vs the bf16 family); the select half owns and
/// releases the buffer.
pub enum DecodeLogits {
    F32(S),
    Bf16(B),
}

/// Persistent device buffers for the optimized decode loop.
/// Allocated once, reused every step — eliminates all per-step allocations and large host↔device copies.
pub struct DecodeBuffers {
    pub pos_dev: cudarc::driver::CudaSlice<i32>,
    pub tokens_dev: cudarc::driver::CudaSlice<i32>,
    pub slot_ids_dev: cudarc::driver::CudaSlice<i32>,
    pub cos_dev: S,
    pub sin_dev: S,
    pub token_ids_dev: cudarc::driver::CudaSlice<i32>,
    pub temps_dev: cudarc::driver::CudaSlice<f32>,
    pub topk_dev: cudarc::driver::CudaSlice<i32>,
    pub topp_dev: cudarc::driver::CudaSlice<f32>,
    pub seeds_dev: cudarc::driver::CudaSlice<u32>,
    pub rep_pen_dev: cudarc::driver::CudaSlice<f32>,
    pub presence_dev: cudarc::driver::CudaSlice<f32>,
    pub frequency_dev: cudarc::driver::CudaSlice<f32>,
    /// LOOPFIX (2026-11-06): whole-output penalty state (vLLM semantics), one row per physical lane
    /// slot: output-token counts [slots][vocab] u32 and the prompt bitset [slots][pen_words] u32
    /// (`pen_apply_b`). Replaces the 64-token window list + device ring, which never penalized a
    /// repeated block longer than the window. `pen_add_*` stage newly committed tokens for
    /// `pen_count_add_b` (host-driven: `pen_count_add`). Zero-initialized (htod), never alloc_zeros.
    pub pen_cnt_dev: cudarc::driver::CudaSlice<u32>,
    pub pen_pbits_dev: cudarc::driver::CudaSlice<u32>,
    pub pen_add_slots_dev: cudarc::driver::CudaSlice<i32>,
    pub pen_add_toks_dev: cudarc::driver::CudaSlice<i32>,
    pub pen_slots: usize,
    /// Device-resident token-loop state (EXPERT_DEVICE_ARGMAX_LOOP_RESPONSE): the per-lane
    /// splitmix64 keys the sample graph's seed_advance_b advances. Uploaded on DIRTY steps only
    /// (admission/finish/MTP/param change); clean steps keep them device-resident.
    pub seeds_key_dev: cudarc::driver::CudaSlice<u64>,
    /// W2 (Phase 13) JSON-schema masks for the DECODE path: per-lane vocabulary bitsets
    /// (`mask_words_dev` = [max_batch][mask_words_for(vocab)] u32 little-endian, bit t = token t
    /// allowed) plus the per-lane enable flags (`mask_flags_dev` = [max_batch] i32, 0 =
    /// unconstrained). `mask_any` is the host-side gate: false on every non-schema step, so the
    /// mask kernel is not launched at all on the normal path (bit-identical behaviour).
    pub mask_words_dev: cudarc::driver::CudaSlice<u32>,
    pub mask_flags_dev: cudarc::driver::CudaSlice<i32>,
    pub mask_any: bool,
}

/// Draft-side penalty window (DraftPenalty, the MTP head's own sampling heuristic only — the
/// TARGET's penalties are whole-output since LOOPFIX, see VerifyPenalty / pen_apply_b).
pub const MAX_PEN_TOKENS: usize = 64;

/// Extra tokens a verify column can add on top of its lane's committed output (its preceding
/// drafts). Matches PEN_EXTRA_MAX in gpu_batch.cu; a verify is at most MAX_VERIFY columns wide.
pub const PEN_EXTRA: usize = MAX_VERIFY;
/// Staging capacity of `pen_count_add` (tokens per launch; larger syncs are chunked).
pub const PEN_ADD_CAP: usize = 4096;

/// Words of the per-slot prompt bitset for a vocabulary of `v` tokens.
pub fn pen_words_for(v: usize) -> usize { (v + 31) / 32 }

/// Penalty device-buffer pointers for the MTP verify path (so greedy MTP lanes keep their
/// repetition/presence/frequency penalty → no repetition). LOOPFIX: whole-output state — column c
/// penalizes with slot `slots[c]`'s committed output + prompt (DecodeBuffers::pen_cnt_dev /
/// pen_pbits_dev) plus `extra[c][0..PEN_EXTRA)` (its preceding drafts, -1 padded); slot -1 = none.
/// Passed as `Option` to `verify_forward`.
#[derive(Clone, Copy)]
pub struct VerifyPenalty {
    pub slots_ptr: u64,     // [n] i32 (-1 = this column is not penalized)
    pub extra_ptr: u64,     // [n * PEN_EXTRA] i32, -1 padded
    pub cnt_ptr: u64,       // DecodeBuffers::pen_cnt_dev
    pub pbits_ptr: u64,     // DecodeBuffers::pen_pbits_dev
    pub rep_pen_ptr: u64,   // [n] f32
    pub presence_ptr: u64,  // [n] f32
    pub freq_ptr: u64,      // [n] f32
}

/// Output of the stochastic MTP verify forward — per-position target probabilities, residual
/// resample tokens, and the bonus token (for the all-accepted case). Host reads these small scalars
/// back and runs the speculative rejection-sampling accept loop.
/// Goodness-of-fit of one empirical histogram against a reference distribution.
///
/// Total-variation distance is a *scale-dependent* statistic: its noise floor grows with the support
/// size and shrinks with the trial count, so a fixed "TVD < 0.01" bar is meaningless on its own — on
/// a 47-token nucleus at 100k trials even a perfect sampler sits around 0.009. A chi-square
/// goodness-of-fit is calibrated: `z` is comparable across temperatures, support sizes and trial
/// counts, so one threshold (|z| < 4) is honest everywhere.
#[derive(Debug, Clone, Copy)]
pub struct GofStat {
    pub tvd: f32,        // reported for intuition only — NOT the pass criterion
    pub chi2_over_df: f32,
    pub z: f32,          // Wilson-Hilferty normal score; |z| < 4 is the bar
    pub bins: usize,
    pub outside: u64,    // emissions outside the nucleus — must be 0 (they have probability 0)
}

/// Result of the stochastic-MTP distribution gate (`bench_mtp_sample`).
#[derive(Debug, Clone)]
pub struct MtpSampleStats {
    pub x_draft: u32,               // the greedy draft token whose acceptance is under test
    pub nucleus_size: usize,        // |support| of the target nucleus (after top-k ∩ top-p)
    pub p_draft_analytic: f32,      // p(x_draft) computed on the host from the same logits
    pub p_draft_device: f32,        // p(x_draft) as reported by spec_verify_b — must agree
    pub accept_rate: f32,           // empirical accept rate; converges to p(x_draft)
    pub accept_z: f32,              // binomial z of the accept rate vs p(x_draft)
    pub p_agree_mean: f32,          // S5F2 real-q: mean |device p(draft) - host p(draft)| per trial
    pub accept_e: f32,              // S5F2 real-q: E[min(1,p/q)] over the draft distribution
    pub trials: usize,
    pub bonus_trials: usize,
    pub sampler: GofStat,           // diagnostic: plain sampler vs the analytic nucleus
    pub mtp: GofStat,               // diagnostic: stochastic MTP vs the analytic nucleus
    pub bonus: GofStat,             // diagnostic: the all-accepted bonus column
    pub mtp_vs_sampler: GofStat,    // THE GATE: two-sample test, MTP against the plain sampler
    pub bonus_vs_sampler: GofStat,  // THE GATE: the bonus column against the plain sampler
}

/// Wilson-Hilferty normal score for a chi-square statistic.
fn wh_z(chi2: f64, df: usize) -> f32 {
    if df == 0 { return 0.0; }
    let dff = df as f64;
    let r = chi2 / dff;
    ((r.cbrt() - (1.0 - 2.0 / (9.0 * dff))) / (2.0 / (9.0 * dff)).sqrt()) as f32
}

/// TWO-SAMPLE chi-square test of homogeneity: are `a` and `b` draws from the same distribution?
///
/// This — not the one-sample fit below — is the real gate. Comparing an empirical histogram against
/// a *host-computed* analytic nucleus requires replicating the kernels' float arithmetic exactly, and
/// the top-p cut is razor-sensitive deep in the tail (`cum >= top_p * sum` on bf16 logits with
/// `__expf`): the host and device disagree on the cutoff by a token or two, so BOTH the MTP path and
/// the plain sampler "fail" against the analytic reference, in lockstep. But the claim we actually
/// need is narrower and reference-free: *stochastic MTP emits from the same law as the plain
/// sampler*. Testing the two empirical histograms directly against each other says exactly that, and
/// is immune to the reference-model problem.
pub(super) fn gof2(a: &[u64], na: u64, b: &[u64], nb: u64) -> GofStat {
    let (naf, nbf) = (na as f64, nb as f64);
    let tot = naf + nbf;
    // TVD between the two empirical distributions (diagnostic only).
    let tvd = 0.5 * (0..a.len())
        .map(|t| (a[t] as f64 / naf - b[t] as f64 / nbf).abs())
        .sum::<f64>();

    // Pool the two samples per token; merge low-expectation tokens into a tail bin.
    let mut bins: Vec<(f64, f64)> = Vec::new(); // (O_a, O_b) per usable bin
    let (mut tail_a, mut tail_b) = (0.0f64, 0.0f64);
    for t in 0..a.len() {
        let (oa, ob) = (a[t] as f64, b[t] as f64);
        let pooled = oa + ob;
        if pooled == 0.0 { continue; }
        // Expected counts under H0 (same law): n_i * pooled/total.
        if naf * pooled / tot >= 5.0 && nbf * pooled / tot >= 5.0 {
            bins.push((oa, ob));
        } else {
            tail_a += oa;
            tail_b += ob;
        }
    }
    let tail_pooled = tail_a + tail_b;
    if naf * tail_pooled / tot >= 5.0 && nbf * tail_pooled / tot >= 5.0 {
        bins.push((tail_a, tail_b));
    }

    let mut chi2 = 0.0f64;
    for &(oa, ob) in &bins {
        let pooled = oa + ob;
        let ea = naf * pooled / tot;
        let eb = nbf * pooled / tot;
        chi2 += (oa - ea) * (oa - ea) / ea + (ob - eb) * (ob - eb) / eb;
    }
    let df = bins.len().saturating_sub(1);
    GofStat {
        tvd: tvd as f32,
        chi2_over_df: if df == 0 { 0.0 } else { (chi2 / df as f64) as f32 },
        z: wh_z(chi2, df),
        bins: bins.len(),
        outside: 0,
    }
}

/// Chi-square goodness-of-fit of `hist` against `support`, with the Wilson-Hilferty normal score.
/// Bins whose expected count falls below 5 are merged into a tail bin so the chi-square stays valid.
/// Diagnostic only — see `gof2` for why this is not the gate.
pub(super) fn gof(hist: &[u64], n: u64, support: &[(u32, f32)]) -> GofStat {
    let nf = n as f64;
    let inside: u64 = support.iter().map(|&(t, _)| hist[t as usize]).sum();
    let outside = n - inside;

    // TVD (diagnostic only).
    let mut pmap = vec![0.0f64; hist.len()];
    for &(t, p) in support { pmap[t as usize] = p as f64; }
    let tvd = 0.5 * (0..hist.len())
        .map(|t| (hist[t] as f64 / nf - pmap[t]).abs())
        .sum::<f64>();

    // Chi-square over merged bins (expected >= 5).
    let mut sorted: Vec<(u32, f32)> = support.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let (mut chi2, mut bins) = (0.0f64, 0usize);
    let (mut tail_o, mut tail_e) = (0.0f64, 0.0f64);
    for &(t, p) in &sorted {
        let e = nf * p as f64;
        let o = hist[t as usize] as f64;
        if e >= 5.0 {
            chi2 += (o - e) * (o - e) / e;
            bins += 1;
        } else {
            tail_o += o;
            tail_e += e;
        }
    }
    if tail_e >= 5.0 {
        chi2 += (tail_o - tail_e) * (tail_o - tail_e) / tail_e;
        bins += 1;
    }

    let df = bins.saturating_sub(1);
    let chi2_over_df = if df == 0 { 0.0 } else { (chi2 / df as f64) as f32 };
    GofStat { tvd: tvd as f32, chi2_over_df, z: wh_z(chi2, df), bins, outside }
}

/// The target nucleus distribution, replicating `sample_b`/`spec_verify_b` exactly:
/// top-k by logit → scale by 1/temp → softmax (max-subtracted) → smallest prefix whose cumulative
/// mass reaches `top_p * sum` → renormalize over that prefix. Returns (token, prob) over the support.
pub(super) fn nucleus_dist(lg: &[f32], temp: f32, top_k: usize, top_p: f32) -> Vec<(u32, f32)> {
    let k = top_k.clamp(1, 64).min(lg.len());
    let mut idx: Vec<u32> = (0..lg.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| {
        lg[b as usize].partial_cmp(&lg[a as usize]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let top: Vec<u32> = idx.into_iter().take(k).collect();
    let inv = 1.0f32 / temp;
    let scaled: Vec<f32> = top.iter().map(|&t| lg[t as usize] * inv).collect();
    let mx = scaled[0];
    let probs: Vec<f32> = scaled.iter().map(|&v| (v - mx).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let mut cum = 0.0f32;
    let mut nc = k - 1;
    for j in 0..k {
        cum += probs[j];
        if cum >= top_p * sum { nc = j; break; }
    }
    (0..=nc).map(|j| (top[j], probs[j] / cum)).collect()
}

pub struct VerifySample {
    pub p_of_draft: Vec<f32>, // [depth-1] p_j(drafted_token) — target prob of drafted token
    pub resid_tok:  Vec<u32>, // [depth-1] token ~ normalize(max(0, p_j \ {draft}))
    pub bonus_tok:  u32,      // token ~ p_{depth-1} (all-accept bonus)
}

/// Penalty device-buffer pointers for the MTP DRAFT path (single column, for the MTP head before
/// sampling). Like VerifyPenalty but for batch=1. Host fills and uploads before mtp_draft_step_sample.
pub struct DraftPenalty {
    pub tokens_ptr: u64,    // [MAX_PEN_TOKENS] i32
    pub counts_ptr: u64,    // [MAX_PEN_TOKENS] i16
    pub rep_pen_ptr: u64,   // [1] f32
    pub presence_ptr: u64,  // [1] f32
    pub freq_ptr: u64,      // [1] f32
}
