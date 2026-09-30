//! `dspark` — the P8 (RadixArk/Qwen3.8-27B-DSpark) block-diffusion speculator substrate.
//!
//! This module family is the CPU-first, license-clean reference implementation of the P8 anatomy.
//! It is NOT the DSV4-family drafter in `src/dsv4_dspark.rs` (a different model family — see
//! `PLAN/B8_DSPARK_IMPLEMENTATION_PLAN.md` §0.1 / risk R9). It is also NOT a fork of
//! `src/dflash.rs` (Hy3-DFlash-B8); that module is precedent for PORT DISCIPLINE only.
//!
//! Contents (S2 session, per `PLAN/B8_S2_WORKDOC.md`):
//!   * [`oracle`]  — the reference-exact CPU f32 oracle for one full DSpark speculation round,
//!     exposed PIECEWISE so S3–S5 kernels can diff against each piece.
//!   * [`synth`]   — the deterministic synthetic-weight generator (`--gen-dspark-synth`), which
//!     writes a REAL safetensors artifact SHAPE-IDENTICAL to the real 62-tensor checkpoint.
//!   * [`load`]    — the loader + inventory/shape/dtype/sha256 assertions (`--probe-dspark-synth`).
//!
//! The anatomy constants below are the single source of truth for the exact 62-tensor inventory
//! and its param reconciliation (derived + verified = exactly 1,359,284,737 params).

pub mod oracle;
pub mod synth;
pub mod load;
pub mod round;

// ---------------------------------------------------------------------------
// Anatomy constants (PLAN/B8_S2_WORKDOC.md §3 — authoritative; use as-is).
// ---------------------------------------------------------------------------

/// Hidden size of the draft backbone and the tap projection output.
pub const HIDDEN: usize = 5120;
/// Query heads (GQA).
pub const NUM_HEADS: usize = 40;
/// KV heads (GQA) — 8 KV heads shared by the 40 Q heads (5:1).
pub const NUM_KV_HEADS: usize = 8;
/// Per-head dimension (also the full rotary dimension — no partial_rotary_factor in the drafter).
pub const HEAD_DIM: usize = 128;
/// SwiGLU intermediate size.
pub const INTER: usize = 10240;
/// Number of draft-backbone layers.
pub const N_LAYERS: usize = 5;
/// Draft block length: anchor + 6× MASK = 7 positions, 7 draft tokens in one forward.
pub const BLOCK: usize = 7;
/// The MASK token id filling the 6 undrafted block positions.
pub const MASK_TOKEN_ID: u32 = 248_077;
/// Vocabulary size (the target's vocab; embed/lm_head are borrowed from the target at runtime).
pub const VOCAB: usize = 248_320;
/// The TARGET trunk layers whose post-layer outputs form the conditioning feature. The reference
/// model code reads `hidden_states[l + 1]` for each `l` here (post-layer output, 0-based).
pub const TAP_LAYERS: [usize; 5] = [4, 16, 28, 40, 52];
/// Concatenated tap dimension = 5 layers × hidden.
pub const TAP_CONCAT_DIM: usize = 5 * HIDDEN; // 25600
/// RoPE base (YaRN).
pub const ROPE_THETA: f32 = 1e7;
/// RoPE interpolation factor (YaRN).
pub const ROPE_FACTOR: f32 = 32.0;
/// YaRN correction-range beta_fast.
pub const BETA_FAST: u32 = 32;
/// YaRN correction-range beta_slow.
pub const BETA_SLOW: u32 = 1;
/// YaRN original (unscaled) max position embeddings.
pub const ORIG_CTX: usize = 8192;
/// Maximum RoPE positions.
pub const MAX_POSITIONS: usize = 262_144;
/// RMSNorm epsilon (qwen3 family default).
pub const RMS_EPS: f32 = 1e-6;
/// Markov bigram rank (W1/W2 latent dim).
pub const MARKOV_RANK: usize = 256;
/// Confidence-head input dim = [draft hidden ∥ Markov latent].
pub const CONF_IN_DIM: usize = HIDDEN + MARKOV_RANK; // 5376
/// Exact tensor count of the checkpoint (5 layers × 11 + 7 global).
pub const N_TENSORS: usize = 62;
/// Exact parameter count of the checkpoint (reconciled; see `PLAN/B8_S2_WORKDOC.md` §3).
pub const N_PARAMS: u64 = 1_359_284_737;

/// S7 CLOSE-OUT: sha256 of the REAL published artifact (`model.safetensors`, identical bytes in
/// Qwen3.8-27B-DSpark{,-512K,-vLLM,-vLLM-512K}; verified against `--probe-dspark-bind`'s
/// prefix/suffix pin 2026-09-06). Loaders pin the full hex.
pub const REAL_SHA256: &str = "9d26d5e637551c244d543c67c790bd0947f360e005c569e5851a185ffe692786";

/// Default synthetic-artifact output directory — CWD-relative (never a hardcoded box path;
/// owner rule 2026-08-23), overridable via --dspark-synth-dir or the CLI value.
pub const DEFAULT_SYNTH_DIR: &str = "dspark-synth-qwen38";

/// The fixed generator seed for the synthetic artifact. Deterministic by construction — no system
/// entropy, no HashMap iteration. Documented here so regeneration is reproducible.
pub const SYNTH_SEED: u64 = 0xD5A2_2026_5D5A_0001;

/// The seed for the oracle's deterministic synthetic embed/lm_head tables (the target-side tensors
/// the checkpoint deliberately omits — bound to the target at S4/S7). Independent of [`SYNTH_SEED`]
/// so re-rolling the artifact weights never changes the borrowed embed/head surface.
pub const SYNTH_EMBED_HEAD_SEED: u64 = 0xE11B_2026_D5A2_0002;

/// The 62-tensor inventory in a FIXED, deterministic order (globals first, then layers 0..4).
///
/// Every tensor is BF16. Shapes are the workdoc §3 table verbatim. `markov_head.markov_w1.weight`
/// is the `Embedding(248320→256)` table (stored `[num_embeddings, embedding_dim]`);
/// `markov_head.markov_w2.weight` is the `Linear(256→248320, no bias)` weight stored `[out, in]` —
/// both are `[248320, 256]`. `confidence_head.proj.weight` is `[1, 5376]` (Linear(5376→1) over
/// `[h ∥ latent]`); `confidence_head.proj.bias` is the single scalar bias in the whole model.
///
/// S7 CLOSE-OUT (2026-09-06, DECISION-K resolved): the real checkpoint's safetensors header
/// (sha256 `9d26d5e6…fe692786`, 62 tensors, all BF16, params == N_PARAMS) uses the
/// `markov_head.markov_w1/w2` + `confidence_head.proj.*` names — the earlier DFlash-convention
/// guesses (`markov.W1` / `confidence.*`) never existed on disk. The synth generator writes the
/// real names, so the oracle/probe contract is unchanged.
pub fn inventory() -> Vec<(String, Vec<usize>)> {
    let mut v = Vec::with_capacity(N_TENSORS);
    v.push(("fc.weight".to_string(), vec![HIDDEN, TAP_CONCAT_DIM]));
    v.push(("hidden_norm.weight".to_string(), vec![HIDDEN]));
    v.push(("norm.weight".to_string(), vec![HIDDEN]));
    v.push(("markov_head.markov_w1.weight".to_string(), vec![VOCAB, MARKOV_RANK]));
    v.push(("markov_head.markov_w2.weight".to_string(), vec![VOCAB, MARKOV_RANK]));
    v.push(("confidence_head.proj.weight".to_string(), vec![1, CONF_IN_DIM]));
    v.push(("confidence_head.proj.bias".to_string(), vec![1]));
    for i in 0..N_LAYERS {
        let lp = format!("layers.{i}");
        v.push((format!("{lp}.self_attn.q_proj.weight"), vec![HIDDEN, HIDDEN]));
        v.push((format!("{lp}.self_attn.k_proj.weight"), vec![NUM_KV_HEADS * HEAD_DIM, HIDDEN]));
        v.push((format!("{lp}.self_attn.v_proj.weight"), vec![NUM_KV_HEADS * HEAD_DIM, HIDDEN]));
        v.push((format!("{lp}.self_attn.o_proj.weight"), vec![HIDDEN, HIDDEN]));
        v.push((format!("{lp}.self_attn.q_norm.weight"), vec![HEAD_DIM]));
        v.push((format!("{lp}.self_attn.k_norm.weight"), vec![HEAD_DIM]));
        v.push((format!("{lp}.input_layernorm.weight"), vec![HIDDEN]));
        v.push((format!("{lp}.post_attention_layernorm.weight"), vec![HIDDEN]));
        v.push((format!("{lp}.mlp.gate_proj.weight"), vec![INTER, HIDDEN]));
        v.push((format!("{lp}.mlp.up_proj.weight"), vec![INTER, HIDDEN]));
        v.push((format!("{lp}.mlp.down_proj.weight"), vec![HIDDEN, INTER]));
    }
    debug_assert_eq!(v.len(), N_TENSORS);
    v
}

/// Reconcile the inventory to the exact published parameter count. Returns the summed param count.
/// This is the load-bearing arithmetic the generator AND loader both assert at runtime.
pub fn reconcile_params() -> u64 {
    inventory().iter().map(|(_, s)| s.iter().map(|&d| d as u64).product::<u64>()).sum()
}
