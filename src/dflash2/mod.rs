//! `dflash2` — the incoai/Qwen3.8-27B-DFlash2 block speculator substrate (post-pivot B8 target).
//!
//! This module family is the CPU-first, reference-exact implementation of the DFlash2 anatomy.
//! It is NOT `src/dspark/` (the parked RadixArk anatomy — S2) and NOT `src/dflash.rs` (the
//! Hy3-DFlash-B8 GPU drafter); both are precedent for PORT DISCIPLINE only (risk R9: three
//! distinct "dspark/dflash" names already; this one is `dflash2` everywhere).
//!
//! Contents (S2F session, per `PLAN/B8_S2F_WORKDOC.md`):
//!   * [`oracle`]  — the reference-exact CPU f32 oracle for one full DFlash2 speculation round,
//!     exposed PIECEWISE so S3F–S5F kernels can diff against each piece.
//!   * [`synth`]   — deterministic synthetic-weight generator + the target-borrowed
//!     embed/head/tap tables (fixed seeds; the golden harness ports the same algorithm).
//!   * [`load`]    — the REAL artifact loader + inventory/shape/dtype/sha256 assertions
//!     (`--probe-dflash2`).
//!
//! The anatomy constants below are the single source of truth for the exact 81-tensor inventory
//! and its param reconciliation (parsed from the REAL safetensors header = exactly
//! 1,924,404,480 params). Binding semantics: `ref/dflash/dflash/model.py` (the vendor
//! reference; where docs disagree, the code wins — see the oracle's DECISIONS ledger).

use std::sync::atomic::{AtomicUsize, Ordering};

pub mod oracle;
pub mod synth;
pub mod load;
pub mod mirror;
pub mod gpu;
pub mod capture;
pub mod round;
pub mod stepdump;

// ---------------------------------------------------------------------------
// Anatomy constants (config.json + parsed safetensors header + model.py — verified 2026-08-19).
// ---------------------------------------------------------------------------

/// Hidden size of the draft backbone and the tap projection output.
pub const HIDDEN: usize = 5120;
/// Query heads (GQA).
pub const NUM_HEADS: usize = 32;
/// KV heads (GQA) — 8 KV heads shared by the 32 Q heads (4:1).
pub const NUM_KV_HEADS: usize = 8;
/// Per-head dimension (full rotary — no partial_rotary_factor; rope_type "default").
pub const HEAD_DIM: usize = 128;
/// SwiGLU intermediate size (target-sized).
pub const INTER: usize = 17_408;
/// Number of draft-backbone layers (all `sliding_attention`).
pub const N_LAYERS: usize = 5;
/// Draft block length: anchor + 7× MASK = 8 positions, 7 draft tokens in one forward.
/// This is the DEFAULT; the live value is [`block()`] (see `--df2-block 8|16`).
pub const BLOCK: usize = 8;
/// Bit layout of the packed `window_B` launch word shared with `gpu_kernels.cu`'s
/// `DF2_WB_SHIFT`/`DF2_WB_MASK`: `(window << SHIFT) | block`. The field MUST be wide enough for
/// the block (16 needs 5 bits) or a block-16 launch decodes B=0 and the band attention breaks.
/// Keep this in lockstep with the `#define`s at the top of kernels/gpu_kernels.cu.
pub const DF2_WB_SHIFT: usize = 5;

/// The largest draft block this build supports. 16 is the verify ceiling (`gpu::MAX_VERIFY`):
/// a block-16 round emits anchor + 15 MASK and verifies 16 columns, which is exactly MAX_VERIFY.
/// Wider is refused at the CLI rather than silently falling to the prefill dequant path.
pub const MAX_BLOCK: usize = 16;

/// The process-wide DFlash2 draft block. Written ONCE from the resolved `--df2-block` before any
/// [`round::Df2Round`] is constructed, then read by every sizing and launch site in this module.
///
/// Why a process global rather than a field threaded through the round: the block appears in ~120
/// places across `round.rs`/`gpu.rs` (allocation sizes, launch dims, loop bounds, packed launch
/// words). A single accessor means there is no site that can disagree with the others, and with
/// the value left at 8 every allocation and launch is bit-for-bit today's — so block-8 serving
/// identity is structural, not something that has to be re-verified site by site.
static DF2_BLOCK: AtomicUsize = AtomicUsize::new(BLOCK);

/// The live draft block (8 by default, 16 under `--df2-block 16`). Always in `2..=MAX_BLOCK`.
#[inline]
pub fn block() -> usize {
    DF2_BLOCK.load(Ordering::Relaxed)
}

/// The live chain length = `block() - 1` (7 by default, 15 at block 16): the number of MASK rows
/// in the round, of draft tokens it emits, and of levels the selector walk runs.
#[inline]
pub fn levels() -> usize {
    block() - 1
}

/// Install the draft block. Idempotent for the same value; refuses anything outside 8 or 16.
/// Must be called before any DFlash2 round/buffer is constructed.
pub fn set_block(b: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        b == BLOCK || b == MAX_BLOCK,
        "--df2-block must be 8 or 16 (got {b}); 16 is the verify ceiling (MAX_VERIFY), and the \
         draft artifacts are block-agnostic so no re-bake is needed"
    );
    let prev = DF2_BLOCK.swap(b, Ordering::Relaxed);
    anyhow::ensure!(
        prev == BLOCK || prev == b,
        "df2 block already set to {prev}, cannot change to {b} after construction"
    );
    Ok(())
}

/// The MASK token id filling the 7 undrafted block positions (config `dflash_config`).
pub const MASK_TOKEN_ID: u32 = 248_070;
/// Vocabulary size (embed/lm_head are borrowed from the target at runtime; not in the checkpoint).
pub const VOCAB: usize = 248_320;
/// The TARGET trunk layers whose `hidden_states[l + 1]` outputs form the conditioning feature
/// (model.py:42 `extract_context_feature` with offset 1; config `target_layer_ids`).
pub const TAP_LAYERS: [usize; 5] = [5, 19, 33, 47, 61];
/// Concatenated tap dimension = 5 layers × hidden.
pub const TAP_CONCAT_DIM: usize = 5 * HIDDEN; // 25600
/// RoPE base (config `rope_parameters.rope_theta`), rope_type "default" → NO scaling factors.
pub const ROPE_THETA: f32 = 1e7;
/// Maximum RoPE positions (config `max_position_embeddings`).
pub const MAX_POSITIONS: usize = 262_144;
/// RMSNorm epsilon (config `rms_norm_eps`).
pub const RMS_EPS: f32 = 1e-6;
/// Sliding-window size on the context KV (config `sliding_window`; all 5 layers).
pub const SLIDING_WINDOW: usize = 2048;

/// S10R — window-bounded attention smem: the band kernels' scores region needs at most
/// `min(window + BLOCK, ntot)` slots (visit range [lo, ntot) spans <= window+B-1 entries;
/// PLAN/B8_S10R_DISSECTION.md §4). Constant at any ctx for window=SLIDING_WINDOW:
/// (128 + 2056 + 32) x 4 = 8,864 B — never approaches the 48 KiB dynamic default cap
/// (the old `(HEAD_DIM + ntot + 32) * 4` capped ctx at 12120).
pub fn band_smem(window: usize, ntot: usize) -> usize {
    let scores_len = if window + block() < ntot { window + block() } else { ntot };
    (HEAD_DIM + scores_len + 32) * 4
}
/// S10R (2026-08-21) — the SAFE absolute-context bound for the round is now the MODEL's
/// `max_position_embeddings` (262,144): the S10R window-bounding fix (`scores[j - lo]`,
/// `band_smem()` above — `PLAN/B8_S10R_DISSECTION.md` §4 + the Phase C gates) made the
/// band kernels' smem ctx-INDEPENDENT (8,864 B constant), so the old 12,120 ceiling is gone.
/// HISTORY (S10' §3, the pre-fix measurement, kept as provenance): the kernels used to size
/// their scores smem by ABSOLUTE ctx (`(HEAD_DIM + ntot + 32) * 4`, no opt-in => the 48 KiB
/// default cap) — bisect `/tmp/s10/ctx_bisect/results.txt`: seq 12120 PASSED (49,152 B),
/// seq 12121 FAILED (49,156 B, CUDA_ERROR_INVALID_VALUE at round.rs:1094); a 256K capture
/// would have needed ~1.049 MB/block — physically impossible (other implementations' dissected cubins
/// prove nobody stages O(ctx) scores: `PLAN/B8_S10R_DISSECTION.md` §2.2).
/// Above THIS bound the round's RoPE tables would exceed the model's trained positions —
/// the auto-fallback to MTP (main.rs `load_df2_round_dir`) remains the standing directive.
pub const MAX_CTX_SAFE: usize = MAX_POSITIONS;
/// Top-level `is_causal: false` — the explicit override (vLLM `test_dflash_causality.py`
/// resolution): every backbone layer is block-mask governed (non-causal); the window still applies.
pub const IS_CAUSAL: bool = false;
/// Dynamic-conv kernel size (config `conv_kernel_size`) — causal 2-tap.
pub const CONV_KERNEL: usize = 2;
/// Dynamic-conv channel group size (config `conv_group_size`) → 5120/16 = 320 groups.
pub const CONV_GROUP: usize = 16;
/// Dynamic-conv groups (= HIDDEN / CONV_GROUP).
pub const CONV_GROUPS: usize = HIDDEN / CONV_GROUP; // 320
/// Candidate-selector codebook/hidden-projection rank (config `selector_rank`).
pub const SELECTOR_RANK: usize = 256;
/// Candidate-selector top-k per position (config `selector_top_k`).
pub const SELECTOR_TOP_K: usize = 16;
/// Exact tensor count of the checkpoint: 5 layers × 15 + 6 global = 81.
///
/// RECONCILIATION (resolves the workdoc §2 "5×16 − log" overcount note): the workdoc's per-layer
/// list SAID 16 but LISTED 15 unique names (it double-counted its own list). The parsed real
/// header has exactly 15 per layer (q/o/k/v proj, q/k norm, input/post LN, gate/up/down,
/// attention_conv.{base_kernel, kernel_projection.weight}, mlp_conv.{base_kernel,
/// kernel_projection.weight}) × 5 = 75, plus 6 globals (fc.weight, hidden_norm.weight,
/// norm.weight, candidate_selector.{hidden_projection.weight, predecessor_codebook,
/// successor_codebook}) = 81. No names are shared and no proj is absent.
pub const N_TENSORS: usize = 81;
/// Exact parameter count of the checkpoint (parsed header sum; loader + generator both assert).
pub const N_PARAMS: u64 = 1_924_404_480;

/// The published sha256 of the REAL `model.safetensors` (HF LFS oid; staged copy verified).
pub const REAL_SHA256: &str = "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c";

/// The REAL artifact directory is USER-SUPPLIED (`--draft-dir`, mandatory on every DFlash2
/// path; owner rule 2026-08-23: no default, no fallback constant, a bad path stops the app).
/// A TP node never resolves a draft path at all — the head ships the artifact through the
/// cluster sync's content-addressed cache (cluster.rs `Msg::DraftManifest`) and rewrites the
/// shipped config's draft dir to the cache path, so nodes need NO local copy.

/// Default synthetic-artifact output directory (gitignored `tool_probe/` per the S2F prompt).
pub const DEFAULT_SYNTH_DIR: &str = "tool_probe/dflash2-synth";

/// The fixed generator seed for the synthetic artifact (deterministic by construction).
pub const SYNTH_SEED: u64 = 0xDF2A_2026_5DF2_0001;

/// The seed for the oracle's deterministic synthetic embed/head tables (the target-side tensors
/// the checkpoint deliberately omits — bound to the target at S4F). The golden harness
/// (`tool_probe/dflash2_golden.py`) ports this exact algorithm so oracle and reference see
/// bit-identical inputs.
pub const SYNTH_EMBED_HEAD_SEED: u64 = 0xDF2B_2026_E11B_0002;

/// The seed for the deterministic synthetic TAP hiddens (the trunk-captured conditioning feature;
/// stands in until S4F wires the real capture). Also ported by the golden harness.
pub const SYNTH_TAP_SEED: u64 = 0xDF2C_2026_7A95_0003;

/// The 81-tensor inventory in a FIXED, deterministic order (globals first, then layers 0..4),
/// listing the EXACT names parsed from the real safetensors header (8,928 header bytes).
///
/// Every tensor is BF16. Linears are `[out, in]`; norms `[n]`;
/// `{attention,mlp}_conv.base_kernel` is `[2 sides, kernel=2, hidden]`;
/// `{attention,mlp}_conv.kernel_projection.weight` is `[2*sides… = 2*kernel*groups, hidden]` =
/// `[1280, 5120]`; the selector codebooks are `nn.Embedding` tables `[vocab, rank]` stored WITHOUT
/// the `.weight` suffix (the reference's `from_pretrained` key_mapping adds it — model.py:633).
pub fn inventory() -> Vec<(String, Vec<usize>)> {
    let mut v = Vec::with_capacity(N_TENSORS);
    v.push(("fc.weight".to_string(), vec![HIDDEN, TAP_CONCAT_DIM]));
    v.push(("hidden_norm.weight".to_string(), vec![HIDDEN]));
    v.push(("norm.weight".to_string(), vec![HIDDEN]));
    v.push((
        "candidate_selector.hidden_projection.weight".to_string(),
        vec![SELECTOR_RANK, HIDDEN],
    ));
    v.push((
        "candidate_selector.predecessor_codebook".to_string(),
        vec![VOCAB, SELECTOR_RANK],
    ));
    v.push((
        "candidate_selector.successor_codebook".to_string(),
        vec![VOCAB, SELECTOR_RANK],
    ));
    for i in 0..N_LAYERS {
        let lp = format!("layers.{i}");
        v.push((format!("{lp}.self_attn.q_proj.weight"), vec![NUM_HEADS * HEAD_DIM, HIDDEN]));
        v.push((format!("{lp}.self_attn.k_proj.weight"), vec![NUM_KV_HEADS * HEAD_DIM, HIDDEN]));
        v.push((format!("{lp}.self_attn.v_proj.weight"), vec![NUM_KV_HEADS * HEAD_DIM, HIDDEN]));
        v.push((format!("{lp}.self_attn.o_proj.weight"), vec![HIDDEN, NUM_HEADS * HEAD_DIM]));
        v.push((format!("{lp}.self_attn.q_norm.weight"), vec![HEAD_DIM]));
        v.push((format!("{lp}.self_attn.k_norm.weight"), vec![HEAD_DIM]));
        v.push((format!("{lp}.input_layernorm.weight"), vec![HIDDEN]));
        v.push((format!("{lp}.post_attention_layernorm.weight"), vec![HIDDEN]));
        v.push((format!("{lp}.mlp.gate_proj.weight"), vec![INTER, HIDDEN]));
        v.push((format!("{lp}.mlp.up_proj.weight"), vec![INTER, HIDDEN]));
        v.push((format!("{lp}.mlp.down_proj.weight"), vec![HIDDEN, INTER]));
        v.push((
            format!("{lp}.attention_conv.base_kernel"),
            vec![2, CONV_KERNEL, HIDDEN],
        ));
        v.push((
            format!("{lp}.attention_conv.kernel_projection.weight"),
            vec![2 * CONV_KERNEL * CONV_GROUPS, HIDDEN],
        ));
        v.push((format!("{lp}.mlp_conv.base_kernel"), vec![2, CONV_KERNEL, HIDDEN]));
        v.push((
            format!("{lp}.mlp_conv.kernel_projection.weight"),
            vec![2 * CONV_KERNEL * CONV_GROUPS, HIDDEN],
        ));
    }
    debug_assert_eq!(v.len(), N_TENSORS);
    v
}

/// Reconcile the inventory to the exact published parameter count. Returns the summed param count.
/// This is the load-bearing arithmetic the generator AND loader both assert at runtime.
pub fn reconcile_params() -> u64 {
    inventory().iter().map(|(_, s)| s.iter().map(|&d| d as u64).product::<u64>()).sum()
}
