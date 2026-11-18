//! Policy predicates for the serving GEMMs and the quantisation stay-lists: split-K selection (a function of the weight
//! shape only, so decode and verify share one geometry: batch-invariant, AGENTS.md section 2.4), `m4_*`, name classifiers. Map: `mod.rs`.

use super::*;

/// Split-K factor for the NVFP4 mma serving GEMM: 1 = the plain persistent-grid kernel, S > 1 =
/// `gemm_mma_fp4_splitk_b` partials + `gemm_mma_splitk_reduce_b`. A function of the WEIGHT SHAPE
/// (outn = M, inn = K) ONLY — never of the batch width N: decode (N=1) and verify (N<=16) see
/// identical split geometry, and the cross-split reduce is a fixed-order sum (s = 0..S-1, no
/// atomics), so column 0 stays bit-identical at every N (AGENTS.md §2.4; gate --probe-binv).
///
/// DEFAULT ON (E15, 2026-08-09 — D1 measured +6.9% e2e 27B, +15.4% hy3 down, binv/LOSSLESS
/// green, acceptance better; serving A/B +9% on matched samples). Escape: --splitk-gemm=0
/// (SPMD-safe — ships through TpConfig). --splitk-gemm=<S>=2..8 stays a diagnostics-only
/// FORCE override (single-node A/B only — it does NOT ship the value to the node).
///
/// Split where it pays: a long-K reduction (K >= 8192) with too few output tiles to fill the 288
/// resident blocks. The 27B FFN down-proj (M=5120, K=17408) gets S=4 — 34 k-blocks per warp instead
/// of 136, and 1280 work items instead of 320 tiles (1.11 waves); its TP=2 half (K=8704) gets S=2.
/// S = clamp(K/4096, 1, 8) targets ~32-64 k-blocks per warp per split.
pub(super) fn gemm_fp4_nsplit(outn: usize, inn: usize) -> usize {
    if outn / 16 > SPLITK_MAX_TILES { return 1; }
    static FORCE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    if let Some(f) = FORCE.get_or_init(|| {
        crate::opts::var(crate::opt!("splitk-gemm")).ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|s| *s >= 2)
    }) {
        // Forced S is head-process env and never ships to the node: under TP the ranks would
        // pick different split geometry and the all-reduce would mix divergent partials. The
        // force is single-node diagnostics only — refuse it (loudly) when TP is live.
        if crate::tp::tp_config().is_some() {
            eprintln!("[splitk] WARNING: --splitk-gemm=<S> force ignored under TP (head-only env; ranks would desync)");
        } else {
            return (*f).clamp(2, SPLITK_MAX_S);
        }
    }
    if !splitk_gemm_enabled() { return 1; }
    if inn >= 8192 { (inn / 4096).clamp(1, SPLITK_MAX_S) } else { 1 }
}

/// `--splitk-gemm` (CLI) or a shipped TpConfig from a TP head. Env-first, then the shipped config,
/// then the OFF default — the node resolves the same value as the head (SPMD rule, AGENTS.md §5a).
fn splitk_gemm_enabled() -> bool {
    if crate::opts::var(crate::opt!("splitk-gemm")).as_deref() == Ok("0") { return false; }
    crate::opts::var(crate::opt!("splitk-gemm")).is_ok()
        || crate::tp::tp_config().map(|c| c.splitk_gemm).unwrap_or(false)
}

/// MXFP4-native design §1/§3.3 stay-list: tensors that run the bf16 chain even in native mode —
/// the LM head (its logits decide the argmax; a W4A4 head flips near-ties, measured 2026-08-07)
/// and the embed (gather-only, no activation quant; design keeps it standard). Same rule as the
/// MTP-head allowlist; the loader remembers their qweight ptrs.
pub(super) fn design_stay_list(name: &str) -> bool {
    name == "lm_head.weight" || name.ends_with(".embed_tokens.weight")
}

/// MTP-head tensor in the loader's gwn names? qwen names the head `mtp.*`; hy_v3's MTP block
/// lives at `model.layers.<num_layers>.*` (the layer-80 block). mxfp4-native mode routes
/// these to the bf16 chain — the head's logits decide draft acceptance, and a W4A4 head
/// erodes near-tie acceptance (the C4 lesson, measured on qwen). Same rule as the qwen
/// `"mtp."` prefix check it generalizes.
pub(super) fn m4_safe_on(env: &str) -> bool { env == "safe" }

/// --pf-mixer4 membership: 'all'/'1' = every mixer; 'safe' = out_proj/o_proj/qkv_proj
/// (everything but the GDN fused in_proj); otherwise a comma list of name substrings
/// ('out_proj,o_proj') for bisecting which projection family breaks the gate.
pub(super) fn m4_member(env: &str, name: &str) -> bool {
    if env == "all" || env == "1" { return true; }
    if env == "safe" { return !name.contains("in_proj"); }
    env.split(',').map(str::trim).any(|p| !p.is_empty() && name.contains(p))
}

pub(super) fn is_qkv_projection_name(name: &str) -> bool {
    // GDN/attention projection family — outputs feed the softmax or the GDN RECURRENCE
    // (a/b/z gates), both of which amplify activation-quant noise at structural
    // boundaries. in_proj_qkv/a/b/z + out_proj under linear_attn; classic q/k/v/o too.
    name.contains("in_proj") || name.contains("out_proj")
        || name.contains("q_proj") || name.contains("k_proj") || name.contains("v_proj")
        || name.contains("o_proj") || name.contains("qkv_proj")
}
pub(super) fn is_mtp_head_name(name: &str, is_hy3: bool, num_layers: usize) -> bool {
    name.starts_with("mtp.")
        || (is_hy3 && name.starts_with(&format!("model.layers.{}.", num_layers)))
}

/// Bisect knob (35B MoE binv hunt, 2026-08-07): allowlist the stacked expert weights so they run
/// the mode-off (dequant+bf16) grouped/slot kernels — proven bitwise binv on the 122B — instead
/// of the native W4A4 grouped/slot kernels. Lossless-at-depth-2 flipping on this knob pins the
/// break to the native MoE kernels vs the GDN/attention path.
pub(super) fn experts_stay_list(name: &str) -> bool {
    crate::opts::var(crate::opt!("mxfp4-allow-experts")).is_ok() && name.contains(".mlp.experts.")
}
