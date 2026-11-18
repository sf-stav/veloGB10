//! FP4-Local counters: host dispatches of the FP4-Local weight kernels by weight class and row class, the pack marker, the exact-precision twins and the resolved precision levers.
//! A CUDA-graph replay does not count here (see the doc comments): the pack marker and the trunk's round counters are the served-traffic receipts. Map: `mod.rs`.

use super::*;

/// FP4L (nvidia/Qwen3.8-Flash-Next-FP4-Local): host DISPATCHES of the FP4-Local weight kernels by
/// weight class [mxfp8, f16, bf16 (stored bits), nvfp4-experts] x row class [decode m = 1, verify/packed 2..=16, prefill > 16].
/// A CUDA-graph replay re-runs captured kernels without a host dispatch, so in graphed steady decode these
/// grow at capture only; `velogb10_fp4l_pack` + the trunk's step / round counters are the served-traffic
/// receipts (every linear of an FP4-Local model is an FP4L weight).
pub(super) static FP4L_DISPATCH: [[AtomicU64; 3]; 5] = [const { [const { AtomicU64::new(0) }; 3] }; 5];
/// 1 once an FP4-Local pack is loaded (the feature marker the boot also prints).
pub(super) static FP4L_PACK: AtomicU64 = AtomicU64::new(0);
/// FP4L precision levers (PLAN/FP4L_DESIGN.md §3a): the resolved values, set once at load.
pub(super) static FP4L_PREC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
/// FP4L: host dispatches of the exact-precision TWINS of trunk kernels, by op (same capture caveat as
/// FP4L_DISPATCH): [hc_mixer, hc_norm, hc_mix, embed, shared_gate, ple, router].
pub const FP4L_TWIN_OPS: [&str; 7] = ["hc_mixer", "hc_norm", "hc_mix", "embed", "shared_gate", "ple", "router"];
pub(super) static FP4L_TWIN: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

fn fp4l_row_class(m: usize) -> usize { if m <= 1 { 0 } else if m <= 16 { 1 } else { 2 } }
/// FP4L: one dense GEMM dispatch of weight class `kind` ("mxfp8" | "f16" | "bf16") at `m` rows.
pub fn fp4l_gemm(kind: &str, m: usize) {
    let k = match kind { "mxfp8" => 0, "bf16" => 2, "trellis" => 4, _ => 1 };
    FP4L_DISPATCH[k][fp4l_row_class(m)].fetch_add(1, Relaxed);
}
/// FP4L: one NVFP4 routed-expert layer dispatch at `m` rows (prefill passes m > 16).
pub fn fp4l_moe(m: usize) { FP4L_DISPATCH[3][fp4l_row_class(m)].fetch_add(1, Relaxed); }
/// FP4L: an FP4-Local pack was loaded.
pub fn fp4l_pack_loaded() { FP4L_PACK.store(1, Relaxed); }
/// FP4L: one dispatch of an exact-precision twin (`op` = one of FP4L_TWIN_OPS).
pub fn fp4l_twin(op: &str) {
    if let Some(i) = FP4L_TWIN_OPS.iter().position(|o| *o == op) { FP4L_TWIN[i].fetch_add(1, Relaxed); }
}
/// FP4L: the resolved precision levers (once, at load).
pub fn fp4l_precision(plain_exact: bool, norms_f32: bool, lm_head: &str, draft_head: &str, prefill_exact: bool) {
    let _ = FP4L_PREC.set(format!("plain=\"{}\",norms=\"{}\",lm_head=\"{lm_head}\",draft_head=\"{draft_head}\",prefill_dense=\"{}\"",
                                  if plain_exact { "exact" } else { "f16" }, if norms_f32 { "f32" } else { "f16" },
                                  if prefill_exact { "exact" } else { "lt" }));
}
