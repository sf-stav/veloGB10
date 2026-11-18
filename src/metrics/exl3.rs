//! EXL3 engagement and graceful-fallback counters: PLE / w4moe / fractional-width expert receipts, KV-cache and kernel-width fallbacks, forced reconstruct and width fallbacks.
//! Receipts, not gates (AGENTS.md section 4: identical hashes are not a liveness receipt). Rendered by `render.rs`. Map: `mod.rs`.

use super::*;


// TP2BIT5 (#26) / EXL3 width-closure: graceful-fallback engagement counters. AGENTS §1b-bis.2: a
// feature is done only when its engagement counter is non-zero on the real posture; a counter that
// never moves is a path that never runs.
pub(super) static EXL3_PLE_WIDTH_ACCEPTED: AtomicU64 = AtomicU64::new(0); // B9: a PLE width outside {5,6} loaded and served
pub fn exl3_ple_width_accepted() { EXL3_PLE_WIDTH_ACCEPTED.fetch_add(1, Relaxed); }
/// FP4L S5 (shared): the n-gram (PLE) table read-ahead — prefill chunks whose NEXT chunk's n-gram rows were read into
/// the page cache on background threads while the current chunk ran (SSD-resident table only; 0 = never ran), the
/// n-gram rows that covered and the deduplicated 4 KB page reads issued. scripts/owner_check.py prints them.
pub(super) static PLE_WARM_AHEAD_CHUNKS: AtomicU64 = AtomicU64::new(0);
pub(super) static PLE_WARM_AHEAD_ROWS: AtomicU64 = AtomicU64::new(0);
pub(super) static PLE_WARM_AHEAD_READS: AtomicU64 = AtomicU64::new(0);
pub fn ple_warm_ahead(rows: u64, reads: u64) {
    PLE_WARM_AHEAD_CHUNKS.fetch_add(1, Relaxed);
    PLE_WARM_AHEAD_ROWS.fetch_add(rows, Relaxed);
    PLE_WARM_AHEAD_READS.fetch_add(reads, Relaxed);
}

pub(super) static EXL3_W4MOE_WIDTH_SERVED: AtomicU64 = AtomicU64::new(0); // B10: persistent grouped experts serving a non-3 width (2,4,5,6,7; 0 = never ran)
pub fn exl3_w4moe_width_served() { EXL3_W4MOE_WIDTH_SERVED.fetch_add(1, Relaxed); }

pub(super) static EXL3_DEVROUTE_FALLBACK: AtomicU64 = AtomicU64::new(0);
pub(super) static EXL3_FRAC_PF_BRIDGE_ROWS: AtomicU64 = AtomicU64::new(0); // LEG-B: prefill rows served by the fractional-k2 decode-moe bridge
pub fn exl3_frac_pf_bridge_rows(add: u64) { EXL3_FRAC_PF_BRIDGE_ROWS.fetch_add(add, Relaxed); } // TP-B: host-routed arm asked under TP -> device-routed EP arm ran instead
pub fn exl3_devroute_fallback() { EXL3_DEVROUTE_FALLBACK.fetch_add(1, Relaxed); }

pub(super) static EXL3_NGRAM_EMBED_ROWS: AtomicU64 = AtomicU64::new(0); // 1.66-class packs: token-embed rows decoded by the ngram codec at load (0 = plain embed path)
pub fn exl3_ngram_embed_decoded(rows: u64) { EXL3_NGRAM_EMBED_ROWS.fetch_add(rows, Relaxed); }
pub(super) static EXL3_MTP_SKIPPED: AtomicU64 = AtomicU64::new(0); // #26 Fix 4: boots whose draft head exceeds the served width set -> served WITHOUT speculation
pub fn exl3_mtp_skipped() { EXL3_MTP_SKIPPED.fetch_add(1, Relaxed); }
pub(super) static EXL3_PF_KERNEL_WIDTH_FALLBACK: AtomicU64 = AtomicU64::new(0);
// TPSHIP (2026-11-07): EXL3 boots asked for --kv-cache bf16 (no EXL3 store) and served f16 instead (AGENTS 1b-bis.8).
pub(super) static EXL3_KV_CACHE_FALLBACK: AtomicU64 = AtomicU64::new(0);
pub fn exl3_kv_cache_fallback() { EXL3_KV_CACHE_FALLBACK.fetch_add(1, Relaxed); } // #26 Fix 3: a prefill layer asked for the pipelined expert kernel (3/4-bit) and was served by the wide tiles kernel (3..5) instead
pub fn exl3_pf_kernel_width_fallback() { EXL3_PF_KERNEL_WIDTH_FALLBACK.fetch_add(1, Relaxed); }

// K35 (2026-10-31): fractional-width (K = 1.5 / 2.5 / 3.5 = k2 3 / 5 / 7) routed-expert serving receipts, by k2.
// EXL3_FRAC_EXPERT_ROUNDS = MoE layer rounds whose experts ran the fractional grouped body (decode, verify, MTP draft,
// the prefill bridge): eager calls plus graph replays x the rounds each captured graph holds (a capture executes
// nothing and is not counted; the capture sites take the per-capture tally into the CudaGraph). EXL3_FRAC_PF_WIDE_ROWS
// = prefill expert rows served by the fractional wide-tiles body (exl3_hmma_gemm_wide_tiles, EXL3_FRAC_BITS case).
// 0 everywhere = the fractional path never ran (AGENTS §4: identical hashes are not a liveness receipt).
// K8EXP (2026-11-07): the same tally also counts the newly admitted INTEGER widths K = 7 / 8 (k2 14 / 16), published as
// velogb10_exl3_k78_expert_rounds_total; the arrays are indexed by k2 (0..=16), so EXL3_WIDTH_SLOTS = 17.
pub const EXL3_WIDTH_SLOTS: usize = 17;
pub(super) static EXL3_FRAC_EXPERT_ROUNDS: [AtomicU64; EXL3_WIDTH_SLOTS] = [const { AtomicU64::new(0) }; EXL3_WIDTH_SLOTS];
pub(super) static EXL3_FRAC_PF_WIDE_ROWS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
// PFHALF (2026-11-06): prefill expert rows served by the PIPELINED kernel's HALF body (xq_moe_pf, EXL3_FRAC_BITS case).
pub(super) static EXL3_FRAC_PF_PIPE_ROWS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
thread_local! { static EXL3_FRAC_CAPTURE: std::cell::Cell<[u32; EXL3_WIDTH_SLOTS]> = const { std::cell::Cell::new([0; EXL3_WIDTH_SLOTS]) }; }
pub fn exl3_frac_expert_round(k2: usize, capturing: bool) {
    if k2 >= EXL3_WIDTH_SLOTS { return; }
    if capturing {
        EXL3_FRAC_CAPTURE.with(|c| { let mut v = c.get(); v[k2] += 1; c.set(v); });
    } else {
        EXL3_FRAC_EXPERT_ROUNDS[k2].fetch_add(1, Relaxed);
    }
}
/// Capture bookkeeping: reset right after cuStreamBeginCapture, take at the instantiation (the rounds the graph holds).
pub fn exl3_frac_capture_reset() { EXL3_FRAC_CAPTURE.with(|c| c.set([0; EXL3_WIDTH_SLOTS])); }
pub fn exl3_frac_capture_take() -> [u32; EXL3_WIDTH_SLOTS] { EXL3_FRAC_CAPTURE.with(|c| c.replace([0; EXL3_WIDTH_SLOTS])) }
pub fn exl3_frac_expert_rounds_replayed(n: &[u32; EXL3_WIDTH_SLOTS]) {
    for (k2, v) in n.iter().enumerate() { if *v > 0 { EXL3_FRAC_EXPERT_ROUNDS[k2].fetch_add(*v as u64, Relaxed); } }
}
pub fn exl3_frac_pf_wide_rows(k2: usize, rows: u64) { if k2 < 8 { EXL3_FRAC_PF_WIDE_ROWS[k2].fetch_add(rows, Relaxed); } }
pub fn exl3_frac_pf_pipe_rows(k2: usize, rows: u64) { if k2 < 8 { EXL3_FRAC_PF_PIPE_ROWS[k2].fetch_add(rows, Relaxed); } }
pub fn exl3_frac_expert_rounds(k2: usize) -> u64 { if k2 < EXL3_WIDTH_SLOTS { EXL3_FRAC_EXPERT_ROUNDS[k2].load(Relaxed) } else { 0 } }
/// F3: EXL3 prefill-chain GEMMs that a module the wide trellis kernel cannot decode (bit width
/// outside 3..=7 since P7 B5) served through the reconstruct + cuBLASLt arm instead of the arm the option asked
/// for, by the module's bit width (`velogb10_exl3_recon_forced_total`).
pub(super) static RECON_FORCED: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

/// F3: count one forced-reconstruct GEMM for a module of this bit width (1..=8; others ignored).
pub fn exl3_recon_forced(bits: i32) {
    let b = bits as usize;
    if (1..RECON_FORCED.len()).contains(&b) {
        RECON_FORCED[b].fetch_add(1, Relaxed);
    }
}

/// P7 B6: PSK grids refused by the width guard in `psk_grid` (bit width outside 2..=6): the
/// chain falls back to the plain split-K grid (`exl3_hmma_gemm_ks`), correct but without the
/// persistent-grid scheduling. By bit width (`velogb10_exl3_psk_width_fallback_total`).
pub(super) static PSK_WIDTH_FALLBACK: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

/// P7 B6: count one PSK width-guard fallback for a module of this bit width (1..=9; others ignored).
pub fn exl3_psk_width_fallback(bits: i32) {
    let b = bits as usize;
    if (1..PSK_WIDTH_FALLBACK.len()).contains(&b) {
        PSK_WIDTH_FALLBACK[b].fetch_add(1, Relaxed);
    }
}

/// PCKG-ALL (2026-10-30): WP21 fused split-K (exl3_hmma_gemm_ks_fx*) refused for a module
/// whose k2 is outside the fused bodies (3..=14; k2 16 = the 4.18bpw 8-bit MTP head):
/// served by the plain 3-launch tail instead (which carries case 16). By k2
/// (`velogb10_exl3_wp21_width_fallback_total`); the boot log prints one WARN, first time.
pub(super) static WP21_WIDTH_FALLBACK: [AtomicU64; 33] = [const { AtomicU64::new(0) }; 33];

/// PCKG-ALL: count one WP21 width-guard refusal for a module of this k2 (1..=32; else ignored).
pub fn exl3_wp21_width_fallback(k2: i32) {
    let b = k2 as usize;
    if (1..WP21_WIDTH_FALLBACK.len()).contains(&b) {
        WP21_WIDTH_FALLBACK[b].fetch_add(1, Relaxed);
    }
}
