//! Scheduling policy pieces that carry no GPU state: lane order (`LaneOrder`, `fcfs_*`), `SpecLanes`, the arm-rate meter constants and their
//! pure helpers, the regime ordering of slots, boot banners for the MTP head, and the trace switches. `step.rs` and `pack.rs` are the main callers.

use super::*;

/// WP23: the DDS target for this round. `base` = --draft-confidence (0.4 served). Once the lane
/// has 16 rounds and its cost EMAs, a draft must be worth its cost: accept probability >= ĉ ms x
/// the lane's tokens/ms — max(base, clamp(ĉ·ema_tok/ema_ms, 0.15, 0.8)). The guard only RAISES
/// the target (long context: rows cost more; fast code rounds: every ms is worth more tokens).
pub(crate) fn wp23_target(base: f64, guard: bool, rounds: usize, c_hat: f64, ema_tok: f64, ema_ms: f64) -> f64 {
    if !guard || rounds < 16 || ema_ms <= 0.0 || ema_tok <= 0.0 {
        return base;
    }
    base.max((c_hat * ema_tok / ema_ms).clamp(0.15, 0.8))
}

/// `--spec-lanes-max`: never / a fixed lane threshold / cost-based (default).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum SpecLanes { Never, Over(usize), Auto }

/// The switch position `step_plain` orders its lanes around: qsa_limit() when the pack has a live
/// sparse attention and the per-row regime is on; usize::MAX = no switch (no indexer, --qsa-exl3 0
/// or --qsa-row-regime 0), which makes every order below the identity.
pub(super) fn regime_switch(qsa_live: bool, row_regime_on: bool, qsa_limit: usize) -> usize {
    if qsa_live && row_regime_on { qsa_limit } else { usize::MAX }
}

/// The scheduler side of the per-row regime (S-B9-REL-QSAREGIME; a free function since
/// S-B9-REL-SMALLFIXES 9a(iv) so the order is testable without a model): `slots` with the lanes
/// at/below the switch `lim` first, each class in its given order. `pos_of(slot)` = the lane's
/// next position.
pub(super) fn regime_ordered_slots(slots: &[usize], pos_of: impl Fn(usize) -> usize, lim: usize) -> Vec<usize> {
    let pos: Vec<usize> = slots.iter().map(|&s| pos_of(s)).collect();
    crate::exl3_forward::regime_order(&pos, lim).into_iter().map(|i| slots[i]).collect()
}

/// S-B9-REL-SMALLFIXES 9a(iii): the idle-point line of the lazily captured mixed-step graphs
/// (None = nothing new since the last line).
pub(super) fn mixed_graphs_idle_line(printed: u64, now: u64, lanes: usize) -> Option<String> {
    (now > printed).then(|| format!("[exl3-serve] mixed-step graphs idle: {printed}->{now} captured since boot (lazy, one per step shape \
seen, not capped; {lanes} lanes can form {} mixed step shapes per sparse grid bucket and penalty state)",
                                    crate::exl3_forward::mixed_step_shapes(lanes)))
}

/// S-B9-REL-SMALLFIXES item 5(a): the statistics a scheduler may report for a finishing stop
/// token. A greedy row's token is that row's argmax, so its rank must be 1: any other rank means
/// the logits row read back is not the row that chose the token, and nothing is reported rather
/// than a number about another row. A sampled row has no such check (any rank is possible).
pub(super) fn stop_info_checked(stats: Option<(f64, u32)>, greedy: bool) -> Option<(f64, u32)> {
    match stats {
        Some((p, rank)) if p.is_finite() && (!greedy || rank == 1) => Some((p, rank)),
        _ => None,
    }
}

/// SURFACE2 item 4: does this boot speculate at all — the pack has a draft head and neither
/// `--mtp off` nor the hatch turned it off? The boot lines about the speculative rounds (their lane
/// order, the shared-step policy, the DDS depth posture, the sampling mode) are printed only then.
pub(super) fn spec_live(has_head: bool, off: Option<crate::exl3_forward::MtpOff>) -> bool {
    has_head && off.is_none()
}

/// The boot line about the pack's draft head (None = the pack has none). The head is loaded either
/// way (it is part of the pack); with speculation off the line says it is not used.
pub(super) fn head_loaded_line(has_head: bool, off: Option<crate::exl3_forward::MtpOff>) -> Option<String> {
    if !has_head { return None; }
    Some(match off {
        None => "[exl3-serve] M1 pack head loaded (MTP draft head for chain-verify serving)".to_string(),
        Some(o) => format!("[exl3-serve] M1 pack head loaded (MTP draft head; NOT used on this boot: {})", o.flag()),
    })
}

/// The boot line of the MTP posture (None = the pack has no MTP head: nothing to say).
/// S-B9-REL-SMALLFIXES item 1: `--mtp off` is honoured and says so ('MTP OFF (--mtp off)'); the
/// two OFF lines keep the `[exl3-serve] MTP chain-verify OFF` prefix the gate scripts match.
pub(super) fn mtp_banner(has_head: bool, off: Option<crate::exl3_forward::MtpOff>, mtp_k: usize, draft_conf: f64) -> Option<String> {
    use crate::exl3_forward::MtpOff;
    if !has_head { return None; }
    Some(match off {
        None => format!("[exl3-serve] MTP chain-verify ON (k={mtp_k}, greedy + sampled lanes, dynamic draft stop {}; --mtp off / --exl3-mtp-k override)",
                        if draft_conf > 0.0 { format!("at {draft_conf}") } else { "OFF".to_string() }),
        Some(MtpOff::MtpFlag) => "[exl3-serve] MTP chain-verify OFF: MTP OFF (--mtp off) — plain decoding only, no draft head, no speculative rounds".to_string(),
        Some(MtpOff::Hatch) => "[exl3-serve] MTP chain-verify OFF (--exl3-no-mtp set — diagnostics hatch)".to_string(),
    })
}

/// CF-FCFS `--lane-order`: the order the SERIAL speculative rounds run in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LaneOrder {
    /// Today's order, unchanged: every scheduler step gives each busy MTP-capable lane one round,
    /// in slot order. This arm must stay bit-for-bit identical to the pre-flag build.
    Rr,
    /// Run to completion, first come first served: only the front lane (the oldest busy
    /// MTP-capable lane that still has quantum left) steps; it does so until it finishes or
    /// exhausts its quantum, then it rotates to the back.
    Fcfs,
}

/// CF-FCFS: the one lane that gets this scheduler step's speculative round, or None when this step
/// runs no speculative round at all.
///
/// PURE FUNCTION of replicated state (AGENTS 2.10, TP SPMD): it reads only `mtp_capable`,
/// `admit_seq` and `fcfs_credit` per lane — no wall clock, no rank-local counter, nothing the
/// head alone sees. Every rank admits the same requests in the same order (the head ships the
/// admits), so every rank's `admit_seq` and token counters are identical and every rank picks the
/// same lane. The rr arm returns None here and the caller keeps today's loop verbatim.
///
/// `lanes` is the busy-lane list in slot order (the caller's `slots`), already split into the
/// lanes this round would speculate (`capable`, MTP on and lane.mtp armed) and the MTP-off lanes
/// that batch separately. Rotation accounting (`fcfs_credit`, the rotation log) is the caller's.
pub(super) fn fcfs_front<'a>(capable: &[(usize, u64, usize)], quantum: usize) -> Option<usize> {
    // Oldest first (admit_seq is unique per live request: the counter only ever increments).
    let mut order: Vec<&(usize, u64, usize)> = capable.iter().collect();
    order.sort_by_key(|(_, seq, _)| *seq);
    // A lane that is not the front yet starts with a full quantum; only a lane that already had a
    // turn can be over its quantum (its credit is reset when it rotated to the back).
    order.into_iter().find(|(_, _, credit)| *credit < quantum).map(|(slot, _, _)| *slot)
}

/// CF-FCFSB: charge the front lane for the tokens it just generated and rotate it to the BACK of
/// the queue when it used its quantum. PURE function of replicated state: on rotation the lane is
/// re-stamped with the next admission sequence number (`*admit_ctr`, then the counter advances) so
/// `fcfs_front` — which orders by `admit_seq` — now sees it as the youngest lane. Without the
/// re-stamp the rotated lane keeps the lowest `admit_seq` and is picked again on the very next
/// step: the "rotation" was a log line and fcfs degenerated to run-to-completion by admission
/// order. Returns true when the lane rotated. `rr` never calls this.
pub(super) fn fcfs_charge(seq: &mut u64, credit: &mut usize, delta: usize, quantum: usize, admit_ctr: &mut u64) -> bool {
    *credit += delta;
    if *credit >= quantum {
        *credit = 0;
        *seq = *admit_ctr;
        *admit_ctr += 1;
        true
    } else {
        false
    }
}

impl SpecLanes {
    pub(crate) fn parse(v: Option<&str>) -> Result<SpecLanes> {
        match v.map(str::trim) {
            None | Some("auto") | Some("") => Ok(SpecLanes::Auto),
            Some("0") | Some("never") | Some("off") => Ok(SpecLanes::Never),
            Some(n) => n.parse::<usize>().map(SpecLanes::Over)
                .map_err(|_| anyhow::anyhow!("--spec-lanes-max must be auto, 0 or a lane count (got '{n}')")),
        }
    }
}

/// Model of one shared plain step over n lanes (ms, TP=1 code class, fitted on .14 2026-10-02: n=4 46.5, n=8 61.7,
/// n=16 90): the learned `shared_scale` corrects it for the topology / content actually served.
pub(super) fn shared_step_prior_ms(n: usize) -> f64 { 31.5 + 3.6 * n as f64 }

/// F4 (S-B9-REL-FINALFIX, from PACK5G finding 1): under `--lane-pack on` the shared arm of the
/// `use_shared_plain` comparison is priced as `shared_scale * shared_step_prior_ms(n)`, but the
/// scale is only LEARNED by running shared rounds (`learn_shared_ms`) — and with the scale
/// unlearned the prior never wins at c=4, so an on-boot rides packed pairs for the whole request
/// (19/20 PACK5G boots) while the shared step measured FASTER in 3 of the 4 standard cells (only
/// code/thinking-off pays for pairs). The probe breaks the loop: one shared plain step early,
/// then periodically, so both arms are priced from measurement and the comparison picks the
/// cheaper of the two. Cadence counts rounds on the OTHER arm:
pub(super) const SHARED_PROBE_FIRST: u32 = 16;  // while the scale has never been learned (boot start)

pub(super) const SHARED_PROBE_EVERY: u32 = 128;

/// P18 SPEC8K: weight of a new sample in the round-to-round WALL latency EMA of a lane
/// (Lane::ema_wall_ms). The second round of a lane already measures the whole cycle under load, so
/// this only smooths it.
pub(super) const WALL_EMA_W: f64 = 0.75;

/// P18 SPEC8K: a wall gap past this multiple of the round it measured is a stall (a prefill on
/// another lane, a TP seam, the idle gap between two requests), not the lateness of this lane.
pub(super) const WALL_STALL_X: f64 = 6.0;

/// P18 SPEC8K: the ms that ONE lane contributes to the cycle time of the serial arm: the time from
/// one round of the lane to its NEXT round, which is what its client waits, NOT the ms of the round
/// itself. A packed round charges each lane one SHARE of the verify (lane_pack::lane_share_ms), so
/// summing ema_ms over n lanes prices n rounds as if they all ran at once; the shared-batching arm
/// then looks slower than a cycle that in truth is about n times longer (spec-lanes auto picked
/// serial speculation for four busy lanes at 8K prose, where the shared plain step measured faster).
/// ema_wall_ms holds the measured queueing, so this is the whole cycle for one lane; for a lone lane
/// it equals ema_ms.
// S-C4XOVER arm indices: 0 = the shared plain step arm, 1 = the serial speculation arm
// (packed pairs where two lanes pack); they index the measured-rate arrays and the
// /metrics arm labels (SPEC_LANES_TO), and pick as usize maps true->0 / false->1.
pub(crate) const ARM_SHARED: usize = 0;

pub(crate) const ARM_SERIAL: usize = 1;

// S-C4XOVER cadences: a measured arm rate is FRESH for 512 scheduler steps at a busy
// count within +-1 of now; the window folds every 8 accumulated rounds; the idle arm is
// probed (a 4-step window of it) when its rate has gone stale, at most every 256
// riding-arm rounds (16 while it has never been measured, so a boot converges in ~2 s).
pub(super) const SPEC_ARM_FRESH_STEPS: u64 = 512;

pub(super) const SPEC_ARM_N_TOL: i64 = 1;

pub(super) const SPEC_ARM_FOLD_EVERY: u32 = 8;

pub(super) const SPEC_ARM_PROBE_FIRST: u32 = 16;

pub(super) const SPEC_ARM_PROBE_EVERY: u32 = 256;

pub(super) const SPEC_ARM_PROBE_WIN: u32 = 4;

/// POSTURE1 `--spec-arm-meter v2`: rates are kept per exact busy-lane count up to this many lanes (more share the
/// last entry); a window folds every SPEC_V2_FOLD rounds and is discarded when the busy count changes inside it;
/// rounds run inside a prefill (`--prefill-interleave`: lanes do not pack there) are not metered; the idle arm is
/// re-measured once its rate at this count is older than SPEC_V2_REFRESH steps, and after SPEC_ARM_PROBE_FIRST riding
/// steps when it has no fresh rate at this count.
pub(super) const SPEC_RATE_NMAX: usize = 16;

pub(super) const SPEC_V2_FOLD: u32 = 4;

pub(super) const SPEC_V2_REFRESH: u64 = 192;

/// ULTRA v2 fixes (the TP=2 POSTURE trace, 2026-11-13: a 4-step speculation window opened right after a release's
/// prefills measured 110-122 tok/s against ~175 steady, the pick sat on the shared plain step for 2 x 192 steps):
/// - SPEC_V2_SETTLE: after the busy count changes or a prefill step, this many steps are neither metered nor probed
///   (the first rounds of a just-admitted request are not the arm's rate);
/// - SPEC_V2_FOLD_SPEC: the speculation arm folds over this many metered steps (its tokens per step swing with the
///   sampled acceptance: 4 steps is noise), and its probe window is that long; the shared arm keeps SPEC_V2_FOLD;
/// - SPEC_V2_REPLACE: a PROBE's fold that contradicts the stored rate by more than this fraction replaces it instead
///   of averaging at 0.5 (a stale or warm-up estimate is wrong, not half right);
/// - SPEC_V2_REFRESH_FAST: the idle arm is re-probed after this many steps (not SPEC_V2_REFRESH) while the two measured
///   rates are within SPEC_V2_CLOSE of each other or the idle arm's rate at this count has fewer than 2 folds.
pub(super) const SPEC_V2_SETTLE: u32 = 4;

pub(super) const SPEC_V2_FOLD_SPEC: u32 = 12;

pub(super) const SPEC_V2_REPLACE: f64 = 0.15;

pub(super) const SPEC_V2_REFRESH_FAST: u64 = 48;

pub(super) const SPEC_V2_CLOSE: f64 = 0.10;

pub(super) fn spec_v2_fold(arm: usize) -> u32 { if arm == ARM_SHARED { SPEC_V2_FOLD } else { SPEC_V2_FOLD_SPEC } }

/// `--spec-arm-meter`: v1 (default) = the S-C4XOVER meter as it was; v2 = per-exact-count rates (above).
pub(super) fn spec_meter_v2() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::choice(crate::opt!("spec-arm-meter")).as_str() == "v2")
}

pub(crate) fn serial_cycle_ms(ema_ms: f64, ema_wall_ms: f64) -> f64 {
    if ema_wall_ms > 0.0 { ema_wall_ms } else { ema_ms }
} // (was the note on the SHARED_PROBE_EVERY line)

pub(super) fn shared_probe_due(ctr: u32, scale_unlearned: bool) -> bool {
    ctr >= if scale_unlearned { SHARED_PROBE_FIRST } else { SHARED_PROBE_EVERY }
}

/// Generated token ids a request keeps for its `[mtp-stats] toks=` field (--round-trace).
pub(super) const FIRST_TOKS: usize = 8;

/// `--round-trace` in the EXL3 server (S-B9-CF-PACK4S): `[mtp-stats]` also prints the request's
/// seed, its per-round arm trace and its first generated token ids. Diagnostics; no effect on any round.
pub(super) fn round_trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("round-trace")).is_ok())
}

/// POSTURE1 `--sched-trace` (diagnostic): one `[sched-trace]` line per scheduler step and per prefill chunk,
/// stamped with the wall clock. Reads lane state only; no decision reads anything it keeps.
pub(super) fn sched_trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("sched-trace")).is_ok())
}

pub(super) fn wall_now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}
