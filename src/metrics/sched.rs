//! Long-run scheduler gauges and serving-path counters: rejects, lane-reserve admits, KV spill, prefill packing / interleave, naive-engine state, prefix pool, scheduler busy / age, graph entries.
//! The scheduler thread writes these with relaxed atomics only; nothing is read on the hot path (module header in `mod.rs`). Rendered by `render.rs`. Map: `mod.rs`.

use super::*;

// ─── v0.7.3 long-run gauges (REL_LONGRUN_REPORT.md §3) ───────────────────────────────────
// Head-only, all atomics; the /proc-backed ones are read AT SCRAPE (no hot-path cost).
pub(super) static REQ_REJECTED: AtomicU64 = AtomicU64::new(0);
pub(super) static STREAMS_CANCELLED_BACKLOG: AtomicU64 = AtomicU64::new(0);
pub(super) static SCHED_STEPS: AtomicU64 = AtomicU64::new(0);
/// LANERES1: admissions decided by the --lane-reserve policy, by rule: 0 = own (the waiting
/// request's prompt-end snapshot sat in a free lane — admitted first, there), 1 = unreserved (a
/// free lane no reservation protects), 2 = forced (the queue head out-waited
/// --lane-reserve-patience and took a reserved lane; rate-limited by --lane-reserve-swap).
/// All zero when the policy is off.
pub(super) static LANE_RESERVE_ADMITS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// LANERES3: TP scoped admits whose allowed lanes were no longer free at apply time —
/// requeued by the head (never refused), skipped symmetrically by the mirrors. Non-zero
/// under bursty arrivals with the policy on; a sustained climb means the planner's view
/// is chronically stale and should be looked at.
pub(super) static LANE_RESERVE_TP_REQUEUED: AtomicU64 = AtomicU64::new(0);
/// Requests the --lane-reserve policy is holding back right now: waiting while free lanes sit
/// reserved for their conversations (0 when the policy is off or nothing waits).
pub(super) static LANE_RESERVE_WAITING: AtomicU64 = AtomicU64::new(0);
/// One admission through the --lane-reserve policy (`kind`: 0 own, 1 unreserved, 2 forced).
pub fn lane_reserve_admit(kind: usize) {
    if let Some(c) = LANE_RESERVE_ADMITS.get(kind) { c.fetch_add(1, Relaxed); }
}
/// The lane-reserve holdback gauge (requests waiting while free lanes are reserved for others).
pub fn lane_reserve_waiting(n: u64) { LANE_RESERVE_WAITING.store(n, Relaxed); }
/// LANERES3: a TP scoped admit whose allowed set had no free lane at apply time —
/// requeued (head) / skipped (mirror), never refused.
pub fn lane_reserve_tp_requeued() { LANE_RESERVE_TP_REQUEUED.fetch_add(1, Relaxed); }
/// LANERES2 SSD spill tier counters (--kv-spill): writes, restores, recompute fallbacks, bytes
/// stored, and prompt tokens whose re-prefill a restore avoided.
pub static KV_SPILL_WRITES: AtomicU64 = AtomicU64::new(0);
/// PERFFIX1: --prefill-interleave engagement — decode steps run between the chunks of joining prompts, and the
/// chunk boundaries that ran at least one (0 with the flag off).
/// PACKPF (--prefill-pack): prefill passes run by the packed admission (every pass of a packed run,
/// the single-prompt ones included), the passes that carried 2+ prompts, the prompt segments and rows
/// those carried, the prompts prefilled that way, the admission holds for a fuller pack, and the
/// runs skipped per reason (tp, df2, lane_reserve, wp16_xcheck, model, error).
pub static PREFILL_PACK_PASSES: AtomicU64 = AtomicU64::new(0);
pub static PREFILL_PACK_PACKED: AtomicU64 = AtomicU64::new(0);
pub static PREFILL_PACK_SEGS: AtomicU64 = AtomicU64::new(0);
pub static PREFILL_PACK_ROWS: AtomicU64 = AtomicU64::new(0);
pub static PREFILL_PACK_PROMPTS: AtomicU64 = AtomicU64::new(0);
pub static PREFILL_PACK_HOLDS: AtomicU64 = AtomicU64::new(0);
pub const PREFILL_PACK_SKIP_REASONS: [&str; 6] = ["tp", "df2", "lane_reserve", "wp16_xcheck", "model", "error"];
pub static PREFILL_PACK_SKIPS: [AtomicU64; 6] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                                                 AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
pub fn prefill_pack_pass(segs: u64, rows: u64) {
    PREFILL_PACK_PASSES.fetch_add(1, Relaxed);
    if segs > 1 {
        PREFILL_PACK_PACKED.fetch_add(1, Relaxed);
        PREFILL_PACK_SEGS.fetch_add(segs, Relaxed);
        PREFILL_PACK_ROWS.fetch_add(rows, Relaxed);
    }
}
pub fn prefill_pack_prompts(n: u64) { PREFILL_PACK_PROMPTS.fetch_add(n, Relaxed); }
pub fn prefill_pack_hold() { PREFILL_PACK_HOLDS.fetch_add(1, Relaxed); }
pub fn prefill_pack_skip(why: &str) {
    if let Some(i) = PREFILL_PACK_SKIP_REASONS.iter().position(|r| *r == why) { PREFILL_PACK_SKIPS[i].fetch_add(1, Relaxed); }
}
pub static PREFILL_INTERLEAVE_STEPS: AtomicU64 = AtomicU64::new(0);
pub static PREFILL_INTERLEAVE_POINTS: AtomicU64 = AtomicU64::new(0);
pub fn prefill_interleave(points: u64, steps: u64) {
    PREFILL_INTERLEAVE_POINTS.fetch_add(points, Relaxed);
    PREFILL_INTERLEAVE_STEPS.fetch_add(steps, Relaxed);
}
/// Naive exact speculative sampling (work item A): sampled verify rounds, drafts accepted by the ratio test, and the
/// rows whose residual draw the kernel could not make (no candidate list / empty residual: the target's own draw was committed)
pub static NAIVE_SPEC_SAMPLED_ROUNDS: AtomicU64 = AtomicU64::new(0);
pub static NAIVE_SPEC_SAMPLED_ACCEPTED: AtomicU64 = AtomicU64::new(0);
pub static NAIVE_SPEC_FALLBACK_NOLIST: AtomicU64 = AtomicU64::new(0);
pub static NAIVE_SPEC_FALLBACK_EMPTY: AtomicU64 = AtomicU64::new(0);
/// Naive TP lever 1: the transport ctx addresses (0 = none) whose DEVICE epoch counters are the liveness receipt of
/// each reduce path: every single-stage reduce advances the single ctx's counter by exactly 1, every tree reduce the
/// primary's by log2(world) (graph replays included — the counter lives on the device, so no host count can lag it).
pub static NAIVE_TP_PRIMARY_CTX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static NAIVE_TP_SINGLE_CTX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Lever 1b: the single-stage ctx runs the GPU-validated receive (`--tp-gpu-recv on`).
pub static NAIVE_TP_SINGLE_GPU_RECV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Lever 1c: the single-stage ctx's K1 is the multi-block one (`--naive-tp-k1 multi`, default).
pub static NAIVE_TP_SINGLE_K1_MB: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Naive expert tensor parallelism is live on this boot (`--naive-moe-shard tp`), and the G1 lane count (`--naive-lanes`).
pub static NAIVE_MOE_TP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static NAIVE_LANES: AtomicU64 = AtomicU64::new(0);
/// G1: every lane keeps its own drafter context (`--naive-lane-ctx on`, with lanes > 1 and a drafter).
pub static NAIVE_LANE_CTX: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Naive prefix reuse (`--prefix-cache on`): 1 = on, 2 = asked for but off on this boot (--naive-lanes > 1), 0 = not asked;
/// prompts that resumed / were prefilled whole, and the prompt tokens not prefilled again.
pub static NAIVE_PREFIX_ON: AtomicU64 = AtomicU64::new(0);
pub static NAIVE_PREFIX_HITS: AtomicU64 = AtomicU64::new(0);
pub static NAIVE_PREFIX_MISSES: AtomicU64 = AtomicU64::new(0);
pub static NAIVE_PREFIX_REUSED: AtomicU64 = AtomicU64::new(0);
pub static KV_SPILL_HITS: AtomicU64 = AtomicU64::new(0);
pub static KV_SPILL_MISSES: AtomicU64 = AtomicU64::new(0);
pub static KV_SPILL_BYTES: AtomicU64 = AtomicU64::new(0);
pub static KV_SPILL_SAVED: AtomicU64 = AtomicU64::new(0);
/// LANERES3: TP spill-agreement outcomes — a capture dropped because SOME rank's write
/// failed, and a restore skipped because some rank could not read the snapshot (both
/// re-prefill; the counters prove the cross-rank safety net engaged, not that disks fail).
pub static KV_SPILL_CAPTURE_PEER_DROPS: AtomicU64 = AtomicU64::new(0);
pub static KV_SPILL_RESTORE_PEER_MISSES: AtomicU64 = AtomicU64::new(0);
pub fn kv_spill_write(bytes: u64, tokens: u64) {
    KV_SPILL_WRITES.fetch_add(1, Relaxed);
    KV_SPILL_BYTES.fetch_add(bytes, Relaxed);
}
pub fn kv_spill_hit(tokens: u64) {
    KV_SPILL_HITS.fetch_add(1, Relaxed);
    KV_SPILL_SAVED.fetch_add(tokens, Relaxed);
}
pub fn kv_spill_miss_recompute() { KV_SPILL_MISSES.fetch_add(1, Relaxed); }
// PFXPOOL: the prefix pool (prompt states kept in memory across lanes, --prefix-pool-gb).
pub static PFX_POOL_CAP: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_BYTES: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_ENTRIES: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_CAPTURES: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_CAPTURE_US: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_SKIPS: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_REFUSED: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_EVICTIONS: AtomicU64 = AtomicU64::new(0);
/// PFXPOOL2: entry slabs one box freed on its own because it went under the pool's memory floor.
pub static PFX_POOL_EMERGENCY: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_HITS: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_HIT_TOKENS: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_RESTORE_US: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_MISSES: AtomicU64 = AtomicU64::new(0);
/// PFXPOOL: the pool's state after a change (bytes held, entries) and its cap.
pub fn pfx_pool_state(cap: u64, bytes: u64, entries: u64) {
    PFX_POOL_CAP.store(cap, Relaxed);
    PFX_POOL_BYTES.store(bytes, Relaxed);
    PFX_POOL_ENTRIES.store(entries, Relaxed);
}
/// PFXPOOL: a lane's outgoing prompt state was copied into the pool (`us` = the in-round copy time).
pub fn pfx_pool_capture(us: u64) { PFX_POOL_CAPTURES.fetch_add(1, Relaxed); PFX_POOL_CAPTURE_US.fetch_add(us, Relaxed); }
/// PFXPOOL: not captured because the pool already holds these tokens, or the prompt is too short to be worth it.
pub fn pfx_pool_skip() { PFX_POOL_SKIPS.fetch_add(1, Relaxed); }
/// PFXPOOL: not captured: larger than the cap, the memory floor (on any rank), or a copy that failed (on any rank).
pub fn pfx_pool_refused() { PFX_POOL_REFUSED.fetch_add(1, Relaxed); }
pub fn pfx_pool_evict(n: u64) { PFX_POOL_EVICTIONS.fetch_add(n, Relaxed); }
pub fn pfx_pool_emergency(n: u64) { PFX_POOL_EMERGENCY.fetch_add(n, Relaxed); }
/// PFXPOOL: a prompt resumed from a pool entry (`tokens` not prefilled again, `us` = the restore copy time).
pub fn pfx_pool_hit(tokens: u64, us: u64) {
    PFX_POOL_HITS.fetch_add(1, Relaxed);
    PFX_POOL_HIT_TOKENS.fetch_add(tokens, Relaxed);
    PFX_POOL_RESTORE_US.fetch_add(us, Relaxed);
}
/// PFXPOOL: the lane's own snapshot missed and no pool entry (deeper than the lane's WP16 checkpoint) matched.
pub fn pfx_pool_miss() { PFX_POOL_MISSES.fetch_add(1, Relaxed); }
/// LANERES3: a TP capture dropped because at least one rank's local write failed (agreed
/// across ranks; the key is forgotten everywhere — the policy stays replicated-identical).
pub fn kv_spill_capture_peer_drop() { KV_SPILL_CAPTURE_PEER_DROPS.fetch_add(1, Relaxed); }
/// LANERES3: a TP restore skipped because at least one rank could not read the snapshot
/// (agreed across ranks; both re-prefill — the rounds stay identical).
pub fn kv_spill_restore_peer_miss() { KV_SPILL_RESTORE_PEER_MISSES.fetch_add(1, Relaxed); }
/// Milliseconds since `START` when the scheduler last completed a round; 0 = never stepped.
pub(super) static SCHED_LAST_MS: AtomicU64 = AtomicU64::new(0);
pub(super) static SCHED_BUSY: AtomicU8 = AtomicU8::new(0);
pub(super) static GRAPH_ENTRIES: AtomicU64 = AtomicU64::new(0);

/// LR-4: a request rejected at admission because too many were already waiting.
pub fn reject_request() { REQ_REJECTED.fetch_add(1, Relaxed); }
/// LR-5: a stream cancelled by the scheduler for exceeding --stream-backlog-events.
pub fn stream_backlog_cancel() { STREAMS_CANCELLED_BACKLOG.fetch_add(1, Relaxed); }
/// A scheduler round STARTING (called from BOTH schedulers' step functions; ~35/s). Sets busy;
/// the age timestamp is written when work COMPLETES (`sched_touch`), so
/// velogb10_scheduler_last_step_age_seconds measures time since the last COMPLETED round or
/// prefill chunk — a 100 s prefill advances the age per chunk instead of reading as a wedge.
pub fn sched_step() {
    let round = SCHED_STEPS.fetch_add(1, Relaxed) + 1;
    SCHED_BUSY.store(1, Relaxed);
    // `--inject-stall` (diagnostic, unset by default): the keep-alive drill sleeps HERE, on the
    // scheduler thread, at the start of the chosen round (one OnceLock read per round otherwise).
    crate::server::stall_drill(round);
}
/// H3 (REL_V0_7_3): one unit of scheduler work COMPLETED (the end of a decode step, or one
/// prefill chunk). One relaxed store; keeps the age gauge honest during long prefills.
pub fn sched_touch() {
    if let Some(s) = START.get() { SCHED_LAST_MS.store(s.elapsed().as_millis() as u64, Relaxed); }
}
/// What the scheduler gauges say right now, for the SSE keep-alive (`server::keepalive_frames`):
/// `busy` = velogb10_scheduler_busy (work in progress since the last idle wait), `work_age` = the
/// time since it last COMPLETED a unit of work (`sched_touch`: the end of a decode round, or one
/// prefill chunk; None before the first one) = velogb10_scheduler_last_step_age_seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SchedState {
    pub busy: bool,
    pub work_age: Option<std::time::Duration>,
}

pub fn sched_state() -> SchedState {
    let last = SCHED_LAST_MS.load(Relaxed);
    let work_age = START.get().filter(|_| last != 0)
        .map(|s| std::time::Duration::from_millis((s.elapsed().as_millis() as u64).saturating_sub(last)));
    SchedState { busy: SCHED_BUSY.load(Relaxed) != 0, work_age }
}
/// The scheduler is about to block waiting for work (between rounds): busy goes 0. On an idle
/// server the age gauge then grows — that is EXPECTED (no rounds are running; nothing is
/// wedged); the alert rule pairs busy==1 with a high age, never busy==0.
pub fn sched_idle() { SCHED_BUSY.store(0, Relaxed); }
/// CUDA graphs currently held (published by the schedulers at boot-precapture end and per
/// lazy capture — the audit's LR-6 visibility).
pub fn set_graph_entries(n: u64) { GRAPH_ENTRIES.store(n, Relaxed); }

/// S-B9-REL-SMALLFIXES 9a(iii): the mixed-regime decode-step graphs (per-row attention regime).
/// They are captured lazily, one per step shape seen, and are not capped; this counts them (they
/// are also part of velogb10_graph_cache_entries). Returns the new count.
pub(super) static MIXED_STEP_GRAPHS: AtomicU64 = AtomicU64::new(0);
pub fn mixed_step_graph_captured() -> u64 { MIXED_STEP_GRAPHS.fetch_add(1, Relaxed) + 1 }
pub fn mixed_step_graphs() -> u64 { MIXED_STEP_GRAPHS.load(Relaxed) }
