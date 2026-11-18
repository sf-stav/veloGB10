//! Lane-pack and spec-lanes gauges: packed / serial rounds, skip reasons, in-flight and lane capacity, batched primes, the spec-lanes auto-policy estimates and meters.
//! Observability only; the policy itself lives in the schedulers. Rendered by `render.rs`. Map: `mod.rs`.

use super::*;

/// Admitted-but-unfinished requests (running + waiting) — exact, from the Req guard.
pub fn inflight() -> u64 { IN_FLIGHT.load(Relaxed) }
/// S-B9-CF-PACK4 (`--lane-pack`): scheduler rounds by arm and the lane rounds that stayed out of a
/// pack, by reason. The arms are counted with the flag on or off (off: `packed` stays 0).
pub const LANE_PACK_ARMS: [&str; 3] = ["packed", "serial", "shared_plain"];
pub(super) static LANE_PACK_ROUNDS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
pub(super) static LANE_PACK_SKIP: [AtomicU64; crate::lane_pack::Skip::ALL.len()] =
    [const { AtomicU64::new(0) }; crate::lane_pack::Skip::ALL.len()];
/// S-B9-CF-PACK4S: LANE rounds served by packed verifies, by the lane's kind (two per packed
/// round): real-q sampled, match-rule sampled, greedy.
pub const LANE_PACK_KINDS: [&str; 3] = ["sampled_real_q", "sampled_match", "greedy"];
pub(super) static LANE_PACK_LANES: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
pub fn lane_pack_lane_round(kind: usize) { LANE_PACK_LANES[kind].fetch_add(1, Relaxed); }
/// PACK4-REVIEW A3: 1 once a packed-round error turned the arm off for the rest of the boot.
pub(super) static LANE_PACK_DISABLED: AtomicU64 = AtomicU64::new(0);
pub fn lane_pack_disabled() { LANE_PACK_DISABLED.store(1, Relaxed); }
pub fn lane_pack_is_disabled() -> bool { LANE_PACK_DISABLED.load(Relaxed) != 0 }
/// One round of arm `arm` (index into LANE_PACK_ARMS) ran.
pub fn lane_pack_round(arm: usize) { LANE_PACK_ROUNDS[arm].fetch_add(1, Relaxed); }
pub fn lane_pack_rounds(arm: usize) -> u64 { LANE_PACK_ROUNDS[arm].load(Relaxed) }
/// One lane round ran outside a pack for reason index `i` (lane_pack::Skip::index).
pub fn lane_pack_skip(i: usize) { LANE_PACK_SKIP[i].fetch_add(1, Relaxed); }
pub fn lane_pack_skips(i: usize) -> u64 { LANE_PACK_SKIP[i].load(Relaxed) }
/// Lane capacity as set at boot (`set_max_batch`).
pub fn max_batch() -> u64 { MAX_BATCH.load(Relaxed) }
/// S-C4PRIMEA: shared-step primes served by the BATCHED single-launch path
/// (xq_gather_tap_rows + one re-prime pass + xq_attn_kv_prefill_pairs), and
/// the lanes those steps carried. 0 with --exl3-prime-batch off.
pub(super) static PRIME_BATCH: AtomicU64 = AtomicU64::new(0);
pub(super) static PRIME_BATCH_LANES: AtomicU64 = AtomicU64::new(0);
pub fn prime_batch_round(lanes: u64) {
    PRIME_BATCH.fetch_add(1, Relaxed);
    PRIME_BATCH_LANES.fetch_add(lanes, Relaxed);
}
pub fn prime_batch_rounds() -> u64 { PRIME_BATCH.load(Relaxed) }
pub fn prime_batch_lanes() -> u64 { PRIME_BATCH_LANES.load(Relaxed) }

/// S-C4XOVER: the spec-lanes auto policy (--spec-lanes-max auto, the default under --mtp auto)
/// made legible: the two arms' estimated AGGREGATE tok/s at the last measured decision, the
/// learned shared-step scale, the busy-lane count at that decision, how many decisions compared
/// real estimates, and how often the pick FLIPPED (thrash / stuck detection). Observability only.
pub const SPEC_LANES_TO: [&str; 2] = ["shared", "serial"];
pub(super) static SPEC_LANES_SWITCHES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
pub(super) static SPEC_LANES_DECISIONS: AtomicU64 = AtomicU64::new(0);
pub(super) static SPEC_LANES_EST_SERIAL: AtomicU64 = AtomicU64::new(0); // f64 bits: est serial aggregate tok/s
pub(super) static SPEC_LANES_EST_SHARED: AtomicU64 = AtomicU64::new(0); // f64 bits: est shared aggregate tok/s
pub(super) static SPEC_LANES_SCALE_MILLI: AtomicU64 = AtomicU64::new(0); // learned shared_scale x1000
pub(super) static SPEC_LANES_BUSY: AtomicU64 = AtomicU64::new(0);       // busy lanes at the last decision
pub(super) static SPEC_ARM_MS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];   // S-C4XOVER: cumulative ms each arm ran
pub(super) static SPEC_ARM_TOK: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];  // ... and the tokens it delivered
pub fn spec_lanes_switch(to: usize) { SPEC_LANES_SWITCHES[to].fetch_add(1, Relaxed); }
pub fn spec_arm_charge(arm: usize, ms: f64, tok: f64) {
    SPEC_ARM_MS[arm].fetch_add(ms as u64, Relaxed);
    SPEC_ARM_TOK[arm].fetch_add(tok as u64, Relaxed);
}
pub fn spec_lanes_decision(serial_tps: f64, shared_tps: f64, scale: f64, busy: usize) {
    SPEC_LANES_EST_SERIAL.store(serial_tps.to_bits(), Relaxed);
    SPEC_LANES_EST_SHARED.store(shared_tps.to_bits(), Relaxed);
    SPEC_LANES_SCALE_MILLI.store((scale * 1000.0) as u64, Relaxed);
    SPEC_LANES_BUSY.store(busy as u64, Relaxed);
    SPEC_LANES_DECISIONS.fetch_add(1, Relaxed);
}
pub(super) static SPEC_LANES_MEASURED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)]; // f64 bits: measured tok/s
pub(super) static SPEC_LANES_PROBES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// S-C4XOVER: the arm's MEASURED delivered aggregate tok/s (EWMA over its last windows) and
/// the arm-rate probe counter. arm: 0 = shared plain step, 1 = serial speculation.
static SPEC_METER_TP_SYNC: AtomicU64 = AtomicU64::new(0); // ULTRA: v2 metered steps whose ms was taken from the TP head
static SPEC_METER_REPLACE: AtomicU64 = AtomicU64::new(0); // ULTRA: v2 probe folds that replaced a contradicted rate
pub fn spec_meter_replace() { SPEC_METER_REPLACE.fetch_add(1, Relaxed); }
pub fn spec_meter_replaces() -> u64 { SPEC_METER_REPLACE.load(Relaxed) }
pub fn spec_meter_tp_sync() { SPEC_METER_TP_SYNC.fetch_add(1, Relaxed); }
pub fn spec_meter_tp_syncs() -> u64 { SPEC_METER_TP_SYNC.load(Relaxed) }
pub fn spec_lanes_measured(arm: usize, tps: f64) { SPEC_LANES_MEASURED[arm].store(tps.to_bits(), Relaxed); }
pub fn spec_lanes_probe(arm: usize) { SPEC_LANES_PROBES[arm].fetch_add(1, Relaxed); }
