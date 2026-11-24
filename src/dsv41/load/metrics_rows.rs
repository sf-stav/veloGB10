//! DSV4.1 loader (leg LS): the `/metrics` rows of the load path — engagement receipts (AGENTS §1b-bis.2): a
//! counter that never moves is a path that never ran. Rendered by `metrics::render` through one line; the
//! rows exist only after a dsv41 load / ship / fit ran in this process, so every other family's `/metrics`
//! is byte-identical.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

static RAN: AtomicU64 = AtomicU64::new(0);
static PLANNED_WORLD: AtomicU64 = AtomicU64::new(0);
static PLANNED_RANK_BYTES: AtomicU64 = AtomicU64::new(0);
static ENGRAM_PACKED: AtomicU64 = AtomicU64::new(0);
static ENGRAM_STREAMED: AtomicU64 = AtomicU64::new(0);
static FIT_CAPS: AtomicU64 = AtomicU64::new(0);
static FIT_LANES_ASKED: AtomicU64 = AtomicU64::new(0);
static FIT_LANES_SERVED: AtomicU64 = AtomicU64::new(0);
static FIT_CTX_ASKED: AtomicU64 = AtomicU64::new(0);
static FIT_CTX_SERVED: AtomicU64 = AtomicU64::new(0);
static FIT_USED: AtomicU64 = AtomicU64::new(0);
static FIT_BUDGET: AtomicU64 = AtomicU64::new(0);
static STAGE_WARM: AtomicU64 = AtomicU64::new(0);
static STAGE_COLD: AtomicU64 = AtomicU64::new(0);
static STAGE_SKIPPED: AtomicU64 = AtomicU64::new(0);

/// A shard plan was built (the head, per rank).
pub fn planned(world: usize, _rank: usize, bytes: u64) { RAN.store(1, Relaxed); PLANNED_WORLD.store(world as u64, Relaxed); PLANNED_RANK_BYTES.store(bytes, Relaxed); }
pub fn engram_heads_packed(n: u64) { RAN.store(1, Relaxed); ENGRAM_PACKED.fetch_add(n, Relaxed); }
/// A head file was packed and streamed to a node without being written here (LS2).
pub fn engram_heads_streamed(n: u64) { RAN.store(1, Relaxed); ENGRAM_STREAMED.fetch_add(n, Relaxed); }
/// The memory-fit result of this boot.
pub fn fit(asked_lanes: usize, lanes: usize, asked_ctx: usize, ctx: usize, caps: usize, used: u64, budget: u64) {
    RAN.store(1, Relaxed);
    FIT_CAPS.store(caps as u64, Relaxed);
    FIT_LANES_ASKED.store(asked_lanes as u64, Relaxed); FIT_LANES_SERVED.store(lanes as u64, Relaxed);
    FIT_CTX_ASKED.store(asked_ctx as u64, Relaxed); FIT_CTX_SERVED.store(ctx as u64, Relaxed);
    FIT_USED.store(used, Relaxed); FIT_BUDGET.store(budget, Relaxed);
}
pub fn stage(outcome: super::stage::Outcome) {
    RAN.store(1, Relaxed);
    match outcome { super::stage::Outcome::Warm => STAGE_WARM.fetch_add(1, Relaxed), super::stage::Outcome::Cold => STAGE_COLD.fetch_add(1, Relaxed), super::stage::Outcome::Skipped => STAGE_SKIPPED.fetch_add(1, Relaxed) };
}

/// Append the rows (nothing when no dsv41 load ran).
pub fn append(o: &mut String) {
    if RAN.load(Relaxed) == 0 { return; }
    let g = |a: &AtomicU64| a.load(Relaxed);
    o.push_str("# HELP velogb10_dsv41_load_shard_plan_world World size of the last dsv41 shard plan built in this process; velogb10_dsv41_load_shard_plan_bytes = the bytes of its last rank's shard dir (shard + Engram heads)\n");
    o.push_str(&format!("velogb10_dsv41_load_shard_plan_world {}\nvelogb10_dsv41_load_shard_plan_bytes {}\n", g(&PLANNED_WORLD), g(&PLANNED_RANK_BYTES)));
    o.push_str("# HELP velogb10_dsv41_load_engram_heads_packed_total Engram head files this process packed from the native checkpoint because they were missing (automatic Engram)\n");
    o.push_str(&format!("velogb10_dsv41_load_engram_heads_packed_total {}\n", g(&ENGRAM_PACKED)));
    o.push_str("# HELP velogb10_dsv41_load_engram_heads_streamed_total Engram head files this process packed and streamed to a node without writing them on its own disk\n");
    o.push_str(&format!("velogb10_dsv41_load_engram_heads_streamed_total {}\n", g(&ENGRAM_STREAMED)));
    o.push_str("# HELP velogb10_dsv41_load_fit_caps Settings the memory-fit check lowered at boot (0 = the asked posture fits); asked / served lanes and context next to it; used / budget in bytes\n");
    o.push_str(&format!("velogb10_dsv41_load_fit_caps {}\nvelogb10_dsv41_load_fit_lanes_asked {}\nvelogb10_dsv41_load_fit_lanes_served {}\nvelogb10_dsv41_load_fit_ctx_asked {}\nvelogb10_dsv41_load_fit_ctx_served {}\nvelogb10_dsv41_load_fit_used_bytes {}\nvelogb10_dsv41_load_fit_budget_bytes {}\n",
                        g(&FIT_CAPS), g(&FIT_LANES_ASKED), g(&FIT_LANES_SERVED), g(&FIT_CTX_ASKED), g(&FIT_CTX_SERVED), g(&FIT_USED), g(&FIT_BUDGET)));
    o.push_str("# HELP velogb10_dsv41_load_stage_total The head's staged shard (fast reboot): warm = read from the local stage, cold = built now, skipped = no room / off\n");
    o.push_str(&format!("velogb10_dsv41_load_stage_total{{outcome=\"warm\"}} {}\nvelogb10_dsv41_load_stage_total{{outcome=\"cold\"}} {}\nvelogb10_dsv41_load_stage_total{{outcome=\"skipped\"}} {}\n",
                        g(&STAGE_WARM), g(&STAGE_COLD), g(&STAGE_SKIPPED)));
}
