//! Phase-9 A.2 — PER-WINDOW DECODE TELEMETRY (host-side, read-only).
//!
//! Why: the owner's client could not see TRUE alpha (accepted tokens per verify forward) and the
//! campaign had to reconstruct acceptance from the head's stderr log after the fact. This module
//! publishes the engine's own decode-window numbers to the HTTP status route so a client can ask
//! the server what it is actually doing, per window:
//!
//!   mode (mtp|dflash2), tp width, df2 block, chosen depth, accept@k (k=1..depth),
//!   yield (tokens emitted per verify forward), step p50/p90, samples.
//!
//! Design rules (deliberate, do not "improve" casually):
//!  * ZERO hot-path structure change: the counters are published at the SAME places the engine
//!    already prints its `[mtp]` / `[df2]` window meters (every 50 steps) plus one timestamp per
//!    step (`note_step`, ~50 ns against a 100 ms step). No scheduler, no kernel, no lock.
//!  * Lock-free: plain relaxed atomics; a reader can never block a decode step. The step ring is
//!    fixed-size (RING) and overwritten in place — a torn read is impossible for a single u32,
//!    and the p50 is computed at read time from whatever is in the ring.
//!  * Windows are CUMULATIVE since boot, exactly like the `[mtp]`/`[df2]` log lines, so the
//!    endpoint and the log can never disagree about acceptance.
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::OnceLock;
use std::time::Instant;

/// Number of per-step samples kept for the p50/p90 view.
pub const RING: usize = 256;
/// Max speculate depth the accept@k vector tracks.
pub const MAXK: usize = 8;
/// Max accept@k columns the BLOCK drafter (P14 DFlash v1) tracks — the lane proposes up to
/// `block-1` = 15 columns, wider than the MTP depth ceiling `MAXK`.
pub const MAXK_BLOCK: usize = 16;

static STEP_US: [AtomicU32; RING] = [const { AtomicU32::new(0) }; RING];
static RING_POS: AtomicUsize = AtomicUsize::new(0);
static RING_N: AtomicU64 = AtomicU64::new(0);

// MTP window
static M_STEPS: AtomicU64 = AtomicU64::new(0);
static M_DRAFTS: AtomicU64 = AtomicU64::new(0);
static M_ACCEPTED: AtomicU64 = AtomicU64::new(0);
static M_EMITTED: AtomicU64 = AtomicU64::new(0);
static M_VERIFY: AtomicU64 = AtomicU64::new(0);
static M_DEPTH: AtomicU64 = AtomicU64::new(0);
static M_ACC: [AtomicU64; MAXK] = [const { AtomicU64::new(0) }; MAXK];
static M_ACCN: [AtomicU64; MAXK] = [const { AtomicU64::new(0) }; MAXK];

// DFlash2 window
static D_STEPS: AtomicU64 = AtomicU64::new(0);
static D_DRAFTS: AtomicU64 = AtomicU64::new(0);
static D_ACCEPTED: AtomicU64 = AtomicU64::new(0);
static D_EMITTED: AtomicU64 = AtomicU64::new(0);

// P14 DFlash v1 (BLOCK drafter) window. The lane verifies the artifact's whole block in one go
// (16 columns on 3.6-35B), so it has a real per-column acceptance profile — tracked to MAXK_BLOCK,
// not MAXK.
static DF_STEPS: AtomicU64 = AtomicU64::new(0);
static DF_DRAFTS: AtomicU64 = AtomicU64::new(0);
static DF_ACCEPTED: AtomicU64 = AtomicU64::new(0);
static DF_EMITTED: AtomicU64 = AtomicU64::new(0);
static DF_BLOCK: AtomicU64 = AtomicU64::new(0);
static DF_ACC: [AtomicU64; MAXK_BLOCK] = [const { AtomicU64::new(0) }; MAXK_BLOCK];
static DF_ACCN: [AtomicU64; MAXK_BLOCK] = [const { AtomicU64::new(0) }; MAXK_BLOCK];

// Boot context + active mode
static TP_WORLD: AtomicU64 = AtomicU64::new(0);
static DF2_BLOCK: AtomicU64 = AtomicU64::new(0);
// DFlash2 TREE window (the fork-then-chain lane's own counters; it has no drafts/emitted meters)
static DT_STEPS: AtomicU64 = AtomicU64::new(0);
static DT_NODES: AtomicU64 = AtomicU64::new(0);
static DT_RESCUES: AtomicU64 = AtomicU64::new(0);
/// 0 = no speculation yet, 1 = mtp, 2 = dflash2, 3 = dflash2-tree, 4 = dflash (P14 v1 block lane)
static MODE: AtomicU64 = AtomicU64::new(0);
static UPDATED_MS: AtomicU64 = AtomicU64::new(0);

fn now_ms() -> u64 {
    static T0: OnceLock<Instant> = OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_millis() as u64
}

static LAST_STEP: AtomicU64 = AtomicU64::new(0);

/// Boot context, called once when the TP config is installed.
pub fn set_boot(tp_world: usize, df2_block: usize) {
    TP_WORLD.store(tp_world as u64, Relaxed);
    DF2_BLOCK.store(df2_block as u64, Relaxed);
}

fn ring_push_us(us: u64) {
    let i = RING_POS.fetch_add(1, Relaxed) % RING;
    STEP_US[i].store(us.min(u32::MAX as u64) as u32, Relaxed);
    RING_N.fetch_add(1, Relaxed);
}

/// One MTP lane step boundary: pushes the inter-step interval (the decode step time).
#[inline]
pub fn note_step() {
    let t = now_ms();
    let prev = LAST_STEP.swap(t, Relaxed);
    if prev != 0 && t > prev {
        ring_push_us((t - prev) * 1000);
    }
}

/// DF2 lane step: the engine already measured this step (`step_t0`), so take it as given.
#[inline]
pub fn note_step_ms(ms: f32) {
    if ms.is_finite() && ms > 0.0 {
        ring_push_us((ms as f64 * 1000.0) as u64);
    }
}

/// Publish the MTP window (called where the `[mtp]` meter prints, i.e. every 50 steps).
pub fn publish_mtp(steps: u64, drafts: u64, accepted: u64, emitted: u64, verify: u64, depth: usize,
                   acc: &[(u64, u64)]) {
    M_STEPS.store(steps, Relaxed);
    M_DRAFTS.store(drafts, Relaxed);
    M_ACCEPTED.store(accepted, Relaxed);
    M_EMITTED.store(emitted, Relaxed);
    M_VERIFY.store(verify, Relaxed);
    M_DEPTH.store(depth as u64, Relaxed);
    for (k, &(a, n)) in acc.iter().enumerate().take(MAXK) {
        M_ACC[k].store(a, Relaxed);
        M_ACCN[k].store(n, Relaxed);
    }
    MODE.store(1, Relaxed);
    UPDATED_MS.store(now_ms(), Relaxed);
}

/// Publish the DFlash2 window (called where the `[df2]` meter prints, every 50 steps).
pub fn publish_df2(steps: u64, drafts: u64, accepted: u64, emitted: u64) {
    D_STEPS.store(steps, Relaxed);
    D_DRAFTS.store(drafts, Relaxed);
    D_ACCEPTED.store(accepted, Relaxed);
    D_EMITTED.store(emitted, Relaxed);
    MODE.store(2, Relaxed);
    UPDATED_MS.store(now_ms(), Relaxed);
}

/// Publish the DFlash2-TREE window (called where the `[df2-tree]` meter prints, every 50 steps).
/// The tree lane has no drafts/emitted meters of its own, and without this the status route
/// reported `mode: mtp` for tree-served lanes (the MTP window's samples were the only ones on the
/// ring) — assert on THIS mode, never on the MTP window.
pub fn publish_df2_tree(steps: u64, nodes: u64, rescues: u64) {
    DT_STEPS.store(steps, Relaxed);
    DT_NODES.store(nodes, Relaxed);
    DT_RESCUES.store(rescues, Relaxed);
    MODE.store(3, Relaxed);
    UPDATED_MS.store(now_ms(), Relaxed);
}

/// Publish the P14 DFlash v1 (block-drafter) window. Unlike MTP/DF2 this is called EVERY lane step
/// (the lane commits up to `block` tokens per step, so a request is ~6-16 steps and a 50-step
/// window would never print for a probe-length request). `acc` is column-indexed:
/// `acc[k] = (accepted_at_k, reached_k)`.
pub fn publish_dflash(steps: u64, drafts: u64, accepted: u64, emitted: u64, block: usize,
                      acc: &[(u64, u64)]) {
    DF_STEPS.store(steps, Relaxed);
    DF_DRAFTS.store(drafts, Relaxed);
    DF_ACCEPTED.store(accepted, Relaxed);
    DF_EMITTED.store(emitted, Relaxed);
    DF_BLOCK.store(block as u64, Relaxed);
    for (k, &(a, n)) in acc.iter().enumerate().take(MAXK_BLOCK) {
        DF_ACC[k].store(a, Relaxed);
        DF_ACCN[k].store(n, Relaxed);
    }
    MODE.store(4, Relaxed);
    UPDATED_MS.store(now_ms(), Relaxed);
}

fn ring_pct(v: &mut [u32], p: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_unstable();
    let k = (v.len() - 1) as f64 * p;
    let lo = k.floor() as usize;
    let hi = (lo + 1).min(v.len() - 1);
    Some(v[lo] as f64 + (v[hi] as f64 - v[lo] as f64) * (k - lo as f64))
}

// EXL3 engine: per-request speculation totals, added when a request's [mtp-stats] line prints.
static X_ROUNDS: AtomicU64 = AtomicU64::new(0);
static X_DRAFTS: AtomicU64 = AtomicU64::new(0);
static X_ACCEPTED: AtomicU64 = AtomicU64::new(0);

/// EXL3: add one finished request's speculation counts (rounds, drafted, accepted).
pub fn add_spec(rounds: u64, drafted: u64, accepted: u64) {
    X_ROUNDS.fetch_add(rounds, Relaxed);
    X_DRAFTS.fetch_add(drafted, Relaxed);
    X_ACCEPTED.fetch_add(accepted, Relaxed);
}

/// Cumulative speculation totals across every source (MTP, DFlash2, DFlash v1, EXL3 MTP):
/// (rounds, drafted, accepted, emitted). Feeds `/metrics`.
pub fn spec_totals() -> (u64, u64, u64, u64) {
    let rounds = M_VERIFY.load(Relaxed) + D_STEPS.load(Relaxed) + DF_STEPS.load(Relaxed) + X_ROUNDS.load(Relaxed);
    let drafts = M_DRAFTS.load(Relaxed) + D_DRAFTS.load(Relaxed) + DF_DRAFTS.load(Relaxed) + X_DRAFTS.load(Relaxed);
    let accepted = M_ACCEPTED.load(Relaxed) + D_ACCEPTED.load(Relaxed) + DF_ACCEPTED.load(Relaxed) + X_ACCEPTED.load(Relaxed);
    let emitted = M_EMITTED.load(Relaxed) + D_EMITTED.load(Relaxed) + DF_EMITTED.load(Relaxed)
        + X_ACCEPTED.load(Relaxed) + X_ROUNDS.load(Relaxed);
    (rounds, drafts, accepted, emitted)
}

/// Read-side snapshot for the status route. Never blocks a decode step.
pub fn snapshot_json() -> serde_json::Value {
    let n = (RING_N.load(Relaxed) as usize).min(RING);
    let mut v: Vec<u32> = (0..RING).map(|i| STEP_US[i].load(Relaxed)).filter(|&x| x > 0).collect();
    v.truncate(n.max(v.len().min(RING)));
    let (p50, p90) = {
        let mut c = v.clone();
        (ring_pct(&mut c, 0.50), ring_pct(&mut c, 0.90))
    };
    let mode = match MODE.load(Relaxed) {
        1 => "mtp",
        2 => "dflash2",
        3 => "dflash2-tree",
        4 => "dflash",
        _ => "none",
    };
    let m_steps = M_STEPS.load(Relaxed);
    let d_steps = D_STEPS.load(Relaxed);
    let mut ak = serde_json::Map::new();
    let depth = M_DEPTH.load(Relaxed) as usize;
    for k in 0..MAXK.min(depth.max(1)) {
        let a = M_ACC[k].load(Relaxed);
        let cnt = M_ACCN[k].load(Relaxed);
        if cnt > 0 {
            ak.insert(format!("{}", k + 1),
                      serde_json::json!({"acc_pct": (a as f64 / cnt as f64) * 100.0, "n": cnt}));
        }
    }
    let yield_mtp = if M_VERIFY.load(Relaxed) > 0 {
        Some(M_EMITTED.load(Relaxed) as f64 / M_VERIFY.load(Relaxed) as f64)
    } else {
        None
    };
    let yield_df2 = if d_steps > 0 {
        Some(D_EMITTED.load(Relaxed) as f64 / d_steps as f64)
    } else {
        None
    };
    // P14 DFlash v1 window (block lane).
    let df_steps = DF_STEPS.load(Relaxed);
    let slope = DF_BLOCK.load(Relaxed) as usize;
    let mut df_ak = serde_json::Map::new();
    for k in 0..MAXK_BLOCK.min(slope.max(1)) {
        let a = DF_ACC[k].load(Relaxed);
        let cnt = DF_ACCN[k].load(Relaxed);
        if cnt > 0 {
            df_ak.insert(format!("{}", k + 1),
                         serde_json::json!({"acc_pct": (a as f64 / cnt as f64) * 100.0, "n": cnt}));
        }
    }
    let yield_dflash = if df_steps > 0 {
        Some(DF_EMITTED.load(Relaxed) as f64 / df_steps as f64)
    } else {
        None
    };
    serde_json::json!({
        "mode": mode,
        "tp_width": TP_WORLD.load(Relaxed),
        "df2_block": DF2_BLOCK.load(Relaxed),
        "updated_ms": UPDATED_MS.load(Relaxed),
        "step_ms_p50": p50.map(|x| x / 1000.0),
        "step_ms_p90": p90.map(|x| x / 1000.0),
        "step_samples": v.len(),
        "mtp": {
            "steps": m_steps,
            "drafts": M_DRAFTS.load(Relaxed),
            "accepted": M_ACCEPTED.load(Relaxed),
            "accepted_pct": if M_DRAFTS.load(Relaxed) > 0 {
                (M_ACCEPTED.load(Relaxed) as f64 / M_DRAFTS.load(Relaxed) as f64) * 100.0
            } else { 0.0 },
            "emitted": M_EMITTED.load(Relaxed),
            "verify_fwds": M_VERIFY.load(Relaxed),
            "depth": depth,
            "accept_at_k": ak,
            "yield_tok_per_step": yield_mtp,
        },
        "dflash2": {
            "steps": d_steps,
            "drafts": D_DRAFTS.load(Relaxed),
            "accepted": D_ACCEPTED.load(Relaxed),
            "emitted": D_EMITTED.load(Relaxed),
            "yield_tok_per_step": yield_df2,
        },
        "df2_tree": {
            "steps": DT_STEPS.load(Relaxed),
            "nodes": DT_NODES.load(Relaxed),
            "rescues": DT_RESCUES.load(Relaxed),
            "avg_nodes": if DT_STEPS.load(Relaxed) > 0 {
                DT_NODES.load(Relaxed) as f64 / DT_STEPS.load(Relaxed) as f64
            } else { 0.0 },
            "b_rescue_pct": if DT_STEPS.load(Relaxed) > 0 {
                DT_RESCUES.load(Relaxed) as f64 / DT_STEPS.load(Relaxed) as f64 * 100.0
            } else { 0.0 },
        },
        // P14: the DFlash v1 BLOCK lane (source=dflash). Column-indexed accept@k (k = 1..block-1),
        // so `@1` is the anchor-column match rate and `@k` the rate at the k-th proposal GIVEN the
        // first k-1 matched — the same reading as the MTP meter, at block-1 columns instead of depth.
        "dflash": {
            "steps": df_steps,
            "block": DF_BLOCK.load(Relaxed),
            "drafts": DF_DRAFTS.load(Relaxed),
            "accepted": DF_ACCEPTED.load(Relaxed),
            "accepted_pct": if DF_DRAFTS.load(Relaxed) > 0 {
                (DF_ACCEPTED.load(Relaxed) as f64 / DF_DRAFTS.load(Relaxed) as f64) * 100.0
            } else { 0.0 },
            "emitted": DF_EMITTED.load(Relaxed),
            "verify_fwds": df_steps,
            "accept_at_k": df_ak,
            "yield_tok_per_step": yield_dflash,
        },
    })
}
