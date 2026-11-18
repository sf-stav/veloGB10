//! T2 definitions (PLAN/AUTOTUNE_DESIGN.md sections 6-8): the search targets, knob families, spec tables, prompts and the `Tuner` struct.
//! The `impl Tuner` blocks live in `tuner_plan.rs`, `tuner_race.rs` and `tuner_gate.rs` (see `mod.rs` for the file map).

use super::*;

// =============================================================================================
// T2 (PLAN/AUTOTUNE_DESIGN.md §6-§8, §11 T2): the search, the final gate and the table.
//
//   gb10_inference --autotune --model-dir <EXL3 pack> [served posture flags] [--autotune-budget 20]
//
// 1. refuse unless this is an idle .13/.14 (§6.4 / G-T8); boot the registry at the defaults;
//    load the model at the SERVED posture (--max-batch / --max-seq-len / --kv-cache /
//    --prefill-chunk / --draft-confidence, the server's own defaults) — the table's posture section
//    must equal what will be served or boot falls back.
// 2. probe: base-only SR rounds at every served verify width — the base unit medians (tau, budget)
//    and, per unit, which registry entries the capture consulted (§4.8a).
// 3. static pruning (§6.1): class S only, a proven domain (non-empty xcheck), no Load scope, no
//    structural / harness / umbrella entries, no env-overridden entry, consulted by some unit,
//    valid() values only.
// 4. dense sweep (§6.2): families in ledger-weighted order (ev / seconds), each a §6.3 race of
//    paired within-round SR differences (HL + Wilcoxon, tau, default preference, sequential
//    stopping, confirmation on fresh rounds), every sample digest-checked (§5.6) and every
//    candidate graph node-hashed (§4.8b: an identical graph leaves the race as INERT). Width
//    groups: race over the representative widths, confirm over every served width, split a width
//    back only when it is clearly slower there. Declared interacting pairs race as a joint grid.
//    Sweep 2 re-races the adopted families against the final base until nothing changes.
// 5. sparse point: the QSA families past the dense/sparse switch (a real long prompt, labelled).
// 6. prefill point: the chunk-class MT rule (the B3 known answer, §5.7.4) — snapshot-restored
//    chunk replays, paired per repetition, KV-row digests.
// 7. the final gate (§6.5): LI whole-round A/B (table vs defaults) in the served posture (DDS on,
//    the served calibrator + WP23 guard), 3 classes; the automatic hash gate (fixed chain AND
//    DDS: greedy tokens + rounds / drafted / accepted identical).
// 8. the table (§7.1): fingerprint (build, PTX, registry, model, GPU, driver, TP degree, clock
//    profile), served posture, provenance, decisions with evidence, family statuses, the gate.
//    Written only when the gate passes; a rejected table goes to the receipts directory only.
// =============================================================================================

/// What a family's candidates are measured on (§5.1 units).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Target {
    /// verify_dry replays -> Verify-family decisions (per width, or per family for a Fam knob)
    Verify,
    /// draft-chain replays -> DraftChain + DraftPass decisions (m = 1)
    Draft,
    /// both units summed per round (verify + draft x passes/round share) -> one Global / Shape value
    Round,
}

impl Target {
    pub(super) fn name(self) -> &'static str {
        match self { Target::Verify => "verify_dry", Target::Draft => "draft", Target::Round => "round(verify+draft)" }
    }
}

/// One searchable knob instance: a registry entry, optionally pinned to one key (a shape).
#[derive(Clone)]
pub(super) struct Knob {
    pub(super) def: &'static tune::TunableDef,
    pub(super) key: Option<tune::Key>,
    pub(super) label: String,
}

/// One family of the plan: one knob (a coordinate step) or a declared interacting pair (§6.2 joint
/// grid, <= 9 combinations). `cands` = value tuples (one per knob), the base tuple excluded.
#[derive(Clone)]
pub(super) struct Family {
    pub(super) name: String,
    pub(super) knobs: Vec<Knob>,
    pub(super) cands: Vec<Vec<i32>>,
    pub(super) regime: u8,
    pub(super) ev_ms: f64,
}

/// §6.2 ledger (dense regime): (registry id, expected realizable ms per round at the §10 central
/// estimate, shares of the m = 6 inventory). Everything else registered is listed NOT SEARCHED.
pub(super) const DENSE_SPECS: &[(&str, f64)] = &[
    ("moe.a1b3", 0.25), ("moe.coop", 0.15), ("moe.fh", 0.10),
    ("hc.rb", 0.15), ("wp11.r2", 0.20), ("wp11.r4", 0.10), ("wp11.cvt_prmt", 0.05), ("hc.mix4", 0.05),
    ("wp12.step", 0.30), ("wp12.ab", 0.07), ("wp12.diet", 0.05), ("wp12.conv", 0.05),
    ("lmh.nst", 0.25), ("lmh.had", 0.10), ("lmh.off", 0.10),
    ("wp10.fold", 0.10), ("wp10.gemv", 0.05), ("router.fused", 0.05),
    ("wp09.am_nb", 0.05), ("wp09.argmax", 0.03),
    ("attn.dense_v3", 0.10),
    ("wp27.forks", 0.30), ("wp27.prio", 0.10),
    // T0c (p5e): W4/HC geometry + levers (default on; every value bitwise)
    ("w4hc.g", 0.10), ("w4hc.pf", 0.08), ("w4hc.la", 0.05),
    ("w4hc.fuse", 0.05), ("w4hc.mix", 0.03), ("w4hc.inj", 0.05), ("w4hc.m1", 0.03),
    // W4/SMALL items (default on); w4s.gdn lives in verify_commit (LI only: INERT for the SR units)
    ("w4s.router", 0.05), ("w4s.dattn", 0.05), ("w4s.ple", 0.03), ("w4s.draft", 0.05), ("w4s.gdn", 0.03),
    // the opt-in packages' levers: consulted (so raced) only where the package is on — INERT otherwise
    ("w4dense.fix", 0.03), ("w4dense.suh", 0.03), ("w4dense.multi", 0.03), ("w4dense.silu", 0.03),
    ("w4dense.gsat", 0.05),
    ("w4moe.order", 0.05), ("w4moe.g", 0.05), ("w4moe.shovl", 0.05),
    // A5-K1: shared expert ‖ routed gate/up on the default WP20 path
    ("moe.shovl", 0.30),
    // A5-K2: router fold with coalesced weight staging
    ("moe.router_coal", 0.30),
    // A5-K7: xq_had_suh_multi folded into the WP20 gate/up prologue
    ("moe.gu_fold", 0.30),
];
/// §6.2 declared interacting pairs (joint grids): (a, a's values, b, b's values, ev).
pub(super) const PAIRS: &[(&str, &[i32], &str, &[i32], f64)] = &[
    ("wp20.mode", &[0, 1, 2], "wp20.diet", &[0, 1], 0.60),
    ("hc.bxr", &[32, 40, 48], "wp11.r1", &[0, 1, 2], 0.50),
    // p5e: PSK is opt-in (default off) — the package re-evaluated per width at each trellis prefetch
    // distance (psk.on = 0 with another pf is an identical graph: INERT, never timed)
    ("psk.on", &[0, 1], "psk.pf", &[0, 1, 2], 0.30),
    // T0c (p5e): the opt-in W4 packages re-evaluated in-round per width (their standalone estimates
    // lacked exactly this), each at its ring depths (an off package at another depth is INERT)
    ("w4dense.on", &[0, 1], "w4dense.nst", &[4, 6, 8], 0.30),
    ("w4moe.parts", &[0, 3, 7], "w4moe.nst", &[0, 4, 8], 0.30),
];
/// Sparse-regime families (S17-S19): searched at the sparse context point only.
pub(super) const SPARSE_SPECS: &[(&str, f64)] = &[
    ("qsa.score_mr_ctas", 0.30), ("qsa.sel_gather", 0.20), ("qsa.topk_v2", 0.10),
    ("qsa.score_generic", 0.05), ("qsa.sel_vec", 0.05),
    // T0c (p5e): the opt-in row-shared union gather (sparse verifies, m >= 2)
    ("w4qsa.on", 0.20),
    // A5-K3: latency-lean top-k select (asc3), sparse regime only
    ("attn.qsa_select", 0.30),
];
/// Per-shape PSK grid families (T4e): ev per shape instance.
pub(super) const PSK_EV_PER_SHAPE: f64 = 0.08;
/// Registered but never searched by default, with the reason (the report lists them).
pub(super) const NOT_SEARCHED: &[(&str, &str)] = &[
    ("qsa.score_mr", "structural: 0 brings back the bucket-keyed graph set (qsa_bucket_free)"),
    ("selftest.spin_ns", "harness: the self-test's known positive"),
    ("reprime.graph", "graph vs eager: the tuner measures graphs; eager is the diagnostic escape"),
    ("draft.graph", "graph vs eager: the tuner measures graphs; eager is the diagnostic escape"),
    ("wp27.graphs", "redundant with the per-family wp27.forks"),
    ("wp09.off", "umbrella escape: its parts (wp09.argmax, wp09.am_nb) are searched"),
    ("wp11.off", "umbrella escape: its parts (wp11.r1/r2/r4/cvt) are searched"),
    ("prefill.wide_mt", "searched at the prefill point (chunk classes)"),
    ("w4hc.off", "umbrella escape: its levers (w4hc.fuse/mix/inj/m1) and geometry (w4hc.g/pf/la) are searched"),
];
/// Share of a draft chain's replay delta that lands on a served DDS round (§10: ~3.5 passes per
/// round against the fixed chain's s passes): Round-target objective = dv + DRAFT_SHARE/s * dd.
pub(super) const DRAFT_PASSES_PER_ROUND: f64 = 3.5;

pub(super) const PY_PROMPT: &str = "Write a Python module that implements an LRU cache class with get, put and \
    delete methods, backed by a dict and a doubly linked list, with type hints, docstrings and a \
    small unittest suite at the end.\n\n```python\n";
pub(super) const SCIFI_PROMPT: &str = "Write the opening scene of a science fiction short story. A salvage crew \
    boards a derelict generation ship that has been drifting between stars for three hundred \
    years, and discovers that the ship's crew is still alive and has no idea how much time has \
    passed.\n\n";

/// One adopted decision (with the overlay keys it contributes to the base of later families).
#[derive(Clone)]
pub(super) struct Dec {
    pub(super) family: String,
    pub(super) target: Target,
    pub(super) knob: Knob,
    pub(super) value: i32,
    /// table decision scope(s) (one per family written: Draft fans out to DraftChain + DraftPass)
    pub(super) scopes: Vec<tune::DecScope>,
    pub(super) keys: Vec<tune::Key>,
    pub(super) evidence: serde_json::Value,
    /// the confirmed per-round gain (ms, negative = faster) the prediction check sums
    pub(super) gain_ms: f64,
}

/// The served DDS lane state the final gate mirrors (exl3_serve::step_mtp's DDS branch).
pub(super) struct DdsLane {
    pub(super) cal: crate::exl3_serve::DraftCal,
    pub(super) fit: crate::exl3_serve::CostFit,
    pub(super) ema_ms: f64,
    pub(super) ema_tok: f64,
    pub(super) rounds: usize,
    pub(super) ctr: u32,
    pub(super) drafted: usize,
    pub(super) accepted: usize,
}

impl DdsLane {
    pub(super) fn new() -> DdsLane {
        DdsLane { cal: crate::exl3_serve::DraftCal::new(), fit: Default::default(), ema_ms: 0.0, ema_tok: 0.0,
                  rounds: 0, ctr: 0, drafted: 0, accepted: 0 }
    }
}

/// One generation's outcome (hash gate).
#[derive(Clone, Debug, PartialEq)]
pub(super) struct GenOut {
    pub(super) tokens: Vec<u32>,
    pub(super) rounds: usize,
    pub(super) drafted: usize,
    pub(super) accepted: usize,
}

pub(super) struct Tuner<'a> {
    pub(super) model: &'a FwdModel,
    pub(super) head: &'a DraftHead,
    pub(super) sc: Scratch,
    pub(super) psc: Option<PrefillScratch>,
    pub(super) chunk: usize,
    pub(super) k: usize,
    pub(super) draft_conf: f64,
    pub(super) max_new: usize,
    pub(super) corpus: Vec<Vec<u32>>,
    pub(super) corpus_label: &'static str,
    pub(super) gen_next: usize,
    /// (b, p, tap_row, emitted) of the running tuning generation
    pub(super) gen: Option<(i32, usize, usize, usize)>,
    pub(super) round_ms: Vec<f64>,
    pub(super) widths_rep: Vec<usize>,
    pub(super) widths_all: Vec<usize>,
    /// base unit medians: ("verify", m, regime) / ("draft", s, regime)
    pub(super) base_ms: HashMap<(&'static str, usize, u8), f64>,
    pub(super) base_mad: HashMap<(&'static str, usize, u8), f64>,
    pub(super) decisions: Vec<Dec>,
    pub(super) families: serde_json::Map<String, serde_json::Value>,
    pub(super) jsonl: std::io::BufWriter<std::fs::File>,
    pub(super) t0: std::time::Instant,
    pub(super) budget_s: f64,
    pub(super) rcfg: tune::RaceCfg,
    pub(super) regime: u8,
    pub(super) gpu_log: Vec<serde_json::Value>,
    pub(super) throttle_events: usize,
    pub(super) mem_cap: usize,
    /// §5.2 v2: untimed base replays at the start of every block / generation (--autotune-warmup)
    pub(super) warmup: usize,
    /// §5.2 v2: remove the fitted position effect before pairing (--autotune-stratify, default on)
    pub(super) stratify: bool,
    /// v2: the prefill point's per-chunk balanced order cursors (persist across race calls)
    pub(super) pf_order: tune::BalancedOrder,
    /// the prefill point's B3 known answer: None = not run, Some(true) = HOLDS, Some(false) = DEVIATES
    pub(super) known_answer: Option<bool>,
}

pub(super) fn sm_clock(env: &serde_json::Value) -> Option<f64> {
    env.get("clocks.sm").and_then(|v| v.as_str()).and_then(|s| s.trim().parse::<f64>().ok())
}
pub(super) fn throttled(env: &serde_json::Value) -> bool {
    // clocks_event_reasons: 0x4 SW power cap, 0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal,
    // 0x80 HW power brake (0x1 idle / 0x2 app clocks / 0x100 sync boost are not throttling)
    env.get("clocks_event_reasons.active").and_then(|v| v.as_str())
        .and_then(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
        .map_or(false, |m| m & (0x4 | 0x8 | 0x20 | 0x40 | 0x80) != 0)
}
