//! S-B9-CF-PACK4 (CF-P1e LEG 4): the PURE half of `--lane-pack` — the flag, the reasons a lane
//! stays out of a pack, the pairing with its fair rotation, the row allocation and the packed
//! staging metadata. No GPU, no scheduler state: everything here is a function of its arguments,
//! so the unit suite pins the whole decision (AGENTS 1a-ter: the suite never builds a `Scratch`,
//! which is why the decision lives apart from the code that needs one).
//!
//! v1 = PAIRS. One packed speculative verify serves TWO requests (`PACK_K`); three busy
//! requests run one packed pair plus one lone round, the lone role rotating (see `plan`). k = 3
//! in one verify is not gated (PACK1: 0.59 ms per distinct expert at m = 15 against a 0.60 stop
//! line). Rows are EXACT: a pair of m0 + m1 rows runs at width R = m0 + m1 with no padding row.
//! The design (PLAN/CF_P1E_PACKED_VERIFY_DESIGN.md 1.3/1.4) rounds R up to even with masked
//! padding rows; the mask that would suppress a padding row's K/V and indexer-key writes
//! (`pad_of`) was never built — PACK1-3 never staged a padding row inside a verify — and such a
//! row skips the segmented GDN kernels, so its later writes would come from an un-mixed hidden.
//! Exact widths need no such row; the cost is 13 graph widths (4..=16) per regime instead of 7.
//!
//! S-B9-CF-PACK4S: SAMPLED lanes pack too (unpenalized; `--spec-sampling ratio`, the default,
//! and `match`). The sampler kernels read one 8-word block per ROW, so a sampled lane needs only
//! staging (`pack_samp`: its own parameters, seed and counters on its own rows, whatever its
//! partner is), the hand-off of its real-q draft lists to its row offset (`list_handoff`) and,
//! for a pair holding a real-q lane, the rq graph family. Pairs may be greedy + greedy,
//! sampled + sampled or mixed; a greedy lane's rows stay all-zero blocks (argmax) in every pair.
//!
//! File map (this was the single `lane_pack.rs`; every item keeps its path): `mod.rs` = the pure lane-pack decision (flag, reasons, pairing, row allocation, packed staging metadata; no GPU); `tests.rs` = the inline unit tests (module `tests`, same name).

use anyhow::Result;
use crate::exl3_forward::{SampParams, MTP_MAX_K};
use crate::wp24::RQ_ROW;

/// Requests per packed verify in v1.
pub(crate) const PACK_K: usize = 2;
/// The packed verify's row ceiling (the batch-invariance limit of the serving GEMM, AGENTS 2.4).
pub(crate) const PACK_ROWS_MAX: usize = 16;
/// The deepest draft chain one request brings to a pack: the lone verify's own ceiling
/// (`MTP_MAX_K`), lowered if a pair of such chains would not fit the packed verify's rows.
/// PACK4-REVIEW A4: this was a literal 8 beside `MTP_MAX_K + 1` with no link between them; a
/// raised `MTP_MAX_K` would have let a planned pair fail inside `pack_rows` AFTER both lanes
/// drafted (all lanes fail). The scheduler's planner and `step_pair` both clamp a lane's depth
/// to this, so a planned pair always fits.
pub(crate) const PACK_DEPTH_MAX: usize =
    if MTP_MAX_K + 1 <= PACK_ROWS_MAX / PACK_K { MTP_MAX_K } else { PACK_ROWS_MAX / PACK_K - 1 };
/// Rows one request may bring: its draft depth + 1, capped at the lone verify's width.
pub(crate) const PACK_SEG_ROWS_MAX: usize = PACK_DEPTH_MAX + 1;
const _: () = assert!(PACK_K * PACK_SEG_ROWS_MAX <= PACK_ROWS_MAX && PACK_SEG_ROWS_MAX <= MTP_MAX_K + 1,
                      "a pair of maximal segments must fit the packed verify and never exceed the lone verify's width");
/// The packed widths a boot precaptures: every R a pair can form (2 + 2 ..= 8 + 8).
pub(crate) const PACK_WIDTHS: std::ops::RangeInclusive<usize> = 4..=PACK_ROWS_MAX;

/// `--lane-pack off|on|2` (default ON since v0.7.4, owner ruling 2026-10-12; `on` == 2 in v1). The registry row
/// (`opts_table.rs`) holds the default; every reader goes through `LanePack::from_opts`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LanePack {
    /// The scheduler decision is master's, bit for bit: no packed scratch, no packed graph.
    Off,
    /// Two packable busy requests share one packed speculative verify.
    Pairs,
}

impl LanePack {
    pub(crate) fn parse(v: Option<&str>) -> Result<LanePack> {
        match v.map(str::trim) {
            None | Some("") | Some("off") => Ok(LanePack::Off),
            Some("on") | Some("2") => Ok(LanePack::Pairs),
            Some("3") => anyhow::bail!("--lane-pack 3: three requests in one verify are not built (v1 packs pairs: \
                                        the expert kernel costs 0.59 ms per distinct expert at 15 rows against a 0.60 stop line); use on or 2"),
            Some(other) => anyhow::bail!("--lane-pack must be off, on or 2 (got '{other}')"),
        }
    }

    /// The boot's mode as the REGISTRY resolves it: the value the operator typed (this head's command line, or the
    /// head's shipped snapshot on a node), else the row's default. One place states the default, so a head and a
    /// node of one binary always resolve the same mode without the flag having been typed (F10).
    pub(crate) fn from_opts() -> Result<LanePack> {
        LanePack::parse(Some(crate::opts::choice(crate::opt!("lane-pack")).as_str()))
    }

    pub(crate) fn on(self) -> bool { self == LanePack::Pairs }

    pub(crate) fn name(self) -> &'static str {
        match self { LanePack::Off => "off", LanePack::Pairs => "on (pairs)" }
    }
}

/// The largest TP world the packed arm runs at: 4 = TP=1, TP=2 and TP=4. Owner, 2026-10-10: the
/// TP=4 refusal is lifted ("we do not need refusals like that; we test before we release"). The
/// world > 2 hub frame (`TpSync::pre_verify_pack`) is built and unit-tested over a fake hub; NO packed
/// round has run on hardware at TP=2 or TP=4 yet. One definition for every check below and in
/// `xtp::world_gt2_flag_refusals`; above it there is no served topology.
pub(crate) const LANE_PACK_MAX_WORLD: u32 = 4;

/// `--lane-pack on` above `LANE_PACK_MAX_WORLD` is REFUSED at boot, on every rank. PACK4-REVIEW B1:
/// the refusal used to sit in `run_tp_*_serve`, AFTER discovery, the config ship and the RDMA link, so
/// `--tp 2 --lane-pack on` brought the whole cluster up and then both ranks exited. The pre-network
/// hook (`xtp::world_gt2_flag_refusals`: head before discovery, node before the link) now applies
/// the same rule first; this function is the rule for the entry points that already hold the world.
/// The world > 2 hub frame is built (`TpSync::pre_verify_pack` runs it) but never ran on four boxes.
pub(crate) fn tp_refusal(mode: LanePack, world: u32) -> Result<()> {
    anyhow::ensure!(!(mode.on() && world > LANE_PACK_MAX_WORLD), "--lane-pack runs at TP=1, TP=2 and TP=4 (TP={world} is \
                                         not a topology the packed arm has a frame for): start the head with \
                                         --lane-pack off");
    Ok(())
}

/// The packed verify's graph FAMILIES. The verify tail a graph records depends on the pair's
/// sampler state: under TP the greedy family ends in the vocabulary-parallel key exchange (no
/// logits rows exist), a sampled pair needs full logits rows (the row all-gather with
/// `--tp-vp-sampled`, else the replicated head) and a real-q pair the real-q sampler
/// (`xq_sample_rows_rq`). At TP=1 the Match family IS the Greedy one (same launches, so the same
/// key: nothing extra is captured).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PackFam {
    /// every row greedy: argmax ids (the PACK4 family).
    Greedy,
    /// a sampled lane under `--spec-sampling match`: rows sampled by `xq_sample_rows`.
    Match,
    /// a sampled lane under `--spec-sampling ratio` (the default): real-q rows.
    Rq,
}

impl PackFam {
    /// The (`samp_live`, `rq_live`) shadows of the packed scratch while this family's graphs are
    /// captured and replayed (what `vp_key_st` and the real-q key bit read).
    pub(crate) fn state(self) -> (bool, bool) {
        match self { PackFam::Greedy => (false, false), PackFam::Match => (true, false), PackFam::Rq => (true, true) }
    }
    pub(crate) fn rq(self) -> bool { self == PackFam::Rq }
    pub(crate) fn label(self) -> &'static str {
        match self { PackFam::Greedy => "greedy", PackFam::Match => "match-rule", PackFam::Rq => "real-q" }
    }
}

/// The families a boot holds: greedy always; with `sampled` (sampled lanes may pack on this
/// topology) the family the boot's sampling mode asks for — real-q under `--spec-sampling ratio`,
/// the match rule's own otherwise. A pure function of two resolved flags: both TP ranks derive
/// the same list in the same order.
pub(crate) fn fams(spec_ratio: bool, sampled: bool) -> Vec<PackFam> {
    let mut v = vec![PackFam::Greedy];
    if sampled { v.push(if spec_ratio { PackFam::Rq } else { PackFam::Match }); }
    v
}

/// The family a staged pair replays: any real-q lane -> Rq; else any sampled lane -> Match; else Greedy.
pub(crate) fn fam_of(samp_live: bool, rq_live: bool) -> PackFam {
    if rq_live { PackFam::Rq } else if samp_live { PackFam::Match } else { PackFam::Greedy }
}

/// PACK4-REVIEW B3: the boot postures `--lane-pack on` was never gated under, refused at boot with
/// the reason (the flag's help promises byte identity without a condition). Checked BEFORE the
/// model load, so a refused boot costs nothing (the `--lane-order fcfs` refusal used to fire after
/// the 414 MB packed scratch and the 26 captures).
pub(crate) struct Posture<'a> {
    /// the resolved `--kv-cache` name (q8 is the default; q8 and q4k are the admitted formats)
    pub kv: &'a str,
    /// the resolved speculation depth ceiling (`--exl3-mtp-k`; MTP_MAX_K is the default and the gated one)
    pub mtp_k: usize,
    /// per-row attention regime (`--qsa-row-regime`; off lets a round straddle the sparse switch)
    pub row_regime: bool,
    /// `--a-ctrl old`: a lane whose last step was a control round drafts from a clobbered `sc.resid`
    pub a_ctrl_old: bool,
    /// `--lane-order fcfs`: one lane per scheduler step; a pack steps two
    pub fcfs: bool,
    /// `--exl3-dense-v3=0`: `xq_attn_decode_group` resolves ONE slot for the whole group, so a
    /// packed verify's segment B would read and write segment A's K/V (q8 and q4k alike)
    pub dense_v3: bool,
    /// `--lane-pack-kv any`: packing may run on a KV format other than q8 (opt-in, ungated)
    pub kv_any: bool,
}

/// F1 (owner 2026-10-10, AGENTS §1b-bis item 8 — "support things properly or gracefully fall back"):
/// `--lane-pack on` never refuses a boot. Every option combination the packed arm was never gated
/// under becomes a FALLBACK: the server boots with packed rounds OFF for this boot, prints ONE plain
/// line (`lane packing (--lane-pack, on by default) is NOT ACTIVE on this boot: <reason>; requests are
/// served without packed rounds`), and the decision rides `/metrics` and the TP boot agree. Returns the plain-word
/// reasons (empty = this posture can pack).
/// `--lane-pack-kv q8|any` (default q8): `any` lets the packed arm run on an f16 / fp8 KV cache (ungated opt-in).
/// q4k is ADMITTED (owner 2026-11-13, PLAN/Q4K_PACK_WORK_2026-11-13.md): the default posture packs on it.
pub(crate) fn kv_any() -> bool {
    crate::opts::var(crate::opt!("lane-pack-kv")).as_deref() == Ok("any")
}

pub(crate) fn posture_fallbacks(p: &Posture) -> Vec<String> {
    let mut v = Vec::new();
    if p.fcfs {
        v.push("--lane-order fcfs: fcfs steps ONE lane per scheduler step, a pack steps two (the rr order steps both)".to_string());
    }
    // q4k is admitted (owner 2026-11-13, Option A on PLAN/Q4K_PACK_WORK_2026-11-13.md): its rows are
    // symmetric (144 B) and the packed arm reaches KV only through xq_kv_make, with attention on
    // xq_attn_decode_group (per-row slot, issue #35). Evidence: the pack2/packstate/binv receipts in
    // docs/KV_CACHE_Q4K.md §6 were genuine packed verifies on q4k — the gate only ever applied to the
    // serving boot, never the probe paths.
    if !matches!(p.kv, "q8" | "q4k") && !p.kv_any {
        v.push(format!("--kv-cache {}: the packed verify's per-segment attention and commit were never run on this KV format (gated: q8, q4k; \
                        --lane-pack-kv any runs them anyway, ungated)", p.kv));
    }
    if p.mtp_k != MTP_MAX_K {
        v.push(format!("--exl3-mtp-k {}: the packed widths and the control-cadence arithmetic were never run at another depth (gated: {MTP_MAX_K})", p.mtp_k));
    }
    if !p.row_regime {
        v.push("--qsa-row-regime 0: a round would straddle the sparse switch, whose regime then depends on the draft width the pack only learns after drafting".to_string());
    }
    if p.a_ctrl_old {
        v.push("--a-ctrl old: a lane whose last step was a control round drafts from a clobbered residual — acceptance moves, and the pair's control cadence arithmetic assumes the default".to_string());
    }
    if !p.dense_v3 {
        v.push("--exl3-dense-v3 0: xq_attn_decode_group resolves one KV slot for the whole group, so a packed verify's second segment would read and write the first lane's K/V (dense-v3 resolves a slot per row)".to_string());
    }
    v
}

/// F1: the boot's `--lane-pack` decision, recorded ONCE by `build_serve` (on every rank): `Some(why)`
/// when the resolved mode is on (F10: the default is on, no flag needs to be typed) — empty `why` =
/// active, non-empty = this boot serves without packed rounds and why; `None` = the mode is off. Every reason comes from shipped CLI options and
/// shipped model facts, so both TP ranks record the same word; `tp_boot_agree` hashes it so a
/// rank-local difference dies loudly at boot instead of one rank packing while its peer fell back.
static BOOT_PACK: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

/// F1: record the boot's lane-pack decision (see `BOOT_PACK`).
pub(crate) fn record_boot_pack(why: Option<String>) {
    let _ = BOOT_PACK.set(why);
}

/// F1: `Some(word)` when the resolved mode was on this boot (empty = active, non-empty = the
/// fallback reason); `None` when the mode is off (`--lane-pack off`). Read by the metrics render and `tp_boot_agree`.
pub(crate) fn boot_pack_word() -> Option<&'static String> {
    BOOT_PACK.get().and_then(|o| o.as_ref())
}

/// F10 (owner 2026-10-12: the default is on, so a boot must never die for the packed arm's sake): the one
/// place that turns a failed boot-time build of the packed scratch or of its graph family into a FALLBACK
/// reason. `build` is the real allocation + capture call (`pack_scratch`, `pack_precapture`, the family
/// check); `Ok` = the arm is built and the caller records the empty reason, `Err(why)` = this boot serves
/// without packed rounds (the half-built scratch is dropped by the caller) and `why` is the line's reason.
pub(crate) fn build_or_fall_back<T>(build: impl FnOnce() -> Result<T>) -> std::result::Result<T, String> {
    build().map_err(|e| format!("the packed scratch or its graphs could not be built at boot ({e:#})"))
}

/// Why a busy lane's round ran OUTSIDE a packed verify while `--lane-pack` was on and at least two
/// requests were busy. One per lane round (`velogb10_lane_pack_skip_total{reason=...}`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Skip {
    /// Fewer than two packable busy lanes: this packable lane has nobody to pack with.
    FewPackable,
    /// The odd lane of an odd-sized group this round (the rotation's lone role).
    OddLane,
    /// A prompt is mid-prefill (a `--prefill-interleave` step): the decoding lanes run serial.
    Prefill,
    /// The lane's MTP is off (the per-request profitability control): it rides the plain batch.
    MtpOff,
    /// The lane owes a control round (the cadence, or a pending re-probe confirm).
    Control,
    /// The lane is inside a re-probe window: its rounds are timed alone for the verdict.
    Probe,
    /// A sampled request on a boot that keeps sampled requests on the lone path: the sampler
    /// diagnostics read the LONE verify (`--wp24-dump`), or the boot runs under TP with
    /// `--lane-pack-sampled-tp off` (the default is ON: sampled requests join packed rounds under TP as
    /// on one box, PACK4S). Sampled requests pack otherwise.
    Sampled,
    /// A penalized request (repetition / presence / frequency): per-row penalty histories.
    Penalty,
    /// A request with images (per-slot 3-axis RoPE map): not proven for packed rows.
    Image,
    /// No draft fits this round (end of window, last token of the budget, the sparse switch).
    Depth0,
    /// The pair's lanes are on different sides of the sparse-attention switch (or grid buckets).
    MixedRegime,
    /// One lane's round is clamped at the sparse switch this round and its partner's is not.
    RegimeClamp,
    /// The boot holds no packed graph for this width and regime: never captured while serving.
    NoGraph,
    /// A request that reports log probabilities on a TP boot: its rows must be full logits rows and no
    /// packed graph family records them (the greedy family ends in the vocab-parallel key exchange).
    LogprobsTp,
}

impl Skip {
    pub(crate) const ALL: [Skip; 14] = [Skip::FewPackable, Skip::OddLane, Skip::Prefill, Skip::MtpOff, Skip::Control,
        Skip::Probe, Skip::Sampled, Skip::Penalty, Skip::Image, Skip::Depth0, Skip::MixedRegime, Skip::RegimeClamp, Skip::NoGraph,
        Skip::LogprobsTp];

    /// The metric label value.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Skip::FewPackable => "few_packable", Skip::OddLane => "odd_lane", Skip::Prefill => "prefill",
            Skip::MtpOff => "mtp_off", Skip::Control => "control", Skip::Probe => "probe", Skip::Sampled => "sampled",
            Skip::Penalty => "penalty", Skip::Image => "image", Skip::Depth0 => "depth0",
            Skip::MixedRegime => "mixed_regime", Skip::RegimeClamp => "regime_clamp", Skip::NoGraph => "no_graph",
            Skip::LogprobsTp => "logprobs_tp",
        }
    }

    /// The log line's reason text (`[exl3-serve] lane-pack skip: <text>`).
    pub(crate) fn text(self) -> &'static str {
        match self {
            Skip::FewPackable => "fewer than two packable busy lanes",
            Skip::OddLane => "odd lane of this round's rotation",
            Skip::Prefill => "a prompt is mid-prefill",
            Skip::MtpOff => "MTP off for this request",
            Skip::Control => "control round owed",
            Skip::Probe => "re-probe window",
            Skip::Sampled => "sampled request on a boot that keeps sampled requests on the lone path (--wp24-dump, or TP with --lane-pack-sampled-tp off)",
            Skip::Penalty => "penalized request",
            Skip::Image => "image request",
            Skip::Depth0 => "no draft fits this round",
            Skip::MixedRegime => "mixed regime",
            Skip::RegimeClamp => "round clamped at the sparse switch for one lane only",
            Skip::NoGraph => "no packed graph for this width",
            Skip::LogprobsTp => "log-probability request under TP (no packed family keeps full logits rows)",
        }
    }

    pub(crate) fn index(self) -> usize { Skip::ALL.iter().position(|&s| s == self).unwrap() }
}

/// What the scheduler knows about one busy lane before the round (all of it lane state).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LaneFacts {
    pub mtp_armed: bool,
    pub force_control: bool,
    pub probing: bool,
    /// A sampled request (temperature > 0 and top_k != 1).
    pub sampled: bool,
    /// This boot keeps sampled requests on the lone path (`--wp24-dump` records the lone
    /// verify's rows): false on every boot without that diagnostic.
    pub sampled_lone: bool,
    pub penalized: bool,
    pub image: bool,
    /// The lane reports log probabilities on a TP boot (`Skip::LogprobsTp`).
    pub logprobs_tp: bool,
    /// The round's draft depth ceiling after the window, budget and regime clamps (0 = plain step).
    pub depth: usize,
}

/// The lane-level reason a lane cannot join a pack (None = packable). Checked in this order.
pub(crate) fn lane_block(f: &LaneFacts) -> Option<Skip> {
    if !f.mtp_armed { return Some(Skip::MtpOff); }
    if f.force_control { return Some(Skip::Control); }
    if f.probing { return Some(Skip::Probe); }
    if f.sampled && f.sampled_lone { return Some(Skip::Sampled); }
    if f.penalized { return Some(Skip::Penalty); }
    if f.image { return Some(Skip::Image); }
    if f.logprobs_tp { return Some(Skip::LogprobsTp); }
    if f.depth == 0 { return Some(Skip::Depth0); }
    None
}

/// PACK4-REVIEW B4: one lane's view for the pairing, as a PURE function of what the scheduler reads
/// (the logic used to live inline in `Exl3Scheduler::pack_plan`, untested, although two of
/// `pack_verify`'s `ensure!`s rest on it). `kd_reg` = the round depth after the window, budget and
/// regime clamps (`round_depth`'s second value), `k_pre` = before the regime clamp, `qsa_live(p)` =
/// is position `p` on the sparse side of the attention switch, `ready[sparse]` = the boot holds the
/// packed family for that regime. Precedence of the reasons: a prefill in flight, then the lane's own
/// facts (`lane_block`), then a round that straddles the switch (`--qsa-row-regime 0` only), then a
/// missing graph. The regime is the verify's LAST row's (p + depth, at least p + 1).
pub(crate) fn lane_view(slot: usize, mut facts: LaneFacts, pos: usize, k_pre: usize, kd_reg: usize, in_prefill: bool,
                        ready: [bool; 2], qsa_live: &dyn Fn(usize) -> bool) -> LaneView {
    let kd = kd_reg.min(PACK_DEPTH_MAX); // PACK4-REVIEW A4: a segment never brings more than a pair can hold
    facts.depth = kd;
    let sparse = qsa_live(pos + kd.max(1));
    let straddles = kd >= 1 && qsa_live(pos + 1) != sparse;
    let block = if in_prefill { Some(Skip::Prefill) } else { lane_block(&facts) }
        .or(if straddles { Some(Skip::MixedRegime) } else { None })
        .or(if !ready[sparse as usize] { Some(Skip::NoGraph) } else { None });
    LaneView { slot, block, regime: sparse as u64, clamped: kd_reg < k_pre }
}

/// One busy lane as the pairing sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaneView {
    pub slot: usize,
    pub block: Option<Skip>,
    /// The lane's attention-regime class for this round (regime + scorer grid bucket): two lanes
    /// pack only when it is equal — the SAME predicate the lone round keys its verify graph on.
    pub regime: u64,
    /// This round's depth is clamped at the sparse switch (the QSAREGIME round clamp).
    pub clamped: bool,
}

/// One scheduler step's packing: `pairs` (slots, ascending inside a pair) and every other busy
/// lane with the reason it runs outside a pack.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub(crate) struct Plan {
    pub pairs: Vec<[usize; 2]>,
    pub lone: Vec<(usize, Skip)>,
}

impl Plan {
    /// The slot's partner and whether the slot is the pair's FIRST (lower) slot.
    pub(crate) fn pair_of(&self, slot: usize) -> Option<[usize; 2]> {
        self.pairs.iter().copied().find(|p| p[0] == slot || p[1] == slot)
    }
    pub(crate) fn skip_of(&self, slot: usize) -> Option<Skip> {
        self.lone.iter().find(|(s, _)| *s == slot).map(|(_, r)| *r)
    }
}

/// The step's plan, or None = take master's path untouched: the flag is off, or fewer than two
/// lanes are busy (a lone request never packs: same functions, scratch, graphs and kernels as
/// today — the single-request protection rule is this early return, not a check deep inside).
///
/// `lanes` = the busy lanes in slot order; `rot` = the scheduler's rotation counter (it advances
/// once per step that consulted a plan). Packable lanes (no `block`) are grouped by (regime,
/// clamped); a group of n >= 2 is rotated left by `rot % n` and paired off in that order, so with
/// an odd n the lone role moves to a different lane every step and every lane is the odd one
/// exactly once per n steps (three lanes: lone = c, a, b, c, ...). Every busy lane still gets
/// exactly one round per scheduler step, packed or not, so no lane can starve; the rotation
/// only spreads the odd lane's slightly dearer accounting (its round time is not shared).
/// A PURE function of lane state and `rot`: under TP every rank would form the same pairs.
pub(crate) fn plan(mode: LanePack, lanes: &[LaneView], rot: u64) -> Option<Plan> {
    if !mode.on() || lanes.len() < 2 { return None; }
    let mut out = Plan::default();
    for l in lanes {
        if let Some(why) = l.block { out.lone.push((l.slot, why)); }
    }
    let packable: Vec<&LaneView> = lanes.iter().filter(|l| l.block.is_none()).collect();
    // groups in order of their first lane (slot order)
    let mut groups: Vec<Vec<&LaneView>> = Vec::new();
    for l in &packable {
        match groups.iter_mut().find(|g| g[0].regime == l.regime && g[0].clamped == l.clamped) {
            Some(g) => g.push(l),
            None => groups.push(vec![l]),
        }
    }
    for g in &groups {
        let n = g.len();
        if n == 1 {
            let me = g[0];
            let why = if packable.len() == 1 {
                Skip::FewPackable
            } else if packable.iter().any(|o| o.slot != me.slot && o.regime == me.regime) {
                Skip::RegimeClamp // a same-regime lane exists; only the clamp differs
            } else {
                Skip::MixedRegime
            };
            out.lone.push((me.slot, why));
            continue;
        }
        let r0 = (rot % n as u64) as usize;
        let order: Vec<usize> = (0..n).map(|i| g[(r0 + i) % n].slot).collect();
        for c in order.chunks(2) {
            match c {
                [a, b] => out.pairs.push([*a.min(b), *a.max(b)]),
                [a] => out.lone.push((*a, Skip::OddLane)),
                _ => unreachable!(),
            }
        }
    }
    out.pairs.sort();
    out.lone.sort_by_key(|(s, _)| *s);
    Some(out)
}

/// The control cadence of a speculative round (step_mtp's WP03 rule, one definition): 1/1024
/// while this request's MTP cost is clearly below plain (< 0.8x), else 1/128. `adaptive` =
/// `--a-ctrl` is at its default.
pub(crate) fn ctrl_period(adaptive: bool, ema_plain: f64, lane_cost: f64) -> usize {
    if adaptive && ema_plain > 0.0 && lane_cost > 0.0 && lane_cost < 0.8 * ema_plain { 1024 } else { 128 }
}

/// Would packing this pair now swallow a control round? The serial arm advances the scheduler's
/// control counter once per lane round and runs a plain control round when it lands on the
/// lane's period. A packed pair advances it twice (first slot, then second), exactly as the two
/// serial rounds would, so the pair must not pack when either landing is a control: the two
/// lanes then run serial and the cadence is the serial arm's, count for count.
pub(crate) fn pair_owes_control(ctrl_round: usize, period_a: usize, period_b: usize) -> bool {
    (ctrl_round + 1) % period_a.max(1) == 0 || (ctrl_round + 2) % period_b.max(1) == 0
}

/// A packed round's wall time as ONE lane's round time: the pair's time divided by its lanes.
/// The lane's cost EMAs, its DDS cost fit and the per-request profitability control all read a
/// lane round time; fed the whole packed time they would see every round twice as dear as the
/// lane's share and the cost guard would mis-price a draft (design LEG 4 risk note).
pub(crate) fn lane_share_ms(round_ms: f64, lanes: usize) -> f64 {
    round_ms / lanes.max(1) as f64
}

/// Rows of a pair whose lanes verified `w0` and `w1` drafts: (m0, m1, R) with m_j = w_j + 1 (the
/// bonus row), each at most 8, R = m0 + m1 at most 16. None when a lane has no draft (a plain
/// step is not a verify) or a width is out of range.
pub(crate) fn pack_rows(w0: usize, w1: usize) -> Option<(usize, usize, usize)> {
    if w0 == 0 || w1 == 0 { return None; }
    let (m0, m1) = (w0 + 1, w1 + 1);
    if m0 > PACK_SEG_ROWS_MAX || m1 > PACK_SEG_ROWS_MAX || m0 + m1 > PACK_ROWS_MAX { return None; }
    Some((m0, m1, m0 + m1))
}

/// One request's half of a packed verify: its slot, the bonus token `b` at position `p`, and its
/// verified draft chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SegIn<'a> {
    pub slot: usize,
    pub b: i32,
    pub p: usize,
    pub drafts: &'a [i32],
}

/// The packed verify's staging, as the device kernels read it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PackMeta {
    /// (slot, token, position) per row: segment 0's rows, then segment 1's. Exactly R rows.
    pub rows: Vec<(usize, i32, usize)>,
    /// `{slot, row_start, len, flags}` per segment; flags bit 0 = nostore (the verify never
    /// writes live recurrent state: the per-segment commit replays rows 0..=a from live).
    pub segs: [[i32; 4]; PACK_K],
    /// Draft ids at their segment's row offset (`xq_accept_seg` compares d[row0 + i] with the
    /// verify's row row0 + i for i < len - 1); `d_len` words, zero elsewhere.
    pub d_dev: Vec<i32>,
    pub r: usize,
}

/// Build the staging of one packed pair. Segment j occupies rows row0_j .. row0_j + m_j with
/// m_j = drafts + 1: row 0 carries `b`, row i carries draft i - 1, positions p .. p + m_j - 1.
/// No row past R is staged (see the module note on padding rows).
pub(crate) fn pack_meta(s0: &SegIn, s1: &SegIn, d_len: usize) -> Result<PackMeta> {
    let (m0, m1, r) = pack_rows(s0.drafts.len(), s1.drafts.len())
        .ok_or_else(|| anyhow::anyhow!("lane-pack: a pair of {} + {} drafts is not packable", s0.drafts.len(), s1.drafts.len()))?;
    anyhow::ensure!(s0.slot != s1.slot, "lane-pack: a pair needs two slots (both are slot {})", s0.slot);
    anyhow::ensure!(d_len >= r, "lane-pack: draft buffer of {d_len} words is narrower than {r} rows");
    let mut rows = Vec::with_capacity(r);
    for s in [s0, s1] {
        for i in 0..=s.drafts.len() {
            rows.push((s.slot, if i == 0 { s.b } else { s.drafts[i - 1] }, s.p + i));
        }
    }
    let segs = [[s0.slot as i32, 0, m0 as i32, 1], [s1.slot as i32, m0 as i32, m1 as i32, 1]];
    let mut d_dev = vec![0i32; d_len];
    d_dev[..s0.drafts.len()].copy_from_slice(s0.drafts);
    d_dev[m0..m0 + s1.drafts.len()].copy_from_slice(s1.drafts);
    Ok(PackMeta { rows, segs, d_dev, r })
}

/// One segment's result out of the packed readback: `accept_out[row0 .. row0 + len]` holds the
/// emitted tokens at i <= a and -1 after; `acc_seg[j]` = {slot, a}. Returns (a, emitted).
pub(crate) fn seg_result(accept_out: &[i32], acc_seg: &[i32], j: usize, seg: &[i32; 4]) -> Result<(usize, Vec<i32>)> {
    let (slot, row0, len) = (seg[0], seg[1] as usize, seg[2] as usize);
    anyhow::ensure!(acc_seg.len() >= j * 2 + 2 && accept_out.len() >= row0 + len, "lane-pack: readback narrower than segment {j}");
    anyhow::ensure!(acc_seg[j * 2] == slot, "lane-pack: segment {j} committed slot {} but staged slot {slot}", acc_seg[j * 2]);
    let a = acc_seg[j * 2 + 1];
    anyhow::ensure!(a >= 0 && (a as usize) < len, "lane-pack: segment {j} accept {a} out of range (rows {len})");
    let a = a as usize;
    let emitted = accept_out[row0..=row0 + a].to_vec();
    anyhow::ensure!(emitted.iter().all(|&t| t >= 0), "lane-pack: segment {j} emitted a negative id {emitted:?}");
    Ok((a, emitted))
}

// ---- S-B9-CF-PACK4S: sampled lanes in a packed verify ----

/// One request's sampler as a packed verify stages it: what the lone round stages for it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SegSamp {
    /// The lane's sampler at its CURRENT counter (`Lane::samp_rows(1)[0]`); None = a greedy lane.
    pub samp: Option<SampParams>,
    /// The lane's draft rows are real-q rows (`--spec-sampling ratio`: its `RqStage` is mode 2).
    pub ratio: bool,
}

impl SegSamp {
    pub(crate) const GREEDY: SegSamp = SegSamp { samp: None, ratio: false };
}

/// The packed verify's sampler block and the two host shadows that follow from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PackSamp {
    /// `rows * 8` u32, row r at `r * 8` (`SampParams::pack_flag`); an all-zero row is greedy.
    pub words: Vec<u32>,
    /// Some row is sampled (`Scratch::samp_live`).
    pub samp_live: bool,
    /// Some row is a real-q row: the verify replays the rq graph family (`Scratch::rq_live`).
    pub rq_live: bool,
}

/// Sampler staging of one packed pair (`segs` = `PackMeta::segs`, `samp[j]` = segment j's lane).
/// Segment j's row i (i in 0..len_j) carries lane j's parameters and seed with counter
/// `ctr_j + i`: exactly the counter the lone verify gives that lane's row i, so a lane's draws
/// are a function of its own (seed, counter) and never of its partner or of its row offset.
/// Flag: 2 on a real-q lane's DRAFT rows (i < len_j - 1: `xq_sample_rows_rq` judges d_i under
/// min(1, p / q') against list row row0_j + i) and 1 on its bonus row; 1 on every row of a
/// match-mode sampled lane (`xq_sample_rows`' own draw); a greedy lane's rows stay all-zero
/// (both kernels return on flag 0 and the argmax id stands). The lone round stages flag 2 on
/// all k + 1 rows and lets the launch argument `nratio = w` select the draft rows; here the
/// flag itself is the selector, because segment 0's bonus row sits below the packed `nratio`.
pub(crate) fn pack_samp(segs: &[[i32; 4]; PACK_K], samp: &[SegSamp; PACK_K], rows: usize) -> Result<PackSamp> {
    let mut words = vec![0u32; rows * 8];
    let (mut samp_live, mut rq_live) = (false, false);
    for (sg, s) in segs.iter().zip(samp.iter()) {
        let (row0, len) = (sg[1] as usize, sg[2] as usize);
        anyhow::ensure!(sg[1] >= 0 && len >= 2 && row0 + len <= rows, "lane-pack: sampler rows {row0}..{} outside {rows} staged rows", row0 + len);
        let Some(p) = s.samp else {
            anyhow::ensure!(!s.ratio, "lane-pack: a greedy lane (slot {}) cannot stage real-q rows", sg[0]);
            continue;
        };
        samp_live = true;
        rq_live |= s.ratio;
        for i in 0..len {
            let flag = if s.ratio && i + 1 < len { 2 } else { 1 };
            let row = SampParams { counter: p.counter.wrapping_add(i as u32), ..p };
            words[(row0 + i) * 8..(row0 + i) * 8 + 8].copy_from_slice(&row.pack_flag(flag));
        }
    }
    Ok(PackSamp { words, samp_live, rq_live })
}

/// Rows of `Scratch::rq_list` (one `RQ_ROW`-word row per draft pass) for a scratch of `w`
/// rows. The served scratch needs one per draft pass (`MTP_MAX_K + 1` with the spare the
/// speculative pass may write); a packed verify reads list row `row0_j + i` for every real-q
/// draft row, up to row 14 of 16, so the list follows the scratch width. Unchanged for every
/// scratch of at most 8 rows.
pub(crate) fn rq_list_rows(w: usize) -> usize {
    w.max(MTP_MAX_K + 1)
}

/// The real-q list hand-off of one drafted lane: its chain wrote list rows 0..w of the SERVED
/// scratch (row i = draft pass i); its verify rows sit at row offset `row0` of the packed
/// scratch, and `xq_sample_rows_rq` reads `list + row * RQ_ROW`. Returns (destination word
/// offset, words) of the copy from the served list's word 0, or None for a lane without a list
/// (greedy, match mode). Both lanes' chains write the served rows 0..w, so the copy of the
/// first lane must be issued BEFORE the second lane drafts.
pub(crate) fn list_handoff(row0: usize, w: usize, ratio: bool, dst_words: usize) -> Result<Option<(usize, usize)>> {
    if !ratio { return Ok(None); }
    anyhow::ensure!(w >= 1 && w <= MTP_MAX_K && (row0 + w) * RQ_ROW <= dst_words,
                    "lane-pack: {w} list rows at row {row0} outside the packed list ({} rows)", dst_words / RQ_ROW);
    Ok(Some((row0 * RQ_ROW, w * RQ_ROW)))
}

// ---- S-B9-CF-PACK5: the packed round's lockstep words (TP) ----

/// One segment of a pack as the pre-verify lockstep point sees it: everything the packed verify's
/// staging is built from (`PackMeta`: slot, bonus token, position, drafts).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PreSeg<'a> {
    pub slot: usize,
    pub pos: usize,
    pub bonus: i32,
    pub drafts: &'a [i32],
    /// the segment's sampler as the packed verify stages its row 0: kind 0 = greedy, 1 = match
    /// rule, 2 = real-q, and the row's 8 words (`SampParams::pack_flag`: flag, temperature, top_p,
    /// top_k, seed lo / hi, counter, min_p; all zero for a greedy lane). Carried so a rank that
    /// disagrees about WHICH sampler (or seed, or counter) a lane's rows use fails at the
    /// pre-verify, before the verify launches, not one round later in the emitted ids.
    pub kind: u32,
    pub samp: [u32; 8],
    /// head draft passes THIS lane's chain issued on this rank (loop + speculative launches) and
    /// whether one of them was a wasted speculative pass: per lane, not only their sum in the
    /// exchange word (two ranks whose per-lane launch counts differ in opposite directions would
    /// sum alike; their collectives would already have mis-paired inside lane B's draft).
    pub launched: u32,
    pub wasted: u32,
}

impl<'a> PreSeg<'a> {
    /// A greedy segment (the sampler words all zero, no pass accounting).
    pub(crate) fn greedy(slot: usize, pos: usize, bonus: i32, drafts: &'a [i32]) -> Self {
        PreSeg { slot, pos, bonus, drafts, kind: 0, samp: [0; 8], launched: 0, wasted: 0 }
    }

    /// The segment of a drafted lane: its sampler kind and row-0 block from `SegSamp`, and its
    /// chain's pass accounting.
    pub(crate) fn of(slot: usize, pos: usize, bonus: i32, drafts: &'a [i32], s: &SegSamp, launched: usize, wasted: bool) -> Self {
        let (kind, samp) = match s.samp {
            None => (0, [0u32; 8]),
            Some(p) => (if s.ratio { 2 } else { 1 }, p.pack_flag(1)),
        };
        PreSeg { slot, pos, bonus, drafts, kind, samp, launched: launched as u32, wasted: wasted as u32 }
    }
}

/// Width of the fixed part of a segment in `pre_words`: rows, slot, pos, bonus, kind, 8 sampler
/// words, launched, wasted.
const PRE_SEG_HEAD: usize = 15;

/// A one-line, human-readable rendering of a `pre_words` list (the words are self-delimiting), for
/// the failure message of the pre-verify: which lane brought which slot / position / bonus / draft
/// chain, so a divergence is diagnosable from the error itself (the decision trace is printed only
/// for the first rounds).
pub(crate) fn describe_pre_words(w: &[i32]) -> String {
    let (mut i, mut out) = (0usize, Vec::new());
    while i < w.len() {
        let rows = w[i].max(0) as usize;
        if rows == 0 || i + PRE_SEG_HEAD + (rows - 1) > w.len() {
            out.push(format!("(malformed tail at word {i})"));
            break;
        }
        let d = &w[i + PRE_SEG_HEAD..i + PRE_SEG_HEAD + rows - 1];
        out.push(format!("seg {}: {rows} rows slot {} pos {} bonus {} kind {} passes {}{} drafts {d:?}",
                         out.len(), w[i + 1], w[i + 2], w[i + 3], w[i + 4], w[i + 13], if w[i + 14] != 0 { " (wasted)" } else { "" }));
        i += PRE_SEG_HEAD + rows - 1;
    }
    out.join(" | ")
}

/// The words the packed round's ONE pre-verify exchange hashes (design 5.1, OQ1: a single combined
/// hash over the concatenation, with `R` carried explicitly next to it). Per segment, in segment
/// order: `[rows, slot, pos, bonus, kind, sampler row 0 (8 words), launched, wasted, d_0 .. d_{rows-2}]`. The leading row count of every segment
/// makes the concatenation self-delimiting, so no two different packs (a different split of the
/// same drafts, a swapped pair, a shifted position, another slot, one changed draft) can produce
/// the same word list: the combined FNV is exactly as strict as a per-request vector would be (a
/// vector would only change the diagnostic message).
pub(crate) fn pre_words(segs: &[PreSeg]) -> Vec<i32> {
    let mut w = Vec::with_capacity(segs.iter().map(|s| PRE_SEG_HEAD + s.drafts.len()).sum());
    for s in segs {
        w.push((s.drafts.len() + 1) as i32);
        w.push(s.slot as i32);
        w.push(s.pos as i32);
        w.push(s.bonus);
        w.push(s.kind as i32);
        w.extend(s.samp.iter().map(|&x| x as i32));
        w.push(s.launched as i32);
        w.push(s.wasted as i32);
        w.extend_from_slice(s.drafts);
    }
    w
}

/// Total rows of a pack = the one number the verify's barrier sequence depends on.
pub(crate) fn pack_rows_total(segs: &[PreSeg]) -> usize {
    segs.iter().map(|s| s.drafts.len() + 1).sum()
}

/// One lane's outcome as the post-verify lockstep point hashes it (the lone `tp_round` hashes the
/// same three things for its single lane: emitted ids, the drafts, the draft confidences).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PostLane<'a> {
    pub accepted: usize,
    pub rows: usize,
    pub emitted: &'a [i32],
    pub drafts: &'a [i32],
    pub confs: &'a [f32],
}

/// The packed round's post-verify words: per lane `[accepted, rows, n_emitted, emitted.., n_drafts,
/// drafts.., n_confs, conf bits..]` (counts make the list self-delimiting, as in `pre_words`).
pub(crate) fn post_words(lanes: &[PostLane]) -> Vec<u32> {
    let mut w = Vec::new();
    for l in lanes {
        w.push(l.accepted as u32);
        w.push(l.rows as u32);
        w.push(l.emitted.len() as u32);
        w.extend(l.emitted.iter().map(|&x| x as u32));
        w.push(l.drafts.len() as u32);
        w.extend(l.drafts.iter().map(|&x| x as u32));
        w.push(l.confs.len() as u32);
        w.extend(l.confs.iter().map(|c| c.to_bits()));
    }
    w
}

/// Rounds a request's `arms=` trace spells out; later rounds are only counted.
pub(crate) const ARM_TRACE_ROUNDS: usize = 48;

/// Per-request round provenance (the `packed=` field of `[mtp-stats]`; with `--round-trace`
/// also `arms=`): one entry per lane round in order, a letter for the arm that produced the
/// round's tokens followed by how many it produced. `P` = a packed verify, `S` = a lone
/// speculative verify, `c` = a control round (plain), `p` = a plain step (end of the window or
/// budget, an MTP-off lane's batch, a shared plain step). The request's first token comes from
/// the prefill seam step and is in no round, so the token with generated index g >= 1 belongs
/// to the first entry whose running token count reaches g.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ArmTrace {
    s: String,
    /// lane rounds recorded in all, and those served by a packed verify
    pub rounds: usize,
    pub packed: usize,
}

impl ArmTrace {
    pub(crate) fn push(&mut self, arm: char, tokens: usize) {
        debug_assert!(matches!(arm, 'P' | 'S' | 'c' | 'p'));
        if self.rounds < ARM_TRACE_ROUNDS {
            self.s.push(arm);
            self.s.push_str(&tokens.to_string());
        }
        self.rounds += 1;
        self.packed += (arm == 'P') as usize;
    }

    /// The trace text; `+` marks rounds past `ARM_TRACE_ROUNDS` (counted, not spelled out).
    pub(crate) fn text(&self) -> String {
        if self.rounds == 0 { return "-".to_string(); }
        if self.rounds > ARM_TRACE_ROUNDS { format!("{}+", self.s) } else { self.s.clone() }
    }

    /// The arm of the round that produced the token with generated index `g` (g >= 1), None
    /// when g is 0 (the seam token) or past the spelled-out rounds. Mirrored by the gate driver.
    pub(crate) fn arm_of(text: &str, g: usize) -> Option<char> {
        if g == 0 { return None; }
        let (mut cum, mut it) = (0usize, text.chars().peekable());
        while let Some(arm) = it.next() {
            if !matches!(arm, 'P' | 'S' | 'c' | 'p') { return None; }
            let mut n = 0usize;
            while let Some(d) = it.peek().and_then(|c| c.to_digit(10)) { n = n * 10 + d as usize; it.next(); }
            cum += n;
            if g <= cum { return Some(arm); }
        }
        None
    }
}

#[cfg(test)]
mod tests;
