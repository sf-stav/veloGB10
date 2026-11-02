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
    /// the resolved `--kv-cache` name (q8 is the default; q8 and q4k are the gated formats)
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
}

/// F1 (owner 2026-10-10, AGENTS §1b-bis item 8 — "support things properly or gracefully fall back"):
/// `--lane-pack on` never refuses a boot. Every option combination the packed arm was never gated
/// under becomes a FALLBACK: the server boots with packed rounds OFF for this boot, prints ONE plain
/// line (`lane packing (--lane-pack, on by default) is NOT ACTIVE on this boot: <reason>; requests are
/// served without packed rounds`), and the decision rides `/metrics` and the TP boot agree. Returns the plain-word
/// reasons (empty = this posture can pack).
pub(crate) fn posture_fallbacks(p: &Posture) -> Vec<String> {
    let mut v = Vec::new();
    if p.fcfs {
        v.push("--lane-order fcfs: fcfs steps ONE lane per scheduler step, a pack steps two (the rr order steps both)".to_string());
    }
    // q4k is admitted: its rows are symmetric (144 B) and the packed arm reaches KV only through
    // xq_kv_make, with attention on xq_attn_decode_group (per-row slot). The older formats stay
    // gated until they are run here.
    if !matches!(p.kv, "q8" | "q4k") {
        v.push(format!("--kv-cache {}: the packed verify's per-segment attention and commit were never run on this KV format (gated: q8, q4k)", p.kv));
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
}

impl Skip {
    pub(crate) const ALL: [Skip; 13] = [Skip::FewPackable, Skip::OddLane, Skip::Prefill, Skip::MtpOff, Skip::Control,
        Skip::Probe, Skip::Sampled, Skip::Penalty, Skip::Image, Skip::Depth0, Skip::MixedRegime, Skip::RegimeClamp, Skip::NoGraph];

    /// The metric label value.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Skip::FewPackable => "few_packable", Skip::OddLane => "odd_lane", Skip::Prefill => "prefill",
            Skip::MtpOff => "mtp_off", Skip::Control => "control", Skip::Probe => "probe", Skip::Sampled => "sampled",
            Skip::Penalty => "penalty", Skip::Image => "image", Skip::Depth0 => "depth0",
            Skip::MixedRegime => "mixed_regime", Skip::RegimeClamp => "regime_clamp", Skip::NoGraph => "no_graph",
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
mod tests {
    use super::*;

    fn lv(slot: usize, block: Option<Skip>, regime: u64, clamped: bool) -> LaneView {
        LaneView { slot, block, regime, clamped }
    }
    fn ok(slot: usize) -> LaneView { lv(slot, None, 0, false) }

    #[test]
    fn flag_parse() {
        for v in [None, Some(""), Some("off"), Some(" off ")] {
            assert_eq!(LanePack::parse(v).unwrap(), LanePack::Off, "{v:?}");
        }
        for v in [Some("on"), Some("2"), Some(" 2 ")] {
            assert_eq!(LanePack::parse(v).unwrap(), LanePack::Pairs, "{v:?}");
        }
        let e = LanePack::parse(Some("3")).unwrap_err().to_string();
        assert!(e.contains("not built") && e.contains("use on or 2"), "{e}");
        for bad in ["1", "0", "4", "auto", "yes", "true"] {
            let e = LanePack::parse(Some(bad)).unwrap_err().to_string();
            assert!(e.starts_with("--lane-pack must be off, on or 2"), "{bad}: {e}");
        }
        assert!(!LanePack::Off.on() && LanePack::Pairs.on());
    }

    #[test]
    fn tp_is_refused_only_above_four_boxes_and_only_when_on() {
        for w in [1u32, 2, 4, 8] { assert!(tp_refusal(LanePack::Off, w).is_ok(), "the default must boot at TP={w}"); }
        // owner 2026-10-10: TP=4 is no longer refused (unverified on hardware until its gates run)
        for w in [1u32, 2, 3, 4] { assert!(tp_refusal(LanePack::Pairs, w).is_ok(), "--lane-pack on boots at TP={w}"); }
        for w in [5u32, 8] {
            let e = tp_refusal(LanePack::Pairs, w).unwrap_err().to_string();
            assert!(e.starts_with("--lane-pack runs at TP=1, TP=2 and TP=4") && e.contains(&format!("TP={w} is")), "{e}");
        }
    }

    /// OQ1 (design 9): the combined hash is exactly as strict as a per-request vector. Every
    /// single-field change of a two-lane pack, and every re-split of the same words, moves the word
    /// list; the total row count is explicit.
    #[test]
    fn the_pre_verify_words_tell_every_pack_apart() {
        let (d0, d1) = ([11, 12, 13], [21, 22]);
        let base = [PreSeg::greedy(1, 100, 7, &d0), PreSeg::greedy(2, 200, 8, &d1)];
        let w = pre_words(&base);
        assert_eq!(w, vec![4, 1, 100, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13,
                           3, 2, 200, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 22]);
        assert_eq!(pack_rows_total(&base), 7);
        let mut seen = std::collections::HashSet::new();
        seen.insert(w.clone());
        let variants: Vec<Vec<i32>> = vec![
            pre_words(&[PreSeg { slot: 3, ..base[0] }, base[1]]),                       // another slot
            pre_words(&[PreSeg { pos: 101, ..base[0] }, base[1]]),                      // a shifted position
            pre_words(&[PreSeg { bonus: 9, ..base[0] }, base[1]]),                      // another bonus token
            pre_words(&[base[0], PreSeg { drafts: &[21, 23], ..base[1] }]),             // one draft differs
            pre_words(&[base[1], base[0]]),                                            // the pair swapped
            pre_words(&[PreSeg { drafts: &[11, 12], ..base[0] }, PreSeg { drafts: &[13, 21, 22], ..base[1] }]), // re-split
            pre_words(&[base[0]]),                                                     // a lone segment
            // the sampler of a lane: kind, then every word of its row-0 block (seed, counter, temperature ...)
            pre_words(&[PreSeg { kind: 1, ..base[0] }, base[1]]),
            pre_words(&[PreSeg { kind: 2, ..base[0] }, base[1]]),
            pre_words(&[PreSeg { samp: [1, 0x3f80_0000, 0x3f73_3333, 20, 5, 0, 9, 0], kind: 2, ..base[0] }, base[1]]),
            pre_words(&[PreSeg { samp: [1, 0x3f80_0000, 0x3f73_3333, 20, 5, 0, 10, 0], kind: 2, ..base[0] }, base[1]]), // counter + 1
            pre_words(&[PreSeg { samp: [1, 0x3f80_0000, 0x3f73_3333, 20, 6, 0, 9, 0], kind: 2, ..base[0] }, base[1]]),  // another seed
            // per-lane pass accounting: two ranks whose launch counts differ in OPPOSITE directions have equal sums
            pre_words(&[PreSeg { launched: 3, ..base[0] }, PreSeg { launched: 4, ..base[1] }]),
            pre_words(&[PreSeg { launched: 4, ..base[0] }, PreSeg { launched: 3, ..base[1] }]),
            pre_words(&[PreSeg { launched: 3, wasted: 1, ..base[0] }, PreSeg { launched: 4, ..base[1] }]),
        ];
        for v in variants { assert!(seen.insert(v.clone()), "a different pack produced the same words: {v:?}"); }
        // same total rows, different split: the explicit per-segment row count tells them apart
        let a = pre_words(&[PreSeg::greedy(0, 0, 0, &[1, 2, 3]), PreSeg::greedy(1, 0, 0, &[4])]);
        let b = pre_words(&[PreSeg::greedy(0, 0, 0, &[1, 2]), PreSeg::greedy(1, 0, 0, &[3, 4])]);
        assert_ne!(a, b);
        assert_eq!(pack_rows_total(&[PreSeg::greedy(0, 0, 0, &[1, 2, 3]), PreSeg::greedy(1, 0, 0, &[4])]), 6);
        // a SegSamp maps to its kind: None = greedy, Some + ratio = real-q, Some = match rule
        let sp = SampParams::sampled(0.7, 0.8, 20, 0.0, 42, 5).unwrap();
        assert_eq!(PreSeg::of(0, 0, 0, &[1], &SegSamp::GREEDY, 3, false).kind, 0);
        assert_eq!(PreSeg::of(0, 0, 0, &[1], &SegSamp { samp: Some(sp), ratio: false }, 3, false).kind, 1);
        let q = PreSeg::of(0, 0, 0, &[1], &SegSamp { samp: Some(sp), ratio: true }, 5, true);
        assert_eq!((q.kind, q.launched, q.wasted), (2, 5, 1));
        assert_eq!(q.samp, sp.pack_flag(1), "the row-0 block the verify stages (flag 1 = the lane's own words)");
        // the failure message can name every lane of a pack from the words alone
        let (d0, d1) = ([11, 12, 13], [21, 22]);
        let txt = describe_pre_words(&pre_words(&[PreSeg::of(1, 100, 7, &d0, &SegSamp::GREEDY, 4, true),
                                                  PreSeg::of(2, 200, 8, &d1, &SegSamp { samp: Some(sp), ratio: true }, 3, false)]));
        assert!(txt.contains("seg 0: 4 rows slot 1 pos 100 bonus 7 kind 0 passes 4 (wasted) drafts [11, 12, 13]")
                && txt.contains("seg 1: 3 rows slot 2 pos 200 bonus 8 kind 2 passes 3 drafts [21, 22]"), "{txt}");
        assert!(describe_pre_words(&[9, 1]).contains("malformed"));
    }

    #[test]
    fn the_post_verify_words_carry_each_lane_and_every_value() {
        let l0 = PostLane { accepted: 2, rows: 4, emitted: &[5, 6, 7], drafts: &[5, 6, 9], confs: &[1.5, 0.5, 0.25] };
        let l1 = PostLane { accepted: 0, rows: 3, emitted: &[8], drafts: &[1, 2], confs: &[0.1, 0.2] };
        let w = post_words(&[l0, l1]);
        assert_eq!(&w[..3], &[2, 4, 3]);
        assert_eq!(w.len(), (3 + 3 + 1 + 3 + 1 + 3) + (3 + 1 + 1 + 2 + 1 + 2));
        let again = post_words(&[l0, l1]);
        assert_eq!(w, again, "deterministic");
        let swapped = post_words(&[l1, l0]);
        assert_ne!(w, swapped);
        let mut moved = l0; moved.accepted = 1;
        assert_ne!(w, post_words(&[moved, l1]), "an accept count change");
        let mut conf = l1; conf.confs = &[0.1, 0.2000001];
        assert_ne!(w, post_words(&[l0, conf]), "a one-ulp confidence change is a different word list");
    }

    #[test]
    fn graph_families_follow_the_sampling_mode_and_never_duplicate() {
        assert_eq!(fams(true, false), vec![PackFam::Greedy]);
        assert_eq!(fams(false, false), vec![PackFam::Greedy]);
        assert_eq!(fams(true, true), vec![PackFam::Greedy, PackFam::Rq]);
        assert_eq!(fams(false, true), vec![PackFam::Greedy, PackFam::Match]);
        // state() and fam_of() are inverse over the three families
        for f in [PackFam::Greedy, PackFam::Match, PackFam::Rq] {
            let (s, r) = f.state();
            assert_eq!(fam_of(s, r), f);
            assert_eq!(f.rq(), r);
        }
        // a pair with a real-q lane replays Rq whatever its partner is; samp_live without rq is Match
        assert_eq!(fam_of(true, true), PackFam::Rq);
        assert_eq!(fam_of(true, false), PackFam::Match);
        assert_eq!(fam_of(false, false), PackFam::Greedy);
    }

    /// PACK4-REVIEW B4: the lane view, as a function. Regime from the last row, the straddle, the
    /// clamp flag, the missing-graph reason and the precedence of the reasons.
    #[test]
    fn the_lane_view_is_a_pure_function_of_what_the_scheduler_reads() {
        let lim = 2051usize; // the sparse switch: positions above it are sparse
        let live = |p: usize| p > lim;
        let base = LaneFacts { mtp_armed: true, ..Default::default() };
        let v = |pos, k_pre, kd, pre, ready: [bool; 2], f: LaneFacts| lane_view(3, f, pos, k_pre, kd, pre, ready, &live);
        // dense, depth 7, graph present: packable, dense, not clamped
        let a = v(100, 7, 7, false, [true, true], base);
        assert_eq!(a, LaneView { slot: 3, block: None, regime: 0, clamped: false });
        // fully past the switch: sparse regime
        assert_eq!(v(3000, 7, 7, false, [true, true], base).regime, 1);
        // the round's regime is its LAST row's: p = 2045, depth 7 reaches 2052 > 2051: sparse, and the
        // first row (2046) is dense, so the round straddles (only reachable with --qsa-row-regime 0:
        // the per-row regime clamps the depth so it never does)
        let s = v(2045, 7, 7, false, [true, true], base);
        assert_eq!((s.regime, s.block), (1, Some(Skip::MixedRegime)));
        // with the regime clamp the depth is 6 (p + 6 = 2051: dense), no straddle, and `clamped` says so
        let c = v(2045, 7, 6, false, [true, true], base);
        assert_eq!((c.regime, c.block, c.clamped), (0, None, true));
        // the clamp at the switch itself: depth 0 = a plain step, never packable
        let z = v(2051, 7, 0, false, [true, true], base);
        assert_eq!(z.block, Some(Skip::Depth0));
        // a missing graph for THIS regime, not the other one
        assert_eq!(v(100, 7, 7, false, [false, true], base).block, Some(Skip::NoGraph));
        assert_eq!(v(3000, 7, 7, false, [false, true], base).block, None);
        assert_eq!(v(3000, 7, 7, false, [true, false], base).block, Some(Skip::NoGraph));
        // precedence: prefill beats every lane fact; a lane fact beats the straddle and the missing graph
        let bad = LaneFacts { penalized: true, ..base };
        assert_eq!(v(2045, 7, 7, true, [false, false], bad).block, Some(Skip::Prefill));
        assert_eq!(v(2045, 7, 7, false, [false, false], bad).block, Some(Skip::Penalty));
        assert_eq!(v(2045, 7, 7, false, [false, false], base).block, Some(Skip::MixedRegime));
        // an MTP-off lane is MtpOff whatever else is true
        assert_eq!(v(100, 7, 7, false, [true, true], LaneFacts::default()).block, Some(Skip::MtpOff));
        // the depth cap (A4): a depth above PACK_DEPTH_MAX cannot move the regime past what a pair can hold
        let capped = v(2040, 12, 12, false, [true, true], base);
        assert_eq!(capped.regime, (2040 + PACK_DEPTH_MAX > lim) as u64);
    }

    /// PACK4-REVIEW B3, now F1 (owner 2026-10-10): every posture the arm was never gated under is a
    /// FALLBACK with its plain-word reason; the default posture packs.
    #[test]
    fn ungated_postures_fall_back_at_boot() {
        let ok = Posture { kv: "q8", mtp_k: MTP_MAX_K, row_regime: true, a_ctrl_old: false, fcfs: false, dense_v3: true };
        assert!(posture_fallbacks(&ok).is_empty(), "the default posture packs");
        let q4k = Posture { kv: "q4k", ..ok };
        assert!(posture_fallbacks(&q4k).is_empty(), "--kv-cache q4k packs");
        for (name, p, want) in [
            ("fcfs", Posture { fcfs: true, ..ok }, "--lane-order fcfs"),
            ("kv f16", Posture { kv: "f16", ..ok }, "--kv-cache f16"),
            ("kv fp8", Posture { kv: "fp8", ..ok }, "--kv-cache fp8"),
            ("depth 5", Posture { mtp_k: 5, ..ok }, "--exl3-mtp-k 5"),
            ("row regime off", Posture { row_regime: false, ..ok }, "--qsa-row-regime 0"),
            ("a-ctrl old", Posture { a_ctrl_old: true, ..ok }, "--a-ctrl old"),
            ("dense v3 off", Posture { dense_v3: false, ..ok }, "--exl3-dense-v3 0"),
        ] {
            let v = posture_fallbacks(&p);
            assert_eq!(v.len(), 1, "{name}: one reason per fallback, got {v:?}");
            assert!(v[0].contains(want), "{name}: {} misses {want}", v[0]);
        }
        // F1: several ungated options at once = several reasons in one line, joined by the boot.
        let multi = Posture { kv: "f16", mtp_k: 5, ..ok };
        assert_eq!(posture_fallbacks(&multi).len(), 2, "each reason is listed");
    }

    /// F10 (owner 2026-10-12): the default is ON, resolved from the registry row; `parse(None)` stays the
    /// raw-argv meaning (off) that pure callers use; a failed boot-time build is a fallback reason, never an abort.
    #[test]
    fn the_default_is_on_and_a_failed_build_falls_back() {
        let row = crate::opts::lookup("lane-pack").expect("registered").def();
        assert_eq!(row.def, "on", "the registry default");
        assert_eq!(LanePack::parse(Some(row.def)).unwrap(), LanePack::Pairs);
        assert_eq!(LanePack::parse(Some("off")).unwrap(), LanePack::Off, "--lane-pack off restores the old behaviour");
        assert_eq!(LanePack::parse(Some("2")).unwrap(), LanePack::Pairs, "2 keeps its meaning");
        assert_eq!(LanePack::parse(None).unwrap(), LanePack::Off, "an absent raw value is the raw-argv meaning");
        // the real call shape: Ok passes through, Err becomes a plain reason that names the cause
        let ok: std::result::Result<u32, String> = build_or_fall_back(|| Ok(7));
        assert_eq!(ok, Ok(7));
        let err = build_or_fall_back::<u32>(|| anyhow::bail!("CUDA_ERROR_OUT_OF_MEMORY")).unwrap_err();
        assert!(err.contains("could not be built at boot") && err.contains("CUDA_ERROR_OUT_OF_MEMORY"), "{err}");
    }

    /// PACK4-REVIEW A5: `--help` quoted x1.18 / 95.8 -> 113.0 tok/s from a pre-final binary (the final
    /// binary's single pass read x1.13). Measured numbers go stale with every build, so the flag's help
    /// says what the flag does and where the numbers live; this pin keeps a figure from coming back.
    #[test]
    fn the_help_text_quotes_no_measured_numbers() {
        let t = include_str!("opts_table.rs");
        let i = t.find("flag: \"lane-pack\"").expect("the flag's table row");
        let row = &t[i..i + t[i..].find("\" },").expect("end of the row")];
        assert!(row.len() > 500, "the help text is there");
        for bad in ["tok/s", "x1.", "95.8", "113.0", "GB of", "0.4 GB"] {
            assert!(!row.contains(bad), "a measured figure ({bad}) in the --lane-pack help: it goes stale with every build");
        }
        assert!(row.contains("Speed factors are measured per build and are not quoted here"));
    }

    /// PACK4-REVIEW A4: the segment cap is derived from the lone verify's width, a pair of maximal
    /// segments always fits, and every depth the planner can hand to `pack_rows` is accepted.
    #[test]
    fn the_segment_cap_follows_the_lone_verify_width() {
        assert_eq!(PACK_DEPTH_MAX, MTP_MAX_K.min(PACK_ROWS_MAX / PACK_K - 1));
        assert_eq!(PACK_SEG_ROWS_MAX, PACK_DEPTH_MAX + 1);
        assert!(PACK_SEG_ROWS_MAX <= MTP_MAX_K + 1 && PACK_K * PACK_SEG_ROWS_MAX <= PACK_ROWS_MAX);
        for w0 in 1..=PACK_DEPTH_MAX {
            for w1 in 1..=PACK_DEPTH_MAX {
                let (m0, m1, r) = pack_rows(w0, w1).unwrap_or_else(|| panic!("a planned pair {w0} + {w1} must fit"));
                assert!(m0 + m1 == r && r <= PACK_ROWS_MAX && PACK_WIDTHS.contains(&r));
            }
        }
        assert!(pack_rows(PACK_DEPTH_MAX + 1, 1).is_none(), "one past the cap is refused by the pure function too");
    }

    /// Off == today: no plan for ANY lane set, so the caller cannot leave master's path.
    #[test]
    fn off_and_single_lane_never_plan() {
        let sets: Vec<Vec<LaneView>> = vec![
            vec![], vec![ok(0)], vec![ok(0), ok(1)], vec![ok(0), ok(1), ok(2)],
            vec![ok(0), lv(1, Some(Skip::Sampled), 0, false)], (0..16).map(ok).collect(),
        ];
        for s in &sets {
            for rot in 0..5 {
                assert_eq!(plan(LanePack::Off, s, rot), None, "off must never plan ({} lanes)", s.len());
            }
        }
        // on: a lone busy request never packs and is not even planned
        for rot in 0..5 {
            assert_eq!(plan(LanePack::Pairs, &[], rot), None);
            assert_eq!(plan(LanePack::Pairs, &[ok(3)], rot), None);
        }
    }

    #[test]
    fn every_lane_level_reason() {
        let base = LaneFacts { mtp_armed: true, depth: 4, ..Default::default() };
        assert_eq!(lane_block(&base), None);
        assert_eq!(lane_block(&LaneFacts { mtp_armed: false, ..base }), Some(Skip::MtpOff));
        assert_eq!(lane_block(&LaneFacts { force_control: true, ..base }), Some(Skip::Control));
        assert_eq!(lane_block(&LaneFacts { probing: true, ..base }), Some(Skip::Probe));
        // PACK4S: a sampled lane packs; only a boot with the lone-verify sampler dump keeps it out
        assert_eq!(lane_block(&LaneFacts { sampled: true, ..base }), None);
        assert_eq!(lane_block(&LaneFacts { sampled: true, sampled_lone: true, ..base }), Some(Skip::Sampled));
        assert_eq!(lane_block(&LaneFacts { sampled_lone: true, ..base }), None, "the dump keeps GREEDY lanes packable");
        assert_eq!(lane_block(&LaneFacts { penalized: true, ..base }), Some(Skip::Penalty));
        assert_eq!(lane_block(&LaneFacts { sampled: true, penalized: true, ..base }), Some(Skip::Penalty),
                   "a penalized sampled lane still stays out (penalties are step 2)");
        assert_eq!(lane_block(&LaneFacts { image: true, ..base }), Some(Skip::Image));
        assert_eq!(lane_block(&LaneFacts { depth: 0, ..base }), Some(Skip::Depth0));
        // precedence: an MTP-off lane is "mtp_off" whatever else is true of it
        assert_eq!(lane_block(&LaneFacts { mtp_armed: false, sampled: true, depth: 0, ..base }), Some(Skip::MtpOff));
        assert_eq!(lane_block(&LaneFacts { sampled: true, sampled_lone: true, penalized: true, image: true, ..base }), Some(Skip::Sampled));
        assert_eq!(lane_block(&LaneFacts { sampled: true, image: true, ..base }), Some(Skip::Image));
    }

    #[test]
    fn two_lanes() {
        let p = plan(LanePack::Pairs, &[ok(0), ok(1)], 0).unwrap();
        assert_eq!(p, Plan { pairs: vec![[0, 1]], lone: vec![] });
        // the rotation does not change a two-lane pair (and the pair is stored slot-ascending)
        for rot in 0..6 {
            assert_eq!(plan(LanePack::Pairs, &[ok(2), ok(5)], rot).unwrap().pairs, vec![[2, 5]]);
        }
        // one blocked lane: the other has nobody to pack with
        for why in [Skip::MtpOff, Skip::Control, Skip::Probe, Skip::Sampled, Skip::Penalty, Skip::Image, Skip::Depth0] {
            let p = plan(LanePack::Pairs, &[ok(0), lv(1, Some(why), 0, false)], 0).unwrap();
            assert!(p.pairs.is_empty());
            assert_eq!(p.lone, vec![(0, Skip::FewPackable), (1, why)], "{why:?}");
        }
        // both blocked: each lane keeps its own reason
        let p = plan(LanePack::Pairs, &[lv(0, Some(Skip::Sampled), 0, false), lv(1, Some(Skip::Penalty), 0, false)], 0).unwrap();
        assert_eq!(p.lone, vec![(0, Skip::Sampled), (1, Skip::Penalty)]);
    }

    #[test]
    fn regime_rules() {
        // different sides of the switch: mixed regime, both serial
        let p = plan(LanePack::Pairs, &[lv(0, None, 0, false), lv(1, None, 1, false)], 0).unwrap();
        assert!(p.pairs.is_empty());
        assert_eq!(p.lone, vec![(0, Skip::MixedRegime), (1, Skip::MixedRegime)]);
        // same regime, one lane clamped at the switch this round
        let p = plan(LanePack::Pairs, &[lv(0, None, 0, true), lv(1, None, 0, false)], 0).unwrap();
        assert!(p.pairs.is_empty());
        assert_eq!(p.lone, vec![(0, Skip::RegimeClamp), (1, Skip::RegimeClamp)]);
        // both clamped: every row of both is at/below the switch, they pack
        let p = plan(LanePack::Pairs, &[lv(0, None, 0, true), lv(1, None, 0, true)], 0).unwrap();
        assert_eq!(p.pairs, vec![[0, 1]]);
        // three lanes, one past the switch: the two short lanes pack, the long one is "mixed regime"
        for rot in 0..4 {
            let p = plan(LanePack::Pairs, &[lv(0, None, 0, false), lv(1, None, 1, false), lv(2, None, 0, false)], rot).unwrap();
            assert_eq!(p.pairs, vec![[0, 2]], "rot {rot}");
            assert_eq!(p.lone, vec![(1, Skip::MixedRegime)], "rot {rot}");
        }
        // two regime groups of two: two pairs
        let p = plan(LanePack::Pairs, &[lv(0, None, 0, false), lv(1, None, 1, false), lv(2, None, 0, false), lv(3, None, 1, false)], 0).unwrap();
        assert_eq!(p.pairs, vec![[0, 2], [1, 3]]);
        assert!(p.lone.is_empty());
    }

    /// The three-lane rotation, deterministically: the lone role visits every lane once per three
    /// steps, every lane runs exactly one round per step, and no pairing is fixed.
    #[test]
    fn three_lane_rotation_is_fair() {
        let lanes = [ok(0), ok(1), ok(2)];
        let mut lone_seq = Vec::new();
        let mut pair_cnt = std::collections::BTreeMap::<[usize; 2], usize>::new();
        let mut rounds = [0usize; 3];
        for rot in 0..300u64 {
            let p = plan(LanePack::Pairs, &lanes, rot).unwrap();
            assert_eq!(p.pairs.len(), 1);
            assert_eq!(p.lone.len(), 1);
            assert_eq!(p.lone[0].1, Skip::OddLane);
            lone_seq.push(p.lone[0].0);
            *pair_cnt.entry(p.pairs[0]).or_default() += 1;
            for s in p.pairs[0] { rounds[s] += 1; }
            rounds[p.lone[0].0] += 1;
        }
        assert_eq!(&lone_seq[..6], &[2, 0, 1, 2, 0, 1], "lone = c, a, b, ...");
        for s in 0..3 {
            assert_eq!(rounds[s], 300, "every lane runs one round per step");
            assert_eq!(lone_seq.iter().filter(|&&x| x == s).count(), 100, "lane {s} is lone once per three steps");
            // never lone twice in a row, never packed more than twice in a row
            let gaps: Vec<usize> = lone_seq.iter().enumerate().filter(|(_, &x)| x == s).map(|(i, _)| i).collect();
            assert!(gaps.windows(2).all(|w| w[1] - w[0] == 3), "lane {s}: lone turns are exactly three steps apart");
        }
        assert_eq!(pair_cnt.len(), 3, "all three pairings occur");
        assert!(pair_cnt.values().all(|&c| c == 100));
    }

    /// Lanes arriving and leaving (the FCFSB-style simulation): whatever the busy set does, every
    /// busy lane gets one round per step, and a lane that stays busy among an odd number of
    /// packable lanes is never the odd one for more than one step in any `n` consecutive steps
    /// of an unchanged busy set.
    #[test]
    fn rotation_with_arrivals_and_departures() {
        // (first step, last step) of each of 6 lanes; deterministic schedule
        let life = [(0u64, 400u64), (0, 137), (10, 400), (50, 260), (138, 320), (261, 400)];
        let mut odd_run = [0usize; 6];
        let mut packed = [0usize; 6];
        let mut lone = [0usize; 6];
        let mut prev_set: Vec<usize> = Vec::new();
        let mut since_change = 0usize;
        let mut odd_in_window: Vec<usize> = vec![0; 6];
        for step in 0..=400u64 {
            let busy: Vec<usize> = (0..6).filter(|&s| life[s].0 <= step && step <= life[s].1).collect();
            let views: Vec<LaneView> = busy.iter().map(|&s| ok(s)).collect();
            if busy != prev_set { since_change = 0; odd_in_window = vec![0; 6]; prev_set = busy.clone(); }
            let Some(p) = plan(LanePack::Pairs, &views, step) else {
                assert!(busy.len() < 2);
                continue;
            };
            // every busy lane appears exactly once: in a pair or lone
            let mut seen: Vec<usize> = p.pairs.iter().flatten().copied().chain(p.lone.iter().map(|(s, _)| *s)).collect();
            seen.sort();
            assert_eq!(seen, busy, "step {step}");
            assert_eq!(p.pairs.len(), busy.len() / 2, "step {step}: every possible pair is formed");
            for s in p.pairs.iter().flatten() { packed[*s] += 1; odd_run[*s] = 0; }
            for (s, why) in &p.lone {
                assert_eq!(*why, Skip::OddLane);
                lone[*s] += 1;
                odd_run[*s] += 1;
                odd_in_window[*s] += 1;
                assert!(odd_run[*s] <= 1 || since_change == 0, "step {step}: lane {s} was the odd lane twice in a row");
            }
            since_change += 1;
            if since_change == busy.len() && busy.len() % 2 == 1 {
                for &s in &busy { assert_eq!(odd_in_window[s], 1, "step {step}: lane {s} odd once per {} steps", busy.len()); }
                since_change = 0;
                odd_in_window = vec![0; 6];
            }
        }
        for s in 0..6 {
            assert!(packed[s] > 0, "lane {s} never packed");
            assert!(packed[s] >= lone[s], "lane {s}: packed {} lone {} — the odd role is the minority", packed[s], lone[s]);
        }
    }

    #[test]
    fn control_cadence_is_the_serial_arms() {
        assert_eq!(ctrl_period(true, 0.0, 10.0), 128);
        assert_eq!(ctrl_period(true, 30.0, 0.0), 128);
        assert_eq!(ctrl_period(true, 30.0, 23.9), 1024);
        assert_eq!(ctrl_period(true, 30.0, 24.0), 128, "0.8x is not clearly below");
        assert_eq!(ctrl_period(false, 30.0, 10.0), 128, "--a-ctrl set: the fixed cadence");
        // the pair consumes counter values c+1 and c+2
        assert!(!pair_owes_control(0, 128, 128));
        assert!(pair_owes_control(127, 128, 128), "the first slot lands on 128");
        assert!(pair_owes_control(126, 128, 128), "the second slot lands on 128");
        assert!(!pair_owes_control(128, 128, 128));
        assert!(pair_owes_control(1022, 128, 1024), "second slot lands on 1024");
        assert!(pair_owes_control(1022, 1024, 128), "second slot lands on a multiple of its own period, 128");
        assert!(!pair_owes_control(1020, 1024, 1024), "1021 and 1022: neither lane is due");
        assert!(pair_owes_control(1023, 1024, 1024), "first slot lands on 1024");
        // count for count: over a long run a pair-or-serial scheduler runs exactly as many controls
        // as the serial arm (every counter value is consumed once; multiples of 128 are controls)
        let (mut c, mut packed, mut controls) = (0usize, 0usize, 0usize);
        while c < 128 * 50 {
            if pair_owes_control(c, 128, 128) {
                for _ in 0..2 { c += 1; if c % 128 == 0 { controls += 1; } } // two serial rounds
            } else {
                c += 2;
                packed += 1;
                assert!((c - 1) % 128 != 0 && c % 128 != 0, "a packed pair swallowed a control at {c}");
            }
        }
        assert_eq!(controls, 50);
        assert!(packed > 3000);
    }

    #[test]
    fn time_division() {
        assert_eq!(lane_share_ms(49.4, 2), 24.7);
        assert_eq!(lane_share_ms(35.2, 1), 35.2, "a lone round keeps its time");
        assert_eq!(lane_share_ms(10.0, 0), 10.0);
    }

    #[test]
    fn row_allocation() {
        assert_eq!(pack_rows(4, 4), Some((5, 5, 10)));
        assert_eq!(pack_rows(7, 7), Some((8, 8, 16)));
        assert_eq!(pack_rows(1, 1), Some((2, 2, 4)));
        assert_eq!(pack_rows(5, 3), Some((6, 4, 10)));
        assert_eq!(pack_rows(1, 2), Some((2, 3, 5)), "odd totals are exact: no padding row");
        assert_eq!(pack_rows(0, 4), None);
        assert_eq!(pack_rows(4, 0), None);
        assert_eq!(pack_rows(8, 1), None, "a lone verify is at most 8 rows");
        // every pair of legal depths fits the 16-row ceiling and lands on a precaptured width
        for w0 in 1..=7 {
            for w1 in 1..=7 {
                let (m0, m1, r) = pack_rows(w0, w1).unwrap();
                assert!(m0 <= PACK_SEG_ROWS_MAX && m1 <= PACK_SEG_ROWS_MAX && r <= PACK_ROWS_MAX);
                assert!(PACK_WIDTHS.contains(&r), "R = {r}");
            }
        }
        assert_eq!(PACK_WIDTHS.clone().count(), 13);
    }

    #[test]
    fn staging_metadata() {
        let (d0, d1) = ([11, 12, 13, 14], [21, 22]);
        let s0 = SegIn { slot: 2, b: 10, p: 100, drafts: &d0 };
        let s1 = SegIn { slot: 0, b: 20, p: 7000, drafts: &d1 };
        let m = pack_meta(&s0, &s1, 16).unwrap();
        assert_eq!(m.r, 8);
        assert_eq!(m.rows, vec![(2, 10, 100), (2, 11, 101), (2, 12, 102), (2, 13, 103), (2, 14, 104),
                                (0, 20, 7000), (0, 21, 7001), (0, 22, 7002)]);
        // {slot, row_start, len, flags = nostore}
        assert_eq!(m.segs, [[2, 0, 5, 1], [0, 5, 3, 1]]);
        // drafts at their segment's row offset; the bonus row's slot (row0 + len - 1) stays 0
        assert_eq!(m.d_dev, vec![11, 12, 13, 14, 0, 21, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        // exactly R rows: NO padding row is staged (module note); rows past R are the caller's
        // zero fill, as in the lone verify
        assert_eq!(m.rows.len(), m.r);
        // an odd total stays exact
        let m = pack_meta(&SegIn { slot: 0, b: 1, p: 5, drafts: &[2] }, &SegIn { slot: 1, b: 3, p: 9, drafts: &[4, 5] }, 16).unwrap();
        assert_eq!((m.r, m.rows.len()), (5, 5));
        assert_eq!(m.segs, [[0, 0, 2, 1], [1, 2, 3, 1]]);
        assert_eq!(&m.d_dev[..5], &[2, 0, 4, 5, 0]);
        // refusals
        assert!(pack_meta(&SegIn { slot: 0, b: 1, p: 5, drafts: &[] }, &s1, 16).is_err(), "a lane without a draft");
        assert!(pack_meta(&SegIn { slot: 0, b: 1, p: 5, drafts: &d1 }, &s1, 16).is_err(), "one slot twice");
        assert!(pack_meta(&s0, &SegIn { slot: 1, b: 1, p: 5, drafts: &d1 }, 7).is_err(), "draft buffer too narrow");
    }

    #[test]
    fn readback_per_segment() {
        let segs = [[2, 0, 5, 1], [0, 5, 3, 1]];
        // segment 0 accepted 2 drafts (emits d0, d1, e2), segment 1 accepted 0 (emits e0)
        let out = [11, 12, 99, -1, -1, 77, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0];
        let acc = [2, 2, 0, 0, 0, 0];
        assert_eq!(seg_result(&out, &acc, 0, &segs[0]).unwrap(), (2, vec![11, 12, 99]));
        assert_eq!(seg_result(&out, &acc, 1, &segs[1]).unwrap(), (0, vec![77]));
        // full accept of segment 1: a = len - 1
        let out2 = [11, -1, -1, -1, -1, 21, 22, 55, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(seg_result(&out2, &[2, 0, 0, 2, 0, 0], 1, &segs[1]).unwrap(), (2, vec![21, 22, 55]));
        // tripwires: a commit key for another slot, an accept past the segment, a -1 inside the emitted run
        assert!(seg_result(&out, &[3, 2, 0, 0, 0, 0], 0, &segs[0]).is_err());
        assert!(seg_result(&out, &[2, 5, 0, 0, 0, 0], 0, &segs[0]).is_err());
        assert!(seg_result(&[11, -1, 99, -1, -1, 77, -1, -1], &acc, 0, &segs[0]).is_err());
    }

    #[test]
    fn skip_table_is_consistent() {
        let labels: std::collections::BTreeSet<&str> = Skip::ALL.iter().map(|s| s.label()).collect();
        assert_eq!(labels.len(), Skip::ALL.len(), "labels are unique");
        for (i, s) in Skip::ALL.iter().enumerate() {
            assert_eq!(s.index(), i);
            assert!(s.label().chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'), "{}", s.label());
            assert!(!s.text().is_empty());
        }
        assert_eq!(Skip::MixedRegime.text(), "mixed regime", "the R7 gate greps this text");
    }

    // ---- S-B9-CF-PACK4S ----

    fn sp(t: f32, top_p: f32, top_k: usize, seed: u64, ctr: u32) -> SampParams {
        SampParams::sampled(t, top_p, top_k, 0.0, seed, ctr).unwrap()
    }
    fn row(ps: &PackSamp, r: usize) -> [u32; 8] { ps.words[r * 8..r * 8 + 8].try_into().unwrap() }

    /// The planner with sampled lanes: they pair with each other and with greedy lanes (MIXED
    /// pairs); a penalized lane still runs alone; the dump diagnostic keeps sampled lanes out.
    #[test]
    fn sampled_lanes_pack_and_penalized_still_skip() {
        let base = LaneFacts { mtp_armed: true, depth: 4, ..Default::default() };
        let view = |slot: usize, f: LaneFacts| lv(slot, lane_block(&f), 0, false);
        let (greedy, sampled) = (base, LaneFacts { sampled: true, ..base });
        let penalized = LaneFacts { sampled: true, penalized: true, ..base };
        // sampled + sampled
        assert_eq!(plan(LanePack::Pairs, &[view(0, sampled), view(1, sampled)], 0).unwrap().pairs, vec![[0, 1]]);
        // mixed greedy + sampled, either slot order
        assert_eq!(plan(LanePack::Pairs, &[view(0, greedy), view(1, sampled)], 0).unwrap().pairs, vec![[0, 1]]);
        assert_eq!(plan(LanePack::Pairs, &[view(2, sampled), view(3, greedy)], 0).unwrap().pairs, vec![[2, 3]]);
        // three lanes of mixed kinds: one pair + the rotating lone role, whatever the kinds
        let three = [view(0, greedy), view(1, sampled), view(2, sampled)];
        let lone: Vec<usize> = (0..3).map(|rot| plan(LanePack::Pairs, &three, rot).unwrap().lone[0].0).collect();
        assert_eq!(lone, vec![2, 0, 1]);
        // a penalized lane stays out; the other two (one sampled, one greedy) still pack
        let p = plan(LanePack::Pairs, &[view(0, sampled), view(1, penalized), view(2, greedy)], 0).unwrap();
        assert_eq!(p.pairs, vec![[0, 2]]);
        assert_eq!(p.lone, vec![(1, Skip::Penalty)]);
        // sampled next to penalized only: nobody to pack with
        let p = plan(LanePack::Pairs, &[view(0, sampled), view(1, penalized)], 0).unwrap();
        assert!(p.pairs.is_empty());
        assert_eq!(p.lone, vec![(0, Skip::FewPackable), (1, Skip::Penalty)]);
        // the lone-verify sampler dump: sampled lanes are counted as `sampled`, greedy lanes pack
        let dump = |f: LaneFacts| LaneFacts { sampled_lone: true, ..f };
        let p = plan(LanePack::Pairs, &[view(0, dump(sampled)), view(1, dump(greedy)), view(2, dump(greedy))], 0).unwrap();
        assert_eq!(p.pairs, vec![[1, 2]]);
        assert_eq!(p.lone, vec![(0, Skip::Sampled)]);
    }

    /// Staging per row: flag 2 on a real-q lane's draft rows and 1 on its bonus row, 1 on every
    /// row of a match-mode lane, all-zero rows for a greedy lane; counters ctr_j + i.
    #[test]
    fn sampler_staging_flags_and_counters() {
        let (a, b) = (sp(1.0, 0.95, 20, 0xAAAA_0000_1111, 100), sp(0.7, 0.8, 20, 0xBBBB_0000_2222, 7));
        // segment 0: slot 2, 4 drafts (5 rows at 0); segment 1: slot 0, 2 drafts (3 rows at 5)
        let segs = [[2, 0, 5, 1], [0, 5, 3, 1]];
        // ratio + ratio
        let ps = pack_samp(&segs, &[SegSamp { samp: Some(a), ratio: true }, SegSamp { samp: Some(b), ratio: true }], 16).unwrap();
        assert_eq!(ps.words.len(), 16 * 8);
        assert!(ps.samp_live && ps.rq_live);
        let flags: Vec<u32> = (0..16).map(|r| row(&ps, r)[0]).collect();
        assert_eq!(flags, vec![2, 2, 2, 2, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0], "draft rows 2, bonus rows 1, unstaged rows 0");
        let ctrs: Vec<u32> = (0..8).map(|r| row(&ps, r)[6]).collect();
        assert_eq!(ctrs, vec![100, 101, 102, 103, 104, 7, 8, 9], "row i of segment j draws counter ctr_j + i");
        for r in 0..5 {
            assert_eq!(row(&ps, r), a_row(a, r as u32, if r < 4 { 2 } else { 1 }));
        }
        for i in 0..3 {
            assert_eq!(row(&ps, 5 + i), a_row(b, i as u32, if i < 2 { 2 } else { 1 }));
        }
        assert!(ps.words[8 * 8..].iter().all(|&w| w == 0), "no word past R rows is staged");
        // match + match: flag 1 on every row, the greedy graph family (rq_live false)
        let ps = pack_samp(&segs, &[SegSamp { samp: Some(a), ratio: false }, SegSamp { samp: Some(b), ratio: false }], 16).unwrap();
        assert!(ps.samp_live && !ps.rq_live);
        assert_eq!((0..8).map(|r| row(&ps, r)[0]).collect::<Vec<_>>(), vec![1; 8]);
        assert_eq!((0..8).map(|r| row(&ps, r)[6]).collect::<Vec<_>>(), vec![100, 101, 102, 103, 104, 7, 8, 9]);
        // greedy + greedy: nothing staged, nothing live (the PACK4 arm's state exactly)
        let ps = pack_samp(&segs, &[SegSamp::GREEDY, SegSamp::GREEDY], 16).unwrap();
        assert!(!ps.samp_live && !ps.rq_live && ps.words.iter().all(|&w| w == 0));
        // MIXED: greedy segment 0 stays all-zero (argmax) next to a real-q segment 1, and the reverse
        let ps = pack_samp(&segs, &[SegSamp::GREEDY, SegSamp { samp: Some(b), ratio: true }], 16).unwrap();
        assert!(ps.samp_live && ps.rq_live);
        assert!(ps.words[..5 * 8].iter().all(|&w| w == 0), "the greedy lane's rows are flag 0: both sampler kernels return");
        assert_eq!((5..8).map(|r| row(&ps, r)[0]).collect::<Vec<_>>(), vec![2, 2, 1]);
        let ps = pack_samp(&segs, &[SegSamp { samp: Some(a), ratio: true }, SegSamp::GREEDY], 16).unwrap();
        assert!(ps.words[5 * 8..].iter().all(|&w| w == 0));
        assert_eq!((0..5).map(|r| row(&ps, r)[0]).collect::<Vec<_>>(), vec![2, 2, 2, 2, 1]);
        // the counter wraps like the lone path's `ctr.wrapping_add(i)`
        let ps = pack_samp(&[[0, 0, 3, 1], [1, 3, 2, 1]], &[SegSamp { samp: Some(sp(1.0, 1.0, 0, 5, u32::MAX - 1)), ratio: true }, SegSamp::GREEDY], 16).unwrap();
        assert_eq!((0..3).map(|r| row(&ps, r)[6]).collect::<Vec<_>>(), vec![u32::MAX - 1, u32::MAX, 0]);
        // refusals: real-q without a sampler, rows outside the staged width
        assert!(pack_samp(&segs, &[SegSamp { samp: None, ratio: true }, SegSamp::GREEDY], 16).is_err());
        assert!(pack_samp(&[[0, 0, 8, 1], [1, 8, 8, 1]], &[SegSamp::GREEDY, SegSamp { samp: Some(b), ratio: true }], 15).is_err());
    }

    /// What the lone path stages for row i of a lane (`Lane::samp_rows` + `pack_flag`).
    fn a_row(p: SampParams, i: u32, flag: u32) -> [u32; 8] {
        SampParams::sampled(p.temperature, p.top_p, p.top_k as usize, p.min_p, p.seed, p.counter.wrapping_add(i)).unwrap().pack_flag(flag)
    }

    /// A lane's draws must not depend on its partner: the SAME lane staged next to two different
    /// partners (greedy and short; sampled with other parameters and long) and at either segment
    /// position has identical row blocks, and every one equals the lone path's row for that
    /// lane (same parameters, seed, counter ctr + i) up to the bonus row's flag.
    #[test]
    fn staged_rows_do_not_depend_on_the_partner() {
        let x = sp(1.0, 0.95, 20, 0x0123_4567_89AB_CDEF, 4242);
        for ratio in [true, false] {
            let lane = SegSamp { samp: Some(x), ratio };
            let partners = [(SegSamp::GREEDY, 3usize), (SegSamp { samp: Some(sp(0.7, 0.8, 40, 99, 1)), ratio }, 8usize),
                            (SegSamp { samp: Some(sp(1.3, 1.0, 0, 0xFFFF_FFFF_FFFF_FFFF, u32::MAX)), ratio }, 2usize)];
            for m in 2..=8usize {
                let mut seen: Vec<Vec<[u32; 8]>> = Vec::new();
                for &(partner, pm) in &partners {
                    // the lane as segment 0 (its slot is the lower one) ...
                    let ps = pack_samp(&[[0, 0, m as i32, 1], [1, m as i32, pm as i32, 1]], &[lane, partner], 16).unwrap();
                    seen.push((0..m).map(|i| row(&ps, i)).collect());
                    // ... and as segment 1, behind a partner of pm rows
                    let ps = pack_samp(&[[0, 0, pm as i32, 1], [1, pm as i32, m as i32, 1]], &[partner, lane], 16).unwrap();
                    seen.push((0..m).map(|i| row(&ps, pm + i)).collect());
                    assert_eq!(ps.rq_live, ratio, "the family follows the pair, the rows follow the lane");
                }
                assert!(seen.windows(2).all(|w| w[0] == w[1]), "m {m} ratio {ratio}: the lane's rows moved with its partner");
                for (i, r) in seen[0].iter().enumerate() {
                    let lone_flag = if ratio { 2 } else { 1 };
                    let want = a_row(x, i as u32, if ratio && i + 1 == m { 1 } else { lone_flag });
                    assert_eq!(*r, want, "row {i}");
                    // only the flag word can differ from the lone staging, and only on the bonus row
                    assert_eq!(r[1..], a_row(x, i as u32, lone_flag)[1..]);
                }
            }
        }
    }

    /// The draft-list hand-off: the list is sized from the scratch width (16 rows for the packed
    /// scratch, unchanged for <= 8 rows), a lane's rows 0..w go to its row offset, and every pair
    /// of legal depths stays inside the packed list.
    #[test]
    fn list_rows_and_handoff() {
        assert_eq!(MTP_MAX_K + 1, 8, "the served list: one row per draft pass + the spare");
        for w in 0..=8 { assert_eq!(rq_list_rows(w), 8, "scratch of {w} rows: the size before PACK4S"); }
        assert_eq!(rq_list_rows(PACK_ROWS_MAX), 16, "the packed scratch: a 16-row verify reads list rows up to 14");
        assert_eq!(rq_list_rows(12), 12);
        let dst = rq_list_rows(PACK_ROWS_MAX) * RQ_ROW;
        assert_eq!(RQ_ROW, 96);
        // segment 0 (row0 = 0) and segment 1 (row0 = m0)
        assert_eq!(list_handoff(0, 4, true, dst).unwrap(), Some((0, 4 * 96)));
        assert_eq!(list_handoff(5, 2, true, dst).unwrap(), Some((5 * 96, 2 * 96)));
        // greedy and match-mode lanes have no list
        assert_eq!(list_handoff(5, 2, false, dst).unwrap(), None);
        // every legal pair: both copies inside the list, disjoint, at the rows the kernel reads
        for w0 in 1..=7usize {
            for w1 in 1..=7usize {
                let (m0, _, r) = pack_rows(w0, w1).unwrap();
                let (o0, n0) = list_handoff(0, w0, true, dst).unwrap().unwrap();
                let (o1, n1) = list_handoff(m0, w1, true, dst).unwrap().unwrap();
                assert!(o0 + n0 <= o1 && o1 + n1 <= dst, "{w0}+{w1}");
                assert_eq!((o0 / RQ_ROW, o1 / RQ_ROW, (o1 + n1) / RQ_ROW), (0, m0, r - 1), "draft row i of segment j reads list row row0_j + i");
            }
        }
        // the OLD 8-row list cannot hold a pair's second segment: the resize is load-bearing
        assert!(list_handoff(8, 7, true, 8 * RQ_ROW).is_err());
        assert!(list_handoff(0, 8, true, dst).is_err(), "at most MTP_MAX_K draft rows");
        assert!(list_handoff(0, 0, true, dst).is_err());
    }

    /// The `arms=` provenance trace: which arm produced the token with generated index g.
    #[test]
    fn arm_trace() {
        let mut t = ArmTrace::default();
        assert_eq!(t.text(), "-");
        for (arm, n) in [('S', 2), ('P', 3), ('P', 1), ('c', 1), ('p', 1), ('P', 12)] { t.push(arm, n); }
        assert_eq!(t.text(), "S2P3P1c1p1P12");
        assert_eq!((t.rounds, t.packed), (6, 3));
        let s = t.text();
        assert_eq!(ArmTrace::arm_of(&s, 0), None, "the seam token is in no round");
        let arms: String = (1..=20).map(|g| ArmTrace::arm_of(&s, g).unwrap()).collect();
        assert_eq!(arms, "SSPPPPcpPPPPPPPPPPPP");
        assert_eq!(ArmTrace::arm_of(&s, 21), None, "past the recorded tokens");
        assert_eq!(ArmTrace::arm_of("-", 1), None);
        // the cap: later rounds are counted, not spelled out, and the text says so
        let mut t = ArmTrace::default();
        for _ in 0..ARM_TRACE_ROUNDS + 5 { t.push('P', 2); }
        assert_eq!((t.rounds, t.packed), (ARM_TRACE_ROUNDS + 5, ARM_TRACE_ROUNDS + 5));
        assert!(t.text().ends_with('+') && t.text().matches('P').count() == ARM_TRACE_ROUNDS);
        assert_eq!(ArmTrace::arm_of(&t.text(), 2 * ARM_TRACE_ROUNDS), Some('P'));
        assert_eq!(ArmTrace::arm_of(&t.text(), 2 * ARM_TRACE_ROUNDS + 1), None);
    }
}
