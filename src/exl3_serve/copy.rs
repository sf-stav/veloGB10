//! Copy-span drafting on the serve path (the `copy_span` series on /metrics): the arm state machine, boot lines, resolvers for the EXL3 and
//! trunk servers, the counters and their /metrics rendering, and the scheduler methods that propose, price and verify copy rounds
//! (`cl_*`, `cpv_*`, `step_cpv`, `copy_propose*`). The proposer is pure integer logic so every TP rank computes the same proposal (see the comment below).

use super::*;

// ---- WIDEVERIFY1: copy-span drafting (the /metrics series `velogb10_copy_span_*`) ----------
// A lone lane's speculative round whose drafts are tokens copied from the lane's own context
// (src/copy_span.rs) instead of the neural drafter's. Scope: the EXL3 server at TP=1 and at the TP
// worlds in `COPY_SPAN_TP_WORLDS`, a solo lane's step (step_mtp with `solo`); packed / multi-lane
// steps keep the neural paths.
//
// WIDEVERIFY1 TP leg: under TP every rank holds the SAME per-lane proposer and computes the SAME
// proposal locally, with no message: the proposer is pure integer logic (fixed-seed hash map, no
// clock, no randomness) over inputs every rank already holds at the same program point — the prompt
// ids the admit ships (`WireRequest::prompt`), the seam token (agreed by the `seam` tp_round), every
// emitted token (agreed by each round's tp_round digest before round_emit appends it), and the
// round's depth `k_eff` / `plain_step` / `solo` (lockstep state the neural arms already rely on for
// their own widths). The copied ids and the width are then agreed BEFORE anything of the round
// launches (`TpSync::pre_verify` with 0 head passes launched, before the head prime), and the
// round's tp_round digest covers the drafts and emitted tokens after it; a disagreement aborts the
// link loudly like every other TP disagreement.

/// How `--copy-span` resolved on this boot (the `velogb10_copy_span_state` gauge).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CopySpanState {
    /// not asked (`--copy-span off`, the default)
    NotAsked = 0,
    /// asked and armed (EXL3 server, TP=1 or a TP world in `COPY_SPAN_TP_WORLDS`, speculation live)
    On = 1,
    /// asked, runs off: a TP server whose world size is not in `COPY_SPAN_TP_WORLDS` (not gated)
    OffTp = 2,
    /// asked, runs off: not an EXL3 pack (the arm lives in the EXL3 speculative round only)
    OffFamily = 3,
    /// asked, runs off: speculation itself is off (`--mtp off` or a pack without a draft head)
    OffNoSpec = 4,
}

impl CopySpanState {
    pub(crate) fn reason(self) -> &'static str {
        match self {
            CopySpanState::NotAsked | CopySpanState::On => "",
            CopySpanState::OffTp => "tp",
            CopySpanState::OffFamily => "family",
            CopySpanState::OffNoSpec => "no_speculation",
        }
    }
}

/// The armed copy arm of one EXL3 scheduler: the proposer config (with the start ramp
/// `--copy-span-start-w` in `cfg.start_w`) and the most tokens one copy round proposes
/// (`--copy-span-max-w`, clamped again per round by `copy_width`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CopyArm {
    pub cfg: crate::copy_span::CopySpanCfg,
    pub max_w: usize,
}

/// The TP worlds the copy arm serves (wired and gated: TP=2 and TP=4, WIDEVERIFY1 TP leg). Any other
/// world > 1 runs the flag off with one counted boot line (`velogb10_copy_span_state 2`).
pub(crate) const COPY_SPAN_TP_WORLDS: [usize; 2] = [2, 4];

/// `--copy-span` resolution, a pure function of the boot (unit-tested): `asked` = the flag,
/// `exl3` = an EXL3 pack, `world` = the TP world (1 = single box), `spec_live` = speculation can
/// run on this boot. Never a refusal (AGENTS 1b-bis item 8): an unsupported setup runs off.
/// Every rank of a TP server resolves it from the same shipped options and the same world, so the
/// head and the nodes arm (or skip) the arm alike.
pub(crate) fn copy_span_decide(asked: bool, exl3: bool, world: usize, spec_live: bool) -> CopySpanState {
    if !asked {
        CopySpanState::NotAsked
    } else if !exl3 {
        CopySpanState::OffFamily
    } else if world > 1 && !COPY_SPAN_TP_WORLDS.contains(&world) {
        CopySpanState::OffTp
    } else if !spec_live {
        CopySpanState::OffNoSpec
    } else {
        CopySpanState::On
    }
}

/// The width of one copy round: the scheduler's clamped depth (`round_depth`: budget, end of the
/// KV window, the sparse-attention switch), `--copy-span-max-w`, and the chain verify's width
/// (`copy_depth_cap`). The depth passed in is the neural round's `k_eff` (<= MTP_MAX_K) unless the
/// wide copy verify is armed, when step_mtp clamps COPY_MAX_K the same way (`round_depth_for`).
pub(crate) fn copy_width(k_eff: usize, max_w: usize) -> usize {
    k_eff.min(max_w).min(copy_depth_cap(max_w))
}

/// WIDEVERIFY1 refinement 2: `--copy-span-max-w` above MTP_MAX_K arms the WIDE copy verify (a copy
/// round verifies up to COPY_MAX_K = 15 copied tokens, 16 rows; the neural rounds keep MTP_MAX_K).
/// Default (10) = not armed: the served scratch, its planes, graphs and every round are as before.
/// UNVERIFIED ON GPU until gated on hardware.
pub(crate) fn copy_span_wide(max_w: usize) -> bool {
    max_w > crate::exl3_forward::MTP_MAX_K
}

/// The verify-width ceiling of a copy round: COPY_MAX_K when the wide copy verify is armed, else
/// the chain verify's MTP_MAX_K.
pub(crate) fn copy_depth_cap(max_w: usize) -> usize {
    if copy_span_wide(max_w) { crate::exl3_forward::COPY_MAX_K } else { crate::exl3_forward::MTP_MAX_K }
}

/// WIDEVERIFY1 refinement 2: the lone-verify rows of the served scratch (`Scratch::vrows`) — COPY_VROWS
/// only when the copy arm will be armed (same resolution as `copy_span_resolve`) with the wide
/// verify; MTP_MAX_K + 1 otherwise (the pre-refinement layout).
pub(crate) fn copy_span_scratch_vrows(asked: bool, world: usize, spec_live: bool, max_w: usize) -> usize {
    if copy_span_decide(asked, true, world, spec_live) == CopySpanState::On && copy_span_wide(max_w) {
        crate::exl3_forward::COPY_VROWS
    } else {
        crate::exl3_forward::MTP_MAX_K + 1
    }
}

/// The one boot line for `--copy-span` (None when not asked). `world` = the TP world (1 = one box).
pub(crate) fn copy_span_boot_line(st: CopySpanState, min_n: usize, max_w: usize, world: usize, yield_gate: bool,
                                  start_w: usize) -> Option<String> {
    match st {
        CopySpanState::NotAsked => None,
        CopySpanState::On => {
            let tp = if world > 1 {
                format!("; TP={world}: every rank runs the same proposer over the shipped prompt and the lockstep-agreed tokens, \
and the copied ids + width are agreed across ranks (pre-verify) before the round launches")
            } else { String::new() };
            let yg = if yield_gate {
                "; yield gate ON (--copy-span-yield-gate: per request, a NEW copy starts only while its copy rounds emit enough tokens \
per round against its neural rounds; every 16th held-back start runs as a probe; a running copy is never held back)"
            } else {
                "; yield gate OFF (--copy-span-yield-gate off: every established match starts a copy)"
            };
            let cap = copy_width(usize::MAX, max_w);
            let ramp = if start_w < cap {
                format!("; start ramp: a copy's first round proposes at most {start_w} tokens, each following round of it doubles that \
up to {cap} (--copy-span-start-w {start_w})")
            } else {
                format!("; no start ramp: every copy round proposes up to {cap} tokens (--copy-span-start-w {start_w} >= the cap)")
            };
            let wide = if copy_span_wide(max_w) {
                format!("; WIDE copy verify: copy rounds verify up to {} rows, neural rounds keep {} \
(--copy-span-max-w {max_w} > {}; not yet gated on hardware)",
                        crate::exl3_forward::COPY_MAX_K + 1, crate::exl3_forward::MTP_MAX_K + 1, crate::exl3_forward::MTP_MAX_K)
            } else { String::new() };
            Some(format!("[exl3-serve] copy-span: ON (a lone request's speculative round drafts tokens copied from its own \
context when a copy is established: search match >= {min_n} tokens (--copy-span-min-n), then the copy cursor; up to {cap} tokens per round \
(--copy-span-max-w {max_w}, verify width {}){ramp}{wide}; otherwise the neural drafter, unchanged; 2-4 concurrent requests keep the neural paths in v1{tp}{yg}; \
counters velogb10_copy_span_*)", copy_depth_cap(max_w)))
        }
        CopySpanState::OffTp => Some(format!("[exl3-serve] copy-span: asked (--copy-span on), runs OFF on this TP={world} server: the copy \
arm serves one box and TP worlds {:?} only (this world size is not wired or gated); speculation is unchanged (velogb10_copy_span_state 2)",
                                             COPY_SPAN_TP_WORLDS)),
        CopySpanState::OffFamily => Some("[copy-span] asked (--copy-span on), runs OFF: copy-span drafting is built into the EXL3 \
and trunk (NVFP4/bf16) speculative rounds only; this pack's server is unchanged (velogb10_copy_span_state 3)".to_string()),
        CopySpanState::OffNoSpec => Some("[exl3-serve] copy-span: asked (--copy-span on), runs OFF: speculation is off on this boot \
(--mtp off or no draft head), and a copy round is a speculative round (velogb10_copy_span_state 4)".to_string()),
    }
}

/// Resolve `--copy-span` for an EXL3 boot: prints the boot line, sets the state gauge, returns the
/// armed arm (None = off). Reads `--copy-span-min-n` / `--copy-span-max-w` only when armed.
pub(crate) fn copy_span_resolve(world: usize, spec_live: bool) -> Result<Option<CopyArm>> {
    let asked = crate::opts::on(crate::opt!("copy-span"));
    let st = copy_span_decide(asked, true, world, spec_live);
    let (min_n, max_w, yg, start_w) = if st == CopySpanState::On {
        (crate::opts::ranged(crate::opt!("copy-span-min-n")).map_err(|e| anyhow::anyhow!("{e}"))? as usize,
         crate::opts::ranged(crate::opt!("copy-span-max-w")).map_err(|e| anyhow::anyhow!("{e}"))? as usize,
         // default OFF (registry def "off"): opt-in until its benefit is measured; an unset value reads as off
         crate::opts::var(crate::opt!("copy-span-yield-gate")).map_or(false, |v| crate::opts::is_on_value(&v)),
         crate::opts::ranged(crate::opt!("copy-span-start-w")).map_err(|e| anyhow::anyhow!("{e}"))? as usize)
    } else { (0, 0, false, 0) };
    copy_span_set_state(st, min_n, max_w);
    CS_WORLD.store(world as u64, std::sync::atomic::Ordering::Relaxed);
    CS_YG_ON.store(yg as u64, std::sync::atomic::Ordering::Relaxed);
    CS_START_W.store(start_w as u64, std::sync::atomic::Ordering::Relaxed);
    if let Some(line) = copy_span_boot_line(st, min_n, max_w, world, yg, start_w) { crate::rprintln!("{line}"); }
    Ok((st == CopySpanState::On).then(|| CopyArm { cfg: copy_arm_cfg(min_n, max_w, yg, start_w), max_w }))
}

/// COPYREC1: resolve `--copy-span-record` for an EXL3 boot (every TP rank resolves it alike from the shipped options):
/// None when not asked or when the copy arm is not armed (one line, counted, never a refusal). With
/// `--copy-span-record-dir` the record is loaded from (and persisted to) that directory on THIS rank's box; any trouble
/// there degrades with a line and a counted reason. Prints ONE boot line stating the record's state and size, so a
/// benchmark is never run against a populated record unknowingly.
pub(crate) fn copy_record_resolve(copy_armed: bool, model_dir: &str, vocab: usize, world: usize)
                                  -> Option<Box<crate::copy_record::CopyRecord>> {
    use crate::copy_record as cr;
    let asked = crate::opts::on(crate::opt!("copy-span-record"));
    let ro = crate::opts::on(crate::opt!("copy-span-record-read-only"));
    let st = cr::resolve(asked, copy_armed);
    cr::set_state(st, ro);
    match st {
        cr::State::NotAsked => return None,
        cr::State::OffNoCopySpan => {
            crate::rprintln!("[exl3-serve] copy-span-record: asked (--copy-span-record on), runs OFF: the record drafts through the copy \
arm, which is not armed on this boot (--copy-span off, or it runs off); nothing is recorded (velogb10_copy_span_record_state 2)");
            return None;
        }
        cr::State::On => {}
    }
    let tok_bytes = std::fs::read(format!("{}/tokenizer.json", model_dir.trim_end_matches('/'))).unwrap_or_default();
    let ident = cr::Identity::from_tokenizer_bytes(&tok_bytes, vocab as u32);
    let mib = crate::opts::ranged(crate::opt!("copy-span-record-mib")).map_or(cr::DEFAULT_MIB, |v| v as u64);
    let o = cr::Options { dir: crate::opts::get(crate::opt!("copy-span-record-dir")).map(std::path::PathBuf::from),
                          cap_bytes: mib << 20, read_only: ro };
    let (r, notes) = cr::CopyRecord::open(ident, &o);
    for f in &notes.fallbacks { cr::count_fallback(*f); }
    for l in &notes.lines { crate::rprintln!("[exl3-serve] copy-span-record: {l}"); }
    cr::publish(&r);
    crate::rprintln!("[exl3-serve] copy-span-record: ON (--copy-span-record on{}): {}. When a lone request's own context has \
nothing to copy, its copy arm drafts from the record's longest match (>= --copy-span-min-n tokens) of its last tokens; the model \
verifies every copied token (output unchanged); record copies are offered while their landing rate per match band beats the \
request's own drafts. A POPULATED RECORD MAKES REPEATED PROMPTS FASTER: benchmark with it off or empty (GET / DELETE \
/v1/copy-record){}. Counters velogb10_copy_span_record_*",
                     if ro { ", read-only: nothing is added" } else { "" }, r.describe(),
                     if world > 1 { format!("; TP={world}: every rank keeps its own identical record (same ids, same order) and the ranks \
compare it at boot") } else { String::new() });
    Some(Box::new(r))
}

/// COPYREC1 under TP: every rank loaded its own record; compare the digests (all ranks, at the same boot point, after
/// the boot config agree). A difference (another topology wrote one rank's directory, a disk failed, ...) makes EVERY
/// rank start empty and stop persisting for this boot (the files stay): one line, counted. The exchange runs whenever
/// the record is armed (the arming is part of the agreed boot config), so every rank calls it.
pub(super) fn copy_record_tp_agree(s: &mut Exl3Scheduler, rank: i32) -> Result<()> {
    let Some(r) = s.record.as_mut() else { return Ok(()) };
    let d = r.ids.digest();
    let held = r.ids.held();
    let all = crate::net::exchange_u32s_all(&[0xC0DE_EC01, d as u32, (d >> 32) as u32, held as u32], 8)
        .context("TP copy-record digest exchange (peer dead or link aborted)")?;
    anyhow::ensure!(all.iter().all(|f| f.len() >= 4 && f[0] == 0xC0DE_EC01),
                    "TP copy-record digest exchange: a rank sent another frame ({:?})", all.iter().map(|f| f.first()).collect::<Vec<_>>());
    if all.iter().all(|f| f[1..4] == all[0][1..4]) {
        crate::drprintln!("[exl3-serve] TP rank {rank}: copy-span-record identical on every rank (digest {d:016x}, {held} ids)");
        return Ok(());
    }
    crate::copy_record::count_fallback(crate::copy_record::Fallback::TpDigest);
    let ids: Vec<String> = all.iter().map(|f| f[3].to_string()).collect();
    crate::rprintln!("[exl3-serve] copy-span-record: the TP ranks loaded DIFFERENT records (ids held per rank: {}); every rank starts \
EMPTY and keeps the record in memory only for this boot (the directories are left as they are) (velogb10_copy_span_record_fallback_total{{reason=\"tp_digest\"}})",
                     ids.join(" / "));
    r.forget_for_boot();
    crate::copy_record::publish(r);
    Ok(())
}

/// The proposer config of the armed copy arm: `--copy-span-min-n`, the start ramp
/// `--copy-span-start-w`, and the yield gate's W = the copy width cap (`copy_width(..,
/// --copy-span-max-w)`, 10 on the EXL3 chain verify) when `--copy-span-yield-gate` is on.
pub(crate) fn copy_arm_cfg(min_n: usize, max_w: usize, yield_gate: bool, start_w: usize) -> crate::copy_span::CopySpanCfg {
    let c = crate::copy_span::CopySpanCfg::new(min_n).with_start_w(start_w.max(1));
    if yield_gate { c.with_yield_gate(copy_width(usize::MAX, max_w)) } else { c }
}

/// The start ramp as the request-end debug line prints it (`usize::MAX` = no ramp).
pub(super) fn cs_start_w_str(w: usize) -> String {
    if w == usize::MAX { "off".to_string() } else { w.to_string() }
}

// ---- COPY27B: the copy arm of the trunk server (NVFP4 / bf16 packs served by `batch::BatchScheduler`) ----
// Same flags, same proposer (`CopySpan`), same `velogb10_copy_span_*` series. A lone request's speculative
// round (DFlash2 chain or the MTP head) may be replaced by a copy round whose drafts are the copied tokens;
// the trunk's verify (batch-invariant GEMM, N padded to 16) checks them exactly like the drafter's.

/// The TP worlds the trunk copy arm serves besides one box (none until gated; another world runs it off,
/// counted, never refused).
pub(crate) const COPY_SPAN_TRUNK_TP_WORLDS: [usize; 2] = [2, 4];

/// The most copied tokens one trunk copy round verifies: the verify ceiling MAX_VERIFY = 16 rows.
pub(crate) fn copy_span_trunk_width(max_w: usize) -> usize {
    max_w.min(crate::gpu::MAX_VERIFY - 1)
}

/// Pure resolution for the trunk server (unit-tested): like `copy_span_decide` for an EXL3 pack, with the
/// trunk's own TP worlds.
pub(crate) fn copy_span_decide_trunk(asked: bool, world: usize, spec_live: bool) -> CopySpanState {
    if !asked {
        CopySpanState::NotAsked
    } else if world > 1 && !COPY_SPAN_TRUNK_TP_WORLDS.contains(&world) {
        CopySpanState::OffTp
    } else if !spec_live {
        CopySpanState::OffNoSpec
    } else {
        CopySpanState::On
    }
}

/// Resolve `--copy-span` for a trunk (`BatchScheduler`) boot: sets the gauges, prints one line, returns the
/// armed arm (None = off). `spec_live` = the boot can speculate (an MTP head or a DFlash2 round, policy not
/// forced off). Every TP rank resolves it alike from the shipped options and the same world.
pub(crate) fn copy_span_resolve_trunk(world: usize, spec_live: bool) -> Option<CopyArm> {
    let asked = crate::opts::on(crate::opt!("copy-span"));
    let st = copy_span_decide_trunk(asked, world, spec_live);
    let rd = |o| crate::opts::ranged(o).ok().map(|v| v as usize);
    let (min_n, max_w, yg, start_w) = if st == CopySpanState::On {
        (rd(crate::opt!("copy-span-min-n")).unwrap_or(16),
         rd(crate::opt!("copy-span-max-w")).unwrap_or(10),
         crate::opts::var(crate::opt!("copy-span-yield-gate")).map_or(false, |v| crate::opts::is_on_value(&v)),
         rd(crate::opt!("copy-span-start-w")).unwrap_or(4))
    } else { (0, 0, false, 0) };
    copy_span_set_state(st, min_n, max_w);
    CS_WORLD.store(world as u64, std::sync::atomic::Ordering::Relaxed);
    CS_YG_ON.store(yg as u64, std::sync::atomic::Ordering::Relaxed);
    CS_START_W.store(start_w as u64, std::sync::atomic::Ordering::Relaxed);
    CS_TRUNK.store(1, std::sync::atomic::Ordering::Relaxed);
    let cap = copy_span_trunk_width(max_w.max(1));
    let line = match st {
        CopySpanState::NotAsked => None,
        CopySpanState::On => Some(format!("[serve] copy-span: ON (a lone request's speculative round drafts tokens copied from its own \
context when a copy is established: search match >= {min_n} tokens (--copy-span-min-n), then the copy cursor; up to {cap} copied tokens per \
round (--copy-span-max-w {max_w}, verify ceiling {} rows); {}; a copy round replaces the round's drafter (DFlash2 chain or MTP head), \
otherwise the drafter runs unchanged; 2-16 concurrent requests keep the batched paths (counted in \
velogb10_copy_span_skipped_rounds_total){}; yield gate {}; counters velogb10_copy_span_*)", crate::gpu::MAX_VERIFY,
            if start_w < cap { format!("start ramp {start_w} doubling per round (--copy-span-start-w)") }
            else { "no start ramp".to_string() },
            if world > 1 { format!("; TP={world}: every rank runs the same proposer over the shipped prompt and the agreed tokens, and \
the ranks exchange the copied ids + width before each copy round (a difference makes every rank run the drafter round)") } else { String::new() },
            if yg { "ON (--copy-span-yield-gate: a new copy starts only while this request's copy rounds emit at least as many \
tokens per round as its drafter rounds; every 16th held-back start runs as a probe)" } else { "off" })),
        CopySpanState::OffTp => Some(format!("[serve] copy-span: asked (--copy-span on), runs OFF on this TP={world} server: the trunk \
copy arm serves one box{} only (not wired or gated for this world); speculation is unchanged (velogb10_copy_span_state 2)",
            if COPY_SPAN_TRUNK_TP_WORLDS.is_empty() { String::new() } else { format!(" and TP worlds {:?}", COPY_SPAN_TRUNK_TP_WORLDS) })),
        CopySpanState::OffNoSpec => Some("[serve] copy-span: asked (--copy-span on), runs OFF: speculation is off on this boot \
(--mtp off or no draft head), and a copy round is a speculative round (velogb10_copy_span_state 4)".to_string()),
        CopySpanState::OffFamily => None,
    };
    if let Some(l) = line { crate::rprintln!("{l}"); }
    (st == CopySpanState::On).then(|| CopyArm { cfg: copy_arm_cfg_w(min_n.max(1), cap, yg, start_w.max(1)), max_w: cap })
}

/// `copy_arm_cfg` with the yield gate's W given directly (the trunk's width cap is not the EXL3 one).
pub(crate) fn copy_arm_cfg_w(min_n: usize, cap: usize, yield_gate: bool, start_w: usize) -> crate::copy_span::CopySpanCfg {
    let c = crate::copy_span::CopySpanCfg::new(min_n).with_start_w(start_w.max(1));
    if yield_gate { c.with_yield_gate(cap).with_yield_ratio(TRUNK_YIELD_RATIO) } else { c }
}

/// COPY27B: the trunk yield gate's bar (x256): a new copy may start while this request's copy rounds emit at least
/// as many tokens per round as its drafter rounds. Measured on the 27B (sampled verify, one box): a copy round costs
/// 111 ms at 3 rows and 148 ms at 16 rows, a depth-8 MTP round about 146 ms, so the two round costs are about equal.
pub(crate) const TRUNK_YIELD_RATIO: i64 = 256;

/// Why a trunk speculative step could not take a copy round (the round ran as before). Counted per step in
/// `velogb10_copy_span_skipped_rounds_total{reason}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CopySkip {
    /// 2+ requests in the step: the batched paths serve them (v1: lone requests only)
    Concurrent = 0,
    /// a JSON-schema constrained request (its mask chain rides the drafter's rounds only)
    Schema = 1,
    /// the step's speculation source has no copy wiring (DSpark, DFlash v1)
    Source = 2,
    /// the lane's drafter state is not live (never primed / went stale): it decodes plain
    NotPrimed = 3,
    /// speculation is off for this step (policy off / plain decode)
    SpecOff = 4,
    /// TP: the ranks proposed different copy rounds (every rank ran the drafter round instead)
    TpDisagree = 5,
}

const CS_SKIP_NAMES: [&str; 6] = ["concurrent", "schema", "source", "not_primed", "spec_off", "tp_disagree"];

static CS_SKIP: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// 1 = the copy arm was resolved by the trunk server (not the EXL3 one)
static CS_TRUNK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn copy_span_count_skip(r: CopySkip) {
    CS_SKIP[r as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn copy_span_add_host_us(us: u64) {
    CS_HOST_US.fetch_add(us, std::sync::atomic::Ordering::Relaxed);
}

/// COPY27B: copy rounds served to a lane without live drafter state (after a batched period, or speculation off).
static CS_BARE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn copy_span_count_bare() {
    CS_BARE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// TP: one copy round whose ids + width every rank agreed on before it launched.
pub(crate) fn copy_span_count_tp_agreed() {
    CS_TP_AGREED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// A non-EXL3 server was asked for `--copy-span on`: one line, counted, the server runs unchanged.
pub fn copy_span_family_fallback() {
    if crate::opts::on(crate::opt!("copy-span")) {
        let st = copy_span_decide(true, false, 1, true);
        copy_span_set_state(st, 0, 0);
        if let Some(line) = copy_span_boot_line(st, 0, 0, 1, false, 0) { eprintln!("{line}"); }
    }
    // ULTRA2: --copy-span-lanes on a DeepSeek V4 / Naive bundle: one line + the fallback counter, never a refusal
    let _ = crate::copy_lanes::resolve_other_server();
}

static CS_STATE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// the TP world `--copy-span` resolved for (1 = one box; the `velogb10_copy_span_tp_world` gauge)
static CS_WORLD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// copy rounds whose copied ids + width were agreed across ranks before launch (TP only)
pub(super) static CS_TP_AGREED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static CS_MIN_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static CS_MAX_W: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static CS_FALLBACK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static CS_ROUNDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static CS_OFFERED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static CS_ACCEPTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static CS_EMITTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static CS_ZERO: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static CS_MS_X1000: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static CS_PROP_SEARCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(super) static CS_PROP_CURSOR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static CS_MISS_NOMATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(super) static CS_MISS_BELOW: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(super) static CS_END_MISMATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static CS_END_CTX: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static CS_END_SHORT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static CS_END_UNUSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(super) static CS_HOST_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// --copy-span-yield-gate as armed (the `velogb10_copy_span_yield_gate` gauge) + its two counters
static CS_YG_ON: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static CS_YG_DENIED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static CS_YG_PROBES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// --copy-span-start-w as armed (the `velogb10_copy_span_start_w` gauge) + the narrow proposals it made
static CS_START_W: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static CS_NARROW_SEARCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static CS_NARROW_CURSOR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn copy_span_set_state(st: CopySpanState, min_n: usize, max_w: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    CS_STATE.store(st as u64, Relaxed);
    CS_MIN_N.store(min_n as u64, Relaxed);
    CS_MAX_W.store(max_w as u64, Relaxed);
    if !matches!(st, CopySpanState::NotAsked | CopySpanState::On) {
        CS_FALLBACK.fetch_add(1, Relaxed);
    }
}

/// Fold the change of one lane's proposer counters (before -> after a propose or a commit) into the
/// process totals (proposals by source, misses by reason, how copies ended).
pub(crate) fn copy_span_count_delta(a: &crate::copy_span::CopySpanCounters, b: &crate::copy_span::CopySpanCounters) {
    use std::sync::atomic::Ordering::Relaxed;
    let d = |x: u64, y: u64| y.saturating_sub(x);
    CS_PROP_SEARCH.fetch_add(d(a.proposals_search, b.proposals_search), Relaxed);
    CS_PROP_CURSOR.fetch_add(d(a.proposals_cursor, b.proposals_cursor), Relaxed);
    CS_MISS_NOMATCH.fetch_add(d(a.miss_no_match, b.miss_no_match), Relaxed);
    CS_MISS_BELOW.fetch_add(d(a.miss_below_min, b.miss_below_min), Relaxed);
    CS_END_MISMATCH.fetch_add(d(a.end_mismatch, b.end_mismatch), Relaxed);
    CS_END_CTX.fetch_add(d(a.end_of_context, b.end_of_context), Relaxed);
    CS_END_SHORT.fetch_add(d(a.end_short_emit, b.end_short_emit), Relaxed);
    CS_END_UNUSED.fetch_add(d(a.end_unused, b.end_unused), Relaxed);
    CS_YG_DENIED.fetch_add(d(a.gate_denied, b.gate_denied), Relaxed);
    CS_YG_PROBES.fetch_add(d(a.gate_probes, b.gate_probes), Relaxed);
    CS_NARROW_SEARCH.fetch_add(d(a.narrow_search, b.narrow_search), Relaxed);
    CS_NARROW_CURSOR.fetch_add(d(a.narrow_cursor, b.narrow_cursor), Relaxed);
}

/// Count one served copy round (`offered` drafts, `accepted` of them, `emitted` tokens, its ms).
pub(crate) fn copy_span_count_round(offered: usize, accepted: usize, emitted: usize, ms: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    if CS_ROUNDS.fetch_add(1, Relaxed) == 0 {
        let tag = if CS_TRUNK.load(Relaxed) == 1 { "[serve]" } else { "[exl3-serve]" };
        crate::rprintln!("{tag} copy-span: first copy round served (a speculative round drafted from the context, no draft pass)");
    }
    CS_OFFERED.fetch_add(offered as u64, Relaxed);
    CS_ACCEPTED.fetch_add(accepted as u64, Relaxed);
    CS_EMITTED.fetch_add(emitted as u64, Relaxed);
    CS_ZERO.fetch_add((accepted == 0) as u64, Relaxed);
    CS_MS_X1000.fetch_add((ms.max(0.0) * 1000.0) as u64, Relaxed);
}

/// Append the `velogb10_copy_span_*` series (rendered by every server; the state gauge says
/// whether the arm was asked for and armed).
pub fn copy_span_metrics_append(o: &mut String) {
    use std::sync::atomic::Ordering::Relaxed;
    let st = CS_STATE.load(Relaxed);
    o.push_str("# HELP velogb10_copy_span_state --copy-span on this boot: 0 = not asked, 1 = on, 2 = asked but off (a TP world size the arm is not wired for), 3 = asked but off (a pack whose server has no copy arm: DeepSeek V4 / Naive bundles), 4 = asked but off (speculation off).\n# TYPE velogb10_copy_span_state gauge\n");
    o.push_str(&format!("velogb10_copy_span_state {st}\n"));
    o.push_str("# HELP velogb10_copy_span_tp_world The TP world --copy-span resolved for (1 = one box).\n# TYPE velogb10_copy_span_tp_world gauge\n");
    o.push_str(&format!("velogb10_copy_span_tp_world {}\n", CS_WORLD.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_tp_agreed_total Copy rounds whose copied ids and width every TP rank agreed on before the round launched (0 on one box).\n# TYPE velogb10_copy_span_tp_agreed_total counter\n");
    o.push_str(&format!("velogb10_copy_span_tp_agreed_total {}\n", CS_TP_AGREED.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_min_n Resolved --copy-span-min-n (0 when the arm is off).\n# TYPE velogb10_copy_span_min_n gauge\n");
    o.push_str(&format!("velogb10_copy_span_min_n {}\n", CS_MIN_N.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_max_w Resolved --copy-span-max-w (0 when the arm is off).\n# TYPE velogb10_copy_span_max_w gauge\n");
    o.push_str(&format!("velogb10_copy_span_max_w {}\n", CS_MAX_W.load(Relaxed)));
    let reason = match st { 2 => "tp", 3 => "family", 4 => "no_speculation", _ => "" };
    o.push_str("# HELP velogb10_copy_span_fallback_total Boots that were asked for --copy-span on and run it off (never a refusal), by reason.\n# TYPE velogb10_copy_span_fallback_total counter\n");
    for r in ["tp", "family", "no_speculation"] {
        let v = if r == reason { CS_FALLBACK.load(Relaxed) } else { 0 };
        o.push_str(&format!("velogb10_copy_span_fallback_total{{reason=\"{r}\"}} {v}\n"));
    }
    o.push_str("# HELP velogb10_copy_span_rounds_total Speculative rounds whose drafts were copied from the context (no draft pass). Also counted in velogb10_spec_rounds_total.\n# TYPE velogb10_copy_span_rounds_total counter\n");
    o.push_str(&format!("velogb10_copy_span_rounds_total {}\n", CS_ROUNDS.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_offered_tokens_total Copied tokens offered to the verify by copy rounds.\n# TYPE velogb10_copy_span_offered_tokens_total counter\n");
    o.push_str(&format!("velogb10_copy_span_offered_tokens_total {}\n", CS_OFFERED.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_accepted_tokens_total Copied tokens the verify accepted.\n# TYPE velogb10_copy_span_accepted_tokens_total counter\n");
    o.push_str(&format!("velogb10_copy_span_accepted_tokens_total {}\n", CS_ACCEPTED.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_emitted_tokens_total Tokens emitted by copy rounds (accepted + 1 per round).\n# TYPE velogb10_copy_span_emitted_tokens_total counter\n");
    o.push_str(&format!("velogb10_copy_span_emitted_tokens_total {}\n", CS_EMITTED.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_zero_accept_rounds_total Copy rounds whose first copied token was rejected.\n# TYPE velogb10_copy_span_zero_accept_rounds_total counter\n");
    o.push_str(&format!("velogb10_copy_span_zero_accept_rounds_total {}\n", CS_ZERO.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_round_seconds_total Wall time of the copy rounds (prime + verify + re-prime; host side included).\n# TYPE velogb10_copy_span_round_seconds_total counter\n");
    o.push_str(&format!("velogb10_copy_span_round_seconds_total {:.6}\n", CS_MS_X1000.load(Relaxed) as f64 / 1e6));
    o.push_str("# HELP velogb10_copy_span_host_seconds_total Host time of the proposer (index upkeep + proposals) on the scheduler thread.\n# TYPE velogb10_copy_span_host_seconds_total counter\n");
    o.push_str(&format!("velogb10_copy_span_host_seconds_total {:.6}\n", CS_HOST_US.load(Relaxed) as f64 / 1e6));
    o.push_str("# HELP velogb10_copy_span_proposals_total Copy proposals made, by source (search = a fresh context match >= min_n; cursor = continuing a live copy).\n# TYPE velogb10_copy_span_proposals_total counter\n");
    o.push_str(&format!("velogb10_copy_span_proposals_total{{source=\"search\"}} {}\n", CS_PROP_SEARCH.load(Relaxed)));
    o.push_str(&format!("velogb10_copy_span_proposals_total{{source=\"cursor\"}} {}\n", CS_PROP_CURSOR.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_misses_total Searches that made no proposal, by reason (no_match = no earlier occurrence; below_min = a match shorter than min_n).\n# TYPE velogb10_copy_span_misses_total counter\n");
    o.push_str(&format!("velogb10_copy_span_misses_total{{reason=\"no_match\"}} {}\n", CS_MISS_NOMATCH.load(Relaxed)));
    o.push_str(&format!("velogb10_copy_span_misses_total{{reason=\"below_min\"}} {}\n", CS_MISS_BELOW.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_copy_ends_total How copies ended (mismatch = the model wrote something else; end_of_context; short_emit = the request ended; unused = a round of another kind ran in between).\n# TYPE velogb10_copy_span_copy_ends_total counter\n");
    o.push_str(&format!("velogb10_copy_span_copy_ends_total{{reason=\"mismatch\"}} {}\n", CS_END_MISMATCH.load(Relaxed)));
    o.push_str(&format!("velogb10_copy_span_copy_ends_total{{reason=\"end_of_context\"}} {}\n", CS_END_CTX.load(Relaxed)));
    o.push_str(&format!("velogb10_copy_span_copy_ends_total{{reason=\"short_emit\"}} {}\n", CS_END_SHORT.load(Relaxed)));
    o.push_str(&format!("velogb10_copy_span_copy_ends_total{{reason=\"unused\"}} {}\n", CS_END_UNUSED.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_yield_gate --copy-span-yield-gate as armed on this boot (1 = on; 0 = off or the arm is off).\n# TYPE velogb10_copy_span_yield_gate gauge\n");
    o.push_str(&format!("velogb10_copy_span_yield_gate {}\n", CS_YG_ON.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_yield_gate_denied_total Copy starts (a context match >= min_n) the yield gate held back because copies on that request were not paying against its neural rounds; the neural drafter served those rounds.\n# TYPE velogb10_copy_span_yield_gate_denied_total counter\n");
    o.push_str(&format!("velogb10_copy_span_yield_gate_denied_total {}\n", CS_YG_DENIED.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_yield_gate_probes_total Copy starts the yield gate would have held back but let through as a probe (every 16th in a row).\n# TYPE velogb10_copy_span_yield_gate_probes_total counter\n");
    o.push_str(&format!("velogb10_copy_span_yield_gate_probes_total {}\n", CS_YG_PROBES.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_start_w Resolved --copy-span-start-w (the start ramp: a copy's first round proposes at most this many tokens, doubling per continued round; 0 when the arm is off).\n# TYPE velogb10_copy_span_start_w gauge\n");
    o.push_str(&format!("velogb10_copy_span_start_w {}\n", CS_START_W.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_narrow_proposals_total Copy proposals the start ramp held below the round's width cap, by source (search = a copy's first round; cursor = a later round still ramping up).\n# TYPE velogb10_copy_span_narrow_proposals_total counter\n");
    o.push_str(&format!("velogb10_copy_span_narrow_proposals_total{{source=\"search\"}} {}\n", CS_NARROW_SEARCH.load(Relaxed)));
    o.push_str(&format!("velogb10_copy_span_narrow_proposals_total{{source=\"cursor\"}} {}\n", CS_NARROW_CURSOR.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_trunk 1 = the copy arm on this boot is the trunk server's (NVFP4/bf16 packs: DFlash2 chain or MTP head rounds), 0 = the EXL3 server's (or not resolved).\n# TYPE velogb10_copy_span_trunk gauge\n");
    o.push_str(&format!("velogb10_copy_span_trunk {}\n", CS_TRUNK.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_skipped_rounds_total Trunk server with the copy arm on: decode steps that could not take a copy round, by reason (concurrent = 2+ requests batched; schema = JSON-schema request; source = a drafter without copy wiring; not_primed = the lane's drafter state is not live; spec_off = speculation off for the step; tp_disagree = TP ranks proposed different copy rounds, all ran the drafter round). The step ran as without the flag.\n# TYPE velogb10_copy_span_skipped_rounds_total counter\n");
    o.push_str("# HELP velogb10_copy_span_bare_rounds_total Trunk server: copy rounds served to a lone request whose drafter state is not live (it decoded batched or plain steps before), replacing plain decode steps. Also in velogb10_copy_span_rounds_total.\n# TYPE velogb10_copy_span_bare_rounds_total counter\n");
    o.push_str(&format!("velogb10_copy_span_bare_rounds_total {}\n", CS_BARE.load(Relaxed)));
    for (k, r) in CS_SKIP_NAMES.iter().enumerate() {
        o.push_str(&format!("velogb10_copy_span_skipped_rounds_total{{reason=\"{r}\"}} {}\n", CS_SKIP[k].load(Relaxed)));
    }
    // COPYPF1: the prefill-width copy verify's series (velogb10_copy_span_prefill_*)
    crate::copy_prefill::metrics_append(o);
    // COPYLANES: copy work in busy steps (velogb10_copy_span_lanes_*)
    crate::copy_lanes::metrics_append(o);
    crate::exl3_forward::dres::metrics_append(o);
    crate::family_defaults::metrics_append(o);
    crate::exl3_forward::dres::mid_metrics_append(o);
    crate::lane_wide::metrics_append(o);
    // COPYREC1: the cross-request copy record's series (velogb10_copy_span_record_*)
    crate::copy_record::metrics_append(o);
}

impl Exl3Scheduler {
    /// COPYLANES: the batch's delivered rate (tokens per ms over all busy lanes) the copy work of a busy step is priced
    /// against (copy_lanes::batch_rate: the largest of the measured arm rates fresh at this busy count, the lanes' own
    /// speculative-round estimate and the shared-step prior). Every input is replicated on every TP rank (agreed round
    /// ms, lockstep-adopted rates); the lanes' wall-clock latency (ema_wall_ms, local) is NOT read.
    fn cl_batch_rate(&self, slots: &[usize]) -> f64 {
        let n = slots.len();
        let mut measured: Vec<f64> = Vec::new();
        for arm in [ARM_SERIAL, ARM_SHARED] {
            if self.arm_fresh(arm, n) {
                measured.push(if spec_meter_v2() { self.spec_rate_n[n.min(SPEC_RATE_NMAX)][arm].0 } else { self.spec_arm_rate[arm] });
            }
        }
        // the lanes' speculative rounds: sum tokens / sum ms (a packed lane's ms is its share of the verify)
        let (mut tok, mut ms, mut known) = (0.0f64, 0.0f64, true);
        for &s in slots {
            if let Some(l) = self.lanes[s].as_ref().filter(|l| l.mtp.is_some()) {
                if l.ema_tok > 0.0 && l.ema_ms > 0.0 { tok += l.ema_tok; ms += l.ema_ms; } else { known = false; }
            }
        }
        let lanes_est = (known && ms > 0.0).then(|| tok / ms);
        let prior = n as f64 / (self.shared_scale * shared_step_prior_ms(n)).max(1e-9);
        crate::copy_lanes::batch_rate(&measured, lanes_est, prior)
    }

    /// COPYLANES: the rest of a step in which some lanes ran copy work: the arm the scheduler is riding (`shared_last`:
    /// the shared plain step, else packed pairs / serial rounds), with NO arm decision, probe window or rate sample —
    /// a step with a lane missing is not a sample of either arm at this busy count, and must not move the hysteresis.
    pub(super) fn cl_rest(&mut self, rest: Vec<usize>) -> Result<()> {
        // a fixed policy (--spec-lanes-max N / never) keeps no arm state: its own rule decides as without the flag
        if self.spec_policy != SpecLanes::Auto { return self.step_rounds_rest(rest, true); }
        if self.shared_last {
            crate::metrics::lane_pack_round(2); // arm = shared_plain
            return self.step_plain(&rest, true);
        }
        if let Some(plan) = self.pack_plan(&rest) {
            return self.step_packed(&rest, plan);
        }
        let mut plain: Vec<usize> = Vec::new();
        for &s in &rest {
            if self.lanes[s].as_ref().map_or(false, |l| l.mtp.is_some()) { self.step_mtp(s, false)?; } else { plain.push(s); }
        }
        if !plain.is_empty() { self.step_plain(&plain, false)?; }
        Ok(())
    }

    /// `cpv_pick`'s decision without counting anything: (decision, rung, true = a stretch the lone rule took that the
    /// busy-step batch price held back). COPYLANES' `cl_step` asks it to price a running copy by its stretch.
    #[allow(clippy::too_many_arguments)]
    fn cpv_eval(&self, s: usize, solo: bool, plain_step: bool, p: usize, remaining: usize,
                think_room: Option<usize>, busy: Option<f64>) -> Option<(crate::copy_prefill::Decision, usize, bool)> {
        use crate::copy_prefill as cp;
        let lane = self.lanes[s].as_ref()?;
        let cs = lane.copy.as_ref();
        let cursor = cs.and_then(|c| c.cursor());
        let cursor_live = cursor.is_some() && lane.copy_new.is_empty();
        let mut rung = if cursor.is_some() && cursor == Some(lane.cpv_next.1) { lane.cpv_next.0 } else { 0 };
        // the schedule's top rung: the prefill chunk capped by --copy-span-prefill-max-rows (the longest stream pause)
        let served = cp::schedule_top(self.chunk.max(1), cp::max_rows());
        let rows = self.psc.as_ref().map_or(0, |x| x.c);
        // the window / budget / thinking-budget clamps of a round; NOT the sparse-switch clamp: the prefill
        // assigns each row its own attention regime (rows past the switch take the selection path)
        let depth = self.round_depth_for(served, p, remaining, think_room).0;
        let left = cs.map_or(0, |c| c.cursor_left());
        let run = cs.map_or(0, |c| c.cursor_run());
        // ECHOPRICE (--copy-span-prefill-price, default on): start at the rung that covers the expected copy, take the
        // stretch only when its expected tokens beat the copy rounds' throughput over its measured cost
        let mut pays = true;
        if cp::priced() && cursor_live {
            let want = cp::expected_want(run, cs.map_or(0.0, |c| c.copy_len_hist()));
            (rung, pays) = cp::priced_pick(&self.cpv_price, rung, want, |g| cp::offered_len(g, served, rows, left, depth));
        }
        let n = cp::offered_len(rung, served, rows, left, depth);
        // COPYLANES: in a busy step the stretch must also pay for the batch (its expected emit over its rung's measured
        // cost against the batch's rate); held back, the copy round serves the lane. Prefill jobs in flight keep the
        // prefill scratch (a busy step never takes a stretch then).
        let mut held = false;
        if let Some(r) = busy {
            if pays && cursor_live && n >= cp::CPV_MIN_N {
                let e = (cp::expected_want(run, cs.map_or(0.0, |c| c.copy_len_hist())).min(n) + 1) as f64;
                let busy_pf = !self.pf_jobs.is_empty() || !self.pf_done.is_empty();
                if busy_pf || (self.cl_priced && !crate::copy_lanes::stretch_pays(e, self.cpv_price.stretch_ms(rung), r)) {
                    held = true;
                    pays = false;
                }
            }
        }
        let f = cp::Facts {
            on: self.cpv_on, solo, plain_step, cursor_live, in_prefill: self.in_prefill,
            penalized: lane.pen.is_some(), logprobs: lane.logprobs.is_some(), image: lane.img,
            tap_ok: lane.mtp.map_or(false, |t| t != usize::MAX), scratch_ok: self.psc.is_some(), n,
            run, min_run: cp::min_run(), pays,
        };
        Some((cp::decide(&f), rung, held))
    }

    /// COPYLANES: one busy step's copy work. Per busy lane (slot order): bring its proposer up to date
    /// (`commit_follow`), ask what it would propose at the round's copy depth (`peek`), price it against the batch
    /// (`copy_lanes::decide`); every lane that takes it then runs the lone path's copy work (`step_mtp_ex`: a copy
    /// round, or a stretch when `--copy-span-prefill` picks one and it pays for the batch). Returns the lanes served.
    /// Every input is replicated state, so every TP rank takes the same lanes in the same order.
    pub(super) fn cl_step(&mut self, slots: &[usize]) -> Result<Vec<usize>> {
        use crate::copy_lanes as cl;
        let Some(arm) = self.copy_arm else { return Ok(Vec::new()) };
        let batch = self.cl_batch_rate(slots);
        cl::publish(batch, &self.cl_price);
        let wide_ok = copy_span_wide(arm.max_w) && self.sc.vrows >= crate::exl3_forward::COPY_VROWS;
        let depth_cap = if wide_ok { crate::exl3_forward::COPY_MAX_K } else { self.mtp_k };
        let mut take: Vec<(usize, bool)> = Vec::new();
        for &s in slots {
            let Some(l) = self.lanes[s].as_ref() else { continue };
            let (p, remaining, think_room, spec_on) = (l.pos, l.max_new.saturating_sub(l.generated), l.think_room(),
                                                       l.mtp.is_some() && l.copy.is_some());
            let depth = if spec_on { self.round_depth_for(depth_cap, p, remaining, think_room).1 } else { 0 };
            let (peek, lane_rate, lane_n) = {
                let lane = self.lanes[s].as_mut().unwrap();
                let (lane_rate, lane_n) = (lane.cl_work.rate(), lane.cl_work.n);
                match lane.copy.as_mut() {
                    Some(cs) if spec_on => {
                        let th = std::time::Instant::now();
                        if !lane.copy_new.is_empty() {
                            let before = *cs.counters();
                            cs.commit_follow(&lane.copy_new);
                            lane.copy_new.clear();
                            copy_span_count_delta(&before, cs.counters());
                        }
                        let pk = if depth > 0 { cs.peek(copy_width(depth, arm.max_w)) } else { None };
                        CS_HOST_US.fetch_add(th.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
                        (pk, lane_rate, lane_n)
                    }
                    _ => (None, lane_rate, lane_n),
                }
            };
            // a running copy whose next stretch pays for the batch (`--copy-span-prefill`) is priced by the stretch: copy
            // rounds alone rarely beat a busy batch at TP=1 (a 16-row copy round ~85 ms), the stretch they lead to does
            let stretch = self.cpv_on && peek.map_or(false, |x| x.1)
                && matches!(self.cpv_eval(s, true, false, p, remaining, think_room, Some(batch)),
                            Some((crate::copy_prefill::Decision::Take(_), _, _)));
            let f = cl::Facts { spec_on, depth, peek, wide: peek.map_or(false, |x| x.0 > crate::exl3_forward::MTP_MAX_K),
                                lane_rate, lane_n, batch, priced: self.cl_priced, stretch };
            match cl::decide(&self.cl_price, &f) {
                cl::Verdict::Take { probe } => take.push((s, probe)),
                cl::Verdict::Skip(r) => cl::count_skip(r),
            }
        }
        if take.is_empty() { return Ok(Vec::new()); }
        cl::count_step();
        let mut served = Vec::with_capacity(take.len());
        for (s, probe) in take {
            if self.lanes[s].is_none() { continue; }
            let (r0, p0, k0) = (CS_ROUNDS.load(std::sync::atomic::Ordering::Relaxed), crate::copy_prefill::steps_total(),
                                crate::lane_wide::tokens_emitted());
            self.step_mtp_ex(s, false, Some(batch))?;
            let kind = if crate::copy_prefill::steps_total() != p0 { 1 }
                       else if CS_ROUNDS.load(std::sync::atomic::Ordering::Relaxed) != r0 { 0 } else { 2 };
            cl::count_work(kind, probe, crate::lane_wide::tokens_emitted().saturating_sub(k0));
            served.push(s);
        }
        Ok(served)
    }

    /// COPYPF1: should this lone lane's step be a prefill-width copy verify? Some((copied tokens, schedule rung))
    /// = yes; None = the normal round runs (an established copy that cannot take it is counted by reason).
    /// The schedule rung is the lane's only while its copy cursor is where the previous stretch left it (any
    /// other round in between, or a new copy, starts again at the first rung).
    pub(super) fn cpv_pick(&self, s: usize, solo: bool, plain_step: bool, p: usize, remaining: usize,
                think_room: Option<usize>, busy: Option<f64>) -> Option<(usize, usize)> {
        use crate::copy_prefill as cp;
        let (d, rung, held) = self.cpv_eval(s, solo, plain_step, p, remaining, think_room, busy)?;
        if held { crate::copy_lanes::count_stretch_priced(); }
        match d {
            cp::Decision::Take(n) => Some((n, rung)),
            cp::Decision::Skip(r) => { cp::count_skip(r); None }
            cp::Decision::NotApplicable => None,
        }
    }

    /// COPYPF1: one prefill-width copy verify for the lone lane in slot `s` (b unconsumed at p, its tap in row
    /// `tap_row`): the stretch (FwdModel::copy_prefill_verify: one chunked prefill of [b, c_1..c_n], every row
    /// judged by the lane's sampler rows ctr..ctr+n, roll back past the first disagreement), then the copy round's
    /// own sequence around it: the round's bookkeeping, the emit through round_emit (EOS, max_tokens, the loop
    /// detector, the thinking-budget filter and a gone client stop it mid-list, as for every round), and the
    /// proposer commit with exactly the emitted tokens. The neural drafter's statistics are not touched.
    pub(super) fn step_cpv(&mut self, s: usize, n: usize, rung: usize, b: i32, p: usize, tap_row: usize) -> Result<()> {
        let t0 = std::time::Instant::now();
        let (prop, rows) = {
            let lane = self.lanes[s].as_mut().unwrap();
            let cs = lane.copy.as_mut().context("copy-span-prefill: the lane has no copy proposer")?;
            let th = std::time::Instant::now();
            let before = *cs.counters();
            let prop = cs.propose_span(n).context("copy-span-prefill: no live copy cursor")?;
            copy_span_count_delta(&before, cs.counters());
            CS_HOST_US.fetch_add(th.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            let rows = lane.samp_rows(prop.tokens.len() + 1);
            (prop, rows)
        };
        let offered: Vec<i32> = prop.tokens.iter().map(|&x| x as i32).collect();
        let nn = offered.len();
        // COPYPF1-TP: lockstep point 1, BEFORE any launch of the stretch. Every rank computed this proposal from
        // replicated state (the copy round's argument); agree the stretch (offered count, schedule rung, seam token,
        // position, tap row, every offered id) before the chunk, whose row count selects the prefill's collective
        // sequence. tp_round's shape (fence A, exchange, fence B) because the previous lockstep op may be the
        // step-go's agree. The token's accept byte / width bits cannot name a 511-token stretch, so they carry 0 and
        // the stretch rides the words hash in full (a divergence fails the hash, never aliases). TP=1: no call.
        if self.tp.is_some() {
            let mut w: Vec<u32> = vec![0xC9F0_0001, nn as u32, rung as u32, b as u32, p as u32, tap_row as u32];
            w.extend(offered.iter().map(|&x| x as u32));
            self.tp_round("copy-prefill-pre", 0, 0, &w, 0.0, 0.0, false, false)?;
            crate::copy_prefill::count_tp_lockstep();
        }
        // an unpenalized lane: nothing to stage, and no mask (schema lanes never speculate)
        self.model.set_penalties(&mut self.sc, &[None])?;
        if self.cpv.is_none() { self.cpv = Some(crate::exl3_forward::CpvBufs::new()); }
        let psc = self.psc.as_mut().context("copy-span-prefill: no prefill scratch")?;
        // COPYPF1-TP: lockstep point 2, between the judge and the re-run: the accept count and every judged choice
        // compared on all ranks (the judge's TP tail already gives every rank the same choices; this catches any
        // divergence before a rank runs a re-run chunk whose width depends on it). Fence A first: the chunk and the
        // judge's collectives just ran. (A closure over the TP state only: the scratches are borrowed by the step.)
        let (tp, model) = (&mut self.tp, &self.model);
        let mut agree = |a: usize, ch: &[i32]| -> Result<()> {
            let Some(t) = tp.as_mut() else { return Ok(()) };
            let mut w: Vec<u32> = vec![0xC9F0_0002, nn as u32, a as u32, ch.len() as u32];
            w.extend(ch.iter().map(|&x| x as u32));
            Self::tp_round_with(t, model, None, "copy-prefill-judge", a & 0xFF, 0, &w, 0.0, 0.0, false)?;
            crate::copy_prefill::count_tp_lockstep();
            Ok(())
        };
        let out = model.copy_prefill_verify(&mut self.sc, psc, self.cpv.as_mut().unwrap(), s, b, p, tap_row,
                                            &offered, &rows, &mut agree)?;
        let ms_local = t0.elapsed().as_secs_f64() * 1e3;
        // COPYPF1-TP: lockstep point 3, after the re-run (fence A: its collectives are done on every rank before the
        // next step's step-go exchange reuses ring memory): the emitted tokens and the offered ids, and every rank
        // adopts the HEAD's wall time for the lane's statistics. The accept byte carries a mod 256; the hash has a.
        let ms = if self.tp.is_some() {
            let mut w: Vec<u32> = vec![0xC9F0_0003, nn as u32, out.a as u32, out.rerun_rows as u32];
            w.extend(out.emitted.iter().map(|&x| x as u32));
            w.extend(offered.iter().map(|&x| x as u32));
            let r = self.tp_round("copy-prefill", out.a & 0xFF, 0, &w, ms_local, 0.0, false, false)?;
            crate::copy_prefill::count_tp_lockstep();
            r.0
        } else { ms_local };
        let ro = crate::exl3_forward::MtpRoundOut {
            a: out.a,
            emitted: out.emitted.clone(),
            pos_next: p + out.a + 1,
            last_tok: *out.emitted.last().context("copy-span-prefill: nothing emitted")?,
            tap_row: 0, // copy_prefill_verify wrote the new bonus's tap into row 0 of the lane's window
            drafts: offered,
            diag_ms: 0.0,
        };
        if round_trace_on() {
            let r = self.tp.as_ref().map_or(0, |t| t.rank);
            eprintln!("[round-trace] rank={r} s={s} p={p} copy-prefill n={nn} a={} rerun={} judged={} emitted={:?}",
                      out.a, out.rerun_rows, out.judged, ro.emitted);
        }
        {
            let lane = self.lanes[s].as_mut().unwrap();
            lane.last_wall = Some(std::time::Instant::now());
            lane.arms.push('S', ro.emitted.len());
            // every sampler row of the stretch is consumed (rows past a were drawn and discarded: never reused)
            lane.ctr = lane.ctr.wrapping_add((nn + 1) as u32);
            lane.gate_first = false;
            lane.st_rounds += 1;
            lane.st_drafted += nn;
            lane.st_accepted += out.a;
            lane.st_ms += ms;
            lane.st_cpv_steps += 1;
            lane.st_cpv_offered += nn;
            lane.st_cpv_accepted += out.a;
            lane.st_cpv_emitted += ro.emitted.len();
            lane.st_cpv_rerun += out.rerun_rows;
        }
        crate::copy_prefill::count_step(nn, out.a, ro.emitted.len(), out.rerun_rows, out.judged, ms);
        self.cpv_price.note_stretch(rung, ms); // ECHOPRICE (ms = the head's under TP)
        if self.cl_on {
            // COPYLANES: a stretch is copy work (the server's and the lane's copy-work rates)
            self.cl_price.work.note(ro.emitted.len(), ms);
            if let Some(lane) = self.lanes[s].as_mut() { lane.cl_work.note(ro.emitted.len(), ms); }
        }
        self.round_emit_rows(s, &ro, None, false)?;
        // the proposer commit, exactly as after a copy round (a finished lane is gone: nothing to commit)
        if let Some(lane) = self.lanes[s].as_mut() {
            if let Some(cs) = lane.copy.as_mut() {
                let th = std::time::Instant::now();
                let before = *cs.counters();
                cs.commit(Some(&prop), out.a, &lane.copy_new);
                copy_span_count_delta(&before, cs.counters());
                lane.copy_new.clear();
                CS_HOST_US.fetch_add(th.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
                // the schedule: one rung up after a fully accepted stretch whose copy continues; else the first
                lane.cpv_next = match cs.cursor() {
                    Some(c) if out.a == nn => (crate::copy_prefill::next_rung(rung, true), c),
                    _ => (0, usize::MAX),
                };
            }
        }
        Ok(())
    }

    /// WIDEVERIFY1: commit the lane's emitted-but-uncommitted tokens to its copy-span proposer (as
    /// rounds without a proposal: a live copy cursor ends `unused`), then ask for a proposal of up to
    /// `copy_width(k_eff, max_w)` tokens. None = no proposal (or the arm is off for this lane).
    /// (An associated fn over the lane, not `&mut self`: step_mtp holds a borrow of the draft head.)
    pub(super) fn copy_propose(arm: Option<CopyArm>, lane: Option<&mut Lane>, k_eff: usize,
                    rec: Option<&mut crate::copy_record::CopyRecord>) -> Option<crate::copy_span::Proposal> {
        let max_w = arm?.max_w;
        let lane = lane?;
        Self::copy_propose_parts(max_w, &mut lane.copy, &mut lane.copy_new, k_eff, rec)
    }

    /// WIDEVERIFY1 yield gate: one neural speculative round of the lane emitted `emitted` tokens (no-op
    /// when the lane has no proposer). Called from exactly one place in step_mtp.
    pub(super) fn copy_note_neural(copy: &mut Option<Box<crate::copy_span::CopySpan>>, emitted: usize) {
        if let Some(cs) = copy.as_mut() { cs.note_neural_round(emitted); }
    }

    /// The body of `copy_propose` over the two lane fields it reads (WIDEVERIFY1 TP leg: split out so the
    /// unit suite drives the exact code every TP rank runs, without a whole `Lane`).
    /// COPYREC1: `rec` = the server's record (None = off: exactly the context-only proposer); asked only when the
    /// context has no proposal (`CopySpan::propose_with`).
    pub(super) fn copy_propose_parts(max_w: usize, copy: &mut Option<Box<crate::copy_span::CopySpan>>, copy_new: &mut Vec<u32>,
                          k_eff: usize, rec: Option<&mut crate::copy_record::CopyRecord>) -> Option<crate::copy_span::Proposal> {
        let cs = copy.as_mut()?;
        let th = std::time::Instant::now();
        let before = *cs.counters();
        if !copy_new.is_empty() {
            cs.commit(None, 0, copy_new);
            copy_new.clear();
        }
        let prop = match rec {
            Some(r) => {
                let p = cs.propose_with(copy_width(k_eff, max_w), Some(&mut *r));
                crate::copy_record::publish(r);
                p
            }
            None => cs.propose(copy_width(k_eff, max_w)),
        };
        copy_span_count_delta(&before, cs.counters());
        CS_HOST_US.fetch_add(th.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
        prop
    }
}
