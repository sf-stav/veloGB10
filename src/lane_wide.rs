//! LANEWIDE (2026-11-12): `--lane-wide-verify off|on|always`, the HOST half (no GPU; unit-tested here).
//!
//! With several busy speculating lanes, the wide arm verifies every eligible lane's speculative rows (its bonus token
//! and its MTP drafts) in ONE prefill-class pass (`FwdModel::wide_verify`, exl3_forward/lwv.rs) instead of one
//! decode-class verify per lane (or per packed pair) or the shared plain step. Each lane accepts its own agreeing
//! prefix; the per-segment commit replays exactly the accepted rows (the replay rollback).
//!
//! `on`: the load-adaptive policy picks, per scheduler step with at least two eligible lanes, between the wide arm
//! and today's arms (serial / packed speculation or the shared plain step, which keep their own choice between them)
//! by MEASURED aggregate tokens per second at that busy-lane count: short probe windows keep both rates fresh, and the
//! wide arm runs only while its measured rate beats today's by the hysteresis margin. `always`: every step with two or
//! more eligible lanes takes the wide arm (diagnostic / measurement). A lone request never takes it.
//!
//! The prefill-class kernels are not bit-identical to the decode-class verify, so output can differ on near-ties from
//! the same request served without the flag: opt-in, default off.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode { Off, On, Always }

impl Mode {
    pub fn parse(v: &str) -> Mode {
        match v.trim() { "on" | "1" | "auto" => Mode::On, "always" => Mode::Always, _ => Mode::Off }
    }
    pub fn from_opts() -> Mode { Mode::parse(&crate::opts::choice(crate::opt!("lane-wide-verify"))) }
}

/// How the flag resolved on this boot (`velogb10_lane_wide_state`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State { NotAsked = 0, On = 1, Always = 2, OffTp = 3, OffNoMtp = 4, OffPack = 5, OffDf2 = 6 }

impl State {
    pub fn reason(self) -> &'static str {
        match self { State::OffTp => "tp", State::OffNoMtp => "no_mtp", State::OffPack => "unsupported_pack", State::OffDf2 => "df2", _ => "" }
    }
    pub fn live(self) -> bool { matches!(self, State::On | State::Always) }
}

/// Pure boot resolution. `unsupported` = `FwdModel::wide_unsupported()` (None = the model can run it). Never a refusal.
pub fn resolve(mode: Mode, world: usize, unsupported: Option<&str>, df2: bool) -> State {
    if mode == Mode::Off { return State::NotAsked; }
    if world > 2 { return State::OffTp; } // LANEWIDE-TP: TP=1 and TP=2
    if df2 { return State::OffDf2; }
    match unsupported {
        Some("no_mtp") => State::OffNoMtp,
        Some("tp") => State::OffTp,
        Some(_) => State::OffPack,
        None => if mode == Mode::Always { State::Always } else { State::On },
    }
}

pub fn boot_line(st: State, rows_max: usize) -> Option<String> {
    match st {
        State::NotAsked => None,
        State::On | State::Always => Some(format!("[exl3-serve] lane-wide-verify: {} (--lane-wide-verify {}): with two or more busy speculating \
requests, ONE prefill-class pass verifies every eligible request's bonus token and drafts (at most {rows_max} rows, --lane-wide-rows-max; \
drafts of all lanes run as one batched head pass per depth), each request keeps its own agreeing prefix and only those rows are \
committed. {}OUTPUT MAY DIFFER ON NEAR-TIES from the same request served without the flag (the prefill-class kernels are not \
bit-identical to the decode-class verify). Penalized, logprob and image requests and requests past the sparse-attention switch keep \
today's rounds (counted). TP=1 and TP=2 (under TP every rank runs the same rounds; the policy reads the head's round times). Counters velogb10_lane_wide_*",
            if st == State::Always { "ALWAYS" } else { "ON" }, if st == State::Always { "always" } else { "on" },
            if st == State::Always { "Every step with two or more eligible requests takes it (diagnostic). " }
            else { "With three or more eligible requests the load-adaptive policy takes it only while its MEASURED aggregate tokens/s beats today's arms at the same number of busy requests. " })),
        _ => Some(format!("[exl3-serve] lane-wide-verify: asked (--lane-wide-verify), runs OFF on this boot ({}); today's rounds are unchanged \
(velogb10_lane_wide_state {})", match st {
            State::OffTp => "TP=1 and TP=2 only in this version (TP=4 is not wired)",
            State::OffNoMtp => "no MTP draft head",
            State::OffDf2 => "the DFlash2 chain is armed",
            _ => "this pack's layer kinds are not wired for it",
        }, st as u8)),
    }
}

/// Why a lane stays out of a wide round (or why a step chosen for the wide arm ran today's path).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Skip {
    Penalty, Logprobs, Image, Sparse, Control, NoMtp, Depth0, Prefill, TooFew, Rows, CopyStretch,
}

impl Skip {
    pub const ALL: [Skip; 11] = [Skip::Penalty, Skip::Logprobs, Skip::Image, Skip::Sparse, Skip::Control, Skip::NoMtp, Skip::Depth0,
                                 Skip::Prefill, Skip::TooFew, Skip::Rows, Skip::CopyStretch];
    pub fn label(self) -> &'static str {
        match self {
            Skip::Penalty => "penalty", Skip::Logprobs => "logprobs", Skip::Image => "image", Skip::Sparse => "sparse_regime",
            Skip::Control => "control", Skip::NoMtp => "no_mtp", Skip::Depth0 => "depth0", Skip::Prefill => "prefill_interleave",
            Skip::TooFew => "too_few_lanes", Skip::Rows => "rows_cap", Skip::CopyStretch => "copy_stretch",
        }
    }
    fn idx(self) -> usize { Skip::ALL.iter().position(|&s| s == self).unwrap() }
}

/// What the scheduler knows about one busy lane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LaneFacts {
    pub mtp_armed: bool,
    pub control: bool,
    pub penalized: bool,
    pub logprobs: bool,
    pub image: bool,
    pub sparse: bool,
    pub depth: usize,
    pub in_prefill: bool,
}

pub fn lane_skip(f: &LaneFacts) -> Option<Skip> {
    if !f.mtp_armed { Some(Skip::NoMtp) }
    else if f.penalized { Some(Skip::Penalty) }
    else if f.logprobs { Some(Skip::Logprobs) }
    else if f.image { Some(Skip::Image) }
    else if f.in_prefill { Some(Skip::Prefill) }
    else if f.control { Some(Skip::Control) }
    else if f.sparse { Some(Skip::Sparse) }
    else if f.depth == 0 { Some(Skip::Depth0) }
    else { None }
}

/// Batched drafts' depth for a lane whose speculative rounds deliver `ema_tok` tokens per round (0 = no estimate
/// yet): one and a half times its usual yield plus one, at least 2 (a fully accepted round raises the next depth).
pub fn auto_depth(ema_tok: f64) -> usize {
    if !(ema_tok > 0.0) { return 4; }
    ((ema_tok * 1.5).ceil() as usize + 1).max(2)
}

/// Per-lane draft depth so that n lanes of (depth + 1) rows fit `rows_max` rows.
pub fn depth_cap(n: usize, rows_max: usize) -> usize {
    if n == 0 { return 0; }
    (rows_max / n).saturating_sub(1)
}

// ---- the load-adaptive choice (measured aggregate tok/ms per busy-lane count) ----

/// today's arms, as today's own load-adaptive choice runs them (serial / packed speculation or the shared plain step)
pub const ARM_OTHER: usize = 0;
pub const ARM_WIDE: usize = 1;
/// the shared plain step for every busy lane, forced (today's choice may sit on serial speculation during a short
/// window; measuring the shared step explicitly keeps "the best of today's arms" honest)
pub const ARM_SHARED: usize = 2;
pub const NARMS: usize = 3;
/// `on`: the wide arm is considered from this many eligible lanes (`always`: from 2)
pub const WIDE_MIN_LANES: usize = 3;
/// the forced shared step is a policy arm from this many eligible lanes
pub const SHARED_MIN: usize = 4;
/// steps per probe window
pub const PROBE_WIN: u32 = 3;
/// a measured rate older than this many steps is stale
pub const STALE_STEPS: u64 = 1500;
/// an arm not being ridden is re-measured when its rate is older than this many steps (and probed at most once
/// per this many steps): content changes (thinking on / off, code / prose) move the arms' rates
pub const REFRESH: u64 = 160;
/// steps folded into one rate sample
pub const FOLD: u32 = 3;
/// hysteresis: the wide arm must beat the best of today's arms by this ratio to be taken (and keeps it until it
/// falls below the best of them)
pub const HYST: f64 = 1.04;

#[derive(Clone, Copy, Debug, Default)]
struct Rate { rate: f64, step: u64, tok: f64, ms: f64, n_acc: u32, probed: u64 }

#[derive(Clone, Debug)]
pub struct Policy {
    rates: Vec<[Rate; NARMS]>,
    win: Option<(usize, u32)>,
    pub last: usize,
}

impl Default for Policy {
    fn default() -> Self { Policy { rates: vec![[Rate::default(); NARMS]; 33], win: None, last: ARM_OTHER } }
}

/// One step's choice: the arm, and whether it is a measurement window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pick { pub arm: usize, pub probe: bool }

impl Pick {
    pub const WIDE: Pick = Pick { arm: ARM_WIDE, probe: false };
    pub fn wide(self) -> bool { self.arm == ARM_WIDE }
}

/// Busy-lane counts that share one measured rate (the arms' rates move slowly with the count; per-count rates
/// multiplied the measurement windows as requests finish one by one): 2, 3, 4-5, 6-8, 9-12, 13+.
pub fn bucket(n: usize) -> usize {
    match n { 0..=2 => 2, 3 => 3, 4..=5 => 4, 6..=8 => 6, 9..=12 => 9, _ => 13 }
}

impl Policy {
    fn rate(&self, n: usize, arm: usize, step: u64) -> Option<f64> {
        let r = &self.rates[n.min(32)][arm];
        if r.rate > 0.0 && step.saturating_sub(r.step) <= STALE_STEPS { Some(r.rate) } else { None }
    }

    /// `n` = eligible busy lanes this step (>= 2), `step` = the scheduler's step counter.
    pub fn pick(&mut self, n: usize, step: u64) -> Pick {
        if let Some((arm, rem)) = self.win {
            self.win = if rem <= 1 { None } else { Some((arm, rem - 1)) };
            return Pick { arm, probe: true };
        }
        let nn = bucket(n);
        // the forced shared step is an arm only from 4 busy lanes: below that today's own choice (packed pairs) is the
        // reference, and 3-step windows of the shared step at 2 lanes mis-measured it (W7 battery, 2 lanes thinking on)
        let arms: &[usize] = if n >= SHARED_MIN { &[ARM_WIDE, ARM_SHARED, ARM_OTHER] } else { &[ARM_WIDE, ARM_OTHER] };
        for &arm in arms {
            let (probed, at) = (self.rates[nn][arm].probed, self.rates[nn][arm].step);
            // never measured / stale, or (an arm not being ridden) older than REFRESH steps: the content may have changed
            let due = self.rate(nn, arm, step).is_none() || (arm != self.last && step.saturating_sub(at) > REFRESH);
            if due && (probed == 0 || step.saturating_sub(probed) >= REFRESH) {
                self.rates[nn][arm].probed = step.max(1);
                self.win = Some((arm, PROBE_WIN - 1));
                return Pick { arm, probe: true };
            }
        }
        let r = |a: usize| self.rate(nn, a, step).unwrap_or(0.0);
        let (w, o, sh) = (r(ARM_WIDE), r(ARM_OTHER), if n >= SHARED_MIN { r(ARM_SHARED) } else { 0.0 });
        let today = if sh > o { ARM_SHARED } else { ARM_OTHER };
        let best_today = o.max(sh);
        let wide = if self.last == ARM_WIDE { w > 0.0 && w >= best_today } else { w > best_today * HYST };
        self.last = if wide { ARM_WIDE } else { today };
        Pick { arm: self.last, probe: false }
    }

    /// One step's delivery (tokens emitted across lanes, wall ms) for the arm that ran it at `n` busy lanes.
    pub fn record(&mut self, n: usize, arm: usize, tok: f64, ms: f64, step: u64) {
        if n < 2 || ms <= 0.0 { return; }
        let r = &mut self.rates[bucket(n)][arm];
        r.tok += tok;
        r.ms += ms;
        r.n_acc += 1;
        if r.n_acc >= FOLD {
            let x = r.tok / r.ms;
            r.rate = if r.rate > 0.0 && step.saturating_sub(r.step) <= STALE_STEPS { 0.5 * r.rate + 0.5 * x } else { x };
            r.step = step;
            r.tok = 0.0;
            r.ms = 0.0;
            r.n_acc = 0;
            set_rate(arm, r.rate * 1e3);
        }
    }
}

// ---- counters (`velogb10_lane_wide_*`) ----

static ST: AtomicU64 = AtomicU64::new(0);
static ROUNDS: AtomicU64 = AtomicU64::new(0);
static LANE_ROUNDS: AtomicU64 = AtomicU64::new(0);
static ROWS: AtomicU64 = AtomicU64::new(0);
static TOKS: AtomicU64 = AtomicU64::new(0);
static ACCEPTED: AtomicU64 = AtomicU64::new(0);
static DRAFTED: AtomicU64 = AtomicU64::new(0);
static US: AtomicU64 = AtomicU64::new(0);
static US_DRAFT: AtomicU64 = AtomicU64::new(0);
static US_TRUNK: AtomicU64 = AtomicU64::new(0);
static US_JUDGE: AtomicU64 = AtomicU64::new(0);
static US_COMMIT: AtomicU64 = AtomicU64::new(0);
static US_FINISH: AtomicU64 = AtomicU64::new(0);
static SKIPS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
static PICKS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
static RATE: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static TOKEN_CTR: AtomicU64 = AtomicU64::new(0);
static FIRST: AtomicU64 = AtomicU64::new(0);

pub fn set_state(st: State) { ST.store(st as u64, Relaxed); }

static SERIAL_DRAFT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Boot: `--lane-wide-draft` (batched, the default, or serial = each lane's own lone draft chain, for A/B).
pub fn set_draft_mode(serial: bool) { SERIAL_DRAFT.store(serial, Relaxed); }
pub fn batched_draft() -> bool { !SERIAL_DRAFT.load(Relaxed) }
pub fn count_skip(s: Skip) { SKIPS[s.idx()].fetch_add(1, Relaxed); }
pub fn skips(s: Skip) -> u64 { SKIPS[s.idx()].load(Relaxed) }
pub fn count_pick(p: Pick) {
    PICKS[p.arm * 2 + p.probe as usize].fetch_add(1, Relaxed);
}
fn set_rate(arm: usize, tok_s: f64) { RATE[arm].store(tok_s.to_bits(), Relaxed); }
/// Every token a lane emits (any arm): the policy's delivery measure.
pub fn note_token() { TOKEN_CTR.fetch_add(1, Relaxed); }
pub fn tokens_emitted() -> u64 { TOKEN_CTR.load(Relaxed) }

#[allow(clippy::too_many_arguments)]
pub fn count_round(lanes: usize, rows: usize, toks: usize, accepted: usize, drafted: usize, ms: f64, ms_parts: [f64; 5]) {
    if FIRST.fetch_add(1, Relaxed) == 0 {
        crate::rprintln!("[exl3-serve] lane-wide-verify: first wide round served ({lanes} lanes, {rows} rows, {toks} tokens)");
    }
    ROUNDS.fetch_add(1, Relaxed);
    LANE_ROUNDS.fetch_add(lanes as u64, Relaxed);
    ROWS.fetch_add(rows as u64, Relaxed);
    TOKS.fetch_add(toks as u64, Relaxed);
    ACCEPTED.fetch_add(accepted as u64, Relaxed);
    DRAFTED.fetch_add(drafted as u64, Relaxed);
    let us = |x: f64| (x.max(0.0) * 1000.0) as u64;
    US.fetch_add(us(ms), Relaxed);
    for (a, x) in [&US_DRAFT, &US_TRUNK, &US_JUDGE, &US_COMMIT, &US_FINISH].iter().zip(ms_parts) { a.fetch_add(us(x), Relaxed); }
}

pub fn rounds() -> u64 { ROUNDS.load(Relaxed) }

pub fn metrics_append(o: &mut String) {
    let st = ST.load(Relaxed);
    let c = |o: &mut String, name: &str, help: &str, v: String| {
        o.push_str(&format!("# HELP velogb10_lane_wide_{name} {help}\n# TYPE velogb10_lane_wide_{name} {}\nvelogb10_lane_wide_{name} {v}\n",
                            if name.ends_with("_total") { "counter" } else { "gauge" }));
    };
    c(o, "state", "--lane-wide-verify on this boot: 0 = not asked, 1 = on (policy), 2 = always, 3 = asked but off (TP), 4 = off (no MTP head), 5 = off (pack not wired), 6 = off (DFlash2 chain).", st.to_string());
    c(o, "rounds_total", "Wide verify rounds (one prefill-class pass over several requests' speculative rows).", ROUNDS.load(Relaxed).to_string());
    c(o, "lane_rounds_total", "Request rounds served inside wide rounds.", LANE_ROUNDS.load(Relaxed).to_string());
    c(o, "rows_total", "Rows verified by wide rounds (bonus tokens + drafts).", ROWS.load(Relaxed).to_string());
    c(o, "tokens_total", "Tokens emitted by wide rounds.", TOKS.load(Relaxed).to_string());
    c(o, "accepted_total", "Drafts accepted in wide rounds.", ACCEPTED.load(Relaxed).to_string());
    c(o, "drafted_total", "Drafts judged in wide rounds.", DRAFTED.load(Relaxed).to_string());
    c(o, "seconds_total", "Wall time of wide rounds (drafts, pass, judge, commit, re-prime).", format!("{:.6}", US.load(Relaxed) as f64 / 1e6));
    o.push_str("# HELP velogb10_lane_wide_phase_seconds_total Wide round wall time by phase.\n# TYPE velogb10_lane_wide_phase_seconds_total counter\n");
    for (n, a) in [("draft", &US_DRAFT), ("trunk", &US_TRUNK), ("judge", &US_JUDGE), ("commit", &US_COMMIT), ("reprime", &US_FINISH)] {
        o.push_str(&format!("velogb10_lane_wide_phase_seconds_total{{phase=\"{n}\"}} {:.6}\n", a.load(Relaxed) as f64 / 1e6));
    }
    o.push_str("# HELP velogb10_lane_wide_fallback_total Requests kept out of a wide round (or steps that ran today's path though the wide arm was chosen), by reason.\n# TYPE velogb10_lane_wide_fallback_total counter\n");
    for s in Skip::ALL { o.push_str(&format!("velogb10_lane_wide_fallback_total{{reason=\"{}\"}} {}\n", s.label(), SKIPS[s.idx()].load(Relaxed))); }
    o.push_str("# HELP velogb10_lane_wide_policy_steps_total Scheduler steps with two or more eligible requests, by the policy's choice (other = today's own choice of arm, shared = the shared plain step forced, wide; probe_* = the measurement windows).\n# TYPE velogb10_lane_wide_policy_steps_total counter\n");
    for (a, n) in ["other", "wide", "shared"].iter().enumerate() {
        o.push_str(&format!("velogb10_lane_wide_policy_steps_total{{arm=\"{n}\"}} {}\n", PICKS[a * 2].load(Relaxed)));
        o.push_str(&format!("velogb10_lane_wide_policy_steps_total{{arm=\"probe_{n}\"}} {}\n", PICKS[a * 2 + 1].load(Relaxed)));
    }
    o.push_str("# HELP velogb10_lane_wide_measured_tok_s The policy's latest measured aggregate tokens/s per arm (other = today's arms).\n# TYPE velogb10_lane_wide_measured_tok_s gauge\n");
    for (i, n) in ["other", "wide", "shared"].iter().enumerate() {
        o.push_str(&format!("velogb10_lane_wide_measured_tok_s{{arm=\"{n}\"}} {:.1}\n", f64::from_bits(RATE[i].load(Relaxed))));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_registered_cli_only() {
        let o = crate::opt!("lane-wide-verify").def();
        assert_eq!((o.def, o.env), ("off", ""));
        assert!(matches!(o.scope, crate::opts::Scope::Spmd));
        let d = crate::opt!("lane-wide-draft").def();
        assert_eq!((d.def, d.env), ("batched", ""));
        let r = crate::opt!("lane-wide-rows-max").def();
        assert_eq!((r.def, r.env), ("128", ""));
        assert_eq!(Mode::parse("on"), Mode::On);
        assert_eq!(Mode::parse("always"), Mode::Always);
        assert_eq!(Mode::parse("off"), Mode::Off);
    }

    #[test]
    fn resolve_never_refuses() {
        assert_eq!(resolve(Mode::Off, 1, None, false), State::NotAsked);
        assert_eq!(resolve(Mode::On, 1, None, false), State::On);
        assert_eq!(resolve(Mode::Always, 1, None, false), State::Always);
        assert_eq!(resolve(Mode::On, 2, None, false), State::On); // LANEWIDE-TP
        assert_eq!(resolve(Mode::On, 4, None, false), State::OffTp);
        assert_eq!(resolve(Mode::On, 1, Some("no_mtp"), false), State::OffNoMtp);
        assert_eq!(resolve(Mode::On, 1, Some("unsupported_pack"), false), State::OffPack);
        assert_eq!(resolve(Mode::On, 1, None, true), State::OffDf2);
        assert!(boot_line(State::NotAsked, 128).is_none());
        assert!(boot_line(State::On, 128).unwrap().contains("MAY DIFFER ON NEAR-TIES"));
        assert!(boot_line(State::OffTp, 128).unwrap().contains("runs OFF"));
    }

    #[test]
    fn lane_skips_in_priority_order() {
        let ok = LaneFacts { mtp_armed: true, depth: 3, ..Default::default() };
        assert_eq!(lane_skip(&ok), None);
        assert_eq!(lane_skip(&LaneFacts { mtp_armed: false, ..ok }), Some(Skip::NoMtp));
        assert_eq!(lane_skip(&LaneFacts { penalized: true, logprobs: true, ..ok }), Some(Skip::Penalty));
        assert_eq!(lane_skip(&LaneFacts { logprobs: true, ..ok }), Some(Skip::Logprobs));
        assert_eq!(lane_skip(&LaneFacts { image: true, ..ok }), Some(Skip::Image));
        assert_eq!(lane_skip(&LaneFacts { sparse: true, ..ok }), Some(Skip::Sparse));
        assert_eq!(lane_skip(&LaneFacts { depth: 0, ..ok }), Some(Skip::Depth0));
        assert_eq!(depth_cap(8, 128), 15);
        assert_eq!(depth_cap(16, 128), 7);
        assert_eq!(depth_cap(16, 64), 3);
        assert_eq!(auto_depth(0.0), 4);
        assert_eq!(auto_depth(2.6), 5);
        assert_eq!(auto_depth(5.5), 10);
        assert_eq!(auto_depth(1.0), 3);
    }

    #[test]
    fn policy_probes_every_arm_then_follows_the_measured_rates() {
        // rates: (other, wide, shared) tokens per 10 ms step
        let run = |n: usize, rates: [f64; 3]| -> Policy {
            let mut p = Policy::default();
            let mut seen = [false; 3];
            for step in 1..=(3 * PROBE_WIN as u64) {
                let k = p.pick(n, step);
                assert!(k.probe, "the first windows measure");
                seen[k.arm] = true;
                p.record(n, k.arm, rates[k.arm], 10.0, step);
            }
            assert!(seen.iter().all(|&x| x), "every arm measured");
            p
        };
        let mut p = run(8, [10.0, 20.0, 14.0]);
        assert_eq!(p.pick(8, 40), Pick::WIDE);
        let mut q = run(8, [10.0, 13.0, 14.0]);
        assert_eq!(q.pick(8, 40), Pick { arm: ARM_SHARED, probe: false }, "wide loses to the shared step");
        assert_eq!((bucket(2), bucket(5), bucket(7), bucket(16)), (2, 4, 6, 13));
        // below SHARED_MIN only wide and today's arm are measured
        let mut t = Policy::default();
        for step in 1..=(2 * PROBE_WIN as u64) { let k = t.pick(2, step); assert!(k.arm != ARM_SHARED); t.record(2, k.arm, 10.0, 10.0, step); }
        let mut r = run(4, [15.0, 15.2, 9.0]);
        assert_eq!(r.pick(4, 40), Pick { arm: ARM_OTHER, probe: false }, "inside the hysteresis: today's arm stays");
        // a stale rate is re-probed
        assert!(r.pick(4, 40 + REFRESH + 1).probe, "an idle arm is re-measured");
    }

    #[test]
    fn metrics_render() {
        count_skip(Skip::Penalty);
        let mut o = String::new();
        metrics_append(&mut o);
        for s in ["velogb10_lane_wide_state ", "velogb10_lane_wide_rounds_total ", "velogb10_lane_wide_tokens_total ",
                  "velogb10_lane_wide_fallback_total{reason=\"penalty\"} ", "velogb10_lane_wide_policy_steps_total{arm=\"wide\"} ",
                  "velogb10_lane_wide_measured_tok_s{arm=\"wide\"} ", "velogb10_lane_wide_phase_seconds_total{phase=\"trunk\"} "] {
            assert!(o.contains(s), "missing {s}");
        }
    }
}

/// The scheduler's wide-arm state (one per scheduler).
pub struct WideSched {
    pub st: State,
    pub rows_max: usize,
    pub bufs: Option<crate::exl3_forward::WideBufs>,
    pub pol: Policy,
    /// a wide round failed: the arm stays off for the rest of the boot (one line, counted in state)
    pub disabled: bool,
}

impl Default for WideSched {
    fn default() -> Self { WideSched { st: State::NotAsked, rows_max: 128, bufs: None, pol: Policy::default(), disabled: false } }
}

impl WideSched {
    pub fn live(&self) -> bool { self.st.live() && !self.disabled }
}
