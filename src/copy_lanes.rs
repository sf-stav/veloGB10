//! COPYLANES (2026-11-13): copy-span work for EVERY busy lane, the HOST logic (no GPU; unit-tested here).
//!
//! `--copy-span-lanes on` (default OFF, meaningful only with `--copy-span on`). Without it a copy round (and a
//! `--copy-span-prefill` stretch) serves only a LONE request: `step_mtp` asks the proposer only when one lane is busy,
//! and with 2-4 busy lanes the scheduler runs its packed / serial / shared-plain arms, none of which ever asks. With it,
//! every scheduler step with two or more busy lanes first prices each lane's copy work against the batch:
//!
//!   * the lane's proposer is brought up to date (`CopySpan::commit_follow`: tokens written by packed or shared steps
//!     keep a live copy alive when they are exactly the source's next tokens) and asked what it WOULD propose
//!     (`CopySpan::peek`, no state change);
//!   * a lane whose copy work pays for the whole batch leaves the step's arms and runs the lone path's own copy work
//!     (`step_mtp`: a copy round, or a stretch when `--copy-span-prefill` picks one); the other lanes run the step's
//!     arms (packed / serial / shared plain) among themselves, unchanged;
//!   * a lane whose copy work does not pay stays in the step's arms, as without the flag. In a step where some lanes
//!     ran copy work, the others ride the arm the scheduler is on (no arm decision, probe or rate sample: a step with
//!     a lane missing is not a sample of either arm and must not move the arm choice).
//!
//! THE PRICE (all inputs replicated on every TP rank: TP-agreed round times, agreed accept counts, lockstep state).
//! R = the batch's delivered rate (tokens per ms over all busy lanes): the larger of the measured arm rates that are
//! fresh at this busy count, the lanes' own round estimates (sum tokens / sum ms of their speculative rounds; a packed
//! lane's ms is its share) and the shared-step prior. Taking the larger is the conservative side: copy work must beat
//! the best the batch is known to do. A lane that leaves the step for copy work costs the other lanes the work's time
//! and delivers its tokens; the step's aggregate rate rises iff the work's own rate beats R, so:
//!
//!   * a CURSOR proposal (a running copy) of n tokens: E = 1 + a * n tokens (a = the accept fraction of cursor copy
//!     rounds, measured) for C ms (the measured copy-round time at its width class); taken iff E / C >= MARGIN * R;
//!   * a SEARCH proposal (a copy would START): one start round rarely pays by itself (the start ramp proposes a few
//!     tokens); what pays is the copy it starts. So a start is priced by the lane's measured COPY-WORK rate (all its
//!     copy rounds and stretches: tokens over ms, decayed sums), or the server's while the lane has fewer than
//!     LANE_MIN_SAMPLES; taken iff that rate >= MARGIN * R. An agent turn that quotes a line and goes on (copies that
//!     land a few tokens) prices itself out; a file return keeps going;
//!   * a STRETCH (`--copy-span-prefill`): the lone priced rule picks it as before, and in a busy step it must also pay
//!     for the batch: E / P >= MARGIN * R (E = the stretch's expected emit, P = its rung's measured cost); a running
//!     copy whose next stretch pays this way is taken even when its copy round alone would not pay;
//!   * nothing measured yet: the lane takes the work as a probe (counted) so the costs get measured.
//!
//! Memory: none on the device (the copy round runs on the served scratch, a stretch on the prefill scratch, exactly as
//! for a lone request); a few f64 per lane on the host.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// A lane's copy work must beat the batch's rate by this factor (noise margin: the batch rate is an estimate).
pub const MARGIN: f64 = 1.15;
/// Exponential weight of a new copy-round cost / accept sample.
const W: f64 = 0.25;
/// Decay of the copy-work sums per new sample (lane and server).
const DECAY: f64 = 0.75;
/// A lane's own copy-work rate prices its starts once it has this many samples (else the server's).
pub const LANE_MIN_SAMPLES: u64 = 4;
/// The accept fraction assumed for cursor copy rounds before one is measured.
pub const PRIOR_ACC_CURSOR: f64 = 0.85;

/// How `--copy-span-lanes` resolved on this boot (the `velogb10_copy_span_lanes_state` gauge).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    NotAsked = 0,
    On = 1,
    /// asked without an armed copy arm (`--copy-span off`, or the copy arm runs off)
    OffNoCopySpan = 2,
    /// asked with `--lane-order fcfs` (one lane steps at a time: there is no busy step to share)
    OffFcfs = 3,
    /// asked with `--max-batch 1` (never two busy lanes)
    OffOneLane = 4,
    /// ULTRA2: asked on a server without the busy-step copy work (the NVFP4/bf16 trunk server, whose copy arm
    /// serves a lone request only; DeepSeek V4 / Naive bundles have no copy arm at all)
    OffServer = 5,
}

impl State {
    pub fn reason(self) -> &'static str {
        match self {
            State::NotAsked | State::On => "",
            State::OffNoCopySpan => "copy_span_off",
            State::OffFcfs => "lane_order_fcfs",
            State::OffOneLane => "max_batch_1",
            State::OffServer => "server",
        }
    }
}

/// Pure boot resolution. Never a refusal.
pub fn resolve(asked: bool, copy_armed: bool, fcfs: bool, width: usize) -> State {
    if !asked { State::NotAsked }
    else if !copy_armed { State::OffNoCopySpan }
    else if fcfs { State::OffFcfs }
    else if width < 2 { State::OffOneLane }
    else { State::On }
}

/// The one boot line (None when not asked).
pub fn boot_line(st: State, world: usize, priced: bool) -> Option<String> {
    match st {
        State::NotAsked => None,
        State::On => Some(format!("[exl3-serve] copy-span-lanes: ON (--copy-span-lanes on): with 2-4 requests busy, a request whose \
answer is copying its context runs its own copy rounds (and --copy-span-prefill stretches) in the step while the other requests \
run the packed / serial / shared step among themselves; the model verifies every copied token as for a lone request (greedy \
output unchanged per request). {}{}No device memory. Counters velogb10_copy_span_lanes_*",
            if priced { format!("Priced (--copy-span-lanes-price on): a request leaves the step for copy work only when that work \
delivers at least {MARGIN:.2}x the batch's measured tokens per second (a running copy by its round's expected tokens and measured \
cost, a copy start by that request's measured copy-work rate, a stretch by its expected tokens and measured cost); otherwise it \
stays in the step, as without the flag. ") }
            else { "Price OFF (--copy-span-lanes-price off, diagnostic): every request with a copy proposal runs it. ".to_string() },
            if world > 1 { format!("TP={world}: every rank prices the same lanes from replicated inputs; each copy round's copied ids are \
agreed before it runs, as for a lone request. ") } else { String::new() })),
        State::OffNoCopySpan => Some("[exl3-serve] copy-span-lanes: asked (--copy-span-lanes on), runs OFF: it serves copies made by \
--copy-span, which is not armed on this boot (velogb10_copy_span_lanes_state 2)".to_string()),
        State::OffFcfs => Some("[exl3-serve] copy-span-lanes: asked (--copy-span-lanes on), runs OFF: --lane-order fcfs steps one \
request at a time, so a lone request's copy rounds already serve it (velogb10_copy_span_lanes_state 3)".to_string()),
        State::OffOneLane => Some("[exl3-serve] copy-span-lanes: asked (--copy-span-lanes on), runs OFF: --max-batch 1 never has two \
busy requests; copy rounds serve the lone request as before (velogb10_copy_span_lanes_state 4)".to_string()),
        State::OffServer => Some(server_fallback_line()),
    }
}

/// ULTRA2: the one boot line of `--copy-span-lanes on` on a server without busy-step copy work.
pub fn server_fallback_line() -> String {
    "[serve] copy-span-lanes: asked (--copy-span-lanes on), runs OFF: copy work for busy requests is built into the EXL3 \
server only; this server's copy arm (--copy-span) serves a lone request, and 2+ requests at once keep the batched paths \
(velogb10_copy_span_lanes_state 5, velogb10_copy_span_lanes_fallback_total{reason=\"server\"})".to_string()
}

/// ULTRA2: resolve `--copy-span-lanes` on a non-EXL3 server (the trunk NVFP4/bf16 server, DeepSeek V4, Naive):
/// asked = one line + the fallback counter, never a refusal; not asked = nothing (the gauge stays 0).
pub fn resolve_other_server() -> State {
    if !crate::opts::on(crate::opt!("copy-span-lanes")) {
        return State::NotAsked;
    }
    let priced = !matches!(crate::opts::var(crate::opt!("copy-span-lanes-price")).as_deref(), Ok("0"));
    set_state(State::OffServer, priced);
    crate::rprintln!("{}", server_fallback_line());
    State::OffServer
}

/// Decayed (tokens, ms) sums of copy work: rate = tokens / ms.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Work {
    pub tok: f64,
    pub ms: f64,
    pub n: u64,
}

impl Work {
    pub fn note(&mut self, tok: usize, ms: f64) {
        if !(ms > 0.0) { return; }
        self.tok = DECAY * self.tok + tok as f64;
        self.ms = DECAY * self.ms + ms;
        self.n += 1;
    }
    pub fn rate(&self) -> Option<f64> {
        (self.n > 0 && self.ms > 0.0).then(|| self.tok / self.ms)
    }
}

/// The measured copy costs the price reads (one per scheduler; fed from TP-agreed round times and accept counts).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Price {
    /// copy-round ms by width class (0 = a chain-verify width, 1 = the wide copy verify), and samples
    c_ms: [f64; 2],
    c_n: [u64; 2],
    /// accept fraction of cursor copy rounds (accepted / offered)
    acc: f64,
    acc_n: u64,
    /// the server's copy-work sums (every copy round and stretch, any lane, lone or busy)
    pub work: Work,
}

impl Price {
    /// One copy round: `ms` (the head's under TP), `offered` copied tokens, `accepted` of them, `cursor` = it continued
    /// a running copy, `wide` = its width needed the wide copy verify.
    pub fn note_copy_round(&mut self, ms: f64, offered: usize, accepted: usize, cursor: bool, wide: bool) {
        if !(ms > 0.0) { return; }
        let i = wide as usize;
        self.c_ms[i] = if self.c_n[i] == 0 { ms } else { self.c_ms[i] + W * (ms - self.c_ms[i]) };
        self.c_n[i] += 1;
        if cursor && offered > 0 {
            let f = accepted.min(offered) as f64 / offered as f64;
            self.acc = if self.acc_n == 0 { f } else { self.acc + W * (f - self.acc) };
            self.acc_n += 1;
        }
    }
    /// The cost of a copy round of this width class: measured, else the other class's, else None.
    pub fn round_ms(&self, wide: bool) -> Option<f64> {
        let i = wide as usize;
        if self.c_n[i] > 0 { Some(self.c_ms[i]) } else if self.c_n[1 - i] > 0 { Some(self.c_ms[1 - i]) } else { None }
    }
    /// The accept fraction of cursor copy rounds (the prior before one is measured).
    pub fn accept(&self) -> f64 {
        if self.acc_n > 0 { self.acc } else { PRIOR_ACC_CURSOR }
    }
    pub fn round_ms_raw(&self, wide: bool) -> f64 { self.c_ms[wide as usize] }
}

/// The batch's delivered rate (tokens per ms): the largest of the known estimates (measured arm rates fresh at this
/// busy count, the lanes' own round estimate, the shared-step prior). Non-positive / non-finite inputs are ignored.
pub fn batch_rate(measured: &[f64], lanes_est: Option<f64>, shared_prior: f64) -> f64 {
    measured.iter().copied().chain(lanes_est).chain(std::iter::once(shared_prior))
        .filter(|x| x.is_finite() && *x > 0.0)
        .fold(0.0f64, f64::max)
}

/// Why a busy lane stayed in the step's arms (counted in `velogb10_copy_span_lanes_skip_total{reason}`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Skip {
    /// no copy proposal: no context match (or one below --copy-span-min-n, or held back by the yield gate)
    NoCopy,
    /// a proposal whose copy work does not beat the batch's rate
    PricedOut,
    /// the request does not speculate (speculation off for it, or a JSON-schema request)
    SpecOff,
    /// the round's depth is 0 (end of the KV window, the token budget, the thinking budget)
    Clamped,
}

impl Skip {
    pub const ALL: [Skip; 4] = [Skip::NoCopy, Skip::PricedOut, Skip::SpecOff, Skip::Clamped];
    pub fn label(self) -> &'static str {
        match self {
            Skip::NoCopy => "no_copy",
            Skip::PricedOut => "priced_out",
            Skip::SpecOff => "spec_off",
            Skip::Clamped => "clamped",
        }
    }
    fn idx(self) -> usize { Skip::ALL.iter().position(|&s| s == self).unwrap() }
}

/// What the step knows about one busy lane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Facts {
    /// the lane speculates (a draft tap is live)
    pub spec_on: bool,
    /// the round depth a copy round would get (0 = clamped)
    pub depth: usize,
    /// `CopySpan::peek` at that depth's width: (tokens, cursor)
    pub peek: Option<(usize, bool)>,
    /// the proposal needs the wide copy verify (more than the chain verify's drafts)
    pub wide: bool,
    /// the lane's own copy-work rate and samples (tokens per ms)
    pub lane_rate: Option<f64>,
    pub lane_n: u64,
    /// the batch's rate (tokens per ms)
    pub batch: f64,
    /// `--copy-span-lanes-price` (false = every proposal is taken)
    pub priced: bool,
    /// a running copy whose next `--copy-span-prefill` stretch pays for the batch (`stretch_pays`, from measured costs)
    pub stretch: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict {
    /// run the lone path's copy work for this lane in this step (`probe` = nothing measured to price it yet)
    Take { probe: bool },
    Skip(Skip),
}

/// The expected tokens of a cursor copy round of `n` copied tokens.
pub fn expected_round(price: &Price, n: usize) -> f64 {
    1.0 + price.accept() * n as f64
}

pub fn decide(price: &Price, f: &Facts) -> Verdict {
    if !f.spec_on { return Verdict::Skip(Skip::SpecOff); }
    if f.depth == 0 { return Verdict::Skip(Skip::Clamped); }
    let Some((n, cursor)) = f.peek else { return Verdict::Skip(Skip::NoCopy) };
    if n == 0 { return Verdict::Skip(Skip::NoCopy); }
    if !f.priced { return Verdict::Take { probe: false }; }
    let bar = MARGIN * f.batch;
    if cursor {
        if f.stretch { return Verdict::Take { probe: false }; }
        let Some(c) = price.round_ms(f.wide) else { return Verdict::Take { probe: true } };
        if expected_round(price, n) / c.max(1e-9) >= bar { Verdict::Take { probe: false } } else { Verdict::Skip(Skip::PricedOut) }
    } else {
        let rate = if f.lane_n >= LANE_MIN_SAMPLES { f.lane_rate } else { price.work.rate() };
        match rate {
            None => Verdict::Take { probe: true },
            Some(r) if r >= bar => Verdict::Take { probe: false },
            Some(_) => Verdict::Skip(Skip::PricedOut),
        }
    }
}

/// A stretch in a busy step: its expected tokens `e` over its rung's measured cost `p_ms` must beat the batch.
/// No cost known: not taken (the copy round serves it and gets measured).
pub fn stretch_pays(e: f64, p_ms: Option<f64>, batch: f64) -> bool {
    p_ms.map_or(false, |p| p > 0.0 && e / p >= MARGIN * batch)
}

// ---- counters (`velogb10_copy_span_lanes_*`) ------------------------------------------------------------------

static ST: AtomicU64 = AtomicU64::new(0);
static BOOT_FALLBACK: AtomicU64 = AtomicU64::new(0);
static STEPS: AtomicU64 = AtomicU64::new(0);
static ROUNDS: AtomicU64 = AtomicU64::new(0);
static STRETCHES: AtomicU64 = AtomicU64::new(0);
static OTHER: AtomicU64 = AtomicU64::new(0);
static PROBES: AtomicU64 = AtomicU64::new(0);
static EMITTED: AtomicU64 = AtomicU64::new(0);
static STRETCH_PRICED: AtomicU64 = AtomicU64::new(0);
static SKIPS: [AtomicU64; Skip::ALL.len()] = [const { AtomicU64::new(0) }; Skip::ALL.len()];
static RATE: AtomicU64 = AtomicU64::new(0);
static C_MS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static ACC: AtomicU64 = AtomicU64::new(0);
static WORK_RATE: AtomicU64 = AtomicU64::new(0);
static PRICED: AtomicU64 = AtomicU64::new(0);

pub fn set_state(st: State, priced: bool) {
    ST.store(st as u64, Relaxed);
    PRICED.store(priced as u64, Relaxed);
    if !matches!(st, State::NotAsked | State::On) { BOOT_FALLBACK.fetch_add(1, Relaxed); }
}
pub fn state() -> u64 { ST.load(Relaxed) }
pub fn count_skip(s: Skip) { SKIPS[s.idx()].fetch_add(1, Relaxed); }
pub fn skips(s: Skip) -> u64 { SKIPS[s.idx()].load(Relaxed) }
/// One busy step in which at least one lane took copy work.
pub fn count_step() { STEPS.fetch_add(1, Relaxed); }
/// What one taken lane's work turned out to be: 0 = a copy round, 1 = a stretch, 2 = neither (a control round or a
/// plain step had priority, or the proposer returned nothing: the lone path served the lane its own round).
pub fn count_work(kind: usize, probe: bool, emitted: u64) {
    let first = ROUNDS.load(Relaxed) + STRETCHES.load(Relaxed) == 0 && kind < 2;
    match kind { 0 => &ROUNDS, 1 => &STRETCHES, _ => &OTHER }.fetch_add(1, Relaxed);
    if probe { PROBES.fetch_add(1, Relaxed); }
    EMITTED.fetch_add(emitted, Relaxed);
    if first {
        crate::rprintln!("[exl3-serve] copy-span-lanes: first copy work served while other requests were busy ({})",
                         if kind == 0 { "a copy round" } else { "a prefill-width stretch" });
    }
}
pub fn rounds() -> u64 { ROUNDS.load(Relaxed) }
pub fn stretches() -> u64 { STRETCHES.load(Relaxed) }
pub fn count_stretch_priced() { STRETCH_PRICED.fetch_add(1, Relaxed); }
pub fn publish(batch: f64, p: &Price) {
    RATE.store(batch.to_bits(), Relaxed);
    for (i, a) in C_MS.iter().enumerate() { a.store(p.round_ms_raw(i == 1).to_bits(), Relaxed); }
    ACC.store(p.accept().to_bits(), Relaxed);
    WORK_RATE.store(p.work.rate().unwrap_or(0.0).to_bits(), Relaxed);
}

fn f(a: &AtomicU64) -> f64 { f64::from_bits(a.load(Relaxed)) }

pub fn metrics_append(o: &mut String) {
    let st = ST.load(Relaxed);
    o.push_str("# HELP velogb10_copy_span_lanes_state --copy-span-lanes on this boot: 0 = not asked, 1 = on, 2 = asked but off (--copy-span not armed), 3 = asked but off (--lane-order fcfs), 4 = asked but off (--max-batch 1), 5 = asked but off (a server without busy-step copy work: NVFP4/bf16 trunk, DeepSeek V4, Naive).\n# TYPE velogb10_copy_span_lanes_state gauge\n");
    o.push_str(&format!("velogb10_copy_span_lanes_state {st}\n"));
    o.push_str("# HELP velogb10_copy_span_lanes_priced --copy-span-lanes-price as armed (1 = on).\n# TYPE velogb10_copy_span_lanes_priced gauge\n");
    o.push_str(&format!("velogb10_copy_span_lanes_priced {}\n", PRICED.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_lanes_steps_total Steps with two or more busy requests in which at least one request ran copy work.\n# TYPE velogb10_copy_span_lanes_steps_total counter\n");
    o.push_str(&format!("velogb10_copy_span_lanes_steps_total {}\n", STEPS.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_lanes_work_total Copy work run by one request while other requests were busy, by kind (round = a copy round, stretch = a --copy-span-prefill stretch, other = the request was taken but its lone path ran a control round / plain step / neural round instead). Also counted in velogb10_copy_span_rounds_total / velogb10_copy_span_prefill_steps_total.\n# TYPE velogb10_copy_span_lanes_work_total counter\n");
    o.push_str(&format!("velogb10_copy_span_lanes_work_total{{kind=\"round\"}} {}\n", ROUNDS.load(Relaxed)));
    o.push_str(&format!("velogb10_copy_span_lanes_work_total{{kind=\"stretch\"}} {}\n", STRETCHES.load(Relaxed)));
    o.push_str(&format!("velogb10_copy_span_lanes_work_total{{kind=\"other\"}} {}\n", OTHER.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_lanes_emitted_tokens_total Tokens emitted by that copy work.\n# TYPE velogb10_copy_span_lanes_emitted_tokens_total counter\n");
    o.push_str(&format!("velogb10_copy_span_lanes_emitted_tokens_total {}\n", EMITTED.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_lanes_probes_total Copy work taken with nothing measured to price it yet (it measures the costs).\n# TYPE velogb10_copy_span_lanes_probes_total counter\n");
    o.push_str(&format!("velogb10_copy_span_lanes_probes_total {}\n", PROBES.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_lanes_skip_total Busy requests kept in the step's arms (per request per step), by reason: no_copy = nothing to copy; priced_out = copy work that would not beat the batch's rate; spec_off = the request does not speculate; clamped = its round depth is 0.\n# TYPE velogb10_copy_span_lanes_skip_total counter\n");
    for s in Skip::ALL {
        o.push_str(&format!("velogb10_copy_span_lanes_skip_total{{reason=\"{}\"}} {}\n", s.label(), SKIPS[s.idx()].load(Relaxed)));
    }
    o.push_str("# HELP velogb10_copy_span_lanes_stretch_priced_out_total Stretches the lone rule picked in a busy step but the batch price held back (a copy round served instead).\n# TYPE velogb10_copy_span_lanes_stretch_priced_out_total counter\n");
    o.push_str(&format!("velogb10_copy_span_lanes_stretch_priced_out_total {}\n", STRETCH_PRICED.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_lanes_fallback_total Boots asked for --copy-span-lanes on that run it off (never a refusal), by reason.\n# TYPE velogb10_copy_span_lanes_fallback_total counter\n");
    let boot = match st { 2 => "copy_span_off", 3 => "lane_order_fcfs", 4 => "max_batch_1", 5 => "server", _ => "" };
    for r in ["copy_span_off", "lane_order_fcfs", "max_batch_1", "server"] {
        let v = if r == boot { BOOT_FALLBACK.load(Relaxed) } else { 0 };
        o.push_str(&format!("velogb10_copy_span_lanes_fallback_total{{reason=\"{r}\"}} {v}\n"));
    }
    o.push_str("# HELP velogb10_copy_span_lanes_batch_tok_s The batch rate the last busy step priced against (tokens/s over all busy requests).\n# TYPE velogb10_copy_span_lanes_batch_tok_s gauge\n");
    o.push_str(&format!("velogb10_copy_span_lanes_batch_tok_s {:.1}\n", f(&RATE) * 1e3));
    o.push_str("# HELP velogb10_copy_span_lanes_copy_round_ms Measured copy-round time by width (chain = up to the chain verify's drafts, wide = the wide copy verify); 0 = not measured.\n# TYPE velogb10_copy_span_lanes_copy_round_ms gauge\n");
    o.push_str(&format!("velogb10_copy_span_lanes_copy_round_ms{{width=\"chain\"}} {:.3}\n", f(&C_MS[0])));
    o.push_str(&format!("velogb10_copy_span_lanes_copy_round_ms{{width=\"wide\"}} {:.3}\n", f(&C_MS[1])));
    o.push_str("# HELP velogb10_copy_span_lanes_cursor_accept Accept fraction of copy rounds that continued a running copy (the prior until measured).\n# TYPE velogb10_copy_span_lanes_cursor_accept gauge\n");
    o.push_str(&format!("velogb10_copy_span_lanes_cursor_accept {:.4}\n", f(&ACC)));
    o.push_str("# HELP velogb10_copy_span_lanes_work_tok_s The server's measured copy-work rate (copy rounds and stretches, tokens/s; decayed sums).\n# TYPE velogb10_copy_span_lanes_work_tok_s gauge\n");
    o.push_str(&format!("velogb10_copy_span_lanes_work_tok_s {:.1}\n", f(&WORK_RATE) * 1e3));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(peek: Option<(usize, bool)>, batch: f64) -> Facts {
        Facts { spec_on: true, depth: 15, peek, wide: true, lane_rate: None, lane_n: 0, batch, priced: true, stretch: false }
    }

    #[test]
    fn resolve_never_refuses() {
        assert_eq!(resolve(false, true, false, 4), State::NotAsked);
        assert_eq!(resolve(true, false, false, 4), State::OffNoCopySpan);
        assert_eq!(resolve(true, true, true, 4), State::OffFcfs);
        assert_eq!(resolve(true, true, false, 1), State::OffOneLane);
        assert_eq!(resolve(true, true, false, 2), State::On);
        for st in [State::On, State::OffNoCopySpan, State::OffFcfs, State::OffOneLane] {
            for w in [1, 2, 4] {
                assert!(boot_line(st, w, true).unwrap().starts_with("[exl3-serve] copy-span-lanes: "));
            }
        }
        assert!(boot_line(State::NotAsked, 1, true).is_none());
        assert!(boot_line(State::On, 4, true).unwrap().contains("TP=4"));
        assert!(boot_line(State::On, 1, false).unwrap().contains("Price OFF"));
    }

    /// ULTRA2: --copy-span-lanes on the trunk (NVFP4/bf16) server or a DeepSeek V4 / Naive bundle falls back with one
    /// line, state 5 and the fallback counter (never a refusal); the trunk calls the resolver once, right after it
    /// resolves --copy-span, and reads the flag nowhere else.
    #[test]
    fn other_servers_fall_back_with_one_line() {
        assert_eq!(State::OffServer as u64, 5);
        assert_eq!(State::OffServer.reason(), "server");
        let l = boot_line(State::OffServer, 1, true).unwrap();
        assert!(l.starts_with("[serve] copy-span-lanes: asked (--copy-span-lanes on), runs OFF") && !l.contains('\n'));
        assert!(l.contains("velogb10_copy_span_lanes_state 5"));
        let mut o = String::new();
        metrics_append(&mut o);
        assert!(o.contains("velogb10_copy_span_lanes_fallback_total{reason=\"server\"} "));
        let batch = src_text!("batch");
        let r = batch.find("crate::exl3_serve::copy_span_resolve_trunk(").expect("trunk resolve");
        let f = batch.find("crate::copy_lanes::resolve_other_server()").expect("trunk lanes fallback");
        assert!(f > r && f - r < 600, "the lanes fallback follows the trunk's --copy-span resolution");
        assert_eq!(batch.matches("copy-span-lanes\")").count(), 0, "the trunk reads --copy-span-lanes only through the resolver");
        let ex = src_text!("exl3_serve");
        let ff = ex.find("pub fn copy_span_family_fallback()").unwrap();
        assert!(ex[ff..ff + 900].contains("crate::copy_lanes::resolve_other_server()"), "DeepSeek V4 / Naive bundles");
    }

    #[test]
    fn batch_rate_takes_the_best_known() {
        assert_eq!(batch_rate(&[], None, 0.087), 0.087);
        assert_eq!(batch_rate(&[0.2, 0.15], Some(0.18), 0.087), 0.2);
        assert_eq!(batch_rate(&[f64::NAN, -1.0], Some(0.3), 0.087), 0.3);
        assert_eq!(batch_rate(&[0.0], Some(f64::INFINITY), 0.05), 0.05);
    }

    #[test]
    fn decide_reasons_and_probes() {
        let p = Price::default();
        let mut f = facts(Some((15, true)), 0.2);
        f.spec_on = false;
        assert_eq!(decide(&p, &f), Verdict::Skip(Skip::SpecOff));
        let mut f = facts(Some((15, true)), 0.2);
        f.depth = 0;
        assert_eq!(decide(&p, &f), Verdict::Skip(Skip::Clamped));
        assert_eq!(decide(&p, &facts(None, 0.2)), Verdict::Skip(Skip::NoCopy));
        assert_eq!(decide(&p, &facts(Some((0, true)), 0.2)), Verdict::Skip(Skip::NoCopy));
        // nothing measured: a probe (cursor and start alike)
        assert_eq!(decide(&p, &facts(Some((15, true)), 0.2)), Verdict::Take { probe: true });
        assert_eq!(decide(&p, &facts(Some((4, false)), 0.2)), Verdict::Take { probe: true });
        // price off: every proposal
        let mut f = facts(Some((1, false)), 1e9);
        f.priced = false;
        assert_eq!(decide(&p, &f), Verdict::Take { probe: false });
    }

    /// The cursor rule at worked points: a 16-row copy round of 25 ms at 85% accept = 13.75 tokens = 0.55 tok/ms.
    #[test]
    fn decide_cursor_worked_points() {
        let mut p = Price::default();
        p.note_copy_round(25.0, 15, 15, true, true);
        assert_eq!(p.round_ms(true), Some(25.0));
        assert_eq!(p.round_ms(false), Some(25.0), "the other class stands in until measured");
        assert!((p.accept() - 1.0).abs() < 1e-12);
        p.note_copy_round(25.0, 15, 9, true, true); // 0.6 -> acc = 1 + 0.25 (0.6 - 1) = 0.9
        assert!((p.accept() - 0.9).abs() < 1e-12);
        // E = 1 + 0.9 * 15 = 14.5 over 25 ms = 0.58 tok/ms
        assert_eq!(decide(&p, &facts(Some((15, true)), 0.5)), Verdict::Take { probe: false }); // bar 0.575
        assert_eq!(decide(&p, &facts(Some((15, true)), 0.51)), Verdict::Skip(Skip::PricedOut)); // bar 0.5865
        // a running copy whose stretch pays is taken even when its copy round alone would not
        let mut fs = facts(Some((15, true)), 0.51);
        fs.stretch = true;
        assert_eq!(decide(&p, &fs), Verdict::Take { probe: false });
        // a short tail of a copy: 2 tokens -> E = 2.8 over 25 ms = 0.112
        assert_eq!(decide(&p, &facts(Some((2, true)), 0.2)), Verdict::Skip(Skip::PricedOut));
        assert_eq!(decide(&p, &facts(Some((2, true)), 0.09)), Verdict::Take { probe: false });
        // search proposals do not touch the cursor accept
        p.note_copy_round(10.0, 4, 0, false, false);
        assert!((p.accept() - 0.9).abs() < 1e-12);
        assert_eq!(p.round_ms(false), Some(10.0));
    }

    /// Starts are priced by copy-work rates: the lane's own once it has LANE_MIN_SAMPLES, else the server's.
    #[test]
    fn decide_start_by_work_rate() {
        let mut p = Price::default();
        p.work.note(16, 25.0); // 0.64 tok/ms
        let mut f = facts(Some((4, false)), 0.2);
        assert_eq!(decide(&p, &f), Verdict::Take { probe: false });
        // an agent lane whose copies land ~2 tokens per 20 ms round: 0.1 tok/ms
        let mut lane = Work::default();
        for _ in 0..LANE_MIN_SAMPLES { lane.note(2, 20.0); }
        f.lane_rate = lane.rate();
        f.lane_n = 3;
        assert_eq!(decide(&p, &f), Verdict::Take { probe: false }, "below the lane minimum: the server's rate");
        f.lane_n = lane.n;
        assert_eq!(decide(&p, &f), Verdict::Skip(Skip::PricedOut));
        f.batch = 0.08;
        assert_eq!(decide(&p, &f), Verdict::Take { probe: false });
    }

    #[test]
    fn work_sums_decay() {
        let mut w = Work::default();
        assert_eq!(w.rate(), None);
        w.note(10, 0.0); // ignored
        assert_eq!(w.rate(), None);
        w.note(10, 10.0);
        assert!((w.rate().unwrap() - 1.0).abs() < 1e-12);
        for _ in 0..40 { w.note(1, 10.0); }
        assert!((w.rate().unwrap() - 0.1).abs() < 1e-3, "old samples decay away");
    }

    #[test]
    fn stretch_price() {
        assert!(!stretch_pays(500.0, None, 0.2));
        assert!(!stretch_pays(500.0, Some(0.0), 0.2));
        assert!(stretch_pays(100.0, Some(200.0), 0.2)); // 0.5 >= 0.23
        assert!(!stretch_pays(40.0, Some(200.0), 0.2)); // 0.2 < 0.23
    }

    #[test]
    fn metrics_render_every_series() {
        let mut o = String::new();
        metrics_append(&mut o);
        for name in ["velogb10_copy_span_lanes_state ", "velogb10_copy_span_lanes_priced ", "velogb10_copy_span_lanes_steps_total ",
                     "velogb10_copy_span_lanes_work_total{kind=\"round\"} ", "velogb10_copy_span_lanes_work_total{kind=\"stretch\"} ",
                     "velogb10_copy_span_lanes_work_total{kind=\"other\"} ", "velogb10_copy_span_lanes_emitted_tokens_total ",
                     "velogb10_copy_span_lanes_probes_total ", "velogb10_copy_span_lanes_skip_total{reason=\"no_copy\"} ",
                     "velogb10_copy_span_lanes_skip_total{reason=\"priced_out\"} ", "velogb10_copy_span_lanes_skip_total{reason=\"spec_off\"} ",
                     "velogb10_copy_span_lanes_skip_total{reason=\"clamped\"} ", "velogb10_copy_span_lanes_stretch_priced_out_total ",
                     "velogb10_copy_span_lanes_fallback_total{reason=\"copy_span_off\"} ", "velogb10_copy_span_lanes_batch_tok_s ",
                     "velogb10_copy_span_lanes_copy_round_ms{width=\"wide\"} ", "velogb10_copy_span_lanes_cursor_accept ",
                     "velogb10_copy_span_lanes_work_tok_s "] {
            assert!(o.contains(name), "missing {name}");
        }
    }
}
