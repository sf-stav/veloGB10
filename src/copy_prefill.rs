//! COPYPF1 (2026-11-12): prefill-width copy verify, the HOST logic (no GPU; unit-tested here).
//!
//! `--copy-span-prefill on` (default OFF, meaningful only with `--copy-span on`): when a lone
//! request's copy is ESTABLISHED (its copy cursor is live, which by `CopySpan::commit` means the
//! previous copy round accepted every copied token and the model's own next token continued the
//! source), the next N copied tokens are verified in ONE chunked-prefill forward (the prompt-reading
//! path, `FwdModel::prefill_chunk`) whose rows all go through the lm_head and the verify's sampler:
//!
//!   chunk = [b, c_1 .. c_N] at positions p .. p+N  (b = the lane's unconsumed bonus token)
//!   row i's choice x_i (greedy argmax, or a draw with the lane's (seed, ctr + i)) is compared with
//!   c_{i+1}; a = the longest agreeing prefix; emitted = c_1 .. c_a, x_a (x_N when all agree).
//!
//! That is the speculative verify's sample-then-compare rule over N+1 rows: every emitted token is
//! the model's own choice at its position, and the copy is only a guess. The state is committed for
//! rows 0..=a only: when a < N the running state (GDN, conv, PLE ring, head-fill carry, PLE n-gram
//! pair) is restored from a checkpoint taken before the chunk and the a+1 rows are prefilled again.
//! The prefill path is not bit-identical to the decode path, so with the flag on, output can differ
//! on near-ties from the same request served without it: that is why it is opt-in.
//!
//! This module holds what can be decided without a device: the chunk-width schedule, the accept
//! prefix, the emitted tokens, the per-step decision and its fallback reasons, the boot resolution
//! and its line, and the `velogb10_copy_span_prefill_*` counters. The device step is
//! `FwdModel::copy_prefill_verify` (exl3_forward/cpv.rs); the serving arm is
//! `Exl3Scheduler::step_cpv` (exl3_serve.rs).
//!
//! Hardware: gated at TP=1 on one Spark (probe exact, file returns exact). TP=2 / TP=4 (COPYPF1-TP): every rank
//! runs the stretch at the same program point; the serving arm agrees the stretch before the chunk, the accept
//! count and every judged choice before the re-run, and the emitted tokens after it (three tp_round lockstep
//! points, counted in velogb10_copy_span_prefill_tp_lockstep_total).

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Fewer copied tokens than this: the normal copy round (a <= 16-row verify) serves the round.
pub const CPV_MIN_N: usize = 16;

/// The chunk widths (rows = offered tokens + 1) of the schedule's first rungs; the last rung is the
/// served prefill chunk (`--prefill-chunk`). 32 / 160 / 512 are widths the boot's prefill warm-up
/// already runs (`FwdModel::prefill_warmup_widths`), so the first stretches pay no first-use cost.
pub const CPV_RUNGS: [usize; 3] = [32, 160, 512];

/// `--copy-span-prefill-max-rows` default: the largest stretch in prefill rows. A stretch is one scheduler step (its
/// tokens are emitted together and every other lane waits for it), so this bounds the longest pause in a stream.
pub const CPV_MAX_ROWS_DEFAULT: usize = 512;

static MAX_ROWS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(CPV_MAX_ROWS_DEFAULT);

/// Boot: `--copy-span-prefill-max-rows` (never below one row).
pub fn set_max_rows(n: usize) { MAX_ROWS.store(n.max(1), Relaxed); }

pub fn max_rows() -> usize { MAX_ROWS.load(Relaxed) }

/// `--copy-span-prefill-min-run` (default 0 = a stretch may follow the first fully accepted copy round, as gated).
static MIN_RUN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn set_min_run(n: usize) { MIN_RUN.store(n, Relaxed); }

pub fn min_run() -> usize { MIN_RUN.load(Relaxed) }

/// `--copy-span-prefill-price` (ECHOPRICE, default on): take a stretch only when it is priced to beat the copy rounds
/// it replaces (`Price`); off = the gated rule (every established copy, from the first rung).
static PRICED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn set_priced(on: bool) { PRICED.store(on, Relaxed); }

pub fn priced() -> bool { PRICED.load(Relaxed) }

// ---- ECHOPRICE: the priced stretch rule ---------------------------------------------------------
//
// A stretch costs one prefill-class pass (plus a re-run of the agreed rows when the model disagrees); its fixed part
// (the dense weights at >= 17 rows) is paid whether the copy then lands 20 tokens or 500. Copy rounds cost one
// decode-class verify per (up to) --copy-span-max-w + 1 tokens. So a stretch pays only when the copy goes on long
// enough: with ρ = the copy rounds' throughput on a copy that keeps going (tokens per ms) and P = what a stretch at
// this rung costs (measured, re-runs included), the stretch is taken iff the tokens it is expected to emit, E,
// satisfy E >= ρ · P (the copy rounds would need at least as long for them).
//
// E is not known in advance. The estimate: a copy that has already landed r tokens is expected to go on about as far
// again (the longer a copy has run, the longer it tends to run: a file return keeps going, an agent's quote of a
// config line stops), and a request whose earlier copies ran L tokens on average is expected to copy about L again;
// E = min(max(r, L), n) + 1 over the n tokens the rung offers. The rung is chosen among the lane's schedule rung and
// the ones above it by the best expected rate E / P (a long expected copy starts wide instead of climbing; a short one
// stays on a narrow, cheaper pass), and that best rate must reach the copy rounds' throughput. Both costs are measured while serving: ρ from fully accepted copy rounds
// ((max_w + 1) tokens over the round's ms), P per rung from the stretches served (exponential means, weight 1/4).
// Under TP every input is replicated: the round times are the head's (tp_round), the copy state is committed
// identically on every rank. Until a rung has a measured stretch, P = the largest measured rung, else
// PRICE_PRIOR_ROUNDS copy rounds.

/// The stretch cost assumed (in copy-round times) before any stretch has been measured on this server.
pub const PRICE_PRIOR_ROUNDS: f64 = 4.0;
const PRICE_W: f64 = 0.25;

/// The measured costs the priced rule compares (one per scheduler; updated from TP-agreed round times).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Price {
    /// copy rounds: mean ms of a fully accepted copy round, and the tokens such a round delivers at full width
    c_ms: f64,
    c_tok: f64,
    c_n: u64,
    /// stretches: mean ms per rung (rungs past the last fixed one share the last slot)
    p_ms: [f64; CPV_RUNGS.len() + 1],
    p_n: [u64; CPV_RUNGS.len() + 1],
}

fn ema(old: f64, x: f64, n: u64) -> f64 { if n == 0 { x } else { old + PRICE_W * (x - old) } }

impl Price {
    /// A copy round whose every copied token was accepted, at `max_w` (the round's width cap), took `ms`.
    pub fn note_copy_round(&mut self, ms: f64, max_w: usize) {
        if !(ms > 0.0) { return; }
        self.c_ms = ema(self.c_ms, ms, self.c_n);
        self.c_tok = (max_w + 1) as f64;
        self.c_n += 1;
        publish(self);
    }

    /// A stretch at schedule rung `rung` took `ms` (chunk, judge, re-run).
    pub fn note_stretch(&mut self, rung: usize, ms: f64) {
        if !(ms > 0.0) { return; }
        let i = rung.min(CPV_RUNGS.len());
        self.p_ms[i] = ema(self.p_ms[i], ms, self.p_n[i]);
        self.p_n[i] += 1;
        publish(self);
    }

    /// Copy rounds' throughput (tokens per ms) on a copy that keeps going; None before any full copy round.
    pub fn copy_rate(&self) -> Option<f64> {
        (self.c_n > 0 && self.c_ms > 0.0).then(|| self.c_tok / self.c_ms)
    }

    /// The cost of a stretch at `rung` (ms): measured, else the largest measured rung, else the prior.
    pub fn stretch_ms(&self, rung: usize) -> Option<f64> {
        let i = rung.min(CPV_RUNGS.len());
        if self.p_n[i] > 0 { return Some(self.p_ms[i]); }
        let m = (0..self.p_ms.len()).filter(|&j| self.p_n[j] > 0).map(|j| self.p_ms[j]).fold(0.0f64, f64::max);
        if m > 0.0 { return Some(m); }
        (self.c_n > 0).then(|| PRICE_PRIOR_ROUNDS * self.c_ms)
    }

    /// The tokens a stretch must be expected to emit to pay at `rung` (ρ · P); None = nothing measured yet.
    pub fn need(&self, rung: usize) -> Option<f64> {
        Some(self.copy_rate()? * self.stretch_ms(rung)?)
    }
}

/// ECHOPRICE: the rung a stretch starts at and whether it pays. Over the rungs from the lane's schedule rung up,
/// the expected emit at rung g is E_g = min(want, offer(g)) + 1 and its rate E_g / P_g (P_g = `Price::stretch_ms`);
/// the rung with the best expected rate is chosen (the first on a tie), and the stretch pays iff that rate is at
/// least the copy rounds' throughput. Returns (rung, pays); with nothing measured yet: (the lane rung, true).
pub fn priced_pick(price: &Price, lane_rung: usize, want: usize, offer: impl Fn(usize) -> usize) -> (usize, bool) {
    let top = CPV_RUNGS.len();
    let g0 = lane_rung.min(top);
    let Some(rho) = price.copy_rate() else { return (g0, true) };
    let mut best: Option<(usize, f64)> = None;
    for g in g0..=top {
        let Some(p) = price.stretch_ms(g) else { return (g0, true) };
        let e = (want.min(offer(g)) + 1) as f64;
        let r = e / p.max(1e-9);
        if best.map_or(true, |(_, b)| r > b) { best = Some((g, r)); }
    }
    let (g, r) = best.unwrap();
    (g, r >= rho)
}

/// ECHOPRICE: the copy length a stretch is expected to cover: the longer of the live copy's landed tokens and the
/// request's mean ended-copy length.
pub fn expected_want(run: usize, hist: f64) -> usize {
    run.max(hist.round() as usize)
}

static PRICE_C_MS: AtomicU64 = AtomicU64::new(0);
static PRICE_C_TOK: AtomicU64 = AtomicU64::new(0);
static PRICE_P_MS: [AtomicU64; CPV_RUNGS.len() + 1] = [const { AtomicU64::new(0) }; CPV_RUNGS.len() + 1];

fn publish(p: &Price) {
    PRICE_C_MS.store(p.c_ms.to_bits(), Relaxed);
    PRICE_C_TOK.store(p.c_tok.to_bits(), Relaxed);
    for (i, a) in PRICE_P_MS.iter().enumerate() {
        a.store(if p.p_n[i] > 0 { p.p_ms[i] } else { 0.0 }.to_bits(), Relaxed);
    }
}

/// The schedule's top rung: the served prefill chunk capped by `--copy-span-prefill-max-rows`. Every rung is capped
/// by it (`chunk_width` takes it as its `served_chunk`).
pub fn schedule_top(served_chunk: usize, max_rows: usize) -> usize {
    served_chunk.min(max_rows).max(1)
}

/// Rows of the stretch chunk at schedule rung `rung` (0 = the first): the rung's width, the last
/// rung = `served_chunk`; never above `served_chunk` or the prefill scratch (`psc_rows`).
pub fn chunk_width(rung: usize, served_chunk: usize, psc_rows: usize) -> usize {
    let cap = served_chunk.min(psc_rows).max(1);
    let w = if rung < CPV_RUNGS.len() { CPV_RUNGS[rung] } else { served_chunk };
    w.min(cap)
}

/// The next rung: one up after a fully accepted stretch (capped at the served-chunk rung), back to
/// the first after a disagreement.
pub fn next_rung(rung: usize, full: bool) -> usize {
    if full { (rung + 1).min(CPV_RUNGS.len()) } else { 0 }
}

/// Copied tokens offered by one stretch: the chunk width minus the bonus row, then the copy source
/// left after the cursor (`src_left`) and the round clamps (`depth_cap` = `round_depth_for` before
/// the sparse-switch clamp: end of the KV window, the remaining token budget, the thinking budget).
pub fn offered_len(rung: usize, served_chunk: usize, psc_rows: usize, src_left: usize, depth_cap: usize) -> usize {
    chunk_width(rung, served_chunk, psc_rows).saturating_sub(1).min(src_left).min(depth_cap)
}

/// The accepted prefix: the number of leading rows whose choice equals the copied token that
/// follows it (`choices[i]` judges `offered[i]`). `choices` may stop early (the device step stops
/// at the first disagreeing group); a row it does not hold is never counted as agreeing.
pub fn accept_prefix(choices: &[i32], offered: &[i32]) -> usize {
    choices.iter().zip(offered).take_while(|(x, c)| x == c).count()
}

/// The tokens a stretch emits: the `a` accepted copied tokens, then the model's own choice at row
/// `a` (the bonus row `N` when all agreed). None when row `a` was not computed (a caller bug).
pub fn emitted(offered: &[i32], choices: &[i32], a: usize) -> Option<Vec<i32>> {
    if a > offered.len() || a >= choices.len() || offered[..a] != choices[..a] { return None; }
    let mut v = offered[..a].to_vec();
    v.push(choices[a]);
    Some(v)
}

/// Rows re-run after a disagreement at row `a` of an `n`-token stretch (0 = all agreed: no re-run).
pub fn rerun_rows(a: usize, n: usize) -> usize {
    if a < n { a + 1 } else { 0 }
}

/// Why an applicable step (a live copy cursor on a lone speculating lane) did not take the
/// prefill-width verify. Counted in `velogb10_copy_span_prefill_fallback_total{reason}`; the normal
/// copy round serves the round instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Skip {
    /// the request has presence / repetition / frequency penalties (per-row penalties over the
    /// copied history are not built in this version)
    Penalty,
    /// the request asked for log probabilities (the stretch's rows are not kept for them)
    Logprobs,
    /// an image request (its 3-axis RoPE map is not wired into the stretch)
    Image,
    /// a prompt is being prefilled between this lane's steps (`--prefill-interleave`): the
    /// prefill scratch holds its state
    PrefillInterleave,
    /// fewer than CPV_MIN_N copied tokens are available (source end, budget, window)
    Short,
    /// the lane has no saved draft tap (it predicts the bonus; needed for the head's KV rows)
    NoTap,
    /// the prefill scratch is not built yet
    NoScratch,
    /// the copy has produced fewer than --copy-span-prefill-min-run tokens so far (a short copy is cheaper in copy
    /// rounds: a 32-row prefill reads about as many expert weights as a 512-row one)
    Young,
    /// ECHOPRICE (--copy-span-prefill-price on, the default): the stretch is not expected to emit enough tokens to beat
    /// the copy rounds it would replace (measured costs; the copy is expected to stop too soon)
    Priced,
}

impl Skip {
    pub const ALL: [Skip; 9] = [Skip::Penalty, Skip::Logprobs, Skip::Image, Skip::PrefillInterleave, Skip::Short,
                                Skip::NoTap, Skip::NoScratch, Skip::Young, Skip::Priced];
    pub fn reason(self) -> &'static str {
        match self {
            Skip::Penalty => "penalty",
            Skip::Logprobs => "logprobs",
            Skip::Image => "image",
            Skip::PrefillInterleave => "prefill_interleave",
            Skip::Short => "short",
            Skip::NoTap => "no_tap",
            Skip::NoScratch => "no_scratch",
            Skip::Young => "young_copy",
            Skip::Priced => "priced_out",
        }
    }
    fn idx(self) -> usize { Skip::ALL.iter().position(|&s| s == self).unwrap() }
}

/// What the serving loop knows about the lane when it asks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Facts {
    /// `--copy-span-prefill` resolved on for this boot
    pub on: bool,
    /// a lone lane's step (no packed / shared step)
    pub solo: bool,
    /// a control round or a plain step was chosen (they keep priority)
    pub plain_step: bool,
    /// the lane's copy cursor is live and no emitted token is waiting to be committed to it
    pub cursor_live: bool,
    pub in_prefill: bool,
    pub penalized: bool,
    pub logprobs: bool,
    pub image: bool,
    pub tap_ok: bool,
    pub scratch_ok: bool,
    /// the offered length `offered_len` computed
    pub n: usize,
    /// tokens the live copy has produced so far (`CopySpan::cursor_run`) and the --copy-span-prefill-min-run floor
    pub run: usize,
    pub min_run: usize,
    /// ECHOPRICE: the priced rule's verdict for this stretch (true = expected to pay, or the rule is off)
    pub pays: bool,
}

/// One step's decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// not this step's business (flag off, not solo, a plain/control step, no established copy):
    /// nothing counted, the round runs exactly as without the flag
    NotApplicable,
    /// an established copy, but this request / moment cannot take it: counted, the copy round serves
    Skip(Skip),
    /// run the prefill-width verify over `n` copied tokens
    Take(usize),
}

pub fn decide(f: &Facts) -> Decision {
    if !f.on || !f.solo || f.plain_step || !f.cursor_live { return Decision::NotApplicable; }
    let skip = if f.penalized { Some(Skip::Penalty) }
        else if f.logprobs { Some(Skip::Logprobs) }
        else if f.image { Some(Skip::Image) }
        else if f.in_prefill { Some(Skip::PrefillInterleave) }
        else if !f.scratch_ok { Some(Skip::NoScratch) }
        else if !f.tap_ok { Some(Skip::NoTap) }
        else if f.run < f.min_run { Some(Skip::Young) }
        else if f.n < CPV_MIN_N { Some(Skip::Short) }
        else if !f.pays { Some(Skip::Priced) }
        else { None };
    match skip { Some(s) => Decision::Skip(s), None => Decision::Take(f.n) }
}

/// How `--copy-span-prefill` resolved on this boot (the `velogb10_copy_span_prefill_state` gauge).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    NotAsked = 0,
    On = 1,
    /// asked on a TP server of a world the copy arm does not serve (it serves 1, 2 and 4)
    OffTp = 2,
    /// asked without an armed copy arm (`--copy-span off`, or the copy arm itself runs off)
    OffNoCopySpan = 3,
    /// asked on a boot whose DFlash2 chain is armed (its ring is primed from prefill chunks; not wired)
    OffDf2 = 4,
}

impl State {
    pub fn reason(self) -> &'static str {
        match self {
            State::NotAsked | State::On => "",
            State::OffTp => "tp",
            State::OffNoCopySpan => "copy_span_off",
            State::OffDf2 => "df2",
        }
    }
}

/// Pure boot resolution: `asked` = the flag, `copy_armed` = the copy-span arm is armed on this boot,
/// `world` = the TP world (1 = one box), `df2` = the DFlash2 chain is armed. Never a refusal.
/// COPYPF1-TP: worlds 1, 2 and 4 (the copy arm's worlds); any other world runs it off.
pub fn resolve(asked: bool, copy_armed: bool, world: usize, df2: bool) -> State {
    if !asked { State::NotAsked }
    else if !matches!(world, 1 | 2 | 4) { State::OffTp }
    else if !copy_armed { State::OffNoCopySpan }
    else if df2 { State::OffDf2 }
    else { State::On }
}

/// The one boot line (None when not asked).
/// `served_chunk` = the schedule's top rung (`schedule_top`: the prefill chunk capped by --copy-span-prefill-max-rows).
pub fn boot_line(st: State, served_chunk: usize, world: usize) -> Option<String> {
    match st {
        State::NotAsked => None,
        State::On => Some(format!("[exl3-serve] copy-span-prefill: ON (--copy-span-prefill on): once a lone request's copy is established \
(a copy round accepted every copied token), the next copied stretch is verified by the PROMPT-READING path - one chunked prefill of \
{} then {} then {} then {} rows ({}) whose every row is checked by the model's own choice (greedy or sampled with the request's own \
random stream), keeping the agreeing prefix. OUTPUT MAY DIFFER ON NEAR-TIES from the same request served without this flag (the \
prompt-reading path is not bit-identical to the decoding path). Not for penalized, logprob or image requests (they keep copy rounds). \
{}{}Counters velogb10_copy_span_prefill_*",
                                  CPV_RUNGS[0].min(served_chunk), CPV_RUNGS[1].min(served_chunk), CPV_RUNGS[2].min(served_chunk),
                                  served_chunk, "the last = --prefill-chunk capped by --copy-span-prefill-max-rows; one rung up per fully accepted stretch, back to the \
first after a disagreement; a stretch is one scheduler step, so its rows bound the longest pause in a stream",
                                  if world > 1 { format!("TP={world}: every rank runs the stretch, agreed before the chunk, before the re-run and \
after it. {} ", GATED_TP) } else { format!("{} ", GATED_TP1) },
                                  if priced() { "Priced rule ON (--copy-span-prefill-price): a stretch is taken only when the copy is expected to run long \
enough to beat the copy rounds it replaces (costs measured while serving; counted as priced_out otherwise). " }
                                  else { "Priced rule OFF (--copy-span-prefill-price off): every established copy takes a stretch. " })),
        State::OffTp => Some(format!("[exl3-serve] copy-span-prefill: asked (--copy-span-prefill on), runs OFF on this TP={world} server: the \
copy arm serves TP=1, 2 and 4 only; copy rounds are unchanged (velogb10_copy_span_prefill_state 2)")),
        State::OffNoCopySpan => Some("[exl3-serve] copy-span-prefill: asked (--copy-span-prefill on), runs OFF: it verifies copies made by \
--copy-span, which is not armed on this boot (velogb10_copy_span_prefill_state 3)".to_string()),
        State::OffDf2 => Some("[exl3-serve] copy-span-prefill: asked (--copy-span-prefill on), runs OFF: the DFlash2 chain is armed on this \
boot and its ring is fed from prefill chunks (not wired for stretches); copy rounds are unchanged (velogb10_copy_span_prefill_state 4)".to_string()),
    }
}

/// What has run on hardware (the boot line says it): kept in one place so the help text, the boot line and the
/// report cannot disagree.
pub const GATED_TP1: &str = "Gated at TP=1 on one Spark (exactness probe, file returns, identity of copied text).";
pub const GATED_TP: &str = "Gated at TP=2 and TP=4 on Sparks (greedy thinking-off identity with the no-speculation reference, file returns, \
32K/100K-token returns, stop / max_tokens / disconnect, zero lockstep failures); the exactness probe runs at TP=1 only.";

// ---- counters (`velogb10_copy_span_prefill_*`) ------------------------------------------------

static ST: AtomicU64 = AtomicU64::new(0);
static BOOT_FALLBACK: AtomicU64 = AtomicU64::new(0);
static STEPS: AtomicU64 = AtomicU64::new(0);
static OFFERED: AtomicU64 = AtomicU64::new(0);
static ACCEPTED: AtomicU64 = AtomicU64::new(0);
static EMITTED: AtomicU64 = AtomicU64::new(0);
static DISAGREE: AtomicU64 = AtomicU64::new(0);
static RERUN_ROWS: AtomicU64 = AtomicU64::new(0);
static ROWS_JUDGED: AtomicU64 = AtomicU64::new(0);
static US: AtomicU64 = AtomicU64::new(0);
static TP_LOCKSTEP: AtomicU64 = AtomicU64::new(0);
static SKIPS: [AtomicU64; Skip::ALL.len()] = [const { AtomicU64::new(0) }; Skip::ALL.len()];

pub fn set_state(st: State) {
    ST.store(st as u64, Relaxed);
    if !matches!(st, State::NotAsked | State::On) { BOOT_FALLBACK.fetch_add(1, Relaxed); }
}

pub fn state() -> u64 { ST.load(Relaxed) }

/// COPYLANES: stretches served since boot (`--copy-span-lanes` tells a stretch from a copy round by it).
pub fn steps_total() -> u64 { STEPS.load(Relaxed) }

pub fn count_skip(s: Skip) { SKIPS[s.idx()].fetch_add(1, Relaxed); }

/// COPYPF1-TP: one lockstep point of a stretch passed (three per stretch under TP; never at TP=1).
pub fn count_tp_lockstep() { TP_LOCKSTEP.fetch_add(1, Relaxed); }

/// One served stretch: `offered` copied tokens, `accepted` of them, `emitted` tokens, `rerun` rows
/// prefilled again after a disagreement, `judged` rows that went through the lm_head, its ms.
pub fn count_step(offered: usize, accepted: usize, emitted: usize, rerun: usize, judged: usize, ms: f64) {
    if STEPS.fetch_add(1, Relaxed) == 0 {
        crate::rprintln!("[exl3-serve] copy-span-prefill: first prefill-width copy verify served ({offered} copied tokens offered, \
{accepted} kept)");
    }
    OFFERED.fetch_add(offered as u64, Relaxed);
    ACCEPTED.fetch_add(accepted as u64, Relaxed);
    EMITTED.fetch_add(emitted as u64, Relaxed);
    DISAGREE.fetch_add((accepted < offered) as u64, Relaxed);
    RERUN_ROWS.fetch_add(rerun as u64, Relaxed);
    ROWS_JUDGED.fetch_add(judged as u64, Relaxed);
    US.fetch_add((ms.max(0.0) * 1000.0) as u64, Relaxed);
}

pub fn metrics_append(o: &mut String) {
    let st = ST.load(Relaxed);
    o.push_str("# HELP velogb10_copy_span_prefill_state --copy-span-prefill on this boot: 0 = not asked, 1 = on, 2 = asked but off (a TP world other than 2 or 4), 3 = asked but off (--copy-span not armed), 4 = asked but off (DFlash2 chain armed).\n# TYPE velogb10_copy_span_prefill_state gauge\n");
    o.push_str(&format!("velogb10_copy_span_prefill_state {st}\n"));
    o.push_str("# HELP velogb10_copy_span_prefill_steps_total Copied stretches verified by one chunked prefill (the prompt-reading path) instead of copy rounds. Also counted in the speculative-round totals.\n# TYPE velogb10_copy_span_prefill_steps_total counter\n");
    o.push_str(&format!("velogb10_copy_span_prefill_steps_total {}\n", STEPS.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_prefill_offered_tokens_total Copied tokens offered to prefill-width verifies.\n# TYPE velogb10_copy_span_prefill_offered_tokens_total counter\n");
    o.push_str(&format!("velogb10_copy_span_prefill_offered_tokens_total {}\n", OFFERED.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_prefill_accepted_tokens_total Copied tokens the model's own choices agreed with.\n# TYPE velogb10_copy_span_prefill_accepted_tokens_total counter\n");
    o.push_str(&format!("velogb10_copy_span_prefill_accepted_tokens_total {}\n", ACCEPTED.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_prefill_emitted_tokens_total Tokens emitted by prefill-width verifies (accepted + the model's own token).\n# TYPE velogb10_copy_span_prefill_emitted_tokens_total counter\n");
    o.push_str(&format!("velogb10_copy_span_prefill_emitted_tokens_total {}\n", EMITTED.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_prefill_disagree_steps_total Prefill-width verifies that ended at a row where the model chose another token than the copy.\n# TYPE velogb10_copy_span_prefill_disagree_steps_total counter\n");
    o.push_str(&format!("velogb10_copy_span_prefill_disagree_steps_total {}\n", DISAGREE.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_prefill_rerun_rows_total Rows prefilled a second time after a disagreement (state restored to before the stretch, the agreed rows run again).\n# TYPE velogb10_copy_span_prefill_rerun_rows_total counter\n");
    o.push_str(&format!("velogb10_copy_span_prefill_rerun_rows_total {}\n", RERUN_ROWS.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_prefill_rows_judged_total Stretch rows that went through the lm_head and the sampler (stops at the first disagreeing group of 16).\n# TYPE velogb10_copy_span_prefill_rows_judged_total counter\n");
    o.push_str(&format!("velogb10_copy_span_prefill_rows_judged_total {}\n", ROWS_JUDGED.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_prefill_seconds_total Wall time of the prefill-width verifies (chunk, heads, re-run, host side).\n# TYPE velogb10_copy_span_prefill_seconds_total counter\n");
    o.push_str(&format!("velogb10_copy_span_prefill_seconds_total {:.6}\n", US.load(Relaxed) as f64 / 1e6));
    o.push_str("# HELP velogb10_copy_span_prefill_tp_lockstep_total TP only: stretch lockstep points passed (before the chunk, before the re-run, after it: three per stretch; a rank difference aborts the link instead).\n# TYPE velogb10_copy_span_prefill_tp_lockstep_total counter\n");
    o.push_str(&format!("velogb10_copy_span_prefill_tp_lockstep_total {}\n", TP_LOCKSTEP.load(Relaxed)));
    o.push_str("# HELP velogb10_copy_span_prefill_fallback_total Asked for but not taken, by reason: boot reasons (tp, copy_span_off, df2: the flag runs off for the boot) and per-step reasons (an established copy whose request or moment cannot take it: the copy round served it).\n# TYPE velogb10_copy_span_prefill_fallback_total counter\n");
    let boot = match st { 2 => "tp", 3 => "copy_span_off", 4 => "df2", _ => "" };
    for r in ["tp", "copy_span_off", "df2"] {
        let v = if r == boot { BOOT_FALLBACK.load(Relaxed) } else { 0 };
        o.push_str(&format!("velogb10_copy_span_prefill_fallback_total{{reason=\"{r}\"}} {v}\n"));
    }
    for s in Skip::ALL {
        o.push_str(&format!("velogb10_copy_span_prefill_fallback_total{{reason=\"{}\"}} {}\n", s.reason(), SKIPS[s.idx()].load(Relaxed)));
    }
    // ECHOPRICE: the priced rule's measured inputs (0 = not measured yet on this server)
    o.push_str("# HELP velogb10_copy_span_prefill_price_rule --copy-span-prefill-price: 1 = a stretch is taken only when priced to beat the copy rounds it replaces, 0 = every established copy (the gated rule).\n# TYPE velogb10_copy_span_prefill_price_rule gauge\n");
    o.push_str(&format!("velogb10_copy_span_prefill_price_rule {}\n", priced() as u8));
    o.push_str("# HELP velogb10_copy_span_prefill_price_copy_round_ms Mean time of a fully accepted copy round (the priced rule's copy cost).\n# TYPE velogb10_copy_span_prefill_price_copy_round_ms gauge\n");
    o.push_str(&format!("velogb10_copy_span_prefill_price_copy_round_ms {:.3}\n", f64::from_bits(PRICE_C_MS.load(Relaxed))));
    o.push_str("# HELP velogb10_copy_span_prefill_price_copy_round_tokens Tokens a full copy round delivers (--copy-span-max-w + 1).\n# TYPE velogb10_copy_span_prefill_price_copy_round_tokens gauge\n");
    o.push_str(&format!("velogb10_copy_span_prefill_price_copy_round_tokens {:.0}\n", f64::from_bits(PRICE_C_TOK.load(Relaxed))));
    o.push_str("# HELP velogb10_copy_span_prefill_price_stretch_ms Mean time of a stretch per schedule rung (re-runs included).\n# TYPE velogb10_copy_span_prefill_price_stretch_ms gauge\n");
    for (i, a) in PRICE_P_MS.iter().enumerate() {
        o.push_str(&format!("velogb10_copy_span_prefill_price_stretch_ms{{rung=\"{i}\"}} {:.3}\n", f64::from_bits(a.load(Relaxed))));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_grows_on_full_and_resets_on_disagreement() {
        let c = 4095;
        assert_eq!(chunk_width(0, c, c), 32);
        assert_eq!(chunk_width(1, c, c), 160);
        assert_eq!(chunk_width(2, c, c), 512);
        assert_eq!(chunk_width(3, c, c), 4095);
        assert_eq!(chunk_width(9, c, c), 4095, "past the last rung = the served chunk");
        let mut r = 0;
        for want in [1, 2, 3, 3, 3] { r = next_rung(r, true); assert_eq!(r, want); }
        assert_eq!(next_rung(r, false), 0);
        assert_eq!(next_rung(1, false), 0);
    }

    /// The cap flag (--copy-span-prefill-max-rows) bounds every rung through the schedule's top; the default keeps the
    /// first three rungs and stops at 512 rows.
    #[test]
    fn schedule_top_caps_every_rung() {
        let c = 4095;
        let top = schedule_top(c, CPV_MAX_ROWS_DEFAULT);
        assert_eq!(top, 512);
        assert_eq!((0..6).map(|r| chunk_width(r, top, c)).collect::<Vec<_>>(), vec![32, 160, 512, 512, 512, 512]);
        let top = schedule_top(c, 160);
        assert_eq!((0..5).map(|r| chunk_width(r, top, c)).collect::<Vec<_>>(), vec![32, 160, 160, 160, 160]);
        let top = schedule_top(c, 2048);
        assert_eq!((0..5).map(|r| chunk_width(r, top, c)).collect::<Vec<_>>(), vec![32, 160, 512, 2048, 2048]);
        assert_eq!(schedule_top(c, 65536), c, "a cap above the chunk = the chunk");
        assert_eq!(schedule_top(2048, 4095), 2048);
        assert_eq!(schedule_top(c, 0), 1, "never zero rows");
        assert_eq!(offered_len(3, schedule_top(c, 512), c, 10_000, 10_000), 511);
        assert_eq!(crate::opt!("copy-span-prefill-max-rows").def().def, "512");
        assert_eq!(crate::opt!("copy-span-prefill-max-rows").def().env, "");
        assert_eq!(crate::opts::ranged_from(crate::opt!("copy-span-prefill-max-rows"), None), Ok(CPV_MAX_ROWS_DEFAULT as i64));
        assert!(crate::opts::ranged_from(crate::opt!("copy-span-prefill-max-rows"), Some("31")).is_err(), "below the first rung");
        let on = boot_line(State::On, schedule_top(c, 512), 1).unwrap();
        assert!(on.contains("32 then 160 then 512 then 512") && on.contains("--copy-span-prefill-max-rows"));
    }

    #[test]
    fn schedule_respects_the_chunk_and_the_scratch() {
        // a served chunk smaller than a rung caps every rung
        assert_eq!(chunk_width(2, 256, 256), 256);
        assert_eq!(chunk_width(0, 16, 16), 16);
        // the prefill scratch caps too (it is never smaller than the chunk in serving, but stay safe)
        assert_eq!(chunk_width(3, 4095, 2048), 2048);
        assert_eq!(chunk_width(0, 0, 0), 1, "never zero rows");
    }

    #[test]
    fn offered_len_takes_every_clamp() {
        assert_eq!(offered_len(0, 4095, 4095, 10_000, 10_000), 31);
        assert_eq!(offered_len(3, 4095, 4095, 10_000, 10_000), 4094);
        assert_eq!(offered_len(3, 4095, 4095, 700, 10_000), 700, "the copy source ends");
        assert_eq!(offered_len(3, 4095, 4095, 10_000, 99), 99, "budget / window / thinking clamp");
        assert_eq!(offered_len(2, 4095, 4095, 0, 99), 0);
    }

    #[test]
    fn accept_prefix_counts_leading_agreement_only() {
        assert_eq!(accept_prefix(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(accept_prefix(&[1, 2, 9, 4], &[1, 2, 3, 4]), 2, "no agreement counted past the first miss");
        assert_eq!(accept_prefix(&[9], &[1, 2, 3]), 0);
        assert_eq!(accept_prefix(&[1, 2], &[1, 2, 3]), 2, "rows not computed never agree");
        assert_eq!(accept_prefix(&[], &[1]), 0);
        // the bonus row (choices has one more row than offered) is never compared
        assert_eq!(accept_prefix(&[1, 2, 7], &[1, 2]), 2);
    }

    #[test]
    fn emitted_is_accepted_prefix_then_the_models_choice() {
        // disagreement at row 2: c1 c2 then the model's x2
        assert_eq!(emitted(&[10, 11, 12, 13], &[10, 11, 99], 2), Some(vec![10, 11, 99]));
        // all agree: every copied token, then the bonus row's choice
        assert_eq!(emitted(&[10, 11], &[10, 11, 55], 2), Some(vec![10, 11, 55]));
        // first row disagrees: only the model's choice
        assert_eq!(emitted(&[10, 11], &[42], 0), Some(vec![42]));
        // row a not computed, or a claimed prefix that does not agree: refused
        assert_eq!(emitted(&[10, 11], &[10, 11], 2), None);
        assert_eq!(emitted(&[10, 11], &[10, 7, 3], 2), None);
        assert_eq!(emitted(&[10], &[10, 1, 2], 3), None);
        // emitted length = accepted + 1, always
        for a in 0..=3usize {
            let off = [1, 2, 3];
            let mut ch: Vec<i32> = off[..a.min(3)].to_vec();
            ch.push(77);
            assert_eq!(emitted(&off, &ch, a).unwrap().len(), a + 1);
        }
    }

    #[test]
    fn rerun_only_after_a_disagreement() {
        assert_eq!(rerun_rows(5, 5), 0);
        assert_eq!(rerun_rows(0, 5), 1, "row 0 (the bonus token b) is re-run");
        assert_eq!(rerun_rows(3, 5), 4);
    }

    fn take() -> Facts {
        Facts { on: true, solo: true, plain_step: false, cursor_live: true, in_prefill: false, penalized: false,
                logprobs: false, image: false, tap_ok: true, scratch_ok: true, n: 31, run: 5, min_run: 0, pays: true }
    }

    #[test]
    fn decision_takes_the_step_only_when_every_condition_holds() {
        assert_eq!(decide(&take()), Decision::Take(31));
        // not applicable: nothing counted, the round runs as without the flag
        for f in [Facts { on: false, ..take() }, Facts { solo: false, ..take() },
                  Facts { plain_step: true, ..take() }, Facts { cursor_live: false, ..take() }] {
            assert_eq!(decide(&f), Decision::NotApplicable, "{f:?}");
        }
        // skips, in priority order
        assert_eq!(decide(&Facts { penalized: true, logprobs: true, ..take() }), Decision::Skip(Skip::Penalty));
        assert_eq!(decide(&Facts { logprobs: true, ..take() }), Decision::Skip(Skip::Logprobs));
        assert_eq!(decide(&Facts { image: true, ..take() }), Decision::Skip(Skip::Image));
        assert_eq!(decide(&Facts { in_prefill: true, ..take() }), Decision::Skip(Skip::PrefillInterleave));
        assert_eq!(decide(&Facts { scratch_ok: false, ..take() }), Decision::Skip(Skip::NoScratch));
        assert_eq!(decide(&Facts { tap_ok: false, ..take() }), Decision::Skip(Skip::NoTap));
        assert_eq!(decide(&Facts { n: CPV_MIN_N - 1, ..take() }), Decision::Skip(Skip::Short));
        assert_eq!(decide(&Facts { n: CPV_MIN_N, ..take() }), Decision::Take(CPV_MIN_N));
        // --copy-span-prefill-min-run: a copy younger than the floor keeps copy rounds (counted); at the floor it is taken
        assert_eq!(decide(&Facts { run: 63, min_run: 64, ..take() }), Decision::Skip(Skip::Young));
        assert_eq!(decide(&Facts { run: 64, min_run: 64, ..take() }), Decision::Take(31));
        assert_eq!(decide(&Facts { run: 0, min_run: 0, ..take() }), Decision::Take(31), "default 0 = as gated");
        assert_eq!(decide(&Facts { run: 1, min_run: 64, cursor_live: false, ..take() }), Decision::NotApplicable);
        assert_eq!(crate::opt!("copy-span-prefill-min-run").def().def, "0");
        assert_eq!(crate::opt!("copy-span-prefill-min-run").def().env, "");
        // ECHOPRICE: a stretch not priced to pay keeps copy rounds (counted); the floor is checked first
        assert_eq!(decide(&Facts { pays: false, ..take() }), Decision::Skip(Skip::Priced));
        assert_eq!(decide(&Facts { pays: false, run: 1, min_run: 64, ..take() }), Decision::Skip(Skip::Young));
        assert_eq!(crate::opt!("copy-span-prefill-price").def().def, "on");
        assert_eq!(crate::opt!("copy-span-prefill-price").def().env, "");
    }

    #[test]
    fn boot_resolution_falls_back_and_never_refuses() {
        assert_eq!(resolve(false, true, 1, false), State::NotAsked);
        assert_eq!(resolve(true, true, 1, false), State::On);
        assert_eq!(resolve(true, true, 2, false), State::On, "COPYPF1-TP: TP=2");
        assert_eq!(resolve(true, true, 4, false), State::On, "COPYPF1-TP: TP=4");
        assert_eq!(resolve(true, true, 3, false), State::OffTp);
        assert_eq!(resolve(true, true, 8, false), State::OffTp);
        assert_eq!(resolve(true, false, 4, false), State::OffNoCopySpan);
        assert_eq!(resolve(true, false, 1, false), State::OffNoCopySpan);
        assert_eq!(resolve(true, true, 1, true), State::OffDf2);
        assert!(boot_line(State::NotAsked, 4095, 1).is_none());
        let on = boot_line(State::On, 4095, 1).unwrap();
        assert!(on.contains("MAY DIFFER ON NEAR-TIES") && on.contains("prompt-reading path") && on.contains("32 then 160 then 512 then 4095"));
        let tp = boot_line(State::OffTp, 4095, 8).unwrap();
        assert!(tp.contains("TP=8") && tp.contains("runs OFF"));
        let on4 = boot_line(State::On, 512, 4).unwrap();
        assert!(on4.contains("TP=4: every rank runs the stretch") && !on4.contains("UNVERIFIED"));
        assert!(on.contains(GATED_TP1) && !on.contains("UNVERIFIED"));
        assert!(boot_line(State::OffNoCopySpan, 4095, 1).unwrap().contains("--copy-span"));
        assert!(boot_line(State::OffDf2, 4095, 1).unwrap().contains("DFlash2"));
        for s in [State::OffTp, State::OffNoCopySpan, State::OffDf2] { assert!(!s.reason().is_empty()); }
    }

    /// The flag is a registered CLI option (no environment variable), Bool, default OFF; the probe's options are
    /// registered with their documented defaults; on/off spellings read as the store reads them.
    #[test]
    fn flags_are_registered_cli_only_and_off_by_default() {
        let o = crate::opt!("copy-span-prefill").def();
        assert_eq!((o.def, o.env), ("off", ""));
        assert!(matches!(o.ty, crate::opts::Ty::Bool));
        let p = crate::opt!("probe-copy-prefill").def();
        assert_eq!((p.def, p.env), ("off", ""));
        assert_eq!(crate::opt!("probe-copy-prefill-n").def().def, "256");
        assert_eq!(crate::opt!("probe-copy-prefill-break").def().def, "0");
        assert_eq!(crate::opts::ranged_from(crate::opt!("probe-copy-prefill-n"), None), Ok(256));
        assert!(crate::opts::ranged_from(crate::opt!("probe-copy-prefill-n"), Some("15")).is_err(), "below the 16-token minimum");
        assert!(crate::opts::is_on_value("1") && !crate::opts::is_on_value("0"));
    }

    #[test]
    fn metrics_render_every_series() {
        count_skip(Skip::Short);
        let mut o = String::new();
        metrics_append(&mut o);
        for name in ["velogb10_copy_span_prefill_state ", "velogb10_copy_span_prefill_steps_total ",
                     "velogb10_copy_span_prefill_offered_tokens_total ", "velogb10_copy_span_prefill_accepted_tokens_total ",
                     "velogb10_copy_span_prefill_emitted_tokens_total ", "velogb10_copy_span_prefill_disagree_steps_total ",
                     "velogb10_copy_span_prefill_rerun_rows_total ", "velogb10_copy_span_prefill_rows_judged_total ",
                     "velogb10_copy_span_prefill_seconds_total ", "velogb10_copy_span_prefill_tp_lockstep_total ",
                     "velogb10_copy_span_prefill_fallback_total{reason=\"tp\"} ",
                     "velogb10_copy_span_prefill_fallback_total{reason=\"short\"} ",
                     "velogb10_copy_span_prefill_fallback_total{reason=\"penalty\"} "] {
            assert!(o.contains(name), "missing {name}");
        }
        for s in Skip::ALL { assert!(o.contains(&format!("reason=\"{}\"", s.reason()))); }
    }
}

#[cfg(test)]
mod price_tests {
    use super::*;

    #[test]
    fn nothing_measured_means_no_price() {
        let p = Price::default();
        assert_eq!(p.copy_rate(), None);
        assert_eq!(p.need(0), None);
    }

    #[test]
    fn need_is_copy_rate_times_stretch_cost() {
        let mut p = Price::default();
        p.note_copy_round(40.0, 15); // 16 tokens / 40 ms
        assert_eq!(p.copy_rate(), Some(0.4));
        // prior before any stretch: PRICE_PRIOR_ROUNDS copy rounds = 160 ms -> 64 tokens
        assert!((p.need(0).unwrap() - 64.0).abs() < 1e-9);
        p.note_stretch(0, 100.0);
        assert!((p.need(0).unwrap() - 40.0).abs() < 1e-9);
        // an unmeasured rung borrows the largest measured one
        p.note_stretch(2, 150.0);
        assert!((p.need(1).unwrap() - 60.0).abs() < 1e-9);
        assert!((p.need(3).unwrap() - 60.0).abs() < 1e-9);
        // exponential mean, weight 1/4
        p.note_stretch(0, 200.0);
        assert!((p.stretch_ms(0).unwrap() - 125.0).abs() < 1e-9);
        // non-positive times are ignored
        p.note_stretch(0, 0.0);
        p.note_copy_round(-1.0, 15);
        assert!((p.stretch_ms(0).unwrap() - 125.0).abs() < 1e-9);
        assert_eq!(p.copy_rate(), Some(0.4));
    }

    /// The TP=1 costs measured on .15 (no dense-resident): full copy round 61.8 ms for 16 tokens, stretches 151 / 252 /
    /// 483 / 520 ms per rung.
    fn measured() -> Price {
        let mut p = Price::default();
        p.note_copy_round(61.8, 15);
        for (g, ms) in [151.0, 252.0, 483.0, 520.0].into_iter().enumerate() { p.note_stretch(g, ms); }
        p
    }

    #[test]
    fn the_best_rate_rung_is_chosen_and_must_beat_copy_rounds() {
        let offer = |g: usize| chunk_width(g, 512, 4095) - 1; // 31, 159, 511, 511
        let p = measured();
        // rho = 16 / 61.8 = 0.259 tok/ms
        assert_eq!(priced_pick(&p, 0, 20, offer), (0, false), "a short copy: 21 / 151 = 0.14 < rho");
        assert_eq!(priced_pick(&p, 0, 31, offer), (0, false), "32 / 151 = 0.21 < rho");
        assert_eq!(priced_pick(&p, 0, 64, offer), (1, false), "65 / 252 = 0.258 < 0.259");
        assert_eq!(priced_pick(&p, 0, 70, offer), (1, true), "71 / 252 = 0.28: the narrower 160-row pass, not 512");
        assert_eq!(priced_pick(&p, 0, 130, offer), (1, true), "131 / 252 beats 131 / 483");
        assert_eq!(priced_pick(&p, 0, 4000, offer), (2, true), "a file return: 512 / 483 beats 160 / 252");
        assert_eq!(priced_pick(&p, 3, 10, offer), (3, false), "never below the lane's own rung");
        // nothing measured: the gated behaviour (the lane rung, taken)
        assert_eq!(priced_pick(&Price::default(), 1, 5, offer), (1, true));
        // only copy rounds measured: every rung costs the prior (4 copy rounds = 247 ms) -> the first rung covering want
        let mut q = Price::default();
        q.note_copy_round(61.8, 15);
        assert_eq!(priced_pick(&q, 0, 100, offer), (1, true));
        assert_eq!(priced_pick(&q, 0, 40, offer), (1, false), "41 / 247 < rho");
    }

    #[test]
    fn expected_want_is_the_longer_of_run_and_history() {
        assert_eq!(expected_want(20, 0.0), 20);
        assert_eq!(expected_want(20, 300.4), 300);
        assert_eq!(expected_want(200, 10.0), 200);
    }
}
