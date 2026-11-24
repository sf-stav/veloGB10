//! The DSpark draft-length policy (plan §2.6, T0 §6, X-DI §4.3; `--dsv41-draft-len auto|0..5`).
//!
//! A round with k drafts per lane emits `E(k) = 1 + S_1 + ... + S_k` tokens per lane, where `S_i` is the chance that
//! draft i is accepted, i.e. that drafts 1..i ALL survive the verify (a prefix survival), and costs
//! `T(lanes, k) = verify(lanes, k) + drafter + host`. `auto` picks the k that maximises `lanes * E(k) / T(lanes, k)`.
//!
//! Two rules decide `auto` (`--dsv41-spec-rule`, leg PS1):
//! * **`conf` (the default): the per-round rule of the baseline TensorFold EXL3 recipe** (RX4 F.2: their lane core's
//!   `allocate`, `CORE2/lanes/allocate.zig:9-49`; RX5 §4 confirms it on the live server). The drafter runs its full
//!   depth (5 drafts, as every served round of theirs does: RX5 §4, 115 kernels in 98-99 % of rounds), its confidence
//!   head gives each draft a logit z_i, the draft's landing chance is `sigmoid(z_i + offset)` (offset 0 in their TP2
//!   binary, 0.19 in their TP4 recipe: `--dsv41-spec-conf-offset`), a draft's chance of being accepted is the prefix
//!   product of the landing chances, and the round verifies the c drafts that maximise
//!   `(1 + sum of the first c chances) / (T(1 + c rows) + overhead)`. Their table is a fixed one; ours is MEASURED on
//!   this engine (boot calibration per lane count and draft count, cached per binary + pack + TP size, then corrected
//!   online per shape), and the overhead is learned as theirs (`max(0, round wall - table)`, first value stored, then
//!   EMA 0.2; their 8.0 ms before any observation: `--dsv41-spec-overhead-ms`). Their cap of 16 verify rows per round
//!   (`max_rows`) bounds c (`--dsv41-spec-max-rows`). Several lanes: ONE c for every lane (our allocation, kept: their
//!   multi-stream `allocateRows` is RX4 gap X6, not closed by RX5), the c maximising the sum over lanes of
//!   `1 + chances` per unit time. Acceptance statistics are not an input (as theirs); they are still counted for the
//!   receipts. **Leg PS3 (`--dsv41-spec-alloc lanes`, the default): with two or more lanes, each lane gets its own count**
//!   (their lane core's allocation, `alloc.rs`: one pending row per lane, then the likeliest next draft over all lanes
//!   while expected tokens / round ms(total rows) rises, at most `--dsv41-spec-max-rows` rows; a lane offers at most its
//!   last shared grant + 2 drafts; the chances are plain sigmoid prefixes, as both reference recipes use in shared
//!   rounds). `shared` keeps the one-c rule above. A lane alone is decided as before (`choose_conf`).
//! * **`stats`**: the k is chosen BEFORE the draft from the acceptance measured online: `S_i` = accepted / offered at
//!   position i (that count IS the prefix survival; leg X1 found the old formula multiplying these as if they were
//!   per-step survivals, so deep drafts were under-priced and k stuck at 2). A position with fewer than `MIN_OBS`
//!   offers is extrapolated from the last measured conditional survival `S_i / S_{i-1}`; and every `PROBE_EVERY`-th
//!   round offers all `MAX_K` drafts when the policy sits below it (a bounded probe: deep positions are re-measured,
//!   no starvation at k = 2).
//!
//! Exactness: the rule only chooses how many drafts are verified; the accept / residual rule is untouched. A choice
//! that reads the drafts themselves (the confidence of draft i depends on draft i - 1 through the Markov embedding)
//! is exact only when a draft's noise is never reused by a later round, so the draft draws are keyed by the lane's
//! round counter as well as the position (`91_ds_vocab.cuh`, leg PS1).
//!
//! `0` = no speculation (a bench lever). A pinned 1..5 = that length.
//!
//! **Confidence-scheduled variable length** (`--dsv41-draft-varlen on`, off by default as in DeepSeek's engine; not
//! wired into the TP serve loop): per lane, the confidence head's sigmoid(z_i), calibrated online per position
//! (observed acceptance / mean predicted), ranked across lanes (DeepSeek's `MTP_VARLEN` scheduler shape).

use anyhow::{Context, Result};
use std::collections::BTreeMap;

pub const MAX_K: usize = super::BLOCK;
/// Observations below this are noise, not a measurement.
pub const MIN_OBS: f64 = 8.0;
/// EWMA weight of a new round-cost sample.
pub const EWMA: f64 = 0.1;
/// A different length must beat the current one by this factor to switch (hysteresis; the `stats` rule).
pub const SWITCH_MARGIN: f64 = 1.03;
/// The conditional survival assumed for a position before any position was measured (`stats` rule).
pub const PRIOR_Q: f64 = 0.6;
/// `stats` rule: every this many decisions below `MAX_K`, one round offers `MAX_K` drafts (re-measures deep positions).
pub const PROBE_EVERY: u64 = 16;
/// `conf` rule: the overhead EMA weight after the first observation (theirs: `depth.Rule.observeOverhead`, 0.2).
pub const OVERHEAD_EMA: f64 = 0.2;
/// `conf` rule defaults (theirs): the overhead before any observation, the verify-row cap of a round.
pub const OVERHEAD0_MS: f64 = 8.0;
pub const MAX_ROWS: usize = 16;
/// Leg PS3: the most rows one round can carry on this engine (`tp::frame::MAX_ROWS`: 4 lanes x 6), the default row cap at
/// TP4 and above (their TP4 recipe's 48 is above it: every draft of every lane fits), and the allocation modes.
pub const ROUND_ROWS_MAX: usize = 32;
pub const MAX_ROWS_TP2: usize = 24;
pub const MAX_ROWS_TP4: usize = 24;

/// Leg PS3 (exactness): how the verify count is cut from the priced rates along the drafts. Draft j's chance S_j is a
/// prefix product of sigmoids of confidences computed from the drafts BEFORE it (draft j's confidence reads draft j - 1
/// through the Markov embedding, never draft j), so a rule is distribution-exact iff its decision "verify draft j" reads
/// S_1..S_j and history only: a stopping rule (report "exactness of the draft-count rule").
/// * `Lookahead` (default for a lane alone and `shared`): decide draft j from S_1..S_j and PROJECTED later chances that read
///   no draft (S_j x m^i, m = `LOOKAHEAD_M`): verify it iff some longer count beats the current one on that projection.
/// * `First`: stop at the first draft that does not raise expected tokens per ms (the per-lane allocation's rule).
/// * `Scan`: the count with the best rate over all counts (PS1 as shipped, the reference's best prefix). It reads S_{j+1}
///   (a function of draft j's draw) to decide draft j: NOT distribution-exact for sampled drafts. Reachable only through
///   `--dsv41-spec-inexact-scan on`, a measurement switch (boot line + /metrics say so), never a default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Stop { First, Lookahead, Scan }

/// The projected conditional chance of a draft beyond the last known one (`Stop::Lookahead`); a constant (it reads nothing
/// of this round). The independent review (RVX) simulated m = 0.7 exact and within 0.5 % of the scan's tokens per ms.
pub const LOOKAHEAD_M: f64 = 0.7;

impl Stop {
    /// `--dsv41-spec-stop` (the exact rules only; the scan has its own switch).
    pub fn parse(s: &str) -> Result<Stop> {
        match s { "lookahead" => Ok(Stop::Lookahead), "first" => Ok(Stop::First), x => anyhow::bail!("--dsv41-spec-stop {x} (lookahead | first)") }
    }
    pub fn name(self) -> &'static str { match self { Stop::First => "first", Stop::Lookahead => "lookahead", Stop::Scan => "scan (inexact)" } }
}

/// The exact first-stop index of a rate sequence r[0..]: the last c before the first c + 1 that does not strictly raise
/// the rate; and the scan's (the first best).
pub fn stop_index(r: &[f64]) -> usize { let mut c = 0; while c + 1 < r.len() && r[c + 1] > r[c] { c += 1; } c }
pub fn scan_index(r: &[f64]) -> usize { let mut b = 0; for c in 1..r.len() { if r[c] > r[b] { b = c; } } b }

/// The exact look-ahead count: `ch[l]` = lane l's chances (S_1.. at index 0..; one c for every lane), `t[c]` = the priced
/// round with c drafts per lane (c = 0..=cmax, cmax <= every lane's chances). Draft j + 1 (index j) is verified iff draft j
/// was and some count c' > j beats count j, priced with the KNOWN chances S_1..S_{j+1} and the projection S_{j+1} m^i
/// beyond: the decision reads S_{j+1} (from draft j) and no later chance.
pub fn lookahead_index(ch: &[Vec<f64>], t: &[f64], m: f64) -> usize {
    let cmax = t.len().saturating_sub(1);
    let mut c = 0;
    for j in 0..cmax {
        let val = |l: usize, i: usize| -> f64 { if i <= j { ch[l][i] } else { ch[l][j] * m.powi((i - j) as i32) } };
        let e = |cc: usize| -> f64 { (0..ch.len()).map(|l| 1.0 + (0..cc).map(|i| val(l, i)).sum::<f64>()).sum() };
        let rj = e(j) / t[j];
        if !((j + 1)..=cmax).any(|cc| e(cc) / t[cc] > rj) { break; }
        c = j + 1;
    }
    c
}

/// How a shared round's verify rows are split between its lanes (`--dsv41-spec-alloc`, leg PS3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Alloc { Lanes, Shared }

impl Alloc {
    pub fn parse(s: &str) -> Result<Alloc> {
        match s { "lanes" => Ok(Alloc::Lanes), "shared" => Ok(Alloc::Shared), x => anyhow::bail!("--dsv41-spec-alloc {x} (lanes | shared)") }
    }
    pub fn name(self) -> &'static str { match self { Alloc::Lanes => "lanes", Alloc::Shared => "shared" } }
}

/// Leg PS3: one per-lane decision (the receipt of a shared round): drafts verified per lane (the lanes' order as given),
/// the expected tokens (sum over lanes of 1 + the chances verified), the priced round, whether the reference's
/// scan-to-the-cap would have chosen otherwise, and how many lanes the grant + 2 rule capped.
#[derive(Clone, Debug, PartialEq)]
pub struct LaneChoice { pub c: Vec<usize>, pub expected: f64, pub ms: f64, pub scan_differs: bool, pub capped: usize,
                        /// leg PS4: the reference scan's counts and whether the decision was one draft from flipping
                        pub scan_c: Vec<usize>, pub sensitive: bool,
                        /// the exact rule's counts (= `c` unless the measurement scan is on)
                        pub exact_c: Vec<usize> }
/// `conf` rule: the confidence offset of their TP4 recipe (`conf_cal` {1, 0.19}); their TP2 binary uses 0.
pub const CONF_OFFSET_TP4: f64 = 0.19;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode { Off, Fixed(usize), Auto }

impl Mode {
    /// `--dsv41-draft-len` value.
    pub fn parse(s: &str) -> Result<Mode> {
        Ok(match s {
            "auto" => Mode::Auto,
            "0" => Mode::Off,
            k => { let k: usize = k.parse().context("--dsv41-draft-len")?; anyhow::ensure!((1..=MAX_K).contains(&k), "--dsv41-draft-len {k}"); Mode::Fixed(k) }
        })
    }
}

/// How `auto` decides (`--dsv41-spec-rule`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rule { Conf, Stats }

impl Rule {
    pub fn parse(s: &str) -> Result<Rule> {
        match s { "conf" => Ok(Rule::Conf), "stats" => Ok(Rule::Stats), x => anyhow::bail!("--dsv41-spec-rule {x} (conf | stats)") }
    }
    pub fn name(self) -> &'static str { match self { Rule::Conf => "conf", Rule::Stats => "stats" } }
}

/// Measured round costs (ms): the verify by total rows (`verify`, the older single-lane-count table) and by
/// (lanes, drafts per lane) (`vk`), the drafter by (lanes, k).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Costs {
    pub verify: BTreeMap<usize, f64>,
    pub vk: BTreeMap<(usize, usize), f64>,
    pub draft: BTreeMap<(usize, usize), f64>,
    pub host_ms: f64,
    /// leg PS2: every timed repetition of the (lanes, drafts) verify (the prior of the online model is fitted from them),
    /// and when the table was measured (unix seconds; a cached table keeps it)
    pub vk_reps: BTreeMap<(usize, usize), Vec<f64>>,
    pub measured_unix: u64,
}

impl Costs {
    /// Leg PS2: the boot prior of the online round-cost model, per lane count 1..=lanes: the verify repetitions of c drafts
    /// per lane plus the draft step at `depth` plus `extra_ms` (the per-round bookkeeping the boot calibration does not
    /// time), fitted as fixed + per-row cost (`roundcost::prior_from_reps`). A lane count without repetitions uses the
    /// medians (an older cache). None when a lane count has nothing measured.
    pub fn prior(&self, lanes: usize, depth: usize, extra_ms: f64) -> Option<Vec<[f64; super::roundcost::NC]>> {
        (1..=lanes).map(|l| {
            let d = self.draft_ms(l, depth);
            let reps: Vec<Vec<f64>> = (0..super::roundcost::NC).map(|c| {
                self.vk_reps.get(&(l, c)).filter(|v| !v.is_empty()).cloned()
                    .or_else(|| self.verify_lk(l, c).map(|v| vec![v])).unwrap_or_default()
                    .into_iter().map(|x| x + d).collect()
            }).collect();
            super::roundcost::prior_from_reps(&reps, extra_ms)
        }).collect()
    }

    /// Linear interpolation over measured row counts; beyond the ends, the nearest slope.
    pub fn verify_ms(&self, rows: usize) -> Option<f64> {
        if let Some(v) = self.verify.get(&rows) { return Some(*v); }
        let below = self.verify.range(..rows).next_back();
        let above = self.verify.range(rows..).next();
        match (below, above) {
            (Some((&a, &va)), Some((&b, &vb))) => Some(va + (vb - va) * (rows - a) as f64 / (b - a) as f64),
            (Some((&a, &va)), None) => {
                let prev = self.verify.range(..a).next_back();
                Some(match prev { Some((&p, &vp)) => va + (va - vp) / (a - p) as f64 * (rows - a) as f64, None => va })
            }
            (None, Some((_, &vb))) => Some(vb),
            (None, None) => None,
        }
    }
    /// The verify of `lanes` lanes with `k` drafts each: the (lanes, k) table when measured, else the rows table.
    pub fn verify_lk(&self, lanes: usize, k: usize) -> Option<f64> {
        self.vk.get(&(lanes, k)).copied().or_else(|| self.verify_ms(lanes * (k + 1)))
    }
    /// `verify_lk` made non-decreasing in k (the running maximum over 0..=k): one more verify row never costs less;
    /// a noisy calibration (X1: verify(4 rows) 46.6 > verify(5 rows) 45.7) must not make a longer draft look cheaper.
    pub fn verify_mono(&self, lanes: usize, k: usize) -> Option<f64> {
        let mut m: Option<f64> = None;
        for j in 0..=k { if let Some(v) = self.verify_lk(lanes, j) { m = Some(m.map_or(v, |x: f64| x.max(v))); } }
        m
    }
    pub fn draft_ms(&self, lanes: usize, k: usize) -> f64 {
        if k == 0 { return 0.0; }
        if let Some(v) = self.draft.get(&(lanes, k)) { return *v; }
        // the same k at the neighbouring lane counts (linear in lanes), then the same lanes at another k (linear in k)
        let same_k: Vec<(usize, f64)> = self.draft.iter().filter(|((_, kk), _)| *kk == k).map(|((l, _), v)| (*l, *v)).collect();
        let lo = same_k.iter().filter(|(l, _)| *l < lanes).last().copied();
        let hi = same_k.iter().find(|(l, _)| *l > lanes).copied();
        match (lo, hi) {
            (Some((a, va)), Some((b, vb))) => return va + (vb - va) * (lanes - a) as f64 / (b - a) as f64,
            (Some((_, v)), None) | (None, Some((_, v))) => return v,
            _ => {}
        }
        self.draft.iter().filter(|((l, _), _)| *l == lanes).map(|((_, kk), v)| v * k as f64 / *kk as f64).next()
            .or_else(|| self.draft.values().next().copied()).unwrap_or(0.0)
    }
    /// A round of `lanes` lanes verifying `k` drafts each after a draft step of `depth` drafts (`stats`: depth = k).
    pub fn round_ms_d(&self, lanes: usize, k: usize, depth: usize) -> Option<f64> { Some(self.verify_mono(lanes, k)? + self.draft_ms(lanes, depth) + self.host_ms) }
    pub fn round_ms(&self, lanes: usize, k: usize) -> Option<f64> { self.round_ms_d(lanes, k, k) }
    /// Fold one measured sample into the table.
    pub fn observe_verify(&mut self, rows: usize, ms: f64) { let e = self.verify.entry(rows).or_insert(ms); *e += EWMA * (ms - *e); }
    pub fn observe_vk(&mut self, lanes: usize, k: usize, ms: f64) { let e = self.vk.entry((lanes, k)).or_insert(ms); *e += EWMA * (ms - *e); }
    pub fn observe_draft(&mut self, lanes: usize, k: usize, ms: f64) { let e = self.draft.entry((lanes, k)).or_insert(ms); *e += EWMA * (ms - *e); }
}

/// The prefix survivals S_1..S_k from the per-position counts `hz[i] = (accepted, offered)` at position i + 1
/// (accepted at position i + 1 means drafts 1..i + 1 all survived). A position with fewer than `MIN_OBS` offers
/// carries the last measured conditional survival `S_i / S_{i-1}` (`PRIOR_Q` before any measurement); the result is
/// non-increasing (a deeper draft never lands more often than a shallower one).
pub fn prefix_survivals(hz: &[(f64, f64)], k: usize) -> Vec<f64> {
    let (mut prev, mut q) = (1.0f64, PRIOR_Q);
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let s = match hz.get(i) {
            Some(&(a, n)) if n >= MIN_OBS => {
                let s = (a / n).min(prev);
                if prev > 0.0 { q = s / prev; }
                s
            }
            _ => prev * q,
        };
        out.push(s);
        prev = s;
    }
    out
}

/// Expected tokens of a k-draft round: `1 + S_1 + ... + S_k` (prefix survivals; see `prefix_survivals`).
pub fn expected_tokens(hz: &[(f64, f64)], k: usize) -> f64 { 1.0 + prefix_survivals(hz, k).iter().sum::<f64>() }

/// The landing chances of one lane's drafts: the prefix products of `sigmoid(z_i + offset)` (theirs: `survivals`).
pub fn conf_chances(z: &[f32], offset: f64) -> Vec<f64> {
    let mut c = 1.0f64;
    z.iter().map(|&x| { c *= 1.0 / (1.0 + (-(x as f64 + offset)).exp()); c }).collect()
}

/// One `conf` decision (the receipt of a round).
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    /// drafts verified per lane
    pub c: usize,
    /// expected tokens of the round over its lanes (sum of `1 + chances`)
    pub expected: f64,
    /// the priced round time (table + overhead) of the chosen c
    pub ms: f64,
    /// leg PS3: the scan (the inexact best-count rule) would have chosen another c on this round
    pub scan_differs: bool,
    /// leg PS4: the scan's c, and whether the decision was one draft from flipping (a neighbouring c's rate within
    /// `alloc::SENSITIVE_GAP` of the chosen one): the rounds a count-rule bias can live in (PIT buckets)
    pub scan_c: usize,
    pub sensitive: bool,
}

#[derive(Clone, Debug)]
pub struct Policy {
    pub mode: Mode,
    pub varlen: bool,
    pub rule: Rule,
    pub costs: Costs,
    /// per position: (accepted, offered)
    pub hz: [(f64, f64); MAX_K],
    /// per position: (sum of sigmoid(confidence) of offered drafts, accepted) for the varlen calibration
    pub conf_cal: [(f64, f64); MAX_K],
    pub current: usize,
    /// `conf`: the landing-chance offset, the verify-row cap, the learned overhead per lane count (None = not yet
    /// observed: `overhead0` stands in)
    pub conf_offset: f64,
    pub max_rows: usize,
    pub overhead0: f64,
    pub overhead: BTreeMap<usize, f64>,
    /// `stats`: decisions made (the probe cadence) and probes taken
    pub decisions: u64,
    pub probes: u64,
    /// leg PS2: the online round-cost model the `conf` rule prices with once the boot calibration built it (fixed +
    /// per-row cost per lane count, refreshed from every served round; `roundcost`). None = the PS1 path (table + overhead).
    pub model: Option<super::roundcost::RoundCost>,
    /// leg PS3: how a shared round's rows are split, and the row cap as asked (before the engine's 24-row clamp)
    pub alloc: Alloc,
    pub max_rows_asked: usize,
    /// leg PS3: the stop rule (`--dsv41-spec-stop`; exact by default)
    pub stop: Stop,
    /// leg PS4, GATE ONLY (`--dsv41-spec-gate-jump-ms`): a synthetic fixed cost added to every priced round that verifies a
    /// draft (a concave step in the cost table: the review's hard case), so that the exact rule and the scan part often and
    /// the PIT gate can see a count rule that reads its drafts. 0 = off (served rounds are priced as measured).
    pub gate_jump_ms: f64,
}

impl Policy {
    pub fn new(mode: Mode, varlen: bool, costs: Costs) -> Policy { Policy::with_rule(mode, varlen, Rule::Stats, costs) }

    pub fn with_rule(mode: Mode, varlen: bool, rule: Rule, costs: Costs) -> Policy {
        let current = match mode { Mode::Off => 0, Mode::Fixed(k) => k, Mode::Auto => MAX_K };
        Policy { mode, varlen, rule, costs, hz: [(0.0, 0.0); MAX_K], conf_cal: [(0.0, 0.0); MAX_K], current,
                 conf_offset: 0.0, max_rows: MAX_ROWS, overhead0: OVERHEAD0_MS, overhead: BTreeMap::new(), decisions: 0, probes: 0, model: None,
                 alloc: Alloc::Shared, max_rows_asked: MAX_ROWS, stop: Stop::Lookahead, gate_jump_ms: 0.0 }
    }

    /// True when `auto` decides AFTER the draft step from the drafter's confidences (the drafter runs `MAX_K` drafts).
    pub fn decides_after_draft(&self) -> bool { self.mode == Mode::Auto && self.rule == Rule::Conf }

    /// Record one lane's round: `offered` drafts verified, `accepted` of them taken (a prefix).
    pub fn record(&mut self, offered: usize, accepted: usize) {
        for i in 0..offered.min(MAX_K) {
            self.hz[i].1 += 1.0;
            if i < accepted { self.hz[i].0 += 1.0; }
        }
    }
    /// Record the confidences of a lane's offered drafts and how many were accepted (varlen calibration).
    pub fn record_conf(&mut self, sig: &[f32], accepted: usize) {
        for (i, s) in sig.iter().enumerate().take(MAX_K) {
            self.conf_cal[i].0 += *s as f64;
            if i < accepted { self.conf_cal[i].1 += 1.0; }
        }
    }

    /// The k of the next round's DRAFT step for `lanes` lanes (the same for every lane). `conf` rule: `MAX_K` (the
    /// verify count is chosen after the draft by `choose_conf`). `stats` rule: the priced k, or `MAX_K` on a probe.
    pub fn choose(&mut self, lanes: usize) -> usize {
        match self.mode {
            Mode::Off => 0,
            Mode::Fixed(k) => k,
            Mode::Auto if self.rule == Rule::Conf => MAX_K,
            Mode::Auto => {
                let rate = |k: usize| -> Option<f64> { Some(lanes as f64 * expected_tokens(&self.hz, k) / self.costs.round_ms(lanes, k)?) };
                let Some(cur) = rate(self.current) else { return self.current };   // no costs yet: DeepSeek's default (5)
                let (mut best, mut bv) = (self.current, cur);
                for k in 0..=MAX_K {
                    if let Some(r) = rate(k) { if r > bv * SWITCH_MARGIN { best = k; bv = r; } }
                }
                self.current = best;
                self.decisions += 1;
                if best < MAX_K && self.decisions % PROBE_EVERY == 0 { self.probes += 1; return MAX_K; }
                best
            }
        }
    }

    /// The `conf` rule's overhead for `lanes` lanes (their `overheadMs(streams)`).
    pub fn overhead_ms(&self, lanes: usize) -> f64 { self.overhead.get(&lanes).copied().unwrap_or(self.overhead0) }

    /// The `conf` rule (theirs): `z[l]` = the confidence logits of lane l's `depth` drafts. Returns the drafts every lane
    /// verifies (one c for all lanes), or None when no cost is measured (the caller verifies all `depth`).
    pub fn choose_conf(&mut self, z: &[Vec<f32>], depth: usize) -> Option<Choice> {
        let lanes = z.len().max(1);
        let cmax = depth.min(MAX_K).min((self.max_rows / lanes).saturating_sub(1));
        let ch: Vec<Vec<f64>> = z.iter().map(|v| conf_chances(&v[..depth.min(v.len())], self.conf_offset)).collect();
        let o = self.overhead_ms(lanes);
        let mut te: Vec<(f64, f64)> = Vec::with_capacity(cmax + 1);
        for c in 0..=cmax {
            // leg PS2: the served-round model when it covers this lane count (its prices are whole served rounds: no
            // separate overhead); else PS1's boot table + learned overhead
            let t = match self.model.as_ref().and_then(|m| m.price(lanes, c)) {
                Some(t) => t,
                None => self.costs.round_ms_d(lanes, c, depth)? + o,
            } + if c > 0 { self.gate_jump_ms } else { 0.0 };
            let e: f64 = ch.iter().map(|v| 1.0 + v.iter().take(c).sum::<f64>()).sum();
            te.push((t, e));
        }
        // leg PS3: an exact rule (`Stop`: the look-ahead by default); the scan (PS1 as shipped) only on the measurement switch
        let r: Vec<f64> = te.iter().map(|(t, e)| e / t).collect();
        let cs = scan_index(&r);
        // the exact count (the look-ahead, or the first stop when asked) is computed in every mode: with the measurement scan
        // on, `scan_differs` still marks the rounds where the scan served a count the exact rule would not have (PIT buckets)
        let cx = if self.stop == Stop::First { stop_index(&r) } else { lookahead_index(&ch, &te.iter().map(|x| x.0).collect::<Vec<_>>(), LOOKAHEAD_M) };
        let c = if self.stop == Stop::Scan { cs } else { cx };
        let near = |a: usize, b: usize| ((r[a] - r[b]) / r[b]).abs() < super::alloc::SENSITIVE_GAP;
        let sensitive = (c + 1 < r.len() && near(c + 1, c)) || (c > 0 && near(c - 1, c));
        let ch = Choice { c, expected: te[c].1, ms: te[c].0, scan_differs: cs != cx, scan_c: cs, sensitive };
        self.current = ch.c;
        self.decisions += 1;
        Some(ch)
    }

    /// Leg PS3: the rows a shared round of `lanes` lanes may carry (their `max(plans.len, batch_rows)`, clamped to the
    /// engine's 24-row round).
    pub fn round_rows(&self, lanes: usize) -> usize { self.max_rows.min(ROUND_ROWS_MAX).max(lanes) }

    /// Leg PS3: the priced wall time of a round of `lanes` lanes verifying `d` drafts in total: the online model when it
    /// covers the lane count (whole served rounds), else PS1's boot table at d / lanes drafts per lane (linear between the
    /// even counts) + the learned overhead.
    pub fn price_total(&self, lanes: usize, d: usize, depth: usize) -> Option<f64> {
        let jump = if d > 0 { self.gate_jump_ms } else { 0.0 };
        if let Some(t) = self.model.as_ref().and_then(|m| m.price_d(lanes, d)) { return Some(t + jump); }
        let (c0, r) = (d / lanes.max(1), d % lanes.max(1));
        let t0 = self.costs.round_ms_d(lanes, c0, depth)?;
        let t = if r == 0 { t0 } else { t0 + (self.costs.round_ms_d(lanes, c0 + 1, depth)? - t0) * r as f64 / lanes as f64 };
        Some(t + self.overhead_ms(lanes) + jump)
    }

    /// Leg PS3, the per-lane rule (`--dsv41-spec-alloc lanes`) of a round of two or more lanes: `z[l]` = the confidence logits
    /// of lane l's `depth` drafts, `held[l]` = the drafts lane l offers (its last shared grant + 2, or all). Landing chances
    /// = prefix products of sigmoid(z) (no offset: both reference recipes' shared rounds use the plain sigmoid), allocated by
    /// `alloc::allocate` against `price_total`. None when the round cannot be priced (the caller verifies every draft).
    pub fn choose_lanes(&mut self, z: &[Vec<f32>], depth: usize, held: &[usize]) -> Option<LaneChoice> {
        let lanes = z.len();
        let depth = depth.min(MAX_K);
        let mut capped = 0;
        let probs: Vec<Vec<f64>> = z.iter().enumerate().map(|(i, v)| {
            let h = held.get(i).copied().unwrap_or(depth).min(depth).min(v.len());
            if h < depth.min(v.len()) { capped += 1; }
            conf_chances(&v[..h], 0.0)
        }).collect();
        let la = if self.stop == Stop::Lookahead { Some(LOOKAHEAD_M) } else { None };
        let a = super::alloc::allocate_with_lookahead(&probs, |d| self.price_total(lanes, d, depth), self.round_rows(lanes), la)?;
        self.decisions += 1;
        if self.stop == Stop::Scan {
            // `--dsv41-spec-stop scan` (A/B only): the reference's best prefix, priced and expected as such
            let total: usize = a.scan_counts.iter().sum();
            let e = lanes as f64 + a.scan_counts.iter().zip(&probs).map(|(&c, p)| p[..c].iter().sum::<f64>()).sum::<f64>();
            let ms = self.price_total(lanes, total, depth).unwrap_or(f64::NAN);
            let sensitive = a.margin < super::alloc::SENSITIVE_GAP;
            return Some(LaneChoice { c: a.scan_counts.clone(), expected: e, ms, scan_differs: a.scan_differs, capped, scan_c: a.scan_counts, sensitive, exact_c: a.counts });
        }
        Some(LaneChoice { c: a.counts.clone(), expected: a.expected, ms: a.ms, scan_differs: a.scan_differs, capped, scan_c: a.scan_counts, sensitive: a.margin < super::alloc::SENSITIVE_GAP, exact_c: a.counts })
    }

    /// Fold one measured `conf` round into the overhead (theirs: `max(0, wall - table)`, the first value stored, then
    /// EMA 0.2). Call with the table as it priced the round (before the round's own cost samples are folded in).
    pub fn observe_overhead(&mut self, lanes: usize, wall_ms: f64, table_ms: f64) {
        let x = (wall_ms - table_ms).max(0.0);
        match self.overhead.get_mut(&lanes) { Some(e) => *e += OVERHEAD_EMA * (x - *e), None => { self.overhead.insert(lanes, x); } }
    }

    /// Per-position calibration factor of the confidence head (observed acceptance / mean predicted), 1 until measured.
    fn cal(&self, i: usize) -> f64 {
        let (s, a) = self.conf_cal[i];
        let n = self.hz[i].1;
        if n >= MIN_OBS && s > 0.0 { (a / s).clamp(0.2, 5.0) } else { 1.0 }
    }

    /// Confidence-scheduled lengths: `sig[l][i]` = sigmoid(confidence) of lane l's draft i (the drafter ran all
    /// `kmax`); returns the drafts each lane verifies (a prefix of its chain).
    pub fn choose_varlen(&self, sig: &[[f32; MAX_K]], kmax: usize) -> Vec<usize> {
        let lanes = sig.len();
        if !self.varlen || lanes == 0 { return vec![kmax; lanes]; }
        // cumulative survival estimates, all lanes' drafts ranked
        let mut cand: Vec<(f64, usize, usize)> = Vec::new();
        for (l, s) in sig.iter().enumerate() {
            let mut c = 1.0f64;
            for i in 0..kmax { c *= (s[i] as f64 * self.cal(i)).min(1.0); cand.push((c, l, i)); }
        }
        cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let mut best = (0usize, f64::NEG_INFINITY);
        let mut e = lanes as f64;
        for take in 0..=cand.len() {
            if take > 0 { e += cand[take - 1].0; }
            let rows = lanes + take;
            let kk = (take + lanes - 1) / lanes.max(1);
            let Some(t) = self.costs.verify_ms(rows).map(|v| v + self.costs.draft_ms(lanes, kk.max(1).min(MAX_K)) + self.costs.host_ms) else { return vec![kmax; lanes] };
            if e / t > best.1 { best = (take, e / t); }
        }
        let mut k = vec![0usize; lanes];
        for &(_, l, i) in cand.iter().take(best.0) { k[l] = k[l].max(i + 1); }
        k
    }
}

/// `--mtp off` (also `0` / `false`, the older spellings) is plain decoding on every server: here that is no DSpark drafts,
/// the same as `--dsv41-draft-len 0` and winning over a pinned length. `on` / `auto` / absent leave `--dsv41-draft-len` as it is.
pub fn effective_len<'a>(len: &'a str, mtp: Option<&str>) -> &'a str {
    if matches!(mtp.map(|m| m.trim().to_ascii_lowercase()).as_deref(), Some("off" | "0" | "false")) { "0" } else { len }
}

/// The policy the serve loop builds at boot from `--dsv41-draft-len` / `--dsv41-draft-varlen` / `--dsv41-spec-rule` /
/// `--dsv41-spec-conf-offset` / `--dsv41-spec-overhead-ms` / `--dsv41-spec-max-rows` and the cached costs; `world` = the
/// TP size (the confidence offset defaults to their TP2 binary's 0 at 1-2 ranks and their TP4 recipe's 0.19 above).
pub fn from_flags(costs: Costs, world: usize) -> Result<Policy> {
    let get = |f: &str| crate::opts::lookup(f).and_then(crate::opts::get);
    let len = get("dsv41-draft-len").unwrap_or_else(|| "auto".into());
    let len = effective_len(&len, get("mtp").as_deref()).to_string();
    let varlen = get("dsv41-draft-varlen").map_or(false, |v| v == "on");
    let rule = Rule::parse(get("dsv41-spec-rule").as_deref().unwrap_or("conf"))?;
    let mut p = Policy::with_rule(Mode::parse(&len)?, varlen, rule, costs);
    p.conf_offset = match get("dsv41-spec-conf-offset") {
        Some(v) => v.trim().parse::<f64>().with_context(|| format!("--dsv41-spec-conf-offset {v}"))?,
        None => default_conf_offset(world),
    };
    if let Some(v) = get("dsv41-spec-overhead-ms") { p.overhead0 = v.trim().parse::<f64>().with_context(|| format!("--dsv41-spec-overhead-ms {v}"))?.max(0.0); }
    p.max_rows = default_max_rows(world);
    if let Some(v) = get("dsv41-spec-max-rows") {
        let mr = v.trim().parse::<usize>().with_context(|| format!("--dsv41-spec-max-rows {v}"))?;
        p.max_rows = if mr == 0 { default_max_rows(world) } else { mr };
    }
    // leg PS3 / P6-4: a cap above the engine's 32-row round is applied as 32 (the boot line and /metrics show both)
    p.max_rows_asked = p.max_rows;
    p.max_rows = p.max_rows.min(ROUND_ROWS_MAX);
    p.alloc = Alloc::parse(get("dsv41-spec-alloc").as_deref().unwrap_or("lanes"))?;
    p.stop = Stop::parse(get("dsv41-spec-stop").as_deref().unwrap_or("lookahead"))?;
    // leg PS3: the inexact scan, a measurement switch only (what exactness costs); never a default
    if get("dsv41-spec-inexact-scan").as_deref() == Some("on") { p.stop = Stop::Scan; }
    if let Some(v) = get("dsv41-spec-gate-jump-ms") { p.gate_jump_ms = v.trim().parse::<f64>().with_context(|| format!("--dsv41-spec-gate-jump-ms {v}"))?.max(0.0); }
    Ok(p)
}

/// The confidence offset when `--dsv41-spec-conf-offset` is not given: their TP2 binary's 0 (`draft_cal` 0) at 1-2 ranks,
/// their TP4 recipe's 0.19 at 4 ranks and more.
pub fn default_conf_offset(world: usize) -> f64 { if world >= 4 { CONF_OFFSET_TP4 } else { 0.0 } }

/// Leg PS3: the row cap when `--dsv41-spec-max-rows` is not given: their TP2 recipe's 16 at 1-2 ranks, 24 at 4 ranks and
/// more (their TP4 recipe's 48 is above this engine's 24-row round: every draft of every lane fits).
pub fn default_max_rows(world: usize) -> usize { if world >= 4 { MAX_ROWS_TP4 } else if world >= 2 { MAX_ROWS_TP2 } else { MAX_ROWS } }

/// The boot line of the policy (the rule and its constants), for the serve loop's boot report.
pub fn boot_line(p: &Policy) -> String {
    match (p.mode, p.rule) {
        (Mode::Auto, Rule::Conf) => format!("[dsv41 spec] draft-length rule conf (the baseline recipe's per-round allocation): the drafter runs {MAX_K} drafts, \
            {}landing chances = prefix products of sigmoid(confidence + {:.2}), the round verifies drafts while each \
            pays ({}), priced from the boot calibration fitted as fixed + per-row cost ({:.1} ms bookkeeping \
            added until served rounds replace it) and then refreshed from every served round's own time (leg PS2), at most {} verify rows{}; {}",
            if p.alloc == Alloc::Lanes { "a lane alone: " } else { "each lane's (one c for all lanes, the sum over lanes) " }, p.conf_offset,
            match p.stop {
                Stop::Lookahead => format!("exact look-ahead: a draft is verified iff a longer count beats the current one on the known chances and a projection x {LOOKAHEAD_M} per later draft that reads no draft"),
                Stop::First => "exact first stop".to_string(),
                Stop::Scan => "!! --dsv41-spec-inexact-scan on: the best count over all counts, NOT distribution-exact for sampled requests: a measurement build only".to_string(),
            },
            p.overhead0, p.max_rows,
            if p.max_rows_asked != p.max_rows { format!(" (asked {}: a round carries at most {ROUND_ROWS_MAX} rows on this engine, capped)", p.max_rows_asked) } else { String::new() },
            match p.alloc {
                Alloc::Lanes => format!("two or more lanes (leg PS3, --dsv41-spec-alloc lanes): EACH LANE ITS OWN COUNT, the baseline recipe's lane allocation: \
                    one pending row per lane, then the likeliest next draft over all lanes (chances = prefix products of sigmoid(confidence), ties to the lower lane) \
                    while expected tokens / round ms(lanes, total rows) rises, at most {} rows; a lane offers at most its last shared count + {}", p.max_rows, super::alloc::DRAFT_SLACK),
                Alloc::Shared => "two or more lanes: one c for every lane (--dsv41-spec-alloc shared)".to_string(),
            }),
        (Mode::Auto, Rule::Stats) => format!("[dsv41 spec] draft-length rule stats: k chosen before the draft from the acceptance by position \
            (prefix survivals) and the measured round costs; every {PROBE_EVERY}th round below {MAX_K} offers {MAX_K} drafts to re-measure deep positions"),
        (Mode::Fixed(k), _) => format!("[dsv41 spec] draft length pinned at {k} (--dsv41-draft-len {k})"),
        (Mode::Off, _) => "[dsv41 spec] draft length 0: speculation off".to_string(),
    }
}

/// Measure round costs: `run(rows)` executes one verify round of `rows` total rows and returns its ms; each shape
/// is run `reps` times after one warm-up and its median kept (the boot calibration of the TP forward).
pub fn calibrate_verify(shapes: &[usize], reps: usize, mut run: impl FnMut(usize) -> Result<f64>) -> Result<Costs> {
    let mut c = Costs::default();
    for &rows in shapes {
        run(rows)?;
        let mut v: Vec<f64> = (0..reps.max(1)).map(|_| run(rows)).collect::<Result<_>>()?;
        v.sort_by(|a, b| a.total_cmp(b));
        c.verify.insert(rows, v[v.len() / 2]);
    }
    Ok(c)
}

// ------------------------------------------------------------------------------------------------ the calibration cache

/// `<binary dir>/dsv41_spec_calib/<pack basename>.json`.
pub fn calib_path(pack_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let base = pack_dir.file_name()?.to_string_lossy().replace(['/', ' '], "_");
    Some(exe.parent()?.join("dsv41_spec_calib").join(format!("{base}.json")))
}

/// The stamp a cached table must carry: kernel build id, the binary's mtime, the pack's census fingerprint, TP size, and
/// (leg PS2: the stale-cache hazard of P1M section 6) the lane count, the node list (topology) and the digest of every
/// set Spmd option (any flag that rides to the nodes can change a round's cost: graphs, PDL, dense grouping, prefetch,
/// the spec rule's own flags...). The table it guards is only the online model's starting point: served rounds replace it.
pub fn calib_stamp(pack_fingerprint: &str, tp: usize, lanes: usize) -> String {
    calib_stamp_with_dual(pack_fingerprint, tp, lanes, false)
}

pub fn calib_stamp_with_dual(pack_fingerprint: &str, tp: usize, lanes: usize, dual_rail: bool) -> String {
    let mtime = std::env::current_exe().ok().and_then(|e| std::fs::metadata(e).ok()).and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
    let nodes = crate::opts::lookup("nodes").and_then(crate::opts::get).unwrap_or_default();
    let spmd = crate::opts::spmd_digest().map_or("none".to_string(), |d| format!("{d:08x}"));
    let dual = if dual_rail { ";dual=1" } else { "" };
    format!("kernel={};bin_mtime={mtime};pack={pack_fingerprint};tp={tp};lanes={lanes};nodes={nodes};spmd={spmd}{dual}", env!("KERNEL_BUILD_ID"))
}

/// A pack's fingerprint for the stamp: FNV-1a over its config.json, quantization_config.json and safetensors index
/// (another pack, or the same name re-quantised, gets another fingerprint).
pub fn pack_fingerprint(dir: &std::path::Path) -> String {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for f in ["config.json", "quantization_config.json", "model.safetensors.index.json"] {
        if let Ok(b) = std::fs::read(dir.join(f)) { for x in b { h = (h ^ x as u64).wrapping_mul(0x0000_0100_0000_01B3); } }
    }
    format!("{}:{h:016x}", dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default())
}

pub fn write_calib(path: &std::path::Path, stamp: &str, c: &Costs) -> Result<()> {
    if let Some(d) = path.parent() { std::fs::create_dir_all(d)?; }
    let v = serde_json::json!({
        "fmt": 3, "stamp": stamp, "host_ms": c.host_ms, "measured_unix": c.measured_unix,
        "verify": c.verify.iter().map(|(r, ms)| serde_json::json!([r, ms])).collect::<Vec<_>>(),
        "vk": c.vk.iter().map(|((l, k), ms)| serde_json::json!([l, k, ms])).collect::<Vec<_>>(),
        "vk_reps": c.vk_reps.iter().map(|((l, k), v)| serde_json::json!([l, k, v])).collect::<Vec<_>>(),
        "draft": c.draft.iter().map(|((l, k), ms)| serde_json::json!([l, k, ms])).collect::<Vec<_>>(),
    });
    std::fs::write(path, serde_json::to_string_pretty(&v)?)?;
    Ok(())
}

/// The cached table when its stamp equals `stamp` (a rebuild, another pack, TP size, lane count, node list or Spmd flag
/// set re-measures).
pub fn read_calib(path: &std::path::Path, stamp: &str) -> Option<Costs> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    // fmt 3 (leg PS2) adds every repetition and the measuring time; an older cache re-measures
    if v.get("fmt")?.as_u64()? != 3 || v.get("stamp")?.as_str()? != stamp { return None; }
    let mut c = Costs { host_ms: v.get("host_ms")?.as_f64()?, measured_unix: v.get("measured_unix")?.as_u64()?, ..Default::default() };
    for e in v.get("verify")?.as_array()? { c.verify.insert(e.get(0)?.as_u64()? as usize, e.get(1)?.as_f64()?); }
    for e in v.get("vk")?.as_array()? { c.vk.insert((e.get(0)?.as_u64()? as usize, e.get(1)?.as_u64()? as usize), e.get(2)?.as_f64()?); }
    for e in v.get("vk_reps")?.as_array()? {
        let reps: Option<Vec<f64>> = e.get(2)?.as_array()?.iter().map(|x| x.as_f64()).collect();
        c.vk_reps.insert((e.get(0)?.as_u64()? as usize, e.get(1)?.as_u64()? as usize), reps?);
    }
    for e in v.get("draft")?.as_array()? { c.draft.insert((e.get(0)?.as_u64()? as usize, e.get(1)?.as_u64()? as usize), e.get(2)?.as_f64()?); }
    Some(c)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn costs() -> Costs {
        // T0 §5.3-shaped: a verify row is ~8-15 % of a 1-row round; the drafter ~2.5 ms
        let mut c = Costs { host_ms: 0.3, ..Default::default() };
        for (r, ms) in [(1, 16.0), (2, 17.2), (3, 18.4), (4, 19.6), (5, 20.8), (6, 22.0), (12, 29.0), (24, 43.0)] { c.verify.insert(r, ms); }
        for k in 1..=5 { for l in [1, 2, 4] { c.draft.insert((l, k), 2.0 + 0.1 * k as f64 + 0.3 * l as f64); } }
        c
    }
    #[test]
    fn dsv41_ds_policy_modes_parse() {
        assert_eq!(Mode::parse("auto").unwrap(), Mode::Auto);
        assert_eq!(Mode::parse("0").unwrap(), Mode::Off);
        // UXF F5: --mtp off is --dsv41-draft-len 0, whatever length was pinned; on / auto / unset change nothing
        for off in ["off", "OFF", "0", "false"] { assert_eq!(effective_len("auto", Some(off)), "0"); assert_eq!(effective_len("3", Some(off)), "0"); }
        for keep in [None, Some("auto"), Some("on"), Some("1"), Some("true")] { assert_eq!(effective_len("auto", keep), "auto"); assert_eq!(effective_len("3", keep), "3"); }
        assert_eq!(Mode::parse(effective_len("4", Some("off"))).unwrap(), Mode::Off);
        assert_eq!(Mode::parse("3").unwrap(), Mode::Fixed(3));
        assert!(Mode::parse("6").is_err());
    }
    #[test]
    fn dsv41_opt1_choose_lanes_uses_plain_sigmoid_while_choose_conf_uses_offset() {
        let mut c = Costs::default();
        for (d, ms) in [(0, 32.0), (1, 40.0), (2, 48.0), (3, 56.0), (4, 64.0), (5, 72.0)] {
            c.vk.insert((1, d), 20.0 + 2.6 * d as f64);
            c.vk.insert((2, d), ms);
        }
        let mut p_off = Policy::with_rule(Mode::Auto, false, Rule::Conf, c.clone());
        p_off.conf_offset = 0.0;
        let mut p_tp4 = Policy::with_rule(Mode::Auto, false, Rule::Conf, c);
        p_tp4.conf_offset = CONF_OFFSET_TP4;
        let z2 = vec![vec![0.1f32, -0.2, -0.5, -0.8, -1.2], vec![0.0, -0.3, -0.6, -0.9, -1.4]];
        let a0 = p_off.choose_lanes(&z2, 5, &[5, 5]).unwrap();
        let a4 = p_tp4.choose_lanes(&z2, 5, &[5, 5]).unwrap();
        assert_eq!(a0.c, a4.c, "multi-lane choose_lanes must ignore conf_offset");
        assert!((a0.expected - a4.expected).abs() < 1e-12);
        let z1 = vec![vec![0.0f32; 5]];
        let s0 = p_off.choose_conf(&z1, 5).unwrap();
        let s4 = p_tp4.choose_conf(&z1, 5).unwrap();
        assert!(s4.expected > s0.expected || s4.c >= s0.c, "1-lane choose_conf still applies conf_offset");
        assert_ne!(calib_stamp_with_dual("pk", 2, 4, false), calib_stamp_with_dual("pk", 2, 4, true));
    }
    #[test]
    fn dsv41_ds_policy_prices_length_from_costs_and_survival() {
        let mut p = Policy::new(Mode::Auto, false, costs());
        // high acceptance (code): every draft survives -> the longest length pays
        for _ in 0..100 { p.record(5, 5); }
        assert_eq!(p.choose(1), 5);
        // low acceptance (prose, survival ~0.3): short drafts pay
        let mut q = Policy::new(Mode::Auto, false, costs());
        for t in 0..100 { q.record(5, if t % 10 < 3 { 1 } else { 0 }); }
        let k = q.choose(1);
        assert!(k <= 2, "low survival chose {k}");
        // no costs yet: DeepSeek's fixed default
        let mut r = Policy::new(Mode::Auto, false, Costs::default());
        assert_eq!(r.choose(1), 5);
        assert_eq!(Policy::new(Mode::Fixed(3), false, Costs::default()).choose(4), 3);
        assert_eq!(Policy::new(Mode::Off, false, costs()).choose(4), 0);
    }
    #[test]
    fn dsv41_ds_expected_tokens_carries_the_last_conditional_survival() {
        // positions 1-2 measured (prefix survivals .9, .4), 3-4 never offered: they carry the last conditional .4 / .9
        let hz = [(90.0, 100.0), (40.0, 100.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
        let e = expected_tokens(&hz, 4);
        let q = 0.4 / 0.9;
        assert!((e - (1.0 + 0.9 + 0.4 + 0.4 * q + 0.4 * q * q)).abs() < 1e-9, "{e}");
        // nothing measured: the prior conditional survival
        let none = [(0.0, 0.0); MAX_K];
        assert!((expected_tokens(&none, 2) - (1.0 + PRIOR_Q + PRIOR_Q * PRIOR_Q)).abs() < 1e-12);
        // a noisy deeper position never lands more often than a shallower one
        let noisy = [(50.0, 100.0), (30.0, 40.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
        assert!(prefix_survivals(&noisy, 2)[1] <= 0.5 + 1e-12);
    }
    /// DoD 1 (leg PS1): a known acceptance process (per-step conditional acceptance q_i, the verify stops at the first
    /// rejection) recorded through `Policy::record`; `expected_tokens` must equal the closed form 1 + sum_i prod_{j<=i} q_j.
    #[test]
    fn dsv41_ps1_expected_tokens_matches_the_closed_form_of_a_known_process() {
        let q = [0.9f64, 0.8, 0.7, 0.6, 0.5];
        let closed = |k: usize| 1.0 + (1..=k).map(|i| q[..i].iter().product::<f64>()).sum::<f64>();
        // (a) exact counts: of N rounds offering 5, a fraction prod(q_1..q_a)(1 - q_{a+1}) accepts exactly a
        let mut p = Policy::new(Mode::Auto, false, Costs::default());
        let n = 100_000.0f64;
        for a in 0..=5usize {
            let pa = q[..a].iter().product::<f64>() * if a < 5 { 1.0 - q[a] } else { 1.0 };
            for _ in 0..(n * pa).round() as usize { p.record(5, a); }
        }
        for k in 0..=5 { assert!((expected_tokens(&p.hz, k) - closed(k)).abs() < 1e-3, "k {k}: {} vs {}", expected_tokens(&p.hz, k), closed(k)); }
        // the formula X1 found (counts multiplied as if they were per-step survivals) is far off at depth
        let old = |k: usize| { let (mut c, mut a) = (1.0, 1.0); for i in 0..k { c *= p.hz[i].0 / p.hz[i].1; a += c; } a };
        assert!(closed(5) - old(5) > 0.5, "old {} closed {}", old(5), closed(5));
        // (b) a simulated process (fixed-seed LCG), the same estimate within sampling noise
        let mut s = Policy::new(Mode::Auto, false, Costs::default());
        let mut x: u64 = 0x2545_F491_4F6C_DD1D;
        let mut u = || { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (x >> 11) as f64 / (1u64 << 53) as f64 };
        for _ in 0..50_000 { let mut a = 0; while a < 5 && u() < q[a] { a += 1; } s.record(5, a); }
        for k in 0..=5 { assert!((expected_tokens(&s.hz, k) - closed(k)).abs() < 0.03, "k {k}: {} vs {}", expected_tokens(&s.hz, k), closed(k)); }
    }
    /// X1's boot E counters (k = 5 pinned, code: prefix survivals .834 / .628 / .464 / .342 / .283) on X1's measured TP2
    /// round times (3 verify rows 43.6 ms, 6 rows 57.06 ms, linear between, 1 row 29.3 ms; drafter included): the old
    /// formula picks k = 2, the corrected one keeps 5 (measured: k = 2 59.4 tok/s, k = 5 67.7).
    #[test]
    fn dsv41_ps1_x1_counters_pick_the_deep_draft() {
        let mut c = Costs::default();
        for (k, ms) in [(0, 29.3), (1, 38.6), (2, 43.6), (3, 48.1), (4, 52.6), (5, 57.06)] { c.vk.insert((1, k), ms); }
        let mut p = Policy::new(Mode::Auto, false, c.clone());
        let s = [0.834f64, 0.628, 0.464, 0.342, 0.283];
        for (i, x) in s.iter().enumerate() { p.hz[i] = (x * 1000.0, 1000.0); }
        assert_eq!(p.choose(1), 5);
        // the old formula's pick from the same start (current 5, the same hysteresis): 2
        let old_e = |k: usize| { let (mut ch, mut a) = (1.0, 1.0); for x in &s[..k] { ch *= x; a += ch; } a };
        let (mut best, mut bv) = (5usize, old_e(5) / c.round_ms(1, 5).unwrap());
        for k in 0..=5 { let r = old_e(k) / c.round_ms(1, k).unwrap(); if r > bv * SWITCH_MARGIN { best = k; bv = r; } }
        assert_eq!(best, 2);
    }
    /// DoD 1: positions beyond the current k are re-measured (the bounded probe): no starvation at k = 2.
    #[test]
    fn dsv41_ps1_stats_probe_re_measures_deep_positions() {
        let mut c = Costs::default();
        for (k, ms) in [(0, 20.0), (1, 21.0), (2, 22.0), (3, 40.0), (4, 60.0), (5, 80.0)] { c.vk.insert((1, k), ms); }
        let mut p = Policy::new(Mode::Auto, false, c);
        let (mut deep, mut at2) = (0, 0);
        for _ in 0..(PROBE_EVERY * 4) {
            let k = p.choose(1);
            if k == MAX_K { deep += 1; } else if k == 2 { at2 += 1; }
            p.record(k, k);   // every draft accepted
        }
        assert!(at2 > 0, "the costs make k = 2 the choice");
        assert_eq!(deep as u64, p.probes);
        assert!(deep >= 3 && deep <= 5, "{deep} probes in {} decisions", PROBE_EVERY * 4);
        for i in 2..MAX_K { assert!(p.hz[i].1 >= 3.0, "position {} offered {}", i + 1, p.hz[i].1); }
    }
    /// The `conf` rule on their numbers (RX4 F.2): table 30.0 / 35.2 ms for 1 / 2 rows, overhead learned to 0: one draft
    /// pays when its chance exceeds 35.2 / 30 - 1 = 0.173.
    #[test]
    fn dsv41_ps1_conf_rule_reproduces_their_threshold() {
        let mut c = Costs::default();
        for (k, ms) in [(0, 30.0), (1, 35.2), (2, 39.9), (3, 44.9), (4, 48.6), (5, 52.1)] { c.vk.insert((1, k), ms); }
        let mut p = Policy::with_rule(Mode::Auto, false, Rule::Conf, c);
        p.overhead.insert(1, 0.0);
        let logit = |pr: f64| (pr / (1.0 - pr)).ln() as f32;
        let one = |p: &mut Policy, pr: f64| p.choose_conf(&[vec![logit(pr), -30.0, -30.0, -30.0, -30.0]], 5).unwrap().c;
        assert_eq!(one(&mut p, 0.17), 0);
        assert_eq!(one(&mut p, 0.18), 1);
        // all drafts sure: the longest; all hopeless: none
        assert_eq!(p.choose_conf(&[vec![30.0; 5]], 5).unwrap().c, 5);
        assert_eq!(p.choose_conf(&[vec![-30.0; 5]], 5).unwrap().c, 0);
        // the expected tokens of the receipt = 1 + the chances taken
        let ch = p.choose_conf(&[vec![30.0; 5]], 5).unwrap();
        assert!((ch.expected - 6.0).abs() < 1e-9 && (ch.ms - 52.1).abs() < 1e-9, "{ch:?}");
        // the offset raises every chance (their TP4 recipe: + 0.19)
        assert!(conf_chances(&[0.0], CONF_OFFSET_TP4)[0] > 0.547 && conf_chances(&[0.0], 0.0)[0] == 0.5);
        // the decision moves current, the stats-time draft step stays at MAX_K
        assert!(p.decides_after_draft());
        assert_eq!(p.choose(1), MAX_K);
        // no measured costs: no decision (the caller verifies every draft)
        let mut q = Policy::with_rule(Mode::Auto, false, Rule::Conf, Costs::default());
        assert_eq!(q.choose_conf(&[vec![0.0; 5]], 5), None);
    }
    /// The `conf` rule at several lanes: one c for every lane, the verify-row cap (theirs: 16) bounds it, the overhead is
    /// learned per lane count (first value stored, then EMA 0.2), and a non-monotone table never makes a longer draft cheaper.
    #[test]
    fn dsv41_ps1_conf_rule_lanes_cap_and_overhead() {
        let mut c = Costs::default();
        for l in 1..=4usize { for k in 0..=5usize { c.vk.insert((l, k), 20.0 + 4.0 * l as f64 * k as f64); } c.draft.insert((l, 5), 3.5); }
        let mut p = Policy::with_rule(Mode::Auto, false, Rule::Conf, c);
        let sure = vec![vec![30.0f32; 5]; 4];
        assert_eq!(p.choose_conf(&sure, 5).unwrap().c, 3, "4 lanes x (3 + 1) = 16 rows: the cap");
        assert_eq!(p.choose_conf(&sure[..3], 5).unwrap().c, 4, "3 lanes x 5 = 15 rows");
        // a confident lane and a hopeless lane share one c
        let mixed = vec![vec![30.0f32; 5], vec![-30.0f32; 5]];
        let ch = p.choose_conf(&mixed, 5).unwrap();
        assert!(ch.c >= 1 && (ch.expected - (2.0 + ch.c as f64)).abs() < 1e-9, "{ch:?}");
        // overhead: their 8 ms until observed, then the first observation, then EMA 0.2
        assert_eq!(p.overhead_ms(2), OVERHEAD0_MS);
        p.observe_overhead(2, 30.0, 25.0);
        assert_eq!(p.overhead_ms(2), 5.0);
        p.observe_overhead(2, 20.0, 25.0);
        assert!((p.overhead_ms(2) - 4.0).abs() < 1e-12);
        assert_eq!(p.overhead_ms(1), OVERHEAD0_MS);
        // monotone verify: a cheaper deeper entry is priced at the shallower maximum
        let mut m = Costs::default();
        for (k, ms) in [(0, 30.0), (1, 34.0), (2, 38.0), (3, 46.6), (4, 45.7), (5, 49.0)] { m.vk.insert((1, k), ms); }
        assert_eq!(m.verify_mono(1, 4), Some(46.6));
        assert_eq!(m.verify_mono(1, 5), Some(49.0));
        // draft costs at an unmeasured lane count interpolate at the same depth
        let mut d = Costs::default();
        d.draft.insert((1, 5), 4.0);
        d.draft.insert((4, 5), 7.0);
        assert!((d.draft_ms(2, 5) - 5.0).abs() < 1e-12);
    }
    /// Leg PS2: P1M's cached first-boot table (1 lane: 30.5 35.8 42.3 42.3 46.1 49.5 ms) priced c = 2 and c = 3 the same, so
    /// no confidence could ever choose 2 (E(3) > E(2) at an equal price). Fitted as fixed + per-row cost (the conf rule's
    /// model), the same measurements price every count strictly above the one before, and a flat landing chance between
    /// the 1 / 2 and 2 / 3 thresholds chooses 2.
    #[test]
    fn dsv41_ps2_a_tied_boot_table_no_longer_starves_a_draft_count() {
        let tab = [30.5, 35.8, 42.3, 42.3, 46.1, 49.5];
        let mut c = Costs::default();
        for (k, ms) in tab.iter().enumerate() { c.vk.insert((1, k), *ms); }
        let logit = |pr: f64| (pr / (1.0 - pr)).ln() as f32;
        let picks = |p: &mut Policy| -> Vec<usize> { (1..100).map(|i| p.choose_conf(&[vec![logit(i as f64 / 100.0); 5]], 5).unwrap().c).collect() };
        let mut old = Policy::with_rule(Mode::Auto, false, Rule::Conf, c.clone());
        old.overhead.insert(1, 0.0);
        assert!(!picks(&mut old).contains(&2), "the tied table never chooses 2");
        // the model from the same numbers (one repetition each, no bookkeeping added): strictly increasing, 2 is reachable
        let prior = c.prior(1, 5, 0.0).unwrap();
        assert!(prior[0].windows(2).all(|w| w[1] > w[0]), "{:?}", prior[0]);
        let mut new = Policy::with_rule(Mode::Auto, false, Rule::Conf, c);
        new.model = Some(super::super::roundcost::RoundCost::new(prior, super::super::roundcost::Source::Measured, 0));
        let got = picks(&mut new);
        for k in 0..=5 { assert!(got.contains(&k), "c = {k} never chosen over the chance sweep: {got:?}"); }
        // the model's prices, not the boot table + overhead, are what the receipt carries
        let ch = new.choose_conf(&[vec![30.0; 5]], 5).unwrap();
        assert!((ch.ms - new.model.as_ref().unwrap().price(1, 5).unwrap()).abs() < 1e-9);
    }

    /// Leg PS3 on the real policy code (`choose_lanes` -> `alloc::allocate` against the online model): lanes with a known
    /// acceptance process each (lane l's draft j lands with probability q_l given the earlier ones; the confidence head is
    /// calibrated: z = logit(q_l)). A confident lane verifies more drafts than an uncertain one; the expected tokens the rule
    /// prices match the tokens a simulated verify emits (the same process); the round never exceeds the row cap; the model's
    /// price of the round is its total rows'; and against the one-c rule on the same rounds, the per-lane rule expects at
    /// least as many tokens per priced ms (it can always choose the one-c split).
    #[test]
    fn dsv41_ps3_lanes_rule_on_a_known_process() {
        use super::super::roundcost::{RoundCost, Source, NC};
        let logit = |p: f64| (p / (1.0 - p)).ln() as f32;
        // a round of l lanes and d total drafts costs 30 + 2 l + 3.5 (l + d) ms (their table's shape: ~+3.5 ms a row)
        let prior: Vec<[f64; NC]> = (1..=4).map(|l| std::array::from_fn(|c| 30.0 + 2.0 * l as f64 + 3.5 * (l * (c + 1)) as f64)).collect();
        let mk = |alloc: Alloc| { let mut p = Policy::with_rule(Mode::Auto, false, Rule::Conf, Costs::default()); p.alloc = alloc;
                                  p.model = Some(RoundCost::new(prior.clone(), Source::Measured, 0)); p };
        let mut lanes = mk(Alloc::Lanes);
        let mut shared = mk(Alloc::Shared);
        let mut x: u64 = 0xD1B5_4A32_D192_ED03;
        let mut u = || { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (x >> 11) as f64 / (1u64 << 53) as f64 };
        for (qs, n_l) in [(vec![0.92, 0.35], 2usize), (vec![0.92, 0.35, 0.92, 0.35], 4), (vec![0.8, 0.8, 0.5], 3)] {
            let z: Vec<Vec<f32>> = qs.iter().map(|&q| vec![logit(q); MAX_K]).collect();
            let held = vec![MAX_K; n_l];
            let ch = lanes.choose_lanes(&z, MAX_K, &held).unwrap();
            assert_eq!(ch.c.len(), n_l);
            assert!(n_l + ch.c.iter().sum::<usize>() <= lanes.round_rows(n_l));
            for i in 0..n_l { for j in 0..n_l { if qs[i] > qs[j] { assert!(ch.c[i] > ch.c[j], "{qs:?} -> {:?}", ch.c); } } }
            assert!((ch.ms - lanes.model.as_ref().unwrap().price_d(n_l, ch.c.iter().sum()).unwrap()).abs() < 1e-9);
            // the expected tokens are the process's: simulate the verify of the chosen counts
            let rounds = 40_000;
            let mut got = 0usize;
            for _ in 0..rounds { for (i, &c) in ch.c.iter().enumerate() { let mut a = 0; while a < c && u() < qs[i] { a += 1; } got += a + 1; } }
            let real = got as f64 / rounds as f64;
            assert!((real - ch.expected).abs() < 0.03 * ch.expected, "{qs:?}: expected {:.3} realised {real:.3}", ch.expected);
            // the per-lane rule's rate is at least the one-c rule's on the same round
            let one = shared.choose_conf(&z, MAX_K).unwrap();
            assert!(ch.expected / ch.ms >= one.expected / one.ms - 1e-12, "lanes {:?} {:.4} < shared c {} {:.4}", ch.c, ch.expected / ch.ms, one.c, one.expected / one.ms);
        }
        // the grant + 2 slack: a lane that got 0 last shared round offers 2 drafts, however sure its drafter is
        let ch = lanes.choose_lanes(&[vec![30.0; MAX_K], vec![30.0; MAX_K]], MAX_K, &[2, MAX_K]).unwrap();
        assert_eq!(ch.c[0], 2);
        assert_eq!(ch.capped, 1);
        // the cap: 4 sure lanes at the TP2 default fill 16 rows; at the TP4 default every draft (24 rows)
        let sure = vec![vec![30.0f32; MAX_K]; 4];
        let ch = lanes.choose_lanes(&sure, MAX_K, &[MAX_K; 4]).unwrap();
        assert_eq!(4 + ch.c.iter().sum::<usize>(), 16);
        lanes.max_rows = default_max_rows(4);
        assert_eq!(lanes.choose_lanes(&sure, MAX_K, &[MAX_K; 4]).unwrap().c, vec![5; 4]);
        lanes.max_rows = 64;
        assert_eq!(lanes.round_rows(4), ROUND_ROWS_MAX, "a cap above the engine's round is applied as 32");
        assert_eq!((default_max_rows(1), default_max_rows(2), default_max_rows(4), default_max_rows(8)), (16, 24, 24, 24));
        assert_eq!(crate::dsv41::tp::frame::MAX_ROWS, ROUND_ROWS_MAX);
    }

    /// Leg PS3 without the online model (PS1's boot table + learned overhead): a total between two even counts is priced
    /// between them; an unpriced lane count gives no decision (every draft verified).
    #[test]
    fn dsv41_ps3_lanes_rule_prices_from_the_boot_table_without_a_model() {
        let mut c = Costs::default();
        for k in 0..=5usize { c.vk.insert((2, k), 40.0 + 7.0 * k as f64); }
        let mut p = Policy::with_rule(Mode::Auto, false, Rule::Conf, c);
        p.alloc = Alloc::Lanes;
        p.overhead.insert(2, 1.0);
        assert!((p.price_total(2, 3, 5).unwrap() - (40.0 + 7.0 * 1.5 + 1.0)).abs() < 1e-9);
        assert!((p.price_total(2, 4, 5).unwrap() - (40.0 + 14.0 + 1.0)).abs() < 1e-9);
        assert!(p.choose_lanes(&[vec![0.0; 5], vec![0.0; 5]], 5, &[5, 5]).is_some());
        assert!(p.choose_lanes(&[vec![0.0; 5], vec![0.0; 5], vec![0.0; 5]], 5, &[5, 5, 5]).is_none(), "3 lanes: no table");
        assert_eq!(Alloc::parse("lanes").unwrap(), Alloc::Lanes);
        assert_eq!(Alloc::parse("shared").unwrap(), Alloc::Shared);
        assert!(Alloc::parse("x").is_err());
    }

    #[test]
    fn dsv41_ds_varlen_keeps_prefixes_and_prefers_confident_lanes() {
        let p = Policy::new(Mode::Auto, true, costs());
        let sig = vec![[0.95f32, 0.9, 0.9, 0.85, 0.8], [0.2, 0.1, 0.1, 0.1, 0.1]];
        let k = p.choose_varlen(&sig, 5);
        assert!(k[0] >= k[1], "{k:?}");
        assert!(k[1] <= 1, "{k:?}");
        let off = Policy::new(Mode::Auto, false, costs());
        assert_eq!(off.choose_varlen(&sig, 5), vec![5, 5]);
    }
    #[test]
    fn dsv41_ds_calib_cache_roundtrip_and_stamp() {
        let dir = std::env::temp_dir().join(format!("dsv41_ds_calib_{}", std::process::id()));
        let path = dir.join("pack.json");
        let c = costs();
        let mut c = c;
        c.vk.insert((2, 3), 41.5);
        c.vk_reps.insert((2, 3), vec![41.0, 41.5, 42.25]);
        c.measured_unix = 1_763_600_000;
        write_calib(&path, "s1", &c).unwrap();
        assert_eq!(read_calib(&path, "s1"), Some(c));
        // a fmt-1 cache (no (lanes, drafts) table) and PS1's fmt 2 (no repetitions, no time) re-measure
        std::fs::write(&path, r#"{"fmt": 1, "stamp": "s1", "host_ms": 0.3, "verify": [[1, 16.0]], "draft": []}"#).unwrap();
        assert_eq!(read_calib(&path, "s1"), None);
        std::fs::write(&path, r#"{"fmt": 2, "stamp": "s1", "host_ms": 0.3, "verify": [[1, 16.0]], "vk": [], "draft": []}"#).unwrap();
        assert_eq!(read_calib(&path, "s1"), None);
        assert_eq!(read_calib(&path, "s2"), None);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[test]
    fn dsv41_ds_calibrate_takes_medians() {
        let mut t = 0.0;
        let c = calibrate_verify(&[1, 6], 3, |rows| { t += 1.0; Ok(rows as f64 * 10.0 + t % 2.0) }).unwrap();
        assert_eq!(c.verify.len(), 2);
        assert!((c.verify[&6] - 60.0).abs() <= 1.0);
    }
    #[test]
    fn dsv41_ds_costs_interpolate() {
        let c = costs();
        assert_eq!(c.verify_ms(6), Some(22.0));
        assert!((c.verify_ms(9).unwrap() - 25.5).abs() < 1e-9);
        assert!(c.verify_ms(30).unwrap() > 43.0);
    }

    /// Leg P7.4: every rank constructs `Policy` + `RoundCost` from Rank 0's boot prior and updates it exclusively with the
    /// `(wall_ms, gap_ms)` `f32` bits carried in `Op::Round` headers, producing bit-identical `cs` across 1,000 1..4-lane rounds
    /// with zero mid-round `bcast_small` collectives.
    #[test]
    fn dsv41_p7_lockstep_cs_agrees_across_ranks_without_mid_round_bcast() {
        use super::super::roundcost::{RoundCost, Source, NC};
        let prior: Vec<[f64; NC]> = (1..=4).map(|l| std::array::from_fn(|c| 26.0 + 1.8 * l as f64 + 3.2 * (l * (c + 1)) as f64)).collect();
        for alloc in [Alloc::Lanes, Alloc::Shared] {
            for stop in [Stop::Lookahead, Stop::First] {
                let mut ranks: Vec<(Policy, [usize; 4])> = (0..4).map(|_| {
                    let mut p = Policy::with_rule(Mode::Auto, false, Rule::Conf, Costs::default());
                    p.alloc = alloc;
                    p.stop = stop;
                    p.max_rows = MAX_ROWS_TP2;
                    p.model = Some(RoundCost::new(prior.clone(), Source::Measured, 1_763_800_000));
                    (p, [MAX_K; 4])
                }).collect();
                let mut s: u64 = 0x9E37_79B9_7F4A_7C15 ^ (alloc as u64 * 17 + stop as u64 * 31);
                let mut next_u64 = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); s };
                let mut pending_bits: Option<(u32, u32, usize, usize)> = None;
                for step in 0..1000usize {
                    if let Some((wb, gb, prev_na, prev_tc)) = pending_bits.take() {
                        let wall = f32::from_bits(wb) as f64;
                        let gf = f32::from_bits(gb);
                        for (p, _) in &mut ranks {
                            let m = p.model.as_mut().unwrap();
                            m.observe_d(prev_na, prev_tc, wall);
                            if gf >= 0.0 { m.observe_gap(gf as f64); }
                        }
                    }
                    let na = 1 + (next_u64() as usize % 4);
                    let z: Vec<Vec<f32>> = (0..na).map(|_| {
                        (0..MAX_K).map(|_| (((next_u64() >> 40) as f32 / 16_777_216.0) * 8.0 - 4.0)).collect()
                    }).collect();
                    let mut round_cs: Vec<Vec<usize>> = Vec::with_capacity(ranks.len());
                    let mut round_ms: Vec<u64> = Vec::with_capacity(ranks.len());
                    for (p, held) in &mut ranks {
                        let (cs, ms) = if p.alloc == Alloc::Lanes && na >= 2 {
                            let ch = p.choose_lanes(&z, MAX_K, &held[..na]).unwrap();
                            (ch.c, ch.ms)
                        } else {
                            let depth = if p.alloc == Alloc::Lanes { MAX_K.min(held[0]) } else { MAX_K };
                            let ch = p.choose_conf(&z.iter().map(|v| v[..depth].to_vec()).collect::<Vec<_>>(), depth).unwrap();
                            (vec![ch.c; na], ch.ms)
                        };
                        for i in 0..na {
                            if p.alloc == Alloc::Lanes {
                                held[i] = super::super::alloc::next_held(MAX_K, na, cs[i]);
                            }
                        }
                        round_cs.push(cs);
                        round_ms.push(ms.to_bits());
                    }
                    for r in 1..ranks.len() {
                        assert_eq!(round_cs[r], round_cs[0], "step {step}: rank {r} cs != rank 0");
                        assert_eq!(round_ms[r], round_ms[0], "step {step}: rank {r} priced ms != rank 0");
                    }
                    let total_c: usize = round_cs[0].iter().sum();
                    let rank0_wall = (28.0 + 3.1 * (na + total_c) as f64 + (step % 7) as f64 * 0.37) as f32;
                    let rank0_gap = if step % 5 == 0 { -1.0f32 } else { (0.18 + (step % 4) as f32 * 0.05) };
                    pending_bits = Some((rank0_wall.to_bits(), rank0_gap.to_bits(), na, total_c));
                }
            }
        }
    }
}
