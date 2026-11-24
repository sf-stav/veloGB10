//! The round-cost table the `conf` draft-length rule prices with (leg PS2; replaces PS1's "measured once at boot, cached
//! for life" table).
//!
//! What was wrong (P1M section 6): the boot table was one noisy calibration (3 timed reps per shape, the shapes measured one
//! after the other while the box heated), clamped by a running maximum, then cached per binary + pack + TP size for every
//! later boot. Across 4 fresh calibrations its 1-lane entries moved by 3.3-8.1 ms and two tables priced adjacent draft
//! counts the same (c = 2 and c = 3 both 42.3 ms: c = 2 was then never chosen), which moved tokens per round by 7-9 %
//! from one boot to the next.
//!
//! What this is:
//! * **A shape per lane count, fitted as fixed cost + per-row cost** (the external recipe's table is that shape: ~+4.4 ms a
//!   row on a ~30 ms base, RX5 section 4): a weighted least-squares line over c = 0..5 with the per-row cost floored above
//!   zero, plus a per-c residual ONLY where the data shows one beyond 4 standard errors (a real non-linearity, e.g. a
//!   kernel-class change between widths). Then strictly increasing in c (each extra draft row costs at least half the
//!   fitted per-row cost): a round with more verify rows is never priced at or below one with fewer.
//! * **Refreshed from the served rounds** (every round knows its lane count, its verified drafts and its wall time): each
//!   (lanes, c) cell keeps a ring of its last 32 rounds (median: an outlier round, a graph capture or an Engram stall,
//!   does not move it), the cells' weights decay with age (a cell not chosen for ~1,000 rounds hands its say back to the
//!   fit and the boot prior), and one **speed factor** shared by every cell follows the box's clock: a robust EWMA of
//!   log(wall / table) that rejects single outliers (beyond +-15 %) but jumps to a new level after 6 consecutive rounds
//!   on the same side (a clock step, a thermal cap, a long context). Samples enter the cells divided by the speed factor,
//!   so cells measured at different clocks stay comparable; the price is speed factor x shape + the host gap between
//!   rounds (median of the last 32 gaps under 50 ms).
//! * **The boot prior** is the boot calibration fitted the same way (interleaved repetitions, see `TpSpec::calibrate`)
//!   plus the overhead the boot calibration cannot see (`--dsv41-spec-overhead-ms`, 8.0 ms), until served rounds replace it.
//!
//! The policy only reads `price(lanes, c)`: it never changes which tokens a round emits (the verify count only), so greedy
//! output is unaffected (AGENTS §2.7).

use std::time::Instant;

/// drafts per lane 0..=CMAX
pub const CMAX: usize = super::BLOCK;
pub const NC: usize = CMAX + 1;
/// rounds kept per (lanes, c) cell
pub const RING: usize = 32;
/// the boot prior's pseudo-weight per c (a cell with 2 served rounds already weighs as much)
pub const PRIOR_W: f64 = 2.0;
/// a cell's weight is its kept rounds, capped
pub const W_CAP: f64 = 16.0;
/// a cell's weight halves every this many rounds served at its lane count without it
pub const HALF_LIFE: f64 = 1000.0;
/// speed factor: EWMA weight, the band (log) inside which a round moves it, the run that moves it to a new level
pub const LAM_ALPHA: f64 = 0.05;
pub const LAM_BAND: f64 = 0.15;
pub const SHIFT_RUN: usize = 6;
/// a residual is used only from a cell this heavy, beyond this many standard errors (floor SE_FLOOR ms); 4, not 3: the MAD
/// of 8 boot repetitions is itself noisy, and a false residual is a mispriced count (synthetic test below)
pub const RES_MIN_W: f64 = 8.0;
pub const RES_SIGMA: f64 = 4.0;
pub const SE_FLOOR: f64 = 0.2;
/// the per-row cost is at least max(B_FLOOR_MS, B_FLOOR_FRAC x fixed); each extra draft row at least MONO_FRAC x per-row
pub const B_FLOOR_MS: f64 = 0.25;
pub const B_FLOOR_FRAC: f64 = 0.02;
pub const MONO_FRAC: f64 = 0.5;
/// host gaps between consecutive rounds counted in the price (longer ones are a prefill or the queue, not the loop)
pub const GAP_MAX_MS: f64 = 50.0;

/// One c of one lane count: a point estimate, its standard error, its weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pt { pub v: f64, pub se: f64, pub w: f64 }

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n == 0 { return 0.0; }
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

/// (median, standard error of the median from the MAD, count) of a sample.
pub fn robust(xs: &[f64]) -> Pt {
    let mut v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    let m = median(&mut v);
    let mut d: Vec<f64> = v.iter().map(|x| (x - m).abs()).collect();
    let mad = median(&mut d) * 1.4826;
    let n = v.len().max(1) as f64;
    Pt { v: m, se: 1.2533 * mad / n.sqrt(), w: v.len() as f64 }
}

/// Weighted least squares line through (c, v) with weights w; the slope floored at max(B_FLOOR_MS, B_FLOOR_FRAC x
/// intercept) (a verify row never costs nothing); a single c gives the floor slope through it. None without weight.
pub fn fit_line(pts: &[(f64, f64, f64)]) -> Option<(f64, f64)> {
    let sw: f64 = pts.iter().map(|p| p.2).sum();
    if !(sw > 0.0) { return None; }
    let cm = pts.iter().map(|p| p.2 * p.0).sum::<f64>() / sw;
    let vm = pts.iter().map(|p| p.2 * p.1).sum::<f64>() / sw;
    let sxx: f64 = pts.iter().map(|p| p.2 * (p.0 - cm) * (p.0 - cm)).sum();
    let sxy: f64 = pts.iter().map(|p| p.2 * (p.0 - cm) * (p.1 - vm)).sum();
    let b = if sxx > 1e-12 { sxy / sxx } else { 0.0 };
    let a = vm - b * cm;
    let floor = B_FLOOR_MS.max(B_FLOOR_FRAC * a.max(0.0));
    if b >= floor { return Some((a, b)); }
    Some((vm - floor * cm, floor))
}

/// The table of one lane count from its points (c = 0..=CMAX, None = no data at that c): the fitted line, a residual
/// where a point is heavy enough and further from the line than RES_SIGMA standard errors, then strictly increasing.
/// Returns (table, fixed, per-row).
pub fn shape(pts: &[Option<Pt>; NC], extra: &[(f64, f64, f64)]) -> Option<([f64; NC], f64, f64)> {
    let mut line: Vec<(f64, f64, f64)> = extra.to_vec();
    for (c, p) in pts.iter().enumerate() { if let Some(p) = p { if p.w > 0.0 { line.push((c as f64, p.v, p.w)); } } }
    let (a, b) = fit_line(&line)?;
    let mut t = [0.0; NC];
    for c in 0..NC {
        let f = a + b * c as f64;
        t[c] = match pts[c] {
            Some(p) if p.w >= RES_MIN_W && (p.v - f).abs() > RES_SIGMA * p.se.max(SE_FLOOR) => p.v,
            _ => f,
        };
    }
    for c in 1..NC { t[c] = t[c].max(t[c - 1] + MONO_FRAC * b); }
    Some((t, a, b))
}

#[derive(Clone, Debug, Default)]
struct Cell { ring: Vec<f64>, head: usize, n: u64, last: u64 }

impl Cell {
    fn push(&mut self, y: f64, at: u64) {
        if self.ring.len() < RING { self.ring.push(y); } else { self.ring[self.head] = y; self.head = (self.head + 1) % RING; }
        self.n += 1;
        self.last = at;
    }
    fn point(&self, now: u64) -> Option<Pt> {
        if self.ring.is_empty() { return None; }
        let mut p = robust(&self.ring);
        let age = now.saturating_sub(self.last) as f64;
        p.w = p.w.min(W_CAP) * 0.5f64.powf(age / HALF_LIFE);
        Some(p)
    }
}

/// Where the table the rule prices with came from (for /metrics and owner_check).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Source { Measured, Cached }

/// The online round-cost model (see the module doc).
///
/// Leg PS3: a round's cost is keyed on (lanes, TOTAL drafts verified over the lanes) = (lanes, total verify rows - lanes),
/// so a round whose lanes verify different counts (the per-lane allocation) is priced and observed like the even ones.
/// The line is still fitted in drafts per lane (x = total drafts / lanes: the same units, floors and constants as PS2's
/// per-c fit), the boot prior sits at the multiples of the lane count (the calibration times even rounds only) and is
/// interpolated between them, a served cell at any total keeps its own residual under PS2's rule, and the table is
/// strictly increasing in the total rows (each extra row at least MONO_FRAC of the fitted per-row cost). At one lane
/// nothing changed (total drafts = c); at 2-4 lanes the price of an even round equals PS2's except where a served cell
/// shows a 4-sigma CHEAPER-than-line residual, which the per-row monotone pass now lifts to stay above the row before.
#[derive(Clone, Debug)]
pub struct RoundCost {
    /// per lane count (index lanes - 1): the boot prior (ms, already shaped, per drafts per lane), its line (fixed, per draft per lane)
    prior: Vec<[f64; NC]>,
    prior_line: Vec<(f64, f64)>,
    /// per lane count: one cell per total drafts 0..=lanes x CMAX
    cells: Vec<Vec<Cell>>,
    /// served rounds per lane count
    rounds: Vec<u64>,
    /// the priced shape per lane count by total drafts (speed-factor units) and its line (per drafts per lane)
    table: Vec<Vec<f64>>,
    line: Vec<(f64, f64)>,
    log_lam: f64,
    run: Vec<f64>,
    run_side: i8,
    gaps: Vec<f64>,
    gap_head: usize,
    pub source: Source,
    /// unix seconds the boot table was measured (a cached table keeps the time it was first measured)
    pub boot_unix: u64,
    pub served: u64,
    pub shifts: u64,
    pub rejected: u64,
    refreshed: Option<Instant>,
    born: Instant,
}

/// The value at total drafts `d` of `lanes` lanes of a per-c table (linear between the multiples of `lanes`).
fn at_total(t: &[f64; NC], lanes: usize, d: usize) -> f64 {
    let (c0, r) = (d / lanes, d % lanes);
    if r == 0 || c0 + 1 >= NC { return t[c0.min(NC - 1)]; }
    t[c0] + (t[c0 + 1] - t[c0]) * r as f64 / lanes as f64
}

impl RoundCost {
    /// From a boot prior per lane count (index lanes - 1; each already a shaped table) and its source.
    pub fn new(prior: Vec<[f64; NC]>, source: Source, boot_unix: u64) -> RoundCost {
        let prior_line: Vec<(f64, f64)> = prior.iter().map(|t| {
            let pts: Vec<(f64, f64, f64)> = t.iter().enumerate().map(|(c, v)| (c as f64, *v, 1.0)).collect();
            fit_line(&pts).unwrap_or((t[0], 0.0))
        }).collect();
        let n = prior.len();
        let table: Vec<Vec<f64>> = prior.iter().enumerate().map(|(li, t)| (0..=(li + 1) * CMAX).map(|d| at_total(t, li + 1, d)).collect()).collect();
        let cells = (0..n).map(|li| vec![Cell::default(); (li + 1) * CMAX + 1]).collect();
        RoundCost { table, line: prior_line.clone(), prior, prior_line, cells, rounds: vec![0; n],
                    log_lam: 0.0, run: Vec::new(), run_side: 0, gaps: Vec::new(), gap_head: 0, source, boot_unix, served: 0, shifts: 0, rejected: 0,
                    refreshed: None, born: Instant::now() }
    }

    /// The lane counts covered (1..=lanes()).
    pub fn lanes(&self) -> usize { self.prior.len() }
    pub fn speed(&self) -> f64 { self.log_lam.exp() }
    /// The host gap priced on top of every round (ms).
    pub fn gap_ms(&self) -> f64 { if self.gaps.is_empty() { 0.0 } else { robust(&self.gaps).v } }
    /// (fixed ms, per-draft-per-lane ms) of the priced line at `lanes` (speed factor applied).
    pub fn fit(&self, lanes: usize) -> Option<(f64, f64)> { self.line.get(lanes.checked_sub(1)?).map(|(a, b)| (a * self.speed(), b * self.speed())) }
    /// Seconds since the table was last refreshed from a served round (None = never: still the boot prior).
    pub fn age_s(&self) -> Option<f64> { self.refreshed.map(|t| t.elapsed().as_secs_f64()) }
    pub fn since_boot_s(&self) -> f64 { self.born.elapsed().as_secs_f64() }

    /// The priced wall time (ms) of a round of `lanes` lanes verifying `c` drafts each.
    pub fn price(&self, lanes: usize, c: usize) -> Option<f64> { self.price_d(lanes, lanes.checked_mul(c)?) }

    /// Leg PS3: the priced wall time (ms) of a round of `lanes` lanes verifying `d` drafts in total (any split over the
    /// lanes: `lanes + d` verify rows). None beyond `lanes` x CMAX drafts or an uncovered lane count.
    pub fn price_d(&self, lanes: usize, d: usize) -> Option<f64> {
        let t = self.table.get(lanes.checked_sub(1)?)?;
        Some(t.get(d)? * self.speed() + self.gap_ms())
    }

    /// The host gap between the end of one round and the start of the next (ms); gaps over GAP_MAX_MS are not the loop.
    pub fn observe_gap(&mut self, ms: f64) {
        if !(ms >= 0.0 && ms < GAP_MAX_MS) { return; }
        if self.gaps.len() < RING { self.gaps.push(ms); } else { self.gaps[self.gap_head] = ms; self.gap_head = (self.gap_head + 1) % RING; }
    }

    /// One served round: `lanes` lanes verified `c` drafts each and took `ms` (draft + choice + verify + commit).
    pub fn observe(&mut self, lanes: usize, c: usize, ms: f64) { if c <= CMAX { self.observe_d(lanes, lanes * c, ms); } }

    /// Leg PS3: one served round of `lanes` lanes that verified `d` drafts in total (any split) and took `ms`.
    pub fn observe_d(&mut self, lanes: usize, d: usize, ms: f64) {
        let Some(li) = lanes.checked_sub(1) else { return };
        if li >= self.table.len() || d > lanes * CMAX || !(ms > 0.0) || !ms.is_finite() { return; }
        // the speed factor: log(wall / table), banded, with a level shift after a run on one side
        let r = (ms / self.table[li][d].max(1e-6)).ln() - self.log_lam;
        if r.abs() <= LAM_BAND {
            self.log_lam += LAM_ALPHA * r;
            self.run.clear();
            self.run_side = 0;
        } else {
            let side = if r > 0.0 { 1 } else { -1 };
            if side != self.run_side { self.run.clear(); self.run_side = side; }
            self.run.push(r + self.log_lam);
            if self.run.len() >= SHIFT_RUN {
                let mut v = std::mem::take(&mut self.run);
                self.log_lam = median(&mut v);
                self.run_side = 0;
                self.shifts += 1;
            } else {
                self.rejected += 1;
            }
        }
        self.rounds[li] += 1;
        let y = ms / self.speed();
        let at = self.rounds[li];
        self.cells[li][d].push(y, at);
        self.served += 1;
        self.refit(li);
        self.refreshed = Some(Instant::now());
    }

    fn refit(&mut self, li: usize) {
        let lanes = li + 1;
        let now = self.rounds[li];
        let data: Vec<Option<Pt>> = self.cells[li].iter().map(|c| c.point(now)).collect();
        // the line over the boot prior (a pseudo-point per c) and the served cells (weight = kept rounds, decayed by age), in
        // drafts per lane (a total of d drafts sits at d / lanes)
        let mut pts: Vec<(f64, f64, f64)> = self.prior[li].iter().enumerate().map(|(c, v)| (c as f64, *v, PRIOR_W)).collect();
        for (d, p) in data.iter().enumerate() { if let Some(p) = p { if p.w > 0.0 { pts.push((d as f64 / lanes as f64, p.v, p.w)); } } }
        let Some((a, b)) = fit_line(&pts) else { return };
        let (pa, pb) = self.prior_line[li];
        let pres: [f64; NC] = std::array::from_fn(|c| self.prior[li][c] - (pa + pb * c as f64));
        let mut t = vec![0.0; data.len()];
        for d in 0..data.len() {
            let x = d as f64 / lanes as f64;
            let f = a + b * x;
            let wd = data[d].map_or(0.0, |p| p.w);
            t[d] = match data[d] {
                // a served non-linearity the data supports
                Some(p) if p.w >= RES_MIN_W && (p.v - f).abs() > RES_SIGMA * p.se.max(SE_FLOOR) => p.v,
                // else the line, plus the boot prior's own residual (interpolated between the lane count's multiples) while
                // the served data there is thin
                _ => f + at_total(&pres, lanes, d) * PRIOR_W / (PRIOR_W + wd),
            };
        }
        let step = MONO_FRAC * b / lanes as f64;
        for d in 1..t.len() { t[d] = t[d].max(t[d - 1] + step); }
        self.table[li] = t;
        self.line[li] = (a, b);
    }

    /// The priced table, every lane count x c (ms, speed factor and host gap applied).
    pub fn rows(&self) -> Vec<(usize, usize, f64)> {
        let mut v = Vec::new();
        for l in 1..=self.lanes() { for c in 0..NC { if let Some(ms) = self.price(l, c) { v.push((l, c, ms)); } } }
        v
    }
}

/// The boot prior of one lane count from the calibration's repetitions: `reps[c]` = the timed walls of the verify of c
/// drafts per lane (draft step + `extra_ms` added to each). Shaped as the online table (fit, significant residuals,
/// strictly increasing). None when no c was measured.
pub fn prior_from_reps(reps: &[Vec<f64>], extra_ms: f64) -> Option<[f64; NC]> {
    let pts: [Option<Pt>; NC] = std::array::from_fn(|c| reps.get(c).filter(|v| !v.is_empty()).map(|v| {
        let mut p = robust(v);
        p.v += extra_ms;
        p
    }));
    shape(&pts, &[]).map(|(t, _, _)| t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed-seed generator: uniform, normal (Box-Muller).
    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> f64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64 }
        fn n(&mut self) -> f64 { (-2.0 * self.u().ln()).sqrt() * (2.0 * std::f64::consts::PI * self.u()).cos() }
    }

    /// A synthetic engine: a round of `lanes` lanes and c drafts costs `fixed + per_row x lanes x (c + 1)` x `clock`, plus
    /// gaussian noise of `sd` ms, plus an outlier (x 2-6) in `p_out` of the rounds.
    fn sample(r: &mut Rng, lanes: usize, c: usize, clock: f64, sd: f64, p_out: f64) -> f64 {
        let base = (24.0 + 2.0 * lanes as f64 + 3.8 * (lanes * (c + 1)) as f64) * clock;
        let x = base + sd * r.n();
        if r.u() < p_out { x * (2.0 + 4.0 * r.u()) } else { x }
    }
    fn truth(lanes: usize, c: usize) -> f64 { 24.0 + 2.0 * lanes as f64 + 3.8 * (lanes * (c + 1)) as f64 }

    fn calib(r: &mut Rng, reps: usize, sd: f64, p_out: f64) -> [f64; NC] {
        let v: Vec<Vec<f64>> = (0..NC).map(|c| (0..reps).map(|_| sample(r, 1, c, 1.0, sd, p_out)).collect()).collect();
        prior_from_reps(&v, 0.0).unwrap()
    }

    fn strictly_increasing(t: &[f64]) -> bool { t.windows(2).all(|w| w[1] > w[0]) }

    /// Reproducible. 400 independent boot calibrations of 8 interleaved repetitions per shape, on a harsh synthetic engine
    /// (2 ms gaussian noise on a 30-50 ms round, plus 10 % of rounds x 2-6 outliers): every table is strictly increasing
    /// (no tie between adjacent draft counts, ever); the table's SHAPE (each entry less the table's mean offset, which is
    /// common to every c and cannot move a choice) is within 1.5 ms of the truth in 95 % of boots; and the mean range of an
    /// entry over 4 boots is under 1.6 ms (P1M measured 3.3-8.1 ms over its 4 boots). With 1 ms noise and no outliers the
    /// shape error is under 0.8 ms in 95 % of boots. The old rule (median of 3 + running maximum) on the harsh engine ties
    /// adjacent counts in a large share of boots.
    #[test]
    fn dsv41_ps2_boot_table_is_reproducible_and_strictly_increasing() {
        let run = |seed: u64, sd: f64, p_out: f64| -> (f64, f64) {
            let mut r = Rng(seed);
            let mut err = vec![Vec::new(); NC];
            let mut worst = Vec::new();
            for _ in 0..400 {
                let t = calib(&mut r, 8, sd, p_out);
                assert!(strictly_increasing(&t), "{t:?}");
                let off: f64 = (0..NC).map(|c| t[c] - truth(1, c)).sum::<f64>() / NC as f64;
                worst.push((0..NC).map(|c| (t[c] - truth(1, c) - off).abs()).fold(0.0, f64::max));
                for c in 0..NC { err[c].push(t[c] - truth(1, c)); }
            }
            worst.sort_by(|a, b| a.total_cmp(b));
            let (mut range, mut k) = (0.0, 0.0);
            for e in &err { for w in e.chunks(4) { range += w.iter().cloned().fold(f64::MIN, f64::max) - w.iter().cloned().fold(f64::MAX, f64::min); k += 1.0; } }
            (worst[380], range / k)
        };
        let (p95, range4) = run(0x5EED_0001, 2.0, 0.10);
        assert!(p95 < 1.5 && range4 < 1.6, "harsh engine: shape error p95 {p95:.2} ms, mean 4-boot range {range4:.2} ms");
        let (p95, _) = run(0x5EED_0002, 1.0, 0.0);
        assert!(p95 < 0.8, "mild engine: shape error p95 {p95:.2} ms");
        // the old calibration: median of 3 per shape, running maximum (ties allowed)
        let mut r = Rng(0x5EED_0003);
        let mut ties = 0;
        for _ in 0..200 {
            let mut m = [0.0f64; NC];
            for c in 0..NC { let mut v: Vec<f64> = (0..3).map(|_| sample(&mut r, 1, c, 1.0, 2.0, 0.10)).collect(); m[c] = median(&mut v); }
            for c in 1..NC { m[c] = m[c].max(m[c - 1]); }
            if m.windows(2).any(|w| w[1] <= w[0]) { ties += 1; }
        }
        assert!(ties >= 10, "the old rule tied adjacent counts in only {ties} of 200 boots");
    }

    /// A non-linearity the data supports (c = 1 costs 3 ms more than the line: a kernel class change) is kept; the same
    /// shift buried in noise is not.
    #[test]
    fn dsv41_ps2_residual_only_where_the_data_supports_it() {
        let mut r = Rng(7);
        let reps: Vec<Vec<f64>> = (0..NC).map(|c| (0..16).map(|_| truth(1, c) + if c == 1 { 3.0 } else { 0.0 } + 0.5 * r.n()).collect()).collect();
        let t = prior_from_reps(&reps, 0.0).unwrap();
        assert!((t[1] - (truth(1, 1) + 3.0)).abs() < 0.6, "kept: {t:?}");
        assert!(strictly_increasing(&t));
        let noisy: Vec<Vec<f64>> = (0..NC).map(|c| (0..3).map(|_| truth(1, c) + if c == 1 { 3.0 } else { 0.0 } + 4.0 * r.n()).collect()).collect();
        let (_, a, b) = shape(&std::array::from_fn(|c| Some(robust(&noisy[c]))), &[]).unwrap();
        let tn = prior_from_reps(&noisy, 0.0).unwrap();
        assert!((tn[1] - (a + b)).abs() < 1e-9, "3 noisy reps are not enough for a residual: {tn:?}");
    }

    /// Monotone in the verify rows even on adversarial data (a deeper count measured cheaper), and the per-row floor holds.
    #[test]
    fn dsv41_ps2_table_never_prices_more_rows_cheaper() {
        let reps: Vec<Vec<f64>> = [30.0, 36.0, 36.0, 35.0, 46.0, 45.0].iter().map(|&v| vec![v; 10]).collect();
        let t = prior_from_reps(&reps, 0.0).unwrap();
        assert!(strictly_increasing(&t), "{t:?}");
        let flat: Vec<Vec<f64>> = (0..NC).map(|_| vec![40.0; 10]).collect();
        let (tf, _, b) = shape(&std::array::from_fn(|c| Some(robust(&flat[c]))), &[]).unwrap();
        assert!(b >= B_FLOOR_MS && strictly_increasing(&tf), "{tf:?}");
    }

    /// Served rounds refresh the table: the prior is off by +30 % and mis-shaped; after a few hundred served rounds (noise,
    /// outliers) the prices are within 3 % of the truth at every c, and strictly increasing.
    #[test]
    fn dsv41_ps2_served_rounds_replace_a_wrong_prior() {
        let mut r = Rng(11);
        let prior: Vec<[f64; NC]> = (1..=4).map(|l| std::array::from_fn(|c| truth(l, c) * 1.3 + if c == 2 { 6.0 } else { 0.0 })).collect();
        let mut m = RoundCost::new(prior, Source::Cached, 0);
        for i in 0..1200 {
            let lanes = 1 + (i % 4);
            let c = (i * 7 + i / 5) % NC;
            m.observe(lanes, c, sample(&mut r, lanes, c, 1.0, 1.5, 0.05));
        }
        for l in 1..=4 {
            let t: Vec<f64> = (0..NC).map(|c| m.price(l, c).unwrap()).collect();
            assert!(strictly_increasing(&t), "lanes {l}: {t:?}");
            for c in 0..NC { assert!((t[c] / truth(l, c) - 1.0).abs() < 0.03, "lanes {l} c {c}: {:.2} vs {:.2}", t[c], truth(l, c)); }
        }
        assert!(m.age_s().is_some() && m.served == 1200);
    }

    /// A clock step mid-run (the GPU drops to 0.6x speed: every round 1/0.6 longer) moves the speed factor to the new level
    /// within ~SHIFT_RUN rounds, the shape (which c is cheapest per row) does not change, and single outliers never move it.
    #[test]
    fn dsv41_ps2_clock_step_and_outliers() {
        let mut r = Rng(23);
        let prior: Vec<[f64; NC]> = vec![std::array::from_fn(|c| truth(1, c))];
        let mut m = RoundCost::new(prior, Source::Measured, 0);
        for i in 0..400 { m.observe(1, i % NC, sample(&mut r, 1, i % NC, 1.0, 1.0, 0.0)); }
        assert!((m.speed() - 1.0).abs() < 0.03, "{}", m.speed());
        let before: Vec<f64> = (0..NC).map(|c| m.price(1, c).unwrap()).collect();
        // single outliers (one in 10 rounds x 2-6) never move the factor by more than 2 %
        for i in 0..200 { let c = i % NC; let x = if i % 10 == 5 { sample(&mut r, 1, c, 1.0, 1.0, 1.0) } else { sample(&mut r, 1, c, 1.0, 1.0, 0.0) }; m.observe(1, c, x); }
        assert!((m.speed() - 1.0).abs() < 0.02, "outliers moved the speed factor to {}", m.speed());
        // the step
        let step = 1.0 / 0.6;
        let mut caught = None;
        for i in 0..60 {
            let c = (i * 3) % NC;
            m.observe(1, c, sample(&mut r, 1, c, step, 1.0, 0.0));
            if caught.is_none() && (m.speed() / step - 1.0).abs() < 0.05 { caught = Some(i + 1); }
        }
        let n = caught.expect("the speed factor never reached the new clock");
        assert!(n <= SHIFT_RUN + 2, "caught after {n} rounds");
        for i in 0..400 { let c = i % NC; m.observe(1, c, sample(&mut r, 1, c, step, 1.0, 0.0)); }
        let after: Vec<f64> = (0..NC).map(|c| m.price(1, c).unwrap()).collect();
        for c in 0..NC { assert!((after[c] / before[c] / step - 1.0).abs() < 0.03, "c {c}: {:.2} -> {:.2}", before[c], after[c]); }
        assert!(m.shifts >= 1 && strictly_increasing(&after));
    }

    /// A cell the policy stopped choosing goes stale: its weight decays and the fit + prior price it, so an old cool-box
    /// sample cannot keep a count looking cheap (or dear) for ever; and the host gap is priced from short gaps only.
    #[test]
    fn dsv41_ps2_stale_cells_hand_back_to_the_fit_and_the_gap_is_bounded() {
        let mut r = Rng(31);
        let mut m = RoundCost::new(vec![std::array::from_fn(|c| truth(1, c))], Source::Measured, 0);
        // c = 3 measured 8 ms cheaper than truth (a cool burst), then never again
        for _ in 0..32 { m.observe(1, 3, truth(1, 3) - 8.0 + 0.3 * r.n()); }
        let fresh = m.price(1, 3).unwrap();
        for i in 0..6000 { let c = [0usize, 1, 2, 4, 5][i % 5]; m.observe(1, c, sample(&mut r, 1, c, 1.0, 1.0, 0.0)); }
        let t: Vec<f64> = (0..NC).map(|c| m.price(1, c).unwrap()).collect();
        assert!((t[3] - truth(1, 3)).abs() < 1.5, "stale c = 3 still priced {:.2} (fresh {fresh:.2}, truth {:.2})", t[3], truth(1, 3));
        assert!(strictly_increasing(&t));
        m.observe_gap(0.4);
        m.observe_gap(0.6);
        m.observe_gap(900.0);
        assert!((m.gap_ms() - 0.5).abs() < 1e-9);
    }

    /// PS2's model as it was (keyed on drafts per lane c), kept here as the reference the D-keyed model must match on even
    /// rounds: the same observe / refit arithmetic, line for line, on a table of NC entries per lane count.
    struct Ps2Ref { prior: Vec<[f64; NC]>, prior_line: Vec<(f64, f64)>, cells: Vec<Vec<Cell>>, rounds: Vec<u64>, table: Vec<[f64; NC]>, log_lam: f64, run: Vec<f64>, run_side: i8 }
    impl Ps2Ref {
        fn new(prior: Vec<[f64; NC]>) -> Ps2Ref {
            let prior_line: Vec<(f64, f64)> = prior.iter().map(|t| fit_line(&t.iter().enumerate().map(|(c, v)| (c as f64, *v, 1.0)).collect::<Vec<_>>()).unwrap()).collect();
            let n = prior.len();
            Ps2Ref { table: prior.clone(), prior, prior_line, cells: vec![vec![Cell::default(); NC]; n], rounds: vec![0; n], log_lam: 0.0, run: Vec::new(), run_side: 0 }
        }
        fn price(&self, lanes: usize, c: usize) -> f64 { self.table[lanes - 1][c] * self.log_lam.exp() }
        fn observe(&mut self, lanes: usize, c: usize, ms: f64) {
            let li = lanes - 1;
            let r = (ms / self.table[li][c].max(1e-6)).ln() - self.log_lam;
            if r.abs() <= LAM_BAND { self.log_lam += LAM_ALPHA * r; self.run.clear(); self.run_side = 0; } else {
                let side = if r > 0.0 { 1 } else { -1 };
                if side != self.run_side { self.run.clear(); self.run_side = side; }
                self.run.push(r + self.log_lam);
                if self.run.len() >= SHIFT_RUN { let mut v = std::mem::take(&mut self.run); self.log_lam = median(&mut v); self.run_side = 0; }
            }
            self.rounds[li] += 1;
            let at = self.rounds[li];
            self.cells[li][c].push(ms / self.log_lam.exp(), at);
            let now = self.rounds[li];
            let data: Vec<Option<Pt>> = self.cells[li].iter().map(|c| c.point(now)).collect();
            let mut pts: Vec<(f64, f64, f64)> = self.prior[li].iter().enumerate().map(|(c, v)| (c as f64, *v, PRIOR_W)).collect();
            for (c, p) in data.iter().enumerate() { if let Some(p) = p { if p.w > 0.0 { pts.push((c as f64, p.v, p.w)); } } }
            let (a, b) = fit_line(&pts).unwrap();
            let (pa, pb) = self.prior_line[li];
            let mut t = [0.0; NC];
            for c in 0..NC {
                let f = a + b * c as f64;
                let wd = data[c].map_or(0.0, |p| p.w);
                t[c] = match data[c] {
                    Some(p) if p.w >= RES_MIN_W && (p.v - f).abs() > RES_SIGMA * p.se.max(SE_FLOOR) => p.v,
                    _ => f + (self.prior[li][c] - (pa + pb * c as f64)) * PRIOR_W / (PRIOR_W + wd),
                };
            }
            for c in 1..NC { t[c] = t[c].max(t[c - 1] + MONO_FRAC * b); }
            self.table[li] = t;
        }
    }

    /// Leg PS3: on even rounds (every lane the same c) the D-keyed model prices every (lanes, c) exactly as PS2's c-keyed
    /// model did (1-4 lanes, a wrong prior, noise, outliers, a clock step): a lane alone and the shared rule are unchanged.
    #[test]
    fn dsv41_ps3_even_rounds_price_as_ps2_did() {
        let mut r = Rng(41);
        let prior: Vec<[f64; NC]> = (1..=4).map(|l| std::array::from_fn(|c| truth(l, c) * 1.2 + if c == 3 { 2.0 } else { 0.0 })).collect();
        let mut m = RoundCost::new(prior.clone(), Source::Measured, 0);
        let mut old = Ps2Ref::new(prior);
        for l in 1..=4 { for c in 0..NC { assert_eq!(m.price(l, c).unwrap(), old.price(l, c), "before any round: lanes {l} c {c}"); } }
        for i in 0..3000 {
            let lanes = 1 + (i * 3) % 4;
            let c = (i * 7 + i / 3) % NC;
            let clock = if i > 1500 { 1.4 } else { 1.0 };
            let ms = sample(&mut r, lanes, c, clock, 1.5, 0.05);
            m.observe(lanes, c, ms);
            old.observe(lanes, c, ms);
            for l in 1..=4 { for cc in 0..NC {
                let (a, b) = (m.price(l, cc).unwrap(), old.price(l, cc));
                assert!((a - b).abs() <= 1e-9 * b, "round {i}: lanes {l} c {cc}: {a} vs PS2 {b}");
            } }
        }
    }

    /// Leg PS3: rounds whose lanes verify different counts are observed at their total drafts; the model then prices every
    /// total (not only the multiples of the lane count) within 3 % of an engine whose cost follows the total rows, strictly
    /// increasing in the total.
    #[test]
    fn dsv41_ps3_uneven_rounds_price_the_total() {
        let mut r = Rng(43);
        let truth_d = |lanes: usize, d: usize| 24.0 + 2.0 * lanes as f64 + 3.8 * (lanes + d) as f64;
        let prior: Vec<[f64; NC]> = (1..=4).map(|l| std::array::from_fn(|c| truth(l, c) * 1.25)).collect();
        let mut m = RoundCost::new(prior, Source::Cached, 0);
        for i in 0..4000 {
            let lanes = 2 + i % 3;
            let d = (i * 5 + i / 7) % (lanes * CMAX + 1);
            let ms = truth_d(lanes, d) + 1.5 * r.n();
            m.observe_d(lanes, d, if r.u() < 0.04 { ms * 3.0 } else { ms });
        }
        for lanes in 2..=4 {
            let t: Vec<f64> = (0..=lanes * CMAX).map(|d| m.price_d(lanes, d).unwrap()).collect();
            assert!(strictly_increasing(&t), "lanes {lanes}: {t:?}");
            for (d, x) in t.iter().enumerate() { assert!((x / truth_d(lanes, d) - 1.0).abs() < 0.03, "lanes {lanes} d {d}: {x:.2} vs {:.2}", truth_d(lanes, d)); }
            assert!(m.price_d(lanes, lanes * CMAX + 1).is_none());
            for c in 0..NC { assert_eq!(m.price(lanes, c), m.price_d(lanes, lanes * c)); }
        }
        // before any served round, a total between two even counts is priced between them (the boot prior interpolated)
        let p: Vec<[f64; NC]> = vec![std::array::from_fn(|c| truth(1, c)), std::array::from_fn(|c| truth(2, c))];
        let f = RoundCost::new(p, Source::Measured, 0);
        assert!((f.price_d(2, 3).unwrap() - 0.5 * (f.price(2, 1).unwrap() + f.price(2, 2).unwrap())).abs() < 1e-9);
    }
}
