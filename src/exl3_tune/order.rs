//! Measurement protocol v2 (pure CPU): the exactly balanced Williams arm order, the position-effect model and the plan ordering by expected value.
//! See `mod.rs` for the file map.

use super::*;

// ---------------------------------------------------------------------------------------------
// Measurement protocol v2 (§5.2 amendment, 2026-09-27): an exactly balanced arm order and the
// position model. Pure CPU. The v1 protocol re-randomized the arm order every round; the .14
// self-test receipts (selftest-1790470336 / -1790470533) showed the first verify replay of every
// round ~0.6-1.1 ms slower than the next, a random order that did not balance first positions
// (base first 49x vs base2 35x), and a paired HL biased by the imbalance.
// ---------------------------------------------------------------------------------------------

/// Rows in one period of the balanced order over `n` arms: n for even n, 2n for odd n.
pub fn williams_period(n: usize) -> usize {
    if n <= 1 { 1 } else if n % 2 == 0 { n } else { 2 * n }
}

/// Row `r` (mod the period) of a Williams design over arms 0..n: every row is a permutation; over one
/// period every arm sits at every position exactly period/n times (once for even n, twice for odd
/// n: the cyclic square plus its mirror) and every ordered pair of distinct arms is adjacent exactly
/// period/n times (first-order carry-over balance: each arm follows every other arm equally often).
pub fn williams_row(n: usize, r: usize) -> Vec<usize> {
    if n <= 1 { return vec![0; n]; }
    let r = r % williams_period(n);
    // first row 0, 1, n-1, 2, n-2, ...; row r adds r (mod n); odd n: rows n..2n are mirrored
    let row: Vec<usize> = (0..n).map(|j| {
        let a = if j == 0 { 0 } else if j % 2 == 1 { (j + 1) / 2 } else { n - j / 2 };
        (a + r % n) % n
    }).collect();
    if r >= n { row.into_iter().rev().collect() } else { row }
}

/// Per-cell cursors over the balanced order. `next(cell, n)` returns the next row for that analysis
/// cell (a unit x width x regime; the caller encodes it). Cursors are keyed by (cell, n) and persist
/// across arm-set changes, so short race phases that revisit a cell CONTINUE the design instead of
/// restarting it (a restart at a fixed row would put the same arm first every time). The start row
/// of a new cursor is derived from the seed — deterministic, and balanced over every complete period
/// from any start.
#[derive(Clone, Debug, Default)]
pub struct BalancedOrder {
    seed: u64,
    cur: HashMap<(u64, usize), usize>,
}

impl BalancedOrder {
    pub fn new(seed: u64) -> BalancedOrder { BalancedOrder { seed, cur: HashMap::new() } }

    pub fn next(&mut self, cell: u64, n: usize) -> Vec<usize> {
        let seed = self.seed;
        let c = self.cur.entry((cell, n)).or_insert_with(|| {
            let mut z = seed ^ cell.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (n as u64).rotate_left(32);
            z = (z ^ (z >> 31)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            (z >> 17) as usize % williams_period(n)
        });
        let row = williams_row(n, *c);
        *c += 1;
        row
    }
}

/// One timed replay for the position model: `round` (the pairing unit: same inputs), `arm` (the
/// configuration, unique per arm-set generation), `pos` (0 = the first TIMED replay of the block).
#[derive(Clone, Copy, Debug)]
pub struct PosObs {
    pub round: u64,
    pub arm: u64,
    pub pos: usize,
    pub ms: f64,
}

/// Position offsets f[p] (ms; mean 0 over the positions that have data; unseen positions read 0) of
/// ONE analysis cell (same unit, width, regime and number of positions), fitted by median polish of
/// the additive model  t = round + arm + f(position) + noise  — the round absorbs the content
/// (routing, context, drafts), the arm its configuration. A position-stratified paired difference
/// is then (t_c - f[p_c]) - (t_b - f[p_b]).
/// * With a balanced order, arm and position are orthogonal and the fit is the position effect.
/// * It never invents an effect: the polish starts at f = 0; rounds with < 2 replays and arms seen
///   in < 2 rounds carry no information about f and are dropped; a confounded design (an arm always
///   at one position) leaves the effect with the arm (f stays ~0 = no adjustment).
pub fn position_offsets(obs: &[PosObs], n_pos: usize) -> Vec<f64> {
    let mut f = vec![0.0f64; n_pos];
    if n_pos < 2 || obs.len() < 4 { return f; }
    // informative subset: rounds with >= 2 replays, arms in >= 2 such rounds (twice, to a fixpoint-ish)
    let mut keep: Vec<PosObs> = obs.iter().copied().filter(|o| o.pos < n_pos && o.ms.is_finite()).collect();
    for _ in 0..2 {
        let mut per_round: HashMap<u64, usize> = HashMap::new();
        for o in &keep { *per_round.entry(o.round).or_default() += 1; }
        keep.retain(|o| per_round[&o.round] >= 2);
        let mut arm_rounds: HashMap<u64, std::collections::BTreeSet<u64>> = HashMap::new();
        for o in &keep { arm_rounds.entry(o.arm).or_default().insert(o.round); }
        keep.retain(|o| arm_rounds[&o.arm].len() >= 2);
    }
    if keep.len() < 4 { return f; }
    let idx = |ids: Vec<u64>| -> (Vec<usize>, usize) {
        let mut map: HashMap<u64, usize> = HashMap::new();
        let v: Vec<usize> = ids.into_iter().map(|x| { let n = map.len(); *map.entry(x).or_insert(n) }).collect();
        let n = map.len();
        (v, n)
    };
    let (ri, nr) = idx(keep.iter().map(|o| o.round).collect());
    let (ai, na) = idx(keep.iter().map(|o| o.arm).collect());
    let group = |ix: &[usize], n: usize| -> Vec<Vec<usize>> {
        let mut g = vec![Vec::new(); n];
        for (i, &k) in ix.iter().enumerate() { g[k].push(i); }
        g
    };
    let (gr, ga) = (group(&ri, nr), group(&ai, na));
    let gp = group(&keep.iter().map(|o| o.pos).collect::<Vec<_>>(), n_pos);
    let (mut mu, mut beta) = (vec![0.0f64; nr], vec![0.0f64; na]);
    let med = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        median_sorted(&v)
    };
    for _ in 0..16 {
        for (r, g) in gr.iter().enumerate() {
            mu[r] = med(g.iter().map(|&i| keep[i].ms - beta[ai[i]] - f[keep[i].pos]).collect());
        }
        for (a, g) in ga.iter().enumerate() {
            beta[a] = med(g.iter().map(|&i| keep[i].ms - mu[ri[i]] - f[keep[i].pos]).collect());
        }
        for (p, g) in gp.iter().enumerate() {
            if g.is_empty() { continue; }
            f[p] = med(g.iter().map(|&i| keep[i].ms - mu[ri[i]] - beta[ai[i]]).collect());
        }
        let seen: Vec<usize> = (0..n_pos).filter(|&p| !gp[p].is_empty()).collect();
        let c = seen.iter().map(|&p| f[p]).sum::<f64>() / seen.len().max(1) as f64;
        for &p in &seen { f[p] -= c; }
    }
    f
}

/// One family of the search plan (a knob, or a declared interacting pair) with its ledger value.
#[derive(Clone, Debug)]
pub struct PlanItem {
    pub name: String,
    /// expected realizable gain per round (ms, the §10 mid estimate)
    pub ev_ms: f64,
    /// estimated tuning seconds (candidates x rounds x round cost)
    pub est_s: f64,
}

/// §6.2 ledger-weighted order: descending expected value density ev / seconds (stable on ties).
pub fn order_by_ev(items: &mut [PlanItem]) {
    items.sort_by(|a, b| {
        let da = a.ev_ms / a.est_s.max(1e-3);
        let db = b.ev_ms / b.est_s.max(1e-3);
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// §6.6 budget planner: the prefix of `items` (in order) whose cumulative estimate fits
/// `budget_s - spent_s`; the rest are `UNTUNED (budget)`.
pub fn plan_fits(items: &[PlanItem], spent_s: f64, budget_s: f64) -> usize {
    let mut t = spent_s;
    for (i, it) in items.iter().enumerate() {
        t += it.est_s;
        if t > budget_s { return i; }
    }
    items.len()
}
