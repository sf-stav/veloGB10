//! Leg PS3: the per-lane verify-row allocation of a shared round (the baseline TensorFold EXL3 recipe's lane core:
//! `windows.allocate` -> `allocate.allocate` at TP2, `allocateRows` with no draft-row price at TP4, which is the same rule).
//!
//! Each live lane brings one pending row (always verified) and its drafts' landing chances (prefix products of the
//! confidences' sigmoids: non-increasing along the lane's chain). Starting from the pending rows (`expected` = one token
//! per lane), the allocation repeatedly adds the likeliest next draft over ALL lanes (a lane can only offer its next
//! undrafted node; ties go to the lowest lane) and prices the round by its TOTAL row count:
//! `rate = expected tokens / round ms(total rows)`; it never goes past `max_rows`. A confident lane therefore gets more
//! verify rows than an uncertain one in the same round.
//!
//! **Where we differ from the reference, and why (exactness).** Their loop scans to `max_rows` and keeps the best prefix
//! seen. Whether lane l's draft j is verified can then depend on lane l's LATER chances, which come from the confidence
//! head fed with draft j itself (the Markov embedding of d_j feeds draft j + 1's confidence): the decision to verify d_j
//! would read d_j, and speculative sampling is exact only when the verify count is a stopping rule that decides on d_j
//! before looking at it (a counter-example is in the tests). We stop at the first added node that does not raise the rate
//! (every node then decides on chances that precede it in the order: other lanes' nodes, whose draws are independent of
//! lane l's, and lane l's earlier drafts). With nodes in non-increasing order and a round cost that is convex in the
//! total rows (a line, as the round-cost model prices it), the rate is unimodal along the order and the two rules choose
//! the same counts (property-tested below); they can differ only where the priced cost has a concave step. Every round
//! also runs their scan and counts the rounds where it would have chosen otherwise (`velogb10_dsv41_spec_alloc_lookahead_total`).

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

/// The rows a lane offers beyond its last shared grant (theirs: `depth.Constants.draft_slack`).
pub const DRAFT_SLACK: usize = 2;

/// One allocation: drafts verified per lane, the expected tokens of the round (lanes + the chances taken), its price,
/// its total verify rows, and whether their scan-to-the-cap rule would have chosen other counts.
#[derive(Clone, Debug, PartialEq)]
pub struct LaneAlloc { pub counts: Vec<usize>, pub expected: f64, pub ms: f64, pub rows: usize, pub scan_differs: bool, pub scan_counts: Vec<usize>,
                       /// leg PS4: how close the exact decision was to flipping: the smallest relative rate change of the last
                       /// draft taken and of the first draft refused (inf: no step was decided)
                       pub margin: f64 }

/// Leg PS4: a decision is "sensitive" (one draft from flipping) when a neighbouring count's rate is within this relative gap.
pub const SENSITIVE_GAP: f64 = 0.02;

/// `probs[l]` = lane l's draft chances (non-increasing; a non-finite or negative chance counts as 0), `price(d)` = the
/// round's ms with `d` drafts over all lanes (`lanes + d` rows; None = unpriced), `max_rows` = the round's row cap
/// (at least one row per lane). None when the round without drafts is unpriced (the caller verifies every draft).
pub fn allocate(probs: &[Vec<f64>], price: impl Fn(usize) -> Option<f64>, max_rows: usize) -> Option<LaneAlloc> {
    allocate_with_lookahead(probs, price, max_rows, None)
}

/// Exact multi-lane allocation with optional constant-decay frontier look-ahead (`lookahead = Some(m)` matches 1-lane
/// `lookahead_index` at n = 1 and sees across the `d = 0 -> 1` fixed drafter-cost step at n >= 2 without ever reading
/// any lane's unverified token draw `d_{l, >= counts[l]}`).
pub fn allocate_with_lookahead(
    probs: &[Vec<f64>],
    price: impl Fn(usize) -> Option<f64>,
    max_rows: usize,
    lookahead: Option<f64>,
) -> Option<LaneAlloc> {
    let n = probs.len();
    if n == 0 { return None; }
    let p: Vec<Vec<f64>> = probs.iter().map(|v| v.iter().map(|&x| if x.is_finite() && x > 0.0 { x.min(1.0) } else { 0.0 }).collect()).collect();
    let rate = |d: usize, e: f64| price(d).filter(|t| *t > 0.0 && t.is_finite()).map(|t| e / t);
    let start = rate(0, n as f64)?;
    let mut counts = vec![0usize; n];
    let (mut rows, mut d, mut expected) = (n, 0usize, n as f64);
    // theirs: the best prefix over the whole scan; ours: exact stopping rule (first-stop or F-measurable frontier look-ahead)
    let (mut scan_best, mut scan_counts) = (start, counts.clone());
    let (mut stop_best, mut stop_counts, mut stopped) = (start, counts.clone(), false);
    let mut margin = f64::INFINITY;
    while rows < max_rows {
        let mut pick: Option<(usize, f64)> = None;
        for (s, ps) in p.iter().enumerate() {
            let Some(&v) = ps.get(counts[s]) else { continue };
            if pick.map_or(true, |(_, best)| v > best) { pick = Some((s, v)); }
        }
        let Some((s, v)) = pick else { break };
        // Before incrementing counts[s], capture the F-measurable projected continuation if look-ahead is enabled:
        // each lane l only uses its known frontier p[l][counts[l]] decayed by m^k, never p[l][> counts[l]].
        let lookahead_beats = if !stopped {
            if let Some(m) = lookahead {
                let base_rate = rate(d, expected).unwrap_or(stop_best);
                let mut sim_c = counts.clone();
                let mut sim_f: Vec<f64> = (0..n).map(|l| p[l].get(counts[l]).copied().unwrap_or(0.0)).collect();
                let mut sim_rows = rows;
                let mut sim_d = d;
                let mut sim_e = expected;
                // Step 1 of continuation is always (s, v)
                sim_c[s] += 1;
                sim_f[s] *= m;
                sim_rows += 1;
                sim_d += 1;
                sim_e += v;
                let mut best_proj = rate(sim_d, sim_e).unwrap_or(-1.0);
                while best_proj <= base_rate && sim_rows < max_rows {
                    let mut sp: Option<(usize, f64)> = None;
                    for l in 0..n {
                        if sim_c[l] >= p[l].len() { continue; }
                        let fv = sim_f[l];
                        if sp.map_or(true, |(_, b)| fv > b) { sp = Some((l, fv)); }
                    }
                    let Some((ls, fv)) = sp else { break };
                    sim_c[ls] += 1;
                    sim_f[ls] *= m;
                    sim_rows += 1;
                    sim_d += 1;
                    sim_e += fv;
                    let r_proj = rate(sim_d, sim_e).unwrap_or(-1.0);
                    if r_proj > best_proj { best_proj = r_proj; }
                }
                Some((best_proj > base_rate, ((best_proj - base_rate) / base_rate).abs()))
            } else {
                None
            }
        } else {
            None
        };
        counts[s] += 1;
        rows += 1;
        d += 1;
        expected += v;
        let now = rate(d, expected).unwrap_or(-1.0);
        if now > scan_best { scan_best = now; scan_counts = counts.clone(); }
        if !stopped {
            if let Some((beats, gap)) = lookahead_beats {
                if gap.is_finite() { margin = margin.min(gap); }
                if beats {
                    stop_best = now;
                    stop_counts = counts.clone();
                } else {
                    stopped = true;
                }
            } else {
                let gap = ((now - stop_best) / stop_best).abs();
                if gap.is_finite() { margin = margin.min(gap); }
                if now > stop_best { stop_best = now; stop_counts = counts.clone(); } else { stopped = true; }
            }
        }
    }
    let e = n as f64 + stop_counts.iter().zip(&p).map(|(&c, ps)| ps[..c].iter().sum::<f64>()).sum::<f64>();
    let total: usize = stop_counts.iter().sum();
    Some(LaneAlloc { ms: price(total).unwrap_or(f64::NAN), expected: e, rows: n + total, scan_differs: scan_counts != stop_counts, counts: stop_counts, scan_counts, margin })
}

/// The drafts lane l offers to the next round: all `depth` after a round it had alone (or a fresh request), at most its
/// grant + DRAFT_SLACK after a shared round (theirs: `Rule.budgets` with node probabilities).
pub fn next_held(depth: usize, lanes_in_round: usize, granted: usize) -> usize { if lanes_in_round >= 2 { depth.min(granted + DRAFT_SLACK) } else { depth } }

// ------------------------------------------------------------------------------------------------ receipts (/metrics)

/// lane-rounds by (lanes in the round 1..=8, drafts verified 0..=5)
static LANE_C: Mutex<[[u64; super::BLOCK + 1]; 8]> = Mutex::new([[0; super::BLOCK + 1]; 8]);
/// per lanes in the round: (rounds, expected tokens, realised tokens, verify rows)
static PER_LANES: Mutex<[(u64, f64, u64, u64); 8]> = Mutex::new([(0, 0.0, 0, 0); 8]);
pub static UNEVEN: AtomicU64 = AtomicU64::new(0);
pub static LOOKAHEAD: AtomicU64 = AtomicU64::new(0);
pub static SLACK_CAPPED: AtomicU64 = AtomicU64::new(0);
pub static NO_PRICE: AtomicU64 = AtomicU64::new(0);
/// the stop rule of this boot (`exact` / `scan`)
static STOP: Mutex<Option<&'static str>> = Mutex::new(None);
pub fn set_stop(stop: &'static str) { if let Ok(mut s) = STOP.lock() { *s = Some(stop); } }
/// leg PS4: the gate-only synthetic jump (ms, f64 bits; 0 = off)
static GATE_JUMP: AtomicU64 = AtomicU64::new(0);
pub fn set_gate_jump(ms: f64) { GATE_JUMP.store(ms.to_bits(), Relaxed); }
/// the allocation mode of this boot (`lanes` / `shared`), the row cap asked and used
static MODE: Mutex<Option<(&'static str, usize, usize)>> = Mutex::new(None);

pub fn set_mode(mode: &'static str, asked: usize, used: usize) { if let Ok(mut m) = MODE.lock() { *m = Some((mode, asked, used)); } }

/// One decided round of `counts.len()` lanes: the drafts each verified, the tokens the rule expected and those emitted,
/// the verify rows, whether their scan would have chosen otherwise, how many lanes the grant + 2 rule capped.
pub fn record(counts: &[usize], expected: f64, realised: u64, rows: usize, scan_differs: bool, capped: usize) {
    let l = counts.len();
    if l == 0 || l > 8 { return; }
    if let Ok(mut h) = LANE_C.lock() { for &c in counts { h[l - 1][c.min(super::BLOCK)] += 1; } }
    if let Ok(mut p) = PER_LANES.lock() { let e = &mut p[l - 1]; e.0 += 1; e.1 += expected; e.2 += realised; e.3 += rows as u64; }
    if counts.iter().any(|&c| c != counts[0]) { UNEVEN.fetch_add(1, Relaxed); }
    if scan_differs { LOOKAHEAD.fetch_add(1, Relaxed); }
    SLACK_CAPPED.fetch_add(capped as u64, Relaxed);
}

fn row(o: &mut String, name: &str, help: &str, v: u64) { o.push_str(&format!("# HELP {name} {help}\n# TYPE {name} counter\n{name} {v}\n")); }

/// The rows (empty unless the mode was announced or a round was recorded).
pub fn render(o: &mut String) {
    let mode = MODE.lock().ok().and_then(|m| *m);
    let any = PER_LANES.lock().map_or(false, |p| p.iter().any(|e| e.0 > 0));
    if mode.is_none() && !any { return; }
    if let Some((m, asked, used)) = mode {
        o.push_str("# HELP velogb10_dsv41_spec_alloc How a round's verify rows are shared between lanes (lanes = per lane by landing chance, the baseline recipe's lane allocation; shared = one draft count for every lane), with the row cap asked and used.\n# TYPE velogb10_dsv41_spec_alloc gauge\n");
        o.push_str(&format!("velogb10_dsv41_spec_alloc{{mode=\"{m}\",max_rows_asked=\"{asked}\",max_rows=\"{used}\"}} 1\n"));
    }
    if let Some(st) = STOP.lock().ok().and_then(|s| *s) {
        o.push_str("# HELP velogb10_dsv41_spec_stop_rule How the draft count is cut: lookahead (a lane alone / shared c: the known chances + a projection that reads no draft) and first (stop at the first draft that does not pay) are distribution-exact; scan (inexact) = the best count over all counts, NOT exact for sampled requests (--dsv41-spec-inexact-scan on, a measurement switch).\n# TYPE velogb10_dsv41_spec_stop_rule gauge\n");
        o.push_str(&format!("velogb10_dsv41_spec_stop_rule{{rule=\"{st}\"}} 1\n"));
        o.push_str(&format!("# HELP velogb10_dsv41_spec_gate_jump_ms GATE ONLY (--dsv41-spec-gate-jump-ms): a synthetic cost added to every priced round with a draft (0 = off: rounds priced as measured).\n# TYPE velogb10_dsv41_spec_gate_jump_ms gauge\nvelogb10_dsv41_spec_gate_jump_ms {}\n", f64::from_bits(GATE_JUMP.load(Relaxed))));
        o.push_str(&format!("# HELP velogb10_dsv41_spec_inexact_scan 1 = this server runs the INEXACT scan rule (--dsv41-spec-inexact-scan on): sampled output is not exactly the model's distribution; a measurement build, never a hand-over build.\n# TYPE velogb10_dsv41_spec_inexact_scan gauge\nvelogb10_dsv41_spec_inexact_scan {}\n", u8::from(st.starts_with("scan"))));
    }
    if let Ok(h) = LANE_C.lock() {
        o.push_str("# HELP velogb10_dsv41_spec_lane_drafts_total Lane-rounds by the lanes in the round and the drafts that lane verified (one count per lane per decided round).\n# TYPE velogb10_dsv41_spec_lane_drafts_total counter\n");
        for (l, r) in h.iter().enumerate() { for (c, x) in r.iter().enumerate() { if *x > 0 { o.push_str(&format!("velogb10_dsv41_spec_lane_drafts_total{{lanes=\"{}\",c=\"{c}\"}} {x}\n", l + 1)); } } }
    }
    if let Ok(p) = PER_LANES.lock() {
        o.push_str("# HELP velogb10_dsv41_spec_lanes_rounds_total Decided rounds by the lanes in the round.\n# TYPE velogb10_dsv41_spec_lanes_rounds_total counter\n");
        for (l, e) in p.iter().enumerate() { if e.0 > 0 { o.push_str(&format!("velogb10_dsv41_spec_lanes_rounds_total{{lanes=\"{}\"}} {}\n", l + 1, e.0)); } }
        o.push_str("# HELP velogb10_dsv41_spec_lanes_expected_tokens_total Tokens the rule expected from those rounds (sum over the round's lanes of 1 + the chances verified).\n# TYPE velogb10_dsv41_spec_lanes_expected_tokens_total counter\n");
        for (l, e) in p.iter().enumerate() { if e.0 > 0 { o.push_str(&format!("velogb10_dsv41_spec_lanes_expected_tokens_total{{lanes=\"{}\"}} {:.3}\n", l + 1, e.1)); } }
        o.push_str("# HELP velogb10_dsv41_spec_lanes_realised_tokens_total Tokens those rounds emitted (sum over the round's lanes).\n# TYPE velogb10_dsv41_spec_lanes_realised_tokens_total counter\n");
        for (l, e) in p.iter().enumerate() { if e.0 > 0 { o.push_str(&format!("velogb10_dsv41_spec_lanes_realised_tokens_total{{lanes=\"{}\"}} {}\n", l + 1, e.2)); } }
        o.push_str("# HELP velogb10_dsv41_spec_lanes_rows_total Verify rows of those rounds (all lanes; pending rows included).\n# TYPE velogb10_dsv41_spec_lanes_rows_total counter\n");
        for (l, e) in p.iter().enumerate() { if e.0 > 0 { o.push_str(&format!("velogb10_dsv41_spec_lanes_rows_total{{lanes=\"{}\"}} {}\n", l + 1, e.3)); } }
    }
    row(o, "velogb10_dsv41_spec_uneven_rounds_total", "Decided rounds whose lanes verified different draft counts (the per-lane allocation at work).", UNEVEN.load(Relaxed));
    row(o, "velogb10_dsv41_spec_alloc_lookahead_total", "Decided rounds where the inexact scan (the best count over all counts: the rule before leg PS3, the reference's best prefix) and the exact rule served choose differently: how often exactness changed the count; with --dsv41-spec-inexact-scan on, the rounds that were served inexactly.", LOOKAHEAD.load(Relaxed));
    row(o, "velogb10_dsv41_spec_slack_capped_total", "Lane-rounds whose offered drafts were cut to the lane's last shared grant + 2 (the reference's draft slack).", SLACK_CAPPED.load(Relaxed));
    row(o, "velogb10_dsv41_spec_alloc_unpriced_total", "Shared rounds the allocation could not price (no round-cost entry): every lane verified every draft.", NO_PRICE.load(Relaxed));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Their table (RX5 / owner analysis 10 s2.1: 30.0, 35.2, ... ms for 1..24 rows), priced by total drafts at `lanes` lanes.
    const THEIRS: [f64; 24] = [30.0, 35.2, 39.9, 44.9, 48.6, 52.1, 55.6, 59.1, 62.7, 66.3, 69.6, 72.9, 75.5, 78.2, 81.3, 84.4, 87.5, 90.6, 93.7, 96.8, 99.9, 103.0, 106.1, 109.2];
    fn theirs(lanes: usize) -> impl Fn(usize) -> Option<f64> { move |d| THEIRS.get(lanes + d - 1).copied() }
    fn linear(a: f64, b: f64) -> impl Fn(usize) -> Option<f64> { move |d| Some(a + b * d as f64) }
    fn chain(q: &[f64]) -> Vec<f64> { let mut c = 1.0; q.iter().map(|x| { c *= x; c }).collect() }

    /// The reference rule, line for line (`allocate.zig:9-48`): scan to the cap, keep the best prefix.
    fn reference(probs: &[Vec<f64>], price: &dyn Fn(usize) -> Option<f64>, max_rows: usize) -> Vec<usize> {
        let n = probs.len();
        let mut counts = vec![0usize; n];
        let mut best_counts = counts.clone();
        let mut rows = n;
        let mut expected = n as f64;
        let rate = |rows: usize, e: f64| price(rows - n).map_or(-1.0, |t| e / t);
        let mut best = rate(rows, expected);
        while rows < max_rows {
            let mut pick: Option<usize> = None;
            let mut pick_p = 0.0;
            for (s, p) in probs.iter().enumerate() {
                if counts[s] >= p.len() { continue; }
                let v = p[counts[s]];
                if pick.is_none() || v > pick_p { pick = Some(s); pick_p = v; }
            }
            let Some(s) = pick else { break };
            counts[s] += 1;
            rows += 1;
            expected += pick_p;
            let now = rate(rows, expected);
            if now > best { best = now; best_counts.copy_from_slice(&counts); }
        }
        best_counts
    }

    /// A confident lane gets more verify rows than an uncertain one in the same round; their table, 2 lanes, cap 16.
    #[test]
    fn dsv41_ps3_confident_lane_gets_more_rows() {
        let sure = chain(&[0.95, 0.95, 0.9, 0.9, 0.85]);
        let unsure = chain(&[0.3, 0.3, 0.3, 0.3, 0.3]);
        let a = allocate(&[sure.clone(), unsure.clone()], theirs(2), 16).unwrap();
        assert!(a.counts[0] >= 4 && a.counts[1] <= 1, "{a:?}");
        assert_eq!(a.counts, reference(&[sure.clone(), unsure.clone()], &theirs(2), 16));
        assert!((a.expected - (2.0 + sure[..a.counts[0]].iter().sum::<f64>() + unsure[..a.counts[1]].iter().sum::<f64>())).abs() < 1e-12);
        assert_eq!(a.rows, 2 + a.counts.iter().sum::<usize>());
        assert!((a.ms - THEIRS[a.rows - 1]).abs() < 1e-12);
        // the same pair in the other lane order: the same rows per lane (the allocation follows the lanes, not their slots)
        let b = allocate(&[unsure, sure], theirs(2), 16).unwrap();
        assert_eq!(b.counts, vec![a.counts[1], a.counts[0]]);
    }

    /// Ties go to the lowest lane: two identical lanes, an odd number of paying drafts.
    #[test]
    fn dsv41_ps3_ties_go_to_the_lowest_lane() {
        let p = vec![0.6, 0.6 * 0.6, 0.0, 0.0, 0.0];
        // per-row cost 0.25 ms on a 10 ms round: chances 0.6 and 0.36 both pay; the 0.6 of lane 0 is taken before lane 1's
        let a = allocate(&[p.clone(), p.clone()], linear(10.0, 0.25), 24).unwrap();
        assert_eq!(a.counts, vec![2, 2]);
        // a cost where exactly 3 drafts pay: lane 0 gets the tied third
        let price = |d: usize| Some(if d <= 3 { 10.0 + 0.1 * d as f64 } else { 100.0 });
        let a = allocate(&[p.clone(), p.clone()], price, 24).unwrap();
        assert_eq!(a.counts, vec![2, 1]);
        assert_eq!(a.counts, reference(&[p.clone(), p], &price, 24));
    }

    /// Fairness: no lane starves. Every lane always verifies its pending row (the allocation only adds drafts), and over
    /// rounds where lanes' chances alternate, each lane gets drafts in proportion to its chances, never zero for a lane
    /// whose drafts pay.
    #[test]
    fn dsv41_ps3_no_lane_starves() {
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut u = || { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (x >> 11) as f64 / (1u64 << 53) as f64 };
        let mut got = [0usize; 4];
        for _ in 0..2000 {
            let probs: Vec<Vec<f64>> = (0..4).map(|_| chain(&(0..5).map(|_| 0.35 + 0.6 * u()).collect::<Vec<_>>())).collect();
            let a = allocate(&probs, theirs(4), 16).unwrap();
            assert!(a.rows <= 16 && a.rows == 4 + a.counts.iter().sum::<usize>(), "{a:?}");
            for l in 0..4 { got[l] += a.counts[l]; }
        }
        let mean = got.iter().sum::<usize>() as f64 / 4.0;
        for g in got { assert!((g as f64 / mean - 1.0).abs() < 0.08, "drafts per lane {got:?}"); }
        // a lane alone in a cap-limited round still gets the drafts that pay; a hopeless lane gets none, but keeps its row
        let a = allocate(&[vec![0.0; 5], chain(&[0.9; 5])], theirs(2), 16).unwrap();
        assert_eq!(a.counts[0], 0);
        assert!(a.counts[1] >= 4);
    }

    /// The row cap: never more rows than max_rows; at least the pending rows even when the cap is below the lane count.
    #[test]
    fn dsv41_ps3_row_cap() {
        let sure = vec![1.0; 5];
        let a = allocate(&vec![sure.clone(); 4], linear(10.0, 0.01), 16).unwrap();
        assert_eq!(a.rows, 16);
        // every chance exactly 1: every node ties, so the lowest lane takes all its drafts first (theirs: ties to the lowest
        // stream; real chances are prefix products of sigmoids < 1, so a lane's next node is below the others' first)
        assert_eq!(a.counts, vec![5, 5, 2, 0]);
        assert_eq!(a.counts, reference(&vec![sure.clone(); 4], &linear(10.0, 0.01), 16));
        let a = allocate(&vec![sure.clone(); 4], linear(10.0, 0.01), 24).unwrap();
        assert_eq!(a.counts, vec![5, 5, 5, 5]);
        let near: Vec<f64> = chain(&[0.99; 5]);
        let a = allocate(&vec![near.clone(); 4], linear(10.0, 0.01), 16).unwrap();
        assert_eq!(a.counts, vec![3, 3, 3, 3], "chances below 1: the rows spread over the lanes");
        let a = allocate(&vec![near; 3], linear(10.0, 0.01), 16).unwrap();
        assert_eq!(a.counts, vec![5, 4, 4]);   // the 13th draft: ties to the lowest lane
        let a = allocate(&vec![sure; 4], linear(10.0, 0.01), 2).unwrap();
        assert_eq!(a.counts, vec![0, 0, 0, 0]);
        // an unpriced round without drafts: no decision
        assert!(allocate(&[vec![0.5]], |_| None, 16).is_none());
    }

    /// With a cost that is a line (or convex) in the total rows, the exact stop rule and the reference scan always agree
    /// (random chances, random lines, every lane count and cap).
    #[test]
    fn dsv41_ps3_stop_rule_equals_their_scan_on_convex_costs() {
        let mut x: u64 = 0x1234_5678_9ABC_DEF1;
        let mut u = || { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (x >> 11) as f64 / (1u64 << 53) as f64 };
        for i in 0..20_000 {
            let lanes = 1 + i % 4;
            let probs: Vec<Vec<f64>> = (0..lanes).map(|_| chain(&(0..5).map(|_| u().powf(0.5)).collect::<Vec<_>>())).collect();
            let (a, b, q) = (20.0 + 40.0 * u(), 0.5 + 6.0 * u(), 0.2 * u());
            let price = move |d: usize| Some(a + b * d as f64 + q * (d * d) as f64);
            let cap = [8, 12, 16, 20, 24][i % 5].max(lanes);
            let got = allocate(&probs, price, cap).unwrap();
            assert_eq!(got.counts, reference(&probs, &price, cap), "lanes {lanes} cap {cap} {probs:?}");
            assert!(!got.scan_differs);
        }
        // their own table is not convex everywhere (row 13 adds 2.6 ms, row 14 2.7, row 15 3.1, ...): count how often the
        // two rules part on it; they part rarely, and the receipt says so
        let mut differ = 0;
        for i in 0..20_000 {
            let lanes = 2 + i % 3;
            let probs: Vec<Vec<f64>> = (0..lanes).map(|_| chain(&(0..5).map(|_| u().powf(0.5)).collect::<Vec<_>>())).collect();
            let got = allocate(&probs, theirs(lanes), 16).unwrap();
            let r = reference(&probs, &theirs(lanes), 16);
            assert_eq!(got.scan_differs, got.counts != r);
            assert_eq!(got.scan_counts, r);
            if got.scan_differs { differ += 1; }
        }
        assert!(differ < 2_000, "{differ} of 20,000 rounds part on their table");
    }

    /// Why the reference's look-ahead is not exact (the module doc), as arithmetic: one lane, drafter p uniform on {a, b},
    /// target q = (0.9, 0.1). Draft 2's chance depends on draft 1 (through the Markov embedding); here it is high after a
    /// and low after b, and draft 1 alone does not pay. The scan verifies d1 only when d1 = a (it looks at draft 2's chance),
    /// so P(emit a) = P(d1 = b) q(a) + P(d1 = a) min(1, q(a) / p(a)) = 0.5 x 0.9 + 0.5 x 1 = 0.95, not 0.9. The stop rule
    /// decides on draft 1 from its own chance: d1 never verified here, every token a fresh target draw, P(emit a) = 0.9.
    #[test]
    fn dsv41_ps3_lookahead_reads_the_draft_it_decides_on() {
        let price = |d: usize| Some([10.0, 12.0, 12.2, 30.0, 40.0, 50.0][d]);
        let p1 = 0.15;   // draft 1 alone: 1.15 / 12 < 1 / 10, it does not pay
        let after_a = vec![p1, p1 * 0.99];
        let after_b = vec![p1, p1 * 0.01];
        let scan = |pr: &Vec<f64>| reference(&[pr.clone()], &price, 6)[0];
        let stop = |pr: &Vec<f64>| allocate(&[pr.clone()], price, 6).unwrap().counts[0];
        assert_eq!((scan(&after_a), scan(&after_b)), (2, 0), "the scan verifies d1 only after a");
        assert_eq!((stop(&after_a), stop(&after_b)), (0, 0), "the stop rule decides before d1");
        let (pa, qa) = (0.5f64, 0.9f64);
        let scan_emit_a = (1.0 - pa) * qa + pa * (qa / pa).min(1.0);
        assert!((scan_emit_a - 0.95).abs() < 1e-12 && (scan_emit_a - qa).abs() > 0.04);
        assert!(allocate(&[after_a], price, 6).unwrap().scan_differs);
    }

    #[test]
    fn dsv41_ps3_held_drafts_follow_the_last_grant() {
        assert_eq!(next_held(5, 1, 0), 5, "a lane alone drafts its full depth");
        assert_eq!(next_held(5, 2, 0), 2);
        assert_eq!(next_held(5, 4, 2), 4);
        assert_eq!(next_held(5, 3, 5), 5);
    }

    #[test]
    fn dsv41_ps3_receipts_render() {
        set_mode("lanes", 16, 16);
        record(&[3, 1], 4.2, 5, 6, false, 1);
        record(&[2, 2, 2, 2], 7.0, 8, 12, true, 0);
        let mut o = String::new();
        render(&mut o);
        for want in ["velogb10_dsv41_spec_alloc{mode=\"lanes\",max_rows_asked=\"16\",max_rows=\"16\"} 1", "velogb10_dsv41_spec_lane_drafts_total{lanes=\"2\",c=\"3\"} ",
                     "velogb10_dsv41_spec_lane_drafts_total{lanes=\"4\",c=\"2\"} ", "velogb10_dsv41_spec_lanes_rounds_total{lanes=\"2\"} ",
                     "velogb10_dsv41_spec_lanes_expected_tokens_total{lanes=\"4\"} ", "velogb10_dsv41_spec_lanes_realised_tokens_total{lanes=\"2\"} ",
                     "velogb10_dsv41_spec_lanes_rows_total{lanes=\"4\"} ", "velogb10_dsv41_spec_uneven_rounds_total ", "velogb10_dsv41_spec_alloc_lookahead_total ",
                     "velogb10_dsv41_spec_slack_capped_total ", "velogb10_dsv41_spec_alloc_unpriced_total "] {
            assert!(o.contains(want), "{want} missing:\n{o}");
        }
        for l in o.lines().filter(|l| !l.starts_with('#')) { assert!(l.rsplit(' ').next().unwrap().parse::<f64>().is_ok(), "bad sample line {l}"); }
    }

    /// The receipts in plain words: `scripts/owner_check.py` on this module's REAL rendered rows (inside the speculation
    /// section, which needs an announced `auto` policy and a round: process-global rows, so the pair is retried).
    #[test]
    fn dsv41_ps3_owner_check_reads_the_allocation() {
        use super::super::metrics as dm;
        set_mode("lanes", 16, 16);
        record(&[4, 1], 4.6, 5, 7, false, 0);
        let mut body = String::new();
        for _ in 0..50 {
            let _ = dm::announce("dspark (TP serve, leg SE)", "Auto");
            dm::set_rule(Some("conf"));
            dm::round(4, 3, true, true, false);
            body = String::from("velogb10_dsv41_serving 1\n");
            dm::render(&mut body);
            if body.contains("velogb10_dsv41_spec_policy{policy=\"auto\"} 1") { break; }
        }
        assert!(body.contains("velogb10_dsv41_spec_alloc{mode=\"lanes\""), "{body}");
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/owner_check.py");
        if !script.exists() || std::process::Command::new("python3").arg("--version").output().is_err() { return; }
        let dir = std::env::temp_dir().join(format!("ps3_oc_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("metrics.txt");
        std::fs::write(&f, &body).unwrap();
        let out = std::process::Command::new("python3").arg("-I").arg(&script).arg("--file").arg(&f).output().expect("run owner_check.py");
        let _ = std::fs::remove_dir_all(&dir);
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        assert!(out.status.success(), "{text}\n{}", String::from_utf8_lossy(&out.stderr));
        for want in ["rows of a round with several requests: each request its own count", "2 request(s) in a round:", "verify rows a round", "drafts verified per request:",
                     "round(s) gave their requests different counts"] {
            assert!(text.contains(want), "owner_check lacks {want:?}:\n{text}");
        }
    }
}
