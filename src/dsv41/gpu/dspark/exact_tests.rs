//! Leg PS3: is a sampled lane's output an exact draw from the target, under each draft-count rule? A statistical test
//! with power, on a 3-token synthetic model, through the REAL policy code (`Policy::choose_conf`, `Policy::choose_lanes`,
//! `alloc::next_held`) and a host twin of the round's draw / accept / residual kernels (`91_ds_vocab.cuh`: the drafter's
//! keyed Gumbel draw in the round-keyed draft namespace, accept `u < min(1, q / p)` with `u` keyed by position, residual
//! `max(q - alpha p, 0)` drawn with its own keyed Gumbel, the bonus draw from q when every verified draft is accepted),
//! built on the engine's host noise functions (`vocab::{key, uniform, gumbel, counter, draft_seed}`).
//!
//! The model: target q(. | prev) and drafter p(. | prev), 3 x 3; the confidence logit of draft j is z[d_{j-1}] (as the
//! engine's: draft j's confidence is computed from the token before it, never from draft j). Every trial starts a lane at
//! token 2 with a fresh seed and runs rounds until it has emitted 2 tokens; the first two emitted tokens (9 cells) are
//! compared with the target's joint q(a | 2) q(b | a) by a one-sample chi-square.
//!
//! Power. Screening arms: N = 200,000 trials (1 % absolute between two cells of probability <= 0.5: noncentrality
//! >= 2e5 x (1e-4 / 0.5 x 2) = 80 at <= 8 df, rejected at p < 1e-3 with probability > 0.999). The hard-case arms (concave
//! table, `dsv41_ps3_exactness_hard_case_0p1_percent`): N = 4,000,000 trials: 0.1 % absolute between two cells of
//! probability <= 0.16 (the largest cells of this model) gives noncentrality >= 4e6 x (1e-6 / 0.16 x 2) = 50 at 8 df,
//! rejected at p < 1e-3 (chi2 > 26.1) with probability ~0.99; a control that reads its draft in 3 % of rounds (a 0.3 %
//! deviation) must be rejected there, showing the power on the real harness. An exact arm must give p > 1e-3 (fixed
//! seeds: reproducible); a biased arm p < 1e-6.

use super::alloc;
use super::policy::{Alloc, Costs, Mode, Policy, Rule, Stop};
use super::vocab::{counter, draft_seed, gumbel, key, uniform};
use std::collections::HashMap;

const Q: [[f64; 3]; 3] = [[0.6, 0.3, 0.1], [0.3, 0.3, 0.4], [0.2, 0.4, 0.4]];
const P: [[f64; 3]; 3] = [[0.3, 0.4, 0.3], [0.5, 0.2, 0.3], [0.5, 0.25, 0.25]];
const X0: usize = 2;
const DEPTH: usize = 3;
const N: usize = 200_000;
const N_HARD: usize = 4_000_000;

fn logit(p: f64) -> f32 { (p / (1.0 - p)).ln() as f32 }
/// draft j's confidence logit from the token before it: after 0 a sure next draft, after 1 a hopeless one, after 2 (the
/// start) 0.15
fn z_of(prev: usize) -> f32 { [logit(0.99), logit(0.01), logit(0.15)][prev] }

/// A Gumbel-max draw from `dist` under `k` (zero-mass ids never drawn).
fn draw(dist: &[f64; 3], k: u64) -> usize {
    let mut best = (f64::NEG_INFINITY, 0usize);
    for (id, &w) in dist.iter().enumerate() { if w > 0.0 { let s = w.ln() + gumbel(k, id as u32); if s > best.0 { best = (s, id); } } }
    best.1
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Arm { Policy(Stop, Alloc), BiasedControl, SlightlyBiased }

struct LaneSim { seed: u64, x: usize, pos: i64, rounds: u32, out: Vec<usize>, held: usize }

/// One trial of `lanes` lanes; returns lane 0's first two tokens.
fn trial(p: &mut Policy, arm: Arm, lanes: usize, argmax_drafts: bool, t: u64) -> (usize, usize) {
    let alpha = super::alpha() as f64;
    let mut ls: Vec<LaneSim> = (0..lanes).map(|l| LaneSim { seed: super::vocab::mix(t.wrapping_mul(0x100) + l as u64 + 1), x: X0, pos: 100, rounds: 0, out: Vec::new(), held: DEPTH }).collect();
    while ls[0].out.len() < 2 {
        // the drafts and their confidences
        let mut d = vec![[0usize; DEPTH]; lanes];
        let mut z = vec![vec![0f32; DEPTH]; lanes];
        for (l, s) in ls.iter().enumerate() {
            let mut prev = s.x;
            for j in 0..DEPTH {
                z[l][j] = z_of(prev);
                d[l][j] = if argmax_drafts { (0..3).max_by(|&a, &b| P[prev][a].total_cmp(&P[prev][b])).unwrap() }
                          else { draw(&P[prev], key(draft_seed(s.seed, s.rounds), counter(s.pos + 1 + j as i64, 0))) };
                prev = d[l][j];
            }
        }
        // the counts
        let cs: Vec<usize> = match arm {
            Arm::BiasedControl => d.iter().map(|dl| if dl[0] == 0 { 1 } else { 0 }).collect(),
            // the exact first stop, except in ~3 % of rounds (an independent keyed coin) where draft 1 is verified iff it is token 0
            Arm::SlightlyBiased => {
                let c = p.choose_conf(&z, DEPTH).expect("priced").c;
                d.iter().zip(&ls).map(|(dl, s)| if uniform(s.seed ^ 0xC0FFEE, s.rounds as u64) < 0.03 { usize::from(dl[0] == 0) } else { c }).collect()
            }
            Arm::Policy(_, a) if lanes >= 2 && a == Alloc::Lanes => {
                let held: Vec<usize> = ls.iter().map(|s| s.held).collect();
                p.choose_lanes(&z, DEPTH, &held).expect("priced").c
            }
            Arm::Policy(..) => vec![p.choose_conf(&z, DEPTH).expect("priced").c; lanes],
        };
        // the verify of each lane (host twin of accept / residual / bonus)
        for (l, s) in ls.iter_mut().enumerate() {
            let c = cs[l];
            let mut prev = s.x;
            let mut emitted = Vec::new();
            let mut rejected = false;
            for j in 0..c {
                let tp = s.pos + 1 + j as i64;
                let (q, dj) = (&Q[prev], d[l][j]);
                let pd: [f64; 3] = if argmax_drafts { std::array::from_fn(|v| if v == dj { 1.0 } else { 0.0 }) } else { P[prev] };
                if uniform(s.seed, counter(tp, 2)) < (q[dj] / pd[dj]).min(1.0) { emitted.push(dj); prev = dj; continue; }
                let rr: [f64; 3] = std::array::from_fn(|v| (q[v] - alpha * pd[v]).max(0.0));
                let tok = if rr.iter().sum::<f64>() > 0.0 { draw(&rr, key(s.seed, counter(tp, 1))) } else { (0..3).max_by(|&a, &b| q[a].total_cmp(&q[b])).unwrap() };
                emitted.push(tok);
                rejected = true;
                break;
            }
            if !rejected { emitted.push(draw(&Q[prev], key(s.seed, counter(s.pos + 1 + c as i64, 0)))); }
            s.pos += emitted.len() as i64;
            s.x = *emitted.last().unwrap();
            s.out.extend(emitted);
            s.rounds += 1;
            if matches!(arm, Arm::Policy(_, Alloc::Lanes)) { s.held = alloc::next_held(DEPTH, lanes, c); }
        }
    }
    (ls[0].out[0], ls[0].out[1])
}

/// A policy pricing rounds from a table of verify costs by drafts per lane (no draft-step cost, no overhead).
fn policy(table: &[f64; 6], lanes_max: usize, stop: Stop, alloc: Alloc) -> Policy {
    let mut c = Costs::default();
    for l in 1..=lanes_max { for (k, ms) in table.iter().enumerate() { c.vk.insert((l, k), *ms); } }
    let mut p = Policy::with_rule(Mode::Auto, false, Rule::Conf, c);
    for l in 1..=lanes_max { p.overhead.insert(l, 0.0); }
    p.stop = stop;
    p.alloc = alloc;
    p.max_rows = 16;
    p
}

/// (chi-square p of lane 0's first two tokens against the target, the share of trials whose first round verified a draft)
fn run(table: &[f64; 6], lanes: usize, arm: Arm, argmax: bool, salt: u64) -> f64 { run_n(table, lanes, arm, argmax, salt, N) }
fn run_n(table: &[f64; 6], lanes: usize, arm: Arm, argmax: bool, salt: u64, n: usize) -> f64 {
    let (stop, al) = match arm { Arm::Policy(s, a) => (s, a), _ => (Stop::First, Alloc::Shared) };
    let mut p = policy(table, lanes, stop, al);
    let mut h: HashMap<i32, usize> = HashMap::new();
    for t in 0..n as u64 { let (a, b) = trial(&mut p, arm, lanes, argmax, t ^ (salt << 40)); *h.entry((a * 3 + b) as i32).or_default() += 1; }
    let probs: Vec<(u32, f64)> = (0..9).map(|i| (i as u32, Q[X0][i / 3] * Q[i / 3][i % 3])).collect();
    super::chisq::one_sample(&h, &probs).2
}

/// The concave table: one draft alone does not pay from the start token (chance 0.15: 1.15 / 12 < 1 / 10), two do when
/// draft 2's chance is high (1 + 0.15 + 0.1485) / 12.2 > 1 / 10, and draft 2's chance is high only after draft 1 = token 0.
const CONCAVE: [f64; 6] = [10.0, 12.0, 12.2, 30.0, 40.0, 50.0];
/// A line: drafts pay often, and whether draft 2 is verified depends on draft 1 (an allowed dependence: draft 1 is part of
/// the prefix being verified when draft 2 is decided).
const LINE: [f64; 6] = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0];

#[test]
fn dsv41_ps3_exactness_one_lane() {
    let scan = run(&CONCAVE, 1, Arm::Policy(Stop::Scan, Alloc::Lanes), false, 1);
    assert!(scan < 1e-6, "PS1 as shipped (scan) on a concave table must be rejected: p {scan}");
    let exact = run(&CONCAVE, 1, Arm::Policy(Stop::First, Alloc::Lanes), false, 2);
    assert!(exact > 1e-3, "the exact stop rule, concave table: p {exact}");
    let exact_line = run(&LINE, 1, Arm::Policy(Stop::First, Alloc::Lanes), false, 3);
    assert!(exact_line > 1e-3, "the exact stop rule, line table: p {exact_line}");
    for (t, salt) in [(&CONCAVE, 7u64), (&LINE, 8)] {
        let la = run(t, 1, Arm::Policy(Stop::Lookahead, Alloc::Lanes), false, salt);
        assert!(la > 1e-3, "the exact look-ahead: p {la}");
    }
    let scan_line = run(&LINE, 1, Arm::Policy(Stop::Scan, Alloc::Lanes), false, 4);
    assert!(scan_line > 1e-3, "on a line the scan picks what the stop rule picks: p {scan_line}");
    let control = run(&LINE, 1, Arm::BiasedControl, false, 5);
    assert!(control < 1e-6, "the control (verify draft 1 only when it is token 0) must be rejected: p {control}");
    // the reference recipe's drafts are the drafter's argmax (a function of the context alone): there even the scan is exact
    let argmax_scan = run(&CONCAVE, 1, Arm::Policy(Stop::Scan, Alloc::Lanes), true, 6);
    assert!(argmax_scan > 1e-3, "argmax drafts under the scan: p {argmax_scan}");
}

#[test]
fn dsv41_ps3_exactness_several_lanes() {
    let scan = run(&CONCAVE, 2, Arm::Policy(Stop::Scan, Alloc::Lanes), false, 11);
    assert!(scan < 1e-6, "per-lane allocation with the reference's best-prefix scan, concave: p {scan}");
    let exact = run(&CONCAVE, 2, Arm::Policy(Stop::First, Alloc::Lanes), false, 12);
    assert!(exact > 1e-3, "per-lane allocation, exact stop, concave: p {exact}");
    for lanes in [2usize, 3] {
        let e = run(&LINE, lanes, Arm::Policy(Stop::First, Alloc::Lanes), false, 13 + lanes as u64);
        assert!(e > 1e-3, "per-lane allocation, exact stop, line, {lanes} lanes: p {e}");
        let s = run(&LINE, lanes, Arm::Policy(Stop::First, Alloc::Shared), false, 23 + lanes as u64);
        assert!(s > 1e-3, "one c for all lanes, exact stop, line, {lanes} lanes: p {s}");
    }
    let shared_scan = run(&CONCAVE, 2, Arm::Policy(Stop::Scan, Alloc::Shared), false, 31);
    assert!(shared_scan < 1e-6, "one c for all lanes with the scan, concave: p {shared_scan}");
}

/// The construction exercises verification (not only c = 0): on the line table the exact rule verifies drafts in most
/// first rounds, and verifies draft 2 only after draft 1 = token 0 (the allowed dependence on the prefix).
#[test]
fn dsv41_ps3_exactness_arms_verify_drafts() {
    let mut p = policy(&LINE, 1, Stop::First, Alloc::Lanes);
    let z = |d1: usize| vec![vec![z_of(X0), z_of(d1), z_of(0)]];
    assert_eq!(p.choose_conf(&z(0), DEPTH).unwrap().c, 3);
    assert_eq!(p.choose_conf(&z(1), DEPTH).unwrap().c, 1);
    let mut s = policy(&CONCAVE, 1, Stop::Scan, Alloc::Lanes);
    assert_eq!(s.choose_conf(&z(0), DEPTH).unwrap().c, 2, "the scan verifies draft 1 because draft 2's chance (from draft 1) is high");
    assert_eq!(s.choose_conf(&z(1), DEPTH).unwrap().c, 0);
    let mut e = policy(&CONCAVE, 1, Stop::First, Alloc::Lanes);
    assert_eq!(e.choose_conf(&z(0), DEPTH).unwrap().c, 0);
    assert!(e.choose_conf(&z(0), DEPTH).unwrap().scan_differs);
}

/// The hard case at 0.1 % resolution (N_HARD = 4,000,000 trials an arm): the concave table on which the scan is biased;
/// the exact first stop and the exact look-ahead at one lane, the per-lane allocation at two lanes, must pass; the scan and
/// a rule that reads its draft in only ~3 % of rounds must be rejected.
#[test]
fn dsv41_ps3_exactness_hard_case_0p1_percent() {
    let scan = run_n(&CONCAVE, 1, Arm::Policy(Stop::Scan, Alloc::Lanes), false, 41, N_HARD);
    assert!(scan < 1e-6, "scan: p {scan}");
    let slight = run_n(&CONCAVE_PAYS, 1, Arm::SlightlyBiased, false, 42, N_HARD);
    assert!(slight < 1e-6, "a 3 %-of-rounds draft-reading rule must be rejected at this N: p {slight}");
    for (arm, lanes, t, salt) in [(Arm::Policy(Stop::First, Alloc::Lanes), 1, &CONCAVE_PAYS, 43u64), (Arm::Policy(Stop::Lookahead, Alloc::Lanes), 1, &CONCAVE, 44),
                                  (Arm::Policy(Stop::Lookahead, Alloc::Lanes), 1, &CONCAVE_PAYS, 45), (Arm::Policy(Stop::First, Alloc::Lanes), 2, &CONCAVE_PAYS, 46)] {
        let pv = run_n(t, lanes, arm, false, salt, N_HARD);
        assert!(pv > 1e-3, "{arm:?} {lanes} lane(s): p {pv}");
    }
}

/// A concave table on which the exact rules DO verify drafts from the start token (draft 1 alone pays: 1.15 / 10.9 > 1 / 10)
/// and the scan still looks past a concave step (rows 2 -> 3 cost 0.2 ms after a 0.9 ms step).
const CONCAVE_PAYS: [f64; 6] = [10.0, 10.9, 12.0, 12.2, 30.0, 40.0];

/// GATE (the property every count rule must hold, RVX section 8.1): the decision "verify draft j" must not change when the
/// chances of drafts j + 1.. (the only quantities that read draft j's draw) are replaced by any other non-increasing
/// tail. Checked over random chances and random, concave and real cached tables (p1 / pm1 medians: 4 rows cheaper than 2
/// or 3), for the exact first stop, the exact look-ahead (one lane and one c for 2-3 lanes) and the per-lane allocation;
/// the scan (one lane, shared c) and the reference's best prefix are the controls that MUST fail it.
#[test]
fn dsv41_ps3_gate_decision_never_reads_later_chances() {
    let mut x: u64 = 0x5EED_6A7E;
    let mut u = || { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (x >> 11) as f64 / (1u64 << 53) as f64 };
    let real: [[f64; 6]; 3] = [[36.0, 43.8, 49.2, 40.9, 42.3, 46.9], [38.5, 45.4, 46.9, 36.9, 40.3, 45.1], [52.0, 59.9, 64.6, 65.2, 65.8, 66.4]];
    let chains = |n: usize, u: &mut dyn FnMut() -> f64| -> Vec<f64> { let mut c = 1.0; (0..n).map(|_| { c *= u().powf(0.4); c }).collect() };
    let (mut fail_scan, mut fail_ref) = (0usize, 0usize);
    for it in 0..6000 {
        let table: [f64; 6] = match it % 4 {
            0 => real[(it / 4) % 3],
            1 => { let a = 20.0 + 30.0 * u(); let b = 0.5 + 6.0 * u(); std::array::from_fn(|c| a + b * c as f64) }
            _ => { let mut t = [20.0 + 20.0 * u(); 6]; for c in 1..6 { t[c] = t[c - 1] + 8.0 * u() * u(); } t }   // random steps: concave pieces
        };
        let lanes = 1 + it % 3;
        let ch: Vec<Vec<f64>> = (0..lanes).map(|_| chains(DEPTH + 2, &mut u)).collect();
        let z: Vec<Vec<f32>> = ch.iter().map(|v| { let mut prev = 1.0; v.iter().map(|&s| { let r = (s / prev).clamp(1e-9, 1.0 - 1e-9); prev = s; logit(r) }).collect() }).collect();
        let depth = DEPTH + 2;
        for (stop, must_hold) in [(Stop::First, true), (Stop::Lookahead, true), (Stop::Scan, false)] {
            let mut p = policy(&table, lanes, stop, Alloc::Shared);
            p.max_rows = 64;
            let c0 = p.choose_conf(&z, depth).unwrap().c;
            for j in 1..=depth {
                // replace the chances of drafts j + 1.. (their conditional chances) with a random non-increasing tail
                let mut z2 = z.clone();
                for zl in z2.iter_mut() { for i in j..depth { zl[i] = logit(u().clamp(1e-6, 1.0 - 1e-6)); } }
                let c1 = p.choose_conf(&z2, depth).unwrap().c;
                if (c0 >= j) != (c1 >= j) {
                    assert!(!must_hold, "{stop:?} {lanes} lane(s), table {table:?}: verify draft {j} changed with later chances ({c0} -> {c1})");
                    fail_scan += 1;
                }
            }
        }
        if lanes >= 2 {
            // the per-lane allocation: lane l's decision on draft j against lane l's tail; the reference best prefix must fail
            let price = |d: usize| { let (c0, r) = (d / lanes, d % lanes); if c0 + 1 >= 6 { Some(table[5] + 2.0 * d as f64) } else { Some(table[c0] + (table[c0 + 1] - table[c0]) * r as f64 / lanes as f64) } };
            let a = alloc::allocate(&ch, price, 64).unwrap();
            let al = alloc::allocate_with_lookahead(&ch, price, 64, Some(super::policy::LOOKAHEAD_M)).unwrap();
            for l in 0..lanes { for j in 1..=depth {
                let mut ch2 = ch.clone();
                let mut c = ch2[l][j - 1];
                for i in j..depth { c *= u(); ch2[l][i] = c; }
                let b = alloc::allocate(&ch2, price, 64).unwrap();
                let bl = alloc::allocate_with_lookahead(&ch2, price, 64, Some(super::policy::LOOKAHEAD_M)).unwrap();
                assert_eq!(a.counts[l] >= j, b.counts[l] >= j, "per-lane allocation, lane {l} draft {j}: {:?} -> {:?}", a.counts, b.counts);
                assert_eq!(al.counts[l] >= j, bl.counts[l] >= j, "per-lane lookahead allocation, lane {l} draft {j}: {:?} -> {:?}", al.counts, bl.counts);
                if (a.scan_counts[l] >= j) != (b.scan_counts[l] >= j) { fail_ref += 1; }
            } }
        }
    }
    assert!(fail_scan > 0, "the control: the scan must fail the gate");
    assert!(fail_ref > 0, "the control: the reference's best prefix must fail the gate");
}

/// Leg PS4: the decision-sensitive bits the PIT log carries. `scan_differs` marks exactly the rounds where the scan and the
/// exact rule part, in EVERY mode (with the measurement scan on it still compares against the exact count); `scan_c` is the
/// scan's count; `sensitive` marks a decision one draft from flipping. The gate-only jump (a synthetic fixed cost on every
/// round with a draft) makes the two rules part where the plain line never does.
#[test]
fn dsv41_ps4_decision_bits_and_the_gate_jump() {
    let mut x: u64 = 0xBADC_0FFE;
    let mut u = || { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (x >> 11) as f64 / (1u64 << 53) as f64 };
    let line = [38.5, 42.5, 46.5, 50.5, 54.5, 58.5];
    let mut parted = [0usize; 2];
    for (ji, jump) in [0.0f64, 32.0].iter().enumerate() {
        let mut exact = policy(&line, 1, Stop::Lookahead, Alloc::Lanes);
        let mut scan = policy(&line, 1, Stop::Scan, Alloc::Lanes);
        exact.gate_jump_ms = *jump;
        scan.gate_jump_ms = *jump;
        for _ in 0..4000 {
            let z: Vec<Vec<f32>> = vec![(0..5).map(|_| (0.3 + 1.6 * (u() * 2.0 - 1.0) * 1.7) as f32).collect()];
            let a = exact.choose_conf(&z, 5).unwrap();
            let b = scan.choose_conf(&z, 5).unwrap();
            assert_eq!(a.scan_c, b.c, "the exact rule reports the scan's count");
            assert_eq!(b.scan_c, b.c);
            assert_eq!(a.scan_differs, a.c != b.c);
            assert_eq!(b.scan_differs, a.c != b.c, "the scan mode still marks where the exact rule would have chosen otherwise");
            if a.c != b.c { parted[ji] += 1; }
            // the jump is in the price of every count with a draft, never in the count without
            if a.c > 0 { assert!(a.ms >= line[a.c] + jump - 1e-9); } else { assert!((a.ms - line[0]).abs() < 1e-9); }
        }
    }
    assert_eq!(parted[0], 0, "a line: the rules never part");
    assert!(parted[1] > 100, "the gate jump makes them part: {} of 4000", parted[1]);
    // sensitive: a decision with a neighbour within 2 % of the chosen count's rate
    let mut p = policy(&[10.0, 10.0 * 1.15 / 1.0 * 1.0, 30.0, 40.0, 50.0, 60.0], 1, Stop::First, Alloc::Lanes);
    let logit = |q: f64| ((q / (1.0 - q)).ln()) as f32;
    let ch = p.choose_conf(&[vec![logit(0.149), -30.0, -30.0, -30.0, -30.0]], 5).unwrap();
    assert!(ch.c == 0 && ch.sensitive, "{ch:?}");
    let ch = p.choose_conf(&[vec![logit(0.01), -30.0, -30.0, -30.0, -30.0]], 5).unwrap();
    assert!(ch.c == 0 && !ch.sensitive, "{ch:?}");
    // the per-lane allocation's margin: a step that barely pays is sensitive
    let near = alloc::allocate(&[vec![0.1505, 0.0]], |d| Some([10.0, 11.5, 50.0][d]), 6).unwrap();
    assert!(near.margin < alloc::SENSITIVE_GAP && near.counts == vec![1], "{near:?}");
    let far = alloc::allocate(&[vec![0.9, 0.0]], |d| Some([10.0, 11.5, 50.0][d]), 6).unwrap();
    assert!(far.margin > alloc::SENSITIVE_GAP, "{far:?}");
}
