//! The sampled-verification gate (plan Revision 4; AGENTS §3: "gate distribution-exactness with a reference-free
//! two-sample chi-square, always draw a control, vary the RNG stream across conditions").
//!
//! For each cell: k + 1 target rows q_j and k draft rows p_j (synthetic, or REFb's real target rows with a
//! perturbed draft). Every trial is one lane of a batched round with its own seed and its own positions (the
//! streams vary across trials and conditions). The drafts are drawn from p_j by the drafter's sampler (draft
//! namespace), the verify runs DeepSeek's rule on the GPU (`Round::verify`: accept u < min(1, q / p), residual
//! max(q - alpha p, 0), bonus row), and the emitted token at position j is collected. The control is a direct
//! draw from q_j (`Round::stats`, another seed) for every trial. The gate: for every position reached by enough
//! trials, the two-sample chi-square of (emitted, control) has p >= `P_MIN` (the speculative chain emits the target
//! distribution; alpha's bias <= 5e-5 is far below the test's resolution). Negative controls (the swapped rule, no
//! residual) must FAIL with p < `P_NEG` somewhere: the test has power. Greedy cells are exact (the argmax chain).
//! A one-sample fit of the control against the exact q (host f64) checks the Gumbel sampler itself, and a 1-vs-4
//! shard run checks that a vocab split merges to the same records (TP invariance of the draws).

use super::vocab::{self, Cut, Round, Rows, Shard};
use super::xchg::Local;
use super::{Lane, OUT_TOK, TEST_NO_RESIDUAL, TEST_SWAP_RULE};
use crate::dsv41::fwd::dev::{Buf, Gpu};
use crate::dsv41::serve::sampler::{kept_set, Sampling};
use anyhow::{ensure, Result};

/// The per-position pass threshold of the two-sample test (fixed seeds: the verdict is reproducible).
pub const P_MIN: f64 = 1e-4;
/// A negative control must reach p below this at some position.
pub const P_NEG: f64 = 1e-6;
/// Positions reached by fewer trials are reported, not tested.
pub const MIN_N: usize = 400;

#[derive(Clone, Debug)]
pub struct Cell {
    pub name: &'static str,
    pub k: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub trials: usize,
    pub batch: usize,
    pub test_flags: i32,
    pub expect_pass: bool,
}

/// Host RNG: splitmix64 stream with normal deviates.
struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 { self.0 = self.0.wrapping_add(1); ((vocab::mix(self.0) >> 11) as f64 + 0.5) / (1u64 << 53) as f64 }
    fn normal(&mut self) -> f64 { let (a, b) = (self.u(), self.u()); (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos() }
}

/// Synthetic rows: target logits N(0, sigma) per token (a few hot tokens), draft = target + N(0, pert).
pub fn synth_rows(v: usize, rows: usize, sigma: f64, pert: f64, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut r = Rng(seed);
    let mut q = Vec::with_capacity(rows * v);
    let mut p = Vec::with_capacity(rows * v);
    for _ in 0..rows {
        for _ in 0..v {
            let a = r.normal() * sigma;
            q.push(a as f32);
            p.push((a + r.normal() * pert) as f32);
        }
    }
    (q, p)
}

/// The cut of `row` under `s` (the last kept element in the (a desc, id asc) order), or None when nothing is cut.
pub fn host_cut(row: &[f32], s: &Sampling) -> Option<(f32, i32)> {
    if s.top_k == 0 && s.top_p >= 1.0 { return None; }
    let kept = kept_set(row, s, &[]);
    let ca = kept.iter().map(|&(id, _)| row[id as usize]).fold(f32::INFINITY, f32::min);
    let cid = kept.iter().filter(|&&(id, _)| row[id as usize] == ca).map(|&(id, _)| id as i32).max()?;
    Some((ca, cid))
}

/// Upper tail of chi-square(df) at x: Q(df / 2, x / 2) (regularized upper incomplete gamma).
pub fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 { return 1.0; }
    let (a, z) = (df / 2.0, x / 2.0);
    let gln = ln_gamma(a);
    if z < a + 1.0 {
        let (mut sum, mut del, mut ap) = (1.0 / a, 1.0 / a, a);
        for _ in 0..10_000 { ap += 1.0; del *= z / ap; sum += del; if del.abs() < sum.abs() * 1e-15 { break; } }
        1.0 - (sum * (-z + a * z.ln() - gln).exp())
    } else {
        let (mut b, mut c, mut d) = (z + 1.0 - a, 1.0 / 1e-300, 1.0 / (z + 1.0 - a));
        let mut h = d;
        for i in 1..10_000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b; if d.abs() < 1e-300 { d = 1e-300; }
            c = b + an / c; if c.abs() < 1e-300 { c = 1e-300; }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 { break; }
        }
        (-z + a * z.ln() - gln).exp() * h
    }
}
fn ln_gamma(x: f64) -> f64 {
    let c = [76.180_091_729_471_46, -86.505_320_329_416_77, 24.014_098_240_830_91, -1.231_739_572_450_155, 0.001_208_650_973_866_179, -0.000_005_395_239_384_953];
    let mut y = x;
    let tmp = x + 5.5 - (x + 0.5) * (x + 5.5).ln();
    let mut ser = 1.000_000_000_190_015;
    for ci in c { y += 1.0; ser += ci / y; }
    -tmp + (2.506_628_274_631_000_5 * ser / x).ln()
}

/// Two-sample chi-square of two count maps (bins with >= 10 combined counts, the rest pooled): (statistic, df, p).
pub fn two_sample(a: &std::collections::HashMap<i32, usize>, b: &std::collections::HashMap<i32, usize>) -> (f64, usize, f64) {
    let (n1, n2) = (a.values().sum::<usize>() as f64, b.values().sum::<usize>() as f64);
    if n1 == 0.0 || n2 == 0.0 { return (0.0, 0, 1.0); }
    let (k1, k2) = ((n2 / n1).sqrt(), (n1 / n2).sqrt());
    let mut keys: Vec<i32> = a.keys().chain(b.keys()).copied().collect();
    keys.sort_unstable();
    keys.dedup();
    let (mut stat, mut bins, mut pa, mut pb) = (0.0, 0usize, 0usize, 0usize);
    for t in keys {
        let (x, y) = (*a.get(&t).unwrap_or(&0), *b.get(&t).unwrap_or(&0));
        if x + y >= 10 { stat += (k1 * x as f64 - k2 * y as f64).powi(2) / (x + y) as f64; bins += 1; } else { pa += x; pb += y; }
    }
    if pa + pb >= 10 { stat += (k1 * pa as f64 - k2 * pb as f64).powi(2) / (pa + pb) as f64; bins += 1; }
    let df = bins.saturating_sub(1);
    (stat, df, if df == 0 { 1.0 } else { chi2_sf(stat, df as f64) })
}

/// One-sample fit of counts against exact probabilities (bins with expected >= 10, the rest pooled): (stat, df, p).
pub fn one_sample(c: &std::collections::HashMap<i32, usize>, probs: &[(u32, f64)]) -> (f64, usize, f64) {
    let n = c.values().sum::<usize>() as f64;
    let (mut stat, mut bins, mut po, mut pe) = (0.0, 0usize, 0.0, 0.0);
    let mut seen = 0usize;
    for &(id, p) in probs {
        let o = *c.get(&(id as i32)).unwrap_or(&0) as f64;
        seen += o as usize;
        let e = p * n;
        if e >= 10.0 { stat += (o - e).powi(2) / e; bins += 1; } else { po += o; pe += e; }
    }
    po += n - seen as f64;   // draws outside the kept set (must be none)
    if pe >= 10.0 || po > 0.0 { let e = pe.max(1e-9); stat += (po - e).powi(2) / e; bins += 1; }
    let df = bins.saturating_sub(1);
    (stat, df, if df == 0 { 1.0 } else { chi2_sf(stat, df as f64) })
}

type Counts = Vec<std::collections::HashMap<i32, usize>>;

/// Run one cell over rows q [k + 1][v] and p [k][v] (broadcast to every trial). Returns (emitted, control, mean accepted).
fn run_cell(g: &Gpu, c: &Cell, v: usize, q: &[f32], p: &[f32], seed0: u64) -> Result<(Counts, Counts, f64)> {
    let rpl = c.k + 1;
    ensure!(q.len() == rpl * v && p.len() >= c.k * v, "cell {}: rows", c.name);
    let sh = Shard { v0: 0, vlocal: v };
    let x = Local { max_bytes: c.batch * 8 * vocab::REC };
    let rd = Round::new(g, c.batch, rpl, sh.nblk(), 1)?;
    let (dq, dp) = (g.buf(q.len() * 4)?, g.buf(super::BLOCK * v * 4)?);
    g.h2d(dq.ptr(), vocab::as_bytes(q))?;
    g.h2d(dp.ptr(), vocab::as_bytes(&p[..c.k * v]))?;
    let s = Sampling { temperature: c.temperature, top_p: c.top_p, top_k: c.top_k, ..Default::default() };
    let greedy = c.temperature <= crate::dsv41::serve::sampler::GREEDY_T;
    // per-row cuts (host, SV's kept_set semantics), replicated per lane: [batch][rpl] / [batch][5]
    let mk_cut = |rows: &[f32], n: usize, stride: usize| -> Result<Option<(Buf, Buf)>> {
        let cuts: Vec<Option<(f32, i32)>> = (0..n).map(|j| host_cut(&rows[j * v..(j + 1) * v], &s)).collect();
        if cuts.iter().all(|x| x.is_none()) { return Ok(None); }
        let mut ca = vec![f32::NEG_INFINITY; c.batch * stride];
        let mut ci = vec![i32::MAX; c.batch * stride];
        for l in 0..c.batch { for (j, cu) in cuts.iter().enumerate() { if let Some((a, i)) = cu { ca[l * stride + j] = *a; ci[l * stride + j] = *i; } } }
        let (ba, bi) = (g.buf(ca.len() * 4)?, g.buf(ci.len() * 4)?);
        g.h2d(ba.ptr(), vocab::as_bytes(&ca))?;
        g.h2d(bi.ptr(), vocab::as_bytes(&ci))?;
        Ok(Some((ba, bi)))
    };
    let tcut = mk_cut(q, rpl, rpl)?;
    let dcut = mk_cut(p, c.k, super::BLOCK)?;
    let tc = tcut.as_ref().map(|(a, i)| Cut { a: a.ptr(), id: i.ptr(), rpl });
    let dcc = dcut.as_ref().map(|(a, i)| Cut { a: a.ptr(), id: i.ptr(), rpl: super::BLOCK });
    let mut emitted: Counts = vec![Default::default(); rpl];
    let mut control: Counts = vec![Default::default(); rpl];
    let (mut acc_sum, mut rounds) = (0usize, 0usize);
    let mut done = 0usize;
    let mut bi = 0u64;
    while done < c.trials {
        let n = (c.trials - done).min(c.batch);
        // a fresh seed and fresh positions per trial (the streams vary across trials, batches and cells)
        let lanes: Vec<Lane> = (0..n).map(|l| {
            let sd = vocab::mix(seed0 ^ vocab::mix(bi * 1_000_003 + l as u64));
            let vb = 1000 + ((sd >> 40) as i32 & 0xFFFF);
            Lane { seed: sd, inv_t: if greedy { 0.0 } else { 1.0 / c.temperature }, vbase: vb, k: c.k as i32, wend: vb - 1, tok: 7, active: 1, ..Default::default() }
        }).collect();
        rd.set_lanes(g, &lanes)?;
        if c.k > 0 {
            rd.draft_pick(g, &x, sh, Rows { ptr: dp.ptr(), lane_stride: 0, row_stride: v }, n, 0, c.k, dcc.map(|d| Cut { rpl: super::BLOCK, ..d }))?;
        }
        rd.verify(g, &x, sh, Rows { ptr: dq.ptr(), lane_stride: 0, row_stride: v }, Rows { ptr: dp.ptr(), lane_stride: 0, row_stride: v }, n, tc, dcc, c.test_flags)?;
        let out = rd.out_tokens(g, n)?;
        let ls = rd.get_lanes(g, n)?;
        for l in 0..n {
            let ne = ls[l].n_emit as usize;
            ensure!(ne >= 1 && ne <= rpl, "cell {}: lane {l} emitted {ne}", c.name);
            for j in 0..ne { *emitted[j].entry(out[l][j]).or_insert(0) += 1; }
            acc_sum += ls[l].n_acc as usize;
            rounds += 1;
        }
        // the control: direct draws from q_j with another seed (stream 0 at the same kind of positions)
        let ctl: Vec<Lane> = lanes.iter().map(|ln| Lane { seed: ln.seed ^ 0x5DEE_CE66_D1CE_4E5B, ..*ln }).collect();
        rd.set_lanes(g, &ctl)?;
        let sres = rd.stats(g, &x, sh, Rows { ptr: dq.ptr(), lane_stride: 0, row_stride: v }, n, rpl, tc)?;
        let recs = rd.recs(g, sres, n * rpl)?;
        for l in 0..n { for j in 0..rpl { let r = &recs[l * rpl + j]; *control[j].entry(if greedy { r.aid } else { r.gid }).or_insert(0) += 1; } }
        done += n;
        bi += 1;
    }
    let _ = OUT_TOK;
    Ok((emitted, control, acc_sum as f64 / rounds.max(1) as f64))
}

/// The cells and their verdicts. `real` = REFb target rows (k + 1 consecutive decode positions, full vocab).
pub fn run(g: &Gpu, real: Option<Vec<f32>>, lines: &mut Vec<String>, fails: &mut Vec<String>) -> Result<()> {
    let v = 4096;
    let cells = [
        Cell { name: "T=1 k=5", k: 5, temperature: 1.0, top_k: 0, top_p: 1.0, trials: 40_000, batch: 2048, test_flags: 0, expect_pass: true },
        Cell { name: "T=0.7 k=5", k: 5, temperature: 0.7, top_k: 0, top_p: 1.0, trials: 40_000, batch: 2048, test_flags: 0, expect_pass: true },
        Cell { name: "T=1 top-k 40 k=3", k: 3, temperature: 1.0, top_k: 40, top_p: 1.0, trials: 40_000, batch: 2048, test_flags: 0, expect_pass: true },
        Cell { name: "T=1 top-p 0.9 k=5", k: 5, temperature: 1.0, top_k: 0, top_p: 0.9, trials: 40_000, batch: 2048, test_flags: 0, expect_pass: true },
        Cell { name: "T=1 k=1", k: 1, temperature: 1.0, top_k: 0, top_p: 1.0, trials: 40_000, batch: 2048, test_flags: 0, expect_pass: true },
        Cell { name: "NEG swapped rule", k: 5, temperature: 1.0, top_k: 0, top_p: 1.0, trials: 40_000, batch: 2048, test_flags: TEST_SWAP_RULE, expect_pass: false },
        Cell { name: "NEG no residual", k: 5, temperature: 1.0, top_k: 0, top_p: 1.0, trials: 40_000, batch: 2048, test_flags: TEST_NO_RESIDUAL, expect_pass: false },
    ];
    lines.push(format!("sampler chi-square: two-sample (emitted vs a direct control draw) per position, pass p >= {P_MIN:.0e}, positions with >= {MIN_N} trials; negative controls must reach p < {P_NEG:.0e}"));
    for (ci, c) in cells.iter().enumerate() {
        let (q, p) = synth_rows(v, c.k + 1, 2.5, 1.0, 0xC0DE + ci as u64);
        cell_verdict(g, c, v, &q, &p, 0xA11CE ^ (ci as u64) << 32, lines, fails)?;
    }
    // greedy: exact
    greedy_cell(g, lines, fails)?;
    batch_invariance(g, lines, fails)?;
    // real rows: REFb's target distributions at 6 consecutive decode positions (full vocabulary)
    if let Some(q) = real {
        let vr = q.len() / 6;
        let mut r = Rng(77);
        let p: Vec<f32> = q[..5 * vr].iter().map(|&a| (a as f64 + r.normal() * 0.7) as f32).collect();
        let c = Cell { name: "REAL rows (REFb decode logits, V 129,280) T=1 k=5", k: 5, temperature: 1.0, top_k: 0, top_p: 1.0, trials: 12_000, batch: 512, test_flags: 0, expect_pass: true };
        cell_verdict(g, &c, vr, &q, &p, 0xBEEF, lines, fails)?;
        tp_invariance(g, &q, vr, lines, fails)?;
    } else {
        lines.push("  real-row cell and the shard-split check: skipped (no --dsv41-dump-ref fixture with G.decode_logits)".into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cell_verdict(g: &Gpu, c: &Cell, v: usize, q: &[f32], p: &[f32], seed: u64, lines: &mut Vec<String>, fails: &mut Vec<String>) -> Result<()> {
    let t0 = std::time::Instant::now();
    let (em, ctl, acc) = run_cell(g, c, v, q, p, seed)?;
    let mut pmin = 1.0f64;
    let mut parts = Vec::new();
    for j in 0..=c.k {
        let n1: usize = em[j].values().sum();
        if n1 < MIN_N { parts.push(format!("pos{j}: n {n1} (untested)")); continue; }
        let (st, df, pv) = two_sample(&em[j], &ctl[j]);
        pmin = pmin.min(pv);
        parts.push(format!("pos{j}: n {n1} chi2 {st:.1}/df {df} p {pv:.3}"));
    }
    // the sampler itself: the control against the exact q of row 0 (host f64)
    let s = Sampling { temperature: c.temperature, top_p: c.top_p, top_k: c.top_k, ..Default::default() };
    let exact = kept_set(&q[..v], &s, &[]);
    let (fst, fdf, fp) = one_sample(&ctl[0], &exact);
    let pass = if c.expect_pass { pmin >= P_MIN && fp >= P_MIN } else { pmin < P_NEG };
    lines.push(format!("  {:<44} accepted/round {acc:.2}; {}; control vs exact q(row 0): chi2 {fst:.1}/df {fdf} p {fp:.3} -> {} ({:.1} s)", c.name, parts.join(", "),
                       if pass { if c.expect_pass { "PASS" } else { "PASS (detected)" } } else { "FAIL" }, t0.elapsed().as_secs_f64()));
    if !pass { fails.push(format!("chi-square cell {}", c.name)); }
    Ok(())
}

/// Greedy verification is exact: drafts = argmax p_j, accepted while draft == argmax q_j, then argmax q.
fn greedy_cell(g: &Gpu, lines: &mut Vec<String>, fails: &mut Vec<String>) -> Result<()> {
    let (v, k, nl) = (4096usize, 5usize, 256usize);
    let rpl = k + 1;
    let sh = Shard { v0: 0, vlocal: v };
    let x = Local { max_bytes: nl * 8 * vocab::REC };
    let rd = Round::new(g, nl, rpl, sh.nblk(), 1)?;
    // per-lane rows; p equals q on a random prefix of rows so acceptance lengths vary
    let mut r = Rng(5);
    let mut q = vec![0f32; nl * rpl * v];
    let mut p = vec![0f32; nl * super::BLOCK * v];
    for l in 0..nl {
        let same = (r.u() * 6.0) as usize;
        for j in 0..rpl { for t in 0..v { q[(l * rpl + j) * v + t] = (r.normal() * 3.0) as f32; } }
        for j in 0..k { for t in 0..v { p[(l * super::BLOCK + j) * v + t] = if j < same { q[(l * rpl + j) * v + t] } else { (r.normal() * 3.0) as f32 }; } }
    }
    let (dq, dp) = (g.buf(q.len() * 4)?, g.buf(p.len() * 4)?);
    g.h2d(dq.ptr(), vocab::as_bytes(&q))?;
    g.h2d(dp.ptr(), vocab::as_bytes(&p))?;
    let lanes: Vec<Lane> = (0..nl).map(|l| Lane { seed: l as u64, inv_t: 0.0, vbase: 100 + l as i32, k: k as i32, wend: 99 + l as i32, tok: 3, active: 1, ..Default::default() }).collect();
    rd.set_lanes(g, &lanes)?;
    rd.draft_pick(g, &x, sh, Rows { ptr: dp.ptr(), lane_stride: super::BLOCK * v, row_stride: v }, nl, 0, k, None)?;
    rd.verify(g, &x, sh, Rows { ptr: dq.ptr(), lane_stride: rpl * v, row_stride: v }, Rows { ptr: dp.ptr(), lane_stride: super::BLOCK * v, row_stride: v }, nl, None, None, 0)?;
    let out = rd.out_tokens(g, nl)?;
    let ls = rd.get_lanes(g, nl)?;
    let am = |row: &[f32]| row.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as i32;
    let mut bad = 0usize;
    let mut hist = [0usize; 6];
    for l in 0..nl {
        let mut n = 0;
        let mut want = Vec::new();
        while n < k {
            let d = am(&p[(l * super::BLOCK + n) * v..(l * super::BLOCK + n + 1) * v]);
            if d != am(&q[(l * rpl + n) * v..(l * rpl + n + 1) * v]) { break; }
            want.push(d);
            n += 1;
        }
        want.push(am(&q[(l * rpl + n) * v..(l * rpl + n + 1) * v]));
        hist[n] += 1;
        if ls[l].n_acc as usize != n || out[l][..=n] != want[..] { bad += 1; }
    }
    lines.push(format!("  {:<44} {nl} lanes, accepted-length histogram {hist:?}: {} lanes differ from the host's exact argmax chain -> {}", "GREEDY exact (k=5)", bad, if bad == 0 { "PASS" } else { "FAIL" }));
    if bad > 0 { fails.push("greedy verification".into()); }
    Ok(())
}

/// A lane's round in a 1-lane call and as lane 2 of a 4-lane call (the other lanes with other rows, seeds and
/// positions): its drafts, records, accept decisions and emitted tokens are bitwise equal (batch invariance: a
/// sampled reply does not depend on what else is in the batch).
fn batch_invariance(g: &Gpu, lines: &mut Vec<String>, fails: &mut Vec<String>) -> Result<()> {
    let (v, k) = (4096usize, 5usize);
    let rpl = k + 1;
    let sh = Shard { v0: 0, vlocal: v };
    let mut r = Rng(31);
    let (mut q, mut p) = (vec![0f32; 4 * rpl * v], vec![0f32; 4 * super::BLOCK * v]);
    for x in q.iter_mut() { *x = (r.normal() * 2.5) as f32; }
    for l in 0..4 { for j in 0..k { for t in 0..v { p[(l * super::BLOCK + j) * v + t] = q[(l * rpl + j) * v + t] + (r.normal() * 0.8) as f32; } } }
    let (dq, dp) = (g.buf(q.len() * 4)?, g.buf(p.len() * 4)?);
    g.h2d(dq.ptr(), vocab::as_bytes(&q))?;
    g.h2d(dp.ptr(), vocab::as_bytes(&p))?;
    let lane = |l: usize| Lane { seed: 1000 + l as u64 * 77, inv_t: 1.0 / 0.9, vbase: 7000 + 31 * l as i32, k: k as i32, wend: 6999 + 31 * l as i32, tok: 5, active: 1, ..Default::default() };
    let x = Local { max_bytes: 64 * rpl * vocab::REC };
    let mut res = Vec::new();
    for (n, pick) in [(1usize, 2usize), (4, 2)] {
        let rd = Round::new(g, 4, rpl, sh.nblk(), 1)?;
        let lanes: Vec<Lane> = if n == 1 { vec![lane(pick)] } else { (0..4).map(lane).collect() };
        rd.set_lanes(g, &lanes)?;
        let off = if n == 1 { pick } else { 0 };
        let qr = Rows { ptr: dq.ptr() + (off * rpl * v * 4) as u64, lane_stride: rpl * v, row_stride: v };
        let pr = Rows { ptr: dp.ptr() + (off * super::BLOCK * v * 4) as u64, lane_stride: super::BLOCK * v, row_stride: v };
        rd.draft_pick(g, &x, sh, pr, n, 0, k, None)?;
        rd.verify(g, &x, sh, qr, pr, n, None, None, 0)?;
        let li = if n == 1 { 0 } else { pick };
        let ids = rd.ids_table(g)?;
        let drafts: Vec<i32> = (1..=k).map(|j| ids[j * n + li]).collect();
        let tres = rd.recs(g, rd.tres.ptr(), n * rpl)?[li * rpl..(li + 1) * rpl].to_vec();
        let ln = rd.get_lanes(g, n)?[li];
        let out = rd.out_tokens(g, n)?[li];
        res.push((drafts, tres, ln.n_acc, ln.n_emit, out));
    }
    let same = res[0] == res[1];
    lines.push(format!("  {:<44} lane 2 alone vs in a 4-lane call: drafts {:?} / {:?}, accepted {} / {}, emitted {:?} / {:?}, target records bitwise {} -> {}",
                       "BATCH invariance (k=5, T=0.9)", res[0].0, res[1].0, res[0].2, res[1].2, &res[0].4[..res[0].3 as usize], &res[1].4[..res[1].3 as usize],
                       if res[0].1 == res[1].1 { "equal" } else { "DIFFER" }, if same { "PASS" } else { "FAIL" }));
    if !same { fails.push("batch invariance of the round".into()); }
    Ok(())
}

/// 1 shard vs 4 simulated shards of the same rows: the merged records give the same draws.
fn tp_invariance(g: &Gpu, q: &[f32], v: usize, lines: &mut Vec<String>, fails: &mut Vec<String>) -> Result<()> {
    let (nl, rpl) = (64usize, 6usize);
    let dq = g.buf(q.len() * 4)?;
    g.h2d(dq.ptr(), vocab::as_bytes(q))?;
    let lanes: Vec<Lane> = (0..nl).map(|l| Lane { seed: 900 + l as u64, inv_t: 1.0, vbase: 5000 + 7 * l as i32, k: 0, wend: 4999 + 7 * l as i32, tok: 1, active: 1, ..Default::default() }).collect();
    let rows = Rows { ptr: dq.ptr(), lane_stride: 0, row_stride: v };
    let one = Round::new(g, nl, rpl, Shard { v0: 0, vlocal: v }.nblk(), 1)?;
    one.set_lanes(g, &lanes)?;
    let a = one.recs(g, one.stats(g, &Local { max_bytes: nl * rpl * vocab::REC }, Shard { v0: 0, vlocal: v }, rows, nl, rpl, None)?, nl * rpl)?;
    let shards: Vec<Shard> = (0..4).map(|r| { let (v0, len) = super::xchg::vocab_range(v, 4, r); Shard { v0, vlocal: len } }).collect();
    let four = Round::new(g, nl, rpl, shards.iter().map(|s| s.nblk()).max().unwrap_or(1), 4)?;
    four.set_lanes(g, &lanes)?;
    let b = four.recs(g, four.stats_split(g, &shards, rows, nl, rpl)?, nl * rpl)?;
    let same_gid = a.iter().zip(&b).filter(|(x, y)| x.gid == y.gid && x.aid == y.aid).count();
    let dl = a.iter().zip(&b).map(|(x, y)| ((x.lse() - y.lse()) as f64).abs()).fold(0.0, f64::max);
    let pass = same_gid == a.len() && dl <= 1e-4;
    lines.push(format!("  {:<44} {} rows: draws and argmax identical on {same_gid}; max |lse(1 shard) - lse(4 shards)| {dl:.2e} -> {}",
                       "SHARD split 1 vs 4 (TP invariance, V 129,280)", a.len(), if pass { "PASS" } else { "FAIL" }));
    if !pass { fails.push("shard-split invariance".into()); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dsv41_ds_chi2_sf_matches_known_values() {
        // chi-square(1) at 3.841 -> 0.05; chi-square(10) at 18.307 -> 0.05; chi-square(100) at 124.342 -> 0.05
        assert!((chi2_sf(3.841_459, 1.0) - 0.05).abs() < 1e-4);
        assert!((chi2_sf(18.307_04, 10.0) - 0.05).abs() < 1e-4);
        assert!((chi2_sf(124.342, 100.0) - 0.05).abs() < 1e-3);
        assert!(chi2_sf(0.0, 5.0) == 1.0);
    }
    #[test]
    fn dsv41_ds_two_sample_detects_a_shift_and_accepts_equal_draws() {
        let mut r = Rng(1);
        let mut a = std::collections::HashMap::new();
        let mut b = std::collections::HashMap::new();
        let mut c = std::collections::HashMap::new();
        for _ in 0..20_000 {
            *a.entry((r.u() * 20.0) as i32).or_insert(0) += 1;
            *b.entry((r.u() * 20.0) as i32).or_insert(0) += 1;
            *c.entry((r.u().powf(1.2) * 20.0) as i32).or_insert(0) += 1;
        }
        assert!(two_sample(&a, &b).2 > 1e-3);
        assert!(two_sample(&a, &c).2 < 1e-6);
    }
    #[test]
    fn dsv41_ds_host_cut_matches_kept_set() {
        let row = [3.0f32, 1.0, 3.0, 2.0, 3.0];
        let s = Sampling { top_k: 2, ..Default::default() };
        assert_eq!(host_cut(&row, &s), Some((3.0, 2)));   // ids 0 and 2 kept: tie at 3.0 keeps the lowest ids
        assert_eq!(host_cut(&row, &Sampling::default()), None);
    }
}
