//! Pure-CPU paired statistics for the tuner: Hodges-Lehmann estimators, the Wilcoxon critical value, the sequential `race` and `width_split`.
//! See `mod.rs` for the file map.

// ---------------------------------------------------------------------------------------------
// Statistics (§6.3): Hodges–Lehmann estimates with distribution-free 95% intervals
// ---------------------------------------------------------------------------------------------

/// An estimate with its 95% interval (ms or µs — the caller's unit).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Est {
    pub n: usize,
    pub est: f64,
    pub lo: f64,
    pub hi: f64,
}

pub(super) fn median_sorted(v: &[f64]) -> f64 {
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

/// Largest k with P(T+ <= k) <= alpha/2 under H0 for the Wilcoxon signed-rank statistic of n
/// pairs (exact DP for n <= 300, normal approximation above). None when no such k exists (n tiny).
pub fn wilcoxon_crit(n: usize, alpha: f64) -> Option<usize> {
    if n == 0 { return None; }
    let nn = n * (n + 1) / 2;
    if n <= 300 {
        // p[k] = P(T+ = k) after adding ranks 1..i (each rank in or out with prob 1/2)
        let mut p = vec![0f64; nn + 1];
        p[0] = 1.0;
        for i in 1..=n {
            let top = i * (i + 1) / 2;
            for k in (0..=top).rev() {
                let with = if k >= i { p[k - i] } else { 0.0 };
                p[k] = 0.5 * (p[k] + with);
            }
        }
        let mut cdf = 0f64;
        let mut best: Option<usize> = None;
        for (k, &pk) in p.iter().enumerate() {
            cdf += pk;
            if cdf <= alpha / 2.0 + 1e-12 { best = Some(k); } else { break; }
        }
        best
    } else {
        let z = 1.959963984540054;
        let sd = ((n * (n + 1) * (2 * n + 1)) as f64 / 24.0).sqrt();
        let k = (nn as f64 / 2.0 - z * sd - 0.5).floor();
        if k < 0.0 { None } else { Some(k as usize) }
    }
}

/// One-sample (paired) Hodges–Lehmann estimate = median of the Walsh averages (d_i + d_j)/2, i <= j,
/// with the Wilcoxon signed-rank 95% interval [W_(k+1), W_(N-k)] (k = wilcoxon_crit).
pub fn hl_paired(d: &[f64]) -> Option<Est> {
    let n = d.len();
    if n == 0 || d.iter().any(|x| !x.is_finite()) { return None; }
    let mut w: Vec<f64> = Vec::with_capacity(n * (n + 1) / 2);
    for i in 0..n {
        for j in i..n { w.push(0.5 * (d[i] + d[j])); }
    }
    w.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let est = median_sorted(&w);
    let nn = w.len();
    let (lo, hi) = match wilcoxon_crit(n, 0.05) {
        Some(k) if k < nn / 2 => (w[k], w[nn - 1 - k]),
        _ => (f64::NEG_INFINITY, f64::INFINITY), // too few pairs for a 95% interval
    };
    Some(Est { n, est, lo, hi })
}

/// Two-sample (unpaired, LI) Hodges–Lehmann shift of y over x = median of all y_j - x_i, with the
/// Mann–Whitney 95% interval (normal approximation) [D_(k+1), D_(nm-k)].
pub fn hl_two_sample(x: &[f64], y: &[f64]) -> Option<Est> {
    let (n, m) = (x.len(), y.len());
    if n == 0 || m == 0 || x.iter().chain(y.iter()).any(|v| !v.is_finite()) { return None; }
    let mut d: Vec<f64> = Vec::with_capacity(n * m);
    for &a in x { for &b in y { d.push(b - a); } }
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let est = median_sorted(&d);
    let nm = (n * m) as f64;
    let z = 1.959963984540054;
    let k = (nm / 2.0 - z * (nm * (n + m + 1) as f64 / 12.0).sqrt() - 0.5).floor();
    let (lo, hi) = if k >= 0.0 && (k as usize) < d.len() / 2 {
        (d[k as usize], d[d.len() - 1 - k as usize])
    } else { (f64::NEG_INFINITY, f64::INFINITY) };
    Some(Est { n: n.min(m), est, lo, hi })
}

// ---------------------------------------------------------------------------------------------
// T2 (§6): the decision rule, the sequential race, plan ordering, the table writer and report.
// Pure CPU — the GPU driver (exl3_autotune) feeds it paired observations.
// ---------------------------------------------------------------------------------------------

/// §6.3 practical threshold: τ = max(0.02 ms, 0.1% of the base unit's median time).
pub fn tau_ms(base_median_ms: f64) -> f64 {
    if !base_median_ms.is_finite() { return 0.02; }
    (0.001 * base_median_ms.abs()).max(0.02)
}

/// Median of a sample (None when empty).
pub fn median(v: &[f64]) -> Option<f64> {
    if v.is_empty() { return None; }
    let mut s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() { return None; }
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(median_sorted(&s))
}

/// The §6.3 decision on a paired estimate Δ = t_cand − t_base: adopt only when the WHOLE 95% CI
/// lies below −τ (a preference for the canonical default: hysteresis, reproducible picks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// CI_upper < −τ: faster beyond the practical threshold
    Adopt,
    /// not shown faster beyond τ: the default stays
    Default,
    /// CI_lower > 0: clearly slower
    Slower,
}

pub fn verdict(e: &Est, tau: f64) -> Verdict {
    if e.hi < -tau { Verdict::Adopt } else if e.lo > 0.0 { Verdict::Slower } else { Verdict::Default }
}

/// One paired observation: (width tag, Δ ms) — the tag is the verify width m' of the round (0 for a
/// pooled/unit-less observation), so a race pools widths while the width split can separate them.
pub type Obs = (u16, f64);

/// What a sampler returns for one candidate: fresh paired observations, or the reason it leaves
/// the race (a digest mismatch = a kernel bug on live activations; an identical graph = inert).
#[derive(Clone, Debug)]
pub enum Sampled {
    Obs(Vec<Obs>),
    Out(String),
}

/// Race parameters (§6.3): stage 1 = `n_stage1` pairs for every candidate; stage 2 = sequential
/// batches of `step` pairs up to `n_max`; confirmation = `n_confirm` pairs on fresh rounds.
#[derive(Clone, Copy, Debug)]
pub struct RaceCfg {
    pub n_stage1: usize,
    pub n_max: usize,
    pub n_confirm: usize,
    pub step: usize,
}

impl Default for RaceCfg {
    fn default() -> Self { RaceCfg { n_stage1: 8, n_max: 40, n_confirm: 24, step: 4 } }
}

#[derive(Clone, Debug)]
pub struct CandOut {
    pub idx: usize,
    pub fate: String,
    pub est: Option<Est>,
    pub obs: Vec<Obs>,
}

#[derive(Clone, Debug)]
pub struct RaceOut {
    pub cands: Vec<CandOut>,
    /// adopted AND confirmed on fresh rounds
    pub winner: Option<usize>,
    /// the best estimate after stage 2 (adopted or not) — the report's "leader"
    pub leader: Option<usize>,
    pub confirm: Option<Est>,
    pub confirm_obs: Vec<Obs>,
    pub verdict: String,
    /// sampler calls made (each call = fresh rounds on the device)
    pub calls: usize,
}

/// Pooled paired estimate of an observation list.
pub fn est_of(obs: &[Obs]) -> Option<Est> {
    let d: Vec<f64> = obs.iter().map(|o| o.1).collect();
    hl_paired(&d)
}

/// The §6.3 race over `n` candidates (indices 0..n; the base is implicit — every observation is
/// already a paired difference against it). `sample(active, want, confirm)` returns, per active
/// candidate in order, at least `want` NEW observations (fresh rounds) or its exit reason;
/// `confirm` marks the confirmation draw (the driver widens it to every served width).
///   stage 1: `n_stage1` pairs each; drop CI_lo > 0 (clearly slower); keep the best ceil(C/2).
///   stage 2: batches of `step` until each survivor stops (CI half-width < τ/2, or n >= n_max) or
///            drops (its CI lies wholly above the leader's).
///   adopt the leader only if its CI_upper < −τ; then CONFIRM on `n_confirm` fresh pairs — it
///   stands only if the confirmation CI lies wholly below 0 (two independent significant results:
///   the winner's-curse control over the ~400 comparisons of a full run).
pub fn race(n: usize, cfg: &RaceCfg, tau: f64,
            sample: &mut dyn FnMut(&[usize], usize, bool) -> Vec<Sampled>) -> RaceOut {
    let mut out = RaceOut { cands: (0..n).map(|i| CandOut { idx: i, fate: String::new(), est: None, obs: Vec::new() })
                                .collect(),
                            winner: None, leader: None, confirm: None, confirm_obs: Vec::new(),
                            verdict: String::new(), calls: 0 };
    if n == 0 {
        out.verdict = "DEFAULT (no candidates)".into();
        return out;
    }
    let mut take = |active: &[usize], want: usize, out: &mut RaceOut| {
        let r = sample(active, want, false);
        out.calls += 1;
        for (j, &i) in active.iter().enumerate() {
            match r.get(j) {
                // a call that adds no observation can never tighten a CI: out, never re-queued
                Some(Sampled::Obs(o)) if o.is_empty() => out.cands[i].fate = "OUT (no new samples this call)".into(),
                Some(Sampled::Obs(o)) => out.cands[i].obs.extend(o.iter().copied()),
                Some(Sampled::Out(why)) => out.cands[i].fate = why.clone(),
                None => out.cands[i].fate = "OUT (no samples)".into(),
            }
        }
    };
    // ---- stage 1
    let all: Vec<usize> = (0..n).collect();
    take(&all, cfg.n_stage1, &mut out);
    let mut live: Vec<usize> = Vec::new();
    for c in out.cands.iter_mut() {
        if !c.fate.is_empty() { continue; }
        c.est = est_of(&c.obs);
        match c.est {
            Some(e) if e.lo > 0.0 => c.fate = "SLOWER (stage 1: CI wholly above 0)".into(),
            Some(_) => live.push(c.idx),
            None => c.fate = "OUT (no estimate)".into(),
        }
    }
    live.sort_by(|a, b| out.cands[*a].est.unwrap().est.partial_cmp(&out.cands[*b].est.unwrap().est).unwrap());
    let keep = (n + 1) / 2;
    for &i in live.iter().skip(keep) { out.cands[i].fate = "PRUNED (stage 1: outside the best half)".into(); }
    live.truncate(keep);
    // ---- stage 2 (sequential; bounded even if a sampler under-delivers)
    let mut stopped: Vec<usize> = Vec::new();
    let max_iters = (cfg.n_max.saturating_sub(cfg.n_stage1)) / cfg.step.max(1) + 2;
    let mut iters = 0usize;
    loop {
        iters += 1;
        if iters > max_iters {
            stopped.extend(live.drain(..));
            break;
        }
        // leader = best estimate among the survivors (stopped or still sampling)
        let pool: Vec<usize> = live.iter().chain(stopped.iter()).copied().collect();
        let leader = pool.iter().copied().min_by(|a, b| out.cands[*a].est.unwrap().est
            .partial_cmp(&out.cands[*b].est.unwrap().est).unwrap());
        let Some(ld) = leader else { break };
        let lhi = out.cands[ld].est.unwrap().hi;
        let mut next: Vec<usize> = Vec::new();
        for &i in &live {
            let e = out.cands[i].est.unwrap();
            if i != ld && e.lo > lhi {
                out.cands[i].fate = "DROPPED (stage 2: CI wholly above the leader's)".into();
            } else if e.n >= cfg.n_max || (e.hi - e.lo) / 2.0 < tau / 2.0 {
                stopped.push(i);
            } else {
                next.push(i);
            }
        }
        live = next;
        if live.is_empty() { break; }
        take(&live, cfg.step, &mut out);
        let mut still: Vec<usize> = Vec::new();
        for &i in &live {
            if !out.cands[i].fate.is_empty() { continue; }
            match est_of(&out.cands[i].obs) {
                Some(e) => { out.cands[i].est = Some(e); still.push(i); }
                None => out.cands[i].fate = "OUT (no estimate)".into(),
            }
        }
        live = still;
    }
    let leader = stopped.iter().copied().min_by(|a, b| out.cands[*a].est.unwrap().est
        .partial_cmp(&out.cands[*b].est.unwrap().est).unwrap());
    out.leader = leader;
    for &i in &stopped {
        if Some(i) != leader && out.cands[i].fate.is_empty() { out.cands[i].fate = "RUNNER-UP".into(); }
    }
    let Some(ld) = leader else {
        out.verdict = "DEFAULT (no surviving candidate)".into();
        return out;
    };
    let e = out.cands[ld].est.unwrap();
    match verdict(&e, tau) {
        Verdict::Adopt => {}
        v => {
            out.cands[ld].fate = format!("LEADER, not adopted ({v:?}: CI [{:+.4}, {:+.4}] vs tau {tau:.4})", e.lo, e.hi);
            out.verdict = "DEFAULT (no candidate beats tau)".into();
            return out;
        }
    }
    // ---- confirmation on fresh rounds
    let r = sample(&[ld], cfg.n_confirm, true);
    out.calls += 1;
    match r.into_iter().next() {
        Some(Sampled::Obs(o)) => {
            out.confirm = est_of(&o);
            out.confirm_obs = o;
        }
        Some(Sampled::Out(why)) => {
            out.cands[ld].fate = why;
            out.verdict = "DEFAULT (winner left the race during confirmation)".into();
            return out;
        }
        None => {}
    }
    match out.confirm {
        Some(c) if c.hi < 0.0 => {
            out.cands[ld].fate = "ADOPTED (confirmed)".into();
            out.winner = Some(ld);
            out.verdict = "ADOPTED".into();
        }
        c => {
            out.cands[ld].fate = format!("ADOPTED then REJECTED by confirmation ({:?})", c.map(|c| (c.est, c.lo, c.hi)));
            out.verdict = "DEFAULT (confirmation failed)".into();
        }
    }
    out
}

/// Per-width view of a winner's observations (§6.2 width groups): (m', estimate, keep). A width is
/// split back to the default only when its own CI lies wholly above 0 (clearly slower there).
pub fn width_split(obs: &[Obs]) -> Vec<(u16, Option<Est>, bool)> {
    let mut ws: Vec<u16> = obs.iter().map(|o| o.0).collect();
    ws.sort_unstable();
    ws.dedup();
    ws.into_iter().map(|w| {
        let d: Vec<f64> = obs.iter().filter(|o| o.0 == w).map(|o| o.1).collect();
        let e = hl_paired(&d);
        let keep = !matches!(e, Some(x) if x.lo > 0.0);
        (w, e, keep)
    }).collect()
}
