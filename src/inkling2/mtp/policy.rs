//! S6: how many drafts a round verifies, and whether a lane speculates at all.
//!
//! EXACTNESS (sampled requests). The accept is the standard rule (accept draft j with probability min(1, p(d)/q(d)),
//! on the first rejection draw from (p - q)+, after c accepted drafts draw from p_c). The emitted law is the target's
//! iff the decision "verify draft j" is a stopping rule: it may read the history (earlier rounds, the context) and the
//! drafter's distributions up to q_j (which depend on d_1..d_{j-1}), but NEVER the draw d_j itself
//! (PLAN/DSV41_PS3_REPORT.md section E; proof sketch: the emitted law is (1 - abar) p + a min(p, q) + ... and equals p
//! for every p, q iff the verify probability a does not depend on the draw). Here:
//!   * `conf[j]` = the top probability of q_{j+1}'s distribution (the step that DRAWS d_{j+1}); it is computed from
//!     d_1..d_j only (the Markov step reads the previous token), so a rule that decides "verify d_{j+1}" from
//!     conf[0..=j] reads d_1..d_j and not d_{j+1}: exact.
//!   * the acceptance estimates per confidence bucket and the round costs are learned from PAST rounds only.
//!   * whether a lane drafts at all is decided from past rounds only.
//! Greedy requests: drafts and accept are deterministic argmax equality; any count is exact (byte identity rests on
//! the verify rows being bit-identical to decode rows, AGENTS §2.4).
//!
//! The count rule is the look-ahead stop: verify d_{j+1} iff d_j was verified and some count c' > j beats count j,
//! with the acceptances a_1..a_{j+1} from the known confidences and the later ones projected as a_{j+1} x DECAY^i
//! (a constant; reads nothing of the round).

pub const NB: usize = 20;
const DECAY: f64 = 0.85;

#[derive(Clone, Debug)]
pub struct Policy {
    /// per confidence bucket: (accepted, tried) with a prior
    acc: [(f64, f64); NB],
    /// verify cost (ms) by rows m = 1..=16 (EMA), and the drafter's cost per round (ms)
    pub verify_ms: [f64; 17],
    pub draft_ms: f64,
    /// most drafts per round
    pub max_drafts: usize,
    /// speculation on/off for a lane from the realised rates (EMA of tokens / ms in spec rounds vs plain steps)
    pub spec_rate: f64,
    pub plain_rate: f64,
    /// plain steps left before the next probe round when speculation lost
    pub backoff: usize,
    pub rounds: u64,
}

pub fn bucket(conf: f32) -> usize { ((conf.clamp(0.0, 0.999_999) as f64) * NB as f64) as usize }

impl Policy {
    /// `step_ms` = a plain decode step (one row), `row_ms` = the extra cost per verify row, `draft_ms` = the drafter.
    pub fn new(step_ms: f64, row_ms: f64, draft_ms: f64, max_drafts: usize) -> Policy {
        let mut acc = [(0.0, 0.0); NB];
        for (b, a) in acc.iter_mut().enumerate() {
            // prior: the drafter is roughly calibrated (acceptance ~ its top probability), weight 4 rounds
            let mid = (b as f64 + 0.5) / NB as f64;
            *a = (4.0 * mid * 0.9, 4.0);
        }
        let mut verify_ms = [0.0; 17];
        for (m, v) in verify_ms.iter_mut().enumerate().skip(1) { *v = step_ms + row_ms * (m as f64 - 1.0); }
        Policy { acc, verify_ms, draft_ms, max_drafts: max_drafts.min(super::gpu::GMAX), spec_rate: 0.0, plain_rate: 1.0 / step_ms,
                 backoff: 0, rounds: 0 }
    }

    pub fn est(&self, conf: f32) -> f64 { let (a, n) = self.acc[bucket(conf)]; (a / n).clamp(0.0, 1.0) }

    fn rate(&self, a: &[f64], c: usize) -> f64 {
        let mut e = 1.0;
        let mut pr = 1.0;
        for &x in &a[..c] { pr *= x; e += pr; }
        e / (self.draft_ms + self.verify_ms[(c + 1).min(16)])
    }

    /// The count for one round from the drafter's confidences (`conf[j]` = top probability of the step drawing
    /// d_{j+1}), at most `cap` drafts. Reads conf[0..=j] to decide d_{j+1} (exact, see the module doc).
    pub fn choose(&self, conf: &[f32], cap: usize) -> usize {
        let cap = cap.min(self.max_drafts).min(conf.len());
        let mut c = 0;
        while c < cap && self.more(conf, c, cap) { c += 1; }
        c
    }

    /// One step of `choose`: with `c` drafts already chosen and conf[0..=c] known, is draft c + 1 verified too?
    /// (`choose` == repeated `more` from c = 0; the serve path calls this after each Markov step to stop drafting.)
    pub fn more(&self, conf: &[f32], c: usize, cap: usize) -> bool {
        let cap = cap.min(self.max_drafts);
        if c >= cap || conf.len() < c + 1 { return false; }
        let mut a: Vec<f64> = conf[..=c].iter().map(|&x| self.est(x)).collect();
        let last = *a.last().unwrap();
        for i in 1..=(cap - c - 1) { a.push(last * DECAY.powi(i as i32)); }
        let r0 = self.rate(&a, c);
        (c + 1..=cap).any(|c2| self.rate(&a, c2) > r0)
    }

    /// Learn from a verified round: `conf` of the verified drafts, `c` verified, `a` accepted, round time and tokens.
    pub fn learn(&mut self, conf: &[f32], c: usize, a: usize, verify_ms: f64, draft_ms: f64) {
        for j in 0..c.min(a + 1) {
            let b = bucket(conf[j]);
            let e = &mut self.acc[b];
            let ok = if j < a { 1.0 } else { 0.0 };
            // keep the effective window bounded (~500 samples) so the estimate tracks the request's text
            if e.1 > 500.0 { e.0 *= 0.998; e.1 *= 0.998; }
            e.0 += ok;
            e.1 += 1.0;
        }
        let m = (c + 1).min(16);
        self.verify_ms[m] = 0.9 * self.verify_ms[m] + 0.1 * verify_ms;
        self.draft_ms = 0.9 * self.draft_ms + 0.1 * draft_ms;
        let r = (a + 1) as f64 / (verify_ms + draft_ms);
        self.spec_rate = if self.rounds == 0 { r } else { 0.85 * self.spec_rate + 0.15 * r };
        self.rounds += 1;
    }

    pub fn learn_plain(&mut self, step_ms: f64) { self.plain_rate = 0.9 * self.plain_rate + 0.1 / step_ms; self.verify_ms[1] = 0.9 * self.verify_ms[1] + 0.1 * step_ms; }

    /// Whether this lane drafts on its next step (history only). After a losing stretch it decodes plain for a
    /// while and then probes again.
    pub fn should_draft(&mut self) -> bool {
        if self.backoff > 0 { self.backoff -= 1; return self.backoff == 0; }
        if self.rounds >= 8 && self.spec_rate < self.plain_rate * 0.98 {
            self.backoff = 16;
            self.spec_rate = self.plain_rate; // the probe round re-measures from neutral
            return false;
        }
        true
    }
}

/// Host twin of the device accept (`ink6_accept` + `ink6_final_part`) on probability vectors: drafts `d[j]` drawn
/// from `q[j]`, target rows `p[0..=c]`, uniforms `u` for the accept tests and `v` for the final draw.
pub fn accept_host(p: &[Vec<f64>], q: &[Vec<f64>], d: &[usize], c: usize, u: &[f64], v: f64) -> (usize, usize) {
    let mut a = 0;
    while a < c {
        let x = d[a];
        if !(u[a] < p[a][x] / q[a][x]) { break; }
        a += 1;
    }
    let w: Vec<f64> = if a < c { p[a].iter().zip(&q[a]).map(|(&x, &y)| (x - y).max(0.0)).collect() } else { p[a].clone() };
    let s: f64 = w.iter().sum();
    let mut acc = 0.0;
    let t = v * s;
    for (i, &x) in w.iter().enumerate() { acc += x; if t < acc { return (a, i); } }
    (a, w.len() - 1)
}

/// S8: the TRUNCATED target law of one logits row (temperature `t` > 0, `top_k` > 0 keeps the k largest, `top_p` < 1
/// keeps the shortest prefix of the descending order whose mass reaches top_p): full-width probabilities, 0 outside the
/// kept set. The kept set is exactly `model::serve::cpu_sample`'s (descending logit, ties by lower index; top-k first,
/// then the nucleus of the top-k renormalised), so a speculative round of a top_p / top_k request emits the same law
/// the plain path draws from. Only the candidates are sorted (every non-candidate is below every candidate), so the
/// cost is one pass over the row plus a sort of the kept neighbourhood, not a 200K sort.
pub fn trunc_probs(l: &[f32], t: f32, top_p: f32, top_k: usize) -> Vec<f64> {
    let n = l.len();
    let it = 1.0 / t as f64;
    let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let w: Vec<f64> = l.iter().map(|&x| ((x as f64 - mx) * it).exp()).collect();
    let tot: f64 = w.iter().sum();
    // candidates: the k largest (top_k), else every token above a logit floor whose mass reaches top_p (lowered until
    // it does); always a superset of the kept set and closed under "logit >= the smallest candidate"
    let desc = |a: &usize, b: &usize| l[*b].partial_cmp(&l[*a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b));
    let mut cand: Vec<usize>;
    if top_k > 0 && top_k < n {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.select_nth_unstable_by(top_k - 1, desc);
        cand = idx[..top_k].to_vec();
    } else if top_p < 1.0 {
        let mut gap = 8.0f64; // in units of t (logit distance below the max)
        loop {
            let floor = mx - gap * t as f64;
            cand = (0..n).filter(|&i| l[i] as f64 >= floor).collect();
            let m: f64 = cand.iter().map(|&i| w[i]).sum();
            if m / tot >= top_p as f64 || cand.len() == n { break; }
            gap *= 2.0;
        }
    } else {
        cand = (0..n).collect();
    }
    cand.sort_by(desc);
    // top-k mass (the nucleus is taken within the top-k, renormalised over it, as cpu_sample does)
    let kmass: f64 = if top_k > 0 && top_k < n { cand.iter().map(|&i| w[i]).sum() } else { tot };
    let mut keep = cand.len();
    if top_p < 1.0 {
        let mut c = 0.0;
        for (j, &i) in cand.iter().enumerate() { c += w[i] / kmass; if c >= top_p as f64 { keep = j + 1; break; } }
    }
    let s: f64 = cand[..keep].iter().map(|&i| w[i]).sum();
    let mut p = vec![0f64; n];
    for &i in &cand[..keep] { p[i] = w[i] / s; }
    p
}

/// S8: the softmax of one drafter step row at temperature `t` (the law the device drew the draft from).
pub fn softmax_t(l: &[f32], t: f32) -> Vec<f64> {
    let it = 1.0 / t as f64;
    let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let w: Vec<f64> = l.iter().map(|&x| ((x as f64 - mx) * it).exp()).collect();
    let s: f64 = w.iter().sum();
    w.into_iter().map(|x| x / s).collect()
}

/// S8: the exact accept of a round on the host, with the target rows built lazily (`p(j)`: the target law of verify
/// row j, `q(j)`: the drafter law of draft j + 1): accept d_{j+1} iff u_j < p_j(d) / q_j(d); on the first rejection
/// draw from (p_j - q_j)+, after c accepted drafts from p_c. Same rule as `accept_host` (it is `accept_host` with rows
/// computed on demand: only rows 0..=a are built). Returns (accepted, the emitted token after them).
pub fn accept_lazy(c: usize, d: &[usize], p: &mut dyn FnMut(usize) -> Vec<f64>, q: &mut dyn FnMut(usize) -> Vec<f64>, u: &[f64], v: f64)
                   -> (usize, usize) {
    let mut a = 0;
    let mut last: Option<(Vec<f64>, Vec<f64>)> = None;
    while a < c {
        let (pa, qa) = (p(a), q(a));
        let x = d[a];
        if !(u[a] < pa[x] / qa[x]) { last = Some((pa, qa)); break; }
        a += 1;
    }
    let w: Vec<f64> = match last {
        Some((pa, qa)) => pa.iter().zip(&qa).map(|(&x, &y)| (x - y).max(0.0)).collect(),
        None => p(c),
    };
    let s: f64 = w.iter().sum();
    let t = v * s;
    let mut acc = 0.0;
    let mut lastnz = 0;
    for (i, &x) in w.iter().enumerate() { if x > 0.0 { lastnz = i; } acc += x; if t < acc { return (a, i); } }
    (a, lastnz)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 { self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15); let mut z = self.0; z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9); z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb); z ^ (z >> 31) }
        fn u(&mut self) -> f64 { ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64 }
        fn draw(&mut self, p: &[f64]) -> usize { let t = self.u(); let mut a = 0.0; for (i, &x) in p.iter().enumerate() { a += x; if t < a { return i; } } p.len() - 1 }
    }

    // 3-token toy: target p(next | prev) and drafter q(next | prev) (deliberately different).
    const PT: [[f64; 3]; 3] = [[0.2, 0.4, 0.4], [0.5, 0.3, 0.2], [0.1, 0.1, 0.8]];
    const QT: [[f64; 3]; 3] = [[0.5, 0.25, 0.25], [0.2, 0.6, 0.2], [0.3, 0.3, 0.4]];

    /// One round from anchor `x0`, count rule `rule(conf, drafts) -> c`; returns the first two emitted tokens.
    fn emit2(r: &mut Rng, x0: usize, rule: &dyn Fn(&[f32], &[usize]) -> usize) -> (usize, usize) {
        let mut out = Vec::new();
        let mut prev = x0;
        while out.len() < 2 {
            // draft 3 tokens
            let mut d = Vec::new();
            let mut q = Vec::new();
            let mut conf = Vec::new();
            let mut pp = prev;
            for _ in 0..3 {
                let qq = QT[pp].to_vec();
                conf.push(qq.iter().cloned().fold(0.0, f64::max) as f32);
                let x = r.draw(&qq);
                q.push(qq);
                d.push(x);
                pp = x;
            }
            let c = rule(&conf, &d);
            let mut p = Vec::new();
            let mut pv = prev;
            for j in 0..=c { p.push(PT[pv].to_vec()); if j < c { pv = d[j]; } }
            let u: Vec<f64> = (0..c).map(|_| r.u()).collect();
            let (a, t) = accept_host(&p, &q, &d, c, &u, r.u());
            for &x in &d[..a] { out.push(x); }
            out.push(t);
            prev = t;
        }
        (out[0], out[1])
    }

    fn chi2(rule: &dyn Fn(&[f32], &[usize]) -> usize, n: usize, seed: u64) -> f64 {
        let mut r = Rng(seed);
        let mut cnt = [[0f64; 3]; 3];
        for _ in 0..n { let (a, b) = emit2(&mut r, 0, rule); cnt[a][b] += 1.0; }
        let mut x2 = 0.0;
        for a in 0..3 { for b in 0..3 {
            let e = n as f64 * PT[0][a] * PT[a][b];
            x2 += (cnt[a][b] - e).powi(2) / e;
        } }
        x2
    }

    // chi-square 8 df: p = 1e-3 at 26.12
    const CRIT: f64 = 26.12;

    #[test]
    fn inkling2_mtp_accept_exact_fixed_and_policy_counts() {
        let n = 400_000;
        for c in 0..=3 {
            let x = chi2(&|_, _| c, n, 11 + c as u64);
            assert!(x < CRIT, "fixed count {c}: chi2 {x}");
        }
        // the real policy's look-ahead stop on the confidences, with a cost table that makes counts differ
        let pol = Policy::new(10.0, 1.5, 3.0, 3);
        let x = chi2(&|conf, _| pol.choose(conf, 3), n, 99);
        assert!(x < CRIT, "policy count: chi2 {x}");
        // a stop on the first confidence below 0.45 (reads conf only)
        let x = chi2(&|conf, _| conf.iter().position(|&z| z < 0.45).unwrap_or(3), n, 7);
        assert!(x < CRIT, "first-stop on confidence: chi2 {x}");
    }

    #[test]
    fn inkling2_mtp_accept_control_reading_the_draw_fails() {
        // CONTROL: verify d_1 only when d_1 == 0 (reads the draw it decides on): must be rejected.
        let x = chi2(&|_, d| if d[0] == 0 { 2 } else { 0 }, 400_000, 5);
        assert!(x > CRIT, "the control must fail the exactness test, chi2 {x}");
    }

    #[test]
    fn inkling2_mtp_policy_choose_reads_only_earlier_confidences() {
        // decision "verify d_{j+1}" (c >= j + 1) must not change when conf[j + 1..] is replaced
        let pol = Policy::new(31.0, 2.5, 9.0, 7);
        let mut r = Rng(3);
        for _ in 0..20_000 {
            let conf: Vec<f32> = (0..7).map(|_| r.u() as f32).collect();
            let c = pol.choose(&conf, 7);
            for j in 0..7 {
                let mut alt = conf.clone();
                for x in alt.iter_mut().skip(j + 1) { *x = r.u() as f32; }
                let c2 = pol.choose(&alt, 7);
                assert_eq!(c >= j + 1, c2 >= j + 1, "decision on draft {} read a later confidence", j + 1);
            }
        }
    }

    #[test]
    fn inkling2_mtp_policy_drafts_more_when_confident() {
        let pol = Policy::new(31.0, 2.5, 9.0, 7);
        assert!(pol.choose(&[0.95; 7], 7) >= 5);
        assert!(pol.choose(&[0.15; 7], 7) <= 1);
    }

    /// S8: a top_p / top_k request: the rounds' emitted law must be the TRUNCATED target (the draws come from the
    /// untruncated drafter; the accept reads the truncated rows through the real `trunc_probs` + `accept_lazy`).
    fn chi2_trunc(top_p: f32, top_k: usize, truncate_in_accept: bool, rule: &dyn Fn(&[f32], &[usize]) -> usize, n: usize, seed: u64) -> f64 {
        let t = 1.0f32;
        let lg = |row: &[f64; 3]| -> Vec<f32> { row.iter().map(|&x| x.ln() as f32).collect() };
        let pt = |prev: usize| -> Vec<f64> { trunc_probs(&lg(&PT[prev]), t, top_p, top_k) };
        let mut r = Rng(seed);
        let mut cnt = [[0f64; 3]; 3];
        for _ in 0..n {
            let mut out = Vec::new();
            let mut prev = 0usize;
            while out.len() < 2 {
                let (mut d, mut conf, mut qs) = (Vec::new(), Vec::new(), Vec::new());
                let mut pp = prev;
                for _ in 0..3 {
                    let qq = softmax_t(&lg(&QT[pp]), t);
                    conf.push(qq.iter().cloned().fold(0.0, f64::max) as f32);
                    let x = r.draw(&qq);
                    qs.push(qq); d.push(x); pp = x;
                }
                let c = rule(&conf, &d);
                let mut ctx = vec![prev];
                ctx.extend_from_slice(&d);
                let u: Vec<f64> = (0..c).map(|_| r.u()).collect();
                let v = r.u();
                let (a, tk) = accept_lazy(c, &d, &mut |j| if truncate_in_accept { pt(ctx[j]) } else { softmax_t(&lg(&PT[ctx[j]]), t) },
                                          &mut |j| qs[j].clone(), &u, v);
                for &x in &d[..a] { out.push(x); }
                out.push(tk);
                prev = tk;
            }
            cnt[out[0]][out[1]] += 1.0;
        }
        // expected: the truncated chain
        let mut x2 = 0.0;
        let (mut df, p0) = (0, pt(0));
        for a in 0..3 { for b in 0..3 {
            let e = n as f64 * p0[a] * pt(a)[b];
            if e > 0.0 { x2 += (cnt[a][b] - e).powi(2) / e; df += 1; } else if cnt[a][b] > 0.0 { x2 += 1e9; }
        } }
        // one-sample chi-square over the df + 1 reachable cells: x2 / (the p = 1e-3 critical value at df - 1 dof), so
        // < 1 passes
        const C: [f64; 9] = [1.0, 10.83, 13.82, 16.27, 18.47, 20.52, 22.46, 24.32, 26.12];
        x2 / C[(df - 1).max(1)]
    }

    #[test]
    fn inkling2_s8_trunc_probs_matches_full_sort() {
        // the kept set and law of the fast candidate path == a full descending sort (cpu_sample's construction)
        let mut r = Rng(17);
        for case in 0..300 {
            let n = 50 + (r.next() % 3000) as usize;
            let mut l: Vec<f32> = (0..n).map(|_| (r.u() * 12.0 - 6.0) as f32).collect();
            if case % 5 == 0 { for i in 0..n / 3 { l[i] = l[n - 1 - i]; } } // ties
            let t = [0.6f32, 1.0, 1.7][case % 3];
            let (tp, tk) = [(0.9f32, 0usize), (1.0, 40), (0.5, 100), (0.95, 7), (0.999, 0), (0.3, 0)][case % 6];
            let got = trunc_probs(&l, t, tp, tk);
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap().then(a.cmp(&b)));
            let kmax = if tk > 0 { tk.min(n) } else { n };
            let mx = l[idx[0]] as f64;
            let p: Vec<f64> = idx[..kmax].iter().map(|&i| ((l[i] as f64 - mx) / t as f64).exp()).collect();
            let tot: f64 = p.iter().sum();
            let mut keep = kmax;
            if tp < 1.0 { let mut c = 0.0; for (j, &x) in p.iter().enumerate() { c += x / tot; if c >= tp as f64 { keep = j + 1; break; } } }
            let s: f64 = p[..keep].iter().sum();
            let mut want = vec![0f64; n];
            for j in 0..keep { want[idx[j]] = p[j] / s; }
            for i in 0..n { assert!((got[i] - want[i]).abs() < 1e-12, "case {case} token {i}: {} vs {}", got[i], want[i]); }
        }
    }

    #[test]
    fn inkling2_s8_accept_exact_under_top_p_top_k() {
        let n = 400_000;
        let pol = Policy::new(10.0, 1.5, 3.0, 3);
        // top_k 2 (token 0 dropped after prev 0 and 2), top_p 0.6, and both
        for (i, (tp, tk)) in [(1.0f32, 2usize), (0.6, 0), (0.7, 2)].iter().enumerate() {
            for (j, c) in [0usize, 1, 3].iter().enumerate() {
                let x = chi2_trunc(*tp, *tk, true, &|_, _| *c, n, 100 + (i * 10 + j) as u64);
                eprintln!("S8 exact top_p {tp} top_k {tk} fixed count {c}: chi2/crit {x:.3}");
                assert!(x < 1.0, "top_p {tp} top_k {tk} fixed count {c}: chi2 / crit {x}");
            }
            let x = chi2_trunc(*tp, *tk, true, &|conf, _| pol.choose(conf, 3), n, 200 + i as u64);
            eprintln!("S8 exact top_p {tp} top_k {tk} policy count: chi2/crit {x:.3}");
            assert!(x < 1.0, "top_p {tp} top_k {tk} policy count: chi2 / crit {x}");
        }
    }

    #[test]
    fn inkling2_s8_accept_control_untruncated_accept_fails() {
        // CONTROL: accepting against the UNtruncated target while the request asks top_k 2: must be rejected.
        let x = chi2_trunc(1.0, 2, false, &|_, _| 3, 400_000, 9);
        eprintln!("S8 control (untruncated accept, top_k 2): chi2/crit {x:.3e}");
        assert!(x > 1.0, "the control must fail, chi2 / crit {x}");
    }
}
