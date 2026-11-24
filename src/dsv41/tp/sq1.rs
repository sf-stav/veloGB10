//! DSV4.1 TP (leg SQ1): is the served sampler at TP exact, and how often does it write off-script text? The part `sq1`
//! of the TP model program (`--probe-dsv41-tp --dsv41-tp-program model --dsv41-tp-run sq1 --dsv41-tp-lanes 4`), every
//! rank running the same program. Started by the owner's hand test (2026-11-19): a sampled "hello" (thinking on, no
//! sampler fields, TP2, DSpark) answered "Hello! How can第一行 must be ignored. How can I help you today?".
//! Sections (`--dsv41-sq1-budget TRIALS,SECONDS,REPLIES` sizes S3 and S4):
//! * S1 exact q: the served path (prefill, then a plain k = 0 round of the same DSpark verify) at four contexts: the
//!   owner's reply up to the stray token ("... Hello! How can"), the same + "第一", the bare "hello" prompt (the
//!   probability of an empty reasoning, `</think>` first) and a high-entropy list context. The target row is gathered over
//!   the vocab shards (an exact fp32 sum of zero-padded slices) and normalised in f64 on the host.
//! * S2 vocab-parallel == one rank: 4,096 keyed draws per row through the TP vocab pass + record exchange, and through the
//!   same kernels over the gathered row on one rank (`Local` exchange): every draw and argmax must be identical; the
//!   draws must fit the exact q (one-sample chi-square).
//! * S3 speculative exactness: from one lane state (4 lanes, the same context), one DSpark round with k = 0, 1, 2, 4, 5
//!   drafts, then plain rounds until G = 4 tokens, the seed varied per trial, the lanes rolled back to the context after
//!   every trial (all writes stay inside one verify's 6-row footprint). Per position, the two-sample chi-square of every k
//!   against k = 0 and the fit of the first token to S1's exact q; pass = p >= 1e-4 everywhere. A negative control
//!   (DeepSeek's acceptance ratio swapped, k = 5) must be detected (p < 1e-4 somewhere), else the cell has no power.
//! * S4 off-script rate: sampled replies (THE POSTURE: temperature 1, top_p 1, thinking on) to short chat prompts on all
//!   lanes, speculation on (the policy's draft length, as the server) and off (k = 0). Every committed token is booked
//!   against the target row it was drawn from: CJK tokens and rare tokens (q < 1e-3, q < 1e-4) observed against their
//!   expected counts under the engine's own q (exact Poisson tails; a sampler that inflates the tail fails here), and
//!   replies with a CJK run. In the plain arm every token is also redrawn on one rank from the gathered row with the
//!   lane's key: bit-identical or FAIL.
//! * S6 state carried between requests: the owner's row of S1 again after S3 / S4 (hundreds of requests on every lane),
//!   on lane 0 and on the last lane: bit-identical to the fresh boot's row, or FAIL.
//! * S5 rows for the native comparison: the owner's whole reply teacher-forced in one every-layer call, the full logits
//!   of every row written by rank 0 to `--dsv41-q1-score-out` (`sq1_rows.f32` [rows][vocab] + `sq1_rows.json`) for
//!   `tools/dsv41_sq1/compare_native.py` against DeepSeek's reference forward of the same ids.

use super::fwd::TpModel;
use super::fwprog::{encode, prefill};
use super::prog::{Guard, ModelOpts};
use super::spec::{TpSpec, TpXchg, RPL};
use crate::dsv41::ced::schedule::Mode;
use crate::dsv41::fwd::dev::{Buf, Gpu};
use crate::dsv41::gpu::dspark::chisq::{chi2_sf, one_sample, two_sample};
use crate::dsv41::gpu::dspark::vocab::{Round, Rows, Shard, REC};
use crate::dsv41::gpu::dspark::xchg::Local;
use crate::dsv41::gpu::dspark::{Lane as DsLane, VRec, TEST_SWAP_RULE};
use crate::tokenizer::QwenTokenizer;
use anyhow::{ensure, Context, Result};
use std::collections::HashMap;

/// `</think>` / `<think>` in the V4.1 vocabulary.
pub const THINK_END: u32 = 128_822;
pub const THINK_START: u32 = 128_821;
/// The owner's reply (2026-11-19, TP2, sampled, thinking on, empty reasoning).
pub const OWNER_REPLY: &str = "Hello! How can第一行 must be ignored. How can I help you today?";
/// The owner's reply up to the stray token.
pub const OWNER_HEAD: &str = "Hello! How can";
/// Short chat prompts of S4 (English, the kind of turn the owner sent).
pub const SHORT: [&str; 20] = [
    "hello", "hi", "Hello!", "hey", "good morning", "how are you?", "what's up", "thanks!", "who are you?", "Can you help me?",
    "tell me a joke", "what is 2+2?", "hi there!", "yo", "Hello, how are you today?", "nice to meet you", "good night",
    "what can you do?", "help", "hey, quick question",
];
const G_TOK: usize = 4;
const DRAWS_ROW: usize = 4096;
const ONE_CAP: usize = 64;

/// Is a token's byte string Chinese / Japanese / Korean text: a whole character in the CJK blocks (Han, kana, Hangul,
/// CJK punctuation, full-width forms), or a partial character whose lead byte only starts such characters (U+3000..U+DFFF).
pub fn is_cjk_bytes(b: &[u8]) -> bool {
    let s = String::from_utf8_lossy(b);
    if s.chars().any(is_cjk_char) { return true; }
    s.contains('\u{FFFD}') && b.iter().any(|&x| (0xE3..=0xED).contains(&x))
}
pub fn is_cjk_char(c: char) -> bool {
    matches!(c as u32, 0x2E80..=0x2FDF | 0x3000..=0x31FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xAC00..=0xD7AF | 0xF900..=0xFAFF | 0xFF00..=0xFFEF)
}

/// Exact Poisson tails of `o` events at mean `e`: (P[X <= o], P[X >= o]).
pub fn poisson_tails(o: u64, e: f64) -> (f64, f64) {
    if e <= 0.0 { return (1.0, if o == 0 { 1.0 } else { 0.0 }); }
    let le = chi2_sf(2.0 * e, 2.0 * (o as f64 + 1.0));
    let ge = if o == 0 { 1.0 } else { 1.0 - chi2_sf(2.0 * e, 2.0 * o as f64) };
    (le, ge)
}

/// f64 softmax of one full row at temperature 1.
pub fn softmax64(x: &[f32]) -> Vec<f64> {
    let m = x.iter().filter(|v| !v.is_nan()).fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let e: Vec<f64> = x.iter().map(|&v| if v.is_nan() { 0.0 } else { (v as f64 - m).exp() }).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// The parsed `--dsv41-sq1-budget TRIALS,SECONDS,REPLIES`.
#[derive(Clone, Copy, Debug)]
pub struct Budget { pub trials: usize, pub secs: f64, pub replies: usize }
pub fn budget(spec: Option<&str>) -> Result<Budget> {
    let s = spec.unwrap_or("800,420,1000");
    let v: Vec<&str> = s.split(',').map(|x| x.trim()).collect();
    ensure!(v.len() == 3, "--dsv41-sq1-budget {s:?}: TRIALS,SECONDS,REPLIES");
    let b = Budget { trials: v[0].parse()?, secs: v[1].parse()?, replies: v[2].parse()? };
    ensure!(b.trials >= 4 && b.secs >= 0.0, "--dsv41-sq1-budget {s:?}: at least 4 trials");
    Ok(b)
}

struct Ctx { name: &'static str, ids: Vec<u32> }

/// Sizes and buffers shared by the sections.
struct Kit<'a> {
    g: &'a Gpu,
    v: usize,
    sh: Shard,
    world: usize,
    /// [ONE_CAP][vocab] f32: gathered rows
    full: Buf,
    /// this rank's copy of one target shard row
    srow: Buf,
    /// [ONE_CAP * 4] f32: the per-row partial sums of S4
    red: Buf,
    one: Round,
    tp: Round,
    xtp: TpXchg<'a>,
    cjk_local: Vec<u32>,
    cjk: Vec<bool>,
}

impl<'a> Kit<'a> {
    /// Gather `rows` (this rank's shard rows, device pointers) into `full` rows 0.. (an exact sum: every other slot is 0).
    fn gather(&self, m: &mut TpModel, rows: &[u64]) -> Result<()> {
        ensure!(rows.len() <= ONE_CAP, "gather: {} rows", rows.len());
        let n = rows.len();
        self.g.memset(self.full.ptr(), n * self.v * 4, 0)?;
        for (i, &r) in rows.iter().enumerate() { self.g.d2d(self.full.ptr() + ((i * self.v + self.sh.v0) * 4) as u64, r, self.sh.vlocal * 4)?; }
        if self.world > 1 { m.reduce(self.full.ptr(), n * self.v)?; }
        Ok(())
    }
    fn host_row(&self, i: usize) -> Result<Vec<f32>> { self.g.down::<f32>(self.full.ptr() + (i * self.v * 4) as u64, self.v) }

    /// One-rank draws of the gathered rows for `lanes` (lane i reads gathered row `row_of[i]`): the records.
    fn one_rank(&self, lanes: &[DsLane], row_of: &[usize]) -> Result<Vec<VRec>> {
        ensure!(lanes.len() <= ONE_CAP && lanes.len() == row_of.len(), "one_rank: {} lanes", lanes.len());
        // the pass reads lane l's row at full + l * lane_stride: lay the requested rows out in lane order
        let n = lanes.len();
        let same = row_of.iter().all(|&r| r == row_of[0]);
        let rows = if same { Rows { ptr: self.full.ptr() + (row_of[0] * self.v * 4) as u64, lane_stride: 0, row_stride: 0 } } else {
            ensure!(row_of.iter().enumerate().all(|(i, &r)| r == i), "one_rank: rows must be the lanes' own or one shared row");
            Rows { ptr: self.full.ptr(), lane_stride: self.v, row_stride: 0 }
        };
        self.one.set_lanes(self.g, lanes)?;
        let whole = Shard { v0: 0, vlocal: self.v };
        let p = self.one.stats(self.g, &Local { max_bytes: self.one.rows_max * REC }, whole, rows, n, 1, None)?;
        self.one.recs(self.g, p, n)
    }
    /// TP draws of this rank's shard row `srow` for `lanes` (the vocab pass + the record exchange).
    fn tp_rank(&self, lanes: &[DsLane]) -> Result<Vec<VRec>> {
        let n = lanes.len();
        self.tp.set_lanes(self.g, lanes)?;
        let p = self.tp.stats(self.g, &self.xtp, self.sh, Rows { ptr: self.srow.ptr(), lane_stride: 0, row_stride: 0 }, n, 1, None)?;
        self.tp.recs(self.g, p, n)
    }
}

fn hash_chains(ch: &[Vec<u32>]) -> u32 { ch.iter().flatten().fold(0x9E37_79B9u32, |h, &t| h.wrapping_mul(31) ^ t) }

/// Lanes 0..nl set to the state "context `ctx[..p]` fed, pending token `ctx[p]` at position p" (p = len - 1): prefill
/// with the DSpark taps, the drafter's rings seeded by a first-token start whose draw is discarded. Returns (p, states).
fn setup_lane(m: &mut TpModel, sp: &mut TpSpec, l: usize, ctx: &[u32], chunk: usize, mode: Mode) -> Result<usize> {
    ensure!(ctx.len() >= 2, "a context of {} tokens", ctx.len());
    let p = ctx.len() - 1;
    m.reset_lane_n(l)?;
    sp.reset_lane(m.f.g, l)?;
    let (_, last) = prefill(m, Some(sp), l, &ctx[..p], chunk, mode)?;
    let _ = sp.start_lane(m, l, p, last, 1.0, 0x51_0000 + l as u64)?;
    sp.host[l].tok = ctx[p] as i32;
    sp.host[l].active = 1;
    Ok(p)
}
fn setup_ctx(m: &mut TpModel, sp: &mut TpSpec, nl: usize, ctx: &[u32], chunk: usize, mode: Mode) -> Result<(usize, Vec<DsLane>)> {
    let mut p = 0;
    for l in 0..nl { p = setup_lane(m, sp, l, ctx, chunk, mode)?; }
    for l in nl..sp.cap { sp.host[l].active = 0; }
    Ok((p, sp.host[..nl].to_vec()))
}

/// S1: the target row after the pending token of a context on lane `l` alone (served path), gathered; the lane rolled
/// back. This rank's shard row is kept in `kit.srow` (S2).
fn exact_row(m: &mut TpModel, sp: &mut TpSpec, kit: &Kit, l: usize, ctx: &[u32], chunk: usize, mode: Mode) -> Result<Vec<f32>> {
    let p = setup_lane(m, sp, l, ctx, chunk, mode)?;
    let st = sp.host[l];
    for i in 0..sp.cap { if i != l { sp.host[i].active = 0; } }
    let _ = sp.round(m, l + 1, 0)?;
    let row = sp.tlog.ptr() + (l * kit.sh.vlocal * 4) as u64;
    kit.g.d2d(kit.srow.ptr(), row, kit.sh.vlocal * 4)?;
    kit.gather(m, &[row])?;
    m.rollback(l, p)?;
    sp.host[l] = st;
    sp.host[l].active = 0;
    kit.host_row(0)
}

fn top_line(tok: &QwenTokenizer, q: &[f64], n: usize) -> String {
    let mut ix: Vec<usize> = (0..q.len()).collect();
    ix.sort_by(|&a, &b| q[b].total_cmp(&q[a]).then(a.cmp(&b)));
    ix.iter().take(n).map(|&i| format!("{:?} {:.3e}", tok.decode(&[i as u32], false).unwrap_or_default(), q[i])).collect::<Vec<_>>().join(", ")
}
fn rank_of(q: &[f64], id: usize) -> usize { q.iter().filter(|&&x| x > q[id]).count() + 1 }

/// S3: `n` trials from the saved lane states with k drafts in the first round (then plain rounds) up to G_TOK tokens.
#[allow(clippy::too_many_arguments)]
fn trials(m: &mut TpModel, sp: &mut TpSpec, guard: &mut Guard, saved: &[DsLane], p: usize, k: usize, flags: i32, n: usize, seed0: u64) -> Result<(Vec<Vec<u32>>, f64)> {
    let nl = saved.len();
    let mut out = Vec::with_capacity(n);
    let mut acc = (0u64, 0u64);
    sp.test_flags = flags;
    let mut t = 0usize;
    let r = (|| -> Result<()> {
        while out.len() < n {
            for l in 0..nl { sp.host[l] = saved[l]; sp.host[l].seed = seed0 + (t + l) as u64; sp.host[l].active = 1; }
            let mut ch: Vec<Vec<u32>> = vec![Vec::new(); nl];
            let em = sp.round(m, nl, k)?;
            for l in 0..nl {
                ch[l].extend(em[l].iter().take(G_TOK));
                acc.0 += sp.host[l].n_acc.max(0) as u64;
                acc.1 += 1;
            }
            sp.test_flags = 0;
            loop {
                let mut any = false;
                for l in 0..nl { let a = ch[l].len() < G_TOK; sp.host[l].active = a as i32; any |= a; }
                if !any { break; }
                let em = sp.round(m, nl, 0)?;
                for l in 0..nl { let have = ch[l].len(); if have < G_TOK { ch[l].extend(em[l].iter().take(G_TOK - have)); } }
            }
            sp.test_flags = flags;
            guard.agree("sq1 trial", hash_chains(&ch))?;
            for l in 0..nl { m.rollback(l, p)?; }
            out.extend(ch);
            t += nl;
        }
        Ok(())
    })();
    sp.test_flags = 0;
    for l in 0..nl { sp.host[l] = saved[l]; }
    r?;
    out.truncate(n);
    Ok((out, acc.0 as f64 / acc.1.max(1) as f64))
}

fn counts_at(ch: &[Vec<u32>], j: usize) -> HashMap<i32, usize> {
    let mut c = HashMap::new();
    for v in ch { if let Some(&t) = v.get(j) { *c.entry(t as i32).or_insert(0) += 1; } }
    c
}
fn probs_of(q: &[f64]) -> Vec<(u32, f64)> { q.iter().enumerate().filter(|(_, &p)| p > 0.0).map(|(i, &p)| (i as u32, p)).collect() }

/// Per-row books of S4 for committed tokens: (cjk mass, mass of q < 1e-3, mass of q < 1e-4, q of the token).
#[derive(Clone, Copy, Debug, Default)]
struct RowBook { cjk: f64, r3: f64, r4: f64, qt: f64 }

/// S4's books for `rows` = (lane, row j, token) of the last verify with `k` drafts (or a start: k = 0): this rank's
/// shard partials from the target rows and the merged log-sum-exp, summed over the ranks.
fn books(m: &mut TpModel, sp: &TpSpec, kit: &Kit, rows: &[(usize, usize, u32)], k: usize) -> Result<Vec<RowBook>> {
    if rows.is_empty() { return Ok(Vec::new()); }
    ensure!(rows.len() <= ONE_CAP, "books: {} rows", rows.len());
    let g = kit.g;
    let vl = kit.sh.vlocal;
    let tres = sp.round.recs(g, sp.round.tres.ptr(), sp.cap * RPL)?;
    let mut part = vec![0f32; ONE_CAP * 4];
    for (i, &(l, j, t)) in rows.iter().enumerate() {
        let x = g.down::<f32>(sp.tlog.ptr() + ((l * (k + 1) + j) * vl * 4) as u64, vl)?;
        let lse = tres[l * RPL + j].lse() as f64;
        let (mut cj, mut r3, mut r4) = (0f64, 0f64, 0f64);
        for &a in &x {
            if a.is_nan() || a == f32::NEG_INFINITY { continue; }
            let q = (a as f64 - lse).exp();
            if q < 1e-3 { r3 += q; if q < 1e-4 { r4 += q; } }
        }
        for &v in &kit.cjk_local { let a = x[v as usize]; if !a.is_nan() { cj += (a as f64 - lse).exp(); } }
        let tl = t as usize;
        let qt = if tl >= kit.sh.v0 && tl < kit.sh.v0 + vl { (x[tl - kit.sh.v0] as f64 - lse).exp() } else { 0.0 };
        part[i * 4..i * 4 + 4].copy_from_slice(&[cj as f32, r3 as f32, r4 as f32, qt as f32]);
    }
    g.h2d_async(kit.red.ptr(), &part)?;
    if kit.world > 1 { m.reduce(kit.red.ptr(), ONE_CAP * 4)?; }
    let s = g.down::<f32>(kit.red.ptr(), rows.len() * 4)?;
    Ok(s.chunks_exact(4).map(|c| RowBook { cjk: c[0] as f64, r3: c[1] as f64, r4: c[2] as f64, qt: c[3] as f64 }).collect())
}

#[derive(Default)]
struct Arm {
    replies: usize, tokens: u64, rounds: u64, drafted: u64, accepted: u64,
    e_cjk: f64, o_cjk: u64, e_r3: f64, o_r3: u64, e_r4: f64, o_r4: u64,
    cjk_content: usize, cjk_reason: usize, empty_reason: usize, truncated: usize,
    ident_ok: u64, ident_bad: u64, examples: Vec<String>,
}
impl Arm {
    fn book(&mut self, b: &RowBook, t: u32, cjk: &[bool]) {
        self.tokens += 1;
        self.e_cjk += b.cjk; self.e_r3 += b.r3; self.e_r4 += b.r4;
        if cjk.get(t as usize).copied().unwrap_or(false) { self.o_cjk += 1; }
        if b.qt < 1e-3 { self.o_r3 += 1; }
        if b.qt < 1e-4 { self.o_r4 += 1; }
    }
}

/// S4: one arm of sampled replies (k: None = the policy's draft length, Some(0) = plain).
#[allow(clippy::too_many_arguments)]
fn rate_arm(m: &mut TpModel, sp: &mut TpSpec, kit: &Kit, guard: &mut Guard, tok: &QwenTokenizer, stops: &[u32], o: &ModelOpts, b: Budget,
            nl: usize, k_pin: Option<usize>, seed0: u64, chunk: usize, mode: Mode, say: &mut dyn FnMut(String)) -> Result<Arm> {
    let prompts: Vec<Vec<u32>> = SHORT.iter().map(|s| encode(tok, s, true)).collect::<Result<_>>()?;
    let cap = o.gen_tokens.max(8);
    let mut a = Arm::default();
    let mut slot: Vec<Option<Vec<u32>>> = vec![None; nl];
    for l in 0..sp.cap { sp.host[l].active = 0; }
    let (mut next, mut stop, mut calibrated) = (0usize, false, k_pin.is_some());
    let t0 = std::time::Instant::now();
    let finish = |a: &mut Arm, toks: &[u32], cut: bool| {
        a.replies += 1;
        if cut { a.truncated += 1; }
        let body: Vec<u32> = toks.iter().copied().filter(|t| !stops.contains(t)).collect();
        let text = tok.decode(&body, false).unwrap_or_default();
        let (reason, content) = match body.iter().position(|&t| t == THINK_END) {
            Some(i) => (tok.decode(&body[..i], false).unwrap_or_default(), tok.decode(&body[i + 1..], false).unwrap_or_default()),
            None => (text.clone(), String::new()),
        };
        if body.first() == Some(&THINK_END) { a.empty_reason += 1; }
        let (cr, cc) = (reason.chars().any(is_cjk_char), content.chars().any(is_cjk_char));
        if cr { a.cjk_reason += 1; }
        if cc { a.cjk_content += 1; }
        if (cr || cc) && a.examples.len() < 6 { a.examples.push(text.chars().take(240).collect()); }
    };
    loop {
        // free lanes take the next prompt (a new request: prefill, then the first draw = a k = 0 verify)
        for l in 0..nl {
            if slot[l].is_some() || stop { continue; }
            let pi = next % prompts.len();
            let seed = seed0 + next as u64;
            next += 1;
            m.reset_lane_n(l)?;
            sp.reset_lane(m.f.g, l)?;
            let (_, last) = prefill(m, Some(sp), l, &prompts[pi], chunk, mode)?;
            let p = prompts[pi].len();
            let t = sp.start_lane(m, l, p, last, 1.0, seed)?;
            guard.agree("sq1 start", t)?;
            let bk = books(m, sp, kit, &[(l, 0, t)], 0)?;
            // the first draw on one rank, from the gathered row, with the lane's key
            kit.gather(m, &[sp.tlog.ptr() + (l * kit.sh.vlocal * 4) as u64])?;
            let ln = DsLane { seed, inv_t: 1.0, vbase: p as i32 - 1, k: 0, wend: p as i32 - 2, tok: 0, active: 1, ..Default::default() };
            let r = kit.one_rank(&[ln], &[0])?;
            if r[0].gid as u32 == t { a.ident_ok += 1 } else { a.ident_bad += 1 }
            a.book(&bk[0], t, &kit.cjk);
            let done = stops.contains(&t);
            if done { finish(&mut a, &[t], false); } else { slot[l] = Some(vec![t]); }
        }
        let live: Vec<usize> = (0..nl).filter(|&l| slot[l].is_some()).collect();
        for l in 0..sp.cap { sp.host[l].active = (l < nl && slot[l].is_some()) as i32; }
        if live.is_empty() { if stop { break; } else { continue; } }
        if !calibrated && live.len() == nl {
            let line = sp.calibrate(m, nl, &o.model_dir)?;
            guard.agree("sq1 calibration", 0x5CA1)?;
            say(format!("SQ1 S4: {line}"));
            calibrated = true;
        }
        let k = match k_pin { Some(k) => k, None => if calibrated { sp.choose_k(m, nl)? } else { 1 } };
        let pre: Vec<DsLane> = sp.host[..nl].to_vec();
        let em = sp.round(m, nl, k)?;
        a.rounds += 1;
        // commit in order up to a stop token or the cap; book every committed token against its row
        let mut rows = Vec::new();
        let mut take = vec![0usize; nl];
        for &l in &live {
            let toks = slot[l].as_ref().unwrap();
            let mut n = 0;
            for &t in &em[l] {
                if toks.len() + n >= cap { break; }
                n += 1;
                if stops.contains(&t) { break; }
            }
            take[l] = n;
            for (j, &t) in em[l].iter().take(n).enumerate() { rows.push((l, j, t)); }
            if k > 0 { a.drafted += k as u64; a.accepted += sp.host[l].n_acc.max(0) as u64; }
        }
        let bk = books(m, sp, kit, &rows, k)?;
        for (i, &(_, _, t)) in rows.iter().enumerate() { a.book(&bk[i], t, &kit.cjk); }
        if k == 0 {
            // the plain arm: every draw again on one rank from the gathered row with the lane's key
            let ptrs: Vec<u64> = (0..nl).map(|l| sp.tlog.ptr() + (l * kit.sh.vlocal * 4) as u64).collect();
            kit.gather(m, &ptrs)?;
            let mut lanes = pre.clone();
            for ln in lanes.iter_mut() { ln.k = 0; }
            let r = kit.one_rank(&lanes, &(0..nl).collect::<Vec<_>>())?;
            for &l in &live { if r[l].gid as u32 == em[l][0] { a.ident_ok += 1 } else { a.ident_bad += 1 } }
        }
        let mut h = k as u32;
        for &l in &live {
            let toks = slot[l].as_mut().unwrap();
            toks.extend_from_slice(&em[l][..take[l]]);
            h = hash_chains(&[toks.clone()]) ^ h.rotate_left(5);
            let ended = toks.last().map_or(false, |t| stops.contains(t));
            if ended || toks.len() >= cap {
                let t = toks.clone();
                finish(&mut a, &t, !ended);
                slot[l] = None;
            }
        }
        guard.agree("sq1 round", h)?;
        // rank 0 decides when the arm stops taking new prompts (time or count)
        let want = if m.rank == 0 { (t0.elapsed().as_secs_f64() >= b.secs || next >= b.replies) as u32 as f32 } else { 0.0 };
        if !stop && m.bcast_small(&[want])?[0] != 0.0 { stop = true; }
    }
    Ok(a)
}

fn arm_lines(name: &str, a: &Arm, say: &mut dyn FnMut(String)) -> bool {
    let per = |x: usize| 1000.0 * x as f64 / a.replies.max(1) as f64;
    say(format!("SQ1 S4 {name}: {} replies, {} tokens, {} rounds ({:.2} tokens / round), accepted {} of {} drafts; replies with a CJK run: {} in the \
                 answer ({:.1} per 1,000), {} in the reasoning ({:.1} per 1,000); empty reasoning {}; cut at the token cap {}",
                a.replies, a.tokens, a.rounds, a.tokens as f64 / a.rounds.max(1) as f64, a.accepted, a.drafted, a.cjk_content, per(a.cjk_content),
                a.cjk_reason, per(a.cjk_reason), a.empty_reason, a.truncated));
    let mut ok = true;
    for (what, o, e) in [("CJK tokens", a.o_cjk, a.e_cjk), ("tokens with q < 1e-3", a.o_r3, a.e_r3), ("tokens with q < 1e-4", a.o_r4, a.e_r4)] {
        let (le, ge) = poisson_tails(o, e);
        let pass = le >= 1e-4 && ge >= 1e-4;
        ok &= pass;
        say(format!("  {what}: observed {o}, expected {e:.1} under the engine's own q (ratio {:.2}); Poisson P[X<=o] {le:.3e}, P[X>=o] {ge:.3e} -> {}",
                    o as f64 / e.max(1e-12), if pass { "PASS" } else { "FAIL" }));
    }
    if a.ident_ok + a.ident_bad > 0 {
        let pass = a.ident_bad == 0;
        ok &= pass;
        say(format!("  plain draws redrawn on one rank from the gathered rows: {} identical, {} different -> {}", a.ident_ok, a.ident_bad, if pass { "PASS" } else { "FAIL" }));
    }
    for e in &a.examples { say(format!("  example with CJK: <<<{e}>>>")); }
    ok
}

/// Run the part. Returns whether every gate held.
#[allow(clippy::too_many_arguments)]
pub fn run(m: &mut TpModel, sp: &mut TpSpec, guard: &mut Guard, tok: &QwenTokenizer, stops: &[u32], o: &ModelOpts, b: Budget, say: &mut dyn FnMut(String)) -> Result<bool> {
    let g = m.f.g;
    let chunk = o.chunk;
    let mode = crate::dsv41::ced::schedule::mode_from_flags()?;
    let v = m.cfg.vocab;
    let sh = sp.sh;
    let nl = m.lanes.len().min(sp.cap).min(4);
    ensure!(nl >= 1, "sq1: no lane");
    let cjk: Vec<bool> = (0..v as u32).map(|i| is_cjk_bytes(&tok.token_bytes(i, false))).collect();
    let cjk_local: Vec<u32> = (sh.v0..sh.v0 + sh.vlocal).filter(|&i| cjk[i]).map(|i| (i - sh.v0) as u32).collect();
    let tp = Round::new(g, ONE_CAP, 1, sh.nblk(), m.world)?;
    let kit = Kit {
        g, v, sh, world: m.world,
        full: g.buf(ONE_CAP * v * 4)?, srow: g.buf(sh.vlocal * 4)?, red: g.buf(ONE_CAP * 4 * 4)?,
        one: Round::new(g, ONE_CAP, 1, Shard { v0: 0, vlocal: v }.nblk(), 1)?,
        xtp: TpXchg::new(g, m.comm, m.cs, tp.rows_max * REC)?, tp, cjk_local, cjk,
    };
    let mut ok = true;
    say(format!("SQ1: lanes {nl}, vocabulary {v} ({} CJK tokens, {} on this rank's slice {}..{}), budget {b:?}, reply cap {} tokens",
                kit.cjk.iter().filter(|&&x| x).count(), kit.cjk_local.len(), sh.v0, sh.v0 + sh.vlocal, o.gen_tokens));

    // ---------------------------------------------------------------- contexts
    let hello = encode(tok, "hello", true)?;
    let mut owner = hello.clone();
    owner.push(THINK_END);
    owner.extend(tok.encode(OWNER_HEAD, false)?);
    let stray = tok.encode(OWNER_REPLY, false)?;
    let n_head = tok.encode(OWNER_HEAD, false)?.len();
    ensure!(stray.len() > n_head + 1 && stray[..n_head] == owner[hello.len() + 1..], "the owner's reply does not tokenize with its head as a prefix");
    let (s1, s2) = (stray[n_head], stray[n_head + 1]);
    let mut owner2 = owner.clone();
    owner2.push(s1);
    let mut free = encode(tok, "Give me ten random English words.", false)?;
    free.extend(tok.encode("Here are ten random English words:\n\n1.", false)?);
    say(format!("SQ1 contexts: hello prompt {} tokens {:?}; owner = prompt + </think> + {:?} ({} tokens); stray tokens {s1} {:?}, {s2} {:?}",
                hello.len(), hello, OWNER_HEAD, owner.len(), tok.decode(&[s1], false).unwrap_or_default(), tok.decode(&[s2], false).unwrap_or_default()));
    let ctxs = [Ctx { name: "owner", ids: owner.clone() }, Ctx { name: "owner+stray1", ids: owner2 }, Ctx { name: "hello prompt", ids: hello.clone() },
                Ctx { name: "free list", ids: free.clone() }];

    // ---------------------------------------------------------------- S1 + S2
    m.barrier(0x5610)?;
    let mut exact: HashMap<&'static str, Vec<f64>> = HashMap::new();
    let mut owner_row: Vec<f32> = Vec::new();
    for (ci, c) in ctxs.iter().enumerate() {
        let row = exact_row(m, sp, &kit, 0, &c.ids, chunk, mode)?;
        if ci == 0 { owner_row = row.clone(); }
        let q = softmax64(&row);
        let cj: f64 = q.iter().zip(&kit.cjk).filter(|(_, &c)| c).map(|(p, _)| p).sum();
        let ent: f64 = -q.iter().filter(|&&p| p > 0.0).map(|p| p * p.ln()).sum::<f64>();
        let mut line = format!("SQ1 S1 {} (pending {:?} at position {}): entropy {ent:.3} nats, CJK mass {cj:.3e}; top: {}", c.name,
                               tok.decode(&c.ids[c.ids.len() - 1..], false).unwrap_or_default(), c.ids.len() - 1, top_line(tok, &q, 8));
        match c.name {
            "owner" => line += &format!("; q(stray {s1}) = {:.4e} (rank {})", q[s1 as usize], rank_of(&q, s1 as usize)),
            "owner+stray1" => line += &format!("; q(stray {s2}) = {:.4e} (rank {})", q[s2 as usize], rank_of(&q, s2 as usize)),
            "hello prompt" => line += &format!("; q(</think>: empty reasoning) = {:.4e}", q[THINK_END as usize]),
            _ => {}
        }
        say(line);
        if c.name == "owner" || c.name == "free list" {
            // S2: the same row drawn through the TP pass and on one rank, 4,096 keys
            let mut counts: HashMap<i32, usize> = HashMap::new();
            let (mut same, mut aid_same, mut dl) = (0usize, 0usize, 0f64);
            for bt in 0..DRAWS_ROW / ONE_CAP {
                let lanes: Vec<DsLane> = (0..ONE_CAP).map(|i| DsLane { seed: 0x5120_0000 + ((ci * 1000 + bt) * ONE_CAP + i) as u64, inv_t: 1.0,
                                                                     vbase: c.ids.len() as i32 - 1, k: 0, wend: c.ids.len() as i32 - 2, active: 1, ..Default::default() }).collect();
                let a = kit.tp_rank(&lanes)?;
                let r = kit.one_rank(&lanes, &vec![0; ONE_CAP])?;
                for (x, y) in a.iter().zip(&r) {
                    if x.gid == y.gid { same += 1; }
                    if x.aid == y.aid { aid_same += 1; }
                    dl = dl.max(((x.lse() - y.lse()) as f64).abs());
                    *counts.entry(x.gid).or_insert(0) += 1;
                }
                guard.agree("sq1 draws", a.iter().fold(0u32, |h, x| h.wrapping_mul(31) ^ x.gid as u32))?;
            }
            let (st, df, pv) = one_sample(&counts, &probs_of(&q));
            let pass = same == DRAWS_ROW && aid_same == DRAWS_ROW && pv >= 1e-4;
            ok &= pass;
            say(format!("SQ1 S2 {}: {DRAWS_ROW} keyed draws, TP{} vocab pass vs one rank on the gathered row: {same} draws and {aid_same} argmax identical, \
                         max |lse diff| {dl:.2e}; draws vs exact q: chi2 {st:.1} on {df} df, p {pv:.3} -> {}", c.name, m.world, if pass { "PASS" } else { "FAIL" }));
        }
        exact.insert(c.name, q);
    }
    m.settle()?;

    // ---------------------------------------------------------------- S3
    m.barrier(0x5620)?;
    for (ci, (cname, ctx, ks)) in [("owner", &owner, vec![0usize, 2, 5]), ("free list", &free, vec![0, 1, 2, 4, 5])].into_iter().enumerate() {
        let (p, saved) = setup_ctx(m, sp, nl, ctx, chunk, mode)?;
        let q0 = exact.get(cname).context("exact row")?.clone();
        let mut base: Option<Vec<Vec<u32>>> = None;
        let mut conds: Vec<(String, usize, i32)> = ks.iter().map(|&k| (format!("k {k}"), k, 0)).collect();
        if cname == "free list" { conds.push(("NEGATIVE CONTROL k 5, acceptance ratio swapped".into(), 5, TEST_SWAP_RULE)); }
        for (di, (dname, k, flags)) in conds.iter().enumerate() {
            let t = std::time::Instant::now();
            let seed0 = 0x5130_0000_0000u64 + ((ci as u64) << 32) + ((di as u64) << 24);
            let (ch, mean_acc) = trials(m, sp, guard, &saved, p, *k, *flags, b.trials, seed0)?;
            let fit = one_sample(&counts_at(&ch, 0), &probs_of(&q0));
            let mut cells = vec![format!("first token vs exact q: chi2 {:.1} / {} df, p {:.3}", fit.0, fit.1, fit.2)];
            let mut pmin = fit.2;
            if let Some(bs) = &base {
                for j in 0..G_TOK {
                    let (st, df, pv) = two_sample(&counts_at(bs, j), &counts_at(&ch, j));
                    cells.push(format!("pos {j} vs k 0: chi2 {st:.1} / {df} df, p {pv:.3}"));
                    pmin = pmin.min(pv);
                }
            }
            let verdict = if *flags != 0 {
                let det = pmin < 1e-4;
                ok &= det;
                if det { "DETECTED (the cell has power)" } else { "NOT DETECTED (no power: FAIL)" }
            } else {
                let pass = pmin >= 1e-4;
                ok &= pass;
                if pass { "PASS" } else { "FAIL" }
            };
            say(format!("SQ1 S3 {cname}, {dname}: {} trials x {G_TOK} tokens in {:.0} s, {mean_acc:.2} drafts accepted per first round; {}; min p {pmin:.2e} -> {verdict}",
                        ch.len(), t.elapsed().as_secs_f64(), cells.join("; ")));
            if *k == 0 && *flags == 0 { base = Some(ch); }
        }
        for l in 0..nl { m.reset_lane_n(l)?; sp.reset_lane(g, l)?; }
    }
    m.settle()?;

    // ---------------------------------------------------------------- S4
    m.barrier(0x5630)?;
    for (ai, (name, k)) in [("speculation on (the policy's draft length)", None), ("speculation off (k = 0)", Some(0usize))].into_iter().enumerate() {
        let t = std::time::Instant::now();
        let a = rate_arm(m, sp, &kit, guard, tok, stops, o, b, nl, k, 0x5140_0000_0000u64 + ((ai as u64) << 32), chunk, mode, say)?;
        ok &= arm_lines(&format!("{name}, {:.0} s", t.elapsed().as_secs_f64()), &a, say);
        for l in 0..nl { m.reset_lane_n(l)?; sp.reset_lane(g, l)?; }
    }
    m.settle()?;

    // ---------------------------------------------------------------- S6: state carried between requests
    // the owner's row again, after S3 / S4 ran hundreds of requests on every lane: on lane 0 and on the last lane, bit for bit
    m.barrier(0x5635)?;
    for l in [0, nl - 1] {
        let row = exact_row(m, sp, &kit, l, &owner, chunk, mode)?;
        let diff = row.iter().zip(&owner_row).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let pass = diff == 0 && row.len() == owner_row.len();
        ok &= pass;
        say(format!("SQ1 S6 owner row on lane {l} after the S3 / S4 requests vs the fresh boot's lane 0: {diff} of {v} logits differ -> {}", if pass { "PASS" } else { "FAIL" }));
        m.reset_lane_n(l)?;
        sp.reset_lane(g, l)?;
    }
    m.settle()?;

    // ---------------------------------------------------------------- S5
    m.barrier(0x5640)?;
    let mut ids = hello.clone();
    ids.push(THINK_END);
    ids.extend_from_slice(&stray);
    ids.push(1);
    let n = ids.len();
    m.reset_lane_n(0)?;
    let _ = m.forward_segs(&[(0, ids.as_slice())], super::fwd::HeadWant::None, None)?;
    let rows_buf = g.buf(n * sh.vlocal * 4)?;
    let mut r0 = 0;
    while r0 < n {
        let c = (n - r0).min(super::fwd::HEAD_MAX);
        m.head_logits(r0, c, rows_buf.ptr() + (r0 * sh.vlocal * 4) as u64)?;
        r0 += c;
    }
    let mut all = Vec::with_capacity(n * v);
    for c0 in (0..n).step_by(ONE_CAP) {
        let c = (n - c0).min(ONE_CAP);
        let ptrs: Vec<u64> = (0..c).map(|i| rows_buf.ptr() + ((c0 + i) * sh.vlocal * 4) as u64).collect();
        kit.gather(m, &ptrs)?;
        all.extend(g.down::<f32>(kit.full.ptr(), c * v)?);
    }
    m.reset_lane_n(0)?;
    let mut tf = Vec::new();
    for r in hello.len() - 1..n - 1 {
        let q = softmax64(&all[r * v..(r + 1) * v]);
        tf.push(format!("{:?} {:.3e}", tok.decode(&[ids[r + 1]], false).unwrap_or_default(), q[ids[r + 1] as usize]));
    }
    say(format!("SQ1 S5 owner's reply teacher-forced (one every-layer call of {n} rows; q of each next token from row {}): {}", hello.len() - 1, tf.join(", ")));
    if m.rank == 0 {
        if let Some(dir) = &o.q1_out {
            std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
            let bytes: Vec<u8> = all.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(dir.join("sq1_rows.f32"), bytes)?;
            let meta = serde_json::json!({ "ids": ids, "rows": n, "vocab": v, "world": m.world, "layout": "f32 little-endian [rows][vocab], row r = the logits after ids[0..=r]" });
            std::fs::write(dir.join("sq1_rows.json"), serde_json::to_string_pretty(&meta)?)?;
            say(format!("SQ1 S5: {n} x {v} logits -> {}", dir.join("sq1_rows.f32").display()));
        } else {
            say("SQ1 S5: no --dsv41-q1-score-out: the rows were not written".into());
        }
    }
    m.settle()?;
    say(format!("SQ1 RESULT: {}", if ok { "PASS" } else { "FAIL" }));
    Ok(ok)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv41::chat::{jx, render};
    use crate::dsv4_chat::Json;

    fn render_msgs(msgs: Vec<Json>, thinking: bool) -> String {
        let mode = if thinking { render::ThinkingMode::Thinking } else { render::ThinkingMode::Chat };
        render::encode_messages(&msgs, &render::EncodeOptions::new(mode)).unwrap().prompt
    }
    fn user(s: &str) -> Json { jx::obj(vec![("role", jx::s("user")), ("content", jx::s(s))]) }

    #[test]
    fn dsv41_sq1_hello_prompt_and_history_match_encoding_py() {
        // DeepSeek's encoding/encoding.py (encode_messages, thinking mode, default effort), run 2026-11-19
        let head = "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>hello";
        assert_eq!(render_msgs(vec![user("hello")], true), format!("{head}<｜Assistant｜><think>"));
        let want = format!("{head}<｜Assistant｜></think>{OWNER_REPLY}<｜end▁of▁sentence｜><｜User｜>what?<｜Assistant｜><think>");
        for rc in [Some(""), None] {
            let mut a = vec![("role", jx::s("assistant")), ("content", jx::s(OWNER_REPLY))];
            if let Some(r) = rc { a.push(("reasoning_content", jx::s(r))); }
            assert_eq!(render_msgs(vec![user("hello"), jx::obj(a), user("what?")], true), want, "reasoning_content {rc:?}");
        }
    }

    #[test]
    fn dsv41_sq1_cjk_classes() {
        assert!(is_cjk_bytes("第一".as_bytes()));
        assert!(is_cjk_bytes("行".as_bytes()));
        assert!(is_cjk_bytes("，".as_bytes()));
        assert!(is_cjk_bytes(&"第".as_bytes()[..2]));   // a partial Han character
        assert!(!is_cjk_bytes(b" Hello"));
        assert!(!is_cjk_bytes("\u{2019}s".as_bytes()));  // a curly apostrophe is not CJK
        assert!(!is_cjk_bytes(&"\u{2014}".as_bytes()[..1]));
        assert!(!is_cjk_bytes("é".as_bytes()));
    }

    #[test]
    fn dsv41_sq1_poisson_tails() {
        // P[X <= 0 | 3] = e^-3; P[X >= 1 | 3] = 1 - e^-3; P[X >= 10 | 3] ~ 1.1e-3
        let (le, ge) = poisson_tails(0, 3.0);
        assert!((le - (-3f64).exp()).abs() < 1e-9 && ge == 1.0);
        let (_, ge) = poisson_tails(1, 3.0);
        assert!((ge - (1.0 - (-3f64).exp())).abs() < 1e-9);
        let (_, ge) = poisson_tails(10, 3.0);
        assert!((ge - 0.001_102).abs() < 2e-5, "{ge}");
        assert_eq!(budget(None).unwrap().trials, 800);
        assert!(budget(Some("1,2")).is_err());
    }

    #[test]
    fn dsv41_sq1_hello_tokens_match_the_reference_tokenizer() {
        // the prompt's ids: DeepSeek's encoding.py (thinking, default effort) through tokenizer.json (tokenizers, no special-token
        // insertion), run 2026-11-19. "hello" is id 33310; "hi" (6366) gives a 31-token prompt too.
        let want: [u32; 31] = [0, 128799, 77666, 288, 9504, 482, 28, 223, 2402, 343, 16801, 223, 19, 15, 1457, 14, 270, 4077, 270, 1990, 14, 270, 850,
                               15146, 270, 22805, 868, 128803, 33310, 128804, 128821];
        let dir = std::path::Path::new("/models/DeepSeek-V4.1-Flash-1017");
        let Ok(tok) = QwenTokenizer::from_file(&dir.join("tokenizer.json").to_string_lossy()) else { eprintln!("skipped: no tokenizer at {}", dir.display()); return; };
        assert_eq!(encode(&tok, "hello", true).unwrap(), want);
        let r = tok.encode(OWNER_REPLY, false).unwrap();
        assert_eq!(&r[..6], &[19923, 3, 1730, 588, 2879, 719]);
    }
}
