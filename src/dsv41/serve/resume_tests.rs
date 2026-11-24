//! Legs CE2 / CE3: the prefix-pool resume of a CED prefill through the REAL lane loop, pool, snapshot plans and CED schedule
//! (no GPU).
//!
//! `CStub` is a rounds forward that keeps, per lane, exactly what the TP forward's ranks keep for this question: the position, the
//! state kind (the real `ced::schedule`: `Schedule::new_with` / `resumed_with` / `state_kind_at`), and since CE3 whether the shared
//! DSpark taps buffer holds the lane's prompt tail and whether a restored state carries one. It applies the TP forward's own rules
//! (`tp::serve::full_at_for`, `tp::serve::tail_resume_ok`, `schedule::snapshot_label`, `schedule::resume_check`); a refusal inside its
//! `prefill` makes it dead (`fatal`), as an op failure makes `TpForward` dead. It logs every prefill call, so the tests count the calls
//! a cold prompt costs (the price PP1 measured: each extra call runs all 40 layers, ~0.9 s).
//!
//! Rows: FIXB's death (`87183a55`) is `Behaviour::Old`; CE2's near-end full grid state is `Behaviour::NearEnd` (= CE3 with
//! `--dsv41-prefix-near-end on`); CE3's default is `Behaviour::Ce3`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{bail, ensure, Context, Result};

use super::engine::{self, EngineCfg, Job, Out};
use super::forward::{Forward, RowOut, Snapshot, StepRow};
use super::prefix::MIN_KEEP;
use super::sampler::Sampling;
use crate::dsv41::ced::schedule::{self, Mode, Schedule, StateKind};
use crate::dsv41::config::Dsv41Config;
use crate::dsv41::tp::serve::{full_at_for, tail_resume_ok};

const EOS: u32 = 1;
const TAP: usize = crate::dsv41::tp::spec::TAP_ROWS;
type Log = Arc<Mutex<Vec<String>>>;

fn cfg() -> Option<Dsv41Config> {
    let d = std::path::Path::new("/models/DeepSeek-V4.1-Flash-1017");
    if !d.join("config.json").exists() { return None; }
    Dsv41Config::load(d).ok()
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Behaviour {
    /// `87183a55`: encoder-only states labelled `MIN_KEEP`, the 128-grid near the end, no check before the restore, no tails
    Old,
    /// the old labels and grid, but the shell's check before the restore (defence in depth alone)
    OldChecked,
    /// CE3 with `--dsv41-prefix-near-end on` (CE2's full grid state near the end; prompt-end tails too)
    NearEnd,
    /// CE3 default: snapshots only at call ends below the cone + the prompt end (full, with its tail)
    Ce3,
}

#[derive(Clone, Copy)]
struct CLane { pos: usize, kind: StateKind, origin: Option<(usize, StateKind)>, emitted: usize, prompt: usize }
const FRESH: CLane = CLane { pos: 0, kind: StateKind::Full, origin: None, emitted: 0, prompt: 0 };

/// A kept state: (position, kind, carries a tail, bytes)
type St = (usize, StateKind, bool, u64);

#[derive(Default)]
struct GpuMem {
    live_bytes: u64,
    peak_bytes: u64,
    free_q: Vec<u64>,
    flushes: u64,
    freed_snaps: u64,
}

struct StubSnapHandle { id: u64, mem: Arc<Mutex<GpuMem>> }
impl Drop for StubSnapHandle {
    fn drop(&mut self) {
        if let Ok(mut m) = self.mem.lock() { m.free_q.push(self.id); }
    }
}

struct CStub {
    cfg: Dsv41Config,
    b: Behaviour,
    sized: bool,
    lanes: Vec<CLane>,
    total: Vec<Option<usize>>,
    first: Vec<bool>,
    full_at: Vec<Option<usize>>,
    snaps: Arc<Mutex<HashMap<u64, St>>>,
    mem: Arc<Mutex<GpuMem>>,
    tap_owner: Option<(usize, usize)>,
    lane_tail: Vec<Option<usize>>,
    next: u64,
    dead: bool,
    log: Log,
}

impl CStub {
    fn new(cfg: Dsv41Config, b: Behaviour, lanes: usize, log: Log) -> Self {
        CStub { cfg, b, sized: false, lanes: vec![FRESH; lanes], total: vec![None; lanes], first: vec![true; lanes], full_at: vec![None; lanes],
                snaps: Arc::default(), mem: Arc::default(), tap_owner: None, lane_tail: vec![None; lanes], next: 1, dead: false, log }
    }
    fn with_realistic_bytes(mut self) -> Self { self.sized = true; self }
    fn say(&self, s: String) { self.log.lock().unwrap().push(s); }
    fn ce3(&self) -> bool { matches!(self.b, Behaviour::NearEnd | Behaviour::Ce3) }
    fn calc_bytes(&self, pos: usize, tail: bool) -> u64 {
        if !self.sized { return 1 << 20; }
        let t = if tail { (pos.min(TAP) * self.cfg.dspark_targets.len() * self.cfg.hidden * 2) as u64 } else { 0 };
        crate::dsv41::tp::mem::snap_bytes(&self.cfg, pos) + t
    }
    fn drain_free_q(&self) {
        let ids: Vec<u64> = {
            let mut m = self.mem.lock().unwrap();
            if m.free_q.is_empty() { return; }
            m.flushes += 1;
            std::mem::take(&mut m.free_q)
        };
        let mut snaps = self.snaps.lock().unwrap();
        let mut m = self.mem.lock().unwrap();
        for id in ids {
            if let Some((_, _, _, b)) = snaps.remove(&id) {
                m.live_bytes = m.live_bytes.saturating_sub(b);
                m.freed_snaps += 1;
            }
        }
    }
}

impl Forward for CStub {
    fn vocab(&self) -> usize { 130_000 }
    fn lanes(&self) -> usize { self.lanes.len() }
    fn max_seq_len(&self) -> usize { 1 << 20 }
    fn name(&self) -> &'static str { "ced-stub" }
    fn rounds(&self) -> bool { true }
    fn fatal(&self) -> bool { self.dead }
    fn prefill_tail(&self) -> usize { TAP }
    fn has_tail_snaps(&self) -> bool { self.ce3() }
    fn state_bytes(&self, pos: usize) -> Option<u64> { Some(self.calc_bytes(pos, false)) }
    fn state_bytes_for(&self, pos: usize, is_tail: bool) -> Option<u64> { Some(self.calc_bytes(pos, is_tail)) }
    fn flush_frees(&mut self) -> Result<()> { self.drain_free_q(); Ok(()) }
    fn encoder_cone(&self) -> Option<usize> { if self.ce3() { Some(schedule::snapshot_label(&self.cfg, Mode::Ced, StateKind::Encoder)) } else { None } }
    fn begin_prefill(&mut self, lane: usize, total: usize) { self.total[lane] = Some(total); self.first[lane] = true; self.full_at[lane] = None; }
    fn prefill_snapshots(&mut self, lane: usize, from: usize, total: usize, at: &[usize]) {
        if self.ce3() { self.full_at[lane] = full_at_for(from, total, at); }
    }
    /// `TpForward::can_resume`
    fn can_resume(&self, snap: &Snapshot, at: usize, prompt_len: usize) -> Result<()> {
        if self.b == Behaviour::Old { return Ok(()); }
        let id = snap.data.downcast_ref::<StubSnapHandle>().context("foreign snapshot")?.id;
        let (pos, kind, tail, _) = *self.snaps.lock().unwrap().get(&id).context("no such state")?;
        let r = (|| {
            ensure!(at <= pos && pos - at <= 6, "a state at {pos} cannot be rolled back to {at}");
            schedule::resume_check(&self.cfg, Mode::Ced, prompt_len, at, kind)?;
            ensure!(tail_resume_ok(prompt_len, at, tail.then_some(pos)), "no tail");
            Ok(())
        })();
        if let Err(e) = &r { self.say(format!("refused {pos} {kind:?}: {e:#}")); }
        r
    }
    /// The ranks' side of `Op::Prefill` (`TpServe::exec` + `TpModel::prefill_part`), reduced to the schedule, the state kind and the taps.
    fn prefill(&mut self, lane: usize, from: usize, tokens: &[u32]) -> Result<RowOut> {
        if self.dead { bail!("dead"); }
        let ln = self.lanes[lane];
        if ln.pos != from { bail!("lane {lane} at {} asked to resume at {from}", ln.pos); }
        let total = self.total[lane].unwrap_or(from + tokens.len()).max(from + tokens.len());
        let first = std::mem::replace(&mut self.first[lane], false);
        if first { self.lanes[lane].origin = Some((ln.pos, ln.kind)); }
        let (k0, kind0) = self.lanes[lane].origin.unwrap();
        let fa = self.full_at[lane];
        let end = from + tokens.len();
        self.tap_owner = None;
        let tail = self.lane_tail[lane];
        let r = (|| {
            let sched = if k0 == 0 { Schedule::new_with(&self.cfg, total, Mode::Ced, fa)? } else { Schedule::resumed_with(&self.cfg, total, Mode::Ced, k0, kind0, fa)? };
            if end == total { ensure!(tail_resume_ok(total, k0, tail), "the taps of rows below {k0} are missing"); }
            Ok(sched)
        })();
        let sched = match r {
            Ok(s) => s,
            Err(e) => { self.dead = true; self.say(format!("FATAL prefill: {e:#}")); return Err(e); }
        };
        self.lanes[lane].pos = end;
        self.lanes[lane].kind = sched.state_kind_at(end);
        if end >= total {
            self.total[lane] = None; self.first[lane] = true; self.full_at[lane] = None;
            self.lane_tail[lane] = None;
            self.lanes[lane].prompt = end;
            self.tap_owner = Some((lane, end));
        }
        self.say(format!("call {lane} {from}..{end} of {total}"));
        Ok(RowOut::default())
    }
    fn step(&mut self, _rows: &[StepRow]) -> Result<Vec<RowOut>> { bail!("rounds only") }
    fn snapshot(&mut self, lane: usize, pos: usize) -> Result<Snapshot> {
        self.drain_free_q();
        let ln = self.lanes[lane];
        let kind = if pos > ln.pos { StateKind::Full } else { ln.kind }; // a decode position is full
        let tail = self.ce3() && self.tap_owner == Some((lane, pos)) && kind == StateKind::Full;
        let min_suffix = match self.b {
            Behaviour::Old | Behaviour::OldChecked => if kind == StateKind::Full { 0 } else { MIN_KEEP },
            _ => schedule::snapshot_label(&self.cfg, Mode::Ced, kind),
        };
        let bytes = self.calc_bytes(pos, tail);
        let id = self.next;
        self.next += 1;
        {
            let mut m = self.mem.lock().unwrap();
            m.live_bytes += bytes;
            m.peak_bytes = m.peak_bytes.max(m.live_bytes);
        }
        self.snaps.lock().unwrap().insert(id, (pos, kind, tail, bytes));
        self.say(format!("snap {pos} {kind:?} min_suffix {min_suffix} tail {tail}"));
        Ok(Snapshot { pos, bytes, data: Box::new(StubSnapHandle { id, mem: self.mem.clone() }), min_suffix, tail })
    }
    fn restore(&mut self, lane: usize, snap: &Snapshot) -> Result<()> {
        let id = snap.data.downcast_ref::<StubSnapHandle>().context("foreign snapshot")?.id;
        let (pos, kind, tail, _) = *self.snaps.lock().unwrap().get(&id).context("no such state")?;
        self.lanes[lane] = CLane { pos, kind, origin: None, emitted: 0, prompt: 0 };
        self.lane_tail[lane] = tail.then_some(pos);
        if self.tap_owner.map_or(false, |t| t.0 == lane) { self.tap_owner = None; }
        self.say(format!("restore {pos} {kind:?}"));
        Ok(())
    }
    fn truncate(&mut self, lane: usize, to: usize) -> Result<()> {
        let l = &mut self.lanes[lane];
        if l.emitted == 0 {
            // a restored state rolled back before its prefill (`TpModel::rollback`: at most 6 rows, the state becomes full)
            if to > l.pos || l.pos - to > 6 { bail!("rollback {} -> {to}", l.pos); }
            l.pos = to;
            l.kind = StateKind::Full;
            self.say(format!("rollback {to}"));
            return Ok(());
        }
        if to < l.prompt { bail!("truncate below the prompt"); }
        l.emitted = to - l.prompt + 1;
        Ok(())
    }
    fn release(&mut self, lane: usize) {
        self.lanes[lane] = FRESH;
        self.lane_tail[lane] = None;
        if self.tap_owner.map_or(false, |t| t.0 == lane) { self.tap_owner = None; }
    }
    fn start(&mut self, lane: usize, _s: &Sampling, _seed: u64) -> Result<u32> { self.lanes[lane].emitted = 1; Ok(1000) }
    fn round(&mut self, lanes: &[usize]) -> Result<Vec<Vec<u32>>> {
        self.tap_owner = None;
        Ok(lanes.iter().map(|&li| {
            let l = &mut self.lanes[li];
            let c: Vec<u32> = (0..3).map(|i| if l.emitted + i >= 4 { EOS } else { 1000 + (l.emitted + i) as u32 }).collect();
            l.emitted += 3;
            c
        }).collect())
    }
    fn lane_pos(&self, lane: usize) -> Option<usize> { let l = &self.lanes[lane]; Some(l.prompt.max(l.pos) + l.emitted.saturating_sub(1)) }
}

fn job(prompt: Vec<u32>) -> (Job, tokio::sync::mpsc::UnboundedReceiver<Out>) {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    (Job { prompt, max_new: 64, min_tokens: 0, sampling: Sampling::default(), seed: 1, ignore_eos: false, starts_in_think: false, think_budget: None,
           cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)), tx, score: false,
           vl_types: Vec::new(), vl_images: Vec::new(), image_spans: Vec::new() }, rx)
}

/// One request's outcome: cached tokens at admission, finished, the failure text, the prefill calls it made.
#[derive(Debug, Clone)]
struct R { cached: Option<usize>, done: bool, err: Option<String>, calls: usize }

fn drain(mut rx: tokio::sync::mpsc::UnboundedReceiver<Out>) -> (Option<usize>, bool, Option<String>) {
    let (mut cached, mut done, mut err) = (None, false, None);
    while let Some(o) = rx.blocking_recv() {
        match o { Out::Admitted { cached: c } => cached = Some(c), Out::Done { .. } => { done = true; break; } Out::Failed(e) => { err = Some(e); break; } _ => {} }
    }
    (cached, done, err)
}

fn ecfg(b: Behaviour) -> EngineCfg {
    let mut c = EngineCfg::simple(1, vec![EOS]);
    c.prefix = Some((1 << 40, MIN_KEEP));
    c.prefill_chunk = 2048;
    if matches!(b, Behaviour::NearEnd | Behaviour::Ce3) { c.grid_snaps = 1; } // as the TP server sets it
    c.near_end = b == Behaviour::NearEnd;
    c
}

/// Send `prompts` one after the other to a fresh engine; per request its outcome, and the stub's log.
fn serve(c: &Dsv41Config, b: Behaviour, prompts: &[Vec<u32>]) -> (Vec<R>, Vec<String>) {
    let log: Log = Arc::default();
    let h = engine::spawn(Box::new(CStub::new(c.clone(), b, 1, log.clone())), ecfg(b));
    let mut out = Vec::new();
    for p in prompts {
        let before = log.lock().unwrap().iter().filter(|l| l.starts_with("call ")).count();
        let (j, rx) = job(p.clone());
        if h.submit(j).is_err() { out.push(R { cached: None, done: false, err: Some("not accepted".into()), calls: 0 }); continue; }
        let (cached, done, err) = drain(rx);
        let calls = log.lock().unwrap().iter().filter(|l| l.starts_with("call ")).count() - before;
        out.push(R { cached, done, err, calls });
    }
    let l = log.lock().unwrap().clone();
    (out, l)
}

fn ok(r: &[R], log: &[String]) {
    assert!(r.iter().all(|x| x.done && x.err.is_none()), "every request finishes: {r:?}\n{log:?}");
    assert!(!log.iter().any(|l| l.starts_with("refused") || l.starts_with("FATAL")), "no refusal, no death: {log:?}");
}

/// FIXB's G4 agent row: a 5,286-token prompt.
fn agent() -> Vec<u32> { words(5286, 3) }
fn words(n: usize, salt: u32) -> Vec<u32> { (0..n as u32).map(|i| 3 + (i.wrapping_mul(2_654_435_761) ^ salt.wrapping_mul(0x9E37_79B9)) % 100_000).collect() }
fn cat(a: &[u32], b: &[u32]) -> Vec<u32> { let mut v = a.to_vec(); v.extend_from_slice(b); v }

#[test]
fn dsv41_ce2_fixb_repro_the_old_labels_kill_the_forward_on_the_repeat() {
    let Some(c) = cfg() else { return };
    let (r, log) = serve(&c, Behaviour::Old, &[agent(), agent(), agent()]);
    assert!(r[0].done, "the first request finishes: {r:?}");
    // the pool offered the encoder-only state at 5,120 (labelled 134), the lane driver refused it, the forward died
    assert_eq!(r[1].cached, Some(5120), "{r:?}");
    let e = r[1].err.as_deref().unwrap_or("");
    assert!(e.contains("an encoder-only state at 5120 cannot resume a 5286-row prompt exactly"), "{e}\n{log:?}");
    assert!(log.iter().any(|l| l.starts_with("FATAL prefill")), "{log:?}");
    assert!(!r[2].done, "{r:?}");
}

#[test]
fn dsv41_ce2_the_check_alone_falls_back_cold_and_the_engine_lives() {
    let Some(c) = cfg() else { return };
    let (r, log) = serve(&c, Behaviour::OldChecked, &[agent(), agent(), agent()]);
    assert!(r.iter().all(|x| x.done && x.err.is_none()), "every request finishes: {r:?}\n{log:?}");
    assert_eq!(r[1].cached, Some(0), "{r:?}");
    assert_eq!(log.iter().filter(|l| l.starts_with("refused")).count(), 8, "4 refusals per repeat: {log:?}");
    assert!(!log.iter().any(|l| l.starts_with("FATAL")), "{log:?}");
}

/// CE3: a cold prompt costs exactly the calls of a prefill without the prefix cache (no cut), at 900 tokens to 128K; CE2's near-end
/// state costs one more call on every one of them.
#[test]
fn dsv41_ce3_a_cold_prompt_makes_no_extra_prefill_call() {
    let Some(c) = cfg() else { return };
    for (n, salt) in [(5286usize, 1u32), (6000, 2), (32_080, 3), (131_072, 4), (2600, 5), (900, 6)] {
        let p = words(n, salt);
        let plain = n.div_ceil(2048) - usize::from(n > 2048 && n % 2048 != 0 && n % 2048 < TAP);
        let (r, log) = serve(&c, Behaviour::Ce3, &[p.clone()]);
        ok(&r, &log);
        assert_eq!(r[0].calls, plain, "{n}: CE3 cold calls");
        assert!(log.iter().any(|l| l == &format!("snap {n} Full min_suffix 0 tail true")), "{n}: prompt end kept with its tail: {log:?}");
        let mid: Vec<&String> = log.iter().filter(|l| l.starts_with("snap ") && l.contains(" min_suffix ") && !l.starts_with(&format!("snap {n} "))
                                                 && l.split(' ').nth(1).and_then(|x| x.parse::<usize>().ok()).map_or(false, |x| x < n)).collect();
        assert!(mid.iter().all(|l| l.contains("Encoder min_suffix 2541 tail false")), "{n}: mid-prompt states are encoder-only and labelled: {mid:?}");
        let (r2, log2) = serve(&c, Behaviour::NearEnd, &[p.clone()]);
        ok(&r2, &log2);
        assert_eq!(r2[0].calls, plain + 1, "{n}: CE2's near-end state costs one more call");
    }
}

/// CE3: where each later prompt resumes, for every resume class, default vs `--dsv41-prefix-near-end on`.
#[test]
fn dsv41_ce3_every_resume_class_resumes_where_stated() {
    let Some(c) = cfg() else { return };
    let a = agent(); // 5,286 tokens; call ends 2,048 / 4,096 / 5,286; usable encoder state: 2,048 (<= 5,286 - 2,541)
    let edit3 = cat(&a[..5283], &words(3, 77));
    let edit150 = cat(&a[..5136], &words(150, 78));
    let edit300 = cat(&a[..4986], &words(300, 79));
    let follow = cat(&a, &words(700, 80));
    let ext50 = cat(&a, &words(50, 81));
    let shorter = a[..5000].to_vec();
    let prompts = [a.clone(), a.clone(), edit3, edit150, edit300, follow, ext50, shorter];
    let (r, log) = serve(&c, Behaviour::Ce3, &prompts);
    ok(&r, &log);
    let cached: Vec<usize> = r.iter().map(|x| x.cached.unwrap()).collect();
    // cold; identical repeat (tail, rolled back 1); last 3 tokens edited (tail, back 3); edit 150 / 300 before the end (encoder state at
    // 2,048); agent follow-up (prompt end); a 50-token extension (tail, no rollback); a shorter prefix (encoder 2,048)
    assert_eq!(cached, vec![0, 5285, 5283, 2048, 2048, 5286, 5286, 2048], "{log:?}");
    assert_eq!((r[1].calls, r[2].calls, r[6].calls), (1, 1, 1), "the repeat, the 3-token edit and the short extension are one small call each");
    let (r2, log2) = serve(&c, Behaviour::NearEnd, &prompts);
    ok(&r2, &log2);
    let cached2: Vec<usize> = r2.iter().map(|x| x.cached.unwrap()).collect();
    // with the near-end full grid state (5,120): an edit 150 before the end resumes at 5,120; the rest as above
    assert_eq!(cached2, vec![0, 5285, 5283, 5120, 2048, 5286, 5286, 2048], "{log2:?}");
}

/// CE3: a second conversation that shares the system prompt and the tool catalogue resumes at the highest kept call end inside the shared
/// part (and below its own cone).
#[test]
fn dsv41_ce3_shared_system_prompt_across_conversations() {
    let Some(c) = cfg() else { return };
    for (sys_n, u1, u2, want) in [(3000usize, 3000usize, 2500usize, 2048usize), (9000, 4000, 3000, 8192), (40_000, 3000, 3000, 38_912), (1500, 2000, 1500, 0)] {
        let sys = words(sys_n, 11);
        let c1 = cat(&sys, &words(u1, 12));
        let c2 = cat(&sys, &words(u2, 13));
        let (r, log) = serve(&c, Behaviour::Ce3, &[c1, c2]);
        ok(&r, &log);
        assert_eq!(r[1].cached, Some(want), "system prompt {sys_n}");
    }
}

/// CE3: an agent conversation of several turns (each turn = the previous prompt + more) resumes every turn at the previous prompt's end,
/// and its cold first turn costs no extra call.
#[test]
fn dsv41_ce3_agent_turns_resume_at_the_previous_prompt_end() {
    let Some(c) = cfg() else { return };
    let mut p = words(8460, 21);
    let mut prompts = vec![p.clone()];
    for t in 0..4u32 { p = cat(&p, &words(3100 + 50 * t as usize, 30 + t)); prompts.push(p.clone()); }
    let (r, log) = serve(&c, Behaviour::Ce3, &prompts);
    ok(&r, &log);
    for i in 1..prompts.len() { assert_eq!(r[i].cached, Some(prompts[i - 1].len()), "turn {i}"); }
    assert_eq!(r[0].calls, 5);
}

/// The pool's tail rule (pure): which prompts a tail state resumes, and where.
#[test]
fn dsv41_ce3_tail_states_resume_only_what_their_taps_cover() {
    use super::prefix::PrefixPool;
    let t = words(6000, 5);
    let mut pool = PrefixPool::new(1 << 40, MIN_KEEP);
    assert!(pool.insert(&t[..5000], Snapshot { pos: 5000, bytes: 1, data: Box::new(()), min_suffix: 0, tail: true }));
    assert_eq!(pool.lookup(&t[..5000]), Some((1, 4999)), "identical: rolled back one row");
    assert_eq!(pool.lookup(&t[..5050]), Some((1, 5000)), "short extension");
    assert_eq!(pool.lookup(&t[..5200]), Some((1, 5000)), "long extension (the ordinary rule)");
    assert_eq!(pool.lookup(&cat(&t[..4996], &[1, 1, 1, 1])), Some((1, 4996)), "last 4 tokens changed");
    assert_eq!(pool.lookup(&cat(&t[..4990], &[1; 10])), None, "last 10 changed: beyond the rollback");
    assert_eq!(pool.lookup(&t[..4999]), None, "shorter than the state: its taps are of other positions");
    let mut p2 = PrefixPool::new(1 << 40, MIN_KEEP);
    assert!(p2.insert(&t[..5000], Snapshot { pos: 5000, bytes: 1, data: Box::new(()), min_suffix: 0, tail: false }));
    assert_eq!(p2.lookup(&t[..5000]), None, "no tail: the 134-row floor holds");
    // the TP side's own rule
    assert!(tail_resume_ok(5000, 4999, Some(5000)) && tail_resume_ok(5050, 5000, Some(5000)) && !tail_resume_ok(5000, 4999, None));
    assert!(tail_resume_ok(5200, 5000, None), "128 rows recomputed: no taps needed");
    assert!(!tail_resume_ok(4999, 4990, Some(5000)), "a shorter prompt's taps window starts below the state's");
}

/// The cold-prompt plan (pure): only call ends, below the cone, dense first, sparse after, the top one, plus the end.
#[test]
fn dsv41_ce3_boundary_plan_adds_no_call() {
    use super::prefix::boundary_plan;
    assert_eq!(boundary_plan(5286, 0, 2048, 128, 2541), vec![2048, 5286]);
    assert_eq!(boundary_plan(2600, 0, 2048, 128, 2541), vec![2600]);
    let p = boundary_plan(131_072, 0, 2048, 128, 2541);
    let enc: Vec<usize> = p[..p.len() - 1].to_vec();
    assert_eq!(enc, vec![2048, 4096, 6144, 8192, 10_240, 12_288, 14_336, 16_384, 32_768, 65_536, 126_976]);
    // a resumed prompt: call ends from the resume point
    assert_eq!(boundary_plan(12_000, 5286, 2048, 128, 2541), vec![7334, 9382, 12_000]);
    // a remainder under the tail rides the last call: never a plan point
    assert_eq!(boundary_plan(4196, 0, 2048, 128, 0), vec![2048, 4196]);
}

/// Leg PFX2: short prompts (`16 <= pos < 128`, odd and even) capture `tail == true` states and resume on identical
/// repeats (`at = pos - 1`), 1..=6-row tail edits (`at = pos - d`), short extensions (`< 128`), and cross-128 extensions.
#[test]
fn dsv41_pfx2_short_prompts_and_odd_even_rollbacks_resume_through_the_real_engine() {
    use crate::dsv41::config::test_config_json;
    let c = cfg().unwrap_or_else(|| Dsv41Config::from_value(&test_config_json(), None).unwrap());
    for &n in &[16usize, 31, 46, 47, 68, 79, 87] {
        let p = words(n, 400 + n as u32);
        let edit1 = cat(&p[..n - 1], &[999_001]);
        let edit5 = cat(&p[..n - 5], &[999_002; 5]);
        let edit6 = cat(&p[..n - 6], &[999_003; 6]);
        let ext_short = cat(&p, &words(15, 700 + n as u32));
        let ext_cross128 = cat(&p, &words(130 - n, 800 + n as u32)); // total = 130 (crosses TAP_ROWS = 128, t0 = 2 < n)
        let prompts = [p.clone(), p.clone(), edit1, edit5, edit6, ext_short, ext_cross128];
        let (r, log) = serve(&c, Behaviour::Ce3, &prompts);
        ok(&r, &log);
        let got: Vec<usize> = r.iter().map(|x| x.cached.unwrap()).collect();
        assert_eq!(got, vec![0, n - 1, n - 1, n - 5, n - 6, n, n], "n={n}: {log:?}");
    }
}

/// Leg PFX2 Definition of Done: 3 consecutive VeloBenchmark `1_FULL_TEST` runs (`[31, 87, 68, 79, 46, 47, 3354, 1174, 2324,
/// 4598, 9173, 18293, 36565, 73082, 146133, 219190]`) through the REAL lane loop (`engine::spawn`), `boundary_plan`,
/// `Schedule`, and `end_rounds`, with realistic `tp::mem::snap_bytes` + tail tap bytes inside a `656 MB` TP=2 Rank 0 pool:
/// Run 2 and Run 3 achieve 100% `tail_resume` hits on every single prompt (`31..87` and `1174..219190`), `0` `tail` evictions,
/// and GPU live snapshot memory never spikes past the `656 MB` budget (pre-evict + `flush_frees` before `fwd.snapshot`).
#[test]
fn dsv41_pfx2_velobenchmark_3_consecutive_runs_through_engine_in_656mb_pool() {
    use crate::dsv41::config::test_config_json;
    let c = cfg().unwrap_or_else(|| Dsv41Config::from_value(&test_config_json(), None).unwrap());
    let lens: [usize; 16] = [31, 87, 68, 79, 46, 47, 3354, 1174, 2324, 4598, 9173, 18293, 36565, 73082, 146133, 219190];
    let budget: u64 = 656_000_000;
    let prompts: Vec<Vec<u32>> = lens.iter().enumerate().map(|(i, &n)| words(n, 1000 + i as u32)).collect();
    let log: Log = Arc::default();
    let stub = CStub::new(c, Behaviour::Ce3, 4, log.clone()).with_realistic_bytes();
    let mem = stub.mem.clone();
    let mut ec = ecfg(Behaviour::Ce3);
    ec.lanes = 4;
    ec.prefix = Some((budget, MIN_KEEP));
    let tail_resumes_before = super::metrics::get(&super::metrics::PFX_TAIL_RESUMES);
    let h = engine::spawn(Box::new(stub), ec);

    for run in 1..=3 {
        let mut cached_run = Vec::with_capacity(lens.len());
        for (pi, p) in prompts.iter().enumerate() {
            let (j, rx) = job(p.clone());
            assert!(h.submit(j).is_ok(), "run {run} prompt #{pi}");
            let (cached, done, err) = drain(rx);
            assert!(done && err.is_none(), "run {run} prompt #{pi} (len {}): err={err:?}", p.len());
            cached_run.push(cached.unwrap());
        }
        if run == 1 {
            assert!(cached_run.iter().all(|&c| c == 0), "run 1 distinct prompts start cold: {cached_run:?}");
        } else {
            let want: Vec<usize> = lens.iter().map(|&n| n - 1).collect();
            assert_eq!(cached_run, want, "run {run}: 100% tail_resume hits on all 16 prompts");
        }
    }
    drop(h);
    let l = log.lock().unwrap().clone();
    assert!(!l.iter().any(|s| s.starts_with("refused") || s.starts_with("FATAL")), "{l:?}");
    let tail_resumes_delta = super::metrics::get(&super::metrics::PFX_TAIL_RESUMES) - tail_resumes_before;
    assert!(tail_resumes_delta >= 32, "expected >= 32 tail_resumes across Run 2 + Run 3, got {tail_resumes_delta}");
    assert_eq!(super::metrics::get(&super::metrics::PFX_TAIL_EVICTIONS), 0, "0 tail == true evictions");
    let gm = mem.lock().unwrap();
    assert!(gm.peak_bytes <= budget, "GPU live snapshot bytes peaked at {} > budget {budget} (pre-evict + flush_frees must precede alloc)", gm.peak_bytes);
    assert!(gm.flushes > 0 && gm.freed_snaps > 0, "flush_frees ran before snapshot allocations");
}
