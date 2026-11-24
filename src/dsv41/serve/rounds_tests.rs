//! Lane-loop tests with a scripted ROUNDS forward (the shape of the TP forward: `start` draws the first token, `round` returns a
//! chain per lane, `force` replaces the tail of the last chain, the prompt is prefilled over several calls). No model, no GPU:
//! they pin the engine's side of `thinking_token_budget` (leg SE2) and of the prefill yield.

use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};

use super::engine::{self, EngineCfg, Finish, Job, Out};
use super::forward::{Forward, PrefillItem, RowOut, Snapshot, StartItem, StepRow};
use super::sampler::Sampling;
use crate::dsv41::chat::special::THINK_END_ID;

const EOS: u32 = 1;
type Log = Arc<Mutex<Vec<String>>>;

struct RLane { prompt: usize, emitted: Vec<u32>, pos: usize }

/// "The model": reasoning tokens `1000 + n` until `</think>` is emitted, then five answer tokens `2000 + i`, then EOS
/// (or `3000 + i` on rows where EOS is masked by `min_tokens` / `ignore_eos`).
fn model_next_sup(emitted: &[u32], suppress_eos: bool) -> u32 {
    match emitted.iter().position(|&t| t == THINK_END_ID) {
        None => 1000 + emitted.len() as u32,
        Some(p) => {
            let i = emitted.len() - p - 1;
            if i < 5 { 2000 + i as u32 } else if suppress_eos { 3000 + i as u32 } else { EOS }
        }
    }
}

struct RStub { lanes: Vec<RLane>, chain: usize, tail: usize, log: Log, fail_force: bool, always_eos: bool }

impl RStub {
    fn new(lanes: usize, chain: usize, tail: usize, log: Log) -> Self {
        RStub { lanes: (0..lanes).map(|_| RLane { prompt: 0, emitted: Vec::new(), pos: 0 }).collect(), chain, tail, log, fail_force: false, always_eos: false }
    }
    fn say(&self, s: String) { self.log.lock().unwrap().push(s); }
    fn draw(&self, emitted: &[u32], suppress_eos: bool) -> u32 {
        if self.always_eos {
            if suppress_eos { 4000 + emitted.len() as u32 } else { EOS }
        } else {
            model_next_sup(emitted, suppress_eos)
        }
    }
}

impl Forward for RStub {
    fn vocab(&self) -> usize { 130_000 }
    fn lanes(&self) -> usize { self.lanes.len() }
    fn max_seq_len(&self) -> usize { 1_000_000 }
    fn name(&self) -> &'static str { "rounds-stub" }
    fn rounds(&self) -> bool { true }
    fn prefill_tail(&self) -> usize { self.tail }
    fn begin_prefill(&mut self, lane: usize, total: usize) { self.say(format!("begin {lane} {total}")); }
    fn prefill(&mut self, lane: usize, from: usize, tokens: &[u32]) -> Result<RowOut> {
        if self.lanes[lane].pos != from { bail!("lane {lane} at {} asked to resume at {from}", self.lanes[lane].pos); }
        self.lanes[lane].pos += tokens.len();
        self.lanes[lane].prompt = self.lanes[lane].pos;
        self.say(format!("prefill {lane} {from}+{}", tokens.len()));
        Ok(RowOut::default())
    }
    fn prefill_batch(&mut self, items: &[PrefillItem<'_>]) -> Result<Vec<RowOut>> {
        let ls: Vec<usize> = items.iter().map(|it| it.lane).collect();
        self.say(format!("prefill_batch {ls:?}"));
        for it in items {
            if self.lanes[it.lane].pos != it.from { bail!("lane {} at {} asked to resume at {}", it.lane, self.lanes[it.lane].pos, it.from); }
            self.lanes[it.lane].pos += it.tokens.len();
            self.lanes[it.lane].prompt = self.lanes[it.lane].pos;
        }
        Ok(vec![RowOut::default(); items.len()])
    }
    fn step(&mut self, _rows: &[StepRow]) -> Result<Vec<RowOut>> { bail!("rounds only") }
    fn snapshot(&mut self, _lane: usize, _pos: usize) -> Result<Snapshot> { bail!("no snapshots") }
    fn restore(&mut self, _lane: usize, _snap: &Snapshot) -> Result<()> { bail!("no snapshots") }
    fn truncate(&mut self, _lane: usize, _to: usize) -> Result<()> { bail!("no truncate") }
    fn release(&mut self, lane: usize) { self.lanes[lane] = RLane { prompt: 0, emitted: Vec::new(), pos: 0 }; }
    fn start(&mut self, lane: usize, s: &Sampling, seed: u64) -> Result<u32> {
        self.start_ex(lane, s, seed, false)
    }
    fn start_ex(&mut self, lane: usize, _s: &Sampling, _seed: u64, suppress_eos: bool) -> Result<u32> {
        let t = self.draw(&self.lanes[lane].emitted, suppress_eos);
        self.lanes[lane].emitted.push(t);
        self.say(format!("start {lane}"));
        Ok(t)
    }
    fn start_lanes(&mut self, items: &[StartItem<'_>]) -> Result<Vec<u32>> {
        let ls: Vec<usize> = items.iter().map(|it| it.lane).collect();
        self.say(format!("start_lanes {ls:?}"));
        let mut out = Vec::with_capacity(items.len());
        for it in items {
            let t = self.draw(&self.lanes[it.lane].emitted, it.suppress_eos);
            self.lanes[it.lane].emitted.push(t);
            out.push(t);
        }
        Ok(out)
    }
    fn round(&mut self, lanes: &[usize]) -> Result<Vec<Vec<u32>>> {
        let ls: Vec<(usize, usize)> = lanes.iter().map(|&l| (l, 0)).collect();
        self.round_ex(&ls)
    }
    fn round_ex(&mut self, lanes_sup: &[(usize, usize)]) -> Result<Vec<Vec<u32>>> {
        let ls: Vec<usize> = lanes_sup.iter().map(|&(l, _)| l).collect();
        self.say(format!("round {ls:?}"));
        let n = self.chain;
        Ok(lanes_sup.iter().map(|&(li, sup)| {
            let mut c = Vec::new();
            for j in 0..n {
                let t = self.draw(&self.lanes[li].emitted, j < sup);
                self.lanes[li].emitted.push(t);
                c.push(t);
            }
            c
        }).collect())
    }
    fn lane_pos(&self, lane: usize) -> Option<usize> { let l = &self.lanes[lane]; Some(l.pos + l.emitted.len().saturating_sub(1)) }
    fn force(&mut self, lane: usize, drop: usize, tok: u32) -> Result<()> {
        if self.fail_force { bail!("scripted failure"); }
        let l = &mut self.lanes[lane];
        if drop == 0 || drop > l.emitted.len() { bail!("force: drop {drop} of {}", l.emitted.len()); }
        let keep = l.emitted.len() - drop;
        l.emitted.truncate(keep);
        l.emitted.push(tok);
        self.say(format!("force {lane} drop {drop}"));
        Ok(())
    }
}

fn job(prompt: usize, max_new: usize, budget: Option<u64>) -> (Job, tokio::sync::mpsc::UnboundedReceiver<Out>) {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    (Job { prompt: (0..prompt as u32).collect(), max_new, min_tokens: 0, sampling: Sampling::default(), seed: 1, ignore_eos: false, starts_in_think: true,
           think_budget: budget, cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)), tx, score: false,
           vl_types: Vec::new(), vl_images: Vec::new(), image_spans: Vec::new() }, rx)
}

fn drain(mut rx: tokio::sync::mpsc::UnboundedReceiver<Out>) -> (Vec<u32>, Option<Finish>, Option<String>) {
    let (mut toks, mut fin, mut err) = (Vec::new(), None, None);
    while let Some(o) = rx.blocking_recv() {
        match o { Out::Tok(t) => toks.push(t), Out::Done { finish, .. } => { fin = Some(finish); break; } Out::Failed(e) => { err = Some(e); break; } _ => {} }
    }
    (toks, fin, err)
}

fn cfg(lanes: usize, chunk: usize) -> EngineCfg { let mut c = EngineCfg::simple(lanes, vec![EOS]); c.prefill_chunk = chunk; c }

#[test]
fn the_thinking_budget_closes_the_reasoning_block_exactly_at_the_budget() {
    for chain in 1..=6usize {
        for budget in 0..=14u64 {
            let log: Log = Arc::default();
            let h = engine::spawn(Box::new(RStub::new(2, chain, 0, log.clone())), cfg(2, 2048));
            let (j, rx) = job(10, 1000, Some(budget));
            assert!(h.submit(j).is_ok());
            let (toks, fin, err) = drain(rx);
            assert!(err.is_none(), "chain {chain} budget {budget}: {err:?}");
            assert_eq!(fin, Some(Finish::Stop), "chain {chain} budget {budget}");
            let b = budget as usize;
            assert_eq!(toks.len(), b + 1 + 5, "chain {chain} budget {budget}: {toks:?}\n{:?}", log.lock().unwrap());
            assert!(toks[..b].iter().all(|&t| (1000..2000).contains(&t) && t != THINK_END_ID), "chain {chain} budget {budget}: {toks:?}");
            assert_eq!(toks[b], THINK_END_ID, "chain {chain} budget {budget}: {toks:?}");
            assert_eq!(&toks[b + 1..], &[2000, 2001, 2002, 2003, 2004], "chain {chain} budget {budget}");
        }
    }
}

#[test]
fn a_reply_inside_its_budget_is_not_touched() {
    let log: Log = Arc::default();
    let h = engine::spawn(Box::new(RStub::new(1, 3, 0, log.clone())), cfg(1, 2048));
    let (j, rx) = job(10, 9, Some(1000));
    assert!(h.submit(j).is_ok());
    let (toks, fin, _) = drain(rx);
    assert_eq!(fin, Some(Finish::Length));
    assert_eq!(toks.len(), 9);
    assert!(!log.lock().unwrap().iter().any(|l| l.starts_with("force")), "no force without a spent budget");
}

#[test]
fn a_failed_force_fails_the_request_not_the_server() {
    let log: Log = Arc::default();
    let mut s = RStub::new(1, 3, 0, log);
    s.fail_force = true;
    let h = engine::spawn(Box::new(s), cfg(1, 2048));
    let (j, rx) = job(10, 100, Some(4));
    assert!(h.submit(j).is_ok());
    let (_, _, err) = drain(rx);
    assert!(err.unwrap_or_default().contains("forcing the reasoning end failed"));
    assert!(h.alive());
}

#[test]
fn a_long_prompt_is_prefilled_in_chunks_with_the_other_lanes_rounds_between_them() {
    let log: Log = Arc::default();
    let h = engine::spawn(Box::new(RStub::new(2, 2, 0, log.clone())), cfg(2, 100));
    // lane A: a short prompt and a long reply
    let (ja, rxa) = job(10, 1_000_000, None);
    assert!(h.submit(ja).is_ok());
    // wait until A decodes
    let t0 = std::time::Instant::now();
    while !log.lock().unwrap().iter().any(|l| l.starts_with("round")) { assert!(t0.elapsed().as_secs() < 10); std::thread::sleep(std::time::Duration::from_millis(2)); }
    let (jb, rxb) = job(450, 8, None);
    assert!(h.submit(jb).is_ok());
    let (tb, _, _) = drain(rxb);
    assert_eq!(tb.len(), 8);
    drop(rxa);
    let l = log.lock().unwrap().clone();
    let pre: Vec<usize> = l.iter().enumerate().filter(|(_, s)| s.starts_with("prefill 1")).map(|(i, _)| i).collect();
    assert_eq!(pre.len(), 5, "450 rows at 100 per call: {l:?}");
    for w in pre.windows(2) {
        assert!(l[w[0]..w[1]].iter().any(|s| s.starts_with("round")), "a decode round runs between two prefill calls of the long prompt: {l:?}");
    }
    assert!(super::metrics::get(&super::metrics::PREFILL_YIELDS) >= 4);
}

#[test]
fn the_last_prefill_call_carries_the_tail_rows() {
    let log: Log = Arc::default();
    let h = engine::spawn(Box::new(RStub::new(1, 2, 20, log.clone())), cfg(1, 100));
    let (j, rx) = job(210, 3, None);
    assert!(h.submit(j).is_ok());
    let _ = drain(rx);
    let calls: Vec<String> = log.lock().unwrap().iter().filter(|s| s.starts_with("prefill")).cloned().collect();
    assert_eq!(calls, vec!["prefill 0 0+100".to_string(), "prefill 0 100+110".to_string()], "the 10 rows left after 200 join the last call");
}

#[test]
fn dsv41_p7_min_tokens_suppresses_eos_inside_speculative_chains_greedy_and_sampled() {
    for chain in 1..=6usize {
        for temp in [0.0f32, 0.85] {
            for min_toks in [1usize, 4, 7, 13] {
                let log: Log = Arc::default();
                let mut stub = RStub::new(2, chain, 0, log.clone());
                stub.always_eos = true;
                let h = engine::spawn(Box::new(stub), cfg(2, 2048));
                let (mut j, rx) = job(8, 32, None);
                j.min_tokens = min_toks;
                j.sampling.temperature = temp;
                assert!(h.submit(j).is_ok());
                let (toks, fin, err) = drain(rx);
                assert!(err.is_none(), "chain={chain} temp={temp} min={min_toks}: {err:?}");
                assert_eq!(fin, Some(Finish::Stop), "chain={chain} temp={temp} min={min_toks}");
                assert_eq!(toks.len(), min_toks, "chain={chain} temp={temp} min={min_toks}: {toks:?}");
                assert!(toks.iter().all(|&t| t != EOS), "chain={chain} temp={temp} min={min_toks}: {toks:?}");
            }
        }
    }
}

#[test]
fn dsv41_p7_ignore_eos_generates_exact_max_tokens_across_speculative_chains() {
    for chain in [1usize, 3, 6] {
        let log: Log = Arc::default();
        let mut stub = RStub::new(2, chain, 0, log.clone());
        stub.always_eos = true;
        let h = engine::spawn(Box::new(stub), cfg(2, 2048));
        let (mut j, rx) = job(12, 23, None);
        j.ignore_eos = true;
        j.sampling.temperature = 0.7;
        assert!(h.submit(j).is_ok());
        let (toks, fin, err) = drain(rx);
        assert!(err.is_none(), "{err:?}");
        assert_eq!(fin, Some(Finish::Length));
        assert_eq!(toks.len(), 23);
        assert!(toks.iter().all(|&t| t != EOS));
    }
}

#[test]
fn dsv41_p7_concurrent_burst_runs_batched_prefill_and_batched_start() {
    let log: Log = Arc::default();
    let h = engine::spawn(Box::new(RStub::new(4, 3, 0, log.clone())), cfg(4, 2048));
    let mut rxs = Vec::new();
    for i in 0..4usize {
        let (mut j, rx) = job(64 + i * 8, 9, None);
        j.min_tokens = 9;
        rxs.push((j, rx));
    }
    let jobs: Vec<Job> = rxs.iter_mut().map(|(j, _)| {
        let (tx, dummy_rx) = tokio::sync::mpsc::unbounded_channel();
        drop(dummy_rx);
        std::mem::replace(j, Job {
            prompt: Vec::new(), max_new: 1, min_tokens: 0, sampling: Sampling::default(), seed: 0,
            ignore_eos: false, starts_in_think: false, think_budget: None,
            cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)), tx, score: false,
            vl_types: Vec::new(), vl_images: Vec::new(), image_spans: Vec::new(),
        })
    }).collect();
    // Submit all 4 requests while holding a PrepGuard until the 4th enters so they land in one admission batch.
    let g = h.prep_guard();
    for j in jobs { assert!(h.submit(j).is_ok()); }
    drop(g);
    for (_, rx) in rxs {
        let (toks, fin, err) = drain(rx);
        assert!(err.is_none(), "{err:?}");
        assert_eq!(fin, Some(Finish::Length));
        assert_eq!(toks.len(), 9);
    }
    let l = log.lock().unwrap().clone();
    assert!(l.iter().any(|s| s == "prefill_batch [0, 1, 2, 3]"), "expected batched prefill across all 4 lanes: {l:?}");
    assert!(l.iter().any(|s| s == "start_lanes [0, 1, 2, 3]"), "expected batched start across all 4 lanes: {l:?}");
}
