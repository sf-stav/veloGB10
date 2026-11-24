//! The lane loop: 1-4 requests share one decode step, each in its own lane of the forward.
//!
//! One thread owns the `Forward` and the prefix pool. Requests arrive as `Job`s; a free lane admits the oldest waiting
//! job (a kept prefix state is restored first when `--prefix-cache on` finds one), the prompt is prefilled ONE CHUNK
//! per loop iteration (so lanes already decoding do not stall for a whole long prefill), and every iteration ends
//! with one decode step over every decoding lane. Tokens go back to the HTTP side as `Out` events.
//!
//! Sampling is the host sampler (`sampler.rs`): a draw is keyed by (seed, position), so a sampled reply does not
//! depend on which other requests share the step. With `--dsv41-sink-reject` a drawn token is held one step and
//! verified by the row it is fed at (`sink.rs`); otherwise it is committed at once.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

use super::forward::{Forward, PrefillItem, RowOut, StartItem, StepRow};
use super::metrics as m;
use super::prefix::{snapshot_plan, PrefixPool};
use super::sampler::{self, Sampling};
use super::sink::{verdict, SinkCfg, Verdict};
use crate::dsv41::chat::special::THINK_END_ID;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish { Stop, Length }

/// Leg P7.3: multi-lane short prefill + start batching mode (`--dsv41-batch-admit`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BatchAdmit { On, StartOnly, Off }

fn batch_admit_mode() -> BatchAdmit {
    match crate::opts::lookup("dsv41-batch-admit").and_then(crate::opts::get).as_deref() {
        Some("off") => BatchAdmit::Off,
        Some("start-only") => BatchAdmit::StartOnly,
        _ => BatchAdmit::On,
    }
}

/// What the engine tells the HTTP side about one request.
#[derive(Debug)]
pub enum Out {
    /// admitted to a lane; `cached` prompt tokens came from a kept prefix state
    Admitted { cached: usize },
    /// one committed token (a stop id is never sent)
    Tok(u32),
    Done { finish: Finish, prompt_tokens: usize, completion_tokens: usize },
    Failed(String),
    /// a score job's log-probs (`Forward::score`: `prompt.len() - 1` values), sent before its `Done`
    Scores(Vec<f32>),
}

pub struct Job {
    pub prompt: Vec<u32>,
    pub max_new: usize,
    pub min_tokens: usize,
    pub sampling: Sampling,
    pub seed: u64,
    pub ignore_eos: bool,
    /// the prompt ended with `<think>`: the reply starts inside a reasoning block
    pub starts_in_think: bool,
    /// `thinking_token_budget`: reasoning tokens allowed before the engine closes the block
    pub think_budget: Option<u64>,
    pub cancel: Arc<AtomicBool>,
    pub tx: tokio::sync::mpsc::UnboundedSender<Out>,
    /// a score job (leg CE): an exact prefill of `prompt` and its per-token log-probs, no generation, no prefix state
    pub score: bool,
    /// leg VIS1: per-prompt-token VL types (`empty` on text-only jobs), Rank-0 patch inputs, and prefix-cache span keys
    pub vl_types: Vec<i8>,
    pub vl_images: Vec<crate::dsv41::vision::preproc::PreparedImage>,
    pub image_spans: Vec<crate::dsv41::vision::preproc::ImageSpanKey>,
}

#[derive(Clone, Debug)]
pub struct EngineCfg {
    pub lanes: usize,
    pub stop_ids: Vec<u32>,
    pub sink: SinkCfg,
    /// `Some((pool bytes, replay keep))` = `--prefix-cache on`
    pub prefix: Option<(u64, usize)>,
    pub grid_snaps: usize,
    /// prompt tokens per prefill chunk
    pub prefill_chunk: usize,
    /// the tokens the engine feeds to close a reasoning block at its budget (`</think>`, or `--reasoning-config`'s text)
    pub force_think_end: Vec<u32>,
    /// leg CE3 (`--dsv41-prefix-near-end on`): with a forward whose mid-prompt states are encoder-only, also keep CE2's FULL grid
    /// state near the prompt end (the prefill ends a call there and computes a second decoder cone: ~0.9 s on every cold prompt)
    pub near_end: bool,
    /// leg P4 (`--dsv41-prefill-tail-merge`): a prompt's last call takes a remainder of up to this many rows along (0 = off)
    pub tail_merge: usize,
}

impl EngineCfg {
    pub fn simple(lanes: usize, stop_ids: Vec<u32>) -> Self {
        EngineCfg { lanes, stop_ids, sink: SinkCfg::OFF, prefix: None, grid_snaps: 4, prefill_chunk: 2048, force_think_end: vec![THINK_END_ID], near_end: false, tail_merge: 0 }
    }
}

/// RAII token tracking HTTP requests currently inside `prepare()` so concurrent bursts coalesce cleanly with zero solo latency.
pub struct PrepGuard(Option<Arc<AtomicUsize>>);
impl Drop for PrepGuard {
    fn drop(&mut self) {
        if let Some(p) = self.0.take() { p.fetch_sub(1, Ordering::Release); }
    }
}

pub struct EngineHandle {
    tx: mpsc::Sender<Job>,
    waiting: Arc<AtomicUsize>,
    preparing: Arc<AtomicUsize>,
    alive: Arc<AtomicBool>,
}

impl EngineHandle {
    pub fn prep_guard(&self) -> PrepGuard {
        self.preparing.fetch_add(1, Ordering::AcqRel);
        PrepGuard(Some(self.preparing.clone()))
    }
    pub fn submit(&self, job: Job) -> Result<(), Job> {
        self.waiting.fetch_add(1, Ordering::SeqCst);
        m::inc(&m::WAITING);
        match self.tx.send(job) {
            Ok(()) => Ok(()),
            Err(e) => { self.waiting.fetch_sub(1, Ordering::SeqCst); m::dec(&m::WAITING); Err(e.0) }
        }
    }
    /// Jobs submitted and not yet in a lane.
    pub fn waiting(&self) -> usize { self.waiting.load(Ordering::SeqCst) }
    pub fn alive(&self) -> bool { self.alive.load(Ordering::SeqCst) }
}

struct Prov { tok: u32, logits: Vec<f32>, rejected: Vec<u32> }

struct Lane {
    job: Job,
    /// positions the forward holds for this lane (= `fed.len()`)
    pos: usize,
    /// prompt + every token fed so far
    fed: Vec<u32>,
    /// prefill progress: next prompt index to prefill; == prompt.len() once decoding
    pf_next: usize,
    /// leg P7.3: prefill reached `prompt.len()`, awaiting `Forward::start_ex` / `Forward::start_lanes`
    need_start: bool,
    plan: Vec<usize>,
    full: Vec<usize>,
    mid_snaps: Vec<u64>,
    history: Vec<u32>,
    /// committed tokens
    gen: usize,
    /// the token to feed at the next step
    feed: u32,
    prov: Option<Prov>,
    think_left: Option<u64>,
    forced: VecDeque<u32>,
    cached: usize,
}

/// The receiving half of the engine's channel and its shared flags (what `run_loop` runs on).
pub struct LoopParts { rx: mpsc::Receiver<Job>, waiting: Arc<AtomicUsize>, preparing: Arc<AtomicUsize>, alive: Arc<AtomicBool> }

/// The handle the HTTP side submits to, and the parts a loop (on any thread) runs on.
pub fn channel() -> (EngineHandle, LoopParts) {
    let (tx, rx) = mpsc::channel::<Job>();
    let waiting = Arc::new(AtomicUsize::new(0));
    let preparing = Arc::new(AtomicUsize::new(0));
    let alive = Arc::new(AtomicBool::new(true));
    (EngineHandle { tx, waiting: waiting.clone(), preparing: preparing.clone(), alive: alive.clone() },
     LoopParts { rx, waiting, preparing, alive })
}

/// Run the lane loop on the CURRENT thread until every handle is dropped (the TP head: the forward holds device
/// resources that stay on the thread that built them). Marks the engine dead when it returns.
pub fn run_loop(fwd: &mut dyn Forward, cfg: EngineCfg, parts: LoopParts) {
    let LoopParts { rx, waiting, preparing, alive } = parts;
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(fwd, cfg, rx, waiting, preparing)));
    alive.store(false, Ordering::SeqCst);
    if r.is_err() { eprintln!("\n*** FATAL: the dsv41 lane loop panicked; the server answers 503 from now on ***\n"); }
}

pub fn spawn(mut fwd: Box<dyn Forward + Send>, cfg: EngineCfg) -> EngineHandle {
    let (h, parts) = channel();
    std::thread::Builder::new().name("dsv41-lanes".into()).spawn(move || run_loop(&mut *fwd, cfg, parts)).expect("spawn the dsv41 lane thread");
    h
}

fn suppress_eos_now(job: &Job, gen: usize) -> bool { job.ignore_eos || gen < job.min_tokens }
fn terminal(cfg: &EngineCfg, job: &Job, gen: usize, t: u32) -> bool { !suppress_eos_now(job, gen) && cfg.stop_ids.contains(&t) }

fn run(fwd: &mut dyn Forward, cfg: EngineCfg, rx: mpsc::Receiver<Job>, waiting: Arc<AtomicUsize>, preparing: Arc<AtomicUsize>) {
    let mut cfg = cfg;
    if fwd.rounds() {
        // a rounds forward (TP + DSpark) cannot hold a token for a sink verdict
        cfg.sink.enabled = false;
    }
    let n_lanes = cfg.lanes.min(fwd.lanes()).max(1);
    let mut lanes: Vec<Option<Lane>> = (0..n_lanes).map(|_| None).collect();
    let mut queue: VecDeque<Job> = VecDeque::new();
    let mut pool = cfg.prefix.map(|(b, k)| PrefixPool::new(b, k));
    let mut pfx_pub = PoolPublished::default();
    let pool_budget = cfg.prefix.map_or(0, |p| p.0);
    fwd.want_sink(cfg.sink.enabled);
    if let Some(p) = pool.as_mut() { govern(&mut *fwd, p, pool_budget); m::set(&m::PFX_BUDGET, p.budget()); }
    let max_seq = fwd.max_seq_len();
    let ba = batch_admit_mode();

    'outer: loop {
        // ---- intake: block when completely idle, drain otherwise ----
        let idle = lanes.iter().all(|l| l.is_none()) && queue.is_empty();
        if idle {
            // wake every 250 ms so a forward with peers (TP) can keep them awake
            match rx.recv_timeout(std::time::Duration::from_millis(250)) {
                Ok(j) => queue.push_back(j),
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    m::set(&m::ACTIVE, 0);
                    if let Some(p) = pool.as_mut() { govern(&mut *fwd, p, pool_budget); publish(p, &mut pfx_pub); }
                    publish_kv(&*fwd, &lanes);
                    if let Err(e) = fwd.idle() { eprintln!("dsv41: the forward's idle call failed: {e:#}"); }
                    if fwd.fatal() { break 'outer; }
                    continue 'outer;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break 'outer,
            }
        }
        while let Ok(j) = rx.try_recv() { queue.push_back(j); }
        // leg P7.3: coalesce concurrent HTTP requests currently tokenizing in `prepare()` when no lane is decoding yet
        if ba != BatchAdmit::Off {
            let any_dec = lanes.iter().any(|l| l.as_ref().is_some_and(|l| l.pf_next >= l.job.prompt.len() && !l.need_start));
            let free_slots = lanes.iter().filter(|l| l.is_none()).count();
            if !any_dec && free_slots > queue.len() && preparing.load(Ordering::Acquire) > 0 {
                let deadline = std::time::Instant::now() + std::time::Duration::from_millis(4);
                while queue.len() < free_slots && preparing.load(Ordering::Acquire) > 0 {
                    let now = std::time::Instant::now();
                    if now >= deadline { break; }
                    let wait = (deadline - now).min(std::time::Duration::from_micros(400));
                    match rx.recv_timeout(wait) {
                        Ok(j) => queue.push_back(j),
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
                while let Ok(j) = rx.try_recv() { queue.push_back(j); }
            }
        }
        // leg MEM1: the prefix pool follows the box's free memory (it gives states back before the memory watchdog's floor)
        if let Some(p) = pool.as_mut() { govern(&mut *fwd, p, pool_budget); }

        // ---- admission ----
        while let Some(li) = lanes.iter().position(|l| l.is_none()) {
            let Some(job) = queue.pop_front() else { break };
            waiting.fetch_sub(1, Ordering::SeqCst);
            m::dec(&m::WAITING);
            if job.cancel.load(Ordering::SeqCst) || job.tx.is_closed() { m::inc(&m::REQ_CANCELLED); continue; }
            if job.prompt.is_empty() || job.prompt.len() + 1 > max_seq {
                let _ = job.tx.send(Out::Failed(format!("the prompt has {} tokens and the lane holds {max_seq}", job.prompt.len())));
                continue;
            }
            if job.score {
                // score mode: one exact prefill in this lane, the log-probs back, the lane free again
                fwd.release(li);
                let _ = job.tx.send(Out::Admitted { cached: 0 });
                m::inc(&m::SCORE_REQS);
                match fwd.score(li, &job.prompt) {
                    Ok(lp) => {
                        m::add(&m::SCORE_TOKENS, job.prompt.len() as u64);
                        let _ = job.tx.send(Out::Scores(lp));
                        let _ = job.tx.send(Out::Done { finish: Finish::Length, prompt_tokens: job.prompt.len(), completion_tokens: 0 });
                    }
                    Err(e) => { let _ = job.tx.send(Out::Failed(format!("prompt scoring failed: {e:#}"))); }
                }
                fwd.release(li);
                continue;
            }
            match admit(&mut *fwd, pool.as_mut(), &cfg, li, job) {
                Ok(l) => lanes[li] = Some(l),
                Err((job, e)) => { let _ = job.tx.send(Out::Failed(format!("admission failed: {e:#}"))); fwd.release(li); }
            }
        }
        m::set(&m::ACTIVE, lanes.iter().flatten().count() as u64);

        // ---- cancellations ----
        for (li, slot) in lanes.iter_mut().enumerate() {
            if slot.as_ref().map_or(false, |l| l.job.cancel.load(Ordering::SeqCst) || l.job.tx.is_closed()) {
                *slot = None;
                fwd.release(li);
                m::inc(&m::REQ_CANCELLED);
            }
        }

        // ---- prefill (leg P7.3 batched short/tail multi-lane prefill when >= 2 qualify, else one prefill chunk) ----
        let batch_pf: Vec<usize> = if fwd.rounds() && ba == BatchAdmit::On {
            collect_batch_prefills(&*fwd, &cfg, &lanes)
        } else {
            Vec::new()
        };
        if batch_pf.len() >= 2 {
            if let Err(e) = prefill_batch_lanes(&mut *fwd, pool.as_mut(), &cfg, &mut lanes, &batch_pf) {
                let msg = format!("prefill failed: {e:#}");
                for &li in &batch_pf { fail_lane(&mut *fwd, &mut lanes, li, &msg); }
            }
        } else if fwd.rounds() && ba == BatchAdmit::StartOnly
            && !lanes.iter().any(|l| l.as_ref().is_some_and(|l| l.pf_next >= l.job.prompt.len() && !l.need_start))
        {
            // start-only A/B mode: let single-chunk admitted lanes finish solo prefills before one batched StartBatch
            let single_cap = cfg.prefill_chunk.max(1) + eff_tail(&*fwd, &cfg);
            let mut ran_any = false;
            for li in 0..lanes.len() {
                let can_one = lanes[li].as_ref().is_some_and(|l| {
                    let plen = l.job.prompt.len();
                    l.pf_next < plen && plen - l.pf_next <= single_cap
                        && l.plan.iter().all(|&p| p <= l.pf_next || p >= plen)
                });
                if can_one {
                    ran_any = true;
                    if let Err(e) = prefill_chunk(&mut *fwd, pool.as_mut(), &cfg, &mut lanes, li, true) {
                        fail_lane(&mut *fwd, &mut lanes, li, &format!("prefill failed: {e:#}"));
                    }
                }
            }
            if !ran_any {
                if let Some(li) = lanes.iter().position(|l| l.as_ref().map_or(false, |l| l.pf_next < l.job.prompt.len())) {
                    match prefill_chunk(&mut *fwd, pool.as_mut(), &cfg, &mut lanes, li, false) {
                        Ok(Some(f)) => { end(&mut *fwd, pool.as_mut(), li, &mut lanes[li], f); lanes[li] = None; fwd.release(li); }
                        Ok(None) => {}
                        Err(e) => fail_lane(&mut *fwd, &mut lanes, li, &format!("prefill failed: {e:#}")),
                    }
                }
            }
        } else if let Some(li) = lanes.iter().position(|l| l.as_ref().map_or(false, |l| l.pf_next < l.job.prompt.len())) {
            match prefill_chunk(&mut *fwd, pool.as_mut(), &cfg, &mut lanes, li, false) {
                Ok(Some(f)) => { end(&mut *fwd, pool.as_mut(), li, &mut lanes[li], f); lanes[li] = None; fwd.release(li); }
                Ok(None) => {
                    // the prompt goes on after this chunk: the lanes already decoding get their round first
                    let still = lanes[li].as_ref().map_or(false, |l| l.pf_next < l.job.prompt.len());
                    if still && lanes.iter().enumerate().any(|(i, l)| i != li && l.as_ref().map_or(false, |l| l.pf_next >= l.job.prompt.len() && !l.need_start)) { m::inc(&m::PREFILL_YIELDS); }
                }
                Err(e) => fail_lane(&mut *fwd, &mut lanes, li, &format!("prefill failed: {e:#}")),
            }
        }

        // ---- batched or solo Start for newly prefilled rounds lanes ----
        if fwd.rounds() {
            start_ready_lanes(&mut *fwd, pool.as_mut(), &cfg, &mut lanes);
        }

        // ---- one decode step over the decoding lanes ----
        if fwd.rounds() {
            let dec: Vec<(usize, usize)> = lanes.iter().enumerate().filter_map(|(i, l)| {
                let l = l.as_ref().filter(|l| l.pf_next >= l.job.prompt.len() && !l.need_start)?;
                let sup = if l.job.ignore_eos { 6 } else { l.job.min_tokens.saturating_sub(l.gen).min(6) };
                Some((i, sup))
            }).collect();
            if !dec.is_empty() {
                match fwd.round_ex(&dec) {
                    Ok(chains) => {
                        m::inc(&m::TP_ROUNDS);
                        for (&(li, _), chain) in dec.iter().zip(chains) {
                            if on_round(&mut *fwd, pool.as_mut(), &cfg, &mut lanes[li], li, chain, max_seq) { lanes[li] = None; fwd.release(li); }
                        }
                    }
                    Err(e) => { for &(li, _) in &dec { fail_lane(&mut *fwd, &mut lanes, li, &format!("forward round failed: {e:#}")); } }
                }
            }
            if fwd.fatal() {
                m::inc(&m::TP_FATAL);
                let why = "the TP forward failed and stopped (the ranks may disagree); restart the server";
                for slot in lanes.iter_mut() { if let Some(l) = slot.take() { let _ = l.job.tx.send(Out::Failed(why.into())); } }
                for j in queue.drain(..) { let _ = j.tx.send(Out::Failed(why.into())); }
                break 'outer;
            }
            if let Some(p) = &pool { publish(p, &mut pfx_pub); }
            publish_kv(&*fwd, &lanes);
            continue 'outer;
        }
        let rows: Vec<StepRow> = lanes.iter().enumerate()
            .filter_map(|(li, l)| l.as_ref().filter(|l| l.pf_next >= l.job.prompt.len())
                .map(|l| StepRow { lane: li, pos: l.pos, token: l.feed })).collect();
        if !rows.is_empty() {
            match fwd.step(&rows) {
                Ok(outs) => {
                    for (r, out) in rows.iter().zip(outs) {
                        let done = on_row(&mut *fwd, pool.as_mut(), &cfg, &mut lanes[r.lane], r, out, max_seq);
                        if done { lanes[r.lane] = None; fwd.release(r.lane); }
                    }
                }
                Err(e) => {
                    for r in &rows { fail_lane(&mut *fwd, &mut lanes, r.lane, &format!("forward step failed: {e:#}")); }
                }
            }
        }
        if let Some(p) = &pool { publish(p, &mut pfx_pub); }
        publish_kv(&*fwd, &lanes);
    }
    m::set(&m::ACTIVE, 0);
    fwd.shutdown();
}

/// Leg MEM1 / PFX2: the pool's budget from the forward's allowance (the box's free memory above the floor + reserve);
/// any evicted device snapshots are freed and trimmed immediately so `MemAvailable` recovers before the next poll.
fn govern(fwd: &mut dyn Forward, p: &mut PrefixPool, budget: u64) {
    if let Some(a) = fwd.pool_allowance(p.bytes(), budget) {
        let ev = p.set_budget(a);
        if ev > 0 {
            crate::dsv41::tp::mem::pressure_evicted(ev);
            if let Err(e) = fwd.flush_frees() { crate::reprintln!("dsv41: flush_frees after governor eviction failed: {e:#}"); }
        }
        m::set(&m::PFX_BUDGET, p.budget());
    }
}

/// Leg PFX2 / VIS1: pre-evict and flush queued device frees BEFORE `fwd.snapshot` allocates on the GPU, skipping non-tail states
/// that would displace a `tail == true` entry and skipping duplicate snapshots already covered by `(tokens, spans)`.
fn take_snapshot(
    fwd: &mut dyn Forward,
    pool: &mut PrefixPool,
    li: usize,
    tokens: &[u32],
    spans: &[crate::dsv41::vision::preproc::ImageSpanKey],
    is_tail: bool,
    is_full: bool,
) -> anyhow::Result<Option<u64>> {
    let pos = tokens.len();
    let min_pos = if is_tail { super::prefix::TAIL_MIN_POS } else { super::prefix::BLOCK };
    if pos < min_pos { return Ok(None); }
    let est = fwd.state_bytes_for(pos, is_tail).unwrap_or(0);
    let (ev0, len0) = (pool.evictions, pool.len());
    if !pool.prepare_insert_for_vl(tokens, spans, est, is_tail, is_full) {
        if pool.evictions > ev0 || pool.len() < len0 { fwd.flush_frees()?; }
        return Ok(None);
    }
    if pool.evictions > ev0 || pool.len() < len0 {
        fwd.flush_frees()?;
    }
    let snap = fwd.snapshot(li, pos)?;
    let id = pool.insert_id_vl(tokens, spans, snap);
    if id.is_some() { m::inc(&m::PFX_SNAPS); } else { fwd.flush_frees()?; }
    Ok(id)
}

/// The KV-occupancy gauges (`velogb10_kv_*`, as the other families publish them): positions each lane holds now. A free lane
/// reads 0; the prefix pool's states are not lane-resident, so `cached` stays 0.
fn publish_kv(fwd: &dyn Forward, lanes: &[Option<Lane>]) {
    crate::metrics::kv_publish(lanes.iter().enumerate().map(|(i, l)| (l.as_ref().map_or(0, |l| fwd.lane_pos(i).unwrap_or(l.pos)), 0)));
}

#[derive(Default)]
struct PoolPublished {
    hits: u64,
    misses: u64,
    reused: u64,
    evictions: u64,
    tail_evictions: u64,
    refused: u64,
}

fn sync_counter(c: & std::sync::atomic::AtomicU64, cur: u64, prev: &mut u64) {
    if cur >= *prev {
        let d = cur - *prev;
        if d > 0 { m::add(c, d); }
    } else {
        let sub = *prev - cur;
        let _ = c.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| Some(v.saturating_sub(sub)));
    }
    *prev = cur;
}

fn publish(p: &PrefixPool, prev: &mut PoolPublished) {
    sync_counter(&m::PFX_HITS, p.hits, &mut prev.hits);
    sync_counter(&m::PFX_MISSES, p.misses, &mut prev.misses);
    sync_counter(&m::PFX_REUSED, p.reused_tokens, &mut prev.reused);
    sync_counter(&m::PFX_EVICTIONS, p.evictions, &mut prev.evictions);
    sync_counter(&m::PFX_TAIL_EVICTIONS, p.tail_evictions, &mut prev.tail_evictions);
    sync_counter(&m::PFX_RESUME_REFUSED, p.refused, &mut prev.refused);
    m::set(&m::PFX_BYTES, p.bytes());
    m::set(&m::PFX_TAIL_BYTES, p.tail_bytes());
    m::set(&m::PFX_ENTRIES, p.len() as u64);
}

fn fail_lane(fwd: &mut dyn Forward, lanes: &mut [Option<Lane>], li: usize, why: &str) {
    if let Some(l) = lanes[li].take() {
        m::inc(&m::REQ_ENGINE_ERR);
        let _ = l.job.tx.send(Out::Failed(why.to_string()));
        fwd.release(li);
    }
}

/// Where a prompt's prefill snapshots, and which of those points the forward must make FULL (`prefill_snapshots`).
/// * a forward whose mid-prompt states are full (`encoder_cone() == None`, the stub): CE's 128-token grid near the end + the end;
/// * the TP CED forward (leg CE3): the call ends it makes anyway (`prefix::boundary_plan`: encoder-only states, free) + the end
///   (full); CE2's full grid state near the end only with `--dsv41-prefix-near-end on` (it costs an extra call on every cold prompt).
pub(crate) fn plan_for(fwd: &dyn Forward, cfg: &EngineCfg, plen: usize, cached: usize, keep: usize) -> (Vec<usize>, Vec<usize>) {
    match fwd.encoder_cone() {
        None => { let p = snapshot_plan(plen, cached, cfg.grid_snaps, keep); (p.clone(), p) }
        Some(cone) => {
            let mut plan = super::prefix::boundary_plan(plen, cached, cfg.prefill_chunk, eff_tail(fwd, cfg), cone);
            let mut full = Vec::new();
            if cfg.near_end {
                full = snapshot_plan(plen, cached, cfg.grid_snaps, keep).into_iter().filter(|&p| p < plen).collect();
                plan.extend(full.iter().copied());
                plan.sort_unstable();
                plan.dedup();
            }
            (plan, full)
        }
    }
}

fn admit(fwd: &mut dyn Forward, pool: Option<&mut PrefixPool>, cfg: &EngineCfg, li: usize, mut job: Job) -> Result<Lane, (Job, anyhow::Error)> {
    fwd.release(li);
    let mut cached = 0;
    let mut fed = Vec::new();
    let mut refused: Vec<u64> = Vec::new();
    if let Some(pool) = pool {
        // leg CE2: the forward checks every offered state BEFORE the lane is committed to it; a state it cannot resume this
        // prompt from exactly is skipped (one line, counted), the next usable one or a cold prefill serves the request
        while let Some((id, pos)) = pool.lookup_skip_vl(&job.prompt, &job.image_spans, &refused) {
            let snap = pool.snapshot(id).expect("entry exists");
            if let Err(e) = fwd.can_resume(snap, pos, job.prompt.len()) {
                let label = snap.min_suffix;
                pool.refuse(id, pos);
                refused.push(id);
                crate::reprintln!("dsv41: prefix state at {pos} (min suffix {}) cannot resume this {}-token prompt ({e:#}); trying the next kept state or a cold prefill (velogb10_dsv41_prefix_resume_refused_total)",
                                  label, job.prompt.len());
                continue;
            }
            let state_pos = snap.pos;
            match fwd.restore(li, snap) {
                Ok(()) => {}
                Err(e) => return Err((job, e)),
            }
            // leg CE3: a tail state resumes up to TAIL_BACK rows before its own end (the same prompt again, or its last tokens changed)
            if pos < state_pos {
                if let Err(e) = fwd.truncate(li, pos) { return Err((job, e)); }
            }
            if job.prompt.len() - pos < super::prefix::MIN_KEEP { m::inc(&m::PFX_TAIL_RESUMES); }
            cached = pos;
            fed = job.prompt[..pos].to_vec();
            break;
        }
    }
    // grid states are placed so a follow-up of the same length can resume from them: below the forward's
    // mid-prompt floor (CED: its decoder cone; the pool's own 134 otherwise); never place a snapshot strictly inside an image span
    let keep = cfg.prefix.map_or(0, |p| p.1).max(fwd.mid_prompt_keep());
    let (mut plan, mut full) = if cfg.prefix.is_some() { plan_for(&*fwd, cfg, job.prompt.len(), cached, keep) } else { (Vec::new(), Vec::new()) };
    if !job.image_spans.is_empty() {
        plan.retain(|&p| !super::prefix::inside_image_span(&job.image_spans, p));
        full.retain(|&p| !super::prefix::inside_image_span(&job.image_spans, p));
    }
    let _ = job.tx.send(Out::Admitted { cached });
    fwd.begin_prefill(li, job.prompt.len());
    if !job.vl_types.is_empty() {
        let imgs = std::mem::take(&mut job.vl_images);
        if let Err(e) = fwd.set_vl_inputs(li, cached, job.vl_types.clone(), imgs) {
            return Err((job, e));
        }
    }
    if !full.is_empty() { fwd.prefill_snapshots(li, cached, job.prompt.len(), &full); }
    let think_left = if job.starts_in_think { job.think_budget } else { None };
    Ok(Lane { pos: cached, fed, pf_next: cached, need_start: false, plan, full, mid_snaps: Vec::new(), history: Vec::new(), gen: 0, feed: 0, prov: None, think_left,
              forced: VecDeque::new(), cached, job })
}

/// Leg PP1: the end of the look-ahead after a prefill call that ends at `end` (exclusive): the next call's chunk, plus the tail
/// rows the next call may take with it (`prefill_tail`); `end` itself when the prompt is done.
/// Leg P4: the tail rule's row count: the forward's own (`prefill_tail`) or, when `--dsv41-prefill-tail-merge N` asks, N + 1
/// (a remainder of N rows or fewer rides the call before it), whichever is larger.
pub(crate) fn eff_tail(fwd: &dyn Forward, cfg: &EngineCfg) -> usize { fwd.prefill_tail().max(if cfg.tail_merge > 0 { cfg.tail_merge + 1 } else { 0 }) }

pub(crate) fn lookahead_end(end: usize, plen: usize, chunk: usize, tail: usize) -> usize { (end + chunk + tail).min(plen).max(end) }

/// Leg P7.3: collect text-only prefilling lanes that each finish their remaining prompt in a single call and together fit the
/// prefill arena (`prefill_chunk + tail_merge`).
fn collect_batch_prefills(fwd: &dyn Forward, cfg: &EngineCfg, lanes: &[Option<Lane>]) -> Vec<usize> {
    let single_cap = cfg.prefill_chunk.max(1) + eff_tail(fwd, cfg);
    let total_cap = cfg.prefill_chunk.max(1) + cfg.tail_merge;
    let mut out = Vec::new();
    let mut sum = 0usize;
    for (li, slot) in lanes.iter().enumerate() {
        let Some(l) = slot.as_ref() else { continue };
        let plen = l.job.prompt.len();
        if l.pf_next >= plen || !l.job.vl_types.is_empty() { continue; }
        let rem = plen - l.pf_next;
        if rem > single_cap || sum + rem > total_cap { continue; }
        if !l.plan.iter().all(|&p| p <= l.pf_next || p >= plen) { continue; }
        out.push(li);
        sum += rem;
    }
    out
}

fn prefill_batch_lanes(
    fwd: &mut dyn Forward,
    mut pool: Option<&mut PrefixPool>,
    cfg: &EngineCfg,
    lanes: &mut [Option<Lane>],
    batch: &[usize],
) -> anyhow::Result<()> {
    let outs = {
        let items: Vec<PrefillItem<'_>> = batch.iter().map(|&li| {
            let l = lanes[li].as_ref().expect("lane");
            PrefillItem { lane: li, from: l.pf_next, tokens: &l.job.prompt[l.pf_next..] }
        }).collect();
        fwd.prefill_batch(&items)?
    };
    for (&li, row) in batch.iter().zip(outs) {
        let l = lanes[li].as_mut().expect("lane");
        let plen = l.job.prompt.len();
        l.fed.extend_from_slice(&l.job.prompt[l.pf_next..plen]);
        l.pf_next = plen;
        l.pos = plen;
        l.need_start = true;
        if l.plan.contains(&plen) {
            let is_tail = fwd.has_tail_snaps();
            if let Some(pool) = pool.as_deref_mut() {
                if let Some(id) = take_snapshot(fwd, pool, li, &l.job.prompt[..plen], &l.job.image_spans, is_tail, true)? {
                    if !is_tail { l.mid_snaps.push(id); }
                }
                if is_tail && !l.mid_snaps.is_empty() && pool.bytes().saturating_mul(2) > pool.budget() {
                    let dropped = pool.prune_unhit_mid(&l.mid_snaps, 2);
                    l.mid_snaps.clear();
                    if dropped > 0 { fwd.flush_frees()?; }
                }
            }
        }
        if cfg.sink.enabled && cfg.sink.flagged(row.sink) { m::inc(&m::SINK_PROMPT_ROWS); }
    }
    Ok(())
}

fn finish_round_start(fwd: &mut dyn Forward, cfg: &EngineCfg, li: usize, l: &mut Lane, mut tok: u32) -> anyhow::Result<Option<Finish>> {
    if l.think_left == Some(0) {
        // `thinking_token_budget` 0: the reply's first token is the end of the reasoning block
        l.forced = cfg.force_think_end.iter().copied().collect();
        m::inc(&m::THINK_BUDGET_FORCED);
        let f = l.forced.pop_front().expect("force sequence is never empty");
        fwd.force(li, 1, f)?;
        tok = f;
    }
    let fin = consider(cfg, l, tok, Vec::new());
    if fin.is_none() && l.think_left == Some(0) && l.forced.is_empty() {
        // the first token spent the budget: the closing sequence goes in at the first round
        l.forced = cfg.force_think_end.iter().copied().collect();
        m::inc(&m::THINK_BUDGET_FORCED);
    }
    Ok(fin)
}

fn start_ready_lanes(fwd: &mut dyn Forward, mut pool: Option<&mut PrefixPool>, cfg: &EngineCfg, lanes: &mut [Option<Lane>]) {
    let ready: Vec<usize> = lanes.iter().enumerate().filter_map(|(i, l)| l.as_ref().filter(|l| l.need_start).map(|_| i)).collect();
    if ready.is_empty() { return; }
    for &li in &ready { if let Some(l) = lanes[li].as_mut() { l.need_start = false; } }
    let res = if ready.len() == 1 {
        let li = ready[0];
        let l = lanes[li].as_ref().expect("lane");
        fwd.start_ex(li, &l.job.sampling, l.job.seed, suppress_eos_now(&l.job, 0)).map(|t| vec![t])
    } else {
        let items: Vec<StartItem<'_>> = ready.iter().map(|&li| {
            let l = lanes[li].as_ref().expect("lane");
            StartItem { lane: li, sampling: &l.job.sampling, seed: l.job.seed, suppress_eos: suppress_eos_now(&l.job, 0) }
        }).collect();
        fwd.start_lanes(&items)
    };
    match res {
        Ok(toks) => {
            for (&li, tok) in ready.iter().zip(toks) {
                let step = {
                    let l = lanes[li].as_mut().expect("lane");
                    finish_round_start(fwd, cfg, li, l, tok)
                };
                match step {
                    Ok(Some(f)) => { end(fwd, pool.as_deref_mut(), li, &mut lanes[li], f); lanes[li] = None; fwd.release(li); }
                    Ok(None) => {}
                    Err(e) => fail_lane(fwd, lanes, li, &format!("start failed: {e:#}")),
                }
            }
        }
        Err(e) => {
            let msg = format!("start failed: {e:#}");
            for &li in &ready { fail_lane(fwd, lanes, li, &msg); }
        }
    }
}

fn prefill_chunk(fwd: &mut dyn Forward, pool: Option<&mut PrefixPool>, cfg: &EngineCfg, lanes: &mut [Option<Lane>], li: usize, defer_start: bool) -> anyhow::Result<Option<Finish>> {
    let l = lanes[li].as_mut().expect("lane");
    let plen = l.job.prompt.len();
    let mut end = (l.pf_next + cfg.prefill_chunk.max(1)).min(plen);
    // the last call carries the prompt's tail rows (the forward reads them right after it: the DSpark taps), so no other lane's round runs in between
    if end < plen && plen - end < eff_tail(fwd, cfg) { end = plen; }
    if let Some(&p) = l.plan.iter().find(|&&p| p > l.pf_next && p < end) { end = p; }
    // leg PP1: the next call's rows (a chunk, plus the tail rule's extension), so the forward can read ahead for them
    fwd.prefill_lookahead(li, &l.job.prompt[end..lookahead_end(end, plen, cfg.prefill_chunk.max(1), eff_tail(fwd, cfg))]);
    let row = fwd.prefill(li, l.pf_next, &l.job.prompt[l.pf_next..end])?;
    l.fed.extend_from_slice(&l.job.prompt[l.pf_next..end]);
    l.pf_next = end;
    l.pos = end;
    if l.plan.contains(&end) {
        let is_tail = end == plen && fwd.has_tail_snaps();
        let is_full = end == plen || fwd.encoder_cone().is_none() || l.full.contains(&end);
        if let Some(pool) = pool {
            if let Some(id) = take_snapshot(fwd, pool, li, &l.job.prompt[..end], &l.job.image_spans, is_tail, is_full)? {
                if !is_tail { l.mid_snaps.push(id); }
            }
            if is_tail && !l.mid_snaps.is_empty() && pool.bytes().saturating_mul(2) > pool.budget() {
                let dropped = pool.prune_unhit_mid(&l.mid_snaps, 2);
                l.mid_snaps.clear();
                if dropped > 0 { fwd.flush_frees()?; }
            }
        }
    }
    if end == plen {
        if cfg.sink.enabled && cfg.sink.flagged(row.sink) { m::inc(&m::SINK_PROMPT_ROWS); }
        // the first draw of the request
        if fwd.rounds() {
            let l = lanes[li].as_mut().expect("lane");
            if defer_start {
                l.need_start = true;
                return Ok(None);
            }
            let tok = fwd.start_ex(li, &l.job.sampling, l.job.seed, suppress_eos_now(&l.job, 0))?;
            return finish_round_start(fwd, cfg, li, l, tok);
        }
        let l = lanes[li].as_mut().expect("lane");
        return Ok(draw_next(cfg, l, row));
    }
    Ok(None)
}

/// Draw the next token from `row` for `l` and either commit it (returns the finish when it ends the request) or hold it.
fn draw_next(cfg: &EngineCfg, l: &mut Lane, row: RowOut) -> Option<Finish> {
    let position = l.pos; // the position the drawn token will occupy
    let tok = if let Some(t) = l.forced.pop_front() { t } else {
        let budget_hit = l.think_left == Some(0);
        if budget_hit {
            l.forced = cfg.force_think_end.iter().copied().collect();
            m::inc(&m::THINK_BUDGET_FORCED);
            l.forced.pop_front().expect("force sequence is never empty")
        } else {
            let excl: &[u32] = if suppress_eos_now(&l.job, l.gen) { &cfg.stop_ids } else { &[] };
            let drawn = sampler::sample(&row.logits, &l.job.sampling, &l.history, l.job.seed, position, excl, 0)
                .or_else(|| if excl.is_empty() { None } else { sampler::sample(&row.logits, &l.job.sampling, &l.history, l.job.seed, position, &[], 0) });
            match drawn {
                Some(t) => t,
                None => { let _ = l.job.tx.send(Out::Failed("the logits row had no finite entry".into())); return Some(Finish::Stop); }
            }
        }
    };
    consider(cfg, l, tok, row.logits)
}

/// Commit `tok` now, or hold it as provisional (sink handling on and the token does not end the request).
fn consider(cfg: &EngineCfg, l: &mut Lane, tok: u32, logits: Vec<f32>) -> Option<Finish> {
    let ends = terminal(cfg, &l.job, l.gen, tok);
    let last_by_budget = l.gen + 1 >= l.job.max_new;
    if cfg.sink.enabled && !ends && !last_by_budget {
        l.prov = Some(Prov { tok, logits, rejected: Vec::new() });
        l.feed = tok;
        return None;
    }
    commit(cfg, l, tok, ends)
}

/// Commit a token (emit it unless it is a stop id) and decide whether the request is over.
fn commit(_cfg: &EngineCfg, l: &mut Lane, tok: u32, ends: bool) -> Option<Finish> {
    if ends { return Some(Finish::Stop); }
    let _ = l.job.tx.send(Out::Tok(tok));
    l.history.push(tok);
    l.gen += 1;
    if let Some(left) = l.think_left.as_mut() {
        if tok == THINK_END_ID { l.think_left = None; } else { *left = left.saturating_sub(1); }
    }
    l.feed = tok;
    if l.gen >= l.job.max_new { return Some(Finish::Length); }
    None
}

/// Handle one decode row's output. Returns true when the lane is finished (the caller frees it).
fn on_row(fwd: &mut dyn Forward, mut pool: Option<&mut PrefixPool>, cfg: &EngineCfg, slot: &mut Option<Lane>, r: &StepRow, out: RowOut, max_seq: usize) -> bool {
    let li = r.lane;
    let Some(l) = slot.as_mut() else { return true };
    l.fed.push(r.token);
    l.pos += 1;
    // ---- the provisional token's verdict ----
    if let Some(prov) = l.prov.take() {
        let flagged = cfg.sink.flagged(out.sink);
        if flagged { m::inc(&m::SINK_FLAGGED); }
        match verdict(flagged, prov.rejected.len() as u32, cfg.sink.retries) {
            Verdict::Accept | Verdict::Fallthrough => {
                if matches!(verdict(flagged, prov.rejected.len() as u32, cfg.sink.retries), Verdict::Fallthrough) { m::inc(&m::SINK_FALLTHROUGHS); }
                if let Some(f) = commit(cfg, l, prov.tok, false) { return end(fwd, pool.take(), li, slot, f); }
            }
            Verdict::Reject => {
                // the next alternate of the draw that produced the token (rejected ids excluded, an independent draw)
                let mut rejected = prov.rejected;
                rejected.push(prov.tok);
                let mut excl = rejected.clone();
                if suppress_eos_now(&l.job, l.gen) {
                    for &s in &cfg.stop_ids { if !excl.contains(&s) { excl.push(s); } }
                }
                let alt = sampler::sample(&prov.logits, &l.job.sampling, &l.history, l.job.seed, l.pos - 1, &excl, rejected.len() as u64);
                match alt {
                    Some(t) => {
                        m::inc(&m::SINK_REJECTIONS);
                        // roll the fed token back
                        l.fed.pop();
                        l.pos -= 1;
                        if let Err(e) = fwd.truncate(li, l.pos) { let _ = l.job.tx.send(Out::Failed(format!("rollback failed: {e:#}"))); return true; }
                        if terminal(cfg, &l.job, l.gen, t) {
                            return end(fwd, pool.take(), li, slot, Finish::Stop);
                        }
                        l.feed = t;
                        l.prov = Some(Prov { tok: t, logits: prov.logits, rejected });
                        return false;
                    }
                    None => {
                        // nothing else can be drawn: commit the sunk token
                        m::inc(&m::SINK_FALLTHROUGHS);
                        if let Some(f) = commit(cfg, l, prov.tok, false) { return end(fwd, pool.take(), li, slot, f); }
                    }
                }
            }
        }
    }
    // ---- the next draw ----
    if l.pos >= max_seq {
        return end(fwd, pool.take(), li, slot, Finish::Length);
    }
    let finished = draw_next(cfg, slot.as_mut().expect("lane"), out);
    if let Some(f) = finished { return end(fwd, pool.take(), li, slot, f); }
    false
}

/// A lane's request ended: send Done, keep the generation-end state when the pool is on (and non-tail room allows).
fn end(fwd: &mut dyn Forward, pool: Option<&mut PrefixPool>, li: usize, slot: &mut Option<Lane>, f: Finish) -> bool {
    let Some(l) = slot.as_ref() else { return true };
    if let Some(pool) = pool {
        if l.pos > l.job.prompt.len() && l.pos >= super::prefix::BLOCK && l.prov.is_none() {
            let _ = take_snapshot(fwd, pool, li, &l.fed, &l.job.image_spans, false, true);
        }
    }
    let _ = l.job.tx.send(Out::Done { finish: f, prompt_tokens: l.job.prompt.len(), completion_tokens: l.gen });
    true
}

/// One lane's chain of a round (a rounds forward): commit the tokens in order up to a stop id or the token budget; the
/// rest of the chain is discarded (the lane's state is rolled back to the committed tokens when its snapshot is taken).
fn on_round(fwd: &mut dyn Forward, pool: Option<&mut PrefixPool>, cfg: &EngineCfg, slot: &mut Option<Lane>, li: usize, chain: Vec<u32>, max_seq: usize) -> bool {
    let Some(l) = slot.as_mut() else { return true };
    let n = chain.len();
    // `thinking_token_budget`: tokens of the closing sequence still to place (the budget ran out on the lane's pending token, or the
    // sequence has several tokens): this round's whole chain is dropped and the next token of the sequence takes its place
    if let Some(f) = l.forced.pop_front() {
        if let Err(e) = fwd.force(li, n, f) { let _ = l.job.tx.send(Out::Failed(format!("forcing the reasoning end failed: {e:#}"))); return true; }
        let ends = terminal(cfg, &l.job, l.gen, f);
        if let Some(fin) = commit(cfg, l, f, ends) { return end_rounds(fwd, pool, li, slot, fin); }
    } else {
        for (i, &t) in chain.iter().enumerate() {
            let ends = terminal(cfg, &l.job, l.gen, t);
            if let Some(f) = commit(cfg, l, t, ends) { return end_rounds(fwd, pool, li, slot, f); }
            if l.think_left == Some(0) {
                // the budget is spent: the reasoning block ends here. The rest of the chain is dropped and the closing sequence replaces it; when the
                // spent token was the pending one (nothing left to drop) the sequence goes in at the next round
                l.forced = cfg.force_think_end.iter().copied().collect();
                m::inc(&m::THINK_BUDGET_FORCED);
                let drop = n - 1 - i;
                if drop >= 1 {
                    let f = l.forced.pop_front().expect("force sequence is never empty");
                    if let Err(e) = fwd.force(li, drop, f) { let _ = l.job.tx.send(Out::Failed(format!("forcing the reasoning end failed: {e:#}"))); return true; }
                    let ends = terminal(cfg, &l.job, l.gen, f);
                    if let Some(fin) = commit(cfg, l, f, ends) { return end_rounds(fwd, pool, li, slot, fin); }
                }
                break;
            }
        }
    }
    if let Some(p) = fwd.lane_pos(li) {
        l.pos = p;
        // a round writes up to 6 rows past the pending token: stop before the lane's cache would overflow
        if p + 8 >= max_seq { return end_rounds(fwd, pool, li, slot, Finish::Length); }
    }
    false
}

/// A rounds lane's request ended: keep the generation-end state (prompt + the committed tokens but the last, which the
/// model has not been fed) when the pool is on and non-tail room allows, then send Done.
fn end_rounds(fwd: &mut dyn Forward, pool: Option<&mut PrefixPool>, li: usize, slot: &mut Option<Lane>, f: Finish) -> bool {
    let Some(l) = slot.as_ref() else { return true };
    if let Some(pool) = pool {
        let (plen, g) = (l.job.prompt.len(), l.history.len());
        if g >= 2 && plen + g - 1 >= super::prefix::BLOCK {
            let target = plen + g - 1;
            let est = fwd.state_bytes_for(target, false).unwrap_or(0);
            if pool.room_for_kind(est, false) {
                let pos = fwd.lane_pos(li).unwrap_or(0);
                let ready = if pos > target { fwd.truncate(li, target).is_ok() } else { pos == target };
                if ready {
                    let mut fed = l.job.prompt.clone();
                    fed.extend_from_slice(&l.history[..g - 1]);
                    let _ = take_snapshot(fwd, pool, li, &fed, &l.job.image_spans, false, true);
                }
            }
        }
    }
    let _ = l.job.tx.send(Out::Done { finish: f, prompt_tokens: l.job.prompt.len(), completion_tokens: l.gen });
    true
}
