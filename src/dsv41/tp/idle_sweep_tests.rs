//! Leg PS2, defect 2 (P1M: a request arriving ~1.0-1.1 s after the previous one ended waited 0.45-1.37 s for its first
//! token instead of ~0.2 s). The sweep runs the REAL lane loop (`serve::engine`), the REAL idle tier (`idle::Idle`, head and
//! node over a TCP control connection: the park op, the wake byte, the node blocked in `wait_for_wake`), the REAL op
//! prologue / epilogue of `TpServe::exec` (`serve::op_begin` / `op_end`, i.e. NIGHT1's thermal governor) and the REAL
//! governor arithmetic with the box held hot (88 C against the 85 / 90 C limits). Only the GPU is simulated: an op of GPU
//! work sleeps a fixed time, and the collective that matches the head with the node is a channel (header) plus an ack.
//!
//! What it shows: with the governor billing an op by its own GPU time (PS2), a request's first op starts within a few
//! milliseconds of its arrival and its first token comes after the request's own work, at every arrival time across the
//! park transition; every park gets exactly one wake and no wake is lost. The negative control (the governor's old
//! accounting: everything between two paced ops is "GPU work") reproduces P1M's cliff on the same loops: the first token
//! waits about as long again as the idle gap.

use std::sync::atomic::{AtomicU64, Ordering::SeqCst};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Result};

use super::idle::{Idle, Link, Tick};
use super::serve::{op_begin, op_end, op_paces, Op};
use crate::dsv41::serve::engine::{self, EngineCfg, Job, Out};
use crate::dsv41::serve::forward::{Forward, RowOut, Snapshot, StepRow};
use crate::dsv41::serve::sampler::Sampling;

/// GPU time of one op of work (prefill, start, round) on every rank
const WORK: Duration = Duration::from_millis(12);
/// the box: 88.0 C against the 85 / 90 C defaults (deci-degrees): the governor's duty cycle is ~0.49 there
const HOT: (u32, u32, u32) = (880, 850, 900);
const PARK_AFTER: Duration = Duration::from_millis(300);
const EOS: u32 = 1;

#[derive(Default)]
struct Probe {
    /// when the client submitted the current request, and when its first op of real work began on the head
    submitted: Mutex<Option<Instant>>,
    first_op: Mutex<Option<Instant>>,
    head_parks: AtomicU64,
    node_wakes: AtomicU64,
}

struct ParkFwd {
    idle: Idle,
    hdr: mpsc::Sender<Op>,
    ack: mpsc::Receiver<()>,
    bill_end: bool,
    probe: Arc<Probe>,
    emitted: Vec<usize>,
    hot: bool,
}

impl ParkFwd {
    /// `TpForward::op` + `TpServe::exec` on the head, the GPU and the collective simulated.
    fn op(&mut self, op: Op) -> Result<()> {
        if !self.hot { crate::thermal::test_temp(Some(HOT)); self.hot = true; }
        if op != Op::Nop && op != Op::Park {
            self.idle.begin_op()?;
            let sub = *self.probe.submitted.lock().unwrap();
            let mut f = self.probe.first_op.lock().unwrap();
            if sub.is_some() && f.is_none() { *f = Some(Instant::now()); }
        }
        self.hdr.send(op)?;   // the header broadcast
        op_begin(&op);
        if op_paces(&op) { std::thread::sleep(WORK); }
        if self.ack.recv_timeout(Duration::from_secs(3)).is_err() { bail!("op {} never matched by the node (a lost wake-up?)", op.name()); }
        if self.bill_end { op_end(&op); }
        if op == Op::Park { self.idle.mark_parked(); self.probe.head_parks.fetch_add(1, SeqCst); }
        self.idle.op_done();
        Ok(())
    }
}

impl Forward for ParkFwd {
    fn vocab(&self) -> usize { 130_000 }
    fn lanes(&self) -> usize { 1 }
    fn max_seq_len(&self) -> usize { 1_000_000 }
    fn name(&self) -> &'static str { "park-sweep-stub" }
    fn rounds(&self) -> bool { true }
    fn prefill(&mut self, lane: usize, _from: usize, tokens: &[u32]) -> Result<RowOut> {
        self.op(Op::Prefill { lane, taps: true, total: tokens.len() as u32, first: true, full_at: 0, ahead: 0 })?;
        Ok(RowOut::default())
    }
    fn step(&mut self, _rows: &[StepRow]) -> Result<Vec<RowOut>> { bail!("rounds only") }
    fn snapshot(&mut self, _lane: usize, _pos: usize) -> Result<Snapshot> { bail!("no snapshots") }
    fn restore(&mut self, _lane: usize, _snap: &Snapshot) -> Result<()> { bail!("no snapshots") }
    fn truncate(&mut self, _lane: usize, _to: usize) -> Result<()> { bail!("no truncate") }
    fn release(&mut self, lane: usize) { self.emitted[lane] = 0; let _ = self.op(Op::Release { lane }); }
    fn start(&mut self, lane: usize, _s: &Sampling, seed: u64) -> Result<u32> {
        self.op(Op::Start { lane, temp: 1.0, seed, pen: [0.0, 0.0, 1.0], suppress_eos: false })?;
        self.emitted[lane] = 1;
        Ok(1000)
    }
    fn round(&mut self, lanes: &[usize]) -> Result<Vec<Vec<u32>>> {
        self.op(Op::Round { nl: 1, mask: 1, copy: true, sup_mask: 0, obs: None })?;
        Ok(lanes.iter().map(|&l| { self.emitted[l] += 2; vec![1001, 1002] }).collect())
    }
    fn idle(&mut self) -> Result<()> {
        match self.idle.tick() {
            Tick::Parked => Ok(()),
            Tick::Park => self.op(Op::Park),
            Tick::Heartbeat => self.op(Op::Nop),
        }
    }
    fn shutdown(&mut self) {
        if self.idle.wake_nodes().is_ok() { let _ = self.hdr.send(Op::Shutdown); }
    }
}

/// The node: `TpServe::follow` with the GPU and the collective simulated (the header arrives on a channel the node reads only
/// when it is not parked, as the real node reads its next header only after the wake byte).
fn node(mut idle: Idle, hdr: mpsc::Receiver<Op>, ack: mpsc::Sender<()>, bill_end: bool, probe: Arc<Probe>) {
    crate::thermal::test_temp(Some(HOT));
    while let Ok(op) = hdr.recv() {
        if op == Op::Shutdown { return; }
        op_begin(&op);
        if op_paces(&op) { std::thread::sleep(WORK); }
        if ack.send(()).is_err() { return; }
        if bill_end { op_end(&op); }
        if op == Op::Park {
            idle.mark_parked();
            if idle.wait_for_wake().is_err() { return; }
            probe.node_wakes.fetch_add(1, SeqCst);
        }
    }
}

fn pair() -> (std::net::TcpStream, std::net::TcpStream) {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let a = std::net::TcpStream::connect(l.local_addr().unwrap()).unwrap();
    let (b, _) = l.accept().unwrap();
    (a, b)
}

struct Rig { h: engine::EngineHandle, probe: Arc<Probe>, node: Option<std::thread::JoinHandle<()>>, arrive: Box<dyn Fn() + Send> }

fn rig(bill_end: bool) -> Rig {
    let (hs, ns) = pair();
    let (htx, hrx) = mpsc::channel::<Op>();
    let (atx, arx) = mpsc::channel::<()>();
    let probe = Arc::new(Probe::default());
    let p2 = probe.clone();
    let node_idle = Idle::new(Link::Node(ns), PARK_AFTER);
    let node = std::thread::spawn(move || node(node_idle, hrx, atx, bill_end, p2));
    let head = Idle::new(Link::Head(vec![hs]), PARK_AFTER);
    let arrive = Box::new(head.arrival_handle().expect("a head tier"));
    let f = ParkFwd { idle: head, hdr: htx, ack: arx, bill_end, probe: probe.clone(), emitted: vec![0], hot: false };
    Rig { h: engine::spawn(Box::new(f), EngineCfg::simple(1, vec![EOS])), probe, node: Some(node), arrive }
}

/// One request (16 prompt tokens, 6 new): (arrival -> first op of real work on the head, arrival -> first token). `http`:
/// the arrival wakes a parked server first, as the HTTP handlers do (`idle::prewake`).
fn request(r: &Rig, http: bool) -> (Duration, Duration) {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    *r.probe.first_op.lock().unwrap() = None;
    let t0 = Instant::now();
    *r.probe.submitted.lock().unwrap() = Some(t0);
    if http { (r.arrive)(); }
    let job = Job { prompt: (100..116).collect(), max_new: 6, min_tokens: 0, sampling: Sampling::default(), seed: 7, ignore_eos: true, starts_in_think: false,
                    think_budget: None, cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)), tx, score: false,
                    vl_types: Vec::new(), vl_images: Vec::new(), image_spans: Vec::new() };
    assert!(r.h.submit(job).is_ok(), "the engine is not running");
    let mut first: Option<Duration> = None;
    while let Some(o) = rx.blocking_recv() {
        match o {
            Out::Tok(_) if first.is_none() => first = Some(t0.elapsed()),
            Out::Done { .. } => break,
            Out::Failed(e) => panic!("the request failed: {e}"),
            _ => {}
        }
    }
    *r.probe.submitted.lock().unwrap() = None;
    let fo = r.probe.first_op.lock().unwrap().expect("no op ran for the request") - t0;
    (fo, first.expect("no token"))
}

/// Sweep the arrival time across the park transition (the server parks at the first idle tick past PARK_AFTER = 300 ms,
/// i.e. ~500 ms after the previous request ended): every request's first op starts within 10 ms and its first token comes
/// after its own work (prefill + start: ~2 x 12 ms) plus at most 60 ms, whether it lands before, during or after the park;
/// the node woke once per park. Every other request wakes the server on arrival as the HTTP handlers do.
#[test]
fn dsv41_ps2_arrival_sweep_across_the_park_transition_has_no_cliff() {
    let mut r = rig(true);
    request(&r, true);   // warm
    let mut worst = (Duration::ZERO, Duration::ZERO, 0u64);
    let mut rows = Vec::new();
    for i in 0..=25u64 {
        let wait = Duration::from_millis(300 + 20 * i);   // 0.30 .. 0.80 s after the previous request ended
        std::thread::sleep(wait);
        let parked_before = r.probe.head_parks.load(SeqCst);
        let (fo, ttft) = request(&r, i % 2 == 0);
        rows.push(format!("{} ms: first op {:.1} ms, first token {:.1} ms, parks {parked_before}", wait.as_millis(), fo.as_secs_f64() * 1e3, ttft.as_secs_f64() * 1e3));
        if ttft > worst.1 { worst = (fo, ttft, wait.as_millis() as u64); }
        assert!(fo < Duration::from_millis(10), "arrival {} ms after the previous request: the first op began after {:?}\n{}", wait.as_millis(), fo, rows.join("\n"));
        assert!(ttft < 2 * WORK + Duration::from_millis(60), "arrival {} ms: first token after {:?}\n{}", wait.as_millis(), ttft, rows.join("\n"));
    }
    let parks = r.probe.head_parks.load(SeqCst);
    assert!(parks >= 8, "the sweep must cross the park transition many times ({parks} parks)\n{}", rows.join("\n"));
    drop(r.h);
    r.node.take().unwrap().join().unwrap();
    assert_eq!(r.probe.node_wakes.load(SeqCst), parks, "one wake per park (none lost, none doubled)");
    let (g, _, _) = crate::thermal::idle_credit();
    assert!(g > 0, "the governor saw idle gaps and did not bill them");
    assert!(worst.1 < 2 * WORK + Duration::from_millis(60), "worst: first op {:?}, first token {:?} at {} ms", worst.0, worst.1, worst.2);
}

/// The negative control on the same loops: the governor's old accounting (an op is never closed, so everything between two
/// paced ops is billed as GPU work) makes a hot box sleep about as long again as the idle gap before the first prefill:
/// P1M's 0.45-1.37 s for ~1 s gaps. (A request after more than 2 s of quiet paid nothing: the old rule's WORK_GAP.)
#[test]
fn dsv41_ps2_the_old_pacing_reproduces_the_cliff_on_the_same_loops() {
    let mut r = rig(false);
    request(&r, true);
    for ms in [400u64, 600] {
        let wait = Duration::from_millis(ms);
        std::thread::sleep(wait);
        let (fo, ttft) = request(&r, ms == 400);
        assert!(fo < Duration::from_millis(10), "the first op itself is not late ({fo:?}): the wake is not the cause");
        assert!(ttft > wait.mul_f64(0.8), "{ms} ms gap: first token after {ttft:?}: the old accounting should sleep ~1x the gap at 88 C");
    }
    drop(r.h);
    r.node.take().unwrap().join().unwrap();
}
