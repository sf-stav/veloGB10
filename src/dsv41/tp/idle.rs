//! DSV4.1 TP (leg HT1): the serve program's idle tier.
//!
//! Before HT1 an idle DSV4.1 TP server never stopped working (TP4B defect D1): the head sent a `Nop` op every 250 ms so a
//! node would never wait in a collective past the 10 s transport watchdog, which means every node sat INSIDE a collective
//! all the time: its main thread spinning in `cuStreamSynchronize` behind a `tp_wait_add` kernel that itself spun on the
//! GPU, and its RDMA proxy spinning too because the node's posted epoch was owed. The head's proxies never reached their
//! deep idle tier (a heartbeat every 250 ms, the deep tier starts after 1 s parked). Measured on TP4: 4-5 threads at
//! 100 % per rank, boards at 78-91 C with no request.
//!
//! The tier: after `--dsv41-idle-park-ms` with no request the head broadcasts `Op::Park` (an ordinary op: every rank runs
//! it, every collective is matched, nothing is owed). Then the nodes block on the head's TCP control connection (the
//! session's cluster stream, idle during serving), so no collective, no GPU kernel and no spinning thread is in flight on
//! any rank; the proxies fall into WFE and then real sleeps, the Engram readers park, the heartbeat stops. The next op the
//! head runs (a request, a prefix-state free, the shutdown) first writes one byte ([`WAKE`]) to every node, then
//! broadcasts its header as always: the nodes come out of `read`, wake their Engram readers and enter the header's
//! collective. Cost: one thread wake-up per node (~0.1-1 ms on GB10) on that op only. While requests run nothing changes.
//!
//! Liveness while parked: both ends set TCP keep-alive on the control stream (a box that lost power is noticed in ~11 s),
//! a node whose head is gone reads EOF / an error and ends its session (the supervisor re-arms), and the head peeks its
//! node streams on every idle tick: a node that closed its stream or wrote to it (a node only writes on failure) stops the
//! forward with a clear message instead of a request hanging into the transport watchdog.

use anyhow::{bail, Result};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The byte the head writes to wake a parked node.
pub const WAKE: u8 = 0xA7;

/// The control connection(s) the tier uses: the head holds one stream per node, a node its stream to the head.
pub enum Link {
    None,
    Head(Vec<TcpStream>),
    Node(TcpStream),
}

// ---- /metrics (rank 0 renders them; a node counts its own for its log line) ----
static PARK_MS: AtomicU64 = AtomicU64::new(0);
static ON: AtomicU64 = AtomicU64::new(0);
static PARKS: AtomicU64 = AtomicU64::new(0);
static WAKES: AtomicU64 = AtomicU64::new(0);
static PARKED_NS: AtomicU64 = AtomicU64::new(0);
/// process-relative ns when the current park began (0 = not parked): the parked-seconds row moves during a long park
static PARKED_AT: AtomicU64 = AtomicU64::new(0);
static FIRST_OP_US_LAST: AtomicU64 = AtomicU64::new(0);
static FIRST_OP_US_MAX: AtomicU64 = AtomicU64::new(0);
static WAKE_WRITE_US_MAX: AtomicU64 = AtomicU64::new(0);

fn epoch() -> Instant { static E: std::sync::OnceLock<Instant> = std::sync::OnceLock::new(); *E.get_or_init(Instant::now) }
fn now_ns() -> u64 { epoch().elapsed().as_nanos() as u64 + 1 }

/// The resolved `--dsv41-idle-park-ms` (0 = off).
pub fn park_after_from_registry() -> Duration {
    Duration::from_millis(crate::opts::lookup("dsv41-idle-park-ms").and_then(|o| crate::opts::ranged(o).ok()).unwrap_or(1000).max(0) as u64)
}

/// The head's side of the tier, shared with the HTTP threads ([`prewake`]): one lock orders every wake write against the
/// park, so the nodes get exactly one wake byte per park (a second byte would wake a node out of its NEXT park into a
/// collective the head does not join).
struct HeadState {
    streams: Vec<TcpStream>,
    parked: bool,
    since: Option<Instant>,
    /// the wake that the next op's end closes (the receipt of the wake cost)
    pending_first: Option<Instant>,
    /// a wake write failed: the forward must stop (the nodes are no longer all in the same place)
    broken: Option<String>,
}

/// The head tier of this process (the HTTP threads' handle; the last head created wins: one per server process).
static HEAD: Mutex<Option<Arc<Mutex<HeadState>>>> = Mutex::new(None);
static PREWAKES: AtomicU64 = AtomicU64::new(0);

enum Side {
    None,
    Head(Arc<Mutex<HeadState>>),
    Node { s: TcpStream, parked: bool },
}

fn lock(h: &Mutex<HeadState>) -> std::sync::MutexGuard<'_, HeadState> { h.lock().unwrap_or_else(|e| e.into_inner()) }

/// Wake the parked nodes under the head's lock (one byte each); a failed write marks the tier broken.
fn wake_locked(g: &mut HeadState) -> Result<()> {
    if let Some(b) = &g.broken { bail!("{b}"); }
    if !g.parked { return Ok(()); }
    let t0 = Instant::now();
    for i in 0..g.streams.len() {
        if let Err(e) = g.streams[i].write_all(&[WAKE]) {
            let why = format!("waking node rank {} from the idle tier: its control connection is gone ({e}; its session ended, see its log)", i + 1);
            g.broken = Some(why.clone());
            bail!("{why}");
        }
    }
    WAKE_WRITE_US_MAX.fetch_max(t0.elapsed().as_micros() as u64, Relaxed);
    g.parked = false;
    PARKED_NS.fetch_add(g.since.take().map_or(0, |t| t.elapsed().as_nanos() as u64), Relaxed);
    PARKED_AT.store(0, Relaxed);
    WAKES.fetch_add(1, Relaxed);
    g.pending_first = Some(t0);
    Ok(())
}

/// The HTTP side (a chat / completion request just arrived): wake a parked server NOW, so the nodes' wake-up overlaps the
/// request's parsing, chat rendering and tokenization instead of adding to its first op. No-op when awake or not a TP head.
/// A failed write is recorded; the lane loop's next op reports it and stops the forward.
pub fn prewake() {
    let h = match HEAD.lock() { Ok(g) => g.clone(), Err(e) => e.into_inner().clone() };
    if let Some(h) = h { prewake_on(&h); }
}

fn prewake_on(h: &Mutex<HeadState>) {
    let mut g = lock(h);
    if g.parked && wake_locked(&mut g).is_ok() { PREWAKES.fetch_add(1, Relaxed); }
}

/// What the head's idle tick does ([`Idle::tick`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tick {
    /// parked: no op (the nodes are out of every collective); check that the nodes are still there
    Parked,
    /// quiet for `--dsv41-idle-park-ms`: run `Op::Park`
    Park,
    /// awake and not yet due: run the `Nop` heartbeat
    Heartbeat,
}

/// One rank's idle tier.
pub struct Idle {
    side: Side,
    after: Duration,
    last_work: Instant,
}

impl Idle {
    /// The tier is on when `after` > 0 and the link is a control connection; `Link::None` (world 1, tests) is always off.
    pub fn new(link: Link, after: Duration) -> Idle {
        let on = !after.is_zero() && !matches!(link, Link::None);
        let side = match link {
            Link::Head(v) => {
                for s in &v { let _ = s.set_nodelay(true); keepalive(s); }
                HEAD_SEEN.store(true, Relaxed);
                ON.store(on as u64, Relaxed);
                PARK_MS.store(if on { after.as_millis() as u64 } else { 0 }, Relaxed);
                Side::Head(Arc::new(Mutex::new(HeadState { streams: v, parked: false, since: None, pending_first: None, broken: None })))
            }
            Link::Node(s) => { let _ = s.set_nodelay(true); keepalive(&s); Side::Node { s, parked: false } }
            Link::None => Side::None,
        };
        Idle { side, after: if on { after } else { Duration::ZERO }, last_work: Instant::now() }
    }
    pub fn off() -> Idle { Idle::new(Link::None, Duration::ZERO) }
    /// The head of a serving process: hand this tier to the HTTP side ([`prewake`]). Called once by `serve::run`; never by
    /// the unit tests (the HTTP tests of the serve shell would otherwise wake a test's parked tier).
    pub fn register_for_prewake(&self) {
        if let Side::Head(h) = &self.side { *HEAD.lock().unwrap_or_else(|e| e.into_inner()) = Some(h.clone()); }
    }
    /// Tests only (leg PS2's sweep): the HTTP side's arrival wake ([`prewake`]) aimed at THIS head tier, without registering it
    /// process-wide.
    #[cfg(test)]
    pub fn arrival_handle(&self) -> Option<impl Fn() + Send + 'static> {
        match &self.side { Side::Head(h) => { let h = h.clone(); Some(move || prewake_on(&h)) } _ => None }
    }
    pub fn on(&self) -> bool { !self.after.is_zero() }
    pub fn parked(&self) -> bool {
        match &self.side { Side::Head(h) => lock(h).parked, Side::Node { parked, .. } => *parked, Side::None => false }
    }

    /// The boot line (rank 0).
    pub fn boot_line(&self) -> String {
        if self.on() {
            format!("idle tier ON (--dsv41-idle-park-ms {}): after {} ms with no request every rank parks (the nodes block on the head's control connection: no collective, no GPU kernel, no spinning thread; proxies and Engram readers sleep); a new request wakes them as it arrives (one thread wake-up per node, ~0.1-1 ms, overlapping its tokenization). /metrics velogb10_dsv41_tp_idle_*",
                    self.after.as_millis(), self.after.as_millis())
        } else {
            "idle tier OFF (--dsv41-idle-park-ms 0): the head sends an idle heartbeat every 250 ms and every node keeps a core and a GPU kernel spinning while the server is idle".to_string()
        }
    }

    /// The head ran an op of real work (anything but the heartbeat / the park itself).
    pub fn worked(&mut self) { self.last_work = Instant::now(); }

    /// The head: is it time to park (on, awake, no work for `after`)?
    pub fn due(&self) -> bool { self.on() && !self.parked() && self.last_work.elapsed() >= self.after }

    /// The head, before an op of real work (anything but the heartbeat / the park): wake parked nodes (one byte each, nothing
    /// when a request's arrival already did), then count the op as work. An error names the node whose stream is gone.
    pub fn begin_op(&mut self) -> Result<()> {
        self.wake_nodes()?;
        self.worked();
        Ok(())
    }

    /// The head's idle tick (the lane loop found no request for 250 ms): what to do now.
    pub fn tick(&self) -> Tick {
        if self.parked() { Tick::Parked } else if self.due() { Tick::Park } else { Tick::Heartbeat }
    }

    /// Every rank, after running `Op::Park`: the tier is now parked (the head's next op must call [`Idle::wake_nodes`] first).
    pub fn mark_parked(&mut self) {
        match &mut self.side {
            Side::Head(h) => {
                let mut g = lock(h);
                g.parked = true;
                g.since = Some(Instant::now());
                PARKS.fetch_add(1, Relaxed);
                PARKED_AT.store(now_ns(), Relaxed);
            }
            Side::Node { parked, .. } => *parked = true,
            Side::None => {}
        }
    }

    /// The head: wake every parked node (one byte each; nothing when a request's arrival already did). An error names the node
    /// whose stream is gone.
    pub fn wake_nodes(&mut self) -> Result<()> {
        match &self.side { Side::Head(h) => wake_locked(&mut lock(h)), _ => Ok(()) }
    }

    /// The head: an op finished; the first one after a wake closes the wake's receipt (wake to the op's end).
    pub fn op_done(&mut self) {
        let Side::Head(h) = &self.side else { return };
        if let Some(t) = lock(h).pending_first.take() {
            let us = t.elapsed().as_micros() as u64;
            FIRST_OP_US_LAST.store(us, Relaxed);
            FIRST_OP_US_MAX.fetch_max(us, Relaxed);
        }
    }

    /// The head, on an idle tick while parked: a node that closed its control stream (EOF / reset) or wrote to it (a node
    /// writes there only to report a failure) is gone. None = every node looks alive.
    pub fn node_gone(&self) -> Option<String> {
        let Side::Head(h) = &self.side else { return None };
        let g = lock(h);
        if let Some(b) = &g.broken { return Some(b.clone()); }
        for (i, s) in g.streams.iter().enumerate() {
            let mut b = 0u8;
            let r = unsafe { libc::recv(s.as_raw_fd(), &mut b as *mut u8 as *mut libc::c_void, 1, libc::MSG_PEEK | libc::MSG_DONTWAIT) };
            if r == 0 { return Some(format!("node rank {} closed its control connection while the server was parked (its session ended; see its log)", i + 1)); }
            if r > 0 { return Some(format!("node rank {} reported a failure on its control connection while the server was parked (see its log)", i + 1)); }
            let e = std::io::Error::last_os_error();
            if !matches!(e.raw_os_error(), Some(libc::EAGAIN) | Some(libc::EINTR)) {
                return Some(format!("node rank {}'s control connection failed while the server was parked: {e}", i + 1));
            }
        }
        None
    }

    /// A node, after running `Op::Park`: block until the head's wake byte. Returns the time parked. Err when the head is
    /// gone (EOF, reset, keep-alive timeout) or wrote something that is not the wake byte (another build).
    pub fn wait_for_wake(&mut self) -> Result<Duration> {
        let t0 = Instant::now();
        match &mut self.side {
            Side::Node { s, parked } => { let r = wait_byte(s); *parked = false; r.map(|_| t0.elapsed()) }
            Side::Head(_) => Ok(Duration::ZERO),
            // never reached on a served node (`run_node` refuses to start without the stream): a node that cannot wait for the
            // wake byte must not fall back into a collective the parked head will not join for minutes (the transport watchdog)
            Side::None => bail!("parked without a control connection to wait on"),
        }
    }
}

/// Block on `s` until one byte arrives; it must be [`WAKE`]. Read timeouts set elsewhere on the socket only loop.
fn wait_byte<R: Read>(s: &mut R) -> Result<()> {
    let mut b = [0u8; 1];
    loop {
        match s.read(&mut b) {
            Ok(0) => bail!("the head closed the control connection while this node was parked (the head process ended)"),
            Ok(_) if b[0] == WAKE => return Ok(()),
            Ok(_) => bail!("the head sent byte {:#04x} instead of the idle tier's wake byte (head and node from different builds?)", b[0]),
            Err(e) if matches!(e.kind(), std::io::ErrorKind::Interrupted | std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => continue,
            Err(e) => bail!("the head's control connection failed while this node was parked: {e}"),
        }
    }
}

/// TCP keep-alive on a control stream: a peer box that lost power is noticed in ~11 s (5 s idle, then 3 probes 2 s apart).
fn keepalive(s: &TcpStream) {
    let fd = s.as_raw_fd();
    let set = |lvl: i32, opt: i32, v: i32| unsafe { libc::setsockopt(fd, lvl, opt, &v as *const i32 as *const libc::c_void, std::mem::size_of::<i32>() as libc::socklen_t) };
    set(libc::SOL_SOCKET, libc::SO_KEEPALIVE, 1);
    set(libc::IPPROTO_TCP, libc::TCP_KEEPIDLE, 5);
    set(libc::IPPROTO_TCP, libc::TCP_KEEPINTVL, 2);
    set(libc::IPPROTO_TCP, libc::TCP_KEEPCNT, 3);
}

/// The `velogb10_dsv41_tp_idle_*` rows (rank 0 / the head; nothing until a head tier exists in this process).
pub fn append(o: &mut String) {
    use std::fmt::Write as _;
    // a process without a TP serve head prints nothing; a head with the tier off still shows its 0 gauge
    if !HEAD_SEEN.load(Relaxed) { return; }
    let live = match PARKED_AT.load(Relaxed) { 0 => 0, t => now_ns().saturating_sub(t) };
    let rows: [(&str, &str, &str, String); 7] = [
        ("velogb10_dsv41_tp_idle_park_ms", "gauge", "Leg HT1 --dsv41-idle-park-ms: ms with no request before every rank parks (0 = the idle tier is off: the 250 ms heartbeat keeps every node spinning).", PARK_MS.load(Relaxed).to_string()),
        ("velogb10_dsv41_tp_idle_parked", "gauge", "1 while every rank is parked (no collective, GPU kernel or spinning thread in flight).", ((PARKED_AT.load(Relaxed) != 0) as u8).to_string()),
        ("velogb10_dsv41_tp_idle_parks_total", "counter", "Times the server parked every rank (about once per quiet gap longer than --dsv41-idle-park-ms; never during a request).", PARKS.load(Relaxed).to_string()),
        ("velogb10_dsv41_tp_idle_wakes_total", "counter", "Times the head woke the parked nodes (a request or another op arrived).", WAKES.load(Relaxed).to_string()),
        ("velogb10_dsv41_tp_idle_parked_seconds_total", "counter", "Seconds every rank spent parked instead of spinning (moves during a park).", format!("{:.3}", (PARKED_NS.load(Relaxed) + live) as f64 / 1e9)),
        ("velogb10_dsv41_tp_idle_first_op_seconds_max", "gauge", "The longest first op after a wake, from the wake bytes to the op's end (includes the request's tokenization when the HTTP side woke the nodes, and the op's own GPU work: compare with the same op while awake).", format!("{:.6}", FIRST_OP_US_MAX.load(Relaxed) as f64 / 1e6)),
        ("velogb10_dsv41_tp_idle_first_op_seconds_last", "gauge", "The last first op after a wake, from the wake bytes to its end.", format!("{:.6}", FIRST_OP_US_LAST.load(Relaxed) as f64 / 1e6)),
    ];
    for (n, k, h, v) in rows { let _ = writeln!(o, "# HELP {n} {h}\n# TYPE {n} {k}\n{n} {v}"); }
    let _ = writeln!(o, "# HELP velogb10_dsv41_tp_round_wait_hybrid Leg HT1 --dsv41-round-wait: 1 = a rank's main thread waits for the GPU inside a request with the hybrid wait (opt-in), 0 = it spins in the driver (default).\n# TYPE velogb10_dsv41_tp_round_wait_hybrid gauge\nvelogb10_dsv41_tp_round_wait_hybrid {}",
                     crate::dsv41::fwd::dev::round_wait_hybrid() as u8);
    let _ = writeln!(o, "# HELP velogb10_dsv41_tp_idle_prewakes_total Wakes done by a request's arrival on the HTTP side (the nodes wake while the request is tokenized), out of velogb10_dsv41_tp_idle_wakes_total.\n# TYPE velogb10_dsv41_tp_idle_prewakes_total counter\nvelogb10_dsv41_tp_idle_prewakes_total {}", PREWAKES.load(Relaxed));
    let _ = writeln!(o, "# HELP velogb10_dsv41_tp_idle_wake_write_seconds_max The longest time the head took to write the wake bytes to every node.\n# TYPE velogb10_dsv41_tp_idle_wake_write_seconds_max gauge\nvelogb10_dsv41_tp_idle_wake_write_seconds_max {:.6}",
                     WAKE_WRITE_US_MAX.load(Relaxed) as f64 / 1e6);
    // leg PS2: the thermal governor bills an op by its own GPU time; the idle gap before the next op is not work
    let (gaps, gap_s, credit_s) = crate::thermal::idle_credit();
    let _ = writeln!(o, "# HELP velogb10_dsv41_tp_thermal_idle_gaps_total Leg PS2: idle gaps before an op of GPU work that the thermal governor did not bill as work while it was pacing (this rank; before PS2 a gap under 2 s was billed and a hot box slept again before the next request's first op).\n# TYPE velogb10_dsv41_tp_thermal_idle_gaps_total counter\nvelogb10_dsv41_tp_thermal_idle_gaps_total {gaps}");
    let _ = writeln!(o, "# HELP velogb10_dsv41_tp_thermal_idle_gap_seconds_total Seconds of those idle gaps.\n# TYPE velogb10_dsv41_tp_thermal_idle_gap_seconds_total counter\nvelogb10_dsv41_tp_thermal_idle_gap_seconds_total {gap_s:.3}");
    let _ = writeln!(o, "# HELP velogb10_dsv41_tp_thermal_idle_credit_seconds_total Pacing seconds the governor did not sleep because the GPU had already been idle that long.\n# TYPE velogb10_dsv41_tp_thermal_idle_credit_seconds_total counter\nvelogb10_dsv41_tp_thermal_idle_credit_seconds_total {credit_s:.3}");
}

/// set once a head tier exists in this process (a head with the tier off still shows its gauge)
static HEAD_SEEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn pair() -> (TcpStream, TcpStream) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let a = TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let (b, _) = l.accept().unwrap();
        (a, b)
    }

    /// A parked node blocks (no CPU) until the head's wake byte, then returns; its time parked is reported.
    #[test]
    fn a_parked_node_wakes_on_the_head_byte() {
        let (h, n) = pair();
        let mut head = Idle::new(Link::Head(vec![h]), Duration::from_millis(1));
        let mut node = Idle::new(Link::Node(n), Duration::from_millis(1));
        head.mark_parked();
        node.mark_parked();
        let t = std::thread::spawn(move || { let r = node.wait_for_wake(); (r.map_err(|e| e.to_string()), node.parked()) });
        std::thread::sleep(Duration::from_millis(50));
        if t.is_finished() { panic!("the node must stay parked until the byte; it returned {:?}", t.join().unwrap().0); }
        let w0 = Instant::now();
        head.wake_nodes().unwrap();
        let (r, still) = t.join().unwrap();
        let parked = r.unwrap();
        assert!(parked >= Duration::from_millis(45) && !still && !head.parked());
        assert!(w0.elapsed() < Duration::from_millis(200), "the wake reached the node at once");
    }

    /// No lost wake-up: the byte written BEFORE the node starts waiting is still there when it waits (TCP buffers it),
    /// over many park / wake cycles with the order raced.
    #[test]
    fn no_lost_wake_up_over_raced_cycles() {
        let (h, n) = pair();
        let mut head = Idle::new(Link::Head(vec![h]), Duration::from_millis(1));
        let mut node = Idle::new(Link::Node(n), Duration::from_millis(1));
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let t = std::thread::spawn(move || {
            for i in 0..300u32 {
                node.mark_parked();
                if i % 3 == 0 { std::thread::sleep(Duration::from_micros(200)); }
                node.wait_for_wake().unwrap();
                tx.send(()).unwrap();
            }
        });
        for i in 0..300u32 {
            head.mark_parked();
            if i % 2 == 0 { std::thread::sleep(Duration::from_micros(150)); }
            head.wake_nodes().unwrap();
            rx.recv_timeout(Duration::from_secs(5)).expect("a wake was lost");
        }
        t.join().unwrap();
        assert!(!head.parked());
    }

    /// Shutdown while parked: the head process going away (its stream dropped) ends the node's wait with an error, at once.
    #[test]
    fn a_parked_node_ends_when_the_head_goes_away() {
        let (h, n) = pair();
        let mut node = Idle::new(Link::Node(n), Duration::from_millis(1));
        node.mark_parked();
        let t = std::thread::spawn(move || node.wait_for_wake().map_err(|e| e.to_string()));
        std::thread::sleep(Duration::from_millis(30));
        drop(h);
        let e = t.join().unwrap().unwrap_err();
        assert!(e.contains("closed the control connection"), "{e}");
    }

    /// The head notices a node that went away (or reported a failure) while parked, and never mistakes silence for death.
    #[test]
    fn the_head_sees_a_node_that_left_while_parked() {
        let (h1, n1) = pair();
        let (h2, mut n2) = pair();
        let head = Idle::new(Link::Head(vec![h1, h2]), Duration::from_millis(1));
        assert_eq!(head.node_gone(), None, "silent nodes are alive");
        n2.write_all(b"x").unwrap();
        std::thread::sleep(Duration::from_millis(20));
        assert!(head.node_gone().unwrap().contains("node rank 2 reported a failure"));
        drop(n1);
        std::thread::sleep(Duration::from_millis(20));
        assert!(head.node_gone().unwrap().contains("node rank 1 closed"));
    }

    /// A request's arrival (the HTTP side) wakes a parked server once; the lane loop's op that follows writes no second byte
    /// (one would wake a node out of its NEXT park into a collective the head does not join).
    #[test]
    fn a_request_arrival_wakes_once_and_the_op_sends_no_second_byte() {
        let (h, n) = pair();
        let mut head = Idle::new(Link::Head(vec![h]), Duration::from_millis(1));
        let mut node = Idle::new(Link::Node(n), Duration::from_millis(1));
        head.mark_parked();
        node.mark_parked();
        let hs = match &head.side { Side::Head(h) => h.clone(), _ => panic!("a head") };
        prewake_on(&hs);
        assert!(!head.parked(), "the arrival woke it");
        head.wake_nodes().unwrap();
        head.op_done();
        node.wait_for_wake().unwrap();
        std::thread::sleep(Duration::from_millis(20));
        let Side::Node { s, .. } = &node.side else { panic!("a node") };
        let mut b = 0u8;
        let r = unsafe { libc::recv(s.as_raw_fd(), &mut b as *mut u8 as *mut libc::c_void, 1, libc::MSG_PEEK | libc::MSG_DONTWAIT) };
        assert_eq!(r, -1, "exactly one wake byte was sent (found another: {b:#x})");
        prewake_on(&hs);
        node.mark_parked();
        let Side::Node { s, .. } = &node.side else { panic!("a node") };
        let r = unsafe { libc::recv(s.as_raw_fd(), &mut b as *mut u8 as *mut libc::c_void, 1, libc::MSG_PEEK | libc::MSG_DONTWAIT) };
        assert_eq!(r, -1, "an arrival on an awake server sends nothing");
    }

    /// The receipt: the head's /metrics rows and `scripts/owner_check.py` on the REAL rendered text of this process say the
    /// tier parked, and (TP4B D6) a DeepSeek-V4.1 server's report now carries the HEAT section (thermal pacing) too.
    #[test]
    fn owner_check_reports_the_idle_tier_and_the_heat_of_a_dsv41_server() {
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/owner_check.py");
        if !script.exists() || std::process::Command::new("python3").arg("--version").output().is_err() { return; }
        let (h, _n) = pair();
        let mut head = Idle::new(Link::Head(vec![h]), Duration::from_millis(1000));
        head.mark_parked();
        let mut rows = String::new();
        append(&mut rows);
        for name in ["velogb10_dsv41_tp_idle_park_ms 1000", "velogb10_dsv41_tp_idle_parked 1", "velogb10_dsv41_tp_idle_parks_total", "velogb10_dsv41_tp_idle_parked_seconds_total",
                     "velogb10_dsv41_tp_idle_first_op_seconds_max"] {
            assert!(rows.lines().any(|l| l.starts_with(name)), "{name} missing:\n{rows}");
        }
        for l in rows.lines().filter(|l| !l.starts_with('#')) { assert!(l.rsplit(' ').next().unwrap().parse::<f64>().is_ok(), "bad sample line {l}"); }
        use crate::dsv41::serve::metrics as m;
        m::set(&m::SERVING, 1);
        m::set(&m::LANES, 4);
        crate::dsv41::engram::io_metrics::ACTIVE.store(true, Relaxed);
        crate::dsv41::engram::io_metrics::READER_PARKS.fetch_add(2, Relaxed);
        let mut body = m::render_all(true);
        if !body.ends_with('\n') { body.push('\n'); }
        body.push_str(&rows);
        let dir = std::env::temp_dir().join(format!("ht1_oc_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("metrics.txt");
        std::fs::write(&f, &body).unwrap();
        let out = std::process::Command::new("python3").arg("-I").arg(&script).arg("--file").arg(&f).output().expect("run owner_check.py");
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(out.status.success(), "{text}\n{}", String::from_utf8_lossy(&out.stderr));
        for want in ["DeepSeek-V4.1 server", "HEAT (this box)", "thermal guard", "idle tier ON (--dsv41-idle-park-ms 1000): PARKED now", "Engram readers (this box): parked"] {
            assert!(text.contains(want), "owner_check output lacks {want:?}:\n{text}");
        }
        head.wake_nodes().unwrap();
    }

    /// A byte that is not the wake byte is refused (never read as a wake).
    #[test]
    fn a_foreign_byte_is_refused() {
        let mut c = std::io::Cursor::new(vec![0x00u8]);
        assert!(wait_byte(&mut c).unwrap_err().to_string().contains("different builds"));
        let mut c = std::io::Cursor::new(vec![WAKE]);
        wait_byte(&mut c).unwrap();
    }

    /// Off (0 ms, or no control connection) never parks; on parks only after the quiet time and never while parked.
    #[test]
    fn due_only_after_the_quiet_time() {
        assert!(!Idle::new(Link::None, Duration::from_millis(1)).due());
        let (h, _n) = pair();
        assert!(!Idle::new(Link::Head(vec![h.try_clone().unwrap()]), Duration::ZERO).due());
        let mut i = Idle::new(Link::Head(vec![h]), Duration::from_millis(30));
        assert!(!i.due());
        std::thread::sleep(Duration::from_millis(40));
        assert!(i.due());
        i.worked();
        assert!(!i.due());
        std::thread::sleep(Duration::from_millis(40));
        i.mark_parked();
        assert!(!i.due(), "already parked");
    }
}
