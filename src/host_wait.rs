//! CPUIDLE (2026-11-06): how the host waits for the GPU (`--gpu-wait spin|hybrid`).
//!
//! The engine never set a CUDA scheduling flag, so the driver's default applied: with one context on a
//! 20-core box that is a SPIN — the thread that waits in `cuStreamSynchronize` holds a big core at 100%
//! for the whole GPU step (measured 2026-11-06 on the 9B NVFP4 decode: ~97% of the scheduler thread's
//! samples were inside that spin, ~56% of them under `verify_forward_sample`). On GB10 the CPU and GPU share
//! one package and one power/thermal budget, so that core's power comes out of the GPU's headroom.
//!
//! `hybrid` keeps the spin for the part of a wait that decides latency and sleeps through the rest. Each
//! call site (`#[track_caller]`) remembers the shortest of its last 8 waits; the next wait at that site
//! sleeps in short slices (with a 1 µs timer slack, polling `cuStreamQuery` after each slice) until
//! [`MARGIN`] before that predicted end, then finishes in the driver's own spin. In steady state the GPU
//! completes inside the spin window, so the wake-up is exactly as fast as before; a wait shorter than the
//! window's minimum by more than the margin (a policy change, a narrower batch) is caught by the poll after
//! the slice it lands in — at most [`SLICE`] late, once, and the site's minimum adapts on that wait. Waits
//! predicted shorter than [`SLEEP_MIN`] never sleep. No value or order changes: only when the host looks.
//! `spin` = the previous behaviour and the DEFAULT (owner 2026-11-06: ship hybrid as an option first).
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};
use std::time::{Duration, Instant};

use cudarc::driver::sys::{self, CUresult, CUstream};

/// Spin for this long before the predicted end (must exceed SLICE + a sleep's overshoot).
pub const MARGIN: Duration = Duration::from_micros(300);
/// Longest single sleep; bounds the extra latency when a wait ends unexpectedly early.
pub const SLICE: Duration = Duration::from_micros(200);
/// A wait predicted shorter than this is spun (sleeping would save little and risk more).
pub const SLEEP_MIN: Duration = Duration::from_micros(700);
const WINDOW: usize = 8;

static MODE: AtomicU8 = AtomicU8::new(0); // 0 = unresolved, 1 = spin, 2 = hybrid
static HYBRID_WAITS: AtomicU64 = AtomicU64::new(0); // waits that slept at least once
static SLEPT_NS: AtomicU64 = AtomicU64::new(0); // time spent asleep instead of spinning
static EARLY: AtomicU64 = AtomicU64::new(0); // waits whose GPU finished before the spin window opened
static TRACE: AtomicU8 = AtomicU8::new(0); // 0 = unresolved, 1 = off, 2 = on (--gpu-wait-trace)

fn trace() -> bool {
    match TRACE.load(Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = crate::opts::var(crate::opt!("gpu-wait-trace")).is_ok();
            TRACE.store(if on { 2 } else { 1 }, Relaxed);
            on
        }
    }
}

/// The policy names `--gpu-wait` accepts.
pub const MODES: &[&str] = &["spin", "hybrid"];
/// The default when `--gpu-wait` is not given.
pub const DEFAULT_MODE: &str = "spin"; // opt-in (owner 2026-11-06): --gpu-wait hybrid

fn hybrid() -> bool {
    match MODE.load(Relaxed) {
        1 => false,
        2 => true,
        _ => {
            // --gpu-wait is a Ty::Choice (spin|hybrid): the parser already refused anything else.
            let v = crate::opts::var(crate::opt!("gpu-wait")).unwrap_or_else(|_| DEFAULT_MODE.to_string());
            let h = v.trim() != "spin";
            MODE.store(if h { 2 } else { 1 }, Relaxed);
            crate::reprintln!("[gpu-wait] host GPU waits: {}", if h {
                "hybrid (sleep to 300 us before each call site's predicted end, then spin)"
            } else {
                "spin (the thread holds its core for the whole wait)"
            });
            h
        }
    }
}

/// The resolved policy name (for the boot line and /metrics).
pub fn mode_name() -> &'static str { if hybrid() { "hybrid" } else { "spin" } }
/// Waits that slept at least once / seconds slept / waits that ended before the spin window (late wake-ups).
pub fn counters() -> (u64, f64, u64) {
    (HYBRID_WAITS.load(Relaxed), SLEPT_NS.load(Relaxed) as f64 / 1e9, EARLY.load(Relaxed))
}

#[derive(Default)]
struct Site { recent: [u32; WINDOW], n: usize, next: usize }

impl Site {
    fn predicted(&self) -> Option<Duration> {
        (self.n >= WINDOW).then(|| Duration::from_micros(*self.recent.iter().min().unwrap() as u64))
    }
    fn record(&mut self, d: Duration) {
        self.recent[self.next] = d.as_micros().min(u32::MAX as u128) as u32;
        self.next = (self.next + 1) % WINDOW;
        self.n = (self.n + 1).min(WINDOW);
    }
}

/// Inside this scope the calling thread's waits SPIN whatever --gpu-wait says. For timing measurements that
/// feed a policy (the MTP depth calibration: a hybrid wait shorter than its site's prediction can wake up to
/// one slice late, which would price the cheaper depths a little high). Serving keeps the hybrid wait.
pub struct SpinScope(());
impl SpinScope {
    pub fn new() -> Self { FORCE_SPIN.with(|c| c.set(c.get() + 1)); SpinScope(()) }
}
impl Drop for SpinScope {
    fn drop(&mut self) { FORCE_SPIN.with(|c| c.set(c.get().saturating_sub(1))); }
}

thread_local! {
    static FORCE_SPIN: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static SITES: RefCell<HashMap<usize, Site>> = RefCell::new(HashMap::new());
    static SLACK_SET: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn check(r: CUresult) -> Result<(), cudarc::driver::DriverError> {
    if r == CUresult::CUDA_SUCCESS { Ok(()) } else { Err(cudarc::driver::DriverError(r)) }
}

/// Wait until all work queued on `stream` has finished (the policy above). Same contract and errors as
/// `cuStreamSynchronize`.
#[track_caller]
pub fn stream_wait(stream: CUstream) -> Result<(), cudarc::driver::DriverError> {
    let loc = std::panic::Location::caller();
    if !hybrid() || FORCE_SPIN.with(|c| c.get() > 0) {
        if !trace() { return check(unsafe { sys::cuStreamSynchronize(stream) }); }
        let t0 = Instant::now();
        let r = unsafe { sys::cuStreamSynchronize(stream) };
        crate::reprintln!("[gpu-wait] {}:{} spin took_us={}", loc.file(), loc.line(), t0.elapsed().as_micros());
        return check(r);
    }
    let site = loc as *const _ as usize;
    let t0 = Instant::now();
    let pred = SITES.with(|s| s.borrow().get(&site).and_then(Site::predicted));
    let mut slept = Duration::ZERO;
    let mut early = false;
    if let Some(p) = pred.filter(|p| *p >= SLEEP_MIN) {
        let wake_at = p - MARGIN;
        if !SLACK_SET.with(|c| c.replace(true)) {
            // 1 µs timer slack for this thread (the default 50 µs would make every slice overshoot).
            unsafe { libc::prctl(libc::PR_SET_TIMERSLACK, 1000 as libc::c_ulong, 0, 0, 0) };
        }
        loop {
            let el = t0.elapsed();
            if el >= wake_at { break; }
            let s = (wake_at - el).min(SLICE);
            let a = Instant::now();
            std::thread::sleep(s);
            slept += a.elapsed();
            match unsafe { sys::cuStreamQuery(stream) } {
                CUresult::CUDA_SUCCESS => { early = true; break; }
                CUresult::CUDA_ERROR_NOT_READY => {}
                e => return check(e),
            }
        }
    }
    let woke = t0.elapsed();
    let r = if early { CUresult::CUDA_SUCCESS } else { unsafe { sys::cuStreamSynchronize(stream) } };
    let took = t0.elapsed();
    if trace() {
        crate::reprintln!("[gpu-wait] {}:{} hybrid pred_us={} took_us={} slept_us={} woke_us={} early={}",
                          loc.file(), loc.line(), pred.map_or(-1, |p| p.as_micros() as i64), took.as_micros(),
                          slept.as_micros(), woke.as_micros(), early as u8);
    }
    SITES.with(|s| s.borrow_mut().entry(site).or_default().record(took));
    if slept > Duration::ZERO {
        HYBRID_WAITS.fetch_add(1, Relaxed);
        SLEPT_NS.fetch_add(slept.as_nanos() as u64, Relaxed);
        if early { EARLY.fetch_add(1, Relaxed); }
    }
    check(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn site_predicts_only_after_a_full_window_and_takes_the_minimum() {
        let mut s = Site::default();
        for us in [900u64, 800, 1200, 1000, 950, 870, 1100] {
            s.record(Duration::from_micros(us));
            assert_eq!(s.predicted(), None);
        }
        s.record(Duration::from_micros(990));
        assert_eq!(s.predicted(), Some(Duration::from_micros(800)));
        // a shorter wait lowers the prediction at once; a longer one only after it ages out
        s.record(Duration::from_micros(500));
        assert_eq!(s.predicted(), Some(Duration::from_micros(500)));
        for _ in 0..WINDOW { s.record(Duration::from_micros(2000)); }
        assert_eq!(s.predicted(), Some(Duration::from_micros(2000)));
    }

    #[test]
    fn margin_covers_a_slice_and_short_waits_never_sleep() {
        assert!(MARGIN > SLICE + Duration::from_micros(50));
        assert!(SLEEP_MIN > MARGIN);
        assert!(MODES.contains(&DEFAULT_MODE));
    }
}
