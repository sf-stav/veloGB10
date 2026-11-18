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
//! `spin` = the previous behaviour (owner 2026-11-06: ship hybrid as an option first).
//!
//! NIGHT1 (2026-11-16, forum report by jetspark + the owner's prefill overheating): `hybrid` only covers the
//! NVFP4/bf16 trunk's `GpuModel::sync_stream` — the EXL3 and FP4-Local paths wait through ~400
//! `synchronize()` calls and ~600 blocking copies that never reach this module, so they kept spinning, and
//! every hybrid wait still spins its last 300 us. `block` sets the CUDA scheduling policy of the process's
//! primary context to CU_CTX_SCHED_BLOCKING_SYNC: EVERY wait in every path (stream / event / context
//! synchronize, blocking copies, the trunk's sites) sleeps in the driver until the GPU signals — no spin
//! at all. Cost: the wake-up of a blocked thread (measured by `--probe-gpu-wait`). Applied in `main` right
//! after the command line is parsed (before any context exists) and again on a TP node after it installs the
//! head's options; the resolved flags are read back from the driver and printed.
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
/// NIGHT1 hybrid v2: the tail (and any unpredicted wait) polls `cuStreamQuery` instead of spinning in the
/// driver: back-to-back queries for the first TAIL_SPIN, then a query after each TAIL_SLICE sleep. A blocking
/// (interrupt) wake-up costs 0.6-1.4 ms on GB10 (`--probe-gpu-wait`, 2026-11-16), a TAIL_SLICE sleep tens of us.
pub const TAIL_SPIN: Duration = Duration::from_micros(50);
pub const TAIL_SLICE: Duration = Duration::from_micros(20);
const WINDOW: usize = 8;

static MODE: AtomicU8 = AtomicU8::new(0); // 0 = unresolved, 1 = spin, 2 = hybrid, 3 = block
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
pub const MODES: &[&str] = &["spin", "hybrid", "block"];
/// The default when `--gpu-wait` is not given.
pub const DEFAULT_MODE: &str = "hybrid"; // NIGHT1 (owner 2026-11-16 overheating plan): hybrid v2 by default; --gpu-wait spin restores the old wait

fn mode() -> u8 {
    match MODE.load(Relaxed) {
        0 => {
            // --gpu-wait is a Ty::Choice (spin|hybrid|block): the parser already refused anything else.
            let v = crate::opts::var(crate::opt!("gpu-wait")).unwrap_or_else(|_| DEFAULT_MODE.to_string());
            let m = match v.trim() { "spin" => 1, "hybrid" => 2, _ => 3 };
            MODE.store(m, Relaxed);
            m
        }
        m => m,
    }
}

fn hybrid() -> bool { mode() == 2 }

/// The resolved policy name (for the boot line and /metrics).
pub fn mode_name() -> &'static str { match mode() { 1 => "spin", 2 => "hybrid", _ => "block" } }

static APPLIED: AtomicU8 = AtomicU8::new(0); // the scheduling flags last written to the primary context (0 = none yet)

/// Write the resolved `--gpu-wait` policy into the primary context's scheduling flags and print what the
/// driver reports back. `block` = CU_CTX_SCHED_BLOCKING_SYNC (every wait sleeps until the GPU signals);
/// `spin` / `hybrid` = CU_CTX_SCHED_SPIN (the thread polls; hybrid sleeps at the trunk's own sites first).
/// Call once the options are final (main after the command line; a TP node again after `install_head`):
/// re-resolves the option, idempotent per value. Never fails the boot: a driver that refuses keeps its
/// default and the line says so.
pub fn apply_context_policy() {
    MODE.store(0, Relaxed);
    let m = mode();
    let want: u32 = if m == 3 { sys::CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC as u32 } else { sys::CUctx_flags::CU_CTX_SCHED_SPIN as u32 };
    if APPLIED.load(Relaxed) as u32 == want { return; }
    let res = unsafe {
        let r = sys::cuInit(0);
        if r != CUresult::CUDA_SUCCESS { Err(format!("cuInit {r:?}")) } else {
            let mut dev: sys::CUdevice = 0;
            let r = sys::cuDeviceGet(&mut dev, 0);
            if r != CUresult::CUDA_SUCCESS { Err(format!("cuDeviceGet {r:?}")) } else {
                let r = sys::cuDevicePrimaryCtxSetFlags_v2(dev, want);
                let (mut flags, mut active) = (0u32, 0i32);
                let g = sys::cuDevicePrimaryCtxGetState(dev, &mut flags, &mut active);
                if r != CUresult::CUDA_SUCCESS { Err(format!("cuDevicePrimaryCtxSetFlags {r:?} (context active: {active})")) }
                else if g != CUresult::CUDA_SUCCESS { Err(format!("cuDevicePrimaryCtxGetState {g:?}")) }
                else { Ok((flags, active)) }
            }
        }
    };
    match res {
        Ok((flags, active)) if flags & 0x7 == want => {
            APPLIED.store(want as u8, Relaxed);
            crate::reprintln!("[gpu-wait] host GPU waits: {} (driver scheduling flags {:#x}{})", match m {
                1 => "spin (every waiting thread holds its core for the whole wait)",
                2 => "hybrid (the serving waits sleep until just before their predicted end, then poll the stream with 20 us sleeps; no driver spin)",
                _ => "block (every wait in every path sleeps in the driver until the GPU signals; no spin)",
            }, flags, if active != 0 { ", set on the active context" } else { "" });
        }
        Ok((flags, _)) => crate::reprintln!("[gpu-wait] --gpu-wait {}: the driver kept scheduling flags {flags:#x} (asked {want:#x}); waits keep the driver's default", mode_name()),
        Err(e) => crate::reprintln!("[gpu-wait] --gpu-wait {}: could not set the context's scheduling policy ({e}); waits keep the driver's default", mode_name()),
    }
}

/// The scheduling flags this process wrote to the primary context (0 = none; 1 = spin, 4 = blocking sync).
pub fn applied_flags() -> u8 { APPLIED.load(Relaxed) }
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
    wait_impl(loc, || unsafe { sys::cuStreamQuery(stream) }, || unsafe { sys::cuStreamSynchronize(stream) })
}

/// NIGHT1: wait until `event` has completed, under the same policy as [`stream_wait`] (the EXL3 speculative
/// round's pinned readbacks wait on events: `cuEventSynchronize` spun the decode thread at 100%). Same contract
/// and errors as `cuEventSynchronize`.
#[track_caller]
pub fn event_wait(event: sys::CUevent) -> Result<(), cudarc::driver::DriverError> {
    let loc = std::panic::Location::caller();
    wait_impl(loc, || unsafe { sys::cuEventQuery(event) }, || unsafe { sys::cuEventSynchronize(event) })
}

fn wait_impl(loc: &'static std::panic::Location<'static>, query: impl Fn() -> CUresult, sync: impl Fn() -> CUresult)
    -> Result<(), cudarc::driver::DriverError> {
    if !hybrid() || FORCE_SPIN.with(|c| c.get() > 0) {
        // spin: the driver spins; block: the context's BLOCKING_SYNC flag makes this sleep in the driver.
        if !trace() { return check(sync()); }
        let t0 = Instant::now();
        let r = sync();
        crate::reprintln!("[gpu-wait] {}:{} {} took_us={}", loc.file(), loc.line(), mode_name(), t0.elapsed().as_micros());
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
            match query() {
                CUresult::CUDA_SUCCESS => { early = true; break; }
                CUresult::CUDA_ERROR_NOT_READY => {}
                e => return check(e),
            }
        }
    }
    let woke = t0.elapsed();
    // NIGHT1 hybrid v2: finish by polling, not by the driver's spin (jetspark's report: the old hybrid still
    // spun the last 300 us of every wait). Same completion contract: the stream has drained (or its sticky
    // error is returned) when this returns.
    let r = if early { CUresult::CUDA_SUCCESS } else {
        if !SLACK_SET.with(|c| c.replace(true)) {
            unsafe { libc::prctl(libc::PR_SET_TIMERSLACK, 1000 as libc::c_ulong, 0, 0, 0) };
        }
        let tail0 = Instant::now();
        loop {
            match query() {
                CUresult::CUDA_ERROR_NOT_READY => {}
                done => break done,
            }
            if tail0.elapsed() < TAIL_SPIN { std::hint::spin_loop(); continue; }
            let a = Instant::now();
            std::thread::sleep(TAIL_SLICE);
            slept += a.elapsed();
        }
    };
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

/// NIGHT1: `--probe-gpu-wait [--probe-gpu-wait-mib 1024,16] [--probe-gpu-wait-reps 200]` (GPU, one box): the cost
/// of the resolved `--gpu-wait` policy on real waits. For each size: `reps` x (an async memset of that many MiB
/// on the null stream, then the stream wait this module serves), reporting per wait the GPU time (events), the
/// wall time, the wake-up overhead (wall - GPU) and the waiting thread's CPU time (CLOCK_THREAD_CPUTIME_ID:
/// a spin costs ~the whole wait, a blocked thread ~0). Run once per policy: the policy is fixed per process
/// at boot (`apply_context_policy`, before the context exists).
pub fn probe(args: &[String]) -> anyhow::Result<()> {
    use anyhow::Context;
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let sizes: Vec<usize> = arg("--probe-gpu-wait-mib").unwrap_or_else(|| "1024,16".into())
        .split(',').map(|x| x.trim().parse::<usize>()).collect::<Result<_, _>>().context("--probe-gpu-wait-mib")?;
    let reps: usize = arg("--probe-gpu-wait-reps").map(|x| x.parse()).transpose().context("--probe-gpu-wait-reps")?.unwrap_or(200);
    let dev = cudarc::driver::CudaDevice::new(0)?;
    let max = *sizes.iter().max().unwrap_or(&16) << 20;
    let buf = dev.alloc_zeros::<u8>(max)?;
    use cudarc::driver::DevicePtr;
    let ptr = *buf.device_ptr();
    let thread_cpu = || {
        let mut t = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut t) };
        t.tv_sec as f64 * 1e3 + t.tv_nsec as f64 / 1e6
    };
    let (mut e0, mut e1): (sys::CUevent, sys::CUevent) = (std::ptr::null_mut(), std::ptr::null_mut());
    unsafe {
        check(sys::cuEventCreate(&mut e0, 0))?;
        check(sys::cuEventCreate(&mut e1, 0))?;
    }
    let null: CUstream = std::ptr::null_mut();
    let med = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    crate::rprintln!("GPU_WAIT_PROBE policy={} driver_flags={:#x} reps={reps}", mode_name(), applied_flags());
    for &mib in &sizes {
        let n = mib << 20;
        let (mut gpu, mut wall, mut cpu) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..reps + 5 {
            unsafe {
                check(sys::cuEventRecord(e0, null))?;
                check(sys::cuMemsetD8Async(ptr, (i & 0xFF) as u8, n, null))?;
                check(sys::cuEventRecord(e1, null))?;
            }
            let (t0, c0) = (Instant::now(), thread_cpu());
            stream_wait(null)?;
            let (w, c) = (t0.elapsed().as_secs_f64() * 1e3, thread_cpu() - c0);
            let mut g = 0f32;
            unsafe { check(sys::cuEventElapsedTime(&mut g, e0, e1))?; }
            if i >= 5 { gpu.push(g as f64); wall.push(w); cpu.push(c); }
        }
        let (g, w, c) = (med(&mut gpu), med(&mut wall), med(&mut cpu));
        crate::rprintln!("GPU_WAIT_PROBE policy={} size_mib={mib} gpu_ms_med={g:.3} wall_ms_med={w:.3} wake_us_med={:.1} \
                          cpu_ms_per_wait_med={c:.3} cpu_share={:.2}", mode_name(), (w - g).max(0.0) * 1e3, c / w.max(1e-9));
    }
    unsafe { sys::cuEventDestroy_v2(e0); sys::cuEventDestroy_v2(e1); }
    Ok(())
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
