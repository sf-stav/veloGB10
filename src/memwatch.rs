//! Host-memory watchdog. On a GB10 the GPU allocations live in the same 128 GB of unified memory as
//! the host: when a process exhausts it, the kernel's OOM path cannot reclaim the pinned device
//! pages and the whole box can hang hard (two reboots on 2026-08-28 while bringing up
//! Qwen3.8-Flash-Next). A process that EXITS frees everything instantly. So: sample
//! `/proc/meminfo` every 200 ms and, below a floor of available memory, print what happened and
//! `exit(3)` — losing the process is always better than losing the machine (and the SSH session).
//!
//! `--mem-watchdog-gb=<floor>` (default 5; `0` disables). The minimum seen is reported by
//! `min_available_gb()` for the load/serve summaries.

use std::sync::atomic::{AtomicU64, Ordering};

static MIN_AVAIL: AtomicU64 = AtomicU64::new(u64::MAX);
static STARTED: std::sync::Once = std::sync::Once::new();
static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

fn t0() -> f64 {
    T0.get_or_init(std::time::Instant::now).elapsed().as_secs_f64()
}

pub fn mem_available_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let l = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    l.split_whitespace().nth(1)?.parse::<u64>().ok().map(|kb| kb * 1024)
}

pub fn min_available_gb() -> f64 {
    let v = MIN_AVAIL.load(Ordering::Relaxed);
    if v == u64::MAX { f64::NAN } else { v as f64 / 1e9 }
}

/// B14/B13 serve-prep instrumentation: log one NAMED phase of the boot memory timeline —
/// host-available, this process's RSS, and the running minimum. On GB10 the CUDA allocations
/// live in the same 128 GB as the host, so MemAvailable is the ground truth; the per-phase
/// deltas name the allocation site that owns a sink (method: instrument, then bisect).
pub fn phase(tag: &str) {
    let avail = mem_available_bytes().map(|b| b as f64 / 1e9).unwrap_or(f64::NAN);
    let rss = std::fs::read_to_string("/proc/self/status").ok()
        .and_then(|s| s.lines()
            .find(|l| l.starts_with("VmRSS:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok()))
        .map(|kb| kb as f64 / 1e6)
        .unwrap_or(f64::NAN);
    eprintln!("[memwatch] phase {tag} @{:.1}s: avail {avail:.1} GB | rss {rss:.1} GB | min-so-far {:.1} GB",
              t0(), min_available_gb());
}

/// Env-gated 1 Hz sampler (`--mem-trace=1`): the timeline BETWEEN phases. An oscillating
/// MemAvailable inside one phase is an allocate/free retry loop, not a monotonic leak — the
/// boot-6 signature of the TP=2 serve-prep sink. Started explicitly where serving begins.
pub fn trace_loop() {
    if crate::opts::var(crate::opt!("mem-trace")).is_err() { return; }
    std::thread::Builder::new().name("memtrace".into()).spawn(|| {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
            if let Some(a) = mem_available_bytes() {
                MIN_AVAIL.fetch_min(a, Ordering::Relaxed);
                eprintln!("[memtrace] @{:.1}s avail {:.1} GB", t0(), a as f64 / 1e9);
            }
        }
    }).expect("memtrace thread");
}

/// Start the watchdog thread once per process (idempotent).
pub fn start() {
    STARTED.call_once(|| {
        let floor_gb: f64 = crate::opts::var(crate::opt!("mem-watchdog-gb")).ok()
            .and_then(|v| v.parse().ok()).unwrap_or(5.0);
        if floor_gb <= 0.0 { return; }
        let floor = (floor_gb * 1e9) as u64;
        std::thread::Builder::new().name("memwatch".into()).spawn(move || {
            let mut strikes = 0u32;
            loop {
                std::thread::sleep(std::time::Duration::from_millis(200));
                let Some(avail) = mem_available_bytes() else { continue };
                MIN_AVAIL.fetch_min(avail, Ordering::Relaxed);
                if avail < floor {
                    // Two consecutive samples below the floor (400 ms) — a single spike from a
                    // transient allocation the kernel is already reclaiming should not kill us.
                    strikes += 1;
                    if strikes >= 2 {
                        eprintln!("\n[memwatch] MemAvailable {:.1} GB < floor {:.1} GB — exiting NOW to keep the box alive \
                                   (--mem-watchdog-gb raises/lowers the floor, 0 disables).",
                                  avail as f64 / 1e9, floor_gb);
                        std::process::exit(3);
                    }
                } else {
                    strikes = 0;
                }
            }
        }).expect("memwatch thread");
    });
}
