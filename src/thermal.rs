//! NIGHT1 (owner 2026-11-16): the engine's thermal governor.
//!
//! GB10 boxes (the ASUS GX10s above all) cut their power — no shutdown sequence, no log line — when the
//! hottest ACPI board zone stays around 95-96 C. Sustained prompt reading (prefill) gets there in 2-3 minutes
//! on a hot box, while the GPU's own limiter never acts: at 96 C board temperature the GPU (84-88 C itself)
//! kept its full clock and reported no throttle reason. Neither vLLM nor SGLang regulate this; the community
//! workaround is a root clock cap (`nvidia-smi -lgc 300,2200`), which an engine cannot set as a normal user.
//!
//! So the engine paces itself. A sampler thread reads the hottest `/sys/class/thermal/thermal_zone*` (the board
//! zones run 8-15 C above what `nvidia-smi` reports for the GPU) every 250 ms. [`pace`] is called once per unit
//! of GPU work on every rank (each prefill pass, each scheduler step): below `--thermal-soft-c` it returns at
//! once; between the soft and the hard limit it sleeps a share of the time the GPU just worked (the GPU duty cycle
//! falls linearly to 15% at the hard limit), so the temperature settles inside the band; inside every prefill pass
//! [`pace_pass`] does the same every 4 layers (it first drains the queued GPU work, so the sleep idles the GPU); at `--thermal-hard-c` it
//! holds (sleeps in 100 ms slices until the box is 1 C under the limit, at most 2 s per call — well inside the
//! TP watchdog's 10 s — then lets one unit of work through). Under TP each rank reads its OWN box: the hottest
//! box sets the pace of the whole group (its peers wait for it at the next collective).
//!
//! Timing only: no value, order or protocol change. On by default (`--thermal-guard off` disables it, one boot
//! line says so). Counters on /metrics (`velogb10_thermal_*`) and an owner_check line.
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering::Relaxed};
use std::time::{Duration, Instant};

static STATE: AtomicU8 = AtomicU8::new(0); // 0 = unresolved, 1 = off, 2 = on (sampler running)
static HOT_DC: AtomicU32 = AtomicU32::new(0); // hottest zone, deci-degrees C (0 = no reading yet)
static EFF_DC: AtomicU32 = AtomicU32::new(0); // what the governor acts on: HOT_DC + the rise expected over LOOKAHEAD
static PEAK_DC: AtomicU32 = AtomicU32::new(0); // highest reading since boot
static SOFT_DC: AtomicU32 = AtomicU32::new(850);
static HARD_DC: AtomicU32 = AtomicU32::new(900);
static PACED_NS: AtomicU64 = AtomicU64::new(0); // time slept by pace() between the limits
static HELD_NS: AtomicU64 = AtomicU64::new(0); // time slept by pace() at or above the hard limit
static PACED_CALLS: AtomicU64 = AtomicU64::new(0);
static HOLDS: AtomicU64 = AtomicU64::new(0);
static IN_BAND: AtomicBool = AtomicBool::new(false); // for the transition lines
static ZONES: AtomicU32 = AtomicU32::new(0);

/// The longest single sleep of one `pace` call (the TP proxy watchdog fires after 10 s of unmatched debt).
pub const MAX_SLEEP: Duration = Duration::from_secs(2);
/// The GPU duty cycle at the hard limit (15%: sleep 5.7x the work); 100% at the soft limit, linear in between.
pub const MIN_DUTY: f64 = 0.15;
/// `pace_pass` checkpoints: every this many layers of a prefill pass (12 per 48-layer Flash-Next pass).
pub const PASS_EVERY_LAYERS: usize = 4;
/// The governor acts on the temperature expected this far ahead (the current rise rate over the last 2 s, rising
/// only, capped at MAX_LEAD): the board keeps climbing for seconds after the GPU slows down (2026-11-17 .11 TP=1
/// ladder: pacing from 88 C still overshot to 94 C).
pub const LOOKAHEAD_S: f64 = 2.0;
const MAX_LEAD_DC: u32 = 40;
/// A gap longer than this since the previous call was not GPU work (an idle server): no pacing debt.
const WORK_GAP: Duration = Duration::from_secs(2);

fn zone_paths() -> Vec<std::path::PathBuf> {
    let mut v = Vec::new();
    if let Ok(rd) = std::fs::read_dir("/sys/class/thermal") {
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            if n.starts_with("thermal_zone") { v.push(e.path().join("temp")); }
        }
    }
    v.sort();
    v
}

/// Hottest of the given zone files in deci-degrees C (None when nothing is readable).
fn read_hottest(paths: &[std::path::PathBuf]) -> Option<u32> {
    let mut best: Option<i64> = None;
    for p in paths {
        if let Ok(s) = std::fs::read_to_string(p) {
            if let Ok(mc) = s.trim().parse::<i64>() {
                // plausibility: a broken zone can read 0 or absurd values; keep 1..150 C
                if (1_000..150_000).contains(&mc) { best = Some(best.map_or(mc, |b| b.max(mc))); }
            }
        }
    }
    best.map(|mc| (mc / 100) as u32)
}

/// The rise expected over LOOKAHEAD_S from the readings 2 s apart (rising only, capped at MAX_LEAD_DC).
pub fn lead_dc(two_s_ago: u32, now: u32) -> u32 {
    if now <= two_s_ago { return 0; }
    ((((now - two_s_ago) as f64) / 2.0 * LOOKAHEAD_S).round() as u32).min(MAX_LEAD_DC)
}

/// The sleep `pace` owes after `work` of GPU time at temperature `t_dc`: 0 below `soft`; between the limits the
/// GPU duty cycle falls linearly from 100% to MIN_DUTY at `hard` (sleep = work x (1 - d) / d, capped at
/// MAX_SLEEP). The hold at or above `hard` is separate.
pub fn pacing_sleep(t_dc: u32, soft_dc: u32, hard_dc: u32, work: Duration) -> Duration {
    if t_dc < soft_dc || hard_dc <= soft_dc { return Duration::ZERO; }
    let x = (((t_dc - soft_dc) as f64) / ((hard_dc - soft_dc) as f64)).min(1.0);
    let d = 1.0 - (1.0 - MIN_DUTY) * x;
    work.mul_f64((1.0 - d) / d).min(MAX_SLEEP)
}

fn on() -> bool {
    match STATE.load(Relaxed) {
        1 => false,
        2 => true,
        _ => start(),
    }
}

/// Resolve `--thermal-guard` / `--thermal-soft-c` / `--thermal-hard-c` and start the sampler. Idempotent;
/// `pace` calls it on first use, the boot paths call it early so the boot line lands with the others.
pub fn start() -> bool {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    match STATE.load(Relaxed) { 1 => return false, 2 => return true, _ => {} }
    let opt = |k: crate::opts::OptId| crate::opts::var(k).ok();
    let guard = opt(crate::opt!("thermal-guard")).map(|v| v.trim().to_string()).unwrap_or_else(|| "on".into());
    let mut soft = opt(crate::opt!("thermal-soft-c")).and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(85.0);
    let mut hard = opt(crate::opt!("thermal-hard-c")).and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(90.0);
    if guard == "off" {
        STATE.store(1, Relaxed);
        crate::reprintln!("[thermal] guard OFF (--thermal-guard off): GPU work is never paced for temperature; \
                           a GB10 can power itself off under sustained prefill");
        return false;
    }
    if hard <= soft {
        crate::reprintln!("[thermal] --thermal-hard-c {hard} is not above --thermal-soft-c {soft}: using soft = hard - 4");
        soft = hard - 4.0;
    }
    if hard > 95.0 {
        crate::reprintln!("[thermal] --thermal-hard-c {hard} is above 95 C, where GX10 boxes have powered off: capped at 95");
        hard = 95.0;
        if soft >= hard { soft = hard - 4.0; }
    }
    let paths = zone_paths();
    let first = read_hottest(&paths);
    let Some(t0) = first else {
        STATE.store(1, Relaxed);
        crate::reprintln!("[thermal] guard OFF: no readable thermal zone under /sys/class/thermal (nothing to regulate on)");
        return false;
    };
    SOFT_DC.store((soft * 10.0).round() as u32, Relaxed);
    HARD_DC.store((hard * 10.0).round() as u32, Relaxed);
    HOT_DC.store(t0, Relaxed);
    EFF_DC.store(t0, Relaxed);
    PEAK_DC.store(t0, Relaxed);
    ZONES.store(paths.len() as u32, Relaxed);
    let spawned = std::thread::Builder::new().name("thermal".into()).spawn(move || {
        // the median of the last three readings (750 ms): a single-sample zone glitch (seen on GB10: 93 C -> 76 C in one
        // second with nothing running) never pauses work; real heating persists across samples and passes
        let mut last = [t0; 3];
        let mut hist = [t0; 9]; // filtered readings of the last 2 s (250 ms apart)
        let mut i = 0usize;
        loop {
            if let Some(t) = read_hottest(&paths) {
                last[i % 3] = t;
                let mut s = last; s.sort_unstable();
                let med = s[1];
                hist[i % 9] = med;
                i += 1;
                let old = hist[i % 9]; // the reading 2 s ago (the slot written next)
                HOT_DC.store(med, Relaxed);
                PEAK_DC.fetch_max(med, Relaxed);
                EFF_DC.store(med + lead_dc(old, med), Relaxed);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    });
    if spawned.is_err() {
        STATE.store(1, Relaxed);
        crate::reprintln!("[thermal] guard OFF: could not start the sampler thread");
        return false;
    }
    STATE.store(2, Relaxed);
    crate::reprintln!("[thermal] guard on: GPU work is paced from {soft:.0} C and held at {hard:.0} C (hottest of {} \
                       board zones, now {:.1} C; --thermal-soft-c / --thermal-hard-c, --thermal-guard off disables)",
                      ZONES.load(Relaxed), t0 as f64 / 10.0);
    true
}

thread_local! {
    static LAST: std::cell::Cell<Option<Instant>> = const { std::cell::Cell::new(None) };
}

/// Call once per unit of GPU work (a prefill pass, a scheduler step), on every rank, BEFORE the work is
/// launched. Sleeps when the box is hot (see the module doc); returns at once otherwise.
pub fn pace() {
    if !on() { return; }
    pace_core();
}

/// Inside a long GPU pass (every PASS_EVERY_LAYERS layers of a prefill chunk), on every rank. Cool box: returns
/// at once (one atomic load, no sync). Hot box (at or above the soft limit): `sync` drains the queued GPU work
/// first — the host enqueues a whole chunk ahead of the GPU, so a sleep without it would not idle the GPU — then
/// the same pacing / hold as [`pace`], at ~100 ms granularity instead of once per 1-2 s chunk (the per-chunk
/// pace let a 2,048-row chunk at ~70 W push the hottest zone several C past the hard limit: 2026-11-17 .11 TP=1
/// ladder, guard kill at 94.8 C). `capturing`: true while the stream is being captured into a CUDA graph (no
/// sync or sleep then).
pub fn pace_pass(capturing: bool, sync: impl FnOnce() -> anyhow::Result<()>) -> anyhow::Result<()> {
    if !on() || capturing { return Ok(()); }
    if EFF_DC.load(Relaxed) < SOFT_DC.load(Relaxed) { return Ok(()); }
    sync()?;
    pace_core();
    Ok(())
}

fn pace_core() {
    let now = Instant::now();
    let work = LAST.with(|c| c.replace(Some(now))).map(|t| now - t).filter(|d| *d <= WORK_GAP).unwrap_or(Duration::ZERO);
    let (soft, hard) = (SOFT_DC.load(Relaxed), HARD_DC.load(Relaxed));
    let t = EFF_DC.load(Relaxed);
    if t < soft {
        // the transition lines use a 2 C hysteresis: a box hovering at the soft limit logs once, not every crossing
        if t + 20 <= soft && IN_BAND.swap(false, Relaxed) {
            crate::dreprintln!("[thermal] {:.1} C: 2 C below {:.0} C again, pacing ended", t as f64 / 10.0, soft as f64 / 10.0);
        }
        return;
    }
    if !IN_BAND.swap(true, Relaxed) {
        crate::reprintln!("[thermal] {:.1} C: at or above {:.0} C, pacing GPU work (hold at {:.0} C)",
                          t as f64 / 10.0, soft as f64 / 10.0, hard as f64 / 10.0);
    }
    if t >= hard {
        // hold: wait for the box to come 1 C under the hard limit, at most MAX_SLEEP per call
        HOLDS.fetch_add(1, Relaxed);
        let t0 = Instant::now();
        // release 2 C under the limit (the measured value, not the projection: a cooling box projects no rise)
        while HOT_DC.load(Relaxed) + 20 > hard && t0.elapsed() < MAX_SLEEP {
            std::thread::sleep(Duration::from_millis(100));
        }
        HELD_NS.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    } else {
        let s = pacing_sleep(t, soft, hard, work);
        if !s.is_zero() {
            std::thread::sleep(s);
            PACED_NS.fetch_add(s.as_nanos() as u64, Relaxed);
            PACED_CALLS.fetch_add(1, Relaxed);
        }
    }
    // the sleep is not GPU work: the next call's work starts now
    LAST.with(|c| c.set(Some(Instant::now())));
}

/// For /metrics and owner_check: (on, hottest C, peak C, soft C, hard C, seconds paced, seconds held, holds, paced calls).
pub fn counters() -> (bool, f64, f64, f64, f64, f64, f64, u64, u64) {
    let on = STATE.load(Relaxed) == 2;
    (on, HOT_DC.load(Relaxed) as f64 / 10.0, PEAK_DC.load(Relaxed) as f64 / 10.0, SOFT_DC.load(Relaxed) as f64 / 10.0,
     HARD_DC.load(Relaxed) as f64 / 10.0, PACED_NS.load(Relaxed) as f64 / 1e9, HELD_NS.load(Relaxed) as f64 / 1e9,
     HOLDS.load(Relaxed), PACED_CALLS.load(Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pacing_is_zero_below_soft_and_rises_to_max_factor_at_hard() {
        let w = Duration::from_millis(100);
        assert_eq!(pacing_sleep(879, 880, 920, w), Duration::ZERO);
        assert_eq!(pacing_sleep(880, 880, 920, w), Duration::ZERO);
        let mid = pacing_sleep(900, 880, 920, w);
        // half way: duty 1 - 0.85/2 = 0.575 -> sleep 0.425/0.575 x work
        assert!((mid.as_secs_f64() - 0.1 * 0.425 / 0.575).abs() < 1e-9, "half way: {mid:?}");
        let full = pacing_sleep(920, 880, 920, w);
        assert!((full.as_secs_f64() - 0.1 * 0.85 / 0.15).abs() < 1e-9, "at the hard limit duty = MIN_DUTY: {full:?}");
        assert_eq!(pacing_sleep(990, 880, 920, w), full, "the band saturates at the hard limit");
        assert_eq!(pacing_sleep(919, 880, 920, Duration::from_secs(10)), MAX_SLEEP, "capped at MAX_SLEEP");
        assert!(MIN_DUTY > 0.0 && MIN_DUTY < 1.0 && PASS_EVERY_LAYERS >= 1);
        assert_eq!(pacing_sleep(900, 900, 900, w), Duration::ZERO, "degenerate band never paces");
    }

    #[test]
    fn lead_projects_a_rise_only_and_is_capped() {
        assert_eq!(lead_dc(880, 870), 0, "cooling projects nothing");
        assert_eq!(lead_dc(880, 880), 0);
        assert_eq!(lead_dc(880, 900), 20, "+2 C over 2 s projects +2 C at a 2 s lookahead");
        assert_eq!(lead_dc(800, 950), MAX_LEAD_DC, "capped");
    }

    #[test]
    fn hottest_zone_skips_implausible_readings() {
        let d = std::env::temp_dir().join(format!("thermal_test_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let w = |n: &str, v: &str| { let p = d.join(n); std::fs::write(&p, v).unwrap(); p };
        let paths = vec![w("a", "56800\n"), w("b", "91250\n"), w("c", "0\n"), w("d", "255000\n"), d.join("missing")];
        assert_eq!(read_hottest(&paths), Some(912));
        assert_eq!(read_hottest(&[d.join("missing")]), None);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn max_sleep_stays_well_inside_the_tp_watchdog() {
        assert!(MAX_SLEEP <= Duration::from_secs(2), "the proxy watchdog fires at 10 s of unmatched debt");
    }
}
