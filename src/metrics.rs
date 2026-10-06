//! Prometheus `/metrics` (community ask, plan CF-P1b): request-level serving metrics in the Prometheus text
//! exposition format, for both engines (NVFP4 and EXL3 share the HTTP layer) and every topology (the TP head
//! serves the API).
//!
//! Cost rule (owner 2026-10-02: on by default only if it costs nothing): nothing here runs on the GPU path or
//! inside the scheduler. A request carries one `Req` (a few plain fields, a local token counter); at request end
//! it does ~a dozen relaxed atomic adds. Everything else is computed when `/metrics` is scraped.
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed};
use std::time::Instant;

const TTFT_S: [f64; 12] = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0];
const E2E_S: [f64; 11] = [0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0];
const DECODE_TPS: [f64; 11] = [5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0];
const PREFILL_TPS: [f64; 10] = [100.0, 250.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 6000.0, 8000.0, 12000.0];

/// A fixed-bucket histogram of relaxed atomics (counts per bucket + sum in micro-units + count).
struct Hist<const N: usize> {
    bounds: &'static [f64; N],
    buckets: [AtomicU64; N],
    over: AtomicU64,
    sum_micro: AtomicU64,
    count: AtomicU64,
}

impl<const N: usize> Hist<N> {
    const fn new(bounds: &'static [f64; N]) -> Self {
        Hist { bounds, buckets: [const { AtomicU64::new(0) }; N], over: AtomicU64::new(0),
               sum_micro: AtomicU64::new(0), count: AtomicU64::new(0) }
    }
    fn observe(&self, v: f64) {
        if !v.is_finite() || v < 0.0 { return; }
        match self.bounds.iter().position(|&b| v <= b) {
            Some(i) => self.buckets[i].fetch_add(1, Relaxed),
            None => self.over.fetch_add(1, Relaxed),
        };
        self.sum_micro.fetch_add((v * 1e6) as u64, Relaxed);
        self.count.fetch_add(1, Relaxed);
    }
    fn render(&self, out: &mut String, name: &str, help: &str) {
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} histogram\n"));
        let mut cum = 0u64;
        for (i, b) in self.bounds.iter().enumerate() {
            cum += self.buckets[i].load(Relaxed);
            out.push_str(&format!("{name}_bucket{{le=\"{b}\"}} {cum}\n"));
        }
        cum += self.over.load(Relaxed);
        out.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {cum}\n"));
        out.push_str(&format!("{name}_sum {}\n", self.sum_micro.load(Relaxed) as f64 / 1e6));
        out.push_str(&format!("{name}_count {}\n", self.count.load(Relaxed)));
    }
}

const FINISHES: [&str; 5] = ["stop", "length", "tool_calls", "error", "cancelled"];
static REQ_BY_FINISH: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
static PROMPT_TOKENS: AtomicU64 = AtomicU64::new(0);
static CACHED_PROMPT: AtomicU64 = AtomicU64::new(0); // v0.7.3 #8.2: prefix-cache-served prompt tokens
static GEN_TOKENS: AtomicU64 = AtomicU64::new(0);
static IN_FLIGHT: AtomicU64 = AtomicU64::new(0);
static MAX_BATCH: AtomicU64 = AtomicU64::new(0);
static TTFT: Hist<12> = Hist::new(&TTFT_S);
static E2E: Hist<11> = Hist::new(&E2E_S);
static DECODE: Hist<11> = Hist::new(&DECODE_TPS);
static PREFILL: Hist<10> = Hist::new(&PREFILL_TPS);
static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

/// Called once at boot by every server (lane capacity: running vs waiting is derived from it).
pub fn set_max_batch(n: usize) {
    MAX_BATCH.store(n as u64, Relaxed);
    START.get_or_init(Instant::now);
}

/// One accepted request, from the moment the scheduler took it to the moment its response ended. Dropping it
/// without `finish` (client disconnected, handler returned early) records it as `cancelled`.
pub struct Req {
    t0: Instant,
    first: Option<Instant>,
    prompt: usize,
    ntok: u64,
    cached: usize,
    finish: Option<&'static str>,
}

impl Req {
    pub fn start(prompt_tokens: usize) -> Req {
        IN_FLIGHT.fetch_add(1, Relaxed);
        Req { t0: Instant::now(), first: None, prompt: prompt_tokens, ntok: 0, cached: 0, finish: None }
    }
    /// One generated token reached the HTTP layer.
    #[inline]
    pub fn tok(&mut self) {
        if self.first.is_none() { self.first = Some(Instant::now()); }
        self.ntok += 1;
    }
    /// v0.7.3 (#8.2): prompt tokens this request served from the prefix cache
    /// (TokEvent::Admitted). Counted once, at request end, into
    /// `velogb10_prompt_tokens_cached_total`.
    pub fn cached(&mut self, n: u64) { self.cached = n as usize; }
    /// The engine's finish reason ("stop", "length", "tool_calls", "error: ...", ...).
    pub fn finish(&mut self, reason: &str) {
        self.finish = Some(if reason.starts_with("error") { "error" }
                           else if reason == "length" { "length" }
                           else if reason == "tool_calls" { "tool_calls" }
                           else if reason == "cancelled" { "cancelled" }
                           else { "stop" });
    }
}

impl Drop for Req {
    fn drop(&mut self) {
        IN_FLIGHT.fetch_sub(1, Relaxed);
        let f = self.finish.unwrap_or("cancelled");
        if let Some(i) = FINISHES.iter().position(|&x| x == f) { REQ_BY_FINISH[i].fetch_add(1, Relaxed); }
        PROMPT_TOKENS.fetch_add(self.prompt as u64, Relaxed);
        CACHED_PROMPT.fetch_add(self.cached as u64, Relaxed);
        GEN_TOKENS.fetch_add(self.ntok, Relaxed);
        let end = Instant::now();
        E2E.observe((end - self.t0).as_secs_f64());
        if let Some(first) = self.first {
            let ttft = (first - self.t0).as_secs_f64();
            TTFT.observe(ttft);
            if ttft > 0.0 && self.prompt > 0 { PREFILL.observe(self.prompt as f64 / ttft); }
            let dec = (end - first).as_secs_f64();
            if self.ntok > 1 && dec > 0.0 { DECODE.observe((self.ntok - 1) as f64 / dec); }
        }
    }
}

// ─── v0.7.3 long-run gauges (REL_LONGRUN_REPORT.md §3) ───────────────────────────────────
// Head-only, all atomics; the /proc-backed ones are read AT SCRAPE (no hot-path cost).
static REQ_REJECTED: AtomicU64 = AtomicU64::new(0);
static STREAMS_CANCELLED_BACKLOG: AtomicU64 = AtomicU64::new(0);
static SCHED_STEPS: AtomicU64 = AtomicU64::new(0);
/// Milliseconds since `START` when the scheduler last completed a round; 0 = never stepped.
static SCHED_LAST_MS: AtomicU64 = AtomicU64::new(0);
static SCHED_BUSY: AtomicU8 = AtomicU8::new(0);
static GRAPH_ENTRIES: AtomicU64 = AtomicU64::new(0);

/// LR-4: a request rejected at admission because too many were already waiting.
pub fn reject_request() { REQ_REJECTED.fetch_add(1, Relaxed); }
/// LR-5: a stream cancelled by the scheduler for exceeding --stream-backlog-events.
pub fn stream_backlog_cancel() { STREAMS_CANCELLED_BACKLOG.fetch_add(1, Relaxed); }
/// A scheduler round STARTING (called from BOTH schedulers' step functions; ~35/s). Sets busy;
/// the age timestamp is written when work COMPLETES (`sched_touch`), so
/// velogb10_scheduler_last_step_age_seconds measures time since the last COMPLETED round or
/// prefill chunk — a 100 s prefill advances the age per chunk instead of reading as a wedge.
pub fn sched_step() {
    SCHED_STEPS.fetch_add(1, Relaxed);
    SCHED_BUSY.store(1, Relaxed);
}
/// H3 (REL_V0_7_3): one unit of scheduler work COMPLETED (the end of a decode step, or one
/// prefill chunk). One relaxed store; keeps the age gauge honest during long prefills.
pub fn sched_touch() {
    if let Some(s) = START.get() { SCHED_LAST_MS.store(s.elapsed().as_millis() as u64, Relaxed); }
}
/// The scheduler is about to block waiting for work (between rounds): busy goes 0. On an idle
/// server the age gauge then grows — that is EXPECTED (no rounds are running; nothing is
/// wedged); the alert rule pairs busy==1 with a high age, never busy==0.
pub fn sched_idle() { SCHED_BUSY.store(0, Relaxed); }
/// CUDA graphs currently held (published by the schedulers at boot-precapture end and per
/// lazy capture — the audit's LR-6 visibility).
pub fn set_graph_entries(n: u64) { GRAPH_ENTRIES.store(n, Relaxed); }
/// Admitted-but-unfinished requests (running + waiting) — exact, from the Req guard.
pub fn inflight() -> u64 { IN_FLIGHT.load(Relaxed) }
/// Lane capacity as set at boot (`set_max_batch`).
pub fn max_batch() -> u64 { MAX_BATCH.load(Relaxed) }

fn proc_status_kb(field: &str) -> Option<u64> {
    std::fs::read_to_string("/proc/self/status").ok()?.lines()
        .find(|l| l.starts_with(field))?
        .split_whitespace().nth(1)?.parse().ok()
}

fn open_fds() -> Option<u64> {
    let mut n = 0u64;
    let mut rd = std::fs::read_dir("/proc/self/fd").ok()?;
    while let Some(_e) = rd.next() { n += 1; }
    Some(n)
}

/// The `/metrics` body (Prometheus text format 0.0.4).
pub fn render(engine_alive: bool) -> String {
    let mut o = String::with_capacity(8192);
    o.push_str("# HELP velogb10_build_info Build identity (value is always 1).\n# TYPE velogb10_build_info gauge\n");
    o.push_str(&format!("velogb10_build_info{{version=\"{}\",commit=\"{}\",dirty=\"{}\",kernel_build=\"{}\"}} 1\n",
                        env!("CARGO_PKG_VERSION"), option_env!("TUNE_GIT_COMMIT").unwrap_or("unknown"),
                        option_env!("TUNE_GIT_DIRTY").unwrap_or("unknown"), env!("KERNEL_BUILD_ID")));
    o.push_str("# HELP velogb10_engine_alive 1 while the engine serves, 0 once it is dead (/health answers 503).\n");
    o.push_str(&format!("# TYPE velogb10_engine_alive gauge\nvelogb10_engine_alive {}\n", engine_alive as u8));
    if let Some(s) = START.get() {
        o.push_str(&format!("# HELP velogb10_uptime_seconds Seconds since the server started.\n# TYPE velogb10_uptime_seconds gauge\nvelogb10_uptime_seconds {:.1}\n",
                            s.elapsed().as_secs_f64()));
    }
    let inflight = IN_FLIGHT.load(Relaxed);
    let cap = MAX_BATCH.load(Relaxed);
    let running = if cap > 0 { inflight.min(cap) } else { inflight };
    o.push_str(&format!("# HELP velogb10_max_batch Lane capacity (--max-batch).\n# TYPE velogb10_max_batch gauge\nvelogb10_max_batch {cap}\n"));
    o.push_str(&format!("# HELP velogb10_requests_running Requests holding a lane.\n# TYPE velogb10_requests_running gauge\nvelogb10_requests_running {running}\n"));
    o.push_str(&format!("# HELP velogb10_requests_waiting Requests queued for a lane.\n# TYPE velogb10_requests_waiting gauge\nvelogb10_requests_waiting {}\n",
                        inflight - running));
    o.push_str("# HELP velogb10_requests_total Finished requests by finish reason.\n# TYPE velogb10_requests_total counter\n");
    for (i, f) in FINISHES.iter().enumerate() {
        o.push_str(&format!("velogb10_requests_total{{finish=\"{f}\"}} {}\n", REQ_BY_FINISH[i].load(Relaxed)));
    }
    o.push_str(&format!("# HELP velogb10_prompt_tokens_total Prompt tokens of finished requests.\n# TYPE velogb10_prompt_tokens_total counter\nvelogb10_prompt_tokens_total {}\n",
                        PROMPT_TOKENS.load(Relaxed)));
    o.push_str(&format!("# HELP velogb10_prompt_tokens_cached_total Prompt tokens served from the prefix cache (prefill skipped) of finished requests.\n# TYPE velogb10_prompt_tokens_cached_total counter\nvelogb10_prompt_tokens_cached_total {}\n",
                        CACHED_PROMPT.load(Relaxed)));
    o.push_str(&format!("# HELP velogb10_generation_tokens_total Generated tokens of finished requests.\n# TYPE velogb10_generation_tokens_total counter\nvelogb10_generation_tokens_total {}\n",
                        GEN_TOKENS.load(Relaxed)));
    TTFT.render(&mut o, "velogb10_time_to_first_token_seconds", "Request receipt to first generated token (queueing + prefill).");
    E2E.render(&mut o, "velogb10_request_duration_seconds", "Request receipt to the end of its response.");
    PREFILL.render(&mut o, "velogb10_prefill_tokens_per_second", "Prompt tokens / time to first token, per request (includes queueing).");
    DECODE.render(&mut o, "velogb10_decode_tokens_per_second", "Per-request generation rate after the first token.");
    // v0.7.3 long-run gauges (REL_LONGRUN §3) — /proc reads happen here, at scrape time.
    let rss_kb = proc_status_kb("VmRSS:");
    if let Some(kb) = rss_kb {
        o.push_str("# HELP velogb10_process_rss_bytes Resident set size of the server process.\n# TYPE velogb10_process_rss_bytes gauge\n");
        o.push_str(&format!("velogb10_process_rss_bytes {}\n", kb * 1024));
    }
    if let Some(t) = proc_status_kb("Threads:") {
        o.push_str("# HELP velogb10_thread_count OS threads of the server process (leak tripwire).\n# TYPE velogb10_thread_count gauge\n");
        o.push_str(&format!("velogb10_thread_count {t}\n"));
    }
    if let Some(f) = open_fds() {
        o.push_str("# HELP velogb10_open_fds Open file descriptors (leak tripwire).\n# TYPE velogb10_open_fds gauge\n");
        o.push_str(&format!("velogb10_open_fds {f}\n"));
    }
    if let Some(a) = crate::memwatch::mem_available_bytes() {
        o.push_str("# HELP velogb10_mem_available_bytes Box MemAvailable (the memwatch metric).\n# TYPE velogb10_mem_available_bytes gauge\n");
        o.push_str(&format!("velogb10_mem_available_bytes {a}\n"));
    }
    let min_av = crate::memwatch::min_available_gb();
    if min_av.is_finite() {
        o.push_str("# HELP velogb10_mem_min_available_bytes Minimum MemAvailable seen since boot.\n# TYPE velogb10_mem_min_available_bytes gauge\n");
        o.push_str(&format!("velogb10_mem_min_available_bytes {}\n", (min_av * 1e9) as u64));
    }
    o.push_str("# HELP velogb10_scheduler_steps_total Scheduler rounds completed since boot.\n# TYPE velogb10_scheduler_steps_total counter\n");
    o.push_str(&format!("velogb10_scheduler_steps_total {}\n", SCHED_STEPS.load(Relaxed)));
    o.push_str("# HELP velogb10_scheduler_busy 1 while scheduler work (round, admission, prefill) is in progress, 0 while blocked idle-waiting for a request.\n# TYPE velogb10_scheduler_busy gauge\n");
    o.push_str(&format!("velogb10_scheduler_busy {}\n", SCHED_BUSY.load(Relaxed)));
    if let Some(s0) = START.get() {
        let now_ms = s0.elapsed().as_millis() as u64;
        let last = SCHED_LAST_MS.load(Relaxed);
        let age = if last == 0 { now_ms } else { now_ms.saturating_sub(last) }; // H9: no wrap
        o.push_str("# HELP velogb10_scheduler_last_step_age_seconds Seconds since the scheduler last COMPLETED a round or a prefill chunk (uptime if none yet; grows while idle — alert only with scheduler_busy == 1).\n# TYPE velogb10_scheduler_last_step_age_seconds gauge\n");
        o.push_str(&format!("velogb10_scheduler_last_step_age_seconds {:.3}\n", age as f64 / 1000.0));
    }
    o.push_str("# HELP velogb10_graph_cache_entries CUDA graphs currently held (LR-6 visibility).\n# TYPE velogb10_graph_cache_entries gauge\n");
    o.push_str(&format!("velogb10_graph_cache_entries {}\n", GRAPH_ENTRIES.load(Relaxed)));
    o.push_str("# HELP velogb10_log_lines_dropped_total Log lines dropped by logq (LR-1: the log reader stalled).\n# TYPE velogb10_log_lines_dropped_total counter\n");
    o.push_str(&format!("velogb10_log_lines_dropped_total {}\n", crate::logq::dropped()));
    o.push_str("# HELP velogb10_requests_rejected_total Requests refused at admission (--max-waiting exceeded; LR-4).\n# TYPE velogb10_requests_rejected_total counter\n");
    o.push_str(&format!("velogb10_requests_rejected_total {}\n", REQ_REJECTED.load(Relaxed)));
    o.push_str("# HELP velogb10_streams_cancelled_backlog_total Streams cancelled for exceeding --stream-backlog-events (LR-5).\n# TYPE velogb10_streams_cancelled_backlog_total counter\n");
    o.push_str(&format!("velogb10_streams_cancelled_backlog_total {}\n", STREAMS_CANCELLED_BACKLOG.load(Relaxed)));
    let (steps, drafts, accepted, emitted) = crate::tel::spec_totals();
    o.push_str("# HELP velogb10_spec_draft_tokens_total Speculative draft tokens proposed (MTP / DFlash2 / DFlash).\n# TYPE velogb10_spec_draft_tokens_total counter\n");
    o.push_str(&format!("velogb10_spec_draft_tokens_total {drafts}\n"));
    o.push_str("# HELP velogb10_spec_accepted_tokens_total Speculative draft tokens accepted by the verify.\n# TYPE velogb10_spec_accepted_tokens_total counter\n");
    o.push_str(&format!("velogb10_spec_accepted_tokens_total {accepted}\n"));
    o.push_str("# HELP velogb10_spec_rounds_total Speculative rounds (verify forwards).\n# TYPE velogb10_spec_rounds_total counter\n");
    o.push_str(&format!("velogb10_spec_rounds_total {steps}\n"));
    o.push_str("# HELP velogb10_spec_emitted_tokens_total Tokens emitted by speculative rounds (accepted + 1 per round).\n# TYPE velogb10_spec_emitted_tokens_total counter\n");
    o.push_str(&format!("velogb10_spec_emitted_tokens_total {emitted}\n"));
    o
}

#[cfg(test)]
mod tests {
    /// H3 tests drive the same scheduler statics — serialize them so busy/idle reads are exact.
    static SCHED_GAUGE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn render_has_longrun_gauges() {
        let _g = SCHED_GAUGE_LOCK.lock().unwrap();
        // Same value the sibling test asserts (2): initialises START and can never race it.
        super::set_max_batch(2);
        super::reject_request();
        super::stream_backlog_cancel();
        super::sched_step();
        super::sched_idle();
        super::set_graph_entries(3);
        let s = super::render(true);
        for name in [
            "velogb10_process_rss_bytes", "velogb10_thread_count", "velogb10_open_fds",
            "velogb10_mem_available_bytes", "velogb10_scheduler_steps_total",
            "velogb10_scheduler_busy", "velogb10_scheduler_last_step_age_seconds",
            "velogb10_graph_cache_entries", "velogb10_log_lines_dropped_total",
            "velogb10_requests_rejected_total", "velogb10_streams_cancelled_backlog_total",
        ] {
            assert!(s.contains(name), "missing gauge {name}");
        }
        for l in s.lines().filter(|l| !l.starts_with('#')) {
            let v = l.rsplit(' ').next().unwrap();
            assert!(v.parse::<f64>().is_ok(), "not a sample line: {l}");
        }
        assert!(s.lines().any(|l| l.starts_with("velogb10_requests_rejected_total") && l.ends_with(" 1")));
    }

    /// H3 (REL_V0_7_3): the busy/idle transitions and the END-of-work age timestamp, on the
    /// REAL gauge functions (what the schedulers call): idle -> 0, round start -> 1, and
    /// `sched_touch` is what writes the age (a completed unit of work), so a scrape right
    /// after work reads a small age even though the round started earlier.
    #[test]
    fn sched_gauges_busy_idle_and_age_transitions() {
        let _g = SCHED_GAUGE_LOCK.lock().unwrap();
        super::set_max_batch(2); // initialises START (age timestamps need it)
        let busy = || super::render(true).lines()
            .find(|l| l.starts_with("velogb10_scheduler_busy")).unwrap()
            .rsplit(' ').next().unwrap().to_string();
        super::sched_idle();
        assert_eq!(busy(), "0", "blocked idle-waiting must read busy 0 (H3: it stuck at 1)");
        super::sched_step();
        assert_eq!(busy(), "1", "a started round must read busy 1");
        std::thread::sleep(std::time::Duration::from_millis(80));
        super::sched_touch(); // the round COMPLETES now — the age measures from HERE
        std::thread::sleep(std::time::Duration::from_millis(30));
        let age = super::render(true).lines()
            .find(|l| l.starts_with("velogb10_scheduler_last_step_age_seconds")).unwrap()
            .rsplit(' ').next().unwrap().parse::<f64>().unwrap();
        assert!(age < 0.060, "age {age} must count from sched_touch (end of work): from the round start it would be >= 0.110");
        assert!(age >= 0.0, "age is a sane non-negative gauge");
        super::sched_idle();
        assert_eq!(busy(), "0", "idle again reads 0");
    }

    #[test]
    fn render_is_prometheus_text_and_counts_a_request() {
        super::set_max_batch(2);
        {
            let mut r = super::Req::start(100);
            r.tok(); r.tok(); r.tok();
            r.cached(40); // v0.7.3 #8.2: the request's prefix-cache hit
            r.finish("length");
        }
        { let _dropped = super::Req::start(7); } // never finished: a cancelled request
        let s = super::render(true);
        assert!(s.contains("velogb10_engine_alive 1"));
        assert!(s.contains("velogb10_max_batch 2"));
        assert!(s.lines().any(|l| l.starts_with("velogb10_requests_total{finish=\"length\"}") && !l.ends_with(" 0")));
        assert!(s.lines().any(|l| l.starts_with("velogb10_requests_total{finish=\"cancelled\"}") && !l.ends_with(" 0")));
        assert!(s.contains("velogb10_time_to_first_token_seconds_bucket{le=\"+Inf\"}"));
        // v0.7.3 #8.2: the cached-prompt-tokens counter, HELP + TYPE + a sample >= 40.
        assert!(s.contains("# HELP velogb10_prompt_tokens_cached_total"));
        assert!(s.contains("# TYPE velogb10_prompt_tokens_cached_total counter"));
        let cv: f64 = s.lines().find(|l| l.starts_with("velogb10_prompt_tokens_cached_total "))
            .expect("cached counter sample line").rsplit(' ').next().unwrap().parse().unwrap();
        assert!(cv >= 40.0, "cached counter should count the request's 40 cached tokens, got {cv}");
        for l in s.lines().filter(|l| !l.starts_with('#')) {
            let v = l.rsplit(' ').next().unwrap();
            assert!(v.parse::<f64>().is_ok(), "not a sample line: {l}");
        }
    }
}
