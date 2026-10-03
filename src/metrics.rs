//! Prometheus `/metrics` (community ask, plan CF-P1b): request-level serving metrics in the Prometheus text
//! exposition format, for both engines (NVFP4 and EXL3 share the HTTP layer) and every topology (the TP head
//! serves the API).
//!
//! Cost rule (owner 2026-10-02: on by default only if it costs nothing): nothing here runs on the GPU path or
//! inside the scheduler. A request carries one `Req` (a few plain fields, a local token counter); at request end
//! it does ~a dozen relaxed atomic adds. Everything else is computed when `/metrics` is scraped.
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
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
    finish: Option<&'static str>,
}

impl Req {
    pub fn start(prompt_tokens: usize) -> Req {
        IN_FLIGHT.fetch_add(1, Relaxed);
        Req { t0: Instant::now(), first: None, prompt: prompt_tokens, ntok: 0, finish: None }
    }
    /// One generated token reached the HTTP layer.
    #[inline]
    pub fn tok(&mut self) {
        if self.first.is_none() { self.first = Some(Instant::now()); }
        self.ntok += 1;
    }
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
    o.push_str(&format!("# HELP velogb10_generation_tokens_total Generated tokens of finished requests.\n# TYPE velogb10_generation_tokens_total counter\nvelogb10_generation_tokens_total {}\n",
                        GEN_TOKENS.load(Relaxed)));
    TTFT.render(&mut o, "velogb10_time_to_first_token_seconds", "Request receipt to first generated token (queueing + prefill).");
    E2E.render(&mut o, "velogb10_request_duration_seconds", "Request receipt to the end of its response.");
    PREFILL.render(&mut o, "velogb10_prefill_tokens_per_second", "Prompt tokens / time to first token, per request (includes queueing).");
    DECODE.render(&mut o, "velogb10_decode_tokens_per_second", "Per-request generation rate after the first token.");
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
    #[test]
    fn render_is_prometheus_text_and_counts_a_request() {
        super::set_max_batch(2);
        {
            let mut r = super::Req::start(100);
            r.tok(); r.tok(); r.tok();
            r.finish("length");
        }
        { let _dropped = super::Req::start(7); } // never finished: a cancelled request
        let s = super::render(true);
        assert!(s.contains("velogb10_engine_alive 1"));
        assert!(s.contains("velogb10_max_batch 2"));
        assert!(s.lines().any(|l| l.starts_with("velogb10_requests_total{finish=\"length\"}") && !l.ends_with(" 0")));
        assert!(s.lines().any(|l| l.starts_with("velogb10_requests_total{finish=\"cancelled\"}") && !l.ends_with(" 0")));
        assert!(s.contains("velogb10_time_to_first_token_seconds_bucket{le=\"+Inf\"}"));
        for l in s.lines().filter(|l| !l.starts_with('#')) {
            let v = l.rsplit(' ').next().unwrap();
            assert!(v.parse::<f64>().is_ok(), "not a sample line: {l}");
        }
    }
}
