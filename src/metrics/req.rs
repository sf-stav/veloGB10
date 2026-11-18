//! Request-level metrics: the fixed-bucket latency histograms, the per-finish and token totals, and the `Req` guard one request carries from admission to its end.
//! Cost rule (module header in `mod.rs`): a request does about a dozen relaxed atomic adds at its end; nothing here runs on the GPU path. Map: `mod.rs`.

use super::*;

const TTFT_S: [f64; 12] = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0];
const E2E_S: [f64; 11] = [0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0];
const DECODE_TPS: [f64; 11] = [5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0];
const PREFILL_TPS: [f64; 10] = [100.0, 250.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 6000.0, 8000.0, 12000.0];

/// A fixed-bucket histogram of relaxed atomics (counts per bucket + sum in micro-units + count).
pub(super) struct Hist<const N: usize> {
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
    pub(super) fn render(&self, out: &mut String, name: &str, help: &str) {
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

pub(super) const FINISHES: [&str; 5] = ["stop", "length", "tool_calls", "error", "cancelled"];
pub(super) static REQ_BY_FINISH: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
pub(super) static PROMPT_TOKENS: AtomicU64 = AtomicU64::new(0);
pub(super) static CACHED_PROMPT: AtomicU64 = AtomicU64::new(0); // v0.7.3 #8.2: prefix-cache-served prompt tokens
pub(super) static GEN_TOKENS: AtomicU64 = AtomicU64::new(0);
pub(super) static TTFT: Hist<12> = Hist::new(&TTFT_S);
pub(super) static E2E: Hist<11> = Hist::new(&E2E_S);
pub(super) static DECODE: Hist<11> = Hist::new(&DECODE_TPS);
pub(super) static PREFILL: Hist<10> = Hist::new(&PREFILL_TPS);
pub(super) static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

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
