//! Prometheus `/metrics` (community ask, plan CF-P1b): request-level serving metrics in the Prometheus text
//! exposition format, for both engines (NVFP4 and EXL3 share the HTTP layer) and every topology (the TP head
//! serves the API).
//!
//! Cost rule (owner 2026-10-02: on by default only if it costs nothing): nothing here runs on the GPU path or
//! inside the scheduler. A request carries one `Req` (a few plain fields, a local token counter); at request end
//! it does ~a dozen relaxed atomic adds. Everything else is computed when `/metrics` is scraped.
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::{Acquire, Relaxed, Release}};
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
// TOOLSCRUB1: spans redacted by --scrub-tool-secrets, by layer, since boot (advisory
// liveness: a span split by the pending cap can count twice). Zero forever when the
// flag is off (the default): the counters exist but never move.
static SCRUB_PROV_HITS: AtomicU64 = AtomicU64::new(0);
static SCRUB_PAT_HITS: AtomicU64 = AtomicU64::new(0);

pub fn scrub_hit_provenance() { SCRUB_PROV_HITS.fetch_add(1, Relaxed); }
pub fn scrub_hit_pattern() { SCRUB_PAT_HITS.fetch_add(1, Relaxed); }

// DFLASH2-GEN: the DF2 chain's engagement counters on the NON-EXL3 (nvfp4) serve path.
// scripts/owner_check.py prints them; the exl3_df2_* series are the EXL3 server's arm.
static DF2_ROUNDS: AtomicU64 = AtomicU64::new(0);
static DF2_MASK_OVERRIDE: AtomicU64 = AtomicU64::new(0);

pub fn df2_round() { DF2_ROUNDS.fetch_add(1, Relaxed); }
pub fn df2_mask_override(on: bool) { DF2_MASK_OVERRIDE.store(on as u64, Relaxed); }

// TP2BIT5 (#26) / EXL3 width-closure: graceful-fallback engagement counters. AGENTS §1b-bis.2: a
// feature is done only when its engagement counter is non-zero on the real posture; a counter that
// never moves is a path that never runs.
static EXL3_PLE_WIDTH_ACCEPTED: AtomicU64 = AtomicU64::new(0); // B9: a PLE width outside {5,6} loaded and served
pub fn exl3_ple_width_accepted() { EXL3_PLE_WIDTH_ACCEPTED.fetch_add(1, Relaxed); }
/// FP4L S5 (shared): the n-gram (PLE) table read-ahead — prefill chunks whose NEXT chunk's n-gram rows were read into
/// the page cache on background threads while the current chunk ran (SSD-resident table only; 0 = never ran), the
/// n-gram rows that covered and the deduplicated 4 KB page reads issued. scripts/owner_check.py prints them.
static PLE_WARM_AHEAD_CHUNKS: AtomicU64 = AtomicU64::new(0);
static PLE_WARM_AHEAD_ROWS: AtomicU64 = AtomicU64::new(0);
static PLE_WARM_AHEAD_READS: AtomicU64 = AtomicU64::new(0);
pub fn ple_warm_ahead(rows: u64, reads: u64) {
    PLE_WARM_AHEAD_CHUNKS.fetch_add(1, Relaxed);
    PLE_WARM_AHEAD_ROWS.fetch_add(rows, Relaxed);
    PLE_WARM_AHEAD_READS.fetch_add(reads, Relaxed);
}

static EXL3_W4MOE_WIDTH_SERVED: AtomicU64 = AtomicU64::new(0); // B10: persistent grouped experts serving a non-3 width (2,4,5,6,7; 0 = never ran)
pub fn exl3_w4moe_width_served() { EXL3_W4MOE_WIDTH_SERVED.fetch_add(1, Relaxed); }

static EXL3_DEVROUTE_FALLBACK: AtomicU64 = AtomicU64::new(0);
static EXL3_FRAC_PF_BRIDGE_ROWS: AtomicU64 = AtomicU64::new(0); // LEG-B: prefill rows served by the fractional-k2 decode-moe bridge
pub fn exl3_frac_pf_bridge_rows(add: u64) { EXL3_FRAC_PF_BRIDGE_ROWS.fetch_add(add, Relaxed); } // TP-B: host-routed arm asked under TP -> device-routed EP arm ran instead
pub fn exl3_devroute_fallback() { EXL3_DEVROUTE_FALLBACK.fetch_add(1, Relaxed); }

static EXL3_NGRAM_EMBED_ROWS: AtomicU64 = AtomicU64::new(0); // 1.66-class packs: token-embed rows decoded by the ngram codec at load (0 = plain embed path)
pub fn exl3_ngram_embed_decoded(rows: u64) { EXL3_NGRAM_EMBED_ROWS.fetch_add(rows, Relaxed); }
static EXL3_MTP_SKIPPED: AtomicU64 = AtomicU64::new(0); // #26 Fix 4: boots whose draft head exceeds the served width set -> served WITHOUT speculation
pub fn exl3_mtp_skipped() { EXL3_MTP_SKIPPED.fetch_add(1, Relaxed); }
static EXL3_PF_KERNEL_WIDTH_FALLBACK: AtomicU64 = AtomicU64::new(0);
// TPSHIP (2026-11-07): EXL3 boots asked for --kv-cache bf16 (no EXL3 store) and served f16 instead (AGENTS 1b-bis.8).
static EXL3_KV_CACHE_FALLBACK: AtomicU64 = AtomicU64::new(0);
pub fn exl3_kv_cache_fallback() { EXL3_KV_CACHE_FALLBACK.fetch_add(1, Relaxed); } // #26 Fix 3: a prefill layer asked for the pipelined expert kernel (3/4-bit) and was served by the wide tiles kernel (3..5) instead
pub fn exl3_pf_kernel_width_fallback() { EXL3_PF_KERNEL_WIDTH_FALLBACK.fetch_add(1, Relaxed); }

// K35 (2026-10-31): fractional-width (K = 1.5 / 2.5 / 3.5 = k2 3 / 5 / 7) routed-expert serving receipts, by k2.
// EXL3_FRAC_EXPERT_ROUNDS = MoE layer rounds whose experts ran the fractional grouped body (decode, verify, MTP draft,
// the prefill bridge): eager calls plus graph replays x the rounds each captured graph holds (a capture executes
// nothing and is not counted; the capture sites take the per-capture tally into the CudaGraph). EXL3_FRAC_PF_WIDE_ROWS
// = prefill expert rows served by the fractional wide-tiles body (exl3_hmma_gemm_wide_tiles, EXL3_FRAC_BITS case).
// 0 everywhere = the fractional path never ran (AGENTS §4: identical hashes are not a liveness receipt).
// K8EXP (2026-11-07): the same tally also counts the newly admitted INTEGER widths K = 7 / 8 (k2 14 / 16), published as
// velogb10_exl3_k78_expert_rounds_total; the arrays are indexed by k2 (0..=16), so EXL3_WIDTH_SLOTS = 17.
pub const EXL3_WIDTH_SLOTS: usize = 17;
static EXL3_FRAC_EXPERT_ROUNDS: [AtomicU64; EXL3_WIDTH_SLOTS] = [const { AtomicU64::new(0) }; EXL3_WIDTH_SLOTS];
static EXL3_FRAC_PF_WIDE_ROWS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
// PFHALF (2026-11-06): prefill expert rows served by the PIPELINED kernel's HALF body (xq_moe_pf, EXL3_FRAC_BITS case).
static EXL3_FRAC_PF_PIPE_ROWS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
thread_local! { static EXL3_FRAC_CAPTURE: std::cell::Cell<[u32; EXL3_WIDTH_SLOTS]> = const { std::cell::Cell::new([0; EXL3_WIDTH_SLOTS]) }; }
pub fn exl3_frac_expert_round(k2: usize, capturing: bool) {
    if k2 >= EXL3_WIDTH_SLOTS { return; }
    if capturing {
        EXL3_FRAC_CAPTURE.with(|c| { let mut v = c.get(); v[k2] += 1; c.set(v); });
    } else {
        EXL3_FRAC_EXPERT_ROUNDS[k2].fetch_add(1, Relaxed);
    }
}
/// Capture bookkeeping: reset right after cuStreamBeginCapture, take at the instantiation (the rounds the graph holds).
pub fn exl3_frac_capture_reset() { EXL3_FRAC_CAPTURE.with(|c| c.set([0; EXL3_WIDTH_SLOTS])); }
pub fn exl3_frac_capture_take() -> [u32; EXL3_WIDTH_SLOTS] { EXL3_FRAC_CAPTURE.with(|c| c.replace([0; EXL3_WIDTH_SLOTS])) }
pub fn exl3_frac_expert_rounds_replayed(n: &[u32; EXL3_WIDTH_SLOTS]) {
    for (k2, v) in n.iter().enumerate() { if *v > 0 { EXL3_FRAC_EXPERT_ROUNDS[k2].fetch_add(*v as u64, Relaxed); } }
}
pub fn exl3_frac_pf_wide_rows(k2: usize, rows: u64) { if k2 < 8 { EXL3_FRAC_PF_WIDE_ROWS[k2].fetch_add(rows, Relaxed); } }
pub fn exl3_frac_pf_pipe_rows(k2: usize, rows: u64) { if k2 < 8 { EXL3_FRAC_PF_PIPE_ROWS[k2].fetch_add(rows, Relaxed); } }
pub fn exl3_frac_expert_rounds(k2: usize) -> u64 { if k2 < EXL3_WIDTH_SLOTS { EXL3_FRAC_EXPERT_ROUNDS[k2].load(Relaxed) } else { 0 } }

// --draft-vocab (FR-Spec, opt-in): the reduced draft head's size (0 = not built: the drafter reads the
// full vocabulary) and the MTP draft tokens it picked. scripts/owner_check.py prints both.
static DRAFT_VOCAB_ROWS: AtomicU64 = AtomicU64::new(0);
static DRAFT_VOCAB_FULL: AtomicU64 = AtomicU64::new(0);
static DRAFT_VOCAB_STEPS: AtomicU64 = AtomicU64::new(0);

pub fn set_draft_vocab(rows: usize, vocab: usize) {
    DRAFT_VOCAB_ROWS.store(rows as u64, Relaxed);
    DRAFT_VOCAB_FULL.store(vocab as u64, Relaxed);
}
pub fn draft_vocab_step() { DRAFT_VOCAB_STEPS.fetch_add(1, Relaxed); }
static IN_FLIGHT: AtomicU64 = AtomicU64::new(0);
static MAX_BATCH: AtomicU64 = AtomicU64::new(0);
/// F2: the `--max-batch` the command line asked for, BEFORE the EXL3 server's 16-lane cap. A number
/// here above `velogb10_max_batch` means the boot was capped (the boot log prints the line).
static MAX_BATCH_REQUESTED: AtomicU64 = AtomicU64::new(0);
/// DSPLITK (owner 2026-11-07): trunk dense split-K attention launches (eager calls and graph captures; replays
/// of a captured graph are not counted). Absent = --exl3-dense-splitk off or no dense layer served.
static EXL3_DENSE_SPLITK_CALLS: AtomicU64 = AtomicU64::new(0);
/// PFTC (owner 2026-11-07): prefill rows whose dense attention ran on the tensor-core kernels (eager prefill;
/// graph replays are not a prefill path). Absent = --exl3-dense-pf-tc off or no such layer served.
static EXL3_PF_TC_ROWS: AtomicU64 = AtomicU64::new(0);
pub fn exl3_pf_tc_rows(n: u64) { EXL3_PF_TC_ROWS.fetch_add(n, Relaxed); }
pub fn exl3_dense_splitk_call() { EXL3_DENSE_SPLITK_CALLS.fetch_add(1, Relaxed); }
/// CTXCLAMP (owner 2026-11-07): an EXL3 TP = 1 boot whose asked --max-seq-len did not fit even one lane booted with one
/// lane at the longest context that fits; asked / served (0 / 0 = no context clamp on this boot).
static EXL3_CTX_ASKED: AtomicU64 = AtomicU64::new(0);
static EXL3_CTX_SERVED: AtomicU64 = AtomicU64::new(0);
pub fn set_exl3_ctx_clamped(asked: usize, served: usize) {
    EXL3_CTX_ASKED.store(asked as u64, Relaxed);
    EXL3_CTX_SERVED.store(served as u64, Relaxed);
}

/// F2: record the uncapped `--max-batch` the argv asked for (next to the served lane capacity).
pub fn set_max_batch_requested(n: usize) {
    MAX_BATCH_REQUESTED.store(n as u64, Relaxed);
}

/// MAXBATCH1: requests that arrived while EVERY lane was busy (above the served cap) and were
/// QUEUED — they run when a lane frees; the cap itself is applied and shown at boot (F2). This is
/// the engagement counter for that cap: non-zero after a load run means over-cap traffic really
/// flowed through the queue and was served, not refused. The refused arm of the same arrival
/// class (LR-4 waiting-limit 503) is `velogb10_over_cap_arrivals_total{disposition="rejected"}`.
static OVER_CAP_QUEUED: AtomicU64 = AtomicU64::new(0);
/// MAXBATCH1: over-cap arrivals refused by the `--max-waiting` gate (503 `server_busy`).
static OVER_CAP_REJECTED: AtomicU64 = AtomicU64::new(0);
/// TOPKM1: sampled requests on the batched GPU samplers (NVFP4/bf16 family) that asked for no
/// top-k truncation (0, or a negative value the server maps to 0) or a top_k above the kernels'
/// 64-candidate limit; served with top_k 64 (`crate::sampler::gpu_top_k`).
static TOP_K_CAPPED: AtomicU64 = AtomicU64::new(0);

/// THINKBUDGET1: requests that carried `thinking_token_budget` and had it applied (the server's model
/// has reasoning markers and the serving path supports the budget).
static THINK_BUDGET_APPLIED: AtomicU64 = AtomicU64::new(0);
/// THINKBUDGET1: requests that carried a budget the serving path could not apply (another model
/// family's scheduler, or a model without reasoning markers): served without it, said once per boot.
static THINK_BUDGET_UNSUPPORTED: AtomicU64 = AtomicU64::new(0);
/// THINKBUDGET1: reasoning blocks the budget ended (the end-of-thinking marker forced).
static THINK_BUDGET_FORCED: AtomicU64 = AtomicU64::new(0);

/// PR #30 port: requests that asked for log probabilities, by outcome (`logprobs_request`).
static LOGPROBS_SERVED: AtomicU64 = AtomicU64::new(0);
static LOGPROBS_UNSUPPORTED: AtomicU64 = AtomicU64::new(0);
static LOGPROBS_WITHHELD: AtomicU64 = AtomicU64::new(0);
static LOGPROBS_STREAM: AtomicU64 = AtomicU64::new(0);
/// PR #31 port: `response_format` json_schema requests: enforced (token mask armed) / unenforced (served with
/// `x-json-schema-enforced: none`: enforcement off, schema not compiled, engine or reasoning mode cannot apply it).
static SCHEMA_ENFORCED: AtomicU64 = AtomicU64::new(0);
static SCHEMA_UNENFORCED: AtomicU64 = AtomicU64::new(0);
/// PR #30 / #31 ports: EXL3 lanes that decoded WITHOUT speculation because the request asked for log probabilities
/// or a schema mask (the verify rows are not read back / masked yet): the speed those requests pay.
static SPEC_OFF_LOGPROBS: AtomicU64 = AtomicU64::new(0);
static SPEC_OFF_SCHEMA: AtomicU64 = AtomicU64::new(0);

/// PR #30 port: one logprobs request's outcome: "served", "unsupported" (engine), "withheld" (scrubber), "stream".
pub fn logprobs_request(disposition: &str) {
    let c = match disposition {
        "served" => &LOGPROBS_SERVED,
        "unsupported" => &LOGPROBS_UNSUPPORTED,
        "withheld" => &LOGPROBS_WITHHELD,
        _ => &LOGPROBS_STREAM,
    };
    c.fetch_add(1, Relaxed);
}

/// Logprobs with speculation (owner 2026-11-02): emitted tokens whose log probabilities came from a speculative round's
/// verify rows (the liveness receipt of the path: zero while no logprobs request ran a speculative round).
static LOGPROBS_SPEC_ROWS: AtomicU64 = AtomicU64::new(0);
pub fn logprobs_spec_rows(n: u64) {
    LOGPROBS_SPEC_ROWS.fetch_add(n, Relaxed);
}

/// PR #31 port: one json_schema request, enforced or served unconstrained.
pub fn schema_request(enforced: bool) {
    let c = if enforced { &SCHEMA_ENFORCED } else { &SCHEMA_UNENFORCED };
    c.fetch_add(1, Relaxed);
}

/// PR #30 / #31 ports: one lane admitted without speculation for log probabilities (`schema` false) or a schema.
pub fn spec_off_lane(schema: bool) {
    let c = if schema { &SPEC_OFF_SCHEMA } else { &SPEC_OFF_LOGPROBS };
    c.fetch_add(1, Relaxed);
}

/// THINKBUDGET1: a request's thinking budget was applied.
pub fn think_budget_applied() {
    THINK_BUDGET_APPLIED.fetch_add(1, Relaxed);
}

/// THINKBUDGET1: a request's thinking budget could not be applied (served without it).
pub fn think_budget_unsupported() {
    THINK_BUDGET_UNSUPPORTED.fetch_add(1, Relaxed);
}

/// THINKBUDGET1: the budget ended one reasoning block.
pub fn think_budget_forced() {
    THINK_BUDGET_FORCED.fetch_add(1, Relaxed);
}

/// TOPKM1: count one request whose top_k the batched GPU samplers capped at 64 (owner_check prints it).
pub fn top_k_capped() {
    TOP_K_CAPPED.fetch_add(1, Relaxed);
}

/// MAXBATCH1: count one over-cap arrival that waits in the queue (owner_check prints it in words).
pub fn over_cap_queued() {
    over_cap_queued_n(1);
}

/// PREHARD1 [ISSUES 104]: the trunk scheduler's queue IS the channel, so it observes arrivals in
/// bulk — a burst's delta counted once per loop sample (the same `over_cap_queued` series).
pub fn over_cap_queued_n(n: u64) {
    OVER_CAP_QUEUED.fetch_add(n, Relaxed);
}

/// MAXBATCH1: count one over-cap arrival refused by the waiting limit (503 server_busy).
pub fn over_cap_rejected() {
    OVER_CAP_REJECTED.fetch_add(1, Relaxed);
}

/// F3: EXL3 prefill-chain GEMMs that a module the wide trellis kernel cannot decode (bit width
/// outside 3..=7 since P7 B5) served through the reconstruct + cuBLASLt arm instead of the arm the option asked
/// for, by the module's bit width (`velogb10_exl3_recon_forced_total`).
static RECON_FORCED: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

/// F3: count one forced-reconstruct GEMM for a module of this bit width (1..=8; others ignored).
pub fn exl3_recon_forced(bits: i32) {
    let b = bits as usize;
    if (1..RECON_FORCED.len()).contains(&b) {
        RECON_FORCED[b].fetch_add(1, Relaxed);
    }
}

/// P7 B6: PSK grids refused by the width guard in `psk_grid` (bit width outside 2..=6): the
/// chain falls back to the plain split-K grid (`exl3_hmma_gemm_ks`), correct but without the
/// persistent-grid scheduling. By bit width (`velogb10_exl3_psk_width_fallback_total`).
static PSK_WIDTH_FALLBACK: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

/// P7 B6: count one PSK width-guard fallback for a module of this bit width (1..=9; others ignored).
pub fn exl3_psk_width_fallback(bits: i32) {
    let b = bits as usize;
    if (1..PSK_WIDTH_FALLBACK.len()).contains(&b) {
        PSK_WIDTH_FALLBACK[b].fetch_add(1, Relaxed);
    }
}

/// PCKG-ALL (2026-10-30): WP21 fused split-K (exl3_hmma_gemm_ks_fx*) refused for a module
/// whose k2 is outside the fused bodies (3..=14; k2 16 = the 4.18bpw 8-bit MTP head):
/// served by the plain 3-launch tail instead (which carries case 16). By k2
/// (`velogb10_exl3_wp21_width_fallback_total`); the boot log prints one WARN, first time.
static WP21_WIDTH_FALLBACK: [AtomicU64; 33] = [const { AtomicU64::new(0) }; 33];

/// PCKG-ALL: count one WP21 width-guard refusal for a module of this k2 (1..=32; else ignored).
pub fn exl3_wp21_width_fallback(k2: i32) {
    let b = k2 as usize;
    if (1..WP21_WIDTH_FALLBACK.len()).contains(&b) {
        WP21_WIDTH_FALLBACK[b].fetch_add(1, Relaxed);
    }
}
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
/// LANERES1: admissions decided by the --lane-reserve policy, by rule: 0 = own (the waiting
/// request's prompt-end snapshot sat in a free lane — admitted first, there), 1 = unreserved (a
/// free lane no reservation protects), 2 = forced (the queue head out-waited
/// --lane-reserve-patience and took a reserved lane; rate-limited by --lane-reserve-swap).
/// All zero when the policy is off.
static LANE_RESERVE_ADMITS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// LANERES3: TP scoped admits whose allowed lanes were no longer free at apply time —
/// requeued by the head (never refused), skipped symmetrically by the mirrors. Non-zero
/// under bursty arrivals with the policy on; a sustained climb means the planner's view
/// is chronically stale and should be looked at.
static LANE_RESERVE_TP_REQUEUED: AtomicU64 = AtomicU64::new(0);
/// Requests the --lane-reserve policy is holding back right now: waiting while free lanes sit
/// reserved for their conversations (0 when the policy is off or nothing waits).
static LANE_RESERVE_WAITING: AtomicU64 = AtomicU64::new(0);
/// One admission through the --lane-reserve policy (`kind`: 0 own, 1 unreserved, 2 forced).
pub fn lane_reserve_admit(kind: usize) {
    if let Some(c) = LANE_RESERVE_ADMITS.get(kind) { c.fetch_add(1, Relaxed); }
}
/// The lane-reserve holdback gauge (requests waiting while free lanes are reserved for others).
pub fn lane_reserve_waiting(n: u64) { LANE_RESERVE_WAITING.store(n, Relaxed); }
/// LANERES3: a TP scoped admit whose allowed set had no free lane at apply time —
/// requeued (head) / skipped (mirror), never refused.
pub fn lane_reserve_tp_requeued() { LANE_RESERVE_TP_REQUEUED.fetch_add(1, Relaxed); }
/// LANERES2 SSD spill tier counters (--kv-spill): writes, restores, recompute fallbacks, bytes
/// stored, and prompt tokens whose re-prefill a restore avoided.
pub static KV_SPILL_WRITES: AtomicU64 = AtomicU64::new(0);
/// PERFFIX1: --prefill-interleave engagement — decode steps run between the chunks of joining prompts, and the
/// chunk boundaries that ran at least one (0 with the flag off).
pub static PREFILL_INTERLEAVE_STEPS: AtomicU64 = AtomicU64::new(0);
pub static PREFILL_INTERLEAVE_POINTS: AtomicU64 = AtomicU64::new(0);
pub fn prefill_interleave(points: u64, steps: u64) {
    PREFILL_INTERLEAVE_POINTS.fetch_add(points, Relaxed);
    PREFILL_INTERLEAVE_STEPS.fetch_add(steps, Relaxed);
}
/// Naive exact speculative sampling (work item A): sampled verify rounds, drafts accepted by the ratio test, and the
/// rows whose residual draw the kernel could not make (no candidate list / empty residual: the target's own draw was committed)
pub static NAIVE_SPEC_SAMPLED_ROUNDS: AtomicU64 = AtomicU64::new(0);
pub static NAIVE_SPEC_SAMPLED_ACCEPTED: AtomicU64 = AtomicU64::new(0);
pub static NAIVE_SPEC_FALLBACK_NOLIST: AtomicU64 = AtomicU64::new(0);
pub static NAIVE_SPEC_FALLBACK_EMPTY: AtomicU64 = AtomicU64::new(0);
/// Naive TP lever 1: the transport ctx addresses (0 = none) whose DEVICE epoch counters are the liveness receipt of
/// each reduce path: every single-stage reduce advances the single ctx's counter by exactly 1, every tree reduce the
/// primary's by log2(world) (graph replays included — the counter lives on the device, so no host count can lag it).
pub static NAIVE_TP_PRIMARY_CTX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static NAIVE_TP_SINGLE_CTX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Lever 1b: the single-stage ctx runs the GPU-validated receive (`--tp-gpu-recv on`).
pub static NAIVE_TP_SINGLE_GPU_RECV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Lever 1c: the single-stage ctx's K1 is the multi-block one (`--naive-tp-k1 multi`, default).
pub static NAIVE_TP_SINGLE_K1_MB: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Naive expert tensor parallelism is live on this boot (`--naive-moe-shard tp`), and the G1 lane count (`--naive-lanes`).
pub static NAIVE_MOE_TP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static NAIVE_LANES: AtomicU64 = AtomicU64::new(0);
/// G1: every lane keeps its own drafter context (`--naive-lane-ctx on`, with lanes > 1 and a drafter).
pub static NAIVE_LANE_CTX: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Naive prefix reuse (`--prefix-cache on`): 1 = on, 2 = asked for but off on this boot (--naive-lanes > 1), 0 = not asked;
/// prompts that resumed / were prefilled whole, and the prompt tokens not prefilled again.
pub static NAIVE_PREFIX_ON: AtomicU64 = AtomicU64::new(0);
pub static NAIVE_PREFIX_HITS: AtomicU64 = AtomicU64::new(0);
pub static NAIVE_PREFIX_MISSES: AtomicU64 = AtomicU64::new(0);
pub static NAIVE_PREFIX_REUSED: AtomicU64 = AtomicU64::new(0);
pub static KV_SPILL_HITS: AtomicU64 = AtomicU64::new(0);
pub static KV_SPILL_MISSES: AtomicU64 = AtomicU64::new(0);
pub static KV_SPILL_BYTES: AtomicU64 = AtomicU64::new(0);
pub static KV_SPILL_SAVED: AtomicU64 = AtomicU64::new(0);
/// LANERES3: TP spill-agreement outcomes — a capture dropped because SOME rank's write
/// failed, and a restore skipped because some rank could not read the snapshot (both
/// re-prefill; the counters prove the cross-rank safety net engaged, not that disks fail).
pub static KV_SPILL_CAPTURE_PEER_DROPS: AtomicU64 = AtomicU64::new(0);
pub static KV_SPILL_RESTORE_PEER_MISSES: AtomicU64 = AtomicU64::new(0);
pub fn kv_spill_write(bytes: u64, tokens: u64) {
    KV_SPILL_WRITES.fetch_add(1, Relaxed);
    KV_SPILL_BYTES.fetch_add(bytes, Relaxed);
}
pub fn kv_spill_hit(tokens: u64) {
    KV_SPILL_HITS.fetch_add(1, Relaxed);
    KV_SPILL_SAVED.fetch_add(tokens, Relaxed);
}
pub fn kv_spill_miss_recompute() { KV_SPILL_MISSES.fetch_add(1, Relaxed); }
// PFXPOOL: the prefix pool (prompt states kept in memory across lanes, --prefix-pool-gb).
pub static PFX_POOL_CAP: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_BYTES: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_ENTRIES: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_CAPTURES: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_CAPTURE_US: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_SKIPS: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_REFUSED: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_EVICTIONS: AtomicU64 = AtomicU64::new(0);
/// PFXPOOL2: entry slabs one box freed on its own because it went under the pool's memory floor.
pub static PFX_POOL_EMERGENCY: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_HITS: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_HIT_TOKENS: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_RESTORE_US: AtomicU64 = AtomicU64::new(0);
pub static PFX_POOL_MISSES: AtomicU64 = AtomicU64::new(0);
/// PFXPOOL: the pool's state after a change (bytes held, entries) and its cap.
pub fn pfx_pool_state(cap: u64, bytes: u64, entries: u64) {
    PFX_POOL_CAP.store(cap, Relaxed);
    PFX_POOL_BYTES.store(bytes, Relaxed);
    PFX_POOL_ENTRIES.store(entries, Relaxed);
}
/// PFXPOOL: a lane's outgoing prompt state was copied into the pool (`us` = the in-round copy time).
pub fn pfx_pool_capture(us: u64) { PFX_POOL_CAPTURES.fetch_add(1, Relaxed); PFX_POOL_CAPTURE_US.fetch_add(us, Relaxed); }
/// PFXPOOL: not captured because the pool already holds these tokens, or the prompt is too short to be worth it.
pub fn pfx_pool_skip() { PFX_POOL_SKIPS.fetch_add(1, Relaxed); }
/// PFXPOOL: not captured: larger than the cap, the memory floor (on any rank), or a copy that failed (on any rank).
pub fn pfx_pool_refused() { PFX_POOL_REFUSED.fetch_add(1, Relaxed); }
pub fn pfx_pool_evict(n: u64) { PFX_POOL_EVICTIONS.fetch_add(n, Relaxed); }
pub fn pfx_pool_emergency(n: u64) { PFX_POOL_EMERGENCY.fetch_add(n, Relaxed); }
/// PFXPOOL: a prompt resumed from a pool entry (`tokens` not prefilled again, `us` = the restore copy time).
pub fn pfx_pool_hit(tokens: u64, us: u64) {
    PFX_POOL_HITS.fetch_add(1, Relaxed);
    PFX_POOL_HIT_TOKENS.fetch_add(tokens, Relaxed);
    PFX_POOL_RESTORE_US.fetch_add(us, Relaxed);
}
/// PFXPOOL: the lane's own snapshot missed and no pool entry (deeper than the lane's WP16 checkpoint) matched.
pub fn pfx_pool_miss() { PFX_POOL_MISSES.fetch_add(1, Relaxed); }
/// LANERES3: a TP capture dropped because at least one rank's local write failed (agreed
/// across ranks; the key is forgotten everywhere — the policy stays replicated-identical).
pub fn kv_spill_capture_peer_drop() { KV_SPILL_CAPTURE_PEER_DROPS.fetch_add(1, Relaxed); }
/// LANERES3: a TP restore skipped because at least one rank could not read the snapshot
/// (agreed across ranks; both re-prefill — the rounds stay identical).
pub fn kv_spill_restore_peer_miss() { KV_SPILL_RESTORE_PEER_MISSES.fetch_add(1, Relaxed); }
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
    let round = SCHED_STEPS.fetch_add(1, Relaxed) + 1;
    SCHED_BUSY.store(1, Relaxed);
    // `--inject-stall` (diagnostic, unset by default): the keep-alive drill sleeps HERE, on the
    // scheduler thread, at the start of the chosen round (one OnceLock read per round otherwise).
    crate::server::stall_drill(round);
}
/// H3 (REL_V0_7_3): one unit of scheduler work COMPLETED (the end of a decode step, or one
/// prefill chunk). One relaxed store; keeps the age gauge honest during long prefills.
pub fn sched_touch() {
    if let Some(s) = START.get() { SCHED_LAST_MS.store(s.elapsed().as_millis() as u64, Relaxed); }
}
/// What the scheduler gauges say right now, for the SSE keep-alive (`server::keepalive_frames`):
/// `busy` = velogb10_scheduler_busy (work in progress since the last idle wait), `work_age` = the
/// time since it last COMPLETED a unit of work (`sched_touch`: the end of a decode round, or one
/// prefill chunk; None before the first one) = velogb10_scheduler_last_step_age_seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SchedState {
    pub busy: bool,
    pub work_age: Option<std::time::Duration>,
}

pub fn sched_state() -> SchedState {
    let last = SCHED_LAST_MS.load(Relaxed);
    let work_age = START.get().filter(|_| last != 0)
        .map(|s| std::time::Duration::from_millis((s.elapsed().as_millis() as u64).saturating_sub(last)));
    SchedState { busy: SCHED_BUSY.load(Relaxed) != 0, work_age }
}
/// The scheduler is about to block waiting for work (between rounds): busy goes 0. On an idle
/// server the age gauge then grows — that is EXPECTED (no rounds are running; nothing is
/// wedged); the alert rule pairs busy==1 with a high age, never busy==0.
pub fn sched_idle() { SCHED_BUSY.store(0, Relaxed); }
/// CUDA graphs currently held (published by the schedulers at boot-precapture end and per
/// lazy capture — the audit's LR-6 visibility).
pub fn set_graph_entries(n: u64) { GRAPH_ENTRIES.store(n, Relaxed); }

/// S-B9-REL-SMALLFIXES 9a(iii): the mixed-regime decode-step graphs (per-row attention regime).
/// They are captured lazily, one per step shape seen, and are not capped; this counts them (they
/// are also part of velogb10_graph_cache_entries). Returns the new count.
static MIXED_STEP_GRAPHS: AtomicU64 = AtomicU64::new(0);
pub fn mixed_step_graph_captured() -> u64 { MIXED_STEP_GRAPHS.fetch_add(1, Relaxed) + 1 }
pub fn mixed_step_graphs() -> u64 { MIXED_STEP_GRAPHS.load(Relaxed) }

/// FP4L (nvidia/Qwen3.8-Flash-Next-FP4-Local): host DISPATCHES of the FP4-Local weight kernels by
/// weight class [mxfp8, f16, bf16 (stored bits), nvfp4-experts] x row class [decode m = 1, verify/packed 2..=16, prefill > 16].
/// A CUDA-graph replay re-runs captured kernels without a host dispatch, so in graphed steady decode these
/// grow at capture only; `velogb10_fp4l_pack` + the trunk's step / round counters are the served-traffic
/// receipts (every linear of an FP4-Local model is an FP4L weight).
static FP4L_DISPATCH: [[AtomicU64; 3]; 5] = [const { [const { AtomicU64::new(0) }; 3] }; 5];
/// 1 once an FP4-Local pack is loaded (the feature marker the boot also prints).
static FP4L_PACK: AtomicU64 = AtomicU64::new(0);
/// FP4L precision levers (PLAN/FP4L_DESIGN.md §3a): the resolved values, set once at load.
static FP4L_PREC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
/// FP4L: host dispatches of the exact-precision TWINS of trunk kernels, by op (same capture caveat as
/// FP4L_DISPATCH): [hc_mixer, hc_norm, hc_mix, embed, shared_gate, ple, router].
pub const FP4L_TWIN_OPS: [&str; 7] = ["hc_mixer", "hc_norm", "hc_mix", "embed", "shared_gate", "ple", "router"];
static FP4L_TWIN: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

fn fp4l_row_class(m: usize) -> usize { if m <= 1 { 0 } else if m <= 16 { 1 } else { 2 } }
/// FP4L: one dense GEMM dispatch of weight class `kind` ("mxfp8" | "f16" | "bf16") at `m` rows.
pub fn fp4l_gemm(kind: &str, m: usize) {
    let k = match kind { "mxfp8" => 0, "bf16" => 2, "trellis" => 4, _ => 1 };
    FP4L_DISPATCH[k][fp4l_row_class(m)].fetch_add(1, Relaxed);
}
/// FP4L: one NVFP4 routed-expert layer dispatch at `m` rows (prefill passes m > 16).
pub fn fp4l_moe(m: usize) { FP4L_DISPATCH[3][fp4l_row_class(m)].fetch_add(1, Relaxed); }
/// FP4L: an FP4-Local pack was loaded.
pub fn fp4l_pack_loaded() { FP4L_PACK.store(1, Relaxed); }
/// FP4L: one dispatch of an exact-precision twin (`op` = one of FP4L_TWIN_OPS).
pub fn fp4l_twin(op: &str) {
    if let Some(i) = FP4L_TWIN_OPS.iter().position(|o| *o == op) { FP4L_TWIN[i].fetch_add(1, Relaxed); }
}
/// FP4L: the resolved precision levers (once, at load).
pub fn fp4l_precision(plain_exact: bool, norms_f32: bool, lm_head: &str, draft_head: &str, prefill_exact: bool) {
    let _ = FP4L_PREC.set(format!("plain=\"{}\",norms=\"{}\",lm_head=\"{lm_head}\",draft_head=\"{draft_head}\",prefill_dense=\"{}\"",
                                  if plain_exact { "exact" } else { "f16" }, if norms_f32 { "f32" } else { "f16" },
                                  if prefill_exact { "exact" } else { "lt" }));
}
/// S-B9-REL-VISIONFIX2 V2b: CUDA graph captures that produced no graph. A capture ends
/// `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` when a CUDA call that is illegal inside stream capture lands in
/// the capture window: from another thread (the vision tower, measured in
/// tests/vision_capture_race_test.rs) or from the capturing thread itself (a NULL-stream allocation, a
/// library-handle destruction). The step then runs eagerly; what happens to the SHAPE depends on the graph
/// family (`capture_next`): the lazily captured ones are captured again at their next use, three are
/// captured once at boot and are not retried. The first loss of each (kind, shape) is logged; every one is
/// counted. S-B9-REL-INTEG C1 added the counting of the sites that used to return silently (BeginCapture
/// refused at the NVFP4 / DFlash2 sites, instantiate failures, the TP spine audit's refusal).
static CAPTURE_INVALIDATED: AtomicU64 = AtomicU64::new(0);
static CAPTURE_FAILED: AtomicU64 = AtomicU64::new(0);
/// The (kind, shape) keys already logged this boot; bounded, so a pathological stream of distinct
/// shapes cannot grow it (the counters keep counting past the bound).
static CAPTURE_LOGGED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
const CAPTURE_LOG_KEYS: usize = 64;
/// The status name `cuStreamEndCapture` returns for an invalidated capture.
pub const CAPTURE_INVALIDATED_STATUS: &str = "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED";

/// Record a capture that produced no graph. `kind` names the graph family ("step", "verify", ...),
/// `shape` its key in words ("width 8"), `status` the `{:?}` of the failing CUDA result (or a short
/// reason for a body error). Returns true when this call logged the line (the first loss of that
/// kind and shape). The caller keeps its own, existing `... capture failed ...` line.
pub fn graph_capture_lost(kind: &str, shape: &str, status: &str) -> bool {
    let invalidated = status == CAPTURE_INVALIDATED_STATUS;
    // (the key and the line are `capture_lost_line`'s: a pure function the tests read)
    if invalidated { CAPTURE_INVALIDATED.fetch_add(1, Relaxed); } else { CAPTURE_FAILED.fetch_add(1, Relaxed); }
    let key = format!("{kind}|{shape}|{}", if invalidated { "invalidated" } else { "failed" });
    let first = {
        let mut seen = CAPTURE_LOGGED.lock().unwrap_or_else(|e| e.into_inner());
        if seen.iter().any(|k| *k == key) || seen.len() >= CAPTURE_LOG_KEYS { false } else { seen.push(key); true }
    };
    if first { crate::rprintln!("{}", capture_lost_line(kind, shape, status)); }
    first
}

/// What happens after a lost capture, by graph family. The lazily captured families (the EXL3 step /
/// verify / packed-verify / draft-pass / draft-chain / reprime graphs, the NVFP4 verify graph, the DFlash2
/// inject graph) run the step eagerly and capture the shape again at its next use. Three are captured ONCE,
/// at boot, and are never retried (S-B9-REL-INTEG C1: the texts used to promise a retry for every family).
fn capture_next(kind: &str) -> &'static str {
    match kind {
        "nvfp4-decode" => "This decode-graph set is captured once, at boot, for every batch size: one loss drops ALL decode \
graphs for this boot and decode runs eagerly (nothing is retried)",
        "nvfp4-decode-sample" => "This graph is captured once per batch size, at boot, and is never retried: that batch size \
takes the non-graph sampling path for this boot",
        "df2-draft-round" => "This graph is captured once, at boot, and is never retried: the DFlash2 draft round runs \
eagerly for this boot",
        _ => "This step ran eagerly and the shape is captured again at its next use",
    }
}

/// The once-per-shape `[graph]` line of a lost capture (pure, so the tests read it).
fn capture_lost_line(kind: &str, shape: &str, status: &str) -> String {
    if status == CAPTURE_INVALIDATED_STATUS {
        format!("[graph] a {kind} capture ({shape}) was INVALIDATED ({status}): a CUDA call that is illegal inside stream \
capture landed in its window (from another thread, e.g. the vision tower, or from the capturing thread itself), so no \
graph was produced. {}. Counted in velogb10_graph_capture_invalidated_total; this line prints once per shape per boot",
                capture_next(kind))
    } else {
        format!("[graph] a {kind} capture ({shape}) produced no graph ({status}). {}. Counted in \
velogb10_graph_capture_failed_total; this line prints once per shape per boot", capture_next(kind))
    }
}
pub fn graph_captures_invalidated() -> u64 { CAPTURE_INVALIDATED.load(Relaxed) }
pub fn graph_captures_failed() -> u64 { CAPTURE_FAILED.load(Relaxed) }

// ─── KV pool occupancy (public issue #24) ────────────────────────────────────────────────
// The KV cache is allocated once at boot: --max-batch lanes, each `window` positions long (the
// served --max-seq-len). These gauges say how much of that a server actually touches.
//
// The unit is a POSITION (one token of a lane's context), not a byte: a position costs the same
// bytes in every lane (`velogb10_kv_allocated_bytes` / `velogb10_kv_positions_capacity` is the per-position
// figure), and a lane's positions are a prefix 0..n of its own window; lanes do not share or borrow
// rows, so the per-lane series are what right-sizes --max-seq-len.
//
// Cost rule (as the rest of this file): the scheduler thread is the ONLY writer and publishes after
// each of its steps / prefill chunks with plain loads and stores (no RMW, no lock, nothing on the
// GPU path); everything else happens at scrape time. An engine that never calls
// `kv_pool_init` emits none of these series.

/// Per-lane series are emitted for at most this many lanes (label cardinality bound); the totals
/// always cover every lane.
pub const KV_LANE_SERIES_MAX: usize = 64;
/// Byte components of the boot allocation (`velogb10_kv_allocated_bytes{kind=...}`).
pub const KV_BYTE_KINDS: [&str; 3] = ["trunk_kv", "indexer_keys", "draft_head"];

struct KvLane {
    used: AtomicU64,
    cached: AtomicU64,
    peak: AtomicU64,
}

struct KvPool {
    window: u64,
    lanes: Box<[KvLane]>,
    used: AtomicU64,
    cached: AtomicU64,
    peak: AtomicU64,
}

static KV_POOL: std::sync::OnceLock<KvPool> = std::sync::OnceLock::new();
static KV_BYTES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static KV_BYTES_SET: AtomicU8 = AtomicU8::new(0);

fn kv_pool_new(lanes: usize, window: usize) -> KvPool {
    KvPool {
        window: window as u64,
        lanes: (0..lanes).map(|_| KvLane { used: AtomicU64::new(0), cached: AtomicU64::new(0), peak: AtomicU64::new(0) }).collect(),
        used: AtomicU64::new(0),
        cached: AtomicU64::new(0),
        peak: AtomicU64::new(0),
    }
}

/// Called once at boot by an engine that reports KV occupancy: `lanes` lanes of `window`
/// positions each (the served context window, after any clamp of --max-seq-len). First call wins.
pub fn kv_pool_init(lanes: usize, window: usize) {
    KV_POOL.get_or_init(|| kv_pool_new(lanes, window));
}

/// The bytes the boot allocation took for one component (`KV_BYTE_KINDS` index), on THIS process
/// (under TP: the head rank's share). Called by the model load; adds up over the layers.
pub fn kv_pool_add_bytes(kind: usize, bytes: u64) {
    if let Some(b) = KV_BYTES.get(kind) {
        b.fetch_add(bytes, Relaxed);
        KV_BYTES_SET.store(1, Relaxed);
    }
}

impl KvPool {
    /// One publication: per lane (positions a running request holds, positions an idle lane keeps
    /// for the prefix cache). Single writer (the scheduler thread). For a concurrent scrape: every
    /// peak is raised BEFORE the value it bounds, and that value is stored with Release and read
    /// with Acquire BEFORE its peak is read (`render`), so `peak >= used` holds in any scrape (two
    /// Relaxed stores to different atomics may become visible out of order on aarch64). Values are
    /// clamped to the lane window, so `used <= total` holds too. The per-lane series of one scrape
    /// can be one publication apart from the aggregate; they agree whenever the scheduler is not
    /// mid-publication.
    fn publish(&self, lanes: impl Iterator<Item = (usize, usize)>) {
        let (mut sum_used, mut sum_cached) = (0u64, 0u64);
        for (i, (used, cached)) in lanes.enumerate() {
            let Some(l) = self.lanes.get(i) else { break };
            let used = (used as u64).min(self.window);
            let cached = (cached as u64).min(self.window);
            sum_used += used;
            sum_cached += cached;
            if used > l.peak.load(Relaxed) { l.peak.store(used, Relaxed); }
            l.used.store(used, Release);
            l.cached.store(cached, Relaxed);
        }
        if sum_used > self.peak.load(Relaxed) { self.peak.store(sum_used, Relaxed); }
        self.used.store(sum_used, Release);
        self.cached.store(sum_cached, Relaxed);
    }

    fn render(&self, o: &mut String, bytes: Option<[u64; 3]>) {
        let n = self.lanes.len() as u64;
        let total = n * self.window;
        o.push_str("# HELP velogb10_kv_positions_capacity Static KV allocation in positions: --max-batch lanes x the served context window (--max-seq-len after any clamp). A position is one token of one lane's context, not a byte; every lane owns its own window and cannot borrow from another.\n# TYPE velogb10_kv_positions_capacity gauge\n");
        o.push_str(&format!("velogb10_kv_positions_capacity {total}\n"));
        o.push_str("# HELP velogb10_kv_positions_used Positions held by requests that are running now, summed over lanes: each lane's prompt (prefilled so far, including a prefix served from the cache) plus the tokens it has generated, to within one token (the newest token is stored when the next one is computed). Updated once per scheduler step and per prefill chunk. 0 when no request runs; speculative rows not yet accepted are not counted.\n# TYPE velogb10_kv_positions_used gauge\n");
        o.push_str(&format!("velogb10_kv_positions_used {}\n", self.used.load(Acquire))); // before the peak (see publish)
        o.push_str("# HELP velogb10_kv_positions_peak Highest velogb10_kv_positions_used since boot.\n# TYPE velogb10_kv_positions_peak gauge\n");
        o.push_str(&format!("velogb10_kv_positions_peak {}\n", self.peak.load(Relaxed)));
        o.push_str("# HELP velogb10_kv_positions_cached Prompt positions that idle lanes keep for the prefix cache, summed over lanes (reused when the next prompt on that lane starts with the same tokens, overwritten otherwise). Not part of positions_used; a lane that is running counts as used only.\n# TYPE velogb10_kv_positions_cached gauge\n");
        o.push_str(&format!("velogb10_kv_positions_cached {}\n", self.cached.load(Relaxed)));
        let shown = self.lanes.len().min(KV_LANE_SERIES_MAX);
        o.push_str("# HELP velogb10_kv_lane_positions_capacity One lane's window in positions (the same for every lane).\n# TYPE velogb10_kv_lane_positions_capacity gauge\n");
        for i in 0..shown { o.push_str(&format!("velogb10_kv_lane_positions_capacity{{lane=\"{i}\"}} {}\n", self.window)); }
        o.push_str("# HELP velogb10_kv_lane_positions_used Positions the request running on this lane holds now (0 = the lane is free). Sums to velogb10_kv_positions_used.\n# TYPE velogb10_kv_lane_positions_used gauge\n");
        for (i, l) in self.lanes.iter().take(shown).enumerate() {
            o.push_str(&format!("velogb10_kv_lane_positions_used{{lane=\"{i}\"}} {}\n", l.used.load(Acquire))); // before its peak
        }
        o.push_str("# HELP velogb10_kv_lane_positions_peak Highest velogb10_kv_lane_positions_used of this lane since boot (the longest context it has held; compare with velogb10_kv_lane_positions_capacity to size --max-seq-len).\n# TYPE velogb10_kv_lane_positions_peak gauge\n");
        for (i, l) in self.lanes.iter().take(shown).enumerate() {
            o.push_str(&format!("velogb10_kv_lane_positions_peak{{lane=\"{i}\"}} {}\n", l.peak.load(Relaxed)));
        }
        o.push_str("# HELP velogb10_kv_lane_positions_cached Prompt positions this lane keeps for the prefix cache while it is free (0 while a request runs on it).\n# TYPE velogb10_kv_lane_positions_cached gauge\n");
        for (i, l) in self.lanes.iter().take(shown).enumerate() {
            o.push_str(&format!("velogb10_kv_lane_positions_cached{{lane=\"{i}\"}} {}\n", l.cached.load(Relaxed)));
        }
        if let Some(b) = bytes {
            o.push_str("# HELP velogb10_kv_allocated_bytes Bytes allocated at boot for per-position state, on this process (with --tp: the head's share only). trunk_kv = the attention K/V cache in the --kv-cache format; indexer_keys = the sparse-attention index planes; draft_head = the speculation head's own cache. Fixed for the life of the process; the per-lane recurrent state and the prefix checkpoints are not included.\n# TYPE velogb10_kv_allocated_bytes gauge\n");
            for (k, name) in KV_BYTE_KINDS.iter().enumerate() {
                o.push_str(&format!("velogb10_kv_allocated_bytes{{kind=\"{name}\"}} {}\n", b[k]));
            }
        }
    }
}

/// The scheduler's publication (see `KvPool::publish`); a no-op on an engine without a pool.
#[inline]
pub fn kv_publish(lanes: impl Iterator<Item = (usize, usize)>) {
    if let Some(p) = KV_POOL.get() { p.publish(lanes); }
}

// ─── tool-call parser shadow counting (S-B9-TOOLPARSE3 T1) ──────────────────────────────
// What the parser's OTHER rules would have returned for the text it parsed. Facts only: which
// reading returns fewer calls and which rule was served; nothing here says which reading is right.
static TC_BLOCKS: AtomicU64 = AtomicU64::new(0);
static TC_COMPARED: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static TC_DIFF: [AtomicU64; 36] = [const { AtomicU64::new(0) }; 36];
static TC_SHADOW_FAILED: AtomicU64 = AtomicU64::new(0);
static TC_SKIPPED: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
/// responses that ended inside a `<tool_call>` block that never closed: [finish stop / length / other] x
/// [the text ends the way a finished call ends: no / yes]
static TC_UNCLOSED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static UNKNOWN_OPTIONS: AtomicU64 = AtomicU64::new(0);
/// F7b: distinct "--"-options this head ignored at startup with one "[opts] unknown option" line
/// each (warn, never refuse; a typo used to disappear silently). Head only; nodes do not warn.
pub fn set_unknown_options(n: u64) { UNKNOWN_OPTIONS.store(n, Relaxed); }
/// Time the thread that parses a response spent in the tool-call parser, per response that held a
/// `<tool_call>` block: [served parse, shadow comparison], a histogram in microsecond buckets, its sum
/// and its worst case. On the EXL3 server that thread is the one that serves every stream, /health and
/// /metrics: this is how long they waited.
const TC_TIME_LE_US: [u64; 11] = [50, 100, 250, 500, 1_000, 2_000, 5_000, 10_000, 25_000, 100_000, 1_000_000];
static TC_TIME_BUCKETS: [[AtomicU64; 12]; 2] = [[const { AtomicU64::new(0) }; 12], [const { AtomicU64::new(0) }; 12]];
static TC_TIME_SUM_US: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static TC_TIME_MAX_US: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// blocks the cost model read by the v0.7.3 rule because they exceeded its work budget: [served, shadow]
static TC_COST_DECLINED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// One response's time in the parser: `part` 0 = the served parse, 1 = the shadow comparison.
pub fn toolcall_time(part: usize, us: u64) {
    let b = TC_TIME_LE_US.iter().position(|&le| us <= le).unwrap_or(TC_TIME_LE_US.len());
    TC_TIME_BUCKETS[part][b].fetch_add(1, Relaxed);
    TC_TIME_SUM_US[part].fetch_add(us, Relaxed);
    TC_TIME_MAX_US[part].fetch_max(us, Relaxed);
}

/// `blocks` closed `<tool_call>` blocks were read by the served parse of one response (counted
/// whether or not the shadow comparison is made, and before it runs).
pub fn toolcall_blocks(blocks: usize) {
    if blocks > 0 { TC_BLOCKS.fetch_add(blocks as u64, Relaxed); }
}

/// A response ended inside a block the model never closed (F8). `outcome`: "returned" (the model
/// ended by itself, finish stop, and function-closed elements came back as calls), "dropped_cut"
/// (the model ended by itself but no complete element was there to return), "dropped_length" (the
/// token budget, the client or the engine cut the response: dropped whole, as v0.7.3).
pub fn toolcall_unclosed_outcome(outcome: &str) {
    let i = match outcome { "returned" => 0, "dropped_cut" => 1, _ => 2 };
    TC_UNCLOSED[i].fetch_add(1, Relaxed);
}

/// The shadow comparison was not made, or was cut short: `size` (text over `tools::SHADOW_MAX_BYTES`),
/// `blocks` (more than `tools::SHADOW_MAX_BLOCKS`), `time` (its wall-clock budget was spent).
pub fn toolcall_shadow_skipped(reason: &str) {
    let i = match reason { "size" => 0, "blocks" => 1, _ => 2 };
    TC_SKIPPED[i].fetch_add(1, Relaxed);
}

/// The cost model read a SERVED block by the v0.7.3 rule because it exceeded its work budget (events,
/// states or work units).
pub fn toolcall_cost_declined() { TC_COST_DECLINED[0].fetch_add(1, Relaxed); }

/// The shadow comparison panicked (a defect in a rule that is not served); the served result was
/// returned unchanged.
pub fn toolcall_shadow_failed() {
    TC_SHADOW_FAILED.fetch_add(1, Relaxed);
    crate::reprintln!("[toolcall-shadow] the comparison panicked; the served result is unaffected");
}

fn tc_pairing_name(p: crate::tools::QuotePairing) -> &'static str {
    match p { crate::tools::QuotePairing::Off => "off", crate::tools::QuotePairing::On => "on", crate::tools::QuotePairing::Cost => "cost" }
}

/// Count one response's shadow comparison and log one content-free line per disagreement.
pub fn toolcall_shadow(sh: &crate::tools::Shadow, served: &crate::tools::Rules) {
    if sh.cost_declined > 0 { TC_COST_DECLINED[1].fetch_add(sh.cost_declined as u64, Relaxed); }
    for (i, n) in sh.compared.iter().enumerate() { if *n > 0 { TC_COMPARED[i].fetch_add(*n as u64, Relaxed); } }
    for (n, d) in sh.diffs.iter().enumerate() {
        TC_DIFF[d.other.idx() * 9 + d.kind.idx() * 3 + d.fewer.idx()].fetch_add(1, Relaxed);
        // one content-free line per disagreement, at most 8 per response (the counters count them all)
        if n == 8 { crate::reprintln!("[toolcall-shadow] {} more disagreements of this response are counted, not logged", sh.diffs.len() - 8); }
        if n >= 8 { continue; }
        crate::reprintln!("[toolcall-shadow] block={} len={} served={}/{} other={} kind={} calls_served={} calls_other={} fewer={}",
                          d.hash, d.len, tc_pairing_name(served.pairing),
                          if served.block == crate::tools::BlockEnd::First { "first" } else { "after-function" },
                          d.other.as_str(), d.kind.as_str(), d.calls_served, d.calls_other, d.fewer.as_str());
    }
}

fn render_toolcall(o: &mut String) {
    use crate::tools::{DiffKind, Fewer, Other};
    let served = crate::tools::Rules::served();
    o.push_str("# HELP velogb10_toolcall_rules_info The tool-call parser rules this server serves (value is always 1): pairing = --tool-quote-pairing (how the end of an argument value is found), block = --tool-block-end (how a <tool_call> block ends), unclosed = --tool-unclosed (what a block that never closed does when the model stopped by itself).\n# TYPE velogb10_toolcall_rules_info gauge\n");
    o.push_str(&format!("velogb10_toolcall_rules_info{{pairing=\"{}\",block=\"{}\",unclosed=\"{}\"}} 1\n", tc_pairing_name(served.pairing),
                        if served.block == crate::tools::BlockEnd::First { "first" } else { "after-function" },
                        if served.unclosed == crate::tools::Unclosed::Complete { "complete" } else { "drop" }));
    o.push_str("# HELP velogb10_toolcall_shadow_failed_total Shadow comparisons that panicked (a defect in a parser rule that is not served; the served result was returned unchanged). Expected 0.\n# TYPE velogb10_toolcall_shadow_failed_total counter\n");
    o.push_str(&format!("velogb10_toolcall_shadow_failed_total {}\n", TC_SHADOW_FAILED.load(Relaxed)));
    o.push_str("# HELP velogb10_toolcall_shadow_skipped_total Responses whose shadow comparison was not made or was cut short, by reason: size (text over 512 KiB), blocks (more than 64 <tool_call> blocks), time (the comparison's 100 ms wall-clock budget was spent between two rules). The served result is unaffected. The default comparison reads two cheap rules (quote pairing and the block rule); the cost model is read only under --tool-shadow-cost on, and has its own work budget (see cost_declined).\n# TYPE velogb10_toolcall_shadow_skipped_total counter\n");
    for (i, r) in ["size", "blocks", "time"].iter().enumerate() {
        o.push_str(&format!("velogb10_toolcall_shadow_skipped_total{{reason=\"{r}\"}} {}\n", TC_SKIPPED[i].load(Relaxed)));
    }
    o.push_str(&format!("# HELP velogb10_toolcall_cost_declined_total Blocks the cost model (--tool-quote-pairing cost, or --tool-shadow-cost on) did not read because they exceeded its work budget (events, states or work units: {} per block and {} per response, counted work and not a clock): they were read by the v0.7.3 rule. role=served: blocks of served responses, where the v0.7.3 reading is what the client got; role=shadow: blocks the comparison skipped. A block counted here is a block on which cost and v0.7.3 agree by construction.\n# TYPE velogb10_toolcall_cost_declined_total counter\n",
                        crate::tools::cost::BLOCK_UNITS, crate::tools::cost::RESPONSE_UNITS));
    o.push_str(&format!("velogb10_toolcall_cost_declined_total{{role=\"served\"}} {}\n", TC_COST_DECLINED[0].load(Relaxed)));
    o.push_str(&format!("velogb10_toolcall_cost_declined_total{{role=\"shadow\"}} {}\n", TC_COST_DECLINED[1].load(Relaxed)));
    o.push_str("# HELP velogb10_toolcall_blocks_total <tool_call> blocks (closed) read by the served parse of finished responses that held at least one. Counted for every such response, whether or not the shadow comparison was made.\n# TYPE velogb10_toolcall_blocks_total counter\n");
    o.push_str(&format!("velogb10_toolcall_blocks_total {}\n", TC_BLOCKS.load(Relaxed)));
    o.push_str("# HELP velogb10_toolcall_parse_seconds Time the thread that parses a response spent in the tool-call parser, per response that held a <tool_call> block. part=served: the parse that is returned; part=shadow: the comparison with the rules that are not served. On the EXL3 server this thread serves every stream, /health and /metrics: it is how long they waited. Histogram, with the worst case below.\n# TYPE velogb10_toolcall_parse_seconds histogram\n");
    for (pi, part) in ["served", "shadow"].iter().enumerate() {
        let mut cum = 0u64;
        for (bi, le) in TC_TIME_LE_US.iter().enumerate() {
            cum += TC_TIME_BUCKETS[pi][bi].load(Relaxed);
            o.push_str(&format!("velogb10_toolcall_parse_seconds_bucket{{part=\"{part}\",le=\"{}\"}} {cum}\n", *le as f64 / 1e6));
        }
        cum += TC_TIME_BUCKETS[pi][TC_TIME_LE_US.len()].load(Relaxed);
        o.push_str(&format!("velogb10_toolcall_parse_seconds_bucket{{part=\"{part}\",le=\"+Inf\"}} {cum}\n"));
        o.push_str(&format!("velogb10_toolcall_parse_seconds_sum{{part=\"{part}\"}} {}\n", TC_TIME_SUM_US[pi].load(Relaxed) as f64 / 1e6));
        o.push_str(&format!("velogb10_toolcall_parse_seconds_count{{part=\"{part}\"}} {cum}\n"));
    }
    o.push_str("# HELP velogb10_toolcall_parse_seconds_max The longest single response in the tool-call parser since the process started, by part (see velogb10_toolcall_parse_seconds).\n# TYPE velogb10_toolcall_parse_seconds_max gauge\n");
    for (pi, part) in ["served", "shadow"].iter().enumerate() {
        o.push_str(&format!("velogb10_toolcall_parse_seconds_max{{part=\"{part}\"}} {}\n", TC_TIME_MAX_US[pi].load(Relaxed) as f64 / 1e6));
    }
    o.push_str("# HELP velogb10_toolcall_unclosed_total Responses that ended inside a <tool_call> block the model never closed. outcome = returned: the model ended by itself (finish stop) and function elements that were closed inside the wrapper came back as calls (--tool-unclosed complete); dropped_cut: the model ended by itself but nothing complete was there to return (cut inside an argument, or a JSON body); dropped_length: the token budget, the client or the engine cut the response (dropped whole, as v0.7.3, and as EVERY finish under --tool-unclosed drop).\n# TYPE velogb10_toolcall_unclosed_total counter\n");
    for (i, name) in [(0usize, "returned"), (1, "dropped_cut"), (2, "dropped_length")].iter() {
        o.push_str(&format!("velogb10_toolcall_unclosed_total{{outcome=\"{name}\"}} {}\n", TC_UNCLOSED[*i].load(Relaxed)));
    }
    o.push_str("# HELP velogb10_toolcall_shadow_compared_total Blocks (other=\"block_end\": responses) that were also read under the rule named by other, which is NOT the served one. other=off / on / cost are the --tool-quote-pairing settings, other=block_end is the --tool-block-end setting that is not served. The comparison changes nothing that is returned.\n# TYPE velogb10_toolcall_shadow_compared_total counter\n");
    for other in Other::ALL {
        o.push_str(&format!("velogb10_toolcall_shadow_compared_total{{other=\"{}\"}} {}\n", other.as_str(), TC_COMPARED[other.idx()].load(Relaxed)));
    }
    o.push_str("# HELP velogb10_toolcall_parse_disagreement_total Blocks (other=\"block_end\": responses) where the rule named by other, read the same text differently from the rule that was served. kind=call_count: a different number of calls; kind=arg_boundary: the same number of calls, an argument or name differs; kind=content: the same calls, the prose differs. fewer says which of the two readings returned FEWER calls (served / other / same). It is a fact about the two readings; the server cannot know which one is right, and a nonzero count means only that the text was ambiguous under the rules.\n# TYPE velogb10_toolcall_parse_disagreement_total counter\n");
    for other in Other::ALL {
        for kind in DiffKind::ALL {
            for fewer in Fewer::ALL {
                o.push_str(&format!("velogb10_toolcall_parse_disagreement_total{{other=\"{}\",kind=\"{}\",fewer=\"{}\"}} {}\n",
                                    other.as_str(), kind.as_str(), fewer.as_str(),
                                    TC_DIFF[other.idx() * 9 + kind.idx() * 3 + fewer.idx()].load(Relaxed)));
            }
        }
    }
}

/// Admitted-but-unfinished requests (running + waiting) — exact, from the Req guard.
pub fn inflight() -> u64 { IN_FLIGHT.load(Relaxed) }
/// S-B9-CF-PACK4 (`--lane-pack`): scheduler rounds by arm and the lane rounds that stayed out of a
/// pack, by reason. The arms are counted with the flag on or off (off: `packed` stays 0).
pub const LANE_PACK_ARMS: [&str; 3] = ["packed", "serial", "shared_plain"];
static LANE_PACK_ROUNDS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static LANE_PACK_SKIP: [AtomicU64; crate::lane_pack::Skip::ALL.len()] =
    [const { AtomicU64::new(0) }; crate::lane_pack::Skip::ALL.len()];
/// S-B9-CF-PACK4S: LANE rounds served by packed verifies, by the lane's kind (two per packed
/// round): real-q sampled, match-rule sampled, greedy.
pub const LANE_PACK_KINDS: [&str; 3] = ["sampled_real_q", "sampled_match", "greedy"];
static LANE_PACK_LANES: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
pub fn lane_pack_lane_round(kind: usize) { LANE_PACK_LANES[kind].fetch_add(1, Relaxed); }
/// PACK4-REVIEW A3: 1 once a packed-round error turned the arm off for the rest of the boot.
static LANE_PACK_DISABLED: AtomicU64 = AtomicU64::new(0);
pub fn lane_pack_disabled() { LANE_PACK_DISABLED.store(1, Relaxed); }
pub fn lane_pack_is_disabled() -> bool { LANE_PACK_DISABLED.load(Relaxed) != 0 }
/// One round of arm `arm` (index into LANE_PACK_ARMS) ran.
pub fn lane_pack_round(arm: usize) { LANE_PACK_ROUNDS[arm].fetch_add(1, Relaxed); }
pub fn lane_pack_rounds(arm: usize) -> u64 { LANE_PACK_ROUNDS[arm].load(Relaxed) }
/// One lane round ran outside a pack for reason index `i` (lane_pack::Skip::index).
pub fn lane_pack_skip(i: usize) { LANE_PACK_SKIP[i].fetch_add(1, Relaxed); }
pub fn lane_pack_skips(i: usize) -> u64 { LANE_PACK_SKIP[i].load(Relaxed) }
/// Lane capacity as set at boot (`set_max_batch`).
pub fn max_batch() -> u64 { MAX_BATCH.load(Relaxed) }
/// S-C4PRIMEA: shared-step primes served by the BATCHED single-launch path
/// (xq_gather_tap_rows + one re-prime pass + xq_attn_kv_prefill_pairs), and
/// the lanes those steps carried. 0 with --exl3-prime-batch off.
static PRIME_BATCH: AtomicU64 = AtomicU64::new(0);
static PRIME_BATCH_LANES: AtomicU64 = AtomicU64::new(0);
pub fn prime_batch_round(lanes: u64) {
    PRIME_BATCH.fetch_add(1, Relaxed);
    PRIME_BATCH_LANES.fetch_add(lanes, Relaxed);
}
pub fn prime_batch_rounds() -> u64 { PRIME_BATCH.load(Relaxed) }
pub fn prime_batch_lanes() -> u64 { PRIME_BATCH_LANES.load(Relaxed) }

/// S-C4XOVER: the spec-lanes auto policy (--spec-lanes-max auto, the default under --mtp auto)
/// made legible: the two arms' estimated AGGREGATE tok/s at the last measured decision, the
/// learned shared-step scale, the busy-lane count at that decision, how many decisions compared
/// real estimates, and how often the pick FLIPPED (thrash / stuck detection). Observability only.
pub const SPEC_LANES_TO: [&str; 2] = ["shared", "serial"];
static SPEC_LANES_SWITCHES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static SPEC_LANES_DECISIONS: AtomicU64 = AtomicU64::new(0);
static SPEC_LANES_EST_SERIAL: AtomicU64 = AtomicU64::new(0); // f64 bits: est serial aggregate tok/s
static SPEC_LANES_EST_SHARED: AtomicU64 = AtomicU64::new(0); // f64 bits: est shared aggregate tok/s
static SPEC_LANES_SCALE_MILLI: AtomicU64 = AtomicU64::new(0); // learned shared_scale x1000
static SPEC_LANES_BUSY: AtomicU64 = AtomicU64::new(0);       // busy lanes at the last decision
static SPEC_ARM_MS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];   // S-C4XOVER: cumulative ms each arm ran
static SPEC_ARM_TOK: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];  // ... and the tokens it delivered
pub fn spec_lanes_switch(to: usize) { SPEC_LANES_SWITCHES[to].fetch_add(1, Relaxed); }
pub fn spec_arm_charge(arm: usize, ms: f64, tok: f64) {
    SPEC_ARM_MS[arm].fetch_add(ms as u64, Relaxed);
    SPEC_ARM_TOK[arm].fetch_add(tok as u64, Relaxed);
}
pub fn spec_lanes_decision(serial_tps: f64, shared_tps: f64, scale: f64, busy: usize) {
    SPEC_LANES_EST_SERIAL.store(serial_tps.to_bits(), Relaxed);
    SPEC_LANES_EST_SHARED.store(shared_tps.to_bits(), Relaxed);
    SPEC_LANES_SCALE_MILLI.store((scale * 1000.0) as u64, Relaxed);
    SPEC_LANES_BUSY.store(busy as u64, Relaxed);
    SPEC_LANES_DECISIONS.fetch_add(1, Relaxed);
}
static SPEC_LANES_MEASURED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)]; // f64 bits: measured tok/s
static SPEC_LANES_PROBES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// S-C4XOVER: the arm's MEASURED delivered aggregate tok/s (EWMA over its last windows) and
/// the arm-rate probe counter. arm: 0 = shared plain step, 1 = serial speculation.
pub fn spec_lanes_measured(arm: usize, tps: f64) { SPEC_LANES_MEASURED[arm].store(tps.to_bits(), Relaxed); }
pub fn spec_lanes_probe(arm: usize) { SPEC_LANES_PROBES[arm].fetch_add(1, Relaxed); }

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
    o.push_str("# HELP velogb10_df2_rounds_total DFlash2 draft rounds served on this boot (the --draft-dir speculator; nvfp4 trunk-path arm; scripts/owner_check.py prints it).\n# TYPE velogb10_df2_rounds_total counter\n");
    o.push_str(&format!("velogb10_df2_rounds_total {}\n", DF2_ROUNDS.load(Relaxed)));
    o.push_str("# HELP velogb10_tp_proxy_parks_total CPUIDLE: times a TP proxy on this process left its spin for the idle tier (a WFE wait after --proxy-idle-us with no epoch owed). 0 on a TP server that has sat idle = the tier never ran; it should grow about once per quiet gap (between requests), not once per step.\n# TYPE velogb10_tp_proxy_parks_total counter\n");
    o.push_str(&format!("velogb10_tp_proxy_parks_total {}\n", crate::net::proxy_parks()));
    o.push_str("# HELP velogb10_tp_proxy_parked_seconds_total CPUIDLE: seconds TP proxies on this process spent in the idle tier instead of spinning a core.\n# TYPE velogb10_tp_proxy_parked_seconds_total counter\n");
    o.push_str(&format!("velogb10_tp_proxy_parked_seconds_total {:.3}\n", crate::net::proxy_parked_secs()));
    {
        let (w, slept, early) = crate::host_wait::counters();
        o.push_str("# HELP velogb10_gpu_wait_hybrid_waits_total CPUIDLE --gpu-wait hybrid: GPU waits that slept part of the wait instead of spinning (0 with requests served = the hybrid wait never ran: --gpu-wait spin, or no wait was long enough).\n# TYPE velogb10_gpu_wait_hybrid_waits_total counter\n");
        o.push_str(&format!("velogb10_gpu_wait_hybrid_waits_total {w}\n"));
        o.push_str("# HELP velogb10_gpu_wait_slept_seconds_total CPUIDLE: seconds the waiting thread slept instead of spinning a core.\n# TYPE velogb10_gpu_wait_slept_seconds_total counter\n");
        o.push_str(&format!("velogb10_gpu_wait_slept_seconds_total {slept:.3}\n"));
        o.push_str("# HELP velogb10_gpu_wait_early_total CPUIDLE: hybrid waits whose GPU finished before the spin window opened (each one woke up to one 200 us slice late; should stay a small fraction of the waits).\n# TYPE velogb10_gpu_wait_early_total counter\n");
        o.push_str(&format!("velogb10_gpu_wait_early_total {early}\n"));
    }
    o.push_str("# HELP velogb10_exl3_ple_width_accepted_total PLE n-gram rows whose group width K was outside {5,6} but inside 1..=8, loaded and served (B9 width closure; 0 = never ran).\n# TYPE velogb10_exl3_ple_width_accepted_total counter\n");
    o.push_str(&format!("velogb10_exl3_ple_width_accepted_total {}\n", EXL3_PLE_WIDTH_ACCEPTED.load(Relaxed)));
    o.push_str("# HELP velogb10_ple_warm_ahead_chunks_total prefill chunks whose next chunk's n-gram (PLE) rows were read ahead into the page cache (SSD-resident table only; 0 = never ran).\n# TYPE velogb10_ple_warm_ahead_chunks_total counter\n");
    o.push_str(&format!("velogb10_ple_warm_ahead_chunks_total {}\n", PLE_WARM_AHEAD_CHUNKS.load(Relaxed)));
    o.push_str("# HELP velogb10_ple_warm_ahead_rows_total n-gram (PLE) rows covered by the read-ahead.\n# TYPE velogb10_ple_warm_ahead_rows_total counter\n");
    o.push_str(&format!("velogb10_ple_warm_ahead_rows_total {}\n", PLE_WARM_AHEAD_ROWS.load(Relaxed)));
    o.push_str("# HELP velogb10_ple_warm_ahead_reads_total deduplicated 4 KB page reads the read-ahead issued.\n# TYPE velogb10_ple_warm_ahead_reads_total counter\n");
    o.push_str(&format!("velogb10_ple_warm_ahead_reads_total {}\n", PLE_WARM_AHEAD_READS.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_w4moe_width_served_total persistent grouped-expert (--w4moe) launches at a non-3 expert width (B10 width closure; 0 = never ran).\n# TYPE velogb10_exl3_w4moe_width_served_total counter\n");
    o.push_str(&format!("velogb10_exl3_w4moe_width_served_total {}\n", EXL3_W4MOE_WIDTH_SERVED.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_devroute_fallback_total #26 TP-B: prefill rounds that asked for the host-routed MoE arm under TP and were served by the device-routed EP arm instead (one clear line at boot; this counter moves when it ran).\n# TYPE velogb10_exl3_devroute_fallback_total counter\n");
    o.push_str(&format!("velogb10_exl3_devroute_fallback_total {}\n", EXL3_DEVROUTE_FALLBACK.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_pf_kernel_width_fallback_total #26 Fix 3: prefill layers whose width the pipelined expert kernel (xq_moe_pf*: K 2..7 and 1.5/2.5/3.5) cannot decode, served by the wide tiles kernel instead (one clear line per boot; 0 = never ran).\n# TYPE velogb10_exl3_pf_kernel_width_fallback_total counter\n");
    o.push_str(&format!("velogb10_exl3_pf_kernel_width_fallback_total {}\n", EXL3_PF_KERNEL_WIDTH_FALLBACK.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_kv_cache_fallback_total EXL3 boots that were asked for --kv-cache bf16 (the EXL3 engine has no bf16 KV store) and served f16 instead (the same 2 bytes per value; one clear line at boot). 0 = never happened.\n# TYPE velogb10_exl3_kv_cache_fallback_total counter\n");
    o.push_str(&format!("velogb10_exl3_kv_cache_fallback_total {}\n", EXL3_KV_CACHE_FALLBACK.load(Relaxed)));

    o.push_str("# HELP velogb10_exl3_mtp_skipped_total #26 Fix 4: boots whose draft head exceeded the served width set and were served WITHOUT speculation\n# TYPE velogb10_exl3_mtp_skipped_total counter\n");
    o.push_str(&format!("velogb10_exl3_mtp_skipped_total {}\n", EXL3_MTP_SKIPPED.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_ngram_embed_rows_total Token-embedding rows decoded from ngram trellis rings at load (exl3 1.6.3 packs; 0 = the plain embed path ran).\n# TYPE velogb10_exl3_ngram_embed_rows_total counter\n");
    o.push_str(&format!("velogb10_exl3_ngram_embed_rows_total {}\n", EXL3_NGRAM_EMBED_ROWS.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_frac_pf_bridge_rows_total Prefill rows served by the fractional-k2 decode-moe bridge (LEG-B; no pf kernel decodes k2=3/5/7).\n# TYPE velogb10_exl3_frac_pf_bridge_rows_total counter\n");
    o.push_str(&format!("velogb10_exl3_frac_pf_bridge_rows_total {}\n", EXL3_FRAC_PF_BRIDGE_ROWS.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_frac_expert_rounds_total K35: MoE layer rounds whose routed experts ran the fractional-width grouped body (K = 1.5/2.5/3.5, label k2 = 3/5/7; decode, verify, MTP draft and the prefill bridge; eager calls plus graph replays — a capture executes nothing and is not counted). Absent = that width never served.\n# TYPE velogb10_exl3_frac_expert_rounds_total counter\n");
    for k2 in 0..8usize {
        let v = EXL3_FRAC_EXPERT_ROUNDS[k2].load(Relaxed);
        if v > 0 { o.push_str(&format!("velogb10_exl3_frac_expert_rounds_total{{k2=\"{k2}\"}} {v}\n")); }
    }
    o.push_str("# HELP velogb10_exl3_k78_expert_rounds_total K8EXP: MoE layer rounds whose routed experts are K = 7 or K = 8 (label k2 = 14 / 16; decode, verify, MTP draft; eager calls plus graph replays). K = 7 runs the integer expert family (a1b7 / WP20 b7); K = 8 runs the plain grouped chain on its own entry (xq_gemm_grouped_b8). Absent = that width never served.\n# TYPE velogb10_exl3_k78_expert_rounds_total counter\n");
    for k2 in [14usize, 16] {
        let v = EXL3_FRAC_EXPERT_ROUNDS[k2].load(Relaxed);
        if v > 0 { o.push_str(&format!("velogb10_exl3_k78_expert_rounds_total{{k2=\"{k2}\"}} {v}\n")); }
    }
    o.push_str("# HELP velogb10_exl3_frac_pf_wide_rows_total K35: prefill expert rows (token x top-k) served by the fractional-width wide-tiles body (exl3_hmma_gemm_wide_tiles, EXL3_FRAC_BITS case), by k2. Absent = no fractional prefill ran there (the bridge counter above covers the host-routed arm).\n# TYPE velogb10_exl3_frac_pf_wide_rows_total counter\n");
    for k2 in 0..8usize {
        let v = EXL3_FRAC_PF_WIDE_ROWS[k2].load(Relaxed);
        if v > 0 { o.push_str(&format!("velogb10_exl3_frac_pf_wide_rows_total{{k2=\"{k2}\"}} {v}\n")); }
    }
    o.push_str("# HELP velogb10_exl3_frac_pf_pipelined_rows_total PFHALF: prefill expert rows (token x top-k) served by the pipelined kernel's fractional-width HALF body (xq_moe_pf, EXL3_FRAC_BITS case), by k2. Absent = no fractional layer prefilled on the pipelined kernel (it took the wide-tiles body above, or never ran).\n# TYPE velogb10_exl3_frac_pf_pipelined_rows_total counter\n");
    for k2 in 0..8 {
        let v = EXL3_FRAC_PF_PIPE_ROWS[k2].load(Relaxed);
        if v > 0 { o.push_str(&format!("velogb10_exl3_frac_pf_pipelined_rows_total{{k2=\"{k2}\"}} {v}\n")); }
    }
    o.push_str("# HELP velogb10_df2_mask_row_override 1 if the draft's mask row measured DAMAGED at load and the artifact's BF16 copy override is engaged (0 otherwise).\n# TYPE velogb10_df2_mask_row_override gauge\n");
    o.push_str(&format!("velogb10_df2_mask_row_override {}\n", DF2_MASK_OVERRIDE.load(Relaxed)));
    o.push_str("# HELP velogb10_draft_vocab_rows Token ids the MTP drafter may propose (--draft-vocab, FR-Spec); 0 = no reduced draft head, the drafter reads the full vocabulary.\n# TYPE velogb10_draft_vocab_rows gauge\n");
    o.push_str(&format!("velogb10_draft_vocab_rows {}\n", DRAFT_VOCAB_ROWS.load(Relaxed)));
    o.push_str("# HELP velogb10_draft_vocab_full_rows The model's full vocabulary (the denominator of velogb10_draft_vocab_rows; 0 when no reduced draft head was built).\n# TYPE velogb10_draft_vocab_full_rows gauge\n");
    o.push_str(&format!("velogb10_draft_vocab_full_rows {}\n", DRAFT_VOCAB_FULL.load(Relaxed)));
    o.push_str("# HELP velogb10_draft_vocab_steps_total MTP draft tokens picked from the reduced draft head (--draft-vocab engaged).\n# TYPE velogb10_draft_vocab_steps_total counter\n");
    o.push_str(&format!("velogb10_draft_vocab_steps_total {}\n", DRAFT_VOCAB_STEPS.load(Relaxed)));
    let inflight = IN_FLIGHT.load(Relaxed);
    let cap = MAX_BATCH.load(Relaxed);
    let running = if cap > 0 { inflight.min(cap) } else { inflight };
    o.push_str(&format!("# HELP velogb10_max_batch Lane capacity (--max-batch).\n# TYPE velogb10_max_batch gauge\nvelogb10_max_batch {cap}\n"));
    let req = MAX_BATCH_REQUESTED.load(Relaxed);
    o.push_str("# HELP velogb10_max_batch_requested The --max-batch the command line asked for, before the 16-lane serving cap (EXL3 and NVFP4 trunk servers; PREHARD1).\n# TYPE velogb10_max_batch_requested gauge\n");
    o.push_str(&format!("velogb10_max_batch_requested {req}\n"));
    let dsk = EXL3_DENSE_SPLITK_CALLS.load(Relaxed);
    o.push_str("# HELP velogb10_exl3_dense_splitk_calls_total DSPLITK: EXL3 trunk dense-attention calls served split-K over the context (--exl3-dense-splitk on; eager calls and graph captures, replays not counted). Absent = never.\n# TYPE velogb10_exl3_dense_splitk_calls_total counter\n");
    if dsk > 0 { o.push_str(&format!("velogb10_exl3_dense_splitk_calls_total {dsk}\n")); }
    let (ca, cs) = (EXL3_CTX_ASKED.load(Relaxed), EXL3_CTX_SERVED.load(Relaxed));
    o.push_str("# HELP velogb10_exl3_ctx_clamped 1 when this EXL3 TP=1 boot could not fit even one lane at the asked --max-seq-len and booted one lane at the longest context that fits (CTXCLAMP, owner 2026-11-07); labels: asked, served. Absent = no context clamp.\n# TYPE velogb10_exl3_ctx_clamped gauge\n");
    if ca > 0 { o.push_str(&format!("velogb10_exl3_ctx_clamped{{asked=\"{ca}\",served=\"{cs}\"}} 1\n")); }
    let pft = EXL3_PF_TC_ROWS.load(Relaxed);
    o.push_str("# HELP velogb10_exl3_pf_tc_rows_total PFTC: EXL3 prefill rows (per attention layer) whose dense attention ran on the tensor-core kernels (--exl3-dense-pf-tc on). Absent = never.\n# TYPE velogb10_exl3_pf_tc_rows_total counter\n");
    if pft > 0 { o.push_str(&format!("velogb10_exl3_pf_tc_rows_total {pft}\n")); }
    o.push_str(&format!("# HELP velogb10_requests_running Requests holding a lane.\n# TYPE velogb10_requests_running gauge\nvelogb10_requests_running {running}\n"));
    o.push_str(&format!("# HELP velogb10_requests_waiting Requests queued for a lane.\n# TYPE velogb10_requests_waiting gauge\nvelogb10_requests_waiting {}\n",
                        inflight - running));
    // MAXBATCH1: the F2 cap's engagement counter — arrivals that found every lane busy. Queued ran
    // when a lane freed; rejected hit the LR-4 waiting-limit 503. scripts/owner_check.py prints it.
    o.push_str("# HELP velogb10_over_cap_arrivals_total Requests that arrived beyond what the lane cap can run at once (a busy server, or a burst the queue must absorb): queued = waited for a lane, then served; rejected = waiting limit, 503 server_busy.\n# TYPE velogb10_over_cap_arrivals_total counter\n");
    o.push_str(&format!("velogb10_over_cap_arrivals_total{{disposition=\"queued\"}} {}\n", OVER_CAP_QUEUED.load(Relaxed)));
    o.push_str(&format!("velogb10_over_cap_arrivals_total{{disposition=\"rejected\"}} {}\n", OVER_CAP_REJECTED.load(Relaxed)));
    // TOPKM1: the batched GPU samplers' top_k cap (0 / negative / > 64 -> 64). scripts/owner_check.py prints it.
    o.push_str("# HELP velogb10_top_k_capped_total Sampled requests that asked for no top-k truncation (top_k 0 or negative) or more than 64 candidates on a model family whose GPU sampler keeps at most 64; served with top_k 64 (the EXL3 server and --cpu-sample apply no such limit).\n# TYPE velogb10_top_k_capped_total counter\n");
    o.push_str(&format!("velogb10_top_k_capped_total {}\n", TOP_K_CAPPED.load(Relaxed)));
    // THINKBUDGET1: the per-request thinking budget (vLLM's thinking_token_budget). scripts/owner_check.py prints it.
    o.push_str("# HELP velogb10_thinking_budget_requests_total Requests that sent thinking_token_budget: applied = the budget was enforced; unsupported = this model or serving path cannot enforce it, so the request ran without it.\n# TYPE velogb10_thinking_budget_requests_total counter\n");
    o.push_str(&format!("velogb10_thinking_budget_requests_total{{disposition=\"applied\"}} {}\n", THINK_BUDGET_APPLIED.load(Relaxed)));
    o.push_str(&format!("velogb10_thinking_budget_requests_total{{disposition=\"unsupported\"}} {}\n", THINK_BUDGET_UNSUPPORTED.load(Relaxed)));
    o.push_str("# HELP velogb10_thinking_budget_forced_total Reasoning blocks the thinking budget ended (the end-of-thinking marker was emitted by the server at the budget).\n# TYPE velogb10_thinking_budget_forced_total counter\n");
    o.push_str(&format!("velogb10_thinking_budget_forced_total {}\n", THINK_BUDGET_FORCED.load(Relaxed)));
    // PR #30 / #31 ports: log probabilities, json_schema enforcement, and the lanes they ran without speculation.
    o.push_str("# HELP velogb10_logprobs_requests_total Requests that asked for logprobs: served = returned; unsupported = this serving engine does not compute them; withheld = --scrub-tool-secrets is on; stream = stream=true, not wired yet. All but served get logprobs null.\n# TYPE velogb10_logprobs_requests_total counter\n");
    for (d, c) in [("served", &LOGPROBS_SERVED), ("unsupported", &LOGPROBS_UNSUPPORTED), ("withheld", &LOGPROBS_WITHHELD), ("stream", &LOGPROBS_STREAM)] {
        o.push_str(&format!("velogb10_logprobs_requests_total{{disposition=\"{d}\"}} {}\n", c.load(Relaxed)));
    }
    o.push_str("# HELP velogb10_logprobs_spec_rows_total Generated tokens whose log probabilities were read from a speculative round's verify rows (logprobs keep speculation).\n# TYPE velogb10_logprobs_spec_rows_total counter\n");
    o.push_str(&format!("velogb10_logprobs_spec_rows_total {}\n", LOGPROBS_SPEC_ROWS.load(Relaxed)));
    o.push_str("# HELP velogb10_json_schema_requests_total response_format json_schema requests: enforced = token mask armed (x-json-schema-enforced: full); unenforced = served unconstrained (x-json-schema-enforced: none).\n# TYPE velogb10_json_schema_requests_total counter\n");
    o.push_str(&format!("velogb10_json_schema_requests_total{{disposition=\"enforced\"}} {}\n", SCHEMA_ENFORCED.load(Relaxed)));
    o.push_str(&format!("velogb10_json_schema_requests_total{{disposition=\"unenforced\"}} {}\n", SCHEMA_UNENFORCED.load(Relaxed)));
    o.push_str("# HELP velogb10_spec_off_lanes_total EXL3 lanes that decoded without speculation because the request asked for logprobs or a schema mask.\n# TYPE velogb10_spec_off_lanes_total counter\n");
    o.push_str(&format!("velogb10_spec_off_lanes_total{{why=\"logprobs\"}} {}\n", SPEC_OFF_LOGPROBS.load(Relaxed)));
    o.push_str(&format!("velogb10_spec_off_lanes_total{{why=\"schema\"}} {}\n", SPEC_OFF_SCHEMA.load(Relaxed)));
    // CLIPRUNE1: rendered since it was set (F7b) but never shown — scripts/owner_check.py
    // already reads it and tolerates absence; keep that guard working.
    o.push_str("# HELP velogb10_unknown_options Unknown '--' options seen at startup (warned and ignored; F7b).\n# TYPE velogb10_unknown_options gauge\n");
    o.push_str(&format!("velogb10_unknown_options {}\n", UNKNOWN_OPTIONS.load(Relaxed)));
    o.push_str("# HELP velogb10_requests_total Finished requests by finish reason.\n# TYPE velogb10_requests_total counter\n");
    for (i, f) in FINISHES.iter().enumerate() {
        o.push_str(&format!("velogb10_requests_total{{finish=\"{f}\"}} {}\n", REQ_BY_FINISH[i].load(Relaxed)));
    }
    o.push_str("# HELP velogb10_scrub_hits_total Spans redacted by --scrub-tool-secrets since boot, by layer (advisory liveness; a span split by the pending cap can count twice). layer=provenance: matches of strings extracted from this conversation's tool results; layer=pattern: key-shaped text. Zero forever when the flag is off (the default).\n# TYPE velogb10_scrub_hits_total counter\n");
    o.push_str(&format!("velogb10_scrub_hits_total{{layer=\"provenance\"}} {}\nvelogb10_scrub_hits_total{{layer=\"pattern\"}} {}\n",
                       SCRUB_PROV_HITS.load(Relaxed), SCRUB_PAT_HITS.load(Relaxed)));
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
    o.push_str("# HELP velogb10_scheduler_steps_total Scheduler rounds STARTED since boot (a round in progress, or one that never completes, is counted; so is a pass whose lanes were all cancelled before it ran).\n# TYPE velogb10_scheduler_steps_total counter\n");
    o.push_str(&format!("velogb10_scheduler_steps_total {}\n", SCHED_STEPS.load(Relaxed)));
    o.push_str("# HELP velogb10_scheduler_busy 1 from the start of a decode round until the scheduler next blocks idle-waiting for a request, else 0. Only a decode round sets it: a server that was idle still reads 0 while it admits and prefills its first request.\n# TYPE velogb10_scheduler_busy gauge\n");
    o.push_str(&format!("velogb10_scheduler_busy {}\n", SCHED_BUSY.load(Relaxed)));
    if let Some(s0) = START.get() {
        let now_ms = s0.elapsed().as_millis() as u64;
        let last = SCHED_LAST_MS.load(Relaxed);
        let age = if last == 0 { now_ms } else { now_ms.saturating_sub(last) }; // H9: no wrap
        o.push_str("# HELP velogb10_scheduler_last_step_age_seconds Seconds since the scheduler last COMPLETED a round or a prefill chunk (uptime if none yet; grows while idle: alert on a high value only together with scheduler_busy == 1, or with requests_running > 0 for a stall during the first request's prefill).\n# TYPE velogb10_scheduler_last_step_age_seconds gauge\n");
        o.push_str(&format!("velogb10_scheduler_last_step_age_seconds {:.3}\n", age as f64 / 1000.0));
    }
    o.push_str("# HELP velogb10_graph_cache_entries CUDA graphs the engine holds now (captured at boot and, lazily, the first time a new step shape runs).\n# TYPE velogb10_graph_cache_entries gauge\n");
    o.push_str(&format!("velogb10_graph_cache_entries {}\n", GRAPH_ENTRIES.load(Relaxed)));
    o.push_str("# HELP velogb10_graph_cache_mixed_step_entries Mixed-regime decode-step CUDA graphs the engine holds now (per-row attention regime: one is captured the first time a step shape is seen; they are never dropped and not capped, so the value only grows; they are also counted in velogb10_graph_cache_entries).\n# TYPE velogb10_graph_cache_mixed_step_entries gauge\n");
    o.push_str(&format!("velogb10_graph_cache_mixed_step_entries {}\n", MIXED_STEP_GRAPHS.load(Relaxed)));
    o.push_str("# HELP velogb10_graph_capture_invalidated_total CUDA graph captures that ended CUDA_ERROR_STREAM_CAPTURE_INVALIDATED since boot: a CUDA call that is illegal inside stream capture landed in the capture window (from another thread, e.g. the vision tower, or from the capturing thread itself), so no graph was produced and the step ran eagerly. The lazily captured shapes (EXL3 step/verify/packed-verify/draft/reprime graphs, the NVFP4 verify graph, the DFlash2 inject graph) are captured again at their next use; the three boot-time graphs (NVFP4 decode set, NVFP4 decode+sample per batch size, the DFlash2 draft round) are not retried and stay eager for the boot (the [graph] log line says which). Expected 0.\n# TYPE velogb10_graph_capture_invalidated_total counter\n");
    o.push_str(&format!("velogb10_graph_capture_invalidated_total {}\n", CAPTURE_INVALIDATED.load(Relaxed)));
    o.push_str("# HELP velogb10_graph_capture_failed_total CUDA graph captures that produced no graph for a reason other than an invalidated capture, since boot: cuStreamBeginCapture refused, cuStreamEndCapture returned another status, a body error during capture, cuGraphInstantiate failed, or the TP graph-spine audit refused the graph (also counted by the idle line's spine tally). Same recovery per graph family as the invalidated counter (eager now; captured again at the next use for the lazy shapes, not retried for the three boot-time graphs). Expected 0.\n# TYPE velogb10_graph_capture_failed_total counter\n");
    o.push_str(&format!("velogb10_graph_capture_failed_total {}\n", CAPTURE_FAILED.load(Relaxed)));
    if let Some(p) = KV_POOL.get() {
        let bytes = (KV_BYTES_SET.load(Relaxed) != 0).then(|| [0, 1, 2].map(|k| KV_BYTES[k].load(Relaxed)));
        p.render(&mut o, bytes);
    }
    o.push_str("# HELP velogb10_lane_pack_rounds_total EXL3 scheduler rounds by arm since boot: packed = packed speculative verifies (ONE verify serving two requests; lane packing is on by default, --lane-pack off disables it), serial = lone speculative rounds (one request each; a request's control rounds and its plain steps at the end of its window or budget count here), shared_plain = shared plain steps of the load-adaptive policy (--spec-lanes-max). The plain batch of MTP-off requests is not counted.\n# TYPE velogb10_lane_pack_rounds_total counter\n");
    for (i, arm) in LANE_PACK_ARMS.iter().enumerate() {
        o.push_str(&format!("velogb10_lane_pack_rounds_total{{arm=\"{arm}\"}} {}\n", LANE_PACK_ROUNDS[i].load(Relaxed)));
    }
    o.push_str("# HELP velogb10_lane_pack_lane_rounds_total LANE rounds (one per request per packed verify: two per packed round) served by a packed speculative verify since boot, by the kind of request: sampled_real_q = a sampled request whose rows verify by the real-q ratio rule (--spec-sampling ratio, the default), sampled_match = a sampled request under --spec-sampling match, greedy = temperature 0. Always 0 with --lane-pack off.\n# TYPE velogb10_lane_pack_lane_rounds_total counter\n");
    for (i, kind) in LANE_PACK_KINDS.iter().enumerate() {
        o.push_str(&format!("velogb10_lane_pack_lane_rounds_total{{kind=\"{kind}\"}} {}\n", LANE_PACK_LANES[i].load(Relaxed)));
    }

    o.push_str("# HELP velogb10_mtp_prime_batch_total Shared plain-step primes served by the BATCHED single-launch path (xq_gather_tap_rows + one re-prime pass + xq_attn_kv_prefill_pairs) since boot; velogb10_mtp_prime_batch_lanes_total counts the head-row lanes those steps carried. 0 with --exl3-prime-batch off; steps that fall back to the per-lane loop are NOT counted.\n# TYPE velogb10_mtp_prime_batch_total counter\n");
    o.push_str(&format!("velogb10_mtp_prime_batch_total {}\n", PRIME_BATCH.load(Relaxed)));
    o.push_str("# TYPE velogb10_mtp_prime_batch_lanes_total counter\n");
    o.push_str(&format!("velogb10_mtp_prime_batch_lanes_total {}\n", PRIME_BATCH_LANES.load(Relaxed)));    o.push_str("# HELP velogb10_lane_pack_skip_total LANE rounds (one per request per scheduler step, not requests) that ran their own lone round outside a packed verify while lane packing was active and at least two requests were busy, by the reason that lane was not packed. NOT counted: steps the load-adaptive policy gave to the shared plain step (--spec-lanes-max; see velogb10_lane_pack_rounds_total{arm=\"shared_plain\"}), and a lone request. When a pair falls back to serial rounds for the control cadence BOTH of its lanes are counted as `control`, although only one of them owes the control round. Always 0 with --lane-pack off.\n# TYPE velogb10_lane_pack_skip_total counter\n");
    for (i, why) in crate::lane_pack::Skip::ALL.iter().enumerate() {
        o.push_str(&format!("velogb10_lane_pack_skip_total{{reason=\"{}\"}} {}\n", why.label(), LANE_PACK_SKIP[i].load(Relaxed)));
    }
    o.push_str("# HELP velogb10_lane_pack_disabled 1 once a packed round failed and the packed arm was turned off for the rest of this boot (the server then runs the serial arm; restart to re-enable). Always 0 with --lane-pack off and on a healthy boot.\n# TYPE velogb10_lane_pack_disabled gauge\n");
    o.push_str("# HELP velogb10_spec_lanes_est_tok_s The spec-lanes auto policy's estimate of the AGGREGATE tok/s the busy clients would see on each arm, at its last decision that compared real measurements (the busy lanes' speculation EMAs known): serial = the busy requests taking turns on their own speculative rounds (packed pairs where two lanes pack); shared = one plain step for all of them. The arm with the higher estimate runs, with 4% hysteresis. 0 means no such decision yet this boot (the policy rides its fallback threshold).\n# TYPE velogb10_spec_lanes_est_tok_s gauge\n");
    o.push_str(&format!("velogb10_spec_lanes_est_tok_s{{arm=\"serial\"}} {}\n", f64::from_bits(SPEC_LANES_EST_SERIAL.load(Relaxed))));
    o.push_str(&format!("velogb10_spec_lanes_est_tok_s{{arm=\"shared\"}} {}\n", f64::from_bits(SPEC_LANES_EST_SHARED.load(Relaxed))));
    o.push_str("# HELP velogb10_spec_lanes_shared_scale The learned correction the shared plain step's cost model carries for this boot's topology and content (1 = the fitted prior; learned from real shared steps; frozen at 1 until one runs).\n# TYPE velogb10_spec_lanes_shared_scale gauge\n");
    o.push_str(&format!("velogb10_spec_lanes_shared_scale {}\n", SPEC_LANES_SCALE_MILLI.load(Relaxed) as f64 / 1000.0));
    o.push_str("# HELP velogb10_spec_lanes_busy_lanes Busy lanes at the spec-lanes policy's last measured decision (the n of the shared-step cost model at that moment).\n# TYPE velogb10_spec_lanes_busy_lanes gauge\n");
    o.push_str(&format!("velogb10_spec_lanes_busy_lanes {}\n", SPEC_LANES_BUSY.load(Relaxed)));
    o.push_str("# HELP velogb10_spec_lanes_decisions_total Spec-lanes auto decisions that compared real measurements (both arms' estimates known); decisions taken on the fallback threshold (an estimate missing) are not counted.\n# TYPE velogb10_spec_lanes_decisions_total counter\n");
    o.push_str(&format!("velogb10_spec_lanes_decisions_total {}\n", SPEC_LANES_DECISIONS.load(Relaxed)));
    o.push_str("# HELP velogb10_spec_lanes_switches_total Times the spec-lanes auto pick FLIPPED between arms since boot (to = the arm it flipped to). A large count says thrash; zero with all rounds on one arm says the policy is stuck (see the est gauges for why).\n# TYPE velogb10_spec_lanes_switches_total counter\n");
    for (i, to) in SPEC_LANES_TO.iter().enumerate() {
        o.push_str(&format!("velogb10_spec_lanes_switches_total{{to=\"{to}\"}} {}\n", SPEC_LANES_SWITCHES[i].load(Relaxed)));
    }
    o.push_str("# HELP velogb10_spec_lanes_arm_ms_total Cumulative milliseconds the arm ran since boot (serial speculation rounds vs shared plain steps) and velogb10_spec_lanes_arm_tok_total the tokens it delivered: the exact time split behind the pick. kind=millisecond_count\n");
    for (i, arm) in SPEC_LANES_TO.iter().enumerate() {
        o.push_str(&format!("velogb10_spec_lanes_arm_ms_total{{arm=\"{arm}\"}} {}\n", SPEC_ARM_MS[i].load(Relaxed)));
        o.push_str(&format!("velogb10_spec_lanes_arm_tok_total{{arm=\"{arm}\"}} {}\n", SPEC_ARM_TOK[i].load(Relaxed)));
    }
    o.push_str("# HELP velogb10_spec_lanes_measured_tok_s The MEASURED delivered aggregate tok/s of each arm (EWMA over the windows its rounds last folded; 0 = never ran recently): shared = plain steps for all busy lanes, serial = speculative rounds taking turns (packed pairs where two lanes pack). The auto pick compares these when both are fresh at the current busy count; the est_tok_s gauges are its modelled fallback. kind=rate_tokens_per_second\n");
    o.push_str("# TYPE velogb10_spec_lanes_measured_tok_s gauge\n");
    for (i, arm) in SPEC_LANES_TO.iter().enumerate() {
        o.push_str(&format!("velogb10_spec_lanes_measured_tok_s{{arm=\"{arm}\"}} {}\n", f64::from_bits(SPEC_LANES_MEASURED[i].load(Relaxed))));
    }
    o.push_str("# HELP velogb10_spec_lanes_probes_total Arm-rate probe windows started: while one arm rides, a short window of the idle arm refreshes its measured rate so the pick stays grounded in delivery. kind=window_count\n");
    o.push_str("# TYPE velogb10_spec_lanes_probes_total counter\n");
    for (i, arm) in SPEC_LANES_TO.iter().enumerate() {
        o.push_str(&format!("velogb10_spec_lanes_probes_total{{arm=\"{arm}\"}} {}\n", SPEC_LANES_PROBES[i].load(Relaxed)));
    }
    o.push_str(&format!("velogb10_lane_pack_disabled {}\n", LANE_PACK_DISABLED.load(Relaxed)));
    // F1: the boot's --lane-pack decision — active, fell back with why, or off. The boot line prints
    // the same reason (`lane packing (--lane-pack, on by default) is NOT ACTIVE on this boot: ...; requests
    // are served without packed rounds`). A --lane-pack off boot records the flag as the reason.
    let (lpa, lpr) = match crate::lane_pack::boot_pack_word() {
        Some(w) if w.is_empty() => (1u64, String::new()),
        Some(w) => (0u64, w.clone()),
        None => (0u64, "--lane-pack off".to_string()),
    };
    o.push_str("# HELP velogb10_lane_pack_active 1 when lane packing (--lane-pack, on by default) is ACTIVE on this boot (packed rounds can run); 0 when this boot serves without packed rounds: reason = why (the boot line prints the same), or `--lane-pack off`.\n# TYPE velogb10_lane_pack_active gauge\n");
    o.push_str(&format!("velogb10_lane_pack_active{{reason=\"{}\"}} {}\n",
                        lpr.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n"), lpa));
    // F3: modules the wide trellis kernel cannot decode (widths outside 3..=7 since P7 B5) served through the
    // reconstruct + cuBLASLt arm whatever --exl3-recon-min said; 0 on a 3-, 4- or 5-bit pack.
    o.push_str("# HELP velogb10_exl3_recon_forced_total EXL3 prefill-chain GEMMs that a module the wide trellis kernel cannot decode (bit width outside 3..=7 since P7 B5) served through the reconstruct + cuBLASLt arm instead of the arm --exl3-recon-min asked for, by the module's bit width. The boot log prints the once-per-boot line the first time it happens.\n# TYPE velogb10_exl3_recon_forced_total counter\n");
    o.push_str("# HELP velogb10_exl3_psk_width_fallback_total EXL3 split-K chains that asked for the persistent PSK grid (--psk=1) but whose module bit width is outside the persistent bodies (2..=6): served by the plain split-K grid instead, by bit width. The boot log prints the once-per-boot line the first time it happens.\n# TYPE velogb10_exl3_psk_width_fallback_total counter\n");
    o.push_str("# HELP velogb10_exl3_wp21_width_fallback_total EXL3 split-K chains that were eligible for the WP21 fused kernel but whose module k2 is outside the fused bodies (3..=14, e.g. k2 16 = the 8-bit MTP-head class): served by the plain 3-launch tail instead, by k2. The boot log prints one WARN the first time it happens.\n# TYPE velogb10_exl3_wp21_width_fallback_total counter\n");
    for (b, c) in PSK_WIDTH_FALLBACK.iter().enumerate().skip(1) {
        let v = c.load(Relaxed);
        if v > 0 {
            o.push_str(&format!("velogb10_exl3_psk_width_fallback_total{{bits=\"{b}\"}} {v}\n"));
        }
    }
    for (b, c) in WP21_WIDTH_FALLBACK.iter().enumerate().skip(1) {
        let v = c.load(Relaxed);
        if v > 0 {
            o.push_str(&format!("velogb10_exl3_wp21_width_fallback_total{{k2=\"{b}\"}} {v}\n"));
        }
    }
    for b in 1..RECON_FORCED.len() {
        let v = RECON_FORCED[b].load(Relaxed);
        if v > 0 {
            o.push_str(&format!("velogb10_exl3_recon_forced_total{{bits=\"{b}\"}} {v}\n"));
        }
    }
    o.push_str("# HELP velogb10_fp4l_pack 1 when the served model is an FP4-Local (NVIDIA ModelOpt MXFP8 + NVFP4) pack.\n# TYPE velogb10_fp4l_pack gauge\n");
    o.push_str(&format!("velogb10_fp4l_pack {}\n", FP4L_PACK.load(Relaxed)));
    o.push_str("# HELP velogb10_fp4l_dispatch_total FP4-Local weight-kernel host dispatches by weight class (mxfp8 dense, f16 dense, bf16 = stored bf16 bits, nvfp4 experts, exl3_mtp_sidecar = the opt-in EXL3 trellis MTP-head linears of --fp4l-mtp-sidecar) and row class (decode = 1 row, verify = 2..16 rows incl. packed lanes, prefill = more). Graph replays do not dispatch from the host: in graphed decode these grow at capture.\n# TYPE velogb10_fp4l_dispatch_total counter\n");
    for (ki, kn) in ["mxfp8", "f16", "bf16", "nvfp4_experts", "exl3_mtp_sidecar"].iter().enumerate() {
        for (ri, rn) in ["decode", "verify", "prefill"].iter().enumerate() {
            let v = FP4L_DISPATCH[ki][ri].load(Relaxed);
            if v > 0 { o.push_str(&format!("velogb10_fp4l_dispatch_total{{weights=\"{kn}\",rows=\"{rn}\"}} {v}\n")); }
        }
    }
    if let Some(p) = FP4L_PREC.get() {
        o.push_str("# HELP velogb10_fp4l_precision The FP4-Local precision levers this boot resolved (--fp4l-plain, --fp4l-norms, --fp4l-lm-head, --fp4l-draft-head, --fp4l-prefill-dense); the value is always 1.\n# TYPE velogb10_fp4l_precision gauge\n");
        o.push_str(&format!("velogb10_fp4l_precision{{{p}}} 1\n"));
    }
    o.push_str("# HELP velogb10_fp4l_twin_total Host dispatches of the FP4-Local exact-precision twins of trunk kernels by op (stored bf16 mixers / router / embedding / shared gate / n-gram weights, fp32 norms). 0 under --fp4l-plain f16 --fp4l-norms f16. Graph replays do not dispatch from the host: in graphed decode these grow at capture.\n# TYPE velogb10_fp4l_twin_total counter\n");
    for (i, op) in FP4L_TWIN_OPS.iter().enumerate() {
        let v = FP4L_TWIN[i].load(Relaxed);
        if v > 0 { o.push_str(&format!("velogb10_fp4l_twin_total{{op=\"{op}\"}} {v}\n")); }
    }
    o.push_str("# HELP velogb10_log_lines_dropped_total Log lines dropped because whatever reads the server's output stalled (the log queue was full).\n# TYPE velogb10_log_lines_dropped_total counter\n");
    o.push_str(&format!("velogb10_log_lines_dropped_total {}\n", crate::logq::dropped()));
    o.push_str("# HELP velogb10_requests_rejected_total Requests refused at admission (--max-waiting exceeded).\n# TYPE velogb10_requests_rejected_total counter\n");
    o.push_str(&format!("velogb10_requests_rejected_total {}\n", REQ_REJECTED.load(Relaxed)));
    o.push_str("# HELP velogb10_streams_cancelled_backlog_total Streams cancelled for exceeding --stream-backlog-events (the client stopped reading).\n# TYPE velogb10_streams_cancelled_backlog_total counter\n");
    o.push_str(&format!("velogb10_streams_cancelled_backlog_total {}\n", STREAMS_CANCELLED_BACKLOG.load(Relaxed)));
    o.push_str("# HELP velogb10_lane_reserve_admits_total Admissions decided by the --lane-reserve policy (EXL3; head-authoritative at TP>=2 since LANERES3: the head ships each admit's scoping, every rank picks the same lane), by rule: own = the waiting request's prompt-end snapshot sat in a free lane (admitted first, into that lane); unreserved = a free lane no reservation protects; forced = the queue head out-waited --lane-reserve-patience and took a reserved lane (rate-limited by --lane-reserve-swap). All zero when --lane-reserve is off (default).\n# TYPE velogb10_lane_reserve_admits_total counter\n");
    o.push_str(&format!("velogb10_lane_reserve_admits_total{{rule=\"own\"}} {}\n", LANE_RESERVE_ADMITS[0].load(Relaxed)));
    o.push_str(&format!("velogb10_lane_reserve_admits_total{{rule=\"unreserved\"}} {}\n", LANE_RESERVE_ADMITS[1].load(Relaxed)));
    o.push_str(&format!("velogb10_lane_reserve_admits_total{{rule=\"forced\"}} {}\n", LANE_RESERVE_ADMITS[2].load(Relaxed)));
    o.push_str(&format!("velogb10_lane_reserve_tp_requeued_total {}\n", LANE_RESERVE_TP_REQUEUED.load(Relaxed)));
    o.push_str("# HELP velogb10_lane_reserve_waiting Requests waiting while the --lane-reserve policy holds every free lane reserved for their conversations (0 when the policy is off or nothing waits).\n# TYPE velogb10_lane_reserve_waiting gauge\n");
    o.push_str(&format!("velogb10_lane_reserve_waiting {}\n", LANE_RESERVE_WAITING.load(Relaxed)));
    {
        let (pc, sc) = (NAIVE_TP_PRIMARY_CTX.load(Relaxed), NAIVE_TP_SINGLE_CTX.load(Relaxed));
        if pc != 0 {
            o.push_str("# HELP velogb10_naive_tp_reduce Naive TP all-reduce paths (device epoch counters): single_stage = 1 when --tp-reduce resolved to the one-shot ctx (world 4); single_epochs_total = reduces that took it (one epoch each); tree_epochs_total = exchanges on the primary link (log2(world) per tree reduce: prefill chunks, and every reduce when the single stage is off)\n");
            o.push_str(&format!("velogb10_naive_tp_reduce_single_stage {}\n", (sc != 0) as u8));
            o.push_str(&format!("velogb10_naive_tp_reduce_single_gpu_recv {}\n", (sc != 0 && NAIVE_TP_SINGLE_GPU_RECV.load(Relaxed)) as u8));
            o.push_str(&format!("velogb10_naive_tp_reduce_single_k1_multiblock {}\n", (sc != 0 && NAIVE_TP_SINGLE_K1_MB.load(Relaxed)) as u8));
            o.push_str(&format!("velogb10_naive_tp_reduce_single_epochs_total {}\n", if sc != 0 { crate::net::ctx_device_epoch(sc) } else { 0 }));
            o.push_str(&format!("velogb10_naive_tp_reduce_tree_epochs_total {}\n", crate::net::ctx_device_epoch(pc)));
        }
        // every velogb10_naive_* series renders ONLY on a Naive server (lanes > 0): owner_check keys its NAIVE-N0.5
        // section on their presence, and any other model's server must export none of them
        let nl = NAIVE_LANES.load(Relaxed);
        if nl > 0 {
            o.push_str("# HELP velogb10_naive_config Naive serve configuration of this boot: moe_shard_tp = 1 when every rank holds a slice of every expert (--naive-moe-shard tp), lanes = concurrent requests (--naive-lanes)\n");
            o.push_str(&format!("velogb10_naive_moe_shard_tp {}\n", NAIVE_MOE_TP.load(Relaxed) as u8));
            o.push_str(&format!("velogb10_naive_lanes {nl}\n"));
            o.push_str(&format!("velogb10_naive_lane_ctx {}\n", NAIVE_LANE_CTX.load(Relaxed) as u8));
        o.push_str("# HELP velogb10_naive_spec_sampled Naive exact speculative sampling (sampled requests): rounds, accepted drafts, residual fallbacks by cause\n");
        o.push_str(&format!("velogb10_naive_spec_sampled_rounds_total {}\n", NAIVE_SPEC_SAMPLED_ROUNDS.load(Relaxed)));
        o.push_str(&format!("velogb10_naive_spec_sampled_accepted_total {}\n", NAIVE_SPEC_SAMPLED_ACCEPTED.load(Relaxed)));
        o.push_str(&format!("velogb10_naive_spec_fallback_nolist_total {}\n", NAIVE_SPEC_FALLBACK_NOLIST.load(Relaxed)));
        o.push_str(&format!("velogb10_naive_spec_fallback_empty_residual_total {}\n", NAIVE_SPEC_FALLBACK_EMPTY.load(Relaxed)));
            o.push_str("# HELP velogb10_naive_prefix Naive prefix reuse (--prefix-cache on): state 1 = on (one conversation per KV slot), 0 = not asked (2 = asked but off: builds before 2026-10-21 with --naive-lanes > 1); hits = prompts that resumed after the previous prompt's prefix, misses = prompts prefilled whole, reused_tokens = prompt tokens not prefilled again\n");
            o.push_str(&format!("velogb10_naive_prefix_state {}\n", NAIVE_PREFIX_ON.load(Relaxed)));
            o.push_str(&format!("velogb10_naive_prefix_hits_total {}\n", NAIVE_PREFIX_HITS.load(Relaxed)));
            o.push_str(&format!("velogb10_naive_prefix_misses_total {}\n", NAIVE_PREFIX_MISSES.load(Relaxed)));
            o.push_str(&format!("velogb10_naive_prefix_reused_tokens_total {}\n", NAIVE_PREFIX_REUSED.load(Relaxed)));
        }
    }
    o.push_str("# HELP velogb10_prefix_pool Prefix pool: prompt states kept in memory across lanes (--prefix-pool-gb). cap/bytes/entries = now; captures = lane states copied in (capture_us_total = their in-round copy time); skips = already held or too short; refused = over the cap, under the memory floor, or a failed copy (any rank); hits = prompts resumed from an entry (hit_tokens = prompt tokens not read again; restore_us_total = the copy time); misses = no lane snapshot and no entry deeper than the lane's own checkpoint\n");
    o.push_str(&format!("velogb10_prefix_pool_cap_bytes {}\n", PFX_POOL_CAP.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_bytes {}\n", PFX_POOL_BYTES.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_entries {}\n", PFX_POOL_ENTRIES.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_captures_total {}\n", PFX_POOL_CAPTURES.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_capture_us_total {}\n", PFX_POOL_CAPTURE_US.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_skips_total {}\n", PFX_POOL_SKIPS.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_refused_total {}\n", PFX_POOL_REFUSED.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_evictions_total {}\n", PFX_POOL_EVICTIONS.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_emergency_frees_total {}\n", PFX_POOL_EMERGENCY.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_hits_total {}\n", PFX_POOL_HITS.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_hit_tokens_total {}\n", PFX_POOL_HIT_TOKENS.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_restore_us_total {}\n", PFX_POOL_RESTORE_US.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_misses_total {}\n", PFX_POOL_MISSES.load(Relaxed)));
    o.push_str("# HELP velogb10_kv_spill SSD spill tier for evicted lane snapshots (--kv-spill)\n");
    o.push_str(&format!("velogb10_kv_spill_writes_total {}\n", KV_SPILL_WRITES.load(Relaxed)));
    o.push_str("# HELP velogb10_prefill_interleave_steps_total Decode steps run between prefill chunks of joining prompts (--prefill-interleave)\n");
    o.push_str(&format!("velogb10_prefill_interleave_steps_total {}\n", PREFILL_INTERLEAVE_STEPS.load(Relaxed)));
    o.push_str(&format!("velogb10_prefill_interleave_points_total {}\n", PREFILL_INTERLEAVE_POINTS.load(Relaxed)));
    o.push_str(&format!("velogb10_kv_spill_hits_total {}\n", KV_SPILL_HITS.load(Relaxed)));
    o.push_str(&format!("velogb10_kv_spill_misses_recompute_total {}\n", KV_SPILL_MISSES.load(Relaxed)));
    o.push_str(&format!("velogb10_kv_spill_capture_peer_drop_total {}\n", KV_SPILL_CAPTURE_PEER_DROPS.load(Relaxed)));
    o.push_str(&format!("velogb10_kv_spill_restore_peer_miss_total {}\n", KV_SPILL_RESTORE_PEER_MISSES.load(Relaxed)));
    o.push_str(&format!("velogb10_kv_spill_bytes {}\n", KV_SPILL_BYTES.load(Relaxed)));
    o.push_str(&format!("velogb10_kv_spill_tokens_recompute_saved_total {}\n", KV_SPILL_SAVED.load(Relaxed)));
    let (steps, drafts, accepted, emitted) = crate::tel::spec_totals();
    o.push_str("# HELP velogb10_spec_draft_tokens_total Speculative draft tokens proposed (MTP / DFlash2 / DFlash).\n# TYPE velogb10_spec_draft_tokens_total counter\n");
    o.push_str(&format!("velogb10_spec_draft_tokens_total {drafts}\n"));
    o.push_str("# HELP velogb10_spec_accepted_tokens_total Speculative draft tokens accepted by the verify.\n# TYPE velogb10_spec_accepted_tokens_total counter\n");
    o.push_str(&format!("velogb10_spec_accepted_tokens_total {accepted}\n"));
    o.push_str("# HELP velogb10_spec_rounds_total Speculative rounds (verify forwards).\n# TYPE velogb10_spec_rounds_total counter\n");
    o.push_str(&format!("velogb10_spec_rounds_total {steps}\n"));
    o.push_str("# HELP velogb10_spec_emitted_tokens_total Tokens emitted by speculative rounds (accepted + 1 per round).\n# TYPE velogb10_spec_emitted_tokens_total counter\n");
    o.push_str(&format!("velogb10_spec_emitted_tokens_total {emitted}\n"));
    let (mtp_off, reprobes, reprobe_ok) = crate::tel::mtp_reprobe_totals();
    o.push_str("# HELP velogb10_spec_mtp_off_events_total MTP off EVENTS, not requests (one request can go off several times; re-offs after failed probes count; flushed at request end with [mtp-stats]).\n# TYPE velogb10_spec_mtp_off_events_total counter\n");
    o.push_str(&format!("velogb10_spec_mtp_off_events_total {mtp_off}\n"));
    o.push_str("# HELP velogb10_spec_mtp_reprobes_total Re-probe windows the engine opened after an MTP off (speculation re-armed to test the workload again).\n# TYPE velogb10_spec_mtp_reprobes_total counter\n");
    o.push_str(&format!("velogb10_spec_mtp_reprobes_total {reprobes}\n"));
    o.push_str("# HELP velogb10_spec_mtp_reprobe_ok_total Re-probe windows that ended with speculation back on.\n# TYPE velogb10_spec_mtp_reprobe_ok_total counter\n");
    o.push_str(&format!("velogb10_spec_mtp_reprobe_ok_total {reprobe_ok}\n"));
    crate::exl3_serve::df2_metrics_append(&mut o);
    render_toolcall(&mut o);
    o
}

#[cfg(test)]
pub(crate) mod tests {
    /// H3 tests drive the same scheduler statics — serialize them so busy/idle reads are exact.
    /// (pub(crate): the keep-alive test in server.rs reads the same gauges.)
    pub(crate) static SCHED_GAUGE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// S-B9-TOOLPARSE3 T1: the parser's shadow counters render with every series present (zeros
    /// included, so "no disagreement" and "not running" differ), a disagreement lands in its own
    /// cell, and the HELP texts say what a cell means.
    #[test]
    fn toolcall_shadow_counters_render_and_count() {
        use crate::tools::{DiffKind, Fewer, Other, Rules, Shadow, ShadowDiff};
        let _g = SCHED_GAUGE_LOCK.lock().unwrap();
        let before = super::render(true);
        let get = |s: &str, name: &str| -> f64 {
            s.lines().find(|l| l.starts_with(name)).and_then(|l| l.rsplit(' ').next()).and_then(|v| v.parse().ok()).unwrap_or(f64::NAN)
        };
        let cell = "velogb10_toolcall_parse_disagreement_total{other=\"cost\",kind=\"call_count\",fewer=\"other\"}";
        assert_eq!(before.matches("velogb10_toolcall_parse_disagreement_total{").count(), 36, "every series is present");
        let c0 = get(&before, cell);
        let b0 = get(&before, "velogb10_toolcall_blocks_total");
        let sh = Shadow {
            skipped: None,
            blocks: 2,
            compared: [0, 2, 2, 1],
            cost_declined: 0,
            diffs: vec![ShadowDiff { other: Other::Cost, kind: DiffKind::CallCount, fewer: Fewer::Other, hash: "0123456789abcdef".into(), len: 300, calls_served: 2, calls_other: 1 }],
        };
        super::toolcall_shadow(&sh, &Rules::V073);
        super::toolcall_blocks(2);
        let after = super::render(true);
        // (other tests parse through `parse_served` on parallel threads and bump the same process-wide
        // counters, so a delta is a lower bound)
        assert!(get(&after, cell) >= c0 + 1.0);
        assert!(get(&after, "velogb10_toolcall_blocks_total") >= b0 + 2.0);
        assert!(after.contains("velogb10_toolcall_rules_info{pairing=\"off\",block=\"after-function\",unclosed=\"drop\"} 1"));
        for h in ["velogb10_toolcall_parse_disagreement_total", "velogb10_toolcall_shadow_compared_total", "velogb10_toolcall_blocks_total"] {
            let help = after.lines().find(|l| l.starts_with(&format!("# HELP {h} "))).expect(h);
            assert!(help.len() > 120, "{h}: the HELP text says what a cell means");
        }
        assert!(after.contains("fewer says which of the two readings returned FEWER calls"));
    }

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
            "velogb10_graph_cache_entries", "velogb10_graph_cache_mixed_step_entries", "velogb10_log_lines_dropped_total",
            "velogb10_graph_capture_invalidated_total", "velogb10_graph_capture_failed_total",
            "velogb10_requests_rejected_total", "velogb10_streams_cancelled_backlog_total",
            "velogb10_lane_pack_rounds_total", "velogb10_lane_pack_skip_total", "velogb10_lane_pack_lane_rounds_total",
            "velogb10_lane_pack_disabled",
            "velogb10_spec_lanes_est_tok_s", "velogb10_spec_lanes_shared_scale", "velogb10_spec_lanes_busy_lanes",
            "velogb10_spec_lanes_decisions_total", "velogb10_spec_lanes_switches_total",
            "velogb10_spec_lanes_arm_ms_total", "velogb10_spec_lanes_arm_tok_total",
            "velogb10_spec_lanes_measured_tok_s", "velogb10_spec_lanes_probes_total",
        ] {
            assert!(s.contains(name), "missing gauge {name}");
        }
        for l in s.lines().filter(|l| !l.starts_with('#')) {
            let v = l.rsplit(' ').next().unwrap();
            assert!(v.parse::<f64>().is_ok(), "not a sample line: {l}");
        }
        assert!(s.lines().any(|l| l.starts_with("velogb10_requests_rejected_total") && l.ends_with(" 1")));
        // SURFACE2 item 6: HELP text is read by operators — no internal audit tag (LR-n), and the two
        // `_entries` gauges say what a gauge holds (graphs held NOW), not "captured since boot".
        for l in s.lines().filter(|l| l.starts_with("# HELP ")) {
            assert!(!l.contains("LR-"), "internal audit tag in a HELP text: {l}");
        }
        for name in ["velogb10_graph_cache_entries", "velogb10_graph_cache_mixed_step_entries"] {
            let h = s.lines().find(|l| l.starts_with(&format!("# HELP {name} "))).unwrap();
            assert!(h.contains("holds now") && !h.contains("since boot"), "{h}");
            assert!(s.contains(&format!("# TYPE {name} gauge\n")));
        }
    }

    /// H3 (REL_V0_7_3): the busy/idle transitions and the END-of-work age timestamp, on the
    /// REAL gauge functions (what the schedulers call): idle -> 0, round start -> 1, and
    /// S-B9-REL-SURFACE2B R6(d): the HELP texts say what the code does. The steps counter moves
    /// when a round STARTS (`sched_step`), and busy is set there and nowhere else, so an idle
    /// server prefilling its first request reads busy 0 (source pins: the two writers).
    #[test]
    fn scheduler_help_texts_say_started_and_decode_round() {
        let text = super::render(true);
        let help = |name: &str| text.lines().find(|l| l.starts_with(&format!("# HELP {name} "))).unwrap_or_else(|| panic!("{name}")).to_string();
        let steps = help("velogb10_scheduler_steps_total");
        assert!(steps.contains("rounds STARTED since boot") && !steps.contains("completed since"), "{steps}");
        let busy = help("velogb10_scheduler_busy");
        assert!(busy.contains("from the start of a decode round") && busy.contains("still reads 0 while it admits and prefills its first request"), "{busy}");
        let src = include_str!("metrics.rs");
        let code = &src[..src.find("mod tests").unwrap()];
        assert_eq!(code.matches(&["SCHED_STEPS.", "fetch_add"].concat()).count(), 1);
        assert_eq!(code.matches(&["SCHED_BUSY.", "store(1"].concat()).count(), 1);
        let step = &code[code.find("pub fn sched_step()").unwrap()..code.find("pub fn sched_touch()").unwrap()];
        assert!(step.contains("SCHED_STEPS.fetch_add") && step.contains("SCHED_BUSY.store(1"), "both are written when a round starts");
    }

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
        // S-B9-REL-BENCHFIX2 (W4): the keep-alive's liveness read is these same two gauges
        let st = super::sched_state();
        assert!(st.busy, "the round has not gone idle yet");
        let work = st.work_age.expect("a unit of work completed");
        assert!(work >= std::time::Duration::from_millis(30) && work < std::time::Duration::from_secs(30),
                "counts from sched_touch: {work:?}");
        super::sched_idle();
        assert!(!super::sched_state().busy);
        assert_eq!(busy(), "0", "idle again reads 0");
    }

    fn sample(s: &str, name: &str) -> u64 {
        s.lines().find(|l| l.starts_with(name) && l[name.len()..].starts_with(' '))
            .unwrap_or_else(|| panic!("no sample line for {name}"))
            .rsplit(' ').next().unwrap().parse().unwrap()
    }

    /// Issue #24: the rendering of the KV pool gauges — names, HELP/TYPE lines, the per-lane label,
    /// the byte components, and the documented relations between the series.
    #[test]
    fn kv_pool_renders_the_documented_series() {
        let p = super::kv_pool_new(4, 32768);
        // lane 0 runs (1,500 positions), lane 1 is prefilling (4,095 so far), lane 2 is free and
        // keeps a 900-token prefix snapshot, lane 3 never served
        p.publish([(1500, 0), (4095, 0), (0, 900), (0, 0)].into_iter());
        let mut s = String::new();
        p.render(&mut s, Some([1_000_000, 20_000, 300]));
        assert_eq!(sample(&s, "velogb10_kv_positions_capacity"), 4 * 32768);
        assert_eq!(sample(&s, "velogb10_kv_positions_used"), 5595);
        assert_eq!(sample(&s, "velogb10_kv_positions_peak"), 5595);
        assert_eq!(sample(&s, "velogb10_kv_positions_cached"), 900);
        for (lane, used, cached) in [(0, 1500, 0), (1, 4095, 0), (2, 0, 900), (3, 0, 0)] {
            assert!(s.contains(&format!("velogb10_kv_lane_positions_used{{lane=\"{lane}\"}} {used}\n")), "{s}");
            assert!(s.contains(&format!("velogb10_kv_lane_positions_peak{{lane=\"{lane}\"}} {used}\n")));
            assert!(s.contains(&format!("velogb10_kv_lane_positions_cached{{lane=\"{lane}\"}} {cached}\n")));
            assert!(s.contains(&format!("velogb10_kv_lane_positions_capacity{{lane=\"{lane}\"}} 32768\n")));
        }
        assert!(s.contains("velogb10_kv_allocated_bytes{kind=\"trunk_kv\"} 1000000\n"));
        assert!(s.contains("velogb10_kv_allocated_bytes{kind=\"indexer_keys\"} 20000\n"));
        assert!(s.contains("velogb10_kv_allocated_bytes{kind=\"draft_head\"} 300\n"));
        // every series has exactly one HELP and one TYPE line, and every sample line parses
        for name in ["velogb10_kv_positions_capacity", "velogb10_kv_positions_used", "velogb10_kv_positions_peak",
                     "velogb10_kv_positions_cached", "velogb10_kv_lane_positions_capacity", "velogb10_kv_lane_positions_used",
                     "velogb10_kv_lane_positions_peak", "velogb10_kv_lane_positions_cached", "velogb10_kv_allocated_bytes"] {
            assert_eq!(s.matches(&format!("# HELP {name} ")).count(), 1, "HELP of {name}");
            assert_eq!(s.matches(&format!("# TYPE {name} gauge\n")).count(), 1, "TYPE of {name}");
        }
        for l in s.lines().filter(|l| !l.starts_with('#')) {
            assert!(l.rsplit(' ').next().unwrap().parse::<f64>().is_ok(), "not a sample line: {l}");
        }
        // S-B9-REL-SURFACE2B R3: `_total` is the counter suffix. These are gauges, so none of them
        // carries it (`promtool check metrics` flags a non-counter named `_total`, and dashboards
        // offer rate() on one): capacity for the two position series, allocated_bytes for the third.
        // The same holds for every gauge of the whole /metrics page.
        for text in [s.clone(), super::render(true)] {
            for l in text.lines().filter(|l| l.starts_with("# TYPE ") && l.ends_with(" gauge")) {
                let name = l.split(' ').nth(2).unwrap();
                assert!(!name.ends_with("_total"), "a gauge named like a counter: {name}");
            }
        }
        assert!(!s.contains("kv_positions_total") && !s.contains("kv_lane_positions_total") && !s.contains("kv_bytes_total"));
        // HELP honesty: positions are not bytes, and "used" is what RUNNING requests hold
        let help = |n: &str| s.lines().find(|l| l.starts_with(&format!("# HELP {n} "))).unwrap().to_string();
        assert!(help("velogb10_kv_positions_capacity").contains("not a byte"));
        assert!(help("velogb10_kv_positions_used").contains("running now"));
        assert!(help("velogb10_kv_positions_cached").contains("Not part of positions_used"));
        // the byte series is absent when the engine registered no byte counts
        let mut s2 = String::new();
        p.render(&mut s2, None);
        assert!(!s2.contains("velogb10_kv_allocated_bytes"));
        // the finished request frees lane 0; lane 1 becomes a running lane
        p.publish([(0, 1400), (4200, 0), (0, 900), (0, 0)].into_iter());
        let mut s3 = String::new();
        p.render(&mut s3, None);
        assert_eq!(sample(&s3, "velogb10_kv_positions_used"), 4200);
        assert_eq!(sample(&s3, "velogb10_kv_positions_peak"), 5595, "the peak is a high-water mark");
        assert_eq!(sample(&s3, "velogb10_kv_positions_cached"), 2300);
        assert!(s3.contains("velogb10_kv_lane_positions_peak{lane=\"0\"} 1500\n"));
        assert!(s3.contains("velogb10_kv_lane_positions_peak{lane=\"1\"} 4200\n"));
    }

    /// Issue #24 invariants over an arbitrary publication sequence (including values an engine bug
    /// could produce: past the window): used <= total, peak >= used, a peak never decreases, the
    /// per-lane series sum to the aggregate, and each lane's peak bounds its own series.
    #[test]
    fn kv_pool_peak_never_decreases_and_used_never_exceeds_total() {
        let (lanes, window) = (5usize, 1000u64);
        let p = super::kv_pool_new(lanes, window as usize);
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = move || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
        let (mut last_peak, mut last_lane_peak) = (0u64, vec![0u64; lanes]);
        for step in 0..20_000 {
            let vals: Vec<(usize, usize)> = (0..lanes).map(|_| match rnd() % 4 {
                0 => (0, (rnd() % 1500) as usize),       // free lane, maybe an over-long "snapshot"
                1 => ((rnd() % 1300) as usize, 0),       // running, maybe past the window
                _ => ((rnd() % 900) as usize, 0),
            }).collect();
            // more lanes than the pool has are ignored, fewer leave the rest untouched
            let extra = if step % 97 == 0 { vec![(5000usize, 5000usize)] } else { vec![] };
            p.publish(vals.iter().copied().chain(extra));
            let used = p.used.load(super::Relaxed);
            let peak = p.peak.load(super::Relaxed);
            assert!(used <= lanes as u64 * window, "used {used} exceeds total");
            assert!(peak >= used && peak >= last_peak, "peak {peak} (was {last_peak}) vs used {used}");
            last_peak = peak;
            let mut sum = 0;
            for (i, l) in p.lanes.iter().enumerate() {
                let (u, pk) = (l.used.load(super::Relaxed), l.peak.load(super::Relaxed));
                assert!(u <= window && l.cached.load(super::Relaxed) <= window);
                assert_eq!(u, (vals[i].0 as u64).min(window));
                assert!(pk >= u && pk >= last_lane_peak[i] && pk <= peak);
                last_lane_peak[i] = pk;
                sum += u;
            }
            assert_eq!(sum, used, "the per-lane series sum to the aggregate");
        }
        assert!(last_peak > 0);
    }

    /// Issue #24: a concurrent scrape never sees `peak < used` (the peak is raised first), and the
    /// label cardinality is bounded.
    #[test]
    fn kv_pool_scrape_is_consistent_under_a_live_writer_and_lane_series_are_bounded() {
        let p = std::sync::Arc::new(super::kv_pool_new(2, 1_000_000));
        let w = p.clone();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_w = stop.clone();
        let writer = std::thread::spawn(move || {
            let mut i = 0usize;
            while !stop_w.load(super::Relaxed) { i += 1; w.publish([(i, 0), (i / 2, 0)].into_iter()); }
        });
        for _ in 0..20_000 {
            // read used first (Acquire), then the peak: what the writer stored before `used` is visible
            let used = p.used.load(super::Acquire);
            let peak = p.peak.load(super::Relaxed);
            assert!(peak >= used, "scrape saw peak {peak} < used {used}");
        }
        stop.store(true, super::Relaxed);
        writer.join().unwrap();
        let big = super::kv_pool_new(super::KV_LANE_SERIES_MAX + 6, 10);
        big.publish((0..super::KV_LANE_SERIES_MAX + 6).map(|_| (3, 0)));
        let mut s = String::new();
        big.render(&mut s, None);
        assert_eq!(s.matches("velogb10_kv_lane_positions_used{").count(), super::KV_LANE_SERIES_MAX);
        assert_eq!(sample(&s, "velogb10_kv_positions_used"), 3 * (super::KV_LANE_SERIES_MAX as u64 + 6), "totals cover every lane");
    }

    /// Issue #24: the global wiring — nothing is emitted until an engine registers a pool, then the
    /// scheduler's `kv_publish` reaches `/metrics`.
    #[test]
    fn kv_pool_global_is_absent_until_an_engine_registers_it() {
        // (this is the only test that touches the global pool, so "absent before init" is exact)
        // the pool family only: velogb10_kv_spill_* is a separate family (LANERES2) and may be present
        assert!(!super::render(true).contains("velogb10_kv_pool")
            && !super::render(true).contains("velogb10_kv_positions")
            && !super::render(true).contains("velogb10_kv_allocated_bytes"), "no engine registered a pool yet");
        super::kv_publish([(5, 0)].into_iter()); // a publish before init is a no-op, not a panic
        super::kv_pool_init(3, 4096);
        super::kv_pool_init(9, 9); // first call wins
        super::kv_pool_add_bytes(0, 4096);
        super::kv_pool_add_bytes(0, 4096);
        super::kv_pool_add_bytes(7, 1); // unknown kind: ignored
        super::kv_publish([(100, 0), (0, 50), (0, 0)].into_iter());
        let s = super::render(true);
        assert_eq!(sample(&s, "velogb10_kv_positions_capacity"), 3 * 4096);
        assert_eq!(sample(&s, "velogb10_kv_positions_used"), 100);
        assert_eq!(sample(&s, "velogb10_kv_positions_cached"), 50);
        assert!(s.contains("velogb10_kv_allocated_bytes{kind=\"trunk_kv\"} 8192\n"));
        for l in s.lines().filter(|l| !l.starts_with('#')) {
            assert!(l.rsplit(' ').next().unwrap().parse::<f64>().is_ok(), "not a sample line: {l}");
        }
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

    /// S-B9-REL-VISIONFIX2 V2b: every lost capture is counted, split by cause; the line prints once
    /// per (kind, shape, cause) and never again; /metrics shows both counters.
    #[test]
    fn lost_captures_are_counted_and_logged_once_per_shape() {
        let _g = SCHED_GAUGE_LOCK.lock().unwrap();
        let (i0, f0) = (super::graph_captures_invalidated(), super::graph_captures_failed());
        let inv = super::CAPTURE_INVALIDATED_STATUS;
        assert!(super::graph_capture_lost("unit-test-kind", "width 8", inv), "the first loss of a shape logs");
        assert!(!super::graph_capture_lost("unit-test-kind", "width 8", inv), "the second loss of the same shape stays silent");
        assert!(super::graph_capture_lost("unit-test-kind", "width 9", inv), "another shape logs");
        assert!(super::graph_capture_lost("unit-test-kind", "width 8", "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED"), "another cause logs");
        assert_eq!(super::graph_captures_invalidated() - i0, 3);
        assert_eq!(super::graph_captures_failed() - f0, 1);
        let s = super::render(true);
        for name in ["velogb10_graph_capture_invalidated_total", "velogb10_graph_capture_failed_total"] {
            assert!(s.contains(&format!("# TYPE {name} counter")), "{name}");
            assert!(s.lines().any(|l| l.starts_with(name) && !l.starts_with('#')), "{name} has a sample");
        }
    }

    /// S-B9-REL-INTEG C1: the line and the counters' texts say what is counted and what happens next, per
    /// graph family (they promised "captured again at its next use" for three families that are captured
    /// once, at boot), and every failure kind the code now reports is counted as `failed`.
    #[test]
    fn the_lost_capture_texts_say_what_happens_next_per_family() {
        let inv = super::CAPTURE_INVALIDATED_STATUS;
        for kind in ["step", "verify", "packed-verify", "draft-pass", "draft-chain", "reprime", "nvfp4-verify", "df2-inject"] {
            let l = super::capture_lost_line(kind, "width 8", inv);
            assert!(l.contains("captured again at its next use"), "{kind}: {l}");
            assert!(l.contains("from another thread") && l.contains("capturing thread itself"), "{kind}: the cause is not over-attributed: {l}");
        }
        for (kind, needle) in [("nvfp4-decode", "drops ALL decode graphs"), ("nvfp4-decode-sample", "never retried"),
                               ("df2-draft-round", "never retried")] {
            for st in [inv, "BeginCapture CUDA_ERROR_NOT_SUPPORTED", "instantiate CUDA_ERROR_OUT_OF_MEMORY"] {
                let l = super::capture_lost_line(kind, "s", st);
                assert!(l.contains(needle) && !l.contains("captured again at its next use"), "{kind} {st}: {l}");
                assert!(l.contains(if st == inv { "velogb10_graph_capture_invalidated_total" } else { "velogb10_graph_capture_failed_total" }), "{l}");
            }
        }
        // the statuses the newly counted sites pass are `failed`, not `invalidated`
        let _g = SCHED_GAUGE_LOCK.lock().unwrap();
        let (i0, f0) = (super::graph_captures_invalidated(), super::graph_captures_failed());
        for st in ["BeginCapture CUDA_ERROR_UNKNOWN", "instantiate CUDA_ERROR_OUT_OF_MEMORY", "tp_spine_audit: a barrier has a concurrent node"] {
            super::graph_capture_lost("unit-test-sites", st, st);
        }
        assert_eq!((super::graph_captures_invalidated() - i0, super::graph_captures_failed() - f0), (0, 3));
        // the /metrics texts name what is counted
        let s = super::render(true);
        for needle in ["cuStreamBeginCapture refused", "cuGraphInstantiate failed", "graph-spine audit refused",
                       "the three boot-time graphs", "from the capturing thread itself"] {
            assert!(s.contains(needle), "{needle:?} missing from the HELP texts");
        }
        assert!(!s.contains("a thread besides the scheduler touched the device"));
    }

}
