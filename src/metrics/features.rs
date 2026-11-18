//! Engagement counters for request-visible features: tool-secret scrubbing, DF2 on the NVFP4 path, draft vocab, lane capacity, logprobs / schema / think-budget outcomes, over-cap admission.
//! Each counter is an engagement receipt (AGENTS.md 1b-bis.2): a counter that never moves is a path that never ran. Rendered by `render.rs`. Map: `mod.rs`.

use super::*;

// TOOLSCRUB1: spans redacted by --scrub-tool-secrets, by layer, since boot (advisory
// liveness: a span split by the pending cap can count twice). Zero forever when the
// flag is off (the default): the counters exist but never move.
pub(super) static SCRUB_PROV_HITS: AtomicU64 = AtomicU64::new(0);
pub(super) static SCRUB_PAT_HITS: AtomicU64 = AtomicU64::new(0);

pub fn scrub_hit_provenance() { SCRUB_PROV_HITS.fetch_add(1, Relaxed); }
pub fn scrub_hit_pattern() { SCRUB_PAT_HITS.fetch_add(1, Relaxed); }

// DFLASH2-GEN: the DF2 chain's engagement counters on the NON-EXL3 (nvfp4) serve path.
// scripts/owner_check.py prints them; the exl3_df2_* series are the EXL3 server's arm.
pub(super) static DF2_ROUNDS: AtomicU64 = AtomicU64::new(0);
pub(super) static DF2_MASK_OVERRIDE: AtomicU64 = AtomicU64::new(0);

pub fn df2_round() { DF2_ROUNDS.fetch_add(1, Relaxed); }
pub fn df2_mask_override(on: bool) { DF2_MASK_OVERRIDE.store(on as u64, Relaxed); }

// --draft-vocab (FR-Spec, opt-in): the reduced draft head's size (0 = not built: the drafter reads the
// full vocabulary) and the MTP draft tokens it picked. scripts/owner_check.py prints both.
pub(super) static DRAFT_VOCAB_ROWS: AtomicU64 = AtomicU64::new(0);
pub(super) static DRAFT_VOCAB_FULL: AtomicU64 = AtomicU64::new(0);
pub(super) static DRAFT_VOCAB_STEPS: AtomicU64 = AtomicU64::new(0);

pub fn set_draft_vocab(rows: usize, vocab: usize) {
    DRAFT_VOCAB_ROWS.store(rows as u64, Relaxed);
    DRAFT_VOCAB_FULL.store(vocab as u64, Relaxed);
}
pub fn draft_vocab_step() { DRAFT_VOCAB_STEPS.fetch_add(1, Relaxed); }
pub(super) static IN_FLIGHT: AtomicU64 = AtomicU64::new(0);
pub(super) static MAX_BATCH: AtomicU64 = AtomicU64::new(0);
/// F2: the `--max-batch` the command line asked for, BEFORE the EXL3 server's 16-lane cap. A number
/// here above `velogb10_max_batch` means the boot was capped (the boot log prints the line).
pub(super) static MAX_BATCH_REQUESTED: AtomicU64 = AtomicU64::new(0);
/// DSPLITK (owner 2026-11-07): trunk dense split-K attention launches (eager calls and graph captures; replays
/// of a captured graph are not counted). Absent = --exl3-dense-splitk off or no dense layer served.
pub(super) static EXL3_DENSE_SPLITK_CALLS: AtomicU64 = AtomicU64::new(0);
/// PFTC (owner 2026-11-07): prefill rows whose dense attention ran on the tensor-core kernels (eager prefill;
/// graph replays are not a prefill path). Absent = --exl3-dense-pf-tc off or no such layer served.
pub(super) static EXL3_PF_TC_ROWS: AtomicU64 = AtomicU64::new(0);
pub fn exl3_pf_tc_rows(n: u64) { EXL3_PF_TC_ROWS.fetch_add(n, Relaxed); }
pub fn exl3_dense_splitk_call() { EXL3_DENSE_SPLITK_CALLS.fetch_add(1, Relaxed); }
/// CTXCLAMP (owner 2026-11-07): an EXL3 TP = 1 boot whose asked --max-seq-len did not fit even one lane booted with one
/// lane at the longest context that fits; asked / served (0 / 0 = no context clamp on this boot).
pub(super) static EXL3_CTX_ASKED: AtomicU64 = AtomicU64::new(0);
pub(super) static EXL3_CTX_SERVED: AtomicU64 = AtomicU64::new(0);
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
pub(super) static OVER_CAP_QUEUED: AtomicU64 = AtomicU64::new(0);
/// MAXBATCH1: over-cap arrivals refused by the `--max-waiting` gate (503 `server_busy`).
pub(super) static OVER_CAP_REJECTED: AtomicU64 = AtomicU64::new(0);
/// TOPKM1: sampled requests on the batched GPU samplers (NVFP4/bf16 family) that asked for no
/// top-k truncation (0, or a negative value the server maps to 0) or a top_k above the kernels'
/// 64-candidate limit; served with top_k 64 (`crate::sampler::gpu_top_k`).
pub(super) static TOP_K_CAPPED: AtomicU64 = AtomicU64::new(0);

/// THINKBUDGET1: requests that carried `thinking_token_budget` and had it applied (the server's model
/// has reasoning markers and the serving path supports the budget).
pub(super) static THINK_BUDGET_APPLIED: AtomicU64 = AtomicU64::new(0);
/// THINKBUDGET1: requests that carried a budget the serving path could not apply (another model
/// family's scheduler, or a model without reasoning markers): served without it, said once per boot.
pub(super) static THINK_BUDGET_UNSUPPORTED: AtomicU64 = AtomicU64::new(0);
/// THINKBUDGET1: reasoning blocks the budget ended (the end-of-thinking marker forced).
pub(super) static THINK_BUDGET_FORCED: AtomicU64 = AtomicU64::new(0);

/// PR #30 port: requests that asked for log probabilities, by outcome (`logprobs_request`).
pub(super) static LOGPROBS_SERVED: AtomicU64 = AtomicU64::new(0);
pub(super) static LOGPROBS_UNSUPPORTED: AtomicU64 = AtomicU64::new(0);
pub(super) static LOGPROBS_WITHHELD: AtomicU64 = AtomicU64::new(0);
pub(super) static LOGPROBS_STREAM: AtomicU64 = AtomicU64::new(0);
/// PR #31 port: `response_format` json_schema requests: enforced (token mask armed) / unenforced (served with
/// `x-json-schema-enforced: none`: enforcement off, schema not compiled, engine or reasoning mode cannot apply it).
pub(super) static SCHEMA_ENFORCED: AtomicU64 = AtomicU64::new(0);
pub(super) static SCHEMA_UNENFORCED: AtomicU64 = AtomicU64::new(0);
/// PR #30 / #31 ports: EXL3 lanes that decoded WITHOUT speculation because the request asked for log probabilities
/// or a schema mask (the verify rows are not read back / masked yet): the speed those requests pay.
pub(super) static SPEC_OFF_LOGPROBS: AtomicU64 = AtomicU64::new(0);
pub(super) static SPEC_OFF_SCHEMA: AtomicU64 = AtomicU64::new(0);

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
pub(super) static LOGPROBS_SPEC_ROWS: AtomicU64 = AtomicU64::new(0);
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
