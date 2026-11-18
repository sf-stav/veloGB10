//! Continuous-batching scheduler for the live server.
//!
//! Owns the GPU. Incoming requests are prefilled into a free physical slot, then all active lanes
//! are decoded together via forward_decode (one shared weight read). Tokens stream back per request.
//! Each lane owns a physical slot (`Lane::phys`); the logical→physical map (`bufs.slot_ids_dev`) is
//! uploaded each step so stateful kernels index state by slot_ids[lane]. Finished lanes return their
//! slot to a free list — no state copying on compaction (slot indirection).
//!
//! File map (this is the former single `batch.rs`; every item keeps its `crate::batch::` path):
//! - `mod.rs`: shared types (BatchRequest, TokEvent, Lane, MtpPolicy, SpecSource, BatchScheduler struct) and constants.
//! - `mtp_policy.rs`: `impl MtpPolicy` (windowed depth/enable controller).
//! - `helpers.rs`: RNG domains, prefix compare, DF2 carry check, image identities, tree accept walk.
//! - `lifecycle.rs`: construction, setters, the run / TP head / TP mirror loops, `run_spec_bench`.
//! - `admission.rs`: `admit` and chunked prefill (`pf_advance`, `pf_finish`).
//! - `decode.rs`: `decode_step`, schema masks, penalty uploads, `batched_decode`.
//! - `mtp_forest.rs` / `mtp_lane.rs`: MTP tree, forest and single-lane steps; `copy_span.rs`: copy-span arm + diagnostics.
//! - `df2.rs` / `dspark.rs`: DFlash2 lanes; DFlash and DSpark lanes.
//! - `tree_accept_tests.rs`, `mtp_policy_tests.rs`, `vision_image_cache_tests.rs`, `df2_carry.rs`, `prehard1_serve_cap_tests.rs`, `loopfix_tests.rs`: the former inline unit-test modules (original module names kept).

mod admission;
mod copy_span;
mod decode;
mod df2;
mod dspark;
mod helpers;
mod lifecycle;
mod mtp_forest;
mod mtp_lane;
mod mtp_policy;

use tokio::sync::mpsc;

pub use self::helpers::*;
use self::mtp_policy::*;

use crate::gpu::{BatchGpuState, B, CudaGraph, DecodeBuffers, GpuModel, Pool};

// R9: a process-global handle to the live scheduler's (gpu, state) so net::agree — which every rank
// reaches on the mismatch sentinel — can dump the GDN recurrent state for the cross-rank diff.
static R9_STATE: std::sync::Mutex<Option<(usize, usize)>> = std::sync::Mutex::new(None); // (gpu_ptr, state_ptr)
/// Register the live gpu+state (raw addrs; the scheduler outlives the request). Called once at startup.
pub fn r9_register_state(gpu: &GpuModel, state: &BatchGpuState) {
    *R9_STATE.lock().unwrap() = Some((gpu as *const _ as usize, state as *const _ as usize));
}
/// Dump the GDN recurrent-state checksums for the live run (rank-tagged). No-op if unregistered.
pub fn r9_dump_gdn_state(rank: i32) {
    let g = R9_STATE.lock().unwrap();
    if let Some(&(g_, s_)) = g.as_ref() {
        let gpu = unsafe { &*(g_ as *const GpuModel) };
        let state = unsafe { &*(s_ as *const BatchGpuState) };
        gpu.dump_gdn_state_checksums(state, rank);
    }
}
// DevicePtr provides `.device_ptr()` on CudaSlice<T>; needed to read raw device pointers for the
// MTP KV buffers (passed as u64 bases into the stateful batched kernels).
use cudarc::driver::DevicePtr;

/// S5F — one job for the on-engine τ-matrix driver (`run_spec_bench`): the prompt + decode
/// parameters + the speculation source to serve it under (the driver switches sources between
/// jobs so the whole matrix runs on ONE model load).
#[derive(Clone)]
pub struct SpecBenchJob {
    pub prompt: Vec<u32>,
    pub max_new: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub source: SpecSource,
    /// S8F routing domain (drives the `DFlash2Auto` lane split in the bench harness too).
    pub domain: Domain,
}

/// A token streamed back to a request handler.
pub enum TokEvent {
    Tok(u32),
    Finish { reason: String },
    /// v0.7.3 (issue #8.2): sent ONCE per request, right after admission, carrying the
    /// number of prompt tokens the scheduler's prefix cache served (prefill skipped).
    /// Engines without a prefix cache never send it - handlers default the request's
    /// cached count to 0. Feeds `usage.prompt_tokens_details.cached_tokens` and
    /// `velogb10_prompt_tokens_cached_total`.
    Admitted { cached_tokens: u32 },
    /// S-B9-REL-SMALLFIXES item 5(a): sent at most once per request, right BEFORE its `Finish`,
    /// when the request ended on a model stop token and the scheduler could read that step's
    /// logits row (`exl3_forward::stop_token_stats`). `rank` = 1 + the number of tokens with a
    /// greater logit. `p` is a probability under one of TWO laws, and `sampled` says which:
    /// true = the law the sampler drew from (temperature-scaled, then top-k / min-p / top-p
    /// truncated and renormalized; 0 = the token was truncated away); false = a greedy request,
    /// where nothing is drawn: the plain softmax at temperature 1. The two are not comparable.
    /// Log-only: it feeds `stop_p` / `stop_rank` / `stop_p_law` of the `[gen]` line. An engine
    /// that never sends it logs `na` for all three.
    StopInfo { p: f32, rank: u32, sampled: bool },
    /// Sent right BEFORE the `Tok` it describes, only for a request that asked for log
    /// probabilities (`BatchRequest::logprobs`) on an engine that computes them: the natural log
    /// of the emitted token's probability and the `top` most likely alternatives (token id,
    /// logprob; most likely first). Computed by a log-softmax over the step's logits row after
    /// the repetition / presence / frequency penalties and any JSON-schema mask, before
    /// temperature and top-k / top-p / min-p truncation.
    Logprob { logprob: f32, top: Vec<(u32, f32)> },
}

/// A request submitted to the scheduler.
pub struct BatchRequest {
    pub prompt: Vec<u32>,
    pub max_new: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub rep_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    /// WP08: min-p truncation (0 = off). Honoured by the EXL3 serve path (xq_sample_rows); the
    /// NVFP4 constructors pass 0.0 (that sampler has no min-p).
    pub min_p: f32,
    pub min_new: usize,
    pub ignore_eos: bool,
    pub tx: crate::server::TokTx,
    pub seed: Option<u64>,
    /// Token index of the MESSAGE BOUNDARY — the prompt as rendered without the generation prompt.
    /// The scheduler snapshots the GDN recurrent state here, because this is the longest prefix of this
    /// prompt that the next turn will reproduce exactly. `None` = no checkpoint (raw / non-chat paths).
    pub ckpt_at: Option<usize>,
    /// S8F routing domain (S6F adjudication): drives the DF2 lane split (`greedy` on code, `rq` on
    /// math/chat/prose) under the default `DFlash2Auto` source. Classified from the rendered prompt
    /// by the server; default `General` everywhere else (bench jobs, TP wire reconstruction — where
    /// DF2 under TP is not yet wired, S9F).
    pub domain: Domain,
    /// When the request was created (server.rs stamps it at send; the TP wire reconstruction stamps
    /// it at into_request). Feeds the `[req] ttft=` log line — the server-side TTFT truth the
    /// measurement protocol asserts.
    pub received_at: std::time::Instant,
    /// V3 vision: merged image embeddings (concatenated, [sum(num_tokens), 5120] f32) + the spans
    /// (absolute positions in the EXPANDED prompt). None/empty for text-only traffic (unchanged).
    pub image_embeds: Option<Vec<f32>>,
    pub image_spans: Vec<crate::vision_encoder::ImageSpan>,
    /// W2 (Phase 13): the compiled JSON-schema FSM for this request (None = unconstrained).
    /// Compiled ONCE by the server from `response_format`; the scheduler advances it per committed
    /// token and hands the sampler a per-step vocabulary mask.
    pub schema: Option<std::sync::Arc<crate::json_schema::SchemaMask>>,
    /// THINKBUDGET1: vLLM's `thinking_token_budget` (None = unlimited / not sent): the most tokens
    /// the request may spend inside its reasoning block before the server forces the end marker.
    /// Enforced by the EXL3 server (crate::think_budget); other schedulers serve without it and
    /// say so once (velogb10_thinking_budget_requests_total{disposition="unsupported"}).
    pub thinking_budget: Option<usize>,
    /// Log probabilities for every emitted token: `Some(n)` = send a `TokEvent::Logprob` before
    /// each `Tok`, with the n (0..=20) most likely alternatives. `None` = none (no extra work).
    /// Honoured by the EXL3 serve path at TP=1 (such a lane decodes plain, no speculation).
    pub logprobs: Option<u8>,
}

struct Lane {
    phys: usize,
    pos: usize,
    last_tok: u32,
    max_new: usize,
    generated: usize,
    greedy: bool,
    /// S8F: the request's routing domain (DF2 lane split under `DFlash2Auto`).
    domain: Domain,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    rep_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    min_new: usize,
    ignore_eos: bool,
    history: Vec<u32>,
    /// LOOPFIX (2026-11-06): tokens ever pushed into `history` (monotonic) and how many of them the
    /// device's whole-output penalty counts already hold (`BatchScheduler::pen_sync`). `history` is
    /// trimmed, but never below the not-yet-counted tail (`hist_trim`).
    pen_total: u64,
    pen_synced: u64,
    /// W2 (Phase 13): the request's compiled JSON-schema FSM (None = unconstrained) and the
    /// machine's CURRENT state, advanced once per COMMITTED token.
    schema: Option<std::sync::Arc<crate::json_schema::SchemaMask>>,
    schema_state: u32,
    tx: crate::server::TokTx,
    /// MTP KV cursor: the main-model position of the last committed token = next MTP write pos - 1.
    /// Only meaningful when this lane is served via the MTP path.
    mtp_pos: usize,
    /// True iff this lane's MTP KV was primed over the prompt at admit. A lane admitted while the
    /// MTP policy was inactive has no primed MTP KV and must never take the MTP path.
    mtp_primed: bool,
    /// S5F: true iff this lane's DFlash2 ring was primed over the prompt at admit (the DFlash2
    /// path's analogue of `mtp_primed`). A lane admitted while DFlash2 was unavailable has no
    /// primed ring and must never take the DFlash2 path.
    df2_primed: bool,
    /// S5F: set the moment this lane takes a NON-DFlash2 step. Its DFlash2 ring is missing the
    /// taps for every token decoded since, so the round can never be trusted again for this
    /// request (the mirror of `mtp_stale`, same reasoning).
    df2_stale: bool,
    /// Set the moment this lane takes a NON-MTP step. Its MTP KV is now missing entries for every
    /// token decoded since, so the head can never be trusted again for this request.
    ///
    /// Dropping MTP mid-lane is safe (nothing but the MTP path reads that KV). RESUMING it is not,
    /// and the policy really can flip back on -- MtpPolicy retries after MTP_RETRY_AFTER steps. That
    /// would have restarted drafting against a KV with holes in it: the head would attend over
    /// never-written (zero) K/V rows, which still get exp(0)=1 weight in the softmax and quietly
    /// poison every draft. Output stays correct -- the verify rejects the garbage -- so the only
    /// symptom is acceptance silently collapsing, which is exactly the failure mode AGENTS.md warns
    /// no correctness gate can see. Once stale, always stale.
    mtp_stale: bool,
    /// Per-lane PRNG seed for stochastic MTP (LCG: seed = seed*1664525 + 1013904223, matching the
    /// LCG in sample_b/sample_prob_b/spec_verify_b). Host-side accept decisions and device-side
    /// seeds all derive from this single seed, advanced once per random draw.
    seed: u64,
    /// TP=2 serving (item A): a cancel delivered over the per-step wire protocol. On the head it is
    /// set by `run_tp_head` the moment it observes `tx.is_closed()` (and shipped to the mirror as a
    /// `TpEvent::Cancel`); on the mirror it is set when that event arrives. In TP serving mode this
    /// is the ONLY cancel channel — `decode_step`'s sweep must not consult `is_closed()` there, or a
    /// disconnect landing between the head's detection and its sweep would finish the lane one step
    /// early on the head only, desyncing the lockstep. Always false in single-node serving.
    tp_cancelled: bool,
    /// COPY27B (`--copy-span on`): this lane's copy proposer over prompt + committed tokens, built
    /// lazily at the lane's first lone speculative step; synced from `slot_cache` + `last_tok`.
    copy: Option<Box<crate::copy_span::CopySpan>>,
    /// COPY27B accounting: the lane is inside a `<think>` block (reasoning text) right now.
    in_reason: bool,
    /// COPY27B accounting: per phase (0 = reasoning, 1 = answer) x kind (0 = drafter round, 1 = copy
    /// round, 2 = plain / batched step): (rounds, tokens, microseconds); `depth[phase][d]` = drafter
    /// rounds that ran at MTP depth d. Printed once per request (`[copy-acct]`) at the lane's end.
    acct: [[(u64, u64, u64); 3]; 2],
    acct_depth: [[u64; MAX_AUTO_DEPTH + 1]; 2],
}

/// COPY27B accounting: the `<think>` / `</think>` ids of the served pack (0 = unknown: every request
/// counts as answer text). Set once at boot from the tokenizer.
static THINK_OPEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static THINK_CLOSE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static ACCT_REQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// COPY27B accounting: install the pack's think-tag ids (the server's boot).
pub fn set_think_ids(open: u32, close: u32) {
    THINK_OPEN.store(open, std::sync::atomic::Ordering::Relaxed);
    THINK_CLOSE.store(close, std::sync::atomic::Ordering::Relaxed);
}

/// COPY27B accounting: after `toks`, is the text inside a think block? (`state` = before them)
fn think_state_after(state: bool, toks: &[u32]) -> bool {
    let (o, c) = (THINK_OPEN.load(std::sync::atomic::Ordering::Relaxed), THINK_CLOSE.load(std::sync::atomic::Ordering::Relaxed));
    if o == 0 { return false; }
    let mut st = state;
    for &t in toks { if t == o { st = true; } else if t == c { st = false; } }
    st
}

/// COPY27B: which round a copy round replaces (it keeps that drafter's state live).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CopyMode {
    /// the MTP head's round (the head is re-primed over the accepted span)
    Mtp,
    /// the DFlash2 chain's round (the accepted span's taps are injected into the ring)
    Df2,
    /// no live drafter state (the lane decoded plain steps since admission: 2+ requests were
    /// batched, or speculation is off): the copy round only replaces a plain decode step; no
    /// drafter state is kept (the lane stays on plain steps otherwise)
    Bare,
}

/// LOOPFIX: how many of the oldest `history` entries `Lane::hist_trim` drops — the pre-LOOPFIX rule
/// (drop 128 once past 256) but never into the `pending` tail the device counts still lack.
fn hist_trim_cut(len: usize, pending: usize) -> usize {
    if len > 256 { 128usize.min(len.saturating_sub(pending)) } else { 0 }
}

/// LOOPFIX: the per-column extra tokens of a chain verify penalty (`make_penalty_drafts`), [cap][pe]
/// -1 padded: column c adds drafts[0..c] (what plain decode would have emitted before it); columns
/// past the verify width n = drafts + 1 repeat the deepest prefix.
fn pen_draft_extras(drafts: &[u32], cap: usize, pe: usize) -> Vec<i32> {
    let n = (drafts.len() + 1).min(cap);
    let mut extra = vec![-1i32; cap * pe];
    for c in 0..cap {
        let k = c.min(n - 1).min(pe);
        for j in 0..k { extra[c * pe + j] = drafts[j] as i32; }
    }
    extra
}

impl Lane {
    fn has_penalty(&self) -> bool {
        self.rep_penalty > 1.0 || self.presence_penalty > 0.0 || self.frequency_penalty > 0.0
    }
    /// Commit `t` to the lane's output history (every emit site goes through here).
    fn hist_push(&mut self, t: u32) {
        self.history.push(t);
        self.pen_total += 1;
        // A lane without a penalty never reads the device counts: keep it "synced" so `pen_sync`
        // has nothing to upload and the trim stays as before.
        if !self.has_penalty() { self.pen_synced = self.pen_total; }
    }
    /// Bound `history` (it also feeds the MTP head's draft-side window) exactly as before — drop the
    /// oldest 128 once it passes 256 — but never drop a token the whole-output counts lack.
    fn hist_trim(&mut self) {
        let cut = hist_trim_cut(self.history.len(), (self.pen_total - self.pen_synced) as usize);
        if cut > 0 { self.history.drain(0..cut); }
    }
    /// A lane runs on the MTP path iff its MTP KV was primed at admit AND the policy is currently
    /// active. Whether it then verifies greedily (bitwise lossless) or stochastically
    /// (distribution-exact) follows from `self.greedy` alone — that is a property of the REQUEST
    /// (temperature), never a server setting.
    ///
    /// The priming flag matters because the policy can flip mid-flight: a lane admitted while MTP was
    /// off has no primed MTP KV and must never take the MTP path, and a lane admitted while it was on
    /// can simply stop (its MTP KV is only ever read by the MTP path, so abandoning it is safe).
    fn use_mtp(&self, active: bool) -> bool {
        // W2: a schema-constrained lane may use the GREEDY MTP chain - its verify columns carry
        // the schema mask chain and the host clamps the accepted prefix to the schema-valid draft
        // length. The SAMPLED spec chain (rejection sampling over unmasked draft candidates) is
        // not — and a wrong-but-fast speculative emission is exactly the failure class W2 exists
        // wired for schemas: those lanes take the plain (masked) decode path. W2 v1.
        active && self.mtp_primed && !self.mtp_stale && (self.schema.is_none() || self.greedy)
    }
}

/// PLAN/SCHEDULER_2LANE_WORKDOC.md §3.1 — the per-lane PREFILL CURSOR.
///
/// Before this existed, `admit()` ran the whole chunked-prefill window loop INLINE on the scheduler
/// thread, so every window of a newly-admitted prompt was executed *before* that iteration's
/// `decode_step(b)`: a co-resident lane's next token waited for `ceil(plen / PREFILL_CHUNK)`
/// windows, measured at 25.4 s across a 16,800-token admission (P0, this session). The cursor holds
/// exactly what that loop's locals held, so the loop body could move VERBATIM (§3.2: the window
/// formula must not change — the pf8 padding family is width-dependent and its failures are found
/// by hitting an (n, bucket) gap in production, not by reasoning).
///
/// State machine (§3.4): `Prefilling{cursor}` → (lane install) → `Active`. A cursor is NOT a lane:
/// it is not in `self.lanes`, so it is served by nothing, is not counted in the decode `b`, and is
/// invisible to the prefix-cache matcher (its `slot_cache`/`slot_ring_len` writes happen at
/// COMPLETION, which is exactly where they happen today).
struct PfCursor {
    /// The physical slot reserved at admission (popped from `free_slots`).
    phys: usize,
    prompt: Vec<u32>,
    plen: usize,
    reuse: usize,
    w0: usize,
    ckpt_at: Option<usize>,
    grid_reuse: bool,
    req_imgs: Vec<ImageIdentity>,
    /// W2 (Phase 13): the request's compiled JSON-schema FSM, carried from `admit()` into the
    /// lane-install tail (`pf_finish`) so the lane starts in the machine's start state.
    schema: Option<std::sync::Arc<crate::json_schema::SchemaMask>>,
    /// Speculation decisions taken at ADMISSION (unchanged, §2: `will_use_df2`/`will_use_dspark`
    /// are decided once, never per window).
    will_use_mtp: bool,
    will_use_df2: bool,
    will_use_dspark: bool,
    df2_carry: bool,
    dspark_carry: bool,
    /// Window-loop state (moved verbatim).
    first_tok: u32,
    first_sent: bool,
    pf_hash_str: String,
    df2_primed_ok: bool,
    dspark_primed_ok: bool,
    /// Lane-install payload — `admit()`'s post-loop tail.
    tx: crate::server::TokTx,
    greedy: bool,
    domain: Domain,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    max_new: usize,
    min_new: usize,
    ignore_eos: bool,
    rep_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    seed: u64,
    /// Server-side TTFT accounting (`[req] ttft=`), stamped when the request was created.
    received_at: std::time::Instant,
    trace_pf: bool,
    admit_t0: std::time::Instant,
    t_memsets: f64,
    t_prefill: f64,
    t_prime: f64,
    /// V3 vision: the merged embeddings + spans for THIS request. Armed onto
    /// `state.vision_embeds` around each of this cursor's own windows and disarmed after them —
    /// the splice is a GLOBAL state field and a co-resident lane's window must never see it.
    vision: Option<(crate::gpu::B, Vec<crate::vision_encoder::ImageSpan>)>,
}

/// Whether MTP pays for itself is a measurable question, not a configuration one.
///
/// A depth-`d` MTP step emits `1 + (accepted drafts)` tokens and costs `r` decode-steps, so
///
/// ```text
/// speedup = (tokens per step) / r        =>       MTP pays iff  tokens_per_step > r
/// ```
///
/// `r` is a pure cost ratio — it depends on the model's shape (above all on what fraction of the
/// weights the LM head is, since drafting must read it a second time to pick a draft token) and not
/// on the prompt. So it is measured once at load. Acceptance, by contrast, is workload-dependent
/// (code accepts differently from prose), so it is tracked live and the decision is revisited.
///
/// This is what replaces `RUST_INFER_MTP` / `RUST_INFER_MTP_STOCHASTIC`. Those env vars encoded a
/// judgement ("MTP is good on 9B, bad on 2B") that the engine can simply measure — and that was wrong
/// the moment either the model or the workload changed.
pub struct MtpPolicy {
    head_present: bool,
    /// B8/G3: which DRAFTER this lane's speculation runs. DSpark is wired as a SELECTABLE source
    /// with MTP as the fallback: until the K-DSP kernels land the DSpark arm resolves to the MTP
    /// head (source stays recorded for telemetry + the agree hash), so the switch is exercised
    /// end-to-end today without serving from an absent drafter. Selection is per
    /// (domain, ctx-bucket) via `spec_source_for` — the ship-rule hook.
    spec_source: SpecSource,
    /// Explicit override from `--mtp=on|off`; `None` = auto (the default).
    force: Option<bool>,
    /// Pinned depth from `--mtp-depth`; `None` = the policy chooses.
    pin_depth: Option<usize>,
    depth: usize,
    /// r(d) = MTP step cost / decode step cost, MEASURED per depth per CONTEXT BUCKET by
    /// `calibrate_mtp_r` (E17): (measurement ctx, r table at that ctx). Bucket i covers
    /// (point[i-1]×2, point[i]×2]; the top point doubles as the asymptotic table. The verify's KV
    /// attention bytes grow ∝ context, so the optimal depth SHRINKS at long ctx — a single
    /// short-context table over-reached there.
    r_ctx: Vec<(usize, Vec<(usize, f32)>)>,
    last_ctx: usize,
    active: bool,
    // Rolling evaluation window.
    win_steps: u64,
    win_emitted: u64,
    win_drafts: u64,
    win_accepted: u64,
    /// Per-position CONDITIONAL acceptance: `hz[i]` = P(draft i accepted | drafts 1..i-1 accepted),
    /// as (accepted, offered) counts. This is the whole basis of the depth decision — see `yield_at`.
    hz: [(f64, f64); MAX_AUTO_DEPTH],
    decode_steps: u64,
    retry_at: u64,
    /// First window after (re)activation is SHORT: at TP=2 MTP usually loses (every speculative
    /// pass adds a cross-node all-reduce), so a weak head must be caught in ~32 steps, not 128
    /// (a 128-token request used to run its ENTIRE lifetime at the initial depth-4 before the
    /// policy could disable — measured 11.5 vs 16.4 tok/s wall).
    first_short: bool,
    /// COPY27B: with the copy arm on, every `explore_every`-th evaluation window runs at
    /// MAX_AUTO_DEPTH (0 = never, the default). Copy rounds take the copyable spans away from the
    /// head, so the hazards it measures come from the harder text alone; once the policy steps
    /// down, the deep positions are never offered again and their stale hazards keep it down when
    /// the text turns easy (measured: d=5 for the rest of a boot, agent-turn neural rounds 3.4
    /// tokens against 5.5 at d=8). The probe window refreshes them on the current text.
    explore_every: u64,
    windows: u64,
}

use crate::gpu::MAX_AUTO_DEPTH;

/// COPY27B: the MTP policy's exploration period (windows) while the copy arm is on.
const COPY_MTP_EXPLORE_EVERY: u64 = 8;
/// COPY27B: the evaluation window (drafter rounds) while the copy arm is on. Copy rounds serve most
/// rounds of copy-heavy requests and the window counts drafter rounds only, so the 128-round window
/// took a whole agent turn to climb back from a step-down (measured, --max-batch 4 greedy: an agent
/// turn after two code edits ran 207 of 251 drafter rounds at d4/d5 after the warm-up prose stepped
/// the policy down; 36.9 vs 43.5 tok/s with the flag off).
const COPY_MTP_EVAL_WINDOW: u64 = 32;

/// KV positions kept beyond the requested output: MTP draft/verify plus re-prime slop,
/// or one guard row for plain decode. (PR #4 — the admission contract, shared with the
/// server's context budget; see `admit` for dev's clamp-before-reject refinement.)
pub fn decode_headroom(mtp_active: bool) -> usize {
    if mtp_active { MAX_AUTO_DEPTH + 8 } else { 1 }
}

/// B8/G3 — the runtime speculation-source switch (PLAN/08 scheduler delta). `Mtp` is always
/// available; `Dspark` is the block drafter (falls back to MTP while K-DSP is unbuilt); `DFlash2`
/// is the S4F integrated round (S5F wiring; falls back to MTP when the artifact is absent/failed
/// — the standing directive, never a hard dependency); `Plain` = no speculation (plain decode).
/// The choice must be IDENTICAL on every rank — it rides the extended agree token via k_verify/
/// depth hashing (agree_ext), never a side channel.
///
/// S8F (S6F adjudication): `DFlash2Auto` is the DEFAULT source — DFlash2 with the per-request
/// lane split (greedy on code, real-q on math/chat/prose, S5F4's table). It resolves per lane in
/// the serving loop via `df2_effective_src`; MTP stays permanently selectable (`--spec-source
/// mtp`, the standing directive) and is the fallback whenever the round is absent.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpecSource { Mtp, Dspark, DFlash, DFlash2, DFlash2Rq, DFlash2Auto, DFlash2Tree, Plain }

/// PLAN/25 Phase 1 (tree-verify spike): the DF2 TREE lane — the selector round's candidate
/// table walked into a two-branch tree (the walk chain + the best per-level alternates),
/// verified with per-node KV paths and per-node target argmax (`verify_forward_topo`).
/// Greedy-only in v0; lossless by construction (emitted tokens are target argmax tokens).

/// S8F — the per-request routing domain for the DF2 lane split. `Code` routes the GREEDY selector
/// (S5F4's code cell: greedy τ 6.72 > rq 5.98); `General` (math/chat/prose) routes the real-q
/// sampled selector (rq wins or ties the other four cells). The domain only picks the DF2 LANE —
/// it can never change output correctness (greedy is lossless, rq is distribution-exact), only τ.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum Domain { #[default] General, Code }

/// S8F — classify a rendered prompt into the routing domain. Conservative: strong code markers
/// (≥2 distinct, or a code fence) → `Code`; everything else → `General`. A misclassification
/// costs a little τ, never correctness.
pub fn classify_domain(prompt: &str) -> Domain {
    let p = prompt.to_ascii_lowercase();
    const CODE_MARKERS: &[&str] = &[
        "def ", "class ", "import ", "from ", "func ", "fn ", "function ",
        "public ", "private ", "static ", "void ", "int ", "float ", "double ",
        "#include", "using namespace", "return ", "=>", "struct ", "impl ",
        "pub ", "let mut", "print(", "cout", "std::", "```",
    ];
    if p.contains("```") { return Domain::Code; }
    let hits = CODE_MARKERS.iter().filter(|m| p.contains(**m)).count();
    if hits >= 2 { Domain::Code } else { Domain::General }
}

/// S8F — resolve the effective DF2 lane for a source. `DFlash2Auto` maps `Code` → the greedy
/// selector (`DFlash2`) and `General` → the real-q selector (`DFlash2Rq`); every other source is
/// passed through unchanged (the explicit `--spec-source` values stay explicit).
///
/// P3(a) close (2026-08-23): with `prose_greedy` (the `--df2-prose-lane greedy-drafts` default),
/// GREEDY (temp-0) General requests take the greedy-draft lane and SAMPLED (temp>0) General
/// requests keep the real-q walk — the quad sweep's regime split, made structural: greedy-drafts
/// is +8-19% tau on every temp-0 prose cell (step-weighted +10.5%, step 61.3->58.0 ms) but
/// REGRESSES flat sampled targets (chat_t1_off @T1.0: 37.4->32.1 tok/s vs MTP, 1.010x->0.865x —
/// an argmax draft accepts at p(argmax), tiny under a flat T=1 nucleus). `req_greedy` is the
/// lane's own temp-0 flag, already SPMD-uniform (the dispatch branches on it identically on
/// every rank). `--df2-prose-lane rq` restores the unconditional real-q walk.
pub fn df2_effective_src(src: SpecSource, domain: Domain, prose_greedy: bool, req_greedy: bool) -> SpecSource {
    match src {
        SpecSource::DFlash2Auto => match domain {
            Domain::Code => SpecSource::DFlash2,
            Domain::General => if prose_greedy && req_greedy { SpecSource::DFlash2 } else { SpecSource::DFlash2Rq },
        },
        other => other,
    }
}

/// S8F — is this source a DFlash2 source (any of the three DF2 variants)?
pub fn is_df2_src(src: SpecSource) -> bool {
    matches!(src, SpecSource::DFlash2 | SpecSource::DFlash2Rq | SpecSource::DFlash2Auto
                 | SpecSource::DFlash2Tree)
}

/// CLI surface for `--spec-source {mtp,dflash2,dflash2-rq,none}` (S5F; S6F owns the per-domain
/// routing). `dflash2-rq` is the S5F2 L2 lane: the SAMPLED selector path verified under the
/// real-q rejection-sampling criterion (u·q < p) with the exact relu(p−q) residual.
impl SpecSource {
    pub fn from_cli(s: &str) -> Option<SpecSource> {
        match s {
            "mtp" => Some(SpecSource::Mtp),
            "dspark" => Some(SpecSource::Dspark),
            "dflash" | "dflash-v1" | "dflash1" => Some(SpecSource::DFlash),
            "dflash2" | "df2" => Some(SpecSource::DFlash2),
            "dflash2-rq" | "df2rq" => Some(SpecSource::DFlash2Rq),
            "dflash2-auto" | "df2-auto" | "df2auto" => Some(SpecSource::DFlash2Auto),
            "dflash2-tree" | "df2-tree" => Some(SpecSource::DFlash2Tree),
            "none" | "plain" | "off" => Some(SpecSource::Plain),
            _ => None,
        }
    }
    pub fn cli_name(&self) -> &'static str {
        match self {
            SpecSource::Mtp => "mtp",
            SpecSource::Dspark => "dspark",
            SpecSource::DFlash => "dflash",
            SpecSource::DFlash2 => "dflash2",
            SpecSource::DFlash2Rq => "dflash2-rq",
            SpecSource::DFlash2Auto => "dflash2-auto",
            SpecSource::DFlash2Tree => "dflash2-tree",
            SpecSource::Plain => "none",
        }
    }
}

// The lane owns a blocking CUDA stream (inside the drafter) and device buffers, exactly like
// `GpuModel` (see `unsafe impl Send for GpuModel`, src/gpu.rs). The lane is only ever touched from
// the scheduler's single task/thread — the same discipline the rest of the serving state uses.
unsafe impl Send for DflashLane {}

/// P14 — the DFlash v1 lane's resident state: the 6-layer block drafter, its incremental ctx KV
/// cache, its own device pool and the consumed ctx feature (one column per committed position).
///
/// The drafter is a single-sequence block model (no batch dim), so the lane is b == 1 only; it is
/// greedy-only (the block's argmax drafts are verified against the target's argmax — the MTP
/// chain's losslessness argument, unchanged).
pub struct DflashLane {
    pub draf: crate::dflash::DflashDrafter,
    pub kv: crate::dflash::DflashKv,
    pub pool: crate::gpu::Pool,
    /// `[nctx*h, kv_stride]` bf16: the conditioning columns consumed so far (absolute positions).
    pub feat: cudarc::driver::CudaSlice<half::bf16>,
    pub nctx: usize,
    pub block: usize,
    pub mask: u32,
    /// The span the NEXT draft appends: cache rows `[row0, row0+ncols)`.
    pub row0: usize,
    pub ncols: usize,
}

/// S5F — one recorded speculation step (the on-engine τ matrix telemetry). `drafts` = the
/// offered draft count (depth-1 for MTP, 7 for DFlash2), `nacc` = accepted prefix length,
/// `emitted` = tokens actually emitted (accepted + bonus, EOS/max_new may truncate). Timings:
/// `round_ms` = the draft-side GPU time (the DFlash2 round / the MTP draft chain),
/// `verify_ms` = the trunk verify, `step_ms` = the whole lane step (wall, includes rollback +
/// inject + host bookkeeping).
#[derive(Clone, Copy, Debug)]
pub struct SpecStepRec {
    pub greedy: bool,
    /// The absolute main-model position of the step's committed token (the lane's pos at step
    /// entry) — the harness's per-position acceptance cuts (early/late, inside/outside <think>).
    pub pos: u32,
    pub drafts: u32,
    pub nacc: u32,
    pub emitted: u32,
    pub round_ms: f32,
    pub verify_ms: f32,
    pub step_ms: f32,
}

/// B31 diagnostics: opt-in dispatch trace. Diagnostics-only env knob (AGENTS §7): when
/// --mtp-dispatch-trace is set, the admit → lane-install → decode-dispatch chain prints its
/// speculation decision per step. Never set by default; never changes any decision.
fn mtp_dispatch_trace() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static V: AtomicU8 = AtomicU8::new(0);
    match V.load(Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    let v = if crate::opts::var(crate::opt!("mtp-dispatch-trace")).is_ok() { 2 } else { 1 };
    V.store(v, Ordering::Relaxed);
    v == 2
}

/// Context buckets for the per-domain ship rule (PLAN/08 §validation: 4k/16k/32k/64k/128k).
pub const SPEC_CTX_BUCKETS: &[usize] = &[4096, 16384, 32768, 65536, 131072];

/// The per-bucket decision: DSpark only where the measured tau table (G2 output) clears the
/// parity thresholds; MTP everywhere else. The table ships as data — `dspark_tau_table` is the
/// measured artifact this function consults; empty table (pre-G2) means MTP everywhere, which
/// is exactly the shipped default.
pub fn spec_source_for(domain: &str, ctx: usize, dspark_tau_table: &[(String, usize, f32)]) -> SpecSource {
    // nearest ctx bucket with a measured tau
    let bctx = SPEC_CTX_BUCKETS.iter().copied()
        .min_by_key(|b| b.abs_diff(ctx)).unwrap_or(4096);
    for (d, c, tau) in dspark_tau_table {
        if d == domain && *c == bctx {
            // parity thresholds (PLAN/08): tau > 2.78 (FP8 drafter) / 3.55 (BF16) vs MTP;
            // the conservative shipped rule uses the BF16 threshold until the FP8 study exists.
            return if *tau > 3.55 { SpecSource::Dspark } else { SpecSource::Mtp };
        }
    }
    SpecSource::Mtp
}

/// MTP steps per policy re-evaluation.
/// Prefill window size. Prefill activation memory is O(window) (~1.2 MiB/token on 9B, more on 27B), so
/// a long prompt is prefilled in windows of this size to bound peak memory (a 256K single-shot prefill
/// would need ~400 GB on 27B). A prompt <= this is one window == the old single-shot path, byte-identical.
/// 8192 keeps typical prompts single-shot while bounding peak to ~13 GB/window on 27B.
/// pub(crate): the grouped-MoE prefill scratch in gpu.rs sizes itself off this bound (and asserts
/// batch×k against it at dispatch) — a window-size change must never silently overflow the scratch.
pub const PREFILL_CHUNK: usize = 8192;

/// F8 acceptance A6/B6 (2026-09-06): block-granular GDN checkpoints. The whole-slot cache
/// held the GDN state at exactly TWO points (live end + prompt boundary); tool-calling
/// clients re-render our turns, so the longest common prefix lands a median ~200 tokens
/// before the held state and EVERY turn re-prefilled the full transcript (edit-heavy cell:
/// 71 s of prefill per turn vs the recipe's ~0). RING checkpoints snapshot the GDN state at
/// RING_CKPT_STRIDE-token boundaries of the ABSOLUTE position into a per-slot ring of the
/// most-recent RING_CKPT_K entries — a new request resumes at the best boundary ≤ LCP and
/// prefills only the slack + divergent tail.
pub const RING_CKPT_STRIDE: usize = 512;
pub const RING_CKPT_K: usize = 16;

/// PREHARD1 [ISSUES 104]: the serve-path lane cap the trunk (NVFP4, non-EXL3) scheduler shares
/// with the EXL3 server (MAXBATCH1's `EXL3_MAX_LANES`). One authority: `EXL3_MAX_LANES` is this
/// constant. At most this many lanes serve at once; a larger `--max-batch` BOOTS ANYWAY at the cap
/// with one clear line, and further requests wait in the queue (owner ruling 2026-10-10 §1b-bis#8:
/// apply and show, never refuse).
pub const SERVE_MAX_LANES: usize = 16;

/// PREHARD1 [ISSUES 104]: how many of `busy + waiting` sit beyond what the lane cap can run at
/// once (pure; the delta of this between loop samples is the over-cap arrival count — a burst on
/// an idle box lands exactly like MAXBATCH1's soak counted it: arrivals minus the lanes that
/// could take them).
pub(crate) fn over_cap_excess(busy: usize, waiting: usize, cap: usize) -> usize {
    (busy + waiting).saturating_sub(cap)
}


/// S-B16: the lossless auditor's per-lane cursor. The shadow decodes the GROUND-TRUTH stream
/// (`gt_last` @ `gt_pos`, advancing on its own prediction — never on the MTP stream); `pred` is
/// its prediction for position gt_pos+1, i.e. exactly what the lane's next emitted token must
/// equal for the step to be lossless. `idx` counts the request's MTP-emitted tokens; `tag` is
/// the per-request id in the log line.
#[derive(Clone, Copy)]
struct MtpAuditCursor {
    gt_last: u32,
    gt_pos: usize,
    pred: u32,
    idx: usize,
    tag: u64,
    diverged: bool,
    /// False until the first audited step ran the zero+prime shadow decode.
    primed: bool,
}

pub struct BatchScheduler {
    gpu: GpuModel,
    pool: Pool,
    state: BatchGpuState,
    bufs: DecodeBuffers,
    graphs: std::collections::HashMap<usize, CudaGraph>,
    kv_stride: usize,
    /// EVERY token that ends an assistant turn. Not one token — a SET.
    ///
    /// Qwen3.5's config.json advertises `<|endoftext|>` while a chat turn actually ends with
    /// `<|im_end|>`. Stopping on only the advertised one let the model run past the end of its own
    /// turn and hallucinate the next one. See QwenTokenizer::stop_token_ids.
    eos: Vec<u32>,
    max_batch: usize,
    rx: mpsc::UnboundedReceiver<BatchRequest>,
    lanes: Vec<Option<Lane>>,
    /// PLAN/SCHEDULER_2LANE_WORKDOC.md §3.1 — the ONE in-flight prefill (the per-lane cursor).
    ///
    /// At most one request is ever mid-prefill, and that is a DELIBERATE v1 policy, not a shortcut:
    ///   * the shared draft-round objects (`df2`, `dspark`) are primed window-by-window and their
    ///     `reset`/`rewind`/identity claim are GLOBAL — two interleaved primes would clobber each
    ///     other's rings and degrade τ with no correctness signal at all (the exact class
    ///     PLAN/DSPARK_RING_IDENTITY_SPEC.md exists for). One cursor at a time makes that
    ///     impossible BY CONSTRUCTION, whereas today's atomic `admit()` only makes it unlikely;
    ///   * a new admission's `df2_carry`/`dspark_carry` decision reads the SHARED ring state, so
    ///     admitting while a prime is in flight would decide on half-written state.
    /// The measured win (§4 P0) does not need more than one: the stall is a co-resident DECODING
    /// lane waiting for a prefill, and only one prefill has to be in flight for that.
    pf: Option<Box<PfCursor>>,
    /// Free physical slots (stack) available for new admissions. Each active lane owns a physical
    /// slot (`Lane::phys`) holding its persistent KV + GDN state; on finish the slot is returned here
    /// instead of copying state into a contiguous prefix (slot indirection via `bufs.slot_ids_dev`).
    free_slots: Vec<usize>,
    /// PREFIX CACHE, one entry per PHYSICAL slot: the exact token sequence this slot's KV cache, GDN
    /// recurrent state and conv1d state currently reflect — i.e. the state is the state AFTER these
    /// tokens. Empty when the slot holds nothing reusable.
    ///
    /// Why per-slot and not a shared radix tree: KV is position-addressable, so any prefix of it stands
    /// on its own; THE GDN RECURRENT STATE IS NOT. It exists only at the single point in the sequence
    /// where the scan was left. A prefix is therefore only reusable from the slot that was carried to
    /// exactly that token and no further — which is precisely the append-only shape a chat or an agent
    /// transcript has anyway.
    slot_cache: Vec<Vec<u32>>,
    /// The token sequence at which each slot's PROMPT CHECKPOINT was taken — i.e. the whole prompt of
    /// the request that last ran in that slot, with the GDN state snapshotted at its final token
    /// (before any generation moved it).
    ///
    /// `slot_cache` alone gives a hit only when the client replays our generated tokens VERBATIM.
    /// A tool-calling agent never does: it re-renders our assistant turn from structured JSON, without
    /// the `<think>` block we actually emitted, so the sequence diverges a few tokens into our own
    /// reply. Measured on tool-eval-bench: 0 hits in 93 requests, and 88% of ALL prefill tokens were
    /// ones we had already computed — the match point sat a median of 202 tokens BEFORE the only GDN
    /// state we held. The prompt boundary is exactly where those misses want to resume, so we keep a
    /// second, older state there. KV needs no snapshot: it is position-addressable and still valid.
    slot_ckpt_seq: Vec<Vec<u32>>,
    /// Image-content identities (start + merged-embedding hash) inside each slot's `slot_cache` /
    /// `slot_ckpt_seq`. Parallel to those token sequences; the prefix-reuse matcher consults them so
    /// a changed image at a reused position breaks the reuse (see [`ImageIdentity`] / [`images_compatible`]).
    slot_cache_images: Vec<Vec<ImageIdentity>>,
    slot_ckpt_images: Vec<Vec<ImageIdentity>>,
    /// First of `max_batch` state slots holding the prompt checkpoints (lane i -> prompt_ckpt_slot + i).
    prompt_ckpt_slot: usize,
    /// First of `max_batch * RING_CKPT_K` state slots holding the periodic GDN ring
    /// checkpoints (lane i, entry j -> ring_ckpt_slot + i*RING_CKPT_K + j). Entry lengths
    /// live in `slot_ring_len`; 0 = invalid. States are pure functions of tokens [0, len),
    /// so entries with len ≤ the reused prefix stay valid across suffix re-prefills.
    ring_ckpt_slot: usize,
    /// Absolute lengths of the ring checkpoints per slot (RING_CKPT_K entries; 0 = invalid).
    slot_ring_len: Vec<Vec<usize>>,
    /// Round-robin write cursor per slot for the ring checkpoints.
    ring_cursor: Vec<usize>,
    /// For each slot: the sequence length up to which the DSPARK round's ring rows are valid
    /// (set when a dspark-primed lane finishes in this slot; 0 = none). A prefix-hit lane
    /// with reuse ≤ this carries the ring over (rewind nprev; prime only the suffix) instead
    /// of degrading to MTP — the edit-heavy τ lever.
    slot_dspark_len: Vec<usize>,
    /// Prompt-lookup draft n-gram order (0 = off). When the last `ngram_draft` tokens of a lane's
    /// context recur earlier, the follower is proposed as a draft instead of the 1-layer head's guess.
    /// Free (host-side) and lossless (the verify checks every draft); a big acceptance win on copyable
    /// tool/JSON text (60.3% -> 71.1% measured), a no-op on prose. See gpu.rs::bench_accept.
    ngram_draft: usize,
    /// Fork-then-chain tree drafting (opt-in). Lossless (verify checks every draft); the k=2 fork at
    /// position 1 rescues the chain-killing first-token miss. See mtp_tree_step.
    tree_draft: bool,
    /// Batched MTP verify across lanes (LANES design Step 3c): pack several concurrent lanes' draft
    /// chains into ONE forest verify. Opt-in; needs the full MAX_VERIFY checkpoint region.
    mtp_lanes: bool,
    /// P11 W5a: --prefill-sched inline — drain the whole prefill inside admission (the pre-P10
    /// schedule). Default false = the cursor schedule.
    pf_inline: bool,
    /// Reuse a cached prefix instead of prefilling it again. OFF by default, and that is a deliberate
    /// correctness choice, not caution: reusing a prefix RE-CHUNKS the prefill, and prefill runs on
    /// cuBLAS, which picks a different kernel per shape (AGENTS.md §2.4). So a cached turn is not
    /// bit-identical to a cold one — the same conversation can word its answer slightly differently
    /// depending on what happened to be in the cache. Every engine with prefix caching has this
    /// property; ours states it. The snapshot/restore ITSELF is bit-exact (tests/prompt_ckpt_test.rs).
    prefix_cache: bool,
    /// GPU-side sampling (temp/top-k/top-p/multinomial in `sample_b`) rather than a full-logit dtoh
    /// plus a CPU sampler. Now the default — every run script already set it, and pulling a
    /// 248k-entry logit vector to the host per token is strictly worse. `--cpu-sample=1`
    /// keeps the old path available as an escape hatch.
    gpu_sample: bool,
    /// Captured decode+sample graphs (per batch size) for the GPU-sampling path, when enabled.
    sample_graphs: std::collections::HashMap<usize, CudaGraph>,

    // ---- MTP (multi-token-prediction speculative decoding) ----
    /// Decides for itself whether MTP pays, from a measured cost ratio plus live acceptance.
    /// Greedy vs stochastic verify is NOT part of this — that follows from the request's temperature.
    mtp: MtpPolicy,
    /// Per-physical-slot MTP KV cache: `[nkv, kv_stride, hd]` bf16 each (packed `[nkv, kv_stride,
    /// hd/16×9]` bytes under the 4-bit KV cache). Indexed by `Lane::phys`. Empty when MTP is disabled.
    mtp_kc: Vec<B>,
    mtp_vc: Vec<B>,
    /// Per-physical-slot MTP hidden cursor `[h]` bf16: the pre-norm backbone hidden at the last
    /// committed position (seeds the next draft chain). Empty when MTP is disabled.
    mtp_h_prev: Vec<B>,
    /// Shared (single active MTP lane at a time) scratch buffers, allocated when MTP is enabled.
    mtp_h_save: Option<B>,      // `[h]`: snapshot of h_prev before a step (for post-accept re-prime)
    mtp_h_scratch: Option<B>,   // `[h]`: hidden-column extract scratch
    mtp_cur_hidden: Option<B>,  // `[h]`: draft-chain cursor hidden
    /// All-zero slot ids for the MTP attention (its KV is per-lane, so the slot is always 0).
    /// Sized for the widest MTP forward, NOT for max_batch: the MTP head runs at batch = depth
    /// Reserved physical slot (never assigned to a lane) used as the GDN-rollback snapshot target.
    /// `copy_gdn_slot(state, phys, snapshot_slot)` snapshots; the reverse restores on partial reject.
    mtp_snapshot_slot: usize,
    /// S-B16: env-gated (`--mtp-lossless-audit=1`) in-server lossless auditor — the bench's
    /// lockstep shadow decode moved into the serving path. One extra plain decode forward per
    /// emitted token; DIAGNOSTIC ONLY (halves serving throughput while on, one bool when off).
    /// Refused under TP>1: the shadow decode joins the SPMD collectives, and an env var does not
    /// ride TpConfig to the nodes — an unaudited node would desync the lockstep.
    audit_on: bool,
    /// The auditor's state slot: one extra slot appended after the normal band (valid iff audit_on).
    audit_slot: usize,
    /// Shadow decode buffers (batch 2, slot_ids preloaded with [audit_slot, 0]).
    audit_bufs: Option<DecodeBuffers>,
    /// Per-lane (batch index) audit cursor, armed at pf_finish with the request's install state.
    audit_cur: Vec<Option<MtpAuditCursor>>,
    audit_req_n: u64,
    /// Persistent penalty buffers for the MTP verify path (depth positions). Filled per-step from
    /// the lane's committed history so greedy MTP lanes keep their repetition/presence/frequency
    /// penalty (no repetition). Stored on the compute stream's device.
    mtp_pen_slots: Option<cudarc::driver::CudaSlice<i32>>,   // [MAX_VERIFY] (LOOPFIX: column -> lane slot)
    mtp_pen_extra: Option<cudarc::driver::CudaSlice<i32>>,   // [MAX_VERIFY * PEN_EXTRA] (preceding drafts)
    mtp_pen_rep: Option<cudarc::driver::CudaSlice<f32>>,     // [depth]
    mtp_pen_presence: Option<cudarc::driver::CudaSlice<f32>>, // [depth]
    mtp_pen_freq: Option<cudarc::driver::CudaSlice<f32>>,    // [depth]
    /// Persistent penalty buffers for the MTP DRAFT path (1 column, for the MTP head before
    /// sampling). Like mtp_pen_* but sized for batch=1.
    mtp_draft_pen_tokens: Option<cudarc::driver::CudaSlice<i32>>,  // [MAX_PEN_TOKENS]
    mtp_draft_pen_counts: Option<cudarc::driver::CudaSlice<i16>>,  // [MAX_PEN_TOKENS]
    mtp_draft_pen_rep: Option<cudarc::driver::CudaSlice<f32>>,     // [1]
    mtp_draft_pen_presence: Option<cudarc::driver::CudaSlice<f32>>, // [1]
    mtp_draft_pen_freq: Option<cudarc::driver::CudaSlice<f32>>,    // [1]
    /// Cache key for the CONSTANT penalty values (rep, presence, freq bit patterns + width): the
    /// penalty VALUES are request-constant, so re-uploading them every MTP step is 3 blocking
    /// copies of pure waste. Re-upload only when the key changes (a request with different
    /// penalties arrives). The token/count history still uploads per step — it changes every step.
    pen_const_key: Option<(u32, u32, u32, usize)>,
    /// Whether the last batched_decode step uploaded penalty data. Guards the five penalty-array
    /// uploads: skipped entirely when no lane is penalized (the kernel skips -1 sentinels), with a
    /// one-shot clear when the last penalized lane departs so its values cannot linger.
    pen_had: bool,
    // ---- S5F: the DFlash2 speculation source (the S4F integrated round, b==1 only) ----
    /// The DFlash2 round (S4F): drafter weights + ring KV + the block pass. `None` = absent or
    /// failed artifact → the source falls back to the MTP path (standing directive).
    /// P14 — the DFlash v1 lane (greedy block drafter); present only under `--spec-source dflash`.
    dflash: Option<DflashLane>,
    df2: Option<crate::dflash2::round::Df2Round>,
    /// The trunk's tap-capture sink writer twin (the round reads the staging via attach_sink).
    df2_sink: Option<std::sync::Arc<crate::dflash2::capture::Df2TapSink>>,
    /// PLAN/25 Phase 1: the WIDE tree-verify sink (armed on the trunk when the source is
    /// `dflash2-tree`; the topo verify's n > BLOCK taps land here).
    df2_sink_tree: Option<std::sync::Arc<crate::dflash2::capture::Df2TapSink>>,
    /// The prefill window's wide tap buffer (the prompt-prime capture target).
    df2_prime: Option<std::sync::Arc<crate::dflash2::capture::Df2PrimeSink>>,
    /// WI1: the DSpark round (Qwen3.8-27B-DSpark drafter, `--spec-source dspark`). `None` =
    /// absent/failed artifact, so the standing MTP fallback applies, never a hard failure.
    dspark: Option<crate::dspark::round::DsparkRound>,
    /// The DSpark lane's tap-capture sink twin (dspark::TAP_LAYERS arms on the trunk).
    dspark_sink: Option<std::sync::Arc<crate::dflash2::capture::Df2TapSink>>,
    /// The DSpark lane's prompt-prime wide sink (the prefill window's taps).
    dspark_prime: Option<std::sync::Arc<crate::dflash2::capture::Df2PrimeSink>>,
    /// One-time "DFlash2 requested but unavailable → serving via MTP" log (the fallback proof).
    df2_fallback_logged: bool,
    /// P14 — one-shot notices for the v1 lane (missing artifact / unprimed request).
    dflash_fallback_logged: bool,
    /// P14 — the admit path armed the ctx-feature prime for the (single) window loop to consume.
    dflash_prime_armed: bool,
    /// P14 — the ctx prime for the request in flight completed (see the prefill hook).
    dflash_prime_done: bool,
    dflash_carry_logged: bool,
    /// One-time "DSpark requested but unavailable -> serving via MTP" log (the fallback proof).
    dspark_fallback_logged: bool,
    /// WI1 DSpark lane telemetry (the df2 counters' twins; the [dspark] step log's source).
    dspark_stat_steps: u64,
    dspark_stat_drafts: u64,
    dspark_stat_accepted: u64,
    dspark_stat_emitted: u64,
    /// FIX1 #76: dspark publish watermark (same contract as the DF2 chain).
    dspark_stat_published: u64,
    /// P3(a) close: route GREEDY (temp-0) General requests to GREEDY drafts — the DEFAULT since
    /// the 2026-08-23 quad temp-0 sweep (prose τ +10.5% step-weighted, code control
    /// bit-identical). Sampled-temp General keeps the real-q walk. `--df2-prose-lane rq`
    /// restores the unconditional sampled real-q selector walk. Resolves in
    /// `df2_effective_src`; affects only the `DFlash2Auto` source's General domain (explicit
    /// `--spec-source` values stay explicit).
    prose_lane_greedy: bool,
    /// DF2_CARRY: keep DFlash2 in the draft seat across a PREFIX-CACHE HIT. Default OFF until
    /// the `PLAN/DF2_CARRY_SPEC.md` §7 acceptance run is green.
    ///
    /// Today `will_use_df2` requires `reuse == 0`, so every cache hit degrades to MTP — on the
    /// owner's tool-eval workload that is 184 of 197 requests (DF2 250 steps vs MTP 9 500).
    /// With this ON the lane carries the round's ring (0 extra bytes) and primes only the suffix,
    /// exactly as DSpark's A6 carry does. The guard is NOT DSpark's length-only check: the DF2
    /// ring is a modulo ring, so `df2_carry` also requires the frontier to lie inside the new
    /// prompt (see the admit site).
    df2_carry_enabled: bool,
    /// S5F3 draft-parity step dump (dump-only; None = the standing path, zero overhead).
    step_dump: Option<crate::dflash2::stepdump::StepDump>,
    /// PLAN/25 Phase 0: run the coverage-trace OP SEQUENCE (eager keep-logits verify, logging
    /// MTP chain, per-level top-k GEMMs) even where `step_dump` is None. SPMD-critical under TP:
    /// the node mirror must execute the identical op/collective sequence as the head, so the
    /// trace flag rides TpConfig to every rank while only the head's `step_dump` writes records.
    cov_trace: bool,
    /// PLAN/25 Phase 0: serving-side dump-job counter (reqNNNN tags).
    cov_req_n: usize,
    // ---- Live MTP acceptance telemetry (stderr, every N MTP lane-steps) ----
    mtp_stat_steps: u64,     // number of mtp_lane_step invocations
    mtp_stat_drafts: u64,    // total draft tokens proposed (depth-1 per step)
    mtp_stat_accepted: u64,  // total drafts accepted (matched verify argmax)
    mtp_stat_emitted: u64,   // total tokens emitted via the MTP path (accepted drafts + bonuses)
    mtp_stat_verify_fwds: u64, // total main-model verify+reverify forwards (cost)
    // B31: last mtp_stat_steps value handed to tel::publish_mtp — lets decode_step republish the
    // live counters every step without re-publishing on plain-only steps.
    mtp_stat_published: u64,
    // ---- S5F: DFlash2 lane telemetry (the MTP policy's curve stays MTP-only) ----
    df2_stat_steps: u64,
    df2_stat_drafts: u64,
    df2_stat_accepted: u64,
    df2_stat_emitted: u64,
    /// FIX1 #76: last df2_stat_steps handed to tel::publish_df2 (B31's watermark, applied to the
    /// DF2 chain): publish every step, log every 50. A short request must still light /metrics.
    df2_stat_published: u64,
    df2_tree_stat_steps: u64,   // df2_tree_step invocations
    df2_tree_stat_rescues: u64, // accepted path crossed into the B branch (the tree's whole point)
    df2_tree_stat_nodes: u64,   // verified tree nodes (cost side: the chain would be depth+1)
    /// FIX1 #76: the engagement side of the tree (drafts/accepted/emitted were always known, the
    /// lane just never published them) + the publish watermark.
    df2_tree_stat_drafts: u64,
    df2_tree_stat_accepted: u64,
    df2_tree_stat_emitted: u64,
    df2_tree_stat_published: u64,
    // ---- P14: DFlash v1 lane telemetry (block drafter; the lane was INVISIBLE at /health) ----
    dflash_stat_steps: u64,     // dflash_lane_step invocations
    dflash_stat_drafts: u64,    // block-1 proposals per step
    dflash_stat_accepted: u64,  // matched verify argmax
    dflash_stat_emitted: u64,   // emitted tokens (accepted drafts + the bonus)
    /// accept@k for the BLOCK lane: `dflash_acc_n[k]` = steps that reached column k,
    /// `dflash_acc_a[k]` = steps where column k was accepted. len = MAX_BLOCK (16), not tel::MAXK
    /// (8, the MTP depth ceiling) — the block drafter proposes up to 15 columns.
    dflash_acc_a: [u64; crate::dflash::MAX_BLOCK],
    dflash_acc_n: [u64; crate::dflash::MAX_BLOCK],
    // ---- S5F: per-step speculation recorder (the on-engine τ matrix harness) ----
    /// When `spec_steps_on`, every speculation step (MTP or DFlash2 lane) pushes a record.
    spec_steps: Vec<SpecStepRec>,
    spec_steps_on: bool,
    /// Option-gated (`--mtp-draft-log <path>`) JSONL log of every chain-MTP step:
    /// `{step, lane, pos, committed, drafts, preds, nacc}` (preds = verify argmax per column; empty
    /// on the stochastic path where the verify samples instead). This is the engine-side reference
    /// the head-finetune runbook's B0 parity gate diffs the HF MTP module against. Log-only.
    mtp_draft_log: Option<std::fs::File>,
    /// Option-gated (`--mtp-curve-file <path>`) JSON dump of the cumulative accept-by-depth curve
    /// (`MtpPolicy::hazard_counts`), overwritten every 50 MTP steps — the runbook §0 baseline.
    mtp_curve_path: Option<String>,
    /// TP=2 serving (item A): set by `run_tp_head` / `run_tp_mirror`. Gates the cancel sweep in
    /// `decode_step` to wire-delivered cancels ONLY (`Lane::tp_cancelled`) — see that field.
    tp_serving: bool,
    /// Device-resident token loop (--device-loop): when true, the NEXT batched_decode must
    /// re-upload tokens/pos/slot_ids/ring/keys (lane composition, MTP step, or param change
    /// invalidated the device-resident state). Clean steps upload nothing.
    resident_dirty: bool,
    /// COPY27B (`--copy-span on`, default off): the armed copy arm (proposer config + width cap);
    /// None = the arm is off and no copy code runs.
    copy_arm: Option<crate::exl3_serve::CopyArm>,
}

#[cfg(test)]
mod tree_accept_tests;

#[cfg(test)]
mod mtp_policy_tests;

/// Cross-image-contamination: the slot-level prefix reuse must be keyed on image CONTENT, not just
/// token identity. Two images of the same resolution expand to identical `image_pad` runs, so a
/// token-only key would replay image N's spliced visual content for image N+1. These test the pure
/// host-side keying helpers (the GPU splice offset itself is covered by the end-to-end gate).
#[cfg(test)]
mod vision_image_cache_tests;

/// DF2_CARRY guard tests (PLAN/DF2_CARRY_SPEC.md §5/§7). These are GPU-free: they pin the
/// decision arithmetic, including the two failure modes that would otherwise be invisible
/// (a stale ring from another slot, and a frontier past the new prompt — both of which keep
/// every losslessness gate green while tau silently falls).
#[cfg(test)]
mod df2_carry;

/// PREHARD1 [ISSUES 104]: the trunk serve cap mirrors MAXBATCH1's EXL3 cap — same constant, same
/// queue semantics, wired at the trunk parse site and both trunk serve loops.
#[cfg(test)]
mod prehard1_serve_cap_tests;

#[cfg(test)]
mod loopfix_tests;
