//! Continuous-batching scheduler for the live server.
//!
//! Owns the GPU. Incoming requests are prefilled into a free physical slot, then all active lanes
//! are decoded together via forward_decode (one shared weight read). Tokens stream back per request.
//! Each lane owns a physical slot (`Lane::phys`); the logical→physical map (`bufs.slot_ids_dev`) is
//! uploaded each step so stateful kernels index state by slot_ids[lane]. Finished lanes return their
//! slot to a free list — no state copying on compaction (slot indirection).

use tokio::sync::mpsc;

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
    pub tx: mpsc::UnboundedSender<TokEvent>,
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
    /// W2 (Phase 13): the request's compiled JSON-schema FSM (None = unconstrained) and the
    /// machine's CURRENT state, advanced once per COMMITTED token.
    schema: Option<std::sync::Arc<crate::json_schema::SchemaMask>>,
    schema_state: u32,
    tx: mpsc::UnboundedSender<TokEvent>,
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
}

impl Lane {
    fn has_penalty(&self) -> bool {
        self.rep_penalty > 1.0 || self.presence_penalty > 0.0 || self.frequency_penalty > 0.0
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
    tx: mpsc::UnboundedSender<TokEvent>,
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
}

use crate::gpu::MAX_AUTO_DEPTH;

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

const MTP_EVAL_WINDOW: u64 = 128;
/// First window after activation (and after each re-probe): catch a losing head quickly.
const MTP_EVAL_FIRST: u64 = 32;
/// Decode steps to wait before re-probing a model whose acceptance had fallen below break-even
/// (the workload may have changed — acceptance is not a fixed property of the model).
const MTP_RETRY_AFTER: u64 = 4096;
/// A different depth must beat the current one by this factor to be worth switching to.
const MTP_DEPTH_MARGIN: f32 = 1.05;

/// Expected tokens emitted by one depth-`d` step: `1 + Σ_{i=1..d-1} Π_{j≤i} p_j`.
///
/// **The hazard is NOT constant across positions, and assuming it is was a real bug.** A single `p`
/// fitted at depth 2 is just the FIRST-position hazard — the easiest one — and `p^k` then wildly
/// over-predicts deep yield. Measured on 9B prose: p₁ ≈ 0.83, which predicts 3.94 tokens/step at
/// depth 6; the actual yield at depth 6 is 2.64. The policy believed the model, jumped to depth 6,
/// and sat there ~20% below the optimum (depth 4). Hazards decay because each draft is conditioned
/// on a chain of its own guesses.
///
/// So: measure `p_i` per position, and for positions deeper than anything observed, extrapolate with
/// the LAST observed hazard (the most pessimistic one seen) rather than the first. A depth we have
/// never run is a guess either way — it should be a conservative guess, because the cost of
/// over-reaching (a whole window at a bad depth) is real and the cost of under-reaching is that the
/// next window discovers the truth and goes deeper.
fn yield_at(hz: &[(f64, f64); MAX_AUTO_DEPTH], d: usize) -> f32 {
    const MIN_OBS: f64 = 8.0;                 // below this a hazard is noise, not a measurement
    let mut last = 0.5f64;                    // prior for positions never offered a draft
    let mut acc = 1.0f64;                     // the bonus token, always emitted
    let mut chain = 1.0f64;
    for i in 1..d {
        let p = match hz.get(i - 1) {
            Some(&(a, n)) if n >= MIN_OBS => { last = a / n; last }
            _ => last,                        // never observed this deep: carry the last known hazard
        };
        chain *= p;
        acc += chain;
    }
    acc as f32
}

/// Render the live accept-by-depth curve for the `[mtp]` stats line: one `pos:rate` per observed
/// position (rate = P(draft accepted | earlier drafts accepted); `?` where too few observations).
/// This is the §0 GO/NO-GO signal at a glance — a sharp drop from @1 to @2+ is exactly the gap the
/// chained head-finetune recovers.
fn fmt_accept_by_depth(mtp: &MtpPolicy) -> String {
    mtp.hazard_counts().iter().enumerate().map(|(i, &(a, n))| {
        if n >= 8 { format!("@{}:{:.0}%", i + 1, a as f64 / n as f64 * 100.0) }
        else { format!("@{}:?", i + 1) }
    }).collect::<Vec<_>>().join(" ")
}

impl MtpPolicy {
    pub fn new(head_present: bool, force: Option<bool>, pin_depth: Option<usize>,
               r_ctx: Vec<(usize, Vec<(usize, f32)>)>) -> Self {
        MtpPolicy::with_source(head_present, force, pin_depth, r_ctx, SpecSource::Mtp)
    }

    /// S5F: `new` + an explicit speculation source (`--spec-source`). Default routing stays MTP;
    /// the source is selectable per process (S6F owns per-domain routing).
    pub fn with_source(head_present: bool, force: Option<bool>, pin_depth: Option<usize>,
                       r_ctx: Vec<(usize, Vec<(usize, f32)>)>, spec_source: SpecSource) -> Self {
        // Start ON when auto: MTP is correctness-neutral either way (greedy is bitwise lossless,
        // stochastic is distribution-exact), so the worst case of guessing wrong is a slightly slow
        // first window, which the evaluation below then corrects.
        let active = head_present && force.unwrap_or(true);
        // Open in the MIDDLE of the range, not at 2. The policy learns a per-position hazard curve,
        // and at depth 2 it only ever observes position 1 — so it would have to extrapolate the whole
        // curve from its easiest point, which is exactly how it used to over-reach. Starting at 4
        // gives the first window three positions of real data to reason from.
        let depth = pin_depth.unwrap_or(4).clamp(2, MAX_AUTO_DEPTH);
        Self { head_present, spec_source, force, pin_depth, depth, r_ctx, last_ctx: 0, active,
               win_steps: 0, win_emitted: 0, win_drafts: 0, win_accepted: 0,
               hz: [(0.0, 0.0); MAX_AUTO_DEPTH], decode_steps: 0, retry_at: 0,
               first_short: true }
    }
    pub fn active(&self) -> bool { self.active }
    /// B8/G3: runtime switch (per lane-admission or policy window). Records the source and
    /// returns whether the DRAFT pass should run the block drafter this step. The serving loop
    /// consults this at admission + every eval window; the decision joins the agree hash.
    pub fn set_spec_source(&mut self, src: SpecSource) {
        self.spec_source = src;
    }
    pub fn spec_source(&self) -> SpecSource { self.spec_source }
    /// True when the block drafter should own this step's draft. ALWAYS falls back to the MTP
    /// head when the DSpark model is not resident (the loader sets `dspark_present` false until
    /// G4's artifact is bound); the switch is exercised but never serves from nothing.
    pub fn use_dspark(&self, dspark_present: bool) -> bool {
        self.spec_source == SpecSource::Dspark && dspark_present
    }
    /// S5F: true when the DFlash2 round should own this step's draft. ALWAYS falls back to the
    /// MTP path when the DFlash2 round is not resident (`df2_present` false — absent or failed
    /// artifact: the standing directive's fallback, never a hard failure). b==1 only (the decode
    /// loop enforces it; AGENTS §4).
    pub fn use_dflash2(&self, df2_present: bool) -> bool {
        is_df2_src(self.spec_source) && df2_present
    }
    pub fn depth(&self) -> usize { self.depth }
    pub fn head_present(&self) -> bool { self.head_present }
    /// Cumulative per-position conditional acceptance counts (accepted, offered), truncated to the
    /// deepest position ever offered a draft. `hz[i]` = P(draft i+1 accepted | drafts 1..i accepted)
    /// — this IS the accept-by-depth curve the MTP head-finetune GO/NO-GO check needs. Never reset,
    /// so it integrates over the whole server run.
    pub fn hazard_counts(&self) -> Vec<(u64, u64)> {
        let mut v: Vec<(u64, u64)> = self.hz.iter().map(|&(a, n)| (a as u64, n as u64)).collect();
        while v.last().map_or(false, |&(_, n)| n == 0) { v.pop(); }
        v
    }
    pub fn r(&self) -> f32 { self.r_at(self.depth, self.last_ctx) }
    /// The r(d) table for context `ctx`: the first bucket whose point×2 covers it, else the top
    /// (asymptotic) bucket. Empty table (calibration failed) → every r reads as INFINITY → the
    /// policy disables itself, exactly as before.
    fn bucket_table(&self, ctx: usize) -> &[(usize, f32)] {
        if self.r_ctx.is_empty() { return &[]; }
        let i = self.r_ctx.iter().position(|&(p, _)| ctx <= p * 2)
            .unwrap_or(self.r_ctx.len() - 1);
        &self.r_ctx[i].1
    }
    fn r_at(&self, d: usize, ctx: usize) -> f32 {
        self.bucket_table(ctx).iter().find(|&&(x, _)| x == d).map(|&(_, r)| r).unwrap_or(f32::INFINITY)
    }
    /// Acceptance at which MTP breaks even, for reporting: tokens_per_step must exceed `r`.
    pub fn break_even_accept(&self) -> f32 {
        ((self.r() - 1.0) / (self.depth.max(2) - 1) as f32).clamp(0.0, 1.0)
    }

    /// Record one completed MTP step. `accepted` is the accepted PREFIX length: drafts 1..=accepted
    /// were taken and draft `accepted+1` (if there was one) was rejected. That is exactly the
    /// information a per-position hazard needs.
    fn record_step(&mut self, drafts: u64, accepted: u64, emitted: u64) {
        self.win_steps += 1;
        self.win_drafts += drafts;
        self.win_accepted += accepted;
        self.win_emitted += emitted;
        for i in 0..(accepted as usize).min(MAX_AUTO_DEPTH) {
            self.hz[i].0 += 1.0;
            self.hz[i].1 += 1.0;
        }
        // The first rejected position was offered and refused — that is the observation that makes
        // the hazard a hazard rather than an average.
        if (accepted as usize) < drafts as usize {
            if let Some(e) = self.hz.get_mut(accepted as usize) { e.1 += 1.0; }
        }
    }

    /// Called once per decode step with the deepest live context (max lane pos). Re-evaluates the
    /// decision when a window completes, and re-probes a disabled model after a cooldown.
    fn tick(&mut self, ctx: usize) {
        self.decode_steps += 1;
        self.last_ctx = ctx;
        if self.force.is_some() || !self.head_present {
            if mtp_dispatch_trace() && self.decode_steps <= 3 {
                eprintln!("[b31-trace] tick: no-eval (force={:?} head={} depth={} ctx={})",
                          self.force, self.head_present, self.depth, ctx);
            }
            return;
        }

        if !self.active {
            if self.decode_steps >= self.retry_at {
                self.active = true;
                self.first_short = true;
                self.win_steps = 0; self.win_emitted = 0; self.win_drafts = 0; self.win_accepted = 0;
                eprintln!("[mtp] re-probing (workload may have changed)");
            }
            return;
        }
        let need = if self.first_short { MTP_EVAL_FIRST } else { MTP_EVAL_WINDOW };
        if self.win_steps < need { return; }
        self.first_short = false;

        let observed = self.win_emitted as f32 / self.win_steps as f32;
        let acc = self.win_accepted as f32 / self.win_drafts.max(1) as f32;
        self.win_steps = 0; self.win_emitted = 0; self.win_drafts = 0; self.win_accepted = 0;

        // Re-pick the depth FROM THE CURRENT CONTEXT'S BUCKET (E17): a step buys `yield_at(d)`
        // tokens for r(d) decode steps, so maximise the ratio. r(d) grows with context (the verify's
        // KV bytes), so the same acceptance curve yields a shallower optimum at ≥16K than at 2K.
        let cur = yield_at(&self.hz, self.depth) / self.r_at(self.depth, ctx);
        let (mut best_d, mut best) = (self.depth, cur);
        if self.pin_depth.is_none() {
            let table = self.bucket_table(ctx).to_vec();
            for &(d, r) in &table {
                if r <= 0.0 || !r.is_finite() || d == self.depth { continue; }
                let s = yield_at(&self.hz, d) / r;
                // Hysteresis, ASYMMETRIC (PLAN/10 #12, 2026-08-17): an UP-switch still needs the
                // real margin — adjacent depths score within a fraction of a percent and flapping
                // 4->5->4 costs a window of relearning each way. But a DOWN-switch is CHEAP: the
                // hazard observations from the deeper run are all still valid at the shallower
                // depth (hz[i] for i < new depth were measured under the deep policy and are
                // exactly the same conditional events), so the next window can go straight back
                // up if it was wrong. The P2 canonical run showed the symmetric margin eating a
                // real win: hazards [@1:44% @2:31% @3:26% @4:24%] (accept@4 = 0.54 × accept@1)
                // with r(3)/r(4) = 1.26/1.39 made d3 strictly better, and the policy stayed at
                // d4/d5 — the 5% margin outweighed the ~1% score gap. Down-margin 1.0.
                let margin = if d < self.depth { 1.0 } else { MTP_DEPTH_MARGIN };
                if s > best * margin { best = s; best_d = d; }
            }
        }

        // TP item E: log EVERY window evaluation, not just switches. Under TP=2 both ranks run
        // this policy on bit-identical token history with the head's shipped r(d) tables, so the
        // lines must be byte-identical across ranks — and a gate that only diffed switch events
        // could pass vacuously on a no-decision run (AGENTS.md §4.12). All values are pure
        // functions of the deterministic hazard curve and the shipped table.
        eprintln!("[mtp] window: d={} yield {:.2} acc {:.1}% | ctx {} | cur {:.2}x best d={} {:.2}x",
                  self.depth, observed, acc * 100.0, ctx, cur, best_d, best);

        if best < 1.0 {
            self.active = false;
            self.retry_at = self.decode_steps + MTP_RETRY_AFTER;
            eprintln!("[mtp] DISABLED: acceptance {:.1}% gives {:.2} tok/step, and no depth beats a \
                       plain decode (best {:.2}x). Re-probing in {} steps.",
                      acc * 100.0, observed, best, MTP_RETRY_AFTER);
            return;
        }
        if best_d != self.depth {
            let hzs: Vec<String> = (0..self.depth.max(best_d) - 1)
                .map(|i| match self.hz.get(i) {
                    Some(&(a, n)) if n >= 8.0 => format!("{:.2}", a / n),
                    _ => "?".to_string(),
                }).collect();
            eprintln!("[mtp] depth {} -> {} ({:.2} -> {:.2} tok/step, r {:.2} -> {:.2}, {:.2}x -> {:.2}x) \
                       hazards [{}]",
                      self.depth, best_d, yield_at(&self.hz, self.depth), yield_at(&self.hz, best_d),
                      self.r(), self.r_at(best_d, ctx), cur, best, hzs.join(" "));
            self.depth = best_d;
        }
    }
}

/// Per-step RNG for stochastic MTP. The lane holds a 64-bit key; every draw is an independent
/// splitmix64 of (key, domain, index), and the key advances once per decode step.
///
/// Two properties this has to have, both learned the hard way:
///
/// 1. **Uniforms must actually land in [0,1).** The device kernels run a *32-bit* LCG and take
///    `r = (s >> 8) / 2^24`, which is a 24-bit mantissa. Carrying that state in a u64 and shifting
///    right by 8 leaves a full 32-bit value, so `r` lands in [0, 256) and an `r < ratio` test with
///    `ratio <= 1` succeeds only ~1/256 of the time — silently rejecting ~255/256 of perfectly
///    acceptable drafts (this is what pinned stochastic acceptance at ~1%).
///
/// 2. **Draws must not share a stream across consumers.** Each `spec_verify_b` column advances its
///    seed internally, so handing the columns *consecutive* LCG states makes column `j`'s draw come
///    from the state that seeds column `j+1` — fine only as long as every column draws exactly once,
///    and an outright collision the moment one draws twice. Domain separation removes the coupling
///    entirely, so the host accept decisions, the per-column residual samples, and the bonus sample
///    are independent by construction.
#[inline]
pub(crate) fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Domain separators keeping the host accept stream disjoint from the device column seeds.
pub(crate) const RNG_DOM_VERIFY: u64 = 0x5645_5249_4659_0001; // device seeds for spec_verify_b columns
pub(crate) const RNG_DOM_ACCEPT: u64 = 0x4143_4345_5054_0001; // host accept/reject uniforms
pub(crate) const RNG_DOM_SAMPLE: u64 = 0x5341_4D50_4C45_0001; // device seeds for sample_b (batched path)
pub const RNG_DOM_DF2_SEL: u64 = 0x4446_3253_454C_0001; // device seeds for the sampled selector walk (S5F2 L2)

/// One independent 32-bit draw from the lane's step key, keyed by domain + index.
#[inline]
pub fn rng_u32(key: u64, domain: u64, idx: usize) -> u32 {
    (splitmix64(key ^ domain ^ (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 32) as u32
}

/// One independent uniform in [0,1), matching the device's 24-bit mantissa convention.
#[inline]
pub(crate) fn rng_uniform(key: u64, domain: u64, idx: usize) -> f32 {
    (rng_u32(key, domain, idx) >> 8) as f32 * (1.0f32 / 16777216.0)
}

/// Length of the longest common prefix of two token sequences.
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// DF2_CARRY decision, as a pure function so the cases are unit-testable (see `mod df2_carry`).
///
/// The DFlash2 ring is a single already-resident shared buffer, so carrying it across a
/// prefix-cache hit costs **zero bytes** — the only question is whether the rows the draft will
/// read still hold the right positions. Two conditions make that true, and the second is why a
/// length-only copy of DSpark's A6 carry (`slot_dspark_len[phys] >= reuse`) is WRONG here:
///
/// 1. **Identity** — `ring_slot == Some(phys)`. The round is ONE shared buffer while the
///    per-slot bookkeeping is per-slot, so "this slot once ran the drafter" does not imply the
///    ring still holds THIS slot's prefix. DSpark used to guard on length alone and was therefore
///    exposed to a stale ring after any other slot used the drafter in between; with
///    `max_batch > 1` that interleaving is reachable, so DFlash2 records the identity — and
///    DSpark now does too (`DsparkRound::ring_slot`, PLAN/DSPARK_RING_IDENTITY_SPEC.md).
///
/// 2. **Frontier** — `reuse <= ring_len <= plen`. The DF2 ring is a MODULO ring (`pos % RING`,
///    RING = 2048); DSpark's is LINEAR (`c_ring = max_c`, no wrap), which is exactly why A6's
///    length check suffices there and does not here. After a lane that ran to frontier `N` the
///    ring holds precisely positions `[N-RING, N)`; every older row has been recycled. The
///    draft at the anchor reads the trailing band `[plen-RING+1, plen)`; the suffix prime writes
///    `[reuse, plen)`; those two are complementary residues of that band, so the CARRIED rows
///    must cover `[plen-RING+1, reuse)`, which requires `reuse <= N` (they were written) and
///    `N <= plen` (they have not yet been overwritten). Both bounds are equalities in the
///    natural multi-turn flow (`N` is the previous turn's prompt + its emitted tokens, and the
///    new prompt is that plus the client's next message), which is why this fires at all.
///
/// Carrying with `N > plen` would feed the draft keys from the wrong ring rows. Verify still
/// rejects bad drafts, so the output text would stay correct and losslessness would stay green
/// while tau quietly fell — the "silently wrong but self-consistent" class AGENTS §3 warns
/// about. Refuse instead.
#[allow(clippy::too_many_arguments)]
fn df2_carry_ok(enabled: bool, is_df2_src: bool, reuse: usize, plen: usize,
                phys: usize, ring_slot: Option<usize>, ring_len: usize,
                len_only: bool) -> bool {
    enabled
        && is_df2_src
        && reuse > 0
        && ring_slot == Some(phys)
        && ring_len >= reuse
        // `len_only` is a DIAGNOSTICS-ONLY instrument (--df2-carry-len-only=1): it drops the
        // frontier term, reproducing DSpark's length-only shape so SPEC §7 gate 7 can force a
        // known-bad carry and confirm the measurement catches it. Never a serving option.
        && (len_only || ring_len <= plen)
}

/// An image's content identity inside a slot's cached sequence: the token-span start plus a hash
/// of the merged embeddings that were actually spliced there. The slot-level prefix reuse matches
/// purely on token identity, and two images of the same resolution expand to IDENTICAL `image_pad`
/// token runs — so a token-only prefix key would happily reuse image N's spliced KV/state for image
/// N+1. This fingerprint is the image-content half of that key.
#[derive(Clone, Copy)]
struct ImageIdentity {
    start: usize,        // token-span start in the EXPANDED stream
    hash: u64,           // FNV-1a over the image's merged-embedding bytes
}

/// The per-image content identities for a request's concatenated `image_embeds` (f32 merged
/// embeddings, concatenated in `spans` order). Empty when the request carries no images. A hash of
/// the RAW f32 merged embeddings (the bytes the prefill converts to bf16 and splices) is the most
/// faithful content key: the same image always yields the same embeddings, a different image
/// essentially never does.
fn request_image_identities(
    embeds: Option<&Vec<f32>>,
    spans: &[crate::vision_encoder::ImageSpan],
) -> Vec<ImageIdentity> {
    // Row width = the vision out width (== text hidden for every supported tower). Derive it
    // from the request itself: len == sum(spans.num_tokens) * width, so ANY tower geometry
    // (0.8b 1024 … 27B 5120) is handled without plumbing model-specific constants.
    let total: usize = spans.iter().map(|s| s.num_tokens).sum();
    let out_h = if total > 0 {
        embeds.map(|e| e.len() / total).unwrap_or(0)
    } else {
        0
    };
    let mut out = Vec::new();
    let Some(emb) = embeds else { return out; };
    let mut off = 0usize;
    for s in spans {
        let n = s.num_tokens;
        let row0 = off * out_h;
        let row1 = (off + n) * out_h;
        let slice = if row1 <= emb.len() { &emb[row0..row1] }
                    else if row0 < emb.len() { &emb[row0..emb.len()] }
                    else { &[][..] };
        // Hash the f32 bit patterns directly (no `from_raw_parts` on a possibly-empty slice).
        let mut h: u64 = 0xcbf29ce484222325;
        for &v in slice {
            for b in v.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        out.push(ImageIdentity { start: s.start, hash: h });
        off += n;
    }
    out
}

/// True iff reusing the prefix `[0, l)` from a slot is safe for this request's images: every image
/// whose span begins inside the reused prefix (`start < l`) must match the slot's recorded image at
/// the same position. A differing image at a reused position means the slot's KV/GDN state was built
/// from the OLD image — reusing it would replay that image's content.
fn images_compatible(slot_imgs: &[ImageIdentity], req_imgs: &[ImageIdentity], l: usize) -> bool {
    for r in req_imgs {
        if r.start < l {
            match slot_imgs.iter().find(|s| s.start == r.start) {
                Some(s) if s.hash == r.hash => {}
                _ => return false,
            }
        }
    }
    true
}

/// Greedy tree accept-walk. Follow the target's argmax down the tree: at the current node, if any child
/// carries the token the target predicts (`preds[current]`), descend into it; else stop. Column 0 is the
/// committed token (always the start). This is the tree generalisation of the chain "accept longest
/// prefix", and greedy exactness is preserved by the same induction: every emitted token is the target's
/// argmax given its accepted prefix, so the output is byte-identical to plain autoregressive decode.
///
/// `parent[c]` is c's tree parent (parent[0] = -1). `tokens[c]` is the draft token at column c (tokens[0]
/// is the committed token). `preds[c]` is the target's argmax AFTER the path ending at c. Returns:
/// - `path`: the accepted node indices, root first (`path[0] == 0`).
/// - `emitted`: the tokens to emit = `preds` along the accepted path = each step's correction/accepted
///   token, ending with the bonus `preds[leaf]`.
///
/// A tie prefers the LOWEST child index (deterministic); duplicate sibling tokens are de-duped upstream.
fn tree_accept_walk(parent: &[i32], tokens: &[u32], preds: &[u32]) -> (Vec<usize>, Vec<u32>) {
    let n = parent.len();
    // children[c] in ascending index order (index order == DFS order here, so lowest = first-drafted).
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for c in 1..n { children[parent[c] as usize].push(c); }

    let mut path = vec![0usize];
    let mut emitted = Vec::new();
    let mut cur = 0usize;
    loop {
        let want = preds[cur];
        emitted.push(want); // the token the target actually chose after `cur` (accepted child or bonus)
        match children[cur].iter().copied().find(|&c| tokens[c] == want) {
            Some(c) => { path.push(c); cur = c; }
            None => break, // no child matched: `want` is the correction/bonus, walk ends
        }
    }
    (path, emitted)
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
    mtp_pen_tokens: Option<cudarc::driver::CudaSlice<i32>>,  // [depth * MAX_PEN_TOKENS]
    mtp_pen_counts: Option<cudarc::driver::CudaSlice<i16>>,  // [depth * MAX_PEN_TOKENS]
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
    df2_tree_stat_steps: u64,   // df2_tree_step invocations
    df2_tree_stat_rescues: u64, // accepted path crossed into the B branch (the tree's whole point)
    df2_tree_stat_nodes: u64,   // verified tree nodes (cost side: the chain would be depth+1)
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
}

impl BatchScheduler {
    pub fn new(gpu: GpuModel, max_batch: usize, kv_stride: usize, eos: Vec<u32>,
               rx: mpsc::UnboundedReceiver<BatchRequest>,
               mtp: MtpPolicy, prefix_cache: bool, ngram_draft: usize, tree_draft: bool, mtp_lanes: bool) -> Self {
        BatchScheduler::with_df2(gpu, max_batch, kv_stride, eos, rx, mtp, prefix_cache,
                                 ngram_draft, tree_draft, mtp_lanes,
                                 None, None, None, None, None, None, None)
    }

    /// P11 W5a: the --prefill-sched inline escape (default false = the cursor schedule).
    pub fn set_pf_inline(&mut self, on: bool) { self.pf_inline = on; }

    /// P3(b) L1: set the prose-lane routing (default `false` = rq sampled selector; `true` =
    /// greedy drafts for General-domain requests under `DFlash2Auto`).
    pub fn set_prose_lane_greedy(&mut self, on: bool) {
        self.prose_lane_greedy = on;
    }

    /// DF2_CARRY: `--df2-carry on|off` (default off — SPEC §7 gates the default flip). SPMD:
    /// ships on TpConfig, so every rank resolves the same value and makes the same admit choice.
    pub fn set_df2_carry(&mut self, on: bool) {
        self.df2_carry_enabled = on;
    }

    /// S5F: `new` + the DFlash2 round (loaded by the caller; `None` = absent/failed artifact → the
    /// source falls back to MTP per the standing directive) and its tap-sink twins.
    pub fn with_df2(mut gpu: GpuModel, max_batch: usize, kv_stride: usize, eos: Vec<u32>,
                    rx: mpsc::UnboundedReceiver<BatchRequest>,
                    mtp: MtpPolicy, prefix_cache: bool, ngram_draft: usize, tree_draft: bool,
                    mtp_lanes: bool,
                    df2: Option<crate::dflash2::round::Df2Round>,
                    df2_sink: Option<std::sync::Arc<crate::dflash2::capture::Df2TapSink>>,
                    df2_prime: Option<std::sync::Arc<crate::dflash2::capture::Df2PrimeSink>>,
                    mut dspark: Option<crate::dspark::round::DsparkRound>,
                    dspark_sink: Option<std::sync::Arc<crate::dflash2::capture::Df2TapSink>>,
                    dspark_prime: Option<std::sync::Arc<crate::dflash2::capture::Df2PrimeSink>>,
                    step_dump: Option<crate::dflash2::stepdump::StepDump>) -> Self {
        // When MTP is on, reserve one extra physical slot as a shared GDN-rollback snapshot target
        // (MTP lanes run one at a time, so a single snapshot slot suffices). It is never assigned to
        // a lane; `copy_gdn_slot(state, phys, snapshot_slot)` snapshots before verify, the reverse
        // restores on partial reject.
        // Reserve the snapshot slot whenever the model HAS an MTP head, not merely when MTP is
        // currently active: the policy can switch MTP on later and the slot must already exist.
        let mtp_has_head = mtp.head_present();
        // Size for the DEEPEST depth the policy may choose, not the one it opens at — the policy
        // re-picks depth from live acceptance every window, and a buffer sized to the initial depth
        // would be silently overrun the moment it went deeper.
        let mtp_depth = crate::gpu::MAX_AUTO_DEPTH;
        // ONE checkpoint slot per verify column we might roll back to: nacc can be 0..depth-2, so
        // we need depth-1 of them, contiguous from `mtp_snapshot_slot`. The GDN kernels write
        // column t's post-state into slot (mtp_snapshot_slot + t) via a derived stride.
        //
        // A single slot is only correct at depth 2. With depth >= 3 it silently rolled the recurrent
        // state back past accepted drafts, which is why greedy MTP was not lossless above depth 2.
        let mtp_snapshot_slot = max_batch;
        // PLAN/25 Phase 1: the DFlash2 TREE lane can engage lazily on any scheduler that carries
        // the DF2 round (e.g. the lossless probe switches sources per job), and its verify writes
        // per-column GDN checkpoints for up to MAX_VERIFY columns — size the band for that
        // capability whenever the round is present, not just when the source is the tree.
        let src_is_tree = matches!(mtp.spec_source(), SpecSource::DFlash2Tree);
        // P14: the DFlash v1 lane (--spec-source dflash) verifies the artifact's WHOLE block — for
        // the 3.6-35B drafter that is `1 + (block-1)` = 16 columns, exactly MAX_VERIFY — and the GDN
        // scan writes one checkpoint per column at `mtp_snapshot_slot + t`, t < n-1, with no bounds
        // check (`kernels/gpu_batch.cu:1299`, `:4120`). Sized off `mtp_depth` (8 ⇒ 7 checkpoints) the
        // 16-column verify overran the band by 8 slots per GDN layer: every emitted token came back
        // token 0 (`!`) because the verify's own logits were computed from corrupted state, at ~400
        // tok/s wall — the "step speed without acceptance is not a result" signature
        // (HANDOFF 2026-09-12). Capability, not current source: the v1 lane can engage on ANY
        // scheduler that carries the artifact (the probe/harness flip sources per job), exactly the
        // rule the DF2 tree already follows. `install_dflash_lane` runs after construction, so the
        // capability is detected from the artifact's presence OR the resolved source.
        let dflash_v1 = matches!(mtp.spec_source(), SpecSource::DFlash)
            || crate::draft_dir_env().is_some();
        // The band below is sized MAX_VERIFY; a block constant WIDER than that would make the v1
        // verify overrun it. Compile-time, not a boot warning: this relation is a property of two
        // constants, and a silent runtime warning is exactly the class of notice nobody reads.
        const _: () = assert!(crate::dflash::MAX_BLOCK <= crate::gpu::MAX_VERIFY,
            "DFlash MAX_BLOCK must not exceed MAX_VERIFY: the v1 verify width is bounded by \
             MAX_VERIFY (past it the verify leaves the batch-invariant path) and the GDN \
             checkpoint band is sized MAX_VERIFY");
        let n_ckpt = if mtp_has_head || dflash_v1 {
            if df2.is_some() || tree_draft || mtp_lanes || src_is_tree || dflash_v1 {
                crate::gpu::MAX_VERIFY
            } else {
                mtp_depth.saturating_sub(1).max(1)
            }
        } else { 0 };
        // One PROMPT checkpoint slot per lane, after the MTP snapshot slots. These hold the GDN state
        // as it stood at the END OF PREFILL — see `prompt_ckpt_slot`. They are pure state: no KV, and
        // none at all when prefix caching is off (51 MB/slot on 9B, 154 MB on 27B).
        let prompt_ckpt_slot = max_batch + n_ckpt;
        let ring_ckpt_slot = prompt_ckpt_slot + max_batch;
        // A6: the periodic ring checkpoints add max_batch*RING_CKPT_K state slots (154 MB/slot
        // class on 27B; K=16 x stride 512 keeps the same 8K coverage window as the old K=4 x
        // 2048 at 4x finer boundaries → ~2.5 GB per lane; the cached-turn slack drops from a
        // 2047-token re-prefill (~3.2 s at the mid-M lane) to <=511 tokens. b=1 deployments.)
        let n_state_slots = max_batch + n_ckpt
            + if prefix_cache { max_batch * (1 + RING_CKPT_K) } else { 0 };
        // ---- S-B16: the env-gated in-server lossless auditor (--mtp-lossless-audit=1) ----
        // A lockstep shadow decode of the greedy MTP lane's committed stream (bench_mtp_steps'
        // slot-1 ground truth, moved into the serving path) that pins the FIRST divergence to an
        // exact (request, emitted-index, position). DIAGNOSTIC ONLY: one extra plain decode
        // forward per emitted token roughly halves serving throughput while on; off, it is one
        // bool. TP>1 refuses the audit — see the field doc above.
        let audit_on = crate::opts::var(crate::opt!("mtp-lossless-audit")).is_ok();
        let audit_on = if audit_on && gpu.tp_world() > 1 {
            eprintln!("[lossless-audit] REFUSED under TP={} — the shadow decode is single-node SPMD; run the audit at TP=1",
                      gpu.tp_world());
            false
        } else { audit_on };
        let audit_slot = n_state_slots; // one extra state slot appended after the normal band
        let n_state_slots = n_state_slots + if audit_on { 1 } else { 0 };
        if prefix_cache {
            eprintln!("[cache] prefix cache ON — prompt ckpt + ring {RING_CKPT_K}x{RING_CKPT_STRIDE} \
                       GDN checkpoints per slot ({n_state_slots} state slots)");
        }
        crate::memwatch::phase(&format!("sched:pre-state slots={n_state_slots} kv_stride={kv_stride} max_batch={max_batch}"));
        // S-B16: the auditor's shadow slot lives at index audit_slot in the STATE band, and KV
        // rows are addressed as base + slot*k_bytes — so the KV cache must span that slot INDEX,
        // not just max_batch. (v2/v3 bug: with kv_slots = max_batch = 1 the shadow's KV writes
        // landed ~8 slots past the allocation — silent corruption in v2 (the bogus 99% accept),
        // a wedged first request in v3.) Diagnostic-only: kv_slots = n_state_slots costs a few
        // GB of extra KV allocation on an audit boot; the serving layout is UNCHANGED when off.
        let kv_alloc_slots = if audit_on { n_state_slots } else { max_batch };
        let mut state = gpu.new_batch_state(kv_alloc_slots, n_state_slots, kv_stride);
        gpu.dev().synchronize().unwrap(); // ensure state allocs visible to non-blocking stream
        let mut pool = Pool::new(gpu.dev().clone());
        let mut bufs = gpu.new_decode_buffers(max_batch);
        crate::memwatch::phase("sched:post-state+bufs");
        // The auditor's shadow decode buffers: batch 2 like the bench's slot-1 ground truth, with
        // slot_ids preloaded so every shadow step reads/writes ONLY the audit slot's KV + GDN
        // state (mirror of bench_mtp_steps' [1, 0] preload on its slot-1 bufs2).
        let audit_bufs = if audit_on {
            let mut ab = gpu.new_decode_buffers(2);
            gpu.dev().htod_sync_copy_into(&[audit_slot as i32, 0i32], &mut ab.slot_ids_dev).unwrap();
            eprintln!("[lossless-audit] ARMED slot={audit_slot} — one shadow decode per emitted token (throughput halves)");
            Some(ab)
        } else {
            None
        };

        // Capture decode graphs for all batch sizes, IF every kernel in the captured region fits the
        // default 48 KB/block limit.
        //
        // THIS USED TO BE `(kv_stride + head_dim) * 4`, which described the PRE-split-K attention
        // kernel: it held a kv_stride-sized score array in shared memory, so smem grew with context and
        // graphs were skipped beyond ~12K positions. The old comment even said "split-K will fix this".
        // Split-K shipped — `gqa_attn_splitk` sizes its smem by WARP COUNT, not context — but the guard
        // was never updated, so every long-context deployment silently lost CUDA graphs: 129 KB computed
        // at seq 32768 (the 122B preset), and at the 256K target envelope graphs would never capture at
        // all, on any model.
        //
        // The real requirement is the MAX over the kernels inside `forward_decode_gpu`, all of which are
        // constant in context length:
        //   GDN delta_step_b  18.4 KB  (gdn_launch: kd*(GDN_C+1) + kd + 3*GDN_C + 2*kd floats)  <- max
        //   attention splitk   8.1 KB  ((nw*hd + 2*nw) * 4, nw = hd/32)
        //   argmax_b           8.0 KB
        //   rmsnorm_b          4.0 KB
        // Capture already degrades safely (`capture_decode_graph` returns None and we fall back), so this
        // guard is belt-and-braces — but it must not be the thing that disables graphs everywhere.
        let head_dim = 256;
        let c = gpu.cfg();
        let gdn_smem = crate::gpu::gdn_launch(c.lin_k_dim, c.lin_v_dim).1 as usize;
        let attn_smem = ((head_dim / 32) * head_dim + 2 * (head_dim / 32)) * 4;
        let smem_bytes = gdn_smem.max(attn_smem).max(1024 * 8);
        // A/B switch: --no-decode-graphs=1 forces the non-graph path, so the value of capture can be
        // measured on any config without rebuilding (and gives an escape hatch if capture ever misbehaves).
        let smem_bytes = if crate::opts::var(crate::opt!("no-decode-graphs")).is_ok() { usize::MAX } else { smem_bytes };
        // qwen4_exp with the PLE table on SSD: the forward has a host round-trip (row reads) —
        // not capturable. Everything else keeps the graph path (PR #4 width/graph contract).
        let smem_bytes = if gpu.decode_graphs_supported() { smem_bytes } else {
            println!("Skipping CUDA graphs: PLE table is SSD-resident (host gather inside the forward).");
            usize::MAX
        };
        let mut graphs = std::collections::HashMap::new();
        if smem_bytes <= 48 * 1024 {
            print!("Attempting CUDA graph capture for batch sizes 1..={}... ", max_batch);
            let mut ok = true;
            for b in 1..=max_batch {
                match gpu.capture_decode_graph(&mut pool, &mut bufs, &mut state,
                                                 kv_stride, kv_stride, b) {
                    Some(g) => { graphs.insert(b, g); }
                    None => { ok = false; break; }
                }
            }
            // Graph capture (and its warmup) advanced the stateful GDN recurrent state — reset.
            gpu.zero_state(&mut state);
            gpu.dev().synchronize().unwrap(); // ensure zeroed state visible to non-blocking stream
            if ok {
                println!("captured (smem {} KB).", smem_bytes / 1024);
            } else {
                graphs.clear();
                println!("unsupported (legacy stream); using non-graph decode.");
            }
        } else {
            println!("Skipping CUDA graphs: attention smem {} KB > 48 KB. Using non-graph decode.",
                     smem_bytes / 1024);
        }

        // When GPU sampling is enabled, also capture a decode+sample graph per batch size so that
        // sampling requests get the same graph speedup greedy does. Falls back to the non-graph
        // sample path if capture is unsupported.
        let gpu_sample = crate::opts::var(crate::opt!("cpu-sample")).is_err();
        let mut sample_graphs: std::collections::HashMap<usize, CudaGraph> = std::collections::HashMap::new();
        if gpu_sample && !graphs.is_empty() {
            for b in 1..=max_batch {
                if let Some(g) = gpu.capture_decode_sample_graph(&mut pool, &mut bufs, &mut state,
                                                                  kv_stride, kv_stride, b) {
                    sample_graphs.insert(b, g);
                }
            }
            gpu.zero_state(&mut state);
            gpu.dev().synchronize().unwrap();
            println!("captured {} GPU-sample graph(s).", sample_graphs.len());
        }
        crate::memwatch::phase("sched:post-graphs");

        // Allocate per-slot MTP state when MTP is enabled. The MTP KV (`[nkv, kv_stride, hd]` bf16
        // per slot) must be zeroed: alloc_zeros is cuMemAllocAsync which does NOT zero, and stale
        // GPU garbage in unwritten MTP KV positions would yield nondeterministic drafts. Zero on the
        // COMPUTE stream (ordered with all later kernels), then sync.
        let (mtp_kc, mtp_vc, mtp_h_prev, mtp_h_save, mtp_h_scratch, mtp_cur_hidden,
             mtp_pen_tokens, mtp_pen_counts, mtp_pen_rep, mtp_pen_presence, mtp_pen_freq,
             mtp_draft_pen_tokens, mtp_draft_pen_counts, mtp_draft_pen_rep, mtp_draft_pen_presence,
             mtp_draft_pen_freq) =
            if mtp_has_head {
                let cfg = gpu.cfg();
                // PR #4 width contract: the hidden columns the scheduler moves for the MTP head
                // (h_prev/save/scratch/cur) are the BACKBONE width — the hc-stream stack on
                // qwen4_exp (4*hidden), the hidden size elsewhere (resid_width() == hidden_size
                // whenever hc_count == 1, so this is a no-op for every existing model).
                let h = gpu.mtp_hidden_width();
                debug_assert_eq!(h, cfg.resid_width(),
                    "MTP hidden columns must be the hc-stream (resid) width");
                // §4.1: with --tp-shard-mtp the MTP attention is head-sharded and the draft cache
                // holds only this rank's kv heads (mtp_kv_heads == num_kv_heads when unsharded).
                let nkv = gpu.mtp_kv_heads();
                let hd = cfg.head_dim;
                let kv_bytes = nkv * kv_stride * hd * 2; // bf16 — the DRAFT cache stays bf16 even
                // under the 4-bit main KV cache (quantized draft KV costs real acceptance).
                let dev = gpu.dev().clone();
                let mut kc: Vec<B> = Vec::with_capacity(max_batch);
                let mut vc: Vec<B> = Vec::with_capacity(max_batch);
                let mut hp: Vec<B> = Vec::with_capacity(max_batch);
                for _ in 0..max_batch {
                    // qwen4_exp QSA: the head's raw-key cache rides at the end of its K buffer
                    // (mtp_kc_elems) so the (kc, vc, kv_stride) plumbing stays untouched.
                    let k = dev.alloc_zeros::<half::bf16>(gpu.mtp_kc_elems(kv_stride)).unwrap();
                    let v = dev.alloc_zeros::<half::bf16>(nkv * kv_stride * hd).unwrap();
                    let p = dev.alloc_zeros::<half::bf16>(h).unwrap();
                    gpu.memset_compute_stream(*k.device_ptr(), gpu.mtp_kc_elems(kv_stride) * 2);
                    gpu.memset_compute_stream(*v.device_ptr(), kv_bytes);
                    kc.push(k);
                    vc.push(v);
                    hp.push(p);
                }
                let save = dev.alloc_zeros::<half::bf16>(h).unwrap();
                let scr = dev.alloc_zeros::<half::bf16>(h).unwrap();
                let cur = dev.alloc_zeros::<half::bf16>(h).unwrap();
                gpu.sync_stream(); // ensure MTP KV zeroing visible before any primed lane reads it
                // Penalty buffers for the verify path (depth positions). bf16 lanes already keep
                // penalties via the normal path; MTP greedy lanes now keep theirs too.
                let mp = crate::gpu::MAX_PEN_TOKENS;
                // Sized to MAX_VERIFY (not just the chain depth): a FOREST verify has up to MAX_VERIFY
                // columns spanning several lanes, each column carrying ITS lane's penalty (per-column).
                let pcap = crate::gpu::MAX_VERIFY;
                let pen_tokens = dev.alloc_zeros::<i32>(pcap * mp).unwrap();
                let pen_counts = dev.alloc_zeros::<i16>(pcap * mp).unwrap();
                let pen_rep = dev.alloc_zeros::<f32>(pcap).unwrap();
                let pen_presence = dev.alloc_zeros::<f32>(pcap).unwrap();
                let pen_freq = dev.alloc_zeros::<f32>(pcap).unwrap();
                let dp_tokens = dev.alloc_zeros::<i32>(mp).unwrap();
                let dp_counts = dev.alloc_zeros::<i16>(mp).unwrap();
                let dp_rep = dev.alloc_zeros::<f32>(1usize).unwrap();
                let dp_presence = dev.alloc_zeros::<f32>(1usize).unwrap();
                let dp_freq = dev.alloc_zeros::<f32>(1usize).unwrap();
                (kc, vc, hp, Some(save), Some(scr), Some(cur),
                 Some(pen_tokens), Some(pen_counts), Some(pen_rep), Some(pen_presence), Some(pen_freq),
                 Some(dp_tokens), Some(dp_counts), Some(dp_rep), Some(dp_presence), Some(dp_freq))
            } else {
                // MTP absent: allocate MINIMAL 1-element dummy per-lane buffers so the unconditional
                // `mtp_kc[phys]`/`mtp_vc[phys]` indexing on the prefill/decode paths never panics. These
                // pointers are never dereferenced — every real MTP use is gated by `will_use_mtp` /
                // `self.mtp.active()`, both false without a head.
                let dev = gpu.dev().clone();
                let dummy = |n: usize| -> Vec<B> {
                    (0..n).map(|_| dev.alloc_zeros::<half::bf16>(1).unwrap()).collect()
                };
                (dummy(max_batch), dummy(max_batch), dummy(max_batch),
                 None, None, None, None, None, None, None, None,
                 None, None, None, None, None)
            };

        // The graph replays with per-step (anchor, nprev) written to device ints; the R13
        // volatile kernels are stable under capture (the probe asserts determinism). Env
        // --no-df2-graph=1 keeps the eager path (the captured-vs-eager measurement).
        // PLAN/25 Phase 1: `dflash2-tree` arms the WIDE (MAX_VERIFY-col) tap sink on the trunk —
        // the topo verify's n > BLOCK per-column taps land there; the accepted path is gathered
        // into the round's 8-col staging at commit. No-op for every other source.
        // WI1: arm the DSpark sinks BEFORE `gpu` moves into Self (the DF2 twins' pre-move
        // pattern), and bind the round to its sink (S5F3: keep the Arc; the lane copies the
        // sink's LIVE staging before each inject).
        crate::memwatch::phase("sched:post-mtp-state");
        if let Some(ds) = dspark.as_mut() {
            if let Some(sk) = &dspark_sink { gpu.set_dspark_capture(sk.clone()); ds.attach_sink(sk); }
            if let Some(ps) = &dspark_prime { gpu.set_dspark_prime_sink(ps.clone()); }
        }
        let df2_sink_tree = if matches!(mtp.spec_source(), SpecSource::DFlash2Tree) {
            let ws = std::sync::Arc::new(crate::dflash2::capture::Df2TapSink::new_cols(gpu.dev(), crate::gpu::MAX_VERIFY));
            gpu.set_df2_capture_tree(ws.clone());
            Some(ws)
        } else { None };
        let mut s = Self {
            dflash: None,
            gpu, pool, state, bufs, graphs, kv_stride, eos, max_batch, rx,
            lanes: (0..max_batch).map(|_| None).collect(),
            free_slots: (0..max_batch).rev().collect(),
            slot_cache: vec![Vec::new(); max_batch],
            slot_ckpt_seq: vec![Vec::new(); max_batch],
            slot_cache_images: vec![Vec::new(); max_batch],
            slot_ckpt_images: vec![Vec::new(); max_batch],
            prompt_ckpt_slot,
            ring_ckpt_slot,
            slot_ring_len: vec![vec![0usize; RING_CKPT_K]; max_batch],
            ring_cursor: vec![0usize; max_batch],
            slot_dspark_len: vec![0usize; max_batch],
            prefix_cache,
            ngram_draft,
            tree_draft,
            mtp_lanes,
            pf_inline: false,
            gpu_sample,
            sample_graphs,
            mtp,
            mtp_kc,
            mtp_vc,
            mtp_h_prev,
            mtp_h_save,
            mtp_h_scratch,
            mtp_cur_hidden,
            mtp_snapshot_slot,
            audit_on,
            audit_slot,
            audit_bufs,
            audit_cur: (0..max_batch).map(|_| None).collect(),
            audit_req_n: 0,
            mtp_pen_tokens,
            mtp_pen_counts,
            mtp_pen_rep,
            mtp_pen_presence,
            mtp_pen_freq,
            mtp_draft_pen_tokens,
            mtp_draft_pen_counts,
            mtp_draft_pen_rep,
            mtp_draft_pen_presence,
            mtp_draft_pen_freq,
            pen_const_key: None,
            pen_had: false,
            df2,
            df2_sink,
            df2_sink_tree,
            df2_prime,
            dspark,
            dspark_sink,
            dspark_prime,
            df2_fallback_logged: false,
            dflash_fallback_logged: false,
            dflash_prime_armed: false,
            dflash_prime_done: false,
            dflash_carry_logged: false,
            dspark_fallback_logged: false,
            dspark_stat_steps: 0,
            dspark_stat_drafts: 0,
            dspark_stat_accepted: 0,
            dspark_stat_emitted: 0,
            prose_lane_greedy: false,
            df2_carry_enabled: false,
            cov_trace: step_dump.is_some(),
            cov_req_n: 0,
            step_dump,
            mtp_stat_steps: 0,
            mtp_stat_drafts: 0,
            mtp_stat_accepted: 0,
            mtp_stat_emitted: 0,
            mtp_stat_verify_fwds: 0,
            mtp_stat_published: 0,
            df2_tree_stat_steps: 0,
            df2_tree_stat_rescues: 0,
            df2_tree_stat_nodes: 0,
            df2_stat_steps: 0,
            df2_stat_drafts: 0,
            df2_stat_accepted: 0,
            df2_stat_emitted: 0,
            dflash_stat_steps: 0,
            dflash_stat_drafts: 0,
            dflash_stat_accepted: 0,
            dflash_stat_emitted: 0,
            dflash_acc_a: [0; crate::dflash::MAX_BLOCK],
            dflash_acc_n: [0; crate::dflash::MAX_BLOCK],
            spec_steps: Vec::new(),
            spec_steps_on: false,
            mtp_draft_log: crate::opts::var(crate::opt!("mtp-draft-log")).ok().and_then(|p| {
                match std::fs::File::create(&p) {
                    Ok(f) => { eprintln!("[mtp] draft log -> {}", p); Some(f) }
                    Err(e) => { eprintln!("[mtp] WARN: cannot open --mtp-draft-log {}: {}", p, e); None }
                }
            }),
            mtp_curve_path: crate::opts::var(crate::opt!("mtp-curve-file")).ok(),
            tp_serving: false,
            // Device-resident token loop: the first step must upload everything (the capture
            // warmup left stale values in token_ids_dev/pos/ring), so the state starts dirty.
            resident_dirty: true,
            pf: None,
        };
        // S5F: capture the DFlash2 draft-round CUDA graph once (the MTP verify-graph pattern).
        // The graph replays with per-step (anchor, nprev) written to device ints; the R13
        // volatile kernels are stable under capture (the probe asserts determinism). Env
        // --no-df2-graph=1 keeps the eager path (the captured-vs-eager measurement).
        if crate::opts::var(crate::opt!("no-df2-graph")).is_err() {
            if let Some(df2) = s.df2.as_mut() {
                if df2.capture_round_graph() {
                    eprintln!("[df2] draft-round CUDA graph captured (eager fallback via --no-df2-graph)");
                } else {
                    eprintln!("[df2] draft-round graph capture unsupported — staying eager");
                }
            }
        }
        s
    }

    /// Run the scheduler loop until the request channel closes and no lanes remain.
    ///
    /// PLAN/SCHEDULER_2LANE_WORKDOC.md §3.1/§3.3 — the two-lane (cursor) loop. Shape per iteration:
    ///
    ///   1. DRAIN queued requests (cheap: admission only creates a cursor),
    ///   2. BLOCK for work only when there is none — a pending cursor IS work,
    ///   3. ADVANCE at most one prefill window of the single in-flight cursor,
    ///   4. DECODE the `b` lanes that are actually eligible (a mid-prefill request is not a lane).
    ///
    /// The ONE structural difference from the pre-cursor loop is the position of the window work:
    /// it used to run to completion INSIDE `admit()` (all `ceil(plen/PREFILL_CHUNK)` windows before
    /// that iteration's `decode_step`), so a co-resident lane's next token waited for the whole
    /// prompt — measured at 25,370 ms across a 16,800-token admission (P0). Now it waits for at
    /// most one window (12.6 s), and the window formula itself is unchanged (§3.2).
    pub async fn run(mut self) {
        loop {
            // F8 diagnosis (temporary, --loop-trace): split the wall around decode_step so the
            // serve-vs-bench per-step gap (261 vs 188 ms) can be attributed.
            let loop_trace = crate::opts::var(crate::opt!("loop-trace")).is_ok();
            let t_admit = if loop_trace { Some(std::time::Instant::now()) } else { None };
            // Admit queued requests into free lanes (front-packed). Capacity counts BOTH the
            // decode-eligible lanes and the in-flight cursor: a mid-prefill request already owns a
            // physical slot, so it must not be oversubscribed.
            // `self.pf.is_none()` is re-tested EVERY iteration: a successful admit creates a
            // cursor, and at most one may exist (the shared df2/dspark rounds are primed
            // window-by-window). Without that term a max-batch>=3 server would try to admit a
            // second request and silently drop the first cursor's prefill.
            while self.pf.is_none()
                  && self.num_active() + usize::from(self.pf.is_some()) < self.max_batch
                  && self.lanes[self.num_active()].is_none() {
                match self.rx.try_recv() {
                    Ok(req) => self.admit(req),
                    Err(_) => break,
                }
            }
            if self.num_active() == 0 && self.pf.is_none() {
                match self.rx.recv().await {
                    Some(req) => self.admit(req),
                    // Fall THROUGH (no `continue`): the cursor just created must get its first
                    // window this iteration, or the loop blocks on an empty channel forever.
                    None => break,
                }
            }
            // One window of the in-flight prefill (no-op when nothing is prefilling).
            self.pf_advance();
            let b = self.num_active();
            let t_step = if loop_trace { Some(std::time::Instant::now()) } else { None };
            let t_yield0 = if loop_trace { Some(std::time::Instant::now()) } else { None };
            if b > 0 { self.decode_step(b); }
            let decode_ms = t_step.map(|t0| t0.elapsed().as_secs_f32() * 1e3);
            // Yield to tokio so streaming handlers can flush SSE events between decode steps.
            tokio::task::yield_now().await;
            if let (Some(t0), Some(ty0), Some(ta0), Some(dm)) = (t_step, t_yield0, t_admit, decode_ms) {
                static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                static LAST: std::sync::OnceLock<std::sync::Mutex<std::time::Instant>> =
                    std::sync::OnceLock::new();
                let last = LAST.get_or_init(|| std::sync::Mutex::new(std::time::Instant::now()));
                let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n >= 5 && n < 35 {
                    let mut prev = last.lock().unwrap();
                    let since_prev = prev.elapsed().as_secs_f32() * 1e3;
                    *prev = std::time::Instant::now();
                    let yield_ms = ty0.elapsed().as_secs_f32() * 1e3 - dm;
                    let _ = t0;
                    eprintln!("[loop-trace] step {n}: decode_step={dm:.1}ms post_yield={yield_ms:.1}ms \
                               admit_ms={:.1} since_prev_total={since_prev:.1}ms",
                              ta0.elapsed().as_secs_f32() * 1e3);
                }
            }
        }
    }

    /// TP item D — per-step SPMD divergence guard. After every executed decode step, both ranks
    /// exchange `(step, cumulative emitted, state hash)` over the link's agreement channel
    /// (`net::agree`, 10 s deadline) and REFUSE to continue on mismatch or timeout. The hash folds
    /// the per-lane output state (`last_tok`, `generated`, `pos`, `mtp_pos`) plus the MTP policy
    /// decision (`active`, `depth`) — identical on both ranks by construction, so any acceptance
    /// or policy divergence flips it within one step. On mismatch: abort the link cooperatively
    /// (kernels no-op through the stream, I9) and return Err — the head's die-with-it guard exits
    /// the server and the mirror's supervisor re-arms, rather than serve one more divergent token.
    /// `--tp-agree-drill=<step>` corrupts this rank's hash at one step (the forced-divergence
    /// drill gate; env read directly, test-only).
    fn tp_agree_step(&self, step: u64) -> anyhow::Result<()> {
        let mut h: u32 = 0x811c9dc5;                     // FNV-1a over the lane/policy state
        let mut mix = |x: u32| { h ^= x; h = h.wrapping_mul(0x0100_0193); };
        let mut total_generated: u64 = 0;
        for l in self.lanes.iter().flatten() {
            mix(l.last_tok);
            mix(l.generated as u32);
            mix(l.pos as u32);
            mix(l.mtp_pos as u32);
            total_generated += l.generated as u64;
        }
        // PLAN/SCHEDULER_2LANE §3.4/§3.5: a mid-prefill request is NOT a lane, so its progress is
        // folded EXPLICITLY. Without this term two ranks that advanced different window counts would
        // agree on everything visible (lanes, policy, width) while holding different prefill state —
        // the divergence would only surface as a KV/state mismatch much later.
        if let Some(c) = self.pf.as_ref() {
            mix(c.phys as u32);
            mix(c.w0 as u32);
            mix(c.plen as u32);
        }
        mix(self.mtp.active() as u32);
        mix(self.mtp.depth() as u32);
        // P11/FOREST: the pack flag changes the verify shape per step; fold it so a one-sided
        // mtp_lanes resolution (mixed builds on one link) fails LOUD at step 0, not as drift.
        mix(self.mtp_lanes as u32);
        // B8/G1: k_verify (the verify WIDTH this step ran) joins the extended agree token — ranks
        // that disagree on the width execute different barrier sequences (I9 class). agree_ext
        // folds it into the hash word at bits [27..31); the depth IS the width for chain MTP.
        let k_verify = if self.mtp.active() { self.mtp.depth() as u8 } else { 0u8 };
        if let Ok(d) = crate::opts::var(crate::opt!("tp-agree-drill")) {
            if d.parse::<u64>().ok() == Some(step) {
                eprintln!("[tp-agree] DRILL: corrupting this rank's hash at step {step}");
                h ^= 0xDEAD;
            }
        }
        let count = (total_generated & 0xFF) as u8;
        // B8: agree_ext folds k_verify into the hash word's low bits [27..31); fold it here too so
        // the comparison is against the SAME wire shape. Raw-vs-folded mismatches by exactly
        // (k_verify<<27) whenever MTP is active (k_verify = depth > 0) — the step-0 stall masked this.
        let h_ext = (h ^ ((k_verify as u32 & 0xF) << 27)) as u32;
        let (pc, ph) = match crate::net::agree_ext(step, count, k_verify, h) {
            Some(x) => x,
            None => {
                eprintln!("[tp-agree] link aborted or peer timeout at step {step} — aborting");
                crate::net::abort_link();
                anyhow::bail!("TP agree: link aborted/timeout at step {step}");
            }
        };
        if pc != count || ph != h_ext {
            eprintln!("[tp-agree] MISMATCH at step {step}: local (count {count}, hash {h_ext:#010x}) \
                       vs peer (count {pc}, hash {ph:#010x}) — aborting rather than serving \
                       divergent output");
            // R9 LOCALIZATION note: the per-layer xchain dump happens in net::agree (which sees the
            // mismatch on the head AND the sentinel on the nodes); this branch only runs when the
            // consensus token itself disagrees, which is a subset of that path.
            crate::net::abort_link();
            anyhow::bail!("TP AGREE MISMATCH at step {step}: local ({count}, {h_ext:#010x}) vs peer ({pc}, {ph:#010x})");
        }
        Ok(())
    }

    /// TP serving head loop (item A, N-way): the same loop shape as `run()`, plus a per-step
    /// rendezvous with EVERY node's mirror over the retained sync streams. Per step:
    ///   1. Drain admissions under the SAME capacity gate as `run()`, but DEFER the `admit()` calls
    ///      until after the Step message is sent — admission prefill runs SPMD all-reduces that the
    ///      mirror can only join once it has seen the admission, so shipping first is what keeps
    ///      the barrier sequences paired instead of deadlocked. Each drained request becomes a
    ///      `TpEvent::Admit`; `admit()` itself runs unmodified (its context-length reject is
    ///      deterministic from identical state, so the mirror rejects exactly the same requests).
    ///   2. Cancel detection: lanes whose client went away are marked `tp_cancelled` NOW (host
    ///      state, no race with decode_step's sweep) and shipped as `TpEvent::Cancel`. Indices are
    ///      the post-admission front-packed table; admissions only append past the active region,
    ///      so they cannot renumber these.
    ///   3. Send `ServingMsg::Step` — every executed step, even with an empty event list; it is the
    ///      rendezvous the mirror waits on.
    ///   4. Admit the pendings (in order), then one `decode_step` over the new front-packed table.
    /// On request-channel close (server shutdown): keep stepping until the live lanes drain, then
    /// send `ServingMsg::Shutdown` and return.
    pub async fn run_tp_head(mut self, mut streams: Vec<std::net::TcpStream>) -> anyhow::Result<()> {
        self.tp_serving = true;
        let mut step: u64 = 0;
        let mut closed = false;
        // PERSISTENT across iterations (cursor scheduler). With one prefill cursor at a time, a
        // request can be drained while another is still prefilling; if `pending` were rebuilt every
        // iteration (as it was when admission was unconditional) that request would be DROPPED at
        // the end of the iteration — its `tx` goes with it and the client's SSE stream closes with
        // no tokens and no error. The 2-lane bar caught exactly that: lane 1 was admitted never,
        // and its client saw the body end 1.5 s in.
        let mut pending: Vec<BatchRequest> = Vec::new();
        loop {
            let mut events: Vec<crate::tp_serve::TpEvent> = Vec::new();
            if !closed {
                loop {
                    // Capacity counts the in-flight prefill cursor too (it owns a physical slot).
                    let projected = self.num_active() + pending.len() + usize::from(self.pf.is_some());
                    if projected >= self.max_batch || self.lanes[projected].is_some() { break; }
                    match self.rx.try_recv() {
                        Ok(req) => pending.push(req),
                        Err(mpsc::error::TryRecvError::Empty) => break,
                        Err(mpsc::error::TryRecvError::Disconnected) => { closed = true; break; }
                    }
                }
                // Idle: block for the next request, exactly like run() — the mirror blocks on the
                // session stream meanwhile, so both ranks sleep in step.
                if self.num_active() == 0 && pending.is_empty() && self.pf.is_none() && !closed {
                    match self.rx.recv().await {
                        Some(req) => pending.push(req),
                        None => closed = true,
                    }
                }
            }
            // AT MOST ONE admission per step, and its `Admit` event is shipped in THIS step's event
            // list — the mirror admits exactly what the wire says, so the two ranks can never hold a
            // different number of cursors. (Draining several requests and admitting them all, as the
            // pre-cursor loop did, is impossible now: only one prefill cursor may exist.)
            let admit_now = self.pf.is_none() && !pending.is_empty();
            // Shutdown only when NOTHING is left — including an in-flight prefill cursor: sending
            // Shutdown while a cursor is mid-prefill would exit the mirror with a half-prefilled
            // lane on the head (§3.4's one new teardown hazard).
            if closed && pending.is_empty() && self.num_active() == 0 && self.pf.is_none() {
                for s in streams.iter_mut() {
                    crate::tp_serve::send_serving(s, &crate::tp_serve::ServingMsg::Shutdown)?;
                }
                return Ok(());
            }
            for i in 0..self.num_active() {
                if self.lanes[i].as_ref()
                    .map_or(false, |l| l.tx.is_closed() && !l.tp_cancelled) {
                    self.lanes[i].as_mut().unwrap().tp_cancelled = true;
                    events.push(crate::tp_serve::TpEvent::Cancel { lane: i });
                }
            }
            if admit_now {
                events.push(crate::tp_serve::TpEvent::Admit((&pending[0]).into()));
            }
            // Fan the SAME per-step event list out to every node (world-1 mirrors). The head is the
            // hub; every mirror must replay an identical scheduler state at the same step index.
            let msg = crate::tp_serve::ServingMsg::Step(
                crate::tp_serve::StepEvents { step, events });
            for s in streams.iter_mut() {
                crate::tp_serve::send_serving(s, &msg)?;
            }
            if admit_now { self.admit(pending.remove(0)); }
            // §3.5: the window advance sits at a FIXED POINT in the step — after the Step message
            // has been shipped and after admissions, before `decode_step`. That is what keeps every
            // prefill all-reduce pair-able by the mirror (the barrier structure changes in
            // ORDERING relative to decode's own collectives, never in COUNT).
            self.pf_advance();
            let b = self.num_active();
            if b > 0 { self.decode_step(b); self.tp_agree_step(step)?; }
            // A cursor can exist while b == 0 (the whole batch is prefilling): nothing to agree
            // about yet, but the step index advances on both ranks identically because the cursor
            // policy is a pure function of state (one cursor, one window, no wall clock).

            step += 1;
            // D3: serving-mode barrier histogram. --tp-trace already timestamps every barrier
            // (K1 duration / peer bounce / K2 wait / whole barrier / gap) and the bench path dumps
            // it at exit; this makes the same table available while SERVING, every 512 steps, so
            // the fixed per-step cost can be split sync vs read without a bench harness.
            if b > 0 && step % 512 == 0 && crate::opts::var(crate::opt!("tp-trace")).is_ok() {
                self.gpu.tp_trace_dump(&format!("serve step {step}"));
            }
            // Yield to tokio so streaming handlers can flush SSE events between decode steps.
            tokio::task::yield_now().await;
        }
    }

    /// TP=2 serving mirror loop (node, rank 1): block on the head's per-step events, replay them in
    /// order, and run the identical `decode_step`. Scheduler state stays identical BY CONSTRUCTION —
    /// the forward's all-reduces keep both ranks bitwise in lockstep and every admission/cancel is
    /// replayed from the wire at the same step index. Tokens decode into a dummy channel whose
    /// receiver is held in `keepalive` forever, so a mirror lane's `tx.is_closed()` never fires on
    /// its own: cancels arrive exclusively as wire events (see `Lane::tp_cancelled`).
    pub async fn run_tp_mirror(mut self, mut stream: std::net::TcpStream) -> anyhow::Result<()> {
        self.tp_serving = true;
        let mut keepalive: Vec<mpsc::UnboundedReceiver<TokEvent>> = Vec::new();
        let mut step: u64 = 0;
        loop {
            let msg = match crate::tp_serve::recv_serving(&mut stream) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("[tp-mirror] session stream ended ({e:#}) — mirror exit at step {step}");
                    return Ok(());
                }
            };
            match msg {
                crate::tp_serve::ServingMsg::Shutdown => {
                    eprintln!("[tp-mirror] head shut down at step {step} — mirror exit");
                    return Ok(());
                }
                crate::tp_serve::ServingMsg::Step(se) => {
                    anyhow::ensure!(se.step == step,
                        "TP step desync: head sent step {}, mirror is at step {step}", se.step);
                    for ev in se.events {
                        match ev {
                            crate::tp_serve::TpEvent::Admit(w) => {
                                let (tx, rx) = mpsc::unbounded_channel::<TokEvent>();
                                keepalive.push(rx);
                                self.admit(w.into_request(tx));
                            }
                            crate::tp_serve::TpEvent::Cancel { lane } => {
                                match self.lanes.get_mut(lane).and_then(|l| l.as_mut()) {
                                    Some(l) => l.tp_cancelled = true,
                                    None => anyhow::bail!(
                                        "TP cancel for empty lane {lane} at step {step} — state desync"),
                                }
                            }
                        }
                    }
                    // Symmetric with the head: replay the admissions, advance the SAME one window
                    // of the SAME cursor, then decode. The replayed Admit events are the only input
                    // either rank uses, so the two cursors cannot diverge.
                    self.pf_advance();
                    let b = self.num_active();
                    if b > 0 { self.decode_step(b); self.tp_agree_step(step)?; }
                    step += 1;
                    tokio::task::yield_now().await;
                }
                other => anyhow::bail!("unexpected serving message on the step stream: {other:?}"),
            }
        }
    }

    fn num_active(&self) -> usize {
        self.lanes.iter().take_while(|l| l.is_some()).count()
    }

    /// S5F — record one speculation step (the τ-matrix harness's per-step telemetry). No-op
    /// unless `spec_steps_on` (set only by `run_spec_bench`).
    fn rec_step(&mut self, r: SpecStepRec) {
        if self.spec_steps_on { self.spec_steps.push(r); }
    }

    /// S5F — the on-engine τ-matrix driver: runs the REAL scheduler loop over a list of jobs,
    /// switching the speculation source between jobs (ONE process, ONE model load — the matrix's
    /// amortization; each job is a fresh lane: admit → prefill (+ DFlash2 prime) → decode steps →
    /// finish). Exercises the exact code path the server runs (Phase A/B routing, the lane steps,
    /// EOS handling, the policy tick under a forced/pinned policy). Returns (token streams,
    /// per-step records per job) — the τ/tok-s/step-breakdown inputs.
    ///
    /// The scheduler is CONSUMED (single use, like `run()`); the tokio runtime must be a
    /// current-thread runtime (the scheduler + the round are single-task by contract).
    pub async fn run_spec_bench(mut self, jobs: Vec<SpecBenchJob>, dump_tags: &[String])
        -> (Vec<Vec<u32>>, Vec<Vec<SpecStepRec>>, Vec<f32>) {
        assert!(self.step_dump.is_none() || dump_tags.len() == jobs.len(),
                "run_spec_bench: dump_tags must align with jobs");
        let mut streams: Vec<Vec<u32>> = Vec::with_capacity(jobs.len());
        let mut step_recs: Vec<Vec<SpecStepRec>> = Vec::with_capacity(jobs.len());
        let mut walls: Vec<f32> = Vec::with_capacity(jobs.len());
        for (k, job) in jobs.into_iter().enumerate() {
            self.mtp.set_spec_source(job.source);
            self.spec_steps.clear();
            self.spec_steps_on = true;
            if let Some(d) = self.step_dump.as_mut() {
                d.job_start(&dump_tags[k], &job.prompt, self.gpu.dev());
            }
            let (tx, mut rx) = mpsc::unbounded_channel::<TokEvent>();
            let job_t0 = std::time::Instant::now();
            let job_plen = job.prompt.len();
            self.admit(BatchRequest {
                prompt: job.prompt,
                max_new: job.max_new,
                temperature: job.temperature,
                top_p: job.top_p,
                top_k: job.top_k,
                rep_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                min_p: 0.0,
                min_new: 0,
                ignore_eos: false,
                tx,
                seed: Some(job.seed),
                ckpt_at: None,
                domain: job.domain,
                received_at: std::time::Instant::now(),
                image_embeds: None,
                image_spans: Vec::new(),
                schema: None,
            });
            // The bench harness has no scheduler loop of its own, so it drains the cursor inline —
            // the SAME window sequence (one window per call), just without interleaving. This keeps
            // `run_spec_bench` byte-equivalent to the pre-cursor inline admit.
            while self.pf.is_some() { self.pf_advance(); }
            // The prefill filled the prime sink — copy the prompt's tap rows into the dump.
            if let Some(d) = self.step_dump.as_mut() {
                d.job_prime(job_plen, self.df2_prime.as_ref(), self.gpu.dev());
            }
            // Step until the lane finishes (the admit may have REJECTED a too-long prompt — the
            // lane count stays 0 and the Finish event already carries the reason).
            loop {
                let b = self.num_active();
                if b == 0 { break; }
                self.decode_step(b);
                tokio::task::yield_now().await;
            }
            let mut toks = Vec::new();
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    TokEvent::Tok(t) => toks.push(t),
                    TokEvent::Finish { .. } => {}
                }
            }
            // Keep the source set for the next job's ADMIT (the lane priming consults it).
            self.spec_steps_on = false;
            if let Some(d) = self.step_dump.as_mut() { d.job_end(); }
            walls.push(job_t0.elapsed().as_secs_f32());
            streams.push(toks);
            step_recs.push(std::mem::take(&mut self.spec_steps));
        }
        if let Some(d) = self.step_dump.as_mut() { d.finish(); }
        (streams, step_recs, walls)
    }

    /// Prefill `req` into a free physical slot; emits the first generated token to the client.
    ///
    /// PREFIX REUSE. OpenWebUI and opencode resend the ENTIRE conversation every turn, so without this a
    /// chat pays prefill of its whole history on every turn — per-turn TTFT grows linearly and a session
    /// costs O(T²) in total prefill. We pick the free slot whose cached sequence is the longest prefix of
    /// this prompt and prefill only the suffix.
    /// PLAN/25 Phase 0: flip the coverage-trace op sequence on a rank whose `step_dump` is None
    /// (the TP node mirror — the flag arrives via TpConfig; on the head it is implied by the dump).
    pub fn set_cov_trace(&mut self, on: bool) { self.cov_trace = on; }

    fn admit(&mut self, req: BatchRequest) {
        // W2: the request's compiled schema (moved into the lane below).
        let req_schema = req.schema.clone();
        // R9: register the live gpu+state once so net::agree's mismatch path can dump GDN state.
        if crate::opts::var(crate::opt!("tp-diag")).is_ok() { r9_register_state(&self.gpu, &self.state); }
        // Each free slot offers TWO points we could resume from, because we hold the GDN state at two
        // moments of its last request:
        //
        //   LIVE  — the slot's current state, at the end of the last GENERATION (`slot_cache`).
        //           Hits when the client replays our tokens verbatim (opencode does).
        //   CKPT  — the prompt checkpoint, at the end of the last PREFILL (`slot_ckpt_seq`).
        //           Hits when the client re-renders our turn instead (every tool-calling agent does),
        //           and when a new conversation reuses the same system/tool preamble.
        //
        // Either is usable only if its sequence is a STRICT prefix of this prompt: the recurrent state
        // sits immediately after its last token, so we need at least one token left to prefill and we
        // cannot resume from a point the state never occupied. Prefer the longer.
        #[derive(Clone, Copy, PartialEq)]
        enum From_ { Live, Ckpt, Ring(usize) }
        // Image-content identities for this request (empty for text-only). A prefix candidate whose
        // reused range includes a DIFFERENT image is discarded below, so the visual content of the
        // shared prefix is never replayed from a stale image.
        let req_imgs = request_image_identities(req.image_embeds.as_ref(), &req.image_spans);
        // Part-B (load invariance): reuse ONLY at ring-grid points (absolute-position multiples of
        // RING_CKPT_STRIDE). A Live/Ckpt hit resumes at an arbitrary length, which puts the first
        // prefill window's start mid-grid — its width differs run-to-run with the cache state, and
        // window width perturbs the prefill hiddens by ulps (the documented chunked-prefill caveat).
        // That made the SAME request's token stream depend on WHO ELSE had used the box: the
        // load-invariance failure. Ring-only reuse makes the window decomposition a pure function
        // of plen — [0,512),[512,1024),...,[512k,plen) — identical for cold, warm, solo, and
        // loaded runs, because the ring snapshot at 512k was itself produced by that same
        // window sequence (snapshots are bit-exact). Cost: reuse precision drops from token-exact
        // to the 512-token grid — a hit re-prefills up to RING_CKPT_STRIDE-1 extra tokens
        // (<=0.75 s at ~690 tok/s TP2 prefill; typical ~0.35 s), and verbatim-replay clients
        // (opencode) re-prefill the round-down remainder instead of skipping 100%.
        // --prefix-grid-reuse=0 restores token-exact Live/Ckpt reuse (diagnostics-only A/B).
        let grid_reuse = crate::opts::var(crate::opt!("prefix-grid-reuse")).map_or(true, |v| v != "0");
        let best = if !self.prefix_cache { None } else { self.free_slots.iter().enumerate()
            .flat_map(|(i, &sl)| {
                let live = common_prefix_len(&self.slot_cache[sl], &req.prompt);
                let ckpt = common_prefix_len(&self.slot_ckpt_seq[sl], &req.prompt);
                let mut cands = Vec::new();
                if !grid_reuse {
                    cands.push((i, sl, live, From_::Live, self.slot_cache[sl].len(), live));
                    cands.push((i, sl, ckpt, From_::Ckpt, self.slot_ckpt_seq[sl].len(), live));
                }
                // A6 ring checkpoints: entry j holds the GDN state at absolute length
                // slot_ring_len[sl][j] — reusable when that whole prefix matches (lcp ≥ len).
                // The live LCP against slot_cache bounds every ring candidate at once.
                if live > 0 {
                    for (j, &rlen) in self.slot_ring_len[sl].iter().enumerate() {
                        if rlen > 0 && rlen <= live && rlen < req.prompt.len() {
                            cands.push((i, sl, rlen, From_::Ring(j), rlen, live));
                        }
                    }
                }
                cands
            })
            .filter(|&(_, sl, l, from, seq_len, _live)|
                l > 0 && match from {
                    // live/ckpt: the state sits at the END of the cached sequence — the cache
                    // must be a strict prefix (at least one token left to prefill).
                    From_::Live | From_::Ckpt => l == seq_len && l < req.prompt.len(),
                    // ring: the state sits at rlen — already filtered to rlen ≤ lcp above.
                    From_::Ring(_) => true,
                }
                && match from {
                    From_::Live => images_compatible(&self.slot_cache_images[sl], &req_imgs, l),
                    From_::Ckpt => images_compatible(&self.slot_ckpt_images[sl], &req_imgs, l),
                    From_::Ring(_) => images_compatible(&self.slot_cache_images[sl], &req_imgs, l),
                })
            .max_by_key(|&(_, _, l, _, _, _)| l) };

        let (phys, reuse, from, live_at_admit) = match best {
            Some((idx, sl, l, f, _, live)) => { self.free_slots.remove(idx); (sl, l, Some(f), live) }
            None => match self.free_slots.pop() {
                Some(s) => (s, 0, None, common_prefix_len(&self.slot_cache[s], &req.prompt)),
                None => {
                    // NO SILENT DROPS (P10 lesson): dropping a BatchRequest drops its `tx`, which
                    // closes the client's SSE stream with no tokens and no error — the same
                    // client-visible symptom as the pf8 OOB class. Capacity is the CALLER's job, so
                    // reaching here means the caller's accounting and the slot table disagree;
                    // say so, and tell the client instead of hanging it.
                    eprintln!("[req] DROPPED: no free physical slot (free_slots=0) — caller capacity                                accounting disagrees with the slot table");
                    let _ = req.tx.send(TokEvent::Finish { reason: "capacity".to_string() });
                    return;
                }
            },
        };

        // Resuming from the checkpoint (or a ring entry) means winding the slot's GDN state BACK
        // to that point. (KV is untouched: positions 0..reuse still hold this very prefix's keys,
        // and the suffix prefill overwrites everything after.)
        match from {
            Some(From_::Ckpt) => {
                self.gpu.copy_gdn_slot(&self.state, self.prompt_ckpt_slot + phys, phys);
            }
            Some(From_::Ring(j)) => {
                self.gpu.copy_gdn_slot(&self.state, self.ring_ckpt_slot + phys * RING_CKPT_K + j, phys);
                if crate::opts::var(crate::opt!("dump-pfhash")).is_ok() {
                    // Phase-8 [pfhash]: state ACTUALLY sitting in the slot right after the ring
                    // restore — compare against [ring-hash] printed at snapshot time (cold run).
                    eprintln!("[pre-hash] reuse={}{}", reuse, self.gpu.pf_hash(&self.state, phys, None));
                }
            }
            _ => {}
        }
        // Ring entries past the reuse point are invalidated: their positions get re-occupied by
        // the new prefill's snapshots. Entries at lengths ≤ reuse stay valid (state is a pure
        // function of the token prefix, which did not change).
        let src_early = self.mtp.spec_source();
        for rlen in self.slot_ring_len[phys].iter_mut() {
            if *rlen > reuse { *rlen = 0; }
        }
        // A6: the DSpark ring rows [0, reuse) are exactly valid when the previous dspark lane in
        // THIS slot covered them (single-tenant b==1 ring; rows are a pure function of the token
        // prefix). Carry the ring over (rewind nprev) instead of degrading the lane to MTP.
        // IDENTITY (PLAN/DSPARK_RING_IDENTITY_SPEC.md): length alone is NOT sufficient — the round
        // is ONE shared buffer while this bookkeeping is per slot, so an interleaved request on
        // another slot can leave the ring holding a foreign prefix. Require the claim to name THIS
        // slot; otherwise `rewind` would preserve rows that are not this prefix's and the lane
        // would draft from them (a tau loss, invisible to every losslessness gate).
        let dspark_ring_slot = self.dspark.as_ref().and_then(|d| d.ring_slot());
        // Diagnostics-only negative control (PLAN/DSPARK_RING_IDENTITY_SPEC.md §4 gate 2): drop
        // ONLY the identity term, reproducing the pre-guard length-only carry so the hazard can be
        // made to bite (the twin of DF2's --df2-carry-len-only). Never set outside a gate run.
        let dspark_ring_blind = crate::opts::var(crate::opt!("dspark-ring-blind")).is_ok();
        let dspark_carry = reuse > 0 && src_early == SpecSource::Dspark
            && self.slot_dspark_len[phys] >= reuse
            && (dspark_ring_slot == Some(phys) || dspark_ring_blind);
        if !dspark_carry {
            self.slot_dspark_len[phys] = 0;
            // The claim must be dropped alongside the length — every path that today zeroes the
            // length invalidates the identity too (mirrors `Df2Round::invalidate_ring`).
            if let Some(d) = self.dspark.as_mut() { d.invalidate_ring(); }
        }

        // On a miss, report the prefix we COULD have reused. This is the only way to see the size of
        // the opportunity we are leaving on the floor — and it is what exposed the 88% waste.
        if reuse == 0 {            if let Some((best, cached)) = self.free_slots.iter().chain(std::iter::once(&phys))
                .flat_map(|&sl| [(common_prefix_len(&self.slot_cache[sl], &req.prompt), self.slot_cache[sl].len()),
                                 (common_prefix_len(&self.slot_ckpt_seq[sl], &req.prompt), self.slot_ckpt_seq[sl].len())])
                .max_by_key(|&(l, _)| l) {
                if best > 64 {
                    eprintln!("[req] prefix MISS: {} of {} prompt tokens match a cached sequence of {} \
                               — unusable, the GDN recurrent state only exists at token {}",
                              best, req.prompt.len(), cached, cached);
                }
            }
        }
        let greedy = req.temperature < 1e-6;
        let temperature = req.temperature;
        let top_p = req.top_p;
        let top_k = req.top_k;
        let domain = req.domain;
        let tx = req.tx;
        let plen = req.prompt.len();
        let prompt = req.prompt;
        let max_new = req.max_new;
        let min_new = req.min_new;
        let ignore_eos = req.ignore_eos;
        let req_ckpt_at = req.ckpt_at;
        let rep_penalty = req.rep_penalty;
        let presence_penalty = req.presence_penalty;
        let frequency_penalty = req.frequency_penalty;
        let _has_penalty = rep_penalty > 1.0 || presence_penalty > 0.0 || frequency_penalty > 0.0;
        let will_use_mtp = self.mtp.active();
        if mtp_dispatch_trace() {
            eprintln!("[b31-trace] admit: active={} depth={} src={:?} reuse={} plen={}",
                      will_use_mtp, self.mtp.depth(), self.mtp.spec_source(), reuse, plen);
        }
        // S5F: does THIS lane take the DFlash2 path? Requires the source + a resident round + a
        // full prompt prime (reuse == 0 — a prefix-hit lane's ring cannot be trusted for the
        // reused prefix, so it falls back to MTP/batched; prefix-cache + DFlash2 is out of scope
        // for S5F and documented as such).
        let src = self.mtp.spec_source();
        // DF2_CARRY: the DFlash2 twin of the A6 prefix-cache carve-out. The exact conditions —
        // and why a length-only copy of DSpark's check would be wrong (the DF2 ring is MODULO,
        // DSpark's is LINEAR) — are documented on `df2_carry_ok` and unit-tested in `mod df2_carry`.
        // Capture the ring state BEFORE any invalidation below: the log must report the guard's
        // INPUTS, not the state it leaves behind (reading it after made a refused carry print as
        // if the ring had never been bound).
        let (ring_slot_in, ring_len_in) = self.df2.as_ref()
            .map(|d| (d.ring_slot(), d.ring_len())).unwrap_or((None, 0));
        let df2_carry = self.df2.is_some() && df2_carry_ok(
            self.df2_carry_enabled, is_df2_src(src), reuse, plen, phys,
            ring_slot_in, ring_len_in, crate::opts::var(crate::opt!("df2-carry-len-only")).is_ok());
        // Does the ring's claim on THIS slot survive this admit?
        //
        // The claim is "rows [N-RING, N) hold the k/v of tokens [0, N) of this slot's cached
        // sequence". This admit replaces that cached sequence with the new prompt, so the claim
        // holds afterwards exactly when the new prompt still agrees with the sequence the ring was
        // built from over the whole range the claim covers — [0, min(N, plen)). `live_at_admit` is
        // the match length against the OLD cached sequence, so the test is `live >= min(N, plen)`.
        //
        // Dropping the claim whenever we simply do not carry is WRONG, and measurably so: a
        // shorter follow-up turn (plen < N — e.g. a retry, or a sub-agent re-sending only the
        // shared preamble) fails the frontier test for its own legitimate reason, and the blunt
        // rule then killed the claim for the rest of the session, pinning every later turn to MTP.
        // The refusal is per-request; the claim is about identity, and identity usually survives.
        let ring_claim_survives = ring_len_in > 0 && live_at_admit >= ring_len_in.min(plen);
        if !df2_carry && !ring_claim_survives
            && self.df2.as_ref().is_some_and(|d| d.ring_slot() == Some(phys)) {
            if let Some(d) = self.df2.as_mut() { d.invalidate_ring(); }
        }
        if reuse > 0 {
            let kind = match from {
                Some(From_::Live) => "live",
                Some(From_::Ckpt) => "ckpt",
                Some(From_::Ring(j)) => "ring",
                None => unreachable!(),
            };
            // DF2_CARRY telemetry (spec §6.7): print the guard's own INPUTS, so a refusal is
            // diagnosable rather than inferred. `ring_len` is the frontier N that `N <= plen`
            // bounds. All four booleans are what `df2_carry_ok` actually evaluated.
            eprintln!("[req] prefix HIT: reuse={reuse} of plen={plen} ({kind}, dspark_carry={dspark_carry}, blind={dspark_ring_blind}, \
                       df2_carry={df2_carry} [enabled={} df2_src={} ring_slot={ring_slot_in:?} phys={phys} \
                       ring_len={ring_len_in} reuse<=len={} len<=plen={} live={live_at_admit} claim_kept={}])",
                      self.df2_carry_enabled, is_df2_src(src),
                      ring_len_in >= reuse, ring_len_in <= plen, ring_claim_survives);
        }
        // W2: the DFlash2 round is not schema-masked in this release — a schema request keeps the
        // plain decode path instead of an unmasked draft/verify round.
        let will_use_df2 = will_use_mtp && is_df2_src(src)
            && self.df2.is_some() && self.df2_prime.is_some()
            && (reuse == 0 || df2_carry) && req_schema.is_none();
        // WI1: the DSpark twin of will_use_df2. A6 softens the prefix carve-out: a prefix-hit
        // lane whose slot carries a valid DSpark ring for the reused range (dspark_carry) keeps
        // the DSpark path — the ring rows [0, reuse) are exactly valid and only the suffix gets
        // primed. Without a carried ring the old rule stands (the ring cannot be rebuilt from
        // stored trunk state: the drafter consumes trunk tap hiddens, which are not retained).
        // P14: the v1 lane runs only on a lane whose ctx feature covers the WHOLE prompt (the
        // drafter conditions on every committed position, and a prefix-cache hit skips the prefill
        // that fills it) and only for greedy requests (v1 is greedy-only by construction).
        let will_use_dflash = will_use_mtp && src == SpecSource::DFlash && self.dflash.is_some()
            && reuse == 0 && greedy && req_schema.is_none();
        self.dflash_prime_done = false;
        self.dflash_prime_armed = will_use_dflash;
        if src == SpecSource::DFlash && self.dflash.is_none() && !self.dflash_fallback_logged {
            eprintln!("[dflash] SpecSource=dflash but no v1 lane is resident — serving via MTP \
                       (install the artifact with --draft-dir <dir>, or --spec-source without a drafter to serve MTP)");
            self.dflash_fallback_logged = true;
        }
        if src == SpecSource::DFlash && self.dflash.is_some() && !will_use_dflash
            && !self.dflash_carry_logged {
            eprintln!("[dflash] v1 lane NOT primed for this request (prefix reuse {reuse}, greedy \
                       {greedy}): serving it with MTP — the v1 lane never runs on a partial ctx");
            self.dflash_carry_logged = true;
        }
        let will_use_dspark = will_use_mtp && src == SpecSource::Dspark
            && self.dspark.is_some() && self.dspark_prime.is_some() && (reuse == 0 || dspark_carry)
            && req_schema.is_none();
        if src == SpecSource::Dspark && self.dspark.is_none() && !self.dspark_fallback_logged {
            eprintln!("[dspark] SpecSource=dspark but the DSpark round is NOT resident — serving via \
                       MTP per the standing-directive fallback (absent/failed artifact is never a \
                       hard failure)");
            self.dspark_fallback_logged = true;
        }
        if is_df2_src(src) && self.df2.is_none() && !self.df2_fallback_logged {
            eprintln!("[df2] SpecSource=DFlash2 but the DFlash2 round is NOT resident — serving via \
                       MTP per the standing-directive fallback (absent/failed artifact is never a \
                       hard failure)");
            self.df2_fallback_logged = true;
        }
        // TTFT fix 0 attribution (--prefill-trace): admission-phase wall times. The window's
        // prefill_batch and mtp_prime_prompt each end with a sync, so their timers are honest
        // without extra syncs; `other` is the residue (pool trim, slot zeroing, lane bookkeeping).
        let received_at = req.received_at;
        let trace_pf = crate::opts::var(crate::opt!("prefill-trace")).ok().is_some();
        let admit_t0 = std::time::Instant::now();
        let mut pf_mark = admit_t0;
        let mut t_memsets = 0.0f64;
        let mut t_prefill = 0.0f64;
        let mut t_prime = 0.0f64;

        let cfg = self.gpu.cfg().clone();
        let h = cfg.hidden_size;

        // The KV cache is exactly `kv_stride` positions deep. `write_kv_prefill` had no bound check, so
        // an over-long prompt wrote past the end of it and corrupted the neighbouring allocation. The
        // server rejects these before they get here; this is the backstop for every other caller (the
        // bench paths admit requests directly).
        // B8 blocker B: the MTP draft/verify/re-prime step writes rows up to plen + max_new + depth,
        // so the PLAIN bound (max_new <= kv_stride - plen) is short by `depth` — a request that exactly
        // fills the context OOBs the MTP KV (the mtp_draft_step assert fires = a worker PANIC). Reserve
        // depth + 8 (the τ floor's slop) when MTP is active, using the policy MAX depth so a later
        // depth-upswitch can't cross the line mid-request. Plain decode needs no such headroom (1 slot
        // of slop). All inputs are replicated state, so both TP ranks reject identically.
        // (PR #4 expresses this as `decode_headroom(will_use_mtp)`; the depth it reserved is the
        // same MAX_AUTO_DEPTH + 8 — dev's clamp-before-reject refinement below is kept.)
        let mtp_headroom = decode_headroom(will_use_mtp);
        // BUG2 fix (2026-09-06, PLAN/FIX_8192_CHAT_TRUNCATION): CLAMP before rejecting. The server
        // grants room = max_seq_len - plen and cannot see this spec headroom, so with
        // --max-tokens >= max_seq_len every request that omitted max_tokens asked for the full
        // room and was REJECTED outright: completion_tokens=0, finish mapped to "length" — a
        // dead server for every default client (reproduced on master @ 286fdbd: plen 17 +
        // max_new 65519 + depth 8 + 8 > 65536). A request whose budget doesn't fit must still
        // generate what the KV cache holds (finish_reason: "length" is the contract for running
        // short); reject only when NOTHING fits (prompt + headroom alone fill the cache). The
        // bound this clamp enforces is exactly the one the reject used to enforce, so the
        // OOB-corruption protection is unchanged.
        let asked_max_new = max_new;
        let max_new = max_new.min(self.kv_stride.saturating_sub(plen + mtp_headroom));
        if max_new < asked_max_new {
            eprintln!("[req] max_tokens clamped {} -> {} (KV stride {} − plen {plen} − spec headroom \
                       {mtp_headroom}; finish=length if generation runs out)",
                      asked_max_new, max_new, self.kv_stride);
        }
        let reject_msg = if plen >= self.kv_stride {
            Some(format!("prompt is {plen} tokens but the KV cache holds {} — raise --max-seq-len",
                         self.kv_stride))
        } else if max_new == 0 {
            Some(format!("plen {plen} + spec headroom {mtp_headroom} leaves no generation room in the \
                          KV stride {} (asked {asked_max_new}) — raise --max-seq-len",
                         self.kv_stride))
        } else {
            None
        };
        if let Some(msg) = reject_msg {
            eprintln!("[req] REJECTED: {msg}");
            // Return the physical slot — it was popped above and losing it here permanently shrinks
            // capacity (handoff 6.10). Two consistency rules: (1) the push-back is deterministic, so
            // both TP ranks make the same next pick; (2) if the CKPT restore already ran, the slot's
            // LIVE state was wound back to the prompt checkpoint — the live metadata must follow it,
            // or a later Live pick would resume from a state the slot no longer holds.
            if from == Some(From_::Ckpt) {
                self.slot_cache[phys] = self.slot_ckpt_seq[phys].clone();
                self.slot_cache_images[phys] = self.slot_ckpt_images[phys].clone();
            }
            self.free_slots.push(phys);
            let _ = tx.send(TokEvent::Finish { reason: "context_length_exceeded".to_string() });
            return;
        }
        // (max_new was already clamped to the KV budget above the reject — BUG2 fix, 2026-09-06.)

        // Bound the pool. Safe here: `trim` synchronizes before freeing, and this runs before any GPU
        // work for this request. Without it the pool grows forever — see Pool::trim.
        self.pool.trim();

        // On a cache MISS, wipe the slot. On a HIT we must NOT: the GDN recurrent state and the conv1d
        // tail are exactly what we are reusing, and they only exist at the point the last request left
        // them. (KV beyond `reuse` is stale but unreachable — attention never reads past `pos`.)
        if reuse == 0 {
            self.gpu.zero_slot_state(&mut self.state, phys, self.kv_stride);
        } else {
            // E2 Fix 2: the hit leaves the slot's q4 dequant mirror intact and valid for [0, reuse),
            // but its rows PAST reuse are the previous request's — clamp the watermarks so this
            // request's windows re-dequant their own tail (KV past reuse gets overwritten).
            self.gpu.clamp_kv_mirrors(&mut self.state, phys, reuse);
        }
        let suffix = &prompt[reuse..];
        if reuse > 0 {
            eprintln!("[req] prefix hit ({}): {}/{} tokens cached, prefilling {} ({:.0}% skipped)",
                      match from { Some(From_::Ckpt) => "prompt checkpoint",
                                   Some(From_::Ring(_)) => "ring grid",
                                   _ => "live state" },
                      reuse, plen, suffix.len(), 100.0 * reuse as f32 / plen as f32);
        }

        // MTP prompt-prime: over main positions 0..plen-2, step t writes (h_t, embed(prompt[t+1]))
        // into the lane's MTP KV at position t. Then seed the cursor hidden h_prev = h at plen-1.
        if will_use_mtp && reuse == 0 {
            // Miss: a previously-finished lane may have left speculative KV here. alloc_zeros doesn't
            // zero, and stale KV → nondeterministic drafts. Compute-stream memset.
            // (b3, EXPERT_TTFT_PREFILL_RESPONSE): zero only the positions the prime will write,
            // [0, plen) — positions >= plen are written by drafting before it attends. The cache is
            // head-major ([nkv, kv_stride, hd] bf16, head h at h·kv_stride·hd·2), so a contiguous
            // `nkv·plen·...` memset would only reach head 0 — one memset per head. The old full-cache
            // memset was 1.07 GB ≈ 4.2 ms of every MTP-on admission's TTFT; the full extent stays on
            // any path with `reuse > 0` (the CKPT-restore case, where the safe bound is less obvious).
            if trace_pf { pf_mark = std::time::Instant::now(); }
            let pos_bytes = plen * cfg.head_dim * 2;
            let head_stride = self.kv_stride * cfg.head_dim * 2;
            let kc_ptr = *self.mtp_kc[phys].device_ptr();
            let vc_ptr = *self.mtp_vc[phys].device_ptr();
            // §4.1b ROOT-CAUSE FIX: iterate the SHARD-AWARE head count. Under --tp-shard-mtp the
            // draft cache holds mtp_kv_heads() heads (1 at world=4), but this loop ran
            // cfg.num_kv_heads (4) — memsets at +8/+16/+24 MB past the buffer, corrupting whatever
            // followed in the VA layout (TP flags/rings/pools). That corruption WAS the entire
            // "first-MTP-step transport stall": proxies wedged, payloads arrived empty, K2' tails
            // timed out, and the memsets themselves faulted at ...aa000 when they crossed unmapped
            // VA (GPU coredumps: memset32, 8704 B = plen·hd·2, deterministic grid). Every alloc
            // site must use mtp_kv_heads() — this was the one loop that didn't.
            let nkv = self.gpu.mtp_kv_heads();
            for h in 0..nkv {
                self.gpu.memset_compute_stream(kc_ptr + (h * head_stride) as u64, pos_bytes);
                self.gpu.memset_compute_stream(vc_ptr + (h * head_stride) as u64, pos_bytes);
            }
            if trace_pf { t_memsets += pf_mark.elapsed().as_secs_f64(); }
        }
        let mtp_kc_ptr = *self.mtp_kc[phys].device_ptr();
        let mtp_vc_ptr = *self.mtp_vc[phys].device_ptr();

        // CHUNKED PREFILL. Prefill activation memory is O(prompt length) — ~1.2 MiB/token on 9B,
        // more on 27B — so a single-shot prefill of a long prompt OOMs (256K on 27B ≈ 400 GB). Process
        // the suffix in windows of PREFILL_CHUNK: each window's buffers are bounded by the chunk, and
        // the KV/GDN/conv state carries across windows via pos_start (the same mechanism prefix reuse
        // already uses). A prompt <= PREFILL_CHUNK is ONE window == the old single-shot path, so short
        // prompts are byte-identical to before; only long prompts (which would otherwise OOM) get
        // re-chunked, which perturbs their prefill hiddens by ulps — outside the batch-invariance
        // contract (prefill feeds decode and verify identically), so greedy MTP stays lossless.
        //
        // The message-boundary checkpoint (prefix cache) is honoured by snapshotting the GDN state
        // at `c` when a window ends there. Part-B (load invariance): under grid reuse a window is
        // NEVER SPLIT at `c` — window ends stay a pure function of absolute position (the next
        // RING_CKPT_STRIDE multiple, then plen), never of cache state. The snapshot still fires
        // when `c` coincides with a window end; otherwise slot_ckpt_seq is refreshed at prompt end
        // (it is diagnostics-only now — Ckpt no longer resumes anything under grid reuse).
        let ckpt_at = req_ckpt_at.filter(|_| self.prefix_cache).filter(|&c| c > reuse && c < plen);
        let mut first_sent = false;
        let mut w0 = reuse;
        // window by window (the prefill captures them into the wide prime sink; the round consumes
        // failure).
        if will_use_df2 {
            if let Some(df2) = self.df2.as_mut() {
                // DF2_CARRY: on a carry, move the frontier to the reuse point and KEEP the ring
                // rows [0, reuse) — the suffix prime below only writes [reuse, plen), and its
                // ring rows are the complementary residues of the draft's read band, so the two
                // halves tile the band exactly. Cold start: full reset (which also drops the
                // identity claim). The window loop below already starts at w0 = reuse, and
                // prime_window sets nprev absolutely, so nothing else needs to change.
                if df2_carry { df2.rewind(reuse); } else { df2.reset(); }
            }
            if let Some(ps) = self.df2_prime.as_ref() {
                self.gpu.set_df2_prime_sink(ps.clone());
            }
        }
        if will_use_dspark {
            if let Some(ds) = self.dspark.as_mut() {
                // A6 carry: rewind to the reuse point (rows [0, reuse) are this very prefix's
                // drafter KV); cold start: full reset.
                if dspark_carry {
                    ds.rewind(reuse);
                } else {
                    ds.reset();
                }
            }
            if let Some(ps) = self.dspark_prime.as_ref() {
                self.gpu.set_dspark_prime_sink(ps.clone());
            }
        }
        // ---------------------------------------------------------------------------------------
        // PLAN/SCHEDULER_2LANE_WORKDOC.md §3.1 — hand the WINDOW LOOP to a per-lane cursor.
        //
        // Everything above this line is the DECISION half of admission and stays here: capacity,
        // the context-length reject, the prefix-cache/reuse pick, `will_use_*`/carry, the slot
        // wipe/zero, the MTP memsets, the DF2/DSpark reset-or-rewind and the prime-sink arming.
        // Everything below moves into `PfCursor` and is advanced by the step loop, so a long
        // prompt's prefill no longer runs to completion inside one scheduler iteration.
        //
        // ORDERING NOTE (the reason arming stays here): at most ONE cursor exists at a time
        // (`BatchScheduler::pf`), so admission can never overlap an in-flight prime — the shared
        // `df2`/`dspark` rounds are reset and armed exactly where they always were relative to
        // their own lane's windows.
        // ---------------------------------------------------------------------------------------
        //
        // V3 vision: upload the merged image embeddings. The splice is armed per WINDOW (in
        // `pf_advance`), never for the whole prefill: `state.vision_embeds` is a GLOBAL field, and
        // a co-resident lane's window or decode step must never see this request's splice (§3.4).
        let vision = match req.image_embeds {
            Some(emb) if !req.image_spans.is_empty() && !emb.is_empty() => {
                let hb: Vec<half::bf16> = emb.iter().map(|&x| half::bf16::from_f32(x)).collect();
                let buf = self.gpu.dev().htod_sync_copy(&hb).expect("vision htod");
                Some((buf, req.image_spans.clone()))
            }
            _ => None,
        };
        // Seed: use the explicit seed from the request, or derive from prompt hash + counter.
        // (Kept at admission so the cursor owns the whole lane-install payload.)
        let seed = req.seed.unwrap_or_else(|| {
            use std::hash::{Hash, Hasher};
            let mut sh = std::collections::hash_map::DefaultHasher::new();
            prompt.hash(&mut sh);
            sh.finish()
        });
        debug_assert!(self.pf.is_none(),
                      "one prefill cursor at a time — see BatchScheduler::pf (the shared df2/dspark \
                       rounds are primed window-by-window and cannot be interleaved)");
        self.pf = Some(Box::new(PfCursor {
            phys, plen, reuse, w0: reuse, ckpt_at, grid_reuse, req_imgs,
            schema: req_schema.clone(),
            will_use_mtp, will_use_df2, will_use_dspark, df2_carry, dspark_carry,
            first_tok: 0, first_sent: false, pf_hash_str: String::new(),
            df2_primed_ok: true, dspark_primed_ok: true,
            tx, greedy, domain, temperature, top_p, top_k, max_new, min_new, ignore_eos,
            rep_penalty, presence_penalty, frequency_penalty, seed, received_at,
            trace_pf, admit_t0, t_memsets, t_prefill, t_prime, vision,
            prompt,
        }));
        // P11 W5a (--prefill-sched inline, owner-blessed escape): drain the whole prefill inside
        // this admission — the pre-P10 schedule. Decode lanes wait the FULL prompt (max gap = the
        // entire prefill, the posture the cursor replaced). A/B + emergency use; the cursor stays
        // the default. The window formula/order is untouched — schedule-only.
        if self.pf_inline {
            while self.pf.is_some() { self.pf_advance(); }
        }
    }

    /// Advance the in-flight prefill cursor by ONE window (SCHEDULER_2LANE §3.1/§3.3).
    ///
    /// This is the `while w0 < plen` body of `admit()` moved VERBATIM (§3.2 — the window formula is
    /// a NUMERIC contract, not a schedule: the pf8 padding family fails only when the pool's
    /// power-of-two capacity lands in a width gap, so "same windows, different order" is the whole
    /// safety argument and a re-derived formula would silently re-open the §0.2 counter-case #2
    /// class). No-op when nothing is prefilling.
    ///
    /// Budget policy (§3.3 option 1): ONE window per loop iteration, no knob. Deterministic — a
    /// pure function of scheduler state — which is what `run_tp_mirror`'s replay-by-wire contract
    /// requires (a wall-clock or memory-pressure budget would desync the ranks).
    fn pf_advance(&mut self) {
        let Some(mut c) = self.pf.take() else { return };
        let phys = c.phys;
        let (plen, w0) = (c.plen, c.w0);
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        if w0 >= plen {
            // reuse == plen (a full prefix hit): today's inline loop never ran either, so the
            // post-loop tail must still execute (it streams the first token and installs the lane).
            self.pf_finish(*c);
            return;
        }
        let mut w1 = (w0 + PREFILL_CHUNK).min(plen);
        if !c.grid_reuse {
            // Token-exact reuse (diagnostics): resume points are arbitrary, so honour the
            // message boundary by splitting a window at `c` (the pre-grid-reuse behavior).
            if let Some(ck) = c.ckpt_at { if w0 < ck && ck < w1 { w1 = ck; } }   // stop at the boundary
        }
        // A6: stop at the next ring-checkpoint boundary so the GDN state passes exactly
        // through it and can be snapshotted below (prefix-cache re-chunking is the
        // documented caveat; the ring slack is bounded by RING_CKPT_STRIDE).
        if self.prefix_cache {
            let rb = ((w0 / RING_CKPT_STRIDE) + 1) * RING_CKPT_STRIDE;
            if w0 < rb && rb < w1 { w1 = rb; }
        }
        // PLAN/TRIPWIRE_SPEC.md §3.2 + SCHEDULER_2LANE §5 gate 4: the WINDOW CENSUS. `--pool-census`
        // only, and NEVER on for a timing run. This is the artifact the scheduler's P4 gate diffs:
        // the multiset of `n = w1 - w0` values `prefill_batch` is presented for a fixed prompt mix,
        // plus the pf8 pad arithmetic each one implies (the pool-luck class of §0.2 counter-case #2).
        crate::gpu::pf_census_window(w0, w1, h);

        // V3 vision: arm the splice for THIS window only. `state.vision_embeds` is global (the
        // splice is consumed host-side inside `prefill_batch`), so it is armed immediately before
        // and disarmed immediately after — a co-resident lane's window must never inherit it.
        match c.vision.as_ref() {
            Some((buf, spans)) => {
                self.state.vision_embeds = Some(buf.clone());
                self.state.vision_spans = spans.clone();
            }
            None => { self.state.vision_embeds = None; self.state.vision_spans.clear(); }
        }

        let mut pf_mark = std::time::Instant::now();
        // P14: the v1 lane's ctx feature is filled by the trunk's own tap capture during the REAL
        // prefill (a position-addressed D2D per tapped layer per window, inside
        // `prefill_batch_range`) — no token-by-token prefill anywhere. NOTE: that capture site was
        // MISSING until the P14 repair; this comment described the design while the code primed the
        // lane from a surface no forward had written (see the capture site's comment in gpu.rs).
        if self.dflash_prime_armed { self.gpu.set_dflash_wide_pos(w0); }
        let (tok, hw) = self.gpu.prefill_batch(
            &mut self.pool, &c.prompt[w0..w1], &mut self.state, phys, self.kv_stride, w0);
        self.state.vision_embeds = None;
        self.state.vision_spans.clear();
        if c.trace_pf { c.t_prefill += pf_mark.elapsed().as_secs_f64(); }
        c.first_tok = tok;   // only the LAST window's token (at plen-1) is the prompt's next token
        if w1 == plen && self.dflash_prime_armed {
            // The prompt's columns [0, plen) are captured: hand them to the drafter (its KV append
            // happens lazily in the first draft step) and mark the lane primed.
            self.dflash_prime(w1);
            // The lane struct is created later in the admit path, so record the prime on the
            // scheduler and let the dispatch accept either flag (the v1 lane is b == 1 only).
            if let Some(l) = self.lanes[0].as_mut() { l.df2_primed = true; }
            self.dflash_prime_done = true;
            self.dflash_prime_armed = false;
        }
        if w1 == plen && crate::opts::var(crate::opt!("dump-pfhash")).is_ok() {
            // Phase-8 [pfhash]: warm-vs-cold prefill equality probe (diagnostics).
            c.pf_hash_str = self.gpu.pf_hash(&self.state, phys, Some((&hw, h, w1 - w0)));
        }

        // TTFT (b2, EXPERT_TTFT_PREFILL_RESPONSE): stream the first token the moment the last
        // window's prefill produced it — BEFORE the MTP prime + cursor copy below. The prime
        // only gates DRAFTING, which runs in decode_step after admission completes, so ordering
        // is preserved and semantics are unchanged; the client just sees its first chunk
        // ~6-9 ms earlier on MTP-on servers (P10 removed from the TTFT window).
        if w1 == plen && !c.first_sent {
            // W2 (Phase 13): a schema-constrained lane does NOT emit the prefill's own first
            // token. The prefill selection has no schema mask (it is a different code path), so
            // that token is unconstrained — observed as a leading '```' in the first W2 boot.
            // It becomes the FEED token of the first masked decode step instead, which means
            // every token the client sees came from a masked selection.
            if c.schema.is_some() {
                c.first_sent = true;
            } else {
                let _ = c.tx.send(TokEvent::Tok(c.first_tok));
                c.first_sent = true;
            }
            // The mirror replays the same admit and prints its own line to the node log;
            // the head's line is the measurement's signal. (tp_serving is true on BOTH ranks
            // in TP serving mode, so it cannot gate this — only rank selection could, and
            // the two lines are harmless in separate logs.)
            eprintln!("[req] ttft={:.1}ms plen={}",
                      c.received_at.elapsed().as_secs_f64() * 1000.0, plen);
        }

        if c.will_use_mtp {
            // MTP prime pairs hidden[t] with token[t+1] for t in [w0, min(w1, plen-1)); position
            // plen-1 is never primed (no token plen to pair). hw's columns 0.. map to positions w0..
            let tok_end = w1.min(plen - 1);
            if tok_end > w0 {
                let mtp_kc_ptr = *self.mtp_kc[phys].device_ptr();
                let mtp_vc_ptr = *self.mtp_vc[phys].device_ptr();
                if c.trace_pf { pf_mark = std::time::Instant::now(); }
                self.gpu.mtp_prime_prompt(&mut self.pool, &hw, &c.prompt[w0 + 1..tok_end + 1],
                                          mtp_kc_ptr, mtp_vc_ptr, self.kv_stride, w0);
                if c.trace_pf { c.t_prime += pf_mark.elapsed().as_secs_f64(); }
            }
            // Cursor hidden = pre-norm h at the LAST prompt position, i.e. last column of the
            // final window.
            if w1 == plen {
                self.gpu.copy_hidden_col(*self.mtp_h_prev[phys].device_ptr(), &hw, (w1 - w0) - 1);
            }
        }
        if c.will_use_dspark {
            // WI1: the DSpark prime twin — same window taps (dspark::TAP_LAYERS captures),
            // same large-M gemm_tiled path, same fail-soft contract.
            if let (Some(ds), Some(ps)) = (self.dspark.as_mut(), self.dspark_prime.as_ref()) {
                if let Err(e) = ds.prime_window(&ps.taps, w1 - w0, w0) {
                    eprintln!("[dspark] prompt prime window {w0}..{w1} FAILED ({e:#}) — this lane \
                               will NOT take the DSpark path (falls back to MTP/batched)");
                    c.dspark_primed_ok = false;
                }
            }
        }
        if c.will_use_df2 {
            // S5F: prime the DFlash2 ring with THIS window's taps. prefill_batch synced at its
            // tail, so the window's capture D2Ds are complete before the round reads them.
            if let (Some(df2), Some(ps)) = (self.df2.as_mut(), self.df2_prime.as_ref()) {
                if let Err(e) = df2.prime_window(&ps.taps, w1 - w0, w0) {
                    eprintln!("[df2] prompt prime window {w0}..{w1} FAILED ({e:#}) — this lane \
                               will NOT take the DFlash2 path (falls back to MTP/batched)");
                    c.df2_primed_ok = false;
                    // DF2_CARRY: a failed prime can leave the ring half-overwritten, so the
                    // identity claim is no longer true — drop it rather than let a later
                    // request carry a partially primed ring.
                    df2.invalidate_ring();
                }
            }
        }
        self.pool.release_bf16(hw, h * (w1 - w0));

        // Snapshot the GDN state at the message boundary (prefix cache), before the next window.
        if Some(w1) == c.ckpt_at {
            self.gpu.copy_gdn_slot(&self.state, phys, self.prompt_ckpt_slot + phys);
            self.slot_ckpt_seq[phys] = c.prompt[..w1].to_vec();
            self.slot_ckpt_images[phys] = c.req_imgs.iter()
                .filter(|x| x.start < w1).copied().collect();
        }
        // A6: ring checkpoint at the boundary — round-robin over the K most-recent entries.
        // w1 < plen keeps the prompt-end state in the live/ckpt slots (it must stay the
        // generation-resume state, not a ring candidate).
        if self.prefix_cache && w1 < plen && w1 % RING_CKPT_STRIDE == 0 {
            let j = self.ring_cursor[phys];
            self.ring_cursor[phys] = (j + 1) % RING_CKPT_K;
            self.gpu.copy_gdn_slot(&self.state, phys,
                self.ring_ckpt_slot + phys * RING_CKPT_K + j);
            self.slot_ring_len[phys][j] = w1;
            if crate::opts::var(crate::opt!("dump-pfhash")).is_ok() && (w1 % 1024 == 0 || w1 >= plen - 512) {
                // Phase-8 [pfhash]: what the ring entry ACTUALLY holds at snapshot time, plus
                // the live slot state it was copied from — both hashed after the copy lands.
                eprintln!("[ring-hash] w1={} j={} src{} ring{}",
                          w1, j, self.gpu.pf_hash(&self.state, phys, None),
                          self.gpu.pf_hash(&self.state, self.ring_ckpt_slot + phys * RING_CKPT_K + j, None));
            }
        }
        c.w0 = w1;
        if c.w0 >= plen {   // the last window completed: the lane becomes visible NOW (§3.4)
            self.pf_finish(*c);
        } else {
            self.pf = Some(c);
        }
    }

    /// Complete an admission whose last window has run: `admit()`'s post-loop tail, moved verbatim.
    ///
    /// This is the ONLY place a partially-prefilled request becomes visible to the rest of the
    /// scheduler (§3.4): the prefix-cache entry (`slot_cache`/`slot_ckpt_seq`) is written here, not
    /// per window, so the matcher can never see a half-prefilled length; and the lane itself is
    /// installed here, so `num_active()`/`b`, the speculation gates and the cancel sweep all treat
    /// a mid-prefill request as if it did not exist.
    fn pf_finish(&mut self, mut c: PfCursor) {
        let phys = c.phys;
        self.state.vision_embeds = None;
        self.state.vision_spans.clear();

        if !c.pf_hash_str.is_empty() {
            eprintln!("[pfhash] plen={} reuse={} phys={} a0={}{}",
                      c.plen, c.reuse, phys, c.first_tok, c.pf_hash_str);
        }

        if self.prefix_cache && (c.ckpt_at.is_none() || c.grid_reuse) {
            // No boundary snapshot fired inside the suffix (a raw/non-chat request, one whose whole
            // prompt was already cached, or — grid reuse — a boundary that is not a window end
            // anymore). Snapshot where we ended; it is still the prompt boundary for THIS prompt.
            self.gpu.copy_gdn_slot(&self.state, phys, self.prompt_ckpt_slot + phys);
            self.slot_ckpt_seq[phys] = c.prompt.clone();
            self.slot_ckpt_images[phys] = c.req_imgs.clone();
        }

        // The slot's state now reflects the whole prompt. Decode extends this as tokens commit.
        self.slot_cache[phys] = c.prompt.clone();
        self.slot_cache_images[phys] = c.req_imgs.clone();
        // S5F: the DFlash2 prime is done — the prefill capture must not run for later lanes or
        // requests (it is a per-admit arm, disarmed here regardless of success).
        if c.will_use_df2 { self.gpu.set_df2_prime_off(); }

        let slot = self.num_active();
        if !c.first_sent {
            // No window ran (reuse == plen full-hit edge) — the loop never produced a token.
            if c.schema.is_some() {
                // W2: same rule as the w1 == plen site — the token stays internal (feed), the
                // client's first token comes from the masked decode path.
                c.first_sent = true;
            } else {
                let _ = c.tx.send(TokEvent::Tok(c.first_tok));
            }
            eprintln!("[req] ttft={:.1}ms plen={}",
                      c.received_at.elapsed().as_secs_f64() * 1000.0, c.plen);
        }
        if c.trace_pf {
            let other = c.admit_t0.elapsed().as_secs_f64() - c.t_memsets - c.t_prefill - c.t_prime;
            eprintln!("[pf-admit] plen={} memsets={:.2}ms prefill={:.2}ms prime={:.2}ms other={:.2}ms",
                      c.plen, c.t_memsets * 1000.0, c.t_prefill * 1000.0, c.t_prime * 1000.0,
                      other * 1000.0);
        }

        // PLAN/25 Phase 0: serving-side job marker. Only opens when no job is open (the bench
        // harness emits its own markers; a job left open by a mid-admit reject self-heals at the
        // next lane finish). Records interleave if lanes overlap — max-batch 1 is the clean case.
        if let Some(d) = self.step_dump.as_mut() {
            if !d.job_open() {
                self.cov_req_n += 1;
                let tag = format!("req{:04}", self.cov_req_n);
                d.job_start(&tag, &c.prompt, self.gpu.dev());
            }
        }
        self.lanes[slot] = Some(Lane {
            phys, pos: c.plen, last_tok: c.first_tok, max_new: c.max_new,
            // W2: a schema lane emitted nothing yet (its prefill token is internal) — count from 0.
            generated: if c.schema.is_some() { 0 } else { 1 },
            greedy: c.greedy, domain: c.domain, temperature: c.temperature,
            top_p: c.top_p, top_k: c.top_k,
            rep_penalty: c.rep_penalty, presence_penalty: c.presence_penalty,
            frequency_penalty: c.frequency_penalty, min_new: c.min_new, ignore_eos: c.ignore_eos,
            history: vec![c.first_tok], tx: c.tx,
            schema_state: c.schema.as_ref().map(|sm| sm.start_state()).unwrap_or(0),
            schema: c.schema.clone(),
            mtp_pos: c.plen.saturating_sub(1),
            mtp_primed: c.will_use_mtp,
            mtp_stale: false,
            df2_primed: (c.will_use_df2 && c.df2_primed_ok) || (c.will_use_dspark && c.dspark_primed_ok),
            df2_stale: false,
            seed: c.seed,
            tp_cancelled: false,
        });
        if mtp_dispatch_trace() {
            eprintln!("[b31-trace] lane install: slot={} pos={} mtp_primed={} df2_primed={} greedy={} schema={}",
                      slot, c.plen, c.will_use_mtp,
                      (c.will_use_df2 && c.df2_primed_ok) || (c.will_use_dspark && c.dspark_primed_ok),
                      c.greedy, c.schema.is_some());
        }
        // Device-resident loop: the lane composition changed — the next batched step must
        // re-upload the full buffer set (tokens/pos/slot_ids/ring/keys).
        self.resident_dirty = true;
        // S-B16: (re)arm the audit cursor for this request from the lane's INSTALL state —
        // gt_last/gt_pos must be the prefill token and its position, captured BEFORE any MTP step
        // advances the lane. The shadow's zero+prime decode is deferred to the first audited step
        // (the cursor is only consumed by lanes that actually take the MTP path).
        if self.audit_on {
            // The shadow must start from a REAL prefill of this request's prompt. Zeroed KV rows
            // are NOT softmax-neutral (exp(0)=1 per unwritten key): the v2 auditor's zero-only
            // shadow predicted token 0 at every request's first token (false divergence, caught
            // in the first v2 run). This mirrors bench_mtp_steps' slot-1 ground-truth prefill.
            self.gpu.zero_slot_state(&mut self.state, self.audit_slot, self.kv_stride);
            self.gpu.zero_slot_kv(&mut self.state, self.audit_slot, self.kv_stride);
            let (a0s, hs) = self.gpu.prefill_batch(&mut self.pool, &c.prompt, &mut self.state,
                                                   self.audit_slot, self.kv_stride, 0);
            let mw = self.gpu.mtp_hidden_width();
            self.pool.release_bf16(hs, mw * c.plen);
            if a0s != c.first_tok {
                eprintln!("[lossless-audit] WARN req={} shadow prefill a0={} != lane a0={}",
                          self.audit_req_n, a0s, c.first_tok);
            }
            // The cursor's FIRST prediction must be real: with pred left at 0 the request's first
            // emission compares against token 0 (run4's false "seq=0" divergence at emitted#0).
            let pred0 = self.audit_shadow_decode(c.first_tok, c.plen, self.kv_stride);
            self.audit_cur[slot] = Some(MtpAuditCursor {
                gt_last: c.first_tok, gt_pos: c.plen, pred: pred0, idx: 0,
                tag: self.audit_req_n, diverged: false, primed: true,
            });
            self.audit_req_n += 1;
        }
    }


    /// W2 (Phase 13): per-lane vocabulary masks for the CURRENT FSM state of every lane in
    /// `batch_idx`, in device (batch) order. Returns (words, flags, any) for
    /// `GpuModel::upload_decode_masks`; `any == false` means "no schema lane present" and the
    /// caller uploads nothing (so the normal path is untouched).
    fn schema_decode_masks(&self, batch_idx: &[usize]) -> (Vec<u32>, Vec<i32>, bool) {
        let words = crate::gpu::mask_words_for(self.gpu.cfg().vocab_size);
        let any = batch_idx.iter()
            .any(|&i| self.lanes[i].as_ref().map_or(false, |l| l.schema.is_some()));
        if !any { return (Vec::new(), Vec::new(), false); }
        let mut w = vec![0u32; words * batch_idx.len()];
        let mut f = vec![0i32; batch_idx.len()];
        for (k, &i) in batch_idx.iter().enumerate() {
            if let Some(l) = self.lanes[i].as_ref() {
                if let Some(sm) = &l.schema {
                    let bits = sm.allowed(l.schema_state);
                    let len = bits.len().min(words);
                    w[k * words..k * words + len].copy_from_slice(&bits[..len]);
                    f[k] = 1;
                }
            }
        }
        (w, f, true)
    }

    /// W2: advance a lane's schema FSM by the tokens it just COMMITTED. Returns true when the
    /// document is complete (the turn is over: the JSON is finished) or the machine hit a state
    /// that can never be completed.
    fn schema_advance(&mut self, i: usize, toks: &[u32]) -> bool {
        let lane = self.lanes[i].as_mut().unwrap();
        let sm = match lane.schema.clone() { Some(sm) => sm, None => return false };
        for &t in toks {
            lane.schema_state = match sm.step(lane.schema_state, t) {
                Some(ns) => ns,
                None => {
                    eprintln!("[schema] lane {i}: token {t} is invalid in the current schema state \
                               — finishing the turn (this must not happen with the mask armed)");
                    return true;
                }
            };
        }
        sm.is_complete(lane.schema_state)
    }

    /// One scheduler decode step over the `b` active (front-packed) lanes. Two phases:
    ///   A. MTP-eligible lanes (greedy → mtp_lane_step; sampling → mtp_lane_step_sample when
    ///      stochastic MTP is on) are served one at a time — each emits 1+ tokens.
    ///   B. All remaining lanes are served by a single batched decode (one shared weight read),
    ///      each emitting exactly one token.
    fn decode_step(&mut self, b: usize) {
        let mut finished = vec![false; b];
        // Cancel lanes whose client is gone (disconnect, or the SSE generator dropped the receiver
        // on a stop-string hit): every send would silently fail and the lane would otherwise decode
        // to EOS/max_new, holding a batch slot and a share of every batched step. The compaction at
        // the end of the step does the teardown (frees the slot, attempts the Finish event).
        for i in 0..b {
            // TP serving takes its cancels from the wire (`tp_cancelled`), never from `is_closed()`
            // directly — a disconnect landing between the head's per-step detection and this sweep
            // would otherwise finish the lane a step early on the head only. Single-node is exactly
            // as before (`is_closed()` alone); the mirror's dummy txs never close on their own.
            if self.lanes[i].as_ref()
                .map_or(false, |l| l.tp_cancelled || (!self.tp_serving && l.tx.is_closed())) {
                finished[i] = true;
            }
        }

        // Phase A: per-lane MTP. Greedy lanes verify by argmax (bitwise lossless); sampling lanes
        // verify by speculative rejection sampling (distribution-exact). The split is decided by the
        // request's temperature, not by any server flag.
        //
        // MTP IS A SINGLE-LANE PATH -- this loop runs one lane at a time, each doing its own draft and
        // verify forwards. With two clients that is two full model passes per step, so concurrent
        // throughput was FLAT: measured 37.7 tok/s alone, 21.4 each with two clients, 10.7 each with
        // four, aggregate pinned at ~43 the whole way. Each client simply got 1/N of the machine.
        //
        // But BATCHING IS FREE HERE, and for the same reason speculation is: the quantized GEMM is a
        // fixed-shape kernel with N padded to 16, so a decode step is bound by the weight bytes, which
        // do not change with the batch. A 4-lane batched step (Phase B) costs what a 1-lane step costs.
        //
        // So above one lane, batching beats speculation and it is not close: 4 lanes at ~1 forward
        // total, versus 4 lanes at ~4 speculative forwards for ~2.5 tokens each. Speculation only wins
        // when there is nothing to batch WITH. Hence: MTP at b == 1, batched decode at b >= 2.
        //
        // (The real prize is batching the MTP VERIFY across lanes -- attn_dispatch already takes
        // per-column slot_ids and pos, so the attention supports mixed slots; the GDN recurrent state
        // and the per-lane accept/rollback bookkeeping are what make it a project rather than a patch.
        // That would give both: N lanes AND ~2.5 tokens per lane per step.)
        //
        // (P11 note: the "real prize" above EXISTS — the FOREST path behind --mtp-lanes packs
        //  concurrent greedy lanes' chains into ONE verify (take ≤ 5, per-lane depth =
        //  MAX_VERIFY/p − 1), default OFF pending the promote-or-kill measurement. The b==1
        //  split stands only for the ROUND-based drafters (df2/dspark own one shared ring).)
        let policy_active = self.mtp.active();
        // S5F: which speculation source is live THIS step. DFlash2 (S4F's integrated round) runs
        // ONLY at b==1 (AGENTS §4 — same as the MTP chain), for a lane that primed the round and
        // never went stale. The MTP path stays live for the Mtp/Dspark sources AND as the DFlash2
        // fallback (unprimed lane, failed prime, or a lane that went stale). Plain never speculates.
        let src = self.mtp.spec_source();
        let df2_live = policy_active && is_df2_src(src) && self.df2.is_some() && b == 1
            && !self.lanes.get(0).and_then(|l| l.as_ref())
                   .map_or(false, |l| l.schema.is_some());
        // WI1: the DSpark lane owns the step exactly like df2_live (b==1, primed, not stale).
        let dspark_live = policy_active && !df2_live && src == SpecSource::Dspark
            && self.dspark.is_some() && b == 1
            && !self.lanes.get(0).and_then(|l| l.as_ref())
                   .map_or(false, |l| l.schema.is_some());
        // P14: the DFlash v1 lane is a single-sequence block drafter — b == 1 only, exactly like
        // the other round-based lanes (AGENTS §4: above one lane, batching beats speculation).
        // W2 (Phase 13): a lane step that does not arm the JSON-schema mask would emit
        // UNCONSTRAINED tokens for a schema request (`df2/dspark/dflash/tree/forest/sampled` lane
        // steps are schema-blind today; only the greedy MTP chain step is wired). A schema lane is
        // therefore kept OFF every schema-blind source: it is served by the MTP chain when that is
        // live, else by the masked Phase-B decode — never silently by an unmasked lane.
        let schema_lane = self.lanes.get(0).and_then(|l| l.as_ref())
            .map_or(false, |l| l.schema.is_some() && b == 1);
        let dflash_live = policy_active && !df2_live && !dspark_live && src == SpecSource::DFlash
            && self.dflash.is_some() && b == 1 && !schema_lane;
        let mtp_live = policy_active && !df2_live && !dspark_live && !dflash_live && src != SpecSource::Plain;
        // `served[i]` = lane i was served by Phase A (speculation) this step — Phase B decodes
        // exactly the lanes Phase A did NOT serve (a lane is never double-served, never stranded).
        let mut served = vec![false; b];
        if mtp_dispatch_trace() && b == 1 {
            eprintln!("[b31-trace] dispatch: active={} src={:?} df2_live={} dflash_live={} dspark_live={} mtp_live={}",
                      policy_active, src, df2_live, dflash_live, dspark_live, mtp_live);
        }
        if df2_live {
            let lane = self.lanes[0].as_ref().unwrap();
            if lane.df2_primed && !lane.df2_stale {
                served[0] = true;
                // S8F (S6F adjudication): `DFlash2Auto` resolves to the per-request lane (greedy
                // on code, real-q on math/chat/prose); explicit sources stay explicit. The rq
                // source runs the SAMPLED selector path under the real-q verify (u*q < p);
                // dflash2 keeps the greedy drafts + q=1 default. P3(a) close: `prose_lane_greedy`
                // routes GREEDY (temp-0) General requests to the greedy-draft lane (the quad
                // sweep's +10.5% prose tau); sampled-temp General keeps the real-q walk.
                let src = df2_effective_src(src, lane.domain, self.prose_lane_greedy, lane.greedy);
                let done = match src {
                    SpecSource::DFlash2Rq => self.df2_lane_step_sample_rq(0),
                    // PLAN/25 Phase 1: the tree lane (greedy v0; sampled falls back to the rq walk).
                    SpecSource::DFlash2Tree if lane.greedy => self.df2_tree_step(0),
                    _ if lane.greedy => self.df2_lane_step(0),
                    _ => self.df2_lane_step_sample(0),
                };
                if done { finished[0] = true; }
            } else if lane.use_mtp(true) {
                // The DFlash2 lane degraded (never primed / prime failed / went stale): serve via
                // the MTP path — the standing directive's fallback, never a hard failure.
                served[0] = true;
                let df2_was_primed = lane.df2_primed;
                let is_greedy = lane.greedy;
                if df2_was_primed { self.lanes[0].as_mut().unwrap().df2_stale = true; }
                let done = if is_greedy { self.mtp_lane_step(0) } else { self.mtp_lane_step_sample(0) };
                if done { finished[0] = true; }
            }
        } else if dflash_live {
            let lane = self.lanes[0].as_ref().unwrap();
            // `df2_primed` is the generic "this lane's drafter state was primed at admit" flag
            // (the v1 prime is a ctx-feature copy, the DF2 prime a ring walk — same contract).
            if lane.greedy && (lane.df2_primed || self.dflash_prime_done) && !lane.df2_stale {
                served[0] = true;
                let done = self.dflash_lane_step(0);
                if done { finished[0] = true; }
            } else if lane.use_mtp(true) {
                // Unprimed (prefix-cache reuse) or stale: the standing MTP fallback — never silent,
                // the admit path logs the reason once.
                served[0] = true;
                let was_primed = lane.df2_primed;
                let is_greedy = lane.greedy;
                if was_primed { self.lanes[0].as_mut().unwrap().df2_stale = true; }
                let done = if is_greedy { self.mtp_lane_step(0) } else { self.mtp_lane_step_sample(0) };
                if done { finished[0] = true; }
            }
        } else if dspark_live {
            let lane = self.lanes[0].as_ref().unwrap();
            if lane.df2_primed && !lane.df2_stale {
                served[0] = true;
                // WI1 v1: the greedy drafts on greedy lanes, the q=1 rejection-sampling verify on
                // sampled lanes (the DF2 lane pair's shape; DSpark has no tree/selector variants).
                let done = if lane.greedy { self.dspark_lane_step(0) } else { self.dspark_lane_step_sample(0) };
                if done { finished[0] = true; }
            } else if lane.use_mtp(true) {
                // The DSpark lane degraded (never primed / prime failed / went stale): the MTP
                // fallback — the standing directive, never a hard failure.
                served[0] = true;
                let was_primed = lane.df2_primed;
                let is_greedy = lane.greedy;
                if was_primed { self.lanes[0].as_mut().unwrap().df2_stale = true; }
                let done = if is_greedy { self.mtp_lane_step(0) } else { self.mtp_lane_step_sample(0) };
                if done { finished[0] = true; }
            }
        } else if mtp_live {
            if self.mtp_lanes {
                // FOREST: pack the greedy, primed, non-stale lanes (penalty carried per-column) into ONE
                // verify. Overflow beyond the column budget and any single leftover run the single-lane
                // MTP path, so NO eligible lane sits out (which strands it on plain decode via mtp_stale).
                let forest: Vec<usize> = (0..b).filter(|&i| {
                    let l = self.lanes[i].as_ref().unwrap();
                    l.greedy && l.use_mtp(true)
                }).collect();
                for &i in &forest { served[i] = true; }
                let take = forest.len().min(5);   // 16-column budget: p=5 ⇒ depth 2; p=6 ⇒ depth 1 (no drafts — useless)
                if take >= 2 {
                    let packed: Vec<usize> = forest[..take].to_vec();
                    for (i, fin) in self.mtp_forest_step(&packed) { if fin { finished[i] = true; } }
                    for &i in &forest[take..] { if self.mtp_lane_step(i) { finished[i] = true; } }
                } else {
                    for &i in &forest { if self.mtp_lane_step(i) { finished[i] = true; } }
                }
                // Sampling (non-greedy) primed lanes keep the single-lane stochastic path (v1).
                for i in 0..b {
                    let l = self.lanes[i].as_ref().unwrap();
                    if l.use_mtp(true) && !l.greedy {
                        served[i] = true;
                        if self.mtp_lane_step_sample(i) { finished[i] = true; }
                    }
                }
                // A DFlash2-primed lane that just took an MTP/forest step can never resume the
                // round (its ring is missing this step's taps) — mirror the Phase-B stale rule.
                for i in 0..b {
                    let l = self.lanes[i].as_ref().unwrap();
                    if l.df2_primed && !l.df2_stale {
                        self.lanes[i].as_mut().unwrap().df2_stale = true;
                    }
                }
            } else if b == 1 {
                let lane = self.lanes[0].as_ref().unwrap();
                let is_greedy = lane.greedy;
                if mtp_dispatch_trace() {
                    eprintln!("[b31-trace] step b=1: use_mtp={} primed={} stale={} greedy={} schema={}",
                              lane.use_mtp(true), lane.mtp_primed, lane.mtp_stale, lane.greedy, lane.schema.is_some());
                }
                if lane.use_mtp(true) {
                    served[0] = true;
                    if lane.df2_primed && !lane.df2_stale {
                        // DFlash2 was live for this lane but this step runs MTP (e.g. the source
                        // flipped at a policy window) — the ring is now missing this step's taps.
                        self.lanes[0].as_mut().unwrap().df2_stale = true;
                    }
                    let done = if is_greedy { self.mtp_lane_step(0) } else { self.mtp_lane_step_sample(0) };
                    if done { finished[0] = true; }
                }
            }
        }
        // E17: the depth policy prices r(d) from the CURRENT context's bucket — feed it the deepest
        // live position (identical on both TP ranks: positions are lockstep).
        let mtp_ctx = self.lanes[..b].iter()
            .filter_map(|l| l.as_ref().map(|l| l.pos)).max().unwrap_or(0);
        self.mtp.tick(mtp_ctx);
        // B31 FIX: publish the MTP telemetry EVERY decode step, not only where the `[mtp]` meter
        // prints (every 50 steps). The 50-step cadence left a window where /health reported
        // drafts=0 on a server that was actively speculating — a pinned-d4 short FIRST request
        // finishes in ~32 steps and never crosses the boundary, so --probe-serve-pinned's
        // engagement assert read a false FAIL (B31, reproduced x3 on .13/.14). The publish is a
        // handful of Relaxed atomic stores against a ~70ms step; the watermark skips plain-only
        // steps. The 50-step LOG cadence is unchanged.
        if self.mtp_stat_steps != self.mtp_stat_published {
            crate::tel::publish_mtp(self.mtp_stat_steps, self.mtp_stat_drafts,
                                    self.mtp_stat_accepted, self.mtp_stat_emitted,
                                    self.mtp_stat_verify_fwds, self.mtp.depth(),
                                    &self.mtp.hazard_counts());
            self.mtp_stat_published = self.mtp_stat_steps;
        }
        // Device-resident loop: any MTP step emits 2+ tokens and advances pos by >1, which the
        // per-step ring push / pos increment cannot represent — the next batched step re-uploads.
        if policy_active { self.resident_dirty = true; }

        // Phase B: batched decode for the lanes Phase A did NOT serve (the served set is the
        // single source of truth — a source switch can never strand or double-serve a lane).
        let batch_idx: Vec<usize> = (0..b).filter(|&i| !served[i]).collect();
        if mtp_dispatch_trace() && policy_active && !batch_idx.is_empty() {
            eprintln!("[b31-trace] phaseB: b={} unserved={:?} (policy active but lanes fell to plain decode)",
                      b, batch_idx);
        }
        if !batch_idx.is_empty() {
            // W2 (Phase 13): ARM THE DECODE MASK for this batch. `batched_decode` already refuses
            // the resident loop and the captured graphs for a schema batch, but the mask itself was
            // never uploaded (`schema_decode_masks` / `upload_decode_masks` had no caller), so a
            // Phase-B step emitted UNMASKED — the escape that forced the W2 floor. The mask is
            // per-lane and per-step (the FSM state moves with every committed token).
            //
            // While no lane carries a schema this block does NOTHING — not one upload, not one
            // allocation — so the hot decode path is untouched by it.
            let armed = if batch_idx.iter().any(|&i| self.lanes[i].as_ref().is_some_and(|l| l.schema.is_some())) {
                let (words, flags, any) = self.schema_decode_masks(&batch_idx);
                if any { self.gpu.upload_decode_masks(&mut self.bufs, &words, &flags, true); }
                any
            } else {
                false
            };
            let next_toks = self.batched_decode(&batch_idx);
            // The mask kernel is armed for exactly this step: disarm it immediately after, so a
            // later non-schema caller can never inherit a stale mask.
            if armed { self.gpu.upload_decode_masks(&mut self.bufs, &[], &[], false); }
            for (k, &i) in batch_idx.iter().enumerate() {
                let t = next_toks[k];
                let lane = self.lanes[i].as_mut().unwrap();
                // This lane just advanced WITHOUT writing MTP KV: the head now has a hole at this
                // position and can never be trusted again for this request. See Lane::mtp_stale.
                lane.mtp_stale = true;
                // S5F: same for the DFlash2 ring — a batched step did not advance the round's nprev.
                lane.df2_stale = true;
                let _ = lane.tx.send(TokEvent::Tok(t));
                // THE CACHE RECORDS WHAT THE STATE HAS CONSUMED, NOT WHAT WE EMITTED. This step fed the
                // PREVIOUS token (`last_tok`) through the model at position `pos`; `t` is the model's
                // prediction and has not been fed yet. Caching `t` instead would put a token in the
                // cache that the state has never seen — and the next turn would reuse that state as if
                // it had, silently producing wrong output. Invariant: slot_cache.len() == lane.pos.
                let fed = lane.last_tok;
                let phys = lane.phys;
                lane.last_tok = t;
                lane.pos += 1;
                lane.generated += 1;
                lane.history.push(t);
                self.slot_cache[phys].push(fed);
                debug_assert_eq!(self.slot_cache[phys].len(), lane.pos);
                if lane.history.len() > 256 { lane.history.drain(0..128); }
                let eos_hit = self.eos.contains(&t) && !lane.ignore_eos && lane.generated >= lane.min_new;
                if eos_hit || lane.generated >= lane.max_new {
                    finished[i] = true;
                }
                // W2: advance the schema FSM over the token this step actually emitted, and end the
                // turn when the document is complete (a schema turn stops at the closing brace, it
                // does not run to max_tokens).
                if self.lanes[i].as_ref().unwrap().schema.is_some() {
                    let done = self.schema_advance(i, &[t]);
                    if done { finished[i] = true; }
                }
            }
        }

        // compact: keep non-finished lanes front-packed. Each lane keeps its physical slot —
        // finished lanes return their slot to the free list; no state copying needed.
        let mut write = 0usize;
        for i in 0..b {
            if finished[i] {
                let lane = self.lanes[i].take().unwrap();
                self.free_slots.push(lane.phys);
                let reason = if lane.generated >= lane.max_new { "length" } else { "stop" };
                let _ = lane.tx.send(TokEvent::Finish { reason: reason.to_string() });
                // PLAN/25 Phase 0: close the serving dump job (no-op when none is open; in bench
                // mode run_spec_bench's own job_end then no-ops the same way — taps are written
                // exactly once, here, under the bench tag).
                if let Some(d) = self.step_dump.as_mut() { d.job_end(); }
            } else {
                self.lanes[write] = self.lanes[i].take();
                write += 1;
            }
        }
        // Device-resident loop: lane composition changed (a lane finished) — re-upload next step.
        if write < b { self.resident_dirty = true; }
    }

    /// Build the per-position verify penalty from a lane's committed history (dedup, replicate to all
    /// MAX_AUTO_DEPTH positions). Shared by the chain and tree MTP paths. Returns None if no penalty.
    fn make_penalty(&mut self, history: &[u32], rep_pen: f32, presence_pen: f32, freq_pen: f32,
                    has_penalty: bool) -> Option<crate::gpu::VerifyPenalty> {
        if !has_penalty { return None; }
        let mp = crate::gpu::MAX_PEN_TOKENS;
        let cap = crate::gpu::MAX_VERIFY;   // buffers are MAX_VERIFY-sized (forest may span that many cols)
        let mut pen_tokens = vec![-1i32; cap * mp];
        let mut pen_counts = vec![0i16; cap * mp];
        let mut idx = 0usize;
        for &t in history.iter().rev().take(mp) {
            let ti = t as i32;
            match (0..idx).position(|j| pen_tokens[j] == ti) {
                Some(j) => { pen_counts[j] += 1; }
                None => { if idx < mp { pen_tokens[idx] = ti; pen_counts[idx] = 1; idx += 1; } }
            }
        }
        let head_t: Vec<i32> = pen_tokens[0..mp].to_vec();
        let head_c: Vec<i16> = pen_counts[0..mp].to_vec();
        for p in 1..cap {
            pen_tokens[p*mp..p*mp+mp].copy_from_slice(&head_t);
            pen_counts[p*mp..p*mp+mp].copy_from_slice(&head_c);
        }
        let (rv, pv, fv) = (vec![rep_pen; cap], vec![presence_pen; cap], vec![freq_pen; cap]);
        self.gpu.dev().htod_sync_copy_into(&pen_tokens, self.mtp_pen_tokens.as_mut().unwrap()).unwrap();
        self.gpu.dev().htod_sync_copy_into(&pen_counts, self.mtp_pen_counts.as_mut().unwrap()).unwrap();
        // The VALUES are request-constant: upload them only when they (or the width) change, not
        // every MTP step. The history token/count arrays above still go up per step.
        let key = (rep_pen.to_bits(), presence_pen.to_bits(), freq_pen.to_bits(), cap);
        if self.pen_const_key != Some(key) {
            self.gpu.dev().htod_sync_copy_into(&rv, self.mtp_pen_rep.as_mut().unwrap()).unwrap();
            self.gpu.dev().htod_sync_copy_into(&pv, self.mtp_pen_presence.as_mut().unwrap()).unwrap();
            self.gpu.dev().htod_sync_copy_into(&fv, self.mtp_pen_freq.as_mut().unwrap()).unwrap();
            self.pen_const_key = Some(key);
        }
        // NOTE: no dev().synchronize() here — the copies are host-blocking on the NULL stream and
        // the compute stream is the blocking stream (invariant I1), so ordering is already
        // guaranteed; the sync was a full pipeline drain of pure waste per penalized MTP step.
        Some(crate::gpu::VerifyPenalty {
            tokens_ptr: *self.mtp_pen_tokens.as_ref().unwrap().device_ptr(),
            counts_ptr: *self.mtp_pen_counts.as_ref().unwrap().device_ptr(),
            rep_pen_ptr: *self.mtp_pen_rep.as_ref().unwrap().device_ptr(),
            presence_ptr: *self.mtp_pen_presence.as_ref().unwrap().device_ptr(),
            freq_ptr: *self.mtp_pen_freq.as_ref().unwrap().device_ptr(),
        })
    }

    /// S-B16: MTP verify penalty with PER-COLUMN histories. Plain decode penalizes with the
    /// EVOLVING history (each emitted token enters it before the next argmax); the verify used
    /// to share ONE stale (committed-only) history across all n columns — accepted drafts'
    /// presence/frequency contributions were missed at every column > 0 — a real losslessness
    /// break when penalties are active, with exposure growing with depth (A/B d2 PASS vs d3/d4
    /// FAIL; run7 penalty-free was clean at every depth). Column t now gets H + drafts[0..t] —
    /// exactly what plain decode would have penalized at that column. The kernel already reads
    /// per-column slices (`pen_tokens[b * n_pen + i]`, per-column scalars), so this is a
    /// host-side fill change only.
    fn make_penalty_drafts(&mut self, history: &[u32], drafts: &[u32], rep_pen: f32,
                           presence_pen: f32, freq_pen: f32, has_penalty: bool)
                           -> Option<crate::gpu::VerifyPenalty> {
        if !has_penalty { return None; }
        let mp = crate::gpu::MAX_PEN_TOKENS;
        let cap = crate::gpu::MAX_VERIFY;   // buffers are MAX_VERIFY-sized (forest may span that many cols)
        let n = (drafts.len() + 1).min(cap); // verify width = 1 committed column + drafts
        let mut pen_tokens = vec![-1i32; cap * mp];
        let mut pen_counts = vec![0i16; cap * mp];
        // Column 0: the committed history (dedup, most-recent-first) — same as plain decode at
        // the first verify position.
        let mut idx = 0usize;
        for &t in history.iter().rev().take(mp) {
            let ti = t as i32;
            match (0..idx).position(|j| pen_tokens[j] == ti) {
                Some(j) => { pen_counts[j] += 1; }
                None => { if idx < mp { pen_tokens[idx] = ti; pen_counts[idx] = 1; idx += 1; } }
            }
        }
        // Column t = column t-1 + one accepted draft (the token plain decode just emitted and
        // would penalize with before the next argmax).
        for c in 1..n {
            let (dst, src) = (c * mp, (c - 1) * mp);
            pen_tokens.copy_within(src..src + mp, dst);
            pen_counts.copy_within(src..src + mp, dst);
            let ti = drafts[c - 1] as i32;
            match pen_tokens[dst..dst + idx].iter().position(|&x| x == ti) {
                Some(j) => { pen_counts[dst + j] += 1; }
                None => { if idx < mp { pen_tokens[dst + idx] = ti; pen_counts[dst + idx] = 1; idx += 1; } }
            }
        }
        // Columns ≥ n are unused (the verify launches grid = n) — fill with the deepest prefix
        // so no stale head copies linger if a wider launch ever reads them.
        for c in n..cap {
            let (dst, src) = (c * mp, (n - 1) * mp);
            pen_tokens.copy_within(src..src + mp, dst);
            pen_counts.copy_within(src..src + mp, dst);
        }
        let (rv, pv, fv) = (vec![rep_pen; cap], vec![presence_pen; cap], vec![freq_pen; cap]);
        self.gpu.dev().htod_sync_copy_into(&pen_tokens, self.mtp_pen_tokens.as_mut().unwrap()).unwrap();
        self.gpu.dev().htod_sync_copy_into(&pen_counts, self.mtp_pen_counts.as_mut().unwrap()).unwrap();
        let key = (rep_pen.to_bits(), presence_pen.to_bits(), freq_pen.to_bits(), cap);
        if self.pen_const_key != Some(key) {
            self.gpu.dev().htod_sync_copy_into(&rv, self.mtp_pen_rep.as_mut().unwrap()).unwrap();
            self.gpu.dev().htod_sync_copy_into(&pv, self.mtp_pen_presence.as_mut().unwrap()).unwrap();
            self.gpu.dev().htod_sync_copy_into(&fv, self.mtp_pen_freq.as_mut().unwrap()).unwrap();
            self.pen_const_key = Some(key);
        }
        Some(crate::gpu::VerifyPenalty {
            tokens_ptr: *self.mtp_pen_tokens.as_ref().unwrap().device_ptr(),
            counts_ptr: *self.mtp_pen_counts.as_ref().unwrap().device_ptr(),
            rep_pen_ptr: *self.mtp_pen_rep.as_ref().unwrap().device_ptr(),
            presence_ptr: *self.mtp_pen_presence.as_ref().unwrap().device_ptr(),
            freq_ptr: *self.mtp_pen_freq.as_ref().unwrap().device_ptr(),
        })
    }

    /// FOREST per-column penalty: column c gets ITS lane's rep/presence/freq penalty (from that lane's
    /// deduped committed history). `lanes` is (start, len, rep, presence, freq, history) per packed lane;
    /// columns outside any lane and past the packed width are no-ops (rep=1, presence=freq=0, tokens=-1).
    /// Returns None if NO packed lane has a penalty. Buffers are MAX_VERIFY-sized.
    fn make_forest_penalty(&mut self, lanes: &[(usize, usize, f32, f32, f32, Vec<u32>)])
                           -> Option<crate::gpu::VerifyPenalty> {
        let any = lanes.iter().any(|(_, _, r, pr, f, _)| *r > 1.0 || *pr > 0.0 || *f > 0.0);
        if !any { return None; }
        let mp = crate::gpu::MAX_PEN_TOKENS;
        let cap = crate::gpu::MAX_VERIFY;
        let mut pen_tokens = vec![-1i32; cap * mp];
        let mut pen_counts = vec![0i16; cap * mp];
        let mut rep_v = vec![1.0f32; cap];
        let mut pres_v = vec![0.0f32; cap];
        let mut freq_v = vec![0.0f32; cap];
        for (start, len, rep, pres, freq, history) in lanes {
            // Dedup this lane's recent history into one [mp] block.
            let mut ht = vec![-1i32; mp];
            let mut hc = vec![0i16; mp];
            let mut idx = 0usize;
            for &t in history.iter().rev().take(mp) {
                let ti = t as i32;
                match (0..idx).position(|j| ht[j] == ti) {
                    Some(j) => { hc[j] += 1; }
                    None => { if idx < mp { ht[idx] = ti; hc[idx] = 1; idx += 1; } }
                }
            }
            for c in *start..(*start + *len).min(cap) {
                pen_tokens[c * mp..c * mp + mp].copy_from_slice(&ht);
                pen_counts[c * mp..c * mp + mp].copy_from_slice(&hc);
                rep_v[c] = *rep; pres_v[c] = *pres; freq_v[c] = *freq;
            }
        }
        self.gpu.dev().htod_sync_copy_into(&pen_tokens, self.mtp_pen_tokens.as_mut().unwrap()).unwrap();
        self.gpu.dev().htod_sync_copy_into(&pen_counts, self.mtp_pen_counts.as_mut().unwrap()).unwrap();
        self.gpu.dev().htod_sync_copy_into(&rep_v, self.mtp_pen_rep.as_mut().unwrap()).unwrap();
        self.gpu.dev().htod_sync_copy_into(&pres_v, self.mtp_pen_presence.as_mut().unwrap()).unwrap();
        self.gpu.dev().htod_sync_copy_into(&freq_v, self.mtp_pen_freq.as_mut().unwrap()).unwrap();
        // The forest just overwrote the per-column penalty VALUES with per-lane ones — invalidate
        // the const cache so the next chain make_penalty re-uploads instead of trusting stale data.
        self.pen_const_key = None;
        // No dev().synchronize(): host-blocking NULL-stream copies + the blocking compute stream
        // already order these before the verify kernels that read them (invariant I1).
        Some(crate::gpu::VerifyPenalty {
            tokens_ptr: *self.mtp_pen_tokens.as_ref().unwrap().device_ptr(),
            counts_ptr: *self.mtp_pen_counts.as_ref().unwrap().device_ptr(),
            rep_pen_ptr: *self.mtp_pen_rep.as_ref().unwrap().device_ptr(),
            presence_ptr: *self.mtp_pen_presence.as_ref().unwrap().device_ptr(),
            freq_ptr: *self.mtp_pen_freq.as_ref().unwrap().device_ptr(),
        })
    }

    /// One FORK-THEN-CHAIN tree MTP step (greedy). Drafts a k=2 fork, verifies the tree, walks the
    /// accepted path (target argmax), compacts its KV to contiguous slots, adopts the accepted leaf's
    /// GDN checkpoint, re-primes MTP over the accepted path, and emits. Lossless: every emitted token is
    /// the target's argmax given its accepted prefix (same as the chain). Returns true if finished.
    fn mtp_tree_step(&mut self, i: usize) -> bool {
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let depth = self.mtp.depth();
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let ckpt = self.mtp_snapshot_slot;   // tree checkpoints base (slots ckpt..ckpt+n-2)
        let kv_stride = self.kv_stride;
        let mtp_kc_ptr = *self.mtp_kc[phys].device_ptr();
        let mtp_vc_ptr = *self.mtp_vc[phys].device_ptr();
        let h_save_ptr = *self.mtp_h_save.as_ref().unwrap().device_ptr();

        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
        let mtp_pos = self.lanes[i].as_ref().unwrap().mtp_pos;
        let generated = self.lanes[i].as_ref().unwrap().generated;
        let max_new = self.lanes[i].as_ref().unwrap().max_new;
        let eos = self.eos.clone();
        // Phase-8: the stop rule is a property of the REQUEST, not the serving path —
        // every emit loop must honor ignore_eos/min_new exactly like the batched decode path
        // (batch.rs' `eos_hit`), or spec-on and spec-off streams stop at different tokens.
        let (ignore_eos, min_new) = { let l = self.lanes[i].as_ref().unwrap(); (l.ignore_eos, l.min_new) };
        let (rep_pen, presence_pen, freq_pen, has_penalty) = {
            let l = self.lanes[i].as_ref().unwrap();
            (l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.has_penalty())
        };
        let history: Vec<u32> = self.lanes[i].as_ref().unwrap().history.clone();

        // h_save = pre-verify hidden (for re-prime column 0).
        self.gpu.copy_hidden_col(h_save_ptr, &self.mtp_h_prev[phys], 0);

        // n-gram context (the second branch's source, if enabled).
        let ngram = self.ngram_draft;
        let mut work: Vec<u32> = Vec::new();
        if ngram > 0 {
            work.extend_from_slice(&self.slot_cache[phys]);
            work.push(committed_tok);
        }

        // ---- Draft the fork-then-chain tree. ----
        let (parent, tokens) = self.gpu.mtp_fork_draft(
            &mut self.pool, &self.mtp_h_prev[phys], committed_tok as i32, mtp_pos, depth,
            mtp_kc_ptr, mtp_vc_ptr, kv_stride, &work, ngram);
        let n = tokens.len();

        // ---- Verify the tree. ----
        let penalty = self.make_penalty(&history, rep_pen, presence_pen, freq_pen, has_penalty);
        let topo = self.gpu.topo_from_parent(&parent, main_pos);
        let (preds, vout) = self.gpu.verify_forward_topo(
            &mut self.pool, &tokens, &mut self.state, phys, kv_stride, main_pos, Some(ckpt), penalty, Some(&topo));

        // ---- Accept walk: follow the target argmax down the tree. ----
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        for c in 1..n { children[parent[c] as usize].push(c); }
        let mut path = vec![0usize]; let mut emitted = Vec::new(); let mut cur = 0usize;
        loop {
            let want = preds[cur]; emitted.push(want);
            match children[cur].iter().copied().find(|&c| tokens[c] == want) {
                Some(c) => { path.push(c); cur = c; } None => break,
            }
        }
        let nacc = path.len() - 1;               // accepted drafts (emitted[0..nacc]); emitted[nacc]=bonus
        let leaf = *path.last().unwrap();

        // ---- Commit: compact the accepted path's KV, adopt the leaf's GDN state, re-prime MTP. ----
        let src_pos: Vec<i32> = path.iter().map(|&p| p as i32).collect();
        self.gpu.compact_kv(&mut self.pool, &mut self.state, phys, main_pos, &src_pos, kv_stride);
        // Adopt the accepted leaf's GDN checkpoint. The DFS scan ends at column n-1, so if leaf==n-1 the
        // slot already holds the right state; else restore from its checkpoint slot (ckpt+leaf).
        if leaf != n - 1 { self.gpu.copy_gdn_slot(&self.state, ckpt + leaf, phys); }
        // h_prev = hidden at the accepted leaf.
        self.gpu.copy_hidden_col(*self.mtp_h_prev[phys].device_ptr(), &vout, leaf);
        // Re-prime MTP over the accepted path: column 0 uses h_save + committed; column k uses
        // vout[path[k-1]] + tokens[path[k]].
        let n_rp = nacc + 1;
        let rp_hidden = self.pool.get_bf16(h * n_rp);
        let rp_ptr = *rp_hidden.device_ptr();
        let mut rp_toks: Vec<u32> = vec![committed_tok];
        self.gpu.copy_hidden_col(rp_ptr, self.mtp_h_save.as_ref().unwrap(), 0);
        for k in 1..=nacc {
            self.gpu.copy_hidden_col(rp_ptr + (k * h * 2) as u64, &vout, path[k - 1]);
            rp_toks.push(tokens[path[k]]);
        }
        self.gpu.mtp_reprime(&mut self.pool, &rp_hidden, &rp_toks, main_pos - 1, mtp_kc_ptr, mtp_vc_ptr, kv_stride);
        self.pool.release_bf16(rp_hidden, h * n_rp);
        self.pool.release_bf16(vout, h * n);

        // ---- Emit (same EOS/max_new discipline as the chain). ----
        let mut new_toks: Vec<u32> = Vec::with_capacity(nacc + 1);
        let mut hit_eos = false;
        for k in 0..nacc {
            if generated + new_toks.len() >= max_new { break; }
            new_toks.push(emitted[k]);
            if eos.contains(&emitted[k]) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; break; }
        }
        if !hit_eos && generated + new_toks.len() < max_new {
            new_toks.push(emitted[nacc]);   // bonus
            if eos.contains(&emitted[nacc]) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; }
        }
        let emit_count = new_toks.len();
        let finished = hit_eos || generated + emit_count >= max_new;

        // Cache what was FED: committed + accepted node tokens (emitted[0..nacc]).
        {
            let cache = &mut self.slot_cache[phys];
            cache.push(committed_tok);
            cache.extend_from_slice(&emitted[..nacc]);
        }
        {
            let lane = self.lanes[i].as_mut().unwrap();
            for &t in &new_toks {
                let _ = lane.tx.send(TokEvent::Tok(t));
                lane.history.push(t);
                if lane.history.len() > 256 { lane.history.drain(0..128); }
            }
            lane.generated += emit_count;
            if !finished {
                lane.last_tok = emitted[nacc];
                lane.pos = main_pos + nacc + 1;
                lane.mtp_pos = main_pos + nacc;
            } else {
                lane.last_tok = *new_toks.last().unwrap_or(&committed_tok);
                lane.pos = main_pos + emit_count;
            }
        }
        self.mtp_stat_steps += 1;
        self.mtp_stat_drafts += (n - 1) as u64;
        self.mtp_stat_accepted += nacc as u64;
        self.mtp_stat_emitted += emit_count as u64;
        self.mtp_stat_verify_fwds += 1;
        self.mtp.record_step((n - 1) as u64, nacc as u64, emit_count as u64);
        if self.mtp_stat_steps % 50 == 0 {
            let acc = if self.mtp_stat_drafts > 0 { self.mtp_stat_accepted as f64 / self.mtp_stat_drafts as f64 * 100.0 } else { 0.0 };
            let eff = self.mtp_stat_emitted as f64 / self.mtp_stat_verify_fwds as f64;
            eprintln!("[mtp/tree] steps={} accepted={:.1}% emitted={} tok/verify_fwd={:.3} (depth {} n {}) accept@k [{}]",
                      self.mtp_stat_steps, acc, self.mtp_stat_emitted, eff, depth, n, fmt_accept_by_depth(&self.mtp));
            self.dump_accept_curve();
        }
        finished
    }

    /// One MTP speculative-decoding step for lane `i` (greedy, penalty-free). Mirrors the validated
    /// `bench_mtp` loop body: draft (depth-1) → snapshot GDN → verify (depth) → accept longest
    /// prefix → rollback+reverify on partial reject → re-prime MTP over the accepted prefix with REAL
    /// verify hiddens. Emits the accepted drafts + bonus token, advancing the lane by nacc+1
    /// positions. Returns true if the lane finished (EOS or max_new reached).
    /// FOREST MTP step (LANES design Step 3c): draft each of `lanes`' chains, pack them into ONE forest
    /// verify (proven lossless by `--bench-lanes`), then per-lane accept / rollback / re-prime / emit.
    /// This is `mtp_lane_step` generalized to L lanes sharing one main-model forward — the concurrency
    /// throughput win. Greedy, no-penalty lanes only (v1). Returns (lane_index, finished) per lane.
    fn mtp_forest_step(&mut self, lanes: &[usize]) -> Vec<(usize, bool)> {
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let ck = self.gpu.cfg().conv_kernel;
        let mv = crate::gpu::MAX_VERIFY;
        let kv_stride = self.kv_stride;
        let snapshot = self.mtp_snapshot_slot;
        let p = lanes.len();
        // v1 allocator: uniform per-lane depth so Σ(1+d) = p*(1+d) ≤ MAX_VERIFY, capped by the policy
        // depth. The floor is the pack cap's derivation: p=5 ⇒ depth 2, p=6 ⇒ depth 1 (no drafts,
        // useless) — which is why `take` caps at 5 upstream.
        let depth = ((mv / p).saturating_sub(1)).clamp(1, self.mtp.depth());
        // BATCHED_MTP §5 risk 5: `clamp(1, d)` panics when the policy reports depth 0 (min > max).
        // A depth-0 policy has nothing to speculate with — serve each packed lane via the
        // single-lane path (which handles depth 0) instead of panicking.
        if self.mtp.depth() == 0 {
            eprintln!("[mtp] forest: policy depth 0 — per-lane MTP steps (no forest pack)");
            return lanes.iter().map(|&i| (i, self.mtp_lane_step(i))).collect();
        }
        // BATCHED_MTP P0 (the confound fix): the forest draft loop previously ignored
        // --ngram-draft and the coverage trace, so packed lanes drafted with a DIFFERENT
        // behaviour than the single-lane arms, and coverage studies silently under-sampled
        // exactly the packed lanes. Both hooks now mirror mtp_lane_step verbatim.
        let ngram = self.ngram_draft;
        let cov_on = self.cov_trace;
        let step_t0 = std::time::Instant::now();

        // Per-lane locals + draft chains (drafting is sequential; the shared MTP scratch is reused, and
        // each lane's head KV is its own mtp_kc/mtp_vc slot). Each lane's chain = [committed, drafts...].
        struct L { i: usize, phys: usize, committed: u32, main_pos: usize, mtp_pos: usize,
                   generated: usize, max_new: usize, drafts: Vec<u32>, start: usize, n: usize,
                   rep: f32, pres: f32, freq: f32, history: Vec<u32>, ignore_eos: bool, min_new: usize,
                   cov: Vec<Vec<(u32, f32)>> }
        let cur_ptr = *self.mtp_cur_hidden.as_ref().unwrap().device_ptr();
        let mut ls: Vec<L> = Vec::with_capacity(p);
        let mut global = 0usize;
        for &i in lanes {
            let (phys, committed, main_pos, mtp_pos, generated, max_new, rep, pres, freq, history, ignore_eos, min_new) = {
                let l = self.lanes[i].as_ref().unwrap();
                (l.phys, l.last_tok, l.pos, l.mtp_pos, l.generated, l.max_new,
                 l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.history.clone(),
                 l.ignore_eos, l.min_new)
            };
            let mtp_kc_ptr = *self.mtp_kc[phys].device_ptr();
            let mtp_vc_ptr = *self.mtp_vc[phys].device_ptr();
            self.gpu.copy_hidden_col(cur_ptr, &self.mtp_h_prev[phys], 0);
            // Prompt-lookup context for n-gram drafting (mtp_lane_step's twin): the lane's full
            // realized sequence from slot_cache plus this step's committed token.
            let mut work: Vec<u32> = Vec::new();
            if ngram > 0 {
                work.reserve(self.slot_cache[phys].len() + depth);
                work.extend_from_slice(&self.slot_cache[phys]);
                work.push(committed);
            }
            let mut cov: Vec<Vec<(u32, f32)>> = Vec::new();
            let mut drafts: Vec<u32> = Vec::with_capacity(depth - 1);
            let mut cur_tok = committed as i32;
            let mut dpos = mtp_pos;
            for _ in 0..depth - 1 {
                let m = self.gpu.mtp_draft_step(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap(),
                                                cur_tok, dpos, mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                self.gpu.copy_hidden_col(cur_ptr, &m, 0);
                self.pool.release_bf16(m, h);
                cur_tok = self.gpu.argmax_hidden(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap()) as i32;
                if cov_on {
                    cov.push(self.gpu.topk_hidden_kv(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap(), 8));
                }
                // PROMPT-LOOKUP OVERRIDE (mtp_lane_step's twin): if the last `ngram` tokens recur
                // earlier in this lane's context, propose the token that followed the most recent
                // earlier occurrence — lossless, the packed verify checks every draft.
                if ngram > 0 && work.len() >= ngram {
                    let tail_start = work.len() - ngram;
                    for j in (0..tail_start).rev() {
                        if work[j..j + ngram] == work[tail_start..] {
                            if j + ngram < work.len() { cur_tok = work[j + ngram] as i32; }
                            break;
                        }
                    }
                    work.push(cur_tok as u32);
                }
                drafts.push(cur_tok as u32);
                dpos += 1;
            }
            let n = 1 + drafts.len();
            ls.push(L { i, phys, committed, main_pos, mtp_pos, generated, max_new, drafts, start: global, n, ignore_eos, min_new,
                        rep, pres, freq, history, cov });
            global += n;
        }
        let ntot = global;

        // Build the FOREST topo + packed token stream (see bench_lanes / verify_forward_core_topo).
        let mut tokens: Vec<u32> = Vec::with_capacity(ntot);
        let mut parent = vec![-1i32; ntot];
        let mut slotv = vec![0i32; ntot];
        let mut rope = vec![0i32; ntot];
        let mut kv_pos = vec![0i32; ntot];
        let mut cps = vec![0i32; ntot];
        let mut path = vec![0u8; ntot * mv];
        let mut winsrc = vec![0i32; ntot * ck];
        for l in &ls {
            tokens.push(l.committed);
            tokens.extend_from_slice(&l.drafts);
            for r in 0..l.n {
                let c = l.start + r;
                parent[c] = if r == 0 { -1 } else { (c - 1) as i32 };
                slotv[c] = l.phys as i32;
                rope[c] = (l.main_pos + r) as i32;
                kv_pos[c] = (l.main_pos + r) as i32;
                cps[c] = l.main_pos as i32;
                for dd in 0..=r { path[c * mv + dd] = dd as u8; }
                for j in 0..ck {
                    let wd = r as i32 - (ck as i32 - 1) + j as i32;
                    winsrc[c * ck + j] = if wd < 0 { wd } else { l.start as i32 + wd };
                }
            }
        }
        let topo = crate::gpu::TreeTopo { rope, kv_pos, parent, path, winsrc,
                                          slot: Some(slotv), col_pos_start: Some(cps) };
        let pos_start_max = ls.iter().map(|l| l.main_pos).max().unwrap();

        // Per-lane verify penalty: each column carries ITS lane's rep/presence/freq penalty (buffers are
        // MAX_VERIFY-sized). None if no packed lane has a penalty.
        let ls_pen: Vec<(usize, usize, f32, f32, f32, Vec<u32>)> =
            ls.iter().map(|l| (l.start, l.n, l.rep, l.pres, l.freq, l.history.clone())).collect();
        let penalty = self.make_forest_penalty(&ls_pen);

        // ONE packed forest verify. ckpt writes column t's post-state to (snapshot + t); rollback reads
        // each lane's own accepted checkpoint.
        let (preds, vout) = self.gpu.verify_forward_topo(
            &mut self.pool, &tokens, &mut self.state, 0, kv_stride, pos_start_max, Some(snapshot), penalty, Some(&topo));

        let eos = self.eos.clone();
        let mut results: Vec<(usize, bool)> = Vec::with_capacity(p);
        // Per-lane (main_pos, nacc, emitted), index-aligned with ls, for the step-dump records.
        let mut lane_stats: Vec<(usize, usize, usize)> = Vec::with_capacity(p);
        for l in &ls {
            let draft_count = l.drafts.len();
            let mut nacc = 0usize;
            while nacc < draft_count && preds[l.start + nacc] == l.drafts[nacc] { nacc += 1; }
            let bonus = preds[l.start + nacc];
            // Rollback on partial reject: restore this lane's state as of its last accepted column.
            if nacc + 1 != l.n {
                self.gpu.copy_gdn_slot(&self.state, snapshot + l.start + nacc, l.phys);
            }
            // Re-prime this lane's MTP head over its accepted prefix (k=0 = pre-step h_prev, still intact;
            // k>=1 = the verify's real hidden vout[start+k-1]). Then advance h_prev to vout[start+nacc].
            let mtp_kc_ptr = *self.mtp_kc[l.phys].device_ptr();
            let mtp_vc_ptr = *self.mtp_vc[l.phys].device_ptr();
            let n_rp = nacc + 1;
            let rp_hidden = self.pool.get_bf16(h * n_rp);
            let rp_ptr = *rp_hidden.device_ptr();
            let mut rp_toks: Vec<u32> = Vec::with_capacity(n_rp);
            self.gpu.copy_hidden_col(rp_ptr, &self.mtp_h_prev[l.phys], 0);
            rp_toks.push(l.committed);
            // vout columns l.start..l.start+nacc-1 are CONTIGUOUS — one dtod for the whole prefix
            // (was nacc separate copy_hidden_col driver calls).
            self.gpu.copy_hidden_cols(rp_ptr + (h * 2) as u64, &vout, l.start, nacc);
            for k in 1..=nacc {
                rp_toks.push(l.drafts[k - 1]);
            }
            self.gpu.mtp_reprime(&mut self.pool, &rp_hidden, &rp_toks, l.main_pos - 1,
                                 mtp_kc_ptr, mtp_vc_ptr, kv_stride);
            self.pool.release_bf16(rp_hidden, h * n_rp);
            self.gpu.copy_hidden_col(*self.mtp_h_prev[l.phys].device_ptr(), &vout, l.start + nacc);

            // Emit accepted drafts + bonus (budget on drafts too — see mtp_lane_step).
            let mut new_toks: Vec<u32> = Vec::with_capacity(nacc + 1);
            let mut hit_eos = false;
            for &d in l.drafts.iter().take(nacc) {
                if l.generated + new_toks.len() >= l.max_new { break; }
                new_toks.push(d);
                if eos.contains(&d) && !l.ignore_eos && l.generated + new_toks.len() >= l.min_new { hit_eos = true; break; }
            }
            if !hit_eos && l.generated + new_toks.len() < l.max_new {
                new_toks.push(bonus);
                if eos.contains(&bonus) && !l.ignore_eos && l.generated + new_toks.len() >= l.min_new { hit_eos = true; }
            }
            let emit_count = new_toks.len();
            let finished = hit_eos || l.generated + emit_count >= l.max_new;
            {
                let cache = &mut self.slot_cache[l.phys];
                cache.push(l.committed);
                cache.extend_from_slice(&l.drafts[..nacc]);
            }
            {
                let lane = self.lanes[l.i].as_mut().unwrap();
                for &t in &new_toks {
                    let _ = lane.tx.send(TokEvent::Tok(t));
                    lane.history.push(t);
                    if lane.history.len() > 256 { lane.history.drain(0..128); }
                }
                lane.generated += emit_count;
                if !finished {
                    lane.last_tok = bonus;
                    lane.pos = l.main_pos + nacc + 1;
                    lane.mtp_pos = l.main_pos + nacc;
                } else {
                    lane.last_tok = *new_toks.last().unwrap_or(&l.committed);
                    lane.pos = l.main_pos + emit_count;
                }
            }
            // Telemetry parity with the chain path: the forest lane's step record reaches the
            // same spec-steps ring (acceptance curves must count forest-served lanes too).
            self.rec_step(SpecStepRec {
                greedy: true, pos: l.main_pos as u32, drafts: draft_count as u32, nacc: nacc as u32,
                emitted: emit_count as u32, round_ms: 0.0, verify_ms: 0.0,
                step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
            });
            self.mtp_stat_drafts += draft_count as u64;
            self.mtp_stat_accepted += nacc as u64;
            self.mtp_stat_emitted += emit_count as u64;
            lane_stats.push((l.main_pos, nacc, emit_count));
            results.push((l.i, finished));
        }
        self.pool.release_bf16(vout, h * ntot);
        self.mtp_stat_steps += p as u64;     // p lane-steps served by...
        self.mtp_stat_verify_fwds += 1;      // ...ONE main-model forward — the batching win

        // BATCHED_MTP §0.4 (the telemetry gap): the forest previously ended with the two counters
        // and NO [mtp] line, so a run where every step took the forest branch never showed its
        // accept@k / tok-per-forward — exactly where a collapse would hide. Same cadence and
        // counters as the chain path, labeled with the pack geometry.
        if self.mtp_stat_steps % 50 == 0 {
            let acc = if self.mtp_stat_drafts > 0 {
                self.mtp_stat_accepted as f64 / self.mtp_stat_drafts as f64 * 100.0
            } else { 0.0 };
            let eff = self.mtp_stat_emitted as f64 / self.mtp_stat_verify_fwds as f64;
            eprintln!("[mtp] steps={} drafts={} accepted={:.1}% emitted={} tok/verify_fwd={:.3} (forest p={} d={}) accept@k [{}]",
                      self.mtp_stat_steps, self.mtp_stat_drafts, acc,
                      self.mtp_stat_emitted, eff, p, depth, fmt_accept_by_depth(&self.mtp));
            crate::tel::publish_mtp(self.mtp_stat_steps, self.mtp_stat_drafts,
                                    self.mtp_stat_accepted, self.mtp_stat_emitted,
                                    self.mtp_stat_verify_fwds, depth, &self.mtp.hazard_counts());
            self.dump_accept_curve();
        }
        // Coverage parity (§0.4 divergence 2): with the step dump on, each packed lane's draft
        // top-k reaches the dump — a forest run no longer under-samples its own lanes. Target
        // coverage fields stay empty (the packed verify exposes no per-lane target top-k).
        if self.step_dump.is_some() {
            for (l, &(main_pos, nacc, emit_count)) in ls.iter().zip(lane_stats.iter()) {
                let bonus = preds.get(l.start + nacc).copied().unwrap_or(l.committed);
                let mut dk_ids = Vec::with_capacity(l.cov.len() * 8);
                let mut dk_logit = Vec::with_capacity(l.cov.len() * 8);
                for tk in &l.cov {
                    for (t, v) in tk { dk_ids.push(*t); dk_logit.push(*v); }
                }
                if let Some(d) = self.step_dump.as_mut() {
                    let rec = crate::dflash2::stepdump::MtpStepRec {
                        step: self.mtp_stat_steps, pos: main_pos, committed: l.committed, depth,
                        drafts: l.drafts.clone(), p_draft: Vec::new(), resid: Vec::new(),
                        bonus, nacc, emitted: emit_count,
                        tgt_ids: Vec::new(), tgt_logit: Vec::new(), tgt_p1: Vec::new(), tgt_margin: Vec::new(),
                        draft_topk_ids: dk_ids, draft_topk_logit: dk_logit,
                        tgt_top20: Vec::new(),
                    };
                    d.record_mtp(&rec);
                }
            }
        }
        results
    }

    /// Append one MTP step to the option-gated draft log (`--mtp-draft-log`), if open. JSONL, one object
    /// per line. `preds` is the verify's per-column argmax (greedy paths); pass `&[]` where the verify
    /// samples instead. Costs nothing when the log is closed. This is the reference the head-finetune
    /// B0 parity gate diffs the HF MTP module against, so it records exactly what was fed and predicted.
    fn log_draft_step(&mut self, lane: usize, pos: usize, committed: u32, drafts: &[u32], preds: &[u32], nacc: usize) {
        use std::io::Write;
        let Some(f) = self.mtp_draft_log.as_mut() else { return; };
        let arr = |xs: &[u32]| -> String {
            let mut s = String::from("[");
            for (k, x) in xs.iter().enumerate() { if k > 0 { s.push(','); } s.push_str(&x.to_string()); }
            s.push(']'); s
        };
        let _ = writeln!(f,
            "{{\"step\":{},\"lane\":{},\"pos\":{},\"committed\":{},\"drafts\":{},\"preds\":{},\"nacc\":{}}}",
            self.mtp_stat_steps, lane, pos, committed, arr(drafts), arr(preds), nacc);
    }

    /// Overwrite the accept-by-depth curve file (`--mtp-curve-file`), if set, with the cumulative
    /// per-position conditional acceptance — the runbook §0 baseline curve. Called on the periodic
    /// stats boundary so a running server keeps a fresh snapshot on disk.
    fn dump_accept_curve(&self) {
        let Some(path) = self.mtp_curve_path.as_ref() else { return; };
        let hz = self.mtp.hazard_counts();
        let mut s = String::from("{\"depth\":");
        s.push_str(&self.mtp.depth().to_string());
        s.push_str(",\"steps\":");
        s.push_str(&self.mtp_stat_steps.to_string());
        s.push_str(",\"accept_by_depth\":[");
        // accept_by_depth[i] = {pos:i+1, accepted, offered, rate, cond_chain}. cond_chain = product of
        // rates up to and including pos i+1 = P(a draft chain reaches depth i+1) — the yield driver.
        let mut chain = 1.0f64;
        for (i, &(a, n)) in hz.iter().enumerate() {
            let rate = if n > 0 { a as f64 / n as f64 } else { 0.0 };
            chain *= rate;
            if i > 0 { s.push(','); }
            s.push_str(&format!(
                "{{\"pos\":{},\"accepted\":{},\"offered\":{},\"rate\":{:.4},\"cond_chain\":{:.4}}}",
                i + 1, a, n, rate, chain));
        }
        s.push_str("]}");
        let _ = std::fs::write(path, s);
    }

    fn mtp_lane_step(&mut self, i: usize) -> bool {
        crate::tel::note_step();   // A.2: per-step interval for the status route (lock-free)
        // Fork-then-chain tree path (opt-in): rescues the chain-killing first-token miss.
        if self.tree_draft && self.mtp.depth() >= 3 {
            return self.mtp_tree_step(i);
        }
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let depth = self.mtp.depth();
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let snapshot = self.mtp_snapshot_slot;
        let kv_stride = self.kv_stride;
        let mtp_kc_ptr = *self.mtp_kc[phys].device_ptr();
        let mtp_vc_ptr = *self.mtp_vc[phys].device_ptr();
        let h_save_ptr = *self.mtp_h_save.as_ref().unwrap().device_ptr();
        let cur_ptr = *self.mtp_cur_hidden.as_ref().unwrap().device_ptr();

        // Snapshot lane state into locals (avoids holding &mut self.lanes across GPU calls).
        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
        let mtp_pos = self.lanes[i].as_ref().unwrap().mtp_pos;
        let generated = self.lanes[i].as_ref().unwrap().generated;
        let max_new = self.lanes[i].as_ref().unwrap().max_new;
        let eos = self.eos.clone();
        // Phase-8: the stop rule is a property of the REQUEST, not the serving path —
        // every emit loop must honor ignore_eos/min_new exactly like the batched decode path
        // (batch.rs' `eos_hit`), or spec-on and spec-off streams stop at different tokens.
        let (ignore_eos, min_new) = { let l = self.lanes[i].as_ref().unwrap(); (l.ignore_eos, l.min_new) };
        // Penalty config + history (so the MTP verify keeps the lane's rep/presence/freq penalty).
        let (rep_pen, presence_pen, freq_pen, has_penalty) = {
            let l = self.lanes[i].as_ref().unwrap();
            (l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.has_penalty())
        };
        let history: Vec<u32> = self.lanes[i].as_ref().unwrap().history.clone();
        // W2: the lane's compiled schema + its current FSM state (None/0 = unconstrained).
        let lane_schema = self.lanes[i].as_ref().unwrap().schema.clone();
        let lane_schema_state = self.lanes[i].as_ref().unwrap().schema_state;

        // h_save = h_prev (hidden at main_pos-1); saved for the post-accept re-prime step k=0.
        self.gpu.copy_hidden_col(h_save_ptr, &self.mtp_h_prev[phys], 0);

        // Prompt-lookup context for n-gram drafting: the full realized sequence for this lane, which
        // slot_cache holds (prompt + every committed token), plus committed_tok (this step's first
        // token to verify, which slot_cache does not yet contain). Cheap host-side clone of u32s,
        // trivial next to a GPU forward. `ngram == 0` disables it.
        let ngram = self.ngram_draft;
        let mut work: Vec<u32> = Vec::new();
        if ngram > 0 {
            work.reserve(self.slot_cache[phys].len() + depth);
            work.extend_from_slice(&self.slot_cache[phys]);
            work.push(committed_tok);
        }

        // ---- Draft chain (depth-1 drafts). cur_hidden starts at h_prev; chains via MTP outputs. ----
        let step_t0 = std::time::Instant::now();
        // PLAN/25 Phase 0: coverage capture (dump-only) — the drafter's top-8 per position is
        // read back AFTER the argmax (the draft itself still comes from `argmax_hidden`; the
        // readback never picks a draft). Zero effect when the step dump is off.
        let cov_on = self.cov_trace;
        let mut cov_draft_topk: Vec<Vec<(u32, f32)>> = Vec::new();
        self.gpu.copy_hidden_col(cur_ptr, &self.mtp_h_prev[phys], 0);
        let mut drafts: Vec<u32> = Vec::with_capacity(depth - 1);
        let mut cur_tok = committed_tok as i32;
        let mut dpos = mtp_pos;
        for _ in 0..depth - 1 {
            let m = self.gpu.mtp_draft_step(
                &mut self.pool, self.mtp_cur_hidden.as_ref().unwrap(), cur_tok, dpos,
                mtp_kc_ptr, mtp_vc_ptr, kv_stride);
            self.gpu.copy_hidden_col(cur_ptr, &m, 0);
            self.pool.release_bf16(m, h);
            cur_tok = self.gpu.argmax_hidden(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap()) as i32;
            if cov_on {
                cov_draft_topk.push(
                    self.gpu.topk_hidden_kv(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap(), 8));
            }
            // PROMPT-LOOKUP OVERRIDE (see mtp_lane_step's twin logic in gpu.rs::bench_accept). If the
            // last `ngram` tokens recur earlier in this lane's context, propose the token that followed
            // the most recent earlier occurrence — a free, exact copy that the 1-layer head cannot do.
            // Lossless by construction: the verify checks every draft; a wrong override is just rejected.
            if ngram > 0 && work.len() >= ngram {
                let tail_start = work.len() - ngram;
                for j in (0..tail_start).rev() {
                    if work[j..j + ngram] == work[tail_start..] {
                        if j + ngram < work.len() { cur_tok = work[j + ngram] as i32; }
                        break;
                    }
                }
                work.push(cur_tok as u32);
            }
            drafts.push(cur_tok as u32);
            dpos += 1;
        }

        // ---- Verify [committed_tok, drafts...] on the main model at positions main_pos.. ----
        let round_ms = step_t0.elapsed().as_secs_f32() * 1e3;
        let mut verify_input = vec![committed_tok];
        verify_input.extend(drafts.iter().copied());
        // Build the penalty for the verify: all `depth` positions share the lane's committed-history
        // penalty (the same one the normal decode path applies). This keeps greedy MTP lanes free of
        // repetition without slowing them. htod on the NULL stream, then sync before the compute-side
        // verify reads it.
        let penalty = self.make_penalty_drafts(&history, &drafts, rep_pen, presence_pen, freq_pen, has_penalty);
        // Ping-pong GDN: the verify snapshots S1 (post committed-token state) into the snapshot slot
        // via the kernel checkpoint, so a rejected draft restores S1 with a dtod copy — no reverify.
        let verify_t0 = std::time::Instant::now();
        // PLAN/25 Phase 0 (cov_on): keep the verify logits and derive preds + the per-column
        // target top-4/p1/top1−top2 margin host-side (the bench_accept readback). The eager core
        // is the exact sequence the E13 graph captures, so the values — and therefore preds —
        // are identical to the non-dump path. Dump-only.
        // Penalty-aware stand-down: a penalized request's preds must come from the penalized
        // verify, so the coverage capture quietly skips that step (empty tgt fields in its
        // record). `has_penalty` is a per-request property, identical on both ranks — SPMD-safe.
        // ---- W2 (Phase 13): JSON-schema mask chain over the verify columns ----
        // The mask is the decode path's per-state vocabulary mask, extended along the draft chain:
        // position 0 carries the lane's CURRENT FSM state, position j+1 the state reached by the
        // drafts the earlier positions would have to accept. `schema_valid` is the length of that
        // valid draft prefix — the host clamps the accepted prefix to it, so a draft that breaks
        // the chain can never be committed, and the bonus column always comes from a masked,
        // correct state.
        let schema_valid: Option<usize> = match lane_schema.clone() {
            Some(sm) => {
                let words = crate::gpu::mask_words_for(self.gpu.cfg().vocab_size);
                let npos = verify_input.len();
                let mut masks: Vec<Option<std::sync::Arc<Vec<u32>>>> = Vec::with_capacity(npos);
                let mut st = lane_schema_state;
                let mut valid = 0usize;
                let dead = std::sync::Arc::new(vec![0u32; words]);
                for j in 0..npos {
                    if valid == j {
                        masks.push(Some(sm.allowed(st)));
                        if j < drafts.len() {
                            if let Some(ns) = sm.step(st, drafts[j]) { st = ns; valid = j + 1; }
                        }
                    } else {
                        masks.push(Some(dead.clone()));   // past the break: never emitted (nacc clamped)
                    }
                }
                self.gpu.set_verify_mask(&masks);
                Some(valid)
            }
            None => None,
        };
        let (preds, vout, cov_tgt) = if cov_on {
            let (logits, vout) = self.gpu.verify_forward_keep_logits(
                &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos,
                Some(snapshot));
            let n = verify_input.len();
            let vocab = self.gpu.cfg().vocab_size;
            let lg_f32: Vec<f32> = match logits {
                crate::gpu::VerifyLogits::B16(lg) => {
                    let v: Vec<half::bf16> = self.gpu.dev().dtoh_sync_copy(&lg).unwrap();
                    self.pool.release_bf16(lg, vocab * n);
                    v.iter().map(|x| x.to_f32()).collect()
                }
                crate::gpu::VerifyLogits::F32(lg) => {
                    let v: Vec<f32> = self.gpu.dev().dtoh_sync_copy(&lg).unwrap();
                    self.pool.release(lg, vocab * n);
                    v
                }
            };
            let mut preds = Vec::with_capacity(n);
            let (mut tids, mut tlog) = (Vec::with_capacity(n * 4), Vec::with_capacity(n * 4));
            let (mut tp1, mut tmar) = (Vec::with_capacity(n), Vec::with_capacity(n));
            for c in 0..n {
                let (ids, vals, p1, margin) = crate::dflash2::stepdump::tgt_topk_host(
                    &lg_f32[c * vocab..(c + 1) * vocab], 4);
                preds.push(ids[0]);
                tids.extend_from_slice(&ids); tlog.extend_from_slice(&vals);
                tp1.push(p1); tmar.push(margin);
            }
            (preds, vout, Some((tids, tlog, tp1, tmar)))
        } else {
            let (preds, vout) = self.gpu.verify_forward(
                &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos,
                Some(snapshot), penalty);
            (preds, vout, None)
        };
        let verify_ms = verify_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Accept longest prefix (greedy: drafts[i] accepted iff preds[i]==drafts[i]). ----
        let mut nacc = 0usize;
        while nacc < drafts.len() && preds[nacc] == drafts[nacc] { nacc += 1; }
        if let Some(v) = schema_valid {
            // A draft beyond the schema-valid prefix is never committed (its verify column had no
            // legal continuation), and the mask is disarmed for the next (unmasked) caller.
            nacc = nacc.min(v);
            self.gpu.clear_verify_mask();
        }
        let bonus = preds[nacc];

        // ---- GDN rollback on partial reject: restore S1 (the checkpoint — no second forward). ----
        // vout column nacc is the hidden at the last accepted position, valid in both cases.
        if nacc + 1 != depth {
            // Restore the state as of the LAST ACCEPTED column, not column 0. Checkpoint slots are
            // contiguous: slot (snapshot + t) holds the post-state of verify column t.
            self.gpu.copy_gdn_slot(&self.state, snapshot + nacc, phys);
        }

        // h_prev = hidden at the last accepted position (vout column nacc).
        self.gpu.copy_hidden_col(*self.mtp_h_prev[phys].device_ptr(), &vout, nacc);

        // ---- Re-prime MTP over the accepted prefix with REAL hiddens (vout), in ONE forward. ----
        // Column k=0 uses h_save (the pre-verify hidden); k>=1 uses vout column k-1. Batching all
        // nacc+1 columns into a single MTP-layer forward reads the layer's weights once instead of
        // once per accepted position — the causal-append attention makes column k see the KV that
        // columns < k just wrote, exactly as in the multi-token verify.
        let n_rp = nacc + 1;
        let rp_hidden = self.pool.get_bf16(h * n_rp);
        let rp_ptr = *rp_hidden.device_ptr();
        let mut rp_toks: Vec<u32> = Vec::with_capacity(n_rp);
        self.gpu.copy_hidden_col(rp_ptr, self.mtp_h_save.as_ref().unwrap(), 0);
        rp_toks.push(committed_tok);
        // vout columns 0..nacc-1 are CONTIGUOUS — one dtod for the whole accepted prefix.
        self.gpu.copy_hidden_cols(rp_ptr + (h * 2) as u64, &vout, 0, nacc);
        for k in 1..=nacc {
            rp_toks.push(drafts[k - 1]);
        }
        self.gpu.mtp_reprime(&mut self.pool, &rp_hidden, &rp_toks, main_pos - 1,
                             mtp_kc_ptr, mtp_vc_ptr, kv_stride);
        self.pool.release_bf16(rp_hidden, h * n_rp);
        self.pool.release_bf16(vout, h * depth);

        // ---- Emit accepted drafts + bonus, honoring EOS and max_new. ----
        //
        // The BUDGET CHECK BELONGS ON THE DRAFTS TOO. It used to guard only the bonus, so a step that
        // accepted k drafts emitted all k regardless: a request for exactly max_tokens got back up to
        // max_tokens + depth - 1. Harmless-looking, and it is why a greedy MTP response and a greedy
        // plain response to the same prompt differed -- not in the tokens, which were identical, but
        // in how many of them came back. That cost real time to chase, because "MTP output != plain
        // output" reads as a losslessness failure when it is really an off-by-one in the stop rule.
        let mut new_toks: Vec<u32> = Vec::with_capacity(nacc + 1);
        let mut hit_eos = false;
        for &d in drafts.iter().take(nacc) {
            if generated + new_toks.len() >= max_new { break; }
            new_toks.push(d);
            if eos.contains(&d) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; break; }
        }
        // Bonus (the greedy next token after the last accepted position) — always progress.
        if !hit_eos && generated + new_toks.len() < max_new {
            new_toks.push(bonus);
            if eos.contains(&bonus) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; }
        }

        let emit_count = new_toks.len();
        // W2: advance the JSON-schema FSM over the tokens ACTUALLY emitted (accepted drafts +
        // bonus). A completed document ends the turn - generation must not run past the value.
        let schema_done = self.schema_advance(i, &new_toks);
        let finished = hit_eos || generated + emit_count >= max_new || schema_done;

        // The verify FED [committed_tok] ++ drafts[..nacc] through the model, and the GDN state was
        // rolled back to exactly the last accepted column — so that, and only that, is what the slot's
        // state has consumed. The bonus token was PREDICTED, not fed. Cache what was consumed.
        // (If EOS truncated the emit, the cache still holds every fed token: it is then a longer
        // sequence than the client saw, so the next turn simply matches a shorter prefix. Correct,
        // just less reuse — which is the right way for this to fail.)
        {
            let cache = &mut self.slot_cache[phys];
            cache.push(committed_tok);
            cache.extend_from_slice(&drafts[..nacc]);
        }

        // Apply lane state. last_tok/pos/mtp_pos only matter if the lane continues, but keep them
        // consistent regardless. On full emit (not finished), advance the MTP cursor as in bench_mtp.
        {
            let lane = self.lanes[i].as_mut().unwrap();
            for &t in &new_toks {
                let _ = lane.tx.send(TokEvent::Tok(t));
                lane.history.push(t);
                if lane.history.len() > 256 { lane.history.drain(0..128); }
            }
            lane.generated += emit_count;
            // If the lane continues, it emitted all nacc drafts + bonus (emit_count == nacc+1).
            if !finished {
                lane.last_tok = bonus;
                lane.pos = main_pos + nacc + 1;
                lane.mtp_pos = main_pos + nacc;
            } else {
                lane.last_tok = *new_toks.last().unwrap_or(&committed_tok);
                lane.pos = main_pos + emit_count;
            }
        }

        // ---- S-B16 lossless auditor (--mtp-lossless-audit=1, TP=1): lockstep shadow decode ----
        // bench_mtp_steps' slot-1 ground truth, moved into the serving path: one plain decode
        // forward per emitted token, ALWAYS advancing on the ground-truth token (never on the MTP
        // stream), so the first divergence between the serving lane and plain decode is pinned to
        // an exact (request, emitted-index, position). A lane that later takes plain (non-MTP)
        // steps leaves its cursor behind — the audit for that request simply ends there.
        if self.audit_on && emit_count > 0 && self.audit_cur[i].is_some() {
            let need_prime = self.audit_cur[i].as_ref().map_or(false, |c| !c.primed);
            if need_prime {
                let (gt_last, gt_pos) = {
                    let c = self.audit_cur[i].as_ref().unwrap();
                    (c.gt_last, c.gt_pos)
                };
                self.gpu.zero_slot_state(&mut self.state, self.audit_slot, kv_stride);
                // The shadow slot has no prefill: zero its KV extent too, or the prime decode
                // attends over [0, pos) rows nothing wrote (garbage softmax → false divergence).
                self.gpu.zero_slot_kv(&mut self.state, self.audit_slot, kv_stride);
                let pred = self.audit_shadow_decode(gt_last, gt_pos, kv_stride);
                {
                    let c = self.audit_cur[i].as_mut().unwrap();
                    c.pred = pred;
                    c.primed = true;
                }
            }
            for &e in &new_toks {
                let (pred, gpos, tag, idx, already) = {
                    let c = self.audit_cur[i].as_ref().unwrap();
                    (c.pred, c.gt_pos, c.tag, c.idx, c.diverged)
                };
                if e != pred && !already {
                    eprintln!("[lossless-audit] req={} FIRST DIVERGENCE: emitted#{} pos={} mtp={} seq={} (depth {} nacc {} step {})",
                              tag, idx, gpos + 1, e, pred, depth, nacc, self.mtp_stat_steps);
                    self.audit_cur[i].as_mut().unwrap().diverged = true;
                }
                // Advance the shadow on the GROUND-TRUTH token (pred), never on the MTP stream,
                // mirroring bench_mtp_steps' slot-1 lockstep exactly.
                let new_pred = self.audit_shadow_decode(pred, gpos + 1, kv_stride);
                {
                    let c = self.audit_cur[i].as_mut().unwrap();
                    c.gt_last = pred;
                    c.gt_pos = gpos + 1;
                    c.pred = new_pred;
                    c.idx += 1;
                }
            }
        }
        // (committed_tok/main_pos/mtp_pos are all read above; no further bookkeeping needed.)

        // ---- Telemetry: accumulate per-step MTP stats and log a summary every 50 lane-steps. ----
        // Ping-pong GDN: every step is exactly ONE main-model verify forward (no reverify on reject).
        self.mtp_stat_steps += 1;
        self.mtp_stat_drafts += drafts.len() as u64;
        self.mtp_stat_accepted += nacc as u64;
        self.mtp_stat_emitted += emit_count as u64;
        self.mtp_stat_verify_fwds += 1;
        // Feed the auto-policy: it needs tokens-per-step to decide whether MTP is still paying.
        self.mtp.record_step(drafts.len() as u64, nacc as u64, emit_count as u64);
        self.log_draft_step(i, main_pos, committed_tok, &drafts, &preds, nacc);
        self.rec_step(SpecStepRec {
            greedy: true, pos: main_pos as u32, drafts: drafts.len() as u32, nacc: nacc as u32,
            emitted: emit_count as u32, round_ms, verify_ms,
            step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
        });
        // ---- S5F3 MTP control dump (dump-only; the p-computation control). ----
        if let Some(d) = self.step_dump.as_mut() {
            let (mut tids, mut tlog) = (Vec::new(), Vec::new());
            let (mut tp1, mut tmar) = (Vec::new(), Vec::new());
            if let Some((a, b, c, e)) = cov_tgt.as_ref() {
                tids = a.clone(); tlog = b.clone(); tp1 = c.clone(); tmar = e.clone();
            }
            let mut dk_ids = Vec::with_capacity(cov_draft_topk.len() * 8);
            let mut dk_logit = Vec::with_capacity(cov_draft_topk.len() * 8);
            for tk in &cov_draft_topk {
                for (t, v) in tk { dk_ids.push(*t); dk_logit.push(*v); }
            }
            let rec = crate::dflash2::stepdump::MtpStepRec {
                step: self.mtp_stat_steps, pos: main_pos, committed: committed_tok, depth,
                drafts: drafts.clone(), p_draft: Vec::new(),
                resid: preds[..depth.saturating_sub(1).min(preds.len())].to_vec(),
                bonus, nacc, emitted: emit_count,
                tgt_ids: tids, tgt_logit: tlog, tgt_p1: tp1, tgt_margin: tmar,
                draft_topk_ids: dk_ids, draft_topk_logit: dk_logit,
                tgt_top20: Vec::new(),
            };
            d.record_mtp(&rec);
        }
        if self.mtp_stat_steps % 50 == 0 {
            let acc = if self.mtp_stat_drafts > 0 {
                self.mtp_stat_accepted as f64 / self.mtp_stat_drafts as f64 * 100.0
            } else { 0.0 };
            // Effective speedup ceiling = emitted tokens / verify forwards (how many output tokens
            // we get per main-model forward — 1.0 means no MTP benefit).
            let eff = self.mtp_stat_emitted as f64 / self.mtp_stat_verify_fwds as f64;
            eprintln!("[mtp] steps={} drafts={} accepted={:.1}% emitted={} tok/verify_fwd={:.3} (depth {}) accept@k [{}]",
                      self.mtp_stat_steps, self.mtp_stat_drafts, acc,
                      self.mtp_stat_emitted, eff, depth, fmt_accept_by_depth(&self.mtp));
            // A.2: same numbers to the status route (published where the log line prints).
            crate::tel::publish_mtp(self.mtp_stat_steps, self.mtp_stat_drafts,
                                    self.mtp_stat_accepted, self.mtp_stat_emitted,
                                    self.mtp_stat_verify_fwds, depth, &self.mtp.hazard_counts());
            self.dump_accept_curve();
        }
        finished
    }

    /// S-B16 auditor: one plain decode forward on the audit slot (batch 1), mirroring the SERVE
    /// path's decode flavor — max_pc PINNED to kv_stride, exactly like the captured decode graph
    /// (capture pins max_pc to the run maximum; an eager pos+1 bound selects a different split-K
    /// reduction and flips knife-edge positions — run5's two "divergences" were this artifact:
    /// the seq arm's own text carries the LANE's token at both spots).
    fn audit_shadow_decode(&mut self, tok: u32, pos: usize, kv_stride: usize) -> u32 {
        {
            let ab = self.audit_bufs.as_mut().unwrap();
            self.gpu.dev().htod_sync_copy_into(&[tok as i32, 0], &mut ab.tokens_dev).unwrap();
            self.gpu.dev().htod_sync_copy_into(&[pos as i32, 0], &mut ab.pos_dev).unwrap();
        }
        self.gpu.dev().synchronize().unwrap();
        let ab = self.audit_bufs.as_mut().unwrap();
        let next = self.gpu.forward_decode(&mut self.pool, ab, &mut self.state, kv_stride, kv_stride, 1);
        next[0]
    }

    /// Stochastic MTP step for a sampling lane (temperature > 0). Mirrors mtp_lane_step but:
    /// 1. Drafts GREEDILY via argmax_hidden (point-mass proposal q=1 — the rejection step corrects
    ///    the distribution; on hy_v3 the draft argmax is fp32-exact, gpu.rs argmax_hidden)
    /// 2. Verifies via verify_forward_sample (returns p_of_draft + resid_tok + bonus_tok; fp32
    ///    logits end-to-end on hy_v3)
    /// 3. Accepts via speculative rejection sampling: accept draft with prob min(1, p(x)/q(x)),
    ///    else emit a token from the residual (p \ {draft}, renormalized).
    /// 4. Re-primes MTP with REAL verify hiddens for ACCEPTED positions only.
    /// Emits the accepted drafts + replacement/bonus token, advancing the lane.
    /// Returns true if the lane finished (EOS or max_new reached).
    fn mtp_lane_step_sample(&mut self, i: usize) -> bool {
        crate::tel::note_step();   // A.2: per-step interval for the status route (lock-free)
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let depth = self.mtp.depth();
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let snapshot = self.mtp_snapshot_slot;
        let kv_stride = self.kv_stride;
        let mtp_kc_ptr = *self.mtp_kc[phys].device_ptr();
        let mtp_vc_ptr = *self.mtp_vc[phys].device_ptr();
        let h_save_ptr = *self.mtp_h_save.as_ref().unwrap().device_ptr();
        let cur_ptr = *self.mtp_cur_hidden.as_ref().unwrap().device_ptr();

        // Snapshot lane state into locals.
        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
        let mtp_pos = self.lanes[i].as_ref().unwrap().mtp_pos;
        let generated = self.lanes[i].as_ref().unwrap().generated;
        let max_new = self.lanes[i].as_ref().unwrap().max_new;
        let eos = self.eos.clone();
        // Phase-8: the stop rule is a property of the REQUEST, not the serving path —
        // every emit loop must honor ignore_eos/min_new exactly like the batched decode path
        // (batch.rs' `eos_hit`), or spec-on and spec-off streams stop at different tokens.
        let (ignore_eos, min_new) = { let l = self.lanes[i].as_ref().unwrap(); (l.ignore_eos, l.min_new) };
        let temperature = self.lanes[i].as_ref().unwrap().temperature;
        let top_k = self.lanes[i].as_ref().unwrap().top_k;
        let top_p = self.lanes[i].as_ref().unwrap().top_p;
        let (rep_pen, presence_pen, freq_pen, has_penalty) = {
            let l = self.lanes[i].as_ref().unwrap();
            (l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.has_penalty())
        };
        let history: Vec<u32> = self.lanes[i].as_ref().unwrap().history.clone();
        // All RNG for this step derives from one key (device column seeds + host accept draws,
        // domain-separated); the lane key advances exactly once per step.
        let step_key = self.lanes[i].as_ref().unwrap().seed;

        // h_save = h_prev (hidden at main_pos-1); saved for the post-accept re-prime step k=0.
        self.gpu.copy_hidden_col(h_save_ptr, &self.mtp_h_prev[phys], 0);

        // ---- Draft chain (depth-1 drafts) via greedy argmax from the MTP head. ----
        // Greedy drafting puts the draft token in the target model's high-probability region,
        // dramatically improving acceptance vs. sampling from the weak 1-layer MTP head.
        // Rejection sampling still corrects the output distribution to match non-MTP sampling.
        let step_t0 = std::time::Instant::now();
        self.gpu.copy_hidden_col(cur_ptr, &self.mtp_h_prev[phys], 0);
        let mut drafts: Vec<u32> = Vec::with_capacity(depth - 1);
        let mut qprobs: Vec<f32> = Vec::with_capacity(depth - 1);
        let mut cur_tok = committed_tok as i32;
        let mut dpos = mtp_pos;
        // PLAN/25 Phase 0: coverage capture (dump-only) — drafter top-8 readback per position,
        // after the argmax (the readback never picks the draft).
        let cov_on = self.cov_trace;
        let mut cov_draft_topk: Vec<Vec<(u32, f32)>> = Vec::new();
        for _ in 0..depth - 1 {
            let m = self.gpu.mtp_draft_step(
                &mut self.pool, self.mtp_cur_hidden.as_ref().unwrap(), cur_tok, dpos,
                mtp_kc_ptr, mtp_vc_ptr, kv_stride);
            self.gpu.copy_hidden_col(cur_ptr, &m, 0);
            self.pool.release_bf16(m, h);
            let tok = self.gpu.argmax_hidden(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap());
            cur_tok = tok as i32;
            if cov_on {
                cov_draft_topk.push(
                    self.gpu.topk_hidden_kv(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap(), 8));
            }
            drafts.push(tok);
            qprobs.push(1.0); // greedy draft = point mass
            dpos += 1;
        }
        let round_ms = step_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Build verify penalty (same as greedy). ----
        let verify_penalty = self.make_penalty_drafts(&history, &drafts, rep_pen, presence_pen, freq_pen, has_penalty);

        // ---- Build verify input + seeds for spec_verify_b. ----
        let mut verify_input = vec![committed_tok];
        verify_input.extend(drafts.iter().copied());
        // Per-column device seeds, domain-separated from the host accept draws below.
        let verify_seeds: Vec<u32> =
            (0..depth).map(|j| rng_u32(step_key, RNG_DOM_VERIFY, j)).collect();

        // ---- Verify with stochastic output. ----
        let verify_t0 = std::time::Instant::now();
        // PLAN/25 Phase 0: the dump-only top-20 table rides out on `cov_t20` when the step
        // dump is armed (verify_forward_sample's existing t20_out hook).
        let mut cov_t20: Vec<u64> = Vec::new();
        let (vsample, vout) = self.gpu.verify_forward_sample(
            &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos,
            Some(snapshot), verify_penalty,
            &drafts, &qprobs, temperature, top_k, top_p, &verify_seeds,
            if cov_on { Some(&mut cov_t20) } else { None });
        let verify_ms = verify_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Speculative rejection sampling accept loop (Leviathan et al. 2023). ----
        // For each draft position j: accept with prob min(1, p_j(x_j) / q_j(x_j)).
        // On reject: emit resid_tok[j] and stop. If all accepted: emit bonus_tok.
        let mut nacc = 0usize;
        let mut emitted: Vec<u32> = Vec::with_capacity(depth);
        let mut rejected = false;
        let eps = 1e-12f32;
        for j in 0..drafts.len() {
            let ratio = if qprobs[j] < eps { 1.0 } else { (vsample.p_of_draft[j] / qprobs[j]).min(1.0) };
            let ru = rng_uniform(step_key, RNG_DOM_ACCEPT, j);
            if ru < ratio {
                emitted.push(drafts[j]);
                nacc += 1;
            } else {
                emitted.push(vsample.resid_tok[j]);
                rejected = true;
                break;
            }
        }
        if !rejected {
            emitted.push(vsample.bonus_tok);
        }

        // ---- GDN rollback on partial reject. ----
        if nacc + 1 != depth {
            // Restore the state as of the LAST ACCEPTED column, not column 0. Checkpoint slots are
            // contiguous: slot (snapshot + t) holds the post-state of verify column t.
            self.gpu.copy_gdn_slot(&self.state, snapshot + nacc, phys);
        }

        // ---- h_prev = hidden at the last accepted position (vout column nacc). ----
        self.gpu.copy_hidden_col(*self.mtp_h_prev[phys].device_ptr(), &vout, nacc);

        // ---- Re-prime MTP with the REAL verify hiddens for the accepted positions, in ONE forward.
        // Only ACCEPTED positions are re-primed: `drafts[k-1]` for k<=nacc are by definition the
        // accepted drafts, so they are exactly the tokens that were emitted there. On a rejection the
        // replacement token at index nacc is NOT re-primed here — it becomes the next step's
        // committed token and is primed then.
        let n_rp = nacc + 1;
        let rp_hidden = self.pool.get_bf16(h * n_rp);
        let rp_ptr = *rp_hidden.device_ptr();
        let mut rp_toks: Vec<u32> = Vec::with_capacity(n_rp);
        self.gpu.copy_hidden_col(rp_ptr, self.mtp_h_save.as_ref().unwrap(), 0);
        rp_toks.push(committed_tok);
        // vout columns 0..nacc-1 are CONTIGUOUS — one dtod for the whole accepted prefix.
        self.gpu.copy_hidden_cols(rp_ptr + (h * 2) as u64, &vout, 0, nacc);
        for k in 1..=nacc {
            rp_toks.push(drafts[k - 1]);
        }
        self.gpu.mtp_reprime(&mut self.pool, &rp_hidden, &rp_toks, main_pos - 1,
                             mtp_kc_ptr, mtp_vc_ptr, kv_stride);
        self.pool.release_bf16(rp_hidden, h * n_rp);
        self.pool.release_bf16(vout, h * depth);

        // ---- Emit accepted tokens + replacement/bonus, honoring EOS and max_new. ----
        let mut hit_eos = false;
        let mut to_emit: Vec<u32> = Vec::with_capacity(emitted.len());
        for &t in &emitted {
            to_emit.push(t);
            if eos.contains(&t) && !ignore_eos && generated + to_emit.len() >= min_new { hit_eos = true; break; }
            if generated + to_emit.len() >= max_new { break; }
        }
        let emit_count = to_emit.len();
        let finished = hit_eos || generated + emit_count >= max_new;

        // Same as the greedy path: the verify FED [committed_tok] ++ drafts[..nacc], and the GDN state
        // was rolled back to the last accepted column. On a rejection the REPLACEMENT token emitted at
        // index nacc was never fed — it becomes the next step's committed token and is fed then. So the
        // slot's state has consumed exactly this, and nothing more.
        {
            let cache = &mut self.slot_cache[phys];
            cache.push(committed_tok);
            cache.extend_from_slice(&drafts[..nacc]);
        }

        // Apply lane state.
        {
            let lane = self.lanes[i].as_mut().unwrap();
            for &t in &to_emit {
                let _ = lane.tx.send(TokEvent::Tok(t));
                lane.history.push(t);
                if lane.history.len() > 256 { lane.history.drain(0..128); }
            }
            lane.generated += emit_count;
            lane.seed = splitmix64(step_key);
            if !finished {
                lane.last_tok = *to_emit.last().unwrap_or(&committed_tok);
                lane.pos = main_pos + nacc + 1;
                lane.mtp_pos = main_pos + nacc;
            } else {
                lane.last_tok = *to_emit.last().unwrap_or(&committed_tok);
                lane.pos = main_pos + emit_count;
            }
        }

        // ---- Telemetry. ----
        self.mtp_stat_steps += 1;
        self.mtp_stat_drafts += drafts.len() as u64;
        self.mtp_stat_accepted += nacc as u64;
        self.mtp_stat_emitted += emit_count as u64;
        self.mtp_stat_verify_fwds += 1;
        // Feed the auto-policy: it needs tokens-per-step to decide whether MTP is still paying.
        self.mtp.record_step(drafts.len() as u64, nacc as u64, emit_count as u64);
        // preds omitted: the stochastic verify SAMPLES rather than taking an argmax, so there is no
        // greedy per-column prediction to record (the parity gate uses the greedy path anyway).
        self.log_draft_step(i, main_pos, committed_tok, &drafts, &[], nacc);
        self.rec_step(SpecStepRec {
            greedy: false, pos: main_pos as u32, drafts: drafts.len() as u32, nacc: nacc as u32,
            emitted: emit_count as u32, round_ms, verify_ms,
            step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
        });
        // ---- S5F3 MTP control dump (dump-only; p_of_draft + residual + bonus — the
        // ---- p-computation cross-check against the DFlash2 lane's verify). ----
        if let Some(d) = self.step_dump.as_mut() {
            let mut dk_ids = Vec::with_capacity(cov_draft_topk.len() * 8);
            let mut dk_logit = Vec::with_capacity(cov_draft_topk.len() * 8);
            for tk in &cov_draft_topk {
                for (t, v) in tk { dk_ids.push(*t); dk_logit.push(*v); }
            }
            let rec = crate::dflash2::stepdump::MtpStepRec {
                step: self.mtp_stat_steps, pos: main_pos, committed: committed_tok, depth,
                drafts: drafts.clone(), p_draft: vsample.p_of_draft.clone(),
                resid: vsample.resid_tok.clone(), bonus: vsample.bonus_tok,
                nacc, emitted: emit_count,
                tgt_ids: Vec::new(), tgt_logit: Vec::new(), tgt_p1: Vec::new(),
                tgt_margin: Vec::new(),
                draft_topk_ids: dk_ids, draft_topk_logit: dk_logit,
                tgt_top20: cov_t20.clone(),
            };
            d.record_mtp(&rec);
        }
        if self.mtp_stat_steps % 50 == 0 {
            let acc = if self.mtp_stat_drafts > 0 {
                self.mtp_stat_accepted as f64 / self.mtp_stat_drafts as f64 * 100.0
            } else { 0.0 };
            let eff = self.mtp_stat_emitted as f64 / self.mtp_stat_verify_fwds as f64;
            eprintln!("[mtp] steps={} drafts={} accepted={:.1}% emitted={} tok/verify_fwd={:.3} (depth {}) accept@k [{}]",
                      self.mtp_stat_steps, self.mtp_stat_drafts, acc,
                      self.mtp_stat_emitted, eff, depth, fmt_accept_by_depth(&self.mtp));
            // A.2: same numbers to the status route (published where the log line prints).
            crate::tel::publish_mtp(self.mtp_stat_steps, self.mtp_stat_drafts,
                                    self.mtp_stat_accepted, self.mtp_stat_emitted,
                                    self.mtp_stat_verify_fwds, depth, &self.mtp.hazard_counts());
            self.dump_accept_curve();
        }
        finished
    }

    // ---- S5F: the DFlash2 speculation lane (b==1 only; the S4F integrated round) --------------

    /// One DFlash2 speculative-decoding step for lane `i` (greedy). The round is the DRAFT source
    /// (S4F: trunk taps → fc/hidden_norm → 5-layer block pass → borrowed LM head → top-16 →
    /// selector chain → 7 draft tokens); the verify is the trunk's M=8 chain verify with the
    /// k_verify≡8 constant (the S2F fold). Accept = longest argmax prefix; rejected drafts are
    /// FREE losslessness (the emitted stream is the target's argmax at every position).
    ///
    /// Ring bookkeeping (the S4F round contract): `nprev == lane.pos == main_pos` at entry — the
    /// ring holds the taps of positions 0..main_pos-1; the anchor (last_tok, at position
    /// main_pos-1... see the invariant note) is the block input, NOT a ctx row. The trunk verify
    /// captures the fed span's taps (cols [0,8) of the sink staging, stream-ordered with the
    /// verify); `inject_dev(nacc+1)` then advances the ring by the accepted span, so the invariant
    /// holds at the next step. The drafter's ring KV is drafter-private (never aliases trunk
    /// slots — the probe asserts the pointer ranges).
    ///
    /// Emits the accepted drafts + bonus, advancing the lane by nacc+1. Returns true if finished.
    /// PLAN/25 Phase 1 — the DF2 TREE lane (opt-in `--spec-source dflash2-tree`, greedy v0).
    /// The selector round already computes a top-16 candidate table for each of the 7 levels in
    /// its single block forward; the chain is one walk through it. This lane walks a SECOND
    /// chain (best per-level candidate that the walk did not pick) and verifies BOTH as a
    /// 15-node tree: per-node KV paths (`verify_forward_topo`), per-node target argmax,
    /// longest root-path accept (lossless — every emitted token is a target argmax token).
    /// Commit: GDN checkpoint of the accepted leaf (per-column ckpts written by the topo
    /// verify), the ring staging compacted to the accepted path (`compact_staging`), then the
    /// standing `inject_dev`. The chain lane (`df2_lane_step`) is untouched when this mode is off.
    fn df2_tree_step(&mut self, i: usize) -> bool {
        const LEVELS: usize = 7;
        // DF2 block-16 x tree is REFUSED, loudly: the tree verifies `1 + LEVELS + nb` nodes with
        // nb <= LEVELS, i.e. up to 15 columns at block 8. At block 16 the same shape is
        // `1 + 15 + 15 = 31` columns, which exceeds `gpu::MAX_VERIFY = 16` — the verify would fall
        // to the prefill dequant path (silently wrong work) or overrun the checkpoint region.
        // Never truncate branch B silently to make it fit; the combination is unsupported until
        // the checkpoint/verify width is raised deliberately.
        assert_eq!(crate::dflash2::block(), 8,
            "df2 block-16 x dflash2-tree is unsupported: the tree needs up to 31 verify columns              (1 + 15 + 15) but MAX_VERIFY is 16. Refusing rather than truncating branch B. \
             Serve block 16 with --spec-source dflash2 (chain) or use the tree at --df2-block 8.");
        assert_eq!(LEVELS, crate::dflash2::levels(),
            "df2 tree LEVELS {} != live levels {}", LEVELS, crate::dflash2::levels());
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let ckpt = self.mtp_snapshot_slot;   // per-column GDN checkpoint base (MAX_VERIFY slots)
        // /health (the tree rider): the tree step is otherwise invisible to the step ring (the
        // chain lanes' note_step sites do not cover this path), so a tree-served turn showed
        // `mode: none` with zero samples while the NEXT turn's MTP window mislabelled the mode.
        crate::tel::note_step();
        // Lazy arm: schedulers not built through `with_df2` (the lossless probe) still get the
        // WIDE tap sink on the first tree step — it must be on the trunk BEFORE the verify's
        // capture D2Ds run. One Option test per step.
        if self.df2_sink_tree.is_none() {
            let ws = std::sync::Arc::new(
                crate::dflash2::capture::Df2TapSink::new_cols(self.gpu.dev(), crate::gpu::MAX_VERIFY));
            self.gpu.set_df2_capture_tree(ws.clone());
            self.df2_sink_tree = Some(ws);
        }
        let kv_stride = self.kv_stride;

        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
        let generated = self.lanes[i].as_ref().unwrap().generated;
        let max_new = self.lanes[i].as_ref().unwrap().max_new;
        let eos = self.eos.clone();
        // Phase-8: the stop rule is a property of the REQUEST, not the serving path —
        // every emit loop must honor ignore_eos/min_new exactly like the batched decode path
        // (batch.rs' `eos_hit`), or spec-on and spec-off streams stop at different tokens.
        let (ignore_eos, min_new) = { let l = self.lanes[i].as_ref().unwrap(); (l.ignore_eos, l.min_new) };
        let (rep_pen, presence_pen, freq_pen, has_penalty) = {
            let l = self.lanes[i].as_ref().unwrap();
            (l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.has_penalty())
        };
        let history: Vec<u32> = self.lanes[i].as_ref().unwrap().history.clone();

        // ---- Draft: the S4F round exactly as the chain runs it (graph when captured). ----
        let step_t0 = std::time::Instant::now();
        let dump_on = self.cov_trace;   // op-sequence gate: must match across TP ranks
        let (drafts, full_out): (Vec<u32>, Option<crate::dflash2::round::Df2RoundOut>) = {
            let df2 = self.df2.as_mut().unwrap();
            assert_eq!(df2.nprev(), main_pos,
                "df2 ring nprev {} != lane pos {} (ring stale or unprimed)", df2.nprev(), main_pos);
            if dump_on {
                // Coverage-trace mode keeps the eager FULL round (its op sequence is what the
                // TP node mirrors and the dump parity harness expects).
                let o = df2.draft_round_full(committed_tok).expect("df2 draft_round_full");
                let toks = o.tokens.clone();
                (toks, Some(o))
            } else {
                // Serving path: the CAPTURED graph (the chain lane's round) + one 448 B dtoh for
                // the top-16 candidate table the B branch ranks from — the full round's ~640 KB
                // h_final readback stays out of the serving step.
                let mk_out = |cand: Vec<u32>| crate::dflash2::round::Df2RoundOut {
                    tokens: Vec::new(), candidates: cand, unary: Vec::new(), scores: Vec::new(),
                    h_final: Vec::new(), layer_hiddens: Vec::new(), logits: None, hp: None,
                    stage_ms: None,
                };
                if df2.round_graph.is_some() {
                    let toks = df2.draft_round_graph(committed_tok).expect("df2 draft_round_graph");
                    let cand = df2.candidate_table().expect("df2 candidate_table");
                    (toks, Some(mk_out(cand)))
                } else {
                    let toks = df2.draft_round_dev(committed_tok).expect("df2 draft_round");
                    let cand = df2.candidate_table().expect("df2 candidate_table");
                    (toks, Some(mk_out(cand)))
                }
            }
        };
        let round_ms = step_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Build the tree: chain A + branch B (best per-level alternate by selector q). ----
        // candidate table layout: [level][16] (token, q); the walk's own chain pick per level is
        // `drafts[j]`. Branch B takes, per level, the highest-q candidate that is neither the
        // chain pick at that level nor any B pick so far (dedupe keeps the tree duplicate-free).
        // The candidate table rides out with the FULL round (head-logit rank order per level —
        // the same table the Phase-0 c_k curves measured). Branch B takes, per level, the
        // highest-ranked candidate that is neither the walk's pick at that level nor any B pick
        // so far (dedupe keeps the tree duplicate-free).
        let cand_tok: Vec<u32> = full_out.as_ref()
            .map(|o| o.candidates.clone())
            .expect("df2-tree: the full round must carry the candidate table");
        let mut b_toks: Vec<u32> = Vec::with_capacity(LEVELS);
        for j in 0..LEVELS {
            let row_t = &cand_tok[j * 16..(j + 1) * 16];
            let mut best: Option<u32> = None;
            for c in 0..16 {
                let t = row_t[c];
                if t == drafts[j] || b_toks.contains(&t) { continue; }
                best = Some(t);
                break;
            }
            match best { Some(t) => b_toks.push(t), None => break }
        }
        let nb = b_toks.len();
        let n = 1 + drafts.len() + nb;             // root + A + B
        let mut tokens: Vec<u32> = Vec::with_capacity(n);
        tokens.push(committed_tok);
        tokens.extend_from_slice(&drafts);
        tokens.extend_from_slice(&b_toks);
        let mut parent: Vec<i32> = vec![-1i32; n];
        for c in 1..=drafts.len() { parent[c] = c as i32 - 1; }
        if nb > 0 { parent[1 + drafts.len()] = 0; } // B rooted at the committed token
        for k in 1..nb { parent[1 + drafts.len() + k] = (drafts.len() + k) as i32; }

        // ---- Verify the tree (per-node KV paths + per-node argmax). ----
        let penalty = self.make_penalty(&history, rep_pen, presence_pen, freq_pen, has_penalty);
        let verify_t0 = std::time::Instant::now();
        let topo = self.gpu.topo_from_parent(&parent, main_pos);
        let (preds, vout) = self.gpu.verify_forward_topo(
            &mut self.pool, &tokens, &mut self.state, phys, kv_stride, main_pos, Some(ckpt),
            penalty, Some(&topo));
        let verify_ms = verify_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Accept walk: follow the target argmax down the tree (mtp_tree_step rule). ----
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        for c in 1..n { children[parent[c] as usize].push(c); }
        let mut path = vec![0usize];
        let mut emitted: Vec<u32> = Vec::new();
        let mut cur = 0usize;
        loop {
            let want = preds[cur];
            emitted.push(want);
            match children[cur].iter().copied().find(|&c| tokens[c] == want) {
                Some(c) => { path.push(c); cur = c; }
                None => break,
            }
        }
        let nacc = path.len() - 1;
        let leaf = *path.last().unwrap();

        // ---- Telemetry: did the B branch actually earn its verify width? (greedy v0 has no
        // other visibility -- the serving A/B showed tree = chain tok/s; this one line says
        // whether that is "rescue rate ~ 0, chain never derails" or "tree never engaged".) ----
        self.df2_tree_stat_steps += 1;
        self.df2_tree_stat_nodes += n as u64;
        if leaf >= 1 + drafts.len() { self.df2_tree_stat_rescues += 1; }
        if self.df2_tree_stat_steps % 50 == 0 {
            let resc = self.df2_tree_stat_rescues as f64
                / self.df2_tree_stat_steps.max(1) as f64 * 100.0;
            let width = self.df2_tree_stat_nodes as f64 / self.df2_tree_stat_steps.max(1) as f64;
            eprintln!("[df2-tree] steps={} b_rescue={:.1}% avg_nodes={:.1} (chain equivalent {})",
                      self.df2_tree_stat_steps, resc, width, LEVELS + 1);
            // /health: publish the tree window too, so the status route can be asserted on
            // (mode == "dflash2-tree") instead of inferred from the MTP window.
            crate::tel::publish_df2_tree(self.df2_tree_stat_steps, self.df2_tree_stat_nodes,
                                         self.df2_tree_stat_rescues);
        }

        // ---- Commit: compact the accepted path's KV into sequential slots [main_pos..). The topo
        // ---- verify writes per-COLUMN slots (main_pos+c); a path that crossed into the B branch
        // ---- leaves accepted tokens at scattered slots, and the next verify reads this span
        // ---- DIRECTLY (dd<0 → slot r) — without the compaction its prefix context is corrupted.
        // ---- Identity short-circuit inside: a pure A-chain accept (the common case) is a no-op.
        {
            let src_pos: Vec<i32> = path.iter().map(|&p| p as i32).collect();
            self.gpu.compact_kv(&mut self.pool, &mut self.state, phys, main_pos, &src_pos, kv_stride);
        }
        // ---- Adopt the leaf's GDN checkpoint (the topo verify wrote per-column ckpts).
        if leaf != n - 1 { self.gpu.copy_gdn_slot(&self.state, ckpt + leaf, phys); }

        // ---- Ring: gather the accepted path's taps (wide tree sink -> ring order), then the
        // ---- standing inject. nacc+1 <= BLOCK always (one branch's depth).
        {
            let wide = self.df2_sink_tree.as_ref()
                .expect("dflash2-tree: wide tap sink not armed");
            let df2 = self.df2.as_mut().unwrap();
            df2.sync_staging_from_wide(wide, &path).expect("df2 sync_staging_from_wide");
            df2.inject_dev(nacc + 1, None).expect("df2 inject_dev");
        }
        self.pool.release_bf16(vout, h * n);

        // ---- Emit accepted + bonus (the standing EOS/max_new discipline). ----
        let mut new_toks: Vec<u32> = Vec::with_capacity(nacc + 1);
        let mut hit_eos = false;
        for k in 0..nacc {
            if generated + new_toks.len() >= max_new { break; }
            new_toks.push(emitted[k]);
            if eos.contains(&emitted[k]) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; break; }
        }
        if !hit_eos && generated + new_toks.len() < max_new {
            new_toks.push(emitted[nacc]);
            if eos.contains(&emitted[nacc]) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; }
        }
        let emit_count = new_toks.len();
        let finished = hit_eos || generated + emit_count >= max_new;

        {
            let cache = &mut self.slot_cache[phys];
            cache.push(committed_tok);
            cache.extend_from_slice(&emitted[..nacc]);
        }
        {
            let lane = self.lanes[i].as_mut().unwrap();
            for &t in &new_toks {
                let _ = lane.tx.send(TokEvent::Tok(t));
                lane.history.push(t);
                if lane.history.len() > 256 { lane.history.drain(0..128); }
            }
            lane.generated += emit_count;
            if !finished {
                lane.last_tok = emitted[nacc];
                lane.pos = main_pos + nacc + 1;
                lane.mtp_pos = main_pos + nacc;
            } else {
                lane.df2_stale = true;
            }
        }
        self.mtp.record_step(drafts.len() as u64, nacc as u64, emit_count as u64);
        self.rec_step(SpecStepRec {
            greedy: true, pos: main_pos as u32, drafts: drafts.len() as u32, nacc: nacc as u32,
            emitted: emit_count as u32, round_ms, verify_ms,
            step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
        });
        finished
    }

    /// P14 — install the DFlash v1 lane. Loads the artifact, allocates the lane's ctx feature and
    /// arms the trunk's wide capture surface (the tap sink reads the same geometry from the
    /// artifact's config.json — no geometry is assumed here).
    pub fn install_dflash_lane(&mut self, dir: &str) -> anyhow::Result<()> {
        use crate::dflash::{DflashDrafter, DflashKv};
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let stride = self.kv_stride;
        let draf = DflashDrafter::load_from_dir(std::path::Path::new(dir), stride + 16)?;
        let (nctx, block, mask) = (draf.nctx, draf.block, draf.mask_token_id);
        let kv = DflashKv::new(&draf, stride + crate::dflash::MAX_BLOCK);
        let pool = crate::gpu::Pool::new(draf.dev.clone());
        let feat = self.gpu.dev().alloc_zeros::<half::bf16>(nctx * h * stride)?;
        self.gpu.arm_dflash_wide(dir, stride)?;
        eprintln!("[dflash] v1 lane installed: {} layers, block {block}, {nctx} taps, mask {mask}, \
                   ctx stride {stride}", draf.layers.len());
        self.dflash = Some(DflashLane { draf, kv, pool, feat, nctx, block, mask, row0: 0, ncols: 0 });
        Ok(())
    }

    /// P14 — prime the v1 lane's ctx feature from the prompt prefill (the trunk's wide capture wrote
    /// every prompt position at its absolute column). The drafter's KV cache is appended lazily by
    /// the first draft step, so priming is a D2D plus the frontier bookkeeping.
    fn dflash_prime(&mut self, plen: usize) {
        if let Some(d) = self.dflash.as_mut() {
            let (feat, nctx) = (&d.feat, d.nctx);
            self.gpu.dflash_feat_copy_at(feat, 0, 0, plen);
            drop(nctx);
            d.row0 = 0;
            d.ncols = plen;
        }
    }

    /// P14 — one DFlash v1 lane step: draft a `block` from the target's hidden states, verify
    /// `[committed, drafts...]` with the trunk, accept the longest agreeing prefix plus the trunk's
    /// own bonus, then hand the verified span's taps to the next step's ctx append.
    ///
    /// Every emitted token is a trunk argmax (`verify_forward`'s preds are the plain-decode argmax
    /// per column — invariant 4), so the emitted chain is plain greedy BY CONSTRUCTION: the MTP
    /// chain's losslessness argument, with a block-wide verify instead of a chain.
    fn dflash_lane_step(&mut self, i: usize) -> bool {
        crate::tel::note_step();
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let snapshot = self.mtp_snapshot_slot;
        let kv_stride = self.kv_stride;
        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
        let generated = self.lanes[i].as_ref().unwrap().generated;
        let max_new = self.lanes[i].as_ref().unwrap().max_new;
        let eos = self.eos.clone();
        let (ignore_eos, min_new) = { let l = self.lanes[i].as_ref().unwrap(); (l.ignore_eos, l.min_new) };
        let (rep_pen, presence_pen, freq_pen, has_penalty) = {
            let l = self.lanes[i].as_ref().unwrap();
            (l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.has_penalty())
        };
        let history: Vec<u32> = self.lanes[i].as_ref().unwrap().history.clone();

        let step_t0 = std::time::Instant::now();
        let block = self.dflash.as_ref().unwrap().block;
        // ---- 1. draft: append the pending ctx span, block forward, then the TARGET's own head over
        // the block and argmax per column (the lane is greedy-only). ----
        let drafts: Vec<u32> = {
            let d = self.dflash.as_mut().unwrap();
            assert_eq!(d.row0 + d.ncols, main_pos,
                "dflash ctx frontier {} != lane pos {main_pos} (unprimed or stale lane)",
                d.row0 + d.ncols);
            let mut bt = Vec::with_capacity(block);
            bt.push(committed_tok);
            bt.extend(std::iter::repeat(d.mask).take(block - 1));
            self.gpu.sync_stream();   // the tap D2Ds run on the engine stream
            let DflashLane { draf, kv, pool, feat, nctx, row0, ncols, .. } = d;
            self.gpu.dflash_draft(draf, kv, pool, feat, *nctx, h, *row0, *ncols, *row0, &bt)
        };
        let round_ms = step_t0.elapsed().as_secs_f32() * 1e3;
        // ---- 2. verify [committed, drafts...] at [main_pos, main_pos + block) ----
        let mut verify_input = Vec::with_capacity(block);
        verify_input.push(committed_tok);
        verify_input.extend(drafts.iter().copied());
        // The lane's verify width is the artifact's whole block (`dflash_draft` drops the block's
        // anchor row, so 1 + (block-1) columns), which must stay inside MAX_VERIFY: past it the
        // verify stops being batch-invariant AND walks a per-layer GDN checkpoint band sized
        // MAX_VERIFY (`verify_forward_core_topo` asserts both — this one names the lane).
        assert_eq!(verify_input.len(), block, "dflash verify width != block");
        assert!(block <= crate::gpu::MAX_VERIFY,
                "dflash block {block} exceeds MAX_VERIFY {}: the verify would fall to the prefill \
                 dequant path and the GDN checkpoint band would overflow", crate::gpu::MAX_VERIFY);
        let penalty = self.make_penalty(&history, rep_pen, presence_pen, freq_pen, has_penalty);
        let verify_t0 = std::time::Instant::now();
        let (preds, vout) = self.gpu.verify_forward(&mut self.pool, &verify_input, &mut self.state,
                                                    phys, kv_stride, main_pos, Some(snapshot), penalty);
        let verify_ms = verify_t0.elapsed().as_secs_f32() * 1e3;
        // ---- 3. accept the longest agreeing prefix; the bonus is the trunk's own argmax ----
        let mut nacc = 0usize;
        while nacc < drafts.len() && preds[nacc] == drafts[nacc] { nacc += 1; }
        let bonus = preds[nacc];
        if nacc + 1 != block { self.gpu.copy_gdn_slot(&self.state, snapshot + nacc, phys); }
        // ---- 4. the verified span's taps are the next ctx append ----
        // The verify captured them into the POSITION-INDEPENDENT ring staging — its D2D lives inside
        // a captured CUDA graph, so it cannot address an absolute column (see the tap site in
        // `verify_forward_core_topo`). This host-issued D2D moves the ACCEPTED span
        // `[0, nacc+1)` to its absolute ctx column. It is the P14 twin of S5F3's
        // `sync_staging_from_sink` before `inject_dev`: without it the drafter conditioned every
        // generated position on ctx columns no forward had written, which is exactly how the DF2
        // lane's τ collapsed 4.06 → 1.27 with every gate still green (PLAN/B8_S5F3_REPORT §R3).
        {
            let d = self.dflash.as_ref().unwrap();
            self.gpu.dflash_tap_to_feature(&d.feat, main_pos, nacc + 1);
        }
        {
            let d = self.dflash.as_mut().unwrap();
            d.row0 = main_pos;
            d.ncols = nacc + 1;
        }
        // ---- 5. emit (the mtp/df2 rule: EOS + max_new on the drafts too) ----
        let mut new_toks: Vec<u32> = Vec::with_capacity(nacc + 1);
        let mut hit_eos = false;
        for &d in drafts.iter().take(nacc) {
            if generated + new_toks.len() >= max_new { break; }
            new_toks.push(d);
            if eos.contains(&d) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; break; }
        }
        if !hit_eos && generated + new_toks.len() < max_new {
            new_toks.push(bonus);
            if eos.contains(&bonus) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; }
        }
        let emit_count = new_toks.len();
        let finished = hit_eos || generated + emit_count >= max_new;
        {
            let cache = &mut self.slot_cache[phys];
            cache.push(committed_tok);
            cache.extend_from_slice(&drafts[..nacc]);
        }
        {
            let lane = self.lanes[i].as_mut().unwrap();
            for &t in &new_toks {
                let _ = lane.tx.send(TokEvent::Tok(t));
                lane.history.push(t);
                if lane.history.len() > 256 { lane.history.drain(0..128); }
            }
            lane.generated += emit_count;
            if !finished {
                lane.last_tok = bonus;
                lane.pos = main_pos + nacc + 1;
            } else {
                lane.last_tok = *new_toks.last().unwrap_or(&committed_tok);
                lane.pos = main_pos + emit_count;
            }
        }
        self.pool.release_bf16(vout, h * block);
        self.rec_step(SpecStepRec {
            greedy: true, pos: main_pos as u32, drafts: drafts.len() as u32, nacc: nacc as u32,
            emitted: emit_count as u32, round_ms, verify_ms,
            step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
        });
        if crate::opts::var(crate::opt!("dflash-step-log")).is_ok() {
            // Field layout mirrors the DF2 line so `accept_gate.py dflash` can reuse its parser:
            // pos / nacc / emitted / committed / round / verify / step. `preds` rides along for the
            // same reason the DF2 line carries it — the FIRST thing to check when acceptance
            // disappoints is the drafts-vs-preds pattern (an off-by-one indexing error shows up as
            // `drafts[j] == preds[j-1]` and looks exactly like "the drafter is weak").
            eprintln!("[dflash-step] pos={main_pos} nacc={nacc} emitted={emit_count} \
                       committed={committed_tok} drafts={:?} preds={:?} round={round_ms:.1}ms \
                       verify={verify_ms:.1}ms step={:.1}ms",
                      &drafts[..drafts.len().min(6)], &preds[..preds.len().min(6)],
                      step_t0.elapsed().as_secs_f32() * 1e3);
        }
        // ---- P14 telemetry: the block lane was INVISIBLE at /health (HANDOFF 2026-09-12 made
        // per-window decode telemetry a standing requirement, and `accept_gate.py` a mandatory gate
        // for any draft-path diff). Cumulative since boot, exactly like the `[mtp]`/`[df2]` meters.
        // accept@k here is COLUMN-indexed: n[k] counts the steps that reached column k, a[k] the
        // steps where column k matched the trunk's argmax (k = 0..block-2 = the 15 proposals).
        self.dflash_stat_steps += 1;
        self.dflash_stat_drafts += drafts.len() as u64;
        self.dflash_stat_accepted += nacc as u64;
        self.dflash_stat_emitted += emit_count as u64;
        for k in 0..drafts.len().min(crate::dflash::MAX_BLOCK) {
            if nacc >= k { self.dflash_acc_n[k] += 1; }
            if nacc > k { self.dflash_acc_a[k] += 1; }
        }
        // Published EVERY step (not every 50 like MTP/DF2): the block lane commits up to 16
        // tokens per step, so a short probe request is ~6-16 steps — a 50-step window would leave
        // /health mute exactly when an operator looks at it. The log meter below prints at a
        // 10-step cadence, which for a 16-token block is the same lines-per-token as MTP's 50.
        // Stack array, not a Vec: this is a decode-step hot path and tel.rs's contract is
        // "zero hot-path structure change, no lock, no allocation".
        let mut acc_pairs = [(0u64, 0u64); crate::dflash::MAX_BLOCK];
        for k in 0..drafts.len().min(crate::dflash::MAX_BLOCK) {
            acc_pairs[k] = (self.dflash_acc_a[k], self.dflash_acc_n[k]);
        }
        crate::tel::publish_dflash(
            self.dflash_stat_steps, self.dflash_stat_drafts, self.dflash_stat_accepted,
            self.dflash_stat_emitted, block, &acc_pairs[..drafts.len()]);
        if self.dflash_stat_steps % 10 == 0 {
            let acc = if self.dflash_stat_drafts > 0 {
                self.dflash_stat_accepted as f64 / self.dflash_stat_drafts as f64 * 100.0
            } else { 0.0 };
            let yield_tok = self.dflash_stat_emitted as f64 / self.dflash_stat_steps as f64;
            let mut ak = String::new();
            for k in 0..drafts.len() {
                let n = self.dflash_acc_n[k];
                if n > 0 {
                    ak.push_str(&format!(" @{}:{:.0}%", k + 1,
                        self.dflash_acc_a[k] as f64 / n as f64 * 100.0));
                }
            }
            eprintln!("[dflash] steps={} block={block} drafts={} accepted={acc:.1}% emitted={} \
                       yield={yield_tok:.2} tok/step accept@k[{ak} ]",
                      self.dflash_stat_steps, self.dflash_stat_drafts, self.dflash_stat_emitted);
        }
        finished
    }

    fn df2_lane_step(&mut self, i: usize) -> bool {
        crate::tel::note_step();   // A.2: per-step interval for the status route (lock-free)
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let snapshot = self.mtp_snapshot_slot;
        let kv_stride = self.kv_stride;

        // Snapshot lane state into locals (avoids holding &mut self.lanes across GPU calls).
        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
        let generated = self.lanes[i].as_ref().unwrap().generated;
        let max_new = self.lanes[i].as_ref().unwrap().max_new;
        let eos = self.eos.clone();
        // Phase-8: the stop rule is a property of the REQUEST, not the serving path —
        // every emit loop must honor ignore_eos/min_new exactly like the batched decode path
        // (batch.rs' `eos_hit`), or spec-on and spec-off streams stop at different tokens.
        let (ignore_eos, min_new) = { let l = self.lanes[i].as_ref().unwrap(); (l.ignore_eos, l.min_new) };
        let (rep_pen, presence_pen, freq_pen, has_penalty) = {
            let l = self.lanes[i].as_ref().unwrap();
            (l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.has_penalty())
        };
        let history: Vec<u32> = self.lanes[i].as_ref().unwrap().history.clone();

        // ---- Draft: the S4F round (refresh block positions, then the lean draft). ----
        let step_t0 = std::time::Instant::now();
        let dump_on = self.cov_trace;   // op-sequence gate: must match across TP ranks
        let (drafts, full_out): (Vec<u32>, Option<crate::dflash2::round::Df2RoundOut>) = {
            let df2 = self.df2.as_mut().unwrap();
            assert_eq!(df2.nprev(), main_pos,
                "df2 ring nprev {} != lane pos {} (ring stale or unprimed)", df2.nprev(), main_pos);
            if dump_on {
                // S5F3 dump: the eager FULL round (tokens + candidates/unary/scores + h_final)
                // — behavior-neutral vs the graph/eager lean path (probe: graph == eager
                // bit-identical); the extra readbacks are the dump's payload.
                df2.refresh_block_pos().expect("df2 refresh_block_pos");
                let o = df2.draft_round_full(committed_tok).expect("df2 draft_round_full");
                (o.tokens.clone(), Some(o))
            } else if df2.round_graph.is_some() {
                // S5F graph path: draft_round_graph writes the per-replay device inputs
                // (anchor, nprev, block positions) + gathers the block RoPE itself.
                (df2.draft_round_graph(committed_tok).expect("df2 draft_round_graph"), None)
            } else {
                df2.refresh_block_pos().expect("df2 refresh_block_pos");
                (df2.draft_round_dev(committed_tok).expect("df2 draft_round"), None)
            }
        };
        let round_ms = step_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Verify [committed_tok, drafts...] (M=8 buckets) on the trunk at main_pos.. ----
        // The chain verify captures its own CUDA graph (n=8 key) — the DFlash2 "verify pair".
        let mut verify_input = vec![committed_tok];
        verify_input.extend(drafts.iter().copied());
        let penalty = self.make_penalty(&history, rep_pen, presence_pen, freq_pen, has_penalty);
        let verify_t0 = std::time::Instant::now();
        // PLAN/25 Phase 0 (cov_on): keep the verify logits and derive preds + the per-column
        // target top-4/p1/top1−top2 margin host-side (the bench_accept readback; the eager core
        // is exactly what the E13 graph captures, so preds are identical to the non-dump path).
        // Dump-only. cov_on already folds in !has_penalty (penalized steps keep the plain
        // penalized verify and carry empty tgt fields; has_penalty is rank-identical — SPMD-safe).
        let cov_on = self.cov_trace && !has_penalty;
        let (preds, vout, cov_tgt) = if cov_on {
            let (logits, vout) = self.gpu.verify_forward_keep_logits(
                &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos,
                Some(snapshot));
            let n = verify_input.len();
            let vocab = self.gpu.cfg().vocab_size;
            let lg_f32: Vec<f32> = match logits {
                crate::gpu::VerifyLogits::B16(lg) => {
                    let v: Vec<half::bf16> = self.gpu.dev().dtoh_sync_copy(&lg).unwrap();
                    self.pool.release_bf16(lg, vocab * n);
                    v.iter().map(|x| x.to_f32()).collect()
                }
                crate::gpu::VerifyLogits::F32(lg) => {
                    let v: Vec<f32> = self.gpu.dev().dtoh_sync_copy(&lg).unwrap();
                    self.pool.release(lg, vocab * n);
                    v
                }
            };
            let mut preds = Vec::with_capacity(n);
            let (mut tids, mut tlog) = (Vec::with_capacity(n * 4), Vec::with_capacity(n * 4));
            let (mut tp1, mut tmar) = (Vec::with_capacity(n), Vec::with_capacity(n));
            for c in 0..n {
                let (ids, vals, p1, margin) = crate::dflash2::stepdump::tgt_topk_host(
                    &lg_f32[c * vocab..(c + 1) * vocab], 4);
                preds.push(ids[0]);
                tids.extend_from_slice(&ids); tlog.extend_from_slice(&vals);
                tp1.push(p1); tmar.push(margin);
            }
            (preds, vout, Some((tids, tlog, tp1, tmar)))
        } else {
            let (preds, vout) = self.gpu.verify_forward(
                &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos,
                Some(snapshot), penalty);
            (preds, vout, None)
        };
        let verify_ms = verify_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Accept longest prefix (greedy: drafts[i] accepted iff preds[i]==drafts[i]). ----
        let mut nacc = 0usize;
        while nacc < drafts.len() && preds[nacc] == drafts[nacc] { nacc += 1; }
        let bonus = preds[nacc];

        // ---- GDN rollback on partial reject (restore the state as of the last accepted column). --
        if nacc + 1 != crate::dflash2::block() {
            self.gpu.copy_gdn_slot(&self.state, snapshot + nacc, phys);
        }

        // ---- PLAN/25 Phase 0: the LOGGING-ONLY MTP chain (coverage capture, dump-only). ----
        // Runs the MTP head over the SAME committed prefix this step's DF2 round drafted from,
        // recording its top-8 per position — the union instrument: both drafters' candidate sets
        // on ONE step grid, joined by (tag, pos) offline with zero grid-alignment bias. The
        // chain writes nothing but the record fields: the DF2 drafts above are untouched, and the
        // head's speculative KV entries beyond the re-primed span have exactly the mtp lane's own
        // semantics (always rewritten before they are ever attended).
        let mut cov_mtp_sets: Option<(Vec<u32>, Vec<f32>)> = None;
        if cov_on {
            let kc = *self.mtp_kc[phys].device_ptr();
            let vc = *self.mtp_vc[phys].device_ptr();
            let cur_ptr = *self.mtp_cur_hidden.as_ref().unwrap().device_ptr();
            self.gpu.copy_hidden_col(cur_ptr, &self.mtp_h_prev[phys], 0);
            let mut cur_tok = committed_tok as i32;
            let mut dpos = main_pos - 1;
            let mut mids = Vec::with_capacity(7 * 8);
            let mut mlog = Vec::with_capacity(7 * 8);
            for _ in 0..7 {
                let m = self.gpu.mtp_draft_step(
                    &mut self.pool, self.mtp_cur_hidden.as_ref().unwrap(), cur_tok, dpos,
                    kc, vc, kv_stride);
                self.gpu.copy_hidden_col(cur_ptr, &m, 0);
                self.pool.release_bf16(m, h);
                cur_tok = self.gpu.argmax_hidden(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap()) as i32;
                for (t, v) in self.gpu.topk_hidden_kv(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap(), 8) {
                    mids.push(t); mlog.push(v);
                }
                dpos += 1;
            }
            cov_mtp_sets = Some((mids, mlog));
            // Re-prime the head over the REAL accepted span (mirror of mtp_lane_step's re-prime:
            // col 0 from the cursor hidden, then the verify's real hiddens), so the next step's
            // chain conditions on the committed prefix exactly as an MTP lane would.
            let n_rp = nacc + 1;
            let rp_hidden = self.pool.get_bf16(h * n_rp);
            let rp_ptr = *rp_hidden.device_ptr();
            let mut rp_toks: Vec<u32> = Vec::with_capacity(n_rp);
            self.gpu.copy_hidden_col(rp_ptr, &self.mtp_h_prev[phys], 0);
            rp_toks.push(committed_tok);
            self.gpu.copy_hidden_cols(rp_ptr + (h * 2) as u64, &vout, 0, nacc);
            for k in 1..=nacc { rp_toks.push(drafts[k - 1]); }
            self.gpu.mtp_reprime(&mut self.pool, &rp_hidden, &rp_toks, main_pos - 1, kc, vc, kv_stride);
            self.pool.release_bf16(rp_hidden, h * n_rp);
            // Cursor hidden ← hidden at the last accepted column (the next step's h_prev twin).
            self.gpu.copy_hidden_col(*self.mtp_h_prev[phys].device_ptr(), &vout, nacc);
        }

        // ---- Inject the accepted span's taps into the ring (cols [0, nacc+1) of the sink
        // ---- staging — written by the verify's capture D2Ds, stream-ordered before the
        // ---- verify's own dtoh readback, which this call follows). nprev = main_pos+nacc+1.
        {
            let df2 = self.df2.as_mut().unwrap();
            // S5F3 fix: copy the sink's LIVE staging into the round before the inject (the
            // attach-time deep copy never saw the trunk's captures — the draft-parity root
            // cause). The verify synced the capture before returning.
            df2.sync_staging_from_sink().expect("df2 sync staging");
            df2.inject_dev(nacc + 1, None).expect("df2 inject_dev");
        }
        self.pool.release_bf16(vout, h * 8);


        // ---- Emit accepted drafts + bonus, honoring EOS and max_new (the mtp_lane_step rule:
        // ---- the budget check belongs on the drafts too). ----
        let mut new_toks: Vec<u32> = Vec::with_capacity(nacc + 1);
        let mut hit_eos = false;
        for &d in drafts.iter().take(nacc) {
            if generated + new_toks.len() >= max_new { break; }
            new_toks.push(d);
            if eos.contains(&d) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; break; }
        }
        if !hit_eos && generated + new_toks.len() < max_new {
            new_toks.push(bonus);
            if eos.contains(&bonus) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; }
        }
        let emit_count = new_toks.len();
        let finished = hit_eos || generated + emit_count >= max_new;

        {
            let cache = &mut self.slot_cache[phys];
            cache.push(committed_tok);
            cache.extend_from_slice(&drafts[..nacc]);
        }
        {
            let lane = self.lanes[i].as_mut().unwrap();
            for &t in &new_toks {
                let _ = lane.tx.send(TokEvent::Tok(t));
                lane.history.push(t);
                if lane.history.len() > 256 { lane.history.drain(0..128); }
            }
            lane.generated += emit_count;
            if !finished {
                lane.last_tok = bonus;
                lane.pos = main_pos + nacc + 1;
            } else {
                lane.last_tok = *new_toks.last().unwrap_or(&committed_tok);
                lane.pos = main_pos + emit_count;
            }
        }

        // ---- S5F3 dump record (dump-only; the staging readback is stream-ordered after the
        // ---- verify's capture D2Ds — the verify synced before returning). ----
        if let Some(d) = self.step_dump.as_mut() {
            let staging: Vec<half::bf16> = self.df2_sink.as_ref()
                .and_then(|s| self.gpu.dev().dtoh_sync_copy(&s.staging).ok())
                .unwrap_or_default();
            d.record_span(main_pos, 8, &staging);
            let ck = crate::dflash2::stepdump::StepDump::tap_checksums(&staging);
            // S5F3 deep-copy-clone check: the ROUND's staging vs the SINK's staging (step 0)
            if self.df2_stat_steps == 0 {
                if let Some(df2) = self.df2.as_mut() {
                    if let Ok(rs) = df2.dump_staging() {
                        let sink_f: Vec<f32> = staging.iter().map(|x| x.to_f32()).collect();
                        let mut nd = 0.0f64; let mut dd = 0.0f64;
                        for i in 0..rs.len().min(sink_f.len()) {
                            let a = rs[i] as f64; let b = sink_f[i] as f64;
                            nd += (a - b) * (a - b); dd += b * b;
                        }
                        eprintln!("[df2-dump] STAGING CHECK step0: round-vs-sink relL2 {:.4e} (round[0..4]={:?} sink[0..4]={:?})",
                                 (nd / dd.max(1e-30)).sqrt(), &rs[..4], &sink_f[..4]);
                    }
                }
            }
            let rec = crate::dflash2::stepdump::Df2StepRec {
                step: self.df2_stat_steps, pos: main_pos, committed: committed_tok, greedy: true, realq: false,
                drafts: drafts.clone(), p_draft: Vec::new(),
                resid: preds[..7.min(preds.len())].to_vec(), bonus,
                nacc, emitted: emit_count, q_rows: vec![1.0; crate::dflash2::levels()],
                candidates: full_out.as_ref().map(|o| o.candidates.clone()).unwrap_or_default(),
                cand_q: Vec::new(),
                unary: full_out.as_ref().map(|o| o.unary.clone()).unwrap_or_default(),
                scores: full_out.as_ref().map(|o| o.scores.clone()).unwrap_or_default(),
                top20: Vec::new(), tap_ck: ck,
                hfinal_written: full_out.is_some() && self.df2_stat_steps
                    < crate::dflash2::stepdump::RAW_STEPS as u64,
                // PLAN/25 Phase 0: the target-side coverage readback + the logging-only MTP
                // chain's top-8 per position (the union instrument).
                tgt_ids: cov_tgt.as_ref().map(|(a, ..)| a.clone()).unwrap_or_default(),
                tgt_logit: cov_tgt.as_ref().map(|(_, b, ..)| b.clone()).unwrap_or_default(),
                tgt_p1: cov_tgt.as_ref().map(|(.., c, _)| c.clone()).unwrap_or_default(),
                tgt_margin: cov_tgt.as_ref().map(|(.., e)| e.clone()).unwrap_or_default(),
                mtp_ids: cov_mtp_sets.as_ref().map(|(a, _)| a.clone()).unwrap_or_default(),
                mtp_logit: cov_mtp_sets.as_ref().map(|(_, b)| b.clone()).unwrap_or_default(),
            };
            let hf = full_out.as_ref().map(|o| o.h_final.as_slice());
            d.record_df2(&rec, hf);
            // S5F3 ring rows (first RING_STEPS steps): ctx rows near C + the injected span.
            if self.df2_stat_steps < crate::dflash2::stepdump::RING_STEPS as u64 {
                let df2 = self.df2.as_mut().unwrap();
                let lo = main_pos.saturating_sub(2);
                let mut rk = Vec::new();
                let mut rv = Vec::new();
                for li in 0..crate::dflash2::N_LAYERS {
                    if let Ok((k, v)) = df2.dump_ring_rows(li, lo, main_pos + 8) {
                        rk.push(k); rv.push(v);
                    }
                }
                if rk.len() == crate::dflash2::N_LAYERS { d.record_ring_rows(self.df2_stat_steps, &rk, &rv); }
            }
        }

        // ---- Telemetry (df2-specific; the MTP policy's hazard curve stays MTP-only). ----
        self.rec_step(SpecStepRec {
            greedy: true, pos: main_pos as u32, drafts: drafts.len() as u32, nacc: nacc as u32,
            emitted: emit_count as u32, round_ms, verify_ms,
            step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
        });
        if crate::opts::var(crate::opt!("df2-step-log")).is_ok() {
            let step_ms = step_t0.elapsed().as_secs_f32() * 1e3;
            eprintln!("[df2-step] pos={main_pos} nacc={nacc} emitted={emit_count} committed={committed_tok} \
                       drafts={:?} preds={:?} round={round_ms:.1}ms verify={verify_ms:.1}ms step={step_ms:.1}ms",
                      &drafts[..drafts.len().min(4)], &preds[..preds.len().min(4)]);
        }
        // DF2_CARRY: the ring now reflects THIS slot's committed prefix up to the lane's frontier
        // (`nprev` = the whole committed sequence: prompt + emitted). Record it so the next request
        // that reuses this slot's prefix can carry the ring. Twin of the DSpark
        // `slot_dspark_len` write in `dspark_lane_step`.
        if let Some(d) = self.df2.as_mut() { d.note_ring(phys); }
        self.df2_stat_steps += 1;
        self.df2_stat_drafts += drafts.len() as u64;
        self.df2_stat_accepted += nacc as u64;
        self.df2_stat_emitted += emit_count as u64;
        if self.df2_stat_steps % 50 == 0 {
            let acc = if self.df2_stat_drafts > 0 {
                self.df2_stat_accepted as f64 / self.df2_stat_drafts as f64 * 100.0
            } else { 0.0 };
            eprintln!("[df2] steps={} drafts={} accepted={:.1}% emitted={} tok/step={:.3}",
                      self.df2_stat_steps, self.df2_stat_drafts, acc,
                      self.df2_stat_emitted,
                      self.df2_stat_emitted as f64 / self.df2_stat_steps as f64);
            // A.2: same numbers to the status route (published exactly where the log line prints).
            crate::tel::publish_df2(self.df2_stat_steps, self.df2_stat_drafts,
                                    self.df2_stat_accepted, self.df2_stat_emitted);
        }
        finished
    }

    /// WI1: one DSpark speculative step for a GREEDY lane — the df2_lane_step shape with the
    /// DSpark round in the draft seat. Draft (7 tokens), verify width 8 on the trunk, longest
    /// prefix accept + bonus, GDN rollback on partial reject, tap inject of the accepted span.
    /// The DF2 dump/coverage instruments are DF2-only (this lane keeps the plain path + the
    /// shared SpecStepRec telemetry); the confidence head is diagnostics-only in v1 (adaptive
    /// k_verify OFF — the recipe default).
    fn dspark_lane_step(&mut self, i: usize) -> bool {
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let snapshot = self.mtp_snapshot_slot;
        let kv_stride = self.kv_stride;

        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
        let generated = self.lanes[i].as_ref().unwrap().generated;
        let max_new = self.lanes[i].as_ref().unwrap().max_new;
        let eos = self.eos.clone();
        // Phase-8: the stop rule is a property of the REQUEST, not the serving path —
        // every emit loop must honor ignore_eos/min_new exactly like the batched decode path
        // (batch.rs' `eos_hit`), or spec-on and spec-off streams stop at different tokens.
        let (ignore_eos, min_new) = { let l = self.lanes[i].as_ref().unwrap(); (l.ignore_eos, l.min_new) };
        let (rep_pen, presence_pen, freq_pen, has_penalty) = {
            let l = self.lanes[i].as_ref().unwrap();
            (l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.has_penalty())
        };
        let history: Vec<u32> = self.lanes[i].as_ref().unwrap().history.clone();

        // ---- Draft: the DSpark round (eager; graphs follow the DF2 pattern once oracle-proven).
        let step_t0 = std::time::Instant::now();
        static DUMP_REQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dump_steps: u64 = crate::opts::var(crate::opt!("dspark-dump-steps")).ok()
            .and_then(|v| v.parse().ok()).unwrap_or(6);
        if self.dspark_stat_steps < dump_steps {
            if let Ok(dir) = crate::opts::var(crate::opt!("dspark-dump-step0")) {
                let rseq = if self.dspark_stat_steps == 0 {
                    DUMP_REQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                } else { DUMP_REQ.load(std::sync::atomic::Ordering::Relaxed).saturating_sub(1) };
                let sdir = format!("{dir}/r{rseq}/step{}", self.dspark_stat_steps);
                let _ = std::fs::create_dir_all(&sdir);
                // step0 only: the DRAFT's own inputs — the wide prime sink's taps + plen
                if self.dspark_stat_steps == 0 {
                    let rdir = format!("{dir}/r{rseq}");
                    // the exact prompt token ids (for the offline ground-truth prefill compare)
                    let ids: Vec<u32> = self.slot_cache[phys].iter().take(main_pos).copied().collect();
                    let mut ib = Vec::with_capacity(ids.len() * 4);
                    for t in &ids { ib.extend_from_slice(&t.to_le_bytes()); }
                    let _ = std::fs::write(format!("{rdir}/prompt.bin"), &ib);
                    eprintln!("[dspark] step0 dump: prime sink present={}", self.dspark_prime.is_some());
                    if let Some(ps) = &self.dspark_prime {
                        let _ = std::fs::create_dir_all(&dir);
                        let taps: Vec<half::bf16> = self.gpu.dev().dtoh_sync_copy(&ps.taps).unwrap_or_default();
                        let cols = taps.len() / crate::dspark::TAP_CONCAT_DIM;
                        eprintln!("[dspark] step0: taps readback {} elems ({} cols)", taps.len(), cols);
                        // only the LIVE columns [0..main_pos) — the rest is stale/zero padding
                        let live = main_pos.min(cols);
                        let mut tb = Vec::with_capacity(live * crate::dspark::TAP_CONCAT_DIM * 2);
                        for t in &taps[..live * crate::dspark::TAP_CONCAT_DIM] {
                            tb.extend_from_slice(&t.to_le_bytes());
                        }
                        let _ = std::fs::write(format!("{dir}/r{rseq}/taps_wide.bin"), &tb);
                        let _ = std::fs::write(format!("{dir}/r{rseq}/plen.txt"),
                                               format!("{} {}", main_pos, live));
                    }
                }
                // every dumped step: nothing pre-draft (the logits buffer is only valid AFTER
                // the draft; the post-verify block below dumps it with the staging)
            }
        }
        let drafts: Vec<u32> = {
            let ds = self.dspark.as_mut().unwrap();
            assert_eq!(ds.nprev(), main_pos,
                "dspark ring nprev {} != lane pos {} (ring stale or unprimed)", ds.nprev(), main_pos);
            ds.draft_round_dev(committed_tok).expect("dspark draft_round")
        };
        let round_ms = step_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Verify [committed_tok, drafts...] (M=8 buckets) on the trunk at main_pos.. ----
        let mut verify_input = vec![committed_tok];
        verify_input.extend(drafts.iter().copied());
        let penalty = self.make_penalty(&history, rep_pen, presence_pen, freq_pen, has_penalty);
        let verify_t0 = std::time::Instant::now();
        let (preds, vout) = self.gpu.verify_forward(
            &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos,
            Some(snapshot), penalty);
        let verify_ms = verify_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Accept longest prefix (greedy) + bonus. ----
        let mut nacc = 0usize;
        while nacc < drafts.len() && preds[nacc] == drafts[nacc] { nacc += 1; }
        let bonus = preds[nacc];

        // ---- GDN rollback on partial reject. ----
        if nacc + 1 != 8 {
            self.gpu.copy_gdn_slot(&self.state, snapshot + nacc, phys);
        }

        // ---- Inject the accepted span's taps into the DSpark ring (the S5F3 live-staging copy
        // ---- then the inject; nprev = main_pos+nacc+1). ----
        {
            let ds = self.dspark.as_mut().unwrap();
            ds.sync_staging_from_sink().expect("dspark sync staging");
            ds.inject_dev(nacc + 1).expect("dspark inject_dev");
        }
        self.pool.release_bf16(vout, h * 8);

        // ---- Emit accepted drafts + bonus (the mtp/df2 lane rule: budget check on drafts too). --
        let mut new_toks: Vec<u32> = Vec::with_capacity(nacc + 1);
        let mut hit_eos = false;
        for &d in drafts.iter().take(nacc) {
            if generated + new_toks.len() >= max_new { break; }
            new_toks.push(d);
            if eos.contains(&d) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; break; }
        }
        if !hit_eos && generated + new_toks.len() < max_new {
            new_toks.push(bonus);
            if eos.contains(&bonus) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; }
        }
        let emit_count = new_toks.len();
        let finished = hit_eos || generated + emit_count >= max_new;

        {
            let cache = &mut self.slot_cache[phys];
            cache.push(committed_tok);
            cache.extend_from_slice(&drafts[..nacc]);
        }
        {
            let lane = self.lanes[i].as_mut().unwrap();
            for &t in &new_toks {
                let _ = lane.tx.send(TokEvent::Tok(t));
                lane.history.push(t);
                if lane.history.len() > 256 { lane.history.drain(0..128); }
            }
            lane.generated += emit_count;
            if !finished {
                lane.last_tok = bonus;
                lane.pos = main_pos + nacc + 1;
            } else {
                lane.last_tok = *new_toks.last().unwrap_or(&committed_tok);
                lane.pos = main_pos + emit_count;
            }
        }

        // ---- Telemetry (the df2 shape; own counters so the [dspark] step log stands alone). ----
        self.rec_step(SpecStepRec {
            greedy: true, pos: main_pos as u32, drafts: drafts.len() as u32, nacc: nacc as u32,
            emitted: emit_count as u32, round_ms, verify_ms,
            step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
        });
        if crate::opts::var(crate::opt!("dspark-step-log")).is_ok() {
            let step_ms = step_t0.elapsed().as_secs_f32() * 1e3;
            eprintln!("[dspark-step] pos={main_pos} nacc={nacc} emitted={emit_count} committed={committed_tok} \
                       drafts={:?} preds={:?} round={round_ms:.1}ms verify={verify_ms:.1}ms step={step_ms:.1}ms",
                      &drafts[..drafts.len().min(4)], &preds[..preds.len().min(4)]);
        }
        // --dspark-dump-step0: for the FIRST N dspark steps (--dspark-dump-steps, default
        // 6), dump the staging taps (the round's own [8, 25600] bf16 staging = the ctx-extension
        // the next draft saw), the anchor, the drafts and the trunk preds — the offline
        // oracle-vs-trunk comparison's raw material, and the per-step reconstruction basis.
        if self.dspark_stat_steps < dump_steps {
            if let Ok(dir) = crate::opts::var(crate::opt!("dspark-dump-step0")) {
                let rseq = DUMP_REQ.load(std::sync::atomic::Ordering::Relaxed).saturating_sub(1);
                let sdir = format!("{dir}/r{rseq}/step{}", self.dspark_stat_steps);
                let ds = self.dspark.as_mut().unwrap();
                let _ = std::fs::create_dir_all(&sdir);
                let taps: Vec<half::bf16> = ds.dump_staging().unwrap_or_default();
                let mut tb = Vec::with_capacity(taps.len() * 2);
                for t in &taps { tb.extend_from_slice(&t.to_le_bytes()); }
                let _ = std::fs::write(format!("{sdir}/staging.bin"), &tb);
                let mut meta = String::new();
                meta.push_str(&format!("anchor {committed_tok}\n"));
                meta.push_str(&format!("drafts {drafts:?}\n"));
                meta.push_str(&format!("preds {preds:?}\n"));
                meta.push_str(&format!("nacc {nacc}\n"));
                meta.push_str(&format!("nprev {}\n", ds.nprev()));
                let _ = std::fs::write(format!("{sdir}/meta.txt"), &meta);
                // THIS step's draft logits ([7, VOCAB] f32, token-major) — the chain-formula
                // analysis runs on them offline (variants vs the trunk's preds).
                if let Ok(lg) = ds.dump_logits() {
                    let mut lb = Vec::with_capacity(lg.len() * 4);
                    for v in &lg { lb.extend_from_slice(&v.to_le_bytes()); }
                    let _ = std::fs::write(format!("{sdir}/logits.bin"), &lb);
                }
                eprintln!("[dspark] step{} dump written (anchor {committed_tok})", self.dspark_stat_steps);
            }
        }
        self.dspark_stat_steps += 1;
        self.dspark_stat_drafts += drafts.len() as u64;
        self.dspark_stat_accepted += nacc as u64;
        self.dspark_stat_emitted += emit_count as u64;
        // A6: the round's ring rows are valid through the lane's whole committed sequence —
        // record it so a follow-up request reusing this slot's prefix can carry the ring.
        // The identity claim is bound HERE (after the round ran for `phys`), which is what makes
        // it true; `slot_dspark_len` alone is only the length half of the guard
        // (PLAN/DSPARK_RING_IDENTITY_SPEC.md §3.2).
        self.slot_dspark_len[phys] = self.slot_cache[phys].len();
        if let Some(d) = self.dspark.as_mut() { d.note_ring(phys); }
        if self.dspark_stat_steps % 50 == 0 {
            let acc = if self.dspark_stat_drafts > 0 {
                self.dspark_stat_accepted as f64 / self.dspark_stat_drafts as f64 * 100.0
            } else { 0.0 };
            eprintln!("[dspark] steps={} drafts={} accepted={:.1}% emitted={} tok/step={:.3}",
                      self.dspark_stat_steps, self.dspark_stat_drafts, acc,
                      self.dspark_stat_emitted,
                      self.dspark_stat_emitted as f64 / self.dspark_stat_steps as f64);
        }
        finished
    }

    /// WI1: one DSpark speculative step for a SAMPLING lane — the same 7 greedy drafts,
    /// verified with `verify_forward_sample` under qprobs = 1.0 (the df2_lane_step_sample
    /// pattern; greedy drafts = a point-mass proposal, so rejection sampling reduces to the
    /// greedy accept + the residual replacement, distribution-exact by construction).
    fn dspark_lane_step_sample(&mut self, i: usize) -> bool {
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let snapshot = self.mtp_snapshot_slot;
        let kv_stride = self.kv_stride;

        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
        let generated = self.lanes[i].as_ref().unwrap().generated;
        let max_new = self.lanes[i].as_ref().unwrap().max_new;
        let eos = self.eos.clone();
        // Phase-8: the stop rule is a property of the REQUEST, not the serving path —
        // every emit loop must honor ignore_eos/min_new exactly like the batched decode path
        // (batch.rs' `eos_hit`), or spec-on and spec-off streams stop at different tokens.
        let (ignore_eos, min_new) = { let l = self.lanes[i].as_ref().unwrap(); (l.ignore_eos, l.min_new) };
        let temperature = self.lanes[i].as_ref().unwrap().temperature;
        let top_k = self.lanes[i].as_ref().unwrap().top_k;
        let top_p = self.lanes[i].as_ref().unwrap().top_p;
        let (rep_pen, presence_pen, freq_pen, has_penalty) = {
            let l = self.lanes[i].as_ref().unwrap();
            (l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.has_penalty())
        };
        let history: Vec<u32> = self.lanes[i].as_ref().unwrap().history.clone();
        let step_key = self.lanes[i].as_ref().unwrap().seed;

        // ---- Draft (same as the greedy lane). ----
        let step_t0 = std::time::Instant::now();
        let drafts: Vec<u32> = {
            let ds = self.dspark.as_mut().unwrap();
            assert_eq!(ds.nprev(), main_pos,
                "dspark ring nprev {} != lane pos {} (ring stale or unprimed)", ds.nprev(), main_pos);
            ds.draft_round_dev(committed_tok).expect("dspark draft_round")
        };
        let round_ms = step_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Verify with stochastic output. ----
        let mut verify_input = vec![committed_tok];
        verify_input.extend(drafts.iter().copied());
        let verify_seeds: Vec<u32> =
            (0..crate::dflash2::block()).map(|j| rng_u32(step_key, RNG_DOM_VERIFY, j)).collect();
        let qprobs: Vec<f32> = vec![1.0; crate::dflash2::levels()];
        let penalty = self.make_penalty(&history, rep_pen, presence_pen, freq_pen, has_penalty);
        let verify_t0 = std::time::Instant::now();
        let (vsample, vout) = self.gpu.verify_forward_sample(
            &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos,
            Some(snapshot), penalty,
            &drafts, &qprobs, temperature, top_k, top_p, &verify_seeds, None);
        let verify_ms = verify_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Rejection-sampling accept loop (q = 1). ----
        let mut nacc = 0usize;
        let mut emitted: Vec<u32> = Vec::with_capacity(8);
        let mut rejected = false;
        for j in 0..drafts.len() {
            let ru = rng_uniform(step_key, RNG_DOM_ACCEPT, j);
            if ru < vsample.p_of_draft[j] {
                emitted.push(drafts[j]);
                nacc += 1;
            } else {
                emitted.push(vsample.resid_tok[j]);
                rejected = true;
                break;
            }
        }
        if !rejected { emitted.push(vsample.bonus_tok); }

        // ---- GDN rollback + tap inject. ----
        if nacc + 1 != 8 {
            self.gpu.copy_gdn_slot(&self.state, snapshot + nacc, phys);
        }
        {
            let ds = self.dspark.as_mut().unwrap();
            ds.sync_staging_from_sink().expect("dspark sync staging");
            ds.inject_dev(nacc + 1).expect("dspark inject_dev");
        }
        self.pool.release_bf16(vout, h * 8);

        // ---- Emit. ----
        let mut hit_eos = false;
        let mut to_emit: Vec<u32> = Vec::with_capacity(emitted.len());
        for &t in &emitted {
            to_emit.push(t);
            if eos.contains(&t) && !ignore_eos && generated + to_emit.len() >= min_new { hit_eos = true; break; }
            if generated + to_emit.len() >= max_new { break; }
        }
        let emit_count = to_emit.len();
        let finished = hit_eos || generated + emit_count >= max_new;

        {
            let cache = &mut self.slot_cache[phys];
            cache.push(committed_tok);
            cache.extend_from_slice(&drafts[..nacc]);
        }
        {
            let lane = self.lanes[i].as_mut().unwrap();
            for &t in &to_emit {
                let _ = lane.tx.send(TokEvent::Tok(t));
                lane.history.push(t);
                if lane.history.len() > 256 { lane.history.drain(0..128); }
            }
            lane.generated += emit_count;
            if !finished {
                let last = *to_emit.last().unwrap_or(&committed_tok);
                lane.last_tok = last;
                lane.pos = main_pos + nacc + 1;
            } else {
                lane.last_tok = *to_emit.last().unwrap_or(&committed_tok);
                lane.pos = main_pos + emit_count;
            }
        }

        self.rec_step(SpecStepRec {
            greedy: false, pos: main_pos as u32, drafts: drafts.len() as u32, nacc: nacc as u32,
            emitted: emit_count as u32, round_ms, verify_ms,
            step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
        });
        self.dspark_stat_steps += 1;
        self.dspark_stat_drafts += drafts.len() as u64;
        self.dspark_stat_accepted += nacc as u64;
        self.dspark_stat_emitted += emit_count as u64;
        finished
    }

    /// One DFlash2 speculative step for a SAMPLING lane (temperature > 0): the SAME 7 greedy
    /// drafts from the round, verified with `verify_forward_sample` under qprobs = 1.0 (S3T2 b1 —
    /// the engine's existing sampled-lane pattern; the selector-temperature lever is CLOSED, so no
    /// sampled-selector path is built), accepted by speculative rejection sampling (Leviathan et
    /// al. 2023: accept with prob min(1, p(x)/q(x)) = p(x) since q = 1; else emit the residual).
    /// Distribution-exact by construction; gated by the DFlash2 chi-square probe.
    fn df2_lane_step_sample(&mut self, i: usize) -> bool {
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let snapshot = self.mtp_snapshot_slot;
        let kv_stride = self.kv_stride;

        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
        let generated = self.lanes[i].as_ref().unwrap().generated;
        let max_new = self.lanes[i].as_ref().unwrap().max_new;
        let eos = self.eos.clone();
        // Phase-8: the stop rule is a property of the REQUEST, not the serving path —
        // every emit loop must honor ignore_eos/min_new exactly like the batched decode path
        // (batch.rs' `eos_hit`), or spec-on and spec-off streams stop at different tokens.
        let (ignore_eos, min_new) = { let l = self.lanes[i].as_ref().unwrap(); (l.ignore_eos, l.min_new) };
        let temperature = self.lanes[i].as_ref().unwrap().temperature;
        let top_k = self.lanes[i].as_ref().unwrap().top_k;
        let top_p = self.lanes[i].as_ref().unwrap().top_p;
        let (rep_pen, presence_pen, freq_pen, has_penalty) = {
            let l = self.lanes[i].as_ref().unwrap();
            (l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.has_penalty())
        };
        let history: Vec<u32> = self.lanes[i].as_ref().unwrap().history.clone();
        // All RNG for this step derives from one key (device column seeds + host accept draws,
        // domain-separated); the lane key advances exactly once per step.
        let step_key = self.lanes[i].as_ref().unwrap().seed;

        // ---- Draft: the S4F round (same as the greedy lane). ----
        let step_t0 = std::time::Instant::now();
        let dump_on = self.cov_trace;   // op-sequence gate: must match across TP ranks
        let (drafts, full_out): (Vec<u32>, Option<crate::dflash2::round::Df2RoundOut>) = {
            let df2 = self.df2.as_mut().unwrap();
            assert_eq!(df2.nprev(), main_pos,
                "df2 ring nprev {} != lane pos {} (ring stale or unprimed)", df2.nprev(), main_pos);
            if dump_on {
                df2.refresh_block_pos().expect("df2 refresh_block_pos");
                let o = df2.draft_round_full(committed_tok).expect("df2 draft_round_full");
                (o.tokens.clone(), Some(o))
            } else if df2.round_graph.is_some() {
                (df2.draft_round_graph(committed_tok).expect("df2 draft_round_graph"), None)
            } else {
                df2.refresh_block_pos().expect("df2 refresh_block_pos");
                (df2.draft_round_dev(committed_tok).expect("df2 draft_round"), None)
            }
        };
        let round_ms = step_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Build verify input + per-column device seeds (domain-separated). ----
        let mut verify_input = vec![committed_tok];
        verify_input.extend(drafts.iter().copied());
        let verify_seeds: Vec<u32> =
            (0..crate::dflash2::block()).map(|j| rng_u32(step_key, RNG_DOM_VERIFY, j)).collect();
        let qprobs: Vec<f32> = vec![1.0; crate::dflash2::levels()];   // greedy draft = point mass (S3T2 b1)

        // ---- Verify with stochastic output (the eager spec_verify_b path). ----
        let penalty = self.make_penalty(&history, rep_pen, presence_pen, freq_pen, has_penalty);
        let verify_t0 = std::time::Instant::now();
        let mut t20: Vec<u64> = Vec::new();
        let (vsample, vout) = self.gpu.verify_forward_sample(
            &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos,
            Some(snapshot), penalty,
            &drafts, &qprobs, temperature, top_k, top_p, &verify_seeds,
            if self.cov_trace { Some(&mut t20) } else { None });
        let verify_ms = verify_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Speculative rejection sampling accept loop (q = 1 ⇒ ratio = p(x)). ----
        let mut nacc = 0usize;
        let mut emitted: Vec<u32> = Vec::with_capacity(8);
        let mut rejected = false;
        for j in 0..drafts.len() {
            let ru = rng_uniform(step_key, RNG_DOM_ACCEPT, j);
            if ru < vsample.p_of_draft[j] {
                emitted.push(drafts[j]);
                nacc += 1;
            } else {
                emitted.push(vsample.resid_tok[j]);
                rejected = true;
                break;
            }
        }
        if !rejected { emitted.push(vsample.bonus_tok); }

        // ---- GDN rollback on partial reject. ----
        if nacc + 1 != crate::dflash2::block() {
            self.gpu.copy_gdn_slot(&self.state, snapshot + nacc, phys);
        }
        // ---- Inject the accepted span's taps (same as the greedy lane). ----
        {
            let df2 = self.df2.as_mut().unwrap();
            // S5F3 fix: copy the sink's LIVE staging into the round before the inject (the
            // attach-time deep copy never saw the trunk's captures — the draft-parity root
            // cause). The verify synced the capture before returning.
            df2.sync_staging_from_sink().expect("df2 sync staging");
            df2.inject_dev(nacc + 1, None).expect("df2 inject_dev");
        }
        self.pool.release_bf16(vout, h * 8);


        // ---- Emit accepted tokens + replacement/bonus, honoring EOS and max_new. ----
        let mut hit_eos = false;
        let mut to_emit: Vec<u32> = Vec::with_capacity(emitted.len());
        for &t in &emitted {
            to_emit.push(t);
            if eos.contains(&t) && !ignore_eos && generated + to_emit.len() >= min_new { hit_eos = true; break; }
            if generated + to_emit.len() >= max_new { break; }
        }
        let emit_count = to_emit.len();
        let finished = hit_eos || generated + emit_count >= max_new;

        {
            let cache = &mut self.slot_cache[phys];
            cache.push(committed_tok);
            cache.extend_from_slice(&drafts[..nacc]);
        }
        {
            let lane = self.lanes[i].as_mut().unwrap();
            for &t in &to_emit {
                let _ = lane.tx.send(TokEvent::Tok(t));
                lane.history.push(t);
                if lane.history.len() > 256 { lane.history.drain(0..128); }
            }
            lane.generated += emit_count;
            lane.seed = splitmix64(step_key);
            lane.last_tok = *to_emit.last().unwrap_or(&committed_tok);
            lane.pos = if !finished { main_pos + nacc + 1 } else { main_pos + emit_count };
        }

        // ---- S5F3 dump record (dump-only; the staging + t20 readbacks are stream-ordered
        // ---- after the verify — it synced before returning). ----
        if let Some(d) = self.step_dump.as_mut() {
            let staging: Vec<half::bf16> = self.df2_sink.as_ref()
                .and_then(|s| self.gpu.dev().dtoh_sync_copy(&s.staging).ok())
                .unwrap_or_default();
            d.record_span(main_pos, 8, &staging);
            let ck = crate::dflash2::stepdump::StepDump::tap_checksums(&staging);
            if self.df2_stat_steps == 0 {
                if let Some(df2) = self.df2.as_mut() {
                    if let Ok(rs) = df2.dump_staging() {
                        let sink_f: Vec<f32> = staging.iter().map(|x| x.to_f32()).collect();
                        let mut nd = 0.0f64; let mut dd = 0.0f64;
                        for i in 0..rs.len().min(sink_f.len()) {
                            let a = rs[i] as f64; let b = sink_f[i] as f64;
                            nd += (a - b) * (a - b); dd += b * b;
                        }
                        eprintln!("[df2-dump] STAGING CHECK step0 (sample lane): round-vs-sink relL2 {:.4e} (round[0..4]={:?} sink[0..4]={:?})",
                                 (nd / dd.max(1e-30)).sqrt(), &rs[..4], &sink_f[..4]);
                    }
                }
            }
            let rec = crate::dflash2::stepdump::Df2StepRec {
                step: self.df2_stat_steps, pos: main_pos, committed: committed_tok, greedy: false, realq: false,
                drafts: drafts.clone(), p_draft: vsample.p_of_draft.clone(),
                resid: vsample.resid_tok.clone(), bonus: vsample.bonus_tok,
                nacc, emitted: emit_count, q_rows: vec![1.0; crate::dflash2::levels()],
                candidates: full_out.as_ref().map(|o| o.candidates.clone()).unwrap_or_default(),
                cand_q: Vec::new(),
                unary: full_out.as_ref().map(|o| o.unary.clone()).unwrap_or_default(),
                scores: full_out.as_ref().map(|o| o.scores.clone()).unwrap_or_default(),
                top20: std::mem::take(&mut t20), tap_ck: ck,
                hfinal_written: full_out.is_some() && self.df2_stat_steps
                    < crate::dflash2::stepdump::RAW_STEPS as u64,
                // PLAN/25 Phase 0 fields: the greedy-lane coverage readback and the logging-only
                // MTP chain don't run on the sampled lanes (the sample lane's distribution-side
                // table is `top20`; the temp-0.7 union uses the per-source marginals).
                tgt_ids: Vec::new(), tgt_logit: Vec::new(), tgt_p1: Vec::new(),
                tgt_margin: Vec::new(), mtp_ids: Vec::new(), mtp_logit: Vec::new(),
            };
            let hf = full_out.as_ref().map(|o| o.h_final.as_slice());
            d.record_df2(&rec, hf);
            // S5F3 ring rows (first RING_STEPS steps): ctx rows near C + the injected span.
            if self.df2_stat_steps < crate::dflash2::stepdump::RING_STEPS as u64 {
                let df2 = self.df2.as_mut().unwrap();
                let lo = main_pos.saturating_sub(2);
                let mut rk = Vec::new();
                let mut rv = Vec::new();
                for li in 0..crate::dflash2::N_LAYERS {
                    if let Ok((k, v)) = df2.dump_ring_rows(li, lo, main_pos + 8) {
                        rk.push(k); rv.push(v);
                    }
                }
                if rk.len() == crate::dflash2::N_LAYERS { d.record_ring_rows(self.df2_stat_steps, &rk, &rv); }
            }
        }

        self.rec_step(SpecStepRec {
            greedy: false, pos: main_pos as u32, drafts: drafts.len() as u32, nacc: nacc as u32,
            emitted: emit_count as u32, round_ms, verify_ms,
            step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
        });
        // DF2_CARRY: bind the ring to this slot's frontier (twin of the greedy lane's write).
        if let Some(d) = self.df2.as_mut() { d.note_ring(phys); }
        self.df2_stat_steps += 1;
        self.df2_stat_drafts += drafts.len() as u64;
        self.df2_stat_accepted += nacc as u64;
        self.df2_stat_emitted += emit_count as u64;
        // P2 Phase A finding: the sampled-greedy lane (temp>0 + Code) had NO [df2-step] line —
        // the step tables silently dropped every code-class step. Same format as the greedy lane.
        if crate::opts::var(crate::opt!("df2-step-log")).is_ok() {
            eprintln!("[df2-step] pos={main_pos} nacc={nacc} emitted={emit_count} committed={committed_tok} \
                       round={round_ms:.1}ms verify={verify_ms:.1}ms step={:.1}ms",
                      step_t0.elapsed().as_secs_f32() * 1e3);
        }
        finished
    }

    /// S5F2 L2 — the REAL-Q DFlash2 lane (--spec-source dflash2-rq): the sampled selector path
    /// (`Df2Round::draft_round_dev_sample` — the SGLang `CandidateSelector.sample_path`
    /// multinomial at the request temperature) verified under the real-q rejection-sampling
    /// criterion `u·q < p` (q = the drawn candidate's selector probability) with the EXACT
    /// relu(p−q) residual (the SGLang `speculative_sampling_classic_kernel` semantics). With a
    /// valid proposal q and the exact residual the emitted distribution is exactly the target p
    /// — the L2 chi-square gate's contract. Distribution-exact by construction; gated by
    /// `--bench-df2-sample-realq`.
    fn df2_lane_step_sample_rq(&mut self, i: usize) -> bool {
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let snapshot = self.mtp_snapshot_slot;
        let kv_stride = self.kv_stride;

        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
        let generated = self.lanes[i].as_ref().unwrap().generated;
        let max_new = self.lanes[i].as_ref().unwrap().max_new;
        let eos = self.eos.clone();
        // Phase-8: the stop rule is a property of the REQUEST, not the serving path —
        // every emit loop must honor ignore_eos/min_new exactly like the batched decode path
        // (batch.rs' `eos_hit`), or spec-on and spec-off streams stop at different tokens.
        let (ignore_eos, min_new) = { let l = self.lanes[i].as_ref().unwrap(); (l.ignore_eos, l.min_new) };
        let temperature = self.lanes[i].as_ref().unwrap().temperature;
        let top_k = self.lanes[i].as_ref().unwrap().top_k;
        let top_p = self.lanes[i].as_ref().unwrap().top_p;
        let (rep_pen, presence_pen, freq_pen, has_penalty) = {
            let l = self.lanes[i].as_ref().unwrap();
            (l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.has_penalty())
        };
        let history: Vec<u32> = self.lanes[i].as_ref().unwrap().history.clone();
        let step_key = self.lanes[i].as_ref().unwrap().seed;

        // ---- Draft: the SAMPLED selector path (eager round; per-position selector seeds). ----
        let step_t0 = std::time::Instant::now();
        let dump_on = self.cov_trace;   // op-sequence gate: must match across TP ranks
        let sel_seeds: Vec<u32> = (0..crate::dflash2::levels()).map(|j| rng_u32(step_key, RNG_DOM_DF2_SEL, j)).collect();
        let (drafts, q_rows, cand_tok, cand_q) = {
            let df2 = self.df2.as_mut().unwrap();
            assert_eq!(df2.nprev(), main_pos,
                "df2 ring nprev {} != lane pos {} (ring stale or unprimed)", df2.nprev(), main_pos);
            // S5F4 fix (the rq-lane sibling of R14): refresh the block RoPE positions for THIS
            // step. The greedy/sample lanes do this before every round; the rq lane missed it —
            // every round ran with the STALE pos_blk/cos8/sin8 left by capture_round_graph
            // (max_c..max_c+8), scrambling the block attention against the correctly-rotated
            // ctx rows (step-0 top16 differed from the greedy lane's on the SAME ring/anchor),
            // flattening the sampled chain's realized q (0.47 vs SGLang 0.80) and capping τ at
            // ~1.23. The chi-square gate stayed green because ITS path refreshes (main.rs
            // run_bench_df2_sample_realq) — the gate and the serving lane diverged.
            df2.refresh_block_pos().expect("df2 refresh_block_pos");
            let out = df2.draft_round_dev_sample(committed_tok, &sel_seeds, temperature)
                .expect("df2 sampled draft round");
            (out.tokens, out.q_rows, out.cand_tok, out.cand_q)
        };
        let round_ms = step_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Build verify input + per-column device seeds. ----
        let mut verify_input = vec![committed_tok];
        verify_input.extend(drafts.iter().copied());
        let verify_seeds: Vec<u32> =
            (0..crate::dflash2::block()).map(|j| rng_u32(step_key, RNG_DOM_VERIFY, j)).collect();

        // ---- Verify with the real-q kernel (exact relu(p-q) residual). ----
        let penalty = self.make_penalty(&history, rep_pen, presence_pen, freq_pen, has_penalty);
        let verify_t0 = std::time::Instant::now();
        let mut t20: Vec<u64> = Vec::new();
        let (vsample, vout) = self.gpu.verify_forward_sample_rq(
            &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos,
            Some(snapshot), penalty,
            &drafts, &cand_tok, &cand_q, temperature, top_k, top_p, &verify_seeds,
            if self.cov_trace { Some(&mut t20) } else { None });
        let verify_ms = verify_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Real-q rejection sampling: accept iff u*q < p (min(1, p/q) ratio). ----
        let mut nacc = 0usize;
        let mut emitted: Vec<u32> = Vec::with_capacity(8);
        let mut rejected = false;
        let eps = 1e-12f32;
        for j in 0..drafts.len() {
            let q = q_rows[j];
            let ratio = if q < eps { 1.0 } else { (vsample.p_of_draft[j] / q).min(1.0) };
            let ru = rng_uniform(step_key, RNG_DOM_ACCEPT, j);
            if ru < ratio {
                emitted.push(drafts[j]);
                nacc += 1;
            } else {
                emitted.push(vsample.resid_tok[j]);
                rejected = true;
                break;
            }
        }
        if !rejected { emitted.push(vsample.bonus_tok); }

        // ---- GDN rollback on partial reject. ----
        if nacc + 1 != crate::dflash2::block() {
            self.gpu.copy_gdn_slot(&self.state, snapshot + nacc, phys);
        }
        // ---- Inject the accepted span's taps (same as the other lanes). ----
        {
            let df2 = self.df2.as_mut().unwrap();
            // S5F3 fix: copy the sink's LIVE staging into the round before the inject (the
            // attach-time deep copy never saw the trunk's captures — the draft-parity root
            // cause). The verify synced the capture before returning.
            df2.sync_staging_from_sink().expect("df2 sync staging");
            df2.inject_dev(nacc + 1, None).expect("df2 inject_dev");
        }
        self.pool.release_bf16(vout, h * 8);


        // ---- Emit accepted tokens + replacement/bonus, honoring EOS and max_new. ----
        let mut hit_eos = false;
        let mut to_emit: Vec<u32> = Vec::with_capacity(emitted.len());
        for &t in &emitted {
            to_emit.push(t);
            if eos.contains(&t) && !ignore_eos && generated + to_emit.len() >= min_new { hit_eos = true; break; }
            if generated + to_emit.len() >= max_new { break; }
        }
        let emit_count = to_emit.len();
        let finished = hit_eos || generated + emit_count >= max_new;

        {
            let cache = &mut self.slot_cache[phys];
            cache.push(committed_tok);
            cache.extend_from_slice(&drafts[..nacc]);
        }
        {
            let lane = self.lanes[i].as_mut().unwrap();
            for &t in &to_emit {
                let _ = lane.tx.send(TokEvent::Tok(t));
                lane.history.push(t);
                if lane.history.len() > 256 { lane.history.drain(0..128); }
            }
            lane.generated += emit_count;
            lane.seed = splitmix64(step_key);
            lane.last_tok = *to_emit.last().unwrap_or(&committed_tok);
            lane.pos = if !finished { main_pos + nacc + 1 } else { main_pos + emit_count };
        }

        // ---- S5F3 dump record (dump-only; staging + t20 readbacks stream-ordered after the
        // ---- verify — it synced before returning). The SAMPLED selector's q_rows + candidate
        // ---- table are the exact SGLang `sample_path` analog. ----
        if let Some(d) = self.step_dump.as_mut() {
            let staging: Vec<half::bf16> = self.df2_sink.as_ref()
                .and_then(|s| self.gpu.dev().dtoh_sync_copy(&s.staging).ok())
                .unwrap_or_default();
            d.record_span(main_pos, 8, &staging);
            let ck = crate::dflash2::stepdump::StepDump::tap_checksums(&staging);
            let rec = crate::dflash2::stepdump::Df2StepRec {
                step: self.df2_stat_steps, pos: main_pos, committed: committed_tok, greedy: false, realq: true,
                drafts: drafts.clone(), p_draft: vsample.p_of_draft.clone(),
                resid: vsample.resid_tok.clone(), bonus: vsample.bonus_tok,
                nacc, emitted: emit_count, q_rows,
                candidates: cand_tok, cand_q, unary: Vec::new(), scores: Vec::new(),
                top20: std::mem::take(&mut t20), tap_ck: ck, hfinal_written: false,
                // PLAN/25 Phase 0 fields (see the greedy-lane record): not captured on rq lanes.
                tgt_ids: Vec::new(), tgt_logit: Vec::new(), tgt_p1: Vec::new(),
                tgt_margin: Vec::new(), mtp_ids: Vec::new(), mtp_logit: Vec::new(),
            };
            d.record_df2(&rec, None);
            // S5F3 ring rows (first RING_STEPS steps).
            if self.df2_stat_steps < crate::dflash2::stepdump::RING_STEPS as u64 {
                let df2 = self.df2.as_mut().unwrap();
                let lo = main_pos.saturating_sub(2);
                let mut rk = Vec::new();
                let mut rv = Vec::new();
                for li in 0..crate::dflash2::N_LAYERS {
                    if let Ok((k, v)) = df2.dump_ring_rows(li, lo, main_pos + 8) {
                        rk.push(k); rv.push(v);
                    }
                }
                if rk.len() == crate::dflash2::N_LAYERS { d.record_ring_rows(self.df2_stat_steps, &rk, &rv); }
            }
        }

        self.rec_step(SpecStepRec {
            greedy: false, pos: main_pos as u32, drafts: drafts.len() as u32, nacc: nacc as u32,
            emitted: emit_count as u32, round_ms, verify_ms,
            step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
        });
        // DF2_CARRY: bind the ring to this slot's frontier (twin of the greedy lane's write).
        if let Some(d) = self.df2.as_mut() { d.note_ring(phys); }
        self.df2_stat_steps += 1;
        self.df2_stat_drafts += drafts.len() as u64;
        self.df2_stat_accepted += nacc as u64;
        self.df2_stat_emitted += emit_count as u64;
        if crate::opts::var(crate::opt!("df2-step-log")).is_ok() {
            eprintln!("[df2-step] pos={main_pos} nacc={nacc} emitted={emit_count} committed={committed_tok} \
                       round={round_ms:.1}ms verify={verify_ms:.1}ms step={:.1}ms",
                      step_t0.elapsed().as_secs_f32() * 1e3);
        }
        finished
    }

    /// One batched decode step over a subset of lanes (`batch_idx`). Builds the per-lane input
    /// arrays (tokens/positions/penalties/slot map), uploads them, runs the appropriate forward path
    /// (greedy graph / GPU-sample graph / CPU-sample / non-graph greedy), and returns the next token
    /// per lane (length = batch_idx.len()).
    fn batched_decode(&mut self, batch_idx: &[usize]) -> Vec<u32> {
        let s = batch_idx.len();
        let mb = self.max_batch;
        let mp = crate::gpu::MAX_PEN_TOKENS;
        // W2: a schema lane in this batch disables the resident loop AND the captured graphs —
        // both bake their kernel sequence (and the mask kernel's operands) at capture/admission
        // time, while the mask content changes every step. Schema batches take the eager core.
        let any_schema = batch_idx.iter().any(|&i| self.lanes[i].as_ref().unwrap().schema.is_some());
        let resident = crate::gpu::GpuModel::device_loop_on() && !any_schema;
        let any_sampling = batch_idx.iter().any(|&i| !self.lanes[i].as_ref().unwrap().greedy);
        let can_graph = !any_sampling && !any_schema && self.graphs.contains_key(&s);
        let max_pc = batch_idx.iter()
            .map(|&i| self.lanes[i].as_ref().unwrap().pos + 1).max().unwrap_or(1);

        if resident {
            // ---- DEVICE-RESIDENT TOKEN LOOP (EXPERT_DEVICE_ARGMAX_LOOP_RESPONSE §7.3) ----
            // Clean steps upload NOTHING: tokens/pos/slot_ids/ring/keys are device-resident and
            // current (the graph head's ids_advance_b / epilogue ring push advanced them last
            // step). Dirty steps (admission/finish/compaction/MTP/param change — any lane
            // composition change) re-upload everything below.
            if self.resident_dirty {
                let mut toks = vec![0i32; mb];
                let mut pos = vec![0i32; mb];
                let mut slot_ids: Vec<i32> = (0..s)
                    .map(|k| self.lanes[batch_idx[k]].as_ref().unwrap().phys as i32).collect();
                slot_ids.resize(mb, 0);
                let mut ring = vec![-1i32; mp * mb];
                let mut ring_state = vec![0i32; mb];
                let mut keys = vec![0u64; mb];
                let mut rep_pen = vec![1.0f32; mb];
                let mut presence_pen = vec![0.0f32; mb];
                let mut frequency_pen = vec![0.0f32; mb];
                for (k, &i) in batch_idx.iter().enumerate() {
                    let lane = self.lanes[i].as_ref().unwrap();
                    // The graph head's ids_advance_b copies token_ids_dev -> tokens_dev and
                    // INCREMENTS pos_dev BEFORE the forward reads it, so the uploads are the
                    // PRE-advance values: token_ids_dev = last_tok (the embed input) and
                    // pos = lane.pos - 1 (the forward consumes lane.pos).
                    toks[k] = lane.last_tok as i32;
                    pos[k] = lane.pos as i32 - 1;
                    // Per-lane penalty VALUES are request constants — upload on every dirty
                    // event (they were silently never uploaded on the resident path: the
                    // window kernel only rebuilds the token/count arrays, not these).
                    rep_pen[k] = lane.rep_penalty;
                    presence_pen[k] = lane.presence_penalty;
                    frequency_pen[k] = lane.frequency_penalty;
                    // Ring rebuild from lane.history, MRU-first: entry j (0 = most recent) lands at
                    // ring[head-1-j], head = len % mp — the exact layout penalty_window_b reads.
                    let hist: Vec<i32> = lane.history.iter().rev().take(mp).map(|&t| t as i32).collect();
                    let len = hist.len();
                    for (j, &t) in hist.iter().enumerate() {
                        ring[k * mp + ((len + mp - 1 - j) % mp)] = t;
                    }
                    ring_state[k] = (((len % mp) << 8) | len) as i32;
                    keys[k] = lane.seed;
                }
                self.gpu.dev().htod_sync_copy_into(&toks, &mut self.bufs.token_ids_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&pos, &mut self.bufs.pos_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&slot_ids, &mut self.bufs.slot_ids_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&ring, &mut self.bufs.pen_ring_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&ring_state, &mut self.bufs.pen_ring_state_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&rep_pen, &mut self.bufs.rep_pen_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&presence_pen, &mut self.bufs.presence_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&frequency_pen, &mut self.bufs.frequency_dev).unwrap();
                if any_sampling {
                    // Sampling params are per-request constants; upload them with the keys (the
                    // draws themselves are produced on-device by seed_advance_b from the keys —
                    // seeds_dev is NEVER uploaded under the resident loop).
                    let mut t: Vec<f32> = batch_idx.iter()
                        .map(|&i| self.lanes[i].as_ref().unwrap().temperature).collect();
                    t.resize(mb, 1.0);
                    let mut ki: Vec<i32> = batch_idx.iter()
                        .map(|&i| self.lanes[i].as_ref().unwrap().top_k as i32).collect();
                    ki.resize(mb, 1);
                    let mut p: Vec<f32> = batch_idx.iter()
                        .map(|&i| self.lanes[i].as_ref().unwrap().top_p).collect();
                    p.resize(mb, 1.0);
                    self.gpu.dev().htod_sync_copy_into(&t, &mut self.bufs.temps_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&ki, &mut self.bufs.topk_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&p, &mut self.bufs.topp_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&keys, &mut self.bufs.seeds_key_dev).unwrap();
                }
                self.resident_dirty = false;
            }
            // No dev().synchronize(): host-blocking NULL-stream copies + the blocking compute
            // stream already order these before the kernels/graph replay that read them (I1).

            // Dispatch — every path below runs the resident kernels (captured in the graphs or
            // composed in the eager core) when the flag is on.
            let can_graph = !any_sampling && self.graphs.contains_key(&s);
            let toks = if can_graph {
                let graph = self.graphs.get(&s).unwrap();
                self.gpu.replay_decode(&self.bufs, graph, s)
            } else if any_sampling {
                let temps: Vec<f32> = batch_idx.iter().map(|&i| self.lanes[i].as_ref().unwrap().temperature).collect();
                let tks: Vec<usize> = batch_idx.iter().map(|&i| self.lanes[i].as_ref().unwrap().top_k).collect();
                let tps: Vec<f32> = batch_idx.iter().map(|&i| self.lanes[i].as_ref().unwrap().top_p).collect();
                if self.gpu_sample {
                    match self.sample_graphs.get(&s) {
                        Some(g) => self.gpu.replay_decode_sample(&self.bufs, g, s),
                        None => self.gpu.forward_decode_sample_gpu(
                            &mut self.pool, &mut self.bufs, &mut self.state, self.kv_stride, max_pc, s),
                    }
                } else {
                    // CPU-sampling escape (--cpu-sample): the host needs the full
                    // logits, so the resident loop is pointless AND its pos semantics differ (no
                    // ids_advance) — run today's sequence with today's uploads, then force the
                    // next step to re-upload (the pos_dev/tokens_dev left behind are in
                    // non-resident semantics).
                    let mut t = temps.clone(); t.resize(mb, 1.0);
                    let mut ki: Vec<i32> = tks.iter().map(|&x| x as i32).collect(); ki.resize(mb, 1);
                    let mut p = tps.clone(); p.resize(mb, 1.0);
                    let mut toks_v = vec![0i32; mb];
                    let mut pos_v = vec![0i32; mb];
                    for (k, &i) in batch_idx.iter().enumerate() {
                        let lane = self.lanes[i].as_ref().unwrap();
                        toks_v[k] = lane.last_tok as i32;
                        pos_v[k] = lane.pos as i32;
                    }
                    self.gpu.dev().htod_sync_copy_into(&t, &mut self.bufs.temps_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&ki, &mut self.bufs.topk_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&p, &mut self.bufs.topp_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&toks_v, &mut self.bufs.tokens_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&pos_v, &mut self.bufs.pos_dev).unwrap();
                    self.resident_dirty = true;
                    self.gpu.forward_decode_sample(
                        &mut self.pool, &mut self.bufs, &mut self.state, self.kv_stride, max_pc, s,
                        &temps, &tks, &tps)
                }
            } else {
                self.gpu.forward_decode(
                    &mut self.pool, &mut self.bufs, &mut self.state, self.kv_stride, max_pc, s)
            };
            if any_sampling && self.gpu_sample {
                // Host key bookkeeping: the device (sample graph / resident core) advanced the
                // keys exactly once this step; keep the host keys in lockstep so a dirty
                // re-upload never replays stale draws. (The CPU-sample escape advances nothing
                // on device and leaves the keys untouched — the forced dirty flag makes the
                // next step re-upload them, still in sync.)
                for k in 0..s {
                    let lane = self.lanes[batch_idx[k]].as_mut().unwrap();
                    lane.seed = splitmix64(lane.seed);
                }
            }
            toks
        } else {
            // ---- today's path, VERBATIM (uploads + dispatch) ----
            let mut toks = vec![0i32; mb];
            let mut pos = vec![0i32; mb];
            let mut pen_tokens = vec![-1i32; mp * mb];
            let mut pen_counts = vec![0i16; mp * mb];
            let mut rep_pen = vec![1.0f32; mb];
            let mut presence_pen = vec![0.0f32; mb];
            let mut frequency_pen = vec![0.0f32; mb];
            for (k, &i) in batch_idx.iter().enumerate() {
                let lane = self.lanes[i].as_ref().unwrap();
                toks[k] = lane.last_tok as i32;
                pos[k] = lane.pos as i32;
                rep_pen[k] = lane.rep_penalty;
                presence_pen[k] = lane.presence_penalty;
                frequency_pen[k] = lane.frequency_penalty;
                // Fill this lane's unique recent tokens (with counts) only if it has any penalty;
                // lanes without penalty leave their slots as -1 sentinels (skipped by the kernel).
                if lane.has_penalty() {
                    let base = k * mp;
                    let mut idx = 0usize;
                    for &t in lane.history.iter().rev().take(mp) {
                        let t_i = t as i32;
                        let found = (0..idx).position(|j| pen_tokens[base + j] == t_i);
                        match found {
                            Some(j) => { pen_counts[base + j] += 1; }
                            None => {
                                if idx < mp { pen_tokens[base + idx] = t_i; pen_counts[base + idx] = 1; idx += 1; }
                            }
                        }
                    }
                }
            }
            let mut slot_ids: Vec<i32> = (0..s)
                .map(|k| self.lanes[batch_idx[k]].as_ref().unwrap().phys as i32).collect();
            slot_ids.resize(mb, 0);
            self.gpu.dev().htod_sync_copy_into(&toks, &mut self.bufs.tokens_dev).unwrap();
            self.gpu.dev().htod_sync_copy_into(&pos, &mut self.bufs.pos_dev).unwrap();
            self.gpu.dev().htod_sync_copy_into(&slot_ids, &mut self.bufs.slot_ids_dev).unwrap();
            // The five penalty arrays are read by rep_penalty_b, which skips -1 sentinels — so they
            // only need uploading when some lane is actually penalized, plus ONE clear when the last
            // penalized lane departs (so its values cannot linger into an unpenalized successor).
            let any_pen = batch_idx.iter().any(|&i| self.lanes[i].as_ref().unwrap().has_penalty());
            if any_pen || self.pen_had {
                self.gpu.dev().htod_sync_copy_into(&pen_tokens, &mut self.bufs.penalty_tokens_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&pen_counts, &mut self.bufs.penalty_counts_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&rep_pen, &mut self.bufs.rep_pen_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&presence_pen, &mut self.bufs.presence_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&frequency_pen, &mut self.bufs.frequency_dev).unwrap();
                self.pen_had = any_pen;
            }
            // No dev().synchronize(): host-blocking NULL-stream copies + the blocking compute stream
            // already order these before the kernels/graph replay that read them (invariant I1).

            let toks = if can_graph {
                let graph = self.graphs.get(&s).unwrap();
                self.gpu.replay_decode(&self.bufs, graph, s)
            } else if any_sampling {
                let temps: Vec<f32> = batch_idx.iter().map(|&i| self.lanes[i].as_ref().unwrap().temperature).collect();
                let tks: Vec<usize> = batch_idx.iter().map(|&i| self.lanes[i].as_ref().unwrap().top_k).collect();
                let tps: Vec<f32> = batch_idx.iter().map(|&i| self.lanes[i].as_ref().unwrap().top_p).collect();
                if self.gpu_sample {
                    // htod sampling params + fresh seeds into bufs (NULL stream), then dispatch to the
                    // captured decode+sample graph when available, else the non-graph core.
                    let mut t = temps.clone(); t.resize(mb, 1.0);
                    let mut ki: Vec<i32> = tks.iter().map(|&x| x as i32).collect(); ki.resize(mb, 1);
                    let mut p = tps.clone(); p.resize(mb, 1.0);
                    // Seed sample_b from each lane's own PRNG, not rand::random(). The lane already
                    // carries a seed (from the request's `seed` field), and the MTP path honours it —
                    // drawing a fresh OS-random seed here meant an explicit `"seed": 42` was silently
                    // ignored on the plain-sampler path, so identical requests were irreproducible.
                    // Advance each lane's key once per decode step, exactly as the MTP path does.
                    let mut sd: Vec<u32> = Vec::with_capacity(mb);
                    for k in 0..s {
                        let lane = self.lanes[batch_idx[k]].as_mut().unwrap();
                        sd.push(rng_u32(lane.seed, RNG_DOM_SAMPLE, 0));
                        lane.seed = splitmix64(lane.seed);
                    }
                    sd.resize(mb, 0);
                    self.gpu.dev().htod_sync_copy_into(&t, &mut self.bufs.temps_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&ki, &mut self.bufs.topk_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&p, &mut self.bufs.topp_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&sd, &mut self.bufs.seeds_dev).unwrap();
                    // No dev().synchronize(): host-blocking NULL-stream copies + the blocking compute
                    // stream already order these before the sample kernels/replay (invariant I1).
                    if let Some(g) = self.sample_graphs.get(&s) {
                        self.gpu.replay_decode_sample(&self.bufs, g, s)
                    } else {
                        self.gpu.forward_decode_sample_gpu(
                            &mut self.pool, &mut self.bufs, &mut self.state, self.kv_stride, max_pc, s)
                    }
                } else {
                    self.gpu.forward_decode_sample(
                        &mut self.pool, &mut self.bufs, &mut self.state, self.kv_stride, max_pc, s,
                        &temps, &tks, &tps)
                }
            } else {
                self.gpu.forward_decode(
                    &mut self.pool, &mut self.bufs, &mut self.state, self.kv_stride, max_pc, s)
            };
            toks
        }
    }
}

#[cfg(test)]
mod tree_accept_tests {
    use super::tree_accept_walk;

    // A tree degenerates to a chain (parent[c]=c-1). The walk must reproduce "accept longest prefix".
    #[test]
    fn chain_is_accept_longest_prefix() {
        // committed=100; drafts=[10,20,30]; target preds=[10,20,99,..] => accept 10,20, correct 30->99.
        let parent = [-1, 0, 1, 2];
        let tokens = [100u32, 10, 20, 30];
        let preds  = [10u32, 20, 99, 7];         // preds[2]=99 != tokens[3]=30 -> stop, bonus 99
        let (path, emitted) = tree_accept_walk(&parent, &tokens, &preds);
        assert_eq!(path, vec![0, 1, 2]);          // accepted committed, 10, 20
        assert_eq!(emitted, vec![10, 20, 99]);    // 10, 20 accepted, 99 the correction
    }

    // Fork at position 1: child A (col 1) wrong, child B (col 2) right -> B rescued, then chains.
    #[test]
    fn fork_rescues_second_branch() {
        //        0(committed=100)
        //       / \
        //   1(A=10) 2(B=20)---3(B2=30)
        let parent = [-1, 0, 0, 2];
        let tokens = [100u32, 10, 20, 30];
        let preds  = [20u32, 5, 30, 88];  // after committed target wants 20 (=B), then 30 (=B2), then 88
        let (path, emitted) = tree_accept_walk(&parent, &tokens, &preds);
        assert_eq!(path, vec![0, 2, 3]);          // walked root -> B -> B2
        assert_eq!(emitted, vec![20, 30, 88]);    // B, B2 accepted, 88 the bonus
    }

    // Neither child matches at the root -> emit just the correction, accept nothing past committed.
    #[test]
    fn no_child_matches() {
        let parent = [-1, 0, 0];
        let tokens = [100u32, 10, 20];
        let preds  = [77u32, 1, 2];               // target wants 77, neither child is 77
        let (path, emitted) = tree_accept_walk(&parent, &tokens, &preds);
        assert_eq!(path, vec![0]);
        assert_eq!(emitted, vec![77]);
    }

    // Tie: two children carry the target's token -> prefer the lowest index (deterministic).
    #[test]
    fn tie_prefers_lowest_child() {
        let parent = [-1, 0, 0];
        let tokens = [100u32, 42, 42];
        let preds  = [42u32, 9, 9];
        let (path, _) = tree_accept_walk(&parent, &tokens, &preds);
        assert_eq!(path, vec![0, 1]);             // col 1, not col 2
    }
}

#[cfg(test)]
mod mtp_policy_tests {
    use super::{MtpPolicy, MTP_EVAL_FIRST, MTP_EVAL_WINDOW};

    /// An r(d) table shaped like the MEASURED TP=4 serving calibration at 2k ctx
    /// (PLAN/03 §results: r(2/3/4/5) = 1.13/1.26/1.39/1.48, fresh calibration) — the regime
    /// where the d3 sweet spot lives.
    fn r_tp4_2k() -> Vec<(usize, Vec<(usize, f32)>)> {
        vec![(2048, vec![(2, 1.13), (3, 1.26), (4, 1.39), (5, 1.48), (6, 1.6), (8, 1.9)])]
    }

    /// Simulate `steps` MTP steps at the policy's current depth with a fixed hazard curve
    /// `hz` (conditional acceptance per position), then tick. Repeats until `windows`
    /// evaluations have happened. Returns the depth history.
    fn run(policy: &mut MtpPolicy, hz: &[f64], steps_per_window: usize, windows: usize) -> Vec<usize> {
        let mut hist = vec![policy.depth()];
        for _ in 0..windows {
            for _ in 0..steps_per_window {
                // draw the accepted prefix from the hazard curve
                let mut acc = 0usize;
                while acc < policy.depth() - 1 && rand_step(hz.get(acc).copied().unwrap_or(0.0)) { acc += 1; }
                // record_step is private but the test module is a child of this module — visible.
                let emitted = (acc + 1) as u64;
                policy.record_step((policy.depth() - 1) as u64, acc as u64, emitted);
            }
            policy.tick(2048);
            hist.push(policy.depth());
        }
        hist
    }

    /// Deterministic stand-in for a Bernoulli(h) draw: an LCG whose STATE is the mixed word
    /// (the first version forgot to store it back and drew from a linear sequence — correlated
    /// draws that measured 77% acceptance from a 44% hazard and invalidated the fixture).
    fn rand_step(h: f64) -> bool {
        use std::cell::Cell;
        thread_local! { static I: Cell<u64> = Cell::new(0x853c49e6748fea9b); }
        I.with(|i| {
            let s = i.get().wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            i.set(s);
            let x = (s >> 33) as f64 / (1u64 << 31) as f64;
            x < h
        })
    }

    /// PLAN/10 #12 regression: the falling-conditional weak-head shape (the documented hazard
    /// decay — yield_at's own doc: "hazards decay because each draft is conditioned on a chain
    /// of its own guesses") with the MEASURED TP=4 r(d) at 2k. At hz = [.44, .50, .45]:
    ///   yield(3)/r(3) = 1.660/1.26 = 1.317    yield(4)/r(4) = 1.759/1.39 = 1.266
    /// d3 wins by 4.1% — MORE than 0, LESS than the 5% up-margin. The OLD symmetric margin
    /// blocked exactly this switch (the P2 carry-forward: "policy never tried d3"); with the
    /// asymmetric down-margin (1.0) the descent must happen.
    #[test]
    fn low_alpha_descends_to_d3() {
        let mut p = MtpPolicy::new(true, None, None, r_tp4_2k());
        assert_eq!(p.depth(), 4, "auto policy opens at 4");
        let hz = [0.44, 0.50, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45];
        let hist = run(&mut p, &hz, MTP_EVAL_FIRST as usize + 32, 4);
        assert!(hist.contains(&3), "policy never tried d3 at a low-alpha curve; history {hist:?}");
    }

    /// Control: a HIGH-α curve (@1:83%) keeps deep depths — the asymmetry must not collapse
    /// the policy to d2/d3 when deep drafts pay (the 9B prose regime the symmetric margin was
    /// built for: p1≈0.83 predicted 3.94 tok/step at d6 vs 2.64 actual; the hazard model
    /// handles that, the margin must not fight it).
    #[test]
    fn high_alpha_stays_deep() {
        let mut p = MtpPolicy::new(true, None, None, r_tp4_2k());
        let hz = [0.83, 0.85, 0.86, 0.87, 0.88, 0.88, 0.88, 0.88];
        let hist = run(&mut p, &hz, MTP_EVAL_WINDOW as usize + 8, 4);
        assert!(hist.iter().all(|&d| d >= 4), "policy bailed shallow on a high-alpha curve; history {hist:?}");
    }

    /// Up-switches keep the margin: a d3 policy with a curve that favors d5 by only 1% must
    /// NOT switch (that's the flapping the up-margin exists to prevent).
    #[test]
    fn up_switch_keeps_margin() {
        let mut p = MtpPolicy::new(true, None, Some(3), r_tp4_2k());
        // At d3 the marginal positions 3/4 are unobserved (prior 0.5 carried) — the model
        // extrapolates pessimistically, so a d5 that's really only ~1% better on paper never
        // clears the 1.05 up-margin. Pin at 3, feed a flat hazard where deeper is barely better.
        let hz = [0.80, 0.95, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99];
        let hist = run(&mut p, &hz, MTP_EVAL_WINDOW as usize + 8, 4);
        assert!(hist.iter().all(|&d| d == 3), "up-switched without clearing the margin; history {hist:?}");
    }
}

/// Cross-image-contamination: the slot-level prefix reuse must be keyed on image CONTENT, not just
/// token identity. Two images of the same resolution expand to identical `image_pad` runs, so a
/// token-only key would replay image N's spliced visual content for image N+1. These test the pure
/// host-side keying helpers (the GPU splice offset itself is covered by the end-to-end gate).
#[cfg(test)]
mod vision_image_cache_tests {
    use super::{request_image_identities, images_compatible};
    use crate::vision_encoder::ImageSpan;
    use crate::vision_tower::OUT_HIDDEN;

    fn span(start: usize, n: usize) -> ImageSpan { ImageSpan { start, num_tokens: n } }

    /// Build a deterministic concatenated embeds buffer for two images of width OUT_HIDDEN.
    fn two_image_embeds(n0: usize, n1: usize) -> (Vec<f32>, Vec<ImageSpan>) {
        let mut e = Vec::new();
        let mut seed = 0x12345678u64;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) - 0.5
        };
        for _ in 0..n0 * OUT_HIDDEN { e.push(next()); }
        for _ in 0..n1 * OUT_HIDDEN { e.push(next()); }
        (e, vec![span(60, n0), span(60 + n0, n1)])
    }

    #[test]
    fn identities_are_deterministic_and_positioned() {
        let (emb, spans) = two_image_embeds(4, 5);
        let ids = request_image_identities(Some(&emb), &spans);
        assert_eq!(ids.len(), 2, "two images -> two identities");
        assert_eq!(ids[0].start, 60);
        assert_eq!(ids[1].start, 60 + 4);
        let ids2 = request_image_identities(Some(&emb), &spans);
        assert_eq!(ids[0].hash, ids2[0].hash);
        assert_eq!(ids[1].hash, ids2[1].hash);
        // Different content at image 1 -> different hash. Use a perturbation large enough to
        // change the f32 bits (the LCG values are ~2^30, where the f32 ULP is ~256, so a +0.5
        // would round away).
        let mut emb2 = emb.clone();
        emb2[4 * OUT_HIDDEN] = -12345.678;
        let ids3 = request_image_identities(Some(&emb2), &spans);
        assert_eq!(ids3[0].hash, ids[0].hash, "image 0 unchanged");
        assert_ne!(ids3[1].hash, ids[1].hash, "image 1 changed -> hash must differ");
        assert!(request_image_identities(None, &[]).is_empty());
    }

    #[test]
    fn compatibility_detects_a_changed_image_in_the_reused_prefix() {
        let (emb, spans) = two_image_embeds(4, 5);
        let img_old = request_image_identities(Some(&emb), &spans);
        assert!(images_compatible(&img_old, &img_old, 80), "identical image key must match");
        assert!(images_compatible(&img_old, &[], 80), "text-only request may reuse any prefix");
        // A request with a DIFFERENT image at position 60 (inside the reused prefix) must NOT reuse.
        let mut emb_c = emb.clone();
        emb_c[5] = -12345.678;
        let img_new = request_image_identities(Some(&emb_c), &spans);
        assert!(!images_compatible(&img_old, &img_new, 80), "changed image in the reused prefix must break reuse");
        // An image wholly in the SUFFIX (start >= l) imposes no constraint.
        let (_, spans_suffix) = (emb_c.clone(), vec![span(70, 4), span(80, 5)]);
        let img_sfx = request_image_identities(Some(&emb_c), &spans_suffix);
        assert!(images_compatible(&img_old, &img_sfx, 60), "image wholly in the suffix imposes no constraint");
    }
}

/// DF2_CARRY guard tests (PLAN/DF2_CARRY_SPEC.md §5/§7). These are GPU-free: they pin the
/// decision arithmetic, including the two failure modes that would otherwise be invisible
/// (a stale ring from another slot, and a frontier past the new prompt — both of which keep
/// every losslessness gate green while tau silently falls).
#[cfg(test)]
mod df2_carry {
    use super::df2_carry_ok;

    const ON: bool = true;
    const DF2: bool = true;
    const SLOT_A: usize = 0;
    const SLOT_B: usize = 1;
    /// The real guard (frontier term active). `len_only` is the diagnostics instrument.
    const FULL: bool = false;

    /// The natural multi-turn flow: turn 1 ran to frontier N = 2240; turn 2's prompt is that
    /// whole sequence plus the client's next message (plen = 2440).
    #[test]
    fn multi_turn_carries() {
        assert!(df2_carry_ok(ON, DF2, 2048, 2440, SLOT_A, Some(SLOT_A), 2240, FULL));
    }

    /// Turn 2's prompt is SHORTER than turn 1's frontier: the rows the draft needs are exactly
    /// the ones the generation has already overwritten. Must refuse. THIS is the case the
    /// frontier term exists for -- and the case `len_only` wrongly allows.
    #[test]
    fn frontier_past_prompt_refused() {
        assert!(!df2_carry_ok(ON, DF2, 2048, 2200, SLOT_A, Some(SLOT_A), 2240, FULL));
    }

    /// Negative-control instrument check (SPEC §7 gate 7): with the frontier term dropped the
    /// SAME state is accepted, which is what lets the gate force a known-bad carry.
    #[test]
    fn len_only_instrument_admits_the_bad_carry() {
        assert!(df2_carry_ok(ON, DF2, 2048, 2200, SLOT_A, Some(SLOT_A), 2240, true));
    }

    /// §5 hazard: slot A once ran the drafter at 3000, then slot B clobbered the ring. The
    /// per-slot length for A still says 3000, so a LENGTH-ONLY check (DSpark's shape) would carry
    /// here and feed the draft slot B's rows. The identity guard refuses.
    #[test]
    fn stale_ring_from_another_slot_refused() {
        assert!(!df2_carry_ok(ON, DF2, 2048, 2600, SLOT_A, Some(SLOT_B), 3000, FULL));
    }

    /// Cold start: `reuse == 0` never carries (the lane primes from 0).
    #[test]
    fn cold_never_carries() {
        assert!(!df2_carry_ok(ON, DF2, 0, 2600, SLOT_A, Some(SLOT_A), 3000, FULL));
    }

    /// Unproven ring identity (reset / failed prime) refuses even when the length looks right.
    #[test]
    fn unproven_identity_refused() {
        assert!(!df2_carry_ok(ON, DF2, 2048, 2600, SLOT_A, None, 3000, FULL));
    }

    /// The ring must have covered the reused prefix.
    #[test]
    fn reuse_past_frontier_refused() {
        assert!(!df2_carry_ok(ON, DF2, 2500, 2600, SLOT_A, Some(SLOT_A), 2400, FULL));
    }

    /// Flag OFF is today's behaviour exactly: never carry, whatever the state says.
    #[test]
    fn flag_off_never_carries() {
        assert!(!df2_carry_ok(false, DF2, 2048, 2440, SLOT_A, Some(SLOT_A), 2240, FULL));
    }

    /// Non-DFlash2 source never carries.
    #[test]
    fn non_df2_source_never_carries() {
        assert!(!df2_carry_ok(ON, false, 2048, 2440, SLOT_A, Some(SLOT_A), 2240, FULL));
    }
}
