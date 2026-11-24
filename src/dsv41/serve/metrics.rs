//! `/metrics` rows of the dsv41 server: a counter for every new path (AGENTS §4: identical hashes are not a liveness
//! receipt), appended to the shared server's text (`crate::metrics::render`) so the generic rows (build, uptime, ...)
//! stay in one place. `scripts/owner_check.py` reads the `velogb10_dsv41_*` rows below.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::fmt::Write as _;

macro_rules! counters {
    ($($name:ident),* $(,)?) => { $(pub static $name: AtomicU64 = AtomicU64::new(0);)* };
}

counters!(
    // identity / gauges
    SERVING, STUB, LANES, LANES_REQUESTED, ACTIVE, WAITING, PREFIX_ON, SINK_ON,
    // requests
    REQ_OK, REQ_BAD, REQ_BUSY, REQ_CANCELLED, REQ_ENGINE_ERR, STREAMED, NON_STREAMED,
    PROMPT_TOKENS, GEN_TOKENS, FINISH_STOP, FINISH_LENGTH, FINISH_TOOLS,
    // prefix cache
    PFX_HITS, PFX_MISSES, PFX_REUSED, PFX_EVICTIONS, PFX_TAIL_EVICTIONS, PFX_BYTES, PFX_TAIL_BYTES, PFX_ENTRIES, PFX_SNAPS, PFX_BUDGET,
    // sink handling
    SINK_PROMPT_ROWS, SINK_FLAGGED, SINK_REJECTIONS, SINK_FALLTHROUGHS,
    // chat / tools / reasoning
    THINK_ON, THINK_OFF, TOOL_REQS, TOOL_CALLS, TOOL_REJECTED, TOOL_BREAKS, TOOL_UNTERMINATED, TOOL_OVER_LIMIT, ORPHAN_CLOSERS,
    THINK_BUDGET_FORCED, STOP_STRING_HITS, LOGPROBS_UNSUPPORTED, SCHEMA_UNENFORCED,
    // streaming hygiene
    SSE_KEEPALIVES, BACKLOG_CANCELS, ROLE_BEFORE_TOKEN,
    // sampler
    SAMPLED_REQS, GREEDY_REQS, SEEDLESS_RANDOM, SEEDLESS_HASH,
    // score mode (leg CE)
    SCORE_REQS, SCORE_TOKENS,
    // the TP serve loop (leg SE)
    TP_ROUNDS, TP_NOPS, TP_PREFILLS, TP_PREFILL_TOKENS, THINK_BUDGET_UNENFORCED, CUT_UNAPPLIED, PENALTY_UNAPPLIED, TP_FATAL,
    // leg SE2: top-k / top-p at TP, prefill yields
    CUT_ROWS, CUT_WIDE, CUT_CAPPED, PREFILL_YIELDS,
    // leg CE2: kept prefix states a prompt could not resume from (fallbacks, never a failure)
    PFX_RESUME_REFUSED, TP_RESUME_COLD,
    // leg CE3: resumes that left fewer than 134 rows (a tail state: the same prompt again, its last tokens changed, a short extension)
    PFX_TAIL_RESUMES,
    // leg VIS1: multimodal vision requests, Rank-0 encoded images, expanded image tokens, and GPU ViT+aligner encode microseconds
    VIS_REQS, VIS_IMAGES, VIS_TOKENS, VIS_ENCODE_US,
    // leg P7.2 / P7.3: min_tokens & ignore_eos requests, batched multi-lane prefills and starts
    MIN_TOKENS_REQS, IGNORE_EOS_REQS, TP_BATCH_PREFILLS, TP_BATCH_STARTS,
);

static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
/// Mark the server's start (the HTTP side coming up); `velogb10_dsv41_uptime_seconds` counts from here. The shared uptime row is
/// fed by a scheduler this server does not run, so `owner_check.py` read "uptime not exposed" (RV1 F3).
pub fn mark_start() { START.get_or_init(std::time::Instant::now); }

pub fn inc(c: &AtomicU64) { c.fetch_add(1, Relaxed); }
pub fn add(c: &AtomicU64, n: u64) { c.fetch_add(n, Relaxed); }
pub fn set(c: &AtomicU64, n: u64) { c.store(n, Relaxed); }
pub fn dec(c: &AtomicU64) { let _ = c.fetch_update(Relaxed, Relaxed, |v| Some(v.saturating_sub(1))); }
pub fn get(c: &AtomicU64) -> u64 { c.load(Relaxed) }

/// Per-label reject counters of the DSML catalogue checks (`Reject::label`).
static TOOL_REJECT_LABELS: std::sync::Mutex<Vec<(&'static str, u64)>> = std::sync::Mutex::new(Vec::new());
pub fn tool_reject(label: &'static str) {
    inc(&TOOL_REJECTED);
    let mut v = TOOL_REJECT_LABELS.lock().unwrap_or_else(|e| e.into_inner());
    match v.iter_mut().find(|x| x.0 == label) { Some(x) => x.1 += 1, None => v.push((label, 1)) }
}

fn row(o: &mut String, name: &str, help: &str, kind: &str, v: u64) {
    let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {v}");
}

/// The dsv41 section of `/metrics`.
pub fn render() -> String {
    let mut o = String::new();
    row(&mut o, "velogb10_dsv41_serving", "1 when this server is the DeepSeek-V4.1 serve shell.", "gauge", get(&SERVING));
    if let Some(t) = START.get() {
        let _ = writeln!(o, "# HELP velogb10_dsv41_uptime_seconds Seconds since the DeepSeek-V4.1 server came up.\n# TYPE velogb10_dsv41_uptime_seconds gauge\nvelogb10_dsv41_uptime_seconds {:.1}", t.elapsed().as_secs_f64());
    }
    row(&mut o, "velogb10_dsv41_stub_forward", "1 when the forward is the CPU stub (no model compute): a smoke / test server, never a speed receipt.", "gauge", get(&STUB));
    row(&mut o, "velogb10_dsv41_lanes", "Lanes (requests decoded in one step) this server runs, 1..8.", "gauge", get(&LANES));
    row(&mut o, "velogb10_dsv41_lanes_requested", "The --max-batch the operator asked for (a larger value boots on the lane cap).", "gauge", get(&LANES_REQUESTED));
    row(&mut o, "velogb10_dsv41_lanes_active", "Lanes decoding now.", "gauge", get(&ACTIVE));
    row(&mut o, "velogb10_dsv41_waiting", "Requests queued for a lane.", "gauge", get(&WAITING));
    row(&mut o, "velogb10_dsv41_requests_ok_total", "Chat requests finished normally (stop, length or tool_calls).", "counter", get(&REQ_OK));
    row(&mut o, "velogb10_dsv41_score_requests_total", "Prompt-scoring requests (/v1/completions with echo and max_tokens 0): an exact prefill, every token's log-prob back.", "counter", get(&SCORE_REQS));
    row(&mut o, "velogb10_dsv41_score_tokens_total", "Prompt tokens scored (exact prefill, every layer on every row).", "counter", get(&SCORE_TOKENS));
    row(&mut o, "velogb10_dsv41_ced_keep_raised_total", "Times a --dsv41-ced-keep below 134 was raised to 134 (window 128 + 6 backtracking).", "counter",
        crate::dsv41::ced::schedule::KEEP_RAISED.load(Relaxed));
    row(&mut o, "velogb10_dsv41_requests_bad_total", "Requests answered 400 (unparseable body, bad fields, an encoding error).", "counter", get(&REQ_BAD));
    row(&mut o, "velogb10_dsv41_requests_busy_total", "Requests answered 503 because --max-waiting requests were already queued.", "counter", get(&REQ_BUSY));
    row(&mut o, "velogb10_dsv41_requests_cancelled_total", "Requests cancelled by the client going away, a stop string, or a stream backlog.", "counter", get(&REQ_CANCELLED));
    row(&mut o, "velogb10_dsv41_requests_engine_error_total", "Requests failed by a forward error.", "counter", get(&REQ_ENGINE_ERR));
    row(&mut o, "velogb10_dsv41_streamed_requests_total", "Streamed (SSE) chat requests.", "counter", get(&STREAMED));
    row(&mut o, "velogb10_dsv41_nonstreamed_requests_total", "Non-streamed chat requests.", "counter", get(&NON_STREAMED));
    row(&mut o, "velogb10_dsv41_prompt_tokens_total", "Prompt tokens of finished requests.", "counter", get(&PROMPT_TOKENS));
    row(&mut o, "velogb10_dsv41_generated_tokens_total", "Generated (committed) tokens of finished requests.", "counter", get(&GEN_TOKENS));
    let _ = writeln!(o, "# HELP velogb10_dsv41_finish_total Finished requests by finish_reason.\n# TYPE velogb10_dsv41_finish_total counter");
    for (l, c) in [("stop", &FINISH_STOP), ("length", &FINISH_LENGTH), ("tool_calls", &FINISH_TOOLS)] {
        let _ = writeln!(o, "velogb10_dsv41_finish_total{{reason=\"{l}\"}} {}", get(c));
    }
    row(&mut o, "velogb10_dsv41_prefix_cache", "1 when --prefix-cache on is in effect.", "gauge", get(&PREFIX_ON));
    row(&mut o, "velogb10_dsv41_prefix_hits_total", "Requests that resumed from a kept prefix state.", "counter", get(&PFX_HITS));
    row(&mut o, "velogb10_dsv41_prefix_misses_total", "Requests with no usable kept state (cold prefill).", "counter", get(&PFX_MISSES));
    row(&mut o, "velogb10_dsv41_prefix_reused_tokens_total", "Prompt tokens served by kept state (prefill skipped).", "counter", get(&PFX_REUSED));
    row(&mut o, "velogb10_dsv41_prefix_snapshots_total", "Prefix states taken (prompt grid, prompt end, generation end).", "counter", get(&PFX_SNAPS));
    row(&mut o, "velogb10_dsv41_prefix_resume_refused_total", "Kept prefix states the pool offered that the forward could not resume this prompt from exactly; the request fell back to the next usable state or a cold prefill (a label defect if non-zero).", "counter", get(&PFX_RESUME_REFUSED));
    row(&mut o, "velogb10_dsv41_prefix_tail_resumes_total", "Requests resumed from a kept prompt-end state with fewer than 134 rows left to compute (the same prompt again, its last few tokens changed, or a short extension): the state carries the prompt tail the resumed prefill needs.", "counter", get(&PFX_TAIL_RESUMES));
    row(&mut o, "velogb10_dsv41_tp_resume_cold_fallback_total", "Resumed TP prefills the head found unresumable at their first call and prefilled cold instead of sending them to the ranks (the last-resort fallback).", "counter", get(&TP_RESUME_COLD));
    row(&mut o, "velogb10_dsv41_prefix_evictions_total", "Prefix states evicted from the pool (two-tier LRU: non-tail first, then tail).", "counter", get(&PFX_EVICTIONS));
    row(&mut o, "velogb10_dsv41_prefix_tail_evictions_total", "Prompt-end tail prefix states evicted from the pool (only when zero non-tail states remain).", "counter", get(&PFX_TAIL_EVICTIONS));
    row(&mut o, "velogb10_dsv41_prefix_pool_bytes", "Bytes the pool holds.", "gauge", get(&PFX_BYTES));
    row(&mut o, "velogb10_dsv41_prefix_pool_tail_bytes", "Bytes the pool holds in prompt-end tail states.", "gauge", get(&PFX_TAIL_BYTES));
    row(&mut o, "velogb10_dsv41_prefix_pool_budget_bytes", "The pool budget (--prefix-pool-gb).", "gauge", get(&PFX_BUDGET));
    row(&mut o, "velogb10_dsv41_prefix_pool_entries", "Prefix states held.", "gauge", get(&PFX_ENTRIES));
    row(&mut o, "velogb10_dsv41_sink_reject", "1 when --dsv41-sink-reject (opt-in sink detection + draft-zero rejection) is on.", "gauge", get(&SINK_ON));
    let _ = writeln!(o, "# HELP velogb10_dsv41_sink_flagged_rows_total Rows whose sink magnitude was at or over the threshold.\n# TYPE velogb10_dsv41_sink_flagged_rows_total counter");
    let _ = writeln!(o, "velogb10_dsv41_sink_flagged_rows_total{{row=\"draft_zero\"}} {}\nvelogb10_dsv41_sink_flagged_rows_total{{row=\"prompt\"}} {}",
                     get(&SINK_FLAGGED), get(&SINK_PROMPT_ROWS));
    row(&mut o, "velogb10_dsv41_sink_draft_zero_rejections_total", "Provisional tokens rejected and replaced by an alternate.", "counter", get(&SINK_REJECTIONS));
    row(&mut o, "velogb10_dsv41_sink_fallthroughs_total", "Sunk tokens committed after the alternates ran out.", "counter", get(&SINK_FALLTHROUGHS));
    row(&mut o, "velogb10_dsv41_thinking_requests_total", "Requests rendered in thinking mode.", "counter", get(&THINK_ON));
    row(&mut o, "velogb10_dsv41_chat_mode_requests_total", "Requests rendered in chat (non-thinking) mode.", "counter", get(&THINK_OFF));
    row(&mut o, "velogb10_dsv41_tp_serve_rounds_total", "Speculative rounds the TP lane loop ran (every decoding step of every request is one).", "counter", get(&TP_ROUNDS));
    row(&mut o, "velogb10_dsv41_tp_serve_idle_nops_total", "Idle heartbeats the head sent so the nodes never wait past the transport watchdog.", "counter", get(&TP_NOPS));
    row(&mut o, "velogb10_dsv41_tp_serve_prefills_total", "Prompts prefilled through CE's exact CED at TP.", "counter", get(&TP_PREFILLS));
    row(&mut o, "velogb10_dsv41_tp_serve_prefill_tokens_total", "Prompt tokens the TP prefills computed (a resumed prefix state is not recomputed).", "counter", get(&TP_PREFILL_TOKENS));
    row(&mut o, "velogb10_dsv41_tp_batch_prefills_total", "Multi-lane batched TP prefill ops executed (leg P7.3).", "counter", get(&TP_BATCH_PREFILLS));
    row(&mut o, "velogb10_dsv41_tp_batch_starts_total", "Multi-lane batched TP start ops executed (leg P7.3).", "counter", get(&TP_BATCH_STARTS));
    row(&mut o, "velogb10_dsv41_think_budget_unenforced_total", "Requests with a thinking_token_budget the TP forward could not enforce (always 0 since leg SE2: the budget is enforced by replacing the round chain's tail with the closing sequence; see think_budget_forced_total).", "counter", get(&THINK_BUDGET_UNENFORCED));
    row(&mut o, "velogb10_dsv41_sampler_cut_unapplied_total", "Requests with top_p < 1 or top_k > 0 served without the cut (always 0 since leg SE2: the cut is applied to the target rows of every round; see sampler_cut_rows_total).", "counter", get(&CUT_UNAPPLIED));
    row(&mut o, "velogb10_dsv41_sampler_penalty_unapplied_total", "Requests with presence / frequency / repetition penalties served without them (always 0: penalties are applied on every verify row at TP, tp/pen.rs; the series stays so a dashboard keeps its name).", "counter", get(&PENALTY_UNAPPLIED));
    row(&mut o, "velogb10_dsv41_sampler_cut_rows_total", "Target rows given a top-k / top-p cut (a request with neither never adds to it).", "counter", get(&CUT_ROWS));
    row(&mut o, "velogb10_dsv41_sampler_cut_wide_total", "Cut calls that needed the wide candidate list (a nucleus beyond 256 tokens per rank).", "counter", get(&CUT_WIDE));
    row(&mut o, "velogb10_dsv41_sampler_cut_capped_total", "Cut rows whose nucleus exceeded the wide list (2,048 tokens): the 2,048 best were kept.", "counter", get(&CUT_CAPPED));
    row(&mut o, "velogb10_dsv41_prefill_yields_total", "Prefill chunks of a prompt that were followed by a decode round of the other lanes before the next chunk.", "counter", get(&PREFILL_YIELDS));
    row(&mut o, "velogb10_dsv41_tp_serve_fatal_total", "TP serve ops that failed and stopped the forward (the ranks can no longer be trusted to agree).", "counter", get(&TP_FATAL));
    row(&mut o, "velogb10_dsv41_think_budget_forced_total", "Reasoning blocks closed by the engine at the thinking_token_budget.", "counter", get(&THINK_BUDGET_FORCED));
    row(&mut o, "velogb10_dsv41_tool_requests_total", "Requests that offered tools the parser may read calls for.", "counter", get(&TOOL_REQS));
    row(&mut o, "velogb10_dsv41_tool_calls_total", "DSML tool calls returned to clients.", "counter", get(&TOOL_CALLS));
    row(&mut o, "velogb10_dsv41_tool_blocks_rejected_total", "DSML invoke blocks that failed the catalogue and stayed visible as content.", "counter", get(&TOOL_REJECTED));
    let _ = writeln!(o, "# HELP velogb10_dsv41_tool_reject_reason_total Rejected DSML blocks by the check that failed.\n# TYPE velogb10_dsv41_tool_reject_reason_total counter");
    for (l, n) in TOOL_REJECT_LABELS.lock().unwrap_or_else(|e| e.into_inner()).iter() {
        let _ = writeln!(o, "velogb10_dsv41_tool_reject_reason_total{{reason=\"{l}\"}} {n}");
    }
    row(&mut o, "velogb10_dsv41_tool_structural_breaks_total", "DSML blocks cut by a structural break (the parser's block-end rule).", "counter", get(&TOOL_BREAKS));
    row(&mut o, "velogb10_dsv41_tool_unterminated_total", "Replies that ended inside an unclosed DSML block.", "counter", get(&TOOL_UNTERMINATED));
    row(&mut o, "velogb10_dsv41_tool_calls_dropped_over_limit_total", "Calls dropped by parallel_tool_calls: false.", "counter", get(&TOOL_OVER_LIMIT));
    row(&mut o, "velogb10_dsv41_orphan_dsml_closers_total", "Replies that were only DSML closing tags (left as text).", "counter", get(&ORPHAN_CLOSERS));
    row(&mut o, "velogb10_dsv41_stop_string_hits_total", "Requests ended by a client stop string (never streamed).", "counter", get(&STOP_STRING_HITS));
    row(&mut o, "velogb10_dsv41_logprobs_unsupported_total", "Requests that asked for logprobs and were served logprobs: null.", "counter", get(&LOGPROBS_UNSUPPORTED));
    row(&mut o, "velogb10_dsv41_response_format_unenforced_total", "Requests whose response_format was given to the model in the prompt but not enforced by a token mask.", "counter", get(&SCHEMA_UNENFORCED));
    row(&mut o, "velogb10_dsv41_sse_keepalives_total", "SSE keep-alive frames sent.", "counter", get(&SSE_KEEPALIVES));
    row(&mut o, "velogb10_dsv41_stream_backlog_cancels_total", "Streams cancelled because the client fell --stream-backlog-events behind.", "counter", get(&BACKLOG_CANCELS));
    row(&mut o, "velogb10_dsv41_role_chunk_before_first_token_total", "Role chunks sent before the first token existed (must stay 0).", "counter", get(&ROLE_BEFORE_TOKEN));
    row(&mut o, "velogb10_dsv41_sampled_requests_total", "Requests drawn with temperature > 0.", "counter", get(&SAMPLED_REQS));
    row(&mut o, "velogb10_dsv41_greedy_requests_total", "Greedy requests (temperature 0).", "counter", get(&GREEDY_REQS));
    row(&mut o, "velogb10_dsv41_seedless_random_total", "Requests without a seed given a fresh random one (--seedless-seed random).", "counter", get(&SEEDLESS_RANDOM));
    row(&mut o, "velogb10_dsv41_seedless_prompt_hash_total", "Requests without a seed given a prompt hash (--seedless-seed prompt-hash).", "counter", get(&SEEDLESS_HASH));
    row(&mut o, "velogb10_dsv41_min_tokens_requests_total", "Requests with min_tokens > 0 admitted by the dsv41 server (leg P7.2).", "counter", get(&MIN_TOKENS_REQS));
    row(&mut o, "velogb10_dsv41_ignore_eos_requests_total", "Requests with ignore_eos: true admitted by the dsv41 server (leg P7.2).", "counter", get(&IGNORE_EOS_REQS));
    row(&mut o, "velogb10_dsv41_vision_requests_total", "Multimodal requests containing one or more images admitted by the dsv41 server.", "counter", get(&VIS_REQS));
    row(&mut o, "velogb10_dsv41_vision_images_total", "Images encoded through the Rank-0 ViT + 3x3 aligner (images covered by a prefix-cache hit are skipped).", "counter", get(&VIS_IMAGES));
    row(&mut o, "velogb10_dsv41_vision_tokens_total", "Expanded image-span tokens encoded on Rank 0 (start + patch grid + row newlines + end).", "counter", get(&VIS_TOKENS));
    row(&mut o, "velogb10_dsv41_vision_encode_us_total", "Total Rank-0 GPU microseconds spent in the 32-layer ViT + 3x3 aligner.", "counter", get(&VIS_ENCODE_US));
    // UXF F4: which of PV1's decode paths the forward took (host launch counts; see each HELP)
    crate::dsv41::weights::exl3::moe_dec::render(&mut o);
    crate::dsv41::gpu::attn::render(&mut o);
    o
}

/// The once-per-boot receipt block of the serve path (UXF F4): the SL forward's kernel entries by path and width
/// (`receipts::boot_lines`, once per process), then PV1's decode paths with their counts so far.
pub fn boot_receipt_lines() -> Vec<String> {
    let mut v = crate::dsv41::fwd::receipts::boot_lines();
    v.push(format!("[dsv41 fwd] decode MoE (PV1): {} launches, {} with the chain scratch (K parts), {} on the short-scratch fallback; decode attention: {} launches, {} with split groups",
                   crate::dsv41::weights::exl3::moe_dec::calls(), crate::dsv41::weights::exl3::moe_dec::chain_calls(), crate::dsv41::weights::exl3::moe_dec::short_scratch_calls(),
                   crate::dsv41::gpu::attn::decode_calls(), crate::dsv41::gpu::attn::grouped_calls()));
    v.push(format!("[dsv41 fwd] decode MoE structure (PM1, --dsv41-moe-cta): {} launches on d41x_mc_* (short CTAs, ready flags), {} on d41x_md_* (persistent)",
                   crate::dsv41::weights::exl3::moe_cp::calls(), crate::dsv41::weights::exl3::moe_cp::persistent_calls()));
    v
}

/// The whole `/metrics` body: the shared rows plus ours.
pub fn render_all(engine_alive: bool) -> String {
    let mut s = crate::metrics::render(engine_alive);
    if !s.ends_with('\n') { s.push('\n'); }
    s.push_str(&render());
    s
}
