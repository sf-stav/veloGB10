//! `/metrics` rows of the Inkling-2 server (`velogb10_inkling2_*`), appended to the shared text (`crate::metrics::render`).
//! A counter for every new path (AGENTS §4); `scripts/owner_check.py` reads them (section stub: the integration leg fills the rest).

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

macro_rules! counters { ($($n:ident),* $(,)?) => { $(pub static $n: AtomicU64 = AtomicU64::new(0);)* }; }
counters!(
    SERVING, NOT_BUILT, LANES, LANES_REQUESTED, PREFIX_ON,
    REQ_OK, REQ_BAD, REQ_NOT_BUILT, REQ_ENGINE_ERR, STREAMED, NON_STREAMED, PROMPT_TOKENS, GEN_TOKENS,
    FINISH_STOP, FINISH_LENGTH, FINISH_TOOLS,
    EFFORT_DEFAULT, EFFORT_NAMED, EFFORT_NUMERIC, EFFORT_MAPPED,
    IMAGES, TOOL_REQS, TOOL_CALLS, TOOL_REJECTED, TOOL_OVER_LIMIT, TEXT_INVOCATIONS, CUT_CALLS, HEADERLESS_TEXT, CHOICE_UNENFORCED,
    FWD_PREFILL_TOKENS, FWD_DECODE_STEPS, FWD_DECODE_ROWS, FWD_GPU_SAMPLED, FWD_CPU_SAMPLED, SPEC_ROUNDS, PREFIX_HIT_TOKENS, PREFIX_HITS, FWD_INTERLEAVED_CHUNKS,
    SPEC_ON, SPEC_DRAFTED, SPEC_ACCEPTED, SPEC_PLAIN_ROWS, SPEC_KEPT_OUT_CPU, SPEC_TRUNC_ROUNDS,
    VIS_ON, VIS_MAX_TOKENS, VIS_TOWERS, VIS_CACHE_HITS, VIS_IMAGE_TOKENS, VIS_SPLICED_ROWS, VIS_TOWER_US, VIS_DOWNSCALED, VIS_BAD, VIS_TOWER_TOKENS,
    EXPERT_ACT_NATIVE,
    ULYSSES_WIRE_BF16,
    TP_WORLD,
);
/// Speculative rounds by accepted drafts (0..7).
pub static SPEC_ACCEPT_HIST: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
pub fn mark_start() { START.get_or_init(std::time::Instant::now); }
pub fn inc(c: &AtomicU64) { c.fetch_add(1, Relaxed); }
pub fn add(c: &AtomicU64, n: u64) { c.fetch_add(n, Relaxed); }
pub fn set(c: &AtomicU64, n: u64) { c.store(n, Relaxed); }
pub fn get(c: &AtomicU64) -> u64 { c.load(Relaxed) }

static REJECT_LABELS: std::sync::Mutex<Vec<(&'static str, u64)>> = std::sync::Mutex::new(Vec::new());
pub fn tool_reject(label: &'static str) {
    inc(&TOOL_REJECTED);
    let mut v = REJECT_LABELS.lock().unwrap_or_else(|e| e.into_inner());
    match v.iter_mut().find(|x| x.0 == label) { Some(x) => x.1 += 1, None => v.push((label, 1)) }
}

fn row(o: &mut String, name: &str, help: &str, kind: &str, v: u64) { let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {v}"); }

pub fn render() -> String {
    let mut o = String::new();
    row(&mut o, "velogb10_inkling2_serving", "1 when this server is the Inkling-2 serve shell.", "gauge", get(&SERVING));
    if let Some(t) = START.get() {
        let _ = writeln!(o, "# HELP velogb10_inkling2_uptime_seconds Seconds since the Inkling-2 server came up.\n# TYPE velogb10_inkling2_uptime_seconds gauge\nvelogb10_inkling2_uptime_seconds {:.1}", t.elapsed().as_secs_f64());
    }
    row(&mut o, "velogb10_inkling2_forward_not_built", "1 when no model is behind the server (the skeleton): every chat request is answered with an error, never a made-up reply.", "gauge", get(&NOT_BUILT));
    row(&mut o, "velogb10_inkling2_lanes", "Requests decoded in one step.", "gauge", get(&LANES));
    row(&mut o, "velogb10_inkling2_lanes_requested", "The --max-batch the operator asked for.", "gauge", get(&LANES_REQUESTED));
    row(&mut o, "velogb10_inkling2_prefix_cache", "1 when --prefix-cache on is in effect.", "gauge", get(&PREFIX_ON));
    row(&mut o, "velogb10_inkling2_expert_act_native", "1 when the experts run the pack-declared NVFP4 / MXFP8 activations (--inkling2-expert-act native); 0 = 16-bit activations (default).", "gauge", get(&EXPERT_ACT_NATIVE));
    row(&mut o, "velogb10_inkling2_ulysses_wire_bf16", "1 when DeepSpeed-Ulysses attention All-to-All uses BF16 wire compression (--inkling2-ulysses-wire bf16); 0 = FP32 wire (default, bit-exact).", "gauge", get(&ULYSSES_WIRE_BF16));
    row(&mut o, "velogb10_inkling2_requests_ok_total", "Chat requests finished normally.", "counter", get(&REQ_OK));
    row(&mut o, "velogb10_inkling2_requests_bad_total", "Requests answered 400 (bad body or fields, a template error).", "counter", get(&REQ_BAD));
    row(&mut o, "velogb10_inkling2_requests_not_built_total", "Requests answered with the not-built error (the prompt rendered; no model ran).", "counter", get(&REQ_NOT_BUILT));
    row(&mut o, "velogb10_inkling2_requests_engine_error_total", "Requests failed by a forward error.", "counter", get(&REQ_ENGINE_ERR));
    row(&mut o, "velogb10_inkling2_streamed_requests_total", "Streamed chat requests.", "counter", get(&STREAMED));
    row(&mut o, "velogb10_inkling2_nonstreamed_requests_total", "Non-streamed chat requests.", "counter", get(&NON_STREAMED));
    row(&mut o, "velogb10_inkling2_prompt_tokens_total", "Prompt tokens of rendered requests.", "counter", get(&PROMPT_TOKENS));
    row(&mut o, "velogb10_inkling2_generated_tokens_total", "Generated tokens of finished requests.", "counter", get(&GEN_TOKENS));
    let _ = writeln!(o, "# HELP velogb10_inkling2_finish_total Finished requests by finish_reason.\n# TYPE velogb10_inkling2_finish_total counter");
    for (l, c) in [("stop", &FINISH_STOP), ("length", &FINISH_LENGTH), ("tool_calls", &FINISH_TOOLS)] { let _ = writeln!(o, "velogb10_inkling2_finish_total{{reason=\"{l}\"}} {}", get(c)); }
    let _ = writeln!(o, "# HELP velogb10_inkling2_reasoning_effort_requests_total Requests by how reasoning_effort reached the template.\n# TYPE velogb10_inkling2_reasoning_effort_requests_total counter");
    for (l, c) in [("template_default", &EFFORT_DEFAULT), ("named", &EFFORT_NAMED), ("numeric", &EFFORT_NUMERIC), ("mapped", &EFFORT_MAPPED)] {
        let _ = writeln!(o, "velogb10_inkling2_reasoning_effort_requests_total{{how=\"{l}\"}} {}", get(c));
    }
    row(&mut o, "velogb10_inkling2_image_parts_total", "Image parts rendered into prompts (one placeholder each).", "counter", get(&IMAGES));
    row(&mut o, "velogb10_inkling2_vision_on", "1 when the vision tower is loaded (image placeholders are expanded and filled with image features).", "gauge", get(&VIS_ON));
    row(&mut o, "velogb10_inkling2_image_max_tokens", "--image-max-tokens: the most LM tokens one image may take (above it the image is downscaled).", "gauge", get(&VIS_MAX_TOKENS));
    row(&mut o, "velogb10_inkling2_vision_towers_total", "Images run through the GPU vision tower.", "counter", get(&VIS_TOWERS));
    row(&mut o, "velogb10_inkling2_vision_tower_tokens_total", "LM tokens produced by those tower runs.", "counter", get(&VIS_TOWER_TOKENS));
    row(&mut o, "velogb10_inkling2_vision_cache_hits_total", "Images whose features came from the tower cache (same image content seen before; no tower run).", "counter", get(&VIS_CACHE_HITS));
    let _ = writeln!(o, "# HELP velogb10_inkling2_vision_tower_ms_total GPU tower time in ms (wall, synchronised), summed.\n# TYPE velogb10_inkling2_vision_tower_ms_total counter\nvelogb10_inkling2_vision_tower_ms_total {:.1}", get(&VIS_TOWER_US) as f64 / 1000.0);
    row(&mut o, "velogb10_inkling2_image_tokens_total", "Image tokens in expanded prompts (rows x cols per image).", "counter", get(&VIS_IMAGE_TOKENS));
    row(&mut o, "velogb10_inkling2_image_rows_spliced_total", "Prompt rows whose embedding was replaced by image features in the prefill.", "counter", get(&VIS_SPLICED_ROWS));
    row(&mut o, "velogb10_inkling2_images_downscaled_total", "Images above --image-max-tokens, downscaled to fit (never refused).", "counter", get(&VIS_DOWNSCALED));
    row(&mut o, "velogb10_inkling2_images_bad_total", "Image parts that could not be decoded (answered 400 invalid_image).", "counter", get(&VIS_BAD));
    row(&mut o, "velogb10_inkling2_tool_requests_total", "Requests that offered tools the parser may read calls for.", "counter", get(&TOOL_REQS));
    row(&mut o, "velogb10_inkling2_tool_calls_total", "Tool calls returned to clients.", "counter", get(&TOOL_CALLS));
    row(&mut o, "velogb10_inkling2_tool_blocks_rejected_total", "Call payloads that failed the catalogue checks and stayed visible as content.", "counter", get(&TOOL_REJECTED));
    let _ = writeln!(o, "# HELP velogb10_inkling2_tool_reject_reason_total Rejected call payloads by the check that failed.\n# TYPE velogb10_inkling2_tool_reject_reason_total counter");
    for (l, c) in REJECT_LABELS.lock().unwrap_or_else(|e| e.into_inner()).iter() { let _ = writeln!(o, "velogb10_inkling2_tool_reject_reason_total{{reason=\"{l}\"}} {c}"); }
    row(&mut o, "velogb10_inkling2_tool_calls_over_limit_total", "Calls dropped by parallel_tool_calls: false.", "counter", get(&TOOL_OVER_LIMIT));
    row(&mut o, "velogb10_inkling2_text_invocations_total", "Headerless raw-text invocation blocks (kept as text, never a call).", "counter", get(&TEXT_INVOCATIONS));
    row(&mut o, "velogb10_inkling2_cut_calls_total", "Call blocks cut by the token budget (shown as text, never run).", "counter", get(&CUT_CALLS));
    row(&mut o, "velogb10_inkling2_headerless_text_total", "Replies that began without a message header (read as model text).", "counter", get(&HEADERLESS_TEXT));
    row(&mut o, "velogb10_inkling2_prefill_tokens_total", "Prompt tokens run through the GPU prefill (chunks of 2,048, 512 while other lanes decode).", "counter", get(&FWD_PREFILL_TOKENS));
    row(&mut o, "velogb10_inkling2_decode_steps_total", "GPU decode steps (one forward over every active lane).", "counter", get(&FWD_DECODE_STEPS));
    row(&mut o, "velogb10_inkling2_decode_rows_total", "Lane rows decoded over all steps (rows / steps = mean lanes per step).", "counter", get(&FWD_DECODE_ROWS));
    row(&mut o, "velogb10_inkling2_gpu_sampled_tokens_total", "Tokens drawn by the GPU sampler (greedy, or temperature-only Gumbel-max).", "counter", get(&FWD_GPU_SAMPLED));
    row(&mut o, "velogb10_inkling2_cpu_sampled_tokens_total", "Tokens drawn on the CPU from the downloaded logits (requests with top_p < 1 or top_k > 0).", "counter", get(&FWD_CPU_SAMPLED));
    row(&mut o, "velogb10_inkling2_spec_on", "1 when the bundled drafter is loaded and speculation is on (--mtp auto / on).", "gauge", get(&SPEC_ON));
    row(&mut o, "velogb10_inkling2_spec_rounds_total", "Lane rounds that drafted (drafter run, its drafts verified).", "counter", get(&SPEC_ROUNDS));
    row(&mut o, "velogb10_inkling2_spec_drafts_verified_total", "Drafts verified by the target over all rounds.", "counter", get(&SPEC_DRAFTED));
    row(&mut o, "velogb10_inkling2_spec_drafts_accepted_total", "Drafts accepted (exact rule) over all rounds.", "counter", get(&SPEC_ACCEPTED));
    row(&mut o, "velogb10_inkling2_spec_plain_rows_total", "Lane steps decoded plain while speculation is on (policy back-off, no room, or a CPU-sampled request).", "counter", get(&SPEC_PLAIN_ROWS));
    row(&mut o, "velogb10_inkling2_spec_kept_out_cpu_total", "Lane steps kept out of speculation because the request sets top_p < 1 or top_k > 0 (none since S8: such requests speculate).", "counter", get(&SPEC_KEPT_OUT_CPU));
    row(&mut o, "velogb10_inkling2_spec_trunc_rounds_total", "Drafting rounds of requests with top_p < 1 or top_k > 0, accepted exactly on the host against the truncated target.", "counter", get(&SPEC_TRUNC_ROUNDS));
    let _ = writeln!(o, "# HELP velogb10_inkling2_spec_accepted_rounds_total Drafting rounds by accepted drafts.\n# TYPE velogb10_inkling2_spec_accepted_rounds_total counter");
    for (i, c) in SPEC_ACCEPT_HIST.iter().enumerate() { let _ = writeln!(o, "velogb10_inkling2_spec_accepted_rounds_total{{accepted=\"{i}\"}} {}", get(c)); }
    row(&mut o, "velogb10_inkling2_prefix_hit_tokens_total", "Prompt tokens served from the prefix cache (a saved prompt end resumed instead of prefilled).", "counter", get(&PREFIX_HIT_TOKENS));
    row(&mut o, "velogb10_inkling2_prefix_hits_total", "Requests that resumed from a saved prompt end.", "counter", get(&PREFIX_HITS));
    // S-INK2-TP2: the tensor-parallel engagement receipt (AGENTS §4): world, model calls announced to the nodes, reduces built
    let (red, bytes) = crate::inkling2::tp::xport::counts();
    row(&mut o, "velogb10_inkling2_tp_world", "Ranks the model is split over (1 = TP1 on this box; 2 / 3 / 4 / 6 / 8 = TP2 / TP3 / TP4 / TP6 / TP8, this box = rank 0, the head).", "gauge", get(&TP_WORLD).max(1));
    row(&mut o, "velogb10_inkling2_tp_ops_total", "Model calls the head announced to the nodes over the control stream (TP only; the nodes replay them in order).", "counter", crate::inkling2::tp::ops_sent());
    row(&mut o, "velogb10_inkling2_tp_reduces_built_total", "All-reduces launched eagerly or captured into a decode graph (a graph replay re-runs its captured reduces without counting).", "counter", red);
    row(&mut o, "velogb10_inkling2_tp_reduce_bytes_total", "Bytes of the all-reduces counted in tp_reduces_built_total.", "counter", bytes);
    row(&mut o, "velogb10_inkling2_interleaved_prefill_chunks_total", "Prefill chunks run between decode steps of other lanes (512 rows each; the others kept streaming).", "counter", get(&FWD_INTERLEAVED_CHUNKS));
    row(&mut o, "velogb10_inkling2_tool_choice_unenforced_total", "Requests with tool_choice required or a named function: served, the choice is not forced by this build.", "counter", get(&CHOICE_UNENFORCED));
    o
}

pub fn render_all(engine_alive: bool) -> String {
    let mut s = crate::metrics::render(engine_alive);
    if !s.ends_with('\n') { s.push('\n'); }
    s.push_str(&render());
    s
}
