use axum::{
    extract::{DefaultBodyLimit, Json, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response, Sse},
    response::sse::Event,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::{Stream, StreamExt};
use tower_http::cors::{CorsLayer, Any};
use uuid::Uuid;
use chrono;

use crate::batch::{BatchRequest, TokEvent};
use crate::tokenizer::{QwenTokenizer, ChatMessage, ToolCall, ThinkingMode};
use crate::{Usage, Timings, make_timings, PromptTokensDetails};

/// Sampling values used for the parameters a request leaves out, per thinking mode.
#[derive(Clone, Copy, Debug)]
pub struct SamplingDefaults {
    pub think: (f32, f32, usize),   // (temperature, top_p, top_k) when the prompt opens a <think> block
    pub no_think: (f32, f32, usize), // ... when it does not
}

impl SamplingDefaults {
    /// The historical server-wide values, one set for both modes.
    pub const LEGACY: Self = Self { think: (0.7, 0.8, 20), no_think: (0.7, 0.8, 20) };
    /// Qwen3.8-Flash-Next model card: thinking temperature=1.0 top_p=0.95 top_k=20; instruct
    /// (non-thinking) temperature=0.7 top_p=0.80 top_k=20 (min_p 0 in both).
    pub const QWEN38_CARD: Self = Self { think: (1.0, 0.95, 20), no_think: (0.7, 0.8, 20) };

    /// FLASHNEXT-DF2-B (ISSUES work[113]; owner decision 2026-10-08 "read the sampler triple from
    /// generation_config.json"): the unsent-field defaults come from the PACK's
    /// `generation_config.json` (temperature / top_p / top_k) when it has a temperature;
    /// `fallback` otherwise. The file carries ONE triple, so it becomes the thinking-mode
    /// values; the non-thinking values stay the family card's when the fallback is the Qwen3.8
    /// card (its instruct recommendation 0.7 / 0.8 / 20 is not in the file), else the same
    /// triple. Returns the defaults and a one-word provenance for the boot line.
    pub fn from_pack(model_dir: &str, fallback: Self) -> (Self, &'static str) {
        let path = format!("{}/generation_config.json", model_dir.trim_end_matches('/'));
        let Ok(txt) = std::fs::read_to_string(&path) else { return (fallback, "built-in (no generation_config.json)"); };
        let Ok(j) = serde_json::from_str::<serde_json::Value>(&txt) else { return (fallback, "built-in (generation_config.json unreadable)"); };
        let Some(t) = j.get("temperature").and_then(|v| v.as_f64()) else { return (fallback, "built-in (generation_config.json has no temperature)"); };
        let top_p = j.get("top_p").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        let top_k = j.get("top_k").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let think = (t as f32, top_p, top_k);
        let qwen38 = fallback.think == Self::QWEN38_CARD.think && fallback.no_think == Self::QWEN38_CARD.no_think;
        let no_think = if qwen38 { Self::QWEN38_CARD.no_think } else { think };
        (Self { think, no_think }, "generation_config.json")
    }

    /// The boot line's text for these defaults.
    pub fn describe(&self, source: &str) -> String {
        format!("sampler defaults for unsent fields: thinking T={} top_p={} top_k={}; non-thinking T={} top_p={} top_k={} (source: {source}; a request's own values always win)",
                self.think.0, self.think.1, self.think.2, self.no_think.0, self.no_think.1, self.no_think.2)
    }
}

#[derive(Clone)]
pub struct AppState {
    /// Per-mode fallback sampling (a request's explicit values always win).
    pub sampling_defaults: SamplingDefaults,
    pub scheduler: mpsc::UnboundedSender<BatchRequest>,
    pub tokenizer: Arc<QwenTokenizer>,
    pub model_name: String,
    pub default_max_tokens: usize,
    pub default_rep_penalty: f32,
    pub default_presence_penalty: f32,
    pub default_frequency_penalty: f32,
    /// Server-wide reasoning-effort default from --reasoning-effort. `None` means "unspecified":
    /// the model's OWN chat template picks its baked-in default (Qwen -> `xhigh`, hy_v3 -> `low`),
    /// which is the only value guaranteed to be valid for that family. A request's
    /// `reasoning_effort` field overrides per request.
    pub reasoning_effort: Option<String>,
    /// W1 (Phase 13): the server-wide `--thinking auto|on|off` policy (default Auto = pass nothing,
    /// the model's own chat template decides). A request's `chat_template_kwargs.enable_thinking`
    /// overrides it per call.
    pub thinking: ThinkingMode,
    /// `--output-prompts [cap]`: log every chat-completion request in human-readable form
    /// (effective params, one line per turn, rendered-prompt excerpt up to `cap` chars).
    /// 0 = off (default).
    pub output_prompts: usize,
    /// KV cache depth, in positions. NOTHING used to check a prompt against it: an over-long prompt
    /// ran `write_kv_prefill` straight past the end of the cache and corrupted the next allocation.
    pub max_seq_len: usize,
    /// Decode positions reserved beyond `max_tokens` for speculative verification/re-prime.
    /// Mirrors the scheduler reserve so an HTTP-clamped request is always admissible. (PR #4.)
    pub decode_headroom: usize,
    /// Scheduler prefix-cache flag (mirror of TpConfig.prefix_cache). The message-boundary
    /// checkpoint (`ckpt_at`) is only ever USED when the scheduler's prefix cache is on
    /// (batch.rs filters it again); gating its render+tokenize here saves the double
    /// template work on every request when the cache is off (TTFT fix (e)).
    pub prefix_cache: bool,
    /// Vision tower (visual trunk) loaded at server start, for image requests. `None` if the
    /// build/model has no vision (text-only server behaves exactly as before).
    pub vision_tower: Option<std::sync::Arc<crate::vision_tower::VisualTower>>,
    /// GPU vision tower (the fast path). When `Some` and `vision_cpu` is false, image requests run
    /// the forward on the GPU; `None` + `vision_cpu: true` (or both unset) keeps the CPU tower.
    pub vision_gpu: Option<std::sync::Arc<std::sync::Mutex<crate::vision_gpu::GpuVisualTower>>>,
    /// Force the CPU vision tower (--vision-cpu), as a diagnostic/escape hatch.
    pub vision_cpu: bool,
    /// Every token id that terminates an assistant turn for this model, resolved once at boot
    /// from the model's own config files (QwenTokenizer::stop_token_ids). Phase-2 A3 uses it to
    /// label a generation's terminal token as a stop token — a fact the Phase-1 ledger could
    /// not recover from the transcripts (harness never reads finish_reason/terminal ids).
    pub stop_ids: Vec<u32>,
    /// OTel generation-telemetry emitter (--otel-endpoint). `None` = OFF (the default): every
    /// telemetry hook site in the SSE path compiles to one `if let Some` branch — zero cost.
    /// Some = the lock-free-ring sink; the SSE chunk hooks below forward the SAME chunk bytes
    /// (single source of truth) and the timer-polled sender exports them (crate::otel).
    pub otel: Option<std::sync::Arc<crate::otel::OtelSink>>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: i64,
    owned_by: String,
    /// v0.7.3 (issue #8.1): the served context length — the engine-clamped
    /// `--max-seq-len`, the vLLM convention — so clients stop guessing. Additive:
    /// the pre-existing fields are unchanged.
    max_model_len: usize,
}

#[derive(Serialize)]
struct ModelList {
    object: String,
    data: Vec<ModelInfo>,
}

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let model = ModelInfo {
        id: state.model_name.clone(),
        object: "model".to_string(),
        created: chrono::Utc::now().timestamp(),
        owned_by: "rust_infer".to_string(),
        max_model_len: state.max_seq_len,
    };
    Json(ModelList {
        object: "list".to_string(),
        data: vec![model],
    })
}

async fn get_model(State(state): State<AppState>, axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    if id == state.model_name {
        Json(ModelInfo {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created: chrono::Utc::now().timestamp(),
            owned_by: "rust_infer".to_string(),
            max_model_len: state.max_seq_len,
        })
        .into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            format!("Model '{}' not found. Available: {}", id, state.model_name),
        )
            .into_response()
    }
}

/// The HTTP API listen address: `--host` (default 0.0.0.0, every interface) + the port. Every server
/// (NVFP4, EXL3, DSV4) binds through this, so `--host` means the same thing everywhere.
pub fn http_bind_addr(port: u16) -> anyhow::Result<std::net::SocketAddr> {
    use std::net::ToSocketAddrs;
    let host = crate::opts::var(crate::opt!("host")).unwrap_or_else(|_| "0.0.0.0".to_string());
    let host = host.trim().trim_start_matches('[').trim_end_matches(']').to_string();
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        return Ok(std::net::SocketAddr::new(ip, port));
    }
    (host.as_str(), port).to_socket_addrs().ok().and_then(|mut a| a.next()).ok_or_else(|| anyhow::anyhow!(
        "--host '{host}' is neither an IP address nor a resolvable host name (e.g. 127.0.0.1, 0.0.0.0, ::1)"))
}

/// Bind the HTTP API or exit 2 with a plain message (port in use, address not on this machine, ...),
/// instead of a panic backtrace.
pub async fn http_listen(port: u16) -> tokio::net::TcpListener {
    // S-B9-REL-BENCHFIX2 (W3): the ranged HTTP-layer options are resolved here, before the socket
    // exists; the keep-alive in force is one boot line.
    match serve_options_boot_line() {
        Ok(line) => { if crate::boot_report::debug() { println!("{line}"); } },
        Err(e) => fatal_option(&e),
    }
    let addr = match http_bind_addr(port) {
        Ok(a) => a,
        Err(e) => { eprintln!("error: {e}"); crate::logq::flush_and_exit(std::time::Duration::from_millis(300), 2); } // H7
    };
    match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: cannot listen on {addr}: {e} (another server on this port? --host not an address of this machine?)");
            crate::logq::flush_and_exit(std::time::Duration::from_millis(300), 2); // H7
        }
    }
}

/// The public model id resolution order (v0.7.3, issue #8.3): `--model-name` >
/// `--served-model-name` (vLLM-spelled alias) > the model card's `base_model:` line
/// > the directory name. Flags are honored verbatim; a blank value falls through.
pub fn resolve_model_name(model_name: Option<&str>, served_model_name: Option<&str>, model_path: &str) -> String {
    fn flag(v: Option<&str>) -> Option<&str> { v.map(str::trim).filter(|s| !s.is_empty()) }
    flag(model_name).or(flag(served_model_name)).map(|s| s.to_string())
        .unwrap_or_else(|| model_id_from_dir(model_path))
}

/// The model's PUBLIC id for /v1/models and response `model` fields: the model card's
/// frontmatter `model_name:` line (the pack's own hub id, e.g.
/// `model_name: doth4580/Naive-N0.5-Flash-EXL3-3.5bpw`), else its `base_model:` line, else the
/// directory name. `model_name:` was added 2026-10-20 (owner: the two Naive EXL3 packs both
/// announced themselves as NaiveAI's own releases, since a quantized pack's `base_model:` is the
/// ORIGINAL model, not the pack). Before `base_model:` the server reported the lab directory
/// fragment (`"model": "3.8-27b-nvfp4-full-all"`). `--model-name` still overrides.
pub fn model_id_from_dir(model_path: &str) -> String {
    let dir = std::path::Path::new(model_path.trim_end_matches('/'));
    if let Ok(card) = std::fs::read_to_string(dir.join("README.md")) {
        let mut base: Option<String> = None;
        for line in card.lines() {
            let l = line.trim();
            for (key, own) in [("model_name:", true), ("base_model:", false)] {
                if let Some(v) = l.strip_prefix(key) {
                    let v = v.trim().trim_matches('"').trim_matches('\'').trim();
                    if v.is_empty() { continue; }
                    if own { return v.to_string(); }
                    if base.is_none() { base = Some(v.to_string()); }
                }
            }
        }
        if let Some(b) = base { return b; }
    }
    dir.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// S8 (REL_V0_7_3 review): the hand-built SSE chunk builders, module-level so the escaping
/// is unit-testable. The MODEL ID is escaped exactly like the text: --model-name is honoured
/// verbatim now, and a quote or backslash in it would break every chunk.
fn role_chunk(cid: &str, created: i64, model: &str) -> String {
    format!("{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\"}},\"finish_reason\":null}}]}}",
        esc(cid), created, esc(model))
}

fn content_chunk(cid: &str, created: i64, model: &str, text: &str) -> String {
    format!("{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}",
        esc(cid), created, esc(model), esc(text))
}

fn reasoning_chunk(cid: &str, created: i64, model: &str, text: &str) -> String {
    format!("{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"reasoning_content\":\"{}\"}},\"finish_reason\":null}}]}}",
        esc(cid), created, esc(model), esc(text))
}

/// finish is the spec finish-reason; stop_field is the pre-rendered stop_reason JSON field
/// suffix (empty when the loop detector did not end the stream) — WP08.
fn final_chunk(cid: &str, created: i64, model: &str, finish: &str, stop_field: &str) -> String {
    format!("{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{}\"{}}}]}}",
        esc(cid), created, esc(model), esc(finish), stop_field)
}

/// The comment text of the keep-alive frame: the wire bytes are `: keep-alive\n\n`. An SSE
/// comment is a line that starts with ':'; it has no `data:` field, so no JSON and no `choices`.
const SSE_KEEPALIVE_TEXT: &str = "keep-alive";

/// S-B9-REL-BENCHFIX2 (W4): keep-alive frames must not hide a hung engine. With a comment frame
/// every N seconds a client's idle-read timeout never fires, so a request stuck behind a
/// scheduler that no longer runs would hold its client open for ever (before the keep-alive
/// existed, that timeout ended it). The rule (`keepalive_live`): a frame is sent only while
///   * the scheduler COMPLETED a unit of work (a decode round or a prefill chunk:
///     `metrics::sched_touch`, called by both schedulers) within the last `SSE_KEEPALIVE_STALL`, or
///   * the stream itself is younger than that AND the scheduler is not marked busy (it was idle
///     and is picking this request up: its last completed work may be hours old).
/// Why progress and not a ceiling on total silence: legitimate silence has no useful upper bound
/// (a 262K-token prefill is minutes; `--max-waiting` allows 256 requests queued behind it), but
/// legitimate silence always shows the scheduler completing work every few seconds (a 4095-row
/// prefill chunk takes ~3 s on one GB10, a decode round ~50 ms), and a hung scheduler completes
/// nothing. 60 s is 20 times the measured chunk time. A unit of work that really takes longer
/// only pauses the frames (the stream is then silent exactly as it was before keep-alives
/// existed) and they resume when it completes.
const SSE_KEEPALIVE_STALL: std::time::Duration = std::time::Duration::from_secs(60);

/// `--sse-keepalive N` (seconds; 1..=3600, 0 = off; the default is the registry row's): the
/// longest silence on an open stream of a live engine. Nothing but these comment frames is on
/// the wire during a queue wait or a prefill (a 30 s prefill carries two of them at 10 s).
/// Resolved once; `serve_options_boot_line` resolves it before the server listens.
pub(crate) fn sse_keepalive() -> Option<std::time::Duration> {
    static V: std::sync::OnceLock<Option<std::time::Duration>> = std::sync::OnceLock::new();
    *V.get_or_init(|| resolve_sse_keepalive().unwrap_or_else(|e| fatal_option(&e)))
}

/// Pure over the option store (unit-tested): Ok(None) = off.
fn resolve_sse_keepalive() -> Result<Option<std::time::Duration>, String> {
    let secs = crate::opts::ranged(crate::opt!("sse-keepalive"))? as u64;
    Ok((secs > 0).then(|| std::time::Duration::from_secs(secs)))
}

/// An option value the registry should have refused at startup reached a server reader: stop
/// with the message instead of serving on a silently substituted default.
fn fatal_option(e: &str) -> ! {
    eprintln!("FATAL: {e}");
    crate::logq::flush_and_exit(std::time::Duration::from_millis(300), 2)
}

/// "[serve] sse keep-alive: every 10 s" / "... off" (the boot line; unit-tested).
fn sse_keepalive_boot_line(every: Option<std::time::Duration>) -> String {
    match every {
        Some(d) => format!("[serve] sse keep-alive: every {} s (--sse-keepalive; a comment frame on a silent stream while \
                            the whole scheduler completed work, any request's decode round or prefill chunk, in the last {} s; \
                            a stream opened on a hung engine can still get frames for its first {} s)",
                           d.as_secs(), SSE_KEEPALIVE_STALL.as_secs(), SSE_KEEPALIVE_STALL.as_secs()),
        None => "[serve] sse keep-alive: off (--sse-keepalive 0)".to_string(),
    }
}

/// Resolve every HTTP-layer option that has a range BEFORE the server listens (a bad value is a
/// refusal at boot, never a silent default at the first request) and return the boot line.
fn serve_options_boot_line() -> Result<String, String> {
    let every = resolve_sse_keepalive()?;
    crate::opts::ranged(crate::opt!("max-waiting"))?;
    crate::opts::ranged(crate::opt!("stream-backlog-events"))?;
    let mut line = sse_keepalive_boot_line(every);
    // S-B9-TOOLPARSE3: which parser rules this server serves, and that the others are only counted
    // (the liveness receipt of the shadow counters: `velogb10_toolcall_rules_info` says the same)
    line += &format!("\n[serve] tool-call parser: values end by --tool-quote-pairing {}, blocks end by --tool-block-end {}, unclosed blocks by --tool-unclosed {}; the other cheap rules \
                      (quote pairing, the block rule) are also read on each response that holds a <tool_call> block (text up to {} KiB, \
                      up to {} blocks, one more linear pass per rule) and the disagreements are COUNTED (velogb10_toolcall_*); the cost \
                      model is {} the shadow; wherever it runs it spends at most {} work units per response ({} per block; about {} ms of one \
                      core), and a block over that is read by the v0.7.3 rule and counted; nothing of the shadow is served",
                     crate::opts::choice(crate::opt!("tool-quote-pairing")), crate::opts::choice(crate::opt!("tool-block-end")),
                     crate::opts::choice(crate::opt!("tool-unclosed")),
                     crate::tools::SHADOW_MAX_BYTES / 1024, crate::tools::SHADOW_MAX_BLOCKS,
                     if crate::opts::choice(crate::opt!("tool-shadow-cost")) == "on" { "ALSO read in" } else { "NOT read in (--tool-shadow-cost off)" },
                     crate::tools::cost::RESPONSE_UNITS, crate::tools::cost::BLOCK_UNITS, crate::tools::cost::response_ms_nominal());
    // TOOLSCRUB1: one line naming the scrub mode (off = the default = no served byte changes).
    line += &format!("\n{}", crate::tool_scrub::boot_line(crate::tool_scrub::Mode::from_opt()));
    if let Some((round, secs)) = parse_stall_drill(crate::opts::get(crate::opt!("inject-stall")).as_deref())? {
        line += &format!("\n[serve] DIAGNOSTIC --inject-stall {round}:{secs}: the scheduler thread will sleep {secs} s at round {round} \
                          (a drill: never on a server that serves users)");
    }
    Ok(line)
}

/// The same resolution without the line: `main` calls it right after the command line is parsed,
/// so a value the registry cannot refuse by type (`--inject-stall`) is refused within
/// milliseconds of the start, before anything is loaded. (`http_listen` resolves again, for the
/// line, when the server binds.)
pub fn check_serve_options(args: &[String]) -> Result<(), String> {
    serve_options_boot_line()?;
    stall_drill_topology(crate::opts::get(crate::opt!("inject-stall")).is_some(), args)
}

/// `--inject-stall` is a one-box drill: it puts ONE process's scheduler thread to sleep. On a TP
/// rank (head or node) the other ranks' watchdogs would declare the sleeping one dead, and the
/// option is per box (`Scope::Local`), so a node started with it would keep it whatever the head
/// says. Refused at startup with any TP role on the command line.
fn stall_drill_topology(drill_set: bool, args: &[String]) -> Result<(), String> {
    if !drill_set { return Ok(()); }
    match args.iter().find(|a| matches!(a.as_str(), "--tp" | "--head" | "--node") || a.starts_with("--tp=")) {
        Some(role) => Err(format!("--inject-stall is a one-box drill and is refused with {} (the sleeping rank would stall \
                                   the others and their watchdogs would declare it dead)", role.split('=').next().unwrap_or(role))),
        None => Ok(()),
    }
}

/// The SSE response of both streaming endpoints: `stream` plus, when `every` is set, the
/// keep-alive frames of `keepalive_frames` (liveness = the scheduler's work age). The frame shape
/// is `--sse-keepalive-frame`: the comment line (default) or a data frame (LANERES1).
fn sse_with_keepalive<S, E>(stream: S, every: Option<std::time::Duration>, frame: SseKeepAliveFrame) -> Response
where
    S: Stream<Item = Result<Event, E>> + Send + 'static,
    E: Into<axum::BoxError> + Send + 'static,
{
    match every {
        Some(d) => Sse::new(keepalive_frames(stream, d, frame, SSE_KEEPALIVE_STALL, crate::metrics::sched_state)).into_response(),
        None => Sse::new(stream).into_response(),
    }
}

/// The keep-alive rule (pure; `SSE_KEEPALIVE_STALL` has the reason): may a comment frame be sent
/// on a stream of age `stream_age`, given the scheduler's gauges?
fn keepalive_live(sched: crate::metrics::SchedState, stream_age: std::time::Duration, stall: std::time::Duration) -> bool {
    sched.work_age.map_or(false, |w| w <= stall) || (!sched.busy && stream_age <= stall)
}

/// `inner` plus one comment frame after each `every` of silence (the timer restarts at every
/// event, so a stream that is producing tokens carries none), as long as `keepalive_live` says
/// the engine is alive (`sched()` = the scheduler's gauges at that moment). The data events pass
/// through untouched, in order. This replaces axum's `KeepAlive`, which cannot be conditional
/// and computes `Instant::now() + interval` unchecked at every event.
fn keepalive_frames<S, E, F>(inner: S, every: std::time::Duration, frame: SseKeepAliveFrame, stall: std::time::Duration, sched: F)
    -> impl Stream<Item = Result<Event, E>> + Send + 'static
where
    S: Stream<Item = Result<Event, E>> + Send + 'static,
    E: Send + 'static,
    F: Fn() -> crate::metrics::SchedState + Send + 'static,
{
    async_stream::stream! {
        tokio::pin!(inner);
        let born = tokio::time::Instant::now();
        // an interval too large to add to a clock means "no frame", never a panic
        let after = |t: tokio::time::Instant| t.checked_add(every);
        let mut deadline = after(born);
        loop {
            let next = match deadline {
                Some(d) => tokio::time::timeout_at(d, inner.next()).await.ok(),
                None => Some(inner.next().await),
            };
            match next {
                Some(Some(item)) => { yield item; }
                Some(None) => break,
                None => {
                    if keepalive_live(sched(), born.elapsed(), stall) {
                        yield Ok(match &frame {
                            SseKeepAliveFrame::Comment => Event::default().comment(SSE_KEEPALIVE_TEXT),
                            SseKeepAliveFrame::Data(payload) => Event::default().data(payload.as_ref()),
                        });
                    }
                }
            }
            deadline = after(tokio::time::Instant::now());
        }
    }
}

/// LANERES1 `--sse-keepalive-frame`: the shape of the idle keep-alive frames. `Comment` = today:
/// the SSE comment line `: keep-alive` (no data field; SSE clients ignore it). `Data` = an SSE
/// data frame carrying a chat-completions chunk with an EMPTY `choices` array: SDKs whose stream
/// guard counts only chunks with a choices entry never yield it to the caller, but client idle
/// timers fed by received SSE data frames reset (andrei.gnezdilov: qwen-code is such a client).
/// The house rule stands: no choices ENTRY is ever sent before the request's first real token —
/// the keep-alive chunk's `choices` array is empty.
#[derive(Clone, Debug)]
pub(crate) enum SseKeepAliveFrame {
    Comment,
    /// The prebuilt JSON payload (this stream's id/created/model; `choices: []`).
    Data(std::sync::Arc<str>),
}

/// The configured mode (--sse-keepalive-frame; a registry-validated Choice, default comment).
fn sse_keepalive_frame_mode() -> &'static str {
    static V: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    V.get_or_init(|| match crate::opts::choice(crate::opt!("sse-keepalive-frame")).as_str() {
        "data" => "data",
        _ => "comment",
    })
}

/// The pure payload of a data-mode keep-alive frame (unit-tested): a chunk with an EMPTY choices
/// array and the stream's own id/created/model.
fn sse_keepalive_data_payload(id: &str, created: i64, model: &str, object: &str) -> String {
    serde_json::json!({
        "id": id, "object": object, "created": created, "model": model, "choices": [],
    }).to_string()
}

/// Pure over the mode string (unit-tested): "data" → the data frame, anything else → comment.
fn sse_keepalive_frame_of_mode(mode: &str, id: &str, created: i64, model: &str, object: &str) -> SseKeepAliveFrame {
    match mode {
        "data" => SseKeepAliveFrame::Data(sse_keepalive_data_payload(id, created, model, object).into()),
        _ => SseKeepAliveFrame::Comment,
    }
}

/// The frame argument of `sse_with_keepalive` for THIS stream (id/created/model at the call site).
fn sse_keepalive_frame(id: &str, created: i64, model: &str, object: &str) -> SseKeepAliveFrame {
    sse_keepalive_frame_of_mode(sse_keepalive_frame_mode(), id, created, model, object)
}

/// `--inject-stall ROUND:SECONDS` (diagnostic; unset = no drill): Ok(None) when unset. ROUND is
/// the scheduler round counted from boot (velogb10_scheduler_steps_total + 1 is the next one),
/// SECONDS 1..=600.
fn parse_stall_drill(v: Option<&str>) -> Result<Option<(u64, u64)>, String> {
    let Some(v) = v else { return Ok(None) };
    let bad = || format!("--inject-stall takes ROUND:SECONDS, ROUND >= 1 and SECONDS from 1 to 600 (got '{v}')");
    let (r, s) = v.split_once(':').ok_or_else(bad)?;
    match (r.trim().parse::<u64>(), s.trim().parse::<u64>()) {
        (Ok(r), Ok(s)) if r >= 1 && (1..=600).contains(&s) => Ok(Some((r, s))),
        _ => Err(bad()),
    }
}

fn stall_drill_cfg() -> Option<(u64, u64)> {
    static V: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();
    *V.get_or_init(|| parse_stall_drill(crate::opts::get(crate::opt!("inject-stall")).as_deref())
        .unwrap_or_else(|e| fatal_option(&e)))
}

/// The W4 drill, called by `metrics::sched_step` at the start of every scheduler round, on the
/// scheduler thread: with `--inject-stall ROUND:SECONDS` that thread sleeps SECONDS once, at
/// round ROUND. Nothing completes meanwhile (no decode round, no prefill chunk, no admission),
/// which is what a hung scheduler looks like to the HTTP layer; the HTTP task itself keeps
/// running (/health, /metrics, the keep-alive timers). Unset: one OnceLock read per round.
pub(crate) fn stall_drill(round: u64) {
    if let Some((at, secs)) = stall_drill_cfg() {
        if round == at {
            crate::reprintln!("[drill] --inject-stall: the scheduler thread sleeps {secs} s at round {at}: no work completes; \
                               keep-alive frames must stop once the last completed work is {} s old", SSE_KEEPALIVE_STALL.as_secs());
            std::thread::sleep(std::time::Duration::from_secs(secs));
            crate::reprintln!("[drill] --inject-stall: the scheduler thread resumes after {secs} s");
        }
    }
}

fn esc(t: &str) -> String {
    // SSE chunks interpolate this inside a JSON string. Hand-escaping only backslashes,
    // quotes and newlines left literal tabs (common in Go), carriage returns and other
    // control characters in the payload. Clients discard that invalid JSON, which looks
    // like the first characters of indented lines were truncated. Use the complete JSON
    // escaping rules (PR #4), then remove the surrounding quotes supplied by the serializer.
    let quoted = serde_json::to_string(t).expect("serializing a Rust string cannot fail");
    quoted[1..quoted.len() - 1].to_string()
}

/// Generation room left in the KV cache after the prompt and the scheduler's decode reserve.
/// None = the prompt (plus reserve) does not fit at all. (PR #4 — mirrors `batch::admit`.)
fn generation_room(max_seq_len: usize, prompt_len: usize, decode_headroom: usize) -> Option<usize> {
    let used = prompt_len.checked_add(decode_headroom)?;
    max_seq_len.checked_sub(used).filter(|&room| room > 0)
}

/// (The think close marker is resolved per-request from the model's vocab — see
/// QwenTokenizer::think_tags. Qwen: `</think>`; hy_v3: `</think:opensource>`.)

/// Longest suffix of `s` that is a proper (partial) prefix of `marker` — text that could be the start
/// of the marker arriving across decode chunks, and so must be held back rather than forwarded.
/// `--output-prompts [cap]` — log the chat-completion call in human-readable form: effective
/// parameters, one line per message turn, and the exact rendered prompt the model sees
/// (excerpt up to `cap` chars; --dump-prompt=1 still writes the full string to /tmp
/// for diffing). Diagnostic output only; nothing here touches the serving path.
#[allow(clippy::too_many_arguments)]
fn log_request_human(
    req: &ChatCompletionRequest,
    effort: Option<&str>,
    prompt: &str,
    prompt_tokens: usize,
    cap: usize,
    model_name: &str,
    render_ms: f64,
) {
    let opt_f32 = |v: &Option<f32>| v.map(|x| x.to_string()).unwrap_or_else(|| "default".into());
    eprintln!("[prompt] ══ chat completion request ({model_name}) ════════════════════════════");
    eprintln!("  stream={}  max_tokens={}  seed={}",
        req.stream,
        req.max_tokens.map(|t| t.to_string()).unwrap_or_else(|| "server-default".into()),
        req.seed.map(|s| s.to_string()).unwrap_or_else(|| "-".into()));
    eprintln!("  temperature={}  top_p={}  top_k={}", opt_f32(&req.temperature), opt_f32(&req.top_p),
        req.top_k.map(|k| k.to_string()).unwrap_or_else(|| "default".into()));
    eprintln!("  penalties: repetition={}  presence={}  frequency={}",
        opt_f32(&req.repetition_penalty), opt_f32(&req.presence_penalty), opt_f32(&req.frequency_penalty));
    eprintln!("  reasoning_effort={} (effective: {})  stop={:?}  include_usage={}",
        req.reasoning_effort.as_deref().unwrap_or("-"),
        effort.unwrap_or("template-default"),
        req.stop,
        req.stream_options.as_ref().map(|s| s.include_usage).unwrap_or(false));
    match &req.tools {
        Some(ts) if !ts.is_empty() => {
            let names: Vec<&str> = ts.iter().filter_map(|t| t.get("function")
                .and_then(|f| f.get("name")).and_then(|n| n.as_str())).collect();
            eprintln!("  tools ({}): {}", ts.len(), names.join(", "));
        }
        _ => eprintln!("  tools: none"),
    }
    eprintln!("  messages ({}):", req.messages.len());
    for (i, m) in req.messages.iter().enumerate() {
        let mut line = format!("    {}. {:9}", i + 1, m.role);
        if let Some(c) = &m.content {
            let flat: String = c.chars().map(|ch| if ch == '\n' { '⏎' } else { ch }).collect();
            let n = flat.chars().count();
            let head: String = flat.chars().take(160).collect();
            line.push_str(&format!(" ({n} ch): {head}{}", if n > 160 { " …" } else { "" }));
        }
        if let Some(tc) = &m.tool_calls {
            let names: Vec<&str> = tc.iter().map(|c| c.function.name.as_str()).collect();
            line.push_str(&format!("  [tool_calls: {}]", names.join(", ")));
        }
        if !m.images.is_empty() { line.push_str(&format!("  [{} image(s)]", m.images.len())); }
        if let Some(id) = &m.tool_call_id { line.push_str(&format!("  [result of {id}]")); }
        eprintln!("{line}");
    }
    let total = prompt.chars().count();
    let trunc = cap.min(total);
    eprintln!("[prompt] rendered prompt: {prompt_tokens} tokens, {total} chars ({render_ms:.1} ms render):");
    let head: String = prompt.chars().take(trunc).collect();
    for l in head.lines() { eprintln!("    | {l}"); }
    if total > trunc {
        eprintln!("    … (+{} more chars of {total} — full dump: --dump-prompt=1)", total - trunc);
    }
    eprintln!("[prompt] ══════════════════════════════════════════════════════════════════");
}

/// The longest proper prefix of `marker` that `s` ENDS with, in bytes (0 = none). A marker may be a request's
/// stop string, so only whole characters of it count as a prefix.
///
/// LINEAR (S-B9-REL-INTEG B1): Knuth-Morris-Pratt over the last `marker.len() - 1` bytes of `s` (nothing
/// earlier can be part of a PROPER prefix of the marker), with the failure table of the prefix of `marker` that
/// tail can reach. The old loop tried every length k from the top and compared k bytes for each: quadratic
/// in the stop string, which a client chooses (1.7 ms per call at 10 KB, 308 ms at 200 KB, per token and per
/// stop string, on a tokio worker). The cost is now O(min(|s|, |marker| - 1)) with no cap and no behaviour
/// change; `partial_overlap_steps` returns the step count the tests assert on.
fn partial_overlap(s: &str, marker: &str) -> usize { partial_overlap_steps(s, marker).0 }

fn partial_overlap_steps(s: &str, marker: &str) -> (usize, u64) {
    let (sb, mb) = (s.as_bytes(), marker.as_bytes());
    if mb.len() < 2 || sb.is_empty() { return (0, 0); }
    let t = &sb[sb.len().saturating_sub(mb.len() - 1)..]; // <= marker.len() - 1 bytes
    let l = t.len();                                       // the automaton never passes state l
    let mut steps = 0u64;
    // failure function of marker[..l]
    let mut fail = vec![0u32; l];
    let mut k = 0usize;
    for i in 1..l {
        while k > 0 && mb[i] != mb[k] { k = fail[k - 1] as usize; steps += 1; }
        if mb[i] == mb[k] { k += 1; }
        fail[i] = k as u32;
        steps += 1;
    }
    // match marker's prefixes against the tail: the final state is the longest prefix that is a suffix of t
    let mut q = 0usize;
    for &b in t {
        while q > 0 && mb[q] != b { q = fail[q - 1] as usize; steps += 1; }
        if mb[q] == b { q += 1; }
        steps += 1;
    }
    // q <= l < marker.len(): a proper prefix. (Over valid UTF-8 the borders are whole characters of the
    // marker; the walk is the guard for a tail that starts inside a character.)
    while q > 0 && !marker.is_char_boundary(q) { q = fail[q - 1] as usize; steps += 1; }
    (q, steps)
}

fn partial_think_overlap(s: &str, marker: &str) -> usize { partial_overlap(s, marker) }

/// S3+S4 (REL_V0_7_3 review): the ONE hold-back decision, called by EVERY SSE emit site
/// (the same-chunk think-close branch, the steady-state content split, the reasoning split).
/// `tool_hold` is None exactly when tool-call parsing is off (a keep-tools "none" request):
/// a stray `<tool_call` then streams as plain content — nothing is held back for tools.
/// `tool_hit` is the caller's scanner result for the tool marker (a GrowFind hit, or a plain
/// `str::find` made absolute; None both when the scanner found nothing and when tools are
/// off). `emit_limit` is the buffer end (`acc.len()`, or `safe` for the reasoning region);
/// `region` is the un-emitted tail the partial-marker checks run over; `partial_markers`
/// lists markers whose SPLIT tail must not leak across an emit boundary (the think-open tag
/// for the content sites; none for reasoning).
fn hold_back_end(
    tool_hold: Option<&'static str>,
    tool_hit: Option<usize>,
    emit_limit: usize,
    region: &str,
    partial_markers: &[&str],
) -> usize {
    let partial = partial_markers.iter().map(|&m| partial_overlap(region, m)).max().unwrap_or(0);
    match tool_hold {
        None => emit_limit - partial,
        Some(m) => match tool_hit {
            Some(i) => i, // a call has started: emit nothing more
            None => emit_limit - partial_overlap(region, m).max(partial),
        },
    }
}

/// WP02: incremental first-occurrence search over a buffer that only GROWS (the streaming `acc`).
/// `find(hay, base, needle)` returns exactly `hay[base..].find(needle).map(|i| base + i)` as long as
/// every `hay` is a prefix of the same growing buffer and `base` never decreases (a decrease falls
/// back to a full scan). It skips what an earlier miss already cleared — no occurrence lies wholly
/// inside [seen_base, seen_upto) — so a long stream costs O(new bytes) per token, not O(n).
#[derive(Default)]
struct GrowFind {
    base: usize,
    upto: usize,
}

impl GrowFind {
    fn find(&mut self, hay: &str, base: usize, needle: &str) -> Option<usize> {
        if needle.is_empty() {
            return Some(base); // str::find("") == Some(0)
        }
        let mut start = base;
        if base >= self.base {
            let known = self.upto.min(hay.len());
            let s = known.saturating_sub(needle.len() - 1);
            if s > start {
                start = s;
                while !hay.is_char_boundary(start) { start -= 1; } // base is a boundary: stops >= base
            }
        }
        let r = hay[start..].find(needle).map(|i| start + i);
        if r.is_none() && (base < self.base || hay.len() > self.upto) {
            self.base = base;
            self.upto = hay.len();
        }
        r
    }
}

/// WP02 liveness: the serving engine's state, read by /health and admission. One engine per
/// process; a backend marks it DEAD when its scheduler can no longer serve (sticky CUDA error,
/// scheduler thread panic). Backends that never set it stay OK.
pub static ENGINE_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(ENGINE_OK);
pub const ENGINE_OK: u8 = 0;
pub const ENGINE_DEAD: u8 = 1;

pub fn engine_ok() -> bool {
    ENGINE_STATE.load(std::sync::atomic::Ordering::Acquire) == ENGINE_OK
}

pub fn set_engine_dead(why: &str) {
    ENGINE_STATE.store(ENGINE_DEAD, std::sync::atomic::Ordering::Release);
    eprintln!("[engine] state DEAD: {why} (/health and new requests answer 503)");
}

/// LR-4 (REL_LONGRUN): `--max-waiting N` (default 256, 0 = unlimited; a negative or non-numeric
/// value is refused at startup, it used to become 256 silently). The HTTP handlers push
/// every accepted request into an UNBOUNDED scheduler channel; once N or more requests wait
/// beyond the lanes, refuse instead of queueing. H8 (REL_V0_7_3): the counter is the Req
/// guard's IN-FLIGHT count — a request counts for its WHOLE handler lifetime (a stalled
/// reader's request stays counted, not just its queue wait), and a burst can overshoot by the
/// requests admitted between the check and the send. Each queued request pins its prompt and
/// messages until a lane frees; the only other guard is the memwatch kill-switch.
pub fn max_waiting() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::ranged(crate::opt!("max-waiting")).unwrap_or_else(|e| fatal_option(&e)) as usize)
}

/// Pure bound arithmetic (unit-tested): `inflight` counts running + waiting (the exact Req-guard
/// counter); with all lanes busy, waiting = inflight - cap.
fn over_wait_limit(inflight: u64, cap: u64, max_waiting: usize) -> bool {
    max_waiting > 0 && inflight >= cap + max_waiting as u64
}

fn admit_gate_busy() -> bool {
    over_wait_limit(crate::metrics::inflight(), crate::metrics::max_batch(), max_waiting())
}

/// The LR-4 refusal: 503 in the engine's existing error shape plus `Retry-After: 5`.
fn server_busy() -> Response {
    crate::metrics::reject_request();
    // MAXBATCH1: the refused arm of the over-cap arrival class (every lane busy AND the waiting
    // queue at --max-waiting) — paired with `velogb10_over_cap_arrivals_total{disposition="queued"}`.
    crate::metrics::over_cap_rejected();
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(axum::http::header::RETRY_AFTER, "5")],
        Json(serde_json::json!({"error": {
            "message": "server busy: too many requests are waiting for a lane; retry shortly",
            "type": "server_error", "code": "server_busy",
        }})),
    ).into_response()
}

/// LR-5 mechanism: the per-request event sender with an UNCONSUMED-EVENT counter. Every `send`
/// increments, the HTTP handler's receive loops decrement, and the scheduler's cancel sweeps
/// read `len()`. (tokio 1.52 exposes `len()` only on the RECEIVER, which lives inside the
/// handler task — exactly the task that is blocked when a stream reader stalls — so the count
/// rides with the sender.) Drop-in for `UnboundedSender<TokEvent>` at every existing call site.
#[derive(Clone)]
pub struct TokTx {
    tx: tokio::sync::mpsc::UnboundedSender<TokEvent>,
    backlog: std::sync::Arc<std::sync::atomic::AtomicIsize>,
}

impl TokTx {
    pub fn send(&self, ev: TokEvent) -> Result<(), tokio::sync::mpsc::error::SendError<TokEvent>> {
        let r = self.tx.send(ev);
        if r.is_ok() { self.backlog.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
        r
    }
    pub fn is_closed(&self) -> bool { self.tx.is_closed() }
    /// Unconsumed events (events sent minus events received by the handler).
    pub fn len(&self) -> usize {
        self.backlog.load(std::sync::atomic::Ordering::Relaxed).max(0) as usize
    }
}

/// The handler-side counter handle (decrement once per received event).
pub type TokBacklog = std::sync::Arc<std::sync::atomic::AtomicIsize>;

pub fn tok_channel()
    -> (TokTx, tokio::sync::mpsc::UnboundedReceiver<TokEvent>, TokBacklog) {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<TokEvent>();
    let backlog: TokBacklog = std::sync::Arc::new(std::sync::atomic::AtomicIsize::new(0));
    (TokTx { tx, backlog: std::sync::Arc::clone(&backlog) }, rx, backlog)
}

/// THE single LR-5 receive mechanism (REL_V0_7_3 review H0/H2/H6): receive the next event and
/// decrement the per-request UNCONSUMED-event counter exactly once. Every handler receive loop
/// goes through this helper (the sync bench/gate drains use `try_recv_counted`) — a loop that
/// forgets the decrement makes a HEALTHY long generation look stalled (the scheduler sweep
/// cancels it at `--stream-backlog-events`), and a loop that decrements twice cancels it at
/// half the limit. Exactly one `fetch_sub` on the backlog may exist in the tree: here.
pub async fn recv_counted(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<TokEvent>,
    bl: &TokBacklog,
) -> Option<TokEvent> {
    counted_event(rx.recv().await, bl)
}

/// Sync twin of `recv_counted` for the bench/gate drains (`run_spec_bench`, `wp24_gate_arm`,
/// `spec_arm`): the same single decrement, `try_recv`-based, so a healthy drain cannot be
/// cancelled by the backlog sweep either.
pub fn try_recv_counted(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<TokEvent>,
    bl: &TokBacklog,
) -> Option<TokEvent> {
    counted_event(rx.try_recv().ok(), bl)
}

fn counted_event(ev: Option<TokEvent>, bl: &TokBacklog) -> Option<TokEvent> {
    if ev.is_some() {
        bl.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
    ev
}

/// LR-5 (REL_LONGRUN): `--stream-backlog-events N` (default 65536, 0 = unlimited; a negative or
/// non-numeric value is refused at startup, it used to become 65536 silently). A stream
/// client that stops reading makes its per-request channel grow for the whole generation; past
/// this many UNCONSUMED events the scheduler cancels the request exactly like a client
/// disconnect (the lane sweep / wire Cancel), freeing the GPU lane.
pub fn stream_backlog_events() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::ranged(crate::opt!("stream-backlog-events")).unwrap_or_else(|e| fatal_option(&e)) as usize)
}

/// Pure predicate (unit-tested); `len` is the TokTx send-minus-receive counter (unconsumed
/// events) — tokio's UnboundedSender exposes no len(); ours rides with the sender.
fn over_backlog(len: usize, limit: usize) -> bool {
    limit > 0 && len >= limit
}

pub fn stream_backlog_exceeded(len: usize) -> bool {
    over_backlog(len, stream_backlog_events())
}

/// WP02: 503 for a request the engine cannot take (DEAD, or its scheduler channel is gone).
fn engine_unavailable() -> Response {
    (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": {
        "message": "engine stopped: the inference engine hit a fatal error and cannot serve; restart the server",
        "type": "server_error", "code": "engine_unavailable",
    }}))).into_response()
}

/// WP02: HTTP status for an engine error reason (busy / stopped = 503, anything else = 500).
fn engine_error_status(reason: &str) -> StatusCode {
    // VIS-2: an engine refusal the CLIENT caused carries its class in the reason
    if reason.starts_with("error: bad request:") { StatusCode::BAD_REQUEST }
    else if reason.starts_with("error: unprocessable:") { StatusCode::UNPROCESSABLE_ENTITY }
    else if reason.contains("busy") || reason.contains("engine stopped") { StatusCode::SERVICE_UNAVAILABLE }
    else { StatusCode::INTERNAL_SERVER_ERROR }
}

/// WP02: the SSE error event for a generation that ended in an engine error (OpenAI SDKs raise
/// on a chunk carrying `error`); the final chunk after it carries a spec finish_reason.
fn sse_error_event(reason: &str) -> String {
    serde_json::json!({"error": {"message": reason, "type": "server_error",
                                 "code": engine_error_status(reason).as_u16()}}).to_string()
}

/// The opening marker of a tool call. While streaming we must never forward this (or a partial prefix
/// of it) to the client as CONTENT: a harness would render raw XML in the chat and never invoke the
/// tool. Once it appears, content emission stops and the rest is buffered for the tool_calls delta.
/// `<tool_call` is the shared PREFIX of qwen's `<tool_call>` and hy_v3's `<tool_call:opensource>` /
/// `<tool_calls:opensource>`, so one constant covers both families.
const TOOL_OPEN: &str = "<tool_call";

/// Does a generation start INSIDE a reasoning block? The RENDERED PROMPT decides: it ends in an open
/// think tag. hy_v3 alone (`family_primed == false`, the `<think:opensource>` family) keeps the effort
/// arm: its low|high template primes `…assistant<think:opensource>` even when the tail check cannot see
/// it. The Qwen templates never open the block from `reasoning_effort` (Qwen3.8's only adds a system
/// instruction), so on a Qwen no-think render (a CLOSED empty block) a low|high effort is a direct
/// answer: Qwen3.5-0.8B/2B (template default: no think) under VeloBench's `reasoning_effort: "low"`
/// streamed the whole answer as reasoning_content with content empty (ISSUES 149).
fn stream_starts_in_reasoning(prompt: &str, think_open: &str, family_primed: bool, effort: Option<&str>) -> bool {
    prompt.trim_end().ends_with(think_open)
        || (!family_primed && matches!(effort, Some("low") | Some("high")))
}

/// Split a completed generation into (reasoning, answer). If the close marker is present, everything
/// before it is reasoning (a leading think-open is stripped) and everything after (trimmed) is the
/// answer. If the marker never appears, the whole text is returned as the answer content.
fn split_think(s: &str, think_open: &str, think_close: &str) -> (Option<String>, String) {
    split_think_in(s, think_open, think_close, true)
}

/// `split_think` that knows where the generation starts. When it starts as CONTENT (the prompt
/// closed the think block: thinking off), a close tag that lies inside a tool-call block is text of
/// that block (an argument that explains reasoning tags, a chat template), not a think boundary:
/// cutting there threw the whole call away, in both response modes (independent review M2). When it
/// starts in reasoning (`starts_in_reasoning`) the first close tag is the boundary as it always was
/// (a call drafted inside reasoning is reasoning).
fn split_think_in(s: &str, think_open: &str, think_close: &str, starts_in_reasoning: bool) -> (Option<String>, String) {
    let mut from = 0usize;
    let found = loop {
        let Some(rel) = s[from..].find(think_close) else { break None };
        let idx = from + rel;
        if think_close.is_empty() { break Some(idx); }
        let inside_call = !starts_in_reasoning && {
            let head = &s[..idx];
            match head.rfind(TOOL_OPEN) { Some(o) => head.rfind("</tool_call").map_or(true, |c| c < o), None => false }
        };
        if !inside_call { break Some(idx); }
        from = idx + think_close.len();
    };
    match found {
        Some(idx) => {
            let mut r = s[..idx].to_string();
            if let Some(rest) = r.strip_prefix(think_open) { r = rest.to_string(); }
            let r = r.trim().to_string();
            let c = s[idx + think_close.len()..].trim_start_matches(['\n', '\r', ' ', '\t']).to_string();
            (if r.is_empty() { None } else { Some(r) }, c)
        }
        None => (None, s.to_string()),
    }
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    /// OpenAI spec requires `model`, but single-model agent clients sometimes omit it. Accept
    /// and fall back to the served model name rather than 422 on a missing field.
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    /// None = not sent: the server's per-mode default applies (AppState::sampling_defaults).
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_top_k")]
    top_k: Option<usize>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    /// WP08: min-p truncation in [0, 1] (0 = off), applied after temperature (the reference implementation's
    /// ComboSampler order). Honoured by the EXL3 serve path; the NVFP4 sampler has none.
    #[serde(default)]
    min_p: Option<f32>,
    /// Optional PRNG seed for reproducible sampling (used by stochastic MTP path).
    #[serde(default)]
    seed: Option<u64>,
    /// OpenAI `logprobs`: return the log probability of every output token
    /// (`choices[].logprobs.content`).
    #[serde(default)]
    logprobs: Option<bool>,
    /// OpenAI `top_logprobs` (0..=20): the most likely alternatives listed per output token.
    /// Requires `logprobs: true`.
    #[serde(default)]
    top_logprobs: Option<u32>,
    /// Stop sequences: accept either a string or a list of strings (OpenAI spec).
    #[serde(default, deserialize_with = "deserialize_stop")]
    stop: Vec<String>,
    /// vLLM-compat: suppress EOS until this many tokens (llama-benchy --exact-tg).
    #[serde(default)]
    min_tokens: Option<usize>,
    /// vLLM-compat: never stop on EOS (--exact-tg).
    #[serde(default)]
    ignore_eos: Option<bool>,
    /// OpenAI tool definitions. Passed straight to the model's chat template, which renders them into
    /// a `# Tools` system block. This field simply did not exist, so serde discarded it and the model
    /// was never told the tools were there -- it answered in prose and every agent harness broke.
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
    /// Accepted and echoed for compatibility. We do not force a call: "required"/named choice would
    /// need constrained decoding, and quietly pretending to honour it is worse than not claiming it.
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
    /// hy_v3 optional reasoning: 'no_think'|'low'|'high', forwarded to the model's chat template.
    /// Per-request override of the server's --reasoning-effort default (which is 'no_think').
    #[serde(default)]
    reasoning_effort: Option<String>,
    /// vLLM's `truncate_prompt_tokens`: keep the LAST n tokens of the prompt (left-truncation)
    /// instead of receiving the over-length 400. Same semantics as /v1/tokenize's field; integer >= 1.
    /// (2026-09-06: this field previously worked on /v1/tokenize ONLY — the chat path silently
    /// dropped it, so the documented escape hatch could not actually rescue an over-length chat.)
    #[serde(default)]
    truncate_prompt_tokens: Option<usize>,
    /// OpenAI streaming options. Only meaningful with stream=true; serde used to drop it silently,
    /// so a client asking for include_usage got nothing and no [DONE] sentinel either.
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    /// Non-standard OpenAI `metadata` object, accepted and passed through. Only one key is
    /// consumed: `session_id` (or `conversation_id`) — the OTel generation-telemetry SESSION
    /// key (see crate::otel::SessionRegistry). Absent/other keys are ignored.
    #[serde(default)]
    metadata: Option<serde_json::Value>,
    /// OpenAI `response_format`. Until Phase 13 this field did not exist on the request struct, so
    /// serde DISCARDED it and every `{"type":"json_schema", ...}` request was silently unconstrained
    /// (the F6 accept-and-ignore class). Now: `json_schema` is compiled into a token-level FSM and
    /// the sampler is masked per step; anything outside the V1 subset is a LOUD 400 naming the
    /// offending keyword. `{"type":"text"}`/absent = unconstrained, exactly as before.
    #[serde(default)]
    response_format: Option<serde_json::Value>,
    /// vLLM/SGLang-compatible `chat_template_kwargs`: an ARBITRARY dict forwarded to the model's
    /// own chat template (W1, Phase 13: the customer's `{"enable_thinking": false}` used to be
    /// dropped by serde and the model kept thinking). `enable_thinking` (bool) selects the
    /// template's thinking / no-think branch; every other key passes through verbatim. A
    /// non-object value, or a non-bool `enable_thinking`, is a loud 400 — never accepted-and-ignored.
    #[serde(default)]
    chat_template_kwargs: Option<serde_json::Value>,
    /// OpenAI's current name for the output cap (`max_tokens` is the deprecated alias). When both
    /// are sent it WINS, as in vLLM (`max_completion_tokens or max_tokens`). It used to be dropped
    /// by serde, so a client sending max_completion_tokens=65536 next to a default max_tokens was
    /// silently capped at the latter.
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    /// Every top-level field this server does not read, kept only so it can be LOGGED by name:
    /// an accepted-and-ignored parameter must be visible in the log, never silent.
    #[serde(flatten)]
    unused_fields: serde_json::Map<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

/// S-B9-REL-SMALLFIXES item 6: does a streamed /v1/completions request get the usage chunk?
/// Only when it asks (`stream_options.include_usage: true`, the OpenAI rule). The chat path sends
/// it unless told not to; here an unasked chunk with `choices: []` would be new bytes in the
/// stream of every existing client of this endpoint, so the default stays "no chunk".
fn completions_include_usage(opts: Option<&StreamOptions>) -> bool {
    opts.map_or(false, |o| o.include_usage)
}

/// S-B9-REL-SMALLFIXES item 6: the stream-usage chunk of the legacy completions SSE path — the
/// chat path's chunk in this path's object type: empty `choices`, top-level `usage` with
/// `prompt_tokens_details.cached_tokens` (the prefix-cache-served prompt tokens, #8.2), and the
/// `timings` extension. Sent after the finish chunk and before `[DONE]`.
fn completions_usage_chunk(id: &str, created: i64, model: &str, prompt_tokens: usize,
                           completion_tokens: usize, cached_tokens: usize,
                           timings: crate::Timings) -> serde_json::Value {
    serde_json::json!({
        "id": id, "object": "text_completion.chunk", "created": created, "model": model,
        "choices": [],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens,
                  "prompt_tokens_details": {"cached_tokens": cached_tokens}},
        "timings": timings,
    })
}

fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.8 }
fn default_top_k() -> usize { 20 }

/// `top_k`: a non-negative integer, or any negative integer for "no top-k truncation" (the
/// OpenAI-compatible convention, e.g. `-1`); mapped to 0, which the samplers treat as off.
fn deserialize_top_k<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<usize>, D::Error> {
    use serde::Deserialize;
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum TopK { Unsigned(u64), Signed(i64) }
    match Option::<TopK>::deserialize(d)? {
        None => Ok(None),
        Some(TopK::Signed(k)) if k < 0 => Ok(Some(0)),
        Some(TopK::Signed(k)) => usize::try_from(k).map(Some).map_err(serde::de::Error::custom),
        Some(TopK::Unsigned(k)) => usize::try_from(k).map(Some).map_err(serde::de::Error::custom),
    }
}

fn deserialize_stop<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    use serde::Deserialize;
    let v: serde_json::Value = serde_json::Value::deserialize(d)?;
    Ok(match v {
        serde_json::Value::Null => vec![],
        serde_json::Value::String(s) => vec![s],
        serde_json::Value::Array(a) => a.into_iter().filter_map(|x| x.as_str().map(String::from)).collect(),
        _ => vec![],
    }
    // S-B9-REL-INTEG B2: an EMPTY stop string matches at offset 0 of every text, which emptied the reply
    // (non-streaming did it before the stream fix too); it stops nothing, so it is not a stop string.
    .into_iter().filter(|s| !s.is_empty()).collect())
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
    /// llama.cpp-compatible timing block, emitted as a top-level extension field (strict
    /// clients ignore unknown top-level fields).
    timings: Timings,
    /// OTel generation-telemetry SESSION key (extension; None when telemetry is off).
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
}

#[derive(Serialize)]
struct ResponseMessage {
    role: String,
    /// null when the turn is purely a tool call -- that is what OpenAI does, and harnesses key on it.
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ResponseMessage,
    finish_reason: String,
    /// vLLM's extension field: WHY a `stop` happened when it was not a stop token/string —
    /// here only "loop_detected" (WP08). Omitted otherwise, so normal replies are unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<String>,
    /// `{"content": [...]}` when the request asked for `logprobs`; omitted otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<serde_json::Value>,
}

/// Max alternatives per token for `top_logprobs` (chat) / `logprobs` (completions).
const MAX_TOP_LOGPROBS: u32 = 20;

/// One output token's log probabilities as received from the engine (`TokEvent::Logprob`),
/// aligned with the token list; `None` = the engine sent none for that token.
type TokLp = Option<(f32, Vec<(u32, f32)>)>;

/// The chat request's logprobs setting -> the engine's `BatchRequest::logprobs`.
fn chat_logprobs_param(logprobs: Option<bool>, top: Option<u32>, stream: bool) -> Result<Option<u8>, String> {
    let n = match (logprobs.unwrap_or(false), top) {
        (false, None) => return Ok(None),
        (false, Some(_)) => return Err("top_logprobs requires logprobs to be true".into()),
        (true, Some(n)) if n > MAX_TOP_LOGPROBS =>
            return Err(format!("top_logprobs must be between 0 and {MAX_TOP_LOGPROBS} (got {n})")),
        (true, n) => n.unwrap_or(0) as u8,
    };
    if stream { return Err("logprobs are not supported with stream=true yet".into()); }
    Ok(Some(n))
}

/// The completions request's legacy `logprobs` -> the engine's `BatchRequest::logprobs`.
fn completion_logprobs_param(logprobs: Option<u32>, stream: bool) -> Result<Option<u8>, String> {
    match logprobs {
        None => Ok(None),
        Some(n) if n > MAX_TOP_LOGPROBS => Err(format!("logprobs must be between 0 and {MAX_TOP_LOGPROBS} (got {n})")),
        Some(_) if stream => Err("logprobs are not supported with stream=true yet".into()),
        Some(n) => Ok(Some(n as u8)),
    }
}

/// The output tokens that carry log probabilities for bytes `range` of the decoded text: every
/// generated token except stop tokens that overlaps the range (a token that prints nothing counts
/// when its position lies inside it). Returns (index, byte offset of the token in the decoded
/// text, the token's bytes clipped to the range).
fn logprob_token_span(bytes_of: impl Fn(u32) -> Vec<u8>, stop_ids: &[u32], tokens: &[u32],
                      range: std::ops::Range<usize>) -> Vec<(usize, usize, Vec<u8>)> {
    let mut off = 0usize;
    let mut out = Vec::with_capacity(tokens.len());
    for (i, &t) in tokens.iter().enumerate() {
        if off >= range.end { break; }
        let b = bytes_of(t);
        let end = off + b.len();
        let inside = if b.is_empty() { off >= range.start } else { end > range.start };
        if inside && !stop_ids.contains(&t) {
            let lo = range.start.saturating_sub(off).min(b.len());
            let hi = (range.end - off).min(b.len());
            out.push((i, off, b[lo..hi].to_vec()));
        }
        off = end;
    }
    out
}

/// The byte range of the decoded `text` that the response's `content` came from: `content` as
/// found after the reasoning block (after the first `think_close`, when one was generated).
/// None when the content is not one contiguous span of the text.
fn content_byte_range(text: &str, content: &str, think_close: &str) -> Option<std::ops::Range<usize>> {
    let from = text.find(think_close).map_or(0, |p| p + think_close.len());
    if content.is_empty() { return Some(from..from); }
    let p = from + text[from..].find(content)?;
    Some(p..p + content.len())
}

/// OpenAI chat `logprobs`: `{"content": [{token, logprob, bytes, top_logprobs: [...]}]}` for the
/// tokens of bytes `range` of the decoded text (the response's `content`); a token cut by the
/// range edge is listed with only its bytes inside the range, its probabilities unchanged. None
/// when a kept token has no engine entry (a readback failure: reported as null, never guessed).
fn chat_logprobs_json(tok: &QwenTokenizer, stop_ids: &[u32], tokens: &[u32], lps: &[TokLp],
                      range: std::ops::Range<usize>) -> Option<serde_json::Value> {
    let entry = |b: Vec<u8>, lp: f32| {
        serde_json::json!({"token": String::from_utf8_lossy(&b), "logprob": lp, "bytes": b})
    };
    let mut content = Vec::new();
    for (i, _, b) in logprob_token_span(|t| tok.token_bytes(t, true), stop_ids, tokens, range) {
        let (lp, top) = lps.get(i)?.as_ref()?;
        let mut e = entry(b, *lp);
        e["top_logprobs"] = serde_json::Value::Array(
            top.iter().map(|&(id, l)| entry(tok.token_bytes(id, false), l)).collect());
        content.push(e);
    }
    Some(serde_json::json!({"content": content}))
}

/// Legacy completions `logprobs`: `{tokens, token_logprobs, top_logprobs, text_offset}` with
/// character offsets into `text` (the decoded completion). A `top_logprobs` object keeps the
/// most likely entry when two alternatives print the same string.
fn completion_logprobs_json(tok: &QwenTokenizer, stop_ids: &[u32], tokens: &[u32], lps: &[TokLp],
                            text: &str) -> Option<serde_json::Value> {
    let (mut ts, mut tl, mut tt, mut to) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (i, off, _) in logprob_token_span(|t| tok.token_bytes(t, true), stop_ids, tokens, 0..usize::MAX) {
        let (lp, top) = lps.get(i)?.as_ref()?;
        ts.push(String::from_utf8_lossy(&tok.token_bytes(tokens[i], false)).into_owned());
        tl.push(*lp);
        let mut m = serde_json::Map::new();
        for &(id, l) in top {
            let k = String::from_utf8_lossy(&tok.token_bytes(id, false)).into_owned();
            m.entry(k).or_insert(serde_json::json!(l));
        }
        tt.push(serde_json::Value::Object(m));
        let mut b = off.min(text.len());
        while !text.is_char_boundary(b) { b -= 1; }
        to.push(text[..b].chars().count());
    }
    Some(serde_json::json!({"tokens": ts, "token_logprobs": tl, "top_logprobs": tt, "text_offset": to}))
}

/// The scheduler's internal backstop reason (batch.rs) is not an OpenAI value — the spec is
/// stop|length|tool_calls|content_filter. Map it to what it means: generation ran out of room.
fn spec_finish_reason(reason: &str) -> &str {
    match reason {
        "context_length_exceeded" => "length",
        // WP08: the loop detector ends a response as a normal stop (the reference implementation's eos_reason
        // "loop_detected"); the reason itself rides `stop_reason`.
        "loop_detected" => "stop",
        // WP02: an engine error / cancel is not a spec value; the SSE error event carries the reason
        r if r.starts_with("error") || r == "cancelled" => "stop",
        _ => reason,
    }
}

/// WP08: the `stop_reason` extension for a scheduler finish reason (None = omit the field).
fn stop_reason_of(reason: &str) -> Option<String> {
    (reason == "loop_detected").then(|| reason.to_string())
}

/// S-B9-REL-SMALLFIXES item 5(c): WHY a generation ended, for the `[gen]` line. The API keeps
/// `finish_reason: "stop"` for a loop-detector stop (client compatibility) and a tool call rewrites
/// it to "tool_calls", so the line's `finish=` cannot tell these apart; `stop_kind=` can.
/// `sched_reason` = the scheduler's own Finish reason (None = the channel closed without one: the
/// engine stopped), `stop_string_hit` = a request stop string matched on the HTTP side (the
/// scheduler never sees that: its lane is cancelled).
///   eos = a model stop token | stop_string | length (max_tokens or the context end) | loop (the
///   loop detector) | cancel | error | other (a reason this function does not know)
pub(crate) fn stop_kind(sched_reason: Option<&str>, stop_string_hit: bool) -> &'static str {
    if stop_string_hit { return "stop_string"; }
    match sched_reason {
        Some("stop") => "eos",
        Some("length") | Some("context_length_exceeded") => "length",
        Some("loop_detected") => "loop",
        Some("cancelled") => "cancel",
        Some(r) if r.starts_with("error") => "error",
        Some(_) => "other",
        None => "error",
    }
}

/// S-B9-REL-SURFACE2 item 8: the (reasoning, content) of a NON-streaming chat generation that ended
/// INSIDE the think block its prompt opened: no close tag in `text` (the caller checks both).
/// `split_think` returns such a text wholly as content, so the same request answered
/// `content: <the reasoning>` without `stream` and `reasoning_content` deltas with it. This is the
/// streaming path's split, applied to the finished text:
///   - the text up to the first tool-call marker (all of it when there is none, or when tool
///     parsing is off for the request: `hold_tools` false) is reasoning, trimmed, None when empty;
///   - what starts at the marker is the content the tool parser then reads: parsed calls become
///     `tool_calls`, an unparsed call-looking block stays visible as content (the stream's
///     held-back remainder); with no marker the content is empty.
fn unclosed_think_split(text: &str, hold_tools: bool) -> (Option<String>, String) {
    let cut = if hold_tools { text.find(TOOL_OPEN).unwrap_or(text.len()) } else { text.len() };
    let r = text[..cut].trim();
    (if r.is_empty() { None } else { Some(r.to_string()) }, text[cut..].to_string())
}

/// `--unclosed-think content`: keep the v0.7.3 response of a generation that ended inside the think
/// block (the reasoning returned as `content`); `reasoning` = `unclosed_think_split`. The default
/// is the registry row's (the owner's ruling of 2026-10-09: `reasoning`; read by `opts::choice`).
fn unclosed_think_as_content() -> bool {
    crate::opts::choice(crate::opt!("unclosed-think")) == "content"
}

/// S-B9-REL-SMALLFIXES item 5(b): the (reasoning, content) pair the `[gen]` line counts.
/// `split_think` returns a text WITHOUT a close tag wholly as content. When the prompt opened the
/// think block (`starts_in_reasoning`) and the model never closed it (`closed` = the close tag is
/// in the generated text), the request ended INSIDE the block: that text is reasoning. Log-only —
/// the response's own fields are not touched.
pub(crate) fn gen_log_spans<'a>(reasoning: Option<&'a str>, content: &'a str, closed: bool,
                                starts_in_reasoning: bool) -> (Option<&'a str>, &'a str) {
    if starts_in_reasoning && !closed && reasoning.is_none() && !content.is_empty() {
        (Some(content), "")
    } else {
        (reasoning, content)
    }
}

/// S-B9-REL-SMALLFIXES item 5(a): the `stop_p=.. stop_rank=..` tail of the `[gen]` line
/// (`na` = no scheduler sent it: not a stop-token finish, an engine without the readback, TP).
/// S-B9-REL-SURFACE2 item 1: `stop_p_law=greedy|sampled` follows them (appended, so the older
/// keys keep their order) and names the law `stop_p` is a probability under: `sampled` = the
/// truncated, temperature-scaled law the sampler drew from; `greedy` = the softmax at T=1.
pub(crate) fn stop_info_fields(info: Option<(f32, u32, bool)>) -> String {
    match info {
        Some((p, rank, sampled)) => format!("stop_p={p:.4} stop_rank={rank} stop_p_law={}",
                                            if sampled { "sampled" } else { "greedy" }),
        None => "stop_p=na stop_rank=na stop_p_law=na".to_string(),
    }
}

/// WP08: validate a request's penalty / min_p values (the OpenAI/vLLM 400 contract; semantics are
/// the reference implementation's ComboSampler, WP15). repetition_penalty must be finite and > 0 — a value < 1
/// REWARDS repetition and is accepted as the reference implementation accepts it, with a log warning (owner decision
/// 2026-09-26); presence/frequency_penalty finite in [-2, 2]; min_p finite in [0, 1].
/// Err = the client-facing message.
pub fn validate_penalties(rep: Option<f32>, pres: Option<f32>, freq: Option<f32>,
                          min_p: Option<f32>) -> Result<(), String> {
    if let Some(r) = rep {
        if !r.is_finite() || r <= 0.0 {
            return Err(format!("repetition_penalty must be a finite number > 0 (got {r})"));
        }
        if r < 1.0 {
            eprintln!("[req] WARNING: repetition_penalty {r} < 1 rewards repetition (accepted, as the reference engine does)");
        }
    }
    for (name, v) in [("presence_penalty", pres), ("frequency_penalty", freq)] {
        if let Some(x) = v {
            if !x.is_finite() || !(-2.0..=2.0).contains(&x) {
                return Err(format!("{name} must be a number in [-2, 2] (got {x})"));
            }
        }
    }
    if let Some(m) = min_p {
        if !m.is_finite() || !(0.0..=1.0).contains(&m) {
            return Err(format!("min_p must be a number in [0, 1] (got {m})"));
        }
    }
    Ok(())
}

/// WP08: a request body that does not parse (wrong JSON types, missing required fields) is the
/// client's error — 400 invalid_request_error, as OpenAI/vLLM answer — never axum's bare 422.
fn bad_json(e: axum::extract::rejection::JsonRejection) -> Response {
    (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
        "message": e.body_text(), "type": "invalid_request_error", "code": "invalid_request_body",
    }}))).into_response()
}

fn bad_param(msg: String) -> Response {
    (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
        "message": msg, "type": "invalid_request_error", "code": "invalid_sampling_parameter",
    }}))).into_response()
}

/// The chat-template reasoning effort for THIS request: the request's per-call override, else the
/// server's `--reasoning-effort` default, normalized onto THIS model family's template vocabulary.
///
/// SHARED by /v1/chat/completions and the `messages` mode of /v1/tokenize — the two must render the
/// SAME prompt for the same request, or the tokenize-side count diverges from the chat-side
/// `usage.prompt_tokens` (the exact invariant a token-counting client measures).
///
/// Two families, two vocabularies:
///   - hy_v3's template accepts `no_think|low|high` (default low), and its Rust dsv4 path treats
///     None/"" as low.
///   - Qwen3.5's template accepts `xhigh|medium|low` (default xhigh) and RAISES on anything else.
/// Forward the client's value verbatim when it is valid for THIS family's template; convert
/// the OpenAI API convention onto the nearest native level. When NEITHER the request nor
/// --reasoning-effort specifies one, pass None so the model's own template default wins
/// (xhigh for Qwen 3.8, low for hy_v3) — never a hardcoded guess.
///
/// Qwen 3.8 native (the ONLY values its template accepts; anything else raises => 500):
///   xhigh (default) | medium | low          ["no_think"/"off" => enable_thinking=false]
/// OpenAI API -> Qwen 3.8 (owner spec 2026-08-30):
///   none    -> thinking off      (latency-critical; no reasoning)
///   low     -> low               (efficient reasoning)
///   medium  -> medium            (balanced; OpenAI's default)
///   high    -> xhigh             (hard reasoning)
///   xhigh   -> xhigh             (deep research)
///   max     -> xhigh             (maximum)
/// hy_v3 native: no_think | low | high  =>  none->no_think, low->low, medium/high/xhigh/max->high
///
/// REGRESSION FIX (2026-08-30): the 289e1a1 refactor lumped "high" into the no_think arm,
/// so every OpenAI-convention client sending reasoning_effort=high silently LOST thinking.
fn resolve_reasoning_effort(tokenizer: &QwenTokenizer, req_effort: Option<&str>,
                            server_default: Option<&str>) -> Option<String> {
    let (_, think_close_tag, _) = tokenizer.think_tags();
    let hy_family = think_close_tag != "</think>";
    req_effort.or(server_default).map(|e| {
        let n = match (e, hy_family) {
            ("high", true) | ("medium", true) | ("xhigh", true) | ("max", true) => "high",
            ("high", false) | ("xhigh", false) | ("max", false) => "xhigh",
            ("low", _) | ("medium", false) => e,
            ("no_think", _) | ("none", _) | ("minimal", _) | ("off", _) | ("", _) => "no_think",
            (other, _) => other,
        };
        if n != e {
            crate::reprintln!("[req] reasoning_effort '{e}' normalized to '{n}' for this model family");
        }
        n.to_string()
    })
}

/// W2 (Phase 13), 2026-09-14: a `response_format` request this build serves WITHOUT enforcing the
/// schema must never be silent — but it must also never be REFUSED. `e737d08` refunded such
/// requests with HTTP 400 to avoid the "F6 class" (accepted-and-quietly-ignored); that intent was
/// right and the mechanism was wrong: a 400 is a hard failure of a request that used to succeed,
/// and it cost real measured quality on the public tool-eval-bench (TC-64..TC-69: "not valid JSON"
/// → 400; 88 → 85). The honest form is to SERVE the request and ADVERTISE that it is
/// unconstrained: every reply carries `x-json-schema-enforced: none`, and the reason is logged
/// loudly once per distinct reason. A client that needs the guarantee checks the header.
/// LR-8 (REL_LONGRUN): the dedup set is CAPPED so a client sending endless distinct invalid
/// tool schemas cannot grow host memory. Under the cap: warn once per distinct reason (the
/// original behaviour). At/over the cap: never warn, never grow, and say so exactly once.
const SCHEMA_SEEN_CAP: usize = 1024;

/// Pure decision core (unit-tested): returns Some(true) to warn (new reason), Some(false) to
/// stay silent (duplicate), None once the cap is hit ("further warnings suppressed").
fn schema_seen_note(
    seen: &mut std::collections::HashSet<String>,
    suppressed: &std::sync::atomic::AtomicBool,
    why: &str,
) -> Option<bool> {
    use std::sync::atomic::Ordering;
    if seen.contains(why) { return Some(false); }
    if seen.len() >= SCHEMA_SEEN_CAP {
        if !suppressed.swap(true, Ordering::Relaxed) { return None; }
        return Some(false);
    }
    seen.insert(why.to_string());
    Some(true)
}

fn warn_unenforced_schema_once(why: &str) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    static SUPPRESSED: OnceLock<std::sync::atomic::AtomicBool> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let suppressed = SUPPRESSED.get_or_init(|| std::sync::atomic::AtomicBool::new(false));
    // H10 (REL_V0_7_3): a POISONED mutex used to fall into the None arm and print the
    // cap-suppression notice on EVERY call. Recover the inner set and keep the normal
    // decision (warn once per distinct reason) instead.
    let verdict = {
        let mut g = seen.lock().unwrap_or_else(|p| p.into_inner());
        schema_seen_note(&mut g, suppressed, why)
    };
    match verdict {
        Some(true) => {}                 // new reason: warn below
        Some(false) => return,           // duplicate: silent
        None => {                        // cap reached, first time: the suppression notice
            eprintln!(
                "[schema] {} distinct unenforced-schema reasons recorded — further first-warnings suppressed (set capped at {SCHEMA_SEEN_CAP}; memory bounded)",
                SCHEMA_SEEN_CAP
            );
            return;
        }
    }
    {
        eprintln!(
            "[schema] WARNING: response_format is NOT enforced by this build — the reply is \
             unconstrained and may not match the requested schema (responses carry \
             'x-json-schema-enforced: none'). Reason: {why}"
        );
    }
}

/// Marks a served response whose `response_format` was not enforced (see above).
fn attach_schema_unenforced_header(resp: &mut Response, unenforced: bool) {
    if !unenforced { return; }
    if let Ok(v) = HeaderValue::from_str("none") {
        resp.headers_mut().insert("x-json-schema-enforced", v);
    }
}

/// S-B9-REL-NONETOOLS: the tool_choice render/parse decision, extracted pure so the
/// OFF-byte-identity and ON-prefix-sharing contracts are unit-testable without a model.
/// Returns (tools_for_template, force_line_on_last_message, parse_reply_as_tool_calls).
///
/// Flag off (the default) this is EXACTLY the historical decision: "none" drops the tools from
/// the template and forces nothing; "required"/a named function append their forcing line; auto
/// forces nothing. Flag on, a "none" request that carries tools keeps the IDENTICAL tools block
/// (the same slice — nothing reordered or re-serialized), gets a do-not-call line on the last
/// message, and its reply is NEVER parsed into tool_calls. `parse_as_tool_calls == false` is the
/// single value both response modes and the streaming hold-back follow; parse() alone cannot
/// express it (it still extracts an XML call at tools=None — only the JSON form checks the
/// offered list), so the callers skip parse entirely.
/// S4 (REL_V0_7_3 review): the server's ONE force-line append, extracted from the handler
/// so the test drives the REAL code (the old test re-implemented the append, so it could not
/// fail). Appends to the LAST message: recency dominates instruction-following; a system-
/// level line is routinely outweighed by the turn's own phrasing.
fn apply_force_line(messages: &mut [crate::tokenizer::ChatMessage], line: &str) {
    if let Some(last) = messages.last_mut() {
        match &mut last.content {
            Some(c) => { c.push_str("\n\n"); c.push_str(line); }
            None => { last.content = Some(line.to_string()); }
        }
    }
}

fn tool_choice_decision<'a>(
    tool_choice: Option<&serde_json::Value>,
    tools: Option<&'a [serde_json::Value]>,
    keep_tools_when_none: bool,
) -> (Option<&'a [serde_json::Value]>, Option<String>, bool) {
    let forced_fn: Option<String> = match tool_choice {
        Some(v) => {
            let s = v.as_str().unwrap_or("");
            if s == "none" { Some("__none__".to_string()) }
            else if s == "required" || s == "auto" { None }  // auto = no forcing
            else { v.pointer("/function/name").and_then(|n| n.as_str()).map(|n| n.to_string()) }
        }
        None => None,
    };
    let none_keeps_tools = forced_fn.as_deref() == Some("__none__") && tools.is_some() && keep_tools_when_none;
    let tools_for_template = if forced_fn.as_deref() == Some("__none__") && !none_keeps_tools { None } else { tools };
    let force_line = match forced_fn.as_deref() {
        Some("__none__") => if none_keeps_tools {
            // The tools stay in the prompt (prefix reuse); forbid them explicitly instead.
            Some("IMPORTANT: Do not call any tools. Answer in plain text only.".to_string())
        } else {
            None // tools already removed; nothing to force (legacy default)
        },
        Some(name) => Some(format!("IMPORTANT: You MUST call the function `{name}` with appropriate arguments before answering. Do not answer in plain text.")),
        None if tool_choice.and_then(|v| v.as_str()) == Some("required") =>
            Some("IMPORTANT: You MUST use one of the provided tools to answer. Do not answer in plain text.".to_string()),
        None => None,
    };
    (tools_for_template, force_line, !none_keeps_tools)
}

/// Everything the chat stream's generator owns. The handler builds it from the request; the
/// unit tests build it with a scripted event channel and a table decoder, and drive the same
/// generator (`chat_sse_stream`) the server runs.
pub(crate) struct ChatSse {
    pub rx: mpsc::UnboundedReceiver<TokEvent>,
    pub bl: TokBacklog,
    pub mt: crate::metrics::Req,
    pub otel_req: Option<crate::otel::OtelRequest>,
    pub otel_session: Option<String>,
    pub role_chunk: String,
    pub completion_id: String,
    pub created: i64,
    pub model_name: String,
    pub stops: Vec<String>,
    pub think_open: &'static str,
    pub think_close: &'static str,
    pub starts_in_reasoning: bool,
    pub tool_hold: Option<&'static str>,
    pub parse_as_tool_calls: bool,
    pub req_tools: Option<Vec<serde_json::Value>>,
    pub include_usage: bool,
    pub prompt_len: usize,
    pub t0: std::time::Instant,
    /// The `[gen]` log line's inputs (None = no log line: the unit tests have no tokenizer).
    pub gen_log: Option<GenLog>,
    /// TOOLEVAL: the armed per-turn tool trace (None = flag off). Read once at the finalize
    /// site; nothing in the token loop touches it.
    pub tool_trace: Option<crate::tool_trace::Turn>,
    /// TOOLSCRUB1: the per-request emission-side scrubber (off = passthrough that never
    /// buffers; see src/tool_scrub.rs).
    pub scrub: crate::tool_scrub::Scrubber,
}

pub(crate) struct GenLog {
    pub tokenizer: Arc<QwenTokenizer>,
    pub stop_ids: Vec<u32>,
    pub effort: Option<String>,
    pub presence_penalty: f32,
    pub pp_source: &'static str,
}

/// `--dump-tools`: the WHOLE text the tool-call parser was given for one request, as one log line
/// `[dump-tools] {"raw": "..."}` (a JSON object, so the text survives the log byte for byte). The
/// `[req] raw model output` line next to it stops at 1200 characters, which cuts a call that
/// writes a file: a cut block has no close tag and the parser drops it, so that line cannot be
/// replayed (S-B9-REL-SURFACE2B R1, the offline replay: examples/toolparse_replay.rs).
fn dump_tools_line(raw: &str) -> String {
    format!("[dump-tools] {}", serde_json::json!({ "raw": raw }))
}

/// The text a finished stream has not sent yet, as (reasoning, content): `acc[emitted..]`, where
/// `emitted` is the live emit cursor (the larger of the reasoning and the content cursor: in
/// reasoning mode the content cursor stays 0, and using it alone would re-send the whole reply).
/// Called only when no tool call was parsed, so all of it is text.
///  * In content mode it is content.
///  * Inside a think block that was never closed (`in_reasoning`) it is reasoning, up to a held
///    tool-call marker; from the marker on it is content: a call-looking block the parser could
///    not read stays visible as content, which is what this stream always did with it.
///  * A think block that opened AND closed inside the reply's last piece (the emit sites hand
///    over to reasoning at the open tag and look at the rest only with the next piece, which
///    never came): reasoning up to the close tag, content after it; the tag itself is not sent.
/// Empty when everything was sent (also when a stop string cut the buffer below the cursor).
fn held_tail<'a>(acc: &'a str, emitted: usize, in_reasoning: bool, tool_hold: Option<&str>, think_close: &str)
    -> (&'a str, &'a str) {
    if emitted >= acc.len() { return ("", ""); }
    let tail = &acc[emitted..];
    if !in_reasoning { return ("", tail); }
    let marker = tool_hold.and_then(|m| tail.find(m)).unwrap_or(tail.len());
    match tail.find(think_close).filter(|&c| !think_close.is_empty() && c < marker) {
        Some(c) => (&tail[..c], tail[c + think_close.len()..].trim_start_matches(['\n', '\r', ' ', '\t'])),
        None => tail.split_at(marker),
    }
}

fn tool_calls_chunk(cid: &str, created: i64, model: &str, calls: &[ToolCall]) -> String {
    let arr: Vec<serde_json::Value> = calls.iter().enumerate().map(|(i, c)| serde_json::json!({
        "index": i, "id": c.id, "type": c.kind,
        "function": {"name": c.function.name, "arguments": c.function.arguments},
    })).collect();
    serde_json::json!({
        "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": {"tool_calls": arr}, "finish_reason": null}],
    }).to_string()
}

/// The `/v1/chat/completions` SSE stream: every data event of a streamed chat response, in
/// order. `make_decoder` builds the per-request token decoder (token id -> text piece) when the
/// stream is first polled.
///
/// THE ROLE-CHUNK CONTRACT (S-B9-REL-BENCHFIX / BENCHFIX2). The role chunk is the first chunk
/// with a `choices` entry. It leaves
///   * when the request's first token has been GENERATED (the first `TokEvent::Tok`), whether
///     or not that token produces a delta (a tool call being buffered and a held-back marker
///     prefix produce none until later), or
///   * immediately before the terminal events (error event / finish chunk) of a request that
///     generates no token (an engine error before the first token, a finish without a token,
///     an engine that went away),
/// and never before. So a client that takes "first chunk with choices" as "first output"
/// (llama-benchy's ttfr, most TTFT tooling) reads the time of the first generated token: not a
/// few milliseconds (the role chunk used to be yielded at stream start, before the prefill:
/// 2,048 prompt tokens / 4.6 ms = "443,198 tok/s" in a public benchmark table), and not the end
/// of the generation either (a reply that is only a tool call yields its first delta at the
/// end). The data events keep their bytes and their order: role chunk first, then exactly the
/// deltas, error event, finish chunk, usage chunk and `[DONE]` as before. While the generator
/// is silent the response carries SSE comment frames only (`sse_with_keepalive`).
pub(crate) fn chat_sse_stream<F, D>(cx: ChatSse, make_decoder: F)
    -> impl Stream<Item = Result<Event, axum::Error>> + Send + 'static
where
    F: FnOnce() -> D + Send + 'static,
    D: FnMut(u32) -> String + Send + 'static,
{
    let ChatSse {
        mut rx, bl, mut mt, otel_req, otel_session, role_chunk, completion_id, created, model_name,
        stops, think_open, think_close, starts_in_reasoning, tool_hold, parse_as_tool_calls,
        req_tools, include_usage, prompt_len, t0, gen_log, tool_trace, mut scrub,
    } = cx;
    // B2: an empty stop string matches at offset 0 of every text (it would empty the reply); the request
    // deserializer drops it already, this keeps the stream itself safe for any caller.
    let stops: Vec<String> = stops.into_iter().filter(|s| !s.is_empty()).collect();
    async_stream::stream! {
        // event=stream_start, token.index 0: recorded when the stream is first polled, as it
        // always was (the telemetry timeline is unchanged). Its body is the role chunk this
        // stream sends with its first token; a stream the client drops before that records
        // the stream_start but never puts the role chunk on the wire.
        if let Some(r) = &otel_req { r.start(&role_chunk); }
        let mut role_pending = Some(role_chunk);
        let mut stream_dec = make_decoder();
        let mut acc = String::new();
        let mut n = 0usize;
        let mut cached = 0usize; // v0.7.3 (#8.2): prefix-cache-served prompt tokens
        let mut last_tok: Option<u32> = None;
        let mut stop_hit = false;
        // [gen] line (item 5): the scheduler's own finish reason and its stop-token statistics
        let mut sched_reason: Option<String> = None;
        let mut stop_info: Option<(f32, u32, bool)> = None;
        // WP02: a channel closed without a Finish is an engine failure, not "length"
        let mut finish = "error: engine stopped".to_string();
        let mut first_tok: Option<std::time::Instant> = None;
        // WP02: tail-window scans (O(new bytes) per token, identical results — see GrowFind)
        let mut f_close = GrowFind::default();
        let mut f_tool_r = GrowFind::default();
        let mut f_open = GrowFind::default();
        let mut f_tool_c = GrowFind::default();
        let mut f_stops: Vec<GrowFind> = stops.iter().map(|_| GrowFind::default()).collect();
        // Thinking-model split: qwen's prompt is primed with `<think>\n`, so the generated stream
        // is `…reasoning…</think>\n\nanswer`. Pre-close text -> reasoning_content, post-close
        // -> content. hy_v3's no_think prompt already closed the (empty) block, so it starts as
        // content. The close marker may span decode chunks, so we hold back a tail that could be
        // its prefix until more text arrives.
        let mut content_start: Option<usize> = if starts_in_reasoning { None } else { Some(0) };
        let mut reason_emitted: usize = 0;
        let mut content_emitted: usize = 0;
        // Markers whose split tail must not leave before the next token decides it: the
        // think-open tag (content sites) and the request's stop strings (every site). A stop
        // string ends the reply BEFORE itself, so a tail that could be the start of one is held.
        let reason_marks: Vec<&str> = stops.iter().map(String::as_str).collect();
        let content_marks: Vec<&str> = std::iter::once(think_open).chain(reason_marks.iter().copied()).collect();
        while let Some(ev) = recv_counted(&mut rx, &bl).await { // LR-5: the one decrement (H0)
            match ev {
                TokEvent::Admitted { cached_tokens } => {
                    cached = cached_tokens as usize;
                    mt.cached(cached_tokens as u64); // /metrics counter
                }
                TokEvent::Tok(t) => {
                    // the request's first generated token: the role chunk leaves now
                    if let Some(r) = role_pending.take() { yield Ok(Event::default().data(r)); }
                    n += 1;
                    mt.tok();
                    last_tok = Some(t);
                    if first_tok.is_none() { first_tok = Some(std::time::Instant::now()); }
                    let text = stream_dec(t);
                    if !text.is_empty() {
                        acc.push_str(&text);
                        // A stop string ends the reply before itself: cut the buffer FIRST, so
                        // the emit sites below never send the stop string or what follows it in
                        // the same token (the check used to come after them: a streamed reply
                        // carried its stop string, and the start of one that spanned tokens,
                        // which the non-streaming response never returns).
                        if !stops.is_empty() {
                            if let Some(p) = stops.iter().zip(f_stops.iter_mut())
                                .filter_map(|(s, f)| f.find(&acc, 0, s)).min() {
                                acc.truncate(p);
                                stop_hit = true;
                                finish = "stop".to_string();
                            }
                        }
                        // (the cut is at or after every emit cursor, because a tail that could
                        // start a stop string is held back at the three hold-back sites. Three
                        // emits have no such hold-back: reasoning before a found close tag,
                        // content before a found open tag, text before a found tool marker. A
                        // stop string that OVERLAPS one of those tags can therefore cut the
                        // buffer below a cursor; the test below then skips the emit sites, and
                        // the text already sent stays sent: the reply differs from the
                        // non-streaming one for such a stop string, and only for such a one)
                        if reason_emitted.max(content_emitted) <= acc.len() && content_start.map_or(true, |cs| cs <= acc.len()) {
                            match content_start {
                                None => {
                                    // Search the close tag from reason_emitted, not from 0:
                                    // a second think block must not match the first one's close.
                                    if let Some(idx) = f_close.find(&acc, reason_emitted, think_close) {
                                        if idx > reason_emitted {
                                            let c = reasoning_chunk(&completion_id, created, &model_name, &scrub.push_reasoning(&acc[reason_emitted..idx]));
                                            if let Some(r) = &otel_req { r.delta(&c); }
                                            yield Ok(Event::default().data(c));
                                        }
                                        let cs = idx + think_close.len();
                                        let mut lead = cs;
                                        while lead < acc.len() && matches!(acc.as_bytes()[lead], b'\n' | b'\r' | b' ' | b'\t') { lead += 1; }
                                        content_start = Some(lead);
                                        // Same hold-back as the steady-state content branch
                                        // below: if a tool-call marker arrived in the same
                                        // decode chunk as the think close, it must NOT be
                                        // forwarded as content.
                                        let region = &acc[lead..];
                                        // S3: the SAME decision as the other emit sites
                                        // (gated on tool_hold) — a stray <tool_call> in
                                        // the close chunk streams as content when tool
                                        // parsing is off (keep-tools "none").
                                        let safe_end = hold_back_end(
                                            tool_hold,
                                            tool_hold.and_then(|_| region.find(TOOL_OPEN).map(|i| lead + i)),
                                            acc.len(), region, &content_marks);
                                        if safe_end > lead {
                                            let c = content_chunk(&completion_id, created, &model_name, &scrub.push_content(&acc[lead..safe_end]));
                                            if let Some(r) = &otel_req { r.delta(&c); }
                                            yield Ok(Event::default().data(c));
                                        }
                                        content_emitted = safe_end;
                                    } else {
                                        let overlap = partial_think_overlap(&acc, think_close);
                                        let safe = (acc.len() - overlap).max(reason_emitted);
                                        // Tool-call hold-back in REASONING mode too. A model
                                        // that calls a tool without ever emitting `</think>`
                                        // (qwen's first-turn behavior on trivial calls: the
                                        // template primes `<think>` and the model jumps
                                        // straight to the call) stays in this branch, which
                                        // had NO TOOL_OPEN hold-back — the raw call markup
                                        // streamed out as reasoning_content while
                                        // finalize_parsed ALSO emitted the structured
                                        // tool_calls delta: the client saw the same call
                                        // twice (2026-08-30 user report). Same contract as
                                        // the content branch below: once TOOL_OPEN appears,
                                        // reasoning emission stops; the buffer is either
                                        // parsed into the tool_calls delta or surfaced
                                        // post-loop by held_tail.
                                        let region = &acc[reason_emitted..safe];
                                        let safe_end = hold_back_end(
                                            tool_hold,
                                            tool_hold.and_then(|m| f_tool_r.find(&acc[..safe], reason_emitted, m)),
                                            safe, region, &reason_marks);
                                        if safe_end > reason_emitted {
                                            let c = reasoning_chunk(&completion_id, created, &model_name, &scrub.push_reasoning(&acc[reason_emitted..safe_end]));
                                            if let Some(r) = &otel_req { r.delta(&c); }
                                            yield Ok(Event::default().data(c));
                                            reason_emitted = safe_end;
                                        }
                                    }
                                }
                                Some(cs) => {
                                    // If the model OPENS a think block (hy_v3 with
                                    // reasoning_effort low|high), hand off to the reasoning
                                    // branch: emit the content before the marker, then split
                                    // reasoning until the close tag. Without this the raw
                                    // think tags would leak into `content`.
                                    let region = &acc[cs..];
                                    // A `<think>` that lies AFTER a held tool-call marker is text of
                                    // the held call (an argument that mentions the tag; independent
                                    // review M2: the raw call XML left as content, then the rest as
                                    // reasoning, then the correct call): only a hit before the marker
                                    // is a think block.
                                    let tool_at = tool_hold.and_then(|m| f_tool_c.find(&acc, cs, m));
                                    if let Some(upto) = f_open.find(&acc, cs, think_open).filter(|&u| tool_at.map_or(true, |t| u < t)) {
                                        if upto > content_emitted {
                                            let c = content_chunk(&completion_id, created, &model_name, &scrub.push_content(&acc[content_emitted..upto]));
                                            if let Some(r) = &otel_req { r.delta(&c); }
                                            yield Ok(Event::default().data(c));
                                        }
                                        content_start = None;
                                        reason_emitted = upto + think_open.len();
                                    } else {
                                    // Hold back anything that is, or could become, a tool call.
                                    // Forwarding `<tool_call>` as content makes the harness render
                                    // XML in the chat and never invoke the tool. Same hold-back
                                    // for a think-open prefix spanning decode chunks.
                                    let safe_end = hold_back_end(
                                        tool_hold,
                                        tool_hold.and_then(|m| f_tool_c.find(&acc, cs, m)),
                                        acc.len(), region, &content_marks);
                                    if safe_end > content_emitted {
                                        let c = content_chunk(&completion_id, created, &model_name, &scrub.push_content(&acc[content_emitted..safe_end]));
                                        if let Some(r) = &otel_req { r.delta(&c); }
                                        yield Ok(Event::default().data(c));
                                        content_emitted = safe_end;
                                    }
                                    }
                                }
                            }
                        }
                        }
                    if stop_hit { break; }
                }
                TokEvent::StopInfo { p, rank, sampled } => { stop_info = Some((p, rank, sampled)); }
                TokEvent::Logprob { .. } => {} // refused with stream=true (chat_logprobs_param)
                TokEvent::Finish { reason } => { sched_reason = Some(reason.clone()); finish = reason; break; }
            }
        }
        // WP02: release the lane NOW (a stop-string hit cancels it within one scheduler step)
        drop(rx);
        // no token was generated (an engine error, a finish without a token, the engine gone):
        // the role chunk goes immediately before the terminal events
        if let Some(r) = role_pending.take() { yield Ok(Event::default().data(r)); }
        if finish.starts_with("error") {
            // WP02: a mid-stream engine error is an SSE error event (+ a spec finish_reason below)
            crate::reprintln!("[req] stream ended in an engine error after {n} tokens: {finish}");
            yield Ok(Event::default().data(sse_error_event(&finish)));
        }
        // The call was buffered, not streamed (see the hold-back above). The DECISION is
        // crate::tools::finalize_parsed — the one canonical serializer shared with the
        // non-streaming mode — and the held-back text is surfaced by
        // held_tail, so streaming can never silently drop text the JSON
        // mode returns (2026-08-27 user report: a malformed `function=NAME>` block with the
        // `<` missing vanished from the SSE stream while the JSON response leaked it).
        let (_, done_content) = split_think_in(&acc, think_open, think_close, starts_in_reasoning);
        // parse_as_tool_calls == false (keep-tools "none"): never parse — the reply is plain
        // content; finalize_parsed then returns the raw text with finish stop/length.
        let parsed = if parse_as_tool_calls {
            crate::tools::parse_served(&done_content, req_tools.as_deref(), &finish)
        } else {
            crate::tools::ParsedOutput { content: done_content.trim().to_string(), tool_calls: Vec::new() }
        };
        if req_tools.is_some() {
            let dump = crate::opts::var(crate::opt!("dump-tools")).is_ok();
            if dump || parsed.tool_calls.is_empty() {
                crate::dreprintln!("[req] raw model output ({} chars): {:?}", done_content.chars().count(),
                          done_content.chars().take(1200).collect::<String>());
            }
            if dump { crate::reprintln!("{}", dump_tools_line(&done_content)); }
        }
        // Phase-4 A2: same empty-argument alarm as the non-streaming path — an agent harness
        // streams, so this branch is the one that actually gets used.
        for a in crate::tools::empty_arg_alarms(&done_content, req_tools.as_deref(), &parsed.tool_calls) {
            crate::reprintln!("[tool-args-alarm] {a}");
        }
        let (tc_content, tool_calls, fin) = crate::tools::finalize_parsed(&done_content, parsed, &finish);
        // TOOLEVAL: the per-turn tool trace (None = flag off -> one Option check, nothing else).
        if let Some(t) = &tool_trace {
            crate::tool_trace::log(t, &acc, &done_content, tc_content.as_deref(), &tool_calls, &fin);
        }
        mt.finish(if finish.starts_with("error") { finish.as_str() } else { fin.as_str() });
        if !tool_calls.is_empty() {
            // Log the ARGUMENTS, not just the names — see the note on the non-streaming path.
            // Agent harnesses stream, so this is the branch that actually gets used, and it was the
            // one printing a bare `tool_calls 1: ["write"]` while a file silently failed to appear.
            for t in &tool_calls {
                crate::reprintln!("[req] tool_call  {} {}({})", t.id, t.function.name, t.function.arguments);
            }
            // TOOLSCRUB1: flush any text the scrubber still holds (it belongs before the
            // call), then scrub the call arguments themselves. The [req] log above keeps
            // the RAW arguments (same class as --tool-trace / --dump-tools: server-side
            // diagnostics of what the model actually produced).
            let r_flush = scrub.finish_reasoning("");
            if !r_flush.is_empty() {
                let c = reasoning_chunk(&completion_id, created, &model_name, &r_flush);
                if let Some(r) = &otel_req { r.delta(&c); }
                yield Ok(Event::default().data(c));
            }
            let c_flush = scrub.finish_content("");
            if !c_flush.is_empty() {
                let c = content_chunk(&completion_id, created, &model_name, &c_flush);
                if let Some(r) = &otel_req { r.delta(&c); }
                yield Ok(Event::default().data(c));
            }
            let tc_scrubbed: Vec<ToolCall> = tool_calls.iter().map(|t| {
                let mut t2 = t.clone();
                t2.function.arguments = scrub.scrub_full(&t2.function.arguments).into_owned();
                t2
            }).collect();
            let tc = tool_calls_chunk(&completion_id, created, &model_name, &tc_scrubbed);
            if let Some(r) = &otel_req { r.delta(&tc); }
            yield Ok(Event::default().data(tc));
            finish = fin;
        } else if !finish.starts_with("error") {
            // Nothing became a tool call, so whatever the hold-back still keeps is TEXT and
            // leaves now (S-B9-REL-SURFACE2B R5). NOT after an engine error (S-B9-REL-INTEG B5): the
            // error event has gone out, and nothing follows it but the finish chunk, unless the held
            // span holds a COMPLETE call (the branch above: master's rule). The emit sites above hold back a tool-call
            // marker and everything after it, and the tail of the buffer while it could still
            // become a marker (`<`, `<tool`, `<thi`, `</thi` are prefixes of `<tool_call`,
            // `<think>`, `</think>`). The marker case was surfaced here; a tail that was only
            // a PREFIX was not, so a reply that ended in one lost those characters (a reply cut
            // by max_tokens right after a `<`, or simply ending in `<`), and so did the text
            // after a `<tool_call` that was not a call tag (`<tool_call_x`). The non-streaming
            // response returns all of them.
            let (reasoning_tail, content_tail) = held_tail(&acc, reason_emitted.max(content_emitted),
                                                           content_start.is_none(), tool_hold, think_close);
            // TOOLSCRUB1: the tails (and anything the scrubber still holds) leave through
            // the scrubber's finish path; off mode returns them byte-identical.
            let rs = scrub.finish_reasoning(&reasoning_tail);
            if !rs.is_empty() {
                let c = reasoning_chunk(&completion_id, created, &model_name, &rs);
                if let Some(r) = &otel_req { r.delta(&c); }
                yield Ok(Event::default().data(c));
            }
            let cs = scrub.finish_content(&content_tail);
            if !cs.is_empty() {
                let c = content_chunk(&completion_id, created, &model_name, &cs);
                if let Some(r) = &otel_req { r.delta(&c); }
                yield Ok(Event::default().data(c));
            }
            content_emitted = acc.len();
        }
        {
            let (r_txt, c_txt) = split_think_in(&acc, think_open, think_close, starts_in_reasoning);
            let (r_log, c_log) = gen_log_spans(r_txt.as_deref(), &c_txt, acc.contains(think_close), starts_in_reasoning);
            if let Some(g) = &gen_log {
                log_generation(&g.tokenizer, &g.stop_ids, &completion_id, &finish,
                               last_tok, prompt_len, n, r_log, c_log,
                               tool_calls.len(), req_tools.as_ref().map(|t| t.len()).unwrap_or(0),
                               g.effort.as_deref(), g.presence_penalty, g.pp_source,
                               stop_kind(sched_reason.as_deref(), stop_hit), stop_info);
            }
        }
        // WP08: `stop_reason` only when the loop detector ended the stream (else unchanged).
        let stop_reason_field = stop_reason_of(&finish)
            .map(|r| format!(",\"stop_reason\":\"{r}\"")).unwrap_or_default();
        let final_chunk = final_chunk(&completion_id, created, &model_name,
            spec_finish_reason(&finish), &stop_reason_field); // S8: esc(model)
        if let Some(r) = &otel_req { r.end(&final_chunk); }   // event=stream_end (carries finish_reason)
        yield Ok(Event::default().data(final_chunk));
        if include_usage {
            // Spec stream-usage chunk: empty choices, top-level usage. `timings` rides along
            // as the extension field (strict clients ignore it), and `session_id` echoes the
            // OTel session key so the client can label the stream without computing anything.
            let mut usage_chunk = serde_json::json!({
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model_name, "choices": [],
                "usage": {"prompt_tokens": prompt_len, "completion_tokens": n,
                          "total_tokens": prompt_len + n,
                          "prompt_tokens_details": {"cached_tokens": cached}},
                "timings": make_timings(t0, first_tok, prompt_len, n),
            });
            if let Some(s) = &otel_session { usage_chunk["session_id"] = serde_json::json!(s); }
            yield Ok(Event::default().data(usage_chunk.to_string()));
        }
        // The OpenAI SSE terminator. Without it a strict client sits on an open stream
        // waiting for more events after the finish chunk.
        yield Ok(Event::default().data("[DONE]"));
        let dt = t0.elapsed().as_secs_f32();
        crate::reprintln!("[req] done   tok={} ({:.1} tok/s wall) finish={} stop_hit={}", n, if dt>1e-6 {n as f32/dt} else {0.0}, finish, stop_hit);
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<ChatCompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(mut req) = match payload { Ok(j) => j, Err(e) => return bad_json(e) };
    // Log the request parameters the client sent (useful for debugging OpenWebUI behavior)
    crate::dreprintln!(
        "[req] params  temp={:?} top_p={:?} top_k={:?} max_tok={:?} max_completion_tok={:?} rep_pen={:?} presence={:?} freq={:?} stream={} effort={:?} ctk={}",
        req.temperature, req.top_p, req.top_k,
        req.max_tokens, req.max_completion_tokens, req.repetition_penalty, req.presence_penalty, req.frequency_penalty,
        req.stream, req.reasoning_effort,
        req.chat_template_kwargs.as_ref().map(|v| v.to_string()).unwrap_or_else(|| "-".into())
    );
    if !req.unused_fields.is_empty() {
        let names: Vec<&str> = req.unused_fields.keys().map(|k| k.as_str()).collect();
        crate::dreprintln!("[req] fields not used by this server: {names:?}");
    }
    if let Err(msg) = validate_penalties(req.repetition_penalty, req.presence_penalty,
                                         req.frequency_penalty, req.min_p) {
        return bad_param(msg);
    }
    if let Some(m) = req.max_completion_tokens {
        req.max_tokens = Some(m);
    }
    if let Some(t) = &req.tools {
        let names: Vec<&str> = t.iter()
            .filter_map(|x| x.pointer("/function/name").and_then(|v| v.as_str())).collect();
        crate::reprintln!("[req] tools   {} offered: {:?} tool_choice={:?}", t.len(), names, req.tool_choice);
    }
    // Resolve the effective reasoning effort. Two families, two vocabularies:
    //   - hy_v3's template accepts `no_think|low|high` (default low), and its Rust dsv4 path treats
    //     None/"" as low.
    //   - Qwen3.5's template accepts `xhigh|medium|low` (default xhigh) and RAISES on anything else.
    // Forward the client's value verbatim when it is valid for THIS family's template; convert
    // the OpenAI API convention onto the nearest native level. When NEITHER the request nor
    // --reasoning-effort specifies one, pass None so the model's own template default wins
    // (xhigh for Qwen 3.8, low for hy_v3) — never a hardcoded guess.
    //
    // Qwen 3.8 native (the ONLY values its template accepts; anything else raises => 500):
    //   xhigh (default) | medium | low          ["no_think"/"off" => enable_thinking=false]
    // OpenAI API -> Qwen 3.8 (owner spec 2026-08-30):
    //   none    -> thinking off      (latency-critical; no reasoning)
    //   low     -> low               (efficient reasoning)
    //   medium  -> medium            (balanced; OpenAI's default)
    //   high    -> xhigh             (hard reasoning)
    //   xhigh   -> xhigh             (deep research)
    //   max     -> xhigh             (maximum)
    // hy_v3 native: no_think | low | high  =>  none->no_think, low->low, medium/high/xhigh/max->high
    //
    // REGRESSION FIX (2026-08-30): the 289e1a1 refactor lumped "high" into the no_think arm,
    // so every OpenAI-convention client sending reasoning_effort=high silently LOST thinking.
    // W2 (Phase 13): `response_format` — compiled HERE, before any work. A supported schema arms
    // the token-level mask when enforcement is on; when it is off the request is still SERVED and
    // the response is marked `x-json-schema-enforced: none` (see warn_unenforced_schema_once).
    // NEVER a 400 for a schema this build can parse: refusing was `e737d08`, and it broke the
    // public quality benchmark (88 → 85) by hard-failing requests that used to succeed.
    // Flip JSON_SCHEMA_ENFORCEMENT_ENABLED to true only when the emission path is proven
    // end-to-end: `cargo test --lib json_schema::tests` green AND the W2 probe 6/6 enforced
    // across temperature × TP × spec source. Until then, serving + advertising beats refusing.
    const JSON_SCHEMA_ENFORCEMENT_ENABLED: bool = false;
    let mut schema_mask: Option<std::sync::Arc<crate::json_schema::SchemaMask>> = None;
    let mut schema_unenforced: Option<String> = None;
    if let Some(rf) = req.response_format.as_ref() {
        match crate::json_schema::compile_response_format(rf) {
            Ok(None) => {}
            Ok(Some(mut m)) if JSON_SCHEMA_ENFORCEMENT_ENABLED => {
                m.set_vocab(state.tokenizer.vocab_pieces());
                crate::reprintln!("[req] response_format: constrained decoding armed ({})", m.summary());
                schema_mask = Some(std::sync::Arc::new(m));
            }
            Ok(Some(m)) => {
                schema_unenforced = Some(format!(
                    "schema compiled ({}) but this build does not enforce constrained decoding yet",
                    m.summary()));
            }
            Err(e) => {
                schema_unenforced = Some(format!("schema not compiled: {e}"));
            }
        }
        if let Some(why) = schema_unenforced.as_deref() { warn_unenforced_schema_once(why); }
    }
    // W1: `chat_template_kwargs` (arbitrary dict → the model's own template). Validate LOUDLY
    // before any render: a malformed value used to be invisible (serde dropped the field), and
    // the customer was left with a model that ignored the request. 400, never accept-and-ignore.
    let logprobs_n = match chat_logprobs_param(req.logprobs, req.top_logprobs, req.stream) {
        Ok(n) => n,
        Err(msg) => return bad_param(msg),
    };
    // Per-token entries carry the generated bytes; a secret the response scrubber removes from
    // the text can span several tokens, so the two cannot be combined safely.
    if logprobs_n.is_some() && crate::tool_scrub::Mode::from_opt() != crate::tool_scrub::Mode::Off {
        return bad_param("logprobs are not available while --scrub-tool-secrets is enabled".into());
    }
    if let Err(e) = crate::tokenizer::enable_thinking_kwarg(req.chat_template_kwargs.as_ref()) {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
            "message": e.to_string(), "type": "invalid_request_error", "code": "invalid_chat_template_kwargs",
        }}))).into_response();
    }
    // An unknown effort is the CLIENT's error: a 400 naming the accepted values, never the
    // template's raise surfacing as a plain-text 500.
    if let Some(e) = req.reasoning_effort.as_deref() {
        if !matches!(e, "" | "none" | "off" | "minimal" | "no_think" | "low" | "medium" | "high" | "xhigh" | "max") {
            return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
                "message": format!("reasoning_effort must be one of none|minimal|low|medium|high|xhigh|max (got '{e}')"),
                "type": "invalid_request_error", "code": "invalid_reasoning_effort",
            }}))).into_response();
        }
    }
    let effort_owned = resolve_reasoning_effort(&state.tokenizer, req.reasoning_effort.as_deref(),
                                                state.reasoning_effort.as_deref());
    let effort: Option<&str> = effort_owned.as_deref();
    // OpenAI/vLLM tool_choice contract. The engine has no guided-decode forcing, so the
    // semantics are implemented at the prompt level: "required" / a specific function append an
    // explicit forcing instruction naming the constraint; the "none" render/parse DECISION lives
    // in tool_choice_decision() below. Scoring harnesses (tool-eval-bench) drive scenarios
    // through these modes.
    //
    // S-B9-REL-NONETOOLS: with --keep-tools-when-tool-choice-none (default off) a "none"
    // request keeps the tools in the prompt (vLLM's default contract; our legacy default is the
    // llama.cpp approach — tools removed, "the model cannot call what it cannot see"). The
    // prompt then shares its long prefix with the conversation's normal requests, so the prefix
    // cache / WP16 checkpoints survive a compaction/summary turn (the Qwen template renders the
    // tools inside the system block, so removing them invalidates the cache from token 0).
    let (tools_for_template, force_line, parse_as_tool_calls) = tool_choice_decision(
        req.tool_choice.as_ref(), req.tools.as_deref(),
        crate::opts::var(crate::opt!("keep-tools-when-tool-choice-none")).is_ok(),
    );
    let mut messages = req.messages.clone();
    if let Some(line) = force_line {
        apply_force_line(&mut messages, &line); // S4: the ONE append (unit-tested)
    }
    let t_render = std::time::Instant::now();
    let prompt = match state.tokenizer.apply_chat_template(&messages, tools_for_template, effort,
                                                          req.chat_template_kwargs.as_ref(),
                                                          state.thinking) {
        Ok(p) => p,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };
    let render_ms = t_render.elapsed().as_secs_f64() * 1000.0;

    // Optional diagnostic: dump the exact rendered prompt string so the bytes a model
    // actually sees can be inspected/diffed across models or turns. Enable with
    // --dump-prompt=1. Writes /tmp/rust_infer_prompt_<n>.txt per request.
    if crate::opts::var(crate::opt!("dump-prompt")).is_ok() {
        static DUMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = DUMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dump_path = format!("/tmp/rust_infer_prompt_{}.txt", n);
        if std::fs::write(&dump_path, &prompt).is_ok() {
            crate::reprintln!("[req] dumped prompt ({} chars) -> {}", prompt.chars().count(), dump_path);
        }
    }

    let t_encode = std::time::Instant::now();
    let mut prompt_tokens = match encode_off_thread(&state.tokenizer, &prompt, true).await {
        Ok(t) => t,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };
    let encode_ms = t_encode.elapsed().as_secs_f64() * 1000.0;
    // TTFT fix 0 attribution (--prefill-trace): the request-path pre-model costs.
    if crate::opts::var(crate::opt!("prefill-trace")).ok().is_some() {
        crate::reprintln!("[pf] server render={render_ms:.3}ms encode={encode_ms:.3}ms");
    }

    // V3 vision dispatch: if any message carries images (and the server has a vision tower),
    // decode+preprocess+run the tower, expand the image_pad span in the token stream, and carry the
    // merged embeddings + spans for the model prefill splice. Text-only traffic is unchanged.
    let mut image_embeds: Option<Vec<f32>> = None;
    let mut image_spans: Vec<crate::vision_encoder::ImageSpan> = Vec::new();
    // VID-0 (VIS-5): a content part this server cannot take (video, audio, file, ...) is a 400 naming the
    // type — it used to be dropped silently (the model then answered about content it never saw).
    if let Some(t) = req.messages.iter().flat_map(|m| m.unsupported_parts.iter()).next() {
        return (StatusCode::BAD_REQUEST,
                format!("content part type '{t}' is not supported (text and images only; video input is not available yet)"))
            .into_response();
    }
    let urls: Vec<String> = req.messages.iter()
        .flat_map(|m| m.images.iter().filter_map(|i| i.url.clone()))
        .collect();
    if !urls.is_empty() {
        let vt0 = std::time::Instant::now();
        // Prefer the GPU tower (fast path) unless --vision-cpu forces the CPU reference.
        // VIS-2: decode + preprocess + tower run on a blocking worker, not inline on the async
        // handler (a long tower used to pin a tokio worker under the std Mutex).
        let (gpu, cpu_tower, force_cpu) = (state.vision_gpu.clone(), state.vision_tower.clone(), state.vision_cpu);
        let toks = prompt_tokens.clone();
        let urls_w = urls.clone();
        // PR #16: keep a second CPU-tower handle + request copies for the worker-PANIC arm
        // (the moved values are gone once the spawn_blocking closure owns them).
        let (cpu_fb, urls_fb, toks_fb) = (state.vision_tower.clone(), urls.clone(), prompt_tokens.clone());
        let prep = match tokio::task::spawn_blocking(move || {
            let cpu = || match &cpu_tower {
                Some(tower) => crate::vision_encoder::prepare_vision_request(tower, &urls_w, &toks),
                None => Err(anyhow::anyhow!("no vision tower loaded")),
            };
            match gpu.filter(|_| !force_cpu) {
                Some(g) => vision_dispatch_with_recovery(&g, |gvt| {
                    crate::vision_encoder::prepare_vision_request_gpu(gvt, &urls_w, &toks)
                }, cpu),
                None => cpu(),
            }
        }).await {
            Ok(r) => r,
            // PR #16: a panic inside the GPU worker (kernel launch, decode of the payload) used
            // to fail the request outright; the CPU tower serves the image instead — the request
            // is never silently served without its image. A lock poisoned by that panic is
            // recovered by vision_dispatch_with_recovery on the NEXT request.
            // S-B9-REL-PRPORT3 (review R4): the CPU fallback itself runs on a blocking worker —
            // the CPU tower is multi-second and must never run on the async handler thread.
            Err(e) => vision_panic_fallback(cpu_fb, urls_fb, toks_fb, e).await,
        };
        crate::reprintln!("[vision] dispatch {} images, prepare took {} ms, len={}",
            urls.len(), vt0.elapsed().as_millis(), prep.as_ref().map(|p| p.image_embeds.len()).unwrap_or(0));
        match prep {
            Ok(prep) => {
                prompt_tokens = prep.expanded_tokens;
                image_embeds = Some(prep.image_embeds);
                image_spans = prep.spans;
            }
            Err(e) => return (StatusCode::BAD_REQUEST,
                format!("vision preprocessing failed: {e}")).into_response(),
        }
    }

    // truncate_prompt_tokens (vLLM's field): keep the LAST n tokens — the same left-truncation
    // convention the /v1/tokenize handler implements. Runs BEFORE the max_seq_len check so any n
    // that fits turns the over-length 400 into a served request; the boundary snapshot (ckpt_at)
    // below re-renders the FULL template, so a truncated stream simply fails its `n < prompt_len`
    // filter and skips the checkpoint this turn (truncated history is not a prefix of the next
    // full-history turn anyway — the cache can't be trusted across the cut).
    if let Some(n) = req.truncate_prompt_tokens {
        if n == 0 {
            return (StatusCode::BAD_REQUEST,
                    "'truncate_prompt_tokens' must be an integer >= 1").into_response();
        }
        if prompt_tokens.len() > n && !image_spans.is_empty() {
            // VIS-2: dropping the front of an image prompt would leave the image spans pointing at the
            // wrong rows (the spans index the expanded stream) — refuse instead of splicing wrongly.
            return (StatusCode::BAD_REQUEST,
                    "'truncate_prompt_tokens' cannot cut a prompt that carries images").into_response();
        }
        if prompt_tokens.len() > n {
            let dropped = prompt_tokens.len() - n;
            prompt_tokens.drain(..dropped); // keep the LAST n — vLLM's left-truncation convention
            crate::reprintln!("[req] truncate_prompt_tokens: dropped the OLDEST {dropped} of {} prompt \
                       tokens (kept the last {n})", dropped + n);
        }
    }

    let prompt_len = prompt_tokens.len();

    if state.output_prompts > 0 {
        let mname = req.model.clone().unwrap_or_else(|| state.model_name.clone());
        log_request_human(&req, effort, &prompt, prompt_len, state.output_prompts, &mname, render_ms);
    }

    // Where to snapshot the GDN state: the message boundary, i.e. this prompt without its trailing
    // generation prompt. Everything up to here is what the NEXT turn replays verbatim. Rendering the
    // template a second time costs microseconds and saves a whole re-prefill per turn — but only
    // when the scheduler's prefix cache is actually on (batch.rs filters ckpt_at again); with the
    // cache off this second render+encode is pure TTFT cost (fix (e), EXPERT_TTFT_PREFILL_RESPONSE).
    let ckpt_at = if state.prefix_cache {
        state.tokenizer
            .apply_chat_template_no_gen(&req.messages, req.tools.as_deref(), effort,
                                        req.chat_template_kwargs.as_ref(), state.thinking).ok()
            .and_then(|s| state.tokenizer.encode(&s, true).ok())
            .map(|t| t.len())
            .filter(|&n| n > 0 && n < prompt_len)
    } else { None };

    // The KV cache holds exactly `max_seq_len` positions. A prompt past that end used to be written
    // out of bounds — silently, corrupting whatever allocation followed, which showed up as two
    // identical prefills disagreeing. Reject what cannot fit, and cap generation at the room left:
    // running short is a `finish_reason: "length"`, which is in the contract. Corruption is not.
    if generation_room(state.max_seq_len, prompt_len, state.decode_headroom).is_none() {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
            "message": format!("This model's maximum context length is {} tokens, but your messages \
                                came to {} tokens and decoding reserves {} more positions. Shorten the input or \
                                restart the server with a larger --max-seq-len.",
                                state.max_seq_len, prompt_len, state.decode_headroom),
            "type": "invalid_request_error", "code": "context_length_exceeded",
        }}))).into_response();
    }
    let room = generation_room(state.max_seq_len, prompt_len, state.decode_headroom)
        .expect("context room checked above");
    let asked = req.max_tokens.unwrap_or(state.default_max_tokens);
    let req_max = asked.min(room);
    // If the KV cache forced generation shorter than asked, SAY SO. A thinking model spends a big fixed
    // chunk on its <think> block, so a silently-shrunk budget looks like "truncated output / only
    // reasoning" as a conversation grows — which is exactly how this surfaced in the wild. Raise
    // --max-seq-len (graphs cost ~nothing here; KV is ~64 KB/token) to give multi-turn room.
    if req_max < asked && req.max_tokens.is_some() {
        crate::reprintln!("[req] max_tokens clamped {} -> {} (KV cache: {}-token prompt + {} reserved decode positions of {}; \
                   raise --max-seq-len)", asked, req_max, prompt_len, state.decode_headroom, state.max_seq_len);
    }
    // Sampling: explicit request values win; anything left out takes the server's default for
    // THIS request's mode (the rendered prompt opens a <think> block or it does not).
    let thinking_on = prompt.trim_end().ends_with(state.tokenizer.think_tags().0);
    let (d_temp, d_top_p, d_top_k) = if thinking_on { state.sampling_defaults.think }
                                     else { state.sampling_defaults.no_think };
    let temperature = req.temperature.unwrap_or(d_temp);
    let top_p = req.top_p.unwrap_or(d_top_p).max(0.01);
    let top_k = req.top_k.unwrap_or(d_top_k);
    crate::dreprintln!("[req] sampling temp={temperature} top_p={top_p} top_k={top_k} (thinking={thinking_on}; \
               sent: temp={:?} top_p={:?} top_k={:?})", req.temperature, req.top_p, req.top_k);

    // Submit to the batching scheduler and receive tokens on a channel.
    // Use request's penalties if explicitly set, else fall back to server defaults.
    let rep_penalty = req.repetition_penalty.unwrap_or(state.default_rep_penalty);
    let presence_penalty = req.presence_penalty.unwrap_or(state.default_presence_penalty);
    let pp_source = if req.presence_penalty.is_some() { "request" } else { "server-default" };
    let frequency_penalty = req.frequency_penalty.unwrap_or(state.default_frequency_penalty);
    // TOOLEVAL (P18): arm the per-turn tool trace for requests that involve tools - the
    // request offers tools, or its history carries a tool result. None (and no further cost)
    // when the flag is off. The trace only READS the prompt, the raw output and the parse the
    // server already computed: no served byte changes with it on or off.
    let tool_trace = {
        let tool_turn = req.tools.is_some() || req.messages.iter().any(|m| m.role == "tool");
        if !tool_turn { None } else {
            crate::tool_trace::capture(&prompt, serde_json::json!({
                "id": format!("tt-{}", Uuid::new_v4()),
                "ts": chrono::Utc::now().to_rfc3339(),
                "stream": req.stream,
                "seed": req.seed,
                "sampling": {
                    "temperature": temperature, "top_p": top_p, "top_k": top_k,
                    "presence_penalty": presence_penalty, "frequency_penalty": frequency_penalty,
                    "sent": { "temperature": req.temperature, "top_p": req.top_p, "top_k": req.top_k },
                },
                "thinking_on": thinking_on,
                "n_tools": req.tools.as_ref().map_or(0, |t| t.len()),
                "tool_choice": req.tool_choice.clone(),
                "parse_as_tool_calls": parse_as_tool_calls,
                "prompt_tokens": prompt_len,
                "roles": req.messages.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
            }))
        }
    };

    let (tx, mut rx, bl) = tok_channel(); // LR-5: `bl` counts consumed events in the loops below
    let request = BatchRequest {
        prompt: prompt_tokens.clone(),
        max_new: req_max,
        temperature,
        received_at: std::time::Instant::now(),
        top_p,
        top_k,
        rep_penalty,
        presence_penalty,
        frequency_penalty,
        min_p: req.min_p.unwrap_or(0.0),
        min_new: req.min_tokens.unwrap_or(0),
        ignore_eos: req.ignore_eos.unwrap_or(false),
        tx,
        seed: req.seed,
        ckpt_at,
        domain: crate::batch::classify_domain(&prompt),
        image_embeds,
        image_spans,
        schema: schema_mask.clone(),
        logprobs: logprobs_n,
    };
    if admit_gate_busy() { return server_busy(); } // LR-4: --max-waiting
    // WP02: a DEAD engine or a gone scheduler is a 503, never a request that hangs or reads "length"
    if !engine_ok() || state.scheduler.send(request).is_err() {
        return engine_unavailable();
    }
    let mut mt = crate::metrics::Req::start(prompt_len); // /metrics (recorded when the response ends)

    // SESSION identity for the OTel generation-telemetry (crate::otel::SessionRegistry). One
    // resolution per REQUEST, only when the emitter is on (off = no session work at all):
    // explicit client key first (X-Session-Id header, then metadata.session_id / .conversation_id),
    // else the engine infers the continuous conversation from the messages-prefix rule — turn
    // N+1's messages array contains turn N's as an exact prefix, so a follow-up lands in the
    // SAME session id and the client sees one continuous session, not one per response.
    let otel_session: Option<String> = state.otel.as_ref().map(|s| {
        let explicit = headers.get("x-session-id").and_then(|v| v.to_str().ok())
            .map(str::to_string)
            .or_else(|| req.metadata.as_ref().and_then(|m| m.get("session_id"))
                .and_then(|v| v.as_str()).map(str::to_string))
            .or_else(|| req.metadata.as_ref().and_then(|m| m.get("conversation_id"))
                .and_then(|v| v.as_str()).map(str::to_string));
        // Canonical per-message JSON, in order — the fingerprint material (serialized once).
        let per_msg: Vec<String> = req.messages.iter()
            .map(|m| serde_json::to_string(m).unwrap_or_default()).collect();
        let sess = s.resolve_session(explicit, &per_msg);
        crate::reprintln!("[req] session {sess} (turn of {} message(s))", req.messages.len());
        sess
    });

    // TOOLSCRUB1: the per-request emission-side scrubber (one registry read +, when
    // provenance is on, candidate extraction from this conversation's tool results).
    let scrub = crate::tool_scrub::Scrubber::new(crate::tool_scrub::Mode::from_opt(), &req.messages);
    if req.stream {
        crate::dreprintln!("[req] stream  prompt_tokens={} max_tokens={} stop={:?}", prompt_len, req_max, req.stop);
        let tokenizer = Arc::clone(&state.tokenizer);
        let model_name = req.model.clone().unwrap_or_else(|| state.model_name.clone());
        let stops = req.stop.clone();
        let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();
        let t0 = std::time::Instant::now();
        let req_tools = req.tools.clone();
        // S-B9-REL-NONETOOLS: the parse decision for this request. `tool_hold` is the
        // <tool_call hold-back needle — None for a keep-tools "none" request, whose stray
        // call streams through as plain content instead of being buffered for a tool_calls delta.
        let tool_hold: Option<&'static str> = if parse_as_tool_calls { Some(TOOL_OPEN) } else { None };
        let include_usage = req.stream_options.as_ref().map(|o| o.include_usage).unwrap_or(true);
        // Owned copy: the SSE generator is 'static, so it cannot capture the `&str` that borrows
        // effort_owned (A3 log line only; the render/ckpt calls above still use `effort`).
        let effort_for_log: Option<String> = effort.map(|s| s.to_string());
        // Think markers + the initial reasoning/content state. Derive the start mode from the
        // RENDERED PROMPT TAIL, not a family constant: qwen's template primes an OPEN think block
        // when thinking (prompt ends with `<think>`), but its no-think branch (enable_thinking=
        // false — effort none/no_think/off) emits a CLOSED empty block `<think>\n\n</think>\n\n`,
        // so that stream starts in CONTENT. A family-constant `primed` mislabeled the direct
        // answer as reasoning_content forever (content stayed empty; the model card's non-thinking
        // mode is real and respected by the model — verified sync+bf16 2026-08-30). hy_v3 keeps
        // the effort arm: its low|high template primes `…assistant<think:opensource>` even when
        // the tail check can't see it (see `stream_starts_in_reasoning`).
        let (think_open, think_close, family_primed) = tokenizer.think_tags();
        // trim_end: the primed form is `<think>\n` — the trailing newline must not defeat the
        // tail check (a plain ends_with(think_open) sent thinking streams to content: exactly
        // the 35b4b15 follow-up bug). The no-think tail `<think>\n\n</think>\n\n` trims to
        // `</think>` and stays content.
        let starts_in_reasoning = stream_starts_in_reasoning(&prompt, think_open, family_primed, effort);

        // OTel generation telemetry (--otel-endpoint). Handle built ONCE per request; the hooks
        // below forward the SAME chunk strings the SSE path yields (single source of truth —
        // the telemetry path never re-derives or re-encodes a delta). `None` (endpoint absent)
        // = the default = every hook below compiles away. The hooks are pure observers: they
        // cannot alter the SSE bytes (see crate::otel — lock-free ring, drop-on-full, the
        // sender runs on its own timer off the compute stream). request.id = the SESSION key
        // (stable across the conversation's turns); generation.id = this POST's execution id.
        let otel_req = match (&state.otel, &otel_session) {
            (Some(s), Some(sess)) =>
                Some(s.open_request(sess, &format!("gen-{}", Uuid::new_v4()))),
            _ => None,
        };

        // S-B9-REL-BENCHFIX2: the stream's generator is `chat_sse_stream` (the unit tests drive that
        // same function); it sends the role chunk when the first token has been generated.
        let cx = ChatSse {
            rx, bl, mt, otel_req, otel_session,
            role_chunk: role_chunk(&completion_id, created, &model_name), // S8: esc(model)
            completion_id, created, model_name, stops,
            think_open, think_close, starts_in_reasoning, tool_hold, parse_as_tool_calls,
            req_tools, include_usage, prompt_len, t0,
            tool_trace, scrub,
            gen_log: Some(GenLog {
                tokenizer: Arc::clone(&state.tokenizer), stop_ids: state.stop_ids.clone(),
                effort: effort_for_log, presence_penalty, pp_source,
            }),
        };
        let ka_frame = sse_keepalive_frame(&cx.completion_id, cx.created, &cx.model_name, "chat.completion.chunk");
        let stream = chat_sse_stream(cx, move || {
            // Byte-level stream decoder: the per-token decode path above would mangle every
            // multi-byte char split across tokens (all emoji) into "�" — the crate's ByteLevel
            // decode is String::from_utf8_lossy per call. Reassembles raw bytes across tokens.
            let mut dec = tokenizer.stream_decoder();
            move |t| dec.push(t)
        });
        let mut resp = sse_with_keepalive(stream, sse_keepalive(), ka_frame);
        attach_schema_unenforced_header(&mut resp, schema_unenforced.is_some());
        resp
    } else {
        crate::reprintln!("[req] sync   prompt_tokens={} max_tokens={} stop={:?}", prompt_len, req_max, req.stop);
        let t0 = std::time::Instant::now();
        let mut tokens = Vec::new();
        // log probabilities (logprobs requested): one entry per token, from the Logprob event
        // the engine sends right before the token
        let mut lps: Vec<TokLp> = Vec::new();
        let mut pending_lp: TokLp = None;
        // WP02: a channel closed without a Finish is an engine failure, not "length"
        let mut finish = "error: engine stopped".to_string();
        let mut first_tok: Option<std::time::Instant> = None;
        let mut cached = 0usize; // v0.7.3 (#8.2): prefix-cache-served prompt tokens
        // [gen] line (item 5): the scheduler's own finish reason, a stop-string hit, the stop-token statistics
        let mut sched_reason: Option<String> = None;
        let mut stop_string_hit = false;
        let mut stop_info: Option<(f32, u32, bool)> = None;
        while let Some(ev) = recv_counted(&mut rx, &bl).await { // LR-5: the one decrement
            match ev {
                TokEvent::Admitted { cached_tokens } => {
                    cached = cached_tokens as usize;
                    mt.cached(cached_tokens as u64); // /metrics counter
                }
                TokEvent::Tok(t) => {
                    tokens.push(t);
                    if logprobs_n.is_some() { lps.push(pending_lp.take()); }
                    mt.tok();
                    if first_tok.is_none() { first_tok = Some(std::time::Instant::now()); }
                    // Apply stop strings LIVE, not just post-hoc: on a hit, break AND let rx drop —
                    // the scheduler sees the closed channel and cancels the lane instead of decoding
                    // to EOS/max_new. Only the tail is searched (a stop string spans a few tokens;
                    // one longer than the window is still honoured post-hoc below, just not early).
                    if !req.stop.is_empty() && tokens.len() % 4 == 0 {
                        let tail = &tokens[tokens.len().saturating_sub(96)..];
                        let s = state.tokenizer.decode(tail, true).unwrap_or_default();
                        if req.stop.iter().any(|x| !x.is_empty() && s.contains(x.as_str())) {
                            finish = "stop".to_string();
                            stop_string_hit = true;
                            break;
                        }
                    }
                }
                TokEvent::StopInfo { p, rank, sampled } => { stop_info = Some((p, rank, sampled)); }
                TokEvent::Logprob { logprob, top } => { pending_lp = Some((logprob, top)); }
                TokEvent::Finish { reason } => { sched_reason = Some(reason.clone()); finish = reason; break; }
            }
        }
        drop(rx); // WP02: release the lane now
        let dt = t0.elapsed().as_secs_f32();
        let mut text = state.tokenizer.decode(&tokens, true).unwrap_or_default();
        if !req.stop.is_empty() {
            if let Some(p) = req.stop.iter().filter(|s| !s.is_empty()).filter_map(|s| text.find(s.as_str())).min() {
                text.truncate(p); finish = "stop".to_string();
                stop_string_hit = true;
            }
        }
        crate::reprintln!("[req] done   tok={} ({:.1} tok/s wall) finish={}", tokens.len(), if dt>1e-6 {tokens.len() as f32/dt} else {0.0}, finish);
        // An engine error is an HTTP error, never a 200 that a client (or a benchmark) would read
        // as an answer. WP02: also when some tokens came first (the text is incomplete).
        if finish.starts_with("error") {
            mt.finish(&finish);
            return (engine_error_status(&finish), Json(serde_json::json!({"error": {
                "message": finish, "type": "server_error", "completion_tokens": tokens.len(),
            }}))).into_response();
        }
        let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
        let (think_open, think_close, family_primed) = state.tokenizer.think_tags();
        // [gen] line (item 5b): did the request end inside the think block its prompt opened?
        // (the same rule as the streaming branch's `starts_in_reasoning`)
        let think_closed = text.contains(think_close);
        let starts_in_reasoning = stream_starts_in_reasoning(&prompt, think_open, family_primed, effort);
        let (reasoning, content) = split_think_in(&text, think_open, think_close, starts_in_reasoning);
        // SURFACE2 item 8: the generation ended inside the block (budget exhausted while reasoning,
        // or a stop string hit there) — the text is reasoning, as the streaming branch sends it.
        // The RENDERED PROMPT decides, never the effort arm of `starts_in_reasoning`: with
        // reasoning_effort low|high and thinking switched off by something that outranks it
        // (enable_thinking false, --thinking off), the prompt ends in a CLOSED block and the model
        // answers directly; that answer has no close tag either, and it is content (review finding).
        let prompt_opened_think = prompt.trim_end().ends_with(think_open);
        let (reasoning, content) = if prompt_opened_think && !think_closed && !unclosed_think_as_content() {
            unclosed_think_split(&text, parse_as_tool_calls)
        } else {
            (reasoning, content)
        };

        // The model emits calls as <tool_call><function=..><parameter=..>..  -- NOT as JSON. Turn them
        // into OpenAI tool_calls, or the harness just sees XML in the content and never invokes
        // anything. finish_reason MUST become "tool_calls": that is the flag every harness branches on.
        // The (content, tool_calls, finish) DECISION is crate::tools::finalize_parsed — the one
        // canonical serializer shared with the streaming mode, so the two can never diverge again
        // (2026-08-27 user report: a malformed call block vanished in streaming and leaked in JSON).
        // With tools offered, the model's LITERAL output is the only artifact that settles a "the tool
        // ran but nothing happened" report. Log it when asked (--dump-tools=1), and ALWAYS log
        // it when tools were offered and we parsed nothing — that combination means either the model
        // declined, or it emitted a call we failed to understand, and those need very different fixes.
        // parse_as_tool_calls == false (keep-tools "none"): never parse — the reply is plain
        // content; finalize_parsed then returns the raw text with finish stop/length.
        let parsed = if parse_as_tool_calls {
            crate::tools::parse_served(&content, req.tools.as_deref(), &finish)
        } else {
            crate::tools::ParsedOutput { content: content.trim().to_string(), tool_calls: Vec::new() }
        };
        if req.tools.is_some() {
            let dump = crate::opts::var(crate::opt!("dump-tools")).is_ok();
            if dump || parsed.tool_calls.is_empty() {
                crate::dreprintln!("[req] raw model output ({} chars): {:?}", content.chars().count(),
                          content.chars().take(1200).collect::<String>());
            }
            if dump { crate::reprintln!("{}", dump_tools_line(&content)); }
        }
        // Phase-4 A2: a call to a tool whose schema declares parameters that arrives with EMPTY
        // arguments while the model emitted a NON-EMPTY body is an argument drop, not a quiet
        // success. Log the raw body so the class can never regress unseen again.
        for a in crate::tools::empty_arg_alarms(&content, req.tools.as_deref(), &parsed.tool_calls) {
            crate::reprintln!("[tool-args-alarm] {a}");
        }
        let parser_input = if tool_trace.is_some() { content.clone() } else { String::new() };
        let (content, tool_calls, finish) = crate::tools::finalize_parsed(&content, parsed, &finish);
        // TOOLEVAL: the same per-turn trace as the streaming branch - raw output (thinking
        // included), the parser input, and the finalized verdict the JSON response carries.
        if let Some(t) = &tool_trace {
            crate::tool_trace::log(t, &text, &parser_input, content.as_deref(), &tool_calls, &finish);
        }
        mt.finish(&finish);
        if !tool_calls.is_empty() {
            // Log the ARGUMENTS, not just the names. When opencode reported a write as successful and
            // no file appeared, the log said `tool_calls 1: ["write"]` — which is exactly enough to
            // know a tool was called and not nearly enough to know what it was told to do. The path the
            // model chose is the whole question.
            for t in &tool_calls {
                crate::reprintln!("[req] tool_call  {} {}({})", t.id, t.function.name, t.function.arguments);
            }
        }
        let tool_calls = if tool_calls.is_empty() { None } else { Some(tool_calls) };
        // the decoded-text span `content` came from (reasoning, tool-call markup and trimmed
        // whitespace carry no content logprobs)
        let content_range = logprobs_n
            .and_then(|_| content_byte_range(&text, content.as_deref().unwrap_or(""), think_close));

        // TOOLSCRUB1: emission-side scrub - AFTER the raw-args log above (the server log
        // stays raw, same class as --tool-trace / --dump-tools) and after the trace; only
        // the response body text changes. Off mode = borrow passthrough, no allocation.
        let content = content.map(|c| scrub.scrub_full(&c).into_owned());
        let reasoning = reasoning.map(|r| scrub.scrub_full(&r).into_owned());
        let tool_calls = tool_calls.map(|v| v.into_iter().map(|mut t| {
            t.function.arguments = scrub.scrub_full(&t.function.arguments).into_owned();
            t }).collect::<Vec<_>>());

        let (r_log, c_log) = gen_log_spans(reasoning.as_deref(), content.as_deref().unwrap_or(""),
                                           think_closed, starts_in_reasoning);
        log_generation(&state.tokenizer, &state.stop_ids, &completion_id, &finish,
                       tokens.last().copied(), prompt_len, tokens.len(), r_log, c_log,
                       tool_calls.as_ref().map(|v| v.len()).unwrap_or(0),
                       req.tools.as_ref().map(|t| t.len()).unwrap_or(0),
                       effort, presence_penalty, pp_source,
                       stop_kind(sched_reason.as_deref(), stop_string_hit), stop_info);
        dump_tokens(&completion_id, &tokens);
        let logprobs_json = logprobs_n.map(|_| {
            let Some(range) = content_range.clone() else {
                crate::reprintln!("[req] logprobs: the content is not one span of the generated text; returning null");
                return serde_json::Value::Null;
            };
            chat_logprobs_json(&state.tokenizer, &state.stop_ids, &tokens, &lps, range).unwrap_or_else(|| {
                crate::reprintln!("[req] logprobs: the engine sent no entry for some output token(s); returning null");
                serde_json::Value::Null
            })
        });
        let response = ChatCompletionResponse {
            id: completion_id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: req.model.clone().unwrap_or_else(|| state.model_name.clone()),
            choices: vec![ChatChoice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant".to_string(), content,
                    reasoning_content: reasoning, tool_calls,
                },
                finish_reason: spec_finish_reason(&finish).to_string(),
                stop_reason: stop_reason_of(&finish),
                logprobs: logprobs_json,
            }],
            usage: Usage {
                prompt_tokens: prompt_len,
                completion_tokens: tokens.len(),
                total_tokens: prompt_len + tokens.len(),
                prompt_tokens_details: PromptTokensDetails { cached_tokens: cached },
            },
            timings: make_timings(t0, first_tok, prompt_len, tokens.len()),
            session_id: otel_session,
        };
        let mut resp = Json(response).into_response();
        attach_schema_unenforced_header(&mut resp, schema_unenforced.is_some());
        resp
    }
}
/// Diagnostics-only (env `--dump-tokens=1`): one line per generation with the EXACT
/// generated token ids. Why it exists: a served-text comparison cannot distinguish "the same
/// tokens" from "different tokens that detokenize alike", and the bitwise-losslessness claim for a
/// speculative lane (AGENTS §2.6/§3, P14's `--spec-source dflash` A/B) is a statement about the
/// TOKEN sequence. Harness plumbing only — no response byte, no scheduler behaviour is touched.
/// (Twin of `--dump-prompt`, which does the same for the prompt ids.)
fn dump_tokens(id: &str, tokens: &[u32]) {
    if crate::opts::var(crate::opt!("dump-tokens")).is_err() { return; }
    let ids: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
    crate::reprintln!("[gen-ids] id={id} n={} ids=[{}]", tokens.len(), ids.join(","));
}

/// PERFFIX1: tokenize a large text on Tokio's blocking pool instead of the server's single request thread (render +
/// encode measured ~1 ms per 1,000 prompt tokens: a 100K-token prompt held every other stream's SSE writes, /metrics
/// and /health for ~100 ms). Same tokenizer, same call, same tokens; texts under 64 KiB stay inline (a thread hop
/// would cost more than it saves).
async fn encode_off_thread(tok: &Arc<QwenTokenizer>, text: &str, add_special: bool) -> anyhow::Result<Vec<u32>> {
    if text.len() < 64 * 1024 {
        return tok.encode(text, add_special);
    }
    let (tok, text) = (tok.clone(), text.to_owned());
    tokio::task::spawn_blocking(move || tok.encode(&text, add_special)).await
        .map_err(|e| anyhow::anyhow!("tokenizer task failed: {e}"))?
}

/// Phase-2 A3 observability: ONE line per generation carrying exactly the fields the Phase-1
/// ledger had to reconstruct — or could not recover at all — from the harness transcripts:
/// the final finish_reason, the terminal token id and whether it is a model stop token, and the
/// reasoning/content token split. `reasoning_tokens`/`content_tokens` are ENCODE-based (the two
/// text spans are re-tokenized), so their sum can differ from `completion_tokens` by
/// template/special-token effects; they are diagnostics, not billing. Log lines only: no
/// response byte, no sampling parameter and no scheduler behaviour is touched.
fn log_generation(tok: &QwenTokenizer, stop_ids: &[u32], id: &str, finish: &str,
                  terminal: Option<u32>, prompt_tokens: usize, completion_tokens: usize,
                  reasoning: Option<&str>, content: &str, tool_calls: usize,
                  tools_offered: usize, effort: Option<&str>, presence_penalty: f32,
                  pp_source: &str, stop_kind: &str, stop_info: Option<(f32, u32, bool)>) {
    // PERFFIX1: the line below prints in debug mode only — skip the two re-tokenizations (up to the whole generation)
    // otherwise; they ran on the request thread after every request for a line nobody saw.
    if !crate::boot_report::debug() { return; }
    let count = |s: Option<&str>| s.map(|x| tok.encode(x, false).map(|v| v.len()).unwrap_or(0))
        .unwrap_or(0);
    let r_tok = count(reasoning);
    let c_tok = count(Some(content));
    let is_stop = terminal.map(|t| stop_ids.contains(&t)).unwrap_or(false);
    crate::dreprintln!("[gen] id={} finish={} terminal_tok={} is_stop_tok={} reasoning_tokens={} \
               content_tokens={} completion_tokens={} prompt_tokens={} tool_calls={} \
               tools_offered={} effort={} presence_penalty={} pp_source={} stop_kind={} {}",
        id, finish, terminal.map(|t| t.to_string()).unwrap_or_else(|| "none".into()), is_stop,
        r_tok, c_tok, completion_tokens, prompt_tokens, tool_calls, tools_offered,
        effort.unwrap_or("<template-default>"), presence_penalty, pp_source,
        stop_kind, stop_info_fields(stop_info));
}

/// Phase-9 A.2 — the status route now carries the engine's OWN per-window decode telemetry
/// (`crate::tel`): mode (mtp|dflash2), tp width, df2 block, chosen depth, accept@k, yield
/// (tokens per verify forward), step p50/p90. Lock-free read; it never touches a decode step.
/// Clients (owner harness, accept_gate.py) use it to see TRUE alpha instead of inferring it
/// from wall-clock tokens. `status` stays "ok" so existing liveness probes are unaffected.
/// Prometheus text exposition (CF-P1b). Cheap to scrape: atomics read, one string built.
async fn metrics() -> Response {
    ([(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
     crate::metrics::render(engine_ok())).into_response()
}

async fn health() -> Response {
    // WP02: a DEAD engine answers 503 (a supervisor / load balancer acts on it)
    if !engine_ok() {
        return (StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"status": "dead", "telemetry": crate::tel::snapshot_json()}))).into_response();
    }
    Json(serde_json::json!({"status": "ok", "telemetry": crate::tel::snapshot_json()})).into_response()
}

// ─── POST /v1/tokenize ────────────────────────────────────────────────────────────────
// vLLM-compatible de-facto tokenization endpoint. /v1/tokenize has NO OpenAI spec — it is a
// community convention (vLLM, SGLang, llama.cpp's /tokenize, LiteLLM). The OpenAI-spec'd token
// count is the Responses API's /v1/responses/input_tokens/count — a DIFFERENT API family, not
// implemented here. Matched shape (PLAN/ADD_V1_TOKENIZE_PROMPT.md; vLLM's TokenizeRequest/
// TokenizeResponse): response fields {tokens, count, max_model_len}, truncation keeps the LAST n
// tokens (vLLM's left-truncation convention, floor 1), empty prompt -> count 0 (vLLM behavior),
// over-length (>= max_seq_len after truncate) -> 400 `context_length_exceeded` — the same
// threshold the chat path enforces.
//
// PURE TOKENIZER: `QwenTokenizer::encode` (or the chat-template render for `messages`) only —
// no scheduler submit, no forward, no KV, no GPU work. Cheap and synchronous.
//
// Deliberate divergence (owner-pinned contract): stock vLLM defaults `add_special_tokens` to
// FALSE on this endpoint; ours defaults TRUE, mirroring the engine's own serving path
// (chat_completions encodes the rendered template with `true`). The flag's meaning is the HF
// fast-tokenizer POST-PROCESSOR, not the chat template — vLLM never applies a chat template to a
// raw /tokenize prompt either. The shipped Qwen3.5/Hy3/GLM post-processors add nothing for raw
// text (verified against `transformers`), so true==false on those families BY REFERENCE and raw
// counts match vLLM's false default anyway; a tokenizer whose post-processor injects specials
// (Llama-3-style BOS) gets them with `true` (proven by tests/tokenize_golden_test.rs fixture).
async fn tokenize(State(state): State<AppState>, Json(body): Json<serde_json::Value>) -> Response {
    // The engine's standard error body (same shape as the context_length_exceeded 400 above).
    let bad = |msg: String| -> Response {
        (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
            "message": msg, "type": "invalid_request_error", "code": "invalid_request_error",
        }}))).into_response()
    };

    // `model`: required; must name the served model (must match GET /v1/models).
    let Some(model) = body.get("model").and_then(|v| v.as_str()) else {
        return bad("'model' is required and must name the served model (see GET /v1/models)".into());
    };
    if model != state.model_name {
        return bad(format!("model '{model}' not found. Available: {}", state.model_name));
    }

    // `add_special_tokens`: optional bool, default true (see the divergence note above).
    let add_special = match body.get("add_special_tokens") {
        None | Some(serde_json::Value::Null) => true,
        Some(serde_json::Value::Bool(b)) => *b,
        Some(_) => return bad("'add_special_tokens' must be a boolean".into()),
    };

    // `truncate_prompt_tokens`: optional int >= 1 (vLLM's field, vLLM's validation floor).
    let truncate: Option<usize> = match body.get("truncate_prompt_tokens") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::Number(n)) => match n.as_u64() {
            Some(v @ 1..) => Some(v as usize),
            _ => return bad("'truncate_prompt_tokens' must be an integer >= 1".into()),
        },
        Some(_) => return bad("'truncate_prompt_tokens' must be an integer >= 1".into()),
    };

    // `prompt` | `messages`: exactly one. Raw text is the vLLM shape; the token-id list is OUR
    // extension (golden/corpus round-trips); `messages` is vLLM's chat-template mode — the model's
    // chat template is applied (generation prompt included), so `count` equals the TRUE prompt
    // size of the equivalent chat request, usage.prompt_tokens included.
    let has_prompt = body.get("prompt").map_or(false, |v| !v.is_null());
    let has_messages = body.get("messages").map_or(false, |v| !v.is_null());
    if has_prompt && has_messages {
        return bad("provide exactly one of 'prompt' or 'messages'".into());
    }
    let mut tokens: Vec<u32> = if has_messages {
        // Messages mode: render EXACTLY as chat_completions does — same ChatMessage deserializer
        // (string or content-array, null content, tool_calls), same optional tools passthrough,
        // same reasoning_effort family normalization — then encode. Any divergence here would show
        // up as tokenize(messages) != usage.prompt_tokens for the same conversation.
        let msgs: Vec<ChatMessage> = match serde_json::from_value(body["messages"].clone()) {
            Ok(m) => m,
            Err(e) => return bad(format!("'messages' is not a valid chat array: {e}")),
        };
        let tools = match body.get("tools") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::Array(a)) => Some(a.clone()),
            Some(_) => return bad("'tools' must be an array".into()),
        };
        let effort = resolve_reasoning_effort(&state.tokenizer,
                                              body.get("reasoning_effort").and_then(|v| v.as_str()),
                                              state.reasoning_effort.as_deref());
        // W1: the messages mode of /v1/tokenize renders EXACTLY like chat_completions, including
        // `chat_template_kwargs` — otherwise tokenize(messages) diverges from usage.prompt_tokens.
        let kw = match body.get("chat_template_kwargs") {
            None | Some(serde_json::Value::Null) => None,
            Some(v @ serde_json::Value::Object(_)) => Some(v.clone()),
            Some(_) => return bad("'chat_template_kwargs' must be a JSON object".into()),
        };
        if let Err(e) = crate::tokenizer::enable_thinking_kwarg(kw.as_ref()) {
            return bad(e.to_string());
        }
        let rendered = match state.tokenizer.apply_chat_template(
            &msgs, tools.as_deref(), effort.as_deref(), kw.as_ref(), state.thinking) {
            Ok(p) => p,
            Err(e) => return bad(format!("chat template failed: {e}")),
        };
        // The model's ACTUAL resident tokenizer — token-identity with the reference is the
        // whole point of this endpoint. Never a naive split.
        match encode_off_thread(&state.tokenizer, &rendered, add_special).await {
            Ok(t) => t,
            Err(e) => return bad(format!("tokenization failed: {e}")),
        }
    } else {
        match body.get("prompt") {
            // An EMPTY prompt is not an error (vLLM returns count 0): availability probes and
            // count-additivity arithmetic want the empty result, not a 400.
            Some(serde_json::Value::String(text)) => {
                if text.is_empty() {
                    Vec::new()
                } else {
                    match encode_off_thread(&state.tokenizer, text, add_special).await {
                        Ok(t) => t,
                        Err(e) => return bad(format!("tokenization failed: {e}")),
                    }
                }
            }
            // Token-id round-trip (our extension per the plan; stock vLLM only accepts a string):
            // echo the ids verbatim. add_special_tokens has nothing to encode here — the ids ARE
            // tokens — so it does not apply (same as vLLM's id-prompt paths elsewhere).
            Some(serde_json::Value::Array(items)) => match prompt_token_ids(items, state.tokenizer.vocab_size()) {
                Ok(ids) => ids,
                Err(e) => return bad(e),
            },
            _ => return bad("'prompt' must be a string or a list of token ids, or use 'messages'"
                .into()),
        }
    };

    if let Some(n) = truncate {
        if tokens.len() > n {
            tokens.drain(..tokens.len() - n); // keep the LAST n — vLLM's left-truncation convention
        }
    }
    // Over-length: the SAME threshold the chat path enforces (prompt >= max_seq_len has zero
    // generation room and is rejected there), so tokenize(messages) can never bless a prompt the
    // chat request would 400. `code: context_length_exceeded` is the machine-readable signal
    // distinguishing "too long" from transient errors; truncate_prompt_tokens is the escape hatch
    // (it runs BEFORE this check — cap to any n that fits).
    if tokens.len() >= state.max_seq_len {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
            "message": format!("This model's maximum context length is {} tokens, but the prompt \
                came to {} tokens. Shorten the input, or pass truncate_prompt_tokens to keep the \
                last n tokens.", state.max_seq_len, tokens.len()),
            "type": "invalid_request_error", "code": "context_length_exceeded",
        }}))).into_response();
    }
    let count = tokens.len();
    crate::dreprintln!("[tokenize] model={model} n={count} add_special={add_special} truncate={truncate:?}");
    Json(serde_json::json!({
        "tokens": tokens,
        "count": count,
        "max_model_len": state.max_seq_len,
    }))
    .into_response()
}

// ─── POST /v1/detokenize ──────────────────────────────────────────────────────────────
// vLLM-compatible detokenization: the decode half of the tokenize pair. With BOTH endpoints a
// client can build a prompt of EXACTLY N tokens (encode corpus -> slice N ids -> decode) — the
// llama-bench approach to exact-context benchmarks — instead of converging on a count by
// tokenize->trim->re-tokenize iteration.
//
// Matched shape (vLLM's DetokenizeRequest/DetokenizeResponse):
//   POST {"model": "<served id>",          // required, must match /v1/models (our convention)
//         "tokens": [ids...],              // required; ints in [0, vocab); may be empty -> ""
//         "skip_special_tokens": false}    // optional bool, default FALSE — vLLM's default:
//                                          //   special tokens ARE included in the output text
//   -> 200 {"model": <echo>, "prompt": "<decoded text>"}
//
// PURE TOKENIZER: one whole-sequence `QwenTokenizer::decode` — the byte-level BPE decoder handles
// multi-byte characters split across tokens correctly when the FULL id list is decoded at once
// (the StreamByteDecoder exists only for incremental streaming chunks).
async fn detokenize(State(state): State<AppState>, Json(body): Json<serde_json::Value>) -> Response {
    let bad = |msg: String| -> Response {
        (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
            "message": msg, "type": "invalid_request_error", "code": "invalid_request_error",
        }}))).into_response()
    };

    // `model`: required; must name the served model (must match GET /v1/models).
    let Some(model) = body.get("model").and_then(|v| v.as_str()) else {
        return bad("'model' is required and must name the served model (see GET /v1/models)".into());
    };
    if model != state.model_name {
        return bad(format!("model '{model}' not found. Available: {}", state.model_name));
    }

    // `tokens`: required; a (possibly empty) array of ints within the vocab.
    let Some(items) = body.get("tokens").and_then(|v| v.as_array()) else {
        return bad("'tokens' is required and must be a list of token ids".into());
    };
    let vocab = state.tokenizer.vocab_size();
    let mut ids: Vec<u32> = Vec::with_capacity(items.len());
    for it in items {
        match it.as_u64() {
            Some(id) if (id as usize) < vocab => ids.push(id as u32),
            _ => return bad(format!(
                "'tokens' ids must be integers in [0, {}) (got {it})", vocab)),
        }
    }

    // `skip_special_tokens`: optional bool, default false (vLLM's default: specials INCLUDED).
    let skip_special = match body.get("skip_special_tokens") {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Bool(b)) => *b,
        Some(_) => return bad("'skip_special_tokens' must be a boolean".into()),
    };

    let prompt = match state.tokenizer.decode(&ids, skip_special) {
        Ok(p) => p,
        Err(e) => return bad(format!("detokenization failed: {e}")),
    };
    crate::reprintln!("[detokenize] model={model} n={} skip_special={skip_special}", ids.len());
    Json(serde_json::json!({
        "model": model,
        "prompt": prompt,
    }))
    .into_response()
}

/// vLLM-style RAW completion endpoint: the prompt continues verbatim — NO chat template, NO
/// thinking markers. This is the surface llama-benchy (the user's benchmark of record) drives;
/// parity requires it. Also carries `min_tokens`/`ignore_eos` (--exact-tg) end to end.
#[derive(Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    model: Option<String>,
    /// vLLM accepts string | token-id array (token arrays are used verbatim, untokenized).
    #[serde(default)]
    prompt: Option<serde_json::Value>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_top_k")]
    top_k: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    min_tokens: Option<usize>,
    #[serde(default)]
    ignore_eos: Option<bool>,
    #[serde(default)]
    seed: Option<u64>,
    /// Legacy completions `logprobs` (0..=20): return per-token log probabilities with this many
    /// alternatives (`choices[].logprobs` = tokens / token_logprobs / top_logprobs / text_offset).
    #[serde(default)]
    logprobs: Option<u32>,
    /// S-B9-REL-SMALLFIXES item 6: OpenAI streaming options. `include_usage: true` adds the final
    /// usage chunk to a streamed completion (it was accepted and dropped: a known v0.7.3 gap).
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    /// WP08/WP15: the same penalty / min_p fields and validation as the chat endpoint (unset =
    /// off: raw completions never took the server-wide penalty defaults).
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    min_p: Option<f32>,
}

async fn completions(
    State(state): State<AppState>,
    payload: Result<Json<CompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match payload { Ok(j) => j, Err(e) => return bad_json(e) };
    if let Err(msg) = validate_penalties(req.repetition_penalty, req.presence_penalty,
                                         req.frequency_penalty, req.min_p) {
        return bad_param(msg);
    }
    let logprobs_n = match completion_logprobs_param(req.logprobs, req.stream) {
        Ok(n) => n,
        Err(msg) => return bad_param(msg),
    };
    let prompt_tokens: Vec<u32> = match req.prompt.as_ref() {
        Some(serde_json::Value::String(t)) => match encode_off_thread(&state.tokenizer, t, true).await {
            Ok(v) => v,
            Err(e) => return (StatusCode::BAD_REQUEST, format!("tokenize failed: {e}")).into_response(),
        },
        // ISSUES 129: every element must be an integer in [0, vocab) — no element dropped, none truncated to u32
        // (an out-of-range id used to reach the GPU embedding gather unchecked)
        Some(serde_json::Value::Array(a)) if !a.is_empty() => match prompt_token_ids(a, state.tokenizer.vocab_size()) {
            Ok(v) => v,
            Err(e) => return (StatusCode::BAD_REQUEST, e).into_response(),
        },
        _ => return (StatusCode::BAD_REQUEST,
                     "prompt must be a string or a non-empty token-id array".to_string()).into_response(),
    };
    let prompt_len = prompt_tokens.len();
    // WP02: the same context budget as the chat path — the decode headroom is reserved (this cap
    // ignored it and ran 3 rows past the KV window at the end of an uncapped request)
    let room = generation_room(state.max_seq_len, prompt_len, state.decode_headroom);
    if prompt_len + 8 >= state.max_seq_len || room.is_none() {
        return (StatusCode::BAD_REQUEST, format!(
            "prompt {} tokens leaves no room within max_seq_len {}", prompt_len, state.max_seq_len)).into_response();
    }
    let req_max = req.max_tokens.unwrap_or(16).min(room.unwrap_or(0));
    let (tx, mut rx, bl) = tok_channel(); // LR-5: `bl` counts consumed events in the loops below
    let request = BatchRequest {
        prompt: prompt_tokens.clone(),
        max_new: req_max,
        temperature: req.temperature.unwrap_or_else(default_temperature),
        top_p: req.top_p.unwrap_or_else(default_top_p),
        top_k: req.top_k.unwrap_or_else(default_top_k),
        rep_penalty: req.repetition_penalty.unwrap_or(1.0),
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        min_p: req.min_p.unwrap_or(0.0),
        min_new: req.min_tokens.unwrap_or(0).min(req_max),
        ignore_eos: req.ignore_eos.unwrap_or(false),
        tx,
        seed: req.seed,
        ckpt_at: None,
        domain: crate::batch::Domain::General,
        received_at: std::time::Instant::now(),
        image_embeds: None,
        image_spans: Vec::new(),
        schema: None,
        logprobs: logprobs_n,
    };
    if admit_gate_busy() { return server_busy(); } // LR-4: --max-waiting
    let (_mn, _ie) = (request.min_new, request.ignore_eos);
    if !engine_ok() || state.scheduler.send(request).is_err() {
        return engine_unavailable(); // WP02
    }
    let mut mt = crate::metrics::Req::start(prompt_len); // /metrics
    crate::reprintln!("[req] completions prompt_tokens={} max_tokens={} min_tokens={} ignore_eos={} stream={}",
              prompt_len, req_max, _mn, _ie, req.stream);
    let model_name = req.model.clone().unwrap_or_else(|| state.model_name.clone());
    let cid = format!("cmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();

    if req.stream {
        let t0 = std::time::Instant::now();
        let include_usage = completions_include_usage(req.stream_options.as_ref()); // item 6
        // LANERES1: the keep-alive frame shape (comment or empty-choices data frame) is decided
        // before the generator below moves cid/created/model_name into the stream.
        let ka_frame = sse_keepalive_frame(&cid, created, &model_name, "text_completion.chunk");
        let stream = async_stream::stream! {
            let mut ntok: usize = 0;
            let mut cached = 0usize; // #8.2: prefix-cache-served prompt tokens (the usage chunk)
            let mut first_tok: Option<std::time::Instant> = None;
            // WP02: closed without a Finish = engine failure; errors are an SSE error event + "stop"
            let mut finish = "error: engine stopped".to_string();
            while let Some(ev) = recv_counted(&mut rx, &bl).await { // LR-5: the one decrement (H0)
                match ev {
                    // v0.7.3 (#8.2): counted for /metrics, and reported in the usage chunk
                    // when the request asked for one (stream_options.include_usage).
                    TokEvent::Admitted { cached_tokens } => {
                        cached = cached_tokens as usize;
                        mt.cached(cached_tokens as u64);
                    }
                    TokEvent::StopInfo { .. } => {} // [gen]-line data; this path prints no [gen] line
                    TokEvent::Logprob { .. } => {} // refused with stream=true (completion_logprobs_param)
                    TokEvent::Tok(t) => {
                        if first_tok.is_none() { first_tok = Some(std::time::Instant::now()); }
                        ntok += 1;
                        mt.tok();
                        let text = state.tokenizer.decode(&[t], true).unwrap_or_default();
                        let chunk = serde_json::json!({
                            "id": cid, "object": "text_completion.chunk", "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "text": text, "finish_reason": null}],
                        });
                        yield Ok::<_, std::convert::Infallible>(Event::default().data(chunk.to_string()));
                    }
                    TokEvent::Finish { reason } => { finish = reason; break; }
                }
            }
            drop(rx);
            if finish.starts_with("error") {
                crate::reprintln!("[req] completions stream ended in an engine error after {ntok} tokens: {finish}");
                yield Ok::<_, std::convert::Infallible>(Event::default().data(sse_error_event(&finish)));
            }
            mt.finish(&finish);
            let fr = if finish == "length" { "length" } else { "stop" };
            let mut chunk = serde_json::json!({
                "id": cid, "object": "text_completion.chunk", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "text": "", "finish_reason": fr}],
            });
            if let Some(r) = stop_reason_of(&finish) {
                chunk["choices"][0]["stop_reason"] = serde_json::json!(r); // WP08
            }
            yield Ok::<_, std::convert::Infallible>(Event::default().data(chunk.to_string()));
            if include_usage {
                let u = completions_usage_chunk(&cid, created, &model_name, prompt_len, ntok, cached,
                                                make_timings(t0, first_tok, prompt_len, ntok));
                yield Ok::<_, std::convert::Infallible>(Event::default().data(u.to_string()));
            }
            yield Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"));
            let dt = t0.elapsed().as_secs_f32();
            crate::reprintln!("[req] done   completions tok={} ({:.1} tok/s wall) finish={}", ntok, if dt>1e-6 {ntok as f32/dt} else {0.0}, fr);
        };
        // S-B9-REL-BENCHFIX: this stream never sent anything before its first token; it gets the
        // same idle comment frames as the chat stream (queue wait / long prefill).
        return sse_with_keepalive(stream, sse_keepalive(), ka_frame);
    }
    // non-streaming: collect everything, detokenize once, single response.
    let mut toks: Vec<u32> = Vec::with_capacity(req_max);
    let mut lps: Vec<TokLp> = Vec::new();
    let mut pending_lp: TokLp = None;
    // WP02: an engine error (or a channel closed without a Finish) is an HTTP error, not 200 "stop"
    let mut finish = "error: engine stopped".to_string();
    let mut stop_reason: Option<String> = None;
    let mut cached = 0usize; // v0.7.3 (#8.2): prefix-cache-served prompt tokens
    while let Some(ev) = recv_counted(&mut rx, &bl).await { // LR-5: the one decrement (H2: this loop never decremented)
        match ev {
            TokEvent::Admitted { cached_tokens } => {
                cached = cached_tokens as usize;
                mt.cached(cached_tokens as u64); // /metrics counter
            }
            TokEvent::StopInfo { .. } => {} // [gen]-line data; this path prints no [gen] line
            TokEvent::Logprob { logprob, top } => { pending_lp = Some((logprob, top)); }
            TokEvent::Tok(t) => { toks.push(t); lps.push(pending_lp.take()); mt.tok(); }
            TokEvent::Finish { reason } => {
                stop_reason = stop_reason_of(&reason); // WP08
                finish = reason;
                break;
            }
        }
    }
    drop(rx);
    if finish.starts_with("error") {
        mt.finish(&finish);
        return (engine_error_status(&finish), Json(serde_json::json!({"error": {
            "message": finish, "type": "server_error", "completion_tokens": toks.len(),
        }}))).into_response();
    }
    mt.finish(&finish);
    let finish = if finish == "length" { "length".to_string() } else { "stop".to_string() };
    let text = state.tokenizer.decode(&toks, true).unwrap_or_default();
    dump_tokens(&cid, &toks);
    let mut json = serde_json::json!({
        "id": cid, "object": "text_completion", "created": created, "model": model_name,
        "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": null}],
        "usage": {"prompt_tokens": prompt_len, "completion_tokens": toks.len(),
                  "total_tokens": prompt_len + toks.len(),
                  "prompt_tokens_details": {"cached_tokens": cached}},
    });
    if let Some(r) = stop_reason {
        json["choices"][0]["stop_reason"] = serde_json::json!(r); // WP08
    }
    if logprobs_n.is_some() {
        json["choices"][0]["logprobs"] = completion_logprobs_json(&state.tokenizer, &state.stop_ids, &toks, &lps, &text)
            .unwrap_or_else(|| {
                crate::reprintln!("[req] logprobs: the engine sent no entry for some output token(s); returning null");
                serde_json::Value::Null
            });
    }
    (StatusCode::OK, axum::Json(json)).into_response()
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/tokenize", post(tokenize))
        .route("/v1/detokenize", post(detokenize))
        .route("/v1/models", get(list_models))
        // S10 (REL_V0_7_3): the served id contains a slash (base_model, e.g.
        // Qwen/Qwen3.8-Flash-Next); a :id param matched only the %2F form. The wildcard
        // serves the raw-slash spelling AND the %2F form (decoded after routing); the
        // handler id comparison is unchanged. :id + *id cannot coexist in matchit.
        .route("/v1/models/*id", get(get_model))
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        // Base64 image bodies inflate ~4/3x; a high-res PNG at ~2-4 MB exceeds axum's 2 MB default
        // (""Failed to buffer the request body: length limit exceeded"" on image requests). Raise it
        // so images that other engines accept also arrive here.
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(state)
}

/// ISSUES 129: a token-id prompt is accepted only when EVERY element is an integer in `[0, vocab)`: nothing is
/// dropped, nothing is truncated to u32, so no out-of-range id can reach the embedding gather. Shared by
/// `/v1/tokenize` and `/v1/completions`.
pub(crate) fn prompt_token_ids(items: &[serde_json::Value], vocab: usize) -> Result<Vec<u32>, String> {
    items.iter().map(|it| match it.as_u64() {
        Some(id) if id < vocab as u64 => Ok(id as u32),
        _ => Err(format!("'prompt' ids must be integers in [0, {vocab}) (got {it})")),
    }).collect()
}

#[cfg(test)]
mod top_k_parse_tests {
    use super::{ChatCompletionRequest, CompletionRequest};

    fn chat(top_k: &str) -> Result<Option<usize>, String> {
        let body = format!(r#"{{"messages":[],"top_k":{top_k}}}"#);
        serde_json::from_str::<ChatCompletionRequest>(&body).map(|r| r.top_k).map_err(|e| e.to_string())
    }

    #[test]
    fn negative_top_k_means_off() {
        assert_eq!(chat("-1"), Ok(Some(0)));
        assert_eq!(chat("-7"), Ok(Some(0)));
        assert_eq!(chat("0"), Ok(Some(0)));
        assert_eq!(chat("20"), Ok(Some(20)));
        assert_eq!(chat("null"), Ok(None));
        assert!(chat("1.5").is_err());
        assert!(chat("\"5\"").is_err());
        let c: CompletionRequest = serde_json::from_str(r#"{"prompt":"x","top_k":-1}"#).unwrap();
        assert_eq!(c.top_k, Some(0));
        let c: CompletionRequest = serde_json::from_str(r#"{"prompt":"x"}"#).unwrap();
        assert_eq!(c.top_k, None);
    }

    #[test]
    fn full_unsigned_range_still_parses() {
        assert_eq!(chat("9223372036854775807"), Ok(Some(i64::MAX as usize)));
        assert_eq!(chat("9223372036854775808"), Ok(Some(1usize << 63)));
        assert_eq!(chat("18446744073709551615"), Ok(Some(u64::MAX as usize)));
        assert_eq!(chat("-9223372036854775808"), Ok(Some(0)));
        assert!(chat("18446744073709551616").is_err());
    }
}

#[cfg(test)]
mod logprobs_span_tests {
    use super::{content_byte_range, logprob_token_span};

    // ids 0..: "Hel", "lo", " wor", "ld", <stop>
    fn bytes_of(t: u32) -> Vec<u8> {
        ["Hel", "lo", " wor", "ld", ""][t as usize].as_bytes().to_vec()
    }

    #[test]
    fn span_clips_tokens_at_the_range_edges() {
        let toks = [0u32, 1, 2, 3, 4];
        // a stop string "lo" cut the text to "Hel"
        let v = logprob_token_span(bytes_of, &[4], &toks, 0..3);
        assert_eq!(v, vec![(0, 0, b"Hel".to_vec())]);
        // a stop string starting inside "lo" ("o w"): "Hello w" -> "Hell"
        let v = logprob_token_span(bytes_of, &[4], &toks, 0..4);
        assert_eq!(v, vec![(0, 0, b"Hel".to_vec()), (1, 3, b"l".to_vec())]);
        // content "world" of "Hello world": starts inside " wor"
        let v = logprob_token_span(bytes_of, &[4], &toks, 6..11);
        assert_eq!(v, vec![(2, 5, b"wor".to_vec()), (3, 9, b"ld".to_vec())]);
        // the whole text; the stop token is not listed
        assert_eq!(logprob_token_span(bytes_of, &[4], &toks, 0..usize::MAX).len(), 4);
        // empty content: nothing
        assert!(logprob_token_span(bytes_of, &[4], &toks, 11..11).is_empty());
    }

    #[test]
    fn content_range_skips_reasoning_and_trim() {
        let t = "thinking about it</think>\n\nThe answer.";
        let r = content_byte_range(t, "The answer.", "</think>").unwrap();
        assert_eq!(&t[r], "The answer.");
        // the same words inside the reasoning are not matched
        let t = "The answer. maybe</think> The answer.";
        let r = content_byte_range(t, "The answer.", "</think>").unwrap();
        assert_eq!(r.start, 26);
        // no reasoning block
        assert_eq!(content_byte_range("  hi ", "hi", "</think>"), Some(2..4));
        // tool-only reply: empty content
        let t = "<tool_call>x</tool_call>";
        assert_eq!(content_byte_range(t, "", "</think>"), Some(0..0));
        // content not one span of the text
        assert_eq!(content_byte_range("a b", "ab", "</think>"), None);
    }
}

#[cfg(test)]
mod logprobs_param_tests {
    use super::{chat_logprobs_param, completion_logprobs_param};

    #[test]
    fn chat_and_completion_logprobs_params() {
        assert_eq!(chat_logprobs_param(None, None, false), Ok(None));
        assert_eq!(chat_logprobs_param(Some(false), None, false), Ok(None));
        assert_eq!(chat_logprobs_param(Some(true), None, false), Ok(Some(0)));
        assert_eq!(chat_logprobs_param(Some(true), Some(20), false), Ok(Some(20)));
        assert!(chat_logprobs_param(Some(true), Some(21), false).is_err());
        assert!(chat_logprobs_param(None, Some(5), false).is_err(), "top_logprobs needs logprobs");
        assert!(chat_logprobs_param(Some(true), Some(5), true).is_err(), "not with stream=true yet");
        assert_eq!(chat_logprobs_param(None, None, true), Ok(None));
        assert_eq!(completion_logprobs_param(None, true), Ok(None));
        assert_eq!(completion_logprobs_param(Some(5), false), Ok(Some(5)));
        assert!(completion_logprobs_param(Some(21), false).is_err());
        assert!(completion_logprobs_param(Some(1), true).is_err());
    }
}

#[cfg(test)]
mod prompt_token_id_tests {
    use super::prompt_token_ids;
    use serde_json::json;

    #[test]
    fn every_id_must_be_an_integer_in_the_vocab() {
        assert_eq!(prompt_token_ids(&[json!(0), json!(5), json!(99)], 100), Ok(vec![0, 5, 99]));
        for bad in [json!(-1), json!(1.5), json!(2.0), json!("7"), json!(null), json!(100), json!(4294967296u64),
                    json!(4294967297u64)] {
            let e = prompt_token_ids(&[json!(1), bad.clone(), json!(2)], 100).unwrap_err();
            assert!(e.contains("[0, 100)") && e.contains(&bad.to_string()), "{bad}: {e}");
        }
    }
}

#[cfg(test)]
mod reasoning_start_tests {
    use super::{split_think_in, stream_starts_in_reasoning};

    /// ISSUES 149: a Qwen no-think render (the 0.8B/2B template default) with reasoning_effort
    /// low|high starts in CONTENT; only an open think tag (or hy_v3's effort arm) starts in reasoning.
    #[test]
    fn qwen_no_think_render_with_an_effort_is_content() {
        let no_think = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let think = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
        for effort in [None, Some("low"), Some("high"), Some("medium")] {
            assert!(!stream_starts_in_reasoning(no_think, "<think>", true, effort), "{effort:?}");
            assert!(stream_starts_in_reasoning(think, "<think>", true, effort), "{effort:?}");
        }
        // the reproduced answer (v0.7.5, streaming, effort low): a direct code block is content
        let (r, c) = split_think_in("```c\nvoid f(void) {}\n```", "<think>", "</think>",
                                    stream_starts_in_reasoning(no_think, "<think>", true, Some("low")));
        assert_eq!((r, c.as_str()), (None, "```c\nvoid f(void) {}\n```"));
        // hy_v3 keeps its effort arm, unchanged
        let hy = "<|hy_User|>hi<|hy_Assistant|>";
        assert!(stream_starts_in_reasoning(hy, "<think:opensource>", false, Some("low")));
        assert!(stream_starts_in_reasoning(hy, "<think:opensource>", false, Some("high")));
        assert!(!stream_starts_in_reasoning(hy, "<think:opensource>", false, None));
    }
}

#[cfg(test)]
mod context_budget_tests {
    // PR #4's context-budget contract. The PR's two further tests
    // (streaming_state_follows_the_rendered_prompt_not_the_model_family,
    // reasoning_effort_high_stays_a_thinking_level) test `prompt_ends_inside_think` /
    // `normalize_reasoning_effort`, which this leg deliberately does NOT port — dev's
    // reasoning semantics (P13 thinking toggle + the 35b4b15 effort-derived start) are kept.
    use super::{esc, generation_room};

    #[test]
    fn mtp_headroom_is_reserved_before_clamping() {
        assert_eq!(generation_room(4096, 1314, 16), Some(2766));
        assert_eq!(generation_room(4096, 1324, 16), Some(2756));
    }

    #[test]
    fn prompt_that_leaves_only_headroom_has_no_generation_room() {
        assert_eq!(generation_room(4096, 4080, 16), None);
        assert_eq!(generation_room(4096, usize::MAX, 16), None);
    }

    #[test]
    fn sse_json_escape_preserves_indented_go_code() {
        let text = concat!(
            "// New crée un cache.\n",
            "func New(capacity int) *CacheLRU {\n",
            "\treturn &CacheLRU{\r\n",
            "\t\tcapacity: capacity,\n",
            "\t\titems: make(map[interface{}]*list.Element),\n",
            "\t}\n",
            "}\n",
        );
        let payload = format!(r#"{{"delta":{{"content":"{}"}}}}"#, esc(text));
        let parsed: serde_json::Value =
            serde_json::from_str(&payload).expect("SSE data must contain valid JSON");

        assert_eq!(parsed["delta"]["content"], text);
        assert!(!payload.contains('\t'), "JSON payload must not contain literal tabs");
        assert!(!payload.contains('\r'), "JSON payload must not contain literal CRs");
    }
}

#[cfg(test)]
mod wp08_validation_tests {
    use super::{spec_finish_reason, stop_reason_of, validate_penalties};
    use crate::exl3_forward::PenParams;

    #[test]
    fn penalty_validation_contract() {
        assert!(validate_penalties(None, None, None, None).is_ok());
        assert!(validate_penalties(Some(1.05), Some(1.5), Some(0.3), Some(0.05)).is_ok());
        // rep < 1 is accepted (owner decision 2026-09-26); <= 0 / non-finite are 400s
        assert!(validate_penalties(Some(0.05), None, None, None).is_ok());
        assert!(validate_penalties(Some(0.0), None, None, None).is_err());
        assert!(validate_penalties(Some(-1.0), None, None, None).is_err());
        assert!(validate_penalties(Some(f32::NAN), None, None, None).is_err());
        assert!(validate_penalties(Some(f32::INFINITY), None, None, None).is_err());
        assert!(validate_penalties(None, Some(2.0), Some(-2.0), None).is_ok());
        assert!(validate_penalties(None, Some(2.01), None, None).is_err());
        assert!(validate_penalties(None, None, Some(-2.5), None).is_err());
        assert!(validate_penalties(None, None, None, Some(0.0)).is_ok());
        assert!(validate_penalties(None, None, None, Some(1.0)).is_ok());
        assert!(validate_penalties(None, None, None, Some(1.5)).is_err());
        assert!(validate_penalties(None, None, None, Some(-0.1)).is_err());
    }

    #[test]
    fn loop_detected_is_a_stop_with_a_reason() {
        assert_eq!(spec_finish_reason("loop_detected"), "stop");
        assert_eq!(stop_reason_of("loop_detected").as_deref(), Some("loop_detected"));
        assert_eq!(stop_reason_of("stop"), None);
        assert_eq!(spec_finish_reason("length"), "length");
    }

    #[test]
    fn inactive_penalties_stage_nothing() {
        // the reference implementation's alt() NoOp rule: defaults never launch a penalty kernel
        assert_eq!(PenParams::new(1.0, 0.0, 0.0, 1024, 1024), None);
        assert_eq!(PenParams::new(1.1, 0.0, 0.0, 0, 0), None);
        assert!(PenParams::new(1.1, 0.0, 0.0, 1024, 1024).is_some());
        assert!(PenParams::new(1.0, 0.0, 0.3, 1024, 1024).is_some());
        assert!(PenParams::new(0.9, 0.0, 0.0, 1024, 1024).is_some());
        assert!(PenParams::new(1.0, -0.5, 0.0, 16, 0).is_some());
    }
}

#[cfg(test)]
mod grow_find_tests {
    // WP02: the incremental streaming search must return exactly what the O(n) `str::find` did.
    use super::GrowFind;

    #[test]
    fn grow_find_matches_str_find_on_growing_buffers() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5eed);
        let alphabet = ["<", "tool", "_call", "</think>", "<think>", "a", "é", "日本", "\n", "<tool_call>", "</thi", "nk>", "int ", "main"];
        let needles = ["<tool_call", "</think>", "<think>", "int main", "日本", "é", ""];
        for _ in 0..400 {
            let mut acc = String::new();
            let mut finders: Vec<GrowFind> = needles.iter().map(|_| GrowFind::default()).collect();
            let mut base = 0usize;
            for _ in 0..60 {
                acc.push_str(alphabet[rng.gen_range(0..alphabet.len())]);
                // a monotone base on a char boundary, as the stream cursors are
                if rng.gen_bool(0.2) {
                    let mut b = rng.gen_range(base..=acc.len());
                    while !acc.is_char_boundary(b) { b -= 1; }
                    base = b.max(base);
                }
                // hay = a prefix of acc (the reasoning branch searches acc[..safe]), never below base
                let mut cut = acc.len() - rng.gen_range(0..=acc.len().min(4));
                while !acc.is_char_boundary(cut) { cut -= 1; }
                let cut = cut.max(base);
                for (f, n) in finders.iter_mut().zip(needles.iter()) {
                    let hay = &acc[..cut];
                    let want = hay[base..].find(n).map(|i| base + i);
                    assert_eq!(f.find(hay, base, n), want, "hay={hay:?} base={base} needle={n:?}");
                }
            }
        }
    }
}

#[cfg(test)]
mod model_id_tests {
    use super::{resolve_model_name, model_id_from_dir};

    fn temp_model_dir(tag: &str, readme: Option<&str>) -> String {
        let d = std::env::temp_dir().join(format!("velogb10-mid-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        if let Some(r) = readme { std::fs::write(d.join("README.md"), r).unwrap(); }
        d.to_string_lossy().to_string()
    }

    /// 2026-10-20: a quantized pack names ITSELF with `model_name:`; `base_model:` is the original
    /// model and only serves as the id when the card has no `model_name:` line.
    #[test]
    fn model_id_prefers_the_packs_own_name() {
        let card = "---\nlicense: mit\nmodel_name: doth4580/Naive-N0.5-Flash-EXL3-3.5bpw\nbase_model: NaiveAI/Naive-N0.5-Flash\nbase_model_relation: quantized\n---\ncard\n";
        let dir = temp_model_dir("own", Some(card));
        assert_eq!(model_id_from_dir(&dir), "doth4580/Naive-N0.5-Flash-EXL3-3.5bpw");
        let card = "---\nbase_model: NaiveAI/Naive-N0.5-Flash\nmodel_name: \"doth4580/Naive-N0.5-Flash-EXL3-4bpw\"\n---\n";
        let dir2 = temp_model_dir("own2", Some(card));
        assert_eq!(model_id_from_dir(&dir2), "doth4580/Naive-N0.5-Flash-EXL3-4bpw", "order in the card and quotes do not matter");
        let card = "---\nmodel_name:   \nbase_model: NaiveAI/Naive-N0.5-Flash\n---\n";
        let dir3 = temp_model_dir("own3", Some(card));
        assert_eq!(model_id_from_dir(&dir3), "NaiveAI/Naive-N0.5-Flash", "a blank model_name: falls through to base_model:");
        for d in [dir, dir2, dir3] { let _ = std::fs::remove_dir_all(&d); }
    }

    /// v0.7.3 (#8.3): the public model id resolves as flag > alias > model_name > base_model > dir name.
    #[test]
    fn model_id_resolution_order() {
        let card = "---\nbase_model: Qwen/Qwen3.8-27B\n---\nsome card\n";
        let dir = temp_model_dir("order", Some(card));
        assert_eq!(resolve_model_name(Some("flag-id"), Some("alias-id"), &dir), "flag-id");
        assert_eq!(resolve_model_name(None, Some("alias-id"), &dir), "alias-id");
        assert_eq!(resolve_model_name(Some("  flag-id  "), None, &dir), "flag-id", "verbatim + trimmed");
        assert_eq!(resolve_model_name(Some(""), Some("alias-id"), &dir), "alias-id",
                   "an empty flag value falls through to the alias");
        assert_eq!(resolve_model_name(None, None, &dir), "Qwen/Qwen3.8-27B",
                   "no flag: the model card base_model: line");
        let bare = temp_model_dir("bare", None);
        let dirname = std::path::Path::new(&bare).file_name().unwrap().to_string_lossy().to_string();
        assert_eq!(resolve_model_name(None, None, &bare), dirname, "no card: the directory name");
        assert_eq!(model_id_from_dir(&bare), dirname);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&bare);
    }
}

#[cfg(test)]
mod host_tests {
    use super::http_bind_addr;

    /// `--host`: default every interface; IPv4, bracketed/unbracketed IPv6 and `localhost` resolve; junk is refused
    /// with a message naming the flag (one test, so the shared option store is not raced).
    #[test]
    fn host_flag_resolves_and_refuses() {
        let o = crate::opt!("host");
        crate::opts::unset(o);
        assert_eq!(http_bind_addr(9000).unwrap().to_string(), "0.0.0.0:9000");
        crate::opts::set(o, "127.0.0.1");
        assert_eq!(http_bind_addr(9000).unwrap().to_string(), "127.0.0.1:9000");
        crate::opts::set(o, "[::1]");
        assert_eq!(http_bind_addr(9000).unwrap().to_string(), "[::1]:9000");
        crate::opts::set(o, "localhost");
        assert!(http_bind_addr(9000).unwrap().ip().is_loopback());
        crate::opts::set(o, "not a host!");
        assert!(http_bind_addr(9000).unwrap_err().to_string().contains("--host"));
        crate::opts::unset(o);
    }
}

#[cfg(test)]
mod harden_tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::atomic::AtomicBool;

    /// LR-8 proof: first N distinct reasons warn (Some(true)); duplicates never re-warn
    /// (Some(false)); past the cap nothing grows and exactly one suppression verdict (None)
    /// is returned, then silence.
    #[test]
    fn schema_seen_cap_bounds_the_set() {
        let mut seen = HashSet::new();
        let suppressed = AtomicBool::new(false);
        for i in 0..SCHEMA_SEEN_CAP {
            assert_eq!(schema_seen_note(&mut seen, &suppressed, &format!("reason {i}")), Some(true));
        }
        assert_eq!(seen.len(), SCHEMA_SEEN_CAP);
        // A repeat is still a duplicate.
        assert_eq!(schema_seen_note(&mut seen, &suppressed, "reason 0"), Some(false));
        // The first NEW reason past the cap yields the one-time suppression verdict...
        assert_eq!(schema_seen_note(&mut seen, &suppressed, "brand new"), None);
        // ...and the set did not grow.
        assert_eq!(seen.len(), SCHEMA_SEEN_CAP);
        // Every later new reason is silent (Some(false)) and still does not grow the set.
        for i in 0..50 {
            assert_eq!(schema_seen_note(&mut seen, &suppressed, &format!("over {i}")), Some(false));
        }
        assert_eq!(seen.len(), SCHEMA_SEEN_CAP);
    }
}

#[cfg(test)]
mod harden_lr5_toktx_tests {
    use super::*;

    /// LR-5 proof: the wrapper's unconsumed-event count is exact across sends, consumes and
    /// disconnect — this is the number the scheduler's cancel sweep compares against
    /// `--stream-backlog-events`.
    #[test]
    fn toktx_counts_unconsumed_backlog() {
        let (tx, mut rx, bl) = tok_channel();
        assert_eq!(tx.len(), 0);
        for i in 0..5u32 { tx.send(TokEvent::Tok(i)).unwrap(); }
        assert_eq!(tx.len(), 5, "5 sent, none consumed");
        try_recv_counted(&mut rx, &bl).unwrap();
        try_recv_counted(&mut rx, &bl).unwrap();
        assert_eq!(tx.len(), 3, "3 unconsumed after the handler read 2 via try_recv_counted");
        let clone = tx.clone();
        assert_eq!(clone.len(), 3, "clones share the count");
        drop(rx);
        assert!(tx.is_closed(), "a dropped receiver closes the sender");
    }
}

#[cfg(test)]
mod nonetools_tests {
    //! S-B9-REL-NONETOOLS: the --keep-tools-when-tool-choice-none contracts.
    //!  * OFF — the decision is the historical one for EVERY input (prompts byte-identical).
    //!  * ON  — a "none" request renders the SAME tools block as auto (same slice), differs
    //!          only by the appended do-not-call instruction (asserted on a Qwen-shaped
    //!          template: the common prefix is the whole conversation, the only difference is
    //!          the inserted line), and its reply is never parsed into tool_calls.
    use super::{tool_choice_decision, partial_overlap, GrowFind, TOOL_OPEN, hold_back_end, apply_force_line};

    fn tools_json() -> serde_json::Value {
        serde_json::json!([
            {"type": "function", "function": {"name": "get_weather",
             "description": "Get the weather for a city",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
            {"type": "function", "function": {"name": "read_file",
             "description": "Read a file from disk",
             "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
        ])
    }

    const DO_NOT_CALL: &str = "IMPORTANT: Do not call any tools. Answer in plain text only.";

    /// Flag OFF: the decision matches the legacy one for every tool_choice mode.
    #[test]
    fn off_flag_decision_matches_legacy_for_every_mode() {
        let tools = tools_json();
        let ts = tools.as_array().unwrap();
        for mode in ["none", "auto", "required"] {
            let v = serde_json::json!(mode);
            let (tft, line, parse) = tool_choice_decision(Some(&v), Some(ts.as_slice()), false);
            match mode {
                "none"     => { assert!(tft.is_none(), "legacy none removes the tools");
                                assert!(line.is_none(), "legacy none forces nothing"); }
                "auto"     => { assert!(tft.is_some()); assert!(line.is_none()); }
                "required" => { assert!(tft.is_some());
                                assert_eq!(line.as_deref(),
                                    Some("IMPORTANT: You MUST use one of the provided tools to answer. Do not answer in plain text.")); }
                _ => unreachable!(),
            }
            assert!(parse, "{mode}: reply parsing stays on with the flag off");
        }
        // named function
        let named = serde_json::json!({"type": "function", "function": {"name": "get_weather"}});
        let (tft, line, parse) = tool_choice_decision(Some(&named), Some(ts.as_slice()), false);
        assert!(tft.is_some());
        assert_eq!(line.as_deref(),
            Some("IMPORTANT: You MUST call the function `get_weather` with appropriate arguments before answering. Do not answer in plain text."));
        assert!(parse);
        // no tool_choice at all
        let (tft, line, parse) = tool_choice_decision(None, Some(ts.as_slice()), false);
        assert!(tft.is_some() && line.is_none() && parse);
        // flag ON but the request is not "none": decision unchanged
        let v = serde_json::json!("auto");
        let (tft, line, parse) = tool_choice_decision(Some(&v), Some(ts.as_slice()), true);
        assert!(tft.is_some() && line.is_none() && parse);
        let v = serde_json::json!("required");
        let (tft, line, parse) = tool_choice_decision(Some(&v), Some(ts.as_slice()), true);
        assert!(tft.is_some() && line.is_some() && parse);
    }

    /// Flag ON + "none" + tools: the tools slice is kept IDENTICAL and the do-not-call line
    /// is the only forcing; the reply is never parsed. "none" WITHOUT tools: unchanged.
    #[test]
    fn on_flag_none_keeps_identical_tools_slice_and_forbids() {
        let tools = tools_json();
        let ts = tools.as_array().unwrap();
        let v = serde_json::json!("none");
        let (tft, line, parse) = tool_choice_decision(Some(&v), Some(ts.as_slice()), true);
        let kept = tft.expect("the tools stay in the template");
        assert!(std::ptr::eq(kept.as_ptr(), ts.as_slice().as_ptr()),
                "the SAME tools slice — nothing reordered or re-serialized");
        assert_eq!(line.as_deref(), Some(DO_NOT_CALL));
        assert!(!parse, "the reply is never parsed into tool_calls");
        // "none" without tools on the request: nothing to keep, nothing to forbid
        let (tft, line, parse) = tool_choice_decision(Some(&v), None, true);
        assert!(tft.is_none() && line.is_none() && parse);
    }

    /// The prefix contract, end to end at the template level: a Qwen-shaped template (the
    /// structural skeleton of the real one — tools render inside the system block at the START)
    /// is fed exactly what the server feeds it (decision outputs + the last-message append).
    /// ON: the none-prompt shares the auto-prompt's whole prefix; removing the single inserted
    /// instruction makes them byte-equal. OFF: the none-prompt diverges at the tools block.
    #[test]
    fn on_flag_none_prompt_shares_the_auto_prefix() {
        let tpl = concat!(
            "{%- if tools %}{{ '<|im_start|>system\\n# Tools\\nYou can call: ' }}",
            "{%- for t in tools %}{{ t.function.name }} {% endfor %}{{ '<|im_end|>\\n' }}{%- endif %}",
            "{%- for m in messages %}{{ '<|im_start|>' }}{{ m.role }}\\n{{ m.content }}{{ '<|im_end|>\\n' }}{%- endfor %}",
            "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}",
        );
        let mut env = minijinja::Environment::new();
        env.add_template("t", tpl).unwrap();
        let render = |messages: &serde_json::Value, tools: Option<&serde_json::Value>| -> String {
            env.get_template("t").unwrap().render(minijinja::context! {
                messages => messages, tools => tools, add_generation_prompt => true,
            }).unwrap()
        };
        let tools = tools_json();
        let ts = tools.as_array().unwrap();
        let mut messages: Vec<serde_json::Value> = vec![
            serde_json::json!({"role": "user", "content": "What is the weather in Paris?"}),
            serde_json::json!({"role": "assistant", "content": "I will check.",
                "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}),
            serde_json::json!({"role": "tool", "content": "18C, clear"}),
            serde_json::json!({"role": "user", "content": "Summarize the conversation so far."}),
        ];

        // AUTO request: tools in the template, no forcing line.
        let prompt_auto = render(&serde_json::json!(messages), Some(&tools));

        // NONE + flag ON: tools stay (decision), the do-not-call line is appended to the LAST
        // message exactly as the server appends it.
        let v_none = serde_json::json!("none");
        let (tft, line, parse) = tool_choice_decision(Some(&v_none), Some(ts.as_slice()), true);
        let tft = tft.expect("tools kept");
        assert_eq!(tft.len(), ts.len());
        assert_eq!(line.as_deref(), Some(DO_NOT_CALL));
        assert!(!parse);
        // S4: drive the SERVER's append (apply_force_line), not a re-implementation — the
        // old body re-implemented the splice, so a handler change could not fail this test.
        let mut typed: Vec<crate::tokenizer::ChatMessage> =
            serde_json::from_value(serde_json::json!(messages)).unwrap();
        apply_force_line(&mut typed, DO_NOT_CALL);
        let prompt_none_on = render(&serde_json::to_value(&typed).unwrap(), Some(&tools));

        // The common prefix is everything up to the inserted instruction, and removing the
        // single inserted segment makes the prompts byte-equal.
        let cp = prompt_none_on.bytes().zip(prompt_auto.bytes())
            .take_while(|(a, b)| a == b).count();
        assert!(cp > prompt_auto.find("Summarize").unwrap(),
                "the shared prefix covers the system block AND the earlier messages");
        let inserted = format!("\n\n{DO_NOT_CALL}");
        assert_eq!(&prompt_none_on[..cp], &prompt_auto[..cp]);
        assert!(prompt_none_on[cp..].starts_with(&inserted),
                "the only difference at the split is the inserted instruction");
        let stripped = format!("{}{}", &prompt_none_on[..cp], &prompt_none_on[cp + inserted.len()..]);
        assert_eq!(stripped, prompt_auto,
                   "removing the single inserted instruction makes the none-prompt byte-equal to auto");

        // NONE + flag OFF (legacy): the tools vanish from the system block, so the divergence
        // starts inside the system head — long before any conversation content.
        let (tft, line, parse) = tool_choice_decision(Some(&v_none), Some(ts.as_slice()), false);
        assert!(tft.is_none() && line.is_none() && parse);
        let messages: Vec<serde_json::Value> = vec![
            serde_json::json!({"role": "user", "content": "What is the weather in Paris?"}),
            serde_json::json!({"role": "assistant", "content": "I will check.",
                "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}),
            serde_json::json!({"role": "tool", "content": "18C, clear"}),
            serde_json::json!({"role": "user", "content": "Summarize the conversation so far."}),
        ];
        let prompt_none_off = render(&serde_json::json!(messages), None);
        assert!(!prompt_none_off.contains("You can call:"),
                "legacy none renders NO tools block");
        let cp_off = prompt_none_off.bytes().zip(prompt_auto.bytes())
            .take_while(|(a, b)| a == b).count();
        assert!(cp_off < prompt_auto.find("You can call:").unwrap(),
                "the legacy none-prompt diverges inside the system head (cache miss from token 0)");
        assert!(cp_off < cp, "flag ON shares a far longer prefix than flag OFF");
    }

    /// The reply contract: a <tool_call> the model emits anyway is a tool call under auto,
    /// and plain content for a keep-tools "none" request (the server skips parse entirely).
    #[test]
    fn on_flag_none_reply_with_tool_call_stays_plain_content() {
        let raw = "Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        let tools = tools_json();
        let ts = tools.as_array().unwrap();
        // auto: the call is extracted and finish becomes tool_calls
        let parsed = crate::tools::parse(raw, Some(ts.as_slice()));
        assert_eq!(parsed.tool_calls.len(), 1, "auto parses the call");
        let (content, calls, finish) = crate::tools::finalize_parsed(raw, parsed, "stop");
        assert_eq!(content.as_deref(), Some("Let me check."), "prose before the call stays content");
        assert_eq!(calls.len(), 1);
        assert_eq!(finish, "tool_calls");
        // keep-tools "none": the server constructs the empty ParsedOutput — the raw text is
        // returned unchanged as content and the finish stays the plain stop/length.
        let parsed = crate::tools::ParsedOutput { content: raw.trim().to_string(), tool_calls: Vec::new() };
        let (content, calls, finish) = crate::tools::finalize_parsed(raw, parsed, "stop");
        assert_eq!(content.as_deref(), Some(raw), "the stray call stays plain content");
        assert!(calls.is_empty());
        assert_eq!(finish, "stop");
    }

    /// The streaming hold-back: the needle is None for a keep-tools "none" request, so a
    /// stray <tool_call> streams through as content; legacy keeps the hold-back.
    #[test]
    fn hold_back_disabled_for_keep_tools_none() {
        // S4 rewrite: drives the REAL hold_back_end (the old body asserted
        // `None.and_then(..)`, which cannot fail).
        let acc = "answer text <tool_call>\n<function=x>\n</function>\n</tool_call>";
        // keep-tools "none": tool_hold = None — the stray call streams through as content
        let none: Option<&'static str> = None;
        assert_eq!(hold_back_end(none, none.and_then(|_| acc.find(TOOL_OPEN)), acc.len(), acc, &["<think"]),
                   acc.len(), "with tool parsing off NOTHING is held back for tools");
        // legacy: tool_hold = Some(TOOL_OPEN) — emission stops AT the marker
        let legacy: Option<&'static str> = Some(TOOL_OPEN);
        assert_eq!(hold_back_end(legacy, legacy.and_then(|_| acc.find(TOOL_OPEN)), acc.len(), acc, &["<think"]),
                   "answer text ".len(), "legacy stops at the call");
        // a partial marker at the tail is held back so a split <tool_c cannot leak
        let tail = "answer <tool_c";
        assert_eq!(hold_back_end(legacy, legacy.and_then(|_| tail.find(TOOL_OPEN)), tail.len(), tail, &["<think"]),
                   "answer ".len(), "the partial marker prefix is not emitted");
    }

    /// S3: the same-chunk think-close decision — with the flag ON (tool_hold None) a stray
    /// `<tool_call>` arriving in the same chunk as `</think>` is NOT held back (was: held).
    #[test]
    fn think_close_same_chunk_uses_the_hold_back_decision() {
        let acc = "</think> sure, calling <tool_call>\n<function=x>";
        let lead = "</think> ".len(); // what the branch computes: past the close tag + lead ws
        let region = &acc[lead..];
        let hit = |hold: Option<&'static str>| hold.and_then(|_| region.find(TOOL_OPEN).map(|i| lead + i));
        // keep-tools "none": the whole chunk streams as content
        assert_eq!(hold_back_end(None, hit(None), acc.len(), region, &["<think"]), acc.len(),
                   "S3: with tool parsing off the same-chunk call is plain content");
        // legacy: held at the call, exactly like the steady-state content branch
        let legacy: Option<&'static str> = Some(TOOL_OPEN);
        assert_eq!(hold_back_end(legacy, hit(legacy), acc.len(), region, &["<think"]),
                   lead + "sure, calling ".len(), "legacy holds at the call");
    }
}

#[cfg(test)]
/// REL_V0_7_3 review H1: the tests the HARDEN report CLAIMED but never wrote, now real.
/// Every test below drives the actual production functions (`over_wait_limit`, `server_busy`,
/// `over_backlog`, `tok_channel` + `recv_counted`/`try_recv_counted`) — no hand-simulated
/// counters, no re-implemented predicates. Listed by name in the dated note in
/// PLAN/REL_HARDEN_REPORT.md.
mod harden_review_h1_tests {
    use super::*;

    /// LR-4 admit-bound arithmetic: unlimited at 0, refusal starts exactly at cap + max_waiting.
    #[test]
    fn wait_limit_arithmetic_is_exact() {
        assert!(!over_wait_limit(4, 4, 256), "all lanes busy, nobody waiting");
        assert!(!over_wait_limit(4 + 255, 4, 256), "max_waiting-1 waiting is still admissible");
        assert!(over_wait_limit(4 + 256, 4, 256), "the boundary fires: inflight == cap + max_waiting");
        assert!(over_wait_limit(10_000, 4, 256), "far beyond the bound still fires");
        assert!(!over_wait_limit(u64::MAX, 4, 0), "0 = unlimited: never over");
        assert!(!over_wait_limit(0, 0, 0), "0 = unlimited holds at zero inflight too");
    }

    /// LR-4 refusal shape: 503 + `Retry-After: 5` + the engine's error JSON with code
    /// `server_busy`, and the rejection is COUNTED (velogb10_requests_rejected_total).
    #[tokio::test]
    async fn server_busy_response_shape() {
        let before = crate::metrics::render(true);
        let resp = server_busy();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(resp.headers().get(axum::http::header::RETRY_AFTER).unwrap(), "5");
        let body = axum::body::to_bytes(resp.into_body(), 1 << 16).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "server_busy", "body: {v}");
        assert_eq!(v["error"]["type"], "server_error");
        assert!(v["error"]["message"].as_str().unwrap().contains("server busy"));
        let after = crate::metrics::render(true);
        let cnt = |s: &str| s.lines().find(|l| l.starts_with("velogb10_requests_rejected_total"))
            .unwrap().split_whitespace().last().unwrap().parse::<u64>().unwrap();
        assert!(cnt(&after) > cnt(&before), "server_busy must count the rejection");
        // MAXBATCH1: the 503 is the refused arm of the over-cap arrival class.
        let oc = |s: &str| s.lines().find(|l| l.starts_with("velogb10_over_cap_arrivals_total{disposition=\"rejected\"}"))
            .unwrap().split_whitespace().last().unwrap().parse::<u64>().unwrap();
        assert!(oc(&after) > oc(&before), "server_busy must count the over-cap rejection");
        assert!(after.contains("velogb10_over_cap_arrivals_total{disposition=\"queued\"}"), "the queued series is always present");
    }

    /// LR-5 backlog predicate: fires AT the limit (>=), never below it; 0 = unlimited.
    #[test]
    fn backlog_predicate_fires_at_limit_only() {
        assert!(!over_backlog(0, 8192));
        assert!(!over_backlog(8191, 8192), "one below the limit is admissible");
        assert!(over_backlog(8192, 8192), "AT the limit the sweep fires");
        assert!(over_backlog(99_999, 8192));
        assert!(!over_backlog(99_999, 0), "0 = unlimited");
        assert!(!over_backlog(0, 0));
    }

    /// Handler-level (a): the REAL helper moves the unconsumed counter by exactly ONE per
    /// received event (H0: the double decrement left N-2K).
    #[tokio::test]
    async fn recv_counted_moves_backlog_exactly_once() {
        let (tx, mut rx, bl) = tok_channel();
        for i in 0..10u32 { tx.send(TokEvent::Tok(i)).unwrap(); }
        assert_eq!(tx.len(), 10, "N sent, none consumed");
        for k in 0..3u32 {
            assert!(matches!(recv_counted(&mut rx, &bl).await, Some(TokEvent::Tok(t)) if t == k));
        }
        assert_eq!(tx.len(), 7, "N-K after the handler read K via recv_counted");
        // the sync twin agrees (the bench/gate drains' path)
        assert!(matches!(try_recv_counted(&mut rx, &bl), Some(TokEvent::Tok(_))));
        assert_eq!(tx.len(), 6, "try_recv_counted decrements exactly once too");
    }

    /// Handler-level (b): a reader that consumed 20,000 events and THEN stalls trips the
    /// predicate only when UNCONSUMED events reach the limit — the test that would have
    /// caught both the H0 double decrement and the H2 missing decrement.
    #[tokio::test]
    async fn stalled_reader_trips_only_on_unconsumed_events() {
        let (tx, mut rx, bl) = tok_channel();
        let limit = 8192usize;
        for i in 0..20_000u32 { tx.send(TokEvent::Tok(i)).unwrap(); }
        for _ in 0..20_000 { recv_counted(&mut rx, &bl).await.unwrap(); }
        assert_eq!(tx.len(), 0, "a healthy reader leaves zero unconsumed events");
        assert!(!over_backlog(tx.len(), limit), "consumed history must not count");
        // ...then the reader stalls: 8191 unread are admissible, the 8192nd trips.
        for i in 0..8191u32 { tx.send(TokEvent::Tok(i)).unwrap(); }
        assert!(!over_backlog(tx.len(), limit), "8191 unconsumed < 8192");
        tx.send(TokEvent::Tok(0)).unwrap();
        assert!(over_backlog(tx.len(), limit), "8192 unconsumed trips the sweep");
    }

    /// Handler-level (c): a 70,000-event generation consumed by a healthy reader never trips
    /// even the default limit (65536) — the H2 bug cancelled exactly this non-streaming run.
    #[tokio::test]
    async fn healthy_70k_event_generation_never_trips_default_limit() {
        let (tx, mut rx, bl) = tok_channel();
        for i in 0..70_000u32 { tx.send(TokEvent::Tok(i)).unwrap(); }
        for _ in 0..70_000 { recv_counted(&mut rx, &bl).await.unwrap(); }
        assert_eq!(tx.len(), 0, "70,000 events, all consumed by the helper");
        assert!(!stream_backlog_exceeded(tx.len()),
            "a healthy 70k-token completion must never be cancelled by the sweep");
    }
}

#[cfg(test)]
/// S8 (REL_V0_7_3 review): the model id is escaped in every hand-built SSE chunk — a quote or
/// backslash in `--model-name` (honoured verbatim since v0.7.3) used to break every chunk.
mod s8_sse_escape_tests {
    use super::*;

    #[test]
    fn sse_chunks_escape_the_model_id() {
        let evil = "Qwen/Qwen3.8\"Flash\\Next";
        for chunk in [
            role_chunk("chatcmpl-1", 1, evil),
            content_chunk("chatcmpl-1", 1, evil, "answer"),
            reasoning_chunk("chatcmpl-1", 1, evil, "thinking"),
            final_chunk("chatcmpl-1", 1, evil, "stop", ""),
        ] {
            let v: serde_json::Value = serde_json::from_str(&chunk)
                .unwrap_or_else(|e| panic!("chunk with an evil model id must stay valid JSON ({e}): {chunk}"));
            assert_eq!(v["model"].as_str(), Some(evil), "the id round-trips verbatim: {chunk}");
        }
        // and the text field stays escaped too (regression guard on the extraction)
        let c = content_chunk("chatcmpl-1", 1, "m", "line1\nline2\"q");
        let v: serde_json::Value = serde_json::from_str(&c).unwrap();
        assert_eq!(v["choices"][0]["delta"]["content"].as_str(), Some("line1\nline2\"q"));
    }
}

#[cfg(test)]
/// S10 (REL_V0_7_3 review): GET /v1/models/<id-with-slash> — both the raw-slash spelling and
/// the %2F form must route to the handler. Drives a real TCP listener (no tower test-util in
/// the tree) with the SAME two route patterns create_router registers.
mod s10_models_route_tests {
    #[tokio::test]
    async fn models_route_accepts_raw_slash_and_encoded_ids() {
        use axum::routing::get;
        async fn echo_id(axum::extract::Path(id): axum::extract::Path<String>) -> String { id }
        let app = axum::Router::new()
            .route("/v1/models/*id", get(echo_id));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
        let fetch = |path: String| async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
            let req = format!("GET {path} HTTP/1.1\r\nhost: t\r\nconnection: close\r\n\r\n");
            s.write_all(req.as_bytes()).await.unwrap();
            let mut buf = String::new();
            s.read_to_string(&mut buf).await.unwrap();
            buf
        };
        // raw slash (the EXL3 default id is slashed now): 200 + the full id extracts
        let r = fetch("/v1/models/Qwen/Qwen3.8-Flash-Next".into()).await;
        assert!(r.starts_with("HTTP/1.1 200"), "raw slash must route (S10): {r}");
        assert!(r.ends_with("Qwen/Qwen3.8-Flash-Next"), "body: {r}");
        // the percent-encoded form keeps working exactly as before
        let r = fetch("/v1/models/Qwen%2FQwen3.8-Flash-Next".into()).await;
        assert!(r.starts_with("HTTP/1.1 200"), "encoded form must keep routing: {r}");
        assert!(r.ends_with("Qwen/Qwen3.8-Flash-Next"), "body: {r}");
    }
}

/// S-B9-REL-PRPORT3 (review R4): the worker-PANIC arm's CPU fallback, on a blocking worker.
/// No CPU tower loaded: the request fails loudly (never a silently image-less 200).
pub(crate) async fn vision_panic_fallback(
    cpu_tower: Option<std::sync::Arc<crate::vision_tower::VisualTower>>,
    urls: Vec<String>,
    toks: Vec<u32>,
    panic: tokio::task::JoinError,
) -> anyhow::Result<crate::vision_encoder::PreparedVision> {
    match cpu_tower {
        Some(tower) => {
            crate::reprintln!("[vision] GPU worker panicked ({panic}), falling back to CPU tower");
            tokio::task::spawn_blocking(move || {
                crate::vision_encoder::prepare_vision_request(&tower, &urls, &toks)
            })
            .await
            .unwrap_or_else(|e2| Err(anyhow::anyhow!(
                "CPU fallback worker failed after the GPU panic ({panic}); {e2}"
            )))
        }
        None => Err(anyhow::anyhow!("vision worker failed: {panic}")),
    }
}

/// PR #16 (ported): GPU-tower dispatch with poisoned-lock recovery and CPU-tower fallback.
/// A cuBLAS/GEMM failure in the GPU tower now returns `Err` (never a panic mid-forward), and
/// the request falls back to the CPU reference tower so the image is never silently dropped
/// (the `len=0` hazard). A lock poisoned by an earlier panic is recovered via `into_inner()`:
/// the CUDA state is still owned by the guard and the next forward re-syncs its stream, so
/// vision keeps working instead of dying until restart (text traffic was never affected).
pub(crate) fn vision_dispatch_with_recovery<T>(
    gpu: &std::sync::Mutex<T>,
    run_gpu: impl FnOnce(&mut T) -> anyhow::Result<crate::vision_encoder::PreparedVision>,
    run_cpu: impl FnOnce() -> anyhow::Result<crate::vision_encoder::PreparedVision>,
) -> anyhow::Result<crate::vision_encoder::PreparedVision> {
    // S-B9-REL-PRPORT3 (review R1): the guard is scoped to `run_gpu` ONLY. The CPU reference
    // tower can take tens of seconds (53.8 s in the LEG A drill); holding the mutex across
    // `run_cpu()` serialized every other image request behind one slow fallback although the
    // GPU tower was free. Drop the guard before the CPU path — it is not under this mutex.
    let gpu_err = {
        let mut recovered = false;
        let mut gvt = match gpu.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                crate::reprintln!("[vision] vision_gpu lock was poisoned, recovering inner tower");
                recovered = true;
                poisoned.into_inner()
            }
        };
        match run_gpu(&mut gvt) {
            Ok(p) => {
                // S-B9-REL-PRPORT3 (review R5): a successful cycle through the recovered tower
                // proves the CUDA state is usable again — clear the poison flag so later requests
                // take the normal lock path instead of logging "poisoned" on every request.
                if recovered {
                    gpu.clear_poison();
                }
                return Ok(p);
            }
            Err(e) => e,
        }
    }; // guard dropped here — run_cpu runs unlocked
    crate::reprintln!("[vision] GPU tower failed ({gpu_err:#}), falling back to CPU tower");
    match run_cpu() {
        Ok(p) => Ok(p),
        // No CPU tower (or it failed too): the request errors out — with the GPU cause
        // in the chain (PR #16: never a silently image-less 200).
        Err(cpu) => Err(cpu.context(format!("GPU tower failed first ({gpu_err:#})"))),
    }
}

#[cfg(test)]
mod vision_fallback_tests {
    use super::*;

    fn cpu_ok() -> anyhow::Result<crate::vision_encoder::PreparedVision> {
        Ok(crate::vision_encoder::PreparedVision {
            expanded_tokens: vec![1, 2, 3],
            image_embeds: vec![0.25, -0.5, 1.0, 2.0],
            spans: vec![],
        })
    }

    fn gpu_err() -> anyhow::Result<crate::vision_encoder::PreparedVision> {
        Err(anyhow::anyhow!("vision gemm f32: injected cuBLAS failure (--vision-fail-gemm drill)"))
    }

    /// The core PR #16 guarantee: a failing GPU GEMM returns Err (no panic while the mutex is
    /// held — the lock is NOT poisoned) and the request still gets its image via the CPU tower.
    #[test]
    fn gpu_error_falls_back_to_cpu_and_lock_stays_clean() {
        let g = std::sync::Mutex::new(0usize);
        let out = vision_dispatch_with_recovery(&g, |_| gpu_err(), cpu_ok).unwrap();
        assert_eq!(out.image_embeds, vec![0.25, -0.5, 1.0, 2.0]);
        assert!(!g.is_poisoned(), "a returned Err must not poison the tower mutex");
        assert!(g.lock().is_ok(), "the next GPU request must reach the tower again");
    }

    /// A panic in the GPU worker poisons the lock; the NEXT request still runs the GPU path
    /// (into_inner recovery) instead of dying until restart.
    #[test]
    fn poisoned_lock_is_recovered_and_gpu_path_resumes() {
        let g = std::sync::Mutex::new(0usize);
        let gg = &g;
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = gg.lock().unwrap();
            panic!("kernel launch boom (drill)");
        }));
        assert!(g.is_poisoned());
        let out = vision_dispatch_with_recovery(&g,
            |t| { *t = 7; cpu_ok() },           // into_inner() gave us the tower back
            || panic!("CPU must not run when the GPU path succeeds")).unwrap();
        assert_eq!(out.image_embeds.len(), 4);
        // S-B9-REL-PRPORT3 (review R5): the successful recovered cycle now clears the poison
        // flag — later requests take the normal lock path (no "poisoned" log every request).
        assert!(!g.is_poisoned(), "a successful recovered GPU cycle must clear the poison flag");
        assert_eq!(*g.lock().unwrap(), 7);
    }

    /// S-B9-REL-PRPORT3 (review R1): the mutex guard must NOT be held across `run_cpu()` — a
    /// slow CPU fallback (53.8 s in the LEG A drill) must not block a concurrent healthy GPU
    /// request: the second request is served by the GPU tower while the first still sits in
    /// its CPU fallback.
    #[test]
    fn cpu_fallback_does_not_block_the_gpu_tower() {
        let g = std::sync::Mutex::new(0usize);
        let (entered, entered_rx) = std::sync::mpsc::channel::<()>();
        let (release, release_rx) = std::sync::mpsc::channel::<()>();
        let g_ref = &g; // move-closures capture the reference, not the Mutex (Receiver is !Sync)
        std::thread::scope(|s| {
            s.spawn(move || vision_dispatch_with_recovery(g_ref, |_| gpu_err(), move || {
                entered.send(()).unwrap();
                // bound the block so a regression cannot hang the suite forever
                let _ = release_rx.recv_timeout(std::time::Duration::from_secs(10));
                cpu_ok()
            }));
            entered_rx.recv_timeout(std::time::Duration::from_secs(5))
                .expect("first request must reach its CPU fallback");
            let (done, done_rx) = std::sync::mpsc::channel();
            let g2 = &g; // the spawn must OWN done (its lifetime is inside the scope body)
            s.spawn(move || {
                let r = vision_dispatch_with_recovery(g2,
                    |t| { *t += 1; cpu_ok() },
                    || panic!("the second request must be served by the GPU tower"));
                let _ = done.send(r.map(|p| p.image_embeds.len()));
            });
            match done_rx.recv_timeout(std::time::Duration::from_secs(5)) {
                Ok(Ok(n)) => assert_eq!(n, 4, "the GPU path must serve the second request"),
                Ok(Err(e)) => panic!("second request failed: {e:#}"),
                Err(_) => panic!("GPU path blocked behind a concurrent CPU fallback (guard held across run_cpu)"),
            }
            release.send(()).unwrap();
        });
    }

    /// No CPU tower loaded: the GPU error surfaces as a proper Err (HTTP 400), never a
    /// silently image-less 200.
    #[test]
    fn gpu_error_with_no_cpu_tower_is_an_error() {
        let g = std::sync::Mutex::new(0usize);
        let err = vision_dispatch_with_recovery(&g, |_| gpu_err(),
            || Err(anyhow::anyhow!("no vision tower loaded")))
            .err().expect("must be Err");
        assert!(format!("{err:#}").contains("no vision tower loaded"));
        assert!(format!("{err:#}").contains("injected cuBLAS failure"),
                "the GPU cause must stay in the error chain: {err:#}");
    }

    /// The worker-PANIC arm's no-CPU case: the request fails loudly, it is not served
    /// image-less (same message shape the .await unwrap_or_else arm produces).
    /// S-B9-REL-PRPORT3 (reviews R4+R9): the worker-PANIC arm with no CPU tower really errors —
    /// a REAL JoinError from a panicking blocking worker, through the actual fallback helper
    /// (the previous test asserted a string against itself and exercised nothing).
    #[tokio::test]
    async fn panic_fallback_without_cpu_tower_errors_loudly() {
        let panic = tokio::task::spawn_blocking(|| -> usize { panic!("vision worker drill") })
            .await
            .expect_err("a panicking blocking worker must surface as a JoinError");
        // PreparedVision is not Debug: route the Err out by match, not expect_err
        let err = match vision_panic_fallback(None, Vec::new(), Vec::new(), panic).await {
            Ok(_) => panic!("no CPU tower: the request must fail loudly"),
            Err(e) => e,
        };
        assert!(
            format!("{err:#}").contains("vision worker failed"),
            "the message must name the failing arm: {err:#}"
        );
    }
}

/// S-B9-REL-SMALLFIXES item 5: the pure parts of the `[gen]` line.
#[cfg(test)]
mod gen_line_tests {
    use super::*;

    /// (c) a loop-detector stop is `finish_reason: "stop"` for the client and `stop_kind=loop` in the log.
    #[test]
    fn stop_kind_names_the_cause() {
        assert_eq!(stop_kind(Some("stop"), false), "eos");
        assert_eq!(stop_kind(Some("length"), false), "length");
        assert_eq!(stop_kind(Some("context_length_exceeded"), false), "length");
        assert_eq!(stop_kind(Some("loop_detected"), false), "loop");
        assert_eq!(stop_kind(Some("cancelled"), false), "cancel");
        assert_eq!(stop_kind(Some("error: CUDA_ERROR_ILLEGAL_ADDRESS"), false), "error");
        assert_eq!(stop_kind(None, false), "error", "a channel closed without a Finish is an engine failure");
        assert_eq!(stop_kind(Some("capacity"), false), "other");
        // a stop string is matched on the HTTP side: it wins whatever the scheduler said (or did not)
        for r in [None, Some("stop"), Some("length"), Some("loop_detected")] {
            assert_eq!(stop_kind(r, true), "stop_string", "{r:?}");
        }
        // the client-facing mapping is unchanged: a loop stop is still a plain "stop" + stop_reason
        assert_eq!(spec_finish_reason("loop_detected"), "stop");
        assert_eq!(stop_reason_of("loop_detected").as_deref(), Some("loop_detected"));
        assert_eq!(stop_reason_of("stop"), None);
    }

    /// (b) a request that ends inside the think block its prompt opened: the text is reasoning.
    #[test]
    fn an_unclosed_reasoning_block_is_counted_as_reasoning() {
        let (open, close) = ("<think>", "</think>");
        // unclosed (e.g. max_tokens reached while thinking): split_think calls it content
        let text = "Let me work through this step by step";
        let (r, c) = split_think(text, open, close);
        assert_eq!((r.as_deref(), c.as_str()), (None, text), "split_think: no close tag = all content");
        assert_eq!(gen_log_spans(r.as_deref(), &c, text.contains(close), true), (Some(text), ""),
                   "the prompt opened the block and it never closed: reasoning");
        // the same text when the prompt did NOT open a think block (thinking off): content
        assert_eq!(gen_log_spans(r.as_deref(), &c, false, false), (None, text));
        // a closed block is untouched, with or without an answer after it
        let full = "plan the answer</think>\n\nThe answer is 4.";
        let (r, c) = split_think(full, open, close);
        assert_eq!(gen_log_spans(r.as_deref(), &c, full.contains(close), true), (Some("plan the answer"), "The answer is 4."));
        let only = "plan the answer</think>";
        let (r, c) = split_think(only, open, close);
        assert_eq!(gen_log_spans(r.as_deref(), &c, true, true), (Some("plan the answer"), ""));
        // nothing generated: nothing moves
        assert_eq!(gen_log_spans(None, "", false, true), (None, ""));
    }

    /// SURFACE2 item 8: a non-streaming generation that ends inside the think block the prompt
    /// opened. The canned stream is decoded text, token by token; the request ran out of budget
    /// before any close tag.
    #[test]
    fn a_generation_that_ends_inside_the_think_block_is_reasoning_not_content() {
        let (open, close) = ("<think>", "</think>");
        let toks = ["The", " user", " asks", " for", " 17", " *", " 23", ".", " 17", " *", " 20", " =", " 340", ","];
        let text: String = toks.concat();
        // BEFORE (reproduction): the non-streaming split hands the whole reasoning back as content
        let (r_old, c_old) = split_think(&text, open, close);
        assert_eq!((r_old, c_old.as_str()), (None, text.as_str()), "split_think alone: reasoning lost, content = the reasoning");
        // NOW: reasoning_content carries it and content is empty, at every length of the stream
        for n in 1..=toks.len() {
            let t: String = toks[..n].concat();
            assert!(!t.contains(close));
            let (r, c) = unclosed_think_split(&t, true);
            assert_eq!(r.as_deref(), Some(t.trim()), "after {n} tokens");
            assert_eq!(c, "", "content stays empty after {n} tokens");
            // the message as the response builder completes it: no call parsed, finish passes through
            let parsed = crate::tools::parse(&c, None);
            let (content, calls, finish) = crate::tools::finalize_parsed(&c, parsed, "length");
            assert_eq!((content.as_deref(), calls.len(), finish.as_str()), (Some(""), 0, "length"));
            // and the [gen] line counts the same spans as before the change
            assert_eq!(gen_log_spans(r.as_deref(), &c, false, true), (Some(t.trim()), ""));
        }
        // nothing generated at all: no reasoning, empty content
        assert_eq!(unclosed_think_split("", true), (None, String::new()));
        assert_eq!(unclosed_think_split(" \n", true), (None, String::new()));

        // a tool call made without closing the block (the model jumps from its thoughts straight to
        // the call): the thoughts are reasoning, the call is parsed from the rest — the stream's
        // hold-back at the marker, on the finished text
        let call = "I need the weather.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nAthens\n</parameter>\n</function>\n</tool_call>";
        let (r, c) = unclosed_think_split(call, true);
        assert_eq!(r.as_deref(), Some("I need the weather."));
        assert!(c.starts_with("<tool_call>"), "{c}");
        let parsed = crate::tools::parse(&c, None);
        let (content, calls, finish) = crate::tools::finalize_parsed(&c, parsed, "stop");
        assert_eq!((content, calls.len(), finish.as_str()), (None, 1, "tool_calls"));
        assert_eq!(calls[0].function.name, "get_weather");
        // the same calls as parsing the whole text (what the old split fed the parser)
        let whole = crate::tools::parse(call, None);
        assert_eq!(whole.tool_calls.len(), 1);
        assert_eq!(whole.tool_calls[0].function.arguments, calls[0].function.arguments);
        // a call-looking block the parser cannot read stays visible as content (never dropped)
        let broken = "Thinking.\n<tool_call>\nnot a call";
        let (r, c) = unclosed_think_split(broken, true);
        assert_eq!((r.as_deref(), c.as_str()), (Some("Thinking."), "<tool_call>\nnot a call"));
        let (content, calls, _) = crate::tools::finalize_parsed(&c, crate::tools::parse(&c, None), "length");
        assert_eq!((content.as_deref(), calls.len()), (Some("<tool_call>\nnot a call"), 0));
        // tool parsing off for the request (tool_choice none): the marker is ordinary reasoning text
        let (r, c) = unclosed_think_split(call, false);
        assert_eq!((r.as_deref(), c.as_str()), (Some(call), ""));

        // NARROW: the response builder takes this path only when the prompt opened the block AND
        // no close tag was generated; --unclosed-think content restores the old bytes
        let src = include_str!("server.rs");
        let guard = ["if prompt_opened_think && !think_closed && !unclosed_think_as_content() {\n", "            unclosed_think_split("].concat();
        assert_eq!(src.matches(guard.as_str()).count(), 1, "one call site, behind the narrow guard");
        // ...and "the prompt opened the block" is read from the rendered prompt's tail alone (the
        // effort arm of the streaming rule can be true for a prompt that CLOSED the block)
        let def = ["let prompt_opened_think = ", "prompt.trim_end().ends_with(think_open);"].concat();
        assert_eq!(src.matches(def.as_str()).count(), 1);
        let (open_tail, closed_tail) = ("<|im_start|>assistant\n<think>\n", "<|im_start|>assistant\n<think>\n\n</think>\n\n");
        assert!(open_tail.trim_end().ends_with(open) && !closed_tail.trim_end().ends_with(open));
        let o = &crate::opts::REG[crate::opts::idx("unclosed-think")];
        // the default is an owner ruling (2026-10-09: `reasoning`): the row's `def` is the one
        // place that states it; here and `opts::tests::choice_reads_the_row_default`
        assert_eq!((o.def, o.env), ("reasoning", ""), "a CLI option, no environment variable");
        if crate::opts::get(crate::opt!("unclosed-think")).is_none() { assert!(!unclosed_think_as_content()); }
        assert!(o.help.contains("v0.7.3 and earlier") && o.help.contains("Streaming responses"), "{}", o.help);
        // a closed block is untouched: split_think still decides
        let (r, c) = split_think("plan</think>\n\nThe answer is 391.", open, close);
        assert_eq!((r.as_deref(), c.as_str()), (Some("plan"), "The answer is 391."));
    }

    /// (a) the tail of the line.
    #[test]
    fn stop_info_fields_format() {
        assert_eq!(stop_info_fields(Some((0.97314, 1, false))), "stop_p=0.9731 stop_rank=1 stop_p_law=greedy");
        assert_eq!(stop_info_fields(Some((0.0, 4821, true))), "stop_p=0.0000 stop_rank=4821 stop_p_law=sampled");
        assert_eq!(stop_info_fields(None), "stop_p=na stop_rank=na stop_p_law=na");
        // SURFACE2 item 1: the law tag is APPENDED — a script that reads `stop_p=` / `stop_rank=`
        // (key followed by `=`) finds the same two fields at the same place as before.
        for info in [Some((0.5f32, 2u32, true)), Some((1.0, 1, false)), None] {
            let s = stop_info_fields(info);
            let keys: Vec<&str> = s.split(' ').map(|kv| kv.split('=').next().unwrap()).collect();
            assert_eq!(keys, ["stop_p", "stop_rank", "stop_p_law"]);
            assert!(s.starts_with("stop_p=") && s.matches("stop_p=").count() == 1, "`stop_p=` still matches one field: {s}");
        }
    }

    /// The three fields are APPENDED to the line (scripts parse the existing keys in order).
    #[test]
    fn the_gen_line_keeps_its_key_order_and_appends_the_new_fields() {
        let src = include_str!("server.rs");
        let at = |pat: &str| src.find(pat).unwrap_or_else(|| panic!("missing `{pat}`"));
        let a = at("crate::dreprintln!(\"[gen] id={} finish={} terminal_tok={} is_stop_tok={} reasoning_tokens={}");
        let mut end = a + 600;
        while !src.is_char_boundary(end) { end -= 1; } // OUT2: 1-byte rename shifted this onto an em-dash
        let tail = &src[a..end];
        let pos = |k: &str| tail.find(k).unwrap_or_else(|| panic!("[gen] line lost `{k}`"));
        let order = ["content_tokens={}", "completion_tokens={}", "prompt_tokens={}", "tool_calls={}", "tools_offered={}",
                     "effort={}", "presence_penalty={}", "pp_source={}", "stop_kind={}"];
        for w in order.windows(2) { assert!(pos(w[0]) < pos(w[1]), "{} must precede {}", w[0], w[1]); }
        assert!(tail.contains("pp_source={} stop_kind={} {}\","), "stop_kind and the stop_p/stop_rank fields close the line");
    }
}

/// S-B9-REL-SMALLFIXES item 6: the streamed /v1/completions usage chunk (a known gap in the
/// v0.7.3 CHANGELOG: `stream_options.include_usage` was accepted and dropped on this path).
#[cfg(test)]
mod completions_usage_tests {
    use super::*;

    #[test]
    fn the_request_field_is_parsed_and_only_an_explicit_true_asks_for_the_chunk() {
        let parse = |s: &str| serde_json::from_str::<CompletionRequest>(s).expect("valid request");
        let r = parse(r#"{"prompt": "x", "stream": true, "stream_options": {"include_usage": true}}"#);
        assert!(completions_include_usage(r.stream_options.as_ref()));
        let r = parse(r#"{"prompt": "x", "stream": true, "stream_options": {"include_usage": false}}"#);
        assert!(!completions_include_usage(r.stream_options.as_ref()));
        let r = parse(r#"{"prompt": "x", "stream": true, "stream_options": {}}"#);
        assert!(!completions_include_usage(r.stream_options.as_ref()));
        // not sent: no chunk (an existing client's stream keeps its bytes)
        let r = parse(r#"{"prompt": "x", "stream": true}"#);
        assert!(r.stream_options.is_none() && !completions_include_usage(r.stream_options.as_ref()));
    }

    #[test]
    fn the_usage_chunk() {
        let t = Timings { prompt_ms: 120.0, predicted_ms: 800.0, prompt_per_second: 100.0, predicted_per_second: 40.0 };
        let c = completions_usage_chunk("cmpl-abc", 1_760_000_000, "Qwen/Qwen3.8-Flash-Next", 12, 32, 8, t);
        // the OpenAI stream-usage shape: empty choices, top-level usage
        assert_eq!(c["choices"], serde_json::json!([]));
        assert_eq!(c["usage"]["prompt_tokens"], 12);
        assert_eq!(c["usage"]["completion_tokens"], 32);
        assert_eq!(c["usage"]["total_tokens"], 44);
        assert_eq!(c["usage"]["prompt_tokens_details"]["cached_tokens"], 8);
        // this path's ids and object type, as its text chunks carry them
        assert_eq!(c["id"], "cmpl-abc");
        assert_eq!(c["object"], "text_completion.chunk");
        assert_eq!(c["created"], 1_760_000_000i64);
        assert_eq!(c["model"], "Qwen/Qwen3.8-Flash-Next");
        // the timings extension, as the chat path's chunk
        assert_eq!(c["timings"]["predicted_per_second"], 40.0);
        // one line of JSON for one SSE data event
        let s = c.to_string();
        assert!(!s.contains('\n') && serde_json::from_str::<serde_json::Value>(&s).is_ok());
        // a cold request reports zero cached tokens, never omits the field
        let z = completions_usage_chunk("cmpl-0", 0, "m", 5, 0, 0, Timings { prompt_ms: 0.0, predicted_ms: 0.0, prompt_per_second: 0.0, predicted_per_second: 0.0 });
        assert_eq!(z["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
        assert_eq!(z["usage"]["total_tokens"], 5);
    }

    /// The chunk goes out after the finish chunk and before [DONE], only when asked (source pin:
    /// the unit suite has no scheduler to stream from).
    #[test]
    fn the_stream_sends_it_between_the_finish_chunk_and_done() {
        let src = include_str!("server.rs");
        let body = &src[src.find("async fn completions(").expect("the completions handler")..];
        let fin = body.find("\"choices\": [{\"index\": 0, \"text\": \"\", \"finish_reason\": fr}],").expect("the finish chunk");
        let gate = body.find(format!("if {}_usage {{", "include").as_str()).expect("the include_usage gate");
        let done = body[gate..].find("data(\"[DONE]\")").expect("[DONE] after the usage chunk") + gate;
        assert!(fin < gate && gate < done);
        assert!(body[gate..done].contains("completions_usage_chunk(&cid, created, &model_name, prompt_len, ntok, cached,"));
    }
}

/// S-B9-REL-BENCHFIX / BENCHFIX2: no chunk with a `choices` entry leaves the server before the
/// request's first generated token, and the role chunk leaves WITH that token.
///
/// These tests drive the generator the server runs (`chat_sse_stream`) through the response
/// builder the server uses (axum's SSE framing + `keepalive_frames`), with a scripted `TokEvent`
/// channel in place of the scheduler and a table in place of the tokenizer. Time is tokio's
/// paused clock: it advances only when every task is waiting, so each frame has an exact
/// virtual timestamp and nothing here depends on how fast or how loaded the machine is. One
/// test (`wire_bytes_over_a_real_socket`) reads the bytes from a TCP socket in real time and
/// asserts order only.
#[cfg(test)]
pub(crate) mod benchfix_sse_tests {
    use super::*;
    use crate::metrics::SchedState;
    use std::time::Duration;
    use tokio::time::Instant;

    fn idle() -> SchedState { SchedState { busy: false, work_age: None } }

    const CID: &str = "chatcmpl-benchfix";
    const MODEL: &str = "Qwen/Qwen3.8-Flash-Next";
    const KA: &str = ": keep-alive\n\n";

    /// The test vocabulary: token id = index.
    const PIECES: &[&str] = &[
        "Hello",                 // 0
        " world",                // 1
        "",                      // 2  a token that decodes to no text (a special token, half a UTF-8 char)
        "plan",                  // 3
        "</think>",              // 4
        "\n\n",                  // 5
        "<tool_call>",           // 6
        "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n", // 7
        "</tool_call>",          // 8
        " oops",                 // 9
        "<",                     // 10 could become "<tool_call" or "<think>": held back
        "</thi",                 // 11 could become "</think>": held back inside a think block
        "<tool_call_x y",        // 12 holds the marker "<tool_call" but is not a call tag
        " and <think>why</think>\n\nso", // 13 a think block that opens and closes inside one piece
    ];

    fn secs(n: u64) -> Duration { Duration::from_secs(n) }
    fn tok(t: u32) -> TokEvent { TokEvent::Tok(t) }
    fn fin(reason: &str) -> TokEvent { TokEvent::Finish { reason: reason.into() } }

    #[derive(Clone, Copy)]
    struct Shape { thinking: bool, tools: bool, usage: bool }
    const PLAIN: Shape = Shape { thinking: false, tools: false, usage: false };

    /// The generator's inputs, as the handler builds them (no tokenizer: no `[gen]` log line).
    fn cx(rx: mpsc::UnboundedReceiver<TokEvent>, bl: TokBacklog, sh: Shape, otel_req: Option<crate::otel::OtelRequest>) -> ChatSse {
        let weather = serde_json::json!({"type": "function", "function": {"name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}});
        ChatSse {
            rx, bl, mt: crate::metrics::Req::start(7), otel_req, otel_session: None,
            role_chunk: role_json(), completion_id: CID.into(), created: 1, model_name: MODEL.into(),
            stops: Vec::new(), think_open: "<think>", think_close: "</think>",
            starts_in_reasoning: sh.thinking,
            tool_hold: sh.tools.then_some(TOOL_OPEN), parse_as_tool_calls: sh.tools,
            req_tools: sh.tools.then(|| vec![weather]),
            include_usage: sh.usage, prompt_len: 7, t0: std::time::Instant::now(), gen_log: None,
            tool_trace: None,
            scrub: crate::tool_scrub::Scrubber::new(crate::tool_scrub::Mode::Off, &[]),
        }
    }

    /// The real generator over a scripted channel.
    fn chat(rx: mpsc::UnboundedReceiver<TokEvent>, bl: TokBacklog, sh: Shape)
        -> impl Stream<Item = Result<Event, axum::Error>> + Send + 'static {
        chat_sse_stream(cx(rx, bl, sh, None), || |t: u32| PIECES[t as usize].to_string())
    }

    /// The real generator with an OTel request handle (the telemetry test in otel.rs).
    pub(crate) fn chat_with_otel(rx: mpsc::UnboundedReceiver<TokEvent>, bl: TokBacklog, req: crate::otel::OtelRequest)
        -> impl Stream<Item = Result<Event, axum::Error>> + Send + 'static {
        chat_sse_stream(cx(rx, bl, PLAIN, Some(req)), || |t: u32| PIECES[t as usize].to_string())
    }

    /// The server's response builder with explicit scheduler gauges (`sse_with_keepalive` passes
    /// `metrics::sched_state`). `idle` = the scheduler has completed nothing and is not busy, so
    /// only the stream's own age counts.
    fn respond<S, E, F>(stream: S, every: Option<Duration>, sched: F) -> Response
    where
        S: Stream<Item = Result<Event, E>> + Send + 'static,
        E: Into<axum::BoxError> + Send + 'static,
        F: Fn() -> SchedState + Send + 'static,
    {
        match every {
            Some(d) => Sse::new(keepalive_frames(stream, d, SseKeepAliveFrame::Comment, SSE_KEEPALIVE_STALL, sched)).into_response(),
            None => Sse::new(stream).into_response(),
        }
    }

    /// Send each event at its time (seconds on the paused clock, from now; an Admitted event
    /// goes out at once, like the scheduler's). The sender is dropped after the last event.
    fn feed(tx: TokTx, script: Vec<(u64, TokEvent)>) {
        let t0 = Instant::now();
        tokio::spawn(async move {
            let _ = tx.send(TokEvent::Admitted { cached_tokens: 0 });
            for (at, ev) in script {
                tokio::time::sleep_until(t0 + secs(at)).await;
                let _ = tx.send(ev);
            }
        });
    }

    /// One wire frame: when it arrived (paused clock) and its raw bytes (one frame per event).
    struct Frame { at: Duration, raw: String }

    impl Frame {
        fn is_comment(&self) -> bool { self.raw.starts_with(':') }
        /// The `data:` payload of a data frame (None for a comment frame).
        fn data(&self) -> Option<&str> {
            self.raw.strip_prefix("data: ").map(|s| s.trim_end_matches('\n'))
        }
        fn json(&self) -> Option<serde_json::Value> {
            self.data().filter(|d| *d != "[DONE]").map(|d| serde_json::from_str(d)
                .unwrap_or_else(|e| panic!("data frame is not JSON ({e}): {d}")))
        }
        fn has_choices(&self) -> bool {
            self.json().map_or(false, |j| j["choices"].as_array().map_or(false, |a| !a.is_empty()))
        }
    }

    /// Every frame of the response body until it ends (or, for a stream that never ends, until
    /// `until` on the paused clock).
    async fn read_frames_until(resp: Response, until: Option<Duration>) -> Vec<Frame> {
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers()["content-type"], "text/event-stream");
        let t0 = Instant::now();
        let mut body = resp.into_body().into_data_stream();
        let mut out = Vec::new();
        loop {
            let next = match until {
                Some(u) => match tokio::time::timeout_at(t0 + u, body.next()).await { Ok(n) => n, Err(_) => break },
                None => body.next().await,
            };
            let Some(chunk) = next else { break };
            let raw = String::from_utf8(chunk.expect("body frame").to_vec()).expect("utf-8");
            assert!(raw.ends_with("\n\n"), "every SSE frame ends with a blank line: {raw:?}");
            out.push(Frame { at: t0.elapsed(), raw });
        }
        out
    }

    async fn read_frames(resp: Response) -> Vec<Frame> { read_frames_until(resp, None).await }

    fn data_frames(frames: &[Frame]) -> Vec<String> {
        frames.iter().filter_map(|f| f.data().map(str::to_string)).collect()
    }

    /// (seconds, "ka" | "role" | "data") per frame: the timeline of a response.
    fn timeline(frames: &[Frame]) -> Vec<(u64, &'static str)> {
        frames.iter().map(|f| {
            assert_eq!(f.at.subsec_nanos(), 0, "frames land on whole seconds of the paused clock: {:?}", f.at);
            (f.at.as_secs(), if f.is_comment() { "ka" } else if f.data() == Some(role_json().as_str()) { "role" } else { "data" })
        }).collect()
    }

    pub(crate) fn role_json() -> String { role_chunk(CID, 1, MODEL) }
    fn content(t: &str) -> String { content_chunk(CID, 1, MODEL, t) }
    fn reasoning(t: &str) -> String { reasoning_chunk(CID, 1, MODEL, t) }
    fn finish(reason: &str) -> String { final_chunk(CID, 1, MODEL, reason, "") }

    /// THE defect of post #46: with a first token after 35 s (queue wait + prefill) nothing with
    /// a `choices` entry is on the wire before it; the wait carries comment frames only, one per
    /// 10 s; the role chunk arrives with the first token and before its content.
    #[tokio::test(start_paused = true)]
    async fn no_choices_chunk_before_the_first_generated_token() {
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(35, tok(0)), (36, tok(1)), (37, fin("stop"))]);
        let sh = Shape { usage: true, ..PLAIN };
        let frames = read_frames(respond(chat(rx, bl, sh), Some(secs(10)), idle)).await;

        assert_eq!(timeline(&frames), [(10, "ka"), (20, "ka"), (30, "ka"), (35, "role"), (35, "data"), (36, "data"),
                                       (37, "data"), (37, "data"), (37, "data")]);
        for f in &frames[..3] {
            assert_eq!(f.raw, KA, "a waiting frame is the comment frame, nothing else");
            assert!(!f.raw.contains('{') && !f.raw.contains("data:") && !f.raw.contains("choices"),
                    "a comment frame carries no JSON and no choices");
        }
        // the first frame with a `choices` entry is the role chunk
        let first_choices = frames.iter().position(|f| f.has_choices()).unwrap();
        assert_eq!(first_choices, 3);
        assert_eq!(frames[3].json().unwrap()["choices"][0]["delta"], serde_json::json!({"role": "assistant"}));
        let d = data_frames(&frames);
        assert_eq!(d[..4], [role_json(), content("Hello"), content(" world"), finish("stop")]);
        let usage = frames[7].json().unwrap();
        assert!(usage["choices"].as_array().unwrap().is_empty(), "the usage chunk has empty choices");
        assert_eq!(usage["usage"]["prompt_tokens"], 7);
        assert_eq!(usage["usage"]["completion_tokens"], 2);
        assert_eq!(d[5], "[DONE]");
        // every data frame is one line of valid JSON (or the terminator)
        for f in frames.iter().filter(|f| !f.is_comment()) {
            let p = f.data().unwrap();
            assert!(!p.contains('\n') && (p == "[DONE]" || serde_json::from_str::<serde_json::Value>(p).is_ok()));
        }
    }

    /// W2 (review m3): thinking off, the reply is ONLY a tool call. The call is buffered, so
    /// the generator yields no delta until the end; the role chunk still leaves at the first
    /// generated token (35 s), not with the tool_calls delta (51 s). The silence while the call
    /// is buffered carries a comment frame.
    #[tokio::test(start_paused = true)]
    async fn a_held_tool_call_sends_the_role_chunk_at_the_first_generated_token() {
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(35, tok(6)), (40, tok(7)), (50, tok(8)), (51, fin("stop"))]);
        let sh = Shape { tools: true, ..PLAIN };
        let frames = read_frames(respond(chat(rx, bl, sh), Some(secs(10)), idle)).await;
        assert_eq!(timeline(&frames), [(10, "ka"), (20, "ka"), (30, "ka"), (35, "role"), (45, "ka"),
                                       (51, "data"), (51, "data"), (51, "data")]);
        let j: Vec<serde_json::Value> = frames.iter().filter_map(|f| f.json()).collect();
        assert_eq!(j.len(), 3);
        assert_eq!(j[0]["choices"][0]["delta"], serde_json::json!({"role": "assistant"}));
        let call = &j[1]["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(call["function"]["name"], "get_weather");
        assert_eq!(serde_json::from_str::<serde_json::Value>(call["function"]["arguments"].as_str().unwrap()).unwrap(),
                   serde_json::json!({"city": "Paris"}));
        assert_eq!(j[2]["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(frames.last().unwrap().data(), Some("[DONE]"));
    }

    /// The same rule for every way a first token can produce no delta: a token without text,
    /// and a held-back marker prefix. The role chunk is at the first token; the first delta is
    /// later.
    #[tokio::test(start_paused = true)]
    async fn the_role_chunk_does_not_wait_for_the_first_delta() {
        // a first token that decodes to nothing
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(5, tok(2)), (9, tok(0)), (10, fin("stop"))]);
        let frames = read_frames(respond(chat(rx, bl, PLAIN), None, idle)).await;
        assert_eq!(timeline(&frames), [(5, "role"), (9, "data"), (10, "data"), (10, "data")]);
        assert_eq!(data_frames(&frames), [role_json(), content("Hello"), finish("stop"), "[DONE]".to_string()]);
        // a first token that is held back as a possible marker prefix, in reasoning mode too
        for thinking in [false, true] {
            let (tx, rx, bl) = tok_channel();
            feed(tx, vec![(5, tok(10)), (9, tok(9)), (10, fin("length"))]);
            let sh = Shape { thinking, tools: true, usage: false };
            let frames = read_frames(respond(chat(rx, bl, sh), None, idle)).await;
            assert_eq!(timeline(&frames), [(5, "role"), (9, "data"), (10, "data"), (10, "data")], "thinking {thinking}");
            let text = if thinking { reasoning("< oops") } else { content("< oops") };
            assert_eq!(data_frames(&frames), [role_json(), text, finish("length"), "[DONE]".to_string()]);
        }
    }

    /// The data events are the old stream's: role chunk first, then the deltas, the finish
    /// chunk and the terminator, each with the bytes of its chunk builder.
    #[tokio::test(start_paused = true)]
    async fn the_data_events_keep_their_bytes_and_order() {
        // reasoning, the close tag, then content
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(3, tok(3)), (4, tok(4)), (5, tok(5)), (6, tok(0)), (7, tok(1)), (8, fin("length"))]);
        let sh = Shape { thinking: true, ..PLAIN };
        let frames = read_frames(respond(chat(rx, bl, sh), None, idle)).await;
        // (white space after the close tag is dropped only when it arrives in the same decode
        // chunk as the tag: the "\n\n" token here is a content delta, as it always was)
        assert_eq!(data_frames(&frames), [role_json(), reasoning("plan"), content("\n\n"), content("Hello"),
                                          content(" world"), finish("length"), "[DONE]".to_string()]);
        assert_eq!(timeline(&frames)[..2], [(3, "role"), (3, "data")], "role and the first reasoning delta together");
        // reasoning, then a tool call (the shape of the banked GPU gate `tool_call`)
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(3, tok(3)), (4, tok(4)), (5, tok(6)), (6, tok(7)), (7, tok(8)), (8, fin("stop"))]);
        let sh = Shape { thinking: true, tools: true, usage: false };
        let frames = read_frames(respond(chat(rx, bl, sh), None, idle)).await;
        let d = data_frames(&frames);
        assert_eq!(d.len(), 5, "{d:?}");
        assert_eq!(d[..2], [role_json(), reasoning("plan")]);
        let tc: serde_json::Value = serde_json::from_str(&d[2]).unwrap();
        assert_eq!(tc["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(d[3..], [finish("tool_calls"), "[DONE]".to_string()]);
        // a block that looked like a call but did not parse is surfaced as content at the end
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(3, tok(6)), (4, tok(9)), (5, fin("stop"))]);
        let sh = Shape { tools: true, ..PLAIN };
        let frames = read_frames(respond(chat(rx, bl, sh), None, idle)).await;
        assert_eq!(data_frames(&frames), [role_json(), content("<tool_call> oops"), finish("stop"), "[DONE]".to_string()]);
        assert_eq!(timeline(&frames)[..2], [(3, "role"), (5, "data")]);
    }

    /// A request that generates NO token sends the role chunk immediately before its terminal
    /// events and nothing while it waits: a finish without a token, an engine error before the
    /// first token (the error event has no `choices`), and an engine that went away.
    #[tokio::test(start_paused = true)]
    async fn a_request_without_a_token_sends_the_role_chunk_before_its_terminal_events() {
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(25, fin("stop"))]);
        let frames = read_frames(respond(chat(rx, bl, PLAIN), Some(secs(10)), idle)).await;
        assert_eq!(timeline(&frames), [(10, "ka"), (20, "ka"), (25, "role"), (25, "data"), (25, "data")]);
        assert_eq!(data_frames(&frames), [role_json(), finish("stop"), "[DONE]".to_string()]);

        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(8, fin("error: prefill failed: boom"))]);
        let frames = read_frames(respond(chat(rx, bl, PLAIN), None, idle)).await;
        assert_eq!(timeline(&frames), [(8, "role"), (8, "data"), (8, "data"), (8, "data")]);
        let d = data_frames(&frames);
        let err: serde_json::Value = serde_json::from_str(&d[1]).unwrap();
        assert_eq!(err["error"]["message"], "error: prefill failed: boom");
        assert_eq!(err["error"]["code"], 500);
        assert!(err.get("choices").is_none());
        assert_eq!(d[2..], [finish("stop"), "[DONE]".to_string()]);

        // the scheduler went away without a Finish (the sender is dropped at 4 s): never a hang
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(4, TokEvent::Admitted { cached_tokens: 0 })]);
        let frames = read_frames(respond(chat(rx, bl, PLAIN), None, idle)).await;
        let d = data_frames(&frames);
        assert_eq!(d[0], role_json());
        assert!(d[1].contains("engine stopped") && frames[0].at == secs(4));
    }

    /// S-B9-REL-INTEG B5: after an in-band engine error event, nothing but the finish chunk follows, unless the
    /// held span holds a COMPLETE call (the rule before SURFACE2B's held-tail flush; the flush now runs only on a
    /// finish that is not an error).
    #[tokio::test(start_paused = true)]
    async fn nothing_follows_an_engine_error_but_the_finish_chunk() {
        // text with a held tail ("<" could become a marker), then the engine fails
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(1, tok(0)), (2, tok(10)), (3, fin("error: engine broke"))]);
        let d = data_frames(&read_frames(respond(chat(rx, bl, Shape { tools: true, ..PLAIN }), None, idle)).await);
        assert_eq!(d.len(), 5, "{d:?}");
        assert_eq!(d[..2], [role_json(), content("Hello")]);
        let err: serde_json::Value = serde_json::from_str(&d[2]).unwrap();
        assert_eq!(err["error"]["message"], "error: engine broke");
        assert_eq!(d[3..], [finish("stop"), "[DONE]".to_string()], "the held `<` must not follow the error");
        // a COMPLETE call in the held span still becomes its tool_calls delta (master's rule)
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(1, tok(6)), (2, tok(7)), (3, tok(8)), (4, fin("error: engine broke"))]);
        let d = data_frames(&read_frames(respond(chat(rx, bl, Shape { tools: true, ..PLAIN }), None, idle)).await);
        assert_eq!(d.len(), 5, "{d:?}");
        assert!(d[1].contains("\"error\""), "{}", d[1]);
        assert!(d[2].contains("\"tool_calls\"") && d[2].contains("get_weather"), "{}", d[2]);
        assert_eq!(d[3..], [finish("tool_calls"), "[DONE]".to_string()]);
    }

    /// S-B9-REL-SURFACE2B R5: a reply that ENDS in text the hold-back was keeping. The emit sites
    /// hold the tail of the buffer while it could still become a marker; when the generation
    /// ended there, that tail was never sent: a streamed reply ending in `<` (or cut by
    /// max_tokens right after one) lost it, where the non-streaming response keeps it. Every
    /// case: the deltas of the stream add up to the generated text.
    #[tokio::test(start_paused = true)]
    async fn a_held_tail_leaves_at_the_finish() {
        async fn run(sh: Shape, toks: &[u32], reason: &str) -> Vec<String> {
            let (tx, rx, bl) = tok_channel();
            let mut script: Vec<(u64, TokEvent)> = toks.iter().enumerate().map(|(i, t)| (i as u64 + 1, tok(*t))).collect();
            script.push((toks.len() as u64 + 1, fin(reason)));
            feed(tx, script);
            let d = data_frames(&read_frames(respond(chat(rx, bl, sh), None, idle)).await);
            // nothing is lost and nothing is sent twice: reasoning + content deltas = the pieces
            // (a closed think block's tag is the one thing a stream never sends)
            let sent: String = d.iter().filter_map(|x| serde_json::from_str::<serde_json::Value>(x).ok()).map(|j| {
                let delta = &j["choices"][0]["delta"];
                format!("{}{}", delta["reasoning_content"].as_str().unwrap_or(""), delta["content"].as_str().unwrap_or(""))
            }).collect();
            let made: String = toks.iter().map(|t| PIECES[*t as usize]).collect();
            assert_eq!(sent, made.replace("</think>", ""), "{toks:?}");
            d
        }
        let end = |reason: &str| [finish(reason), "[DONE]".to_string()];
        let sh = |thinking, tools| Shape { thinking, tools, usage: false };
        for tools in [false, true] {
            // content: "<" may start "<think>" (always watched) or "<tool_call" (with tools)
            let d = run(sh(false, tools), &[0, 10], "length").await;
            assert_eq!(d, [vec![role_json(), content("Hello"), content("<")], end("length").to_vec()].concat(), "tools {tools}");
            let d = run(sh(false, tools), &[10], "stop").await;
            assert_eq!(d, [vec![role_json(), content("<")], end("stop").to_vec()].concat(), "a reply that is only the held character");
            // inside a think block that never closes, the tail is reasoning: "<" and "</thi"
            for held in [10u32, 11] {
                let d = run(sh(true, tools), &[3, held], "length").await;
                assert_eq!(d, [vec![role_json(), reasoning("plan"), reasoning(PIECES[held as usize])], end("length").to_vec()].concat(), "tools {tools}");
            }
            // after the block closed it is content again
            let d = run(sh(true, tools), &[3, 4, 0, 10], "length").await;
            assert_eq!(d, [vec![role_json(), reasoning("plan"), content("Hello"), content("<")], end("length").to_vec()].concat());
        }
        // text that holds the marker "<tool_call" but is not a call: held as a possible call,
        // not parsed, so it is content (inside an unclosed think block too: a call-looking block
        // the parser cannot read stays visible as content)
        let d = run(sh(false, true), &[0, 12], "stop").await;
        assert_eq!(d, [vec![role_json(), content("Hello"), content("<tool_call_x y")], end("stop").to_vec()].concat());
        let d = run(sh(true, true), &[3, 12, 9], "length").await;
        assert_eq!(d, [vec![role_json(), reasoning("plan"), content("<tool_call_x y oops")], end("length").to_vec()].concat());
        // without tools nothing watches "<tool_call": it streams as it comes
        let d = run(sh(false, false), &[0, 12], "stop").await;
        assert_eq!(d, [vec![role_json(), content("Hello"), content("<tool_call_x y")], end("stop").to_vec()].concat());
        // a think block that opens and closes inside the reply's LAST piece: reasoning, then
        // content, no raw tag (review 2026-10-10: the flush sent `why</think>\n\nso` as reasoning)
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(1, tok(0)), (2, tok(13)), (3, fin("stop"))]);
        let d = data_frames(&read_frames(respond(chat(rx, bl, sh(false, false)), None, idle)).await);
        assert_eq!(d, [vec![role_json(), content("Hello"), content(" and "), reasoning("why"), content("so")], end("stop").to_vec()].concat());
        // a parsed call is still the tool_calls delta alone: its text is not sent as content too
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(1, tok(0)), (2, tok(6)), (3, tok(7)), (4, tok(8)), (5, fin("stop"))]);
        let d = data_frames(&read_frames(respond(chat(rx, bl, sh(false, true)), None, idle)).await);
        assert_eq!(d.len(), 5, "{d:?}");
        assert_eq!(d[..2], [role_json(), content("Hello")]);
        assert!(d[2].contains("\"tool_calls\"") && !d[2].contains("\"content\""), "{}", d[2]);
        assert_eq!(d[3..], [finish("tool_calls"), "[DONE]".to_string()]);
    }

    /// A stop string ends the reply BEFORE itself, in a stream as in the non-streaming response
    /// (S-B9-REL-SURFACE2B, found while fixing R5). The stream used to check for a stop string
    /// after it had sent the token's text: a streamed reply carried its stop string, the rest of
    /// the token that completed it, and the start of a stop string that spanned tokens. Every
    /// case states the text the stream delivers; it is the text up to the stop string.
    #[tokio::test(start_paused = true)]
    async fn a_stop_string_is_not_part_of_the_stream() {
        async fn run(sh: Shape, stop: &str, toks: &[u32], reason: &str) -> Vec<String> {
            let (tx, rx, bl) = tok_channel();
            let mut script: Vec<(u64, TokEvent)> = toks.iter().enumerate().map(|(i, t)| (i as u64 + 1, tok(*t))).collect();
            script.push((toks.len() as u64 + 1, fin(reason)));
            feed(tx, script);
            let mut c = cx(rx, bl, sh, None);
            c.stops = vec![stop.to_string()];
            let d = data_frames(&read_frames(respond(chat_sse_stream(c, || |t: u32| PIECES[t as usize].to_string()), None, idle)).await);
            // the delivered text is the generated text up to the stop string (or all of it)
            let sent: String = d.iter().filter_map(|x| serde_json::from_str::<serde_json::Value>(x).ok()).map(|j| {
                let delta = &j["choices"][0]["delta"];
                format!("{}{}", delta["reasoning_content"].as_str().unwrap_or(""), delta["content"].as_str().unwrap_or(""))
            }).collect();
            let made: String = toks.iter().map(|t| PIECES[*t as usize]).collect();
            let want = made.find(stop).map_or(made.as_str(), |p| &made[..p]);
            // (a parsed tool call is its tool_calls delta, not text)
            if !d.iter().any(|x| x.contains("\"tool_calls\"")) {
                assert_eq!(sent, want.replace("</think>", ""), "stop {stop:?}, tokens {toks:?}");
            }
            d
        }
        let end = |reason: &str| [finish(reason), "[DONE]".to_string()];
        // the stop string is a whole token: it is not sent, and nothing after it is
        let d = run(PLAIN, "\n\n", &[0, 5, 1], "length").await;
        assert_eq!(d, [vec![role_json(), content("Hello")], end("stop").to_vec()].concat());
        // ...inside a token: the token's text before it is sent, the rest is not
        let d = run(PLAIN, "wor", &[0, 1, 9], "length").await;
        assert_eq!(d, [vec![role_json(), content("Hello"), content(" ")], end("stop").to_vec()].concat());
        // ...spanning two tokens: its first half was never sent
        let d = run(PLAIN, "lo wor", &[0, 1, 9], "length").await;
        assert_eq!(d, [vec![role_json(), content("Hel")], end("stop").to_vec()].concat());
        // a tail that only LOOKED like the start of the stop string leaves with the next token...
        let d = run(PLAIN, "lo!", &[0, 1], "stop").await;
        assert_eq!(d, [vec![role_json(), content("Hel"), content("lo world")], end("stop").to_vec()].concat());
        // ...or at the finish, when the reply ends there
        let d = run(PLAIN, "lo!", &[0], "length").await;
        assert_eq!(d, [vec![role_json(), content("Hel"), content("lo")], end("length").to_vec()].concat());
        // inside a think block, and after it closed
        let think = Shape { thinking: true, ..PLAIN };
        let d = run(think, " oo", &[3, 9, 4, 0], "length").await;
        assert_eq!(d, [vec![role_json(), reasoning("plan")], end("stop").to_vec()].concat());
        let d = run(think, "lo wor", &[3, 4, 0, 1], "length").await;
        assert_eq!(d, [vec![role_json(), reasoning("plan"), content("Hel")], end("stop").to_vec()].concat());
        // a stop string after a tool call: the call is parsed from the text before it
        let d = run(Shape { tools: true, ..PLAIN }, " oops", &[6, 7, 8, 9, 0], "length").await;
        assert_eq!(d.len(), 4, "{d:?}");
        assert!(d[1].contains("\"tool_calls\"") && d[1].contains("get_weather"), "{}", d[1]);
        assert_eq!(d[2..], [finish("tool_calls"), "[DONE]".to_string()]);
        // a stop string that never comes changes nothing
        let d = run(PLAIN, "zzz", &[0, 1], "stop").await;
        assert_eq!(d, [vec![role_json(), content("Hello"), content(" world")], end("stop").to_vec()].concat());
    }

    /// A stop string may hold any characters: only whole characters of it count as a prefix.
    #[test]
    fn a_partial_marker_is_whole_characters() {
        assert_eq!(partial_overlap("abc<to", "<tool_call"), 3);
        assert_eq!(partial_overlap("abc", "<tool_call"), 0);
        assert_eq!(partial_overlap("x\u{e9}", "\u{e9}!"), 2, "the two bytes of the first character");
        assert_eq!(partial_overlap("x\u{c3}\u{a0}", "\u{e9}!"), 0, "a different character with the same lead byte");
        assert_eq!(partial_overlap("\u{4f60}", "\u{4f60}\u{597d}"), 3);
        assert_eq!(partial_overlap("a\u{1f600}", "\u{1f600}\u{1f600}"), 4);
    }

    /// S-B9-REL-INTEG B1: `partial_overlap` is the old rule, in linear time. The stop string is the client's,
    /// and the old loop (every length from the top, `k` bytes compared for each) was quadratic in it: 1.7 ms
    /// per call at 10 KB, 308 ms at 200 KB, per token and per stop string. The assertion counts steps, not time.
    #[test]
    fn partial_overlap_is_the_old_rule_in_linear_time() {
        let naive = |s: &str, m: &str| (1..m.len()).rev().find(|&k| m.is_char_boundary(k) && s.ends_with(&m[..k])).unwrap_or(0);
        // every (text, marker) pair of short strings over a small alphabet that includes 2-, 3- and 4-byte characters
        let alpha = ["a", "b", "\u{e9}", "\u{4f60}", "\u{1f600}", "<"];
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
        let mut pairs = 0;
        for _ in 0..30_000 {
            let mut mk = |max: u64| -> String { (0..(next() % max)).map(|_| alpha[(next() % alpha.len() as u64) as usize]).collect() };
            let (s, m) = (mk(14), mk(9));
            assert_eq!(partial_overlap(&s, &m), naive(&s, &m), "text {s:?} marker {m:?}");
            pairs += 1;
        }
        assert_eq!(pairs, 30_000);
        // the hostile shapes: stop = a^(L-1) b against text a^L c (no overlap) and against a^L (overlap L-1)
        let mut last = 0u64;
        for l in [1_000usize, 10_000, 100_000] {
            let marker = format!("{}b", "a".repeat(l - 1));
            let (none, steps_none) = partial_overlap_steps(&format!("{}c", "a".repeat(l)), &marker);
            let (all, steps_all) = partial_overlap_steps(&"a".repeat(l), &marker);
            assert_eq!((none, all), (0, l - 1));
            for st in [steps_none, steps_all] {
                assert!(st <= 6 * l as u64, "L={l}: {st} steps is not linear in the stop string");
            }
            assert!(last == 0 || steps_none <= last * 11, "growth between sizes is not linear");
            last = steps_none;
        }
    }

    /// S-B9-REL-INTEG B2: an empty stop string stops nothing. It matches at offset 0 of every text, so the reply
    /// came back EMPTY (non-streaming before the stream fix, and the stream with it).
    #[tokio::test(start_paused = true)]
    async fn an_empty_stop_string_stops_nothing() {
        let r: ChatCompletionRequest = serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}],"stop":""}"#).unwrap();
        assert!(r.stop.is_empty(), "{:?}", r.stop);
        let r: ChatCompletionRequest = serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}],"stop":["","x",""]}"#).unwrap();
        assert_eq!(r.stop, vec!["x".to_string()]);
        // the stream itself, handed an empty stop string by a caller that did not go through the request
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(1, tok(0)), (2, tok(1)), (3, fin("stop"))]);
        let mut c = cx(rx, bl, PLAIN, None);
        c.stops = vec![String::new(), "zzz".to_string()];
        let d = data_frames(&read_frames(respond(chat_sse_stream(c, || |t: u32| PIECES[t as usize].to_string()), None, idle)).await);
        assert_eq!(d, [vec![role_json(), content("Hello"), content(" world")], vec![finish("stop"), "[DONE]".to_string()]].concat());
        // the non-streaming handler's post-hoc search skips empty strings too
        let src = include_str!("server.rs");
        let code = &src[..src.find("mod benchfix_sse_tests").unwrap()];
        assert!(code.contains("req.stop.iter().filter(|s| !s.is_empty()).filter_map(|s| text.find(s.as_str())).min()"));
    }

    /// `--dump-tools`: one line, the whole text, and it reads back exactly.
    #[test]
    fn the_tool_dump_line_carries_the_whole_text() {
        let raw = format!("<tool_call>\n<function=write_file>\n<parameter=content>\n{}\n\"q\" \u{e9}\t\u{1f600}\n</parameter>\n</function>\n</tool_call>", "x".repeat(5000));
        let line = dump_tools_line(&raw);
        assert!(!line.contains('\n') && line.starts_with("[dump-tools] {"), "{}", &line[..40]);
        let back: serde_json::Value = serde_json::from_str(line.strip_prefix("[dump-tools] ").unwrap()).unwrap();
        assert_eq!(back["raw"].as_str(), Some(raw.as_str()));
        // both response modes write it, next to the older 1200-character line
        let src = include_str!("server.rs");
        let code = &src[..src.find("mod benchfix_sse_tests").unwrap()];
        assert_eq!(code.matches(&["if dump { crate::reprintln!(\"{}\", ", "dump_tools_line(&"].concat()).count(), 2);
        assert!(crate::opt!("dump-tools").def().help.contains("[dump-tools]"));
    }

    /// `held_tail`, the split itself.
    #[test]
    fn the_held_tail_split() {
        let t = Some(TOOL_OPEN);
        let ht = |acc: &'static str, emitted, in_reasoning, tool| held_tail(acc, emitted, in_reasoning, tool, "</think>");
        assert_eq!(ht("a <", 2, false, t), ("", "<"));
        assert_eq!(ht("a <", 3, false, t), ("", ""), "everything was sent");
        assert_eq!(ht("a", 5, false, t), ("", ""), "a stop string cut the buffer below the cursor");
        assert_eq!(ht("plan</thi", 4, true, t), ("</thi", ""));
        assert_eq!(ht("plan<tool", 4, true, t), ("<tool", ""), "a marker PREFIX is reasoning text");
        assert_eq!(ht("plan<tool_call_x y", 4, true, t), ("", "<tool_call_x y"), "from a marker on: content");
        assert_eq!(ht("pl</<tool_call> x", 2, true, t), ("</", "<tool_call> x"));
        assert_eq!(ht("plan<tool_call_x", 4, true, None), ("<tool_call_x", ""), "tool parsing off: no marker");
        assert_eq!(ht("é<", 2, false, t), ("", "<"), "cursors are char boundaries");
        // a think block that closed inside the last piece: the tag is not sent
        assert_eq!(ht("a<think>b</think>\n\nc", 8, true, t), ("b", "c"));
        assert_eq!(ht("a<think>b</think>", 8, true, t), ("b", ""));
        assert_eq!(ht("x<tool_call> </think> y", 1, true, t), ("", "<tool_call> </think> y"), "a close tag after a held marker is the marker's text");
        assert_eq!(ht("a</think>b", 1, false, t), ("", "</think>b"), "in content mode a close tag is text");
    }

    /// --sse-keepalive 0: no comment frames; the data events are unchanged.
    #[tokio::test(start_paused = true)]
    async fn keepalive_off_sends_no_comment_frames() {
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(150, tok(0)), (151, fin("stop"))]);
        let frames = read_frames(respond(chat(rx, bl, PLAIN), None, idle)).await;
        assert_eq!(timeline(&frames), [(150, "role"), (150, "data"), (151, "data"), (151, "data")]);
    }

    /// The legacy /v1/completions stream has no role chunk; it gets the same idle comment frames.
    #[tokio::test(start_paused = true)]
    async fn a_stream_without_a_role_chunk_gets_the_same_comment_frames() {
        let s = async_stream::stream! {
            tokio::time::sleep(secs(25)).await;
            yield Ok::<_, std::convert::Infallible>(Event::default().data("{\"choices\":[{\"index\":0,\"text\":\"x\"}]}"));
            yield Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"));
        };
        let frames = read_frames(respond(s, Some(secs(10)), idle)).await;
        assert_eq!(timeline(&frames), [(10, "ka"), (20, "ka"), (25, "data"), (25, "data")]);
        assert!(frames[..2].iter().all(|f| f.raw == KA));
    }

    /// A client that leaves during the wait: dropping the response drops the generator and its
    /// receiver, so the scheduler's `tx.is_closed()` sweep sees it (the lane is freed) without
    /// any data having been written, before and after a keep-alive frame.
    #[tokio::test(start_paused = true)]
    async fn dropping_the_response_before_the_first_token_closes_the_event_channel() {
        for after_a_frame in [false, true] {
            let (tx, rx, bl) = tok_channel();
            let _ = tx.send(TokEvent::Admitted { cached_tokens: 0 });
            let mut body = respond(chat(rx, bl, PLAIN), Some(secs(10)), idle).into_body().into_data_stream();
            if after_a_frame {
                let first = String::from_utf8(body.next().await.unwrap().unwrap().to_vec()).unwrap();
                assert_eq!(first, KA, "the request is still waiting: only a comment frame so far");
            } else {
                // polled once (the generator has started and holds the receiver), nothing sent
                assert!(tokio::time::timeout(secs(3), body.next()).await.is_err());
            }
            assert!(!tx.is_closed());
            drop(body);
            assert!(tx.is_closed(), "the scheduler's cancel signal: the receiver is gone");
        }
    }

    /// W4 (review m4): keep-alive frames do not hide a hung engine. A stream that never
    /// produces an event behind a scheduler that never completes a unit of work gets frames for
    /// SSE_KEEPALIVE_STALL (60 s) and is then silent, exactly like a stream without keep-alive,
    /// so its client's idle timeout can fire. A stream that arrives while the scheduler is
    /// already wedged (busy, last completed work older than the bound) gets no frame at all.
    #[tokio::test(start_paused = true)]
    async fn keepalive_stops_when_the_scheduler_completes_no_work() {
        assert_eq!(SSE_KEEPALIVE_STALL, secs(60));
        let never = || tokio_stream::pending::<Result<Event, std::convert::Infallible>>();
        let frames = read_frames_until(respond(never(), Some(secs(10)), idle), Some(secs(600))).await;
        assert_eq!(timeline(&frames), [(10, "ka"), (20, "ka"), (30, "ka"), (40, "ka"), (50, "ka"), (60, "ka")]);
        // an interval longer than the bound: the first tick is already past it
        assert!(read_frames_until(respond(never(), Some(secs(61)), idle), Some(secs(600))).await.is_empty());
        // already wedged when the stream starts: busy, nothing completed for 10 minutes
        let wedged = || SchedState { busy: true, work_age: Some(secs(600)) };
        assert!(read_frames_until(respond(never(), Some(secs(10)), wedged), Some(secs(600))).await.is_empty());
        // busy and nothing ever completed (a first request whose first unit of work hangs)
        let stuck_first = || SchedState { busy: true, work_age: None };
        assert!(read_frames_until(respond(never(), Some(secs(10)), stuck_first), Some(secs(600))).await.is_empty());
        // the largest accepted interval and an interval no clock can add: no frame, no panic
        // (axum's own KeepAlive computes `Instant::now() + interval` unchecked at every event)
        for every in [secs(3600), Duration::MAX] {
            let (tx, rx, bl) = tok_channel();
            feed(tx, vec![(5, tok(0)), (6, fin("stop"))]);
            let alive = || SchedState { busy: true, work_age: Some(secs(0)) };
            let frames = read_frames(respond(chat(rx, bl, PLAIN), Some(every), alive)).await;
            assert_eq!(timeline(&frames), [(5, "role"), (5, "data"), (6, "data"), (6, "data")]);
        }
    }

    /// The rule itself, as a table.
    #[test]
    fn the_keepalive_rule() {
        let st = |busy, age: Option<u64>| SchedState { busy, work_age: age.map(secs) };
        let stall = SSE_KEEPALIVE_STALL;
        // the scheduler completed work within the bound: alive, whatever else
        for busy in [false, true] {
            assert!(keepalive_live(st(busy, Some(0)), secs(100_000), stall));
            assert!(keepalive_live(st(busy, Some(60)), secs(100_000), stall), "60 s is inside the bound");
        }
        // idle for hours, a request has just arrived: the stream's own age covers the pick-up
        assert!(keepalive_live(st(false, Some(7200)), secs(10), stall));
        assert!(keepalive_live(st(false, None), secs(60), stall));
        assert!(!keepalive_live(st(false, Some(7200)), secs(61), stall), "nothing completed for this stream in 61 s");
        assert!(!keepalive_live(st(false, None), secs(61), stall));
        // busy and nothing completed for longer than the bound: wedged, for old and new streams
        assert!(!keepalive_live(st(true, Some(61)), secs(10), stall));
        assert!(!keepalive_live(st(true, Some(61)), secs(100_000), stall));
        assert!(!keepalive_live(st(true, None), secs(10), stall));
    }

    /// W4: the frames follow the scheduler's progress. Work completes every 3 s (prefill
    /// chunks, the decode rounds of other requests: a long legitimate wait) from 1 s to 118 s,
    /// then nothing (a hang) until 301 s, then again every 3 s until 331 s. Frames: every 10 s
    /// while the last completed work is at most 60 s old (the last one at 170 s; 180 s is 62 s
    /// after 118 s), silence during the hang, frames again from 310 s to 390 s.
    #[tokio::test(start_paused = true)]
    async fn keepalive_follows_the_scheduler_progress() {
        let last_work: Arc<std::sync::Mutex<Option<Instant>>> = Arc::new(std::sync::Mutex::new(None));
        let t0 = Instant::now();
        let w = Arc::clone(&last_work);
        tokio::spawn(async move {
            let ticks = (1..=118).step_by(3).chain((301..=331).step_by(3));
            for at in ticks {
                tokio::time::sleep_until(t0 + secs(at)).await;
                *w.lock().unwrap() = Some(Instant::now());
            }
        });
        let w = Arc::clone(&last_work);
        let sched = move || SchedState { busy: true, work_age: w.lock().unwrap().map(|t| t.elapsed()) };
        let never = tokio_stream::pending::<Result<Event, std::convert::Infallible>>();
        let frames = read_frames_until(respond(never, Some(secs(10)), sched), Some(secs(600))).await;
        let at: Vec<u64> = timeline(&frames).iter().map(|(t, k)| { assert_eq!(*k, "ka"); *t }).collect();
        let want: Vec<u64> = (10..=170).step_by(10).chain((310..=390).step_by(10)).collect();
        assert_eq!(at, want);
    }

    /// The real scheduler gauges (`metrics::sched_state`) through `sse_with_keepalive`, the
    /// function both endpoints call. The gauges are process globals that the metrics tests
    /// drive, so this test takes their lock and sets the state it asserts on.
    #[tokio::test(start_paused = true)]
    async fn sse_with_keepalive_reads_the_scheduler_gauges() {
        let _g = crate::metrics::tests::SCHED_GAUGE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::metrics::set_max_batch(2); // the value the metrics tests assert; initialises the gauge clock
        // idle scheduler, young stream: frames
        crate::metrics::sched_idle();
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(25, tok(0)), (26, fin("stop"))]);
        let frames = read_frames(sse_with_keepalive(chat(rx, bl, PLAIN), Some(secs(10)), SseKeepAliveFrame::Comment)).await;
        assert_eq!(timeline(&frames), [(10, "ka"), (20, "ka"), (25, "role"), (25, "data"), (26, "data"), (26, "data")]);
        // work completed just now (real clock: the whole test takes milliseconds): alive although busy
        crate::metrics::sched_step();
        crate::metrics::sched_touch();
        std::thread::sleep(Duration::from_millis(2)); // the gauge has millisecond resolution; 0 ms reads as "never"
        crate::metrics::sched_touch();
        assert!(keepalive_live(crate::metrics::sched_state(), secs(100_000), SSE_KEEPALIVE_STALL));
        crate::metrics::sched_idle();
    }

    /// LANERES1 `--sse-keepalive-frame data`: the pure payload is a chunk with an EMPTY choices
    /// array (the first real choices entry is still the request's first token — the keep-alive
    /// adds none), carrying the stream's own id/created/model; "data" mode builds that frame and
    /// the default mode stays the comment line.
    #[test]
    fn the_keepalive_data_frame_is_an_empty_choices_chunk() {
        let p = sse_keepalive_data_payload("chatcmpl-x", 1_700_000_000, "test-model", "chat.completion.chunk");
        let v: serde_json::Value = serde_json::from_str(&p).expect("valid JSON");
        assert_eq!(v["id"], "chatcmpl-x");
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["created"], 1_700_000_000);
        assert_eq!(v["model"], "test-model");
        assert_eq!(v["choices"], serde_json::json!([]), "no choices entry before the first real token");
        assert!(matches!(sse_keepalive_frame_of_mode("data", "id", 1, "m", "text_completion.chunk"), SseKeepAliveFrame::Data(_)));
        assert!(matches!(sse_keepalive_frame_of_mode("comment", "id", 1, "m", "chat.completion.chunk"), SseKeepAliveFrame::Comment));
        assert!(matches!(sse_keepalive_frame_of_mode("", "id", 1, "m", "chat.completion.chunk"), SseKeepAliveFrame::Comment));
    }

    /// `--inject-stall ROUND:SECONDS`, the W4 drill flag: parsed strictly (the store is not
    /// written here; unset in a unit-test process, so `sched_step` never sleeps).
    #[test]
    fn the_stall_drill_flag() {
        assert_eq!(parse_stall_drill(None), Ok(None));
        assert_eq!(parse_stall_drill(Some("40:100")), Ok(Some((40, 100))));
        assert_eq!(parse_stall_drill(Some(" 1 : 600 ")), Ok(Some((1, 600))));
        for bad in ["", "40", "40:", ":100", "0:100", "40:0", "40:601", "-1:5", "a:b", "40:100:1"] {
            let e = parse_stall_drill(Some(bad)).unwrap_err();
            assert!(e.starts_with("--inject-stall takes ROUND:SECONDS") && e.ends_with(&format!("(got '{bad}')")), "{e}");
        }
        let o = crate::opt!("inject-stall").def();
        assert_eq!((o.env, o.def, o.cat, o.scope), ("", "unset", crate::opts::Cat::Diag, crate::opts::Scope::Local));
        assert!(crate::opts::get(crate::opt!("inject-stall")).is_none(), "never set in the unit suite");
        // refused at the start of `main`, with the command-line errors (source pin: main.rs is the binary)
        let main_src = include_str!("main.rs");
        assert_eq!(main_src.matches(&format!("parse_cli(&args).and_then(|_| gb10_inference::server::{}(&args))", "check_serve_options")).count(), 1);
        assert_eq!(check_serve_options(&[]), Ok(()), "nothing is set in the unit suite: the defaults resolve");
        // R6(b): a one-box drill. Refused with any TP role on the command line; fine without one,
        // and no refusal at all when the drill is not set.
        let argv = |a: &[&str]| a.iter().map(|x| x.to_string()).collect::<Vec<String>>();
        for role in [&["--server", "--tp", "2"][..], &["--server", "--tp=4"], &["--tp"], &["--head", "--model-dir", "m"], &["--node", "--port", "29500"]] {
            let e = stall_drill_topology(true, &argv(role)).unwrap_err();
            assert!(e.starts_with("--inject-stall is a one-box drill and is refused with --"), "{e}");
            assert!(e.contains(role.iter().find(|a| a.starts_with("--tp") || **a == "--head" || **a == "--node").unwrap().split('=').next().unwrap()), "{e}");
            assert_eq!(stall_drill_topology(false, &argv(role)), Ok(()));
        }
        assert_eq!(stall_drill_topology(true, &argv(&["--server", "--model-dir", "m", "--tp-like-name", "--port", "9000"])), Ok(()));
        assert!(o.help.contains("REFUSED at startup together with --tp, --head or --node") && o.help.contains("nothing on the DSV4 server"), "{}", o.help);
    }

    /// W3: the boot line, and the values behind it (the option store is not written here: other
    /// tests latch `stream_backlog_events()` from it).
    #[test]
    fn the_keepalive_boot_line_and_its_default() {
        assert_eq!(sse_keepalive_boot_line(None), "[serve] sse keep-alive: off (--sse-keepalive 0)");
        let on = sse_keepalive_boot_line(Some(secs(10)));
        assert!(on.starts_with("[serve] sse keep-alive: every 10 s (--sse-keepalive;"), "{on}");
        assert!(on.contains("in the last 60 s"), "{on}");
        assert!(on.contains("the whole scheduler") && on.contains("for its first 60 s"), "{on}");
        // nothing is set in a unit-test process: the default is the registry row's (an owner
        // ruling, 2026-10-09: on, every 10 s), and it boots
        let id = crate::opt!("sse-keepalive");
        if crate::opts::get(id).is_none() {
            assert_eq!(resolve_sse_keepalive(), Ok(Some(secs(10))));
            let line = serve_options_boot_line().unwrap();
            assert!(line.starts_with(&on), "{line}");
            // S-B9-TOOLPARSE3: the parser rules and the counters are named on the next boot line
            assert!(line.contains("[serve] tool-call parser: values end by --tool-quote-pairing off, blocks end by --tool-block-end after-function, unclosed blocks by --tool-unclosed drop"), "{line}");
        }
        assert_eq!(crate::opts::ranged_from(id, Some("0")), Ok(0), "0 = off");
        assert_eq!(crate::opts::ranged_from(id, Some("10")), Ok(10));
        // the help text states the bound that is in the code, and which servers honour the flag
        let help = id.def().help;
        assert!(help.contains(&format!("for {} s they stop", SSE_KEEPALIVE_STALL.as_secs())), "{help}");
        assert!(help.contains("the DSV4 server sends no keep-alive frames"), "{help}");
        // ...and what the bound is NOT (review of BENCHFIX2): it is server-wide, not per request
        assert!(help.contains("the WHOLE scheduler") && help.contains("of any request"), "{help}");
        assert!(help.contains("keeps receiving frames") && help.contains("already hung"), "{help}");
    }

    /// Real wire bytes: the response is served by axum on a TCP socket and read back as bytes.
    /// Real time, so only ORDER is asserted (no upper bound on any duration): the test waits
    /// until two comment frames are on the wire, checks that nothing else is, and only then lets
    /// the first token out.
    #[tokio::test]
    async fn wire_bytes_over_a_real_socket() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let (tx, rx, bl) = tok_channel();
        let slot = Arc::new(std::sync::Mutex::new(Some((rx, bl))));
        let app = axum::Router::new().route("/s", get(move || {
            let slot = Arc::clone(&slot);
            async move {
                let (rx, bl) = slot.lock().unwrap().take().expect("one request");
                respond(chat(rx, bl, Shape { usage: false, thinking: false, tools: false }),
                        Some(Duration::from_millis(20)), idle)
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
        let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
        // HTTP/1.0: the body is the SSE bytes themselves (no chunked transfer framing)
        s.write_all(b"GET /s HTTP/1.0\r\nhost: t\r\n\r\n").await.unwrap();
        let mut wire = Vec::new();
        let mut buf = [0u8; 4096];
        let body_of = |wire: &[u8]| -> Option<String> {
            let text = String::from_utf8(wire.to_vec()).expect("utf-8");
            text.find("\r\n\r\n").map(|i| text[i + 4..].to_string())
        };
        // the wait: read until two comment frames have arrived
        loop {
            let n = s.read(&mut buf).await.unwrap();
            assert!(n > 0, "the stream ended during the wait");
            wire.extend_from_slice(&buf[..n]);
            if body_of(&wire).map_or(false, |b| b.matches(KA).count() >= 2) { break; }
        }
        let head = String::from_utf8(wire.clone()).unwrap();
        assert!(head.starts_with("HTTP/1.0 200"), "{head}");
        assert!(head.to_ascii_lowercase().contains("content-type: text/event-stream"), "{head}");
        assert!(!head.to_ascii_lowercase().contains("transfer-encoding"), "{head}");
        let waited = body_of(&wire).unwrap();
        assert!(waited.len() >= 2 * KA.len() && waited.replace(KA, "").is_empty(),
                "before the first token the body is comment frames only: {waited:?}");
        // now the first token, a second one, the end
        tx.send(tok(0)).unwrap();
        tx.send(tok(1)).unwrap();
        tx.send(fin("stop")).unwrap();
        s.read_to_end(&mut wire).await.unwrap();
        let body = body_of(&wire).unwrap();
        let tail = &body[waited.len()..];
        let data_part = tail.replace(KA, ""); // a frame may still land between two reads
        assert_eq!(data_part, format!("data: {}\n\ndata: {}\n\ndata: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                                      role_json(), content("Hello"), content(" world"), finish("stop")));
        assert_eq!(body.find("\"choices\""), body.find(&role_json()).map(|i| i + role_json().find("\"choices\"").unwrap()),
                   "the first `choices` on the wire is the role chunk's");
    }

    /// What the keep-alive rule ASSUMES of the two schedulers (source pins: the unit suite cannot
    /// build a scheduler). The rule reads one signal, `metrics::sched_touch`, and takes it to mean
    /// "a decode round or a prefill chunk completed". S-B9-REL-SURFACE2B R0: on the EXL3
    /// scheduler that signal used to be written at the end of ONE arm of the step (round-robin)
    /// and per prefill chunk; the shared plain step and the `--lane-order fcfs` arm returned
    /// before it, so a request waiting behind lanes that decode on those arms stopped receiving
    /// frames after 60 s and velogb10_scheduler_last_step_age_seconds read a false wedge.
    #[test]
    fn every_arm_of_both_scheduler_steps_reaches_the_touch() {
        // a function's text: from its signature to the next method of the impl
        fn body<'a>(src: &'a str, sig: &str) -> &'a str {
            let at = src.find(sig).unwrap_or_else(|| panic!("{sig} not found"));
            let end = src[at + sig.len()..].find("\n    fn ").map_or(src.len(), |e| at + sig.len() + e);
            &src[at..end]
        }
        // code only: a `//` comment may say "return"
        fn code(text: &str) -> String {
            text.lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("\n")
        }
        let (touch, round_start, idle) = (["crate::metrics::", "sched_touch()"].concat(), ["crate::metrics::", "sched_step()"].concat(),
                                          ["crate::metrics::", "sched_idle()"].concat());

        // --- EXL3 (src/exl3_serve.rs)
        let exl3 = include_str!("exl3_serve.rs");
        let step = body(exl3, "fn step(&mut self) -> Result<()> {");
        let rounds = body(exl3, "fn step_rounds(&mut self) -> Result<()> {");
        // `step` is the wrapper: it runs the rounds once and touches after ANY arm that ran one
        assert_eq!(step.matches("let r = self.step_rounds();").count(), 1);
        assert!(step.contains(&["if r.is_ok() && self.steps != before { ", touch.as_str(), "; }"].concat()), "{step}");
        assert_eq!(exl3.matches(&["self.", "step_rounds()"].concat()).count(), 1, "nothing but `step` runs the rounds");
        // the rounds: the round counter moves exactly once, before any arm, and the only return
        // ahead of it is the one that ran nothing (no lane holds a request)
        assert_eq!(code(rounds).matches("self.steps += 1;").count(), 1);
        assert_eq!(exl3.matches("self.steps += 1;").count(), 1, "the round counter moves in step_rounds only");
        let before_count = code(&rounds[..rounds.find("self.steps += 1;").unwrap()]);
        assert_eq!(before_count.matches("return").count(), 1, "one return before the round is counted");
        assert!(before_count.contains("if slots.is_empty() {\n            return Ok(());"), "...and it is the empty one");
        assert!(code(rounds).trim_start_matches(|c| c != '{').contains(&round_start), "a round starts with sched_step");
        assert!(!code(rounds).contains(&touch), "no arm writes the signal itself: the arms return early");
        // the arms (speculative lane, plain batch) are called from the rounds, and from the packed
        // arm (`--lane-pack`: `step_packed` and its `step_pair`, which run the lone round / the plain
        // batch for lanes outside a pair and for the control-cadence fallback), and from nowhere else.
        // The packed arm itself is entered from the rounds only, so it ends in `step`'s touch too.
        let packed = body(exl3, "fn step_packed(&mut self,");
        let pair = body(exl3, "fn step_pair(&mut self,");
        assert_eq!(exl3.matches(&["self.", "step_packed("].concat()).count(), 1, "one call of the packed arm");
        assert_eq!(code(rounds).matches(&["self.", "step_packed("].concat()).count(), 1, "...and it is in step_rounds");
        assert_eq!(exl3.matches(&["self.", "step_pair("].concat()).count(), 1, "one call of the packed round");
        assert_eq!(code(packed).matches(&["self.", "step_pair("].concat()).count(), 1, "...and it is in step_packed");
        for arm in [["self.", "step_mtp("].concat(), ["self.", "step_plain("].concat()] {
            assert!(code(rounds).matches(arm.as_str()).count() >= 3, "{arm}: rr, fcfs and shared-plain arms");
            let inside = [rounds, packed, pair].iter().map(|f| code(f).matches(arm.as_str()).count()).sum::<usize>();
            assert_eq!(exl3.matches(arm.as_str()).count(), inside, "{arm} outside step_rounds / step_packed / step_pair");
        }
        // a prefill chunk completes -> the signal; both scheduler loops mark idle before they block
        assert_eq!(body(exl3, "fn prefill_range(&mut self,").matches(&touch).count(), 1);
        assert_eq!(exl3.matches(&touch).count(), 2, "step and prefill_range, nowhere else");
        assert!(exl3.matches(&idle).count() >= 2, "run and run_tp_head");

        // --- NVFP4 (src/batch.rs): one arm, no early return
        let nv = include_str!("batch.rs");
        let dstep = body(nv, "fn decode_step(&mut self, b: usize) {");
        let dcode = code(dstep);
        let first = dcode.lines().skip(1).map(str::trim).find(|l| !l.is_empty()).unwrap();
        assert_eq!(first, format!("{round_start};"), "the round starts with sched_step");
        assert_eq!(dcode.matches(&touch).count(), 1);
        assert!(!dcode[..dcode.find(&touch).unwrap()].contains("return"), "decode_step returns nowhere before the signal");
        assert_eq!(nv.matches(&["self.", "decode_step("].concat()).count(), 4, "the scheduler loops (one box, TP head, TP node, inline) all go through it");
        // a prefill window completes -> the signal; the two returns ahead of it did no prefill
        // work (no cursor; a full prefix hit, which goes straight to pf_finish)
        let pf = code(body(nv, "fn pf_advance(&mut self) {"));
        assert_eq!(pf.matches(&touch).count(), 1);
        assert_eq!(pf[..pf.find(&touch).unwrap()].matches("return").count(), 2);
        assert_eq!((nv.matches(&touch).count(), nv.matches(&round_start).count()), (2, 1));
        assert!(nv.matches(&idle).count() >= 2);
    }

    /// The link from the handlers to the tested code (source pins; the generator and the
    /// response builder themselves are exercised above). The streaming branch of
    /// /v1/chat/completions returns exactly `sse_with_keepalive(chat_sse_stream(..))`, the
    /// /v1/completions stream goes through `sse_with_keepalive`, and nothing in this file builds
    /// an SSE response any other way.
    #[test]
    fn the_handlers_build_their_streams_from_the_tested_functions() {
        let src = include_str!("server.rs");
        let code = &src[..src.find("mod benchfix_sse_tests").unwrap()];
        let n = |needle: String| code.matches(&needle).count();
        assert_eq!(n(format!("let stream = {}(cx, move || {{", "chat_sse_stream")), 1, "the chat handler's stream");
        assert_eq!(n(format!("let mut resp = {}(stream, sse_keepalive(), ka_frame);", "sse_with_keepalive")), 1);
        assert_eq!(n(format!("return {}(stream, sse_keepalive(), ka_frame);", "sse_with_keepalive")), 1, "/v1/completions");
        // the only Sse responses are the two arms of sse_with_keepalive, and the only generator
        // that writes a role chunk is chat_sse_stream (once at the first token, once before the
        // terminal events)
        assert_eq!(n(format!("{}::new(", "Sse")), 2);
        assert_eq!(n(format!("role_pending.{}()", "take")), 2);
        assert_eq!(n(format!("{}!", "async_stream::stream")), 3, "keepalive_frames, chat_sse_stream, /v1/completions");
        // the OTel stream_start hook: one call site, the generator's first statement
        assert_eq!(n(format!("r.start(&{});", "role_chunk")), 1);
        let gen = &code[code.find("pub(crate) fn chat_sse_stream").unwrap()..];
        let body = &gen[gen.find("async_stream::stream! {").unwrap() + "async_stream::stream! {".len()..];
        let first = body.lines().map(str::trim).find(|l| !l.is_empty() && !l.starts_with("//")).unwrap();
        assert_eq!(first, "if let Some(r) = &otel_req { r.start(&role_chunk); }");
    }

    // ===================================================================================
    // S-B9-TOOLPARSE3 T8 (b)(c)(d): the serial fixtures, the fuzz class D reproduction and
    // two stated behaviours, all on the REAL generator (`chat_sse_stream`).
    // ===================================================================================

    /// What a client assembles from one chat response, streamed or not.
    #[derive(Debug, Default, PartialEq)]
    struct Wire {
        /// every `reasoning_content` delta, concatenated
        reasoning: String,
        /// every `content` delta, concatenated
        content: String,
        /// (name, arguments as JSON) of every tool call, in order
        calls: Vec<(String, serde_json::Value)>,
        /// the finish chunk's `finish_reason`
        finish: String,
        /// the delta kinds in wire order: 'r' reasoning, 'c' content, 't' tool_calls
        order: String,
    }

    fn wire_of(data: &[String]) -> Wire {
        let mut w = Wire::default();
        for f in data {
            if f == "[DONE]" { continue; }
            let j: serde_json::Value = serde_json::from_str(f).expect("a data frame is JSON");
            let Some(ch) = j["choices"].as_array().and_then(|a| a.first()) else { continue };
            let d = &ch["delta"];
            if let Some(s) = d["reasoning_content"].as_str() { w.reasoning.push_str(s); w.order.push('r'); }
            if let Some(s) = d["content"].as_str() { w.content.push_str(s); w.order.push('c'); }
            if let Some(a) = d["tool_calls"].as_array() {
                w.order.push('t');
                for c in a {
                    let args = c["function"]["arguments"].as_str().expect("arguments is a string");
                    w.calls.push((c["function"]["name"].as_str().unwrap().to_string(),
                                  serde_json::from_str(args).expect("arguments is JSON")));
                }
            }
            if let Some(r) = ch["finish_reason"].as_str() { w.finish = r.to_string(); }
        }
        w
    }

    /// The request's `tools` array as `cx` builds it.
    fn t8_tools() -> Vec<serde_json::Value> {
        vec![serde_json::json!({"type": "function", "function": {"name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}})]
    }

    /// The REAL generator over a pre-queued event channel: one token per piece (the decoder
    /// returns the piece), then `Finish`. `thinking` = the prompt opened the think block.
    async fn t8_stream(thinking: bool, tools: bool, pieces: &[String]) -> Wire {
        let (tx, rx, bl) = tok_channel();
        let _ = tx.send(TokEvent::Admitted { cached_tokens: 0 });
        for i in 0..pieces.len() { let _ = tx.send(tok(i as u32)); }
        let _ = tx.send(fin("stop"));
        let table = pieces.to_vec();
        let sh = Shape { thinking, tools, usage: false };
        let s = chat_sse_stream(cx(rx, bl, sh, None), move || move |t: u32| table[t as usize].clone());
        wire_of(&data_frames(&read_frames(respond(s, None, idle)).await))
    }

    /// The non-streaming handler's decision for a finished generation, call for call (the
    /// non-streaming branch of `chat_completions`: `split_think`, the SURFACE2 unclosed-think
    /// split, `tools::parse`, `finalize_parsed`). The handler itself needs an AppState (a
    /// tokenizer and a scheduler), so this is the one hand-written half of the comparison.
    fn t8_non_stream(thinking: bool, tools: bool, text: &str) -> Wire {
        let (open, close) = ("<think>", "</think>");
        let (reasoning, content) = split_think_in(text, open, close, thinking);
        let closed = text.contains(close);
        let (reasoning, content) = if thinking && !closed && !unclosed_think_as_content() {
            unclosed_think_split(text, tools)
        } else {
            (reasoning, content)
        };
        let t = t8_tools();
        let parsed = if tools {
            // F8: the real non-streaming branch calls parse_served with the engine finish (a
            // wrapper the model never closed is completed on `stop`): the reference must too
            crate::tools::parse_served(&content, Some(&t), "stop")
        } else {
            crate::tools::ParsedOutput { content: content.trim().to_string(), tool_calls: Vec::new() }
        };
        let (content, calls, finish) = crate::tools::finalize_parsed(&content, parsed, "stop");
        Wire {
            reasoning: reasoning.unwrap_or_default(),
            content: content.unwrap_or_default(),
            calls: calls.iter().map(|c| (c.function.name.clone(),
                                         serde_json::from_str(&c.function.arguments).unwrap())).collect(),
            finish: spec_finish_reason(&finish).to_string(),
            order: String::new(),
        }
    }

    /// `text` as pieces of the given sizes in CHARACTERS (cycled; the last piece takes the rest).
    fn t8_cut(text: &str, sizes: &[usize]) -> Vec<String> {
        let mut out = Vec::new();
        let mut it = text.chars().peekable();
        let mut k = 0;
        while it.peek().is_some() {
            let n = sizes[k % sizes.len()].max(1);
            out.push(it.by_ref().take(n).collect::<String>());
            k += 1;
        }
        out
    }

    /// Every way to cut `text` at ONE character boundary, plus: one piece, one character per
    /// piece, and eight pseudo-random cuts into pieces of 1 to 9 characters (a fixed LCG: the
    /// same cuts on every run). A marker that spans a cut is held at that cut.
    fn t8_chunkings(text: &str) -> Vec<Vec<String>> {
        let mut all: Vec<Vec<String>> = vec![vec![text.to_string()], t8_cut(text, &[1])];
        for (p, _) in text.char_indices().skip(1) {
            all.push(vec![text[..p].to_string(), text[p..].to_string()]);
        }
        let mut s: u64 = 0x2545_f491_4f6c_dd1d;
        for _ in 0..8 {
            let sizes: Vec<usize> = (0..16).map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                1 + ((s >> 33) % 9) as usize
            }).collect();
            all.push(t8_cut(text, &sizes));
        }
        all
    }

    /// The `tests/toolcall_serial_test.rs` fixtures (the same strings: that file asserts the
    /// finish-time DECISION on them, this test asserts what the stream SENDS). Each entry:
    /// (name, generated text, the calls the response must carry, the content it must carry).
    fn t8_fixtures() -> Vec<(&'static str, &'static str, Vec<(&'static str, serde_json::Value)>, &'static str)> {
        let w = |c: &str| serde_json::json!({"city": c});
        vec![
            ("malformed captured",
             "<tool_call>\nfunction=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=units>\nc\n</parameter>\n</function>\n</tool_call>",
             vec![("get_weather", serde_json::json!({"city": "Paris", "units": "c"}))], ""),
            ("well-formed",
             "<tool_call>\n<function=get_weather>\n<parameter=city>\nBerlin\n</parameter>\n</function>\n</tool_call>",
             vec![("get_weather", w("Berlin"))], ""),
            ("f8: wrapper never closed, element closed",
             "Writing it down.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>",
             vec![("get_weather", w("Paris"))], "Writing it down."),
            ("f8: two closed elements then a cut one",
             "Both.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n<function=get_weather>\n<parameter=city>\nOs",
             vec![("get_weather", w("Paris")), ("get_weather", w("Tokyo"))], "Both."),
            ("two calls, prose before",
             "I'll check both cities.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>",
             vec![("get_weather", w("Paris")), ("get_weather", w("Tokyo"))], "I'll check both cities."),
            ("no arguments",
             "<tool_call>\n<function=list_cities>\n</function>\n</tool_call>",
             vec![("list_cities", serde_json::json!({}))], ""),
            ("unicode",
             "<tool_call>\n<function=get_weather>\n<parameter=city>\nZürich 🇨🇭 北çå\n</parameter>\n<parameter=note>\ntempérature — très élevée\n</parameter>\n</function>\n</tool_call>",
             vec![("get_weather", serde_json::json!({"city": "Zürich 🇨🇭 北çå", "note": "température — très élevée"}))], ""),
            ("hy_v3 opensource",
             "Checking.<tool_calls:opensource>\n<tool_call:opensource>get_weather<tool_sep:opensource>\n<arg_key:opensource>city</arg_key:opensource>\n<arg_value:opensource>\nParis\n</arg_value:opensource>\n</tool_call:opensource>\n</tool_calls:opensource>",
             vec![("get_weather", w("Paris"))], "Checking."),
            ("truncated block",
             "Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis",
             vec![], "Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis"),
            ("plain prose",
             "The sky is blue because Rayleigh scattering.",
             vec![], "The sky is blue because Rayleigh scattering."),
            // a `<` the model never completed, and a marker-looking word that is not a call
            ("ends in a marker prefix", "Done <", vec![], "Done <"),
            ("marker-looking text", "see <tool_call_x y> and <thinking>", vec![], "see <tool_call_x y> and <thinking>"),
        ]
    }

    /// (b) The serial fixtures through the REAL stream, in every chunking `t8_chunkings` makes,
    /// content mode and think mode (`plan</think>\n\n` first, the close tag cut anywhere too):
    /// the client's reasoning, content, calls and finish_reason equal the non-streaming
    /// response's AND the fixture's stated outcome. The success signal is asserted per
    /// fixture: the calls it must carry are present, the content it must carry is non-empty
    /// where it should be, and a parsed call's text is never also sent as content.
    #[tokio::test(start_paused = true)]
    async fn the_serial_fixtures_stream_what_the_non_streaming_response_returns() {
        let _complete = crate::tools::test_unclosed(crate::tools::Unclosed::Complete); // F11: the default is drop; the F8 fixtures are about `complete`
        let mut streams = 0usize;
        for (name, raw, calls, content) in t8_fixtures() {
            for thinking in [false, true] {
                let text = if thinking { format!("plan</think>\n\n{raw}") } else { raw.to_string() };
                let want_reasoning = if thinking { "plan" } else { "" };
                let reference = t8_non_stream(thinking, true, &text);
                // the reference itself says what the fixture must produce (not only "agrees")
                let names: Vec<&str> = reference.calls.iter().map(|c| c.0.as_str()).collect();
                assert_eq!(names, calls.iter().map(|c| c.0).collect::<Vec<_>>(), "{name}: reference calls");
                for ((_, got), (_, want)) in reference.calls.iter().zip(&calls) {
                    assert_eq!(got, want, "{name}: reference arguments");
                }
                assert_eq!(reference.finish, if calls.is_empty() { "stop" } else { "tool_calls" }, "{name}");
                assert_eq!(reference.content.trim(), content, "{name}: reference content");
                assert_eq!(reference.reasoning.trim(), want_reasoning, "{name}: reference reasoning");
                for pieces in t8_chunkings(&text) {
                    streams += 1;
                    let got = t8_stream(thinking, true, &pieces).await;
                    let at = format!("{name} / thinking {thinking} / {pieces:?}");
                    assert_eq!(got.calls, reference.calls, "calls: {at}");
                    assert_eq!(got.finish, reference.finish, "finish: {at}");
                    assert_eq!(got.reasoning.trim(), reference.reasoning.trim(), "reasoning: {at}");
                    assert_eq!(got.content.trim(), reference.content.trim(), "content: {at}");
                    // the success signals, stated again on the stream's own bytes
                    assert_eq!(got.content.trim(), content, "stream content: {at}");
                    if !calls.is_empty() {
                        assert!(!got.calls.is_empty() && got.order.ends_with('t'), "the call is the last delta: {at}");
                        assert!(!got.content.contains("<tool_call") && !got.reasoning.contains("<tool_call"),
                                "a parsed call is not also sent as text: {at}");
                    } else {
                        assert!(got.calls.is_empty() && !got.order.contains('t'), "no call: {at}");
                        assert!(!got.content.trim().is_empty(), "the text is not dropped: {at}");
                    }
                }
            }
        }
        assert!(streams > 2000, "the fixtures ran through {streams} streams");
    }

    /// F8: a response cut by the token limit inside a wrapper it never closed returns NO call in
    /// the stream (the finish chunk carries `length`, never `tool_calls`) and none in the
    /// non-streaming response — under every chunking. The same text stopped by itself DOES return
    /// the closed element (the fixtures above, every chunking).
    #[tokio::test(start_paused = true)]
    async fn f8_unclosed_cut_by_length_returns_no_call_in_stream_or_json() {
        let _complete = crate::tools::test_unclosed(crate::tools::Unclosed::Complete); // F11: the default is drop
        let text = "Prose.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>";
        for pieces in t8_chunkings(text) {
            let (tx, rx, bl) = tok_channel();
            let _ = tx.send(TokEvent::Admitted { cached_tokens: 0 });
            for i in 0..pieces.len() { let _ = tx.send(tok(i as u32)); }
            let _ = tx.send(fin("length"));
            let table = pieces.to_vec();
            let sh = Shape { thinking: false, tools: true, usage: false };
            let s = chat_sse_stream(cx(rx, bl, sh, None), move || move |t: u32| table[t as usize].clone());
            let got = wire_of(&data_frames(&read_frames(respond(s, None, idle)).await));
            assert!(got.calls.is_empty(), "cut by length: no call: {pieces:?}");
            assert_eq!(got.finish, "length", "cut by length: finish carried as is: {pieces:?}");
        }
        assert!(crate::tools::parse_served(text, None, "length").tool_calls.is_empty());
        assert_eq!(crate::tools::parse_served(text, None, "stop").tool_calls.len(), 1);
    }

    /// (c) FUZZ CLASS D (the independent review of SURFACE2B, B12: 4 of 300,000 fuzzed streams;
    /// pre-existing on master). REPRODUCTION, PINNED AS IT IS TODAY.
    ///
    /// Input (thinking off, tools offered, the pieces below, each one decode chunk):
    ///   `Hello` | `<tool_call>` | `\n<function=get_weather>…Paris…</function>\n` | `</tool_call>`
    ///   | `<think>` | `hmm`
    /// What the client receives: content `Hello`; then content with the RAW call text
    /// (`<tool_call>…</tool_call>`, sent when the `<think>` arrives); then reasoning `hmm`;
    /// then the `tool_calls` delta carrying the same call; finish `tool_calls`. The call is
    /// delivered twice: as text and as a structured call.
    ///
    /// Verdict: a defect of the stream's hold-back in `chat_sse_stream`, not of `tools::parse`
    /// or `split_think`. In content mode the `<think>` hand-off (`f_open.find(&acc, cs,
    /// think_open)` -> `Some(upto)`) emits `acc[content_emitted..upto]` WITHOUT clamping at a
    /// held tool marker, where every other emit site goes through `hold_back_end` (which
    /// stops at the marker). `content_emitted` is parked AT the marker by the hold-back, so
    /// the whole held call text is inside `acc[content_emitted..upto]` and leaves as content.
    /// The parser then reads the whole `acc` at the finish (no close tag: `split_think` returns
    /// it as content) and finds the call, so the structured call follows. The non-streaming
    /// response for the same text is content `Hello<think>hmm` + the call (`<think>` is
    /// returned as text there: no close tag); it never sends the call as text.
    ///
    /// FIXED (S-B9-TOOLPARSE3 part B item 5, independent review M2): a `<think>` that lies after a
    /// held tool marker is text of the held call, so the hand-off to reasoning takes only a hit
    /// BEFORE the marker. What the client receives now: content `Hello`, no raw call text, the
    /// structured call last. (`hmm`, which follows the stray `<think>` after the call, is held text and
    /// is dropped because a call parsed: as in the non-streaming response.)
    #[tokio::test(start_paused = true)]
    async fn fuzz_class_d_a_think_open_after_a_held_tool_marker_no_longer_sends_the_call_as_content_too() {
        let body = "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n";
        let raw_call = format!("<tool_call>{body}</tool_call>");
        let pieces: Vec<String> = ["Hello", "<tool_call>", body, "</tool_call>", "<think>", "hmm"]
            .iter().map(|s| s.to_string()).collect();
        let text = pieces.concat();
        let w = t8_stream(false, true, &pieces).await;

        assert_eq!(w.calls, vec![("get_weather".to_string(), serde_json::json!({"city": "Paris"}))]);
        assert_eq!(w.finish, "tool_calls");
        let n = t8_non_stream(false, true, &text);
        assert_eq!(n.calls, w.calls, "non-streaming: the same single call");
        assert_eq!(n.content, "Hello<think>hmm", "non-streaming: the prose, `<think>` left in (no close tag)");
        assert!(!n.content.contains("<tool_call"), "non-streaming never sends the call as text");

        // the raw call text does NOT leave as content any more
        assert!(!w.content.contains(&raw_call) && !w.content.contains("<tool_call"), "the raw call is not content: {:?}", w.content);
        assert_eq!(w.content, "Hello");
        assert_eq!(w.reasoning, "");
        assert_eq!(w.order, "ct", "content `Hello`, then the tool_calls delta");

        // inside ONE decode chunk (the whole reply is one piece)
        let one = t8_stream(false, true, &[text.clone()]).await;
        assert_eq!(one.calls, w.calls);
        assert_eq!((one.content.as_str(), one.reasoning.as_str(), one.order.as_str()), ("Hello", "", "ct"));

        // and after a think block that closed: thinking on, `plan</think>\n\nHello<call><think>hmm`
        let think_pieces: Vec<String> = std::iter::once("plan</think>\n\n".to_string()).chain(pieces.iter().cloned()).collect();
        let t = t8_stream(true, true, &think_pieces).await;
        assert_eq!(t.calls, w.calls);
        assert_eq!((t.reasoning.as_str(), t.content.as_str(), t.order.as_str()), ("plan", "Hello", "rct"));

        // CONTROL: the same reply WITHOUT the `<think>` is held to the end and sent as the call only
        let ctl: Vec<String> = pieces[..4].to_vec();
        let c = t8_stream(false, true, &ctl).await;
        assert_eq!((c.content.as_str(), c.order.as_str()), ("Hello", "ct"), "no `<think>`: no duplicate");
        assert_eq!(c.calls, w.calls);
    }

    /// Item 5, the natural trigger: an ARGUMENT that contains the text `<think>` (a chat template, a
    /// document about reasoning tags, this repository's own server.rs), in EVERY chunking, thinking off
    /// and on, streamed and not: one call whose argument is whole, the prose before it, nothing of the
    /// call's markup in content or reasoning. And an argument that contains `</think>`, with thinking
    /// OFF: the call used to be thrown away (the first close tag was taken for the end of reasoning).
    #[tokio::test(start_paused = true)]
    async fn a_think_tag_inside_an_argument_is_text_of_the_call_in_every_chunking() {
        for (arg, label) in [
            ("The model starts its reasoning with <think> and then answers.", "open tag"),
            ("A close tag </think> inside an argument.", "close tag"),
            ("<think>draft</think> both, as a template shows", "both"),
        ] {
            let call = format!("<tool_call>\n<function=get_weather>\n<parameter=city>\n{arg}\n</parameter>\n</function>\n</tool_call>");
            let want = vec![("get_weather".to_string(), serde_json::json!({"city": arg}))];
            // thinking OFF
            let text = format!("I will write it.\n{call}");
            let n = t8_non_stream(false, true, &text);
            assert_eq!(n.calls, want, "{label}: non-streaming, thinking off");
            assert_eq!(n.content, "I will write it.", "{label}");
            for pieces in t8_chunkings(&text) {
                let w = t8_stream(false, true, &pieces).await;
                assert_eq!(w.calls, want, "{label}: streamed {pieces:?}");
                // (the stream sends the prose before a held call as it arrives, newline included;
                // clients trim it, the non-streaming response does)
                assert_eq!(w.content.trim(), "I will write it.", "{label}: {pieces:?}");
                assert_eq!(w.reasoning, "", "{label}: {pieces:?}");
                assert_eq!(w.finish, "tool_calls");
                assert!(w.order.ends_with('t') && !w.order.contains('r'), "{label}: {pieces:?} order {}", w.order);
            }
            // thinking ON, the block closed first
            let text = format!("plan</think>\n\nI will write it.\n{call}");
            let n = t8_non_stream(true, true, &text);
            assert_eq!(n.calls, want, "{label}: non-streaming, thinking on");
            assert_eq!((n.reasoning.as_str(), n.content.as_str()), ("plan", "I will write it."), "{label}");
            for pieces in t8_chunkings(&text) {
                let w = t8_stream(true, true, &pieces).await;
                assert_eq!(w.calls, want, "{label}: streamed thinking on {pieces:?}");
                assert_eq!((w.reasoning.as_str(), w.content.trim()), ("plan", "I will write it."), "{label}: {pieces:?}");
                assert_eq!(w.finish, "tool_calls");
            }
        }
    }

    /// (d) 1 of 2, STATED AS IT IS TODAY. A tool call drafted INSIDE a think block that is never
    /// closed (the prompt opened it; the model writes its thoughts and jumps to the call without
    /// `</think>`) is RETURNED AS A CALL, in both response modes.
    ///  * non-streaming: `unclosed_think_split(&text, parse_as_tool_calls)` (the narrow guard
    ///    `prompt_opened_think && !think_closed && !unclosed_think_as_content()`): the text up to
    ///    the first `<tool_call` marker is `reasoning_content`, the text from the marker on is
    ///    what `tools::parse` reads: the call becomes `tool_calls`, finish `tool_calls`.
    ///    (`--unclosed-think content` restores the v0.7.3 arm: `split_think` returns the whole
    ///    text as content, the parser reads all of it, the SAME call comes out and the thoughts
    ///    are `content`.)
    ///  * streaming: the reasoning branch holds back from the marker (`f_tool_r`), the thoughts
    ///    leave as `reasoning_content`; at the finish `split_think(&acc)` has no close tag, so
    ///    `tools::parse` reads the whole buffer: the same call, finish `tool_calls`.
    /// Minimal input: the prompt opened the block; the model wrote
    /// `I need the weather.\n<tool_call>\n<function=get_weather>…Athens…</function>\n</tool_call>`
    /// and nothing else.
    #[tokio::test(start_paused = true)]
    async fn a_call_drafted_in_an_unclosed_think_block_is_returned_as_a_call_is_pinned() {
        let text = "I need the weather.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nAthens\n</parameter>\n</function>\n</tool_call>";
        let want = vec![("get_weather".to_string(), serde_json::json!({"city": "Athens"}))];
        let n = t8_non_stream(true, true, text);
        assert_eq!(n.calls, want);
        assert_eq!(n.finish, "tool_calls");
        assert_eq!(n.reasoning, "I need the weather.", "the thoughts are reasoning_content");
        assert_eq!(n.content, "", "and content is empty");
        for pieces in t8_chunkings(text) {
            let s = t8_stream(true, true, &pieces).await;
            assert_eq!(s.calls, want, "{pieces:?}");
            assert_eq!(s.finish, "tool_calls", "{pieces:?}");
            assert_eq!(s.reasoning.trim(), "I need the weather.", "{pieces:?}");
            assert_eq!(s.content, "", "{pieces:?}");
        }
        // the v0.7.3 arm (`--unclosed-think content`): the same call, the thoughts are content
        let (r, c) = split_think(text, "<think>", "</think>");
        assert_eq!((r, c.as_str()), (None, text));
        let p = crate::tools::parse(&c, Some(&t8_tools()));
        assert_eq!((p.tool_calls.len(), p.content.as_str()), (1, "I need the weather."));
        // tool parsing off for the request: nothing is a call; the marker is ordinary reasoning text
        let (r, c) = unclosed_think_split(text, false);
        assert_eq!((r.as_deref(), c.as_str()), (Some(text), ""));
    }

    /// (d) 2 of 2, STATED AS IT IS TODAY. A stream DROPS the prose that follows a parsed call;
    /// the non-streaming response returns it. `tools::parse` (Qwen markup) keeps the prose
    /// before, between and after the calls as `content`; the stream stops sending content at the
    /// first tool marker (the hold-back), and when a call parsed the finish sends the
    /// `tool_calls` delta ONLY (`held_tail` runs only when nothing parsed), so everything after
    /// the marker that is not a call is never sent. The hy_v3 markup is the same in both modes
    /// (`parse_hy3` returns only the prose BEFORE the wrapper).
    /// Minimal input: `<tool_call>…Paris…</tool_call>\nI will report back.`
    /// non-streaming: content `I will report back.`, one call, finish `tool_calls`;
    /// streaming: no content delta at all, one call, finish `tool_calls`.
    #[tokio::test(start_paused = true)]
    async fn a_stream_drops_the_prose_that_follows_a_parsed_call_is_pinned() {
        let call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        let want = vec![("get_weather".to_string(), serde_json::json!({"city": "Paris"}))];
        // prose only after the call
        let text = format!("{call}\nI will report back.");
        let n = t8_non_stream(false, true, &text);
        assert_eq!((n.calls.clone(), n.content.as_str(), n.finish.as_str()), (want.clone(), "I will report back.", "tool_calls"));
        for pieces in t8_chunkings(&text) {
            let s = t8_stream(false, true, &pieces).await;
            assert_eq!(s.calls, want, "{pieces:?}");
            assert_eq!(s.finish, "tool_calls");
            // PINNED: the prose after the call never leaves
            assert_eq!(s.content, "", "dropped: {pieces:?}");
            assert_eq!(s.order, "t", "{pieces:?}");
        }
        // prose before AND after: the stream keeps only the part before
        let text = format!("Checking.\n{call}\nDone.");
        let n = t8_non_stream(false, true, &text);
        assert_eq!(n.content, "Checking.\n\nDone.", "non-streaming: before and after, joined");
        for pieces in t8_chunkings(&text) {
            let s = t8_stream(false, true, &pieces).await;
            assert_eq!(s.calls, want, "{pieces:?}");
            assert_eq!(s.content.trim(), "Checking.", "streaming: only the prose before the call: {pieces:?}");
        }
        // hy_v3 markup: both modes drop the prose after the wrapper, so they agree
        let hy = "Checking.<tool_calls:opensource>\n<tool_call:opensource>get_weather<tool_sep:opensource>\n<arg_key:opensource>city</arg_key:opensource>\n<arg_value:opensource>\nParis\n</arg_value:opensource>\n</tool_call:opensource>\n</tool_calls:opensource>\nAfter.";
        let n = t8_non_stream(false, true, hy);
        assert_eq!((n.calls.len(), n.content.as_str()), (1, "Checking."));
        let s = t8_stream(false, true, &t8_cut(hy, &[3])).await;
        assert_eq!((s.calls.len(), s.content.as_str()), (1, "Checking."));
    }
}
