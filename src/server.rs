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
/// frontmatter `base_model:` line (every model dir ships one, e.g. `base_model: Qwen/Qwen3.8-27B`),
/// falling back to the directory name when the card or the line is absent. Before this, the
/// server reported the lab directory fragment (`"model": "3.8-27b-nvfp4-full-all"`) — an
/// internal path name that no client or catalog can resolve. `--model-name` still overrides.
pub fn model_id_from_dir(model_path: &str) -> String {
    let dir = std::path::Path::new(model_path.trim_end_matches('/'));
    if let Ok(card) = std::fs::read_to_string(dir.join("README.md")) {
        for line in card.lines() {
            let l = line.trim();
            if let Some(v) = l.strip_prefix("base_model:") {
                let v = v.trim().trim_matches('"').trim_matches('\'').trim();
                if !v.is_empty() { return v.to_string(); }
            }
        }
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

fn partial_overlap(s: &str, marker: &str) -> usize {
    (1..marker.len()).rev().find(|&k| s.ends_with(&marker[..k])).unwrap_or(0)
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

/// LR-4 (REL_LONGRUN): `--max-waiting N` (default 256, 0 = unlimited). The HTTP handlers push
/// every accepted request into an UNBOUNDED scheduler channel; once N or more requests wait
/// beyond the lanes, refuse instead of queueing. H8 (REL_V0_7_3): the counter is the Req
/// guard's IN-FLIGHT count — a request counts for its WHOLE handler lifetime (a stalled
/// reader's request stays counted, not just its queue wait), and a burst can overshoot by the
/// requests admitted between the check and the send. Each queued request pins its prompt and
/// messages until a lane frees; the only other guard is the memwatch kill-switch.
pub fn max_waiting() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::var(crate::opt!("max-waiting")).ok()
        .and_then(|v| v.parse().ok()).unwrap_or(256))
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

/// LR-5 (REL_LONGRUN): `--stream-backlog-events N` (default 65536, 0 = unlimited). A stream
/// client that stops reading makes its per-request channel grow for the whole generation; past
/// this many UNCONSUMED events the scheduler cancels the request exactly like a client
/// disconnect (the lane sweep / wire Cancel), freeing the GPU lane.
pub fn stream_backlog_events() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::var(crate::opt!("stream-backlog-events")).ok()
        .and_then(|v| v.parse().ok()).unwrap_or(65_536))
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

/// Split a completed generation into (reasoning, answer). If the close marker is present, everything
/// before it is reasoning (a leading think-open is stripped) and everything after (trimmed) is the
/// answer. If the marker never appears, the whole text is returned as the answer content.
fn split_think(s: &str, think_open: &str, think_close: &str) -> (Option<String>, String) {
    match s.find(think_close) {
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
    #[serde(default)]
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

fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.8 }
fn default_top_k() -> usize { 20 }

fn deserialize_stop<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    use serde::Deserialize;
    let v: serde_json::Value = serde_json::Value::deserialize(d)?;
    Ok(match v {
        serde_json::Value::Null => vec![],
        serde_json::Value::String(s) => vec![s],
        serde_json::Value::Array(a) => a.into_iter().filter_map(|x| x.as_str().map(String::from)).collect(),
        _ => vec![],
    })
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

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<ChatCompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(mut req) = match payload { Ok(j) => j, Err(e) => return bad_json(e) };
    // Log the request parameters the client sent (useful for debugging OpenWebUI behavior)
    crate::reprintln!(
        "[req] params  temp={:?} top_p={:?} top_k={:?} max_tok={:?} max_completion_tok={:?} rep_pen={:?} presence={:?} freq={:?} stream={} effort={:?} ctk={}",
        req.temperature, req.top_p, req.top_k,
        req.max_tokens, req.max_completion_tokens, req.repetition_penalty, req.presence_penalty, req.frequency_penalty,
        req.stream, req.reasoning_effort,
        req.chat_template_kwargs.as_ref().map(|v| v.to_string()).unwrap_or_else(|| "-".into())
    );
    if !req.unused_fields.is_empty() {
        let names: Vec<&str> = req.unused_fields.keys().map(|k| k.as_str()).collect();
        crate::reprintln!("[req] fields not used by this server: {names:?}");
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
    let mut prompt_tokens = match state.tokenizer.encode(&prompt, true) {
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
        let prep = tokio::task::spawn_blocking(move || {
            if let Some(g) = gpu.filter(|_| !force_cpu) {
                let mut gvt = g.lock().expect("vision_gpu lock");
                crate::vision_encoder::prepare_vision_request_gpu(&mut gvt, &urls_w, &toks)
            } else if let Some(tower) = cpu_tower {
                crate::vision_encoder::prepare_vision_request(&tower, &urls_w, &toks)
            } else {
                Err(anyhow::anyhow!("no vision tower loaded"))
            }
        }).await.unwrap_or_else(|e| Err(anyhow::anyhow!("vision worker failed: {e}")));
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
    crate::reprintln!("[req] sampling temp={temperature} top_p={top_p} top_k={top_k} (thinking={thinking_on}; \
               sent: temp={:?} top_p={:?} top_k={:?})", req.temperature, req.top_p, req.top_k);

    // Submit to the batching scheduler and receive tokens on a channel.
    // Use request's penalties if explicitly set, else fall back to server defaults.
    let rep_penalty = req.repetition_penalty.unwrap_or(state.default_rep_penalty);
    let presence_penalty = req.presence_penalty.unwrap_or(state.default_presence_penalty);
    let pp_source = if req.presence_penalty.is_some() { "request" } else { "server-default" };
    let frequency_penalty = req.frequency_penalty.unwrap_or(state.default_frequency_penalty);

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

    let tool_calls_chunk = |cid: &str, created: i64, model: &str, calls: &[ToolCall]| {
        let arr: Vec<serde_json::Value> = calls.iter().enumerate().map(|(i, c)| serde_json::json!({
            "index": i, "id": c.id, "type": c.kind,
            "function": {"name": c.function.name, "arguments": c.function.arguments},
        })).collect();
        serde_json::json!({
            "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"tool_calls": arr}, "finish_reason": null}],
        }).to_string()
    };

    if req.stream {
        crate::reprintln!("[req] stream  prompt_tokens={} max_tokens={} stop={:?}", prompt_len, req_max, req.stop);
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
        // the tail check can't see it.
        let (think_open, think_close, _) = tokenizer.think_tags();
        // trim_end: the primed form is `<think>\n` — the trailing newline must not defeat the
        // tail check (a plain ends_with(think_open) sent thinking streams to content: exactly
        // the 35b4b15 follow-up bug). The no-think tail `<think>\n\n</think>\n\n` trims to
        // `</think>` and stays content.
        let starts_in_reasoning = prompt.trim_end().ends_with(think_open)
            || matches!(effort, Some("low") | Some("high"));

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

        let stream = async_stream::stream! {
            let role_chunk = role_chunk(&completion_id, created, &model_name); // S8: esc(model)
            if let Some(r) = &otel_req { r.start(&role_chunk); }   // event=stream_start, token.index 0
            yield Ok::<Event, axum::Error>(Event::default().data(role_chunk));
            // Byte-level stream decoder: the per-token decode path above would mangle every
            // multi-byte char split across tokens (all emoji) into "�" — the crate's ByteLevel
            // decode is String::from_utf8_lossy per call. Reassembles raw bytes across tokens.
            let mut stream_dec = tokenizer.stream_decoder();
            let mut acc = String::new();
            let mut n = 0usize;
            let mut cached = 0usize; // v0.7.3 (#8.2): prefix-cache-served prompt tokens
            let mut last_tok: Option<u32> = None;
            let mut stop_hit = false;
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
            while let Some(ev) = recv_counted(&mut rx, &bl).await { // LR-5: the one decrement (H0)
                match ev {
                    TokEvent::Admitted { cached_tokens } => {
                        cached = cached_tokens as usize;
                        mt.cached(cached_tokens as u64); // /metrics counter
                    }
                    TokEvent::Tok(t) => {
                        n += 1;
                        mt.tok();
                        last_tok = Some(t);
                        if first_tok.is_none() { first_tok = Some(std::time::Instant::now()); }
                        let text = stream_dec.push(t);
                        if !text.is_empty() {
                            acc.push_str(&text);
                                match content_start {
                                    None => {
                                        // Search the close tag from reason_emitted, not from 0:
                                        // a second think block must not match the first one's close.
                                        if let Some(idx) = f_close.find(&acc, reason_emitted, think_close) {
                                            if idx > reason_emitted {
                                                let c = reasoning_chunk(&completion_id, created, &model_name, &acc[reason_emitted..idx]);
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
                                                acc.len(), region, &[think_open]);
                                            if safe_end > lead {
                                                let c = content_chunk(&completion_id, created, &model_name, &acc[lead..safe_end]);
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
                                            // post-loop by held_back_remainder.
                                            let region = &acc[reason_emitted..safe];
                                            let safe_end = hold_back_end(
                                                tool_hold,
                                                tool_hold.and_then(|m| f_tool_r.find(&acc[..safe], reason_emitted, m)),
                                                safe, region, &[]);
                                            if safe_end > reason_emitted {
                                                let c = reasoning_chunk(&completion_id, created, &model_name, &acc[reason_emitted..safe_end]);
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
                                        if let Some(upto) = f_open.find(&acc, cs, think_open) {
                                            if upto > content_emitted {
                                                let c = content_chunk(&completion_id, created, &model_name, &acc[content_emitted..upto]);
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
                                            acc.len(), region, &[think_open]);
                                        if safe_end > content_emitted {
                                            let c = content_chunk(&completion_id, created, &model_name, &acc[content_emitted..safe_end]);
                                            if let Some(r) = &otel_req { r.delta(&c); }
                                            yield Ok(Event::default().data(c));
                                            content_emitted = safe_end;
                                        }
                                        }
                                    }
                                }
                            }
                        if !stops.is_empty() {
                            if let Some(p) = stops.iter().zip(f_stops.iter_mut())
                                .filter_map(|(s, f)| f.find(&acc, 0, s)).min() {
                                acc.truncate(p);
                                stop_hit = true;
                                finish = "stop".to_string();
                                break;
                            }
                        }
                    }
                    TokEvent::Finish { reason } => { finish = reason; break; }
                }
            }
            // WP02: release the lane NOW (a stop-string hit cancels it within one scheduler step)
            drop(rx);
            if finish.starts_with("error") {
                // WP02: a mid-stream engine error is an SSE error event (+ a spec finish_reason below)
                crate::reprintln!("[req] stream ended in an engine error after {n} tokens: {finish}");
                yield Ok(Event::default().data(sse_error_event(&finish)));
            }
            // The call was buffered, not streamed (see the hold-back above). The DECISION is
            // crate::tools::finalize_parsed — the one canonical serializer shared with the
            // non-streaming mode — and the held-back text is surfaced by
            // tools::held_back_remainder, so streaming can never silently drop text the JSON
            // mode returns (2026-08-27 user report: a malformed `function=NAME>` block with the
            // `<` missing vanished from the SSE stream while the JSON response leaked it).
            let (_, done_content) = split_think(&acc, think_open, think_close);
            // parse_as_tool_calls == false (keep-tools "none"): never parse — the reply is plain
            // content; finalize_parsed then returns the raw text with finish stop/length.
            let parsed = if parse_as_tool_calls {
                crate::tools::parse(&done_content, req_tools.as_deref())
            } else {
                crate::tools::ParsedOutput { content: done_content.trim().to_string(), tool_calls: Vec::new() }
            };
            if req_tools.is_some() {
                let dump = crate::opts::var(crate::opt!("dump-tools")).is_ok();
                if dump || parsed.tool_calls.is_empty() {
                    crate::reprintln!("[req] raw model output ({} chars): {:?}", done_content.chars().count(),
                              done_content.chars().take(1200).collect::<String>());
                }
            }
            // Phase-4 A2: same empty-argument alarm as the non-streaming path — an agent harness
            // streams, so this branch is the one that actually gets used.
            for a in crate::tools::empty_arg_alarms(&done_content, req_tools.as_deref(), &parsed.tool_calls) {
                crate::reprintln!("[tool-args-alarm] {a}");
            }
            let (_, tool_calls, fin) = crate::tools::finalize_parsed(&done_content, parsed, &finish);
            mt.finish(if finish.starts_with("error") { finish.as_str() } else { fin.as_str() });
            if !tool_calls.is_empty() {
                // Log the ARGUMENTS, not just the names — see the note on the non-streaming path.
                // Agent harnesses stream, so this is the branch that actually gets used, and it was the
                // one printing a bare `tool_calls 1: ["write"]` while a file silently failed to appear.
                for t in &tool_calls {
                    crate::reprintln!("[req] tool_call  {} {}({})", t.id, t.function.name, t.function.arguments);
                }
                let tc = tool_calls_chunk(&completion_id, created, &model_name, &tool_calls);
                if let Some(r) = &otel_req { r.delta(&tc); }
                yield Ok(Event::default().data(tc));
                finish = fin;
            // Watermark is whichever cursor is live: in reasoning mode content_emitted stays 0
            // and the held-back span lives after reason_emitted (tool call before any
            // </think>) — using content_emitted alone would re-emit the whole request's text.
            } else if let Some(held) = crate::tools::held_back_remainder(&acc, reason_emitted.max(content_emitted)) {
                // A tool-call marker was held back but nothing parsed: surface the buffered text
                // as content, exactly what the non-streaming mode returns for the same output.
                let c = content_chunk(&completion_id, created, &model_name, held);
                if let Some(r) = &otel_req { r.delta(&c); }
                yield Ok(Event::default().data(c));
                content_emitted = acc.len();
            }
            {
                let (r_txt, c_txt) = split_think(&acc, think_open, think_close);
                log_generation(&state.tokenizer, &state.stop_ids, &completion_id, &finish,
                               last_tok, prompt_len, n, r_txt.as_deref(), &c_txt,
                               tool_calls.len(), req_tools.as_ref().map(|t| t.len()).unwrap_or(0),
                               effort_for_log.as_deref(), presence_penalty, pp_source);
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
        };
        let mut resp = Sse::new(stream).into_response();
        attach_schema_unenforced_header(&mut resp, schema_unenforced.is_some());
        resp
    } else {
        crate::reprintln!("[req] sync   prompt_tokens={} max_tokens={} stop={:?}", prompt_len, req_max, req.stop);
        let t0 = std::time::Instant::now();
        let mut tokens = Vec::new();
        // WP02: a channel closed without a Finish is an engine failure, not "length"
        let mut finish = "error: engine stopped".to_string();
        let mut first_tok: Option<std::time::Instant> = None;
        let mut cached = 0usize; // v0.7.3 (#8.2): prefix-cache-served prompt tokens
        while let Some(ev) = recv_counted(&mut rx, &bl).await { // LR-5: the one decrement
            match ev {
                TokEvent::Admitted { cached_tokens } => {
                    cached = cached_tokens as usize;
                    mt.cached(cached_tokens as u64); // /metrics counter
                }
                TokEvent::Tok(t) => {
                    tokens.push(t);
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
                            break;
                        }
                    }
                }
                TokEvent::Finish { reason } => { finish = reason; break; }
            }
        }
        drop(rx); // WP02: release the lane now
        let dt = t0.elapsed().as_secs_f32();
        let mut text = state.tokenizer.decode(&tokens, true).unwrap_or_default();
        if !req.stop.is_empty() {
            if let Some(p) = req.stop.iter().filter_map(|s| text.find(s)).min() {
                text.truncate(p); finish = "stop".to_string();
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
        let (think_open, think_close, _) = state.tokenizer.think_tags();
        let (reasoning, content) = split_think(&text, think_open, think_close);

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
            crate::tools::parse(&content, req.tools.as_deref())
        } else {
            crate::tools::ParsedOutput { content: content.trim().to_string(), tool_calls: Vec::new() }
        };
        if req.tools.is_some() {
            let dump = crate::opts::var(crate::opt!("dump-tools")).is_ok();
            if dump || parsed.tool_calls.is_empty() {
                crate::reprintln!("[req] raw model output ({} chars): {:?}", content.chars().count(),
                          content.chars().take(1200).collect::<String>());
            }
        }
        // Phase-4 A2: a call to a tool whose schema declares parameters that arrives with EMPTY
        // arguments while the model emitted a NON-EMPTY body is an argument drop, not a quiet
        // success. Log the raw body so the class can never regress unseen again.
        for a in crate::tools::empty_arg_alarms(&content, req.tools.as_deref(), &parsed.tool_calls) {
            crate::reprintln!("[tool-args-alarm] {a}");
        }
        let (content, tool_calls, finish) = crate::tools::finalize_parsed(&content, parsed, &finish);
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

        log_generation(&state.tokenizer, &state.stop_ids, &completion_id, &finish,
                       tokens.last().copied(), prompt_len, tokens.len(), reasoning.as_deref(),
                       content.as_deref().unwrap_or(""),
                       tool_calls.as_ref().map(|v| v.len()).unwrap_or(0),
                       req.tools.as_ref().map(|t| t.len()).unwrap_or(0),
                       effort, presence_penalty, pp_source);
        dump_tokens(&completion_id, &tokens);
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
                  pp_source: &str) {
    let count = |s: Option<&str>| s.map(|x| tok.encode(x, false).map(|v| v.len()).unwrap_or(0))
        .unwrap_or(0);
    let r_tok = count(reasoning);
    let c_tok = count(Some(content));
    let is_stop = terminal.map(|t| stop_ids.contains(&t)).unwrap_or(false);
    crate::reprintln!("[gen] id={} finish={} terminal_tok={} is_stop_tok={} reasoning_tokens={} \
               content_tokens={} completion_tokens={} prompt_tokens={} tool_calls={} \
               tools_offered={} effort={} presence_penalty={} pp_source={}",
        id, finish, terminal.map(|t| t.to_string()).unwrap_or_else(|| "none".into()), is_stop,
        r_tok, c_tok, completion_tokens, prompt_tokens, tool_calls, tools_offered,
        effort.unwrap_or("<template-default>"), presence_penalty, pp_source);
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
        match state.tokenizer.encode(&rendered, add_special) {
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
                    match state.tokenizer.encode(text, add_special) {
                        Ok(t) => t,
                        Err(e) => return bad(format!("tokenization failed: {e}")),
                    }
                }
            }
            // Token-id round-trip (our extension per the plan; stock vLLM only accepts a string):
            // echo the ids verbatim. add_special_tokens has nothing to encode here — the ids ARE
            // tokens — so it does not apply (same as vLLM's id-prompt paths elsewhere).
            Some(serde_json::Value::Array(items)) => {
                let vocab = state.tokenizer.vocab_size();
                let mut ids: Vec<u32> = Vec::with_capacity(items.len());
                for it in items {
                    match it.as_u64() {
                        Some(id) if (id as usize) < vocab => ids.push(id as u32),
                        _ => return bad(format!(
                            "'prompt' ids must be integers in [0, {}) (got {it})", vocab)),
                    }
                }
                ids
            }
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
    crate::reprintln!("[tokenize] model={model} n={count} add_special={add_special} truncate={truncate:?}");
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
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    min_tokens: Option<usize>,
    #[serde(default)]
    ignore_eos: Option<bool>,
    #[serde(default)]
    seed: Option<u64>,
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
    let prompt_tokens: Vec<u32> = match req.prompt.as_ref() {
        Some(serde_json::Value::String(t)) => match state.tokenizer.encode(t, true) {
            Ok(v) => v,
            Err(e) => return (StatusCode::BAD_REQUEST, format!("tokenize failed: {e}")).into_response(),
        },
        Some(serde_json::Value::Array(a)) if !a.is_empty() && a[0].is_number() =>
            a.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect(),
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
        let stream = async_stream::stream! {
            let mut ntok: usize = 0;
            let mut first_tok: Option<std::time::Instant> = None;
            // WP02: closed without a Finish = engine failure; errors are an SSE error event + "stop"
            let mut finish = "error: engine stopped".to_string();
            while let Some(ev) = recv_counted(&mut rx, &bl).await { // LR-5: the one decrement (H0)
                match ev {
                    // v0.7.3 (#8.2): counted for /metrics; this SSE path emits no usage
                    // chunk today, so no response byte changes here.
                    TokEvent::Admitted { cached_tokens } => { mt.cached(cached_tokens as u64); }
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
            yield Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"));
            let dt = t0.elapsed().as_secs_f32();
            crate::reprintln!("[req] done   completions tok={} ({:.1} tok/s wall) finish={}", ntok, if dt>1e-6 {ntok as f32/dt} else {0.0}, fr);
        };
        return Sse::new(stream).into_response();
    }
    // non-streaming: collect everything, detokenize once, single response.
    let mut toks: Vec<u32> = Vec::with_capacity(req_max);
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
            TokEvent::Tok(t) => { toks.push(t); mt.tok(); }
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

    /// v0.7.3 (#8.3): the public model id resolves as flag > alias > base_model > dir name.
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
