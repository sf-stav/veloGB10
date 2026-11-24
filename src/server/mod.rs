//! The OpenAI-compatible HTTP server: shared state, request/response types and the router; the routes and helpers live in submodules.
//!
//! Map of this directory (items keep their `crate::server::` paths via `pub use self::<sub>::*`):
//! - `sse.rs`: SSE chunk framing, keep-alive, stall drill, serve-option boot lines
//! - `helpers.rs`: request helpers: generation room, human log line, hold-back, token dump, encode
//! - `engine.rs`: engine liveness, waiting cap, TokTx channel, engine-error responses
//! - `reasoning.rs`: thinking split and finish-reason mapping
//! - `logprobs.rs`: logprobs validation and JSON
//! - `chat_params.rs`: chat request preparation (effort, schema, tool_choice)
//! - `chat_stream.rs`: the streaming chat generator
//! - `chat.rs`: POST /v1/chat/completions
//! - `admin.rs`: /metrics, /health, copy-record routes
//! - `tokenize.rs`: /tokenize and /detokenize
//! - `completions.rs`: POST /v1/completions
//! - `vision.rs`: vision panic fallback and recovery
//! - `*_tests.rs`, `*_tests` files: the inline unit-test modules, one file each (declared at the end of this file).
//!
//! Types whose fields are touched by several submodules (`AppState`, the request structs, `ChatSse`, `NextTurn`, `GenLog`) stay here.

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

mod sse;
mod helpers;
mod engine;
mod reasoning;
mod logprobs;
mod chat_params;
mod chat_stream;
mod chat;
mod admin;
mod tokenize;
mod completions;
mod vision;

pub use self::sse::*;
use self::helpers::*;
pub use self::engine::*;
pub use self::reasoning::*;
use self::logprobs::*;
use self::chat_params::*;
pub(crate) use self::chat_stream::*;
use self::chat::*;
use self::admin::*;
use self::tokenize::*;
use self::completions::*;
pub(crate) use self::vision::*;

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

/// LOOPFIX (2026-11-06): what the Qwen3.5 model cards recommend, for a pack WITHOUT a
/// `generation_config.json` (Qwen's own 0.8B-9B checkpoints ship none, nor do our packs of them).
/// Keyed on the model's IDENTITY in config.json (model_type + text size), never the folder name.
/// Returns (unsent-field sampler defaults, presence-penalty default) for a Qwen3.5 / 3.6 / 3.8
/// trunk model (`qwen3_5`, `qwen3_5_moe`), None for anything else (callers keep their values).
///   0.8B / 2B (non-thinking by default; card "non-thinking mode for text tasks"):
///       non-thinking 1.0 / 1.00 / 20, thinking 1.0 / 0.95 / 20, presence 2.0
///   every larger Qwen3.5 / 3.6 / 3.8 (thinking by default; card "thinking mode for general tasks"
///   and "instruct mode for general tasks"): thinking 1.0 / 0.95 / 20, non-thinking 0.7 / 0.8 / 20,
///   presence 1.5
pub fn qwen35_card(model_dir: &str) -> Option<(SamplingDefaults, f32)> {
    let path = format!("{}/config.json", model_dir.trim_end_matches('/'));
    let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).ok()?).ok()?;
    let mt = j.get("model_type").and_then(|v| v.as_str())?;
    if mt != "qwen3_5" && mt != "qwen3_5_moe" { return None; }
    let t = j.get("text_config").unwrap_or(&j);
    let hidden = t.get("hidden_size").and_then(|v| v.as_u64())?;
    let layers = t.get("num_hidden_layers").and_then(|v| v.as_u64())?;
    let small = mt == "qwen3_5" && hidden <= 2048 && layers <= 24;   // Qwen3.5-0.8B (1024) / 2B (2048)
    Some(if small {
        (SamplingDefaults { think: (1.0, 0.95, 20), no_think: (1.0, 1.0, 20) }, 2.0)
    } else {
        (SamplingDefaults { think: (1.0, 0.95, 20), no_think: (0.7, 0.8, 20) }, 1.5)
    })
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
    /// The scheduler can start a `response_format` constraint after the reasoning block
    /// (`SchemaMask::defer_until`). When false, a thinking request's schema is not enforced.
    pub schema_defer: bool,
    /// PR #30 port: the scheduler returns per-token log probabilities (`TokEvent::Logprob`). When false a
    /// request that asks for them is served with `logprobs: null`, said once and counted (owner rule: no refusals).
    pub logprobs_ok: bool,
    /// PR #31 port: the scheduler applies a `response_format` token mask (`BatchRequest::schema`). When false the
    /// request is served unconstrained with `x-json-schema-enforced: none` (never a `full` it did not get).
    pub schema_ok: bool,
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
    // KVPOOL (veloGB10 #43): an asked-for --kv-pool that does not run here (a TP rank, or a model the pool does not
    // back) is one notice line and a /metrics gauge, never a refusal (AGENTS 1b-bis(8)).
    crate::kv_vmm::bind_notice();
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
    /// THINKBUDGET1: vLLM's `thinking_token_budget` — the most tokens this request may spend inside
    /// its reasoning block; at the budget the server emits the end-of-thinking marker and the model
    /// answers. A non-negative integer, or -1 = unlimited; anything else is a 400 with vLLM's
    /// message (crate::think_budget). Absent = no budget: the request is served exactly as before.
    #[serde(default, deserialize_with = "crate::think_budget::deserialize")]
    thinking_token_budget: Option<usize>,
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
    /// The content is a schema-constrained document: never re-enter reasoning on a think tag
    /// inside it (`split_constrained`).
    pub literal_content: bool,
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
    /// TURNRESUME: renders the next request's history once the reply is done (None = --next-turn-prefill off).
    pub next_turn: Option<NextTurn>,
}

/// TURNRESUME (--next-turn-prefill): what the server needs to render this conversation the way its NEXT request will
/// send it once the reply is done: the history as the client sent it, plus this reply as an assistant message (no
/// reasoning, unless the client already sends reasoning back for earlier turns). The ids go to the scheduler
/// (crate::turn_resume::post), which prefills them onto the lane's pre-turn state while the box is idle.
pub(crate) struct NextTurn {
    pub tokenizer: Arc<QwenTokenizer>,
    pub messages: Vec<ChatMessage>,
    pub tools: Option<Vec<serde_json::Value>>,
    pub effort: Option<String>,
    pub kwargs: Option<serde_json::Value>,
    pub thinking: crate::tokenizer::ThinkingMode,
}

impl NextTurn {
    /// None unless a scheduler takes hints, or when the history carries images (their prompt rows are not text).
    pub(crate) fn arm(state: &AppState, messages: &[ChatMessage], tools: Option<&[serde_json::Value]>, effort: Option<&str>,
                      kwargs: Option<&serde_json::Value>) -> Option<NextTurn> {
        if !crate::turn_resume::accepting() || messages.iter().any(|m| !m.images.is_empty()) { return None; }
        Some(NextTurn { tokenizer: Arc::clone(&state.tokenizer), messages: messages.to_vec(), tools: tools.map(|t| t.to_vec()),
                        effort: effort.map(str::to_string), kwargs: kwargs.cloned(), thinking: state.thinking })
    }

    /// The assistant message the client will send back for this reply.
    fn reply_message(echo_reasoning: bool, reasoning: Option<String>, content: String, tool_calls: Vec<ToolCall>) -> ChatMessage {
        ChatMessage {
            role: "assistant".to_string(), content: Some(content), images: Vec::new(),
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls), tool_call_id: None, name: None,
            reasoning_content: if echo_reasoning { reasoning.filter(|r| !r.trim().is_empty()) } else { None },
            layout: Vec::new(), unsupported_parts: Vec::new(),
        }
    }

    /// Render + encode off the async thread, then post the ids. Never fails the request.
    pub(crate) fn post(self, reasoning: Option<String>, content: String, tool_calls: Vec<ToolCall>) {
        let job = move || {
            // a client that sends reasoning back for its earlier turns will send this one's too
            let echo = self.messages.iter()
                .any(|m| m.role == "assistant" && m.reasoning_content.as_deref().map_or(false, |r| !r.trim().is_empty()));
            let mut msgs = self.messages;
            msgs.push(NextTurn::reply_message(echo, reasoning, content, tool_calls));
            match self.tokenizer.apply_chat_template_no_gen(&msgs, self.tools.as_deref(), self.effort.as_deref(),
                                                            self.kwargs.as_ref(), self.thinking)
                .and_then(|s| self.tokenizer.encode(&s, true)) {
                Ok(ids) => crate::turn_resume::post(ids),
                Err(e) => crate::dreprintln!("[req] next-turn render failed (no background prefill for this reply): {e:#}"),
            }
        };
        match tokio::runtime::Handle::try_current() {
            Ok(h) => { h.spawn_blocking(job); }
            Err(_) => job(),
        }
    }
}

pub(crate) struct GenLog {
    pub tokenizer: Arc<QwenTokenizer>,
    pub stop_ids: Vec<u32>,
    pub effort: Option<String>,
    pub presence_penalty: f32,
    pub pp_source: &'static str,
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
    /// THINKBUDGET1: vLLM's `thinking_token_budget` — the most tokens this request may spend inside
    /// its reasoning block; at the budget the server emits the end-of-thinking marker and the model
    /// answers. A non-negative integer, or -1 = unlimited; anything else is a 400 with vLLM's
    /// message (crate::think_budget). Absent = no budget: the request is served exactly as before.
    #[serde(default, deserialize_with = "crate::think_budget::deserialize")]
    thinking_token_budget: Option<usize>,
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
        // COPYREC1: the cross-request copy record's state, and emptying it (--copy-span-record)
        .route("/v1/copy-record", get(copy_record_get).delete(copy_record_delete))
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
mod thinking_budget_parse_tests;

#[cfg(test)]
mod top_k_parse_tests;

#[cfg(test)]
mod logprobs_span_tests;

#[cfg(test)]
mod logprobs_param_tests;

#[cfg(test)]
mod prompt_token_id_tests;

#[cfg(test)]
mod reasoning_start_tests;

#[cfg(test)]
mod context_budget_tests;

#[cfg(test)]
mod wp08_validation_tests;

#[cfg(test)]
mod grow_find_tests;

#[cfg(test)]
mod model_id_tests;

#[cfg(test)]
mod host_tests;

#[cfg(test)]
mod harden_tests;

#[cfg(test)]
mod harden_lr5_toktx_tests;

#[cfg(test)]
mod nonetools_tests;

#[cfg(test)]
/// REL_V0_7_3 review H1: the tests the HARDEN report CLAIMED but never wrote, now real.
/// Every test below drives the actual production functions (`over_wait_limit`, `server_busy`,
/// `over_backlog`, `tok_channel` + `recv_counted`/`try_recv_counted`) — no hand-simulated
/// counters, no re-implemented predicates. Listed by name in the dated note in
/// PLAN/REL_HARDEN_REPORT.md.
mod harden_review_h1_tests;

#[cfg(test)]
/// S8 (REL_V0_7_3 review): the model id is escaped in every hand-built SSE chunk — a quote or
/// backslash in `--model-name` (honoured verbatim since v0.7.3) used to break every chunk.
mod s8_sse_escape_tests;

#[cfg(test)]
/// S10 (REL_V0_7_3 review): GET /v1/models/<id-with-slash> — both the raw-slash spelling and
/// the %2F form must route to the handler. Drives a real TCP listener (no tower test-util in
/// the tree) with the SAME two route patterns create_router registers.
mod s10_models_route_tests;

#[cfg(test)]
mod vision_fallback_tests;

/// S-B9-REL-SMALLFIXES item 5: the pure parts of the `[gen]` line.
#[cfg(test)]
mod gen_line_tests;

/// S-B9-REL-SMALLFIXES item 6: the streamed /v1/completions usage chunk (a known gap in the
/// v0.7.3 CHANGELOG: `stream_options.include_usage` was accepted and dropped on this path).
#[cfg(test)]
mod completions_usage_tests;

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
pub(crate) mod benchfix_sse_tests;

#[cfg(test)]
mod loopfix_card_tests;
