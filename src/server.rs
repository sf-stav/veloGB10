use axum::{
    extract::{DefaultBodyLimit, Json, State},
    http::StatusCode,
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
use crate::tokenizer::{QwenTokenizer, ChatMessage, ToolCall};
use crate::{Usage, Timings, make_timings};

#[derive(Clone)]
pub struct AppState {
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
    /// `--output-prompts [cap]`: log every chat-completion request in human-readable form
    /// (effective params, one line per turn, rendered-prompt excerpt up to `cap` chars).
    /// 0 = off (default).
    pub output_prompts: usize,
    /// KV cache depth, in positions. NOTHING used to check a prompt against it: an over-long prompt
    /// ran `write_kv_prefill` straight past the end of the cache and corrupted the next allocation.
    pub max_seq_len: usize,
    /// Decode positions reserved beyond `max_tokens` for speculative verification/re-prime.
    /// Mirrors the scheduler reserve so an HTTP-clamped request is always admissible.
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
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: i64,
    owned_by: String,
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

fn esc(t: &str) -> String {
    // SSE chunks interpolate this inside a JSON string. Hand-escaping only backslashes,
    // quotes and newlines left literal tabs (common in Go), carriage returns and other
    // control characters in the payload. Clients discard that invalid JSON, which looks
    // like the first characters of indented lines were truncated. Use the complete JSON
    // escaping rules, then remove the surrounding quotes supplied by the serializer.
    let quoted = serde_json::to_string(t).expect("serializing a Rust string cannot fail");
    quoted[1..quoted.len() - 1].to_string()
}

/// (The think close marker is resolved per-request from the model's vocab — see
/// QwenTokenizer::think_tags. Qwen: `</think>`; hy_v3: `</think:opensource>`.)

/// Longest suffix of `s` that is a proper (partial) prefix of `marker` — text that could be the start
/// of the marker arriving across decode chunks, and so must be held back rather than forwarded.
/// `--output-prompts [cap]` — log the chat-completion call in human-readable form: effective
/// parameters, one line per message turn, and the exact rendered prompt the model sees
/// (excerpt up to `cap` chars; RUST_INFER_DUMP_PROMPT=1 still writes the full string to /tmp
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
    eprintln!("  temperature={}  top_p={}  top_k={}", req.temperature, req.top_p, req.top_k);
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
        eprintln!("    … (+{} more chars of {total} — full dump: RUST_INFER_DUMP_PROMPT=1)", total - trunc);
    }
    eprintln!("[prompt] ══════════════════════════════════════════════════════════════════");
}

fn partial_overlap(s: &str, marker: &str) -> usize {
    (1..marker.len()).rev().find(|&k| s.ends_with(&marker[..k])).unwrap_or(0)
}

fn partial_think_overlap(s: &str, marker: &str) -> usize { partial_overlap(s, marker) }

/// Whether generation starts inside an unclosed think block in the prompt that was ACTUALLY
/// rendered. This must not be inferred from the model family: Qwen normally primes `<think>`, but
/// `reasoning_effort=no_think` renders the same family with thinking disabled. Conversely hy_v3 can
/// start either inside or outside its think block depending on the request.
fn prompt_ends_inside_think(prompt: &str, think_open: &str) -> bool {
    // Only the generation-prompt suffix is authoritative. Searching the whole rendered conversation
    // would let a literal `<think>` in the user's text alter the stream state.
    prompt.trim_end().ends_with(think_open)
}

/// Map the union accepted by the HTTP/CLI surface onto the vocabulary of the active template.
/// In particular, OpenAI's `high` must never mean "disable thinking" (the old mapping did exactly
/// that for Qwen, which also made the streaming-state bug intermittent across clients).
fn normalize_reasoning_effort(effort: &str, hy_v3: bool) -> &str {
    match (effort, hy_v3) {
        ("high" | "medium" | "xhigh" | "max", true) => "high",
        ("high" | "max", false) => "xhigh",
        ("xhigh" | "medium" | "low", false) | ("low", true) => effort,
        ("no_think" | "minimal" | "none" | "off" | "", _) => "no_think",
        (other, _) => other,
    }
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
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    stream: bool,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    /// Optional PRNG seed for reproducible sampling (used by stochastic MTP path).
    #[serde(default)]
    seed: Option<u64>,
    /// Stop sequences: accept either a string or a list of strings (OpenAI spec).
    #[serde(default, deserialize_with = "deserialize_stop")]
    stop: Vec<String>,
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
    /// OpenAI streaming options. Only meaningful with stream=true; serde used to drop it silently,
    /// so a client asking for include_usage got nothing and no [DONE] sentinel either.
    #[serde(default)]
    stream_options: Option<StreamOptions>,
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
    /// Extension (llama.cpp naming). Strict clients ignore unknown top-level fields.
    timings: Timings,
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
}

/// The scheduler's internal backstop reason (batch.rs) is not an OpenAI value — the spec is
/// stop|length|tool_calls|content_filter. Map it to what it means: generation ran out of room.
fn spec_finish_reason(reason: &str) -> &str {
    if reason == "context_length_exceeded" { "length" } else { reason }
}

fn generation_room(max_seq_len: usize, prompt_len: usize, decode_headroom: usize) -> Option<usize> {
    let used = prompt_len.checked_add(decode_headroom)?;
    max_seq_len.checked_sub(used).filter(|&room| room > 0)
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    // Log the request parameters the client sent (useful for debugging OpenWebUI behavior)
    eprintln!(
        "[req] params  temp={:?} top_p={:?} top_k={} max_tok={:?} rep_pen={:?} presence={:?} freq={:?} stream={}",
        req.temperature, req.top_p, req.top_k,
        req.max_tokens, req.repetition_penalty, req.presence_penalty, req.frequency_penalty,
        req.stream
    );
    if let Some(t) = &req.tools {
        let names: Vec<&str> = t.iter()
            .filter_map(|x| x.pointer("/function/name").and_then(|v| v.as_str())).collect();
        eprintln!("[req] tools   {} offered: {:?} tool_choice={:?}", t.len(), names, req.tool_choice);
    }
    // Resolve the effective reasoning effort. Two families, two vocabularies:
    //   - hy_v3's template accepts `no_think|low|high` (default low), and its Rust dsv4 path treats
    //     None/"" as low.
    //   - Qwen3.5's template accepts `xhigh|medium|low` (default xhigh) and RAISES on anything else.
    // Forward the client's value verbatim when it is one of the Qwen values; normalize the OpenAI
    // convention (minimal|low|medium|high) onto a family-agnostic low/high only when a value is
    // given. When NEITHER the request nor --reasoning-effort specifies one, pass None so the model's
    // own template default wins (xhigh for Qwen, low for hy_v3) — never a hardcoded guess.
    let hy_v3 = !state.tokenizer.think_tags().2;
    let effort: Option<&str> = req.reasoning_effort.as_deref()
        .or(state.reasoning_effort.as_deref())
        .map(|e| {
            let n = normalize_reasoning_effort(e, hy_v3);
            if n != e {
                eprintln!("[req] reasoning_effort '{e}' normalized to '{n}' for {}",
                          if hy_v3 { "hy_v3" } else { "Qwen" });
            }
            if !matches!(n, "xhigh" | "medium" | "low" | "no_think" | "high") {
                eprintln!("[req] reasoning_effort '{e}' not a known level (passing through verbatim)");
            }
            n
        });
    // tool_choice: "none" renders the turn without tools; "required" / a named function FORCE the
    // call by seeding the assistant turn with the template's tool-call opener (the thinking block
    // is closed empty first — a forced call cannot start inside <think>). The seed is prepended to
    // the generated text before parsing so the serializer sees a complete call.
    let (tools_for_render, forced_prefix): (Option<&[serde_json::Value]>, String) = match &req.tool_choice {
        Some(serde_json::Value::String(c)) if c == "none" => (None, String::new()),
        Some(serde_json::Value::String(c)) if c == "required" && req.tools.is_some() => (req.tools.as_deref(), "<tool_call>\n<function=".to_string()),
        Some(v) if v.get("type").and_then(|t| t.as_str()) == Some("function") && req.tools.is_some() => {
            let name = v["function"]["name"].as_str().unwrap_or("").to_string();
            (req.tools.as_deref(), format!("<tool_call>\n<function={name}>\n"))
        }
        _ => (req.tools.as_deref(), String::new()),
    };
    let t_render = std::time::Instant::now();
    let mut prompt = match state.tokenizer.apply_chat_template(&req.messages, tools_for_render, effort) {
        Ok(p) => p,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };
    if !forced_prefix.is_empty() {
        if let Some(p) = prompt.strip_suffix("<think>\n") { prompt = format!("{p}<think>\n\n</think>\n\n"); }
        prompt.push_str(&forced_prefix);
        eprintln!("[req] tool_choice forces a call: seeded {:?}", forced_prefix);
    }
    let forced_prefix_stream = forced_prefix.clone();
    let render_ms = t_render.elapsed().as_secs_f64() * 1000.0;

    // Optional diagnostic: dump the exact rendered prompt string so the bytes a model
    // actually sees can be inspected/diffed across models or turns. Enable with
    // RUST_INFER_DUMP_PROMPT=1. Writes /tmp/rust_infer_prompt_<n>.txt per request.
    if std::env::var("RUST_INFER_DUMP_PROMPT").is_ok() {
        static DUMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = DUMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dump_path = format!("/tmp/rust_infer_prompt_{}.txt", n);
        if std::fs::write(&dump_path, &prompt).is_ok() {
            eprintln!("[req] dumped prompt ({} chars) -> {}", prompt.chars().count(), dump_path);
        }
    }

    let t_encode = std::time::Instant::now();
    let mut prompt_tokens = match state.tokenizer.encode(&prompt, true) {
        Ok(t) => t,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };
    let encode_ms = t_encode.elapsed().as_secs_f64() * 1000.0;
    // TTFT fix 0 attribution (GB10_PREFILL_TRACE): the request-path pre-model costs.
    if crate::env_knob("GB10_PREFILL_TRACE", "DSV4_PREFILL_TRACE").is_some() {
        eprintln!("[pf] server render={render_ms:.3}ms encode={encode_ms:.3}ms");
    }

    // V3 vision dispatch: if any message carries images (and the server has a vision tower),
    // decode+preprocess+run the tower, expand the image_pad span in the token stream, and carry the
    // merged embeddings + spans for the model prefill splice. Text-only traffic is unchanged.
    let mut image_embeds: Option<Vec<f32>> = None;
    let mut image_spans: Vec<crate::vision_encoder::ImageSpan> = Vec::new();
    let urls: Vec<String> = req.messages.iter()
        .flat_map(|m| m.images.iter().filter_map(|i| i.url.clone()))
        .collect();
    if !urls.is_empty() {
        let vt0 = std::time::Instant::now();
        // Prefer the GPU tower (fast path) unless --vision-cpu forces the CPU reference.
        let prep = if let Some(g) = state.vision_gpu.clone() {
            if !state.vision_cpu {
                let mut gvt = g.lock().expect("vision_gpu lock");
                crate::vision_encoder::prepare_vision_request_gpu(&mut gvt, &urls, &prompt_tokens)
            } else {
                state.vision_tower.as_ref().map(|t| crate::vision_encoder::prepare_vision_request(t, &urls, &prompt_tokens)).unwrap_or_else(|| Err(anyhow::anyhow!("vision_gpu forced but no CPU tower")))
            }
        } else if let Some(tower) = state.vision_tower.clone() {
            crate::vision_encoder::prepare_vision_request(&tower, &urls, &prompt_tokens)
        } else {
            Err(anyhow::anyhow!("no vision tower loaded"))
        };
        eprintln!("[vision] dispatch {} images, prepare took {} ms, len={}",
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
            .apply_chat_template_no_gen(&req.messages, req.tools.as_deref(), effort).ok()
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
    if req_max < asked {
        eprintln!("[req] max_tokens clamped {} -> {} (KV cache: {}-token prompt + {} reserved decode positions of {}; \
                   raise --max-seq-len)", asked, req_max, prompt_len, state.decode_headroom, state.max_seq_len);
    }
    let temperature = req.temperature;
    let top_p = req.top_p.max(0.01);

    // Submit to the batching scheduler and receive tokens on a channel.
    // Use request's penalties if explicitly set, else fall back to server defaults.
    let rep_penalty = req.repetition_penalty.unwrap_or(state.default_rep_penalty);
    let presence_penalty = req.presence_penalty.unwrap_or(state.default_presence_penalty);
    let frequency_penalty = req.frequency_penalty.unwrap_or(state.default_frequency_penalty);

    let (tx, mut rx) = mpsc::unbounded_channel::<TokEvent>();
    let request = BatchRequest {
        prompt: prompt_tokens.clone(),
        max_new: req_max,
        temperature,
        received_at: std::time::Instant::now(),
        top_p,
        top_k: req.top_k,
        rep_penalty,
        presence_penalty,
        frequency_penalty,
        tx,
        seed: req.seed,
        ckpt_at,
        domain: crate::batch::classify_domain(&prompt),
        image_embeds,
        image_spans,
    };
    let _ = state.scheduler.send(request);

    let content_chunk = |cid: &str, created: i64, model: &str, text: &str| {
        format!("{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}",
            cid, created, model, esc(text))
    };
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
    let reasoning_chunk = |cid: &str, created: i64, model: &str, text: &str| {
        format!("{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"reasoning_content\":\"{}\"}},\"finish_reason\":null}}]}}",
            cid, created, model, esc(text))
    };

    if req.stream {
        eprintln!("[req] stream  prompt_tokens={} max_tokens={} stop={:?}", prompt_len, req_max, req.stop);
        let tokenizer = Arc::clone(&state.tokenizer);
        let model_name = req.model.clone().unwrap_or_else(|| state.model_name.clone());
        let stops = req.stop.clone();
        let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();
        let t0 = std::time::Instant::now();
        let req_tools = req.tools.clone();
        let include_usage = req.stream_options.as_ref().map(|o| o.include_usage).unwrap_or(true);
        // Resolve markers from the model's vocab, but derive the initial state from the prompt that
        // was ACTUALLY rendered. Family-based inference is wrong for request-level no-think: Qwen's
        // family default is a primed `<think>` block, while `reasoning_effort=no_think` renders a
        // plain assistant prompt. The old inference consequently streamed the whole normal answer as
        // `reasoning_content` because no `</think>` was ever supposed to arrive. Inspecting the
        // generation-prompt suffix also handles hy_v3 and forced tool-call prefixes without special
        // cases, while ignoring literal markers in user messages.
        let (think_open, think_close, _) = tokenizer.think_tags();
        let starts_in_reasoning = prompt_ends_inside_think(&prompt, think_open);

        let stream = async_stream::stream! {
            yield Ok::<Event, axum::Error>(Event::default().data(
                format!("{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\"}},\"finish_reason\":null}}]}}",
                    completion_id, created, model_name)));
            // Byte-level stream decoder: the per-token decode path above would mangle every
            // multi-byte char split across tokens (all emoji) into "�" — the crate's ByteLevel
            // decode is String::from_utf8_lossy per call. Reassembles raw bytes across tokens.
            let mut stream_dec = tokenizer.stream_decoder();
            let mut acc = forced_prefix_stream.clone();
            let mut n = 0usize;
            let mut stop_hit = false;
            let mut finish = "length".to_string();
            let mut first_tok: Option<std::time::Instant> = None;
            // Thinking-model split: qwen's prompt is primed with `<think>\n`, so the generated stream
            // is `…reasoning…</think>\n\nanswer`. Pre-close text -> reasoning_content, post-close
            // -> content. hy_v3's no_think prompt already closed the (empty) block, so it starts as
            // content. The close marker may span decode chunks, so we hold back a tail that could be
            // its prefix until more text arrives.
            let mut content_start: Option<usize> = if starts_in_reasoning { None } else { Some(0) };
            let mut reason_emitted: usize = 0;
            let mut content_emitted: usize = 0;
            while let Some(ev) = rx.recv().await {
                match ev {
                    TokEvent::Tok(t) => {
                        n += 1;
                        if first_tok.is_none() { first_tok = Some(std::time::Instant::now()); }
                        let text = stream_dec.push(t);
                        if !text.is_empty() {
                            acc.push_str(&text);
                                match content_start {
                                    None => {
                                        // Search the close tag from reason_emitted, not from 0:
                                        // a second think block must not match the first one's close.
                                        if let Some(idx) = acc[reason_emitted..].find(think_close).map(|i| reason_emitted + i) {
                                            if idx > reason_emitted {
                                                yield Ok(Event::default().data(reasoning_chunk(&completion_id, created, &model_name, &acc[reason_emitted..idx])));
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
                                            let safe_end = match region.find(TOOL_OPEN) {
                                                Some(i) => lead + i,
                                                None => acc.len() - partial_overlap(region, TOOL_OPEN)
                                                    .max(partial_overlap(region, think_open)),
                                            };
                                            if safe_end > lead {
                                                yield Ok(Event::default().data(content_chunk(&completion_id, created, &model_name, &acc[lead..safe_end])));
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
                                            let safe_end = match region.find(TOOL_OPEN) {
                                                Some(i) => reason_emitted + i,
                                                None => safe - partial_overlap(region, TOOL_OPEN),
                                            };
                                            if safe_end > reason_emitted {
                                                yield Ok(Event::default().data(reasoning_chunk(&completion_id, created, &model_name, &acc[reason_emitted..safe_end])));
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
                                        if let Some(tp) = region.find(think_open) {
                                            let upto = cs + tp;
                                            if upto > content_emitted {
                                                yield Ok(Event::default().data(content_chunk(&completion_id, created, &model_name, &acc[content_emitted..upto])));
                                            }
                                            content_start = None;
                                            reason_emitted = upto + think_open.len();
                                        } else {
                                        // Hold back anything that is, or could become, a tool call.
                                        // Forwarding `<tool_call>` as content makes the harness render
                                        // XML in the chat and never invoke the tool. Same hold-back
                                        // for a think-open prefix spanning decode chunks.
                                        let safe_end = match region.find(TOOL_OPEN) {
                                            Some(i) => cs + i,          // a call has started: emit nothing more
                                            None => acc.len() - partial_overlap(region, TOOL_OPEN)
                                                .max(partial_overlap(region, think_open)),
                                        };
                                        if safe_end > content_emitted {
                                            yield Ok(Event::default().data(content_chunk(&completion_id, created, &model_name, &acc[content_emitted..safe_end])));
                                            content_emitted = safe_end;
                                        }
                                        }
                                    }
                                }
                            }
                        if !stops.is_empty() {
                            if let Some(p) = stops.iter().filter_map(|s| acc.find(s)).min() {
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
            // The call was buffered, not streamed (see the hold-back above). The DECISION is
            // crate::tools::finalize_parsed — the one canonical serializer shared with the
            // non-streaming mode — and the held-back text is surfaced by
            // tools::held_back_remainder, so streaming can never silently drop text the JSON
            // mode returns (2026-08-27 user report: a malformed `function=NAME>` block with the
            // `<` missing vanished from the SSE stream while the JSON response leaked it).
            let (_, done_content) = split_think(&acc, think_open, think_close);
            let parsed = crate::tools::parse(&done_content, req_tools.as_deref());
            if req_tools.is_some() {
                let dump = std::env::var("RUST_INFER_DUMP_TOOLS").is_ok();
                if dump || parsed.tool_calls.is_empty() {
                    eprintln!("[req] raw model output ({} chars): {:?}", done_content.chars().count(),
                              done_content.chars().take(1200).collect::<String>());
                }
            }
            let (_, tool_calls, fin) = crate::tools::finalize_parsed(&done_content, parsed, &finish);
            if !tool_calls.is_empty() {
                // Log the ARGUMENTS, not just the names — see the note on the non-streaming path.
                // Agent harnesses stream, so this is the branch that actually gets used, and it was the
                // one printing a bare `tool_calls 1: ["write"]` while a file silently failed to appear.
                for t in &tool_calls {
                    eprintln!("[req] tool_call  {} {}({})", t.id, t.function.name, t.function.arguments);
                }
                yield Ok(Event::default().data(tool_calls_chunk(&completion_id, created, &model_name, &tool_calls)));
                finish = fin;
            // Watermark is whichever cursor is live: in reasoning mode content_emitted stays 0
            // and the held-back span lives after reason_emitted (tool call before any
            // </think>) — using content_emitted alone would re-emit the whole request's text.
            } else if let Some(held) = crate::tools::held_back_remainder(&acc, reason_emitted.max(content_emitted)) {
                // A tool-call marker was held back but nothing parsed: surface the buffered text
                // as content, exactly what the non-streaming mode returns for the same output.
                yield Ok(Event::default().data(content_chunk(&completion_id, created, &model_name, held)));
                content_emitted = acc.len();
            }
            let final_chunk = format!("{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{}\"}}]}}",
                completion_id, created, model_name, spec_finish_reason(&finish));
            yield Ok(Event::default().data(final_chunk));
            if include_usage {
                // Spec stream-usage chunk: empty choices, top-level usage. `timings` rides along
                // as the extension field (strict clients ignore it).
                let usage_chunk = serde_json::json!({
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_name, "choices": [],
                    "usage": {"prompt_tokens": prompt_len, "completion_tokens": n,
                              "total_tokens": prompt_len + n},
                    "timings": make_timings(t0, first_tok, prompt_len, n),
                });
                yield Ok(Event::default().data(usage_chunk.to_string()));
            }
            // The OpenAI SSE terminator. Without it a strict client sits on an open stream
            // waiting for more events after the finish chunk.
            yield Ok(Event::default().data("[DONE]"));
            let dt = t0.elapsed().as_secs_f32();
            eprintln!("[req] done   tok={} ({:.1} tok/s wall) finish={} stop_hit={}", n, if dt>1e-6 {n as f32/dt} else {0.0}, finish, stop_hit);
        };
        Sse::new(stream).into_response()
    } else {
        eprintln!("[req] sync   prompt_tokens={} max_tokens={} stop={:?}", prompt_len, req_max, req.stop);
        let t0 = std::time::Instant::now();
        let mut tokens = Vec::new();
        let mut finish = "length".to_string();
        let mut first_tok: Option<std::time::Instant> = None;
        while let Some(ev) = rx.recv().await {
            match ev {
                TokEvent::Tok(t) => {
                    tokens.push(t);
                    if first_tok.is_none() { first_tok = Some(std::time::Instant::now()); }
                    // Apply stop strings LIVE, not just post-hoc: on a hit, break AND let rx drop —
                    // the scheduler sees the closed channel and cancels the lane instead of decoding
                    // to EOS/max_new. Only the tail is searched (a stop string spans a few tokens;
                    // one longer than the window is still honoured post-hoc below, just not early).
                    if !req.stop.is_empty() && tokens.len() % 4 == 0 {
                        let tail = &tokens[tokens.len().saturating_sub(96)..];
                        let s = state.tokenizer.decode(tail, true).unwrap_or_default();
                        if req.stop.iter().any(|x| !x.is_empty() && s.contains(x.as_str())) { break; }
                    }
                }
                TokEvent::Finish { reason } => { finish = reason; break; }
            }
        }
        let dt = t0.elapsed().as_secs_f32();
        let mut text = format!("{forced_prefix}{}", state.tokenizer.decode(&tokens, true).unwrap_or_default());
        if !req.stop.is_empty() {
            if let Some(p) = req.stop.iter().filter_map(|s| text.find(s)).min() {
                text.truncate(p); finish = "stop".to_string();
            }
        }
        eprintln!("[req] done   tok={} ({:.1} tok/s wall) finish={}", tokens.len(), if dt>1e-6 {tokens.len() as f32/dt} else {0.0}, finish);
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
        // ran but nothing happened" report. Log it when asked (RUST_INFER_DUMP_TOOLS=1), and ALWAYS log
        // it when tools were offered and we parsed nothing — that combination means either the model
        // declined, or it emitted a call we failed to understand, and those need very different fixes.
        let parsed = crate::tools::parse(&content, req.tools.as_deref());
        if req.tools.is_some() {
            let dump = std::env::var("RUST_INFER_DUMP_TOOLS").is_ok();
            if dump || parsed.tool_calls.is_empty() {
                eprintln!("[req] raw model output ({} chars): {:?}", content.chars().count(),
                          content.chars().take(1200).collect::<String>());
            }
        }
        let (content, tool_calls, finish) = crate::tools::finalize_parsed(&content, parsed, &finish);
        if !tool_calls.is_empty() {
            // Log the ARGUMENTS, not just the names. When opencode reported a write as successful and
            // no file appeared, the log said `tool_calls 1: ["write"]` — which is exactly enough to
            // know a tool was called and not nearly enough to know what it was told to do. The path the
            // model chose is the whole question.
            for t in &tool_calls {
                eprintln!("[req] tool_call  {} {}({})", t.id, t.function.name, t.function.arguments);
            }
        }
        let tool_calls = if tool_calls.is_empty() { None } else { Some(tool_calls) };

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
            }],
            usage: Usage {
                prompt_tokens: prompt_len,
                completion_tokens: tokens.len(),
                total_tokens: prompt_len + tokens.len(),
            },
            timings: make_timings(t0, first_tok, prompt_len, tokens.len()),
        };
        Json(response).into_response()
    }
}
async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/v1/models/:id", get(get_model))
        .route("/health", get(health))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        // Base64 image bodies inflate ~4/3x; a high-res PNG at ~2-4 MB exceeds axum's 2 MB default
        // (""Failed to buffer the request body: length limit exceeded"" on image requests). Raise it
        // so images that other engines accept also arrive here.
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(state)
}

#[cfg(test)]
mod context_budget_tests {
    use super::{esc, generation_room, normalize_reasoning_effort, prompt_ends_inside_think};

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

    #[test]
    fn streaming_state_follows_the_rendered_prompt_not_the_model_family() {
        assert!(prompt_ends_inside_think(
            "<|im_start|>assistant\n<think>\n",
            "<think>",
        ));

        // Qwen with reasoning_effort=no_think: same tokenizer family, but the template does not
        // prime a think block. A normal answer must therefore stream through `content`.
        assert!(!prompt_ends_inside_think(
            "<|im_start|>assistant\n",
            "<think>",
        ));

        // Forced tool calls explicitly close the template's primed block before their prefix.
        assert!(!prompt_ends_inside_think(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=",
            "<think>",
        ));

        // hy_v3 uses different markers and can be rendered in either state as well.
        assert!(prompt_ends_inside_think(
            "assistant<think:opensource>",
            "<think:opensource>",
        ));
        assert!(!prompt_ends_inside_think(
            "assistant<think:opensource></think:opensource>",
            "<think:opensource>",
        ));

        // A literal marker in user content must not prime a no-think generation.
        assert!(!prompt_ends_inside_think(
            "<|im_start|>user\nplease print <think><|im_end|>\n<|im_start|>assistant\n",
            "<think>",
        ));
    }

    #[test]
    fn reasoning_effort_high_stays_a_thinking_level() {
        assert_eq!(normalize_reasoning_effort("high", false), "xhigh");
        assert_eq!(normalize_reasoning_effort("high", true), "high");
        assert_eq!(normalize_reasoning_effort("medium", true), "high");
        assert_eq!(normalize_reasoning_effort("max", false), "xhigh");
        assert_eq!(normalize_reasoning_effort("max", true), "high");
        assert_eq!(normalize_reasoning_effort("minimal", false), "no_think");
    }
}
