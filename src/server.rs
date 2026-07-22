use axum::{
    extract::{Json, State},
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

#[derive(Clone)]
pub struct AppState {
    pub scheduler: mpsc::UnboundedSender<BatchRequest>,
    pub tokenizer: Arc<QwenTokenizer>,
    pub model_name: String,
    pub default_max_tokens: usize,
    pub default_rep_penalty: f32,
    pub default_presence_penalty: f32,
    pub default_frequency_penalty: f32,
    /// KV cache depth, in positions. NOTHING used to check a prompt against it: an over-long prompt
    /// ran `write_kv_prefill` straight past the end of the cache and corrupted the next allocation.
    pub max_seq_len: usize,
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

fn esc(t: &str) -> String {
    t.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

/// (The think close marker is resolved per-request from the model's vocab — see
/// QwenTokenizer::think_tags. Qwen: `</think>`; hy_v3: `</think:opensource>`.)

/// Longest suffix of `s` that is a proper (partial) prefix of `marker` — text that could be the start
/// of the marker arriving across decode chunks, and so must be held back rather than forwarded.
fn partial_overlap(s: &str, marker: &str) -> usize {
    (1..marker.len()).rev().find(|&k| s.ends_with(&marker[..k])).unwrap_or(0)
}

fn partial_think_overlap(s: &str, marker: &str) -> usize { partial_overlap(s, marker) }

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
    model: String,
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

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
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
    let prompt = match state.tokenizer.apply_chat_template(&req.messages, req.tools.as_deref()) {
        Ok(p) => p,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };

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

    let prompt_tokens = match state.tokenizer.encode(&prompt, true) {
        Ok(t) => t,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };

    let prompt_len = prompt_tokens.len();

    // Where to snapshot the GDN state: the message boundary, i.e. this prompt without its trailing
    // generation prompt. Everything up to here is what the NEXT turn replays verbatim. Rendering the
    // template a second time costs microseconds and saves a whole re-prefill per turn.
    let ckpt_at = state.tokenizer
        .apply_chat_template_no_gen(&req.messages, req.tools.as_deref()).ok()
        .and_then(|s| state.tokenizer.encode(&s, true).ok())
        .map(|t| t.len())
        .filter(|&n| n > 0 && n < prompt_len);

    // The KV cache holds exactly `max_seq_len` positions. A prompt past that end used to be written
    // out of bounds — silently, corrupting whatever allocation followed, which showed up as two
    // identical prefills disagreeing. Reject what cannot fit, and cap generation at the room left:
    // running short is a `finish_reason: "length"`, which is in the contract. Corruption is not.
    if prompt_len >= state.max_seq_len {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
            "message": format!("This model's maximum context length is {} tokens, but your messages \
                                came to {} tokens. Shorten the input or restart the server with a \
                                larger --max-seq-len.", state.max_seq_len, prompt_len),
            "type": "invalid_request_error", "code": "context_length_exceeded",
        }}))).into_response();
    }
    let room = state.max_seq_len - prompt_len;
    let asked = req.max_tokens.unwrap_or(state.default_max_tokens);
    let req_max = asked.min(room);
    // If the KV cache forced generation shorter than asked, SAY SO. A thinking model spends a big fixed
    // chunk on its <think> block, so a silently-shrunk budget looks like "truncated output / only
    // reasoning" as a conversation grows — which is exactly how this surfaced in the wild. Raise
    // --max-seq-len (graphs cost ~nothing here; KV is ~64 KB/token) to give multi-turn room.
    if req_max < asked {
        eprintln!("[req] max_tokens clamped {} -> {} (KV cache room: {} of {} used by the {}-token prompt; \
                   raise --max-seq-len)", asked, req_max, prompt_len, state.max_seq_len, prompt_len);
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
        top_p,
        top_k: req.top_k,
        rep_penalty,
        presence_penalty,
        frequency_penalty,
        tx,
        seed: req.seed,
        ckpt_at,
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
        let model_name = req.model.clone();
        let stops = req.stop.clone();
        let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();
        let t0 = std::time::Instant::now();
        let req_tools = req.tools.clone();
        // Think markers + the initial reasoning/content state, from the model's vocab: qwen is
        // primed into a think block (starts in reasoning); hy_v3's no_think prompt closes the empty
        // block itself (starts as content).
        let (think_open, think_close, starts_in_reasoning) = tokenizer.think_tags();

        let stream = async_stream::stream! {
            yield Ok::<Event, axum::Error>(Event::default().data(
                format!("{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\"}},\"finish_reason\":null}}]}}",
                    completion_id, created, model_name)));
            let mut acc = String::new();
            let mut n = 0usize;
            let mut stop_hit = false;
            let mut finish = "length".to_string();
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
                        if let Ok(text) = tokenizer.decode(&[t], true) {
                            if !text.is_empty() {
                                acc.push_str(&text);
                                match content_start {
                                    None => {
                                        if let Some(idx) = acc.find(think_close) {
                                            if idx > reason_emitted {
                                                yield Ok(Event::default().data(reasoning_chunk(&completion_id, created, &model_name, &acc[reason_emitted..idx])));
                                            }
                                            let cs = idx + think_close.len();
                                            let mut lead = cs;
                                            while lead < acc.len() && matches!(acc.as_bytes()[lead], b'\n' | b'\r' | b' ' | b'\t') { lead += 1; }
                                            content_start = Some(lead);
                                            if lead < acc.len() {
                                                yield Ok(Event::default().data(content_chunk(&completion_id, created, &model_name, &acc[lead..acc.len()])));
                                            }
                                            content_emitted = acc.len();
                                        } else {
                                            let overlap = partial_think_overlap(&acc, think_close);
                                            let safe = acc.len() - overlap;
                                            if safe > reason_emitted {
                                                yield Ok(Event::default().data(reasoning_chunk(&completion_id, created, &model_name, &acc[reason_emitted..safe])));
                                                reason_emitted = safe;
                                            }
                                        }
                                    }
                                    Some(cs) => {
                                        // Hold back anything that is, or could become, a tool call.
                                        // Forwarding `<tool_call>` as content makes the harness render
                                        // XML in the chat and never invoke the tool.
                                        let region = &acc[cs..];
                                        let safe_end = match region.find(TOOL_OPEN) {
                                            Some(i) => cs + i,          // a call has started: emit nothing more
                                            None => acc.len() - partial_overlap(region, TOOL_OPEN),
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
            // The call was buffered, not streamed (see the hold-back above). Emit it as one
            // tool_calls delta and flip finish_reason -- that is the flag every harness branches on.
            let (_, done_content) = split_think(&acc, think_open, think_close);
            let parsed = crate::tools::parse(&done_content, req_tools.as_deref());
            if req_tools.is_some() {
                let dump = std::env::var("RUST_INFER_DUMP_TOOLS").is_ok();
                if dump || parsed.tool_calls.is_empty() {
                    eprintln!("[req] raw model output ({} chars): {:?}", done_content.chars().count(),
                              done_content.chars().take(1200).collect::<String>());
                }
            }
            if !parsed.tool_calls.is_empty() {
                // Log the ARGUMENTS, not just the names — see the note on the non-streaming path.
                // Agent harnesses stream, so this is the branch that actually gets used, and it was the
                // one printing a bare `tool_calls 1: ["write"]` while a file silently failed to appear.
                for t in &parsed.tool_calls {
                    eprintln!("[req] tool_call  {} {}({})", t.id, t.function.name, t.function.arguments);
                }
                yield Ok(Event::default().data(tool_calls_chunk(&completion_id, created, &model_name, &parsed.tool_calls)));
                finish = "tool_calls".to_string();
            }
            let final_chunk = format!("{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{}\"}}]}}",
                completion_id, created, model_name, finish);
            yield Ok(Event::default().data(final_chunk));
            let dt = t0.elapsed().as_secs_f32();
            eprintln!("[req] done   tok={} ({:.1} tok/s wall) finish={} stop_hit={}", n, if dt>1e-6 {n as f32/dt} else {0.0}, finish, stop_hit);
        };
        Sse::new(stream).into_response()
    } else {
        eprintln!("[req] sync   prompt_tokens={} max_tokens={} stop={:?}", prompt_len, req_max, req.stop);
        let t0 = std::time::Instant::now();
        let mut tokens = Vec::new();
        let mut finish = "length".to_string();
        while let Some(ev) = rx.recv().await {
            match ev {
                TokEvent::Tok(t) => {
                    tokens.push(t);
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
        let mut text = state.tokenizer.decode(&tokens, true).unwrap_or_default();
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
        let (content, tool_calls, finish) = if parsed.tool_calls.is_empty() {
            (Some(content), None, finish)
        } else {
            // Log the ARGUMENTS, not just the names. When opencode reported a write as successful and
            // no file appeared, the log said `tool_calls 1: ["write"]` — which is exactly enough to
            // know a tool was called and not nearly enough to know what it was told to do. The path the
            // model chose is the whole question.
            for t in &parsed.tool_calls {
                eprintln!("[req] tool_call  {} {}({})", t.id, t.function.name, t.function.arguments);
            }
            let c = if parsed.content.is_empty() { None } else { Some(parsed.content) };
            (c, Some(parsed.tool_calls), "tool_calls".to_string())
        };

        let response = ChatCompletionResponse {
            id: completion_id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: req.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant".to_string(), content,
                    reasoning_content: reasoning, tool_calls,
                },
                finish_reason: finish,
            }],
            usage: Usage {
                prompt_tokens: prompt_len,
                completion_tokens: tokens.len(),
                total_tokens: prompt_len + tokens.len(),
            },
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
        .with_state(state)
}
