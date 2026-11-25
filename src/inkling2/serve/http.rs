//! The Inkling-2 HTTP surface: `/v1/chat/completions` (streamed and not), `/v1/models`, `/v1/tokenize`, `/v1/detokenize`, `/health`, `/metrics`.
//!
//! Why not the shared `server::create_router` handler: that handler splits `<think>` text and finds `<tool_call` markup in the decoded
//! string; this model frames reasoning and calls with control TOKENS (`<|content_thinking|>`, `<|content_invoke_tool_json|>`), which
//! decode to nothing. Serving it through the shared parser would return the control-free text as one answer: a reply that looks fine
//! and has silently lost its reasoning and its calls. So this family parses by token id (`chat::parse`) behind its own router, as
//! DeepSeek-V4.1 does, and reuses the shared listener, boot report, logger and metrics.

use std::convert::Infallible;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use serde_json::Value;
use tokio::sync::mpsc;

use super::super::chat::request::{self, ApiError, Defaults, Rendered};
use super::forward::{Forward, ForwardError, Job};
use super::metrics as m;
use super::output::{self as o, Assembler, Delta};
use crate::tokenizer::QwenTokenizer;

/// What the boot resolved.
pub struct Settings {
    pub model_name: String,
    pub max_seq_len: usize,
    pub default_max_tokens: Option<usize>,
    /// the pack's `generation_config.json` (temperature 1.0; top_p / top_k when it has them)
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub effort: Option<String>,
    /// `--image-max-tokens`: the per-image LM token cap (above it an image is downscaled)
    pub image_max_tokens: usize,
}

pub struct App {
    pub settings: Settings,
    pub tok: Arc<QwenTokenizer>,
    pub forward: Arc<dyn Forward>,
    pub created: i64,
}

pub fn router(app: Arc<App>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/models", get(models))
        .route("/v1/models/*id", get(model_one))
        .route("/v1/tokenize", post(tokenize))
        .route("/v1/detokenize", post(detokenize))
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .layer(tower_http::cors::CorsLayer::new().allow_origin(tower_http::cors::Any).allow_methods(tower_http::cors::Any).allow_headers(tower_http::cors::Any))
        .layer(axum::extract::DefaultBodyLimit::max(128 * 1024 * 1024))
        .with_state(app)
}

fn json_response(status: u16, body: String) -> Response {
    let mut r = Response::new(Body::from(body));
    *r.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    r
}
fn api_error(e: &ApiError) -> Response { json_response(e.status, e.json()) }
fn model_json(a: &App) -> Value { serde_json::json!({"id": a.settings.model_name, "object": "model", "created": a.created, "owned_by": "velogb10", "max_model_len": a.settings.max_seq_len}) }

async fn models(State(a): State<Arc<App>>) -> Response { json_response(200, serde_json::json!({"object": "list", "data": [model_json(&a)]}).to_string()) }
async fn model_one(State(a): State<Arc<App>>, axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    if id == a.settings.model_name { return json_response(200, model_json(&a).to_string()); }
    let mut r = Response::new(Body::from(format!("Model '{}' not found. Available: {}", id, a.settings.model_name)));
    *r.status_mut() = StatusCode::NOT_FOUND;
    r
}

/// The shared server's error body for the tokenizer routes.
fn tok_bad(msg: String) -> Response {
    json_response(400, serde_json::json!({"error": {"message": msg, "type": "invalid_request_error", "code": "invalid_request_error"}}).to_string())
}

/// The `model` field the tokenizer routes require: it must name the served model.
fn tok_model(a: &App, v: &Value) -> Result<String, Response> {
    let Some(model) = v.get("model").and_then(|x| x.as_str()) else {
        return Err(tok_bad("'model' is required and must name the served model (see GET /v1/models)".into()));
    };
    if model != a.settings.model_name {
        return Err(tok_bad(format!("model '{model}' not found. Available: {}", a.settings.model_name)));
    }
    Ok(model.to_string())
}

/// `POST /v1/tokenize`, matching the shared server and DeepSeek-V4.1: `prompt` (text or token ids) or `messages`
/// (rendered by Inkling-2's chat template, with `tools` / `reasoning_effort` / `chat_template_kwargs` and image
/// expansion when vision is enabled, so `count` equals the `prompt_tokens` of the same chat request);
/// `add_special_tokens` (default true), `truncate_prompt_tokens` (keeps the LAST n). Answers `{tokens, count, max_model_len}`.
async fn tokenize(State(a): State<Arc<App>>, body: String) -> Response {
    let v: Value = match serde_json::from_str(&body) { Ok(v) => v, Err(e) => return tok_bad(format!("body is not JSON: {e}")) };
    if let Err(r) = tok_model(&a, &v) { return r; }
    let add_special = match v.get("add_special_tokens") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(b)) => *b,
        Some(_) => return tok_bad("'add_special_tokens' must be a boolean".into()),
    };
    let truncate: Option<usize> = match v.get("truncate_prompt_tokens") {
        None | Some(Value::Null) => None,
        Some(Value::Number(n)) => match n.as_u64() { Some(x @ 1..) => Some(x as usize), _ => return tok_bad("'truncate_prompt_tokens' must be an integer >= 1".into()) },
        Some(_) => return tok_bad("'truncate_prompt_tokens' must be an integer >= 1".into()),
    };
    let has_prompt = v.get("prompt").map_or(false, |x| !x.is_null());
    let has_messages = v.get("messages").map_or(false, |x| !x.is_null());
    if has_prompt && has_messages { return tok_bad("provide exactly one of 'prompt' or 'messages'".into()); }
    let a2 = a.clone();
    let mut tokens: Vec<u32> = if has_messages {
        let r = tokio::task::spawn_blocking(move || -> Result<Vec<u32>, ApiError> {
            // max_seq_len: 0 defers the context-length check until after truncate_prompt_tokens below.
            let d = Defaults { effort: a2.settings.effort.clone(), max_seq_len: 0 };
            let mut rendered = request::render(&v, &d, &a2.tok)?;
            if rendered.n_images > 0 && a2.forward.vision() {
                crate::inkling2::vision::serve::prepare(&mut rendered, a2.settings.image_max_tokens, 0)?;
            }
            Ok(rendered.prompt_ids)
        }).await;
        match r {
            Ok(Ok(ids)) => ids,
            Ok(Err(e)) => return api_error(&e),
            Err(_) => return tok_bad("tokenization panicked".into()),
        }
    } else {
        match v.get("prompt") {
            Some(Value::String(t)) if t.is_empty() => Vec::new(),
            Some(Value::String(t)) => {
                let text = t.clone();
                match tokio::task::spawn_blocking(move || a2.tok.encode(&text, add_special)).await {
                    Ok(Ok(x)) => x,
                    Ok(Err(e)) => return tok_bad(format!("tokenization failed: {e}")),
                    Err(_) => return tok_bad("tokenization panicked".into()),
                }
            }
            Some(Value::Array(items)) => {
                let vocab = a.tok.vocab_size();
                let mut ids = Vec::with_capacity(items.len());
                for it in items {
                    match it.as_u64() {
                        Some(id) if (id as usize) < vocab => ids.push(id as u32),
                        _ => return tok_bad(format!("'prompt' ids must be integers in [0, {vocab}) (got {it})")),
                    }
                }
                ids
            }
            _ => return tok_bad("'prompt' must be a string or a list of token ids, or use 'messages'".into()),
        }
    };
    if let Some(n) = truncate { if tokens.len() > n { tokens.drain(..tokens.len() - n); } }
    if tokens.len() >= a.settings.max_seq_len {
        return json_response(400, serde_json::json!({"error": {
            "message": format!("This model's maximum context length is {} tokens, but the prompt came to {} tokens. Shorten the input, or pass truncate_prompt_tokens to keep the last n tokens.", a.settings.max_seq_len, tokens.len()),
            "type": "invalid_request_error", "code": "context_length_exceeded"
        }}).to_string());
    }
    let count = tokens.len();
    json_response(200, serde_json::json!({"tokens": tokens, "count": count, "max_model_len": a.settings.max_seq_len}).to_string())
}

/// `POST /v1/detokenize`: `{model, tokens, skip_special_tokens?}` -> `{model, prompt}`.
async fn detokenize(State(a): State<Arc<App>>, body: String) -> Response {
    let v: Value = match serde_json::from_str(&body) { Ok(v) => v, Err(e) => return tok_bad(format!("body is not JSON: {e}")) };
    let model = match tok_model(&a, &v) { Ok(m) => m, Err(r) => return r };
    let Some(items) = v.get("tokens").and_then(|x| x.as_array()) else { return tok_bad("'tokens' is required and must be a list of token ids".into()) };
    let vocab = a.tok.vocab_size();
    let mut ids: Vec<u32> = Vec::with_capacity(items.len());
    for it in items {
        match it.as_u64() {
            Some(id) if (id as usize) < vocab => ids.push(id as u32),
            _ => return tok_bad(format!("'tokens' ids must be integers in [0, {vocab}) (got {it})")),
        }
    }
    let skip = match v.get("skip_special_tokens") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return tok_bad("'skip_special_tokens' must be a boolean".into()),
    };
    let a2 = a.clone();
    match tokio::task::spawn_blocking(move || a2.tok.decode(&ids, skip)).await {
        Ok(Ok(p)) => json_response(200, serde_json::json!({"model": model, "prompt": p}).to_string()),
        Ok(Err(e)) => tok_bad(format!("detokenization failed: {e}")),
        Err(_) => tok_bad("detokenization panicked".into()),
    }
}
async fn health() -> Response { json_response(200, "{\"status\":\"ok\"}".into()) }
async fn metrics() -> Response {
    let mut r = Response::new(Body::from(m::render_all(true)));
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; version=0.0.4"));
    r
}

fn num(v: Option<&Value>, what: &str) -> Result<Option<f64>, ApiError> {
    match v { None | Some(Value::Null) => Ok(None), Some(x) => x.as_f64().map(Some).ok_or_else(|| ApiError::bad(format!("'{what}' must be a number"))) }
}

struct Plan { rendered: Rendered, max_new: usize, temperature: f32, top_p: f32, top_k: usize, seed: Option<u64>, stream: bool, include_usage: bool }

fn plan(body: &str, a: &App) -> Result<Plan, ApiError> {
    let v: Value = serde_json::from_str(body).map_err(|e| ApiError::bad(format!("the request body is not valid JSON ({e})")))?;
    if !v.is_object() { return Err(ApiError::bad("the request body must be a JSON object")); }
    let d = Defaults { effort: a.settings.effort.clone(), max_seq_len: a.settings.max_seq_len };
    let mut rendered = request::render(&v, &d, &a.tok)?;
    if a.forward.vision() {
        crate::inkling2::vision::serve::prepare(&mut rendered, a.settings.image_max_tokens, a.settings.max_seq_len)?;
    } else if rendered.n_images > 0 && a.forward.name() != "not-built" {
        // a model without its vision tower must not answer about an image it never saw
        return Err(ApiError::bad("this server's vision tower did not load at boot (see its log): image requests cannot be served; send text only").code("vision_unavailable"));
    }
    let room = a.settings.max_seq_len.saturating_sub(rendered.prompt_ids.len());
    let asked = num(v.get("max_completion_tokens"), "max_completion_tokens")?.or(num(v.get("max_tokens"), "max_tokens")?).map(|x| x as usize)
        .or(a.settings.default_max_tokens).unwrap_or(usize::MAX);
    if asked == 0 { return Err(ApiError::bad("'max_tokens' must be at least 1")); }
    let temperature = num(v.get("temperature"), "temperature")?.map_or(a.settings.temperature, |x| x as f32);
    let top_p = num(v.get("top_p"), "top_p")?.map_or(a.settings.top_p, |x| x as f32);
    if !(temperature >= 0.0) || !(top_p > 0.0 && top_p <= 1.0) { return Err(ApiError::bad("'temperature' must be >= 0 and 'top_p' in (0, 1]")); }
    let top_k = num(v.get("top_k"), "top_k")?.map_or(a.settings.top_k, |x| if x <= 0.0 { 0 } else { x as usize });
    let seed = num(v.get("seed"), "seed")?.map(|x| x as u64);
    let stream = matches!(v.get("stream"), Some(Value::Bool(true)));
    let include_usage = v.get("stream_options").and_then(|o| o.get("include_usage")).and_then(|x| x.as_bool()).unwrap_or(true);
    Ok(Plan { rendered, max_new: asked.min(room), temperature, top_p, top_k, seed, stream, include_usage })
}

fn count_request(p: &Plan) {
    m::add(&m::PROMPT_TOKENS, p.rendered.prompt_ids.len() as u64);
    m::add(&m::IMAGES, p.rendered.n_images as u64);
    if p.rendered.tools_offered { m::inc(&m::TOOL_REQS); }
    if p.rendered.choice_unenforced { m::inc(&m::CHOICE_UNENFORCED); }
    let e = &p.rendered.effort;
    m::inc(match (&e.value, &e.mapped) { (None, _) => &m::EFFORT_DEFAULT, (_, Some(_)) => &m::EFFORT_MAPPED, (Some(serde_json::Value::Number(_)), _) => &m::EFFORT_NUMERIC, _ => &m::EFFORT_NAMED });
    if let Some(line) = &e.mapped { crate::dprintln!("[inkling2] {line}"); }
}

fn forward_error(e: &anyhow::Error) -> Response {
    match e.downcast_ref::<ForwardError>() {
        Some(fe) if fe.code == "model_not_built" => {
            m::inc(&m::REQ_NOT_BUILT);
            json_response(501, serde_json::json!({"error": {"message": fe.message, "type": "not_implemented_error", "param": null, "code": fe.code}}).to_string())
        }
        _ => { m::inc(&m::REQ_ENGINE_ERR); json_response(500, serde_json::json!({"error": {"message": format!("{e:#}"), "type": "server_error", "param": null, "code": null}}).to_string()) }
    }
}

enum Msg { Frame(String), Err(anyhow::Error), End }

async fn chat(State(a): State<Arc<App>>, body: String) -> Response {
    let a2 = a.clone();
    let p = match tokio::task::spawn_blocking(move || plan(&body, &a2)).await {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => { m::inc(&m::REQ_BAD); return api_error(&e); }
        Err(_) => { m::inc(&m::REQ_BAD); return api_error(&ApiError::bad("request preparation panicked")); }
    };
    count_request(&p);
    let (cid, created, model) = (format!("chatcmpl-{:x}", rand_id()), a.created, a.settings.model_name.clone());
    let (tx, mut rx) = mpsc::unbounded_channel::<Msg>();
    let stream = p.stream;
    let prompt_n = p.rendered.prompt_ids.len();
    let a3 = a.clone();
    let (cid2, model2) = (cid.clone(), model.clone());
    tokio::task::spawn_blocking(move || run(a3, p, tx, cid2, created, model2));
    if !stream {
        let mut text = None;
        while let Some(msg) = rx.recv().await {
            match msg { Msg::Frame(s) => text = Some(s), Msg::Err(e) => return forward_error(&e), Msg::End => break }
        }
        m::inc(&m::NON_STREAMED);
        return json_response(200, text.unwrap_or_default());
    }
    // streaming: an error before the first frame is an HTTP error, never an empty stream
    let first = match rx.recv().await { Some(Msg::Err(e)) => return forward_error(&e), Some(x) => x, None => Msg::End };
    m::inc(&m::STREAMED);
    let _ = (cid, model, prompt_n);
    let (btx, brx) = mpsc::channel::<Result<String, Infallible>>(256);
    tokio::spawn(async move {
        let mut msg = first;
        loop {
            match msg { Msg::Frame(s) => { if btx.send(Ok(s)).await.is_err() { break; } } Msg::Err(_) | Msg::End => break }
            msg = match rx.recv().await { Some(x) => x, None => break };
        }
    });
    let mut r = Response::new(Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(brx)));
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    r
}

fn rand_id() -> u64 {
    let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    n ^ (std::process::id() as u64).rotate_left(32)
}

/// The blocking part of a request: the forward drives the assembler; frames go to the response.
fn run(a: Arc<App>, p: Plan, tx: mpsc::UnboundedSender<Msg>, cid: String, created: i64, model: String) {
    let mut asm = Assembler::new(&a.tok, p.rendered.catalogue.clone(), p.rendered.parse.clone());
    let job = Job::from_rendered(&p.rendered, p.max_new, p.temperature, p.top_p, p.top_k, p.seed);
    let prompt_n = p.rendered.prompt_ids.len();
    let mut all: Vec<Delta> = Vec::new();
    let mut sent_role = false;
    let mut bad: Option<anyhow::Error> = None;
    let streaming = p.stream;
    let mut emit = |id: u32| -> bool {
        match asm.push(id) {
            Err(e) => { bad = Some(e.into()); false }
            Ok(ds) => {
                for d in ds {
                    if streaming {
                        if !sent_role { sent_role = true; if tx.send(Msg::Frame(o::role_chunk(&cid, created, &model))).is_err() { return false; } }
                        if tx.send(Msg::Frame(o::delta_chunk(&cid, created, &model, &d))).is_err() { return false; }
                    } else { all.push(d); }
                }
                !asm.parser_done()
            }
        }
    };
    let res = a.forward.generate(&job, &mut emit);
    drop(emit);
    if let Some(e) = bad { m::inc(&m::REQ_ENGINE_ERR); asm.abort(); let _ = tx.send(Msg::Err(e)); return; }
    let end = match res { Ok(e) => e, Err(e) => { let _ = tx.send(Msg::Err(e)); return; } };
    let tail = asm.finish(end);
    o::count_stats(asm.stats());
    let completion = asm.tokens;
    m::add(&m::GEN_TOKENS, completion as u64);
    let calls = asm.calls();
    m::inc(match o::finish_reason(end, calls) { "tool_calls" => &m::FINISH_TOOLS, "length" => &m::FINISH_LENGTH, _ => &m::FINISH_STOP });
    m::inc(&m::REQ_OK);
    if streaming {
        if !sent_role { let _ = tx.send(Msg::Frame(o::role_chunk(&cid, created, &model))); }
        for d in tail { let _ = tx.send(Msg::Frame(o::delta_chunk(&cid, created, &model, &d))); }
        let _ = tx.send(Msg::Frame(o::finish_chunk(&cid, created, &model, end, calls)));
        if p.include_usage { let _ = tx.send(Msg::Frame(o::usage_chunk(&cid, created, &model, prompt_n, completion))); }
        let _ = tx.send(Msg::Frame(o::DONE.to_string()));
    } else {
        all.extend(tail);
        let _ = tx.send(Msg::Frame(o::chat_json(&cid, created, &model, &all, end, prompt_n, completion).to_string()));
    }
    let _ = tx.send(Msg::End);
}
