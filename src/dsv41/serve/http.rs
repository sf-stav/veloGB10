//! The dsv41 HTTP surface: `/v1/chat/completions` (streamed and not), `/v1/models`, `/health`, `/metrics`.
//!
//! Streaming hygiene (UX row 11): no role chunk before the first token exists (or the finish chunk when none came), SSE
//! keep-alives on a silent stream (`--sse-keepalive`, `--sse-keepalive-frame`), a client stop string never streamed, a
//! usage chunk, `--max-waiting` (503), `--stream-backlog-events` (cancel a client that fell that far behind), and a
//! lane freed the moment the client goes away.

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use tokio::sync::mpsc;

use super::api::{prepare, prepare_completion, ApiError, Prepared, ServeSettings};
use super::engine::{EngineHandle, Finish, Job, Out};
use super::metrics as m;
use super::output::{self as o, Assembler, Delta, Usage};
use crate::tokenizer::QwenTokenizer;

pub struct HttpOpts {
    /// `--max-waiting` (0 = unlimited)
    pub max_waiting: usize,
    /// `--stream-backlog-events` (0 = unlimited)
    pub backlog: usize,
    /// `--sse-keepalive` seconds (0 = off)
    pub keepalive_secs: u64,
    /// `--sse-keepalive-frame data`
    pub keepalive_data: bool,
}

pub struct App {
    pub settings: ServeSettings,
    pub tok: Arc<QwenTokenizer>,
    pub engine: EngineHandle,
    pub http: HttpOpts,
    pub max_model_len: usize,
    pub created: i64,
}

pub fn router(app: Arc<App>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
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

fn log_bad_request(route: &str, code: &str, message: &str) {
    let clean: String = message.chars().map(|c| if c == '\n' || c == '\r' { ' ' } else { c }).take(240).collect();
    crate::logq::send(false, format!("[req] 400 {route} ({code}): {clean}"));
}

fn json_response(status: u16, body: String) -> Response {
    let mut r = Response::new(Body::from(body));
    *r.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    r
}

fn api_error(e: &ApiError) -> Response { json_response(e.status, e.json()) }

fn model_json(app: &App) -> serde_json::Value {
    serde_json::json!({"id": app.settings.model_name, "object": "model", "created": app.created, "owned_by": "velogb10", "max_model_len": app.max_model_len})
}

async fn models(State(app): State<Arc<App>>) -> Response {
    json_response(200, serde_json::json!({"object": "list", "data": [model_json(&app)]}).to_string())
}

/// `GET /v1/models/{id}` (the id may hold slashes, as a hub id does): the served model, else 404 naming it.
async fn model_one(State(app): State<Arc<App>>, axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    if id == app.settings.model_name { return json_response(200, model_json(&app).to_string()); }
    let mut r = Response::new(Body::from(format!("Model '{}' not found. Available: {}", id, app.settings.model_name)));
    *r.status_mut() = StatusCode::NOT_FOUND;
    r
}

/// The shared server's error body for the tokenizer routes.
fn tok_bad(msg: String) -> Response {
    json_response(400, serde_json::json!({"error": {"message": msg, "type": "invalid_request_error", "code": "invalid_request_error"}}).to_string())
}

/// The `model` field the tokenizer routes require: it must name the served model.
fn tok_model(app: &App, v: &serde_json::Value) -> Result<String, Response> {
    let Some(model) = v.get("model").and_then(|x| x.as_str()) else { return Err(tok_bad("'model' is required and must name the served model (see GET /v1/models)".into())) };
    if model != app.settings.model_name { return Err(tok_bad(format!("model '{model}' not found. Available: {}", app.settings.model_name))); }
    Ok(model.to_string())
}

/// `POST /v1/tokenize`, as the shared server's: `prompt` (text or token ids) or `messages` (rendered by DeepSeek's chat encoder, with
/// `tools` / `thinking` as a chat request, so `count` equals the prompt_tokens of the same chat request); `add_special_tokens`
/// (default true), `truncate_prompt_tokens` (keeps the LAST n). Answers {tokens, count, max_model_len}.
async fn tokenize(State(app): State<Arc<App>>, body: String) -> Response {
    let v: serde_json::Value = match serde_json::from_str(&body) { Ok(v) => v, Err(e) => return tok_bad(format!("body is not JSON: {e}")) };
    if let Err(r) = tok_model(&app, &v) { return r; }
    let add_special = match v.get("add_special_tokens") {
        None | Some(serde_json::Value::Null) => true,
        Some(serde_json::Value::Bool(b)) => *b,
        Some(_) => return tok_bad("'add_special_tokens' must be a boolean".into()),
    };
    let truncate: Option<usize> = match v.get("truncate_prompt_tokens") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::Number(n)) => match n.as_u64() { Some(x @ 1..) => Some(x as usize), _ => return tok_bad("'truncate_prompt_tokens' must be an integer >= 1".into()) },
        Some(_) => return tok_bad("'truncate_prompt_tokens' must be an integer >= 1".into()),
    };
    let has_prompt = v.get("prompt").map_or(false, |x| !x.is_null());
    let has_messages = v.get("messages").map_or(false, |x| !x.is_null());
    if has_prompt && has_messages { return tok_bad("provide exactly one of 'prompt' or 'messages'".into()); }
    let a2 = app.clone();
    let mut tokens: Vec<u32> = if has_messages {
        // the chat request's own preparation (the prompt-length check is the same one a chat request meets, below)
        let mut chat = v.clone();
        if let Some(o) = chat.as_object_mut() { o.remove("truncate_prompt_tokens"); }
        let r = tokio::task::spawn_blocking(move || prepare(&chat.to_string(), &a2.settings, &a2.tok)).await;
        match r {
            Ok(Ok(p)) => p.prompt_ids,
            Ok(Err(e)) => {
                log_bad_request("/v1/tokenize", e.code.unwrap_or(e.kind), &e.message);
                return api_error(&e);
            }
            Err(_) => return tok_bad("tokenization panicked".into()),
        }
    } else {
        match v.get("prompt") {
            Some(serde_json::Value::String(t)) if t.is_empty() => Vec::new(),
            Some(serde_json::Value::String(t)) => match a2.tok.encode(t, add_special) { Ok(x) => x, Err(e) => return tok_bad(format!("tokenization failed: {e}")) },
            Some(serde_json::Value::Array(items)) => {
                let vocab = app.tok.vocab_size();
                let mut ids = Vec::with_capacity(items.len());
                for it in items { match it.as_u64() { Some(id) if (id as usize) < vocab => ids.push(id as u32), _ => return tok_bad(format!("'prompt' ids must be integers in [0, {vocab}) (got {it})")) } }
                ids
            }
            _ => return tok_bad("'prompt' must be a string or a list of token ids, or use 'messages'".into()),
        }
    };
    if let Some(n) = truncate { if tokens.len() > n { tokens.drain(..tokens.len() - n); } }
    if tokens.len() >= app.max_model_len {
        return json_response(400, serde_json::json!({"error": {"message": format!("This model's maximum context length is {} tokens, but the prompt came to {} tokens. Shorten the input, or pass truncate_prompt_tokens to keep the last n tokens.", app.max_model_len, tokens.len()),
            "type": "invalid_request_error", "code": "context_length_exceeded"}}).to_string());
    }
    let count = tokens.len();
    json_response(200, serde_json::json!({"tokens": tokens, "count": count, "max_model_len": app.max_model_len}).to_string())
}

/// `POST /v1/detokenize`: `{model, tokens, skip_special_tokens?}` -> `{model, prompt}`.
async fn detokenize(State(app): State<Arc<App>>, body: String) -> Response {
    let v: serde_json::Value = match serde_json::from_str(&body) { Ok(v) => v, Err(e) => return tok_bad(format!("body is not JSON: {e}")) };
    let model = match tok_model(&app, &v) { Ok(m) => m, Err(r) => return r };
    let Some(items) = v.get("tokens").and_then(|x| x.as_array()) else { return tok_bad("'tokens' is required and must be a list of token ids".into()) };
    let vocab = app.tok.vocab_size();
    let mut ids: Vec<u32> = Vec::with_capacity(items.len());
    for it in items { match it.as_u64() { Some(id) if (id as usize) < vocab => ids.push(id as u32), _ => return tok_bad(format!("'tokens' ids must be integers in [0, {vocab}) (got {it})")) } }
    let skip = match v.get("skip_special_tokens") {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Bool(b)) => *b,
        Some(_) => return tok_bad("'skip_special_tokens' must be a boolean".into()),
    };
    match app.tok.decode(&ids, skip) {
        Ok(p) => json_response(200, serde_json::json!({"model": model, "prompt": p}).to_string()),
        Err(e) => tok_bad(format!("detokenization failed: {e}")),
    }
}

async fn health(State(app): State<Arc<App>>) -> Response {
    if app.engine.alive() { json_response(200, "{\"status\":\"ok\"}".into()) }
    else { json_response(503, "{\"status\":\"engine_dead\"}".into()) }
}

async fn metrics(State(app): State<Arc<App>>) -> Response {
    let mut r = Response::new(Body::from(m::render_all(app.engine.alive())));
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; version=0.0.4"));
    r
}

/// Cancels the request's lane when the handler's future is dropped (the client went away).
struct CancelGuard(Arc<AtomicBool>, bool);
impl Drop for CancelGuard { fn drop(&mut self) { if self.1 { self.0.store(true, Ordering::SeqCst); } } }

async fn chat(State(app): State<Arc<App>>, body: String) -> Response {
    // leg HT1: a parked TP server starts waking its nodes now, under this request's parsing and tokenization (no-op otherwise)
    crate::dsv41::tp::idle::prewake();
    let prep_guard = app.engine.prep_guard();
    // ---- parse + render + tokenize off the async threads (a long prompt tokenizes for a while) ----
    let a2 = app.clone();
    let prepared = tokio::task::spawn_blocking(move || prepare(&body, &a2.settings, &a2.tok)).await;
    let p = match prepared {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            m::inc(&m::REQ_BAD);
            log_bad_request("/v1/chat/completions", e.code.unwrap_or(e.kind), &e.message);
            return api_error(&e);
        }
        Err(_) => return api_error(&ApiError { status: 500, kind: "server_error", code: None, message: "request preparation panicked".into() }),
    };
    serve_prepared(app, p, prep_guard).await
}

/// Admit a prepared request (chat or completion) and answer it, streamed or whole.
async fn serve_prepared(app: Arc<App>, p: Prepared, prep_guard: super::engine::PrepGuard) -> Response {
    if !app.engine.alive() {
        return api_error(&ApiError { status: 503, kind: "server_error", code: Some("engine_dead"), message: "the engine is not running".into() });
    }
    if app.http.max_waiting > 0 && app.engine.waiting() >= app.http.max_waiting {
        m::inc(&m::REQ_BUSY);
        let mut r = api_error(&ApiError { status: 503, kind: "server_error", code: Some("busy"), message: format!("{} requests are already waiting for a lane (--max-waiting)", app.http.max_waiting) });
        r.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from_static("5"));
        return r;
    }
    // ---- counters of what this request asks for ----
    if p.sampling.greedy() { m::inc(&m::GREEDY_REQS) } else { m::inc(&m::SAMPLED_REQS) }
    if !p.seeded { if app.settings.seedless_mode == "prompt-hash" { m::inc(&m::SEEDLESS_HASH) } else { m::inc(&m::SEEDLESS_RANDOM) } }
    if p.thinking { m::inc(&m::THINK_ON) } else { m::inc(&m::THINK_OFF) }
    if p.tools_offered { m::inc(&m::TOOL_REQS) }
    if p.wants_logprobs { m::inc(&m::LOGPROBS_UNSUPPORTED) }
    if p.unenforced_format { m::inc(&m::SCHEMA_UNENFORCED) }
    if p.stream { m::inc(&m::STREAMED) } else { m::inc(&m::NON_STREAMED) }
    if p.min_tokens > 0 { m::inc(&m::MIN_TOKENS_REQS) }
    if p.ignore_eos { m::inc(&m::IGNORE_EOS_REQS) }

    let cancel = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::unbounded_channel::<Out>();
    let job = Job {
        prompt: p.prompt_ids.clone(), max_new: p.max_new, min_tokens: p.min_tokens, sampling: p.sampling.clone(), seed: p.seed, ignore_eos: p.ignore_eos,
        starts_in_think: p.thinking, think_budget: p.think_budget, cancel: cancel.clone(), tx, score: false,
        vl_types: p.vl_types.clone(), vl_images: p.vl_images.clone(), image_spans: p.image_spans.clone(),
    };
    drop(prep_guard);
    if app.engine.submit(job).is_err() {
        return api_error(&ApiError { status: 503, kind: "server_error", code: Some("engine_dead"), message: "the engine is not running".into() });
    }
    let t0 = Instant::now();
    if p.stream { stream_reply(app, p, rx, cancel, t0) } else { full_reply(app, p, rx, cancel, t0).await }
}

/// `POST /v1/completions`. Generation (the default): raw text or token ids in, `text_completion` out, streamed or not, `max_tokens`
/// 16 unless given. Prompt SCORING (GA's G1 `--source api-echo`): `{prompt: [ids] | "text", echo: true, max_tokens: 0, logprobs: N}` ->
/// the OpenAI text-completion shape with `logprobs.token_logprobs` = [null, log p(t1 | t0), ...] from an EXACT prefill
/// (`Forward::score`).
async fn completions(State(app): State<Arc<App>>, body: String) -> Response {
    crate::dsv41::tp::idle::prewake();   // leg HT1 (see `chat`)
    let bad = |msg: String| {
        m::inc(&m::REQ_BAD);
        log_bad_request("/v1/completions", "invalid_request_error", &msg);
        api_error(&ApiError::bad(msg))
    };
    let v: serde_json::Value = match serde_json::from_str(&body) { Ok(v) => v, Err(e) => return bad(format!("body is not JSON: {e}")) };
    let echo = v.get("echo").and_then(|x| x.as_bool()).unwrap_or(false);
    let max_tokens = v.get("max_tokens").and_then(|x| x.as_u64()).unwrap_or(16);
    if !echo || max_tokens != 0 {
        // generation: raw text in, raw text out (no chat encoding, no thinking, no tools)
        let prep_guard = app.engine.prep_guard();
        let a2 = app.clone();
        let prepared = tokio::task::spawn_blocking(move || prepare_completion(&body, &a2.settings, &a2.tok)).await;
        return match prepared {
            Ok(Ok(p)) => serve_prepared(app, p, prep_guard).await,
            Ok(Err(e)) => {
                m::inc(&m::REQ_BAD);
                log_bad_request("/v1/completions", e.code.unwrap_or(e.kind), &e.message);
                api_error(&e)
            }
            Err(_) => api_error(&ApiError { status: 500, kind: "server_error", code: None, message: "request preparation panicked".into() }),
        };
    }
    let ids: Vec<u32> = match v.get("prompt") {
        Some(serde_json::Value::Array(a)) => {
            let mut ids = Vec::with_capacity(a.len());
            for x in a { match x.as_u64() { Some(t) if t <= u32::MAX as u64 => ids.push(t as u32), _ => return bad("prompt: token ids (integers) or one string".into()) } }
            ids
        }
        Some(serde_json::Value::String(s)) => match app.tok.encode(s, false) { Ok(t) => t, Err(e) => return bad(format!("tokenizer: {e:#}")) },
        _ => return bad("prompt: token ids (integers) or one string".into()),
    };
    if ids.len() < 2 { return bad("prompt: at least 2 tokens to score".into()); }
    if ids.len() > app.max_model_len { return bad(format!("prompt: {} tokens > max_model_len {}", ids.len(), app.max_model_len)); }
    if !app.engine.alive() { return api_error(&ApiError { status: 503, kind: "server_error", code: Some("engine_dead"), message: "the engine is not running".into() }); }
    let cancel = Arc::new(AtomicBool::new(false));
    let _guard = CancelGuard(cancel.clone(), true);
    let (tx, mut rx) = mpsc::unbounded_channel::<Out>();
    let job = Job { prompt: ids.clone(), max_new: 0, min_tokens: 0, sampling: Default::default(), seed: 0, ignore_eos: true, starts_in_think: false, think_budget: None,
                    cancel: cancel.clone(), tx, score: true, vl_types: Vec::new(), vl_images: Vec::new(), image_spans: Vec::new() };
    if app.engine.submit(job).is_err() { return api_error(&ApiError { status: 503, kind: "server_error", code: Some("engine_dead"), message: "the engine is not running".into() }); }
    let mut lps = None;
    while let Some(ev) = rx.recv().await {
        match ev {
            Out::Scores(v) => lps = Some(v),
            Out::Done { .. } => break,
            Out::Failed(e) => return api_error(&ApiError { status: 500, kind: "server_error", code: Some("engine_error"), message: e }),
            _ => {}
        }
    }
    let Some(lps) = lps else { return api_error(&ApiError { status: 500, kind: "server_error", code: Some("engine_error"), message: "no scores".into() }) };
    let tokens: Vec<String> = ids.iter().map(|&t| String::from_utf8_lossy(&app.tok.token_bytes(t, false)).into_owned()).collect();
    let mut off = 0usize;
    let text_offset: Vec<usize> = tokens.iter().map(|s| { let o = off; off += s.len(); o }).collect();
    let token_logprobs: Vec<serde_json::Value> = std::iter::once(serde_json::Value::Null).chain(lps.iter().map(|&x| serde_json::json!(x))).collect();
    let body = serde_json::json!({
        "id": format!("cmpl-score-{}", app.created), "object": "text_completion", "created": app.created, "model": app.settings.model_name,
        "choices": [{"index": 0, "text": tokens.concat(), "finish_reason": "length",
                     "logprobs": {"tokens": tokens, "token_logprobs": token_logprobs, "top_logprobs": serde_json::Value::Null, "text_offset": text_offset}}],
        "usage": {"prompt_tokens": ids.len(), "completion_tokens": 0, "total_tokens": ids.len()}
    });
    json_response(200, body.to_string())
}

/// What every reply tracks while it runs.
struct Reply {
    asm: Assembler,
    call_ids: Vec<String>,
    cached: usize,
    tokens: usize,
    first_tok: Option<Instant>,
    finish: Option<Finish>,
    failed: Option<String>,
    /// `--scrub-tool-secrets`: redacts SERVED text only (the model's context and the parser's input are untouched)
    scrub: Option<crate::tool_scrub::Scrubber>,
    /// `--tool-trace`: the reply's raw bytes
    raw: Vec<u8>,
    trace_on: bool,
}

impl Reply {
    fn new(p: &Prepared) -> Self { Self::with_scrub(p, crate::tool_scrub::Mode::from_opt()) }

    fn with_scrub(p: &Prepared, mode: crate::tool_scrub::Mode) -> Self {
        Reply { asm: Assembler::new(p.parse.clone(), p.catalogue.clone(), p.stop_strings.clone()), call_ids: Vec::new(), cached: 0, tokens: 0,
                first_tok: None, finish: None, failed: None,
                scrub: p.scrub_msgs.as_ref().map(|m| crate::tool_scrub::Scrubber::new(mode, m)), raw: Vec::new(), trace_on: p.trace.is_some() }
    }

    /// The deltas as the client may see them: redacted when `--scrub-tool-secrets` is on (held text is released as it settles;
    /// the reasoning tail is flushed when the reasoning ends).
    fn served(&mut self, ds: Vec<Delta>) -> Vec<Delta> {
        let Some(sc) = self.scrub.as_mut() else { return ds };
        let mut out = Vec::with_capacity(ds.len());
        for d in ds {
            match d {
                Delta::Reasoning(t) => { let o = sc.push_reasoning(&t).into_owned(); if !o.is_empty() { out.push(Delta::Reasoning(o)); } }
                Delta::Content(t) => {
                    let r = sc.finish_reasoning(""); if !r.is_empty() { out.push(Delta::Reasoning(r)); }
                    let o = sc.push_content(&t).into_owned(); if !o.is_empty() { out.push(Delta::Content(o)); }
                }
                Delta::Call { index, mut call } => {
                    let r = sc.finish_reasoning(""); if !r.is_empty() { out.push(Delta::Reasoning(r)); }
                    let c = sc.finish_content(""); if !c.is_empty() { out.push(Delta::Content(c)); }
                    call.arguments = sc.scrub_full(&call.arguments).into_owned();
                    out.push(Delta::Call { index, call });
                }
            }
        }
        out
    }

    /// What is still held at the end of the reply.
    fn served_tail(&mut self) -> Vec<Delta> {
        let Some(sc) = self.scrub.as_mut() else { return Vec::new() };
        let mut out = Vec::new();
        let r = sc.finish_reasoning(""); if !r.is_empty() { out.push(Delta::Reasoning(r)); }
        let c = sc.finish_content(""); if !c.is_empty() { out.push(Delta::Content(c)); }
        out
    }

    /// Apply one engine event; returns the reply parts it produced.
    fn on_out(&mut self, out: Out, tok: &QwenTokenizer) -> Vec<Delta> {
        match out {
            Out::Admitted { cached } => { self.cached = cached; Vec::new() }
            Out::Tok(t) => {
                self.tokens += 1;
                self.first_tok.get_or_insert_with(Instant::now);
                if self.trace_on { self.raw.extend_from_slice(&tok.token_bytes(t, false)); }
                let ds = self.asm.push_token(tok, t);
                self.note_calls(&ds);
                ds
            }
            Out::Done { finish, .. } => {
                self.finish = Some(finish);
                let ds = self.asm.finish();
                self.note_calls(&ds);
                ds
            }
            Out::Failed(e) => { self.failed = Some(e); Vec::new() }
            // chat jobs never carry scores (only score jobs from /v1/completions do)
            Out::Scores(_) => Vec::new(),
        }
    }

    fn note_calls(&mut self, ds: &[Delta]) {
        for d in ds { if let Delta::Call { .. } = d { self.call_ids.push(o::new_id("call_")); } }
    }

    fn finish_reason(&self) -> &'static str {
        if self.asm.stopped() { return "stop"; }
        match self.finish {
            Some(Finish::Length) => "length",
            _ if !self.asm.calls.is_empty() => "tool_calls",
            _ => "stop",
        }
    }

    /// Book the finished request into the counters and the request log line.
    fn account(&self, p: &Prepared, t0: Instant, lane_cancelled: bool) {
        let st = &self.asm.parsed().stats;
        for r in &st.rejected { m::tool_reject(r.label()); }
        m::add(&m::TOOL_BREAKS, st.structural_breaks as u64);
        m::add(&m::TOOL_UNTERMINATED, st.unterminated as u64);
        m::add(&m::TOOL_OVER_LIMIT, st.dropped_over_limit as u64);
        m::add(&m::ORPHAN_CLOSERS, self.asm.parsed().warnings.len() as u64);
        m::add(&m::TOOL_CALLS, self.asm.calls.len() as u64);
        if self.asm.stopped() { m::inc(&m::STOP_STRING_HITS); }
        if self.failed.is_some() { return; }
        m::add(&m::PROMPT_TOKENS, p.prompt_ids.len() as u64);
        m::add(&m::GEN_TOKENS, self.tokens as u64);
        let fr = self.finish_reason();
        if let Some(turn) = p.trace.as_ref() {
            let raw = String::from_utf8_lossy(&self.raw).into_owned();
            let after = raw.rsplit_once("</think>").map_or(raw.as_str(), |x| x.1);
            let calls: Vec<crate::tokenizer::ToolCall> = self.asm.calls.iter().zip(&self.call_ids).map(|(c, id)| crate::tokenizer::ToolCall {
                id: id.clone(), kind: "function".into(), function: crate::tokenizer::FunctionCall { name: c.qualified_name(), arguments: c.arguments.clone() } }).collect();
            crate::tool_trace::log(turn, &raw, after, Some(&self.asm.content), &calls, fr);
        }
        match fr { "length" => m::inc(&m::FINISH_LENGTH), "tool_calls" => m::inc(&m::FINISH_TOOLS), _ => m::inc(&m::FINISH_STOP) }
        if !lane_cancelled { m::inc(&m::REQ_OK); }
        let secs = t0.elapsed().as_secs_f64();
        let ttft = self.first_tok.map_or(0.0, |t| t.duration_since(t0).as_secs_f64() * 1e3);
        crate::logq::send(false, format!("[req] prompt={} cached={} gen={} finish={} ttft={:.0}ms {:.1}tok/s{}", p.prompt_ids.len(), self.cached, self.tokens, fr, ttft,
                                         if secs > 0.0 { self.tokens as f64 / secs } else { 0.0 }, if p.thinking { " thinking" } else { "" }));
        crate::drprintln!("[req] detail: model={} seed={}{} temp={} top_p={} top_k={} effort={} think_budget={:?} calls={} rejected_blocks={}",
                          p.model, p.seed, if p.seeded { "" } else { " (seedless)" }, p.sampling.temperature, p.sampling.top_p, p.sampling.top_k, p.effort_desc,
                          p.think_budget, self.asm.calls.len(), st.rejected.len());
    }

    fn usage(&self, p: &Prepared) -> Usage { Usage { prompt: p.prompt_ids.len(), completion: self.tokens, cached: self.cached } }
}

async fn full_reply(app: Arc<App>, p: Prepared, mut rx: mpsc::UnboundedReceiver<Out>, cancel: Arc<AtomicBool>, t0: Instant) -> Response {
    let mut guard = CancelGuard(cancel.clone(), true);
    let mut r = Reply::new(&p);
    while let Some(out) = rx.recv().await {
        let done = matches!(out, Out::Done { .. } | Out::Failed(_));
        r.on_out(out, &app.tok);
        if r.asm.stopped() && !done { cancel.store(true, Ordering::SeqCst); let _ = r.on_out(Out::Done { finish: Finish::Stop, prompt_tokens: 0, completion_tokens: 0 }, &app.tok); break; }
        if done { break; }
    }
    guard.1 = false;
    if let Some(e) = &r.failed {
        r.account(&p, t0, false);
        return api_error(&ApiError { status: 500, kind: "server_error", code: Some("engine_error"), message: e.clone() });
    }
    if r.finish.is_none() {
        return api_error(&ApiError { status: 500, kind: "server_error", code: Some("engine_error"), message: "the engine ended the request without a result".into() });
    }
    r.account(&p, t0, r.asm.stopped());
    if let Some(sc) = r.scrub.as_ref() {
        r.asm.reasoning = sc.scrub_full(&r.asm.reasoning).into_owned();
        r.asm.content = sc.scrub_full(&r.asm.content).into_owned();
        for c in r.asm.calls.iter_mut() { c.arguments = sc.scrub_full(&c.arguments).into_owned(); }
    }
    if p.text_shape {
        let mut text = p.echo_text.clone().unwrap_or_default();
        text.push_str(&r.asm.reasoning);
        text.push_str(&r.asm.content);
        return json_response(200, o::text_completion_json(&o::new_id("cmpl-"), app.created_now(), &p.model, &text, r.finish_reason(), &r.usage(&p)));
    }
    let cid = o::new_id("chatcmpl-");
    let body = o::completion_json(&cid, app.created_now(), &p.model, &r.asm, &r.call_ids, r.finish_reason(), &r.usage(&p));
    json_response(200, body)
}

impl App {
    fn created_now(&self) -> i64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64) }
}

fn sse(frame: &str) -> Bytes { Bytes::from(format!("data: {frame}\n\n")) }

fn stream_reply(app: Arc<App>, p: Prepared, mut rx: mpsc::UnboundedReceiver<Out>, cancel: Arc<AtomicBool>, t0: Instant) -> Response {
    let cap = if app.http.backlog == 0 { 1 << 20 } else { app.http.backlog.min(1 << 20) };
    let (btx, brx) = mpsc::channel::<Result<Bytes, Infallible>>(cap);
    tokio::spawn(async move {
        let text = p.text_shape;
        let cid = o::new_id(if text { "cmpl-" } else { "chatcmpl-" });
        let created = app.created_now();
        let model = p.model.clone();
        let mut r = Reply::new(&p);
        // a completion has no role chunk: it counts as sent, and its first frame follows the first token all the same
        let mut role_sent = text;
        let mut echo_pending = p.echo_text.clone();
        let ka = app.http.keepalive_secs;
        let mut last = Instant::now();
        let mut alive = true;
        // send one frame; a full backlog cancels the request, a closed channel means the client left
        macro_rules! send {
            ($b:expr) => {{
                match btx.try_send(Ok($b)) {
                    Ok(()) => {}
                    Err(mpsc::error::TrySendError::Full(_)) => { m::inc(&m::BACKLOG_CANCELS); alive = false; }
                    Err(mpsc::error::TrySendError::Closed(_)) => { alive = false; }
                }
            }};
        }
        loop {
            if !alive { cancel.store(true, Ordering::SeqCst); m::inc(&m::REQ_CANCELLED); return; }
            let wait = if ka == 0 { Duration::from_secs(86_400) } else { Duration::from_secs(ka).saturating_sub(last.elapsed()) };
            let next = tokio::select! {
                got = tokio::time::timeout(wait, rx.recv()) => match got {
                    Ok(x) => x,
                    Err(_) => {
                        if ka > 0 && app.engine.alive() {
                            m::inc(&m::SSE_KEEPALIVES);
                            if app.http.keepalive_data {
                                send!(sse(&serde_json::json!({"id": cid, "object": if text { "text_completion" } else { "chat.completion.chunk" }, "created": created, "model": model, "choices": []}).to_string()));
                            } else { send!(Bytes::from_static(b": keep-alive\n\n")); }
                        }
                        last = Instant::now();
                        continue;
                    }
                },
                _ = btx.closed() => { alive = false; continue; }
            };
            last = Instant::now();
            let Some(out) = next else { break };
            let is_tok = matches!(out, Out::Tok(_));
            let done = matches!(out, Out::Done { .. } | Out::Failed(_));
            if is_tok && !role_sent { role_sent = true; send!(sse(&o::role_chunk(&cid, created, &model))); }
            if is_tok { if let Some(e) = echo_pending.take() { send!(sse(&o::text_delta_chunk(&cid, created, &model, &e))); } }
            let ds = r.on_out(out, &app.tok);
            let ds = r.served(ds);
            for d in ds {
                if text {
                    match d {
                        Delta::Reasoning(t) | Delta::Content(t) => send!(sse(&o::text_delta_chunk(&cid, created, &model, &t))),
                        Delta::Call { .. } => {}
                    }
                    continue;
                }
                match d {
                    Delta::Reasoning(t) => send!(sse(&o::reasoning_chunk(&cid, created, &model, &t))),
                    Delta::Content(t) => send!(sse(&o::content_chunk(&cid, created, &model, &t))),
                    Delta::Call { index, call } => {
                        let id = r.call_ids.get(index).cloned().unwrap_or_else(|| o::new_id("call_"));
                        send!(sse(&o::call_chunk(&cid, created, &model, index, &id, &call)));
                    }
                }
            }
            if r.asm.stopped() && !done {
                // a stop string matched: end the reply here and free the lane
                cancel.store(true, Ordering::SeqCst);
                let _ = r.on_out(Out::Done { finish: Finish::Stop, prompt_tokens: 0, completion_tokens: 0 }, &app.tok);
                break;
            }
            if done { break; }
        }
        if let Some(e) = r.failed.clone() {
            r.account(&p, t0, false);
            send!(sse(&serde_json::json!({"error": {"message": e, "type": "server_error", "code": "engine_error"}}).to_string()));
            send!(Bytes::from_static(b"data: [DONE]\n\n"));
            return;
        }
        if !role_sent { send!(sse(&o::role_chunk(&cid, created, &model))); }
        if let Some(e) = echo_pending.take() { send!(sse(&o::text_delta_chunk(&cid, created, &model, &e))); }
        for d in r.served_tail() {
            match d {
                Delta::Reasoning(t) | Delta::Content(t) if text => send!(sse(&o::text_delta_chunk(&cid, created, &model, &t))),
                Delta::Reasoning(t) => send!(sse(&o::reasoning_chunk(&cid, created, &model, &t))),
                Delta::Content(t) => send!(sse(&o::content_chunk(&cid, created, &model, &t))),
                Delta::Call { .. } => {}
            }
        }
        let fr = r.finish_reason();
        if text {
            send!(sse(&o::text_final_chunk(&cid, created, &model, fr)));
            if p.include_usage { send!(sse(&o::text_usage_chunk(&cid, created, &model, &r.usage(&p)))); }
        } else {
            send!(sse(&o::final_chunk(&cid, created, &model, fr)));
            if p.include_usage { send!(sse(&o::usage_chunk(&cid, created, &model, &r.usage(&p)))); }
        }
        send!(Bytes::from_static(b"data: [DONE]\n\n"));
        r.account(&p, t0, r.asm.stopped());
        if !alive { cancel.store(true, Ordering::SeqCst); }
    });
    let mut resp = Response::new(Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(brx)));
    resp.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    resp.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    resp
}

#[cfg(test)]
mod scrub_tests {
    use super::*;
    use crate::tokenizer::ChatMessage;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage { role: role.into(), content: Some(content.into()), images: vec![], tool_calls: None, tool_call_id: None, name: None, reasoning_content: None,
                      layout: vec![], unsupported_parts: vec![] }
    }

    /// `Reply::served` redacts what the model echoes of a tool result's credential, in reasoning, content and call arguments,
    /// across delta boundaries; the assembler's own (raw) text is untouched.
    #[test]
    fn the_served_deltas_of_a_reply_are_scrubbed_across_chunk_boundaries() {
        let secret = "sk-live-9f8e7d6c5b4a39281706f5e4d3c2b1a0";
        let msgs = vec![msg("user", "show the config"), msg("tool", &format!("API_KEY={secret}\nregion=eu"))];
        let mut sc = crate::tool_scrub::Scrubber::new(crate::tool_scrub::Mode::Provenance, &msgs);
        // fed in 5-char slices, as tokens would arrive
        let text = format!("The key is {secret} and the region is eu.");
        let mut out = String::new();
        let mut i = 0;
        while i < text.len() { let j = (i + 5).min(text.len()); out.push_str(&sc.push_content(&text[i..j])); i = j; }
        out.push_str(&sc.finish_content(""));
        assert!(!out.contains(secret), "{out}");
        assert!(out.contains("[redacted:tool-secret]") && out.contains("region is eu"), "{out}");
        let full = sc.scrub_full(&format!("{{\"k\": \"{secret}\"}}")).into_owned();
        assert!(!full.contains(secret), "{full}");
        // off: nothing changes
        let mut off = crate::tool_scrub::Scrubber::new(crate::tool_scrub::Mode::Off, &msgs);
        assert_eq!(off.push_content(&text), text);
    }
}
