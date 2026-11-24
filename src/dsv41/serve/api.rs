//! One chat request, from the HTTP body to a prompt and a job: the request fields, DeepSeek's chat encoding (ENC), the
//! tool catalogue, the thinking toggles and the sampler defaults. No I/O here: `prepare` is a pure function of the body and
//! the server's settings (the tokenizer aside), so it is unit-tested without a socket.

use std::sync::Arc;

use crate::dsv41::chat::catalogue::Catalogue;
use crate::dsv41::chat::request::{prepare_tools, resolve_thinking, thinking_token_budget, ThinkingRequest, ToolChoice};
use crate::dsv41::chat::stream::ParseOptions;
use crate::dsv41::chat::{attach_tools, encode_messages, jx, EncodeOptions, ThinkingMode};
use crate::dsv4_chat::{parse_json, Json};
use crate::tokenizer::QwenTokenizer;

use super::sampler::Sampling;

/// A client-visible error: HTTP status, OpenAI `type`, `code` and message.
#[derive(Debug, Clone)]
pub struct ApiError { pub status: u16, pub kind: &'static str, pub code: Option<&'static str>, pub message: String }

impl ApiError {
    pub fn bad(message: impl Into<String>) -> Self { ApiError { status: 400, kind: "invalid_request_error", code: None, message: message.into() } }
    pub fn code(mut self, c: &'static str) -> Self { self.code = Some(c); self }
    pub fn json(&self) -> String {
        serde_json::json!({"error": {"message": self.message, "type": self.kind, "param": null, "code": self.code}}).to_string()
    }
}

/// What the server decided at boot (flags and the model's files).
#[derive(Clone)]
pub struct ServeSettings {
    pub model_name: String,
    pub max_seq_len: usize,
    /// `--max-tokens`: the default budget of a request that sends none (None = fill the context)
    pub default_max_tokens: Option<usize>,
    /// sampler defaults (DeepSeek's 1.0 / 1.0 unless the pack's generation_config says otherwise)
    pub default_sampling: Sampling,
    /// `--thinking on|off` (None = auto = on, the trained default)
    pub server_thinking: Option<bool>,
    /// `--reasoning-effort`
    pub server_effort: Option<Json>,
    pub keep_tools_when_none: bool,
    pub seedless_mode: String,
    pub server_think_budget: Option<u64>,
    pub include_usage_default: bool,
    pub vision: Option<crate::dsv41::config::Vision>,
    pub image_token_id: u32,
    pub vision_on: bool,
    pub max_images: usize,
}

/// A request ready for the engine and the output side.
pub struct Prepared {
    pub model: String,
    pub prompt_ids: Vec<u32>,
    pub prompt_text: String,
    pub max_new: usize,
    pub min_tokens: usize,
    pub ignore_eos: bool,
    pub sampling: Sampling,
    pub seed: u64,
    pub seeded: bool,
    pub stop_strings: Vec<String>,
    pub stream: bool,
    pub include_usage: bool,
    pub thinking: bool,
    pub think_budget: Option<u64>,
    pub catalogue: Option<Catalogue>,
    pub parse: ParseOptions,
    pub wants_logprobs: bool,
    pub unenforced_format: bool,
    pub tools_offered: bool,
    pub effort_desc: String,
    /// `--scrub-tool-secrets` (not off): the request's messages, for the provenance layer
    pub scrub_msgs: Option<Vec<crate::tokenizer::ChatMessage>>,
    /// `--tool-trace FILE`: the request side of a tool-involved turn
    pub trace: Option<std::sync::Arc<crate::tool_trace::Turn>>,
    /// a `/v1/completions` request: the reply is raw text in the `text_completion` shape (no chat framing, no tools)
    pub text_shape: bool,
    /// `/v1/completions` `echo: true`: the prompt text leads the reply
    pub echo_text: Option<String>,
    /// leg VIS1: per-prompt-token VL types, preprocessed images, and prefix-cache span keys
    pub vl_types: Vec<i8>,
    pub vl_images: Vec<crate::dsv41::vision::preproc::PreparedImage>,
    pub image_spans: Vec<crate::dsv41::vision::preproc::ImageSpanKey>,
}

/// The request's messages as the shared scrubber reads them (role + text; content parts joined).
fn scrub_messages(v: Option<&Json>) -> Vec<crate::tokenizer::ChatMessage> {
    let Some(Json::Array(ms)) = v else { return Vec::new() };
    ms.iter().filter_map(|m| {
        let role = jx::get_str(m, "role")?.to_string();
        let content = match jx::get(m, "content") {
            Some(Json::Str(t)) => Some(t.clone()),
            Some(Json::Array(parts)) => Some(parts.iter().filter_map(|p| jx::get_str(p, "text")).collect::<Vec<_>>().join("\n")),
            _ => None,
        };
        Some(crate::tokenizer::ChatMessage { role, content, images: Vec::new(), tool_calls: None, tool_call_id: None, name: None, reasoning_content: None,
                                             layout: Vec::new(), unsupported_parts: Vec::new() })
    }).collect()
}

fn num(v: Option<&Json>, what: &str) -> Result<Option<f64>, ApiError> {
    match v {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Num(s)) => s.parse::<f64>().map(Some).map_err(|_| ApiError::bad(format!("'{what}' is not a number"))),
        Some(_) => Err(ApiError::bad(format!("'{what}' must be a number"))),
    }
}

fn uint(v: Option<&Json>, what: &str) -> Result<Option<u64>, ApiError> {
    match num(v, what)? {
        None => Ok(None),
        Some(x) if x >= 0.0 && x.fract() == 0.0 && x < 1.8e19 => Ok(Some(x as u64)),
        Some(_) => Err(ApiError::bad(format!("'{what}' must be a non-negative integer"))),
    }
}

/// Non-negative integer or `-1` = unlimited (`max_tokens`, `max_completion_tokens`, `thinking_token_budget`).
fn uint_or_minus_one(v: Option<&Json>, what: &str) -> Result<Option<u64>, ApiError> {
    match num(v, what)? {
        None => Ok(None),
        Some(x) if x == -1.0 => Ok(None),
        Some(x) if x >= 0.0 && x.fract() == 0.0 && x < 1.8e19 => Ok(Some(x as u64)),
        Some(_) => Err(ApiError::bad(format!("'{what}' must be a non-negative integer or -1"))),
    }
}

fn min_tokens_opt(v: Option<&Json>) -> Result<usize, ApiError> {
    Ok(uint_or_minus_one(v, "min_tokens")?.map_or(0, |n| n as usize))
}

fn bool_opt(v: Option<&Json>, what: &str) -> Result<bool, ApiError> {
    match v {
        None | Some(Json::Null) => Ok(false),
        Some(Json::Bool(b)) => Ok(*b),
        Some(_) => Err(ApiError::bad(format!("'{what}' must be a boolean"))),
    }
}

/// Non-negative integer seed, or any negative integer (e.g. `-1`) = unseeded random.
fn seed_uint(v: Option<&Json>) -> Result<Option<u64>, ApiError> {
    match num(v, "seed")? {
        None => Ok(None),
        Some(x) if x < 0.0 && x.fract() == 0.0 => Ok(None),
        Some(x) if x >= 0.0 && x.fract() == 0.0 && x < 1.8e19 => Ok(Some(x as u64)),
        Some(_) => Err(ApiError::bad("'seed' must be an integer")),
    }
}

fn stop_list(v: Option<&Json>) -> Result<Vec<String>, ApiError> {
    match v {
        None | Some(Json::Null) => Ok(Vec::new()),
        Some(Json::Str(s)) => Ok(vec![s.clone()].into_iter().filter(|s| !s.is_empty()).collect()),
        Some(Json::Array(a)) => a.iter().map(|x| x.as_str().map(str::to_string).ok_or_else(|| ApiError::bad("'stop' entries must be strings")))
            .collect::<Result<Vec<_>, _>>().map(|v| v.into_iter().filter(|s| !s.is_empty()).collect()),
        Some(_) => Err(ApiError::bad("'stop' must be a string or a list of strings")),
    }
}

/// The request-level thinking switch: a boolean, `"on"`/`"off"`, or DeepSeek's API form `{"type": "enabled" | "disabled"}`.
fn thinking_switch(v: Option<&Json>) -> Result<Option<bool>, ApiError> {
    match v {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Bool(b)) => Ok(Some(*b)),
        Some(Json::Str(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "on" | "true" | "enabled" | "1" => Ok(Some(true)),
            "off" | "false" | "disabled" | "0" => Ok(Some(false)),
            _ => Err(ApiError::bad("'thinking' must be a boolean or {\"type\": \"enabled\" | \"disabled\"}")),
        },
        Some(o @ Json::Object(_)) => match jx::get(o, "type").and_then(|t| t.as_str()) {
            Some("enabled") => Ok(Some(true)),
            Some("disabled") => Ok(Some(false)),
            _ => Err(ApiError::bad("'thinking.type' must be \"enabled\" or \"disabled\"")),
        },
        Some(_) => Err(ApiError::bad("'thinking' must be a boolean or {\"type\": \"enabled\" | \"disabled\"}")),
    }
}

/// Normalize real-world OpenAI / Responses API / Messages API / OpenWebUI / Vercel AI SDK message shapes
/// onto the strict `encoding.py` forms (`encode_messages` itself remains untouched and byte-exact).
fn normalize_content_parts(parts: &mut [Json]) -> Result<(), ApiError> {
    for part in parts.iter_mut() {
        if let Json::Str(t) = part {
            *part = jx::obj(vec![("type", jx::s("text")), ("text", Json::Str(t.clone()))]);
            continue;
        }
        if !matches!(part, Json::Object(_)) {
            continue;
        }
        let raw_kind = jx::get_str(part, "type").map(str::to_string);
        match raw_kind.as_deref() {
            Some(bad @ ("video_url" | "video" | "input_video" | "audio_url" | "audio" | "input_audio")) => {
                return Err(ApiError::bad(format!(
                    "content part type '{bad}' is not supported (text and images only; video/audio input is not supported)"
                ))
                .code("unsupported_content_part"));
            }
            Some("input_text" | "output_text") => {
                jx::set(part, "type", jx::s("text"));
                if !matches!(jx::get(part, "text"), Some(Json::Str(_))) {
                    if matches!(jx::get(part, "text"), None | Some(Json::Null)) {
                        jx::set(part, "text", jx::s(""));
                    }
                }
            }
            Some("text") => {
                if matches!(jx::get(part, "text"), Some(Json::Null)) {
                    jx::set(part, "text", jx::s(""));
                }
            }
            None if jx::present(part, "image_url") => {
                normalize_image_part(part);
            }
            None if jx::present(part, "text") => {
                jx::set(part, "type", jx::s("text"));
                if matches!(jx::get(part, "text"), Some(Json::Null)) {
                    jx::set(part, "text", jx::s(""));
                }
            }
            Some("image" | "image_url" | "input_image") => {
                normalize_image_part(part);
            }
            Some("tool_result") => {
                if let Some(Json::Array(inner)) = jx::get_mut(part, "content") {
                    normalize_content_parts(inner)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Normalize any image content block (`image`, `image_url`, `input_image`, untyped `{image_url}`) into
/// the exact shape `dsv41::chat::preprocess::extract_image` and `vision::preproc::load_image_bytes` expect.
fn normalize_image_part(part: &mut Json) {
    if let Some(src @ Json::Object(_)) = jx::get(part, "source").cloned() {
        *part = jx::obj(vec![("type", jx::s("image")), ("source", src)]);
        return;
    }
    for key in ["data", "b64_json", "file_data"] {
        if let Some(Json::Str(d)) = jx::get(part, key) {
            if !d.is_empty() {
                let d = d.clone();
                // `file_data` in Responses API may be either `data:image/...;base64,...` or bare base64.
                if d.starts_with("data:") {
                    *part = jx::obj(vec![("type", jx::s("image")), ("url", Json::Str(d))]);
                } else {
                    *part = jx::obj(vec![("type", jx::s("image")), ("data", Json::Str(d))]);
                }
                return;
            }
        }
    }
    if let Some(iu) = jx::get(part, "image_url").cloned() {
        match iu {
            Json::Str(u) if !u.is_empty() => {
                *part = jx::obj(vec![("type", jx::s("image_url")), ("image_url", jx::obj(vec![("url", Json::Str(u))]))]);
                return;
            }
            Json::Object(_) => {
                if let Some(Json::Str(u)) = jx::get(&iu, "url") {
                    if !u.is_empty() {
                        let u = u.clone();
                        *part = jx::obj(vec![("type", jx::s("image_url")), ("image_url", jx::obj(vec![("url", Json::Str(u))]))]);
                        return;
                    }
                }
                for dkey in ["data", "b64_json"] {
                    if let Some(Json::Str(d)) = jx::get(&iu, dkey) {
                        if !d.is_empty() {
                            let d = d.clone();
                            *part = jx::obj(vec![("type", jx::s("image")), ("data", Json::Str(d))]);
                            return;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    for ukey in ["url", "image"] {
        if let Some(Json::Str(u)) = jx::get(part, ukey) {
            if !u.is_empty() {
                let u = u.clone();
                *part = jx::obj(vec![("type", jx::s("image")), ("url", Json::Str(u))]);
                return;
            }
        }
    }
    if jx::get_str(part, "type") == Some("input_image") {
        jx::set(part, "type", jx::s("image"));
    }
}

fn normalize_http_messages(messages: &mut [Json]) -> Result<(), ApiError> {
    for msg in messages.iter_mut() {
        if !matches!(msg, Json::Object(_)) {
            continue;
        }
        if jx::get_str(msg, "role") == Some("developer") {
            jx::set(msg, "role", jx::s("system"));
        }
        if jx::get_str(msg, "role") == Some("user")
            && matches!(jx::get(msg, "content"), Some(Json::Null))
            && !jx::has(msg, "content_blocks")
        {
            jx::set(msg, "content", jx::s(""));
        }
        if let Some(Json::Array(parts)) = jx::get_mut(msg, "content") {
            normalize_content_parts(parts)?;
        }
        if let Some(Json::Array(parts)) = jx::get_mut(msg, "content_blocks") {
            normalize_content_parts(parts)?;
        }
    }
    Ok(())
}

fn effort_json(s: &str) -> Json { if let Ok(n) = s.parse::<u32>() { Json::Num(n.to_string()) } else { Json::Str(s.to_string()) } }
pub fn effort_from_flag(s: &str) -> Json { effort_json(s) }

/// The response_format a request asks for, as the system-message value the reference renders, and whether the answer
/// is a JSON channel (the parser then never reads calls out of it).
fn response_format(v: Option<&Json>) -> Result<Option<Json>, ApiError> {
    let Some(rf) = v.filter(|v| !v.is_null()) else { return Ok(None) };
    match jx::get(rf, "type").and_then(|t| t.as_str()) {
        Some("text") | None => Ok(None),
        Some("json_object") => Ok(Some(parse_json("{\"type\": \"object\"}").expect("static"))),
        Some("json_schema") => {
            let js = jx::get(rf, "json_schema").ok_or_else(|| ApiError::bad("response_format json_schema needs a 'json_schema' object"))?;
            Ok(Some(jx::get(js, "schema").cloned().unwrap_or_else(|| js.clone())))
        }
        Some(other) => Err(ApiError::bad(format!("unsupported response_format type '{other}'"))),
    }
}

/// Parse `body` and build everything a request needs. `tok` tokenizes the rendered prompt (no special-token insertion:
/// the BOS is in the prompt).
pub fn prepare(body: &str, st: &ServeSettings, tok: &QwenTokenizer) -> Result<Prepared, ApiError> {
    let body = body.strip_prefix('\u{feff}').unwrap_or(body);
    let req = parse_json(body).map_err(|e| ApiError::bad(format!("the request body is not valid JSON (byte {}: {})", e.0, e.1)))?;
    if !matches!(req, Json::Object(_)) { return Err(ApiError::bad("the request body must be a JSON object")); }
    let g = |k: &str| jx::get(&req, k);
    let mut messages: Vec<Json> = match g("messages") {
        Some(Json::Array(a)) if !a.is_empty() => a.clone(),
        _ => return Err(ApiError::bad("'messages' must be a non-empty list")),
    };
    normalize_http_messages(&mut messages)?;
    // ---- thinking, effort ----
    let kw = g("chat_template_kwargs").filter(|v| !v.is_null());
    let req_thinking = thinking_switch(g("thinking").filter(|v| !v.is_null()).or_else(|| g("enable_thinking")))?;
    let resolved = resolve_thinking(&ThinkingRequest {
        chat_template_kwargs: kw, reasoning_effort: g("reasoning_effort"), thinking: req_thinking,
        server_effort: st.server_effort.as_ref(), server_thinking: st.server_thinking,
    }).map_err(|e| ApiError::bad(e.to_string()))?;
    let thinking_on = resolved.mode == ThinkingMode::Thinking;
    // ---- tools ----
    let choice = ToolChoice::from_json(g("tool_choice")).map_err(|e| ApiError::bad(e.to_string()))?;
    let tools = g("tools").filter(|v| !v.is_null());
    let prepared = prepare_tools(tools, &choice, st.keep_tools_when_none).map_err(|e| ApiError::bad(e.to_string()))?;
    if prepared.prompt_tools.is_some() && jx::get(&messages[0], "role").and_then(|r| r.as_str()) != Some("system") {
        // encoding.py renders tools only on a system message: on a user message they are silently dropped and the model
        // answers that it has no tool. Lead with an empty system message, as response_format does below.
        messages.insert(0, jx::obj(vec![("role", jx::s("system")), ("content", jx::s(""))]));
    }
    if let Some(t) = &prepared.prompt_tools { attach_tools(&mut messages, t).map_err(|e| ApiError::bad(e.to_string()))?; }
    // ---- response_format ----
    let rf = response_format(g("response_format"))?;
    let json_channel = rf.is_some();
    if let Some(rf) = &rf {
        if jx::get(&messages[0], "role").and_then(|r| r.as_str()) == Some("system") {
            jx::set(&mut messages[0], "response_format", rf.clone());
        } else {
            messages.insert(0, jx::obj(vec![("role", jx::s("system")), ("content", jx::s("")), ("response_format", rf.clone())]));
        }
    }
    // ---- the prompt ----
    let mut opts = EncodeOptions::new(resolved.mode);
    opts.reasoning_effort = resolved.effort.clone();
    let enc = encode_messages(&messages, &opts).map_err(|e| ApiError::bad(e.to_string()))?;
    let raw_ids = tok.encode(&enc.prompt, false).map_err(|e| ApiError::bad(format!("tokenizer: {e:#}")))?;
    let (prompt_ids, vl_types, vl_images, image_spans) = if enc.images.is_empty() {
        (raw_ids, Vec::new(), Vec::new(), Vec::new())
    } else {
        let Some(vis) = st.vision.as_ref().filter(|_| st.vision_on) else {
            return Err(ApiError::bad("vision is disabled (--dsv41-vision off) or this pack has no vision_config").code("images_unsupported"));
        };
        let vlp = crate::dsv41::vision::preproc::prepare_vl_inputs(&raw_ids, &enc.images, st.image_token_id, vis, st.max_images)?;
        (vlp.prompt_ids, vlp.token_types, vlp.images, vlp.spans)
    };
    if prompt_ids.is_empty() { return Err(ApiError::bad("the prompt is empty")); }
    if prompt_ids.len() + 1 > st.max_seq_len {
        return Err(ApiError::bad(format!("this model's context is {} tokens; the prompt has {} (no room to generate). Shorten the messages or start the server with a larger --max-seq-len",
                                         st.max_seq_len, prompt_ids.len())).code("context_length_exceeded"));
    }
    // ---- budget ----
    let room = st.max_seq_len - prompt_ids.len();
    let asked = uint_or_minus_one(g("max_completion_tokens"), "max_completion_tokens")?
        .or(uint_or_minus_one(g("max_tokens"), "max_tokens")?)
        .map(|n| n as usize).or(st.default_max_tokens).unwrap_or(usize::MAX);
    if asked == 0 { return Err(ApiError::bad("'max_tokens' must be at least 1")); }
    let max_new = asked.min(room);
    let min_tokens = min_tokens_opt(g("min_tokens"))?;
    let ignore_eos = bool_opt(g("ignore_eos"), "ignore_eos")?;
    // ---- sampling ----
    let (sampling, seed, seeded) = read_sampling(&req, st, &prompt_ids)?;
    // ---- the rest ----
    let stream = matches!(g("stream"), Some(Json::Bool(true)));
    let include_usage = match g("stream_options") {
        Some(o @ Json::Object(_)) => jx::get(o, "include_usage").and_then(|v| v.as_bool()).unwrap_or(st.include_usage_default),
        _ => st.include_usage_default,
    };
    let think_budget = if thinking_on { thinking_token_budget(uint_or_minus_one(g("thinking_token_budget"), "thinking_token_budget")?, st.server_think_budget) } else { None };
    let max_calls = match g("parallel_tool_calls") { Some(Json::Bool(false)) => Some(1), _ => None };
    let mut parse = ParseOptions::new(thinking_on);
    parse.max_calls = max_calls;
    parse.json_channel = json_channel;
    let wants_logprobs = matches!(g("logprobs"), Some(Json::Bool(true))) || g("top_logprobs").map_or(false, |v| !v.is_null());
    let effort_desc = resolved.effort.as_ref().map_or("default".to_string(), |e| crate::dsv41::chat::pyjson::to_json(e));
    // ---- opt-in hardening / tracing (the shared flags: --scrub-tool-secrets, --tool-trace) ----
    let scrub_msgs = (crate::tool_scrub::Mode::from_opt() != crate::tool_scrub::Mode::Off).then(|| scrub_messages(g("messages")));
    let involves_tools = prepared.catalogue.is_some() || matches!(g("messages"), Some(Json::Array(ms)) if ms.iter().any(|m| jx::get_str(m, "role") == Some("tool")));
    let trace = if involves_tools {
        crate::tool_trace::capture(&enc.prompt, serde_json::json!({
            "family": "dsv41", "stream": matches!(g("stream"), Some(Json::Bool(true))), "seed": seed, "seeded": seeded, "thinking": thinking_on,
            "temperature": sampling.temperature, "top_p": sampling.top_p, "top_k": sampling.top_k, "tools": prepared.catalogue.is_some(),
            "effort": effort_desc, "prompt_tokens": prompt_ids.len(),
        })).map(std::sync::Arc::new)
    } else { None };
    Ok(Prepared {
        model: g("model").and_then(|m| m.as_str()).unwrap_or(&st.model_name).to_string(),
        prompt_ids, prompt_text: enc.prompt, max_new, min_tokens, ignore_eos, sampling, seed, seeded,
        stop_strings: stop_list(g("stop"))?, stream, include_usage, thinking: thinking_on, think_budget,
        tools_offered: prepared.catalogue.is_some(), catalogue: prepared.catalogue, parse, wants_logprobs,
        unenforced_format: json_channel, effort_desc, scrub_msgs, trace, text_shape: false, echo_text: None,
        vl_types, vl_images, image_spans,
    })
}

/// The sampler fields of a request (temperature, top_p, top_k, penalties) over the server's defaults, and the seed.
fn read_sampling(req: &Json, st: &ServeSettings, prompt_ids: &[u32]) -> Result<(Sampling, u64, bool), ApiError> {
    let g = |k: &str| jx::get(req, k);
    let d = &st.default_sampling;
    let temperature = num(g("temperature"), "temperature")?.map_or(d.temperature, |x| x as f32);
    let top_p = num(g("top_p"), "top_p")?.map_or(d.top_p, |x| x as f32);
    if !(temperature >= 0.0) || !(top_p > 0.0 && top_p <= 1.0) {
        return Err(ApiError::bad("'temperature' must be >= 0 and 'top_p' in (0, 1]"));
    }
    let top_k = match num(g("top_k"), "top_k")? { None => d.top_k, Some(x) if x <= 0.0 => 0, Some(x) => x as usize };
    let sampling = Sampling {
        temperature, top_p, top_k,
        presence_penalty: num(g("presence_penalty"), "presence_penalty")?.map_or(d.presence_penalty, |x| x as f32),
        frequency_penalty: num(g("frequency_penalty"), "frequency_penalty")?.map_or(d.frequency_penalty, |x| x as f32),
        repetition_penalty: num(g("repetition_penalty"), "repetition_penalty")?.map_or(d.repetition_penalty, |x| x as f32),
    };
    let (seed, seeded) = match seed_uint(g("seed"))? {
        Some(s) => (s, true),
        None => (super::sampler::seedless_seed(&st.seedless_mode, prompt_ids), false),
    };
    Ok((sampling, seed, seeded))
}

/// `POST /v1/completions` (generation): the prompt is raw text (or token ids), no chat encoding, no thinking, no tools; the reply
/// is raw text. `max_tokens` defaults to 16, as on the shared server. (Prompt SCORING, `echo` with `max_tokens` 0, is `http.rs`.)
pub fn prepare_completion(body: &str, st: &ServeSettings, tok: &QwenTokenizer) -> Result<Prepared, ApiError> {
    let body = body.strip_prefix('\u{feff}').unwrap_or(body);
    let req = parse_json(body).map_err(|e| ApiError::bad(format!("the request body is not valid JSON (byte {}: {})", e.0, e.1)))?;
    if !matches!(req, Json::Object(_)) { return Err(ApiError::bad("the request body must be a JSON object")); }
    let g = |k: &str| jx::get(&req, k);
    let (prompt_ids, text) = match g("prompt") {
        Some(Json::Str(s)) => (tokenize_text(tok, s)?, Some(s.clone())),
        Some(Json::Array(a)) if a.len() == 1 && matches!(a[0], Json::Str(_)) => {
            let s = a[0].as_str().unwrap_or_default().to_string();
            (tokenize_text(tok, &s)?, Some(s))
        }
        Some(Json::Array(a)) if !a.is_empty() && a.iter().all(|x| matches!(x, Json::Num(_))) => {
            let vocab = tok.vocab_size() as u64;
            let mut ids = Vec::with_capacity(a.len());
            for x in a {
                match uint(Some(x), "prompt")? { Some(t) if t < vocab => ids.push(t as u32), _ => return Err(ApiError::bad(format!("'prompt' ids must be integers in [0, {vocab})"))) }
            }
            (ids, None)
        }
        _ => return Err(ApiError::bad("'prompt' must be a string or a list of token ids")),
    };
    if prompt_ids.is_empty() { return Err(ApiError::bad("the prompt is empty")); }
    if prompt_ids.len() + 1 > st.max_seq_len {
        return Err(ApiError::bad(format!("this model's context is {} tokens; the prompt has {} (no room to generate). Shorten the prompt or start the server with a larger --max-seq-len",
                                         st.max_seq_len, prompt_ids.len())).code("context_length_exceeded"));
    }
    let room = st.max_seq_len - prompt_ids.len();
    let asked = uint(g("max_tokens"), "max_tokens")?.map(|n| n as usize).unwrap_or(16);
    if asked == 0 { return Err(ApiError::bad("'max_tokens' must be at least 1 (0 scores the prompt: send echo: true)")); }
    let max_new = asked.min(room);
    let min_tokens = min_tokens_opt(g("min_tokens"))?;
    let ignore_eos = bool_opt(g("ignore_eos"), "ignore_eos")?;
    let (sampling, seed, seeded) = read_sampling(&req, st, &prompt_ids)?;
    let stream = matches!(g("stream"), Some(Json::Bool(true)));
    let include_usage = match g("stream_options") {
        Some(o @ Json::Object(_)) => jx::get(o, "include_usage").and_then(|v| v.as_bool()).unwrap_or(st.include_usage_default),
        _ => st.include_usage_default,
    };
    let echo_text = if matches!(g("echo"), Some(Json::Bool(true))) { Some(text.unwrap_or_else(|| tok.decode(&prompt_ids, false).unwrap_or_default())) } else { None };
    let wants_logprobs = match g("logprobs") { Some(Json::Bool(true)) => true, Some(v @ Json::Num(_)) => uint(Some(v), "logprobs")?.map_or(false, |n| n > 0), _ => false };
    // no catalogue, no thinking: the parser reads the bytes as plain text (a literal DSML block stays visible, never a call)
    Ok(Prepared {
        model: g("model").and_then(|m| m.as_str()).unwrap_or(&st.model_name).to_string(),
        prompt_ids, prompt_text: String::new(), max_new, min_tokens, ignore_eos, sampling, seed, seeded,
        stop_strings: stop_list(g("stop"))?, stream, include_usage, thinking: false, think_budget: None,
        tools_offered: false, catalogue: None, parse: ParseOptions::new(false), wants_logprobs,
        unenforced_format: false, effort_desc: "default".into(), scrub_msgs: None, trace: None, text_shape: true, echo_text,
        vl_types: Vec::new(), vl_images: Vec::new(), image_spans: Vec::new(),
    })
}

/// Raw text to ids as the shared server's `/v1/tokenize` does by default: the tokenizer's own special-token handling.
pub fn tokenize_text(tok: &QwenTokenizer, s: &str) -> Result<Vec<u32>, ApiError> {
    tok.encode(s, true).map_err(|e| ApiError::bad(format!("tokenizer: {e:#}")))
}

pub type SharedTok = Arc<QwenTokenizer>;
