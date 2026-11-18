//! `POST /tokenize` and `POST /detokenize` (vLLM-compatible), including chat-template and tools rendering.
//! Siblings: `admin.rs` holds the other utility routes; `mod.rs` registers all routes in `create_router`.

use super::*;

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
pub(super) async fn tokenize(State(state): State<AppState>, Json(body): Json<serde_json::Value>) -> Response {
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
pub(super) async fn detokenize(State(state): State<AppState>, Json(body): Json<serde_json::Value>) -> Response {
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

