//! Chat request preparation: JSON rejection answers, reasoning-effort resolution, the JSON-schema
//! constraint notes and `tool_choice` decisions.
//! Siblings: `chat.rs` (the handler) calls these before building the prompt; `mod.rs` has the map.

use super::*;

/// WP08: a request body that does not parse (wrong JSON types, missing required fields) is the
/// client's error — 400 invalid_request_error, as OpenAI/vLLM answer — never axum's bare 422.
pub(super) fn bad_json(e: axum::extract::rejection::JsonRejection) -> Response {
    (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
        "message": e.body_text(), "type": "invalid_request_error", "code": "invalid_request_body",
    }}))).into_response()
}

pub(super) fn bad_param(msg: String) -> Response {
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
pub(super) fn resolve_reasoning_effort(tokenizer: &QwenTokenizer, req_effort: Option<&str>,
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
pub(super) const SCHEMA_SEEN_CAP: usize = 1024;

/// Pure decision core (unit-tested): returns Some(true) to warn (new reason), Some(false) to
/// stay silent (duplicate), None once the cap is hit ("further warnings suppressed").
pub(super) fn schema_seen_note(
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

pub(super) fn warn_unenforced_schema_once(why: &str) {
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
            "[schema] WARNING: response_format is NOT enforced by this server — the reply is \
             unconstrained and may not match the requested schema (responses carry \
             'x-json-schema-enforced: none'). Reason: {why}"
        );
    }
}

/// W2-EXL3: the vocabulary table every constrained request shares (built once per process from
/// the tokenizer's output-byte tables; the server serves one tokenizer).
pub(super) fn schema_vocab_table(tok: &crate::tokenizer::QwenTokenizer) -> crate::json_schema::VocabTable {
    static T: std::sync::OnceLock<crate::json_schema::VocabTable> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let t0 = std::time::Instant::now();
        let t = crate::json_schema::VocabTable::build(tok.vocab_pieces());
        crate::reprintln!("[schema] vocabulary table built: {} ids ({:.0} ms, once per process)",
                          t.len(), t0.elapsed().as_secs_f64() * 1e3);
        t
    }).clone()
}

/// Marks a served response whose `response_format` was not enforced (see above) with
/// `x-json-schema-enforced: none`, and (W2-EXL3, `--json-schema-enforce on`) one whose schema
/// WAS armed as a token mask with `x-json-schema-enforced: full`. No header without a schema.
pub(super) fn attach_schema_unenforced_header(resp: &mut Response, unenforced: bool, enforced: bool) {
    let val = if unenforced { "none" } else if enforced { "full" } else { return };
    if let Ok(v) = HeaderValue::from_str(val) {
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
pub(super) fn apply_force_line(messages: &mut [crate::tokenizer::ChatMessage], line: &str) {
    if let Some(last) = messages.last_mut() {
        match &mut last.content {
            Some(c) => { c.push_str("\n\n"); c.push_str(line); }
            None => { last.content = Some(line.to_string()); }
        }
    }
}

pub(super) fn tool_choice_decision<'a>(
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

