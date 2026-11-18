//! Logprobs request validation and the OpenAI-shaped logprobs JSON for chat and completions.
//! Siblings: `chat.rs` and `completions.rs` call into this; the request types live in `mod.rs`.

use super::*;

/// Max alternatives per token for `top_logprobs` (chat) / `logprobs` (completions).
pub(super) const MAX_TOP_LOGPROBS: u32 = 20;

/// One output token's log probabilities as received from the engine (`TokEvent::Logprob`),
/// aligned with the token list; `None` = the engine sent none for that token.
pub(super) type TokLp = Option<(f32, Vec<(u32, f32)>)>;

/// The chat request's logprobs setting -> the engine's `BatchRequest::logprobs`.
pub(super) fn chat_logprobs_param(logprobs: Option<bool>, top: Option<u32>) -> Result<Option<u8>, String> {
    let n = match (logprobs.unwrap_or(false), top) {
        (false, None) => return Ok(None),
        (false, Some(_)) => return Err("top_logprobs requires logprobs to be true".into()),
        (true, Some(n)) if n > MAX_TOP_LOGPROBS =>
            return Err(format!("top_logprobs must be between 0 and {MAX_TOP_LOGPROBS} (got {n})")),
        (true, n) => n.unwrap_or(0) as u8,
    };
    Ok(Some(n))
}

/// The completions request's legacy `logprobs` -> the engine's `BatchRequest::logprobs`.
pub(super) fn completion_logprobs_param(logprobs: Option<u32>) -> Result<Option<u8>, String> {
    match logprobs {
        None => Ok(None),
        Some(n) if n > MAX_TOP_LOGPROBS => Err(format!("logprobs must be between 0 and {MAX_TOP_LOGPROBS} (got {n})")),
        Some(n) => Ok(Some(n as u8)),
    }
}

/// PR #30 port, owner rule (support it or fall back, never refuse): why THIS server cannot return the log
/// probabilities a valid request asked for, as (disposition, reason); None = it can. Pure (unit-tested).
pub(super) fn logprobs_fallback(engine_ok: bool, scrub_on: bool, stream: bool) -> Option<(&'static str, &'static str)> {
    if !engine_ok {
        Some(("unsupported", "this serving engine does not compute them (the EXL3 server at TP=1, 2 and 4 does)"))
    } else if scrub_on {
        Some(("withheld", "--scrub-tool-secrets is on, and per-token bytes could rebuild text the scrubber removed"))
    } else if stream {
        Some(("stream", "stream=true is not wired for them yet"))
    } else {
        None
    }
}

/// The logprobs to ask the engine for: the request's, or None when `logprobs_fallback` says this server cannot
/// return them; the reply then carries `logprobs: null`, the reason prints once per boot, and every request is
/// counted on velogb10_logprobs_requests_total.
pub(super) fn logprobs_gate(n: Option<u8>, state: &AppState, stream: bool) -> Option<u8> {
    n?;
    let scrub_on = crate::tool_scrub::Mode::from_opt() != crate::tool_scrub::Mode::Off;
    match logprobs_fallback(state.logprobs_ok, scrub_on, stream) {
        None => { crate::metrics::logprobs_request("served"); n }
        Some((disp, why)) => {
            crate::metrics::logprobs_request(disp);
            static SAID: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
            let mut g = SAID.lock().unwrap_or_else(|p| p.into_inner());
            if !g.contains(&disp) {
                g.push(disp);
                crate::reprintln!("[serve] logprobs: {why}: such requests are served with logprobs null \
(velogb10_logprobs_requests_total{{disposition=\"{disp}\"}}; this line prints once per boot)");
            }
            None
        }
    }
}

/// The output tokens that carry log probabilities for bytes `range` of the decoded text: every
/// generated token except stop tokens that overlaps the range (a token that prints nothing counts
/// when its position lies inside it). Returns (index, byte offset of the token in the decoded
/// text, the token's bytes clipped to the range).
pub(super) fn logprob_token_span(bytes_of: impl Fn(u32) -> Vec<u8>, stop_ids: &[u32], tokens: &[u32],
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
pub(super) fn content_byte_range(text: &str, content: &str, think_close: &str) -> Option<std::ops::Range<usize>> {
    let from = text.find(think_close).map_or(0, |p| p + think_close.len());
    if content.is_empty() { return Some(from..from); }
    let p = from + text[from..].find(content)?;
    Some(p..p + content.len())
}

/// OpenAI chat `logprobs`: `{"content": [{token, logprob, bytes, top_logprobs: [...]}]}` for the
/// tokens of bytes `range` of the decoded text (the response's `content`); a token cut by the
/// range edge is listed with only its bytes inside the range, its probabilities unchanged. None
/// when a kept token has no engine entry (a readback failure: reported as null, never guessed).
pub(super) fn chat_logprobs_json(tok: &QwenTokenizer, stop_ids: &[u32], tokens: &[u32], lps: &[TokLp],
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
pub(super) fn completion_logprobs_json(tok: &QwenTokenizer, stop_ids: &[u32], tokens: &[u32], lps: &[TokLp],
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

