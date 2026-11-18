//! Thinking/answer splitting (`split_think*`, `split_constrained`) and the finish-reason mapping
//! (`spec_finish_reason`, `stop_reason_of`, `stop_kind`, generation-log spans, penalty validation).
//! Siblings: `chat_stream.rs` and `chat.rs` use these to shape the response; `mod.rs` has the map.

/// The opening marker of a tool call. While streaming we must never forward this (or a partial prefix
/// of it) to the client as CONTENT: a harness would render raw XML in the chat and never invoke the
/// tool. Once it appears, content emission stops and the rest is buffered for the tool_calls delta.
/// `<tool_call` is the shared PREFIX of qwen's `<tool_call>` and hy_v3's `<tool_call:opensource>` /
/// `<tool_calls:opensource>`, so one constant covers both families.
pub(super) const TOOL_OPEN: &str = "<tool_call";

/// Does a generation start INSIDE a reasoning block? The RENDERED PROMPT decides: it ends in an open
/// think tag. hy_v3 alone (`family_primed == false`, the `<think:opensource>` family) keeps the effort
/// arm: its low|high template primes `…assistant<think:opensource>` even when the tail check cannot see
/// it. The Qwen templates never open the block from `reasoning_effort` (Qwen3.8's only adds a system
/// instruction), so on a Qwen no-think render (a CLOSED empty block) a low|high effort is a direct
/// answer: Qwen3.5-0.8B/2B (template default: no think) under VeloBench's `reasoning_effort: "low"`
/// streamed the whole answer as reasoning_content with content empty (ISSUES 149).
pub(super) fn stream_starts_in_reasoning(prompt: &str, think_open: &str, family_primed: bool, effort: Option<&str>) -> bool {
    prompt.trim_end().ends_with(think_open)
        || (!family_primed && matches!(effort, Some("low") | Some("high")))
}

/// Split a completed generation into (reasoning, answer). If the close marker is present, everything
/// before it is reasoning (a leading think-open is stripped) and everything after (trimmed) is the
/// answer. If the marker never appears, the whole text is returned as the answer content.
pub(super) fn split_think(s: &str, think_open: &str, think_close: &str) -> (Option<String>, String) {
    split_think_in(s, think_open, think_close, true)
}

/// `split_think` that knows where the generation starts. When it starts as CONTENT (the prompt
/// closed the think block: thinking off), a close tag that lies inside a tool-call block is text of
/// that block (an argument that explains reasoning tags, a chat template), not a think boundary:
/// cutting there threw the whole call away, in both response modes (independent review M2). When it
/// starts in reasoning (`starts_in_reasoning`) the first close tag is the boundary as it always was
/// (a call drafted inside reasoning is reasoning).
pub(super) fn split_think_in(s: &str, think_open: &str, think_close: &str, starts_in_reasoning: bool) -> (Option<String>, String) {
    let mut from = 0usize;
    let found = loop {
        let Some(rel) = s[from..].find(think_close) else { break None };
        let idx = from + rel;
        if think_close.is_empty() { break Some(idx); }
        let inside_call = !starts_in_reasoning && {
            let head = &s[..idx];
            match head.rfind(TOOL_OPEN) { Some(o) => head.rfind("</tool_call").map_or(true, |c| c < o), None => false }
        };
        if !inside_call { break Some(idx); }
        from = idx + think_close.len();
    };
    match found {
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

/// `split_think_in` for a schema-constrained reply: the reasoning (only when the prompt opened a
/// think block) ends at the first close tag, where the constraint starts; everything after it is
/// the constrained document, taken literally (a think tag inside one of its strings is text).
pub(super) fn split_constrained(s: &str, think_close: &str, starts_in_reasoning: bool) -> (Option<String>, String) {
    if !starts_in_reasoning {
        return (None, s.to_string());
    }
    let (r, c) = match s.find(think_close) {
        Some(i) => (&s[..i], s[i + think_close.len()..].trim_start_matches(['\n', '\r', ' ', '\t'])),
        None => (s, ""),
    };
    let r = r.trim();
    (if r.is_empty() { None } else { Some(r.to_string()) }, c.to_string())
}

/// The scheduler's internal backstop reason (batch.rs) is not an OpenAI value — the spec is
/// stop|length|tool_calls|content_filter. Map it to what it means: generation ran out of room.
pub(super) fn spec_finish_reason(reason: &str) -> &str {
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
pub(super) fn stop_reason_of(reason: &str) -> Option<String> {
    (reason == "loop_detected").then(|| reason.to_string())
}

/// S-B9-REL-SMALLFIXES item 5(c): WHY a generation ended, for the `[gen]` line. The API keeps
/// `finish_reason: "stop"` for a loop-detector stop (client compatibility) and a tool call rewrites
/// it to "tool_calls", so the line's `finish=` cannot tell these apart; `stop_kind=` can.
/// `sched_reason` = the scheduler's own Finish reason (None = the channel closed without one: the
/// engine stopped), `stop_string_hit` = a request stop string matched on the HTTP side (the
/// scheduler never sees that: its lane is cancelled).
///   eos = a model stop token | stop_string | length (max_tokens or the context end) | loop (the
///   loop detector) | cancel | error | other (a reason this function does not know)
pub(crate) fn stop_kind(sched_reason: Option<&str>, stop_string_hit: bool) -> &'static str {
    if stop_string_hit { return "stop_string"; }
    match sched_reason {
        Some("stop") => "eos",
        Some("length") | Some("context_length_exceeded") => "length",
        Some("loop_detected") => "loop",
        Some("cancelled") => "cancel",
        Some(r) if r.starts_with("error") => "error",
        Some(_) => "other",
        None => "error",
    }
}

/// S-B9-REL-SURFACE2 item 8: the (reasoning, content) of a NON-streaming chat generation that ended
/// INSIDE the think block its prompt opened: no close tag in `text` (the caller checks both).
/// `split_think` returns such a text wholly as content, so the same request answered
/// `content: <the reasoning>` without `stream` and `reasoning_content` deltas with it. This is the
/// streaming path's split, applied to the finished text:
///   - the text up to the first tool-call marker (all of it when there is none, or when tool
///     parsing is off for the request: `hold_tools` false) is reasoning, trimmed, None when empty;
///   - what starts at the marker is the content the tool parser then reads: parsed calls become
///     `tool_calls`, an unparsed call-looking block stays visible as content (the stream's
///     held-back remainder); with no marker the content is empty.
pub(super) fn unclosed_think_split(text: &str, hold_tools: bool) -> (Option<String>, String) {
    let cut = if hold_tools { text.find(TOOL_OPEN).unwrap_or(text.len()) } else { text.len() };
    let r = text[..cut].trim();
    (if r.is_empty() { None } else { Some(r.to_string()) }, text[cut..].to_string())
}

/// `--unclosed-think content`: keep the v0.7.3 response of a generation that ended inside the think
/// block (the reasoning returned as `content`); `reasoning` = `unclosed_think_split`. The default
/// is the registry row's (the owner's ruling of 2026-10-09: `reasoning`; read by `opts::choice`).
pub(super) fn unclosed_think_as_content() -> bool {
    crate::opts::choice(crate::opt!("unclosed-think")) == "content"
}

/// S-B9-REL-SMALLFIXES item 5(b): the (reasoning, content) pair the `[gen]` line counts.
/// `split_think` returns a text WITHOUT a close tag wholly as content. When the prompt opened the
/// think block (`starts_in_reasoning`) and the model never closed it (`closed` = the close tag is
/// in the generated text), the request ended INSIDE the block: that text is reasoning. Log-only —
/// the response's own fields are not touched.
pub(crate) fn gen_log_spans<'a>(reasoning: Option<&'a str>, content: &'a str, closed: bool,
                                starts_in_reasoning: bool) -> (Option<&'a str>, &'a str) {
    if starts_in_reasoning && !closed && reasoning.is_none() && !content.is_empty() {
        (Some(content), "")
    } else {
        (reasoning, content)
    }
}

/// S-B9-REL-SMALLFIXES item 5(a): the `stop_p=.. stop_rank=..` tail of the `[gen]` line
/// (`na` = no scheduler sent it: not a stop-token finish, an engine without the readback, TP).
/// S-B9-REL-SURFACE2 item 1: `stop_p_law=greedy|sampled` follows them (appended, so the older
/// keys keep their order) and names the law `stop_p` is a probability under: `sampled` = the
/// truncated, temperature-scaled law the sampler drew from; `greedy` = the softmax at T=1.
pub(crate) fn stop_info_fields(info: Option<(f32, u32, bool)>) -> String {
    match info {
        Some((p, rank, sampled)) => format!("stop_p={p:.4} stop_rank={rank} stop_p_law={}",
                                            if sampled { "sampled" } else { "greedy" }),
        None => "stop_p=na stop_rank=na stop_p_law=na".to_string(),
    }
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

