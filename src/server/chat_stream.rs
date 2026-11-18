//! The streaming chat generator `chat_sse_stream` (token events to SSE frames, tool-call holding, reasoning
//! split) and its small helpers. The request/state types it consumes (`ChatSse`, `NextTurn`, `GenLog`) live in `mod.rs`.
//! Siblings: `chat.rs` builds a `ChatSse` and drives this; `sse.rs` frames the output.

use super::*;

/// `--dump-tools`: the WHOLE text the tool-call parser was given for one request, as one log line
/// `[dump-tools] {"raw": "..."}` (a JSON object, so the text survives the log byte for byte). The
/// `[req] raw model output` line next to it stops at 1200 characters, which cuts a call that
/// writes a file: a cut block has no close tag and the parser drops it, so that line cannot be
/// replayed (S-B9-REL-SURFACE2B R1, the offline replay: examples/toolparse_replay.rs).
pub(super) fn dump_tools_line(raw: &str) -> String {
    format!("[dump-tools] {}", serde_json::json!({ "raw": raw }))
}

/// The text a finished stream has not sent yet, as (reasoning, content): `acc[emitted..]`, where
/// `emitted` is the live emit cursor (the larger of the reasoning and the content cursor: in
/// reasoning mode the content cursor stays 0, and using it alone would re-send the whole reply).
/// Called only when no tool call was parsed, so all of it is text.
///  * In content mode it is content.
///  * Inside a think block that was never closed (`in_reasoning`) it is reasoning, up to a held
///    tool-call marker; from the marker on it is content: a call-looking block the parser could
///    not read stays visible as content, which is what this stream always did with it.
///  * A think block that opened AND closed inside the reply's last piece (the emit sites hand
///    over to reasoning at the open tag and look at the rest only with the next piece, which
///    never came): reasoning up to the close tag, content after it; the tag itself is not sent.
/// Empty when everything was sent (also when a stop string cut the buffer below the cursor).
pub(super) fn held_tail<'a>(acc: &'a str, emitted: usize, in_reasoning: bool, tool_hold: Option<&str>, think_close: &str)
    -> (&'a str, &'a str) {
    if emitted >= acc.len() { return ("", ""); }
    let tail = &acc[emitted..];
    if !in_reasoning { return ("", tail); }
    let marker = tool_hold.and_then(|m| tail.find(m)).unwrap_or(tail.len());
    match tail.find(think_close).filter(|&c| !think_close.is_empty() && c < marker) {
        Some(c) => (&tail[..c], tail[c + think_close.len()..].trim_start_matches(['\n', '\r', ' ', '\t'])),
        None => tail.split_at(marker),
    }
}

pub(super) fn tool_calls_chunk(cid: &str, created: i64, model: &str, calls: &[ToolCall]) -> String {
    let arr: Vec<serde_json::Value> = calls.iter().enumerate().map(|(i, c)| serde_json::json!({
        "index": i, "id": c.id, "type": c.kind,
        "function": {"name": c.function.name, "arguments": c.function.arguments},
    })).collect();
    serde_json::json!({
        "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": {"tool_calls": arr}, "finish_reason": null}],
    }).to_string()
}

/// The `/v1/chat/completions` SSE stream: every data event of a streamed chat response, in
/// order. `make_decoder` builds the per-request token decoder (token id -> text piece) when the
/// stream is first polled.
///
/// THE ROLE-CHUNK CONTRACT (S-B9-REL-BENCHFIX / BENCHFIX2). The role chunk is the first chunk
/// with a `choices` entry. It leaves
///   * when the request's first token has been GENERATED (the first `TokEvent::Tok`), whether
///     or not that token produces a delta (a tool call being buffered and a held-back marker
///     prefix produce none until later), or
///   * immediately before the terminal events (error event / finish chunk) of a request that
///     generates no token (an engine error before the first token, a finish without a token,
///     an engine that went away),
/// and never before. So a client that takes "first chunk with choices" as "first output"
/// (llama-benchy's ttfr, most TTFT tooling) reads the time of the first generated token: not a
/// few milliseconds (the role chunk used to be yielded at stream start, before the prefill:
/// 2,048 prompt tokens / 4.6 ms = "443,198 tok/s" in a public benchmark table), and not the end
/// of the generation either (a reply that is only a tool call yields its first delta at the
/// end). The data events keep their bytes and their order: role chunk first, then exactly the
/// deltas, error event, finish chunk, usage chunk and `[DONE]` as before. While the generator
/// is silent the response carries SSE comment frames only (`sse_with_keepalive`).
pub(crate) fn chat_sse_stream<F, D>(cx: ChatSse, make_decoder: F)
    -> impl Stream<Item = Result<Event, axum::Error>> + Send + 'static
where
    F: FnOnce() -> D + Send + 'static,
    D: FnMut(u32) -> String + Send + 'static,
{
    let ChatSse {
        mut rx, bl, mut mt, otel_req, otel_session, role_chunk, completion_id, created, model_name,
        stops, think_open, think_close, starts_in_reasoning, tool_hold, parse_as_tool_calls,
        literal_content, req_tools, include_usage, prompt_len, t0, gen_log, tool_trace, mut scrub, next_turn,
    } = cx;
    let split = move |acc: &str| if literal_content {
        split_constrained(acc, think_close, starts_in_reasoning)
    } else {
        split_think_in(acc, think_open, think_close, starts_in_reasoning)
    };
    // B2: an empty stop string matches at offset 0 of every text (it would empty the reply); the request
    // deserializer drops it already, this keeps the stream itself safe for any caller.
    let stops: Vec<String> = stops.into_iter().filter(|s| !s.is_empty()).collect();
    async_stream::stream! {
        // event=stream_start, token.index 0: recorded when the stream is first polled, as it
        // always was (the telemetry timeline is unchanged). Its body is the role chunk this
        // stream sends with its first token; a stream the client drops before that records
        // the stream_start but never puts the role chunk on the wire.
        if let Some(r) = &otel_req { r.start(&role_chunk); }
        let mut role_pending = Some(role_chunk);
        let mut stream_dec = make_decoder();
        let mut acc = String::new();
        let mut n = 0usize;
        let mut cached = 0usize; // v0.7.3 (#8.2): prefix-cache-served prompt tokens
        let mut last_tok: Option<u32> = None;
        let mut stop_hit = false;
        // [gen] line (item 5): the scheduler's own finish reason and its stop-token statistics
        let mut sched_reason: Option<String> = None;
        let mut stop_info: Option<(f32, u32, bool)> = None;
        // WP02: a channel closed without a Finish is an engine failure, not "length"
        let mut finish = "error: engine stopped".to_string();
        let mut first_tok: Option<std::time::Instant> = None;
        // WP02: tail-window scans (O(new bytes) per token, identical results — see GrowFind)
        let mut f_close = GrowFind::default();
        let mut f_tool_r = GrowFind::default();
        let mut f_open = GrowFind::default();
        let mut f_tool_c = GrowFind::default();
        let mut f_stops: Vec<GrowFind> = stops.iter().map(|_| GrowFind::default()).collect();
        // Thinking-model split: qwen's prompt is primed with `<think>\n`, so the generated stream
        // is `…reasoning…</think>\n\nanswer`. Pre-close text -> reasoning_content, post-close
        // -> content. hy_v3's no_think prompt already closed the (empty) block, so it starts as
        // content. The close marker may span decode chunks, so we hold back a tail that could be
        // its prefix until more text arrives.
        let mut content_start: Option<usize> = if starts_in_reasoning { None } else { Some(0) };
        let mut reason_emitted: usize = 0;
        let mut content_emitted: usize = 0;
        // Markers whose split tail must not leave before the next token decides it: the
        // think-open tag (content sites) and the request's stop strings (every site). A stop
        // string ends the reply BEFORE itself, so a tail that could be the start of one is held.
        let reason_marks: Vec<&str> = stops.iter().map(String::as_str).collect();
        let content_marks: Vec<&str> = std::iter::once(think_open).filter(|_| !literal_content)
            .chain(reason_marks.iter().copied()).collect();
        while let Some(ev) = recv_counted(&mut rx, &bl).await { // LR-5: the one decrement (H0)
            match ev {
                TokEvent::Admitted { cached_tokens } => {
                    cached = cached_tokens as usize;
                    mt.cached(cached_tokens as u64); // /metrics counter
                }
                TokEvent::Tok(t) => {
                    // the request's first generated token: the role chunk leaves now
                    if let Some(r) = role_pending.take() { yield Ok(Event::default().data(r)); }
                    n += 1;
                    mt.tok();
                    last_tok = Some(t);
                    if first_tok.is_none() { first_tok = Some(std::time::Instant::now()); }
                    let text = stream_dec(t);
                    if !text.is_empty() {
                        acc.push_str(&text);
                        // A stop string ends the reply before itself: cut the buffer FIRST, so
                        // the emit sites below never send the stop string or what follows it in
                        // the same token (the check used to come after them: a streamed reply
                        // carried its stop string, and the start of one that spanned tokens,
                        // which the non-streaming response never returns).
                        if !stops.is_empty() {
                            if let Some(p) = stops.iter().zip(f_stops.iter_mut())
                                .filter_map(|(s, f)| f.find(&acc, 0, s)).min() {
                                acc.truncate(p);
                                stop_hit = true;
                                finish = "stop".to_string();
                            }
                        }
                        // (the cut is at or after every emit cursor, because a tail that could
                        // start a stop string is held back at the three hold-back sites. Three
                        // emits have no such hold-back: reasoning before a found close tag,
                        // content before a found open tag, text before a found tool marker. A
                        // stop string that OVERLAPS one of those tags can therefore cut the
                        // buffer below a cursor; the test below then skips the emit sites, and
                        // the text already sent stays sent: the reply differs from the
                        // non-streaming one for such a stop string, and only for such a one)
                        if reason_emitted.max(content_emitted) <= acc.len() && content_start.map_or(true, |cs| cs <= acc.len()) {
                            match content_start {
                                None => {
                                    // Search the close tag from reason_emitted, not from 0:
                                    // a second think block must not match the first one's close.
                                    if let Some(idx) = f_close.find(&acc, reason_emitted, think_close) {
                                        if idx > reason_emitted {
                                            let c = reasoning_chunk(&completion_id, created, &model_name, &scrub.push_reasoning(&acc[reason_emitted..idx]));
                                            if let Some(r) = &otel_req { r.delta(&c); }
                                            yield Ok(Event::default().data(c));
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
                                        // S3: the SAME decision as the other emit sites
                                        // (gated on tool_hold) — a stray <tool_call> in
                                        // the close chunk streams as content when tool
                                        // parsing is off (keep-tools "none").
                                        let safe_end = hold_back_end(
                                            tool_hold,
                                            tool_hold.and_then(|_| region.find(TOOL_OPEN).map(|i| lead + i)),
                                            acc.len(), region, &content_marks);
                                        if safe_end > lead {
                                            let c = content_chunk(&completion_id, created, &model_name, &scrub.push_content(&acc[lead..safe_end]));
                                            if let Some(r) = &otel_req { r.delta(&c); }
                                            yield Ok(Event::default().data(c));
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
                                        // post-loop by held_tail.
                                        let region = &acc[reason_emitted..safe];
                                        let safe_end = hold_back_end(
                                            tool_hold,
                                            tool_hold.and_then(|m| f_tool_r.find(&acc[..safe], reason_emitted, m)),
                                            safe, region, &reason_marks);
                                        if safe_end > reason_emitted {
                                            let c = reasoning_chunk(&completion_id, created, &model_name, &scrub.push_reasoning(&acc[reason_emitted..safe_end]));
                                            if let Some(r) = &otel_req { r.delta(&c); }
                                            yield Ok(Event::default().data(c));
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
                                    // A `<think>` that lies AFTER a held tool-call marker is text of
                                    // the held call (an argument that mentions the tag; independent
                                    // review M2: the raw call XML left as content, then the rest as
                                    // reasoning, then the correct call): only a hit before the marker
                                    // is a think block.
                                    let tool_at = tool_hold.and_then(|m| f_tool_c.find(&acc, cs, m));
                                    if let Some(upto) = f_open.find(&acc, cs, think_open)
                                        .filter(|&u| !literal_content && tool_at.map_or(true, |t| u < t)) {
                                        if upto > content_emitted {
                                            let c = content_chunk(&completion_id, created, &model_name, &scrub.push_content(&acc[content_emitted..upto]));
                                            if let Some(r) = &otel_req { r.delta(&c); }
                                            yield Ok(Event::default().data(c));
                                        }
                                        content_start = None;
                                        reason_emitted = upto + think_open.len();
                                    } else {
                                    // Hold back anything that is, or could become, a tool call.
                                    // Forwarding `<tool_call>` as content makes the harness render
                                    // XML in the chat and never invoke the tool. Same hold-back
                                    // for a think-open prefix spanning decode chunks.
                                    let safe_end = hold_back_end(
                                        tool_hold,
                                        tool_hold.and_then(|m| f_tool_c.find(&acc, cs, m)),
                                        acc.len(), region, &content_marks);
                                    if safe_end > content_emitted {
                                        let c = content_chunk(&completion_id, created, &model_name, &scrub.push_content(&acc[content_emitted..safe_end]));
                                        if let Some(r) = &otel_req { r.delta(&c); }
                                        yield Ok(Event::default().data(c));
                                        content_emitted = safe_end;
                                    }
                                    }
                                }
                            }
                        }
                        }
                    if stop_hit { break; }
                }
                TokEvent::StopInfo { p, rank, sampled } => { stop_info = Some((p, rank, sampled)); }
                TokEvent::Logprob { .. } => {} // refused with stream=true (chat_logprobs_param)
                TokEvent::Finish { reason } => { sched_reason = Some(reason.clone()); finish = reason; break; }
            }
        }
        // WP02: release the lane NOW (a stop-string hit cancels it within one scheduler step)
        drop(rx);
        // no token was generated (an engine error, a finish without a token, the engine gone):
        // the role chunk goes immediately before the terminal events
        if let Some(r) = role_pending.take() { yield Ok(Event::default().data(r)); }
        if finish.starts_with("error") {
            // WP02: a mid-stream engine error is an SSE error event (+ a spec finish_reason below)
            crate::reprintln!("[req] stream ended in an engine error after {n} tokens: {finish}");
            yield Ok(Event::default().data(sse_error_event(&finish)));
        }
        // The call was buffered, not streamed (see the hold-back above). The DECISION is
        // crate::tools::finalize_parsed — the one canonical serializer shared with the
        // non-streaming mode — and the held-back text is surfaced by
        // held_tail, so streaming can never silently drop text the JSON
        // mode returns (2026-08-27 user report: a malformed `function=NAME>` block with the
        // `<` missing vanished from the SSE stream while the JSON response leaked it).
        let (_, done_content) = split(&acc);
        // parse_as_tool_calls == false (keep-tools "none"): never parse — the reply is plain
        // content; finalize_parsed then returns the raw text with finish stop/length.
        let parsed = if parse_as_tool_calls {
            crate::tools::parse_served(&done_content, req_tools.as_deref(), &finish)
        } else {
            crate::tools::ParsedOutput { content: done_content.trim().to_string(), tool_calls: Vec::new() }
        };
        if req_tools.is_some() {
            let dump = crate::opts::var(crate::opt!("dump-tools")).is_ok();
            if dump || parsed.tool_calls.is_empty() {
                crate::dreprintln!("[req] raw model output ({} chars): {:?}", done_content.chars().count(),
                          done_content.chars().take(1200).collect::<String>());
            }
            if dump { crate::reprintln!("{}", dump_tools_line(&done_content)); }
        }
        // Phase-4 A2: same empty-argument alarm as the non-streaming path — an agent harness
        // streams, so this branch is the one that actually gets used.
        for a in crate::tools::empty_arg_alarms(&done_content, req_tools.as_deref(), &parsed.tool_calls) {
            crate::reprintln!("[tool-args-alarm] {a}");
        }
        let (tc_content, tool_calls, fin) = crate::tools::finalize_parsed(&done_content, parsed, &finish);
        // TOOLEVAL: the per-turn tool trace (None = flag off -> one Option check, nothing else).
        if let Some(t) = &tool_trace {
            crate::tool_trace::log(t, &acc, &done_content, tc_content.as_deref(), &tool_calls, &fin);
        }
        mt.finish(if finish.starts_with("error") { finish.as_str() } else { fin.as_str() });
        if !tool_calls.is_empty() {
            // Log the ARGUMENTS, not just the names — see the note on the non-streaming path.
            // Agent harnesses stream, so this is the branch that actually gets used, and it was the
            // one printing a bare `tool_calls 1: ["write"]` while a file silently failed to appear.
            for t in &tool_calls {
                crate::reprintln!("[req] tool_call  {} {}({})", t.id, t.function.name, t.function.arguments);
            }
            // TOOLSCRUB1: flush any text the scrubber still holds (it belongs before the
            // call), then scrub the call arguments themselves. The [req] log above keeps
            // the RAW arguments (same class as --tool-trace / --dump-tools: server-side
            // diagnostics of what the model actually produced).
            let r_flush = scrub.finish_reasoning("");
            if !r_flush.is_empty() {
                let c = reasoning_chunk(&completion_id, created, &model_name, &r_flush);
                if let Some(r) = &otel_req { r.delta(&c); }
                yield Ok(Event::default().data(c));
            }
            let c_flush = scrub.finish_content("");
            if !c_flush.is_empty() {
                let c = content_chunk(&completion_id, created, &model_name, &c_flush);
                if let Some(r) = &otel_req { r.delta(&c); }
                yield Ok(Event::default().data(c));
            }
            let tc_scrubbed: Vec<ToolCall> = tool_calls.iter().map(|t| {
                let mut t2 = t.clone();
                t2.function.arguments = scrub.scrub_full(&t2.function.arguments).into_owned();
                t2
            }).collect();
            let tc = tool_calls_chunk(&completion_id, created, &model_name, &tc_scrubbed);
            if let Some(r) = &otel_req { r.delta(&tc); }
            yield Ok(Event::default().data(tc));
            finish = fin;
        } else if !finish.starts_with("error") {
            // Nothing became a tool call, so whatever the hold-back still keeps is TEXT and
            // leaves now (S-B9-REL-SURFACE2B R5). NOT after an engine error (S-B9-REL-INTEG B5): the
            // error event has gone out, and nothing follows it but the finish chunk, unless the held
            // span holds a COMPLETE call (the branch above: master's rule). The emit sites above hold back a tool-call
            // marker and everything after it, and the tail of the buffer while it could still
            // become a marker (`<`, `<tool`, `<thi`, `</thi` are prefixes of `<tool_call`,
            // `<think>`, `</think>`). The marker case was surfaced here; a tail that was only
            // a PREFIX was not, so a reply that ended in one lost those characters (a reply cut
            // by max_tokens right after a `<`, or simply ending in `<`), and so did the text
            // after a `<tool_call` that was not a call tag (`<tool_call_x`). The non-streaming
            // response returns all of them.
            let (reasoning_tail, content_tail) = held_tail(&acc, reason_emitted.max(content_emitted),
                                                           content_start.is_none(), tool_hold, think_close);
            // TOOLSCRUB1: the tails (and anything the scrubber still holds) leave through
            // the scrubber's finish path; off mode returns them byte-identical.
            let rs = scrub.finish_reasoning(&reasoning_tail);
            if !rs.is_empty() {
                let c = reasoning_chunk(&completion_id, created, &model_name, &rs);
                if let Some(r) = &otel_req { r.delta(&c); }
                yield Ok(Event::default().data(c));
            }
            let cs = scrub.finish_content(&content_tail);
            if !cs.is_empty() {
                let c = content_chunk(&completion_id, created, &model_name, &cs);
                if let Some(r) = &otel_req { r.delta(&c); }
                yield Ok(Event::default().data(c));
            }
            content_emitted = acc.len();
        }
        // TURNRESUME: the reply as the client will send it back (content + the calls it received) -> the scheduler
        if let Some(nt) = next_turn {
            if !finish.starts_with("error") && !stop_hit {
                let (r_txt, c_txt) = split(&acc);
                let content = if tool_calls.is_empty() { c_txt } else { tc_content.clone().unwrap_or_default() };
                let calls: Vec<ToolCall> = tool_calls.iter().map(|t| {
                    let mut t2 = t.clone();
                    t2.function.arguments = scrub.scrub_full(&t2.function.arguments).into_owned();
                    t2
                }).collect();
                nt.post(r_txt, content, calls);
            }
        }
        {
            let (r_txt, c_txt) = split(&acc);
            let (r_log, c_log) = gen_log_spans(r_txt.as_deref(), &c_txt, acc.contains(think_close), starts_in_reasoning);
            if let Some(g) = &gen_log {
                log_generation(&g.tokenizer, &g.stop_ids, &completion_id, &finish,
                               last_tok, prompt_len, n, r_log, c_log,
                               tool_calls.len(), req_tools.as_ref().map(|t| t.len()).unwrap_or(0),
                               g.effort.as_deref(), g.presence_penalty, g.pp_source,
                               stop_kind(sched_reason.as_deref(), stop_hit), stop_info);
            }
        }
        // WP08: `stop_reason` only when the loop detector ended the stream (else unchanged).
        let stop_reason_field = stop_reason_of(&finish)
            .map(|r| format!(",\"stop_reason\":\"{r}\"")).unwrap_or_default();
        let final_chunk = final_chunk(&completion_id, created, &model_name,
            spec_finish_reason(&finish), &stop_reason_field); // S8: esc(model)
        if let Some(r) = &otel_req { r.end(&final_chunk); }   // event=stream_end (carries finish_reason)
        yield Ok(Event::default().data(final_chunk));
        if include_usage {
            // Spec stream-usage chunk: empty choices, top-level usage. `timings` rides along
            // as the extension field (strict clients ignore it), and `session_id` echoes the
            // OTel session key so the client can label the stream without computing anything.
            let mut usage_chunk = serde_json::json!({
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model_name, "choices": [],
                "usage": {"prompt_tokens": prompt_len, "completion_tokens": n,
                          "total_tokens": prompt_len + n,
                          "prompt_tokens_details": {"cached_tokens": cached}},
                "timings": make_timings(t0, first_tok, prompt_len, n),
            });
            if let Some(s) = &otel_session { usage_chunk["session_id"] = serde_json::json!(s); }
            yield Ok(Event::default().data(usage_chunk.to_string()));
        }
        // The OpenAI SSE terminator. Without it a strict client sits on an open stream
        // waiting for more events after the finish chunk.
        yield Ok(Event::default().data("[DONE]"));
        let dt = t0.elapsed().as_secs_f32();
        crate::reprintln!("[req] done   tok={} ({:.1} tok/s wall) finish={} stop_hit={}", n, if dt>1e-6 {n as f32/dt} else {0.0}, finish, stop_hit);
    }
}

