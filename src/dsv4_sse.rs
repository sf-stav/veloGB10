//! The DSV4 server's blocking SSE writer (`--server-dspark`, src/main.rs `dsv4_tp_serve`).
//! Moved here from src/main.rs by S-B9-REL-BENCHFIX2 (W1) so the unit suite (`--lib`) runs the
//! code the server runs: the streamer writes to any `Write` and decodes through a closure.
//!
//! THE ROLE-CHUNK RULE (the same as src/server.rs `chat_sse_stream`): the role chunk, the first
//! chunk with a `choices` entry, is written when the request's first token has been generated
//! (the first `feed` of a non-EOS token), or immediately before the finish chunk when none is
//! generated; never before. The DSpark branch used to write it right after the response
//! headers, BEFORE `decode_request` (which begins with the prefill), so a client that takes the
//! first chunk with `choices` as the first output measured a prefill of a few milliseconds.
//!
//! No keep-alive here: the DSV4 server handles one connection at a time on the thread that
//! drives the model, and its prefill is one blocking call; nothing can write to the socket
//! during it without a second writer. `--sse-keepalive` is not read by this server.

use std::io::Write;

/// Write one SSE frame (`data: <json>\r\n\r\n`) and flush. Used by both DSV4 streaming paths
/// (the DSpark streamer below and the greedy loop in src/main.rs).
pub fn dsv4_sse_chunk<W: Write>(sock: &mut W, json: &str) -> std::io::Result<()> {
    sock.write_all(b"data: ")?;
    sock.write_all(json.as_bytes())?;
    sock.write_all(b"\r\n\r\n")?;
    sock.flush()
}

/// Longest suffix of `s` that is a proper (partial) prefix of `marker` — mirrors
/// src/server.rs:partial_overlap. Held back during streaming so a think-close marker arriving
/// across decode chunks is not partially forwarded as reasoning_content.
pub fn dsv4_partial_overlap(s: &str, marker: &str) -> usize {
    (1..marker.len()).rev().find(|&k| s.ends_with(&marker[..k])).unwrap_or(0)
}

/// Item 3.4 — SSE streaming for the DSpark server path (the tool-eval harness streams every
/// perf request and turn 1 of every scenario, so a DSpark server must stream too). Mirrors the
/// greedy stream path's per-token reasoning→content split (THINKING_END_TOKEN boundary, partial
/// close-tag hold-back) — one SSE delta per generated TOKEN (not per accept-burst), so the
/// harness's token-per-chunk counting stays honest. `decode` maps one token id to its text
/// (None = not decodable: the token is counted and produces no delta).
pub struct Dsv4SseStreamer<'a, W: Write, D: Fn(u32) -> Option<String>> {
    decode: D,
    sock: &'a mut W,
    model_id: &'a str,
    role_sent: bool,
    acc: String,
    reason_emitted: usize,
    content_emitted: usize,
    content_start: Option<usize>,
    t0: std::time::Instant,
    first_tok: Option<std::time::Instant>,
    n_emitted: usize,
    include_usage: bool,
}

impl<'a, W: Write, D: Fn(u32) -> Option<String>> Dsv4SseStreamer<'a, W, D> {
    pub fn new(
        decode: D,
        sock: &'a mut W,
        raw_completion: bool,
        thinking_mode: &str,
        model_id: &'a str,
        include_usage: bool,
        t0: std::time::Instant,
    ) -> Self {
        // thinking mode starts INSIDE the think block (prompt primed `imd`); raw completion
        // and chat mode are all content (no reasoning split).
        let content_start = if !raw_completion && thinking_mode == "thinking" { None } else { Some(0) };
        Self { decode, sock, model_id, role_sent: false, acc: String::new(),
               reason_emitted: 0, content_emitted: 0, content_start,
               t0, first_tok: None, n_emitted: 0, include_usage }
    }

    /// The role chunk, once: with the first generated token, or before the finish chunk.
    fn role_once(&mut self) -> anyhow::Result<()> {
        if self.role_sent { return Ok(()); }
        self.role_sent = true;
        let role = serde_json::json!({"id":"dsv4-tp-1","object":"chat.completion.chunk","model":self.model_id,
            "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]});
        dsv4_sse_chunk(&mut self.sock, &serde_json::to_string(&role)?)?;
        Ok(())
    }

    /// Feed one generated token (the EOS token itself is not emitted — mirrors the greedy
    /// stream path's break-before-push).
    pub fn feed(&mut self, t: u32) -> anyhow::Result<()> {
        if t == 1 { return Ok(()); } // eos
        self.role_once()?; // the request's first generated token: the role chunk leaves now
        if self.first_tok.is_none() { self.first_tok = Some(std::time::Instant::now()); }
        self.n_emitted += 1;
        let think_close = crate::dsv4_chat::THINKING_END_TOKEN;
        let Some(piece) = (self.decode)(t) else { return Ok(()) };
        if piece.is_empty() { return Ok(()); }
        self.acc.push_str(&piece);
        match self.content_start {
            None => {
                if let Some(idx) = self.acc[self.reason_emitted..].find(think_close).map(|i| self.reason_emitted + i) {
                    if idx > self.reason_emitted {
                        let d = serde_json::json!({"id":"dsv4-tp-1","object":"chat.completion.chunk","model":self.model_id,
                            "choices":[{"index":0,"delta":{"reasoning_content":&self.acc[self.reason_emitted..idx]},"finish_reason":null}]});
                        dsv4_sse_chunk(&mut self.sock, &serde_json::to_string(&d)?)?;
                    }
                    let cs = idx + think_close.len();
                    let mut lead = cs;
                    while lead < self.acc.len() && matches!(self.acc.as_bytes()[lead], b'\n'|b'\r'|b' '|b'\t') { lead += 1; }
                    if lead < self.acc.len() {
                        let d = serde_json::json!({"id":"dsv4-tp-1","object":"chat.completion.chunk","model":self.model_id,
                            "choices":[{"index":0,"delta":{"content":&self.acc[lead..self.acc.len()]},"finish_reason":null}]});
                        dsv4_sse_chunk(&mut self.sock, &serde_json::to_string(&d)?)?;
                    }
                    self.content_start = Some(lead);
                    self.content_emitted = self.acc.len();
                } else {
                    let overlap = dsv4_partial_overlap(&self.acc, think_close);
                    let safe = self.acc.len() - overlap;
                    if safe > self.reason_emitted {
                        let d = serde_json::json!({"id":"dsv4-tp-1","object":"chat.completion.chunk","model":self.model_id,
                            "choices":[{"index":0,"delta":{"reasoning_content":&self.acc[self.reason_emitted..safe]},"finish_reason":null}]});
                        dsv4_sse_chunk(&mut self.sock, &serde_json::to_string(&d)?)?;
                        self.reason_emitted = safe;
                    }
                }
            }
            Some(_) => {
                if self.acc.len() > self.content_emitted {
                    let d = serde_json::json!({"id":"dsv4-tp-1","object":"chat.completion.chunk","model":self.model_id,
                        "choices":[{"index":0,"delta":{"content":&self.acc[self.content_emitted..self.acc.len()]},"finish_reason":null}]});
                    dsv4_sse_chunk(&mut self.sock, &serde_json::to_string(&d)?)?;
                    self.content_emitted = self.acc.len();
                }
            }
        }
        Ok(())
    }

    /// Flush the held-back tail, the final chunk, usage chunk (if enabled), and the SSE terminator.
    pub fn finish(&mut self, hit_eos: bool, prompt_len: usize) -> anyhow::Result<()> {
        self.role_once()?; // no token was generated: the role chunk goes before the finish chunk
        match self.content_start {
            Some(_) => {
                if self.acc.len() > self.content_emitted {
                    let d = serde_json::json!({"id":"dsv4-tp-1","object":"chat.completion.chunk","model":self.model_id,
                        "choices":[{"index":0,"delta":{"content":&self.acc[self.content_emitted..self.acc.len()]},"finish_reason":null}]});
                    dsv4_sse_chunk(&mut self.sock, &serde_json::to_string(&d)?)?;
                }
            }
            None => {
                if self.acc.len() > self.reason_emitted {
                    let d = serde_json::json!({"id":"dsv4-tp-1","object":"chat.completion.chunk","model":self.model_id,
                        "choices":[{"index":0,"delta":{"reasoning_content":&self.acc[self.reason_emitted..self.acc.len()]},"finish_reason":null}]});
                    dsv4_sse_chunk(&mut self.sock, &serde_json::to_string(&d)?)?;
                }
            }
        }
        let finish = if hit_eos { "stop" } else { "length" };
        let final_chunk = serde_json::json!({"id":"dsv4-tp-1","object":"chat.completion.chunk","model":self.model_id,
            "choices":[{"index":0,"delta":{},"finish_reason":finish}]});
        dsv4_sse_chunk(&mut self.sock, &serde_json::to_string(&final_chunk)?)?;
        if self.include_usage {
            let timings = crate::make_timings(self.t0, self.first_tok, prompt_len, self.n_emitted);
            let usage_chunk = serde_json::json!({
                "id": "dsv4-tp-1", "object": "chat.completion.chunk", "model": self.model_id,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": self.n_emitted,
                    "total_tokens": prompt_len + self.n_emitted,
                },
                "timings": {
                    "prompt_ms": timings.prompt_ms,
                    "predicted_ms": timings.predicted_ms,
                    "prompt_per_second": timings.prompt_per_second,
                    "predicted_per_second": timings.predicted_per_second,
                },
            });
            dsv4_sse_chunk(&mut self.sock, &serde_json::to_string(&usage_chunk)?)?;
        }
        self.sock.write_all(b"data: [DONE]\r\n\r\n")?;
        self.sock.flush()?;
        Ok(())
    }
}

/// S-B9-REL-BENCHFIX2 (W1). The streamer the DSV4 server runs, written to a buffer and read
/// back. UNVERIFIED ON GPU: a DSV4 serve run needs its model and a TP=2 pair (the exact check
/// is in PLAN/REL_BENCHFIX_REPORT.md, pair-window checklist).
#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: &str = "deepseek-v4-flash";
    /// The role chunk, built the way src/main.rs built it before the move.
    fn role() -> String {
        serde_json::to_string(&serde_json::json!({"id":"dsv4-tp-1","object":"chat.completion.chunk","model":MODEL,
            "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]})).unwrap()
    }

    /// token id -> text (1 = EOS and is never decoded; 9 = not decodable)
    fn decode(t: u32) -> Option<String> {
        match t {
            2 => Some("Hello".into()), 3 => Some(" world".into()), 4 => Some("plan".into()),
            5 => Some(crate::dsv4_chat::THINKING_END_TOKEN.to_string()), 6 => Some(String::new()),
            _ => None,
        }
    }

    /// The `data:` payloads written so far (every frame is `data: <payload>\r\n\r\n`).
    fn payloads(wire: &[u8]) -> Vec<String> {
        let text = std::str::from_utf8(wire).unwrap();
        assert!(text.is_empty() || text.ends_with("\r\n\r\n"), "whole frames only: {text:?}");
        text.split_terminator("\r\n\r\n").map(|f| f.strip_prefix("data: ").expect("a data frame").to_string()).collect()
    }

    fn delta(key: &str, text: &str) -> String {
        serde_json::to_string(&serde_json::json!({"id":"dsv4-tp-1","object":"chat.completion.chunk","model":MODEL,
            "choices":[{"index":0,"delta":{key:text},"finish_reason":null}]})).unwrap()
    }

    fn finish_chunk(reason: &str) -> String {
        serde_json::to_string(&serde_json::json!({"id":"dsv4-tp-1","object":"chat.completion.chunk","model":MODEL,
            "choices":[{"index":0,"delta":{},"finish_reason":reason}]})).unwrap()
    }

    /// THE defect: nothing is written when the streamer is created (the DSpark branch creates
    /// it, then runs the prefill inside `decode_request`); the role chunk is the first frame of
    /// the first generated token, followed by that token's delta.
    #[test]
    fn nothing_is_written_before_the_first_generated_token() {
        let mut wire: Vec<u8> = Vec::new();
        let mut s = Dsv4SseStreamer::new(decode, &mut wire, false, "chat", MODEL, false, std::time::Instant::now());
        assert!(s.sock.is_empty(), "the prefill runs now: no byte, so no `choices` chunk, is on the wire");
        s.feed(2).unwrap();
        let first: serde_json::Value = serde_json::from_str(&payloads(&s.sock[..])[0]).unwrap();
        assert_eq!(first["choices"][0]["delta"], serde_json::json!({"role": "assistant"}));
        assert_eq!(payloads(&s.sock[..]), [role(), delta("content", "Hello")]);
        s.feed(3).unwrap();
        s.feed(1).unwrap(); // EOS: not emitted
        s.finish(true, 11).unwrap();
        assert_eq!(payloads(&wire), [role(), delta("content", "Hello"), delta("content", " world"),
                                     finish_chunk("stop"), "[DONE]".to_string()]);
    }

    /// The role chunk leaves at the first generated TOKEN even when that token writes no delta
    /// (no text, or not decodable), and exactly once.
    #[test]
    fn the_role_chunk_does_not_wait_for_the_first_delta() {
        for first in [6u32, 9] {
            let mut wire: Vec<u8> = Vec::new();
            let mut s = Dsv4SseStreamer::new(decode, &mut wire, false, "chat", MODEL, false, std::time::Instant::now());
            s.feed(first).unwrap();
            assert_eq!(payloads(&s.sock[..]), [role()], "first token {first}");
            s.feed(2).unwrap();
            s.finish(false, 11).unwrap();
            assert_eq!(payloads(&wire), [role(), delta("content", "Hello"), finish_chunk("length"), "[DONE]".to_string()]);
        }
    }

    /// A request that generates no token (EOS first, or nothing at all): role, finish, [DONE].
    #[test]
    fn a_request_without_a_token_sends_the_role_chunk_before_the_finish_chunk() {
        for eos_first in [true, false] {
            let mut wire: Vec<u8> = Vec::new();
            let mut s = Dsv4SseStreamer::new(decode, &mut wire, false, "chat", MODEL, false, std::time::Instant::now());
            if eos_first {
                s.feed(1).unwrap();
                assert!(s.sock.is_empty(), "EOS is not a generated token: still nothing on the wire");
            }
            s.finish(eos_first, 11).unwrap();
            assert_eq!(payloads(&wire), [role(), finish_chunk(if eos_first { "stop" } else { "length" }), "[DONE]".to_string()]);
        }
    }

    /// Thinking mode: role, reasoning deltas, then content after the close marker; with usage.
    /// The data frames after the role chunk are the bytes the streamer always wrote.
    #[test]
    fn thinking_stream_and_usage_keep_their_frames() {
        let mut wire: Vec<u8> = Vec::new();
        let mut s = Dsv4SseStreamer::new(decode, &mut wire, false, "thinking", MODEL, true, std::time::Instant::now());
        for t in [4u32, 5, 2, 3, 1] { s.feed(t).unwrap(); }
        s.finish(true, 11).unwrap();
        let p = payloads(&wire);
        assert_eq!(p[..5], [role(), delta("reasoning_content", "plan"), delta("content", "Hello"),
                            delta("content", " world"), finish_chunk("stop")]);
        let usage: serde_json::Value = serde_json::from_str(&p[5]).unwrap();
        assert_eq!(usage["choices"], serde_json::json!([]));
        assert_eq!(usage["usage"], serde_json::json!({"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}));
        assert_eq!(p[6], "[DONE]");
        assert_eq!(p.len(), 7);
        // raw completion: no reasoning split even in thinking mode
        let mut wire: Vec<u8> = Vec::new();
        let mut s = Dsv4SseStreamer::new(decode, &mut wire, true, "thinking", MODEL, false, std::time::Instant::now());
        s.feed(4).unwrap();
        s.finish(false, 3).unwrap();
        assert_eq!(payloads(&wire)[..2], [role(), delta("content", "plan")]);
    }

    /// The link from the server to this module (source pin): in src/main.rs the DSpark streaming
    /// branch writes the response head, builds the streamer and calls `decode_request`, with no
    /// chunk written in between; the greedy branch's role chunk comes after its prefill.
    #[test]
    fn the_dsv4_server_writes_no_chunk_before_decode_request() {
        let src = src_text!("main");
        let a = src.find("if let Some(st) = dspark_state.as_mut() {\n            let _gen = if stream {").expect("the DSpark streaming branch");
        let branch = &src[a..];
        let call = branch.find("st.decode_request(").unwrap();
        let before = &branch[..call];
        assert_eq!(before.matches("sock.write_all(head.as_bytes())?;").count(), 1, "the response head");
        assert_eq!(before.matches("Dsv4SseStreamer::new(decode, &mut sock,").count(), 1);
        assert_eq!(before.matches(&format!("{}(", "dsv4_sse_chunk")).count(), 0, "no chunk before the prefill");
        assert_eq!(before.matches("\"choices\"").count(), 0);
        // greedy branch: prefill (`forward_prefix_cached`) first, the role chunk after it
        let g = src.find("let m = m.as_mut().expect(\"greedy server owns the model\");").unwrap();
        let greedy = &src[g..];
        let prefill = greedy.find("m.forward_prefix_cached").unwrap();
        let role = greedy.find("\"delta\":{\"role\":\"assistant\"}").unwrap();
        assert!(prefill < role, "the greedy role chunk is written after its prefill");
    }

    /// S-B9-REL-SURFACE2B R6(f): the `timings` block of the DSV4 greedy path. `prompt_ms` is
    /// request -> first token, so the request clock must start BEFORE the prefill. It was taken
    /// after it (both greedy branches): `prompt_per_second` was the prompt length divided by the
    /// few microseconds between two statements, the same class as the public benchmark defect
    /// ("443,198 tok/s"). UNVERIFIED ON GPU: the DSV4 server needs the pair.
    #[test]
    fn the_dsv4_greedy_timings_start_before_the_prefill() {
        // what the block computes: 2,000 prompt tokens, first token 400 ms after the request
        let t0 = std::time::Instant::now();
        let first = t0 + std::time::Duration::from_millis(400);
        let t = crate::make_timings(t0, Some(first), 2_000, 16);
        assert!((t.prompt_ms - 400.0).abs() < 1e-6 && (t.prompt_per_second - 5_000.0).abs() < 1e-6, "{t:?}");
        // ...and what the old clock gave: the "request" starts when the first token exists
        let late = crate::make_timings(first, Some(first + std::time::Duration::from_micros(3)), 2_000, 16);
        assert!(late.prompt_per_second > 1e8, "prompt tokens over microseconds: {late:?}");
        // source pin (src/main.rs is the binary): in the greedy path ONE clock is started, before
        // the prefill, and both of its `timings` blocks read that clock
        let src = src_text!("main");
        let g = src.find("let m = m.as_mut().expect(\"greedy server owns the model\");").unwrap();
        let end = g + src[g..].find("\nfn ").expect("the end of dsv4_tp_serve_server");
        let greedy = &src[g..end];
        let clock = ["let t0 = ", "std::time::Instant::now();"].concat();
        assert_eq!(greedy.matches(&clock).count(), 1, "one request clock in the greedy path");
        let (at_clock, at_prefill) = (greedy.find(&clock).unwrap(), greedy.find("m.forward_prefix_cached").unwrap());
        assert!(at_clock < at_prefill, "the clock starts before the prefill");
        assert_eq!(greedy.matches(&["make_timings(", "t0, first_tok, prompt_len,"].concat()).count(), 2, "streaming and non-streaming");
        // the DSpark branch keeps its own clocks, each started before `decode_request` (the prefill)
        let d = src.find("if let Some(st) = dspark_state.as_mut() {").unwrap();
        let dspark = &src[d..g];
        assert_eq!(dspark.matches(&clock).count(), 2);
        for (i, _) in dspark.match_indices(&clock) {
            let next_decode = dspark[i..].find("st.decode_request(").expect("a decode_request after each clock");
            assert!(!dspark[i..i + next_decode].contains("make_timings"), "the clock precedes its decode_request");
        }
    }
}
