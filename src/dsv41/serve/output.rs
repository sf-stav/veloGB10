//! Tokens in, reply parts out: the byte stream of one reply through ENC's DSML parser (reasoning / content / calls)
//! and the client's stop strings, plus the OpenAI JSON builders for chunks and whole responses.
//!
//! Rules kept here (the streaming-hygiene rows of the UX addendum):
//! * a client stop string is matched in CONTENT only and is never streamed: text that could be the start of one is held
//!   back until it is settled, and a match ends the reply (the caller cancels the lane);
//! * a tool call is reported whole when its invoke closes (ENC streams calls whole); a block the catalogue rejects stays
//!   visible as content (the 2026-10-10 hardening rule: never invent a call);
//! * the reply's bytes are decoded by the parser, not by a lossy per-token decode, so a character split across tokens
//!   is whole in the output.

use crate::dsv41::chat::catalogue::Catalogue;
use crate::dsv41::chat::stream::{DsmlStream, Event, ParseOptions, ParsedCall, Parsed};
use crate::tokenizer::QwenTokenizer;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum Delta {
    Reasoning(String),
    Content(String),
    Call { index: usize, call: ParsedCall },
}

pub struct Assembler {
    parser: DsmlStream,
    stops: Vec<String>,
    held: String,
    stopped: bool,
    n_calls: usize,
    pub reasoning: String,
    pub content: String,
    pub calls: Vec<ParsedCall>,
}

impl Assembler {
    pub fn new(opts: ParseOptions, catalogue: Option<Catalogue>, stops: Vec<String>) -> Self {
        Assembler { parser: DsmlStream::new(opts, catalogue), stops, held: String::new(), stopped: false, n_calls: 0,
                    reasoning: String::new(), content: String::new(), calls: Vec::new() }
    }

    /// A client stop string matched: the reply is over (nothing after it is reported).
    pub fn stopped(&self) -> bool { self.stopped }

    /// One committed token's bytes (special tokens such as `<think>` and `｜DSML｜` are their literal text).
    pub fn push_token(&mut self, tok: &QwenTokenizer, id: u32) -> Vec<Delta> {
        if self.stopped { return Vec::new(); }
        let bytes = tok.token_bytes(id, false);
        let evs = self.parser.push_bytes(&bytes);
        self.convert(evs, false)
    }

    /// The reply ended (EOS, length, or a stop string): flush what is held.
    pub fn finish(&mut self) -> Vec<Delta> {
        if self.stopped { return Vec::new(); }
        let evs = self.parser.finish();
        self.convert(evs, true)
    }

    pub fn parsed(&self) -> &Parsed { self.parser.parsed() }

    fn convert(&mut self, evs: Vec<Event>, flush: bool) -> Vec<Delta> {
        let mut out = Vec::new();
        for ev in evs {
            if self.stopped { break; }
            match ev {
                Event::Reasoning(s) => if !s.is_empty() { self.reasoning.push_str(&s); out.push(Delta::Reasoning(s)); },
                Event::Content(s) => self.content_text(&s, &mut out),
                Event::Call(c) => {
                    // text held for a stop string precedes the call: settle it first
                    self.settle(true, &mut out);
                    self.calls.push(c.clone());
                    out.push(Delta::Call { index: self.n_calls, call: c });
                    self.n_calls += 1;
                }
            }
        }
        if flush { self.settle(true, &mut out); }
        out
    }

    fn content_text(&mut self, s: &str, out: &mut Vec<Delta>) {
        if self.stops.is_empty() {
            if !s.is_empty() { self.content.push_str(s); out.push(Delta::Content(s.to_string())); }
            return;
        }
        self.held.push_str(s);
        self.settle(false, out);
    }

    /// Emit the held text that can no longer become a stop string (all of it when `all`).
    fn settle(&mut self, all: bool, out: &mut Vec<Delta>) {
        if self.stops.is_empty() || self.held.is_empty() { return; }
        // the earliest full match ends the reply
        let hit = self.stops.iter().filter_map(|st| self.held.find(st.as_str())).min();
        if let Some(i) = hit {
            let keep = self.held[..i].to_string();
            self.held.clear();
            self.stopped = true;
            if !keep.is_empty() { self.content.push_str(&keep); out.push(Delta::Content(keep)); }
            return;
        }
        let hold = if all { 0 } else { self.stops.iter().map(|st| partial_overlap(&self.held, st)).max().unwrap_or(0) };
        let cut = self.held.len() - hold;
        if cut > 0 {
            let emit: String = self.held.drain(..cut).collect();
            self.content.push_str(&emit);
            out.push(Delta::Content(emit));
        }
    }
}

/// Longest suffix of `s` that is a proper prefix of `marker` (so it must be held back).
pub fn partial_overlap(s: &str, marker: &str) -> usize {
    let mut best = 0;
    for (k, _) in marker.char_indices().skip(1) {
        if s.ends_with(&marker[..k]) { best = k; }
    }
    best
}

// ---------------------------------------------------------------------------------------------------------------------
// JSON builders
// ---------------------------------------------------------------------------------------------------------------------

pub fn new_id(prefix: &str) -> String { format!("{prefix}{}", uuid::Uuid::new_v4().simple()) }

fn call_json(id: &str, c: &ParsedCall) -> Value {
    let mut v = json!({"id": id, "type": "function", "function": {"name": c.qualified_name(), "arguments": c.arguments}});
    if let Some(ns) = &c.namespace { v["namespace"] = Value::String(ns.clone()); }
    v
}

fn chunk(cid: &str, created: i64, model: &str, delta: Value, finish: Value) -> String {
    json!({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}).to_string()
}

pub fn role_chunk(cid: &str, created: i64, model: &str) -> String { chunk(cid, created, model, json!({"role": "assistant"}), Value::Null) }
pub fn reasoning_chunk(cid: &str, created: i64, model: &str, t: &str) -> String { chunk(cid, created, model, json!({"reasoning_content": t}), Value::Null) }
pub fn content_chunk(cid: &str, created: i64, model: &str, t: &str) -> String { chunk(cid, created, model, json!({"content": t}), Value::Null) }
pub fn call_chunk(cid: &str, created: i64, model: &str, index: usize, call_id: &str, c: &ParsedCall) -> String {
    let mut tc = call_json(call_id, c);
    tc["index"] = json!(index);
    chunk(cid, created, model, json!({"tool_calls": [tc]}), Value::Null)
}
pub fn final_chunk(cid: &str, created: i64, model: &str, finish: &str) -> String { chunk(cid, created, model, json!({}), json!(finish)) }
pub fn usage_chunk(cid: &str, created: i64, model: &str, u: &Usage) -> String {
    json!({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [], "usage": u.json()}).to_string()
}

// the `/v1/completions` shapes (`text_completion`): the same events, no role chunk and no deltas
fn text_chunk(cid: &str, created: i64, model: &str, t: &str, finish: Value) -> String {
    json!({"id": cid, "object": "text_completion", "created": created, "model": model,
           "choices": [{"text": t, "index": 0, "logprobs": null, "finish_reason": finish}]}).to_string()
}
pub fn text_delta_chunk(cid: &str, created: i64, model: &str, t: &str) -> String { text_chunk(cid, created, model, t, Value::Null) }
pub fn text_final_chunk(cid: &str, created: i64, model: &str, finish: &str) -> String { text_chunk(cid, created, model, "", json!(finish)) }
pub fn text_usage_chunk(cid: &str, created: i64, model: &str, u: &Usage) -> String {
    json!({"id": cid, "object": "text_completion", "created": created, "model": model, "choices": [], "usage": u.json()}).to_string()
}
/// The non-streaming `/v1/completions` body: `text` is the whole reply (the echoed prompt first when asked).
pub fn text_completion_json(cid: &str, created: i64, model: &str, text: &str, finish: &str, u: &Usage) -> String {
    json!({"id": cid, "object": "text_completion", "created": created, "model": model,
           "choices": [{"text": text, "index": 0, "logprobs": null, "finish_reason": finish}], "usage": u.json()}).to_string()
}

#[derive(Clone, Debug, Default)]
pub struct Usage { pub prompt: usize, pub completion: usize, pub cached: usize }
impl Usage {
    pub fn json(&self) -> Value {
        json!({"prompt_tokens": self.prompt, "completion_tokens": self.completion, "total_tokens": self.prompt + self.completion,
               "prompt_tokens_details": {"cached_tokens": self.cached}})
    }
}

/// The non-streaming response body.
pub fn completion_json(cid: &str, created: i64, model: &str, a: &Assembler, call_ids: &[String], finish: &str, u: &Usage) -> String {
    let mut msg = json!({"role": "assistant"});
    msg["content"] = if a.content.is_empty() && !a.calls.is_empty() { Value::Null } else { Value::String(a.content.clone()) };
    if !a.reasoning.is_empty() { msg["reasoning_content"] = Value::String(a.reasoning.clone()); }
    if !a.calls.is_empty() {
        msg["tool_calls"] = Value::Array(a.calls.iter().zip(call_ids).map(|(c, id)| call_json(id, c)).collect());
    }
    json!({"id": cid, "object": "chat.completion", "created": created, "model": model,
           "choices": [{"index": 0, "message": msg, "finish_reason": finish, "logprobs": null}], "usage": u.json()}).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_overlap_is_the_longest_proper_prefix_suffix() {
        assert_eq!(partial_overlap("abc ST", "STOP"), 2);
        assert_eq!(partial_overlap("abc STOP", "STOP"), 0, "a full match is not partial");
        assert_eq!(partial_overlap("abc", "STOP"), 0);
        assert_eq!(partial_overlap("ab\u{e9}", "\u{e9}x"), 2, "multi-byte prefix");
    }

    fn asm(stops: &[&str]) -> Assembler {
        Assembler::new(ParseOptions::new(false), None, stops.iter().map(|s| s.to_string()).collect())
    }

    fn feed(a: &mut Assembler, parts: &[&str]) -> (String, bool) {
        let mut text = String::new();
        for p in parts {
            for ev in a.parser.push_bytes(p.as_bytes()) {
                let mut out = Vec::new();
                if let Event::Content(s) = ev { a.content_text(&s, &mut out); }
                for d in out { if let Delta::Content(s) = d { text.push_str(&s); } }
            }
        }
        let mut tail = Vec::new();
        a.settle(true, &mut tail);
        for d in tail { if let Delta::Content(s) = d { text.push_str(&s); } }
        (text, a.stopped())
    }

    #[test]
    fn a_stop_string_split_across_chunks_is_never_streamed() {
        let mut a = asm(&["STOP"]);
        let (t, stopped) = feed(&mut a, &["hello ST", "O", "P and more"]);
        assert_eq!(t, "hello ");
        assert!(stopped);
        assert_eq!(a.content, "hello ");
    }

    #[test]
    fn held_text_that_does_not_complete_a_stop_is_released() {
        let mut a = asm(&["STOP"]);
        let (t, stopped) = feed(&mut a, &["hello ST", "ART here"]);
        assert_eq!(t, "hello START here");
        assert!(!stopped);
    }

    #[test]
    fn the_earliest_of_several_stops_wins() {
        let mut a = asm(&["zz", "b"]);
        let (t, stopped) = feed(&mut a, &["aaabzz"]);
        assert_eq!(t, "aaa");
        assert!(stopped);
    }
}
