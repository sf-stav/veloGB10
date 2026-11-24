//! The reply's tokens through the parser to OpenAI shapes: `Assembler` (token ids in, deltas out), the non-streamed
//! `chat.completion` JSON and the streamed `chat.completion.chunk` frames. Structural tokens are told by id; ordinary tokens decode
//! through the shared byte-level stream decoder (a multi-byte character split across tokens is held until whole).

use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Value};

use super::super::chat::catalogue::Catalogue;
use super::super::chat::parse::{Event, Finish, ParseError, ParseOptions, ReplyParser, Stats, Tok};
use super::super::chat::special::{is_reserved, TABLE};
use super::forward::End;
use super::metrics as m;
use crate::tokenizer::{QwenTokenizer, StreamByteDecoder};

static CALL_SEQ: AtomicU64 = AtomicU64::new(0);

/// One thing the client sees, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum Delta { Reasoning(String), Content(String), Call { index: usize, id: String, name: String, arguments: String } }

pub struct Assembler {
    parser: ReplyParser,
    dec: StreamByteDecoder,
    calls: usize,
    pub tokens: usize,
}

fn is_structural(id: u32) -> bool { is_reserved(id) || TABLE.iter().any(|(_, i)| *i == id) }

impl Assembler {
    pub fn new(tok: &QwenTokenizer, cat: Catalogue, opt: ParseOptions) -> Self {
        Self { parser: ReplyParser::new(cat, opt), dec: tok.stream_decoder(), calls: 0, tokens: 0 }
    }
    fn lift(&mut self, evs: Vec<Event>) -> Vec<Delta> {
        evs.into_iter().map(|e| match e {
            Event::Reasoning(s) => Delta::Reasoning(s),
            Event::Content(s) => Delta::Content(s),
            Event::Call(c) => {
                let index = self.calls; self.calls += 1;
                Delta::Call { index, id: format!("call_{}", CALL_SEQ.fetch_add(1, Ordering::Relaxed)), name: c.name, arguments: c.arguments }
            }
        }).collect()
    }
    pub fn push(&mut self, id: u32) -> Result<Vec<Delta>, ParseError> {
        self.tokens += 1;
        let evs = if is_structural(id) { self.parser.feed(Tok::Special(id))? } else {
            let t = self.dec.push(id);
            if t.is_empty() { Vec::new() } else { self.parser.feed(Tok::Text(&t))? }
        };
        Ok(self.lift(evs))
    }
    pub fn finish(&mut self, end: End) -> Vec<Delta> {
        let evs = self.parser.finish(if end == End::Length { Finish::Length } else { Finish::Stop });
        self.lift(evs)
    }
    pub fn abort(&mut self) { self.parser.abort(); }
    pub fn stats(&self) -> &Stats { &self.parser.stats }
    pub fn calls(&self) -> usize { self.calls }
    pub fn parser_done(&self) -> bool { self.parser.is_done() }
}

/// Fold the parser's stats into `/metrics` (once per finished request).
pub fn count_stats(s: &Stats) {
    for r in &s.rejected { m::tool_reject(r.label()); }
    m::add(&m::TOOL_OVER_LIMIT, s.dropped_over_limit as u64);
    m::add(&m::TEXT_INVOCATIONS, s.text_invocations as u64);
    m::add(&m::CUT_CALLS, s.cut_calls as u64);
    m::add(&m::HEADERLESS_TEXT, s.headerless_text as u64);
    m::add(&m::TOOL_CALLS, s.calls as u64);
}

pub fn finish_reason(end: End, calls: usize) -> &'static str { if calls > 0 { "tool_calls" } else if end == End::Length { "length" } else { "stop" } }

fn call_json(index: Option<usize>, id: &str, name: &str, arguments: &str) -> Value {
    let mut c = json!({"id": id, "type": "function", "function": {"name": name, "arguments": arguments}});
    if let Some(i) = index { c["index"] = json!(i); }
    c
}

/// The non-streamed reply body.
pub fn chat_json(id: &str, created: i64, model: &str, deltas: &[Delta], end: End, prompt_tokens: usize, completion_tokens: usize) -> Value {
    let (mut reasoning, mut content, mut calls) = (String::new(), String::new(), Vec::new());
    for d in deltas { match d {
        Delta::Reasoning(s) => reasoning.push_str(s), Delta::Content(s) => content.push_str(s),
        Delta::Call { id, name, arguments, .. } => calls.push(call_json(None, id, name, arguments)),
    } }
    let mut msg = json!({"role": "assistant", "content": if content.is_empty() && !calls.is_empty() { Value::Null } else { Value::String(content) }});
    if !reasoning.is_empty() { msg["reasoning_content"] = Value::String(reasoning); }
    let fr = finish_reason(end, calls.len());
    if !calls.is_empty() { msg["tool_calls"] = Value::Array(calls); }
    json!({"id": id, "object": "chat.completion", "created": created, "model": model,
           "choices": [{"index": 0, "message": msg, "finish_reason": fr}],
           "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}})
}

fn chunk(id: &str, created: i64, model: &str, delta: Value, finish: Option<&str>) -> String {
    format!("data: {}\n\n", json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}))
}
pub fn role_chunk(id: &str, created: i64, model: &str) -> String { chunk(id, created, model, json!({"role": "assistant", "content": ""}), None) }
pub fn delta_chunk(id: &str, created: i64, model: &str, d: &Delta) -> String {
    let delta = match d {
        Delta::Reasoning(s) => json!({"reasoning_content": s}),
        Delta::Content(s) => json!({"content": s}),
        Delta::Call { index, id: cid, name, arguments } => json!({"tool_calls": [call_json(Some(*index), cid, name, arguments)]}),
    };
    chunk(id, created, model, delta, None)
}
pub fn finish_chunk(id: &str, created: i64, model: &str, end: End, calls: usize) -> String { chunk(id, created, model, json!({}), Some(finish_reason(end, calls))) }
pub fn usage_chunk(id: &str, created: i64, model: &str, prompt: usize, completion: usize) -> String {
    format!("data: {}\n\n", json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}}))
}
pub const DONE: &str = "data: [DONE]\n\n";
