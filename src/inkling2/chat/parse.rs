//! The reply parser. The model writes a sequence of messages, each `<|message_model|> [header text] <|content_KIND|> body <|end_message|>`,
//! ending with `<|content_model_end_sampling|>` (EOS, the stop token). Block kinds: `thinking` is reasoning, `text` is the answer,
//! `invoke_tool_json` is a call (`{"name":..,"args":{..}}`, the header text names the callee), `invoke_tool_text` is a headerless
//! raw-text invocation (no structured name: kept as text, counted), `tool_error` is text. Several blocks of one kind concatenate.
//!
//! Input is TOKEN events (`Tok`): control tokens are told by ID, so a user's or a model's literal `<|end_message|>` spelling in ordinary
//! text is text, never framing. The whole-reply parse is "feed everything, finish", so it cannot disagree with streaming.
//!
//! Reading rules (owner hardening of 2026-10-10, as the other formats):
//! - a call exists only if tools were offered and the payload passes `Catalogue::check_payload`; a payload that fails stays visible as
//!   content text, nothing is repaired; an undeclared tool name is such a failure (the other stacks surface it; this engine does not
//!   invent a call the request never offered);
//! - the block structure decides where a call ends (`<|end_message|>` by id), never a search for a closer in the text;
//! - a call is reported once its block is closed, whole (name and complete arguments in one event); a reply cut by the token budget
//!   (`Finish::Length`) shows an open call's payload as text, an abort (`abort`) shows nothing: neither completes a call;
//! - messages of another author (a model echoing `<|message_user|>`) and text before the first header after a skipped special never stream.

use super::catalogue::{Catalogue, Reject};
use super::special::{self as sp, is_content_kind, is_reserved};

/// One decoded step of the model's output: a structural token by id, or the text an ordinary token decodes to.
#[derive(Debug, Clone, Copy)]
pub enum Tok<'a> { Special(u32), Text(&'a str) }

#[derive(Debug, Clone)]
pub struct ParseOptions {
    /// Reasoning blocks go to `Event::Reasoning` (false: into content, for clients that do not read `reasoning_content`).
    pub separate_reasoning: bool,
    /// Report at most this many calls (`parallel_tool_calls: false` is 1); the rest are dropped and counted.
    pub max_calls: Option<usize>,
}
impl Default for ParseOptions { fn default() -> Self { Self { separate_reasoning: true, max_calls: None } } }

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedCall { pub name: String, /** JSON object text */ pub arguments: String }

#[derive(Debug, Clone, PartialEq)]
pub enum Event { Reasoning(String), Content(String), Call(ParsedCall) }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish { Stop, Length }

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub rejected: Vec<Reject>,
    pub text_invocations: u32,
    pub dropped_over_limit: u32,
    pub headerless_text: u32,
    pub foreign_messages: u32,
    pub cut_calls: u32,
    pub calls: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError { Reserved(u32) }
impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { ParseError::Reserved(id) => write!(f, "generated token {id} is a reserved token of this checkpoint") }
    }
}
impl std::error::Error for ParseError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind { Thinking, Text, ToolError, Xml, InvokeJson, InvokeText, Other }

#[derive(Debug)]
enum State {
    /// No message open. `skipping`: a skipped special came first, so text before the next header is noise.
    Boundary { skipping: bool },
    Header { model: bool, text: String },
    Body { model: bool, kind: Kind, header: String, buf: String, unframed: bool },
    Done,
}

pub struct ReplyParser {
    cat: Catalogue,
    opt: ParseOptions,
    st: State,
    pub stats: Stats,
}

impl ReplyParser {
    pub fn new(cat: Catalogue, opt: ParseOptions) -> Self { Self { cat, opt, st: State::Boundary { skipping: false }, stats: Stats::default() } }

    pub fn is_done(&self) -> bool { matches!(self.st, State::Done) }

    /// Feed one step; the events it completed, in order.
    pub fn feed(&mut self, t: Tok<'_>) -> Result<Vec<Event>, ParseError> {
        let mut out = Vec::new();
        if let Tok::Special(id) = t { if is_reserved(id) { return Err(ParseError::Reserved(id)); } }
        self.step(t, &mut out);
        Ok(out)
    }

    fn step(&mut self, t: Tok<'_>, out: &mut Vec<Event>) {
        let st = std::mem::replace(&mut self.st, State::Done);
        self.st = match (st, t) {
            (State::Done, _) => State::Done,
            // ---- no message open ----
            (State::Boundary { .. }, Tok::Special(sp::EOS)) => State::Done,
            (State::Boundary { skipping }, Tok::Text(s)) => {
                if skipping || s.is_empty() { State::Boundary { skipping } } else {
                    // headerless model output: synthesize the opener (the model's text block)
                    self.stats.headerless_text += 1;
                    let mut b = State::Body { model: true, kind: Kind::Text, header: String::new(), buf: String::new(), unframed: true };
                    self.body_text(&mut b, s, out);
                    b
                }
            }
            (State::Boundary { .. }, Tok::Special(id)) => match id {
                sp::MESSAGE_MODEL => State::Header { model: true, text: String::new() },
                sp::MESSAGE_USER | sp::MESSAGE_SYSTEM | sp::MESSAGE_TOOL => State::Header { model: false, text: String::new() },
                _ => State::Boundary { skipping: true },   // begin_of_text, a stray end_message, a content kind without a header, ...
            },
            // ---- header ----
            (State::Header { .. }, Tok::Special(sp::EOS)) => State::Done,
            (State::Header { model, mut text }, Tok::Text(s)) => { text.push_str(s); State::Header { model, text } }
            (State::Header { model, text }, Tok::Special(id)) => match id {
                _ if is_content_kind(id) => State::Body { model, kind: kind_of(id), header: text, buf: String::new(), unframed: false },
                sp::MESSAGE_MODEL => State::Header { model: true, text: String::new() },
                sp::MESSAGE_USER | sp::MESSAGE_SYSTEM | sp::MESSAGE_TOOL => State::Header { model: false, text: String::new() },
                sp::END_MESSAGE => State::Boundary { skipping: false },
                _ => State::Header { model, text },
            },
            // ---- body ----
            (mut b @ State::Body { .. }, Tok::Text(s)) => { self.body_text(&mut b, s, out); b }
            (State::Body { unframed: true, model, kind, header, buf }, Tok::Special(id)) if id != sp::END_MESSAGE => {
                // a special inside headerless text is framing: close the block, then read the special at the boundary
                self.close(model, kind, &header, buf, Finish::Stop, out);
                self.st = State::Boundary { skipping: false };
                self.step(Tok::Special(id), out);
                return;
            }
            (State::Body { model, kind, header, buf, .. }, Tok::Special(sp::END_MESSAGE)) => {
                self.close(model, kind, &header, buf, Finish::Stop, out);
                State::Boundary { skipping: false }
            }
            (State::Body { model, kind, header, buf, .. }, Tok::Special(sp::EOS)) => {
                self.close(model, kind, &header, buf, Finish::Stop, out);
                State::Done
            }
            (mut b @ State::Body { .. }, Tok::Special(id)) => {
                // any other special inside a framed body keeps its spelling, as the reference parser's text does
                if let Some((name, _)) = sp::TABLE.iter().find(|(_, i)| *i == id) { self.body_text(&mut b, name, out); }
                b
            }
        };
    }

    fn body_text(&mut self, b: &mut State, s: &str, out: &mut Vec<Event>) {
        let State::Body { model, kind, buf, .. } = b else { return };
        if !*model { return; }
        match kind {
            Kind::Thinking => if !s.is_empty() { out.push(if self.opt.separate_reasoning { Event::Reasoning(s.into()) } else { Event::Content(s.into()) }) },
            Kind::Text | Kind::ToolError | Kind::Xml => if !s.is_empty() { out.push(Event::Content(s.into())) },
            Kind::InvokeJson | Kind::InvokeText => buf.push_str(s),
            Kind::Other => {}
        }
    }

    fn close(&mut self, model: bool, kind: Kind, header: &str, buf: String, how: Finish, out: &mut Vec<Event>) {
        if !model { self.stats.foreign_messages += 1; return; }
        match kind {
            Kind::InvokeText => { self.stats.text_invocations += 1; if !buf.is_empty() { out.push(Event::Content(buf)); } }
            Kind::InvokeJson => {
                if how == Finish::Length { self.stats.cut_calls += 1; if !buf.is_empty() { out.push(Event::Content(buf)); } return; }
                match self.cat.check_payload(header, &buf) {
                    Ok((name, arguments)) => {
                        if self.opt.max_calls.map_or(false, |m| self.stats.calls as usize >= m) {
                            self.stats.dropped_over_limit += 1;
                        } else {
                            self.stats.calls += 1;
                            out.push(Event::Call(ParsedCall { name, arguments }));
                        }
                    }
                    Err(r) => { self.stats.rejected.push(r); if !buf.is_empty() { out.push(Event::Content(buf)); } }
                }
            }
            _ => {}
        }
    }

    /// The reply ended (EOS, a stop, or the token budget). A block still open is closed as the stream ended it.
    pub fn finish(&mut self, how: Finish) -> Vec<Event> {
        let mut out = Vec::new();
        if let State::Body { model, kind, header, buf, .. } = std::mem::replace(&mut self.st, State::Done) {
            self.close(model, kind, &header, buf, how, &mut out);
        }
        self.st = State::Done;
        out
    }

    /// The client went away: an open call is never completed, nothing more is emitted.
    pub fn abort(&mut self) { self.st = State::Done; }
}

fn kind_of(id: u32) -> Kind {
    match id {
        sp::CONTENT_THINKING => Kind::Thinking, sp::CONTENT_TEXT => Kind::Text, sp::CONTENT_TOOL_ERROR => Kind::ToolError,
        sp::CONTENT_XML => Kind::Xml, sp::CONTENT_INVOKE_TOOL_JSON => Kind::InvokeJson, sp::CONTENT_INVOKE_TOOL_TEXT => Kind::InvokeText,
        _ => Kind::Other,
    }
}

/// The whole reply, accumulated.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Parsed { pub reasoning: String, pub content: String, pub calls: Vec<ParsedCall> }

impl Parsed {
    pub fn push(&mut self, ev: Event) {
        match ev { Event::Reasoning(s) => self.reasoning.push_str(&s), Event::Content(s) => self.content.push_str(&s), Event::Call(c) => self.calls.push(c) }
    }
}

/// Parse a whole reply (tests, non-streaming responses). Returns the accumulated reply and the parser's stats.
pub fn parse_all<'a>(toks: impl IntoIterator<Item = Tok<'a>>, cat: Catalogue, opt: ParseOptions, how: Finish) -> Result<(Parsed, Stats), ParseError> {
    let mut p = ReplyParser::new(cat, opt);
    let mut r = Parsed::default();
    for t in toks { for e in p.feed(t)? { r.push(e); } }
    for e in p.finish(how) { r.push(e); }
    Ok((r, p.stats))
}
