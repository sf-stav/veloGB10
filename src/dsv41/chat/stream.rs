//! The completion parser: reasoning, content and DSML tool calls, one engine for the whole text and
//! for byte-split streaming (the whole-text parse is "feed everything, then finish", so the two cannot
//! disagree by construction; the fuzz tests check that anyway).
//!
//! Reading rules (owner hardening of 2026-10-10, adapted to the V4.1 DSML grammar):
//! - a call exists only if tools were offered and the call passes `Catalogue::check`; nothing is invented;
//! - the block structure decides where things end, never a search for the first closer: a parameter
//!   value ends at the first `</｜DSML｜ parameter>` that is followed (after blanks) by the next
//!   parameter opener or the invoke closer, so a closer or opener inside a value stays in the value;
//! - an invoke that is structurally whole but fails the catalogue stays visible as content;
//!   a block that is cut off or breaks structurally stays text from where it broke;
//! - a JSON answer (response format channel) is never parsed for calls;
//! - orphan closers are plain content, flagged by a warning, never repaired.
//!
//! A call is reported the moment its invoke closer is read (not at the end of the reply), whole:
//! name and complete arguments in one event.

use super::catalogue::{Catalogue, RawParam, Reject};
use super::pyjson::to_json;
use super::tools::{split_tool_name, DSML_TOKEN, TC_BLOCK_NAME, TC_TAG_NAME, TP_TAG_NAME};
use crate::dsv4_chat::Json;

pub const EOS: &str = "<｜end▁of▁sentence｜>";
const THINK_END: &str = "</think>";

#[derive(Debug, Clone)]
pub struct ParseOptions {
    /// The prompt ended with `<think>`: the reply starts with reasoning.
    pub thinking: bool,
    /// Report at most this many calls (`parallel_tool_calls: false` is 1); the rest are dropped and counted.
    pub max_calls: Option<usize>,
    /// The reply is a structured JSON answer: never look for calls.
    pub json_channel: bool,
}

impl ParseOptions {
    pub fn new(thinking: bool) -> Self {
        Self { thinking, max_calls: None, json_channel: false }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedCall {
    pub namespace: Option<String>,
    /// The bare tool name (without the namespace).
    pub name: String,
    /// JSON object text: `{"k": v, ...}` exactly as the reference parser writes it.
    pub arguments: String,
}

impl ParsedCall {
    /// `ns::name` or `name`, as offered in the prompt.
    pub fn qualified_name(&self) -> String {
        match &self.namespace {
            Some(ns) => format!("{ns}::{}", self.name),
            None => self.name.clone(),
        }
    }

    /// The OpenAI `tool_calls[]` element (`namespace` rides at the top level when present, as the
    /// release's `tool_calls_to_openai_format` does). `id` is chosen by the caller.
    pub fn to_openai_json(&self, id: &str) -> Json {
        let mut o = vec![
            ("id".to_string(), Json::Str(id.to_string())),
            ("type".to_string(), Json::Str("function".to_string())),
        ];
        if let Some(ns) = &self.namespace {
            o.push(("namespace".to_string(), Json::Str(ns.clone())));
        }
        o.push((
            "function".to_string(),
            Json::Object(vec![
                ("name".to_string(), Json::Str(self.name.clone())),
                ("arguments".to_string(), Json::Str(self.arguments.clone())),
            ]),
        ));
        Json::Object(o)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Reasoning(String),
    Content(String),
    Call(ParsedCall),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Warning {
    /// The whole answer is DSML closing tags: a model slip, left as text, counted.
    OrphanDsmlClosers,
}

#[derive(Debug, Clone, Default)]
pub struct ParseStats {
    pub rejected: Vec<Reject>,
    pub structural_breaks: u32,
    pub unterminated: u32,
    pub dropped_over_limit: u32,
    pub blocks: u32,
}

#[derive(Debug, Clone, Default)]
pub struct Parsed {
    pub reasoning_content: String,
    pub content: String,
    pub calls: Vec<ParsedCall>,
    pub warnings: Vec<Warning>,
    pub stats: ParseStats,
}

fn calls_open() -> String {
    super::tools::calls_open()
}
fn calls_close() -> String {
    super::tools::calls_close()
}
fn invoke_open() -> String {
    format!("<{DSML_TOKEN}{TC_TAG_NAME} name=\"")
}
fn invoke_close() -> String {
    format!("</{DSML_TOKEN}{TC_TAG_NAME}>")
}
fn param_open() -> String {
    format!("<{DSML_TOKEN}{TP_TAG_NAME} name=\"")
}
fn param_close() -> String {
    format!("</{DSML_TOKEN}{TP_TAG_NAME}>")
}

#[derive(PartialEq)]
enum Cls {
    Full,
    Prefix,
    No,
}

fn classify(rest: &str, tag: &str) -> Cls {
    if rest.starts_with(tag) {
        Cls::Full
    } else if tag.starts_with(rest) {
        Cls::Prefix
    } else {
        Cls::No
    }
}

/// Length in bytes of the longest suffix of `rest` that is a proper prefix of one of `tokens`.
fn holdback(rest: &str, tokens: &[&str]) -> usize {
    let mut best = 0;
    for t in tokens {
        let max = (t.len() - 1).min(rest.len());
        for k in (best + 1..=max).rev() {
            if rest.is_char_boundary(rest.len() - k) && t.starts_with(&rest[rest.len() - k..]) {
                best = k;
                break;
            }
        }
    }
    best
}

fn skip_ws(s: &str, mut p: usize) -> usize {
    let b = s.as_bytes();
    while p < b.len() && matches!(b[p], b' ' | b'\n' | b'\r' | b'\t') {
        p += 1;
    }
    p
}

fn next_boundary(s: &str, mut p: usize) -> usize {
    p += 1;
    while p < s.len() && !s.is_char_boundary(p) {
        p += 1;
    }
    p
}

enum Blk {
    /// after the block opener or an invoke: blanks, then an invoke opener or the block closer
    Between,
    /// after an invoke header: blanks, then a parameter opener or the invoke closer
    InInvoke,
    /// inside a parameter value (`scan`: where the closer search resumes)
    InValue { pname: String, flag: Option<bool>, vstart: usize, scan: usize },
}

enum St {
    Reasoning,
    Content { after_block: bool },
    Block(Blk),
    Done,
}

struct BlockCtx {
    /// raw start of the block, including the `\n\n` before its opener when there was one
    raw_start: usize,
    opener_end: usize,
    /// start of the invoke being read
    invoke_start: usize,
    name: String,
    params: Vec<RawParam>,
    /// invokes read whole in this block (reported, rejected or dropped)
    done: usize,
}

type Step = (bool, Option<Blk>);

pub struct DsmlStream {
    opts: ParseOptions,
    catalogue: Option<Catalogue>,
    buf: String,
    /// start of text not yet settled into content/reasoning
    pos: usize,
    /// read position inside a block
    cur: usize,
    st: St,
    ctx: Option<BlockCtx>,
    pending: Vec<u8>,
    finished: bool,
    out: Vec<Event>,
    parsed: Parsed,
    calls_enabled: bool,
}

impl DsmlStream {
    pub fn new(opts: ParseOptions, catalogue: Option<Catalogue>) -> Self {
        let calls_enabled = !opts.json_channel && catalogue.as_ref().map_or(false, |c| !c.is_empty());
        let st = if opts.thinking { St::Reasoning } else { St::Content { after_block: false } };
        Self {
            opts,
            catalogue,
            buf: String::new(),
            pos: 0,
            cur: 0,
            st,
            ctx: None,
            pending: Vec::new(),
            finished: false,
            out: Vec::new(),
            parsed: Parsed::default(),
            calls_enabled,
        }
    }

    /// Everything read so far.
    pub fn parsed(&self) -> &Parsed {
        &self.parsed
    }

    pub fn into_parsed(self) -> Parsed {
        self.parsed
    }

    /// Feed raw bytes (a chunk may end inside a UTF-8 character).
    pub fn push_bytes(&mut self, bytes: &[u8]) -> Vec<Event> {
        if self.finished {
            return Vec::new();
        }
        self.pending.extend_from_slice(bytes);
        let mut text = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    text.push_str(s);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    text.push_str(std::str::from_utf8(&self.pending[..valid]).unwrap());
                    match e.error_len() {
                        None => {
                            self.pending.drain(..valid);
                            break;
                        }
                        Some(n) => {
                            text.push('\u{FFFD}');
                            self.pending.drain(..valid + n);
                        }
                    }
                }
            }
        }
        self.push_str_inner(&text);
        std::mem::take(&mut self.out)
    }

    pub fn push(&mut self, chunk: &str) -> Vec<Event> {
        self.push_bytes(chunk.as_bytes())
    }

    fn push_str_inner(&mut self, text: &str) {
        let mut from = self.buf.len().saturating_sub(EOS.len() - 1);
        while !self.buf.is_char_boundary(from) {
            from -= 1;
        }
        self.buf.push_str(text);
        // the end-of-sentence token ends the reply wherever it is
        if let Some(i) = self.buf[from..].find(EOS) {
            self.buf.truncate(from + i);
            self.finish_inner();
        } else {
            self.run(false);
        }
    }

    /// End of the reply (EOS or the length limit): resolve everything still pending.
    pub fn finish(&mut self) -> Vec<Event> {
        if !self.pending.is_empty() && !self.finished {
            let rest = String::from_utf8_lossy(&std::mem::take(&mut self.pending)).into_owned();
            self.push_str_inner(&rest);
        }
        self.finish_inner();
        std::mem::take(&mut self.out)
    }

    fn finish_inner(&mut self) {
        if self.finished {
            return;
        }
        self.run(true);
        self.finished = true;
        if self.calls_enabled && self.parsed.calls.is_empty() && is_orphan_closers(&self.parsed.content) {
            self.parsed.warnings.push(Warning::OrphanDsmlClosers);
        }
    }

    fn emit_reasoning(&mut self, s: &str) {
        if !s.is_empty() {
            self.parsed.reasoning_content.push_str(s);
            self.out.push(Event::Reasoning(s.to_string()));
        }
    }

    fn emit_content(&mut self, s: &str) {
        if !s.is_empty() {
            self.parsed.content.push_str(s);
            self.out.push(Event::Content(s.to_string()));
        }
    }

    /// Advance the state machine as far as the text allows. `fin`: no more text will come.
    fn run(&mut self, fin: bool) {
        loop {
            let progressed = match &self.st {
                St::Done => false,
                St::Reasoning => self.step_reasoning(fin),
                St::Content { .. } => self.step_content(fin),
                St::Block(_) => {
                    let St::Block(blk) = std::mem::replace(&mut self.st, St::Done) else { unreachable!() };
                    let (progress, next) = self.step_block(blk, fin);
                    if let Some(b) = next {
                        self.st = St::Block(b);
                    }
                    progress
                }
            };
            if !progressed {
                break;
            }
        }
    }

    fn step_reasoning(&mut self, fin: bool) -> bool {
        let rest = &self.buf[self.pos..];
        if let Some(i) = rest.find(THINK_END) {
            let r = rest[..i].to_string();
            self.pos += i + THINK_END.len();
            self.emit_reasoning(&r);
            self.st = St::Content { after_block: false };
            return true;
        }
        let hb = if fin { 0 } else { holdback(rest, &[THINK_END, EOS]) };
        let take = rest.len() - hb;
        if take > 0 {
            let r = rest[..take].to_string();
            self.pos += take;
            self.emit_reasoning(&r);
        }
        if fin {
            self.st = St::Done;
        }
        false
    }

    fn step_content(&mut self, fin: bool) -> bool {
        if matches!(self.st, St::Content { after_block: true }) {
            // blanks right after a block are the format's own
            self.pos = skip_ws(&self.buf, self.pos);
            if self.pos >= self.buf.len() {
                if fin {
                    self.st = St::Done;
                }
                return false;
            }
            self.st = St::Content { after_block: false };
        }
        let open = calls_open();
        let rest = &self.buf[self.pos..];
        if self.calls_enabled {
            if let Some(i) = rest.find(&open) {
                let abs = self.pos + i;
                let raw_start = if abs - self.pos >= 2 && self.buf[..abs].ends_with("\n\n") { abs - 2 } else { abs };
                let c = self.buf[self.pos..raw_start].to_string();
                self.emit_content(&c);
                self.pos = raw_start;
                let oe = abs + open.len();
                self.cur = oe;
                self.ctx = Some(BlockCtx {
                    raw_start,
                    opener_end: oe,
                    invoke_start: oe,
                    name: String::new(),
                    params: Vec::new(),
                    done: 0,
                });
                self.parsed.stats.blocks += 1;
                self.st = St::Block(Blk::Between);
                return true;
            }
        }
        let nn_open = format!("\n\n{open}");
        let toks: Vec<&str> = if self.calls_enabled { vec![nn_open.as_str(), open.as_str(), EOS] } else { vec![EOS] };
        let hb = if fin { 0 } else { holdback(rest, &toks) };
        let take = rest.len() - hb;
        if take > 0 {
            let c = rest[..take].to_string();
            self.pos += take;
            self.emit_content(&c);
        }
        if fin {
            self.st = St::Done;
        }
        false
    }

    /// The block broke or was cut off. What was not read as a call stays text, from `at`.
    fn fallback(&mut self, at: usize, broke: bool) -> Step {
        let ctx = self.ctx.take().expect("block ctx");
        if broke {
            self.parsed.stats.structural_breaks += 1;
        } else {
            self.parsed.stats.unterminated += 1;
        }
        let resume = if ctx.done == 0 {
            // nothing of the block was a call: the opener is literal text too
            let lit = self.buf[ctx.raw_start..ctx.opener_end].to_string();
            self.emit_content(&lit);
            ctx.opener_end
        } else {
            at
        };
        self.pos = resume;
        self.st = St::Content { after_block: false };
        (true, None)
    }

    fn close_block(&mut self, p: usize) -> Step {
        self.ctx = None;
        self.pos = p;
        self.st = St::Content { after_block: true };
        (true, None)
    }

    /// Not enough text to decide: wait, or at the end resolve as cut off.
    fn need_more(&mut self, blk: Blk, fin: bool) -> Step {
        if !fin {
            return (false, Some(blk));
        }
        match blk {
            Blk::Between => {
                if self.ctx.as_ref().unwrap().done > 0 {
                    // every invoke was whole and already reported; only the block closer is missing
                    self.parsed.stats.unterminated += 1;
                    let c = self.cur;
                    self.close_block(c)
                } else {
                    let c = self.cur;
                    self.fallback(c, false)
                }
            }
            _ => {
                let at = self.ctx.as_ref().unwrap().invoke_start;
                self.fallback(at, false)
            }
        }
    }

    fn broke_invoke(&mut self) -> Step {
        let at = self.ctx.as_ref().unwrap().invoke_start;
        self.fallback(at, true)
    }

    fn step_block(&mut self, blk: Blk, fin: bool) -> Step {
        match blk {
            Blk::Between => {
                self.cur = skip_ws(&self.buf, self.cur);
                let p = self.cur;
                let rest = &self.buf[p..];
                let close = calls_close();
                match classify(rest, &close) {
                    Cls::Full => return self.close_block(p + close.len()),
                    Cls::Prefix => return self.need_more(Blk::Between, fin),
                    Cls::No => {}
                }
                let iopen = invoke_open();
                match classify(rest, &iopen) {
                    Cls::Full => {
                        let ns = p + iopen.len();
                        let Some(q) = self.buf[ns..].find('"') else {
                            if self.buf[ns..].contains(['<', '\n']) {
                                return self.fallback(p, true);
                            }
                            return self.need_more(Blk::Between, fin);
                        };
                        let q = ns + q;
                        let name = &self.buf[ns..q];
                        if name.contains(['<', '\n']) || name.len() > 512 {
                            return self.fallback(p, true);
                        }
                        match classify(&self.buf[q..], "\">") {
                            Cls::Full => {
                                let name = name.to_string();
                                let c = self.ctx.as_mut().unwrap();
                                c.name = name;
                                c.params.clear();
                                c.invoke_start = p;
                                self.cur = q + 2;
                                (true, Some(Blk::InInvoke))
                            }
                            Cls::Prefix => self.need_more(Blk::Between, fin),
                            Cls::No => self.fallback(p, true),
                        }
                    }
                    Cls::Prefix => self.need_more(Blk::Between, fin),
                    Cls::No => self.fallback(p, true),
                }
            }
            Blk::InInvoke => {
                self.cur = skip_ws(&self.buf, self.cur);
                let p = self.cur;
                let rest = &self.buf[p..];
                let iclose = invoke_close();
                match classify(rest, &iclose) {
                    Cls::Full => {
                        self.finish_invoke(p + iclose.len());
                        return (true, Some(Blk::Between));
                    }
                    Cls::Prefix => return self.need_more(Blk::InInvoke, fin),
                    Cls::No => {}
                }
                let popen = param_open();
                match classify(rest, &popen) {
                    Cls::Full => {
                        let ns = p + popen.len();
                        let Some(q) = self.buf[ns..].find('"') else {
                            if self.buf[ns..].contains(['<', '\n']) {
                                return self.broke_invoke();
                            }
                            return self.need_more(Blk::InInvoke, fin);
                        };
                        let q = ns + q;
                        let pname = self.buf[ns..q].to_string();
                        if pname.contains(['<', '\n']) || pname.is_empty() || pname.len() > 512 {
                            return self.broke_invoke();
                        }
                        let after = &self.buf[q + 1..];
                        let variants: [(&str, Option<bool>); 3] =
                            [(" string=\"true\">", Some(true)), (" string=\"false\">", Some(false)), (">", None)];
                        for (tag, flag) in variants {
                            if after.starts_with(tag) {
                                let vstart = q + 1 + tag.len();
                                self.cur = vstart;
                                return (true, Some(Blk::InValue { pname, flag, vstart, scan: vstart }));
                            }
                        }
                        if variants.iter().any(|(t, _)| t.starts_with(after)) {
                            return self.need_more(Blk::InInvoke, fin);
                        }
                        self.broke_invoke()
                    }
                    Cls::Prefix => self.need_more(Blk::InInvoke, fin),
                    Cls::No => self.broke_invoke(),
                }
            }
            Blk::InValue { pname, flag, vstart, mut scan } => {
                let pclose = param_close();
                let popen = param_open();
                let iclose = invoke_close();
                loop {
                    let Some(rel) = self.buf[scan..].find(&pclose) else {
                        // no closer yet: only the tail that could still start one is unsettled
                        let from = self.buf.len().saturating_sub(pclose.len() - 1).max(scan);
                        scan = (from..=self.buf.len()).find(|i| self.buf.is_char_boundary(*i)).unwrap_or(self.buf.len());
                        return self.need_more(Blk::InValue { pname, flag, vstart, scan }, fin);
                    };
                    let c = scan + rel;
                    let e = c + pclose.len();
                    let w = skip_ws(&self.buf, e);
                    let rest = &self.buf[w..];
                    let cp = classify(rest, &popen);
                    let ci = classify(rest, &iclose);
                    if cp == Cls::Full || ci == Cls::Full {
                        let value = self.buf[vstart..c].to_string();
                        self.ctx.as_mut().unwrap().params.push(RawParam { name: pname, value, string: flag });
                        self.cur = e;
                        return (true, Some(Blk::InInvoke));
                    }
                    if cp == Cls::Prefix || ci == Cls::Prefix {
                        // the text after the closer is still too short to tell
                        return self.need_more(Blk::InValue { pname, flag, vstart, scan: c }, fin);
                    }
                    // not a boundary: the closer is part of the value
                    scan = next_boundary(&self.buf, c);
                }
            }
        }
    }

    /// An invoke was read whole, up to `after` (just past its closer): report it, or keep it as text.
    fn finish_invoke(&mut self, after: usize) {
        let (name, params, start) = {
            let c = self.ctx.as_mut().unwrap();
            (std::mem::take(&mut c.name), std::mem::take(&mut c.params), c.invoke_start)
        };
        self.ctx.as_mut().unwrap().done += 1;
        self.cur = after;
        let checked = match &self.catalogue {
            Some(cat) => cat.check(&name, &params),
            None => Err(Reject::UnknownTool),
        };
        match checked {
            Ok(list) => {
                if self.opts.max_calls.map_or(false, |m| self.parsed.calls.len() >= m) {
                    self.parsed.stats.dropped_over_limit += 1;
                    return;
                }
                let parts: Vec<String> =
                    list.iter().map(|p| format!("{}: {}", to_json(&Json::Str(p.name.clone())), p.json_text)).collect();
                let arguments = format!("{{{}}}", parts.join(", "));
                match split_tool_name(&name, None) {
                    Ok((namespace, bare)) => {
                        let call = ParsedCall { namespace, name: bare, arguments };
                        self.parsed.calls.push(call.clone());
                        self.out.push(Event::Call(call));
                    }
                    Err(_) => {
                        self.parsed.stats.rejected.push(Reject::UnknownTool);
                        let raw = self.buf[start..after].to_string();
                        self.emit_content(&raw);
                    }
                }
            }
            Err(rej) => {
                self.parsed.stats.rejected.push(rej);
                let raw = self.buf[start..after].to_string();
                self.emit_content(&raw);
            }
        }
    }
}

/// True when the whole text is DSML closing tags and blanks (either tag spelling).
pub fn is_orphan_closers(text: &str) -> bool {
    let d = DSML_TOKEN;
    let closers = [
        format!("</{d} parameter>"),
        format!("</{d} invoke>"),
        format!("</{d} calls>"),
        format!("</{d}parameter>"),
        format!("</{d}invoke>"),
        format!("</{d}tool_calls>"),
    ];
    let mut rest = text.trim_matches([' ', '\t', '\r', '\n']);
    if rest.is_empty() {
        return false;
    }
    while !rest.is_empty() {
        match closers.iter().find(|c| rest.starts_with(c.as_str())) {
            Some(c) => rest = rest[c.len()..].trim_start_matches([' ', '\t', '\r', '\n']),
            None => return false,
        }
    }
    true
}

/// Parse a whole completion (EOS already stripped or still present).
pub fn parse_completion(text: &str, opts: ParseOptions, catalogue: Option<Catalogue>) -> Parsed {
    let mut s = DsmlStream::new(opts, catalogue);
    s.push(text);
    s.finish();
    s.into_parsed()
}
