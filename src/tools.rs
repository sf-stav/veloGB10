//! Parse the model's tool-call syntax back into OpenAI `tool_calls`.
//!
//! Qwen3.5's own chat template instructs the model to emit calls in an XML-ish form, NOT the JSON
//! blob most people expect:
//!
//! ```text
//! <tool_call>
//! <function=get_weather>
//! <parameter=city>
//! Paris
//! </parameter>
//! <parameter=units>
//! c
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Every value arrives as TEXT. OpenAI's `arguments` is a JSON string whose values must have the types
//! the tool's JSON Schema declares -- a harness will feed them straight into a real function, so
//! sending `"count": "3"` where the schema says `integer` is a bug that surfaces in the caller, not
//! here. So we coerce each parameter against the declared schema, and fall back to string when the
//! schema is silent or the value doesn't parse.

use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;

use crate::tokenizer::{FunctionCall, ToolCall};

/// Tool-call ids must be unique across the WHOLE conversation, not just within one response.
///
/// They used to be `call_{index-within-this-response}`, so every turn of an agent loop emitted
/// `call_0` again. Harnesses match a tool RESULT back to its call by id — duplicate ids across turns
/// are exactly how a tool appears to run and then quietly has no effect, because the result gets
/// attached to the wrong (earlier) call. A process-wide counter costs nothing and removes the class.
static CALL_SEQ: AtomicU64 = AtomicU64::new(0);

const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";

// hy_v3's tool-call markup (its chat template instructs this form; the `:opensource` suffix is the
// family signature). A call is:
//   <tool_calls:opensource>
//   <tool_call:opensource>get_weather<tool_sep:opensource>
//   <arg_key:opensource>city</arg_key:opensource>
//   <arg_value:opensource>Paris</arg_value:opensource>
//   </tool_call:opensource>
//   </tool_calls:opensource>
const HY_CALL_OPEN: &str = "<tool_call:opensource>";
const HY_CALL_CLOSE: &str = "</tool_call:opensource>";
const HY_SEP: &str = "<tool_sep:opensource>";
const HY_AK: &str = "<arg_key:opensource>";
const HY_AK_END: &str = "</arg_key:opensource>";
const HY_AV: &str = "<arg_value:opensource>";
const HY_AV_END: &str = "</arg_value:opensource>";

/// Everything the model produced, split into the prose it wrote and the calls it made.
pub struct ParsedOutput {
    /// Text outside any `<tool_call>` block. The template explicitly permits reasoning BEFORE a call.
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

/// True if `s` contains the start of a tool call — used by the streaming path to decide whether it
/// must hold text back rather than forward it to the client as content. `<tool_call` is the shared
/// prefix of both families' tags.
pub fn has_tool_call(s: &str) -> bool {
    s.contains("<tool_call")
}

/// Parse a completed generation. `tools` is the request's `tools` array, used only to type-coerce
/// arguments; parsing still works (as strings) when it is absent. Format dispatch is by content:
/// hy_v3's `:opensource` markup or qwen's `<function=…>` markup.
pub fn parse(text: &str, tools: Option<&[Value]>) -> ParsedOutput {
    if text.contains(HY_CALL_OPEN) {
        return parse_hy3(text, tools);
    }
    let mut content = String::new();
    let mut tool_calls = Vec::new();
    let mut rest = text;

    while let Some(open) = rest.find(CALL_OPEN) {
        content.push_str(&rest[..open]);
        let after = &rest[open + CALL_OPEN.len()..];
        // A truncated call (hit max_tokens mid-emit) has no close tag. Drop it entirely: a half-parsed
        // call is worse than none, because the harness would invoke a real function with missing
        // arguments. `rest` MUST be cleared before breaking -- leaving it would append the partial XML
        // to content (and duplicate the prose before it), which is the exact leak this guards against.
        let Some(close) = after.find(CALL_CLOSE) else { rest = ""; break };
        for tc in parse_block(&after[..close], tools, tool_calls.len()) {
            tool_calls.push(tc);
        }
        rest = &after[close + CALL_CLOSE.len()..];
    }
    content.push_str(rest);

    ParsedOutput { content: content.trim().to_string(), tool_calls }
}

/// hy_v3 variant: prose is everything before the wrapper; every `<tool_call:opensource>…</…>`
/// block inside it becomes a call. A truncated block is dropped (same half-call rule as qwen).
fn parse_hy3(text: &str, tools: Option<&[Value]>) -> ParsedOutput {
    let mut tool_calls = Vec::new();
    // The wrapper `<tool_calls:opensource>` and the first call block start one char apart —
    // prose ends at whichever comes first, so the wrapper never leaks into the content.
    let first = ["<tool_calls:opensource>", HY_CALL_OPEN].iter()
        .filter_map(|t| text.find(t)).min().unwrap();
    let prose = &text[..first];
    let mut rest = &text[first..];
    while let Some(open) = rest.find(HY_CALL_OPEN) {
        let after = &rest[open + HY_CALL_OPEN.len()..];
        let Some(close) = after.find(HY_CALL_CLOSE) else { break };
        if let Some(tc) = parse_one_hy3(&after[..close], tools) {
            tool_calls.push(tc);
        }
        rest = &after[close + HY_CALL_CLOSE.len()..];
    }
    ParsedOutput { content: prose.trim().to_string(), tool_calls }
}

/// Parse one hy_v3 call body: `NAME<tool_sep:opensource>` then
/// `<arg_key:opensource>K</arg_key:opensource><arg_value:opensource>V</arg_value:opensource>` pairs.
fn parse_one_hy3(body: &str, tools: Option<&[Value]>) -> Option<ToolCall> {
    let sep = body.find(HY_SEP)?;
    let name = body[..sep].trim().to_string();
    if name.is_empty() { return None; }
    let schema = tools.and_then(|ts| param_schema(ts, &name));
    let mut args = serde_json::Map::new();
    let mut rest = &body[sep + HY_SEP.len()..];
    loop {
        let Some(k0) = rest.find(HY_AK) else { break };
        let a = &rest[k0 + HY_AK.len()..];
        let Some(k1) = a.find(HY_AK_END) else { break };
        let key = a[..k1].trim().to_string();
        let v = &a[k1 + HY_AK_END.len()..];
        let Some(v0) = v.find(HY_AV) else { break };
        let v = &v[v0 + HY_AV.len()..];
        let Some(v1) = v.find(HY_AV_END) else { break };
        let raw = v[..v1].trim_matches('\n');
        args.insert(key.clone(), coerce(raw, schema.and_then(|s| s.get(&key))));
        rest = &v[v1 + HY_AV_END.len()..];
    }
    Some(ToolCall {
        id: format!("call_{}", CALL_SEQ.fetch_add(1, Ordering::Relaxed)),
        kind: "function".to_string(),
        function: FunctionCall {
            name,
            arguments: serde_json::to_string(&Value::Object(args)).unwrap_or_else(|_| "{}".into()),
        },
    })
}

/// Locate the function tag inside one `<tool_call>` body.
///
/// The well-formed tag is `<function=NAME>`. In live serving (2026-08-27, 2 of 38 sampled
/// tool requests) the model SOMETIMES DROPS THE `<` and emits `<tool_call>\nfunction=NAME>`.
/// That used to fail `parse_one` entirely, and the two response modes then diverged: streaming
/// (which had already held the block back from the content stream) dropped the text, while
/// non-streaming leaked the raw block into `content`. Inside a `<tool_call>` block a `function=`
/// at the start of a line is unambiguous, so accept the bare form as a repair — the well-formed
/// match always wins when both are present. Returns `(byte offset, tag length)`.
fn find_function_tag(body: &str) -> Option<(usize, usize)> {
    find_function_tag_mode(body, false)
}

/// `wf_only`: the whole `<tool_call>` body contains a well-formed `<function=` somewhere, so a bare `function=NAME>` is never a
/// repair candidate in it (the well-formed form always wins — also AFTER the first call, where a bare-looking line inside a
/// parameter value used to be taken for a second call).
fn find_function_tag_mode(body: &str, wf_only: bool) -> Option<(usize, usize)> {
    const WF: &str = "<function=";
    const BARE: &str = "function=";
    if let Some(i) = body.find(WF) {
        return Some((i, WF.len()));
    }
    if wf_only { return None; }
    let mut from = 0usize;
    while let Some(rel) = body[from..].find(BARE) {
        let i = from + rel;
        let boundary_ok = match body[..i].chars().next_back() {
            None | Some('\n') | Some('\r') | Some(' ') | Some('\t') => true,
            _ => false,
        };
        if boundary_ok {
            return Some((i, BARE.len()));
        }
        from = i + BARE.len();
    }
    None
}

/// Parse one `<tool_call>…</tool_call>` body into ZERO OR MORE calls.
///
/// Two shapes are supported; the body decides:
///  * a JSON object (the Froggeric `tool_call_format=json` emission) — one call, with the
///    tolerant salvage rules in `parse_json_call`;
///  * the XML form — EVERY `<function=NAME>` in the block becomes its own call. Before Phase 2
///    only the FIRST function tag produced a call and every later function's `<parameter=…>`
///    pairs were merged into it, same-key values overwritten (RENDER_AUDIT.md 9.2).
fn parse_block(body: &str, tools: Option<&[Value]>, idx: usize) -> Vec<ToolCall> {
    let trimmed = body.trim_start();
    if trimmed.starts_with('{') {
        if let Some(tc) = parse_json_call(trimmed, tools, idx) {
            return vec![tc];
        }
        // Unparseable JSON body: fall through to the XML path rather than dropping the call.
    }
    parse_xml_calls(body, tools)
}

/// The XML form: one call per `<function=NAME>` tag, each owning the `<parameter=…>` pairs up
/// to the next tag. Also accepts a JSON-object body inside the function tag (the Froggeric
/// template's string-arguments history form) when no `<parameter=` pair is present.
fn parse_xml_calls(body: &str, tools: Option<&[Value]>) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    // A body with any well-formed tag takes no bare repairs at all (see find_function_tag_mode).
    let wf_only = body.contains("<function=");
    // The scan is SEQUENTIAL: a call owns its `<parameter=…>…</parameter>` spans, and a function tag is only a tag when it sits
    // OUTSIDE every value span consumed so far. A parameter value that merely contains the text `function=x>` or `<function=x>`
    // (a coding agent writing a file about this very format) stays payload and never becomes a second call.
    let mut from = 0usize;
    while let Some((i, tlen)) = find_function_tag_mode(&body[from..], wf_only) {
        let start = from + i;
        let name_from = start + tlen;
        let Some(gt) = body[name_from..].find('>') else { from = name_from; continue };
        let name = body[name_from..name_from + gt].trim().to_string();
        let after_name = name_from + gt + 1;
        if name.is_empty() { from = after_name; continue; }

        let schema = tools.and_then(|ts| param_schema(ts, &name));
        let mut args = serde_json::Map::new();
        // `cursor` is always outside a value span. The next `<parameter=` belongs to THIS call unless the next function tag comes
        // first (then it is that call's); the end of this call's text is the next tag after the last consumed span.
        let mut cursor = after_name;
        loop {
            let next_tag = find_function_tag_mode(&body[cursor..], wf_only).map(|(ti, _)| cursor + ti);
            let next_param = body[cursor..].find("<parameter=").map(|pi| cursor + pi);
            let Some(popen) = next_param else { break };
            if next_tag.map_or(false, |t| t < popen) { break; }
            let a = &body[popen + "<parameter=".len()..];
            let Some(gt2) = a.find('>') else { break };
            let key = a[..gt2].trim().to_string();
            let vstart = &a[gt2 + 1..];
            let Some(pclose) = vstart.find("</parameter>") else { break };
            // The template puts a newline after `>` and before `</parameter>`; they are delimiters,
            // not part of the value. Trim only those, so interior whitespace of a multi-line value
            // survives.
            let raw = vstart[..pclose].trim_matches('\n');
            args.insert(key.clone(), coerce(raw, schema.and_then(|s| s.get(&key))));
            cursor = body.len() - vstart.len() + pclose + "</parameter>".len();
        }
        // This call's text runs to the next function tag after the last consumed value span (or the end of the block).
        let end = find_function_tag_mode(&body[cursor..], wf_only).map(|(ti, _)| cursor + ti).unwrap_or(body.len());
        let after = &body[start + tlen..end];
        if args.is_empty() {
            let jb = after[gt + 1..].trim();
            let jb = jb.strip_suffix("</function>").unwrap_or(jb).trim();
            // A body that LOOKS like JSON is a JSON body: the Froggeric template's
            // `raw_tool_args_history` branch renders the model's own previous calls as a JSON
            // object inside the function tag, and the model imitates that while closing with the
            // `</parameter>` tag it also sees in the XML example. So the emitted shape is
            // `{…json…}\n</parameter>` — ONE stray trailing close tag. Strip it (and the
            // whitespace between) before parsing. Before this, every such body fell through to
            // `args` = {} and the harness invoked a real tool with no arguments: 6/6 lost
            // (phase3/EMPTY_ARGS_EVIDENCE.md, PHASE3_REPORT.md 4.3), while the 5/5 clean-JSON
            // bodies parsed. A non-empty body that still fails to yield arguments is ALARMED,
            // never silently dropped.
            let jsonish = jb.starts_with('{') || jb.starts_with('[');
            if jsonish {
                let jb = strip_stray_param_close(jb);
                // Phase-5 B1: `json_object_lenient` = strict parse, then the unescaped-quote
                // repair. An array, an unrepairable body or a truncated one still alarms.
                match json_object_lenient(jb) {
                    Some(o) => for (k, v) in o { args.insert(k, v); },
                    None => eprintln!("[tool-args-alarm] {jb}"),
                }
            } else if !jb.is_empty() {
                eprintln!("[tool-args-alarm] {jb}");
            }
        }
        calls.push(ToolCall {
            id: format!("call_{}", CALL_SEQ.fetch_add(1, Ordering::Relaxed)),
            kind: "function".to_string(),
            function: crate::tokenizer::FunctionCall {
                name,
                arguments: serde_json::to_string(&Value::Object(args)).unwrap_or_else(|_| "{}".into()),
            },
        });
        from = end;
    }
    calls
}

/// One JSON-body call. Strict `{"name": T, "arguments": …}` is the shipped form and is tried
/// first; the three MALFORMED shapes the Phase-1 ledger proved the model actually emits inside
/// `<tool_call>` are salvaged after it (LEDGER_REPORT.md 1.1.1):
///   * wrapper    `{"function": T, "arguments": {…}}` / `{"function": {"name": T, …}}`
///   * flattened  `{"function": T, "k": v, …}` — the remaining keys ARE the arguments
///   * name-first `{T, "k": "v"}` — not valid JSON; repaired, T is the tool name
/// The salvage shapes require the recovered name to be a tool of the request catalog (when the
/// catalog is known), so structured-output JSON with an unrelated `name` field is never promoted
/// to a call. The strict form keeps its original behaviour (no catalog check).
fn parse_json_call(body: &str, tools: Option<&[Value]>, idx: usize) -> Option<ToolCall> {
    let t = body.trim();
    if let Ok(v) = serde_json::from_str::<Value>(t) {
        return call_from_object(v.as_object()?, tools, idx);
    }
    // 2b. Phase-5 B1: valid JSON except for UNESCAPED quotes inside a string value —
    //     `{"language":"python","code":"print("correct")"}`. The NVFP4 lane emits this ~4x
    //     more often than FP8 (phase-4 alarm census); before the repair every argument was
    //     lost. Repair first, then run the SAME shape resolution as the strict path.
    if repair_quotes_enabled() {
        if let Some(rep) = repair_unescaped_quotes(t) {
            if let Ok(Value::Object(o)) = serde_json::from_str::<Value>(&rep) {
                if let Some(tc) = call_from_object(&o, tools, idx) {
                    return Some(tc);
                }
            }
        }
    }
    // 3. not valid JSON: `{T, "k": "v"}` (name-first, unquoted name) — repair.
    let (name, args) = salvage_name_first(t, tools)?;
    Some(json_call(name, args, idx))
}

/// Resolve ONE already-parsed JSON object into a call, using the four shapes the ledger proved
/// the model emits: strict OpenAI `{"name":T,"arguments":…}`, flattened `{"name":T,"k":v,…}`,
/// wrapper `{"function":T|{…},"arguments":…}` and flattened `{"function":T,"k":v,…}`. Returns
/// `None` for any object that is not one of them (never guesses a name).
fn call_from_object(o: &serde_json::Map<String, Value>, tools: Option<&[Value]>, idx: usize) -> Option<ToolCall> {
    // 1. strict OpenAI form.
    if let Some(name) = o.get("name").and_then(Value::as_str).filter(|s| !s.is_empty()) {
        if let Some(args) = o.get("arguments") {
            return Some(json_call(name.to_string(), args_to_string(args), idx));
        }
        // 1b. flattened name form: {"name": T, "k": v, …}
        if let Some(args) = flat_args(o, &["name"]) {
            return Some(json_call(name.to_string(), args, idx));
        }
    }
    // 2. function-key wrapper / flattened form.
    if let Some(f) = o.get("function") {
        let (name, args) = match f {
            Value::String(s) if !s.is_empty() => match o.get("arguments") {
                Some(a) => (s.clone(), args_to_string(a)),
                None => (s.clone(), flat_args(o, &["function"]).unwrap_or_else(|| "{}".into())),
            },
            Value::Object(fo) => {
                let name = fo.get("name").and_then(Value::as_str)?.to_string();
                let a = fo.get("arguments").or_else(|| o.get("arguments"));
                (name, a.map(args_to_string).unwrap_or_else(|| "{}".into()))
            }
            _ => return None,
        };
        if !name.is_empty() && known_tool(tools, &name) {
            return Some(json_call(name, args, idx));
        }
        return None;
    }
    None
}

/// Phase-5 B1 negative control switch. Production is ALWAYS on (`#[cfg(not(test))]` below
/// returns true); tests flip it off in their own thread to prove the repair — and not some
/// other salvage path — is what recovers the archived bodies.
#[cfg(not(test))]
fn repair_quotes_enabled() -> bool { true }

#[cfg(test)]
thread_local! {
    static REPAIR_QUOTES: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}
#[cfg(test)]
fn repair_quotes_enabled() -> bool { REPAIR_QUOTES.with(|c| c.get()) }
#[cfg(test)]
fn set_repair_quotes(on: bool) { REPAIR_QUOTES.with(|c| c.set(on)) }

/// Phase-5 B1 — repair UNESCAPED double quotes inside JSON string VALUES.
///
/// Lex the text and escape every `"` that is INSIDE a string but cannot be its closing
/// delimiter. A closing quote is followed (after optional whitespace) by one of `, } ] :` —
/// the only characters that can legally follow a string in JSON; a quote followed by anything
/// else is part of the value and must be escaped. A backslash escapes the next character, so
/// already-escaped input (`\"`) is copied verbatim and never re-escaped.
///
/// `None` means NOT REPAIRABLE: nothing changed, a string was left unterminated, or the
/// repaired text still does not parse as a JSON object. The caller then keeps the
/// `[tool-args-alarm]` — an ambiguous body is never guessed (and a TRUNCATED body, which this
/// never completes, stays alarmed by policy: see the phase-5 truncated-body decision).
fn repair_unescaped_quotes(s: &str) -> Option<String> {
    let mut out = String::with_capacity(s.len() + 8);
    let mut in_string = false;
    let mut changed = false;
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if !in_string {
            if c == '"' { in_string = true; }
            out.push(c);
            continue;
        }
        match c {
            '\\' => {
                out.push(c);
                if let Some(n) = it.next() { out.push(n); }
            }
            '"' => {
                let closes = it.clone()
                    .find(|ch| !ch.is_whitespace())
                    .map_or(true, |ch| matches!(ch, ',' | '}' | ']' | ':'));
                if closes {
                    in_string = false;
                    out.push('"');
                } else {
                    out.push_str("\\\"");
                    changed = true;
                }
            }
            _ => out.push(c),
        }
    }
    if !changed || in_string { return None; }
    match serde_json::from_str::<Value>(&out) {
        Ok(Value::Object(_)) => Some(out),
        _ => None,
    }
}

/// Parse a function body as a JSON arguments object, applying the unescaped-quote repair when
/// strict parsing fails. `None` => the caller emits the alarm.
fn json_object_lenient(jb: &str) -> Option<serde_json::Map<String, Value>> {
    if let Ok(Value::Object(o)) = serde_json::from_str::<Value>(jb) {
        return Some(o);
    }
    if repair_quotes_enabled() {
        if let Some(rep) = repair_unescaped_quotes(jb) {
            if let Ok(Value::Object(o)) = serde_json::from_str::<Value>(&rep) {
                return Some(o);
            }
        }
    }
    None
}

/// Strip ONE stray trailing `</parameter>` (plus the whitespace between the JSON and that tag)
/// from a function body that otherwise looks like JSON. The model emits
/// `{…json…}\n</parameter>` because it imitates the JSON body our render puts in the history
/// while closing with the XML tag the system prompt shows. Returns the input unchanged when the
/// tag is absent.
fn strip_stray_param_close(jb: &str) -> &str {
    let t = jb.trim_end();
    match t.strip_suffix("</parameter>") {
        Some(rest) => rest.trim_end(),
        None => t,
    }
}

/// The RAW function body the model emitted for `name`, verbatim (only outer whitespace
/// trimmed). Used by the empty-argument alarm; `None` when the tag is not in the text.
fn raw_function_body<'a>(raw: &'a str, name: &str) -> Option<&'a str> {
    let wf = format!("<function={name}>");
    let bare = format!("function={name}>");
    let i = raw.find(&wf).or_else(|| raw.find(&bare))?;
    let tlen = if raw[i..].starts_with(&wf) { wf.len() } else { bare.len() };
    let rest = &raw[i + tlen..];
    let end = ["</function>", "</tool_call>", "<function="].iter()
        .filter_map(|m| rest.find(m))
        .min()
        .unwrap_or(rest.len());
    Some(rest[..end].trim())
}

/// Phase-4 A2 — the empty-argument alarm.
///
/// For every parsed call whose arguments are EMPTY while the tool's schema declares parameters,
/// return the raw function body the model emitted, when that body is non-empty. Callers log each
/// returned body as `[tool-args-alarm] <raw body>`. This is the net that makes ANY future
/// regression of the silent-argument-drop class visible in every serving log, instead of
/// surfacing as a tool that "ran" and did nothing.
pub fn empty_arg_alarms<'a>(raw: &'a str, tools: Option<&[Value]>, calls: &[ToolCall]) -> Vec<&'a str> {
    let mut out = Vec::new();
    let Some(ts) = tools else { return out };
    for c in calls {
        let a = c.function.arguments.trim();
        if !(a.is_empty() || a == "{}") { continue; }
        let declares = param_schema(ts, &c.function.name).map_or(false, |p| !p.is_empty());
        if !declares { continue; }
        if let Some(body) = raw_function_body(raw, &c.function.name) {
            if !body.is_empty() { out.push(body); }
        }
    }
    out
}

fn json_call(name: String, arguments: String, idx: usize) -> ToolCall {
    ToolCall {
        id: format!("call_{:02x}", idx),
        kind: "function".to_string(),
        function: crate::tokenizer::FunctionCall { name, arguments },
    }
}

/// `arguments` value → the JSON string our ToolCall carries: a string is kept verbatim, an
/// object/array is re-serialized, null becomes `{}`.
fn args_to_string(a: &Value) -> String {
    match a {
        Value::String(s) => s.clone(),
        Value::Null => "{}".to_string(),
        other => other.to_string(),
    }
}

/// Every object key except `skip`, serialized as an arguments object (the flattened shape).
fn flat_args(o: &serde_json::Map<String, Value>, skip: &[&str]) -> Option<String> {
    let m: serde_json::Map<String, Value> = o.iter()
        .filter(|(k, _)| !skip.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if m.is_empty() { None } else { Some(Value::Object(m).to_string()) }
}

/// Is `name` a tool of the request catalog? Unknown/absent catalog → true (the call is inside a
/// `<tool_call>` block, i.e. the model already declared the intent).
fn known_tool(tools: Option<&[Value]>, name: &str) -> bool {
    match tools {
        Some(ts) if !ts.is_empty() => ts.iter().any(|t| {
            t.pointer("/function/name").and_then(Value::as_str) == Some(name)
                || t.get("name").and_then(Value::as_str) == Some(name)
        }),
        _ => true,
    }
}

/// Repair the invalid `{T, "k": "v"}` shape: first token is the tool name (quoted or bare), the
/// rest is the argument object once wrapped in braces.
fn salvage_name_first(t: &str, tools: Option<&[Value]>) -> Option<(String, String)> {
    let inner = t.strip_prefix('{')?.trim_start();
    let (name, rest) = if let Some(stripped) = inner.strip_prefix('"') {
        let end = stripped.find('"')?;
        (stripped[..end].to_string(), &stripped[end + 1..])
    } else {
        let end = inner.find(|c| c == ',' || c == '}')?;
        (inner[..end].trim().to_string(), &inner[end..])
    };
    if name.is_empty() || !known_tool(tools, &name) { return None; }
    let rest = rest.trim_start().trim_start_matches(',').trim();
    let rest = rest.strip_suffix('}').unwrap_or(rest).trim();
    if rest.is_empty() { return Some((name, "{}".to_string())); }
    let repaired = format!("{{{rest}}}");
    let v: Value = serde_json::from_str(&repaired).ok()?;
    if !v.is_object() { return None; }
    Some((name, v.to_string()))
}

/// `tools[i].function.parameters.properties` for the named function.
fn param_schema<'a>(tools: &'a [Value], name: &str) -> Option<&'a serde_json::Map<String, Value>> {
    tools.iter()
        .find(|t| t.pointer("/function/name").and_then(Value::as_str) == Some(name))
        .and_then(|t| t.pointer("/function/parameters/properties"))
        .and_then(Value::as_object)
}

/// The ONE final (content, tool_calls, finish_reason) decision for a completed generation,
/// shared by BOTH response modes — streaming SSE and non-streaming JSON.
///
/// This exists because the modes used to decide independently and could diverge (the
/// 2026-08-27 user report): with a call the parser could not read, streaming dropped the
/// block's text entirely while non-streaming returned it verbatim. Both modes now call this
/// on the same post-think-split answer text, so they can only ever agree.
///
/// Rules (the exact old non-streaming behavior, now canonical for both):
///  - at least one call parsed  -> content becomes the prose outside the calls (`None` when
///    empty) and finish becomes `"tool_calls"` — the flag every harness branches on;
///  - no call parsed            -> content is the model's literal answer text UNCHANGED
///    (recoverable by the operator) and `finish` passes through.
pub fn finalize(raw: &str, tools: Option<&[Value]>, finish: &str) -> (Option<String>, Vec<ToolCall>, String) {
    let parsed = parse(raw, tools);
    finalize_parsed(raw, parsed, finish)
}

/// Decision half of [`finalize`] for callers that already ran `parse` (e.g. to log the raw
/// output). Takes the parse result by value so call ids are minted exactly once.
pub fn finalize_parsed(
    raw: &str,
    parsed: ParsedOutput,
    finish: &str,
) -> (Option<String>, Vec<ToolCall>, String) {
    if parsed.tool_calls.is_empty() {
        (Some(raw.to_string()), Vec::new(), finish.to_string())
    } else {
        let c = if parsed.content.is_empty() { None } else { Some(parsed.content) };
        (c, parsed.tool_calls, "tool_calls".to_string())
    }
}

/// What the STREAMING mode must still surface after its incremental emission, so it never
/// silently drops text the non-streaming mode returns. `acc` is the full generated text and
/// `emitted` the byte offset already sent as content chunks (the hold-back in server.rs stops
/// at the first `<tool_call`, so everything after it sits in the remainder).
///
/// `None` when nothing is held back. Some(text) only when the held-back span actually contains
/// a tool-call marker: if the parser read the calls, the block is represented by the
/// `tool_calls` delta instead (and unparsed block text is dropped by BOTH modes identically —
/// that is `finalize_parsed`'s rule); if the parser read NOTHING from a block that looked like
/// a call, the raw text is surfaced as a final content chunk, exactly what non-streaming does.
pub fn held_back_remainder<'a>(acc: &'a str, emitted: usize) -> Option<&'a str> {
    if emitted >= acc.len() {
        return None;
    }
    let rest = &acc[emitted..];
    if rest.contains(CALL_OPEN) || rest.contains(HY_CALL_OPEN) || rest.contains("<tool_calls") {
        Some(rest)
    } else {
        None
    }
}

/// Coerce one text value to the type its schema declares. Unknown/absent schema, or a value that does
/// not parse as the declared type, stays a string — never guess a type the schema did not ask for.
fn coerce(raw: &str, schema: Option<&Value>) -> Value {
    let ty = schema.and_then(|s| s.get("type")).and_then(Value::as_str);
    match ty {
        Some("integer") => raw.trim().parse::<i64>().map(Value::from).unwrap_or_else(|_| Value::String(raw.to_string())),
        Some("number")  => raw.trim().parse::<f64>().map(Value::from).unwrap_or_else(|_| Value::String(raw.to_string())),
        Some("boolean") => match raw.trim().to_ascii_lowercase().as_str() {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            _ => Value::String(raw.to_string()),
        },
        // The template serialises objects/arrays with `| tojson`, so they come back as JSON text.
        Some("object") | Some("array") =>
            serde_json::from_str(raw.trim()).unwrap_or_else(|_| Value::String(raw.to_string())),
        _ => Value::String(raw.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Vec<Value> {
        vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {
                    "city":  {"type": "string"},
                    "days":  {"type": "integer"},
                    "exact": {"type": "boolean"},
                    "opts":  {"type": "object"}
                }}
            }
        })]
    }

    #[test]
    fn parses_a_call_and_coerces_types() {
        let out = parse("Let me check.\n<tool_call>\n<function=get_weather>\n\
                         <parameter=city>\nParis\n</parameter>\n\
                         <parameter=days>\n3\n</parameter>\n\
                         <parameter=exact>\ntrue\n</parameter>\n\
                         <parameter=opts>\n{\"a\":1}\n</parameter>\n\
                         </function>\n</tool_call>", Some(&tools()));
        assert_eq!(out.content, "Let me check.");
        assert_eq!(out.tool_calls.len(), 1);
        let tc = &out.tool_calls[0];
        assert_eq!(tc.function.name, "get_weather");
        let a: Value = serde_json::from_str(&tc.function.arguments).unwrap();
        assert_eq!(a["city"], "Paris");
        assert_eq!(a["days"], 3);              // integer, not "3"
        assert_eq!(a["exact"], true);          // bool, not "true"
        assert_eq!(a["opts"]["a"], 1);         // nested object
    }

    #[test]
    fn multiple_calls() {
        let out = parse("<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n\
                         <tool_call>\n<function=b>\n</function>\n</tool_call>", None);
        assert_eq!(out.tool_calls.len(), 2);
        assert_eq!(out.tool_calls[0].function.name, "a");
        assert_eq!(out.tool_calls[1].function.name, "b");
        assert_ne!(out.tool_calls[0].id, out.tool_calls[1].id);   // distinct within a response
        // ...and across responses: a harness matches tool RESULTS to calls by id, so an agent loop
        // that re-emits `call_0` every turn attaches results to the wrong call.
        let again = parse("<tool_call>\n<function=a>\n</function>\n</tool_call>", None);
        assert_ne!(again.tool_calls[0].id, out.tool_calls[0].id);
    }

    /// A parameter value is payload: text in it that LOOKS like a function tag (a coding agent writing a file about this very
    /// format) must never become a second tool call — well-formed or bare, on its own line or not.
    #[test]
    fn function_looking_text_inside_a_value_is_payload() {
        let file = "# notes\n<function=decoy>\nfunction=other>\n<parameter=q>\nnot a param\n";
        let wf = format!("<tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/a.md\n</parameter>\n<parameter=content>\n{file}</parameter>\n</function>\n</tool_call>");
        let out = parse(&wf, None);
        assert_eq!(out.tool_calls.len(), 1, "{:?}", out.tool_calls);
        assert_eq!(out.tool_calls[0].function.name, "write_file");
        let a: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        assert_eq!(a["path"], "/tmp/a.md");
        assert_eq!(a["content"], file.trim_matches('\n'), "the whole value survives verbatim");
        // bare form (the model dropped the `<`): the in-value lines are still payload
        let bare = "<tool_call>\nfunction=write_file>\n<parameter=content>\nfunction=decoy>\nx\n</parameter>\n</tool_call>";
        let out = parse(bare, None);
        assert_eq!(out.tool_calls.len(), 1, "{:?}", out.tool_calls);
        assert_eq!(out.tool_calls[0].function.name, "write_file");
        let a: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        assert_eq!(a["content"], "function=decoy>\nx");
    }

    /// Several real calls in ONE block still split, each owning only its own parameters.
    #[test]
    fn several_functions_in_one_block_keep_their_own_parameters() {
        let out = parse("<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n\
                          <function=b>\n<parameter=x>\n2\n</parameter>\n<parameter=y>\nz\n</parameter>\n</function>\n</tool_call>", None);
        assert_eq!(out.tool_calls.len(), 2);
        let (a, b): (Value, Value) = (serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap(),
                                       serde_json::from_str(&out.tool_calls[1].function.arguments).unwrap());
        assert_eq!((out.tool_calls[0].function.name.as_str(), out.tool_calls[1].function.name.as_str()), ("a", "b"));
        assert_eq!(a, serde_json::json!({"x": "1"}));   // no schema: values stay strings
        assert_eq!(b, serde_json::json!({"x": "2", "y": "z"}));
    }

    #[test]
    fn a_truncated_call_is_dropped_not_half_parsed() {
        // Hit max_tokens mid-call: no </tool_call>. Emitting a call with missing arguments would make
        // the harness invoke a real function with a hole in it.
        let out = parse("thinking\n<tool_call>\n<function=get_weather>\n<parameter=city>\nPar", Some(&tools()));
        assert!(out.tool_calls.is_empty());
        assert_eq!(out.content, "thinking");
    }

    #[test]
    fn plain_prose_is_untouched() {
        let out = parse("There is no tool for that.", Some(&tools()));
        assert!(out.tool_calls.is_empty());
        assert_eq!(out.content, "There is no tool for that.");
    }

    #[test]
    fn multi_line_values_keep_interior_whitespace() {
        let out = parse("<tool_call>\n<function=f>\n<parameter=code>\nline1\n  line2\n</parameter>\n\
                         </function>\n</tool_call>", None);
        let a: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        assert_eq!(a["code"], "line1\n  line2");
    }

    #[test]
    fn hy_v3_opensource_markup() {
        let out = parse("Let me check.\n<tool_calls:opensource>\n\
                         <tool_call:opensource>get_weather<tool_sep:opensource>\n\
                         <arg_key:opensource>city</arg_key:opensource>\n\
                         <arg_value:opensource>Paris</arg_value:opensource>\n\
                         <arg_key:opensource>days</arg_key:opensource>\n\
                         <arg_value:opensource>3</arg_value:opensource>\n\
                         </tool_call:opensource>\n</tool_calls:opensource>", Some(&tools()));
        assert_eq!(out.content, "Let me check.");
        assert_eq!(out.tool_calls.len(), 1);
        let tc = &out.tool_calls[0];
        assert_eq!(tc.function.name, "get_weather");
        let a: Value = serde_json::from_str(&tc.function.arguments).unwrap();
        assert_eq!(a["city"], "Paris");
        assert_eq!(a["days"], 3);   // coerced to the schema's integer
        // A truncated hy_v3 call drops cleanly too.
        let out2 = parse("thinking\n<tool_call:opensource>get_weather<tool_sep:opensource>\n\
                          <arg_key:opensource>city", Some(&tools()));
        assert!(out2.tool_calls.is_empty());
        assert_eq!(out2.content, "thinking");
    }

    /// Phase-2 A2 acceptance: a `<tool_call>` with TWO `<function=>` blocks yields TWO calls,
    /// each with its own arguments. Before the fix the second function was merged into the first
    /// and same-key values overwrote each other (the probe in RENDER_AUDIT.md 9.2).
    #[test]
    fn two_functions_in_one_tool_call_block() {
        let out = parse("<tool_call>\n<function=read_file>\n<parameter=file_id>\n/a\n</parameter>\n\
                         </function>\n<function=read_file>\n<parameter=file_id>\n/b\n</parameter>\n\
                         </function>\n</tool_call>", Some(&tools()));
        assert_eq!(out.tool_calls.len(), 2, "each <function=> must be its own call");
        let a0: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        let a1: Value = serde_json::from_str(&out.tool_calls[1].function.arguments).unwrap();
        assert_eq!(out.tool_calls[0].function.name, "read_file");
        assert_eq!(out.tool_calls[1].function.name, "read_file");
        assert_eq!(a0["file_id"], "/a");
        assert_eq!(a1["file_id"], "/b", "same-key values must not overwrite across functions");
        // Distinct-key variant: no cross-contamination either.
        let out = parse("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n\
                         </function>\n<function=get_weather>\n<parameter=days>\n3\n</parameter>\n\
                         </function>\n</tool_call>", Some(&tools()));
        assert_eq!(out.tool_calls.len(), 2);
        let b0: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        let b1: Value = serde_json::from_str(&out.tool_calls[1].function.arguments).unwrap();
        assert_eq!(b0["city"], "Paris");
        assert!(b0.get("days").is_none(), "second function's params leaked into the first");
        assert_eq!(b1["days"], 3);   // schema coercion still applies per call
        assert!(b1.get("city").is_none());
        // Multiple <tool_call> blocks per turn keep working (regression guard).
        let out = parse("<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n\
                         </function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n\
                         <parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>", Some(&tools()));
        assert_eq!(out.tool_calls.len(), 2);
    }

    /// Phase-2 A5: the three malformed JSON shapes the Phase-1 ledger proved are salvaged.
    #[test]
    fn json_salvage_shapes() {
        let tools = vec![
            serde_json::json!({"type": "function", "function": {"name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                               "days": {"type": "integer"}}}}}),
        ];
        // (a) wrapper {"function": T, "arguments": {...}}
        let out = parse("<tool_call>\n{\"function\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>", Some(&tools));
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].function.name, "get_weather");
        let a: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        assert_eq!(a["city"], "Paris");
        // (b) flattened {"function": T, k: v, ...}
        let out = parse("<tool_call>\n{\"function\": \"get_weather\", \"city\": \"Paris\", \"days\": 3}\n</tool_call>", Some(&tools));
        assert_eq!(out.tool_calls.len(), 1);
        let a: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        assert_eq!(a["city"], "Paris");
        assert_eq!(a["days"], 3);
        // (c) invalid name-first {T, "k": "v"}
        let out = parse("<tool_call>\n{\"get_weather\", \"city\": \"Oslo\"}\n</tool_call>", Some(&tools));
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].function.name, "get_weather");
        let a: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        assert_eq!(a["city"], "Oslo");
        // Strict OpenAI form still wins unchanged.
        let out = parse("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}\n</tool_call>", Some(&tools));
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].function.name, "get_weather");
        // A JSON object that is NOT a tool call is not promoted to one.
        let out = parse("<tool_call>\n{\"function\": \"not_a_tool\", \"arguments\": {}}\n</tool_call>", Some(&tools));
        assert!(out.tool_calls.is_empty(), "unknown name must not become a call");
    }

    /// Phase-4 A1 acceptance: the SIX archived `json-body + stray </parameter>` bodies the model
    /// literally emitted in the Phase-3 audit (phase3/EMPTY_ARGS_EVIDENCE.md, requests 2, 33, 38,
    /// 40, 46, 50). Every one of them reached the harness as `{}` before the fix — 6/6 lost.
    /// The fixtures are the emitted bytes, and the assertions are the exact arguments.
    #[test]
    fn phase4_archived_json_body_with_stray_parameter_tag() {
        let tools = vec![
            serde_json::json!({"type":"function","function":{"name":"send_email","parameters":{"type":"object","properties":{
                "to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}}}}}),
            serde_json::json!({"type":"function","function":{"name":"book_room","parameters":{"type":"object","properties":{
                "room_id":{"type":"string"},"date":{"type":"string"},"time":{"type":"string"},
                "duration_minutes":{"type":"integer"},"attendees":{"type":"array"}}}}}),
            serde_json::json!({"type":"function","function":{"name":"get_incident","parameters":{"type":"object","properties":{
                "incident_id":{"type":"string"}}}}}),
            serde_json::json!({"type":"function","function":{"name":"update_incident","parameters":{"type":"object","properties":{
                "incident_id":{"type":"string"},"expected_version":{"type":"integer"},"severity":{"type":"string"},
                "assignee":{"type":"string"},"tags":{"type":"array"}}}}}),
            serde_json::json!({"type":"function","function":{"name":"list_incidents","parameters":{"type":"object","properties":{
                "status":{"type":"string"},"quarter":{"type":"string"},"page_token":{"type":"string"}}}}}),
            serde_json::json!({"type":"function","function":{"name":"get_oncall_route","parameters":{"type":"object","properties":{}}}}),
        ];
        // (tool name, raw emitted body, expected arguments) — verbatim from the archive.
        let cases: Vec<(&str, &str, Value)> = vec![
            // request #2 — TC-18 / TC-87
            ("send_email",
             r#"{"to":"hans.mueller@firma.de","subject":"Meeting Terminänderung","body":"Der Termin wurde auf 15 Uhr verschoben. Bitte bestätigen Sie Ihre Teilnahme."}"#,
             serde_json::json!({"to":"hans.mueller@firma.de","subject":"Meeting Terminänderung",
                                "body":"Der Termin wurde auf 15 Uhr verschoben. Bitte bestätigen Sie Ihre Teilnahme."})),
            // request #33 — TC-84
            ("book_room",
             r#"{"room_id":"berlin_5b","date":"2026-03-25","time":"14:00","duration_minutes":45,"attendees":["elena@company.com","ravi@company.com"]}"#,
             serde_json::json!({"room_id":"berlin_5b","date":"2026-03-25","time":"14:00","duration_minutes":45,
                                "attendees":["elena@company.com","ravi@company.com"]})),
            // request #38 — TC-86
            ("get_incident", r#"{"incident_id":"INC-442"}"#,
             serde_json::json!({"incident_id":"INC-442"})),
            // request #40 — TC-86
            ("update_incident",
             r#"{"incident_id":"INC-442","expected_version":8,"severity":"P1","assignee":"Mika","tags":["customer-impact","database"]}"#,
             serde_json::json!({"incident_id":"INC-442","expected_version":8,"severity":"P1","assignee":"Mika",
                                "tags":["customer-impact","database"]})),
            // request #46 — TC-87 (page 2)
            ("list_incidents", r#"{"status":"open","quarter":"Q3","page_token":"p2"}"#,
             serde_json::json!({"status":"open","quarter":"Q3","page_token":"p2"})),
            // request #50 — TC-87 (the digest email; body is the longest archived emission)
            ("send_email",
             r#"{"to":"oncall@company.com","subject":"Q3 Open P1 Incident Digest","body":"Q3 Open P1 Incident Digest\n\nPagination completed (final page confirmed, next_page_token = null).\n\nDeduplicated open P1 incidents (first-seen order):\n1. INC-901\n2. INC-902\n3. INC-903\n4. INC-904\n5. INC-905\n6. INC-906\n\nExact count: 6\n\nDuplicates removed: INC-902 (repeated on page 2), INC-905 (repeated on page 4).","incident_ids":["INC-901","INC-902","INC-903","INC-904","INC-905","INC-906"],"exact_count":6}"#,
             serde_json::json!({"to":"oncall@company.com","subject":"Q3 Open P1 Incident Digest",
                "body":"Q3 Open P1 Incident Digest\n\nPagination completed (final page confirmed, next_page_token = null).\n\nDeduplicated open P1 incidents (first-seen order):\n1. INC-901\n2. INC-902\n3. INC-903\n4. INC-904\n5. INC-905\n6. INC-906\n\nExact count: 6\n\nDuplicates removed: INC-902 (repeated on page 2), INC-905 (repeated on page 4).",
                "incident_ids":["INC-901","INC-902","INC-903","INC-904","INC-905","INC-906"],"exact_count":6})),
        ];
        for (name, body, want) in cases {
            let raw = format!("prose\n<tool_call>\n<function={name}>\n{body}\n</parameter>\n</function>\n</tool_call>");
            let out = parse(&raw, Some(&tools));
            assert_eq!(out.tool_calls.len(), 1, "one call for {name}");
            let tc = &out.tool_calls[0];
            assert_eq!(tc.function.name, name);
            let got: Value = serde_json::from_str(&tc.function.arguments).unwrap();
            assert_eq!(got, want, "arguments lost/incorrect for {name}: {}", tc.function.arguments);
            assert_ne!(tc.function.arguments, "{}", "arguments must not come back empty");
            // The A2 alarm must be silent when the arguments were recovered.
            assert!(empty_arg_alarms(&raw, Some(&tools), &out.tool_calls).is_empty());
        }
        // Variant with no `</function>` (the model sometimes omits it before `</tool_call>`).
        let raw = "<tool_call>\n<function=get_incident>\n{\"incident_id\":\"INC-442\"}\n</parameter>\n</tool_call>";
        let out = parse(raw, Some(&tools));
        let a: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        assert_eq!(a["incident_id"], "INC-442");
        // Clean JSON body inside the function tag (5/5 in the audit) keeps working unchanged.
        let raw = "<tool_call>\n<function=get_incident>\n{\"incident_id\":\"INC-999\"}\n</function>\n</tool_call>";
        let out = parse(raw, Some(&tools));
        let a: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        assert_eq!(a["incident_id"], "INC-999");
        // A genuinely EMPTY body (the one legitimate zero-arg call in the audit) is still `{}` and
        // is NOT alarmed: there was nothing to lose.
        let raw = "<tool_call>\n<function=get_oncall_route>\n</function>\n</tool_call>";
        let out = parse(raw, Some(&tools));
        assert_eq!(out.tool_calls[0].function.arguments, "{}");
        assert!(empty_arg_alarms(raw, Some(&tools), &out.tool_calls).is_empty());
        // Two `<function=>` blocks, the first carrying the hybrid body — the second must still own
        // its own arguments (Phase-2 A2 must not regress).
        let raw = "<tool_call>\n<function=list_incidents>\n{\"status\":\"open\",\"quarter\":\"Q3\",\"page_token\":\"p2\"}\n</parameter>\n\
                   </function>\n<function=get_incident>\n{\"incident_id\":\"INC-442\"}\n</parameter>\n</function>\n</tool_call>";
        let out = parse(raw, Some(&tools));
        assert_eq!(out.tool_calls.len(), 2);
        let a0: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        let a1: Value = serde_json::from_str(&out.tool_calls[1].function.arguments).unwrap();
        assert_eq!(a0["page_token"], "p2");
        assert_eq!(a1["incident_id"], "INC-442");
    }

    /// Phase-4 A2: a parameterized tool arriving with `{}` while the emitted body was NON-EMPTY is
    /// reported with the raw body; a zero-arg tool (schema declares no parameters) never is.
    #[test]
    fn empty_arg_alarm_reports_raw_body() {
        let tools = vec![
            serde_json::json!({"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{
                "city":{"type":"string"}}}}}),
            serde_json::json!({"type":"function","function":{"name":"get_oncall_route","parameters":{"type":"object","properties":{}}}}),
        ];
        // A body the JSON fallback cannot read (unterminated object): args stay {}, call survives,
        // and the raw body is alarmed rather than dropped silently.
        let raw = "<tool_call>\n<function=get_weather>\n{\"city\": \"Paris\"\n</function>\n</tool_call>";
        let out = parse(raw, Some(&tools));
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].function.arguments, "{}");
        let alarms = empty_arg_alarms(raw, Some(&tools), &out.tool_calls);
        assert_eq!(alarms.len(), 1);
        assert!(alarms[0].contains("\"city\": \"Paris\""), "alarm must carry the raw body: {alarms:?}");
        // Zero-arg tool: empty arguments are legitimate.
        let raw = "<tool_call>\n<function=get_oncall_route>\n</function>\n</tool_call>";
        let out = parse(raw, Some(&tools));
        assert!(empty_arg_alarms(raw, Some(&tools), &out.tool_calls).is_empty());
        // Unknown catalog: no schema to compare against, so no alarm (never guess).
        assert!(empty_arg_alarms(raw, None, &out.tool_calls).is_empty());
    }

    // ───────────────────────── Phase-5 B1: unescaped-quote repair ─────────────────────────

    fn p5_tools() -> Vec<Value> {
        vec![
            serde_json::json!({"type":"function","function":{"name":"run_code","parameters":{"type":"object","properties":{
                "language":{"type":"string"},"code":{"type":"string"}}}}}),
            serde_json::json!({"type":"function","function":{"name":"list_incidents","parameters":{"type":"object","properties":{
                "status":{"type":"string"},"quarter":{"type":"string"},"page_token":{"type":"string"}}}}}),
            serde_json::json!({"type":"function","function":{"name":"get_oncall_route","parameters":{"type":"object","properties":{}}}}),
        ]
    }

    fn p5_args(raw: &str, tools: &[Value]) -> (Value, usize) {
        let out = parse(raw, Some(tools));
        assert_eq!(out.tool_calls.len(), 1, "exactly one call expected from {raw}");
        let a: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        (a, empty_arg_alarms(raw, Some(tools), &out.tool_calls).len())
    }

    /// Phase-5 B1 acceptance: the ARCHIVED NVFP4 alarm body — the model wrote a code string
    /// containing unescaped quotes (`print("correct")`) inside a JSON body, so serde rejected
    /// the whole object and every argument was lost (phase4/N1a+N1b `[tool-args-alarm]`, 1
    /// occurrence per leg). The repair escapes only the quotes that cannot close the string.
    #[test]
    fn phase5_unescaped_quote_body_is_repaired() {
        let tools = p5_tools();
        let (a, alarms) = p5_args(
            "<tool_call>\n<function=run_code>\n{\"language\":\"python\",\"code\":\"print(\"correct\")\"}\n</function>\n</tool_call>",
            &tools);
        assert_eq!(a["language"], "python");
        assert_eq!(a["code"], r#"print("correct")"#, "the inner quotes must survive verbatim");
        assert_eq!(alarms, 0, "a recovered body must NOT be alarmed");

        // Same body in the hybrid shape (JSON body + stray </parameter>) the FP8 lane emits.
        let (a, alarms) = p5_args(
            "<tool_call>\n<function=run_code>\n{\"language\":\"python\",\"code\":\"print(\"correct\")\"}\n</parameter>\n</function>\n</tool_call>",
            &tools);
        assert_eq!(a["code"], r#"print("correct")"#);
        assert_eq!(alarms, 0);

        // Hostile case: a code snippet with braces AND unescaped quotes inside the value.
        let (a, _) = p5_args(
            "<tool_call>\n<function=run_code>\n{\"language\":\"python\",\"code\":\"if (x) { print(\"y\") }\"}\n</function>\n</tool_call>",
            &tools);
        assert_eq!(a["code"], r#"if (x) { print("y") }"#);

        // Multiple stray quotes in one value.
        let (a, _) = p5_args(
            "<tool_call>\n<function=run_code>\n{\"code\":\"a(\"b\")c(\"d\")e\"}\n</function>\n</tool_call>",
            &tools);
        assert_eq!(a["code"], r#"a("b")c("d")e"#);

        // A stray quote in a KEY is repaired too (the same lexer rule applies).
        let (a, _) = p5_args(
            "<tool_call>\n<function=run_code>\n{\"co\"de\":\"x\"}\n</function>\n</tool_call>",
            &tools);
        assert_eq!(a["co\"de"], "x");

        // Empty string values are untouched by the repair (already valid JSON).
        let (a, _) = p5_args(
            "<tool_call>\n<function=run_code>\n{\"language\":\"python\",\"code\":\"\"}\n</function>\n</tool_call>",
            &tools);
        assert_eq!(a["code"], "");
    }

    /// NEGATIVE CONTROL: with the repair DISABLED the archived body is NOT recovered — it comes
    /// back as `{}` and the alarm fires, which is exactly the pre-Phase-5 behavior. If this test
    /// ever passes with the repair off, some other path is doing the work and the B1 claim is
    /// false.
    #[test]
    fn phase5_negative_control_repair_disabled() {
        let tools = p5_tools();
        let raw = "<tool_call>\n<function=run_code>\n{\"language\":\"python\",\"code\":\"print(\"correct\")\"}\n</function>\n</tool_call>";
        set_repair_quotes(false);
        let (a, alarms) = p5_args(raw, &tools);
        assert_eq!(a, serde_json::json!({}), "with the repair OFF the arguments must be lost");
        assert_eq!(alarms, 1, "with the repair OFF the alarm must fire");
        set_repair_quotes(true);
        let (a, alarms) = p5_args(raw, &tools);
        assert_eq!(a["code"], r#"print("correct")"#, "with the repair ON the arguments return");
        assert_eq!(alarms, 0);
    }

    /// The repair must never COMPLETE a truncated body (Phase-5 B2 policy: a silently closed
    /// JSON is a wrong-args hazard). Archived occurrence: phase4/R3 (FP8, parallel 3).
    #[test]
    fn phase5_truncated_body_is_not_salvaged() {
        let tools = p5_tools();
        let raw = "<tool_call>\n<function=list_incidents>\n{\"status\":\"open\",\"quarter\":\"Q3\",\"page_token\":\"p4\"\n</parameter>\n</function>\n</tool_call>";
        let (a, alarms) = p5_args(raw, &tools);
        assert_eq!(a, serde_json::json!({}), "a truncated body must never be auto-completed");
        assert_eq!(alarms, 1, "a truncated body must stay alarmed");
        // A body whose final `}` never arrived AND whose string never closed, inside a block
        // that was itself cut off (no </function>): the whole call is dropped — never
        // half-parsed, never auto-completed (the pre-existing truncation rule).
        let raw2 = "<tool_call>\n<function=run_code>\n{\"code\":\"print(\"correct)";
        let out2 = parse(raw2, Some(&tools));
        assert!(out2.tool_calls.is_empty(), "a truncated block must yield NO call at all");
        // Control: the SAME content with a closing `}` and closing quote IS a complete object
        // (the model wrote an unbalanced snippet, not a truncated one) — the repair salvages it
        // and the value is what the model literally wrote. Documented, not accidental.
        let raw2b = "<tool_call>\n<function=run_code>\n{\"code\":\"print(\"correct)\"}\n</function>\n</tool_call>";
        let (a2b, alarms2b) = p5_args(raw2b, &tools);
        assert_eq!(a2b["code"], "print(\"correct)");
        assert_eq!(alarms2b, 0);
        // Missing comma between two pairs: escaping both stray quotes must NOT rescue it (the
        // repaired text still fails to parse), so the alarm stays.
        let raw3 = "<tool_call>\n<function=run_code>\n{\"language\":\"x\" \"code\":\"y\"}\n</function>\n</tool_call>";
        let (a3, alarms3) = p5_args(raw3, &tools);
        assert_eq!(a3, serde_json::json!({}));
        assert_eq!(alarms3, 1);
    }

    /// The repair is a no-op on input that is already valid JSON: `repair_unescaped_quotes`
    /// returns None (nothing changed), and the strict path handles the body unchanged.
    #[test]
    fn phase5_repair_leaves_valid_json_alone() {
        let body = r#"{"language":"python","code":"print(\"correct\")"}"#;
        assert!(repair_unescaped_quotes(body).is_none(), "valid JSON must not be rewritten");
        let tools = p5_tools();
        let (a, alarms) = p5_args(
            &format!("<tool_call>\n<function=run_code>\n{body}\n</function>\n</tool_call>"),
            &tools);
        assert_eq!(a["code"], r#"print("correct")"#);
        assert_eq!(alarms, 0);
        // A nested object/array value with braces is untouched.
        let body2 = r#"{"language":"python","code":"d = {\"k\": [1, 2]}"}"#;
        assert!(repair_unescaped_quotes(body2).is_none());
    }

    /// Strict-form repair: the model wraps an unescaped-quote body in the OpenAI shape, which
    /// used to fall through to the XML path and produce an `arguments` object containing the
    /// whole envelope. With the repair the call is resolved by the strict path.
    #[test]
    fn phase5_strict_form_with_unescaped_quotes() {
        let tools = p5_tools();
        let raw = "<tool_call>\n{\"name\":\"run_code\",\"arguments\":{\"language\":\"python\",\"code\":\"print(\"hi\")\"}}\n</tool_call>";
        let out = parse(raw, Some(&tools));
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].function.name, "run_code");
        let a: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        assert_eq!(a["code"], r#"print("hi")"#);
        assert!(empty_arg_alarms(raw, Some(&tools), &out.tool_calls).is_empty());
    }
}
