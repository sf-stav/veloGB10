//! JSON-body calls inside `<tool_call>`: the strict form, the salvaged malformed shapes, unescaped-quote repair and
//! the empty-argument alarm. Map of the directory in `mod.rs`.

use super::*;

/// One JSON-body call. Strict `{"name": T, "arguments": …}` is the shipped form and is tried
/// first; the three MALFORMED shapes the Phase-1 ledger proved the model actually emits inside
/// `<tool_call>` are salvaged after it (LEDGER_REPORT.md 1.1.1):
///   * wrapper    `{"function": T, "arguments": {…}}` / `{"function": {"name": T, …}}`
///   * flattened  `{"function": T, "k": v, …}` — the remaining keys ARE the arguments
///   * name-first `{T, "k": "v"}` — not valid JSON; repaired, T is the tool name
/// The salvage shapes require the recovered name to be a tool of the request catalog (when the
/// catalog is known), so structured-output JSON with an unrelated `name` field is never promoted
/// to a call. The strict form keeps its original behaviour (no catalog check).
pub(super) fn parse_json_call(body: &str, tools: Option<&[Value]>, idx: usize) -> Option<ToolCall> {
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
pub(super) fn set_repair_quotes(on: bool) { REPAIR_QUOTES.with(|c| c.set(on)) }

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
pub(super) fn repair_unescaped_quotes(s: &str) -> Option<String> {
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
pub(super) fn json_object_lenient(jb: &str) -> Option<serde_json::Map<String, Value>> {
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
pub(super) fn strip_stray_param_close(jb: &str) -> &str {
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
pub(super) fn param_schema<'a>(tools: &'a [Value], name: &str) -> Option<&'a serde_json::Map<String, Value>> {
    tools.iter()
        .find(|t| t.pointer("/function/name").and_then(Value::as_str) == Some(name))
        .and_then(|t| t.pointer("/function/parameters/properties"))
        .and_then(Value::as_object)
}
