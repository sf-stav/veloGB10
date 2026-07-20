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

/// Everything the model produced, split into the prose it wrote and the calls it made.
pub struct ParsedOutput {
    /// Text outside any `<tool_call>` block. The template explicitly permits reasoning BEFORE a call.
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

/// True if `s` contains the start of a tool call — used by the streaming path to decide whether it
/// must hold text back rather than forward it to the client as content.
pub fn has_tool_call(s: &str) -> bool {
    s.contains(CALL_OPEN)
}

/// Parse a completed generation. `tools` is the request's `tools` array, used only to type-coerce
/// arguments; parsing still works (as strings) when it is absent.
pub fn parse(text: &str, tools: Option<&[Value]>) -> ParsedOutput {
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
        if let Some(tc) = parse_one(&after[..close], tools, tool_calls.len()) {
            tool_calls.push(tc);
        }
        rest = &after[close + CALL_CLOSE.len()..];
    }
    content.push_str(rest);

    ParsedOutput { content: content.trim().to_string(), tool_calls }
}

/// Parse the inside of one `<tool_call>…</tool_call>`.
fn parse_one(body: &str, tools: Option<&[Value]>, idx: usize) -> Option<ToolCall> {
    let fopen = body.find("<function=")?;
    let after = &body[fopen + "<function=".len()..];
    let gt = after.find('>')?;
    let name = after[..gt].trim().to_string();
    if name.is_empty() { return None; }

    let schema = tools.and_then(|ts| param_schema(ts, &name));

    let mut args = serde_json::Map::new();
    let mut rest = &after[gt + 1..];
    while let Some(popen) = rest.find("<parameter=") {
        let a = &rest[popen + "<parameter=".len()..];
        let Some(gt2) = a.find('>') else { break };
        let key = a[..gt2].trim().to_string();
        let vstart = &a[gt2 + 1..];
        let Some(pclose) = vstart.find("</parameter>") else { break };
        // The template puts a newline after `>` and before `</parameter>`; they are delimiters, not
        // part of the value. Trim only those, so interior whitespace of a multi-line value survives.
        let raw = vstart[..pclose].trim_matches('\n');
        args.insert(key.clone(), coerce(raw, schema.and_then(|s| s.get(&key))));
        rest = &vstart[pclose + "</parameter>".len()..];
    }

    let _ = idx;
    Some(ToolCall {
        id: format!("call_{}", CALL_SEQ.fetch_add(1, Ordering::Relaxed)),
        kind: "function".to_string(),
        function: FunctionCall {
            name,
            arguments: serde_json::to_string(&Value::Object(args)).unwrap_or_else(|_| "{}".into()),
        },
    })
}

/// `tools[i].function.parameters.properties` for the named function.
fn param_schema<'a>(tools: &'a [Value], name: &str) -> Option<&'a serde_json::Map<String, Value>> {
    tools.iter()
        .find(|t| t.pointer("/function/name").and_then(Value::as_str) == Some(name))
        .and_then(|t| t.pointer("/function/parameters/properties"))
        .and_then(Value::as_object)
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
}
