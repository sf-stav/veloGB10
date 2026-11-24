//! Tool rendering of `encoding.py`: namespace-qualified names, the `## Tools` block, and the DSML
//! encoding of assistant tool calls (spaced V4.1 tag names).

use crate::dsv4_chat::Json;

use super::error::ChatError;
use super::jx::{self, get, set, truthy};
use super::pyjson::{loads, py_str, to_json};

pub const DSML_TOKEN: &str = "｜DSML｜";
pub const THINKING_START: &str = "<think>";
pub const THINKING_END: &str = "</think>";
/// V4.1 tag names carry a leading space.
pub const TC_BLOCK_NAME: &str = " calls";
pub const TC_TAG_NAME: &str = " invoke";
pub const TP_TAG_NAME: &str = " parameter";

/// `<｜DSML｜ calls>`
pub fn calls_open() -> String {
    format!("<{DSML_TOKEN}{TC_BLOCK_NAME}>")
}
/// `</｜DSML｜ calls>`
pub fn calls_close() -> String {
    format!("</{DSML_TOKEN}{TC_BLOCK_NAME}>")
}

/// Python `str.format` for the simple `{name}` placeholders of these templates. One pass: substituted
/// text is never scanned again, so braces in tool schemas cannot collide with placeholders.
pub fn fmt_named(tmpl: &str, vars: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(tmpl.len() + 64);
    let mut rest = tmpl;
    while let Some(b) = rest.find('{') {
        out.push_str(&rest[..b]);
        match rest[b..].find('}') {
            Some(e) => {
                let name = &rest[b + 1..b + e];
                match vars.iter().find(|(n, _)| *n == name) {
                    Some((_, v)) => {
                        out.push_str(v);
                        rest = &rest[b + e + 1..];
                    }
                    None => {
                        out.push('{');
                        rest = &rest[b + 1..];
                    }
                }
            }
            None => {
                out.push('{');
                rest = &rest[b + 1..];
            }
        }
    }
    out.push_str(rest);
    out
}

const TOOLS_TEMPLATE: &str = "## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<{dsml_token}{tc_block_name}>\" block like the following:

<{dsml_token}{tc_block_name}>
<{dsml_token}{tool_call_tag_name} name=\"$TOOL_NAME\">
<{dsml_token}{tool_parameter_tag_name} name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{dsml_token}{tool_parameter_tag_name}>
...
</{dsml_token}{tool_call_tag_name}>
<{dsml_token}{tool_call_tag_name} name=\"$TOOL_NAME2\">
...
</{dsml_token}{tool_call_tag_name}>
</{dsml_token}{tc_block_name}>

String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
";

pub const RESPONSE_FORMAT_TEMPLATE: &str =
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}";

/// `render_tools`: one `json.dumps` line per function object.
pub fn render_tools(tools: &[Json]) -> String {
    let schemas: Vec<String> = tools.iter().map(to_json).collect();
    fmt_named(
        TOOLS_TEMPLATE,
        &[
            ("tool_schemas", schemas.join("\n").as_str()),
            ("dsml_token", DSML_TOKEN),
            ("tc_block_name", TC_BLOCK_NAME),
            ("tool_call_tag_name", TC_TAG_NAME),
            ("tool_parameter_tag_name", TP_TAG_NAME),
            ("thinking_start_token", THINKING_START),
            ("thinking_end_token", THINKING_END),
        ],
    )
}

/// `_split_tool_name`
pub fn split_tool_name(name: &str, namespace: Option<&str>) -> Result<(Option<String>, String), ChatError> {
    let mut namespace = namespace.map(str::to_string);
    let mut name = name.to_string();
    if let Some((prefix, bare)) = name.clone().split_once("::") {
        if !(namespace.is_none() || namespace.as_deref() == Some(prefix)) {
            return Err(ChatError::assertion(format!(
                "Conflicting tool namespaces: {} != {prefix}",
                namespace.unwrap_or_default()
            )));
        }
        namespace = Some(prefix.to_string());
        name = bare.to_string();
    }
    if name.contains("::") {
        return Err(ChatError::assertion(format!("Tool name must not contain '::': {name}")));
    }
    if let Some(ns) = &namespace {
        if ns.contains("::") {
            return Err(ChatError::assertion(format!("Tool namespace must not contain '::': {ns}")));
        }
    }
    Ok((namespace, name))
}

/// The namespace of a tool/function dict: a string, or a dict with a `name`.
fn namespace_of(tool: &Json) -> Result<Option<String>, ChatError> {
    match get(tool, "namespace") {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Str(s)) => Ok(Some(s.clone())),
        Some(o @ Json::Object(_)) => match get(o, "name") {
            Some(Json::Str(s)) => Ok(Some(s.clone())),
            _ => Err(ChatError::input("tool namespace dict has no string name")),
        },
        Some(_) => Err(ChatError::input("tool namespace must be a string or an object")),
    }
}

/// `_tool_name_for_encoding`
pub fn tool_name_for_encoding(tool: &Json) -> Result<String, ChatError> {
    let ns = namespace_of(tool)?;
    let name = jx::get_str(tool, "name").ok_or_else(|| ChatError::input("tool has no string name"))?;
    let (ns, name) = split_tool_name(name, ns.as_deref())?;
    Ok(match ns {
        None => name,
        Some(ns) => format!("{ns}::{name}"),
    })
}

/// `tools_from_openai_format`: the function objects, names qualified, namespace folded in.
pub fn tools_from_openai_format(tools: &Json) -> Result<Vec<Json>, ChatError> {
    let arr = match tools {
        Json::Array(a) => a,
        _ => return Err(ChatError::input("tools must be a list")),
    };
    let mut out = Vec::with_capacity(arr.len());
    for tool in arr {
        let mut function = match get(tool, "function") {
            Some(f @ Json::Object(_)) => f.clone(),
            _ => return Err(ChatError::input("tool has no function object")),
        };
        if jx::present(tool, "namespace") {
            set(&mut function, "namespace", get(tool, "namespace").unwrap().clone());
        }
        let qualified = tool_name_for_encoding(&function)?;
        set(&mut function, "name", Json::Str(qualified));
        let namespace = jx::remove(&mut function, "namespace");
        if let Some(ns @ Json::Object(_)) = &namespace {
            if truthy(get(ns, "description")) {
                let nsd = py_str(get(ns, "description").unwrap())?;
                let own = match get(&function, "description") {
                    Some(d) if super::pyjson::truthy(d) => py_str(d)?,
                    _ => String::new(),
                };
                set(&mut function, "description", Json::Str(format!("{nsd}\n{own}")));
            }
        }
        out.push(function);
    }
    Ok(out)
}

/// One assistant tool call in the internal shape of `tool_calls_from_openai_format`.
#[derive(Debug, Clone)]
pub struct InternalCall {
    pub namespace: Option<String>,
    pub name: String,
    pub arguments: Json,
}

/// `tool_calls_from_openai_format`
pub fn tool_calls_from_openai_format(tool_calls: &Json) -> Result<Vec<InternalCall>, ChatError> {
    let arr = match tool_calls {
        Json::Array(a) => a,
        _ => return Err(ChatError::input("tool_calls must be a list")),
    };
    let mut calls = Vec::with_capacity(arr.len());
    for tc in arr {
        let function = get(tc, "function").ok_or_else(|| ChatError::input("tool call has no function"))?;
        let fname = jx::get_str(function, "name").ok_or_else(|| ChatError::input("tool call function has no string name"))?;
        // `tool_call.get("namespace") or function.get("namespace")`
        let ns_val = match get(tc, "namespace") {
            Some(v) if super::pyjson::truthy(v) => Some(v),
            _ => get(function, "namespace"),
        };
        let ns = match ns_val {
            None | Some(Json::Null) => None,
            Some(Json::Str(s)) => Some(s.as_str()),
            Some(_) => return Err(ChatError::input("call namespace must be a string")),
        };
        let (namespace, name) = split_tool_name(fname, ns)?;
        let arguments = get(function, "arguments").ok_or_else(|| ChatError::input("tool call function has no arguments"))?.clone();
        calls.push(InternalCall { namespace, name, arguments });
    }
    Ok(calls)
}

impl InternalCall {
    /// `_tool_name_for_encoding` of the internal call.
    pub fn encoded_name(&self) -> String {
        match &self.namespace {
            None => self.name.clone(),
            Some(ns) => format!("{ns}::{}", self.name),
        }
    }
}

/// `encode_arguments_to_dsml`: JSON strings (also double encoded) become one parameter per key.
pub fn encode_arguments_to_dsml(call: &InternalCall) -> Result<String, ChatError> {
    let mut arguments = call.arguments.clone();
    if !matches!(arguments, Json::Object(_)) {
        for _ in 0..2 {
            if let Json::Str(s) = &arguments {
                match loads(s) {
                    Ok(v) => arguments = v,
                    Err(_) => break,
                }
            } else {
                break;
            }
        }
        if !matches!(arguments, Json::Object(_)) {
            arguments = Json::Object(vec![("arguments".to_string(), call.arguments.clone())]);
        }
    }
    let Json::Object(pairs) = super::pyjson::canon(&arguments) else { unreachable!() };
    let mut lines = Vec::with_capacity(pairs.len());
    for (k, v) in &pairs {
        let (is_str, value) = match v {
            Json::Str(s) => ("true", s.clone()),
            other => ("false", to_json(other)),
        };
        lines.push(fmt_named(
            "<{dsml_token}{tag} name=\"{key}\" string=\"{is_str}\">{value}</{dsml_token}{tag}>",
            &[("dsml_token", DSML_TOKEN), ("tag", TP_TAG_NAME), ("key", k), ("is_str", is_str), ("value", &value)],
        ));
    }
    Ok(lines.join("\n"))
}

/// The `tool_calls` section of an assistant turn (`"\n\n"` + the calls block), or empty.
pub fn render_tool_calls(calls: &[InternalCall]) -> Result<String, ChatError> {
    if calls.is_empty() {
        return Ok(String::new());
    }
    let mut list = Vec::with_capacity(calls.len());
    for c in calls {
        let args = encode_arguments_to_dsml(c)?;
        list.push(fmt_named(
            "<{dsml_token}{tag} name=\"{name}\">\n{arguments}\n</{dsml_token}{tag}>",
            &[("dsml_token", DSML_TOKEN), ("tag", TC_TAG_NAME), ("name", &c.encoded_name()), ("arguments", &args)],
        ));
    }
    Ok(format!("\n\n{}\n{}\n{}", calls_open(), list.join("\n"), calls_close()))
}
