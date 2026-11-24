//! The tools a request offered, and the checks a parsed call must pass before it is reported as a
//! call (owner hardening rules of 2026-10-10: never invent a call, check against the offered
//! catalogue). A call that fails stays text; nothing is repaired.

use crate::dsv4_chat::{parse_json, Json};

use super::error::ChatError;
use super::jx::{get, get_str};
use super::pyjson::{canon, py_number, to_json};
use super::tools::tools_from_openai_format;

/// Why a call was refused (the label goes to `/metrics`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reject {
    UnknownTool,
    DuplicateParam,
    UnknownParam,
    MissingRequired,
    TypeMismatch,
    BadJson,
    NotInEnum,
    EmptyName,
}

impl Reject {
    pub fn label(self) -> &'static str {
        match self {
            Reject::UnknownTool => "unknown_tool",
            Reject::DuplicateParam => "duplicate_param",
            Reject::UnknownParam => "unknown_param",
            Reject::MissingRequired => "missing_required",
            Reject::TypeMismatch => "type_mismatch",
            Reject::BadJson => "bad_json",
            Reject::NotInEnum => "not_in_enum",
            Reject::EmptyName => "empty_name",
        }
    }
}

/// A parameter as the model wrote it: `string` is the attribute, `None` when it was absent.
#[derive(Debug, Clone)]
pub struct RawParam {
    pub name: String,
    pub value: String,
    pub string: Option<bool>,
}

#[derive(Debug, Clone)]
struct ToolDef {
    /// Namespace-qualified name as rendered in the prompt (`ns::name` or `name`).
    name: String,
    parameters: Option<Json>,
}

#[derive(Debug, Clone)]
pub struct Catalogue {
    tools: Vec<ToolDef>,
    any: bool,
}

/// How a checked value is written into the `arguments` JSON.
#[derive(Debug, Clone)]
pub struct CheckedParam {
    pub name: String,
    /// The text for `arguments`: a JSON string literal for string values, the model's own JSON text otherwise.
    pub json_text: String,
}

impl Catalogue {
    /// From the request's `tools` array (OpenAI shape, namespaces folded in as the encoder does).
    pub fn from_openai_tools(tools: &Json) -> Result<Self, ChatError> {
        let funcs = tools_from_openai_format(tools)?;
        let tools = funcs
            .into_iter()
            .filter_map(|f| {
                let name = get_str(&f, "name")?.to_string();
                Some(ToolDef { name, parameters: get(&f, "parameters").cloned() })
            })
            .collect();
        Ok(Self { tools, any: false })
    }

    /// Accept any tool name and any parameters. For tests and diagnostics only: serving never uses it.
    pub fn allow_any() -> Self {
        Self { tools: Vec::new(), any: true }
    }

    /// Only the named tool (`tool_choice` naming a function).
    pub fn only(&self, name: &str) -> Self {
        Self { tools: self.tools.iter().filter(|t| t.name == name).cloned().collect(), any: false }
    }

    /// True when no call can ever pass (so the parser need not look for calls at all).
    pub fn is_empty(&self) -> bool {
        !self.any && self.tools.is_empty()
    }

    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name.as_str()).collect()
    }

    /// Check a call. On success, the parameters in the order the model wrote them, ready to join.
    pub fn check(&self, name: &str, params: &[RawParam]) -> Result<Vec<CheckedParam>, Reject> {
        if name.is_empty() {
            return Err(Reject::EmptyName);
        }
        let schema: Option<&Json> = if self.any {
            None
        } else {
            match self.tools.iter().find(|t| t.name == name) {
                Some(t) => t.parameters.as_ref(),
                None => return Err(Reject::UnknownTool),
            }
        };
        for (i, p) in params.iter().enumerate() {
            if params[..i].iter().any(|q| q.name == p.name) {
                return Err(Reject::DuplicateParam);
            }
        }
        let props = schema.and_then(|s| get(s, "properties")).and_then(|p| p.as_object());
        if let (Some(s), Some(props)) = (schema, props) {
            let extra_ok = match get(s, "additionalProperties") {
                Some(Json::Bool(true)) | Some(Json::Object(_)) => true,
                _ => false,
            };
            if !extra_ok && params.iter().any(|p| !props.iter().any(|(k, _)| k == &p.name)) {
                return Err(Reject::UnknownParam);
            }
        }
        if let Some(Json::Array(req)) = schema.and_then(|s| get(s, "required")) {
            for r in req {
                if let Json::Str(r) = r {
                    if !params.iter().any(|p| &p.name == r) {
                        return Err(Reject::MissingRequired);
                    }
                }
            }
        }
        let mut out = Vec::with_capacity(params.len());
        for p in params {
            let prop = props.and_then(|ps| ps.iter().find(|(k, _)| k == &p.name)).map(|(_, v)| v);
            out.push(check_value(p, prop)?);
        }
        Ok(out)
    }
}

/// The JSON types a property schema allows; `None` when the schema does not say.
fn allowed_types(schema: &Json) -> Option<Vec<String>> {
    if let Some(t) = get(schema, "type") {
        return match t {
            Json::Str(s) => Some(vec![s.clone()]),
            Json::Array(a) => {
                let v: Vec<String> = a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect();
                if v.len() == a.len() { Some(v) } else { None }
            }
            _ => None,
        };
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(Json::Array(alts)) = get(schema, key) {
            let mut all = Vec::new();
            for a in alts {
                all.extend(allowed_types(a)?);
            }
            return Some(all);
        }
    }
    None
}

fn json_type_matches(v: &Json, types: &[String]) -> bool {
    types.iter().any(|t| match (t.as_str(), v) {
        ("string", Json::Str(_)) | ("boolean", Json::Bool(_)) | ("null", Json::Null) | ("array", Json::Array(_)) | ("object", Json::Object(_)) => true,
        ("number", Json::Num(_)) => true,
        ("integer", Json::Num(raw)) => {
            let n = py_number(raw);
            !n.contains(['.', 'e', 'E']) || n.parse::<f64>().map_or(false, |f| f.is_finite() && f.fract() == 0.0)
        }
        _ => false,
    })
}

fn in_enum(schema: &Json, v: &Json) -> bool {
    match get(schema, "enum") {
        Some(Json::Array(items)) => {
            let cv = canon(v);
            items.iter().any(|i| canon(i) == cv)
        }
        _ => true,
    }
}

fn check_value(p: &RawParam, prop: Option<&Json>) -> Result<CheckedParam, Reject> {
    let types = prop.and_then(allowed_types);
    let string_ok = types.as_ref().map_or(true, |t| t.iter().any(|x| x == "string"));
    let as_string = |p: &RawParam| -> Result<CheckedParam, Reject> {
        if !string_ok {
            return Err(Reject::TypeMismatch);
        }
        if let Some(pr) = prop {
            if !in_enum(pr, &Json::Str(p.value.clone())) {
                return Err(Reject::NotInEnum);
            }
        }
        Ok(CheckedParam { name: p.name.clone(), json_text: to_json(&Json::Str(p.value.clone())) })
    };
    let as_json = |p: &RawParam| -> Result<CheckedParam, Reject> {
        let v = parse_json(&p.value).map_err(|_| Reject::BadJson)?;
        if let Some(t) = &types {
            if !json_type_matches(&v, t) {
                return Err(Reject::TypeMismatch);
            }
        }
        if let Some(pr) = prop {
            if !in_enum(pr, &v) {
                return Err(Reject::NotInEnum);
            }
        }
        Ok(CheckedParam { name: p.name.clone(), json_text: p.value.clone() })
    };
    match p.string {
        Some(true) => as_string(p),
        Some(false) => as_json(p),
        None => {
            // attribute absent: the schema decides, else a JSON value reads as JSON and anything else as text
            match &types {
                Some(t) if t.iter().all(|x| x == "string") => as_string(p),
                _ => as_json(p).or_else(|_| as_string(p)),
            }
        }
    }
}
