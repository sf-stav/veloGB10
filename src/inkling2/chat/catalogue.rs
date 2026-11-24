//! The tools a request offered, and the checks a parsed call must pass before it is reported as a call (owner hardening rules of
//! 2026-10-10: never invent a call, check against the offered catalogue, prefer fewer calls on doubt). A call that fails stays text
//! in the reply; nothing is repaired.

use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reject { NoCatalogue, EmptyName, NameMismatch, UnknownTool, ArgsNotObject, DuplicateParam, MissingRequired, TypeMismatch, NotInEnum, BadJson }

impl Reject {
    pub fn label(self) -> &'static str {
        match self {
            Reject::NoCatalogue => "no_catalogue", Reject::EmptyName => "empty_name", Reject::NameMismatch => "name_mismatch", Reject::UnknownTool => "unknown_tool",
            Reject::ArgsNotObject => "args_not_object", Reject::DuplicateParam => "duplicate_param",
            Reject::MissingRequired => "missing_required", Reject::TypeMismatch => "type_mismatch",
            Reject::NotInEnum => "not_in_enum", Reject::BadJson => "bad_json",
        }
    }
}

#[derive(Debug, Clone)]
struct ToolDef { name: String, parameters: Option<Value> }

#[derive(Debug, Clone, Default)]
pub struct Catalogue { tools: Vec<ToolDef> }

impl Catalogue {
    /// From the request's `tools` array (OpenAI `{type, function: {name, parameters}}`; a bare `{name, parameters}` is accepted, as the template does).
    pub fn from_openai_tools(tools: &[Value]) -> Self {
        Self { tools: tools.iter().filter_map(|t| {
            let f = t.get("function").unwrap_or(t);
            Some(ToolDef { name: f.get("name")?.as_str()?.to_string(), parameters: f.get("parameters").cloned() })
        }).collect() }
    }
    /// `tool_choice` naming one function: only that call can pass.
    pub fn only(&self, name: &str) -> Self { Self { tools: self.tools.iter().filter(|t| t.name == name).cloned().collect() } }
    pub fn is_empty(&self) -> bool { self.tools.is_empty() }
    pub fn names(&self) -> Vec<&str> { self.tools.iter().map(|t| t.name.as_str()).collect() }

    /// Check the payload `{"name": ..., "args": {...}}` as the model wrote it (`body` is the text between the content-kind token and
    /// `<|end_message|>`). On success: the tool name and the `arguments` JSON text (compact, keys in the model's order, non-ASCII as written).
    pub fn check_payload(&self, header: &str, body: &str) -> Result<(String, String), Reject> {
        if self.tools.is_empty() { return Err(Reject::NoCatalogue); }
        let v: Value = serde_json::from_str(body.trim()).map_err(|_| Reject::BadJson)?;
        let obj = v.as_object().ok_or(Reject::BadJson)?;
        let name = obj.get("name").and_then(|n| n.as_str()).ok_or(Reject::BadJson)?;
        if name.is_empty() { return Err(Reject::EmptyName); }
        let args = obj.get("args").ok_or(Reject::ArgsNotObject)?;
        if !args.is_object() { return Err(Reject::ArgsNotObject); }
        let def = self.tools.iter().find(|t| t.name == name).ok_or(Reject::UnknownTool)?;
        // the header names the callee too (`<|message_model|>NAME<|content_invoke_tool_json|>`); two names for one call is doubt
        if !header.trim().is_empty() && header.trim() != name { return Err(Reject::NameMismatch); }
        // a repeated key is a call the model did not make (the last one would silently win): look at the raw text
        if has_duplicate_keys(body) { return Err(Reject::DuplicateParam); }
        if let Some(schema) = def.parameters.as_ref() {
            let a = args.as_object().expect("checked");
            if let Some(Value::Array(req)) = schema.get("required") {
                if req.iter().filter_map(|r| r.as_str()).any(|r| !a.contains_key(r)) { return Err(Reject::MissingRequired); }
            }
            if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
                for (k, val) in a {
                    let Some(ps) = props.get(k) else { continue };
                    if let Some(t) = ps.get("type") { if !type_ok(t, val) { return Err(Reject::TypeMismatch); } }
                    if let Some(Value::Array(en)) = ps.get("enum") { if !en.contains(val) { return Err(Reject::NotInEnum); } }
                }
            }
        }
        Ok((name.to_string(), serde_json::to_string(args).map_err(|_| Reject::BadJson)?))
    }
}

fn type_ok(t: &Value, v: &Value) -> bool {
    match t {
        Value::String(s) => match s.as_str() {
            "string" => v.is_string(),
            "integer" => v.is_i64() || v.is_u64() || v.as_f64().map_or(false, |x| x.fract() == 0.0),
            "number" => v.is_number(),
            "boolean" => v.is_boolean(),
            "array" => v.is_array(),
            "object" => v.is_object(),
            "null" => v.is_null(),
            _ => true,
        },
        Value::Array(a) => a.iter().any(|x| type_ok(x, v)),
        _ => true,
    }
}

/// Does any object in `text` (valid JSON) repeat a key? Done on the raw text with a visitor that sees every pair.
fn has_duplicate_keys(text: &str) -> bool {
    use serde::de::{Deserializer, MapAccess, SeqAccess, Visitor};
    struct Any<'a>(&'a mut bool);
    impl<'de, 'a> serde::de::DeserializeSeed<'de> for Any<'a> {
        type Value = ();
        fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<(), D::Error> { d.deserialize_any(Walk(self.0)) }
    }
    struct Walk<'a>(&'a mut bool);
    impl<'de, 'a> Visitor<'de> for Walk<'a> {
        type Value = ();
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("json") }
        fn visit_bool<E>(self, _: bool) -> Result<(), E> { Ok(()) }
        fn visit_i64<E>(self, _: i64) -> Result<(), E> { Ok(()) }
        fn visit_u64<E>(self, _: u64) -> Result<(), E> { Ok(()) }
        fn visit_f64<E>(self, _: f64) -> Result<(), E> { Ok(()) }
        fn visit_str<E>(self, _: &str) -> Result<(), E> { Ok(()) }
        fn visit_unit<E>(self) -> Result<(), E> { Ok(()) }
        fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<(), A::Error> {
            while a.next_element_seed(Any(self.0))?.is_some() {}
            Ok(())
        }
        fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> Result<(), A::Error> {
            let mut seen: Vec<String> = Vec::new();
            while let Some(k) = m.next_key::<String>()? {
                if seen.contains(&k) { *self.0 = true; }
                seen.push(k);
                m.next_value_seed(Any(self.0))?;
            }
            Ok(())
        }
    }
    let mut dup = false;
    let mut de = serde_json::Deserializer::from_str(text.trim());
    let _ = serde::de::DeserializeSeed::deserialize(Any(&mut dup), &mut de);
    dup
}
