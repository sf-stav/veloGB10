//! `reasoning_effort`: the template accepts a number in [0.0, 0.99] or one of the names below (default 0.9). The shared tokenizer path
//! can pass an effort only as a string, so the value goes to the template through `chat_template_kwargs` (a JSON number stays a
//! number). OpenAI's no-think spellings and `enable_thinking: false` (which this template ignores) map to effort "none" with a note
//! (support it or fall back gracefully, AGENTS 1b-bis item 8); an unusable value is a 400 that names the accepted ones.

use serde_json::Value;

pub const NAMES: [(&str, f64); 7] = [("none", 0.0), ("minimal", 0.1), ("low", 0.2), ("medium", 0.7), ("high", 0.9), ("xhigh", 0.99), ("max", 0.99)];

/// The resolved effort and what to tell the operator, if anything.
#[derive(Debug, Clone, PartialEq)]
pub struct Effort {
    /// What the template receives as `reasoning_effort` (None = pass nothing: the template's own 0.9).
    pub value: Option<Value>,
    /// The number the template will print (`Thinking effort level: N`), for `/metrics` and logs.
    pub level: f64,
    /// One-line notice when the request's spelling was mapped (counted by the caller).
    pub mapped: Option<String>,
}

fn name_level(s: &str) -> Option<f64> { NAMES.iter().find(|(n, _)| *n == s).map(|(_, v)| *v) }

fn one(v: &Value) -> Result<(Value, f64, Option<String>), String> {
    match v {
        Value::Number(n) => {
            let x = n.as_f64().ok_or("reasoning_effort is not a finite number")?;
            if !(0.0..=0.99).contains(&x) { return Err(format!("reasoning_effort {x} is outside [0.0, 0.99]")); }
            Ok((v.clone(), x, None))
        }
        Value::String(s) => {
            let k = s.trim();
            if let Some(x) = name_level(k) { return Ok((Value::String(k.to_string()), x, None)); }
            // OpenAI's / this engine's no-think spellings: the template's own off value is "none"
            if matches!(k.to_ascii_lowercase().as_str(), "off" | "no_think" | "disabled" | "false") {
                return Ok((Value::String("none".into()), 0.0, Some(format!("reasoning_effort {k:?} mapped to \"none\" (this template's off value)"))));
            }
            // a number sent as text
            if let Ok(x) = k.parse::<f64>() {
                if (0.0..=0.99).contains(&x) {
                    return Ok((serde_json::json!(x), x, Some(format!("reasoning_effort {k:?} read as the number {x}"))));
                }
            }
            Err(format!("reasoning_effort {s:?} is not one of none, minimal, low, medium, high, xhigh, max or a number in [0.0, 0.99]"))
        }
        _ => Err("reasoning_effort must be a string or a number".into()),
    }
}

/// Precedence, most specific first: `chat_template_kwargs.reasoning_effort`, the request's `reasoning_effort`,
/// `chat_template_kwargs.enable_thinking == false` (no-think), the server's `--reasoning-effort`, nothing (the template's 0.9).
pub fn resolve(kwargs: Option<&Value>, request: Option<&Value>, server: Option<&str>) -> Result<Effort, String> {
    let kw_effort = kwargs.and_then(|k| k.get("reasoning_effort")).filter(|v| !v.is_null());
    let req = request.filter(|v| !v.is_null());
    let no_think_kw = kwargs.and_then(|k| k.get("enable_thinking")).and_then(|v| v.as_bool()) == Some(false);
    let server_v = server.map(|s| Value::String(s.to_string()));
    let pick = kw_effort.or(req);
    if let Some(v) = pick {
        let (val, level, mapped) = one(v)?;
        return Ok(Effort { value: Some(val), level, mapped });
    }
    if no_think_kw {
        return Ok(Effort { value: Some(Value::String("none".into())), level: 0.0,
                           mapped: Some("chat_template_kwargs.enable_thinking=false: this template has no such switch; rendered with reasoning_effort \"none\"".into()) });
    }
    if let Some(v) = server_v.as_ref() {
        let (val, level, mapped) = one(v).map_err(|e| format!("--reasoning-effort: {e}"))?;
        return Ok(Effort { value: Some(val), level, mapped });
    }
    Ok(Effort { value: None, level: 0.9, mapped: None })
}
