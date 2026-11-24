//! `reasoning_effort` of V4.1: an integer budget 1..=100 or the aliases `low` / `high` / `max`.

use crate::dsv4_chat::Json;

use super::error::ChatError;

pub const REASONING_EFFORT_MAPPINGS: [(&str, u32); 3] = [("low", 50), ("high", 75), ("max", 100)];
pub const DEFAULT_REASONING_EFFORT: &str = "high";

/// The numeric budget for a request value. `None` is the default (`high`, 75). A bool or a float is
/// refused (`type(effort) is int` in the reference), as is any other string or an integer out of range.
pub fn budget(effort: Option<&Json>) -> Result<u32, ChatError> {
    let bad = |what: String| {
        ChatError::assertion(format!(
            "Invalid reasoning effort for deepseek_v41: {what}, should be int within [1,100] or [\"low\", \"high\", \"max\"]"
        ))
    };
    match effort {
        None | Some(Json::Null) => Ok(75),
        Some(Json::Num(raw)) => {
            if raw.bytes().any(|b| matches!(b, b'.' | b'e' | b'E')) {
                return Err(bad(raw.clone()));
            }
            match raw.parse::<i64>() {
                Ok(n) if (1..=100).contains(&n) => Ok(n as u32),
                _ => Err(bad(raw.clone())),
            }
        }
        Some(Json::Str(s)) => REASONING_EFFORT_MAPPINGS
            .iter()
            .find(|(k, _)| k == s)
            .map(|(_, v)| *v)
            .ok_or_else(|| bad(s.clone())),
        // a list or an object is unhashable in the reference: a TypeError there, an input error here
        Some(Json::Array(_)) | Some(Json::Object(_)) => Err(ChatError::input("reasoning_effort must be an integer or a string")),
        Some(other) => Err(bad(format!("{other:?}"))),
    }
}

/// `REASONING_EFFORT_TEMPLATE` for a budget.
pub fn effort_prefix(budget: u32) -> String {
    format!("Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n")
}

/// Convenience constructors for request plumbing.
pub fn int(n: u32) -> Json {
    Json::Num(n.to_string())
}
pub fn named(s: &str) -> Json {
    Json::Str(s.to_string())
}
