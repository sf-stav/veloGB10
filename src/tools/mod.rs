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
//!
//! Map of this directory (items keep their `crate::tools::` paths via `pub use self::<sub>::*`):
//! - `rules.rs`: the opt-in rule enums (block end, value trim, unclosed block) and `Rules`
//! - `parse.rs`: `parse` / `parse_rules` / `parse_core`, `parse_with`, `block_spans`, `finalize`
//! - `shadow.rs`: the shadow comparison of the served reading against the other readings
//! - `blocks.rs`: where a `<tool_call>` block ends (block rules, `BlockIndex`)
//! - `hy3.rs`: the hy_v3 (`:opensource` markup) parser
//! - `values.rs`: where a `<parameter=...>` value ends (v0.7.3 rule, quote pairing, caches)
//! - `calls.rs`: block bodies to calls (XML form) and `coerce`
//! - `json.rs`: the JSON-body call forms, salvage and quote repair
//! - `mod.rs` also holds the ids, tag constants, `ParsedOutput`, `is_hy3`, `has_tool_call` and two test-only thread-local counters
//! - `cost.rs` (own directory when split): the cost model; `bound_tests.rs`, `v073_frozen.rs`, `default_identity.rs`, `tests.rs`: the test modules

use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;

use crate::tokenizer::{FunctionCall, ToolCall};

pub mod cost;

#[cfg(test)]
mod bound_tests;

/// The v0.7.3 parser, frozen, for the default-identity test (test builds only).
#[cfg(test)]
mod v073_frozen;
#[cfg(test)]
mod default_identity;
#[cfg(test)]
pub(crate) mod tests;

mod blocks;
mod calls;
mod hy3;
mod json;
mod parse;
mod rules;
mod shadow;
mod values;

use self::blocks::*;
pub(crate) use self::calls::*;
use self::hy3::*;
pub use self::json::*;
pub use self::parse::*;
pub use self::rules::*;
pub use self::shadow::*;
pub use self::values::*;

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

/// Is this text a hy_v3 call (its `:opensource` markup) rather than a Qwen one? Decided by the
/// FIRST block opener, the block's own structure: a Qwen call whose value merely contains the text
/// `<tool_call:opensource>` (an agent writing a file about hy_v3 markup, this repository's own
/// src/tools.rs) is a Qwen call. (Independent review M4 / audit X01: the dispatch used to be "the text
/// contains the hy_v3 tag anywhere", which handed such a call to the hy_v3 parser: 0 calls, the raw XML
/// returned as content, finish `stop`.)
fn is_hy3(text: &str) -> bool {
    let Some(h) = text.find(HY_CALL_OPEN) else { return false };
    let first_hy = text.find("<tool_calls:opensource>").map_or(h, |w| w.min(h));
    text.find(CALL_OPEN).map_or(true, |q| first_hy < q)
}

/// True if `s` contains the start of a tool call — used by the streaming path to decide whether it
/// must hold text back rather than forward it to the client as content. `<tool_call` is the shared
/// prefix of both families' tags.
pub fn has_tool_call(s: &str) -> bool {
    s.contains("<tool_call")
}

/// Test instrument: make the shadow comparison of this thread panic (the drill of
/// `a_panic_in_the_shadow_is_caught_counted_and_changes_nothing_served`).
#[cfg(test)]
thread_local! { static SHADOW_FAULT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

/// Test instrument: bytes the DEFAULT path's searches have scanned on this thread (the linearity
/// tests compare it between a text and a text four times its size: counted work, not a clock).
#[cfg(test)]
thread_local! { static SCAN_OPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
