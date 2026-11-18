//! Host-side JSON-schema constrained decoding (V1 subset) for the OpenAI-compatible API.
//!
//! **Why this module exists.** `response_format: {"type":"json_schema", ...}` used to be *accepted
//! and then silently ignored* — the F6 failure class. A client that sends a schema and gets
//! free-form text back has no way to notice until it parses the answer; the failure looks like a
//! model quality problem, not a server contract violation. This module replaces "silently ignored"
//! with two loud behaviours:
//!
//! 1. [`compile_response_format`] validates the schema and returns `Err(msg)` for anything outside
//!    the V1 subset, **naming the offending keyword**. An unsupported schema can never degrade into
//!    "no constraint applied".
//! 2. [`SchemaMask`] is a byte-level JSON machine plus a token-level allowed-set so the sampler can
//!    mask every token that would leave the schema.
//!
//! **V1 subset:** `type` / `properties` / `required` / `additionalProperties` / `items` /
//! `minItems` / `maxItems` / `minLength` / `maxLength` / `minimum` / `maximum` /
//! `exclusiveMinimum` / `exclusiveMaximum` / `enum` (strings and integers), nesting depth <= 12.
//! Everything else — `pattern`, `format`, `oneOf`/`anyOf`/`allOf`/`not`, `$ref`, `const`,
//! `uniqueItems`, `$defs`, conditionals, unknown keywords — is a hard error.
//!
//! **`additionalProperties` default (deliberate deviation from JSON Schema).** When `properties` is
//! present, an absent `additionalProperties` is treated as **false**; a bare `{"type":"object"}` /
//! `{"type":"json_object"}` (no `properties`) allows any key. JSON Schema's own default is `true`,
//! but an accepted key the caller never declared is indistinguishable from a hallucination, so we
//! take the strict reading and document it here. `"additionalProperties": true` opts back in.
//!
//! **Byte machine semantics.** A document is: optional leading whitespace, one JSON value matching
//! the schema, trailing whitespace, end. The machine is an explicit continuation stack of immutable
//! frames; a *state* is an interned stack snapshot, so states are cheap to compare/deduplicate and
//! `allowed()` results are cacheable per state. `step()` advances by one emitted token; `allowed()`
//! walks the vocabulary trie from the state and prunes branches the moment the machine refuses a
//! byte (a full-vocab rescan per state is never needed).
//!
//! No regex, no new dependencies (std + serde_json), no panics on user input — every rejection is an
//! `Err`/`None`/zeroed bitset.
//!
//! Map of this directory (items keep their `crate::json_schema::` paths):
//! - `schema.rs`: the schema IR (`Ty`, `Schema`) and `compile_schema` with its validation
//! - `automata.rs`: the string and number sub-machines (`StrState`, `str_advance`, `NumState`)
//! - `machine.rs`: the continuation-stack frames and the `Machine` state, `canon_match`, `Roll`
//! - `vocab.rs`: the vocabulary byte trie
//! - `mask.rs`: `SchemaMask` / `VocabTable` and the public methods (`start_state`, `allowed`, `step`, ...)
//! - `walk.rs`: `SchemaMask` internals: the trie walk and the byte feed (`value_start`, `feed`)
//! - `tests.rs`: the unit tests
//! `mod.rs` keeps the entry points `compile_response_format` / `build_mask`, `vocab_words` and the constants.

use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

mod automata;
mod machine;
mod mask;
mod schema;
mod vocab;
mod walk;

use self::automata::*;
use self::machine::*;
pub use self::mask::*;
use self::schema::*;
use self::vocab::*;

/// Number of `u32` words needed for a `vocab_size`-token bitset (bit `t` = token `t` allowed).
pub fn vocab_words(n: usize) -> usize {
    n / 32 + if n % 32 == 0 { 0 } else { 1 }
}

/// Nesting cap. Beyond this the schema is refused rather than truncated (silent truncation would be
/// a constraint that is not the one the caller asked for).
const MAX_DEPTH: usize = 12;

const SUPPORTED_LIST: &str = "type/properties/required/enum/items/minItems/maxItems/minLength/maxLength/minimum/maximum/exclusiveMinimum/exclusiveMaximum/additionalProperties";

/// Every keyword V1 knows about. Anything else (including the explicitly-unsupported list in the
/// module docs) is refused by name.
const KNOWN_KEYWORDS: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "enum",
];

const PREFIX: &str = "response_format json_schema: ";

fn err_msg(msg: impl AsRef<str>) -> String {
    format!("{PREFIX}{}", msg.as_ref())
}

fn err_unsupported(kw: &str) -> String {
    err_msg(format!(
        "keyword '{kw}' is not supported in this release (V1 supports: {SUPPORTED_LIST})"
    ))
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Parse + validate an OpenAI `response_format` value (the whole `{"type": "json_schema", ...}`
/// object).
///
/// Returns `Ok(None)` when there is no schema to enforce (absent/`Null`/`{"type":"text"}`), and
/// `Err(msg)` with a LOUD, user-facing message naming the offending keyword for everything else —
/// an unsupported schema is refused, never silently ignored.
pub fn compile_response_format(rf: &serde_json::Value) -> Result<Option<SchemaMask>, String> {
    let obj = match rf {
        Value::Null => return Ok(None),
        Value::Object(o) => o,
        other => {
            return Err(err_msg(format!(
                "response_format must be an object (found {})",
                type_name(other)
            )))
        }
    };

    let ty = match obj.get("type") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => {
            return Err(err_msg(format!(
                "response_format 'type' must be a string (found {})",
                type_name(other)
            )))
        }
        None => return Err(err_msg("response_format is missing 'type'")),
    };

    match ty.as_str() {
        "text" => {
            for k in obj.keys() {
                if k != "type" {
                    return Err(err_msg(format!(
                        "response_format field '{k}' is not supported in this release"
                    )));
                }
            }
            Ok(None)
        }
        "json_object" => {
            for k in obj.keys() {
                if k != "type" {
                    return Err(err_msg(format!(
                        "response_format field '{k}' is not supported in this release"
                    )));
                }
            }
            // `{"type":"json_object"}` means "any JSON object": compile a real permissive object
            // schema, so arrays/scalars are rejected instead of any value being accepted.
            Ok(Some(build_mask(&serde_json::json!({"type": "object"}), 1, "type")?))
        }
        "json_schema" => {
            for k in obj.keys() {
                if k != "type" && k != "json_schema" && k != "schema" {
                    return Err(err_msg(format!(
                        "response_format field '{k}' is not supported in this release"
                    )));
                }
            }
            let inner = obj.get("json_schema");
            let (schema_v, name) = match inner {
                Some(Value::Object(io)) => {
                    for k in io.keys() {
                        if !matches!(k.as_str(), "name" | "description" | "strict" | "schema") {
                            return Err(err_msg(format!(
                                "json_schema field '{k}' is not supported in this release \
                                 (supported: name/description/strict/schema)"
                            )));
                        }
                    }
                    let name = match io.get("name") {
                        Some(Value::String(n)) => Some(n.clone()),
                        Some(other) => {
                            return Err(err_msg(format!(
                                "json_schema 'name' must be a string (found {})",
                                type_name(other)
                            )))
                        }
                        None => None,
                    };
                    (io.get("schema"), name)
                }
                Some(other) => {
                    return Err(err_msg(format!(
                        "'json_schema' must be an object (found {})",
                        type_name(other)
                    )))
                }
                None => (obj.get("schema"), None),
            };
            let Some(sv) = schema_v else {
                return Err(err_msg(
                    "keyword 'schema' is required: send \
                     {\"type\":\"json_schema\",\"json_schema\":{\"name\":...,\"schema\":{...}}}",
                ));
            };
            let what = name.map(|n| format!("json_schema '{n}'")).unwrap_or_else(|| "json_schema".to_string());
            let mask = build_mask(sv, 1, "schema")?;
            // Loud success line: a wrong-but-accepted schema must be visible in the server log.
            crate::deprintln!("[json_schema] enforcing {what}: {}", mask.summary());
            Ok(Some(mask))
        }
        other => Err(err_msg(format!(
            "response_format 'type' = '{other}' is not supported in this release \
             (V1 supports: text/json_object/json_schema)"
        ))),
    }
}

fn build_mask(schema: &Value, depth: usize, via: &str) -> Result<SchemaMask, String> {
    let mut table = vec![Schema::any()]; // index 0 = universal permissive schema
    // Validation errors MUST reach the caller: swallowing them here would turn an unsupported
    // schema into a silently-unenforced one — exactly the failure this module exists to remove.
    let root = compile_schema(schema, depth, via, &mut table)?;
    let mut m = SchemaMask {
        schemas: table,
        root,
        start: 0,
        interner: Mutex::new(Interner::default()),
        cache: Mutex::new(HashMap::new()),
        mask_cache: Mutex::new(HashMap::new()),
        vocab: Arc::new(Vocab::default()),
        vocab_size: 0,
        words: 0,
        defer_until: None,
    };
    let start = m.intern(Machine { stack: vec![Frame::DocEnd, Frame::Value { si: root }], ws: 0 });
    m.start = start;
    Ok(m)
}

// ===========================================================================================
// Tests
// ===========================================================================================

#[cfg(test)]
mod tests;
