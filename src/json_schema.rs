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

use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

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

// ===========================================================================================
// Schema IR + validation
// ===========================================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    /// No `type` and no constraint: any JSON value. Also the "additional property" child schema.
    Any,
    Object,
    Array,
    String,
    Integer,
    Number,
    Boolean,
    Null,
}

impl Ty {
    fn name(self) -> &'static str {
        match self {
            Ty::Any => "any",
            Ty::Object => "object",
            Ty::Array => "array",
            Ty::String => "string",
            Ty::Integer => "integer",
            Ty::Number => "number",
            Ty::Boolean => "boolean",
            Ty::Null => "null",
        }
    }
}

/// One compiled schema. Children are indices into the shared table (never `Box`), so machine frames
/// can reference schemas by `u32` and stay `Copy`-cheap inside the state snapshots.
#[derive(Clone, Debug)]
struct Schema {
    ty: Ty,
    props: Vec<(String, u32)>,
    /// Indices into `props`.
    required: Vec<u32>,
    /// `false` = keys outside `props` are rejected. See the module docs for the default.
    additional: bool,
    items: Option<u32>,
    min_items: Option<u64>,
    max_items: Option<u64>,
    min_len: Option<u64>,
    max_len: Option<u64>,
    minimum: Option<f64>,
    maximum: Option<f64>,
    excl_min: Option<f64>,
    excl_max: Option<f64>,
    enum_str: Vec<String>,
    enum_int: Vec<i64>,
    has_enum: bool,
    depth: usize,
}

impl Schema {
    fn any() -> Schema {
        Schema {
            ty: Ty::Any,
            props: Vec::new(),
            required: Vec::new(),
            additional: true,
            items: None,
            min_items: None,
            max_items: None,
            min_len: None,
            max_len: None,
            minimum: None,
            maximum: None,
            excl_min: None,
            excl_max: None,
            enum_str: Vec::new(),
            enum_int: Vec::new(),
            has_enum: false,
            depth: 0,
        }
    }
}

/// Is `kw` meaningful for `ty`? Anything else is refused by name (never ignored).
fn keyword_allowed_for(ty: Ty, kw: &str) -> bool {
    match ty {
        Ty::Object => matches!(kw, "type" | "properties" | "required" | "additionalProperties" | "enum"),
        Ty::Array => matches!(kw, "type" | "items" | "minItems" | "maxItems"),
        Ty::String => matches!(kw, "type" | "minLength" | "maxLength" | "enum"),
        Ty::Integer | Ty::Number => {
            matches!(kw, "type" | "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum" | "enum")
        }
        Ty::Boolean | Ty::Null => matches!(kw, "type"),
        // Untyped schemas may only constrain through `enum`; everything else needs an explicit type
        // so we never *infer* a constraint the caller did not state.
        Ty::Any => matches!(kw, "type" | "enum"),
    }
}

fn non_neg_u64(kw: &str, v: &Value) -> Result<u64, String> {
    v.as_u64()
        .ok_or_else(|| err_msg(format!("keyword '{kw}' must be a non-negative integer (found {})", v)))
}

fn num_f64(kw: &str, v: &Value) -> Result<f64, String> {
    v.as_f64()
        .ok_or_else(|| err_msg(format!("keyword '{kw}' must be a number (found {})", v)))
}

/// Compile one schema into `table`, returning its index. `via` names the keyword that pulled this
/// schema in (`"properties"` / `"items"`), used for depth errors.
fn compile_schema(v: &Value, depth: usize, via: &str, table: &mut Vec<Schema>) -> Result<u32, String> {
    if depth > MAX_DEPTH {
        return Err(err_msg(format!(
            "schema is nested deeper than the V1 limit of {MAX_DEPTH} levels (via keyword '{via}')"
        )));
    }
    let obj = v.as_object().ok_or_else(|| {
        err_msg(format!(
            "schema for keyword '{via}' must be a JSON object (found {})",
            type_name(v)
        ))
    })?;

    // Keyword gate FIRST: an unsupported/unknown keyword is refused before its value is interpreted,
    // so the error always names the thing the caller has to change.
    let mut declared: Option<String> = None;
    for k in obj.keys() {
        if !KNOWN_KEYWORDS.contains(&k.as_str()) {
            return Err(err_unsupported(k));
        }
        if k == "type" {
            declared = obj["type"].as_str().map(|s| s.to_string());
        }
    }

    let ty = match obj.get("type") {
        None => Ty::Any,
        Some(Value::String(s)) => match s.as_str() {
            "object" | "json_object" => Ty::Object,
            "array" => Ty::Array,
            "string" => Ty::String,
            "integer" => Ty::Integer,
            "number" => Ty::Number,
            "boolean" => Ty::Boolean,
            "null" => Ty::Null,
            other => {
                return Err(err_msg(format!(
                    "keyword 'type' has unsupported value '{other}' (V1 supports: object/array/string/integer/number/boolean/null/json_object)"
                )))
            }
        },
        Some(Value::Array(_)) => {
            return Err(err_msg(
                "keyword 'type' with a union type array is not supported in this release".to_string(),
            ))
        }
        Some(other) => {
            return Err(err_msg(format!("keyword 'type' must be a string (found {})", type_name(other))))
        }
    };

    let ty_label = match &declared {
        Some(s) => format!("\"type\":\"{s}\""),
        None => "no \"type\"".to_string(),
    };
    for k in obj.keys() {
        if !keyword_allowed_for(ty, k) {
            return Err(err_msg(format!(
                "keyword '{k}' is not supported with {ty_label} in this release (V1 supports: {SUPPORTED_LIST})"
            )));
        }
    }

    let mut s = Schema::any();
    s.ty = ty;
    s.depth = depth;
    // Strict default when a key set is declared; permissive for a bare object/json_object.
    s.additional = !matches!(ty, Ty::Object) || !obj.contains_key("properties");

    match obj.get("additionalProperties") {
        Some(Value::Bool(b)) => s.additional = *b,
        Some(other) => {
            return Err(err_msg(format!(
                "keyword 'additionalProperties' only supports a boolean in this release (found {}); \
                 a subschema form is not supported",
                type_name(other)
            )))
        }
        None => {}
    }

    if let Some(p) = obj.get("properties") {
        let pm = p
            .as_object()
            .ok_or_else(|| err_msg("keyword 'properties' must be an object".to_string()))?;
        for (name, sub) in pm {
            let csi = compile_schema(sub, depth + 1, "properties", table)?;
            s.props.push((name.clone(), csi));
        }
    }

    if let Some(r) = obj.get("required") {
        let arr = r
            .as_array()
            .ok_or_else(|| err_msg("keyword 'required' must be an array of property names".to_string()))?;
        for item in arr {
            let name = item.as_str().ok_or_else(|| {
                err_msg("keyword 'required' must contain only strings".to_string())
            })?;
            let idx = s
                .props
                .iter()
                .position(|(n, _)| n == name)
                .ok_or_else(|| {
                    err_msg(format!(
                        "keyword 'required' names '{name}' which is not declared in 'properties'"
                    ))
                })? as u32;
            if !s.required.contains(&idx) {
                s.required.push(idx);
            }
        }
    }

    if let Some(i) = obj.get("items") {
        let csi = compile_schema(i, depth + 1, "items", table)?;
        s.items = Some(csi);
    }
    if let Some(v) = obj.get("minItems") {
        s.min_items = Some(non_neg_u64("minItems", v)?);
    }
    if let Some(v) = obj.get("maxItems") {
        s.max_items = Some(non_neg_u64("maxItems", v)?);
    }
    if let Some((lo, hi)) = s.min_items.zip(s.max_items) {
        if lo > hi {
            return Err(err_msg(format!("keyword 'minItems' ({lo}) exceeds 'maxItems' ({hi})")));
        }
    }
    if let Some(v) = obj.get("minLength") {
        s.min_len = Some(non_neg_u64("minLength", v)?);
    }
    if let Some(v) = obj.get("maxLength") {
        s.max_len = Some(non_neg_u64("maxLength", v)?);
    }
    if let Some((lo, hi)) = s.min_len.zip(s.max_len) {
        if lo > hi {
            return Err(err_msg(format!("keyword 'minLength' ({lo}) exceeds 'maxLength' ({hi})")));
        }
    }
    if let Some(v) = obj.get("minimum") {
        s.minimum = Some(num_f64("minimum", v)?);
    }
    if let Some(v) = obj.get("maximum") {
        s.maximum = Some(num_f64("maximum", v)?);
    }
    if let Some(v) = obj.get("exclusiveMinimum") {
        s.excl_min = Some(num_f64("exclusiveMinimum", v)?);
    }
    if let Some(v) = obj.get("exclusiveMaximum") {
        s.excl_max = Some(num_f64("exclusiveMaximum", v)?);
    }

    if let Some(e) = obj.get("enum") {
        let arr = e.as_array().ok_or_else(|| {
            err_msg("keyword 'enum' must be an array of allowed values".to_string())
        })?;
        for item in arr {
            match item {
                Value::String(x) => s.enum_str.push(x.clone()),
                Value::Number(n) => match n.as_i64() {
                    Some(i) => s.enum_int.push(i),
                    None => {
                        return Err(err_msg(
                            "keyword 'enum': only string and integer values are supported in this release"
                                .to_string(),
                        ))
                    }
                },
                other => {
                    return Err(err_msg(format!(
                        "keyword 'enum': only string and integer values are supported in this release (found {})",
                        type_name(other)
                    )))
                }
            }
        }
        if s.enum_str.is_empty() && s.enum_int.is_empty() {
            return Err(err_msg("keyword 'enum' must not be empty".to_string()));
        }
        let compatible = match ty {
            Ty::Any => true,
            Ty::String => s.enum_int.is_empty(),
            Ty::Integer | Ty::Number => s.enum_str.is_empty(),
            _ => false,
        };
        if !compatible {
            return Err(err_msg(format!(
                "keyword 'enum' values do not match {ty_label} (V1 supports string and integer enum values only)"
            )));
        }
        // Only values the schema's other keywords also accept are candidates: an enum value
        // outside minLength/maxLength or the numeric bounds would let decoding start a prefix
        // with no valid completion.
        let (min_len, max_len) = (s.min_len, s.max_len);
        s.enum_str.retain(|e| {
            let n = e.chars().count() as u64;
            min_len.map_or(true, |m| n >= m) && max_len.map_or(true, |m| n <= m)
        });
        let in_bounds = |v: f64| {
            s.minimum.map_or(true, |m| v >= m) && s.excl_min.map_or(true, |m| v > m)
                && s.maximum.map_or(true, |m| v <= m) && s.excl_max.map_or(true, |m| v < m)
        };
        let ints: Vec<i64> = s.enum_int.iter().copied().filter(|&e| in_bounds(e as f64)).collect();
        s.enum_int = ints;
        if s.enum_str.is_empty() && s.enum_int.is_empty() {
            return Err(err_msg(
                "keyword 'enum': no value satisfies the schema's other constraints".to_string(),
            ));
        }
        s.has_enum = true;
    }

    let idx = table.len() as u32;
    table.push(s);
    Ok(idx)
}

// ===========================================================================================
// Byte machine: string / number / literal sub-states
// ===========================================================================================

/// Escape-sequence position inside a JSON string.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Esc {
    None,
    /// Just consumed `\`.
    Bslash,
    /// Inside `\uXXXX`, `n` hex digits collected in `acc`.
    Hex { n: u8, acc: u32 },
    /// High surrogate seen, waiting for `\`.
    WantLowBslash { hi: u32 },
    /// High surrogate seen, waiting for `u`.
    WantLowU { hi: u32 },
    /// Inside the low half `\uYYYY` of a surrogate pair.
    LowHex { n: u8, acc: u32, hi: u32 },
}

/// Incremental JSON-string state. `dec` holds the *decoded* bytes and is only kept when a prefix
/// constraint (enum values, property names) needs it — that keeps the buffers bounded by the
/// longest candidate instead of by the emitted document.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct StrState {
    esc: Esc,
    dec: Vec<u8>,
    /// The raw bytes consumed inside the string (kept with `dec`): enum values and declared keys
    /// are matched in their canonical spelling (see [`canon_match`]).
    raw: Vec<u8>,
    cp: u32,
    /// Trailing bytes of a not-yet-complete UTF-8 sequence (a token can split a code point).
    pend: Vec<u8>,
    /// Skip this string's minLength / maxLength checks. Only set on the machine used to build a
    /// length-free base mask (see [`SchemaMask::allowed_len_bounded`]); never on a live state.
    free_len: bool,
}

impl StrState {
    fn new() -> StrState {
        StrState { esc: Esc::None, dec: Vec::new(), raw: Vec::new(), cp: 0, pend: Vec::new(), free_len: false }
    }
    fn is_closed_interior(&self) -> bool {
        self.esc == Esc::None && self.pend.is_empty()
    }
}

enum Step {
    Open,
    Closed,
    Bad,
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn push_cp(st: &mut StrState, cp: u32, keep: bool) -> bool {
    let Some(c) = char::from_u32(cp) else { return false };
    let mut buf = [0u8; 4];
    let enc = c.encode_utf8(&mut buf);
    st.cp += 1;
    if keep {
        st.dec.extend_from_slice(enc.as_bytes());
    }
    true
}

/// Append one raw (non-escape) content byte, validating UTF-8 across token boundaries.
fn push_raw(st: &mut StrState, b: u8, keep: bool) -> bool {
    if b < 0x80 {
        st.cp += 1;
        if keep {
            st.dec.push(b);
        }
        return true;
    }
    st.pend.push(b);
    // The borrow of `pend` ends inside `map`, so the arms may mutate `st`.
    let decoded = std::str::from_utf8(&st.pend).map(|s| (s.chars().count() as u32, s.as_bytes().to_vec()));
    match decoded {
        Ok((n, bytes)) => {
            st.pend.clear();
            st.cp += n;
            if keep {
                st.dec.extend_from_slice(&bytes);
            }
            true
        }
        Err(e) => e.error_len().is_none() && st.pend.len() < 4,
    }
}

/// Advance one byte inside a string. `keep` mirrors the caller's prefix-constraint need.
fn str_advance(st: &mut StrState, b: u8, keep: bool) -> Step {
    if keep && !(st.esc == Esc::None && b == b'"') {
        st.raw.push(b);
    }
    match st.esc {
        // An ASCII byte (including `"` and `\`) while a multi-byte UTF-8 sequence is still
        // incomplete would make the string invalid UTF-8.
        Esc::None if !st.pend.is_empty() && b < 0x80 => Step::Bad,
        Esc::None => match b {
            b'"' => Step::Closed,
            b'\\' => {
                st.esc = Esc::Bslash;
                Step::Open
            }
            // Raw control characters must be escaped in JSON.
            0x00..=0x1F => Step::Bad,
            _ => {
                if push_raw(st, b, keep) {
                    Step::Open
                } else {
                    Step::Bad
                }
            }
        },
        Esc::Bslash => {
            let cp = match b {
                b'"' => 0x22,
                b'\\' => 0x5C,
                b'/' => 0x2F,
                b'b' => 0x08,
                b'f' => 0x0C,
                b'n' => 0x0A,
                b'r' => 0x0D,
                b't' => 0x09,
                b'u' => {
                    st.esc = Esc::Hex { n: 0, acc: 0 };
                    return Step::Open;
                }
                _ => return Step::Bad,
            };
            st.esc = Esc::None;
            if push_cp(st, cp, keep) {
                Step::Open
            } else {
                Step::Bad
            }
        }
        Esc::Hex { n, acc } => {
            let Some(d) = hex_val(b) else { return Step::Bad };
            let acc = acc * 16 + d as u32;
            if n + 1 < 4 {
                // `\uDC`..`\uDF` can only become a lone low surrogate: refuse it now, while
                // another byte can still follow, instead of at the fourth digit
                if n + 1 == 2 && (0xDC..=0xDF).contains(&acc) {
                    return Step::Bad;
                }
                st.esc = Esc::Hex { n: n + 1, acc };
                return Step::Open;
            }
            st.esc = Esc::None;
            if (0xD800..=0xDBFF).contains(&acc) {
                st.esc = Esc::WantLowBslash { hi: acc };
                return Step::Open;
            }
            if (0xDC00..=0xDFFF).contains(&acc) {
                // Lone low surrogate: serde_json rejects it too, so we must not emit it.
                return Step::Bad;
            }
            if push_cp(st, acc, keep) {
                Step::Open
            } else {
                Step::Bad
            }
        }
        Esc::WantLowBslash { hi } => {
            if b != b'\\' {
                return Step::Bad;
            }
            st.esc = Esc::WantLowU { hi };
            Step::Open
        }
        Esc::WantLowU { hi } => {
            if b != b'u' {
                return Step::Bad;
            }
            st.esc = Esc::LowHex { n: 0, acc: 0, hi };
            Step::Open
        }
        Esc::LowHex { n, acc, hi } => {
            let Some(d) = hex_val(b) else { return Step::Bad };
            let acc = acc * 16 + d as u32;
            if n + 1 < 4 {
                // the second half of a pair must be `\uDC00`..`\uDFFF`: refuse a prefix that
                // cannot become one
                if (n + 1 == 1 && acc != 0xD) || (n + 1 == 2 && !(0xDC..=0xDF).contains(&acc)) {
                    return Step::Bad;
                }
                st.esc = Esc::LowHex { n: n + 1, acc, hi };
                return Step::Open;
            }
            st.esc = Esc::None;
            if !(0xDC00..=0xDFFF).contains(&acc) {
                return Step::Bad;
            }
            let cp = 0x10000 + ((hi - 0xD800) << 10) + (acc - 0xDC00);
            if push_cp(st, cp, keep) {
                Step::Open
            } else {
                Step::Bad
            }
        }
    }
}

/// JSON number grammar position.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum NumState {
    /// Saw `-`, a digit must follow.
    NegStart,
    Zero,
    Int,
    FracDot,
    Frac,
    ExpChar,
    ExpSign,
    Exp,
}

impl NumState {
    fn terminal(self) -> bool {
        matches!(self, NumState::Zero | NumState::Int | NumState::Frac | NumState::Exp)
    }
}

fn num_start(b: u8) -> Option<(NumState, Vec<u8>)> {
    match b {
        b'-' => Some((NumState::NegStart, vec![b'-'])),
        b'0' => Some((NumState::Zero, vec![b'0'])),
        b'1'..=b'9' => Some((NumState::Int, vec![b])),
        _ => None,
    }
}

fn num_advance(st: NumState, b: u8) -> Option<NumState> {
    use NumState::*;
    match (st, b) {
        (NegStart, b'0') => Some(Zero),
        (NegStart, b'1'..=b'9') => Some(Int),
        (Zero, b'.') => Some(FracDot),
        (Zero, b'e' | b'E') => Some(ExpChar),
        (Int, b'0'..=b'9') => Some(Int),
        (Int, b'.') => Some(FracDot),
        (Int, b'e' | b'E') => Some(ExpChar),
        (FracDot, b'0'..=b'9') => Some(Frac),
        (Frac, b'0'..=b'9') => Some(Frac),
        (Frac, b'e' | b'E') => Some(ExpChar),
        (ExpChar, b'0'..=b'9') => Some(Exp),
        (ExpChar, b'+' | b'-') => Some(ExpSign),
        (ExpSign, b'0'..=b'9') => Some(Exp),
        (Exp, b'0'..=b'9') => Some(Exp),
        _ => None,
    }
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

// ===========================================================================================
// Byte machine: frames + states
// ===========================================================================================

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum ObjPhase {
    /// Whitespace, then `}` or the opening quote of a key.
    KeyOrEnd,
    /// Directly after a `,`: a key must follow (`}` would be a trailing comma).
    KeyReq,
    /// Inside a key string. Parsed *inside* the object frame so phase transitions never need to
    /// reach across stack frames.
    Key(StrState),
    /// Key closed; expect whitespace then `:`. `child` is the schema for the member value.
    Colon { child: u32 },
    /// The member value has been parsed; expect whitespace, `,` or `}`.
    AfterValue,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum ArrPhase {
    /// Whitespace, then `]` or the start of an item.
    ItemOrEnd,
    /// Directly after a `,`: an item must follow (`]` would be a trailing comma).
    ItemReq,
    /// An item has been parsed; expect whitespace, `,` or `]`.
    AfterItem,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Frame {
    /// Top-level frame: after the value only whitespace (then end) may follow.
    DocEnd,
    /// The byte stream left the schema / the JSON grammar. Absorbs everything after it.
    Dead,
    /// A value matching schema `si` must start here.
    Value { si: u32 },
    Obj { si: u32, phase: ObjPhase, seen: Vec<u32>, extra: Vec<String> },
    Arr { si: u32, n: u64, phase: ArrPhase },
    Str { si: u32, st: StrState },
    /// Remaining bytes of `true` / `false` / `null`.
    Lit { rest: &'static [u8] },
    Num { si: u32, buf: Vec<u8>, st: NumState },
}

/// An interned machine state: a stack of continuation frames.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
struct Machine {
    stack: Vec<Frame>,
    /// Consecutive STRUCTURAL whitespace bytes just consumed (whitespace between tokens of the
    /// document, never string content). Capped at [`MAX_WS_RUN`]: without a cap, whitespace is
    /// legal in every structural position, so a model that does not want to start the document
    /// can emit blank space until max_tokens and the reply never parses.
    ws: u8,
}

/// Longest run of structural whitespace the machine accepts (enough for pretty-printing a deeply
/// nested document: a newline plus indentation).
const MAX_WS_RUN: u8 = 24;

impl Machine {
    fn is_dead(&self) -> bool {
        self.stack.iter().any(|f| matches!(f, Frame::Dead))
    }
    /// Exactly the completed-document shape: only `DocEnd` left (whitespace is absorbed in place).
    fn at_doc_end(&self) -> bool {
        self.stack.len() == 1 && matches!(self.stack[0], Frame::DocEnd)
    }
    /// True while the next byte is string CONTENT (a value string or an object key).
    fn in_string(&self) -> bool {
        matches!(self.stack.last(), Some(Frame::Str { .. }) | Some(Frame::Obj { phase: ObjPhase::Key(_), .. }))
    }
}

/// The canonical JSON encoding (serde_json's: `"` `\\` and the short control escapes, other
/// control characters as lowercase `\u00XX`, everything else as raw UTF-8) of string `v`,
/// without the quotes, compared against `raw` (the bytes emitted so far inside the string).
/// Returns (`raw` is a prefix of the encoding, `raw` is the whole encoding).
///
/// Enum values and declared property names are matched in this ONE canonical spelling. Matching
/// the decoded value instead admits every escaped spelling (`"\u0074"` for `t`), and a model
/// under the mask then wanders into long escape sequences (or into escapes and partial UTF-8
/// sequences that no candidate completes) instead of writing the value.
fn canon_match(v: &str, raw: &[u8]) -> (bool, bool) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut i = 0usize;
    let mut buf = [0u8; 6];
    for c in v.chars() {
        let enc: &[u8] = match c {
            '"' => b"\\\"",
            '\\' => b"\\\\",
            '\u{08}' => b"\\b",
            '\u{0C}' => b"\\f",
            '\n' => b"\\n",
            '\r' => b"\\r",
            '\t' => b"\\t",
            c if (c as u32) < 0x20 => {
                buf = [b'\\', b'u', b'0', b'0', HEX[(c as usize) >> 4], HEX[(c as usize) & 15]];
                &buf[..]
            }
            c => c.encode_utf8(&mut buf).as_bytes(),
        };
        for &e in enc {
            if i == raw.len() {
                return (true, false);
            }
            if raw[i] != e {
                return (false, false);
            }
            i += 1;
        }
    }
    // the encoding is exhausted: `raw` matches only if it is exhausted too
    (i == raw.len(), i == raw.len())
}

/// Undo log for a single byte, used by the trie walk to roll a shared machine back instead of
/// cloning it per node. Frames are only ever mutated *after* being popped, so logging the frames
/// that were popped (first occurrence per index) plus the deepest index touched is a complete
/// record of the change.
struct Roll {
    orig_len: usize,
    frames: Vec<(usize, Frame)>,
}

impl Roll {
    fn new(orig_len: usize) -> Roll {
        Roll { orig_len, frames: Vec::new() }
    }
    fn note_pop(&mut self, idx: usize, f: &Frame) {
        if idx >= self.orig_len {
            // Pushed earlier during this same byte — truncation removes it.
            return;
        }
        if self.frames.iter().any(|(i, _)| *i == idx) {
            return;
        }
        self.frames.push((idx, f.clone()));
    }
    fn restore(&self, m: &mut Machine) {
        if self.frames.is_empty() {
            return;
        }
        let base = self.frames.iter().map(|(i, _)| *i).min().unwrap_or(0);
        m.stack.truncate(base);
        let mut sorted: Vec<&(usize, Frame)> = self.frames.iter().collect();
        sorted.sort_by_key(|(i, _)| *i);
        for (_, f) in sorted {
            m.stack.push(f.clone());
        }
    }
}

// ===========================================================================================
// Vocabulary trie
// ===========================================================================================

#[derive(Default, Debug)]
struct TrieNode {
    children: Vec<(u8, usize)>,
    token: Option<u32>,
}

#[derive(Default, Debug)]
struct Vocab {
    nodes: Vec<TrieNode>,
    by_id: HashMap<u32, Vec<u8>>,
    /// Longest piece in bytes: an upper bound on the characters one token adds to a string.
    max_piece: usize,
    /// Per token id, fed from the interior of a value string (no pending escape or UTF-8
    /// sequence): the characters it adds before the string closes or the piece ends, with
    /// [`STR_RUN_CLOSES`] set when it closes the string. [`STR_RUN_BAD`] when the bytes are not
    /// valid string content (such a token is never in a base mask, so it is never consulted).
    str_run: Vec<u16>,
}

const STR_RUN_CLOSES: u16 = 0x8000;
const STR_RUN_BAD: u16 = 0xFFFF;

/// 1 while a started escape or UTF-8 sequence has not produced its character yet.
fn pending_char(st: &StrState) -> u64 {
    u64::from(!st.is_closed_interior())
}

/// [`Vocab::str_run`] entry for one piece.
fn str_run_of(bytes: &[u8]) -> u16 {
    let mut st = StrState::new();
    for &b in bytes {
        match str_advance(&mut st, b, false) {
            Step::Open => {}
            Step::Closed => return (st.cp.min(0x7FFE) as u16) | STR_RUN_CLOSES,
            Step::Bad => return STR_RUN_BAD,
        }
    }
    ((st.cp as u64 + pending_char(&st)).min(0x7FFE)) as u16
}

impl Vocab {
    /// Build the byte trie. Empty pieces are skipped: a token with no bytes can never advance the
    /// machine, so it is never allowed (see `step`).
    fn build(pieces: Vec<(u32, Vec<u8>)>) -> (Vocab, usize) {
        let mut v = Vocab::default();
        v.nodes.push(TrieNode::default());
        let mut vocab_size = 0usize;
        for (id, bytes) in pieces {
            vocab_size = vocab_size.max(id as usize + 1);
            if bytes.is_empty() {
                continue;
            }
            v.max_piece = v.max_piece.max(bytes.len());
            v.by_id.insert(id, bytes.clone());
            let mut cur = 0usize;
            for &b in &bytes {
                let nxt = match v.nodes[cur].children.iter().find(|(c, _)| *c == b) {
                    Some(&(_, n)) => n,
                    None => {
                        v.nodes.push(TrieNode::default());
                        let n = v.nodes.len() - 1;
                        v.nodes[cur].children.push((b, n));
                        n
                    }
                };
                cur = nxt;
            }
            v.nodes[cur].token = Some(id); // duplicate piece: last id wins
        }
        v.str_run = vec![STR_RUN_BAD; vocab_size];
        for (&id, bytes) in &v.by_id {
            v.str_run[id as usize] = str_run_of(bytes);
        }
        (v, vocab_size)
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A poisoned lock means a previous panic; the bitsets/states are still structurally valid, so
    // keep serving instead of panicking a second time.
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Default, Debug)]
struct Interner {
    states: Vec<Arc<Machine>>,
    index: HashMap<Machine, u32>,
}

// ===========================================================================================
// Public API
// ===========================================================================================

/// A compiled schema: byte-level machine + token-level allowed sets, lazily built.
#[derive(Debug)]
pub struct SchemaMask {
    schemas: Vec<Schema>,
    root: u32,
    /// Index 0 is always the universal permissive schema (`Ty::Any`), used for additional
    /// properties and for `{"type":"array"}` without `items`.
    start: u32,
    interner: Mutex<Interner>,
    /// `state -> allowed bitset`. Computed on demand; states are interned so identical machine
    /// snapshots share one entry.
    cache: Mutex<HashMap<u32, Arc<Vec<u32>>>>,
    /// `mask key -> allowed bitset` (see [`SchemaMask::mask_key`]): states that differ only in a
    /// string's character count, where the count cannot change the allowed set, share one entry.
    /// Without it every token inside a free-text string is a new state and a full trie walk.
    mask_cache: Mutex<HashMap<Machine, Arc<Vec<u32>>>>,
    /// Shared, immutable byte trie of the model vocabulary (see [`VocabTable`]): built once per
    /// process and handed to every request's mask, instead of rebuilding a 248K-entry trie per
    /// request.
    vocab: Arc<Vocab>,
    vocab_size: usize,
    words: usize,
    /// When `Some(tok)`, the mask is NOT applied until the lane has emitted `tok` (the model's
    /// think-close token): a request whose prompt opens a reasoning block reasons freely, and
    /// only the answer after the block is constrained. `None` = constrained from the first token.
    defer_until: Option<u32>,
}

/// A model vocabulary prepared for constrained decoding: token id -> raw output bytes plus the
/// byte trie `allowed()` walks. Immutable after `build`; share it with `Arc` across requests and
/// attach it with [`SchemaMask::set_vocab_table`].
#[derive(Debug, Clone)]
pub struct VocabTable {
    vocab: Arc<Vocab>,
    size: usize,
}

impl VocabTable {
    /// `pieces` = (token id, the exact bytes that token adds to the output text). Special /
    /// control tokens must be left out (or given empty bytes): they are never allowed.
    pub fn build(pieces: Vec<(u32, Vec<u8>)>) -> Self {
        let (v, size) = Vocab::build(pieces);
        VocabTable { vocab: Arc::new(v), size }
    }

    /// Number of token ids covered (max id + 1).
    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl SchemaMask {
    /// The start state id.
    pub fn start_state(&self) -> u32 {
        self.start
    }

    /// Allowed-token bitset for `state`: `vocab_words(vocab_size)` little-endian `u32` words,
    /// bit `t` set = token `t` is allowed. Cached; safe to call from several threads.
    ///
    /// Returns an all-zero bitset if `set_vocab` was never called (`fail closed`: nothing is
    /// emitted rather than unconstrained text).
    pub fn allowed(&self, state: u32) -> Arc<Vec<u32>> {
        if let Some(hit) = lock(&self.cache).get(&state).cloned() {
            return hit;
        }
        let arc = match self.machine(state) {
            Some(m) => self.allowed_for(&m),
            None => Arc::new(vec![0u32; self.words]),
        };
        lock(&self.cache).insert(state, arc.clone());
        arc
    }

    /// Allowed set of a machine snapshot, shared through `mask_cache` by [`SchemaMask::mask_key`].
    fn allowed_for(&self, m: &Machine) -> Arc<Vec<u32>> {
        let key = self.mask_key(m);
        if let Some(hit) = lock(&self.mask_cache).get(&key).cloned() {
            return hit;
        }
        let a = Arc::new(match self.allowed_len_bounded(m) {
            Some(bits) => bits,
            None => self.compute_allowed_machine(m),
        });
        lock(&self.mask_cache).insert(key, a.clone());
        a
    }

    /// Inside a value string whose minLength / maxLength can refuse the next token, derive the
    /// allowed set from the length-free set of the same position instead of walking the trie.
    ///
    /// The string's character count only gates acceptance: it never changes what the bytes
    /// parse to. It only grows while a token is fed, and a token is refused by it exactly when
    /// the count after its last in-string byte exceeds maxLength, or the string closes below
    /// minLength. From an interior state (no pending escape or UTF-8 sequence) the characters a
    /// token adds depend only on its bytes ([`Vocab::str_run`]), so
    ///
    /// `allowed = base ∩ { t : added(t) <= room  and  (closes(t) ⇒ added(t) >= to_min) }`
    ///
    /// where `base` ignores this string's bounds. `base` is one cached mask per position, so the
    /// last characters before maxLength (and the first before minLength) cost a bitset pass each
    /// instead of a full trie walk. `None` when the shortcut does not apply.
    fn allowed_len_bounded(&self, m: &Machine) -> Option<Vec<u32>> {
        let Some(Frame::Str { si, st }) = m.stack.last() else { return None };
        let s = &self.schemas[*si as usize];
        if !s.enum_str.is_empty() || st.free_len || !st.is_closed_interior() || self.vocab.str_run.is_empty() {
            return None;
        }
        if s.min_len.is_none() && s.max_len.is_none() {
            return None;
        }
        let cp = st.cp as u64;
        let to_min = s.min_len.map_or(0, |lo| lo.saturating_sub(cp));
        let room = s.max_len.map_or(u64::MAX, |hi| hi.saturating_sub(cp));
        let mut free = m.clone();
        if let Some(Frame::Str { st, .. }) = free.stack.last_mut() {
            st.free_len = true;
            st.cp = 0;
        }
        let base = self.allowed_for(&free);
        let run = &self.vocab.str_run;
        let mut bits = vec![0u32; base.len()];
        for (w, &word) in base.iter().enumerate() {
            let mut rest = word;
            while rest != 0 {
                let b = rest.trailing_zeros();
                rest &= rest - 1;
                let t = (w << 5) | b as usize;
                let r = run.get(t).copied().unwrap_or(STR_RUN_BAD);
                if r == STR_RUN_BAD {
                    continue;
                }
                let added = (r & !STR_RUN_CLOSES) as u64;
                if added <= room && (r & STR_RUN_CLOSES == 0 || added >= to_min) {
                    bits[w] |= 1u32 << b;
                }
            }
        }
        Some(bits)
    }

    /// The machine with every value string's character count replaced by what the next token
    /// can observe of it: the distance to `minLength` and the room left below `maxLength`, each
    /// capped at one more than the longest piece (a token adds at most that many characters).
    /// Two states with the same key allow exactly the same tokens. Strings whose content is
    /// kept (enum values) are left exact.
    fn mask_key(&self, m: &Machine) -> Machine {
        let cap = self.vocab.max_piece as u64 + 1;
        let mut k = m.clone();
        for f in k.stack.iter_mut() {
            if let Frame::Str { si, st } = f {
                let s = &self.schemas[*si as usize];
                if !s.enum_str.is_empty() || st.free_len {
                    continue;
                }
                let cp = st.cp as u64;
                let to_min = s.min_len.map_or(0, |lo| lo.saturating_sub(cp).min(cap));
                let room = s.max_len.map_or(cap, |hi| hi.saturating_sub(cp).min(cap));
                st.cp = ((to_min << 16) | room) as u32;
            }
        }
        k
    }

    /// Advance the machine by ONE EMITTED TOKEN. `None` when the token is unknown, empty, or its
    /// bytes are refused in `state`.
    pub fn step(&self, state: u32, tok: u32) -> Option<u32> {
        let bytes = self.vocab.by_id.get(&tok)?;
        if bytes.is_empty() {
            return None;
        }
        let mut m = (*self.machine(state)?).clone();
        if m.is_dead() {
            return None;
        }
        for &b in bytes {
            if !self.feed(&mut m, b, None) {
                return None;
            }
        }
        Some(self.intern(m))
    }

    /// True when the JSON value is finished and only trailing whitespace (or end) may follow.
    /// A pending number is finalized here, so `42` counts as finished even though the frame is
    /// still open until the next byte.
    pub fn is_complete(&self, state: u32) -> bool {
        let Some(stored) = self.machine(state) else { return false };
        if stored.is_dead() {
            return false;
        }
        let mut m = (*stored).clone();
        if !self.finalize_pending_number(&mut m) {
            return false;
        }
        m.at_doc_end()
    }

    /// True when a state is DEAD (used to reject after a broken draft chain).
    pub fn is_dead(&self, state: u32) -> bool {
        self.machine(state).map(|m| m.is_dead()).unwrap_or(true)
    }

    /// Set the token-id → piece-bytes table (built from the model tokenizer). MUST be called once
    /// before `allowed()`; also fixes `vocab_size`/`words`. Cached bitsets are dropped because they
    /// were computed against the previous vocabulary.
    pub fn set_vocab(&mut self, pieces: Vec<(u32, Vec<u8>)>) {
        self.set_vocab_table(&VocabTable::build(pieces));
    }

    /// `set_vocab` with a prebuilt, shared table (no per-request trie build).
    pub fn set_vocab_table(&mut self, table: &VocabTable) {
        self.vocab = table.vocab.clone();
        self.vocab_size = table.size;
        self.words = vocab_words(table.size);
        lock(&self.cache).clear();
        lock(&self.mask_cache).clear();
    }

    /// Defer the constraint until the lane emits `tok` (see the `defer_until` field).
    pub fn set_defer_until(&mut self, tok: Option<u32>) {
        self.defer_until = tok;
    }

    /// The token that arms the constraint, if the constraint is deferred.
    pub fn defer_until(&self) -> Option<u32> {
        self.defer_until
    }

    /// A bitset with every token disallowed EXCEPT the model's end-of-generation tokens: when the
    /// document is complete, generation must stop rather than run on. End tokens are deliberately
    /// never produced by `allowed()`.
    pub fn end_mask(&self, stop_ids: &[u32]) -> Arc<Vec<u32>> {
        let words = if self.words > 0 {
            self.words
        } else {
            // Vocab not attached yet: size the bitset to the stop ids so the caller still gets a
            // usable mask instead of an empty one.
            vocab_words(stop_ids.iter().map(|&s| s as usize + 1).max().unwrap_or(0))
        };
        let mut bits = vec![0u32; words];
        for &s in stop_ids {
            let t = s as usize;
            if t < words * 32 {
                bits[t >> 5] |= 1u32 << (t & 31);
            }
        }
        Arc::new(bits)
    }

    /// Human-readable summary for logs.
    pub fn summary(&self) -> String {
        let root = &self.schemas[self.root as usize];
        let max_depth = self.schemas.iter().map(|s| s.depth).max().unwrap_or(0);
        let enum_values: usize = self.schemas.iter().map(|s| s.enum_str.len() + s.enum_int.len()).sum();
        let states = lock(&self.interner).states.len();
        let cached = lock(&self.cache).len();
        let vocab = if self.vocab_size == 0 {
            "unset".to_string()
        } else {
            format!("{} tokens/{} words", self.vocab_size, self.words)
        };
        format!(
            "json_schema: root={} properties={} required={} additionalProperties={} items={} enum_values={} schemas={} max_depth={} states={} cached_masks={} vocab={}",
            root.ty.name(),
            root.props.len(),
            root.required.len(),
            root.additional,
            root.items.is_some(),
            enum_values,
            self.schemas.len(),
            max_depth,
            states,
            cached,
            vocab,
        )
    }

    // -----------------------------------------------------------------------------------
    // internals
    // -----------------------------------------------------------------------------------

    fn machine(&self, state: u32) -> Option<Arc<Machine>> {
        lock(&self.interner).states.get(state as usize).cloned()
    }

    fn intern(&self, m: Machine) -> u32 {
        let mut it = lock(&self.interner);
        if let Some(&id) = it.index.get(&m) {
            return id;
        }
        let id = it.states.len() as u32;
        it.index.insert(m.clone(), id);
        it.states.push(Arc::new(m));
        id
    }

    fn compute_allowed(&self, state: u32) -> Vec<u32> {
        let mut bits = vec![0u32; self.words];
        if self.words == 0 || self.vocab.nodes.is_empty() {
            return bits;
        }
        let Some(stored) = self.machine(state) else { return bits };
        self.walk_into(&stored, &mut bits);
        bits
    }

    /// [`SchemaMask::compute_allowed`] for a machine snapshot that need not be interned.
    fn compute_allowed_machine(&self, m: &Machine) -> Vec<u32> {
        let mut bits = vec![0u32; self.words];
        if self.words == 0 || self.vocab.nodes.is_empty() {
            return bits;
        }
        self.walk_into(m, &mut bits);
        bits
    }

    fn walk_into(&self, stored: &Machine, bits: &mut Vec<u32>) {
        if stored.is_dead() {
            return;
        }
        let mut m = stored.clone();
        self.trie_walk(&mut m, 0, bits);
    }

    /// DFS the vocabulary trie from `node`, pruning the moment the machine refuses a byte.
    /// One machine is reused and rolled back per byte (see [`Roll`]) — no per-node clones and no
    /// full-vocab rescan.
    fn trie_walk(&self, m: &mut Machine, node: usize, out: &mut Vec<u32>) {
        // Iterate the shared trie by reference: cloning the child list at every node cost one
        // allocation per visited node (~1M on a 248K vocabulary inside a free string).
        let vocab = &*self.vocab;
        for &(b, child) in &vocab.nodes[node].children {
            let mut roll = Roll::new(m.stack.len());
            let ws0 = m.ws; // not covered by the Roll undo log (it records frames only)
            let alive = self.feed(m, b, Some(&mut roll));
            if alive {
                if let Some(tok) = vocab.nodes[child].token {
                    let t = tok as usize;
                    if t < self.vocab_size {
                        out[t >> 5] |= 1u32 << (t & 31);
                    }
                }
                self.trie_walk(m, child, out);
            }
            roll.restore(m);
            m.ws = ws0;
        }
    }

    /// Push the frame(s) that start a JSON value matching `si` in response to byte `b`.
    fn value_start(&self, m: &mut Machine, si: u32, b: u8) -> bool {
        let s = &self.schemas[si as usize];
        let (has_enum, any_str, any_int) = (s.has_enum, !s.enum_str.is_empty(), !s.enum_int.is_empty());
        let push_str = |m: &mut Machine, si: u32| {
            m.stack.push(Frame::Str { si, st: StrState::new() });
        };
        let push_obj = |m: &mut Machine, si: u32| {
            m.stack.push(Frame::Obj {
                si,
                phase: ObjPhase::KeyOrEnd,
                seen: Vec::new(),
                extra: Vec::new(),
            });
        };
        let push_arr = |m: &mut Machine, si: u32| {
            m.stack.push(Frame::Arr { si, n: 0, phase: ArrPhase::ItemOrEnd });
        };
        match s.ty {
            Ty::Object => {
                if b != b'{' {
                    return false;
                }
                push_obj(m, si);
                true
            }
            Ty::Array => {
                if b != b'[' {
                    return false;
                }
                push_arr(m, si);
                true
            }
            Ty::String => {
                if b != b'"' {
                    return false;
                }
                push_str(m, si);
                true
            }
            Ty::Integer | Ty::Number => match num_start(b) {
                Some((st, buf)) => {
                    if !self.num_prefix_viable(si, &buf, st) {
                        return false;
                    }
                    m.stack.push(Frame::Num { si, buf, st });
                    true
                }
                None => false,
            },
            Ty::Boolean => match b {
                b't' => {
                    m.stack.push(Frame::Lit { rest: b"rue" });
                    true
                }
                b'f' => {
                    m.stack.push(Frame::Lit { rest: b"alse" });
                    true
                }
                _ => false,
            },
            Ty::Null => {
                if b != b'n' {
                    return false;
                }
                m.stack.push(Frame::Lit { rest: b"ull" });
                true
            }
            Ty::Any => {
                // Untyped: accept whatever the schema constrains. With an enum, only the value
                // kinds the enum actually contains may start.
                if has_enum && !any_str && b == b'"' {
                    return false;
                }
                if has_enum && !any_int && (b == b'-' || b.is_ascii_digit()) {
                    return false;
                }
                // enum values are strings and integers only: an object or array never matches
                if has_enum && (b == b'{' || b == b'[') {
                    return false;
                }
                match b {
                    b'{' => {
                        push_obj(m, si);
                        true
                    }
                    b'[' => {
                        push_arr(m, si);
                        true
                    }
                    b'"' => {
                        push_str(m, si);
                        true
                    }
                    b't' => {
                        if has_enum {
                            return false;
                        }
                        m.stack.push(Frame::Lit { rest: b"rue" });
                        true
                    }
                    b'f' => {
                        if has_enum {
                            return false;
                        }
                        m.stack.push(Frame::Lit { rest: b"alse" });
                        true
                    }
                    b'n' => {
                        if has_enum {
                            return false;
                        }
                        m.stack.push(Frame::Lit { rest: b"ull" });
                        true
                    }
                    _ => match num_start(b) {
                        Some((st, buf)) => {
                            if !self.num_prefix_viable(si, &buf, st) {
                                return false;
                            }
                            m.stack.push(Frame::Num { si, buf, st });
                            true
                        }
                        None => false,
                    },
                }
            }
        }
    }

    fn required_satisfied(&self, si: u32, seen: &[u32]) -> bool {
        self.schemas[si as usize].required.iter().all(|r| seen.contains(r))
    }

    /// Value-string constraint check. `closing` = the terminating quote was consumed.
    fn str_value_ok(&self, si: u32, st: &StrState, closing: bool) -> bool {
        let s = &self.schemas[si as usize];
        if let Some(mx) = s.max_len.filter(|_| !st.free_len) {
            // A started escape or UTF-8 sequence already commits one more character: counting it
            // only once complete let a string at maxLength begin one, after which no byte could
            // finish it (a dead end that ends the reply mid-document).
            if st.cp as u64 + pending_char(st) > mx {
                return false;
            }
        }
        if closing {
            if !st.is_closed_interior() {
                return false;
            }
            if let Some(mn) = s.min_len.filter(|_| !st.free_len) {
                if (st.cp as u64) < mn {
                    return false;
                }
            }
            if !s.enum_str.is_empty() && !s.enum_str.iter().any(|e| canon_match(e, &st.raw).1) {
                return false;
            }
        } else if !s.enum_str.is_empty() && !s.enum_str.iter().any(|e| canon_match(e, &st.raw).0) {
            // enum prefix rule: only bytes that keep us on the way to an allowed value.
            return false;
        }
        true
    }

    fn num_ok(&self, si: u32, buf: &[u8], st: NumState) -> bool {
        if !st.terminal() {
            return false;
        }
        let s = &self.schemas[si as usize];
        let Ok(text) = std::str::from_utf8(buf) else { return false };
        let Ok(v) = text.parse::<f64>() else { return false };
        if s.ty == Ty::Integer && !matches!(st, NumState::Zero | NumState::Int) {
            return false; // integer forbids fraction/exponent
        }
        // a number matches only the enum's integer values (an untyped enum may list strings too)
        if s.has_enum && !s.enum_int.iter().any(|&e| e as f64 == v) {
            return false;
        }
        if let Some(m) = s.minimum {
            if v < m {
                return false;
            }
        }
        if let Some(m) = s.excl_min {
            if v <= m {
                return false;
            }
        }
        if let Some(m) = s.maximum {
            if v > m {
                return false;
            }
        }
        if let Some(m) = s.excl_max {
            if v >= m {
                return false;
            }
        }
        true
    }

    /// Can a number whose bytes so far are `buf` (grammar position `st`) still END UP valid?
    ///
    /// `num_ok` runs only when the number TERMINATES, so without this check the allowed set let the
    /// model run an unbounded digit sequence after a prefix that could no longer complete.
    ///
    /// The rule may prune the mask (refuse a prefix) only when no valid completion exists, or
    /// when it narrows the spelling of a value that is still reachable another way:
    ///   - an integer enum: the canonical spelling of a listed value only (as for string enums);
    ///   - `integer`: no fraction or exponent (`num_ok` refuses them), and the digits must still
    ///     reach an integer inside the bounds (the values `p*10^k ..= p*10^k + 10^k - 1`);
    ///   - `number`: a sign the bounds exclude (a nonzero negative with no negative value allowed,
    ///     a nonzero positive with no positive value allowed). Any other float prefix can still
    ///     be moved into range by `.`/`e` (`70e-1 == 7`).
    fn num_prefix_viable(&self, si: u32, buf: &[u8], st: NumState) -> bool {
        let s = &self.schemas[si as usize];
        if !s.enum_int.is_empty() {
            return s.enum_int.iter().any(|e| e.to_string().as_bytes().starts_with(buf));
        }
        let negative = buf.first() == Some(&b'-');
        let digits = if negative { &buf[1..] } else { buf };
        let lead = digits.first().copied();
        match s.ty {
            Ty::Integer => {}
            Ty::Number => {
                let neg_ok = s.minimum.map_or(true, |m| m < 0.0) && s.excl_min.map_or(true, |m| m < 0.0);
                let pos_ok = s.maximum.map_or(true, |m| m > 0.0) && s.excl_max.map_or(true, |m| m > 0.0);
                let zero_ok = s.minimum.map_or(true, |m| m <= 0.0) && s.excl_min.map_or(true, |m| m < 0.0)
                    && s.maximum.map_or(true, |m| m >= 0.0) && s.excl_max.map_or(true, |m| m > 0.0);
                let lower = s.minimum.unwrap_or(f64::NEG_INFINITY).max(s.excl_min.unwrap_or(f64::NEG_INFINITY));
                let upper = s.maximum.unwrap_or(f64::INFINITY).min(s.excl_max.unwrap_or(f64::INFINITY));
                return match lead {
                    None => neg_ok || zero_ok, // a bare `-`: `-0` or a negative
                    // `0`, `0.x`: a value in [0, 1); `-0`, `-0.x`: a value in (-1, 0]
                    Some(b'0') if negative => zero_ok || (neg_ok && upper > -1.0),
                    Some(b'0') => zero_ok || (pos_ok && lower < 1.0),
                    Some(_) if negative => neg_ok,
                    Some(_) => pos_ok,
                };
            }
            _ => return true,
        }
        if !matches!(st, NumState::NegStart | NumState::Zero | NumState::Int) {
            return false;
        }
        // the integer bounds
        let lo = s.minimum.map_or(f64::NEG_INFINITY, f64::ceil)
            .max(s.excl_min.map_or(f64::NEG_INFINITY, |m| m.floor() + 1.0));
        let hi = s.maximum.map_or(f64::INFINITY, f64::floor)
            .min(s.excl_max.map_or(f64::INFINITY, |m| m.ceil() - 1.0));
        if lo > hi {
            return false;
        }
        if digits.is_empty() {
            // a bare `-`: `-0` (= 0) or a negative integer
            return lo <= hi.min(0.0);
        }
        // `0` / `-0` cannot be extended by another digit (leading zeros are illegal JSON)
        if st == NumState::Zero {
            return self.num_ok(si, buf, st);
        }
        let Ok(text) = std::str::from_utf8(digits) else { return true };
        let Ok(p) = text.parse::<f64>() else { return true };
        // the magnitudes this prefix can still become: [p*10^k, p*10^k + 10^k - 1], k = 0, 1, ...
        let mut scale = 1.0f64;
        loop {
            let (a, b) = (p * scale, p * scale + scale - 1.0);
            let (vlo, vhi) = if negative { (-b, -a) } else { (a, b) };
            if vlo <= hi && vhi >= lo {
                return true;
            }
            // every longer number lies further from zero, so past the bound means never
            if (!negative && a > hi) || (negative && -a < lo) {
                return false;
            }
            if scale > 1e19 {
                return true; // beyond the integer range checked exactly here; `num_ok` decides
            }
            scale *= 10.0;
        }
    }

    /// Pop a terminal pending number frame (used by `is_complete`); false = it was not valid.
    fn finalize_pending_number(&self, m: &mut Machine) -> bool {
        if let Some(Frame::Num { si, buf, st }) = m.stack.last().cloned() {
            if self.num_ok(si, &buf, st) {
                m.stack.pop();
                return true;
            }
            return false;
        }
        true
    }

    /// Can the key being parsed still become a key this object accepts? Declared keys already
    /// present (`seen`) are excluded: a duplicate is refused when the key closes, so letting it
    /// through here led into a state with no allowed continuation.
    fn key_prefix_ok(&self, si: u32, st: &StrState, seen: &[u32]) -> bool {
        let s = &self.schemas[si as usize];
        s.additional
            || s.props.iter().enumerate().any(|(i, (n, _))| !seen.contains(&(i as u32)) && canon_match(n, &st.raw).0)
    }

    /// May another key start in this object (`"` after `{`, `,` after a member)?
    fn can_add_key(&self, si: u32, seen: &[u32]) -> bool {
        let s = &self.schemas[si as usize];
        s.additional || (0..s.props.len() as u32).any(|i| !seen.contains(&i))
    }

    /// Feed ONE byte. Returns true when the byte was consumed by a live machine; false when the
    /// machine died (a [`Frame::Dead`] is pushed and absorbs everything afterwards).
    ///
    /// Frames are mutated only after being popped, which is what makes the [`Roll`] undo log and
    /// the "pop, then let the parent reprocess the same byte" loop sound.
    fn feed(&self, m: &mut Machine, b: u8, roll: Option<&mut Roll>) -> bool {
        // Trailing whitespace after a complete document (top = DocEnd) is not counted: the
        // completed state stays one interned state, and the scheduler ends the turn there anyway.
        let structural_ws = is_ws(b) && !m.in_string() && !matches!(m.stack.last(), Some(Frame::DocEnd));
        // No whitespace before the root value: the document starts at its first value byte (a
        // leading run carries no information and only spends tokens).
        if structural_ws && m.stack.len() == 2 && matches!(m.stack[1], Frame::Value { .. }) {
            return false;
        }
        if structural_ws && m.ws >= MAX_WS_RUN {
            // Refused without touching the stack: both callers discard the machine (step) or
            // roll it back (trie_walk) on `false`.
            return false;
        }
        let alive = self.feed_inner(m, b, roll);
        if alive {
            m.ws = if structural_ws { m.ws + 1 } else { 0 };
        }
        alive
    }

    fn feed_inner(&self, m: &mut Machine, b: u8, mut roll: Option<&mut Roll>) -> bool {
        loop {
            if m.stack.is_empty() {
                m.stack.push(Frame::Dead);
                return false;
            }
            let idx = m.stack.len() - 1;
            let top = m.stack.pop().expect("checked non-empty");
            if let Some(r) = roll.as_deref_mut() {
                r.note_pop(idx, &top);
            }
            return match top {
                Frame::Dead => {
                    m.stack.push(Frame::Dead);
                    return false;
                }
                Frame::DocEnd => {
                    if is_ws(b) {
                        m.stack.push(Frame::DocEnd);
                        return true;
                    }
                    m.stack.push(Frame::Dead);
                    return false;
                }
                Frame::Value { si } => {
                    if is_ws(b) {
                        m.stack.push(Frame::Value { si });
                        return true;
                    }
                    if self.value_start(m, si, b) {
                        return true;
                    }
                    m.stack.push(Frame::Dead);
                    return false;
                }
                Frame::Obj { si, phase, mut seen, mut extra } => match phase {
                    ObjPhase::KeyOrEnd => {
                        if is_ws(b) {
                            m.stack.push(Frame::Obj { si, phase: ObjPhase::KeyOrEnd, seen, extra });
                            return true;
                        }
                        if b == b'}' {
                            if self.required_satisfied(si, &seen) {
                                return true; // object closed; the byte belongs to this frame
                            }
                            m.stack.push(Frame::Dead);
                            return false;
                        }
                        if b == b'"' && self.can_add_key(si, &seen) {
                            m.stack.push(Frame::Obj {
                                si,
                                phase: ObjPhase::Key(StrState::new()),
                                seen,
                                extra,
                            });
                            return true;
                        }
                        m.stack.push(Frame::Dead);
                        return false;
                    }
                    ObjPhase::Key(mut st) => {
                        match str_advance(&mut st, b, true) {
                            Step::Open => {
                                if !self.key_prefix_ok(si, &st, &seen) {
                                    m.stack.push(Frame::Dead);
                                    return false;
                                }
                                m.stack.push(Frame::Obj { si, phase: ObjPhase::Key(st), seen, extra });
                                true
                            }
                            Step::Closed => {
                                if !st.is_closed_interior() {
                                    m.stack.push(Frame::Dead);
                                    return false;
                                }
                                let Ok(key) = String::from_utf8(st.dec.clone()) else {
                                    m.stack.push(Frame::Dead);
                                    return false;
                                };
                                let s = &self.schemas[si as usize];
                                let child = match s.props.iter().position(|(n, _)| *n == key) {
                                    Some(i) => {
                                        if seen.contains(&(i as u32)) {
                                            m.stack.push(Frame::Dead);
                                            return false;
                                        }
                                        seen.push(i as u32);
                                        s.props[i].1
                                    }
                                    None => {
                                        if !s.additional || extra.iter().any(|k| *k == key) {
                                            m.stack.push(Frame::Dead);
                                            return false;
                                        }
                                        extra.push(key);
                                        0 // universal permissive schema
                                    }
                                };
                                m.stack.push(Frame::Obj {
                                    si,
                                    phase: ObjPhase::Colon { child },
                                    seen,
                                    extra,
                                });
                                true
                            }
                            Step::Bad => {
                                m.stack.push(Frame::Dead);
                                false
                            }
                        }
                    }
                    // After a `,` a key must follow: `}` here would be a trailing comma (invalid
                    // JSON), so it can never be emitted.
                    ObjPhase::KeyReq => {
                        if is_ws(b) {
                            m.stack.push(Frame::Obj { si, phase: ObjPhase::KeyReq, seen, extra });
                            return true;
                        }
                        if b == b'"' {
                            m.stack.push(Frame::Obj {
                                si,
                                phase: ObjPhase::Key(StrState::new()),
                                seen,
                                extra,
                            });
                            return true;
                        }
                        m.stack.push(Frame::Dead);
                        return false;
                    }
                    ObjPhase::Colon { child } => {
                        if is_ws(b) {
                            m.stack.push(Frame::Obj { si, phase: ObjPhase::Colon { child }, seen, extra });
                            return true;
                        }
                        if b == b':' {
                            // Consume the separator, then require a value above us.
                            m.stack.push(Frame::Obj { si, phase: ObjPhase::AfterValue, seen, extra });
                            m.stack.push(Frame::Value { si: child });
                            return true;
                        }
                        m.stack.push(Frame::Dead);
                        false
                    }
                    ObjPhase::AfterValue => {
                        if is_ws(b) {
                            m.stack.push(Frame::Obj { si, phase: ObjPhase::AfterValue, seen, extra });
                            return true;
                        }
                        if b == b',' && self.can_add_key(si, &seen) {
                            m.stack.push(Frame::Obj { si, phase: ObjPhase::KeyReq, seen, extra });
                            return true;
                        }
                        if b == b'}' {
                            if self.required_satisfied(si, &seen) {
                                return true;
                            }
                            m.stack.push(Frame::Dead);
                            return false;
                        }
                        m.stack.push(Frame::Dead);
                        false
                    }
                },
                Frame::Arr { si, n, phase } => match phase {
                    ArrPhase::ItemOrEnd | ArrPhase::ItemReq => {
                        // Just after a `,` (`ItemReq`) a `]` would be a trailing comma: refuse it.
                        let need_item = matches!(phase, ArrPhase::ItemReq);
                        if is_ws(b) {
                            let here = if need_item { ArrPhase::ItemReq } else { ArrPhase::ItemOrEnd };
                            m.stack.push(Frame::Arr { si, n, phase: here });
                            return true;
                        }
                        if b == b']' {
                            let s = &self.schemas[si as usize];
                            if !need_item && s.min_items.map_or(true, |m0| n >= m0) {
                                return true; // array closed
                            }
                            m.stack.push(Frame::Dead);
                            return false;
                        }
                        let s = &self.schemas[si as usize];
                        if let Some(mx) = s.max_items {
                            if n >= mx {
                                m.stack.push(Frame::Dead);
                                return false;
                            }
                        }
                        let item_si = s.items.unwrap_or(0);
                        m.stack.push(Frame::Arr { si, n: n + 1, phase: ArrPhase::AfterItem });
                        m.stack.push(Frame::Value { si: item_si });
                        continue; // same byte starts the item
                    }
                    ArrPhase::AfterItem => {
                        if is_ws(b) {
                            m.stack.push(Frame::Arr { si, n, phase: ArrPhase::AfterItem });
                            return true;
                        }
                        // A `,` must leave room for one more item (maxItems), or the array can only
                        // dead-end.
                        if b == b',' && self.schemas[si as usize].max_items.map_or(true, |mx| n < mx) {
                            m.stack.push(Frame::Arr { si, n, phase: ArrPhase::ItemReq });
                            return true;
                        }
                        if b == b']' {
                            let s = &self.schemas[si as usize];
                            if s.min_items.map_or(true, |m0| n >= m0) {
                                return true;
                            }
                            m.stack.push(Frame::Dead);
                            return false;
                        }
                        m.stack.push(Frame::Dead);
                        false
                    }
                },
                Frame::Str { si, mut st } => {
                    let keep = !self.schemas[si as usize].enum_str.is_empty();
                    match str_advance(&mut st, b, keep) {
                        Step::Open => {
                            if self.str_value_ok(si, &st, false) {
                                m.stack.push(Frame::Str { si, st });
                                true
                            } else {
                                m.stack.push(Frame::Dead);
                                false
                            }
                        }
                        Step::Closed => {
                            if self.str_value_ok(si, &st, true) {
                                true
                            } else {
                                m.stack.push(Frame::Dead);
                                false
                            }
                        }
                        Step::Bad => {
                            m.stack.push(Frame::Dead);
                            false
                        }
                    }
                }
                Frame::Lit { rest } => {
                    if rest.is_empty() || rest[0] != b {
                        m.stack.push(Frame::Dead);
                        return false;
                    }
                    if rest.len() == 1 {
                        return true; // literal complete
                    }
                    m.stack.push(Frame::Lit { rest: &rest[1..] });
                    true
                }
                Frame::Num { si, mut buf, st } => {
                    if let Some(next) = num_advance(st, b) {
                        buf.push(b);
                        // W2: refuse a digit that puts the number permanently out of range.
                        if !self.num_prefix_viable(si, &buf, next) {
                            m.stack.push(Frame::Dead);
                            return false;
                        }
                        m.stack.push(Frame::Num { si, buf, st: next });
                        return true;
                    }
                    // The number cannot continue: validate it, then hand the byte to the parent.
                    if self.num_ok(si, &buf, st) {
                        continue;
                    }
                    m.stack.push(Frame::Dead);
                    false
                }
            };
        }
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
mod tests {
    use super::*;

    fn byte_vocab() -> Vec<(u32, Vec<u8>)> {
        (0u32..256).map(|b| (b, vec![b as u8])).collect()
    }

    fn compile_err(schema_json: &str) -> String {
        let v: Value = serde_json::from_str(schema_json).expect("test schema is valid JSON");
        let rf = serde_json::json!({
            "type": "json_schema",
            "json_schema": {"name": "t", "strict": true, "schema": v}
        });
        compile_response_format(&rf).expect_err("schema must be refused")
    }

    fn mask(schema_json: &str) -> SchemaMask {
        let v: Value = serde_json::from_str(schema_json).expect("test schema is valid JSON");
        let rf = serde_json::json!({
            "type": "json_schema",
            "json_schema": {"name": "t", "strict": true, "schema": v}
        });
        let mut m = compile_response_format(&rf)
            .unwrap_or_else(|e| panic!("schema unexpectedly refused: {e}"))
            .expect("schema must be enforced");
        m.set_vocab(byte_vocab());
        m
    }

    /// Feed a byte string as a sequence of one-byte tokens; `None` = refused.
    fn walk(m: &SchemaMask, s: &str) -> Option<u32> {
        let mut st = m.start_state();
        for &b in s.as_bytes() {
            st = m.step(st, b as u32)?;
        }
        Some(st)
    }

    /// Full-document acceptance: every byte feeds AND the document is finished.
    fn ok(m: &SchemaMask, s: &str) -> bool {
        match walk(m, s) {
            Some(st) => m.is_complete(st),
            None => false,
        }
    }

    fn bits(ids: &[u32], words: usize) -> Vec<u32> {
        let mut v = vec![0u32; words];
        for &i in ids {
            v[(i >> 5) as usize] |= 1u32 << (i & 31);
        }
        v
    }

    // ---- 0: W2 ENFORCEMENT — the escapes the 0.8B probe actually produced --------------------
    //
    // These are the exact failure shapes seen on hardware (2026-09-14, `w2_probe.py` against the
    // 0.8B trunk): a truncated \uXXXX escape, an integer far outside `maximum`, a raw control
    // character inside a string, and a digit run that never closes the document. The byte-level
    // walk isolates the MACHINE from the token plumbing, so a failure here is the FSM's own bug
    // (and a pass here means the escape came from a path that never armed the mask).

    const OBJ_SCHEMA: &str = r#"{"type":"object","properties":{
        "answer":{"type":"string","enum":["ALPHA","BETA"]},
        "n":{"type":"integer","minimum":1,"maximum":3},
        "why":{"type":"string","minLength":1,"maxLength":40}},
        "required":["answer","n","why"],"additionalProperties":false}"#;

    #[test]
    fn w2_accepts_a_valid_document() {
        let m = mask(OBJ_SCHEMA);
        assert!(ok(&m, r#"{"answer":"ALPHA","n":2,"why":"because"}"#),
                "the positive control must be accepted");
    }

    #[test]
    fn w2_refuses_the_observed_escapes() {
        let m = mask(OBJ_SCHEMA);
        let bad: &[(&str, &str)] = &[
            ("truncated escape", r#"{"answer":"ALPHA","n":2,"why":"\u043!"}"#),
            ("raw newline in string", "{\"answer\":\"ALPHA\",\"n\":2,\"why\":\"a\nb\"}"),
            ("below minimum", r#"{"answer":"ALPHA","n":-1,"why":"x"}"#),
            ("above maximum", r#"{"answer":"ALPHA","n":4,"why":"x"}"#),
            ("far above maximum", r#"{"answer":"ALPHA","n":100000000000,"why":"x"}"#),
            ("enum violation", r#"{"answer":"GAMMA","n":2,"why":"x"}"#),
            ("additionalProperties", r#"{"answer":"ALPHA","n":2,"why":"x","extra":1}"#),
            ("missing required", r#"{"answer":"ALPHA","n":2}"#),
        ];
        let mut escaped = Vec::new();
        for (what, doc) in bad {
            if ok(&m, doc) { escaped.push(*what); }
        }
        assert!(escaped.is_empty(), "these documents were ACCEPTED by the FSM: {escaped:?}");
    }

    #[test]
    fn w2_allowed_set_prunes_at_the_number_bound() {
        // The bitset is what the kernel masks with: after `{"n":3` only `,` / `}` may follow.
        let m = mask(OBJ_SCHEMA);
        let st = walk(&m, r#"{"answer":"ALPHA","n":3"#).expect("prefix must be walkable");
        let bits = m.allowed(st);
        for d in b'0'..=b'9' {
            let t = d as usize;
            assert_eq!(bits[t >> 5] & (1u32 << (t & 31)), 0,
                       "digit '{}' is allowed after the maximum", d as char);
        }
        let st0 = walk(&m, r#"{"answer":"ALPHA","n":"#).expect("prefix must be walkable");
        let b0 = m.allowed(st0);
        for c in [b'-', b'0'] {
            let t = c as usize;
            assert_eq!(b0[t >> 5] & (1u32 << (t & 31)), 0,
                       "'{}' is allowed below the minimum", c as char);
        }
    }

    #[test]
    fn w2_allowed_set_tracks_the_escape_state() {
        let m = mask(OBJ_SCHEMA);
        let st = walk(&m, r#"{"answer":"ALPHA","n":2,"why":"\u04"#).expect("prefix must be walkable");
        let bits = m.allowed(st);
        let allowed = |c: u8| { let t = c as usize; bits[t >> 5] & (1u32 << (t & 31)) != 0 };
        assert!(allowed(b'A') && allowed(b'9'), "a 4th hex digit must be allowed");
        assert!(!allowed(b'!'), "'!' must NOT be allowed while a hex digit is pending");
        assert!(!allowed(b'"'), "the string must not close inside an escape");
    }

    // ---- 1: every rejected keyword is named loudly ------------------------------------------

    #[test]
    fn rejected_keywords_are_named_loudly() {
        let cases: &[(&str, &str)] = &[
            ("pattern", r#"{"type":"string","pattern":"^a$"}"#),
            ("format", r#"{"type":"string","format":"date-time"}"#),
            ("oneOf", r#"{"oneOf":[{"type":"string"}]}"#),
            ("anyOf", r#"{"anyOf":[{"type":"string"}]}"#),
            ("allOf", r#"{"allOf":[{"type":"string"}]}"#),
            ("not", r#"{"not":{"type":"string"}}"#),
            ("$ref", r##"{"$ref":"#/defs/x"}"##),
            ("patternProperties", r#"{"type":"object","patternProperties":{"a":{"type":"string"}}}"#),
            ("minProperties", r#"{"type":"object","minProperties":1}"#),
            ("maxProperties", r#"{"type":"object","maxProperties":3}"#),
            ("$defs", r#"{"$defs":{"a":{"type":"string"}}}"#),
            ("definitions", r#"{"definitions":{"a":{"type":"string"}}}"#),
            ("if", r#"{"if":{"type":"string"}}"#),
            ("then", r#"{"then":{"type":"string"}}"#),
            ("else", r#"{"else":{"type":"string"}}"#),
            ("const", r#"{"const":1}"#),
            ("uniqueItems", r#"{"type":"array","uniqueItems":true}"#),
            ("contains", r#"{"type":"array","contains":{"type":"string"}}"#),
            ("prefixItems", r#"{"type":"array","prefixItems":[{"type":"string"}]}"#),
            ("additionalItems", r#"{"type":"array","additionalItems":false}"#),
            ("propertyNames", r#"{"type":"object","propertyNames":{"type":"string"}}"#),
            ("dependencies", r#"{"type":"object","dependencies":{"a":["b"]}}"#),
            ("dependentSchemas", r#"{"type":"object","dependentSchemas":{"a":{}}}"#),
            ("dependentRequired", r#"{"type":"object","dependentRequired":{"a":["b"]}}"#),
            ("unevaluatedProperties", r#"{"type":"object","unevaluatedProperties":false}"#),
            ("unevaluatedItems", r#"{"type":"array","unevaluatedItems":false}"#),
            // unknown keywords are refused just as loudly
            ("title", r#"{"type":"string","title":"nope"}"#),
            ("x-vendor", r#"{"type":"string","x-vendor":true}"#),
        ];
        for (kw, js) in cases {
            let e = compile_err(js);
            assert!(e.contains(&format!("'{kw}'")), "error must name '{kw}': {e}");
            assert!(e.contains("not supported"), "error for '{kw}' must say so: {e}");
        }
        // ... and the message tells the caller what to do instead.
        let e = compile_err(r#"{"type":"string","pattern":"^a$"}"#);
        assert!(e.contains("V1 supports"), "must list the supported keywords: {e}");
    }

    #[test]
    fn type_dependent_keywords_are_named_loudly() {
        // A supported keyword on the wrong type is refused by name, never ignored.
        for (kw, js) in [
            ("minLength", r#"{"minLength":3}"#),
            ("minItems", r#"{"type":"string","minItems":1}"#),
            ("minimum", r#"{"type":"string","minimum":1}"#),
            ("properties", r#"{"type":"array","properties":{"a":{"type":"string"}}}"#),
            ("enum", r#"{"type":"boolean","enum":["a"]}"#),
        ] {
            let e = compile_err(js);
            assert!(e.contains(&format!("'{kw}'")), "error must name '{kw}': {e}");
        }
        let e = compile_err(r#"{"type":"union_thing"}"#);
        assert!(e.contains("'type'"), "{e}");
        let e = compile_err(r#"{"type":["string","null"]}"#);
        assert!(e.contains("'type'"), "{e}");
    }

    // ---- 2: basic object ---------------------------------------------------------------------

    #[test]
    fn object_required_and_additional_properties() {
        let m = mask(r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}"#);
        assert!(ok(&m, r#"{"a":"x"}"#));
        assert!(ok(&m, r#"{ "a" : "x" }"#)); // whitespace is free
        assert!(ok(&m, r#"{"a":"\u0078"}"#)); // escapes decode to the same string
        assert!(!ok(&m, r#"{"a":1}"#), "string property must reject a number");
        assert!(!ok(&m, r#"{}"#), "missing required key");
        assert!(!ok(&m, r#"{"a":"x","b":1}"#), "additionalProperties defaults to false");
        assert!(!ok(&m, r#"{"a":"x"}extra"#), "trailing junk after the document");
        assert!(!ok(&m, r#"{"a":"x","a":"x"}"#), "duplicate key");
        assert!(!ok(&m, r#"{"a":"x",}"#), "trailing comma in object");
        assert!(!ok(&m, r#"{"a":"x""#), "unterminated document is not complete");
        assert!(!ok(&m, r#"{"b":"x"}"#), "unknown key with additionalProperties false");
        assert!(!ok(&m, r#"[]"#), "wrong value kind");

        // explicit additionalProperties: true re-opens the key set (each at most once)
        let open = mask(r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":true}"#);
        assert!(ok(&open, r#"{"a":"x","b":1}"#));
        assert!(ok(&open, r#"{"b":null,"a":"x"}"#));
        assert!(!ok(&open, r#"{"a":"x","a":"y"}"#), "duplicates are rejected even for declared keys");
        assert!(!ok(&open, r#"{"b":1,"b":2}"#), "duplicates are rejected for extra keys too");
        assert!(!ok(&open, r#"{"b":1}"#), "required still applies");

        // a bare object / json_object accepts any shape of object
        let any_obj = mask(r#"{"type":"json_object"}"#);
        assert!(ok(&any_obj, r#"{}"#));
        assert!(ok(&any_obj, r#"{"anything":[1,{"x":null}]}"#));
        assert!(!ok(&any_obj, r#"[]"#));
        assert!(ok(&mask(r#"{"type":"object"}"#), r#"{"z":1}"#));
    }

    // ---- 3: enum -----------------------------------------------------------------------------

    #[test]
    fn enum_prefix_rule() {
        let m = mask(r#"{"enum":["alpha","beta"]}"#);
        assert!(ok(&m, r#""alpha""#));
        assert!(ok(&m, r#""beta""#));
        assert!(!ok(&m, r#""gamma""#), "must die as soon as the prefix leaves the enum");
        assert!(!ok(&m, r#""alph""#), "an unterminated string is not a complete value");
        // the prefix rule is visible at the byte level: after `"` only a/b may start
        let st = walk(&m, r#"""#).unwrap();
        assert!(m.step(st, b'a' as u32).is_some());
        assert!(m.step(st, b'b' as u32).is_some());
        assert!(m.step(st, b'g' as u32).is_none());

        let typed = mask(r#"{"type":"string","enum":["a","b"]}"#);
        assert!(ok(&typed, r#""a""#));
        assert!(!ok(&typed, r#""c""#));
        assert!(!ok(&typed, r#"1"#), "declared type still applies");

        let ints = mask(r#"{"type":"integer","enum":[1,2]}"#);
        assert!(ok(&ints, "1"));
        assert!(ok(&ints, "2"));
        assert!(!ok(&ints, "3"));
        assert!(!ok(&ints, "12"));
        assert!(!ok(&ints, "1.0"), "integer forbids a fraction");
        assert!(!ok(&ints, r#""1""#));
    }

    // ---- 4: array bounds ---------------------------------------------------------------------

    #[test]
    fn array_item_bounds() {
        let m = mask(r#"{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":2}"#);
        assert!(!ok(&m, "[]"), "minItems 1");
        assert!(ok(&m, "[1]"));
        assert!(ok(&m, "[1,2]"));
        assert!(ok(&m, "[ 1 , 2 ]"));
        assert!(!ok(&m, "[1,2,3]"), "maxItems 2");
        assert!(!ok(&m, "[[1]]"), "items must be integers");
        assert!(!ok(&m, "[1,]"), "trailing comma");
        assert!(!ok(&m, "[1")); // unterminated
        let unbounded = mask(r#"{"type":"array"}"#);
        assert!(ok(&unbounded, "[]"));
        assert!(ok(&unbounded, r#"[1,"a",null,{"b":[]}]"#));
    }

    // ---- 5: nesting --------------------------------------------------------------------------

    #[test]
    fn nested_object_array_object() {
        let m = mask(
            r#"{"type":"object","properties":{"xs":{"type":"array","items":{"type":"object",
               "properties":{"n":{"type":"integer","minimum":0}},"required":["n"],
               "additionalProperties":false}}},"required":["xs"],"additionalProperties":false}"#,
        );
        assert!(ok(&m, r#"{"xs":[{"n":1},{"n":2}]}"#));
        assert!(ok(&m, r#"{"xs":[]}"#));
        assert!(!ok(&m, r#"{"xs":[{"n":-1}]}"#), "minimum applies at depth 3");
        assert!(!ok(&m, r#"{"xs":[{"n":1,"z":0}]}"#), "additionalProperties applies at depth 3");
        assert!(!ok(&m, r#"{"xs":[{"z":1}]}"#), "required applies at depth 3");
        assert!(!ok(&m, r#"{"xs":[{"n":1}]"#), "unterminated nesting");
        assert!(!ok(&m, r#"{"xs":{"n":1}}"#), "object where the array belongs");
        assert!(m.summary().contains("root=object"), "{}", m.summary());
        assert!(m.summary().contains("max_depth=4"), "{}", m.summary());
    }

    #[test]
    fn depth_cap_is_loud() {
        let deep = |levels: usize| {
            let mut v = serde_json::json!({"type":"string"});
            for _ in 0..levels {
                v = serde_json::json!({"type":"object","properties":{"x": v}});
            }
            v.to_string()
        };
        // 12 nested property levels are fine; 13 are refused by name.
        assert!(compile_response_format(&serde_json::json!({
            "type":"json_schema",
            "json_schema":{"name":"deep","schema": serde_json::from_str::<Value>(&deep(11)).unwrap()}
        }))
        .is_ok());
        let v: Value = serde_json::from_str(&deep(13)).unwrap();
        let rf = serde_json::json!({"type":"json_schema","json_schema":{"name":"deep","schema": v}});
        let e = compile_response_format(&rf).expect_err("13 levels must be refused");
        assert!(e.contains("'properties'"), "{e}");
        assert!(e.contains("deeper than the V1 limit"), "{e}");
    }

    // ---- 6: token-level layer ----------------------------------------------------------------

    #[test]
    fn tiny_vocab_allowed_and_step_agree() {
        let mut m = mask(r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}"#);
        // Pieces exercise the trie: single bytes, mid-document fragments, and whole documents.
        let pieces: Vec<(u32, Vec<u8>)> = vec![
            (1, b"{".to_vec()),
            (2, b"\"a\"".to_vec()),
            (3, b"}".to_vec()),
            (4, b"1".to_vec()),
            (5, b"\"x\"".to_vec()),
            (6, b":".to_vec()),
            (7, b",".to_vec()),
            (8, b"{\"".to_vec()),
            (9, b"a\":".to_vec()),
            (10, b"\"x\"}".to_vec()),
            (11, b" ".to_vec()),
            (12, b"{\"a\":\"x\"}".to_vec()),
        ];
        m.set_vocab(pieces);
        let words = vocab_words(13);
        assert_eq!(words, 1);

        let s0 = m.start_state();
        // At the start only tokens whose whole byte run is accepted may fire: `{`, `{"` and the
        // complete document (no whitespace before the root value).
        assert_eq!(m.allowed(s0).as_slice(), bits(&[1, 8, 12], words).as_slice());
        for bad in [2u32, 3, 4, 5, 6, 7, 9, 10, 11] {
            assert_eq!(m.step(s0, bad), None, "token {bad} must not be usable at the start");
        }

        // Walk a full document with the tokenizer's pieces, asserting each use was allowed.
        let used = [1u32, 2, 6, 5, 3];
        let mut st = s0;
        let mut produced = String::new();
        for tok in used {
            let allowed = m.allowed(st);
            assert!(allowed[(tok >> 5) as usize] & (1u32 << (tok & 31)) != 0, "token {tok} was not allowed");
            st = m.step(st, tok).expect("used token must be steppable");
            produced.push_str(match tok {
                1 => "{",
                2 => "\"a\"",
                6 => ":",
                5 => "\"x\"",
                3 => "}",
                _ => unreachable!(),
            });
        }
        assert_eq!(produced, r#"{"a":"x"}"#);
        assert!(m.is_complete(st), "document finished");

        // State-specific expectations (also shows required-key handling inside `allowed`).
        let after_brace = m.step(s0, 1).unwrap();
        assert_eq!(m.allowed(after_brace).as_slice(), bits(&[2, 11], words).as_slice());
        assert_eq!(m.step(after_brace, 3), None, "cannot close before 'a' is present");

        let after_key = m.step(after_brace, 2).unwrap();
        assert_eq!(m.allowed(after_key).as_slice(), bits(&[6, 11], words).as_slice());

        let after_colon = m.step(after_key, 6).unwrap();
        assert_eq!(m.allowed(after_colon).as_slice(), bits(&[2, 5, 10, 11], words).as_slice());
        assert_eq!(m.step(after_colon, 4), None, "1 is not a string");

        let after_value = m.step(after_colon, 5).unwrap();
        // No `,` (7): the only declared key is present and additional properties are off, so a
        // comma could only lead to a key slot that nothing can fill.
        assert_eq!(m.allowed(after_value).as_slice(), bits(&[3, 11], words).as_slice());

        let done = m.step(after_value, 3).unwrap();
        assert!(m.is_complete(done));
        // Trailing whitespace keeps the machine in the same (interned) completed state.
        assert_eq!(m.step(done, 11), Some(done));
        assert_eq!(m.allowed(done).as_slice(), bits(&[11], words).as_slice());

        // End-of-generation tokens are only ever reachable through end_mask.
        let em = m.end_mask(&[3]);
        assert_eq!(em.as_slice(), bits(&[3], words).as_slice());
        assert_eq!(m.allowed(done).as_slice(), bits(&[11], words).as_slice());
    }

    #[test]
    fn unknown_and_empty_tokens_are_never_allowed() {
        let mut m = mask(r#"{"type":"integer"}"#);
        m.set_vocab(vec![(0, Vec::new()), (1, b"4".to_vec()), (2, b"2".to_vec())]);
        let s0 = m.start_state();
        let a = m.allowed(s0);
        assert_eq!(a.as_slice(), bits(&[1, 2], 1).as_slice());
        assert_eq!(m.step(s0, 0), None, "empty piece cannot advance the machine");
        assert_eq!(m.step(s0, 7), None, "token missing from the table");
        assert_eq!(m.step(s0, 9999), None);
    }

    #[test]
    fn vocab_not_set_fails_closed() {
        // A mask with no vocabulary attached must emit nothing, not everything.
        let bare = compile_response_format(&serde_json::json!({
            "type": "json_schema",
            "json_schema": {"name": "t", "schema": {"type": "string"}}
        }))
        .unwrap()
        .expect("schema is enforced");
        assert!(!bare.is_dead(bare.start_state()), "the machine itself is alive");
        assert!(bare.allowed(bare.start_state()).is_empty(), "no vocab: nothing may be emitted");
        // ... and an explicitly empty vocabulary behaves the same way.
        let mut empty = mask(r#"{"type":"string"}"#);
        empty.set_vocab(Vec::new());
        assert!(empty.allowed(empty.start_state()).is_empty());
    }

    // ---- 7: completeness ---------------------------------------------------------------------

    #[test]
    fn complete_state_tracks_document_end() {
        let m = mask(r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}"#);
        let s0 = m.start_state();
        assert!(!m.is_complete(s0), "empty output is not a document");
        for prefix in [r#"{"#, r#"{"a"#, r#"{"a":"#, "{\"a\":1"] {
            let st = walk(&m, prefix).unwrap_or_else(|| panic!("prefix {prefix} should be steppable"));
            assert!(!m.is_complete(st), "{prefix} is not finished");
        }
        let mid = walk(&m, r#"{"a":1"#).unwrap();
        assert!(!m.is_complete(mid), "the object is still open");
        // A pending number IS finished at the top level: `is_complete` finalizes it.
        let nums = mask(r#"{"type":"integer","maximum":10}"#);
        assert!(nums.is_complete(walk(&nums, "7").unwrap()));
        assert!(walk(&nums, "70").is_none(),
                "an integer prefix past `maximum` can never return to range — refused mid-prefix");
        // The same bound on a FLOAT is only decidable once the number ends: `70e-1` is 7, so the
        // prefix stays allowed and the refusal lands at completion.
        let floats = mask(r#"{"type":"number","maximum":10}"#);
        assert!(!floats.is_complete(walk(&floats, "70").unwrap()), "over the bound");
        let finished = walk(&m, r#"{"a":1}"#).unwrap();
        assert!(m.is_complete(finished));
        assert!(m.is_complete(walk(&m, "{\"a\":1} \n\t").unwrap()), "trailing whitespace");
        assert!(walk(&m, r#"{"a":1} x"#).is_none(), "junk is refused outright");
        assert!(walk(&m, r#"{"a":false}"#).is_none(), "wrong value kind is refused");
        assert!(!m.is_dead(finished), "a finished document is alive");
        assert!(m.is_dead(12345), "unknown states report dead (fail closed)");
    }

    // ---- 8: cache ----------------------------------------------------------------------------

    #[test]
    fn allowed_is_cached_and_idempotent() {
        let m = mask(r#"{"type":"integer","minimum":5,"maximum":10}"#);
        let s0 = m.start_state();
        let a = m.allowed(s0);
        let b = m.allowed(s0);
        assert_eq!(a, b, "same bitset");
        assert!(Arc::ptr_eq(&a, &b), "second call must be the cached Arc");
        let before = m.summary();
        assert!(before.contains("cached_masks=1"), "{before}");
        assert!(before.contains("vocab=256 tokens/8 words"), "{before}");
        let _ = m.allowed(s0);
        assert!(m.summary().contains("cached_masks=1"), "no growth on repeat");
    }

    // ---- additional coverage: numbers, strings, response_format shapes ------------------------

    #[test]
    fn number_grammar_and_bounds() {
        let m = mask(r#"{"type":"integer","minimum":5,"maximum":10}"#);
        assert!(ok(&m, "5"));
        assert!(ok(&m, "10"));
        assert!(!ok(&m, "4"));
        assert!(!ok(&m, "11"));
        assert!(!ok(&m, "05"), "leading zero");
        assert!(!ok(&m, "-5"));
        assert!(!ok(&m, "5.0"), "integer forbids fraction");
        assert!(!ok(&m, "1e1"), "integer forbids exponent");
        assert!(!ok(&m, "5x"));
        assert!(!ok(&m, "-"));

        let n = mask(r#"{"type":"number","exclusiveMinimum":0,"exclusiveMaximum":1}"#);
        assert!(ok(&n, "0.5"));
        assert!(ok(&n, "1e-3"));
        assert!(!ok(&n, "0"), "exclusive bound");
        assert!(!ok(&n, "1"), "exclusive bound");
        assert!(!ok(&n, "-1"));
        assert!(!ok(&n, "1."), "dangling fraction point");
        assert!(!ok(&n, "1e"), "dangling exponent");

        let e = mask(r#"{"type":"number","maximum":2,"exclusiveMaximum":2}"#);
        assert!(!ok(&e, "2"));
    }

    #[test]
    fn string_lengths_are_code_points() {
        let m = mask(r#"{"type":"string","minLength":2,"maxLength":3}"#);
        assert!(ok(&m, r#""ab""#));
        assert!(ok(&m, r#""abc""#));
        assert!(!ok(&m, r#""a""#));
        assert!(!ok(&m, r#""abcd""#));
        assert!(ok(&m, r#""\u0041B""#), "an escape is one code point");
        assert!(ok(&m, r#""\ud83d\ude00x""#), "a surrogate pair is one code point");
        assert!(!ok(&m, r#""\ud83d\ude00""#), "one code point < minLength 2");
        let one = mask(r#"{"type":"string","maxLength":1}"#);
        assert!(ok(&one, "\"\u{1F600}\""), "4 UTF-8 bytes, 1 code point");
        assert!(!ok(&one, r#""ab""#));
        assert!(!ok(&m, "\"a\nb\""), "raw newline must be escaped");
        assert!(ok(&m, r#""a\nb""#), "escaped newline is fine");
        assert!(!ok(&m, r#""\ud83d""#), "lone high surrogate");
        assert!(!ok(&m, r#""\q""#), "unknown escape");
    }

    #[test]
    fn response_format_shapes() {
        // nothing to enforce
        assert!(compile_response_format(&Value::Null).unwrap().is_none());
        assert!(compile_response_format(&serde_json::json!({"type":"text"})).unwrap().is_none());
        // json_object = any object
        let m = compile_response_format(&serde_json::json!({"type":"json_object"}))
            .unwrap()
            .expect("json_object is enforced");
        assert!(m.summary().contains("root=object"), "{}", m.summary());
        // bare form: {"type":"json_schema","schema":{...}}
        let bare = compile_response_format(
            &serde_json::json!({"type":"json_schema","schema":{"type":"string"}}),
        )
        .unwrap()
        .expect("bare form is enforced");
        assert!(bare.summary().contains("root=string"), "{}", bare.summary());
        // strict/name/description are accepted and ignored
        assert!(compile_response_format(&serde_json::json!({
            "type":"json_schema",
            "json_schema":{"name":"n","description":"d","strict":true,"schema":{"type":"null"}}
        }))
        .is_ok());
        // loud failures
        let e = compile_response_format(&serde_json::json!({
            "type":"json_schema","json_schema":{"name":"n"}
        }))
        .expect_err("missing schema");
        assert!(e.contains("'schema'"), "{e}");
        let e = compile_response_format(&serde_json::json!({"type":"yaml"})).expect_err("bad type");
        assert!(e.contains("'type'"), "{e}");
        let e = compile_response_format(&serde_json::json!({"nope":1})).expect_err("no type");
        assert!(e.contains("'type'"), "{e}");
        assert!(compile_response_format(&serde_json::json!([1,2])).is_err());
        assert!(compile_response_format(&serde_json::json!({"type":"json_schema","json_schema":{"name":"n","schema":{"type":"string"}},"extra":1})).is_err());
    }

    #[test]
    fn permissive_and_literal_values() {
        let m = mask("{}");
        assert!(ok(&m, "123"));
        assert!(ok(&m, "-1.5e3"));
        assert!(ok(&m, r#""s""#));
        assert!(ok(&m, "true"));
        assert!(ok(&m, "null"));
        assert!(ok(&m, r#"[1,"a"]"#));
        assert!(!ok(&m, "tru"));
        assert!(!ok(&m, "[1,]"));
        assert!(ok(&mask(r#"{"type":"boolean"}"#), "false"));
        assert!(!ok(&mask(r#"{"type":"boolean"}"#), "truex"));
        assert!(ok(&mask(r#"{"type":"null"}"#), "null"));
        assert!(!ok(&mask(r#"{"type":"null"}"#), "nul"));
    }

    // ---- dead-end freedom: every reachable state allows a token or is complete -----------------

    /// Random walks over the allowed sets (single-byte vocabulary). A state that accepts no byte
    /// and is not a complete document is a dead end: under the token mask the sampler would have
    /// nothing to pick. Every allowed byte must also step.
    fn no_dead_ends(schema: &str, walks: usize, max_len: usize) {
        let m = mask(schema);
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        for w in 0..walks {
            let mut st = m.start_state();
            let mut doc = Vec::new();
            for _ in 0..max_len {
                if m.is_complete(st) { break; }
                let bits = m.allowed(st);
                let ids: Vec<u32> = (0..256u32).filter(|&t| bits[(t >> 5) as usize] >> (t & 31) & 1 == 1).collect();
                assert!(!ids.is_empty(), "walk {w}: dead end after {:?}", String::from_utf8_lossy(&doc));
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let t = ids[((rng >> 33) as usize) % ids.len()];
                st = m.step(st, t).unwrap_or_else(|| panic!("walk {w}: allowed byte {t} refused by step"));
                doc.push(t as u8);
            }
        }
    }

    #[test]
    fn no_dead_end_at_max_length() {
        // At maxLength a started escape or UTF-8 sequence could never be finished.
        no_dead_ends(r#"{"type":"object","properties":{"t":{"type":"string","maxLength":3}},"required":["t"]}"#, 400, 80);
        no_dead_ends(r#"{"type":"array","items":{"type":"string","minLength":1,"maxLength":1},"minItems":2,"maxItems":2}"#, 400, 80);
        let m = mask(r#"{"type":"object","properties":{"t":{"type":"string","maxLength":2}}}"#);
        assert!(walk(&m, r#"{"t":"a\u00"#).is_some(), "an escape may start while there is room for its character");
        assert!(walk(&m, r#"{"t":"ab\"#).is_none(), "no escape at maxLength");
        let at_max = walk(&m, r#"{"t":"ab"#).unwrap();
        assert!(m.step(at_max, 0xC3).is_none(), "no UTF-8 lead byte at maxLength");
        let closed = m.step(at_max, b'"' as u32).unwrap();
        assert!(m.step(closed, b'}' as u32).is_some());
    }

    #[test]
    fn enum_candidates_respect_length_bounds() {
        let m = mask(r#"{"type":"string","enum":["a","bbb"],"minLength":3}"#);
        assert!(walk(&m, r#""a"#).is_none(), "a value shorter than minLength is not a candidate");
        assert!(ok(&m, r#""bbb""#));
        let m = mask(r#"{"type":"string","enum":["abc","de"],"maxLength":2}"#);
        assert!(walk(&m, r#""a"#).is_none(), "a value longer than maxLength is not a candidate");
        assert!(ok(&m, r#""de""#));
        assert!(compile_err(r#"{"type":"string","enum":["a"],"minLength":2}"#).contains("enum"));
        let m = mask(r#"{"type":"integer","enum":[1,20],"maximum":10}"#);
        assert!(walk(&m, "2").is_none(), "an integer outside the bounds is not a candidate");
        assert!(ok(&m, "1"));
    }

    #[test]
    fn integer_prefixes_must_reach_a_valid_value() {
        let m = mask(r#"{"type":"integer"}"#);
        assert!(walk(&m, "1.").is_none(), "no fraction for an integer");
        assert!(walk(&m, "1e").is_none(), "no exponent for an integer");
        assert!(ok(&m, "-12"));
        let m = mask(r#"{"type":"integer","minimum":5,"maximum":10}"#);
        assert!(walk(&m, "2").is_none(), "2, 20.. are all outside 5..10");
        assert!(walk(&m, "1").is_some(), "1 can still become 10");
        assert!(walk(&m, "11").is_none());
        assert!(walk(&m, "-").is_none());
        assert!(ok(&m, "5") && ok(&m, "10") && !ok(&m, "1"));
        let m = mask(r#"{"type":"integer","minimum":-10,"maximum":-5}"#);
        assert!(walk(&m, "-1").is_some() && walk(&m, "-2").is_none() && walk(&m, "0").is_none());
        assert!(ok(&m, "-5") && ok(&m, "-10"));
        let m = mask(r#"{"type":"integer","exclusiveMinimum":4.5,"exclusiveMaximum":7}"#);
        assert!(walk(&m, "4").is_none() && walk(&m, "7").is_none());
        assert!(ok(&m, "5") && ok(&m, "6"));
        let m = mask(r#"{"type":"integer","enum":[1,2]}"#);
        assert!(walk(&m, "3").is_none() && walk(&m, "10").is_none() && walk(&m, "1.").is_none());
        assert!(ok(&m, "1") && ok(&m, "2"));
        let m = mask(r#"{"type":"number","minimum":0}"#);
        assert!(walk(&m, "-1").is_none(), "no negative number when none is allowed");
        assert!(ok(&m, "-0") && ok(&m, "2.5"));
        let m = mask(r#"{"type":"number","maximum":-1}"#);
        assert!(walk(&m, "5").is_none() && walk(&m, "0").is_none() && walk(&m, "-0").is_none());
        assert!(ok(&m, "-1.5"));
        no_dead_ends(r#"{"type":"object","properties":{"a":{"type":"integer","minimum":5,"maximum":10},"b":{"type":"integer","enum":[1,2,30]}},"required":["a","b"]}"#, 400, 80);
    }

    #[test]
    fn untyped_enum_admits_only_its_values() {
        let m = mask(r#"{"enum":["alpha","beta"]}"#);
        assert!(walk(&m, "{").is_none() && walk(&m, "[").is_none());
        assert!(ok(&m, r#""alpha""#));
        let m = mask(r#"{"type":"object","properties":{"k":{"enum":["x",3]}},"required":["k"]}"#);
        assert!(walk(&m, r#"{"k":{"#).is_none() && walk(&m, r#"{"k":["#).is_none());
        assert!(ok(&m, r#"{"k":"x"}"#) && ok(&m, r#"{"k":3}"#));
        assert!(walk(&m, r#"{"k":4"#).is_none());
    }

    #[test]
    fn no_lone_surrogate_escape_prefix() {
        let m = mask(r#"{"type":"string"}"#);
        assert!(walk(&m, r#""\udc"#).is_none(), "a low surrogate cannot start a pair");
        assert!(walk(&m, r#""\ud834\u"#).is_some());
        assert!(walk(&m, r#""\ud834\ue"#).is_none(), "a pair needs a low surrogate second");
        assert!(walk(&m, r#""\ud834\ud8"#).is_none());
        assert!(ok(&m, r#""\ud834\udd1e""#));
        no_dead_ends(r#"{"type":"object","properties":{"t":{"type":"string"}},"required":["t"]}"#, 2000, 60);
    }

    #[test]
    fn no_dead_end_enum_with_escapes_and_spaces() {
        no_dead_ends(r#"{"type":"object","properties":{"color":{"type":"string","enum":["red","green","light blue","naïve","𝄞 clef"]}},"required":["color"]}"#, 400, 200);
    }

    #[test]
    fn no_dead_end_object_keys_and_arrays() {
        no_dead_ends(r#"{"type":"object","properties":{"a":{"type":"integer","minimum":0,"maximum":9},"ab":{"type":"boolean"},"list":{"type":"array","items":{"type":"string","enum":["x","y"]},"maxItems":2}},"required":["a"]}"#, 400, 200);
    }

    #[test]
    fn enum_values_and_keys_use_the_canonical_spelling() {
        let m = mask(r#"{"type":"string","enum":["no","not x","𝄞","say \"hi\"\n"]}"#);
        assert!(ok(&m, r#""not x""#));
        assert!(ok(&m, r#""𝄞""#), "non-ASCII is written raw");
        assert!(ok(&m, r#""say \"hi\"\n""#), "quote and newline use their short escapes");
        assert!(walk(&m, r#""no\u0074"#).is_none(), "an escaped spelling of a plain character is refused");
        assert!(walk(&m, r#""\ud834"#).is_none(), "so is a surrogate escape of a raw-written character");
        assert!(walk(&m, "\"no\u{e9}").is_none(), "a raw multi-byte char no candidate continues with");
        assert!(walk(&m, r#""no\"#).is_none(), "a backslash no candidate needs");
        let k = mask(r#"{"type":"object","properties":{"ab":{"type":"integer"}},"required":["ab"]}"#);
        assert!(ok(&k, r#"{"ab":1}"#));
        assert!(walk(&k, r#"{"\u0061"#).is_none(), "declared keys too");
        assert_eq!(canon_match("a\u{1}b", b"a\\u0001b"), (true, true));
        assert_eq!(canon_match("ab", b"a"), (true, false));
        assert_eq!(canon_match("ab", b"abc"), (false, false));
    }

    #[test]
    fn mask_key_shares_free_string_states() {
        let m = mask(r#"{"type":"object","properties":{"t":{"type":"string"}}}"#);
        let a = walk(&m, r#"{"t":"a"#).unwrap();
        let b = walk(&m, r#"{"t":"abcdef"#).unwrap();
        assert_ne!(a, b, "distinct machine states");
        assert!(Arc::ptr_eq(&m.allowed(a), &m.allowed(b)), "one cached mask for both");
    }

    #[test]
    fn mask_key_is_exact_near_length_bounds() {
        // multi-byte pieces, so a token can add several characters at once
        let mut pieces = byte_vocab();
        let n = pieces.len() as u32;
        for (i, p) in ["ab", "abc", "abcd\"", "\"}", "xyz"].iter().enumerate() {
            pieces.push((n + i as u32, p.as_bytes().to_vec()));
        }
        for schema in [r#"{"type":"object","properties":{"t":{"type":"string","minLength":3,"maxLength":7}}}"#,
                       r#"{"type":"object","properties":{"t":{"type":"string","maxLength":4}}}"#,
                       r#"{"type":"object","properties":{"t":{"type":"string","minLength":5}}}"#] {
            let mut m = mask(schema);
            m.set_vocab(pieces.clone());
            for len in 0..9 {
                let Some(st) = walk(&m, &format!(r#"{{"t":"{}"#, "q".repeat(len))) else { continue };
                assert_eq!(*m.allowed(st), m.compute_allowed(st), "{schema}: {len} chars");
            }
        }
    }

    /// Multi-byte pieces with escapes, UTF-8, closing quotes and structure after the close.
    fn mixed_vocab() -> Vec<(u32, Vec<u8>)> {
        let mut pieces = byte_vocab();
        let n = pieces.len() as u32;
        let extra: [&[u8]; 14] = [b"ab", b"abc", b"abcd\"", b"\"}", b"xyz", b"\\n", b"q\\t",
            "\u{e9}".as_bytes(), "a\u{e9}b\"".as_bytes(), b"\\u00e9", b"\\u00", b"e9\"}",
            b"\", \"", &[0xC3]];
        for (i, p) in extra.iter().enumerate() {
            pieces.push((n + i as u32, p.to_vec()));
        }
        pieces
    }

    #[test]
    fn length_bounded_masks_match_full_walk() {
        // Every state reached by a random walk over multi-byte pieces: the derived mask near the
        // length bounds must equal a full trie walk (including mid-escape / mid-UTF-8 states,
        // which take the full walk).
        for schema in [r#"{"type":"object","properties":{"t":{"type":"string","minLength":3,"maxLength":7}}}"#,
                       r#"{"type":"object","properties":{"t":{"type":"string","maxLength":4},"u":{"type":"string","maxLength":2}}}"#,
                       r#"{"type":"object","properties":{"t":{"type":"string","minLength":5}}}"#,
                       r#"{"type":"array","items":{"type":"string","maxLength":3},"maxItems":4}"#] {
            let mut m = mask(schema);
            m.set_vocab(mixed_vocab());
            let mut rng = 0x2545_F491_4F6C_DD1Du64;
            let mut checked = 0;
            for _ in 0..200 {
                let mut st = m.start_state();
                for _ in 0..24 {
                    if m.is_complete(st) { break; }
                    let bits = m.allowed(st);
                    assert_eq!(*bits, m.compute_allowed(st), "{schema}");
                    checked += 1;
                    let ids: Vec<u32> = (0..m.vocab_size as u32)
                        .filter(|&t| bits[(t >> 5) as usize] >> (t & 31) & 1 == 1).collect();
                    if ids.is_empty() { break; }
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    st = m.step(st, ids[((rng >> 33) as usize) % ids.len()]).expect("allowed token refused");
                }
            }
            assert!(checked > 200, "{schema}: only {checked} states checked");
        }
    }

    #[test]
    fn str_run_counts_characters_not_bytes() {
        assert_eq!(str_run_of(b"abc"), 3);
        assert_eq!(str_run_of(b"ab\"}"), 2 | STR_RUN_CLOSES);
        assert_eq!(str_run_of(b"\\n"), 1);
        assert_eq!(str_run_of(b"\\u00e9x"), 2);
        assert_eq!(str_run_of(b"\\u00"), 1); // escape still open: its character is committed
        assert_eq!(str_run_of("\u{e9}\u{e9}".as_bytes()), 2);
        assert_eq!(str_run_of(&[0xC3]), 1); // UTF-8 sequence still open: same
        assert_eq!(str_run_of(&[0x01]), STR_RUN_BAD);
    }

    #[test]
    fn structural_whitespace_is_capped() {
        let m = mask(r#"{"type":"object","properties":{"a":{"type":"string"}}}"#);
        let lead = " ".repeat(MAX_WS_RUN as usize);
        assert!(walk(&m, " ").is_none(), "no whitespace before the root value");
        assert!(ok(&m, &format!("{{{lead}\"a\":\"x\"}}")), "a run up to the cap is fine");
        assert!(walk(&m, &format!("{{{lead} ")).is_none(), "one more is refused");
        assert!(ok(&m, &format!("{{\"a\":\"{}\"}}", " ".repeat(100))), "string content is not structural");
        assert!(ok(&m, &format!("{{{lead}\"a\"{lead}:{lead}\"x\"{lead}}}")), "the run resets at every token");
    }

    #[test]
    fn no_key_or_item_slot_that_cannot_be_filled() {
        let m = mask(r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}}}"#);
        assert!(walk(&m, r#"{"a":1,"a"#).is_none(), "a duplicate declared key cannot start");
        assert!(ok(&m, r#"{"a":1,"b":2}"#));
        assert!(walk(&m, r#"{"a":1,"b":2,"#).is_none(), "no key left: the comma is refused");
        let arr = mask(r#"{"type":"array","items":{"type":"integer"},"maxItems":2}"#);
        assert!(ok(&arr, "[1,2]"));
        assert!(walk(&arr, "[1,2,").is_none(), "maxItems reached: the comma is refused");
        let none = mask(r#"{"type":"object","properties":{},"additionalProperties":false}"#);
        assert!(ok(&none, "{}"));
        assert!(walk(&none, "{\"").is_none(), "no key can ever be accepted");
    }
}
