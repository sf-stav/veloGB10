//! The schema IR and its compiler: `Ty`, `Schema`, keyword validation and `compile_schema` (V1 subset; every
//! unsupported keyword is a loud error naming it). Feeds `build_mask` in `mod.rs`; map of the directory in `mod.rs`.

use super::*;

// ===========================================================================================
// Schema IR + validation
// ===========================================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Ty {
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
    pub(super) fn name(self) -> &'static str {
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
pub(super) struct Schema {
    pub(super) ty: Ty,
    pub(super) props: Vec<(String, u32)>,
    /// Indices into `props`.
    pub(super) required: Vec<u32>,
    /// `false` = keys outside `props` are rejected. See the module docs for the default.
    pub(super) additional: bool,
    pub(super) items: Option<u32>,
    pub(super) min_items: Option<u64>,
    pub(super) max_items: Option<u64>,
    pub(super) min_len: Option<u64>,
    pub(super) max_len: Option<u64>,
    pub(super) minimum: Option<f64>,
    pub(super) maximum: Option<f64>,
    pub(super) excl_min: Option<f64>,
    pub(super) excl_max: Option<f64>,
    pub(super) enum_str: Vec<String>,
    pub(super) enum_int: Vec<i64>,
    pub(super) has_enum: bool,
    pub(super) depth: usize,
}

impl Schema {
    pub(super) fn any() -> Schema {
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
pub(super) fn compile_schema(v: &Value, depth: usize, via: &str, table: &mut Vec<Schema>) -> Result<u32, String> {
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
