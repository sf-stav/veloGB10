//! Python `json` compatibility on top of the ordered `Json` value of `dsv4_chat` (additive use only,
//! `dsv4_chat` is not changed). `json.loads` then `json.dumps(ensure_ascii=False)` differs from a
//! verbatim re-emit in two places that reach the prompt: duplicate object keys (the last value wins,
//! at the position of the first) and numbers (`1e3` becomes `1000.0`, `-0` becomes `0`, `1.50` becomes
//! `1.5`). `canon` applies both, so `dumps(&canon(v))` is byte-identical to `json.dumps(json.loads(..))`.

use crate::dsv4_chat::{dumps, parse_json, Json};

use super::error::ChatError;

/// Python `repr(float)` for a finite or infinite f64, as `json.dumps` prints it.
pub fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // `{:e}` gives the shortest round-trip digits as d.ddde<exp>.
    let s = format!("{:e}", f.abs());
    let (mant, exp) = s.split_once('e').expect("exp form");
    let exp: i32 = exp.parse().expect("exp int");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let sign = if f < 0.0 { "-" } else { "" };
    let decpt = exp + 1; // value = 0.DIGITS * 10^decpt
    let mut out = String::from(sign);
    if decpt > 16 || decpt < -3 {
        // scientific: d[.ddd]e(+|-)XX with at least two exponent digits
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        out.push('e');
        out.push(if e < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", e.abs()));
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..(-decpt) {
            out.push('0');
        }
        out.push_str(&digits);
    } else if (decpt as usize) >= digits.len() {
        out.push_str(&digits);
        for _ in 0..(decpt as usize - digits.len()) {
            out.push('0');
        }
        out.push_str(".0");
    } else {
        out.push_str(&digits[..decpt as usize]);
        out.push('.');
        out.push_str(&digits[decpt as usize..]);
    }
    out
}

/// A JSON number's source text as Python would print it after `loads`/`dumps`.
pub fn py_number(raw: &str) -> String {
    if raw.bytes().any(|b| matches!(b, b'.' | b'e' | b'E')) {
        match raw.parse::<f64>() {
            Ok(f) => py_float_repr(f),
            Err(_) => raw.to_string(),
        }
    } else {
        // an int: Python keeps arbitrary precision; only "-0" changes
        let t = raw.trim_start_matches('-');
        if t.bytes().all(|b| b == b'0') {
            "0".to_string()
        } else {
            raw.to_string()
        }
    }
}

/// Apply Python's dict and number semantics to a parsed value.
pub fn canon(v: &Json) -> Json {
    match v {
        Json::Num(s) => Json::Num(py_number(s)),
        Json::Array(a) => Json::Array(a.iter().map(canon).collect()),
        Json::Object(o) => {
            let mut out: Vec<(String, Json)> = Vec::with_capacity(o.len());
            for (k, x) in o {
                let x = canon(x);
                if let Some(slot) = out.iter_mut().find(|(kk, _)| kk == k) {
                    slot.1 = x;
                } else {
                    out.push((k.clone(), x));
                }
            }
            Json::Object(out)
        }
        other => other.clone(),
    }
}

/// `json.dumps(v, ensure_ascii=False)` of a value that came from `json.loads`.
pub fn to_json(v: &Json) -> String {
    dumps(&canon(v))
}

/// `json.loads(text)` into a canonical value.
pub fn loads(text: &str) -> Result<Json, ChatError> {
    parse_json(text).map(|j| canon(&j)).map_err(|e| ChatError::value(format!("invalid JSON: {}", e.1)))
}

/// Python truthiness of a value (`if x:`).
pub fn truthy(v: &Json) -> bool {
    match v {
        Json::Null => false,
        Json::Bool(b) => *b,
        Json::Num(s) => py_number(s) != "0" && py_number(s) != "0.0" && py_number(s) != "-0.0",
        Json::Str(s) => !s.is_empty(),
        Json::Array(a) => !a.is_empty(),
        Json::Object(o) => !o.is_empty(),
    }
}

/// `str(x)` for the scalar values a template can receive; containers are refused (their Python repr
/// is not something an API client can rely on).
pub fn py_str(v: &Json) -> Result<String, ChatError> {
    Ok(match v {
        Json::Null => "None".into(),
        Json::Bool(true) => "True".into(),
        Json::Bool(false) => "False".into(),
        Json::Num(s) => py_number(s),
        Json::Str(s) => s.clone(),
        _ => return Err(ChatError::value("a container where text is expected".to_string())),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        let cases: &[(&str, &str)] = &[
            ("1.0", "1.0"), ("1.50", "1.5"), ("1e3", "1000.0"), ("1E5", "100000.0"), ("0.1", "0.1"),
            ("1e16", "1e+16"), ("1e15", "1000000000000000.0"), ("0.0001", "0.0001"), ("0.00001", "1e-05"),
            ("1.5e-7", "1.5e-07"), ("123456789.125", "123456789.125"), ("-2.5", "-2.5"), ("0.0", "0.0"),
            ("-0.0", "-0.0"), ("1e400", "Infinity"), ("12345678901234567890.0", "1.2345678901234567e+19"),
            ("-0", "0"), ("007".trim_start_matches('0'), "7"), ("9007199254740993", "9007199254740993"),
        ];
        for (i, o) in cases {
            assert_eq!(&py_number(i), o, "input {i}");
        }
    }

    #[test]
    fn canon_dedupes_keys_last_wins_first_position() {
        let v = parse_json(r#"{"a":1,"b":2,"a":3}"#).unwrap();
        assert_eq!(to_json(&v), r#"{"a": 3, "b": 2}"#);
    }
}
