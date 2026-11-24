//! Small accessors over the ordered `Json` value (it has no mutators of its own).

use crate::dsv4_chat::Json;

pub fn get<'a>(v: &'a Json, key: &str) -> Option<&'a Json> {
    v.get(key)
}

pub fn get_str<'a>(v: &'a Json, key: &str) -> Option<&'a str> {
    v.get(key).and_then(|x| x.as_str())
}

pub fn has(v: &Json, key: &str) -> bool {
    v.get(key).is_some()
}

/// Set a key: keep its position when present, append otherwise (Python dict assignment).
pub fn set(v: &mut Json, key: &str, val: Json) {
    if let Json::Object(o) = v {
        if let Some(slot) = o.iter_mut().find(|(k, _)| k == key) {
            slot.1 = val;
        } else {
            o.push((key.to_string(), val));
        }
    }
}

pub fn remove(v: &mut Json, key: &str) -> Option<Json> {
    if let Json::Object(o) = v {
        if let Some(i) = o.iter().position(|(k, _)| k == key) {
            return Some(o.remove(i).1);
        }
    }
    None
}

pub fn get_mut<'a>(v: &'a mut Json, key: &str) -> Option<&'a mut Json> {
    if let Json::Object(o) = v {
        o.iter_mut().find(|(k, _)| k == key).map(|(_, x)| x)
    } else {
        None
    }
}

pub fn obj(pairs: Vec<(&str, Json)>) -> Json {
    Json::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

pub fn s(x: &str) -> Json {
    Json::Str(x.to_string())
}

/// Python truthiness.
pub fn truthy(v: Option<&Json>) -> bool {
    v.map_or(false, super::pyjson::truthy)
}

/// `dict.get(key)` followed by `is not None`.
pub fn present(v: &Json, key: &str) -> bool {
    !matches!(v.get(key), None | Some(Json::Null))
}
