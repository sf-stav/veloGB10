//! Message preprocessing of `encoding.py`: `merge_tool_messages`, `sort_tool_results_by_call_order`,
//! image placeholder validation and `process_image_messages`. Each function mirrors the Python one and
//! works on copies; the caller's messages are never changed.

use crate::dsv4_chat::Json;

use super::error::ChatError;
use super::jx::{self, get, get_mut, has, obj, present, remove, s, set, truthy};

pub const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

fn role_of(m: &Json) -> Option<&str> {
    jx::get_str(m, "role")
}

/// `_validate_no_image_sp_tokens`
pub fn validate_no_image_sp_tokens(msg: &Json) -> Result<(), ChatError> {
    if let Some(Json::Str(c)) = get(msg, "content") {
        if c.contains(IMAGE_PLACEHOLDER) {
            return Err(ChatError::value(format!(
                "Message content contains image special token '{IMAGE_PLACEHOLDER}'. Images should be provided as image content blocks."
            )));
        }
    }
    if let Some(Json::Str(c)) = get(msg, "reasoning_content") {
        if c.contains(IMAGE_PLACEHOLDER) {
            return Err(ChatError::value(format!("reasoning_content contains image special token '{IMAGE_PLACEHOLDER}'")));
        }
    }
    Ok(())
}

/// `merge_tool_messages`: tool results become `tool_result` blocks of a user message.
pub fn merge_tool_messages(messages: &[Json]) -> Result<Vec<Json>, ChatError> {
    let mut merged: Vec<Json> = Vec::with_capacity(messages.len());
    for msg in messages {
        let mut msg = msg.clone();
        let role = role_of(&msg).map(str::to_string);
        match role.as_deref() {
            Some("tool") => {
                let mut content = get(&msg, "content").cloned().unwrap_or_else(|| s(""));
                if truthy(get(&msg, "content_blocks")) && !matches!(get(&msg, "content"), Some(Json::Str(_))) {
                    content = get(&msg, "content_blocks").cloned().unwrap();
                }
                let tool_block = obj(vec![
                    ("type", s("tool_result")),
                    ("tool_use_id", get(&msg, "tool_call_id").cloned().unwrap_or_else(|| s(""))),
                    ("content", content),
                ]);
                let into_prev = merged.last().map_or(false, |m| role_of(m) == Some("user") && has(m, "content_blocks"));
                if into_prev {
                    push_block(merged.last_mut().unwrap(), tool_block)?;
                } else {
                    merged.push(obj(vec![("role", s("user")), ("content_blocks", Json::Array(vec![tool_block]))]));
                }
            }
            Some("user") => {
                let blocks: Vec<Json> = match get(&msg, "content_blocks") {
                    Some(Json::Array(a)) => a.clone(),
                    Some(Json::Null) | None => match get(&msg, "content") {
                        Some(Json::Array(a)) => a.clone(),
                        Some(other) => vec![obj(vec![("type", s("text")), ("text", other.clone())])],
                        None => vec![obj(vec![("type", s("text")), ("text", s(""))])],
                    },
                    Some(_) => return Err(ChatError::input("content_blocks must be a list")),
                };
                let into_prev = merged.last().map_or(false, |m| {
                    role_of(m) == Some("user") && has(m, "content_blocks") && !present(m, "task")
                });
                if into_prev {
                    let last = merged.last_mut().unwrap();
                    if let Some(Json::Array(a)) = get_mut(last, "content_blocks") {
                        a.extend(blocks);
                    } else {
                        return Err(ChatError::input("content_blocks must be a list"));
                    }
                } else {
                    set(&mut msg, "content_blocks", Json::Array(blocks));
                    merged.push(msg);
                }
            }
            _ => merged.push(msg),
        }
    }
    Ok(merged)
}

fn push_block(msg: &mut Json, block: Json) -> Result<(), ChatError> {
    match get_mut(msg, "content_blocks") {
        Some(Json::Array(a)) => {
            a.push(block);
            Ok(())
        }
        _ => Err(ChatError::input("content_blocks must be a list")),
    }
}

/// `sort_tool_results_by_call_order`: in place, over the combined context + messages list.
pub fn sort_tool_results_by_call_order(messages: &mut [Json]) {
    let mut order: Vec<(String, usize)> = Vec::new();
    for msg in messages.iter_mut() {
        let role = role_of(msg).map(str::to_string);
        if role.as_deref() == Some("assistant") && truthy(get(msg, "tool_calls")) {
            order.clear();
            if let Some(Json::Array(tcs)) = get(msg, "tool_calls") {
                for (idx, tc) in tcs.iter().enumerate() {
                    // `tc.get("id") or tc.get("function", {}).get("id", "")`
                    let id = match get(tc, "id") {
                        Some(Json::Str(x)) if !x.is_empty() => x.clone(),
                        _ => get(tc, "function").and_then(|f| jx::get_str(f, "id")).unwrap_or("").to_string(),
                    };
                    if !id.is_empty() {
                        // a repeated id keeps its LAST index (dict assignment)
                        if let Some(slot) = order.iter_mut().find(|(k, _)| *k == id) {
                            slot.1 = idx;
                        } else {
                            order.push((id, idx));
                        }
                    }
                }
            }
        } else if role.as_deref() == Some("user") && truthy(get(msg, "content_blocks")) {
            let blocks = match get(msg, "content_blocks") {
                Some(Json::Array(a)) => a.clone(),
                _ => continue,
            };
            let idxs: Vec<usize> = blocks
                .iter()
                .enumerate()
                .filter(|(_, b)| jx::get_str(b, "type") == Some("tool_result"))
                .map(|(i, _)| i)
                .collect();
            if idxs.len() > 1 && !order.is_empty() {
                let mut keyed: Vec<(usize, Json)> = idxs
                    .iter()
                    .map(|&i| {
                        let id = jx::get_str(&blocks[i], "tool_use_id").unwrap_or("");
                        let key = order.iter().find(|(k, _)| k == id).map(|(_, p)| *p).unwrap_or(0);
                        (key, blocks[i].clone())
                    })
                    .collect();
                keyed.sort_by_key(|(k, _)| *k); // stable, like Python's sorted
                let mut it = keyed.into_iter();
                let new_blocks: Vec<Json> = blocks
                    .iter()
                    .map(|b| {
                        if jx::get_str(b, "type") == Some("tool_result") {
                            it.next().unwrap().1
                        } else {
                            b.clone()
                        }
                    })
                    .collect();
                set(msg, "content_blocks", Json::Array(new_blocks));
            }
        }
    }
}

/// `parse_tagged_text`: `<image>path</image>` text into content blocks (a plain string when it has none).
pub fn parse_tagged_text(text: &str) -> Result<Json, ChatError> {
    const OPEN: &str = "<image>";
    const CLOSE: &str = "</image>";
    // regex `<image>(.*?)</image>` with DOTALL: leftmost, lazy
    let mut matches: Vec<(usize, usize, &str)> = Vec::new();
    let mut from = 0;
    while let Some(o) = text[from..].find(OPEN) {
        let o = from + o;
        let Some(c) = text[o + OPEN.len()..].find(CLOSE) else { break };
        let c = o + OPEN.len() + c;
        matches.push((o, c + CLOSE.len(), &text[o + OPEN.len()..c]));
        from = c + CLOSE.len();
    }
    let mut remaining = String::new();
    let mut cur = 0;
    for (st, en, _) in &matches {
        remaining.push_str(&text[cur..*st]);
        cur = *en;
    }
    remaining.push_str(&text[cur..]);
    if remaining.contains(OPEN) || remaining.contains(CLOSE) {
        return Err(ChatError::value("Malformed <image>path</image> tag"));
    }
    if matches.is_empty() {
        return Ok(s(text));
    }
    let mut blocks = Vec::new();
    let mut cursor = 0;
    for (st, en, path) in &matches {
        if *st > cursor {
            blocks.push(obj(vec![("type", s("text")), ("text", s(&text[cursor..*st]))]));
        }
        if path.is_empty() {
            return Err(ChatError::value("Image path must not be empty"));
        }
        blocks.push(obj(vec![("type", s("image_url")), ("image_url", obj(vec![("url", s(path))]))]));
        cursor = *en;
    }
    if cursor < text.len() {
        blocks.push(obj(vec![("type", s("text")), ("text", s(&text[cursor..]))]));
    }
    Ok(Json::Array(blocks))
}

fn is_image_block(b: &Json) -> bool {
    matches!(jx::get_str(b, "type"), Some("image") | Some("image_url"))
}

fn extract_image(b: &Json) -> Result<Json, ChatError> {
    let mut rec: Vec<(&str, Json)> = vec![("type", s("image"))];
    if jx::get_str(b, "type") == Some("image_url") {
        let url = match get(b, "image_url") {
            Some(Json::Str(u)) => Json::Str(u.clone()),
            Some(Json::Object(_)) => get(get(b, "image_url").unwrap(), "url").cloned().unwrap_or_else(|| s("")),
            _ => s(""),
        };
        rec.push(("url", url));
    } else {
        for key in ["source", "url", "data"] {
            if let Some(v) = get(b, key) {
                rec.push((key, v.clone()));
            }
        }
    }
    let ok = ["source", "url", "data"].iter().any(|k| rec.iter().any(|(rk, v)| rk == k && super::pyjson::truthy(v)));
    if !ok {
        return Err(ChatError::value("Image block does not contain a valid source"));
    }
    Ok(obj(rec))
}

fn process_image_blocks(blocks: &[Json]) -> Result<(Vec<Json>, Vec<Json>), ChatError> {
    let mut new_blocks = Vec::with_capacity(blocks.len());
    let mut images = Vec::new();
    for block in blocks {
        if !matches!(block, Json::Object(_)) {
            new_blocks.push(block.clone());
            continue;
        }
        if is_image_block(block) {
            new_blocks.push(obj(vec![("type", s("text")), ("text", s(IMAGE_PLACEHOLDER))]));
            images.push(extract_image(block)?);
        } else if jx::get_str(block, "type") == Some("tool_result") && matches!(get(block, "content"), Some(Json::Array(_))) {
            let inner = match get(block, "content") {
                Some(Json::Array(a)) => a.clone(),
                _ => unreachable!(),
            };
            let (nb, nested) = process_image_blocks(&inner)?;
            let mut b2 = block.clone();
            set(&mut b2, "content", Json::Array(nb));
            new_blocks.push(b2);
            images.extend(nested);
        } else if jx::get_str(block, "type") == Some("text") {
            if let Some(Json::Str(t)) = get(block, "text") {
                if t.contains(IMAGE_PLACEHOLDER) {
                    return Err(ChatError::value(format!(
                        "Text block contains image placeholder '{IMAGE_PLACEHOLDER}': '{}'. Images should be separate content blocks.",
                        t.chars().take(100).collect::<String>()
                    )));
                }
            }
            new_blocks.push(block.clone());
        } else {
            new_blocks.push(block.clone());
        }
    }
    Ok((new_blocks, images))
}

/// `process_image_messages`: normalize image blocks, return their records in prompt order.
pub fn process_image_messages(messages: &[Json]) -> Result<(Vec<Json>, Vec<Json>), ChatError> {
    let mut processed = Vec::with_capacity(messages.len());
    let mut images = Vec::new();
    for msg in messages {
        let mut msg = msg.clone();
        validate_no_image_sp_tokens(&msg)?;
        if matches!(get(&msg, "content"), Some(Json::Array(_))) && !has(&msg, "content_blocks") {
            let c = remove(&mut msg, "content").unwrap();
            set(&mut msg, "content_blocks", c);
        }
        if truthy(get(&msg, "content_blocks")) {
            let blocks = match get(&msg, "content_blocks") {
                Some(Json::Array(a)) => a.clone(),
                _ => return Err(ChatError::input("content_blocks must be a list")),
            };
            let (nb, imgs) = process_image_blocks(&blocks)?;
            images.extend(imgs);
            let content_is_str = matches!(get(&msg, "content"), Some(Json::Str(_)));
            if !content_is_str {
                let mut texts: Vec<String> = Vec::new();
                for b in &nb {
                    if matches!(b, Json::Object(_)) && jx::get_str(b, "type") == Some("text") {
                        texts.push(match get(b, "text") {
                            Some(Json::Str(t)) => t.clone(),
                            None => String::new(),
                            Some(_) => return Err(ChatError::input("text block text must be a string")),
                        });
                    }
                }
                set(&mut msg, "content", Json::Str(texts.join("\n\n")));
            }
            set(&mut msg, "content_blocks", Json::Array(nb));
        }
        processed.push(msg);
    }
    Ok((processed, images))
}
