//! The hy_v3 variant: `<tool_call:opensource>` blocks with `<arg_key>/<arg_value>` pairs. Chosen by `is_hy3` (in `mod.rs`)
//! from the first block opener. Map of the directory in `mod.rs`.

use super::*;

/// hy_v3 variant: prose is everything before the wrapper; every `<tool_call:opensource>…</…>`
/// block inside it becomes a call. A truncated block is dropped (same half-call rule as qwen).
pub(super) fn parse_hy3(text: &str, tools: Option<&[Value]>) -> ParsedOutput {
    let mut tool_calls = Vec::new();
    // The wrapper `<tool_calls:opensource>` and the first call block start one char apart —
    // prose ends at whichever comes first, so the wrapper never leaks into the content.
    let first = ["<tool_calls:opensource>", HY_CALL_OPEN].iter()
        .filter_map(|t| text.find(t)).min().unwrap();
    let prose = &text[..first];
    let mut rest = &text[first..];
    while let Some(open) = rest.find(HY_CALL_OPEN) {
        let after = &rest[open + HY_CALL_OPEN.len()..];
        let Some(close) = after.find(HY_CALL_CLOSE) else { break };
        if let Some(tc) = parse_one_hy3(&after[..close], tools) {
            tool_calls.push(tc);
        }
        rest = &after[close + HY_CALL_CLOSE.len()..];
    }
    ParsedOutput { content: prose.trim().to_string(), tool_calls }
}

/// Parse one hy_v3 call body: `NAME<tool_sep:opensource>` then
/// `<arg_key:opensource>K</arg_key:opensource><arg_value:opensource>V</arg_value:opensource>` pairs.
fn parse_one_hy3(body: &str, tools: Option<&[Value]>) -> Option<ToolCall> {
    let sep = body.find(HY_SEP)?;
    let name = body[..sep].trim().to_string();
    if name.is_empty() { return None; }
    let schema = tools.and_then(|ts| param_schema(ts, &name));
    let mut args = serde_json::Map::new();
    let mut rest = &body[sep + HY_SEP.len()..];
    loop {
        let Some(k0) = rest.find(HY_AK) else { break };
        let a = &rest[k0 + HY_AK.len()..];
        let Some(k1) = a.find(HY_AK_END) else { break };
        let key = a[..k1].trim().to_string();
        let v = &a[k1 + HY_AK_END.len()..];
        let Some(v0) = v.find(HY_AV) else { break };
        let v = &v[v0 + HY_AV.len()..];
        let Some(v1) = v.find(HY_AV_END) else { break };
        let raw = v[..v1].trim_matches('\n');
        args.insert(key.clone(), coerce(raw, schema.and_then(|s| s.get(&key))));
        rest = &v[v1 + HY_AV_END.len()..];
    }
    Some(ToolCall {
        id: format!("call_{}", CALL_SEQ.fetch_add(1, Ordering::Relaxed)),
        kind: "function".to_string(),
        function: FunctionCall {
            name,
            arguments: serde_json::to_string(&Value::Object(args)).unwrap_or_else(|_| "{}".into()),
        },
    })
}

/// Locate the function tag inside one `<tool_call>` body.
///
/// The well-formed tag is `<function=NAME>`. In live serving (2026-08-27, 2 of 38 sampled
/// tool requests) the model SOMETIMES DROPS THE `<` and emits `<tool_call>\nfunction=NAME>`.
/// That used to fail `parse_one` entirely, and the two response modes then diverged: streaming
/// (which had already held the block back from the content stream) dropped the text, while
/// non-streaming leaked the raw block into `content`. Inside a `<tool_call>` block a `function=`
/// at the start of a line is unambiguous, so accept the bare form as a repair — the well-formed
/// match always wins when both are present. Returns `(byte offset, tag length)`.
fn find_function_tag(body: &str) -> Option<(usize, usize)> {
    find_function_tag_mode(body, false)
}

/// `wf_only`: the whole `<tool_call>` body contains a well-formed `<function=` somewhere, so a bare `function=NAME>` is never a
/// repair candidate in it (the well-formed form always wins — also AFTER the first call, where a bare-looking line inside a
/// parameter value used to be taken for a second call).
pub(super) fn find_function_tag_mode(body: &str, wf_only: bool) -> Option<(usize, usize)> {
    const WF: &str = "<function=";
    const BARE: &str = "function=";
    if let Some(i) = body.find(WF) {
        scan_ops(i + 1);
        return Some((i, WF.len()));
    }
    scan_ops(body.len());
    if wf_only { return None; }
    let mut from = 0usize;
    while let Some(rel) = body[from..].find(BARE) {
        scan_ops(rel + 1);
        let i = from + rel;
        let boundary_ok = match body[..i].chars().next_back() {
            None | Some('\n') | Some('\r') | Some(' ') | Some('\t') => true,
            _ => false,
        };
        if boundary_ok {
            return Some((i, BARE.len()));
        }
        from = i + BARE.len();
    }
    None
}
