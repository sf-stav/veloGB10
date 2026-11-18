//! XML-form calls: block body to `ToolCall`s (`parse_block`, `parse_xml_calls*`) and `coerce` of each value to the
//! type its schema declares. Map of the directory in `mod.rs`.

use super::*;

/// Parse one `<tool_call>…</tool_call>` body into ZERO OR MORE calls.
///
/// Two shapes are supported; the body decides:
///  * a JSON object (the Froggeric `tool_call_format=json` emission) — one call, with the
///    tolerant salvage rules in `parse_json_call`;
///  * the XML form — EVERY `<function=NAME>` in the block becomes its own call. Before Phase 2
///    only the FIRST function tag produced a call and every later function's `<parameter=…>`
///    pairs were merged into it, same-key values overwritten (RENDER_AUDIT.md 9.2).
/// F8 recovery read (`--tool-unclosed complete`, finish stop): the body of a wrapper that never
/// closed. XML elements whose own `</function>` IS there come back (vLLM's qwen3 parser ends a call
/// at FUNC_END; the wrapper is only preamble); a trailing element cut inside an argument yields
/// nothing (its value never closed and its `</function>` never came); a JSON body stays dropped
/// (vLLM's XML parser makes no call of it).
pub(super) fn parse_unclosed_body(body: &str, tools: Option<&[Value]>, idx: usize, rules: &Rules, ctx: &cost::Ctx) -> Vec<ToolCall> {
    let _ = idx; // ids are minted by the caller's numbering either way
    if body.trim_start().starts_with('{') { return Vec::new(); }
    parse_xml_calls_ctx(body, tools, rules.pairing, None, &rules.weights, ctx, rules.trim, true)
}

pub(super) fn parse_block(body: &str, tools: Option<&[Value]>, idx: usize, rules: &Rules, ctx: &cost::Ctx) -> Vec<ToolCall> {
    let trimmed = body.trim_start();
    if trimmed.starts_with('{') {
        if let Some(tc) = parse_json_call(trimmed, tools, idx) {
            return vec![tc];
        }
        // Unparseable JSON body: fall through to the XML path rather than dropping the call.
    }
    parse_xml_calls_ctx(body, tools, rules.pairing, None, &rules.weights, ctx, rules.trim, false)
}

/// The XML form: one call per `<function=NAME>` tag, each owning the `<parameter=…>` pairs up
/// to the next tag. Also accepts a JSON-object body inside the function tag (the Froggeric
/// template's string-arguments history form) when no `<parameter=` pair is present.
pub(super) fn parse_xml_calls(body: &str, tools: Option<&[Value]>, pairing: QuotePairing) -> Vec<ToolCall> {
    parse_xml_calls_w(body, tools, pairing, None, &cost::Weights::SHIPPED)
}

/// `parse_xml_calls`, optionally with the close of every value given (`forced`: value start offset
/// -> close offset, as `cost::Plan::closes`): how the tests and the cost breakdowns read a block
/// under a segmentation of their own choosing. A value the map does not hold is ended by the
/// v0.7.3 rule.
pub(crate) fn parse_xml_calls_plan(body: &str, tools: Option<&[Value]>, pairing: QuotePairing,
                                   forced: Option<&std::collections::HashMap<usize, usize>>) -> Vec<ToolCall> {
    parse_xml_calls_w(body, tools, pairing, forced, &cost::Weights::SHIPPED)
}

/// `parse_xml_calls_plan` with the weights of the cost model given.
pub(crate) fn parse_xml_calls_w(body: &str, tools: Option<&[Value]>, pairing: QuotePairing,
                                forced: Option<&std::collections::HashMap<usize, usize>>, weights: &cost::Weights) -> Vec<ToolCall> {
    parse_xml_calls_ctx(body, tools, pairing, forced, weights, &cost::Ctx::new(tools), ValueTrim::All, false)
}

/// `parse_xml_calls_w` inside a response: the cost model's context (catalogue lookup, work budget)
/// is the response's, shared by its blocks.
pub(crate) fn parse_xml_calls_ctx(body: &str, tools: Option<&[Value]>, pairing: QuotePairing,
                                  forced: Option<&std::collections::HashMap<usize, usize>>, weights: &cost::Weights,
                                  ctx: &cost::Ctx, trim: ValueTrim, require_close: bool) -> Vec<ToolCall> {
    let pairing = if forced.is_some() { QuotePairing::Cost } else { pairing };
    let mut calls = Vec::new();
    let mut calls_skipped = 0usize;
    let mut pair = PairState::new(body.len());
    // A body with any well-formed tag takes no bare repairs at all (see find_function_tag_mode).
    let wf_only = body.contains("<function=");
    // `--tool-quote-pairing cost`: the cheapest reading of the whole block fixes where each value ends
    // (a value the plan does not know is ended by the v0.7.3 rule).
    let own_plan;
    let plan: Option<&std::collections::HashMap<usize, usize>> = match forced {
        Some(f) => Some(f),
        None if pairing == QuotePairing::Cost => { own_plan = cost::closes_for(body, ctx, wf_only, weights); own_plan.as_ref() }
        None => None,
    };
    // The scan is SEQUENTIAL: a call owns its `<parameter=…>…</parameter>` spans, and a function tag is only a tag when it sits
    // OUTSIDE every value span consumed so far. A parameter value that merely contains the text `function=x>` or `<function=x>`
    // (a coding agent writing a file about this very format) stays payload and never becomes a second call.
    let mut from = 0usize;
    // (S-B9-TOOLPARSE3 T8a) two searches that ran to the end of the block from every parameter made a
    // block of thousands of parameters quadratic: 1.6 MB took 2.3 s and 26.7 s on the two hostile
    // shapes. Both results are position-independent, so they are cached; the answers are identical.
    let mut dead_from = usize::MAX;
    let mut tags = TagCache::new(wf_only);
    let mut params = NeedleCache::new("<parameter=");
    while let Some((i, tlen)) = tags.next(body, from) {
        let start = from + i;
        let name_from = start + tlen;
        let Some(gt) = body[name_from..].find('>') else { from = name_from; continue };
        let name = body[name_from..name_from + gt].trim().to_string();
        let after_name = name_from + gt + 1;
        if name.is_empty() { from = after_name; continue; }

        let schema = tools.and_then(|ts| param_schema(ts, &name));
        let mut args = serde_json::Map::new();
        // `cursor` is always outside a value span. The next `<parameter=` belongs to THIS call unless the next function tag comes
        // first (then it is that call's); the end of this call's text is the next tag after the last consumed span.
        let mut cursor = after_name;
        loop {
            let next_tag = tags.next(body, cursor).map(|(ti, _)| cursor + ti);
            let next_param = params.next_abs(body, cursor);
            let Some(popen) = next_param else { break };
            if next_tag.map_or(false, |t| t < popen) { break; }
            let a = &body[popen + "<parameter=".len()..];
            let Some(gt2) = a.find('>') else { break };
            let key = a[..gt2].trim().to_string();
            let vstart = &a[gt2 + 1..];
            // v0.7.3: the STRUCTURAL close, not merely the first close tag — a value that quotes
            // the tool format keeps its embedded `</parameter>` as content. `--tool-quote-pairing
            // on` pairs the tags of quoted calls before that rule is asked.
            let pclose = match pairing {
                QuotePairing::Off => param_end_v073_cached(vstart, body.len() - vstart.len(), &mut dead_from),
                QuotePairing::On => param_end_paired(vstart, &mut pair, &ValueCx { tools, name: &name, key: &key, given: &args, abs: body.len() - vstart.len() }),
                QuotePairing::Cost => {
                    let vs = body.len() - vstart.len();
                    match plan.and_then(|p| p.get(&vs)) {
                        Some(&c) => Some(c - vs),
                        None => param_end_v073_cached(vstart, vs, &mut dead_from),
                    }
                }
            };
            let Some(pclose) = pclose else { break };
            // The template puts a newline after `>` and before `</parameter>`; they are delimiters,
            // not part of the value. Trim only those, so interior whitespace of a multi-line value
            // survives.
            let raw = match trim {
                ValueTrim::All => vstart[..pclose].trim_matches('\n'),
                ValueTrim::One => trim_wrapping_newlines(&vstart[..pclose]),
            };
            args.insert(key.clone(), coerce(raw, schema.and_then(|s| s.get(&key))));
            cursor = body.len() - vstart.len() + pclose + "</parameter>".len();
        }
        // This call's text runs to the next function tag after the last consumed value span (or the end of the block).
        let end = tags.next(body, cursor).map(|(ti, _)| cursor + ti).unwrap_or(body.len());
        let after = &body[start + tlen..end];
        if args.is_empty() {
            let jb = after[gt + 1..].trim();
            let jb = jb.strip_suffix("</function>").unwrap_or(jb).trim();
            // A body that LOOKS like JSON is a JSON body: the Froggeric template's
            // `raw_tool_args_history` branch renders the model's own previous calls as a JSON
            // object inside the function tag, and the model imitates that while closing with the
            // `</parameter>` tag it also sees in the XML example. So the emitted shape is
            // `{…json…}\n</parameter>` — ONE stray trailing close tag. Strip it (and the
            // whitespace between) before parsing. Before this, every such body fell through to
            // `args` = {} and the harness invoked a real tool with no arguments: 6/6 lost
            // (phase3/EMPTY_ARGS_EVIDENCE.md, PHASE3_REPORT.md 4.3), while the 5/5 clean-JSON
            // bodies parsed. A non-empty body that still fails to yield arguments is ALARMED,
            // never silently dropped.
            let jsonish = jb.starts_with('{') || jb.starts_with('[');
            if jsonish {
                let jb = strip_stray_param_close(jb);
                // Phase-5 B1: `json_object_lenient` = strict parse, then the unescaped-quote
                // repair. An array, an unrepairable body or a truncated one still alarms.
                match json_object_lenient(jb) {
                    Some(o) => for (k, v) in o { args.insert(k, v); },
                    None => if !shadow_parse() { eprintln!("[tool-args-alarm] {jb}") },
                }
            } else if !jb.is_empty() {
                if !shadow_parse() { eprintln!("[tool-args-alarm] {jb}"); }
            }
        }
        // F8 (`require_close`, the recovery read of a wrapper that never closed): an element whose
        // own `</function>` is absent is the one the model was cut inside — vLLM's parser never
        // reaches FUNC_END for it, so it yields nothing here too (the alarm above still fires: its
        // body failed to yield arguments).
        if require_close && !after.contains("</function>") { calls_skipped += 1; from = end; continue; }
        calls.push(ToolCall {
            id: mint_id(),
            kind: "function".to_string(),
            function: crate::tokenizer::FunctionCall {
                name,
                arguments: serde_json::to_string(&Value::Object(args)).unwrap_or_else(|_| "{}".into()),
            },
        });
        from = end;
    }
    calls
}

/// Coerce one text value to the type its schema declares. Unknown/absent schema, or a value that does
/// not parse as the declared type, stays a string — never guess a type the schema did not ask for.
pub(super) fn coerce(raw: &str, schema: Option<&Value>) -> Value {
    let ty = schema.and_then(|s| s.get("type")).and_then(Value::as_str);
    match ty {
        Some("integer") => raw.trim().parse::<i64>().map(Value::from).unwrap_or_else(|_| Value::String(raw.to_string())),
        Some("number")  => raw.trim().parse::<f64>().map(Value::from).unwrap_or_else(|_| Value::String(raw.to_string())),
        Some("boolean") => match raw.trim().to_ascii_lowercase().as_str() {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            _ => Value::String(raw.to_string()),
        },
        // The template serialises objects/arrays with `| tojson`, so they come back as JSON text.
        Some("object") | Some("array") =>
            serde_json::from_str(raw.trim()).unwrap_or_else(|_| Value::String(raw.to_string())),
        _ => Value::String(raw.to_string()),
    }
}
