//! Unit tests of the chat surface. The parser cases are the reference stack's own (SGLang `test_inkling2_renderer.py`: interleaved calls,
//! adjacent/empty blocks, tool-error recovery, literal framing spellings in text, malformed calls recovering as text, leading ignored
//! specials, abort), re-expressed as token events. Tests that need the pack's tokenizer skip (loudly) when the file is not on this box.

use super::catalogue::{Catalogue, Reject};
use super::effort;
use super::parse::*;
use super::special::{self as sp, TABLE};
use serde_json::json;

/// `<|name|>` pieces become structural tokens by id, everything else is text.
fn toks(parts: &[&str]) -> Vec<Tok<'static>> {
    parts.iter().map(|p| {
        let leaked: &'static str = Box::leak(p.to_string().into_boxed_str());
        match TABLE.iter().find(|(n, _)| n == p) { Some((_, id)) => Tok::Special(*id), None => Tok::Text(leaked) }
    }).collect()
}
fn message(kind: &str, body: &str, name: Option<&str>, closed: bool) -> Vec<String> {
    let mut v = vec!["<|message_model|>".to_string()];
    if let Some(n) = name { v.push(n.into()); }
    v.push(format!("<|content_{kind}|>")); v.push(body.into());
    if closed { v.push("<|end_message|>".into()); }
    v
}
fn call(body: &str, closed: bool) -> Vec<String> { message("invoke_tool_json", body, Some("lookup"), closed) }
fn flat(blocks: Vec<Vec<String>>) -> Vec<String> { blocks.into_iter().flatten().collect() }
fn refs(v: &[String]) -> Vec<&str> { v.iter().map(|s| s.as_str()).collect() }
fn lookup_cat() -> Catalogue {
    Catalogue::from_openai_tools(&[json!({"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}})])
}
const END: &str = "<|content_model_end_sampling|>";

fn run(parts: &[String], cat: Catalogue) -> (Parsed, Stats) {
    parse_all(toks(&refs(parts)), cat, ParseOptions::default(), Finish::Stop).unwrap()
}
/// every way of cutting each text piece in two (UTF-8 safe) must parse identically (streaming == whole)
fn run_split(parts: &[String], cat: Catalogue, per_char: bool) -> Parsed {
    let mut pieces: Vec<Tok<'static>> = Vec::new();
    for p in parts {
        match TABLE.iter().find(|(n, _)| n == p) {
            Some((_, id)) => pieces.push(Tok::Special(*id)),
            None if per_char => for c in p.chars() { pieces.push(Tok::Text(Box::leak(c.to_string().into_boxed_str()))); },
            None => pieces.push(Tok::Text(Box::leak(p.clone().into_boxed_str()))),
        }
    }
    parse_all(pieces, cat, ParseOptions::default(), Finish::Stop).unwrap().0
}
fn args(c: &ParsedCall) -> serde_json::Value { serde_json::from_str(&c.arguments).unwrap() }

#[test]
fn unicode_reasoning_then_answer() {
    let p = flat(vec![message("thinking", "🦉中文", None, true), message("text", "answer", None, true), vec![END.into()]]);
    let (r, _) = run(&p, Catalogue::default());
    assert_eq!((r.reasoning.as_str(), r.content.as_str(), r.calls.len()), ("🦉中文", "answer", 0));
    assert_eq!(run_split(&p, Catalogue::default(), true), r);
}

#[test]
fn interleaved_calls_keep_order() {
    let p = flat(vec![call(r#"{"name":"lookup","args":{"id":1}}"#, true), message("thinking", "Consider 雪", None, true),
        message("text", "First 中文", None, true), call(r#"{"name":"lookup","args":{"id":2}}"#, true), message("text", "Last", None, true), vec![END.into()]]);
    let (r, st) = run(&p, lookup_cat());
    assert_eq!((r.reasoning.as_str(), r.content.as_str()), ("Consider 雪", "First 中文Last"));
    assert_eq!(r.calls.iter().map(|c| args(c)).collect::<Vec<_>>(), vec![json!({"id":1}), json!({"id":2})]);
    assert_eq!(st.calls, 2);
    assert_eq!(run_split(&p, lookup_cat(), true), r);
}

#[test]
fn adjacent_and_empty_blocks_concatenate() {
    let kinds = [("text", ""), ("text", "One"), ("text", ""), ("text", "二"), ("thinking", ""), ("thinking", "Why"), ("thinking", ""), ("thinking", "雪"), ("text", "")];
    let mut blocks: Vec<Vec<String>> = kinds.iter().map(|(k, b)| message(k, b, None, true)).collect();
    blocks.push(vec![END.into()]);
    let (r, _) = run(&flat(blocks), Catalogue::default());
    assert_eq!((r.reasoning.as_str(), r.content.as_str()), ("Why雪", "One二"));
}

#[test]
fn tool_error_and_unclosed_payload_recover_as_text() {
    let p = flat(vec![call(r#"{"name":"lookup","args":"#, true), message("tool_error", "Failed", None, true),
        call(r#"{"name":"lookup","args":{}}"#, true), message("text", "Recovered", None, true), vec![END.into()]]);
    let (r, st) = run(&p, lookup_cat());
    assert_eq!(r.content, r#"{"name":"lookup","args":FailedRecovered"#);
    assert_eq!(r.calls.len(), 1);
    assert_eq!(st.rejected, vec![Reject::BadJson]);
}

#[test]
fn literal_framing_spellings_in_text_stay_text() {
    // ordinary tokens that SPELL the control tokens: never framing (control tokens are told by id)
    let lit = r#"<|end_message|><|content_invoke_tool_json|> 中文 "quote" \ tail"#;
    let call_body = serde_json::to_string(&json!({"name":"lookup","args":{"key": lit}})).unwrap();
    // the spelled markers are TEXT: build the token list by hand so they are not looked up as ids
    let mut t: Vec<Tok<'_>> = Vec::new();
    let s = |id| Tok::Special(id);
    let (think, answer) = (format!("Consider {lit}"), format!("Literal answer: {lit}"));
    t.extend([s(sp::MESSAGE_MODEL), s(sp::CONTENT_THINKING), Tok::Text(&think), s(sp::END_MESSAGE)]);
    t.extend([s(sp::MESSAGE_MODEL), s(sp::CONTENT_TEXT), Tok::Text(&answer), s(sp::END_MESSAGE)]);
    t.extend([s(sp::MESSAGE_MODEL), Tok::Text("lookup"), s(sp::CONTENT_INVOKE_TOOL_JSON), Tok::Text(&call_body), s(sp::END_MESSAGE), s(sp::EOS)]);
    let (r, _) = parse_all(t, lookup_cat(), ParseOptions::default(), Finish::Stop).unwrap();
    assert_eq!(r.reasoning, format!("Consider {lit}"));
    assert_eq!(r.content, format!("Literal answer: {lit}"));
    assert_eq!(args(&r.calls[0]), json!({"key": lit}));
}

#[test]
fn malformed_calls_recover_as_text_and_later_calls_survive() {
    let bad = [r#"{"name":"lookup","args":{"key":"<|end_message|>"}"#, r#"{"name":"lookup","args":[]}"#, r#"{"name":7,"args":{}}"#, r#""<|content_invoke_tool_json|>""#];
    for body in bad {
        let p = flat(vec![call(body, true), call(r#"{"name":"lookup","args":{"id":1}}"#, true), call(r#"{"name":"lookup","args":{"id":2}}"#, false)]);
        let (r, st) = run(&p, lookup_cat());
        assert_eq!(r.content, body, "{body}");
        assert_eq!(r.calls.iter().map(args).collect::<Vec<_>>(), vec![json!({"id":1}), json!({"id":2})], "{body}");
        assert_eq!(st.rejected.len(), 1);
        for per_char in [false, true] { assert_eq!(run_split(&p, lookup_cat(), per_char), r); }
    }
}

#[test]
fn catalogue_checks_never_invent_a_call() {
    let tools = [json!({"type":"function","function":{"name":"weather","parameters":{"type":"object",
        "properties":{"city":{"type":"string"},"n":{"type":"integer"},"u":{"type":"string","enum":["c","f"]}},"required":["city"]}}})];
    let cat = Catalogue::from_openai_tools(&tools);
    let c = |h: &str, b: &str| cat.check_payload(h, b);
    assert!(c("weather", r#"{"name":"weather","args":{"city":"SF","n":3,"u":"c"}}"#).is_ok());
    assert_eq!(c("weather", r#"{"name":"weather","args":{"city":"SF"}}"#).unwrap().1, r#"{"city":"SF"}"#);
    assert_eq!(c("", r#"{"name":"weather","args":{"city":"SF"}}"#).unwrap().0, "weather");
    assert_eq!(c("sh", r#"{"name":"sh","args":{}}"#), Err(Reject::UnknownTool));
    assert_eq!(c("weather", r#"{"name":"weather","args":{"n":1}}"#), Err(Reject::MissingRequired));
    assert_eq!(c("weather", r#"{"name":"weather","args":{"city":"SF","n":"3"}}"#), Err(Reject::TypeMismatch));
    assert_eq!(c("weather", r#"{"name":"weather","args":{"city":"SF","u":"k"}}"#), Err(Reject::NotInEnum));
    assert_eq!(c("weather", r#"{"name":"weather","args":{"city":"SF","city":"LA"}}"#), Err(Reject::DuplicateParam));
    assert_eq!(c("weather", r#"{"name":"weather","args":{"x":{"a":1,"a":2},"city":"SF"}}"#), Err(Reject::DuplicateParam));
    assert_eq!(c("other", r#"{"name":"weather","args":{"city":"SF"}}"#), Err(Reject::NameMismatch));
    assert_eq!(c("", r#"{"name":"","args":{}}"#), Err(Reject::EmptyName));
    assert_eq!(c("", r#"{"name":"weather"}"#), Err(Reject::ArgsNotObject));
    assert_eq!(Catalogue::default().check_payload("", r#"{"name":"weather","args":{}}"#), Err(Reject::NoCatalogue));
    assert_eq!(cat.only("none-such").check_payload("", r#"{"name":"weather","args":{"city":"a"}}"#), Err(Reject::NoCatalogue));
    // non-ASCII stays as written
    assert_eq!(c("weather", "{\"name\":\"weather\",\"args\":{\"city\":\"Zürich\"}}").unwrap().1, "{\"city\":\"Zürich\"}");
}

#[test]
fn a_rejected_call_stays_visible_and_no_catalogue_means_no_calls() {
    let p = flat(vec![call(r#"{"name":"rm","args":{"path":"/"}}"#, true), vec![END.into()]]);
    let (r, st) = run(&p, lookup_cat());
    assert!(r.calls.is_empty());
    assert_eq!(r.content, r#"{"name":"rm","args":{"path":"/"}}"#);
    assert_eq!(st.rejected, vec![Reject::UnknownTool]);
    let (r2, st2) = run(&flat(vec![call(r#"{"name":"lookup","args":{}}"#, true)]), Catalogue::default());
    assert!(r2.calls.is_empty());
    assert_eq!(st2.rejected, vec![Reject::NoCatalogue]);
}

#[test]
fn parallel_tool_calls_false_keeps_the_first_only() {
    let p = flat(vec![call(r#"{"name":"lookup","args":{"id":1}}"#, true), call(r#"{"name":"lookup","args":{"id":2}}"#, true)]);
    let (r, st) = parse_all(toks(&refs(&p)), lookup_cat(), ParseOptions { max_calls: Some(1), ..Default::default() }, Finish::Stop).unwrap();
    assert_eq!(r.calls.len(), 1);
    assert_eq!(st.dropped_over_limit, 1);
}

#[test]
fn leading_ignored_special_and_foreign_authors_never_stream() {
    for prefix in [vec!["<|begin_of_text|>"], vec!["<|begin_of_text|>", "noise"]] {
        for (role, shown) in [("model", "echoed text"), ("user", ""), ("system", ""), ("tool", "")] {
            let mut p: Vec<&str> = prefix.clone();
            let h = format!("<|message_{role}|>");
            p.extend([h.as_str(), "<|content_text|>", "echoed text", "<|end_message|>"]);
            let (r, _) = parse_all(toks(&p), Catalogue::default(), ParseOptions::default(), Finish::Stop).unwrap();
            assert_eq!(r.content, shown, "{prefix:?} {role}");
        }
    }
    // an <|end_message|> before any header closes nothing; the text after it is still pre-header noise
    let (r, _) = parse_all(toks(&["<|begin_of_text|>", "noise", "<|end_message|>", "echoed text"]), Catalogue::default(), ParseOptions::default(), Finish::Stop).unwrap();
    assert_eq!(r.content, "");
}

#[test]
fn headerless_output_is_model_text() {
    let (r, st) = parse_all(toks(&["Hello ", "there", "<|content_model_end_sampling|>"]), Catalogue::default(), ParseOptions::default(), Finish::Stop).unwrap();
    assert_eq!(r.content, "Hello there");
    assert_eq!(st.headerless_text, 1);
    // a special inside headerless text is framing
    let (r, _) = parse_all(toks(&["Hi", "<|end_message|>", "<|message_model|>", "<|content_text|>", " more", "<|end_message|>"]), Catalogue::default(), ParseOptions::default(), Finish::Stop).unwrap();
    assert_eq!(r.content, "Hi more");
}

#[test]
fn reasoning_can_be_folded_into_content() {
    let p = flat(vec![message("thinking", "t", None, true), message("text", "a", None, true)]);
    let (r, _) = parse_all(toks(&refs(&p)), Catalogue::default(), ParseOptions { separate_reasoning: false, ..Default::default() }, Finish::Stop).unwrap();
    assert_eq!((r.reasoning.as_str(), r.content.as_str()), ("", "ta"));
}

#[test]
fn text_invocation_is_text_never_a_call() {
    let p = flat(vec![message("invoke_tool_text", "raw words", None, true)]);
    let (r, st) = run(&p, lookup_cat());
    assert_eq!((r.content.as_str(), r.calls.len(), st.text_invocations), ("raw words", 0, 1));
}

#[test]
fn a_reply_cut_by_the_budget_or_aborted_never_completes_a_call() {
    let open = flat(vec![message("thinking", "plan", None, true), call(r#"{"name":"lookup","args":{"id":1}}"#, false)]);
    // Stop (the stream ended on its own): the complete payload is the call
    let (r, _) = parse_all(toks(&refs(&open)), lookup_cat(), ParseOptions::default(), Finish::Stop).unwrap();
    assert_eq!(r.calls.len(), 1);
    // Length: the budget cut the reply; the open payload is shown as text, not run
    let (r, st) = parse_all(toks(&refs(&open)), lookup_cat(), ParseOptions::default(), Finish::Length).unwrap();
    assert!(r.calls.is_empty());
    assert_eq!(r.content, r#"{"name":"lookup","args":{"id":1}}"#);
    assert_eq!(st.cut_calls, 1);
    // abort: nothing more is emitted
    let mut p = ReplyParser::new(lookup_cat(), ParseOptions::default());
    for t in toks(&refs(&open)) { p.feed(t).unwrap(); }
    p.abort();
    assert!(p.finish(Finish::Stop).is_empty());
}

#[test]
fn reserved_tokens_are_an_engine_error() {
    let mut p = ReplyParser::new(Catalogue::default(), ParseOptions::default());
    p.feed(Tok::Special(sp::MESSAGE_MODEL)).unwrap();
    assert_eq!(p.feed(Tok::Special(sp::CONTENT_AUDIO_INPUT)).unwrap_err(), ParseError::Reserved(200020));
}

#[test]
fn eos_ends_the_reply_and_closes_the_open_block() {
    let (r, _) = parse_all(toks(&["<|message_model|>", "<|content_text|>", "ab", "<|content_model_end_sampling|>", "<|message_model|>", "<|content_text|>", "never"]),
                           Catalogue::default(), ParseOptions::default(), Finish::Stop).unwrap();
    assert_eq!(r.content, "ab");
}

#[test]
fn streaming_emits_text_as_it_arrives_and_holds_the_call_until_closed() {
    let mut p = ReplyParser::new(lookup_cat(), ParseOptions::default());
    let mut seen = Vec::new();
    let all = flat(vec![message("thinking", "a", None, false), vec!["b".into(), "<|end_message|>".into()], call(r#"{"name":"lookup","args":{}}"#, false), vec!["<|end_message|>".into()]]);
    for t in toks(&refs(&all)) { seen.push(p.feed(t).unwrap()); }
    // events arrive per piece: 'a' and 'b' each as reasoning, the call only at its <|end_message|>
    assert_eq!(seen.iter().flatten().filter(|e| matches!(e, Event::Reasoning(_))).count(), 2);
    let last_call_ix = seen.iter().position(|v| v.iter().any(|e| matches!(e, Event::Call(_)))).unwrap();
    assert_eq!(last_call_ix, seen.len() - 1);
}

// ------------------------------------------------------------------------------------------------ effort
#[test]
fn effort_names_numbers_and_graceful_mapping() {
    let e = effort::resolve(None, None, None).unwrap();
    assert_eq!((e.value, e.level), (None, 0.9));
    let e = effort::resolve(None, Some(&json!("low")), None).unwrap();
    assert_eq!((e.value, e.level), (Some(json!("low")), 0.2));
    let e = effort::resolve(Some(&json!({"reasoning_effort": 0.5})), Some(&json!("low")), None).unwrap();
    assert_eq!((e.value, e.level), (Some(json!(0.5)), 0.5), "kwargs win; a number stays a number");
    assert_eq!(effort::resolve(None, Some(&json!(0.3)), None).unwrap().value, Some(json!(0.3)));
    let e = effort::resolve(None, Some(&json!("off")), None).unwrap();
    assert_eq!((e.value, e.level), (Some(json!("none")), 0.0));
    assert!(e.mapped.is_some());
    let e = effort::resolve(Some(&json!({"enable_thinking": false})), None, None).unwrap();
    assert_eq!((e.value, e.level), (Some(json!("none")), 0.0));
    assert_eq!(effort::resolve(None, Some(&json!("0.5")), None).unwrap().value, Some(json!(0.5)));
    assert_eq!(effort::resolve(None, None, Some("high")).unwrap().level, 0.9);
    assert!(effort::resolve(None, Some(&json!("bogus")), None).is_err());
    assert!(effort::resolve(None, Some(&json!(1.5)), None).is_err());
    assert!(effort::resolve(None, Some(&json!(true)), None).is_err());
    assert!(effort::resolve(None, None, Some("bogus")).is_err());
}

// ------------------------------------------------------------------------------------------------ the pack (skipped where it is not on this box)
const PACK: &str = "/models/Inkling-2-Mini";
const REF: &str = "/models/Inkling-2-Mini/tok_cases.json";

fn pack_tok() -> Option<crate::tokenizer::QwenTokenizer> {
    let p = format!("{PACK}/tokenizer.json");
    if !std::path::Path::new(&p).exists() { eprintln!("SKIP: {p} not on this box"); return None; }
    Some(crate::tokenizer::QwenTokenizer::from_file(&p).unwrap())
}

#[test]
fn special_ids_match_the_pack_tokenizer() {
    let Some(tok) = pack_tok() else { return };
    for (name, id) in TABLE { assert_eq!(tok.encode(name, false).unwrap(), vec![id], "{name}"); }
    assert_eq!(tok.encode("<|unused_200054|>", false).unwrap(), vec![sp::IMAGE]);
    // the checkpoint keeps the native audio ids as unused tokens
    assert_eq!(tok.encode("<|unused_200020|>", false).unwrap(), vec![sp::CONTENT_AUDIO_INPUT]);
    assert_eq!(tok.encode("<|unused_200043|>", false).unwrap(), vec![sp::AUDIO_END]);
}

#[test]
fn template_renders_the_seven_reference_prompts_through_the_request_path() {
    let Some(tok) = pack_tok() else { return };
    let Ok(raw) = std::fs::read(REF) else { eprintln!("SKIP: {REF} not on this box"); return };
    let cases: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    let mut n = 0;
    for c in cases.as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let mut body = json!({"messages": c["messages"]});
        // the reference messages carry parsed tool-call arguments; the wire form is a JSON string
        for m in body["messages"].as_array_mut().unwrap() {
            if let Some(tcs) = m.get_mut("tool_calls").and_then(|t| t.as_array_mut()) {
                for tc in tcs { if let Some(a) = tc.pointer_mut("/function/arguments") { if !a.is_string() { *a = json!(a.to_string()); } } }
            }
        }
        if let Some(t) = c.get("tools") { body["tools"] = t.clone(); }
        if let Some(e) = c.get("reasoning_effort") { body["reasoning_effort"] = e.clone(); }
        if let Some(k) = c.get("chat_template_kwargs") { body["chat_template_kwargs"] = k.clone(); }
        let r = super::request::render(&body, &Default::default(), &tok).unwrap_or_else(|e| panic!("{name}: {}", e.message));
        assert_eq!(r.prompt_text, c["text"].as_str().unwrap(), "{name}: text");
        let want: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        assert_eq!(r.prompt_ids, want, "{name}: ids");
        if name == "image_user" { assert_eq!((r.n_images, r.image_splices.len()), (1, 1)); }
        n += 1;
    }
    assert_eq!(n, 7);
}

#[test]
fn request_render_handles_effort_number_images_tool_choice_and_errors() {
    let Some(tok) = pack_tok() else { return };
    let d = super::request::Defaults { effort: None, max_seq_len: 4096 };
    let r = super::request::render(&json!({"messages":[{"role":"user","content":"hi"}], "reasoning_effort": 0.5}), &d, &tok).unwrap();
    assert!(r.prompt_text.contains("Thinking effort level: 0.5"), "{}", r.prompt_text);
    let r = super::request::render(&json!({"messages":[{"role":"user","content":"hi"}], "reasoning_effort": "off"}), &d, &tok).unwrap();
    assert!(r.prompt_text.contains("Thinking effort level: 0<|end_message|>"), "{}", r.prompt_text);
    // images: one splice per image part, in order
    let r = super::request::render(&json!({"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA"}},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,AA"}}]}]}), &d, &tok).unwrap();
    assert_eq!((r.n_images, r.image_splices.len()), (2, 2));
    assert!(r.image_splices[0] < r.image_splices[1]);
    assert_eq!(r.prompt_ids[r.image_splices[0]], sp::IMAGE);
    // tool_choice none: the tools are not offered
    let tools = json!([{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]);
    let r = super::request::render(&json!({"messages":[{"role":"user","content":"hi"}], "tools": tools, "tool_choice":"none"}), &d, &tok).unwrap();
    assert!(!r.tools_offered && !r.prompt_text.contains("tool_declare"));
    let r = super::request::render(&json!({"messages":[{"role":"user","content":"hi"}], "tools": tools, "tool_choice":"required"}), &d, &tok).unwrap();
    assert!(r.tools_offered && r.choice_unenforced && r.prompt_text.contains("tool_declare"));
    // errors: 400s that say what to do
    assert_eq!(super::request::render(&json!({"messages":[]}), &d, &tok).err().unwrap().status, 400);
    assert!(super::request::render(&json!({"messages":[{"role":"user","content":"hi"}], "reasoning_effort": "bogus"}), &d, &tok).err().unwrap().message.contains("reasoning_effort"));
    let small = super::request::Defaults { effort: None, max_seq_len: 8 };
    assert_eq!(super::request::render(&json!({"messages":[{"role":"user","content":"hi there friend"}]}), &small, &tok).err().unwrap().code, Some("context_length_exceeded"));
    // an unknown role is the template's own error, as a 400
    assert!(super::request::render(&json!({"messages":[{"role":"wizard","content":"hi"}]}), &d, &tok).is_err());
    // a developer message is a system message
    let r = super::request::render(&json!({"messages":[{"role":"developer","content":"be terse"},{"role":"user","content":"hi"}]}), &d, &tok).unwrap();
    assert!(r.prompt_text.starts_with("<|message_system|><|content_text|>be terse<|end_message|>"));
}
