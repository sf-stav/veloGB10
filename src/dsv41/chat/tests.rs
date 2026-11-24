//! Tests of the V4.1 chat layer: the reference corpus (`tests/fixtures/dsv41_enc/corpus.json`, made by
//! `tools/dsv41_ref/encoding_corpus.py` from the release's own `encoding.py`), the release's own
//! assertions, the DSML parser (adversarial cases and byte-split fuzz) and the request rules.

use std::path::{Path, PathBuf};

use crate::dsv4_chat::{parse_json, Json};

use super::catalogue::{Catalogue, Reject};
use super::error::ChatErrorKind;
use super::jx::{get, get_str};
use super::pyjson::canon;
use super::render::*;
use super::request::*;
use super::special;
use super::stream::*;

const MODEL_DIR: &str = "/models/DeepSeek-V4.1-Flash-1017";

fn j(s: &str) -> Json {
    parse_json(s).unwrap_or_else(|e| panic!("bad test json {s}: {}", e.1))
}

fn corpus() -> Option<Json> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv41_enc/corpus.json");
    let text = std::fs::read_to_string(p).ok()?;
    Some(parse_json(&text).expect("corpus parses"))
}

fn tokenizer() -> Option<tokenizers::Tokenizer> {
    let p = PathBuf::from(MODEL_DIR).join("tokenizer.json");
    if !p.exists() {
        return None;
    }
    tokenizers::Tokenizer::from_file(p).ok()
}

fn kind_name(k: ChatErrorKind) -> &'static str {
    match k {
        ChatErrorKind::Assertion => "assertion",
        ChatErrorKind::Value => "value",
        ChatErrorKind::NotImplemented => "not_implemented",
        ChatErrorKind::Input => "input",
    }
}

fn arr<'a>(v: Option<&'a Json>) -> Vec<Json> {
    match v {
        Some(Json::Array(a)) => a.clone(),
        _ => Vec::new(),
    }
}

#[test]
fn corpus_encoder_is_byte_exact() {
    let Some(c) = corpus() else {
        eprintln!("SKIP: corpus fixture missing");
        return;
    };
    let tok = tokenizer();
    if tok.is_none() {
        eprintln!("NOTE: tokenizer missing, comparing strings only");
    }
    let cases = arr(get(&c, "cases"));
    let (mut n, mut errs, mut ids_checked) = (0, 0, 0);
    for case in cases.iter().filter(|c| get_str(c, "kind") == Some("encode")) {
        let id = get(case, "id").and_then(|x| x.as_u64()).unwrap();
        let args = get(case, "args").unwrap();
        let messages = match get_str(case, "raw_messages_json") {
            Some(raw) => arr(Some(&j(raw))),
            None => arr(get(args, "messages")),
        };
        let mut opts = EncodeOptions::new(ThinkingMode::parse(get_str(args, "thinking_mode").unwrap()).unwrap_or(ThinkingMode::Chat));
        // an invalid thinking_mode is itself a case
        let bad_mode = ThinkingMode::parse(get_str(args, "thinking_mode").unwrap()).is_err();
        opts.context = arr(get(args, "context"));
        opts.drop_thinking = get(args, "drop_thinking").and_then(|x| x.as_bool()).unwrap_or(true);
        opts.add_default_bos_token = get(args, "add_bos").and_then(|x| x.as_bool()).unwrap_or(true);
        opts.reasoning_effort = get(args, "reasoning_effort").cloned().filter(|e| !e.is_null());
        let res = if bad_mode {
            Err(super::error::ChatError::assertion("Invalid thinking_mode"))
        } else {
            encode_messages(&messages, &opts)
        };
        n += 1;
        if let Some(exp) = get_str(case, "expected_error") {
            errs += 1;
            let e = res.expect_err(&format!("case {id} ({}) must fail with {exp}", get_str(case, "tag").unwrap()));
            assert_eq!(kind_name(e.kind), exp, "case {id} ({}): {e}", get_str(case, "tag").unwrap());
            continue;
        }
        let out = res.unwrap_or_else(|e| panic!("case {id} ({}) failed: {e}", get_str(case, "tag").unwrap()));
        let want = get_str(case, "expected_prompt").unwrap();
        if out.prompt != want {
            let i = out.prompt.bytes().zip(want.bytes()).position(|(a, b)| a != b).unwrap_or(out.prompt.len().min(want.len()));
            let lo = (0..=i.saturating_sub(60)).rev().find(|k| want.is_char_boundary(*k)).unwrap_or(0);
            panic!(
                "case {id} ({}) prompt differs at byte {i}\n  got : {:?}\n  want: {:?}",
                get_str(case, "tag").unwrap(),
                out.prompt.get(lo..(i + 80).min(out.prompt.len())).unwrap_or("?"),
                want.get(lo..(i + 80).min(want.len())).unwrap_or("?")
            );
        }
        let want_images = arr(get(case, "expected_images"));
        assert_eq!(out.images.iter().map(canon).collect::<Vec<_>>(), want_images.iter().map(canon).collect::<Vec<_>>(), "case {id} images");
        if let Some(t) = &tok {
            let ids: Vec<u32> = t.encode(out.prompt.as_str(), false).unwrap().get_ids().to_vec();
            let want_ids: Vec<u32> = arr(get(case, "expected_ids")).iter().map(|x| x.as_u64().unwrap() as u32).collect();
            assert_eq!(ids, want_ids, "case {id} ({}) token ids", get_str(case, "tag").unwrap());
            ids_checked += 1;
        }
    }
    eprintln!("corpus encode: {n} cases ({errs} expected errors), {ids_checked} token-id comparisons");
    assert!(n >= 200, "corpus must hold at least 200 conversations, has {n}");
}

#[test]
fn corpus_parser_matches_reference_on_reference_output() {
    let Some(c) = corpus() else { return };
    let mut n = 0;
    for case in arr(get(&c, "cases")).iter().filter(|c| get_str(c, "kind") == Some("parse")) {
        let Some(exp) = get(case, "expected") else { continue };
        let thinking = get_str(case, "thinking_mode") == Some("thinking");
        let mut tools = get(case, "tools").cloned().unwrap_or(Json::Array(vec![]));
        // the reference parser does not check required parameters; strip them so the format is what is compared
        if let Json::Array(ts) = &mut tools {
            for t in ts.iter_mut() {
                if let Some(Json::Object(f)) = super::jx::get_mut(t, "function") {
                    if let Some((_, Json::Object(p))) = f.iter_mut().find(|(k, _)| k == "parameters") {
                        p.retain(|(k, _)| k != "required");
                    }
                }
            }
        }
        let cat = if matches!(&tools, Json::Array(a) if a.is_empty()) { Some(Catalogue::allow_any()) } else { Some(Catalogue::from_openai_tools(&tools).unwrap()) };
        let text = get_str(case, "text").unwrap();
        if text.find(EOS_TOKEN).map_or(false, |i| i + EOS_TOKEN.len() < text.len()) {
            continue; // an end-of-sequence token inside the text ends generation in serving: not a parity case
        }
        let p = parse_completion(text, ParseOptions::new(thinking), cat);
        assert_eq!(Some(p.content.as_str()), get_str(exp, "content"), "case {} content", get(case, "id").unwrap().as_u64().unwrap());
        assert_eq!(Some(p.reasoning_content.as_str()), get_str(exp, "reasoning_content"), "reasoning");
        let want: Vec<(Option<String>, String, String)> = arr(get(exp, "tool_calls"))
            .iter()
            .map(|t| (get_str(t, "namespace").map(str::to_string), get_str(t, "name").unwrap().to_string(), get_str(t, "arguments").unwrap().to_string()))
            .collect();
        let got: Vec<_> = p.calls.iter().map(|t| (t.namespace.clone(), t.name.clone(), t.arguments.clone())).collect();
        assert_eq!(got, want, "case {} calls", get(case, "id").unwrap().as_u64().unwrap());
        assert!(p.stats.rejected.is_empty(), "{:?}", p.stats.rejected);
        n += 1;
    }
    assert!(n >= 30, "parse cases {n}");
}

// ---- the release's own assertions (test_encoding.py), ported ----

const TOOL_CALL_OUTPUT: &str = "  reason  </think>summary\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">value</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"limit\" string=\"false\">2</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>";

fn tool_call_messages() -> Vec<Json> {
    arr(Some(&j(r#"[{"role":"user","content":"question"},{"role":"assistant","reasoning_content":"  reason  ","content":"summary","tool_calls":[{"type":"function","function":{"name":"lookup","arguments":"{\"query\":\"value\",\"limit\":2}"}}]}]"#)))
}

fn lookup_tools() -> Json {
    j(r#"[{"type":"function","function":{"name":"lookup","description":"Look up a value","parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"}}}}}]"#)
}

#[test]
fn release_assertions_render_and_roundtrip() {
    let msgs = tool_call_messages();
    assert_eq!(render_message(1, &msgs, ThinkingMode::Thinking, true, None).unwrap(), TOOL_CALL_OUTPUT);
    let cat = Catalogue::from_openai_tools(&lookup_tools()).unwrap();
    let p = parse_completion(TOOL_CALL_OUTPUT, ParseOptions::new(true), Some(cat));
    assert_eq!(p.reasoning_content, "  reason  ");
    assert_eq!(p.content, "summary");
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.calls[0].name, "lookup");
    assert_eq!(j(&p.calls[0].arguments), j(r#"{"query": "value", "limit": 2}"#));
    // re-encoding the parsed message reproduces the completion
    let parsed_msg = j(&format!(
        r#"{{"role":"assistant","content":"summary","reasoning_content":"  reason  ","tool_calls":[{{"type":"function","function":{{"name":"lookup","arguments":{}}}}}]}}"#,
        crate::dsv4_chat::dumps(&Json::Str(p.calls[0].arguments.clone()))
    ));
    let mut o = EncodeOptions::new(ThinkingMode::Thinking);
    o.context = msgs[..1].to_vec();
    assert_eq!(encode_messages(&[parsed_msg], &o).unwrap().prompt, TOOL_CALL_OUTPUT);
}

#[test]
fn release_assertion_unspaced_v4_dsml_is_not_a_call() {
    let v4 = TOOL_CALL_OUTPUT.replace("｜DSML｜ calls", "｜DSML｜tool_calls").replace("｜DSML｜ invoke", "｜DSML｜invoke").replace("｜DSML｜ parameter", "｜DSML｜parameter");
    let cat = Catalogue::from_openai_tools(&lookup_tools()).unwrap();
    let p = parse_completion(&v4, ParseOptions::new(true), Some(cat));
    assert!(p.calls.is_empty());
    assert!(p.content.contains("tool_calls"), "kept as text: {:?}", p.content);
}

#[test]
fn effort_mapping_and_rejections() {
    for (e, b) in [(None, 75u32), (Some("low"), 50), (Some("high"), 75), (Some("max"), 100)] {
        let ej = e.map(super::effort::named);
        assert_eq!(super::effort::budget(ej.as_ref()).unwrap(), b);
    }
    for n in [1u32, 42, 100] {
        assert_eq!(super::effort::budget(Some(&super::effort::int(n))).unwrap(), n);
    }
    for bad in ["-1", "0", "101", "1.5", "true", "false", "\"medium\"", "[]"] {
        assert!(super::effort::budget(Some(&j(bad))).is_err(), "{bad}");
    }
}

// ---- parser ----

fn offered() -> Catalogue {
    Catalogue::from_openai_tools(&j(r#"[
      {"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"integer"},"flags":{"type":"array","items":{"type":"string"}}},"required":["path"]}}},
      {"type":"function","namespace":"search","function":{"name":"find","parameters":{"type":"object","properties":{"q":{"type":"string"},"k":{"enum":[1,2,3]}}}}},
      {"type":"function","function":{"name":"noargs"}}]"#)).unwrap()
}

fn block(inner: &str) -> String {
    format!("\n\n<｜DSML｜ calls>\n{inner}\n</｜DSML｜ calls>")
}

fn inv(name: &str, params: &[(&str, &str, &str)]) -> String {
    let ps: Vec<String> = params.iter().map(|(k, s, v)| format!("<｜DSML｜ parameter name=\"{k}\" string=\"{s}\">{v}</｜DSML｜ parameter>")).collect();
    format!("<｜DSML｜ invoke name=\"{name}\">\n{}\n</｜DSML｜ invoke>", ps.join("\n"))
}

fn parse(text: &str) -> Parsed {
    parse_completion(text, ParseOptions::new(false), Some(offered()))
}

#[test]
fn a_plain_call() {
    let p = parse(&format!("ok{}", block(&inv("write", &[("path", "true", "/a b"), ("mode", "false", "420"), ("flags", "false", "[\"x\", \"y\"]")]))));
    assert_eq!(p.content, "ok");
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.calls[0].arguments, r#"{"path": "/a b", "mode": 420, "flags": ["x", "y"]}"#);
}

#[test]
fn namespaced_and_parallel_calls() {
    let p = parse(&block(&format!("{}\n{}\n{}", inv("search::find", &[("q", "true", "hi"), ("k", "false", "2")]), inv("noargs", &[]), inv("write", &[("path", "true", "p")]))));
    assert_eq!(p.calls.len(), 3);
    assert_eq!(p.calls[0].namespace.as_deref(), Some("search"));
    assert_eq!(p.calls[0].name, "find");
    assert_eq!(p.calls[1].arguments, "{}");
    let o = p.calls[0].to_openai_json("call_1");
    assert_eq!(get(&o, "namespace").and_then(|x| x.as_str()), Some("search"));
    assert_eq!(p.content, "");
}

#[test]
fn closer_and_opener_inside_a_value_stay_in_the_value() {
    let value = "a </｜DSML｜ invoke> b <｜DSML｜ invoke name=\"evil\"> c </｜DSML｜ calls> d </｜DSML｜ parameter> e";
    let p = parse(&block(&inv("write", &[("path", "true", "p"), ("content", "true", value)])));
    assert_eq!(p.calls.len(), 1, "{:?}", p.stats);
    let args = j(&p.calls[0].arguments);
    assert_eq!(get_str(&args, "content"), Some(value));
    assert_eq!(p.content, "");
}

#[test]
fn unknown_tool_unknown_param_and_type_errors_stay_text() {
    let cases: &[(&str, Reject)] = &[
        (&inv("rm_rf", &[("path", "true", "/")]), Reject::UnknownTool),
        (&inv("write", &[("path", "true", "p"), ("evil", "true", "x")]), Reject::UnknownParam),
        (&inv("write", &[("content", "true", "x")]), Reject::MissingRequired),
        (&inv("write", &[("path", "true", "p"), ("mode", "true", "7")]), Reject::TypeMismatch),
        (&inv("write", &[("path", "false", "5")]), Reject::TypeMismatch),
        (&inv("write", &[("path", "true", "p"), ("mode", "false", "{oops")]), Reject::BadJson),
        (&inv("write", &[("path", "true", "p"), ("path", "true", "q")]), Reject::DuplicateParam),
        (&inv("search::find", &[("k", "false", "9")]), Reject::NotInEnum),
    ];
    for (raw, why) in cases {
        let p = parse(&block(raw));
        assert!(p.calls.is_empty(), "{why:?}: {:?}", p.calls);
        assert_eq!(p.stats.rejected, vec![*why]);
        assert!(p.content.contains(raw), "the refused invoke stays visible: {:?}", p.content);
    }
}

#[test]
fn good_calls_survive_a_bad_neighbour() {
    let p = parse(&block(&format!("{}\n{}", inv("write", &[("path", "true", "ok")]), inv("nope", &[]))));
    assert_eq!(p.calls.len(), 1);
    assert!(p.content.contains("nope"));
}

#[test]
fn no_tools_offered_means_no_calls() {
    let text = format!("x{}", block(&inv("write", &[("path", "true", "p")])));
    for cat in [None, Some(Catalogue::from_openai_tools(&j("[]")).unwrap())] {
        let p = parse_completion(&text, ParseOptions::new(false), cat);
        assert!(p.calls.is_empty());
        assert_eq!(p.content, text);
    }
}

#[test]
fn a_json_answer_is_data_even_with_tools_offered() {
    let answer = r#"{"name": "write", "arguments": {"path": "x"}}"#;
    let p = parse_completion(answer, ParseOptions::new(false), Some(offered()));
    assert_eq!(p.content, answer);
    assert!(p.calls.is_empty());
    // and a DSML block inside a JSON-channel reply is text too
    let mut o = ParseOptions::new(false);
    o.json_channel = true;
    let text = format!("{{\"a\": 1}}{}", block(&inv("write", &[("path", "true", "p")])));
    let p = parse_completion(&text, o, Some(offered()));
    assert_eq!(p.content, text);
    assert!(p.calls.is_empty());
}

#[test]
fn orphan_closers_are_text_with_a_warning() {
    for text in ["</｜DSML｜ calls>", " \n</｜DSML｜ invoke>\t</｜DSML｜ parameter></｜DSML｜ calls>", "</｜DSML｜tool_calls>"] {
        let p = parse(text);
        assert_eq!(p.content, text);
        assert_eq!(p.warnings, vec![Warning::OrphanDsmlClosers], "{text:?}");
    }
    for text in ["Example: </｜DSML｜ calls>", "\"</｜DSML｜ calls>\"", "</｜DSML｜ calls> suffix", "", "  "] {
        assert!(parse(text).warnings.is_empty(), "{text:?}");
    }
    // orphan closers after a real call are not a warning
    let p = parse(&format!("{}</｜DSML｜ calls>", block(&inv("noargs", &[]))));
    assert!(p.warnings.is_empty());
    assert_eq!(p.calls.len(), 1);
}

#[test]
fn short_and_cut_off_payloads_stay_text_and_never_panic() {
    let open = "<｜DSML｜ calls>";
    let close = "</｜DSML｜ calls>";
    let texts = [
        open.to_string(),
        format!("{open}{}", &close[..close.len() - 1]),
        format!("{open}\n"),
        format!("\n\n{open}"),
        format!("x\n\n{open}\n<｜DSML｜ invoke name=\"wri"),
        format!("{open}\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">unfinished"),
        format!("{open}\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">p</｜DSML｜ parameter>"),
        format!("{open}{close}"),
        format!("{open}\n{close}"),
    ];
    for t in &texts {
        let p = parse(t);
        assert!(p.calls.is_empty(), "{t:?}");
    }
    // an empty block makes no call and no text; everything cut off stays visible
    assert_eq!(parse(&format!("{open}{close}")).content, "");
    assert_eq!(parse(&texts[1]).content, texts[1]);
    assert_eq!(parse(&texts[5]).content, texts[5]);
    assert_eq!(parse(&texts[0]).content, texts[0]);
}

#[test]
fn calls_whole_but_block_closer_missing_are_kept() {
    let t = format!("\n\n<｜DSML｜ calls>\n{}\n", inv("write", &[("path", "true", "p")]));
    let p = parse(&t);
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.content, "");
}

#[test]
fn prose_between_and_after_blocks() {
    let t = format!("a{}\n\nb{}", block(&inv("noargs", &[])), block(&inv("noargs", &[])));
    let p = parse(&t);
    assert_eq!(p.calls.len(), 2);
    assert_eq!(p.content, "ab");
}

#[test]
fn max_calls_limits_and_counts() {
    let mut o = ParseOptions::new(false);
    o.max_calls = Some(1);
    let t = block(&format!("{}\n{}", inv("noargs", &[]), inv("noargs", &[])));
    let p = parse_completion(&t, o, Some(offered()));
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.stats.dropped_over_limit, 1);
}

#[test]
fn thinking_split_and_unclosed_think() {
    let p = parse_completion("why</think>answer<｜end▁of▁sentence｜>", ParseOptions::new(true), None);
    assert_eq!((p.reasoning_content.as_str(), p.content.as_str()), ("why", "answer"));
    let p = parse_completion("only reasoning, cut off", ParseOptions::new(true), None);
    assert_eq!((p.reasoning_content.as_str(), p.content.as_str()), ("only reasoning, cut off", ""));
    // a call opener inside reasoning is reasoning
    let t = format!("think{}</think>done", block(&inv("noargs", &[])));
    let p = parse_completion(&t, ParseOptions::new(true), Some(offered()));
    assert!(p.calls.is_empty());
    assert_eq!(p.content, "done");
}

// ---- streaming: byte-split fuzz ----

fn same(a: &Parsed, b: &Parsed) -> bool {
    a.content == b.content && a.reasoning_content == b.reasoning_content && a.calls == b.calls && a.warnings == b.warnings && a.stats.rejected == b.stats.rejected
}

fn stream_all(text: &str, sizes: &mut dyn FnMut() -> usize, opts: ParseOptions) -> (Parsed, Vec<Event>) {
    let mut s = DsmlStream::new(opts, Some(offered()));
    let bytes = text.as_bytes();
    let mut events = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let n = sizes().max(1).min(bytes.len() - i);
        events.extend(s.push_bytes(&bytes[i..i + n]));
        i += n;
    }
    events.extend(s.finish());
    (s.into_parsed(), events)
}

fn events_match_parsed(events: &[Event], p: &Parsed) {
    let (mut c, mut r, mut calls) = (String::new(), String::new(), Vec::new());
    for e in events {
        match e {
            Event::Content(x) => c.push_str(x),
            Event::Reasoning(x) => r.push_str(x),
            Event::Call(x) => calls.push(x.clone()),
        }
    }
    assert_eq!((c, r, calls), (p.content.clone(), p.reasoning_content.clone(), p.calls.clone()));
}

fn fuzz_texts() -> Vec<(String, bool)> {
    let call = inv("write", &[("path", "true", "/tmp/é"), ("content", "true", "a </｜DSML｜ parameter> b\n</｜DSML｜ invoke> c 😀"), ("mode", "false", "5")]);
    let mut v: Vec<(String, bool)> = vec![
        (format!("hello{}", block(&call)), false),
        (format!("r1 r2</think>text\n\nmore{}", block(&format!("{call}\n{}", inv("search::find", &[("q", "true", "x")])))), true),
        (format!("{}tail", block(&call)), false),
        ("plain text with\n\nnewlines\n and <｜DSML｜ nothing".to_string(), false),
        (format!("x\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n garbage"), false),
        (format!("{}", block(&inv("write", &[("path", "true", "p"), ("nope", "true", "q")]))), false),
        ("</｜DSML｜ calls></｜DSML｜ invoke>".to_string(), false),
        ("{\"json\": true}".to_string(), false),
        (format!("é\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"noargs\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\n"), false),
        (format!("a<｜end▁of▁sentence｜>ignored{}", block(&call)), false),
        (format!("{}<｜end▁of▁sentence｜>", block(&call)), false),
    ];
    // blocks with the value-ambiguity shapes
    v.push((block(&inv("write", &[("path", "true", "p</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"path\">")])), false));
    v
}

#[test]
fn byte_split_streaming_equals_whole_parse_at_every_split() {
    for (text, thinking) in fuzz_texts() {
        let whole = parse_completion(&text, ParseOptions::new(thinking), Some(offered()));
        // two chunks, every split offset (including inside UTF-8 characters)
        for cut in 0..=text.len() {
            let mut s = DsmlStream::new(ParseOptions::new(thinking), Some(offered()));
            let mut ev = s.push_bytes(&text.as_bytes()[..cut]);
            ev.extend(s.push_bytes(&text.as_bytes()[cut..]));
            ev.extend(s.finish());
            let p = s.into_parsed();
            assert!(same(&whole, &p), "split at {cut} of {text:?}\n whole: {whole:?}\n split: {p:?}");
            events_match_parsed(&ev, &p);
        }
        // one byte at a time
        let (p, ev) = stream_all(&text, &mut || 1, ParseOptions::new(thinking));
        assert!(same(&whole, &p), "bytewise {text:?}");
        events_match_parsed(&ev, &p);
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[test]
fn random_fragment_soup_is_chunk_invariant_and_never_panics() {
    let frags: Vec<String> = vec![
        "<｜DSML｜ calls>".into(), "</｜DSML｜ calls>".into(), "<｜DSML｜ invoke name=\"write\">".into(), "<｜DSML｜ invoke name=\"noargs\">".into(),
        "</｜DSML｜ invoke>".into(), "<｜DSML｜ parameter name=\"path\" string=\"true\">".into(), "<｜DSML｜ parameter name=\"mode\" string=\"false\">".into(),
        "<｜DSML｜ parameter name=\"path\">".into(), "</｜DSML｜ parameter>".into(), "\n".into(), "\n\n".into(), " ".into(), "x".into(), "é".into(), "😀".into(),
        "5".into(), "\"".into(), ">".into(), "<".into(), "</think>".into(), "｜DSML｜".into(), "<｜DSML｜ ".into(), "<｜end▁of▁sentence｜>".into(), "{\"a\":1}".into(),
    ];
    let mut rng = Rng(0x9E3779B97F4A7C15);
    for _ in 0..3000 {
        let n = 1 + rng.below(14);
        let text: String = (0..n).map(|_| frags[rng.below(frags.len())].clone()).collect();
        let thinking = rng.below(3) == 0;
        let whole = parse_completion(&text, ParseOptions::new(thinking), Some(offered()));
        let (p1, ev1) = stream_all(&text, &mut || 1, ParseOptions::new(thinking));
        assert!(same(&whole, &p1), "bytewise differs for {text:?}\n{whole:?}\n{p1:?}");
        events_match_parsed(&ev1, &p1);
        let mut r2 = Rng(rng.next() | 1);
        let (p2, ev2) = stream_all(&text, &mut || 1 + r2.below(9), ParseOptions::new(thinking));
        assert!(same(&whole, &p2), "random chunks differ for {text:?}");
        events_match_parsed(&ev2, &p2);
        // nothing is invented: every call names an offered tool
        for c in &whole.calls {
            assert!(offered().names().contains(&c.qualified_name().as_str()));
        }
    }
}

#[test]
fn nothing_is_lost_or_invented_in_text() {
    // with no recognizable call, the reply comes back byte for byte
    let mut rng = Rng(12345);
    let frags = ["a", "\n", "\n\n", "<", "｜DSML｜", " calls>", "é", "<｜DSML｜", "</｜DSML｜ calls", "x y"];
    for _ in 0..500 {
        let n = 1 + rng.below(10);
        let text: String = (0..n).map(|_| frags[rng.below(frags.len())]).collect();
        let p = parse_completion(&text, ParseOptions::new(false), Some(offered()));
        if p.calls.is_empty() && p.stats.blocks == 0 {
            assert_eq!(p.content, text);
        }
    }
}

// ---- request rules ----

#[test]
fn thinking_resolution_order() {
    let kw = |s: &str| j(s);
    let r = |t: ThinkingRequest| resolve_thinking(&t).unwrap();
    assert_eq!(r(ThinkingRequest::default()).mode, ThinkingMode::Thinking);
    assert_eq!(r(ThinkingRequest { thinking: Some(false), ..Default::default() }).mode, ThinkingMode::Chat);
    let k = kw(r#"{"enable_thinking": false}"#);
    assert_eq!(r(ThinkingRequest { chat_template_kwargs: Some(&k), thinking: Some(true), ..Default::default() }).mode, ThinkingMode::Chat);
    let k = kw(r#"{"thinking_mode": "thinking", "enable_thinking": false}"#);
    assert_eq!(r(ThinkingRequest { chat_template_kwargs: Some(&k), ..Default::default() }).mode, ThinkingMode::Thinking);
    let none = super::effort::named("none");
    let x = r(ThinkingRequest { reasoning_effort: Some(&none), ..Default::default() });
    assert_eq!(x.mode, ThinkingMode::Chat);
    assert_eq!(super::effort::budget(x.effort.as_ref()).unwrap(), 75);
    let x = r(ThinkingRequest { reasoning_effort: Some(&none), thinking: Some(true), ..Default::default() });
    assert_eq!(x.mode, ThinkingMode::Thinking);
    let e = kw("\"max\"");
    let k = kw(r#"{"reasoning_effort": 33}"#);
    let x = r(ThinkingRequest { chat_template_kwargs: Some(&k), reasoning_effort: Some(&e), ..Default::default() });
    assert_eq!(super::effort::budget(x.effort.as_ref()).unwrap(), 33);
    let (m, xh) = (kw("\"medium\""), kw("\"xhigh\""));
    // INT12: medium is the pack template's 62 (it was 75); xhigh stays 75 (the template's value; 88 elsewhere is an owner decision)
    assert_eq!(super::effort::budget(r(ThinkingRequest { reasoning_effort: Some(&m), ..Default::default() }).effort.as_ref()).unwrap(), 62);
    assert_eq!(super::effort::budget(r(ThinkingRequest { server_effort: Some(&m), ..Default::default() }).effort.as_ref()).unwrap(), 62);
    let mk = kw(r#"{"reasoning_effort": "medium"}"#);
    assert_eq!(super::effort::budget(r(ThinkingRequest { chat_template_kwargs: Some(&mk), ..Default::default() }).effort.as_ref()).unwrap(), 62);
    let mn = kw("\"minimal\"");
    assert_eq!(super::effort::budget(r(ThinkingRequest { reasoning_effort: Some(&mn), ..Default::default() }).effort.as_ref()).unwrap(), 50, "minimal unchanged (owner decision)");
    assert_eq!(super::effort::budget(r(ThinkingRequest { server_effort: Some(&xh), ..Default::default() }).effort.as_ref()).unwrap(), 75);
    assert!(resolve_thinking(&ThinkingRequest { chat_template_kwargs: Some(&kw("5")), ..Default::default() }).is_err());
    assert_eq!(r(ThinkingRequest { server_thinking: Some(false), ..Default::default() }).mode, ThinkingMode::Chat);
}

#[test]
fn tool_choice_rules() {
    let tools = lookup_tools();
    let none = ToolChoice::from_json(Some(&j("\"none\""))).unwrap();
    let p = prepare_tools(Some(&tools), &none, false).unwrap();
    assert!(p.prompt_tools.is_none() && p.catalogue.is_none());
    let p = prepare_tools(Some(&tools), &none, true).unwrap();
    assert!(p.prompt_tools.is_some(), "kept in the prompt");
    assert!(p.catalogue.is_none(), "but no call is ever read");
    let named = ToolChoice::from_json(Some(&j(r#"{"type":"function","function":{"name":"other"}}"#))).unwrap();
    let p = prepare_tools(Some(&tools), &named, false).unwrap();
    assert!(p.catalogue.unwrap().is_empty());
    assert_eq!(thinking_token_budget(Some(100), Some(5)), Some(100));
    assert_eq!(thinking_token_budget(None, Some(5)), Some(5));
    assert_eq!(thinking_token_budget(Some(0), None), None);
}

#[test]
fn special_ids_match_the_release() {
    let dir = Path::new(MODEL_DIR);
    let Some(t) = tokenizer() else { return };
    for (s, id) in [
        ("<｜begin▁of▁sentence｜>", special::BOS_ID), (EOS_TOKEN, special::EOS_ID), ("<think>", special::THINK_START_ID),
        ("</think>", special::THINK_END_ID), ("｜DSML｜", special::DSML_ID), (super::IMAGE_PLACEHOLDER, special::IMAGE_PLACEHOLDER_ID),
    ] {
        assert_eq!(t.token_to_id(s), Some(id), "{s}");
    }
    assert_eq!(special::stop_token_ids(dir), vec![special::EOS_ID]);
    let ids = vec![5, special::IMAGE_PLACEHOLDER_ID, 6, special::IMAGE_PLACEHOLDER_ID];
    assert_eq!(special::placeholder_positions(&ids, special::IMAGE_PLACEHOLDER_ID), vec![1, 3]);
    let (out, starts) = special::expand_image_spans(&ids, special::IMAGE_PLACEHOLDER_ID, &[2, 3]).unwrap();
    assert_eq!(out.len(), 2 + 2 + 3);
    assert_eq!(starts, vec![1, 4]);
    assert!(special::expand_image_spans(&ids, special::IMAGE_PLACEHOLDER_ID, &[1]).is_err());
}

#[test]
fn tagged_text_matches_reference_shape() {
    let b = super::preprocess::parse_tagged_text("a<image>x.png</image>b").unwrap();
    assert_eq!(arr(Some(&b)).len(), 3);
    assert!(super::preprocess::parse_tagged_text("a<image>x").is_err());
    assert!(super::preprocess::parse_tagged_text("<image></image>").is_err());
    assert_eq!(super::preprocess::parse_tagged_text("plain").unwrap(), Json::Str("plain".into()));
}

// ---- the generic Jinja path (`chat_template.jinja`, upstream main 08c8657f) ----

/// The upstream template, if a copy is on this box. The generic tokenizer loads `chat_template.jinja`
/// from the tokenizer's directory, so a scratch directory holds a link to the real tokenizer and the copy.
fn jinja_dir() -> Option<PathBuf> {
    let tpl = Path::new("/models/DeepSeek-V4.1-Flash-1017/chat_template.jinja");
    let tokj = PathBuf::from(MODEL_DIR).join("tokenizer.json");
    if !tpl.exists() || !tokj.exists() {
        return None;
    }
    let d = std::env::temp_dir().join(format!("dsv41_jinja_{}", std::process::id()));
    std::fs::create_dir_all(&d).ok()?;
    let _ = std::fs::remove_file(d.join("tokenizer.json"));
    std::os::unix::fs::symlink(&tokj, d.join("tokenizer.json")).ok()?;
    std::fs::copy(tpl, d.join("chat_template.jinja")).ok()?;
    Some(d)
}

#[test]
fn generic_jinja_path_renders_the_upstream_template_like_encoding_py() {
    let Some(d) = jinja_dir() else {
        eprintln!("SKIP: upstream chat_template.jinja copy not present");
        return;
    };
    let tk = crate::tokenizer::QwenTokenizer::from_file(d.join("tokenizer.json").to_str().unwrap()).expect("tokenizer + template load");
    let convs: Vec<(&str, &str)> = vec![
        (r#"[{"role":"system","content":"You are brief."},{"role":"user","content":"hello"}]"#, "plain"),
        (r#"[{"role":"user","content":"q1"},{"role":"assistant","content":"a1","reasoning_content":"r1"},{"role":"user","content":"q2"}]"#, "multi"),
        (r#"[{"role":"user","content":"weather?"},{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\",\"days\":3}"}}]},{"role":"tool","tool_call_id":"c1","content":"sunny"}]"#, "tool_turn"),
    ];
    let tools = j(r#"[{"type":"function","function":{"name":"get_weather","description":"Weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}]"#);
    for (conv, tag) in convs {
        for mode in ["thinking", "chat"] {
            let msgs_json: serde_json::Value = serde_json::from_str(conv).unwrap();
            let msgs: Vec<crate::tokenizer::ChatMessage> = serde_json::from_value(msgs_json).unwrap();
            let tools_sj: Vec<serde_json::Value> = serde_json::from_str::<serde_json::Value>(&crate::dsv4_chat::dumps(&tools)).unwrap().as_array().unwrap().clone();
            let kw = serde_json::json!({"thinking_mode": mode});
            let got = tk.apply_chat_template(&msgs, Some(&tools_sj), None, Some(&kw), crate::tokenizer::ThinkingMode::Auto);
            let got = match got {
                Ok(g) => g,
                Err(e) => panic!("generic path cannot render the upstream template ({tag}/{mode}): {e:?}"),
            };
            let mut messages = arr(Some(&j(conv)));
            // top-level tools attach to a leading system message, or to a new empty one (as the template does)
            if get_str(&messages[0], "role") != Some("system") {
                messages.insert(0, j(r#"{"role":"system","content":""}"#));
            }
            attach_tools(&mut messages, &tools).unwrap();
            let want = encode_messages(&messages, &EncodeOptions::new(ThinkingMode::parse(mode).unwrap())).unwrap().prompt;
            assert_eq!(got, want, "{tag}/{mode}: generic Jinja path differs from encoding.py");
        }
    }
}
