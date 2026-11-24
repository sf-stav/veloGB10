//! End-to-end tests of the serve shell: a REAL HTTP server on a socket, the real lane loop, ENC's encoder and DSML parser,
//! and the CPU `StubForward` standing in for the model (whose "reply" is scripted per prompt). They need the V4.1
//! tokenizer (`tokenizer.json` of the native checkpoint); without it each test says so and passes (the public repository
//! suite has no lab fixtures).

use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::api::ServeSettings;
use super::engine::{self, EngineCfg};
use super::forward::{Forward, StubForward};
use super::http::{App, HttpOpts};
use super::sampler::Sampling;
use super::sink::SinkCfg;
use crate::dsv41::chat::special::EOS_ID;
use crate::tokenizer::QwenTokenizer;

const MODEL_DIR: &str = "/models/DeepSeek-V4.1-Flash-1017";

fn tokenizer() -> Option<Arc<QwenTokenizer>> {
    let p = format!("{MODEL_DIR}/tokenizer.json");
    if !std::path::Path::new(&p).exists() {
        eprintln!("[dsv41 serve tests] {p} not found: skipping (lab fixture)");
        return None;
    }
    Some(Arc::new(QwenTokenizer::from_file(&p).expect("tokenizer")))
}

/// What the stub says for a prompt: its decoded text in, the reply text out.
type Script = Box<dyn Fn(&str) -> String + Send + Sync>;

struct Srv { addr: std::net::SocketAddr }

struct Knobs {
    lanes: usize,
    prefix: bool,
    sink: Option<SinkCfg>,
    /// the stub flags the row a token with this text is fed at
    sink_tok: Option<&'static str>,
    second: Option<String>,
    delay_ms: u64,
    keepalive: u64,
    keepalive_data: bool,
}
impl Default for Knobs {
    fn default() -> Self { Knobs { lanes: 2, prefix: false, sink: None, sink_tok: None, second: None, delay_ms: 0, keepalive: 0, keepalive_data: false } }
}

async fn start(tok: &Arc<QwenTokenizer>, script: Script, k: Knobs) -> Srv {
    let t2 = tok.clone();
    let reply: super::forward::ReplyFn = Box::new(move |prompt: &[u32]| {
        let text = t2.decode(prompt, false).unwrap_or_default();
        t2.encode(&script(&text), false).unwrap_or_default()
    });
    let mut stub = StubForward::new(129_280, EOS_ID, k.lanes, 8192, reply);
    if k.delay_ms > 0 { stub = stub.with_prefill_delay(Duration::from_millis(k.delay_ms)); }
    if let Some(s) = &k.second { stub = stub.with_second(tok.encode(s, false).unwrap()[0]); }
    if let Some(t) = k.sink_tok {
        let id = tok.encode(t, false).unwrap()[0];
        stub = stub.with_sink(Box::new(move |toks: &[u32]| toks.last() == Some(&id)));
    }
    let mut cfg = EngineCfg::simple(k.lanes, vec![EOS_ID]);
    if k.prefix { cfg.prefix = Some((1 << 30, super::prefix::MIN_KEEP)); cfg.prefill_chunk = 512; }
    if let Some(s) = k.sink { cfg.sink = s; }
    let handle = engine::spawn(Box::new(stub) as Box<dyn Forward + Send>, cfg);
    let settings = ServeSettings {
        model_name: "dsv41-test".into(), max_seq_len: 8192, default_max_tokens: None, default_sampling: Sampling::default(),
        server_thinking: None, server_effort: None, keep_tools_when_none: false, seedless_mode: "random".into(), server_think_budget: None,
        include_usage_default: true,
        vision: Some(crate::dsv41::config::Vision {
            layers: 32, hidden: 1024, heads: 16, inter: 2816, patch: 14, rope_theta: 10000.0,
            downsample: 3, max_image_tokens: 1024, min_pixels: 295_936, max_wh_ratio: None,
        }),
        image_token_id: 129_264,
        vision_on: true,
        max_images: 4,
    };
    let app = Arc::new(App {
        settings, tok: tok.clone(), engine: handle, max_model_len: 8192, created: 1,
        http: HttpOpts { max_waiting: 256, backlog: 65_536, keepalive_secs: k.keepalive, keepalive_data: k.keepalive_data },
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { let _ = super::serve_on(listener, app).await; });
    Srv { addr }
}

/// One HTTP/1.1 request, `Connection: close`; returns (status, header text, body with chunked framing removed).
async fn http(addr: std::net::SocketAddr, method: &str, path: &str, body: Option<&Value>) -> (u16, String, String) {
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let b = body.map(|b| b.to_string()).unwrap_or_default();
    let req = format!("{method} {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}", b.len());
    s.write_all(req.as_bytes()).await.unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).await.unwrap();
    let text = String::from_utf8_lossy(&raw).to_string();
    let (head, rest) = text.split_once("\r\n\r\n").expect("http response");
    let status: u16 = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    let body = if head.to_ascii_lowercase().contains("transfer-encoding: chunked") { dechunk(rest) } else { rest.to_string() };
    (status, head.to_string(), body)
}

fn dechunk(mut s: &str) -> String {
    let mut out = String::new();
    while let Some((len, rest)) = s.split_once("\r\n") {
        let n = usize::from_str_radix(len.trim(), 16).unwrap_or(0);
        if n == 0 { break; }
        out.push_str(&rest[..n]);
        s = &rest[n + 2..];
    }
    out
}

/// The SSE frames of a body: Ok(json) for data frames, Err(text) for comment lines; "[DONE]" as Err("[DONE]").
fn frames(body: &str) -> Vec<Result<Value, String>> {
    body.split("\n\n").filter(|f| !f.is_empty()).map(|f| {
        if let Some(d) = f.strip_prefix("data: ") { if d == "[DONE]" { Err("[DONE]".into()) } else { Ok(serde_json::from_str(d).expect("json frame")) } }
        else { Err(f.to_string()) }
    }).collect()
}

fn chat(msgs: Value) -> Value { json!({"model": "dsv41-test", "messages": msgs, "seed": 7}) }
fn user(t: &str) -> Value { json!([{"role": "user", "content": t}]) }

struct Streamed { reasoning: String, content: String, calls: Vec<Value>, finish: Option<String>, usage: Option<Value>, order: Vec<String>, comments: usize }

fn collect(body: &str) -> Streamed {
    let mut r = Streamed { reasoning: String::new(), content: String::new(), calls: vec![], finish: None, usage: None, order: vec![], comments: 0 };
    for f in frames(body) {
        match f {
            Err(c) if c == "[DONE]" => r.order.push("done".into()),
            Err(_) => { r.comments += 1; r.order.push("comment".into()); }
            Ok(v) => {
                let ch = &v["choices"];
                if ch.as_array().map_or(true, |a| a.is_empty()) {
                    if let Some(u) = v.get("usage") { r.usage = Some(u.clone()); r.order.push("usage".into()); } else { r.order.push("empty-choices".into()); }
                    continue;
                }
                let d = &ch[0]["delta"];
                if d.get("role").is_some() { r.order.push("role".into()); }
                if let Some(t) = d.get("reasoning_content").and_then(|x| x.as_str()) { r.reasoning.push_str(t); r.order.push("reasoning".into()); }
                if let Some(t) = d.get("content").and_then(|x| x.as_str()) { r.content.push_str(t); r.order.push("content".into()); }
                if let Some(tc) = d.get("tool_calls") { r.calls.extend(tc.as_array().unwrap().iter().cloned()); r.order.push("call".into()); }
                if let Some(f) = ch[0]["finish_reason"].as_str() { r.finish = Some(f.to_string()); r.order.push("finish".into()); }
            }
        }
    }
    r
}

macro_rules! need_tok { () => { match tokenizer() { Some(t) => t, None => return } }; }

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn streams_a_thinking_chat_completion_through_the_real_server() {
    let tok = need_tok!();
    let srv = start(&tok, Box::new(|_| "Let me think.</think>\n\nHello there!".into()), Knobs::default()).await;
    let mut req = chat(user("Say hello"));
    req["stream"] = json!(true);
    let (st, head, body) = http(srv.addr, "POST", "/v1/chat/completions", Some(&req)).await;
    assert_eq!(st, 200, "{body}");
    assert!(head.to_ascii_lowercase().contains("text/event-stream"));
    let c = collect(&body);
    // hygiene: the first frame is the role chunk (it follows the first token), the last is [DONE], a usage chunk precedes it
    assert_eq!(c.order.first().map(String::as_str), Some("role"), "{:?}", c.order);
    assert_eq!(c.order.last().map(String::as_str), Some("done"));
    assert_eq!(c.order[c.order.len() - 2], "usage");
    assert_eq!(c.order.iter().filter(|o| *o == "role").count(), 1);
    assert_eq!(c.reasoning, "Let me think.");
    assert_eq!(c.content, "\n\nHello there!");
    assert_eq!(c.finish.as_deref(), Some("stop"));
    let u = c.usage.expect("usage chunk");
    assert!(u["prompt_tokens"].as_u64().unwrap() > 5 && u["completion_tokens"].as_u64().unwrap() > 5);
    assert_eq!(u["prompt_tokens_details"]["cached_tokens"], 0);
    assert_eq!(u["total_tokens"].as_u64().unwrap(), u["prompt_tokens"].as_u64().unwrap() + u["completion_tokens"].as_u64().unwrap());
    assert!(super::metrics::get(&super::metrics::THINK_ON) >= 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_thinking_toggle_changes_the_prompt_and_the_reply_shape() {
    let tok = need_tok!();
    // the script answers by what the prompt ends with, so the test sees the toggle reach the model
    let srv = start(&tok, Box::new(|p| if p.trim_end().ends_with("<think>") { "pondering</think>answer".into() } else { "plain answer".into() }), Knobs::default()).await;
    for (extra, think) in [(json!({}), true), (json!({"thinking": {"type": "disabled"}}), false), (json!({"chat_template_kwargs": {"thinking": false}}), false),
                           (json!({"thinking": true}), true)] {
        let mut req = chat(user("hi"));
        for (k, v) in extra.as_object().unwrap() { req[k] = v.clone(); }
        let (st, _, body) = http(srv.addr, "POST", "/v1/chat/completions", Some(&req)).await;
        assert_eq!(st, 200, "{body}");
        let m: Value = serde_json::from_str(&body).unwrap();
        let msg = &m["choices"][0]["message"];
        if think {
            assert_eq!(msg["reasoning_content"], "pondering", "{extra}");
            assert_eq!(msg["content"], "answer");
        } else {
            assert!(msg.get("reasoning_content").is_none(), "{extra}: {msg}");
            assert_eq!(msg["content"], "plain answer");
        }
    }
    let (st, _, body) = http(srv.addr, "POST", "/v1/chat/completions", Some(&{ let mut r = chat(user("hi")); r["thinking"] = json!("maybe"); r })).await;
    assert_eq!(st, 400, "{body}");
}

fn dsml_call(name: &str, args: Value) -> String {
    let calls = crate::dsv4_chat::parse_json(&json!([{"id": "x", "type": "function", "function": {"name": name, "arguments": args.to_string()}}]).to_string()).unwrap();
    let internal = crate::dsv41::chat::tools::tool_calls_from_openai_format(&calls).unwrap();
    crate::dsv41::chat::tools::render_tool_calls(&internal).unwrap()
}

fn weather_tools() -> Value {
    json!([{"type": "function", "function": {"name": "get_weather", "description": "weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}])
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_tool_call_streams_and_returns_whole_with_finish_tool_calls() {
    let tok = need_tok!();
    let call = dsml_call("get_weather", json!({"city": "Paris"}));
    let c2 = call.clone();
    let srv = start(&tok, Box::new(move |_| format!("Checking.</think>Sure.\n\n{c2}")), Knobs::default()).await;
    let mut req = chat(user("weather in Paris?"));
    req["tools"] = weather_tools();
    req["stream"] = json!(true);
    let (st, _, body) = http(srv.addr, "POST", "/v1/chat/completions", Some(&req)).await;
    assert_eq!(st, 200, "{body}");
    let c = collect(&body);
    assert_eq!(c.finish.as_deref(), Some("tool_calls"), "{body}");
    assert_eq!(c.calls.len(), 1, "{body}");
    assert_eq!(c.calls[0]["function"]["name"], "get_weather");
    assert_eq!(c.calls[0]["index"], 0);
    assert!(c.calls[0]["id"].as_str().unwrap().starts_with("call_"));
    let args: Value = serde_json::from_str(c.calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args, json!({"city": "Paris"}));
    assert_eq!(c.content.trim(), "Sure.", "text before the call is content, the DSML block is not");
    assert!(!c.content.contains("DSML"));
    // not streamed
    req["stream"] = json!(false);
    let (st, _, body) = http(srv.addr, "POST", "/v1/chat/completions", Some(&req)).await;
    assert_eq!(st, 200);
    let m: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(m["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(m["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "get_weather");
    // tool_choice none: the same reply is plain text, no call is ever read
    req["tool_choice"] = json!("none");
    let (_, _, body) = http(srv.addr, "POST", "/v1/chat/completions", Some(&req)).await;
    let m: Value = serde_json::from_str(&body).unwrap();
    assert!(m["choices"][0]["message"].get("tool_calls").is_none(), "{body}");
    assert!(m["choices"][0]["message"]["content"].as_str().unwrap().contains("get_weather"));
    assert!(super::metrics::get(&super::metrics::TOOL_CALLS) >= 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_call_to_an_unknown_tool_is_never_invented() {
    let tok = need_tok!();
    let call = dsml_call("delete_everything", json!({"path": "/"}));
    let srv = start(&tok, Box::new(move |_| format!("x</think>{call}")), Knobs::default()).await;
    let mut req = chat(user("go"));
    req["tools"] = weather_tools();
    let (_, _, body) = http(srv.addr, "POST", "/v1/chat/completions", Some(&req)).await;
    let m: Value = serde_json::from_str(&body).unwrap();
    assert!(m["choices"][0]["message"].get("tool_calls").is_none(), "{body}");
    assert_eq!(m["choices"][0]["finish_reason"], "stop");
    assert!(super::metrics::get(&super::metrics::TOOL_REJECTED) >= 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stop_string_ends_the_reply_is_never_streamed_and_frees_the_lane() {
    let tok = need_tok!();
    let srv = start(&tok, Box::new(|_| "x</think>alpha beta STOP gamma delta".into()), Knobs { lanes: 1, ..Default::default() }).await;
    let mut req = chat(user("go"));
    req["stop"] = json!(["STOP"]);
    req["stream"] = json!(true);
    let (_, _, body) = http(srv.addr, "POST", "/v1/chat/completions", Some(&req)).await;
    let c = collect(&body);
    assert_eq!(c.content, "alpha beta ");
    assert!(!body.contains("STOP") && !body.contains("gamma"));
    assert_eq!(c.finish.as_deref(), Some("stop"));
    // the single lane is free again: a second request is served
    let (st, _, body) = http(srv.addr, "POST", "/v1/chat/completions", Some(&chat(user("again")))).await;
    assert_eq!(st, 200, "{body}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_requests_share_the_lanes_and_each_gets_its_own_reply() {
    let tok = need_tok!();
    let srv = start(&tok, Box::new(|p| if p.contains("AAA") { "x</think>reply for A".into() } else { "x</think>reply for B".into() }), Knobs::default()).await;
    let a = chat(user("AAA question"));
    let b = chat(user("BBB question"));
    let (ra, rb) = tokio::join!(http(srv.addr, "POST", "/v1/chat/completions", Some(&a)), http(srv.addr, "POST", "/v1/chat/completions", Some(&b)));
    let (ja, jb): (Value, Value) = (serde_json::from_str(&ra.2).unwrap(), serde_json::from_str(&rb.2).unwrap());
    assert_eq!(ja["choices"][0]["message"]["content"], "reply for A");
    assert_eq!(jb["choices"][0]["message"]["content"], "reply for B");
    // more requests than lanes queue and all finish
    let reqs: Vec<_> = (0..5).map(|i| { let r = chat(user(&format!("BBB {i}"))); let addr = srv.addr; tokio::spawn(async move { http(addr, "POST", "/v1/chat/completions", Some(&r)).await }) }).collect();
    for h in reqs { let (st, _, body) = h.await.unwrap(); assert_eq!(st, 200, "{body}"); assert!(body.contains("reply for B")); }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_thinking_token_budget_closes_the_reasoning_block() {
    let tok = need_tok!();
    let srv = start(&tok, Box::new(|_| "a b c d e f".into()), Knobs::default()).await;
    let mut req = chat(user("go"));
    req["thinking_token_budget"] = json!(3);
    let before = super::metrics::get(&super::metrics::THINK_BUDGET_FORCED);
    let (st, _, body) = http(srv.addr, "POST", "/v1/chat/completions", Some(&req)).await;
    assert_eq!(st, 200, "{body}");
    let m: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(m["choices"][0]["message"]["reasoning_content"], "a b c");
    assert!(super::metrics::get(&super::metrics::THINK_BUDGET_FORCED) > before);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_prefix_cache_resumes_a_follow_up_exactly_and_reports_cached_tokens() {
    let tok = need_tok!();
    let srv = start(&tok, Box::new(|p| format!("x</think>len={}", p.len())), Knobs { lanes: 1, prefix: true, ..Default::default() }).await;
    let big = "The quick brown fox jumps over the lazy dog near the river bank. ".repeat(60);
    let r1 = chat(user(&big));
    let (_, _, b1) = http(srv.addr, "POST", "/v1/chat/completions", Some(&r1)).await;
    let j1: Value = serde_json::from_str(&b1).unwrap();
    assert_eq!(j1["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
    let p1 = j1["usage"]["prompt_tokens"].as_u64().unwrap();
    assert!(p1 > 800, "prompt too short for the test: {p1}");
    // the follow-up: the same first message, the assistant's answer, a new question
    let reply1 = j1["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    let r2 = chat(json!([{"role": "user", "content": big}, {"role": "assistant", "content": reply1}, {"role": "user", "content": "and now?"}]));
    let (_, _, b2) = http(srv.addr, "POST", "/v1/chat/completions", Some(&r2)).await;
    let j2: Value = serde_json::from_str(&b2).unwrap();
    let cached = j2["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap();
    let p2 = j2["usage"]["prompt_tokens"].as_u64().unwrap();
    assert!(cached >= 128, "cached {cached}");
    assert!(cached + 134 <= p2, "the hit cap is L - 134: cached {cached}, prompt {p2}");
    assert!(cached + 400 >= p1, "the hit should be close to the shared prefix: cached {cached}, first prompt {p1}");
    // the resumed reply equals a cold one: the same follow-up on a server without the cache
    let cold = start(&tok, Box::new(|p| format!("x</think>len={}", p.len())), Knobs { lanes: 1, prefix: false, ..Default::default() }).await;
    let (_, _, c2) = http(cold.addr, "POST", "/v1/chat/completions", Some(&r2)).await;
    let jc: Value = serde_json::from_str(&c2).unwrap();
    assert_eq!(jc["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
    assert_eq!(jc["choices"][0]["message"]["content"], j2["choices"][0]["message"]["content"]);
    assert!(super::metrics::get(&super::metrics::PFX_HITS) >= 1 && super::metrics::get(&super::metrics::PFX_REUSED) >= cached);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sink_rejection_replaces_a_sunk_token_with_the_next_alternate_and_is_off_by_default() {
    let tok = need_tok!();
    let script = || -> Script { Box::new(|_| "x</think> world is big".into()) };
    // OFF (the default): the sink magnitude is never read; the reply is the scripted one
    let off = start(&tok, script(), Knobs { sink_tok: Some(" world"), second: Some("Z".into()), ..Default::default() }).await;
    let mut req = chat(user("go"));
    req["temperature"] = json!(0);
    let (_, _, b) = http(off.addr, "POST", "/v1/chat/completions", Some(&req)).await;
    let j: Value = serde_json::from_str(&b).unwrap();
    assert_eq!(j["choices"][0]["message"]["reasoning_content"], "x");
    assert_eq!(j["choices"][0]["message"]["content"], " world is big");
    // ON: the row " world" is fed at is a sink: the token is rejected and the alternate "Z" takes its place
    let on = start(&tok, script(), Knobs { sink: Some(SinkCfg { enabled: true, threshold: 85_000.0, retries: 3 }), sink_tok: Some(" world"), second: Some("Z".into()), ..Default::default() }).await;
    let before = super::metrics::get(&super::metrics::SINK_REJECTIONS);
    let (_, _, b) = http(on.addr, "POST", "/v1/chat/completions", Some(&req)).await;
    let j: Value = serde_json::from_str(&b).unwrap();
    let content = j["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(content.starts_with('Z') && content.ends_with("is big"), "{content:?}");
    assert!(super::metrics::get(&super::metrics::SINK_REJECTIONS) > before, "the rejection counter is the liveness receipt");
    // retries 0: the sunk token is committed and counted as a fall-through
    let zero = start(&tok, script(), Knobs { sink: Some(SinkCfg { enabled: true, threshold: 85_000.0, retries: 0 }), sink_tok: Some(" world"), ..Default::default() }).await;
    let ft = super::metrics::get(&super::metrics::SINK_FALLTHROUGHS);
    let (_, _, b) = http(zero.addr, "POST", "/v1/chat/completions", Some(&req)).await;
    let j: Value = serde_json::from_str(&b).unwrap();
    assert_eq!(j["choices"][0]["message"]["content"], " world is big");
    assert!(super::metrics::get(&super::metrics::SINK_FALLTHROUGHS) > ft);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_silent_stream_gets_keep_alives_before_the_first_token_and_never_a_role_chunk() {
    let tok = need_tok!();
    let srv = start(&tok, Box::new(|_| "x</think>done".into()), Knobs { delay_ms: 2300, keepalive: 1, ..Default::default() }).await;
    let mut req = chat(user("slow"));
    req["stream"] = json!(true);
    let (_, _, body) = http(srv.addr, "POST", "/v1/chat/completions", Some(&req)).await;
    let c = collect(&body);
    assert!(c.comments >= 1, "{:?}", c.order);
    let first_comment = c.order.iter().position(|o| o == "comment").unwrap();
    let role = c.order.iter().position(|o| o == "role").unwrap();
    assert!(first_comment < role, "keep-alive precedes the role chunk, which follows the first token: {:?}", c.order);
    assert_eq!(super::metrics::get(&super::metrics::ROLE_BEFORE_TOKEN), 0);
    assert_eq!(c.content, "done");
    // the data-frame variant carries an empty choices array
    let srv = start(&tok, Box::new(|_| "x</think>done".into()), Knobs { delay_ms: 1500, keepalive: 1, keepalive_data: true, ..Default::default() }).await;
    let (_, _, body) = http(srv.addr, "POST", "/v1/chat/completions", Some(&req)).await;
    assert!(collect(&body).order.contains(&"empty-choices".to_string()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn models_health_metrics_and_bad_requests() {
    let tok = need_tok!();
    let srv = start(&tok, Box::new(|_| "x</think>ok".into()), Knobs::default()).await;
    let (st, _, b) = http(srv.addr, "GET", "/v1/models", None).await;
    assert_eq!(st, 200);
    let m: Value = serde_json::from_str(&b).unwrap();
    assert_eq!(m["data"][0]["id"], "dsv41-test");
    assert_eq!(m["data"][0]["max_model_len"], 8192);
    assert_eq!(http(srv.addr, "GET", "/health", None).await.0, 200);
    let (st, _, b) = http(srv.addr, "GET", "/metrics", None).await;
    assert_eq!(st, 200);
    assert!(b.contains("velogb10_dsv41_requests_ok_total") && b.contains("velogb10_dsv41_prefix_hits_total") && b.contains("velogb10_dsv41_sink_reject"));
    for (body, code) in [(json!({"messages": []}), 400), (json!({"nomessages": 1}), 400), (json!({"messages": [{"role": "user", "content": "x"}], "temperature": -1}), 400),
                         (json!({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}), 400)] {
        let (st, _, b) = http(srv.addr, "POST", "/v1/chat/completions", Some(&body)).await;
        assert_eq!(st, code, "{body} -> {b}");
        assert_eq!(serde_json::from_str::<Value>(&b).unwrap()["error"]["type"], "invalid_request_error");
    }
    // a prompt longer than the context is a 400 naming the limit, not a hang
    let (st, _, b) = http(srv.addr, "POST", "/v1/chat/completions", Some(&chat(user(&"word ".repeat(9000))))).await;
    assert_eq!(st, 400);
    assert!(b.contains("context_length_exceeded"), "{b}");
    // finish length: max_tokens caps the reply
    let mut req = chat(user("go"));
    req["max_tokens"] = json!(1);
    let (_, _, b) = http(srv.addr, "POST", "/v1/chat/completions", Some(&req)).await;
    let j: Value = serde_json::from_str(&b).unwrap();
    assert_eq!(j["choices"][0]["finish_reason"], "length");
    assert_eq!(j["usage"]["completion_tokens"], 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_same_seed_gives_the_same_sampled_reply_whatever_else_runs() {
    let tok = need_tok!();
    let srv = start(&tok, Box::new(|_| "x</think>stable".into()), Knobs::default()).await;
    let mut a = chat(user("same"));
    a["temperature"] = json!(1.0);
    let (_, _, solo) = http(srv.addr, "POST", "/v1/chat/completions", Some(&a)).await;
    let other = chat(user("other"));
    let (r1, _) = tokio::join!(http(srv.addr, "POST", "/v1/chat/completions", Some(&a)), http(srv.addr, "POST", "/v1/chat/completions", Some(&other)));
    let content = |b: &str| serde_json::from_str::<Value>(b).unwrap()["choices"][0]["message"]["content"].clone();
    assert_eq!(content(&solo), content(&r1.2));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dsv41_ce_prompt_scoring_on_v1_completions_returns_every_tokens_logprob() {
    let tok = need_tok!();
    let srv = start(&tok, Box::new(|_| "x".into()), Knobs::default()).await;
    let before = super::metrics::get(&super::metrics::SCORE_REQS);
    // GA's g1 api-echo request: ids, echo, max_tokens 0, logprobs 0
    let ids: Vec<u32> = vec![0, 128803, 5, 77, 1023, 9];
    let (st, _, body) = http(srv.addr, "POST", "/v1/completions", Some(&json!({"model": "m", "prompt": ids, "echo": true, "max_tokens": 0, "logprobs": 0}))).await;
    assert_eq!(st, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let lp = v["choices"][0]["logprobs"]["token_logprobs"].as_array().unwrap().clone();
    assert_eq!(lp.len(), ids.len());
    assert!(lp[0].is_null(), "the first token has no log-prob");
    for (i, x) in lp.iter().enumerate().skip(1) { assert_eq!(x.as_f64().unwrap() as f32, super::forward::stub_score(ids[i])); }
    assert_eq!(v["usage"]["prompt_tokens"], ids.len());
    assert_eq!(v["choices"][0]["logprobs"]["tokens"].as_array().unwrap().len(), ids.len());
    assert!(super::metrics::get(&super::metrics::SCORE_REQS) > before);
    // a text prompt is tokenized; generation on this route is served (UXF F7: text_completion), see completions_generate_streamed_and_whole
    let (st, _, _) = http(srv.addr, "POST", "/v1/completions", Some(&json!({"prompt": "Hello world", "echo": true, "max_tokens": 0, "logprobs": 0}))).await;
    assert_eq!(st, 200);
    let (st, _, body) = http(srv.addr, "POST", "/v1/completions", Some(&json!({"prompt": "Hello", "max_tokens": 5}))).await;
    assert_eq!(st, 200, "{body}");
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["object"], "text_completion");
    // the lane is free again: a chat request still completes
    let (st, _, _) = http(srv.addr, "POST", "/v1/chat/completions", Some(&chat(user("hi")))).await;
    assert_eq!(st, 200);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tools_reach_the_prompt_when_the_request_has_no_system_message() {
    let tok = need_tok!();
    // the stub answers "has-tool" when the decoded prompt carries the tool block, "no-tool" otherwise
    let srv = start(&tok, Box::new(|p| if p.contains("get_weather") && p.contains("## Tools") { "has-tool".into() } else { "no-tool".into() }), Knobs::default()).await;
    let mut req = chat(user("weather in Paris?"));
    req["tools"] = weather_tools();
    req["thinking"] = json!(false);
    let (st, _, body) = http(srv.addr, "POST", "/v1/chat/completions", Some(&req)).await;
    assert_eq!(st, 200, "{body}");
    assert!(body.contains("has-tool"), "the tool block was dropped from the prompt: {body}");
    // no tools: no system message appears
    let mut req = chat(user("hi"));
    req["thinking"] = json!(false);
    let (_, _, body) = http(srv.addr, "POST", "/v1/chat/completions", Some(&req)).await;
    assert!(body.contains("no-tool"), "{body}");
}

// ---------------------------------------------------------------------------------------------------------------------
// UXF (RV1 F3 / F4 / F5 / F7 / L4)
// ---------------------------------------------------------------------------------------------------------------------

/// The `text_completion` frames of a streamed completion: (text, finish, usage, every object name, [DONE] seen).
fn text_frames(body: &str) -> (String, Option<String>, Option<Value>, Vec<String>, bool) {
    let (mut text, mut fin, mut usage, mut objs, mut done) = (String::new(), None, None, vec![], false);
    for f in frames(body) {
        match f {
            Err(c) if c == "[DONE]" => done = true,
            Err(_) => {}
            Ok(v) => {
                objs.push(v["object"].as_str().unwrap_or("?").to_string());
                if let Some(u) = v.get("usage") { usage = Some(u.clone()); }
                if let Some(c) = v["choices"].as_array().and_then(|a| a.first()) {
                    if let Some(t) = c["text"].as_str() { text.push_str(t); }
                    if let Some(f) = c["finish_reason"].as_str() { fin = Some(f.to_string()); }
                    assert!(c.get("delta").is_none(), "a completion chunk carries text, never a chat delta: {v}");
                }
            }
        }
    }
    (text, fin, usage, objs, done)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn completions_generate_streamed_and_whole() {
    let tok = need_tok!();
    let srv = start(&tok, Box::new(|p| if p.contains("capital of France") { " Paris, and then STOP more text".into() } else { "?".into() }), Knobs::default()).await;
    let req = json!({"model": "dsv41-test", "prompt": "The capital of France is", "max_tokens": 64, "seed": 3});
    // whole
    let (st, head, body) = http(srv.addr, "POST", "/v1/completions", Some(&req)).await;
    assert_eq!(st, 200, "{body}");
    assert!(head.to_ascii_lowercase().contains("application/json"));
    let m: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(m["object"], "text_completion");
    assert!(m["id"].as_str().unwrap().starts_with("cmpl-"));
    assert_eq!(m["choices"][0]["text"], " Paris, and then STOP more text");
    assert_eq!(m["choices"][0]["finish_reason"], "stop");
    assert!(m["choices"][0].get("message").is_none());
    let u = &m["usage"];
    assert!(u["prompt_tokens"].as_u64().unwrap() >= 5 && u["completion_tokens"].as_u64().unwrap() >= 5);
    // streamed: no role chunk, text frames, a finish frame, a usage frame, [DONE]
    let mut s = req.clone();
    s["stream"] = json!(true);
    let (st, head, body) = http(srv.addr, "POST", "/v1/completions", Some(&s)).await;
    assert_eq!(st, 200, "{body}");
    assert!(head.to_ascii_lowercase().contains("text/event-stream"));
    let (text, fin, usage, objs, done) = text_frames(&body);
    assert_eq!(text, " Paris, and then STOP more text");
    assert_eq!(fin.as_deref(), Some("stop"));
    assert!(usage.is_some() && done, "{body}");
    assert!(objs.iter().all(|o| o == "text_completion"), "{objs:?}");
    // a stop string ends the reply and is never streamed
    s["stop"] = json!(["STOP"]);
    let (_, _, body) = http(srv.addr, "POST", "/v1/completions", Some(&s)).await;
    assert_eq!(text_frames(&body).0, " Paris, and then ", "{body}");
    // max_tokens ends it with finish_reason length
    let mut l = req.clone();
    l["max_tokens"] = json!(3);
    let (_, _, body) = http(srv.addr, "POST", "/v1/completions", Some(&l)).await;
    let m: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(m["choices"][0]["finish_reason"], "length", "{body}");
    assert_eq!(m["usage"]["completion_tokens"], 3);
    // echo leads with the prompt; token-id prompts work; an empty / bad prompt is a 400
    let mut e = req.clone();
    e["echo"] = json!(true);
    let (_, _, body) = http(srv.addr, "POST", "/v1/completions", Some(&e)).await;
    let m: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(m["choices"][0]["text"], "The capital of France is Paris, and then STOP more text", "{body}");
    let ids = tok.encode("The capital of France is", true).unwrap();
    let (st, _, body) = http(srv.addr, "POST", "/v1/completions", Some(&json!({"prompt": ids, "max_tokens": 8}))).await;
    assert_eq!(st, 200, "{body}");
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["choices"][0]["text"], " Paris, and then STOP more text");
    for bad in [json!({"prompt": ""}), json!({"max_tokens": 4}), json!({"prompt": {"a": 1}}), json!({"prompt": "x", "max_tokens": 0}), json!({"prompt": [999999999]})] {
        let (st, _, body) = http(srv.addr, "POST", "/v1/completions", Some(&bad)).await;
        assert_eq!(st, 400, "{bad}: {body}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn completions_scoring_still_answers_with_echo_and_max_tokens_zero() {
    let tok = need_tok!();
    let srv = start(&tok, Box::new(|_| "x".into()), Knobs::default()).await;
    let (st, _, body) = http(srv.addr, "POST", "/v1/completions", Some(&json!({"prompt": "Hello there, my friend", "echo": true, "max_tokens": 0, "logprobs": 0}))).await;
    assert_eq!(st, 200, "{body}");
    let m: Value = serde_json::from_str(&body).unwrap();
    assert!(m["choices"][0]["logprobs"]["token_logprobs"][0].is_null());
    assert!(m["choices"][0]["logprobs"]["token_logprobs"].as_array().unwrap().len() >= 4);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tokenize_detokenize_and_the_model_route_behave_like_the_shared_server() {
    let tok = need_tok!();
    let srv = start(&tok, Box::new(|_| "Fine.".into()), Knobs::default()).await;
    // raw text: ids equal the tokenizer's, count and max_model_len are reported
    let (st, _, body) = http(srv.addr, "POST", "/v1/tokenize", Some(&json!({"model": "dsv41-test", "prompt": "Hello, world"}))).await;
    assert_eq!(st, 200, "{body}");
    let m: Value = serde_json::from_str(&body).unwrap();
    let want = tok.encode("Hello, world", true).unwrap();
    assert_eq!(m["tokens"], json!(want));
    assert_eq!(m["count"].as_u64().unwrap() as usize, want.len());
    assert_eq!(m["max_model_len"], 8192);
    // an empty prompt counts 0; truncation keeps the LAST n
    let (_, _, body) = http(srv.addr, "POST", "/v1/tokenize", Some(&json!({"model": "dsv41-test", "prompt": ""}))).await;
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["count"], 0);
    let (_, _, body) = http(srv.addr, "POST", "/v1/tokenize", Some(&json!({"model": "dsv41-test", "prompt": "Hello, world", "truncate_prompt_tokens": 2}))).await;
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["tokens"], json!(want[want.len() - 2..]));
    // messages mode counts exactly what the same chat request reports as prompt_tokens (tools included)
    let msgs = json!([{"role": "user", "content": "weather in Paris?"}]);
    let (st, _, body) = http(srv.addr, "POST", "/v1/tokenize", Some(&json!({"model": "dsv41-test", "messages": msgs, "tools": weather_tools()}))).await;
    assert_eq!(st, 200, "{body}");
    let counted = serde_json::from_str::<Value>(&body).unwrap()["count"].as_u64().unwrap();
    let mut req = chat(msgs);
    req["tools"] = weather_tools();
    let (_, _, body) = http(srv.addr, "POST", "/v1/chat/completions", Some(&req)).await;
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["usage"]["prompt_tokens"].as_u64().unwrap(), counted);
    // errors: no model, wrong model, both prompt and messages, a bad id
    for bad in [json!({"prompt": "x"}), json!({"model": "other", "prompt": "x"}), json!({"model": "dsv41-test", "prompt": "x", "messages": user("y")}),
                json!({"model": "dsv41-test", "prompt": [99999999]}), json!({"model": "dsv41-test"}), json!({"model": "dsv41-test", "prompt": "x", "truncate_prompt_tokens": 0})] {
        let (st, _, body) = http(srv.addr, "POST", "/v1/tokenize", Some(&bad)).await;
        assert_eq!(st, 400, "{bad}: {body}");
        assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["error"]["type"], "invalid_request_error");
    }
    // detokenize round-trips
    let (st, _, body) = http(srv.addr, "POST", "/v1/detokenize", Some(&json!({"model": "dsv41-test", "tokens": want}))).await;
    assert_eq!(st, 200, "{body}");
    let m: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(m["model"], "dsv41-test");
    assert_eq!(m["prompt"], tok.decode(&want, false).unwrap());
    assert!(m["prompt"].as_str().unwrap().contains("Hello, world"));
    let (_, _, body) = http(srv.addr, "POST", "/v1/detokenize", Some(&json!({"model": "dsv41-test", "tokens": []}))).await;
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["prompt"], "");
    for bad in [json!({"model": "dsv41-test"}), json!({"model": "dsv41-test", "tokens": [-1]}), json!({"model": "dsv41-test", "tokens": [99999999]}), json!({"tokens": [1]})] {
        let (st, _, _) = http(srv.addr, "POST", "/v1/detokenize", Some(&bad)).await;
        assert_eq!(st, 400, "{bad}");
    }
    // /v1/models/{id}
    let (st, _, body) = http(srv.addr, "GET", "/v1/models/dsv41-test", None).await;
    assert_eq!(st, 200, "{body}");
    let m: Value = serde_json::from_str(&body).unwrap();
    assert_eq!((m["id"].as_str(), m["object"].as_str(), m["max_model_len"].as_u64()), (Some("dsv41-test"), Some("model"), Some(8192)));
    let (st, _, body) = http(srv.addr, "GET", "/v1/models/nope/with/slashes", None).await;
    assert_eq!(st, 404);
    assert!(body.contains("not found") && body.contains("dsv41-test"), "{body}");
}

#[test]
fn prefix_cache_defaults_on_and_the_pack_name_is_its_directory() {
    use super::{pack_name, prefix_cache_on};
    assert!(prefix_cache_on(None) && prefix_cache_on(Some("on")) && prefix_cache_on(Some("true")) && prefix_cache_on(Some("1")) && prefix_cache_on(Some("auto")));
    for off in ["off", "OFF", "0", "false", "no"] { assert!(!prefix_cache_on(Some(off)), "{off}"); }
    assert_eq!(pack_name(std::path::Path::new("/home/x/models/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw/")), "DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw");
    assert_eq!(pack_name(std::path::Path::new("pack")), "pack");
}

#[test]
fn the_metrics_body_names_pv1_paths_and_the_receipt_block_ends_with_their_counts() {
    // the real renderer, and the real boot block
    crate::dsv41::fwd::receipts::kernel("uxf.test", "d41_uxf_test", 2);
    super::metrics::mark_start();
    let body = super::metrics::render_all(true);
    for name in ["velogb10_dsv41_moe_dec_calls_total", "velogb10_dsv41_moe_dec_chain_calls_total", "velogb10_dsv41_moe_dec_short_scratch_total",
                 "velogb10_dsv41_attn_dec_calls_total", "velogb10_dsv41_attn_dec_grouped_calls_total", "velogb10_dsv41_uptime_seconds",
                 "velogb10_dsv41_moe_dec_entry_calls_total{entry=\"d41x_mc_gu_a8\"}", "velogb10_dsv41_moe_dec_entry_calls_total{entry=\"d41x_md_gu_a32\"}",
                 "velogb10_dsv41_moe_cp_short_scratch_total"] {
        assert!(body.lines().any(|l| l.starts_with(name) && !l.starts_with('#')), "{name} missing from /metrics");
    }
    let lines = super::metrics::boot_receipt_lines();
    // PV1's decode paths, then PM1's decode MoE structure (the block's last line)
    assert!(lines[lines.len() - 2].starts_with("[dsv41 fwd] decode MoE (PV1):"), "{lines:?}");
    assert!(lines.last().unwrap().starts_with("[dsv41 fwd] decode MoE structure (PM1, --dsv41-moe-cta):"), "{lines:?}");
    // the stale text is gone
    assert!(!body.contains("not built under TP yet"));
}

/// `scripts/owner_check.py` on the REAL `/metrics` text of this process: the Engram rows, the forward block, the skipped / capped
/// section and the uptime print for a DeepSeek-V4.1 server (RV1 F3), and the SE2 / SE counters are reported in plain words.
#[test]
fn owner_check_tells_the_truth_about_a_dsv41_server() {
    use std::sync::atomic::Ordering::Relaxed;
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/owner_check.py");
    if !script.exists() || std::process::Command::new("python3").arg("--version").output().is_err() {
        return; // the public repository suite has neither the script nor (maybe) python3
    }
    use super::metrics as m;
    m::mark_start();
    m::set(&m::SERVING, 1);
    m::set(&m::LANES, 4);
    m::set(&m::LANES_REQUESTED, 4);
    m::add(&m::CUT_CAPPED, 3);
    m::add(&m::TP_FATAL, 1);
    m::add(&m::PREFILL_YIELDS, 2);
    m::add(&m::CUT_UNAPPLIED, 1);
    m::add(&m::VIS_REQS, 2);
    m::add(&m::VIS_IMAGES, 2);
    m::add(&m::VIS_TOKENS, 378);
    m::add(&m::VIS_ENCODE_US, 18_500);
    crate::dsv41::fwd::receipts::count("uxf.path", crate::dsv41::weights::BackendKind::Exl3, "d41x_uxf", 24, crate::dsv41::weights::RowClass::Generic);
    crate::dsv41::engram::io_metrics::ACTIVE.store(true, Relaxed);
    crate::dsv41::engram::io_metrics::STEPS[0].fetch_add(5, Relaxed);
    crate::dsv41::engram::io_metrics::STALLS.fetch_add(2, Relaxed);
    let dir = std::env::temp_dir().join(format!("uxf_oc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("metrics.txt");
    std::fs::write(&f, m::render_all(true)).unwrap();
    let out = std::process::Command::new("python3").arg("-I").arg(&script).arg("--file").arg(&f).output().expect("run owner_check.py");
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(out.status.success(), "{text}\n{}", String::from_utf8_lossy(&out.stderr));
    for want in ["DeepSeek-V4.1 server", "DEEPSEEK-V4.1 ENGRAM ROWS", "LATE ROWS", "DEEPSEEK-V4.1 FORWARD", "GENERIC entry", "WHAT WAS SKIPPED, CAPPED OR LIMITED",
                 "TP OP(S) FAILED", "CAPPED: ", "WITHOUT the cut", "prefill yields: ", "decode MoE (PV1", "CUDA-graph replays",
                 "vision (ViT + 3x3 aligner on rank 0): 2 multimodal request(s)"] {
        assert!(text.contains(want), "owner_check output lacks {want:?}:\n{text}");
    }
    assert!(!text.contains("uptime not exposed"), "{text}");
}

#[tokio::test]
async fn vision_inline_base64_serves_and_remote_or_local_urls_are_rejected_400() {
    use base64::Engine as _;
    let Some(tok) = tokenizer() else { return };
    let s = start(&tok, Box::new(|_| "crimson swatch".into()), Knobs { prefix: true, ..Knobs::default() }).await;
    let mut png = Vec::new();
    let img = image::RgbImage::from_pixel(64, 48, image::Rgb([180, 40, 60]));
    image::DynamicImage::ImageRgb8(img).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
    let b64 = base64::engine::general_purpose::STANDARD.encode(&png);

    // Valid data:image/png;base64,... -> 200 OK with expanded prompt_tokens (> 150)
    let ok_req = serde_json::json!({
        "model": "dsv41-test",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{b64}")}},
            {"type": "text", "text": "Describe this color."}
        ]}],
        "max_tokens": 8,
        "temperature": 0.0
    });
    let (st, _, body) = http(s.addr, "POST", "/v1/chat/completions", Some(&ok_req)).await;
    assert_eq!(st, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let ptoks = v["usage"]["prompt_tokens"].as_u64().unwrap();
    assert!(ptoks > 150, "placeholder was not expanded: prompt_tokens={ptoks}");

    // Remote http:// and local path -> 400 invalid_image_url
    for bad_url in ["http://127.0.0.1:9999/a.png", "https://example.com/a.png", "/etc/passwd", "file:///etc/passwd"] {
        let bad_req = serde_json::json!({
            "model": "dsv41-test",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": bad_url}},
                {"type": "text", "text": "What is this?"}
            ]}]
        });
        let (st_b, _, body_b) = http(s.addr, "POST", "/v1/chat/completions", Some(&bad_req)).await;
        assert_eq!(st_b, 400, "{bad_url}: {body_b}");
        let vb: Value = serde_json::from_str(&body_b).unwrap();
        assert_eq!(vb["error"]["code"].as_str(), Some("invalid_image_url"), "{bad_url}: {body_b}");
    }

    // Real-world client multimodal variants (`input_image` + `input_text`, `{type: "image", image_url: ...}`,
    // `{type: "image_url", url: ...}`, bare base64 in `url`, `developer` role, `reasoning_effort: "off"`,
    // `max_tokens: -1`, `seed: -1`, `thinking_token_budget: -1`) all serve cleanly with 200 OK.
    for variant in [
        serde_json::json!([
            {"role": "developer", "content": "Be concise."},
            {"role": "user", "content": [
                {"type": "input_image", "image_url": format!("data:image/png;base64,{b64}")},
                {"type": "input_text", "text": "Color?"}
            ]}
        ]),
        serde_json::json!([{"role": "user", "content": [
            {"type": "image", "image_url": {"url": b64}},
            {"text": "Color?"}
        ]}]),
        serde_json::json!([{"role": "user", "content": [
            {"type": "image_url", "url": format!("data:base64,{b64}")},
            {"type": "text", "text": serde_json::Value::Null}
        ]}]),
        serde_json::json!([{"role": "user", "content": [
            {"type": "image", "image": format!("data:image/png;base64,{b64}")},
            "Color?"
        ]}]),
    ] {
        let req_v = serde_json::json!({
            "model": "dsv41-test",
            "messages": variant,
            "max_tokens": -1,
            "seed": -1,
            "thinking_token_budget": -1,
            "reasoning_effort": "off",
        });
        let (st_v, _, body_v) = http(s.addr, "POST", "/v1/chat/completions", Some(&req_v)).await;
        assert_eq!(st_v, 200, "{variant}: {body_v}");
        let vv: Value = serde_json::from_str(&body_v).unwrap();
        assert!(vv["usage"]["prompt_tokens"].as_u64().unwrap() > 150, "{variant}: {body_v}");
    }

    // Video/audio content parts are explicitly refused with 400 naming the part type (VIS-5 / VID-0 parity).
    let vid_req = serde_json::json!({
        "model": "dsv41-test",
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
            {"type": "text", "text": "What happens?"}
        ]}]
    });
    let (st_vid, _, body_vid) = http(s.addr, "POST", "/v1/chat/completions", Some(&vid_req)).await;
    assert_eq!(st_vid, 400, "{body_vid}");
    assert!(body_vid.contains("video_url"), "{body_vid}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn min_tokens_and_ignore_eos_work_on_chat_and_text_completions() {
    let tok = need_tok!();
    // Scripted reply is short ("ok</think>hi"), then StubForward wants EOS on every subsequent row (`second = "Z"`).
    let srv = start(
        &tok,
        Box::new(|p| if p.contains("raw") { "hi".into() } else { "ok</think>hi".into() }),
        Knobs { second: Some("Z".into()), ..Default::default() },
    ).await;

    // 1. Default / min_tokens: 0 / min_tokens: -1 stops normally on EOS.
    for extra in [json!({}), json!({"min_tokens": 0}), json!({"min_tokens": -1}), json!({"ignore_eos": false})] {
        let mut req = chat(user("go"));
        for (k, v) in extra.as_object().unwrap() { req[k] = v.clone(); }
        let (st, _, body) = http(srv.addr, "POST", "/v1/chat/completions", Some(&req)).await;
        assert_eq!(st, 200, "{extra}: {body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "stop", "{extra}: {body}");
        assert!(v["usage"]["completion_tokens"].as_u64().unwrap() <= 6, "{extra}: {body}");
    }

    // 2. min_tokens prevents early EOS stop on both greedy and sampled chat completions, then allows stop at >= min_tokens.
    for temp in [0.0, 0.8] {
        let mut req = chat(user("go"));
        req["temperature"] = json!(temp);
        req["min_tokens"] = json!(14);
        req["max_tokens"] = json!(32);
        let (st, _, body) = http(srv.addr, "POST", "/v1/chat/completions", Some(&req)).await;
        assert_eq!(st, 200, "temp={temp}: {body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "stop", "temp={temp}: {body}");
        assert_eq!(v["usage"]["completion_tokens"], 14, "temp={temp}: {body}");
    }

    // 3. ignore_eos: true generates exact max_tokens on both /v1/chat/completions (streamed + non-streamed) and /v1/completions.
    let mut ign = chat(user("go"));
    ign["ignore_eos"] = json!(true);
    ign["max_tokens"] = json!(19);
    let (st, _, body) = http(srv.addr, "POST", "/v1/chat/completions", Some(&ign)).await;
    assert_eq!(st, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{body}");
    assert_eq!(v["usage"]["completion_tokens"], 19, "{body}");

    ign["stream"] = json!(true);
    let (st, _, sbody) = http(srv.addr, "POST", "/v1/chat/completions", Some(&ign)).await;
    assert_eq!(st, 200, "{sbody}");
    let sc = collect(&sbody);
    assert_eq!(sc.finish.as_deref(), Some("length"));
    assert_eq!(sc.usage.unwrap()["completion_tokens"], 19);

    for (extra, want_fin, want_toks) in [
        (json!({"min_tokens": -1, "max_tokens": 20}), "stop", 1u64),
        (json!({"min_tokens": 11, "max_tokens": 25, "temperature": 0.0}), "stop", 11),
        (json!({"min_tokens": 11, "max_tokens": 25, "temperature": 0.7}), "stop", 11),
        (json!({"ignore_eos": true, "max_tokens": 15}), "length", 15),
    ] {
        let mut creq = json!({"model": "dsv41-test", "prompt": "raw prompt", "seed": 5});
        for (k, v) in extra.as_object().unwrap() { creq[k] = v.clone(); }
        let (st, _, body) = http(srv.addr, "POST", "/v1/completions", Some(&creq)).await;
        assert_eq!(st, 200, "{creq}: {body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], want_fin, "{creq}: {body}");
        assert_eq!(v["usage"]["completion_tokens"].as_u64().unwrap(), want_toks, "{creq}: {body}");
    }

    // 4. Invalid types / values return 400 invalid_request_error.
    for bad in [
        json!({"model": "dsv41-test", "messages": user("x"), "min_tokens": -2}),
        json!({"model": "dsv41-test", "messages": user("x"), "min_tokens": 1.5}),
        json!({"model": "dsv41-test", "messages": user("x"), "ignore_eos": "yes"}),
        json!({"model": "dsv41-test", "prompt": "x", "max_tokens": 8, "min_tokens": -5}),
        json!({"model": "dsv41-test", "prompt": "x", "max_tokens": 8, "ignore_eos": 1}),
    ] {
        let route = if bad.get("messages").is_some() { "/v1/chat/completions" } else { "/v1/completions" };
        let (st, _, body) = http(srv.addr, "POST", route, Some(&bad)).await;
        assert_eq!(st, 400, "{bad}: {body}");
        assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["error"]["type"], "invalid_request_error");
    }

    let (_, _, mbody) = http(srv.addr, "GET", "/metrics", None).await;
    assert!(mbody.contains("velogb10_dsv41_min_tokens_requests_total"), "{mbody}");
    assert!(mbody.contains("velogb10_dsv41_ignore_eos_requests_total"), "{mbody}");
}
