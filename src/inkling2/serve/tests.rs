//! End-to-end tests of the serve shell: a real HTTP server on a socket, the pack's tokenizer and template, and a scripted forward
//! (the model's reply as text with control-token spellings, encoded to ids). They skip (loudly) when the pack is not on this box.

use std::sync::Arc;

use anyhow::Result;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::forward::{End, Forward, Job, NotBuilt};
use super::http::{App, Settings};
use crate::tokenizer::QwenTokenizer;

const PACK: &str = "/models/Inkling-2-Mini";

struct Scripted(Vec<u32>);
impl Forward for Scripted {
    fn name(&self) -> &'static str { "scripted" }
    fn generate(&self, _job: &Job<'_>, emit: &mut dyn FnMut(u32) -> bool) -> Result<End> {
        for &t in &self.0 { if !emit(t) { return Ok(End::Stop); } }
        Ok(End::Stop)
    }
}

fn tok() -> Option<Arc<QwenTokenizer>> {
    let p = format!("{PACK}/tokenizer.json");
    if !std::path::Path::new(&p).exists() { eprintln!("SKIP: {p} not on this box"); return None; }
    Some(Arc::new(QwenTokenizer::from_file(&p).unwrap()))
}

async fn server(fwd: Arc<dyn Forward>, tok: Arc<QwenTokenizer>) -> std::net::SocketAddr {
    let settings = Settings { model_name: "ink2-test".into(), max_seq_len: 8192, default_max_tokens: None, temperature: 1.0, top_p: 1.0, top_k: 0, effort: None, image_max_tokens: 4096 };
    let app = Arc::new(App { settings, tok, forward: fwd, created: 1 });
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { let _ = super::serve_on(l, app).await; });
    addr
}

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

fn reply(tok: &QwenTokenizer, text: &str) -> Arc<dyn Forward> { Arc::new(Scripted(tok.encode(text, false).unwrap())) }
const THINK_ANSWER: &str = "<|message_model|><|content_thinking|>plan 🦉<|end_message|><|message_model|><|content_text|>Hello!<|end_message|><|content_model_end_sampling|>";
const CALL: &str = "<|message_model|><|content_thinking|>need a tool<|end_message|><|message_model|>weather<|content_invoke_tool_json|>{\"name\":\"weather\",\"args\":{\"city\":\"Zürich\"}}<|end_message|><|content_model_end_sampling|>";
fn tools() -> Value { json!([{"type":"function","function":{"name":"weather","description":"w","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]) }
fn user(t: &str) -> Value { json!([{"role":"user","content":t}]) }

#[tokio::test]
async fn the_skeleton_answers_the_surface_and_refuses_to_invent_a_reply() {
    let Some(tok) = tok() else { return };
    let addr = server(Arc::new(NotBuilt), tok).await;
    assert_eq!(http(addr, "GET", "/health", None).await.0, 200);
    let (s, _, b) = http(addr, "GET", "/v1/models", None).await;
    assert_eq!(s, 200);
    assert!(b.contains("ink2-test") && b.contains("8192"));
    for stream in [false, true] {
        let (s, _, b) = http(addr, "POST", "/v1/chat/completions", Some(&json!({"messages": user("hi"), "stream": stream}))).await;
        assert_eq!(s, 501, "stream={stream}");
        let e: Value = serde_json::from_str(&b).unwrap();
        assert_eq!(e["error"]["code"], "model_not_built");
        assert!(e["error"]["message"].as_str().unwrap().contains("not built yet"));
    }
    let (_, _, m) = http(addr, "GET", "/metrics", None).await;
    assert!(m.contains("velogb10_inkling2_serving"), "{m}");
    assert!(m.contains("velogb10_inkling2_requests_not_built_total"));
}

#[tokio::test]
async fn bad_requests_are_400_with_the_reason() {
    let Some(tok) = tok() else { return };
    let addr = server(Arc::new(NotBuilt), tok).await;
    for (body, needle) in [
        (json!({"messages": []}), "non-empty"),
        (json!({"messages": user("hi"), "reasoning_effort": "bogus"}), "reasoning_effort"),
        (json!({"messages": user("hi"), "reasoning_effort": 7}), "outside"),
        (json!({"messages": [{"role":"wizard","content":"x"}]}), "role"),
        (json!({"messages": user("hi"), "temperature": -1}), "temperature"),
        (json!({"messages": user("hi"), "tool_choice": "sometimes"}), "tool_choice"),
    ] {
        let (s, _, b) = http(addr, "POST", "/v1/chat/completions", Some(&body)).await;
        assert_eq!(s, 400, "{body}: {b}");
        assert!(b.contains(needle), "{needle}: {b}");
    }
    assert_eq!(http(addr, "POST", "/v1/chat/completions", None).await.0, 400);
}

#[tokio::test]
async fn reasoning_and_answer_split_in_the_openai_shape() {
    let Some(tok) = tok() else { return };
    let addr = server(reply(&tok, THINK_ANSWER), tok.clone()).await;
    let (s, _, b) = http(addr, "POST", "/v1/chat/completions", Some(&json!({"messages": user("hi")}))).await;
    assert_eq!(s, 200, "{b}");
    let v: Value = serde_json::from_str(&b).unwrap();
    let m = &v["choices"][0]["message"];
    assert_eq!((m["reasoning_content"].as_str(), m["content"].as_str()), (Some("plan 🦉"), Some("Hello!")));
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert!(m.get("tool_calls").is_none());
    assert!(v["usage"]["prompt_tokens"].as_u64().unwrap() > 0 && v["usage"]["completion_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn a_declared_call_becomes_a_tool_call_and_an_undeclared_one_stays_text() {
    let Some(tok) = tok() else { return };
    let addr = server(reply(&tok, CALL), tok.clone()).await;
    let (_, _, b) = http(addr, "POST", "/v1/chat/completions", Some(&json!({"messages": user("weather?"), "tools": tools()}))).await;
    let v: Value = serde_json::from_str(&b).unwrap();
    let m = &v["choices"][0]["message"];
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    assert!(m["content"].is_null());
    assert_eq!(m["reasoning_content"], "need a tool");
    let c = &m["tool_calls"][0];
    assert_eq!((c["type"].as_str(), c["function"]["name"].as_str()), (Some("function"), Some("weather")));
    assert_eq!(serde_json::from_str::<Value>(c["function"]["arguments"].as_str().unwrap()).unwrap(), json!({"city": "Zürich"}));
    assert!(c["id"].as_str().unwrap().starts_with("call_"));
    // no tools offered: the same reply never produces a call; the payload stays visible
    let (_, _, b) = http(addr, "POST", "/v1/chat/completions", Some(&json!({"messages": user("weather?")}))).await;
    let v: Value = serde_json::from_str(&b).unwrap();
    assert!(v["choices"][0]["message"].get("tool_calls").is_none());
    assert!(v["choices"][0]["message"]["content"].as_str().unwrap().contains("\"name\":\"weather\""));
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    // tool_choice none: not offered either
    let (_, _, b) = http(addr, "POST", "/v1/chat/completions", Some(&json!({"messages": user("w"), "tools": tools(), "tool_choice": "none"}))).await;
    assert!(serde_json::from_str::<Value>(&b).unwrap()["choices"][0]["message"].get("tool_calls").is_none());
    let (_, _, m) = http(addr, "GET", "/metrics", None).await;
    assert!(m.contains("velogb10_inkling2_tool_calls_total 1"), "{m}");
    assert!(m.contains("reason=\"no_catalogue\""), "{m}");
}

#[tokio::test]
async fn streaming_frames_carry_reasoning_content_calls_finish_and_usage() {
    let Some(tok) = tok() else { return };
    let addr = server(reply(&tok, CALL), tok.clone()).await;
    let (s, head, b) = http(addr, "POST", "/v1/chat/completions", Some(&json!({"messages": user("w"), "tools": tools(), "stream": true}))).await;
    assert_eq!(s, 200);
    assert!(head.to_ascii_lowercase().contains("text/event-stream"));
    let frames: Vec<&str> = b.split("\n\n").filter(|f| !f.is_empty()).collect();
    assert_eq!(*frames.last().unwrap(), "data: [DONE]");
    let js: Vec<Value> = frames.iter().filter_map(|f| f.strip_prefix("data: ")).filter(|d| *d != "[DONE]").map(|d| serde_json::from_str(d).unwrap()).collect();
    assert_eq!(js[0]["choices"][0]["delta"]["role"], "assistant", "the role chunk leads");
    let reasoning: String = js.iter().filter_map(|j| j["choices"][0]["delta"]["reasoning_content"].as_str()).collect();
    assert_eq!(reasoning, "need a tool");
    let calls: Vec<&Value> = js.iter().filter_map(|j| j["choices"][0]["delta"]["tool_calls"].get(0)).collect();
    assert_eq!(calls.len(), 1);
    assert_eq!((calls[0]["index"].as_u64(), calls[0]["function"]["name"].as_str()), (Some(0), Some("weather")));
    let fin: Vec<&Value> = js.iter().filter(|j| j["choices"].as_array().map_or(false, |c| !c.is_empty() && !c[0]["finish_reason"].is_null())).collect();
    assert_eq!(fin.len(), 1);
    assert_eq!(fin[0]["choices"][0]["finish_reason"], "tool_calls");
    assert!(js.last().unwrap()["usage"]["completion_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn a_reply_cut_by_the_budget_is_length_and_never_completes_a_call() {
    let Some(tok) = tok() else { return };
    // the script stops (Length) in the middle of the call block: the generator reports End::Length
    struct Cut(Vec<u32>);
    impl Forward for Cut {
        fn name(&self) -> &'static str { "cut" }
        fn generate(&self, _j: &Job<'_>, emit: &mut dyn FnMut(u32) -> bool) -> Result<End> { for &t in &self.0 { if !emit(t) { break; } } Ok(End::Length) }
    }
    let ids = tok.encode("<|message_model|>weather<|content_invoke_tool_json|>{\"name\":\"weather\",\"args\":{\"city\":\"Ber", false).unwrap();
    let addr = server(Arc::new(Cut(ids)), tok.clone()).await;
    let (_, _, b) = http(addr, "POST", "/v1/chat/completions", Some(&json!({"messages": user("w"), "tools": tools()}))).await;
    let v: Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert!(v["choices"][0]["message"].get("tool_calls").is_none());
}
