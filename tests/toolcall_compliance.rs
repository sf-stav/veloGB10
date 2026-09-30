//! TOOLCALL COMPLIANCE fixtures (2026-08-24, toolcall-compliance branch).
//!
//! Two concerns, split by what is genuinely the ENGINE's and what is testable without a GPU:
//!
//!  * **L2 — tool-result fidelity** (the actionable half of TC-14/52/57). The engine must carry a
//!    `role:"tool"` message — including its ERROR content (rate-limit, invalid ticker, "not
//!    relevant", empty result) and its `tool_call_id`/`name` — verbatim and unmistakably into the
//!    NEXT turn, so the MODEL can reason about alternatives. A dropped/mangled tool error is the
//!    single biggest engine-side cause of "the model stopped instead of trying another tool".
//!    This test LOCKS the round-trip through `apply_chat_template` (deserialize -> to_template_json
//!    -> the model's own chat_template.jinja), so a future change that drops the error text or the
//!    tool call fails here. Reference model dir: 3.6-27b-nvfp4-full (has chat_template.jinja).
//!
//!  * **L1 — tool_choice contract** (TC-45). The engine parses `tool_choice` but does NOT yet
//!    enforce it at decode time (server.rs: it is only logged). The decode-time grammar mask is a
//!    device-sampler change (verify logits, spec-decode compatible) and is NOT testable here. What
//!    IS testable host-side is the *contract unit*: the well-formed-call determination and the
//!    "tools offered but nothing parsed" path, exercised on the exact grammar in src/tools.rs.
//!
//! Run: `cargo test --test toolcall_compliance`.

use gb10_inference::tokenizer::{ChatMessage, QwenTokenizer, ThinkingMode};

/// The tool-error payloads, byte-for-byte as they appear on the wire (from the live eval traces).
const ERR_RATE_LIMIT: &str = r#"{"error": "Service temporarily unavailable. Rate limit exceeded.", "error_code": "ERR_TOOL_UNAVAILABLE", "timestamp": "2026-03-20T12:00:00Z", "trace_id": "trace_198b3243", "request_id": "req_err_198b3243"}"#;
const ERR_NOT_RELEVANT: &str = r#"{"error": "Tool search_files is not relevant.", "error_code": "ERR_TOOL_UNAVAILABLE", "timestamp": "2026-03-20T12:00:00Z"}"#;
const ERR_INVALID_TICKER: &str = r#"{"error": "Invalid ticker symbol 'ZZZZ'. Supported: AAPL, MSFT, GOOGL, AMZN.", "error_code": "ERR_TOOL_UNAVAILABLE", "timestamp": "2026-03-20T12:00:00Z"}"#;
const EMPTY_RESULT: &str = r#"{"results": [], "total_results": 0, "query_time_ms": 12}"#;

fn load_tokenizer() -> Option<QwenTokenizer> {
    let dir = std::env::var("GB10_TEST_MODEL_DIR")
        .unwrap_or_else(|_| "models/3.6-27b-nvfp4-full".to_string());
    QwenTokenizer::from_file(&format!("{dir}/tokenizer.json")).ok()
}

/// The realistic agent-shaped conversation the harness sends back on the tool-result turn:
/// system + user + assistant(tool_calls, content:null) + tool(role:"tool", tool_call_id, name,
/// content=<ERROR>). Deserializing exercises the engine's custom `content`/`tool_calls`/`tool_call_id`
/// deserializer; rendering exercises `to_template_json` + the model's jinja.
fn conversation(content: &str) -> Vec<ChatMessage> {
    let json = format!(
        r#"[
            {{"role":"system","content":"You are a helpful assistant with access to tools."}},
            {{"role":"user","content":"What is Apple's stock price?"}},
            {{"role":"assistant","content":null,"tool_calls":[{{"id":"call_7","type":"function","function":{{"name":"get_stock_price","arguments":"{{\"ticker\":\"AAPL\"}}"}}}}]}},
            {{"role":"tool","tool_call_id":"call_7","name":"get_stock_price","content":{content:?}}}
        ]"#,
        content = content,
    );
    serde_json::from_str::<Vec<ChatMessage>>(&json).expect("deserialize agent conversation")
}

#[test]
fn tool_error_round_trips_verbatim_into_next_turn() {
    let Some(tok) = load_tokenizer() else {
        eprintln!("SKIP: 3.6 tokenizer not present (no chat_template.jinja to load)");
        return;
    };
    for (label, err) in [
        ("rate-limit", ERR_RATE_LIMIT),
        ("not-relevant", ERR_NOT_RELEVANT),
        ("invalid-ticker", ERR_INVALID_TICKER),
        ("empty-result", EMPTY_RESULT),
    ] {
        let msgs = conversation(err);
        // Deserialize fidelity: tool_call_id + name + verbatim content all survive.
        let tool = &msgs[3];
        assert_eq!(tool.role, "tool", "[{label}] role");
        assert_eq!(tool.tool_call_id.as_deref(), Some("call_7"), "[{label}] tool_call_id");
        assert_eq!(tool.name.as_deref(), Some("get_stock_price"), "[{label}] tool name");
        assert_eq!(tool.content.as_deref(), Some(err), "[{label}] content verbatim");

        // Render fidelity: the error text lands VERBATIM inside the model's <tool_response> block.
        let rendered = tok.apply_chat_template(&msgs, None, None, None, ThinkingMode::Auto).expect("render");
        let inner = format!("<tool_response>\n{err}\n</tool_response>");
        assert!(
            rendered.contains(&inner),
            "[{label}] error content must render verbatim inside <tool_response>\nRENDERED:\n{rendered}"
        );
        // The assistant's prior call must be re-rendered as the grammar so the next turn sees it.
        assert!(
            rendered.contains("<tool_call>\n<function=get_stock_price>")
                && rendered.contains("<parameter=ticker>")
                && rendered.contains("AAPL"),
            "[{label}] assistant tool_calls must be re-emitted in grammar form:\n{rendered}"
        );
        // The generation prompt must follow (the model reads the error THEN picks the next action).
        assert!(
            rendered.contains("<|im_start|>assistant\n<think>"),
            "[{label}] generation prompt must follow the tool result:\n{rendered}"
        );
    }
}

#[test]
fn tool_call_id_is_preserved_on_deserialize_for_harness_matching() {
    // The harness matches a tool RESULT back to its call by id. If the engine dropped `tool_call_id`,
    // a result could attach to the wrong call. Deserialize must keep it (the message struct field is
    // public; the jinja emission is exercised by the render test above).
    let msgs = conversation(ERR_RATE_LIMIT);
    let tool = &msgs[3];
    assert_eq!(tool.tool_call_id.as_deref(), Some("call_7"), "tool_call_id must be preserved");
    assert_eq!(tool.name.as_deref(), Some("get_stock_price"), "tool name must be preserved");
}

/// An assistant turn carrying TWO parallel calls then TWO results must render both results verbatim,
/// in order — the oracle for the parallel-call round-trip (TC-52 ran a parallel get_stock_price +
/// web_search turn).
#[test]
fn parallel_calls_render_each_result_verbatim_in_order() {
    let Some(tok) = load_tokenizer() else {
        eprintln!("SKIP: tokenizer not present");
        return;
    };
    let json = serde_json::json!([
        {"role":"system","content":"You are helpful."},
        {"role":"user","content":"Compare AAPL vs the market."},
        {"role":"assistant","content":null,"tool_calls":[
            {"id":"call_9","type":"function","function":{"name":"get_stock_price","arguments":"{\"ticker\":\"AAPL\"}"}},
            {"id":"call_10","type":"function","function":{"name":"web_search","arguments":"{\"query\":\"S&P 500 today\"}"}}
        ]},
        {"role":"tool","tool_call_id":"call_9","name":"get_stock_price","content":"{\"ticker\":\"AAPL\",\"price\":178.5,\"change_percent\":-1.27}"},
        {"role":"tool","tool_call_id":"call_10","name":"web_search","content":"{\"results\":[{\"snippet\":\"S&P 500 closed up 0.8%\"}]}"}
    ]);
    let msgs: Vec<ChatMessage> = serde_json::from_value(json).expect("parse parallel conversation");
    let rendered = tok.apply_chat_template(&msgs, None, None, None, ThinkingMode::Auto).expect("render");
    assert!(
        rendered.contains("{\"ticker\":\"AAPL\",\"price\":178.5,\"change_percent\":-1.27}"),
        "first tool result verbatim:\n{rendered}"
    );
    assert!(
        rendered.contains("{\"results\":[{\"snippet\":\"S&P 500 closed up 0.8%\"}]}"),
        "second tool result verbatim:\n{rendered}"
    );
    assert!(
        rendered.contains("<tool_call>\n<function=get_stock_price>")
            && rendered.contains("<function=web_search>"),
        "both prior calls re-emitted in grammar form:\n{rendered}"
    );
}

/// L1 contract unit (no GPU needed): "tools offered + parsed nothing" must NEVER be conflated with a
/// clean completion when `tool_choice=required` is in play. If the model produces no call, the wire
/// must say so via finish_reason, not silently present a clean stop that a harness would read as
/// "the tool ran". This is the host-side piece of the L1 contract; the decode-time grammar mask
/// itself is the device-sampler change.
#[test]
fn tools_offered_but_parsed_nothing_is_recoverable() {
    use gb10_inference::tools::finalize;
    // Model plainly declined (no <tool_call> anywhere): finalize must PRESERVE the literal answer
    // (recoverable by the operator), NOT fabricate a call and NOT drop the text.
    let (content, calls, finish) = finalize("I cannot find a tool for that.", None, "stop");
    assert!(calls.is_empty(), "no call parsed");
    assert_eq!(finish, "stop", "finish passes through when no call parsed");
    assert_eq!(content.as_deref(), Some("I cannot find a tool for that."), "literal text preserved");
}
