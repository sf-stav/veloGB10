// hy_v3's tokenizer.json uses the pair-array merges form ([["a","b"],...]) which tokenizers
// 0.19's BPE deserializer rejects; the engine upgrades it transparently in load_tokenizer.
use gb10_inference::tokenizer::{ChatMessage, QwenTokenizer};

#[test]
fn load_hy3_tokenizer_via_engine() {
    let tok = QwenTokenizer::from_file("/mnt/models/hy3-nvfp4/tokenizer.json")
        .expect("hy3 tokenizer must load (pair-merge upgrade)");
    let ids = tok.encode("The history of the railway", true).expect("encode");
    assert_eq!(ids[..4], [628, 4043, 279, 252]);   // oracle p1 prefix

    // Stop-token union: generation_config (120025) + config.eod_token_id (120026) + hy_EOT (120008).
    let stops = tok.stop_token_ids(120025);
    for want in [120025u32, 120026, 120008] {
        assert!(stops.contains(&want), "stop union missing {want}: {stops:?}");
    }
    println!("hy3 tokenizer OK; stop union {stops:?}");
}

#[test]
fn hy3_chat_template_renders_with_tools() {
    let tok = QwenTokenizer::from_file("/mnt/models/hy3-nvfp4/tokenizer.json").unwrap();
    let msgs = vec![
        ChatMessage { role: "system".into(), content: Some("You are helpful.".into()),
                      tool_calls: None, tool_call_id: None, name: None, reasoning_content: None },
        ChatMessage::user("What is 2+2?"),
    ];
    let plain = tok.apply_chat_template(&msgs, None, None).expect("template without tools");
    assert!(!plain.contains("{}"), "Python .format must interpolate (no literal braces): {plain}");
    assert!(plain.contains("<｜hy_begin_of_sentence:opensource｜>"),
            "the :opensource suffix must be formatted in: {plain}");
    println!("--- plain ---\n{plain}");

    // hy_v3 optional reasoning: effort must reach the template ('no_think' default vs 'high').
    let thinking = tok.apply_chat_template(&msgs, None, Some("high")).expect("template with effort");
    assert!(thinking.contains("reasoning_effort:high"),
            "reasoning_effort=high must render into the prompt: {thinking}");

    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {"name": "calc", "description": "calculator",
                     "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}}
    })];
    let with_tools = tok.apply_chat_template(&msgs, Some(&tools), None)
        .expect("template WITH tools (tojson/raise_exception path)");
    assert!(with_tools.contains("calc"), "tool block must render the tool name");
    println!("--- with tools (first 500 chars) ---\n{}", &with_tools[..with_tools.len().min(500)]);
    // the rendered prompt must tokenize (specials are in the vocab)
    let ids = tok.encode(&with_tools, true).expect("encode rendered prompt");
    assert!(!ids.is_empty());
}
