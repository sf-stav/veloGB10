//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `nonetools_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    //! S-B9-REL-NONETOOLS: the --keep-tools-when-tool-choice-none contracts.
    //!  * OFF — the decision is the historical one for EVERY input (prompts byte-identical).
    //!  * ON  — a "none" request renders the SAME tools block as auto (same slice), differs
    //!          only by the appended do-not-call instruction (asserted on a Qwen-shaped
    //!          template: the common prefix is the whole conversation, the only difference is
    //!          the inserted line), and its reply is never parsed into tool_calls.
    use super::{tool_choice_decision, partial_overlap, GrowFind, TOOL_OPEN, hold_back_end, apply_force_line};

    fn tools_json() -> serde_json::Value {
        serde_json::json!([
            {"type": "function", "function": {"name": "get_weather",
             "description": "Get the weather for a city",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
            {"type": "function", "function": {"name": "read_file",
             "description": "Read a file from disk",
             "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
        ])
    }

    const DO_NOT_CALL: &str = "IMPORTANT: Do not call any tools. Answer in plain text only.";

    /// Flag OFF: the decision matches the legacy one for every tool_choice mode.
    #[test]
    fn off_flag_decision_matches_legacy_for_every_mode() {
        let tools = tools_json();
        let ts = tools.as_array().unwrap();
        for mode in ["none", "auto", "required"] {
            let v = serde_json::json!(mode);
            let (tft, line, parse) = tool_choice_decision(Some(&v), Some(ts.as_slice()), false);
            match mode {
                "none"     => { assert!(tft.is_none(), "legacy none removes the tools");
                                assert!(line.is_none(), "legacy none forces nothing"); }
                "auto"     => { assert!(tft.is_some()); assert!(line.is_none()); }
                "required" => { assert!(tft.is_some());
                                assert_eq!(line.as_deref(),
                                    Some("IMPORTANT: You MUST use one of the provided tools to answer. Do not answer in plain text.")); }
                _ => unreachable!(),
            }
            assert!(parse, "{mode}: reply parsing stays on with the flag off");
        }
        // named function
        let named = serde_json::json!({"type": "function", "function": {"name": "get_weather"}});
        let (tft, line, parse) = tool_choice_decision(Some(&named), Some(ts.as_slice()), false);
        assert!(tft.is_some());
        assert_eq!(line.as_deref(),
            Some("IMPORTANT: You MUST call the function `get_weather` with appropriate arguments before answering. Do not answer in plain text."));
        assert!(parse);
        // no tool_choice at all
        let (tft, line, parse) = tool_choice_decision(None, Some(ts.as_slice()), false);
        assert!(tft.is_some() && line.is_none() && parse);
        // flag ON but the request is not "none": decision unchanged
        let v = serde_json::json!("auto");
        let (tft, line, parse) = tool_choice_decision(Some(&v), Some(ts.as_slice()), true);
        assert!(tft.is_some() && line.is_none() && parse);
        let v = serde_json::json!("required");
        let (tft, line, parse) = tool_choice_decision(Some(&v), Some(ts.as_slice()), true);
        assert!(tft.is_some() && line.is_some() && parse);
    }

    /// Flag ON + "none" + tools: the tools slice is kept IDENTICAL and the do-not-call line
    /// is the only forcing; the reply is never parsed. "none" WITHOUT tools: unchanged.
    #[test]
    fn on_flag_none_keeps_identical_tools_slice_and_forbids() {
        let tools = tools_json();
        let ts = tools.as_array().unwrap();
        let v = serde_json::json!("none");
        let (tft, line, parse) = tool_choice_decision(Some(&v), Some(ts.as_slice()), true);
        let kept = tft.expect("the tools stay in the template");
        assert!(std::ptr::eq(kept.as_ptr(), ts.as_slice().as_ptr()),
                "the SAME tools slice — nothing reordered or re-serialized");
        assert_eq!(line.as_deref(), Some(DO_NOT_CALL));
        assert!(!parse, "the reply is never parsed into tool_calls");
        // "none" without tools on the request: nothing to keep, nothing to forbid
        let (tft, line, parse) = tool_choice_decision(Some(&v), None, true);
        assert!(tft.is_none() && line.is_none() && parse);
    }

    /// The prefix contract, end to end at the template level: a Qwen-shaped template (the
    /// structural skeleton of the real one — tools render inside the system block at the START)
    /// is fed exactly what the server feeds it (decision outputs + the last-message append).
    /// ON: the none-prompt shares the auto-prompt's whole prefix; removing the single inserted
    /// instruction makes them byte-equal. OFF: the none-prompt diverges at the tools block.
    #[test]
    fn on_flag_none_prompt_shares_the_auto_prefix() {
        let tpl = concat!(
            "{%- if tools %}{{ '<|im_start|>system\\n# Tools\\nYou can call: ' }}",
            "{%- for t in tools %}{{ t.function.name }} {% endfor %}{{ '<|im_end|>\\n' }}{%- endif %}",
            "{%- for m in messages %}{{ '<|im_start|>' }}{{ m.role }}\\n{{ m.content }}{{ '<|im_end|>\\n' }}{%- endfor %}",
            "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}",
        );
        let mut env = minijinja::Environment::new();
        env.add_template("t", tpl).unwrap();
        let render = |messages: &serde_json::Value, tools: Option<&serde_json::Value>| -> String {
            env.get_template("t").unwrap().render(minijinja::context! {
                messages => messages, tools => tools, add_generation_prompt => true,
            }).unwrap()
        };
        let tools = tools_json();
        let ts = tools.as_array().unwrap();
        let mut messages: Vec<serde_json::Value> = vec![
            serde_json::json!({"role": "user", "content": "What is the weather in Paris?"}),
            serde_json::json!({"role": "assistant", "content": "I will check.",
                "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}),
            serde_json::json!({"role": "tool", "content": "18C, clear"}),
            serde_json::json!({"role": "user", "content": "Summarize the conversation so far."}),
        ];

        // AUTO request: tools in the template, no forcing line.
        let prompt_auto = render(&serde_json::json!(messages), Some(&tools));

        // NONE + flag ON: tools stay (decision), the do-not-call line is appended to the LAST
        // message exactly as the server appends it.
        let v_none = serde_json::json!("none");
        let (tft, line, parse) = tool_choice_decision(Some(&v_none), Some(ts.as_slice()), true);
        let tft = tft.expect("tools kept");
        assert_eq!(tft.len(), ts.len());
        assert_eq!(line.as_deref(), Some(DO_NOT_CALL));
        assert!(!parse);
        // S4: drive the SERVER's append (apply_force_line), not a re-implementation — the
        // old body re-implemented the splice, so a handler change could not fail this test.
        let mut typed: Vec<crate::tokenizer::ChatMessage> =
            serde_json::from_value(serde_json::json!(messages)).unwrap();
        apply_force_line(&mut typed, DO_NOT_CALL);
        let prompt_none_on = render(&serde_json::to_value(&typed).unwrap(), Some(&tools));

        // The common prefix is everything up to the inserted instruction, and removing the
        // single inserted segment makes the prompts byte-equal.
        let cp = prompt_none_on.bytes().zip(prompt_auto.bytes())
            .take_while(|(a, b)| a == b).count();
        assert!(cp > prompt_auto.find("Summarize").unwrap(),
                "the shared prefix covers the system block AND the earlier messages");
        let inserted = format!("\n\n{DO_NOT_CALL}");
        assert_eq!(&prompt_none_on[..cp], &prompt_auto[..cp]);
        assert!(prompt_none_on[cp..].starts_with(&inserted),
                "the only difference at the split is the inserted instruction");
        let stripped = format!("{}{}", &prompt_none_on[..cp], &prompt_none_on[cp + inserted.len()..]);
        assert_eq!(stripped, prompt_auto,
                   "removing the single inserted instruction makes the none-prompt byte-equal to auto");

        // NONE + flag OFF (legacy): the tools vanish from the system block, so the divergence
        // starts inside the system head — long before any conversation content.
        let (tft, line, parse) = tool_choice_decision(Some(&v_none), Some(ts.as_slice()), false);
        assert!(tft.is_none() && line.is_none() && parse);
        let messages: Vec<serde_json::Value> = vec![
            serde_json::json!({"role": "user", "content": "What is the weather in Paris?"}),
            serde_json::json!({"role": "assistant", "content": "I will check.",
                "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}),
            serde_json::json!({"role": "tool", "content": "18C, clear"}),
            serde_json::json!({"role": "user", "content": "Summarize the conversation so far."}),
        ];
        let prompt_none_off = render(&serde_json::json!(messages), None);
        assert!(!prompt_none_off.contains("You can call:"),
                "legacy none renders NO tools block");
        let cp_off = prompt_none_off.bytes().zip(prompt_auto.bytes())
            .take_while(|(a, b)| a == b).count();
        assert!(cp_off < prompt_auto.find("You can call:").unwrap(),
                "the legacy none-prompt diverges inside the system head (cache miss from token 0)");
        assert!(cp_off < cp, "flag ON shares a far longer prefix than flag OFF");
    }

    /// The reply contract: a <tool_call> the model emits anyway is a tool call under auto,
    /// and plain content for a keep-tools "none" request (the server skips parse entirely).
    #[test]
    fn on_flag_none_reply_with_tool_call_stays_plain_content() {
        let raw = "Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        let tools = tools_json();
        let ts = tools.as_array().unwrap();
        // auto: the call is extracted and finish becomes tool_calls
        let parsed = crate::tools::parse(raw, Some(ts.as_slice()));
        assert_eq!(parsed.tool_calls.len(), 1, "auto parses the call");
        let (content, calls, finish) = crate::tools::finalize_parsed(raw, parsed, "stop");
        assert_eq!(content.as_deref(), Some("Let me check."), "prose before the call stays content");
        assert_eq!(calls.len(), 1);
        assert_eq!(finish, "tool_calls");
        // keep-tools "none": the server constructs the empty ParsedOutput — the raw text is
        // returned unchanged as content and the finish stays the plain stop/length.
        let parsed = crate::tools::ParsedOutput { content: raw.trim().to_string(), tool_calls: Vec::new() };
        let (content, calls, finish) = crate::tools::finalize_parsed(raw, parsed, "stop");
        assert_eq!(content.as_deref(), Some(raw), "the stray call stays plain content");
        assert!(calls.is_empty());
        assert_eq!(finish, "stop");
    }

    /// The streaming hold-back: the needle is None for a keep-tools "none" request, so a
    /// stray <tool_call> streams through as content; legacy keeps the hold-back.
    #[test]
    fn hold_back_disabled_for_keep_tools_none() {
        // S4 rewrite: drives the REAL hold_back_end (the old body asserted
        // `None.and_then(..)`, which cannot fail).
        let acc = "answer text <tool_call>\n<function=x>\n</function>\n</tool_call>";
        // keep-tools "none": tool_hold = None — the stray call streams through as content
        let none: Option<&'static str> = None;
        assert_eq!(hold_back_end(none, none.and_then(|_| acc.find(TOOL_OPEN)), acc.len(), acc, &["<think"]),
                   acc.len(), "with tool parsing off NOTHING is held back for tools");
        // legacy: tool_hold = Some(TOOL_OPEN) — emission stops AT the marker
        let legacy: Option<&'static str> = Some(TOOL_OPEN);
        assert_eq!(hold_back_end(legacy, legacy.and_then(|_| acc.find(TOOL_OPEN)), acc.len(), acc, &["<think"]),
                   "answer text ".len(), "legacy stops at the call");
        // a partial marker at the tail is held back so a split <tool_c cannot leak
        let tail = "answer <tool_c";
        assert_eq!(hold_back_end(legacy, legacy.and_then(|_| tail.find(TOOL_OPEN)), tail.len(), tail, &["<think"]),
                   "answer ".len(), "the partial marker prefix is not emitted");
    }

    /// S3: the same-chunk think-close decision — with the flag ON (tool_hold None) a stray
    /// `<tool_call>` arriving in the same chunk as `</think>` is NOT held back (was: held).
    #[test]
    fn think_close_same_chunk_uses_the_hold_back_decision() {
        let acc = "</think> sure, calling <tool_call>\n<function=x>";
        let lead = "</think> ".len(); // what the branch computes: past the close tag + lead ws
        let region = &acc[lead..];
        let hit = |hold: Option<&'static str>| hold.and_then(|_| region.find(TOOL_OPEN).map(|i| lead + i));
        // keep-tools "none": the whole chunk streams as content
        assert_eq!(hold_back_end(None, hit(None), acc.len(), region, &["<think"]), acc.len(),
                   "S3: with tool parsing off the same-chunk call is plain content");
        // legacy: held at the call, exactly like the steady-state content branch
        let legacy: Option<&'static str> = Some(TOOL_OPEN);
        assert_eq!(hold_back_end(legacy, hit(legacy), acc.len(), region, &["<think"]),
                   lead + "sure, calling ".len(), "legacy holds at the call");
    }
