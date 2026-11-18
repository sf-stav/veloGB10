//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `gen_line_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::*;

    /// (c) a loop-detector stop is `finish_reason: "stop"` for the client and `stop_kind=loop` in the log.
    #[test]
    fn stop_kind_names_the_cause() {
        assert_eq!(stop_kind(Some("stop"), false), "eos");
        assert_eq!(stop_kind(Some("length"), false), "length");
        assert_eq!(stop_kind(Some("context_length_exceeded"), false), "length");
        assert_eq!(stop_kind(Some("loop_detected"), false), "loop");
        assert_eq!(stop_kind(Some("cancelled"), false), "cancel");
        assert_eq!(stop_kind(Some("error: CUDA_ERROR_ILLEGAL_ADDRESS"), false), "error");
        assert_eq!(stop_kind(None, false), "error", "a channel closed without a Finish is an engine failure");
        assert_eq!(stop_kind(Some("capacity"), false), "other");
        // a stop string is matched on the HTTP side: it wins whatever the scheduler said (or did not)
        for r in [None, Some("stop"), Some("length"), Some("loop_detected")] {
            assert_eq!(stop_kind(r, true), "stop_string", "{r:?}");
        }
        // the client-facing mapping is unchanged: a loop stop is still a plain "stop" + stop_reason
        assert_eq!(spec_finish_reason("loop_detected"), "stop");
        assert_eq!(stop_reason_of("loop_detected").as_deref(), Some("loop_detected"));
        assert_eq!(stop_reason_of("stop"), None);
    }

    /// (b) a request that ends inside the think block its prompt opened: the text is reasoning.
    #[test]
    fn an_unclosed_reasoning_block_is_counted_as_reasoning() {
        let (open, close) = ("<think>", "</think>");
        // unclosed (e.g. max_tokens reached while thinking): split_think calls it content
        let text = "Let me work through this step by step";
        let (r, c) = split_think(text, open, close);
        assert_eq!((r.as_deref(), c.as_str()), (None, text), "split_think: no close tag = all content");
        assert_eq!(gen_log_spans(r.as_deref(), &c, text.contains(close), true), (Some(text), ""),
                   "the prompt opened the block and it never closed: reasoning");
        // the same text when the prompt did NOT open a think block (thinking off): content
        assert_eq!(gen_log_spans(r.as_deref(), &c, false, false), (None, text));
        // a closed block is untouched, with or without an answer after it
        let full = "plan the answer</think>\n\nThe answer is 4.";
        let (r, c) = split_think(full, open, close);
        assert_eq!(gen_log_spans(r.as_deref(), &c, full.contains(close), true), (Some("plan the answer"), "The answer is 4."));
        let only = "plan the answer</think>";
        let (r, c) = split_think(only, open, close);
        assert_eq!(gen_log_spans(r.as_deref(), &c, true, true), (Some("plan the answer"), ""));
        // nothing generated: nothing moves
        assert_eq!(gen_log_spans(None, "", false, true), (None, ""));
    }

    /// SURFACE2 item 8: a non-streaming generation that ends inside the think block the prompt
    /// opened. The canned stream is decoded text, token by token; the request ran out of budget
    /// before any close tag.
    #[test]
    fn a_generation_that_ends_inside_the_think_block_is_reasoning_not_content() {
        let (open, close) = ("<think>", "</think>");
        let toks = ["The", " user", " asks", " for", " 17", " *", " 23", ".", " 17", " *", " 20", " =", " 340", ","];
        let text: String = toks.concat();
        // BEFORE (reproduction): the non-streaming split hands the whole reasoning back as content
        let (r_old, c_old) = split_think(&text, open, close);
        assert_eq!((r_old, c_old.as_str()), (None, text.as_str()), "split_think alone: reasoning lost, content = the reasoning");
        // NOW: reasoning_content carries it and content is empty, at every length of the stream
        for n in 1..=toks.len() {
            let t: String = toks[..n].concat();
            assert!(!t.contains(close));
            let (r, c) = unclosed_think_split(&t, true);
            assert_eq!(r.as_deref(), Some(t.trim()), "after {n} tokens");
            assert_eq!(c, "", "content stays empty after {n} tokens");
            // the message as the response builder completes it: no call parsed, finish passes through
            let parsed = crate::tools::parse(&c, None);
            let (content, calls, finish) = crate::tools::finalize_parsed(&c, parsed, "length");
            assert_eq!((content.as_deref(), calls.len(), finish.as_str()), (Some(""), 0, "length"));
            // and the [gen] line counts the same spans as before the change
            assert_eq!(gen_log_spans(r.as_deref(), &c, false, true), (Some(t.trim()), ""));
        }
        // nothing generated at all: no reasoning, empty content
        assert_eq!(unclosed_think_split("", true), (None, String::new()));
        assert_eq!(unclosed_think_split(" \n", true), (None, String::new()));

        // a tool call made without closing the block (the model jumps from its thoughts straight to
        // the call): the thoughts are reasoning, the call is parsed from the rest — the stream's
        // hold-back at the marker, on the finished text
        let call = "I need the weather.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nAthens\n</parameter>\n</function>\n</tool_call>";
        let (r, c) = unclosed_think_split(call, true);
        assert_eq!(r.as_deref(), Some("I need the weather."));
        assert!(c.starts_with("<tool_call>"), "{c}");
        let parsed = crate::tools::parse(&c, None);
        let (content, calls, finish) = crate::tools::finalize_parsed(&c, parsed, "stop");
        assert_eq!((content, calls.len(), finish.as_str()), (None, 1, "tool_calls"));
        assert_eq!(calls[0].function.name, "get_weather");
        // the same calls as parsing the whole text (what the old split fed the parser)
        let whole = crate::tools::parse(call, None);
        assert_eq!(whole.tool_calls.len(), 1);
        assert_eq!(whole.tool_calls[0].function.arguments, calls[0].function.arguments);
        // a call-looking block the parser cannot read stays visible as content (never dropped)
        let broken = "Thinking.\n<tool_call>\nnot a call";
        let (r, c) = unclosed_think_split(broken, true);
        assert_eq!((r.as_deref(), c.as_str()), (Some("Thinking."), "<tool_call>\nnot a call"));
        let (content, calls, _) = crate::tools::finalize_parsed(&c, crate::tools::parse(&c, None), "length");
        assert_eq!((content.as_deref(), calls.len()), (Some("<tool_call>\nnot a call"), 0));
        // tool parsing off for the request (tool_choice none): the marker is ordinary reasoning text
        let (r, c) = unclosed_think_split(call, false);
        assert_eq!((r.as_deref(), c.as_str()), (Some(call), ""));

        // NARROW: the response builder takes this path only when the prompt opened the block AND
        // no close tag was generated; --unclosed-think content restores the old bytes
        let src = src_text!("server");
        let guard = ["if prompt_opened_think && !think_closed && !unclosed_think_as_content() {\n", "            unclosed_think_split("].concat();
        assert_eq!(src.matches(guard.as_str()).count(), 1, "one call site, behind the narrow guard");
        // ...and "the prompt opened the block" is read from the rendered prompt's tail alone (the
        // effort arm of the streaming rule can be true for a prompt that CLOSED the block)
        let def = ["let prompt_opened_think = ", "prompt.trim_end().ends_with(think_open);"].concat();
        assert_eq!(src.matches(def.as_str()).count(), 1);
        let (open_tail, closed_tail) = ("<|im_start|>assistant\n<think>\n", "<|im_start|>assistant\n<think>\n\n</think>\n\n");
        assert!(open_tail.trim_end().ends_with(open) && !closed_tail.trim_end().ends_with(open));
        let o = &crate::opts::REG[crate::opts::idx("unclosed-think")];
        // the default is an owner ruling (2026-10-09: `reasoning`): the row's `def` is the one
        // place that states it; here and `opts::tests::choice_reads_the_row_default`
        assert_eq!((o.def, o.env), ("reasoning", ""), "a CLI option, no environment variable");
        if crate::opts::get(crate::opt!("unclosed-think")).is_none() { assert!(!unclosed_think_as_content()); }
        assert!(o.help.contains("v0.7.3 and earlier") && o.help.contains("Streaming responses"), "{}", o.help);
        // a closed block is untouched: split_think still decides
        let (r, c) = split_think("plan</think>\n\nThe answer is 391.", open, close);
        assert_eq!((r.as_deref(), c.as_str()), (Some("plan"), "The answer is 391."));
    }

    /// (a) the tail of the line.
    #[test]
    fn stop_info_fields_format() {
        assert_eq!(stop_info_fields(Some((0.97314, 1, false))), "stop_p=0.9731 stop_rank=1 stop_p_law=greedy");
        assert_eq!(stop_info_fields(Some((0.0, 4821, true))), "stop_p=0.0000 stop_rank=4821 stop_p_law=sampled");
        assert_eq!(stop_info_fields(None), "stop_p=na stop_rank=na stop_p_law=na");
        // SURFACE2 item 1: the law tag is APPENDED — a script that reads `stop_p=` / `stop_rank=`
        // (key followed by `=`) finds the same two fields at the same place as before.
        for info in [Some((0.5f32, 2u32, true)), Some((1.0, 1, false)), None] {
            let s = stop_info_fields(info);
            let keys: Vec<&str> = s.split(' ').map(|kv| kv.split('=').next().unwrap()).collect();
            assert_eq!(keys, ["stop_p", "stop_rank", "stop_p_law"]);
            assert!(s.starts_with("stop_p=") && s.matches("stop_p=").count() == 1, "`stop_p=` still matches one field: {s}");
        }
    }

    /// The three fields are APPENDED to the line (scripts parse the existing keys in order).
    #[test]
    fn the_gen_line_keeps_its_key_order_and_appends_the_new_fields() {
        let src = src_text!("server");
        let at = |pat: &str| src.find(pat).unwrap_or_else(|| panic!("missing `{pat}`"));
        let a = at("crate::dreprintln!(\"[gen] id={} finish={} terminal_tok={} is_stop_tok={} reasoning_tokens={}");
        let mut end = a + 600;
        while !src.is_char_boundary(end) { end -= 1; } // OUT2: 1-byte rename shifted this onto an em-dash
        let tail = &src[a..end];
        let pos = |k: &str| tail.find(k).unwrap_or_else(|| panic!("[gen] line lost `{k}`"));
        let order = ["content_tokens={}", "completion_tokens={}", "prompt_tokens={}", "tool_calls={}", "tools_offered={}",
                     "effort={}", "presence_penalty={}", "pp_source={}", "stop_kind={}"];
        for w in order.windows(2) { assert!(pos(w[0]) < pos(w[1]), "{} must precede {}", w[0], w[1]); }
        assert!(tail.contains("pp_source={} stop_kind={} {}\","), "stop_kind and the stop_p/stop_rank fields close the line");
    }
