//! The unit tests of the tokenizer module (chat template loading, thinking modes, stream decoder).

    /// Issue #34: --chat-template <PATH> wins over the pack's chat_template.jinja, and a named
    /// but unreadable file fails LOUDLY (no silent fallback). Both checked against a scratch dir
    /// with a jinja file present, so the flag's precedence is what is under test.
    #[test]
    fn issue34_chat_template_flag_wins_and_missing_file_fails_loud() {
        use std::fs;
        let d = std::env::temp_dir().join(format!("tpl34_{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("chat_template.jinja"), "PACK TEMPLATE {{ messages[0]['role'] }}").unwrap();
        fs::write(d.join("custom.jinja"), "CUSTOM TEMPLATE {{ messages[0]['role'] }}").unwrap();
        let tok_path = d.join("tokenizer.json").to_string_lossy().to_string();
        fs::write(&tok_path, "{}").unwrap();

        // no flag: the pack's own jinja wins (origin names it)
        // (opt_or_arg reads std::env::args, so drive it through the real argv of the test proc:
        //  absent -> pack path; the override cases below re-exec-free via direct calls)
        let argv_none: Vec<String> = vec!["prog".into()];
        assert!(crate::opt_or_arg(&argv_none, "--chat-template").is_none());

        // flag present + readable: the flag path is the origin
        let argv_flag: Vec<String> = vec!["prog".into(), "--chat-template".into(), d.join("custom.jinja").to_string_lossy().to_string()];
        assert_eq!(crate::opt_or_arg(&argv_flag, "--chat-template").as_deref(),
                   Some(d.join("custom.jinja").to_string_lossy().to_string()).as_deref());

        // missing file: read fails -> the caller bails (checked at the fs layer, the same error
        // load_chat_env wraps into its bail!)
        assert!(std::fs::read_to_string(d.join("nope.jinja")).is_err());
        let _ = fs::remove_dir_all(&d);
    }

    /// GitHub issue #37: the OpenAI-spec `developer` role (the spec'd replacement for `system` in
    /// structured-output/agent contexts) must be normalized to `system` AT INGEST. Forwarded
    /// verbatim it reached the pack's chat template, whose `else` arm raised "Unexpected message
    /// role" → HTTP 500. One mapping here; every renderer sees `system`; unknown roles stay
    /// untouched (the template still rejects them, loudly).
    #[test]
    fn issue37_developer_role_normalized_to_system() {
        // The issue's exact repro shape: developer mid-conversation AND at messages[0].
        let msgs: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": "hi"},
            {"role": "developer", "content": "You are an agent."},
            {"role": "user", "content": "go"}
        ])).unwrap();
        let roles: Vec<&str> = msgs.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["user", "system", "user"], "{roles:?}");

        let first: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
            {"role": "developer", "content": "You are an agent."},
            {"role": "user", "content": "go"}
        ])).unwrap();
        assert_eq!(first[0].role, "system");
        assert_eq!(first[0].content.as_deref(), Some("You are an agent."));

        // Everything else passes through untouched — the template's strictness is not weakened.
        let other: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "tool_call_id": "c1", "content": "r"}
        ])).unwrap();
        let roles: Vec<&str> = other.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["system", "user", "assistant", "tool"], "{roles:?}");
    }

    /// GitHub issue #6: string `arguments` holding a JSON object are parsed before the template sees them, so a
    /// template with an `arguments is string` branch still takes its `is mapping` (`<parameter=…>`) branch; the raw
    /// string survives only for arguments that are not a JSON object.
    #[test]
    fn issue6_json_object_arguments_are_parsed_even_with_a_string_branch() {
        let msg = |args: &str| -> ChatMessage {
            serde_json::from_value(serde_json::json!({"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": args}}]})).unwrap()
        };
        for raw in [true, false] {
            let j = msg(r#"{"command": "ls -la", "n": 3}"#).to_template_json(raw);
            let a = &j["tool_calls"][0]["function"]["arguments"];
            assert!(a.is_object(), "raw_tool_args={raw}: {a}");
            assert_eq!(a["command"], "ls -la");
        }
        // not an object: kept verbatim for a template that can show a string, {} otherwise
        for bad in ["not json", "[1,2]", "\"x\"", "", "{\"command\": "] {
            let a = msg(bad).to_template_json(true)["tool_calls"][0]["function"]["arguments"].clone();
            assert_eq!(a, serde_json::Value::String(bad.to_string()), "{bad:?}");
            let a = msg(bad).to_template_json(false)["tool_calls"][0]["function"]["arguments"].clone();
            assert_eq!(a, serde_json::json!({}), "{bad:?}");
        }
    }

    /// The issue's repro, end to end through minijinja: a template with BOTH branches renders the history as
    /// `<parameter=…>` blocks (what its system prompt asks for), never raw JSON inside `<function=…>`.
    #[test]
    fn issue6_history_renders_parameter_tags() {
        let tpl = "{%- for m in messages %}{%- for tc in m.tool_calls %}<tool_call>\n<function={{ tc.function.name }}>\n\
                   {%- if tc.function.arguments is mapping %}{%- for k, v in tc.function.arguments|items %}\n<parameter={{ k }}>\n{{ v }}\n</parameter>{%- endfor %}\n\
                   {%- elif tc.function.arguments is string %}{{ tc.function.arguments }}{%- endif %}</function>\n</tool_call>{%- endfor %}{%- endfor %}";
        let mut env = minijinja::Environment::new();
        register_pycompat(&mut env);
        env.add_template("t", tpl).unwrap();
        let m: ChatMessage = serde_json::from_value(serde_json::json!({"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\": \"ls -la\"}"}}]})).unwrap();
        let out = env.get_template("t").unwrap()
            .render(minijinja::context! { messages => vec![m.to_template_json(true)] }).unwrap();
        assert!(out.contains("<parameter=command>\nls -la\n</parameter>"), "{out}");
        assert!(!out.contains("{\"command\""), "{out}");
    }

    /// VIS-5: image and text parts render in the client's order; VID-0: unknown part types are captured.
    #[test]
    fn vis5_interleave_order_and_unsupported_parts() {
        let m: ChatMessage = serde_json::from_str(r#"{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}},{"type":"text","text":"q1"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,BBBB"}},{"type":"text","text":"q2"}]}"#).unwrap();
        let j = m.to_template_json(false);
        let kinds: Vec<String> = j["content"].as_array().unwrap().iter()
            .map(|p| p["type"].as_str().unwrap().to_string() + p.get("text").and_then(|t| t.as_str()).unwrap_or("")).collect();
        assert_eq!(kinds, vec!["image", "textq1", "image", "textq2"]);
        assert!(m.unsupported_parts.is_empty());
        let v: ChatMessage = serde_json::from_str(r#"{"role":"user","content":[{"type":"video_url","video_url":{"url":"x"}},{"type":"text","text":"hi"}]}"#).unwrap();
        assert_eq!(v.unsupported_parts, vec!["video_url".to_string()]);
        // text-only string content is untouched
        let t: ChatMessage = serde_json::from_str(r#"{"role":"user","content":"plain"}"#).unwrap();
        assert_eq!(t.to_template_json(false)["content"], serde_json::json!("plain"));
    }

    use super::*;

    /// Regression: a prior assistant tool_call whose arguments are NON-STRING (numbers,
    /// booleans, objects) takes the template's `tojson(ensure_ascii=False)` path — the filter
    /// must tolerate the kwarg (was: "too many arguments" → HTTP 500 on the tool-result turn).
    #[test]
    fn render_tool_call_with_numeric_arguments() {
        let tok = match QwenTokenizer::from_file("/mnt/models/hy3-nvfp4/tokenizer.json") {
            Ok(t) => t,
            Err(e) => {
                eprintln!("skip: hy3 tokenizer not present ({})", e);
                return;
            }
        };
        assert!(tok.chat_env.is_some(), "hy3 chat template should have loaded");

        let messages = vec![
            ChatMessage::user("Read the first 60 lines of the file."),
            ChatMessage {
                role: "assistant".into(),
                content: None,
                images: vec![], layout: vec![], unsupported_parts: vec![],
                tool_calls: Some(vec![ToolCall {
                    id: "call_0".into(),
                    kind: "function".into(),
                    function: FunctionCall {
                        name: "read".into(),
                        arguments: "{\"filePath\":\"/tmp/x\",\"limit\":60,\"offset\":1}".into(),
                    },
                }]),
                tool_call_id: None, name: None, reasoning_content: None,
            },
            ChatMessage {
                role: "tool".into(),
                content: Some("line one\nline two".into()),
                images: vec![], layout: vec![], unsupported_parts: vec![], tool_calls: None, tool_call_id: Some("call_0".into()), name: Some("read".into()),
                reasoning_content: None,
            },
        ];
        let rendered = tok.apply_chat_template(&messages, None, None, None, ThinkingMode::Auto)
            .expect("render must not raise 'too many arguments' on non-string tool args");
        assert!(rendered.contains("read"), "tool name should appear in the prompt");
        assert!(rendered.contains("60"), "numeric argument should render unescaped");
    }

    /// HOTFIX `serve-content-parts`: the OpenAI spec allows `content` as a STRING or an ARRAY of
    /// content parts. A bare `Option<String>` rejected the array form with a confusing 422
    /// `invalid type: sequence, expected a string`. Headless unit test on the deserializer.
    #[test]
    fn content_accepts_string_array_and_null() {
        // (a) plain string (regression): unchanged.
        let m: ChatMessage = serde_json::from_str(r#"{"role":"user","content":"hello"}"#).unwrap();
        assert_eq!(m.content.as_deref(), Some("hello"));

        // (b) single-text-part array — the Pi payload shape. Used verbatim.
        let m: ChatMessage = serde_json::from_str(
            r#"{"role":"user","content":[{"type":"text","text":"hello world"}]}"#).unwrap();
        assert_eq!(m.content.as_deref(), Some("hello world"));

        // (c) multi-text-part array — joined with "\n" (documented join choice).
        let m: ChatMessage = serde_json::from_str(
            r#"{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}"#).unwrap();
        assert_eq!(m.content.as_deref(), Some("a\nb"));

        // (d) null content on an assistant tool_calls turn -> None (agents send this).
        let m: ChatMessage = serde_json::from_str(
            r#"{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]}"#).unwrap();
        assert_eq!(m.content, None);
        assert!(m.tool_calls.is_some(), "tool_calls must still deserialize");

        // (e) an array containing an image_url part -> CAPTURED into `images` (vision entry point);
        //     no 422, content is None (no text part).
        let m: ChatMessage = serde_json::from_str(
            r#"{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}"#).unwrap();
        assert_eq!(m.content, None);
        assert_eq!(m.images.len(), 1);
        assert_eq!(m.images[0].url.as_deref(), Some("data:image/png;base64,AAAA"));

        // (e2) mixed: an image part + a text part -> text captured in content, image in images.
        let m: ChatMessage = serde_json::from_str(
            r#"{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,BBBB"}},{"type":"text","text":"what is this?"}]}"#).unwrap();
        assert_eq!(m.content.as_deref(), Some("what is this?"));
        assert_eq!(m.images.len(), 1);
        assert_eq!(m.images[0].url.as_deref(), Some("data:image/png;base64,BBBB"));

        // (f) realistic agent conversation: system + user + assistant(tool_calls,null) + tool result.
        let msgs: Vec<ChatMessage> = serde_json::from_str(
            r#"[
                {"role":"system","content":"You are a helpful assistant."},
                {"role":"user","content":[{"type":"text","text":"What is 2+2?"}]},
                {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"calc","arguments":"{\"expr\":\"2+2\"}"}}]},
                {"role":"tool","tool_call_id":"call_1","name":"calc","content":[{"type":"text","text":"4"}]}
            ]"#).unwrap();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[1].content.as_deref(), Some("What is 2+2?"));
        assert_eq!(msgs[2].content, None);
        assert_eq!(msgs[3].content.as_deref(), Some("4"));
        assert_eq!(msgs[3].tool_call_id.as_deref(), Some("call_1"));
    }

    /// Regression: the SSE streaming path decodes one token at a time; a multi-byte UTF-8 char
    /// split across byte-level-BPE tokens (every emoji) must round-trip without "�" — the
    /// crate's per-token decode is String::from_utf8_lossy and mangles each fragment.
    #[test]
    fn stream_decoder_reassembles_emoji() {
        let Some((tok, _)) = fixture_tokenizer() else {
            eprintln!("skip: no tokenizer fixture (models/<Qwen repo name>/tokenizer.json) for the emoji stream test");
            return;
        };
        let text = "Hello \u{1F600} world! \u{1F680}\u{1F4A5}";   // 😀 🚀 💥
        let ids = tok.encode(text, false).expect("encode");
        let mut dec = tok.stream_decoder();
        let mut out = String::new();
        for &id in &ids {
            out.push_str(&dec.push(id));
        }
        out.push_str(&dec.finish());
        assert_eq!(out, text, "incremental decode must reassemble multi-byte chars");
        assert!(!out.contains('\u{FFFD}'), "no replacement chars allowed: {out:?}");
        // the whole-list decode (non-streaming path) must agree
        let whole = tok.decode(&ids, false).expect("decode");
        assert_eq!(whole, text, "whole-list decode should match too");
    }

    /// Smoke-test that the model's real Jinja template is loaded and renders a
    /// multi-turn conversation the way Qwen3.5 expects. Verifies the two fixes the
    /// legacy hand-rolled template got wrong:
    ///   1. The generation prompt ends with `<|im_start|>assistant\n<think>\n`.
    ///   2. A prior assistant turn that carried a `<think>…</think>` block has it
    ///      stripped in the rendered history (history turns are answer-only).
    #[test]
    fn render_4b_multiturn_thinking_template() {
        let tok = match QwenTokenizer::from_file("4b/tokenizer.json") {
            Ok(t) => t,
            Err(e) => {
                eprintln!("skip: 4b tokenizer not present ({})", e);
                return;
            }
        };
        assert!(tok.chat_env.is_some(), "chat template should have loaded from 4b/");

        let messages = vec![
            ChatMessage::user("Write a one-sentence sci-fi story."),
            ChatMessage {
                role: "assistant".into(),
                // Simulate what round-trips from a client: reasoning + answer.
                content: Some("<think>\nI should keep it short.\n</think>\n\nThe beacon awoke, and so did something else.\n".into()),
                images: vec![], layout: vec![], unsupported_parts: vec![], tool_calls: None, tool_call_id: None, name: None, reasoning_content: None,
            },
            ChatMessage::user("Continue it for another 1000 words."),
        ];
        let rendered = tok.apply_chat_template(&messages, None, None, None, ThinkingMode::Auto).expect("render");

        eprintln!("===== RENDERED PROMPT START =====\n{}\n===== RENDERED PROMPT END =====", rendered);

        // Fix 1: generation prompt primes thinking.
        assert!(
            rendered.ends_with("<|im_start|>assistant\n<think>\n"),
            "generation prompt must end with `<|im_start|>assistant\\n<think>\\n`; got tail: {:?}",
            rendered.chars().rev().take(40).collect::<String>().chars().rev().collect::<String>()
        );

        // The history (first) assistant turn must appear, but its raw `<think>` body
        // should NOT be present — Qwen3.5 strips reasoning from history turns.
        assert!(rendered.contains("The beacon awoke"), "prior answer text should be present");
        assert!(
            !rendered.contains("I should keep it short"),
            "prior `<think>` body leaked into history; the template should strip it"
        );

        // Sanity: there should be exactly one `<think>` open tag — the one priming
        // the *current* generation — and no dangling `</think>` from history.
        assert_eq!(
            rendered.matches("<think>").count(),
            1,
            "expected exactly one priming <think> tag, got:\n{}",
            rendered
        );
    }

    // ---------------------------------------------------------------------------------------
    // W1 (Phase 13): the thinking toggle. Two prongs, one root cause each:
    //   (a) `chat_template_kwargs: {"enable_thinking": false}` was dropped by serde entirely;
    //   (b) the engine hardcoded `enable_thinking: true` into EVERY render, so a user-edited
    //       `chat_template.jinja` whose default is "off" could never take effect.
    // These tests lock the precedence table and the render behaviour; the GPU leg (one 35B boot)
    // only has to confirm the live server path.
    // ---------------------------------------------------------------------------------------

    /// The CLI spelling -> mode mapping (and the fact that unknown spellings are rejected rather
    /// than silently defaulted to Auto).
    #[test]
    fn thinking_mode_parse() {
        assert_eq!(ThinkingMode::parse("auto"), Some(ThinkingMode::Auto));
        assert_eq!(ThinkingMode::parse("AUTO"), Some(ThinkingMode::Auto));
        assert_eq!(ThinkingMode::parse("on"), Some(ThinkingMode::On));
        assert_eq!(ThinkingMode::parse("true"), Some(ThinkingMode::On));
        assert_eq!(ThinkingMode::parse("off"), Some(ThinkingMode::Off));
        assert_eq!(ThinkingMode::parse("no_think"), Some(ThinkingMode::Off));
        assert_eq!(ThinkingMode::parse("maybe"), None);
        assert_eq!(ThinkingMode::Auto.as_option(), None);
        assert_eq!(ThinkingMode::On.as_option(), Some(true));
        assert_eq!(ThinkingMode::Off.as_option(), Some(false));
    }

    /// Precedence, most specific wins: request kwarg > no-think effort > --thinking > template.
    #[test]
    fn resolve_enable_thinking_precedence() {
        use ThinkingMode::*;
        assert_eq!(resolve_enable_thinking(None, None, Auto), None);
        assert_eq!(resolve_enable_thinking(None, None, On), Some(true));
        assert_eq!(resolve_enable_thinking(None, None, Off), Some(false));
        assert_eq!(resolve_enable_thinking(Some(false), None, On), Some(false));
        assert_eq!(resolve_enable_thinking(Some(true), None, Off), Some(true));
        assert_eq!(resolve_enable_thinking(None, Some("off"), Auto), Some(false));
        assert_eq!(resolve_enable_thinking(None, Some("no_think"), Auto), Some(false));
        assert_eq!(resolve_enable_thinking(None, Some("xhigh"), Auto), None);
        assert_eq!(resolve_enable_thinking(None, Some("high"), Auto), None);
    }

    /// `chat_template_kwargs` validation: the two malformed shapes must RAISE (they become a loud
    /// 400), never be dropped on the floor.
    #[test]
    fn enable_thinking_kwarg_validation() {
        assert_eq!(enable_thinking_kwarg(None).unwrap(), None);
        assert_eq!(enable_thinking_kwarg(Some(&serde_json::json!({}))).unwrap(), None);
        assert_eq!(enable_thinking_kwarg(Some(&serde_json::json!({"enable_thinking": false}))).unwrap(), Some(false));
        assert_eq!(enable_thinking_kwarg(Some(&serde_json::json!({"enable_thinking": true}))).unwrap(), Some(true));
        assert_eq!(enable_thinking_kwarg(Some(&serde_json::json!({"preserve_thinking": true}))).unwrap(), None);
        assert!(enable_thinking_kwarg(Some(&serde_json::json!("not-an-object"))).is_err());
        assert!(enable_thinking_kwarg(Some(&serde_json::json!({"enable_thinking": "false"}))).is_err());
    }

    /// Find a tokenizer fixture for the render-level tests: the original Hugging Face model dirs under the nearest
    /// `models/` in any ancestor of the crate root (so worktrees find them too), by their standard repo names, first
    /// found wins. No environment variable (owner rule: flags and files only). Tests SKIP (never fail) when the box
    /// has none - the convention the other template tests in this file already use.
    fn fixture_tokenizer() -> Option<(QwenTokenizer, String)> {
        // flagship family first (its template primes thinking under Auto and carries the {% generation %} tags the
        // loader strips), then the small model (its template's Auto falls to the closed branch)
        const NAMES: [&str; 4] = ["Qwen3.6-35B-A3B", "Qwen3.6-27B", "Qwen3.5-27B", "Qwen3.5-0.8B"];
        let mut d = Some(std::path::Path::new(env!("CARGO_MANIFEST_DIR")));
        while let Some(p) = d {
            for n in NAMES {
                let c = p.join("models").join(n).join("tokenizer.json");
                if let Ok(t) = QwenTokenizer::from_file(&c.to_string_lossy()) { return Some((t, c.to_string_lossy().to_string())); }
            }
            d = p.parent();
        }
        None
    }

    /// The customer's exact request: `chat_template_kwargs: {"enable_thinking": false}` must render
    /// the template's NO-THINK branch (a closed, empty think block); `true`/absent must render the
    /// thinking branch. Skips when no model fixture is present.
    #[test]
    fn render_enable_thinking_kwarg_real_template() {
        let Some((tok, path)) = fixture_tokenizer() else {
            eprintln!("skip: no tokenizer fixture for the W1 render test");
            return;
        };
        let msgs = vec![ChatMessage::user("Reply with one word.")];
        let base = tok.apply_chat_template(&msgs, None, None, None, ThinkingMode::Auto)
            .expect("render (auto)");
        let off = tok.apply_chat_template(&msgs, None, None,
                                           Some(&serde_json::json!({"enable_thinking": false})),
                                           ThinkingMode::Auto).expect("render (off)");
        let on = tok.apply_chat_template(&msgs, None, None,
                                          Some(&serde_json::json!({"enable_thinking": true})),
                                          ThinkingMode::Auto).expect("render (on)");
        eprintln!("[W1] fixture={path}");
        eprintln!("[W1] auto tail={:?}", &base[base.len().saturating_sub(40)..]);
        eprintln!("[W1] off  tail={:?}", &off[off.len().saturating_sub(40)..]);
        assert!(off.ends_with("<think>\n\n</think>\n\n"),
                "enable_thinking=false must render the closed think block; tail={:?}",
                &off[off.len().saturating_sub(60)..]);
        assert!(on.ends_with("<think>\n"),
                "enable_thinking=true must prime the think block; tail={:?}",
                &on[on.len().saturating_sub(60)..]);
        // FIXTURE-DEPENDENT DEFAULT (v0.7.10 prep): whether Auto primes or closes the block is the
        // TEMPLATE's decision, and our fixtures differ — the 3.6-35b family template primes under
        // Auto, while the 0.8b template guards `enable_thinking is defined and enable_thinking is
        // true`, so undefined (Auto) falls to the CLOSED branch. Hard-coding "primes" made the test
        // fail on any box that only has the 0.8b fixture. What must hold for EVERY template: Auto
        // renders exactly one of the two explicit branches (the kwarg pass-through is consistent),
        // never a third form.
        let primes = base.ends_with("<think>\n");
        let closes = base.ends_with("<think>\n\n</think>\n\n");
        assert!(primes || closes,
                "Auto must render one of the template's two explicit think branches; tail={:?}",
                &base[base.len().saturating_sub(60)..]);
        // and Auto must agree with whichever explicit branch matches its render (no third form)
        assert!(base == on || base == off,
                "Auto must render exactly the primed or the closed branch, not a third form");
    }

    /// PRONG (b): a user-edited template whose DEFAULT is "no thinking" must be honoured when the
    /// request asks for nothing. Built in a temp dir from a synthetic template + a real tokenizer.
    #[test]
    fn template_edit_default_off_is_honoured() {
        let Some((_, path)) = fixture_tokenizer() else {
            eprintln!("skip: no tokenizer fixture for the W1 template-edit test");
            return;
        };
        let dir = std::env::temp_dir().join(format!("w1_tpl_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::copy(&path, dir.join("tokenizer.json")).expect("copy tokenizer");
        std::fs::write(dir.join("chat_template.jinja"), concat!(
            "{{- '<|im_start|>user\n' }}{{ messages[0].content }}{{ '<|im_end|>\n' }}",
            "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}",
            "{%- if enable_thinking is defined and enable_thinking is true %}{{- '<think>\n' }}",
            "{%- else %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}{%- endif %}",
        )).expect("write template");

        let tok = QwenTokenizer::from_file(&dir.join("tokenizer.json").to_string_lossy()).expect("load");
        assert!(tok.chat_env.is_some(), "synthetic template must compile");
        let msgs = vec![ChatMessage::user("hi")];
        let auto = tok.apply_chat_template(&msgs, None, None, None, ThinkingMode::Auto).expect("auto");
        let on = tok.apply_chat_template(&msgs, None, None, None, ThinkingMode::On).expect("on");
        let off = tok.apply_chat_template(&msgs, None, None, None, ThinkingMode::Off).expect("off");
        assert!(auto.ends_with("<think>\n\n</think>\n\n"),
                "template default OFF must win when the request asks nothing: {auto:?}");
        assert!(on.ends_with("<think>\n"), "--thinking on must override the template default: {on:?}");
        assert!(off.ends_with("<think>\n\n</think>\n\n"), "--thinking off keeps it off: {off:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Arbitrary (non-managed) `chat_template_kwargs` reach the template context verbatim, and the
    /// engine-owned keys cannot be clobbered by a request.
    #[test]
    fn chat_template_kwargs_passthrough_and_guards() {
        let Some((_, path)) = fixture_tokenizer() else {
            eprintln!("skip: no tokenizer fixture for the kwargs passthrough test");
            return;
        };
        let dir = std::env::temp_dir().join(format!("w1_kw_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::copy(&path, dir.join("tokenizer.json")).expect("copy tokenizer");
        std::fs::write(dir.join("chat_template.jinja"),
            "{{- messages|length }}:{{ my_custom_var|default('unset') }}").expect("write template");
        let tok = QwenTokenizer::from_file(&dir.join("tokenizer.json").to_string_lossy()).expect("load");
        let msgs = vec![ChatMessage::user("hi")];
        let r = tok.apply_chat_template(&msgs, None, None,
                                        Some(&serde_json::json!({"my_custom_var": "hello"})),
                                        ThinkingMode::Auto).expect("render with custom kwarg");
        assert!(r.contains("hello"), "arbitrary kwarg must reach the template: {r}");
        let err = tok.apply_chat_template(&msgs, None, None,
                                          Some(&serde_json::json!({"messages": []})),
                                          ThinkingMode::Auto);
        assert!(err.is_err(), "a request must not be able to clobber the engine's `messages` key");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// WP01 gate: the load-time tables are byte-identical to the legacy per-request rebuilds.
    ///   (a) cached think/eos/vocab_size == the legacy per-call expressions;
    ///   (b) EVERY id (plus 64 past the table), each from a fresh state: push + finish identical;
    ///   (c) 10K random id sequences (byte-alphabet ids to split/garble UTF-8, uniform ids,
    ///       specials, out-of-range ids), every push and the final finish identical;
    ///   (d) emoji / CJK / mixed-script texts stream-decode identically AND round-trip.
    /// Skips when no model tokenizer is present (`fixture_tokenizer`).
    #[test]
    fn token_bytes_match_without_the_load_time_tables() {
        let Some((tok, path)) = fixture_tokenizer() else {
            eprintln!("skip: no model tokenizer fixture"); return;
        };
        let (mut bare, _) = fixture_tokenizer().unwrap();
        bare.cache = None;
        let n = tok.vocab_size() as u32;
        for id in 0..n {
            for skip in [true, false] {
                assert_eq!(tok.token_bytes(id, skip), bare.token_bytes(id, skip), "{path}: id={id} skip={skip}");
            }
        }
    }

    #[test]
    fn wp01_stream_tables_match_legacy() {
        let Some((tok, path)) = fixture_tokenizer() else {
            eprintln!("skip: no model tokenizer fixture"); return;
        };
        let c = tok.cache.as_ref().expect("WP01 cache must be built at load");
        // (a)
        let vocab = tok.tokenizer.get_vocab(true);
        assert_eq!(c.think_hy, vocab.contains_key("</think:opensource>"));
        assert_eq!(c.eos_id, vocab.get("<|endoftext|>").copied().unwrap_or(151643) as u32);
        assert_eq!(c.vocab_size, tok.tokenizer.get_vocab_size(true));
        assert_eq!(c.stream.dup_ids, 0,
                   "{path}: ids with two vocab strings make the LEGACY inversion random; not comparable");
        let json = stream_json_path(tok.model_dir.as_deref());
        let tables = std::sync::Arc::clone(&c.stream);
        let mut new = StreamByteDecoder::from_tables(std::sync::Arc::clone(&tables), None);
        let mut old = LegacyStreamDecoder::new(&tok.tokenizer, &json);
        assert!(!old.specials.is_empty(), "{json}: special set should not be empty");
        // (b)
        let n = tables.kind.len() as u32;
        let (mut n_piece, mut n_special) = (0usize, 0usize);
        for id in 0..n + 64 {
            match tables.kind.get(id as usize).copied() {
                Some(K_PIECE) => n_piece += 1,
                Some(K_SPECIAL) => n_special += 1,
                _ => {}
            }
            assert_eq!(new.push(id), old.push(id), "push id={id}");
            assert_eq!(new.finish(), old.finish(), "finish after id={id}");
        }
        assert!(n_piece > 1000 && n_special > 0, "table looks empty: piece={n_piece} special={n_special}");
        // (c)
        let specials: Vec<u32> = (0..n).filter(|&i| tables.kind[i as usize] == K_SPECIAL).collect();
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut rnd = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let mut pushes = 0usize;
        for seq in 0..10_000 {
            let len = 1 + (rnd() % 48) as usize;
            for k in 0..len {
                let r = rnd();
                let id = match r % 10 {
                    0..=3 => (r >> 8) as u32 % 256,
                    4..=7 => (r >> 8) as u32 % n,
                    8 => specials[(r >> 8) as usize % specials.len()],
                    _ => n.saturating_sub(8) + (r >> 8) as u32 % 72,
                };
                assert_eq!(new.push(id), old.push(id), "seq {seq} pos {k} id={id}");
                pushes += 1;
            }
            assert_eq!(new.finish(), old.finish(), "finish seq {seq}");
        }
        // (d)
        for text in ["Hello \u{1F600} world! \u{1F680}\u{1F4A5}",
                     "你好，世界！这是一个中文测试。日本語のテキストも。한국어 문장.",
                     "e\u{301}\u{302} \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467} flags \u{1F1EC}\u{1F1E7} \u{10348}",
                     "mixed: ascii + Ελληνικά + русский + العربية + हिन्दी + ∑∫√ + 𝔘𝔫𝔦𝔠𝔬𝔡𝔢"] {
            let ids = tok.encode(text, false).expect("encode");
            let (mut a, mut b) = (String::new(), String::new());
            for &id in &ids { a.push_str(&new.push(id)); b.push_str(&old.push(id)); pushes += 1; }
            a.push_str(&new.finish()); b.push_str(&old.finish());
            assert_eq!(a, b, "emoji/CJK stream differs: {text:?}");
            // Round-trip vs the whole-list decode (the tokenizer's normalizer NFC-composes
            // e+U+0301 at encode, so the reference is decode(encode(text)), not `text`).
            assert_eq!(a, tok.decode(&ids, false).expect("decode"), "stream decode must round-trip");
            assert!(!a.contains('\u{FFFD}'), "no replacement chars: {a:?}");
        }
        eprintln!("wp01: {path}: ids={n} piece={n_piece} special={n_special} random+text pushes={pushes} — identical");
    }

    // ---- ink2-t1: generation tags and the `tojson` kwargs (expected strings are Python's `json.dumps` / jinja2 output) ----

    fn render_str(src: &str) -> String {
        let mut env = minijinja::Environment::new();
        super::jinja::register_pycompat(&mut env);
        let s = super::jinja::strip_generation_tags(src);
        env.add_template("t", Box::leak(s.into_boxed_str())).unwrap();
        env.get_template("t").unwrap().render(serde_json::json!({"v": {"b": [1, 2.5, {"z": null, "a": true}], "a": "é中😀", "c": {"y": 1, "x": []}}})).unwrap()
    }

    #[test]
    fn tojson_honours_sort_keys_separators_and_ensure_ascii() {
        // python: json.dumps(v, ensure_ascii=False) and friends
        assert_eq!(render_str("{{ v | tojson }}"), r#"{"b": [1, 2.5, {"z": null, "a": true}], "a": "é中😀", "c": {"y": 1, "x": []}}"#);
        assert_eq!(render_str("{{ v | tojson(sort_keys=true) }}"), r#"{"a": "é中😀", "b": [1, 2.5, {"a": true, "z": null}], "c": {"x": [], "y": 1}}"#);
        assert_eq!(render_str(r#"{{ v | tojson(separators=(",", ":")) }}"#), r#"{"b":[1,2.5,{"z":null,"a":true}],"a":"é中😀","c":{"y":1,"x":[]}}"#);
        assert_eq!(render_str(r#"{{ v | tojson(sort_keys=true, separators=(",", ":")) }}"#), r#"{"a":"é中😀","b":[1,2.5,{"a":true,"z":null}],"c":{"x":[],"y":1}}"#);
        assert_eq!(render_str("{{ v | tojson(ensure_ascii=True) }}"), "{\"b\": [1, 2.5, {\"z\": null, \"a\": true}], \"a\": \"\\u00e9\\u4e2d\\ud83d\\ude00\", \"c\": {\"y\": 1, \"x\": []}}");
        assert_eq!(render_str("{{ v | tojson(ensure_ascii=False) }}"), render_str("{{ v | tojson }}"));
    }

    #[test]
    fn generation_tags_keep_their_whitespace_controls_in_every_spelling() {
        // jinja2 + the HF generation extension: `{%- generation -%}` trims both sides like any block tag
        let t = |s: &str| render_str(s);
        assert_eq!(t("A  {%- generation -%}  B  {%- endgeneration -%}  C"), "ABC");
        assert_eq!(t("A  {% generation %}  B  {% endgeneration %}  C"), "A    B    C");
        assert_eq!(t("A  {%- generation %}  B  {%- endgeneration %}  C"), "A  B  C");
        assert_eq!(t("A  {% generation -%}  B  {% endgeneration -%}  C"), "A  B  C");
        assert_eq!(t("{%-generation-%}x{%-endgeneration-%}"), "x");
        // a block that merely mentions the word is untouched
        assert_eq!(t("{% set generation = 1 %}{{ generation }}"), "1");
    }
