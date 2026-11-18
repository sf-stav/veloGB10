//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `completions_usage_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::*;

    #[test]
    fn the_request_field_is_parsed_and_only_an_explicit_true_asks_for_the_chunk() {
        let parse = |s: &str| serde_json::from_str::<CompletionRequest>(s).expect("valid request");
        let r = parse(r#"{"prompt": "x", "stream": true, "stream_options": {"include_usage": true}}"#);
        assert!(completions_include_usage(r.stream_options.as_ref()));
        let r = parse(r#"{"prompt": "x", "stream": true, "stream_options": {"include_usage": false}}"#);
        assert!(!completions_include_usage(r.stream_options.as_ref()));
        let r = parse(r#"{"prompt": "x", "stream": true, "stream_options": {}}"#);
        assert!(!completions_include_usage(r.stream_options.as_ref()));
        // not sent: no chunk (an existing client's stream keeps its bytes)
        let r = parse(r#"{"prompt": "x", "stream": true}"#);
        assert!(r.stream_options.is_none() && !completions_include_usage(r.stream_options.as_ref()));
    }

    #[test]
    fn the_usage_chunk() {
        let t = Timings { prompt_ms: 120.0, predicted_ms: 800.0, prompt_per_second: 100.0, predicted_per_second: 40.0 };
        let c = completions_usage_chunk("cmpl-abc", 1_760_000_000, "Qwen/Qwen3.8-Flash-Next", 12, 32, 8, t);
        // the OpenAI stream-usage shape: empty choices, top-level usage
        assert_eq!(c["choices"], serde_json::json!([]));
        assert_eq!(c["usage"]["prompt_tokens"], 12);
        assert_eq!(c["usage"]["completion_tokens"], 32);
        assert_eq!(c["usage"]["total_tokens"], 44);
        assert_eq!(c["usage"]["prompt_tokens_details"]["cached_tokens"], 8);
        // this path's ids and object type, as its text chunks carry them
        assert_eq!(c["id"], "cmpl-abc");
        assert_eq!(c["object"], "text_completion.chunk");
        assert_eq!(c["created"], 1_760_000_000i64);
        assert_eq!(c["model"], "Qwen/Qwen3.8-Flash-Next");
        // the timings extension, as the chat path's chunk
        assert_eq!(c["timings"]["predicted_per_second"], 40.0);
        // one line of JSON for one SSE data event
        let s = c.to_string();
        assert!(!s.contains('\n') && serde_json::from_str::<serde_json::Value>(&s).is_ok());
        // a cold request reports zero cached tokens, never omits the field
        let z = completions_usage_chunk("cmpl-0", 0, "m", 5, 0, 0, Timings { prompt_ms: 0.0, predicted_ms: 0.0, prompt_per_second: 0.0, predicted_per_second: 0.0 });
        assert_eq!(z["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
        assert_eq!(z["usage"]["total_tokens"], 5);
    }

    /// The chunk goes out after the finish chunk and before [DONE], only when asked (source pin:
    /// the unit suite has no scheduler to stream from).
    #[test]
    fn the_stream_sends_it_between_the_finish_chunk_and_done() {
        let src = src_text!("server");
        let body = &src[src.find("async fn completions(").expect("the completions handler")..];
        let fin = body.find("\"choices\": [{\"index\": 0, \"text\": \"\", \"finish_reason\": fr}],").expect("the finish chunk");
        let gate = body.find(format!("if {}_usage {{", "include").as_str()).expect("the include_usage gate");
        let done = body[gate..].find("data(\"[DONE]\")").expect("[DONE] after the usage chunk") + gate;
        assert!(fin < gate && gate < done);
        assert!(body[gate..done].contains("completions_usage_chunk(&cid, created, &model_name, prompt_len, ntok, cached,"));
    }
