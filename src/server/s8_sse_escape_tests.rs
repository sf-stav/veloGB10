//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `s8_sse_escape_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::*;

    #[test]
    fn sse_chunks_escape_the_model_id() {
        let evil = "Qwen/Qwen3.8\"Flash\\Next";
        for chunk in [
            role_chunk("chatcmpl-1", 1, evil),
            content_chunk("chatcmpl-1", 1, evil, "answer"),
            reasoning_chunk("chatcmpl-1", 1, evil, "thinking"),
            final_chunk("chatcmpl-1", 1, evil, "stop", ""),
        ] {
            let v: serde_json::Value = serde_json::from_str(&chunk)
                .unwrap_or_else(|e| panic!("chunk with an evil model id must stay valid JSON ({e}): {chunk}"));
            assert_eq!(v["model"].as_str(), Some(evil), "the id round-trips verbatim: {chunk}");
        }
        // and the text field stays escaped too (regression guard on the extraction)
        let c = content_chunk("chatcmpl-1", 1, "m", "line1\nline2\"q");
        let v: serde_json::Value = serde_json::from_str(&c).unwrap();
        assert_eq!(v["choices"][0]["delta"]["content"].as_str(), Some("line1\nline2\"q"));
    }
