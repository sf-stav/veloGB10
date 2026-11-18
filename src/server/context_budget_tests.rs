//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `context_budget_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    // PR #4's context-budget contract. The PR's two further tests
    // (streaming_state_follows_the_rendered_prompt_not_the_model_family,
    // reasoning_effort_high_stays_a_thinking_level) test `prompt_ends_inside_think` /
    // `normalize_reasoning_effort`, which this leg deliberately does NOT port — dev's
    // reasoning semantics (P13 thinking toggle + the 35b4b15 effort-derived start) are kept.
    use super::{esc, generation_room};

    #[test]
    fn mtp_headroom_is_reserved_before_clamping() {
        assert_eq!(generation_room(4096, 1314, 16), Some(2766));
        assert_eq!(generation_room(4096, 1324, 16), Some(2756));
    }

    #[test]
    fn prompt_that_leaves_only_headroom_has_no_generation_room() {
        assert_eq!(generation_room(4096, 4080, 16), None);
        assert_eq!(generation_room(4096, usize::MAX, 16), None);
    }

    #[test]
    fn sse_json_escape_preserves_indented_go_code() {
        let text = concat!(
            "// New crée un cache.\n",
            "func New(capacity int) *CacheLRU {\n",
            "\treturn &CacheLRU{\r\n",
            "\t\tcapacity: capacity,\n",
            "\t\titems: make(map[interface{}]*list.Element),\n",
            "\t}\n",
            "}\n",
        );
        let payload = format!(r#"{{"delta":{{"content":"{}"}}}}"#, esc(text));
        let parsed: serde_json::Value =
            serde_json::from_str(&payload).expect("SSE data must contain valid JSON");

        assert_eq!(parsed["delta"]["content"], text);
        assert!(!payload.contains('\t'), "JSON payload must not contain literal tabs");
        assert!(!payload.contains('\r'), "JSON payload must not contain literal CRs");
    }
