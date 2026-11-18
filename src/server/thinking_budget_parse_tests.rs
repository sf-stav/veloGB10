//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `thinking_budget_parse_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::{ChatCompletionRequest, CompletionRequest};

    fn chat(v: &str) -> Result<Option<usize>, String> {
        let body = format!(r#"{{"messages":[],"thinking_token_budget":{v}}}"#);
        serde_json::from_str::<ChatCompletionRequest>(&body).map(|r| r.thinking_token_budget).map_err(|e| e.to_string())
    }

    fn completion(v: &str) -> Result<Option<usize>, String> {
        let body = format!(r#"{{"prompt":"x","thinking_token_budget":{v}}}"#);
        serde_json::from_str::<CompletionRequest>(&body).map(|r| r.thinking_token_budget).map_err(|e| e.to_string())
    }

    /// THINKBUDGET1: vLLM's validate_thinking_token_budget, on both endpoints.
    #[test]
    fn both_endpoints_follow_vllm() {
        for f in [chat as fn(&str) -> Result<Option<usize>, String>, completion] {
            assert_eq!(f("0"), Ok(Some(0)));
            assert_eq!(f("2048"), Ok(Some(2048)));
            assert_eq!(f("-1"), Ok(None), "-1 = unlimited");
            assert_eq!(f("null"), Ok(None));
            for bad in ["-2", "1.5", "2.0", "true", "\"512\""] {
                let e = f(bad).unwrap_err();
                assert!(e.contains(crate::think_budget::INVALID), "{bad}: {e}");
            }
        }
        let c: ChatCompletionRequest = serde_json::from_str(r#"{"messages":[]}"#).unwrap();
        assert_eq!(c.thinking_token_budget, None, "absent = no budget");
        let c: CompletionRequest = serde_json::from_str(r#"{"prompt":"x"}"#).unwrap();
        assert_eq!(c.thinking_token_budget, None);
    }

    /// Before THINKBUDGET1 the chat struct's catch-all swallowed the field; it must be a real field now.
    #[test]
    fn the_field_is_not_an_unused_field_any_more() {
        let c: ChatCompletionRequest = serde_json::from_str(r#"{"messages":[],"thinking_token_budget":64}"#).unwrap();
        assert!(!c.unused_fields.contains_key("thinking_token_budget"));
        assert_eq!(c.thinking_token_budget, Some(64));
    }
