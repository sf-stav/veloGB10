//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `top_k_parse_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::{ChatCompletionRequest, CompletionRequest};

    fn chat(top_k: &str) -> Result<Option<usize>, String> {
        let body = format!(r#"{{"messages":[],"top_k":{top_k}}}"#);
        serde_json::from_str::<ChatCompletionRequest>(&body).map(|r| r.top_k).map_err(|e| e.to_string())
    }

    #[test]
    fn negative_top_k_means_off() {
        assert_eq!(chat("-1"), Ok(Some(0)));
        assert_eq!(chat("-7"), Ok(Some(0)));
        assert_eq!(chat("0"), Ok(Some(0)));
        assert_eq!(chat("20"), Ok(Some(20)));
        assert_eq!(chat("null"), Ok(None));
        assert!(chat("1.5").is_err());
        assert!(chat("\"5\"").is_err());
        let c: CompletionRequest = serde_json::from_str(r#"{"prompt":"x","top_k":-1}"#).unwrap();
        assert_eq!(c.top_k, Some(0));
        let c: CompletionRequest = serde_json::from_str(r#"{"prompt":"x"}"#).unwrap();
        assert_eq!(c.top_k, None);
    }

    #[test]
    fn full_unsigned_range_still_parses() {
        assert_eq!(chat("9223372036854775807"), Ok(Some(i64::MAX as usize)));
        assert_eq!(chat("9223372036854775808"), Ok(Some(1usize << 63)));
        assert_eq!(chat("18446744073709551615"), Ok(Some(u64::MAX as usize)));
        assert_eq!(chat("-9223372036854775808"), Ok(Some(0)));
        assert!(chat("18446744073709551616").is_err());
    }
