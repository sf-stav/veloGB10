//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `logprobs_param_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::{chat_logprobs_param, completion_logprobs_param, logprobs_fallback};

    #[test]
    fn chat_and_completion_logprobs_params() {
        assert_eq!(chat_logprobs_param(None, None), Ok(None));
        assert_eq!(chat_logprobs_param(Some(false), None), Ok(None));
        assert_eq!(chat_logprobs_param(Some(true), None), Ok(Some(0)));
        assert_eq!(chat_logprobs_param(Some(true), Some(20)), Ok(Some(20)));
        assert!(chat_logprobs_param(Some(true), Some(21)).is_err());
        assert!(chat_logprobs_param(None, Some(5)).is_err(), "top_logprobs needs logprobs");
        assert_eq!(completion_logprobs_param(None), Ok(None));
        assert_eq!(completion_logprobs_param(Some(5)), Ok(Some(5)));
        assert!(completion_logprobs_param(Some(21)).is_err());
        // owner rule (no refusals): what the server cannot return falls back to logprobs null, with a reason
        assert_eq!(logprobs_fallback(true, false, false), None);
        assert_eq!(logprobs_fallback(false, false, false).map(|f| f.0), Some("unsupported"));
        assert_eq!(logprobs_fallback(true, true, false).map(|f| f.0), Some("withheld"));
        assert_eq!(logprobs_fallback(true, false, true).map(|f| f.0), Some("stream"));
        assert_eq!(logprobs_fallback(false, true, true).map(|f| f.0), Some("unsupported"), "the engine reason wins");
    }
