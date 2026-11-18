//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `logprobs_span_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::{content_byte_range, logprob_token_span};

    // ids 0..: "Hel", "lo", " wor", "ld", <stop>
    fn bytes_of(t: u32) -> Vec<u8> {
        ["Hel", "lo", " wor", "ld", ""][t as usize].as_bytes().to_vec()
    }

    #[test]
    fn span_clips_tokens_at_the_range_edges() {
        let toks = [0u32, 1, 2, 3, 4];
        // a stop string "lo" cut the text to "Hel"
        let v = logprob_token_span(bytes_of, &[4], &toks, 0..3);
        assert_eq!(v, vec![(0, 0, b"Hel".to_vec())]);
        // a stop string starting inside "lo" ("o w"): "Hello w" -> "Hell"
        let v = logprob_token_span(bytes_of, &[4], &toks, 0..4);
        assert_eq!(v, vec![(0, 0, b"Hel".to_vec()), (1, 3, b"l".to_vec())]);
        // content "world" of "Hello world": starts inside " wor"
        let v = logprob_token_span(bytes_of, &[4], &toks, 6..11);
        assert_eq!(v, vec![(2, 5, b"wor".to_vec()), (3, 9, b"ld".to_vec())]);
        // the whole text; the stop token is not listed
        assert_eq!(logprob_token_span(bytes_of, &[4], &toks, 0..usize::MAX).len(), 4);
        // empty content: nothing
        assert!(logprob_token_span(bytes_of, &[4], &toks, 11..11).is_empty());
    }

    #[test]
    fn content_range_skips_reasoning_and_trim() {
        let t = "thinking about it</think>\n\nThe answer.";
        let r = content_byte_range(t, "The answer.", "</think>").unwrap();
        assert_eq!(&t[r], "The answer.");
        // the same words inside the reasoning are not matched
        let t = "The answer. maybe</think> The answer.";
        let r = content_byte_range(t, "The answer.", "</think>").unwrap();
        assert_eq!(r.start, 26);
        // no reasoning block
        assert_eq!(content_byte_range("  hi ", "hi", "</think>"), Some(2..4));
        // tool-only reply: empty content
        let t = "<tool_call>x</tool_call>";
        assert_eq!(content_byte_range(t, "", "</think>"), Some(0..0));
        // content not one span of the text
        assert_eq!(content_byte_range("a b", "ab", "</think>"), None);
    }
