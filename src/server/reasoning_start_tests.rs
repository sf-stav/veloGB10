//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `reasoning_start_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::{split_constrained, split_think_in, stream_starts_in_reasoning};

    #[test]
    fn constrained_reply_keeps_think_markers_literal() {
        // thinking off: the whole reply is the document, markers inside its strings are text
        let doc = r#"{"s":"</think>","t":"<think>x"}"#;
        assert_eq!(split_constrained(doc, "</think>", false), (None, doc.to_string()));
        // the generic splitter would cut it at the embedded tag
        assert_ne!(split_think_in(doc, "<think>", "</think>", false).1, doc);
        // thinking on: reasoning ends at the first close tag; later tags are document text
        let r = format!("plan it\n</think>\n\n{doc}");
        assert_eq!(split_constrained(&r, "</think>", true), (Some("plan it".into()), doc.to_string()));
        // budget exhausted while reasoning: no document
        assert_eq!(split_constrained("still thinking", "</think>", true), (Some("still thinking".into()), String::new()));
    }

    /// ISSUES 149: a Qwen no-think render (the 0.8B/2B template default) with reasoning_effort
    /// low|high starts in CONTENT; only an open think tag (or hy_v3's effort arm) starts in reasoning.
    #[test]
    fn qwen_no_think_render_with_an_effort_is_content() {
        let no_think = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let think = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
        for effort in [None, Some("low"), Some("high"), Some("medium")] {
            assert!(!stream_starts_in_reasoning(no_think, "<think>", true, effort), "{effort:?}");
            assert!(stream_starts_in_reasoning(think, "<think>", true, effort), "{effort:?}");
        }
        // the reproduced answer (v0.7.5, streaming, effort low): a direct code block is content
        let (r, c) = split_think_in("```c\nvoid f(void) {}\n```", "<think>", "</think>",
                                    stream_starts_in_reasoning(no_think, "<think>", true, Some("low")));
        assert_eq!((r, c.as_str()), (None, "```c\nvoid f(void) {}\n```"));
        // hy_v3 keeps its effort arm, unchanged
        let hy = "<|hy_User|>hi<|hy_Assistant|>";
        assert!(stream_starts_in_reasoning(hy, "<think:opensource>", false, Some("low")));
        assert!(stream_starts_in_reasoning(hy, "<think:opensource>", false, Some("high")));
        assert!(!stream_starts_in_reasoning(hy, "<think:opensource>", false, None));
    }
