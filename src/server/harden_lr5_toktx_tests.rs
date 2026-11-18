//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `harden_lr5_toktx_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::*;

    /// LR-5 proof: the wrapper's unconsumed-event count is exact across sends, consumes and
    /// disconnect — this is the number the scheduler's cancel sweep compares against
    /// `--stream-backlog-events`.
    #[test]
    fn toktx_counts_unconsumed_backlog() {
        let (tx, mut rx, bl) = tok_channel();
        assert_eq!(tx.len(), 0);
        for i in 0..5u32 { tx.send(TokEvent::Tok(i)).unwrap(); }
        assert_eq!(tx.len(), 5, "5 sent, none consumed");
        try_recv_counted(&mut rx, &bl).unwrap();
        try_recv_counted(&mut rx, &bl).unwrap();
        assert_eq!(tx.len(), 3, "3 unconsumed after the handler read 2 via try_recv_counted");
        let clone = tx.clone();
        assert_eq!(clone.len(), 3, "clones share the count");
        drop(rx);
        assert!(tx.is_closed(), "a dropped receiver closes the sender");
    }
