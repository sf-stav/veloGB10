//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `harden_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::*;
    use std::collections::HashSet;
    use std::sync::atomic::AtomicBool;

    /// LR-8 proof: first N distinct reasons warn (Some(true)); duplicates never re-warn
    /// (Some(false)); past the cap nothing grows and exactly one suppression verdict (None)
    /// is returned, then silence.
    #[test]
    fn schema_seen_cap_bounds_the_set() {
        let mut seen = HashSet::new();
        let suppressed = AtomicBool::new(false);
        for i in 0..SCHEMA_SEEN_CAP {
            assert_eq!(schema_seen_note(&mut seen, &suppressed, &format!("reason {i}")), Some(true));
        }
        assert_eq!(seen.len(), SCHEMA_SEEN_CAP);
        // A repeat is still a duplicate.
        assert_eq!(schema_seen_note(&mut seen, &suppressed, "reason 0"), Some(false));
        // The first NEW reason past the cap yields the one-time suppression verdict...
        assert_eq!(schema_seen_note(&mut seen, &suppressed, "brand new"), None);
        // ...and the set did not grow.
        assert_eq!(seen.len(), SCHEMA_SEEN_CAP);
        // Every later new reason is silent (Some(false)) and still does not grow the set.
        for i in 0..50 {
            assert_eq!(schema_seen_note(&mut seen, &suppressed, &format!("over {i}")), Some(false));
        }
        assert_eq!(seen.len(), SCHEMA_SEEN_CAP);
    }
