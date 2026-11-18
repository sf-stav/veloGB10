//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `wp08_validation_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::{spec_finish_reason, stop_reason_of, validate_penalties};
    use crate::exl3_forward::PenParams;

    #[test]
    fn penalty_validation_contract() {
        assert!(validate_penalties(None, None, None, None).is_ok());
        assert!(validate_penalties(Some(1.05), Some(1.5), Some(0.3), Some(0.05)).is_ok());
        // rep < 1 is accepted (owner decision 2026-09-26); <= 0 / non-finite are 400s
        assert!(validate_penalties(Some(0.05), None, None, None).is_ok());
        assert!(validate_penalties(Some(0.0), None, None, None).is_err());
        assert!(validate_penalties(Some(-1.0), None, None, None).is_err());
        assert!(validate_penalties(Some(f32::NAN), None, None, None).is_err());
        assert!(validate_penalties(Some(f32::INFINITY), None, None, None).is_err());
        assert!(validate_penalties(None, Some(2.0), Some(-2.0), None).is_ok());
        assert!(validate_penalties(None, Some(2.01), None, None).is_err());
        assert!(validate_penalties(None, None, Some(-2.5), None).is_err());
        assert!(validate_penalties(None, None, None, Some(0.0)).is_ok());
        assert!(validate_penalties(None, None, None, Some(1.0)).is_ok());
        assert!(validate_penalties(None, None, None, Some(1.5)).is_err());
        assert!(validate_penalties(None, None, None, Some(-0.1)).is_err());
    }

    #[test]
    fn loop_detected_is_a_stop_with_a_reason() {
        assert_eq!(spec_finish_reason("loop_detected"), "stop");
        assert_eq!(stop_reason_of("loop_detected").as_deref(), Some("loop_detected"));
        assert_eq!(stop_reason_of("stop"), None);
        assert_eq!(spec_finish_reason("length"), "length");
    }

    #[test]
    fn inactive_penalties_stage_nothing() {
        // the reference implementation's alt() NoOp rule: defaults never launch a penalty kernel
        assert_eq!(PenParams::new(1.0, 0.0, 0.0, 1024, 1024), None);
        assert_eq!(PenParams::new(1.1, 0.0, 0.0, 0, 0), None);
        assert!(PenParams::new(1.1, 0.0, 0.0, 1024, 1024).is_some());
        assert!(PenParams::new(1.0, 0.0, 0.3, 1024, 1024).is_some());
        assert!(PenParams::new(0.9, 0.0, 0.0, 1024, 1024).is_some());
        assert!(PenParams::new(1.0, -0.5, 0.0, 16, 0).is_some());
    }
