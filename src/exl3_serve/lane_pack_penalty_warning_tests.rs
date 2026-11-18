//! Unit tests (module `lane_pack_penalty_warning_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::*;

    #[test]
    fn the_warning_follows_the_server_defaults() {
        // the no-op defaults (rep 1, presence 0, frequency 0): nothing to warn about
        assert!(lane_pack_penalty_warning(1.0, 0.0, 0.0).is_none());
        // the owner's posture, and each sibling on its own
        for (r, p, f) in [(1.0, 1.5, 0.0), (1.0, 0.0, 0.3), (1.1, 0.0, 0.0), (1.1, 1.5, 0.3)] {
            let w = lane_pack_penalty_warning(r, p, f).unwrap_or_else(|| panic!("({r}, {p}, {f}) must warn"));
            assert!(w.starts_with("[exl3-serve] lane-pack WARNING") && w.contains("never joins a packed verify")
                    && w.contains("--default-presence-penalty") && w.contains(&format!("presence {p}")), "{w}");
        }
        // it is asked exactly where the defaults are parsed, and only when the arm is on
        let src = src_text!("exl3_serve");
        let code = &src[..src.find("mod lane_pack_penalty_warning_tests").unwrap()];
        let at = code.find("if let Some(w) = lane_pack_penalty_warning(").expect("the call");
        assert!(code[..at].trim_end().ends_with("if lane_pack.on() {"), "the warning is only for a boot with the arm on");
    }
