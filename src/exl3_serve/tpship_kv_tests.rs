//! Unit tests (module `tpship_kv_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    /// TPSHIP: bf16 (no EXL3 store) is served as f16; the four real formats pass through; anything else is refused.
    #[test]
    fn bf16_kv_falls_back_to_f16_and_the_rest_is_unchanged() {
        assert_eq!(super::exl3_kv_cache_value("bf16").unwrap(), "f16");
        for v in ["f32", "f16", "fp8", "q8"] { assert_eq!(super::exl3_kv_cache_value(v).unwrap(), v); }
        assert!(super::exl3_kv_cache_value("int4").is_err());
    }
