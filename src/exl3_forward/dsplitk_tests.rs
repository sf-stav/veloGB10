//! Unit tests (module `dsplitk_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    use super::*;

    #[test]
    fn the_pf_tc_gate_covers_the_qwen35_family_and_the_wiring_keeps_flash_next_off() {
        for (nh, nkv) in [(16usize, 2usize), (24, 4), (16, 4), (8, 1), (12, 2), (4, 1)] {
            assert!(pf_tc_geometry_ok(nh, nkv, 256), "{nh}/{nkv}");
        }
        assert!(!pf_tc_geometry_ok(16, 2, 128), "head_dim 128");
        assert!(!pf_tc_geometry_ok(20, 4, 256), "g = 5 does not divide 96");
        assert!(!pf_tc_geometry_ok(16, 0, 256));
        let src = src_text!("exl3_forward");
        let code: String = src.lines().filter(|l| !l.trim_start().starts_with("//")).collect::<Vec<_>>().join("\n");
        let code = &code[..code.find("mod dsplitk_tests").unwrap_or(code.len())];
        let gate = &code[code.find("fn pf_tc_ok(").unwrap()..];
        let gate = &gate[..gate.find("\n    }").unwrap()];
        assert!(gate.contains("a.idx.is_none()") && gate.contains("!cfg.has_indexer()"), "PFTC must never take a layer with an indexer");
        assert_eq!(code.matches("if self.pf_tc_ok(a, att_i) {").count(), 1, "one prefill call site");
        for k in ["xq_attn_pf_qstage", "xq_attn_pf_tc256", "xq_attn_pf_tc_epi"] {
            assert!(crate::exl3_bench::MODULE_FNS.contains(&k), "{k} is loaded");
            assert!(src_text!("kernels/exl3_bench").contains(&format!("{k}(")), "{k} exists");
        }
        let split = &code[code.find("fn dense_splitk_ok(").unwrap()..];
        assert!(split[..split.find("\n    }").unwrap()].contains("!self.cfg.has_indexer()"), "DSPLITK must never take Flash-Next");
    }

    #[test]
    fn the_geometry_gate_covers_the_qwen35_family_and_refuses_the_rest() {
        for (nh, nkv) in [(16usize, 2usize), (24, 4), (16, 4), (32, 2), (8, 1)] {
            assert!(dense_splitk_geometry_ok(nh, nkv, 256), "{nh}/{nkv}");
        }
        assert!(!dense_splitk_geometry_ok(16, 2, 128), "head_dim 128: the kernel is 32 lanes x 8 dims");
        assert!(!dense_splitk_geometry_ok(64, 2, 256), "32 query heads per KV head > 16");
        assert!(!dense_splitk_geometry_ok(16, 0, 256));
        assert!(!dense_splitk_geometry_ok(18, 4, 256), "not a whole group");
    }

    /// Both trunk dense call sites (decode step and verify) consult the split-K gate BEFORE their old
    /// kernel and after dense v3, and the option is a CLI flag (no env name) that rides TpConfig (SPMD).
    #[test]
    fn decode_and_verify_both_route_through_the_gate() {
        let src = src_text!("exl3_forward");
        let code: String = src.lines().filter(|l| !l.trim_start().starts_with("//")).collect::<Vec<_>>().join("\n");
        let code = &code[..code.find("mod dsplitk_tests").unwrap_or(code.len())]; // not this test's own literals
        let n = code.matches("} else if self.dense_splitk_ok(cfg.num_heads, cfg.num_kv_heads, cfg.head_dim) {").count();
        assert_eq!(n, 2, "decode step + verify");
        let step = &code[code.find("fn dense_attn_step(").unwrap()..];
        let (g, old) = (step.find("dense_splitk_ok").unwrap(), step.find("\"xq_attn_decode\"").unwrap());
        assert!(g < old, "decode: gate before xq_attn_decode");
        let opts = src_text!("opts_table");
        let line = opts.lines().find(|l| l.contains("flag: \"exl3-dense-splitk\"")).expect("the flag");
        assert!(line.contains("env: \"\"") && line.contains("def: \"on\"") && line.contains("Scope::Spmd"), "{line}");
    }
