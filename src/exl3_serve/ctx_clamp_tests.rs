//! Unit tests (module `ctx_clamp_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::*;

    /// YARN1: the pack config's own rope_scaling feeds the factor (native 262144 x 2 = 524288,
    /// still under the packing ceiling). The flag/TP precedence is unit-tested at
    /// qwen::yarn_tests; this covers the config-file source only.
    #[test]
    fn the_pack_rope_scaling_raises_the_ceiling() {
        let dir = std::env::temp_dir().join(format!("gb10_yarn_clamp_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let d = dir.to_str().unwrap();
        std::fs::write(dir.join("config.json"),
            r#"{"text_config":{"max_position_embeddings":262144,"rope_scaling":{"type":"yarn","factor":2.0}}}"#).unwrap();
        // 400_000 is UNDER the 524_288 ceiling: served as asked, no trim. 600_000 trims to it.
        assert_eq!(clamp_max_seq_len(d, 400_000), 400_000);
        assert_eq!(clamp_max_seq_len(d, 600_000), 524_288);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    const W: usize = 262_144; // the Qwen3.8 family's trained window

    /// Without a rope extension the context stops at the model's trained window.
    #[test]
    fn the_context_stops_at_the_trained_window() {
        assert_eq!(clamp_ctx(300_000, Some(W), 1.0), (W, Some(CtxLimit::Rope)));
        assert_eq!(clamp_ctx(W + 1, Some(W), 1.0), (W, Some(CtxLimit::Rope)));
        assert_eq!(clamp_ctx(W, Some(W), 1.0), (W, None));
        assert_eq!(clamp_ctx(32_768, Some(W), 1.0), (32_768, None));
    }

    /// A rope factor raises the ceiling until the packings stop it: max_pos = max_seq_len + headroom
    /// must stay below 2^20.
    #[test]
    fn a_rope_factor_raises_the_ceiling_to_the_packing_limit() {
        let top = packable_max_seq_len();
        assert_eq!(top + crate::batch::decode_headroom(false), crate::exl3_forward::MAX_POS_PACKED - 1);
        assert_eq!(clamp_ctx(600_000, Some(W), 2.0), (2 * W, Some(CtxLimit::Rope)));
        assert_eq!(clamp_ctx(1_048_576, Some(W), 4.0), (top, Some(CtxLimit::Packing)));
        assert_eq!(clamp_ctx(top, Some(W), 4.0), (top, None));
        // a factor below 1 never shrinks the trained window
        assert_eq!(clamp_ctx(W, Some(W), 0.5), (W, None));
    }

    /// An unreadable config must not disable the guard: the packing ceiling still applies.
    #[test]
    fn an_unknown_window_still_gets_the_packing_ceiling() {
        let top = packable_max_seq_len();
        assert_eq!(clamp_ctx(usize::MAX, None, 1.0), (top, Some(CtxLimit::Packing)));
        assert_eq!(clamp_ctx(W, None, 1.0), (W, None));
    }

    /// Whatever is asked, the window the clamp returns loads: `FwdModel::load_tp` accepts its max_pos.
    #[test]
    fn every_clamped_window_passes_the_loader_guard() {
        for native in [None, Some(4096), Some(W), Some(1 << 20), Some(1 << 24)] {
            for f in [1.0, 2.0, 4.0, 64.0] {
                for asked in [1usize, 4096, W, W + 3, 524_288, 1_048_574, 1_048_575, 1 << 20, 1 << 30] {
                    let (eff, _) = clamp_ctx(asked, native, f);
                    assert!(eff <= asked);
                    let max_pos = eff + crate::batch::decode_headroom(false);
                    assert!(crate::exl3_forward::check_max_pos_packable(max_pos).is_ok(), "{asked} {native:?} x{f} -> {eff}");
                }
            }
        }
    }

    /// The log line names the limit that applied (and only mentions a rope factor when one is set).
    #[test]
    fn the_message_names_the_limit() {
        let m = clamp_ctx_message(300_000, W, CtxLimit::Rope, Some(W), 1.0);
        assert_eq!(m, "[warn] --max-seq-len 300000 exceeds the model's trained window 262144 (max_position_embeddings) - clamping to 262144.");
        let m = clamp_ctx_message(600_000, 2 * W, CtxLimit::Rope, Some(W), 2.0);
        assert!(m.contains("trained window 262144 x rope factor 2") && m.ends_with("clamping to 524288."), "{m}");
        let m = clamp_ctx_message(2_000_000, 1_048_574, CtxLimit::Packing, None, 1.0);
        assert!(m.contains("bit-packings (max_pos must stay below 1048576 = 2^20)") && m.ends_with("clamping to 1048574."), "{m}");
    }

    /// Wiring pin: the serve builder and the TP head resolve --max-seq-len through the clamp (the
    /// served path itself is the GPU gate: a 524288 boot loads at 262145 on every rank).
    #[test]
    fn the_serve_and_the_tp_head_go_through_the_clamp() {
        let code = |src: &str| src.lines().filter(|l| !l.trim_start().starts_with("//")).collect::<Vec<_>>().join("\n");
        let serve = code(src_text!("exl3_serve"));
        // CTXCLAMP: `let mut` (the TP = 1 memory fit may clamp it afterwards, one boot line + /metrics)
        let needle = ["let mut max_seq_len: usize = clamp_max_seq_len(model_dir, ", "arg(args, \"--max-seq-len\")"].concat();
        assert_eq!(serve.matches(&needle).count(), 1, "build_serve");
        let main = code(src_text!("main"));
        let needle = ["tpc.max_seq_len = gb10_inference::exl3_serve::", "resolve_max_seq_len("].concat();
        assert_eq!(main.matches(&needle).count(), 1, "run_exl3_tp_serve_head");
    }

    /// The trained window is read from the pack's config.json, flat or nested under text_config.
    #[test]
    fn the_trained_window_is_read_from_the_pack_config() {
        let dir = std::env::temp_dir().join(format!("gb10_ctx_clamp_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let d = dir.to_str().unwrap();
        assert_eq!(native_window(d), None, "no config.json");
        std::fs::write(dir.join("config.json"), r#"{"max_position_embeddings": 262144}"#).unwrap();
        assert_eq!(native_window(d), Some(W));
        std::fs::write(dir.join("config.json"), r#"{"text_config": {"max_position_embeddings": 131072}, "max_position_embeddings": 7}"#).unwrap();
        assert_eq!(native_window(d), Some(131_072));
        assert_eq!(clamp_max_seq_len(d, 200_000), 131_072);
        assert_eq!(clamp_max_seq_len(d, 65_536), 65_536);
        std::fs::remove_dir_all(&dir).unwrap();
    }
