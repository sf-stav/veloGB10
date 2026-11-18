//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `loopfix_card_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::qwen35_card;

    fn dir_with(cfg: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("loopfix_card_{}_{}", std::process::id(), rand::random::<u32>()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("config.json"), cfg).unwrap();
        d
    }

    #[test]
    fn card_by_identity_not_folder_name() {
        let small = dir_with(r#"{"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text","hidden_size":1024,"num_hidden_layers":24}}"#);
        let two = dir_with(r#"{"model_type":"qwen3_5","text_config":{"hidden_size":2048,"num_hidden_layers":24}}"#);
        let four = dir_with(r#"{"model_type":"qwen3_5","text_config":{"hidden_size":2560,"num_hidden_layers":32}}"#);
        let moe = dir_with(r#"{"model_type":"qwen3_5_moe","text_config":{"hidden_size":2048,"num_hidden_layers":40}}"#);
        let other = dir_with(r#"{"model_type":"hy_v3","hidden_size":1024,"num_hidden_layers":24}"#);
        let (d, p) = qwen35_card(small.to_str().unwrap()).unwrap();
        assert_eq!((d.no_think, d.think, p), ((1.0, 1.0, 20), (1.0, 0.95, 20), 2.0));
        assert_eq!(qwen35_card(two.to_str().unwrap()).unwrap().1, 2.0);
        let (d, p) = qwen35_card(four.to_str().unwrap()).unwrap();
        assert_eq!((d.no_think, d.think, p), ((0.7, 0.8, 20), (1.0, 0.95, 20), 1.5));
        assert_eq!(qwen35_card(moe.to_str().unwrap()).unwrap().1, 1.5);    // 35B-A3B: hidden 2048 but 40 layers
        assert!(qwen35_card(other.to_str().unwrap()).is_none());
        assert!(qwen35_card("/nonexistent/loopfix").is_none());
        for d in [small, two, four, moe, other] { let _ = std::fs::remove_dir_all(d); }
    }
