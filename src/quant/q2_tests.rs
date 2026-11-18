//! Inline unit tests of the quant module, moved out of `quant.rs` verbatim (module `q2_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::*;
    #[test]
    fn layer_rules_parse_and_select() {
        // S5F2 L1 ladder spec: `layers:<sel>:<fmt>` tokens (list via `+`/`,`, range via `-`).
        let rules = parse_layer_rules("all,layers:5+19+33+47+61:fp8");
        assert_eq!(rules.len(), 5);
        assert!(rules.iter().all(|r| r.fmt == Fmt::Fp8));
        let rules2 = parse_layer_rules("all,layers:0-61:bf16");
        assert_eq!(rules2.len(), 1);
        assert_eq!((rules2[0].lo, rules2[0].hi, rules2[0].fmt), (0, 61, Fmt::Bf16));
        // layer_index_of: both qwen3.5 (`model.language_model.layers.N.*`) and hy_v3 roots.
        assert_eq!(layer_index_of("model.language_model.layers.5.self_attn.q_proj.weight"), Some(5));
        assert_eq!(layer_index_of("model.layers.61.mlp.down_proj.weight"), Some(61));
        assert_eq!(layer_index_of("lm_head.weight"), None);
        assert_eq!(layer_index_of("mtp.layers.0.mlp.gate_up_proj.weight"), Some(0));
        // Layer rules WIN over the group map; non-layered tensors fall back to the group map.
        let map = parse_recipe("all").unwrap();
        let f1 = fmt_for_layer(&map, &rules, "model.language_model.layers.19.self_attn.q_proj.weight");
        let f2 = fmt_for_layer(&map, &rules, "model.language_model.layers.20.self_attn.q_proj.weight");
        let f3 = fmt_for_layer(&map, &rules, "lm_head.weight");
        assert!(matches!(f1, Fmt::Fp8), "tap-layer override must win: {f1:?}");
        assert!(matches!(f2, Fmt::Nvfp4), "non-tap layer falls back to the group map: {f2:?}");
        assert!(matches!(f3, Fmt::Nvfp4), "non-layered tensor uses the group map: {f3:?}");
        // bf16 override = "no fake-quant" on the covered layers.
        let f4 = fmt_for_layer(&map, &rules2, "model.language_model.layers.5.self_attn.q_proj.weight");
        assert!(matches!(f4, Fmt::Bf16));
    }
    #[test]
    fn q2_levels_assigned() {
        // one block with a known spread: expect all four levels to appear
        let mut w: Vec<bf16> = vec![-0.249, -0.994, 0.497, -0.994, -1.491, 0.746, -0.249, 0.746,
                                    -0.994, 0.994, -0.994, -0.994, 1.491, 0.0, -0.249, 1.491]
            .iter().map(|&x| bf16::from_f32(x)).collect();
        // scale so the block amax is exactly Q2_MAX (s = e4m3(1)*st ≈ 1.0 path)
        let k = 16;
        let orig: Vec<f32> = w.iter().map(|x| x.to_f32()).collect();
        fake_quant_q2(&mut w, 1, k);
        let got: Vec<f32> = w.iter().map(|x| x.to_f32()).collect();
        eprintln!("orig: {:?}", orig);
        eprintln!("got:  {:?}", got);
        let has_small = got.iter().any(|v| v.abs() < 1.0);
        assert!(has_small, "q2 collapsed everything to the extreme level: {:?}", got);
    }

    #[test]
    fn draft_vocab_defaults_and_parses_every_spelling() {
        // default (no flag): ON for the Qwen3.5/3.6/3.8 family only (owner 2026-10-27)
        assert_eq!(DraftVocab::default_for(crate::qwen::Family::Qwen35), DraftVocab::Top(DRAFT_VOCAB_ON_ROWS));
        for f in [crate::qwen::Family::HyV3, crate::qwen::Family::Qwen4Exp, crate::qwen::Family::Dsv4] {
            assert_eq!(DraftVocab::default_for(f), DraftVocab::Off, "{f:?}");
        }
        for off in ["off", "0", "full", "OFF", " no "] {
            assert_eq!(DraftVocab::parse(off), Ok(DraftVocab::Off), "{off}");
            assert_eq!(DraftVocab::canonical_text(off).unwrap(), "off");
        }
        assert_eq!(DraftVocab::parse("on"), Ok(DraftVocab::Top(DRAFT_VOCAB_ON_ROWS)));
        assert_eq!(DraftVocab::canonical_text("Yes").unwrap(), "on");
        assert_eq!(DraftVocab::parse("40000"), Ok(DraftVocab::Top(40000)));
        assert_eq!(DraftVocab::parse("27%"), Ok(DraftVocab::Frac(0.27)));
        assert_eq!(DraftVocab::parse("0.5"), Ok(DraftVocab::Frac(0.5)));
        assert_eq!(DraftVocab::canonical_text(" 30% ").unwrap(), "30%");
        for bad in ["", "-5", "0%", "101%", "1.5", "0.0", "abc", "12k"] {
            assert!(DraftVocab::parse(bad).is_err(), "{bad:?} must be refused");
        }
        assert_eq!(DraftVocab::Off.top(248320), 0);
        assert_eq!(DraftVocab::Top(65536).top(248320), 65536);
        assert_eq!(DraftVocab::Frac(0.5).top(248320), 124160);
        // the default `on` subset on the Qwen vocabulary: 16-row aligned, keeps <|endoftext|> (248044)
        // and <|im_end|> (248046) proposable, and stays a strict subset of the head
        let rows = draft_vocab_rows(DraftVocab::parse("on").unwrap().top(248320), 248320);
        assert_eq!(rows.len() % MMA_M, 0);
        assert!(rows.contains(&248044) && rows.contains(&248046));
        assert!(rows.len() < 248320 / 2);
    }
