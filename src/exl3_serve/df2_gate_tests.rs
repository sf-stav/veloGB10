//! Unit tests (module `df2_gate_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::*;
    use crate::batch::SpecSource;

    const DIR: Option<&str> = Some("/a/df2/artifact/dir");

    #[test]
    fn df2_gate_engages_every_dflash2_source_on_the_27b_pack_shape() {
        for src in [SpecSource::DFlash2, SpecSource::DFlash2Rq, SpecSource::DFlash2Auto,
                    SpecSource::DFlash2Tree] {
            assert_eq!(df2_gate_reason(src, DIR, 5120, 248_320, 62, 262_144, 1), None,
                       "{src:?} + dir + dims + ctx within the pin must ENGAGE");
        }
    }

    #[test]
    fn df2_gate_source_wins_over_everything() {
        // a non-DF2 source never engages, even with a dir, the right dims and a sane ctx
        for src in [SpecSource::Mtp, SpecSource::Dspark, SpecSource::Plain] {
            assert_eq!(df2_gate_reason(src, DIR, 5120, 248_320, 62, 262_144, 1).as_deref(),
                       Some("source"), "{src:?}");
        }
        // the flag's own parse: unknown values die with the trunk's exact error text; the
        // default (no flag) is dflash2-auto, so a plain boot is a DF2-class boot that falls
        // back on the missing artifact, not on `source`.
        let mut args: Vec<String> = vec!["gb10_inference".into(), "--model-dir".into(), "/m".into()];
        assert!(matches!(resolve_spec_source_exl3(&args), SpecSource::DFlash2Auto));
        args.push("--spec-source".into()); args.push("dflash2-tree".into());
        assert!(matches!(resolve_spec_source_exl3(&args), SpecSource::DFlash2Tree));
        args.push("--spec-source=wharrgarbl".into());
        // (the error path eprintln!s and exits(1): not exercised here — it is the same text
        // main.rs prints, pinned by that file's own tests.)
    }

    #[test]
    fn df2_gate_missing_dir_is_the_default_boot_reason() {
        // dflash2-auto default + no --draft-dir: the old silent-ignore case, now a reason
        assert_eq!(df2_gate_reason(SpecSource::DFlash2Auto, None, 5120, 248_320, 62, 262_144, 1).as_deref(),
                   Some("no --draft-dir"));
        // every DF2-class source without a dir says the same thing
        for src in [SpecSource::DFlash2, SpecSource::DFlash2Rq, SpecSource::DFlash2Tree] {
            assert_eq!(df2_gate_reason(src, None, 5120, 248_320, 62, 262_144, 1).as_deref(),
                       Some("no --draft-dir"));
        }
    }

    #[test]
    fn df2_gate_dims_reason_reports_got_and_want() {
        assert_eq!(df2_gate_reason(SpecSource::DFlash2, DIR, 3072, 248_320, 62, 262_144, 1).as_deref(),
                   Some("dims hidden=3072x5120 vocab=248320x248320 layers=62 (need >61)"));
        assert_eq!(df2_gate_reason(SpecSource::DFlash2, DIR, 5120, 151_936, 62, 262_144, 1).as_deref(),
                   Some("dims hidden=5120x5120 vocab=151936x248320 layers=62 (need >61)"));
        // layers must be MORE than 61: the Flash-Next backbone
        assert_eq!(df2_gate_reason(SpecSource::DFlash2, DIR, 5120, 248_320, 61, 262_144, 1).as_deref(),
                   Some("dims hidden=5120x5120 vocab=248320x248320 layers=61 (need >61)"));
    }

    #[test]
    fn df2_gate_ctx_and_tp_reasons() {
        // one past the artifact's safe ceiling
        assert_eq!(df2_gate_reason(SpecSource::DFlash2, DIR, 5120, 248_320, 62, 262_145, 1).as_deref(),
                   Some("ctx 262145>262144"));
        // EXL27D item 2: the tp reason is RETIRED — a TP boot ENGAGES like TP=1 (the
        // TP-DF2 leg ships the artifact + argv; both ranks run the identical scheduler)
        assert_eq!(df2_gate_reason(SpecSource::DFlash2, DIR, 5120, 248_320, 62, 262_144, 2), None,
                   "tp=2 must ENGAGE now (the single-process contract is retired)");
    }

    #[test]
    fn df2_boot_lines_are_verbatim() {
        let sha = "0123456789abcdef".repeat(4);
        assert_eq!(df2_boot_line_engaged("/d", &sha, 8, false, 1),
                   format!("[exl3-serve] df2: ENGAGED draft=/d sha256={sha} block=8 round-shard=off (DFlash2 chain on EXL3, TP=1; chain verify k<=10 = the pack MTP wide-verify width)"));
        assert_eq!(df2_boot_line_engaged("/d2", &sha, 16, true, 2),
                   format!("[exl3-serve] df2: ENGAGED draft=/d2 sha256={sha} block=16 round-shard=on (DFlash2 chain on EXL3, TP=2; chain verify k<=10 = the pack MTP wide-verify width)"));
        assert_eq!(df2_boot_line_fallback("source"),
                   "[exl3-serve] df2: FALLBACK to pack MTP (source)");
        assert_eq!(df2_boot_line_fallback("no --draft-dir"),
                   "[exl3-serve] df2: FALLBACK to pack MTP (no --draft-dir)");
        assert_eq!(df2_boot_line_fallback("load failed: no such file"),
                   "[exl3-serve] df2: FALLBACK to pack MTP (load failed: no such file)");
    }

    #[test]
    fn df2_metrics_render_carries_the_exact_series_names() {
        let mut o = String::new();
        df2_metrics_append(&mut o);
        for series in ["exl3_df2_engaged ", "exl3_df2_rounds_total ",
                       "exl3_df2_tokens_accepted_total ", "exl3_df2_fallback_mtp_total ",
                       "exl3_df2_dds_bypass_total "] {
            assert!(o.contains(series), "missing series `{series}` in:\n{o}");
        }
    }
