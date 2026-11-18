//! Unit tests (module `fit_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::{fit_parts, fit_plan, FitParts};
    const G: u64 = 1_000_000_000;

    #[test]
    fn fit_plan_refuses_with_the_arithmetic_and_the_way_out() {
        let p = FitParts { weights: 53 * G, table: 33 * G, per_lane: 3 * G, fixed: 19 * G, floor: 5 * G };
        // 1 lane: fits with the table in RAM.
        assert_eq!(fit_plan(p, 1, 124 * G, "auto", 131072), Ok(true));
        // 8 lanes: fits only on SSD; auto serves it, ram is refused and told to use ssd.
        assert_eq!(fit_plan(p, 8, 124 * G, "auto", 131072), Ok(false));
        let e = fit_plan(p, 8, 124 * G, "ram", 131072).unwrap_err();
        assert!(e.contains("--ple-ram ssd") && e.contains("does not fit"), "{e}");
        // 20 lanes: does not fit even on SSD; names the lane and context limits.
        let e = fit_plan(p, 20, 124 * G, "auto", 131072).unwrap_err();
        assert!(e.contains("even with the n-gram table on SSD") && e.contains("--max-batch 15"), "{e}");
        // the weights alone overflow: says TP=2.
        let e = fit_plan(FitParts { weights: 110 * G, ..p }, 1, 124 * G, "ssd", 131072).unwrap_err();
        assert!(e.contains("--tp 2"), "{e}");
    }

    /// MAXBATCH1: the boot-anyway clamp. The same arithmetic fit_plan refuses with becomes the
    /// answer to "what is the largest lane count that BOOTS": the .12 refusal of 2026-10-21
    /// (16 lanes, 262K context, 125.5 GB box) resolves to 8 lanes, the table landing on SSD
    /// exactly like a plain boot that only fits on SSD. None is reserved for "not even one lane
    /// can serve" — the one refusal CF-P1d keeps — and a forced `--ple-ram ram` is honoured
    /// (the explicit user demand), not silently downgraded to SSD.
    #[test]
    fn fit_max_ctx_picks_the_longest_context_one_lane_fits() {
        use super::{fit_max_ctx, fit_plan, FitParts};
        const G: u64 = 1_000_000_000;
        // per-lane KV linear in max_pos: 5.61 GB at 262,145 (the issue-#26 4bpw pack at TP = 1, .11, 2026-11-07)
        let parts_at = |mp: usize| Some(FitParts { weights: 69_900_000_000, table: 39 * G,
            per_lane: (5_610_000_000f64 * mp as f64 / 262_145.0) as u64, fixed: 19_300_000_000, floor: 20 * G });
        let avail = 111_200_000_000;
        let ctx = fit_max_ctx(262_144, 1, avail, "auto", &parts_at).expect("a shorter context fits");
        assert_eq!(ctx % 1024, 0);
        assert!(fit_plan(parts_at(ctx + 1).unwrap(), 1, avail, "auto", ctx + 1).is_ok(), "the clamp fits");
        assert!(fit_plan(parts_at(ctx + 1024 + 1).unwrap(), 1, avail, "auto", ctx + 1024 + 1).is_err(), "1,024 more does not");
        assert!((80_000..110_000).contains(&ctx), "{ctx}");
        // everything fits: the largest candidate below the ask
        assert_eq!(fit_max_ctx(262_144, 1, 400 * G, "auto", &parts_at), Some(261_120));
        // weights alone overflow: no context fits (the refusal stays)
        assert_eq!(fit_max_ctx(262_144, 1, 100 * G, "auto", &parts_at), None);
        // a tiny ask: nothing below 4,096 is offered
        assert_eq!(fit_max_ctx(4096, 1, 400 * G, "auto", &parts_at), None);
    }

    #[test]
    fn fit_max_lanes_picks_the_largest_lane_count_that_boots() {
        use super::fit_max_lanes;
        // The .12 2026-10-21 boot: weights 53.2, working 19.3, floor 5.0, per-lane 5.61 at 262145.
        let p = FitParts { weights: 53_200_000_000, table: 33_000_000_000, per_lane: 5_610_000_000,
                           fixed: 19_300_000_000, floor: 5_000_000_000 };
        assert_eq!(fit_max_lanes(p, 16, 125_500_000_000, "auto", 262145), Some(8),
                   "the refused boot clamps to the 8 lanes that fit (9 lanes = 127.5 GB > 125.5)");
        // A generous box: nothing is clamped away, the table fits in RAM.
        assert_eq!(fit_max_lanes(p, 6, 200 * G, "auto", 262145), Some(6));
        // The clamp never exceeds the asked width.
        assert_eq!(fit_max_lanes(p, 4, 200 * G, "auto", 262145), Some(4));
        // Weights alone overflow: None — CF-P1d's refusal (use --tp 2) survives.
        assert_eq!(fit_max_lanes(FitParts { weights: 110 * G, ..p }, 16, 124 * G, "ssd", 131072), None);
        // Forced --ple-ram ram with a table that cannot sit in RAM: no lane count satisfies the
        // explicit demand → None keeps the refusal; auto on the same box boots 5 lanes.
        assert_eq!(fit_max_lanes(p, 16, 110 * G, "ram", 262145), None);
        assert_eq!(fit_max_lanes(p, 16, 110 * G, "auto", 262145), Some(5));
    }

    /// The real Flash-Next 3.05 pack, where present: the estimate must bracket the 2026-10-02 measurement on .14
    /// (per lane at 131K ~2.8 GB measured; device weights ~53 GB; table 32.6 GB).
    #[test]
    fn flash_next_305_estimate_brackets_the_measurement() {
        let dir = std::path::Path::new(&std::env::var("HOME").unwrap_or_default()).join("models/Qwen3.8-Flash-Next-exl3-3.05bpw");
        if !dir.is_dir() { return; }
        let p = fit_parts(dir.to_str().unwrap(), 131073, 1, true, 1.0625, 4.0).unwrap();
        assert!((2_600_000_000..3_300_000_000).contains(&p.per_lane), "{p:?}");
        assert!((50 * G..56 * G).contains(&p.weights), "{p:?}");
        assert!((32 * G..34 * G).contains(&p.table), "{p:?}");
        // MAXBATCH1: the serve fit reserves the §5 serving-headroom bar (>= 20 GB), not the 5 GB
        // emergency-watchdog floor — a 2026-10-21 .12 boot at the 5 GB floor rode it and died mid-sweep.
        assert!(p.floor >= 20 * G, "serve-fit floor must reserve the serving headroom: {p:?}");
    }
