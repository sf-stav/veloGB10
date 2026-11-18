//! Unit tests (module `qsaregime_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    use super::*;

    const LIM: usize = 2051; // Flash-Next: indexer_budget 2048 + ratio 4 - 1

    #[test]
    fn dense_prefix_counts_only_leading_short_rows_of_a_mixed_step() {
        // one-sided steps are not mixed
        assert_eq!(dense_prefix(&[], LIM), (0, 0));
        assert_eq!(dense_prefix(&[10], LIM), (0, 0));
        assert_eq!(dense_prefix(&[10, 2051, 300], LIM), (0, 0));
        assert_eq!(dense_prefix(&[2052, 20_000], LIM), (0, 0));
        // the switch itself: 2051 is dense, 2052 is the first sparse position
        assert_eq!(dense_prefix(&[2051, 2052], LIM), (1, 0));
        assert_eq!(dense_prefix(&[1500, 1600, 20_000], LIM), (2, 0));
        assert_eq!(dense_prefix(&[1500, 20_000, 30_000, 40_000], LIM), (1, 0));
        // short rows behind a long row are reported, never counted
        assert_eq!(dense_prefix(&[20_000, 1500], LIM), (0, 1));
        assert_eq!(dense_prefix(&[1500, 20_000, 1600, 30_000, 7], LIM), (1, 2));
    }

    #[test]
    fn regime_order_is_a_stable_partition_and_the_identity_on_one_sided_steps() {
        assert_eq!(regime_order(&[5, 9, 100], LIM), vec![0, 1, 2]);
        assert_eq!(regime_order(&[5000, 9000, 2052], LIM), vec![0, 1, 2]);
        assert_eq!(regime_order(&[20_000, 1500, 30_000, 7, 2051, 2052], LIM), vec![1, 3, 4, 0, 2, 5]);
        // no switch (no indexer / QSA off / the fix off): the caller passes usize::MAX
        assert_eq!(regime_order(&[20_000, 1500, 30_000], usize::MAX), vec![0, 1, 2]);
    }

    /// After regime_order every short row is in the dense prefix, and the prefix never holds a row
    /// past the switch (the dense kernels' score plane ends at DV3_TS positions).
    #[test]
    fn ordered_steps_have_no_late_rows_and_the_prefix_never_holds_a_long_row() {
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..4000 {
            let mut pos = Vec::new();
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let n = 1 + (x >> 60) as usize; // 1..=16 rows
            for _ in 0..n {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                pos.push(match (x >> 61) & 3 { 0 => LIM, 1 => LIM + 1, 2 => (x >> 40) as usize % (LIM + 1), _ => LIM + 1 + (x >> 40) as usize % 200_000 });
            }
            let (md0, late0) = dense_prefix(&pos, LIM);
            // as given: the prefix stops at the first long row, and prefix + late = every short row of a mixed step
            assert!(md0 == pos.len() || pos[md0] > LIM || md0 + late0 == 0, "{pos:?}");
            let ord: Vec<usize> = regime_order(&pos, LIM).into_iter().map(|i| pos[i]).collect();
            let (md, late) = dense_prefix(&ord, LIM);
            let short = pos.iter().filter(|&&p| p <= LIM).count();
            let mixed = short > 0 && short < pos.len();
            assert_eq!(late, 0, "{pos:?}");
            assert_eq!(md, if mixed { short } else { 0 }, "{pos:?}");
            assert_eq!(md0 + late0, md, "{pos:?}: unsorted, the same short rows are split into prefix + late");
            assert!(md < pos.len(), "a mixed step keeps at least one sparse row");
            // forward_step looks at the first m positions only: any prefix of the list is safe (no row
            // count can go negative, and a long position beyond the rows makes the step one-sided)
            for m in 0..=pos.len() {
                let (a, b) = dense_prefix(&ord[..m], LIM);
                assert!(a + b <= m && (m == 0 || a < m), "{ord:?} m {m}");
            }
            // every dense row fits the dense-v3 score plane
            assert!(ord[..md].iter().all(|&p| p < DV3_TS));
        }
    }

    /// No speculative round straddles the switch: its rows p..p+depth are all dense or all sparse,
    /// the depth is untouched wherever the full round already is on one side, and a round is only
    /// ever shortened, down to a plain step exactly at the switch.
    #[test]
    fn a_round_never_straddles_the_switch() {
        for p in LIM.saturating_sub(12)..LIM + 12 {
            for k in 0..=7usize {
                let d = regime_depth_at(p, k, Some(LIM));
                assert!(d <= k);
                assert!(p > LIM || p + d <= LIM, "p {p} k {k} -> {d} straddles");
                if p > LIM || p + k <= LIM { assert_eq!(d, k, "p {p} k {k}: a one-sided round must keep its depth"); }
                else { assert_eq!(d, LIM - p); }
                assert_eq!(regime_depth_at(p, k, None), k);
            }
        }
        assert_eq!(regime_depth_at(LIM, 7, Some(LIM)), 0);      // the switch itself: a plain step
        assert_eq!(regime_depth_at(LIM + 1, 7, Some(LIM)), 7);  // the first sparse position
        assert_eq!(regime_depth_at(0, 7, Some(LIM)), 7);
    }

    /// S-B9-REL-SMALLFIXES item 2: the round of the callers outside the scheduler (--bench-mtp, the
    /// draft-quality bench, the KV-NLL probe: `mtp_round_clamped`) at positions 2,049..2,053 — the
    /// serve's clamp exactly, and at the switch itself the plain step, never `mtp_round` with k = 0.
    #[test]
    fn the_bench_round_is_clamped_at_the_switch_like_the_serve() {
        let plan = |p: usize, k: usize| clamped_round_plan(p, k, Some(LIM));
        assert_eq!(LIM, 2051);
        // the gate's --depth 2, the serve's 5, WP23's 7
        assert_eq!((plan(2049, 2), plan(2049, 5), plan(2049, 7)), (Some(2), Some(2), Some(2))); // rows 2049..=2051
        assert_eq!((plan(2050, 2), plan(2050, 5), plan(2050, 7)), (Some(1), Some(1), Some(1))); // rows 2050, 2051
        assert_eq!((plan(2051, 2), plan(2051, 5), plan(2051, 7)), (None, None, None));          // the switch: a plain step
        assert_eq!((plan(2052, 2), plan(2052, 5), plan(2052, 7)), (Some(2), Some(5), Some(7))); // first sparse row: untouched
        assert_eq!((plan(2053, 2), plan(2053, 5), plan(2053, 7)), (Some(2), Some(5), Some(7)));
        for p in 2040..2064usize {
            for k in 1..=7usize {
                // the same number the serve's step_mtp gets from regime_depth
                assert_eq!(plan(p, k).unwrap_or(0), regime_depth_at(p, k, Some(LIM)), "p {p} k {k}");
                match plan(p, k) {
                    Some(d) => {
                        assert!(d >= 1 && d <= k, "p {p} k {k} -> {d}");
                        assert!(p > LIM || p + d <= LIM, "p {p} k {k} -> {d}: verify rows p..=p+d straddle the switch");
                    }
                    None => assert_eq!(p, LIM, "only the switch itself has no room for a draft (p {p} k {k})"),
                }
                // no switch (no indexer, --qsa-exl3 0, --qsa-row-regime 0): the round is untouched
                assert_eq!(clamped_round_plan(p, k, None), Some(k));
            }
            assert_eq!(plan(p, 0), None, "a zero-depth round is a plain step (mtp_round refuses k = 0)");
        }
    }

    /// The unit suite cannot run a round (no Scratch, AGENTS 1a-ter (4)): the callers are pinned in
    /// the source. Every direct call of the unclamped round is listed here; a new one fails this
    /// test until it is either routed through `mtp_round_clamped` or added with its reason.
    #[test]
    fn every_direct_caller_of_the_unclamped_round_is_listed() {
        // the patterns are built from pieces so this test's own text never matches them
        let direct = format!(".mtp_{}(", "round");
        let clamped = format!(".mtp_{}_clamped(", "round");
        let n = |s: &str, pat: &str| s.matches(pat).count();
        let (fwd, serve, kvnll, tune, regime) = (src_text!("exl3_forward"), src_text!("exl3_serve"),
            src_text!("exl3_kvnll"), src_text!("exl3_autotune"), src_text!("exl3_regime"));
        assert_eq!(n(fwd, &direct), 1, "exl3_forward.rs: only mtp_round_clamped itself calls the unclamped round");
        assert_eq!(n(fwd, &clamped), 2, "the LOSSLESS --bench-mtp and the draft-quality bench go through the clamp");
        assert_eq!((n(kvnll, &direct), n(kvnll, &clamped)), (0, 1), "the KV-NLL acceptance probe goes through the clamp");
        assert_eq!(n(regime, &direct), 0);
        assert_eq!(n(serve, &direct), 1, "exl3_serve.rs: step_mtp, which clamps k_eff itself");
        let mtp = &serve[serve.find("fn step_mtp(&mut self, s: usize, solo: bool)").expect("step_mtp")..];
        // S-B9-CF-PACK4: the clamps moved into `round_depth` (one definition for step_mtp and the packed
        // arm's plan): step_mtp must take its depth from it before the round, and it must hold the switch clamp
        assert!(mtp.find("self.round_depth(p, remaining, think_room)").expect("step_mtp's clamp") < mtp.find(direct.as_str()).expect("step_mtp's round"),
                "the serve clamps before it runs the round");
        // WIDEVERIFY1 refinement 2: round_depth delegates to round_depth_for(self.mtp_k, ..) (the wide copy round
        // clamps COPY_MAX_K through the same body); the clamps live in round_depth_for
        assert!(serve.contains("        self.round_depth_for(self.mtp_k, p, remaining, think_room)\n    }"), "round_depth = round_depth_for at mtp_k");
        let rd = &serve[serve.find("fn round_depth_for(&self").expect("round_depth_for")..];
        let rd = &rd[..rd.find("\n    }\n").expect("end of round_depth")];
        assert!(rd.find("saturating_sub(p + 2)").unwrap() < rd.find("rem_clamp").unwrap()
                && rd.find("rem_clamp").unwrap() < rd.find("self.model.regime_depth(p, k_eff)").expect("round_depth clamps at the switch"),
                "round_depth = window clamp, then budget clamp, then the switch clamp, in step_mtp's old order");
        // THINKBUDGET1: the thinking-budget clamp sits after the max_new clamp and before the switch clamp,
        // and it is NOT behind the rem_clamp flag (a round past the budget would force mid-round)
        let tb = rd.find("if let Some(r) = think_room").expect("round_depth clamps at the thinking budget");
        assert!(rd.find("rem_clamp").unwrap() < tb && tb < rd.find("self.model.regime_depth(p, k_eff)").unwrap());
        let rc = &rd[rd.find("rem_clamp").unwrap()..tb];
        assert!(rc.matches('{').count() == rc.matches('}').count(), "the thinking clamp is outside the rem_clamp block");
        // NOT clamped, on purpose (REL_SMALLFIXES_REPORT item 2): the autotune passes run their rounds
        // inside the tuner's armed measurement window; a clamp there needs a GPU-verified change.
        assert_eq!(n(tune, &direct), 3, "exl3_autotune.rs: run_mtp_pass, Session::one_round, gen_once (fixed chain)");
        // the helper is the serve's rule and the serve's plain step, in this order
        let h = &fwd[fwd.find(format!("pub fn mtp_{}_clamped(", "round").as_str()).expect("the helper")..];
        let h = &h[..h.find(format!("pub fn mtp_{}(", "round").as_str()).expect("the round follows the helper")];
        let at = |pat: &str| h.find(pat).unwrap_or_else(|| panic!("mtp_round_clamped must contain `{pat}`"));
        let (prime, step, snap) = (at("self.mtp_plain_prime(sc, head, slot, b, p, tap_row)?;"),
                                   at("self.forward_step(sc, &[b], &[p], &[slot], None)?;"), at("self.tap_snapshot(sc, slot)?;"));
        assert!(prime < step && step < snap, "the plain step is prime -> forward_step -> tap snapshot, as step_mtp");
        at("if self.qsa_live(usize::MAX) && row_regime_on() { Some(self.qsa_limit()) } else { None }");
    }

    /// A mixed step is its own graph per (width, dense rows); every pre-existing key is unchanged
    /// and no mixed key lands on another key family or on the penalty / vp / scratch-id bits.
    #[test]
    fn mixed_step_graph_keys_are_distinct_and_leave_the_old_keys_alone() {
        assert_eq!(mixed_key_bits(0), 0);
        let mut seen = std::collections::HashSet::new();
        for m in 1..=16usize {
            assert!(seen.insert(gkey_of(m, false, 0)));
            for qb in [0usize, 10, 63] {
                for md in 0..m {
                    let k = gkey_of(m, true, qb) | mixed_key_bits(md);
                    assert!(seen.insert(k), "duplicate step key m {m} qb {qb} md {md}");
                    assert_eq!(k >> 27, 0, "step key reaches the family / penalty / vp / scratch bits");
                    assert_eq!(k & 0xFF, m);
                }
            }
        }
        // every other graph family lives at or above bit 30
        for (name, k) in [("verify", vkey_of(0, 1, false, 0)), ("draft", dkey_of(0, 1, false, 0)), ("prime", pkey_of(0, 0, false, 0))] {
            assert!(k >> 30 != 0, "{name} keys must not share the step family's range");
        }
    }

    /// The unit suite cannot build a Scratch (AGENTS 1a-ter (4)), so the wiring is pinned in the
    /// source: forward_step computes md once and hands it to the eager launches AND the capture,
    /// the boot precapture stays one-regime, and the serve orders the lanes before staging a row.
    #[test]
    fn the_per_row_regime_is_wired_through_step_capture_and_scheduler() {
        let f = src_text!("exl3_forward");
        // each pattern is given in two pieces so this test's own text never matches it
        let once = |a: &str, b: &str| {
            let pat = format!("{a}{b}");
            assert_eq!(f.matches(pat.as_str()).count(), 1, "expected exactly one `{pat}`");
        };
        once("let (md, late) = dense_prefix(", "rows, self.qsa_limit());");
        once("let gkey = gkey_of(m, qsa, qb) | ", "mixed_key_bits(md) |");
        once("self.forward_step_kernels(&l, sc, m, ", "ladder.as_deref_mut(), dump_active, qsa, md)?;");
        once("self.capture_step_graph(&l, sc, m, ", "gkey, qsa, md);");
        once("self.capture_step_graph(&l, sc, m, ", "key, sparse, 0);");
        once("self.dense_attn_step(l, sc, a, att_i, ", "md, nh_p, hd_p)?;");
        let s = src_text!("exl3_serve");
        let body = &s[s.find("fn step_plain(&mut self, slots: &[usize], sync_head: bool)").expect("step_plain")..];
        let (o, st) = (body.find("let ordered = self.regime_ordered(slots);").expect("step_plain orders its lanes"),
                       body.find("let toks: Vec<i32> = slots.iter()").expect("step_plain stages rows"));
        assert!(o < st && o < body.find("mtp_plain_prime").unwrap(), "the order must be fixed before any row is staged");
        let sh = body.find("let slots: &[usize] = &ordered;").expect("step_plain must USE the ordered list (shadow `slots`)");
        assert!(o < sh && sh < st && sh < body.find("mtp_plain_prime").unwrap(), "every row of the step must come from the ordered list");
        let mtp = &s[s.find("fn step_mtp(&mut self, s: usize, solo: bool)").expect("step_mtp")..];
        let (c, r) = (mtp.find("self.round_depth(p, remaining, think_room)").expect("step_mtp clamps the round at the switch (round_depth)"),
                      mtp.find("let plain_step = control || k_eff == 0;").expect("step_mtp's plain-step rule"));
        assert!(c < r, "the switch clamp must precede the plain-step decision");
        let t = src_text!("opts_table");
        assert!(t.contains("flag: \"qsa-row-regime\", env: \"\", ty: Ty::Bool, def: \"on\""), "the option row (no environment variable)");
    }

    /// S-B9-REL-VISIONFIX2 V2a: the `pfx1-ltplans` thread runs at an arbitrary time after boot, while the
    /// scheduler thread captures graphs lazily. A cublasLtDestroy on it invalidates the scheduler's
    /// open capture (not refused: it returns ok; measured in tests/vision_capture_race_test.rs), so the
    /// thread must never call it. The body is read from the source, comment lines excluded.
    #[test]
    fn the_ltplans_thread_never_destroys_a_library_handle() {
        let f = src_text!("exl3_forward");
        let a = f.find("pub fn lt_prewarm_start(&self)").expect("lt_prewarm_start");
        let b = a + f[a..].find("/// WP04 (PLAN/SURPASS_PLAN_2026-09-26.md)").expect("end of lt_prewarm_start");
        let code: String = f[a..b].lines().filter(|l| !l.trim_start().starts_with("//")).collect::<Vec<_>>().join("\n");
        for banned in [["destroy", "_handle"].concat(), ["cublasLt", "Destroy"].concat(), ["cublas", "Destroy"].concat()] {
            assert!(!code.contains(banned.as_str()), "the ltplans thread names `{banned}`: it would invalidate a capture on the scheduler thread");
        }
        assert!(code.contains("lt::create_handle()") && code.contains("lt_plan_build("), "the guard reads the right function body");
    }
