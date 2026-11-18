//! Unit tests of the autotune core and the tuner (the former inline `mod tests`; same module name).

    use super::*;

    /// The LI schedule is balanced: every block of `arms` consecutive draws is a permutation, the
    /// reported in-block position is the draw's place in its block, and over a whole period every
    /// arm takes every in-block position equally often (v2: a Williams walk, not random blocks).
    #[test]
    fn li_schedule_is_balanced_blocks() {
        for arms in 1..=5usize {
            for seed in [0u64, 3, 11] {
                let mut s = LiSched::new(arms, seed);
                let p = tune::williams_period(arms);
                let mut at = vec![vec![0usize; arms]; arms];
                for _ in 0..p {
                    let draws: Vec<(usize, u8)> = (0..arms).map(|_| s.next()).collect();
                    let mut seen: Vec<usize> = draws.iter().map(|d| d.0).collect();
                    seen.sort();
                    assert_eq!(seen, (0..arms).collect::<Vec<_>>());
                    for (j, d) in draws.iter().enumerate() {
                        assert_eq!(d.1 as usize, j);
                        at[d.0][j] += 1;
                    }
                }
                for a in 0..arms { assert_eq!(at[a], vec![p / arms; arms], "arms {arms} seed {seed}"); }
            }
        }
    }

    /// The SR block: the warm-up(s) come first and only when the base is runnable; the timed arms
    /// follow in the row's order with positions 0..n (a skipped arm consumes no position).
    #[test]
    fn block_plan_puts_the_warmup_first() {
        let p = block_plan(&[2, 0, 1], 2);
        assert_eq!(p, vec![Step::Warm, Step::Warm, Step::Timed { arm: 2, apos: 0 }, Step::Timed { arm: 0, apos: 1 },
                           Step::Timed { arm: 1, apos: 2 }]);
        // no base graph (memory cap): no warm-up, positions stay dense
        assert_eq!(block_plan(&[2, 1], 1), vec![Step::Timed { arm: 2, apos: 0 }, Step::Timed { arm: 1, apos: 1 }]);
        assert_eq!(block_plan(&[0], 0), vec![Step::Timed { arm: 0, apos: 0 }]);
        // cells separate units, widths and regimes
        assert_ne!(order_cell(0, 7, false), order_cell(1, 7, false));
        assert_ne!(order_cell(1, 3, false), order_cell(1, 6, false));
        assert_ne!(order_cell(1, 3, false), order_cell(1, 3, true));
    }

    /// Region words: a partial trailing word is skipped, alignment is enforced.
    #[test]
    fn digest_regions_are_word_aligned() {
        assert_eq!(reg(0x1000, 10).unwrap().words, 2);
        assert!(reg(0x1002, 8).is_err());
        assert_eq!(fnv64(&[1, 2, 3]), fnv64(&[1, 2, 3]));
        assert_ne!(fnv64(&[1, 2, 3]), fnv64(&[3, 2, 1]));
    }

    fn smp(unit: &'static str, phase: usize, m: usize, arm: usize, round: usize, ms: f64, ok: bool) -> Sample {
        Sample { unit, phase, m, arm, prompt: 0, round, pos: 100 + round, regime: 0, ms, digest_ok: ok, mode: "sr",
                 apos: arm.min(7) as u8, npos: 2 }
    }

    /// Pairing: only the same round's base sample pairs; digest failures and other phases are out;
    /// the Round objective adds the draft delta at the served passes-per-round share.
    #[test]
    fn pair_obs_pairs_within_rounds_only() {
        let s = vec![
            smp("verify_dry", 1, 3, 0, 10, 40.0, true), smp("verify_dry", 1, 3, 1, 10, 39.5, true),
            smp("draft", 1, 7, 0, 10, 8.0, true), smp("draft", 1, 7, 1, 10, 7.3, true),
            smp("verify_dry", 1, 6, 0, 11, 48.0, true), smp("verify_dry", 1, 6, 1, 11, 48.2, true),
            // round 12: the candidate's digest failed -> no pair
            smp("verify_dry", 1, 5, 0, 12, 45.0, true), smp("verify_dry", 1, 5, 1, 12, 44.0, false),
            // another phase -> ignored
            smp("verify_dry", 2, 3, 0, 13, 40.0, true), smp("verify_dry", 2, 3, 1, 13, 30.0, true),
            // the canonical execution (arm usize::MAX) never pairs
            smp("verify", 1, 8, usize::MAX, 10, 55.0, true),
        ];
        let none = PosTable::new();
        let v = pair_obs(&s, &none, 1, Target::Verify, 2, 7, 0);
        assert_eq!(v[1].len(), 2);
        assert!((v[1][0].1 - -0.5).abs() < 1e-12 && v[1][0].0 == 3);
        assert!((v[1][1].1 - 0.2).abs() < 1e-9 && v[1][1].0 == 6);
        let d = pair_obs(&s, &none, 1, Target::Draft, 2, 7, 0);
        assert_eq!(d[1].len(), 1);
        assert!((d[1][0].1 - -0.7).abs() < 1e-9);
        let r = pair_obs(&s, &none, 1, Target::Round, 2, 7, 0);
        assert!(pair_obs(&s, &none, 1, Target::Verify, 2, 7, 1)[1].is_empty()); // another regime never pairs
        // v2: a cell's fitted position offsets are removed before pairing (base at apos 0, arm at 1)
        let mut offs = PosTable::new();
        offs.insert(("verify_dry", 3, 0, 2), vec![0.25, -0.25]);
        let vs = pair_obs(&s, &offs, 1, Target::Verify, 2, 7, 0);
        assert!((vs[1][0].1 - (-0.5 + 0.25 + 0.25)).abs() < 1e-9, "{:?}", vs[1]);
        assert!((vs[1][1].1 - 0.2).abs() < 1e-9); // m = 6 has no offsets: unchanged
        assert_eq!(r[1].len(), 1); // round 11 has no draft pair
        let want = -0.5 + DRAFT_PASSES_PER_ROUND / 7.0 * -0.7;
        assert!((r[1][0].1 - want).abs() < 1e-9, "{:?}", r[1]);
        assert!(v[0].is_empty()); // the base never pairs with itself
    }

    /// Arm signatures: equal assignments share cached graphs; a different value, key or knob does not.
    #[test]
    fn arm_signatures_identify_configurations() {
        let a = Arm { name: "a".into(), assign: vec![(&T_WP09_AM_NB, None, 16)] };
        let b = Arm { name: "b".into(), assign: vec![(&T_WP09_AM_NB, None, 16)] };
        let c = Arm { name: "c".into(), assign: vec![(&T_WP09_AM_NB, None, 32)] };
        let d = Arm { name: "d".into(), assign: vec![(&T_WP09_AM_NB, Some(tune::Key::G), 16)] };
        let e = Arm { name: "e".into(), assign: vec![(&T_HC_BXR, None, 16)] };
        assert_eq!(a.sig(), b.sig());
        assert_ne!(a.sig(), c.sig());
        assert_ne!(a.sig(), d.sig());
        assert_ne!(a.sig(), e.sig());
        assert_ne!(Arm::base().sig(), a.sig());
    }

    #[test]
    fn clock_and_throttle_fields_parse() {
        let e = serde_json::json!({"clocks.sm": "2405", "clocks_event_reasons.active": "0x0000000000000001"});
        assert_eq!(sm_clock(&e), Some(2405.0));
        assert!(!throttled(&e)); // 0x1 = GPU idle, not a throttle
        let t = serde_json::json!({"clocks.sm": "unknown", "clocks_event_reasons.active": "0x0000000000000040"});
        assert_eq!(sm_clock(&t), None); // unsupported reads "unknown", never zero
        assert!(throttled(&t)); // HW thermal slowdown
    }

    /// Every registered entry is either in a default search list, a declared pair, the per-shape
    /// PSK/chain families, or NOT_SEARCHED / pruned by class-scope-xcheck (the report accounts for all).
    #[test]
    fn every_registry_entry_has_a_search_status() {
        for t in tune::REGISTRY.iter() {
            let listed = DENSE_SPECS.iter().any(|x| x.0 == t.id) || SPARSE_SPECS.iter().any(|x| x.0 == t.id)
                || PAIRS.iter().any(|p| p.0 == t.id || p.2 == t.id)
                || ["psk.g", "psk.g_pair", "chain.pair"].contains(&t.id)
                || NOT_SEARCHED.iter().any(|x| x.0 == t.id);
            let pruned = t.class != tune::Class::S || t.xcheck.is_empty()
                || matches!(t.scope, tune::Scope::Load | tune::Scope::Chunk);
            assert!(listed || pruned, "{} is neither searched nor listed as not searched", t.id);
            if listed && !NOT_SEARCHED.iter().any(|x| x.0 == t.id) {
                assert_eq!(t.class, tune::Class::S, "{} searched but not class S", t.id);
                assert!(!t.xcheck.is_empty(), "{} searched without a proven domain", t.id);
            }
        }
        // pair grids stay within the design's 9 combinations
        for (_, a, _, b, _) in PAIRS { assert!(a.len() * b.len() - 1 <= 9); }
    }

    // ---- §5.2 v2: a synthetic sampler through the production order / plan / pairing code ----

    struct Xs(u64);
    impl Xs {
        fn u(&mut self) -> f64 {
            self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        /// symmetric triangular noise, |x| <= amp
        fn tri(&mut self, amp: f64) -> f64 { (self.u() + self.u() - 1.0) * amp }
    }

    /// One self-test cell (verify_dry m = 6) over `rounds` rounds with arms base, base2 (the A/A
    /// twin) and spin (+0.200 ms). Every replay i of a round (warm-ups included) pays `pen(i, P_r)`
    /// on top of the round's content and the arm's effect; P_r varies per round (the cold start is
    /// not a constant). v1 = the old protocol: a fresh Fisher–Yates order every round (the removed
    /// Session::shuffled_arms), no warm-up. v2 = the production path: BalancedOrder rows per cell,
    /// block_plan with the warm-up, Samples recorded as resolve() records them. Returns the samples
    /// and how often each arm ran first among the TIMED replays.
    fn simulate(v2: bool, warmup: usize, rounds: usize, seed: u64, pen: &dyn Fn(usize, f64) -> f64)
        -> (Vec<Sample>, Vec<usize>) {
        let eff = [0.0, 0.0, 0.2];
        let n = eff.len();
        let mut rng = Xs(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut order = tune::BalancedOrder::new(seed);
        let (mut out, mut first) = (Vec::new(), vec![0usize; n]);
        for r in 0..rounds {
            let content = 30.0 + 10.0 * rng.u();
            let p_r = 0.5 + rng.u();
            let row = if v2 { order.next(order_cell(1, 6, false), n) } else {
                let mut v: Vec<usize> = (0..n).collect();
                for i in (1..n).rev() { let j = ((rng.u() * (i + 1) as f64) as usize).min(i); v.swap(i, j); }
                v
            };
            first[row[0]] += 1;
            for (i, st) in block_plan(&row, if v2 { warmup } else { 0 }).into_iter().enumerate() {
                let (arm, apos, mode, e) = match st {
                    Step::Warm => (WARMUP_ARM, NO_APOS, "warm", 0.0),
                    Step::Timed { arm, apos } => (arm, apos, "sr", eff[arm]),
                };
                let ms = content + e + pen(i, p_r) + rng.tri(0.03);
                out.push(Sample { unit: "verify_dry", phase: 1, m: 6, arm, prompt: 0, round: r, pos: 100, regime: 0, ms,
                                  digest_ok: true, mode, apos, npos: n as u8 });
            }
        }
        (out, first)
    }

    fn est(s: &[Sample], arm: usize, offs: &PosTable) -> tune::Est {
        tune::hl_paired(&paired(s, "verify_dry", 6, arm, offs)).expect("pairs")
    }

    /// The receipts' failure, reproduced and fixed: a first-replay penalty (P_r ~ 0.5..1.5 ms, the
    /// .14 evidence) biases the v1 protocol's paired HL by the realized first-position imbalance —
    /// the §5.7 A/A / known-positive criteria fail on many seeds, and the error's sign follows the
    /// imbalance — while v2 (warm-up + balanced order, plain or stratified) is unbiased: every arm
    /// first equally often and the criteria hold on every seed.
    #[test]
    fn synthetic_first_position_penalty_biases_v1_not_v2() {
        let pen = |i: usize, p: f64| if i == 0 { p } else { 0.0 };
        let (rounds, seeds) = (120usize, 1..=24u64);
        let mut v1_fail = 0usize;
        let mut signed = 0usize;
        let mut v1_max = 0.0f64;
        let mut v2_max = 0.0f64;
        let (mut v2_cover, mut v2_n) = (0usize, 0usize);
        let mut log = Vec::new();
        for seed in seeds.clone() {
            // ---- v1
            let (s, first) = simulate(false, 0, rounds, seed, &pen);
            let none = PosTable::new();
            let (aa, sp) = (est(&s, 1, &none), est(&s, 2, &none));
            if !(aa_ok(&aa, 0.02) && spin_ok(&sp, 0.2)) { v1_fail += 1; }
            let d = first[1] as i64 - first[0] as i64; // base2 first more often -> base2 reads slower
            if d.abs() >= 10 {
                signed += 1;
                assert_eq!(aa.est > 0.0, d > 0, "v1 seed {seed}: first {first:?} A/A HL {:+.4}", aa.est);
            }
            v1_max = v1_max.max(aa.est.abs());
            log.push(format!("seed {seed}: v1 first {first:?} A/A {:+.4} spin {:+.4} -> {}", aa.est, sp.est,
                             if aa_ok(&aa, 0.02) && spin_ok(&sp, 0.2) { "pass" } else { "FAIL" }));
            // ---- v2 (the same content/noise model)
            let (s2, first2) = simulate(true, 1, rounds, seed, &pen);
            assert_eq!(first2, vec![rounds / 3; 3], "v2 seed {seed}: the balanced order must put every arm first equally often");
            // no warm-up sample is ever paired, and every timed replay carries its position
            assert!(s2.iter().filter(|x| x.mode == "warm").all(|x| x.arm == WARMUP_ARM && x.apos == NO_APOS));
            assert_eq!(s2.iter().filter(|x| x.mode == "warm").count(), rounds);
            for offs in [PosTable::new(), pos_table(&s2, &|_| true)] {
                let (aa2, sp2) = (est(&s2, 1, &offs), est(&s2, 2, &offs));
                // no bias: the estimate is inside the A/A band on EVERY seed; the 95% CI itself
                // misses 0 by chance on ~5% of seeds (noise only), so coverage is counted
                assert!(aa2.est.abs() < 0.02, "v2 seed {seed} (stratified {}): A/A {aa2:?}", !offs.is_empty());
                assert!(spin_ok(&sp2, 0.2) && (sp2.est - 0.2).abs() < 0.02, "v2 seed {seed}: spin {sp2:?}");
                v2_cover += (aa2.lo <= 0.0 && 0.0 <= aa2.hi) as usize;
                v2_n += 1;
                v2_max = v2_max.max(aa2.est.abs());
            }
        }
        println!("{}\nv1: {v1_fail}/24 seeds fail, max |A/A| {v1_max:.4} ms; v2: max |A/A| {v2_max:.4} ms, A/A CI covers 0 \
                  {v2_cover}/{v2_n}", log.join("\n"));
        assert!(v2_cover * 10 >= v2_n * 9, "v2 A/A coverage {v2_cover}/{v2_n} (nominal 95%)");
        assert!(v1_fail >= 6, "the v1 protocol should fail the §5.7 criteria on many seeds ({v1_fail}/24):\n{}", log.join("\n"));
        assert!(signed >= 3, "too few strongly imbalanced v1 seeds to check the sign rule ({signed}):\n{}", log.join("\n"));
        assert!(v1_max > 3.0 * v2_max && v1_max > 0.04, "v1 max |A/A| {v1_max} vs v2 {v2_max}");
    }

    /// A residual position gradient the warm-up does not absorb (replays 1..3 pay +0.6 / +0.1 /
    /// -0.7 ms): the balanced order keeps the plain paired HL centred but wide; the position model
    /// recovers the gradient and the stratified CI is several times tighter, centred on the truth.
    #[test]
    fn stratification_removes_a_residual_gradient() {
        let pen = |i: usize, p: f64| match i { 0 => p, 1 => 0.6, 2 => 0.1, _ => -0.7 };
        let (s, _) = simulate(true, 1, 240, 7, &pen);
        let offs = pos_table(&s, &|_| true);
        let f = &offs[&("verify_dry", 6, 0, 3)];
        for (p, want) in [0.6, 0.1, -0.7].iter().enumerate() { assert!((f[p] - want).abs() < 0.02, "{f:?}"); }
        let none = PosTable::new();
        let (plain, strat) = (est(&s, 1, &none), est(&s, 1, &offs));
        assert!(plain.lo <= 0.0 && 0.0 <= plain.hi, "{plain:?}");
        assert!(aa_ok(&strat, 0.02), "{strat:?}");
        assert!((strat.hi - strat.lo) * 4.0 < plain.hi - plain.lo, "stratified {strat:?} vs plain {plain:?}");
        let sp = est(&s, 2, &offs);
        assert!((sp.est - 0.2).abs() < 0.01, "{sp:?}");
        // the report view: the warm-up and the timed positions, with the fitted offsets
        let pe = position_effect_json(&s, &offs);
        let c = &pe.as_array().unwrap()[0];
        assert_eq!(c["n_pos"], 3);
        assert_eq!(c["rounds"], 240);
        assert!(c["warmup_median_ms"].as_f64().unwrap() > c["median_ms_by_pos"][2].as_f64().unwrap());
        assert!((c["offsets_ms"][2].as_f64().unwrap() + 0.7).abs() < 0.02);
    }

    #[test]
    fn protocol_flags_parse_and_refuse() {
        let a = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(protocol_args(&a(&["--autotune"])).unwrap(), (2, true)); // v3: warm-up default 2
        assert_eq!(protocol_args(&a(&["--autotune-warmup", "1"])).unwrap(), (1, true));
        assert_eq!(protocol_args(&a(&["--autotune-warmup", "0", "--autotune-stratify", "off"])).unwrap(), (0, false));
        assert_eq!(protocol_args(&a(&["--autotune-warmup", "4"])).unwrap(), (4, true));
        assert!(protocol_args(&a(&["--autotune-warmup", "5"])).is_err());
        assert!(protocol_args(&a(&["--autotune-stratify", "maybe"])).is_err());
        // v3: the self-test's A/A threshold
        assert_eq!(aa_args(&a(&["--autotune"])).unwrap(), AaRule::DEFAULT);
        assert_eq!(aa_args(&a(&["--autotune-selftest-tau", "0.037", "--autotune-selftest-tau-pct", "0.1"])).unwrap(),
                   AaRule { abs_ms: 0.037, pct: 0.1 });
        assert!(aa_args(&a(&["--autotune-selftest-tau", "0"])).is_err());
        assert!(aa_args(&a(&["--autotune-selftest-tau", "x"])).is_err());
        assert!(aa_args(&a(&["--autotune-selftest-tau-pct", "-1"])).is_err());
    }

    /// v3 A/A rule on the .13 receipts (2026-09-27 05:05-05:17 UTC; stratified HL / CI95 and the base
    /// medians of each cell): tau = max(0.02 ms, 0.05 % of the median). The rule is what the brief
    /// specified; the receipt that motivated it (selftest-1790486005 verify m = 3) still FAILS it at
    /// the defaults — the CI's upper end (+0.0368) leaves +-0.020 — and passes from tau 0.0368 ms up.
    #[test]
    fn aa_rule_on_the_selftest_receipts() {
        let r = AaRule::DEFAULT;
        let est = |n: usize, hl: f64, lo: f64, hi: f64| tune::Est { n, est: hl, lo, hi };
        // (receipt cell, estimate, base median ms, verdict under v3 defaults, verdict under v1/v2)
        let cells = [
            ("1790485556 verify m3", est(127, -0.0452, -0.1378, 0.0280), 31.798, false, false),
            ("1790485556 verify m6", est(126, 0.0248, -0.0399, 0.0993), 39.055, false, false),
            ("1790485556 draft m7", est(253, 0.0014, -0.0030, 0.0063), 5.758, true, true),
            ("1790485758 verify m3", est(236, -0.0021, -0.0631, 0.0518), 31.984, true, true),
            ("1790485758 verify m6", est(236, 0.0260, -0.0221, 0.0842), 39.352, false, false),
            ("1790485758 draft m7", est(472, -0.0001, -0.0036, 0.0036), 5.724, true, true),
            ("1790486005 verify m3", est(236, 0.0188, 0.0005, 0.0368), 31.917, false, false),
            ("1790486005 verify m6", est(236, -0.0037, -0.0227, 0.0124), 39.345, true, true),
            ("1790486005 draft m7", est(472, 0.0019, -0.0045, 0.0085), 5.832, true, true),
        ];
        for (name, e, med, v3, old) in cells {
            let tau = r.tau(med);
            assert!((tau - 0.02).abs() < 1e-12, "{name}: tau {tau} (every cell here is under 40 ms)");
            assert_eq!(aa_ok(&e, tau), v3, "{name}: {e:?} tau {tau}");
            assert_eq!(e.est.abs() < 0.02 && e.lo <= 0.0 && 0.0 <= e.hi, old, "{name}: the v1/v2 rule");
        }
        // the motivating cell: the |HL| clause holds, the CI clause does not ...
        let m3 = est(236, 0.0188, 0.0005, 0.0368);
        assert_eq!(aa_verdict(&m3, 0.02), (false, "CI excludes 0 and leaves +-tau"));
        // ... until tau covers the CI's upper end
        assert_eq!(aa_verdict(&m3, 0.0368), (true, "CI within +-tau"));
        assert!(!aa_ok(&m3, 0.0367));
        // the relative part only binds above 40 ms (e.g. the m = 8 canonical verify, 45.7 ms)
        assert!((r.tau(45.726) - 0.022863).abs() < 1e-9);
        assert_eq!(r.tau(f64::NAN), 0.02);
    }
