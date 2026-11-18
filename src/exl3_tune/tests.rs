//! Unit tests of the tunable registry, table format, selection, statistics and plan ordering (the former inline `mod tests`; same module name).

    use super::*;

    #[test]
    fn registry_ids_unique_and_well_formed() {
        let mut seen = std::collections::HashSet::new();
        for (i, t) in REGISTRY.iter().enumerate() {
            assert_eq!(t.slot as usize, i);
            assert!(seen.insert(t.id), "duplicate tune id {}", t.id);
            assert!(!t.id.is_empty() && !t.id.contains(char::is_whitespace), "bad id {:?}", t.id);
            assert!(domain_has(t.domain, t.default));
            let _ = (t.env_parse)(); // must not panic
        }
    }

    /// T0 migration exactness: for every legacy env alias and a spread of values (unset, "0", "1",
    /// "2", "", junk), the registry value (env parse, else default) reproduces the pre-registry
    /// comparison bit for bit.
    /// Tests that set process env vars hold this lock (cargo runs tests on parallel threads).
    pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn legacy_env_semantics_are_exact() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use crate::exl3_forward as f;
        let vals: [Option<&str>; 16] = [None, Some("0"), Some("1"), Some("2"), Some(""), Some("yes"),
                                        Some("off"), Some("8"), Some("33"), Some("4"), Some(" 2 "),
                                        Some("generic"), Some("legacy"), Some("hg1"), Some("hg3"), Some("3")];
        // the old wp_env_flag(name, default_on = true)
        fn flag(v: Option<&str>) -> i32 { match v { Some("0") | Some("off") => 0, _ => 1 } }
        // (def, old semantics of the raw env value -> the registry-equivalent i32)
        type Old = fn(Option<&str>) -> i32;
        let cases: Vec<(&TunableDef, Old)> = vec![
            (&f::T_MOE_FH, |v| (v == Some("1")) as i32),
            (&f::T_HC_MIX4, |v| (v != Some("0")) as i32),
            (&f::T_WP09_OFF, |v| (v == Some("1")) as i32),
            (&f::T_WP09_PLE, |v| (v != Some("0")) as i32),
            (&f::T_WP09_ARGMAX, |v| (v != Some("0")) as i32),
            (&f::T_HC_RB, |v| (v != Some("0")) as i32),
            (&f::T_WP11_OFF, |v| (v == Some("1")) as i32),
            (&f::T_WP11_R1, |v| match v { Some("0") => 0, Some("2") => 2, _ => 1 }),
            (&f::T_WP11_R2, |v| (v != Some("0")) as i32),
            (&f::T_WP11_R3, |v| (v != Some("0")) as i32),
            (&f::T_WP11_R4, |v| (v != Some("0")) as i32),
            (&f::T_WP11_CVT, |v| (v != Some("0")) as i32), // value 0 == i2f (old: CVT == "0")
            (&f::T_WP10_FOLD, |v| (v != Some("1")) as i32),
            (&f::T_WP10_GEMV, |v| (v != Some("0")) as i32),
            (&f::T_PDL, |v| (v == Some("1")) as i32),
            (&f::T_MOE_A1B3, |v| (v != Some("0")) as i32),
            (&f::T_CHAIN_PAIR, |v| (v != Some("0")) as i32),
            (&f::T_MOE_COOP, |v| v.map(|s| s != "0").unwrap_or(false) as i32),
            (&f::T_QSA_SPLITS_DEC, |v| v.and_then(|s| s.parse::<usize>().ok())
                .filter(|n| (1..=32).contains(n)).unwrap_or(32) as i32),
            // p3c line
            (&f::T_PSK_PF, |v| v.and_then(|s| s.trim().parse::<i32>().ok()).unwrap_or(1).clamp(0, 127)),
            // psk.g: any set value hands the decision to the old --psk-g parse (marker -1)
            (&f::T_PSK_G, |v| if v.is_some() { -1 } else { 0 }),
            (&f::T_PSK_G2, |v| if v.is_some() { -1 } else { 0 }),
            (&f::T_LMH_OFF, |v| (v == Some("1")) as i32),
            (&f::T_LMH_HAD, |v| (v != Some("0")) as i32),
            (&f::T_LMH_NST, |v| match v { Some("4") => 4, Some("8") => 8, _ => 6 }),
            (&f::T_SCORE_MR, flag),
            (&f::T_SCORE_GENERIC, |v| (v == Some("generic")) as i32),
            (&f::T_SCORE_MR_CTAS, |v| v.and_then(|s| s.parse::<u32>().ok())
                .filter(|n| (1..=24).contains(n)).unwrap_or(4) as i32),
            (&f::T_TOPK_V2, flag),
            (&f::T_SEL_GATHER, |v| match v { Some("legacy") | Some("0") => 0, Some("hg1") | Some("1") => 1,
                                             Some("hg3") | Some("3") => 3, _ => -1 }),
            (&f::T_SEL_VEC, |v| (v != Some("0")) as i32),
            (&f::T_DENSE_V3, |v| (v != Some("0")) as i32),
            (&f::T_WP20_DIET, |v| (v != Some("0")) as i32),
            (&f::T_ROUTER_FUSED, |v| (v != Some("0")) as i32),
            (&f::T_PF_ROUTER_ROWS, |v| (v != Some("0")) as i32),
            (&f::T_PF_MOE_GROUPED, |v| (v != Some("0")) as i32),
            (&f::T_PF_CONV_PAR, |v| v.is_none() as i32), // the old `.is_err()` (set = serial)
            (&f::T_REPRIME_GRAPH, |v| (v != Some("0")) as i32),
            (&f::T_DRAFT_GRAPH, |v| (v != Some("0")) as i32),
            (&crate::exl3_wp27::T_WP27_PRIO, |v| (v == Some("1")) as i32),
        ];
        for (def, old) in cases {
            let id = crate::opts::lookup(def.env).unwrap_or_else(|| panic!("{}: no option {}", def.id, def.env));
            let saved = crate::opts::get(id);
            for v in vals {
                crate::opts::test_set(id, v);
                let new = (def.env_parse)().unwrap_or(def.default);
                assert_eq!(new, old(v), "{} ({}={:?})", def.id, def.env, v);
            }
            crate::opts::test_set(id, saved.as_deref());
        }
        // --wp12-off parts (the parse is cached per process; test the pure parser)
        let all = 1 | 2 | 4 | 8;
        for (v, want) in [("1", 0u8), ("all", 0), ("0", all), ("", all), ("step", all & !1), ("diet,conv", all & !6),
                          ("ab", all & !8), ("step, ab", all & !9), ("bogus", all)] {
            assert_eq!(f::wp12_mask_of(v), want, "--wp12-off={v:?}");
        }
        // WP20: two variables, the old precedence (OFF=1 -> 0, else EPI=0 -> 1, else 2)
        let two: [Option<&str>; 4] = [None, Some("0"), Some("1"), Some("x")];
        for off in two {
            for epi in two {
                for (k, v) in [(crate::opt!("wp20-off"), off), (crate::opt!("wp20-epi"), epi)] {
                    crate::opts::test_set(k, v)
                }
                let old = if off == Some("1") { 0 } else if epi == Some("0") { 1 } else { 2 };
                let new = (f::T_WP20_MODE.env_parse)().unwrap_or(f::T_WP20_MODE.default);
                assert_eq!(new, old, "--wp20-off={off:?} --wp20-epi={epi:?}");
            }
        }
        crate::opts::unset(crate::opt!("wp20-off"));
        crate::opts::unset(crate::opt!("wp20-epi"));
        // WP27: OFF=1 zeroes forks + graphs; FORKS / GRAPHS lists (ple masked); unset -> defaults
        use crate::exl3_wp27 as w;
        for off in [None, Some("1"), Some("0")] {
            for forks in [None, Some("moe"), Some("gdn,ple"), Some("ple"), Some("none"), Some("all")] {
                for graphs in [None, Some("verify"), Some("step,draft")] {
                    for (k, v) in [(crate::opt!("wp27-off"), off), (crate::opt!("wp27-forks"), forks), (crate::opt!("wp27-graphs"), graphs)] {
                        crate::opts::test_set(k, v)
                    }
                    let parse = |s: Option<&str>, names: &[(&str, i32)], all: i32| -> i32 {
                        match s { None => all, Some(s) => s.split(',').map(|t| t.trim())
                            .map(|t| if t == "all" { all } else { names.iter().find(|n| n.0 == t).map_or(0, |n| n.1) })
                            .fold(0, |a, b| a | b) }
                    };
                    let (old_f, old_g) = if off == Some("1") { (0, 0) } else {
                        (parse(forks, &[("moe", 1), ("gdn", 2), ("ple", 4)], 7) & !4,
                         parse(graphs, &[("verify", 1), ("draft", 2), ("step", 4)], 7))
                    };
                    let nf = (w::T_WP27_FORKS.env_parse)().unwrap_or(w::T_WP27_FORKS.default);
                    let ng = (w::T_WP27_GRAPHS.env_parse)().unwrap_or(w::T_WP27_GRAPHS.default);
                    assert_eq!((nf, ng), (old_f, old_g), "OFF={off:?} FORKS={forks:?} GRAPHS={graphs:?}");
                }
            }
        }
        for k in [crate::opt!("wp27-off"), crate::opt!("wp27-forks"), crate::opt!("wp27-graphs")] { crate::opts::unset(k); }
        // p5e PSK: opt-in (--psk == "1") unless the escape (--psk-off == "1"); unset -> default off
        let pv: [Option<&str>; 5] = [None, Some("1"), Some("0"), Some(""), Some("yes")];
        for on in pv {
            for esc in pv {
                for (k, v) in [(crate::opt!("psk"), on), (crate::opt!("psk-off"), esc)] {
                    crate::opts::test_set(k, v)
                }
                let old = (on == Some("1") && esc != Some("1")) as i32;
                let new = (f::T_PSK_ON.env_parse)().unwrap_or(f::T_PSK_ON.default);
                assert_eq!(new, old, "--psk={on:?} --psk-off={esc:?}");
            }
        }
        for k in [crate::opt!("psk"), crate::opt!("psk-off")] { crate::opts::unset(k); }
        // p5e prefill.wide_mt: the RESOLVED MT (dense wide chains: value 0 = 4) equals PFX1's
        // wide_mt_override().unwrap_or(4) for every alias value (trimmed; only 1|2|4|8 count)
        let saved = crate::opts::get(crate::opt!("exl3-wide-mt"));
        for v in vals.iter().copied().chain([Some(" 8 "), Some("16"), Some("-1")]) {
            match v { Some(s) => crate::opts::set(crate::opt!("exl3-wide-mt"), s), None => crate::opts::unset(crate::opt!("exl3-wide-mt")) }
            let old = match v.map(|s| s.trim().parse::<usize>()) { Some(Ok(mt @ (1 | 2 | 4 | 8))) => mt as i32, _ => 4 };
            let raw = (f::T_WIDE_MT.env_parse)().unwrap_or(f::T_WIDE_MT.default);
            let new = if matches!(raw, 1 | 2 | 4 | 8) { raw } else { 4 };
            assert_eq!(new, old, "--exl3-wide-mt={v:?}");
        }
        match saved { Some(s) => crate::opts::set(crate::opt!("exl3-wide-mt"), s), None => crate::opts::unset(crate::opt!("exl3-wide-mt")) }
    }

    #[test]
    fn wilcoxon_critical_values_match_the_table() {
        // two-sided alpha 0.05 signed-rank critical values (reject when T <= c)
        for (n, c) in [(6usize, 0usize), (7, 2), (8, 3), (9, 5), (10, 8), (12, 13), (15, 25), (20, 52)] {
            assert_eq!(wilcoxon_crit(n, 0.05), Some(c), "n = {n}");
        }
        assert_eq!(wilcoxon_crit(5, 0.05), None);
        // the normal approximation continues the exact table smoothly
        let (a, b) = (wilcoxon_crit(300, 0.05).unwrap() as f64, wilcoxon_crit(301, 0.05).unwrap() as f64);
        assert!(b > a && b - a < 400.0);
    }

    #[test]
    fn hodges_lehmann_recovers_shifts() {
        // symmetric noise around a shift: the estimate lands on the shift and the interval covers it
        let noise: Vec<f64> = (0..60).map(|i| ((i * 37 % 61) as f64 - 30.0) / 300.0).collect();
        for shift in [0.0, 0.2, -0.05] {
            let d: Vec<f64> = noise.iter().map(|x| x + shift).collect();
            let e = hl_paired(&d).unwrap();
            assert!((e.est - shift).abs() < 0.01, "est {} vs {shift}", e.est);
            assert!(e.lo <= shift && shift <= e.hi && e.lo < e.hi);
        }
        // exact constant data: a degenerate interval at the value
        let e = hl_paired(&[0.2; 12]).unwrap();
        assert_eq!((e.est, e.lo, e.hi), (0.2, 0.2, 0.2));
        assert!(hl_paired(&[]).is_none());
        // two-sample shift
        let x: Vec<f64> = noise.iter().map(|v| 40.0 + v).collect();
        let y: Vec<f64> = noise.iter().rev().map(|v| 40.2 + v).collect();
        let e2 = hl_two_sample(&x, &y).unwrap();
        assert!((e2.est - 0.2).abs() < 0.02 && e2.lo <= 0.2 && 0.2 <= e2.hi, "{e2:?}");
    }

    /// T0c (p5e): the line's new aliases, through the registry, reproduce their pre-registry reads
    /// exactly (single-variable aliases swept; the multi-variable rules as products; the cached
    /// parses through their pure parsers).
    #[test]
    fn p5e_legacy_env_semantics_are_exact() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use crate::exl3_forward as f;
        use crate::exl3_forward::dense as d;
        let vals: [Option<&str>; 20] = [None, Some("0"), Some("1"), Some("2"), Some(""), Some("yes"), Some("off"),
                                        Some("OFF"), Some("8"), Some("33"), Some("4"), Some(" 2 "), Some(" 36 "),
                                        Some("il"), Some("IL"), Some("causal"), Some("-1"), Some("96"), Some("512"),
                                        Some("99999999999")];
        type Old = fn(Option<&str>) -> i32;
        let on_unless_0: Old = |v| (v != Some("0")) as i32;
        let cases: Vec<(&TunableDef, Old)> = vec![
            (&f::T_W4HC_OFF, |v| (v == Some("1")) as i32),
            (&f::T_W4HC_FUSE, on_unless_0),
            (&f::T_W4HC_MIX, on_unless_0),
            (&f::T_W4HC_INJ, on_unless_0),
            (&f::T_W4HC_M1, on_unless_0),
            (&f::T_W4HC_G, |v| v.and_then(|s| s.parse::<u32>().ok()).filter(|&g| g >= 1).unwrap_or(96).min(i32::MAX as u32) as i32),
            (&f::T_W4HC_PF, |v| match v { Some("0") => 0, Some("2") => 2, _ => 1 }),
            (&f::T_W4HC_LA, |v| v.and_then(|s| s.parse::<i32>().ok()).unwrap_or(0).clamp(0, 7)),
            (&d::T_W4DENSE_FIX, on_unless_0),
            (&d::T_W4DENSE_SUH, on_unless_0),
            (&d::T_W4DENSE_MULTI, on_unless_0),
            (&d::T_W4DENSE_SILU, on_unless_0),
            (&d::T_W4DENSE_NST, |v| match v { Some("4") => 4, Some("8") => 8, _ => 6 }),
            (&d::T_W4DENSE_GSAT, |v| v.and_then(|s| s.trim().parse::<usize>().ok()).filter(|&x| x >= 1).unwrap_or(36)
                .min(i32::MAX as usize) as i32),
            (&f::T_W4MOE_ORDER, |v| matches!(v, Some("il") | Some("IL") | Some("1")) as i32),
            // 0 = mpk_plan's rule (the old w4moe_nst_env() None)
            (&f::T_W4MOE_NST, |v| match v { Some("4") => 4, Some("8") => 8, _ => 0 }),
            (&f::T_W4MOE_G, |v| v.and_then(|s| s.trim().parse::<u32>().ok()).filter(|&g| g > 0)
                .map_or(0, |g| g.min(i32::MAX as u32) as i32)),
            (&f::T_W4MOE_SHOVL, |v| (v == Some("1")) as i32),
            (&f::T_MOE_SHOVL, on_unless_0),
            (&f::T_MOE_ROUTER_COAL, on_unless_0),
            (&f::T_QSA_SELECT, on_unless_0),
            (&f::T_MOE_GU_FOLD, |v| v.map_or(6, |s| s.trim().parse::<i32>().unwrap_or(0).clamp(0, 16))), // default 6 since e2e7bd2
            (&f::T_PQ8_CAUSAL, |v| (v == Some("causal")) as i32),
            (&f::T_PREFIX_TAIL_CKPT, |v| match v { None => 1, Some("1") => 1, Some("2") => 2, _ => 0 }), // default 1 (owner 2026-09-28)
            (&f::T_DHEADP_OFF, |v| v.map_or(false, |s| !s.is_empty() && s != "0") as i32),
            (&f::T_DHEADP_CAP, |v| v.and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(256).min(i32::MAX as usize) as i32),
            (&f::T_SPEC_RATIO_DHEAD, |v| v.map_or(true, |s| !s.is_empty() && s != "0") as i32), // default 1 (owner 2026-09-28)
        ];
        for (def, old) in cases {
            let id = crate::opts::lookup(def.env).unwrap_or_else(|| panic!("{}: no option {}", def.id, def.env));
            let saved = crate::opts::get(id);
            for v in vals {
                crate::opts::test_set(id, v);
                let new = (def.env_parse)().unwrap_or(def.default);
                assert_eq!(new, old(v), "{} ({}={:?})", def.id, def.env, v);
            }
            crate::opts::test_set(id, saved.as_deref());
        }
        let set = |k: crate::opts::OptId, v: Option<&str>| crate::opts::test_set(k, v);
        let two: [Option<&str>; 6] = [None, Some("1"), Some("0"), Some(""), Some("off"), Some(" 1 ")];
        // W4/DENSE: opt-in (set, trimmed not "", "0", "off", "OFF") unless --w4dense-off == "1"
        for on in two {
            for esc in two {
                set(crate::opt!("w4dense"), on);
                set(crate::opt!("w4dense-off"), esc);
                let old = (on.is_some_and(|v| !matches!(v.trim(), "" | "0" | "off" | "OFF")) && esc != Some("1")) as i32;
                let new = (d::T_W4DENSE_ON.env_parse)().unwrap_or(d::T_W4DENSE_ON.default);
                assert_eq!(new, old, "--w4dense={on:?} --w4dense-off={esc:?}");
            }
        }
        for k in [crate::opt!("w4dense"), crate::opt!("w4dense-off")] { crate::opts::unset(k); }
        // W4/QSA: wp_env_flag(--w4qsa, off) && !wp_env_flag(--w4qsa-off, off) && no SEL_GATHER pin
        let flag0 = |v: Option<&str>| match v { None => false, Some("0") | Some("off") => false, Some(_) => true };
        for on in two {
            for esc in two {
                for pin in [None, Some("legacy"), Some("hg3"), Some("3"), Some("auto")] {
                    set(crate::opt!("w4qsa"), on);
                    set(crate::opt!("w4qsa-off"), esc);
                    set(crate::opt!("exl3-sel-gather"), pin);
                    let pinned = matches!(pin, Some("legacy") | Some("0") | Some("hg1") | Some("1") | Some("hg3") | Some("3"));
                    let old = (flag0(on) && !flag0(esc) && !pinned) as i32;
                    let new = (f::T_W4QSA_ON.env_parse)().unwrap_or(f::T_W4QSA_ON.default);
                    assert_eq!(new, old, "--w4qsa={on:?} --w4qsa-off={esc:?} SEL_GATHER={pin:?}");
                }
            }
        }
        for k in [crate::opt!("w4qsa"), crate::opt!("w4qsa-off"), crate::opt!("exl3-sel-gather")] { crate::opts::unset(k); }
        // PQ8: item i = !wp_env_flag(--pq8-off, off) && wp_env_flag(<item>, on)
        let flag1 = |v: Option<&str>| match v { None => true, Some("0") | Some("off") => false, Some(_) => true };
        for (def, name) in [(&f::T_PQ8_KVW, crate::opt!("pq8-kvw")), (&f::T_PQ8_FLASH, crate::opt!("pq8-flash")),
                            (&f::T_PQ8_GATHER, crate::opt!("pq8-gather"))] {
            for off in two {
                for item in [None, Some("0"), Some("1"), Some("off"), Some("causal")] {
                    set(crate::opt!("pq8-off"), off);
                    set(name, item);
                    let old = (!flag0(off) && flag1(item)) as i32;
                    let new = (def.env_parse)().unwrap_or(def.default);
                    assert_eq!(new, old, "{} --pq8-off={off:?} {name}={item:?}", def.id);
                }
            }
            crate::opts::unset(name);
        }
        crate::opts::unset(crate::opt!("pq8-off"));
        // W4/MOE parts (the alias parse is cached per process: its pure parser, against the old body)
        let (gu, dn, fold) = (f::MPK_GU, f::MPK_DN, f::MPK_FOLD);
        let all = gu | dn | fold;
        for (opt, off, want) in [(None, None, 0u8), (Some("1"), None, all), (Some("0"), None, 0), (Some(" off "), None, 0),
                                 (Some("yes"), None, all), (Some("1"), Some("1"), 0), (Some("1"), Some("all"), 0),
                                 (Some("1"), Some("gu"), dn), (Some("1"), Some("dn"), gu | fold),
                                 (Some("1"), Some("fold"), gu | dn), (Some("1"), Some("GU, fold"), dn),
                                 (Some("1"), Some("0"), all), (Some("1"), Some("bogus"), all), (None, Some("gu"), 0)] {
            assert_eq!(f::w4moe_parts_of(opt, off), want, "--w4moe={opt:?} --w4moe-off={off:?}");
            assert!(f::T_W4MOE_PARTS.domain.contains(&(want as i32)), "mask {want} outside w4moe.parts' domain");
        }
    }

    #[test]
    fn chunk_classes_are_monotone() {
        let mut last = 0u8;
        for c in [1usize, 16, 26, 32, 33, 72, 128, 160, 256, 512, 1024, 2048, 4096] {
            let k = chunk_class(c);
            assert!(k >= last);
            last = k;
        }
        assert_eq!(chunk_class(26), 0);
        assert_eq!(chunk_class(72), 2);
        assert_eq!(chunk_class(2048), 6);
    }

    #[test]
    fn scope_guards_nest_and_unwind() {
        assert_eq!(current(), Ctx::UTIL);
        {
            let _a = enter(Ctx::fam(Fam::Verify, 6).regime(true));
            assert_eq!(current().fam, Fam::Verify);
            assert_eq!(current().m, 6);
            {
                let _b = enter(Ctx::fam(Fam::Reprime, 3));
                assert_eq!(current().fam, Fam::Reprime);
            }
            assert_eq!(current().fam, Fam::Verify);
            assert_eq!(current().regime, 1);
        }
        assert_eq!(current(), Ctx::UTIL);
    }

    #[test]
    fn empty_table_parses_and_keys_expand() {
        let fp = Fingerprint { tp: 1, clock_mhz: 2400, ..Default::default() };
        let t = Table { format: TABLE_FORMAT.into(), origin: "shipped".into(), fingerprint: fp.clone(),
                        posture: Posture::default(), provenance: serde_json::Value::Null,
                        decisions: Vec::new(), families: serde_json::Value::Null, confirm: serde_json::Value::Null,
                        table_sha256: String::new() };
        let s = serde_json::to_string(&t).unwrap();
        let back: Table = serde_json::from_str(&s).unwrap();
        assert_eq!(back.fingerprint, fp);
        assert!(back.decisions.is_empty());
        // a minimal hand-written table (origin defaults to shipped, decisions optional)
        let min = format!(r#"{{"format":"{TABLE_FORMAT}","fingerprint":{},"posture":{}}}"#,
                          serde_json::to_string(&fp).unwrap(), serde_json::to_string(&Posture::default()).unwrap());
        let t2: Table = serde_json::from_str(&min).unwrap();
        assert_eq!(t2.origin, "shipped");
        let (dec, ns, nd, ign) = build_decisions(&t2, false);
        assert_eq!((ns, nd, ign.len()), (0, 0, 0));
        assert!(dec.iter().all(|d| d.is_empty()));
    }

    #[test]
    fn strict_match_names_the_field() {
        let a = Fingerprint { tune_build_id: "x".into(), tp: 1, clock_mhz: 2400, ..Default::default() };
        let mut b = a.clone();
        let p = Posture::default();
        assert!(strict_match(&a, &b, &p, &p).is_ok());
        b.tune_build_id = "y".into();
        assert_eq!(strict_match(&a, &b, &p, &p).unwrap_err(), "tune_build_id");
        let b2 = Fingerprint { tp: 2, ..a.clone() };
        assert_eq!(strict_match(&a, &b2, &p, &p).unwrap_err(), "tp");
        let p2 = Posture { lanes: 4, ..Posture::default() };
        assert_eq!(strict_match(&a, &a, &p, &p2).unwrap_err(), "posture");
        // the clock profile is NOT a strict field (nearest-profile selection happens in boot)
        let b3 = Fingerprint { clock_mhz: 2000, ..a.clone() };
        assert!(strict_match(&a, &b3, &p, &p).is_ok());
    }

    /// FIX1 #78: the PTX hashes come from the EMBEDDED modules — a binary-alone deploy (no
    /// src/ptx next to the executable, any CWD) still computes them, so the table fast path hits
    /// instead of silently falling back to the built-in defaults.
    #[test]
    fn fingerprint_ptx_hashes_come_from_the_embedded_modules() {
        let fp = current_fingerprint("/nonexistent-fix1-model-dir", 1, 0);
        assert_eq!(fp.ptx_sha256.len(), PTX_STEMS.len(), "every PTX stem hashed, CWD-independent");
        assert_eq!(fp.ptx_sha256.get("exl3_bench").map(String::as_str),
                   Some(sha256_hex(crate::ptx_embed::EXL3_BENCH.as_bytes()).as_str()));
        assert_eq!(fp.ptx_sha256.get("gpu_batch").map(String::as_str),
                   Some(sha256_hex(crate::ptx_embed::GPU_BATCH.as_bytes()).as_str()));
        let fp2 = current_fingerprint("/nonexistent-fix1-model-dir", 1, 0);
        assert_eq!(fp.ptx_sha256, fp2.ptx_sha256);
        let p = Posture::default();
        assert!(strict_match(&fp, &fp2, &p, &p).is_ok(),
                "the fast path must hit: a table written by this binary's fingerprint loads back");
    }

    #[test]
    fn ptx_build_id_parses_decimal_and_hex() {
        let ptx = ".visible .entry kernel_build_id(\n .param .u64 kernel_build_id_param_0\n)\n{\n .reg .b64 %rd<4>;\n\
                   ld.param.u64 %rd1, [kernel_build_id_param_0];\n cvta.to.global.u64 %rd2, %rd1;\n\
                   mov.u64 %rd3, 12345678901234567;\n st.global.u64 [%rd2], %rd3;\n ret;\n}\n";
        assert_eq!(ptx_build_id(ptx, 12345678901234567), Some(12345678901234567));
        assert_eq!(ptx_build_id(ptx, 7), Some(12345678901234567));
        let hex = ptx.replace("12345678901234567", "0x2BDC545D6B4B87");
        assert_eq!(ptx_build_id(&hex, 0x2BDC545D6B4B87), Some(0x2BDC545D6B4B87));
        assert_eq!(ptx_build_id("no entry here", 1), None);
        // nvcc's real spelling: the u64 stamp as a signed decimal immediate
        let neg = ptx.replace("12345678901234567", "-7963248470597812263");
        assert_eq!(ptx_build_id(&neg, 0x917c_dbb5_03d8_3fd9), Some(0x917c_dbb5_03d8_3fd9));
    }

    /// The PTX this build just wrote carries THIS binary's KERNEL_BUILD_ID (no GPU needed).
    #[test]
    fn built_ptx_stamps_match_the_binary() {
        let expect = u64::from_str_radix(env!("KERNEL_BUILD_ID"), 16).unwrap();
        for stem in PTX_STEMS {
            let Ok(ptx) = std::fs::read_to_string(format!("src/ptx/{stem}.ptx")) else { continue };
            assert_eq!(ptx_build_id(&ptx, expect), Some(expect), "src/ptx/{stem}.ptx stamp");
        }
    }

    // ---- T2 (pure CPU): statistics on synthetic data, the race, the table, fallback ----

    /// Deterministic pseudo-noise (xorshift -> symmetric triangular, mean 0, |x| <= amp).
    fn noise(seed: &mut u64, amp: f64) -> f64 {
        let mut u = || { *seed ^= *seed << 13; *seed ^= *seed >> 7; *seed ^= *seed << 17;
                         (*seed >> 11) as f64 / (1u64 << 53) as f64 };
        (u() + u() - 1.0) * amp
    }

    #[test]
    fn wilcoxon_interval_has_nominal_coverage_under_the_null() {
        // A/A: 400 experiments of n = 24 symmetric-noise pairs; the 95% interval must cover 0 in
        // about 95% of them (exact signed-rank bounds: coverage >= 95% by construction, a bit more)
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut cover = 0usize;
        for _ in 0..400 {
            let d: Vec<f64> = (0..24).map(|_| noise(&mut seed, 0.08)).collect();
            let e = hl_paired(&d).unwrap();
            if e.lo <= 0.0 && 0.0 <= e.hi { cover += 1; }
        }
        let rate = cover as f64 / 400.0;
        assert!((0.92..=0.995).contains(&rate), "A/A coverage {rate}");
    }

    #[test]
    fn hl_sees_a_known_positive_through_noise() {
        // the self-test's known positive: +0.200 ms on top of +-0.08 ms noise, n = 24
        let mut seed = 42u64;
        let d: Vec<f64> = (0..24).map(|_| 0.2 + noise(&mut seed, 0.08)).collect();
        let e = hl_paired(&d).unwrap();
        assert!((e.est - 0.2).abs() < 0.03, "{e:?}");
        assert!(e.lo > 0.1 && e.hi < 0.3, "{e:?}");
        // and the two-sample (LI) form on unpaired samples with a large common offset
        let x: Vec<f64> = (0..120).map(|_| 50.0 + noise(&mut seed, 1.5)).collect();
        let y: Vec<f64> = (0..120).map(|_| 49.0 + noise(&mut seed, 1.5)).collect();
        let e2 = hl_two_sample(&x, &y).unwrap();
        assert!((e2.est + 1.0).abs() < 0.3 && e2.hi < 0.0, "{e2:?}");
    }

    #[test]
    fn verdict_prefers_the_default() {
        let tau = tau_ms(48.0); // 0.048 ms
        assert!((tau - 0.048).abs() < 1e-12);
        assert_eq!(tau_ms(6.0), 0.02);
        let e = |est, lo, hi| Est { n: 30, est, lo, hi };
        assert_eq!(verdict(&e(-0.2, -0.3, -0.1), tau), Verdict::Adopt);
        // significant but inside tau: the default stays (hysteresis)
        assert_eq!(verdict(&e(-0.04, -0.045, -0.03), tau), Verdict::Default);
        assert_eq!(verdict(&e(0.0, -0.1, 0.1), tau), Verdict::Default);
        assert_eq!(verdict(&e(0.1, 0.05, 0.2), tau), Verdict::Slower);
    }

    /// A synthetic sampler: candidate i has true shift `shift[i]` ms, noise +-amp; `out` marks a
    /// candidate that fails its digest on the first call.
    fn synth(shift: Vec<f64>, amp: f64, out: Option<usize>) -> impl FnMut(&[usize], usize, bool) -> Vec<Sampled> {
        let mut seed = 7u64;
        move |active: &[usize], want: usize, _confirm: bool| {
            active.iter().map(|&i| {
                if Some(i) == out { return Sampled::Out("XCHECK FAIL (synthetic)".into()); }
                Sampled::Obs((0..want).map(|j| ((3 + (j % 4)) as u16, shift[i] + noise(&mut seed, amp))).collect())
            }).collect()
        }
    }

    #[test]
    fn race_adopts_a_real_winner_and_confirms_it() {
        let cfg = RaceCfg::default();
        let tau = 0.05;
        // candidates: slower, neutral, clearly faster (-0.3), slightly faster (-0.1)
        let mut s = synth(vec![0.4, 0.0, -0.3, -0.1], 0.08, None);
        let r = race(4, &cfg, tau, &mut s);
        assert_eq!(r.winner, Some(2), "{r:#?}");
        assert_eq!(r.verdict, "ADOPTED");
        assert!(r.confirm.unwrap().hi < 0.0);
        assert!(r.cands[0].fate.starts_with("SLOWER") || r.cands[0].fate.starts_with("PRUNED"), "{}", r.cands[0].fate);
        assert!(r.calls >= 2);
    }

    #[test]
    fn race_keeps_the_default_on_noise_and_inside_tau() {
        let cfg = RaceCfg::default();
        // pure noise: nothing may be adopted
        let mut s = synth(vec![0.0, 0.0, 0.0], 0.1, None);
        let r = race(3, &cfg, 0.05, &mut s);
        assert_eq!(r.winner, None, "{r:#?}");
        assert!(r.verdict.starts_with("DEFAULT"));
        // a real but sub-threshold gain (-0.02 ms against tau 0.05): the default stays
        let mut s2 = synth(vec![-0.02], 0.01, None);
        let r2 = race(1, &cfg, 0.05, &mut s2);
        assert_eq!(r2.winner, None, "{r2:#?}");
        // no candidates at all
        let mut s3 = synth(vec![], 0.1, None);
        assert_eq!(race(0, &cfg, 0.05, &mut s3).verdict, "DEFAULT (no candidates)");
    }

    /// A sampler that stops delivering (memory cap, a new unit key it cannot capture) must not
    /// livelock stage 2: the candidate leaves the race.
    #[test]
    fn race_terminates_when_a_sampler_underdelivers() {
        let cfg = RaceCfg::default();
        let mut calls = 0usize;
        let mut seed = 11u64;
        let mut s = |active: &[usize], want: usize, _c: bool| -> Vec<Sampled> {
            calls += 1;
            active.iter().map(|_| if calls == 1 {
                // stage 1 delivers a wide, undecided sample; later calls deliver nothing
                Sampled::Obs((0..want).map(|_| (3u16, noise(&mut seed, 1.0))).collect())
            } else { Sampled::Obs(Vec::new()) }).collect()
        };
        let r = race(2, &cfg, 0.01, &mut s);
        assert!(r.calls < 10, "{}", r.calls);
        assert_eq!(r.winner, None);
    }

    #[test]
    fn table_assign_matches_boot() {
        let mut t = table_fixture("user", 2400);
        t.decisions.push(Decision { id: "hc.bxr".into(), rev: 1, class: "S".into(),
                                    scope: DecScope { fam: Some("DraftChain".into()), m: Some(vec![1]), ..Default::default() },
                                    value: 48, default: Some(40), evidence: serde_json::Value::Null });
        let (a, ign) = table_assign(&t, false);
        // the am_nb decision (2 widths) is in force; the invalid hc.bxr at m=1 is not
        assert_eq!(a.len(), 2, "{ign:?}");
        assert!(a.iter().all(|(d, k, v)| d.id == "wp09.am_nb" && *v == 32 && k.is_some()));
        assert!(!key_valid(&crate::exl3_forward::T_HC_BXR, &Key::W(Fam::DraftChain as u8, 1, 0), 48));
        assert!(key_valid(&crate::exl3_forward::T_HC_BXR, &Key::W(Fam::Verify as u8, 6, 0), 48));
    }

    #[test]
    fn race_disqualifies_a_digest_failure_even_if_fast() {
        let cfg = RaceCfg::default();
        let mut s = synth(vec![-1.0, -0.3], 0.05, Some(0));
        let r = race(2, &cfg, 0.05, &mut s);
        assert_eq!(r.winner, Some(1));
        assert!(r.cands[0].fate.contains("XCHECK"), "{}", r.cands[0].fate);
    }

    #[test]
    fn width_split_only_reverts_a_clearly_slower_width() {
        let mut seed = 3u64;
        let mut obs: Vec<Obs> = Vec::new();
        for _ in 0..12 {
            obs.push((3, -0.2 + noise(&mut seed, 0.02)));
            obs.push((6, -0.2 + noise(&mut seed, 0.02)));
            obs.push((8, 0.3 + noise(&mut seed, 0.02)));
        }
        let sp = width_split(&obs);
        let keep: Vec<(u16, bool)> = sp.iter().map(|x| (x.0, x.2)).collect();
        assert_eq!(keep, vec![(3, true), (6, true), (8, false)]);
    }

    #[test]
    fn plan_orders_by_value_density_and_respects_the_budget() {
        let mut v = vec![
            PlanItem { name: "a".into(), ev_ms: 0.1, est_s: 60.0 },
            PlanItem { name: "b".into(), ev_ms: 1.0, est_s: 60.0 },
            PlanItem { name: "c".into(), ev_ms: 0.5, est_s: 10.0 },
        ];
        order_by_ev(&mut v);
        assert_eq!(v.iter().map(|x| x.name.as_str()).collect::<Vec<_>>(), vec!["c", "b", "a"]);
        assert_eq!(plan_fits(&v, 0.0, 75.0), 2);
        assert_eq!(plan_fits(&v, 70.0, 75.0), 0);
        assert_eq!(plan_fits(&v, 0.0, 1e9), 3);
    }

    fn fp_fixture() -> Fingerprint {
        let mut ptx = BTreeMap::new();
        ptx.insert("exl3_bench".to_string(), "aa".repeat(32));
        Fingerprint {
            tune_build_id: "0123456789abcdef".into(), ptx_sha256: ptx, fatbin_sha256: BTreeMap::new(),
            registry_schema: registry_schema(),
            model: ModelFp { config_sha256: "c0".repeat(32), quant_sha256: "q0".repeat(32) },
            gpu: GpuFp { name: "NVIDIA GB10".into(), cc: "12.1".into(), sms: 48, l2_bytes: 25165824 },
            driver: "580.173.02".into(), cuda: "13.0".into(), tp: 1, clock_mhz: 2400,
        }
    }

    fn table_fixture(origin: &str, clock: u32) -> Table {
        let posture = Posture { lanes: 8, mtp_max_k: 7, dds: true, kv_fmt: 3, max_pos: 4097, prefill_chunk: 2048 };
        Table {
            format: TABLE_FORMAT.into(), origin: origin.into(),
            fingerprint: Fingerprint { clock_mhz: clock, ..fp_fixture() }, posture,
            provenance: serde_json::json!({"commit": "x", "box": "test"}),
            decisions: vec![Decision {
                id: "wp09.am_nb".into(), rev: 1, class: "S".into(),
                scope: DecScope { fam: Some("Verify".into()), m: Some(vec![3, 4]), regime: Some("dense".into()), ..Default::default() },
                value: 32, default: Some(0),
                evidence: serde_json::json!({"unit": "verify_dry", "n": 30, "hl_delta_ms": -0.05, "ci95_ms": [-0.07, -0.03]}),
            }],
            families: serde_json::json!({"wp09.am_nb": {"status": "tuned"}}),
            confirm: serde_json::json!({"li_round_ms": {}, "pass": true}),
            table_sha256: String::new(),
        }
    }

    #[test]
    fn table_round_trips_sealed_and_detects_edits() {
        let dir = std::env::temp_dir().join(format!("gb10_tune_test_{}", std::process::id()));
        let mut t = table_fixture("user", 2400);
        let path = dir.join(table_file_name(&t).trim_start_matches("tune/"));
        let fsha = write_table(&mut t, path.to_str().unwrap()).unwrap();
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(sha256_hex(&raw), fsha);
        let back: Table = serde_json::from_slice(&raw).unwrap();
        assert!(back.seal_ok() && !back.table_sha256.is_empty());
        assert_eq!(back.decisions.len(), 1);
        assert_eq!(back.fingerprint, t.fingerprint);
        assert_eq!(back.posture, t.posture);
        // the decision validates against the registry and expands to its width keys
        let (dec, ns, nd, ign) = build_decisions(&back, false);
        assert_eq!((ns, nd), (1, 0), "{ign:?}");
        let slot = crate::exl3_forward::T_WP09_AM_NB.slot as usize;
        assert_eq!(dec[slot].get(&Key::W(Fam::Verify as u8, 3, 0)), Some(&32));
        assert_eq!(dec[slot].get(&Key::W(Fam::Verify as u8, 4, 0)), Some(&32));
        assert_eq!(dec[slot].get(&Key::W(Fam::Verify as u8, 5, 0)), None);
        // a hand edit after sealing is refused
        let edited = String::from_utf8(raw.clone()).unwrap().replace("\"value\": 32", "\"value\": 16");
        assert_ne!(edited.as_bytes(), &raw[..]);
        let tt: Table = serde_json::from_str(&edited).unwrap();
        assert!(!tt.seal_ok());
        // the report names the decision and the seal
        let rep = report(&back, &raw);
        assert!(rep.contains("wp09.am_nb") && rep.contains(" OK"), "{rep}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fingerprint_mismatch_falls_back_with_the_field_named() {
        let t = table_fixture("shipped", 2400);
        let raw = serde_json::to_vec(&t).unwrap();
        let have = fp_fixture();
        let file = |b: Vec<u8>| vec![(PathBuf::from("tune/x.json"), b)];
        // exact match -> picked
        assert!(select_table(file(raw.clone()), &have, &t.posture, 2400).is_ok());
        // a new build (tune_build_id) -> FALLBACK naming the field
        let have2 = Fingerprint { tune_build_id: "fedcba9876543210".into(), ..have.clone() };
        let e = select_table(file(raw.clone()), &have2, &t.posture, 2400).err().unwrap();
        assert!(e.starts_with("tune_build_id"), "{e}");
        // a different kernel PTX
        let mut have3 = have.clone();
        have3.ptx_sha256.insert("exl3_bench".into(), "bb".repeat(32));
        assert!(select_table(file(raw.clone()), &have3, &t.posture, 2400).err().unwrap().starts_with("ptx_sha256"));
        // another TP degree, another posture
        let have4 = Fingerprint { tp: 2, ..have.clone() };
        assert!(select_table(file(raw.clone()), &have4, &t.posture, 2400).err().unwrap().starts_with("tp"));
        let p2 = Posture { max_pos: 262145, ..t.posture.clone() };
        assert!(select_table(file(raw.clone()), &have, &p2, 2400).err().unwrap().starts_with("posture"));
        // a corrupt file and an empty directory
        assert!(select_table(file(b"{not json".to_vec()), &have, &t.posture, 2400).err().unwrap().starts_with("parse error"));
        assert_eq!(select_table(Vec::new(), &have, &t.posture, 2400).err().unwrap(), "no table in ./tune");
        // an unconfirmed table (gate not passed / hand-written): never auto-loaded, loads when named
        let mut unconf = t.clone();
        unconf.confirm = serde_json::json!({"pass": false});
        let e = select_table(file(serde_json::to_vec(&unconf).unwrap()), &have, &t.posture, 2400).err().unwrap();
        assert!(e.starts_with("unconfirmed"), "{e}");
        assert!(select_table_x(file(serde_json::to_vec(&unconf).unwrap()), &have, &t.posture, 2400, true).is_ok());
        // a sealed-then-edited table
        let mut sealed = t.clone();
        sealed.seal();
        sealed.decisions[0].value = 16;
        let e = select_table(file(serde_json::to_vec(&sealed).unwrap()), &have, &t.posture, 2400).err().unwrap();
        assert!(e.starts_with("table_sha256"), "{e}");
        // the clock is not strict: a user table wins over a shipped one; among equals the nearest
        // clock profile wins
        let u = table_fixture("user", 2000);
        let files = vec![(PathBuf::from("tune/a.json"), raw.clone()),
                         (PathBuf::from("tune/b.json"), serde_json::to_vec(&u).unwrap())];
        let pick = select_table(files, &have, &t.posture, 2300).unwrap();
        assert_eq!(pick.table.origin, "user");
        let s2 = table_fixture("shipped", 2000);
        let files = vec![(PathBuf::from("tune/a.json"), raw.clone()),
                         (PathBuf::from("tune/b.json"), serde_json::to_vec(&s2).unwrap())];
        assert_eq!(select_table(files, &have, &t.posture, 2100).unwrap().table.fingerprint.clock_mhz, 2000);
    }

    #[test]
    fn zero_decision_tables_load_failed_confirms_do_not() {
        let have = fp_fixture();
        let file = |b: Vec<u8>| vec![(PathBuf::from("tune/x.json"), b)];
        let sealed = |mut t: Table| { t.seal(); serde_json::to_vec_pretty(&t).unwrap() };
        let mut zero = table_fixture("user", 2400);
        zero.decisions.clear();
        // L10 writer: a clean 0-decision run writes pass = true with its reason -> auto loads it
        zero.confirm = serde_json::json!({"skipped": "no decision adopted", "pass": true, "known_answer": "holds",
            "reason": "0 decisions: the table is the built-in defaults", "hashes_identical": true, "ignored": []});
        let p = zero.posture.clone();
        assert!(select_table(file(sealed(zero.clone())), &have, &p, 2400).is_ok());
        // pre-L10 0-decision table (no pass field, sealed as written then): still validates + loads
        let mut legacy = zero.clone();
        legacy.confirm = serde_json::json!({"skipped": "no decision adopted"});
        assert!(select_table(file(sealed(legacy.clone())), &have, &p, 2400).is_ok());
        // ...but the legacy shape never excuses a table that carries decisions
        let mut legacy_dec = table_fixture("user", 2400);
        legacy_dec.confirm = legacy.confirm.clone();
        let e = select_table(file(sealed(legacy_dec)), &have, &p, 2400).err().unwrap();
        assert!(e.starts_with("unconfirmed"), "{e}");
        // a 0-decision run that did NOT pass (ignored decisions / deviating known answer), a
        // rejected gate and an ungated table: all refused by auto
        let mut failed0 = zero.clone();
        failed0.confirm = serde_json::json!({"skipped": "no decision adopted", "pass": false, "known_answer": "deviates"});
        let mut rejected = table_fixture("user", 2400);
        rejected.confirm = serde_json::json!({"li_round_ms": {}, "beats_defaults": false, "pass": false});
        let mut ungated = zero.clone();
        ungated.confirm = serde_json::json!({"skipped": "--autotune-skip-gate", "pass": false});
        for t in [failed0, rejected, ungated] {
            let e = select_table(file(sealed(t)), &have, &p, 2400).err().unwrap();
            assert!(e.starts_with("unconfirmed"), "{e}");
        }
        // tampering: flipping a failed confirm to pass after sealing is refused by the seal
        let mut tamper = zero.clone();
        tamper.confirm = serde_json::json!({"skipped": "no decision adopted", "pass": false});
        let raw = String::from_utf8(sealed(tamper)).unwrap().replace("\"pass\": false", "\"pass\": true");
        let e = select_table(file(raw.into_bytes()), &have, &p, 2400).err().unwrap();
        assert!(e.starts_with("table_sha256"), "{e}");
        // ...and so is stripping the pass field to fake the legacy shape
        let raw = String::from_utf8(sealed(zero.clone())).unwrap();
        let t2: Table = serde_json::from_str(&raw).unwrap();
        let mut stripped = t2.clone();
        stripped.confirm = serde_json::json!({"skipped": "no decision adopted"});
        let e = select_table(file(serde_json::to_vec_pretty(&stripped).unwrap()), &have, &p, 2400).err().unwrap();
        assert!(e.starts_with("table_sha256"), "{e}");
    }

    #[test]
    fn unproven_and_non_s_decisions_are_ignored_at_boot() {
        let mut t = table_fixture("user", 2400);
        t.decisions.push(Decision { id: "pdl".into(), rev: 1, class: "S".into(),
                                    scope: DecScope { fam: Some("Verify".into()), ..Default::default() },
                                    value: 1, default: Some(0), evidence: serde_json::Value::Null });
        t.decisions.push(Decision { id: "qsa.splits_dec".into(), rev: 1, class: "N".into(),
                                    scope: DecScope::default(), value: 16, default: Some(32),
                                    evidence: serde_json::Value::Null });
        t.decisions.push(Decision { id: "hc.bxr".into(), rev: 1, class: "S".into(),
                                    scope: DecScope { fam: Some("Verify".into()), m: Some(vec![1]), ..Default::default() },
                                    value: 48, default: Some(40), evidence: serde_json::Value::Null });
        let (_, ns, _, ign) = build_decisions(&t, false);
        assert_eq!(ns, 1, "{ign:?}");
        assert!(ign.iter().any(|x| x.contains("pdl") && x.contains("UNPROVEN")), "{ign:?}");
        assert!(ign.iter().any(|x| x.contains("qsa.splits_dec") && x.contains("section N")), "{ign:?}");
        assert!(ign.iter().any(|x| x.contains("hc.bxr") && x.contains("valid")), "{ign:?}");
    }

    // ---- §5.2 v2 (pure CPU): the balanced order and the position model ----

    /// Over one period (from ANY start row) every row is a permutation, every arm sits at every
    /// position exactly period/n times, and every ordered pair of distinct arms is adjacent exactly
    /// period/n times (first-order carry-over balance) — for every arm count the tuner can run.
    #[test]
    fn balanced_order_covers_every_position_equally() {
        for n in 1..=16usize {
            let p = williams_period(n);
            assert_eq!(p % n, 0);
            for start in [0usize, 1, 5, 17] {
                let mut at = vec![vec![0usize; n]; n]; // at[arm][pos]
                let mut adj = vec![vec![0usize; n]; n]; // adj[x][y]: y right after x
                for r in start..start + p {
                    let row = williams_row(n, r);
                    let mut s = row.clone();
                    s.sort_unstable();
                    assert_eq!(s, (0..n).collect::<Vec<_>>(), "n {n} row {r} {row:?} is not a permutation");
                    for (pos, &a) in row.iter().enumerate() { at[a][pos] += 1; }
                    for w in row.windows(2) { adj[w[0]][w[1]] += 1; }
                }
                for a in 0..n {
                    for pos in 0..n { assert_eq!(at[a][pos], p / n, "n {n} start {start}: arm {a} at position {pos}"); }
                    for b in 0..n {
                        if a != b { assert_eq!(adj[a][b], p / n, "n {n} start {start}: {a} -> {b} adjacency"); }
                    }
                }
            }
        }
    }

    /// Cursors are per (cell, n) and persist: interleaved cells each walk their own design; a cell
    /// revisited after another arm count continues where it stopped (never restarts at a fixed row).
    #[test]
    fn balanced_order_cursors_are_per_cell_and_persist() {
        let mut o = BalancedOrder::new(0xABCD);
        let n = 3;
        let p = williams_period(n);
        // two cells interleaved round by round (the verify widths rotate per round): each cell must
        // be exactly balanced on its own after p visits
        let mut first = [[0usize; 3]; 2];
        for _ in 0..p {
            for (ci, cell) in [7u64, 9].iter().enumerate() {
                first[ci][o.next(*cell, n)[0]] += 1;
            }
        }
        for ci in 0..2 { assert_eq!(first[ci], [p / n; 3], "cell {ci}"); }
        // short phases (1 visit each, n = 2, with n = 3 phases in between) still alternate
        let mut firsts = Vec::new();
        for _ in 0..6 {
            firsts.push(o.next(11, 2)[0]);
            let _ = o.next(11, 3);
        }
        assert_eq!(firsts.iter().filter(|&&a| a == 0).count(), 3, "{firsts:?}");
        // deterministic in the seed
        let mut a = BalancedOrder::new(5);
        let mut b = BalancedOrder::new(5);
        for _ in 0..10 { assert_eq!(a.next(3, 4), b.next(3, 4)); }
    }

    /// The position model recovers a known position effect under round (content) and arm effects,
    /// and invents none when there is none — or when the design cannot identify it.
    #[test]
    fn position_offsets_recover_and_never_invent() {
        let mut seed = 99u64;
        let n = 3usize;
        let truth = [0.6, 0.1, -0.7];
        let arms = [0.0, 0.0, 0.2];
        let mut with = Vec::new();
        let mut without = Vec::new();
        let mut confounded = Vec::new();
        for r in 0..120u64 {
            let content = 30.0 + 10.0 * (noise(&mut seed, 0.5) + 0.5);
            let row = williams_row(n, r as usize);
            for (pos, &a) in row.iter().enumerate() {
                let e = noise(&mut seed, 0.05);
                with.push(PosObs { round: r, arm: a as u64, pos, ms: content + arms[a] + truth[pos] + e });
                without.push(PosObs { round: r, arm: a as u64, pos, ms: content + arms[a] + e });
                // confounded: arm a always at position a (the design cannot separate them): the
                // combined effect must stay with the arm, not be split into f
                confounded.push(PosObs { round: r, arm: a as u64, pos: a, ms: content + arms[a] + truth[a] + e });
            }
        }
        let f = position_offsets(&with, n);
        for p in 0..n { assert!((f[p] - truth[p]).abs() < 0.03, "{f:?}"); }
        let f0 = position_offsets(&without, n);
        assert!(f0.iter().all(|x| x.abs() < 0.03), "{f0:?}");
        let fc = position_offsets(&confounded, n);
        assert!(fc.iter().all(|x| x.abs() < 0.03), "confounded design must not move the arm effect into f: {fc:?}");
        // uninformative inputs: no adjustment
        assert_eq!(position_offsets(&with[..3], n), vec![0.0; n]);
        assert_eq!(position_offsets(&with, 1), vec![0.0]);
    }
