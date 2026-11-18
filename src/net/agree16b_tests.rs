    use super::*;

    /// W5: the only narrowing of a width on its way into a lockstep word sees the whole usize.
    #[test]
    fn a_width_is_range_checked_before_it_is_narrowed() {
        for k in 0..=AGREE_WIDTH_MAX { assert_eq!(agree_width(k), Some(k as u8)); }
        for k in AGREE_WIDTH_MAX + 1..=1024 { assert_eq!(agree_width(k), None, "width {k}"); }
        // the wrap an `as u8` before the check would have produced: 256 + k arriving as k
        for k in 0..=AGREE_WIDTH_MAX {
            for wraps in [256usize, 512, 65536, 1 << 32] {
                assert_eq!((wraps + k) as u8 as usize, k, "the cast wraps {} to {k}", wraps + k);
                assert_eq!(agree_width(wraps + k), None, "{} must be refused, not read as {k}", wraps + k);
            }
        }
        assert_eq!(agree_width(usize::MAX), None);
    }

    /// W5, the call sites (they need a live link, so they are pinned in the source): the NVFP4 step agree
    /// and the DFlash agree narrow through `agree_width`, and no `agree_ext(` call takes an `as u8`.
    #[test]
    fn the_nvfp4_and_dflash_agree_sites_narrow_through_agree_width() {
        let code = |s: &str| s.lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("\n");
        let body = |src: &str, from: &str, to: &str| -> String {
            let a = src.find(from).unwrap_or_else(|| panic!("{from} not found"));
            let b = src[a..].find(to).unwrap_or_else(|| panic!("{to} not found after {from}")) + a;
            code(&src[a..b])
        };
        let batch = src_text!("batch");
        let step = body(batch, "    fn tp_agree_step(&self, step: u64)", "    /// TP serving head loop");
        assert!(step.contains("crate::net::agree_width(k_real)"), "tp_agree_step narrows through agree_width");
        assert!(!step.contains("depth() as u8"), "tp_agree_step casts the depth before the range check");
        let gpu = src_text!("gpu");
        let df = body(gpu, "    pub fn tp_generate_dflash(", "\n    }\n");
        assert!(df.contains("crate::net::agree_width(dblock)"), "tp_generate_dflash narrows through agree_width");
        assert!(!df.contains("dblock as u8"), "tp_generate_dflash casts the block before the range check");
        let call = ["agree", "_ext("].concat();
        for (name, src) in [("batch.rs", batch), ("gpu.rs", gpu), ("exl3_serve.rs", src_text!("exl3_serve")),
                            ("exl3_forward/xtp.rs", src_text!("exl3_forward/xtp"))] {
            for (i, l) in code(src).lines().enumerate() {
                assert!(!(l.contains(&call) && l.contains(" as u8")), "src/{name}:{}: a cast inside an agree call: {}", i + 1, l.trim());
            }
        }
    }

    /// W3: what each drill flag does where. Every set flag yields exactly one line: ARMED or INERT.
    #[test]
    fn the_drill_plan_arms_or_says_why_not() {
        assert_eq!(drill_plan(None, None, true, Some((0, 2))), DrillPlan::default());
        assert_eq!(drill_plan(None, None, false, None), DrillPlan::default());
        // EXL3 width drill, armed: the head says N - 1, rank r says N - 1 - r
        for (rank, said) in [(0, 15), (1, 14), (3, 12)] {
            let p = drill_plan(None, Some("16"), true, Some((rank, 4)));
            assert_eq!((p.width, p.hash_step, p.lines.len()), (Some(16), None, 1), "rank {rank}");
            assert!(p.lines[0].starts_with(&format!("[tp-agree] DRILL ARMED (--tp-width-drill 16, rank {rank} of TP=4)")), "{}", p.lines[0]);
            assert!(p.lines[0].contains(&format!("REPORTED as width {said} ")) && p.lines[0].contains("once"), "{}", p.lines[0]);
            assert!(p.lines[0].contains("not a serving boot"));
        }
        // inert, each with its reason
        let inert = |p: DrillPlan, why: &str| {
            assert_eq!((p.width, p.hash_step, p.lines.len()), (None, None, 1), "{p:?}");
            assert!(p.lines[0].contains("INERT") && p.lines[0].contains(why), "{} (wanted: {why})", p.lines[0]);
        };
        inert(drill_plan(None, Some("16"), true, None), "at TP=1");
        inert(drill_plan(None, Some("16"), false, Some((0, 2))), "on an NVFP4 pack");
        inert(drill_plan(None, Some("1"), true, Some((1, 2))), "N > 1");
        inert(drill_plan(None, Some("3"), true, Some((3, 4))), "N > 3");
        for bad in ["0", "-1", "32", "sixteen", ""] { inert(drill_plan(None, Some(bad), true, Some((0, 2))), "not an integer from 1 to 31"); }
        assert_eq!(drill_plan(None, Some("31"), true, Some((0, 2))).width, Some(31));
        assert_eq!(drill_plan(None, Some("2"), true, Some((1, 2))).width, Some(2));
        // NVFP4 hash drill
        let p = drill_plan(Some("5"), None, false, Some((1, 2)));
        assert_eq!((p.width, p.hash_step, p.lines.len()), (None, Some(5), 1));
        assert!(p.lines[0].starts_with("[tp-agree] DRILL ARMED (--tp-agree-drill 5, rank 1 of TP=2, NVFP4 pack)"), "{}", p.lines[0]);
        assert_eq!(drill_plan(Some("0"), None, false, Some((0, 2))).hash_step, Some(0), "step 0 is a step");
        inert(drill_plan(Some("5"), None, true, Some((0, 2))), "on an EXL3 pack");
        inert(drill_plan(Some("5"), None, false, None), "at TP=1");
        inert(drill_plan(Some("-5"), None, false, Some((0, 2))), "not a step number");
        // both flags on one command line: one line each, each judged by itself
        let p = drill_plan(Some("7"), Some("16"), true, Some((0, 2)));
        assert_eq!((p.width, p.hash_step, p.lines.len()), (Some(16), None, 2));
        assert!(p.lines[0].contains("DRILL ARMED") && p.lines[1].contains("INERT on an EXL3 pack"));
    }

    /// W3: the NVFP4 drill's corruption differs on every rank, so two armed ranks cannot agree on it.
    #[test]
    fn the_hash_drill_corrupts_every_rank_differently() {
        let flips: Vec<u32> = (0..16).map(drill_hash_flip).collect();
        for (i, a) in flips.iter().enumerate() {
            assert_ne!(*a, 0, "rank {i}: a zero flip corrupts nothing");
            for (j, b) in flips.iter().enumerate() { if i != j { assert_ne!(a, b, "ranks {i} and {j} corrupt alike"); } }
        }
        // through the word both ranks compare: same digest, same width, both armed -> different words
        for h in [0u32, 0x811c_9dc5, 0xFFFF_FFFF, 0x99915f3e] {
            for k in [0u8, 2, 7] {
                assert_ne!(agree_word(k, h ^ drill_hash_flip(0)), agree_word(k, h ^ drill_hash_flip(1)), "digest {h:08x} width {k}");
                assert_ne!(agree_word(k, h ^ drill_hash_flip(0)), agree_word(k, h), "an armed head against an honest node");
                assert_ne!(agree_word(k, h ^ drill_hash_flip(1)), agree_word(k, h), "an armed node against an honest head");
            }
        }
    }
