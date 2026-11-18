//! Unit tests of `lane_pack` (the former inline `mod tests`; same module name). See `mod.rs` for the code under test.

    use super::*;

    fn lv(slot: usize, block: Option<Skip>, regime: u64, clamped: bool) -> LaneView {
        LaneView { slot, block, regime, clamped }
    }
    fn ok(slot: usize) -> LaneView { lv(slot, None, 0, false) }

    #[test]
    fn flag_parse() {
        for v in [None, Some(""), Some("off"), Some(" off ")] {
            assert_eq!(LanePack::parse(v).unwrap(), LanePack::Off, "{v:?}");
        }
        for v in [Some("on"), Some("2"), Some(" 2 ")] {
            assert_eq!(LanePack::parse(v).unwrap(), LanePack::Pairs, "{v:?}");
        }
        let e = LanePack::parse(Some("3")).unwrap_err().to_string();
        assert!(e.contains("not built") && e.contains("use on or 2"), "{e}");
        for bad in ["1", "0", "4", "auto", "yes", "true"] {
            let e = LanePack::parse(Some(bad)).unwrap_err().to_string();
            assert!(e.starts_with("--lane-pack must be off, on or 2"), "{bad}: {e}");
        }
        assert!(!LanePack::Off.on() && LanePack::Pairs.on());
    }

    #[test]
    fn tp_is_refused_only_above_four_boxes_and_only_when_on() {
        for w in [1u32, 2, 4, 8] { assert!(tp_refusal(LanePack::Off, w).is_ok(), "the default must boot at TP={w}"); }
        // owner 2026-10-10: TP=4 is no longer refused (unverified on hardware until its gates run)
        for w in [1u32, 2, 3, 4] { assert!(tp_refusal(LanePack::Pairs, w).is_ok(), "--lane-pack on boots at TP={w}"); }
        for w in [5u32, 8] {
            let e = tp_refusal(LanePack::Pairs, w).unwrap_err().to_string();
            assert!(e.starts_with("--lane-pack runs at TP=1, TP=2 and TP=4") && e.contains(&format!("TP={w} is")), "{e}");
        }
    }

    /// OQ1 (design 9): the combined hash is exactly as strict as a per-request vector. Every
    /// single-field change of a two-lane pack, and every re-split of the same words, moves the word
    /// list; the total row count is explicit.
    #[test]
    fn the_pre_verify_words_tell_every_pack_apart() {
        let (d0, d1) = ([11, 12, 13], [21, 22]);
        let base = [PreSeg::greedy(1, 100, 7, &d0), PreSeg::greedy(2, 200, 8, &d1)];
        let w = pre_words(&base);
        assert_eq!(w, vec![4, 1, 100, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13,
                           3, 2, 200, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 22]);
        assert_eq!(pack_rows_total(&base), 7);
        let mut seen = std::collections::HashSet::new();
        seen.insert(w.clone());
        let variants: Vec<Vec<i32>> = vec![
            pre_words(&[PreSeg { slot: 3, ..base[0] }, base[1]]),                       // another slot
            pre_words(&[PreSeg { pos: 101, ..base[0] }, base[1]]),                      // a shifted position
            pre_words(&[PreSeg { bonus: 9, ..base[0] }, base[1]]),                      // another bonus token
            pre_words(&[base[0], PreSeg { drafts: &[21, 23], ..base[1] }]),             // one draft differs
            pre_words(&[base[1], base[0]]),                                            // the pair swapped
            pre_words(&[PreSeg { drafts: &[11, 12], ..base[0] }, PreSeg { drafts: &[13, 21, 22], ..base[1] }]), // re-split
            pre_words(&[base[0]]),                                                     // a lone segment
            // the sampler of a lane: kind, then every word of its row-0 block (seed, counter, temperature ...)
            pre_words(&[PreSeg { kind: 1, ..base[0] }, base[1]]),
            pre_words(&[PreSeg { kind: 2, ..base[0] }, base[1]]),
            pre_words(&[PreSeg { samp: [1, 0x3f80_0000, 0x3f73_3333, 20, 5, 0, 9, 0], kind: 2, ..base[0] }, base[1]]),
            pre_words(&[PreSeg { samp: [1, 0x3f80_0000, 0x3f73_3333, 20, 5, 0, 10, 0], kind: 2, ..base[0] }, base[1]]), // counter + 1
            pre_words(&[PreSeg { samp: [1, 0x3f80_0000, 0x3f73_3333, 20, 6, 0, 9, 0], kind: 2, ..base[0] }, base[1]]),  // another seed
            // per-lane pass accounting: two ranks whose launch counts differ in OPPOSITE directions have equal sums
            pre_words(&[PreSeg { launched: 3, ..base[0] }, PreSeg { launched: 4, ..base[1] }]),
            pre_words(&[PreSeg { launched: 4, ..base[0] }, PreSeg { launched: 3, ..base[1] }]),
            pre_words(&[PreSeg { launched: 3, wasted: 1, ..base[0] }, PreSeg { launched: 4, ..base[1] }]),
        ];
        for v in variants { assert!(seen.insert(v.clone()), "a different pack produced the same words: {v:?}"); }
        // same total rows, different split: the explicit per-segment row count tells them apart
        let a = pre_words(&[PreSeg::greedy(0, 0, 0, &[1, 2, 3]), PreSeg::greedy(1, 0, 0, &[4])]);
        let b = pre_words(&[PreSeg::greedy(0, 0, 0, &[1, 2]), PreSeg::greedy(1, 0, 0, &[3, 4])]);
        assert_ne!(a, b);
        assert_eq!(pack_rows_total(&[PreSeg::greedy(0, 0, 0, &[1, 2, 3]), PreSeg::greedy(1, 0, 0, &[4])]), 6);
        // a SegSamp maps to its kind: None = greedy, Some + ratio = real-q, Some = match rule
        let sp = SampParams::sampled(0.7, 0.8, 20, 0.0, 42, 5).unwrap();
        assert_eq!(PreSeg::of(0, 0, 0, &[1], &SegSamp::GREEDY, 3, false).kind, 0);
        assert_eq!(PreSeg::of(0, 0, 0, &[1], &SegSamp { samp: Some(sp), ratio: false }, 3, false).kind, 1);
        let q = PreSeg::of(0, 0, 0, &[1], &SegSamp { samp: Some(sp), ratio: true }, 5, true);
        assert_eq!((q.kind, q.launched, q.wasted), (2, 5, 1));
        assert_eq!(q.samp, sp.pack_flag(1), "the row-0 block the verify stages (flag 1 = the lane's own words)");
        // the failure message can name every lane of a pack from the words alone
        let (d0, d1) = ([11, 12, 13], [21, 22]);
        let txt = describe_pre_words(&pre_words(&[PreSeg::of(1, 100, 7, &d0, &SegSamp::GREEDY, 4, true),
                                                  PreSeg::of(2, 200, 8, &d1, &SegSamp { samp: Some(sp), ratio: true }, 3, false)]));
        assert!(txt.contains("seg 0: 4 rows slot 1 pos 100 bonus 7 kind 0 passes 4 (wasted) drafts [11, 12, 13]")
                && txt.contains("seg 1: 3 rows slot 2 pos 200 bonus 8 kind 2 passes 3 drafts [21, 22]"), "{txt}");
        assert!(describe_pre_words(&[9, 1]).contains("malformed"));
    }

    #[test]
    fn the_post_verify_words_carry_each_lane_and_every_value() {
        let l0 = PostLane { accepted: 2, rows: 4, emitted: &[5, 6, 7], drafts: &[5, 6, 9], confs: &[1.5, 0.5, 0.25] };
        let l1 = PostLane { accepted: 0, rows: 3, emitted: &[8], drafts: &[1, 2], confs: &[0.1, 0.2] };
        let w = post_words(&[l0, l1]);
        assert_eq!(&w[..3], &[2, 4, 3]);
        assert_eq!(w.len(), (3 + 3 + 1 + 3 + 1 + 3) + (3 + 1 + 1 + 2 + 1 + 2));
        let again = post_words(&[l0, l1]);
        assert_eq!(w, again, "deterministic");
        let swapped = post_words(&[l1, l0]);
        assert_ne!(w, swapped);
        let mut moved = l0; moved.accepted = 1;
        assert_ne!(w, post_words(&[moved, l1]), "an accept count change");
        let mut conf = l1; conf.confs = &[0.1, 0.2000001];
        assert_ne!(w, post_words(&[l0, conf]), "a one-ulp confidence change is a different word list");
    }

    #[test]
    fn graph_families_follow_the_sampling_mode_and_never_duplicate() {
        assert_eq!(fams(true, false), vec![PackFam::Greedy]);
        assert_eq!(fams(false, false), vec![PackFam::Greedy]);
        assert_eq!(fams(true, true), vec![PackFam::Greedy, PackFam::Rq]);
        assert_eq!(fams(false, true), vec![PackFam::Greedy, PackFam::Match]);
        // state() and fam_of() are inverse over the three families
        for f in [PackFam::Greedy, PackFam::Match, PackFam::Rq] {
            let (s, r) = f.state();
            assert_eq!(fam_of(s, r), f);
            assert_eq!(f.rq(), r);
        }
        // a pair with a real-q lane replays Rq whatever its partner is; samp_live without rq is Match
        assert_eq!(fam_of(true, true), PackFam::Rq);
        assert_eq!(fam_of(true, false), PackFam::Match);
        assert_eq!(fam_of(false, false), PackFam::Greedy);
    }

    /// PACK4-REVIEW B4: the lane view, as a function. Regime from the last row, the straddle, the
    /// clamp flag, the missing-graph reason and the precedence of the reasons.
    #[test]
    fn the_lane_view_is_a_pure_function_of_what_the_scheduler_reads() {
        let lim = 2051usize; // the sparse switch: positions above it are sparse
        let live = |p: usize| p > lim;
        let base = LaneFacts { mtp_armed: true, ..Default::default() };
        let v = |pos, k_pre, kd, pre, ready: [bool; 2], f: LaneFacts| lane_view(3, f, pos, k_pre, kd, pre, ready, &live);
        // dense, depth 7, graph present: packable, dense, not clamped
        let a = v(100, 7, 7, false, [true, true], base);
        assert_eq!(a, LaneView { slot: 3, block: None, regime: 0, clamped: false });
        // fully past the switch: sparse regime
        assert_eq!(v(3000, 7, 7, false, [true, true], base).regime, 1);
        // the round's regime is its LAST row's: p = 2045, depth 7 reaches 2052 > 2051: sparse, and the
        // first row (2046) is dense, so the round straddles (only reachable with --qsa-row-regime 0:
        // the per-row regime clamps the depth so it never does)
        let s = v(2045, 7, 7, false, [true, true], base);
        assert_eq!((s.regime, s.block), (1, Some(Skip::MixedRegime)));
        // with the regime clamp the depth is 6 (p + 6 = 2051: dense), no straddle, and `clamped` says so
        let c = v(2045, 7, 6, false, [true, true], base);
        assert_eq!((c.regime, c.block, c.clamped), (0, None, true));
        // the clamp at the switch itself: depth 0 = a plain step, never packable
        let z = v(2051, 7, 0, false, [true, true], base);
        assert_eq!(z.block, Some(Skip::Depth0));
        // a missing graph for THIS regime, not the other one
        assert_eq!(v(100, 7, 7, false, [false, true], base).block, Some(Skip::NoGraph));
        assert_eq!(v(3000, 7, 7, false, [false, true], base).block, None);
        assert_eq!(v(3000, 7, 7, false, [true, false], base).block, Some(Skip::NoGraph));
        // precedence: prefill beats every lane fact; a lane fact beats the straddle and the missing graph
        let bad = LaneFacts { penalized: true, ..base };
        assert_eq!(v(2045, 7, 7, true, [false, false], bad).block, Some(Skip::Prefill));
        assert_eq!(v(2045, 7, 7, false, [false, false], bad).block, Some(Skip::Penalty));
        assert_eq!(v(2045, 7, 7, false, [false, false], base).block, Some(Skip::MixedRegime));
        // an MTP-off lane is MtpOff whatever else is true
        assert_eq!(v(100, 7, 7, false, [true, true], LaneFacts::default()).block, Some(Skip::MtpOff));
        // the depth cap (A4): a depth above PACK_DEPTH_MAX cannot move the regime past what a pair can hold
        let capped = v(2040, 12, 12, false, [true, true], base);
        assert_eq!(capped.regime, (2040 + PACK_DEPTH_MAX > lim) as u64);
    }

    /// PACK4-REVIEW B3, now F1 (owner 2026-10-10): every posture the arm was never gated under is a
    /// FALLBACK with its plain-word reason; the default posture packs.
    #[test]
    fn ungated_postures_fall_back_at_boot() {
        let ok = Posture { kv: "q8", mtp_k: MTP_MAX_K, row_regime: true, a_ctrl_old: false, fcfs: false, kv_any: false, dense_v3: true };
        assert!(posture_fallbacks(&ok).is_empty(), "the default posture packs");
        // Q4K admission (owner 2026-11-13): q4k packs by default too, no --lane-pack-kv any needed.
        assert!(posture_fallbacks(&Posture { kv: "q4k", ..ok }).is_empty(), "q4k packs by default (admitted, KV_CACHE_Q4K.md §6 receipts)");
        for (name, p, want) in [
            ("fcfs", Posture { fcfs: true, ..ok }, "--lane-order fcfs"),
            ("kv f16", Posture { kv: "f16", ..ok }, "--kv-cache f16"),
            ("kv fp8", Posture { kv: "fp8", ..ok }, "--kv-cache fp8"),
            ("depth 5", Posture { mtp_k: 5, ..ok }, "--exl3-mtp-k 5"),
            ("row regime off", Posture { row_regime: false, ..ok }, "--qsa-row-regime 0"),
            ("a-ctrl old", Posture { a_ctrl_old: true, ..ok }, "--a-ctrl old"),
            ("dense v3 off", Posture { dense_v3: false, ..ok }, "--exl3-dense-v3 0"),
        ] {
            let v = posture_fallbacks(&p);
            assert_eq!(v.len(), 1, "{name}: one reason per fallback, got {v:?}");
            assert!(v[0].contains(want), "{name}: {} misses {want}", v[0]);
        }
        // F1: several ungated options at once = several reasons in one line, joined by the boot.
        let multi = Posture { kv: "f16", mtp_k: 5, ..ok };
        assert_eq!(posture_fallbacks(&multi).len(), 2, "each reason is listed");
        // --lane-pack-kv any (opt-in, ungated): an f16 / fp8 cache packs; the other reasons still apply
        for kv in ["f16", "fp8"] {
            assert!(posture_fallbacks(&Posture { kv, kv_any: true, ..ok }).is_empty(), "kv {kv} under --lane-pack-kv any packs");
        }
        assert_eq!(posture_fallbacks(&Posture { kv: "f16", kv_any: true, mtp_k: 5, ..ok }).len(), 1, "only the depth reason is left");
    }

    /// F10 (owner 2026-10-12): the default is ON, resolved from the registry row; `parse(None)` stays the
    /// raw-argv meaning (off) that pure callers use; a failed boot-time build is a fallback reason, never an abort.
    #[test]
    fn the_default_is_on_and_a_failed_build_falls_back() {
        let row = crate::opts::lookup("lane-pack").expect("registered").def();
        assert_eq!(row.def, "on", "the registry default");
        assert_eq!(LanePack::parse(Some(row.def)).unwrap(), LanePack::Pairs);
        assert_eq!(LanePack::parse(Some("off")).unwrap(), LanePack::Off, "--lane-pack off restores the old behaviour");
        assert_eq!(LanePack::parse(Some("2")).unwrap(), LanePack::Pairs, "2 keeps its meaning");
        assert_eq!(LanePack::parse(None).unwrap(), LanePack::Off, "an absent raw value is the raw-argv meaning");
        // the real call shape: Ok passes through, Err becomes a plain reason that names the cause
        let ok: std::result::Result<u32, String> = build_or_fall_back(|| Ok(7));
        assert_eq!(ok, Ok(7));
        let err = build_or_fall_back::<u32>(|| anyhow::bail!("CUDA_ERROR_OUT_OF_MEMORY")).unwrap_err();
        assert!(err.contains("could not be built at boot") && err.contains("CUDA_ERROR_OUT_OF_MEMORY"), "{err}");
    }

    /// PACK4-REVIEW A5: `--help` quoted x1.18 / 95.8 -> 113.0 tok/s from a pre-final binary (the final
    /// binary's single pass read x1.13). Measured numbers go stale with every build, so the flag's help
    /// says what the flag does and where the numbers live; this pin keeps a figure from coming back.
    #[test]
    fn the_help_text_quotes_no_measured_numbers() {
        let t = src_text!("opts_table");
        let i = t.find("flag: \"lane-pack\"").expect("the flag's table row");
        let row = &t[i..i + t[i..].find("\" },").expect("end of the row")];
        assert!(row.len() > 500, "the help text is there");
        for bad in ["tok/s", "x1.", "95.8", "113.0", "GB of", "0.4 GB"] {
            assert!(!row.contains(bad), "a measured figure ({bad}) in the --lane-pack help: it goes stale with every build");
        }
        assert!(row.contains("Speed factors are measured per build and are not quoted here"));
    }

    /// PACK4-REVIEW A4: the segment cap is derived from the lone verify's width, a pair of maximal
    /// segments always fits, and every depth the planner can hand to `pack_rows` is accepted.
    #[test]
    fn the_segment_cap_follows_the_lone_verify_width() {
        assert_eq!(PACK_DEPTH_MAX, MTP_MAX_K.min(PACK_ROWS_MAX / PACK_K - 1));
        assert_eq!(PACK_SEG_ROWS_MAX, PACK_DEPTH_MAX + 1);
        assert!(PACK_SEG_ROWS_MAX <= MTP_MAX_K + 1 && PACK_K * PACK_SEG_ROWS_MAX <= PACK_ROWS_MAX);
        for w0 in 1..=PACK_DEPTH_MAX {
            for w1 in 1..=PACK_DEPTH_MAX {
                let (m0, m1, r) = pack_rows(w0, w1).unwrap_or_else(|| panic!("a planned pair {w0} + {w1} must fit"));
                assert!(m0 + m1 == r && r <= PACK_ROWS_MAX && PACK_WIDTHS.contains(&r));
            }
        }
        assert!(pack_rows(PACK_DEPTH_MAX + 1, 1).is_none(), "one past the cap is refused by the pure function too");
    }

    /// Off == today: no plan for ANY lane set, so the caller cannot leave master's path.
    #[test]
    fn off_and_single_lane_never_plan() {
        let sets: Vec<Vec<LaneView>> = vec![
            vec![], vec![ok(0)], vec![ok(0), ok(1)], vec![ok(0), ok(1), ok(2)],
            vec![ok(0), lv(1, Some(Skip::Sampled), 0, false)], (0..16).map(ok).collect(),
        ];
        for s in &sets {
            for rot in 0..5 {
                assert_eq!(plan(LanePack::Off, s, rot), None, "off must never plan ({} lanes)", s.len());
            }
        }
        // on: a lone busy request never packs and is not even planned
        for rot in 0..5 {
            assert_eq!(plan(LanePack::Pairs, &[], rot), None);
            assert_eq!(plan(LanePack::Pairs, &[ok(3)], rot), None);
        }
    }

    #[test]
    fn every_lane_level_reason() {
        let base = LaneFacts { mtp_armed: true, depth: 4, ..Default::default() };
        assert_eq!(lane_block(&base), None);
        assert_eq!(lane_block(&LaneFacts { mtp_armed: false, ..base }), Some(Skip::MtpOff));
        assert_eq!(lane_block(&LaneFacts { force_control: true, ..base }), Some(Skip::Control));
        assert_eq!(lane_block(&LaneFacts { probing: true, ..base }), Some(Skip::Probe));
        // PACK4S: a sampled lane packs; only a boot with the lone-verify sampler dump keeps it out
        assert_eq!(lane_block(&LaneFacts { sampled: true, ..base }), None);
        assert_eq!(lane_block(&LaneFacts { sampled: true, sampled_lone: true, ..base }), Some(Skip::Sampled));
        assert_eq!(lane_block(&LaneFacts { sampled_lone: true, ..base }), None, "the dump keeps GREEDY lanes packable");
        assert_eq!(lane_block(&LaneFacts { penalized: true, ..base }), Some(Skip::Penalty));
        assert_eq!(lane_block(&LaneFacts { sampled: true, penalized: true, ..base }), Some(Skip::Penalty),
                   "a penalized sampled lane still stays out (penalties are step 2)");
        assert_eq!(lane_block(&LaneFacts { image: true, ..base }), Some(Skip::Image));
        assert_eq!(lane_block(&LaneFacts { logprobs_tp: true, ..base }), Some(Skip::LogprobsTp));
        assert_eq!(lane_block(&LaneFacts { depth: 0, ..base }), Some(Skip::Depth0));
        // precedence: an MTP-off lane is "mtp_off" whatever else is true of it
        assert_eq!(lane_block(&LaneFacts { mtp_armed: false, sampled: true, depth: 0, ..base }), Some(Skip::MtpOff));
        assert_eq!(lane_block(&LaneFacts { sampled: true, sampled_lone: true, penalized: true, image: true, ..base }), Some(Skip::Sampled));
        assert_eq!(lane_block(&LaneFacts { sampled: true, image: true, ..base }), Some(Skip::Image));
    }

    #[test]
    fn two_lanes() {
        let p = plan(LanePack::Pairs, &[ok(0), ok(1)], 0).unwrap();
        assert_eq!(p, Plan { pairs: vec![[0, 1]], lone: vec![] });
        // the rotation does not change a two-lane pair (and the pair is stored slot-ascending)
        for rot in 0..6 {
            assert_eq!(plan(LanePack::Pairs, &[ok(2), ok(5)], rot).unwrap().pairs, vec![[2, 5]]);
        }
        // one blocked lane: the other has nobody to pack with
        for why in [Skip::MtpOff, Skip::Control, Skip::Probe, Skip::Sampled, Skip::Penalty, Skip::Image, Skip::Depth0] {
            let p = plan(LanePack::Pairs, &[ok(0), lv(1, Some(why), 0, false)], 0).unwrap();
            assert!(p.pairs.is_empty());
            assert_eq!(p.lone, vec![(0, Skip::FewPackable), (1, why)], "{why:?}");
        }
        // both blocked: each lane keeps its own reason
        let p = plan(LanePack::Pairs, &[lv(0, Some(Skip::Sampled), 0, false), lv(1, Some(Skip::Penalty), 0, false)], 0).unwrap();
        assert_eq!(p.lone, vec![(0, Skip::Sampled), (1, Skip::Penalty)]);
    }

    #[test]
    fn regime_rules() {
        // different sides of the switch: mixed regime, both serial
        let p = plan(LanePack::Pairs, &[lv(0, None, 0, false), lv(1, None, 1, false)], 0).unwrap();
        assert!(p.pairs.is_empty());
        assert_eq!(p.lone, vec![(0, Skip::MixedRegime), (1, Skip::MixedRegime)]);
        // same regime, one lane clamped at the switch this round
        let p = plan(LanePack::Pairs, &[lv(0, None, 0, true), lv(1, None, 0, false)], 0).unwrap();
        assert!(p.pairs.is_empty());
        assert_eq!(p.lone, vec![(0, Skip::RegimeClamp), (1, Skip::RegimeClamp)]);
        // both clamped: every row of both is at/below the switch, they pack
        let p = plan(LanePack::Pairs, &[lv(0, None, 0, true), lv(1, None, 0, true)], 0).unwrap();
        assert_eq!(p.pairs, vec![[0, 1]]);
        // three lanes, one past the switch: the two short lanes pack, the long one is "mixed regime"
        for rot in 0..4 {
            let p = plan(LanePack::Pairs, &[lv(0, None, 0, false), lv(1, None, 1, false), lv(2, None, 0, false)], rot).unwrap();
            assert_eq!(p.pairs, vec![[0, 2]], "rot {rot}");
            assert_eq!(p.lone, vec![(1, Skip::MixedRegime)], "rot {rot}");
        }
        // two regime groups of two: two pairs
        let p = plan(LanePack::Pairs, &[lv(0, None, 0, false), lv(1, None, 1, false), lv(2, None, 0, false), lv(3, None, 1, false)], 0).unwrap();
        assert_eq!(p.pairs, vec![[0, 2], [1, 3]]);
        assert!(p.lone.is_empty());
    }

    /// The three-lane rotation, deterministically: the lone role visits every lane once per three
    /// steps, every lane runs exactly one round per step, and no pairing is fixed.
    #[test]
    fn three_lane_rotation_is_fair() {
        let lanes = [ok(0), ok(1), ok(2)];
        let mut lone_seq = Vec::new();
        let mut pair_cnt = std::collections::BTreeMap::<[usize; 2], usize>::new();
        let mut rounds = [0usize; 3];
        for rot in 0..300u64 {
            let p = plan(LanePack::Pairs, &lanes, rot).unwrap();
            assert_eq!(p.pairs.len(), 1);
            assert_eq!(p.lone.len(), 1);
            assert_eq!(p.lone[0].1, Skip::OddLane);
            lone_seq.push(p.lone[0].0);
            *pair_cnt.entry(p.pairs[0]).or_default() += 1;
            for s in p.pairs[0] { rounds[s] += 1; }
            rounds[p.lone[0].0] += 1;
        }
        assert_eq!(&lone_seq[..6], &[2, 0, 1, 2, 0, 1], "lone = c, a, b, ...");
        for s in 0..3 {
            assert_eq!(rounds[s], 300, "every lane runs one round per step");
            assert_eq!(lone_seq.iter().filter(|&&x| x == s).count(), 100, "lane {s} is lone once per three steps");
            // never lone twice in a row, never packed more than twice in a row
            let gaps: Vec<usize> = lone_seq.iter().enumerate().filter(|(_, &x)| x == s).map(|(i, _)| i).collect();
            assert!(gaps.windows(2).all(|w| w[1] - w[0] == 3), "lane {s}: lone turns are exactly three steps apart");
        }
        assert_eq!(pair_cnt.len(), 3, "all three pairings occur");
        assert!(pair_cnt.values().all(|&c| c == 100));
    }

    /// Lanes arriving and leaving (the FCFSB-style simulation): whatever the busy set does, every
    /// busy lane gets one round per step, and a lane that stays busy among an odd number of
    /// packable lanes is never the odd one for more than one step in any `n` consecutive steps
    /// of an unchanged busy set.
    #[test]
    fn rotation_with_arrivals_and_departures() {
        // (first step, last step) of each of 6 lanes; deterministic schedule
        let life = [(0u64, 400u64), (0, 137), (10, 400), (50, 260), (138, 320), (261, 400)];
        let mut odd_run = [0usize; 6];
        let mut packed = [0usize; 6];
        let mut lone = [0usize; 6];
        let mut prev_set: Vec<usize> = Vec::new();
        let mut since_change = 0usize;
        let mut odd_in_window: Vec<usize> = vec![0; 6];
        for step in 0..=400u64 {
            let busy: Vec<usize> = (0..6).filter(|&s| life[s].0 <= step && step <= life[s].1).collect();
            let views: Vec<LaneView> = busy.iter().map(|&s| ok(s)).collect();
            if busy != prev_set { since_change = 0; odd_in_window = vec![0; 6]; prev_set = busy.clone(); }
            let Some(p) = plan(LanePack::Pairs, &views, step) else {
                assert!(busy.len() < 2);
                continue;
            };
            // every busy lane appears exactly once: in a pair or lone
            let mut seen: Vec<usize> = p.pairs.iter().flatten().copied().chain(p.lone.iter().map(|(s, _)| *s)).collect();
            seen.sort();
            assert_eq!(seen, busy, "step {step}");
            assert_eq!(p.pairs.len(), busy.len() / 2, "step {step}: every possible pair is formed");
            for s in p.pairs.iter().flatten() { packed[*s] += 1; odd_run[*s] = 0; }
            for (s, why) in &p.lone {
                assert_eq!(*why, Skip::OddLane);
                lone[*s] += 1;
                odd_run[*s] += 1;
                odd_in_window[*s] += 1;
                assert!(odd_run[*s] <= 1 || since_change == 0, "step {step}: lane {s} was the odd lane twice in a row");
            }
            since_change += 1;
            if since_change == busy.len() && busy.len() % 2 == 1 {
                for &s in &busy { assert_eq!(odd_in_window[s], 1, "step {step}: lane {s} odd once per {} steps", busy.len()); }
                since_change = 0;
                odd_in_window = vec![0; 6];
            }
        }
        for s in 0..6 {
            assert!(packed[s] > 0, "lane {s} never packed");
            assert!(packed[s] >= lone[s], "lane {s}: packed {} lone {} — the odd role is the minority", packed[s], lone[s]);
        }
    }

    #[test]
    fn control_cadence_is_the_serial_arms() {
        assert_eq!(ctrl_period(true, 0.0, 10.0), 128);
        assert_eq!(ctrl_period(true, 30.0, 0.0), 128);
        assert_eq!(ctrl_period(true, 30.0, 23.9), 1024);
        assert_eq!(ctrl_period(true, 30.0, 24.0), 128, "0.8x is not clearly below");
        assert_eq!(ctrl_period(false, 30.0, 10.0), 128, "--a-ctrl set: the fixed cadence");
        // the pair consumes counter values c+1 and c+2
        assert!(!pair_owes_control(0, 128, 128));
        assert!(pair_owes_control(127, 128, 128), "the first slot lands on 128");
        assert!(pair_owes_control(126, 128, 128), "the second slot lands on 128");
        assert!(!pair_owes_control(128, 128, 128));
        assert!(pair_owes_control(1022, 128, 1024), "second slot lands on 1024");
        assert!(pair_owes_control(1022, 1024, 128), "second slot lands on a multiple of its own period, 128");
        assert!(!pair_owes_control(1020, 1024, 1024), "1021 and 1022: neither lane is due");
        assert!(pair_owes_control(1023, 1024, 1024), "first slot lands on 1024");
        // count for count: over a long run a pair-or-serial scheduler runs exactly as many controls
        // as the serial arm (every counter value is consumed once; multiples of 128 are controls)
        let (mut c, mut packed, mut controls) = (0usize, 0usize, 0usize);
        while c < 128 * 50 {
            if pair_owes_control(c, 128, 128) {
                for _ in 0..2 { c += 1; if c % 128 == 0 { controls += 1; } } // two serial rounds
            } else {
                c += 2;
                packed += 1;
                assert!((c - 1) % 128 != 0 && c % 128 != 0, "a packed pair swallowed a control at {c}");
            }
        }
        assert_eq!(controls, 50);
        assert!(packed > 3000);
    }

    #[test]
    fn time_division() {
        assert_eq!(lane_share_ms(49.4, 2), 24.7);
        assert_eq!(lane_share_ms(35.2, 1), 35.2, "a lone round keeps its time");
        assert_eq!(lane_share_ms(10.0, 0), 10.0);
    }

    #[test]
    fn row_allocation() {
        assert_eq!(pack_rows(4, 4), Some((5, 5, 10)));
        assert_eq!(pack_rows(7, 7), Some((8, 8, 16)));
        assert_eq!(pack_rows(1, 1), Some((2, 2, 4)));
        assert_eq!(pack_rows(5, 3), Some((6, 4, 10)));
        assert_eq!(pack_rows(1, 2), Some((2, 3, 5)), "odd totals are exact: no padding row");
        assert_eq!(pack_rows(0, 4), None);
        assert_eq!(pack_rows(4, 0), None);
        assert_eq!(pack_rows(8, 1), None, "a lone verify is at most 8 rows");
        // every pair of legal depths fits the 16-row ceiling and lands on a precaptured width
        for w0 in 1..=7 {
            for w1 in 1..=7 {
                let (m0, m1, r) = pack_rows(w0, w1).unwrap();
                assert!(m0 <= PACK_SEG_ROWS_MAX && m1 <= PACK_SEG_ROWS_MAX && r <= PACK_ROWS_MAX);
                assert!(PACK_WIDTHS.contains(&r), "R = {r}");
            }
        }
        assert_eq!(PACK_WIDTHS.clone().count(), 13);
    }

    #[test]
    fn staging_metadata() {
        let (d0, d1) = ([11, 12, 13, 14], [21, 22]);
        let s0 = SegIn { slot: 2, b: 10, p: 100, drafts: &d0 };
        let s1 = SegIn { slot: 0, b: 20, p: 7000, drafts: &d1 };
        let m = pack_meta(&s0, &s1, 16).unwrap();
        assert_eq!(m.r, 8);
        assert_eq!(m.rows, vec![(2, 10, 100), (2, 11, 101), (2, 12, 102), (2, 13, 103), (2, 14, 104),
                                (0, 20, 7000), (0, 21, 7001), (0, 22, 7002)]);
        // {slot, row_start, len, flags = nostore}
        assert_eq!(m.segs, [[2, 0, 5, 1], [0, 5, 3, 1]]);
        // drafts at their segment's row offset; the bonus row's slot (row0 + len - 1) stays 0
        assert_eq!(m.d_dev, vec![11, 12, 13, 14, 0, 21, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        // exactly R rows: NO padding row is staged (module note); rows past R are the caller's
        // zero fill, as in the lone verify
        assert_eq!(m.rows.len(), m.r);
        // an odd total stays exact
        let m = pack_meta(&SegIn { slot: 0, b: 1, p: 5, drafts: &[2] }, &SegIn { slot: 1, b: 3, p: 9, drafts: &[4, 5] }, 16).unwrap();
        assert_eq!((m.r, m.rows.len()), (5, 5));
        assert_eq!(m.segs, [[0, 0, 2, 1], [1, 2, 3, 1]]);
        assert_eq!(&m.d_dev[..5], &[2, 0, 4, 5, 0]);
        // refusals
        assert!(pack_meta(&SegIn { slot: 0, b: 1, p: 5, drafts: &[] }, &s1, 16).is_err(), "a lane without a draft");
        assert!(pack_meta(&SegIn { slot: 0, b: 1, p: 5, drafts: &d1 }, &s1, 16).is_err(), "one slot twice");
        assert!(pack_meta(&s0, &SegIn { slot: 1, b: 1, p: 5, drafts: &d1 }, 7).is_err(), "draft buffer too narrow");
    }

    #[test]
    fn readback_per_segment() {
        let segs = [[2, 0, 5, 1], [0, 5, 3, 1]];
        // segment 0 accepted 2 drafts (emits d0, d1, e2), segment 1 accepted 0 (emits e0)
        let out = [11, 12, 99, -1, -1, 77, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0];
        let acc = [2, 2, 0, 0, 0, 0];
        assert_eq!(seg_result(&out, &acc, 0, &segs[0]).unwrap(), (2, vec![11, 12, 99]));
        assert_eq!(seg_result(&out, &acc, 1, &segs[1]).unwrap(), (0, vec![77]));
        // full accept of segment 1: a = len - 1
        let out2 = [11, -1, -1, -1, -1, 21, 22, 55, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(seg_result(&out2, &[2, 0, 0, 2, 0, 0], 1, &segs[1]).unwrap(), (2, vec![21, 22, 55]));
        // tripwires: a commit key for another slot, an accept past the segment, a -1 inside the emitted run
        assert!(seg_result(&out, &[3, 2, 0, 0, 0, 0], 0, &segs[0]).is_err());
        assert!(seg_result(&out, &[2, 5, 0, 0, 0, 0], 0, &segs[0]).is_err());
        assert!(seg_result(&[11, -1, 99, -1, -1, 77, -1, -1], &acc, 0, &segs[0]).is_err());
    }

    #[test]
    fn skip_table_is_consistent() {
        let labels: std::collections::BTreeSet<&str> = Skip::ALL.iter().map(|s| s.label()).collect();
        assert_eq!(labels.len(), Skip::ALL.len(), "labels are unique");
        for (i, s) in Skip::ALL.iter().enumerate() {
            assert_eq!(s.index(), i);
            assert!(s.label().chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'), "{}", s.label());
            assert!(!s.text().is_empty());
        }
        assert_eq!(Skip::MixedRegime.text(), "mixed regime", "the R7 gate greps this text");
    }

    // ---- S-B9-CF-PACK4S ----

    fn sp(t: f32, top_p: f32, top_k: usize, seed: u64, ctr: u32) -> SampParams {
        SampParams::sampled(t, top_p, top_k, 0.0, seed, ctr).unwrap()
    }
    fn row(ps: &PackSamp, r: usize) -> [u32; 8] { ps.words[r * 8..r * 8 + 8].try_into().unwrap() }

    /// The planner with sampled lanes: they pair with each other and with greedy lanes (MIXED
    /// pairs); a penalized lane still runs alone; the dump diagnostic keeps sampled lanes out.
    #[test]
    fn sampled_lanes_pack_and_penalized_still_skip() {
        let base = LaneFacts { mtp_armed: true, depth: 4, ..Default::default() };
        let view = |slot: usize, f: LaneFacts| lv(slot, lane_block(&f), 0, false);
        let (greedy, sampled) = (base, LaneFacts { sampled: true, ..base });
        let penalized = LaneFacts { sampled: true, penalized: true, ..base };
        // sampled + sampled
        assert_eq!(plan(LanePack::Pairs, &[view(0, sampled), view(1, sampled)], 0).unwrap().pairs, vec![[0, 1]]);
        // mixed greedy + sampled, either slot order
        assert_eq!(plan(LanePack::Pairs, &[view(0, greedy), view(1, sampled)], 0).unwrap().pairs, vec![[0, 1]]);
        assert_eq!(plan(LanePack::Pairs, &[view(2, sampled), view(3, greedy)], 0).unwrap().pairs, vec![[2, 3]]);
        // three lanes of mixed kinds: one pair + the rotating lone role, whatever the kinds
        let three = [view(0, greedy), view(1, sampled), view(2, sampled)];
        let lone: Vec<usize> = (0..3).map(|rot| plan(LanePack::Pairs, &three, rot).unwrap().lone[0].0).collect();
        assert_eq!(lone, vec![2, 0, 1]);
        // a penalized lane stays out; the other two (one sampled, one greedy) still pack
        let p = plan(LanePack::Pairs, &[view(0, sampled), view(1, penalized), view(2, greedy)], 0).unwrap();
        assert_eq!(p.pairs, vec![[0, 2]]);
        assert_eq!(p.lone, vec![(1, Skip::Penalty)]);
        // sampled next to penalized only: nobody to pack with
        let p = plan(LanePack::Pairs, &[view(0, sampled), view(1, penalized)], 0).unwrap();
        assert!(p.pairs.is_empty());
        assert_eq!(p.lone, vec![(0, Skip::FewPackable), (1, Skip::Penalty)]);
        // the lone-verify sampler dump: sampled lanes are counted as `sampled`, greedy lanes pack
        let dump = |f: LaneFacts| LaneFacts { sampled_lone: true, ..f };
        let p = plan(LanePack::Pairs, &[view(0, dump(sampled)), view(1, dump(greedy)), view(2, dump(greedy))], 0).unwrap();
        assert_eq!(p.pairs, vec![[1, 2]]);
        assert_eq!(p.lone, vec![(0, Skip::Sampled)]);
    }

    /// Staging per row: flag 2 on a real-q lane's draft rows and 1 on its bonus row, 1 on every
    /// row of a match-mode lane, all-zero rows for a greedy lane; counters ctr_j + i.
    #[test]
    fn sampler_staging_flags_and_counters() {
        let (a, b) = (sp(1.0, 0.95, 20, 0xAAAA_0000_1111, 100), sp(0.7, 0.8, 20, 0xBBBB_0000_2222, 7));
        // segment 0: slot 2, 4 drafts (5 rows at 0); segment 1: slot 0, 2 drafts (3 rows at 5)
        let segs = [[2, 0, 5, 1], [0, 5, 3, 1]];
        // ratio + ratio
        let ps = pack_samp(&segs, &[SegSamp { samp: Some(a), ratio: true }, SegSamp { samp: Some(b), ratio: true }], 16).unwrap();
        assert_eq!(ps.words.len(), 16 * 8);
        assert!(ps.samp_live && ps.rq_live);
        let flags: Vec<u32> = (0..16).map(|r| row(&ps, r)[0]).collect();
        assert_eq!(flags, vec![2, 2, 2, 2, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0], "draft rows 2, bonus rows 1, unstaged rows 0");
        let ctrs: Vec<u32> = (0..8).map(|r| row(&ps, r)[6]).collect();
        assert_eq!(ctrs, vec![100, 101, 102, 103, 104, 7, 8, 9], "row i of segment j draws counter ctr_j + i");
        for r in 0..5 {
            assert_eq!(row(&ps, r), a_row(a, r as u32, if r < 4 { 2 } else { 1 }));
        }
        for i in 0..3 {
            assert_eq!(row(&ps, 5 + i), a_row(b, i as u32, if i < 2 { 2 } else { 1 }));
        }
        assert!(ps.words[8 * 8..].iter().all(|&w| w == 0), "no word past R rows is staged");
        // match + match: flag 1 on every row, the greedy graph family (rq_live false)
        let ps = pack_samp(&segs, &[SegSamp { samp: Some(a), ratio: false }, SegSamp { samp: Some(b), ratio: false }], 16).unwrap();
        assert!(ps.samp_live && !ps.rq_live);
        assert_eq!((0..8).map(|r| row(&ps, r)[0]).collect::<Vec<_>>(), vec![1; 8]);
        assert_eq!((0..8).map(|r| row(&ps, r)[6]).collect::<Vec<_>>(), vec![100, 101, 102, 103, 104, 7, 8, 9]);
        // greedy + greedy: nothing staged, nothing live (the PACK4 arm's state exactly)
        let ps = pack_samp(&segs, &[SegSamp::GREEDY, SegSamp::GREEDY], 16).unwrap();
        assert!(!ps.samp_live && !ps.rq_live && ps.words.iter().all(|&w| w == 0));
        // MIXED: greedy segment 0 stays all-zero (argmax) next to a real-q segment 1, and the reverse
        let ps = pack_samp(&segs, &[SegSamp::GREEDY, SegSamp { samp: Some(b), ratio: true }], 16).unwrap();
        assert!(ps.samp_live && ps.rq_live);
        assert!(ps.words[..5 * 8].iter().all(|&w| w == 0), "the greedy lane's rows are flag 0: both sampler kernels return");
        assert_eq!((5..8).map(|r| row(&ps, r)[0]).collect::<Vec<_>>(), vec![2, 2, 1]);
        let ps = pack_samp(&segs, &[SegSamp { samp: Some(a), ratio: true }, SegSamp::GREEDY], 16).unwrap();
        assert!(ps.words[5 * 8..].iter().all(|&w| w == 0));
        assert_eq!((0..5).map(|r| row(&ps, r)[0]).collect::<Vec<_>>(), vec![2, 2, 2, 2, 1]);
        // the counter wraps like the lone path's `ctr.wrapping_add(i)`
        let ps = pack_samp(&[[0, 0, 3, 1], [1, 3, 2, 1]], &[SegSamp { samp: Some(sp(1.0, 1.0, 0, 5, u32::MAX - 1)), ratio: true }, SegSamp::GREEDY], 16).unwrap();
        assert_eq!((0..3).map(|r| row(&ps, r)[6]).collect::<Vec<_>>(), vec![u32::MAX - 1, u32::MAX, 0]);
        // refusals: real-q without a sampler, rows outside the staged width
        assert!(pack_samp(&segs, &[SegSamp { samp: None, ratio: true }, SegSamp::GREEDY], 16).is_err());
        assert!(pack_samp(&[[0, 0, 8, 1], [1, 8, 8, 1]], &[SegSamp::GREEDY, SegSamp { samp: Some(b), ratio: true }], 15).is_err());
    }

    /// What the lone path stages for row i of a lane (`Lane::samp_rows` + `pack_flag`).
    fn a_row(p: SampParams, i: u32, flag: u32) -> [u32; 8] {
        SampParams::sampled(p.temperature, p.top_p, p.top_k as usize, p.min_p, p.seed, p.counter.wrapping_add(i)).unwrap().pack_flag(flag)
    }

    /// A lane's draws must not depend on its partner: the SAME lane staged next to two different
    /// partners (greedy and short; sampled with other parameters and long) and at either segment
    /// position has identical row blocks, and every one equals the lone path's row for that
    /// lane (same parameters, seed, counter ctr + i) up to the bonus row's flag.
    #[test]
    fn staged_rows_do_not_depend_on_the_partner() {
        let x = sp(1.0, 0.95, 20, 0x0123_4567_89AB_CDEF, 4242);
        for ratio in [true, false] {
            let lane = SegSamp { samp: Some(x), ratio };
            let partners = [(SegSamp::GREEDY, 3usize), (SegSamp { samp: Some(sp(0.7, 0.8, 40, 99, 1)), ratio }, 8usize),
                            (SegSamp { samp: Some(sp(1.3, 1.0, 0, 0xFFFF_FFFF_FFFF_FFFF, u32::MAX)), ratio }, 2usize)];
            for m in 2..=8usize {
                let mut seen: Vec<Vec<[u32; 8]>> = Vec::new();
                for &(partner, pm) in &partners {
                    // the lane as segment 0 (its slot is the lower one) ...
                    let ps = pack_samp(&[[0, 0, m as i32, 1], [1, m as i32, pm as i32, 1]], &[lane, partner], 16).unwrap();
                    seen.push((0..m).map(|i| row(&ps, i)).collect());
                    // ... and as segment 1, behind a partner of pm rows
                    let ps = pack_samp(&[[0, 0, pm as i32, 1], [1, pm as i32, m as i32, 1]], &[partner, lane], 16).unwrap();
                    seen.push((0..m).map(|i| row(&ps, pm + i)).collect());
                    assert_eq!(ps.rq_live, ratio, "the family follows the pair, the rows follow the lane");
                }
                assert!(seen.windows(2).all(|w| w[0] == w[1]), "m {m} ratio {ratio}: the lane's rows moved with its partner");
                for (i, r) in seen[0].iter().enumerate() {
                    let lone_flag = if ratio { 2 } else { 1 };
                    let want = a_row(x, i as u32, if ratio && i + 1 == m { 1 } else { lone_flag });
                    assert_eq!(*r, want, "row {i}");
                    // only the flag word can differ from the lone staging, and only on the bonus row
                    assert_eq!(r[1..], a_row(x, i as u32, lone_flag)[1..]);
                }
            }
        }
    }

    /// The draft-list hand-off: the list is sized from the scratch width (16 rows for the packed
    /// scratch, unchanged for <= 8 rows), a lane's rows 0..w go to its row offset, and every pair
    /// of legal depths stays inside the packed list.
    #[test]
    fn list_rows_and_handoff() {
        // MTP26/DEPTH: the served list follows MTP_MAX_K (one row per draft pass + the spare;
        // was a hard 8 at k=7). <= MTP_MAX_K+1 keeps the pre-PACK4S size; wider scratches
        // return their own width, unchanged.
        assert_eq!(MTP_MAX_K + 1, 11, "the served list: one row per draft pass + the spare");
        for w in 0..=11 { assert_eq!(rq_list_rows(w), 11, "scratch of {w} rows: the size before PACK4S"); }
        assert_eq!(rq_list_rows(PACK_ROWS_MAX), 16, "the packed scratch: a 16-row verify reads list rows up to 14");
        assert_eq!(rq_list_rows(12), 12);
        let dst = rq_list_rows(PACK_ROWS_MAX) * RQ_ROW;
        assert_eq!(RQ_ROW, 96);
        // segment 0 (row0 = 0) and segment 1 (row0 = m0)
        assert_eq!(list_handoff(0, 4, true, dst).unwrap(), Some((0, 4 * 96)));
        assert_eq!(list_handoff(5, 2, true, dst).unwrap(), Some((5 * 96, 2 * 96)));
        // greedy and match-mode lanes have no list
        assert_eq!(list_handoff(5, 2, false, dst).unwrap(), None);
        // every legal pair: both copies inside the list, disjoint, at the rows the kernel reads
        for w0 in 1..=7usize {
            for w1 in 1..=7usize {
                let (m0, _, r) = pack_rows(w0, w1).unwrap();
                let (o0, n0) = list_handoff(0, w0, true, dst).unwrap().unwrap();
                let (o1, n1) = list_handoff(m0, w1, true, dst).unwrap().unwrap();
                assert!(o0 + n0 <= o1 && o1 + n1 <= dst, "{w0}+{w1}");
                assert_eq!((o0 / RQ_ROW, o1 / RQ_ROW, (o1 + n1) / RQ_ROW), (0, m0, r - 1), "draft row i of segment j reads list row row0_j + i");
            }
        }
        // the OLD 8-row list cannot hold a pair's second segment: the resize is load-bearing
        assert!(list_handoff(8, 7, true, 8 * RQ_ROW).is_err());
        assert!(list_handoff(0, MTP_MAX_K + 1, true, dst).is_err(), "at most MTP_MAX_K draft rows");
        assert!(list_handoff(0, 0, true, dst).is_err());
    }

    /// The `arms=` provenance trace: which arm produced the token with generated index g.
    #[test]
    fn arm_trace() {
        let mut t = ArmTrace::default();
        assert_eq!(t.text(), "-");
        for (arm, n) in [('S', 2), ('P', 3), ('P', 1), ('c', 1), ('p', 1), ('P', 12)] { t.push(arm, n); }
        assert_eq!(t.text(), "S2P3P1c1p1P12");
        assert_eq!((t.rounds, t.packed), (6, 3));
        let s = t.text();
        assert_eq!(ArmTrace::arm_of(&s, 0), None, "the seam token is in no round");
        let arms: String = (1..=20).map(|g| ArmTrace::arm_of(&s, g).unwrap()).collect();
        assert_eq!(arms, "SSPPPPcpPPPPPPPPPPPP");
        assert_eq!(ArmTrace::arm_of(&s, 21), None, "past the recorded tokens");
        assert_eq!(ArmTrace::arm_of("-", 1), None);
        // the cap: later rounds are counted, not spelled out, and the text says so
        let mut t = ArmTrace::default();
        for _ in 0..ARM_TRACE_ROUNDS + 5 { t.push('P', 2); }
        assert_eq!((t.rounds, t.packed), (ARM_TRACE_ROUNDS + 5, ARM_TRACE_ROUNDS + 5));
        assert!(t.text().ends_with('+') && t.text().matches('P').count() == ARM_TRACE_ROUNDS);
        assert_eq!(ArmTrace::arm_of(&t.text(), 2 * ARM_TRACE_ROUNDS), Some('P'));
        assert_eq!(ArmTrace::arm_of(&t.text(), 2 * ARM_TRACE_ROUNDS + 1), None);
    }
