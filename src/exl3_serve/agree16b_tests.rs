//! Unit tests (module `agree16b_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::*;
    use crate::net::hub_mock::{run_ranks, Net};
    use std::time::Duration;

    fn armed(rank: usize, world: usize, n: usize) -> TpSync {
        let mut t = TpSync::new(rank as i32, world as i32);
        assert_eq!(t.drill_width, None, "no drill flag is set in the unit suite");
        t.drill_width = Some(n);
        t
    }

    /// W3: the drill lies ONCE (it used to lie on every N-row round), and only about its own width.
    #[test]
    fn the_width_drill_fires_once_then_tells_the_truth() {
        for world in [2usize, 4] {
            for rank in 0..world {
                let mut t = armed(rank, world, 16);
                for m in [0usize, 1, 5, 15, 17] { assert_eq!(t.width_said(m), m, "width {m} is not the drill's"); }
                assert_eq!(t.drill_hits, 0, "still armed");
                assert_eq!(t.width_said(16), 15 - rank, "rank {rank}: the first 16-row round is reported as 16 - 1 - rank");
                assert_eq!(t.drill_hits, 1);
                for _ in 0..8 { assert_eq!(t.width_said(16), 16, "rank {rank}: spent, the truth from here on"); }
                assert_eq!(t.drill_hits, 1);
            }
        }
        // not armed: the truth, always
        let mut t = TpSync::new(0, 2);
        for m in 0..=31 { assert_eq!(t.width_said(m), m); }
        // a width the rank term cannot go below zero for is left alone even if forced past `drill_plan`
        let mut t = armed(1, 2, 1);
        assert_eq!((t.width_said(1), t.drill_hits), (1, 0));
    }

    /// W3, TP=2 (the pairwise token): armed on the head only it is the 15-against-16 fault; armed on the
    /// node only, or on BOTH ranks, the two words still differ, in both directions. Before AGREE16B two
    /// armed ranks both said N - 1 and agreed.
    #[test]
    fn a_width_drill_on_one_rank_or_on_both_fails_the_pairwise_agree() {
        let words = |said: [usize; 2], h: u32| -> [(u8, u8, (u8, u32)); 2] {
            [0, 1].map(|r| {
                let (a, k) = tp_agree_fields(0, said[r]).unwrap();
                (a, k, crate::net::agree_token_reply(crate::net::agree_token(agree_field(87, AGREE_A), a, k, h)))
            })
        };
        let fails = |said: [usize; 2], h: u32| {
            let w = words(said, h);
            let (e0, e1) = (tp_agree_decide(w[0].0, w[0].1, h, Some(w[1].2)), tp_agree_decide(w[1].0, w[1].1, h, Some(w[0].2)));
            assert!(e0.is_err() && e1.is_err(), "widths said {said:?} (digest {h:08x}) must fail on both ranks: {e0:?} / {e1:?}");
            (e0.unwrap_err(), e1.unwrap_err())
        };
        for n in 2..=31usize {
            for h in [0u32, 0x99915f3e, 0xFFFF_FFFF] {
                let said = |on: [bool; 2]| [0, 1].map(|r| if on[r] { armed(r, 2, n).width_said(n) } else { n });
                assert_eq!(said([false, false]), [n, n]);
                assert_eq!(said([true, false]), [n - 1, n], "the head alone");
                assert_eq!(said([false, true]), [n, n - 2], "the node alone");
                assert_eq!(said([true, true]), [n - 1, n - 2], "both ranks: two different lies");
                for on in [[true, false], [false, true], [true, true]] { fails(said(on), h); }
                let w = words(said([false, false]), h);
                assert_eq!(tp_agree_decide(w[0].0, w[0].1, h, Some(w[1].2)), Ok(()), "no drill: the round agrees");
            }
        }
        // the gate's line, head armed with 16: "this rank (accept 0, width 15, ..) vs peer (accept 0, width 16, ..)"
        let (head, node) = fails([15, 16], 0x99915f3e);
        assert!(head.contains("this rank (accept 0, width 15, word 79915f2d) vs peer (accept 0, width 16, word 81915f2d)"), "{head}");
        assert!(node.contains("this rank (accept 0, width 16, word 81915f2d) vs peer (accept 0, width 15, word 79915f2d)"), "{node}");
    }

    /// W3, world 4 (the hub frames): the drill armed on EVERY rank still fails the 16-row round on every
    /// rank (each reports a different width), and a rank armed for another width does not disturb it.
    #[test]
    fn a_width_drill_on_every_rank_fails_the_hub_round_on_every_rank() {
        const W: usize = 4;
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = armed(r, W, 16);
            let first = t.round_hub(hub, "verify", 2, 5, &[1, 2], 1.0, 1.0, false, 0, |_| Ok(vec![])).map(|_| ()).map_err(|e| format!("{e:#}"));
            let second = t.round_hub(hub, "plain-step", 0, 16, &[1, 2], 1.0, 1.0, false, 0, |_| Ok(vec![])).map(|_| ()).map_err(|e| format!("{e:#}"));
            (first, second, t.drill_hits)
        });
        for (r, (first, second, hits)) in out.iter().enumerate() {
            assert!(first.is_ok(), "rank {r}: a 5-row round is not the drill's width: {first:?}");
            let e = second.as_ref().expect_err(&format!("rank {r}: every rank lied differently about the 16-row round — it must fail"));
            assert!(e.contains("verify width"), "rank {r}: {e}");
            assert_eq!(*hits, 1, "rank {r}: fired once");
        }
    }

    /// W4, now F2 (owner 2026-10-10): the lane count from the argv alone, through the ONE resolver the
    /// serve uses. A `--max-batch` above 16 no longer refuses anywhere: the cap resolves it to 16
    /// first (16 always fits the pairwise agree width), the boot prints the one plain line, and
    /// further requests wait in the queue.
    #[test]
    fn the_max_batch_refusal_reads_the_argv_the_serve_reads() {
        let argv = |v: &[&str]| -> Vec<String> { v.iter().map(|s| s.to_string()).collect() };
        assert_eq!(max_batch_arg(&argv(&["--server", "--tp", "2"])), 8, "the serve's default");
        assert_eq!(max_batch_arg(&argv(&["--max-batch", "16"])), 16);
        for ok in [&["--max-batch", "1"][..], &["--max-batch", "16"], &["--max-batch", "31"], &["--server"]] {
            assert!(tp_lane_width_refusal(&argv(ok), 2).is_ok(), "{ok:?} boots at TP=2");
        }
        for bad in ["32", "33", "64", "256", "287"] {
            let raw: usize = bad.parse().unwrap();
            for world in [1u32, 2, 4] {
                assert!(tp_lane_width_refusal(&argv(&["--server", "--max-batch", bad, "--tp", "2"]), world).is_ok(),
                        "--max-batch {bad} at world {world}: the 16-lane cap resolves it, it does not refuse");
            }
            assert_eq!(resolved_max_batch(&argv(&["--max-batch", bad])), (16, raw), "the one resolver caps at 16");
        }
        assert_eq!(resolved_max_batch(&argv(&["--server"])), (8, 8), "the default sits under the cap, uncapped");
    }

    /// W4, now F2, the call sites (pinned in the source: they sit in the boot path of a TP pair). The
    /// cap call sites remain as the BACKSTOP for a caller that bypasses the 16-lane resolver — the
    /// head before discovery and the pack sync, the node before the link, both before the RDMA
    /// attach; and `build_serve` resolves `--max-batch` through the same resolver before its backstop.
    #[test]
    fn the_max_batch_refusal_runs_before_any_network_step() {
        let code = |s: &str| s.lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("\n");
        let body = |src: &str, from: &str, to: &str| -> String {
            let a = src.find(from).unwrap_or_else(|| panic!("{from} not found"));
            let b = src[a + from.len()..].find(to).unwrap_or_else(|| panic!("{to} not found after {from}")) + a + from.len();
            code(&src[a..b])
        };
        let before = |text: &str, first: &str, then: &str, what: &str| {
            let (a, b) = (text.find(first).unwrap_or_else(|| panic!("{what}: `{first}` is missing")),
                          text.find(then).unwrap_or_else(|| panic!("{what}: `{then}` is missing")));
            assert!(a < b, "{what}: `{first}` must come before `{then}`");
        };
        let main = src_text!("main");
        let head = body(main, "fn run_exl3_tp_serve_head(", "\nfn ");
        before(&head, "tp_lane_width_refusal(args, world as u32)", "run_head_retain(", "the head's main");
        before(&head, "tp_lane_width_refusal(args, world as u32)", "TpConfig::from_opts()", "the head's main");
        let prelink = body(main, "fn exl3_tp_node_prelink(", "\nfn ");
        before(&prelink, "tp_lane_width_refusal(&argv, tpc.world as u32)?", "Exl3NodeProgram::Serve(argv)", "the node's prelink");
        let serve = src_text!("exl3_serve");
        let h = body(serve, "pub fn run_tp_head_serve(", "\npub fn ");
        before(&h, "tp_lane_width_refusal(args, ctx.world as u32)?", "tp_attach(ctx)", "run_tp_head_serve");
        let n = body(serve, "pub fn run_tp_node_serve(", "\nfn ");
        before(&n, "tp_lane_width_refusal(args, node_world as u32)?", "tp_attach(ctx)", "run_tp_node_serve");
        let b = body(serve, "fn build_serve(", "\nfn ");
        before(&b, "let (mut width, raw_width) = resolved_max_batch(args);", "tp_lane_width_fits(tp_world, width)", "build_serve (the 16-lane cap, then the backstop)");
        before(&b, "tp_lane_width_fits(tp_world, width)", "FwdModel::load", "build_serve (the backstop, before the load)");
    }

    /// MAXBATCH1: the boot-anyway memory clamp is pinned in the source — it runs in `build_serve`
    /// BEFORE the tune table is selected (the table is keyed by the lane count that will serve),
    /// it reassigns `width` to the clamped count, and the CF-P1d exit(2) refusal survives ONLY
    /// for the nothing-fits arm. The over-cap arrival hooks ride every rx arrival site.
    #[test]
    fn the_memory_fit_clamps_lanes_before_the_tune_table_and_keeps_the_nothing_fits_refusal() {
        let src = src_text!("exl3_serve");
        let code = |s: &str| s.lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("\n");
        let body = |src: &str, from: &str, to: &str| -> String {
            let a = src.find(from).unwrap_or_else(|| panic!("{from} not found"));
            let b = src[a + from.len()..].find(to).unwrap_or_else(|| panic!("{to} not found after {from}")) + a + from.len();
            code(&src[a..b])
        };
        let b = body(src, "fn build_serve(", "\nfn ");
        let clamp = b.find("fit_max_lanes(parts, width, avail, mode, max_pos)").expect("the clamp call in build_serve");
        let tune = b.find("tune::boot(&tune::BootReq").expect("the tune table boot");
        assert!(clamp < tune, "the clamp runs before the tune table is picked (the table is keyed by the serving lane count)");
        let reassign = b.find("width = n;").expect("the clamped lane count is assigned to width");
        assert!(reassign > clamp, "the reassignment follows the clamp");
        assert!(b[clamp..].contains("std::process::exit(2)"), "the nothing-fits refusal survives (CF-P1d's teeth)");
        // The queue path F2 boots counts its over-cap arrivals: the three rx arrival sites in the
        // scheduler (top of loop, the reserve branch, the Plan::Wait arm) all carry the hook.
        assert!(src.matches("crate::metrics::over_cap_queued()").count() >= 3,
                "expected >= 3 over_cap_queued hooks, found {}", src.matches("crate::metrics::over_cap_queued()").count());
    }
