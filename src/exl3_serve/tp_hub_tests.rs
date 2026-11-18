//! Unit tests (module `tp_hub_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::*;
    use crate::net::hub_mock::{run_ranks, Net};
    use std::time::Duration;

    const W: usize = 4;

    fn sync(rank: usize) -> TpSync { TpSync::new(rank as i32, W as i32) }

    #[test]
    fn step_go_hub_adopts_head_words_and_advances_counters_in_lockstep() {
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            let mut got = Vec::new();
            for round in 0..5u64 {
                let (events, local_step) = if r == 0 { ((round % 3) as usize, 100 + round) } else { (0, 100 + round) };
                got.push(t.step_go_hub(hub, local_step, events).expect("a healthy step-go"));
            }
            (got, t.step)
        });
        for (r, (got, step)) in out.iter().enumerate() {
            assert_eq!(*step, 5, "rank {r}: one lockstep counter tick per step-go");
            let want: Vec<(u64, usize)> = (0..5u64).map(|i| (100 + i, (i % 3) as usize)).collect();
            assert_eq!(got, &want, "rank {r} adopts the head's (step, events)");
        }
        assert!(net.clobbers().is_empty(), "control ring clobbers: {:?}", net.clobbers());
    }

    #[test]
    fn a_rank_one_lockstep_point_ahead_fails_every_rank_and_names_it() {
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            if r == 2 { t.step += 1; }
            t.step_go_hub(hub, 7, 0).map_err(|e| format!("{e:#}"))
        });
        for (r, o) in out.iter().enumerate() {
            let e = o.as_ref().expect_err(&format!("rank {r} must fail: the ranks are at different lockstep points"));
            assert!(e.contains("step-go") && e.contains("FAILED") && e.contains("rank 2 counter field"), "rank {r}: {e}");
        }
    }

    #[test]
    fn pre_verify_hub_compares_width_drafts_and_launches_on_every_rank() {
        let net = Net::new(W, true);
        let ok = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            let res = t.pre_verify_hub(hub, 6, &[1, 2, 3, 4, 5], 7);
            (res.is_ok(), t.pre_agrees)
        });
        assert!(ok.iter().all(|&(good, n)| good && n == 1), "{ok:?}");
        for (what, r_bad) in [("drafts", 3usize), ("width", 1), ("launched", 2)] {
            let net = Net::new(W, true);
            let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
                let mut t = sync(r);
                let (w, d, l) = match (what, r == r_bad) {
                    ("drafts", true) => (6, vec![1, 2, 3, 4, 6], 7),
                    ("width", true) => (5, vec![1, 2, 3, 4, 5], 7),
                    ("launched", true) => (6, vec![1, 2, 3, 4, 5], 8),
                    _ => (6, vec![1, 2, 3, 4, 5], 7),
                };
                t.pre_verify_hub(hub, w, &d, l).map_err(|e| format!("{e:#}"))
            });
            for (r, o) in out.iter().enumerate() {
                let e = o.as_ref().expect_err(&format!("{what}: rank {r} must fail on a divergent rank {r_bad}"));
                assert!(e.contains("pre-verify") && e.contains(&format!("rank {r_bad}")), "{what} rank {r}: {e}");
            }
        }
        // S-B9-CF-AGREE16: the widths a packed verify would reach (today's pre-verify sees at most 7 drafts).
        // One rank drafted for a 15-row verify, the others for a 16-row one (and 16 vs 17, 15 vs 17): the SAME
        // drafts hash on every rank, so the width word alone must fail the op, on every rank, before the verify
        // launches. (This frame always carried the whole width: the case pins it, it was not red on master.)
        let drafts: Vec<i32> = (0..16).collect();
        for (w_most, w_bad, r_bad) in [(16usize, 15usize, 1usize), (15, 16, 2), (17, 16, 3), (16, 17, 1), (17, 15, 2)] {
            let net = Net::new(W, true);
            let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
                let mut t = sync(r);
                t.pre_verify_hub(hub, if r == r_bad { w_bad } else { w_most }, &drafts, 3).map_err(|e| format!("{e:#}"))
            });
            for (r, o) in out.iter().enumerate() {
                let e = o.as_ref().expect_err(&format!("width {w_most} vs {w_bad}: rank {r} must fail"));
                assert!(e.contains("pre-verify") && e.contains(&format!("rank {r_bad} width")), "{w_most} vs {w_bad}, rank {r}: {e}");
            }
        }
        for w in [15usize, 16, 17] {
            let net = Net::new(W, true);
            let ok = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| sync(r).pre_verify_hub(hub, w, &drafts, 3).is_ok());
            assert!(ok.iter().all(|&g| g), "every rank at width {w} must agree: {ok:?}");
        }
    }

    /// S-B9-CF-PACK5, the fault drill at unit scale (fake hub transport, every rank a thread): the
    /// PACKED round's pre-verify frame. The words are `lane_pack::pre_words` of a two-lane pack; R is
    /// the explicit row total. One rank disagreeing about the pack (a different draft in either lane,
    /// the same words split differently, another slot, another position) or about R must fail EVERY
    /// rank at this exchange, before any packed verify could launch, with the combined-hash message
    /// naming the offending rank; identical packs pass. (The world-2 path is the same words through
    /// `exchange_u32s` + `tp_agree_eq`, which needs the real link: the live drill covers it.)
    #[test]
    fn the_pack_pre_verify_catches_every_disagreement_about_the_pack_before_the_verify() {
        use crate::lane_pack::{pack_rows_total, pre_words, PreSeg};
        let (d0, d1) = ([11, 12, 13], [21, 22]);
        let mk = |slot1: usize, pos1: usize, e1: [i32; 2]| -> (usize, Vec<i32>) {
            let segs = [PreSeg::greedy(1, 100, 7, &d0), PreSeg::greedy(slot1, pos1, 8, &e1)];
            (pack_rows_total(&segs), pre_words(&segs))
        };
        let base = mk(2, 200, d1);
        let net = Net::new(W, true);
        let ok = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            let res = t.pre_verify_hub(hub, base.0, &base.1, 5);
            (res.is_ok(), t.pre_agrees)
        });
        assert!(ok.iter().all(|&(good, n)| good && n == 1), "{ok:?}");
        assert_eq!(base.0, 7);
        for (what, r_bad, other) in [("a draft of lane 1", 3usize, mk(2, 200, [21, 23])), ("lane 1's slot", 1, mk(3, 200, d1)),
                                     ("lane 1's position", 2, mk(2, 201, d1))] {
            let net = Net::new(W, true);
            let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
                let mut t = sync(r);
                let (rows, words) = if r == r_bad { (other.0, &other.1) } else { (base.0, &base.1) };
                t.pre_verify_hub(hub, rows, words, 5).map_err(|e| format!("{e:#}"))
            });
            for (r, o) in out.iter().enumerate() {
                let e = o.as_ref().expect_err(&format!("{what}: rank {r} must fail on a divergent rank {r_bad}"));
                assert!(e.contains("pre-verify") && e.contains(&format!("rank {r_bad}")) && e.contains("link aborted before the verify"),
                        "{what} rank {r}: {e}");
            }
        }
        // the row total alone: SAME words, one rank believes R is one row wider (a 15- vs 16-row pack)
        for (r_most, r_bad, who) in [(15usize, 16usize, 2usize), (16, 15, 1)] {
            let net = Net::new(W, true);
            let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
                let mut t = sync(r);
                t.pre_verify_hub(hub, if r == who { r_bad } else { r_most }, &base.1, 5).map_err(|e| format!("{e:#}"))
            });
            for (r, o) in out.iter().enumerate() {
                let e = o.as_ref().expect_err(&format!("R {r_most} vs {r_bad}: rank {r} must fail"));
                assert!(e.contains(&format!("rank {who} width")), "R {r_most} vs {r_bad}, rank {r}: {e}");
            }
        }
    }

    /// S-B9-CF-AGREE16: the post-verify round of the world > 2 lockstep. `words` (emitted ids, drafts,
    /// confidences) are the same on every rank here, so the hash is equal and ONLY the width (then only the
    /// accept count) tells the ranks apart: every unequal pair in 0..=17 must fail every rank and name the
    /// offender; every equal pair must pass.
    #[test]
    fn round_hub_tells_every_width_and_accept_count_apart_0_to_17() {
        let words = [11u32, 22, 33];
        let run = |a_most: usize, a_bad: usize, m_most: usize, m_bad: usize| -> Vec<std::result::Result<(), String>> {
            let net = Net::new(W, true);
            run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
                let mut t = sync(r);
                let (a, m) = if r == 1 { (a_bad, m_bad) } else { (a_most, m_most) };
                t.round_hub(hub, "verify", a, m, &words, 1.0, 1.0, false, 0, |_| Ok(vec![])).map(|_| ()).map_err(|e| format!("{e:#}"))
            })
        };
        let mut silent: Vec<String> = Vec::new();
        for m_most in 0..=17usize {
            for m_bad in 0..=17usize {
                let out = run(3, 3, m_most, m_bad);
                if m_most == m_bad {
                    assert!(out.iter().all(|o| o.is_ok()), "width {m_most} on every rank must agree: {out:?}");
                } else if out.iter().any(|o| o.is_ok()) {
                    silent.push(format!("width {m_most} vs {m_bad}"));
                } else {
                    for (r, o) in out.iter().enumerate() {
                        let e = o.as_ref().unwrap_err();
                        assert!(e.contains("rank 1 verify width"), "width {m_most} vs {m_bad}, rank {r}: {e}");
                    }
                }
            }
        }
        for a_most in 0..=17usize {
            for a_bad in 0..=17usize {
                let out = run(a_most, a_bad, 17, 17);
                if a_most == a_bad {
                    assert!(out.iter().all(|o| o.is_ok()), "accept {a_most} on every rank must agree: {out:?}");
                } else if out.iter().any(|o| o.is_ok()) {
                    silent.push(format!("accept {a_most} vs {a_bad}"));
                } else {
                    for (r, o) in out.iter().enumerate() {
                        let e = o.as_ref().unwrap_err();
                        assert!(e.contains("rank 1 accept count"), "accept {a_most} vs {a_bad}, rank {r}: {e}");
                    }
                }
            }
        }
        assert!(silent.is_empty(), "{} unequal pairs passed the round in silence: {silent:?}", silent.len());
        // beyond a byte: the frame words are whole u32s, so 255 vs 256 accepted ids are told apart as well
        assert!(run(255, 256, 17, 17).iter().all(|o| o.is_err()), "accept 255 vs 256 must fail the round");
        assert!(run(3, 3, 31, 32).iter().all(|o| o.is_err()), "width 31 vs 32 must fail the round");
    }

    /// S-B9-CF-AGREE16, the fault drill at unit scale: every rank runs a 16-row round, ONE rank carries
    /// `--tp-width-drill 16` (the flag's name since AGREE16B) and therefore REPORTS 15 - rank. The guard
    /// must fail the round on every rank.
    #[test]
    fn the_width_drill_on_one_rank_fails_a_16_row_round_on_every_rank() {
        for drill_rank in [0usize, 2] {
            let net = Net::new(W, true);
            let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
                let mut t = sync(r);
                if r == drill_rank { t.drill_width = Some(16); }
                // a 5-row round first: the drill leaves every width but its own alone
                let first = t.round_hub(hub, "verify", 2, 5, &[1, 2], 1.0, 1.0, false, 0, |_| Ok(vec![])).map(|_| ()).map_err(|e| format!("{e:#}"));
                let second = t.round_hub(hub, "verify", 2, 16, &[1, 2], 1.0, 1.0, false, 0, |_| Ok(vec![])).map(|_| ()).map_err(|e| format!("{e:#}"));
                (first, second, t.drill_hits)
            });
            for (r, (first, second, hits)) in out.iter().enumerate() {
                assert!(first.is_ok(), "rank {r}: the 5-row round is not the drill's width: {first:?}");
                let e = second.as_ref().expect_err(&format!("rank {r}: rank {drill_rank} reported 15 for a 16-row round — the round must fail"));
                assert!(e.contains("verify width"), "rank {r}: {e}");
                assert_eq!(*hits, (r == drill_rank) as u64, "rank {r}");
            }
        }
    }

    #[test]
    fn round_hub_ships_head_timing_and_reports_ident_digests() {
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            t.ident = true;
            let words = [3u32, 1, 4, 1, 5];
            // rank 3's second digest differs: counted, but it is diagnostic (the round itself stays OK)
            let res = t.round_hub(hub, "verify", 3, 5, &words, if r == 0 { 12.5 } else { 99.0 }, 20.0, true, 1000 + r as u64,
                                  |m| { assert_eq!(m, 5); Ok(vec![0xAA, if r == 3 { 0xBC } else { 0xBB }, 0xCC]) });
            (res.expect("a healthy round"), t.agrees, t.ident_bufs, t.ident_bad, t.step)
        });
        for (r, (res, agrees, bufs, bad, step)) in out.iter().enumerate() {
            assert_eq!(*res, (12.5, 20.0, true), "rank {r} gets the head's timing, not its own");
            assert_eq!((*agrees, *bufs, *bad, *step), (1, 3, 1, 1), "rank {r}");
        }
    }

    #[test]
    fn round_hub_fails_every_rank_on_a_divergent_accept_count() {
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            t.round_hub(hub, "verify", if r == 1 { 2 } else { 3 }, 5, &[1, 2, 3], 1.0, 1.0, false, 0, |_| Ok(vec![]))
                .map(|_| ()).map_err(|e| format!("{e:#}"))
        });
        for (r, o) in out.iter().enumerate() {
            let e = o.as_ref().expect_err(&format!("rank {r} must fail"));
            assert!(e.contains("rank 1 accept count"), "rank {r}: {e}");
        }
    }

    #[test]
    fn a_node_that_dies_fails_every_live_rank_and_a_hung_one_trips_the_deadline() {
        // dead node: rank 2 "crashes" before its step-go (its thread ends: the mock's dead-peer probe fires)
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            if r == 2 { return Err("node crashed".to_string()); }
            sync(r).step_go_hub(hub, 9, 1).map(|_| ()).map_err(|e| format!("{e:#}"))
        });
        for (r, o) in out.iter().enumerate() {
            let e = o.as_ref().expect_err(&format!("rank {r}: a dead rank 2 must fail the op"));
            if r != 2 { assert!(e.contains("step-go") && e.contains("FAILED"), "rank {r}: {e}"); }
        }
        assert!(net.abort_code(0) == 10 || net.abort_code(0) == 0, "the head saw the dead peer (code 10) or a failed verdict");
        // hung node: alive but never posts; the head's deadline fires (code 12) and every other rank fails too
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_millis(300)), |r, hub| {
            if r == 3 { std::thread::sleep(Duration::from_millis(1500)); return Err("hung".to_string()); }
            sync(r).step_go_hub(hub, 9, 1).map(|_| ()).map_err(|e| format!("{e:#}"))
        });
        for r in 0..3 { assert!(out[r].is_err(), "rank {r} must not stay parked behind a hung rank 3"); }
        assert_eq!(net.abort_code(0), 12, "the head's wait hit the lockstep deadline");
    }

    #[test]
    fn checkpoint_allocation_outcome_is_agreed_so_four_stores_stay_identical() {
        use crate::exl3_forward::CkptStore;
        // rank 2's third allocation fails (its device memory is tighter); the others would succeed.
        let local = |r: usize, n_before: usize| -> Result<u32> {
            if r == 2 && n_before >= 2 { anyhow::bail!("CUDA_ERROR_OUT_OF_MEMORY (model)") } else { Ok(n_before as u32 + 1) }
        };
        // WITHOUT the agree (today's world-2 behaviour): the stores diverge — the C2 bug, reproduced
        let plain: Vec<(usize, usize)> = (0..W).map(|r| {
            let mut st: CkptStore<u32> = CkptStore::new(1, 8);
            st.begin_aligned(0, 0, 2048);
            for i in 0..5 {
                let n = st.n_alloc;
                if let Some(b) = st.acquire(|| local(r, n)) { st.insert(0, 2048 * (i + 1), b); }
            }
            (st.n_alloc, st.live())
        }).collect();
        assert!(plain.iter().any(|&p| p != plain[0]), "without the agree rank 2 freezes at 2 while the others grow: {plain:?}");
        // WITH the agree: every rank's store ends identical
        let net = Net::new(W, true);
        let agreed = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            let mut st: CkptStore<u32> = CkptStore::new(1, 8);
            st.begin_aligned(0, 0, 2048);
            let mut hub_err = None;
            for i in 0..5 {
                let n = st.n_alloc;
                let got = st.acquire(|| { let l = local(r, n); t.alloc_outcome(hub, n, l, &mut hub_err) });
                assert!(hub_err.is_none(), "rank {r}: a healthy hub");
                if let Some(b) = got { st.insert(0, 2048 * (i + 1), b); }
            }
            (st.n_alloc, st.live(), st.positions(0))
        });
        assert!(agreed.iter().all(|a| *a == agreed[0]), "the four stores must be identical: {agreed:?}");
        assert_eq!(agreed[0].0, 2, "the cap froze at the two buffers every rank could allocate");
        assert!(net.clobbers().is_empty());
    }

    #[test]
    fn a_failed_alloc_agree_is_reported_not_swallowed_as_an_allocation_failure() {
        use crate::exl3_forward::CkptStore;
        // rank 3 dies before the allocation agree: the live ranks' `acquire` would freeze the cap on the Err, so the
        // transport failure must come back through `hub_err`
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            if r == 3 { return None; }
            let mut t = sync(r);
            let mut st: CkptStore<u32> = CkptStore::new(1, 8);
            let mut hub_err = None;
            let _ = st.acquire(|| t.alloc_outcome(hub, 0, Ok(1u32), &mut hub_err));
            hub_err.map(|e| format!("{e:#}"))
        });
        for r in 0..3 {
            let e = out[r].as_ref().unwrap_or_else(|| panic!("rank {r}: the agree failure must surface in hub_err"));
            assert!(e.contains("wp16 alloc"), "rank {r}: {e}");
        }
    }

    #[test]
    fn boot_hash_verdict_names_every_differing_rank() {
        let ok: Vec<Vec<u32>> = (0..W).map(|_| vec![BOOT_AGREE_MAGIC, 0x1234]).collect();
        assert_eq!(boot_hash_verdict(&ok, 1), Ok(()));
        let mut bad = ok.clone();
        bad[2][1] = 0x9999;
        bad[3][0] = 0x6000_0001;
        let e = boot_hash_verdict(&bad, 1).unwrap_err();
        assert!(e.contains("rank 2 hash 00009999 != head 00001234") && e.contains("rank 3 sent frame class") && e.contains("this is rank 1"), "{e}");
        assert!(!e.contains("rank 1 hash") && !e.contains("rank 0"), "{e}");
    }

    #[test]
    fn tpspec_fold_requires_every_rank_ok_with_equal_agree_counts() {
        let good: Vec<Vec<u32>> = (0..W).map(|_| vec![1, 77]).collect();
        let (ok, line) = tpspec_fold(&good, 2, 77);
        assert!(ok && line.contains("rank 2 (this) OK (agrees 77)"), "{line}");
        let mut one_fail = good.clone();
        one_fail[3][0] = 0;
        let (ok, line) = tpspec_fold(&one_fail, 0, 77);
        assert!(!ok && line.contains("rank 3 FAIL"), "{line}");
        let mut skew = good.clone();
        skew[1][1] = 76;
        let (ok, _) = tpspec_fold(&skew, 0, 77);
        assert!(!ok, "a rank that crossed a different number of lockstep points must fail the fold");
    }
