//! Unit tests (module `spill_agree_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::*;
    use crate::net::hub_mock::{run_ranks, Net};

    #[test]
    fn ranks_agree_on_a_capture_and_drop_it_together_when_one_write_failed() {
        for world in [3usize, 4] {
            for failed in [None, Some(0usize), Some(world - 1)] {
                let net = Net::new(world, true);
                let outs = run_ranks(&net, None, |rank, hub| {
                    let mut t = TpSync::new(rank as i32, world as i32);
                    (0..3).map(|_| t.spill_agree_hub(hub, Some(rank) != failed).map_err(|e| format!("{e:#}")))
                          .collect::<Vec<_>>()
                });
                for (rank, v) in outs.iter().enumerate() {
                    for r in v {
                        assert_eq!(r, &Ok(failed.is_none()), "world {world}, failed write on {failed:?}: rank {rank}");
                    }
                }
                assert!((0..world).all(|r| net.abort_code(r) == 0), "no rank aborts its link on a healthy vote");
                assert!(net.clobbers().is_empty(), "{:?}", net.clobbers());
            }
        }
    }

    /// veloGB10 #44 (reported by @AbsolomBane): under TP a rank whose spill capture fails must still vote, or the peer that
    /// reached the SpillFlag exchange waits for a frame that never comes (TP=2 has no read deadline there).
    #[test]
    fn a_failed_capture_still_casts_its_spill_vote() {
        let src = src_text!("exl3_serve");
        let at = src.find("pub(super) fn spill_capture_slot(").expect("spill_capture_slot");
        let body = &src[at..at + src[at..].find("\n    pub(").unwrap_or(src.len() - at)];
        let fail = body.find("kv spill: capture failed slot=").expect("the capture-failure arm");
        let ret = fail + body[fail..].find("return Ok(());").expect("the arm returns");
        assert!(body[fail..ret].contains("spill_agree(false)"), "the capture-failure arm must vote before it returns");
    }


    /// Evidence for the v0.7.10 capture-failure arm (`pool.rs:439-442` returns before `spill_agree`), on the hub
    /// mock (TP=3). Three evictions, each one spill vote. On the capture of the FIRST, rank 1's D2H capture fails and,
    /// as in v0.7.10, it returns without voting; it stays alive and votes on the next two. A spill frame is only
    /// [tag, ok] (no sequence), so every later vote of rank 1 pairs with the PREVIOUS vote of the others:
    ///   - vote 1: ranks 0 and 2 read rank 1's vote-2 frame and keep key 1, which rank 1 dropped (silent divergence);
    ///   - vote 2: likewise, one behind;
    ///   - vote 3: rank 1 has nothing left to send, so ranks 0 and 2 stall (here the mock's lockstep deadline ends it;
    ///     at TP=2 the SpillFlag read has none).
    /// With the fix (rank 1 votes `false` before returning) all three votes pair, key 1 is dropped on every rank.
    #[test]
    fn a_skipped_spill_vote_mispairs_the_next_votes_and_stalls_the_peers() {
        use std::time::Duration;
        let (world, failed) = (3usize, 1usize);
        for fixed in [false, true] {
            let net = Net::new(world, true);
            let outs = run_ranks(&net, Some(Duration::from_millis(300)), |rank, hub| {
                let mut t = TpSync::new(rank as i32, world as i32);
                let mut votes = Vec::new();
                for key in 1..=3 {
                    if rank == failed && key == 1 {
                        if !fixed { continue; } // v0.7.10: capture failed -> `return Ok(())`, no vote
                        votes.push(t.spill_agree_hub(hub, false).map_err(|e| format!("{e:#}")));
                        continue;
                    }
                    votes.push(t.spill_agree_hub(hub, true).map_err(|e| format!("{e:#}")));
                }
                if rank == failed { std::thread::sleep(Duration::from_millis(900)); } // still serving, not dead
                votes
            });
            if fixed {
                for (rank, v) in outs.iter().enumerate() {
                    assert_eq!(v, &vec![Ok(false), Ok(true), Ok(true)], "fixed, rank {rank}: key 1 dropped everywhere");
                }
                assert!((0..world).all(|r| net.abort_code(r) == 0), "fixed: no rank aborts");
            } else {
                // rank 1 voted twice, and both of its votes were answered: it sees nothing wrong
                assert_eq!(outs[failed], vec![Ok(true), Ok(true)], "v0.7.10: the failed rank");
                for rank in [0usize, 2] {
                    let v = &outs[rank];
                    assert_eq!(v.len(), 3);
                    // vote 1 is answered with rank 1's vote-2 frame: key 1 is KEPT here, dropped on rank 1
                    assert_eq!(v[0], Ok(true), "v0.7.10, rank {rank}: key 1 kept although rank 1 dropped it");
                    assert_eq!(v[1], Ok(true), "v0.7.10, rank {rank}: vote 2, one behind");
                    assert!(v[2].is_err(), "v0.7.10, rank {rank}: vote 3 waits for a frame that never comes: {:?}", v[2]);
                    // 12 = its own lockstep deadline; 10 = the peer it waited on already gave up (the mock marks it dead);
                    // or the agree itself sees the missing frame (scheduling decides which comes first, so all count)
                    let seen = v[2].as_ref().err().map_or(false, |e| e.contains("spill agree"));
                    assert!(matches!(net.abort_code(rank), 10 | 12) || seen,
                            "v0.7.10, rank {rank}: stalled, code {} err {:?}", net.abort_code(rank), v[2]);
                }
            }
        }
    }

    /// veloGB10 #52: pre-D2H budget agreement under TP >= 2 keeps all ranks in lockstep across
    /// full-with-WP16, base-without-WP16, and pre-D2H skip when one rank has tighter MemAvailable.
    #[test]
    fn ranks_agree_on_pre_d2h_full_base_or_skip_mode() {
        #[derive(Debug, PartialEq, Eq)]
        enum Mode { Full, Base, Skip }
        let (base, full) = (2_000_000_000u64, 3_400_000_000u64);
        let scenarios: &[(&str, [u64; 3], Mode)] = &[
            ("all_fit_full", [5_000_000_000, 4_000_000_000, 6_000_000_000], Mode::Full),
            ("one_rank_fits_only_base", [5_000_000_000, 2_500_000_000, 6_000_000_000], Mode::Base),
            ("one_rank_fits_neither", [5_000_000_000, 1_000_000_000, 6_000_000_000], Mode::Skip),
        ];
        for (name, budgets, expect) in scenarios {
            let net = Net::new(3, true);
            let outs = run_ranks(&net, None, |rank, hub| {
                let mut t = TpSync::new(rank as i32, 3);
                let b = budgets[rank];
                let all_full = t.spill_agree_hub(hub, full <= b).unwrap();
                if all_full {
                    Mode::Full
                } else if t.spill_agree_hub(hub, base <= b).unwrap() {
                    Mode::Base
                } else {
                    Mode::Skip
                }
            });
            for (rank, mode) in outs.iter().enumerate() {
                assert_eq!(mode, expect, "{name}: rank {rank}");
            }
            assert!((0..3).all(|r| net.abort_code(r) == 0), "{name}: no aborts");
            assert!(net.clobbers().is_empty(), "{name}: {:?}", net.clobbers());
        }
    }
