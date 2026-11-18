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
