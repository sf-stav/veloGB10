    use super::*;
    use std::collections::HashSet;

    /// Interleaving model of the world>2 control plane (net_exchange_one over the per-sender ring slots):
    /// every rank runs its exchange program, each exchange is POST (write into the peer's slot
    /// (me, g % RING)), WAIT (own slot (peer, g % RING) carries tag g), CONSUME (copy to the stable slot).
    /// Every interleaving is explored; a POST into an unconsumed slot is a clobber, a state with no enabled
    /// step that is not final is a deadlock.
    fn explore(world: usize, calls: usize, ring: usize) -> Result<usize, String> {
        let mut prog: Vec<Vec<usize>> = vec![Vec::new(); world];
        for _ in 0..calls {
            for _round in 0..2 {
                for r in 1..world { prog[0].push(r); prog[r].push(0); }
            }
        }
        let gen = |x: usize, i: usize| 1 + prog[x][..i].iter().filter(|&&p| p == prog[x][i]).count() as u32;
        let idx = |dst: usize, src: usize, g: u32| (dst * world + src) * ring + g as usize % ring;
        #[derive(Clone, Hash, PartialEq, Eq)]
        struct St { pc: Vec<usize>, ph: Vec<u8>, slot: Vec<Option<(u32, bool)>> }
        let s0 = St { pc: vec![0; world], ph: vec![0; world], slot: vec![None; world * world * ring] };
        let mut seen: HashSet<St> = HashSet::new();
        let mut stack = vec![s0];
        while let Some(s) = stack.pop() {
            if !seen.insert(s.clone()) { continue; }
            if (0..world).all(|x| s.pc[x] == prog[x].len()) { continue; }
            let mut any = false;
            for x in 0..world {
                if s.pc[x] == prog[x].len() { continue; }
                let (p, g) = (prog[x][s.pc[x]], gen(x, s.pc[x]));
                let mut n = s.clone();
                match s.ph[x] {
                    0 => {
                        let i = idx(p, x, g);
                        if let Some((og, false)) = s.slot[i] {
                            return Err(format!("clobber: rank {x} gen {g} overwrote unconsumed gen {og} in rank {p}'s slot (ring {ring})"));
                        }
                        n.slot[i] = Some((g, false)); n.ph[x] = 1;
                    }
                    1 => {
                        if !matches!(s.slot[idx(x, p, g)], Some((sg, false)) if sg == g) { continue; }
                        n.ph[x] = 2;
                    }
                    _ => {
                        n.slot[idx(x, p, g)] = Some((g, true)); n.ph[x] = 0; n.pc[x] += 1;
                    }
                }
                any = true;
                stack.push(n);
            }
            if !any { return Err(format!("deadlock at pcs {:?} phases {:?}", s.pc, s.ph)); }
        }
        Ok(seen.len())
    }

    fn header_ring() -> usize {
        let h = include_str!("../../native/tp_doorbell.h");
        let l = h.lines().find(|l| l.contains("#define TP_CTRL_RING")).expect("TP_CTRL_RING in tp_doorbell.h");
        l.split_whitespace().nth(2).unwrap().parse().unwrap()
    }

    // World 4 is the shipping topology (every config here runs in milliseconds). World 8 has 1.1M states and takes
    // ~17 s in release, so it lives in the #[ignore]d sweep below (`cargo test --release --lib -- --ignored hub_control_ring`).
    #[test]
    fn hub_control_ring_is_clobber_free_and_live() {
        let ring = header_ring();
        assert!(ring >= 2, "a depth-1 ring is the R10 clobber");
        for r in [ring, 2] {
            let n = explore(4, 3, r).unwrap_or_else(|e| panic!("world 4 ring {r}: {e}"));
            assert!(n > 1000, "world 4 ring {r}: only {n} states explored");
        }
    }

    #[test]
    #[ignore = "world 8, 2 calls: 1.1M states, ~17 s in release"]
    fn hub_control_ring_world8_full_sweep() {
        let ring = header_ring();
        let n = explore(8, 2, ring).unwrap_or_else(|e| panic!("world 8: {e}"));
        assert!(n > 1_000_000, "world 8: only {n} states explored");
    }

    #[test]
    fn hub_control_ring_depth_one_clobbers() {
        for calls in [1usize, 3] {
            let e = explore(4, calls, 1).unwrap_err();
            assert!(e.starts_with("clobber"), "calls {calls}: {e}");
        }
    }

    fn payload(rank: usize, op: usize, len: usize) -> Vec<u32> {
        (0..len).map(|i| ((op as u32) << 24) | ((rank as u32) << 16) | i as u32).collect()
    }

    /// The REAL `hub_all` (not a re-implementation) over the mock: four ranks run a mixed-size op sequence shaped
    /// like a served round (go / pre-verify / round frames, an ident-sized frame, an agree-sized frame) and every
    /// rank must return every rank's exact payload, with no ring clobber.
    #[test]
    fn hub_all_mixed_sizes_four_ranks() {
        use hub_mock::*;
        let net = Net::new(4, true);
        let sizes: Vec<(usize, usize)> = vec![(4, 6), (5, 8), (12, 14), (8, 10), (40, 44), (4, 6), (5, 8), (12, 14), (4, 6), (12, 14)];
        let outs = run_ranks(&net, None, |rank, hub| {
            sizes.iter().enumerate().map(|(op, &(len, wire))| {
                hub_all(hub, &payload(rank, op, len), wire).expect("a healthy hub op")
            }).collect::<Vec<_>>()
        });
        for (rank, per_rank) in outs.iter().enumerate() {
            assert_eq!(per_rank.len(), sizes.len(), "rank {rank} completed every op");
            for (op, all) in per_rank.iter().enumerate() {
                let want: Vec<Vec<u32>> = (0..4).map(|r| payload(r, op, sizes[op].0)).collect();
                assert_eq!(all, &want, "rank {rank} op {op}: every rank must hold every rank's payload");
            }
        }
        assert!(net.clobbers().is_empty(), "no ring clobber: {:?}", net.clobbers());
        assert!((0..4).all(|r| net.abort_code(r) == 0));
    }

    /// Mixed wire sizes share ring slots. A stale PAYLOAD word pair an older, larger frame left at a later, smaller
    /// frame's tail offset can equal that frame's expected tag `(g << 8) | sender`; without the retirement of the
    /// consumed frame (TP-4D net_shim.c) the head accepts the stale frame before the node has posted. Model of the
    /// hazard AND of the fix — the C code itself is only exercised on hardware.
    #[test]
    fn stale_payload_can_fake_a_tag_until_the_frame_is_retired() {
        use hub_mock::*;
        use std::time::Duration;
        // node 1's op-0 payload plants [(5 << 8) | 1, 0] at words 2..4 = the tag head expects from node 1 at its
        // generation 5 (op 2 round 1); op 2 uses a 4-word frame whose tail is exactly words 2..4.
        let plant = |rank: usize, op: usize| -> Vec<u32> {
            let mut p = payload(rank, op, 8);
            if rank == 1 && op == 0 { p[2] = (5 << 8) | 1; p[3] = 0; }
            p
        };
        let run = |clear_frame: bool| {
            let net = Net::new(4, clear_frame);
            let res = run_ranks(&net, Some(Duration::from_secs(5)), |rank, hub| {
                let r0 = hub_all(hub, &plant(rank, 0), 14);
                let r1 = hub_all(hub, &payload(rank, 1, 4), 6);
                if rank == 1 { std::thread::sleep(Duration::from_millis(150)); } // node 1 is late into op 2
                let r2 = hub_all(hub, &payload(rank, 2, 2), 4);
                (r0.is_ok() && r1.is_ok(), r2)
            });
            res
        };
        let good = run(true);
        let want: Vec<Vec<u32>> = (0..4).map(|r| payload(r, 2, 2)).collect();
        for (rank, (early_ok, r2)) in good.iter().enumerate() {
            assert!(*early_ok, "rank {rank}: ops 0 and 1 are healthy");
            assert_eq!(r2.as_ref().expect("with the frame retired op 2 completes"), &want, "rank {rank}");
        }
        let bad = run(false);
        let broke = bad.iter().any(|(_, r2)| match r2 { Err(_) => true, Ok(all) => all != &want });
        assert!(broke, "without retiring the consumed frame the stale payload must be accepted as node 1's op-2 frame \
                         (the model has to demonstrate the hazard, else the fix is unproven)");
    }

    fn go_payload(rank: usize, op: u32) -> Vec<u32> {
        crate::tp_lockstep::go_frame(rank == 0, 0xC0_0000 + op, 100 + op as u64, 3).to_vec()
    }

    /// Kill-a-node model: rank 3's process is gone before op 1. The head must NOT return early from its gather (nodes
    /// 1 and 2 are already in round 2 — the agree_ext rule), every live rank must come back promptly, and the ranks
    /// that did get a round-2 frame see the dead rank's block as zeros — so the lockstep verdict aborts them too.
    #[test]
    fn dead_node_fails_the_head_and_zeroes_its_block_for_the_rest() {
        use crate::tp_lockstep::{go_verdict, GO_WIRE};
        use hub_mock::*;
        let net = Net::new(4, true);
        let t0 = std::time::Instant::now();
        let res = run_ranks(&net, None, |rank, hub| {
            let first = hub_all(hub, &go_payload(rank, 0), GO_WIRE).expect("op 0 is healthy");
            if rank == 3 { return (first, None); } // the process dies between op 0 and op 1
            (first, Some(hub_all(hub, &go_payload(rank, 1), GO_WIRE)))
        });
        assert!(t0.elapsed() < std::time::Duration::from_secs(5), "no live rank hangs");
        for (rank, (first, _)) in res.iter().enumerate() {
            assert_eq!(go_verdict(first, 4), Ok((100, 3)), "rank {rank}: op 0 reached a healthy verdict before the death");
        }
        let e = format!("{:#}", res[0].1.as_ref().unwrap().as_ref().unwrap_err());
        assert!(e.contains("rank 3") && e.contains("rc=-3"), "the head names the dead rank: {e}");
        for n in [1usize, 2] {
            let all = res[n].1.as_ref().unwrap().as_ref().unwrap_or_else(|e| panic!("node {n} gets the head's round-2 frame: {e:#}"));
            assert_eq!(all[n], go_payload(n, 1));
            assert_eq!(all[3], vec![0; 4], "the dead rank's block is zero, never a stale copy of its previous frame");
            let v = go_verdict(all, 4).unwrap_err();
            assert!(v.contains("rank 3 tag"), "the verdict over a zeroed block aborts node {n} too and names rank 3: {v}");
        }
        assert_eq!(net.abort_code(0), 10, "the head records abort code 10 (peer dead)");
    }

    /// Hung-but-alive model (audit C5): rank 2 never reaches the lockstep point (its thread is alive, so the
    /// dead-peer probe would wait for ever) — the per-wait deadline trips instead, the head aborts with code 12 and
    /// EVERY rank ends in an error rather than parked or in a healthy verdict.
    #[test]
    fn hung_alive_node_trips_the_lockstep_deadline_on_every_rank() {
        use crate::tp_lockstep::{go_verdict, GO_WIRE};
        use hub_mock::*;
        use std::time::Duration;
        let net = Net::new(4, true);
        let res = run_ranks(&net, Some(Duration::from_millis(120)), |rank, hub| {
            if rank == 2 { std::thread::sleep(Duration::from_millis(600)); }
            hub_all(hub, &go_payload(rank, 0), GO_WIRE)
        });
        let e = format!("{:#}", res[0].as_ref().unwrap_err());
        assert!(e.contains("rc=-4") && e.contains("deadline"), "the head reports the deadline: {e}");
        assert_eq!(net.abort_code(0), 12);
        for (rank, r) in res.iter().enumerate() {
            match r {
                Err(_) => {}
                // a node that still got a round-2 frame holds a zeroed block for the hung rank: the verdict aborts it
                Ok(all) => assert!(go_verdict(all, 4).is_err(), "rank {rank} must not reach a healthy verdict"),
            }
        }
        assert!(res.iter().filter(|r| r.is_err()).count() >= 1);
    }

    #[test]
    fn hub_frames_round_trip() {
        let (world, wire) = (4usize, 7usize);
        let all: Vec<Option<Vec<u32>>> = (0..world).map(|k| Some((0..3).map(|i| (k * 100 + i) as u32).collect())).collect();
        let mut frame = vec![0xFFFF_FFFFu32; world * wire];
        hub_pack(&all, wire, &mut frame);
        let out = hub_unpack(&frame, wire, 3, world);
        for k in 0..world { assert_eq!(Some(&out[k]), all[k].as_ref()); }
        assert!(frame[3..wire].iter().all(|&x| x == 0), "headroom of block 0 must be zero");
    }
