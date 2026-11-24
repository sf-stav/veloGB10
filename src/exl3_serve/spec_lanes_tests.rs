//! Unit tests (module `spec_lanes_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::{fcfs_charge, fcfs_front, shared_step_prior_ms, LaneOrder, SpecLanes};

    /// CF-FCFS: the ordering function is a pure function of (admit_seq, credit) — no GPU, no
    /// scheduler. The credit/rotation bookkeeping itself is `fcfs_charge`, exercised by
    /// `fcfs_simulation_three_lanes` below, which drives this same pair exactly as step() does.
    #[test]
    fn fcfs_order_is_pure() {
        // (slot, admit_seq, fcfs_credit)
        let q = 4;
        // Two lanes admitted in order, both fresh: the OLDEST (lowest admit_seq) is the front, not
        // the lowest slot.
        let two = [(0usize, 0u64, 0usize), (1, 1, 0)];
        assert_eq!(fcfs_front(&two, q), Some(0));
        // Slot order and admission order disagree: admission order wins.
        let swapped = [(0usize, 7u64, 0usize), (1, 3, 0)];
        assert_eq!(fcfs_front(&swapped, q), Some(1));
        // A lane whose credit reached the quantum (as fcfs_charge leaves it: credit 0 AND the
        // HIGHEST admit_seq after rotation) would only be the front if it were the sole lane.
        let one_left = [(0usize, 0u64, 0usize)];
        assert_eq!(fcfs_front(&one_left, q), Some(0));
        // The front finishes (its lane is gone from the list) -> the next-oldest runs at once.
        let finished = [(1usize, 1u64, 0)];
        assert_eq!(fcfs_front(&finished, q), Some(1));
        // A new arrival does NOT jump the queue: it has the highest admit_seq.
        let arrived = [(0usize, 0u64, 0), (1, 1, 0), (2, 2, 0)];
        assert_eq!(fcfs_front(&arrived, q), Some(0));
        // A lane with a partially used credit is still the front (its quantum is not exhausted).
        let partial = [(0usize, 0u64, 3), (1, 1, 0)];
        assert_eq!(fcfs_front(&partial, q), Some(0));
        // No capable lane at all (all MTP-off) -> the caller runs the plain batch only.
        assert_eq!(fcfs_front(&[], q), None);
        // Defensive guard on an unreachable state (charge resets credit AT the quantum): still
        // answers, never panics.
        let all_done = [(0usize, 0u64, q), (1, 1, q + 3)];
        assert_eq!(fcfs_front(&all_done, q), None);
        // Purity: the same input always gives the same front, whatever the slot order.
        let a = [(5usize, 2u64, 0), (3, 9, 0), (7, 4, q)];
        let b = [(7usize, 4u64, q), (5, 2, 0), (3, 9, 0)];
        assert_eq!(fcfs_front(&a, q), fcfs_front(&b, q));
        assert_eq!(fcfs_front(&a, q), Some(5));
        // A huge quantum never rotates (one lane runs to completion).
        assert_eq!(fcfs_front(&two, 1 << 20), Some(0));
    }

    /// CF-FCFSB: a multi-step simulation that drives the SAME bookkeeping pair step() uses —
    /// `fcfs_front` to pick, `fcfs_charge` to charge and rotate — over 3 lanes whose turns are
    /// longer than the quantum, one lane finishing mid-turn and a new arrival. Asserts the actual
    /// SERVICE ORDER (who steps at each scheduler step): a lane that used its quantum goes BEHIND
    /// the others, and the old front only gets its next turn after every waiting lane had one.
    /// (Before the fix this test fails: the rotated lane kept the lowest admit_seq and stepped
    /// forever — the order degenerated to [0,0,0,...].)
    #[test]
    fn fcfs_simulation_three_lanes() {
        let q = 4usize;
        // (slot, admit_seq, credit, tokens_left). Lanes 0 and 1 have 10-token turns, lane 2 a
        // 5-token turn; each speculative round generates 2 tokens. Quantum 4 => 2 consecutive
        // rounds per turn, then rotate.
        let mut lanes: Vec<Option<(usize, u64, usize, usize)>> =
            vec![Some((0, 0, 0, 10)), Some((1, 1, 0, 10)), Some((2, 2, 0, 5))];
        let mut admit_ctr = 3u64;
        let mut rotations = 0usize;
        let mut service: Vec<usize> = Vec::new();
        let mut lane3_admitted = false;
        for _step in 0..100 {
            // Lane 3 is admitted once lane 2 has finished (the scheduler admits into a free slot).
            if !lane3_admitted && lanes[2].is_none() {
                lanes[2] = Some((2, admit_ctr, 0, 10));
                admit_ctr += 1;
                lane3_admitted = true;
            }
            let capable: Vec<(usize, u64, usize)> = lanes
                .iter()
                .filter_map(|l| l.as_ref().map(|(s, seq, c, _)| (*s, *seq, *c)))
                .collect();
            let Some(front) = fcfs_front(&capable, q) else { break };
            let lane = lanes
                .iter_mut()
                .find_map(|l| l.as_mut().filter(|(s, ..)| *s == front))
                .unwrap();
            let delta = lane.3.min(2);
            lane.3 -= delta;
            let (mut seq, mut credit) = (lane.1, lane.2);
            if fcfs_charge(&mut seq, &mut credit, delta, q, &mut admit_ctr) {
                rotations += 1;
            }
            lane.1 = seq;
            lane.2 = credit;
            service.push(front);
            if lane.3 == 0 {
                lanes.iter_mut().find(|l| matches!(l, Some((s, ..)) if *s == front)).unwrap().take();
            }
        }
        // Hand-derived from the bookkeeping above: two rounds per quantum, then rotate; lane 2's
        // last round is a 1-token partial (credit 1, no rotation); lanes 0/1 finish with credit 2.
        // The rotated lane is served again ONLY after both other lanes had their turns; the new
        // arrival (lane 3, admitted into slot 2) runs only after lanes 0 and 1 finished entirely.
        assert_eq!(service, [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 0, 1, 2, 2, 2, 2, 2]);
        assert!(lane3_admitted);
        assert_eq!(rotations, 7); // 5 in the first two cycles + 2 from the lane-3 turn
        assert_eq!(admit_ctr, 11); // 3 initial + 7 rotation re-stamps + 1 admission stamp
    }

    #[test]
    fn lane_order_parse() {
        assert_eq!(LaneOrder::parse(None).unwrap(), LaneOrder::Rr);
        assert_eq!(LaneOrder::parse(Some("")).unwrap(), LaneOrder::Rr);
        assert_eq!(LaneOrder::parse(Some("rr")).unwrap(), LaneOrder::Rr);
        assert_eq!(LaneOrder::parse(Some("round-robin")).unwrap(), LaneOrder::Rr);
        assert_eq!(LaneOrder::parse(Some("fcfs")).unwrap(), LaneOrder::Fcfs);
        assert!(LaneOrder::parse(Some("fifo")).is_err());
        assert!(LaneOrder::parse(Some("RR")).is_err());
    }

    #[test]
    fn spec_lanes_parse() {
        assert_eq!(SpecLanes::parse(None).unwrap(), SpecLanes::Auto);
        assert_eq!(SpecLanes::parse(Some("auto")).unwrap(), SpecLanes::Auto);
        assert_eq!(SpecLanes::parse(Some("0")).unwrap(), SpecLanes::Never);
        assert_eq!(SpecLanes::parse(Some("off")).unwrap(), SpecLanes::Never);
        assert_eq!(SpecLanes::parse(Some("4")).unwrap(), SpecLanes::Over(4));
        assert!(SpecLanes::parse(Some("many")).is_err());
        assert!(SpecLanes::parse(Some("-1")).is_err());
    }

    /// The prior reproduces the 2026-10-02 TP=1 measurements within 5% (n = 4: 46.5, 8: 61.7, 16: 90 ms).
    #[test]
    fn shared_step_prior_matches_measurements() {
        for (n, ms) in [(4usize, 46.5f64), (8, 61.7), (16, 90.0)] {
            let p = shared_step_prior_ms(n);
            assert!((p - ms).abs() / ms < 0.05, "n={n}: {p} vs {ms}");
        }
    }

    /// F4: the shared-cost probe fires EARLY while the scale has never been learned (the wrong
    /// arm must not own a quarter of a ~250-round request) and at the 1-in-128 cadence afterwards,
    /// mirroring the shared arm's own stale-estimate refresh.
    #[test]
    fn the_shared_probe_fires_early_then_every_128() {
        assert_eq!(super::SHARED_PROBE_FIRST, 16);
        assert_eq!(super::SHARED_PROBE_EVERY, 128);
        let due = super::shared_probe_due;
        assert!(!due(0, true) && !due(15, true), "not before 16 rounds on the other arm");
        assert!(due(16, true) && due(17, true), "due at 16 while the scale is unlearned");
        assert!(!due(15, false) && !due(127, false), "learned scale: 1-in-128 cadence");
        assert!(due(128, false) && due(1000, false));
    }

    /// ULTRA: `--spec-arm-meter v2` under TP folds the HEAD's step time on every rank (one lockstep point per
    /// metered step, after the world-2 drain), so the ranks' arm picks stay identical; TP=1 takes no exchange.
    #[test]
    fn v2_meter_takes_the_heads_step_ms_under_tp() {
        let src = src_text!("exl3_serve");
        let f = &src[src.find("fn step_rounds_metered(&mut self").unwrap()..];
        let f = &f[..f.find("fn arm_meter_v2(&mut self").unwrap()];
        let drain = f.find("if self.tp.is_some() { self.model.tp_drain_before_exchange()?; }").expect("drain before the exchange");
        let x = f.find("Self::tp_round_with(t, &self.model, None, \"arm-meter\", 0, 0, &w, ms, 0.0, false)?.0").expect("head's ms");
        let fold = f.find("self.arm_meter_v2(arm, n, tok, ms);").expect("fold the agreed ms");
        assert!(drain < x && x < fold, "drain, then the exchange, then the fold");
        assert!(f[x..fold].contains("crate::metrics::spec_meter_tp_sync();"), "counted on /metrics");
        assert!(f.find("let meter = spec_meter_v2() && mtp_on && !self.in_prefill && slots.len() >= 2 && !settling;").map_or(false, |m| m < drain),
                "the gate is lockstep state only");
    }

    /// Issue #45: every lockstep point that can follow another agree with no collective in between (`arm-meter`,
    /// `wide-policy`, `copy-prefill-pre`, `map-ahead`) drains and runs `t.rendezvous(...)` before its agree so the
    /// first rank through never overwrites `net_agree`'s single mirrored token slot before the peer reads it.
    #[test]
    fn the_meter_points_rendezvous_before_their_agree() {
        let src = src_text!("exl3_serve");
        for (what, call) in [
            ("arm-meter", "Self::tp_round_with(t, &self.model, None, \"arm-meter\", 0, 0, &w, ms, 0.0, false)"),
            ("wide-policy", "Self::tp_round_with(t, &self.model, None, \"wide-policy\", 0, 0, &w, ms, 0.0, false)"),
            ("copy-prefill-pre", "self.tp_round(\"copy-prefill-pre\", 0, 0, &w, 0.0, 0.0, false, false)"),
            ("map-ahead", "Self::tp_round_with(t, model, None, \"map-ahead\", 0, slots.len(), &agree, 0.0, 0.0, false)"),
        ] {
            let agree = src.find(call).expect(what);
            let rdv = src.find(&format!("t.rendezvous(\"{what}\")?;")).expect(what);
            let drain = src[..rdv].rfind("self.model.tp_drain_before_exchange()?;").expect(what);
            assert!(drain < rdv && rdv < agree && agree - rdv < 500, "{what}: drain, then rendezvous, then agree");
        }
        let f = &src[src.find("fn rendezvous(&mut self").unwrap()..];
        let f = &f[..f.find("\n    }\n").unwrap()];
        assert!(f.contains("if self.world > 2 { return Ok(()); }") && f.contains("crate::net::exchange_u32s(&[tag], 1 + 4)?"),
                "world 2 only, one pairwise exchange");
    }

    /// ULTRA v2 fixes: settle steps after a busy-count change / prefill are neither metered nor probed; the speculation
    /// arm folds (and is probed) over SPEC_V2_FOLD_SPEC steps, the shared arm over SPEC_V2_FOLD; a probe's contradicting
    /// fold replaces the stored rate; close / young estimates are re-probed after SPEC_V2_REFRESH_FAST steps. v1 paths
    /// read none of it.
    #[test]
    fn v2_settles_folds_per_arm_replaces_and_reprobes_sooner() {
        use super::*;
        assert_eq!((SPEC_V2_SETTLE, SPEC_V2_FOLD, SPEC_V2_FOLD_SPEC), (4, 4, 12));
        assert_eq!((spec_v2_fold(ARM_SHARED), spec_v2_fold(ARM_SERIAL)), (4, 12));
        assert!(SPEC_V2_REFRESH_FAST < SPEC_V2_REFRESH && SPEC_V2_REPLACE > SPEC_V2_CLOSE);
        let src = src_text!("exl3_serve");
        let f = &src[src.find("fn step_rounds_metered(&mut self").unwrap()..];
        let f = &f[..f.find("fn arm_meter_v2(&mut self").unwrap()];
        let set = f.find("if spec_meter_v2() && (self.in_prefill || slots.len() != self.meter_last_n) {").expect("settle restart");
        let run = f.find("let r = self.step_rounds_rest(slots, mtp_on);").unwrap();
        let dec = f.find("if settling && !self.in_prefill { self.meter_settle -= 1; }").expect("decrement");
        assert!(set < run && run < dec, "restart before the step, decrement after it (the probe check sees the settle)");
        let g = &src[src.find("fn arm_meter_v2(&mut self").unwrap()..];
        let g = &g[..g.find("fn step_rounds_rest(&mut self").unwrap()];
        assert!(g.contains("if self.spec_win_acc[arm] >= spec_v2_fold(arm) {"));
        assert!(g.contains("let contra = fresh && probe && (rate / e.0 - 1.0).abs() > SPEC_V2_REPLACE;"));
        assert!(g.contains("e.0 = if fresh && !contra { 0.5 * e.0 + 0.5 * rate } else { rate };"));
        let h = &src[src.find("fn step_rounds_rest(&mut self").unwrap()..];
        let h = &h[..h.find("if mtp_on && probe_force.unwrap_or_else(|| self.use_shared_plain(&slots)) {").unwrap()];
        assert!(h.contains("let due = due && !(v2_idle.is_some() && self.meter_settle > 0);"));
        assert!(h.contains("let win = if v2_idle.is_some() { spec_v2_fold(idle) } else { SPEC_ARM_PROBE_WIN };"), "v1 keeps its window");
        assert!(h.contains("if close || self.spec_rate_folds[nn][idle] < 2 { SPEC_V2_REFRESH_FAST } else { SPEC_V2_REFRESH }"));
        assert!(h.contains("let every = if v2_idle.is_some() { v2_every as u32 } else { every };"), "v1 keeps its cadence");
    }

    /// S-C4XOVER: the arm-rate probe (which generalizes F4's shared-cost probe) sits BEFORE
    /// the arm pick (it can force either arm for a few steps), gated on `--lane-pack on` +
    /// the Auto policy + >=2 busy lanes with at least one MTP-capable lane, and only starts
    /// when the idle arm's measured rate has gone stale. A forced-shared window runs the
    /// shared arm's own step (not a new path) and the 128th-round refresh stays off while a
    /// probe window is in flight. Flag-off boots (G2 byte identity), `Never`/`Over` policies
    /// and single-lane steps never reach it.
    #[test]
    fn the_arm_rate_probe_is_flag_on_auto_two_lanes_and_preempts_the_pick() {
        let src = src_text!("exl3_serve");
        let pick = src.find("if mtp_on && probe_force.unwrap_or_else(|| self.use_shared_plain(&slots)) {")
            .expect("arm pick in step()");
        let probe = src.find("S-C4XOVER arm-rate probe").expect("probe block in step()");
        assert!(probe < pick, "the probe preempts the arm pick when due");
        let gate = &src[probe..pick];
        assert!(gate.contains("self.lane_pack.on() && self.spec_policy == SpecLanes::Auto && slots.len() >= 2"),
                "the gate names the flag, the policy and the two lanes");
        assert!(gate.contains("l.mtp.is_some()"),
                "the gate requires an MTP-capable lane (else the plain batch already runs)");
        // POSTURE1: --spec-arm-meter v2 also re-measures an idle arm whose rate at this busy count is old (`refresh`,
        // always false under the default v1, where the condition reads as before)
        assert!(gate.contains("if due && (!self.arm_fresh(idle, slots.len()) || refresh) {"),
                "a probe only starts when the idle arm's measured rate is stale");
        let branch = &src[pick..pick + 1600];
        assert!(branch.contains("crate::metrics::lane_pack_round(2); // arm = shared_plain"),
                "a forced-shared probe counts itself as a shared_plain round on /metrics");
        assert!(branch.contains(&["return self.", "step_plain(&slots, true);"].concat()),
                "a forced-shared probe runs the shared arm's own step");
        // POSTURE1: under --spec-arm-meter v2 the refresh never runs (the per-count probes keep the serial rate fresh)
        assert!(branch.contains("probe_force.is_none() && !spec_meter_v2() && self.spec_policy == SpecLanes::Auto && self.shared_streak % refresh_every == 0"),
                "the 128th-round refresh is suppressed while a probe window is in flight");
    }
