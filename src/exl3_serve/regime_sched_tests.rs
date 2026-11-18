//! Unit tests (module `regime_sched_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::*;

    const LIM: usize = 2051; // Flash-Next: qsa_limit()

    /// Issue #24: what one lane reports to /metrics in each of its states.
    #[test]
    fn kv_lane_occupancy_states() {
        // a running request: its position is used, and its own prefix snapshot is not "cached"
        assert_eq!(kv_lane_occupancy(Some(1200), None, Some(900)), (1200, 0));
        // a prefill in flight (the lane does not exist yet): the rows written so far
        assert_eq!(kv_lane_occupancy(None, Some(4095), None), (4095, 0));
        assert_eq!(kv_lane_occupancy(None, Some(0), None), (0, 0));
        // a free lane: nothing used; the snapshot it keeps for the prefix cache
        assert_eq!(kv_lane_occupancy(None, None, Some(900)), (0, 900));
        assert_eq!(kv_lane_occupancy(None, None, None), (0, 0));
        // the lane wins over a stale prefill marker (admit clears it; a nested interleave step
        // publishes while it is still set for ANOTHER slot, which the caller filters by slot)
        assert_eq!(kv_lane_occupancy(Some(7), Some(3), None), (7, 0));
    }

    /// Issue #24 wiring: every scheduler function that frees or fills a lane ends in a publish, and
    /// the age gauge is written by the `step` wrapper (every arm), not by one arm of the rounds.
    #[test]
    fn kv_publish_points_are_wired() {
        let src = src_text!("exl3_serve");
        let body = |name: &str| -> &str {
            let at = src.find(name).unwrap_or_else(|| panic!("{name} not found"));
            let end = src[at + name.len()..].find("\n    fn ").map_or(src.len(), |e| at + name.len() + e);
            &src[at..end]
        };
        let publish = ["self.", "publish_kv()"].concat();
        let touch = ["crate::metrics::", "sched_touch()"].concat();
        for f in ["fn step(&mut self) -> Result<()> {", "fn admit(&mut self, req: BatchRequest) {", "fn die(mut self,",
                  "fn tp_fail(&mut self,", "fn prefill_range(&mut self,"] {
            assert!(body(f).contains(&publish), "{f} must publish the KV gauges");
        }
        assert!(body("fn step(&mut self) -> Result<()> {").contains(&touch), "the step wrapper writes the age gauge");
        assert!(!body("fn step_rounds(&mut self) -> Result<()> {").contains(&touch),
                "step_rounds must not write the age gauge itself (the fcfs and shared-plain arms return early)");
        // a request's last round: its final position is published before the lane is dropped
        // (the post-step publication sees a free lane), in the speculative and the plain step
        assert_eq!(src.matches(&[publish.as_str(), "; // issue #24: the lane's final position"].concat()).count(), 1);
        assert_eq!(src.matches(&["if finish { ", publish.as_str(), "; } // issue #24: the lane's final position"].concat()).count(), 1);
        // both scheduler loops publish the idle value before they block
        let idle = ["crate::metrics::", "sched_idle()"].concat();
        let n_idle = src.matches(&idle).count();
        let paired = src.matches(&[publish.as_str(), "; // issue #24: the idle value"].concat()).count();
        assert!(n_idle >= 2 && paired == n_idle, "every sched_idle() site publishes first ({paired} of {n_idle})");
    }

    /// 9a(iv): step_plain's order — the lanes at/below the switch first, stable in each class.
    #[test]
    fn step_plain_orders_short_lanes_first() {
        // slot -> the lane's next position
        let pos = |s: usize| [100usize, 5000, 2051, 2052, 7, 30000][s];
        // slots 0, 2, 4 are at/below the switch (2051 itself is dense); 1, 3, 5 are past it
        assert_eq!(regime_ordered_slots(&[0, 1, 2, 3, 4, 5], pos, LIM), vec![0, 2, 4, 1, 3, 5]);
        // a subset, given in another order: each class keeps the GIVEN order (a stable partition)
        assert_eq!(regime_ordered_slots(&[5, 4, 3, 2], pos, LIM), vec![4, 2, 5, 3]);
        // one-sided steps are untouched
        assert_eq!(regime_ordered_slots(&[4, 0, 2], pos, LIM), vec![4, 0, 2]);
        assert_eq!(regime_ordered_slots(&[5, 1, 3], pos, LIM), vec![5, 1, 3]);
        assert_eq!(regime_ordered_slots(&[1], pos, LIM), vec![1]);
        assert_eq!(regime_ordered_slots(&[], pos, LIM), Vec::<usize>::new());
        // the result is always a permutation of the input
        let mut o = regime_ordered_slots(&[3, 0, 5, 2, 1, 4], pos, LIM);
        o.sort();
        assert_eq!(o, vec![0, 1, 2, 3, 4, 5]);
        // the dense rows lead, so forward_step's dense prefix covers every short lane of the step
        let ord = regime_ordered_slots(&[1, 0, 3, 2, 5, 4], pos, LIM);
        let p: Vec<usize> = ord.iter().map(|&s| pos(s)).collect();
        assert_eq!(crate::exl3_forward::dense_prefix(&p, LIM), (3, 0), "3 leading dense rows, none behind a long row");
    }

    /// 9a(iv): with the flag off (or no sparse attention) the order is the identity.
    #[test]
    fn the_flag_off_order_is_the_identity() {
        assert_eq!(regime_switch(true, true, LIM), LIM);
        assert_eq!(regime_switch(true, false, LIM), usize::MAX, "--qsa-row-regime 0");
        assert_eq!(regime_switch(false, true, LIM), usize::MAX, "no indexer / --qsa-exl3 0");
        assert_eq!(regime_switch(false, false, LIM), usize::MAX);
        let pos = |s: usize| [100usize, 5000, 2051, 2052, 7, 30000][s];
        for slots in [vec![0, 1, 2, 3, 4, 5], vec![5, 4, 3, 2, 1, 0], vec![1, 0, 3], vec![]] {
            assert_eq!(regime_ordered_slots(&slots, pos, regime_switch(true, false, LIM)), slots, "flag off: {slots:?}");
            assert_eq!(regime_ordered_slots(&slots, pos, regime_switch(false, true, LIM)), slots, "no switch: {slots:?}");
        }
    }

    /// The method the scheduler calls is exactly these two functions (source pin).
    #[test]
    fn regime_ordered_is_the_two_pure_functions() {
        let src = src_text!("exl3_serve");
        let f = &src[src.find("fn regime_ordered(&self, slots: &[usize]) -> Vec<usize> {").expect("regime_ordered")..];
        let f = &f[..f.find("\n    }\n").unwrap()];
        assert!(f.contains("regime_switch(self.model.qsa_live(usize::MAX), crate::exl3_forward::row_regime_on(), self.model.qsa_limit())"));
        assert!(f.contains("regime_ordered_slots(slots, |s| self.lanes[s].as_ref().map_or(0, |l| l.pos), lim)"));
    }

    /// 9a(iii): the count of mixed step shapes, and the idle line that reports the graphs.
    #[test]
    fn mixed_step_graph_accounting() {
        use crate::exl3_forward::mixed_step_shapes;
        assert_eq!((mixed_step_shapes(0), mixed_step_shapes(1), mixed_step_shapes(2)), (0, 0, 1));
        assert_eq!(mixed_step_shapes(4), 6);
        assert_eq!(mixed_step_shapes(16), 120);
        // = the number of (width m in 2..=lanes, dense rows md in 1..m) pairs
        for lanes in 0..=16usize {
            let n = (2..=lanes).map(|m| m - 1).sum::<usize>();
            assert_eq!(mixed_step_shapes(lanes), n, "lanes {lanes}");
        }
        assert_eq!(mixed_graphs_idle_line(0, 0, 4), None);
        assert_eq!(mixed_graphs_idle_line(3, 3, 4), None, "nothing new: no line");
        let l = mixed_graphs_idle_line(0, 2, 4).unwrap();
        assert!(l.starts_with("[exl3-serve] mixed-step graphs idle: 0->2 captured since boot") && l.contains("4 lanes can form 6 mixed step shapes"), "{l}");
        // the gauge is rendered (the count itself is a process global other tests may bump)
        assert!(crate::metrics::render(true).contains("velogb10_graph_cache_mixed_step_entries "));
    }
