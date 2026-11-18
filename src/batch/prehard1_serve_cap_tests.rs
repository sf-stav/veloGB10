//! Unit tests (module `prehard1_serve_cap_tests`) moved out of the old inline block; text unchanged.

    #[test]
    fn serve_max_lanes_is_16_and_matches_the_exl3_cap() {
        assert_eq!(crate::batch::SERVE_MAX_LANES, 16);
        assert_eq!(crate::exl3_serve::EXL3_MAX_LANES, crate::batch::SERVE_MAX_LANES,
            "one authority: the EXL3 and trunk lane caps must not diverge");
    }

    #[test]
    fn over_cap_excess_counts_arrivals_like_the_maxbatch1_soak() {
        use crate::batch::over_cap_excess;
        assert_eq!(over_cap_excess(16, 0, 16), 0, "at the cap: nothing over");
        assert_eq!(over_cap_excess(16, 5, 16), 5, "5 waiting beyond 16 busy lanes");
        assert_eq!(over_cap_excess(0, 12, 5), 7,
            "a burst on an idle box: arrivals minus the lanes that could take them (MAXBATCH1 soak semantics)");
        assert_eq!(over_cap_excess(3, 0, 16), 0, "saturating: never negative");
        assert_eq!(over_cap_excess(0, 0, 16), 0);
    }

    /// The wiring is pinned in the source, MAXBATCH1-style: the trunk parse caps --max-batch at
    /// SERVE_MAX_LANES, records the requested width on its gauge, prints the applies-and-shows
    /// line, the W3 kv_lane_check sees the CLAMPED width, and BOTH trunk serve loops
    /// (run, run_tp_head) carry the over-cap accounting.
    #[test]
    fn trunk_serve_path_applies_and_shows_the_cap() {
        let main_src = src_text!("main");
        let cap = main_src.find("let max_batch = max_batch_raw.min(gb10_inference::batch::SERVE_MAX_LANES);")
            .expect("the trunk parse caps --max-batch at SERVE_MAX_LANES");
        let req = main_src.find("set_max_batch_requested(max_batch_raw)")
            .expect("the requested gauge is set on the trunk path");
        assert!(req > cap, "the gauge line follows the clamp");
        assert!(main_src.contains("further requests wait in the queue"),
            "the applies-and-shows boot line");
        let kv = main_src.find("kv_lane_check(").expect("kv_lane_check call");
        assert!(kv > cap, "the KV x lane check sees the clamped width");
        let self_src = src_text!("batch");
        // Assemble the needle so this test's own literal cannot match itself.
        let needle = ["count_over_cap_", "arrivals(self.num_", "active()"].concat();
        assert_eq!(self_src.matches(&needle).count(), 2,
            "exactly the two trunk serve loops (run + run_tp_head) carry the hook");
        assert!(self_src.contains("self.rx.len()"),
            "the TP1 hook samples the channel backlog");
        assert!(self_src.contains("pending.len() + self.rx.len()"),
            "the TP hook adds the pending queue");
    }
