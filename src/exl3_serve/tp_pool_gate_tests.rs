//! Unit tests (module `tp_pool_gate_tests`): TPPOOL (design v3 step 4), the head's capacity gate. CPU only.

    use super::tp_pool_gate::{gate_decide, GateDecision, TP_EVICTS};

    #[test]
    fn a_prompt_that_fits_ships_and_is_charged_its_pages() {
        assert_eq!(gate_decide(10, 10, 0, &[], false, true), GateDecision::Ship { pages: 10 });
        assert_eq!(gate_decide(0, 0, 0, &[], false, false), GateDecision::Ship { pages: 0 }, "the fast path needs nothing");
    }

    #[test]
    fn admits_shipped_earlier_in_the_step_are_charged_before_the_next_one() {
        // 30 free; the first Admit of the step took 25: a second 10-page prompt must not be approved on the same pages
        assert_eq!(gate_decide(10, 30, 0, &[], false, true), GateDecision::Ship { pages: 10 });
        assert_eq!(gate_decide(10, 30, 25, &[], false, true), GateDecision::Hold);
        // a reservation larger than what is free never underflows
        assert_eq!(gate_decide(1, 5, 9, &[], false, true), GateDecision::Hold);
    }

    #[test]
    fn a_short_pool_holds_while_a_lane_runs_and_refuses_when_nothing_will_free_room() {
        assert_eq!(gate_decide(50, 10, 0, &[], false, true), GateDecision::Hold);
        match gate_decide(50, 10, 0, &[], false, false) {
            GateDecision::Refuse(why) => {
                assert!(why.starts_with("error: "), "{why}");
                assert!(why.contains("50 pages") && why.contains("10 free"), "{why}");
            }
            d => panic!("expected a refusal, got {d:?}"),
        }
    }

    #[test]
    fn victims_are_credited_one_by_one_and_one_that_frees_nothing_counts_nothing() {
        // need 20, 5 free: victims free 0, 10, 10 in eviction order -> fits after the third
        assert_eq!(gate_decide(20, 5, 0, &[0, 10, 10], true, false), GateDecision::Ship { pages: 20 });
        // the zero-page victim does not make up a shortfall
        assert_eq!(gate_decide(20, 5, 0, &[0, 0], true, true), GateDecision::Hold);
        assert!(matches!(gate_decide(20, 5, 0, &[0, 0], true, false), GateDecision::Refuse(_)));
    }

    #[test]
    fn victims_are_never_credited_while_tp_eviction_is_off() {
        // the pages exist on paper, but under TP no eviction runs today: crediting them would ship a fatal
        assert!(!TP_EVICTS, "turn the gate's crediting on in the same change that lets vmm_evict_idle run under TP");
        assert_eq!(gate_decide(20, 5, 0, &[100], false, true), GateDecision::Hold);
    }

    #[test]
    fn the_gate_charges_a_restore_target_past_the_next_grid_line() {
        // the apply maps rows_grid(rows + margin): plen 11264 direct = 12288 rows, a ratio-4 restore of the same
        // conversation maps spill_rows(11264) = 11268 rows -> 16384. The gate charges max(plen, restore rows).
        use crate::exl3_forward::{VMM_MARGIN_ROWS, spill_rows};
        let (plen, ratio, max_pos) = (11264usize, 4usize, 1_048_572usize);
        let direct = crate::kv_vmm::rows_grid(plen + VMM_MARGIN_ROWS, max_pos);
        let restore = crate::kv_vmm::rows_grid(plen.max(spill_rows(plen, ratio, max_pos)) + VMM_MARGIN_ROWS, max_pos);
        assert_eq!(direct, 12288);
        assert_eq!(restore, 16384);
    }

    /// Source pin: both head admission arms consult the gate before an Admit ships, and the FIFO arm ships one Admit
    /// per iteration while the gate decides (a second verdict in the same iteration would be stale, see `tp_pool_gated`).
    #[test]
    fn the_head_gates_every_admit_and_ships_one_per_iteration_while_gated() {
        let src = src_text!("exl3_serve");
        assert_eq!(src.matches("self.tp_pool_gate(&req, Some(allowed.as_slice()), &pending_free, gate_reserved)").count(), 1,
                   "the lane-reserve arm");
        assert_eq!(src.matches("self.tp_pool_gate(&req, None, &pending_free, gate_reserved)").count(), 1, "the FIFO arm");
        let fifo = src.find("self.tp_pool_gate(&req, None,").unwrap();
        let ship = fifo + src[fifo..].find("events.push(TpEvent::Admit(w));").expect("the FIFO Admit");
        let after = &src[ship..ship + 300];
        assert!(after.contains("if self.tp_pool_gated() { break; }"), "one Admit per iteration while gated: {after}");
    }
