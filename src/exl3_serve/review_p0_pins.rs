//! Unit tests (module `review_p0_pins`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    fn code(src: &str) -> String {
        src.lines().filter(|l| !l.trim_start().starts_with("//")).collect::<Vec<_>>().join("\n")
    }

    #[test]
    fn df2_round_width_is_clamped_and_staged_for_greedy_and_sampled() {
        let s = code(src_text!("exl3_serve"));
        // 127: one clamped width, staged for greedy and sampled alike
        assert!(s.contains("let k_df2 = self.df2_k.min(crate::exl3_forward::MTP_MAX_K).min(k_eff);"), "DF2 width clamp");
        let k = s.find("let k_df2 = self.df2_k").unwrap();
        let branch = &s[k..k + 2500];
        let stage = branch.find("self.model.set_penalties(&mut self.sc, &vec![pen; k_df2 + 1])?;").expect("penalty rows at k_df2+1");
        assert!(stage < branch.find("if rq_ratio {\n").unwrap(), "rows staged before the sampled/greedy split");
    }

    #[test]
    fn served_draft_chains_stage_the_lane_slot_in_every_regime() {
        let s = code(src_text!("exl3_forward"));
        assert!(!s.contains(&["if dqsa { self.dev.htod_copy_into(vec![slot as i32; sc.slot_ids", ".len()], &mut sc.slot_ids)?; }"].concat()),
                "the fixed chain no longer stages the slot only for a sparse last pass");
        assert!(!s.contains(&["if self.qsa_live(p + kmax - 1) {\n            if stage {\n                let st = sc.wp05", ""].concat()),
                "the adaptive chain no longer stages the slot only for a sparse last pass");
    }
