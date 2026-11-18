//! Unit tests (module `lane_pack_wiring_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    #[test]
    fn tp_is_refused_before_the_attach_on_both_ranks_and_in_build_serve() {
        let src = src_text!("exl3_serve");
        let call = format!("lane_pack::{}(", "tp_refusal");
        assert_eq!(src.matches(call.as_str()).count(), 3, "build_serve (any TP rank) + the head and node entry points");
        // the head and node entry points refuse BEFORE tp_attach, build_serve before the model load
        for entry in ["pub fn run_tp_head_serve(", "pub fn run_tp_node_serve("] {
            let f = &src[src.find(entry).unwrap_or_else(|| panic!("{entry}"))..];
            assert!(f.find(call.as_str()).unwrap() < f.find("tp_attach(ctx)").unwrap(), "{entry} refuses before the attach");
        }
        let b = &src[src.find("fn build_serve(").unwrap()..];
        assert!(b.find(call.as_str()).unwrap() < b.find("let model = ").expect("the model-load anchor of build_serve was renamed: update this pin"),
                "build_serve refuses before the load");
        assert!(src.contains("--lane-pack on needs --lane-order rr"), "fcfs steps one lane, a pack steps two");
    }

    /// PACK4-REVIEW A2: the reasoned pin. `pack_finish` re-primes eagerly; `head_reprime` takes
    /// `rp.pending` FIRST and, if a lone round left one behind (it failed between its
    /// `reprime_dev_enqueue` and its `head_reprime`), would run `reprime_dev_finish` on the wrong
    /// round and fail every live lane. The packed path therefore clears the hand-off first.
    #[test]
    fn a_packed_round_never_takes_a_stale_reprime_handoff() {
        let fwd = src_text!("exl3_forward");
        let pf = &fwd[fwd.find("pub fn pack_finish(&self").unwrap()..];
        let pf = &pf[..pf.find("\n    }\n").unwrap()];
        assert!(pf.find("rp.pending = None;").expect("the stale hand-off is cleared")
                < pf.find("self.head_reprime(sc, head, d.slot").unwrap(), "cleared BEFORE head_reprime can take it");
        // the lone path is untouched: its enqueue still resets the hand-off itself, and head_reprime
        // still consumes it with take()
        let re = &fwd[fwd.find("fn reprime_dev_enqueue(&self").unwrap()..];
        assert!(re[..re.find("\n    }\n").unwrap()].contains("rp.pending = None;"));
        assert!(fwd.contains("sc.rp.as_mut().and_then(|r| r.pending.take())"));
    }

    /// PACK4-REVIEW A3: an error out of the packed part of a pair round switches the arm off (the
    /// planner then returns None: master's loops), loudly and visibly; an error of the control
    /// cadence's serial fallback does not.
    #[test]
    fn a_packed_round_error_turns_the_arm_off_for_the_boot() {
        let src = src_text!("exl3_serve");
        let sk = &src[src.find("fn step_packed(&mut self").unwrap()..];
        let sk = &sk[..sk.find("\n    }\n").unwrap()];
        assert!(sk.contains("std::mem::take(&mut self.pack_stats.in_round)") && sk.contains("self.pack_disable(a, b, &e)"));
        let sp = &src[src.find("fn step_pair(&mut self").unwrap()..];
        let sp = &sp[..sp.find("\n    }\n").unwrap()];
        // the fallback's serial rounds run BEFORE the flag is raised; the packed part raises it and
        // lowers it on completion
        let fallback = sp.find(["return self.", "step_mtp(sb, false);"].concat().as_str()).expect("the control-cadence fallback");
        let raise = sp.find("self.pack_stats.in_round = true;").expect("flag raised");
        let lower = sp.find("self.pack_stats.in_round = false;").expect("flag lowered");
        assert!(fallback < raise && raise < lower);
        let pd = &src[src.find("fn pack_disable(&mut self").unwrap()..];
        let pd = &pd[..pd.find("\n    }\n").unwrap()];
        assert!(pd.contains("self.lane_pack = crate::lane_pack::LanePack::Off;") && pd.contains("crate::metrics::lane_pack_disabled();"));
        assert!(src[src.find("fn pack_plan(&mut self").unwrap()..].contains("if !self.lane_pack.on() || slots.len() < 2 { return None; }"),
                "the planner returns None once the mode is Off: master's loops");
    }

    /// PACK4-REVIEW A4: the planner and the pair round both use the capped depth (one definition:
    /// `lane_pack::PACK_DEPTH_MAX`), so the forward path never meets a pair `pack_rows` refuses.
    #[test]
    fn the_planner_and_the_pair_round_cap_the_depth_alike() {
        let src = src_text!("exl3_serve");
        let pp = &src[src.find("fn pack_plan(&mut self").unwrap()..];
        let pp = &pp[..pp.find("\n    }\n").unwrap()];
        // the per-lane rule is the pure `lane_view` (PACK4-REVIEW B4), which applies the cap and the clamp flag
        assert!(pp.contains("crate::lane_pack::lane_view(s, facts, l.pos, k_pre, kd_reg, self.in_prefill, self.pack_ready"));
        let lp = src_text!("lane_pack");
        let lv = &lp[lp.find("pub(crate) fn lane_view(").unwrap()..];
        let lv = &lv[..lv.find("\n}\n").unwrap()];
        assert!(lv.contains("kd_reg.min(PACK_DEPTH_MAX)") && lv.contains("clamped: kd_reg < k_pre"));
        let sp = &src[src.find("fn step_pair(&mut self").unwrap()..];
        let sp = &sp[..sp.find("\n    }\n").unwrap()];
        assert!(sp.contains("kd.min(crate::lane_pack::PACK_DEPTH_MAX)"));
        // PACK4-REVIEW B4: step_pair drafts at the round's CLAMPED depth (round_depth's regime-clamped value), never at mtp_k
        assert!(sp.contains("let (_, kd) = self.round_depth(l.pos, l.max_new.saturating_sub(l.generated), l.think_room());"));
        assert!(!sp.contains("self.mtp_k.min("), "the pair round must not recompute a depth of its own");
    }

    /// PACK4-REVIEW B2: the packed precapture decides nothing from a free-memory delta; a capture
    /// that did not land still fails the boot, per graph.
    #[test]
    fn the_packed_precapture_has_no_memory_delta_cap() {
        let fwd = src_text!("exl3_forward");
        let pc = &fwd[fwd.find("pub fn pack_precapture(&self").unwrap()..];
        let pc = &pc[..pc.find("\n    }\n").unwrap()];
        assert!(!pc.contains("1usize << 30") && !pc.contains("precapture cap"), "no cuMemGetInfo-delta cap");
        assert!(pc.contains("did not capture (see the EXL3_GRAPH line above)"), "a failed capture still fails the boot");
    }

    /// S-B9-CF-PACK5: the packed round's lockstep points, in source order. After BOTH lanes drafted:
    /// the wasted-pass drain, then the ONE pre-verify exchange (before `pack_verify` launches the
    /// R-keyed graph); after the finishes: the post-verify `tp_round` whose head ms / capture flag
    /// feed `round_account` (never the local clock). The pack is built from lane state only.
    #[test]
    fn the_packed_round_runs_its_lockstep_points_in_order() {
        let src = src_text!("exl3_serve");
        let sp = &src[src.find("fn step_pair(&mut self").unwrap()..];
        let sp = &sp[..sp.find("\n    }\n").unwrap()];
        let at = |pat: &str| sp.find(pat).unwrap_or_else(|| panic!("missing in step_pair: {pat}"));
        let drafts_done = at("drafts.push(d);"); // the end of the per-lane draft loop body: BOTH lanes have drafted after it
        let drain = at("self.model.pack_drain_wasted(&drafts)?;");
        let pre = at("t.pre_verify_pack(r, segs[0].drafts.len() + 1, &words, launched)?;");
        let verify = at("self.model.pack_verify(&mut self.sc, scp, &drafts[0], &drafts[1])?");
        let fin = at("self.model.pack_finish(&mut self.sc, head, &drafts[j], &outs[j])?;");
        let post = at("self.tp_round(\"pack-round\", outs[0].a + outs[1].a, r, &w, ms_local, 0.0, captured_local, true)?");
        let acct = at("self.round_account(s, &outs[j], ms_lane, 0.0, captured,");
        assert!(drafts_done < drain && drain < pre && pre < verify, "both drafts, then the drain, the pre-verify, then the verify");
        assert!(verify < fin && fin < post && post < acct, "verify, finishes, post-verify agree, then the bookkeeping");
        assert!(sp.contains("let ms_lane = lane_share_ms(ms_round, 2);"), "the lane's share of the AGREED (head) round time");
        assert!(sp.contains("if self.tp.is_some() && tp_preverify_on() {"), "the pre-verify honours --exl3-tp-preverify=0 like the lone round");
        // nothing in the pack is decided from rank-local clocks or graph presence: the drafts loop reads
        // lane state and the pre list, the dds target comes from lane state + the head-agreed EMAs
        assert!(!sp.contains("lane_share_ms(ms_of("), "a local clock must not be the lane's cost");
        // the arm goes off after a packed-round error (PACK4-REVIEW A3), and a lone round's stale
        // device re-prime hand-off cannot fail a pair (A2)
        let sk = &src[src.find("fn step_packed(&mut self").unwrap()..];
        let sk = &sk[..sk.find("\n    }\n").unwrap()];
        assert!(sk.contains("self.pack_disable(a, b, &e)"));
        let fwd = src_text!("exl3_forward");
        let pf = &fwd[fwd.find("pub fn pack_finish(&self").unwrap()..];
        let pf = &pf[..pf.find("\n    }\n").unwrap()];
        assert!(pf.find("rp.pending = None;").unwrap() < pf.find("self.head_reprime(sc, head, d.slot").unwrap(),
                "pack_finish clears a stale re-prime hand-off before the eager re-prime");
    }

    #[test]
    fn the_packed_arm_sits_behind_the_flag_and_in_front_of_the_serial_loops() {
        let src = src_text!("exl3_serve");
        // the arms live in `step_rounds`; `step` is the wrapper that publishes after any of them
        let st = &src[src.find("fn step_rounds_rest(&mut self, slots: Vec<usize>, mtp_on: bool) -> Result<()>").unwrap()..]; // LANEWIDE: today's arms
        let st = &st[..st.find("\n    }\n").unwrap()];
        let (shared, pack, fcfs, serial) = (st.find(["return self.", "step_plain(&slots, true);"].concat().as_str()).unwrap(),
            st.find("self.pack_plan(&slots)").expect("the packed arm"), st.find("self.lane_order == LaneOrder::Fcfs").unwrap(),
            st.find("let mut plain: Vec<usize> = Vec::new();\n        for &s in &slots {").unwrap());
        assert!(shared < pack && pack < fcfs && fcfs < serial, "shared-plain, then the packed arm, then the serial loops");
        let pp = &src[src.find("fn pack_plan(&mut self").unwrap()..];
        let pp = &pp[..pp.find("\n    }\n").unwrap()];
        assert!(pp.contains("if !self.lane_pack.on() || slots.len() < 2 { return None; }"), "off or a lone request: master's path");
        // the boot precaptures the packed family (never while serving) and refuses a half-built one
        assert!(src.contains("model.pack_precapture(&mut scp, &fams)?"), "the families (greedy + real-q / match) are captured at boot");
        assert!(src.contains("crate::lane_pack::fams(spec_ratio, sampled_ok)"), "one list of families for capture, readiness and the boot agree");
        assert!(src.contains("the dense packed graph family is incomplete"));
    }

    /// S-B9-CF-PACK4S: the sampled-lane glue, pinned in source order. Per lane, inside ONE loop
    /// body: the lane's own sampler staging on the served scratch (the lone round's two calls),
    /// then its draft chain, then the hand-off of its real-q lists — so the first lane's lists
    /// leave the served scratch before the second lane's chain overwrites them.
    #[test]
    fn a_pair_stages_drafts_and_hands_off_per_lane_in_order() {
        let src = src_text!("exl3_serve");
        let src = &src[..src.find("mod lane_pack_wiring_tests").unwrap()]; // the code, not these pins
        let sp = &src[src.find("fn step_pair(&mut self").unwrap()..];
        let sp = &sp[..sp.find("\n    }\n").unwrap()];
        let body = &sp[sp.find("for &(s, b, p, tap_row, kd, _) in &pre {").expect("the per-lane draft loop")..];
        let body = &body[..body.find("let t1 = std::time::Instant::now();").unwrap()];
        let at = |pat: &str| body.find(pat).unwrap_or_else(|| panic!("missing in the draft loop: {pat}"));
        let (rows, stage, pen) = (at("samp_rows(k + 1), self.rq_stage(s)"), at("self.model.set_spec_sampling(&mut self.sc, &rows, rq)?"),
                                  at("self.model.set_penalties(&mut self.sc, &vec![None; k + 1])?"));
        let (dds, fixed, hand, push) = (at("Some(&mut keep), seg, rq)?"), at("tap_row, None, seg, rq)?"),
                                        at("self.model.pack_list_handoff(&self.sc, scp, at, &d)?"), at("drafts.push(d);"));
        assert!(rows < stage && stage < pen && pen < dds && dds < fixed && fixed < hand && hand < push,
                "stage -> draft -> list hand-off, per lane");
        assert!(at("at += d.drafts.len() + 1;") > hand, "the next segment's row offset follows this lane's rows");
        // the lane's real-q flag reaches the accounting (ctr += k + 1, dctr += k, as in the lone round)
        assert!(sp.contains("captured, false, false, drafts[j].samp.ratio, xcheck, true, false)?")); // WIDEVERIFY1: + copy = false
        // the all-greedy staging of PACK4 is gone: each lane stages its own
        assert!(!sp.contains("set_spec_sampling(&mut self.sc, &vec![None; k + 1], None)"));
        // the fixed-depth chain is allowed under lane-pack (the deterministic gate posture)
        assert!(!src.contains(&format!("{} on needs the dynamic draft stop", "--lane-pack")));
        // a control round never runs with --a-ctrl off, in the lone round and in the pair's check
        assert!(src.contains("let control = (af.ctrl != 3 && self.ctrl_round % period == 0) || force_ctrl;"));
        assert!(sp.contains("if af.ctrl != 3 && pair_owes_control(self.ctrl_round, per_a, per_b) {"));
    }

    /// S-B9-CF-PACK4S: the forward side. The real-q list follows the scratch width (16 rows for
    /// the packed scratch); the packed verify stages its sampler block BEFORE it picks the graph
    /// (the family is the staged pair's); the precapture leaves the scratch greedy.
    #[test]
    fn the_packed_verify_stages_per_row_samplers_and_picks_the_family() {
        let fwd = src_text!("exl3_forward");
        assert_eq!(fwd.matches("rq_list: dev.htod_sync_copy(&vec![u32::MAX; crate::lane_pack::rq_list_rows(w) * crate::wp24::RQ_ROW])?").count(), 1,
                   "the list is sized from the scratch width");
        assert!(!fwd.contains("vec![u32::MAX; (MTP_MAX_K + 1) * crate::wp24::RQ_ROW]"), "the 8-row list is gone");
        let pv = &fwd[fwd.find("pub fn pack_verify(&self").unwrap()..];
        let pv = &pv[..pv.find("\n    }\n").unwrap()];
        let at = |pat: &str| pv.find(pat).unwrap_or_else(|| panic!("missing in pack_verify: {pat}"));
        let (samp, set, key, launch) = (at("crate::lane_pack::pack_samp(&meta.segs, &[d0.samp, d1.samp], scp.samp.len() / 8)?"),
                                        at("self.set_pack_sampling(scp, &ps)?"),
                                        at("pack_key(meta.r, q0, qb, scp.rq_live) | self.vp_key_st(scp.samp_live, false, scp.rq_live, meta.r) | sk(scp)"),
                                        at(".launch();"));
        assert!(samp < set && set < key && key < launch);
        let pc = &fwd[fwd.find("pub fn pack_precapture(&self").unwrap()..];
        let pc = &pc[..pc.find("\n    }\n").unwrap()];
        // the family's sampler shadows are set BEFORE the capture (the verify tail is chosen from them)
        // and restored to greedy after the last one; the graph key is the family's own (vp bits included)
        assert!(pc.find("scp.rq_live = rq;").unwrap() < pc.find("self.capture_verify_graph_packed(").unwrap());
        assert!(pc.find("scp.samp_live = samp;").unwrap() < pc.find("self.capture_verify_graph_packed(").unwrap());
        assert!(pc.find("self.capture_verify_graph_packed(").unwrap() < pc.find("scp.rq_live = false;").unwrap());
        assert!(pc.find("self.capture_verify_graph_packed(").unwrap() < pc.find("scp.samp_live = false;").unwrap());
        assert!(pc.contains("self.pack_fam_key(scp, r, sparse, qb, fam)"), "one key function for capture, readiness and replay");
        let fk = &fwd[fwd.find("fn pack_fam_key(&self").unwrap()..];
        let fk = &fk[..fk.find("\n    }\n").unwrap()];
        assert!(fk.contains("self.vp_key_st(samp, false, rq, r)"), "the vp tail bits are part of the family key");
        // the hand-off is a D2D copy on the COMPUTE stream (AGENTS 2.3), never a sync copy
        let ho = &fwd[fwd.find("pub fn pack_list_handoff(&self").unwrap()..];
        let ho = &ho[..ho.find("\n    }\n").unwrap()];
        assert!(ho.contains("cuMemcpyDtoDAsync_v2(dst, src, words * 4, self.stream.stream)") && !ho.contains("dtod_sync"));
    }
