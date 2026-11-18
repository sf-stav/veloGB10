//! Unit tests (module `copy_span_arm_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::*;

    /// ULTRA2 merge pin: COPY27B's changes to shared copy-span code stay off the EXL3 (Flash-Next) path. The cost-ratio
    /// yield rule is set only by the trunk's `copy_arm_cfg_w`; the depth-policy cadence (open deep, explore every 8th
    /// window, 32-round windows) lives in the trunk's MtpPolicy and is armed only with the trunk copy arm; the trunk
    /// resolver is called only from the trunk scheduler.
    #[test]
    fn copy27b_shared_changes_are_trunk_gated() {
        let me = src_text!("exl3_serve");
        let body = |name: &str| { let i = me.find(name).unwrap(); &me[i..i + 400] };
        let yr = concat!(".with_yield", "_ratio(");
        assert_eq!(me.matches(yr).count(), 1, "one caller in this file");
        assert!(body("pub(crate) fn copy_arm_cfg_w(").contains(concat!(".with_yield", "_ratio(TRUNK_YIELD_RATIO)")));
        assert!(!body("pub(crate) fn copy_arm_cfg(").contains("with_yield_ratio"), "the EXL3 arm keeps ratio 0");
        assert_eq!(copy_arm_cfg(16, 10, true, 4).yield_ratio, 0);
        assert_eq!(copy_arm_cfg(16, 15, false, usize::MAX).yield_ratio, 0);
        let calls = me.matches(concat!("copy_span_resolve", "_trunk(")).count();
        assert_eq!(calls, 1, "only its definition in the EXL3 server (no call)");
        let batch = src_text!("batch");
        assert_eq!(batch.matches(".set_explore(").count(), 2, "one serving call + one test call (copy_explore_probes_max_depth_only_when_armed)");
        assert!(batch.contains("if s.copy_arm.is_some() { s.mtp.set_explore(COPY_MTP_EXPLORE_EVERY); s.mtp.open_deep(); }"));
        assert!(batch.contains("explore_every: 0, windows: 0 }"), "default: no exploration");
    }

    /// COPY27B: the trunk server's resolution and width cap (16 verify rows at most).
    #[test]
    fn copy_span_trunk_decide_and_width() {
        use CopySpanState as S;
        assert_eq!(copy_span_decide_trunk(false, 1, true), S::NotAsked);
        assert_eq!(copy_span_decide_trunk(true, 1, true), S::On);
        assert_eq!(copy_span_decide_trunk(true, 1, false), S::OffNoSpec);
        for w in [2usize, 3, 4, 8] {
            let want = if COPY_SPAN_TRUNK_TP_WORLDS.contains(&w) { S::On } else { S::OffTp };
            assert_eq!(copy_span_decide_trunk(true, w, true), want, "world {w}");
            assert_eq!(copy_span_decide_trunk(false, w, true), S::NotAsked);
        }
        assert_eq!(copy_span_trunk_width(10), 10);
        assert_eq!(copy_span_trunk_width(15), 15);
        assert_eq!(copy_span_trunk_width(64), crate::gpu::MAX_VERIFY - 1);
        assert_eq!(copy_arm_cfg_w(12, 15, false, 4), crate::copy_span::CopySpanCfg::new(12).with_start_w(4));
        assert_eq!(copy_arm_cfg_w(16, 15, true, 4).yield_w, 15);
        assert_eq!(copy_arm_cfg_w(16, 15, true, 4).yield_ratio, TRUNK_YIELD_RATIO);
        assert_eq!(copy_arm_cfg(16, 10, true, 4).yield_ratio, 0, "the EXL3 arm keeps its own rule");
        let mut o = String::new();
        copy_span_metrics_append(&mut o);
        for r in ["concurrent", "schema", "source", "not_primed", "spec_off", "tp_disagree"] {
            assert!(o.contains(&format!("velogb10_copy_span_skipped_rounds_total{{reason=\"{r}\"}} ")), "{r}");
        }
        assert!(o.contains("velogb10_copy_span_trunk "));
        assert!(o.contains("velogb10_copy_span_bare_rounds_total "));
    }

    /// WIDEVERIFY1: the --copy-span resolution — armed only on an EXL3 pack, one box, speculation
    /// live; every other case runs off with its own state (never a refusal).
    #[test]
    fn copy_span_decide_falls_back_never_refuses() {
        use CopySpanState as S;
        assert_eq!(copy_span_decide(false, true, 1, true), S::NotAsked);
        assert_eq!(copy_span_decide(false, false, 4, false), S::NotAsked, "not asked wins over everything");
        assert_eq!(copy_span_decide(true, true, 1, true), S::On);
        // WIDEVERIFY1 TP leg: TP=2 and TP=4 are wired; any other world runs off, named before speculation
        assert_eq!(copy_span_decide(true, true, 2, true), S::On);
        assert_eq!(copy_span_decide(true, true, 4, true), S::On);
        assert_eq!(copy_span_decide(true, true, 2, false), S::OffNoSpec);
        assert_eq!(copy_span_decide(true, true, 3, true), S::OffTp);
        assert_eq!(copy_span_decide(true, true, 8, true), S::OffTp);
        assert_eq!(copy_span_decide(true, true, 8, false), S::OffTp, "an unwired TP world is named first");
        assert_eq!(copy_span_decide(false, true, 2, true), S::NotAsked, "off by default at every world");
        assert_eq!(copy_span_decide(false, true, 4, true), S::NotAsked);
        assert_eq!(copy_span_decide(true, false, 1, true), S::OffFamily);
        assert_eq!(copy_span_decide(true, true, 1, false), S::OffNoSpec);
        assert_eq!((S::OffTp.reason(), S::OffFamily.reason(), S::OffNoSpec.reason(), S::On.reason()),
                   ("tp", "family", "no_speculation", ""));
    }

    /// The copy round's width: the scheduler's clamped depth, --copy-span-max-w and the verify ceiling
    /// (MTP_MAX_K; COPY_MAX_K once --copy-span-max-w > MTP_MAX_K arms the wide copy verify).
    #[test]
    fn copy_width_clamps_to_depth_flag_and_verify() {
        use crate::exl3_forward::{COPY_MAX_K, COPY_VROWS, MTP_MAX_K};
        let k = MTP_MAX_K;
        assert_eq!(copy_width(10, 10), 10.min(k));
        assert_eq!(copy_width(3, 10), 3, "budget / window / sparse-switch clamp wins");
        assert_eq!(copy_width(10, 4), 4, "--copy-span-max-w wins");
        assert_eq!(copy_width(0, 10), 0, "k_eff 0 is a plain step: nothing to propose");
        // the default (max_w <= MTP_MAX_K): never wider than the chain verify, whatever the depth passed in
        for d in 0..=64usize {
            for w in 1..=MTP_MAX_K {
                assert_eq!(copy_width(d, w), d.min(w).min(MTP_MAX_K), "depth {d} max_w {w}: the pre-refinement width");
            }
        }
        assert!(!copy_span_wide(10) && copy_depth_cap(10) == MTP_MAX_K, "the default max_w 10 does not arm the wide verify");
        // refinement 2: max_w > MTP_MAX_K arms the wide copy verify, capped at COPY_MAX_K
        assert!(copy_span_wide(MTP_MAX_K + 1) && copy_depth_cap(64) == COPY_MAX_K);
        assert_eq!(copy_width(64, 64), COPY_MAX_K, "never wider than the wide copy verify");
        assert_eq!(copy_width(usize::MAX, 64), COPY_MAX_K);
        assert_eq!(copy_width(usize::MAX, 12), 12);
        assert_eq!(copy_width(10, 15), 10, "the depth clamp still wins");
        // the served scratch rows: wide only when the arm is armed AND wide
        assert_eq!(copy_span_scratch_vrows(true, 1, true, 15), COPY_VROWS);
        assert_eq!(copy_span_scratch_vrows(true, 4, true, 64), COPY_VROWS);
        for (asked, world, spec, w) in [(true, 1, true, 10), (false, 1, true, 15), (true, 8, true, 15), (true, 1, false, 15), (true, 2, true, 0)] {
            assert_eq!(copy_span_scratch_vrows(asked, world, spec, w), MTP_MAX_K + 1, "{asked} {world} {spec} {w}");
        }
    }

    /// One boot line per state; the armed line names the arm and the resolved min_n / max_w.
    #[test]
    fn copy_span_boot_lines() {
        assert_eq!(copy_span_boot_line(CopySpanState::NotAsked, 16, 10, 1, true, 4), None);
        let on = copy_span_boot_line(CopySpanState::On, 16, 10, 1, true, 4).unwrap();
        assert!(on.starts_with("[exl3-serve] copy-span: ON") && on.contains(">= 16 tokens") && on.contains("--copy-span-max-w 10"), "{on}");
        assert!(!on.contains("TP="), "one box names no TP world: {on}");
        assert!(on.contains("start ramp: a copy's first round proposes at most 4 tokens") && on.contains("--copy-span-start-w 4"), "{on}");
        assert!(on.contains("verify width 10") && !on.contains("WIDE"), "{on}");
        let full = copy_span_boot_line(CopySpanState::On, 16, 10, 1, true, 10).unwrap();
        assert!(full.contains("no start ramp") && full.contains("--copy-span-start-w 10"), "{full}");
        let w = copy_span_boot_line(CopySpanState::On, 4, 64, 1, true, 4).unwrap();
        assert!(w.contains(&format!("up to {} tokens", crate::exl3_forward::COPY_MAX_K)) && w.contains("WIDE copy verify")
                && w.contains("verify width 15") && w.contains("not yet gated"), "{w}");
        for world in COPY_SPAN_TP_WORLDS {
            let t = copy_span_boot_line(CopySpanState::On, 16, 10, world, true, 4).unwrap();
            assert!(t.starts_with("[exl3-serve] copy-span: ON") && t.contains(&format!("TP={world}")) && t.contains("pre-verify"), "{t}");
        }
        let tp = copy_span_boot_line(CopySpanState::OffTp, 0, 0, 8, true, 0).unwrap();
        assert!(tp.contains("asked (--copy-span on), runs OFF") && tp.contains("TP=8") && tp.contains("[2, 4]"), "{tp}");
        assert!(copy_span_boot_line(CopySpanState::OffFamily, 0, 0, 1, true, 0).unwrap().contains("EXL3"));
        assert!(copy_span_boot_line(CopySpanState::OffNoSpec, 0, 0, 1, true, 0).unwrap().contains("--mtp off"));
        assert!(on.contains("yield gate ON"), "{on}");
        let off = copy_span_boot_line(CopySpanState::On, 16, 10, 1, false, 4).unwrap();
        assert!(off.contains("yield gate OFF"), "{off}");
        // the arm's proposer config: W = the copy width cap when the gate is on, 0 (off) otherwise
        assert_eq!(copy_arm_cfg(16, 10, true, 4).yield_w, copy_width(usize::MAX, 10));
        assert_eq!(copy_arm_cfg(16, 64, true, 4).yield_w, crate::exl3_forward::COPY_MAX_K);
        assert_eq!(copy_arm_cfg(16, 10, false, 4), crate::copy_span::CopySpanCfg::new(16).with_start_w(4));
        assert_eq!(copy_arm_cfg(16, 10, false, 4).start_w, 4);
        assert_eq!(cs_start_w_str(usize::MAX), "off");
    }

    /// The metrics block renders every series, labelled reasons included (owner_check.py reads them).
    #[test]
    fn copy_span_metrics_render_every_series() {
        let mut o = String::new();
        copy_span_metrics_append(&mut o);
        for name in ["velogb10_copy_span_state ", "velogb10_copy_span_min_n ", "velogb10_copy_span_max_w ",
                     "velogb10_copy_span_rounds_total ", "velogb10_copy_span_offered_tokens_total ",
                     "velogb10_copy_span_accepted_tokens_total ", "velogb10_copy_span_emitted_tokens_total ",
                     "velogb10_copy_span_zero_accept_rounds_total ", "velogb10_copy_span_round_seconds_total ",
                     "velogb10_copy_span_host_seconds_total ",
                     "velogb10_copy_span_proposals_total{source=\"search\"} ", "velogb10_copy_span_proposals_total{source=\"cursor\"} ",
                     "velogb10_copy_span_misses_total{reason=\"no_match\"} ", "velogb10_copy_span_misses_total{reason=\"below_min\"} ",
                     "velogb10_copy_span_copy_ends_total{reason=\"mismatch\"} ",
                     "velogb10_copy_span_fallback_total{reason=\"tp\"} ",
                     "velogb10_copy_span_tp_world ", "velogb10_copy_span_tp_agreed_total ",
                     "velogb10_copy_span_yield_gate ", "velogb10_copy_span_yield_gate_denied_total ",
                     "velogb10_copy_span_yield_gate_probes_total ", "velogb10_copy_span_start_w ",
                     "velogb10_copy_span_narrow_proposals_total{source=\"search\"} ",
                     "velogb10_copy_span_narrow_proposals_total{source=\"cursor\"} "] {
            assert!(o.contains(name), "missing {name}");
        }
    }

    /// The counter fold: deltas of one lane's proposer counters reach the process totals.
    #[test]
    fn copy_span_counter_delta_folds() {
        use std::sync::atomic::Ordering::Relaxed;
        let a = crate::copy_span::CopySpanCounters::default();
        let mut b = a;
        b.proposals_cursor = 3;
        b.miss_below_min = 2;
        b.end_mismatch = 1;
        let (c0, m0, e0) = (CS_PROP_CURSOR.load(Relaxed), CS_MISS_BELOW.load(Relaxed), CS_END_MISMATCH.load(Relaxed));
        copy_span_count_delta(&a, &b);
        assert!(CS_PROP_CURSOR.load(Relaxed) >= c0 + 3 && CS_MISS_BELOW.load(Relaxed) >= m0 + 2
                && CS_END_MISMATCH.load(Relaxed) >= e0 + 1);
    }

    /// Source pins (the GPU path cannot run in the unit suite): a copy round is asked for only on a
    /// solo, non-plain step; it stages the match rule; it commits with the round's own emitted
    /// tokens; and the neural statistics stay out of it.
    #[test]
    fn copy_span_arm_source_pins() {
        let s = src_text!("exl3_serve");
        assert!(s.contains("let copy_prop = if !plain_step && solo { Self::copy_propose(self.copy_arm, self.lanes[s].as_mut(), k_eff) } else { None };"));
        assert!(s.contains("let rq = if plain_step || copy_prop.is_some() { None } else { self.rq_stage(s) };"));
        assert!(s.contains("self.model.set_spec_sampling(&mut self.sc, &rows_c, None)?;"));
        assert!(s.contains("if !copy { self.arm_acc(ARM_SERIAL, out.emitted.len() as f64, ms); }"));
        assert!(s.contains("if !plain_step && !copy {\n            let dds = self.draft_conf > 0.0;"));
        assert!(s.contains("cs.commit(Some(cp), out.a, &lane.copy_new);"));
        // every emission point feeds the proposer
        assert_eq!(s.matches("if lane.copy.is_some() { lane.copy_new.push(").count(), 2 + 1, "round_emit + step_plain (+ this pin)");
        // yield gate: neural rounds are reported from exactly one call site (+ this pin), and only for a
        // non-plain, non-copy round; copy rounds are counted by the proposer's own commit
        let call = ["Self::copy_note_", "neural(&mut lane.copy, out.emitted.len());"].concat();
        assert_eq!(s.matches(call.as_str()).count(), 1, "one neural-round call site");
        assert_eq!(s.matches(["Self::copy_note_", "neural("].concat().as_str()).count(), 1, "one call site");
        assert_eq!(s.matches([".note_neural", "_round("].concat().as_str()).count(), 1, "only copy_note_neural reaches the proposer");
        assert!(s.contains(["if !plain_step && copy_prop.is_none() {\n            if let Some(lane) = self.lanes[s].as_mut() { Self::copy_note_",
                            "neural("].concat().as_str()));
    }

    /// WIDEVERIFY1 TP leg: the replicated-proposer design rests on the proposer being a pure function of
    /// (prompt ids, emitted tokens, the round's width): two independent proposers (two ranks) fed the same
    /// stream at the same program points through `copy_propose` make the same proposal every round, and
    /// a rank fed a different emitted token diverges (which the pre-verify agree then catches).
    #[test]
    fn copy_propose_is_identical_on_every_rank() {
        let arm = CopyArm { cfg: crate::copy_span::CopySpanCfg::new(4), max_w: 10 };
        // a prompt with a repeated block; the "model" re-emits the block (a copy) then free text
        let block: Vec<u32> = (100..140).collect();
        let mut prompt: Vec<u32> = vec![1, 2, 3];
        prompt.extend(&block);
        prompt.extend([7u32, 8, 9]);
        prompt.extend(&block[..5]); // the seam: the answer has started re-writing the block
        let mk = || PLane { copy: Some(Box::new(crate::copy_span::CopySpan::new(&prompt, arm.cfg))), copy_new: Vec::new() };
        let prop = |l: &mut PLane, k_eff: usize| Exl3Scheduler::copy_propose_parts(arm.max_w, &mut l.copy, &mut l.copy_new, k_eff, None);
        let (mut r0, mut r1) = (mk(), mk());
        let mut out: Vec<u32> = block[5..].to_vec();
        out.extend(500..530u32);
        let (mut i, mut rounds, mut copies) = (0usize, 0usize, 0usize);
        // emulate rounds: k_eff varies like the scheduler's clamps; a copy round emits accepted + 1
        // tokens and commits with commit(Some(..)); other rounds feed copy_new (committed lazily)
        while i < out.len() {
            let k_eff = [10usize, 7, 3, 10, 1][rounds % 5];
            let p0 = prop(&mut r0, k_eff);
            let p1 = prop(&mut r1, k_eff);
            assert_eq!(p0, p1, "round {rounds}: ranks proposed differently");
            let (emitted, a) = match p0.as_ref() {
                Some(pr) => {
                    copies += 1;
                    let a = pr.tokens.iter().zip(&out[i..]).take_while(|(x, y)| x == y).count();
                    (out[i..(i + a + 1).min(out.len())].to_vec(), a)
                }
                None => (out[i..(i + 2).min(out.len())].to_vec(), 0),
            };
            for r in [&mut r0, &mut r1] {
                r.copy_new.extend(&emitted);
                if let Some(pr) = p0.as_ref() {
                    r.copy.as_mut().unwrap().commit(Some(pr), a.min(emitted.len()), &r.copy_new);
                    r.copy_new.clear();
                }
            }
            i += emitted.len();
            rounds += 1;
        }
        assert!(copies > 0, "the repeated block must start a copy");
        assert_eq!(r0.copy.as_ref().unwrap().ctx(), r1.copy.as_ref().unwrap().ctx());
        assert_eq!(r0.copy.as_ref().unwrap().counters(), r1.copy.as_ref().unwrap().counters());
        // a rank that saw a different token proposes differently from here (the agree's job to catch)
        let (mut a, mut b) = (mk(), mk());
        a.copy_new.extend(&block[5..12]);
        b.copy_new.extend(&block[5..11]);
        b.copy_new.push(999_999);
        let (pa, pb) = (prop(&mut a, 10), prop(&mut b, 10));
        assert!(pa.is_some(), "the agreeing stream keeps copying");
        assert_ne!(pa, pb);
        // the arm off (None) or a lane without a proposer: never a proposal, at any width
        let mut c = mk();
        c.copy = None;
        c.copy_new.extend(&block[5..12]);
        assert_eq!(prop(&mut c, 10), None);
        assert_eq!(prop(&mut mk(), 0), None, "k_eff 0 (a plain step) proposes nothing");

        // the yield gate: two ranks fed the same agreed round stream (copy rounds through commit, neural
        // rounds through copy_note_neural) hold the same gate state and make the same decision every round.
        // Agent-echo shape: the output re-quotes the block in pieces, departing after ~5 tokens each time,
        // while neural rounds emit 8, so the gate suspends and later probes.
        let garm = CopyArm { cfg: copy_arm_cfg(4, 10, true, 10), max_w: 10 };
        let gmk = || PLane { copy: Some(Box::new(crate::copy_span::CopySpan::new(&prompt, garm.cfg))), copy_new: Vec::new() };
        let gprop = |l: &mut PLane, k_eff: usize| Exl3Scheduler::copy_propose_parts(garm.max_w, &mut l.copy, &mut l.copy_new, k_eff, None);
        let mut out: Vec<u32> = Vec::new();
        let mut fresh = 600_000u32;
        for rep in 0..40 {
            let st = (rep * 3) % 30;
            out.extend(&block[st..st + 10]); // quote 10 tokens (a 4-token match starts a copy)...
            out.push(fresh); // ...then depart from the source
            fresh += 1;
            out.extend(fresh..fresh + 8); // free text the neural drafter serves
            fresh += 8;
        }
        let (mut g0, mut g1) = (gmk(), gmk());
        let (mut i, mut rounds, mut copies, mut neural) = (0usize, 0usize, 0usize, 0usize);
        while i < out.len() {
            let k_eff = [10usize, 7, 10, 10, 3][rounds % 5];
            let p0 = gprop(&mut g0, k_eff);
            let p1 = gprop(&mut g1, k_eff);
            assert_eq!(p0, p1, "gate round {rounds}: ranks proposed differently");
            assert_eq!(g0.copy.as_ref().unwrap().gate(), g1.copy.as_ref().unwrap().gate(), "gate round {rounds}");
            let (emitted, a) = match p0.as_ref() {
                Some(pr) => {
                    copies += 1;
                    let a = pr.tokens.iter().zip(&out[i..]).take_while(|(x, y)| x == y).count().min(5);
                    (out[i..(i + a + 1).min(out.len())].to_vec(), a)
                }
                None => { neural += 1; (out[i..(i + 8).min(out.len())].to_vec(), 0) }
            };
            for r in [&mut g0, &mut g1] {
                r.copy_new.extend(&emitted);
                match p0.as_ref() {
                    Some(pr) => {
                        r.copy.as_mut().unwrap().commit(Some(pr), a.min(emitted.len()), &r.copy_new);
                        r.copy_new.clear();
                    }
                    None => Exl3Scheduler::copy_note_neural(&mut r.copy, emitted.len()),
                }
            }
            i += emitted.len();
            rounds += 1;
        }
        let (c0, c1) = (g0.copy.as_ref().unwrap(), g1.copy.as_ref().unwrap());
        assert_eq!(c0.gate(), c1.gate());
        assert_eq!(c0.counters(), c1.counters());
        assert!(copies >= 2 && neural > 0, "copies {copies} neural {neural}");
        assert!(c0.counters().gate_denied > 0, "the poor copies suspend the gate: {:?} {:?}", c0.gate(), c0.counters());
        assert!(c0.gate().nn > 0 && c0.gate().nc >= 2);

        // WIDEVERIFY1 refinement 1 (start ramp) and 2 (wide copy width): two ranks fed the same agreed
        // stream through copy_propose_parts make the same proposals — widths included — every round, for
        // the default width (max_w 10) and the wide one (max_w 15), with the depth varying like the
        // scheduler's clamps. Copy-heavy shape: long quotes of the block with an occasional departure.
        for (max_w, start_w) in [(10usize, 4usize), (15, 4), (15, 15), (10, 1)] {
            let rarm = CopyArm { cfg: copy_arm_cfg(4, max_w, false, start_w), max_w };
            let rmk = || PLane { copy: Some(Box::new(crate::copy_span::CopySpan::new(&prompt, rarm.cfg))), copy_new: Vec::new() };
            let rprop = |l: &mut PLane, k: usize| Exl3Scheduler::copy_propose_parts(rarm.max_w, &mut l.copy, &mut l.copy_new, k, None);
            let mut out: Vec<u32> = block[5..].to_vec();
            out.extend(&block[..]);
            out.push(777_777);
            out.extend(&block[..30]);
            let (mut r0, mut r1) = (rmk(), rmk());
            let (mut i, mut rounds, mut widths) = (0usize, 0usize, Vec::new());
            while i < out.len() {
                let k = [15usize, 10, 15, 7, 15, 3][rounds % 6];
                let (p0, p1) = (rprop(&mut r0, k), rprop(&mut r1, k));
                assert_eq!(p0, p1, "ramp max_w {max_w} start_w {start_w} round {rounds}: ranks proposed differently");
                let (emitted, a) = match p0.as_ref() {
                    Some(pr) => {
                        assert!(pr.tokens.len() <= copy_width(k, max_w) && pr.tokens.len() <= pr.ramp,
                                "round {rounds}: {} tokens > width {} / ramp {}", pr.tokens.len(), copy_width(k, max_w), pr.ramp);
                        widths.push(pr.tokens.len());
                        let a = pr.tokens.iter().zip(&out[i..]).take_while(|(x, y)| x == y).count();
                        (out[i..(i + a + 1).min(out.len())].to_vec(), a)
                    }
                    None => (out[i..i + 1].to_vec(), 0),
                };
                for r in [&mut r0, &mut r1] {
                    r.copy_new.extend(&emitted);
                    if let Some(pr) = p0.as_ref() {
                        r.copy.as_mut().unwrap().commit(Some(pr), a.min(emitted.len()), &r.copy_new);
                        r.copy_new.clear();
                    }
                }
                i += emitted.len();
                rounds += 1;
            }
            let (c0, c1) = (r0.copy.as_ref().unwrap(), r1.copy.as_ref().unwrap());
            assert_eq!(c0.counters(), c1.counters());
            assert!(widths.len() >= 3, "max_w {max_w} start_w {start_w}: copies ran ({widths:?})");
            assert!(*widths.iter().max().unwrap() <= copy_depth_cap(max_w));
            if start_w < max_w {
                assert!(c0.counters().narrow_search >= 1, "the first round of a copy is narrow: {:?}", c0.counters());
                assert_eq!(widths[0], start_w, "a copy starts at start_w: {widths:?}");
            } else {
                assert_eq!(c0.counters().narrow_search + c0.counters().narrow_cursor, 0, "no ramp at start_w >= max_w");
            }
            if max_w == 15 && start_w == 15 {
                assert!(widths.contains(&15), "the wide copy verify proposes 15 at depth 15: {widths:?}");
            }
        }
    }

    /// The two lane fields `copy_propose` reads.
    struct PLane {
        copy: Option<Box<crate::copy_span::CopySpan>>,
        copy_new: Vec<u32>,
    }

    /// WIDEVERIFY1 TP leg source pins: the copy round's TP agreement runs before the round launches,
    /// with 0 head passes, only when the arm proposed; the lockstep label names the copy round; the
    /// boot resolves the copy arm from the TP world on every rank.
    #[test]
    fn copy_span_tp_source_pins() {
        let s = src_text!("exl3_serve");
        let agree = s.find(["t.pre_verify(drafts.len(), &drafts, 0)", "?;"].concat().as_str()).expect("the copy round's pre-verify");
        let launch = s.find(["self.model.copy_round(&mut self.sc, head, s, b, p, tap_row, drafts, t0)", "?"].concat().as_str())
            .expect("the copy round launch");
        assert!(agree < launch, "the agree precedes the launch");
        assert!(s.contains("if plain_step { \"plain-round\" } else if copy_prop.is_some() { \"copy-round\" } else { \"mtp-round\" }"));
        assert!(s.contains(["copy_span_resolve(if tp_rank.is_some() { (tp_world as usize).max(2) } else { 1 }, spec_live)", "?;"].concat().as_str()));
    }

    /// COPYPF1 source pins (the step needs a GPU): the arm is reached from ONE place, behind the resolved flag
    /// (off = no call, every round as before); every other emit keeps its row information; the two harnesses never
    /// arm it; the boot resolves it from the same TP world as the copy arm.
    /// COPYLANES (--copy-span-lanes): the busy-step hook is reached only through the resolved flag with two or more busy
    /// lanes; the two harnesses never arm it; the lone path keeps `solo` for the DFlash2 ring and passes `copy_ok` only to
    /// the copy proposal and the stretch pick; the batch price never reads a rank-local wall clock (TP lockstep).
    #[test]
    fn copy_span_lanes_source_pins() {
        let s = src_text!("exl3_serve");
        let hook = ["if self.cl_on && mtp_on && slots.len() >= 2 {\n", "            let took = self.cl_step(&slots)?;"].concat();
        assert_eq!(s.matches(hook.as_str()).count(), 1, "one hook, behind the resolved flag");
        assert_eq!(s.matches(["self.cl_", "step(&"].concat().as_str()).count(), 2, "cl_step has one caller (+ this pin's hook text)");
        assert_eq!(s.matches(["cl_on: false, cl_priced: true, cl_price: Default::default(), // ", "COPYLANES"].concat().as_str()).count(), 2,
                   "both harnesses keep it off");
        assert_eq!(s.matches(["self.step_mtp_ex(s, false, Some(", "batch))?;"].concat().as_str()).count(), 1);
        assert!(s.contains("let copy_ok = solo || busy_copy.is_some();"));
        assert!(s.contains("} else if solo && self.df2.as_ref().is_some_and(|r| r.nprev() + 1 == p) {"), "DF2 keeps solo");
        let b = &s[s.find(["fn cl_", "batch_rate(&self"].concat().as_str()).unwrap()..];
        let b = &b[..b.find("\n    }\n").unwrap()];
        assert!(!b.contains("ema_wall") && !b.contains("Instant"), "the price reads only replicated inputs");
        // every busy-step note of copy costs is behind the flag
        assert_eq!(s.matches(["self.cl_price.work.", "note("].concat().as_str()).count(), 2);
        assert_eq!(s.matches(["if self.cl_on && !plain_", "step {"].concat().as_str()).count(), 1);
        assert_eq!(s.matches(["        if self.cl_on {\n            // COPYLANES: a stretch is ", "copy work"].concat().as_str()).count(), 1);
    }

    #[test]
    fn copy_span_prefill_source_pins() {
        let s = src_text!("exl3_serve");
        let call = ["return self.step_", "cpv(s, n, rung, b, p, tap_row);"].concat();
        assert_eq!(s.matches(call.as_str()).count(), 1, "one call site");
        let i = s.find(call.as_str()).unwrap();
        let ctx = &s[i.saturating_sub(300)..i];
        assert!(ctx.contains("if self.cpv_on {") && ctx.contains("self.cpv_pick(s, copy_ok, plain_step, p, remaining, think_room, busy_copy)"),
                "the call sits behind the resolved flag and the pure decision");
        let off = ["self.round_emit_rows(s, &ro, None, ", "false)"].concat();
        assert_eq!(s.matches(off.as_str()).count(), 1, "only the stretch emits without row information");
        assert!(s.contains("self.round_emit_rows(s, out, packed_row0, true)"), "round_emit keeps it for every other round");
        assert_eq!(s.matches(["cpv_on: false, cpv: ", "None, // COPYPF1"].concat().as_str()).count(), 2, "both harnesses");
        assert!(s.contains("let cpv_on = cpv_st == crate::copy_prefill::State::On;"));
        assert!(s.contains("copy_arm.is_some(),\n                                              if tp_rank.is_some() { (tp_world as usize).max(2) } else { 1 }, df2.is_some());"));
        // the stretch consumes every sampler row it judged (rows past the first disagreement are never reused)
        assert!(s.contains("lane.ctr = lane.ctr.wrapping_add((nn + 1) as u32);"));
    }
