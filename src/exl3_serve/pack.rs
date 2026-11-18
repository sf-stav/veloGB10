//! The lane-pack and wide-verify arms: `pack_plan`, `step_packed`, `step_pair` (two lanes in one verify), `wide_lanes` / `step_wide` / `wide_round`,
//! the arm meters (`arm_acc`, `arm_fresh`, `use_shared_plain`) and the boot-time pack warnings. Another `impl Exl3Scheduler` block.
//! A pair round has lockstep points in a fixed order under TP (drafts, drain, pre-verify exchange, verify, finish, post-verify agree); do not reorder.

use super::*;

/// S-B9-REL-INTEG: the boot warning of lane packing (on by default since F10) under non-default penalty defaults (None = the defaults are
/// the no-op 1.0 / 0 / 0, so an unpenalized request packs). A request that sends no penalty field inherits the
/// server default, so with `--default-presence-penalty 1.5` EVERY such request is penalized and none of them packs
/// (`Skip::Penalty`); a request that sends `presence_penalty: 0` (and the other two neutral) overrides the default
/// and does pack. This is the owner's posture: his hand test showed 0 packed rounds for exactly this reason.
pub(super) fn lane_pack_penalty_warning(rep: f32, pres: f32, freq: f32) -> Option<String> {
    if crate::exl3_forward::PenParams::new(rep, pres, freq, 1, 0).is_none() { return None; }
    Some(format!("[exl3-serve] lane-pack WARNING: this server's penalty defaults are repetition {rep}, presence {pres}, frequency {freq} \
(--default-repetition-penalty / --default-presence-penalty / --default-frequency-penalty). A request that sends no penalty \
field inherits them and is a PENALIZED request, and a penalized request never joins a packed verify (it runs its own \
lone round: /metrics velogb10_lane_pack_skip_total{{reason=\"penalty\"}}). With these defaults lane packing (--lane-pack, on by default) changes \
nothing for such requests; a request that sends presence_penalty 0, frequency_penalty 0 and repetition_penalty 1 itself \
does pack. The packed arm for penalized requests is not built"))
}

/// May SAMPLED lanes join a packed verify under TP? `--lane-pack-sampled-tp on|off`, default ON
/// (owner 2026-10-10: real requests are sampled, and a feature is not kept away from them because
/// its gates have not run: his TP=4 hand test showed 0 packed rounds and 898 `skip sampled`).
/// The sampled tail of a packed verify under TP (the match-rule and real-q families keyed by their
/// vocabulary-parallel tail) was built by S-B9-CF-PACK5 (`lane_pack::fams`, `FwdModel::pack_fam_key`)
/// and has NOT run its hardware gates. `off` = a sampled lane runs its own lone round under TP
/// (counted and logged as `skip sampled`). Spmd scope: the head ships the value, and the boot agree
/// carries `pack_sampled`, so ranks cannot differ.
pub(super) fn pack_sampled_under_tp() -> bool {
    crate::opts::choice(crate::opt!("lane-pack-sampled-tp")) != "off"
}

impl Exl3Scheduler {
    /// S-B9-CF-PACK4: this step's packing (`lane_pack::plan`), or None = the flag is off or
    /// fewer than two lanes are busy, and the caller runs master's loops untouched. Every input
    /// is lane state plus the rotation counter (which advances once per planned step).
    pub(super) fn pack_plan(&mut self, slots: &[usize]) -> Option<crate::lane_pack::Plan> {
        use crate::lane_pack::{lane_block, LaneFacts, LaneView, Skip};
        if !self.lane_pack.on() || slots.len() < 2 { return None; }
        let views: Vec<LaneView> = slots.iter().map(|&s| {
            let l = self.lanes[s].as_ref().unwrap();
            let (k_pre, kd_reg) = self.round_depth(l.pos, l.max_new.saturating_sub(l.generated), l.think_room());
            let facts = LaneFacts {
                mtp_armed: l.mtp.is_some(),
                force_control: l.force_control,
                probing: l.reprobe.phase == crate::mtp_reprobe::Phase::Probing,
                sampled: l.samp_rows(1)[0].is_some(),
                // PACK4S: sampled lanes pack (real-q and match rule); only the lone-verify
                // sampler dump (--wp24-dump records the lone round's rows) keeps them out; and
                // (PACK5) a topology whose packed sampled tail is not gated (TP=2)
                sampled_lone: crate::wp24::dump_on() || !self.pack_sampled,
                penalized: l.pen.is_some(),
                image: l.img,
                logprobs_tp: l.logprobs.is_some() && self.tp.is_some(),
                depth: 0, // set by lane_view from kd_reg (capped at PACK_DEPTH_MAX: PACK4-REVIEW A4)
            };
            // The verify's regime is its LAST row's (p + w for the w <= kd drafts the DDS keeps).
            // With the per-row regime on, a round never straddles the switch, so every w gives
            // the regime of p + kd; with --qsa-row-regime 0 a straddling round's regime depends
            // on w, which is only known after drafting: such a lane does not pack (and the boot
            // refuses that posture, PACK4-REVIEW B3). `lane_view` is the pure, unit-tested rule.
            crate::lane_pack::lane_view(s, facts, l.pos, k_pre, kd_reg, self.in_prefill, self.pack_ready,
                                        &|p| self.model.qsa_live(p))
        }).collect();
        let plan = crate::lane_pack::plan(self.lane_pack, &views, self.pack_rot);
        if plan.is_some() { self.pack_rot += 1; }
        plan
    }

    /// One lane round ran outside a pack: count it, and log the first few (64 lines in all, at
    /// most 8 per reason so one common reason cannot use the whole budget).
    fn note_skip(&mut self, s: usize, why: crate::lane_pack::Skip) {
        crate::metrics::lane_pack_skip(why.index());
        if self.pack_logs < 64 && crate::metrics::lane_pack_skips(why.index()) <= 8 {
            self.pack_logs += 1;
            crate::dreprintln!("[exl3-serve] lane-pack skip: {} (slot {s}; line {}/64, every skip is counted: /metrics \
velogb10_lane_pack_skip_total{{reason=\"{}\"}})", why.text(), self.pack_logs, why.label());
        }
    }

    /// S-B9-CF-PACK4: one scheduler step under `--lane-pack on` with at least two busy lanes.
    /// Lanes are visited in slot order, as in the serial loop: a lane that is the first slot of
    /// a planned pair runs the packed round with its partner; every other lane runs exactly
    /// what the serial loop runs for it (its lone speculative round, or the plain batch for
    /// MTP-off lanes), with the reason it stayed out of a pack counted.
    pub(super) fn step_packed(&mut self, slots: &[usize], plan: crate::lane_pack::Plan) -> Result<()> {
        let mut plain: Vec<usize> = Vec::new();
        for &s in slots {
            match plan.pair_of(s) {
                Some([a, b]) => {
                    if s == a { // b ran with a (slots ascend, a < b)
                        if let Err(e) = self.step_pair(a, b) {
                            // PACK4-REVIEW A3: an error out of the PACKED part of the round leaves
                            // the arm off for the rest of the boot (the caller still fails every
                            // live lane, as for any decode-step error); the control-cadence
                            // fallback's serial rounds are the lone path's own and do not.
                            if std::mem::take(&mut self.pack_stats.in_round) { self.pack_disable(a, b, &e); }
                            return Err(e);
                        }
                    }
                }
                None => {
                    if let Some(why) = plan.skip_of(s) { self.note_skip(s, why); }
                    if self.lanes[s].as_ref().map_or(false, |l| l.mtp.is_some()) {
                        self.step_mtp(s, false)?; // packed arm: never a lone lane
                    } else if self.lanes[s].is_some() {
                        plain.push(s);
                    }
                }
            }
        }
        if !plain.is_empty() {
            self.step_plain(&plain, false)?;
        }
        // H3 (round COMPLETE): written by `step`, the wrapper, for every arm of `step_rounds` (this one included)
        Ok(())
    }

    /// LANEWIDE: ONE wide round (module note of exl3_forward/lwv.rs): per lane the lone round's draft half (the
    /// drafts of all lanes, then the real-q lists handed to their row offsets), ONE wide verify (trunk, judge, the
    /// per-segment commit of the accepted rows, taps into each lane's window), then per lane the head re-prime, the
    /// lone round's bookkeeping (fed the lane's SHARE of the round time) and its emit loop.
    fn wide_round(&mut self, elig: &[usize]) -> Result<()> {
        use crate::lane_pack::{lane_share_ms, SegSamp};
        let k = self.mtp_k;
        let n = elig.len();
        if self.wide.bufs.is_none() {
            self.wide.bufs = Some(self.model.wide_bufs(self.wide.rows_max)?);
        }
        let cap = crate::lane_wide::depth_cap(n, self.wide.rows_max).min(crate::exl3_forward::MTP_MAX_K);
        let pre: Vec<(usize, i32, usize, usize, usize)> = elig.iter().map(|&s| {
            let l = self.lanes[s].as_ref().unwrap();
            let (_, kd) = self.round_depth(l.pos, l.max_new.saturating_sub(l.generated), l.think_room());
            // batched drafts have no confidence stop: the depth follows the lane's own tokens-per-round EMA
            let auto = if crate::lane_wide::batched_draft() { crate::lane_wide::auto_depth(l.ema_tok) } else { usize::MAX };
            (s, l.last_tok as i32, l.pos, l.mtp.unwrap(), kd.min(cap).min(auto).max(1))
        }).collect();
        self.ctrl_round += n;
        let head = self.model.mtp.as_ref().unwrap();
        let t0 = std::time::Instant::now();
        let g0 = self.model.graph_count();
        let dds = self.draft_conf > 0.0 && !crate::lane_wide::batched_draft();
        let mut drafts: Vec<crate::exl3_forward::PackDraft> = Vec::with_capacity(n);
        let mut at = 0usize;
        if crate::lane_wide::batched_draft() {
            let lanes: Vec<crate::exl3_forward::WideLane> = pre.iter().map(|&(s, b, p, tap_row, kd)| {
                let (rows, rq) = (self.lanes[s].as_ref().unwrap().samp_rows(1), self.rq_stage(s));
                crate::exl3_forward::WideLane { slot: s, b, p, tap_row, k: kd,
                    samp: SegSamp { samp: rows[0], ratio: rq.map_or(false, |r| r.mode == 2) }, rq: rq.filter(|r| r.mode == 2) }
            }).collect();
            let wb = self.wide.bufs.as_mut().unwrap();
            drafts = self.model.wide_draft_batch(&mut self.sc, wb, &lanes)?;
        }
        for &(s, b, p, tap_row, kd) in pre.iter().filter(|_| !crate::lane_wide::batched_draft()) {
            let (rows, rq) = (self.lanes[s].as_ref().unwrap().samp_rows(k + 1), self.rq_stage(s));
            let seg = SegSamp { samp: rows[0], ratio: rq.map_or(false, |r| r.mode == 2) };
            self.model.set_spec_sampling(&mut self.sc, &rows, rq)?;
            self.model.set_penalties(&mut self.sc, &vec![None; k + 1])?;
            let d = if dds {
                let target = {
                    let lane = self.lanes[s].as_mut().unwrap();
                    let t = wp23_target(self.draft_conf, wp23_guard_on(k), lane.mtp_rounds,
                                        lane.wp23.fit.slope(), lane.ema_tok, lane.ema_ms);
                    lane.wp23.target = t;
                    t
                };
                let cal = &self.cal;
                let mut reach = 1.0f64;
                let mut keep = |i: usize, c: f32| { reach *= cal.estimate_at(i, c); reach >= target };
                self.model.pack_draft(&mut self.sc, head, s, kd, b, p, tap_row, Some(&mut keep), seg, rq)?
            } else {
                self.model.pack_draft(&mut self.sc, head, s, kd, b, p, tap_row, None, seg, rq)?
            };
            let wb = self.wide.bufs.as_mut().unwrap();
            self.model.wide_list_handoff(&self.sc, wb, at, &d)?;
            at += d.drafts.len() + 1;
            drafts.push(d);
        }
        // LANEWIDE-TP: the wide round's ONE pre-verify lockstep point (the packed round's, over every lane's segment):
        // the round was planned from replicated lane state, so a rank that built a different set of segments (slots,
        // positions, bonus tokens, samplers, drafts, rows) fails here, before the verify's collectives launch.
        if self.tp.is_some() && tp_preverify_on() {
            use crate::lane_pack::PreSeg;
            // the batched draft's head passes are collectives (sharded draft head): their doorbell epochs may still be in
            // flight when the host exchange below stages in send slot 0 / receives in recv slot gen & 7 of the SAME rings
            // (the wasted-pass hazard of pack_drain_wasted, here after EVERY batched draft): device sync + proxy drain
            self.model.tp_drain_before_exchange()?;
            let segs: Vec<PreSeg> = drafts.iter().map(|d| PreSeg::of(d.slot, d.p, d.b, &d.drafts, &d.samp, d.launched, d.wasted)).collect();
            let words = crate::lane_pack::pre_words(&segs);
            let r = crate::lane_pack::pack_rows_total(&segs);
            let launched: usize = drafts.iter().map(|d| d.launched).sum();
            let t = self.tp.as_mut().context("lane-wide: TP state missing")?;
            t.pre_verify_wide(r, segs.len(), &words, launched)?;
        }
        let t1 = std::time::Instant::now();
        let wb = self.wide.bufs.as_mut().unwrap();
        let (outs, wm) = self.model.wide_verify(&mut self.sc, wb, &drafts)?;
        let t2 = std::time::Instant::now();
        for j in 0..n {
            self.model.pack_finish(&mut self.sc, head, &drafts[j], &outs[j])?;
        }
        let t3 = std::time::Instant::now();
        let captured_local = self.model.graph_count() != g0;
        let ms_of = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
        // LANEWIDE-TP: the POST-verify lockstep point (the packed round's): agree every lane's outcome (accept count,
        // rows, emitted ids, drafts) with the lane count as the width, and TAKE THE HEAD's round ms and capture flag, so
        // the lanes' EMAs, the depth choice and the wide policy's measured rates are the same on every rank. No ident
        // digests here (the wide rows live in the wide scratch, wider than the served one).
        let (ms_round, captured) = if let Some(t) = self.tp.as_mut() {
            use crate::lane_pack::PostLane;
            let lanes: Vec<PostLane> = (0..n).map(|j| PostLane {
                accepted: outs[j].a, rows: drafts[j].drafts.len() + 1,
                emitted: &outs[j].emitted, drafts: &outs[j].drafts, confs: &drafts[j].confs }).collect();
            let w = crate::lane_pack::post_words(&lanes);
            let acc: usize = outs.iter().map(|o| o.a).sum();
            self.model.tp_drain_before_exchange()?; // the wide verify's collectives (the same ring-aliasing rule)
            let (ms, _, cap) = Self::tp_round_with(t, &self.model, None, "wide-round", acc, n, &w, ms_of(t0, t3), 0.0, captured_local)?;
            (ms, cap)
        } else { (ms_of(t0, t3), captured_local) };
        let ms_lane = lane_share_ms(ms_round, n);
        let rows: usize = drafts.iter().map(|d| d.drafts.len() + 1).sum();
        let toks: usize = outs.iter().map(|o| o.a + 1).sum();
        let acc: usize = outs.iter().map(|o| o.a).sum();
        crate::lane_wide::count_round(n, rows, toks, acc, rows - n, ms_round,
                                      [ms_of(t0, t1), wm.stage + wm.trunk, wm.judge, wm.commit, ms_of(t2, t3)]);
        for j in 0..n {
            for (i, &c) in drafts[j].confs.iter().enumerate() {
                if i < outs[j].a { self.cal.add_label(i, c, true); } else { self.cal.add_label(i, c, false); break; }
            }
            if dds { self.cal.decay_step(); }
        }
        for j in 0..n {
            let s = pre[j].0;
            self.round_account(s, &outs[j], ms_lane, 0.0, captured, false, false, drafts[j].samp.ratio, false, true, false)?;
            self.round_emit_rows(s, &outs[j], None, false)?;
        }
        Ok(())
    }

    /// S-B9-CF-PACK4: ONE packed speculative round for the lanes in slots `sa` < `sb` (design
    /// 3.2): each request drafts on the served scratch with the lone round's own draft half;
    /// ONE packed verify (graph replay on the packed scratch) accepts and commits per segment;
    /// then, per request, the head re-prime, the lone round's bookkeeping (`round_account`, fed
    /// the request's SHARE of the round time) and its emit loop (`round_emit`). Both lanes are
    /// unpenalized, MTP-armed, on one side of the sparse switch, with depth >= 1.
    /// S-B9-CF-PACK4S: each lane is greedy or sampled (real-q under `--spec-sampling ratio`,
    /// else the match rule), in any combination. A lane drafts under ITS OWN sampler staging,
    /// exactly as its lone round stages it; a real-q lane's draft lists go to its row offset
    /// of the packed scratch before the next lane drafts; the verify stages one sampler block
    /// per row and the lane's counters advance as in its lone round.
    /// LANEWIDE: the busy lanes a wide round can take this step (in slot order) and the rest, each kept-out lane's
    /// reason counted. A lane owing its periodic control round stays out (its lone round runs the control).
    pub(super) fn wide_lanes(&mut self, slots: &[usize]) -> (Vec<usize>, Vec<usize>) {
        use crate::lane_wide::{lane_skip, LaneFacts};
        let af = crate::exl3_forward::a_flags();
        let (mut elig, mut rest) = (Vec::new(), Vec::new());
        let mut ctr = self.ctrl_round;
        for &s in slots {
            let l = self.lanes[s].as_ref().unwrap();
            let (_, kd) = self.round_depth(l.pos, l.max_new.saturating_sub(l.generated), l.think_room());
            let cost = if l.ema_tok > 0.0 { l.ema_ms / l.ema_tok } else { 0.0 };
            let per = crate::lane_pack::ctrl_period(af.ctrl == 0, self.ema_plain, cost);
            let owes = af.ctrl != 3 && l.mtp.is_some() && (ctr + 1) % per.max(1) == 0;
            let facts = LaneFacts {
                mtp_armed: l.mtp.is_some(),
                control: l.force_control || owes || l.reprobe.phase == crate::mtp_reprobe::Phase::Probing,
                penalized: l.pen.is_some(),
                logprobs: l.logprobs.is_some(),
                image: l.img,
                sparse: self.model.qsa_live(l.pos + kd.max(1)),
                depth: kd,
                in_prefill: self.in_prefill,
            };
            match lane_skip(&facts) {
                None => { elig.push(s); ctr += 1; }
                Some(why) => { crate::lane_wide::count_skip(why); rest.push(s); }
            }
        }
        (elig, rest)
    }

    /// LANEWIDE: one step with the wide arm: ONE wide round for `elig` (>= 2 lanes), then today's rounds for `rest`
    /// (their lone speculative rounds, or one plain batch for MTP-off lanes).
    pub(super) fn step_wide(&mut self, elig: &[usize], rest: &[usize]) -> Result<()> {
        if let Err(e) = self.wide_round(elig) {
            if !self.wide.disabled {
                self.wide.disabled = true;
                crate::reprintln!("[exl3-serve] LANE-WIDE DISABLED for the rest of this boot: a wide round of slots {elig:?} failed ({e:#}); \
every busy lane of this step fails with it and every later step runs today's arms. Restart the server to re-enable it");
            }
            return Err(e);
        }
        let mut plain: Vec<usize> = Vec::new();
        for &s in rest {
            if self.lanes[s].as_ref().map_or(false, |l| l.mtp.is_some()) {
                self.step_mtp(s, false)?;
            } else if self.lanes[s].is_some() {
                plain.push(s);
            }
        }
        if !plain.is_empty() { self.step_plain(&plain, false)?; }
        Ok(())
    }

    /// PACK4-REVIEW A3: a packed round failed. The caller fails every live lane (a decode-step error
    /// always does: after an arbitrary error mid-round the shared staging — the pinned stage's
    /// in-flight copies, `sc.rp.pending`, the speculative-pass state — is not proven clean, and
    /// only the pair's own state could be spared, which this does not attempt). Without this the
    /// arm stayed on and a deterministic failure repeated for every later concurrent pair until
    /// a restart. Now the server degrades to the serial arm for the rest of the boot: one loud
    /// line, `velogb10_lane_pack_disabled 1`, and every later round is master's.
    fn pack_disable(&mut self, a: usize, b: usize, why: &anyhow::Error) {
        if !self.lane_pack.on() { return; }
        self.lane_pack = crate::lane_pack::LanePack::Off;
        crate::metrics::lane_pack_disabled();
        crate::reprintln!("[exl3-serve] LANE-PACK DISABLED for the rest of this boot: the packed round of slots {a} + {b} failed ({why:#}); \
every busy lane of this step fails with it, and every later round runs the serial arm (--lane-pack off semantics; \
/metrics velogb10_lane_pack_disabled 1). Restart the server to re-enable packing");
    }

    fn step_pair(&mut self, sa: usize, sb: usize) -> Result<()> {
        use crate::lane_pack::{ctrl_period, lane_share_ms, pair_owes_control, PostLane, PreSeg, SegSamp, Skip};
        let k = self.mtp_k;
        let af = crate::exl3_forward::a_flags();
        // (slot, bonus token, position, tap row, depth, cost) of each lane, before the round
        let pre: Vec<(usize, i32, usize, usize, usize, f64)> = [sa, sb].iter().map(|&s| {
            let l = self.lanes[s].as_ref().unwrap();
            let (_, kd) = self.round_depth(l.pos, l.max_new.saturating_sub(l.generated), l.think_room());
            let kd = kd.min(crate::lane_pack::PACK_DEPTH_MAX); // PACK4-REVIEW A4: the plan's own depth
            (s, l.last_tok as i32, l.pos, l.mtp.unwrap(), kd,
             if l.ema_tok > 0.0 { l.ema_ms / l.ema_tok } else { 0.0 })
        }).collect();
        // The control cadence is the serial arm's, count for count: the pair would consume the
        // counter values ctrl_round + 1 and + 2; if either is a control round for its lane, the
        // two lanes run serial this step (the control round itself is step_mtp's).
        let (per_a, per_b) = (ctrl_period(af.ctrl == 0, self.ema_plain, pre[0].5), ctrl_period(af.ctrl == 0, self.ema_plain, pre[1].5));
        if af.ctrl != 3 && pair_owes_control(self.ctrl_round, per_a, per_b) {
            self.note_skip(sa, Skip::Control);
            self.note_skip(sb, Skip::Control);
            self.step_mtp(sa, false)?; // pair control: two lanes in this step
            return self.step_mtp(sb, false);
        }
        self.pack_stats.in_round = true; // PACK4-REVIEW A3: cleared when the round completes
        let xcheck = crate::exl3_forward::a_xcheck_on();
        if xcheck {
            for &(s, _, _, tap_row, _, _) in &pre { self.model.tap_xcheck_verify(&self.sc, s, tap_row)?; }
        }
        self.ctrl_round += 2;
        let head = self.model.mtp.as_ref().unwrap();
        let t0 = std::time::Instant::now();
        let g0 = self.model.graph_count(); // a round that captures a graph is not cost-fit
        // 1. draft: per request, sequential — the lone round's draft half, under the lone
        // round's own staging of the served scratch (which may still carry another lane's):
        // the lane's sampler rows and, for a real-q lane, its draft sampler (the draft tail and
        // the pass-graph keys read these), and no penalties (a penalized lane does not pack).
        let dds = self.draft_conf > 0.0;
        let mut drafts: Vec<crate::exl3_forward::PackDraft> = Vec::with_capacity(2);
        let mut at = 0usize; // the next segment's first row in the packed verify
        for &(s, b, p, tap_row, kd, _) in &pre {
            let (rows, rq) = (self.lanes[s].as_ref().unwrap().samp_rows(k + 1), self.rq_stage(s));
            let seg = SegSamp { samp: rows[0], ratio: rq.map_or(false, |r| r.mode == 2) };
            self.model.set_spec_sampling(&mut self.sc, &rows, rq)?;
            self.model.set_penalties(&mut self.sc, &vec![None; k + 1])?;
            let d = if dds {
                let target = {
                    let lane = self.lanes[s].as_mut().unwrap();
                    let t = wp23_target(self.draft_conf, wp23_guard_on(k), lane.mtp_rounds,
                                        lane.wp23.fit.slope(), lane.ema_tok, lane.ema_ms);
                    lane.wp23.target = t;
                    t
                };
                let cal = &self.cal;
                let mut reach = 1.0f64;
                let mut keep = |i: usize, c: f32| { reach *= cal.estimate_at(i, c); reach >= target };
                self.model.pack_draft(&mut self.sc, head, s, kd, b, p, tap_row, Some(&mut keep), seg, rq)?
            } else {
                // --draft-confidence 0: the fixed-depth chain of the round's depth (the lone mtp_round's)
                self.model.pack_draft(&mut self.sc, head, s, kd, b, p, tap_row, None, seg, rq)?
            };
            // the real-q list hand-off, issued BEFORE the next lane drafts: both chains write
            // the served list rows 0..w (no-op for a greedy or match-rule lane)
            let scp = self.scp.as_mut().context("lane-pack: the packed scratch is missing")?;
            self.model.pack_list_handoff(&self.sc, scp, at, &d)?;
            at += d.drafts.len() + 1;
            drafts.push(d);
        }
        // S-B9-CF-PACK5 (TP): the pack's ONE pre-verify lockstep point — after BOTH lanes drafted,
        // before the packed verify launches (the verify's R-keyed barrier sequence needs every rank
        // to have built the same pack). The pack was decided by `pack_plan` from replicated lane
        // state, so a divergence here is a bug (or a fault drill), never a legitimate difference.
        // `--exl3-tp-preverify=0` (diagnostic) skips it, as it skips the lone round's.
        let tpre0 = std::time::Instant::now();
        if self.tp.is_some() && tp_preverify_on() {
            self.model.pack_drain_wasted(&drafts)?;
            let segs = [PreSeg::of(drafts[0].slot, drafts[0].p, drafts[0].b, &drafts[0].drafts, &drafts[0].samp, drafts[0].launched, drafts[0].wasted),
                        PreSeg::of(drafts[1].slot, drafts[1].p, drafts[1].b, &drafts[1].drafts, &drafts[1].samp, drafts[1].launched, drafts[1].wasted)];
            let words = crate::lane_pack::pre_words(&segs);
            let r = crate::lane_pack::pack_rows_total(&segs);
            let launched = drafts[0].launched + drafts[1].launched;
            let t = self.tp.as_mut().context("lane-pack: TP state missing")?;
            t.pre_verify_pack(r, segs[0].drafts.len() + 1, &words, launched)?;
            self.pack_stats.ms_pre += tpre0.elapsed().as_secs_f64() * 1e3;
        }
        // S-B9-CF-PACK5: the decision trace of a packed round, on EVERY rank (head and node print the
        // same line: slots, rows, the combined pack hash, head passes, wasted passes), so a divergence
        // is a `diff` of two logs. The first 16 rounds, then every 4096th.
        {
            let n = self.pack_stats.rounds + 1;
            if n <= 16 || n % 4096 == 0 {
                let segs = [PreSeg::of(drafts[0].slot, drafts[0].p, drafts[0].b, &drafts[0].drafts, &drafts[0].samp, drafts[0].launched, drafts[0].wasted),
                            PreSeg::of(drafts[1].slot, drafts[1].p, drafts[1].b, &drafts[1].drafts, &drafts[1].samp, drafts[1].launched, drafts[1].wasted)];
                let h = fnv32(0x5EED, crate::lane_pack::pre_words(&segs).iter().map(|&d| d as u32));
                crate::drprintln!("[exl3-serve] lane-pack: round #{n}{}: slots {}+{} rows {}+{}={} pack hash {h:08x} head passes {}+{} wasted {}+{}",
                                 self.tp.as_ref().map_or(String::new(), |t| format!(" (TP rank {})", t.rank)),
                                 drafts[0].slot, drafts[1].slot, drafts[0].drafts.len() + 1, drafts[1].drafts.len() + 1,
                                 crate::lane_pack::pack_rows_total(&segs), drafts[0].launched, drafts[1].launched,
                                 drafts[0].wasted as u8, drafts[1].wasted as u8);
            }
        }
        let t1 = std::time::Instant::now();
        // 2. ONE packed verify: stage, replay, taps into the served windows, one readback
        let scp = self.scp.as_mut().context("lane-pack: the packed scratch is missing")?;
        let (outs, pvm) = self.model.pack_verify(&mut self.sc, scp, &drafts[0], &drafts[1])?;
        let t2 = std::time::Instant::now();
        // 3. per request: the head re-prime from the taps the verify left in its window
        for j in 0..2 {
            self.model.pack_finish(&mut self.sc, head, &drafts[j], &outs[j])?;
        }
        let t3 = std::time::Instant::now();
        let captured_local = self.model.graph_count() != g0;
        let ms_of = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
        // (--wp24-xcheck's host re-derivation is not round cost: the lone round's `diag_ms` rule)
        let ms_local = (ms_of(t0, t3) - pvm.diag).max(0.0);
        // S-B9-CF-PACK5 (TP): the packed round's POST-verify lockstep point, `tp_round` as the lone
        // round runs it: agree the pair's outcome (per lane the accept count, rows, emitted ids,
        // drafts and draft confidences; the total row count R as the width) and TAKE THE HEAD's
        // round ms and capture flag. `step_pair` fed its LOCAL clock to the cost model before; under
        // TP that would let the EMAs, the DDS fit and the section 6 control diverge between ranks.
        let tpost0 = std::time::Instant::now();
        let (ms_round, _plain_ms, captured) = if self.tp.is_some() {
            let lanes: Vec<PostLane> = (0..2).map(|j| PostLane {
                accepted: outs[j].a, rows: drafts[j].drafts.len() + 1,
                emitted: &outs[j].emitted, drafts: &outs[j].drafts, confs: &drafts[j].confs }).collect();
            let w = crate::lane_pack::post_words(&lanes);
            let r = lanes.iter().map(|l| l.rows).sum::<usize>();
            self.tp_round("pack-round", outs[0].a + outs[1].a, r, &w, ms_local, 0.0, captured_local, true)?
        } else {
            (ms_local, 0.0, captured_local)
        };
        if self.tp.is_some() { self.pack_stats.ms_post += tpost0.elapsed().as_secs_f64() * 1e3; }
        let ms_lane = lane_share_ms(ms_round, 2);
        crate::metrics::lane_pack_round(0); // arm = packed
        {
            let st = &mut self.pack_stats;
            st.rq_rounds += drafts.iter().any(|d| d.samp.ratio) as u64;
            for d in &drafts {
                let kind = if d.samp.samp.is_none() { 2 } else if d.samp.ratio { 0 } else { 1 };
                crate::metrics::lane_pack_lane_round(kind);
                match kind { 0 => st.lanes_rq += 1, 1 => st.lanes_match += 1, _ => st.lanes_greedy += 1 }
            }
            st.rounds += 1;
            st.rows += (drafts[0].drafts.len() + drafts[1].drafts.len() + 2) as u64;
            st.toks += (outs[0].a + outs[1].a + 2) as u64;
            st.ms_draft += ms_of(t0, t1);
            st.ms_verify += ms_of(t1, t2);
            st.ms_finish += ms_of(t2, t3);
            st.ms_stage += pvm.stage;
            st.ms_graph += pvm.graph;
            st.ms_read += pvm.read;
            let r = (drafts[0].drafts.len() + drafts[1].drafts.len() + 2).min(16);
            st.by_r[r].0 += 1;
            st.by_r[r].1 += pvm.graph;
            st.by_r[r].2 += (outs[0].a + outs[1].a + 2) as u64;
        }
        // every verified position labels the calibrator (the lone round's rule, per request;
        // a fixed-depth chain has no confidences and labels nothing, as in the lone mtp_round)
        for j in 0..2 {
            for (i, &c) in drafts[j].confs.iter().enumerate() {
                if i < outs[j].a { self.cal.add_label(i, c, true); } else { self.cal.add_label(i, c, false); break; }
            }
            if dds { self.cal.decay_step(); }
        }
        // 4. per request: the serial arm's bookkeeping and emit loop. A lane that finishes here
        // (stop / length / loop / a gone client) frees its slot; its partner's round is complete
        // on the device and in `outs`, so it is accounted and emitted regardless.
        let row0 = [0usize, drafts[0].drafts.len() + 1];
        for j in 0..2 {
            let s = pre[j].0;
            // the lane's real-q flag: ctr advances by k + 1 and (real-q) dctr by k, as in its lone round
            self.round_account(s, &outs[j], ms_lane, 0.0, captured, false, false, drafts[j].samp.ratio, xcheck, true, false)?;
            self.round_emit(s, &outs[j], Some(row0[j]))?;
        }
        self.pack_stats.in_round = false;
        Ok(())
    }

    /// S-B9-CF-PACK4: the idle-point line of the packed arm (None = nothing new since the last).
    pub(super) fn pack_idle_line(&mut self) -> Option<String> {
        let st = self.pack_stats;
        if st.rounds == st.printed { return None; }
        self.pack_stats.printed = st.rounds;
        let curve: Vec<String> = st.by_r.iter().enumerate().filter(|(_, c)| c.0 > 0)
            .map(|(r, c)| format!("R{r}:{}x {:.1}ms {:.2}tok", c.0, c.1 / c.0 as f64, c.2 as f64 / c.0 as f64)).collect();
        crate::drprintln!("[exl3-serve] lane-pack cost by width (rounds, graph-and-sync ms, tokens per round): {}", curve.join(" | "));
        let n = st.rounds.max(1) as f64;
        let skips: Vec<String> = crate::lane_pack::Skip::ALL.iter()
            .map(|w| (w.label(), crate::metrics::lane_pack_skips(w.index())))
            .filter(|(_, c)| *c > 0).map(|(l, c)| format!("{l} {c}")).collect();
        Some(format!("[exl3-serve] lane-pack idle{}: {} packed rounds since boot ({} on the real-q family; lane rounds in packs: {} real-q sampled, {} match-rule sampled, {} greedy; \
{:.2} rows and {:.2} tokens per packed round; \
host ms per round: drafts {:.2} + verify {:.2} (staging {:.2} + graph and sync {:.2} + readback {:.2}) + re-prime enqueue {:.2}{}); rounds by arm: packed {} serial {} shared_plain {}; \
lane rounds outside a pack: {}", self.tp.as_ref().map_or(String::new(), |t| format!(" (TP rank {})", t.rank)),
                     st.rounds, st.rq_rounds, st.lanes_rq, st.lanes_match, st.lanes_greedy, st.rows as f64 / n, st.toks as f64 / n,
                     st.ms_draft / n, st.ms_verify / n, st.ms_stage / n, st.ms_graph / n, st.ms_read / n, st.ms_finish / n,
                     if self.tp.is_some() { format!("; TP lockstep: pre-verify {:.2} ms (inside the drafts figure) + post-verify tp_round {:.2} ms (after the re-prime enqueue: NOT in the figures above) per round",
                                                    st.ms_pre / n, st.ms_post / n) } else { String::new() },
                     crate::metrics::lane_pack_rounds(0), crate::metrics::lane_pack_rounds(1), crate::metrics::lane_pack_rounds(2),
                     if skips.is_empty() { "none".to_string() } else { skips.join(", ") }))
    }

    /// S-C4XOVER: fold one round's (tokens, ms) into its arm's measured-rate window; every
    /// SPEC_ARM_FOLD_EVERY rounds the window rate folds into the arm's EWMA, tagged with the
    /// busiest lane count it saw. Rank-identical by construction: fed from the same agreed
    /// round-ms values the per-lane EMAs already use, on both TP ranks.
    pub(super) fn arm_acc(&mut self, arm: usize, tok: f64, ms: f64) {
        if ms <= 0.0 || tok <= 0.0 { return; }
        let n = self.lanes.iter().filter(|l| l.is_some()).count() as u8;
        if spec_meter_v2() {
            // POSTURE1 v2: the arms are metered per step (step_rounds_metered), never per round
            return;
        }
        crate::metrics::spec_arm_charge(arm, ms, tok);
        self.spec_win_tok[arm] += tok;
        self.spec_win_ms[arm] += ms;
        self.spec_win_n[arm] = self.spec_win_n[arm].max(n);
        self.spec_win_acc[arm] += 1;
        if self.spec_win_acc[arm] >= SPEC_ARM_FOLD_EVERY {
            let rate = self.spec_win_tok[arm] / self.spec_win_ms[arm];
            self.spec_arm_rate[arm] = if self.spec_arm_rate[arm] > 0.0 {
                0.5 * self.spec_arm_rate[arm] + 0.5 * rate
            } else { rate };
            self.spec_arm_n[arm] = self.spec_win_n[arm];
            self.spec_arm_fold[arm] = self.steps as u64;
            self.spec_win_tok[arm] = 0.0;
            self.spec_win_ms[arm] = 0.0;
            self.spec_win_n[arm] = 0;
            self.spec_win_acc[arm] = 0;
            crate::metrics::spec_lanes_measured(arm, self.spec_arm_rate[arm] * 1e3);
        }
    }

    /// S-C4XOVER: is this arm's measured rate fresh at this busy-lane count?
    pub(super) fn arm_fresh(&self, arm: usize, n: usize) -> bool {
        if spec_meter_v2() {
            let e = self.spec_rate_n[n.min(SPEC_RATE_NMAX)][arm];
            return e.0 > 0.0 && e.1.saturating_add(SPEC_ARM_FRESH_STEPS) > self.steps as u64;
        }
        self.spec_arm_rate[arm] > 0.0
            && self.spec_arm_fold[arm].saturating_add(SPEC_ARM_FRESH_STEPS) > self.steps as u64
            && (self.spec_arm_n[arm] as i64 - n as i64).abs() <= SPEC_ARM_N_TOL
    }

    /// CF-P1e: do the busy lanes share one plain batched step this round (true) or run serial speculative rounds?
    /// `Auto` compares the estimated aggregate tokens/ms of the two arms (4% hysteresis): serial = the busy
    /// MTP-capable lanes' measured round speed (sum tokens / sum ms) plus one batched step for any MTP-off lanes;
    /// shared = n tokens per modelled shared-step time. A lane without an estimate yet falls back to a lane count of 4.
    pub(super) fn use_shared_plain(&mut self, slots: &[usize]) -> bool {
        let n = slots.len();
        if n < 2 { return false; }
        match self.spec_policy {
            SpecLanes::Never => false,
            SpecLanes::Over(k) => n > k,
            SpecLanes::Auto => {
                // S-C4XOVER: measured delivered rates outrank the modelled estimates (both
                // were biased at four busy lanes: the serial cycle estimate ~+30% optimistic
                // vs delivery, the shared prior ~-18% pessimistic; together they pinned the
                // pick to the slower arm, ~13% at c4). When both arms have a fresh measured
                // rate at a compatible busy count, compare THOSE; the estimate path below is
                // the fallback for an arm that has not run recently.
                if self.arm_fresh(ARM_SERIAL, n) && self.arm_fresh(ARM_SHARED, n) {
                    let (s_rate, h_rate) = if spec_meter_v2() {
                        let e = self.spec_rate_n[n.min(SPEC_RATE_NMAX)];
                        (e[ARM_SERIAL].0, e[ARM_SHARED].0)
                    } else { (self.spec_arm_rate[ARM_SERIAL], self.spec_arm_rate[ARM_SHARED]) };
                    let pick = if h_rate > s_rate * 1.04 { true } else if s_rate > h_rate * 1.04 { false } else { self.shared_last };
                    if pick != self.shared_last && self.policy_logs < 64 {
                        self.policy_logs += 1;
                        crate::dreprintln!("[exl3-serve] spec-lanes auto (measured): {} with {n} busy lanes \
                           (serial speculation {:.0} tok/s delivered, one shared step {:.0} tok/s delivered)",
                            if pick { "SHARED plain step" } else { "serial speculation" }, s_rate * 1e3, h_rate * 1e3);
                    }
                    self.spec_margin = if s_rate.min(h_rate) > 0.0 { s_rate.max(h_rate) / s_rate.min(h_rate) - 1.0 } else { 0.0 };
                    self.spec_margin_n = n as u8;
                    crate::metrics::spec_lanes_decision(s_rate * 1e3, h_rate * 1e3, self.shared_scale, n);
                    if pick != self.shared_last {
                        crate::metrics::spec_lanes_switch(pick as usize); // 0 = shared, 1 = serial
                    }
                    self.shared_last = pick;
                    return pick;
                }
                let (mut tok, mut ms, mut capable, mut known) = (0.0f64, 0.0f64, 0usize, 0usize);
                for &s in slots {
                    let l = self.lanes[s].as_ref().unwrap();
                    if l.mtp.is_some() {
                        capable += 1;
                        if l.ema_tok > 0.0 && l.ema_ms > 0.0 { tok += l.ema_tok;
                            // P18 SPEC8K: price the serial arm by the measured CYCLE of the lane
                            // (serial_cycle_ms) instead of the ms of its own round, which for a packed
                            // lane is one SHARE of the verify. Both arms are then read, below, as the
                            // aggregate tokens per second that the n clients see.
                            ms += serial_cycle_ms(l.ema_ms, l.ema_wall_ms);
                            known += 1;
                        }
                    }
                }
                if capable == 0 { return false; }          // nothing speculates: the plain batch already runs
                if known < capable { return n > 4; }       // an estimate is missing: the fixed threshold
                let plain_only = n - capable;
                let shared_ms = self.shared_scale * shared_step_prior_ms(n);
                if plain_only > 0 {
                    tok += plain_only as f64;
                    ms += self.shared_scale * shared_step_prior_ms(plain_only);
                }
                // P18 SPEC8K: both sides are the AGGREGATE tokens per second that the n clients
                // see (n times a per-lane rate), which is the quantity the arm choice is about.
                let (serial_rate, shared_rate) = (n as f64 * tok / ms, n as f64 / shared_ms);
                let pick = if shared_rate > serial_rate * 1.04 { true } else if serial_rate > shared_rate * 1.04 { false } else { self.shared_last };
                if pick != self.shared_last && self.policy_logs < 64 {
                    self.policy_logs += 1;
                    crate::dreprintln!("[exl3-serve] spec-lanes auto: {} with {n} busy lanes (serial speculation {:.0} tok/s est, \
                               one shared step {:.0} tok/s est, scale {:.2}, serial cycle {:.1} ms/lane)",
                              if pick { "SHARED plain step" } else { "serial speculation" },
                              serial_rate * 1e3, shared_rate * 1e3, self.shared_scale, ms / n as f64);
                }
                // S-C4XOVER: the decision on /metrics (est gauges + flip counters; no behavior change).
                self.spec_est_pick = Some(pick);
                crate::metrics::spec_lanes_decision(serial_rate * 1e3, shared_rate * 1e3, self.shared_scale, n);
                if pick != self.shared_last {
                    crate::metrics::spec_lanes_switch(pick as usize); // 0 = shared, 1 = serial
                }
                self.shared_last = pick;
                pick
            }
        }
    }
}
