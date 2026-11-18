//! One scheduler step: `step` (the wrapper that publishes after any arm), `step_rounds` / `step_rounds_metered` / `step_rounds_rest` (arm choice),
//! `tp_round` / `tp_round_with` (the per-round lockstep digest), trace and gauge helpers. Another `impl Exl3Scheduler` block.
//! Binding invariants: the round counter moves once per round and only `step` touches the age gauge (pinned by source-text tests).

use super::*;

/// The phase after `toks` given the phase before: the last reasoning marker in them decides.
fn trace_phase(open: Option<u32>, close: Option<u32>, toks: &[u32], before: bool) -> bool {
    for &t in toks.iter().rev() {
        if Some(t) == close { return true; }
        if Some(t) == open { return false; }
    }
    before
}

/// The step counters `--sched-trace` reads before and after a step: packed, serial, shared, wide rounds.
fn trace_ctrs() -> [u64; 5] {
    [crate::metrics::lane_pack_rounds(0), crate::metrics::lane_pack_rounds(1), crate::metrics::lane_pack_rounds(2),
     crate::lane_wide::rounds(), crate::copy_lanes::rounds() + crate::copy_lanes::stretches()]
}

struct TraceSnap { t0: std::time::Instant, ctr: [u64; 5], gen: Vec<Option<(u64, usize)>> }

impl Exl3Scheduler {
    /// One decode round over all active lanes (batched; lanes are independent slots).
    /// S-A3-f-d: lanes with a loaded M1 head run chain-verify rounds (speculation ON by
    /// default per AGENTS 1b — --exl3-no-mtp is the owner's diagnostics hatch, never a
    /// default). API-parity G2: sampled lanes speculate too (device-sampled verify rows).
    /// Lanes whose request turned MTP off (G1) batch through the plain decode path.
    pub(super) fn step(&mut self) -> Result<()> {
        let before = self.steps;
        // Logprobs at TP (owner 2026-11-02): while a lane reports log probabilities, every step's LM-head tail leaves
        // FULL logits rows (Scratch::lp_live; the row gather is a collective: every rank sets it from the same
        // mirrored lanes). TP=1 has no vocabulary-parallel tail: nothing changes there.
        self.sc.lp_live = self.tp.is_some() && self.lanes.iter().flatten().any(|l| l.logprobs.is_some());
        let snap = if sched_trace_on() {
            Some(TraceSnap { t0: std::time::Instant::now(), ctr: trace_ctrs(),
                             gen: self.lanes.iter().map(|l| l.as_ref().map(|l| (l.admit_seq, l.generated))).collect() })
        } else { None };
        let r = self.step_rounds();
        if let Some(s0) = snap { self.trace_step(s0); }
        // H3: round COMPLETE — the age gauge measures from here. Every arm of `step_rounds` ends
        // here (it used to be written on the rr arm only: a shared plain step and an fcfs step
        // left velogb10_scheduler_last_step_age_seconds growing while busy read 1).
        if r.is_ok() && self.steps != before { crate::metrics::sched_touch(); }
        self.publish_kv(); // issue #24: lane positions after this step (finished lanes read 0)
        r
    }

    /// `--sched-trace`: the step that just ran — wall ms, the arm counters it moved (P packed, S serial, H shared
    /// plain, W wide), and per busy slot `slot:tokens:pos:phase` (T reasoning, A answer; `fin` = ended in the step).
    fn trace_step(&mut self, s0: TraceSnap) {
        let ms = s0.t0.elapsed().as_secs_f64() * 1e3;
        let c1 = trace_ctrs();
        if self.tr.marks.is_none() {
            let id = |s: &str| self.tok.encode(s, false).ok().filter(|v| v.len() == 1).map(|v| v[0]);
            self.tr.marks = Some((id("<think>"), id("</think>")));
        }
        let (open, close) = self.tr.marks.unwrap();
        if self.tr.lanes.len() < self.lanes.len() { self.tr.lanes.resize(self.lanes.len(), (u64::MAX, 0, false)); }
        let mut parts: Vec<String> = Vec::new();
        for s in 0..self.lanes.len() {
            let before = s0.gen.get(s).copied().flatten();
            match (self.lanes[s].as_ref(), before) {
                (Some(l), b) => {
                    let st = &mut self.tr.lanes[s];
                    if st.0 != l.admit_seq {
                        // a new request: its phase is the prompt's (the template opens or closes the reasoning)
                        let plen = l.history.len().saturating_sub(l.generated);
                        let tail = &l.history[plen.saturating_sub(64)..plen];
                        *st = (l.admit_seq, plen, trace_phase(open, close, tail, true));
                    }
                    let from = st.1.min(l.history.len());
                    st.2 = trace_phase(open, close, &l.history[from..], st.2);
                    st.1 = l.history.len();
                    let d = match b { Some((seq, g)) if seq == l.admit_seq => l.generated.saturating_sub(g), _ => 0 };
                    parts.push(format!("{s}:{d}:{}:{}", l.pos, if st.2 { 'A' } else { 'T' }));
                }
                (None, Some(_)) => parts.push(format!("{s}:fin")),
                (None, None) => {}
            }
        }
        // COPYLANES: ` cl=N` (copy rounds + stretches served in a busy step) is appended only with --copy-span-lanes on
        let cl = if self.cl_on { format!(" cl={}", c1[4] - s0.ctr[4]) } else { String::new() };
        crate::rprintln!("[sched-trace] t={:.3} step={} ms={ms:.2} arm=P{}S{}H{}W{} pf={} lanes={}{cl}", wall_now(), self.steps,
                         c1[0] - s0.ctr[0], c1[1] - s0.ctr[1], c1[2] - s0.ctr[2], c1[3] - s0.ctr[3],
                         u8::from(self.in_prefill), parts.join(","));
    }

    /// Logprobs: this rank reads logits rows back for the requests that asked (TP=1, or the TP head: the nodes
    /// run the same full-rows tail but serve no client).
    pub(super) fn lp_reader(&self) -> bool { self.tp.as_ref().map_or(true, |t| t.rank == 0) }

    /// Public issue #24: publish every lane's KV occupancy to /metrics — the positions a running
    /// request holds (`Lane.pos`: prompt + generated; the prefill in flight: the rows written so
    /// far) and, for a free lane, the prompt positions its prefix-cache snapshot keeps. Host-only:
    /// a few relaxed loads/stores per lane, once per scheduler step / prefill chunk / admit.
    pub(super) fn publish_kv(&self) {
        let pf = self.kv_pf;
        crate::metrics::kv_publish(self.lanes.iter().zip(self.cache.iter()).enumerate()
            .map(|(s, (lane, cache))| kv_lane_occupancy(lane.as_ref().map(|l| l.pos), pf.filter(|p| p.0 == s).map(|p| p.1),
                                                        cache.as_ref().map(|c| c.len()))));
    }

    /// The rounds of one `step` (every arm may return early; `step` does the per-step publication).
    fn step_rounds(&mut self) -> Result<()> {
        crate::metrics::sched_step(); // v0.7.3 gauge: rounds + busy
        crate::thermal::pace(); // NIGHT1: the thermal governor (every rank, once per scheduler step)
        // WP02 cancel sweep (the NVFP4 decode_step sweep, batch.rs): a lane whose client is gone
        // (disconnect, or the HTTP side dropped the receiver on a stop-string hit) frees its slot
        // NOW instead of decoding to EOS/max_new while every later request waits behind it.
        // TP-D: under TP the HEAD sweeps (run_tp_head) and ships each cancel as a Step event — a
        // rank-local sweep would free a lane on one rank only.
        for s in 0..self.lanes.len() {
            if self.tp.is_none() {
                if let Some(l) = self.lanes[s].as_ref() {
                    let closed = l.tx.is_closed();
                    let over = !closed && crate::server::stream_backlog_exceeded(l.tx.len()); // LR-5
                    if closed || over {
                        let lane = self.take_lane(s, false).unwrap();
                        if over {
                            crate::metrics::stream_backlog_cancel();
                            crate::dreprintln!("[exl3-serve] slot {s}: stream cancelled — unconsumed event backlog at the --stream-backlog-events limit (reader stalled; LR-5)");
                        }
                        lane.log_cancel(s);
                    }
                }
            }
        }
        let slots: Vec<usize> = self.lanes.iter().enumerate()
            .filter(|(_, l)| l.is_some())
            .map(|(i, _)| i)
            .collect();
        if slots.is_empty() {
            return Ok(());
        }
        self.steps += 1;
        if self.inject_panic == Some(self.steps) {
            panic!("--inject-panic-steps: injected scheduler panic at decode step {} (liveness gate)", self.steps);
        }
        crate::tel::note_step();
        let mtp_on = self.model.mtp.is_some()
            && !crate::exl3_forward::mtp_disabled_by_opt();
        // S-B9-REL-MTPREPROBE: arm due re-probes BEFORE dispatch (pure lane state —
        // plain-step counts and agreed HEAD ms inputs — so every TP rank arms the same
        // lanes; lockstep holds through the transition). The seam is exactly a control
        // round's post-step state: the lane's plain steps kept its head row primed and
        // its tap snapshotted (step_plain), so Some(0) arms it cleanly.
        if mtp_on && self.reprobe_on {
            let mut armed: Vec<(usize, usize, usize)> = Vec::new(); // (slot, plain steps waited, off event)
            for &s in &slots {
                let due = self.lanes[s].as_ref()
                    .map_or(false, |l| l.mtp.is_none() && l.reprobe.probe_due(&self.reprobe_cfg));
                if due {
                    let (steps, ev) = {
                        let l = self.lanes[s].as_ref().unwrap();
                        (l.reprobe.plain_steps, l.reprobe.off_events)
                    };
                    let l = self.lanes[s].as_mut().unwrap();
                    l.mtp = Some(0);
                    l.ema_ms = 0.0;
                    l.ema_tok = 0.0;
                    l.ema_n = 0;
                    l.reprobe.armed_for_probe();
                    armed.push((s, steps, ev));
                }
            }
            for &(s, steps, ev) in &armed {
                crate::drprintln!("[exl3-serve] MTP re-probe (slot {s}): back ON after {steps} plain steps \
(off event {ev}; probe window {} rounds, a {:.0}% sustained loss turns it off again)",
                           self.reprobe_cfg.probe_window, 100.0 * self.reprobe_cfg.margin);
            }
        }
        // COPYLANES (--copy-span-lanes on; `cl_on` false = nothing is called): with two or more busy lanes, each lane whose
        // copy work pays for the batch runs it now (the lone path's copy round or stretch); the others run this step's arms
        // among themselves, unmetered (a step with copy work in it is not a sample of either arm's rate).
        if self.cl_on && mtp_on && slots.len() >= 2 {
            let took = self.cl_step(&slots)?;
            if !took.is_empty() {
                let rest: Vec<usize> = slots.iter().copied()
                    .filter(|s| !took.contains(s) && self.lanes[*s].is_some()).collect();
                return match rest.len() {
                    0 => Ok(()),
                    1 if self.lanes[rest[0]].as_ref().map_or(false, |l| l.mtp.is_some()) => self.step_mtp(rest[0], false),
                    1 => self.step_plain(&rest, false),
                    _ => self.cl_rest(rest),
                };
            }
        }
        // LANEWIDE (--lane-wide-verify): the wide arm as a third choice, by measured aggregate tok/s at this number of
        // eligible lanes (`lane_wide::Policy`); `always` takes it whenever two or more lanes are eligible. The other
        // choice runs today's arms (their own serial / packed / shared choice below), measured the same way.
        if mtp_on && self.wide.live() && slots.len() >= 2 {
            let (elig, rest) = self.wide_lanes(&slots);
            // the wide arm needs at least WIDE_MIN_LANES eligible lanes (at 2 lanes it lost to today's packed pairs in every
            // measurement; the policy's windows there only cost); `always` keeps 2 for diagnostics
            let min_n = if self.wide.st == crate::lane_wide::State::Always { 2 } else { crate::lane_wide::WIDE_MIN_LANES };
            if elig.len() >= min_n {
                let n = elig.len();
                let pick = if self.wide.st == crate::lane_wide::State::Always { crate::lane_wide::Pick::WIDE }
                           else { self.wide.pol.pick(n, self.steps as u64) };
                crate::lane_wide::count_pick(pick);
                let (t0, k0) = (std::time::Instant::now(), crate::lane_wide::tokens_emitted());
                match pick.arm {
                    crate::lane_wide::ARM_WIDE => self.step_wide(&elig, &rest)?,
                    crate::lane_wide::ARM_SHARED => {
                        crate::metrics::lane_pack_round(2); // arm = shared_plain
                        self.step_plain(&slots, true)?;
                    }
                    _ => self.step_rounds_metered(slots, mtp_on)?,
                }
                let (mut ms, tok) = (t0.elapsed().as_secs_f64() * 1e3, (crate::lane_wide::tokens_emitted() - k0) as f64);
                // LANEWIDE-TP: the policy's measured rates must be the same on every rank (its next pick is a function of
                // them): one more lockstep point agrees the arm and the step's token count and takes the HEAD's step ms
                if self.tp.is_some() { self.model.tp_drain_before_exchange()?; }
                if let Some(t) = self.tp.as_mut() {
                    let w = [pick.arm as u32, tok as u32, n as u32];
                    ms = Self::tp_round_with(t, &self.model, None, "wide-policy", 0, 0, &w, ms, 0.0, false)?.0;
                }
                self.wide.pol.record(n, pick.arm, tok, ms, self.steps as u64);
                return Ok(());
            } else if !elig.is_empty() {
                crate::lane_wide::count_skip(crate::lane_wide::Skip::TooFew);
            }
        }
        self.step_rounds_metered(slots, mtp_on)
    }

    /// Today's arms (`step_rounds_rest`) for one step, metered per step under `--spec-arm-meter v2`.
    fn step_rounds_metered(&mut self, slots: Vec<usize>, mtp_on: bool) -> Result<()> {
        // POSTURE1 --spec-arm-meter v2: the arms are metered per STEP (every token the step emitted over its whole wall
        // time, host work included), at the exact busy count, outside prefills; the rounds' own calls are not used.
        // ULTRA: a busy-count change or a prefill step restarts the settle countdown (lockstep state on every rank)
        if spec_meter_v2() && (self.in_prefill || slots.len() != self.meter_last_n) {
            self.meter_settle = SPEC_V2_SETTLE;
            self.meter_last_n = slots.len();
        }
        let settling = spec_meter_v2() && self.meter_settle > 0;
        let meter = spec_meter_v2() && mtp_on && !self.in_prefill && slots.len() >= 2 && !settling;
        let (t0, k0, h0) = (std::time::Instant::now(), crate::lane_wide::tokens_emitted(), crate::metrics::lane_pack_rounds(2));
        let n = slots.len();
        let r = self.step_rounds_rest(slots, mtp_on);
        if settling && !self.in_prefill { self.meter_settle -= 1; } // (after the step: its probe check saw the settle)
        if meter && r.is_ok() && self.lanes.iter().filter(|l| l.is_some()).count() == n {
            // (a step in which a request finished or was cancelled is not metered: its count changed mid-step)
            let arm = if crate::metrics::lane_pack_rounds(2) != h0 { ARM_SHARED } else { ARM_SERIAL };
            let tok = (crate::lane_wide::tokens_emitted() - k0) as f64;
            let mut ms = t0.elapsed().as_secs_f64() * 1e3;
            // ULTRA (TP): the arm pick is a function of these rates, so every rank must fold the SAME step time: one
            // lockstep point per metered step agrees the arm, the token count and the busy count, and takes the HEAD's
            // step ms (the local wall clock differs per rank; folding it would let the ranks pick different arms).
            // Rank-identical gate: meter / n / the lane census / the arm are lockstep state. world 2: drain the step's
            // device collectives first (the ring-aliasing rule of the wide-policy point).
            if self.tp.is_some() { self.model.tp_drain_before_exchange()?; }
            if let Some(t) = self.tp.as_mut() {
                let w = [arm as u32, tok as u32, n as u32];
                ms = Self::tp_round_with(t, &self.model, None, "arm-meter", 0, 0, &w, ms, 0.0, false)?.0;
                crate::metrics::spec_meter_tp_sync();
            }
            self.arm_meter_v2(arm, n, tok, ms);
        }
        r
    }

    /// POSTURE1 v2: fold one metered step into the arm's rate at exactly `n` busy lanes.
    fn arm_meter_v2(&mut self, arm: usize, n: usize, tok: f64, ms: f64) {
        if ms <= 0.0 || tok <= 0.0 { return; }
        crate::metrics::spec_arm_charge(arm, ms, tok);
        if self.spec_win_acc[arm] > 0 && self.spec_win_n[arm] as usize != n {
            self.spec_win_tok[arm] = 0.0;
            self.spec_win_ms[arm] = 0.0;
            self.spec_win_acc[arm] = 0;
        }
        self.spec_win_tok[arm] += tok;
        self.spec_win_ms[arm] += ms;
        self.spec_win_n[arm] = n as u8;
        self.spec_win_acc[arm] += 1;
        if self.spec_win_acc[arm] >= spec_v2_fold(arm) {
            let rate = self.spec_win_tok[arm] / self.spec_win_ms[arm];
            let steps = self.steps as u64;
            let probe = std::mem::replace(&mut self.probe_fold[arm], false);
            self.spec_rate_folds[n.min(SPEC_RATE_NMAX)][arm] = self.spec_rate_folds[n.min(SPEC_RATE_NMAX)][arm].saturating_add(1);
            let e = &mut self.spec_rate_n[n.min(SPEC_RATE_NMAX)][arm];
            let fresh = e.0 > 0.0 && e.1.saturating_add(SPEC_ARM_FRESH_STEPS) > steps;
            // ULTRA: a probe that contradicts the stored rate by a wide margin replaces it
            let contra = fresh && probe && (rate / e.0 - 1.0).abs() > SPEC_V2_REPLACE;
            if contra { crate::metrics::spec_meter_replace(); }
            e.0 = if fresh && !contra { 0.5 * e.0 + 0.5 * rate } else { rate };
            e.1 = steps;
            // the v1 fields mirror the latest fold (the probe's never-measured test and /metrics read them)
            self.spec_arm_rate[arm] = e.0;
            self.spec_arm_n[arm] = n as u8;
            self.spec_arm_fold[arm] = steps;
            self.spec_win_tok[arm] = 0.0;
            self.spec_win_ms[arm] = 0.0;
            self.spec_win_acc[arm] = 0;
            crate::metrics::spec_lanes_measured(arm, e.0 * 1e3);
        }
    }

    /// The rounds of one step on today's arms (serial / packed speculation, the shared plain step, fcfs).
    pub(super) fn step_rounds_rest(&mut self, slots: Vec<usize>, mtp_on: bool) -> Result<()> {
        // CF-P1e load-adaptive batching: past `--spec-lanes-max` busy lanes, serial speculative rounds stop
        // scaling (one lane per round), so every busy lane shares one batched plain step instead; the
        // lanes' MTP heads stay in sync (see step_plain's `sync_head`), so speculation resumes when load drops.
        // S-C4XOVER arm-rate probe (generalizes the F4 shared-cost probe): while one arm
        // rides, the idle arm's MEASURED rate goes stale; at most every 256 riding rounds (16
        // while it has never been measured) a 4-step window of the idle arm runs, refreshing
        // its measured rate (and, for shared, the learned scale) so the pick below stays
        // grounded in DELIVERY, not in the priors. Gated like F4 (lane packing on, Auto
        // policy) but at >=2 busy lanes with at least one MTP-capable; pure scheduler-state
        // decision (counters, rates, policy are rank-identical or adopted), so both TP ranks
        // take it in the same step, like the re-probe arming above.
        self.spec_probe_ctr = self.spec_probe_ctr.saturating_add(1);
        let mut probe_force: Option<bool> = None;
        if mtp_on && self.lane_pack.on() && self.spec_policy == SpecLanes::Auto && slots.len() >= 2
            && slots.iter().any(|&s| self.lanes[s].as_ref().map_or(false, |l| l.mtp.is_some())) {
            if let Some((arm, rem)) = self.spec_probe {
                self.spec_probe = if rem <= 1 { None } else { Some((arm, rem - 1)) };
                probe_force = Some(arm as usize == ARM_SHARED);
                self.shared_probe_ctr = 0;
            } else {
                let riding = if self.shared_last { ARM_SHARED } else { ARM_SERIAL };
                let idle = 1 - riding;
                let first = self.spec_arm_rate[idle] <= 0.0;
                // POSTURE1 v2: the idle arm has no fresh rate at THIS busy count -> probe it soon (as at boot); its rate
                // here is older than SPEC_V2_REFRESH steps -> it is due like a stale one (the content may have changed)
                // ULTRA: re-probe sooner (SPEC_V2_REFRESH_FAST) while the measured rates are close or the idle one is young
                let v2_every = if spec_meter_v2() {
                    let nn = slots.len().min(SPEC_RATE_NMAX);
                    let (a, b) = (self.spec_rate_n[nn][ARM_SHARED].0, self.spec_rate_n[nn][ARM_SERIAL].0);
                    let close = a > 0.0 && b > 0.0 && a.max(b) / a.min(b) - 1.0 < SPEC_V2_CLOSE;
                    if close || self.spec_rate_folds[nn][idle] < 2 { SPEC_V2_REFRESH_FAST } else { SPEC_V2_REFRESH }
                } else { SPEC_V2_REFRESH };
                let v2_idle = if spec_meter_v2() {
                    let e = self.spec_rate_n[slots.len().min(SPEC_RATE_NMAX)][idle];
                    Some((!self.arm_fresh(idle, slots.len()), (self.steps as u64).saturating_sub(e.1) > v2_every))
                } else { None };
                let first = first || v2_idle.map_or(false, |v| v.0);
                let agrees = self.spec_est_pick == Some(riding == ARM_SHARED)
                    && self.spec_margin >= 0.12 && (self.spec_margin_n as i64 - slots.len() as i64).abs() <= 1;
                let backoff = if agrees { 8.0 }
                              else if self.spec_margin >= 0.12 && (self.spec_margin_n as i64 - slots.len() as i64).abs() <= 1 { 4.0 }
                              else if self.spec_margin >= 0.06 { 2.0 } else { 1.0 };
                let every = (SPEC_ARM_PROBE_EVERY as f64 * backoff) as u32;
                let every = if v2_idle.is_some() { v2_every as u32 } else { every };
                let due = self.spec_probe_ctr >= if first { SPEC_ARM_PROBE_FIRST } else { every };
                let refresh = v2_idle.map_or(false, |v| v.1);
                let due = due && !(v2_idle.is_some() && self.in_prefill); // v2: a window inside a prefill measures nothing
                let due = due && !(v2_idle.is_some() && self.meter_settle > 0); // ULTRA: nor one in the settle steps
                if due && (!self.arm_fresh(idle, slots.len()) || refresh) {
                    // ULTRA v2: the window is the arm's fold length (the speculation arm's tokens per step are noisy)
                    let win = if v2_idle.is_some() { spec_v2_fold(idle) } else { SPEC_ARM_PROBE_WIN };
                    self.spec_probe = Some((idle as u8, win));
                    if v2_idle.is_some() {
                        self.probe_fold[idle] = true;
                        // the window's fold must be the probe's own steps, not older ones of the same arm
                        self.spec_win_tok[idle] = 0.0; self.spec_win_ms[idle] = 0.0; self.spec_win_acc[idle] = 0;
                    }
                    self.spec_probe_ctr = 0;
                    probe_force = Some(idle == ARM_SHARED);
                    self.shared_probe_ctr = 0;
                    crate::metrics::spec_lanes_probe(idle);
                    if self.policy_logs < 64 {
                        self.policy_logs += 1;
                        crate::dreprintln!("[exl3-serve] spec-lanes auto: probing the {} arm for {} steps with {} busy lanes \
                           (its measured rate went stale; the pick prices both arms by delivery)",
                            if idle == ARM_SHARED { "shared plain" } else { "serial speculation" }, SPEC_ARM_PROBE_WIN, slots.len());
                    }
                }
            }
        }
        if mtp_on && probe_force.unwrap_or_else(|| self.use_shared_plain(&slots)) {
            self.shared_streak += 1;
            self.shared_probe_ctr = 0; // a shared round ran: the scale just learned, no probe owed
            // Stale-estimate refresh: lanes' speculative-speed EMAs freeze while they ride shared steps, so every
            // 128th shared round one lane (rotating) runs a serial speculative round instead; the others skip
            // this iteration. ~0.5% of rounds; keeps the cost-based choice tracking the content.
            let refresh_every = if self.spec_margin >= 0.12 { 256 } else { 128 };
            if probe_force.is_none() && !spec_meter_v2() && self.spec_policy == SpecLanes::Auto && self.shared_streak % refresh_every == 0 {
                let s = slots[(self.shared_streak / 128) % slots.len()];
                if self.lanes[s].as_ref().map_or(false, |l| l.mtp.is_some()) {
                    // shared-refresh: slots.len() >= 2 in this arm, so the DF2 chain (a single
                    // token stream) stays out of it — the pack-MTP round refreshes the EMA.
                    return self.step_mtp(s, false);
                }
            }
            crate::metrics::lane_pack_round(2); // arm = shared_plain
            return self.step_plain(&slots, true);
        }
        self.shared_streak = 0;
        // S-C4XOVER: F4's single-step shared probe is superseded by the arm-rate probe
        // above (same idea, windowed, both directions, feeding measured rates).
        // S-B9-CF-PACK4 `--lane-pack on`: two packable busy lanes share ONE packed speculative
        // verify. `pack_plan` is None with the flag off or fewer than two busy lanes, so both of
        // those take the loops below untouched (off == master; a lone request never packs).
        if mtp_on {
            if let Some(plan) = self.pack_plan(&slots) {
                return self.step_packed(&slots, plan);
            }
        }
        // CF-FCFS: the fcfs arm steps ONE lane (the front) and leaves the rest untouched this
        // scheduler step; the MTP-off lanes still batch together exactly as below. rr (the default)
        // runs the original loop, unchanged and in slot order.
        if self.lane_order == LaneOrder::Fcfs {
            let mut plain: Vec<usize> = Vec::new();
            let mut capable: Vec<(usize, u64, usize)> = Vec::new();
            for &s in &slots {
                let lane = self.lanes[s].as_ref().unwrap();
                if mtp_on && lane.mtp.is_some() {
                    capable.push((s, lane.admit_seq, lane.fcfs_credit));
                } else {
                    plain.push(s);
                }
            }
            let front = fcfs_front(&capable, self.lane_quantum);
            match front {
                Some(s) => {
                    let before = self.lanes[s].as_ref().unwrap().generated;
                    self.step_mtp(s, slots.len() == 1)?;
                    // step_mtp may have FINISHED this lane (stop / length / cancel / a dead client):
                    // it frees the slot (self.lanes[s] = None) before returning. The front lane is
                    // then simply gone — its turn ended early and the next step picks the next-oldest
                    // — so every post-step access must tolerate the slot being empty.
                    let lane = match self.lanes[s].as_mut() {
                        Some(lane) => lane,
                        None => {
                            if !plain.is_empty() {
                                self.step_plain(&plain, false)?;
                            }
                            return Ok(());
                        }
                    };
                    let delta = lane.generated - before;
                    let (mut seq, mut credit) = (lane.admit_seq, lane.fcfs_credit);
                    // fcfs_charge works on the replicated (seq, credit) tuple; BOTH fields are
                    // written back every step (the credit accumulation lives inside charge —
                    // forgetting this write-back on the no-rotate path pins credit at 0 and the
                    // front lane never reaches the quantum: run-to-completion again).
                    let rotated = fcfs_charge(&mut seq, &mut credit, delta, self.lane_quantum, &mut self.admit_ctr);
                    lane.fcfs_credit = credit;
                    if rotated {
                        lane.admit_seq = seq; // re-stamped: the queue position IS the sequence number
                        self.fcfs_rots += 1; // TOTAL rotations; the first 16 are logged and the total is printed at a clean idle point
                        if self.fcfs_rots <= 16 {
                            crate::dreprintln!("[exl3-serve] lane-order fcfs: slot {s} used its quantum ({} tokens) \
                                       and rotates to the back (rotation {}/16 logged)", self.lane_quantum, self.fcfs_rots);
                        }
                    }
                }
                None => {
                    // Unreachable with the charge-at-quantum bookkeeping: a lane's credit is reset
                    // the moment it reaches the quantum, so some lane always has credit < quantum.
                    // Kept as a guard: this step services no capable lane.
                }
            }
            if !plain.is_empty() {
                self.step_plain(&plain, false)?;
            }
            return Ok(());
        }
        let mut plain: Vec<usize> = Vec::new();
        for &s in &slots {
            let lane = self.lanes[s].as_ref().unwrap();
            if mtp_on && lane.mtp.is_some() {
                self.step_mtp(s, slots.len() == 1)?;
            } else {
                plain.push(s);
            }
        }
        if !plain.is_empty() {
            self.step_plain(&plain, false)?;
        }
        Ok(())
    }
}
