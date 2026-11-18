//! The scheduler loop and the plain decode step: `run`, `die`, `step_plain` (shared batch, one row per lane), `regime_ordered`, `idle_recv`,
//! and the prime-batch switches. Another `impl Exl3Scheduler` block; the TP loops (`run_tp_head`, `run_tp_mirror`) are in `tp_serve.rs`.

use super::*;

impl Exl3Scheduler {
    /// S-B9-REL-QSAREGIME: `slots` with the lanes at/below the sparse switch first (see step_plain).
    fn regime_ordered(&self, slots: &[usize]) -> Vec<usize> {
        let lim = regime_switch(self.model.qsa_live(usize::MAX), crate::exl3_forward::row_regime_on(), self.model.qsa_limit());
        regime_ordered_slots(slots, |s| self.lanes[s].as_ref().map_or(0, |l| l.pos), lim)
    }

    /// `sync_head` (CF-P1e): the lanes are MTP-capable but this round is a shared plain step — keep each lane's draft
    /// head exactly as a control round does (write the head's row p KV-only from the lane's saved tap BEFORE the
    /// step; snapshot the step's tap row into the lane's window AFTER it), so its next speculative round is valid.
    pub(super) fn step_plain(&mut self, slots: &[usize], sync_head: bool) -> Result<()> {
        // S-B9-REL-QSAREGIME: lanes at/below the sparse-attention switch first (stable; the identity
        // unless the lanes straddle the switch), so forward_step keeps their rows on the dense
        // kernels when a longer lane shares the step — a lane's bits must not depend on its
        // neighbours. Everything below indexes rows through this one list. Under TP every rank
        // orders from the same lockstep positions.
        let ordered = self.regime_ordered(slots);
        let slots: &[usize] = &ordered;
        // S-B9-REL-MTPREPROBE: an MTP-off lane under the re-probe policy keeps the SAME
        // head sync an armed lane gets — prime its row p from the saved tap BEFORE the step
        // (the 1-layer head row a control round would write) and snapshot the step's tap
        // AFTER it — so its head KV stays contiguous and its re-arm is exactly a
        // control-round seam. --mtp-reprobe=off primes only when sync_head, as before.
        let reprobe_primes = self.reprobe_on && self.model.mtp.is_some();
        if sync_head || reprobe_primes {
            let model = self.model.clone();
            if let Some(head) = model.mtp.as_ref() {
                // S-C4PRIMEA: collect the step's primeable lanes. When EVERY lane has a real
                // tap row and there are >= 2, ONE batched call (xq_gather_tap_rows + a single
                // re-prime pass at m = lanes + xq_attn_kv_prefill_pairs) replaces the per-lane
                // mtp_plain_prime loop. Any other shape (a sentinel tap, a skipped lane, one
                // lane) runs the VERBATIM legacy loop below — byte-identical to the old code,
                // which is also what the control site and solo re-probe steps keep running.
                let mut lanes: Vec<(usize, i32, usize, usize)> = Vec::with_capacity(slots.len());
                let mut batch_ok = prime_batch_on();
                for &s in slots {
                    let (b, p, tr) = { let l = self.lanes[s].as_ref().unwrap(); (l.last_tok as i32, l.pos, l.mtp) };
                    let tr = tr.or({
                        // an off rearmable lane: its saved tap row (0 once a primed step snapshotted)
                        let l = self.lanes[s].as_ref().unwrap();
                        if l.reprobe.keep_primed() { Some(l.reprobe.prime_tap_row) } else { None }
                    });
                    match tr {
                        Some(tap_row) if tap_row != usize::MAX => lanes.push((s, b, p, tap_row)),
                        _ => batch_ok = false,
                    }
                }
                batch_ok = batch_ok && lanes.len() >= 2 && lanes.len() <= 16; // re-prime width cap; mtp_plain_prime_batched re-checks vs the scratch width
                if batch_ok {
                    if prime_batch_xcheck_take() {
                        // S-C4PRIMEA: ONE-TIME-per-boot GPU identity gate on the real path:
                        // the legacy per-lane head-KV rows vs the batched bytes of the same rows.
                        for &(s, b, p, tr) in &lanes { model.mtp_plain_prime(&mut self.sc, head, s, b, p, tr)?; }
                        let want = model.head_kv_row_bytes(&mut self.sc, head, &lanes)?;
                        model.mtp_plain_prime_batched(&mut self.sc, head, &lanes)?;
                        let got = model.head_kv_row_bytes(&mut self.sc, head, &lanes)?;
                        let same = want == got;
                        eprintln!("[prime-batch] XCHECK {}: {} lanes, {} head-KV row bytes, per-lane == batched",
                                  if same { "PASS" } else { "FAIL" }, lanes.len(), got.len());
                        anyhow::ensure!(same, "prime-batch identity xcheck FAILED (first batched call this boot)");
                    } else {
                        model.mtp_plain_prime_batched(&mut self.sc, head, &lanes)?;
                    }
                    crate::metrics::prime_batch_round(lanes.len() as u64);
                    prime_batch_boot_log(lanes.len());
                } else {
                    for &s in slots {
                        let (b, p, tr) = { let l = self.lanes[s].as_ref().unwrap(); (l.last_tok as i32, l.pos, l.mtp) };
                        let tr = tr.or({
                            // an off rearmable lane: its saved tap row (0 once a primed step snapshotted)
                            let l = self.lanes[s].as_ref().unwrap();
                            if l.reprobe.keep_primed() { Some(l.reprobe.prime_tap_row) } else { None }
                        });
                        if let Some(tap_row) = tr {
                            model.mtp_plain_prime(&mut self.sc, head, s, b, p, tap_row)?;
                        }
                    }
                }
            }
        }
        // S-B9-REL-MTPREPROBE2 W1.3: when this shared step is exactly one slot and that
        // lane is an off re-probe lane, its plain sample must time the STEP ALONE — the
        // same discipline as a control round (prime, synchronize, then time). Without
        // the sync the async mtp_plain_prime's head GPU time lands inside the sample
        // (stream-ordered before forward_step's readback), inflating off_plain_ema and
        // biasing the policy toward staying ON. Multi-lane shared steps keep the CF-P1e
        // costing (primes + step): their number feeds the shared estimate, not the policy.
        let solo_off_sample = self.reprobe_on && slots.len() == 1
            && self.lanes[slots[0]].as_ref().map_or(false, |l| l.reprobe.keep_primed());
        if (sync_head || reprobe_primes) && solo_off_sample {
            self.model.dev().synchronize()?;
        }
        let t_round = std::time::Instant::now();   // CF-P1e: the shared round's cost = head primes + the step
        let toks: Vec<i32> = slots.iter()
            .map(|&s| self.lanes[s].as_ref().unwrap().last_tok as i32)
            .collect();
        let poss: Vec<usize> = slots.iter()
            .map(|&s| self.lanes[s].as_ref().unwrap().pos)
            .collect();
        // API-parity G2: sampled lanes sample ON DEVICE (xq_sample_rows after the argmax;
        // the same sampler the MTP verify rows use) — no 248K-float host readback per token.
        let rows: Vec<Option<crate::exl3_forward::SampParams>> = slots.iter()
            .map(|&s| self.lanes[s].as_ref().unwrap().samp_rows(1)[0])
            .collect();
        self.model.set_sampling(&mut self.sc, &rows)?;
        let pens: Vec<Option<crate::exl3_forward::PenParams>> = slots.iter()
            .map(|&s| self.lanes[s].as_ref().unwrap().pen)
            .collect();
        self.model.set_penalties(&mut self.sc, &pens)?;
        // W2-EXL3: per-row schema masks for the armed schema lanes (nothing staged — and no
        // mask node in the step's graph — when no lane in this step carries a live constraint).
        {
            let masks: Vec<Option<std::sync::Arc<Vec<u32>>>> = slots.iter().map(|&s| {
                let l = self.lanes[s].as_ref().unwrap();
                match (&l.schema, l.schema_armed) {
                    (Some(sm), true) => Some(sm.allowed(l.schema_state)),
                    _ => None,
                }
            }).collect();
            let rows: Vec<Option<&[u32]>> = masks.iter().map(|m| m.as_deref().map(|v| &v[..])).collect();
            self.model.set_vmask(&mut self.sc, &rows)?;
        }
        let ids = self.model.forward_step(&mut self.sc, &toks, &poss, slots, None)?;
        // S-B9: one agreed ms for the step (the HEAD's under TP — a rank-local clock must
        // never drive a decision), reused by the shared estimate and the off-lane sampler.
        let own_ms = t_round.elapsed().as_secs_f64() * 1e3;
        let plain_ms_head;
        if self.tp.is_some() {
            // TP-C: lockstep on the batched plain step's emitted ids (no wall-clock decision here)
            let w: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
            let (ms, _, _) = self.tp_round("plain-step", 0, slots.len(), &w, own_ms, 0.0, false, false)?;
            if sync_head {
                self.learn_shared_ms(slots.len(), ms);
                // S-C4XOVER: a shared plain step delivers one token to every busy lane;
                // fold it into the shared arm's measured rate (2+ lanes: a lone-lane
                // sync step is a reprobe control, not the shared arm at work).
                if slots.len() >= 2 { self.arm_acc(ARM_SHARED, slots.len() as f64, ms); }
            }
            plain_ms_head = ms;
        } else {
            if sync_head {
                self.learn_shared_ms(slots.len(), own_ms);
                if slots.len() >= 2 { self.arm_acc(ARM_SHARED, slots.len() as f64, own_ms); }
            }
            plain_ms_head = own_ms;
        }
        if let Some(g) = self.gate.as_mut() { g.plain += slots.len() as u64; }
        if sync_head || self.reprobe_on {
            let model = self.model.clone();
            // S-B9-REL-MTPREPROBE2 W1.5: --a-xcheck must see these plain-step snapshots
            // too (a control round records its row the same way) — without the record
            // the FIRST speculative round after a re-arm verifies against the lane's
            // last RECORDED (verify-era) row and logs a false "clobbered" line.
            let xcheck = crate::exl3_forward::a_xcheck_on();
            for (j, &s) in slots.iter().enumerate() {
                if self.lanes[s].as_ref().map_or(false, |l| l.mtp.is_some()) {
                    model.tap_snapshot_row(&mut self.sc, s, j)?;
                    if xcheck {
                        if let Err(e) = model.tap_xcheck_record(&self.sc, s, 0) {
                            eprintln!("[a-xcheck] record failed: {e:#}");
                        }
                    }
                    self.lanes[s].as_mut().unwrap().mtp = Some(0);
                } else if self.reprobe_on && self.lanes[s].as_ref().map_or(false, |l| l.reprobe.keep_primed()) {
                    // S-B9: keep the off lane's tap fresh (row 0) so its next head prime is valid
                    model.tap_snapshot_row(&mut self.sc, s, j)?;
                    if xcheck {
                        if let Err(e) = model.tap_xcheck_record(&self.sc, s, 0) {
                            eprintln!("[a-xcheck] record failed: {e:#}");
                        }
                    }
                    self.lanes[s].as_mut().unwrap().reprobe.prime_tap_row = 0;
                }
            }
            // S-B9: off lanes' plain steps count toward the re-probe backoff. A lane ALONE
            // in this batch also yields an exact per-token plain cost at the CURRENT
            // context (the engine's ema_plain can be stale-low: plain grows with context)
            // — it raises the probe's plain reference, so probes cannot fail on a stale
            // baseline.
            if self.reprobe_on {
                for &s in slots {
                    let off = self.lanes[s].as_ref()
                        .map_or(false, |l| l.mtp.is_none() && l.reprobe.keep_primed());
                    if off {
                        let sample = if slots.len() == 1 && plain_ms_head > 0.0 { plain_ms_head } else { 0.0 };
                        self.lanes[s].as_mut().unwrap().reprobe.observe_plain(sample, &self.reprobe_cfg);
                    }
                }
            }
        }

        for (j, &s) in slots.iter().enumerate() {
            let t = {
                let lane = self.lanes[s].as_mut().unwrap();
                let next = lane.think_filter(s, ids[j] as u32, 0, 1); // THINKBUDGET1
                lane.ctr = lane.ctr.wrapping_add(1);
                lane.st_plain += 1;
                lane.arms.push('p', 1);
                if lane.first_toks.len() < FIRST_TOKS { lane.first_toks.push(next); }
                lane.last_tok = next;
                lane.pos += 1;
                lane.generated += 1;
                crate::lane_wide::note_token();
                lane.history.push(next);
                if lane.history.len() > 256 { lane.history.drain(0..128); }
                if lane.copy.is_some() { lane.copy_new.push(next); } // WIDEVERIFY1
                if let Some(r) = lane.rec_reply.as_mut() { r.push(next); } // COPYREC1
                next
            };
            let (finish, reason) = {
                let lane = self.lanes[s].as_mut().unwrap();
                if lane.generated >= lane.max_new {
                    (true, "length")
                } else if self.eos.contains(&t) && !lane.ignore_eos && lane.generated >= lane.min_new {
                    (true, "stop")
                } else if let Some(sm) = lane.schema.clone().filter(|_| lane.schema_armed) {
                    // W2-EXL3: a schema turn ends at the end of its JSON document. The loop
                    // detector does not run on an armed lane: the mask bounds the output, and a
                    // valid document may legitimately repeat (a long array of equal values).
                    match schema_feed(&sm, &mut lane.schema_state, &mut lane.schema_armed, t) {
                        SchemaStep::Continue => (false, ""),
                        SchemaStep::Complete => (true, "stop"),
                        SchemaStep::Invalid => {
                            crate::reprintln!("[schema] slot {s}: token {t} leaves no valid continuation \
                                               — ending the turn with an error");
                            (true, SCHEMA_DEAD_END)
                        }
                    }
                } else if lane.loop_feed(t) {
                    (true, "loop_detected") // WP08
                } else if let Some(sm) = lane.schema.clone() {
                    // not armed yet (reasoning): this token may be the one that arms it
                    schema_feed(&sm, &mut lane.schema_state, &mut lane.schema_armed, t);
                    (false, "")
                } else {
                    (false, "")
                }
            };
            let tp_on = self.tp.is_some();
            // item 5(a): ids[j] is the choice of the step's logits row j
            if finish && reason == "stop" { self.send_stop_info(s, j, t, false); }
            if finish { self.publish_kv(); } // issue #24: the lane's final position (the peaks include the round that ends the request)
            // ids[j] is also the choice of logits row j for the log probabilities
            if let Some(n) = self.lanes[s].as_ref().and_then(|l| l.logprobs).filter(|_| self.lp_reader()) {
                let tx = self.lanes[s].as_ref().unwrap().tx.clone();
                self.send_logprob(&tx, false, j, t, n);
            }
            let lane = self.lanes[s].as_mut().unwrap();
            // TP-D: under TP a failed send is not a cancel here (the head's next sweep ships it)
            if lane.tx.send(TokEvent::Tok(t)).is_err() && !tp_on {
                // WP02: the client is gone — cancel now, free the slot
                lane.log_cancel(s);
                self.release_lane(s, false);
                continue;
            }
            if finish {
                if reason == "loop_detected" { lane.log_loop(s, &self.tok); }
                lane.log_stats(s, reason);
                self.model.dump_expert_hist(); // TP-I #6 diagnostic (no-op unless --tp-ep-hist)
                self.model.dump_esel_hist(); // S-A3-o diagnostic (no-op unless --exl3-esel-hist)
                self.model.dump_route_log(); // CF-P1e step 0 diagnostic (no-op unless --exl3-route-log)
                let _ = lane.tx.send(TokEvent::Finish { reason: reason.to_string() });
                self.release_lane(s, crate::lane_reserve::normal_finish(reason));
                self.model.dhead_request_report(); // DHEADP fallback line (penalized requests) + DHEAD XCHECK lines (when on)
            }
        }
        Ok(())
    }

    pub(super) fn run(mut self, mut rx: UnboundedReceiver<BatchRequest>) {
        // HOST / RO-7: the mask this thread inherited from `pub fn run`'s pin (--cpu-affinity)
        if let Ok(m) = crate::cpu_affinity::current_thread_affinity() {
            println!("[exl3-serve] scheduler thread cpu mask: {}", crate::cpu_affinity::format_cpu_list(&m));
        }
        // Requests beyond the lane count WAIT in arrival order (the standard server contract);
        // they used to be refused as "server busy", which reached a non-streaming client as an
        // empty HTTP 200. A client that hangs up while queued is dropped before admission.
        let mut waiting: std::collections::VecDeque<BatchRequest> = std::collections::VecDeque::new();
        let mut rots_printed = 0usize;
        let mut mixed_printed = 0u64;
        loop {
            if self.fatal.is_some() {
                return self.die(waiting, rx);
            }
            self.pool_emergency(); // PFXPOOL2 (rank-local, rate-limited)
            if waiting.is_empty() && self.lanes.iter().all(|l| l.is_none()) {
                // Clean idle point: report the TOTAL number of fcfs rotations this scheduler made
                // (the per-rotation log caps at 16 lines; G1 needs the true count).
                if self.lane_order == LaneOrder::Fcfs && self.fcfs_rots > rots_printed {
                    crate::drprintln!("[exl3-serve] lane-order fcfs idle: {rots_printed}->{} total rotations", self.fcfs_rots);
                    rots_printed = self.fcfs_rots;
                }
                // 9a(iii): the lazily captured mixed-step graphs, whenever their number grew
                if let Some(line) = mixed_graphs_idle_line(mixed_printed, crate::metrics::mixed_step_graphs(), self.width) {
                    crate::drprintln!("{line}");
                    mixed_printed = crate::metrics::mixed_step_graphs();
                }
                self.publish_kv(); // issue #24: the idle value (lanes freed outside a step included)
                if let Some(line) = self.pack_idle_line() { crate::drprintln!("{line}"); } // S-B9-CF-PACK4
                crate::metrics::sched_idle(); // H3: about to block for work (busy must not stick at 1)
                if self.turn.as_ref().map_or(false, |a| a.nt == crate::turn_resume::NtMode::On) {
                    // TURNRESUME: background next-turn passes run here (every lane idle, nothing waiting)
                    match self.turn_idle(&mut rx) {
                        TurnWake::Req(req) => waiting.push_back(req),
                        TurnWake::Closed => break,
                        TurnWake::Again => continue,
                    }
                } else {
                    match self.idle_recv(&mut rx) {
                        Some(req) => waiting.push_back(req),
                        None => break,
                    }
                }
            }
            // COPYREC1: DELETE /v1/copy-record, applied before the next admission
            if crate::copy_record::take_clear_request() { self.record_empty(); }
            while let Ok(req) = rx.try_recv() {
                // MAXBATCH1: an arrival beyond what the cap can run at once — every lane busy,
                // or the queue already holds the overflow (a burst on an idle box lands here
                // too: all 12 drain before admission runs). Count it (the cap's engagement
                // counter; the 2026-10-21 soak showed the lanes-already-full-only test misses
                // exactly those bursts).
                if self.lanes.iter().filter(|l| l.is_some()).count() + waiting.len() >= self.lanes.len() {
                    crate::metrics::over_cap_queued();
                }
                waiting.push_back(req);
            }
            while self.fatal.is_none() && self.lanes.iter().any(|l| l.is_none()) {
                if self.reserve.is_some() {
                    // LANERES1: plan-driven admission — snapshot-first, reservations, bounded
                    // patience, rate-limited forced takes. Same queue, same rx; only WHICH
                    // waiting request goes to WHICH lane changes, and only while the policy is
                    // on (--lane-reserve). Off = the original body below, byte-for-byte.
                    while let Ok(req) = rx.try_recv() {
                        // MAXBATCH1: over-cap arrival (beyond what the cap runs at once) — count it.
                        if self.lanes.iter().filter(|l| l.is_some()).count() + waiting.len() >= self.lanes.len() {
                            crate::metrics::over_cap_queued();
                        }
                        waiting.push_back(req);
                    }
                    if !waiting.is_empty() {
                        waiting.retain(|r| !r.tx.is_closed());
                    }
                    let now = std::time::Instant::now();
                    let slots = self.slot_views();
                    let wvs: Vec<crate::lane_reserve::WaitView> = waiting.iter().map(|r| crate::lane_reserve::WaitView {
                        arrived_at: r.received_at,
                        own_slot: self.own_snapshot_slot(&r.prompt),
                    }).collect();
                    let Some(cfg) = self.reserve else { unreachable!("checked above") };
                    match crate::lane_reserve::plan(&cfg, &mut self.reserve_st, now, &slots, &wvs) {
                        crate::lane_reserve::Plan::Admit { waiting_index, allowed, kind } => {
                            if let Some(req) = waiting.remove(waiting_index) {
                                if req.tx.is_closed() {
                                    continue;
                                }
                                self.admit_scoped(req, &allowed, kind);
                                crate::metrics::lane_reserve_waiting(waiting.len() as u64);
                            }
                        }
                        crate::lane_reserve::Plan::Wait { until } => {
                            crate::metrics::lane_reserve_waiting(waiting.len() as u64);
                            if self.lanes.iter().any(|l| l.is_some()) {
                                break; // a round is running; re-plan after the next step
                            }
                            // Every lane free but reserved: bounded blocking wait for a new
                            // request or the next deadline (20 ms tick; no engine work exists).
                            loop {
                                if self.fatal.is_some() { break; }
                                if until.map_or(false, |d| std::time::Instant::now() >= d) { break; }
                                match rx.try_recv() {
                                    Ok(req) => {
                                        // MAXBATCH1: over-cap arrival (beyond what the cap runs at once) — count it.
                                        if self.lanes.iter().filter(|l| l.is_some()).count() + waiting.len() >= self.lanes.len() {
                                            crate::metrics::over_cap_queued();
                                        }
                                        waiting.push_back(req);
                                        break;
                                    }
                                    Err(_) => std::thread::sleep(std::time::Duration::from_millis(20)),
                                }
                            }
                        }
                    }
                    continue; // re-plan from the top of this while
                }
                if packpf_on() {
                    // PACKPF hold (--prefill-pack-min / --prefill-pack-wait): run a decode round first
                    // so more lanes free up and the next packed pass carries more prompts
                    if self.pf_hold(&waiting) { break; }
                    if waiting.len() >= 2 || !self.pf_jobs.is_empty() { self.pack_prefill_waiting(&waiting); }
                }
                let Some(req) = waiting.pop_front() else { break };
                if req.tx.is_closed() {
                    continue;
                }
                self.admit(req);
            }
            if !waiting.is_empty() {
                waiting.retain(|r| !r.tx.is_closed());
            }
            if self.fatal.is_some() || self.lanes.iter().all(|l| l.is_none()) {
                continue;
            }
            if let Err(e) = self.step() {
                self.fail_step(&e);
            }
        }
    }

    /// WP02 liveness: the CUDA context is poisoned (sticky error) — nothing on this engine can run
    /// again. Mark it DEAD (/health 503, new requests 503 at the HTTP layer), fail every live and
    /// queued request with an error Finish, then either exit 70 (--exit-on-fatal: the supervisor
    /// restarts the server) or keep answering every late request with the same error.
    fn die(mut self, mut waiting: std::collections::VecDeque<BatchRequest>,
           mut rx: UnboundedReceiver<BatchRequest>) {
        let why = self.fatal.clone().unwrap_or_default();
        crate::server::set_engine_dead(&why);
        let reason = format!("error: engine stopped (fatal CUDA error: {why})");
        eprintln!("[exl3-serve] FATAL: {reason}");
        for s in 0..self.lanes.len() {
            if let Some(lane) = self.lanes[s].take() {
                let _ = lane.tx.send(TokEvent::Finish { reason: reason.clone() });
            }
        }
        self.publish_kv(); // issue #24: a dead engine holds no request
        while let Ok(req) = rx.try_recv() {
            waiting.push_back(req);
        }
        for req in waiting.drain(..) {
            let _ = req.tx.send(TokEvent::Finish { reason: reason.clone() });
        }
        if self.exit_on_fatal {
            eprintln!("[exl3-serve] FATAL: exit 70 (--exit-on-fatal, default; a supervisor restarts the server)");
            std::thread::sleep(std::time::Duration::from_millis(300)); // let the error events flush
            crate::logq::flush_and_exit(std::time::Duration::from_millis(300), 70); // H7 + LR-1
        }
        while let Some(req) = rx.blocking_recv() {
            let _ = req.tx.send(TokEvent::Finish { reason: reason.clone() });
        }
    }

    /// PFXPOOL2: the scheduler's idle wait for the next request. With the pool on it wakes every second to run
    /// `pool_emergency`, so an idle server also gives memory back when its box comes under the floor (the box's other
    /// processes keep running while it waits).
    pub(super) fn idle_recv(&mut self, rx: &mut UnboundedReceiver<BatchRequest>) -> Option<BatchRequest> {
        if !self.pool.on() { return rx.blocking_recv(); }
        let rt = match tokio::runtime::Builder::new_current_thread().enable_time().build() {
            Ok(r) => r,
            Err(_) => return rx.blocking_recv(),
        };
        loop {
            match rt.block_on(async { tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv()).await }) {
                Ok(v) => return v,
                Err(_) => self.pool_emergency(),
            }
        }
    }
}

// ---- S-C4PRIMEA: --exl3-prime-batch (default on; "0"/"off" disables) — batch the shared
// step's per-lane mtp_plain_prime head-row writes into ONE gather + ONE re-prime pass.
// --exl3-prime-batch-xcheck (default on): the FIRST batched call per boot bit-compares the
// legacy per-lane head-KV rows against the batched ones on the real path (one-time cost).
fn prime_batch_on() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        !matches!(crate::opts::var(crate::opt!("exl3-prime-batch")).as_deref(), Ok("0") | Ok("off"))
    })
}

fn prime_batch_xcheck_take() -> bool {
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if matches!(crate::opts::var(crate::opt!("exl3-prime-batch-xcheck")).as_deref(), Ok("0") | Ok("off")) {
        return false;
    }
    !DONE.swap(true, std::sync::atomic::Ordering::Relaxed)
}

fn prime_batch_boot_log(lanes: usize) {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        crate::deprintln!("PRIME-BATCH armed: shared-step head-row primes batched ({lanes} lanes this step; \
xq_gather_tap_rows + ONE re-prime pass + xq_attn_kv_prefill_pairs; --exl3-prime-batch off disables)");
    }
}
