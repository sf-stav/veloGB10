//! TP serving of the EXL3 backend: the head and mirror scheduler loops (`run_tp_head`, `run_tp_mirror`), request shipping and strip rules,
//! `run_tp_head_serve` / `run_tp_node_serve`, attach, boot agreement digest and `parse_spec_sampling`. Another `impl Exl3Scheduler` block plus a `TpSync` helper block.
//! The node is zero-config: the head ships model, config and calibration; do not add a side channel (AGENTS.md section 2, item 10).

use super::*;

// ---- TP-D: serving at TP=2 (design §5: run_tp_head / run_tp_mirror around the SPMD scheduler) ----

/// A request the EXL3 path refuses before admission (checked on the head BEFORE an Admit ships,
/// so a refused request never reaches the node).
fn exl3_reject(req: &BatchRequest) -> Option<String> {
    if req.prompt.is_empty() {
        return Some("error: empty prompt".into());
    }
    None
}

/// PR #30 / #31 ports, owner rule (fall back, never refuse): under TP a request's schema mask is dropped (wired at
/// TP=1), and its log probabilities at a world without row-gather kernels (they are served at TP=2 / 4). The HTTP layer already decides this (AppState schema_ok / logprobs_ok: the reply carries
/// `x-json-schema-enforced: none` / `logprobs: null`); this keeps the head, the node and any direct caller
/// consistent. Said once per boot.
pub(super) fn exl3_tp_strip(req: &mut BatchRequest, world: i32) {
    let had_schema = req.schema.take().is_some();
    // logprobs at TP (owner 2026-11-02): served at world 2 and 4 (the row-gather kernels exist there)
    let had_lp = !matches!(world, 2 | 4) && req.logprobs.take().is_some();
    if had_schema || had_lp {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| crate::reprintln!("[exl3-serve] TP: a request's schema mask / log probabilities are not applied \
under TP yet; such requests are served without them (this line prints once per boot)"));
    }
}

/// The seed a request without one gets (admit's TP=1 derivation, minus the slot term).
fn entropy_seed() -> u64 {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64).unwrap_or(0);
    t ^ 0x5DEECE66D
}

impl Exl3Scheduler {
    /// TP-D: when the engine goes idle, one line of lockstep counters (the session's agree receipts).
    fn tp_log_idle(&mut self, step_no: u64) {
        if self.lanes.iter().all(|l| l.is_none()) {
            // S-B9-CF-PACK5: the packed arm's counters on the node too (its mirror loop never reaches
            // the head's idle-point line); printed once per change, so the head's own call is a no-op
            if let Some(line) = self.pack_idle_line() { crate::drprintln!("{line}"); }
            if let Some(t) = self.tp.as_mut() {
                crate::drprintln!("[tp-serve] rank {} idle after control step {step_no}: lockstep agrees {} ok, pre-verify agrees {} ok{}",
                         t.rank, t.agrees, t.pre_agrees,
                         if t.ident { format!(", graphed digests {} compared, {} differ", t.ident_bufs, t.ident_bad) } else { String::new() });
                crate::drprintln!("{}", t.wp.line(t.rank)); // H5: twin of the line above - never direct on a served path
                t.wp = WaitProf::default();
            }
        }
    }

    /// VIS-4: the binary channel — raw image-row payloads written right after a Step frame, in the
    /// order of that Step's image admits (every node reads them back by `image_bytes`).
    fn tp_ship_raw(&mut self, payloads: &[Vec<u8>]) -> Result<()> {
        if payloads.is_empty() { return Ok(()); }
        use std::io::Write;
        let t = self.tp.as_mut().context("TP ship without TP state")?;
        let t0 = std::time::Instant::now();
        let total: usize = payloads.iter().map(|p| p.len()).sum();
        for s in t.ctl.iter_mut() {
            for p in payloads { s.write_all(p)?; }
            s.flush()?;
        }
        crate::drprintln!("[exl3-serve] VIS-4: shipped {} image payload(s), {:.1} MB to {} node(s) in {:.1} ms",
                 payloads.len(), total as f64 / 1e6, t.ctl.len(), t0.elapsed().as_secs_f64() * 1e3);
        Ok(())
    }

    fn tp_ship(&mut self, m: &crate::tp_serve::ServingMsg) -> Result<()> {
        let live = self.lanes.iter().any(|l| l.is_some());
        let t = self.tp.as_mut().context("TP ship without TP state")?;
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        for s in t.ctl.iter_mut() {
            crate::tp_serve::send_serving(s, m)?;
        }
        if live { t.wp.rec(6, t0, s0); }
        Ok(())
    }

    /// TP-D: a TP rank cannot serve on after a failed lockstep, a failed step or a fatal admit:
    /// fail every live, pending and queued request loudly, mark the engine dead (/health 503 while
    /// the process winds down), and return the error — the caller exits 70.
    fn tp_fail(&mut self, why: String, waiting: &mut std::collections::VecDeque<BatchRequest>,
               rx: Option<&mut UnboundedReceiver<BatchRequest>>) -> anyhow::Error {
        crate::server::set_engine_dead(&why);
        let reason = format!("error: TP engine stopped ({why})");
        for s in 0..self.lanes.len() {
            if let Some(lane) = self.lanes[s].take() {
                let _ = lane.tx.send(TokEvent::Finish { reason: reason.clone() });
            }
        }
        self.publish_kv(); // issue #24: a dead engine holds no request
        if let Some(rx) = rx {
            while let Ok(req) = rx.try_recv() { waiting.push_back(req); }
        }
        for req in waiting.drain(..) {
            let _ = req.tx.send(TokEvent::Finish { reason: reason.clone() });
        }
        anyhow::anyhow!("{why}")
    }

    /// TP-D: the node's mirror loop — apply each head `Step` (cancels, admits with the head's
    /// resolved sampling params + seed) in order, then run the identical SPMD step. The node's
    /// token channels have no receiver (sends are no-ops; no rank-local cancel exists under TP).
    /// Ok = the head shut the session down (or its stream closed); Err = a failed step (the node
    /// process exits and its supervisor re-arms).
    fn run_tp_mirror(mut self) -> Result<()> {
        use crate::tp_serve::{ServingMsg, TpEvent};
        let mut step_no: u64 = 0;
        let mut admitted: u64 = 0;
        let fast_ctl = crate::opts::var(crate::opt!("exl3-tp-fastctl")).map_or(true, |v| v != "0");
        crate::drprintln!("[exl3-serve] TP control (node): {}", if fast_ctl { "step-go over RDMA while live (TP-E)" } else { "TCP Step every step" });
        loop {
            self.pool_emergency(); // PFXPOOL2 (rank-local, rate-limited; an idle node waits on the head's stream)
            let live = self.lanes.iter().any(|l| l.is_some());
            if fast_ctl && live {
                let (hs, n) = self.tp.as_mut().context("mirror without TP state")?.step_go(step_no + 1, 0)?;
                anyhow::ensure!(hs == step_no + 1, "TP mirror: head step-go {hs} after local step {step_no}");
                if n == 0 {
                    step_no = hs;
                    self.step().with_context(|| format!("TP node: decode step {step_no} failed"))?;
                    self.tp_log_idle(step_no);
                    continue;
                }
                // events ride the TCP Step below (the head sent it after the go)
            }
            let msg = {
                let t = self.tp.as_mut().context("mirror without TP state")?;
                let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
                let r = crate::tp_serve::recv_serving(&mut t.ctl[0]);
                if live { t.wp.rec(5, t0, s0); }  // mid-request only: the head's inter-step host time + TCP
                match r {
                    Ok(m) => m,
                    Err(e) => {
                        // H7: END-OF-LIFE line — direct print; the node exits right after the
                        // return and no logq flush can be guaranteed there.
                        crate::deprintln!("[exl3-serve] TP node: head control stream closed ({e:#}) — session over after {step_no} steps, {admitted} requests");
                        return Ok(());
                    }
                }
            };
            match msg {
                ServingMsg::Shutdown => {
                    // H7: END-OF-LIFE line — direct print (same reason as the closed-stream line).
                    crate::deprintln!("[exl3-serve] TP node: Shutdown from the head after {step_no} steps, {admitted} requests");
                    return Ok(());
                }
                ServingMsg::Step(se) => {
                    anyhow::ensure!(se.step == step_no + 1, "TP mirror: head step {} after local step {step_no}", se.step);
                    step_no = se.step;
                    // PR #17 (ported): read ALL binary image payloads right after the Step frame,
                    // in admit order, BEFORE any admit()/prefill runs — prefill multiplexes
                    // HeadFlag traffic on the same control stream, so a payload still unread when
                    // the first admit runs could interleave with HeadFlag frames (stream desync /
                    // digest mismatch with 2+ images admitted in one Step). Digest checks unchanged.
                    // S-B9-REL-PRPORT3 (review R9): the pre-read lives in
                    // read_step_image_payloads (unit-tested); admit-order specs, then read.
                    let specs: Vec<(u64, u64)> = se.events.iter().filter_map(|ev| match ev {
                        TpEvent::Admit(w) => Some((w.image_bytes as u64, w.image_digest as u64)),
                        _ => None,
                    }).collect();
                    let payloads = {
                        let t = self.tp.as_mut().context("mirror without TP state")?;
                        read_step_image_payloads(&mut t.ctl[0], step_no, &specs)?
                    };
                    let mut payload_iter = payloads.into_iter();
                    for ev in se.events {
                        match ev {
                            TpEvent::Cancel { lane } => {
                                if let Some(mut l) = self.lanes.get_mut(lane).and_then(|l| l.take()) {
                                    self.record_release(&mut l); // COPYREC1
                                    l.log_cancel(lane);
                                }
                            }
                            TpEvent::RecordClear => self.record_empty(), // COPYREC1
                            TpEvent::Admit(w) => {
                                let (tx, _rx, _bl) = crate::server::tok_channel(); // mirror dummy; LR-5 gated off under TP
                                let min_p = w.min_p;
                                // LANERES3: the head's reservation scoping rides the wire;
                                // the mirror restricts its pick identically so every rank
                                // lands the same lane (the pick is a pure function of
                                // replicated state, and the restriction is part of that
                                // state now).
                                let scoping = w.reserve.clone();
                                let mut req = w.into_request(tx);
                                req.min_p = min_p;
                                // PR #17: this admit's image rows were already read (in admit
                                // order) right after the Step frame — see payloads above.
                                req.image_embeds = payload_iter.next().flatten();
                                admitted += 1;
                                match scoping {
                                    Some(sc) => {
                                        if !crate::lane_reserve::allowed_free(&self.lanes, &sc.allowed) {
                                            // LANERES3: the head requeued this admit (no allowed lane free
                                            // at apply time); the mirror skips it identically or the slot
                                            // tables diverge. The request re-arrives as a fresh admit event.
                                            crate::metrics::lane_reserve_tp_requeued();
                                        } else {
                                            self.admit_scoped(req, &sc.allowed, crate::lane_reserve::Kind::from_wire(sc.kind));
                                        }
                                    }
                                    None => self.admit(req),
                                }
                            }
                        }
                        if let Some(f) = self.fatal.clone() {
                            anyhow::bail!("TP node: admit failed at step {step_no}: {f}");
                        }
                    }
                    if self.lanes.iter().any(|l| l.is_some()) {
                        self.step().with_context(|| format!("TP node: decode step {step_no} failed"))?;
                    }
                    self.tp_log_idle(step_no);
                }
                other => anyhow::bail!("TP mirror: unexpected control message {other:?}"),
            }
        }
    }

    /// TP-D: the TP=2 HEAD's scheduler loop. Every scheduler-visible decision is taken HERE and
    /// shipped before it is applied: per iteration the head sweeps cancels (a closed client
    /// channel), admits waiting requests (validated here; the seed resolved here so sampled ranks
    /// draw the same stream), ships ONE `Step { events }` to every node, then applies exactly those
    /// events in order and runs the SPMD step the node runs. Everything else the step decides is a
    /// pure function of replicated state (slot pick, prefix-cache / WP16 / C1 resume points, EOS /
    /// max_new / loop stops) or is head-authoritative inside the step (round timing — TP-C's
    /// exchange; a prefill chunk's cancel — HeadFlag); the agree tripwires prove it every
    /// seam / round (and before every DDS verify).
    pub(super) fn run_tp_head(mut self, mut rx: UnboundedReceiver<BatchRequest>, streams: Vec<std::net::TcpStream>) -> Result<()> {
        use crate::tp_serve::{ServingMsg, StepEvents, TpEvent, WireRequest};
        if let Ok(m) = crate::cpu_affinity::current_thread_affinity() {
            crate::dprintln!("[exl3-serve] TP head scheduler thread cpu mask: {}", crate::cpu_affinity::format_cpu_list(&m));
        }
        self.tp.as_mut().context("run_tp_head without TP state")?.ctl = streams;
        let mut waiting: std::collections::VecDeque<BatchRequest> = std::collections::VecDeque::new();
        let mut step_no: u64 = 0;
        let mut rots_printed = 0usize;
        let mut mixed_printed = 0u64;
        let fast_ctl = crate::opts::var(crate::opt!("exl3-tp-fastctl")).map_or(true, |v| v != "0");
        crate::dprintln!("[exl3-serve] TP control: {}", if fast_ctl { "step-go over RDMA while live, TCP Step only with events (TP-E)" }
                 else { "TCP Step every step (--exl3-tp-fastctl=0)" });
        loop {
            self.pool_emergency(); // PFXPOOL2 (rank-local, rate-limited)
            if waiting.is_empty() && self.lanes.iter().all(|l| l.is_none()) {
                // Clean idle point (TP twin of the single-process loop): total fcfs rotations.
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
                match self.idle_recv(&mut rx) {
                    Some(req) => waiting.push_back(req),
                    None => {
                        let _ = self.tp_ship(&ServingMsg::Shutdown);
                        return Ok(());
                    }
                }
            }
            while let Ok(req) = rx.try_recv() {
                waiting.push_back(req);
            }
            let mut events: Vec<TpEvent> = Vec::new();
            let mut admits: std::collections::VecDeque<BatchRequest> = std::collections::VecDeque::new();
            let mut payloads: Vec<Vec<u8>> = Vec::new(); // VIS-4: image rows of this step's admits, in order
            let mut free = 0usize;
            for s in 0..self.lanes.len() {
                match &self.lanes[s] {
                    None => free += 1,
                    Some(l) if l.tx.is_closed()
                             || crate::server::stream_backlog_exceeded(l.tx.len()) => { // LR-5
                        if !l.tx.is_closed() {
                            crate::metrics::stream_backlog_cancel();
                            crate::dreprintln!("[exl3-serve] slot {s}: stream cancelled — unconsumed event backlog at the --stream-backlog-events limit (reader stalled; LR-5; Cancel shipped to the mirrors)");
                        }
                        events.push(TpEvent::Cancel { lane: s });
                        free += 1;
                    }
                    Some(_) => {}
                }
            }
            // TPPOOL (design v3 step 4): lanes this step's Cancels free before its admits apply, and the pages charged to
            // the admits it ships (the head's capacity gate; inert unless a lazy pool runs under TP)
            let pending_free: Vec<usize> = events.iter().filter_map(|e| match e { TpEvent::Cancel { lane } => Some(*lane), _ => None }).collect();
            let mut gate_reserved = 0usize;
            // COPYREC1: DELETE /v1/copy-record is a step event (after this step's cancels, before its admits) so every
            // rank empties its record at the same point
            if crate::copy_record::take_clear_request() && self.record.is_some() {
                events.push(TpEvent::RecordClear);
            }
            // LANERES3: plan-driven admission under TP — the same single-box policy
            // (`lane_reserve::plan`), head-authoritative. The node never runs the policy:
            // each admit's scoping rides the wire (`WireRequest::reserve`), so both ranks
            // restrict `admit_req`'s pick identically and hold the same slot tables.
            // --lane-reserve 0 (default) takes the plain FIFO body: byte-for-byte today.
            if self.reserve.is_some() {
                // LANERES3 fix (2026-10-22): EXACTLY ONE scoped admit per iteration. The
                // plan's `allowed` set is computed from `self.lanes` HERE, but admits apply
                // later (the events-apply phase below) — a sweep of several admits plans
                // them all against the same stale view, two can target one lane, and the
                // second then finds `free ∩ allowed = {}` and the request was REFUSED
                // ("server busy"), the §1b-bis.8 anti-pattern (9/60 requests at conc 4 in
                // the first structural cells run; the FIFO path is immune because its pick
                // recomputes the free set at apply time). One admit per iteration keeps
                // every plan's view equal to the state its admit acts on; bursts take one
                // extra scheduler iteration per request.
                waiting.retain(|r| !r.tx.is_closed());
                if free > 0 && !waiting.is_empty() {
                    let now = std::time::Instant::now();
                    let slots = self.slot_views();
                    let wvs: Vec<crate::lane_reserve::WaitView> = waiting.iter().map(|r| crate::lane_reserve::WaitView {
                        arrived_at: r.received_at,
                        own_slot: self.own_snapshot_slot(&r.prompt),
                    }).collect();
                    if let Some(cfg) = self.reserve {
                        match crate::lane_reserve::plan(&cfg, &mut self.reserve_st, now, &slots, &wvs) {
                            crate::lane_reserve::Plan::Admit { waiting_index, allowed, kind } => {
                                if let Some(mut req) = waiting.remove(waiting_index) {
                                    if !req.tx.is_closed() {
                                        exl3_tp_strip(&mut req, self.tp.as_ref().map_or(1, |t| t.world)); // before the Admit ships: head and node agree
                                        let gate = self.tp_pool_gate(&req, Some(allowed.as_slice()), &pending_free, gate_reserved);
                                        if let Some(why) = exl3_reject(&req) {
                                            let _ = req.tx.send(TokEvent::Finish { reason: why });
                                        } else if gate == super::tp_pool_gate::GateDecision::Hold {
                                            waiting.insert(waiting_index.min(waiting.len()), req); // TPPOOL: same place, next step re-plans
                                        } else if let super::tp_pool_gate::GateDecision::Refuse(why) = gate {
                                            let _ = req.tx.send(TokEvent::Finish { reason: why });
                                        } else {
                                            if let super::tp_pool_gate::GateDecision::Ship { pages } = gate { gate_reserved += pages; }
                                            if req.seed.is_none() {
                                                req.seed = Some(entropy_seed()); // head-resolved: every rank samples one stream
                                            }
                                            let mut w = WireRequest::from(&req);
                                            w.reserve = Some(crate::tp_serve::Scoping { allowed: allowed.clone(), kind: kind as u8 });
                                            if let Some(e) = req.image_embeds.as_ref().filter(|_| !req.image_spans.is_empty()) {
                                                // VIS-4: the rows ride the binary channel right after this step's frame
                                                let (bytes, digest) = crate::tp_serve::image_rows_wire(e);
                                                w.image_bytes = bytes.len() as u64;
                                                w.image_digest = digest;
                                                payloads.push(bytes);
                                            }
                                            events.push(TpEvent::Admit(w));
                                            admits.push_back(req);
                                            free -= 1;
                                        }
                                    }
                                }
                            }
                            crate::lane_reserve::Plan::Wait { until } => {
                                // Admission holds: every free lane is reserved for a waiting
                                // conversation's own snapshot. With a round running there is
                                // nothing to wait for HERE — the next step re-plans. With every
                                // lane free the head ticks (20 ms) for a new arrival or the
                                // next deadline, the single-box shape (no busy spin, no lost
                                // work: the patience deadline still forces a take).
                                if self.lanes.iter().all(|l| l.is_none()) {
                                    loop {
                                        if self.fatal.is_some() {
                                            break;
                                        }
                                        if until.map_or(true, |d| std::time::Instant::now() >= d) {
                                            break;
                                        }
                                        match rx.try_recv() {
                                            Ok(req) => {
                                                waiting.push_back(req);
                                                break;
                                            }
                                            Err(_) => std::thread::sleep(std::time::Duration::from_millis(20)),
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                crate::metrics::lane_reserve_waiting(waiting.len() as u64);
            } else {
            while free > 0 {
                let Some(mut req) = waiting.pop_front() else { break };
                if req.tx.is_closed() {
                    continue;
                }
                exl3_tp_strip(&mut req, self.tp.as_ref().map_or(1, |t| t.world)); // before the Admit ships: head and node agree
                if let Some(why) = exl3_reject(&req) {
                    let _ = req.tx.send(TokEvent::Finish { reason: why });
                    continue;
                }
                // TPPOOL (design v3 step 4): never ship an Admit the shared pool cannot hold (the apply would be a fatal on
                // every rank). Hold keeps FIFO order. Gated, the head ships one Admit per iteration (`tp_pool_gated`).
                match self.tp_pool_gate(&req, None, &pending_free, gate_reserved) {
                    super::tp_pool_gate::GateDecision::Ship { pages } => gate_reserved += pages,
                    super::tp_pool_gate::GateDecision::Hold => { waiting.push_front(req); break; }
                    super::tp_pool_gate::GateDecision::Refuse(why) => { let _ = req.tx.send(TokEvent::Finish { reason: why }); continue; }
                }
                if req.seed.is_none() {
                    req.seed = Some(entropy_seed()); // head-resolved: every rank samples one stream
                }
                let mut w = WireRequest::from(&req);
                if let Some(e) = req.image_embeds.as_ref().filter(|_| !req.image_spans.is_empty()) {
                    // VIS-4: the rows ride the binary channel right after this step's frame
                    let (bytes, digest) = crate::tp_serve::image_rows_wire(e);
                    w.image_bytes = bytes.len() as u64;
                    w.image_digest = digest;
                    payloads.push(bytes);
                }
                events.push(TpEvent::Admit(w));
                admits.push_back(req);
                free -= 1;
                if self.tp_pool_gated() { break; } // TPPOOL: one Admit per iteration while the gate decides
            }
            }
            if !waiting.is_empty() {
                waiting.retain(|r| !r.tx.is_closed());
            }
            if events.is_empty() && self.lanes.iter().all(|l| l.is_none()) {
                continue;
            }
            step_no += 1;
            // TP-E: a live engine's step goes over RDMA; the TCP Step only when it carries events
            let go = fast_ctl && self.lanes.iter().any(|l| l.is_some());
            if go {
                let n = events.len();
                if let Err(e) = self.tp.as_mut().context("TP state")?.step_go(step_no, n) {
                    return Err(self.tp_fail(format!("step-go failed at step {step_no}: {e:#}"), &mut waiting, Some(&mut rx)));
                }
            }
            if !go || !events.is_empty() {
                let msg = ServingMsg::Step(StepEvents { step: step_no, events: events.clone() });
                if let Err(e) = self.tp_ship(&msg) {
                    return Err(self.tp_fail(format!("control plane send failed at step {step_no}: {e:#}"), &mut waiting, Some(&mut rx)));
                }
                if let Err(e) = self.tp_ship_raw(&payloads) {
                    return Err(self.tp_fail(format!("image payload send failed at step {step_no}: {e:#}"), &mut waiting, Some(&mut rx)));
                }
            }
            for ev in events {
                match ev {
                    TpEvent::Cancel { lane } => {
                        if let Some(mut l) = self.lanes[lane].take() {
                            self.record_release(&mut l); // COPYREC1
                            l.log_cancel(lane);
                        }
                    }
                    TpEvent::RecordClear => self.record_empty(), // COPYREC1
                    TpEvent::Admit(w) => {
                        let req = admits.pop_front().context("admit queue out of step")?;
                        // LANERES3: apply the same scoping we shipped (`WireRequest::reserve`)
                        // or the head's own slot table diverges from the mirrors' from the
                        // very first scoped admit. A scoped admit whose allowed lanes are no
                        // longer free at apply time NEVER refuses (§1b-bis.8): the request
                        // returns to the waiting deque, the next iteration re-plans, and the
                        // mirror skips the event identically (same replicated lane state).
                        match w.reserve {
                            Some(sc) => {
                                if !crate::lane_reserve::allowed_free(&self.lanes, &sc.allowed) {
                                    crate::metrics::lane_reserve_tp_requeued();
                                    crate::dreprintln!("[exl3-serve] TP scoped admit requeued (no allowed lane free; replanning)");
                                    waiting.push_back(req);
                                } else {
                                    self.admit_scoped(req, &sc.allowed, crate::lane_reserve::Kind::from_wire(sc.kind));
                                }
                            }
                            None => self.admit(req),
                        }
                    }
                }
                if let Some(f) = self.fatal.clone() {
                    return Err(self.tp_fail(format!("admit failed at step {step_no}: {f}"), &mut waiting, Some(&mut rx)));
                }
            }
            if self.lanes.iter().any(|l| l.is_some()) {
                if let Err(e) = self.step() {
                    eprintln!("[exl3-serve] TP decode step failed: {e:#}");
                    return Err(self.tp_fail(format!("decode step {step_no} failed: {e:#}"), &mut waiting, Some(&mut rx)));
                }
            }
            self.tp_log_idle(step_no);
        }
    }
}

impl TpSync {
    /// TP-4D (audit C2) world > 2: the outcome of ONE WP16 checkpoint allocation attempt as a lockstep decision.
    /// `CkptStore::acquire` freezes its cap when THIS rank's `cuMemAlloc` fails; ranks whose allocation outcomes
    /// differ hold different checkpoint sets and every later store-dependent decision (resume point, chunk grid)
    /// goes rank-local. Every rank sends (buffers held, ok); Ok(ranks that failed) — if any, ALL ranks must treat
    /// the allocation as failed (and drop a buffer they did get) so the stores stay identical.
    pub(crate) fn alloc_agree_hub<X: crate::net::HubXport>(&mut self, x: &mut X, n_alloc: usize, ok: bool)
                                                           -> Result<Vec<usize>> {
        use crate::tp_lockstep as ls;
        let all = crate::net::hub_all(x, &ls::alloc_frame(n_alloc, ok), ls::ALLOC_WIRE)
            .map_err(|e| self.hub_fail("wp16 alloc", &format!("{e:#}")))?;
        ls::alloc_verdict(&all, self.world as usize).map_err(|why| self.hub_fail("wp16 alloc", &why))
    }

    /// The body of `CkptStore::acquire`'s allocation closure with the cross-rank agreement applied: `r` = this rank's
    /// local `ckpt_alloc` result, `n_before` = the buffers this rank held before it. Every rank runs this at the same
    /// program point (the store state is identical by invariant) and gets the same outcome: Ok = every rank got its
    /// buffer, Err = at least one rank's allocation failed (a buffer this rank did get is dropped) so `acquire` freezes
    /// the cap on ALL ranks. A transport failure of the agree itself lands in `hub_err` (the caller must return it:
    /// `acquire` would otherwise swallow it as an allocation failure).
    pub(crate) fn alloc_outcome<B, X: crate::net::HubXport>(&mut self, x: &mut X, n_before: usize, r: Result<B>,
                                                            hub_err: &mut Option<anyhow::Error>) -> Result<B> {
        match self.alloc_agree_hub(x, n_before, r.is_ok()) {
            Ok(failed) if failed.is_empty() => r,
            Ok(failed) => match r {
                Ok(_dropped) => Err(anyhow::anyhow!("checkpoint allocation failed on rank(s) {failed:?} (agreed across ranks)")),
                Err(e) => Err(e),
            },
            Err(e) => { let m = format!("{e:#}"); *hub_err = Some(e); Err(anyhow::anyhow!(m)) }
        }
    }

    /// World 2 keeps today's rank-local decision (NO exchange, no behaviour change: the gap stays open there, see
    /// PLAN/TP-4D_REPORT.md); world > 2 runs `alloc_outcome` over the hub's control slots.
    pub(super) fn wp16_alloc<B>(&mut self, n_before: usize, r: Result<B>, hub_err: &mut Option<anyhow::Error>) -> Result<B> {
        if self.world <= 2 { return r; }
        match crate::net::link_hub(self.deadline_ms) {
            Ok(mut x) => self.alloc_outcome(&mut x, n_before, r, hub_err),
            Err(e) => { let m = format!("{e:#}"); *hub_err = Some(e); Err(anyhow::anyhow!(m)) }
        }
    }
}

/// TP-4D: the head's rank/host map (`rank r = host (addr)`, r >= 1), set once by main after the node sync so the API
/// startup line can name which box is which rank. Display only — nothing reads it for a decision.
static TP_RANK_MAP: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();

pub fn set_tp_rank_map(map: Vec<String>) { let _ = TP_RANK_MAP.set(map); }

/// TP-D: bring up the EXL3 TP attachment from a TpContext (the sanity + branch handshakes).
fn tp_attach(mut ctx: crate::tp::TpContext) -> Result<crate::exl3_forward::xtp::TpAttach> {
    ctx.sanity()?;
    ctx.branch_check(&crate::tp::TpBranch::Exl3Xtp)?;
    let (rank, world, link) = ctx.into_parts();
    Ok(crate::exl3_forward::xtp::TpAttach { rank, world, link })
}

/// TP-D: --exl3-tp-preverify=0 (diagnostic, prices the belt): skip the pre-verify width/drafts
/// exchange + agree (the post-round agree still proves the width, one round late — TP-C's shape).
/// Rides TpConfig's CLI option-registry snapshot, so both ranks decide alike.
pub(super) fn tp_preverify_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("exl3-tp-preverify")).map_or(true, |v| v != "0"))
}

/// TP-D: --exl3-tp-ident=1 (diagnostic; rides TpConfig's CLI option-registry snapshot, so both ranks agree):
/// the served scheduler digests the logits rows, residual and taps at every lockstep point and
/// compares them across ranks (TP-C's G-T1-b graphed identity, through the API). Costs a dtoh per
/// step — a gate knob, never a serving default.
fn tp_ident_knob(s: &mut Exl3Scheduler) {
    if crate::opts::var(crate::opt!("exl3-tp-ident")).map_or(false, |v| v == "1") {
        if let Some(t) = s.tp.as_mut() {
            t.ident = true;
            crate::dprintln!("[exl3-serve] TP ident digests ON (--exl3-tp-ident=1, diagnostic): logits rows + resid + taps compared across ranks every lockstep point");
        }
    }
}

/// TP-4D: the world > 2 boot-agree frame magic (class prefix; a frame of another class at this point is a desync).
pub(super) const BOOT_AGREE_MAGIC: u32 = 0xB007_A612;

/// TP-4D: Ok when every rank sent the boot magic and rank 0's config hash; Err names every differing rank with both
/// hashes (`all[r] = [magic, hash]`).
pub(super) fn boot_hash_verdict(all: &[Vec<u32>], my_rank: usize) -> std::result::Result<(), String> {
    let mut bad = Vec::new();
    for (r, f) in all.iter().enumerate() {
        if f[0] != BOOT_AGREE_MAGIC {
            bad.push(format!("rank {r} sent frame class {:#010x}, not a boot-agree frame", f[0]));
        } else if f[1] != all[0][1] {
            bad.push(format!("rank {r} hash {:08x} != head {:08x}", f[1], all[0][1]));
        }
    }
    if bad.is_empty() { Ok(()) } else { Err(format!("{} (this is rank {my_rank})", bad.join("; "))) }
}

/// TP-D: every resolved scheduler knob, hashed and AGREED across ranks before the first request
/// (both ranks resolved them from the same argv + shipped env; this proves it, including the
/// memory-derived WP16 checkpoint cap and the tune table actually frozen).
fn tp_boot_agree(s: &mut Exl3Scheduler) -> Result<()> {
    let tune = crate::exl3_tune::status().map(|t| format!("{}|{:?}|{}|{}", t.state, t.table_sha8, t.decisions_s, t.decisions_d))
        .unwrap_or_default();
    let mut w: Vec<u32> = vec![s.width as u32, s.chunk as u32, s.mtp_k as u32, s.prefix_on as u32,
                               s.pen_range as u32, s.spec_ratio as u32, s.model.max_pos() as u32,
                               s.model.mtp.is_some() as u32, crate::exl3_forward::mtp_disabled_by_opt() as u32];
    let dc = s.draft_conf.to_bits();
    w.extend([dc as u32, (dc >> 32) as u32, s.draft_temp.map_or(0, |t| t.to_bits())]);
    w.extend(s.loop_cfg.map_or([0, 0], |(a, b)| [a as u32, b as u32]));
    w.extend(s.wp16.as_ref().map_or([0, 0], |st| [1, st.cap as u32]));
    // PFXPOOL: the pool policy is replicated — its cap, floor and per-entry cost (the geometry-derived bytes of a
    // 1K-token entry on this rank) must match, or the ranks would evict differently. An ON-only word, so a pool-off
    // boot's hash is the pre-PFXPOOL one.
    if s.pool.on() {
        let per_k = s.model.pool_entry_bytes(1024).unwrap_or(0);
        w.extend([0x5046_0000, (s.pool.cap_bytes() >> 20) as u32, (s.pool_floor >> 20) as u32, per_k as u32, (per_k >> 32) as u32]);
    }
    w.push(crate::exl3_forward::tail_ckpt_on() as u32);
    w.push(crate::exl3_forward::kv_fmt_from_opts() as u32);
    // TP-I #7: the WP23 cost-guard prior both ranks resolved (it only matters if the targets differ)
    w.push(boot_prior_word(s.tp_world()));
    // TP-I2 item 1: the prefill-overlap granularity both ranks resolved (--tp-prefill-overlap, TpConfig v19)
    w.push(crate::exl3_forward::xtp::pf_overlap_code());
    // TP-H2: the sampled-row vocab-parallel tail both ranks resolved (--tp-vp-sampled, TpConfig v21)
    w.push(crate::exl3_forward::xtp::vp_sampled_code());
    // TP-SP1: sequence-parallel prefill (--tp-seq-parallel, TpConfig v22) — an ON code only, so an OFF
    // boot's hash is the pre-SP posture's
    if crate::exl3_forward::xtp::seq_parallel_on() {
        w.push(0x5350_0000 | crate::exl3_forward::xtp::seq_parallel_code());
    }
    // CLI-1: every SPMD registry option either rank has set (both installed the head's registry
    // (TpConfig v23); this proves it). Pushed only when one is set, so a boot with no option set
    // keeps the pre-registry hash.
    let spmd = crate::opts::spmd_digest();
    if let Some(d) = spmd { w.extend([0x4F50_5453, d]); }
    // S-B9-CF-PACK5: with --lane-pack on, the resolved mode, whether sampled lanes may pack here and
    // THE PACKED GRAPH SET (count + digest of the (family, regime, width) presence table). `pack_plan`
    // reads `pack_ready`, derived from rank-local graph presence: a rank whose set differed (the
    // CF-P1e TP=2 --max-batch 16 death was a decision reading rank-local graph state) now refuses to
    // boot here instead of planning a pack its peer cannot replay. Pushed only when the arm is on,
    // so a --lane-pack off boot adds nothing to the hash.
    if s.lane_pack.on() {
        if let Some(scp) = s.scp.as_ref() {
            let (n, d, _) = s.model.pack_graph_set(scp, &crate::lane_pack::fams(s.spec_ratio, s.pack_sampled));
            w.extend([0x504B_0001, 1, s.pack_sampled as u32, n as u32, d, s.pack_ready[0] as u32, s.pack_ready[1] as u32]);
        }
    }
    // F1/F10: when the RESOLVED mode is on (the default: no flag needs to be typed), hash the fallback decision
    // (active or not, and why) — a rank that packs while its peer fell back would misstate what is served. Every
    // reason comes from shipped options and model facts and the mode itself from the registry (the head's typed
    // value ships in the snapshot, the row's default is the same constant in one binary), so both ranks hash the
    // same words; a --lane-pack off boot records nothing.
    if let Some(why) = crate::lane_pack::boot_pack_word() {
        let mut d = 0x811c9dc5u32;
        for b in why.bytes() { d = (d ^ b as u32).wrapping_mul(0x01000193); }
        w.extend([0x5F50_4B46, why.is_empty() as u32, d]);
    }
    // FLASHNEXT-DF2-C (ISSUES 116): the DFlash2 chain's boot outcome — pushed only on a rank that
    // ENGAGED it (a boot without DF2 keeps the pre-existing hash), so a head that engaged while a
    // node fell back (no artifact on that node, a different artifact, block or verify width)
    // refuses HERE with a named reason instead of diverging at the first round.
    if s.df2.is_some() {
        let sha = crate::dflash2::load::last_load_sha().unwrap_or("unverified");
        w.extend([0xDF20_0001, crate::dflash2::block() as u32, s.df2_k as u32,
                  fnv32(0xD5A, sha.bytes().map(|b| b as u32))]);
    }
    w.extend(s.eos.iter().copied());
    w.extend(tune.bytes().map(|b| b as u32));
    let h = fnv32(0xB007, w.iter().copied());
    let rank = s.tp.as_ref().map_or(0, |t| t.rank);
    let world = s.tp.as_ref().map_or(2, |t| t.world);
    crate::dprintln!("[exl3-serve] TP={world} rank {rank}: boot config hash {h:08x} (width {} chunk {} k {} dds {} prefix {} wp16 cap {} tail {} tune {tune} \
              dds-guard prior {} ms/draft (--tp-dds-prior; TP=1 6.0) prefill-overlap {} vp-sampled {}{})",
             s.width, s.chunk, s.mtp_k, s.draft_conf, s.prefix_on, s.wp16.as_ref().map_or(0, |st| st.cap),
             crate::exl3_forward::tail_ckpt_on(), dds_prior_w(s.tp_world()),
             crate::exl3_forward::xtp::pf_overlap_desc(crate::exl3_forward::xtp::pf_overlap_code()),
             crate::exl3_forward::xtp::vp_sampled_desc(crate::exl3_forward::xtp::vp_sampled_code()),
             if crate::exl3_forward::xtp::seq_parallel_on() {
                 format!(" seq-parallel {}", crate::exl3_forward::xtp::seq_parallel_desc(crate::exl3_forward::xtp::seq_parallel_code()))
             } else { String::new() });
    crate::dprintln!("[exl3-serve] TP={world} rank {rank}: options {} (spmd digest {})", crate::opts::set_summary(),
             spmd.map_or("none".to_string(), |d| format!("{d:08x}")));
    if world > 2 {
        // TP-4D: world > 2 — every rank's hash through the head hub (probe-tolerant, unbounded: the four model
        // loads finish at different times; a dead peer still aborts within the dead-peer probe): EVERY rank sees
        // EVERY hash and names the offender, instead of one head-side verdict. world 2 keeps the pairwise agree.
        let all = crate::net::exchange_u32s_all(&[BOOT_AGREE_MAGIC, h], 4)
            .context("TP boot agree (head hub; peer dead or link aborted)")?;
        if let Err(why) = boot_hash_verdict(&all, rank as usize) {
            crate::net::abort_link_keep_code();
            anyhow::bail!("TP boot agree FAILED: the ranks resolved different serve configs (hash {h:08x} on rank {rank}): {why}");
        }
        return Ok(());
    }
    // TP-D: the world-2 boot agree is the FIRST cross-rank rendezvous after two INDEPENDENT
    // model loads (50-400 s of legitimate skew) — it must use the boot deadline
    // (`net::agree_boot`), not the 10 s per-step watchdog: the 2026-10-30 #26 repro aborted
    // exactly here while the head was still loading weights.
    if let Err(why) = tp_agree_decide(0, 0, h, crate::net::agree_boot(0xB0_0700, h)) {
        crate::net::abort_link();
        anyhow::bail!("TP boot agree FAILED: the ranks resolved different serve configs (hash {h:08x} on rank {rank}): {why}");
    }
    Ok(())
}

/// TP-D: the TP=2 HEAD serve (rank 0): load this rank's shard, prove the config equal with the
/// node, wait for the node's mirror to arm (Ready), then serve HTTP with the scheduler running
/// `run_tp_head` over the retained control streams.
/// PACK-FIX: `watch` (armed by the caller before the link bring-up) keeps watching the node control
/// streams through this rank's load + boot agree — a node boot failure it reports (or its death)
/// exits the head at once with the node's reason; it is disarmed right before the Ready reads.
pub fn run_tp_head_serve(args: &[String], model_dir: &str, ctx: crate::tp::TpContext,
                         mut streams: Vec<std::net::TcpStream>, watch: crate::cluster::NodeWatch) -> Result<()> {
    // S7 (REL_V0_7_3): refuse a bad --otel-endpoint BEFORE the RDMA attach (inside
    // build_serve it runs after tp_attach). Returning Err (not exit) lets the caller report it.
    if let Some(cfg) = crate::otel::config_from_args(|f| arg(args, &format!("--{f}")).map(|s| s.to_string())) {
        if let Err(e) = cfg.hostport() { anyhow::bail!("[otel] {e}"); }
    }
    tp_lane_width_refusal(args, ctx.world as u32)?;   // W4: before the attach (the head's main refused it before the sync)
    // S-B9-CF-PACK5: --lane-pack runs at TP=1, TP=2 and TP=4; a TYPED `--lane-pack on` at a larger world is refused
    // here BEFORE the attach (PACK4-REVIEW B1: and before this, in main.rs, by xtp::world_gt2_flag_refusals,
    // ahead of the discovery). F10: `parse(None)` is the raw-argv meaning (off), so a defaulted-on boot is never
    // refused here; build_serve turns it into a fallback line.
    crate::lane_pack::tp_refusal(crate::lane_pack::LanePack::parse(arg(args, "--lane-pack"))?, ctx.world as u32)?;
    let attach = tp_attach(ctx)?;
    let mut parts = build_serve(args, model_dir, Some(attach))?;
    tp_ident_knob(&mut parts.sched);
    tp_boot_agree(&mut parts.sched)?;
    copy_record_tp_agree(&mut parts.sched, 0)?; // COPYREC1: no-op unless --copy-span-record is armed (on every rank alike)
    watch.disarm(&streams);
    for (i, s) in streams.iter_mut().enumerate() {
        match crate::tp_serve::recv_serving(s).context("node Ready")? {
            crate::tp_serve::ServingMsg::Ready => crate::dprintln!("[exl3-serve] TP node rank {} READY (mirror armed)", i + 1),
            other => anyhow::bail!("expected Ready from node rank {}, got {other:?}", i + 1),
        }
    }
    let world = parts.sched.tp.as_ref().map_or(2, |t| t.world);
    if world <= 2 {
        println!("[exl3-serve] TP={world} serving: every §1b feature as TP=1 (graphs, MTP k={}, DDS {}, WP27 forks, prefix cache {}, \
                  WP16 {}) — head-authoritative Step/Cancel/Admit(seed) + HeadFlag, agree per seam/round + pre-verify",
                 parts.sched.mtp_k, parts.sched.draft_conf, parts.sched.prefix_on,
                 parts.sched.wp16.as_ref().map_or("off".to_string(), |st| format!("cap {}", st.cap)));
    }
    if world > 2 {
        // TP-4D (review F1): at world > 2 some §1b features are OFF, INERT or refused, so the world-2 "every §1b feature as
        // TP=1" claim is not printed. The `TP=N serving: ` prefix is the launcher's exactly-once marker (tp4_launch.sh
        // SERVE_SERVING_RE); the per-feature state rides the next line. Option-resolved state, read from the same predicates
        // the load uses; the `TP-H ...` / `[exl3-tp] ...` load lines above show what attached.
        {
            use crate::exl3_forward::xtp;
            let s = &parts.sched;
            let mtp_on = s.model.mtp.is_some() && !crate::exl3_forward::mtp_disabled_by_opt();
            let graphs = if crate::opts::var(crate::opt!("exl3-no-graph")).is_ok()
                || crate::opts::var(crate::opt!("mtp-dump")).map_or(false, |v| !v.is_empty()) { "OFF (--exl3-no-graph / --mtp-dump)" }
                else if arg(args, "--graph-precapture") == Some("off") { "ON (lazy capture: --graph-precapture off)" }
                else { "ON (the greedy, unpenalized shapes are precaptured at boot; sampled and penalized shapes are captured during traffic)" };
            let mtp = if mtp_on { format!("ON k={}", s.mtp_k) }
                else if s.model.mtp.is_some() {
                    format!("OFF ({})", crate::exl3_forward::mtp_off_by_opt().map_or("--exl3-no-mtp", |o| o.flag()))
                } else { "OFF (pack has no MTP head)".to_string() };
            let dds = if mtp_on && s.draft_conf > 0.0 { format!("ON at {}", s.draft_conf) } else { "OFF".to_string() };
            let dec = xtp::dec_xport_mode() == 1;
            println!("[exl3-serve] TP={world} serving: head-authoritative Step/Cancel/Admit(seed) + HeadFlag, hub lockstep agree per \
                      seam/round + pre-verify — some TP=1 §1b features are OFF / INERT / refused at world {world}; per-feature state on the \
                      next line, table in PLAN/TP-4D_REPORT.md");
            // seltc6: println, not dprintln: the line above promises it at the default log level
            println!("[exl3-serve] TP={world} features: graphs {graphs}; MTP {mtp}; DDS {dds}; prefix cache {}; WP16 {}; WP27 forks not \
                      world-gated (--wp27-off / --wp27-forks / PDL decide); TP-H greedy vocab-parallel head {}; sampled / penalized / \
                      ratio-rule rows = {}; sharded MTP-head screen {}; decode \
                      transport {}; prefill transport {}; prefill overlap {}; sequence-parallel prefill {}; EP deal '{}'; pre-verify \
                      drain SKIPPED at world > 2 (it runs at world 2 only; hardware-unproven); refused if passed: --tp-oneshot, --tp-dec-grecv != 0, \
                      --ep-deal freq outside world 4",
                     if s.prefix_on { "ON" } else { "OFF (--prefix-cache off)" },
                     s.wp16.as_ref().map_or("OFF".to_string(), |st| format!("ON (cap {} checkpoints)", st.cap)),
                     if xtp::vp_head_on() && dec { "ON" } else { "OFF (--tp-vp-head off / --tp-dec-xport 0)" },
                     if xtp::vp_head_on() && dec && xtp::vp_sampled_on() { "ON (vocab-parallel all-gather, TP-4H2)" }
                         else { "REPLICATED head (--tp-vp-sampled off / --tp-vp-head off / --tp-dec-xport 0)" },
                     if mtp_on && xtp::dh_shard_on() && dec { "ON (shown for the greedy, unpenalized requests)" } else { "OFF (no MTP head / --tp-dh-shard off / --tp-dec-xport 0)" },
                     if dec { format!("folded K1 + multi-block K2, {} exchange(s) per reduce{}, receive cpu_done only",
                                      if xtp::tp_reduce_single_for(world) { 1 } else { (world as u32).trailing_zeros() },
                                      if xtp::tp_reduce_single_for(world) { " (--tp-reduce single: ONE all-peers stage, the world-4 default)" } else { "" }) }
                         else { "serial K1/K2 + cvt (--tp-dec-xport 0)".to_string() },
                     match xtp::prefill_xport_mode() { 0 => "serial rounds-aware K1/K2", 1 => "single-rail pipelined (hardware-unproven at world > 2)", _ => "dual-rail pipelined (hardware-unproven at world > 2)" },
                     if xtp::pf_overlap_rows() == 0 { "OFF" } else { "ON (the hooks ride the TP-4E round-aware program: a landed row range is consumed only after its last round's K2m; fold is refused at world > 2)" },
                     if !xtp::seq_parallel_on() { "OFF" }
                         else if world == 4 { "ON (TP-4S quarter rows: recursive-halving reduce-scatter + recursive-doubling all-gather; hardware-unproven until PLAN/TP-4S_GATES.md rungs 1-2 pass)" }
                         else { "INERT" },
                     xtp::ep_deal_kind());
        }
        // TP-4D: the API startup line names the world, the lockstep protocol and the rank/host map
        crate::dprintln!("[exl3-serve] TP={world} API up: head = rank 0; {}; lockstep = head-hub (one merged op per point, per-wait \
                  deadline {} ms — --tp-lockstep-timeout-ms); Step/Cancel/Admit/HeadFlag over {} node control streams",
                 TP_RANK_MAP.get().map_or("rank/host map unavailable".to_string(), |m| m.join("; ")),
                 parts.sched.tp.as_ref().map_or(0, |t| t.deadline_ms), world - 1);
    }
    serve_http(parts, Some(streams))
}

/// TP-D: the TP=2 NODE serve (rank >= 1): the same boot as the head (same argv), then the mirror
/// loop on this thread, pinned to core 9.
/// PACK-FIX: a boot failure (attach, load, boot agree, pinning) is reported to the head over the
/// control stream before returning — the head watches the stream through its own load and exits
/// loudly with this reason, instead of loading for minutes and then losing the node in a probe.
pub fn run_tp_node_serve(args: &[String], model_dir: &str, ctx: crate::tp::TpContext,
                         mut stream: std::net::TcpStream) -> Result<()> {
    let node_world = ctx.world;   // TP-4Z1: the worker mask needs the world to know whether the single-stage proxy (core 17) is live
    // S7 (REL_V0_7_3): same hoisted refusal as on the head - BEFORE the attach, and reported
    // through the boot failure path instead of process::exit(1).
    if let Some(cfg) = crate::otel::config_from_args(|f| arg(args, &format!("--{f}")).map(|s| s.to_string())) {
        if let Err(e) = cfg.hostport() { anyhow::bail!("[otel] {e}"); }
    }
    tp_lane_width_refusal(args, node_world as u32)?;   // W4: before the attach (the prelink refused it before the link)
    // S-B9-CF-PACK5: --lane-pack runs at TP=1, TP=2 and TP=4; a TYPED `--lane-pack on` at a larger world is refused
    // here BEFORE the attach (and before the link, in the prelink, by xtp::world_gt2_flag_refusals: PACK4-REVIEW
    // B1); a defaulted-on boot is handled by build_serve's fallback (F10).
    crate::lane_pack::tp_refusal(crate::lane_pack::LanePack::parse(arg(args, "--lane-pack"))?, node_world as u32)?;
    let boot = (|| -> Result<(Exl3Scheduler, Option<Vec<usize>>)> {
        let attach = tp_attach(ctx)?;
        let parts = build_serve(args, model_dir, Some(attach))?;
        let mut sched = parts.sched;
        tp_ident_knob(&mut sched);
        tp_boot_agree(&mut sched)?;
        let rank = sched.tp.as_ref().map_or(1, |t| t.rank);
        copy_record_tp_agree(&mut sched, rank)?; // COPYREC1: the head runs the same exchange at the same boot point
        anyhow::ensure!(crate::net::pin_thread(9), "TP node mirror thread failed to pin to core 9 — TP refuses to run unpinned");
        Ok((sched, parts.cpu_aff.0))
    })();
    let (mut sched, cpu_mask) = match boot {
        Ok(x) => x,
        Err(e) => {
            crate::cluster::node_report_failure(&mut stream, &format!("TP node boot failed: {e:#}"));
            return Err(e);
        }
    };
    set_worker_mask(tp_worker_mask(&cpu_mask, true, node_world));
    crate::tp_serve::send_serving(&mut stream, &crate::tp_serve::ServingMsg::Ready)?;
    crate::dprintln!("[exl3-serve] TP node: mirror armed (Ready sent) — waiting for the head's steps");
    sched.tp.as_mut().context("node without TP state")?.ctl = vec![stream];
    sched.run_tp_mirror()
}

/// WP24: `--spec-sampling match|ratio` and `--draft-temperature <T>` (EXL3 serve, TP=1 — no
/// TpConfig transport needed while EXL3 has no TP path). match (default) = today: drafts are the
/// head's argmax, a sampled verify row accepts while its sample == the draft. ratio = real-q
/// speculative sampling for SAMPLED requests: each draft pass samples its draft from q' (the
/// head's top-32 under --draft-temperature and the request's top-k / top-p), a verify row accepts
/// it with probability min(1, p/q'), an exact residual on reject — distribution-exact vs plain
/// sampling (gate: --probe-spec-sampling), but seeded sampled bytes change. Greedy requests are
/// unaffected. --draft-temperature default = the request's temperature.
pub fn parse_spec_sampling(args: &[String]) -> Result<(bool, Option<f32>)> {
    let spec_ratio = match arg(args, "--spec-sampling") {
        None | Some("ratio") => true, // owner ruling 2026-09-27: ratio is the default (χ² gate PASS, +5.1% T=1 thinking)
        Some("match") => false,
        Some(v) => anyhow::bail!("--spec-sampling must be match or ratio (got '{v}')"),
    };
    let draft_temp: Option<f32> = match arg(args, "--draft-temperature") {
        None => None,
        Some(v) => {
            let t: f32 = v.parse().map_err(|_| anyhow::anyhow!("--draft-temperature must be a number"))?;
            anyhow::ensure!(t.is_finite() && t > 0.0, "--draft-temperature must be > 0 (omit it = the request's temperature)");
            Some(t)
        }
    };
    // (SURFACE2 item 4: nothing is drafted with speculation switched off, so the mode is not announced)
    if (spec_ratio || draft_temp.is_some()) && !crate::exl3_forward::mtp_disabled_by_opt() {
        crate::dprintln!("[exl3-serve] spec sampling: {} (draft temperature {}){}",
                 if spec_ratio { "ratio (real-q: sampled drafts on the full draft-slice head — on DHEAD when the DHEAD-RQ line says so, accept min(1, p/q'))" } else { "match" },
                 draft_temp.map_or("= request temperature".to_string(), |t| t.to_string()),
                 if !spec_ratio && draft_temp.is_some() { " — --draft-temperature only applies with --spec-sampling ratio" } else { "" });
    }
    Ok((spec_ratio, draft_temp))
}

/// Is the `--vision-fail-gemm` drill armed? Read through the option registry, as the NVFP4 server
/// does (`src/main.rs`). S-B9-REL-VISIONFIX H2: this path read it with `arg(args, ..).is_some()`,
/// the VALUE reader, which answers "is there a token after the flag": a bare flag at the END of
/// the command line was silently not armed (and `--print-config` still showed it as set), while a
/// flag followed by anything at all, `=0` included, was. The registry parses a `Ty::Flag` by
/// presence, wherever it stands.
pub(super) fn vision_fail_gemm_armed() -> bool {
    crate::opts::on(crate::opt!("vision-fail-gemm"))
}

/// S-B9-REL-PRPORT3 (review R9): the PR #17 image-payload pre-read, extracted so the
/// admit-order alignment and the error paths are unit-testable. Reads every image-bearing
/// Admit's payload (in admit order) from the control stream BEFORE any admit() runs —
/// prefill multiplexes HeadFlag frames on the same stream, so a payload still unread when
/// the first admit runs could interleave with them (the stream desync / digest mismatch
/// master could hit with 2+ images admitted in one Step). Digest verified per payload;
/// little-endian f16 rows decoded to f32.
pub(crate) fn read_step_image_payloads<R: std::io::Read>(
    r: &mut R,
    step_no: u64,
    specs: &[(u64, u64)],
) -> anyhow::Result<Vec<Option<Vec<f32>>>> {
    use std::io::Read;
    let mut payloads: Vec<Option<Vec<f32>>> = Vec::new();
    for &(ib, idg) in specs {
        if ib > 0 {
            let mut buf = vec![0u8; ib as usize];
            r.read_exact(&mut buf).context("TP node: image payload read")?;
            anyhow::ensure!(crate::tp_serve::fnv64(&buf) == idg,
                            "TP node: image payload digest mismatch (step {step_no})");
            payloads.push(Some(buf.chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect()));
        } else {
            payloads.push(None);
        }
    }
    Ok(payloads)
}
