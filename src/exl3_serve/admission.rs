//! Request admission for the EXL3 scheduler: `admit` / `admit_scoped` / `admit_req` (prefix cache, prefill hand-off, lane install),
//! lane release and the per-lane bookkeeping around it, and `vision_setup`. Another `impl Exl3Scheduler` block (file map in `mod.rs`).
//! Bodies keep their exact text.

use super::*;

impl Exl3Scheduler {
    /// VIS-2 (TP=1): arm `slot` for an image request — the 3-axis RoPE map + image-token rows
    /// (FwdModel::set_slot_vision) and the prefill splice data (psc.vis_img / vis_src). Returns the
    /// prefix-cache key (image rows replaced by pixel-hash ids, so a cached prefix is reused only
    /// for the SAME image) and a generated-token cap (none since VIS-3).
    fn vision_setup(&mut self, slot: usize, req: &BatchRequest) -> Result<(Vec<u32>, usize)> {
        let sec = self.mrope_section.context("bad request: this server has no vision tower for this model (text-only)")?;
        let embeds = req.image_embeds.as_ref().context("image spans without image embeddings")?;
        let h = self.model.cfg.hidden_size;
        let plen = req.prompt.len();
        let n_img: usize = req.image_spans.iter().map(|s| s.num_tokens).sum();
        anyhow::ensure!(embeds.len() == n_img * h, "image embeddings {} != {n_img} tokens x {h}", embeds.len());
        anyhow::ensure!(n_img <= crate::exl3_forward::VIS_MROPE_ROWS,
                        "bad request: {n_img} image tokens exceed the per-request limit of {}", crate::exl3_forward::VIS_MROPE_ROWS);
        let (pos3, delta) = crate::vision_encoder::mrope_positions(plen, &req.image_spans)?;
        self.model.set_slot_vision(slot, &pos3, delta, sec)?;
        let mut src = vec![-1i32; plen];
        let mut key = req.prompt.clone();
        let mut off = 0usize;
        for sp in &req.image_spans {
            let hh = (sp.pixel_hash as u32) ^ ((sp.pixel_hash >> 32) as u32);
            for j in 0..sp.num_tokens {
                src[sp.start + j] = (off + j) as i32;
                // ids >= 2^31 never collide with a vocabulary id; only the cache compares them
                key[sp.start + j] = 0x8000_0000 | (hh.wrapping_add((j as u32).wrapping_mul(0x9E37_79B1)) & 0x7FFF_FFFF);
            }
            off += sp.num_tokens;
        }
        // the embedding table is fp16: image rows enter the residual in the same precision class
        // (VIS-4: also the TP wire's precision, so every rank splices bit-identical rows)
        let rows: Vec<f32> = embeds.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
        let (_, digest) = crate::tp_serve::image_rows_wire(&rows);
        self.vis_seam = Some((n_img as u32, delta as i32, digest));
        if self.psc.is_none() {
            self.psc = Some(self.model.prefill_scratch(psc_alloc_rows(self.chunk.max(1), self.tp.is_some()))?);
        }
        let img = self.model.vis_rows_upload(&rows)?;
        let psc = self.psc.as_mut().unwrap();
        psc.vis_img = Some(img);
        psc.vis_src = src;
        crate::drprintln!("[exl3-serve] VIS-2 slot={slot}: {} image(s), {n_img} image tokens in a {plen}-token prompt; rope delta {delta}",
                 req.image_spans.len());
        // VIS-3: the QSA indexer ropes its pooled keys through the same per-slot map, so an image
        // request runs past the dense window like text (no generated-token cap)
        Ok((key, usize::MAX))
    }

    /// LANERES1: every lane release of the single-box paths goes through one helper so no site
    /// is missed. `normal` = the request completed its turn (stop / length / a stop string / a
    /// tool call / loop-detected); only a normal release is eligible for a reservation. Cancels,
    /// errors and failed prefills never reserve. No stamping while the policy is off.
    pub(super) fn release_lane(&mut self, s: usize, normal: bool) {
        if let Some(mut lane) = self.lanes[s].take() { self.record_release(&mut lane); }
        if self.reserve.is_some() {
            self.rel[s] = Some((std::time::Instant::now(), normal));
        }
    }

    /// `release_lane` for the sites that still need the lane (`take()`): stamp, then hand over.
    pub(super) fn take_lane(&mut self, s: usize, normal: bool) -> Option<Lane> {
        let mut lane = self.lanes[s].take();
        if let Some(l) = lane.as_mut() { self.record_release(l); }
        if lane.is_some() && self.reserve.is_some() {
            self.rel[s] = Some((std::time::Instant::now(), normal));
        }
        lane
    }

    /// COPYREC1: a lane leaves (any reason: stop, length, loop, cancel, disconnect, a failed step): its record-copy
    /// outcomes reach the landing rates and its reply (what was emitted, after its prompt's tail) goes into the
    /// record. Every TP rank releases the same lane at the same program point (finishes are lockstep-agreed, cancels
    /// are shipped events), so the ranks' records stay identical.
    pub(super) fn record_release(&mut self, lane: &mut Lane) {
        let Some(r) = self.record.as_mut() else { return };
        let th = std::time::Instant::now();
        if let Some(cs) = lane.copy.as_mut() { cs.flush_record(r); }
        if let Some(reply) = lane.rec_reply.take() {
            if let Some(f) = r.add_reply(&lane.rec_tail, &reply) { crate::copy_record::count_fallback(f); }
        }
        crate::copy_record::publish(r);
        crate::copy_record::add_host_us(th.elapsed().as_micros() as u64);
    }

    /// COPYREC1: `DELETE /v1/copy-record` reached this rank (TP=1: at the top of a scheduler iteration; TP: the head
    /// ships it as a step event, applied by every rank before that step's admits).
    pub(super) fn record_empty(&mut self) {
        let Some(r) = self.record.as_mut() else { return };
        match r.empty() {
            Ok(()) => crate::rprintln!("[exl3-serve] copy-span-record: emptied (DELETE /v1/copy-record)"),
            Err(why) => {
                crate::copy_record::count_fallback(crate::copy_record::Fallback::Delete);
                crate::reprintln!("[exl3-serve] copy-span-record: emptied in memory, but {why}; the record stops persisting for this run");
            }
        }
        crate::copy_record::publish(r);
    }

    pub(super) fn admit(&mut self, req: BatchRequest) {
        self.admit_req(req, None);
        // issue #24: the prefill (if any) is over — the lane now carries its own position, or the
        // admit failed / was cancelled and the slot is free again.
        self.kv_pf = None;
        self.publish_kv();
    }

    fn admit_req(&mut self, mut req: BatchRequest, scoped: Option<(&[usize], crate::lane_reserve::Kind)>) {
        let t_admit = std::time::Instant::now();
        // prefix cache: prefer the free lane whose snapshot is the longest prefix of this prompt,
        // then a lane with no snapshot, then any free lane.
        // LANERES1: with the reservation policy on (`scoped`), every candidate list is restricted
        // to the policy's `allowed` lanes, and the fallback after the snapshot hit is the
        // policy's victim order (empty > deepest WP16 > least recently released) instead of
        // "slot 0". `scoped == None` keeps today's pick byte-for-byte.
        let pick = {
            // PACKPF: a lane holding an in-flight packed prefill is not free for this admit
            let free: Vec<usize> = (0..self.lanes.len()).filter(|&i| self.lanes[i].is_none() && !self.pf_busy(i)).collect();
            let free: Vec<usize> = match scoped {
                Some((allowed, _)) => free.into_iter().filter(|i| allowed.contains(i)).collect(),
                None => free,
            };
            // PACKPF: the lane a packed prefill already filled with this prompt comes first
            let packed = if scoped.is_none() {
                free.iter().copied().find(|i| self.pf_done.get(i).map_or(false, |d| d.key == req.prompt))
            } else { None };
            let hit = free.iter().copied().filter_map(|i| match &self.cache[i] {
                Some(t) if t.len() < req.prompt.len() && req.prompt[..t.len()] == t[..] => Some((t.len(), i)),
                _ => None,
            }).max();
            packed.or(hit.map(|(_, i)| i))
                .or_else(|| {
                    // LANERES1: a scoped admit that did not come through the snapshot-hit arm
                    // picks its owner-change victim among the allowed lanes only.
                    if scoped.is_some() {
                        return crate::lane_reserve::victim(&free, &self.slot_views(), &self.wp16_depths(&req.prompt));
                    }
                    // WP16: else the free lane whose checkpoints resume this prompt deepest
                    let st = self.wp16.as_ref()?;
                    let end = req.prompt.len().saturating_sub(1);
                    free.iter().copied()
                        .map(|i| (st.resume_point(i, &req.prompt, end, self.chunk.max(1)).0, i))
                        .filter(|&(q, _)| q > 0).max().map(|(_, i)| i)
                })
                .or_else(|| free.iter().copied().find(|&i| self.cache[i].is_none()))
                .or_else(|| free.first().copied())
        };
        let free = match pick {
            Some(i) => i,
            None => {
                let _ = req.tx.send(TokEvent::Finish {
                    reason: "error: server busy (all lanes occupied)".into(),
                });
                return;
            }
        };
        if let Some((_, kind)) = scoped {
            crate::metrics::lane_reserve_admit(match kind {
                crate::lane_reserve::Kind::Own => 0,
                crate::lane_reserve::Kind::Unreserved => 1,
                crate::lane_reserve::Kind::Forced => 2,
            });
            match kind {
                crate::lane_reserve::Kind::Own => {
                    let snap = self.cache[free].as_ref().map_or(0, |t| t.len());
                    crate::drprintln!("[exl3-serve] lane-reserve: own slot={free} ({snap} of {} prompt tokens)", req.prompt.len());
                }
                crate::lane_reserve::Kind::Unreserved => {
                    crate::drprintln!("[exl3-serve] lane-reserve: unreserved slot={free}");
                }
                crate::lane_reserve::Kind::Forced => {
                    let waited = t_admit.duration_since(req.received_at).as_secs_f32();
                    let victim_age = self.rel[free].map_or("never ran".to_string(), |(t, _)| {
                        format!("{:.0} s ago", t_admit.duration_since(t).as_secs_f32())
                    });
                    crate::drprintln!("[exl3-serve] lane-reserve: forced slot={free} after {waited:.0} s (victim released {victim_age})");
                }
            }
        }
        // W2-EXL3 / PR #30: the masked plain step and the logits readback are wired at TP=1 only. The server
        // already serves such requests without them under TP (AppState schema_ok / logprobs_ok); the head strips
        // them before an Admit ships; this is the structural backstop (never a refusal).
        if let Some(w) = self.tp.as_ref().map(|t| t.world) { exl3_tp_strip(&mut req, w); }
        if req.prompt.is_empty() {
            let _ = req.tx.send(TokEvent::Finish { reason: "error: empty prompt".into() });
            return;
        }
        let mut max_new = req.max_new.min(self.model.max_pos().saturating_sub(req.prompt.len() + 1));
        // VIS-2: an image request arms the slot's 3-axis RoPE map and the prefill splice; every other
        // request puts the slot back on plain text RoPE (a previous image request may have armed it).
        self.vis_seam = None;
        let key: Vec<u32> = if req.image_spans.is_empty() {
            if let Err(e) = self.model.clear_slot_vision(free) {
                let _ = req.tx.send(TokEvent::Finish { reason: format!("error: slot reset failed: {e}") });
                self.fatal = Some(format!("{e:#}"));
                return;
            }
            req.prompt.clone()
        } else {
            match self.vision_setup(free, &req) {
                Ok((key, cap)) => {
                    max_new = max_new.min(cap);
                    key
                }
                Err(e) => {
                    let _ = self.model.clear_slot_vision(free);
                    if let Some(p) = self.psc.as_mut() { p.vis_img = None; p.vis_src.clear(); }
                    let msg = format!("{e:#}");
                    let reason = if msg.starts_with("bad request:") || msg.starts_with("unprocessable:") {
                        format!("error: {msg}")
                    } else {
                        format!("error: vision setup failed: {msg}") // an engine fault, not the client's
                    };
                    let _ = req.tx.send(TokEvent::Finish { reason });
                    return;
                }
            }
        };
        let pre = self.prefill_cached(free, &req.prompt, &key, req.ckpt_at, &req.tx);
        if let Some(p) = self.psc.as_mut() { p.vis_img = None; p.vis_src.clear(); } // VIS-2: splice data is per prefill
        match pre {
            Ok(true) => {}
            // WP02: the client left mid-prefill — the slot is free again, nobody to tell
            Ok(false) => return,
            Err(e) => {
                crate::reprintln!("[exl3-serve] prefill failed (slot {free}): {e:#}");
                let _ = req.tx.send(TokEvent::Finish { reason: format!("error: prefill failed: {e}") });
                if self.tp.is_some() || sticky_cuda_error(&e) { self.fatal = Some(format!("{e:#}")); }
                return;
            }
        }
        // S-A3-h seam step: the LAST prompt token runs through forward_step — the
        // graphed/fused decode step the bench path uses for it — so its KV/GDN state
        // and the tap left in sc.resid are byte-class identical to bench ground
        // truth. Its argmax is the first generated token (the S-A3-e contract).
        let plen = req.prompt.len();
        // API-parity G2: the first generated token is SAMPLED for sampled requests (it was
        // the argmax). Seed: the request's, else fresh entropy; counter 0 = this token.
        let seed = req.seed.unwrap_or_else(|| {
            let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64).unwrap_or(0);
            t ^ ((free as u64) << 56) ^ 0x5DEECE66D
        });
        let first = crate::exl3_forward::SampParams::sampled(
            req.temperature, req.top_p, req.top_k, req.min_p, seed, 0);
        if let Err(e) = self.model.set_sampling(&mut self.sc, &[first]) {
            let _ = req.tx.send(TokEvent::Finish { reason: format!("error: sampler staging failed: {e}") });
            if self.tp.is_some() { self.fatal = Some(format!("{e:#}")); } // TP-D: never a one-rank skip
            return;
        }
        // WP15: penalties from the first generated token on (the seam step's row), over the
        // prompt tail + everything generated — the reference implementation's past_ids = the full sequence.
        let pen = crate::exl3_forward::PenParams::new(req.rep_penalty, req.presence_penalty,
                                                      req.frequency_penalty, self.pen_range, self.pen_range);
        if pen.is_some() {
            crate::dreprintln!("[exl3-serve] penalties slot={free}: {pen:?} (window {} + {} tokens)", self.pen_range, self.pen_range);
        }
        let staged = match pen {
            Some(_) => self.model.pen_seed(&mut self.sc, free, &req.prompt[..plen - 1])
                .and_then(|_| self.model.set_penalties(&mut self.sc, &[pen])),
            None => self.model.set_penalties(&mut self.sc, &[None]),
        };
        if let Err(e) = staged {
            let _ = req.tx.send(TokEvent::Finish { reason: format!("error: penalty staging failed: {e}") });
            if self.tp.is_some() { self.fatal = Some(format!("{e:#}")); } // TP-D: never a one-rank skip
            return;
        }
        // THINKBUDGET1: the request's thinking budget, from the prompt's reasoning state (vLLM's rule).
        // A model without single-token <think> / </think> markers cannot apply it: served without it,
        // said once, counted.
        let mut think = req.thinking_budget.and_then(|b| {
            match crate::think_budget::Markers::from_encoder(|txt| self.tok.encode(txt, false).ok(), crate::think_budget::reasoning_cfg()) {
                Some(m) => {
                    crate::metrics::think_budget_applied();
                    let tb = crate::think_budget::ThinkBudget::new(b, m, &req.prompt);
                    crate::dreprintln!("[exl3-serve] thinking budget slot={free}: {b} tokens (the prompt {} a reasoning block{})",
                        if tb.in_think() { "opens" } else { "does not open" },
                        if tb.in_think() { format!("; {} of the budget used by the prompt", b.saturating_sub(tb.room().unwrap_or(0))) } else { String::new() });
                    Some(tb)
                }
                None => {
                    crate::metrics::think_budget_unsupported();
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| crate::reprintln!("[exl3-serve] thinking_token_budget: this model's tokenizer has no single-token \
<think> / </think> markers, so requests run without a thinking budget (counted in velogb10_thinking_budget_requests_total; \
this line prints once per boot)"));
                    None
                }
            }
        });
        // Logprobs at TP: the seam step's row is this request's first logprob (every rank: the request is mirrored)
        if self.tp.is_some() && req.logprobs.is_some() { self.sc.lp_live = true; }
        // W2-EXL3: the seam step emits the request's FIRST token — mask it when the constraint
        // is live from the start (no think block to wait for).
        let mut schema_state = req.schema.as_ref().map_or(0, |sm| sm.start_state());
        let mut schema_armed = req.schema.as_ref().map_or(false, |sm| sm.defer_until().is_none());
        let vstaged = match req.schema.as_ref().filter(|_| schema_armed) {
            Some(sm) => {
                let bits = sm.allowed(schema_state);
                self.model.set_vmask(&mut self.sc, &[Some(&bits[..])])
            }
            None => self.model.set_vmask(&mut self.sc, &[None]),
        };
        if let Err(e) = vstaged {
            let _ = req.tx.send(TokEvent::Finish { reason: format!("error: schema mask staging failed: {e}") });
            return;
        }
        let last = match self.model.forward_step(
            &mut self.sc,
            &[*req.prompt.last().unwrap() as i32],
            &[plen - 1],
            &[free],
            None,
        ) {
            Ok(ids) => think_emit(think.as_mut(), free, ids[0] as u32, 0, 1),
            Err(e) => {
                eprintln!("[exl3-serve] seam decode step failed (slot {free}): {e:#}");
                let _ = req.tx.send(TokEvent::Finish {
                    reason: format!("error: seam step failed: {e}"),
                });
                if self.tp.is_some() || sticky_cuda_error(&e) { self.fatal = Some(format!("{e:#}")); }
                return;
            }
        };
        // issue #24: the seam step wrote row plen - 1 — the request holds its whole prompt now (a
        // request that ends at its first token never becomes a lane; its context still counts in the peak)
        self.kv_pf = Some((free, plen));
        self.publish_kv();
        if self.tp.is_some() {
            // TP-C: lockstep on the seam step (the request's first token)
            // TP-D: + the slot, the prefix-cache / WP16-C1 resume points and the checkpoint store's
            // shape (live, allocated, this slot's checkpoint positions) — proven equal on every rank
            let (reuse, q) = self.last_resume;
            let mut w = vec![last, plen as u32, free as u32, reuse as u32, q as u32];
            if let Some((n, d, h)) = self.vis_seam {
                // VIS-4: the image rows, their count and the rope delta are identical on every rank
                w.extend([0x7155_0000, n, d as u32, h as u32, (h >> 32) as u32]);
            }
            if let Some(st) = self.wp16.as_ref() {
                w.extend([st.live() as u32, st.n_alloc as u32,
                          fnv32(0, st.positions(free).iter().map(|&x| x as u32))]);
            }
            if let Err(e) = self.tp_round("seam", 0, 1, &w, 0.0, 0.0, false, false) {
                crate::reprintln!("[exl3-serve] TP seam lockstep failed (slot {free}): {e:#}");
                let _ = req.tx.send(TokEvent::Finish { reason: format!("error: TP lockstep failed: {e}") });
                self.fatal = Some(format!("{e:#}"));
                return;
            }
        }
        if let Some(n) = req.logprobs.filter(|_| self.lp_reader()) {
            self.send_logprob(&req.tx, false, 0, last, n);
        }
        if req.tx.send(TokEvent::Tok(last)).is_err() && self.tp.is_none() {
            // WP02: the client is gone — no lane (the prefix snapshot stays valid). TP-D: under TP
            // the lane is created on every rank and the head's next sweep cancels it.
            crate::dreprintln!("[mtp-stats] slot={free} finish=cancelled gen=1 rounds=0 plain_steps=0 temp={}", req.temperature);
            return;
        }
        // WP04-v2: engine-side TTFT (receipt = the handler's hand-off, AFTER tokenization and
        // template rendering), so a client TTFT can be split into HTTP/tokenizer vs engine time.
        if sched_trace_on() {
            crate::rprintln!("[sched-trace] t={:.3} first slot={free} plen={plen} after_receipt_ms={:.1} queued_ms={:.1} reused={}",
                             wall_now(), req.received_at.elapsed().as_secs_f64() * 1e3,
                             t_admit.duration_since(req.received_at).as_secs_f64() * 1e3, self.last_resume.0);
        }
        crate::drprintln!("[exl3-serve] first token slot={free} {:.1} ms after receipt (queued {:.1} ms)",
                 req.received_at.elapsed().as_secs_f64() * 1e3,
                 t_admit.duration_since(req.received_at).as_secs_f64() * 1e3);
        if max_new <= 1 {
            let _ = req.tx.send(TokEvent::Finish { reason: "length".into() });
            return;
        }
        if self.eos.contains(&last) && !req.ignore_eos && 1 >= req.min_new {
            let _ = req.tx.send(TokEvent::Finish { reason: "stop".into() });
            return;
        }
        if let Some(sm) = req.schema.as_ref() {
            match schema_feed(sm, &mut schema_state, &mut schema_armed, last) {
                SchemaStep::Continue => {}
                SchemaStep::Complete => {
                    let _ = req.tx.send(TokEvent::Finish { reason: "stop".into() });
                    return;
                }
                SchemaStep::Invalid => {
                    crate::reprintln!("[schema] slot {free}: first token {last} leaves no valid continuation \
                                       — ending the turn with an error");
                    let _ = req.tx.send(TokEvent::Finish { reason: SCHEMA_DEAD_END.into() });
                    return;
                }
            }
        }
        // S-A3-f-d: greedy lanes on an M1 pack speculate by default (AGENTS 1b).
        // Prime the head once: snapshot the tap that predicted `last` (now the
        // seam decode step's sc.resid — exact, same composition as the bench-mtp
        // prime and every round's re-prime), then feed the head the last prompt
        // token with that tap (round 1's draft-extend reuses tap_row 0).
        // A JSON-schema lane decodes plain steps (its mask does not cover the verify rows yet); a lane that reports log
        // probabilities speculates like any other (round_emit reads the round's verify rows).
        if self.model.mtp.is_some() && !crate::exl3_forward::mtp_disabled_by_opt() && req.schema.is_some() {
            // the speed such a request pays, visible to scripts/owner_check.py (velogb10_spec_off_lanes_total)
            crate::metrics::spec_off_lane(true);
        }
        // log probabilities keep speculation (round_emit reads the round's verify rows); a schema lane decodes
        // plain steps until its mask covers the verify rows
        let mtp = if self.model.mtp.is_some() && !crate::exl3_forward::mtp_disabled_by_opt() && req.schema.is_none() {
            match self.mtp_prime(free, *req.prompt.last().unwrap(), plen) {
                Ok(()) => {
                    if crate::exl3_forward::a_xcheck_on() {
                        if let Err(e) = self.model.tap_xcheck_record(&self.sc, free, 0) {
                            eprintln!("[a-xcheck] record failed: {e:#}");
                        }
                    }
                    Some(0usize)
                }
                Err(e) => {
                    crate::reprintln!("[exl3-serve] MTP prime failed (lane runs plain): {e:#}");
                    if self.tp.is_some() || sticky_cuda_error(&e) { self.fatal = Some(format!("{e:#}")); }
                    None
                }
            }
        } else {
            None
        };
        // WP08: the reference implementation feeds the detector every emitted token, the first one included.
        let mut loop_det = self.loop_cfg
            .map(|(w, r)| crate::loop_detect::LoopDetector::for_stop_on_loop(w, r));
        if let Some(d) = loop_det.as_mut() { d.feed(last); }
        let is_img = !req.image_spans.is_empty(); // S-B9-CF-PACK4: an image lane never packs (v1)
        let admit_seq = self.admit_ctr;   // CF-FCFS: FCFS order; the same on every TP rank
        self.admit_ctr += 1;
        // WIDEVERIFY1: the request's copy-span proposer over its full prompt ids (known on a prefix-cache
        // hit too: the cache restores state, never the ids' availability) + the seam token; only for a
        // lane that speculates (a copy round is a speculative round).
        // COPYREC1: the record reads the prompt (the ids it does not hold yet) and keeps the reply for the release.
        let (rec_reply, rec_tail) = match self.record.as_mut() {
            Some(r) if !r.read_only() => {
                let th = std::time::Instant::now();
                if let Some(f) = r.add_prompt(&req.prompt) { crate::copy_record::count_fallback(f); }
                crate::copy_record::publish(r);
                crate::copy_record::add_host_us(th.elapsed().as_micros() as u64);
                (Some(vec![last]), req.prompt[req.prompt.len().saturating_sub(crate::copy_record::OVERLAP)..].to_vec())
            }
            _ => (None, Vec::new()),
        };
        let copy = match (self.copy_arm, mtp.is_some()) {
            (Some(arm), true) => {
                let th = std::time::Instant::now();
                let mut ctx = Vec::with_capacity(req.prompt.len() + 1);
                ctx.extend_from_slice(&req.prompt);
                ctx.push(last);
                let cs = Box::new(crate::copy_span::CopySpan::new(&ctx, arm.cfg));
                CS_HOST_US.fetch_add(th.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
                Some(cs)
            }
            _ => None,
        };
        self.lanes[free] = Some(Lane {
            pos: req.prompt.len(),
            last_tok: last,
            generated: 1,
            max_new,
            min_new: req.min_new,
            ignore_eos: req.ignore_eos,
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            min_p: req.min_p,
            pen,
            loop_det,
            think: think.take(),
            prompt_hash: prompt_hash(&req.prompt),
            history: req.prompt.clone(),
            copy,
            copy_new: Vec::new(),
            rec_reply, rec_tail,
            st_copy_rounds: 0,
            st_copy_accepted: 0,
            st_copy_emitted: 0,
            cpv_next: (0, usize::MAX),
            st_cpv_steps: 0,
            st_cpv_offered: 0,
            st_cpv_accepted: 0,
            st_cpv_emitted: 0,
            st_cpv_rerun: 0,
            cl_work: Default::default(),
            seed,
            ctr: 1,
            dctr: 0,
            gate_first: true,
            ema_ms: 0.0,
            ema_tok: 0.0,
            ema_n: 0,
            ema_wall_ms: 0.0,
            last_wall: None,
            mtp_rounds: 0,
            // TP-I #7: a TP lane's cost prior is the TP=2 one (dds_prior; TP=1 = 6.0 unchanged)
            wp23: Wp23Lane { fit: CostFit::with_prior(dds_prior_w(self.tp_world())), ..Default::default() },
            st_rounds: 0,
            st_drafted: 0,
            st_accepted: 0,
            st_ms: 0.0,
            st_plain: 0,
            st_ctrl: 0,
            arms: Default::default(),
            first_toks: vec![last],
            reprobe: crate::mtp_reprobe::ReProbe::new(&self.reprobe_cfg),
            force_control: false,
            mtp,
            admit_seq,
            fcfs_credit: 0,
            img: is_img,
            logprobs: req.logprobs,
            schema: req.schema.clone(),
            schema_state,
            schema_armed,
            tx: req.tx,
        });
    }

    /// LANERES1: `admit` through the reservation policy — the pick is restricted to `allowed`,
    /// `kind` drives the counter and the log line. Same post-conditions as `admit`.
    pub(super) fn admit_scoped(&mut self, req: BatchRequest, allowed: &[usize], kind: crate::lane_reserve::Kind) {
        self.admit_req(req, Some((allowed, kind)));
        self.kv_pf = None;
        self.publish_kv();
    }
}
