//! The plain decode step and its helpers: schema masks, `decode_step`, penalty uploads, `batched_decode`.
//! Another `impl BatchScheduler` block (file map in `mod.rs`); the methods keep their exact text.
//! Binding invariants: greedy MTP stays bitwise lossless; spec gating is per source (AGENTS.md section 4).

use super::*;

impl BatchScheduler {
    /// W2 (Phase 13): per-lane vocabulary masks for the CURRENT FSM state of every lane in
    /// `batch_idx`, in device (batch) order. Returns (words, flags, any) for
    /// `GpuModel::upload_decode_masks`; `any == false` means "no schema lane present" and the
    /// caller uploads nothing (so the normal path is untouched).
    fn schema_decode_masks(&self, batch_idx: &[usize]) -> (Vec<u32>, Vec<i32>, bool) {
        let words = crate::gpu::mask_words_for(self.gpu.cfg().vocab_size);
        let any = batch_idx.iter()
            .any(|&i| self.lanes[i].as_ref().map_or(false, |l| l.schema.is_some()));
        if !any { return (Vec::new(), Vec::new(), false); }
        let mut w = vec![0u32; words * batch_idx.len()];
        let mut f = vec![0i32; batch_idx.len()];
        for (k, &i) in batch_idx.iter().enumerate() {
            if let Some(l) = self.lanes[i].as_ref() {
                if let Some(sm) = &l.schema {
                    let bits = sm.allowed(l.schema_state);
                    let len = bits.len().min(words);
                    w[k * words..k * words + len].copy_from_slice(&bits[..len]);
                    f[k] = 1;
                }
            }
        }
        (w, f, true)
    }

    /// W2: advance a lane's schema FSM by the tokens it just COMMITTED. Returns true when the
    /// document is complete (the turn is over: the JSON is finished) or the machine hit a state
    /// that can never be completed.
    pub(super) fn schema_advance(&mut self, i: usize, toks: &[u32]) -> bool {
        let lane = self.lanes[i].as_mut().unwrap();
        let sm = match lane.schema.clone() { Some(sm) => sm, None => return false };
        for &t in toks {
            lane.schema_state = match sm.step(lane.schema_state, t) {
                Some(ns) => ns,
                None => {
                    crate::reprintln!("[schema] lane {i}: token {t} is invalid in the current schema state \
                               — finishing the turn (this must not happen with the mask armed)");
                    return true;
                }
            };
        }
        sm.is_complete(lane.schema_state)
    }

    /// One scheduler decode step over the `b` active (front-packed) lanes. Two phases:
    ///   A. MTP-eligible lanes (greedy → mtp_lane_step; sampling → mtp_lane_step_sample when
    ///      stochastic MTP is on) are served one at a time — each emits 1+ tokens.
    ///   B. All remaining lanes are served by a single batched decode (one shared weight read),
    ///      each emitting exactly one token.
    pub(super) fn decode_step(&mut self, b: usize) {
        crate::metrics::sched_step(); // v0.7.3 gauge: rounds + busy
        crate::thermal::pace(); // NIGHT1: the thermal governor (once per scheduler step)
        let mut finished = vec![false; b];
        // Cancel lanes whose client is gone (disconnect, or the SSE generator dropped the receiver
        // on a stop-string hit): every send would silently fail and the lane would otherwise decode
        // to EOS/max_new, holding a batch slot and a share of every batched step. The compaction at
        // the end of the step does the teardown (frees the slot, attempts the Finish event).
        for i in 0..b {
            // TP serving takes its cancels from the wire (`tp_cancelled`), never from `is_closed()`
            // directly — a disconnect landing between the head's per-step detection and this sweep
            // would otherwise finish the lane a step early on the head only. Single-node is exactly
            // as before (`is_closed()` alone); the mirror's dummy txs never close on their own.
            if self.lanes[i].as_ref()
                .map_or(false, |l| l.tp_cancelled
                    || (!self.tp_serving
                        && (l.tx.is_closed()
                            || crate::server::stream_backlog_exceeded(l.tx.len())))) { // LR-5
                if !self.tp_serving
                    && !self.lanes[i].as_ref().map_or(false, |l| l.tp_cancelled || l.tx.is_closed()) {
                    crate::metrics::stream_backlog_cancel();
                    crate::reprintln!("[req] lane {i}: stream cancelled — unconsumed event backlog at the --stream-backlog-events limit (reader stalled; LR-5)");
                }
                finished[i] = true;
            }
        }

        // Phase A: per-lane MTP. Greedy lanes verify by argmax (bitwise lossless); sampling lanes
        // verify by speculative rejection sampling (distribution-exact). The split is decided by the
        // request's temperature, not by any server flag.
        //
        // MTP IS A SINGLE-LANE PATH -- this loop runs one lane at a time, each doing its own draft and
        // verify forwards. With two clients that is two full model passes per step, so concurrent
        // throughput was FLAT: measured 37.7 tok/s alone, 21.4 each with two clients, 10.7 each with
        // four, aggregate pinned at ~43 the whole way. Each client simply got 1/N of the machine.
        //
        // But BATCHING IS FREE HERE, and for the same reason speculation is: the quantized GEMM is a
        // fixed-shape kernel with N padded to 16, so a decode step is bound by the weight bytes, which
        // do not change with the batch. A 4-lane batched step (Phase B) costs what a 1-lane step costs.
        //
        // So above one lane, batching beats speculation and it is not close: 4 lanes at ~1 forward
        // total, versus 4 lanes at ~4 speculative forwards for ~2.5 tokens each. Speculation only wins
        // when there is nothing to batch WITH. Hence: MTP at b == 1, batched decode at b >= 2.
        //
        // (The real prize is batching the MTP VERIFY across lanes -- attn_dispatch already takes
        // per-column slot_ids and pos, so the attention supports mixed slots; the GDN recurrent state
        // and the per-lane accept/rollback bookkeeping are what make it a project rather than a patch.
        // That would give both: N lanes AND ~2.5 tokens per lane per step.)
        //
        // (P11 note: the "real prize" above EXISTS — the FOREST path behind --mtp-lanes packs
        //  concurrent greedy lanes' chains into ONE verify (take ≤ 5, per-lane depth =
        //  MAX_VERIFY/p − 1), default OFF pending the promote-or-kill measurement. The b==1
        //  split stands only for the ROUND-based drafters (df2/dspark own one shared ring).)
        let policy_active = self.mtp.active();
        // S5F: which speculation source is live THIS step. DFlash2 (S4F's integrated round) runs
        // ONLY at b==1 (AGENTS §4 — same as the MTP chain), for a lane that primed the round and
        // never went stale. The MTP path stays live for the Mtp/Dspark sources AND as the DFlash2
        // fallback (unprimed lane, failed prime, or a lane that went stale). Plain never speculates.
        let src = self.mtp.spec_source();
        let df2_live = policy_active && is_df2_src(src) && self.df2.is_some() && b == 1
            && !self.lanes.get(0).and_then(|l| l.as_ref())
                   .map_or(false, |l| l.schema.is_some());
        // WI1: the DSpark lane owns the step exactly like df2_live (b==1, primed, not stale).
        let dspark_live = policy_active && !df2_live && src == SpecSource::Dspark
            && self.dspark.is_some() && b == 1
            && !self.lanes.get(0).and_then(|l| l.as_ref())
                   .map_or(false, |l| l.schema.is_some());
        // P14: the DFlash v1 lane is a single-sequence block drafter — b == 1 only, exactly like
        // the other round-based lanes (AGENTS §4: above one lane, batching beats speculation).
        // W2 (Phase 13): a lane step that does not arm the JSON-schema mask would emit
        // UNCONSTRAINED tokens for a schema request (`df2/dspark/dflash/tree/forest/sampled` lane
        // steps are schema-blind today; only the greedy MTP chain step is wired). A schema lane is
        // therefore kept OFF every schema-blind source: it is served by the MTP chain when that is
        // live, else by the masked Phase-B decode — never silently by an unmasked lane.
        let schema_lane = self.lanes.get(0).and_then(|l| l.as_ref())
            .map_or(false, |l| l.schema.is_some() && b == 1);
        let dflash_live = policy_active && !df2_live && !dspark_live && src == SpecSource::DFlash
            && self.dflash.is_some() && b == 1 && !schema_lane;
        let mtp_live = policy_active && !df2_live && !dspark_live && !dflash_live && src != SpecSource::Plain;
        // `served[i]` = lane i was served by Phase A (speculation) this step — Phase B decodes
        // exactly the lanes Phase A did NOT serve (a lane is never double-served, never stranded).
        let mut served = vec![false; b];
        if mtp_dispatch_trace() && b == 1 {
            crate::reprintln!("[b31-trace] dispatch: active={} src={:?} df2_live={} dflash_live={} dspark_live={} mtp_live={}",
                      policy_active, src, df2_live, dflash_live, dspark_live, mtp_live);
        }
        // COPY27B (--copy-span on): a lone lane whose output is copying its own context gets a COPY
        // ROUND in place of this step's drafter round (no draft pass; the same verify judges the
        // copied tokens). Off (`copy_arm == None`): nothing below runs and the step is unchanged.
        let mut copy_served = false;
        // COPY27B accounting: per-lane phase/kind snapshot of this step
        let acct_t0 = std::time::Instant::now();
        let acct_depth_now = self.mtp.depth();
        let acct_pre: Vec<(bool, usize)> = (0..b).map(|i| self.lanes[i].as_ref()
            .map_or((false, 0), |l| (l.in_reason, l.generated))).collect();
        if self.copy_arm.is_some() {
            use crate::exl3_serve::{copy_span_count_skip, CopySkip};
            if b != 1 {
                copy_span_count_skip(CopySkip::Concurrent);
            } else {
                let lane = self.lanes[0].as_ref().unwrap();
                // a lane without live drafter state still takes BARE copy rounds when the verify's
                // checkpoint band exists (MTP head present): they replace plain decode steps
                let bare = |r: CopySkip| if self.mtp.head_present() { Ok(CopyMode::Bare) } else { Err(r) };
                let mode = if lane.schema.is_some() {
                    Err(CopySkip::Schema)
                } else if !policy_active {
                    bare(CopySkip::SpecOff)
                } else if df2_live {
                    if lane.df2_primed && !lane.df2_stale && self.df2_ring_at(lane.pos) { Ok(CopyMode::Df2) }
                    else if lane.use_mtp(true) { Ok(CopyMode::Mtp) }
                    else { bare(CopySkip::NotPrimed) }
                } else if dspark_live || dflash_live {
                    Err(CopySkip::Source)
                } else if mtp_live {
                    if lane.use_mtp(true) { Ok(CopyMode::Mtp) } else { bare(CopySkip::NotPrimed) }
                } else {
                    bare(CopySkip::SpecOff)
                };
                match mode {
                    Err(r) => copy_span_count_skip(r),
                    Ok(m) => {
                        let mut p = self.copy_propose(0, m);
                        if self.gpu.tp_world() > 1 { p = Self::copy_tp_agree(p); }
                        if let Some(p) = p {
                            copy_served = true;
                            served[0] = true;
                            if self.copy_lane_step(0, m, &p) { finished[0] = true; }
                        }
                    }
                }
            }
        }
        if copy_served {
            // the copy round served the lone lane (the drafter's round did not run this step)
        } else if df2_live {
            crate::metrics::df2_round();
            let lane = self.lanes[0].as_ref().unwrap();
            // Second line of defence for the shared-ring class (see admission): a lane whose ring
            // frontier is not its own position never enters the round (its asserts would kill the
            // scheduler); it takes the MTP fallback below and is marked stale there.
            let ring_ok = self.df2_ring_at(lane.pos);
            if lane.df2_primed && !lane.df2_stale && !ring_ok {
                crate::reprintln!("[df2] ring frontier {} != lane pos {}: lane continues on MTP",
                          self.df2.as_ref().map_or(0, |d| d.nprev()), lane.pos);
            }
            if lane.df2_primed && !lane.df2_stale && ring_ok {
                served[0] = true;
                // S8F (S6F adjudication): `DFlash2Auto` resolves to the per-request lane (greedy
                // on code, real-q on math/chat/prose); explicit sources stay explicit. The rq
                // source runs the SAMPLED selector path under the real-q verify (u*q < p);
                // dflash2 keeps the greedy drafts + q=1 default. P3(a) close: `prose_lane_greedy`
                // routes GREEDY (temp-0) General requests to the greedy-draft lane (the quad
                // sweep's +10.5% prose tau); sampled-temp General keeps the real-q walk.
                let src = df2_effective_src(src, lane.domain, self.prose_lane_greedy, lane.greedy);
                let done = match src {
                    SpecSource::DFlash2Rq => self.df2_lane_step_sample_rq(0),
                    // PLAN/25 Phase 1: the tree lane (greedy v0; sampled falls back to the rq walk).
                    SpecSource::DFlash2Tree if lane.greedy => self.df2_tree_step(0),
                    _ if lane.greedy => self.df2_lane_step(0),
                    _ => self.df2_lane_step_sample(0),
                };
                if done { finished[0] = true; }
            } else if lane.use_mtp(true) {
                // The DFlash2 lane degraded (never primed / prime failed / went stale): serve via
                // the MTP path — the standing directive's fallback, never a hard failure.
                served[0] = true;
                let df2_was_primed = lane.df2_primed;
                let is_greedy = lane.greedy;
                if df2_was_primed { self.lanes[0].as_mut().unwrap().df2_stale = true; }
                let done = if is_greedy { self.mtp_lane_step(0) } else { self.mtp_lane_step_sample(0) };
                if done { finished[0] = true; }
            }
        } else if dflash_live {
            let lane = self.lanes[0].as_ref().unwrap();
            // `df2_primed` is the generic "this lane's drafter state was primed at admit" flag
            // (the v1 prime is a ctx-feature copy, the DF2 prime a ring walk — same contract).
            if lane.greedy && (lane.df2_primed || self.dflash_prime_done) && !lane.df2_stale {
                served[0] = true;
                let done = self.dflash_lane_step(0);
                if done { finished[0] = true; }
            } else if lane.use_mtp(true) {
                // Unprimed (prefix-cache reuse) or stale: the standing MTP fallback — never silent,
                // the admit path logs the reason once.
                served[0] = true;
                let was_primed = lane.df2_primed;
                let is_greedy = lane.greedy;
                if was_primed { self.lanes[0].as_mut().unwrap().df2_stale = true; }
                let done = if is_greedy { self.mtp_lane_step(0) } else { self.mtp_lane_step_sample(0) };
                if done { finished[0] = true; }
            }
        } else if dspark_live {
            let lane = self.lanes[0].as_ref().unwrap();
            let ring_ok = self.dspark_ring_at(lane.pos);
            if lane.df2_primed && !lane.df2_stale && !ring_ok {
                crate::reprintln!("[dspark] ring frontier {} != lane pos {}: lane continues on MTP",
                          self.dspark.as_ref().map_or(0, |d| d.nprev()), lane.pos);
            }
            if lane.df2_primed && !lane.df2_stale && ring_ok {
                served[0] = true;
                // WI1 v1: the greedy drafts on greedy lanes, the q=1 rejection-sampling verify on
                // sampled lanes (the DF2 lane pair's shape; DSpark has no tree/selector variants).
                let done = if lane.greedy { self.dspark_lane_step(0) } else { self.dspark_lane_step_sample(0) };
                if done { finished[0] = true; }
            } else if lane.use_mtp(true) {
                // The DSpark lane degraded (never primed / prime failed / went stale): the MTP
                // fallback — the standing directive, never a hard failure.
                served[0] = true;
                let was_primed = lane.df2_primed;
                let is_greedy = lane.greedy;
                if was_primed { self.lanes[0].as_mut().unwrap().df2_stale = true; }
                let done = if is_greedy { self.mtp_lane_step(0) } else { self.mtp_lane_step_sample(0) };
                if done { finished[0] = true; }
            }
        } else if mtp_live {
            if self.mtp_lanes {
                // FOREST: pack the greedy, primed, non-stale lanes (penalty carried per-column) into ONE
                // verify. Overflow beyond the column budget and any single leftover run the single-lane
                // MTP path, so NO eligible lane sits out (which strands it on plain decode via mtp_stale).
                let forest: Vec<usize> = (0..b).filter(|&i| {
                    let l = self.lanes[i].as_ref().unwrap();
                    l.greedy && l.use_mtp(true)
                }).collect();
                for &i in &forest { served[i] = true; }
                let take = forest.len().min(5);   // 16-column budget: p=5 ⇒ depth 2; p=6 ⇒ depth 1 (no drafts — useless)
                if take >= 2 {
                    let packed: Vec<usize> = forest[..take].to_vec();
                    for (i, fin) in self.mtp_forest_step(&packed) { if fin { finished[i] = true; } }
                    for &i in &forest[take..] { if self.mtp_lane_step(i) { finished[i] = true; } }
                } else {
                    for &i in &forest { if self.mtp_lane_step(i) { finished[i] = true; } }
                }
                // Sampling (non-greedy) primed lanes keep the single-lane stochastic path (v1).
                for i in 0..b {
                    let l = self.lanes[i].as_ref().unwrap();
                    if l.use_mtp(true) && !l.greedy {
                        served[i] = true;
                        if self.mtp_lane_step_sample(i) { finished[i] = true; }
                    }
                }
                // A DFlash2-primed lane that just took an MTP/forest step can never resume the
                // round (its ring is missing this step's taps) — mirror the Phase-B stale rule.
                for i in 0..b {
                    let l = self.lanes[i].as_ref().unwrap();
                    if l.df2_primed && !l.df2_stale {
                        self.lanes[i].as_mut().unwrap().df2_stale = true;
                    }
                }
            } else if b == 1 {
                let lane = self.lanes[0].as_ref().unwrap();
                let is_greedy = lane.greedy;
                if mtp_dispatch_trace() {
                    crate::reprintln!("[b31-trace] step b=1: use_mtp={} primed={} stale={} greedy={} schema={}",
                              lane.use_mtp(true), lane.mtp_primed, lane.mtp_stale, lane.greedy, lane.schema.is_some());
                }
                if lane.use_mtp(true) {
                    served[0] = true;
                    if lane.df2_primed && !lane.df2_stale {
                        // DFlash2 was live for this lane but this step runs MTP (e.g. the source
                        // flipped at a policy window) — the ring is now missing this step's taps.
                        self.lanes[0].as_mut().unwrap().df2_stale = true;
                    }
                    let done = if is_greedy { self.mtp_lane_step(0) } else { self.mtp_lane_step_sample(0) };
                    if done { finished[0] = true; }
                }
            }
        }
        // E17: the depth policy prices r(d) from the CURRENT context's bucket — feed it the deepest
        // live position (identical on both TP ranks: positions are lockstep).
        let mtp_ctx = self.lanes[..b].iter()
            .filter_map(|l| l.as_ref().map(|l| l.pos)).max().unwrap_or(0);
        self.mtp.tick(mtp_ctx);
        // B31 FIX: publish the MTP telemetry EVERY decode step, not only where the `[mtp]` meter
        // prints (every 50 steps). The 50-step cadence left a window where /health reported
        // drafts=0 on a server that was actively speculating — a pinned-d4 short FIRST request
        // finishes in ~32 steps and never crosses the boundary, so --probe-serve-pinned's
        // engagement assert read a false FAIL (B31, reproduced x3 on .13/.14). The publish is a
        // handful of Relaxed atomic stores against a ~70ms step; the watermark skips plain-only
        // steps. The 50-step LOG cadence is unchanged.
        if self.mtp_stat_steps != self.mtp_stat_published {
            crate::tel::publish_mtp(self.mtp_stat_steps, self.mtp_stat_drafts,
                                    self.mtp_stat_accepted, self.mtp_stat_emitted,
                                    self.mtp_stat_verify_fwds, self.mtp.depth(),
                                    &self.mtp.hazard_counts());
            self.mtp_stat_published = self.mtp_stat_steps;
        }
        // Device-resident loop: any MTP step emits 2+ tokens and advances pos by >1, which the
        // per-step ring push / pos increment cannot represent — the next batched step re-uploads.
        if policy_active { self.resident_dirty = true; }

        // Phase B: batched decode for the lanes Phase A did NOT serve (the served set is the
        // single source of truth — a source switch can never strand or double-serve a lane).
        let batch_idx: Vec<usize> = (0..b).filter(|&i| !served[i]).collect();
        if mtp_dispatch_trace() && policy_active && !batch_idx.is_empty() {
            crate::reprintln!("[b31-trace] phaseB: b={} unserved={:?} (policy active but lanes fell to plain decode)",
                      b, batch_idx);
        }
        if !batch_idx.is_empty() {
            // W2 (Phase 13): ARM THE DECODE MASK for this batch. `batched_decode` already refuses
            // the resident loop and the captured graphs for a schema batch, but the mask itself was
            // never uploaded (`schema_decode_masks` / `upload_decode_masks` had no caller), so a
            // Phase-B step emitted UNMASKED — the escape that forced the W2 floor. The mask is
            // per-lane and per-step (the FSM state moves with every committed token).
            //
            // While no lane carries a schema this block does NOTHING — not one upload, not one
            // allocation — so the hot decode path is untouched by it.
            let armed = if batch_idx.iter().any(|&i| self.lanes[i].as_ref().is_some_and(|l| l.schema.is_some())) {
                let (words, flags, any) = self.schema_decode_masks(&batch_idx);
                if any { self.gpu.upload_decode_masks(&mut self.bufs, &words, &flags, true); }
                any
            } else {
                false
            };
            let next_toks = self.batched_decode(&batch_idx);
            // The mask kernel is armed for exactly this step: disarm it immediately after, so a
            // later non-schema caller can never inherit a stale mask.
            if armed { self.gpu.upload_decode_masks(&mut self.bufs, &[], &[], false); }
            for (k, &i) in batch_idx.iter().enumerate() {
                let t = next_toks[k];
                let lane = self.lanes[i].as_mut().unwrap();
                // This lane just advanced WITHOUT writing MTP KV: the head now has a hole at this
                // position and can never be trusted again for this request. See Lane::mtp_stale.
                lane.mtp_stale = true;
                // S5F: same for the DFlash2 ring — a batched step did not advance the round's nprev.
                lane.df2_stale = true;
                let _ = lane.tx.send(TokEvent::Tok(t));
                // THE CACHE RECORDS WHAT THE STATE HAS CONSUMED, NOT WHAT WE EMITTED. This step fed the
                // PREVIOUS token (`last_tok`) through the model at position `pos`; `t` is the model's
                // prediction and has not been fed yet. Caching `t` instead would put a token in the
                // cache that the state has never seen — and the next turn would reuse that state as if
                // it had, silently producing wrong output. Invariant: slot_cache.len() == lane.pos.
                let fed = lane.last_tok;
                let phys = lane.phys;
                lane.last_tok = t;
                lane.pos += 1;
                lane.generated += 1;
                lane.hist_push(t);
                self.slot_cache[phys].push(fed);
                debug_assert_eq!(self.slot_cache[phys].len(), lane.pos);
                lane.hist_trim();
                let eos_hit = self.eos.contains(&t) && !lane.ignore_eos && lane.generated >= lane.min_new;
                if eos_hit || lane.generated >= lane.max_new {
                    finished[i] = true;
                }
                // W2: advance the schema FSM over the token this step actually emitted, and end the
                // turn when the document is complete (a schema turn stops at the closing brace, it
                // does not run to max_tokens).
                if self.lanes[i].as_ref().unwrap().schema.is_some() {
                    let done = self.schema_advance(i, &[t]);
                    if done { finished[i] = true; }
                }
            }
        }

        // COPY27B accounting: attribute this step to each lane's phase and round kind
        {
            let us = acct_t0.elapsed().as_micros() as u64;
            for i in 0..b {
                let Some(l) = self.lanes[i].as_mut() else { continue };
                let (ph_reason, g0) = acct_pre[i];
                let ph = if ph_reason { 0 } else { 1 };
                let kind = if b == 1 && copy_served { 1 } else if served[i] { 0 } else { 2 };
                let nt = l.generated.saturating_sub(g0);
                let e = &mut l.acct[ph][kind];
                e.0 += 1; e.1 += nt as u64; e.2 += us;
                if kind == 0 { l.acct_depth[ph][acct_depth_now.min(MAX_AUTO_DEPTH)] += 1; }
                let hl = l.history.len();
                let tail: Vec<u32> = l.history[hl.saturating_sub(nt)..].to_vec();
                l.in_reason = think_state_after(l.in_reason, &tail);
            }
        }
        // compact: keep non-finished lanes front-packed. Each lane keeps its physical slot —
        // finished lanes return their slot to the free list; no state copying needed.
        let mut write = 0usize;
        for i in 0..b {
            if finished[i] {
                let lane = self.lanes[i].take().unwrap();
                {
                    let n = ACCT_REQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let f = |a: &[(u64, u64, u64); 3], d: &[u64; MAX_AUTO_DEPTH + 1]| {
                        let ds: Vec<String> = d.iter().enumerate().filter(|(_, &c)| c > 0).map(|(k, c)| format!("d{k}:{c}")).collect();
                        format!("drafter {} {} {} copy {} {} {} plain {} {} {} depth [{}]",
                                a[0].0, a[0].1, a[0].2 / 1000, a[1].0, a[1].1, a[1].2 / 1000, a[2].0, a[2].1, a[2].2 / 1000, ds.join(" "))
                    };
                    crate::reprintln!("[copy-acct] req={n} gen={} reasoning: {} | answer: {}", lane.generated,
                                      f(&lane.acct[0], &lane.acct_depth[0]), f(&lane.acct[1], &lane.acct_depth[1]));
                }
                self.free_slots.push(lane.phys);
                let reason = if lane.generated >= lane.max_new { "length" } else { "stop" };
                let _ = lane.tx.send(TokEvent::Finish { reason: reason.to_string() });
                // PLAN/25 Phase 0: close the serving dump job (no-op when none is open; in bench
                // mode run_spec_bench's own job_end then no-ops the same way — taps are written
                // exactly once, here, under the bench tag).
                if let Some(d) = self.step_dump.as_mut() { d.job_end(); }
            } else {
                self.lanes[write] = self.lanes[i].take();
                write += 1;
            }
        }
        // Device-resident loop: lane composition changed (a lane finished) — re-upload next step.
        if write < b { self.resident_dirty = true; }
        crate::metrics::sched_touch(); // H3: round COMPLETE — the age gauge measures from here
    }

    /// LOOPFIX (2026-11-06): bring the device's whole-output penalty counts up to date for these
    /// lanes — upload every token each penalized lane committed since its last sync. Called before
    /// anything applies a penalty (batched decode, every verify builder). No-op for unpenalized lanes.
    fn pen_sync(&mut self, lane_idx: &[usize]) {
        let mut pairs: Vec<(i32, i32)> = Vec::new();
        for &i in lane_idx {
            let Some(l) = self.lanes[i].as_mut() else { continue };
            if !l.has_penalty() { l.pen_synced = l.pen_total; continue; }
            let pending = (l.pen_total - l.pen_synced) as usize;
            if pending == 0 { continue; }
            let n = l.history.len();
            assert!(pending <= n, "pen_sync: {pending} uncounted tokens but history holds {n} (hist_trim invariant)");
            for &t in &l.history[n - pending..] { pairs.push((l.phys as i32, t as i32)); }
            l.pen_synced = l.pen_total;
        }
        if !pairs.is_empty() { self.gpu.pen_count_add(&mut self.bufs, &pairs); }
    }

    /// Upload a verify penalty's per-column descriptors (`slots`, `extra` = [cap][PEN_EXTRA]) and
    /// VALUES (`uniform` = one (rep, presence, freq) for every column, cached by `pen_const_key`;
    /// else per-column vectors) and return the descriptor the verify kernels read.
    fn verify_pen_upload(&mut self, slots: &[i32], extra: &[i32], uniform: Option<(f32, f32, f32)>,
                         per_col: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>) -> crate::gpu::VerifyPenalty {
        let cap = crate::gpu::MAX_VERIFY;
        debug_assert_eq!(slots.len(), cap);
        debug_assert_eq!(extra.len(), cap * crate::gpu::PEN_EXTRA);
        self.gpu.dev().htod_sync_copy_into(slots, self.mtp_pen_slots.as_mut().unwrap()).unwrap();
        self.gpu.dev().htod_sync_copy_into(extra, self.mtp_pen_extra.as_mut().unwrap()).unwrap();
        match (uniform, per_col) {
            (Some((rep_pen, presence_pen, freq_pen)), _) => {
                // The VALUES are request-constant: upload them only when they (or the width) change.
                let key = (rep_pen.to_bits(), presence_pen.to_bits(), freq_pen.to_bits(), cap);
                if self.pen_const_key != Some(key) {
                    self.gpu.dev().htod_sync_copy_into(&vec![rep_pen; cap], self.mtp_pen_rep.as_mut().unwrap()).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&vec![presence_pen; cap], self.mtp_pen_presence.as_mut().unwrap()).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&vec![freq_pen; cap], self.mtp_pen_freq.as_mut().unwrap()).unwrap();
                    self.pen_const_key = Some(key);
                }
            }
            (None, Some((rv, pv, fv))) => {
                self.gpu.dev().htod_sync_copy_into(&rv, self.mtp_pen_rep.as_mut().unwrap()).unwrap();
                self.gpu.dev().htod_sync_copy_into(&pv, self.mtp_pen_presence.as_mut().unwrap()).unwrap();
                self.gpu.dev().htod_sync_copy_into(&fv, self.mtp_pen_freq.as_mut().unwrap()).unwrap();
                // Per-column values overwrote the cached uniform ones: the next uniform caller re-uploads.
                self.pen_const_key = None;
            }
            (None, None) => unreachable!("verify_pen_upload: no values"),
        }
        // No dev().synchronize(): host-blocking NULL-stream copies + the blocking compute stream
        // already order these before the verify kernels that read them (invariant I1).
        crate::gpu::VerifyPenalty {
            slots_ptr: *self.mtp_pen_slots.as_ref().unwrap().device_ptr(),
            extra_ptr: *self.mtp_pen_extra.as_ref().unwrap().device_ptr(),
            cnt_ptr: *self.bufs.pen_cnt_dev.device_ptr(),
            pbits_ptr: *self.bufs.pen_pbits_dev.device_ptr(),
            rep_pen_ptr: *self.mtp_pen_rep.as_ref().unwrap().device_ptr(),
            presence_ptr: *self.mtp_pen_presence.as_ref().unwrap().device_ptr(),
            freq_ptr: *self.mtp_pen_freq.as_ref().unwrap().device_ptr(),
        }
    }

    /// Verify penalty where every column shares lane `i`'s committed output (no per-column extras) —
    /// the chain/tree paths that predate per-column histories. Returns None if no penalty.
    /// LOOPFIX: whole output (+ prompt for repetition), was the last <= 64 distinct tokens.
    pub(super) fn make_penalty(&mut self, i: usize, rep_pen: f32, presence_pen: f32, freq_pen: f32,
                    has_penalty: bool) -> Option<crate::gpu::VerifyPenalty> {
        if !has_penalty { return None; }
        self.pen_sync(&[i]);
        let phys = self.lanes[i].as_ref().unwrap().phys as i32;
        let cap = crate::gpu::MAX_VERIFY;
        let slots = vec![phys; cap];
        let extra = vec![-1i32; cap * crate::gpu::PEN_EXTRA];
        Some(self.verify_pen_upload(&slots, &extra, Some((rep_pen, presence_pen, freq_pen)), None))
    }

    /// S-B16: MTP verify penalty with PER-COLUMN histories. Plain decode penalizes with the
    /// EVOLVING history (each emitted token enters it before the next argmax); column t of the verify
    /// therefore gets the committed output + drafts[0..t] — exactly what plain decode would have
    /// penalized at that column (spec on/off byte identity for greedy penalized requests). Columns
    /// >= n (unused: the launches use grid = n) repeat the deepest prefix.
    /// LOOPFIX: the committed part is the whole output (+ prompt for repetition) on the device.
    pub(super) fn make_penalty_drafts(&mut self, i: usize, drafts: &[u32], rep_pen: f32,
                           presence_pen: f32, freq_pen: f32, has_penalty: bool)
                           -> Option<crate::gpu::VerifyPenalty> {
        if !has_penalty { return None; }
        self.pen_sync(&[i]);
        let phys = self.lanes[i].as_ref().unwrap().phys as i32;
        let cap = crate::gpu::MAX_VERIFY;
        let pe = crate::gpu::PEN_EXTRA;
        let slots = vec![phys; cap];
        let extra = pen_draft_extras(drafts, cap, pe);
        Some(self.verify_pen_upload(&slots, &extra, Some((rep_pen, presence_pen, freq_pen)), None))
    }

    /// FOREST per-column penalty: column c gets ITS lane's rep/presence/freq penalty over that lane's
    /// committed output. `lanes` is (start, len, rep, presence, freq, lane index) per packed lane;
    /// columns outside any lane and past the packed width are no-ops (slot -1, neutral values).
    /// Returns None if NO packed lane has a penalty. LOOPFIX: whole output, was the last <= 64.
    pub(super) fn make_forest_penalty(&mut self, lanes: &[(usize, usize, f32, f32, f32, usize)])
                           -> Option<crate::gpu::VerifyPenalty> {
        let any = lanes.iter().any(|(_, _, r, pr, f, _)| *r > 1.0 || *pr > 0.0 || *f > 0.0);
        if !any { return None; }
        let idx: Vec<usize> = lanes.iter().map(|l| l.5).collect();
        self.pen_sync(&idx);
        let cap = crate::gpu::MAX_VERIFY;
        let mut slots = vec![-1i32; cap];
        let extra = vec![-1i32; cap * crate::gpu::PEN_EXTRA];
        let mut rep_v = vec![1.0f32; cap];
        let mut pres_v = vec![0.0f32; cap];
        let mut freq_v = vec![0.0f32; cap];
        for &(start, len, rep, pres, freq, li) in lanes {
            let lane = self.lanes[li].as_ref().unwrap();
            if !lane.has_penalty() { continue; } // its counts are not maintained: leave its columns neutral
            let phys = lane.phys as i32;
            for c in start..(start + len).min(cap) {
                slots[c] = phys;
                rep_v[c] = rep; pres_v[c] = pres; freq_v[c] = freq;
            }
        }
        Some(self.verify_pen_upload(&slots, &extra, None, Some((rep_v, pres_v, freq_v))))
    }

    /// One batched decode step over a subset of lanes (`batch_idx`). Builds the per-lane input
    /// arrays (tokens/positions/penalties/slot map), uploads them, runs the appropriate forward path
    /// (greedy graph / GPU-sample graph / CPU-sample / non-graph greedy), and returns the next token
    /// per lane (length = batch_idx.len()).
    fn batched_decode(&mut self, batch_idx: &[usize]) -> Vec<u32> {
        // LOOPFIX: the step's penalty reads each penalized lane's whole-output counts — bring them up
        // to date with everything committed since the last sync (any path: decode, MTP, DF2, ...).
        self.pen_sync(batch_idx);
        let s = batch_idx.len();
        let mb = self.max_batch;
        // W2: a schema lane in this batch disables the resident loop AND the captured graphs —
        // both bake their kernel sequence (and the mask kernel's operands) at capture/admission
        // time, while the mask content changes every step. Schema batches take the eager core.
        let any_schema = batch_idx.iter().any(|&i| self.lanes[i].as_ref().unwrap().schema.is_some());
        let resident = crate::gpu::GpuModel::device_loop_on() && !any_schema;
        let any_sampling = batch_idx.iter().any(|&i| !self.lanes[i].as_ref().unwrap().greedy);
        let can_graph = !any_sampling && !any_schema && self.graphs.contains_key(&s);
        let max_pc = batch_idx.iter()
            .map(|&i| self.lanes[i].as_ref().unwrap().pos + 1).max().unwrap_or(1);

        if resident {
            // ---- DEVICE-RESIDENT TOKEN LOOP (EXPERT_DEVICE_ARGMAX_LOOP_RESPONSE §7.3) ----
            // Clean steps upload NOTHING: tokens/pos/slot_ids/ring/keys are device-resident and
            // current (the graph head's ids_advance_b / epilogue ring push advanced them last
            // step). Dirty steps (admission/finish/compaction/MTP/param change — any lane
            // composition change) re-upload everything below.
            if self.resident_dirty {
                let mut toks = vec![0i32; mb];
                let mut pos = vec![0i32; mb];
                let mut slot_ids: Vec<i32> = (0..s)
                    .map(|k| self.lanes[batch_idx[k]].as_ref().unwrap().phys as i32).collect();
                slot_ids.resize(mb, 0);
                let mut keys = vec![0u64; mb];
                let mut rep_pen = vec![1.0f32; mb];
                let mut presence_pen = vec![0.0f32; mb];
                let mut frequency_pen = vec![0.0f32; mb];
                for (k, &i) in batch_idx.iter().enumerate() {
                    let lane = self.lanes[i].as_ref().unwrap();
                    // The graph head's ids_advance_b copies token_ids_dev -> tokens_dev and
                    // INCREMENTS pos_dev BEFORE the forward reads it, so the uploads are the
                    // PRE-advance values: token_ids_dev = last_tok (the embed input) and
                    // pos = lane.pos - 1 (the forward consumes lane.pos).
                    toks[k] = lane.last_tok as i32;
                    pos[k] = lane.pos as i32 - 1;
                    // Per-lane penalty VALUES are request constants — upload on every dirty
                    // event. LOOPFIX: neutral for a lane without a penalty (its whole-output
                    // counts are not maintained), the same rule as the non-resident path.
                    if lane.has_penalty() {
                        rep_pen[k] = lane.rep_penalty;
                        presence_pen[k] = lane.presence_penalty;
                        frequency_pen[k] = lane.frequency_penalty;
                    }
                    keys[k] = lane.seed;
                }
                self.gpu.dev().htod_sync_copy_into(&toks, &mut self.bufs.token_ids_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&pos, &mut self.bufs.pos_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&slot_ids, &mut self.bufs.slot_ids_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&rep_pen, &mut self.bufs.rep_pen_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&presence_pen, &mut self.bufs.presence_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&frequency_pen, &mut self.bufs.frequency_dev).unwrap();
                if any_sampling {
                    // Sampling params are per-request constants; upload them with the keys (the
                    // draws themselves are produced on-device by seed_advance_b from the keys —
                    // seeds_dev is NEVER uploaded under the resident loop).
                    let mut t: Vec<f32> = batch_idx.iter()
                        .map(|&i| self.lanes[i].as_ref().unwrap().temperature).collect();
                    t.resize(mb, 1.0);
                    let mut ki: Vec<i32> = batch_idx.iter()
                        .map(|&i| crate::sampler::gpu_top_k(self.lanes[i].as_ref().unwrap().top_k)).collect();
                    ki.resize(mb, 1);
                    let mut p: Vec<f32> = batch_idx.iter()
                        .map(|&i| self.lanes[i].as_ref().unwrap().top_p).collect();
                    p.resize(mb, 1.0);
                    self.gpu.dev().htod_sync_copy_into(&t, &mut self.bufs.temps_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&ki, &mut self.bufs.topk_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&p, &mut self.bufs.topp_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&keys, &mut self.bufs.seeds_key_dev).unwrap();
                }
                self.resident_dirty = false;
            }
            // No dev().synchronize(): host-blocking NULL-stream copies + the blocking compute
            // stream already order these before the kernels/graph replay that read them (I1).

            // Dispatch — every path below runs the resident kernels (captured in the graphs or
            // composed in the eager core) when the flag is on.
            let can_graph = !any_sampling && self.graphs.contains_key(&s);
            let toks = if can_graph {
                let graph = self.graphs.get(&s).unwrap();
                self.gpu.replay_decode(&self.bufs, graph, s)
            } else if any_sampling {
                let temps: Vec<f32> = batch_idx.iter().map(|&i| self.lanes[i].as_ref().unwrap().temperature).collect();
                let tks: Vec<usize> = batch_idx.iter().map(|&i| self.lanes[i].as_ref().unwrap().top_k).collect();
                let tps: Vec<f32> = batch_idx.iter().map(|&i| self.lanes[i].as_ref().unwrap().top_p).collect();
                if self.gpu_sample {
                    match self.sample_graphs.get(&s) {
                        Some(g) => self.gpu.replay_decode_sample(&self.bufs, g, s),
                        None => self.gpu.forward_decode_sample_gpu(
                            &mut self.pool, &mut self.bufs, &mut self.state, self.kv_stride, max_pc, s),
                    }
                } else {
                    // CPU-sampling escape (--cpu-sample): the host needs the full
                    // logits, so the resident loop is pointless AND its pos semantics differ (no
                    // ids_advance) — run today's sequence with today's uploads, then force the
                    // next step to re-upload (the pos_dev/tokens_dev left behind are in
                    // non-resident semantics).
                    let mut t = temps.clone(); t.resize(mb, 1.0);
                    let mut ki: Vec<i32> = tks.iter().map(|&x| x as i32).collect(); ki.resize(mb, 1);
                    let mut p = tps.clone(); p.resize(mb, 1.0);
                    let mut toks_v = vec![0i32; mb];
                    let mut pos_v = vec![0i32; mb];
                    for (k, &i) in batch_idx.iter().enumerate() {
                        let lane = self.lanes[i].as_ref().unwrap();
                        toks_v[k] = lane.last_tok as i32;
                        pos_v[k] = lane.pos as i32;
                    }
                    self.gpu.dev().htod_sync_copy_into(&t, &mut self.bufs.temps_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&ki, &mut self.bufs.topk_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&p, &mut self.bufs.topp_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&toks_v, &mut self.bufs.tokens_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&pos_v, &mut self.bufs.pos_dev).unwrap();
                    self.resident_dirty = true;
                    self.gpu.forward_decode_sample(
                        &mut self.pool, &mut self.bufs, &mut self.state, self.kv_stride, max_pc, s,
                        &temps, &tks, &tps)
                }
            } else {
                self.gpu.forward_decode(
                    &mut self.pool, &mut self.bufs, &mut self.state, self.kv_stride, max_pc, s)
            };
            if any_sampling && self.gpu_sample {
                // Host key bookkeeping: the device (sample graph / resident core) advanced the
                // keys exactly once this step; keep the host keys in lockstep so a dirty
                // re-upload never replays stale draws. (The CPU-sample escape advances nothing
                // on device and leaves the keys untouched — the forced dirty flag makes the
                // next step re-upload them, still in sync.)
                for k in 0..s {
                    let lane = self.lanes[batch_idx[k]].as_mut().unwrap();
                    lane.seed = splitmix64(lane.seed);
                }
            }
            toks
        } else {
            // ---- today's path, VERBATIM (uploads + dispatch) ----
            let mut toks = vec![0i32; mb];
            let mut pos = vec![0i32; mb];
            let mut rep_pen = vec![1.0f32; mb];
            let mut presence_pen = vec![0.0f32; mb];
            let mut frequency_pen = vec![0.0f32; mb];
            for (k, &i) in batch_idx.iter().enumerate() {
                let lane = self.lanes[i].as_ref().unwrap();
                toks[k] = lane.last_tok as i32;
                pos[k] = lane.pos as i32;
                // LOOPFIX: the penalty itself is the slot's whole-output state (pen_sync above);
                // only a penalized lane gets non-neutral VALUES (its counts are maintained).
                if lane.has_penalty() {
                    rep_pen[k] = lane.rep_penalty;
                    presence_pen[k] = lane.presence_penalty;
                    frequency_pen[k] = lane.frequency_penalty;
                }
            }
            let mut slot_ids: Vec<i32> = (0..s)
                .map(|k| self.lanes[batch_idx[k]].as_ref().unwrap().phys as i32).collect();
            slot_ids.resize(mb, 0);
            self.gpu.dev().htod_sync_copy_into(&toks, &mut self.bufs.tokens_dev).unwrap();
            self.gpu.dev().htod_sync_copy_into(&pos, &mut self.bufs.pos_dev).unwrap();
            self.gpu.dev().htod_sync_copy_into(&slot_ids, &mut self.bufs.slot_ids_dev).unwrap();
            // The three penalty VALUE arrays only need uploading when some lane is actually penalized,
            // plus ONE clear when the last penalized lane departs (so its values cannot linger into an
            // unpenalized successor — pen_apply_b returns at once on a neutral row).
            let any_pen = batch_idx.iter().any(|&i| self.lanes[i].as_ref().unwrap().has_penalty());
            if any_pen || self.pen_had {
                self.gpu.dev().htod_sync_copy_into(&rep_pen, &mut self.bufs.rep_pen_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&presence_pen, &mut self.bufs.presence_dev).unwrap();
                self.gpu.dev().htod_sync_copy_into(&frequency_pen, &mut self.bufs.frequency_dev).unwrap();
                self.pen_had = any_pen;
            }
            // No dev().synchronize(): host-blocking NULL-stream copies + the blocking compute stream
            // already order these before the kernels/graph replay that read them (invariant I1).

            let toks = if can_graph {
                let graph = self.graphs.get(&s).unwrap();
                self.gpu.replay_decode(&self.bufs, graph, s)
            } else if any_sampling {
                let temps: Vec<f32> = batch_idx.iter().map(|&i| self.lanes[i].as_ref().unwrap().temperature).collect();
                let tks: Vec<usize> = batch_idx.iter().map(|&i| self.lanes[i].as_ref().unwrap().top_k).collect();
                let tps: Vec<f32> = batch_idx.iter().map(|&i| self.lanes[i].as_ref().unwrap().top_p).collect();
                if self.gpu_sample {
                    // htod sampling params + fresh seeds into bufs (NULL stream), then dispatch to the
                    // captured decode+sample graph when available, else the non-graph core.
                    let mut t = temps.clone(); t.resize(mb, 1.0);
                    let mut ki: Vec<i32> = tks.iter().map(|&x| x as i32).collect(); ki.resize(mb, 1);
                    let mut p = tps.clone(); p.resize(mb, 1.0);
                    // Seed sample_b from each lane's own PRNG, not rand::random(). The lane already
                    // carries a seed (from the request's `seed` field), and the MTP path honours it —
                    // drawing a fresh OS-random seed here meant an explicit `"seed": 42` was silently
                    // ignored on the plain-sampler path, so identical requests were irreproducible.
                    // Advance each lane's key once per decode step, exactly as the MTP path does.
                    let mut sd: Vec<u32> = Vec::with_capacity(mb);
                    for k in 0..s {
                        let lane = self.lanes[batch_idx[k]].as_mut().unwrap();
                        sd.push(rng_u32(lane.seed, RNG_DOM_SAMPLE, 0));
                        lane.seed = splitmix64(lane.seed);
                    }
                    sd.resize(mb, 0);
                    self.gpu.dev().htod_sync_copy_into(&t, &mut self.bufs.temps_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&ki, &mut self.bufs.topk_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&p, &mut self.bufs.topp_dev).unwrap();
                    self.gpu.dev().htod_sync_copy_into(&sd, &mut self.bufs.seeds_dev).unwrap();
                    // No dev().synchronize(): host-blocking NULL-stream copies + the blocking compute
                    // stream already order these before the sample kernels/replay (invariant I1).
                    if let Some(g) = self.sample_graphs.get(&s) {
                        self.gpu.replay_decode_sample(&self.bufs, g, s)
                    } else {
                        self.gpu.forward_decode_sample_gpu(
                            &mut self.pool, &mut self.bufs, &mut self.state, self.kv_stride, max_pc, s)
                    }
                } else {
                    self.gpu.forward_decode_sample(
                        &mut self.pool, &mut self.bufs, &mut self.state, self.kv_stride, max_pc, s,
                        &temps, &tks, &tps)
                }
            } else {
                self.gpu.forward_decode(
                    &mut self.pool, &mut self.bufs, &mut self.state, self.kv_stride, max_pc, s)
            };
            toks
        }
    }
}
