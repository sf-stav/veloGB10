//! Single-lane MTP chain steps: greedy (`mtp_lane_step`, bitwise lossless), sampled, and the lossless audit shadow decode.
//! Another `impl BatchScheduler` block (file map in `mod.rs`); the methods keep their exact text.
//! Binding invariants: greedy MTP stays bitwise lossless; spec gating is per source (AGENTS.md section 4).

use super::*;

impl BatchScheduler {
    pub(super) fn mtp_lane_step(&mut self, i: usize) -> bool {
        crate::tel::note_step();   // A.2: per-step interval for the status route (lock-free)
        // Fork-then-chain tree path (opt-in): rescues the chain-killing first-token miss.
        if self.tree_draft && self.mtp.depth() >= 3 {
            return self.mtp_tree_step(i);
        }
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let depth = self.mtp.depth();
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let snapshot = self.mtp_snapshot_slot;
        let kv_stride = self.kv_stride;
        let mtp_kc_ptr = *self.mtp_kc[phys].device_ptr();
        let mtp_vc_ptr = *self.mtp_vc[phys].device_ptr();
        let h_save_ptr = *self.mtp_h_save.as_ref().unwrap().device_ptr();
        let cur_ptr = *self.mtp_cur_hidden.as_ref().unwrap().device_ptr();

        // Snapshot lane state into locals (avoids holding &mut self.lanes across GPU calls).
        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
        let mtp_pos = self.lanes[i].as_ref().unwrap().mtp_pos;
        let generated = self.lanes[i].as_ref().unwrap().generated;
        let max_new = self.lanes[i].as_ref().unwrap().max_new;
        let eos = self.eos.clone();
        // Phase-8: the stop rule is a property of the REQUEST, not the serving path —
        // every emit loop must honor ignore_eos/min_new exactly like the batched decode path
        // (batch.rs' `eos_hit`), or spec-on and spec-off streams stop at different tokens.
        let (ignore_eos, min_new) = { let l = self.lanes[i].as_ref().unwrap(); (l.ignore_eos, l.min_new) };
        // Penalty config + history (so the MTP verify keeps the lane's rep/presence/freq penalty).
        let (rep_pen, presence_pen, freq_pen, has_penalty) = {
            let l = self.lanes[i].as_ref().unwrap();
            (l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.has_penalty())
        };
        let history: Vec<u32> = self.lanes[i].as_ref().unwrap().history.clone();
        // W2: the lane's compiled schema + its current FSM state (None/0 = unconstrained).
        let lane_schema = self.lanes[i].as_ref().unwrap().schema.clone();
        let lane_schema_state = self.lanes[i].as_ref().unwrap().schema_state;

        // h_save = h_prev (hidden at main_pos-1); saved for the post-accept re-prime step k=0.
        self.gpu.copy_hidden_col(h_save_ptr, &self.mtp_h_prev[phys], 0);

        // Prompt-lookup context for n-gram drafting: the full realized sequence for this lane, which
        // slot_cache holds (prompt + every committed token), plus committed_tok (this step's first
        // token to verify, which slot_cache does not yet contain). Cheap host-side clone of u32s,
        // trivial next to a GPU forward. `ngram == 0` disables it.
        let ngram = self.ngram_draft;
        let mut work: Vec<u32> = Vec::new();
        if ngram > 0 {
            work.reserve(self.slot_cache[phys].len() + depth);
            work.extend_from_slice(&self.slot_cache[phys]);
            work.push(committed_tok);
        }

        // ---- Draft chain (depth-1 drafts). cur_hidden starts at h_prev; chains via MTP outputs. ----
        let step_t0 = std::time::Instant::now();
        // PLAN/25 Phase 0: coverage capture (dump-only) — the drafter's top-8 per position is
        // read back AFTER the argmax (the draft itself still comes from `argmax_hidden`; the
        // readback never picks a draft). Zero effect when the step dump is off.
        let cov_on = self.cov_trace;
        let mut cov_draft_topk: Vec<Vec<(u32, f32)>> = Vec::new();
        self.gpu.copy_hidden_col(cur_ptr, &self.mtp_h_prev[phys], 0);
        let mut drafts: Vec<u32> = Vec::with_capacity(depth - 1);
        let mut cur_tok = committed_tok as i32;
        let mut dpos = mtp_pos;
        for _ in 0..depth - 1 {
            let m = self.gpu.mtp_draft_step(
                &mut self.pool, self.mtp_cur_hidden.as_ref().unwrap(), cur_tok, dpos,
                mtp_kc_ptr, mtp_vc_ptr, kv_stride);
            self.gpu.copy_hidden_col(cur_ptr, &m, 0);
            self.pool.release_bf16(m, h);
            cur_tok = self.gpu.argmax_hidden(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap()) as i32;
            if cov_on {
                cov_draft_topk.push(
                    self.gpu.topk_hidden_kv(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap(), 8));
            }
            // PROMPT-LOOKUP OVERRIDE (see mtp_lane_step's twin logic in gpu.rs::bench_accept). If the
            // last `ngram` tokens recur earlier in this lane's context, propose the token that followed
            // the most recent earlier occurrence — a free, exact copy that the 1-layer head cannot do.
            // Lossless by construction: the verify checks every draft; a wrong override is just rejected.
            if ngram > 0 && work.len() >= ngram {
                let tail_start = work.len() - ngram;
                for j in (0..tail_start).rev() {
                    if work[j..j + ngram] == work[tail_start..] {
                        if j + ngram < work.len() { cur_tok = work[j + ngram] as i32; }
                        break;
                    }
                }
                work.push(cur_tok as u32);
            }
            drafts.push(cur_tok as u32);
            dpos += 1;
        }

        // ---- Verify [committed_tok, drafts...] on the main model at positions main_pos.. ----
        let round_ms = step_t0.elapsed().as_secs_f32() * 1e3;
        let mut verify_input = vec![committed_tok];
        verify_input.extend(drafts.iter().copied());
        // Build the penalty for the verify: all `depth` positions share the lane's committed-history
        // penalty (the same one the normal decode path applies). This keeps greedy MTP lanes free of
        // repetition without slowing them. htod on the NULL stream, then sync before the compute-side
        // verify reads it.
        let penalty = self.make_penalty_drafts(i, &drafts, rep_pen, presence_pen, freq_pen, has_penalty);
        // Ping-pong GDN: the verify snapshots S1 (post committed-token state) into the snapshot slot
        // via the kernel checkpoint, so a rejected draft restores S1 with a dtod copy — no reverify.
        let verify_t0 = std::time::Instant::now();
        // PLAN/25 Phase 0 (cov_on): keep the verify logits and derive preds + the per-column
        // target top-4/p1/top1−top2 margin host-side (the bench_accept readback). The eager core
        // is the exact sequence the E13 graph captures, so the values — and therefore preds —
        // are identical to the non-dump path. Dump-only.
        // Penalty-aware stand-down: a penalized request's preds must come from the penalized
        // verify, so the coverage capture quietly skips that step (empty tgt fields in its
        // record). `has_penalty` is a per-request property, identical on both ranks — SPMD-safe.
        // ---- W2 (Phase 13): JSON-schema mask chain over the verify columns ----
        // The mask is the decode path's per-state vocabulary mask, extended along the draft chain:
        // position 0 carries the lane's CURRENT FSM state, position j+1 the state reached by the
        // drafts the earlier positions would have to accept. `schema_valid` is the length of that
        // valid draft prefix — the host clamps the accepted prefix to it, so a draft that breaks
        // the chain can never be committed, and the bonus column always comes from a masked,
        // correct state.
        let schema_valid: Option<usize> = match lane_schema.clone() {
            Some(sm) => {
                let words = crate::gpu::mask_words_for(self.gpu.cfg().vocab_size);
                let npos = verify_input.len();
                let mut masks: Vec<Option<std::sync::Arc<Vec<u32>>>> = Vec::with_capacity(npos);
                let mut st = lane_schema_state;
                let mut valid = 0usize;
                let dead = std::sync::Arc::new(vec![0u32; words]);
                for j in 0..npos {
                    if valid == j {
                        masks.push(Some(sm.allowed(st)));
                        if j < drafts.len() {
                            if let Some(ns) = sm.step(st, drafts[j]) { st = ns; valid = j + 1; }
                        }
                    } else {
                        masks.push(Some(dead.clone()));   // past the break: never emitted (nacc clamped)
                    }
                }
                self.gpu.set_verify_mask(&masks);
                Some(valid)
            }
            None => None,
        };
        let (preds, vout, cov_tgt) = if cov_on {
            let (logits, vout) = self.gpu.verify_forward_keep_logits(
                &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos,
                Some(snapshot));
            let n = verify_input.len();
            let vocab = self.gpu.cfg().vocab_size;
            let lg_f32: Vec<f32> = match logits {
                crate::gpu::VerifyLogits::B16(lg) => {
                    let v: Vec<half::bf16> = self.gpu.dev().dtoh_sync_copy(&lg).unwrap();
                    self.pool.release_bf16(lg, vocab * n);
                    v.iter().map(|x| x.to_f32()).collect()
                }
                crate::gpu::VerifyLogits::F32(lg) => {
                    let v: Vec<f32> = self.gpu.dev().dtoh_sync_copy(&lg).unwrap();
                    self.pool.release(lg, vocab * n);
                    v
                }
            };
            let mut preds = Vec::with_capacity(n);
            let (mut tids, mut tlog) = (Vec::with_capacity(n * 4), Vec::with_capacity(n * 4));
            let (mut tp1, mut tmar) = (Vec::with_capacity(n), Vec::with_capacity(n));
            for c in 0..n {
                let (ids, vals, p1, margin) = crate::dflash2::stepdump::tgt_topk_host(
                    &lg_f32[c * vocab..(c + 1) * vocab], 4);
                preds.push(ids[0]);
                tids.extend_from_slice(&ids); tlog.extend_from_slice(&vals);
                tp1.push(p1); tmar.push(margin);
            }
            (preds, vout, Some((tids, tlog, tp1, tmar)))
        } else {
            let (preds, vout) = self.gpu.verify_forward(
                &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos,
                Some(snapshot), penalty);
            (preds, vout, None)
        };
        let verify_ms = verify_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Accept longest prefix (greedy: drafts[i] accepted iff preds[i]==drafts[i]). ----
        let mut nacc = 0usize;
        while nacc < drafts.len() && preds[nacc] == drafts[nacc] { nacc += 1; }
        if let Some(v) = schema_valid {
            // A draft beyond the schema-valid prefix is never committed (its verify column had no
            // legal continuation), and the mask is disarmed for the next (unmasked) caller.
            nacc = nacc.min(v);
            self.gpu.clear_verify_mask();
        }
        let bonus = preds[nacc];

        // ---- GDN rollback on partial reject: restore S1 (the checkpoint — no second forward). ----
        // vout column nacc is the hidden at the last accepted position, valid in both cases.
        if nacc + 1 != depth {
            // Restore the state as of the LAST ACCEPTED column, not column 0. Checkpoint slots are
            // contiguous: slot (snapshot + t) holds the post-state of verify column t.
            self.gpu.copy_gdn_slot(&self.state, snapshot + nacc, phys);
        }

        // h_prev = hidden at the last accepted position (vout column nacc).
        self.gpu.copy_hidden_col(*self.mtp_h_prev[phys].device_ptr(), &vout, nacc);

        // ---- Re-prime MTP over the accepted prefix with REAL hiddens (vout), in ONE forward. ----
        // Column k=0 uses h_save (the pre-verify hidden); k>=1 uses vout column k-1. Batching all
        // nacc+1 columns into a single MTP-layer forward reads the layer's weights once instead of
        // once per accepted position — the causal-append attention makes column k see the KV that
        // columns < k just wrote, exactly as in the multi-token verify.
        let n_rp = nacc + 1;
        let rp_hidden = self.pool.get_bf16(h * n_rp);
        let rp_ptr = *rp_hidden.device_ptr();
        let mut rp_toks: Vec<u32> = Vec::with_capacity(n_rp);
        self.gpu.copy_hidden_col(rp_ptr, self.mtp_h_save.as_ref().unwrap(), 0);
        rp_toks.push(committed_tok);
        // vout columns 0..nacc-1 are CONTIGUOUS — one dtod for the whole accepted prefix.
        self.gpu.copy_hidden_cols(rp_ptr + (h * 2) as u64, &vout, 0, nacc);
        for k in 1..=nacc {
            rp_toks.push(drafts[k - 1]);
        }
        self.gpu.mtp_reprime(&mut self.pool, &rp_hidden, &rp_toks, main_pos - 1,
                             mtp_kc_ptr, mtp_vc_ptr, kv_stride);
        self.pool.release_bf16(rp_hidden, h * n_rp);
        self.pool.release_bf16(vout, h * depth);

        // ---- Emit accepted drafts + bonus, honoring EOS and max_new. ----
        //
        // The BUDGET CHECK BELONGS ON THE DRAFTS TOO. It used to guard only the bonus, so a step that
        // accepted k drafts emitted all k regardless: a request for exactly max_tokens got back up to
        // max_tokens + depth - 1. Harmless-looking, and it is why a greedy MTP response and a greedy
        // plain response to the same prompt differed -- not in the tokens, which were identical, but
        // in how many of them came back. That cost real time to chase, because "MTP output != plain
        // output" reads as a losslessness failure when it is really an off-by-one in the stop rule.
        let mut new_toks: Vec<u32> = Vec::with_capacity(nacc + 1);
        let mut hit_eos = false;
        for &d in drafts.iter().take(nacc) {
            if generated + new_toks.len() >= max_new { break; }
            new_toks.push(d);
            if eos.contains(&d) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; break; }
        }
        // Bonus (the greedy next token after the last accepted position) — always progress.
        if !hit_eos && generated + new_toks.len() < max_new {
            new_toks.push(bonus);
            if eos.contains(&bonus) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; }
        }

        let emit_count = new_toks.len();
        // W2: advance the JSON-schema FSM over the tokens ACTUALLY emitted (accepted drafts +
        // bonus). A completed document ends the turn - generation must not run past the value.
        let schema_done = self.schema_advance(i, &new_toks);
        let finished = hit_eos || generated + emit_count >= max_new || schema_done;

        // The verify FED [committed_tok] ++ drafts[..nacc] through the model, and the GDN state was
        // rolled back to exactly the last accepted column — so that, and only that, is what the slot's
        // state has consumed. The bonus token was PREDICTED, not fed. Cache what was consumed.
        // (If EOS truncated the emit, the cache still holds every fed token: it is then a longer
        // sequence than the client saw, so the next turn simply matches a shorter prefix. Correct,
        // just less reuse — which is the right way for this to fail.)
        {
            let cache = &mut self.slot_cache[phys];
            cache.push(committed_tok);
            cache.extend_from_slice(&drafts[..nacc]);
        }

        // Apply lane state. last_tok/pos/mtp_pos only matter if the lane continues, but keep them
        // consistent regardless. On full emit (not finished), advance the MTP cursor as in bench_mtp.
        {
            let lane = self.lanes[i].as_mut().unwrap();
            for &t in &new_toks {
                let _ = lane.tx.send(TokEvent::Tok(t));
                lane.hist_push(t);
                lane.hist_trim();
            }
            lane.generated += emit_count;
            // If the lane continues, it emitted all nacc drafts + bonus (emit_count == nacc+1).
            if !finished {
                lane.last_tok = bonus;
                lane.pos = main_pos + nacc + 1;
                lane.mtp_pos = main_pos + nacc;
            } else {
                lane.last_tok = *new_toks.last().unwrap_or(&committed_tok);
                lane.pos = main_pos + emit_count;
            }
        }

        // ---- S-B16 lossless auditor (--mtp-lossless-audit=1, TP=1): lockstep shadow decode ----
        // bench_mtp_steps' slot-1 ground truth, moved into the serving path: one plain decode
        // forward per emitted token, ALWAYS advancing on the ground-truth token (never on the MTP
        // stream), so the first divergence between the serving lane and plain decode is pinned to
        // an exact (request, emitted-index, position). A lane that later takes plain (non-MTP)
        // steps leaves its cursor behind — the audit for that request simply ends there.
        if self.audit_on && emit_count > 0 && self.audit_cur[i].is_some() {
            let need_prime = self.audit_cur[i].as_ref().map_or(false, |c| !c.primed);
            if need_prime {
                let (gt_last, gt_pos) = {
                    let c = self.audit_cur[i].as_ref().unwrap();
                    (c.gt_last, c.gt_pos)
                };
                self.gpu.zero_slot_state(&mut self.state, self.audit_slot, kv_stride);
                // The shadow slot has no prefill: zero its KV extent too, or the prime decode
                // attends over [0, pos) rows nothing wrote (garbage softmax → false divergence).
                self.gpu.zero_slot_kv(&mut self.state, self.audit_slot, kv_stride);
                let pred = self.audit_shadow_decode(gt_last, gt_pos, kv_stride);
                {
                    let c = self.audit_cur[i].as_mut().unwrap();
                    c.pred = pred;
                    c.primed = true;
                }
            }
            for &e in &new_toks {
                let (pred, gpos, tag, idx, already) = {
                    let c = self.audit_cur[i].as_ref().unwrap();
                    (c.pred, c.gt_pos, c.tag, c.idx, c.diverged)
                };
                if e != pred && !already {
                    eprintln!("[lossless-audit] req={} FIRST DIVERGENCE: emitted#{} pos={} mtp={} seq={} (depth {} nacc {} step {})",
                              tag, idx, gpos + 1, e, pred, depth, nacc, self.mtp_stat_steps);
                    self.audit_cur[i].as_mut().unwrap().diverged = true;
                }
                // Advance the shadow on the GROUND-TRUTH token (pred), never on the MTP stream,
                // mirroring bench_mtp_steps' slot-1 lockstep exactly.
                let new_pred = self.audit_shadow_decode(pred, gpos + 1, kv_stride);
                {
                    let c = self.audit_cur[i].as_mut().unwrap();
                    c.gt_last = pred;
                    c.gt_pos = gpos + 1;
                    c.pred = new_pred;
                    c.idx += 1;
                }
            }
        }
        // (committed_tok/main_pos/mtp_pos are all read above; no further bookkeeping needed.)

        // ---- Telemetry: accumulate per-step MTP stats and log a summary every 50 lane-steps. ----
        // Ping-pong GDN: every step is exactly ONE main-model verify forward (no reverify on reject).
        self.mtp_stat_steps += 1;
        self.mtp_stat_drafts += drafts.len() as u64;
        self.mtp_stat_accepted += nacc as u64;
        self.mtp_stat_emitted += emit_count as u64;
        self.mtp_stat_verify_fwds += 1;
        // Feed the auto-policy: it needs tokens-per-step to decide whether MTP is still paying.
        self.mtp.record_step(drafts.len() as u64, nacc as u64, emit_count as u64);
        self.log_draft_step(i, main_pos, committed_tok, &drafts, &preds, nacc);
        self.rec_step(SpecStepRec {
            greedy: true, pos: main_pos as u32, drafts: drafts.len() as u32, nacc: nacc as u32,
            emitted: emit_count as u32, round_ms, verify_ms,
            step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
        });
        // ---- S5F3 MTP control dump (dump-only; the p-computation control). ----
        if let Some(d) = self.step_dump.as_mut() {
            let (mut tids, mut tlog) = (Vec::new(), Vec::new());
            let (mut tp1, mut tmar) = (Vec::new(), Vec::new());
            if let Some((a, b, c, e)) = cov_tgt.as_ref() {
                tids = a.clone(); tlog = b.clone(); tp1 = c.clone(); tmar = e.clone();
            }
            let mut dk_ids = Vec::with_capacity(cov_draft_topk.len() * 8);
            let mut dk_logit = Vec::with_capacity(cov_draft_topk.len() * 8);
            for tk in &cov_draft_topk {
                for (t, v) in tk { dk_ids.push(*t); dk_logit.push(*v); }
            }
            let rec = crate::dflash2::stepdump::MtpStepRec {
                step: self.mtp_stat_steps, pos: main_pos, committed: committed_tok, depth,
                drafts: drafts.clone(), p_draft: Vec::new(),
                resid: preds[..depth.saturating_sub(1).min(preds.len())].to_vec(),
                bonus, nacc, emitted: emit_count,
                tgt_ids: tids, tgt_logit: tlog, tgt_p1: tp1, tgt_margin: tmar,
                draft_topk_ids: dk_ids, draft_topk_logit: dk_logit,
                tgt_top20: Vec::new(),
            };
            d.record_mtp(&rec);
        }
        if self.mtp_stat_steps % 50 == 0 {
            let acc = if self.mtp_stat_drafts > 0 {
                self.mtp_stat_accepted as f64 / self.mtp_stat_drafts as f64 * 100.0
            } else { 0.0 };
            // Effective speedup ceiling = emitted tokens / verify forwards (how many output tokens
            // we get per main-model forward — 1.0 means no MTP benefit).
            let eff = self.mtp_stat_emitted as f64 / self.mtp_stat_verify_fwds as f64;
            crate::reprintln!("[mtp] steps={} drafts={} accepted={:.1}% emitted={} tok/verify_fwd={:.3} (depth {}) accept@k [{}]",
                      self.mtp_stat_steps, self.mtp_stat_drafts, acc,
                      self.mtp_stat_emitted, eff, depth, fmt_accept_by_depth(&self.mtp));
            // A.2: same numbers to the status route (published where the log line prints).
            crate::tel::publish_mtp(self.mtp_stat_steps, self.mtp_stat_drafts,
                                    self.mtp_stat_accepted, self.mtp_stat_emitted,
                                    self.mtp_stat_verify_fwds, depth, &self.mtp.hazard_counts());
            self.dump_accept_curve();
        }
        finished
    }

    /// S-B16 auditor: one plain decode forward on the audit slot (batch 1), mirroring the SERVE
    /// path's decode flavor — max_pc PINNED to kv_stride, exactly like the captured decode graph
    /// (capture pins max_pc to the run maximum; an eager pos+1 bound selects a different split-K
    /// reduction and flips knife-edge positions — run5's two "divergences" were this artifact:
    /// the seq arm's own text carries the LANE's token at both spots).
    pub(super) fn audit_shadow_decode(&mut self, tok: u32, pos: usize, kv_stride: usize) -> u32 {
        {
            let ab = self.audit_bufs.as_mut().unwrap();
            self.gpu.dev().htod_sync_copy_into(&[tok as i32, 0], &mut ab.tokens_dev).unwrap();
            self.gpu.dev().htod_sync_copy_into(&[pos as i32, 0], &mut ab.pos_dev).unwrap();
        }
        self.gpu.dev().synchronize().unwrap();
        let ab = self.audit_bufs.as_mut().unwrap();
        let next = self.gpu.forward_decode(&mut self.pool, ab, &mut self.state, kv_stride, kv_stride, 1);
        next[0]
    }

    /// Stochastic MTP step for a sampling lane (temperature > 0). Mirrors mtp_lane_step but:
    /// 1. Drafts GREEDILY via argmax_hidden (point-mass proposal q=1 — the rejection step corrects
    ///    the distribution; on hy_v3 the draft argmax is fp32-exact, gpu.rs argmax_hidden)
    /// 2. Verifies via verify_forward_sample (returns p_of_draft + resid_tok + bonus_tok; fp32
    ///    logits end-to-end on hy_v3)
    /// 3. Accepts via speculative rejection sampling: accept draft with prob min(1, p(x)/q(x)),
    ///    else emit a token from the residual (p \ {draft}, renormalized).
    /// 4. Re-primes MTP with REAL verify hiddens for ACCEPTED positions only.
    /// Emits the accepted drafts + replacement/bonus token, advancing the lane.
    /// Returns true if the lane finished (EOS or max_new reached).
    pub(super) fn mtp_lane_step_sample(&mut self, i: usize) -> bool {
        crate::tel::note_step();   // A.2: per-step interval for the status route (lock-free)
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let depth = self.mtp.depth();
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let snapshot = self.mtp_snapshot_slot;
        let kv_stride = self.kv_stride;
        let mtp_kc_ptr = *self.mtp_kc[phys].device_ptr();
        let mtp_vc_ptr = *self.mtp_vc[phys].device_ptr();
        let h_save_ptr = *self.mtp_h_save.as_ref().unwrap().device_ptr();
        let cur_ptr = *self.mtp_cur_hidden.as_ref().unwrap().device_ptr();

        // Snapshot lane state into locals.
        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
        let mtp_pos = self.lanes[i].as_ref().unwrap().mtp_pos;
        let generated = self.lanes[i].as_ref().unwrap().generated;
        let max_new = self.lanes[i].as_ref().unwrap().max_new;
        let eos = self.eos.clone();
        // Phase-8: the stop rule is a property of the REQUEST, not the serving path —
        // every emit loop must honor ignore_eos/min_new exactly like the batched decode path
        // (batch.rs' `eos_hit`), or spec-on and spec-off streams stop at different tokens.
        let (ignore_eos, min_new) = { let l = self.lanes[i].as_ref().unwrap(); (l.ignore_eos, l.min_new) };
        let temperature = self.lanes[i].as_ref().unwrap().temperature;
        let top_k = self.lanes[i].as_ref().unwrap().top_k;
        let top_p = self.lanes[i].as_ref().unwrap().top_p;
        let (rep_pen, presence_pen, freq_pen, has_penalty) = {
            let l = self.lanes[i].as_ref().unwrap();
            (l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.has_penalty())
        };
        let history: Vec<u32> = self.lanes[i].as_ref().unwrap().history.clone();
        // All RNG for this step derives from one key (device column seeds + host accept draws,
        // domain-separated); the lane key advances exactly once per step.
        let step_key = self.lanes[i].as_ref().unwrap().seed;

        // h_save = h_prev (hidden at main_pos-1); saved for the post-accept re-prime step k=0.
        self.gpu.copy_hidden_col(h_save_ptr, &self.mtp_h_prev[phys], 0);

        // ---- Draft chain (depth-1 drafts) via greedy argmax from the MTP head. ----
        // Greedy drafting puts the draft token in the target model's high-probability region,
        // dramatically improving acceptance vs. sampling from the weak 1-layer MTP head.
        // Rejection sampling still corrects the output distribution to match non-MTP sampling.
        let step_t0 = std::time::Instant::now();
        self.gpu.copy_hidden_col(cur_ptr, &self.mtp_h_prev[phys], 0);
        let mut drafts: Vec<u32> = Vec::with_capacity(depth - 1);
        let mut qprobs: Vec<f32> = Vec::with_capacity(depth - 1);
        let mut cur_tok = committed_tok as i32;
        let mut dpos = mtp_pos;
        // PLAN/25 Phase 0: coverage capture (dump-only) — drafter top-8 readback per position,
        // after the argmax (the readback never picks the draft).
        let cov_on = self.cov_trace;
        let mut cov_draft_topk: Vec<Vec<(u32, f32)>> = Vec::new();
        for _ in 0..depth - 1 {
            let m = self.gpu.mtp_draft_step(
                &mut self.pool, self.mtp_cur_hidden.as_ref().unwrap(), cur_tok, dpos,
                mtp_kc_ptr, mtp_vc_ptr, kv_stride);
            self.gpu.copy_hidden_col(cur_ptr, &m, 0);
            self.pool.release_bf16(m, h);
            let tok = self.gpu.argmax_hidden(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap());
            cur_tok = tok as i32;
            if cov_on {
                cov_draft_topk.push(
                    self.gpu.topk_hidden_kv(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap(), 8));
            }
            drafts.push(tok);
            qprobs.push(1.0); // greedy draft = point mass
            dpos += 1;
        }
        let round_ms = step_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Build verify penalty (same as greedy). ----
        let verify_penalty = self.make_penalty_drafts(i, &drafts, rep_pen, presence_pen, freq_pen, has_penalty);

        // ---- Build verify input + seeds for spec_verify_b. ----
        let mut verify_input = vec![committed_tok];
        verify_input.extend(drafts.iter().copied());
        // Per-column device seeds, domain-separated from the host accept draws below.
        let verify_seeds: Vec<u32> =
            (0..depth).map(|j| rng_u32(step_key, RNG_DOM_VERIFY, j)).collect();

        // ---- Verify with stochastic output. ----
        let verify_t0 = std::time::Instant::now();
        // PLAN/25 Phase 0: the dump-only top-20 table rides out on `cov_t20` when the step
        // dump is armed (verify_forward_sample's existing t20_out hook).
        let mut cov_t20: Vec<u64> = Vec::new();
        let (vsample, vout) = self.gpu.verify_forward_sample(
            &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos,
            Some(snapshot), verify_penalty,
            &drafts, &qprobs, temperature, top_k, top_p, &verify_seeds,
            if cov_on { Some(&mut cov_t20) } else { None });
        let verify_ms = verify_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Speculative rejection sampling accept loop (Leviathan et al. 2023). ----
        // For each draft position j: accept with prob min(1, p_j(x_j) / q_j(x_j)).
        // On reject: emit resid_tok[j] and stop. If all accepted: emit bonus_tok.
        let mut nacc = 0usize;
        let mut emitted: Vec<u32> = Vec::with_capacity(depth);
        let mut rejected = false;
        let eps = 1e-12f32;
        for j in 0..drafts.len() {
            let ratio = if qprobs[j] < eps { 1.0 } else { (vsample.p_of_draft[j] / qprobs[j]).min(1.0) };
            let ru = rng_uniform(step_key, RNG_DOM_ACCEPT, j);
            if ru < ratio {
                emitted.push(drafts[j]);
                nacc += 1;
            } else {
                emitted.push(vsample.resid_tok[j]);
                rejected = true;
                break;
            }
        }
        if !rejected {
            emitted.push(vsample.bonus_tok);
        }

        // ---- GDN rollback on partial reject. ----
        if nacc + 1 != depth {
            // Restore the state as of the LAST ACCEPTED column, not column 0. Checkpoint slots are
            // contiguous: slot (snapshot + t) holds the post-state of verify column t.
            self.gpu.copy_gdn_slot(&self.state, snapshot + nacc, phys);
        }

        // ---- h_prev = hidden at the last accepted position (vout column nacc). ----
        self.gpu.copy_hidden_col(*self.mtp_h_prev[phys].device_ptr(), &vout, nacc);

        // ---- Re-prime MTP with the REAL verify hiddens for the accepted positions, in ONE forward.
        // Only ACCEPTED positions are re-primed: `drafts[k-1]` for k<=nacc are by definition the
        // accepted drafts, so they are exactly the tokens that were emitted there. On a rejection the
        // replacement token at index nacc is NOT re-primed here — it becomes the next step's
        // committed token and is primed then.
        let n_rp = nacc + 1;
        let rp_hidden = self.pool.get_bf16(h * n_rp);
        let rp_ptr = *rp_hidden.device_ptr();
        let mut rp_toks: Vec<u32> = Vec::with_capacity(n_rp);
        self.gpu.copy_hidden_col(rp_ptr, self.mtp_h_save.as_ref().unwrap(), 0);
        rp_toks.push(committed_tok);
        // vout columns 0..nacc-1 are CONTIGUOUS — one dtod for the whole accepted prefix.
        self.gpu.copy_hidden_cols(rp_ptr + (h * 2) as u64, &vout, 0, nacc);
        for k in 1..=nacc {
            rp_toks.push(drafts[k - 1]);
        }
        self.gpu.mtp_reprime(&mut self.pool, &rp_hidden, &rp_toks, main_pos - 1,
                             mtp_kc_ptr, mtp_vc_ptr, kv_stride);
        self.pool.release_bf16(rp_hidden, h * n_rp);
        self.pool.release_bf16(vout, h * depth);

        // ---- Emit accepted tokens + replacement/bonus, honoring EOS and max_new. ----
        let mut hit_eos = false;
        let mut to_emit: Vec<u32> = Vec::with_capacity(emitted.len());
        for &t in &emitted {
            to_emit.push(t);
            if eos.contains(&t) && !ignore_eos && generated + to_emit.len() >= min_new { hit_eos = true; break; }
            if generated + to_emit.len() >= max_new { break; }
        }
        let emit_count = to_emit.len();
        let finished = hit_eos || generated + emit_count >= max_new;

        // Same as the greedy path: the verify FED [committed_tok] ++ drafts[..nacc], and the GDN state
        // was rolled back to the last accepted column. On a rejection the REPLACEMENT token emitted at
        // index nacc was never fed — it becomes the next step's committed token and is fed then. So the
        // slot's state has consumed exactly this, and nothing more.
        {
            let cache = &mut self.slot_cache[phys];
            cache.push(committed_tok);
            cache.extend_from_slice(&drafts[..nacc]);
        }

        // Apply lane state.
        {
            let lane = self.lanes[i].as_mut().unwrap();
            for &t in &to_emit {
                let _ = lane.tx.send(TokEvent::Tok(t));
                lane.hist_push(t);
                lane.hist_trim();
            }
            lane.generated += emit_count;
            lane.seed = splitmix64(step_key);
            if !finished {
                lane.last_tok = *to_emit.last().unwrap_or(&committed_tok);
                lane.pos = main_pos + nacc + 1;
                lane.mtp_pos = main_pos + nacc;
            } else {
                lane.last_tok = *to_emit.last().unwrap_or(&committed_tok);
                lane.pos = main_pos + emit_count;
            }
        }

        // ---- Telemetry. ----
        self.mtp_stat_steps += 1;
        self.mtp_stat_drafts += drafts.len() as u64;
        self.mtp_stat_accepted += nacc as u64;
        self.mtp_stat_emitted += emit_count as u64;
        self.mtp_stat_verify_fwds += 1;
        // Feed the auto-policy: it needs tokens-per-step to decide whether MTP is still paying.
        self.mtp.record_step(drafts.len() as u64, nacc as u64, emit_count as u64);
        // preds omitted: the stochastic verify SAMPLES rather than taking an argmax, so there is no
        // greedy per-column prediction to record (the parity gate uses the greedy path anyway).
        self.log_draft_step(i, main_pos, committed_tok, &drafts, &[], nacc);
        self.rec_step(SpecStepRec {
            greedy: false, pos: main_pos as u32, drafts: drafts.len() as u32, nacc: nacc as u32,
            emitted: emit_count as u32, round_ms, verify_ms,
            step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
        });
        // ---- S5F3 MTP control dump (dump-only; p_of_draft + residual + bonus — the
        // ---- p-computation cross-check against the DFlash2 lane's verify). ----
        if let Some(d) = self.step_dump.as_mut() {
            let mut dk_ids = Vec::with_capacity(cov_draft_topk.len() * 8);
            let mut dk_logit = Vec::with_capacity(cov_draft_topk.len() * 8);
            for tk in &cov_draft_topk {
                for (t, v) in tk { dk_ids.push(*t); dk_logit.push(*v); }
            }
            let rec = crate::dflash2::stepdump::MtpStepRec {
                step: self.mtp_stat_steps, pos: main_pos, committed: committed_tok, depth,
                drafts: drafts.clone(), p_draft: vsample.p_of_draft.clone(),
                resid: vsample.resid_tok.clone(), bonus: vsample.bonus_tok,
                nacc, emitted: emit_count,
                tgt_ids: Vec::new(), tgt_logit: Vec::new(), tgt_p1: Vec::new(),
                tgt_margin: Vec::new(),
                draft_topk_ids: dk_ids, draft_topk_logit: dk_logit,
                tgt_top20: cov_t20.clone(),
            };
            d.record_mtp(&rec);
        }
        if self.mtp_stat_steps % 50 == 0 {
            let acc = if self.mtp_stat_drafts > 0 {
                self.mtp_stat_accepted as f64 / self.mtp_stat_drafts as f64 * 100.0
            } else { 0.0 };
            let eff = self.mtp_stat_emitted as f64 / self.mtp_stat_verify_fwds as f64;
            crate::reprintln!("[mtp] steps={} drafts={} accepted={:.1}% emitted={} tok/verify_fwd={:.3} (depth {}) accept@k [{}]",
                      self.mtp_stat_steps, self.mtp_stat_drafts, acc,
                      self.mtp_stat_emitted, eff, depth, fmt_accept_by_depth(&self.mtp));
            // A.2: same numbers to the status route (published where the log line prints).
            crate::tel::publish_mtp(self.mtp_stat_steps, self.mtp_stat_drafts,
                                    self.mtp_stat_accepted, self.mtp_stat_emitted,
                                    self.mtp_stat_verify_fwds, depth, &self.mtp.hazard_counts());
            self.dump_accept_curve();
        }
        finished
    }

    // ---- S5F: the DFlash2 speculation lane (b==1 only; the S4F integrated round) --------------
}
