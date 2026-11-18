//! Copy-span speculation arm (`--copy-span`) plus the draft-step logging and accept-curve diagnostics.
//! Another `impl BatchScheduler` block (file map in `mod.rs`); the methods keep their exact text.
//! Binding invariants: greedy MTP stays bitwise lossless; spec gating is per source (AGENTS.md section 4).

use super::*;

impl BatchScheduler {
    /// Append one MTP step to the option-gated draft log (`--mtp-draft-log`), if open. JSONL, one object
    /// per line. `preds` is the verify's per-column argmax (greedy paths); pass `&[]` where the verify
    /// samples instead. Costs nothing when the log is closed. This is the reference the head-finetune
    /// B0 parity gate diffs the HF MTP module against, so it records exactly what was fed and predicted.
    pub(super) fn log_draft_step(&mut self, lane: usize, pos: usize, committed: u32, drafts: &[u32], preds: &[u32], nacc: usize) {
        use std::io::Write;
        let Some(f) = self.mtp_draft_log.as_mut() else { return; };
        let arr = |xs: &[u32]| -> String {
            let mut s = String::from("[");
            for (k, x) in xs.iter().enumerate() { if k > 0 { s.push(','); } s.push_str(&x.to_string()); }
            s.push(']'); s
        };
        let _ = writeln!(f,
            "{{\"step\":{},\"lane\":{},\"pos\":{},\"committed\":{},\"drafts\":{},\"preds\":{},\"nacc\":{}}}",
            self.mtp_stat_steps, lane, pos, committed, arr(drafts), arr(preds), nacc);
    }

    /// Overwrite the accept-by-depth curve file (`--mtp-curve-file`), if set, with the cumulative
    /// per-position conditional acceptance — the runbook §0 baseline curve. Called on the periodic
    /// stats boundary so a running server keeps a fresh snapshot on disk.
    pub(super) fn dump_accept_curve(&self) {
        let Some(path) = self.mtp_curve_path.as_ref() else { return; };
        let hz = self.mtp.hazard_counts();
        let mut s = String::from("{\"depth\":");
        s.push_str(&self.mtp.depth().to_string());
        s.push_str(",\"steps\":");
        s.push_str(&self.mtp_stat_steps.to_string());
        s.push_str(",\"accept_by_depth\":[");
        // accept_by_depth[i] = {pos:i+1, accepted, offered, rate, cond_chain}. cond_chain = product of
        // rates up to and including pos i+1 = P(a draft chain reaches depth i+1) — the yield driver.
        let mut chain = 1.0f64;
        for (i, &(a, n)) in hz.iter().enumerate() {
            let rate = if n > 0 { a as f64 / n as f64 } else { 0.0 };
            chain *= rate;
            if i > 0 { s.push(','); }
            s.push_str(&format!(
                "{{\"pos\":{},\"accepted\":{},\"offered\":{},\"rate\":{:.4},\"cond_chain\":{:.4}}}",
                i + 1, a, n, rate, chain));
        }
        s.push_str("]}");
        let _ = std::fs::write(path, s);
    }

    /// COPY27B: bring lane `i`'s copy proposer up to the committed sequence (`slot_cache[phys]` =
    /// every token the state has consumed, then `last_tok` = the committed token the next round
    /// feeds) and ask it for a proposal sized to this round. The proposer is append-only: a copy
    /// round commits its own outcome right away, every other step's tokens are appended here
    /// (one "round without a proposal"). A sequence that no longer extends the proposer's context
    /// (cannot happen through the serve loop; defensive) rebuilds it from scratch.
    pub(super) fn copy_propose(&mut self, i: usize, mode: CopyMode) -> Option<crate::copy_span::Proposal> {
        let t0 = std::time::Instant::now();
        let arm = self.copy_arm?;
        let kv_stride = self.kv_stride;
        let df2_room = if mode == CopyMode::Df2 {
            self.df2.as_ref().map_or(usize::MAX, |d| d.max_c().saturating_sub(d.nprev()))
        } else { usize::MAX };
        let BatchScheduler { lanes, slot_cache, .. } = self;
        let lane = lanes[i].as_mut().unwrap();
        let cache = &slot_cache[lane.phys];
        let last = lane.last_tok;
        let slen = cache.len() + 1;
        let at = |k: usize| if k < cache.len() { cache[k] } else { last };
        let fresh = match lane.copy.as_ref() {
            None => true,
            Some(cs) => { let l = cs.ctx_len(); l == 0 || l > slen || cs.ctx()[l - 1] != at(l - 1) }
        };
        let before;
        if fresh {
            let mut seq = Vec::with_capacity(slen + 64);
            seq.extend_from_slice(cache);
            seq.push(last);
            lane.copy = Some(Box::new(crate::copy_span::CopySpan::new(&seq, arm.cfg)));
            before = *lane.copy.as_ref().unwrap().counters();
        } else {
            let cs = lane.copy.as_mut().unwrap();
            before = *cs.counters();
            let l = cs.ctx_len();
            if l < slen {
                let tail: Vec<u32> = (l..slen).map(at).collect();
                // the drafter's round(s) since this lane's last proposal (the yield gate's neural sample)
                cs.note_neural_round(tail.len());
                cs.commit(None, 0, &tail);
            }
        }
        // width: the arm's cap, the request's budget (a round emits up to w + 1), the KV stride and
        // the DFlash2 ring's room (a copy round writes w + 1 rows from `pos`)
        let budget = lane.max_new.saturating_sub(lane.generated);
        let w = arm.max_w
            .min(budget.saturating_sub(1))
            .min(kv_stride.saturating_sub(lane.pos + 1))
            .min(df2_room.saturating_sub(1));
        let cs = lane.copy.as_mut().unwrap();
        let p = cs.propose(w);
        crate::exl3_serve::copy_span_count_delta(&before, cs.counters());
        crate::exl3_serve::copy_span_add_host_us(t0.elapsed().as_micros() as u64);
        p
    }

    /// COPY27B under TP: every rank computed its proposal from lockstep state (the prompt shipped
    /// at admit, every agreed token), so they are equal by construction; this exchange makes that
    /// a checked fact BEFORE anything of the round launches (a copy round and a drafter round run
    /// different collectives). Every rank sends (width, digest of the copied ids and source); all
    /// equal = the copy round runs (counted in `velogb10_copy_span_tp_agreed_total`); any
    /// difference is seen by EVERY rank alike, so every rank runs the drafter round instead
    /// (counted as `skipped_rounds_total{reason="tp_disagree"}`) and the lockstep holds.
    pub(super) fn copy_tp_agree(p: Option<crate::copy_span::Proposal>) -> Option<crate::copy_span::Proposal> {
        let (w, h) = match &p {
            None => (0u32, 0u64),
            Some(p) => {
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                for &t in p.tokens.iter().chain(std::iter::once(&(p.src as u32))) {
                    h ^= t as u64;
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
                (p.tokens.len() as u32, h)
            }
        };
        let mine = [0xC0B2_7001u32, w, h as u32, (h >> 32) as u32];
        match crate::net::exchange_u32s_all(&mine, 4) {
            Ok(all) if all.iter().all(|f| f.len() >= 4 && f[..4] == mine) => {
                if p.is_some() { crate::exl3_serve::copy_span_count_tp_agreed(); }
                p
            }
            Ok(all) => {
                crate::reprintln!("[serve] copy-span: TP ranks proposed different copy rounds ({:?}); every rank runs the \
                                   drafter round instead (velogb10_copy_span_skipped_rounds_total{{reason=\"tp_disagree\"}})",
                                  all.iter().map(|f| f.get(1).copied().unwrap_or(0)).collect::<Vec<_>>());
                crate::exl3_serve::copy_span_count_skip(crate::exl3_serve::CopySkip::TpDisagree);
                None
            }
            Err(e) => {
                // the link is gone: the step's own collectives and the per-step agree fail loudly next
                crate::reprintln!("[serve] copy-span: TP copy agreement failed ({e:#}); no copy round this step");
                None
            }
        }
    }

    /// COPY27B: one COPY ROUND for lone lane `i`: verify `[last_tok, copied...]` on the trunk exactly
    /// like the replaced drafter round (greedy: longest argmax prefix + the model's token; sampled:
    /// the q = 1 rejection rule of the point-mass drafts, `p(x)` accept + residual draw), GDN rollback
    /// to the last accepted column, then keep the replaced drafter's state live: the MTP head is
    /// re-primed over the accepted span (`CopyMode::Mtp`), or the accepted span's taps are injected
    /// into the DFlash2 ring (`CopyMode::Df2`; a verify wider than the block reads them from the wide
    /// sink, injected in block-sized pieces). Copy rounds feed neither drafter's statistics nor the
    /// MTP depth policy; they count in `velogb10_copy_span_*` and the speculative-round totals.
    pub(super) fn copy_lane_step(&mut self, i: usize, mode: CopyMode, prop: &crate::copy_span::Proposal) -> bool {
        crate::tel::note_step();
        let step_t0 = std::time::Instant::now();
        let h = self.gpu.mtp_hidden_width();
        let snapshot = self.mtp_snapshot_slot;
        let kv_stride = self.kv_stride;
        let (phys, committed_tok, main_pos, generated, max_new, ignore_eos, min_new, greedy,
             temperature, top_k, top_p, rep_pen, presence_pen, freq_pen, has_penalty, step_key) = {
            let l = self.lanes[i].as_ref().unwrap();
            (l.phys, l.last_tok, l.pos, l.generated, l.max_new, l.ignore_eos, l.min_new, l.greedy,
             l.temperature, l.top_k, l.top_p, l.rep_penalty, l.presence_penalty, l.frequency_penalty,
             l.has_penalty(), l.seed)
        };
        let eos = self.eos.clone();
        let drafts: Vec<u32> = prop.tokens.clone();
        let n = drafts.len() + 1;
        assert!(n >= 2 && n <= crate::gpu::MAX_VERIFY, "copy round width {n} outside 2..={}", crate::gpu::MAX_VERIFY);
        let (mtp_kc_ptr, mtp_vc_ptr) = (*self.mtp_kc[phys].device_ptr(), *self.mtp_vc[phys].device_ptr());
        if mode == CopyMode::Mtp {
            // h_save = h_prev (hidden at main_pos-1): column 0 of the re-prime below
            let h_save_ptr = *self.mtp_h_save.as_ref().unwrap().device_ptr();
            self.gpu.copy_hidden_col(h_save_ptr, &self.mtp_h_prev[phys], 0);
        } else if mode == CopyMode::Df2 {
            let df2 = self.df2.as_ref().unwrap();
            assert_eq!(df2.nprev(), main_pos, "copy round: df2 ring nprev {} != lane pos {main_pos}", df2.nprev());
        }
        let mut verify_input = Vec::with_capacity(n);
        verify_input.push(committed_tok);
        verify_input.extend_from_slice(&drafts);
        let penalty = self.make_penalty_drafts(i, &drafts, rep_pen, presence_pen, freq_pen, has_penalty);
        let (nacc, emitted, vout) = if greedy {
            let (preds, vout) = self.gpu.verify_forward(
                &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos,
                Some(snapshot), penalty);
            let mut nacc = 0usize;
            while nacc < drafts.len() && preds[nacc] == drafts[nacc] { nacc += 1; }
            let mut e = drafts[..nacc].to_vec();
            e.push(preds[nacc]);
            (nacc, e, vout)
        } else {
            let seeds: Vec<u32> = (0..n).map(|j| rng_u32(step_key, RNG_DOM_VERIFY, j)).collect();
            let qprobs = vec![1.0f32; n - 1];   // a copied token is a point-mass proposal
            let (vs, vout) = self.gpu.verify_forward_sample(
                &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos,
                Some(snapshot), penalty, &drafts, &qprobs, temperature, top_k, top_p, &seeds, None);
            let mut nacc = 0usize;
            let mut e: Vec<u32> = Vec::with_capacity(n);
            let mut rejected = false;
            for j in 0..drafts.len() {
                if rng_uniform(step_key, RNG_DOM_ACCEPT, j) < vs.p_of_draft[j] {
                    e.push(drafts[j]);
                    nacc += 1;
                } else {
                    e.push(vs.resid_tok[j]);
                    rejected = true;
                    break;
                }
            }
            if !rejected { e.push(vs.bonus_tok); }
            (nacc, e, vout)
        };

        // GDN rollback to the last accepted column (slot snapshot + t = post-state of column t)
        if nacc + 1 != n {
            self.gpu.copy_gdn_slot(&self.state, snapshot + nacc, phys);
        }
        match mode {
            CopyMode::Mtp => {
                // h_prev = hidden at the last accepted position; re-prime the head over the span
                self.gpu.copy_hidden_col(*self.mtp_h_prev[phys].device_ptr(), &vout, nacc);
                let n_rp = nacc + 1;
                let rp_hidden = self.pool.get_bf16(h * n_rp);
                let rp_ptr = *rp_hidden.device_ptr();
                self.gpu.copy_hidden_col(rp_ptr, self.mtp_h_save.as_ref().unwrap(), 0);
                self.gpu.copy_hidden_cols(rp_ptr + (h * 2) as u64, &vout, 0, nacc);
                let mut rp_toks: Vec<u32> = Vec::with_capacity(n_rp);
                rp_toks.push(committed_tok);
                rp_toks.extend_from_slice(&drafts[..nacc]);
                self.gpu.mtp_reprime(&mut self.pool, &rp_hidden, &rp_toks, main_pos - 1,
                                     mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                self.pool.release_bf16(rp_hidden, h * n_rp);
            }
            CopyMode::Df2 => {
                let m = nacc + 1;
                let blk = crate::dflash2::block();
                let wide = self.df2_sink_tree.clone();
                let df2 = self.df2.as_mut().unwrap();
                if n <= blk {
                    // the verify's taps are in the chain sink (cols [0, n)), exactly as a DF2 round's
                    df2.sync_staging_from_sink().expect("copy round: df2 sync staging");
                    df2.inject_dev(m, None).expect("copy round: df2 inject_dev");
                } else {
                    let wide = wide.expect("copy round wider than the DFlash2 block needs the wide tap sink");
                    let mut s0 = 0usize;
                    while s0 < m {
                        let k = blk.min(m - s0);
                        let path: Vec<usize> = (s0..s0 + k).collect();
                        df2.sync_staging_from_wide(&wide, &path).expect("copy round: df2 staging from wide sink");
                        df2.inject_dev(k, None).expect("copy round: df2 inject_dev");
                        s0 += k;
                    }
                }
                df2.note_ring(phys);
            }
            CopyMode::Bare => {}
        }
        self.pool.release_bf16(vout, h * n);

        // emit, honoring EOS (min_new / ignore_eos) and max_new exactly like the drafter rounds
        let mut hit_eos = false;
        let mut to_emit: Vec<u32> = Vec::with_capacity(emitted.len());
        for &t in &emitted {
            if generated + to_emit.len() >= max_new { break; }
            to_emit.push(t);
            if eos.contains(&t) && !ignore_eos && generated + to_emit.len() >= min_new { hit_eos = true; break; }
        }
        let emit_count = to_emit.len();
        let finished = hit_eos || generated + emit_count >= max_new;
        {
            let cache = &mut self.slot_cache[phys];
            cache.push(committed_tok);
            cache.extend_from_slice(&drafts[..nacc]);
        }
        {
            let lane = self.lanes[i].as_mut().unwrap();
            for &t in &to_emit {
                let _ = lane.tx.send(TokEvent::Tok(t));
                lane.hist_push(t);
                lane.hist_trim();
            }
            lane.generated += emit_count;
            if !greedy { lane.seed = splitmix64(step_key); }
            lane.last_tok = *to_emit.last().unwrap_or(&committed_tok);
            if !finished {
                lane.pos = main_pos + nacc + 1;
                if mode == CopyMode::Mtp { lane.mtp_pos = main_pos + nacc; }
            } else {
                lane.pos = main_pos + emit_count;
            }
            if mode == CopyMode::Mtp && lane.df2_primed && !lane.df2_stale {
                // the DFlash2 ring did not see this round's taps (the MTP-fallback rule)
                lane.df2_stale = true;
            }
            if mode == CopyMode::Bare {
                // no drafter state was kept for these positions (the plain-step rule)
                lane.mtp_stale = true;
                lane.df2_stale = true;
            }
            if let Some(cs) = lane.copy.as_mut() {
                let before = *cs.counters();
                cs.commit(Some(prop), nacc, &to_emit);
                crate::exl3_serve::copy_span_count_delta(&before, cs.counters());
            }
        }
        // the in-server lossless auditor's shadow does not follow copy rounds: its audit ends here
        if self.audit_on { self.audit_cur[i] = None; }
        let ms = step_t0.elapsed().as_secs_f64() * 1e3;
        crate::exl3_serve::copy_span_count_round(drafts.len(), nacc, emit_count, ms);
        if mode == CopyMode::Bare { crate::exl3_serve::copy_span_count_bare(); }
        // the lane advanced by several positions: the device-resident decode loop re-uploads next step
        self.resident_dirty = true;
        crate::tel::add_spec(1, drafts.len() as u64, nacc as u64);
        self.rec_step(SpecStepRec {
            greedy, pos: main_pos as u32, drafts: drafts.len() as u32, nacc: nacc as u32,
            emitted: emit_count as u32, round_ms: 0.0, verify_ms: ms as f32, step_ms: ms as f32,
        });
        if crate::opts::var(crate::opt!("df2-step-log")).is_ok() {
            // ULTRA2 merge: through the log queue (served path; logq's direct-print baseline for batch.rs stays 36)
            crate::reprintln!("[copy-step] pos={main_pos} mode={mode:?} offered={} nacc={nacc} emitted={emit_count} step={ms:.1}ms",
                      drafts.len());
        }
        finished
    }
}
