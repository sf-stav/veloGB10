//! MTP tree and forest steps (multi-lane MTP speculation).
//! Another `impl BatchScheduler` block (file map in `mod.rs`); the methods keep their exact text.
//! Binding invariants: greedy MTP stays bitwise lossless; spec gating is per source (AGENTS.md section 4).

use super::*;

impl BatchScheduler {
    /// One FORK-THEN-CHAIN tree MTP step (greedy). Drafts a k=2 fork, verifies the tree, walks the
    /// accepted path (target argmax), compacts its KV to contiguous slots, adopts the accepted leaf's
    /// GDN checkpoint, re-primes MTP over the accepted path, and emits. Lossless: every emitted token is
    /// the target's argmax given its accepted prefix (same as the chain). Returns true if finished.
    pub(super) fn mtp_tree_step(&mut self, i: usize) -> bool {
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let depth = self.mtp.depth();
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let ckpt = self.mtp_snapshot_slot;   // tree checkpoints base (slots ckpt..ckpt+n-2)
        let kv_stride = self.kv_stride;
        let mtp_kc_ptr = *self.mtp_kc[phys].device_ptr();
        let mtp_vc_ptr = *self.mtp_vc[phys].device_ptr();
        let h_save_ptr = *self.mtp_h_save.as_ref().unwrap().device_ptr();

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
        let (rep_pen, presence_pen, freq_pen, has_penalty) = {
            let l = self.lanes[i].as_ref().unwrap();
            (l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.has_penalty())
        };
        let history: Vec<u32> = self.lanes[i].as_ref().unwrap().history.clone();

        // h_save = pre-verify hidden (for re-prime column 0).
        self.gpu.copy_hidden_col(h_save_ptr, &self.mtp_h_prev[phys], 0);

        // n-gram context (the second branch's source, if enabled).
        let ngram = self.ngram_draft;
        let mut work: Vec<u32> = Vec::new();
        if ngram > 0 {
            work.extend_from_slice(&self.slot_cache[phys]);
            work.push(committed_tok);
        }

        // ---- Draft the fork-then-chain tree. ----
        let (parent, tokens) = self.gpu.mtp_fork_draft(
            &mut self.pool, &self.mtp_h_prev[phys], committed_tok as i32, mtp_pos, depth,
            mtp_kc_ptr, mtp_vc_ptr, kv_stride, &work, ngram);
        let n = tokens.len();

        // ---- Verify the tree. ----
        let penalty = self.make_penalty(i, rep_pen, presence_pen, freq_pen, has_penalty);
        let topo = self.gpu.topo_from_parent(&parent, main_pos);
        let (preds, vout) = self.gpu.verify_forward_topo(
            &mut self.pool, &tokens, &mut self.state, phys, kv_stride, main_pos, Some(ckpt), penalty, Some(&topo));

        // ---- Accept walk: follow the target argmax down the tree. ----
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        for c in 1..n { children[parent[c] as usize].push(c); }
        let mut path = vec![0usize]; let mut emitted = Vec::new(); let mut cur = 0usize;
        loop {
            let want = preds[cur]; emitted.push(want);
            match children[cur].iter().copied().find(|&c| tokens[c] == want) {
                Some(c) => { path.push(c); cur = c; } None => break,
            }
        }
        let nacc = path.len() - 1;               // accepted drafts (emitted[0..nacc]); emitted[nacc]=bonus
        let leaf = *path.last().unwrap();

        // ---- Commit: compact the accepted path's KV, adopt the leaf's GDN state, re-prime MTP. ----
        let src_pos: Vec<i32> = path.iter().map(|&p| p as i32).collect();
        self.gpu.compact_kv(&mut self.pool, &mut self.state, phys, main_pos, &src_pos, kv_stride);
        // Adopt the accepted leaf's GDN checkpoint. The DFS scan ends at column n-1, so if leaf==n-1 the
        // slot already holds the right state; else restore from its checkpoint slot (ckpt+leaf).
        if leaf != n - 1 { self.gpu.copy_gdn_slot(&self.state, ckpt + leaf, phys); }
        // h_prev = hidden at the accepted leaf.
        self.gpu.copy_hidden_col(*self.mtp_h_prev[phys].device_ptr(), &vout, leaf);
        // Re-prime MTP over the accepted path: column 0 uses h_save + committed; column k uses
        // vout[path[k-1]] + tokens[path[k]].
        let n_rp = nacc + 1;
        let rp_hidden = self.pool.get_bf16(h * n_rp);
        let rp_ptr = *rp_hidden.device_ptr();
        let mut rp_toks: Vec<u32> = vec![committed_tok];
        self.gpu.copy_hidden_col(rp_ptr, self.mtp_h_save.as_ref().unwrap(), 0);
        for k in 1..=nacc {
            self.gpu.copy_hidden_col(rp_ptr + (k * h * 2) as u64, &vout, path[k - 1]);
            rp_toks.push(tokens[path[k]]);
        }
        self.gpu.mtp_reprime(&mut self.pool, &rp_hidden, &rp_toks, main_pos - 1, mtp_kc_ptr, mtp_vc_ptr, kv_stride);
        self.pool.release_bf16(rp_hidden, h * n_rp);
        self.pool.release_bf16(vout, h * n);

        // ---- Emit (same EOS/max_new discipline as the chain). ----
        let mut new_toks: Vec<u32> = Vec::with_capacity(nacc + 1);
        let mut hit_eos = false;
        for k in 0..nacc {
            if generated + new_toks.len() >= max_new { break; }
            new_toks.push(emitted[k]);
            if eos.contains(&emitted[k]) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; break; }
        }
        if !hit_eos && generated + new_toks.len() < max_new {
            new_toks.push(emitted[nacc]);   // bonus
            if eos.contains(&emitted[nacc]) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; }
        }
        let emit_count = new_toks.len();
        let finished = hit_eos || generated + emit_count >= max_new;

        // Cache what was FED: committed + accepted node tokens (emitted[0..nacc]).
        {
            let cache = &mut self.slot_cache[phys];
            cache.push(committed_tok);
            cache.extend_from_slice(&emitted[..nacc]);
        }
        {
            let lane = self.lanes[i].as_mut().unwrap();
            for &t in &new_toks {
                let _ = lane.tx.send(TokEvent::Tok(t));
                lane.hist_push(t);
                lane.hist_trim();
            }
            lane.generated += emit_count;
            if !finished {
                lane.last_tok = emitted[nacc];
                lane.pos = main_pos + nacc + 1;
                lane.mtp_pos = main_pos + nacc;
            } else {
                lane.last_tok = *new_toks.last().unwrap_or(&committed_tok);
                lane.pos = main_pos + emit_count;
            }
        }
        self.mtp_stat_steps += 1;
        self.mtp_stat_drafts += (n - 1) as u64;
        self.mtp_stat_accepted += nacc as u64;
        self.mtp_stat_emitted += emit_count as u64;
        self.mtp_stat_verify_fwds += 1;
        self.mtp.record_step((n - 1) as u64, nacc as u64, emit_count as u64);
        if self.mtp_stat_steps % 50 == 0 {
            let acc = if self.mtp_stat_drafts > 0 { self.mtp_stat_accepted as f64 / self.mtp_stat_drafts as f64 * 100.0 } else { 0.0 };
            let eff = self.mtp_stat_emitted as f64 / self.mtp_stat_verify_fwds as f64;
            crate::reprintln!("[mtp/tree] steps={} accepted={:.1}% emitted={} tok/verify_fwd={:.3} (depth {} n {}) accept@k [{}]",
                      self.mtp_stat_steps, acc, self.mtp_stat_emitted, eff, depth, n, fmt_accept_by_depth(&self.mtp));
            self.dump_accept_curve();
        }
        finished
    }

    /// One MTP speculative-decoding step for lane `i` (greedy, penalty-free). Mirrors the validated
    /// `bench_mtp` loop body: draft (depth-1) → snapshot GDN → verify (depth) → accept longest
    /// prefix → rollback+reverify on partial reject → re-prime MTP over the accepted prefix with REAL
    /// verify hiddens. Emits the accepted drafts + bonus token, advancing the lane by nacc+1
    /// positions. Returns true if the lane finished (EOS or max_new reached).
    /// FOREST MTP step (LANES design Step 3c): draft each of `lanes`' chains, pack them into ONE forest
    /// verify (proven lossless by `--bench-lanes`), then per-lane accept / rollback / re-prime / emit.
    /// This is `mtp_lane_step` generalized to L lanes sharing one main-model forward — the concurrency
    /// throughput win. Greedy, no-penalty lanes only (v1). Returns (lane_index, finished) per lane.
    pub(super) fn mtp_forest_step(&mut self, lanes: &[usize]) -> Vec<(usize, bool)> {
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let ck = self.gpu.cfg().conv_kernel;
        let mv = crate::gpu::MAX_VERIFY;
        let kv_stride = self.kv_stride;
        let snapshot = self.mtp_snapshot_slot;
        let p = lanes.len();
        // v1 allocator: uniform per-lane depth so Σ(1+d) = p*(1+d) ≤ MAX_VERIFY, capped by the policy
        // depth. The floor is the pack cap's derivation: p=5 ⇒ depth 2, p=6 ⇒ depth 1 (no drafts,
        // useless) — which is why `take` caps at 5 upstream.
        let depth = ((mv / p).saturating_sub(1)).clamp(1, self.mtp.depth());
        // BATCHED_MTP §5 risk 5: `clamp(1, d)` panics when the policy reports depth 0 (min > max).
        // A depth-0 policy has nothing to speculate with — serve each packed lane via the
        // single-lane path (which handles depth 0) instead of panicking.
        if self.mtp.depth() == 0 {
            crate::reprintln!("[mtp] forest: policy depth 0 — per-lane MTP steps (no forest pack)");
            return lanes.iter().map(|&i| (i, self.mtp_lane_step(i))).collect();
        }
        // BATCHED_MTP P0 (the confound fix): the forest draft loop previously ignored
        // --ngram-draft and the coverage trace, so packed lanes drafted with a DIFFERENT
        // behaviour than the single-lane arms, and coverage studies silently under-sampled
        // exactly the packed lanes. Both hooks now mirror mtp_lane_step verbatim.
        let ngram = self.ngram_draft;
        let cov_on = self.cov_trace;
        let step_t0 = std::time::Instant::now();

        // Per-lane locals + draft chains (drafting is sequential; the shared MTP scratch is reused, and
        // each lane's head KV is its own mtp_kc/mtp_vc slot). Each lane's chain = [committed, drafts...].
        struct L { i: usize, phys: usize, committed: u32, main_pos: usize, mtp_pos: usize,
                   generated: usize, max_new: usize, drafts: Vec<u32>, start: usize, n: usize,
                   rep: f32, pres: f32, freq: f32, history: Vec<u32>, ignore_eos: bool, min_new: usize,
                   cov: Vec<Vec<(u32, f32)>> }
        let cur_ptr = *self.mtp_cur_hidden.as_ref().unwrap().device_ptr();
        let mut ls: Vec<L> = Vec::with_capacity(p);
        let mut global = 0usize;
        for &i in lanes {
            let (phys, committed, main_pos, mtp_pos, generated, max_new, rep, pres, freq, history, ignore_eos, min_new) = {
                let l = self.lanes[i].as_ref().unwrap();
                (l.phys, l.last_tok, l.pos, l.mtp_pos, l.generated, l.max_new,
                 l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.history.clone(),
                 l.ignore_eos, l.min_new)
            };
            let mtp_kc_ptr = *self.mtp_kc[phys].device_ptr();
            let mtp_vc_ptr = *self.mtp_vc[phys].device_ptr();
            self.gpu.copy_hidden_col(cur_ptr, &self.mtp_h_prev[phys], 0);
            // Prompt-lookup context for n-gram drafting (mtp_lane_step's twin): the lane's full
            // realized sequence from slot_cache plus this step's committed token.
            let mut work: Vec<u32> = Vec::new();
            if ngram > 0 {
                work.reserve(self.slot_cache[phys].len() + depth);
                work.extend_from_slice(&self.slot_cache[phys]);
                work.push(committed);
            }
            let mut cov: Vec<Vec<(u32, f32)>> = Vec::new();
            let mut drafts: Vec<u32> = Vec::with_capacity(depth - 1);
            let mut cur_tok = committed as i32;
            let mut dpos = mtp_pos;
            for _ in 0..depth - 1 {
                let m = self.gpu.mtp_draft_step(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap(),
                                                cur_tok, dpos, mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                self.gpu.copy_hidden_col(cur_ptr, &m, 0);
                self.pool.release_bf16(m, h);
                cur_tok = self.gpu.argmax_hidden(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap()) as i32;
                if cov_on {
                    cov.push(self.gpu.topk_hidden_kv(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap(), 8));
                }
                // PROMPT-LOOKUP OVERRIDE (mtp_lane_step's twin): if the last `ngram` tokens recur
                // earlier in this lane's context, propose the token that followed the most recent
                // earlier occurrence — lossless, the packed verify checks every draft.
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
            let n = 1 + drafts.len();
            ls.push(L { i, phys, committed, main_pos, mtp_pos, generated, max_new, drafts, start: global, n, ignore_eos, min_new,
                        rep, pres, freq, history, cov });
            global += n;
        }
        let ntot = global;

        // Build the FOREST topo + packed token stream (see bench_lanes / verify_forward_core_topo).
        let mut tokens: Vec<u32> = Vec::with_capacity(ntot);
        let mut parent = vec![-1i32; ntot];
        let mut slotv = vec![0i32; ntot];
        let mut rope = vec![0i32; ntot];
        let mut kv_pos = vec![0i32; ntot];
        let mut cps = vec![0i32; ntot];
        let mut path = vec![0u8; ntot * mv];
        let mut winsrc = vec![0i32; ntot * ck];
        for l in &ls {
            tokens.push(l.committed);
            tokens.extend_from_slice(&l.drafts);
            for r in 0..l.n {
                let c = l.start + r;
                parent[c] = if r == 0 { -1 } else { (c - 1) as i32 };
                slotv[c] = l.phys as i32;
                rope[c] = (l.main_pos + r) as i32;
                kv_pos[c] = (l.main_pos + r) as i32;
                cps[c] = l.main_pos as i32;
                for dd in 0..=r { path[c * mv + dd] = dd as u8; }
                for j in 0..ck {
                    let wd = r as i32 - (ck as i32 - 1) + j as i32;
                    winsrc[c * ck + j] = if wd < 0 { wd } else { l.start as i32 + wd };
                }
            }
        }
        let topo = crate::gpu::TreeTopo { rope, kv_pos, parent, path, winsrc,
                                          slot: Some(slotv), col_pos_start: Some(cps) };
        let pos_start_max = ls.iter().map(|l| l.main_pos).max().unwrap();

        // Per-lane verify penalty: each column carries ITS lane's rep/presence/freq penalty (buffers are
        // MAX_VERIFY-sized). None if no packed lane has a penalty.
        let ls_pen: Vec<(usize, usize, f32, f32, f32, usize)> =
            ls.iter().map(|l| (l.start, l.n, l.rep, l.pres, l.freq, l.i)).collect();
        let penalty = self.make_forest_penalty(&ls_pen);

        // ONE packed forest verify. ckpt writes column t's post-state to (snapshot + t); rollback reads
        // each lane's own accepted checkpoint.
        let (preds, vout) = self.gpu.verify_forward_topo(
            &mut self.pool, &tokens, &mut self.state, 0, kv_stride, pos_start_max, Some(snapshot), penalty, Some(&topo));

        let eos = self.eos.clone();
        let mut results: Vec<(usize, bool)> = Vec::with_capacity(p);
        // Per-lane (main_pos, nacc, emitted), index-aligned with ls, for the step-dump records.
        let mut lane_stats: Vec<(usize, usize, usize)> = Vec::with_capacity(p);
        for l in &ls {
            let draft_count = l.drafts.len();
            let mut nacc = 0usize;
            while nacc < draft_count && preds[l.start + nacc] == l.drafts[nacc] { nacc += 1; }
            let bonus = preds[l.start + nacc];
            // Rollback on partial reject: restore this lane's state as of its last accepted column.
            if nacc + 1 != l.n {
                self.gpu.copy_gdn_slot(&self.state, snapshot + l.start + nacc, l.phys);
            }
            // Re-prime this lane's MTP head over its accepted prefix (k=0 = pre-step h_prev, still intact;
            // k>=1 = the verify's real hidden vout[start+k-1]). Then advance h_prev to vout[start+nacc].
            let mtp_kc_ptr = *self.mtp_kc[l.phys].device_ptr();
            let mtp_vc_ptr = *self.mtp_vc[l.phys].device_ptr();
            let n_rp = nacc + 1;
            let rp_hidden = self.pool.get_bf16(h * n_rp);
            let rp_ptr = *rp_hidden.device_ptr();
            let mut rp_toks: Vec<u32> = Vec::with_capacity(n_rp);
            self.gpu.copy_hidden_col(rp_ptr, &self.mtp_h_prev[l.phys], 0);
            rp_toks.push(l.committed);
            // vout columns l.start..l.start+nacc-1 are CONTIGUOUS — one dtod for the whole prefix
            // (was nacc separate copy_hidden_col driver calls).
            self.gpu.copy_hidden_cols(rp_ptr + (h * 2) as u64, &vout, l.start, nacc);
            for k in 1..=nacc {
                rp_toks.push(l.drafts[k - 1]);
            }
            self.gpu.mtp_reprime(&mut self.pool, &rp_hidden, &rp_toks, l.main_pos - 1,
                                 mtp_kc_ptr, mtp_vc_ptr, kv_stride);
            self.pool.release_bf16(rp_hidden, h * n_rp);
            self.gpu.copy_hidden_col(*self.mtp_h_prev[l.phys].device_ptr(), &vout, l.start + nacc);

            // Emit accepted drafts + bonus (budget on drafts too — see mtp_lane_step).
            let mut new_toks: Vec<u32> = Vec::with_capacity(nacc + 1);
            let mut hit_eos = false;
            for &d in l.drafts.iter().take(nacc) {
                if l.generated + new_toks.len() >= l.max_new { break; }
                new_toks.push(d);
                if eos.contains(&d) && !l.ignore_eos && l.generated + new_toks.len() >= l.min_new { hit_eos = true; break; }
            }
            if !hit_eos && l.generated + new_toks.len() < l.max_new {
                new_toks.push(bonus);
                if eos.contains(&bonus) && !l.ignore_eos && l.generated + new_toks.len() >= l.min_new { hit_eos = true; }
            }
            let emit_count = new_toks.len();
            let finished = hit_eos || l.generated + emit_count >= l.max_new;
            {
                let cache = &mut self.slot_cache[l.phys];
                cache.push(l.committed);
                cache.extend_from_slice(&l.drafts[..nacc]);
            }
            {
                let lane = self.lanes[l.i].as_mut().unwrap();
                for &t in &new_toks {
                    let _ = lane.tx.send(TokEvent::Tok(t));
                    lane.hist_push(t);
                    lane.hist_trim();
                }
                lane.generated += emit_count;
                if !finished {
                    lane.last_tok = bonus;
                    lane.pos = l.main_pos + nacc + 1;
                    lane.mtp_pos = l.main_pos + nacc;
                } else {
                    lane.last_tok = *new_toks.last().unwrap_or(&l.committed);
                    lane.pos = l.main_pos + emit_count;
                }
            }
            // Telemetry parity with the chain path: the forest lane's step record reaches the
            // same spec-steps ring (acceptance curves must count forest-served lanes too).
            self.rec_step(SpecStepRec {
                greedy: true, pos: l.main_pos as u32, drafts: draft_count as u32, nacc: nacc as u32,
                emitted: emit_count as u32, round_ms: 0.0, verify_ms: 0.0,
                step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
            });
            self.mtp_stat_drafts += draft_count as u64;
            self.mtp_stat_accepted += nacc as u64;
            self.mtp_stat_emitted += emit_count as u64;
            lane_stats.push((l.main_pos, nacc, emit_count));
            results.push((l.i, finished));
        }
        self.pool.release_bf16(vout, h * ntot);
        self.mtp_stat_steps += p as u64;     // p lane-steps served by...
        self.mtp_stat_verify_fwds += 1;      // ...ONE main-model forward — the batching win

        // BATCHED_MTP §0.4 (the telemetry gap): the forest previously ended with the two counters
        // and NO [mtp] line, so a run where every step took the forest branch never showed its
        // accept@k / tok-per-forward — exactly where a collapse would hide. Same cadence and
        // counters as the chain path, labeled with the pack geometry.
        if self.mtp_stat_steps % 50 == 0 {
            let acc = if self.mtp_stat_drafts > 0 {
                self.mtp_stat_accepted as f64 / self.mtp_stat_drafts as f64 * 100.0
            } else { 0.0 };
            let eff = self.mtp_stat_emitted as f64 / self.mtp_stat_verify_fwds as f64;
            crate::reprintln!("[mtp] steps={} drafts={} accepted={:.1}% emitted={} tok/verify_fwd={:.3} (forest p={} d={}) accept@k [{}]",
                      self.mtp_stat_steps, self.mtp_stat_drafts, acc,
                      self.mtp_stat_emitted, eff, p, depth, fmt_accept_by_depth(&self.mtp));
            crate::tel::publish_mtp(self.mtp_stat_steps, self.mtp_stat_drafts,
                                    self.mtp_stat_accepted, self.mtp_stat_emitted,
                                    self.mtp_stat_verify_fwds, depth, &self.mtp.hazard_counts());
            self.dump_accept_curve();
        }
        // Coverage parity (§0.4 divergence 2): with the step dump on, each packed lane's draft
        // top-k reaches the dump — a forest run no longer under-samples its own lanes. Target
        // coverage fields stay empty (the packed verify exposes no per-lane target top-k).
        if self.step_dump.is_some() {
            for (l, &(main_pos, nacc, emit_count)) in ls.iter().zip(lane_stats.iter()) {
                let bonus = preds.get(l.start + nacc).copied().unwrap_or(l.committed);
                let mut dk_ids = Vec::with_capacity(l.cov.len() * 8);
                let mut dk_logit = Vec::with_capacity(l.cov.len() * 8);
                for tk in &l.cov {
                    for (t, v) in tk { dk_ids.push(*t); dk_logit.push(*v); }
                }
                if let Some(d) = self.step_dump.as_mut() {
                    let rec = crate::dflash2::stepdump::MtpStepRec {
                        step: self.mtp_stat_steps, pos: main_pos, committed: l.committed, depth,
                        drafts: l.drafts.clone(), p_draft: Vec::new(), resid: Vec::new(),
                        bonus, nacc, emitted: emit_count,
                        tgt_ids: Vec::new(), tgt_logit: Vec::new(), tgt_p1: Vec::new(), tgt_margin: Vec::new(),
                        draft_topk_ids: dk_ids, draft_topk_logit: dk_logit,
                        tgt_top20: Vec::new(),
                    };
                    d.record_mtp(&rec);
                }
            }
        }
        results
    }
}
