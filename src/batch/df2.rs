//! DFlash2 lanes: tree step, chain step, sampled and rejection-sampled variants.
//! Another `impl BatchScheduler` block (file map in `mod.rs`); the methods keep their exact text.
//! Binding invariants: greedy MTP stays bitwise lossless; spec gating is per source (AGENTS.md section 4).

use super::*;

impl BatchScheduler {
    /// One DFlash2 speculative-decoding step for lane `i` (greedy). The round is the DRAFT source
    /// (S4F: trunk taps → fc/hidden_norm → 5-layer block pass → borrowed LM head → top-16 →
    /// selector chain → 7 draft tokens); the verify is the trunk's M=8 chain verify with the
    /// k_verify≡8 constant (the S2F fold). Accept = longest argmax prefix; rejected drafts are
    /// FREE losslessness (the emitted stream is the target's argmax at every position).
    ///
    /// Ring bookkeeping (the S4F round contract): `nprev == lane.pos == main_pos` at entry — the
    /// ring holds the taps of positions 0..main_pos-1; the anchor (last_tok, at position
    /// main_pos-1... see the invariant note) is the block input, NOT a ctx row. The trunk verify
    /// captures the fed span's taps (cols [0,8) of the sink staging, stream-ordered with the
    /// verify); `inject_dev(nacc+1)` then advances the ring by the accepted span, so the invariant
    /// holds at the next step. The drafter's ring KV is drafter-private (never aliases trunk
    /// slots — the probe asserts the pointer ranges).
    ///
    /// Emits the accepted drafts + bonus, advancing the lane by nacc+1. Returns true if finished.
    /// PLAN/25 Phase 1 — the DF2 TREE lane (opt-in `--spec-source dflash2-tree`, greedy v0).
    /// The selector round already computes a top-16 candidate table for each of the 7 levels in
    /// its single block forward; the chain is one walk through it. This lane walks a SECOND
    /// chain (best per-level candidate that the walk did not pick) and verifies BOTH as a
    /// 15-node tree: per-node KV paths (`verify_forward_topo`), per-node target argmax,
    /// longest root-path accept (lossless — every emitted token is a target argmax token).
    /// Commit: GDN checkpoint of the accepted leaf (per-column ckpts written by the topo
    /// verify), the ring staging compacted to the accepted path (`compact_staging`), then the
    /// standing `inject_dev`. The chain lane (`df2_lane_step`) is untouched when this mode is off.
    pub(super) fn df2_tree_step(&mut self, i: usize) -> bool {
        const LEVELS: usize = 7;
        // DF2 block-16 x tree is REFUSED, loudly: the tree verifies `1 + LEVELS + nb` nodes with
        // nb <= LEVELS, i.e. up to 15 columns at block 8. At block 16 the same shape is
        // `1 + 15 + 15 = 31` columns, which exceeds `gpu::MAX_VERIFY = 16` — the verify would fall
        // to the prefill dequant path (silently wrong work) or overrun the checkpoint region.
        // Never truncate branch B silently to make it fit; the combination is unsupported until
        // the checkpoint/verify width is raised deliberately.
        assert_eq!(crate::dflash2::block(), 8,
            "df2 block-16 x dflash2-tree is unsupported: the tree needs up to 31 verify columns              (1 + 15 + 15) but MAX_VERIFY is 16. Refusing rather than truncating branch B. \
             Serve block 16 with --spec-source dflash2 (chain) or use the tree at --df2-block 8.");
        assert_eq!(LEVELS, crate::dflash2::levels(),
            "df2 tree LEVELS {} != live levels {}", LEVELS, crate::dflash2::levels());
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let ckpt = self.mtp_snapshot_slot;   // per-column GDN checkpoint base (MAX_VERIFY slots)
        // /health (the tree rider): the tree step is otherwise invisible to the step ring (the
        // chain lanes' note_step sites do not cover this path), so a tree-served turn showed
        // `mode: none` with zero samples while the NEXT turn's MTP window mislabelled the mode.
        crate::tel::note_step();
        // Lazy arm: schedulers not built through `with_df2` (the lossless probe) still get the
        // WIDE tap sink on the first tree step — it must be on the trunk BEFORE the verify's
        // capture D2Ds run. One Option test per step.
        if self.df2_sink_tree.is_none() {
            let ws = std::sync::Arc::new(
                crate::dflash2::capture::Df2TapSink::new_cols(self.gpu.dev(), crate::gpu::MAX_VERIFY));
            self.gpu.set_df2_capture_tree(ws.clone());
            self.df2_sink_tree = Some(ws);
        }
        let kv_stride = self.kv_stride;

        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
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

        // ---- Draft: the S4F round exactly as the chain runs it (graph when captured). ----
        let step_t0 = std::time::Instant::now();
        let dump_on = self.cov_trace;   // op-sequence gate: must match across TP ranks
        let (drafts, full_out): (Vec<u32>, Option<crate::dflash2::round::Df2RoundOut>) = {
            let df2 = self.df2.as_mut().unwrap();
            assert_eq!(df2.nprev(), main_pos,
                "df2 ring nprev {} != lane pos {} (ring stale or unprimed)", df2.nprev(), main_pos);
            if dump_on {
                // Coverage-trace mode keeps the eager FULL round (its op sequence is what the
                // TP node mirrors and the dump parity harness expects).
                let o = df2.draft_round_full(committed_tok).expect("df2 draft_round_full");
                let toks = o.tokens.clone();
                (toks, Some(o))
            } else {
                // Serving path: the CAPTURED graph (the chain lane's round) + one 448 B dtoh for
                // the top-16 candidate table the B branch ranks from — the full round's ~640 KB
                // h_final readback stays out of the serving step.
                let mk_out = |cand: Vec<u32>| crate::dflash2::round::Df2RoundOut {
                    tokens: Vec::new(), candidates: cand, unary: Vec::new(), scores: Vec::new(),
                    h_final: Vec::new(), layer_hiddens: Vec::new(), logits: None, hp: None,
                    stage_ms: None,
                };
                if df2.round_graph.is_some() {
                    let toks = df2.draft_round_graph(committed_tok).expect("df2 draft_round_graph");
                    let cand = df2.candidate_table().expect("df2 candidate_table");
                    (toks, Some(mk_out(cand)))
                } else {
                    let toks = df2.draft_round_dev(committed_tok).expect("df2 draft_round");
                    let cand = df2.candidate_table().expect("df2 candidate_table");
                    (toks, Some(mk_out(cand)))
                }
            }
        };
        let round_ms = step_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Build the tree: chain A + branch B (best per-level alternate by selector q). ----
        // candidate table layout: [level][16] (token, q); the walk's own chain pick per level is
        // `drafts[j]`. Branch B takes, per level, the highest-q candidate that is neither the
        // chain pick at that level nor any B pick so far (dedupe keeps the tree duplicate-free).
        // The candidate table rides out with the FULL round (head-logit rank order per level —
        // the same table the Phase-0 c_k curves measured). Branch B takes, per level, the
        // highest-ranked candidate that is neither the walk's pick at that level nor any B pick
        // so far (dedupe keeps the tree duplicate-free).
        let cand_tok: Vec<u32> = full_out.as_ref()
            .map(|o| o.candidates.clone())
            .expect("df2-tree: the full round must carry the candidate table");
        let mut b_toks: Vec<u32> = Vec::with_capacity(LEVELS);
        for j in 0..LEVELS {
            let row_t = &cand_tok[j * 16..(j + 1) * 16];
            let mut best: Option<u32> = None;
            for c in 0..16 {
                let t = row_t[c];
                if t == drafts[j] || b_toks.contains(&t) { continue; }
                best = Some(t);
                break;
            }
            match best { Some(t) => b_toks.push(t), None => break }
        }
        let nb = b_toks.len();
        let n = 1 + drafts.len() + nb;             // root + A + B
        let mut tokens: Vec<u32> = Vec::with_capacity(n);
        tokens.push(committed_tok);
        tokens.extend_from_slice(&drafts);
        tokens.extend_from_slice(&b_toks);
        let mut parent: Vec<i32> = vec![-1i32; n];
        for c in 1..=drafts.len() { parent[c] = c as i32 - 1; }
        if nb > 0 { parent[1 + drafts.len()] = 0; } // B rooted at the committed token
        for k in 1..nb { parent[1 + drafts.len() + k] = (drafts.len() + k) as i32; }

        // ---- Verify the tree (per-node KV paths + per-node argmax). ----
        let penalty = self.make_penalty(i, rep_pen, presence_pen, freq_pen, has_penalty);
        let verify_t0 = std::time::Instant::now();
        let topo = self.gpu.topo_from_parent(&parent, main_pos);
        let (preds, vout) = self.gpu.verify_forward_topo(
            &mut self.pool, &tokens, &mut self.state, phys, kv_stride, main_pos, Some(ckpt),
            penalty, Some(&topo));
        let verify_ms = verify_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Accept walk: follow the target argmax down the tree (mtp_tree_step rule). ----
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        for c in 1..n { children[parent[c] as usize].push(c); }
        let mut path = vec![0usize];
        let mut emitted: Vec<u32> = Vec::new();
        let mut cur = 0usize;
        loop {
            let want = preds[cur];
            emitted.push(want);
            match children[cur].iter().copied().find(|&c| tokens[c] == want) {
                Some(c) => { path.push(c); cur = c; }
                None => break,
            }
        }
        let nacc = path.len() - 1;
        let leaf = *path.last().unwrap();

        // ---- Telemetry: did the B branch actually earn its verify width? (greedy v0 has no
        // other visibility -- the serving A/B showed tree = chain tok/s; this one line says
        // whether that is "rescue rate ~ 0, chain never derails" or "tree never engaged".) ----
        self.df2_tree_stat_steps += 1;
        self.df2_tree_stat_nodes += n as u64;
        self.df2_tree_stat_drafts += drafts.len() as u64;   // FIX1 #76: engagement side
        self.df2_tree_stat_accepted += nacc as u64;
        if leaf >= 1 + drafts.len() { self.df2_tree_stat_rescues += 1; }
        if self.df2_tree_stat_steps % 50 == 0 {
            let resc = self.df2_tree_stat_rescues as f64
                / self.df2_tree_stat_steps.max(1) as f64 * 100.0;
            let width = self.df2_tree_stat_nodes as f64 / self.df2_tree_stat_steps.max(1) as f64;
            crate::reprintln!("[df2-tree] steps={} b_rescue={:.1}% avg_nodes={:.1} (chain equivalent {})",
                      self.df2_tree_stat_steps, resc, width, LEVELS + 1);
            // (/health: the tree window now publishes EVERY step at the emit point below —
            // FIX1 #76 — so mode == "dflash2-tree" is live mid-request, not at step 50.)
        }

        // ---- Commit: compact the accepted path's KV into sequential slots [main_pos..). The topo
        // ---- verify writes per-COLUMN slots (main_pos+c); a path that crossed into the B branch
        // ---- leaves accepted tokens at scattered slots, and the next verify reads this span
        // ---- DIRECTLY (dd<0 → slot r) — without the compaction its prefix context is corrupted.
        // ---- Identity short-circuit inside: a pure A-chain accept (the common case) is a no-op.
        {
            let src_pos: Vec<i32> = path.iter().map(|&p| p as i32).collect();
            self.gpu.compact_kv(&mut self.pool, &mut self.state, phys, main_pos, &src_pos, kv_stride);
        }
        // ---- Adopt the leaf's GDN checkpoint (the topo verify wrote per-column ckpts).
        if leaf != n - 1 { self.gpu.copy_gdn_slot(&self.state, ckpt + leaf, phys); }

        // ---- Ring: gather the accepted path's taps (wide tree sink -> ring order), then the
        // ---- standing inject. nacc+1 <= BLOCK always (one branch's depth).
        {
            let wide = self.df2_sink_tree.as_ref()
                .expect("dflash2-tree: wide tap sink not armed");
            let df2 = self.df2.as_mut().unwrap();
            df2.sync_staging_from_wide(wide, &path).expect("df2 sync_staging_from_wide");
            df2.inject_dev(nacc + 1, None).expect("df2 inject_dev");
        }
        self.pool.release_bf16(vout, h * n);

        // ---- Emit accepted + bonus (the standing EOS/max_new discipline). ----
        let mut new_toks: Vec<u32> = Vec::with_capacity(nacc + 1);
        let mut hit_eos = false;
        for k in 0..nacc {
            if generated + new_toks.len() >= max_new { break; }
            new_toks.push(emitted[k]);
            if eos.contains(&emitted[k]) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; break; }
        }
        if !hit_eos && generated + new_toks.len() < max_new {
            new_toks.push(emitted[nacc]);
            if eos.contains(&emitted[nacc]) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; }
        }
        let emit_count = new_toks.len();
        let finished = hit_eos || generated + emit_count >= max_new;

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
                lane.df2_stale = true;
            }
        }
        self.mtp.record_step(drafts.len() as u64, nacc as u64, emit_count as u64);
        // FIX1 #76: the tree's engagement numbers publish EVERY step here (emit_count is the one
        // value not known before the emit loop). Watermark, not %50: a short request must still
        // light up velogb10_spec_*.
        self.df2_tree_stat_emitted += emit_count as u64;
        if self.df2_tree_stat_steps != self.df2_tree_stat_published {
            self.df2_tree_stat_published = self.df2_tree_stat_steps;
            crate::tel::publish_df2_tree(self.df2_tree_stat_steps, self.df2_tree_stat_drafts,
                                         self.df2_tree_stat_accepted, self.df2_tree_stat_emitted,
                                         self.df2_tree_stat_nodes, self.df2_tree_stat_rescues);
        }
        self.rec_step(SpecStepRec {
            greedy: true, pos: main_pos as u32, drafts: drafts.len() as u32, nacc: nacc as u32,
            emitted: emit_count as u32, round_ms, verify_ms,
            step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
        });
        finished
    }

    pub(super) fn df2_lane_step(&mut self, i: usize) -> bool {
        crate::tel::note_step();   // A.2: per-step interval for the status route (lock-free)
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let snapshot = self.mtp_snapshot_slot;
        let kv_stride = self.kv_stride;

        // Snapshot lane state into locals (avoids holding &mut self.lanes across GPU calls).
        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
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

        // ---- Draft: the S4F round (refresh block positions, then the lean draft). ----
        let step_t0 = std::time::Instant::now();
        let dump_on = self.cov_trace;   // op-sequence gate: must match across TP ranks
        let (drafts, full_out): (Vec<u32>, Option<crate::dflash2::round::Df2RoundOut>) = {
            let df2 = self.df2.as_mut().unwrap();
            assert_eq!(df2.nprev(), main_pos,
                "df2 ring nprev {} != lane pos {} (ring stale or unprimed)", df2.nprev(), main_pos);
            if dump_on {
                // S5F3 dump: the eager FULL round (tokens + candidates/unary/scores + h_final)
                // — behavior-neutral vs the graph/eager lean path (probe: graph == eager
                // bit-identical); the extra readbacks are the dump's payload.
                df2.refresh_block_pos().expect("df2 refresh_block_pos");
                let o = df2.draft_round_full(committed_tok).expect("df2 draft_round_full");
                (o.tokens.clone(), Some(o))
            } else if df2.round_graph.is_some() {
                // S5F graph path: draft_round_graph writes the per-replay device inputs
                // (anchor, nprev, block positions) + gathers the block RoPE itself.
                (df2.draft_round_graph(committed_tok).expect("df2 draft_round_graph"), None)
            } else {
                df2.refresh_block_pos().expect("df2 refresh_block_pos");
                (df2.draft_round_dev(committed_tok).expect("df2 draft_round"), None)
            }
        };
        let round_ms = step_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Verify [committed_tok, drafts...] (M=8 buckets) on the trunk at main_pos.. ----
        // The chain verify captures its own CUDA graph (n=8 key) — the DFlash2 "verify pair".
        let mut verify_input = vec![committed_tok];
        verify_input.extend(drafts.iter().copied());
        let penalty = self.make_penalty(i, rep_pen, presence_pen, freq_pen, has_penalty);
        let verify_t0 = std::time::Instant::now();
        // PLAN/25 Phase 0 (cov_on): keep the verify logits and derive preds + the per-column
        // target top-4/p1/top1−top2 margin host-side (the bench_accept readback; the eager core
        // is exactly what the E13 graph captures, so preds are identical to the non-dump path).
        // Dump-only. cov_on already folds in !has_penalty (penalized steps keep the plain
        // penalized verify and carry empty tgt fields; has_penalty is rank-identical — SPMD-safe).
        let cov_on = self.cov_trace && !has_penalty;
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
        let bonus = preds[nacc];

        // ---- GDN rollback on partial reject (restore the state as of the last accepted column). --
        if nacc + 1 != crate::dflash2::block() {
            self.gpu.copy_gdn_slot(&self.state, snapshot + nacc, phys);
        }

        // ---- PLAN/25 Phase 0: the LOGGING-ONLY MTP chain (coverage capture, dump-only). ----
        // Runs the MTP head over the SAME committed prefix this step's DF2 round drafted from,
        // recording its top-8 per position — the union instrument: both drafters' candidate sets
        // on ONE step grid, joined by (tag, pos) offline with zero grid-alignment bias. The
        // chain writes nothing but the record fields: the DF2 drafts above are untouched, and the
        // head's speculative KV entries beyond the re-primed span have exactly the mtp lane's own
        // semantics (always rewritten before they are ever attended).
        let mut cov_mtp_sets: Option<(Vec<u32>, Vec<f32>)> = None;
        if cov_on {
            let kc = *self.mtp_kc[phys].device_ptr();
            let vc = *self.mtp_vc[phys].device_ptr();
            let cur_ptr = *self.mtp_cur_hidden.as_ref().unwrap().device_ptr();
            self.gpu.copy_hidden_col(cur_ptr, &self.mtp_h_prev[phys], 0);
            let mut cur_tok = committed_tok as i32;
            let mut dpos = main_pos - 1;
            let mut mids = Vec::with_capacity(7 * 8);
            let mut mlog = Vec::with_capacity(7 * 8);
            for _ in 0..7 {
                let m = self.gpu.mtp_draft_step(
                    &mut self.pool, self.mtp_cur_hidden.as_ref().unwrap(), cur_tok, dpos,
                    kc, vc, kv_stride);
                self.gpu.copy_hidden_col(cur_ptr, &m, 0);
                self.pool.release_bf16(m, h);
                cur_tok = self.gpu.argmax_hidden(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap()) as i32;
                for (t, v) in self.gpu.topk_hidden_kv(&mut self.pool, self.mtp_cur_hidden.as_ref().unwrap(), 8) {
                    mids.push(t); mlog.push(v);
                }
                dpos += 1;
            }
            cov_mtp_sets = Some((mids, mlog));
            // Re-prime the head over the REAL accepted span (mirror of mtp_lane_step's re-prime:
            // col 0 from the cursor hidden, then the verify's real hiddens), so the next step's
            // chain conditions on the committed prefix exactly as an MTP lane would.
            let n_rp = nacc + 1;
            let rp_hidden = self.pool.get_bf16(h * n_rp);
            let rp_ptr = *rp_hidden.device_ptr();
            let mut rp_toks: Vec<u32> = Vec::with_capacity(n_rp);
            self.gpu.copy_hidden_col(rp_ptr, &self.mtp_h_prev[phys], 0);
            rp_toks.push(committed_tok);
            self.gpu.copy_hidden_cols(rp_ptr + (h * 2) as u64, &vout, 0, nacc);
            for k in 1..=nacc { rp_toks.push(drafts[k - 1]); }
            self.gpu.mtp_reprime(&mut self.pool, &rp_hidden, &rp_toks, main_pos - 1, kc, vc, kv_stride);
            self.pool.release_bf16(rp_hidden, h * n_rp);
            // Cursor hidden ← hidden at the last accepted column (the next step's h_prev twin).
            self.gpu.copy_hidden_col(*self.mtp_h_prev[phys].device_ptr(), &vout, nacc);
        }

        // ---- Inject the accepted span's taps into the ring (cols [0, nacc+1) of the sink
        // ---- staging — written by the verify's capture D2Ds, stream-ordered before the
        // ---- verify's own dtoh readback, which this call follows). nprev = main_pos+nacc+1.
        {
            let df2 = self.df2.as_mut().unwrap();
            // S5F3 fix: copy the sink's LIVE staging into the round before the inject (the
            // attach-time deep copy never saw the trunk's captures — the draft-parity root
            // cause). The verify synced the capture before returning.
            df2.sync_staging_from_sink().expect("df2 sync staging");
            df2.inject_dev(nacc + 1, None).expect("df2 inject_dev");
        }
        self.pool.release_bf16(vout, h * 8);


        // ---- Emit accepted drafts + bonus, honoring EOS and max_new (the mtp_lane_step rule:
        // ---- the budget check belongs on the drafts too). ----
        let mut new_toks: Vec<u32> = Vec::with_capacity(nacc + 1);
        let mut hit_eos = false;
        for &d in drafts.iter().take(nacc) {
            if generated + new_toks.len() >= max_new { break; }
            new_toks.push(d);
            if eos.contains(&d) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; break; }
        }
        if !hit_eos && generated + new_toks.len() < max_new {
            new_toks.push(bonus);
            if eos.contains(&bonus) && !ignore_eos && generated + new_toks.len() >= min_new { hit_eos = true; }
        }
        let emit_count = new_toks.len();
        let finished = hit_eos || generated + emit_count >= max_new;

        {
            let cache = &mut self.slot_cache[phys];
            cache.push(committed_tok);
            cache.extend_from_slice(&drafts[..nacc]);
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
                lane.last_tok = bonus;
                lane.pos = main_pos + nacc + 1;
            } else {
                lane.last_tok = *new_toks.last().unwrap_or(&committed_tok);
                lane.pos = main_pos + emit_count;
            }
        }

        // ---- S5F3 dump record (dump-only; the staging readback is stream-ordered after the
        // ---- verify's capture D2Ds — the verify synced before returning). ----
        if let Some(d) = self.step_dump.as_mut() {
            let staging: Vec<half::bf16> = self.df2_sink.as_ref()
                .and_then(|s| self.gpu.dev().dtoh_sync_copy(&s.staging).ok())
                .unwrap_or_default();
            d.record_span(main_pos, 8, &staging);
            let ck = crate::dflash2::stepdump::StepDump::tap_checksums(&staging);
            // S5F3 deep-copy-clone check: the ROUND's staging vs the SINK's staging (step 0)
            if self.df2_stat_steps == 0 {
                if let Some(df2) = self.df2.as_mut() {
                    if let Ok(rs) = df2.dump_staging() {
                        let sink_f: Vec<f32> = staging.iter().map(|x| x.to_f32()).collect();
                        let mut nd = 0.0f64; let mut dd = 0.0f64;
                        for i in 0..rs.len().min(sink_f.len()) {
                            let a = rs[i] as f64; let b = sink_f[i] as f64;
                            nd += (a - b) * (a - b); dd += b * b;
                        }
                        eprintln!("[df2-dump] STAGING CHECK step0: round-vs-sink relL2 {:.4e} (round[0..4]={:?} sink[0..4]={:?})",
                                 (nd / dd.max(1e-30)).sqrt(), &rs[..4], &sink_f[..4]);
                    }
                }
            }
            let rec = crate::dflash2::stepdump::Df2StepRec {
                step: self.df2_stat_steps, pos: main_pos, committed: committed_tok, greedy: true, realq: false,
                drafts: drafts.clone(), p_draft: Vec::new(),
                resid: preds[..7.min(preds.len())].to_vec(), bonus,
                nacc, emitted: emit_count, q_rows: vec![1.0; crate::dflash2::levels()],
                candidates: full_out.as_ref().map(|o| o.candidates.clone()).unwrap_or_default(),
                cand_q: Vec::new(),
                unary: full_out.as_ref().map(|o| o.unary.clone()).unwrap_or_default(),
                scores: full_out.as_ref().map(|o| o.scores.clone()).unwrap_or_default(),
                top20: Vec::new(), tap_ck: ck,
                hfinal_written: full_out.is_some() && self.df2_stat_steps
                    < crate::dflash2::stepdump::RAW_STEPS as u64,
                // PLAN/25 Phase 0: the target-side coverage readback + the logging-only MTP
                // chain's top-8 per position (the union instrument).
                tgt_ids: cov_tgt.as_ref().map(|(a, ..)| a.clone()).unwrap_or_default(),
                tgt_logit: cov_tgt.as_ref().map(|(_, b, ..)| b.clone()).unwrap_or_default(),
                tgt_p1: cov_tgt.as_ref().map(|(.., c, _)| c.clone()).unwrap_or_default(),
                tgt_margin: cov_tgt.as_ref().map(|(.., e)| e.clone()).unwrap_or_default(),
                mtp_ids: cov_mtp_sets.as_ref().map(|(a, _)| a.clone()).unwrap_or_default(),
                mtp_logit: cov_mtp_sets.as_ref().map(|(_, b)| b.clone()).unwrap_or_default(),
            };
            let hf = full_out.as_ref().map(|o| o.h_final.as_slice());
            d.record_df2(&rec, hf);
            // S5F3 ring rows (first RING_STEPS steps): ctx rows near C + the injected span.
            if self.df2_stat_steps < crate::dflash2::stepdump::RING_STEPS as u64 {
                let df2 = self.df2.as_mut().unwrap();
                let lo = main_pos.saturating_sub(2);
                let mut rk = Vec::new();
                let mut rv = Vec::new();
                for li in 0..crate::dflash2::n_layers() {
                    if let Ok((k, v)) = df2.dump_ring_rows(li, lo, main_pos + 8) {
                        rk.push(k); rv.push(v);
                    }
                }
                if rk.len() == crate::dflash2::n_layers() { d.record_ring_rows(self.df2_stat_steps, &rk, &rv); }
            }
        }

        // ---- Telemetry (df2-specific; the MTP policy's hazard curve stays MTP-only). ----
        self.rec_step(SpecStepRec {
            greedy: true, pos: main_pos as u32, drafts: drafts.len() as u32, nacc: nacc as u32,
            emitted: emit_count as u32, round_ms, verify_ms,
            step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
        });
        if crate::opts::var(crate::opt!("df2-step-log")).is_ok() {
            let step_ms = step_t0.elapsed().as_secs_f32() * 1e3;
            eprintln!("[df2-step] pos={main_pos} nacc={nacc} emitted={emit_count} committed={committed_tok} \
                       drafts={:?} preds={:?} round={round_ms:.1}ms verify={verify_ms:.1}ms step={step_ms:.1}ms",
                      &drafts[..drafts.len().min(4)], &preds[..preds.len().min(4)]);
        }
        // DF2_CARRY: the ring now reflects THIS slot's committed prefix up to the lane's frontier
        // (`nprev` = the whole committed sequence: prompt + emitted). Record it so the next request
        // that reuses this slot's prefix can carry the ring. Twin of the DSpark
        // `slot_dspark_len` write in `dspark_lane_step`.
        if let Some(d) = self.df2.as_mut() { d.note_ring(phys); }
        self.df2_stat_steps += 1;
        self.df2_stat_drafts += drafts.len() as u64;
        self.df2_stat_accepted += nacc as u64;
        self.df2_stat_emitted += emit_count as u64;
        if self.df2_stat_steps % 50 == 0 {
            let acc = if self.df2_stat_drafts > 0 {
                self.df2_stat_accepted as f64 / self.df2_stat_drafts as f64 * 100.0
            } else { 0.0 };
            eprintln!("[df2] steps={} drafts={} accepted={:.1}% emitted={} tok/step={:.3}",
                      self.df2_stat_steps, self.df2_stat_drafts, acc,
                      self.df2_stat_emitted,
                      self.df2_stat_emitted as f64 / self.df2_stat_steps as f64);
        }
        // FIX1 #76 (B31's lesson, applied to the DF2 chain): publish EVERY step. The old
        // publish-inside-the-%50-guard left velogb10_spec_* at 0 for any request that finished
        // under 50 steps — exactly the owner-posture report (DFlash2 row-9 flags, b==1, ~30
        // steps). A handful of Relaxed stores against a ~70 ms step; the log cadence is unchanged.
        if self.df2_stat_steps != self.df2_stat_published {
            self.df2_stat_published = self.df2_stat_steps;
            crate::tel::publish_df2(self.df2_stat_steps, self.df2_stat_drafts,
                                    self.df2_stat_accepted, self.df2_stat_emitted);
        }
        finished
    }

    /// One DFlash2 speculative step for a SAMPLING lane (temperature > 0): the SAME 7 greedy
    /// drafts from the round, verified with `verify_forward_sample` under qprobs = 1.0 (S3T2 b1 —
    /// the engine's existing sampled-lane pattern; the selector-temperature lever is CLOSED, so no
    /// sampled-selector path is built), accepted by speculative rejection sampling (Leviathan et
    /// al. 2023: accept with prob min(1, p(x)/q(x)) = p(x) since q = 1; else emit the residual).
    /// Distribution-exact by construction; gated by the DFlash2 chi-square probe.
    pub(super) fn df2_lane_step_sample(&mut self, i: usize) -> bool {
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let snapshot = self.mtp_snapshot_slot;
        let kv_stride = self.kv_stride;

        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
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

        // ---- Draft: the S4F round (same as the greedy lane). ----
        let step_t0 = std::time::Instant::now();
        let dump_on = self.cov_trace;   // op-sequence gate: must match across TP ranks
        let (drafts, full_out): (Vec<u32>, Option<crate::dflash2::round::Df2RoundOut>) = {
            let df2 = self.df2.as_mut().unwrap();
            assert_eq!(df2.nprev(), main_pos,
                "df2 ring nprev {} != lane pos {} (ring stale or unprimed)", df2.nprev(), main_pos);
            if dump_on {
                df2.refresh_block_pos().expect("df2 refresh_block_pos");
                let o = df2.draft_round_full(committed_tok).expect("df2 draft_round_full");
                (o.tokens.clone(), Some(o))
            } else if df2.round_graph.is_some() {
                (df2.draft_round_graph(committed_tok).expect("df2 draft_round_graph"), None)
            } else {
                df2.refresh_block_pos().expect("df2 refresh_block_pos");
                (df2.draft_round_dev(committed_tok).expect("df2 draft_round"), None)
            }
        };
        let round_ms = step_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Build verify input + per-column device seeds (domain-separated). ----
        let mut verify_input = vec![committed_tok];
        verify_input.extend(drafts.iter().copied());
        let verify_seeds: Vec<u32> =
            (0..crate::dflash2::block()).map(|j| rng_u32(step_key, RNG_DOM_VERIFY, j)).collect();
        let qprobs: Vec<f32> = vec![1.0; crate::dflash2::levels()];   // greedy draft = point mass (S3T2 b1)

        // ---- Verify with stochastic output (the eager spec_verify_b path). ----
        let penalty = self.make_penalty(i, rep_pen, presence_pen, freq_pen, has_penalty);
        let verify_t0 = std::time::Instant::now();
        let mut t20: Vec<u64> = Vec::new();
        let (vsample, vout) = self.gpu.verify_forward_sample(
            &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos,
            Some(snapshot), penalty,
            &drafts, &qprobs, temperature, top_k, top_p, &verify_seeds,
            if self.cov_trace { Some(&mut t20) } else { None });
        let verify_ms = verify_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Speculative rejection sampling accept loop (q = 1 ⇒ ratio = p(x)). ----
        let mut nacc = 0usize;
        let mut emitted: Vec<u32> = Vec::with_capacity(8);
        let mut rejected = false;
        for j in 0..drafts.len() {
            let ru = rng_uniform(step_key, RNG_DOM_ACCEPT, j);
            if ru < vsample.p_of_draft[j] {
                emitted.push(drafts[j]);
                nacc += 1;
            } else {
                emitted.push(vsample.resid_tok[j]);
                rejected = true;
                break;
            }
        }
        if !rejected { emitted.push(vsample.bonus_tok); }

        // ---- GDN rollback on partial reject. ----
        if nacc + 1 != crate::dflash2::block() {
            self.gpu.copy_gdn_slot(&self.state, snapshot + nacc, phys);
        }
        // ---- Inject the accepted span's taps (same as the greedy lane). ----
        {
            let df2 = self.df2.as_mut().unwrap();
            // S5F3 fix: copy the sink's LIVE staging into the round before the inject (the
            // attach-time deep copy never saw the trunk's captures — the draft-parity root
            // cause). The verify synced the capture before returning.
            df2.sync_staging_from_sink().expect("df2 sync staging");
            df2.inject_dev(nacc + 1, None).expect("df2 inject_dev");
        }
        self.pool.release_bf16(vout, h * 8);


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
            lane.seed = splitmix64(step_key);
            lane.last_tok = *to_emit.last().unwrap_or(&committed_tok);
            lane.pos = if !finished { main_pos + nacc + 1 } else { main_pos + emit_count };
        }

        // ---- S5F3 dump record (dump-only; the staging + t20 readbacks are stream-ordered
        // ---- after the verify — it synced before returning). ----
        if let Some(d) = self.step_dump.as_mut() {
            let staging: Vec<half::bf16> = self.df2_sink.as_ref()
                .and_then(|s| self.gpu.dev().dtoh_sync_copy(&s.staging).ok())
                .unwrap_or_default();
            d.record_span(main_pos, 8, &staging);
            let ck = crate::dflash2::stepdump::StepDump::tap_checksums(&staging);
            if self.df2_stat_steps == 0 {
                if let Some(df2) = self.df2.as_mut() {
                    if let Ok(rs) = df2.dump_staging() {
                        let sink_f: Vec<f32> = staging.iter().map(|x| x.to_f32()).collect();
                        let mut nd = 0.0f64; let mut dd = 0.0f64;
                        for i in 0..rs.len().min(sink_f.len()) {
                            let a = rs[i] as f64; let b = sink_f[i] as f64;
                            nd += (a - b) * (a - b); dd += b * b;
                        }
                        eprintln!("[df2-dump] STAGING CHECK step0 (sample lane): round-vs-sink relL2 {:.4e} (round[0..4]={:?} sink[0..4]={:?})",
                                 (nd / dd.max(1e-30)).sqrt(), &rs[..4], &sink_f[..4]);
                    }
                }
            }
            let rec = crate::dflash2::stepdump::Df2StepRec {
                step: self.df2_stat_steps, pos: main_pos, committed: committed_tok, greedy: false, realq: false,
                drafts: drafts.clone(), p_draft: vsample.p_of_draft.clone(),
                resid: vsample.resid_tok.clone(), bonus: vsample.bonus_tok,
                nacc, emitted: emit_count, q_rows: vec![1.0; crate::dflash2::levels()],
                candidates: full_out.as_ref().map(|o| o.candidates.clone()).unwrap_or_default(),
                cand_q: Vec::new(),
                unary: full_out.as_ref().map(|o| o.unary.clone()).unwrap_or_default(),
                scores: full_out.as_ref().map(|o| o.scores.clone()).unwrap_or_default(),
                top20: std::mem::take(&mut t20), tap_ck: ck,
                hfinal_written: full_out.is_some() && self.df2_stat_steps
                    < crate::dflash2::stepdump::RAW_STEPS as u64,
                // PLAN/25 Phase 0 fields: the greedy-lane coverage readback and the logging-only
                // MTP chain don't run on the sampled lanes (the sample lane's distribution-side
                // table is `top20`; the temp-0.7 union uses the per-source marginals).
                tgt_ids: Vec::new(), tgt_logit: Vec::new(), tgt_p1: Vec::new(),
                tgt_margin: Vec::new(), mtp_ids: Vec::new(), mtp_logit: Vec::new(),
            };
            let hf = full_out.as_ref().map(|o| o.h_final.as_slice());
            d.record_df2(&rec, hf);
            // S5F3 ring rows (first RING_STEPS steps): ctx rows near C + the injected span.
            if self.df2_stat_steps < crate::dflash2::stepdump::RING_STEPS as u64 {
                let df2 = self.df2.as_mut().unwrap();
                let lo = main_pos.saturating_sub(2);
                let mut rk = Vec::new();
                let mut rv = Vec::new();
                for li in 0..crate::dflash2::n_layers() {
                    if let Ok((k, v)) = df2.dump_ring_rows(li, lo, main_pos + 8) {
                        rk.push(k); rv.push(v);
                    }
                }
                if rk.len() == crate::dflash2::n_layers() { d.record_ring_rows(self.df2_stat_steps, &rk, &rv); }
            }
        }

        self.rec_step(SpecStepRec {
            greedy: false, pos: main_pos as u32, drafts: drafts.len() as u32, nacc: nacc as u32,
            emitted: emit_count as u32, round_ms, verify_ms,
            step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
        });
        // DF2_CARRY: bind the ring to this slot's frontier (twin of the greedy lane's write).
        if let Some(d) = self.df2.as_mut() { d.note_ring(phys); }
        self.df2_stat_steps += 1;
        self.df2_stat_drafts += drafts.len() as u64;
        self.df2_stat_accepted += nacc as u64;
        self.df2_stat_emitted += emit_count as u64;

        // FIX1 #76 (sampled lanes): the greedy lane's watermark publish, applied here too. The
        // owner-posture request class (sampled + thinking, b==1) rides THIS function — before
        // this, its rounds bumped the df2 stats but never reached /metrics, so a DF2 chain
        // serving every step still read as zero engagement (ISSUES #76's exact case).
        if self.df2_stat_steps != self.df2_stat_published {
            self.df2_stat_published = self.df2_stat_steps;
            crate::tel::publish_df2(self.df2_stat_steps, self.df2_stat_drafts,
                                    self.df2_stat_accepted, self.df2_stat_emitted);
        }
        // P2 Phase A finding: the sampled-greedy lane (temp>0 + Code) had NO [df2-step] line —
        // the step tables silently dropped every code-class step. Same format as the greedy lane.
        if crate::opts::var(crate::opt!("df2-step-log")).is_ok() {
            eprintln!("[df2-step] pos={main_pos} nacc={nacc} emitted={emit_count} committed={committed_tok} \
                       round={round_ms:.1}ms verify={verify_ms:.1}ms step={:.1}ms",
                      step_t0.elapsed().as_secs_f32() * 1e3);
        }
        finished
    }

    /// S5F2 L2 — the REAL-Q DFlash2 lane (--spec-source dflash2-rq): the sampled selector path
    /// (`Df2Round::draft_round_dev_sample` — the SGLang `CandidateSelector.sample_path`
    /// multinomial at the request temperature) verified under the real-q rejection-sampling
    /// criterion `u·q < p` (q = the drawn candidate's selector probability) with the EXACT
    /// relu(p−q) residual (the SGLang `speculative_sampling_classic_kernel` semantics). With a
    /// valid proposal q and the exact residual the emitted distribution is exactly the target p
    /// — the L2 chi-square gate's contract. Distribution-exact by construction; gated by
    /// `--bench-df2-sample-realq`.
    pub(super) fn df2_lane_step_sample_rq(&mut self, i: usize) -> bool {
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let snapshot = self.mtp_snapshot_slot;
        let kv_stride = self.kv_stride;

        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
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
        let step_key = self.lanes[i].as_ref().unwrap().seed;

        // ---- Draft: the SAMPLED selector path (eager round; per-position selector seeds). ----
        let step_t0 = std::time::Instant::now();
        let dump_on = self.cov_trace;   // op-sequence gate: must match across TP ranks
        let sel_seeds: Vec<u32> = (0..crate::dflash2::levels()).map(|j| rng_u32(step_key, RNG_DOM_DF2_SEL, j)).collect();
        let (drafts, q_rows, cand_tok, cand_q) = {
            let df2 = self.df2.as_mut().unwrap();
            assert_eq!(df2.nprev(), main_pos,
                "df2 ring nprev {} != lane pos {} (ring stale or unprimed)", df2.nprev(), main_pos);
            // S5F4 fix (the rq-lane sibling of R14): refresh the block RoPE positions for THIS
            // step. The greedy/sample lanes do this before every round; the rq lane missed it —
            // every round ran with the STALE pos_blk/cos8/sin8 left by capture_round_graph
            // (max_c..max_c+8), scrambling the block attention against the correctly-rotated
            // ctx rows (step-0 top16 differed from the greedy lane's on the SAME ring/anchor),
            // flattening the sampled chain's realized q (0.47 vs SGLang 0.80) and capping τ at
            // ~1.23. The chi-square gate stayed green because ITS path refreshes (main.rs
            // run_bench_df2_sample_realq) — the gate and the serving lane diverged.
            df2.refresh_block_pos().expect("df2 refresh_block_pos");
            let out = df2.draft_round_dev_sample(committed_tok, &sel_seeds, temperature)
                .expect("df2 sampled draft round");
            (out.tokens, out.q_rows, out.cand_tok, out.cand_q)
        };
        let round_ms = step_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Build verify input + per-column device seeds. ----
        let mut verify_input = vec![committed_tok];
        verify_input.extend(drafts.iter().copied());
        let verify_seeds: Vec<u32> =
            (0..crate::dflash2::block()).map(|j| rng_u32(step_key, RNG_DOM_VERIFY, j)).collect();

        // ---- Verify with the real-q kernel (exact relu(p-q) residual). ----
        let penalty = self.make_penalty(i, rep_pen, presence_pen, freq_pen, has_penalty);
        let verify_t0 = std::time::Instant::now();
        let mut t20: Vec<u64> = Vec::new();
        let (vsample, vout) = self.gpu.verify_forward_sample_rq(
            &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos,
            Some(snapshot), penalty,
            &drafts, &cand_tok, &cand_q, temperature, top_k, top_p, &verify_seeds,
            if self.cov_trace { Some(&mut t20) } else { None });
        let verify_ms = verify_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Real-q rejection sampling: accept iff u*q < p (min(1, p/q) ratio). ----
        let mut nacc = 0usize;
        let mut emitted: Vec<u32> = Vec::with_capacity(8);
        let mut rejected = false;
        let eps = 1e-12f32;
        for j in 0..drafts.len() {
            let q = q_rows[j];
            let ratio = if q < eps { 1.0 } else { (vsample.p_of_draft[j] / q).min(1.0) };
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
        if !rejected { emitted.push(vsample.bonus_tok); }

        // ---- GDN rollback on partial reject. ----
        if nacc + 1 != crate::dflash2::block() {
            self.gpu.copy_gdn_slot(&self.state, snapshot + nacc, phys);
        }
        // ---- Inject the accepted span's taps (same as the other lanes). ----
        {
            let df2 = self.df2.as_mut().unwrap();
            // S5F3 fix: copy the sink's LIVE staging into the round before the inject (the
            // attach-time deep copy never saw the trunk's captures — the draft-parity root
            // cause). The verify synced the capture before returning.
            df2.sync_staging_from_sink().expect("df2 sync staging");
            df2.inject_dev(nacc + 1, None).expect("df2 inject_dev");
        }
        self.pool.release_bf16(vout, h * 8);


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
            lane.seed = splitmix64(step_key);
            lane.last_tok = *to_emit.last().unwrap_or(&committed_tok);
            lane.pos = if !finished { main_pos + nacc + 1 } else { main_pos + emit_count };
        }

        // ---- S5F3 dump record (dump-only; staging + t20 readbacks stream-ordered after the
        // ---- verify — it synced before returning). The SAMPLED selector's q_rows + candidate
        // ---- table are the exact SGLang `sample_path` analog. ----
        if let Some(d) = self.step_dump.as_mut() {
            let staging: Vec<half::bf16> = self.df2_sink.as_ref()
                .and_then(|s| self.gpu.dev().dtoh_sync_copy(&s.staging).ok())
                .unwrap_or_default();
            d.record_span(main_pos, 8, &staging);
            let ck = crate::dflash2::stepdump::StepDump::tap_checksums(&staging);
            let rec = crate::dflash2::stepdump::Df2StepRec {
                step: self.df2_stat_steps, pos: main_pos, committed: committed_tok, greedy: false, realq: true,
                drafts: drafts.clone(), p_draft: vsample.p_of_draft.clone(),
                resid: vsample.resid_tok.clone(), bonus: vsample.bonus_tok,
                nacc, emitted: emit_count, q_rows,
                candidates: cand_tok, cand_q, unary: Vec::new(), scores: Vec::new(),
                top20: std::mem::take(&mut t20), tap_ck: ck, hfinal_written: false,
                // PLAN/25 Phase 0 fields (see the greedy-lane record): not captured on rq lanes.
                tgt_ids: Vec::new(), tgt_logit: Vec::new(), tgt_p1: Vec::new(),
                tgt_margin: Vec::new(), mtp_ids: Vec::new(), mtp_logit: Vec::new(),
            };
            d.record_df2(&rec, None);
            // S5F3 ring rows (first RING_STEPS steps).
            if self.df2_stat_steps < crate::dflash2::stepdump::RING_STEPS as u64 {
                let df2 = self.df2.as_mut().unwrap();
                let lo = main_pos.saturating_sub(2);
                let mut rk = Vec::new();
                let mut rv = Vec::new();
                for li in 0..crate::dflash2::n_layers() {
                    if let Ok((k, v)) = df2.dump_ring_rows(li, lo, main_pos + 8) {
                        rk.push(k); rv.push(v);
                    }
                }
                if rk.len() == crate::dflash2::n_layers() { d.record_ring_rows(self.df2_stat_steps, &rk, &rv); }
            }
        }

        self.rec_step(SpecStepRec {
            greedy: false, pos: main_pos as u32, drafts: drafts.len() as u32, nacc: nacc as u32,
            emitted: emit_count as u32, round_ms, verify_ms,
            step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
        });
        // DF2_CARRY: bind the ring to this slot's frontier (twin of the greedy lane's write).
        if let Some(d) = self.df2.as_mut() { d.note_ring(phys); }
        self.df2_stat_steps += 1;
        self.df2_stat_drafts += drafts.len() as u64;
        self.df2_stat_accepted += nacc as u64;
        self.df2_stat_emitted += emit_count as u64;

        // FIX1 #76 (sampled lanes): the greedy lane's watermark publish, applied here too. The
        // owner-posture request class (sampled + thinking, b==1) rides THIS function — before
        // this, its rounds bumped the df2 stats but never reached /metrics, so a DF2 chain
        // serving every step still read as zero engagement (ISSUES #76's exact case).
        if self.df2_stat_steps != self.df2_stat_published {
            self.df2_stat_published = self.df2_stat_steps;
            crate::tel::publish_df2(self.df2_stat_steps, self.df2_stat_drafts,
                                    self.df2_stat_accepted, self.df2_stat_emitted);
        }
        if crate::opts::var(crate::opt!("df2-step-log")).is_ok() {
            eprintln!("[df2-step] pos={main_pos} nacc={nacc} emitted={emit_count} committed={committed_tok} \
                       round={round_ms:.1}ms verify={verify_ms:.1}ms step={:.1}ms",
                      step_t0.elapsed().as_secs_f32() * 1e3);
        }
        finished
    }
}
