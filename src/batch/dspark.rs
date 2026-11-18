//! DFlash and DSpark lanes: `install_dflash_lane`, `dflash_prime`, the DFlash and DSpark lane steps (greedy and sampled).
//! Another `impl BatchScheduler` block (file map in `mod.rs`); the methods keep their exact text.
//! Binding invariants: greedy MTP stays bitwise lossless; spec gating is per source (AGENTS.md section 4).

use super::*;

impl BatchScheduler {
    /// P14 — install the DFlash v1 lane. Loads the artifact, allocates the lane's ctx feature and
    /// arms the trunk's wide capture surface (the tap sink reads the same geometry from the
    /// artifact's config.json — no geometry is assumed here).
    pub fn install_dflash_lane(&mut self, dir: &str) -> anyhow::Result<()> {
        use crate::dflash::{DflashDrafter, DflashKv};
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let stride = self.kv_stride;
        let draf = DflashDrafter::load_from_dir(std::path::Path::new(dir), stride + 16)?;
        let (nctx, block, mask) = (draf.nctx, draf.block, draf.mask_token_id);
        let kv = DflashKv::new(&draf, stride + crate::dflash::MAX_BLOCK);
        let pool = crate::gpu::Pool::new(draf.dev.clone());
        let feat = self.gpu.dev().alloc_zeros::<half::bf16>(nctx * h * stride)?;
        self.gpu.arm_dflash_wide(dir, stride)?;
        eprintln!("[dflash] v1 lane installed: {} layers, block {block}, {nctx} taps, mask {mask}, \
                   ctx stride {stride}", draf.layers.len());
        self.dflash = Some(DflashLane { draf, kv, pool, feat, nctx, block, mask, row0: 0, ncols: 0 });
        Ok(())
    }

    /// P14 — prime the v1 lane's ctx feature from the prompt prefill (the trunk's wide capture wrote
    /// every prompt position at its absolute column). The drafter's KV cache is appended lazily by
    /// the first draft step, so priming is a D2D plus the frontier bookkeeping.
    pub(super) fn dflash_prime(&mut self, plen: usize) {
        if let Some(d) = self.dflash.as_mut() {
            let (feat, nctx) = (&d.feat, d.nctx);
            self.gpu.dflash_feat_copy_at(feat, 0, 0, plen);
            drop(nctx);
            d.row0 = 0;
            d.ncols = plen;
        }
    }

    /// P14 — one DFlash v1 lane step: draft a `block` from the target's hidden states, verify
    /// `[committed, drafts...]` with the trunk, accept the longest agreeing prefix plus the trunk's
    /// own bonus, then hand the verified span's taps to the next step's ctx append.
    ///
    /// Every emitted token is a trunk argmax (`verify_forward`'s preds are the plain-decode argmax
    /// per column — invariant 4), so the emitted chain is plain greedy BY CONSTRUCTION: the MTP
    /// chain's losslessness argument, with a block-wide verify instead of a chain.
    pub(super) fn dflash_lane_step(&mut self, i: usize) -> bool {
        crate::tel::note_step();
        let h = self.gpu.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let phys = self.lanes[i].as_ref().unwrap().phys;
        let snapshot = self.mtp_snapshot_slot;
        let kv_stride = self.kv_stride;
        let committed_tok = self.lanes[i].as_ref().unwrap().last_tok;
        let main_pos = self.lanes[i].as_ref().unwrap().pos;
        let generated = self.lanes[i].as_ref().unwrap().generated;
        let max_new = self.lanes[i].as_ref().unwrap().max_new;
        let eos = self.eos.clone();
        let (ignore_eos, min_new) = { let l = self.lanes[i].as_ref().unwrap(); (l.ignore_eos, l.min_new) };
        let (rep_pen, presence_pen, freq_pen, has_penalty) = {
            let l = self.lanes[i].as_ref().unwrap();
            (l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.has_penalty())
        };
        let history: Vec<u32> = self.lanes[i].as_ref().unwrap().history.clone();

        let step_t0 = std::time::Instant::now();
        let block = self.dflash.as_ref().unwrap().block;
        // ---- 1. draft: append the pending ctx span, block forward, then the TARGET's own head over
        // the block and argmax per column (the lane is greedy-only). ----
        let drafts: Vec<u32> = {
            let d = self.dflash.as_mut().unwrap();
            assert_eq!(d.row0 + d.ncols, main_pos,
                "dflash ctx frontier {} != lane pos {main_pos} (unprimed or stale lane)",
                d.row0 + d.ncols);
            let mut bt = Vec::with_capacity(block);
            bt.push(committed_tok);
            bt.extend(std::iter::repeat(d.mask).take(block - 1));
            self.gpu.sync_stream();   // the tap D2Ds run on the engine stream
            let DflashLane { draf, kv, pool, feat, nctx, row0, ncols, .. } = d;
            self.gpu.dflash_draft(draf, kv, pool, feat, *nctx, h, *row0, *ncols, *row0, &bt)
        };
        let round_ms = step_t0.elapsed().as_secs_f32() * 1e3;
        // ---- 2. verify [committed, drafts...] at [main_pos, main_pos + block) ----
        let mut verify_input = Vec::with_capacity(block);
        verify_input.push(committed_tok);
        verify_input.extend(drafts.iter().copied());
        // The lane's verify width is the artifact's whole block (`dflash_draft` drops the block's
        // anchor row, so 1 + (block-1) columns), which must stay inside MAX_VERIFY: past it the
        // verify stops being batch-invariant AND walks a per-layer GDN checkpoint band sized
        // MAX_VERIFY (`verify_forward_core_topo` asserts both — this one names the lane).
        assert_eq!(verify_input.len(), block, "dflash verify width != block");
        assert!(block <= crate::gpu::MAX_VERIFY,
                "dflash block {block} exceeds MAX_VERIFY {}: the verify would fall to the prefill \
                 dequant path and the GDN checkpoint band would overflow", crate::gpu::MAX_VERIFY);
        let penalty = self.make_penalty(i, rep_pen, presence_pen, freq_pen, has_penalty);
        let verify_t0 = std::time::Instant::now();
        let (preds, vout) = self.gpu.verify_forward(&mut self.pool, &verify_input, &mut self.state,
                                                    phys, kv_stride, main_pos, Some(snapshot), penalty);
        let verify_ms = verify_t0.elapsed().as_secs_f32() * 1e3;
        // ---- 3. accept the longest agreeing prefix; the bonus is the trunk's own argmax ----
        let mut nacc = 0usize;
        while nacc < drafts.len() && preds[nacc] == drafts[nacc] { nacc += 1; }
        let bonus = preds[nacc];
        if nacc + 1 != block { self.gpu.copy_gdn_slot(&self.state, snapshot + nacc, phys); }
        // ---- 4. the verified span's taps are the next ctx append ----
        // The verify captured them into the POSITION-INDEPENDENT ring staging — its D2D lives inside
        // a captured CUDA graph, so it cannot address an absolute column (see the tap site in
        // `verify_forward_core_topo`). This host-issued D2D moves the ACCEPTED span
        // `[0, nacc+1)` to its absolute ctx column. It is the P14 twin of S5F3's
        // `sync_staging_from_sink` before `inject_dev`: without it the drafter conditioned every
        // generated position on ctx columns no forward had written, which is exactly how the DF2
        // lane's τ collapsed 4.06 → 1.27 with every gate still green (PLAN/B8_S5F3_REPORT §R3).
        {
            let d = self.dflash.as_ref().unwrap();
            self.gpu.dflash_tap_to_feature(&d.feat, main_pos, nacc + 1);
        }
        {
            let d = self.dflash.as_mut().unwrap();
            d.row0 = main_pos;
            d.ncols = nacc + 1;
        }
        // ---- 5. emit (the mtp/df2 rule: EOS + max_new on the drafts too) ----
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
        self.pool.release_bf16(vout, h * block);
        self.rec_step(SpecStepRec {
            greedy: true, pos: main_pos as u32, drafts: drafts.len() as u32, nacc: nacc as u32,
            emitted: emit_count as u32, round_ms, verify_ms,
            step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
        });
        if crate::opts::var(crate::opt!("dflash-step-log")).is_ok() {
            // Field layout mirrors the DF2 line so `accept_gate.py dflash` can reuse its parser:
            // pos / nacc / emitted / committed / round / verify / step. `preds` rides along for the
            // same reason the DF2 line carries it — the FIRST thing to check when acceptance
            // disappoints is the drafts-vs-preds pattern (an off-by-one indexing error shows up as
            // `drafts[j] == preds[j-1]` and looks exactly like "the drafter is weak").
            eprintln!("[dflash-step] pos={main_pos} nacc={nacc} emitted={emit_count} \
                       committed={committed_tok} drafts={:?} preds={:?} round={round_ms:.1}ms \
                       verify={verify_ms:.1}ms step={:.1}ms",
                      &drafts[..drafts.len().min(6)], &preds[..preds.len().min(6)],
                      step_t0.elapsed().as_secs_f32() * 1e3);
        }
        // ---- P14 telemetry: the block lane was INVISIBLE at /health (HANDOFF 2026-09-12 made
        // per-window decode telemetry a standing requirement, and `accept_gate.py` a mandatory gate
        // for any draft-path diff). Cumulative since boot, exactly like the `[mtp]`/`[df2]` meters.
        // accept@k here is COLUMN-indexed: n[k] counts the steps that reached column k, a[k] the
        // steps where column k matched the trunk's argmax (k = 0..block-2 = the 15 proposals).
        self.dflash_stat_steps += 1;
        self.dflash_stat_drafts += drafts.len() as u64;
        self.dflash_stat_accepted += nacc as u64;
        self.dflash_stat_emitted += emit_count as u64;
        for k in 0..drafts.len().min(crate::dflash::MAX_BLOCK) {
            if nacc >= k { self.dflash_acc_n[k] += 1; }
            if nacc > k { self.dflash_acc_a[k] += 1; }
        }
        // Published EVERY step (not every 50 like MTP/DF2): the block lane commits up to 16
        // tokens per step, so a short probe request is ~6-16 steps — a 50-step window would leave
        // /health mute exactly when an operator looks at it. The log meter below prints at a
        // 10-step cadence, which for a 16-token block is the same lines-per-token as MTP's 50.
        // Stack array, not a Vec: this is a decode-step hot path and tel.rs's contract is
        // "zero hot-path structure change, no lock, no allocation".
        let mut acc_pairs = [(0u64, 0u64); crate::dflash::MAX_BLOCK];
        for k in 0..drafts.len().min(crate::dflash::MAX_BLOCK) {
            acc_pairs[k] = (self.dflash_acc_a[k], self.dflash_acc_n[k]);
        }
        crate::tel::publish_dflash(
            self.dflash_stat_steps, self.dflash_stat_drafts, self.dflash_stat_accepted,
            self.dflash_stat_emitted, block, &acc_pairs[..drafts.len()]);
        if self.dflash_stat_steps % 10 == 0 {
            let acc = if self.dflash_stat_drafts > 0 {
                self.dflash_stat_accepted as f64 / self.dflash_stat_drafts as f64 * 100.0
            } else { 0.0 };
            let yield_tok = self.dflash_stat_emitted as f64 / self.dflash_stat_steps as f64;
            let mut ak = String::new();
            for k in 0..drafts.len() {
                let n = self.dflash_acc_n[k];
                if n > 0 {
                    ak.push_str(&format!(" @{}:{:.0}%", k + 1,
                        self.dflash_acc_a[k] as f64 / n as f64 * 100.0));
                }
            }
            eprintln!("[dflash] steps={} block={block} drafts={} accepted={acc:.1}% emitted={} \
                       yield={yield_tok:.2} tok/step accept@k[{ak} ]",
                      self.dflash_stat_steps, self.dflash_stat_drafts, self.dflash_stat_emitted);
        }
        finished
    }

    /// WI1: one DSpark speculative step for a GREEDY lane — the df2_lane_step shape with the
    /// DSpark round in the draft seat. Draft (7 tokens), verify width 8 on the trunk, longest
    /// prefix accept + bonus, GDN rollback on partial reject, tap inject of the accepted span.
    /// The DF2 dump/coverage instruments are DF2-only (this lane keeps the plain path + the
    /// shared SpecStepRec telemetry); the confidence head is diagnostics-only in v1 (adaptive
    /// k_verify OFF — the recipe default).
    pub(super) fn dspark_lane_step(&mut self, i: usize) -> bool {
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
        let (rep_pen, presence_pen, freq_pen, has_penalty) = {
            let l = self.lanes[i].as_ref().unwrap();
            (l.rep_penalty, l.presence_penalty, l.frequency_penalty, l.has_penalty())
        };
        let history: Vec<u32> = self.lanes[i].as_ref().unwrap().history.clone();

        // ---- Draft: the DSpark round (eager; graphs follow the DF2 pattern once oracle-proven).
        let step_t0 = std::time::Instant::now();
        static DUMP_REQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dump_steps: u64 = crate::opts::var(crate::opt!("dspark-dump-steps")).ok()
            .and_then(|v| v.parse().ok()).unwrap_or(6);
        if self.dspark_stat_steps < dump_steps {
            if let Ok(dir) = crate::opts::var(crate::opt!("dspark-dump-step0")) {
                let rseq = if self.dspark_stat_steps == 0 {
                    DUMP_REQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                } else { DUMP_REQ.load(std::sync::atomic::Ordering::Relaxed).saturating_sub(1) };
                let sdir = format!("{dir}/r{rseq}/step{}", self.dspark_stat_steps);
                let _ = std::fs::create_dir_all(&sdir);
                // step0 only: the DRAFT's own inputs — the wide prime sink's taps + plen
                if self.dspark_stat_steps == 0 {
                    let rdir = format!("{dir}/r{rseq}");
                    // the exact prompt token ids (for the offline ground-truth prefill compare)
                    let ids: Vec<u32> = self.slot_cache[phys].iter().take(main_pos).copied().collect();
                    let mut ib = Vec::with_capacity(ids.len() * 4);
                    for t in &ids { ib.extend_from_slice(&t.to_le_bytes()); }
                    let _ = std::fs::write(format!("{rdir}/prompt.bin"), &ib);
                    eprintln!("[dspark] step0 dump: prime sink present={}", self.dspark_prime.is_some());
                    if let Some(ps) = &self.dspark_prime {
                        let _ = std::fs::create_dir_all(&dir);
                        let taps: Vec<half::bf16> = self.gpu.dev().dtoh_sync_copy(&ps.taps).unwrap_or_default();
                        let cols = taps.len() / crate::dspark::TAP_CONCAT_DIM;
                        eprintln!("[dspark] step0: taps readback {} elems ({} cols)", taps.len(), cols);
                        // only the LIVE columns [0..main_pos) — the rest is stale/zero padding
                        let live = main_pos.min(cols);
                        let mut tb = Vec::with_capacity(live * crate::dspark::TAP_CONCAT_DIM * 2);
                        for t in &taps[..live * crate::dspark::TAP_CONCAT_DIM] {
                            tb.extend_from_slice(&t.to_le_bytes());
                        }
                        let _ = std::fs::write(format!("{dir}/r{rseq}/taps_wide.bin"), &tb);
                        let _ = std::fs::write(format!("{dir}/r{rseq}/plen.txt"),
                                               format!("{} {}", main_pos, live));
                    }
                }
                // every dumped step: nothing pre-draft (the logits buffer is only valid AFTER
                // the draft; the post-verify block below dumps it with the staging)
            }
        }
        let drafts: Vec<u32> = {
            let ds = self.dspark.as_mut().unwrap();
            assert_eq!(ds.nprev(), main_pos,
                "dspark ring nprev {} != lane pos {} (ring stale or unprimed)", ds.nprev(), main_pos);
            ds.draft_round_dev(committed_tok).expect("dspark draft_round")
        };
        let round_ms = step_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Verify [committed_tok, drafts...] (M=8 buckets) on the trunk at main_pos.. ----
        let mut verify_input = vec![committed_tok];
        verify_input.extend(drafts.iter().copied());
        let penalty = self.make_penalty(i, rep_pen, presence_pen, freq_pen, has_penalty);
        let verify_t0 = std::time::Instant::now();
        let (preds, vout) = self.gpu.verify_forward(
            &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos,
            Some(snapshot), penalty);
        let verify_ms = verify_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Accept longest prefix (greedy) + bonus. ----
        let mut nacc = 0usize;
        while nacc < drafts.len() && preds[nacc] == drafts[nacc] { nacc += 1; }
        let bonus = preds[nacc];

        // ---- GDN rollback on partial reject. ----
        if nacc + 1 != 8 {
            self.gpu.copy_gdn_slot(&self.state, snapshot + nacc, phys);
        }

        // ---- Inject the accepted span's taps into the DSpark ring (the S5F3 live-staging copy
        // ---- then the inject; nprev = main_pos+nacc+1). ----
        {
            let ds = self.dspark.as_mut().unwrap();
            ds.sync_staging_from_sink().expect("dspark sync staging");
            ds.inject_dev(nacc + 1).expect("dspark inject_dev");
        }
        self.pool.release_bf16(vout, h * 8);

        // ---- Emit accepted drafts + bonus (the mtp/df2 lane rule: budget check on drafts too). --
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

        // ---- Telemetry (the df2 shape; own counters so the [dspark] step log stands alone). ----
        self.rec_step(SpecStepRec {
            greedy: true, pos: main_pos as u32, drafts: drafts.len() as u32, nacc: nacc as u32,
            emitted: emit_count as u32, round_ms, verify_ms,
            step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
        });
        if crate::opts::var(crate::opt!("dspark-step-log")).is_ok() {
            let step_ms = step_t0.elapsed().as_secs_f32() * 1e3;
            eprintln!("[dspark-step] pos={main_pos} nacc={nacc} emitted={emit_count} committed={committed_tok} \
                       drafts={:?} preds={:?} round={round_ms:.1}ms verify={verify_ms:.1}ms step={step_ms:.1}ms",
                      &drafts[..drafts.len().min(4)], &preds[..preds.len().min(4)]);
        }
        // --dspark-dump-step0: for the FIRST N dspark steps (--dspark-dump-steps, default
        // 6), dump the staging taps (the round's own [8, 25600] bf16 staging = the ctx-extension
        // the next draft saw), the anchor, the drafts and the trunk preds — the offline
        // oracle-vs-trunk comparison's raw material, and the per-step reconstruction basis.
        if self.dspark_stat_steps < dump_steps {
            if let Ok(dir) = crate::opts::var(crate::opt!("dspark-dump-step0")) {
                let rseq = DUMP_REQ.load(std::sync::atomic::Ordering::Relaxed).saturating_sub(1);
                let sdir = format!("{dir}/r{rseq}/step{}", self.dspark_stat_steps);
                let ds = self.dspark.as_mut().unwrap();
                let _ = std::fs::create_dir_all(&sdir);
                let taps: Vec<half::bf16> = ds.dump_staging().unwrap_or_default();
                let mut tb = Vec::with_capacity(taps.len() * 2);
                for t in &taps { tb.extend_from_slice(&t.to_le_bytes()); }
                let _ = std::fs::write(format!("{sdir}/staging.bin"), &tb);
                let mut meta = String::new();
                meta.push_str(&format!("anchor {committed_tok}\n"));
                meta.push_str(&format!("drafts {drafts:?}\n"));
                meta.push_str(&format!("preds {preds:?}\n"));
                meta.push_str(&format!("nacc {nacc}\n"));
                meta.push_str(&format!("nprev {}\n", ds.nprev()));
                let _ = std::fs::write(format!("{sdir}/meta.txt"), &meta);
                // THIS step's draft logits ([7, VOCAB] f32, token-major) — the chain-formula
                // analysis runs on them offline (variants vs the trunk's preds).
                if let Ok(lg) = ds.dump_logits() {
                    let mut lb = Vec::with_capacity(lg.len() * 4);
                    for v in &lg { lb.extend_from_slice(&v.to_le_bytes()); }
                    let _ = std::fs::write(format!("{sdir}/logits.bin"), &lb);
                }
                eprintln!("[dspark] step{} dump written (anchor {committed_tok})", self.dspark_stat_steps);
            }
        }
        self.dspark_stat_steps += 1;
        self.dspark_stat_drafts += drafts.len() as u64;
        self.dspark_stat_accepted += nacc as u64;
        self.dspark_stat_emitted += emit_count as u64;
        if self.dspark_stat_steps % 50 == 0 {
            let acc = if self.dspark_stat_drafts > 0 {
                self.dspark_stat_accepted as f64 / self.dspark_stat_drafts as f64 * 100.0
            } else { 0.0 };
            eprintln!("[dspark] steps={} drafts={} accepted={:.1}% emitted={} tok/step={:.3}",
                      self.dspark_stat_steps, self.dspark_stat_drafts, acc,
                      self.dspark_stat_emitted,
                      self.dspark_stat_emitted as f64 / self.dspark_stat_steps.max(1) as f64);
        }
        // FIX1 #76: same watermark contract as the DF2 chain — publish every step, log every 50.
        // Before this the dspark lane accumulated stats that fed only its dump dirs: /metrics and
        // /health never saw a dspark round.
        if self.dspark_stat_steps != self.dspark_stat_published {
            self.dspark_stat_published = self.dspark_stat_steps;
            crate::tel::publish_dspark(self.dspark_stat_steps, self.dspark_stat_drafts,
                                       self.dspark_stat_accepted, self.dspark_stat_emitted);
        }
        // A6: the round's ring rows are valid through the lane's whole committed sequence —
        // record it so a follow-up request reusing this slot's prefix can carry the ring.
        // The identity claim is bound HERE (after the round ran for `phys`), which is what makes
        // it true; `slot_dspark_len` alone is only the length half of the guard
        // (PLAN/DSPARK_RING_IDENTITY_SPEC.md §3.2).
        self.slot_dspark_len[phys] = self.slot_cache[phys].len();
        if let Some(d) = self.dspark.as_mut() { d.note_ring(phys); }
        if self.dspark_stat_steps % 50 == 0 {
            let acc = if self.dspark_stat_drafts > 0 {
                self.dspark_stat_accepted as f64 / self.dspark_stat_drafts as f64 * 100.0
            } else { 0.0 };
            eprintln!("[dspark] steps={} drafts={} accepted={:.1}% emitted={} tok/step={:.3}",
                      self.dspark_stat_steps, self.dspark_stat_drafts, acc,
                      self.dspark_stat_emitted,
                      self.dspark_stat_emitted as f64 / self.dspark_stat_steps as f64);
        }
        finished
    }

    /// WI1: one DSpark speculative step for a SAMPLING lane — the same 7 greedy drafts,
    /// verified with `verify_forward_sample` under qprobs = 1.0 (the df2_lane_step_sample
    /// pattern; greedy drafts = a point-mass proposal, so rejection sampling reduces to the
    /// greedy accept + the residual replacement, distribution-exact by construction).
    pub(super) fn dspark_lane_step_sample(&mut self, i: usize) -> bool {
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

        // ---- Draft (same as the greedy lane). ----
        let step_t0 = std::time::Instant::now();
        let drafts: Vec<u32> = {
            let ds = self.dspark.as_mut().unwrap();
            assert_eq!(ds.nprev(), main_pos,
                "dspark ring nprev {} != lane pos {} (ring stale or unprimed)", ds.nprev(), main_pos);
            ds.draft_round_dev(committed_tok).expect("dspark draft_round")
        };
        let round_ms = step_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Verify with stochastic output. ----
        let mut verify_input = vec![committed_tok];
        verify_input.extend(drafts.iter().copied());
        let verify_seeds: Vec<u32> =
            (0..crate::dflash2::block()).map(|j| rng_u32(step_key, RNG_DOM_VERIFY, j)).collect();
        let qprobs: Vec<f32> = vec![1.0; crate::dflash2::levels()];
        let penalty = self.make_penalty(i, rep_pen, presence_pen, freq_pen, has_penalty);
        let verify_t0 = std::time::Instant::now();
        let (vsample, vout) = self.gpu.verify_forward_sample(
            &mut self.pool, &verify_input, &mut self.state, phys, kv_stride, main_pos,
            Some(snapshot), penalty,
            &drafts, &qprobs, temperature, top_k, top_p, &verify_seeds, None);
        let verify_ms = verify_t0.elapsed().as_secs_f32() * 1e3;

        // ---- Rejection-sampling accept loop (q = 1). ----
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

        // ---- GDN rollback + tap inject. ----
        if nacc + 1 != 8 {
            self.gpu.copy_gdn_slot(&self.state, snapshot + nacc, phys);
        }
        {
            let ds = self.dspark.as_mut().unwrap();
            ds.sync_staging_from_sink().expect("dspark sync staging");
            ds.inject_dev(nacc + 1).expect("dspark inject_dev");
        }
        self.pool.release_bf16(vout, h * 8);

        // ---- Emit. ----
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
            if !finished {
                let last = *to_emit.last().unwrap_or(&committed_tok);
                lane.last_tok = last;
                lane.pos = main_pos + nacc + 1;
            } else {
                lane.last_tok = *to_emit.last().unwrap_or(&committed_tok);
                lane.pos = main_pos + emit_count;
            }
        }

        self.rec_step(SpecStepRec {
            greedy: false, pos: main_pos as u32, drafts: drafts.len() as u32, nacc: nacc as u32,
            emitted: emit_count as u32, round_ms, verify_ms,
            step_ms: step_t0.elapsed().as_secs_f32() * 1e3,
        });
        self.dspark_stat_steps += 1;
        self.dspark_stat_drafts += drafts.len() as u64;
        self.dspark_stat_accepted += nacc as u64;
        self.dspark_stat_emitted += emit_count as u64;
        if self.dspark_stat_steps % 50 == 0 {
            let acc = if self.dspark_stat_drafts > 0 {
                self.dspark_stat_accepted as f64 / self.dspark_stat_drafts as f64 * 100.0
            } else { 0.0 };
            eprintln!("[dspark] steps={} drafts={} accepted={:.1}% emitted={} tok/step={:.3}",
                      self.dspark_stat_steps, self.dspark_stat_drafts, acc,
                      self.dspark_stat_emitted,
                      self.dspark_stat_emitted as f64 / self.dspark_stat_steps.max(1) as f64);
        }
        // FIX1 #76: same watermark contract as the DF2 chain — publish every step, log every 50.
        // Before this the dspark lane accumulated stats that fed only its dump dirs: /metrics and
        // /health never saw a dspark round.
        if self.dspark_stat_steps != self.dspark_stat_published {
            self.dspark_stat_published = self.dspark_stat_steps;
            crate::tel::publish_dspark(self.dspark_stat_steps, self.dspark_stat_drafts,
                                       self.dspark_stat_accepted, self.dspark_stat_emitted);
        }
        finished
    }
}
