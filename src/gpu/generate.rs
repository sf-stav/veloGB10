//! `impl GpuModel`: `tp_generate`, `tp_generate_dflash`, the local DFlash generation loop and the DFlash dump / feature helpers.
//! The TP agree() guard narrows through `agree_width` (checked by a source-text test in `net.rs`). Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// TP=2 masked-replicated SPMD greedy decode — Stage 3 Proof v0. Both boxes run this IDENTICALLY
    /// over the same prompt; the per-layer FFN all-reduce (`mlp_batch`, world==2) keeps their residual
    /// streams bit-identical, so both independently greedy-argmax the SAME token every step with NO
    /// token broadcast. Returns the generated ids (both ranks compute them; the head decodes+prints).
    /// Prefill is token-by-token (batch=1) so every all-reduce is a single ~10 KB exchange. Stops on
    /// EOS or after `max_new` — deterministic on both ranks, so the exchange rendezvous stays in lockstep.
    pub fn tp_generate(&self, prompt: &[u32], max_new: usize, max_seq_len: usize) -> Vec<u32> {
        // E29-B3: --tp-dflash=1 (head env, shipped to the node via TpConfig — resolved the SAME
        // way on both ranks) routes the one-shot through the draft-8-verify-accept loop. The node
        // runs the identical loop; only the drafter itself is rank-0-only (see tp_generate_dflash).
        if crate::opts::var(crate::opt!("tp-dflash")).is_ok()
            || crate::tp::tp_config().map(|c| c.dflash).unwrap_or(false) {
            return self.tp_generate_dflash(prompt, max_new, max_seq_len);
        }
        let v = self.cfg.vocab_size;
        let h = self.cfg.hidden_size;
        let eos = self.cfg.eos_token_id;
        let plen = prompt.len();
        let mut pool = Pool::new(self.dev.clone());
        // Two slots are used: 0 = the MTP chain, 1 = the sequential ground-truth lane (prefilled
        // and decoded in lockstep). A single slot made every slot-1 KV/state access out of bounds —
        // the landing's changed allocation pattern turned the silent-zero OOB reads into garbage.
        let mut state = self.new_batch_state(2, 2, max_seq_len);
        let mut bufs = self.new_decode_buffers(1);

        // ---- prefill, token by token (batch=1) ----
        let mut last = 0u32;
        for (t, &tok) in prompt.iter().enumerate() {
            let hidden = self.embed_batch(&[tok]);
            let out = self.forward_batch(&mut pool, hidden, &[t], &mut state, max_seq_len, 1);
            if t == plen - 1 {
                if self.cfg.lm_head_fp32 {
                    // hy_v3: fp32 logits end-to-end — bf16 rounding flips near-tie top-1s.
                    let logits = self.logits_batch_f32(&mut pool, &out, 1);
                    let block = 1024u32;
                    blaunch!(self, "argmax_f32_b", (1u32,1,1), (block,1,1), (block*8),
                        (*bufs.token_ids_dev.device_ptr() as u64, d(&logits), v as i32, 1i32));
                    self.sync_stream();
                    last = self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap()[0] as u32;
                    pool.release(logits, v);
                } else {
                let logits = self.logits_batch(&mut pool, &out, 1);
                let block = 1024u32;
                blaunch!(self, "argmax_b", (1u32,1,1), (block,1,1), (block*8),
                    (*bufs.token_ids_dev.device_ptr() as u64, d(&logits), v as i32, 1i32));
                self.sync_stream();
                last = self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap()[0] as u32;
                pool.release_bf16(logits, v);
                }
            }
            pool.release_bf16(out, h);
        }

        // ---- decode ----
        // E29-B2 (DFCTX recorder): the token-by-token prefill above pushed one capture per prompt
        // token — keep only the LAST one (position plen-1's hiddens = the first draft's context
        // feature) so the sink holds exactly [final prefill step, decode steps...].
        if crate::opts::var(crate::opt!("xchain-ctx-dump")).is_ok() || crate::opts::var(crate::opt!("fold-xchain-dump")).is_ok() {
            let mut caps = xchain_capture_take();
            eprintln!("[xchain] prefill captures before decode: {} (batch of last: {})", caps.len(),
                      caps.last().map(|c| c.batch).unwrap_or(0));
            if caps.len() > 1 { caps.drain(..caps.len() - 1); }
            for c in caps { XCHAIN.get_or_init(Default::default).lock().push(c); }
        }
        let mut out_tokens = Vec::with_capacity(max_new);
        if last == eos { return out_tokens; }
        out_tokens.push(last);
        // Per-token latency, not aggregate tok/s: an aggregate number folds any startup transient into
        // the steady state and reports the average as if it were the rate. Percentiles over the tokens
        // AFTER a warmup discard are what the comparison actually needs.
        let mut tok_ns: Vec<u64> = Vec::with_capacity(max_new);
        // CUDA-graph capture (--tp-graph=1). Collapses ~320 eager launches/token into one
        // cuGraphLaunch, which is where the measured 9.76 ms/token of non-GEMV time lives.
        //
        // `max_pc` is PINNED to the run maximum for every step, eager and replayed alike. It only feeds
        // `ns_grid`, which is a launch BOUND rather than a length — `gqa_attn_reduce` recomputes each
        // column's true split count from device-side `pos` and uses `ns_grid` purely as an indexing
        // stride — so over-estimating it is always correct and the surplus split blocks exit immediately.
        // Pinning it is what makes one captured graph valid at every position, and it keeps the eager
        // warm-up allocating exactly the pool buffers the capture will close over.
        let want_graph = self.tp_world == 2 && (crate::opts::var(crate::opt!("tp-graph")).is_ok()
            || crate::tp::tp_config().map(|c| c.graph).unwrap_or(false));
        let max_pc_pinned = max_seq_len;
        let mut graph: Option<CudaGraph> = None;
        let (mut host_pre_ns, mut host_post_ns) = (Vec::new(), Vec::new());
        const GRAPH_CAPTURE_AT: usize = 8;   // let the pool and cuBLAS settle first
        for step in 0..max_new.saturating_sub(1) {
            let t_tok = std::time::Instant::now();
            let mut t_gpu_end = 0u64;
            let pos_val = plen + step;
            let max_pc = if want_graph { max_pc_pinned } else { pos_val + 1 };
            let toks_i32 = vec![last as i32];
            let pos_i32 = vec![pos_val as i32];
            self.dev.htod_sync_copy_into(&toks_i32, &mut bufs.tokens_dev).unwrap();
            self.dev.htod_sync_copy_into(&pos_i32, &mut bufs.pos_dev).unwrap();
            self.dev.synchronize().unwrap();
            // Size the prize before building a device-resident loop: this is the per-token HOST round
            // trip (2 htod + a full-device sync) that a captured graph cannot absorb, plus the dtoh
            // below. It matters more under TP than single-node because both ranks pay it and the skew
            // between them lands on the rendezvous critical path.
            let t_pre = t_tok.elapsed().as_nanos() as u64;
            let next = if want_graph && step == GRAPH_CAPTURE_AT && graph.is_none() {
                // capture_decode_graph runs one REAL decode step as its warm-up, so this iteration
                // produces its token normally; the capture itself executes nothing.
                graph = self.capture_decode_graph(&mut pool, &mut bufs, &mut state, max_seq_len, max_pc, 1);
                eprintln!("[tp{}] decode graph {}", self.tp_rank,
                          if graph.is_some() { "CAPTURED" } else { "capture FAILED — staying eager" });
                self.sync_stream();
                self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap()
                    .into_iter().take(1).map(|x| x as u32).collect::<Vec<u32>>()
            } else if let Some(g) = &graph {
                g.launch();
                self.sync_stream();
                t_gpu_end = t_tok.elapsed().as_nanos() as u64;
                self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap()
                    .into_iter().take(1).map(|x| x as u32).collect::<Vec<u32>>()
            } else {
                self.forward_decode(&mut pool, &mut bufs, &mut state, max_seq_len, max_pc, 1)
            };
            last = next[0];
            let t_all = t_tok.elapsed().as_nanos() as u64;
            host_pre_ns.push(t_pre);
            host_post_ns.push(t_all.saturating_sub(t_gpu_end));
            tok_ns.push(t_all);
            if last == eos { break; }
            out_tokens.push(last);
        }
        Self::report_token_latency(&tok_ns, self.tp_rank);
        if host_pre_ns.len() > 20 {
            let med = |v: &mut Vec<u64>| { v.sort_unstable(); v[v.len()/2] as f64 / 1e6 };
            let (mut a, mut b) = (host_pre_ns[16..].to_vec(), host_post_ns[16..].to_vec());
            let (pre, post) = (med(&mut a), med(&mut b));
            eprintln!("[tp{}] host round-trip per token: pre(2×htod+sync) {pre:.3} ms + post(dtoh) \
                       {post:.3} ms = {:.3} ms — the residue a device-resident loop would remove",
                      self.tp_rank, pre + post);
        }
        if self.tp_head_proof() {
            let (checked, bad) = self.verify_head_redzones(&state);
            if bad == 0 {
                eprintln!("[head-proof] rank {} — PASS: {checked} GDN state red zones pristine after {} tokens \
                           (no kernel wrote outside this rank's heads)", self.tp_rank, out_tokens.len());
            } else if crate::opts::var(crate::opt!("tp-head-proof-fault")).is_ok() {
                eprintln!("[head-proof] rank {} — DETECTED (injected fault): {bad} of {checked} red zones \
                           written. Run continues so the token output can be compared: if it is unchanged, \
                           that is the point — this bug is invisible to the correctness gate.",
                          self.tp_rank);
            } else {
                panic!("[head-proof] rank {} — FAIL: {bad} of {checked} red zones were written", self.tp_rank);
            }
        }
        out_tokens
    }

    /// E29-B3: DFlash one-shot decode — draft-8-verify-accept per step, the z-lab `dflash_generate`
    /// semantics ported exactly. Both ranks run this SPMD loop; rank 0 owns the drafter (unsharded,
    /// with full-vocab bf16 copies of the TARGET's embed + lm_head) and ships the 7 drafts to the
    /// node over the link before the verify. The verify is the existing TP chain verify (invariant
    /// 4: the same reduction order as plain decode), so every emitted token is a target argmax —
    /// the chain equals plain greedy BY CONSTRUCTION; the run still asserts it (GATE_TOKENS vs a
    /// plain run).
    ///
    /// Per step at position `start` (ctx feature covers [0, start)):
    ///   1. block = [t_start, mask x7]; drafter forward (ctx k/v over the whole feature) →
    ///      logits rows 1..7 = drafts for positions start+1..start+7 (row j predicts position
    ///      start+j, non-causal over the block — reference `[:, 1-block_size:, :]`).
    ///   2. verify [t_start, d1..d7] at positions [start, start+8) (causal) → target argmax preds;
    ///      acceptance nacc = longest prefix draft == pred; emit drafts[..nacc] + preds[nacc].
    ///   3. accepted span [start, start+nacc] taps (from the verify's staging scratch) append to
    ///      the ctx feature; advance start += nacc + 1.
    pub fn tp_generate_dflash(&self, prompt: &[u32], max_new: usize, max_seq_len: usize) -> Vec<u32> {
        use crate::dflash::{DflashDrafter, DflashKv};
        let v = self.cfg.vocab_size;
        let h = self.cfg.hidden_size;
        let eos = self.cfg.eos_token_id;
        let plen = prompt.len();
        let is_head = self.tp_rank == 0;
        let mut pool = Pool::new(self.dev.clone());
        // 2 + BLOCK state slots: 0 = the lane; 2..=2+BLOCK-2 = the 8-column verify's per-column
        // GDN checkpoints (an N-column verify with a checkpoint at slot 2 touches 2..=2+N-2).
        let mut state = self.new_batch_state(1, 2 + crate::dflash::MAX_BLOCK, max_seq_len);
        let mut bufs = self.new_decode_buffers(1);

        // ---- rank 0: drafter + persistent KV + its pool + the ctx feature buffer ----
        let dflash_dir = crate::draft_dir_env().unwrap_or_default();
        let mut drafter: Option<DflashDrafter> = None;
        let mut dkv: Option<DflashKv> = None;
        let mut dpool: Option<Pool> = None;
        let mut feature: Option<cudarc::driver::CudaSlice<half::bf16>> = None; // [nctx*h, max_seq_len]
        let mut mask = 0u32;   // from the artifact's dflash_config.mask_token_id (never guessed)
        let mut dblock = 0usize; // the artifact's block_size (draft width AND verify width)
        if is_head {
            // No default path (owner rule 2026-08-23): the DFlash (Hy3) drafter dir MUST be
            // supplied via the drafter-dir env knob — a missing var stops the load loudly.
            if dflash_dir.is_empty() {
                panic!("FATAL: no drafter dir — set --draft-dir / [draft-dir] for the DFlash (Hy3) drafter; there is no default path");
            }
            let dir = &dflash_dir;
            let draf = DflashDrafter::load_from_dir(std::path::Path::new(dir), max_seq_len + crate::dflash::MAX_BLOCK)
                .expect("dflash drafter load");
            mask = draf.mask_token_id;
            dblock = draf.block;
            let dk = DflashKv::new(&draf, max_seq_len + dblock);
            let dp = Pool::new(draf.dev.clone());
            let feat = self.dev.alloc_zeros::<half::bf16>(draf.nctx * h * max_seq_len).unwrap();
            eprintln!("[dflash] rank 0: drafter {} layers h={} mask={} rope_theta={} block={} | feature \
                       [{} x {}, {}] | kv stride {}", draf.layers.len(), draf.h, mask, draf.rope_theta,
                      dblock, draf.nctx, h, max_seq_len, max_seq_len + dblock);
            drafter = Some(draf); dkv = Some(dk); dpool = Some(dp); feature = Some(feat);
        }

        // ---- prefill, token by token (batch=1) — the taps feed the ctx feature (rank 0) ----
        let mut last = 0u32;
        for (t, &tok) in prompt.iter().enumerate() {
            let hidden = self.embed_batch(&[tok]);
            let out = self.forward_batch(&mut pool, hidden, &[t], &mut state, max_seq_len, 1);
            if t == plen - 1 {
                if self.cfg.lm_head_fp32 {
                    let logits = self.logits_batch_f32(&mut pool, &out, 1);
                    let block = 1024u32;
                    blaunch!(self, "argmax_f32_b", (1u32,1,1), (block,1,1), (block*8),
                        (*bufs.token_ids_dev.device_ptr() as u64, d(&logits), v as i32, 1i32));
                    self.sync_stream();
                    last = self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap()[0] as u32;
                    pool.release(logits, v);
                } else {
                    let logits = self.logits_batch(&mut pool, &out, 1);
                    let block = 1024u32;
                    blaunch!(self, "argmax_b", (1u32,1,1), (block,1,1), (block*8),
                        (*bufs.token_ids_dev.device_ptr() as u64, d(&logits), v as i32, 1i32));
                    self.sync_stream();
                    last = self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap()[0] as u32;
                    pool.release_bf16(logits, v);
                }
            }
            // rank 0: this forward's tap (staging col 0) -> feature col t.
            if feature.is_some() { self.dflash_tap_to_feature(feature.as_ref().unwrap(), t, 1); }
            pool.release_bf16(out, h);
        }
        // E29-B3 faithfulness check: --dflash-tapdump=<path> writes the raw ctx feature
        // [5h, plen] as the golden DFCT format (position-major, layer-major) so the torch
        // reference can be run on the REAL taps and its drafts compared with the engine's.
        if let Some(feat) = &feature {
            if let Ok(p) = crate::opts::var(crate::opt!("dflash-tapdump")) {
                let fh: Vec<half::bf16> = self.dev.dtoh_sync_copy(feat).unwrap();
                let nctx = drafter.as_ref().map(|d| d.nctx).unwrap_or(0);
                let mut out = Vec::with_capacity(12 + 4 * plen * nctx * h);
                out.extend_from_slice(b"DFCT");
                out.extend_from_slice(&1u32.to_le_bytes());
                out.extend_from_slice(&(plen as u32).to_le_bytes());
                out.extend_from_slice(&1u32.to_le_bytes());
                for pos in 0..plen {
                    for li in 0..nctx {
                        for i in 0..h {
                            let x = fh[(li * h + i) * max_seq_len + pos];
                            out.extend_from_slice(&x.to_f32().to_le_bytes());
                        }
                    }
                }
                std::fs::write(&p, &out).unwrap();
                eprintln!("[dflash] tapdump {} (plen {} x {} x {})", p, plen, nctx, h);
                // ALSO dump the staging scratch (last prefill forward's taps, [BLOCK, 5h] rows):
                // the feature only shows what survived the copy; the scratch shows what the taps wrote.
                let sp = format!("{}.scratch", p);
                let tap = self.dflash_tap.as_ref().unwrap();
                let sh: Vec<half::bf16> = self.dev.dtoh_sync_copy(&tap.scratch).unwrap();
                let nctx = drafter.as_ref().map(|d| d.nctx).unwrap_or(0);
                let mut sb = Vec::with_capacity(4 + 4 * nctx * h * crate::dflash::MAX_BLOCK);
                for x in &sh { sb.extend_from_slice(&x.to_f32().to_le_bytes()); }
                std::fs::write(&sp, &sb).unwrap();
                eprintln!("[dflash] scratch dump {} ({} rows x {} cols)", sp, crate::dflash::MAX_BLOCK, nctx * h);
            }
        }

        let mut fpos = plen;                         // next free ctx feature column
        let mut out_tokens = Vec::with_capacity(max_new);
        if last == eos { return out_tokens; }
        out_tokens.push(last);
        let mut committed = last;                    // t_start: the block's first token
        let mut start = plen;                        // block's first position == ctx length
        let mut block = vec![committed];
        block.extend(std::iter::repeat(mask).take(dblock - 1));
        let (mut drafts_total, mut accepted_total, mut n_steps) = (0usize, 0usize, 0usize);
        let t0 = std::time::Instant::now();
        let mut step_no = 0u64;
        while out_tokens.len() < max_new {
            step_no += 1;
            n_steps += 1;
            // ---- 1. draft (rank 0 only): block forward over the [5h, start] ctx feature ----
            let mut drafts = vec![0u32; dblock - 1];
            if is_head {
                // The tap D2D copies are queued on the ENGINE stream; the drafter runs on its own
                // stream — drain the engine stream so the feature is complete before the drafter
                // reads it (two blocking streams have no implicit ordering).
                self.sync_stream();
                let draf = drafter.as_mut().unwrap();
                let dk = dkv.as_mut().unwrap();
                let dp = dpool.as_mut().unwrap();
                let feat = feature.as_ref().unwrap();
                let nctx = draf.nctx;
                let ctx_view = feat.slice(..nctx * h * start);
                let logits = draf.forward(dp, dk, &ctx_view, start, &block, start,
                                          Some(self.embed.bf16()), self.dflash_lm_head.as_ref())
                    .expect("dflash block forward");
                let top1 = draf.top1(&logits);
                drafts.copy_from_slice(&top1[1..]);
                if step_no == 1 && crate::opts::var(crate::opt!("dflash-debug")).is_ok() {
                    // one-shot diagnosis: the drafter's logits pattern at step 1 (rank 0).
                    let mut f = std::io::BufWriter::new(std::fs::File::create("/tmp/b3_draft_dbg.txt").unwrap());
                    use std::io::Write;
                    writeln!(f, "top1 all 8: {}", top1.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",")).unwrap();
                    for b in 0..dblock {
                        let col = &logits[b * draf.vocab..(b + 1) * draf.vocab];
                        let mut best = 0usize;
                        for (t, &x) in col.iter().enumerate() { if x > col[best] { best = t; } }
                        let mut sorted: Vec<f32> = col.to_vec();
                        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
                        writeln!(f, "row {b}: top1 {best} logit {:.3} | top3 {:.3} {:.3} {:.3} | absmax {:.3}",
                                 col[best], sorted[0], sorted[1], sorted[2], col.iter().fold(0.0f32, |m, &x| m.max(x.abs()))).unwrap();
                    }
                    f.flush().unwrap();
                }
            }
            // ---- 2. ship the drafts: both ranks exchange (BLOCK+1 words; the last 8 bytes of the
            // wire frame are the gen tag, leaving exactly BLOCK-1 usable u32s). The exchange is a
            // SWAP: each side receives the PEER's payload — the head sent its real drafts and the
            // node sent zeros, so the node adopts the received drafts and the head keeps its own.
            let recv = crate::net::exchange_u32s(&drafts, dblock + 1).expect("dflash draft exchange");
            if !is_head { drafts = recv; }
            // ---- 3. verify [committed, drafts...] at positions [start, start+8) ----
            let mut vinput = Vec::with_capacity(dblock);
            vinput.push(committed);
            vinput.extend_from_slice(&drafts);
            let (preds, vout) = self.verify_forward(&mut pool, &vinput, &mut state, 0, max_seq_len,
                                                    start, Some(2), None);
            // ---- 4. accept the longest prefix; emit accepted drafts + the target bonus ----
            // preds[i] = the target's argmax for position start+i+1 (col-0 bit-identical to a
            // decode — invariant 4), so an accepted draft IS the plain-greedy token.
            let mut nacc = 0usize;
            while nacc < drafts.len() && preds[nacc] == drafts[nacc] { nacc += 1; }
            let bonus = preds[nacc];
            // GDN rollback to the last accepted column (full accept needs none).
            if nacc + 1 != dblock { self.copy_gdn_slot(&state, 2 + nacc, 0); }
            let emitted_before = out_tokens.len();
            for &d in drafts.iter().take(nacc) { out_tokens.push(d); }
            if out_tokens.len() < max_new { out_tokens.push(bonus); }
            accepted_total += nacc;
            drafts_total += dblock - 1;
            eprintln!("[dflash] step {step_no}: start={start} block0={committed} drafts={} \
                       nacc={nacc} bonus={bonus}", drafts.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
            // ---- 5. rank 0: accepted span taps (staging cols 0..=nacc) -> feature ----
            if feature.is_some() { self.dflash_tap_to_feature(feature.as_ref().unwrap(), fpos, nacc + 1); }
            fpos += nacc + 1;
            // ---- 6. advance ----
            committed = bonus;
            start += nacc + 1;
            block[0] = committed;
            // eos: stop like plain decode (which breaks right after emitting eos). Any emitted
            // token this step may be eos — truncate the chain at the first one.
            let mut hit_eos = false;
            for &t in &out_tokens[emitted_before..] { if t == eos { hit_eos = true; break; } }
            if hit_eos {
                if let Some(pos) = out_tokens.iter().position(|&t| t == eos) { out_tokens.truncate(pos + 1); }
                pool.release_bf16(vout, h * dblock);
                break;
            }
            // ---- 7. lockstep agreement guard (TP invariant 10): same (step, count, hash) on both
            // ranks, or the link aborts both instead of desyncing the verify all-reduces. ----
            let mut hsh = 0x811c9dc5u32;
            for &t in &out_tokens[emitted_before..] { hsh = hsh.rotate_left(5) ^ t; }
            let cnt = (out_tokens.len() - emitted_before) as u8;
            // B8/G1: the verify WIDTH joins the extended token (k_verify = BLOCK here). agree_ext
            // sends net::agree_word (width in bits 27..31, digest folded to 27 bits), so the local
            // word comes from the same function before comparing.
            // S-B9-CF-AGREE16B (W5): the range check sees the whole usize, before the narrowing
            // (`dblock as u8` first would turn 256 + k into k and slip past agree_ext's own check).
            let Some(kv) = crate::net::agree_width(dblock) else {
                crate::net::abort_link();
                panic!("[dflash] TP agree REFUSED at step {step_no}: block {dblock} does not fit the lockstep token's \
                       verify-width field (max {})", crate::net::AGREE_WIDTH_MAX);
            };
            match crate::net::agree_ext(step_no, cnt, kv, hsh) {
                Some((pc, ph)) if pc == cnt && ph == crate::net::agree_word(kv, hsh) => {}
                Some((pc, ph)) => {
                    let hsh_ext = crate::net::agree_word(kv, hsh);
                    crate::net::abort_link();
                    panic!("[dflash] TP AGREE MISMATCH at step {step_no}: local (count {cnt}, hash {hsh_ext:#010x}) \
                           vs peer (count {pc}, hash {ph:#010x}) — the ranks verified different chains");
                }
                None => {
                    crate::net::abort_link();
                    panic!("[dflash] TP agree: link aborted or peer timeout at step {step_no} — divergence would be silent");
                }
            }
            pool.release_bf16(vout, h * dblock);
        }
        if is_head {
            let dt = t0.elapsed().as_secs_f64().max(1e-6);
            let acc = if drafts_total > 0 { accepted_total as f64 / drafts_total as f64 } else { 0.0 };
            eprintln!("[dflash] {n_steps} draft steps, {} tokens ({:.2}/step), {:.2} ms/step, \
                       {:.1} tok/s wall, acceptance {:.1}% ({accepted_total}/{drafts_total})",
                      out_tokens.len(), out_tokens.len() as f64 / n_steps.max(1) as f64,
                      dt * 1000.0 / n_steps.max(1) as f64, out_tokens.len() as f64 / dt, acc * 100.0);
        }
        out_tokens
    }

    /// P14 — DFlash v1 greedy generation, single process (the lane's reference loop; the batch
    /// scheduler runs the same primitives on its own state/pool). Semantics are the z-lab
    /// `dflash_generate` greedy path: a block `[committed, mask x (block-1)]` is drafted from the
    /// target's hidden states, the trunk verifies `[committed, drafts...]`, the longest agreeing
    /// prefix is accepted and the target's own argmax is the bonus — so the emitted chain IS plain
    /// greedy by construction (every emitted token is a trunk argmax).
    ///
    /// `dump_prefix` (optional) writes `<prefix>.dfct` + `<prefix>.tokens.json` — the real-tap
    /// oracle input for ~/recipe_launch/p14/ref/run_ref.py (AGENTS §3: a wrong-but-self-consistent
    /// drafter passes every in-mode gate, so the drafts must be diffed against the reference on
    /// REAL activations).
    pub fn dflash_generate_local(&self, dir: &str, prompt: &[u32], max_new: usize,
                                 max_seq_len: usize, dump_prefix: Option<&str>) -> Vec<u32> {
        use crate::dflash::{DflashDrafter, DflashKv};
        let h = self.cfg.hidden_size;
        let v = self.cfg.vocab_size;
        let eos = self.cfg.eos_token_id;
        let plen = prompt.len();
        assert!(plen + max_new + crate::dflash::MAX_BLOCK <= max_seq_len, "prompt+gen > max_seq_len");
        let mut pool = Pool::new(self.dev.clone());
        let mut state = self.new_batch_state(1, 2 + crate::dflash::MAX_BLOCK, max_seq_len);
        let mut bufs = self.new_decode_buffers(1);

        let mut draf = DflashDrafter::load_from_dir(std::path::Path::new(dir), max_seq_len + 16)
            .expect("dflash drafter load (drafter dir)");
        let nctx = draf.nctx;
        let block = draf.block;
        let mask = draf.mask_token_id;
        let mut dkv = DflashKv::new(&draf, max_seq_len + crate::dflash::MAX_BLOCK);
        let mut dpool = Pool::new(draf.dev.clone());
        let feat = self.dev.alloc_zeros::<half::bf16>(nctx * h * max_seq_len).unwrap();
        eprintln!("[dflash] local lane: {} layers h={} block={} mask={} nctx={} rope_theta={}",
                  draf.layers.len(), draf.h, block, mask, nctx, draf.rope_theta);

        // ---- prefill, token by token: the taps of every prompt position land in the ctx feature
        // (a token-by-token prefill is the price of a per-position feature; the serving lane arms
        // the WIDE sink and fills the feature in one chunked prefill instead). ----
        let mut last = 0u32;
        for (t, &tok) in prompt.iter().enumerate() {
            let hidden = self.embed_batch(&[tok]);
            let out = self.forward_batch(&mut pool, hidden, &[t], &mut state, max_seq_len, 1);
            if t == plen - 1 {
                let logits = self.logits_batch_f32(&mut pool, &out, 1);
                blaunch!(self, "argmax_f32_b", (1u32,1,1), (1024,1,1), (1024*8),
                    (*bufs.token_ids_dev.device_ptr() as u64, d(&logits), v as i32, 1i32));
                self.sync_stream();
                last = self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap()[0] as u32;
                pool.release(logits, v);
            }
            self.dflash_tap_to_feature(&feat, t, 1);
            pool.release_bf16(out, h);
        }
        let mut out_tokens: Vec<u32> = Vec::with_capacity(max_new + 1);
        if last == eos { return out_tokens; }
        out_tokens.push(last);
        let mut start = plen;              // next block's first position == appended ctx columns
        let mut committed = last;
        let (mut pend_row0, mut pend_ncols) = (0usize, plen);   // the ctx span to append next
        let (mut drafts_total, mut accepted_total, mut n_steps) = (0usize, 0usize, 0usize);
        let mut block_toks = vec![committed];
        block_toks.extend(std::iter::repeat(mask).take(block - 1));
        let t0 = std::time::Instant::now();
        while out_tokens.len() < max_new {
            n_steps += 1;
            // ---- 1. draft: append the pending ctx span (engine stream → sync first), block
            // forward, then the TARGET's head over the block rows 1.. = the draft tokens ----
            self.sync_stream();
            let drafts = self.dflash_draft(&mut draf, &mut dkv, &mut dpool, &feat, nctx, h,
                                                 pend_row0, pend_ncols, pend_row0, &block_toks);
            if n_steps == 1 {
                // Oracle: this step's exact drafter input (feature + block ids + the target's
                // embedding rows) and the object the torch reference must reproduce (the final
                // normalized hidden, i.e. everything before the LM head).
                if let Some(pfx) = dump_prefix {
                    eprintln!("[dflash-dbg] dump hidden begin");
                    self.dflash_dump_hidden(&draf, block, pfx);
                    eprintln!("[dflash-dbg] dump hidden done; embed begin");
                    self.dflash_dump_embed(&[block_toks[0], mask], pfx);
                    eprintln!("[dflash-dbg] dump embed done");
                }
            }
            drafts_total += block - 1;
            // ---- 2. verify [committed, drafts...] at positions [start, start+block) ----
            let mut vinput = Vec::with_capacity(block);
            vinput.push(committed);
            vinput.extend_from_slice(&drafts);
            let (preds, vout) = self.verify_forward(&mut pool, &vinput, &mut state, 0, max_seq_len,
                                                     start, Some(2), None);
            // ---- 3. accept the longest agreeing prefix; the bonus is the trunk's own argmax ----
            let mut nacc = 0usize;
            while nacc < drafts.len() && preds[nacc] == drafts[nacc] { nacc += 1; }
            let bonus = preds[nacc];
            if nacc + 1 != block { self.copy_gdn_slot(&state, 2 + nacc, 0); }
            let before = out_tokens.len();
            for &d in drafts.iter().take(nacc) { out_tokens.push(d); }
            if out_tokens.len() < max_new { out_tokens.push(bonus); }
            accepted_total += nacc;
            eprintln!("[dflash] step {n_steps}: start={start} nacc={nacc} bonus={bonus} drafts={} \
                       preds={} commit={committed}",
                      drafts.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","),
                      preds[..drafts.len().min(preds.len())].iter().map(|t| t.to_string())
                          .collect::<Vec<_>>().join(","));
            // ---- 4. the verify's own tap capture wrote the span's features at their absolute
            // positions; the next step appends [start, start+nacc+1) ----
            self.dflash_tap_to_feature(&feat, start, nacc + 1);
            pend_row0 = start;
            pend_ncols = nacc + 1;
            start += nacc + 1;
            committed = bonus;
            block_toks[0] = committed;
            let mut hit_eos = false;
            for &t in &out_tokens[before..] { if t == eos { hit_eos = true; break; } }
            pool.release_bf16(vout, h * block);
            if hit_eos {
                if let Some(pos) = out_tokens.iter().position(|&t| t == eos) { out_tokens.truncate(pos + 1); }
                break;
            }
        }
        let dt = t0.elapsed().as_secs_f64().max(1e-6);
        let acc = if drafts_total > 0 { accepted_total as f64 / drafts_total as f64 } else { 0.0 };
        eprintln!("[dflash] LOCAL: {n_steps} steps, {} tokens ({:.2}/step), {:.2} ms/step, \
                   {:.1} tok/s, acceptance {:.1}% ({accepted_total}/{drafts_total})",
                  out_tokens.len(), out_tokens.len() as f64 / n_steps.max(1) as f64,
                  dt * 1000.0 / n_steps.max(1) as f64, out_tokens.len() as f64 / dt, acc * 100.0);
        if let Some(pfx) = dump_prefix {
            self.dflash_dump_ctx(&feat, nctx, h, plen, &out_tokens, pfx);
        }
        out_tokens
    }

    /// One v1 draft: append ctx columns `[row0, row0+ncols)` (sourced from the feature columns
    /// `[src_col0, src_col0+ncols)`), run the block forward, then run the TARGET's LM head over the
    /// drafter's final hidden and argmax each column. Returns the block's rows 1.. (the drafts).
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_draft(&self, draf: &mut crate::dflash::DflashDrafter, dkv: &mut crate::dflash::DflashKv,
                          dpool: &mut Pool, feat: &cudarc::driver::CudaSlice<half::bf16>,
                          nctx: usize, _h: usize, row0: usize, ncols: usize, src_col0: usize,
                          block_toks: &[u32]) -> Vec<u32> {
        let blk = block_toks.len();
        let v = self.cfg.vocab_size;
        let pos_start = row0 + ncols;
        let src = feat.slice(src_col0 * nctx * _h..(src_col0 + ncols) * nctx * _h);
        // The TARGET's embedding may be NVFP4-packed (the 35B's is), so the noise block must come
        // from the engine's own embedding path — `W::bf16()` on a packed table is a loud panic by
        // design (the legacy single-stream contract), not a path we may take here.
        let noise = self.embed_batch(block_toks);
        draf.forward_step(dpool, dkv, &src, ncols, row0, block_toks, pos_start,
                          None::<&cudarc::driver::CudaSlice<half::bf16>>, Some(&noise))
            .expect("dflash forward_step");
        drop(noise);
        let logits = self.logits_batch(dpool, &draf.last_hidden, blk);   // [v, blk] bf16
        let ids = self.dev.alloc_zeros::<i32>(blk).unwrap();
        // `argmax_b` indexes the BATCH by `blockIdx.x` (kernels/gpu_batch.cu:3261) — one block per
        // column. A grid of (1,1,1) writes only column 0 and leaves the rest at their buffer
        // contents, which presented as "every draft is token 0".
        blaunch!(self, "argmax_b", (blk as u32,1,1), (1024,1,1), (1024*8),
            (*ids.device_ptr() as u64, d(&logits), v as i32, blk as i32));
        self.sync_stream();
        let host = self.dev.dtoh_sync_copy(&ids).unwrap();
        dpool.release_bf16(logits, v * blk);
        host[1..].iter().map(|&t| t as u32).collect()
    }

    /// P14: absolute position of the next wide tap capture (see `DflashTapSink`).
    pub fn set_dflash_wide_pos(&self, p: usize) {
        if let Some(t) = &self.dflash_tap { if t.wide.is_some() { t.set_wide_pos(p); } }
    }

    /// P14: arm the wide ctx capture surface for the v1 lane (geometry from the artifact's config,
    /// never assumed: the sink reads `nctx` from the same file).
    pub fn arm_dflash_wide(&mut self, dir: &str, stride: usize) -> anyhow::Result<()> {
        let cfg = crate::dflash::DflashCfg::load(std::path::Path::new(dir))?;
        let h = self.cfg.hidden_size;
        let buf = self.dev.alloc_zeros::<half::bf16>(cfg.nctx() * h * stride)?;
        if self.dflash_tap.is_none() {
            // The artifact can be installed AFTER the model load (the serving path resolves
            // --spec-source at schedule time), so create the sink here when the load did not.
            self.dflash_tap = Some(crate::dflash::DflashTapSink::new(
                &self.dev, h, cfg.tap_layers.clone()));
        }
        match self.dflash_tap.as_mut() {
            Some(t) => { t.arm_wide(buf, stride); eprintln!(
                "[dflash] wide ctx surface armed: {} taps x h {} x {} positions", t.nctx, h, stride); }
            None => anyhow::bail!("v1 tap sink could not be created"),
        }
        Ok(())
    }

    /// P14: D2D the wide ctx feature columns `[src_col, src_col+ncols)` into the lane's consumed
    /// feature at `[dst_col, dst_col+ncols)` (both `[nctx*h, stride]` bf16).
    pub fn dflash_feat_copy_at(&self, dst: &cudarc::driver::CudaSlice<half::bf16>,
                               dst_col: usize, src_col: usize, ncols: usize) {
        let t = self.dflash_tap.as_ref().expect("dflash wide sink");
        let h = self.cfg.hidden_size;
        let bytes = ncols * t.nctx * h * 2;
        unsafe {
            let r = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                *dst.device_ptr() as u64 + (dst_col * t.nctx * h * 2) as u64,
                t.wide_ptr() + (src_col * t.nctx * h * 2) as u64,
                bytes, self.stream.stream);
            assert!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "dflash feat copy: {r:?}");
        }
    }

    /// P14 oracle support: gather `ids`' embedding rows from the TARGET's embedding table (bf16)
    /// and write them as JSON `{"ids": [...], "rows": [[f32 x h] ...]}`. The 35B drafter artifact
    /// carries NO embedding of its own — the reference conditions on the target's — so the torch
    /// oracle needs exactly the rows the engine's block gathered (≤ block unique ids, ~64 KB).
    pub fn dflash_dump_embed(&self, ids: &[u32], prefix: &str) {
        let h = self.cfg.hidden_size;
        let n = ids.len();
        // The TARGET's embedding may be NVFP4-packed (`W::bf16()` panics there by design): gather
        // through the ENGINE's own embedding path, which is exactly the bytes the drafter conditions
        // on. `rows` is [h, n] col-major.
        let rows = self.embed_batch(ids);
        let host: Vec<half::bf16> = self.dev.dtoh_sync_copy(&rows).unwrap();
        let mut j = String::from("{\"ids\": [");
        j.push_str(&ids.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
        j.push_str("], \"rows\": [");
        for i in 0..n {
            if i > 0 { j.push(','); }
            j.push('[');
            j.push_str(&host[i * h..(i + 1) * h].iter()
                .map(|x| format!("{:.9e}", x.to_f32())).collect::<Vec<_>>().join(","));
            j.push(']');
        }
        j.push_str("]}");
        std::fs::write(format!("{prefix}.embed.json"), j).unwrap();
        eprintln!("[dflash] embed rows for {:?} -> {prefix}.embed.json", ids);
    }

    /// P14 oracle support: write the drafter's final normalized hidden ([h, block] f32, the exact
    /// input to the LM head) — the object the torch reference must reproduce bit-for-bit-ish.
    pub fn dflash_dump_hidden(&self, draf: &crate::dflash::DflashDrafter, block: usize, prefix: &str) {
        let h = draf.h;
        let v: Vec<half::bf16> = self.dev.dtoh_sync_copy(&draf.last_hidden).unwrap();
        let mut out = Vec::with_capacity(16 + 4 * h * block);
        out.extend_from_slice(b"DFHL");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(block as u32).to_le_bytes());
        out.extend_from_slice(&(h as u32).to_le_bytes());
        for x in &v[..h * block] { out.extend_from_slice(&x.to_f32().to_le_bytes()); }
        std::fs::write(format!("{prefix}.hidden.bin"), &out).unwrap();
        eprintln!("[dflash] drafter hidden [{} x {}] -> {prefix}.hidden.bin", h, block);
    }

    /// Write the ctx feature as a DFCT file (position-major, layer-major) — the oracle input for
    /// ~/recipe_launch/p14/ref/run_ref.py — plus the emitted chain as tokens.json.
    fn dflash_dump_ctx(&self, feat: &cudarc::driver::CudaSlice<half::bf16>, nctx: usize, h: usize,
                       plen: usize, chain: &[u32], prefix: &str) {
        let fh: Vec<half::bf16> = self.dev.dtoh_sync_copy(feat).unwrap();
        // The oracle compares ONE step (the reference's `dflash_generate` first block): ctx = the
        // whole prompt, block = [first emitted token, mask x (block-1)]. Longer chains would feed
        // REAL tokens into the block, which the lane never does (its block is always mask-filled),
        // so the dump keeps exactly the one generated token the block needs.
        let cols = plen;
        let chain = &chain[..chain.len().min(1)];
        let mut out = Vec::with_capacity(16 + 4 * cols * nctx * h);
        out.extend_from_slice(b"DFCT");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(cols as u32).to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        for pos in 0..cols {
            for li in 0..nctx {
                for i in 0..h {
                    let x = fh[(li * h + i) + pos * nctx * h];
                    out.extend_from_slice(&x.to_f32().to_le_bytes());
                }
            }
        }
        std::fs::write(format!("{prefix}.dfct"), &out).unwrap();
        let mut toks = Vec::with_capacity(plen + chain.len());
        toks.extend_from_slice(&chain.to_vec());
        let j = format!("{{\"plen\": {}, \"tokens\": [{}]}}", plen,
                        toks.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
        std::fs::write(format!("{prefix}.tokens.json"), j).unwrap();
        eprintln!("[dflash] dump: {prefix}.dfct ({cols} cols x {nctx} x {h}) + {prefix}.tokens.json");
    }

    /// E29-B3 / P14: D2D (compute stream) copy of the first `ncols` columns of the DFlash tap
    /// staging scratch — this forward's taps, layer-major [nctx*h, ncols] — into the ctx feature
    /// buffer at column `fpos`. Rank 0 only (the sink exists only there). No dtoh in the loop.
    ///
    /// The v1 SERVING lane calls this once per step with `ncols = nacc+1` to place the span the
    /// verify just confirmed at its absolute ctx column: the verify's own capture writes the
    /// POSITION-INDEPENDENT ring staging (a captured graph cannot bake a host-computed column), so
    /// this copy is the only thing that fills the lane's ctx for positions ≥ plen. The single-process
    /// probe reaches the same buffer through the same call at the same place.
    pub fn dflash_tap_to_feature(&self, feat: &cudarc::driver::CudaSlice<half::bf16>, fpos: usize, ncols: usize) {
        let nctx = self.dflash_tap.as_ref().map(|t| t.nctx).unwrap_or(0);
        let h = self.cfg.hidden_size;
        let tap = self.dflash_tap.as_ref().expect("dflash tap sink");
        let src = *tap.scratch.device_ptr() as u64;
        let dst = *feat.device_ptr() as u64 + (fpos as u64) * (nctx * h * 2) as u64;
        unsafe {
            let r = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(dst, src, ncols * nctx * h * 2, self.stream.stream);
            assert!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "dflash tap->feature D2D: {r:?}");
        }
    }
}
