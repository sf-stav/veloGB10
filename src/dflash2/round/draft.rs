//! `Df2Round` draft rounds (host / device / sampled / graph variants, tree tables) and the 5-layer `layer_forward`.
//! Siblings: `inject.rs` (the ctx it attends), `state.rs`; `mod.rs` has the map.

use super::*;

impl Df2Round {
    /// NIGHT1 follow-up: the DFlash2 twin of `FwdModel::dsync` — wait for this round's stream through `host_wait`
    /// (`--gpu-wait`), THEN the original device synchronize, unchanged. The round's kernels and graph replays run on
    /// `self.stream`, so the driver call that follows only waits for whatever is still queued elsewhere. A stream
    /// error is reported by the first wait instead of by the synchronize. `#[track_caller]`: each call site keeps its
    /// own prediction.
    #[track_caller]
    pub(crate) fn dsync(&self) -> std::result::Result<(), cudarc::driver::DriverError> {
        crate::host_wait::stream_wait(self.stream.stream)?;
        self.dev.synchronize()
    }

    /// One draft round. Requires `nprev ≥ 1`. `window` = the attention band (2048; the
    /// negative control passes a huge value). `flip_unary` negates the unary term inside the
    /// walk (the selector sign-flip control). `ctl_dual_write` additionally writes the block
    /// k/v at LINEAR rows `[nprev, nprev+8)` so the S3F `gqa_attn_band_b` control can read
    /// the same cache. `dump` collects probe surfaces (logits, hp) + stage timings.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_round(&mut self, anchor: u32, window: usize, flip_unary: bool,
                       ctl_dual_write: bool, dump: bool) -> Result<Df2RoundOut> {
        self.draft_round_depth(anchor, window, flip_unary, ctl_dual_write, dump, n_layers())
    }

    /// Probe-only layer bisect: run the round with the first `nlayers` block layers (the
    /// block h after the last run layer is what the head/top16 see — for the determinism
    /// hunt we only compare `h_final`).
    pub fn draft_round_depth(&mut self, anchor: u32, window: usize, flip_unary: bool,
                             ctl_dual_write: bool, dump: bool, nlayers: usize) -> Result<Df2RoundOut> {
        self.draft_round_stages(anchor, window, flip_unary, ctl_dual_write, dump, nlayers, true)
    }

    /// Determinism bisection: `post` = false stops after the final norm (no head/top16/hp/walk).
    pub fn draft_round_stages(&mut self, anchor: u32, window: usize, flip_unary: bool,
                              ctl_dual_write: bool, dump: bool, nlayers: usize, post: bool)
                              -> Result<Df2RoundOut> {
        let ntot = self.nprev + block();
        let mut timer = EvTimer::new();
        timer.mark(self.stream.stream);

        // ---- block input: [anchor, MASK×7] from the TRUNK's real embed table ----
        let toks: Vec<i32> = [anchor as i32].iter().copied()
            .chain(std::iter::repeat(crate::dflash2::mask_token_id() as i32).take(block() - 1)).collect();
        self.dev.htod_sync_copy_into(&toks, &mut self.toks_blk)?;
        self.embed_gather();

        // ---- 5-layer backbone (S3F sequence; ring attention) ----
        for li in 0..nlayers.min(n_layers()) {
            self.layer_forward(li, ntot, window, ctl_dual_write);
        }
        // final norm
        self.rmsnorm(&self.blk.h_final, &self.blk.h, &self.glob.norm, hidden(), block());
        timer.mark(self.stream.stream);
        if !post {
            let mut hf: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.h_final)?;
            hf.truncate(hidden() * block());   // the 9th row is the guard row — not part of the output
            return Ok(Df2RoundOut {
                tokens: vec![0; levels()], candidates: vec![0; levels() * 16], unary: vec![0.0; levels() * 16],
                scores: vec![0.0; levels() * 16], h_final: hf.iter().map(|x| x.to_f32()).collect(),
                layer_hiddens: Vec::new(),
                logits: None, hp: None, stage_ms: None,
            });
        }

        // ---- borrowed head at N=7 on h_final cols 1..7 ([5120,7] col-major at +5120) ----
        // Dtype dispatch: NVFP4 (persistent-grid gemm_mma_fp4_b; N<=16 legal) or BF16
        // (gemm_binv_b, NC=7 — the batch-invariant fixed-order reduction).
        self.head_logits();
        timer.mark(self.stream.stream);
        // ---- top-16 on the 7 MASK rows ----
        klaunch!(self, "top16_b", (levels() as u32, 1, 1), (256, 1, 1), 0,
            (d(&self.out_vals), d(&self.out_ids), d(&self.logits), vocab() as i32, levels() as i32));
        timer.mark(self.stream.stream);

        // ---- hidden_projection [256,5120] × h_sel [5120,7] (rows 1..7) → hp ----
        self.gemm_dsp(&self.hp_bf16, &self.hp_w, d(&self.blk.h_final) + (hidden() * 2) as u64, selector_rank(), hidden());
        klaunch!(self, "bf16tof32", grid(selector_rank() * block()), (256, 1, 1), 0,
            (d(&self.hp_f32), d(&self.hp_bf16), (selector_rank() * block()) as i32));

        // ---- the greedy chain (sign-flip control negates the unary term) ----
        let unary_src: &CudaSlice<f32> = if flip_unary {
            let vals: Vec<f32> = self.dev.dtoh_sync_copy(&self.out_vals)?;
            let neg: Vec<f32> = vals.iter().map(|&v| -v).collect();
            self.dev.htod_sync_copy_into(&neg, &mut self.unary_ctl)?;
            &self.unary_ctl
        } else {
            &self.out_vals
        };
        klaunch!(self, "df2_sel_walk_b", (1u32, 1, 1), (256, 1, 1), 0,
            (d(&self.walk_tokens), d(&self.walk_scores), d(&self.hp_f32), d(&self.out_ids),
             d(unary_src), d(&self.pred_cb), d(&self.succ_cb), anchor, self.anchor_dev,
             selector_rank() as i32, levels() as i32));
        timer.mark(self.stream.stream);

        // ---- read back ----
        let tokens: Vec<u32> = self.dev.dtoh_sync_copy(&self.walk_tokens)?.to_vec();
        let scores: Vec<f32> = self.dev.dtoh_sync_copy(&self.walk_scores)?.to_vec();
        let candidates: Vec<u32> = self.dev.dtoh_sync_copy(&self.out_ids)?.to_vec();
        let unary: Vec<f32> = self.dev.dtoh_sync_copy(&self.out_vals)?.to_vec();
        let mut hf: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.h_final)?;
        hf.truncate(hidden() * block());   // guard row never leaves the device
        let h_final: Vec<f32> = hf.iter().map(|x| x.to_f32()).collect();
        let (logits, hp) = if dump {
            let lg: Vec<bf16> = self.dev.dtoh_sync_copy(&self.logits)?;
            let mut hp16: Vec<bf16> = self.dev.dtoh_sync_copy(&self.hp_bf16)?;
            hp16.truncate(selector_rank() * levels());   // 8th gemm column never leaves the device
            (Some(lg.iter().map(|x| x.to_f32()).collect()),
             Some(hp16.iter().map(|x| x.to_f32()).collect()))
        } else {
            (None, None)
        };
        // marks: 0 start, 1 block-end, 2 head-end, 3 top16-end, 4 walk-end
        let stage_ms = Some([timer.elapsed_ms(0), timer.elapsed_ms(1), timer.elapsed_ms(2), timer.elapsed_ms(3)]);
        Ok(Df2RoundOut { tokens, candidates, unary, scores, h_final, layer_hiddens: Vec::new(),
                         logits, hp, stage_ms })
    }

    /// S5F — the ENGINE draft round (lean): the same kernel sequence as `draft_round` with the
    /// probe-only surfaces removed (no EvTimer, no dumps, no ctl/dual-write, no flip) and ONLY the
    /// 7 walk tokens read back. This is the serving-callable round the decode loop drives.
    /// Requires `nprev >= 1` (the committed prefix incl. the prompt prime). The block pass reads
    /// the ring() (ctx) — the sink staging is NOT read here (its fresh contents are consumed by
    /// [`inject_dev`] after the trunk verify). Stream-ordered on the round's stream.
    pub fn draft_round_dev(&mut self, anchor: u32) -> Result<Vec<u32>> {
        // ---- block input: [anchor, MASK×7] from the TRUNK's real embed table ----
        let toks: Vec<i32> = [anchor as i32].iter().copied()
            .chain(std::iter::repeat(crate::dflash2::mask_token_id() as i32).take(block() - 1)).collect();
        self.dev.htod_sync_copy_into(&toks, &mut self.toks_blk)?;
        // The EAGER path must use the packed-arg kernels (ntot/anchor from the launch args, NOT
        // the graph's device ints): after a capture, self.ntot_dev/anchor_dev point at the graph
        // buffers — leaving them set would make the eager kernels read a STALE ntot (the last
        // replay's) against an eager-sized smem allocation (the graph-capture smem bug class,
        // caught by the probe's varying-nprev replays).
        let (ntot_save, anchor_save) = (self.ntot_dev, self.anchor_dev);
        self.ntot_dev = 0;
        self.anchor_dev = 0;
        let r = self.draft_round_kernels(anchor, false, None);
        self.ntot_dev = ntot_save;
        self.anchor_dev = anchor_save;
        r?;
        // ---- read back ONLY the 7 tokens ----
        self.dsync()?;
        let tokens: Vec<u32> = self.dev.dtoh_sync_copy(&self.walk_tokens)?.to_vec();
        Ok(tokens)
    }

    /// S5F3 — the DUMP draft round: the same eager kernel sequence as `draft_round_dev` (incl.
    /// the packed-arg zeroing so a captured graph's stale device ints can't leak in) with the
    /// FULL probe surfaces read back: the 7 tokens, the top-16 candidates/unary, the chain
    /// scores, the block h_final `[8, 5120]` f32, and (dump=true) the [7, vocab] logits + hp.
    /// The step-dump analysis compares these against the S2F oracle's run_round surfaces on the
    /// same taps (the S1 bisect). Behavior-neutral vs `draft_round_dev`: same kernels, same
    /// inputs — only the readbacks differ. Callers must refresh_block_pos() first (as the
    /// serving eager path does).
    pub fn draft_round_full(&mut self, anchor: u32) -> Result<Df2RoundOut> {
        let (ntot_save, anchor_save) = (self.ntot_dev, self.anchor_dev);
        self.ntot_dev = 0;
        self.anchor_dev = 0;
        let r = self.draft_round_stages(anchor, crate::dflash2::sliding_window(), false, false,
                                        true, crate::dflash2::n_layers(), true);
        self.ntot_dev = ntot_save;
        self.anchor_dev = anchor_save;
        r
    }

    /// The kernel-only draft-round sequence (embed gather -> 5 layers -> final norm -> head ->
    /// top16 -> hp -> walk). All inputs must be device-resident: `toks_blk` (anchor+MASK),
    /// `pos_blk` + `cos8`/`sin8` (refreshed by `refresh_block_pos`), the ring() (ctx), and the
    /// per-replay device ints (`ntot_dev`, `anchor_dev` — 0 on the eager path). Shared by the
    /// eager path and the CUDA-graph capture (the capture records exactly this sequence).
    /// `graph_mode` = true for the capture/replay (the walk reads the anchor from `anchor_dev`);
    /// the eager path passes the real `anchor` as the launch arg with `anchor_dev` = 0.
    /// `sample` = Some((seeds, temperature)) runs the S5F2 SAMPLED selector walk
    /// (`df2_sel_walk_sample_b` — multinomial at temperature, q_rows + candidate table out)
    /// instead of the greedy chain; the sampled path is EAGER-only (per-step seeds are host
    /// inputs, not graph-replay-safe) — callers must pass None when capturing.
    fn draft_round_kernels(&mut self, anchor: u32, graph_mode: bool,
                           sample: Option<(&[u32], f32)>) -> Result<()> {
        let ntot = self.nprev + block();
        self.embed_gather();
        // ---- 5-layer backbone (ring attention) ----
        for li in 0..n_layers() {
            self.layer_forward(li, ntot, crate::dflash2::sliding_window(), false);
        }
        // final norm
        self.rmsnorm(&self.blk.h_final, &self.blk.h, &self.glob.norm, hidden(), block());
        // ---- borrowed head at N=7 on h_final cols 1..7 ----
        self.head_logits();
        // ---- top-16 on the 7 MASK rows ----
        klaunch!(self, "top16_b", (levels() as u32, 1, 1), (256, 1, 1), 0,
            (d(&self.out_vals), d(&self.out_ids), d(&self.logits), vocab() as i32, levels() as i32));
        // ---- hidden_projection [256,5120] × h_sel [5120,7] → hp ----
        self.gemm_dsp(&self.hp_bf16, &self.hp_w, d(&self.blk.h_final) + (hidden() * 2) as u64, selector_rank(), hidden());
        klaunch!(self, "bf16tof32", grid(selector_rank() * block()), (256, 1, 1), 0,
            (d(&self.hp_f32), d(&self.hp_bf16), (selector_rank() * block()) as i32));
        // ---- the chain: greedy (device unary, no flip) or S5F2 sampled ----
        match sample {
            Some((seeds, temperature)) => {
                // The sampled walk's per-position seeds: host writes them into the round's
                // scratch; the walk draws the multinomial and writes tokens + q_rows + the
                // candidate (token, q) table for the real-q verify's exact residual.
                self.dev.htod_sync_copy_into(seeds, &mut self.walk_seeds).expect("htod walk seeds");
                klaunch!(self, "df2_sel_walk_sample_b", (1u32, 1, 1), (256, 1, 1), 0,
                    (d(&self.walk_out_tok), d(&self.walk_out_q),
                     d(&self.hp_f32), d(&self.out_ids), d(&self.out_vals),
                     d(&self.pred_cb), d(&self.succ_cb),
                     anchor, if graph_mode { self.anchor_dev } else { 0 },
                     d(&self.walk_seeds), temperature,
                     ((selector_rank() as i32) << 8) | levels() as i32));
            }
            None => {
                klaunch!(self, "df2_sel_walk_b", (1u32, 1, 1), (256, 1, 1), 0,
                    (d(&self.walk_tokens), d(&self.walk_scores), d(&self.hp_f32), d(&self.out_ids),
                     d(&self.out_vals), d(&self.pred_cb), d(&self.succ_cb),
                     anchor, if graph_mode { self.anchor_dev } else { 0 },
                     selector_rank() as i32, levels() as i32));
            }
        }
        Ok(())
    }

    /// S5F2 — the SAMPLED draft round (L2's selector path): the S4F round sequence with the
    /// multinomial walk at `temperature`. Returns the drawn draft tokens, their per-position
    /// selector probabilities (`q_rows`), and the full candidate (token, q) table — the inputs
    /// the real-q verify (`verify_forward_sample_rq`) needs for the accept (u·q < p) and the
    /// exact relu(p−q) residual. Eager-only (the greedy round stays graph-captured).
    pub fn draft_round_dev_sample(&mut self, anchor: u32, seeds: &[u32], temperature: f32)
        -> Result<Df2SampleOut> {
        // DF2 block-16: the chain length is the live block-1 (7 or 15), and the candidate
        // tables the walk writes are `[L]` drawn tokens/q followed by `[L][16]` candidate ids.
        let l_ = levels();
        assert_eq!(seeds.len(), l_,
            "draft_round_dev_sample needs block()-1 = {l_} selector seeds");
        let toks: Vec<i32> = [anchor as i32].iter().copied()
            .chain(std::iter::repeat(crate::dflash2::mask_token_id() as i32).take(block() - 1)).collect();
        self.dev.htod_sync_copy_into(&toks, &mut self.toks_blk)?;
        let (ntot_save, anchor_save) = (self.ntot_dev, self.anchor_dev);
        self.ntot_dev = 0;
        self.anchor_dev = 0;
        let r = self.draft_round_kernels(anchor, false, Some((seeds, temperature)));
        self.ntot_dev = ntot_save;
        self.anchor_dev = anchor_save;
        r?;
        self.dsync()?;
        let out_tok: Vec<u32> = self.dev.dtoh_sync_copy(&self.walk_out_tok)?.to_vec();
        let out_q: Vec<f32> = self.dev.dtoh_sync_copy(&self.walk_out_q)?.to_vec();
        let tokens: Vec<u32> = out_tok[..l_].to_vec();
        let cand_tok: Vec<u32> = out_tok[l_..l_ + l_ * 16].to_vec();
        let q_rows: Vec<f32> = out_q[..l_].to_vec();
        let cand_q: Vec<f32> = out_q[l_..l_ + l_ * 16].to_vec();
        Ok(Df2SampleOut { tokens, q_rows, cand_tok, cand_q })
    }

    /// PLAN/25 Phase 1 (tree-verify spike): read back the per-level candidate table the
    /// selector walk ALREADY wrote — `walk_out_tok[7..7+112]` / `walk_out_q[7..7+112]`, the
    /// top-16 (token, q) pairs for each of the 7 MASK levels. Zero extra GPU work: the buffers
    /// are populated by the same round replay (graph or eager) that produced the chain; this is
    /// one ~760 B dtoh readback. Valid only between a round run and the next one.
    pub fn tree_tables(&self) -> Result<(Vec<u32>, Vec<f32>)> {
        // Only reachable at block 8: `df2_tree_step` hard-refuses block 16 (31 columns > MAX_VERIFY).
        let l_ = levels();
        let cand_tok: Vec<u32> = self.dev.dtoh_sync_copy(&self.walk_out_tok)?.to_vec();
        let cand_q: Vec<f32> = self.dev.dtoh_sync_copy(&self.walk_out_q)?.to_vec();
        Ok((cand_tok[l_..l_ + l_ * 16].to_vec(), cand_q[l_..l_ + l_ * 16].to_vec()))
    }

    /// PLAN/25 Phase 1: gather the accepted tree path's tap columns out of the WIDE tree sink
    /// into this round's 8-column staging, in ring order at cols [0..path.len()) —
    /// `inject_dev(path.len())` then works unchanged (path.len() <= BLOCK: the accepted path
    /// spans at most one branch's depth). Stream-ordered 2D copies, no sync.
    pub fn sync_staging_from_wide(&mut self, wide: &Df2TapSink, path: &[usize]) -> Result<()> {
        let k = path.len();
        assert!(k >= 1 && k <= block(), "sync_staging_from_wide path len {k}");
        let dst = *self.staging.device_ptr() as u64;
        let src = *wide.staging.device_ptr() as u64;
        // Staging layout is [token][TAP_CONCAT_DIM] row-major (token t's features = row t); the
        // gather is a whole-row copy per accepted token — one driver call each.
        for j in 0..k {
            copy_row_into(&self.dev, self.stream.stream,
                          dst + (j * tap_concat_dim() * 2) as u64,
                          src + (path[j] * tap_concat_dim() * 2) as u64,
                          tap_concat_dim());
        }
        Ok(())
    }

    /// S5F — capture the draft-round kernel sequence as a CUDA graph (the MTP verify-graph
    /// pattern): the captured region is kernel-only (embed -> layers -> norm -> head -> top16 ->
    /// hp -> walk); per-replay inputs (anchor, nprev) are device ints written before each replay.
    /// The attention smem is allocated at the MAX ntot (max_c + BLOCK) so every replay fits.
    /// Returns None when capture is unsupported (the eager path stays).
    pub fn capture_round_graph(&mut self) -> bool {
        use cudarc::driver::sys;
        if self.round_graph.is_some() { return true; }
        // allocate the per-replay device ints + the anchor buffer
        let dev = self.dev.clone();
        let (mut ntot_int, mut anchor_buf) = match (dev.alloc_zeros::<i32>(1), dev.alloc_zeros::<u32>(1)) {
            (Ok(a), Ok(b)) => (a, b),
            _ => return false,
        };
        if dev.htod_sync_copy_into(&[(self.max_c + block()) as i32], &mut ntot_int).is_err() { return false; }
        if dev.htod_sync_copy_into(&[0u32], &mut anchor_buf).is_err() { return false; }
        self.ntot_dev = *ntot_int.device_ptr() as u64;
        self.anchor_dev = *anchor_buf.device_ptr() as u64;
        // position the block at max_c so the recorded launches (smem, grid) cover every replay.
        let pos_max: Vec<i32> = (0..block()).map(|b| (self.max_c + b) as i32).collect();
        self.pos_at = None; // the capture parks the block at max_c
        if dev.htod_sync_copy_into(&pos_max, &mut self.pos_blk).is_err() { return false; }
        self.gather_rope(&self.blk.cos8, &self.blk.sin8, d(&self.pos_blk), block());
        if dev.synchronize().is_err() { return false; }
        let stream = self.stream.stream;
        // The recorded launch configs must cover EVERY replay. S10R: the attention's dynamic
        // smem is ctx-INDEPENDENT (band_smem(), 8,864 B constant) and the grid is
        // BLOCK*NUM_HEADS — but capture still positions the block at max_c and sets nprev to
        // the max so the recorded packed args mirror the worst case, then restores. (The
        // kernels are only RECORDED here — nothing executes, so the ring contents are
        // irrelevant; ntot_dev makes the replayed ntot device-driven.)
        let nprev_save = self.nprev;
        self.nprev = self.max_c;
        let r = unsafe { sys::cuStreamBeginCapture_v2(stream, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) };
        if r != sys::CUresult::CUDA_SUCCESS {
            self.nprev = nprev_save;
            crate::metrics::graph_capture_lost("df2-draft-round", "draft round graph", &format!("BeginCapture {r:?}")); // S-B9-REL-INTEG C1
            return false;
        }
        let ok = self.draft_round_kernels(0, true, None).is_ok();
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let r = unsafe { sys::cuStreamEndCapture(stream, &mut graph) };
        self.nprev = nprev_save;
        if !ok || r != sys::CUresult::CUDA_SUCCESS {
            crate::metrics::graph_capture_lost("df2-draft-round", "draft round graph",
                                               &if r != sys::CUresult::CUDA_SUCCESS { format!("{r:?}") } else { "body error".to_string() }); // S-B9-REL-VISIONFIX2 V2b
            return false;
        }
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let r = unsafe { sys::cuGraphInstantiate_v2(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0) };
        unsafe { sys::cuGraphDestroy(graph); }
        if r != sys::CUresult::CUDA_SUCCESS {
            crate::metrics::graph_capture_lost("df2-draft-round", "draft round graph", &format!("instantiate {r:?}")); // S-B9-REL-INTEG C1
            return false;
        }
        self.round_graph = Some(crate::gpu::CudaGraph::from_exec(exec, stream));
        // keep the device ints alive for the lifetime of the graph (they are captured by addr)
        self.ntot_buf = Some(ntot_int);
        self.anchor_buf = Some(anchor_buf);
        true
    }

    /// S5F — replay the captured draft round with the given anchor. Host writes the per-replay
    /// device inputs (anchor, nprev, block positions) + gathers the block RoPE eagerly, then ONE
    /// graph launch, then the 7-token readback. Bit-identical to the eager path (same kernels,
    /// same order — the graph determinism is asserted by the probe). Falls back to the eager
    /// path when no graph is captured.
    /// PLAN/25 Phase 1: the top-16 candidate table ([7][16] u32, head-logit rank order) — the
    /// tree lane's B-branch source. Written by the top16 stage on EVERY round path (the graph
    /// replay included), so the tree lane can run the CAPTURED graph and pay one 448 B dtoh
    /// instead of the eager full round's ~640 KB `h_final` readback.
    pub fn candidate_table(&self) -> Result<Vec<u32>> {
        Ok(self.dev.dtoh_sync_copy(&self.out_ids)?.to_vec())
    }

    pub fn draft_round_graph(&mut self, anchor: u32) -> Result<Vec<u32>> {
        let Some(g) = self.round_graph.as_ref() else {
            return self.draft_round_dev(anchor);
        };
        // F8 (diagnostic, --round-trace): split the replay wall — input H2D syncs, the eager
        // rope gather, the pre/post syncs, the graph launch, the token readback.
        let tr = crate::opts::var(crate::opt!("round-trace")).is_ok();
        let mut t = std::time::Instant::now();
        let ntot = self.nprev + block();
        // device inputs for this replay — ONE host sync, then stream-ordered D2D fanning.
        // (B2: the four per-replay `htod_sync_copy_into` calls cost ~1.75 ms/step of host
        // round-trips; a single merged upload + raw `memcpy_dtod_async` on the blocking stream
        // keeps every captured kernel address intact at ~1/4 the host cost. Bit-exact: the
        // anchor rides the same bytes it always did, only the transport changes.)
        let mut stage: Vec<i32> = Vec::with_capacity(2 * block() + 2);
        stage.push(anchor as i32);
        stage.extend(std::iter::repeat(crate::dflash2::mask_token_id() as i32).take(block() - 1));
        stage.extend((0..block()).map(|b| (self.nprev + b) as i32));
        stage.push(ntot as i32);
        stage.push(anchor as i32);
        self.dev.htod_sync_copy_into(&stage, &mut self.h2d_stage)?;
        {
            let sbase = d(&self.h2d_stage);
            let stream = self.stream.stream;
            let cpy = |dst: u64, src_byte_off: u64, bytes: usize| unsafe {
                cudarc::driver::result::memcpy_dtod_async(dst, sbase + src_byte_off, bytes, stream)
                    .unwrap();
            };
            cpy(d(&self.toks_blk), 0, block() * 4);
            cpy(d(&self.pos_blk), block() as u64 * 4, block() * 4);
            cpy(d(self.ntot_buf.as_ref().expect("ntot")), 2 * block() as u64 * 4, 4);
            cpy(d(self.anchor_buf.as_ref().expect("anchor")), (2 * block() + 1) as u64 * 4, 4);
        }
        if tr { eprintln!("[round-trace] h2d inputs: {:.3} ms", t.elapsed().as_secs_f32() * 1e3); t = std::time::Instant::now(); }
        self.gather_rope(&self.blk.cos8, &self.blk.sin8, d(&self.pos_blk), block());
        if tr { eprintln!("[round-trace] gather_rope launch: {:.3} ms", t.elapsed().as_secs_f32() * 1e3); t = std::time::Instant::now(); }
        self.dsync()?;
        if tr { eprintln!("[round-trace] pre-replay sync: {:.3} ms", t.elapsed().as_secs_f32() * 1e3); t = std::time::Instant::now(); }
        g.launch();
        if tr { eprintln!("[round-trace] graph launch (async): {:.3} ms", t.elapsed().as_secs_f32() * 1e3); t = std::time::Instant::now(); }
        self.dsync()?;
        if tr { eprintln!("[round-trace] replay wait: {:.3} ms", t.elapsed().as_secs_f32() * 1e3); t = std::time::Instant::now(); }
        let tokens: Vec<u32> = self.dev.dtoh_sync_copy(&self.walk_tokens)?.to_vec();
        if tr { eprintln!("[round-trace] tokens dtoh: {:.3} ms", t.elapsed().as_secs_f32() * 1e3); }
        Ok(tokens)
    }

    /// One draft layer on the ring KV (S3F's sequence; the attention is the ring variant).
    /// P2 sharded mode: the q/k/v/o + gate/up/down launches run at the RANK-LOCAL widths and
    /// `o_proj`/`down_proj` produce PARTIAL [5120×B] outputs, landed by an all-reduce at each
    /// sublayer boundary (the trunk's reduce-site pattern — the next sublayer's rmsnorm needs
    /// the full-hidden sum, so the AR cannot be deferred past it). The conv/norm/head path and
    /// every buffer address are IDENTICAL to the unsharded launch sequence, so the captured
    /// graph keeps the same node structure per rank.
    fn layer_forward(&self, li: usize, ntot: usize, window: usize, ctl_dual_write: bool) {
        let l = &self.layers[li];
        let blk = &self.blk;
        let base1_off = (conv_kernel() * hidden() * 2) as u64;
        // P2: local widths (the full constants when unsharded — the dims below are then
        // byte-identical to the pre-P2 launches).
        let (nq, nkv, ni) = (self.nq_l, self.nkv_l, self.ni_l);
        let sharded = self.ar.is_some();
        // attention sublayer
        self.rmsnorm(&blk.normed, &blk.h, &l.input_ln, hidden(), block());
        self.gemm_lin(&blk.dyn_attn, &l.attn_kp, d(&blk.normed), 2 * conv_kernel() * conv_groups(), hidden());
        self.conv2(&blk.x_conv, &blk.normed, &blk.dyn_attn, d(&l.attn_base), block(), 0);
        self.gemm_lin(&blk.q, &l.q_proj, d(&blk.x_conv), nq * head_dim(), hidden());
        self.rmsnorm_perhead(&blk.q, &l.q_norm, nq, block());
        self.rope(&blk.q, &blk.cos8, &blk.sin8, nq, block());
        self.gemm_lin(&blk.k, &l.k_proj, d(&blk.x_conv), nkv * head_dim(), hidden());
        self.rmsnorm_perhead(&blk.k, &l.k_norm, nkv, block());
        self.rope(&blk.k, &blk.cos8, &blk.sin8, nkv, block());
        self.gemm_lin(&blk.v, &l.v_proj, d(&blk.x_conv), nkv * head_dim(), hidden());
        // block rows → ring rows [ring(), ring()+8) (+ the linear control copy at [nprev, nprev+8))
        klaunch!(self, "write_kv_b", grid(block() * nkv * head_dim()), (256, 1, 1), 0,
            (d(&self.k_ring[li]), d(&self.v_ring[li]), d(&blk.k), d(&blk.v),
             d(&self.wrow_blk), ring_stride() as i32, nkv as i32, head_dim() as i32,
             block() as i32, d(&blk.slot_ids)));
        if ctl_dual_write && ntot <= ring() {
            klaunch!(self, "write_kv_b", grid(block() * nkv * head_dim()), (256, 1, 1), 0,
                (d(&self.k_ring[li]), d(&self.v_ring[li]), d(&blk.k), d(&blk.v),
                 d(&self.wrow_ctl), ring_stride() as i32, nkv as i32, head_dim() as i32,
                 block() as i32, d(&blk.slot_ids)));
        }
        let scale = 1.0f32 / (head_dim() as f32).sqrt();
        let smem = crate::dflash2::band_smem(window, ntot);
        let nh_packed = ((nq << 20) | (head_dim() << 10) | nkv) as i32;
        let packed = (((ntot as u64) << 32) | ((ring() as u64) << 16) | ring_stride() as u64);
        let window_b = ((window << crate::dflash2::DF2_WB_SHIFT) | block()) as i32;
        klaunch!(self, "gqa_attn_band_ring_b", ((block() * nq) as u32, 1, 1), (head_dim() as u32, 1, 1), smem as u32,
            (d(&blk.attn), d(&blk.q), d(&self.k_ring[li]), d(&self.v_ring[li]),
             d(&self.pos_blk), packed, self.ntot_dev, nh_packed, window_b, fbits(scale)));
        if ctl_dual_write && li == 0 {
            // the ring-vs-linear control: (a) the ring kernel again into a reference buffer
            // (deterministic -> bitwise copy of what just ran), (b) the S3F linear kernel over
            // the same cache's dual-written linear rows.
            klaunch!(self, "gqa_attn_band_ring_b", ((block() * nq) as u32, 1, 1), (head_dim() as u32, 1, 1), smem as u32,
                (d(&self.ctl_attn_ref), d(&blk.q), d(&self.k_ring[li]), d(&self.v_ring[li]),
                 d(&self.pos_blk), packed, self.ntot_dev, nh_packed, window_b, fbits(scale)));
            let lin_stride = (((ntot as u64) << 16) | ring_stride() as u64) as i64;
            klaunch!(self, "gqa_attn_band_b", ((block() * nq) as u32, 1, 1), (head_dim() as u32, 1, 1), smem as u32,
                (d(&self.ctl_attn), d(&blk.q), d(&self.k_ring[li]), d(&self.v_ring[li]),
                 d(&self.pos_blk), lin_stride, nh_packed, window_b, fbits(scale)));
        }
        self.gemm_lin(&blk.attn_out, &l.o_proj, d(&blk.attn), hidden(), nq * head_dim());
        if sharded {
            // AR site 1: land the rank's o_proj partial (K-split over q heads) — conv2's
            // channel-local finish and the residual then run on the full sum.
            self.tp_ar_bf16(d(&blk.attn_out), hidden() * block());
        }
        self.conv2(&blk.fin, &blk.attn_out, &blk.dyn_attn, d(&l.attn_base) + base1_off, block(), 1);
        klaunch!(self, "add_residual_b", grid(hidden() * block()), (256, 1, 1), 0,
            (d(&blk.h), d(&blk.h), d(&blk.fin), (hidden() * block()) as i32));
        // mlp sublayer
        self.rmsnorm(&blk.normed2, &blk.h, &l.post_ln, hidden(), block());
        self.gemm_lin(&blk.dyn_mlp, &l.mlp_kp, d(&blk.normed2), 2 * conv_kernel() * conv_groups(), hidden());
        self.conv2(&blk.x_conv2, &blk.normed2, &blk.dyn_mlp, d(&l.mlp_base), block(), 0);
        self.gemm_lin(&blk.gate, &l.gate_proj, d(&blk.x_conv2), ni, hidden());
        self.gemm_lin(&blk.up, &l.up_proj, d(&blk.x_conv2), ni, hidden());
        klaunch!(self, "silu_mul_b", grid(ni * block()), (256, 1, 1), 0,
            (d(&blk.gate), d(&blk.gate), d(&blk.up), (ni * block()) as i32));
        self.gemm_lin(&blk.mlp_out, &l.down_proj, d(&blk.gate), hidden(), ni);
        if sharded {
            // AR site 2: land the rank's down_proj partial (K-split over inter rows).
            self.tp_ar_bf16(d(&blk.mlp_out), hidden() * block());
        }
        self.conv2(&blk.fin2, &blk.mlp_out, &blk.dyn_mlp, d(&l.mlp_base) + base1_off, block(), 1);
        klaunch!(self, "add_residual_b", grid(hidden() * block()), (256, 1, 1), 0,
            (d(&blk.h), d(&blk.h), d(&blk.fin2), (hidden() * block()) as i32));
    }

}
