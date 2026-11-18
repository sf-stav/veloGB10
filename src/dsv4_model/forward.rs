//! `Dsv4GpuModel` forwards: trunk-top head, embed, prefill (chunked), decode (eager and graphed), main-hidden
//! capture, verify, verify-state snapshots / restore and `reset_states`.
//! Stream invariant (AGENTS 2.1): every launch rides the runtime's blocking compute stream.
//! Siblings: `load.rs` (loaders), `prefix_cache.rs` (prefix-cached forwards); `mod.rs` has the map.

use super::*;

impl Dsv4GpuModel {
    /// Trunk-top forward: `hc_head` (sigmoid-only collapse) → final RMSNorm → LM head on the LAST
    /// row. `x` [s, hc*dim] bf16 streams → (collapsed [s, dim] bf16, logits [vocab] f32). Mirrors
    /// `dsv4_cpu::run_head_piece` exactly (§B.8 hc_head + §A.1 ParallelHead, last-position-sliced).
    pub fn forward_head<X: crate::dsv4_gpu::Dsv4Buf<bf16>, F: crate::dsv4_gpu::Dsv4Buf<f32>>(&self, x: &X, s: usize) -> Result<(X, F)> {
        use cudarc::driver::result;
        let (dim, vocab) = (self.cfg.dim, self.cfg.vocab_size);
        let collapsed = self
            .rt
            .hc_head(x, &self.hc_head_fn, &self.hc_head_base, &self.hc_head_scale, s, &self.cfg)
            .context("hc_head")?;
        let yn = self
            .rt
            .rmsnorm(&collapsed, &self.norm, s, dim, self.cfg.norm_eps)
            .context("final rmsnorm")?;
        // LM head on the LAST row only (the reference slices yn[-1:] — §A.1 last-position). d2d
        // the last row into a fresh dim-wide buffer (8 KB — noise on the blocking stream, no
        // host round-trip).
        let last = X::alloc_zeros(&self.dev, self.rt.stream.stream, dim)?;
        unsafe {
            result::memcpy_dtod_async(
                last.dptr(),
                yn.dptr() + ((s - 1) * dim * 2) as u64,
                dim * 2,
                self.rt.stream.stream,
            )
            .map_err(|e| anyhow!("forward_head last-row d2d: {e}"))?;
        }
        let logits = self.rt.lm_head(&self.head, &last, 1, dim, vocab).context("lm_head")?;
        Ok((collapsed, logits))
    }

    /// R3.2 (L2): trunk top → next token via the vocab-parallel maxloc head. Returns `None`
    /// when TP is not attached (single-process — the caller uses the full-logits path).
    /// Bitwise: the winner equals the full-vocab host argmax exactly (see `lm_head_maxloc_tp`).
    pub fn forward_head_next(&self, x: &B, s: usize) -> Result<Option<u32>> {
        if self.rt.tp_ctx_dptr == 0 || self.tp_rank < 0 {
            return Ok(None);
        }
        use cudarc::driver::result;
        let (dim, vocab) = (self.cfg.dim, self.cfg.vocab_size);
        let collapsed = self
            .rt
            .hc_head(x, &self.hc_head_fn, &self.hc_head_base, &self.hc_head_scale, s, &self.cfg)
            .context("hc_head")?;
        let yn = self
            .rt
            .rmsnorm(&collapsed, &self.norm, s, dim, self.cfg.norm_eps)
            .context("final rmsnorm")?;
        let mut last = self.dev.alloc_zeros::<bf16>(dim)?;
        unsafe {
            result::memcpy_dtod_async(
                *last.device_ptr(),
                *yn.device_ptr() + ((s - 1) * dim * 2) as u64,
                dim * 2,
                self.rt.stream.stream,
            )
            .map_err(|e| anyhow!("forward_head_next last-row d2d: {e}"))?;
        }
        Ok(Some(self.rt.lm_head_maxloc_tp(&self.head, &last, dim, vocab, self.tp_rank as usize)?))
    }

    /// Trunk forward + next token in one call (the R3.2 greedy-decode step). `None` when TP is
    /// not attached (the caller falls back to `forward` + host argmax).
    pub fn forward_next(&mut self, ids: &[i32], start_pos: usize) -> Result<Option<u32>> {
        if self.rt.tp_ctx_dptr == 0 || self.tp_rank < 0 {
            return Ok(None);
        }
        let s = ids.len();
        let x = self.forward_streams(ids, start_pos)?;
        self.forward_head_next(&x, s)
    }

    /// Embed `ids` [s] → [s, hc*dim] bf16 streams (gather + replicate ×hc, §1.1 forward_embed).
    pub fn embed_tokens<X: crate::dsv4_gpu::Dsv4Buf<bf16>>(&self, ids: &CudaSlice<i32>, s: usize) -> Result<X> {
        self.rt.embed_tokens(&self.embed, ids, s, &self.cfg)
    }

    /// Trunk streams forward (the chunkable body of [`forward`](Self::forward)): embed → N
    /// layers (block_forward) → final `x` streams [s, hc*dim] bf16. NO trunk top (hc_head /
    /// norm / LM head) — the caller runs [`forward_head`](Self::forward_head) once on the
    /// tail. `ids` [s] are the input token ids (hash-router table keys for layers 0–2).
    /// `start_pos` is the absolute position of `ids[0]` (0 for prefill; current len for
    /// decode / chunk continuation). One call processes ≤ s_max rows; chunk boundaries at
    /// multiples of 128 carry all recurrent state (compressor frontier, indexer cache,
    /// window ring) in `self.states` — bitwise-identical to one-shot (§12.B.5).
    pub fn forward_streams(&mut self, ids: &[i32], start_pos: usize) -> Result<B> {
        let s = ids.len();
        let ids_dev = self.dev.htod_sync_copy(ids)?;
        self.forward_streams_dev::<B, S, CudaSlice<i32>, CudaSlice<u8>, CudaSlice<u32>>(&ids_dev, s, start_pos)
    }

    /// `forward_streams` with the ids already on device (the CUDA-graph path: the graph's
    /// persistent ids buffer is re-uploaded OUTSIDE the graph before each replay).
    pub fn forward_streams_dev<X: crate::dsv4_gpu::Dsv4Buf<bf16>, F: crate::dsv4_gpu::Dsv4Buf<f32>, I: crate::dsv4_gpu::Dsv4Buf<i32>, C: crate::dsv4_gpu::Dsv4Buf<u8>, U: crate::dsv4_gpu::Dsv4Buf<u32>>(&mut self, ids_dev: &CudaSlice<i32>, s: usize, start_pos: usize) -> Result<X> {
        anyhow::ensure!(
            self.layers.len() == self.states.len(),
            "layers/states length mismatch ({} vs {}) — load_trunk_top has no layers",
            self.layers.len(),
            self.states.len()
        );
        let mut x = self.embed_tokens::<X>(ids_dev, s)?;
        for i in 0..self.layers.len() {
            let o = self
                .rt
                .block_forward::<X, F, I, C, U>(&self.layers[i], &mut self.states[i], &mut self.scratch, &x, s, start_pos, ids_dev, &self.cfg)
                .with_context(|| format!("block_forward layer {i}"))?;
            x = o.y;
        }
        Ok(x)
    }

    /// Chunked prefill (§12.B.5): process `ids` (a full prompt, start_pos 0) in `chunk`-sized
    /// pieces — `chunk` MUST be a multiple of 128 (compressor-block alignment; PREFILL_CHUNK
    /// in production, smaller in gates) — and return the LAST chunk's streams [cs, hc*dim]
    /// bf16 + its row count (the trunk top runs once on that tail). Recurrent state carries
    /// across chunks in `self.states`; the result is bitwise-identical to one-shot prefill.
    pub fn forward_prefill_chunked(&mut self, ids: &[i32], chunk: usize) -> Result<(B, usize)> {
        anyhow::ensure!(chunk % 128 == 0 && chunk > 0, "prefill chunk {chunk} not a multiple of 128 (§12.B.5)");
        let s = ids.len();
        anyhow::ensure!(s > 0, "forward_prefill_chunked: empty prompt");
        let trace = crate::opts::var(crate::opt!("prefill-trace")).ok().is_some();
        let t0 = if trace { Some(std::time::Instant::now()) } else { None };
        let mut tail: Option<(B, usize)> = None;
        let mut c0 = 0usize;
        let mut n_chunks = 0usize;
        while c0 < s {
            let cs = chunk.min(s - c0);
            let tc = if trace { Some(std::time::Instant::now()) } else { None };
            let x = self
                .forward_streams(&ids[c0..c0 + cs], c0)
                .with_context(|| format!("prefill chunk @{c0} ({cs}/{s} tok)"))?;
            if let (Some(tc), Some(t0)) = (tc, t0) {
                self.rt.dev.synchronize().ok();
                eprintln!("[dsv4-pf] chunk @{c0} cs={cs}/{s} chunk_ms={} total_ms={}", tc.elapsed().as_millis(), t0.elapsed().as_millis());
            }
            tail = Some((x, cs));
            c0 += cs;
            n_chunks += 1;
        }
        if trace {
            eprintln!("[dsv4-pf] forward_prefill_chunked: {s} tok in {n_chunks} chunks, total_ms={}", t0.map(|t| t.elapsed().as_millis()).unwrap_or(0));
        }
        Ok(tail.expect("non-empty prompt produced no chunks"))
    }

    /// Full trunk forward: embed → N layers (block_forward) → hc_head → norm → LM head.
    /// `ids` [s] are the input token ids (hash-router table keys for layers 0–2). `start_pos` is
    /// the absolute position of `ids[0]` (0 for prefill; current len for decode). Returns the
    /// last-row fp32 logits [vocab]. `x` streams flow layer-to-layer in fp32-bf16 (hc_post output).
    /// A prefill longer than PREFILL_CHUNK is chunked (see forward_prefill_chunked) — this is
    /// what makes >4K prompts (200K, 1M) fit chunk-sized prefill scratch.
    pub fn forward(&mut self, ids: &[i32], start_pos: usize) -> Result<S> {
        let s = ids.len();
        let trace = crate::opts::var(crate::opt!("prefill-trace")).ok().is_some();
        let t0 = if trace { Some(std::time::Instant::now()) } else { None };
        if start_pos == 0 && s > PREFILL_CHUNK {
            let (x_tail, tail_len) = self.forward_prefill_chunked(ids, PREFILL_CHUNK)?;
            let (_collapsed, logits) = self.forward_head(&x_tail, tail_len)?;
            if let Some(t0) = t0 {
                self.rt.dev.synchronize().ok();
                eprintln!("[dsv4-pf] forward: {s} tok (chunked) total_ms={} tok/s={:.1}",
                    t0.elapsed().as_millis(), s as f64 / t0.elapsed().as_secs_f64());
            }
            return Ok(logits);
        }
        // CUDA-graph decode (R3A.1/E2, opt-in --graph=1): single-token decode replays a
        // per-fire-variant whole-forward graph instead of ~1600 eager launches. Bitwise by
        // construction (same kernels/args; only the launch vehicle changes) — verified by the
        // classifier's baked-value checks at capture + the replay≡eager gate in the tests.
        // Eager fallback: any capture/classify error poisons that variant (loud, once).
        if s == 1 && start_pos > 0 && crate::dsv4_gpu::env_flag_once(crate::opt!("graph")) {
            // Classifier disambiguation floor (gather CSA-vs-HCA baked values diverge) and
            // the hierarchical-topk regime limit (>16384 blocks ⇒ kernel sequence changes).
            if start_pos >= 130 && (start_pos + 1) / 4 <= 16384 {
                match self.forward_decode_graphed(ids[0], start_pos) {
                    Ok(logits) => return Ok(logits),
                    Err(e) => eprintln!("[dsv4-graph] eager fallback at sp={start_pos}: {e:#}"),
                }
            }
        }
        let x = self.forward_streams(ids, start_pos)?;
        let (_collapsed, logits) = self.forward_head(&x, s)?;
        if let Some(t0) = t0 {
            self.rt.dev.synchronize().ok();
            eprintln!("[dsv4-pf] forward: {s} tok (one-shot sp={start_pos}) total_ms={} tok/s={:.1}",
                t0.elapsed().as_millis(), s as f64 / t0.elapsed().as_secs_f64());
        }
        Ok(logits)
    }

    /// CUDA-graph decode step (the `--graph=1` arm of [`forward`](Self::forward)).
    /// Lazy per-variant capture: the capture run RECORDS (does not execute), so the token's
    /// output is produced by the replay that immediately follows — state advances exactly
    /// once, identical to an eager step. Capture/classify errors poison the variant
    /// (eager forever after, loudly) — they never execute partial GPU work, so the eager
    /// fallback is always safe. Replay errors after cuGraphLaunch propagate (state unknown).
    /// Pub for the replay≡eager bitwise gate (tests/dsv4_graph_test.rs).
    pub fn forward_decode_graphed(&mut self, id: i32, start_pos: usize) -> Result<S> {
        use crate::dsv4_graph::DecodeGraphs;
        // lazy init: func map from the runtime's kernel modules (spine + attn + comp both).
        if self.graphs.is_none() {
            let mut func_names: std::collections::HashMap<usize, &'static str> = std::collections::HashMap::new();
            for (n, f) in self.rt.spine.func_handles() {
                func_names.insert(f as usize, Box::leak(n.to_string().into_boxed_str()));
            }
            for (n, f) in self.rt.attn.func_handles() {
                func_names.insert(f as usize, Box::leak(n.to_string().into_boxed_str()));
            }
            if let Some(ck) = &self.rt.comp {
                for (n, f) in ck.comp.func_handles() {
                    func_names.insert(f as usize, Box::leak(n.to_string().into_boxed_str()));
                }
                for (n, f) in ck.spine.func_handles() {
                    func_names.insert(f as usize, Box::leak(n.to_string().into_boxed_str()));
                }
            }
            crate::dsv4_graph::raise_mempool_threshold(&self.dev)?;
            // create the dedicated GSlice pool EAGERLY (lazy creation inside capture would
            // be an illegal device-state change mid-capture).
            let _ = crate::dsv4_gpu::graph_mempool(&self.dev);
            // the decode-graph workspace slab: all transient GSlice allocs inside capture
            // become bump slices of it (NO alloc nodes in the graph — the only sound
            // multi-launch pattern, measured 2026-07-30). 256 MB >> the ~20 MB forward
            // transient footprint; overflow panics loudly with the required size.
            crate::dsv4_gpu::graph_ws_init(&self.dev, 256 * 1024 * 1024)?;
            self.graphs = Some(DecodeGraphs::new(
                &self.dev, func_names, self.rt.window,
                self.cfg.n_routed_experts, self.cfg.n_activated_experts,
                self.cfg.vocab_size,
            )?);
        }
        // Take the graphs out so the capture closure can call &mut self methods freely;
        // restored on every path (poison marks included).
        let mut graphs = self.graphs.take().unwrap();
        let result = self.graphed_step(&mut graphs, id, start_pos);
        self.graphs = Some(graphs);
        result
    }

    fn graphed_step(
        &mut self,
        graphs: &mut crate::dsv4_graph::DecodeGraphs,
        id: i32,
        start_pos: usize,
    ) -> Result<S> {
        use crate::dsv4_graph::{Slot, Variant};
        let variant = Variant::of(start_pos);
        if matches!(graphs.slot_ref(variant), Slot::Poisoned) {
            return Err(anyhow::anyhow!("variant {variant:?} poisoned (earlier capture failure)"));
        }
        if matches!(graphs.slot_ref(variant), Slot::Unborn) {
            // ---- capture (records, does NOT execute) ----
            let cap_result = (|| {
                // upload this token's id BEFORE capture (the graph reads the buffer, baked).
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(
                        *graphs.ids_dev.device_ptr(),
                        &[id],
                        self.rt.stream.stream,
                    )
                    .map_err(|e| anyhow::anyhow!("graph capture ids upload: {e}"))?;
                }
                let r = unsafe {
                    cudarc::driver::sys::cuStreamBeginCapture_v2(
                        self.rt.stream.stream,
                        cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
                    )
                };
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(anyhow::anyhow!("cuStreamBeginCapture: {r:?}"));
                }
                // The GSlice family: every alloc/free/memset in the forward runs on
                // rt.stream (capture-legal — the blocker that killed the CudaSlice
                // attempt: INVALIDATED / STREAM_CAPTURE_IMPLICIT, DSV4_R3A.md §8).
                // GRAPH_CAPTURE_ACTIVE makes GSlice::drop leak to the graph pool instead
                // of freeing graph-owned memory outside the graph.
                crate::dsv4_gpu::GRAPH_CAPTURE_ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
                crate::dsv4_gpu::graph_ws_begin_capture();
                // ONE slab-wide memset as the graph's first node (replaces ~150 per-alloc
                // memset nodes — the bump regions are contiguous; zero-at-start is
                // value-identical since no two sites share a region within one forward).
                unsafe {
                    let (d, b) = crate::dsv4_gpu::graph_ws_span();
                    cudarc::driver::result::memset_d8_async(d, 0, b, self.rt.stream.stream)
                        .map_err(|e| anyhow::anyhow!("graph slab memset: {e}"))?;
                }
                let fwd = (|| {
                    let x = self.forward_streams_dev::<
                        crate::dsv4_gpu::GB, crate::dsv4_gpu::GS,
                        crate::dsv4_gpu::GSlice<i32>, crate::dsv4_gpu::GSlice<u8>, crate::dsv4_gpu::GSlice<u32>,
                    >(&graphs.ids_dev, 1, start_pos)?;
                    let (_collapsed, logits) = self.forward_head::<crate::dsv4_gpu::GB, crate::dsv4_gpu::GS>(&x, 1)?;
                    // AUTO_FREE_ON_LAUNCH frees every in-graph allocation at launch end —
                    // the output MUST be copied to the persistent eager buffer INSIDE the
                    // graph (a memcpy node; the source is valid during the launch).
                    unsafe {
                        cudarc::driver::result::memcpy_dtod_async(
                            *graphs.logits_out.device_ptr(),
                            logits.dptr(),
                            logits.len() * 4,
                            self.rt.stream.stream,
                        )
                        .map_err(|e| anyhow::anyhow!("graph logits-out memcpy: {e}"))?;
                    }
                    Ok::<_, anyhow::Error>(())
                })();
                // ALWAYS close the capture (even when fwd failed) — an open capture turns
                // every subsequent GSlice drop into a pool-ownership violation.
                let mut graph: cudarc::driver::sys::CUgraph = std::ptr::null_mut();
                let r = unsafe { cudarc::driver::sys::cuStreamEndCapture(self.rt.stream.stream, &mut graph) };
                crate::dsv4_gpu::GRAPH_CAPTURE_ACTIVE.store(false, std::sync::atomic::Ordering::Relaxed);
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(anyhow::anyhow!("cuStreamEndCapture: {r:?}"));
                }
                fwd?;
                let g = graphs.instantiate(graph, start_pos)?;
                g.upload_once(&self.rt.stream)?;
                eprintln!("[dsv4-graph] captured variant {variant:?} at sp={start_pos} (ws high-water {} MB)",
                    crate::dsv4_gpu::graph_ws_high_water() / (1024 * 1024));
                *graphs.slot_mut(variant) = Slot::Ready(g);
                Ok::<_, anyhow::Error>(())
            })();
            if let Err(e) = cap_result {
                *graphs.slot_mut(variant) = Slot::Poisoned;
                return Err(e);
            }
        }
        // ---- replay ----
        let g = match graphs.slot_ref(variant) {
            Slot::Ready(g) => g,
            _ => unreachable!("post-capture slot is Ready"),
        };
        g.replay(&self.dev, &self.rt.stream, &graphs.ids_dev, &graphs.logits_out, id, start_pos)
    }

    /// Trunk forward that ALSO captures main_hidden = h.mean(streams) @ layers 40/41/42
    /// (§B.10 DSpark interface). Returns (last-row logits [vocab], main_hidden [s, 3*dim] bf16).
    /// `main_hidden` is None when the trunk slice doesn't contain all 3 target layers.
    /// Auto-chunks prompts > PREFILL_CHUNK (§12.B.5): main_hidden is captured per chunk and
    /// concatenated → bitwise-identical to one-shot (the chunked-prefill gate). The DSpark warm
    /// then primes the stage rings from the full main_hidden.
    pub fn forward_capture_main(&mut self, ids: &[i32], start_pos: usize) -> Result<(S, Option<B>)> {
        let s = ids.len();
        if start_pos == 0 && s > PREFILL_CHUNK {
            return self.forward_capture_main_chunked(ids, PREFILL_CHUNK);
        }
        self.forward_capture_main_oneshot(ids, start_pos)
    }

    /// One-shot capture (probe-scale; s must fit s_max). The original path.
    fn forward_capture_main_oneshot(&mut self, ids: &[i32], start_pos: usize) -> Result<(S, Option<B>)> {
        let s = ids.len();
        let ids_dev = self.dev.htod_sync_copy(ids)?;
        let mut x = self.embed_tokens(&ids_dev, s)?;
        let targets = &self.cfg.dspark_target_layer_ids;
        let have_all = targets.iter().all(|&t| t < self.layers.len());
        let mut caps: [Option<B>; 3] = [None, None, None];
        for i in 0..self.layers.len() {
            let o = self
                .rt
                .block_forward::<B, S, CudaSlice<i32>, CudaSlice<u8>, CudaSlice<u32>>(&self.layers[i], &mut self.states[i], &mut self.scratch, &x, s, start_pos, &ids_dev, &self.cfg)
                .with_context(|| format!("block_forward layer {i}"))?;
            x = o.y;
            if have_all {
                for (k, &t) in targets.iter().enumerate() {
                    if i == t { caps[k] = Some(x.clone()); }
                }
            }
        }
        let (_collapsed, logits) = self.forward_head(&x, s)?;
        let main_hidden = if have_all {
            Some(self.rt.compute_main_hidden(
                caps[0].as_ref().unwrap(),
                caps[1].as_ref().unwrap(),
                caps[2].as_ref().unwrap(),
                s, &self.cfg,
            )?)
        } else { None };
        Ok((logits, main_hidden))
    }

    /// Chunked prefill + main_hidden capture (§12.B.5): process `ids` in `chunk`-sized 128-aligned
    /// pieces, capturing h.mean(streams) @ 40/41/42 per chunk → concatenated main_hidden
    /// [s, 3*dim]. Bitwise-identical to one-shot (the recurrent state carries across chunks).
    fn forward_capture_main_chunked(&mut self, ids: &[i32], chunk: usize) -> Result<(S, Option<B>)> {
        anyhow::ensure!(chunk % 128 == 0 && chunk > 0, "capture chunk {chunk} not a multiple of 128");
        let s = ids.len();
        let targets = &self.cfg.dspark_target_layer_ids;
        let have_all = targets.iter().all(|&t| t < self.layers.len());
        let three_d = 3 * self.cfg.dim;
        let mut full_mh: Option<B> = if have_all { Some(self.dev.alloc_zeros::<bf16>(s * three_d)?) } else { None };
        let mut tail: Option<(B, usize)> = None;
        let mut c0 = 0usize;
        while c0 < s {
            let cs = chunk.min(s - c0);
            let ids_dev = self.dev.htod_sync_copy(&ids[c0..c0 + cs])?;
            let mut x = self.embed_tokens(&ids_dev, cs)?;
            let mut caps: [Option<B>; 3] = [None, None, None];
            for i in 0..self.layers.len() {
                let o = self
                    .rt
                    .block_forward::<B, S, CudaSlice<i32>, CudaSlice<u8>, CudaSlice<u32>>(&self.layers[i], &mut self.states[i], &mut self.scratch, &x, cs, c0, &ids_dev, &self.cfg)
                    .with_context(|| format!("block_forward layer {i} (chunked capture @{c0})"))?;
                x = o.y;
                if have_all {
                    for (k, &t) in targets.iter().enumerate() {
                        if i == t { caps[k] = Some(x.clone()); }
                    }
                }
            }
            if let Some(full) = full_mh.as_mut() {
                let cmh = self.rt.compute_main_hidden(
                    caps[0].as_ref().unwrap(), caps[1].as_ref().unwrap(), caps[2].as_ref().unwrap(),
                    cs, &self.cfg,
                )?;
                // d2d the chunk's main_hidden [cs, 3*dim] into the full buffer at row c0.
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        (*full.device_ptr()) + (c0 * three_d * 2) as u64,
                        *cmh.device_ptr(), cs * three_d * 2, self.rt.stream.stream,
                    ).map_err(|e| anyhow!("chunked main_hidden dtod: {e}"))?;
                }
            }
            tail = Some((x, cs));
            c0 += cs;
        }
        let (x_tail, tail_len) = tail.expect("non-empty prompt produced no chunks");
        let (_collapsed, logits) = self.forward_head(&x_tail, tail_len)?;
        Ok((logits, full_mh))
    }
    /// Trunk verify forward returning ALL `s` rows of fp32 logits (for DSpark verify: compare
    /// each draft position's argmax). Same as `forward` but the LM head runs over every row
    /// (not last-only). Used at start_pos>0 over [real + drafts] (the sequential per-token
    /// attention path handles s>1 correctly — §6a.1).
    pub fn forward_verify_logits(&mut self, ids: &[i32], start_pos: usize) -> Result<S> {
        let (logits, _mh) = self.forward_verify_capture_main(ids, start_pos)?;
        Ok(logits)
    }

    /// Verify forward + main_hidden capture (§B.10 — h.mean(streams) @ 40/41/42) for ALL `s`
    /// verify rows. The DSpark serve loop uses the committed-prefix rows to re-prime the draft
    /// ring (AGENTS §2.6 "re-prime with real verify hiddens") so the draft attends to a
    /// contiguous main_kv window (no gaps at the accepted-draft positions). Returns
    /// (per-row logits [s, vocab] fp32, main_hidden [s, 3*dim] bf16).
    pub fn forward_verify_capture_main(&mut self, ids: &[i32], start_pos: usize) -> Result<(S, Option<B>)> {
        let s = ids.len();
        let ids_dev = self.dev.htod_sync_copy(ids)?;
        let mut x: B = self.embed_tokens(&ids_dev, s)?;
        // R4 selective rollback: lazily allocate the per-layer verify-activation capture and
        // record each layer's input x (committed-prefix re-application without a full re-forward).
        // Item 3.3: the verify WIDTH is no longer fixed (adaptive depth drafts D rows, verifies
        // D+1) — grow the capture when a wider verify arrives (contents are fully re-captured
        // per forward, so a grow is a fresh zeroed alloc, never a copy).
        let n_hc_dim = self.cfg.hc_mult * self.cfg.dim;
        let need = s * n_hc_dim;
        if self.x_cap.as_ref().map_or(true, |c| c[0].len() < need) {
            let caps: Result<Vec<_>, _> = (0..self.layers.len())
                .map(|_| self.dev.alloc_zeros::<bf16>(need).map_err(|e| anyhow!("x_cap alloc: {e}")))
                .collect();
            self.x_cap = Some(caps?);
        }
        let targets = &self.cfg.dspark_target_layer_ids;
        let have_all = targets.iter().all(|&t| t < self.layers.len());
        let mut caps: [Option<B>; 3] = [None, None, None];
        for i in 0..self.layers.len() {
            {
                // capture this layer's input (the committed rows are what a full re-forward
                // would have seen — width-bitwise: x@p at s=6 == s=1).
                let xc = &mut self.x_cap.as_mut().unwrap()[i];
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        *xc.device_ptr(), *x.device_ptr(), x.len() * 2, self.rt.stream.stream,
                    )
                    .map_err(|e| anyhow!("x_cap d2d layer {i}: {e}"))?;
                }
            }
            let o = self
                .rt
                .block_forward::<B, S, CudaSlice<i32>, CudaSlice<u8>, CudaSlice<u32>>(&self.layers[i], &mut self.states[i], &mut self.scratch, &x, s, start_pos, &ids_dev, &self.cfg)
                .with_context(|| format!("block_forward layer {i} (verify+capture)"))?;
            x = o.y;
            if have_all {
                for (k, &t) in targets.iter().enumerate() {
                    if i == t { caps[k] = Some(x.clone()); }
                }
            }
        }
        let (dim, vocab) = (self.cfg.dim, self.cfg.vocab_size);
        let collapsed = self.rt.hc_head(&x, &self.hc_head_fn, &self.hc_head_base, &self.hc_head_scale, s, &self.cfg)?;
        let yn = self.rt.rmsnorm(&collapsed, &self.norm, s, dim, self.cfg.norm_eps)?;
        let logits = self.rt.lm_head(&self.head, &yn, s, dim, vocab)?;
        let main_hidden = if have_all {
            Some(self.rt.compute_main_hidden(
                caps[0].as_ref().unwrap(), caps[1].as_ref().unwrap(), caps[2].as_ref().unwrap(),
                s, &self.cfg,
            )?)
        } else { None };
        Ok((logits, main_hidden))
    }

    /// DSpark selective rollback (R4): after a rejected verify, re-apply the committed prefix
    /// `start_pos .. start_pos+n-1` (verify rows 0..n-1) WITHOUT a full re-forward. The verify
    /// captured every layer's input x (width-bitwise == a re-forward's), so:
    ///   - the KV ring is rebuilt by `dspark_attn_warm_range` (batched, the gated warm path);
    ///   - compressor + indexer state is re-applied by `forward_tokens` (batched GEMM + the
    ///     same sequential state chain as sequential decode — gates prove the equivalence).
    /// The attention gather output, MoE and LM head are NOT recomputed — nothing downstream
    /// consumes them (the next step's carry forward recomputes from state + the new token).
    /// Final state is identical to the old per-token full re-forward.
    pub fn readvance_committed(&mut self, start_pos: usize, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let dim = self.cfg.dim;
        let eps = self.cfg.norm_eps;
        let ks = self.rt.comp.as_ref().expect("readvance_committed needs comp kernels");
        let x_cap = self.x_cap.as_ref().expect("readvance_committed before any verify capture");
        for i in 0..self.layers.len() {
            let kind = self.layers[i].kind;
            let rope = self.rt.rope_for(kind);
            let layer = &self.layers[i];
            let st = &mut self.states[i];
            // the compressor/KV path consumes yn = rmsnorm(hc_pre(x)) (the attention input
            // [n, dim]), not the raw [n, hc*dim] block input — recompute it from the capture.
            let xc = &x_cap[i];
            let (y, _posts, _combs) = self
                .rt
                .hc_pre::<B, S>(xc, n, &layer.hc_attn_fn, &layer.hc_attn_base, &layer.hc_attn_scale, &self.cfg)
                .with_context(|| format!("readvance hc_pre layer {i}"))?;
            let yn = self
                .rt
                .rmsnorm(&y, &layer.attn_norm, n, dim, eps)
                .with_context(|| format!("readvance norm layer {i}"))?;
            self.rt
                .dspark_attn_warm_range(layer, st, &yn, n, start_pos, rope, &self.cfg)
                .with_context(|| format!("readvance warm layer {i}"))?;
            if let Some(comp) = &st.attn_compressor {
                comp.forward_tokens::<B, S, CudaSlice<i32>, CudaSlice<u32>>(&self.dev, ks, &self.rt.stream, &yn, n, start_pos, rope)
                    .with_context(|| format!("readvance compressor layer {i}"))?;
            }
            if let Some(idx) = &st.indexer {
                idx.comp
                    .forward_tokens::<B, S, CudaSlice<i32>, CudaSlice<u32>>(&self.dev, ks, &self.rt.stream, &yn, n, start_pos, rope)
                    .with_context(|| format!("readvance indexer layer {i}"))?;
            }
        }
        Ok(())
    }

    /// Snapshot the full per-layer attention state (KV ring + compressor + indexer) for DSpark
    /// verify rollback. Call BEFORE `forward_verify_logits`; [`restore_verify_state`] rewinds.
    pub fn snapshot_verify_state(&self) -> Result<Vec<LayerVerifySnap>> {
        let mut snaps = Vec::with_capacity(self.states.len());
        for st in &self.states {
            snaps.push(st.snapshot_verify(&self.dev, &self.rt.stream)?);
        }
        Ok(snaps)
    }

    /// Rewind every layer's attention state to its snapshot (D2D on the compute stream).
    pub fn restore_verify_state(&self, snaps: &[LayerVerifySnap]) -> Result<()> {
        anyhow::ensure!(snaps.len() == self.states.len(), "restore: {} snaps vs {} layers", snaps.len(), self.states.len());
        for (st, snap) in self.states.iter().zip(snaps) {
            st.restore_verify(snap, &self.dev, &self.rt.stream)?;
        }
        Ok(())
    }

    /// Reset ALL per-layer attention state to the prefill-start condition (zeroed KV ring,
    /// compressor-cache tail, and frontier state re-initialized). Used by the persistent
    /// server loop between requests so each request starts from a clean model state.
    pub fn reset_states(&mut self) -> Result<()> {
        let dev = &self.dev;
        let stream = &self.rt.stream;
        let comp_ks = self.rt.comp.as_ref().expect("CSA/HCA needs comp kernels");
        for st in &mut self.states {
            // Zero the unified kv_cache (ring + compressor-cache tail).
            let n = st.kv_cache.len();
            let zeros = vec![bf16::ZERO; n];
            dev.htod_sync_copy_into(&zeros, &mut st.kv_cache)
                .map_err(|e| anyhow!("reset kv_cache: {e}"))?;
            // Reset compressor frontier (kv=0, score=−inf via the state-init kernel).
            if let Some(comp) = &st.attn_compressor {
                comp.reset(comp_ks, stream)?;
            }
            if let Some(idx) = &st.indexer {
                idx.comp.reset(comp_ks, stream)?;
            }
        }
        dev.synchronize().map_err(|e| anyhow!("reset sync: {e}"))?;
        Ok(())
    }

}
