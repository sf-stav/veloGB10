//! `impl GpuModel`: decode path: penalties, verify masks, decode buffers, `forward_decode*` (logits / select / sample),
//! the device-resident loop and decode-graph helpers. Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// Decode-row penalties (vLLM semantics) over `batch` lanes: row k uses slot `slot_ids_dev[k]`'s
    /// committed output + prompt; no extra tokens. The per-row VALUES (rep_pen/presence/frequency_dev)
    /// are neutral for a lane without a penalty, and the kernel returns at once for a neutral row.
    pub fn pen_apply_decode(&self, bufs: &DecodeBuffers, logits: u64, f32_logits: bool, batch: usize) {
        let v = self.cfg.vocab_size;
        let name = if f32_logits { "pen_apply_f32_b" } else { "pen_apply_b" };
        blaunch!(self, name, (batch as u32,1,1), (256,1,1), 0,
            (logits, *bufs.slot_ids_dev.device_ptr() as u64, 0u64, 0i32,
             *bufs.pen_cnt_dev.device_ptr() as u64, *bufs.pen_pbits_dev.device_ptr() as u64,
             *bufs.rep_pen_dev.device_ptr() as u64, *bufs.presence_dev.device_ptr() as u64,
             *bufs.frequency_dev.device_ptr() as u64, v as i32, pen_words_for(v) as i32, batch as i32));
    }

    /// Verify-column penalties (vLLM semantics): column c uses `p.slots[c]`'s committed output +
    /// prompt plus its `PEN_EXTRA` extra tokens (the drafts before it). Same per-token formula as
    /// the decode rows, so a verify column and a plain decode step at the same point penalize
    /// identically (spec on/off byte identity for greedy penalized requests).
    pub fn pen_apply_verify(&self, p: &VerifyPenalty, logits: u64, f32_logits: bool, n: usize) {
        let v = self.cfg.vocab_size;
        let name = if f32_logits { "pen_apply_f32_b" } else { "pen_apply_b" };
        blaunch!(self, name, (n as u32,1,1), (256,1,1), 0,
            (logits, p.slots_ptr, p.extra_ptr, PEN_EXTRA as i32, p.cnt_ptr, p.pbits_ptr,
             p.rep_pen_ptr, p.presence_ptr, p.freq_ptr, v as i32, pen_words_for(v) as i32, n as i32));
    }

    /// A lane takes over `slot`: zero its output counts; mark its prompt tokens when the lane's
    /// repetition penalty is active (`prompt` empty = leave the bitset clear).
    pub fn pen_slot_init(&self, bufs: &mut DecodeBuffers, slot: usize, prompt: &[u32]) {
        assert!(slot < bufs.pen_slots, "pen_slot_init: slot {slot} >= {}", bufs.pen_slots);
        let v = self.cfg.vocab_size;
        let w = pen_words_for(v);
        let blocks = ((v.max(w) + 255) / 256).min(1024) as u32;
        blaunch!(self, "pen_slot_reset_b", (blocks,1,1), (256,1,1), 0,
            (*bufs.pen_cnt_dev.device_ptr() as u64, *bufs.pen_pbits_dev.device_ptr() as u64,
             slot as i32, v as i32, w as i32));
        if !prompt.is_empty() {
            let toks: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
            let dev_toks = self.dev.htod_sync_copy(&toks).unwrap();
            blaunch!(self, "pen_prompt_set_b", (((toks.len() + 255) / 256) as u32,1,1), (256,1,1), 0,
                (*bufs.pen_pbits_dev.device_ptr() as u64, slot as i32, *dev_toks.device_ptr() as u64,
                 toks.len() as i32, v as i32, w as i32));
            // dev_toks is read by the kernel above: keep it alive until the stream passes it.
            self.sync_stream();
            drop(dev_toks);
        }
    }

    /// Count newly committed output tokens: `(slot, token)` pairs into pen_cnt_dev.
    pub fn pen_count_add(&self, bufs: &mut DecodeBuffers, pairs: &[(i32, i32)]) {
        let v = self.cfg.vocab_size;
        for chunk in pairs.chunks(PEN_ADD_CAP) {
            let slots: Vec<i32> = chunk.iter().map(|p| p.0).collect();
            let toks: Vec<i32> = chunk.iter().map(|p| p.1).collect();
            // Host-blocking NULL-stream copies order before the kernel on the blocking compute
            // stream (invariant I1); the staging buffers are reused only after this launch.
            self.dev.htod_sync_copy_into(&slots, &mut bufs.pen_add_slots_dev.slice_mut(0..chunk.len())).unwrap();
            self.dev.htod_sync_copy_into(&toks, &mut bufs.pen_add_toks_dev.slice_mut(0..chunk.len())).unwrap();
            blaunch!(self, "pen_count_add_b", (((chunk.len() + 255) / 256) as u32,1,1), (256,1,1), 0,
                (*bufs.pen_cnt_dev.device_ptr() as u64, *bufs.pen_add_slots_dev.device_ptr() as u64,
                 *bufs.pen_add_toks_dev.device_ptr() as u64, chunk.len() as i32, v as i32));
            if chunk.len() == PEN_ADD_CAP { self.sync_stream(); } // the next chunk reuses the staging
        }
    }

    pub fn new_decode_buffers(&self, max_batch: usize) -> DecodeBuffers {
        let rdim = self.cfg.rotary_dim;
        DecodeBuffers {
            pos_dev: self.dev.htod_sync_copy(&vec![0i32; max_batch]).unwrap(),
            tokens_dev: self.dev.htod_sync_copy(&vec![0i32; max_batch]).unwrap(),
            slot_ids_dev: self.dev.htod_sync_copy(&(0..max_batch as i32).collect::<Vec<_>>()).unwrap(),
            cos_dev: self.dev.alloc_zeros::<f32>(max_batch * rdim).unwrap(),
            sin_dev: self.dev.alloc_zeros::<f32>(max_batch * rdim).unwrap(),
            token_ids_dev: self.dev.alloc_zeros::<i32>(max_batch).unwrap(),
            // LOOPFIX: whole-output penalty state, htod zero-initialized (invariant 2).
            pen_cnt_dev: self.dev.htod_sync_copy(&vec![0u32; max_batch * self.cfg.vocab_size]).unwrap(),
            pen_pbits_dev: self.dev.htod_sync_copy(&vec![0u32; max_batch * pen_words_for(self.cfg.vocab_size)]).unwrap(),
            pen_add_slots_dev: self.dev.htod_sync_copy(&vec![-1i32; PEN_ADD_CAP]).unwrap(),
            pen_add_toks_dev: self.dev.htod_sync_copy(&vec![-1i32; PEN_ADD_CAP]).unwrap(),
            pen_slots: max_batch,
            temps_dev: self.dev.htod_sync_copy(&vec![1.0f32; max_batch]).unwrap(),
            topk_dev: self.dev.htod_sync_copy(&vec![1i32; max_batch]).unwrap(),
            topp_dev: self.dev.htod_sync_copy(&vec![1.0f32; max_batch]).unwrap(),
            seeds_dev: self.dev.htod_sync_copy(&vec![0u32; max_batch]).unwrap(),
            rep_pen_dev: self.dev.htod_sync_copy(&vec![1.0f32; max_batch]).unwrap(),
            // Resident-loop keys: htod-initialized (0). alloc_zeros does NOT zero (AGENTS.md
            // invariant 2) and the graph head reads these before any dirty upload on the first step.
            seeds_key_dev: self.dev.htod_sync_copy(&vec![0u64; max_batch]).unwrap(),
            presence_dev: self.dev.htod_sync_copy(&vec![0.0f32; max_batch]).unwrap(),
            frequency_dev: self.dev.htod_sync_copy(&vec![0.0f32; max_batch]).unwrap(),
            // W2: schema-mask buffers. The words are initialized to ZERO, which would mean
            // "nothing allowed" if a flag were set — but the flags start at 0 (unconstrained) and
            // `mask_any` starts false, so the kernel is never launched from this state.
            mask_words_dev: self.dev.htod_sync_copy(&vec![0u32; mask_words_for(self.cfg.vocab_size) * max_batch]).unwrap(),
            mask_flags_dev: self.dev.htod_sync_copy(&vec![0i32; max_batch]).unwrap(),
            mask_any: false,
        }
    }

    /// W2: upload the per-lane DECODE masks in `batch_idx` order and arm/disarm the kernel.
    /// `flags[k] == 0` marks an unconstrained lane (its words are ignored by the kernel).
    pub fn upload_decode_masks(&self, bufs: &mut DecodeBuffers,
                               words: &[u32], flags: &[i32], active: bool) {
        if !active { bufs.mask_any = false; return; }
        self.dev.htod_sync_copy_into(words, &mut bufs.mask_words_dev).unwrap();
        self.dev.htod_sync_copy_into(flags, &mut bufs.mask_flags_dev).unwrap();
        bufs.mask_any = true;
        // W2 diagnostic (one-shot): how restrictive is the first armed decode mask?
        if !MASK_DIAG_ON.load(std::sync::atomic::Ordering::Relaxed) {
            MASK_DIAG_ON.store(true, std::sync::atomic::Ordering::Relaxed);
            let allowed: u32 = words.iter().map(|w| w.count_ones()).sum();
            eprintln!("[schema] decode mask armed: lanes={} words={} allowed_tokens={} flags={:?}",
                      flags.len(), words.len(), allowed, &flags[..flags.len().min(4)]);
        }
    }

    /// W2: arm the per-POSITION verify mask (one slot per verify column, `MAX_VERIFY` max) and
    /// apply it to whatever verify runs next; `clear_verify_mask` disarms. `masks[j] = None`
    /// leaves position j unconstrained. Returns false when the scratch could not be allocated.
    pub fn set_verify_mask(&self, masks: &[Option<std::sync::Arc<Vec<u32>>>]) -> bool {
        use std::sync::atomic::Ordering;
        let words = mask_words_for(self.cfg.vocab_size);
        let n = masks.len().min(MAX_VERIFY);
        let mut w = vec![0u32; words * MAX_VERIFY];
        let mut f = vec![0i32; MAX_VERIFY];
        for j in 0..n {
            if let Some(m) = &masks[j] {
                let src: &Vec<u32> = m;
                let len = src.len().min(words);
                w[j * words..j * words + len].copy_from_slice(&src[..len]);
                f[j] = 1;
            }
        }
        unsafe {
            let words_dst = *self.vmask_words.device_ptr() as cudarc::driver::sys::CUdeviceptr;
            cudarc::driver::result::memcpy_htod_async(words_dst, &w, self.stream.stream)
                .expect("htod vmask words");
            let flags_dst = *self.vmask_flags.device_ptr() as cudarc::driver::sys::CUdeviceptr;
            cudarc::driver::result::memcpy_htod_async(flags_dst, &f, self.stream.stream)
                .expect("htod vmask flags");
        }
        self.vmask_words_ptr.store(*self.vmask_words.device_ptr(), Ordering::Relaxed);
        self.vmask_flags_ptr.store(*self.vmask_flags.device_ptr(), Ordering::Relaxed);
        if !VMASK_DIAG_ON.load(Ordering::Relaxed) {
            VMASK_DIAG_ON.store(true, Ordering::Relaxed);
            let allowed: u32 = w.iter().map(|x| x.count_ones()).sum();
            eprintln!("[schema] verify mask armed: positions={} words={} allowed_tokens_pos0={} flags={:?}",
                      n, words, w[..words].iter().map(|x| x.count_ones()).sum::<u32>(), &f[..n.min(4)]);
        }
        true
    }

    /// W2: disarm the verify mask (the kernel launch is skipped when the words pointer is 0).
    pub fn clear_verify_mask(&self) {
        self.vmask_words_ptr.store(0, std::sync::atomic::Ordering::Relaxed);
        self.vmask_flags_ptr.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// W2: the armed verify-mask device pointers, or None when no mask is set.
    #[inline]
    pub(super) fn verify_mask_ptrs(&self) -> Option<(u64, u64)> {
        let w = self.vmask_words_ptr.load(std::sync::atomic::Ordering::Relaxed);
        let f = self.vmask_flags_ptr.load(std::sync::atomic::Ordering::Relaxed);
        if w == 0 { None } else { Some((w, f)) }
    }

    /// W2: launch the vocabulary mask over a [vocab, B] logits column block (bf16).
    #[inline]
    fn launch_mask_bf16(&self, logits_ptr: u64, vocab: usize, b: usize) {
        if let Some((mw, mf)) = self.verify_mask_ptrs() {
            let nw = mask_words_for(vocab);
            let gy = ((nw as u32) + 255) / 256;
            blaunch!(self, "logit_mask_b", (b as u32, gy, 1), (256,1,1), 0,
                (logits_ptr, mw, mf, vocab as i32, b as i32));
        }
    }

    /// W2: same for the fp32 (hy_v3) arm.
    #[inline]
    fn launch_mask_f32(&self, logits_ptr: u64, vocab: usize, b: usize) {
        if let Some((mw, mf)) = self.verify_mask_ptrs() {
            let nw = mask_words_for(vocab);
            let gy = ((nw as u32) + 255) / 256;
            blaunch!(self, "logit_mask_f32_b", (b as u32, gy, 1), (256,1,1), 0,
                (logits_ptr, mw, mf, vocab as i32, b as i32));
        }
    }

    /// One fully device-side decode step using pre-computed RoPE tables and persistent buffers.
    /// Caller must write `tokens` and `pos` to bufs before calling.
    /// GPU-only decode step — writes to bufs.token_ids_dev. Fully graph-capturable (no host sync).
    pub fn forward_decode_gpu(&self, pool: &mut Pool, bufs: &mut DecodeBuffers,
                          state: &mut BatchGpuState, kv_stride: usize, max_pc: usize, batch: usize) {
        let logits = self.forward_decode_logits(pool, bufs, state, kv_stride, max_pc, batch);
        self.forward_decode_select(pool, bufs, logits, batch, false);
    }

    /// Device-resident token loop (EXPERT_DEVICE_ARGMAX_LOOP_RESPONSE): is it enabled?
    /// CLI `--device-loop=on|off` rides TpConfig to the node (which installs [device-loop]
    /// from the shipped config before load); the env var is the node-side + bench override.
    /// Default OFF until the token-identity A/B + full gate chain pass; then flip to ON.
    pub(crate) fn device_loop_on() -> bool {
        static D: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            if let Ok(v) = crate::opts::var(crate::opt!("device-loop")) {
                return matches!(v.as_str(), "1" | "on" | "true");
            }
            crate::tp::tp_config().map(|c| c.device_loop).unwrap_or(false)
        });
        *D
    }

    /// The serving decode sequence WITH the device-resident token loop: the four bookkeeping
    /// kernels around the two forward halves. A clean step runs entirely on the GPU:
    ///   ids_advance -> logits -> penalty_window -> select -> ring_push (+ seed_advance for
    ///   sampling), then the host reads back [batch] i32 only.
    /// The new kernels are pool-free persistent-buffer launches (capture-safe); the selection
    /// and penalty-application kernels are byte-identical, so the token stream is identical to
    /// the host-bookkeeping path — the dirty-upload logic in batched_decode keeps the ring/keys
    /// current across admissions/finishes/MTP steps/param changes.
    pub fn decode_step_resident(&self, pool: &mut Pool, bufs: &mut DecodeBuffers,
                                state: &mut BatchGpuState, kv_stride: usize, max_pc: usize,
                                batch: usize, sampling: bool) {
        // 1. graph head: last step's selected id becomes this step's embed input; pos advances.
        blaunch!(self, "ids_advance_b", (1,1,1), (batch as u32,1,1), 0,
            (*bufs.tokens_dev.device_ptr() as u64, *bufs.token_ids_dev.device_ptr() as u64,
             *bufs.pos_dev.device_ptr() as u64, batch as i32));
        // 2. embed .. logits (TP all-reduces inside; unchanged).
        let logits = self.forward_decode_logits(pool, bufs, state, kv_stride, max_pc, batch);
        // 3. penalty application + selection. LOOPFIX: the whole-output penalty state
        //    (pen_cnt_dev) is brought up to date by the host before the step (`pen_count_add`), so
        //    the resident core no longer rebuilds a window ring nor pushes into one.
        self.forward_decode_select(pool, bufs, logits, batch, sampling);
        // 4. epilogue: advance the sample seed keys once.
        if sampling {
            blaunch!(self, "seed_advance_b", (1,1,1), (batch as u32,1,1), 0,
                (*bufs.seeds_dev.device_ptr() as u64, *bufs.seeds_key_dev.device_ptr() as u64,
                 batch as i32));
        }
    }

    /// Graph-capturable half 1: embed -> rope -> forward_batch_dev (64 layers, TP all-reduces
    /// inside) -> LM-head logits. Returns the logits buffer UNRELEASED — the select half owns it.
    pub fn forward_decode_logits(&self, pool: &mut Pool, bufs: &mut DecodeBuffers,
                          state: &mut BatchGpuState, kv_stride: usize, max_pc: usize, batch: usize) -> DecodeLogits {
        if f9_xhash_active() { f9_x_new_call(); F9X_ATTN.with(|c| c.set(usize::MAX)); }
        // Caller must sync NULL stream before calling (after htod copies into bufs, incl. per-lane
        // rep/presence/frequency in bufs). Fully graph-capturable (no host sync).
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let rdim = cfg.rotary_dim;
        let v = cfg.vocab_size;

        let hidden = pool.get_bf16(cfg.resid_width() * batch);
        self.embed_gather_resid(pool, *hidden.device_ptr() as u64,
                          *bufs.tokens_dev.device_ptr() as u64, batch);
        blaunch!(self, "gather_rope_b", grid(batch*rdim), (256,1,1), 0,
            (d(&bufs.cos_dev), d(&bufs.sin_dev),
             d(&self.cos_table), d(&self.sin_table),
             *bufs.pos_dev.device_ptr() as u64, rdim as i32, batch as i32));
        let out = self.forward_batch_dev(pool, hidden, &bufs.pos_dev, &bufs.cos_dev, &bufs.sin_dev,
                                         max_pc, state, *bufs.slot_ids_dev.device_ptr(), kv_stride, batch,
                                         *bufs.tokens_dev.device_ptr() as u64);
        if f9_xhash_active() && batch <= 8 {
            let host = self.dev.dtoh_sync_copy(&out).unwrap_or_default();
            let mut hs = Vec::with_capacity(batch);
            for c in 0..batch {
                let mut hv: u64 = 0xcbf29ce484222325;
                for i in 0..h { let b = host.get(c * h + i).map(|v| v.to_bits()).unwrap_or(0); hv ^= b as u64; hv = hv.wrapping_mul(0x100000001b3); }
                hs.push(hv);
            }
            F9XHASH.lock().unwrap().push((0u8, F9X_CUR.with(|c| c.get()), usize::MAX, hs));
        }
        if cfg.lm_head_fp32 {
            // hy_v3: fp32 logits end-to-end (gemm_binv_f32_b + the f32 penalty/argmax twins).
            let logits = self.logits_batch_f32(pool, &out, batch);
            pool.release_bf16(out, h * batch);
            return DecodeLogits::F32(logits);
        }
        let logits = self.logits_batch(pool, &out, batch);
        // Cross-chain probe capture: the RAW (pre-penalty) logits.
        if xchain_active() { xchain_capture(|c| c.logits = {
            // capture-safe (R9): during graph capture the dtoh is illegal — record empty, don't panic
            match self.dev.dtoh_sync_copy(&logits) {
                Ok(mut lh) => { lh.truncate(v * batch); lh.iter().map(|b| half::bf16::to_f32(*b)).collect() }
                Err(_) => Vec::new(),
            }
        }); }
        pool.release_bf16(out, h * batch);
        DecodeLogits::Bf16(logits)
    }

    /// Graph-capturable half 2: rep_penalty + selection (argmax greedy | sample_b stochastic)
    /// over the logits from `forward_decode_logits`. Consumes (releases) the logits buffer.
    /// `sampling` selects sample_*_b (with the greedy arm inside) vs argmax_*_b — the serving
    /// route is unchanged: mixed batches ride the sample path exactly as before.
    pub fn forward_decode_select(&self, pool: &mut Pool, bufs: &mut DecodeBuffers,
                                 logits: DecodeLogits, batch: usize, sampling: bool) {
        let cfg = &self.cfg;
        let v = cfg.vocab_size;
        let nthr = 256u32;
        let smem = ((2 * 64usize + 2 * nthr as usize) * 4) as u32;
        match logits {
            DecodeLogits::F32(logits) => {
                self.pen_apply_decode(bufs, d(&logits), true, batch);
                // W2: JSON-schema mask — applied to the logits column AFTER the penalties and
                // BEFORE selection, so every downstream reduction sees the same fixed order it
                // always did (allowed entries stay byte-identical).
                if bufs.mask_any {
                    let gy = ((mask_words_for(v) as u32) + 255) / 256;
                    blaunch!(self, "logit_mask_f32_b", (batch as u32, gy, 1), (256,1,1), 0,
                        (d(&logits), *bufs.mask_words_dev.device_ptr() as u64,
                         *bufs.mask_flags_dev.device_ptr() as u64, v as i32, batch as i32));
                }
                if sampling {
                    blaunch!(self, "sample_f32_b", (batch as u32,1,1), (nthr,1,1), smem,
                        (*bufs.token_ids_dev.device_ptr() as u64, d(&logits),
                         *bufs.temps_dev.device_ptr() as u64, *bufs.topk_dev.device_ptr() as u64,
                         *bufs.topp_dev.device_ptr() as u64, *bufs.seeds_dev.device_ptr() as u64,
                         v as i32, batch as i32));
                } else {
                    let block = 1024u32;
                    blaunch!(self, "argmax_f32_b", (batch as u32, 1, 1), (block, 1, 1), (block as u32 * 8),
                        (*bufs.token_ids_dev.device_ptr() as u64, d(&logits), v as i32, batch as i32));
                }
                pool.release(logits, v * batch);
            }
            DecodeLogits::Bf16(logits) => {
                self.pen_apply_decode(bufs, d(&logits), false, batch);
                // W2: JSON-schema mask — applied to the logits column AFTER the penalties and
                // BEFORE selection, so every downstream reduction sees the same fixed order it
                // always did (allowed entries stay byte-identical).
                if bufs.mask_any {
                    let gy = ((mask_words_for(v) as u32) + 255) / 256;
                    blaunch!(self, "logit_mask_b", (batch as u32, gy, 1), (256,1,1), 0,
                        (d(&logits), *bufs.mask_words_dev.device_ptr() as u64,
                         *bufs.mask_flags_dev.device_ptr() as u64, v as i32, batch as i32));
                }
                if sampling {
                    blaunch!(self, "sample_b", (batch as u32,1,1), (nthr,1,1), smem,
                        (*bufs.token_ids_dev.device_ptr() as u64, d(&logits),
                         *bufs.temps_dev.device_ptr() as u64, *bufs.topk_dev.device_ptr() as u64,
                         *bufs.topp_dev.device_ptr() as u64, *bufs.seeds_dev.device_ptr() as u64,
                         v as i32, batch as i32));
                } else {
                    let block = 1024u32;
                    blaunch!(self, "argmax_b", (batch as u32, 1, 1), (block, 1, 1), (block as u32 * 8),
                        (*bufs.token_ids_dev.device_ptr() as u64, d(&logits), v as i32, batch as i32));
                }
                pool.release_bf16(logits, v * batch);
            }
        }
    }

    /// Full decode step: GPU forward + readback. Non-graph path.
    pub fn forward_decode(&self, pool: &mut Pool, bufs: &mut DecodeBuffers,
                          state: &mut BatchGpuState, kv_stride: usize, max_pc: usize, batch: usize) -> Vec<u32> {
        if Self::device_loop_on() {
            self.decode_step_resident(pool, bufs, state, kv_stride, max_pc, batch, false);
        } else {
            self.forward_decode_gpu(pool, bufs, state, kv_stride, max_pc, batch);
        }
        self.sync_stream();
        self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap()
            .into_iter().take(batch).map(|x| x as u32).collect()
    }

    /// Decode with CPU sampling (for non-greedy requests). Reads logits to host,
    /// applies temperature + top_k + top_p, samples per-lane.
    pub fn forward_decode_sample(&self, pool: &mut Pool, bufs: &mut DecodeBuffers,
                                 state: &mut BatchGpuState, kv_stride: usize, max_pc: usize, batch: usize,
                                 temps: &[f32], top_ks: &[usize], top_ps: &[f32]) -> Vec<u32> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let rdim = cfg.rotary_dim;
        let v = cfg.vocab_size;

        // Forward pass (same as forward_decode_gpu but without argmax)
        let hidden = pool.get_bf16(cfg.resid_width() * batch);
        self.embed_gather_resid(pool, *hidden.device_ptr() as u64,
                          *bufs.tokens_dev.device_ptr() as u64, batch);
        blaunch!(self, "gather_rope_b", grid(batch*rdim), (256,1,1), 0,
            (d(&bufs.cos_dev), d(&bufs.sin_dev),
             d(&self.cos_table), d(&self.sin_table),
             *bufs.pos_dev.device_ptr() as u64, rdim as i32, batch as i32));
        let out = self.forward_batch_dev(pool, hidden, &bufs.pos_dev, &bufs.cos_dev, &bufs.sin_dev,
                                         max_pc, state, *bufs.slot_ids_dev.device_ptr(), kv_stride, batch,
                                         *bufs.tokens_dev.device_ptr() as u64);
        if cfg.lm_head_fp32 {
            // hy_v3: fp32 logits, penalized on-GPU, read to host UNROUNDED (the f32 twin kernels).
            let logits = self.logits_batch_f32(pool, &out, batch);
            pool.release_bf16(out, h * batch);
            self.pen_apply_decode(bufs, d(&logits), true, batch);
            self.sync_stream();
            let lh: Vec<f32> = self.dev.dtoh_sync_copy(&logits).unwrap();
            pool.release(logits, v * batch);
            return (0..batch).map(|i| {
                crate::sampler::sample(&lh[i*v..(i+1)*v], temps[i], top_ks[i], top_ps[i])
            }).collect();
        }
        let logits = self.logits_batch(pool, &out, batch);
        pool.release_bf16(out, h * batch);
        self.pen_apply_decode(bufs, d(&logits), false, batch);

        // Read logits to host and sample per-lane
        self.sync_stream();
        let lh: Vec<half::bf16> = self.dev.dtoh_sync_copy(&logits).unwrap();
        pool.release_bf16(logits, v * batch);

        (0..batch).map(|i| {
            let col: Vec<f32> = lh[i*v..(i+1)*v].iter().map(|x| half::bf16::to_f32(*x)).collect();
            crate::sampler::sample(&col, temps[i], top_ks[i], top_ps[i])
        }).collect()
    }

    /// Decode + on-GPU sampling. Assumes the caller has already htod'd tokens/pos/penalty
    /// (rep_pen/presence/frequency) AND the sampling params (temps/topk/topp/seeds) into `bufs`
    /// and synced the NULL stream. Non-graph path: runs the core, syncs, reads back token ids.
    pub fn forward_decode_sample_gpu(&self, pool: &mut Pool, bufs: &mut DecodeBuffers,
                                  state: &mut BatchGpuState, kv_stride: usize, max_pc: usize, batch: usize) -> Vec<u32> {
        if Self::device_loop_on() {
            self.decode_step_resident(pool, bufs, state, kv_stride, max_pc, batch, true);
        } else {
            self.decode_sample_gpu_core(pool, bufs, state, kv_stride, max_pc, batch);
        }
        self.sync_stream();
        let ids: Vec<i32> = self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap();
        ids.into_iter().take(batch).map(|x| x as u32).collect()
    }

    /// Pure-GPU decode + sample kernel sequence (embed → rope → forward → logits → rep_penalty →
    /// sample_b; the f32 twins rep_penalty_f32_b/sample_f32_b when `cfg.lm_head_fp32`, hy_v3).
    /// Reads every input from persistent `bufs` and writes `bufs.token_ids_dev`. No host
    /// sync, so it is CUDA-graph capturable. Caller must htod all inputs first.
    pub(super) fn decode_sample_gpu_core(&self, pool: &mut Pool, bufs: &mut DecodeBuffers,
                              state: &mut BatchGpuState, kv_stride: usize, max_pc: usize, batch: usize) {
        // Shared halves with the greedy path: identical kernel sequence to the old inline body
        // (the sample_*_b block size 256 and smem are unchanged — block size is part of the
        // sample/argmax tie-break, do not touch).
        let logits = self.forward_decode_logits(pool, bufs, state, kv_stride, max_pc, batch);
        self.forward_decode_select(pool, bufs, logits, batch, true);
    }
}
