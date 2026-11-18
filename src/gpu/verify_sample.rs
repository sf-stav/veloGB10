//! `impl GpuModel`: sampled verify (`verify_forward_sample*`), window NLL / argmax helpers and `probe_moe`.
//! Greedy MTP stays bitwise lossless; stochastic paths do not touch `mtp_lane_step` (AGENTS.md section 2.7). Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// Stochastic MTP verify forward: like verify_forward but instead of argmax on every column,
    /// computes per-position target probabilities + residual resamples + bonus token for the
    /// speculative rejection-sampling accept loop. Returns (VerifySample, vout [h, n]).
    pub fn verify_forward_sample(&self, pool: &mut Pool, tokens: &[u32],
                                  state: &mut BatchGpuState, slot: usize, kv_stride: usize,
                                  pos_start: usize, ckpt_slot: Option<usize>,
                                  penalty: Option<VerifyPenalty>,
                                  draft_tokens: &[u32], draft_qprobs: &[f32],
                                  temp: f32, top_k: usize, top_p: f32,
                                  seeds: &[u32],
                                  t20_out: Option<&mut Vec<u64>>) -> (VerifySample, B) {
        let cfg = &self.cfg;
        let _h = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let n = tokens.len();
        assert_eq!(draft_tokens.len(), n - 1, "draft_tokens must be length depth-1");
        assert_eq!(draft_qprobs.len(), n - 1, "draft_qprobs must be length depth-1");
        assert_eq!(seeds.len(), n, "seeds must be length depth");

        // hy_v3 (E21): the verify columns ARE the target distribution the rejection-sampling accept
        // loop judges against — keep them UNROUNDED fp32 (enable_lm_head_fp32) and run the f32 kernel
        // twins (rep_penalty_f32_b, spec_verify_f32_b). A bf16 round perturbs near-tie p(x) values
        // and the p/q accept ratio — a measurable distribution bias at temp>0. qwen keeps bf16.
        let (logits, residual) = self.verify_forward_core(pool, tokens, state, slot, kv_stride, pos_start, ckpt_slot, cfg.lm_head_fp32);

        // Apply penalty (same as verify_forward).
        if let Some(p) = &penalty {
            match &logits {
                VerifyLogits::F32(l) => self.pen_apply_verify(p, d(l), true, n),
                VerifyLogits::B16(l) => self.pen_apply_verify(p, d(l), false, n),
            }
        }

        // Persistent scratch, packed and uploaded on the COMPUTE stream. Everything here used to be
        // a fresh alloc_zeros + a synchronous NULL-stream htod + a drop, per decode step; against a
        // blocking compute stream each of those serializes the pipeline. Two packed async uploads
        // and one readback replace six of each.
        assert!(n <= MAX_VERIFY, "verify width {} exceeds MAX_VERIFY {}", n, MAX_VERIFY);
        let mut pf = vec![0.0f32; 3 * MAX_VERIFY];   // temps | top_ps | draft_qprobs
        let mut ki = vec![0i32; 2 * MAX_VERIFY];     // draft_tokens | top_ks
        for j in 0..n {
            pf[j] = temp;
            pf[MAX_VERIFY + j] = top_p;
            ki[MAX_VERIFY + j] = crate::sampler::gpu_top_k(top_k);   // TOPKM1: 0 / >64 -> the kernels' 64-candidate law, not argmax
        }
        for j in 0..n - 1 {
            pf[2 * MAX_VERIFY + j] = draft_qprobs[j];
            ki[j] = draft_tokens[j] as i32;
        }
        let dptr = |s: &cudarc::driver::CudaSlice<f32>| *s.device_ptr() as u64;
        unsafe {
            use cudarc::driver::sys::CUdeviceptr;
            cudarc::driver::result::memcpy_htod_async(
                *self.sv_pf.device_ptr() as CUdeviceptr, &pf[..], self.stream.stream).expect("htod pf");
            cudarc::driver::result::memcpy_htod_async(
                *self.sv_ki.device_ptr() as CUdeviceptr, &ki[..], self.stream.stream).expect("htod ki");
            cudarc::driver::result::memcpy_htod_async(
                *self.sv_sd.device_ptr() as CUdeviceptr, &seeds[..n], self.stream.stream).expect("htod sd");
        }
        let temps_ptr  = dptr(&self.sv_pf);
        let topp_ptr   = temps_ptr + (MAX_VERIFY * 4) as u64;
        let qprob_ptr  = temps_ptr + (2 * MAX_VERIFY * 4) as u64;
        let dtok_ptr   = *self.sv_ki.device_ptr() as u64;
        let topk_ptr   = dtok_ptr + (MAX_VERIFY * 4) as u64;

        let smem = ((2 * 64usize + 2 * 256usize) * 4) as u32;
        // S5F3 dump: the top-20 (token, renorm-p) table — run on the UNROUNDED logits BEFORE
        // spec_verify_b (which masks them destructively). Dump-only; t20_out None = free path.
        let dump_ptrs: Option<(u64, u64)> = if t20_out.is_some() {
            let vocab_cap = self.cfg.vocab_size;
            let tbl = self.sv_t20.get_or_init(|| {
                self.dev.alloc_zeros::<u64>(MAX_VERIFY * crate::dflash2::stepdump::DUMP_TOPK)
                    .expect("sv_t20 alloc")
            });
            let scr = self.sv_t20_scratch.get_or_init(|| {
                self.dev.alloc_zeros::<half::bf16>(MAX_VERIFY * vocab_cap)
                    .expect("sv_t20 scratch alloc")
            });
            Some((*tbl.device_ptr() as u64, *scr.device_ptr() as u64))
        } else { None };
        // W2 (Phase 13): JSON-schema mask over the verify logits [vocab, n] BEFORE
        // spec_verify_* selects anything, so the p(draft)/residual draws AND the host accept
        // rule see the masked distribution (allowed entries byte-identical).
        if let Some((mw, mf)) = self.verify_mask_ptrs() {
            let gy = ((mask_words_for(vocab) as u32) + 255) / 256;
            match &logits {
                VerifyLogits::B16(lg) => blaunch!(self, "logit_mask_b", (n as u32, gy, 1), (256,1,1), 0,
                    (d(lg), mw, mf, vocab as i32, n as i32)),
                VerifyLogits::F32(lg) => blaunch!(self, "logit_mask_f32_b", (n as u32, gy, 1), (256,1,1), 0,
                    (d(lg), mw, mf, vocab as i32, n as i32)),
            }
        }
        match logits {
            VerifyLogits::F32(l) => {
                blaunch!(self, "spec_verify_f32_b", (n as u32,1,1), (256,1,1), smem,
                    (*self.sv_p.device_ptr() as u64, *self.sv_r.device_ptr() as u64,
                     d(&l), dtok_ptr, qprob_ptr,
                     temps_ptr, topk_ptr, topp_ptr, *self.sv_sd.device_ptr() as u64,
                     vocab as i32, n as i32));
                pool.release(l, vocab*n);
            }
            VerifyLogits::B16(l) => {
                if let Some((tbl, scr)) = dump_ptrs {
                    blaunch!(self, "df2_topk20_dump_b", (n as u32,1,1), (256,1,1), smem,
                        (tbl, scr, d(&l), temps_ptr, topk_ptr, topp_ptr, vocab as i32, n as i32));
                }
                blaunch!(self, "spec_verify_b", (n as u32,1,1), (256,1,1), smem,
                    (*self.sv_p.device_ptr() as u64, *self.sv_r.device_ptr() as u64,
                     d(&l), dtok_ptr, qprob_ptr,
                     temps_ptr, topk_ptr, topp_ptr, *self.sv_sd.device_ptr() as u64,
                     vocab as i32, n as i32));
                pool.release_bf16(l, vocab*n);
            }
        }
        self.sync_stream();

        // One readback each for the f32 and i32 outputs (the bonus rides in resid_tok[n-1]).
        let p_vec = self.dev.dtoh_sync_copy(&self.sv_p).unwrap();
        let r_vec = self.dev.dtoh_sync_copy(&self.sv_r).unwrap();
        if let Some(out) = t20_out {
            let t20 = self.dev.dtoh_sync_copy(self.sv_t20.get().expect("t20 alloc")).unwrap();
            out.clear();
            out.extend_from_slice(&t20[..n * crate::dflash2::stepdump::DUMP_TOPK]);
        }

        let p_of_draft: Vec<f32> = p_vec[..n - 1].to_vec();
        let resid_tok: Vec<u32> = r_vec[..n - 1].iter().map(|&x| x as u32).collect();
        let bonus_tok = r_vec[n - 1] as u32;
        (VerifySample { p_of_draft, resid_tok, bonus_tok }, residual)
    }

    /// S5F2 L2 — `verify_forward_sample` with the REAL-q rejection-sampling residual (the
    /// SGLang `speculative_sampling_classic_kernel` semantics): same nucleus p_of_draft, but the
    /// rejection resample draws from the exact relu(p − q)+ renormalized (q = the selector's
    /// candidate-table weight at every candidate position, 0 elsewhere) instead of the pragmatic
    /// p \ {draft}. Combined with the host's accept rule `u·q < p` (q = the DRAWN candidate's
    /// weight), the emitted distribution is EXACTLY the target p — the L2 chi-square gate's
    /// contract. `cand_tok`/`cand_q` are the per-position candidate tables `[n-1][16]` row-major
    /// (from the sampled selector walk). qwen/bf16 only (the f32 hy_v3 twin is not needed — the
    /// DFlash2 path is Qwen-family).
    #[allow(clippy::too_many_arguments)]
    pub fn verify_forward_sample_rq(&self, pool: &mut Pool, tokens: &[u32],
                                    state: &mut BatchGpuState, slot: usize, kv_stride: usize,
                                    pos_start: usize, ckpt_slot: Option<usize>,
                                    penalty: Option<VerifyPenalty>,
                                    draft_tokens: &[u32], cand_tok: &[u32], cand_q: &[f32],
                                    temp: f32, top_k: usize, top_p: f32,
                                    seeds: &[u32],
                                    t20_out: Option<&mut Vec<u64>>) -> (VerifySample, B) {
        let cfg = &self.cfg;
        let _h = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let n = tokens.len();
        assert_eq!(draft_tokens.len(), n - 1, "draft_tokens must be length depth-1");
        assert_eq!(cand_tok.len(), (n - 1) * 16, "cand_tok must be [depth-1][16]");
        assert_eq!(cand_q.len(), (n - 1) * 16, "cand_q must be [depth-1][16]");
        assert_eq!(seeds.len(), n, "seeds must be length depth");

        let (logits, residual) = self.verify_forward_core(pool, tokens, state, slot, kv_stride, pos_start, ckpt_slot, cfg.lm_head_fp32);

        if let Some(p) = &penalty {
            match &logits {
                VerifyLogits::F32(_) => panic!("verify_forward_sample_rq: the f32 (hy_v3) arm is not wired (Qwen-only path)"),
                VerifyLogits::B16(l) => self.pen_apply_verify(p, d(l), false, n),
            }
        }

        assert!(n <= MAX_VERIFY, "verify width {} exceeds MAX_VERIFY {}", n, MAX_VERIFY);
        // Packed scratch: pf (temps | top_ps | draft_qprobs), ki (draft_tokens | top_ks),
        // cand (packed (q<<32)|tok u64 pairs per column — ONE pointer keeps the kernel launch
        // at the 12-arg cudarc cap), sd (seeds).
        let mut pf = vec![0.0f32; 3 * MAX_VERIFY];
        let mut ki = vec![0i32; 2 * MAX_VERIFY];
        let mut cand = vec![0u64; MAX_VERIFY * 16];
        for j in 0..n {
            pf[j] = temp;
            pf[MAX_VERIFY + j] = top_p;
            ki[MAX_VERIFY + j] = crate::sampler::gpu_top_k(top_k);   // TOPKM1: 0 / >64 -> the kernels' 64-candidate law, not argmax
        }
        for j in 0..n - 1 {
            ki[j] = draft_tokens[j] as i32;
            for k in 0..16 {
                let qbits = cand_q[j * 16 + k].to_bits() as u64;
                cand[j * 16 + k] = (qbits << 32) | cand_tok[j * 16 + k] as u64;
            }
        }
        let dptr = |s: &cudarc::driver::CudaSlice<f32>| *s.device_ptr() as u64;
        unsafe {
            use cudarc::driver::sys::CUdeviceptr;
            cudarc::driver::result::memcpy_htod_async(
                *self.sv_pf.device_ptr() as CUdeviceptr, &pf[..], self.stream.stream).expect("htod pf");
            cudarc::driver::result::memcpy_htod_async(
                *self.sv_ki.device_ptr() as CUdeviceptr, &ki[..], self.stream.stream).expect("htod ki");
            cudarc::driver::result::memcpy_htod_async(
                *self.sv_cand.device_ptr() as CUdeviceptr, &cand[..], self.stream.stream).expect("htod cand");
            cudarc::driver::result::memcpy_htod_async(
                *self.sv_sd.device_ptr() as CUdeviceptr, &seeds[..n], self.stream.stream).expect("htod sd");
        }
        let temps_ptr  = dptr(&self.sv_pf);
        let topp_ptr   = temps_ptr + (MAX_VERIFY * 4) as u64;
        let qprob_ptr  = temps_ptr + (2 * MAX_VERIFY * 4) as u64;
        let dtok_ptr   = *self.sv_ki.device_ptr() as u64;
        let topk_ptr   = dtok_ptr + (MAX_VERIFY * 4) as u64;
        let cand_ptr   = *self.sv_cand.device_ptr() as u64;

        let smem = ((2 * 64usize + 2 * 256usize) * 4) as u32;
        // S5F3 dump: the top-20 table BEFORE the destructive realq kernel (same as the
        // q=1 arm). Dump-only; t20_out None = free path.
        // W2 (Phase 13): JSON-schema mask over the verify logits [vocab, n] BEFORE
        // spec_verify_* selects anything, so the p(draft)/residual draws AND the host accept
        // rule see the masked distribution (allowed entries byte-identical).
        if let Some((mw, mf)) = self.verify_mask_ptrs() {
            let gy = ((mask_words_for(vocab) as u32) + 255) / 256;
            match &logits {
                VerifyLogits::B16(lg) => blaunch!(self, "logit_mask_b", (n as u32, gy, 1), (256,1,1), 0,
                    (d(lg), mw, mf, vocab as i32, n as i32)),
                VerifyLogits::F32(lg) => blaunch!(self, "logit_mask_f32_b", (n as u32, gy, 1), (256,1,1), 0,
                    (d(lg), mw, mf, vocab as i32, n as i32)),
            }
        }
        let dump_ptrs: Option<(u64, u64)> = if t20_out.is_some() {
            let vocab_cap = self.cfg.vocab_size;
            let tbl = self.sv_t20.get_or_init(|| {
                self.dev.alloc_zeros::<u64>(MAX_VERIFY * crate::dflash2::stepdump::DUMP_TOPK)
                    .expect("sv_t20 alloc")
            });
            let scr = self.sv_t20_scratch.get_or_init(|| {
                self.dev.alloc_zeros::<half::bf16>(MAX_VERIFY * vocab_cap)
                    .expect("sv_t20 scratch alloc")
            });
            Some((*tbl.device_ptr() as u64, *scr.device_ptr() as u64))
        } else { None };
        match logits {
            VerifyLogits::F32(_) => panic!("verify_forward_sample_rq: f32 arm not wired"),
            VerifyLogits::B16(l) => {
                if let Some((tbl, scr)) = dump_ptrs {
                    blaunch!(self, "df2_topk20_dump_b", (n as u32,1,1), (256,1,1), smem,
                        (tbl, scr, d(&l), temps_ptr, topk_ptr, topp_ptr, vocab as i32, n as i32));
                }
                blaunch!(self, "spec_verify_realq_b", (n as u32,1,1), (256,1,1), smem,
                    (*self.sv_p.device_ptr() as u64, *self.sv_r.device_ptr() as u64,
                     d(&l), dtok_ptr, qprob_ptr, cand_ptr,
                     temps_ptr, topk_ptr, topp_ptr, *self.sv_sd.device_ptr() as u64,
                     vocab as i32, n as i32));
                pool.release_bf16(l, vocab*n);
            }
        }
        self.sync_stream();

        let p_vec = self.dev.dtoh_sync_copy(&self.sv_p).unwrap();
        let r_vec = self.dev.dtoh_sync_copy(&self.sv_r).unwrap();
        if let Some(out) = t20_out {
            let t20 = self.dev.dtoh_sync_copy(self.sv_t20.get().expect("t20 alloc")).unwrap();
            out.clear();
            out.extend_from_slice(&t20[..n * crate::dflash2::stepdump::DUMP_TOPK]);
        }

        let p_of_draft: Vec<f32> = p_vec[..n - 1].to_vec();
        let resid_tok: Vec<u32> = r_vec[..n - 1].iter().map(|&x| x as u32).collect();
        let bonus_tok = r_vec[n - 1] as u32;
        (VerifySample { p_of_draft, resid_tok, bonus_tok }, residual)
    }
}

impl GpuModel {
    /// End-to-end MTP speculative-decoding probe (single lane, slot 0). Full loop: prefill →
    /// MTP prompt-prime → repeat { draft K-1 → verify K → accept longest prefix → GDN rollback →
    /// re-prime accepted prefix with real hiddens } → compare token-for-token against sequential
    /// greedy on slot 1 (lossless gate). Returns (mtp_tokens, seq_tokens, mtp_tok_s, seq_tok_s,
    /// accept_rate). MTP KV position and RoPE position both track the main model's position.
    /// Mean negative log-likelihood over a token window — the basis of perplexity.
    ///
    /// This is the quality gate for quantization. "The output still looks coherent" is not a gate:
    /// 4-bit round-to-nearest damage shows up as rare-token degradation and long-context drift long
    /// before fluency breaks. Perplexity on held-out text is the standard, sensitive measure, and it
    /// is cheap here — one prefill gives logits for every position at once, and the softmax stays on
    /// device (a [248320, N] logits block would be ~0.5 GB to move per window).
    ///
    /// Position i's logits predict token i+1, so a window of N tokens yields N-1 scored positions.
    pub fn window_nll(&self, pool: &mut Pool, state: &mut BatchGpuState, tokens: &[u32],
                      kv_stride: usize) -> (f64, usize) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let n = tokens.len();
        assert!(n >= 2, "need at least 2 tokens to score one position");

        self.zero_slot_state(state, 0, kv_stride);
        let (_first, residual) = self.prefill_batch(pool, tokens, state, 0, kv_stride, 0);

        // prefill returns the PRE-final-norm hidden; logits_batch expects post-norm. On qwen4_exp
        // the residual is the hyper-connected stream [n, rw] and the "final norm" is the
        // hyper_connection_mixer (the serving path's branch in prefill_batch_range) — applying
        // rmsnorm_b(final_norm) to it scored NaN on every window.
        let rw = cfg.resid_width();
        let out = pool.get_bf16(h * n);
        if let Some(m) = self.hc_mixer.as_ref() {
            self.hc_pre(pool, &out, None, d(&residual), m, n);
        } else {
            blaunch!(self, "rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
                (d(&out), d(&residual), d(&self.final_norm), h as i32, n as i32, fbits(cfg.rms_eps)));
        }
        pool.release_bf16(residual, rw * n);

        let logits = self.logits_batch(pool, &out, n);
        pool.release_bf16(out, h * n);

        // Score positions 0..n-2 against targets tokens[1..n-1].
        let ns = n - 1;
        let tgt: Vec<i32> = tokens[1..].iter().map(|&t| t as i32).collect();
        let mut tgt_dev = self.dev.alloc_zeros::<i32>(ns).unwrap();
        self.dev.memset_zeros(&mut tgt_dev).unwrap();
        self.dev.htod_sync_copy_into(&tgt, &mut tgt_dev).unwrap();
        let nll = pool.get(ns);
        self.dev.synchronize().unwrap();

        let nthr = 1024u32;
        blaunch!(self, "nll_b", (ns as u32,1,1), (nthr,1,1), (nthr * 4),
            (d(&nll), d(&logits), d(&tgt_dev), vocab as i32, ns as i32));
        pool.release_bf16(logits, vocab * n);
        self.sync_stream();

        let v = self.dev.dtoh_sync_copy(&nll).unwrap();
        pool.release(nll, ns);
        let sum: f64 = v.iter().take(ns).map(|&x| x as f64).sum();
        (sum, ns)
    }

    /// DEBUG PROBE (MoE correctness oracle). Run `moe_batch` for the FIRST MoE layer on a caller-given
    /// input `x_host` [hidden*batch] (col-major, bf16) and return `(layer_index, output [hidden*batch])`.
    /// A numpy reference over that layer's checkpoint weights + the same input validates the block.
    pub fn probe_moe(&self, x_host: &[half::bf16], batch: usize) -> (usize, Vec<half::bf16>) {
        let (li, moe) = self.layers.iter().enumerate()
            .find_map(|(i, l)| match &l.mlp { Ffn::Moe(m) => Some((i, m)), _ => None })
            .expect("no MoE layer in this model");
        let mut pool = Pool::new(self.dev.clone());
        let x = self.dev.htod_sync_copy(x_host).unwrap();
        let ReduceOut::Full(out) = self.moe_batch(&mut pool, &x, moe, batch, false) else {
            unreachable!("AR landing 2: the MoE probe never fuses")
        };
        let ov = self.dev.dtoh_sync_copy(&out).unwrap();
        self.sync_stream();
        (li, ov)
    }
}

impl GpuModel {
    /// DEBUG PROBE (cross-model spec-decode acceptance study). Teacher-force `tokens` and return the
    /// per-position greedy argmax: `pred[i] = argmax(logits at position i)` = the model's greedy
    /// next-token given `tokens[..=i]`, for i in 0..n. Mirrors `window_nll` exactly (same prefill +
    /// logits) but reads out argmax via the existing `argmax_b` kernel instead of scoring NLL. Purely
    /// observational — no serving path, no behaviour change. Used by `--dump-argmax`.
    pub fn window_argmax(&self, pool: &mut Pool, state: &mut BatchGpuState, tokens: &[u32],
                         kv_stride: usize) -> Vec<u32> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let n = tokens.len();
        assert!(n >= 1, "need at least 1 token");

        self.zero_slot_state(state, 0, kv_stride);
        let (_first, residual) = self.prefill_batch(pool, tokens, state, 0, kv_stride, 0);

        let out = pool.get_bf16(h * n);
        blaunch!(self, "rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
            (d(&out), d(&residual), d(&self.final_norm), h as i32, n as i32, fbits(cfg.rms_eps)));
        pool.release_bf16(residual, h * n);

        let logits = self.logits_batch(pool, &out, n);
        pool.release_bf16(out, h * n);

        let mut tok_dev = self.dev.alloc_zeros::<i32>(n).unwrap();
        self.dev.memset_zeros(&mut tok_dev).unwrap();
        let block = 1024u32;
        blaunch!(self, "argmax_b", (n as u32,1,1), (block,1,1), (block*8),
            (*tok_dev.device_ptr() as u64, d(&logits), vocab as i32, n as i32));
        pool.release_bf16(logits, vocab * n);
        self.sync_stream();

        let ids = self.dev.dtoh_sync_copy(&tok_dev).unwrap();
        ids.into_iter().take(n).map(|x| x as u32).collect()
    }
}
