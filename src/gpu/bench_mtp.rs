//! `impl GpuModel`: MTP benches and calibration (`calibrate_mtp_r`, `profile_mtp`, `bench_mtp*`, `bench_accept`, `bench_batch`).
//! Bench harnesses; see AGENTS.md section 5 for benchmark discipline. Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// Measure `r` = (cost of one MTP step) / (cost of one plain decode step).
    ///
    /// This is the whole basis of the auto-policy. A depth-2 MTP step emits `1 + acceptance` tokens
    /// (exactly: the accepted draft plus the bonus), and costs `r` decode-steps, so
    ///
    /// ```text
    /// speedup = (1 + acceptance) / r      =>      MTP pays iff  acceptance > r - 1
    /// ```
    ///
    /// `r` is a pure cost ratio: it depends on the model's shape (above all on what fraction of the
    /// weights the LM head is, since drafting reads it a second time) and not on the prompt. So it can
    /// be measured once at load and trusted. Acceptance, by contrast, is workload-dependent and has to
    /// be tracked live — hence the split: `r` here, acceptance in the scheduler's EMA.
    ///
    /// Measured values (r): 2B 1.54, 4B 1.35, 9B 1.29, 27B 1.12 — i.e. break-even acceptance of
    /// 54% / 35% / 29% / 12%. That is exactly why 2B (30% acceptance) loses and 27B (80%) wins big.
    /// r(d) = (cost of one MTP step at depth d) / (cost of a plain decode step), MEASURED per depth
    /// per CONTEXT BUCKET (E17) — returns (ctx point, r table at that ctx) pairs.
    ///
    /// The MTP identity is `speedup = tokens_per_step / r`, so r(d) and the acceptance curve together
    /// decide the optimal depth. Both halves have to be real: this measures r(d) directly rather than
    /// measuring one verify at N=2 and charging it to every depth. And the verify's KV attention
    /// bytes grow ∝ context while the draft chain's cost does not, so a single short-context table
    /// makes deep drafts look cheaper than they are at ≥16K — the policy over-reached exactly there.
    pub fn calibrate_mtp_r(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                           kv_stride: usize) -> Vec<(usize, Vec<(usize, f32)>)> {
        let _spin = crate::host_wait::SpinScope::new(); // CPUIDLE: policy timings see the exact (spin) path
        let _ = prompt;   // content is timing-neutral (see time_mtp_phases_at_ctx); arg kept so the
                          // SPMD head/node call shape is unchanged.
        // Determinism hygiene: zero the KV the phases will read. Timing itself is value-independent
        // (every kernel's loop bounds derive from pc, not content), but zeros keep it reproducible.
        for (li, lt) in self.cfg.layer_types.iter().enumerate() {
            if !matches!(lt, LayerType::FullAttention) { continue; }
            if let Some(k) = state.k_cache[li].as_mut() { self.dev.memset_zeros(k).unwrap(); }
            if let Some(v) = state.v_cache[li].as_mut() { self.dev.memset_zeros(v).unwrap(); }
        }
        // BASICS (owner 2026-11-02): one step per context size, as a percentage at the normal level
        let points = mtp_calib_ctx_points(kv_stride);
        let prog = crate::boot_report::Progress::new("measuring the speculation cost", points.len() as u64,
                                                     crate::boot_report::Unit::Items("context sizes"));
        let out: Vec<(usize, Vec<(usize, f32)>)> = points.iter().map(|&ctx| {
            let rows = self.time_mtp_phases_at_ctx(pool, state, ctx, kv_stride, 3);
            prog.add(1);
            let get = |k: &str| rows.iter().find(|r| r.0.trim().starts_with(k)).map(|r| r.1).unwrap_or(0.0);
            let decode = get("decode step");
            if decode <= 0.0 { return (ctx, vec![]); }
            let per_draft = get("mtp_draft_step") + get("argmax_hidden");
            let tail = get("copy_gdn_slot") + get("mtp_reprime");
            let table = AUTO_DEPTHS.iter().map(|&d| {
                // One MTP step: the draft chain (d-1 MTP forwards, each + an LM-head argmax), the
                // verify at N=d, the GDN rollback, and the re-prime over the accepted prefix.
                let step = (d - 1) as f64 * per_draft
                    + get(&format!("verify_forward_sample N={}", d))
                    + tail;
                (d, (step / decode) as f32)
            }).collect();
            (ctx, table)
        }).collect();
        prog.finish();
        out
    }

    /// E11 measurement harness (`--bench-decode-ctx`): per-phase p50 of one decode/MTP step at
    /// context `ctx` (default 2048 — the "decode p50 at 2K" of the E11 brief). NO prefill (zeros
    /// KV; every kernel's cost is a pure function of pc — see time_mtp_phases_at_ctx). The first
    /// row, "decode step (batch=1, baseline)", IS the per-token decode p50 at ctx. Best-of-3
    /// (three full runs, min per phase) — the discipline's interleaved-best-of-N.
    pub fn bench_decode_at_ctx(&self, ctx: usize, kv_stride: usize, runs: usize) -> Vec<(String, f64)> {
        let _spin = crate::host_wait::SpinScope::new(); // CPUIDLE: policy timings see the exact (spin) path
        let mut pool = Pool::new(self.dev.clone());
        // 0 = the decode/verify lane, 1 = spare, 2 = the MTP snapshot the rollback phase copies
        // from (copy_gdn_slot(s, 2, 0)), then ONE checkpoint slot per verify column: the verify
        // ladder below runs at ckpt_slot=2 with widths up to max(AUTO_DEPTHS), and the GDN kernels
        // write column t's post-state into slot (ckpt_slot + t) UNGUARDED (verify_forward_core
        // asserts cs + n <= state_slots + 1 - fail loudly here, not in device memory). 3 slots
        // panicked at the first depth-3 verify: "verify ckpt band overflow: ckpt_slot 2 + width 3
        // needs 4 GDN state slots, state has 3" (B25). Size for the WIDEST verify this harness can
        // issue: ckpt_slot + MAX_VERIFY - 1 slots, plus one spare. (An earlier 1-slot state made
        // the rollback memcpy read past the allocation -> CUDA_ERROR_INVALID_VALUE, since 8ec059b.)
        let mut state = self.new_batch_state(1, 2 + MAX_VERIFY, kv_stride);
        // best-of-runs per phase (each run re-times every phase in the same order)
        let mut agg: Vec<(String, f64)> = Vec::new();
        for _ in 0..runs {
            let rows = self.time_mtp_phases_at_ctx(&mut pool, &mut state, ctx, kv_stride, 3);
            for (name, ms) in rows {
                match agg.iter_mut().find(|(n, _)| *n == name) {
                    Some((_, best)) => { if ms < *best { *best = ms; } }
                    None => agg.push((name, ms)),
                }
            }
        }
        agg
    }

    /// Time every phase of one MTP step at context `ctx`: the decode baseline, the draft chain, the
    /// verify at every candidate depth (its KV reads span [0, ctx)), and the rollback/re-prime tail.
    /// NO prefill: the caches hold zeros and every kernel's cost is a pure function of pc (loop
    /// bounds, tile counts, split-K fan-out all derive from positions, never from values), so a
    /// scratch state times identically to a real one. That is what makes per-bucket calibration
    /// cheap — measuring at 131072 ctx costs the phases, not a 131072-token prefill.
    fn time_mtp_phases_at_ctx(&self, pool: &mut Pool, state: &mut BatchGpuState, ctx: usize,
                              kv_stride: usize, iters: usize) -> Vec<(String, f64)> {
        let _spin = crate::host_wait::SpinScope::new(); // CPUIDLE: policy timings see the exact (spin) path
        let h = self.cfg.hidden_size;
        let _nkv = self.cfg.num_kv_heads; let hd = self.cfg.head_dim;
        let mut bufs = self.new_decode_buffers(3);
        let mut mtp_kc = self.dev.alloc_zeros::<half::bf16>(self.mtp_kc_elems(kv_stride)).unwrap();
        let mut mtp_vc = self.dev.alloc_zeros::<half::bf16>(self.mtp_kv_heads() * kv_stride * hd).unwrap();
        self.dev.memset_zeros(&mut mtp_kc).unwrap();
        self.dev.memset_zeros(&mut mtp_vc).unwrap();
        self.dev.synchronize().unwrap();
        let mtp_kc_ptr = *mtp_kc.device_ptr();
        let mtp_vc_ptr = *mtp_vc.device_ptr();
        let h_scratch = self.dev.alloc_zeros::<half::bf16>(h).unwrap();
        let a0 = 0u32;   // a valid token id; the actual id is timing-neutral
        let depth = 2usize;

        let mut out: Vec<(String, f64)> = Vec::new();
        let mut time_it = |name: &str, n: usize, f: &mut dyn FnMut()| {
            f(); // warm
            let t0 = std::time::Instant::now();
            for _ in 0..n { f(); }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            out.push((name.to_string(), ms));
        };

        // --- baseline: one plain decode step at pc = ctx (what a token costs without MTP) ---
        self.dev.htod_sync_copy_into(&[a0 as i32, 0, 0], &mut bufs.token_ids_dev).unwrap();
        self.dev.htod_sync_copy_into(&[(ctx - 1) as i32, 0, 0], &mut bufs.pos_dev).unwrap();
        self.dev.htod_sync_copy_into(&[0i32, 1, 2], &mut bufs.slot_ids_dev).unwrap();
        self.dev.synchronize().unwrap();
        {
            let (p, s, b) = (&mut *pool, &mut *state, &mut bufs);
            let mut f = || { self.forward_decode_gpu(p, b, s, kv_stride, ctx, 1); self.sync_stream(); };
            time_it("decode step (batch=1, baseline)", iters, &mut f);
        }

        // --- MTP head forward (the draft), and the LM-head argmax that turns it into a token ---
        {
            let p = &mut *pool;
            let mut f = || {
                let m = self.mtp_draft_step(p, &h_scratch, a0 as i32, ctx - 1,
                                            mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                self.sync_stream();
                p.release_bf16(m, h);
            };
            time_it("  mtp_draft_step (MTP layer fwd)", iters, &mut f);
        }
        {
            let p = &mut *pool;
            let mut f = || { let _ = self.argmax_hidden(p, &h_scratch); };
            time_it("  argmax_hidden (LM head + dtoh)", iters, &mut f);
        }

        // --- GDN rollback ---
        {
            let s = &*state;
            let mut f = || { self.copy_gdn_slot(s, 2, 0); self.sync_stream(); };
            time_it("  copy_gdn_slot (GDN rollback)", iters, &mut f);
        }

        // --- batched re-prime over the accepted prefix (depth columns, one MTP-layer forward) ---
        {
            let rp = pool.get_bf16(h * depth);
            let toks = vec![a0; depth];
            let p = &mut *pool;
            let mut f = || {
                // Start at ctx-depth so the LAST written position is ctx-1: pos_start=ctx-1 would
                // write row ctx == kv_stride at the top bucket — an OOB the unsharded 4-head cache
                // silently absorbed into head 1's region (found by compute-sanitizer under §4.1).
                self.mtp_reprime(p, &rp, &toks, ctx - depth, mtp_kc_ptr, mtp_vc_ptr, kv_stride);
            };
            time_it("  mtp_reprime (batched, N=depth)", iters, &mut f);
            pool.release_bf16(rp, h * depth);
        }

        // --- the FULL stochastic verify at every candidate depth, its KV reads spanning [0, ctx) ---
        for &d in AUTO_DEPTHS {
            let toks = vec![a0; d];
            let drafts = vec![a0; d - 1];
            let qp = vec![1.0f32; d - 1];
            let seeds: Vec<u32> = (0..d as u32).collect();
            let (p, s) = (&mut *pool, &mut *state);
            let mut f = || {
                let (_vs, res) = self.verify_forward_sample(
                    p, &toks, s, 0, kv_stride, ctx - d, Some(2), None,
                    &drafts, &qp, 0.7, 20, 0.8, &seeds, None);
                p.release_bf16(res, h * d);
            };
            time_it(&format!("verify_forward_sample N={}", d), iters, &mut f);
        }

        out
    }

    /// Phase timings for one stochastic-MTP step (`--profile-mtp`).
    ///
    /// A step accepts ~1.76 tokens but costs ~1.6x a plain decode step, so most of the acceptance
    /// win is being eaten somewhere. This times each phase against a plain decode step (the baseline
    /// a token would otherwise cost) so the overhead can be attributed instead of guessed at.
    pub fn profile_mtp(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                       kv_stride: usize, iters: usize) -> Vec<(String, f64)> {
        let h = self.cfg.hidden_size;
        let plen = prompt.len();
        let _nkv = self.cfg.num_kv_heads; let hd = self.cfg.head_dim;
        let mut bufs = self.new_decode_buffers(3);

        let mut mtp_kc = self.dev.alloc_zeros::<half::bf16>(self.mtp_kc_elems(kv_stride)).unwrap();
        let mut mtp_vc = self.dev.alloc_zeros::<half::bf16>(self.mtp_kv_heads() * kv_stride * hd).unwrap();
        self.dev.memset_zeros(&mut mtp_kc).unwrap();
        self.dev.memset_zeros(&mut mtp_vc).unwrap();
        self.dev.synchronize().unwrap();
        let mtp_kc_ptr = *mtp_kc.device_ptr();
        let mtp_vc_ptr = *mtp_vc.device_ptr();
        let h_scratch = self.dev.alloc_zeros::<half::bf16>(h).unwrap();

        self.zero_slot_state(state, 0, kv_stride);
        let (a0, hout0) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        let copy_stream = self.stream.stream;
        let copy_col = |dst_ptr: u64, src_buf: &B, col: usize| unsafe {
            let src = *src_buf.device_ptr() as u64 + (col as u64) * (h as u64) * 2;
            cudarc::driver::result::memcpy_dtod_async(dst_ptr, src, h * 2, copy_stream).unwrap();
        };
        copy_col(*h_scratch.device_ptr(), &hout0, plen - 1);
        pool.release_bf16(hout0, h * plen);
        self.sync_stream();

        // Penalty buffers, sized as the scheduler sizes them.
        let depth = 2usize;
        let mp = MAX_PEN_TOKENS;
        let mut pen_tokens = self.dev.alloc_zeros::<i32>(depth * mp).unwrap();
        let mut pen_counts = self.dev.alloc_zeros::<i16>(depth * mp).unwrap();
        let mut pen_rep = self.dev.alloc_zeros::<f32>(depth).unwrap();
        let mut pen_pres = self.dev.alloc_zeros::<f32>(depth).unwrap();
        let mut pen_freq = self.dev.alloc_zeros::<f32>(depth).unwrap();
        self.dev.synchronize().unwrap();

        let mut out: Vec<(String, f64)> = Vec::new();
        let mut time_it = |name: &str, n: usize, f: &mut dyn FnMut()| {
            f(); // warm
            let t0 = std::time::Instant::now();
            for _ in 0..n { f(); }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            out.push((name.to_string(), ms));
        };

        // --- baseline: one plain decode step (what a token costs without MTP) ---
        self.dev.htod_sync_copy_into(&[a0 as i32, 0, 0], &mut bufs.token_ids_dev).unwrap();
        self.dev.htod_sync_copy_into(&[plen as i32, 0, 0], &mut bufs.pos_dev).unwrap();
        self.dev.htod_sync_copy_into(&[0i32, 1, 2], &mut bufs.slot_ids_dev).unwrap();
        self.dev.synchronize().unwrap();
        {
            let (p, s, b) = (&mut *pool, &mut *state, &mut bufs);
            let mut f = || { self.forward_decode_gpu(p, b, s, kv_stride, plen + 1, 1); self.sync_stream(); };
            time_it("decode step (batch=1, baseline)", iters, &mut f);
        }

        // --- MTP head forward (the draft), and the LM-head argmax that turns it into a token ---
        {
            let p = &mut *pool;
            let mut f = || {
                let m = self.mtp_draft_step(p, &h_scratch, a0 as i32, plen - 1,
                                            mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                self.sync_stream();
                p.release_bf16(m, h);
            };
            time_it("  mtp_draft_step (MTP layer fwd)", iters, &mut f);
        }
        {
            let p = &mut *pool;
            let mut f = || { let _ = self.argmax_hidden(p, &h_scratch); };
            time_it("  argmax_hidden (LM head + dtoh)", iters, &mut f);
        }

        // --- the penalty upload the scheduler does on EVERY step ---
        {
            let tks = vec![1i32; depth * mp];
            let cts = vec![1i16; depth * mp];
            let rp = vec![1.1f32; depth];
            let mut f = || {
                self.dev.htod_sync_copy_into(&tks, &mut pen_tokens).unwrap();
                self.dev.htod_sync_copy_into(&cts, &mut pen_counts).unwrap();
                self.dev.htod_sync_copy_into(&rp, &mut pen_rep).unwrap();
                self.dev.htod_sync_copy_into(&rp, &mut pen_pres).unwrap();
                self.dev.htod_sync_copy_into(&rp, &mut pen_freq).unwrap();
                self.dev.synchronize().unwrap();
            };
            time_it("  penalty upload (5 htod + sync)", iters, &mut f);
        }

        // --- verify: backbone+logits alone, then the greedy and stochastic tails on top ---
        // The verify reads exactly the same weights as a decode step, so at N=1 it *should* cost the
        // same. Sweep N (and the ping-pong checkpoint) to separate "cost of the extra column" from
        // "cost of the prefill-flavoured kernels".
        // Sweep N with the checkpoint on AND off at each width. With a flat-in-N GEMM the weight bytes
        // (which set decode cost) do not change with N, so any growth here is NOT the GEMM: it is the
        // GDN scan, the attention, or the per-column checkpoint writes. Separating them is the point.
        let vin = [a0, a0];
        for (n, ck, label) in [
            (1usize, None,    "verify_core N=1, no ckpt"),
            (1usize, Some(2), "verify_core N=1, +ckpt"),
            (2usize, None,    "verify_core N=2, no ckpt"),
            (2usize, Some(2), "verify_core N=2, +ckpt"),
            (4usize, None,    "verify_core N=4, no ckpt"),
            (4usize, Some(2), "verify_core N=4, +ckpt"),
            (8usize, None,    "verify_core N=8, no ckpt"),
            (8usize, Some(2), "verify_core N=8, +ckpt"),
        ] {
            let toks = vec![a0; n];
            let (p, s) = (&mut *pool, &mut *state);
            let mut f = || {
                let (lg, res) = self.verify_forward_core(p, &toks, s, 0, kv_stride, plen, ck, false);
                let lg = lg.into_b16();
                self.sync_stream();
                p.release_bf16(lg, self.cfg.vocab_size * n);
                p.release_bf16(res, h * n);
            };
            time_it(label, iters, &mut f);
        }
        {
            let (p, s) = (&mut *pool, &mut *state);
            let mut f = || {
                let (lg, res) = self.verify_forward_core(p, &vin, s, 0, kv_stride, plen, Some(2), false);
                let lg = lg.into_b16();
                self.sync_stream();
                p.release_bf16(lg, self.cfg.vocab_size * 2);
                p.release_bf16(res, h * 2);
            };
            time_it("verify_forward_core (backbone+logits, N=2)", iters, &mut f);
        }
        {
            let (p, s) = (&mut *pool, &mut *state);
            let mut f = || {
                let (_pr, res) = self.verify_forward(p, &vin, s, 0, kv_stride, plen, Some(2), None);
                p.release_bf16(res, h * 2);
            };
            time_it("verify_forward (greedy: +argmax+dtoh)", iters, &mut f);
        }
        // The FULL stochastic verify at every candidate depth. This is what `calibrate_mtp_r` composes
        // r(d) from. Measuring it once at N=2 and charging that to every depth (what this used to do)
        // made depth 2 look optimal by construction — harmless while the verify was superlinear in N,
        // and exactly wrong now that it is flat and depth is the whole decision.
        for &d in AUTO_DEPTHS {
            let toks = vec![a0; d];
            let drafts = vec![a0; d - 1];
            let qp = vec![1.0f32; d - 1];
            let seeds: Vec<u32> = (0..d as u32).collect();
            let (p, s) = (&mut *pool, &mut *state);
            let mut f = || {
                let (_vs, res) = self.verify_forward_sample(
                    p, &toks, s, 0, kv_stride, plen, Some(2), None,
                    &drafts, &qp, 0.7, 20, 0.8, &seeds, None);
                p.release_bf16(res, h * d);
            };
            time_it(&format!("verify_forward_sample N={}", d), iters, &mut f);
        }

        // --- GDN rollback ---
        {
            let s = &*state;
            let mut f = || { self.copy_gdn_slot(s, 2, 0); self.sync_stream(); };
            time_it("  copy_gdn_slot (GDN rollback)", iters, &mut f);
        }

        // --- batched re-prime over the accepted prefix (depth columns, one MTP-layer forward) ---
        {
            let rp = pool.get_bf16(h * depth);
            let toks = vec![a0; depth];
            let p = &mut *pool;
            let mut f = || {
                self.mtp_reprime(p, &rp, &toks, plen - 1,
                                 mtp_kc_ptr, mtp_vc_ptr, kv_stride);
            };
            time_it("  mtp_reprime (batched, N=depth)", iters, &mut f);
            pool.release_bf16(rp, h * depth);
        }

        out
    }

    /// Distribution-exactness gate for stochastic MTP (`--bench-mtp-sample`).
    ///
    /// Greedy MTP is *bitwise* lossless, and `--bench-mtp` gates it by direct comparison against
    /// sequential greedy. Stochastic MTP can only ever be *distribution*-exact, so it needs a
    /// statistical gate instead. The backbone is already proven (verify column 0 == decode, see
    /// `--bench-verify` / `--probe-state`), which leaves the accept/residual/RNG math as the only
    /// thing that can break exactness. So this pins the prefix — one prefill, one greedy draft, one
    /// verify, giving a FIXED target distribution `p` and a FIXED draft token `x` — and then draws
    /// `trials` emissions several ways:
    ///
    ///   analytic p  the nucleus distribution the plain sampler is *defined* to emit (ground truth)
    ///   A: sampler  real `sample_b` launches on that same logits column   (the sampling-noise floor)
    ///   B: MTP      real `spec_verify_b` + the production host accept rule (the thing under test)
    ///   C: bonus    `spec_verify_b`'s all-accepted bonus column (also a draw from p)
    ///
    /// With a greedy (point-mass) draft, `q(x) = 1`, so speculative rejection sampling accepts `x`
    /// with probability exactly `p(x)` and otherwise draws from `p` with `x` removed and
    /// renormalized — which composes back to exactly `p` for every token. B must therefore be
    /// statistically indistinguishable from A. The gate is `TVD(B, p) <= TVD(A, p)`-ish: A's distance
    /// is the noise floor at this trial count, so "small" is not enough — B has to reach the floor.
    ///
    /// No penalty is applied here: this isolates the sampling math from the penalty transform.
    pub fn bench_mtp_sample(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                            kv_stride: usize, temp: f32, top_k: usize, top_p: f32, trials: usize,
                            rng_base: u64)
                           -> MtpSampleStats {
        use crate::batch::{rng_u32, rng_uniform, splitmix64, RNG_DOM_VERIFY, RNG_DOM_ACCEPT};
        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let plen = prompt.len();
        let _nkv = self.cfg.num_kv_heads; let hd = self.cfg.head_dim;

        // alloc_zeros is cuMemAllocAsync — it does NOT zero. Explicitly memset anything a compute
        // kernel will read (HANDOFF invariant 1).
        let mut mtp_kc = self.dev.alloc_zeros::<half::bf16>(self.mtp_kc_elems(kv_stride)).unwrap();
        let mut mtp_vc = self.dev.alloc_zeros::<half::bf16>(self.mtp_kv_heads() * kv_stride * hd).unwrap();
        self.dev.memset_zeros(&mut mtp_kc).unwrap();
        self.dev.memset_zeros(&mut mtp_vc).unwrap();
        self.dev.synchronize().unwrap();
        let mtp_kc_ptr = *mtp_kc.device_ptr();
        let mtp_vc_ptr = *mtp_vc.device_ptr();
        let h_scratch = self.dev.alloc_zeros::<half::bf16>(h).unwrap();
        let cur_hidden = self.dev.alloc_zeros::<half::bf16>(h).unwrap();

        self.zero_slot_state(state, 0, kv_stride);
        let (a0, hout0) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);

        let copy_stream = self.stream.stream;
        let copy_col = |dst_ptr: u64, src_buf: &B, col: usize| unsafe {
            let src = *src_buf.device_ptr() as u64 + (col as u64) * (h as u64) * 2;
            cudarc::driver::result::memcpy_dtod_async(dst_ptr, src, h * 2, copy_stream).unwrap();
        };

        // Prompt-prime the MTP KV exactly as the serving path does (positions 0..plen-2).
        // Must stay a call to the SAME primitive the server uses -- this used to be an open-coded
        // per-token loop whose comment claimed it mirrored serving, which is exactly the kind of
        // copy that silently stops mirroring it.
        self.mtp_prime_prompt(pool, &hout0, &prompt[1..plen], mtp_kc_ptr, mtp_vc_ptr, kv_stride, 0);
        copy_col(*cur_hidden.device_ptr(), &hout0, plen - 1);
        pool.release_bf16(hout0, h * plen);

        // One greedy draft at depth-2, exactly as the production stochastic path drafts.
        let m = self.mtp_draft_step(pool, &cur_hidden, a0 as i32, plen - 1,
                                    mtp_kc_ptr, mtp_vc_ptr, kv_stride);
        copy_col(*cur_hidden.device_ptr(), &m, 0);
        pool.release_bf16(m, h);
        let x_draft = self.argmax_hidden(pool, &cur_hidden);

        // S5F: the gate body (verify + CH-column loop + stats) is shared with the DFlash2 source
        // (--bench-df2-sample): the DRAFT source is the only difference, and the gate's claim is
        // about the sampler, never the draft.
        self.bench_rejection_gate(pool, state, prompt, kv_stride, a0, x_draft,
                                  temp, top_k, top_p, trials, rng_base)
    }
}

impl GpuModel {
    pub fn bench_accept(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                        kv_stride: usize, depth: usize, max_new: usize, ngram: usize)
                        -> anyhow::Result<(Vec<AcceptSample>, Vec<u32>)> {
        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let plen = prompt.len();
        let nkv = self.cfg.num_kv_heads; let hd = self.cfg.head_dim;
        let mut mtp_kc = self.dev.alloc_zeros::<half::bf16>(self.mtp_kc_elems(kv_stride)).unwrap();
        let mut mtp_vc = self.dev.alloc_zeros::<half::bf16>(self.mtp_kv_heads() * kv_stride * hd).unwrap();
        self.dev.memset_zeros(&mut mtp_kc).unwrap();
        self.dev.memset_zeros(&mut mtp_vc).unwrap();
        self.dev.synchronize().unwrap();
        let mtp_kc_ptr = *mtp_kc.device_ptr();
        let mtp_vc_ptr = *mtp_vc.device_ptr();
        let h_prev = self.dev.alloc_zeros::<half::bf16>(h).unwrap();
        let h_save = self.dev.alloc_zeros::<half::bf16>(h).unwrap();
        let h_scratch = self.dev.alloc_zeros::<half::bf16>(h).unwrap();
        let cur_hidden = self.dev.alloc_zeros::<half::bf16>(h).unwrap();

        self.zero_slot_state(state, 0, kv_stride);
        let (a0, hout0) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        if crate::opts::var(crate::opt!("accept-debug")).is_ok() {
            let v: Vec<half::bf16> = self.dev.dtoh_sync_copy(&hout0).unwrap();
            let nans = v.iter().filter(|x| x.is_nan()).count();
            eprintln!("  [bench] hout0 prefill[{}]: {nans} NaN, first={:?} last={:?}",
                      v.len(), v.first().map(|x| x.to_f32()), v.last().map(|x| x.to_f32()));
        }
        self.dbg_probe_prefill("ACCEPT", a0, &hout0, h, plen, state, kv_stride);   // --debug-hash

        let copy_stream = self.stream.stream;
        let copy_col = |dst_ptr: u64, src_buf: &B, col: usize| unsafe {
            let src = *src_buf.device_ptr() as u64 + (col as u64) * (h as u64) * 2;
            cudarc::driver::result::memcpy_dtod_async(dst_ptr, src, h * 2, copy_stream).unwrap();
        };

        self.mtp_prime_prompt(pool, &hout0, &prompt[1..plen], mtp_kc_ptr, mtp_vc_ptr, kv_stride, 0);
        // --debug-hash: whole buffers — both were memset+sync'd, so unwritten rows are zeros.
        self.dbg_hash_bf16("ACCEPT.mtpkc", &mtp_kc, nkv * kv_stride * hd);
        self.dbg_hash_bf16("ACCEPT.mtpvc", &mtp_vc, nkv * kv_stride * hd);
        let mut mtp_pos = plen - 1;
        copy_col(*h_prev.device_ptr(), &hout0, plen - 1);
        self.dbg_hash_bf16("ACCEPT.h_prev0", &h_prev, h);   // --debug-hash
        pool.release_bf16(hout0, h * plen);

        let mut out = vec![a0];
        let mut committed_tok = a0;
        let mut main_pos = plen;
        let mut samples: Vec<AcceptSample> = Vec::new();

        while out.len() < max_new {
            copy_col(*h_save.device_ptr(), &h_prev, 0);
            let mut drafts: Vec<u32> = Vec::with_capacity(depth - 1);
            let mut head_top3: Vec<[u32; 3]> = Vec::with_capacity(depth - 1);   // head's top-3 per draft pos
            copy_col(*cur_hidden.device_ptr(), &h_prev, 0);
            let mut cur_tok = committed_tok as i32;
            let mut dpos = mtp_pos;
            // Running context for the n-gram lookup: the full realized sequence, extended with the
            // drafts as we propose them (each draft becomes context for the next lookup).
            let mut work: Vec<u32> = Vec::new();
            if ngram > 0 { work.extend_from_slice(prompt); work.extend_from_slice(&out[1..]); }
            for _ in 0..depth - 1 {
                let m = self.mtp_draft_step(pool, &cur_hidden, cur_tok, dpos,
                                            mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                if out.len() == 1 && drafts.is_empty() { self.dbg_hash_bf16("ACCEPT.m0", &m, h); }   // --debug-hash
                dpos += 1;
                copy_col(*cur_hidden.device_ptr(), &m, 0);
                pool.release_bf16(m, h);
                let t3 = self.topk_hidden(pool, &cur_hidden, 3);   // top-1 == argmax (the draft)
                let a1 = self.argmax_hidden(pool, &cur_hidden);
                if crate::opts::var(crate::opt!("accept-debug")).is_ok() {
                    eprintln!("  [accept-dbg] draft candidates: topk={:?} argmax_hidden={} {}",
                              t3, a1, if a1 == t3[0] { "" } else { "  <-- MISMATCH" });
                }
                head_top3.push([t3[0], *t3.get(1).unwrap_or(&t3[0]), *t3.get(2).unwrap_or(&t3[0])]);
                cur_tok = a1 as i32;
                // PROMPT-LOOKUP OVERRIDE: if the last `ngram` tokens of the context recur earlier, the
                // token that followed the most recent earlier occurrence is a free, exact-copy draft.
                // Prefer it over the 1-layer head's guess on copyable (tool/JSON) text. The MTP hidden
                // was computed for the PREVIOUS token, so overriding this draft keeps the chain valid:
                // the next step feeds (this hidden, the overridden token), exactly as MTP chains.
                if ngram > 0 && work.len() >= ngram {
                    let tail = &work[work.len() - ngram..];
                    for i in (0..work.len() - ngram).rev() {
                        if &work[i..i + ngram] == tail {
                            if i + ngram < work.len() { cur_tok = work[i + ngram] as i32; }
                            break;
                        }
                    }
                }
                drafts.push(cur_tok as u32);
                if ngram > 0 { work.push(cur_tok as u32); }
            }

            let mut verify_input = vec![committed_tok];
            verify_input.extend(drafts.iter().copied());
            let n = verify_input.len();

            // Same verify the serving path runs, but keep the LOGITS so we can read the target's
            // confidence rather than only its argmax. hy_v3: unrounded fp32 (the greedy contract).
            let (logits, vout) = self.verify_forward_core(
                pool, &verify_input, state, 0, kv_stride, main_pos, Some(2), true);
            self.sync_stream();
            // Gate hardening: an aborted link (I9) no-ops the kernels and leaves the buffers stale —
            // reporting an acceptance number from that state is the swallowed-abort failure. Check
            // the cooperative status word every step and fail LOUDLY instead.
            let abort = crate::net::traced_abort_status();
            if abort != 0 {
                anyhow::bail!(
                    "TP link ABORTED (status {abort}, tail_fires {}) mid-acceptance — refusing to \
                     report an acceptance number from an aborted run (I9 no-op'd kernels)",
                    crate::net::traced_tail_fires());
            }
            let lg_f32: Vec<f32> = match logits {
                VerifyLogits::B16(lg) => {
                    let v = self.dev.dtoh_sync_copy(&lg).unwrap();
                    pool.release_bf16(lg, vocab * n);
                    v.iter().map(|x| x.to_f32()).collect()
                }
                VerifyLogits::F32(lg) => {
                    let v = self.dev.dtoh_sync_copy(&lg).unwrap();
                    pool.release(lg, vocab * n);
                    v
                }
            };
            let lg = &lg_f32;

            // Per verify column: target argmax, top-1 probability, and the top1-top2 margin.
            let mut preds = Vec::with_capacity(n);
            let mut conf = Vec::with_capacity(n);
            for c in 0..n {
                let col = &lg[c * vocab..(c + 1) * vocab];
                let (mut b1, mut b2, mut i1) = (f32::NEG_INFINITY, f32::NEG_INFINITY, 0usize);
                for (i, &v) in col.iter().enumerate() {
                    if v > b1 { b2 = b1; b1 = v; i1 = i; } else if v > b2 { b2 = v; }
                }
                // softmax top-1 probability (subtract max for stability)
                let denom: f32 = col.iter().map(|&x| (x - b1).exp()).sum();
                preds.push(i1 as u32);
                conf.push((1.0 / denom, b1 - b2));      // (p_top1, logit margin)
            }

            let mut nacc = 0usize;
            while nacc < drafts.len() && preds[nacc] == drafts[nacc] { nacc += 1; }

            // Record only the draft positions whose chain was STILL ON THE CORRECT PREFIX: 0..=nacc.
            //
            // Past the first rejection the drafts are conditioned on a token the target did not
            // choose, so their failure says nothing about the head — counting them would slander it.
            // Position `nacc` (the first rejection, when there is one) is the single most informative
            // sample we get: correct context, head still missed. That is where "weak head" and "hard
            // text" separate.
            for i in 0..drafts.len().min(nacc + 1) {
                let (p1, margin) = conf[i];
                // Does the target's argmax (preds[i]) land in the head's top-2 / top-3 here? That is
                // exactly the coverage a fork offering the head's top-k at this position would rescue.
                let tgt = preds[i];
                let t3 = head_top3[i];
                samples.push(AcceptSample {
                    depth_idx: i + 1,
                    target_top1_p: p1,
                    target_margin: margin,
                    accepted: i < nacc,          // the prefix rule: accepted iff every earlier one was
                    covered_top2: tgt == t3[0] || tgt == t3[1],
                    covered_top3: tgt == t3[0] || tgt == t3[1] || tgt == t3[2],
                });
                if crate::opts::var(crate::opt!("accept-debug")).is_ok() && samples.len() <= 24 {
                    eprintln!("  [accept-dbg] pos+{} draft={} head_top3={:?} target={} p={:.2} margin={:.2}",
                              i + 1, drafts[i], t3, tgt, p1, margin);
                }
            }

            let bonus = preds[nacc];
            for &d in drafts.iter().take(nacc) { out.push(d); }
            if out.len() < max_new { out.push(bonus); }

            if nacc + 1 != depth { self.copy_gdn_slot(state, 2 + nacc, 0); }
            copy_col(*h_prev.device_ptr(), &vout, nacc);

            for k in 0..=nacc {
                if k == 0 {
                    let m = self.mtp_draft_step(pool, &h_save, committed_tok as i32, main_pos - 1,
                                                mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                    pool.release_bf16(m, h);
                } else {
                    copy_col(*h_scratch.device_ptr(), &vout, k - 1);
                    let m = self.mtp_draft_step(pool, &h_scratch, drafts[k - 1] as i32, main_pos - 1 + k,
                                                mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                    pool.release_bf16(m, h);
                }
            }
            pool.release_bf16(vout, h * depth);

            committed_tok = *out.last().unwrap();
            main_pos += nacc + 1;
            mtp_pos += nacc + 1;
            if main_pos + depth + 2 >= kv_stride { break; }
        }
        Ok((samples, out))
    }

    pub fn bench_mtp(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                     kv_stride: usize, depth: usize, max_new: usize)
                    -> (Vec<u32>, Vec<u32>, f32, f32, f32) {
        let (a, b, c, d, e, _n_steps, _acc_k, _dr_k) = self.bench_mtp_steps(pool, state, prompt, kv_stride, depth, max_new);
        (a, b, c, d, e)
    }

    /// `bench_mtp` + the raw step count (B8/G2: tau = emitted/steps must be COUNTED, not modelled).
    /// S-B4: also returns per-position draft acceptance — accepted_at_k[k] = verify steps where the
    /// (k+1)-th draft was accepted (longest-prefix, so accepted_at_k is monotone decreasing), and
    /// drafts_at_k[k] = steps that DRAFTED position k (every step drafts all k < depth-1). accept@k
    /// = accepted_at_k[k] / drafts_at_k[k].
    pub fn bench_mtp_steps(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                     kv_stride: usize, depth: usize, max_new: usize)
                    -> (Vec<u32>, Vec<u32>, f32, f32, f32, usize, Vec<usize>, Vec<usize>) {
        let h = self.cfg.hidden_size;
        let plen = prompt.len();
        // B8 blocker B: the MTP draft/verify/re-prime step writes KV rows up to
        // plen + max_new + depth (plus the τ floor's +8 slop). A stride == plen + max_new OOBs the
        // MTP KV (caught downstream only by the mtp_draft_step assert, i.e. too late). Make the whole
        // class loud at bench entry instead of corrupting or panicking mid-loop.
        assert!(plen + max_new + depth + 8 <= kv_stride,
                "bench_mtp_steps: plen {} + max_new {} + depth {} + 8 exceeds kv_stride {} — \
                 raise the stride (see run_bench_tau's floor) or shrink the cell",
                plen, max_new, depth, kv_stride);
        let bufs = self.new_decode_buffers(3);
        let nkv = self.cfg.num_kv_heads; let hd = self.cfg.head_dim;
        let mut mtp_kc = self.dev.alloc_zeros::<half::bf16>(self.mtp_kc_elems(kv_stride)).unwrap();
        let mut mtp_vc = self.dev.alloc_zeros::<half::bf16>(self.mtp_kv_heads() * kv_stride * hd).unwrap();
        // alloc_zeros is cuMemAllocAsync — does NOT zero. The MTP attention must not read garbage
        // at unwritten KV positions, so zero both caches explicitly.
        self.dev.memset_zeros(&mut mtp_kc).unwrap();
        self.dev.memset_zeros(&mut mtp_vc).unwrap();
        self.dev.synchronize().unwrap();
        let mtp_kc_ptr = *mtp_kc.device_ptr();
        let mtp_vc_ptr = *mtp_vc.device_ptr();
        // S-B4 ROOT CAUSE FIX: the MTP hidden plumbing must be MW-wide, not h-wide. The
        // qwen4_exp MTP contract (mtp_forward_q4 + argmax_hidden's mtp_mix_for_head) consumes the
        // PRE-final-norm STREAM STACK [resid_width] — prefill_batch_range returns exactly that
        // ("residual ... the MTP head consumes the PRE-norm hidden") and verify's vout is "the
        // PRE-norm residual by contract (MTP head feed)". These scratch buffers were h-sized, so
        // every copy_col truncated each [rw] column to its first h elements (stream 0) and the
        // head then rmsnorm'd rw elements OUT OF BOUNDS — every draft was computed from garbage
        // (constant token 0 / rare-token clusters; k1 0-25% vs the 64-78% reference, k>=2 = 0%).
        let mw = self.mtp_hidden_width();
        let h_prev = self.dev.alloc_zeros::<half::bf16>(mw).unwrap();   // cursor hidden (seeds drafts)
        let h_scratch = self.dev.alloc_zeros::<half::bf16>(mw).unwrap();    // hidden-column extract scratch
        let h_save = self.dev.alloc_zeros::<half::bf16>(mw).unwrap();       // pre-verify hidden (re-prime k=0)
        let cur_hidden = self.dev.alloc_zeros::<half::bf16>(mw).unwrap();   // draft-chain cursor hidden

        // Prefill slot 0 (MTP) and slot 1 (sequential ground truth).
        self.zero_slot_state(state, 0, kv_stride);
        let (a0, hout0) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        self.dbg_probe_prefill("MTP", a0, &hout0, h, plen, state, kv_stride);   // --debug-hash
        self.zero_slot_state(state, 1, kv_stride);
        let (a1, _hout1) = self.prefill_batch(pool, prompt, state, 1, kv_stride, 0);
        assert_eq!(a0, a1, "prefill divergence slot0 vs slot1");

        // Helper: copy hidden column `col` of an [h, n] buffer into a [h] device buffer, on the
        // compute stream (stream-ordered with the kernels that produced/consume these hiddens).
        let copy_stream = self.stream.stream;
        let copy_col = |dst_ptr: u64, src_buf: &B, col: usize| unsafe {
            let src = *src_buf.device_ptr() as u64 + (col as u64) * (mw as u64) * 2;
            cudarc::driver::result::memcpy_dtod_async(dst_ptr, src, mw * 2, copy_stream).unwrap();
        };

        // Prompt-prime MTP over main positions 0..plen-2: step t uses (h_t, prompt[t+1]).
        // Same primitive the server uses -- see the note above.
        self.mtp_prime_prompt(pool, &hout0, &prompt[1..plen], mtp_kc_ptr, mtp_vc_ptr, kv_stride, 0);
        // --debug-hash: whole buffers — both were memset+sync'd, so unwritten rows are zeros.
        self.dbg_hash_bf16("MTP.mtpkc", &mtp_kc, nkv * kv_stride * hd);
        self.dbg_hash_bf16("MTP.mtpvc", &mtp_vc, nkv * kv_stride * hd);
        let mut mtp_pos = plen - 1;   // next main position for an MTP write
        // h_prev = h at plen-1 (last prompt position) — seeds the first draft.
        copy_col(*h_prev.device_ptr(), &hout0, plen - 1);
        self.dbg_hash_bf16("MTP.h_prev0", &h_prev, mw);   // --debug-hash
        pool.release_bf16(hout0, mw * plen);

        let mut mtp_tokens = vec![a0];      // a0 (prefill's first token) is already emitted
        let mut committed_tok = a0;         // token for the next position the verify will process
        let mut main_pos = plen;           // position committed_tok will be verified at
        let mtp_start = std::time::Instant::now();
        // The lossless gate decodes a sequential ground-truth token for every token MTP emits, in the
        // same loop. That work must NOT land in the MTP timer — it used to, which is why this probe
        // reported MTP at roughly half its real rate and then papered over it by *assigning* the MTP
        // duration to the sequential one (`speedup` was 1.00x by construction, on every run).
        let mut seq_dur = 0.0f32;
        let mut total_drafts = 0usize;
        let mut total_accepted = 0usize;
        let mut n_steps = 0usize;
        let mut accepted_at_k = vec![0usize; depth.saturating_sub(1)];
        let mut drafts_at_k = vec![0usize; depth.saturating_sub(1)];

        // Lockstep greedy ground-truth on slot 1: advanced in parallel with the MTP loop so the
        // emitted MTP token stream can be compared against clean sequential greedy (the lossless gate).
        let mut bufs2 = self.new_decode_buffers(2);
        self.dev.htod_sync_copy_into(&[1i32, 0], &mut bufs2.slot_ids_dev).unwrap();
        self.dev.synchronize().unwrap();
        let mut seq_tokens = vec![a0];
        let mut spos = plen;

        while mtp_tokens.len() < max_new {
            n_steps += 1;
            copy_col(*h_save.device_ptr(), &h_prev, 0);

            // ---- Draft chain: d_i from (prev_hidden, prev_token) at MTP positions mtp_pos.. ----
            // Uses persistent buffers + compute-stream copy_col (NOT CudaSlice::clone, which dtod-copies
            // on the device/NULL stream and races with the compute stream).
            let mut drafts: Vec<u32> = Vec::with_capacity(depth - 1);
            copy_col(*cur_hidden.device_ptr(), &h_prev, 0);
            let mut cur_tok = committed_tok as i32;
            let mut dpos = mtp_pos;
            for _ in 0..depth - 1 {
                let m = self.mtp_draft_step(pool, &cur_hidden, cur_tok, dpos,
                                            mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                if mtp_tokens.len() == 1 && drafts.is_empty() { self.dbg_hash_bf16("MTP.m0", &m, mw); }   // --debug-hash
                dpos += 1;
                copy_col(*cur_hidden.device_ptr(), &m, 0);
                pool.release_bf16(m, mw);
                cur_tok = self.argmax_hidden(pool, &cur_hidden) as i32;
                drafts.push(cur_tok as u32);
            if crate::opts::var(crate::opt!("mtp-garbage-draft")).is_ok() {
                for d in drafts.iter_mut() { *d = (*d).wrapping_add(1); }
            }
                total_drafts += 1;
                drafts_at_k[drafts.len() - 1] += 1;
            }

            // ---- Verify [committed_tok, drafts...] with ping-pong GDN checkpoint. ----
            // The GDN kernels snapshot S1 (recurrent state after the always-accepted committed token)
            // into slot 2. On draft rejection we restore S1 with a dtod copy — NO second model
            // forward. vout's column nacc already holds the hidden at the last accepted position in
            // BOTH accept/reject cases (the old reverify's hidden was identical), so it is reused.
            let mut verify_input = vec![committed_tok];
            verify_input.extend(drafts.iter().copied());
            let mut f9_div_count: Option<usize> = None;
            let (preds, vout) = self.verify_forward(pool, &verify_input, state, 0, kv_stride, main_pos, Some(2), None);
            let (vhs, vcall): (Vec<u64>, usize) = if f9_xhash_active() {
                match F9XHASH.lock().unwrap().last().filter(|(l, _, _, _)| *l == 1u8) {
                    Some((_, c, _, hs)) => (hs.clone(), *c),
                    None => (Vec::new(), usize::MAX),
                }
            } else { (Vec::new(), usize::MAX) };

            // ---- Accept longest prefix (committed_tok already emitted, don't re-emit). ----
            let mut nacc = 0usize;
            while nacc < drafts.len() && preds[nacc] == drafts[nacc] { nacc += 1; }
            if crate::opts::var(crate::opt!("accept-debug")).is_ok() && (n_steps <= 4 || nacc < drafts.len()) {
                eprintln!("  [accept-dbg] step {n_steps} emitted_so_far={} committed={committed_tok} drafts={drafts:?} preds[0..3]={:?} nacc={nacc}",
                          mtp_tokens.len(), &preds[..preds.len().min(3)]);
            }
            let bonus = preds[nacc];
            let emitted_count_before = mtp_tokens.len();
            for &d in drafts.iter().take(nacc) { mtp_tokens.push(d); }
            if mtp_tokens.len() < max_new { mtp_tokens.push(bonus); }
            total_accepted += nacc;
            for a in accepted_at_k.iter_mut().take(nacc) { *a += 1; }

            // ---- GDN rollback on partial accept: restore the state as of the LAST ACCEPTED column.
            // Checkpoint slots are contiguous from slot 2: slot (2 + t) holds verify column t's
            // post-state. Restoring slot 2 unconditionally (the old behaviour) is only correct at
            // depth 2 and silently corrupted the recurrent state at any deeper setting.
            if nacc + 1 != depth {
                self.copy_gdn_slot(state, 2 + nacc, 0);
            }
            // h_prev = hidden at the last accepted position (vout column nacc).
            copy_col(*h_prev.device_ptr(), &vout, nacc);

            // ---- Re-prime MTP over the accepted prefix with REAL hiddens (vout) so its KV stays
            // ---- consistent. k=0 uses h_save; k>=1 uses vout col k-1.
            for k in 0..=nacc {
                if k == 0 {
                    let m = self.mtp_draft_step(pool, &h_save, committed_tok as i32, main_pos - 1,
                                                mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                    pool.release_bf16(m, mw);
                } else {
                    copy_col(*h_scratch.device_ptr(), &vout, k - 1);
                    let m = self.mtp_draft_step(pool, &h_scratch, drafts[k - 1] as i32, main_pos - 1 + k,
                                                mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                    pool.release_bf16(m, mw);
                }
            }
            pool.release_bf16(vout, mw * depth);

            // ---- Lockstep greedy check on slot 1: decode one token per emitted MTP token. ----
            // Timed separately: this IS the sequential baseline, and it is not part of the MTP step.
            let seq_t0 = std::time::Instant::now();
            for (li, &_emt) in mtp_tokens[emitted_count_before..].iter().enumerate() {
                if crate::opts::var(crate::opt!("accept-debug")).is_ok() && li == 0 && n_steps > 0 {
                    // (debug aid) nothing yet — mismatch checked after the decode below
                }
                let tok = *seq_tokens.last().unwrap() as i32;
                self.dev.htod_sync_copy_into(&vec![tok, 0], &mut bufs2.tokens_dev).unwrap();
                self.dev.htod_sync_copy_into(&vec![spos as i32, 0], &mut bufs2.pos_dev).unwrap();
                self.dev.synchronize().unwrap();
                let next = self.forward_decode(pool, &mut bufs2, state, kv_stride, spos + 1, 1);
                if f9_xhash_active() && !vhs.is_empty() {
                    let dcall_now = F9X_CUR.with(|c| c.get());
                    let dh = F9XHASH.lock().unwrap().last().filter(|(l, c, _, _)| *l == 0u8 && *c == dcall_now).map(|(_, _, _, hs)| hs[0]);
                    let p_abs = mtp_tokens.len() + li;
                    match (vhs.get(li), dh) {
                        (Some(v), Some(d)) if v != &d => {
                            eprintln!("  [f9x] hidden() DIVERGE at abs pos {p_abs} (step {n_steps} col {li}): verify 0x{v:016x} decode 0x{d:016x}");
                            *f9_div_count.get_or_insert_with(|| 0usize) += 1;
                            if f9_div_count.unwrap() == 1
                               && crate::opts::var(crate::opt!("f9-xhash")).as_deref() == Ok("2") {
                                let ring = F9XHASH.lock().unwrap();
                                let dcall_now = F9X_CUR.with(|c| c.get());
                                // call-id-tagged windows (robust to finals interleaving).
                                {
                                    let dec: Vec<(usize, &Vec<u64>)> = ring.iter()
                                        .filter(|(l, c, _, _)| *l == 10u8 && *c == dcall_now).map(|(_, _, li2, hs)| (*li2, hs)).collect();
                                    let ver: Vec<(usize, &Vec<u64>)> = ring.iter()
                                        .filter(|(l, c, _, _)| *l == 11u8 && *c == vcall).map(|(_, _, li2, hs)| (*li2, hs)).collect();
                                    eprintln!("  [f9x-L] windows: dec={} ver={} layers (dcall {dcall_now} vcall {vcall})", dec.len(), ver.len());
                                    let dm: std::collections::BTreeMap<usize, &Vec<u64>> = dec.into_iter().collect();
                                    let mut printed = 0;
                                    for (lix, vh) in &ver {
                                        if let Some(dh) = dm.get(lix) {
                                            let dv = vh.get(li).copied().unwrap_or(0) != dh[0];
                                            let c0v = vh.first().copied().unwrap_or(0) != dh[0];
                                            if (dv || c0v) && printed < 6 {
                                                let cls = if lix % 4 == 3 {"FA "} else {"GDN"};
                                                eprintln!("  [f9x-L] layer divergence L{lix} ({cls}): col{li} {} col0 {}",
                                                    if dv {"DIFF"} else {"same"}, if c0v {"DIFF"} else {"same"});
                                                printed += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                seq_tokens.push(next[0]);
                spos += 1;
            }
            seq_dur += seq_t0.elapsed().as_secs_f32();

            mtp_pos = main_pos + nacc;
            main_pos += nacc + 1;
            committed_tok = bonus;
            if mtp_tokens.len() >= max_new { break; }
        }
        // MTP time = wall time minus the interleaved sequential ground-truth decode.
        let mtp_dur = (mtp_start.elapsed().as_secs_f32() - seq_dur).max(1e-6);
        drop(h_prev); drop(h_save); drop(h_scratch); drop(cur_hidden); drop(mtp_kc); drop(mtp_vc); drop(bufs);
        let seq_dur = seq_dur.max(1e-6);
        drop(bufs2);

        let mtp_tok_s = if mtp_dur > 0.0 { mtp_tokens.len() as f32 / mtp_dur } else { 0.0 };
        let seq_tok_s = if seq_dur > 0.0 { seq_tokens.len() as f32 / seq_dur } else { 0.0 };
        let accept_rate = if total_drafts > 0 { total_accepted as f32 / total_drafts as f32 } else { 0.0 };
        // PROBE 0 output: the two numbers the --profile-mtp phase table cannot give us. That table's
        // "modelled step" is the DEPTH-1 model (verify N=2, ONE draft), so the real depth-d step cost
        // and tokens/step have to be counted, not derived.
        if n_steps > 0 {
            eprintln!("[probe0] depth {depth}: {n_steps} steps, {} tokens => {:.3} tokens/step, \
                       {:.2} ms/step  |  drafts {total_drafts}, accepted {total_accepted} ({:.1}%)",
                      mtp_tokens.len(), mtp_tokens.len() as f32 / n_steps as f32,
                      mtp_dur * 1000.0 / n_steps as f32, accept_rate * 100.0);
        }
        let n = mtp_tokens.len().min(seq_tokens.len());
        (mtp_tokens[..n].to_vec(), seq_tokens[..n].to_vec(), mtp_tok_s, seq_tok_s, accept_rate, n_steps, accepted_at_k, drafts_at_k)
    }

    /// Batched benchmark: M identical prompts prefilled + decoded together.
    /// Returns (tokens_of_slot0, aggregate_decode_tok_s). Tokens must match single-stream.
    #[allow(non_snake_case)]
    pub fn bench_batch(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                       M: usize, max_new: usize, max_seq_len: usize) -> (Vec<u32>, f32) {
        let v = self.cfg.vocab_size;
        let plen = prompt.len();
        let mut bufs = self.new_decode_buffers(M);
        // ---- prefill (uniform batched; all M slots identical) ----
        let mut last = vec![0u32; M];
        for (t, &tok) in prompt.iter().enumerate() {
            let hidden = self.embed_batch(&vec![tok; M]);
            let out = self.forward_batch(pool, hidden, &vec![t; M], state, max_seq_len, M);
            if t == plen - 1 {
                let logits = self.logits_batch(pool, &out, M);
                let block = 1024u32;
                blaunch!(self, "argmax_b", (M as u32,1,1), (block,1,1), (block*8),
                    (*bufs.token_ids_dev.device_ptr() as u64, d(&logits), v as i32, M as i32));
                self.sync_stream();
                last = self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap()
                    .into_iter().take(M).map(|x| x as u32).collect();
                pool.release_bf16(logits, v * M);
            }
            pool.release_bf16(out, self.cfg.hidden_size * M); // release to pool instead of freeing
        }
        let mut out_tokens = vec![last[0]];
        // ---- decode (optimized path: pre-computed rope, device-side embed/argmax) ----
        let start = std::time::Instant::now();
        for step in 0..max_new {
            let pos_val = plen + step;
            let max_pc = pos_val + 1;
            let toks_i32: Vec<i32> = last.iter().map(|&t| t as i32).collect();
            let pos_i32: Vec<i32> = vec![pos_val as i32; M];
            self.dev.htod_sync_copy_into(&toks_i32, &mut bufs.tokens_dev).unwrap();
            self.dev.htod_sync_copy_into(&pos_i32, &mut bufs.pos_dev).unwrap();
            self.dev.synchronize().unwrap(); // htod on NULL must complete before compute stream
            let next = self.forward_decode(pool, &mut bufs, state, max_seq_len, max_pc, M);
            last.copy_from_slice(&next);
            out_tokens.push(last[0]);
        }
        let dur = start.elapsed().as_secs_f32();
        (out_tokens, (M * max_new) as f32 / dur)
    }
}
