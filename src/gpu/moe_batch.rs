//! `impl GpuModel`: `moe_batch`, the trunk's batched MoE forward with its TP fused-reduce arms (one function, kept whole:
//! a single item over the file budget). Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// MoE FFN forward (qwen3_5_moe), correctness-first bf16 path.
    ///   router → softmax→top-k→renorm → Σ_j w_j·expert_{e_j}(x) + sigmoid(shared_gate·x)·shared(x).
    /// Experts are the STACKED fused weights read directly by `moe_experts_b`. Not yet fused-dequant or
    /// tuned — this is the oracle-correct reference the NVFP4 grouped kernel is validated against.
    /// AR landing 2: with `fuse` set (the FFN epilogue runs `tp_reduce_resnorm_b`), the arms whose
    /// FINAL reduce is the last producer of `out` (the fold_shared combines and the fp32_shared
    /// tail) stop after K1 and hand the LOCAL partial up; the arms that add the replicated shared
    /// contribution POST-reduce keep today's K1+K2 chain (the shared add cannot move pre-reduce
    /// without changing the rounding placement, so they are not bit-exactly fusable).
    pub(super) fn moe_batch(&self, pool: &mut Pool, x: &B, moe: &GpuMoe, batch: usize, fuse: bool) -> ReduceOut {
        let cfg = &self.cfg;
        let (h, ne, k) = (cfg.hidden_size, cfg.num_experts, cfg.num_experts_per_tok);
        let (mi, si) = (cfg.moe_intermediate_size, cfg.shared_expert_intermediate_size);

        // TP=2 expert-parallel (only when THIS layer's experts were halved at attach — a property of
        // the tensor, NOT of the process; the MTP MoE stays replicated and barrier-free). The router
        // is replicated, so `moe_router_topk_b` still emits GLOBAL ids; the expert GEMMs take this
        // rank's band [e_base, e_base+e_span) and contribute EXACT ZEROS for remote experts, and one
        // all-reduce on the expert-combined output (below, before the shared add) restores the sum.
        let (e_base, e_span) = if moe.experts_sharded {
            let span = (ne / self.experts_shard_factor().max(1)) as i32;
            (self.tp_rank.max(0) as i32 * span, span)
        } else {
            (0, ne as i32)
        };

        // 1. Router logits [ne, batch] → top-k expert ids + combine weights. The kernel is the SINGLE
        // routing decision shared by every expert-GEMM path below (bf16 and NVFP4), so the variants
        // cannot drift. qwen3_5_moe: softmax→top-k→renorm. hy_v3: sigmoid scores (fp32), top-k
        // selected by score+expert_bias, weights = UN-biased sigmoid scores renormalized × scaling.
        // Router scratch: reuse the preallocated buffers for graph-CAPTURED small batches (decode/verify,
        // batch<=MAX_VERIFY) — a fresh alloc here is illegal during CUDA-graph capture. Prefill
        // (batch>MAX_VERIFY) is not captured, so a fresh alloc is fine and keeps the scratch bounded.
        let owned = (batch > MAX_VERIFY).then(||
            (self.dev.alloc_zeros::<i32>(k*batch).unwrap(), self.dev.alloc_zeros::<f32>(k*batch).unwrap()));
        let (ids_ptr, wts_ptr) = match &owned {
            Some((i, w)) => (*i.device_ptr() as u64, *w.device_ptr() as u64),
            None => (*self.moe_ids.device_ptr() as u64, *self.moe_wts.device_ptr() as u64),
        };
        if let Some(bias) = &moe.expert_bias {
            if let W::Bf16(rw) = &moe.router {                // hy_v3: the reference computes router logits in FP32 (hidden.float() @ gate.float())
                // and sigmoid-selects on them. Rounding the logits to bf16 first (the qwen softmax
                // path's contract) perturbs Hy3's tightly-clustered sigmoid scores by ~1e-3 and
                // flips near-tie top-8 selections — an O(1) output change per flipped token. Keep
                // them fp32: gemm_binv_f32_b at serving widths (the batch-invariant twin of
                // gemm_binv_b), a cuBLAS fp32-accumulate F32-out GEMM at prefill widths (outside
                // the invariance contract, like every prefill GEMM).
                let mut logits = pool.get(ne * batch);
                if batch <= MAX_VERIFY {
                    let smem = (batch * 256 * 4) as u32;
                    blaunch!(self, "gemm_binv_f32_b", (ne as u32,1,1), (256,1,1), smem,
                        (d(&logits), d(rw), d(x), ne as i32, h as i32, batch as i32));
                } else {
                    use cudarc::cublas::sys::{cudaDataType, cublasComputeType_t, cublasGemmAlgo_t};
                    let (one, zero) = (1.0f32, 0.0f32);
                    unsafe {
                        cudarc::cublas::result::gemm_ex(*self.blas.handle(),
                            OP::CUBLAS_OP_T, OP::CUBLAS_OP_N,
                            ne as i32, batch as i32, h as i32,
                            &one as *const f32 as *const _,
                            *rw.device_ptr() as *const _, cudaDataType::CUDA_R_16BF, h as i32,
                            d(x) as *const _, cudaDataType::CUDA_R_16BF, h as i32,
                            &zero as *const f32 as *const _,
                            d(&logits) as *mut _, cudaDataType::CUDA_R_32F, ne as i32,
                            cublasComputeType_t::CUBLAS_COMPUTE_32F,
                            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT).expect("hy3 router gemm (f32 out)");
                    }
                }
                blaunch!(self, "moe_router_topk_sigmoid_f32_b", (batch as u32,1,1), (256,1,1), (ne*4) as u32,
                    (ids_ptr, wts_ptr, d(&logits), d(bias), ne as i32, k as i32, batch as i32,
                     cfg.route_norm as i32, fbits(cfg.router_scaling)));
                pool.release(logits, ne * batch);
            } else {
                // Quantized router (NOT the hy3 recipe): the legacy bf16-logits path.
                let mut logits = pool.get_bf16(ne * batch);
                self.gemm_act_binv(&moe.router, x, &mut logits, h, ne, batch);
                blaunch!(self, "moe_router_topk_sigmoid_b", (batch as u32,1,1), (256,1,1), (ne*4) as u32,
                    (ids_ptr, wts_ptr, d(&logits), d(bias), ne as i32, k as i32, batch as i32,
                     cfg.route_norm as i32, fbits(cfg.router_scaling)));
                pool.release_bf16(logits, ne * batch);
            }
        } else {
            let mut logits = pool.get_bf16(ne * batch);
            self.gemm_act_binv(&moe.router, x, &mut logits, h, ne, batch);
            if crate::opts::var(crate::opt!("dump-pfhash")).is_ok() {
                // Phase-8 [rtap]: first wide MoE routing per prefill — token 0's top-k ids/weights
                // and its logits, hashed — warm-vs-cold router decision probe.
                static RTAP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                let n = RTAP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 4 {
                    self.sync_stream();
                    let lg = self.dev.dtoh_sync_copy(&logits).unwrap_or_default();
                    let mut f64h = 0u64;
                    for v in lg.iter().take(ne) { f64h ^= (v.to_bits() as u64).wrapping_mul(0x100000001B3); }
                    eprintln!("[rtap#{n}] batch={batch} tok0 logits[:8]={:?} xor={:012x}",
                              &lg.iter().take(8).map(|v| v.to_f32()).collect::<Vec<_>>(), f64h & 0xFFFFFFFFFFFF);
                }
            }
            blaunch!(self, "moe_router_topk_b", (batch as u32,1,1), (256,1,1), (ne*4) as u32,
                (ids_ptr, wts_ptr, d(&logits), ne as i32, k as i32, batch as i32));
            pool.release_bf16(logits, ne * batch);
        }
        self.gptq_profile_moe(ids_ptr, batch, k, ne);

        // 2. Grouped expert MLP over the stacked fused weights → out [h, batch].
        // --gptq: per-expert Hessians of the bf16 experts' inputs (a no-op unless armed).
        if let (W::Bf16(gu), W::Bf16(_)) = (&moe.gate_up, &moe.down) { self.gptq_tap_moe(x, ids_ptr, batch, gu, h, mi, k); }
        let mut out = pool.get_bf16(h * batch);
        // AR landing 2: set by the arms whose final reduce stopped after K1 (the caller then
        // launches tp_reduce_resnorm_b instead of the full-rank norm kernel).
        let mut fused_out: Option<FusedPartial> = None;
        let smem = ((2*mi + mi + h) * 4) as u32;    // gu[2I] + hh[I] + acc[H]
        // E12: set in the NVFP4 arm below (the bf16 arm and the non-fold paths leave it false); the
        // shared-expert section at the end reads it to skip the old separate-launch shared MLP.
        let mut fold = false;
        match (&moe.gate_up, &moe.down) {
            (W::Bf16(gu), W::Bf16(dn)) => {
                let skip = self.gptq_tap.lock().unwrap().as_ref().map_or(false, |t| t.skip_experts);
                if skip {
                    self.dev.memset_zeros(&mut out).unwrap();
                } else {
                blaunch!(self, "moe_experts_b", (batch as u32,1,1), (256,1,1), smem,
                    (d(&out), d(x), ids_ptr, wts_ptr, d(gu), d(dn),
                     h as i32, mi as i32, k as i32, batch as i32));
                }
            }
            (W::Nvfp4 { qweight: guq, scales: gus, gs: gugs, .. },
             W::Nvfp4 { qweight: dnq, scales: dns, gs: dngs, .. }) => {
              // E12 — fold the SHARED expert into the grouped MoE launches as ONE extra slot (slot
              // index k of every token; one launch sequence per layer instead of grouped + a separate
              // small-M shared pair). Numerics: the shared slot's per-element math is the SAME mma
              // chain as gemm_mma_fp4_b (the old gemm_act dispatch), the combine keeps the old
              // two-stage rounding (routed sum → bf16 round → shared add → bf16 round), so the fold
              // is bit-identical to the separate-launch path at the bf16 output level. E8: under
              // the shard the shared slot's gu M is rank-local (2·si_local, from desc[8]) and the
              // silu writes the rank's half of the act row (moe_silu_fold_b) — the partial then
              // joins the routed fp32 chain exactly like the replicated slot. Conditions:
              // (1) --moe-no-fold=1 disables (the escape / A-B); (2) si == mi so the shared slot
              // fits the launch's act width (hy_v3: si = mi·num_shared_experts = 1536; qwen's
              // si ≠ mi falls back); (3) the shared gate_up is the FUSED NVFP4 tensor (hy_v3
              // quantized); (4) the MoE runs the bf16-chain mma kernels (mxfp4-native MoE keeps
              // the old path).
              let fold_shared = moe.experts_sharded && moe.shared_sharded;
              // E8 FP32-partial rule: at the decode/verify widths (batch <= MAX_VERIFY — the
              // binv/LOSSLESS/acceptance regime) the sharded shared PARTIAL must stay UNROUNDED
              // until the final reduce — the fold's combine rounds per-rank AFTER the shared add
              // (a bf16 partial round: the e7950d7 acceptance disease), so the fold's shared slot
              // is DISABLED there and the shared contribution flows through the dense row-parallel
              // pattern (gemm_act_f32 + fp32 add + tp_all_reduce_fp32, single rounding boundary,
              // in the tail below). Wide prefill keeps the fold's bf16 shared slot — prefill
              // reassociation is accepted (the dense path's rule; gemm_act_f32 caps at
              // batch <= MAX_VERIFY). The REPLICATED shared (escape / TP=1) always folds.
              let fp32_shared = fold_shared && batch <= MAX_VERIFY
                  && self.mxfp4.as_ref().map_or(true, |st| {
                      // The SHARED down must run the bf16-chain (gemm_act_f32 has no native arm);
                      // a native shared falls back to the bf16 fold_shared flow below.
                      let ab = st.allow_bf16.read().unwrap();
                      ab.contains(&moe.shared.down.packed_ptr())
                  });
              fold = Self::moe_fold_on()
                  && si == mi
                  && matches!(moe.shared_gate_up, Some(W::Nvfp4 { .. }))
                  && moe.shared_slot_desc.is_some()
                  && !fp32_shared
                  && !self.mxfp4.as_ref().map_or(false, |st| {
                      let ab = st.allow_bf16.read().unwrap();
                      !ab.contains(&(*guq.device_ptr() as u64)) || !ab.contains(&(*dnq.device_ptr() as u64))
                  });
              let gated = moe.shared_gate.is_some();   // hy_v3: None → slot weight exactly 1.0
              // The shared slot's weight geometry lives in the per-layer descriptor (written at
              // load): [0..4) = gate_up { qweight, scales, gs, (SK<<32)|sk_off } (SK = h, sk_off =
              // 0 — the col-shard splits M, not K), [4..8) = down { qweight, scales, gs,
              // (sdn_k<<32)|rank·sdn_k }, [8] = the shared slot's gu M (2·si, or 2·si_local = si
              // under the shard), [12] = the down's M (h, uniform no-op).
              let sdesc_ptr = moe.shared_slot_desc.as_ref()
                  .map(|d| *d.device_ptr() as u64).unwrap_or(0);
              // The sigmoid gate logits (qwen-style gated shared expert). Computed with the SAME
              // gemm_act_binv as the old path, so the slot weight sigmoid is bit-identical. hy_v3 is
              // ungated → None.
              let mut sgate = if fold && gated {
                  let mut sg = pool.get_bf16(batch);
                  self.gemm_act_binv(moe.shared_gate.as_ref().unwrap(), x, &mut sg, h, 1, batch);
                  Some(sg)
              } else { None };
              let sgate_ptr = sgate.as_ref().map(|s| d(s)).unwrap_or(0);
              // Token-gather wins only when tokens/expert > 1 — for the verify regime that used to mean
              // "never": the grouped path's per-layer dtoh sync (host reads poff[ne] to size the grid)
              // dominated a tiny batch. E16 removes the sync: static upper-bound buffers + the grouped
              // GEMM early-exiting on poff[ne] read ON DEVICE. Verify (2..=MAX_VERIFY) now reads each
              // DISTINCT expert's weights once instead of once per (token,slot) — at Hy3's 192 experts
              // top-8 an N=8 verify has ~64 pairs over ~40-60 distinct experts ≈ 1.3-1.6× fewer bytes.
              // The grouped-vs-GEMV crossover is measured via --moe-grouped-min (default 2).
              if batch >= Self::moe_grouped_min() && batch <= MAX_VERIFY {
                // E16 — DEVICE-ONLY grouped MoE for the verify regime. No fresh allocs (scratch is
                // preallocated), no dtoh, no htod — graph-capturable end to end. Bit-identity vs the
                // N=1 path holds per output element: same MMA k-order per element, columns independent,
                // combine in slot order fp32; the scatter permutation moves rows, never values.
                let ne = cfg.num_experts;
                let p = batch * k;
                if fold {
                    // E12 — the shared expert rides the SAME grouped sequence as one extra slot: the
                    // metadata pipeline appends its region ([poff[ne], poff[ne+1])) after the routed
                    // band, the fold GEMMs compute it from the shared MLP's own tensors, and the fold
                    // combine applies the old two-stage rounding (routed fp32 sum → bf16 round →
                    // shared add → bf16 round). Per-element math is bit-identical to the separate
                    // gemm_act launches; the combine's slot order is unchanged (routed slots first).
                    let g = &self.moe_g;
                    let ppad_max = g.ppad_max;
                    self.memset_compute_stream(*g.count.device_ptr() as u64, ne * 4);
                    blaunch!(self, "moe_count_b", grid(p), (256,1,1), 0,
                        (*g.count.device_ptr() as u64, ids_ptr, p as i32, e_base, e_base + e_span));
                    blaunch!(self, "moe_offsets_fold_b", (1u32,1,1), (32,1,1), 0,
                        (*g.poff.device_ptr() as u64, *g.cursor.device_ptr() as u64, *g.count.device_ptr() as u64,
                         ne as i32, batch as i32));
                    self.memset_compute_stream_u32(*g.perm_tok.device_ptr() as u64, 0xFFFF_FFFF, ppad_max);
                    blaunch!(self, "moe_scatter_b", grid(p), (256,1,1), 0,
                        (*g.perm_tok.device_ptr() as u64, *g.perm_wt.device_ptr() as u64, *g.inv_pos.device_ptr() as u64,
                         *g.cursor.device_ptr() as u64, ids_ptr, wts_ptr, p as i32, k as i32,
                         e_base, e_base + e_span));
                    // The shared pairs (slot k of every token) land in the dedicated region in fixed
                    // token order — position of token b is poff[ne]+b, so the combine needs no
                    // per-pair bookkeeping and the fp32 add order is deterministic.
                    blaunch!(self, "moe_shared_scatter_b", grid(batch), (256,1,1), 0,
                        (*g.perm_tok.device_ptr() as u64, *g.perm_wt.device_ptr() as u64, *g.poff.device_ptr() as u64,
                         sgate_ptr, ne as i32, batch as i32, gated as i32));
                    blaunch!(self, "moe_tilemap_fold_b", grid(ne + 1), (256,1,1), 0,
                        (*g.tile_e.device_ptr() as u64, *g.poff.device_ptr() as u64, ne as i32));
                    let x_perm = pool.get_bf16(ppad_max * h);
                    blaunch!(self, "moe_gather_x_b", grid(ppad_max * h), (256,1,1), 0,
                        (d(&x_perm), d(x), *g.perm_tok.device_ptr() as u64, h as i32, ppad_max as i32,
                         *g.poff.device_ptr() as u64, (ne + 1) as i32));
                    if self.rotated.contains(&(*guq.device_ptr() as u64)) { self.rot_inplace(&x_perm, ppad_max * h); }
                    let ngroups_max = (ppad_max / 16) as u32;
                    let gu_p = pool.get_bf16(ppad_max * 2 * mi);
                    // ne+1: the fold GEMM's early-exit bound reads poff[ne+1] (routed + shared total).
                    // Shared gate_up: K = h (same as routed — the col-shard splits M, not K),
                    // sk_off = 0; the shared slot's gu M is desc[8] (2·si, or si under the shard).
                    blaunch!(self, "gemm_moe_grouped_mma_fp4_fold", (((2*mi/16) as u32), ngroups_max, 1), (256,1,1), 0,
                        (d(&gu_p), *guq.device_ptr() as u64, *gus.device_ptr() as u64, d(gugs), d(&x_perm),
                         *g.tile_e.device_ptr() as u64, (2*mi) as i32, h as i32, e_base,
                         *g.poff.device_ptr() as u64, (ne + 1) as i32,
                         sdesc_ptr));
                    let h_p = pool.get_bf16(ppad_max * mi);
                    // E8: fold-aware silu — the shared region's rows write the rank's half of the
                    // act row (moe_silu_fold_b; the routed rows are identical to moe_silu_bf16_b).
                    blaunch!(self, "moe_silu_fold_b", grid(ppad_max * mi), (256,1,1), 0,
                        (d(&h_p), d(&gu_p), mi as i32, ppad_max as i32,
                         *g.poff.device_ptr() as u64, ne as i32, sdesc_ptr + 32));
                    let dn_p = pool.get_bf16(ppad_max * h);
                    // Shared down: K = sdn_k (si, or si/2 under the E8 row-parallel shard), X k-offset
                    // sk_off = the rank's half of the act (bit-identical to the old act_half slice).
                    blaunch!(self, "gemm_moe_grouped_mma_fp4_fold", (((h/16) as u32), ngroups_max, 1), (256,1,1), 0,
                        (d(&dn_p), *dnq.device_ptr() as u64, *dns.device_ptr() as u64, d(dngs), d(&h_p),
                         *g.tile_e.device_ptr() as u64, h as i32, mi as i32, e_base,
                         *g.poff.device_ptr() as u64, (ne + 1) as i32,
                         sdesc_ptr + 32));
                    // fold_shared: the shared PARTIAL is combined pre-reduce (ONE reduce restores both
                    // sums, exactly the E8 fold's math). !fold_shared: routed-only combine, reduce the
                    // routed partial, then add the replicated shared slot post-reduce (the old
                    // add_residual_b / moe_shared_combine_b semantics).
                    blaunch!(self, "moe_combine_grouped_fold_b", grid(batch * h), (256,1,1), 0,
                        (d(&out), d(&dn_p), *g.perm_wt.device_ptr() as u64, *g.inv_pos.device_ptr() as u64,
                         *g.poff.device_ptr() as u64, h as i32, k as i32, batch as i32, ne as i32,
                         fold_shared as i32));
                    if !fold_shared && moe.experts_sharded { self.tp_all_reduce_bf16(&mut out, h * batch); }
                    if !fold_shared {
                        // The shared slot's down rows are [poff[ne], poff[ne]+batch) of dn_p.
                        blaunch!(self, "moe_shared_slot_combine_b", grid(batch * h), (256,1,1), 0,
                            (d(&out), d(&dn_p), *g.poff.device_ptr() as u64, sgate_ptr,
                             h as i32, 0i32, 0i32, batch as i32, ne as i32, 1i32, gated as i32));
                    }
                    if fold_shared {
                        if fuse {
                            if let Some(fl) = self.tp_all_reduce_bf16_k1(&mut out, h * batch) {
                                fused_out = Some(match fl {
                                    FusedLocal::F32Scratch => FusedPartial::F32Scratch,
                                    _ => FusedPartial::Bf16,
                                });
                            }
                        } else {
                            self.tp_all_reduce_bf16(&mut out, h * batch);
                        }
                    }
                    pool.release_bf16(x_perm, ppad_max*h); pool.release_bf16(gu_p, ppad_max*2*mi);
                    pool.release_bf16(h_p, ppad_max*mi); pool.release_bf16(dn_p, ppad_max*h);
                } else {
                let g = &self.moe_g;
                let ppad_max = g.ppad_max;
                self.memset_compute_stream(*g.count.device_ptr() as u64, ne * 4);
                blaunch!(self, "moe_count_b", grid(p), (256,1,1), 0,
                    (*g.count.device_ptr() as u64, ids_ptr, p as i32, e_base, e_base + e_span));
                blaunch!(self, "moe_offsets_b", (1u32,1,1), (32,1,1), 0,
                    (*g.poff.device_ptr() as u64, *g.cursor.device_ptr() as u64, *g.count.device_ptr() as u64, ne as i32));
                self.memset_compute_stream_u32(*g.perm_tok.device_ptr() as u64, 0xFFFF_FFFF, ppad_max);
                blaunch!(self, "moe_scatter_b", grid(p), (256,1,1), 0,
                    (*g.perm_tok.device_ptr() as u64, *g.perm_wt.device_ptr() as u64, *g.inv_pos.device_ptr() as u64,
                     *g.cursor.device_ptr() as u64, ids_ptr, wts_ptr, p as i32, k as i32,
                     e_base, e_base + e_span));
                blaunch!(self, "moe_tilemap_b", grid(ne), (256,1,1), 0,
                    (*g.tile_e.device_ptr() as u64, *g.poff.device_ptr() as u64, ne as i32));
                let x_perm = pool.get_bf16(ppad_max * h);
                blaunch!(self, "moe_gather_x_b", grid(ppad_max * h), (256,1,1), 0,
                    (d(&x_perm), d(x), *g.perm_tok.device_ptr() as u64, h as i32, ppad_max as i32,
                     *g.poff.device_ptr() as u64, ne as i32));
                let ngroups_max = (ppad_max / 16) as u32;
                let gu_p = pool.get_bf16(ppad_max * 2 * mi);
                if let Some(st) = &self.mxfp4 {
                    let gu_ptr = *guq.device_ptr() as u64;
                    // MXFP4-native grouped MoE: quantize x_perm (two 8-token OMMA groups per
                    // 16-token expert group, one launch) then the grouped OMMA GEMV. Same
                    // capture-safe discipline as the bf16 arm (preallocated scratch, no allocs,
                    // device-side bound on poff[ne]). Allowlisted (MTP head) runs the bf16 chain.
                    // 2026-08-17: native grouped OMMA LOSES at prefill widths (--mxfp4=1 alone
                    // cost 3-9% TP prefill vs control — A/B #2 of the B2 gate) — the dequant+mma
                    // bf16 grouped kernel wins there. Route wide batches to bf16; escape knob
                    // --moe-native-pf=1 forces the old behavior.
                    let native_wide = Self::moe_native_pf_on();
                    if native_wide
                        && !st.allow_bf16.read().unwrap().contains(&gu_ptr) {
                        let (gu_a, gu_s2) = st.w.get(&gu_ptr).unwrap_or_else(|| {
                            panic!("mxfp4: MoE gate_up @{gu_ptr:x} has no OMMA repack") }).ptrs();
                        let nks_gu = (h / 64) as i32;
                        if Self::mxfp4_fused_on() {
                            // FUSED (F3, XSILU=0): two 8-token group passes per kstep in-block,
                            // ONE launch; the poff[ne] device-side early-exit stays the FIRST
                            // kernel statement (the F12 padded-grid kill).
                            unsafe {
                                st.moe_grouped_fused0.clone().launch_on_stream(&self.stream,
                                    LaunchConfig { grid_dim: ((2*mi/16) as u32, ngroups_max, 1), block_dim: (256,1,1), shared_mem_bytes: 0 },
                                    (d(&gu_p), gu_a, gu_s2, d(gugs), d(&x_perm),
                                     (2*mi/16) as i32, nks_gu, h as i32, e_base,
                                     *g.tile_e.device_ptr() as u64, *g.poff.device_ptr() as u64, ne as i32))
                                    .unwrap_or_else(|e| panic!("mxfp4 moe_grouped_fused (gu): {e:?}"));
                            }
                        } else {
                            let ng_quant = (ppad_max / 8) as u32;
                            unsafe {
                                st.quant_ng.clone().launch_on_stream(&self.stream,
                                    LaunchConfig { grid_dim: (ng_quant * nks_gu as u32, 1, 1), block_dim: (256,1,1), shared_mem_bytes: 0 },
                                    (d(&x_perm), h as i32, ng_quant as i32, nks_gu, 8i32, 8i32,
                                     d(&st.moe_bp), d(&st.moe_sfb),
                                     *g.poff.device_ptr() as u64, ne as i32)).unwrap_or_else(|e| panic!("mxfp4 quant_ng: {e:?}"));
                                st.moe_gemv.clone().launch_on_stream(&self.stream,
                                    LaunchConfig { grid_dim: ((2*mi/16) as u32, ngroups_max, 1), block_dim: (256,1,1), shared_mem_bytes: 0 },
                                    (d(&gu_p), gu_a, gu_s2, d(gugs), d(&st.moe_bp), d(&st.moe_sfb),
                                     (2*mi/16) as i32, nks_gu, e_base,
                                     *g.tile_e.device_ptr() as u64, *g.poff.device_ptr() as u64, ne as i32))
                                    .unwrap_or_else(|e| panic!("mxfp4 moe_gemv (gu): {e:?}"));
                            }
                        }
                    } else {
                        blaunch!(self, "gemm_moe_grouped_mma_fp4", (((2*mi/16) as u32), ngroups_max, 1), (256,1,1), 0,
                            (d(&gu_p), *guq.device_ptr() as u64, *gus.device_ptr() as u64, d(gugs), d(&x_perm),
                             *g.tile_e.device_ptr() as u64, (2*mi) as i32, h as i32, e_base,
                             *g.poff.device_ptr() as u64, ne as i32));
                    }
                } else {
                    blaunch!(self, "gemm_moe_grouped_mma_fp4", (((2*mi/16) as u32), ngroups_max, 1), (256,1,1), 0,
                        (d(&gu_p), *guq.device_ptr() as u64, *gus.device_ptr() as u64, d(gugs), d(&x_perm),
                         *g.tile_e.device_ptr() as u64, (2*mi) as i32, h as i32, e_base,
                         *g.poff.device_ptr() as u64, ne as i32));
                }
                // Fused-quant path: the fused dn kernel computes silu in its stage from gu_p
                // directly; h_p (and its silu launch) vanish on the fused path (§8).
                let dn_native_fused = self.mxfp4.as_ref().map_or(false, |st| {
                    !st.allow_bf16.read().unwrap().contains(&(*dnq.device_ptr() as u64))
                        && Self::mxfp4_fused_on()
                });
                let h_p: Option<B> = if dn_native_fused { None } else { Some(pool.get_bf16(ppad_max * mi)) };
                if let Some(h_p) = &h_p {
                    blaunch!(self, "moe_silu_bf16_b", grid(ppad_max * mi), (256,1,1), 0,
                        (d(h_p), d(&gu_p), mi as i32, ppad_max as i32,
                         *g.poff.device_ptr() as u64, ne as i32));
                    if self.rotated.contains(&(*dnq.device_ptr() as u64)) { self.rot_inplace(h_p, ppad_max * mi); }
                }
                let dn_p = pool.get_bf16(ppad_max * h);
                if let Some(st) = &self.mxfp4 {
                    let dn_ptr = *dnq.device_ptr() as u64;
                    // Same prefill-width rule as the gu arm above (2026-08-17).
                    let native_wide = Self::moe_native_pf_on();
                    if native_wide
                        && !st.allow_bf16.read().unwrap().contains(&dn_ptr) {
                        let (dn_a, dn_s2) = st.w.get(&dn_ptr).unwrap_or_else(|| {
                            panic!("mxfp4: MoE down @{dn_ptr:x} has no OMMA repack") }).ptrs();
                        let nks_dn = (mi / 64) as i32;
                        if Self::mxfp4_fused_on() {
                            // FUSED (F3, XSILU=1): silu computed in the dn stage from gu_p with
                            // moe_silu_bf16_b's exact math — ONE launch, no h_p, no silu kernel.
                            unsafe {
                                st.moe_grouped_fused1.clone().launch_on_stream(&self.stream,
                                    LaunchConfig { grid_dim: ((h/16) as u32, ngroups_max, 1), block_dim: (256,1,1), shared_mem_bytes: 0 },
                                    (d(&dn_p), dn_a, dn_s2, d(dngs), d(&gu_p),
                                     (h/16) as i32, nks_dn, mi as i32, e_base,
                                     *g.tile_e.device_ptr() as u64, *g.poff.device_ptr() as u64, ne as i32))
                                    .unwrap_or_else(|e| panic!("mxfp4 moe_grouped_fused (dn): {e:?}"));
                            }
                        } else {
                            let ng_quant = (ppad_max / 8) as u32;
                            unsafe {
                                st.quant_ng.clone().launch_on_stream(&self.stream,
                                    LaunchConfig { grid_dim: (ng_quant * nks_dn as u32, 1, 1), block_dim: (256,1,1), shared_mem_bytes: 0 },
                                    (d(h_p.as_ref().unwrap()), mi as i32, ng_quant as i32, nks_dn, 8i32, 8i32,
                                     d(&st.moe_bp), d(&st.moe_sfb),
                                     *g.poff.device_ptr() as u64, ne as i32)).unwrap_or_else(|e| panic!("mxfp4 quant_ng: {e:?}"));
                                st.moe_gemv.clone().launch_on_stream(&self.stream,
                                    LaunchConfig { grid_dim: ((h/16) as u32, ngroups_max, 1), block_dim: (256,1,1), shared_mem_bytes: 0 },
                                    (d(&dn_p), dn_a, dn_s2, d(dngs), d(&st.moe_bp), d(&st.moe_sfb),
                                     (h/16) as i32, nks_dn, e_base,
                                     *g.tile_e.device_ptr() as u64, *g.poff.device_ptr() as u64, ne as i32))
                                    .unwrap_or_else(|e| panic!("mxfp4 moe_gemv (dn): {e:?}"));
                            }
                        }
                    } else {
                        blaunch!(self, "gemm_moe_grouped_mma_fp4", (((h/16) as u32), ngroups_max, 1), (256,1,1), 0,
                            (d(&dn_p), *dnq.device_ptr() as u64, *dns.device_ptr() as u64, d(dngs), d(h_p.as_ref().unwrap()),
                             *g.tile_e.device_ptr() as u64, h as i32, mi as i32, e_base,
                             *g.poff.device_ptr() as u64, ne as i32));
                    }
                } else {
                    blaunch!(self, "gemm_moe_grouped_mma_fp4", (((h/16) as u32), ngroups_max, 1), (256,1,1), 0,
                        (d(&dn_p), *dnq.device_ptr() as u64, *dns.device_ptr() as u64, d(dngs), d(h_p.as_ref().unwrap()),
                         *g.tile_e.device_ptr() as u64, h as i32, mi as i32, e_base,
                         *g.poff.device_ptr() as u64, ne as i32));
                }
                blaunch!(self, "moe_combine_grouped_b", grid(batch * h), (256,1,1), 0,
                    (d(&out), d(&dn_p), *g.perm_wt.device_ptr() as u64, *g.inv_pos.device_ptr() as u64,
                     h as i32, k as i32, batch as i32));
                pool.release_bf16(x_perm, ppad_max*h); pool.release_bf16(gu_p, ppad_max*2*mi);
                if let Some(h_p) = h_p { pool.release_bf16(h_p, ppad_max*mi); }
                pool.release_bf16(dn_p, ppad_max*h);
                }
              } else if batch < 128 {
                // DECODE/VERIFY (N=1 per pair): no token-sharing at small batch, so no gather needed.
                if fold {
                    // E12 — the shared expert is slot k of every token: ONE extra column in the SAME
                    // slot GEMMs (grid.y = batch·(k+1)), silu over the extra rows, and the fold
                    // combine. Per-element math and the two-stage rounding are bit-identical to the
                    // separate gemm_act path (see the kernel comments); fold_shared folds the shared
                    // PARTIAL pre-reduce, !fold_shared reduces the routed partial first and adds the
                    // replicated shared slot post-reduce (the old add_residual_b semantics).
                    let bkf = batch * (k + 1);
                    let gu_s = pool.get_bf16(bkf * 2 * mi);
                    let h_s  = pool.get_bf16(bkf * mi);
                    let dn_s = pool.get_bf16(bkf * h);
                    // Kslots = k+1: the fold GEMM decomposes bslot = b·Kslots + s with the shared
                    // slot at s == Kslots-1 (routed slots s < Kslots-1, ids[b·(Kslots-1)+s]) —
                    // grid.y = batch·(k+1) columns. Packing k (the routed count) misaligned every
                    // slot by one: the block at bslot = b·(k+1)+k took the ROUTED branch with
                    // xrow = b+1 (OOB for the last token) and the shared computation landed at
                    // bslot = b·k+k-1 — the combine read the OOB row as the shared slot → NaN.
                    let kx = ((k + 1) as i64) << 32;   // Kslots<<32 | x_by_slot (packed for the 12-arg cap)
                    self.launch_moe_fold_gemm("gemm_moe_mma_fp4_fold",
                        ((2*mi/16) as u32, bkf as u32), false,
                        (d(&gu_s), *guq.device_ptr() as u64, *gus.device_ptr() as u64, d(gugs), d(x), ids_ptr,
                         (2*mi) as i32, h as i32, kx, e_base, e_span, sdesc_ptr));
                    // E8: fold-aware silu — the shared slot (row%(k+1)==k) writes the rank's half
                    // of the act row; routed slots are identical to moe_silu_bf16_b.
                    blaunch!(self, "moe_silu_fold_b", grid(bkf * mi), (256,1,1), 0,
                        (d(&h_s), d(&gu_s), mi as i32, bkf as i32, 0u64, k as i32, sdesc_ptr + 32));
                    // E11: the DOWN GEMM may run the PDL variant — its weight selection (ids) was
                    // consumed by the up GEMM + silu already, so the weight stream can overlap the
                    // directly-preceding silu (the shared slot's descriptor is a kernel argument).
                    self.launch_moe_fold_gemm("gemm_moe_mma_fp4_fold",
                        ((h/16) as u32, bkf as u32), true,
                        (d(&dn_s), *dnq.device_ptr() as u64, *dns.device_ptr() as u64, d(dngs), d(&h_s), ids_ptr,
                         h as i32, mi as i32, kx | 1, e_base, e_span, sdesc_ptr + 32));
                    blaunch!(self, "moe_combine_experts_fold_b", grid(batch * h), (256,1,1), 0,
                        (d(&out), d(&dn_s), wts_ptr, sgate_ptr, h as i32, k as i32, batch as i32,
                         fold_shared as i32, gated as i32));
                    if !fold_shared && moe.experts_sharded { self.tp_all_reduce_bf16(&mut out, h * batch); }
                    if !fold_shared {
                        // The shared slot's down row for token b is dn_s[(b·(k+1)+k)·h].
                        blaunch!(self, "moe_shared_slot_combine_b", grid(batch * h), (256,1,1), 0,
                            (d(&out), d(&dn_s), 0u64, sgate_ptr, h as i32, (k + 1) as i32, k as i32,
                             batch as i32, 0i32, 0i32, gated as i32));
                    }
                    if fold_shared {
                        if fuse {
                            if let Some(fl) = self.tp_all_reduce_bf16_k1(&mut out, h * batch) {
                                fused_out = Some(match fl {
                                    FusedLocal::F32Scratch => FusedPartial::F32Scratch,
                                    _ => FusedPartial::Bf16,
                                });
                            }
                        } else {
                            self.tp_all_reduce_bf16(&mut out, h * batch);
                        }
                    }
                    if crate::opts::var(crate::opt!("moe-fold-debug")).is_ok() {
                        self.sync_stream();
                        let mk = |buf: &B, n: usize| -> (usize, usize) {
                            let v: Vec<half::bf16> = self.dev.dtoh_sync_copy(buf).unwrap();
                            (v[..n].iter().filter(|x| x.is_nan()).count(),
                             v[..n].iter().filter(|x| x.is_finite()).count())
                        };
                        let (gn, gf) = mk(&gu_s, bkf*2*mi);
                        let (hn, hf) = mk(&h_s, bkf*mi);
                        let (dn, df) = mk(&dn_s, bkf*h);
                        eprintln!("  [fold-dbg] b={batch} k={k} gu_s {gn} NaN/{gf} finite | h_s {hn}/{hf} | dn_s {dn}/{df} (rows {bkf})");
                    }
                    pool.release_bf16(gu_s, bkf*2*mi); pool.release_bf16(h_s, bkf*mi); pool.release_bf16(dn_s, bkf*h);
                } else {
                let bk = batch * k;
                let gu_s = pool.get_bf16(bk * 2 * mi);
                // Fused-quant path (EXPERT_FUSED_QUANT_RESPONSE.md §8): the fused dn kernel
                // reads gu_s directly and computes silu in its stage, so the h_s allocation
                // and the silu launch vanish on the fused path (pool pressure drops); the
                // unfused path keeps both exactly as today.
                let dn_native_fused = self.mxfp4.as_ref().map_or(false, |st| {
                    !st.allow_bf16.read().unwrap().contains(&(*dnq.device_ptr() as u64))
                        && Self::mxfp4_fused_on()
                });
                let h_s: Option<B> = if dn_native_fused { None } else { Some(pool.get_bf16(bk * mi)) };
                let dn_s = pool.get_bf16(bk * h);
                if let Some(st) = &self.mxfp4 {
                    // MXFP4-native slot path: one 8-token OMMA group per TOKEN for gate_up (N=1,
                    // rows 1..7 zeroed — group g = token row g); the down side is SLOT-major (h_s
                    // row = slot), so its quant has bk groups (group s = slot s).
                    let nks_gu = (h / 64) as i32;
                    let gu_ptr = *guq.device_ptr() as u64;
                    if !st.allow_bf16.read().unwrap().contains(&gu_ptr) {
                        let (gu_a, gu_s2) = st.w.get(&gu_ptr).unwrap_or_else(|| {
                            panic!("mxfp4: MoE gate_up @{gu_ptr:x} has no OMMA repack") }).ptrs();
                        if Self::mxfp4_fused_on() {
                            // FUSED (F2, XSILU=0): quantize the slot's row in-block, ONE launch.
                            unsafe {
                                st.moe_slot_fused0.clone().launch_on_stream(&self.stream,
                                    LaunchConfig { grid_dim: ((2*mi/16) as u32, bk as u32, 1), block_dim: (256,1,1), shared_mem_bytes: 0 },
                                    (d(&gu_s), gu_a, gu_s2, d(gugs), d(x), ids_ptr,
                                     nks_gu, h as i32, k as i32, e_base, e_span,
                                     (2*mi/16) as i32)).unwrap_or_else(|e| panic!("mxfp4 moe_slot_fused (gu): {e:?}"));
                            }
                        } else {
                            unsafe {
                                st.quant_ng.clone().launch_on_stream(&self.stream,
                                    LaunchConfig { grid_dim: (batch as u32 * nks_gu as u32, 1, 1), block_dim: (256,1,1), shared_mem_bytes: 0 },
                                    (d(x), h as i32, batch as i32, nks_gu, 1i32, 1i32,
                                     d(&st.moe_bp), d(&st.moe_sfb), 0u64, 0i32)).unwrap_or_else(|e| panic!("mxfp4 quant_ng: {e:?}"));
                                st.moe_slot.clone().launch_on_stream(&self.stream,
                                    LaunchConfig { grid_dim: ((2*mi/16) as u32, bk as u32, 1), block_dim: (256,1,1), shared_mem_bytes: 0 },
                                    (d(&gu_s), gu_a, gu_s2, d(gugs), d(&st.moe_bp), d(&st.moe_sfb), ids_ptr,
                                     nks_gu, k as i32, e_base, e_span,
                                     (2*mi/16) as i32)).unwrap_or_else(|e| panic!("mxfp4 moe_slot (gu): {e:?}"));
                            }
                        }
                    } else {
                        blaunch!(self, "gemm_moe_mma_fp4", (((2*mi/16) as u32), bk as u32, 1), (256,1,1), 0,
                            (d(&gu_s), *guq.device_ptr() as u64, *gus.device_ptr() as u64, d(gugs), d(x), ids_ptr,
                             (2*mi) as i32, h as i32, k as i32, 0i32, e_base, e_span));
                    }
                } else {
                    let _rg; let xg: &B = if self.rotated.contains(&(*guq.device_ptr() as u64)) { _rg = self.rot_x(x, h, batch); &*_rg } else { x };
                    blaunch!(self, "gemm_moe_mma_fp4", (((2*mi/16) as u32), bk as u32, 1), (256,1,1), 0,
                        (d(&gu_s), *guq.device_ptr() as u64, *gus.device_ptr() as u64, d(gugs), d(xg), ids_ptr,
                         (2*mi) as i32, h as i32, k as i32, 0i32, e_base, e_span));
                }
                if let Some(h_s) = &h_s {
                    blaunch!(self, "moe_silu_bf16_b", grid(bk * mi), (256,1,1), 0,
                        (d(h_s), d(&gu_s), mi as i32, bk as i32, 0u64, 0i32));
                    if self.rotated.contains(&(*dnq.device_ptr() as u64)) { self.rot_inplace(h_s, bk * mi); }
                }
                if let Some(st) = &self.mxfp4 {
                    let dn_ptr = *dnq.device_ptr() as u64;
                    if !st.allow_bf16.read().unwrap().contains(&dn_ptr) {
                        let (dn_a, dn_s2) = st.w.get(&dn_ptr).unwrap_or_else(|| {
                            panic!("mxfp4: MoE down @{dn_ptr:x} has no OMMA repack") }).ptrs();
                        let nks_dn = (mi / 64) as i32;
                        if Self::mxfp4_fused_on() {
                            // FUSED (F2, XSILU=1): silu computed in the dn stage from gu_s with
                            // moe_silu_bf16_b's exact math — ONE launch, no h_s, no silu kernel.
                            unsafe {
                                st.moe_slot_fused1.clone().launch_on_stream(&self.stream,
                                    LaunchConfig { grid_dim: ((h/16) as u32, bk as u32, 1), block_dim: (256,1,1), shared_mem_bytes: 0 },
                                    (d(&dn_s), dn_a, dn_s2, d(dngs), d(&gu_s), ids_ptr,
                                     nks_dn, mi as i32, 0i32, e_base, e_span,
                                     (h/16) as i32)).unwrap_or_else(|e| panic!("mxfp4 moe_slot_fused (dn): {e:?}"));
                            }
                        } else {
                            unsafe {
                                st.quant_ng.clone().launch_on_stream(&self.stream,
                                    LaunchConfig { grid_dim: (bk as u32 * nks_dn as u32, 1, 1), block_dim: (256,1,1), shared_mem_bytes: 0 },
                                    (d(h_s.as_ref().unwrap()), mi as i32, bk as i32, nks_dn, 1i32, 1i32,
                                     d(&st.moe_bp), d(&st.moe_sfb), 0u64, 0i32)).unwrap_or_else(|e| panic!("mxfp4 quant_ng: {e:?}"));
                                st.moe_slot.clone().launch_on_stream(&self.stream,
                                    LaunchConfig { grid_dim: ((h/16) as u32, bk as u32, 1), block_dim: (256,1,1), shared_mem_bytes: 0 },
                                    (d(&dn_s), dn_a, dn_s2, d(dngs), d(&st.moe_bp), d(&st.moe_sfb), ids_ptr,
                                     nks_dn, 0i32, e_base, e_span,
                                     (h/16) as i32)).unwrap_or_else(|e| panic!("mxfp4 moe_slot (dn): {e:?}"));
                            }
                        }
                    } else {
                        blaunch!(self, "gemm_moe_mma_fp4", (((h/16) as u32), bk as u32, 1), (256,1,1), 0,
                            (d(&dn_s), *dnq.device_ptr() as u64, *dns.device_ptr() as u64, d(dngs), d(h_s.as_ref().unwrap()), ids_ptr,
                             h as i32, mi as i32, k as i32, 1i32, e_base, e_span));
                    }
                } else {
                    blaunch!(self, "gemm_moe_mma_fp4", (((h/16) as u32), bk as u32, 1), (256,1,1), 0,
                        (d(&dn_s), *dnq.device_ptr() as u64, *dns.device_ptr() as u64, d(dngs), d(h_s.as_ref().unwrap()), ids_ptr,
                         h as i32, mi as i32, k as i32, 1i32, e_base, e_span));
                }
                blaunch!(self, "moe_combine_experts_b", grid(batch * h), (256,1,1), 0,
                    (d(&out), d(&dn_s), wts_ptr, h as i32, k as i32, batch as i32));
                pool.release_bf16(gu_s, bk*2*mi);
                if let Some(h_s) = h_s { pool.release_bf16(h_s, bk*mi); }
                pool.release_bf16(dn_s, bk*h);
                }
              } else {
                // TOKEN-GATHER (prefill): counting-sort pairs by expert → ONE weight read/expert.
                // E3b — the E16 device-only sequence at window scale. The old arm read poff[ne] back
                // to the host to size the grid and the pool buffers (sync_stream + dtoh): a FULL
                // pipeline drain per MoE layer per prefill window — 79 drains per 8K window on Hy3.
                // Metadata lives in the preallocated moe_g_pf scratch, count/perm_tok are memset on
                // the compute stream, and the grid is a host-known bound with per-block early-exit on
                // a DEVICE read of poff[ne] — so the host never reads poff back. poff stays device-
                // only under TP too: each rank's prefix covers only its own expert band, and the only
                // cross-rank sync remains the h*batch all-reduce below (identical on both ranks).
                //
                // SUB-BATCHED at the scratch's pair budget (tok_cap = p_max/k = PREFILL_CHUNK
                // tokens): moe_batch is a library function — the serving scheduler feeds it <=8K
                // windows, but the bench harnesses call prefill_batch DIRECTLY with the full prompt
                // (batch >> 8192), and the pre-scratch dynamic arm handled any batch. Each sub-batch
                // runs the full sequence over its token slice. Per-(token,slot) pair work is
                // independent of how tokens are grouped: every pair gets its own padded row, and a
                // row's gather/GEMM/silu/combine math touches only its own token, its expert's
                // weight and its slot weights (GEMM rows are independent N-columns with a fixed
                // per-element k-order; combine reads pairs by inv_pos, not by row position, so even
                // the scatter's atomicAdd order already doesn't matter). Token-aligned slices keep
                // each token's k pairs together, so the per-token slot-order combine sum is
                // chunking-invariant — out is bit-identical to one big call. batch <= tok_cap (the
                // serving path) is ONE iteration with zero-offset slices: same launches as before.
                let ne = cfg.num_experts;
                let g = &self.moe_g_pf;
                let tok_cap = g.p_max / k;
                let mut t0 = 0usize;
                while t0 < batch {
                    let ns = tok_cap.min(batch - t0);
                    let p = ns * k;
                    // Tripwire: a single sub-batch can never overflow (tok_cap = p_max/k by
                    // construction) — a fire here means the scratch and the constant diverged.
                    assert!(p <= g.p_max,
                            "MoE grouped prefill scratch overflow: sub-batch {ns} × k {k} = {p} pairs > p_max {} \
                             — moe_g_pf mis-sized (tok_cap = p_max/k violated)", g.p_max);
                    // Slices are TOKEN-relative: ids/wts are [batch, k] token-major, x and out are
                    // [h, batch] token-major, and the kernels index pair/token rows relative to the
                    // slice base (perm_tok = p/k; combine token b = idx/H) — self-contained.
                    let ids_s = ids_ptr + (t0 * k * 4) as u64;
                    let wts_s = wts_ptr + (t0 * k * 4) as u64;
                    let x_s = d(x) + (t0 * h * 2) as u64;
                    let out_s = d(&out) + (t0 * h * 2) as u64;
                    // Tightest host-known bound on the padded total: <= 16 rows of 16-alignment
                    // padding per LOCAL expert (moe_offsets_b pads count[e] up to a multiple of 16;
                    // remote experts count 0). Unsharded e_span == ne (the old code's own bound);
                    // TP-sharded it's tighter. Pool buffers size off THIS, not the static ppad_max,
                    // so a short window doesn't grow the pool peak to the full 8K-window size.
                    // E12: +pad32(ns) for the shared-expert region (moe_offsets_fold_b appends it
                    // after the routed band; the old path ignores the slack — its GEMMs early-exit
                    // on poff[ne]). ≤ pad32(tok_cap) by construction, so ppad_max always covers it.
                    let ppad_cap = p + (e_span as usize) * 32
                                 + if fold { ((ns + 31) / 32) * 32 } else { 0 };
                    // Tripwire: the shared region (fold) is pad32(ns) <= pad32(tok_cap) rows, so the
                    // scratch's ppad_max (routed bound + pad32(p_max/k)) must always cover ppad_cap —
                    // a fire here means the constructor's sizing and this bound diverged.
                    assert!(ppad_cap <= g.ppad_max,
                            "MoE grouped prefill scratch ppad overflow: sub-batch {ns} × k {k} = {p} pairs + \
                             {e_span}×32 routed padding + {} shared rows = {ppad_cap} > ppad_max {} \
                             — moe_g_pf mis-sized for the E12 shared region", ((ns + 31) / 32) * 32, g.ppad_max);
                    if fold {
                        // E12 fold (per sub-batch, token-relative slices; see the verify regime for
                        // the shared-region semantics). Wide/16-token respects --moe-grouped-wide
                        // like the old path; the reduce placement mirrors it too.
                        self.memset_compute_stream(*g.count.device_ptr() as u64, ne * 4);
                        blaunch!(self, "moe_count_b", grid(p), (256,1,1), 0,
                            (*g.count.device_ptr() as u64, ids_s, p as i32, e_base, e_base + e_span));
                        blaunch!(self, "moe_offsets_fold_b", (1u32,1,1), (32,1,1), 0,
                            (*g.poff.device_ptr() as u64, *g.cursor.device_ptr() as u64, *g.count.device_ptr() as u64,
                             ne as i32, ns as i32));
                        self.memset_compute_stream_u32(*g.perm_tok.device_ptr() as u64, 0xFFFF_FFFF, ppad_cap);
                        blaunch!(self, "moe_scatter_b", grid(p), (256,1,1), 0,
                            (*g.perm_tok.device_ptr() as u64, *g.perm_wt.device_ptr() as u64, *g.inv_pos.device_ptr() as u64,
                             *g.cursor.device_ptr() as u64, ids_s, wts_s, p as i32, k as i32,
                             e_base, e_base + e_span));
                        let sgate_s = if gated { sgate_ptr + (t0 * 2) as u64 } else { 0 };
                        blaunch!(self, "moe_shared_scatter_b", grid(ns), (256,1,1), 0,
                            (*g.perm_tok.device_ptr() as u64, *g.perm_wt.device_ptr() as u64, *g.poff.device_ptr() as u64,
                             sgate_s, ne as i32, ns as i32, gated as i32));
                        blaunch!(self, "moe_tilemap_fold_b", grid(ne + 1), (256,1,1), 0,
                            (*g.tile_e.device_ptr() as u64, *g.poff.device_ptr() as u64, ne as i32));
                        let x_perm = pool.get_bf16(ppad_cap * h);
                        blaunch!(self, "moe_gather_x_b", grid(ppad_cap * h), (256,1,1), 0,
                            (d(&x_perm), x_s, *g.perm_tok.device_ptr() as u64, h as i32, ppad_cap as i32,
                         *g.poff.device_ptr() as u64, ne as i32));
                        // MR-GPTQ: the gathered rows get the H16/4 micro-rotation (same as the < 128 paths)
                        if self.rotated.contains(&(*guq.device_ptr() as u64)) { self.rot_inplace(&x_perm, ppad_cap * h); }
                        let ngroups_max = (ppad_cap / 16) as u32;
                        let wide_grouped = crate::opts::var(crate::opt!("moe-grouped-wide")).map_or(true, |v| v != "0");
                        let ngroups_x4 = (ppad_cap / 32) as u32;
                        let gu_p = pool.get_bf16(ppad_cap * 2 * mi);
                        if wide_grouped {
                            blaunch!(self, "gemm_moe_grouped_mma_fp4_x4_fold", (((2*mi/16) as u32), ngroups_x4, 1), (256,1,1), 0,
                                (d(&gu_p), *guq.device_ptr() as u64, *gus.device_ptr() as u64, d(gugs), d(&x_perm),
                                 *g.tile_e.device_ptr() as u64, (2*mi) as i32, h as i32, e_base,
                                 *g.poff.device_ptr() as u64, (ne + 1) as i32,
                                 sdesc_ptr));
                        } else {
                            blaunch!(self, "gemm_moe_grouped_mma_fp4_fold", (((2*mi/16) as u32), ngroups_max, 1), (256,1,1), 0,
                                (d(&gu_p), *guq.device_ptr() as u64, *gus.device_ptr() as u64, d(gugs), d(&x_perm),
                                 *g.tile_e.device_ptr() as u64, (2*mi) as i32, h as i32, e_base,
                                 *g.poff.device_ptr() as u64, (ne + 1) as i32,
                                 sdesc_ptr));
                        }
                        let h_p = pool.get_bf16(ppad_cap * mi);
                        // E8: fold-aware silu — the shared region's rows write the rank's half of
                        // the act row (moe_silu_fold_b; routed rows identical to moe_silu_bf16_b).
                        blaunch!(self, "moe_silu_fold_b", grid(ppad_cap * mi), (256,1,1), 0,
                            (d(&h_p), d(&gu_p), mi as i32, ppad_cap as i32,
                             *g.poff.device_ptr() as u64, ne as i32, sdesc_ptr + 32));
                        if self.rotated.contains(&(*dnq.device_ptr() as u64)) { self.rot_inplace(&h_p, ppad_cap * mi); }
                        let dn_p = pool.get_bf16(ppad_cap * h);
                        if wide_grouped {
                            blaunch!(self, "gemm_moe_grouped_mma_fp4_x4_fold", (((h/16) as u32), ngroups_x4, 1), (256,1,1), 0,
                                (d(&dn_p), *dnq.device_ptr() as u64, *dns.device_ptr() as u64, d(dngs), d(&h_p),
                                 *g.tile_e.device_ptr() as u64, h as i32, mi as i32, e_base,
                                 *g.poff.device_ptr() as u64, (ne + 1) as i32,
                                 sdesc_ptr + 32));
                        } else {
                            blaunch!(self, "gemm_moe_grouped_mma_fp4_fold", (((h/16) as u32), ngroups_max, 1), (256,1,1), 0,
                                (d(&dn_p), *dnq.device_ptr() as u64, *dns.device_ptr() as u64, d(dngs), d(&h_p),
                                 *g.tile_e.device_ptr() as u64, h as i32, mi as i32, e_base,
                                 *g.poff.device_ptr() as u64, (ne + 1) as i32,
                                 sdesc_ptr + 32));
                        }
                        // fold_shared: shared PARTIAL combined pre-reduce (one reduce restores both
                        // sums); !fold_shared: routed-only combine, reduce, then the replicated
                        // shared slot post-reduce. Per-sub-batch reduces are elementwise and
                        // token-independent, so the result is identical to one whole-batch reduce.
                        blaunch!(self, "moe_combine_grouped_fold_b", grid(ns * h), (256,1,1), 0,
                            (out_s, d(&dn_p), *g.perm_wt.device_ptr() as u64, *g.inv_pos.device_ptr() as u64,
                             *g.poff.device_ptr() as u64, h as i32, k as i32, ns as i32, ne as i32,
                             fold_shared as i32));
                        if !fold_shared && moe.experts_sharded { self.tp_all_reduce_bf16(&mut out, h * batch); }
                        if !fold_shared {
                            blaunch!(self, "moe_shared_slot_combine_b", grid(ns * h), (256,1,1), 0,
                                (out_s, d(&dn_p), *g.poff.device_ptr() as u64, sgate_s,
                                 h as i32, 0i32, 0i32, ns as i32, ne as i32, 1i32, gated as i32));
                        }
                        if fold_shared { self.tp_all_reduce_bf16(&mut out, h * batch); }
                        pool.release_bf16(x_perm, ppad_cap*h); pool.release_bf16(gu_p, ppad_cap*2*mi);
                        pool.release_bf16(h_p, ppad_cap*mi); pool.release_bf16(dn_p, ppad_cap*h);
                    } else {
                    self.memset_compute_stream(*g.count.device_ptr() as u64, ne * 4);
                    blaunch!(self, "moe_count_b", grid(p), (256,1,1), 0,
                        (*g.count.device_ptr() as u64, ids_s, p as i32, e_base, e_base + e_span));
                    blaunch!(self, "moe_offsets_b", (1u32,1,1), (32,1,1), 0,
                        (*g.poff.device_ptr() as u64, *g.cursor.device_ptr() as u64, *g.count.device_ptr() as u64, ne as i32));
                    self.memset_compute_stream_u32(*g.perm_tok.device_ptr() as u64, 0xFFFF_FFFF, ppad_cap);
                    blaunch!(self, "moe_scatter_b", grid(p), (256,1,1), 0,
                        (*g.perm_tok.device_ptr() as u64, *g.perm_wt.device_ptr() as u64, *g.inv_pos.device_ptr() as u64,
                         *g.cursor.device_ptr() as u64, ids_s, wts_s, p as i32, k as i32,
                         e_base, e_base + e_span));
                    blaunch!(self, "moe_tilemap_b", grid(ne), (256,1,1), 0,
                        (*g.tile_e.device_ptr() as u64, *g.poff.device_ptr() as u64, ne as i32));
                    let x_perm = pool.get_bf16(ppad_cap * h);
                    if self.igs_armed() { self.memset_compute_stream(d(&x_perm), ppad_cap * h * 2); }   // stale tail rows must not feed the amax
                    blaunch!(self, "moe_gather_x_b", grid(ppad_cap * h), (256,1,1), 0,
                        (d(&x_perm), x_s, *g.perm_tok.device_ptr() as u64, h as i32, ppad_cap as i32,
                         *g.poff.device_ptr() as u64, ne as i32));
                    // MR-GPTQ: the gathered rows get the H16/4 micro-rotation (same as the < 128 paths)
                    if self.rotated.contains(&(*guq.device_ptr() as u64)) { self.rot_inplace(&x_perm, ppad_cap * h); }
                    self.igs_tap_x(*guq.device_ptr() as u64, &x_perm, ppad_cap * h);
                    let ngroups_max = (ppad_cap / 16) as u32;   // blocks past poff[ne]/16 early-exit on device
                    // E23: 32-token grouped tiles (x4) halve the expert-weight re-reads at prefill
                    // scale. Bitwise contract identical (see the kernel); --moe-grouped-wide=0
                    // keeps the 16-token kernel for the A/B byte gate.
                    let wide_grouped = crate::opts::var(crate::opt!("moe-grouped-wide")).map_or(true, |v| v != "0");
                    let ngroups_x4 = (ppad_cap / 32) as u32;
                    let gu_p = pool.get_bf16(ppad_cap * 2 * mi);
                    // NVFP4 W4A4 prefill (src/w4a4.rs): both expert GEMMs of this layer on the
                    // block-scaled FP4 tensor cores (x_perm / h_p quantized per 16-block).
                    let pf_fused = Self::mxfp4_fused_on() && Self::mxfp4_fused_prefill_on();
                    let w4_moe = self.w4a4.as_ref().filter(|w| w.on(*guq.device_ptr() as u64)
                        && w.on(*dnq.device_ptr() as u64) && h % 64 == 0 && mi % 64 == 0);
                    if let Some(w4) = w4_moe {
                        self.w4a4_moe(w4, &x_perm, ppad_cap, h, *guq.device_ptr() as u64, *gus.device_ptr() as u64,
                                      *gugs.device_ptr() as u64, 2 * mi, &gu_p, g, ne, e_base, true, w4.xgs(*guq.device_ptr() as u64));
                        if crate::w4a4::check_on() {
                            let r = pool.get_bf16(ppad_cap * 2 * mi);
                            let xf = self.w4a4_fakequant(w4, &x_perm, ppad_cap, h, w4.xgs(*guq.device_ptr() as u64));
                            blaunch!(self, "gemm_moe_grouped_mma_fp4_x4", (((2*mi/16) as u32), (ppad_cap / 32) as u32, 1), (256,1,1), 0,
                                (d(&r), *guq.device_ptr() as u64, *gus.device_ptr() as u64, d(gugs), d(&xf),
                                 *g.tile_e.device_ptr() as u64, (2*mi) as i32, h as i32, e_base,
                                 *g.poff.device_ptr() as u64, ne as i32));
                            let real = { let p: Vec<i32> = self.dev.dtoh_sync_copy(&g.poff).unwrap(); p[ne] as usize };
                            self.w4a4_compare(&gu_p, &r, real, 2 * mi, &format!("moe gate_up rows={real}"));
                            pool.release_bf16(r, ppad_cap * 2 * mi);
                        }
                    } else {
                    // --mxfp4-fused-prefill=0 keeps the separate pair at the prefill windows
                    // (the §10 R3 L2-thrash escape); default fused.
                    if let Some(st) = &self.mxfp4 {
                        if !st.allow_bf16.read().unwrap().contains(&(*guq.device_ptr() as u64)) {
                            // MXFP4-native grouped-prefill: quant the window's x_perm (two 8-token
                            // groups per 16-token expert group) + the grouped OMMA GEMV. The MoE
                            // pack scratch is sized for the window (moe_groups_max in the state).
                            let nks_gu = (h / 64) as i32;
                            let (gu_a, gu_s2) = st.w.get(&(*guq.device_ptr() as u64)).unwrap_or_else(|| {
                                panic!("mxfp4: MoE gate_up @{:x} has no OMMA repack", *guq.device_ptr() as u64) }).ptrs();
                            if pf_fused {
                                // FUSED (F3, XSILU=0): two 8-token group passes per kstep in-block.
                                unsafe {
                                    st.moe_grouped_fused0.clone().launch_on_stream(&self.stream,
                                        LaunchConfig { grid_dim: ((2*mi/16) as u32, ngroups_max, 1), block_dim: (256,1,1), shared_mem_bytes: 0 },
                                        (d(&gu_p), gu_a, gu_s2, d(gugs), d(&x_perm),
                                         (2*mi/16) as i32, nks_gu, h as i32, e_base,
                                         *g.tile_e.device_ptr() as u64, *g.poff.device_ptr() as u64, ne as i32))
                                        .unwrap_or_else(|e| panic!("mxfp4 moe_grouped_fused (gu): {e:?}"));
                                }
                            } else {
                                let ng_quant = (ppad_cap / 8) as u32;
                                unsafe {
                                    st.quant_ng.clone().launch_on_stream(&self.stream,
                                        LaunchConfig { grid_dim: (ng_quant * nks_gu as u32, 1, 1), block_dim: (256,1,1), shared_mem_bytes: 0 },
                                        (d(&x_perm), h as i32, ng_quant as i32, nks_gu, 8i32, 8i32,
                                         d(&st.moe_bp), d(&st.moe_sfb),
                                         *g.poff.device_ptr() as u64, ne as i32)).unwrap_or_else(|e| panic!("mxfp4 quant_ng: {e:?}"));
                                    st.moe_gemv.clone().launch_on_stream(&self.stream,
                                        LaunchConfig { grid_dim: ((2*mi/16) as u32, ngroups_max, 1), block_dim: (256,1,1), shared_mem_bytes: 0 },
                                        (d(&gu_p), gu_a, gu_s2, d(gugs), d(&st.moe_bp), d(&st.moe_sfb),
                                         (2*mi/16) as i32, nks_gu, e_base,
                                         *g.tile_e.device_ptr() as u64, *g.poff.device_ptr() as u64, ne as i32))
                                        .unwrap_or_else(|e| panic!("mxfp4 moe_gemv (gu): {e:?}"));
                                }
                            }
                        } else {
                            if wide_grouped {
                                blaunch!(self, "gemm_moe_grouped_mma_fp4_x4", (((2*mi/16) as u32), ngroups_x4, 1), (256,1,1), 0,
                                    (d(&gu_p), *guq.device_ptr() as u64, *gus.device_ptr() as u64, d(gugs), d(&x_perm),
                                     *g.tile_e.device_ptr() as u64, (2*mi) as i32, h as i32, e_base,
                                     *g.poff.device_ptr() as u64, ne as i32));
                            } else {
                                blaunch!(self, "gemm_moe_grouped_mma_fp4", (((2*mi/16) as u32), ngroups_max, 1), (256,1,1), 0,
                                    (d(&gu_p), *guq.device_ptr() as u64, *gus.device_ptr() as u64, d(gugs), d(&x_perm),
                                     *g.tile_e.device_ptr() as u64, (2*mi) as i32, h as i32, e_base,
                                     *g.poff.device_ptr() as u64, ne as i32));
                            }
                        }
                    } else {
                        if wide_grouped {
                            blaunch!(self, "gemm_moe_grouped_mma_fp4_x4", (((2*mi/16) as u32), ngroups_x4, 1), (256,1,1), 0,
                                (d(&gu_p), *guq.device_ptr() as u64, *gus.device_ptr() as u64, d(gugs), d(&x_perm),
                                 *g.tile_e.device_ptr() as u64, (2*mi) as i32, h as i32, e_base,
                                 *g.poff.device_ptr() as u64, ne as i32));
                        } else {
                            blaunch!(self, "gemm_moe_grouped_mma_fp4", (((2*mi/16) as u32), ngroups_max, 1), (256,1,1), 0,
                                (d(&gu_p), *guq.device_ptr() as u64, *gus.device_ptr() as u64, d(gugs), d(&x_perm),
                                 *g.tile_e.device_ptr() as u64, (2*mi) as i32, h as i32, e_base,
                                 *g.poff.device_ptr() as u64, ne as i32));
                        }
                    }
                    }
                    // Fused-quant path: the fused dn kernel computes silu in its stage from gu_p
                    // directly; h_p (and its silu launch) vanish on the fused path (§8).
                    let dn_native_fused = self.mxfp4.as_ref().map_or(false, |st| {
                        !st.allow_bf16.read().unwrap().contains(&(*dnq.device_ptr() as u64)) && pf_fused
                    });
                    let h_p: Option<B> = if dn_native_fused { None } else { Some(pool.get_bf16(ppad_cap * mi)) };
                    if let Some(h_p) = &h_p {
                        if self.igs_armed() { self.memset_compute_stream(d(h_p), ppad_cap * mi * 2); }
                        blaunch!(self, "moe_silu_bf16_b", grid(ppad_cap * mi), (256,1,1), 0,
                            (d(h_p), d(&gu_p), mi as i32, ppad_cap as i32,
                             *g.poff.device_ptr() as u64, ne as i32));
                        if self.rotated.contains(&(*dnq.device_ptr() as u64)) { self.rot_inplace(h_p, ppad_cap * mi); }
                        self.igs_tap_x(*dnq.device_ptr() as u64, h_p, ppad_cap * mi);
                    } else {
                        assert!(!self.rotated.contains(&(*dnq.device_ptr() as u64)), "MR-GPTQ rotation is not supported on the fused-silu down path");
                    }
                    let dn_p = pool.get_bf16(ppad_cap * h);
                    if let Some(w4) = w4_moe {
                        self.w4a4_moe(w4, h_p.as_ref().unwrap(), ppad_cap, mi, *dnq.device_ptr() as u64, *dns.device_ptr() as u64,
                                      *dngs.device_ptr() as u64, h, &dn_p, g, ne, e_base, false, w4.xgs(*dnq.device_ptr() as u64));
                        if crate::w4a4::check_on() {
                            let r = pool.get_bf16(ppad_cap * h);
                            let xf = self.w4a4_fakequant(w4, h_p.as_ref().unwrap(), ppad_cap, mi, w4.xgs(*dnq.device_ptr() as u64));
                            blaunch!(self, "gemm_moe_grouped_mma_fp4_x4", (((h/16) as u32), (ppad_cap / 32) as u32, 1), (256,1,1), 0,
                                (d(&r), *dnq.device_ptr() as u64, *dns.device_ptr() as u64, d(dngs), d(&xf),
                                 *g.tile_e.device_ptr() as u64, h as i32, mi as i32, e_base,
                                 *g.poff.device_ptr() as u64, ne as i32));
                            let real = { let p: Vec<i32> = self.dev.dtoh_sync_copy(&g.poff).unwrap(); p[ne] as usize };
                            self.w4a4_compare(&dn_p, &r, real, h, &format!("moe down rows={real}"));
                            pool.release_bf16(r, ppad_cap * h);
                        }
                    } else {
                    if let Some(st) = &self.mxfp4 {
                        if !st.allow_bf16.read().unwrap().contains(&(*dnq.device_ptr() as u64)) {
                            let nks_dn = (mi / 64) as i32;
                            let (dn_a, dn_s2) = st.w.get(&(*dnq.device_ptr() as u64)).unwrap_or_else(|| {
                                panic!("mxfp4: MoE down @{:x} has no OMMA repack", *dnq.device_ptr() as u64) }).ptrs();
                            if pf_fused {
                                // FUSED (F3, XSILU=1): silu computed in the dn stage from gu_p with
                                // moe_silu_bf16_b's exact math — ONE launch, no h_p, no silu kernel.
                                unsafe {
                                    st.moe_grouped_fused1.clone().launch_on_stream(&self.stream,
                                        LaunchConfig { grid_dim: ((h/16) as u32, ngroups_max, 1), block_dim: (256,1,1), shared_mem_bytes: 0 },
                                        (d(&dn_p), dn_a, dn_s2, d(dngs), d(&gu_p),
                                         (h/16) as i32, nks_dn, mi as i32, e_base,
                                         *g.tile_e.device_ptr() as u64, *g.poff.device_ptr() as u64, ne as i32))
                                        .unwrap_or_else(|e| panic!("mxfp4 moe_grouped_fused (dn): {e:?}"));
                                }
                            } else {
                                let ng_quant = (ppad_cap / 8) as u32;
                                unsafe {
                                    st.quant_ng.clone().launch_on_stream(&self.stream,
                                        LaunchConfig { grid_dim: (ng_quant * nks_dn as u32, 1, 1), block_dim: (256,1,1), shared_mem_bytes: 0 },
                                        (d(h_p.as_ref().unwrap()), mi as i32, ng_quant as i32, nks_dn, 8i32, 8i32,
                                         d(&st.moe_bp), d(&st.moe_sfb),
                                         *g.poff.device_ptr() as u64, ne as i32)).unwrap_or_else(|e| panic!("mxfp4 quant_ng: {e:?}"));
                                    st.moe_gemv.clone().launch_on_stream(&self.stream,
                                        LaunchConfig { grid_dim: ((h/16) as u32, ngroups_max, 1), block_dim: (256,1,1), shared_mem_bytes: 0 },
                                        (d(&dn_p), dn_a, dn_s2, d(dngs), d(&st.moe_bp), d(&st.moe_sfb),
                                         (h/16) as i32, nks_dn, e_base,
                                         *g.tile_e.device_ptr() as u64, *g.poff.device_ptr() as u64, ne as i32))
                                        .unwrap_or_else(|e| panic!("mxfp4 moe_gemv (dn): {e:?}"));
                                }
                            }
                        } else {
                            if wide_grouped {
                                blaunch!(self, "gemm_moe_grouped_mma_fp4_x4", (((h/16) as u32), ngroups_x4, 1), (256,1,1), 0,
                                    (d(&dn_p), *dnq.device_ptr() as u64, *dns.device_ptr() as u64, d(dngs), d(h_p.as_ref().unwrap()),
                                     *g.tile_e.device_ptr() as u64, h as i32, mi as i32, e_base,
                                     *g.poff.device_ptr() as u64, ne as i32));
                            } else {
                                blaunch!(self, "gemm_moe_grouped_mma_fp4", (((h/16) as u32), ngroups_max, 1), (256,1,1), 0,
                                    (d(&dn_p), *dnq.device_ptr() as u64, *dns.device_ptr() as u64, d(dngs), d(h_p.as_ref().unwrap()),
                                     *g.tile_e.device_ptr() as u64, h as i32, mi as i32, e_base,
                                     *g.poff.device_ptr() as u64, ne as i32));
                            }
                        }
                    } else {
                        if wide_grouped {
                            blaunch!(self, "gemm_moe_grouped_mma_fp4_x4", (((h/16) as u32), ngroups_x4, 1), (256,1,1), 0,
                                (d(&dn_p), *dnq.device_ptr() as u64, *dns.device_ptr() as u64, d(dngs), d(h_p.as_ref().unwrap()),
                                 *g.tile_e.device_ptr() as u64, h as i32, mi as i32, e_base,
                                 *g.poff.device_ptr() as u64, ne as i32));
                        } else {
                            blaunch!(self, "gemm_moe_grouped_mma_fp4", (((h/16) as u32), ngroups_max, 1), (256,1,1), 0,
                                (d(&dn_p), *dnq.device_ptr() as u64, *dns.device_ptr() as u64, d(dngs), d(h_p.as_ref().unwrap()),
                                 *g.tile_e.device_ptr() as u64, h as i32, mi as i32, e_base,
                                 *g.poff.device_ptr() as u64, ne as i32));
                        }
                    }
                    }
                    blaunch!(self, "moe_combine_grouped_b", grid(ns * h), (256,1,1), 0,
                        (out_s, d(&dn_p), *g.perm_wt.device_ptr() as u64, *g.inv_pos.device_ptr() as u64,
                         h as i32, k as i32, ns as i32));
                    pool.release_bf16(x_perm, ppad_cap*h); pool.release_bf16(gu_p, ppad_cap*2*mi);
                    if let Some(h_p) = h_p { pool.release_bf16(h_p, ppad_cap*mi); }
                    pool.release_bf16(dn_p, ppad_cap*h);
                    }
                    t0 += ns;
                }
              }
              // E12: the fold's sigmoid gate logits (None for hy_v3's ungated shared expert).
              if let Some(sg) = sgate.take() { pool.release_bf16(sg, batch); }
            }
            _ => panic!("MoE experts: gate_up and down must both be bf16 or both Nvfp4Raw"),
        }

        // TP=2 expert-parallel: `out` holds only THIS rank's experts (remote slots are exact zeros on
        // both paths — the decode kernel zeroes remote (token,slot) rows; the prefill combine skips
        // inv_pos == -1 pairs), so it is a PARTIAL. ONE all-reduce restores the full Σ. It MUST run
        // after the expert combine and BEFORE the shared-expert add — because the shared expert is
        // replicated: reducing after the shared add would double it. E8 (fold_shared): the routed
        // PARTIAL stays un-reduced here — the shared PARTIAL is combined into it and ONE reduce
        // (inside the fold branch, step 3) restores both sums. Same call shape as mlp_batch's
        // row-parallel epilogue: elementwise, chunked to the ring payload (both ranks compute the same
        // chunk count from the same n), canonical rank0+rank1 order → bit-identical across ranks.
        // bf16 partials only: the combine kernels emit bf16, so there is no unrounded FP32 accumulator
        // to preserve — the exact analog of the dense path's wide-batch bf16 fallback.
        let fold_shared = moe.experts_sharded && moe.shared_sharded;
        // E8 FP32-partial rule at the shared-dispatch tail: decode/verify widths only (batch <=
        // MAX_VERIFY — the binv/LOSSLESS/acceptance regime) and only when the shared down runs the
        // bf16 chain (gemm_act_f32 has no mxfp4-native arm). Same expression as the fold-disable
        // condition in the NVFP4 arm above — the two must never disagree.
        let fp32_shared = fold_shared && batch <= MAX_VERIFY
            && self.mxfp4.as_ref().map_or(true, |st| {
                let ab = st.allow_bf16.read().unwrap();
                ab.contains(&moe.shared.down.packed_ptr())
            });
        // E12: with the fold, the shared expert was already computed as a slot inside the grouped
        // launches above (including the reduce placement — the fold_shared partial folded pre-reduce,
        // the !fold_shared post-reduce add), so the old separate-launch shared MLP is skipped.
        if !fold {
        if moe.experts_sharded && !fold_shared {
            self.tp_all_reduce_bf16(&mut out, h * batch);
        }

        // 3. Shared expert: standard MLP (intermediate = si). qwen3_5_moe gates it by
        // sigmoid(shared_gate·x); hy_v3 has NO gate tensor — the shared expert is a plain add.
        // When the loader fused gate+up (hy_v3 quantized), it's ONE [2*si, h] GEMM instead of two —
        // the [si, h] shapes are small-M latency-bound (~115 GB/s); `moe_silu_bf16_b` == `silu_mul_b`.
        // E8 (revised): with the shared expert SHARDED the fused gate+up is the rank's paired
        // [2·si_local, h] half (gate[i]·up[i] on one rank) and the down is row-parallel [h, si_local]
        // — the act is ALREADY the rank's local si_local rows (0-based, the row-parallel
        // convention), so the down reads it directly and the local K-partial joins the routed
        // all-reduce below. `fold_shared` was computed before the routed reduce.
        let si_local = if fold_shared { si / self.shared_expert_shard_factor().max(1) } else { si };
        // The rank-local act (bf16 — input-side; the FP32-partial rule governs the DOWN partial).
        // Sharded: the rank's own si_local rows. Replicated: the full si.
        let act: B = if let Some(gu) = &moe.shared_gate_up {
            let mut gu_buf = pool.get_bf16(2 * si_local * batch);
            self.gemm_act(gu, x, &mut gu_buf, h, 2 * si_local, batch);
            let mut a = pool.get_bf16(si_local * batch);
            blaunch!(self, "moe_silu_bf16_b", grid(si_local*batch), (256,1,1), 0,
                (d(&a), d(&gu_buf), si_local as i32, batch as i32, 0u64, 0i32));
            pool.release_bf16(gu_buf, 2 * si_local * batch);
            a
        } else {
            let mut sg = pool.get_bf16(si * batch);
            let mut su = pool.get_bf16(si * batch);
            self.gemm_act(&moe.shared.gate, x, &mut sg, h, si, batch);
            self.gemm_act(&moe.shared.up,   x, &mut su, h, si, batch);
            blaunch!(self, "silu_mul_b", grid(si*batch), (256,1,1), 0, (d(&sg), d(&sg), d(&su), (si*batch) as i32));
            pool.release_bf16(su, si*batch);
            if fold_shared {
                // row-parallel down reads THIS rank's half of the full act: copy the half into a
                // fresh buffer (local rows 0..si_local, the row-parallel convention) and reuse the
                // normal gemm_act dispatch (every format and batch width). Offset in BYTES:
                // rank·si_local·batch elements × 2.
                let mut act_half = pool.get_bf16(si_local * batch);
                let src = d(&sg) + (self.tp_rank.max(0) as u64) * (si_local as u64 * batch as u64 * 2);
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(*act_half.device_ptr() as cudarc::driver::sys::CUdeviceptr,
                        src as cudarc::driver::sys::CUdeviceptr, si_local * batch * 2, self.stream.stream).unwrap();
                }
                pool.release_bf16(sg, si*batch);
                act_half
            } else {
                sg
            }
        };
        // The shared contribution + its reduce placement. E8: the shared PARTIAL joins the routed
        // PARTIAL pre-reduce (the gate is a replicated per-token scalar; the combine is
        // elementwise, so ordering it pre-reduce is legal). The fold reassociates the fp32 sum vs
        // the replicated path (shared add moves pre-reduce) — the documented E8 numerics change,
        // gated by the lossless/acceptance battery.
        if fp32_shared {
            // E8 FP32-partial rule (decode/verify — the binv/LOSSLESS/acceptance regime): the
            // shared down partial crosses the rank boundary UNROUNDED (gemm_act_f32's Cf
            // accumulator — the dense row-parallel pattern), the rank's routed partial (bf16 in
            // `out`, exact in fp32) joins it in fp32, and ONE fp32 all-reduce (tp_all_reduce_fp32
            // mode 1: f2b(rank0 + rank1), fixed chunked order, no atomics) applies the single
            // rounding boundary for the whole shared contribution. The fold's shared slot is
            // disabled at these widths on purpose — its combine rounds per-rank AFTER the shared
            // add (a bf16 partial round: the e7950d7 acceptance disease).
            let mut shared_f = pool.get(h * batch);
            self.gemm_act_f32(&moe.shared.down, &act, &mut shared_f, si_local, h, batch);
            blaunch!(self, "add_f32_bf16_b", grid(h*batch), (256,1,1), 0,
                (d(&shared_f), d(&out), (h*batch) as i32));
            if fuse {
                if let Some(fl) = self.tp_all_reduce_fp32_k1(&mut out, &shared_f, h * batch) {
                    match fl {
                        FusedLocal::F32Scratch => { pool.release(shared_f, h * batch); fused_out = Some(FusedPartial::F32Scratch); }
                        _ => fused_out = Some(FusedPartial::F32(shared_f)),   // caller releases shared_f
                    }
                } else {
                    pool.release(shared_f, h * batch);               // chunked — full chain ran
                }
            } else {
                self.tp_all_reduce_fp32(&mut out, &shared_f, h * batch);
                pool.release(shared_f, h * batch);
            }
        } else if fold_shared {
            // Wide-prefill bf16 fold_shared: routed PARTIAL + bf16 shared PARTIAL → ONE bf16
            // all-reduce (prefill reassociation is accepted — the dense path's rule, and
            // gemm_act_f32 caps at batch <= MAX_VERIFY).
            let mut shared_out = pool.get_bf16(h * batch);
            self.gemm_act(&moe.shared.down, &act, &mut shared_out, si_local, h, batch);
            match &moe.shared_gate {
                Some(sgw) => {
                    let mut sgate = pool.get_bf16(batch);
                    self.gemm_act_binv(sgw, x, &mut sgate, h, 1, batch);
                    blaunch!(self, "moe_shared_combine_b", grid(h*batch), (256,1,1), 0,
                        (d(&out), d(&shared_out), d(&sgate), h as i32, batch as i32));
                    pool.release_bf16(sgate, batch);
                }
                None => {
                    let tot = h * batch;
                    blaunch!(self, "add_residual_b", grid(tot), (256,1,1), 0,
                        (d(&out), d(&out), d(&shared_out), tot as i32));
                }
            }
            if fuse {
                if let Some(fl) = self.tp_all_reduce_bf16_k1(&mut out, h * batch) {
                    fused_out = Some(match fl {
                        FusedLocal::F32Scratch => FusedPartial::F32Scratch,
                        _ => FusedPartial::Bf16,
                    });
                }
            } else {
                self.tp_all_reduce_bf16(&mut out, h * batch);
            }
            pool.release_bf16(shared_out, h*batch);
        } else {
            // !fold_shared: the routed partial was ALREADY reduced above (moe.experts_sharded
            // && !fold_shared) or was never a partial; the replicated shared add is post-reduce.
            let mut shared_out = pool.get_bf16(h * batch);
            self.gemm_act(&moe.shared.down, &act, &mut shared_out, si_local, h, batch);
            match &moe.shared_gate {
                Some(sgw) => {
                    let mut sgate = pool.get_bf16(batch);
                    self.gemm_act_binv(sgw, x, &mut sgate, h, 1, batch);
                    blaunch!(self, "moe_shared_combine_b", grid(h*batch), (256,1,1), 0,
                        (d(&out), d(&shared_out), d(&sgate), h as i32, batch as i32));
                    pool.release_bf16(sgate, batch);
                }
                None => {
                    let tot = h * batch;
                    blaunch!(self, "add_residual_b", grid(tot), (256,1,1), 0,
                        (d(&out), d(&out), d(&shared_out), tot as i32));
                }
            }
            pool.release_bf16(shared_out, h*batch);
        }
        pool.release_bf16(act, si_local * batch);
        } // !fold — E12: the shared slot path handled the shared expert above.
        match fused_out {
            Some(FusedPartial::F32(p)) => ReduceOut::Fused(FusedReduce { sink: out, partial: Some(p), mode: 1 }),
            Some(FusedPartial::F32Scratch) => ReduceOut::Fused(FusedReduce { sink: out, partial: None, mode: 1 }),
            Some(FusedPartial::Bf16) => ReduceOut::Fused(FusedReduce { sink: out, partial: None, mode: 0 }),
            None => ReduceOut::Full(out),
        }
    }
}
