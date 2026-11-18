//! MoE sublayer pieces (router, shared / routed experts, combine) and the `block_forward*` entry points of `Dsv4AttnRuntime`.
//! Siblings: `attn.rs` / `ops.rs` provide the attention half and building blocks; `mod.rs` has the map.

use super::*;

impl Dsv4AttnRuntime {
    /// Router on the UN-simmed x (fp32 purity, §9.4/§12.A.5). Returns (sel [s,topk] global expert
    /// ids, router_w [s,topk] fp32). Replicated+identical on both TP ranks (gate_w/bias/tid2eid are
    /// replicated; the input x is identical post-attention) — so the routed-expert SELECTION is the
    /// same on every rank; only the routed-expert COMPUTATION differs (each rank's band).
    /// R3.1: scores/biased ride the persistent workspace (sel/router_w stay per-call — they are
    /// returned across dispatch boundaries, e.g. block_forward_tp_sim shares them across ranks).
    pub fn moe_router<X: dsv4_gpu::Dsv4Buf<bf16>, F: dsv4_gpu::Dsv4Buf<f32>, I: dsv4_gpu::Dsv4Buf<i32>>(
        &self,
        layer: &Dsv4GpuLayer,
        x: &X,
        s: usize,
        ids: &CudaSlice<i32>,
        cfg: &dsv4_load::Dsv4Config,
        scratch: &mut MoeGroupedScratch,
    ) -> Result<(I, F)> {
        let (dim, ne, topk) = (cfg.dim, cfg.n_routed_experts, cfg.n_activated_experts);
        let s_i = s as i32;
        let ws = scratch.dsv4_ws_mut();
        anyhow::ensure!(s <= ws.s_cap, "moe_router: s {s} > workspace s_cap {}", ws.s_cap);
        let scores = &mut ws.scores;
        let (dim_i, ne_i, tk_i) = (dim as i32, ne as i32, topk as i32);
        dsv4_launch!(self.spine, "dsv4_router_score_b", self.stream.stream, (ne as u32, s as u32, 1), (256, 1, 1), 0,
            (scores, &layer.gate_w, x, &s_i, &dim_i, &ne_i))?;
        let sel = I::alloc_zeros(&self.dev, self.stream.stream, s * topk)?;
        match (&layer.tid2eid, &layer.gate_bias) {
            (Some(tbl), _) => {
                dsv4_launch!(self.spine, "dsv4_router_tid2eid_b", self.stream.stream, (ceil256(s * topk), 1, 1), (256, 1, 1), 0,
                    (&sel, tbl, ids, &s_i, &tk_i))?;
            }
            (None, Some(bias)) => {
                let biased = &mut ws.biased;
                dsv4_launch!(self.spine, "dsv4_router_bias_add_b", self.stream.stream, (ceil256(s * ne), 1, 1), (256, 1, 1), 0,
                    (biased, &*scores, bias, &s_i, &ne_i))?;
                dsv4_launch!(self.spine, "dsv4_topk", self.stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
                    (&*biased, &sel, &s_i, &ne_i, &tk_i))?;
            }
            (None, None) => {
                dsv4_launch!(self.spine, "dsv4_topk", self.stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
                    (&*scores, &sel, &s_i, &ne_i, &tk_i))?;
            }
        }
        let router_w = F::alloc_zeros(&self.dev, self.stream.stream, s * topk)?;
        let rs = cfg.route_scale;
        dsv4_launch!(self.spine, "dsv4_router_weights_b", self.stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
            (&router_w, &*scores, &sel, &s_i, &ne_i, &tk_i, &rs))?;
        Ok((sel, router_w))
    }

    /// Shared expert (replicated) on the pristine x (§B.9: shared input is the original bf16 x)
    /// into `ws.sh_out`. Identical on both TP ranks — it is added AFTER the routed all-reduce
    /// (reducing after the add would double it; see `gpu.rs:2577` for the proven Hy3 reference).
    /// R3.1: all intermediates ride the workspace (zero per-call allocs on the serving path).
    pub fn moe_shared_into<X: dsv4_gpu::Dsv4Buf<bf16>>(&self, layer: &Dsv4GpuLayer, x: &X, s: usize, cfg: &dsv4_load::Dsv4Config, ws: &mut crate::gpu::Dsv4MoeWorkspace) -> Result<()> {
        let (dim, inter) = (cfg.dim, cfg.moe_inter_dim);
        anyhow::ensure!(s <= ws.s_cap, "moe_shared_into: s {s} > workspace s_cap {}", ws.s_cap);
        self.quant_g128_into(&mut ws.sh_q, &mut ws.sh_qs, x, s, dim)?;
        self.fp8_bsb_rows_into(&mut ws.sh_gu, &layer.sh_gu, &ws.sh_q, &ws.sh_qs, s)?; // [s, 2*inter] gate|up
        let (inter_i, limit, s_i) = (inter as i32, cfg.swiglu_limit, s as i32);
        dsv4_launch!(self.spine, "dsv4_swiglu_clamp_shared", self.stream.stream, (ceil256(s * inter), 1, 1), (256, 1, 1), 0,
            (&mut ws.sh_h, &ws.sh_gu, &limit, &inter_i, &s_i))?;
        self.quant_g128_into(&mut ws.sh_q2, &mut ws.sh_qs2, &ws.sh_h, s, inter)?;
        self.fp8_bsb_rows_into(&mut ws.sh_out, &layer.sh_w2, &ws.sh_q2, &ws.sh_qs2, s)?; // [s, dim]
        Ok(())
    }

    /// Shared expert (replicated), allocating variant — probes/one-shots. Production uses
    /// [`moe_shared_into`](Self::moe_shared_into) on the workspace.
    pub fn moe_shared(&self, layer: &Dsv4GpuLayer, x: &B, s: usize, cfg: &dsv4_load::Dsv4Config) -> Result<B> {
        let (dim, inter) = (cfg.dim, cfg.moe_inter_dim);
        let (shc, shsa) = self.quant_g128::<B, CudaSlice<u8>>(x, s, dim)?;
        let gu = self.fp8_bsb_rows::<B, CudaSlice<u8>>(&layer.sh_gu, &shc, &shsa, s)?; // [s, 2*inter] gate|up
        let h = self.dev.alloc_zeros::<bf16>(s * inter)?;
        let (inter_i, limit, s_i) = (inter as i32, cfg.swiglu_limit, s as i32);
        dsv4_launch!(self.spine, "dsv4_swiglu_clamp_shared", self.stream.stream, (ceil256(s * inter), 1, 1), (256, 1, 1), 0,
            (&h, &gu, &limit, &inter_i, &s_i))?;
        let (hc, hsa) = self.quant_g128::<B, CudaSlice<u8>>(&h, s, inter)?;
        Ok(self.fp8_bsb_rows(&layer.sh_w2, &hc, &hsa, s)?) // [s, dim]
    }

    /// Routed experts (the TP=2 rank-LOCAL partial) into the workspace's `routed_out`. `sel`/
    /// `router_w` from [`moe_router`]; the runtime sims its input in place → staged chunk copies
    /// (x stays pristine). Under TP=2 each rank's `Dsv4MoeGpu` holds only its `[e_base,
    /// e_base+e_span)` band, so remote (token,slot) pairs contribute exact zeros — this output is
    /// a PARTIAL summed by the all-reduce. R3.1: chunk staging + the expert helpers run entirely
    /// on the compute stream against the persistent workspace — NO host syncs, NO per-call allocs
    /// (the old path did 3 `synchronize()` + ~11 allocs + an htod per dispatch per layer).
    pub fn moe_routed<X: dsv4_gpu::Dsv4Buf<bf16>, F: dsv4_gpu::Dsv4Buf<f32>, I: dsv4_gpu::Dsv4Buf<i32>>(
        &self,
        layer: &Dsv4GpuLayer,
        scratch: &mut MoeGroupedScratch,
        x: &X,
        s: usize,
        sel: &I,
        router_w: &F,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<()> {
        let (dim, topk) = (cfg.dim, cfg.n_activated_experts);
        anyhow::ensure!(s <= scratch.dsv4_ws_ref().s_cap, "moe_routed: s {s} > workspace s_cap");
        // R3A.4: full-width dispatches of up to d_cap rows instead of 128 sequential 16-row
        // dispatches — each touched expert's weights are read ceil(rows_e/16) times per
        // dispatch instead of once per 16-row dispatch (~21x less expert-weight traffic; the
        // grouped GEMM's per-element K-order is ppad-independent, so outputs are
        // bitwise-identical to the 16-row decomposition — gated). s > d_cap (one-shot
        // probes, e.g. the 32k §12.B.5 reference) loops d_cap chunks of the same form.
        let d_cap = gpu::MOE_D_CAP.min(scratch.p_max() / topk);
        anyhow::ensure!(d_cap >= 1, "moe_routed: scratch p_max {} < topk", scratch.p_max());
        let mut r0 = 0usize;
        while r0 < s {
            let n = (s - r0).min(d_cap);
            {
                let ws = scratch.dsv4_ws_mut();
                unsafe {
                    result::memcpy_dtod_async(*ws.xc.device_ptr(), x.dptr() + (r0 * dim * 2) as u64, n * dim * 2, self.stream.stream)
                        .map_err(|e| anyhow!("d2d xc stage: {e}"))?;
                    result::memcpy_dtod_async(*ws.idc.device_ptr(), sel.dptr() + (r0 * topk * 4) as u64, n * topk * 4, self.stream.stream)
                        .map_err(|e| anyhow!("d2d idc stage: {e}"))?;
                    result::memcpy_dtod_async(*ws.wtc.device_ptr(), router_w.dptr() + (r0 * topk * 4) as u64, n * topk * 4, self.stream.stream)
                        .map_err(|e| anyhow!("d2d wtc stage: {e}"))?;
                }
            }
            if n == 1 {
                gpu::dsv4_moe_experts_n1_ws(&self.dev, &self.stream, &self.bk, &self.df, &layer.moe,
                    scratch.dsv4_ws_mut(), 1, topk, cfg.swiglu_limit)?;
            } else {
                gpu::dsv4_moe_experts_grouped_ws(&self.dev, &self.stream, &self.bk, &self.df, &layer.moe,
                    scratch, n, topk, cfg.swiglu_limit)?;
            }
            {
                let ws = scratch.dsv4_ws_mut();
                unsafe {
                    result::memcpy_dtod_async(*ws.routed_out.device_ptr() + (r0 * dim * 2) as u64, *ws.outc.device_ptr(), n * dim * 2, self.stream.stream)
                        .map_err(|e| anyhow!("d2d out fold: {e}"))?;
                }
            }
            r0 += n;
        }
        Ok(())
    }

    /// `ffn_out = bf16(fp32(routed) + fp32(shared))` — the post-all-reduce combine (one bf16 RNE).
    pub fn ffn_combine<X: dsv4_gpu::Dsv4Buf<bf16>>(&self, routed: &B, shared: &B, s: usize, dim: usize) -> Result<X> {
        let ffn_out = X::alloc_zeros(&self.dev, self.stream.stream, s * dim)?;
        let total_i = (s * dim) as i32;
        let cfg_launch = LaunchConfig { grid_dim: (ceil256(s * dim), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let out_v = ffn_out.view(0, s * dim);
        unsafe {
            self.bk["add_residual_b"]
                .clone()
                .launch_on_stream(&self.stream, cfg_launch, (&out_v, routed, shared, total_i))
                .map_err(|e| anyhow!("add_residual_b: {e:?}"))?;
        }
        Ok(ffn_out)
    }

    /// FFN sublayer (§B.9): router (on the un-simmed x) → shared expert + routed experts
    /// (G2 runtime, ≤16-row chunks) → residual add. `x` [s, dim] bf16 is never mutated
    /// (routed chunks sim private copies). Returns (ffn_out, router_w [s,6], router_idx [s,6]).
    /// Single-process path (no TP). Under TP=2 use [`moe_router`]+[`moe_routed`]→all-reduce→
    /// [`moe_shared`]+[`ffn_combine`] (see `block_forward_tp`).
    pub fn moe_forward<X: dsv4_gpu::Dsv4Buf<bf16>, F: dsv4_gpu::Dsv4Buf<f32>, I: dsv4_gpu::Dsv4Buf<i32>>(
        &self,
        layer: &Dsv4GpuLayer,
        scratch: &mut MoeGroupedScratch,
        x: &X,
        s: usize,
        ids: &CudaSlice<i32>,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<(X, F, I)> {
        let (sel, router_w) = self.moe_router(layer, x, s, ids, cfg, scratch)?;
        self.moe_shared_into(layer, x, s, cfg, scratch.dsv4_ws_mut())?;
        self.moe_routed(layer, scratch, x, s, &sel, &router_w, cfg)?;
        let ffn_out = {
            let ws = scratch.dsv4_ws_ref();
            self.ffn_combine(&ws.routed_out, &ws.sh_out, s, cfg.dim)?
        };
        Ok((ffn_out, router_w, sel))
    }

    /// Single-node TP=2 simulation block forward (the model-side TP validator, no transport).
    /// Mirrors [`block_forward`](Self::block_forward) but splits the FFN at the routed|shared
    /// boundary: rank_a's attention runs (advancing `state_a` — attention is replicated, so rank_a
    /// drives it; rank_b's state is untouched), then BOTH ranks' routed-expert partials are computed
    /// on the identical `y2n` and SUMMED (the doorbell all-reduce, simulated here as a bf16 add),
    /// then the replicated shared expert is added, then `hc_post`. Returns the TP block output `x3`.
    /// `layer_b`/`scratch_b` are rank_b's; only rank_b's MoE band is read (its attn/mHC/shared are
    /// identical to rank_a's and unused). Compare the result to a full-256 forward to validate the
    /// §5 sharding + the all-reduce-before-shared boundary compounding across layers.
    #[allow(clippy::too_many_arguments)]
    pub fn block_forward_tp_sim(
        &self,
        layer_a: &Dsv4GpuLayer,
        state_a: &mut Dsv4AttnState,
        scratch_a: &mut MoeGroupedScratch,
        layer_b: &Dsv4GpuLayer,
        scratch_b: &mut MoeGroupedScratch,
        x: &B,
        s: usize,
        start_pos: usize,
        ids: &CudaSlice<i32>,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<B> {
        // --- attention sublayer (rank_a; replicated, so rank_a drives it) ---
        let (y, posts, combs) = self.hc_pre::<B, S>(x, s, &layer_a.hc_attn_fn, &layer_a.hc_attn_base, &layer_a.hc_attn_scale, cfg)?;
        let yn = self.rmsnorm(&y, &layer_a.attn_norm, s, cfg.dim, cfg.norm_eps)?;
        let (attn_out, _topk_idx, _topk_t) = self.attn_forward::<B, S, CudaSlice<i32>, CudaSlice<u8>, CudaSlice<u32>>(layer_a, state_a, &yn, s, start_pos, cfg)?;
        let x2 = self.hc_post(&attn_out, x, &posts, &combs, s, cfg)?;
        // --- ffn sublayer: router (replicated) → routed partials (rank-local) → ALL-REDUCE → +shared → hc_post ---
        let (y2, posts2, combs2) = self.hc_pre::<B, S>(&x2, s, &layer_a.hc_ffn_fn, &layer_a.hc_ffn_base, &layer_a.hc_ffn_scale, cfg)?;
        let y2n = self.rmsnorm(&y2, &layer_a.ffn_norm, s, cfg.dim, cfg.norm_eps)?;
        let (sel, router_w) = self.moe_router::<B, S, CudaSlice<i32>>(layer_a, &y2n, s, ids, cfg, scratch_a)?;
        self.moe_routed(layer_a, scratch_a, &y2n, s, &sel, &router_w, cfg)?;
        self.moe_routed(layer_b, scratch_b, &y2n, s, &sel, &router_w, cfg)?;
        // ALL-REDUCE (simulated): routed_sum = bf16(routed_a + routed_b). The doorbell's tp_wait_add
        // does this same fp32-add-then-bf16-round in-kernel; here it's add_residual_b on the host stream.
        let routed_sum = self.ffn_combine(&scratch_a.dsv4_ws_ref().routed_out, &scratch_b.dsv4_ws_ref().routed_out, s, cfg.dim)?;
        self.moe_shared_into(layer_a, &y2n, s, cfg, scratch_a.dsv4_ws_mut())?;
        let ffn_out = self.ffn_combine(&routed_sum, &scratch_a.dsv4_ws_ref().sh_out, s, cfg.dim)?;
        let x3 = self.hc_post(&ffn_out, &x2, &posts2, &combs2, s, cfg)?;
        Ok(x3)
    }

    /// Full trunk `Block.forward` (§B.8 ordering): hc_pre → norm → sublayer → hc_post, twice.
    /// `x` [s, hc*dim] bf16 streams; `ids` [s] i32 (hash-router table lookups).
    pub fn block_forward<X: dsv4_gpu::Dsv4Buf<bf16>, F: dsv4_gpu::Dsv4Buf<f32>, I: dsv4_gpu::Dsv4Buf<i32>, C: dsv4_gpu::Dsv4Buf<u8>, U: dsv4_gpu::Dsv4Buf<u32>>(
        &self,
        layer: &Dsv4GpuLayer,
        st: &mut Dsv4AttnState,
        scratch: &mut MoeGroupedScratch,
        x: &X,
        s: usize,
        start_pos: usize,
        ids: &CudaSlice<i32>,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<BlockOut<X, F, I>> {
        // --- attention sublayer (replicated under TP — no cross-rank reduction) ---
        let (y, posts, combs) = self.hc_pre::<X, F>(x, s, &layer.hc_attn_fn, &layer.hc_attn_base, &layer.hc_attn_scale, cfg)?;
        let yn = self.rmsnorm(&y, &layer.attn_norm, s, cfg.dim, cfg.norm_eps)?;
        let (attn_out, topk_idx, topk_t) = self.attn_forward::<X, F, I, C, U>(layer, st, &yn, s, start_pos, cfg)?;
        let x2 = self.hc_post(&attn_out, x, &posts, &combs, s, cfg)?;
        // --- ffn sublayer ---
        let (y2, posts2, combs2) = self.hc_pre::<X, F>(&x2, s, &layer.hc_ffn_fn, &layer.hc_ffn_base, &layer.hc_ffn_scale, cfg)?;
        let y2n = self.rmsnorm(&y2, &layer.ffn_norm, s, cfg.dim, cfg.norm_eps)?;
        let (ffn_out, router_w, router_idx) = if self.tp_ctx_dptr != 0 {
            // TP=2: router (replicated) → routed PARTIAL (rank-local) → ALL-REDUCE → +shared (replicated).
            // Boundary = routed combine BEFORE the shared add (shared replicated; reducing after
            // double-counts it — the proven Hy3 pattern, gpu.rs:2577).
            let (sel, rw) = self.moe_router(layer, &y2n, s, ids, cfg, scratch)?;
            self.moe_routed(layer, scratch, &y2n, s, &sel, &rw, cfg)?;
            self.tp_all_reduce_bf16(&mut scratch.dsv4_ws_mut().routed_out, s * cfg.dim)?;
            self.moe_shared_into(layer, &y2n, s, cfg, scratch.dsv4_ws_mut())?;
            let ffn = {
                let ws = scratch.dsv4_ws_ref();
                self.ffn_combine(&ws.routed_out, &ws.sh_out, s, cfg.dim)?
            };
            (ffn, rw, sel)
        } else {
            self.moe_forward(layer, scratch, &y2n, s, ids, cfg)?
        };
        let x3 = self.hc_post(&ffn_out, &x2, &posts2, &combs2, s, cfg)?;
        Ok(BlockOut {
            y: x3,
            attn_out,
            ffn_out,
            router_w,
            router_idx,
            topk_idx,
            topk_t,
        })
    }

    /// Traced `Block.forward` for the batch-invariance localization probe: identical math to
    /// [`block_forward`](Self::block_forward), but also returns the POST-ATTENTION mid-block
    /// streams (`x2` after the attention hc_post) so the caller can tell whether a row-0
    /// divergence enters at the attention sublayer or the MoE/ffn sublayer. Returns
    /// `(mid_post_attn, final_post_ffn)`, both `[s, hc*dim]` bf16.
    pub fn block_forward_traced(
        &self,
        layer: &Dsv4GpuLayer,
        st: &mut Dsv4AttnState,
        scratch: &mut MoeGroupedScratch,
        x: &B,
        s: usize,
        start_pos: usize,
        ids: &CudaSlice<i32>,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<(B, B)> {
        let (y, posts, combs) = self.hc_pre::<B, S>(x, s, &layer.hc_attn_fn, &layer.hc_attn_base, &layer.hc_attn_scale, cfg)?;
        let yn = self.rmsnorm(&y, &layer.attn_norm, s, cfg.dim, cfg.norm_eps)?;
        let (attn_out, _topk_idx, _topk_t) = self.attn_forward::<B, S, CudaSlice<i32>, CudaSlice<u8>, CudaSlice<u32>>(layer, st, &yn, s, start_pos, cfg)?;
        let mid = self.hc_post(&attn_out, x, &posts, &combs, s, cfg)?;
        let (y2, posts2, combs2) = self.hc_pre::<B, S>(&mid, s, &layer.hc_ffn_fn, &layer.hc_ffn_base, &layer.hc_ffn_scale, cfg)?;
        let y2n = self.rmsnorm(&y2, &layer.ffn_norm, s, cfg.dim, cfg.norm_eps)?;
        let (ffn_out, _router_w, _router_idx) = self.moe_forward::<B, S, CudaSlice<i32>>(layer, scratch, &y2n, s, ids, cfg)?;
        let out = self.hc_post(&ffn_out, &mid, &posts2, &combs2, s, cfg)?;
        Ok((mid, out))
    }
}
