//! DSpark stage attention of `Dsv4AttnRuntime`: ring warm, forward (host and device variants) and capture.
//! Siblings: `attn.rs` is the trunk analogue; `mod.rs` has the map.

use super::*;

impl Dsv4AttnRuntime {
    /// DSpark warm (prefill) ring write (§B.10 DSparkAttention, start_pos==0 branch): compute
    /// main_kv = kv_norm(wkv(main_x)) for ALL `s` trunk positions, RoPE 0..s−1, FP8-sim the
    /// nope dims, write into the stage's 128-ring (rotated last-128 for s>win). NO attention,
    /// NO FFN — the warm pass only primes the ring. Each of the 3 stages warms with the SAME
    /// main_x (computed once from main_hidden by the caller).
    pub fn dspark_attn_warm(
        &self,
        layer: &Dsv4GpuLayer,
        st: &mut Dsv4AttnState,
        main_x: &B,
        s: usize,
        rope: &DevRope,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<()> {
        self.dspark_attn_warm_range(layer, st, main_x, s, 0, rope, cfg)
    }

    /// Range-position warm: like [`dspark_attn_warm`] but RoPE + ring-write at positions
    /// `start_pos .. start_pos+s-1` (the DSpark re-prime after a verify — writes main_kv for the
    /// committed verify positions so the draft ring stays contiguous, AGENTS §2.6).
    pub fn dspark_attn_warm_range(
        &self,
        layer: &Dsv4GpuLayer,
        st: &mut Dsv4AttnState,
        main_x: &B,
        s: usize,
        start_pos: usize,
        rope: &DevRope,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<()> {
        let (dim, hd, rd, win, eps) =
            (cfg.dim, cfg.head_dim, cfg.rope_head_dim, cfg.window_size, cfg.norm_eps);
        let (mxc, mxsa) = self.quant_g128(main_x, s, dim)?;
        let mkv = self.fp8_bsb_rows::<B, CudaSlice<u8>>(&layer.wkv, &mxc, &mxsa, s)?;
        let mkv = self.rmsnorm(&mkv, &layer.kv_norm, s, hd, eps)?;
        let pos_mk: Vec<i32> = (0..s).map(|i| (start_pos + i) as i32).collect();
        let pos_mk_dev = self.dev.htod_sync_copy(&pos_mk)?;
        let (s_i, hd_i, rd_i, sp_i) = (s as i32, hd as i32, rd as i32, start_pos as i32);
        dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(s * 32), 1, 1), (256, 1, 1), 0,
            (&mkv, &rope.cos, &rope.sin, &pos_mk_dev, &s_i, &hd_i, &rd_i, &0i32))?;
        let (stride_i, nope_i) = (hd as i32, (hd - rd) as i32);
        dsv4_launch!(self.attn, "dsv4_kv_sim_g64_strided", self.stream.stream,
            (ceil256(s * ((hd - rd) / 64) * 32), 1, 1), (256, 1, 1), 0,
            (&mkv, &s_i, &stride_i, &nope_i))?;
        // ring_write_b writes rows [lo, s) of mkv to cache[(start_pos+r)%win]; lo is 0 unless this
        // is a start_pos==0 prefill with s>win (the rotated last-128 case). Re-prime (start_pos>0)
        // always has lo=0 → writes all s rows at (start_pos+r)%win.
        let lo = if start_pos == 0 && s > win { s - win } else { 0 };
        let win_i = win as i32;
        dsv4_launch!(self.attn, "dsv4_ring_write_b", self.stream.stream, ((s - lo) as u32, 1, 1), (256, 1, 1), 0,
            (&st.kv_cache, &mkv, &s_i, &sp_i, &win_i, &hd_i))?;
        Ok(())
    }

    /// DSpark draft attention (§B.10 DSparkAttention, start_pos>0 branch). Writes main_kv at
    /// slot start_pos%win, then attends the `block` draft rows over [ring ++ draft_kv] using the
    /// 133-entry non-causal index list (window ++ 5 draft slots — every row sees all 5 drafts).
    /// `x` [block, dim] is the attn_norm output; `main_x` [1, dim] the projected trunk hidden.
    /// Returns attn_out [block, dim]. NON-CAUSAL within the block (no sequential per-row loop).
    #[allow(clippy::too_many_arguments)]
    pub fn dspark_attn_forward(
        &self,
        layer: &Dsv4GpuLayer,
        st: &mut Dsv4AttnState,
        x: &B,
        block: usize,
        start_pos: usize,
        main_x: &B,
        rope: &DevRope,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<B> {
        self.dspark_attn_forward_impl(layer, st, x, block, start_pos, main_x, rope, cfg, false).map(|r| r.0)
    }

    /// Debug (R4 audit): capture the attention's internal `q`/`kv` (draft q/kv) and `o`
    /// (gather+de-rotation output) and `oflat` (grouped-LoRA wo_a output) alongside wo_b.
    pub fn dspark_attn_forward_capture(
        &self,
        layer: &Dsv4GpuLayer,
        st: &mut Dsv4AttnState,
        x: &B,
        block: usize,
        start_pos: usize,
        main_x: &B,
        rope: &DevRope,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<(B, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        self.dspark_attn_forward_impl(layer, st, x, block, start_pos, main_x, rope, cfg, true)
    }

    fn dspark_attn_forward_impl(
        &self,
        layer: &Dsv4GpuLayer,
        st: &mut Dsv4AttnState,
        x: &B,
        block: usize,
        start_pos: usize,
        main_x: &B,
        rope: &DevRope,
        cfg: &dsv4_load::Dsv4Config,
        capture: bool,
    ) -> Result<(B, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        use cudarc::driver::result;
        let (dim, qlr, nh, hd, rd, win, eps) =
            (cfg.dim, cfg.q_lora_rank, cfg.n_heads, cfg.head_dim, cfg.rope_head_dim, cfg.window_size, cfg.norm_eps);

        // ---- main_kv at start_pos (written BEFORE draft attention) ----
        let (mxc, mxsa) = self.quant_g128(main_x, 1, dim)?;
        let mkv = self.fp8_bsb_rows::<B, CudaSlice<u8>>(&layer.wkv, &mxc, &mxsa, 1)?;
        let mkv = self.rmsnorm(&mkv, &layer.kv_norm, 1, hd, eps)?;
        let (sp_i, one_i, hd_i, rd_i) = (start_pos as i32, 1i32, hd as i32, rd as i32);
        let sp_dev = self.dev.htod_sync_copy(&[start_pos as i32])?;
        dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(32), 1, 1), (256, 1, 1), 0,
            (&mkv, &rope.cos, &rope.sin, &sp_dev, &one_i, &hd_i, &rd_i, &0i32))?;
        let (stride_i, nope_i) = (hd as i32, (hd - rd) as i32);
        dsv4_launch!(self.attn, "dsv4_kv_sim_g64_strided", self.stream.stream,
            (ceil256(((hd - rd) / 64) * 32), 1, 1), (256, 1, 1), 0,
            (&mkv, &one_i, &stride_i, &nope_i))?;
        let win_i = win as i32;
        dsv4_launch!(self.attn, "dsv4_ring_write_b", self.stream.stream, (1u32, 1, 1), (256, 1, 1), 0,
            (&st.kv_cache, &mkv, &one_i, &sp_i, &win_i, &hd_i))?;

        // ---- draft q/kv at positions start_pos+1 .. start_pos+block ----
        let pos0 = start_pos + 1;
        let (xc, xsa) = self.quant_g128::<B, CudaSlice<u8>>(x, block, dim)?;
        let qr_pre = self.fp8_bsb_rows(&layer.wq_a, &xc, &xsa, block)?;
        let qr = self.rmsnorm(&qr_pre, &layer.q_norm, block, qlr, eps)?;
        let (qrc, qrsa) = self.quant_g128::<B, CudaSlice<u8>>(&qr, block, qlr)?;
        let q = self.fp8_bsb_rows(&layer.wq_b, &qrc, &qrsa, block)?;
        let q_pre_host: Vec<f32> = if capture {
            self.dev.dtoh_sync_copy(&q)?.iter().map(|v| v.to_f32()).collect()
        } else { Vec::new() };
        let rows_q = block * nh;
        let (rows_i, hd_i2) = (rows_q as i32, hd as i32);
        // §B.1.1 per-head rescale: ONE BLOCK PER ROW (kernel maps blockIdx.x -> row; the old
        // ceil256 grid rescaled only rows 0..1 — every head-row >= 2 went UNRESCALED, which
        // was the DSpark draft-quality bug: draft q systematically miscalibrated past the
        // first two head rows, acceptance collapsing at d3).
        dsv4_launch!(self.attn, "dsv4_attn_rescale_b", self.stream.stream, (rows_q as u32, 1, 1), (256, 1, 1), 0,
            (&q, &rows_i, &hd_i2, &eps))?;
        let q_rsc_host: Vec<f32> = if capture {
            self.dev.dtoh_sync_copy(&q)?.iter().map(|v| v.to_f32()).collect()
        } else { Vec::new() };
        let pos_q_dev = dsv4_gpu::iota_positions::<CudaSlice<i32>>(&self.dev, &self.spine, &self.stream, pos0 as i32, 1, nh as i32, rows_q)?;
        dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(rows_q * 32), 1, 1), (256, 1, 1), 0,
            (&q, &rope.cos, &rope.sin, &pos_q_dev, &rows_i, &hd_i2, &rd_i, &0i32))?;
        let q_host: Vec<f32> = if capture {
            self.dev.dtoh_sync_copy(&q)?.iter().map(|v| v.to_f32()).collect()
        } else { Vec::new() };
        let kv = self.fp8_bsb_rows(&layer.wkv, &xc, &xsa, block)?;
        let kv = self.rmsnorm(&kv, &layer.kv_norm, block, hd, eps)?;
        let pos_kv_dev = dsv4_gpu::iota_positions::<CudaSlice<i32>>(&self.dev, &self.spine, &self.stream, pos0 as i32, 1, 1, block)?;
        let blk_i = block as i32;
        dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(block * 32), 1, 1), (256, 1, 1), 0,
            (&kv, &rope.cos, &rope.sin, &pos_kv_dev, &blk_i, &hd_i, &rd_i, &0i32))?;
        dsv4_launch!(self.attn, "dsv4_kv_sim_g64_strided", self.stream.stream,
            (ceil256(block * ((hd - rd) / 64) * 32), 1, 1), (256, 1, 1), 0,
            (&kv, &blk_i, &stride_i, &nope_i))?;
        let kv_host: Vec<f32> = if capture {
            self.dev.dtoh_sync_copy(&kv)?.iter().map(|v| v.to_f32()).collect()
        } else { Vec::new() };

        // ---- kv_cat = [ring(128) ++ draft_kv(block)] ----
        let total_kv = win + block;
        let mut kv_cat = self.dev.alloc_zeros::<bf16>(total_kv * hd)?;
        unsafe {
            result::memcpy_dtod_async(*kv_cat.device_ptr(), *st.kv_cache.device_ptr(), win * hd * 2, self.stream.stream)
                .map_err(|e| anyhow!("dspark ring→kv_cat dtod: {e}"))?;
            result::memcpy_dtod_async(
                (*kv_cat.device_ptr()) + (win * hd * 2) as u64,
                *kv.device_ptr(), block * hd * 2, self.stream.stream,
            ).map_err(|e| anyhow!("dspark draft_kv→kv_cat dtod: {e}"))?;
        }

        // ---- 133-entry non-causal index list [0..min(128,start_pos+1) ++ 128..128+block] × block ----
        // R4/drafter-graph prep: device-side (was a host Vec build + htod = a full sync per
        // draft step). Row-uniform integer math — bit-exact vs the host build.
        let t_win = win.min(start_pos + 1);
        let t = t_win + block;
        let idxs_dev = self.dev.alloc_zeros::<i32>(block * t)?;
        let (blk_i2, t_win_i, t_i0, win_i2) = (block as i32, t_win as i32, t as i32, win as i32);
        dsv4_launch!(self.attn, "dsv4_dspark_draft_idxs_b", self.stream.stream,
            (ceil256(block * t), 1, 1), (256, 1, 1), 0,
            (&idxs_dev, &blk_i2, &t_win_i, &t_i0, &win_i2))?;

        // ---- batched gather over block rows × t keys (non-causal) ----
        let scale = (hd as f64).powf(-0.5) as f32;
        let s_pad = ((block + 15) / 16) * 16;
        let rows_q_pad = s_pad * nh;
        let mut o = self.dev.alloc_zeros::<bf16>(rows_q_pad * hd)?;
        let nh16 = (nh / 16) as u32;
        let (t_i, n_i) = (t as i32, total_kv as i32);
        dsv4_launch!(self.spine, "dsv4_gather_attn", self.stream.stream,
            (block as u32, 1u32, nh16), (256, 1, 1), GATHER_SMEM,
            (&q, &kv_cat, &o, &layer.sink, &idxs_dev, &t_i, &n_i, &scale))?;

        // ---- de-rotation with the DRAFT freqs (inverse RoPE) ----
        dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(rows_q * 32), 1, 1), (256, 1, 1), 0,
            (&o, &rope.cos, &rope.sin, &pos_q_dev, &rows_i, &hd_i2, &rd_i, &1i32))?;
        let o_host: Vec<f32> = if capture {
            self.dev.dtoh_sync_copy(&o)?.iter().map(|v| v.to_f32()).collect()
        } else { Vec::new() };

        // ---- grouped-LoRA O projection (wo_a einsum + wo_b fp8_bsb) — same as trunk ----
        let (g, r, gd) = (cfg.o_groups, cfg.o_lora_rank, nh * hd / cfg.o_groups);
        let mut oflat = self.dev.alloc_zeros::<bf16>(s_pad * g * r)?;
        let (blk_i2, g_i, r_i, gd_i, ors_i) = (block as i32, g as i32, r as i32, gd as i32, (nh * hd) as i32);
        if !exact_gemm_enabled() {
            let wq = layer.wo_a_q.as_ref().expect("wo_a fp8 quant must be loaded");
            olo_einsum_fast::<B, CudaSlice<u8>>(self, &mut oflat, &o, wq, block, g, r, gd)?;
        } else {
            let tiles_n = (r + 15) / 16;
            dsv4_launch!(self.attn, "dsv4_olo_proj_tc_b", self.stream.stream,
                ((tiles_n * g) as u32, s_pad as u32 / 16, 1), (32, 1, 1), 0,
                (&oflat, &o, &layer.wo_a, &blk_i2, &g_i, &r_i, &gd_i, &ors_i))?;
        }
        let oflat_host: Vec<f32> = if capture {
            self.dev.dtoh_sync_copy(&oflat)?.iter().map(|v| v.to_f32()).collect()
        } else { Vec::new() };
        let (oc, osa) = self.quant_g128::<B, CudaSlice<u8>>(&oflat, block, g * r)?;
        let attn_out = self.fp8_bsb_rows(&layer.wo_b, &oc, &osa, block)?;
        Ok((attn_out, q_pre_host, q_rsc_host, kv_host, o_host, oflat_host))
    }

    /// CUDA-graph (--dspark-graph) variant of [`dspark_attn_forward`](Self::dspark_attn_forward):
    /// the SAME kernel sequence and argument values, but capture-legal — every transient is
    /// a `Dsv4Buf` (GSlice bump slices of the workspace slab under capture) and the three
    /// position vectors (main_kv sp, draft q positions, draft kv positions) arrive as
    /// PERSISTENT device buffers refreshed outside the graph per step (the eager impl
    /// htod's/iota-allocs them per call — illegal inside capture). No debug captures.
    /// Bitwise-identical to the eager impl by construction (same kernels, same order).
    #[allow(clippy::too_many_arguments)]
    pub fn dspark_attn_forward_dev<X: dsv4_gpu::Dsv4Buf<bf16>, C: dsv4_gpu::Dsv4Buf<u8>, I: dsv4_gpu::Dsv4Buf<i32>>(
        &self,
        layer: &Dsv4GpuLayer,
        st: &mut Dsv4AttnState,
        x: &X,
        block: usize,
        start_pos: usize,
        main_x: &X,
        rope: &DevRope,
        cfg: &dsv4_load::Dsv4Config,
        pos_sp: &CudaSlice<i32>,
        pos_q: &CudaSlice<i32>,
        pos_kv: &CudaSlice<i32>,
    ) -> Result<X> {
        use cudarc::driver::result;
        let (dim, qlr, nh, hd, rd, win, eps) =
            (cfg.dim, cfg.q_lora_rank, cfg.n_heads, cfg.head_dim, cfg.rope_head_dim, cfg.window_size, cfg.norm_eps);

        // ---- main_kv at start_pos (written BEFORE draft attention) ----
        let (mxc, mxsa) = self.quant_g128::<X, C>(main_x, 1, dim)?;
        let mkv = self.fp8_bsb_rows::<X, C>(&layer.wkv, &mxc, &mxsa, 1)?;
        let mkv = self.rmsnorm(&mkv, &layer.kv_norm, 1, hd, eps)?;
        let (sp_i, one_i, hd_i, rd_i) = (start_pos as i32, 1i32, hd as i32, rd as i32);
        dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(32), 1, 1), (256, 1, 1), 0,
            (&mkv, &rope.cos, &rope.sin, pos_sp, &one_i, &hd_i, &rd_i, &0i32))?;
        let (stride_i, nope_i) = (hd as i32, (hd - rd) as i32);
        dsv4_launch!(self.attn, "dsv4_kv_sim_g64_strided", self.stream.stream,
            (ceil256(((hd - rd) / 64) * 32), 1, 1), (256, 1, 1), 0,
            (&mkv, &one_i, &stride_i, &nope_i))?;
        let win_i = win as i32;
        dsv4_launch!(self.attn, "dsv4_ring_write_b", self.stream.stream, (1u32, 1, 1), (256, 1, 1), 0,
            (&st.kv_cache, &mkv, &one_i, &sp_i, &win_i, &hd_i))?;

        // ---- draft q/kv at positions start_pos+1 .. start_pos+block ----
        let (xc, xsa) = self.quant_g128::<X, C>(x, block, dim)?;
        let qr_pre = self.fp8_bsb_rows::<X, C>(&layer.wq_a, &xc, &xsa, block)?;
        let qr = self.rmsnorm(&qr_pre, &layer.q_norm, block, qlr, eps)?;
        let (qrc, qrsa) = self.quant_g128::<X, C>(&qr, block, qlr)?;
        let q = self.fp8_bsb_rows::<X, C>(&layer.wq_b, &qrc, &qrsa, block)?;
        let rows_q = block * nh;
        let (rows_i, hd_i2) = (rows_q as i32, hd as i32);
        // ONE BLOCK PER ROW (the acceptance-cliff fix — see dspark_attn_forward_impl).
        dsv4_launch!(self.attn, "dsv4_attn_rescale_b", self.stream.stream, (rows_q as u32, 1, 1), (256, 1, 1), 0,
            (&q, &rows_i, &hd_i2, &eps))?;
        dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(rows_q * 32), 1, 1), (256, 1, 1), 0,
            (&q, &rope.cos, &rope.sin, pos_q, &rows_i, &hd_i2, &rd_i, &0i32))?;
        let kv = self.fp8_bsb_rows::<X, C>(&layer.wkv, &xc, &xsa, block)?;
        let kv = self.rmsnorm(&kv, &layer.kv_norm, block, hd, eps)?;
        let blk_i = block as i32;
        dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(block * 32), 1, 1), (256, 1, 1), 0,
            (&kv, &rope.cos, &rope.sin, pos_kv, &blk_i, &hd_i, &rd_i, &0i32))?;
        dsv4_launch!(self.attn, "dsv4_kv_sim_g64_strided", self.stream.stream,
            (ceil256(block * ((hd - rd) / 64) * 32), 1, 1), (256, 1, 1), 0,
            (&kv, &blk_i, &stride_i, &nope_i))?;

        // ---- kv_cat = [ring(128) ++ draft_kv(block)] ----
        let total_kv = win + block;
        let kv_cat = X::alloc_zeros(&self.dev, self.stream.stream, total_kv * hd)?;
        unsafe {
            result::memcpy_dtod_async(kv_cat.dptr(), *st.kv_cache.device_ptr(), win * hd * 2, self.stream.stream)
                .map_err(|e| anyhow!("dspark-dev ring→kv_cat dtod: {e}"))?;
            result::memcpy_dtod_async(
                kv_cat.dptr() + (win * hd * 2) as u64,
                kv.dptr(), block * hd * 2, self.stream.stream,
            ).map_err(|e| anyhow!("dspark-dev draft_kv→kv_cat dtod: {e}"))?;
        }

        // ---- non-causal index list [0..min(128,sp+1) ++ 128..128+block] × block (device-side) ----
        let t_win = win.min(start_pos + 1);
        let t = t_win + block;
        let idxs_dev = I::alloc_zeros(&self.dev, self.stream.stream, block * t)?;
        let (blk_i2, t_win_i, t_i0, win_i2) = (block as i32, t_win as i32, t as i32, win as i32);
        dsv4_launch!(self.attn, "dsv4_dspark_draft_idxs_b", self.stream.stream,
            (ceil256(block * t), 1, 1), (256, 1, 1), 0,
            (&idxs_dev, &blk_i2, &t_win_i, &t_i0, &win_i2))?;

        // ---- batched gather over block rows × t keys (non-causal) ----
        let scale = (hd as f64).powf(-0.5) as f32;
        let s_pad = ((block + 15) / 16) * 16;
        let rows_q_pad = s_pad * nh;
        let o = X::alloc_zeros(&self.dev, self.stream.stream, rows_q_pad * hd)?;
        let nh16 = (nh / 16) as u32;
        let (t_i, n_i) = (t as i32, total_kv as i32);
        dsv4_launch!(self.spine, "dsv4_gather_attn", self.stream.stream,
            (block as u32, 1u32, nh16), (256, 1, 1), GATHER_SMEM,
            (&q, &kv_cat, &o, &layer.sink, &idxs_dev, &t_i, &n_i, &scale))?;

        // ---- de-rotation with the DRAFT freqs (inverse RoPE) ----
        dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(rows_q * 32), 1, 1), (256, 1, 1), 0,
            (&o, &rope.cos, &rope.sin, pos_q, &rows_i, &hd_i2, &rd_i, &1i32))?;

        // ---- grouped-LoRA O projection (wo_a einsum + wo_b fp8_bsb) — same as trunk ----
        let (g, r, gd) = (cfg.o_groups, cfg.o_lora_rank, nh * hd / cfg.o_groups);
        let mut oflat = X::alloc_zeros(&self.dev, self.stream.stream, s_pad * g * r)?;
        let (blk_i2, g_i, r_i, gd_i, ors_i) = (block as i32, g as i32, r as i32, gd as i32, (nh * hd) as i32);
        if !exact_gemm_enabled() {
            let wq = layer.wo_a_q.as_ref().expect("wo_a fp8 quant must be loaded");
            olo_einsum_fast::<X, C>(self, &mut oflat, &o, wq, block, g, r, gd)?;
        } else {
            let tiles_n = (r + 15) / 16;
            dsv4_launch!(self.attn, "dsv4_olo_proj_tc_b", self.stream.stream,
                ((tiles_n * g) as u32, s_pad as u32 / 16, 1), (32, 1, 1), 0,
                (&oflat, &o, &layer.wo_a, &blk_i2, &g_i, &r_i, &gd_i, &ors_i))?;
        }
        let (oc, osa) = self.quant_g128::<X, C>(&oflat, block, g * r)?;
        self.fp8_bsb_rows::<X, C>(&layer.wo_b, &oc, &osa, block)
    }

}
