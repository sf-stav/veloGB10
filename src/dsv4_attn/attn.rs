//! `Dsv4AttnRuntime::attn_forward`: the trunk attention sublayer (SWA window ring, CSA indexer top-k, HCA all blocks).
//! One long function, kept whole (splitting a function is a code change). Siblings: `ops.rs` (its ops); `mod.rs` has the map.

use super::*;

impl Dsv4AttnRuntime {
    /// Trunk attention sublayer (§B.1–B.4) on the attn_norm output `x` [s, dim] bf16.
    /// Kind-dispatched: SWA (3A path — window-only ring gather); CSA (window ++ indexer
    /// top-512 blocks, ratio-4 overlap compressor); HCA (window ++ ALL compressed blocks,
    /// ratio-128 non-overlap compressor). Returns (attn_out [s, dim] bf16, topk_idx
    /// [s, t] i32, t).
    ///
    /// CSA/HCA index space (decode): window indices address `kv_cache[0..win]` (ring
    /// physical slots 0..win-1); compress indices address `kv_cache[win..win+nb]` (the
    /// compressor cache tail, copied from `attn_compressor.cache` after each step).
    /// Prefill gathers over `kv_attn_scratch[0..s+nb]` (kv ++ compressor cache, matching
    /// the CPU reference's `kv_attn = kv.clone(); kv_attn.extend(kvc)`).
    pub fn attn_forward<X: dsv4_gpu::Dsv4Buf<bf16>, F: dsv4_gpu::Dsv4Buf<f32>, I: dsv4_gpu::Dsv4Buf<i32>, C: dsv4_gpu::Dsv4Buf<u8>, U: dsv4_gpu::Dsv4Buf<u32>>(
        &self,
        layer: &Dsv4GpuLayer,
        st: &mut Dsv4AttnState,
        x: &X,
        s: usize,
        start_pos: usize,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<(X, I, usize)> {
        let (dim, qlr, nh, hd, rd, win) =
            (cfg.dim, cfg.q_lora_rank, cfg.n_heads, cfg.head_dim, cfg.rope_head_dim, cfg.window_size);
        let eps = cfg.norm_eps;
        let kind = layer.kind;
        // Per-kind RoPE table (§B.1.3): SWA plain θ vs CSA/HCA YaRN. `rope_for` picks the
        // multikind table when present (full trunk), else the lane's single-kind `rope`.
        let rope = self.rope_for(kind);

        // Q path: quantize x once (K=4096 shared by wq_a and wkv — same codes, bit-identical).
        // R3A.1 E2: wq_a + wkv in ONE fused launch at decode/verify widths (independent ops,
        // same activation — per-element chains identical to the two separate launches, gated).
        let (xc, xsa) = self.quant_g128::<X, C>(x, s, dim)?;
        let (qr_pre, kv) = if s <= 16 {
            self.fp8_bsb2q_rows(&layer.wq_a, &layer.wkv, &xc, &xsa, s)?
        } else {
            (self.fp8_bsb_rows(&layer.wq_a, &xc, &xsa, s)?, self.fp8_bsb_rows(&layer.wkv, &xc, &xsa, s)?)
        };
        let pair_seq = crate::dsv4_gpu::env_flag_once(crate::opt!("pair-seq"));
        let s_i = s as i32;
        let (sp_i, nh_i, rd_i) = (start_pos as i32, nh as i32, rd as i32);
        let rows_q = s * nh;
        let (rows_i, hd_i) = (rows_q as i32, hd as i32);
        // Seq arm only: the q-position iota array (pair arms compute positions inline).
        // Lives to the de-rotation site.
        let pos_q_dev = if pair_seq {
            Some(dsv4_gpu::iota_positions::<I>(&self.dev, &self.spine, &self.stream, start_pos as i32, 1, nh as i32, rows_q)?)
        } else { None };
        // R3A.1 E2 rung 2: the q/kv RMSNorms are independent (same-stage outputs of the fused
        // wq_a/wkv GEMM) — ONE pair launch, bitwise == two singles (the spine pair gate).
        let (qr, kv) = if !pair_seq {
            let qr = X::alloc_zeros(&self.dev, self.stream.stream, s * qlr)?;
            let kv_n = X::alloc_zeros(&self.dev, self.stream.stream, s * hd)?;
            let (qlr_i, hd_i) = (qlr as i32, hd as i32);
            dsv4_launch!(self.spine, "dsv4_rmsnorm_pair_b", self.stream.stream, (s as u32, 2, 1), (256, 1, 1), 0,
                (&qr, &qr_pre, &layer.q_norm, &s_i, &qlr_i,
                 &kv_n, &kv, &layer.kv_norm, &s_i, &hd_i, &eps))?;
            (qr, kv_n)
        } else {
            (self.rmsnorm(&qr_pre, &layer.q_norm, s, qlr, eps)?,
             self.rmsnorm(&kv, &layer.kv_norm, s, hd, eps)?)
        };
        let (qrc, qrsa) = self.quant_g128::<X, C>(&qr, s, qlr)?;
        let q = self.fp8_bsb_rows::<X, C>(&layer.wq_b, &qrc, &qrsa, s)?;
        if !pair_seq && s <= 16 {
            // E2 rung 5 (Tier 1.2): rescale(q) + rope(q,kv) + KV-sim in ONE launch
            // (3 -> 1; bitwise == the separate kernels — the spine fused-tail gate).
            dsv4_launch!(self.attn, "dsv4_rescale_rope_sim_b", self.stream.stream,
                ((rows_q + s) as u32, 1, 1), (256, 1, 1), 0,
                (&q, &kv, &rope.cos, &rope.sin, &sp_i, &nh_i, &rows_i, &s_i, &hd_i, &rd_i, &eps))?;
        } else {
        // §B.1.1 per-head weight-free RMS rescale (bit-exact vs dsv4_cpu::attn_qkv).
        dsv4_launch!(self.attn, "dsv4_attn_rescale_b", self.stream.stream, (rows_q as u32, 1, 1), (256, 1, 1), 0,
            (&q, &rows_i, &hd_i, &eps))?;
        if !pair_seq {
            // R3A.1 E2 rung 2: ONE rope launch for q+kv with INLINE positions (identical
            // integers to the iota arrays it replaces; per-element math unchanged). Replaces
            // 2 iota + 2 rope launches. q: p = start_pos + row/nh; kv: p = start_pos + row.
            dsv4_launch!(self.spine, "dsv4_rope_pair_b", self.stream.stream,
                ((((rows_q + 7) / 8) as u32), 2, 1), (256, 1, 1), 0,
                (&q, &kv, &rope.cos, &rope.sin, &sp_i, &nh_i, &rows_i, &s_i, &hd_i, &rd_i))?;
        } else {
            // RoPE on q's last 64 dims (every head of token t at position start_pos + t).
            dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(rows_q * 32), 1, 1), (256, 1, 1), 0,
                (&q, &rope.cos, &rope.sin, pos_q_dev.as_ref().unwrap(), &rows_i, &hd_i, &rd_i, &0i32))?;
            let pos_kv_dev = dsv4_gpu::iota_positions::<I>(&self.dev, &self.spine, &self.stream, start_pos as i32, 1, 1, s)?;
            dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(s * 32), 1, 1), (256, 1, 1), 0,
                (&kv, &rope.cos, &rope.sin, &pos_kv_dev, &s_i, &hd_i, &rd_i, &0i32))?;
        }
        let stride_i = hd as i32;
        let nope_i = (hd - rd) as i32;
        dsv4_launch!(self.attn, "dsv4_kv_sim_g64_strided", self.stream.stream,
            (ceil256(s * ((hd - rd) / 64) * 32), 1, 1), (256, 1, 1), 0,
            (&kv, &s_i, &stride_i, &nope_i))?;
        }

        // ---- ring write + index list + compressor + gather (prefill batched; start_pos>0 sequential) ----
        let scale = (hd as f64).powf(-0.5) as f32;
        // Pad s up to a multiple of 16 for dsv4_olo_proj_tc_b's [16,8] WMMA tile grid: the kernel
        // launches ceil(s/16) M-tiles and each reads/writes a full 16-row slab of o/oflat, so an
        // unpadded s%16!=0 makes the last tile stride OOB (CUDA_ERROR_ILLEGAL_ADDRESS — silent
        // single-process where the OOB lands in mapped adjacent memory, fatal under TP=2's layout
        // where the 128-expert heap is smaller). Phantom rows are never read by downstream ops
        // (de-rotation, quant_g128, fp8_bsb all use the real `s`/`rows_q` row counts).
        let s_pad = ((s + 15) / 16) * 16;
        let rows_q_pad = s_pad * nh;
        let o = X::alloc_zeros(&self.dev, self.stream.stream, rows_q_pad * hd)?;
        let win_i = win as i32;

        // R2.0 instrument: record which attention path each (layer, start_pos, s) takes —
        // the audit's missing "is the batched chunk-prefill path engaged for chunks 2+?" probe.
        // Gated by --prefill-trace so it is free in production. The sequential per-token loop
        // (start_pos>0) is the structural suspect #1 (§3b #1): chunks 2+ ride it, not the batched
        // prefill path, which would explain the ~27–45 prefill rate sagging with depth.
        let _pf_trace = crate::opts::var(crate::opt!("prefill-trace")).ok().is_some();
        let (idxs, t): (I, usize) = if start_pos == 0 {
            // ===================== PREFILL (start_pos == 0): batched — UNCHANGED (lane 3A/3C + trunk gate) =====================
            // §B.2 ring write BEFORE attention (current token attends to itself), rotated for S>win.
            let lo = if s > win { s - win } else { 0 };
            let sp_zero = 0i32;
            dsv4_launch!(self.attn, "dsv4_ring_write_b", self.stream.stream, ((s - lo) as u32, 1, 1), (256, 1, 1), 0,
                (&st.kv_cache, &kv, &s_i, &sp_zero, &win_i, &hd_i))?;
            let nh16 = (nh / 16) as u32;
            match kind {
                dsv4_load::LayerKind::Swa => {
                    let t_win = s.min(win);
                    let idxs = I::alloc_zeros(&self.dev, self.stream.stream, s * t_win)?;
                    let t_win_i = t_win as i32;
                    dsv4_launch!(self.attn, "dsv4_window_idxs_b", self.stream.stream,
                        (ceil256(s * t_win), 1, 1), (256, 1, 1), 0,
                        (&idxs, &s_i, &sp_zero, &win_i, &t_win_i))?;
                    let (t_i, n_i) = (t_win as i32, s as i32);
                    dsv4_launch!(self.spine, "dsv4_gather_attn", self.stream.stream,
                        (s as u32, 1u32, nh16), (256, 1, 1), GATHER_SMEM,
                        (&q, &kv, &o, &layer.sink, &idxs, &t_i, &n_i, &scale))?;
                    (idxs, t_win)
                }
                dsv4_load::LayerKind::Csa | dsv4_load::LayerKind::Hca => {
                    let comp_ks = self.comp.as_ref().expect("CSA/HCA needs comp kernels");
                    let compressor = st.attn_compressor.as_mut().expect("CSA/HCA needs attn_compressor");
                    let ratio = compressor.spec.ratio;
                    let nb_committed = compressor.prefill::<X, I>(&self.dev, comp_ks, &self.stream, x, s, rope)?;
                    let t_win = s.min(win);
                    let t_comp = match kind {
                        dsv4_load::LayerKind::Csa => {
                            let indexer = st.indexer.as_mut().expect("CSA needs indexer");
                            let f_bsb = &self.bk["gemm_dsv4_fp8_bsb2"];
                            indexer.forward::<X, F, I, U>(&self.dev, comp_ks, &self.stream, f_bsb, x, &qr, s, 0, s, rope)?
                        }
                        dsv4_load::LayerKind::Hca => s / ratio,
                        _ => unreachable!(),
                    };
                    let t_total = t_win + t_comp;
                    let idxs = I::alloc_zeros(&self.dev, self.stream.stream, s * t_total)?;
                    let (t_win_i, t_total_i) = (t_win as i32, t_total as i32);
                    dsv4_launch!(self.attn, "dsv4_window_idxs_strided_b", self.stream.stream,
                        (ceil256(s * t_win), 1, 1), (256, 1, 1), 0,
                        (&idxs, &s_i, &sp_zero, &win_i, &t_win_i, &t_total_i))?;
                    if t_comp > 0 {
                        match kind {
                            dsv4_load::LayerKind::Hca => {
                                let (offset_i, ratio_i, t_comp_i, col_off_i) =
                                    (s as i32, ratio as i32, t_comp as i32, t_win as i32);
                                dsv4_launch!(self.attn, "dsv4_compress_idxs_b", self.stream.stream,
                                    (ceil256(s * t_comp), 1, 1), (256, 1, 1), 0,
                                    (&idxs, &s_i, &sp_zero, &ratio_i, &offset_i, &t_comp_i, &t_total_i, &col_off_i))?;
                            }
                            dsv4_load::LayerKind::Csa => {
                                let (k_i, col_off_i) = (t_comp as i32, t_win as i32);
                                let indexer = st.indexer.as_ref().expect("CSA needs indexer");
                                dsv4_launch!(self.attn, "dsv4_idxs_place_b", self.stream.stream,
                                    (ceil256(s * t_comp), 1, 1), (256, 1, 1), 0,
                                    (&idxs, indexer.idx_dev(), &s_i, &k_i, &t_total_i, &col_off_i))?;
                            }
                            _ => unreachable!(),
                        }
                    }
                    let scratch = st.kv_attn_scratch.as_ref().unwrap();
                    let comp_bytes = nb_committed * hd * 2;
                    if comp_bytes > 0 {
                        // Item 3: the compressor's cache is aliased to kv_cache[win..]. The
                        // epilogue wrote there; read the comp_cache tail from the alias.
                        let comp_src = if compressor.cache_alias_ptr != 0 {
                            compressor.cache_alias_ptr
                        } else {
                            *compressor.cache.device_ptr()
                        };
                        unsafe {
                            result::memcpy_dtod_async(*scratch.device_ptr(), kv.dptr(), s * hd * 2, self.stream.stream)
                                .map_err(|e| anyhow!("prefill kv→scratch dtod: {e}"))?;
                            result::memcpy_dtod_async((*scratch.device_ptr()) + (s * hd * 2) as u64, comp_src, comp_bytes, self.stream.stream)
                                .map_err(|e| anyhow!("prefill comp→scratch dtod: {e}"))?;
                        }
                    }
                    let (t_i, n_i) = (t_total as i32, (s + nb_committed) as i32);
                    dsv4_launch!(self.spine, "dsv4_gather_attn", self.stream.stream,
                        (s as u32, 1u32, nh16), (256, 1, 1), GATHER_SMEM,
                        (&q, scratch, &o, &layer.sink, &idxs, &t_i, &n_i, &scale))?;
                    (idxs, t_total)
                }
            }
        } else if start_pos > 0 && s >= win && start_pos % win == 0 && s % 128 == 0 {
            // ===================== R2.1 BATCHED CONTINUATION (start_pos > 0, prefill chunk 2+) =====================
            // The opt-ladder-1 batched the COMPRESSOR for chunks 2+; this extends the batching
            // to the ATTENTION (window ring-write + gather) — eliminating the sequential per-token
            // loop that ran for every chunk after the first (the root cause of the ~27–45 prefill
            // rate sagging with depth). Gate: start_pos > 0 && s >= win && start_pos % win == 0
            // (prefill chunks have s=PREFILL_CHUNK=4096 >> win=128, start_pos is a multiple of
            // 4096; verify has s ≤ ~6 < 128, so this branch is never reached for verify/decode).
            //
            // Build a UNIFIED scratch = [prefix window (win rows from ring) | new kv (s rows) |
            // comp_cache (nb rows from the aliased kv_cache tail)] and gather from it with logical
            // indices. The window for row r (absolute position start_pos+r) is the contiguous
            // scratch rows [r+1 .. r+win] — prefix fills the early entries (no masking needed).
            // This is bitwise-identical to one-shot prefill (same positions, same order: the
            // gather sums over the same KV vectors in the same oldest→newest order). The ring is
            // written AFTER the gather (for future chunks/decodes); the prefix is saved to scratch
            // BEFORE the ring write clobbers it.
            let nh16 = (nh / 16) as u32;
            let sp_i = start_pos as i32;
            let win_plus_s = win + s;

            // 1. Compressor prefill_at (batched — already proven by opt-ladder-1). For CSA the
            //    indexer.forward below also calls its own compressor; the ATTENTION compressor
            //    here writes the comp_cache to the aliased kv_cache[win..] tail.
            let (nb_committed, t_comp, idxs): (usize, usize, I) = match kind {
                dsv4_load::LayerKind::Swa => {
                    // SWA: no compressor, no comp_cache. scratch = [prefix(win) | new(s)].
                    (0, 0, I::alloc_zeros(&self.dev, self.stream.stream, s * win)?)
                }
                dsv4_load::LayerKind::Csa | dsv4_load::LayerKind::Hca => {
                    let comp_ks = self.comp.as_ref().expect("CSA/HCA needs comp kernels");
                    let compressor = st.attn_compressor.as_mut().expect("CSA/HCA needs attn_compressor");
                    let ratio = compressor.spec.ratio;
                    let nb = compressor.prefill_at::<X, I>(&self.dev, comp_ks, &self.stream, x, s, start_pos, rope)?;
                    let nb_committed = (start_pos + s) / ratio;
                    let t_comp = match kind {
                        dsv4_load::LayerKind::Csa => {
                            let indexer = st.indexer.as_mut().expect("CSA needs indexer");
                            let f_bsb = &self.bk["gemm_dsv4_fp8_bsb2"];
                            // offset = win + s (comp_cache starts at scratch row win+s in the
                            // unified buffer; the remask kernel adds offset to block indices).
                            indexer.forward::<X, F, I, U>(&self.dev, comp_ks, &self.stream, f_bsb, x, &qr, s, start_pos, win + s, rope)?
                        }
                        dsv4_load::LayerKind::Hca => nb_committed,
                        _ => unreachable!(),
                    };
                    let t_total = win + t_comp;
                    (nb_committed, t_comp, I::alloc_zeros(&self.dev, self.stream.stream, s * t_total)?)
                }
            };
            let t_total = win + t_comp;

            // 2. Build the unified scratch: [prefix(win) | new(s) | comp(nb)].
            let scratch = st.kv_attn_scratch.as_ref().unwrap();
            // 2a. Copy prefix window (win rows) from the ring (kv_cache[0..win-1]).
            unsafe {
                result::memcpy_dtod_async(
                    *scratch.device_ptr(),
                    *st.kv_cache.device_ptr(),
                    win * hd * 2,
                    self.stream.stream,
                ).map_err(|e| anyhow!("cont prefill ring→scratch dtod: {e}"))?;
            }
            // 2b. Copy new kv (s rows) into scratch[win..win+s-1].
            unsafe {
                result::memcpy_dtod_async(
                    (*scratch.device_ptr()) + (win * hd * 2) as u64,
                    kv.dptr(),
                    s * hd * 2,
                    self.stream.stream,
                ).map_err(|e| anyhow!("cont prefill kv→scratch dtod: {e}"))?;
            }
            // 2c. Copy comp_cache (nb_committed rows) from the aliased kv_cache[win..] tail
            //     into scratch[win+s..win+s+nb-1]. (CSA/HCA only; SWA skips this.)
            if nb_committed > 0 {
                let comp_src = if let Some(ac) = st.attn_compressor.as_ref() {
                    if ac.cache_alias_ptr != 0 { ac.cache_alias_ptr } else { *ac.cache.device_ptr() }
                } else { unreachable!() };
                let comp_bytes = nb_committed * hd * 2;
                unsafe {
                    result::memcpy_dtod_async(
                        (*scratch.device_ptr()) + (win_plus_s * hd * 2) as u64,
                        comp_src,
                        comp_bytes,
                        self.stream.stream,
                    ).map_err(|e| anyhow!("cont prefill comp→scratch dtod: {e}"))?;
                }
            }

            // 3. Build window idxs (batched, continuation regime: v = row + 1 + j).
            let (t_win_i, t_total_i) = (win as i32, t_total as i32);
            dsv4_launch!(self.attn, "dsv4_window_idxs_strided_b", self.stream.stream,
                (ceil256(s * win), 1, 1), (256, 1, 1), 0,
                (&idxs, &s_i, &sp_i, &win_i, &t_win_i, &t_total_i))?;

            // 4. Build compress idxs (HCA only — CSA already placed by the indexer above).
            if t_comp > 0 {
                match kind {
                    dsv4_load::LayerKind::Hca => {
                        let ratio = st.attn_compressor.as_ref().expect("HCA").spec.ratio;
                        let (offset_i, ratio_i, t_comp_i, col_off_i) =
                            (win_plus_s as i32, ratio as i32, t_comp as i32, win as i32);
                        dsv4_launch!(self.attn, "dsv4_compress_idxs_b", self.stream.stream,
                            (ceil256(s * t_comp), 1, 1), (256, 1, 1), 0,
                            (&idxs, &s_i, &sp_i, &ratio_i, &offset_i, &t_comp_i, &t_total_i, &col_off_i))?;
                    }
                    dsv4_load::LayerKind::Csa => {
                        // The indexer's remask already wrote block_index + (win+s) into idxs.
                        // Place them into the unified idxs buffer at column win.
                        let (k_i, col_off_i) = (t_comp as i32, win as i32);
                        let indexer = st.indexer.as_ref().expect("CSA needs indexer");
                        dsv4_launch!(self.attn, "dsv4_idxs_place_b", self.stream.stream,
                            (ceil256(s * t_comp), 1, 1), (256, 1, 1), 0,
                            (&idxs, indexer.idx_dev(), &s_i, &k_i, &t_total_i, &col_off_i))?;
                    }
                    _ => unreachable!(),
                }
            }

            // 5. Gather from the unified scratch (batched over all s rows).
            let (t_i, n_i) = (t_total as i32, (win_plus_s + nb_committed) as i32);
            dsv4_launch!(self.spine, "dsv4_gather_attn", self.stream.stream,
                (s as u32, 1u32, nh16), (256, 1, 1), GATHER_SMEM,
                (&q, scratch, &o, &layer.sink, &idxs, &t_i, &n_i, &scale))?;

            // 6. Ring write: write the new kv rows into the ring (for future chunks/decodes).
            //    The gather read from scratch (not the ring), so the ring write can happen here
            //    without clobbering anything the gather needed. dsv4_ring_write_b with
            //    start_pos > 0 writes all s rows; the final ring state has the last win rows at
            //    the correct physical slots (wrapping ensures the most recent write wins).
            dsv4_launch!(self.attn, "dsv4_ring_write_b", self.stream.stream,
                (s as u32, 1, 1), (256, 1, 1), 0,
                (&st.kv_cache, &kv, &s_i, &sp_i, &win_i, &hd_i))?;

            (idxs, t_total)
        } else if start_pos > 0 && s <= win
            && !crate::dsv4_gpu::env_flag_once(crate::opt!("verify-seq"))
            && !crate::dsv4_gpu::env_flag_once(crate::opt!("graph"))
        {
            // ===================== SESSION-10 FUSED GATHER (verify/decode, ONE launch/layer) =====================
            // Queue #5: `dsv4_fused_gather_b` replaces, for the verify (s=6) and decode (s=1)
            // shapes at any start_pos, the assembly [window idxs + compress idxs/place + the
            // unified-scratch d2d copies + dsv4_gather_attn] — the index lists are computed
            // in-kernel and the KV is read directly from the ring / the layer's new kv / the
            // comp tail. Per-(row, head) chains are bitwise == the SEQ/R4 assemblies (same
            // key order, tile boundaries, online-softmax and P·V chains — the masked entries
            // keep their tile positions as p=0 / fma(0,0) no-ops). Compressor + indexer stay
            // batched (identical to the assembly arms; the indexer remask uses the batched
            // id space win+s — uniform with comp_off below). --verify-seq (the documented
            // A/B arm) and --graph (the graph policies capture the assembly kernels) keep
            // the old paths. The returned idx buffer is a placeholder (topk_idx is consumed
            // only by the debug replay tool, which runs the assembly path).
            let mut nb_max = 0usize;
            let mut t_comp_max = 0usize;
            let mut idx_dev: u64 = 0;
            let mut ratio_use = 0usize;
            if matches!(kind, dsv4_load::LayerKind::Csa | dsv4_load::LayerKind::Hca) {
                let comp_ks = self.comp.as_ref().expect("CSA/HCA needs comp kernels");
                let compressor = st.attn_compressor.as_mut().expect("CSA/HCA needs attn_compressor");
                let ratio = compressor.spec.ratio;
                // kernel dispatch: ratio > 0 selects the HCA arange path; CSA (indexer
                // idx_dev) needs ratio == 0 (the kernel branches idx_csa != 0 second).
                ratio_use = if kind == dsv4_load::LayerKind::Hca { ratio } else { 0 };
                if s >= ratio && start_pos % ratio == 0 {
                    compressor.prefill_at::<X, I>(&self.dev, comp_ks, &self.stream, x, s, start_pos, rope)?;
                } else {
                    compressor.forward_tokens::<X, F, I, U>(&self.dev, comp_ks, &self.stream, x, s, start_pos, rope)?;
                }
                nb_max = (start_pos + s) / ratio;
                t_comp_max = match kind {
                    dsv4_load::LayerKind::Csa => {
                        let indexer = st.indexer.as_mut().expect("CSA needs indexer");
                        let f_bsb = &self.bk["gemm_dsv4_fp8_bsb2"];
                        let k = indexer.forward::<X, F, I, U>(&self.dev, comp_ks, &self.stream, f_bsb, x, &qr, s, start_pos, win + s, rope)?;
                        idx_dev = *indexer.idx_dev().device_ptr();
                        k
                    }
                    _ => nb_max,
                };
            }
            // ONE fused gather over [ring | new kv | comp tail]; index lists + scratch
            // copies computed in-kernel (bitwise == the assembly arms).
            let nh16 = (nh / 16) as u32;
            let ring = &st.kv_cache;
            let comp_tail = match st.attn_compressor.as_ref() {
                Some(ac) if ac.cache_alias_ptr != 0 => ac.cache_alias_ptr,
                Some(ac) => *ac.cache.device_ptr(),
                None => *ring.device_ptr(), // SWA: never dereferenced (t_comp_max == 0)
            };
            let (s_i, sp_i, win_i, tcm_i, ratio_i, off_i, k_i) = (
                s as i32, start_pos as i32, win as i32,
                t_comp_max as i32, ratio_use as i32, (win + s) as i32, t_comp_max as i32,
            );
            dsv4_launch!(self.spine, "dsv4_fused_gather_b", self.stream.stream,
                (s as u32, 1u32, nh16), (256, 1, 1), GATHER_SMEM,
                (&q, ring, &kv, &comp_tail, &idx_dev, &o, &layer.sink,
                 &s_i, &sp_i, &win_i, &ratio_i, &off_i, &tcm_i, &k_i, &scale))?;
            // Ring write AFTER the gather (batched last-wins — identical to the R4 arm).
            dsv4_launch!(self.attn, "dsv4_ring_write_b", self.stream.stream,
                (s as u32, 1, 1), (256, 1, 1), 0,
                (ring, &kv, &s_i, &sp_i, &win_i, &hd_i))?;
            let t_total = win + t_comp_max;
            let idxs = I::alloc_zeros(&self.dev, self.stream.stream, s * t_total)?;
            (idxs, t_total)
        } else if start_pos > 0 && s > 1 && start_pos >= win
            && !crate::dsv4_gpu::env_flag_once(crate::opt!("verify-seq"))
        {
            // ===================== R4 BATCHED VERIFY (s>1, start_pos >= win) =====================
            // The R2.1 batched-continuation machinery generalized to arbitrary alignment by a
            // rotated, position-ordered prefix copy. The compressor/indexer state work stays
            // batched (identical to the SEQ arm's front part); the window ring's per-token
            // sequencing (3-4 launches × s rows) becomes ONE s-wide gather over a scratch the
            // ring writes cannot clobber, then ONE batched ring write (same last-wins
            // semantics as the prefill/R2.1 arms). Per-row attended positions/KV/order are
            // identical to the SEQ arm — the equivalence gate compares the arms bitwise.
            let nh16 = (nh / 16) as u32;
            let sp_i = start_pos as i32;
            let win_plus_s = win + s;

            // 1. Compressor + (CSA) indexer — batched, identical to the SEQ arm.
            let mut nb_max = 0usize;
            if matches!(kind, dsv4_load::LayerKind::Csa | dsv4_load::LayerKind::Hca) {
                let comp_ks = self.comp.as_ref().expect("CSA/HCA needs comp kernels");
                let compressor = st.attn_compressor.as_mut().expect("CSA/HCA needs attn_compressor");
                let ratio = compressor.spec.ratio;
                if s >= ratio && start_pos % ratio == 0 {
                    compressor.prefill_at::<X, I>(&self.dev, comp_ks, &self.stream, x, s, start_pos, rope)?;
                } else {
                    compressor.forward_tokens::<X, F, I, U>(&self.dev, comp_ks, &self.stream, x, s, start_pos, rope)?;
                }
                nb_max = (start_pos + s) / ratio;
            }

            // 2. Unified scratch [prefix(win, position-ordered) | new(s) | comp(nb)].
            let t_comp = match kind {
                dsv4_load::LayerKind::Swa => 0usize,
                dsv4_load::LayerKind::Csa => {
                    let comp_ks = self.comp.as_ref().expect("CSA needs comp kernels");
                    let indexer = st.indexer.as_mut().expect("CSA needs indexer");
                    let f_bsb = &self.bk["gemm_dsv4_fp8_bsb2"];
                    indexer.forward::<X, F, I, U>(&self.dev, comp_ks, &self.stream, f_bsb, x, &qr, s, start_pos, win + s, rope)?
                }
                dsv4_load::LayerKind::Hca => nb_max,
            };
            let t_total = win + t_comp;
            let idxs = I::alloc_zeros(&self.dev, self.stream.stream, s * t_total)?;
            let scratch = match st.kv_attn_scratch.as_ref() {
                Some(sc) => sc.clone(),
                None => self.dev.alloc_zeros::<bf16>((win + s + nb_max) * hd)?,
            };
            let sp = start_pos % win;
            unsafe {
                // 2a. rotated prefix copy (position-ordered: scratch row m = position
                // start_pos-win+m for every alignment).
                result::memcpy_dtod_async(
                    *scratch.device_ptr(),
                    *st.kv_cache.device_ptr() + (sp * hd * 2) as u64,
                    (win - sp) * hd * 2,
                    self.stream.stream,
                ).map_err(|e| anyhow!("verify prefix rot-a: {e}"))?;
                result::memcpy_dtod_async(
                    (*scratch.device_ptr()) + ((win - sp) * hd * 2) as u64,
                    *st.kv_cache.device_ptr(),
                    sp * hd * 2,
                    self.stream.stream,
                ).map_err(|e| anyhow!("verify prefix rot-b: {e}"))?;
                // 2b. the new kv rows (batched projection from the front of attn_forward).
                result::memcpy_dtod_async(
                    (*scratch.device_ptr()) + (win * hd * 2) as u64,
                    kv.dptr(),
                    s * hd * 2,
                    self.stream.stream,
                ).map_err(|e| anyhow!("verify kv→scratch: {e}"))?;
                // 2c. the committed compressor tail (CSA/HCA).
                if nb_max > 0 {
                    let ac = st.attn_compressor.as_ref().unwrap();
                    let comp_src = if ac.cache_alias_ptr != 0 { ac.cache_alias_ptr } else { *ac.cache.device_ptr() };
                    result::memcpy_dtod_async(
                        (*scratch.device_ptr()) + (win_plus_s * hd * 2) as u64,
                        comp_src,
                        nb_max * hd * 2,
                        self.stream.stream,
                    ).map_err(|e| anyhow!("verify comp→scratch: {e}"))?;
                }
            }

            // 3. window idxs (all rows at once) + compress part (per-row causal by the
            //    existing kernels' own semantics).
            let (s_i, win_i, t_total_i) = (s as i32, win as i32, t_total as i32);
            dsv4_launch!(self.attn, "dsv4_window_idxs_verify_b", self.stream.stream,
                (ceil256(s * win), 1, 1), (256, 1, 1), 0,
                (&idxs, &s_i, &win_i, &win_i, &t_total_i))?;
            match kind {
                dsv4_load::LayerKind::Swa => {}
                dsv4_load::LayerKind::Hca => {
                    if t_comp > 0 {
                        let ratio = st.attn_compressor.as_ref().expect("HCA").spec.ratio;
                        let (offset_i, ratio_i, t_comp_i, col_off_i) =
                            (win_plus_s as i32, ratio as i32, t_comp as i32, win as i32);
                        dsv4_launch!(self.attn, "dsv4_compress_idxs_b", self.stream.stream,
                            (ceil256(s * t_comp), 1, 1), (256, 1, 1), 0,
                            (&idxs, &s_i, &sp_i, &ratio_i, &offset_i, &t_comp_i, &t_total_i, &col_off_i))?;
                    }
                }
                dsv4_load::LayerKind::Csa => {
                    if t_comp > 0 {
                        let (k_i, col_off_i) = (t_comp as i32, win as i32);
                        let indexer = st.indexer.as_ref().expect("CSA needs indexer");
                        dsv4_launch!(self.attn, "dsv4_idxs_place_b", self.stream.stream,
                            (ceil256(s * t_comp), 1, 1), (256, 1, 1), 0,
                            (&idxs, indexer.idx_dev(), &s_i, &k_i, &t_total_i, &col_off_i))?;
                    }
                }
            }

            // 4. ONE s-wide gather over the scratch.
            let (t_i, n_i) = (t_total as i32, (win_plus_s + nb_max) as i32);
            dsv4_launch!(self.spine, "dsv4_gather_attn", self.stream.stream,
                (s as u32, 1u32, nh16), (256, 1, 1), GATHER_SMEM,
                (&q, scratch, &o, &layer.sink, &idxs, &t_i, &n_i, &scale))?;

            // 5. Ring write AFTER the gather (last-wins semantics, identical to the SEQ
            //    arm's sequential per-token writes — the prefill/R2.1 arms prove the batch form).
            dsv4_launch!(self.attn, "dsv4_ring_write_b", self.stream.stream,
                (s as u32, 1, 1), (256, 1, 1), 0,
                (&st.kv_cache, &kv, &s_i, &sp_i, &win_i, &hd_i))?;

            (idxs, t_total)
        } else {
            // ===================== DECODE / VERIFY (start_pos > 0): sequential per-token loop =====================
            // Reference-faithful: the reference's start_pos>0 path is single-token decode
            // (model.py:535 `kv.squeeze(1)` asserts seqlen==1). A batched ring-write of s>1 verify
            // tokens would clobber prefix slots that earlier rows still need (slots [start_pos%win ..
            // (start_pos+s-1)%win] hold prefix positions inside row 0's 128-window). Writing token r →
            // attending row r → writing token r+1 matches N sequential decodes exactly; s==1 decode
            // is one iteration (bit-identical to the former batched s==1 path). Each row also gets its
            // own (start_pos+r) for the window ring order and the HCA compress count — the batched
            // path used a single start_pos (wrong for verify rows >0). The attention compressor cache
            // and the CSA indexer are append-only (row r reads only prefix + rows <r), so they run
            // batched ONCE before the loop; only the window ring needs per-token sequencing.
            let mut nb_max = 0usize;
            if matches!(kind, dsv4_load::LayerKind::Csa | dsv4_load::LayerKind::Hca) {
                let comp_ks = self.comp.as_ref().expect("CSA/HCA needs comp kernels");
                let compressor = st.attn_compressor.as_mut().expect("CSA/HCA needs attn_compressor");
                let ratio = compressor.spec.ratio;
                // Batched prefill pool when the chunk boundary aligns (start_pos%ratio==0 AND
                // s>=ratio) — the parallel chunk-prefill speedup (one batched pool launch
                // instead of s sequential decode+sync round-trips). Sequential decode otherwise
                // (verify/decode with small s — the pool can't run the per-token state machine).
                // Bitwise-identical to one-shot (§12.B.5 — the carry flag handles block 0's
                // overlap from the frontier).
                if s >= ratio && start_pos % ratio == 0 {
                    compressor.prefill_at::<X, I>(&self.dev, comp_ks, &self.stream, x, s, start_pos, rope)?;
                } else {
                    compressor.forward_tokens::<X, F, I, U>(&self.dev, comp_ks, &self.stream, x, s, start_pos, rope)?;
                }
                nb_max = (start_pos + s) / ratio;
                // Item 3 alias: the compressor's epilogue already wrote its cache rows
                // directly to kv_cache[win..] (the alias). No d2d mirror needed — the
                // gather reads from the unified kv_cache (ring + tail) contiguously.
            }
            let (t_total, idxs_out): (usize, I) = match kind {
                dsv4_load::LayerKind::Swa => (win, I::alloc_zeros(&self.dev, self.stream.stream, s * win)?),
                dsv4_load::LayerKind::Csa => {
                    let comp_ks = self.comp.as_ref().expect("CSA needs comp kernels");
                    let indexer = st.indexer.as_mut().expect("CSA needs indexer");
                    let f_bsb = &self.bk["gemm_dsv4_fp8_bsb2"];
                    let k = indexer.forward::<X, F, I, U>(&self.dev, comp_ks, &self.stream, f_bsb, x, &qr, s, start_pos, win, rope)?;
                    let t_total = win + k;
                    // CUDA-graph mode: alloc for the CAP (k ≤ index_topk) so replay never
                    // outgrows the baked buffer; the layout args below stay at the live t.
                    let cap = if crate::dsv4_gpu::env_flag_once(crate::opt!("graph")) {
                        s * (win + indexer.index_topk)
                    } else {
                        s * t_total
                    };
                    let idxs = I::alloc_zeros(&self.dev, self.stream.stream, cap)?;
                    // Place the batched indexer [s, k] output at col win for ALL rows (per-row causal
                    // already enforced by score/remask — selections are a pure function of the prefix).
                    let (k_i, t_total_i, col_off_i) = (k as i32, t_total as i32, win as i32);
                    dsv4_launch!(self.attn, "dsv4_idxs_place_b", self.stream.stream,
                        (ceil256(s * k), 1, 1), (256, 1, 1), 0,
                        (&idxs, indexer.idx_dev(), &s_i, &k_i, &t_total_i, &col_off_i))?;
                    (t_total, idxs)
                }
                dsv4_load::LayerKind::Hca => {
                    let ratio = st.attn_compressor.as_ref().expect("HCA needs compressor").spec.ratio;
                    let t_comp_max = (start_pos + s) / ratio;
                    let t_total = win + t_comp_max;
                    // CUDA-graph mode: alloc for the compressor cache CAPACITY (see CSA note).
                    let cap = if crate::dsv4_gpu::env_flag_once(crate::opt!("graph")) {
                        s * (win + st.attn_compressor.as_ref().expect("HCA").cache_rows)
                    } else {
                        s * t_total
                    };
                    (t_total, I::alloc_zeros(&self.dev, self.stream.stream, cap)?)
                }
            };
            let gather_n = (win + nb_max) as i32;
            let t_total_i = t_total as i32;
            let nh16 = (nh / 16) as u32;
            let one_i = 1i32;
            let t_win_i = win as i32;
            for r in 0..s {
                let sp_r = start_pos + r;
                let sp_r_i = sp_r as i32;
                // (1) ring write kv[r] → slot sp_r % win (write BEFORE attention — self-attend).
                let kv_row = kv.view(r * hd, hd);
                dsv4_launch!(self.attn, "dsv4_ring_write_b", self.stream.stream, (1u32, 1, 1), (256, 1, 1), 0,
                    (&st.kv_cache, &kv_row, &one_i, &sp_r_i, &win_i, &hd_i))?;
                // (2) window idxs for row r at sp=start_pos+r (into idxs_out[r, 0..win]); HCA compress
                //     idxs at idxs_out[r, win..win+t_comp_r] (count grows with r). CSA compress slots
                //     were placed batched above (k uniform; per-row masking carried as −1).
                let idxs_row = idxs_out.view(r * t_total, t_total);
                dsv4_launch!(self.attn, "dsv4_window_idxs_b", self.stream.stream,
                    (ceil256(win), 1, 1), (256, 1, 1), 0,
                    (&idxs_row, &one_i, &sp_r_i, &win_i, &t_win_i))?;
                let t_r = match kind {
                    dsv4_load::LayerKind::Hca => {
                        let ratio = st.attn_compressor.as_ref().expect("HCA").spec.ratio;
                        let t_comp_r = (sp_r + 1) / ratio;
                        if t_comp_r > 0 {
                            let (offset_i, ratio_i, t_comp_r_i, col_off_i) =
                                (win as i32, ratio as i32, t_comp_r as i32, win as i32);
                            dsv4_launch!(self.attn, "dsv4_compress_idxs_b", self.stream.stream,
                                (ceil256(t_comp_r), 1, 1), (256, 1, 1), 0,
                                (&idxs_row, &one_i, &sp_r_i, &ratio_i, &offset_i, &t_comp_r_i, &t_total_i, &col_off_i))?;
                        }
                        win + t_comp_r
                    }
                    _ => t_total, // SWA: win; CSA: win+k (indexer placed batched).
                };
                // (3) gather row r over [ring + comp tail] (kv_cache). Per-row (s=1) — the ring now
                //     holds exactly prefix + tokens 0..r, as a sequential decode would.
                let q_row = q.view(r * nh * hd, nh * hd);
                let o_row = o.view(r * nh * hd, nh * hd);
                let t_r_i = t_r as i32;
                dsv4_launch!(self.spine, "dsv4_gather_attn", self.stream.stream,
                    (1u32, 1u32, nh16), (256, 1, 1), GATHER_SMEM,
                    (&q_row, &st.kv_cache, &o_row, &layer.sink, &idxs_row, &t_r_i, &gather_n, &scale))?;
            }
            (idxs_out, t_total)
        };
        if _pf_trace {
            let path = if start_pos == 0 { "batched" }
                else if start_pos > 0 && s >= win && start_pos % win == 0 { "batched-cont" }
                else { "SEQ-per-token" };
            eprintln!("[dsv4-pf] attn kind={kind:?} sp={start_pos} s={s} path={path}");
        }

        // De-rotation (inverse RoPE on the attention output, compensating K≡V leak).
        if !pair_seq {
            // R3A.1 E2 rung 2: inline q positions (identical integers), inverse=1.
            dsv4_launch!(self.spine, "dsv4_rope_q_inline_b", self.stream.stream, (ceil256(rows_q * 32), 1, 1), (256, 1, 1), 0,
                (&o, &rope.cos, &rope.sin, &sp_i, &nh_i, &rows_i, &hd_i, &rd_i, &1i32))?;
        } else {
            dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(rows_q * 32), 1, 1), (256, 1, 1), 0,
                (&o, &rope.cos, &rope.sin, pos_q_dev.as_ref().unwrap(), &rows_i, &hd_i, &rd_i, &1i32))?;
        }

        // §B.1.4 grouped-LoRA O: bf16 einsum with wo_a.view(8,1024,4096), then fp8_bsb(wo_b).
        // Uses the WMMA tensor-core kernel (dsv4_olo_proj_tc_b) for production prefill speed
        // (~30% of the long8k replay was the scalar variant). One warp per [16,16] output tile.
        // R3A.1: tiles_n matches the kernel's internal (r+15)/16 — the old (r+7)/8 launched
        // 2x the CTAs, half exiting immediately (512 dead CTAs per launch at decode).
        // NOTE (R3.3): a decode-width GEMV replacement was tried and REVERTED — the trunk is
        // width-bitwise end-to-end (a row at s≤16 must be bit-identical to the same row at
        // prefill width; the non-128-aligned chunked-prefill tail rides the sequential s=1 path
        // and is gate-compared against one-shot). Only reduction-order-preserving changes are
        // admissible at decode width.
        let (g, r, gd) = (cfg.o_groups, cfg.o_lora_rank, nh * hd / cfg.o_groups);
        let mut oflat = X::alloc_zeros(&self.dev, self.stream.stream, s_pad * g * r)?;
        anyhow::ensure!(r % 64 == 0, "olo tc4 pack needs r % 64 == 0 (r={r})");
        let tiles_m = (s + 15) / 16;
        let (s_i, g_i, r_i, gd_i, ors_i) = (s as i32, g as i32, r as i32, gd as i32, (nh * hd) as i32);
        if !exact_gemm_enabled() {
            // Item 2.5 fast path (default): fp8 einsum-class per-group BMM — the weights are
            // fp8-quantized at load (per-head-group tiles), o is quantized here, and the
            // reduction order is scheduler-chosen (pf2-class schedule). Tolerance-gated, not
            // bitwise (AGENTS §3). The --exact-gemm path below keeps the locked chains.
            let wq = layer.wo_a_q.as_ref().expect("wo_a fp8 quant must be loaded");
            olo_einsum_fast::<X, C>(self, &mut oflat, &o, wq, s, g, r, gd)?;
        } else if s > 16 {
            // R3A.4 P5: C=4-packed kernel at prefill widths — 4 n-tiles per CTA share one A
            // load (the [16,gd] slab re-read drops 4x); identical per-element chains (gated
            // bitwise vs the C=1 kernel). Decode/verify keeps C=1 (ILP-4 register pressure
            // loses at 1-warp decode geometry — E0c measured).
            let packs_n = r / 64;
            dsv4_launch!(self.attn, "dsv4_olo_proj_tc4_b", self.stream.stream,
                ((packs_n * g) as u32, tiles_m as u32, 1), (32, 1, 1), 0,
                (&oflat, &o, &layer.wo_a, &s_i, &g_i, &r_i, &gd_i, &ors_i))?;
        } else {
            let tiles_n = (r + 15) / 16;
            dsv4_launch!(self.attn, "dsv4_olo_proj_tc_b", self.stream.stream,
                ((tiles_n * g) as u32, tiles_m as u32, 1), (32, 1, 1), 0,
                (&oflat, &o, &layer.wo_a, &s_i, &g_i, &r_i, &gd_i, &ors_i))?;
        }
        let (oc, osa) = self.quant_g128::<X, C>(&oflat, s, g * r)?;
        let attn_out = self.fp8_bsb_rows(&layer.wo_b, &oc, &osa, s)?;
        Ok((attn_out, idxs, t))
    }

}
