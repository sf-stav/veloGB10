//! Launch helpers, the dense chain (`chain_ws*`, split-K + Hadamard) and the eager step (`step_kernels*`,
//! `head_logits_ws`, `layer_forward_ws`). Every launch rides the model's stream; nothing allocates (graph-capturable).
//! Siblings: `moe.rs` (the MoE sublayer it calls), `helpers.rs` (`Ws`); `mod.rs` has the map.

use super::*;

impl GpuModel {
    fn f(&self, name: &str) -> Result<cudarc::driver::CudaFunction> {
        self.dev.get_func(MODULE, name).with_context(|| format!("{name} missing from {MODULE}"))
    }

    /// Launch `name` on the model's stream (the capturable stream every forward kernel rides).
    /// With `--naive-pdl=1` the launch goes through the raw handle with the programmatic-stream-
    /// serialization attribute (gpu.rs `launch_pdl_raw`); the kernel, grid, block, smem and the
    /// parameter bytes are identical either way.
    pub fn l<P>(&self, name: &str, cfg: LaunchConfig, params: P) -> Result<()>
    where cudarc::driver::CudaFunction: LaunchAsync<P>, P: super::gpu::NvParams {
        if super::gpu::pdl_on() {
            if let Some(f) = super::gpu::nv_raw_fn(name) {
                let mut ps = params.nv_params();
                return unsafe { super::gpu::launch_pdl_raw(self.stream.stream, f, cfg.grid_dim, cfg.block_dim, cfg.shared_mem_bytes, &mut ps) }
                    .with_context(|| format!("launch {name} (pdl)"));
            }
        }
        unsafe { self.f(name)?.launch_on_stream(&self.stream, cfg, params) }.with_context(|| format!("launch {name}"))
    }

    /// Wait for everything queued on the model's stream.
    pub fn sync(&self) -> Result<()> {
        unsafe { cudarc::driver::result::stream::synchronize(self.stream.stream) }.context("stream synchronize")
    }

    /// The dense chain for M <= 16 rows on workspace memory: xh = H(x*suh); y_raw = xh W (one pass or
    /// split-K + combine); y = H(y_raw) * svh. `x` f16 [m][k] -> returns y f16 [m][n] (a `ws` address).
    pub fn chain_ws(&self, q: &GpuQuad, x: u64, m: usize, ws: &mut Ws) -> Result<u64> {
        self.chain_ws_in(q, x, 0, m, ws)
    }

    /// The chain with the fused input: `x2 != 0` means the input is silu(x) * x2 (the MLP down input).
    /// Staged split-K (the decode plan): 2 launches — GEMM with the input transform in its staging,
    /// combine with the output transform. The per-step body (lm_head) keeps a separate input pass.
    pub fn chain_ws_in(&self, q: &GpuQuad, x: u64, x2: u64, m: usize, ws: &mut Ws) -> Result<u64> {
        self.chain_ws_out(q, x, x2, m, ws, 0).map(|(y, _)| y)
    }

    /// `out32 != 0`: the TP form — the result lands as fp32 at `out32` (the all-reduce partial) and the
    /// returned flag says so (true); otherwise f16 y. (The fallback paths always produce f16.)
    pub fn chain_ws_out(&self, q: &GpuQuad, x: u64, x2: u64, m: usize, ws: &mut Ws, out32: u64) -> Result<(u64, bool)> {
        ensure!((1..=MAX_PREFILL_ROWS).contains(&m), "chain: m {m} out of 1..{MAX_PREFILL_ROWS}");
        let (k, n) = (q.k, q.n);
        let y = ws.alloc::<u16>(m * n)?;
        if m > MAX_ROWS {
            // S6b wide: input pass (silu pair first when asked), the wide HMMA GEMM (mt 8), output pass
            let mut xin = x;
            if x2 != 0 {
                let din = ws.alloc::<u16>(m * k)?;
                self.l("nv_silu_mul", b1(m * k), (din, x, x2, (m * k) as i64))?;
                xin = din;
            }
            let xh = ws.alloc::<u16>(m * k)?;
            self.l("nv_had_suh", b32(m * k / 128), (xin, &q.suh, xh, k as i32))?;
            let yraw = ws.alloc::<u16>(m * n)?;
            let mt = 8usize;
            self.l("nv_gemm_wide", LaunchConfig { grid_dim: (((m + mt * 16 - 1) / (mt * 16)) as u32, (n / 128) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: (mt * 16 * (8 * 16 + 8) * 2) as u32 },
                   (&q.tr, xh, yraw, m as i32, k as i32, n as i32, q.bpb as i32, mt as i32))?;
            self.l("nv_had_svh", b32(m * n / 128), (yraw, &q.svh, y, n as i32))?;
            return Ok((y, false));
        }
        // T0-c: the staged body stages only THIS CTA's k-window, so the bytes are
        // m * ((k / ks) + 8) * 2, not the pre-T0-c m * (k + 8) * 2. grid_dim.y == q.ks IS the
        // split-K id, so each CTA walks exactly k/q.ks wide. The single-pass nv_gemm below
        // has grid_dim.y == 1, so it needs the ks==1 figure (unchanged from before T0-c).
        let smem = a1_smem(m, k, q.ks);
        let smem_ks1 = a1_smem(m, k, 1);
        if q.ks > 1 && q.a1 && smem > 0 {
            let w = ws.alloc::<f32>(q.ks * m * n)?;
            let fz = if super::gpu::fuse_on() && n / 128 <= super::gpu::FUSE_COUNTERS {
                super::gpu::NvFuse { svh: dptr(&q.svh), y, out32, cnt: dptr(&self.fuse_cnt) }
            } else { super::gpu::NvFuse::default() };
            self.l("nv_gemm_a1ks", LaunchConfig { grid_dim: ((n / 128) as u32, q.ks as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem },
                   (&q.tr, x, w, m as i32, k as i32, n as i32, q.bpb as i32, &q.suh, x2, fz))?;
            if fz.cnt == 0 {
                self.l("nv_ks_combine_had", LaunchConfig { grid_dim: (((m * n / 128) as u32 + 7) / 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                       (w, &q.svh, y, m as i32, n as i32, q.ks as i32, out32))?;
            }
            return Ok((y, out32 != 0));
        }
        // unfused input pass (silu pair first when asked)
        let mut xin = x;
        if x2 != 0 {
            let din = ws.alloc::<u16>(m * k)?;
            self.l("nv_silu_mul", b1(m * k), (din, x, x2, (m * k) as i64))?;
            xin = din;
        }
        let xh = ws.alloc::<u16>(m * k)?;
        self.l("nv_had_suh", b32(m * k / 128), (xin, &q.suh, xh, k as i32))?;
        if q.ks > 1 {
            let kb = k / 16;
            ensure!(kb % q.ks == 0, "ks {} does not divide kb {kb}", q.ks);
            let w = ws.alloc::<f32>(q.ks * m * n)?;
            if q.a1 && smem > 0 {
                self.l("nv_gemm_a1ks", LaunchConfig { grid_dim: ((n / 128) as u32, q.ks as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem },
                       (&q.tr, xh, w, m as i32, k as i32, n as i32, q.bpb as i32, 0u64, 0u64, super::gpu::NvFuse::default()))?;
            } else {
                self.l("nv_gemm_ks", LaunchConfig { grid_dim: ((n / 128) as u32, q.ks as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                       (&q.tr, xh, w, m as i32, k as i32, n as i32, q.bpb as i32))?;
            }
            self.l("nv_ks_combine_had", LaunchConfig { grid_dim: (((m * n / 128) as u32 + 7) / 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                   (w, &q.svh, y, m as i32, n as i32, q.ks as i32, 0u64))?;
        } else {
            let yraw = ws.alloc::<u16>(m * n)?;
            self.l("nv_gemm", LaunchConfig { grid_dim: ((n / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem_ks1 },
                   (&q.tr, xh, yraw, m as i32, k as i32, n as i32, q.bpb as i32))?;
            self.l("nv_had_svh", b32(m * n / 128), (yraw, &q.svh, y, n as i32))?;
        }
        Ok((y, false))
    }

    /// The whole step's kernels: embed -> layers -> (optionally) final norm + lm_head. `ids` / `pos`
    /// are device i32 [m] addresses; returns the logits address (f16 [m][V]) when asked. `taps`
    /// (eager gates): the residual after every layer, copied into `ws` (f32 [m][H] each).
    /// Every launch is on the model's stream; nothing allocates (graph-capturable).
    pub fn step_kernels(&self, ws: &mut Ws, ids: u64, pos: u64, m: usize, want_logits: bool, taps: Option<&mut Vec<u64>>) -> Result<Option<u64>> {
        self.step_kernels_sel(ws, ids, pos, m, want_logits, taps, None)
    }

    /// T2-d: as `step_kernels`, but each per-layer residual copy lands in an EXTERNAL f32 buffer
    /// at `taps_base + (li * tap_rows + row_off) * h` (in floats), not in the workspace.
    ///
    /// This is what makes chunked prefill possible: `tap_rows` is the FULL prompt length while
    /// `m` is the chunk width, so chunk `c0` writes its rows into the right slice of ONE
    /// [layers][rows][h] set. Two consumers need exactly that set -- the drafter's per-layer
    /// context taps, and the prefill head, whose input is the LAST layer's block (the single-shot
    /// path already recovers it from the last tap).
    ///
    /// A separate entry point rather than a parameter on `step_kernels_sel`, so the graph-capture
    /// path and every existing caller stay untouched.
    /// `sids` is the SLOT POINTER FOR THIS CALL'S ROWS (row 0's id). A chunked prefill offsets it by
    /// its first row, exactly as it offsets `ids`/`pos`; passing the model base for every chunk reads
    /// the FIRST chunk's slots for all rows.
    pub fn step_kernels_taps_ext(&self, ws: &mut Ws, ids: u64, pos: u64, m: usize,
                                 taps_base: u64, tap_rows: usize, row_off: usize, sids: u64) -> Result<()> {
        let h = self.cfg.hidden_size;
        ws.reset();
        let x = ws.alloc::<f32>(m * h)?;
        let hbuf = ws.alloc::<u16>(m * h)?;
        debug_assert!(sids != 0, "a step needs its rows' slot ids");
        self.l("nv_embed", LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
               (x, &self.embed, ids, h as i32))?;
        for (li, l) in self.layers.iter().enumerate() {
            let mark = ws.mark();
            self.layer_forward_ws(l, li, x, hbuf, pos, m, sids, ws)?;
            let dst = taps_base + ((li * tap_rows + row_off) * h * 4) as u64;
            unsafe { cudarc::driver::result::memcpy_dtod_async(dst, x, m * h * 4, self.stream.stream) }
                .context("tap copy")?;
            ws.release(mark);
        }
        Ok(())
    }

    /// `step_kernels` + `tap_sel` = (layers, dst): the residual after each listed layer j is copied to
    /// dst + j * m * h * 4 (f32 [m][h] blocks in list order) — the drafter's context taps, inside the graph.
    pub fn step_kernels_sel(&self, ws: &mut Ws, ids: u64, pos: u64, m: usize, want_logits: bool, mut taps: Option<&mut Vec<u64>>,
                            tap_sel: Option<(&[usize], u64)>) -> Result<Option<u64>> {
        let c = &self.cfg;
        let h = c.hidden_size;
        ensure!(m >= 1 && m <= MAX_PREFILL_ROWS, "rows {m}");
        ws.reset();
        let x = ws.alloc::<f32>(m * h)?;
        let hbuf = ws.alloc::<u16>(m * h)?;
        let tap_base = if taps.is_some() { ws.alloc::<f32>(self.layers.len() * m * h)? } else { 0 };
        self.l("nv_embed", LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, (x, &self.embed, ids, h as i32))?;
        for (li, l) in self.layers.iter().enumerate() {
            let mark = ws.mark();
            self.layer_forward_ws(l, li, x, hbuf, pos, m, self.sids, ws)?;
            if let Some(t) = taps.as_deref_mut() {
                let dst = tap_base + (li * m * h * 4) as u64;
                unsafe { cudarc::driver::result::memcpy_dtod_async(dst, x, m * h * 4, self.stream.stream) }.context("tap copy")?;
                t.push(dst);
            }
            if let Some((layers, dst)) = tap_sel {
                if let Some(j) = layers.iter().position(|&l2| l2 == li) {
                    unsafe { cudarc::driver::result::memcpy_dtod_async(dst + (j * m * h * 4) as u64, x, m * h * 4, self.stream.stream) }.context("tap copy")?;
                }
            }
            ws.release(mark);
        }
        if !want_logits { return Ok(None); }
        if m <= MAX_ROWS { return Ok(Some(self.head_logits_ws(x, hbuf, m, ws)?)); }
        // prefill: only the LAST row's logits (returned as a 1-row f16 [vl] buffer)
        let xl = x + ((m - 1) * h * 4) as u64;
        let hl = hbuf + ((m - 1) * h * 2) as u64;
        Ok(Some(self.head_logits_ws(xl, hl, 1, ws)?))
    }

    /// Final norm + lm_head on the residual `x`: logits f16 [m][V] (a `ws` address).
    pub fn head_logits_ws(&self, x: u64, hbuf: u64, m: usize, ws: &mut Ws) -> Result<u64> {
        let c = &self.cfg;
        self.l("nv_rmsnorm", LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 }, (hbuf, x, &self.norm, c.hidden_size as i32, c.rms_eps))?;
        self.chain_ws(&self.lm_head, hbuf, m, ws)
    }

    /// One layer on the residual `x` (f32 [m][H], a device address) in place; `hbuf` f16 [m][H] scratch.
    pub fn layer_forward_ws(&self, l: &GpuLayer, li: usize, x: u64, hbuf: u64, pos: u64, m: usize, sids: u64, ws: &mut Ws) -> Result<()> {
        let c = &self.cfg;
        let (h, nh, hd, vd) = (c.hidden_size, l.nh_l, c.head_dim, c.v_head_dim);
        let kv = l.kv_l;
        let g1 = |n: usize| LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        // ---- attention sublayer
        self.l("nv_rmsnorm", g1(m), (hbuf, x, &l.in_norm, h as i32, c.rms_eps))?;
        let mut q = self.chain_ws(&l.q, hbuf, m, ws)?;
        let mut k = self.chain_ws(&l.k, hbuf, m, ws)?;
        let mut v = self.chain_ws(&l.v, hbuf, m, ws)?;
        if l.k.n != kv * hd {
            let k2 = ws.alloc::<u16>(m * kv * hd)?;
            self.l("nv_cols_copy", b1(m * kv * hd), (k2, k, m as i32, l.k.n as i32, l.k_off as i32, (kv * hd) as i32))?;
            k = k2;
        }
        if l.v.n != kv * hd {
            let v2 = ws.alloc::<u16>(m * kv * hd)?;
            self.l("nv_cols_copy", b1(m * kv * hd), (v2, v, m as i32, l.v.n as i32, l.v_off as i32, (kv * hd) as i32))?;
            v = v2;
        }
        let theta = c.rope_theta_of(li);
        let ring = (l.kind == AttnKind::Swa) as i32;
        debug_assert!(sids != 0, "a layer needs its rows' slot ids");
        self.l("nv_rope_kv", b32(m * (nh + kv)), (q, k, v, &l.kc, &l.vc, pos, sids, (nh | (kv << 16)) as i32, (hd | (vd << 16) | ((ring as usize) << 30)) as i32, c.rotary_dim as i32, theta, l.slots as i32))?;
        let attn = ws.alloc::<u16>(m * nh * vd)?;
        match l.kind {
            AttnKind::Swa => {
                let sink_ref: &CudaSlice<u16> = match &l.sink { Some(s) => s, None => &self.zero_sink };
                let has_sink = l.sink.is_some() as i32;
                // prefill (m > 16): one block per (kv group, row), K/V shared by the group's heads (bitwise the same)
                let (swa_fn, swa_grid) = if m > 16 && nh / kv <= 8 && nh % kv == 0 { ("nv_attn_swa_grp", (kv as u32, m as u32, 1)) } else { ("nv_attn_swa", (nh as u32, m as u32, 1)) };
                self.l(swa_fn, LaunchConfig { grid_dim: swa_grid, block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                       (attn, q, &l.kc, &l.vc, sink_ref, pos, sids, (nh | (kv << 16)) as i32, (hd | (vd << 16)) as i32, l.slots as i32, (c.sliding_window as i32) | (has_sink << 16)))?;
            }
            AttnKind::Dsa => {
                let ix = l.idx.as_ref().context("DSA layer without indexer")?;
                let (ih, ihd) = (c.index_n_heads, c.index_head_dim);
                let iq_raw = self.chain_ws(&ix.wq, hbuf, m, ws)?;
                let ik_raw = self.chain_ws(&ix.wk, hbuf, m, ws)?;
                let wgt = self.chain_ws(&ix.wp, hbuf, m, ws)?; // [m][128] padded, 16 live
                let iq = ws.alloc::<f32>(m * ih * ihd)?;
                self.l("nv_idx_q", LaunchConfig { grid_dim: ((m * ih) as u32, 1, 1), block_dim: (ihd as u32, 1, 1), shared_mem_bytes: 0 },
                       (iq, iq_raw, pos, ih as i32, ihd as i32, c.rotary_dim as i32, theta, c.indexer_fp8 as i32))?;
                self.l("nv_idx_k", LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (ihd as u32, 1, 1), shared_mem_bytes: 0 },
                       (&ix.kq, &ix.ks, ik_raw, &ix.kn_w, &ix.kn_b, pos, sids, ihd as i32, c.rotary_dim as i32, theta, 1e-5f32, self.max_pos as i32))?;
                // scores over the WHOLE position range (stride max_pos): a fixed grid that strides over
                // the keys <= pos[m] (graph-capturable; entries past pos are never written or read)
                let stride = self.max_pos as i64;
                let scores = ws.alloc::<f32>(m * self.max_pos)?;
                if m > 16 && ih == 16 && ihd == 128 {
                    // prefill: tensor-core scorer (grid over key tiles x row groups; nkeys = whole plane)
                    self.l("nv_idx_score_wide", LaunchConfig { grid_dim: (((self.max_pos + 127) / 128) as u32, ((m + 31) / 32) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                           (scores, iq, wgt, &ix.kq, &ix.ks, pos, sids, m as i32, self.max_pos as i32, ix.wp.n as i32, stride))?;
                } else {
                    self.l("nv_idx_score", LaunchConfig { grid_dim: (IDX_SCORE_BLOCKS, m as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                           (scores, iq, wgt, &ix.kq, &ix.ks, pos, sids, ih as i32, ihd as i32, ix.wp.n as i32, stride, self.max_pos as i32))?;
                }
                let kk = c.index_top_k;
                let sel = ws.alloc::<i32>(m * kk)?;
                let nsel = ws.alloc::<i32>(m)?;
                self.l("nv_topk", g1(m), (sel, nsel, scores, pos, kk as i32, stride, 0u64, 0u64))?;
                // decode width: 64 splits of 32 keys so a warp walks 32 keys x 2 heads instead of 256 x 2
                // (the per-key chain is the cost; 8 splits took 181 us per launch). Prefill keeps 8: the
                // group-shared prefill kernel is pinned bitwise to the per-head one at that geometry.
                let nsplits = if m <= MAX_ROWS { 64usize } else { 8usize };
                let split_len = (kk + nsplits - 1) / nsplits;
                let pm = ws.alloc::<f32>(m * nh * nsplits)?;
                let pl = ws.alloc::<f32>(m * nh * nsplits)?;
                let pacc = ws.alloc::<f32>(m * nh * nsplits * vd)?;
                let dsa_fn = if m > 16 && (nh / kv) % 8 == 0 && nh / kv <= 16 && nh % kv == 0 { "nv_attn_dsa_split_grp" } else { "nv_attn_dsa_split" };
                self.l(dsa_fn, LaunchConfig { grid_dim: ((m * kv * nsplits) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                       (pm, pl, pacc, q, &l.kc, &l.vc, sel, nsel, sids, (nh | (kv << 16)) as i32, (hd | (vd << 16)) as i32,
                        pack_geom(kk, nsplits, split_len, l.slots)))?;
                self.l("nv_attn_combine", LaunchConfig { grid_dim: ((m * nh) as u32, 1, 1), block_dim: (vd as u32, 1, 1), shared_mem_bytes: 0 },
                       (attn, pm, pl, pacc, nsplits as i32, vd as i32))?;
            }
        }
        let part = self.tp.as_ref().map_or(0, |t| dptr(&t.part));
        let (o, o32) = self.chain_ws_out(&l.o, attn, 0, m, ws, part)?;
        self.add_reduced_ws(x, o, o32, m)?;
        // ---- MLP sublayer
        self.l("nv_rmsnorm", g1(m), (hbuf, x, &l.post_norm, h as i32, c.rms_eps))?;
        match &l.mlp {
            GpuMlp::Dense { gate, up, down } => {
                let g = self.chain_ws(gate, hbuf, m, ws)?;
                let u = self.chain_ws(up, hbuf, m, ws)?;
                let part = self.tp.as_ref().map_or(0, |t| dptr(&t.part));
                let (d, d32) = self.chain_ws_out(down, g, u, m, ws, part)?; // input = silu(g) * u, fused into the staging
                self.add_reduced_ws(x, d, d32, m)?;
            }
            GpuMlp::Moe(mo) => self.moe_ws(mo, hbuf, x, m, ws)?,
        }
        Ok(())
    }

}
