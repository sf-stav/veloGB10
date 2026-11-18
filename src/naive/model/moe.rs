//! The MoE sublayer of `GpuModel`: the TP partial reduce, the narrow and wide expert paths and the combine.
//! Siblings: `step.rs` calls these from `layer_forward_ws`; `mod.rs` has the map.

use super::*;

impl GpuModel {
    /// x += y (f16 partial [m][H]); under TP the partial is summed across the ranks first (fp32).
    /// `in_part`: the producer already wrote the fp32 partial into tp.part (F6), skip the convert.
    pub(super) fn add_reduced_ws(&self, x: u64, y: u64, in_part: bool, m: usize) -> Result<()> {
        let n = m * self.cfg.hidden_size;
        match &self.tp {
            None => self.l("nv_add_f16", b1(n), (x, y, n as i64)),
            Some(tp) => {
                if !in_part { self.l("nv_cvt_f16_f32", b1(n), (&tp.part, y, n as i64))?; }
                tp.all_reduce(&self.stream, &tp.part, n)?;
                self.l("nv_add_f32", b1(n), (x, &tp.part, n as i64))
            }
        }
    }

    pub(super) fn moe_ws(&self, mo: &GpuMoe, hbuf: u64, x: u64, m: usize, ws: &mut Ws) -> Result<()> {
        let c = &self.cfg;
        // mi = this rank's intermediate width per expert (expert TP: mi / world; the K-slice partial sums of the down
        // projection are completed by the same MoE reduce that sums the expert-parallel deal's partials)
        let (h, e, k, mi) = (c.hidden_size, c.num_experts, c.num_experts_per_tok, mo.mi_l);
        if m > MAX_ROWS { return self.moe_wide_ws(mo, hbuf, x, m, ws); }
        let ecap = e.min(m * k);
        let logits = ws.alloc::<f32>(m * e)?;
        self.l("nv_router_logits", LaunchConfig { grid_dim: (((e + 3) / 4) as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
               (logits, hbuf, &mo.router, m as i32, e as i32, h as i32))?;
        let ids = ws.alloc::<i32>(m * k)?;
        let wts = ws.alloc::<f32>(m * k)?;
        self.l("nv_router_topk", LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: (e * 4) as u32 },
               (ids, wts, logits, &mo.bias, e as i32, k as i32, c.norm_topk_prob as i32, mo.scale))?;
        let slotmap = ws.alloc::<i32>(e)?;
        let idxmap = ws.alloc::<i32>(ecap)?;
        let esel = ws.alloc::<i32>(1)?;
        let offs_g = ws.alloc::<u64>(ecap)?;
        let offs_u = ws.alloc::<u64>(ecap)?;
        let offs_d = ws.alloc::<u64>(ecap)?;
        self.l("nv_moe_route", LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
               (ids, (m | (k << 8) | (e << 16)) as i32, slotmap, idxmap, esel, offs_g, offs_u, offs_d, mo.g_words, mo.u_words, mo.d_words, &self.local_of))?;
        // T0-c: ONE smem arithmetic for the whole naive path (a1_smem, window-sized). The
        // fused form below sizes for the split factors it picks (ks_gu / ks_d); the ks==1
        // figures here are the conservative gate for 'can we stage at all'.
        let a1 = |mm: usize, kk: usize, ks: usize| -> u32 { a1_smem(mm, kk, ks) };
        if a1(m, h, 1) > 0 && a1(m, mi, 1) > 0 {
            // the fused form (S6 F1/F2): 6 launches for the three expert GEMMs
            let gks = |kk: usize, nn: usize| -> usize {
                let tiles = k.min(e) * nn / 128; let kb = kk / 16;
                let mut ks = 2usize;
                while tiles * ks * 2 <= 512 && kb % (ks * 2) == 0 && kb / (ks * 2) >= 16 { ks *= 2; }
                ks
            };
            let (ks_gu, ks_d) = (gks(h, mi), gks(mi, h));
            let cmb = |nn: usize| LaunchConfig { grid_dim: (((ecap * m * nn / 128) as u32 + 7) / 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let gemm = |nn: usize, ks: usize, kk: usize| LaunchConfig { grid_dim: ((ecap * nn / 128) as u32, ks as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: a1(m, kk, ks) };
            let w = ws.alloc::<f32>(ecap * ks_gu * m * mi)?;
            let yg = ws.alloc::<u16>(ecap * m * mi)?;
            // F13: fused combine when the (expert, tile) counters fit; svh/y are resolved per expert in the kernel
            let fuse = super::gpu::fuse_on() && ecap * (h.max(mi) / 128) <= super::gpu::FUSE_COUNTERS;
            let fzf = |svh: &CudaSlice<u16>, y: u64| if fuse { super::gpu::NvFuse { svh: dptr(svh), y, out32: 0, cnt: dptr(&self.fuse_cnt) } } else { super::gpu::NvFuse::default() };
            self.l("nv_gemm_grouped_a1ks", gemm(mi, ks_gu, h), (&mo.g_tr, offs_g, hbuf, w, (m | (mo.bpb << 8)) as i32, (h | (mi << 16)) as i32, esel, &mo.suh_g, idxmap, 0u64, 0i32, fzf(&mo.svh_g, yg)))?;
            if !fuse { self.l("nv_ks_combine_grouped_had", cmb(mi), (w, &mo.svh_g, idxmap, yg, m as i32, mi as i32, ks_gu as i32, esel))?; }
            let yu = ws.alloc::<u16>(ecap * m * mi)?;
            self.l("nv_gemm_grouped_a1ks", gemm(mi, ks_gu, h), (&mo.u_tr, offs_u, hbuf, w, (m | (mo.bpb << 8)) as i32, (h | (mi << 16)) as i32, esel, &mo.suh_u, idxmap, 0u64, 0i32, fzf(&mo.svh_u, yu)))?;
            if !fuse { self.l("nv_ks_combine_grouped_had", cmb(mi), (w, &mo.svh_u, idxmap, yu, m as i32, mi as i32, ks_gu as i32, esel))?; }
            let wd = ws.alloc::<f32>(ecap * ks_d * m * h)?;
            let yd = ws.alloc::<u16>(ecap * m * h)?;
            self.l("nv_gemm_grouped_a1ks", gemm(h, ks_d, mi), (&mo.d_tr, offs_d, yg, wd, (m | (mo.bpb << 8)) as i32, (mi | (h << 16)) as i32, esel, &mo.suh_d, idxmap, yu, 2i32, fzf(&mo.svh_d, yd)))?;
            if !fuse { self.l("nv_ks_combine_grouped_had", cmb(h), (wd, &mo.svh_d, idxmap, yd, m as i32, h as i32, ks_d as i32, esel))?; }
            return self.moe_combine_ws(x, yd, ids, wts, slotmap, m, ws);
        }
        let xh_e = ws.alloc::<u16>(ecap * m * h)?;
        let yg_raw = ws.alloc::<u16>(ecap * m * mi)?;
        let yg = ws.alloc::<u16>(ecap * m * mi)?;
        let yu_raw = ws.alloc::<u16>(ecap * m * mi)?;
        let yu = ws.alloc::<u16>(ecap * m * mi)?;
        // the grouped GEMM: ~512 resident CTAs via split-K with the staged body (bench: 8 experts
        // 174 -> 160 us); at wider M the expert count alone fills the grid
        // batch invariance (AGENTS 2.4): the split factor is a function of the SHAPE only (sized for
        // the decode grid of top-k experts), never of this step's row count
        let gks = |kk: usize, nn: usize| -> usize {
            let tiles = k.min(e) * nn / 128; let kb = kk / 16;
            let mut ks = 1usize;
            while tiles * ks * 2 <= 512 && kb % (ks * 2) == 0 && kb / (ks * 2) >= 16 { ks *= 2; }
            if a1_smem(m, kk, ks) == 0 { 1 } else { ks }
        };
        let mut grouped = |tr: &CudaSlice<u16>, offs: u64, xh: u64, y: u64, kk: usize, nn: usize, ws: &mut Ws| -> Result<()> {
            let ks = gks(kk, nn);
            if ks > 1 {
                let w = ws.alloc::<f32>(ecap * ks * m * nn)?;
                self.l("nv_gemm_grouped_a1ks", LaunchConfig { grid_dim: ((ecap * nn / 128) as u32, ks as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: a1_smem(m, kk, ks) },
                       (tr, offs, xh, w, (m | (mo.bpb << 8)) as i32, (kk | (nn << 16)) as i32, esel, 0u64, 0u64, 0u64, 1i32, super::gpu::NvFuse::default()))?;
                self.l("nv_ks_combine_grouped", b1(ecap * m * nn), (w, y, m as i32, nn as i32, ks as i32, esel))
            } else {
                self.l("nv_gemm_grouped_xh", LaunchConfig { grid_dim: ((ecap * nn / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: a1_smem(m, kk, 1) },
                       (tr, offs, xh, y, m as i32, kk as i32, nn as i32, mo.bpb as i32, esel))
            }
        };
        self.l("nv_had_suh_multi", b32(ecap * m * (h / 128)), (hbuf, &mo.suh_g, idxmap, xh_e, m as i32, h as i32, esel, 0i32))?;
        grouped(&mo.g_tr, offs_g, xh_e, yg_raw, h, mi, ws)?;
        self.l("nv_had_svh_multi", b32(ecap * m * (mi / 128)), (yg_raw, &mo.svh_g, idxmap, yg, m as i32, mi as i32, esel))?;
        self.l("nv_had_suh_multi", b32(ecap * m * (h / 128)), (hbuf, &mo.suh_u, idxmap, xh_e, m as i32, h as i32, esel, 0i32))?;
        grouped(&mo.u_tr, offs_u, xh_e, yu_raw, h, mi, ws)?;
        self.l("nv_had_svh_multi", b32(ecap * m * (mi / 128)), (yu_raw, &mo.svh_u, idxmap, yu, m as i32, mi as i32, esel))?;
        let n = ecap * m * mi;
        let din = ws.alloc::<u16>(n)?;
        self.l("nv_silu_mul", b1(n), (din, yg, yu, n as i64))?;
        let xhd = ws.alloc::<u16>(n)?;
        self.l("nv_had_suh_multi", b32(ecap * m * (mi / 128)), (din, &mo.suh_d, idxmap, xhd, m as i32, mi as i32, esel, 1i32))?;
        let yd_raw = ws.alloc::<u16>(ecap * m * h)?;
        let yd = ws.alloc::<u16>(ecap * m * h)?;
        grouped(&mo.d_tr, offs_d, xhd, yd_raw, mi, h, ws)?;
        self.l("nv_had_svh_multi", b32(ecap * m * (h / 128)), (yd_raw, &mo.svh_d, idxmap, yd, m as i32, h as i32, esel))?;
        self.moe_combine_ws(x, yd, ids, wts, slotmap, m, ws)
    }

    /// S6b prefill MoE (m > 16): router -> histogram over this rank's experts -> gathered rows (the per-expert
    /// input transform applied in the gather) -> grouped wide GEMMs (gate, up; silu pair -> down) -> per-row
    /// output transforms -> the j-ordered combine. Every expert's weights are read ONCE per chunk.
    fn moe_wide_ws(&self, mo: &GpuMoe, hbuf: u64, x: u64, m: usize, ws: &mut Ws) -> Result<()> {
        let c = &self.cfg;
        let (h, e, k, mi) = (c.hidden_size, c.num_experts, c.num_experts_per_tok, mo.mi_l);
        let el = mo.el;
        let logits = ws.alloc::<f32>(m * e)?;
        self.l("nv_router_logits", LaunchConfig { grid_dim: (((e + 3) / 4) as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
               (logits, hbuf, &mo.router, m as i32, e as i32, h as i32))?;
        let ids = ws.alloc::<i32>(m * k)?;
        let wts = ws.alloc::<f32>(m * k)?;
        self.l("nv_router_topk", LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: (e * 4) as u32 },
               (ids, wts, logits, &mo.bias, e as i32, k as i32, c.norm_topk_prob as i32, mo.scale))?;
        let cnt = ws.alloc::<i32>(el)?;
        let offs_row = ws.alloc::<i32>(el)?;
        let cursor = ws.alloc::<i32>(el)?;
        let esel = ws.alloc::<i32>(1)?;
        self.l("nv_zero_i32", b1(el), (cnt, el as i32))?;
        self.l("nv_moe_hist", b1(m * k), (ids, &self.local_of, cnt, (m * k) as i32))?;
        self.l("nv_moe_offsets", LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 }, (cnt, offs_row, cursor, esel, el as i32))?;
        let rmax = m * k; // rows (an upper bound: every pair owned)
        let xh_g = ws.alloc::<u16>(rmax * h)?;
        let xh_u = ws.alloc::<u16>(rmax * h)?;
        let rowmap = ws.alloc::<i32>(m * k)?;
        let row_e = ws.alloc::<i32>(rmax)?;
        // rows past the gathered count (pairs owned by other ranks) keep expert 0: the per-row transform kernels
        // run over rmax rows and must only ever index valid suh / svh (their padded outputs are never read)
        self.l("nv_zero_i32", b1(rmax), (row_e, rmax as i32))?;
        self.l("nv_moe_gather_had", LaunchConfig { grid_dim: ((m * k) as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
               (hbuf, &mo.suh_g, &mo.suh_u, ids, &self.local_of, cursor, xh_g, xh_u, rowmap, row_e, k as i32, h as i32))?;
        // The path is eager: read the per-expert counts so the grid covers the busiest expert, not the m*k bound.
        self.sync()?;
        let cnts = dtoh::<i32>(cnt, el)?;
        let cmax = (cnts.iter().copied().max().unwrap_or(0).max(1)) as usize;
        let mt = if cmax <= 32 { 2usize } else if cmax <= 64 { 4 } else { 8 };
        let msup = (cmax + mt * 16 - 1) / (mt * 16);
        let smem = (mt * 16 * (8 * 16 + 8) * 2) as u32;
        let wide = |tr: &CudaSlice<u16>, offs_w: &CudaSlice<u64>, xin: u64, y: u64, kk: usize, nn: usize| -> Result<()> {
            self.l("nv_gemm_wide_grouped", LaunchConfig { grid_dim: (msup as u32, (nn / 128) as u32, el as u32), block_dim: (256, 1, 1), shared_mem_bytes: smem },
                   (tr, offs_w, xin, y, offs_row, cnt, esel, kk as i32, nn as i32, mo.bpb as i32, mt as i32))
        };
        let rows_blocks = |nn: usize| LaunchConfig { grid_dim: (((rmax * nn / 128) as u32 + 7) / 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let yg_raw = ws.alloc::<u16>(rmax * mi)?;
        let yg = ws.alloc::<u16>(rmax * mi)?;
        wide(&mo.g_tr, &mo.offs_g, xh_g, yg_raw, h, mi)?;
        self.l("nv_had_svh_rows", rows_blocks(mi), (yg_raw, &mo.svh_g, row_e, yg, rmax as i32, mi as i32))?;
        let yu_raw = ws.alloc::<u16>(rmax * mi)?;
        let yu = ws.alloc::<u16>(rmax * mi)?;
        wide(&mo.u_tr, &mo.offs_g, xh_u, yu_raw, h, mi)?;
        self.l("nv_had_svh_rows", rows_blocks(mi), (yu_raw, &mo.svh_u, row_e, yu, rmax as i32, mi as i32))?;
        let xhd = ws.alloc::<u16>(rmax * mi)?;
        self.l("nv_had_suh_rows_silu", rows_blocks(mi), (yg, yu, &mo.suh_d, row_e, xhd, rmax as i32, mi as i32))?;
        let yd_raw = ws.alloc::<u16>(rmax * h)?;
        let yd = ws.alloc::<u16>(rmax * h)?;
        wide(&mo.d_tr, &mo.offs_d, xhd, yd_raw, mi, h)?;
        self.l("nv_had_svh_rows", rows_blocks(h), (yd_raw, &mo.svh_d, row_e, yd, rmax as i32, h as i32))?;
        let g1m = LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        match &self.tp {
            None => self.l("nv_moe_gather_combine", g1m, (x, yd, rowmap, wts, k as i32, h as i32, 1i32)),
            Some(tp) => {
                let n = m * h;
                ensure!(n <= tp.part.len(), "TP partial buffer too small for {m} prefill rows");
                self.l("nv_moe_gather_combine", g1m, (&tp.part, yd, rowmap, wts, k as i32, h as i32, 0i32))?;
                tp.all_reduce(&self.stream, &tp.part, n)?;
                self.l("nv_add_f32", b1(n), (x, &tp.part, n as i64))
            }
        }
    }

    /// The MoE combine into the residual (TP: into the fp32 partial, reduced, then added).
    fn moe_combine_ws(&self, x: u64, yd: u64, ids: u64, wts: u64, slotmap: u64, m: usize, _ws: &mut Ws) -> Result<()> {
        let (h, k) = (self.cfg.hidden_size, self.cfg.num_experts_per_tok);
        let g1m = LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        match &self.tp {
            None => self.l("nv_moe_combine", g1m, (x, yd, ids, wts, slotmap, k as i32, h as i32, m as i32, 1i32))?,
            Some(tp) => {
                let n = m * h;
                self.l("nv_moe_combine", g1m, (&tp.part, yd, ids, wts, slotmap, k as i32, h as i32, m as i32, 0i32))?;
                tp.all_reduce(&self.stream, &tp.part, n)?;
                self.l("nv_add_f32", b1(n), (x, &tp.part, n as i64))?;
            }
        }
        Ok(())
    }

    // ---- eager entry points (the gates): allocate a workspace, run, copy back ----

}
