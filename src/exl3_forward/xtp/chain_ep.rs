//! `impl FwdModel` TP chains: the row-parallel out-projection chain (`chain_o`, `chain_o_rb`) and the
//! EP combine-reduce (decode and prefill rows). See `mod.rs` for the map.

use super::*;

impl FwdModel {
    /// TP-B: the out-projection chain of a mixer. A row-parallel Quad (ShardSpec::Row — this
    /// rank holds a K slice) produces a PARTIAL: chain_rowpar reduces it; any other Quad is the
    /// plain served chain.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::exl3_forward) fn chain_o(&self, l: &Launcher, q: &Quad, x: &CudaSlice<u16>, xh: &mut CudaSlice<u16>,
                          yraw: &mut CudaSlice<u16>, y: &mut CudaSlice<u16>, m: usize, ws: &CudaSlice<f32>) -> Result<()> {
        if q.rowpar { self.chain_rowpar(l, q, x, xh, yraw, y, m, ws, None, None, None).map(|_| ()) } else { exl3_chain(l, q, x, xh, yraw, y, m, ws) }
    }

    /// chain_o for the prefill (m = c) sites: the plain arm is exl3_chain_rb (recon + cuBLASLt).
    /// TP-I2 item 1: `hook` = the next hc_pre's inject+norm to run per landed row range inside the
    /// row-parallel reduce. Returns true iff the hook ran (every row), i.e. the caller must NOT run
    /// that inject+norm again. TP-SP: `sp` = this rank's row block of a sequence-parallel chunk — the
    /// reduce becomes a reduce-scatter and y (+ the hook) is produced for the OWN rows only.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::exl3_forward) fn chain_o_rb(&self, l: &Launcher, q: &Quad, x: &CudaSlice<u16>, xh: &mut CudaSlice<u16>,
                             yraw: &mut CudaSlice<u16>, y: &mut CudaSlice<u16>, m: usize, ws: &CudaSlice<f32>,
                             rb: &ReconBufs, hook: Option<PfHook>, sp: Option<SpRows>) -> Result<bool> {
        if q.rowpar {
            pf_chain_check_n(q, m, xh, yraw)?;
            self.chain_rowpar(l, q, x, xh, yraw, y, m, ws, Some(rb), hook, sp)
        } else {
            anyhow::ensure!(sp.is_none(), "TP-SP: a sequence-parallel chunk reached a non-row-parallel out projection");
            exl3_chain_rb(l, q, x, xh, yraw, y, m, ws, rb).map(|_| false)
        }
    }

    /// TP-B row-parallel chain: y = svh ⊙ H_N( Σ_ranks W_rᵀ·H_K(x_r ⊙ suh_r) ). Each rank computes
    /// its K-slice's y_raw partial in FP32 (m <= 16: the split-K GEMM's fp32 slabs summed in the
    /// ascending order of exl3_ks_combine, NO f16 rounding; m > 16: the served wide / recon GEMM's
    /// f16 y_raw, widened), ONE FP32 all-reduce (canonical rank0 + rank1 on both ranks), then the
    /// SAME single f16 rounding + output Hadamard/svh as the TP=1 chain. Batch invariance: every
    /// step is per row with an m-independent ks (chain_ks of the per-rank shape).
    #[allow(clippy::too_many_arguments)]
    fn chain_rowpar(&self, l: &Launcher, q: &Quad, x: &CudaSlice<u16>, xh: &mut CudaSlice<u16>,
                    yraw: &mut CudaSlice<u16>, y: &mut CudaSlice<u16>, m: usize, ws: &CudaSlice<f32>,
                    rb: Option<&ReconBufs>, hook: Option<PfHook>, sp: Option<SpRows>) -> Result<bool> {
        if let Some(fx) = q.fx.as_deref() {
            return self.fx_chain_rowpar(l, q, fx, x, yraw, y, m, ws, rb, hook, sp);
        }
        let tp = self.tp.as_ref().context("a row-parallel Quad without a TP attachment")?;
        let n = q.n as usize;
        let mn = m * n;
        anyhow::ensure!(mn <= tp.part.len(), "row-parallel chain m={m} N={n} exceeds the TP partial buffer ({})", tp.part.len());
        anyhow::ensure!(mn <= yraw.len() && mn <= y.len(), "row-parallel chain m={m} N={n}: y_raw/y too small");
        let part = tp.part_ptr();
        let g = |e: usize| (((e + 255) / 256) as u32, 1, 1);
        if m <= 16 {
            xqlaunch!(l, "exl3_had_suh", (m as u32 * (q.k as u32 / 128), 1, 1), (32, 1, 1), 0, (x, &q.suh, &mut *xh, q.k))?;
            if let Some(d) = tp.dec_ok(mn) {
                // TP-G: the partial's producer carries K1; K2 writes the f16 y_raw (the cvt folded)
                let arrive = *d.arrive.device_ptr() as u64;
                let yp = *yraw.device_ptr() as u64;
                if q.ks > 1 {
                    let kb = (q.k as u32) / 16;
                    anyhow::ensure!(kb % q.ks == 0, "chain ks {} does not divide kb {}", q.ks, kb);
                    anyhow::ensure!(q.ks as usize * mn <= ws.len(), "chain_ws too small: need {} have {}", q.ks as usize * mn, ws.len());
                    xqlaunch!(l, "exl3_hmma_gemm_ks", ((q.n as u32) / 128, q.ks, 1), (256, 1, 1), 0,
                             (&q.tr, &mut *xh, ws, m as i32, q.k, q.n, q.k2))?;
                    xqlaunch!(l, "xq_ks_combine_f32_k1l", g(mn), (256, 1, 1), 0,
                             (tp.dec_ctx(), ws, part, m as i32, q.n, q.ks as i32, arrive))?;
                } else {
                    xqlaunch!(l, "exl3_hmma_gemm", ((q.n as u32) / 128, 1, 1), (256, 1, 1), 0,
                             (&q.tr, &mut *xh, &mut *yraw, m as i32, q.k, q.n, q.k2))?;
                    xqlaunch!(l, "xq_cvt_f16_f32_k1l", g(mn), (256, 1, 1), 0, (tp.dec_ctx(), part, yp, mn as i32, arrive))?;
                }
                tp.dec_sum_finish(l, d, yp, part, mn)?;
                xqlaunch!(l, "exl3_had_svh", (m as u32 * (q.n as u32 / 128), 1, 1), (32, 1, 1), 0, (&*yraw, &q.svh, &mut *y, q.n))?;
                return Ok(false);
            }
            if q.ks > 1 {
                let kb = (q.k as u32) / 16;
                anyhow::ensure!(kb % q.ks == 0, "chain ks {} does not divide kb {}", q.ks, kb);
                anyhow::ensure!(q.ks as usize * mn <= ws.len(), "chain_ws too small: need {} have {}", q.ks as usize * mn, ws.len());
                xqlaunch!(l, "exl3_hmma_gemm_ks", ((q.n as u32) / 128, q.ks, 1), (256, 1, 1), 0,
                         (&q.tr, &mut *xh, ws, m as i32, q.k, q.n, q.k2))?;
                xqlaunch!(l, "xq_ks_combine_f32", g(mn), (256, 1, 1), 0, (ws, part, m as i32, q.n, q.ks as i32))?;
            } else {
                xqlaunch!(l, "exl3_hmma_gemm", ((q.n as u32) / 128, 1, 1), (256, 1, 1), 0,
                         (&q.tr, &mut *xh, &mut *yraw, m as i32, q.k, q.n, q.k2))?;
                xqlaunch!(l, "xq_cvt_f16_f32", g(mn), (256, 1, 1), 0, (part, &*yraw, mn as i64))?;
            }
        } else {
            chain_wide_raw(l, q, x, xh, yraw, m, rb)?;
            if let Some(sp) = sp {
                // TP-SP: every rank's partial over ALL rows (the mixer ran on the gathered x) ->
                // reduce-scatter on the lossless f16 wire (each rank ships the peer's rows, sums its own:
                // the all-reduce's canonical fp32 sum + single f16 rounding on those rows) -> the output
                // Hadamard + svh on the OWN rows (one CTA = one (row, 128-block): row-offset pointers).
                anyhow::ensure!(sp.c == m, "TP-SP: chunk {} vs chain m {m}", sp.c);
                let yp = *yraw.device_ptr() as u64;
                let (svh, yo) = (*q.svh.device_ptr() as u64, *y.device_ptr() as u64);
                let io = crate::tp_xport::Io { src: yp, src_f16: true, out: yp, out_f16: true, wire_f16: true, mid: 0 };
                if let Some(hk) = hook {
                    anyhow::ensure!(hk.y == yo, "prefill overlap hook: y {:#x} is not this chain's output {yo:#x}", hk.y);
                    let mut f = |r0: usize, r1: usize| -> Result<()> {
                        xqlaunch!(l, "exl3_had_svh", (((r1 - r0) * (n / 128)) as u32, 1, 1), (32, 1, 1), 0,
                                  (yp + (r0 * n * 2) as u64, svh, yo + (r0 * n * 2) as u64, q.n))?;
                        hk.inj_norm(l, r0, r1)
                    };
                    tp.sp_reduce_scatter(l, io, n, &sp, Some(crate::tp_xport::SpHook { min_rows: hk.rows, side: hk.dual, f: &mut f }))?;
                    return Ok(true);
                }
                tp.sp_reduce_scatter(l, io, n, &sp, None)?;
                let (r0, r1) = sp.own;
                xqlaunch!(l, "exl3_had_svh", (((r1 - r0) * (n / 128)) as u32, 1, 1), (32, 1, 1), 0,
                          (yp + (r0 * n * 2) as u64, svh, yo + (r0 * n * 2) as u64, q.n))?;
                return Ok(false);
            }
            // TP-F: y_raw (f16) -> f16 wire (exact: y_raw IS the partial) -> canonical fp32 sum -> y_raw
            // (f16, one __float2half_rn) — the two cvt passes fused, half the wire bytes
            let yp = *yraw.device_ptr() as u64;
            if let Some(io) = tp.all_reduce_io(crate::tp_xport::Io { src: yp, src_f16: true, out: yp, out_f16: true, wire_f16: true, mid: 0 }, mn)? {
                if let Some(hk) = hook {
                    // TP-I2 item 1: per landed row range, the output Hadamard + svh (exl3_had_svh's CTA =
                    // one (row, 128-block): row-offset pointers give the full launch's bits on those
                    // rows), then the next hc_pre's inject+norm reading the rows just written.
                    let (svh, yo) = (*q.svh.device_ptr() as u64, *y.device_ptr() as u64);
                    anyhow::ensure!(hk.y == yo, "prefill overlap hook: y {:#x} is not this chain's output {yo:#x}", hk.y);
                    let mut f = |r0: usize, r1: usize| -> Result<()> {
                        xqlaunch!(l, "exl3_had_svh", (((r1 - r0) * (n / 128)) as u32, 1, 1), (32, 1, 1), 0,
                                  (yp + (r0 * n * 2) as u64, svh, yo + (r0 * n * 2) as u64, q.n))?;
                        hk.inj_norm(l, r0, r1)
                    };
                    // TP-I3 (c1) fold: K2m + exl3_had_svh fused per whole-row epoch (the local partial is y_raw,
                    // which the fold never overwrites), then the inject+norm of those rows
                    let mut fk = |ctx: u64, lag: u32, r0: usize, r1: usize| -> Result<()> {
                        let tasks = (r1 - r0) * (n / 128);
                        xqlaunch!(l, "exl3_had_svh_k2", (tasks.div_ceil(8).min(96) as u32, 1, 1), (256, 1, 1), 0,
                                  (ctx, lag, yp + (r0 * n * 2) as u64, svh, yo + (r0 * n * 2) as u64, q.n, (r1 - r0) as i32))?;
                        hk.inj_norm(l, r0, r1)
                    };
                    let fold: Option<&mut dyn FnMut(u64, u32, usize, usize) -> Result<()>> = if hk.fold { Some(&mut fk) } else { None };
                    tp.reduce_io_hooked(l, io, mn, Some(crate::tp_xport::RowHook { row_len: n, min_rows: hk.rows, f: &mut f,
                                                                                   side: hk.dual, fold }))?;
                    return Ok(true);
                }
                tp.reduce_io(l, io, mn)?;
                xqlaunch!(l, "exl3_had_svh", (m as u32 * (q.n as u32 / 128), 1, 1), (32, 1, 1), 0, (&*yraw, &q.svh, &mut *y, q.n))?;
                return Ok(false);
            }
            xqlaunch!(l, "xq_cvt_f16_f32", g(mn), (256, 1, 1), 0, (part, &*yraw, mn as i64))?;
        }
        tp.all_reduce_f32(l, part, mn)?;
        xqlaunch!(l, "xq_cvt_f32_f16", g(mn), (256, 1, 1), 0, (&mut *yraw, part, mn as i64))?;
        xqlaunch!(l, "exl3_had_svh", (m as u32 * (q.n as u32 / 128), 1, 1), (32, 1, 1), 0, (&*yraw, &q.svh, &mut *y, q.n))?;
        Ok(false)
    }

    /// FP4L twin of `chain_rowpar` (an FP4-Local weight, fp4l_fwd.rs): y = Σ_ranks x_r · W_r^T with NO
    /// input or output transform. Same transports, same canonical fp32 sum and single f16 rounding as the
    /// trellis path; where that path applies exl3_had_svh to the reduced y_raw this one copies y_raw -> y
    /// (per landed row range under the prefill hooks). m <= 16: the xf GEMM's fp32 result (ks == 1) or its
    /// fp32 slabs + the trunk's ascending xq_ks_combine_f32*; m > 16: the wide arm's f16 y_raw partial.
    /// The TP-I3 K2+svh fold has no FP4L twin (the hook runs unfolded).
    #[allow(clippy::too_many_arguments)]
    fn fx_chain_rowpar(&self, l: &Launcher, q: &Quad, fx: &super::fp4l_fwd::FxW, x: &CudaSlice<u16>,
                       yraw: &mut CudaSlice<u16>, y: &mut CudaSlice<u16>, m: usize, ws: &CudaSlice<f32>,
                       rb: Option<&ReconBufs>, hook: Option<PfHook>, sp: Option<SpRows>) -> Result<bool> {
        let tp = self.tp.as_ref().context("a row-parallel Quad without a TP attachment")?;
        let n = q.n as usize;
        let mn = m * n;
        anyhow::ensure!(mn <= tp.part.len(), "FP4L row-parallel chain m={m} N={n} exceeds the TP partial buffer ({})", tp.part.len());
        anyhow::ensure!(mn <= yraw.len() && mn <= y.len(), "FP4L row-parallel chain m={m} N={n}: y_raw/y too small");
        let part = tp.part_ptr();
        let g = |e: usize| (((e + 255) / 256) as u32, 1, 1);
        let (yp, yo) = (*yraw.device_ptr() as u64, *y.device_ptr() as u64);
        let copy_rows = |r0: usize, r1: usize| -> Result<()> {
            let ne = ((r1 - r0) * n) as i64;
            xqlaunch!(l, "xq_copy_u16", (((ne as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                      (yo + (r0 * n * 2) as u64, yp + (r0 * n * 2) as u64, ne, 0i64, 0i64))
        };
        if m <= 16 {
            super::fp4l_fwd::fx_rowpar_small(l, q, fx, x, ws, m)?;
            let ks = q.ks.max(1) as i32;
            if let Some(d) = tp.dec_ok(mn) {
                // TP-G: the combine carries K1 (slabs -> part); K2 writes the f16 y_raw
                let arrive = *d.arrive.device_ptr() as u64;
                xqlaunch!(l, "xq_ks_combine_f32_k1l", g(mn), (256, 1, 1), 0,
                         (tp.dec_ctx(), ws, part, m as i32, q.n, ks, arrive))?;
                tp.dec_sum_finish(l, d, yp, part, mn)?;
                copy_rows(0, m)?;
                return Ok(false);
            }
            xqlaunch!(l, "xq_ks_combine_f32", g(mn), (256, 1, 1), 0, (ws, part, m as i32, q.n, ks))?;
        } else {
            let r = rb.context("FP4L row-parallel m > 16 chain without recon buffers")?;
            crate::metrics::fp4l_gemm(fx.kind(), m);
            super::fp4l_fwd::fx_wide(l, q, fx, *x.device_ptr() as u64, yp, m, r)?;
            if let Some(sp) = sp {
                anyhow::ensure!(sp.c == m, "TP-SP: chunk {} vs chain m {m}", sp.c);
                let io = crate::tp_xport::Io { src: yp, src_f16: true, out: yp, out_f16: true, wire_f16: true, mid: 0 };
                if let Some(hk) = hook {
                    anyhow::ensure!(hk.y == yo, "prefill overlap hook: y {:#x} is not this chain's output {yo:#x}", hk.y);
                    let mut f = |r0: usize, r1: usize| -> Result<()> { copy_rows(r0, r1)?; hk.inj_norm(l, r0, r1) };
                    tp.sp_reduce_scatter(l, io, n, &sp, Some(crate::tp_xport::SpHook { min_rows: hk.rows, side: hk.dual, f: &mut f }))?;
                    return Ok(true);
                }
                tp.sp_reduce_scatter(l, io, n, &sp, None)?;
                let (r0, r1) = sp.own;
                copy_rows(r0, r1)?;
                return Ok(false);
            }
            if let Some(io) = tp.all_reduce_io(crate::tp_xport::Io { src: yp, src_f16: true, out: yp, out_f16: true, wire_f16: true, mid: 0 }, mn)? {
                if let Some(hk) = hook {
                    anyhow::ensure!(hk.y == yo, "prefill overlap hook: y {:#x} is not this chain's output {yo:#x}", hk.y);
                    let mut f = |r0: usize, r1: usize| -> Result<()> { copy_rows(r0, r1)?; hk.inj_norm(l, r0, r1) };
                    tp.reduce_io_hooked(l, io, mn, Some(crate::tp_xport::RowHook { row_len: n, min_rows: hk.rows, f: &mut f,
                                                                                   side: hk.dual, fold: None }))?;
                    return Ok(true);
                }
                tp.reduce_io(l, io, mn)?;
                copy_rows(0, m)?;
                return Ok(false);
            }
            xqlaunch!(l, "xq_cvt_f16_f32", g(mn), (256, 1, 1), 0, (part, &*yraw, mn as i64))?;
        }
        tp.all_reduce_f32(l, part, mn)?;
        xqlaunch!(l, "xq_cvt_f32_f16", g(mn), (256, 1, 1), 0, (&mut *yraw, part, mn as i64))?;
        copy_rows(0, m)?;
        Ok(false)
    }

    /// TP-B: the EP prefill combine — this rank's FP32 partial over its local picks (+ its shared
    /// term), one (chunked) all-reduce, one f16 rounding into psc.moe_out.
    pub(in crate::exl3_forward) fn ep_combine_reduce_rows(&self, l: &Launcher, psc: &mut PrefillScratch, moe: &MoeLayer,
                                         ep: &EpShard, c: usize) -> Result<()> {
        let tp = self.tp.as_ref().context("an expert-parallel MoE layer without a TP attachment")?;
        anyhow::ensure!(ep.rank == tp.rank as usize && ep.world == tp.world as usize,
            "EP shard rank {}/{} != attached TP rank {}/{}", ep.rank, ep.world, tp.rank, tp.world);
        let h = self.cfg.hidden_size;
        let topk = self.cfg.num_experts_per_tok;
        let n = c * h;
        anyhow::ensure!(n <= tp.part.len(), "EP prefill combine: c={c} exceeds the partial buffer ({TP_MAX_ROWS} rows)");
        let part = tp.part_ptr();
        let add_sh = (ep.shared_split || ep.rank == 0) as i32;
        // TP-4M1: the vectorised combine (bitwise equal: same per-element fma chain, same 1024-lane sgv tree) when the
        // shapes / alignment it needs hold (h == 2560, topk <= 16, every base 16-B aligned); else the original kernel.
        let al16 = |p: u64| p % 16 == 0;
        // FP4L stored-bf16 gate: the chunk's gate channels (moe_prefill_tail's xf_sg_gate; moe.sg = their constants)
        let gx = self.gate_x_rows(psc, moe);
        let ep2 = if h == 2560 && topk <= 16 && al16(part)
            && al16(*psc.yd.device_ptr() as u64) && al16(*psc.ysh.device_ptr() as u64)
            && al16(*moe.sg.device_ptr() as u64) && al16(gx) {
            Some(if topk <= 10 { "xq_moe_combine_rows_ep2_k10" } else { "xq_moe_combine_rows_ep2_k16" })
        } else { None };
        if let Some(name) = ep2 {
            xqlaunch!(l, name, (c as u32, 1, 1), (320, 1, 1), 0,
                     (part, &psc.yd, &psc.ysh, &psc.cand, &psc.wts, &moe.sg, gx,
                      topk as i32, h as i32, c as i32, add_sh))?;
        } else {
            xqlaunch!(l, "xq_moe_combine_rows_ep", (c as u32, 1, 1), (1024, 1, 1), 4096,
                     (part, &psc.yd, &psc.ysh, &psc.cand, &psc.wts, &moe.sg, gx,
                      topk as i32, h as i32, c as i32, add_sh))?;
        }
        // TP-F: the fp32 partial -> canonical sum -> moe_out (f16), the cvt pass fused into K2m
        let mo = *psc.moe_out.device_ptr() as u64;
        if let Some(sp) = psc.sp_rows {
            // TP-SP: the partial covers ALL rows (the router + EP experts ran on the gathered x); the
            // reduce-scatter gives this rank the canonical sum + one f16 rounding on its OWN rows of
            // moe_out (the all-reduce's values there), with the next inject+norm hooked on those rows.
            anyhow::ensure!(sp.c == c, "TP-SP: chunk {} vs MoE c {c}", sp.c);
            let io = crate::tp_xport::Io { src: part, src_f16: false, out: mo, out_f16: true, wire_f16: false, mid: 0 };
            if let Some(hk) = psc.pf_hook.take() {
                anyhow::ensure!(hk.y == mo, "prefill overlap hook: y {:#x} is not moe_out {mo:#x}", hk.y);
                let mut f = |r0: usize, r1: usize| -> Result<()> { hk.inj_norm(l, r0, r1) };
                tp.sp_reduce_scatter(l, io, h, &sp, Some(crate::tp_xport::SpHook { min_rows: hk.rows, side: hk.dual, f: &mut f }))?;
                psc.pf_hooked = true;
                return Ok(());
            }
            return tp.sp_reduce_scatter(l, io, h, &sp, None);
        }
        if let Some(io) = tp.all_reduce_io(crate::tp_xport::Io { src: part, src_f16: false, out: mo, out_f16: true, wire_f16: false, mid: 0 }, n)? {
            // TP-I2 item 1: the next hc_pre's inject+norm per landed row range (prefill_chunk set the
            // hook; psc.pf_hooked tells it the norm already ran)
            if let Some(hk) = psc.pf_hook.take() {
                anyhow::ensure!(hk.y == mo, "prefill overlap hook: y {:#x} is not moe_out {mo:#x}", hk.y);
                let mut f = |r0: usize, r1: usize| -> Result<()> { hk.inj_norm(l, r0, r1) };
                // TP-I3 (c1) fold: K2m + the inject+norm fused per whole-row epoch (moe_out is never written)
                let mut fk = |ctx: u64, lag: u32, r0: usize, r1: usize| -> Result<()> { hk.inj_norm_k2(l, ctx, lag, part, r0, r1) };
                let fold: Option<&mut dyn FnMut(u64, u32, usize, usize) -> Result<()>> =
                    if hk.fold && hk.hcn == 4 && h % 512 == 0 && h <= 4096 { Some(&mut fk) } else { None };
                tp.reduce_io_hooked(l, io, n, Some(crate::tp_xport::RowHook { row_len: h, min_rows: hk.rows, f: &mut f,
                                                                              side: hk.dual, fold }))?;
                psc.pf_hooked = true;
                return Ok(());
            }
            return tp.reduce_io(l, io, n);
        }
        tp.all_reduce_f32(l, part, n)?;
        xqlaunch!(l, "xq_cvt_f32_f16", ((n as u32).div_ceil(256), 1, 1), (256, 1, 1), 0,
                 (&mut psc.moe_out, part, n as i64))?;
        Ok(())
    }
}
