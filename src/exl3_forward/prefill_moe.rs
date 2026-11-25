//! Prefill-side MLP / MoE methods: `dense_mlp_prefill`, `moe_split_prefill`, `moe_prefill` (kernel class by bit
//! width; the guards precede the launches), the frac bridge / tail and the WP18 select + cross-check.
//! See `mod.rs` for the file map.

use super::*;

impl FwdModel {
    /// Compact per-expert MoE over C chunk rows (deterministic: experts in
    /// ascending id, rows in ascending token order, combine in ascending j).
    /// EXL27B: a dense pack's MLP over a prefill chunk — the three quads as prefill chains (the trunk's
    /// arm: wide trellis, or reconstruct + cuBLASLt from --exl3-recon-min rows), each with its own
    /// suh/svh/bits, silu(g)*u between, the down chain straight into moe_out.
    fn dense_mlp_prefill(&self, l: &Launcher, psc: &mut PrefillScratch, moe: &MoeLayer, c: usize,
                         rb: &ReconBufs) -> Result<()> {
        let mi = moe.sh_down.k as usize; // EXL27C TP-B: the rank's intermediate slice (full at TP=1)
        exl3_chain_rb_dense(l, &moe.sh_gate, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.sh1, c, &psc.chain_ws, rb)?;
        exl3_chain_rb_dense(l, &moe.sh_up, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.sh2, c, &psc.chain_ws, rb)?;
        let sh1p = *psc.sh1.device_ptr() as u64;
        xqlaunch!(l, "xq_silu_mul", (((c * mi) as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                 (&mut psc.sh1, sh1p, &psc.sh2, (c * mi) as i64))?;
        if moe.sh_down.rowpar {
            self.chain_o_rb(l, &moe.sh_down, &psc.sh1, &mut psc.chain_xh, &mut psc.chain_yraw,
                            &mut psc.moe_out, c, &psc.chain_ws, rb, None, None).map(|_| ())?;
        } else {
            exl3_chain_rb_dense(l, &moe.sh_down, &psc.sh1, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.moe_out, c, &psc.chain_ws, rb)?;
        }
        Ok(())
    }

    /// EXL27B: a Qwen3.5-class MoE layer (MoeLayer::split) over a prefill chunk — a copy of
    /// moe_prefill's device-routed arm on the wide-tiles kernel, gate and up as two GEMMs (each its own
    /// suh / svh) over the per-layer offset tables; xh_e = gate input, yd_raw = up input (dead until the
    /// down GEMM); ygu_raw / ygu hold [gate rows | up rows]. Then the shared FN tail (shared expert +
    /// combine, generic in h / mi).
    fn moe_split_prefill(&self, l: &Launcher, psc: &mut PrefillScratch, moe: &MoeLayer, c: usize,
                         rb: &ReconBufs) -> Result<()> {
        let sp = moe.split.as_ref().expect("moe_split_prefill on a non-split layer");
        let (h, ne, topk, mi) = (self.cfg.hidden_size, self.cfg.num_experts, self.cfg.num_experts_per_tok,
                                 self.cfg.moe_intermediate_size);
        anyhow::ensure!(wide_trellis_supports_bits(moe.bits), "EXL3 prefill MoE: {}-bit routed experts", moe.bits);
        anyhow::ensure!(psc.pf_tiles.len() > 1, "EXL3 prefill MoE: no tile lists in the prefill scratch");
        let r = c * topk;
        let mt = pfx1_moe_mt(c);
        xqlaunch!(l, "xq_gemm_f16_f32_rows32", ((ne as u32) / 128, ((c + 31) / 32) as u32, 1), (128, 1, 1), 0,
                 (&mut psc.logits_r, &moe.router, &psc.x, c as i32, ne as i32, h as i32))?;
        xqlaunch!(l, "xq_router_topk", (c as u32, 1, 1), (256, 1, 1), ((256 + ne + 512) * 4) as u32,
                 (&mut psc.ids, &mut psc.wts, &psc.logits_r, ne as i32, topk as i32, c as i32))?;
        // TP-A: under EP, count/place run over THIS rank's local expert space (ids_ep: stacked
        // local id or -1; cand = -1 marks remote picks for the combine), tiles sized to ne_eff —
        // the stacked tables are local, so pf_ident / sp.offs_* address them unchanged.
        let ep = moe.ep.as_ref();
        if let Some(e) = ep {
            xqlaunch!(l, "xq_moe_ids_ep", (r.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0,
                     (&psc.ids, &mut psc.ids_ep, &mut psc.cand, &e.lut, r as i32, ne as i32))?;
        }
        let ids_route = if ep.is_some() { *psc.ids_ep.device_ptr() as u64 } else { *psc.ids.device_ptr() as u64 };
        let ne = ep.map_or(ne, |e| e.n_local);
        let max_tiles = (r / 16 + ne) as u32;
        xqlaunch!(l, "xq_moe_pf_count", (1, 1, 1), (1024, 1, 1), 0,
                 (ids_route, r as i32, ne as i32, &mut psc.cnt, &mut psc.offs_row, &mut psc.pf_tiles,
                  &mut psc.pf_ntiles, (16 * mt) as i32))?;
        xqlaunch!(l, "xq_moe_pf_place", (ne as u32, 1, 1), (256, 1, 1), 0,
                 (ids_route, r as i32, topk as i32, &psc.offs_row, &mut psc.row_tok, &mut psc.row_eidx, &mut psc.cand))?;
        if ep.is_some() {
            // rows [offs_row[ne_eff], r) hold no local pick: give them a valid index (see kernel)
            xqlaunch!(l, "xq_moe_pf_ep_fill", (r.div_ceil(256).min(1024) as u32, 1, 1), (256, 1, 1), 0,
                     (&psc.offs_row, ne as i32, r as i32, &mut psc.row_tok, &mut psc.row_eidx))?;
        }
        xqlaunch!(l, "xq_had_suh_rows", ((r * (h / 128)) as u32, 1, 1), (32, 1, 1), 0,
                 (&psc.x, &moe.suh_gu, &psc.pf_ident, &psc.row_tok, &psc.row_eidx, &mut psc.xh_e, r as i32, h as i32))?;
        xqlaunch!(l, "xq_had_suh_rows", ((r * (h / 128)) as u32, 1, 1), (32, 1, 1), 0,
                 (&psc.x, &sp.suh_u, &psc.pf_ident, &psc.row_tok, &psc.row_eidx, &mut psc.yd_raw, r as i32, h as i32))?;
        let ygr = *psc.ygu_raw.device_ptr() as u64;
        let yur = ygr + (r * mi * 2) as u64;
        let yg = *psc.ygu.device_ptr() as u64;
        let yu = yg + (r * mi * 2) as u64;
        xqlaunch!(l, "exl3_hmma_gemm_wide_tiles", ((mi as u32) / 128, max_tiles, 1), (256, 1, 1), 0,
                 (&moe.gu_tr, &sp.offs_e, &psc.xh_e, ygr, &psc.pf_tiles, &psc.pf_ntiles,
                  &psc.offs_row, &psc.cnt, h as i32, mi as i32, moe.bits as i32, mt as i32))?;
        xqlaunch!(l, "exl3_hmma_gemm_wide_tiles", ((mi as u32) / 128, max_tiles, 1), (256, 1, 1), 0,
                 (&sp.up_tr, &sp.offs_e, &psc.yd_raw, yur, &psc.pf_tiles, &psc.pf_ntiles,
                  &psc.offs_row, &psc.cnt, h as i32, mi as i32, moe.bits as i32, mt as i32))?;
        xqlaunch!(l, "xq_had_svh_rows", ((r * (mi / 128)) as u32, 1, 1), (32, 1, 1), 0,
                 (ygr, &moe.svh_gu, &psc.pf_ident, &psc.row_eidx, yg, r as i32, mi as i32))?;
        xqlaunch!(l, "xq_had_svh_rows", ((r * (mi / 128)) as u32, 1, 1), (32, 1, 1), 0,
                 (yur, &sp.svh_u, &psc.pf_ident, &psc.row_eidx, yu, r as i32, mi as i32))?;
        xqlaunch!(l, "xq_silu_mul", (((r * mi) as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                 (&mut psc.din, yg, yu, (r * mi) as i64))?;
        xqlaunch!(l, "xq_had_suh_rows", ((r * (mi / 128)) as u32, 1, 1), (32, 1, 1), 0,
                 (&psc.din, &moe.suh_d, &psc.pf_ident, &psc.pf_dident, &psc.row_eidx, &mut psc.xhd, r as i32, mi as i32))?;
        xqlaunch!(l, "exl3_hmma_gemm_wide_tiles", ((h as u32) / 128, max_tiles, 1), (256, 1, 1), 0,
                 (&moe.d_tr, &sp.offs_d, &psc.xhd, &mut psc.yd_raw, &psc.pf_tiles, &psc.pf_ntiles,
                  &psc.offs_row, &psc.cnt, mi as i32, h as i32, moe.bits as i32, mt as i32))?;
        xqlaunch!(l, "xq_had_svh_rows", ((r * (h / 128)) as u32, 1, 1), (32, 1, 1), 0,
                 (&psc.yd_raw, &moe.svh_d, &psc.pf_ident, &psc.row_eidx, &mut psc.yd, r as i32, h as i32))?;
        self.moe_prefill_tail(l, psc, moe, c, rb)
    }

    pub(super) fn moe_prefill(&self, l: &Launcher, psc: &mut PrefillScratch, moe: &MoeLayer, c: usize,
                   rb: &ReconBufs) -> Result<()> {
        if moe.dense { return self.dense_mlp_prefill(l, psc, moe, c, rb); } // EXL27B
        if moe.split.is_some() { return self.moe_split_prefill(l, psc, moe, c, rb); } // EXL27B
        if moe.fx.is_some() || moe.router_bf { return fp4l_fwd::moe_prefill(self, l, psc, moe, c, rb); } // FP4L: its own copy
        // K35 (2026-10-31): fractional-k2 routed experts (k2 = 3/5/7 -> K = 1.5/2.5/3.5) prefill on the
        // device-routed arm's wide-tiles kernel (exl3_hmma_gemm_wide_tiles, EXL3_FRAC_BITS case = the
        // HALF twin of exl3_gemm_body_wide) — the arm every default boot takes. Only the host-routed
        // A/B arm (--exl3-moe-recon-min > 0 / --exl3-moe-devroute=0 / no tile lists) has no fractional
        // body; it keeps LEG-B's bridge (the chunk row-blocked through the decode moe()).
        let frac = exl3_is_frac_bits(moe.bits);
        if frac {
            let rmin: usize = crate::opts::var(crate::opt!("exl3-moe-recon-min")).ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            let devroute_off = crate::opts::var(crate::opt!("exl3-moe-devroute")).as_deref() == Ok("0");
            let ep = moe.ep.is_some();
            let want_dev = (rmin == 0 || ep) && psc.pf_tiles.len() > 1 && (!devroute_off || ep);
            if !want_dev { return self.moe_prefill_frac_bridge(l, psc, moe, c); }
        }
        let h = self.cfg.hidden_size;
        let ne = self.cfg.num_experts;
        let topk = self.cfg.num_experts_per_tok;
        let mi = self.cfg.moe_intermediate_size;
        // PFX1 (a): MT by chunk rows (bitwise across MT; --exl3-wide-mt overrides). TUNE S24:
        // prefill.wide_mt 0 = this rule; 1|2|4|8 = a fixed MT for the chunk class.
        let mt: usize = match tune::get(&T_WIDE_MT, &l.cx()) { v @ (1 | 2 | 4 | 8) => v as usize, _ => pfx1_moe_mt(c) };
        // TP-SP2 (--tp-seq-parallel router): the router GEMM + softmax/top-k on this rank's OWN rows of
        // the chunk (the kernel the replicated chunk would pick, on row-offset pointers: one ascending-k
        // chain per (row, expert), one CTA per row for the top-k), then ONE all-gather of the ids and
        // weights rows — every row's routing is the replicated router's bits on both ranks.
        let sp_router = psc.sp_rows.filter(|_| xtp::seq_parallel_router() && (topk * 4) % 16 == 0);
        if let (Some(s), Some(tp)) = (sp_router, self.tp.as_ref()) {
            anyhow::ensure!(s.c == c, "TP-SP router: chunk {} vs MoE c {c}", s.c);
            let (lo, hi) = s.own;
            let m = hi - lo;
            let (lp, xp) = (*psc.logits_r.device_ptr() as u64, *psc.x.device_ptr() as u64);
            let (ip, wp) = (*psc.ids.device_ptr() as u64, *psc.wts.device_ptr() as u64);
            router_rows_launch(l, lp + (lo * ne * 4) as u64, *moe.router.device_ptr() as u64, xp + (lo * h * 2) as u64,
                               m, c, ne, h, tune::get(&T_PF_ROUTER_ROWS, &l.cx()) != 0)?;
            xqlaunch!(l, "xq_router_topk", (m as u32, 1, 1), (256, 1, 1), ((256 + ne + 512) * 4) as u32,
                     (ip + (lo * topk * 4) as u64, wp + (lo * topk * 4) as u64, lp + (lo * ne * 4) as u64,
                      ne as i32, topk as i32, m as i32))?;
            tp.sp_all_gather_multi(l, &[(ip, topk * 4), (wp, topk * 4)], &s)?;
        } else {
        // S-A3-q P3: row-batched bit-exact twin (router weight read once per 16 tokens).
        if tune::get(&T_PF_ROUTER_ROWS, &l.cx()) != 0 {
            if pfx1_router_rows8(c, ne, h) {
                // PFX1 (d): the same per-(m, n) ascending-k chain on (ne/64) x ceil(c/8) CTAs
                // (rows32 fills 4 x ceil(c/32) CTAs: <= 1 warp per SM at short chunks).
                xqlaunch!(l, "xq_gemm_f16_f32_rows8", ((ne as u32) / 64, c.div_ceil(8) as u32, 1), (64, 1, 1), 0,
                         (&mut psc.logits_r, &moe.router, &psc.x, c as i32, ne as i32, h as i32))?;
            } else if let Some((kn, bm)) = router_tile_kernel(c, ne, h) {
                // TP-4R1: register-tiled, bitwise == rows32 (same ascending-k chain per output)
                xqlaunch!(l, kn, ((ne as u32) / 64, c.div_ceil(bm) as u32, 1), (128, 1, 1), 0,
                         (&mut psc.logits_r, &moe.router, &psc.x, c as i32, ne as i32, h as i32))?;
            } else {
                xqlaunch!(l, "xq_gemm_f16_f32_rows32", ((ne as u32) / 128, ((c + 31) / 32) as u32, 1), (128, 1, 1), 0,
                         (&mut psc.logits_r, &moe.router, &psc.x, c as i32, ne as i32, h as i32))?;
            }
        } else {
            xqlaunch!(l, "xq_gemm_f16_f32", ((ne as u32) / 128, c as u32, 1), (128, 1, 1), 0,
                     (&mut psc.logits_r, &moe.router, &psc.x, c as i32, ne as i32, h as i32))?;
        }
        xqlaunch!(l, "xq_router_topk", (c as u32, 1, 1), (256, 1, 1), ((256 + ne + 512) * 4) as u32,
                 (&mut psc.ids, &mut psc.wts, &psc.logits_r, ne as i32, topk as i32, c as i32))?;
        }
        // S-A3-u D: device-side routing (no host sync) whenever no expert reconstructs (the
        // default). --exl3-moe-devroute=0 = the host routing A/B.
        let moe_rmin: usize = crate::opts::var(crate::opt!("exl3-moe-recon-min")).ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let ep = moe.ep.as_ref();
        // TP2BIT5 #26 (AGENTS 1b-bis.8, no-refusals): under EP the host-routed arm has no
        // cross-rank gather, so it can never run — the "not EP-wired" refusal fired on rank 1's
        // first prefill and aborted the boot; the four flags it suggested were all no-ops
        // (recon-min already 0; devroute already on; no-shard-mixers unparsed; pf-kernel
        // never reached — and `devroute off`, its own best suggestion, disabled EP outright).
        // Never refuse for a posture we can serve: prefer the device-routed EP arm whenever the
        // scratch carries tile lists (it always does under EP), print ONE clear line, count it.
        let devroute_off = crate::opts::var(crate::opt!("exl3-moe-devroute")).as_deref() == Ok("0");
        let want_dev = (moe_rmin == 0 || ep.is_some()) && psc.pf_tiles.len() > 1 && (!devroute_off || ep.is_some());
        if (devroute_off || moe_rmin > 0) && ep.is_some() {
            static DEVROUTE_FB_ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            crate::metrics::exl3_devroute_fallback();
            if !DEVROUTE_FB_ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!("[exl3] #26 fallback: {} asked under EP; serving the device-routed \
expert-parallel arm instead (the host-routed path has no cross-rank gather).",
                          if devroute_off { "--exl3-moe-devroute=0" } else { "--exl3-moe-recon-min" });
            }
        }
        if want_dev {
            let r = c * topk;
            // TP-B: expert-parallel prefill — the router/top-k above ran replicated (global ids on
            // every rank); count/place run over THIS rank's local expert space (ids_ep: local
            // stacked id or -1, cand = -1 for remote picks), so the tile lists, compact rows and
            // offsets are the rank's own. The stacked tables are local, so pf_ident / pf_offs_*
            // (identity over expert index) address them unchanged.
            let ne_eff = ep.map_or(ne, |e| e.n_local);
            if let Some(e) = ep {
                xqlaunch!(l, "xq_moe_ids_ep", (r.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0,
                         (&psc.ids, &mut psc.ids_ep, &mut psc.cand, &e.lut, r as i32, ne as i32))?;
            }
            let ids_route = if ep.is_some() { *psc.ids_ep.device_ptr() as u64 } else { *psc.ids.device_ptr() as u64 };
            let ne = ne_eff;
            let max_tiles = (r / 16 + ne) as u32;
            // A5-P1 knobs (all bitwise; default OFF): pipelined kernel, WP19 per-expert MT, glue fold.
            let pk = tune::get(&T_PF_MOE_KERNEL, &l.cx()) != 0 && mi % 128 == 0 && h % 128 == 0;
            // S-B9-REL-INTEG D1: both arms switch on `bits` with `default: break;` (a launch that writes nothing):
            // xq_moe_pf* (V 0) decodes K 2..7 and the fractional 1.5 / 2.5 / 3.5 (PFHALF), exl3_hmma_gemm_wide_tiles
            // 3..8 and the fractional set. Refuse before the first launch.
            // #26 Fix 3: per-layer width dispatch -- a layer the pipelined kernel cannot decode falls back to the
            // wide tiles kernel instead of refusing (AGENTS 1b-bis.8).
            let pk_asked = pk;
            let pk = pk && moe_pf_kernel_supports_bits(moe.bits);
            if pk_asked && !pk {
                crate::metrics::exl3_pf_kernel_width_fallback();
                static WFB_ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !WFB_ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[exl3] #26 Fix 3: a prefill layer has K={} experts; the pipelined expert kernel does not decode it (its range is 2..7 bits \
and 1.5/2.5/3.5{}), serving it with the wide tiles kernel instead (per-layer dispatch; --pf-moe-kernel controls the pipelined kernel)",
                              exl3_width_name(moe.bits), if frac { "; K35: fractional widths serve on the wide tiles kernel's HALF body" } else { "" });
                }
            }
            // The ONE honest refusal left: neither prefill kernel decodes this width (0/1/8 bits) --
            // nothing we can launch would write the layer's experts (AGENTS 1b-bis.8).
            anyhow::ensure!(pk || wide_tiles_supports_bits(moe.bits),
                "EXL3 prefill MoE: this layer's K={} routed experts have no prefill kernel in this build \
(xq_moe_pf* decodes 2..7 bits and 1.5/2.5/3.5, exl3_hmma_gemm_wide_tiles 3..8 and 1.5/2.5/3.5); refusing rather than serving garbage", exl3_width_name(moe.bits));
            if frac && pk {
                // PFHALF engagement receipt: rows (token x top-k) this layer's experts serve on the pipelined HALF body.
                crate::metrics::exl3_frac_pf_pipe_rows(moe.k2, r as u64);
                static PFP_ONCE: [std::sync::Once; 8] = [const { std::sync::Once::new() }; 8];
                PFP_ONCE[moe.k2.min(7)].call_once(|| crate::reprintln!("[exl3] PFHALF: K={} (k2 {}) routed experts PREFILL ENGAGED on the pipelined kernel — \
xq_moe_pf bits={} -> xq_moe_pf_body<{},MT,..,HALF> (pf2 masked off: no HALF body); /metrics \
velogb10_exl3_frac_pf_pipelined_rows_total{{k2=\"{}\"}} counts the expert rows", exl3_width_name(moe.bits), moe.k2, moe.bits, moe.k2 / 2, moe.k2));
            } else if frac {
                // K35 engagement receipt: rows (token x top-k) this layer's experts serve on the fractional wide body.
                crate::metrics::exl3_frac_pf_wide_rows(moe.k2, r as u64);
                static PF_ONCE: [std::sync::Once; 8] = [const { std::sync::Once::new() }; 8];
                PF_ONCE[moe.k2.min(7)].call_once(|| eprintln!("[exl3] K35: K={} (k2 {}) routed experts PREFILL ENGAGED — chunks served by \
exl3_hmma_gemm_wide_tiles bits={} -> exl3_gemm_body_wide<{},MT,HALF> on the device-routed arm; /metrics \
velogb10_exl3_frac_pf_wide_rows_total{{k2=\"{}\"}} counts the expert rows", exl3_width_name(moe.bits), moe.k2, moe.bits, moe.k2 / 2, moe.k2));
            }
            let pe = tune::get(&T_PF_MOE_MT_PE, &l.cx()) != 0 && psc.pf_tiles4.len() > 1;
            let fold = pk && tune::get(&T_PF_MOE_FOLD, &l.cx()) != 0;
            // A5-L2: MT-class mask of the smem-ring twin (needs the P1 kernel; 16-B aligned trellis runs).
            // PFHALF: a fractional layer stays on the P1 twin (pf2 has no HALF body: its entry would write nothing).
            let pf2 = if pk && !frac && (moe.gu_words * 2) % 16 == 0 && (moe.d_words * 2) % 16 == 0 {
                tune::get(&T_PF_MOE_PF2, &l.cx()).clamp(0, 15)
            } else { 0 };
            {
                static SEEN: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
                let key = 1u8 | (pk as u8) << 1 | (pe as u8) << 2 | (fold as u8) << 3;
                if SEEN.swap(key, std::sync::atomic::Ordering::Relaxed) != key {
                    crate::dprintln!("A5-P1 prefill MoE: kernel={} mt={} fold={} (c {c}, rule mt {mt})",
                             if pk { "xq_moe_pf (pipelined)" } else { "exl3_hmma_gemm_wide_tiles" },
                             if pe { "per-expert (WP19)" } else { "chunk rule" }, fold as u8);
                }
                static SEEN2: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);
                if SEEN2.swap(pf2, std::sync::atomic::Ordering::Relaxed) != pf2 {
                    crate::dprintln!("A5-L2 prefill MoE: pf2 mask={pf2} (xq_moe_pf2 smem trellis ring for MT classes {:?})",
                             (0..4).filter(|k| pf2 & (1 << k) != 0).map(|k| 1 << k).collect::<Vec<i32>>());
                }
            }
            // (mt, tiles ptr, ntiles ptr, grid-y cap) per launch group
            let tl_base = *psc.pf_tiles4.device_ptr() as u64;
            let nt_base = *psc.pf_ntiles4.device_ptr() as u64;
            let groups: Vec<(usize, u64, u64, u32)> = if pe {
                let caps = moe_pf_class_caps(r, ne);
                (0..4).filter(|&k| caps[k] > 0)
                    .map(|k| (1usize << k, tl_base + (k * max_tiles as usize * 2 * 4) as u64, nt_base + (k * 4) as u64,
                              caps[k] as u32))
                    .collect()
            } else {
                vec![(mt, *psc.pf_tiles.device_ptr() as u64, *psc.pf_ntiles.device_ptr() as u64, max_tiles)]
            };
            if pe {
                xqlaunch!(l, "xq_moe_pf_count4", (1, 1, 1), (1024, 1, 1), 0,
                         (ids_route, r as i32, ne as i32, &mut psc.cnt, &mut psc.offs_row, &mut psc.pf_tiles4,
                          &mut psc.pf_ntiles4, max_tiles as i32))?;
            } else {
                xqlaunch!(l, "xq_moe_pf_count", (1, 1, 1), (1024, 1, 1), 0,
                         (ids_route, r as i32, ne as i32, &mut psc.cnt, &mut psc.offs_row, &mut psc.pf_tiles,
                          &mut psc.pf_ntiles, (16 * mt) as i32))?;
            }
            xqlaunch!(l, "xq_moe_pf_place", (ne as u32, 1, 1), (256, 1, 1), 0,
                     (ids_route, r as i32, topk as i32, &psc.offs_row, &mut psc.row_tok, &mut psc.row_eidx, &mut psc.cand))?;
            if ep.is_some() {
                // rows [offs_row[ne_l], r) hold no local pick: give them a valid index (see kernel)
                xqlaunch!(l, "xq_moe_pf_ep_fill", (r.div_ceil(256).min(1024) as u32, 1, 1), (256, 1, 1), 0,
                         (&psc.offs_row, ne as i32, r as i32, &mut psc.row_tok, &mut psc.row_eidx))?;
            }
            if ep.is_some() {
                // TP-4M1: only rows [0, offs_row[ne_l]) are ever read (the rest hold the ep_fill dummy mapping); the live-row
                // twin skips the dead ones and moves 8 B per lane. Same per-element arithmetic (bitwise equal on live rows).
                let live_end = *psc.offs_row.device_ptr() as u64 + (ne * 4) as u64;
                let nitem = (r * 20) as u64;
                let blocks = nitem.div_ceil(8 * XQ_HSL_U as u64).min(XQ_HSL_BLOCKS as u64).max(1) as u32;
                xqlaunch!(l, XQ_HSL_KERNEL, (blocks, 1, 1), (256, 1, 1), 0,
                         (&psc.x, &moe.suh_gu, &psc.pf_ident, &psc.row_tok, &psc.row_eidx, &mut psc.xh_e, live_end, r as i32, 2560i32))?;
            } else {
                // From GitHub andersc's PACK-PF: the vectorised persistent twin over ALL R rows (live_end =
                // null) on the replicated path too — the same per-element arithmetic as xq_had_suh_rows
                // (bitwise; see the kernel note), 8-byte lanes and 8-warp blocks instead of 1-warp blocks
                // (their ncu: the 1-warp grid reached 19% of DRAM bandwidth).
                let nitem = (r * 20) as u64;
                let blocks = nitem.div_ceil(8 * XQ_HSL_U as u64).min(XQ_HSL_FULL_BLOCKS as u64).max(1) as u32;
                xqlaunch!(l, XQ_HSL_KERNEL, (blocks, 1, 1), (256, 1, 1), 0,
                         (&psc.x, &moe.suh_gu, &psc.pf_ident, &psc.row_tok, &psc.row_eidx, &mut psc.xh_e, 0u64, r as i32, 2560i32))?;
            }
            let kn_gu = (2560 | ((mi * 2) << 16)) as i32;
            let kn_d = (mi | (h << 16)) as i32;
            let (svh_gu, suh_d, svh_d) = (*moe.svh_gu.device_ptr() as u64, *moe.suh_d.device_ptr() as u64,
                                          *moe.svh_d.device_ptr() as u64);
            for &(gmt, tp, np, cap) in &groups {
                if fold {
                    let (f, sm, nt) = moe_pf_pick('g', gmt, pf2, moe.bits);
                    xqlaunch!(l, f, ((mi as u32) * 2 / 256, cap, 1), (nt, 1, 1), sm,
                             (&moe.gu_tr, &moe.pf_offs_gu, &psc.xh_e, &mut psc.xhd, tp, np,
                              &psc.offs_row, &psc.cnt, kn_gu, moe.bits as i32, svh_gu, suh_d))?;
                } else if pk {
                    let (f, sm, _) = moe_pf_pick('t', gmt, pf2, moe.bits);
                    xqlaunch!(l, f, ((mi as u32) * 2 / 128, cap, 1), (256, 1, 1), sm,
                             (&moe.gu_tr, &moe.pf_offs_gu, &psc.xh_e, &mut psc.ygu_raw, tp, np,
                              &psc.offs_row, &psc.cnt, kn_gu, moe.bits as i32, 0u64, 0u64))?;
                } else {
                    xqlaunch!(l, "exl3_hmma_gemm_wide_tiles", ((mi as u32) * 2 / 128, cap, 1), (256, 1, 1), 0,
                             (&moe.gu_tr, &moe.pf_offs_gu, &psc.xh_e, &mut psc.ygu_raw, tp, np,
                              &psc.offs_row, &psc.cnt, 2560i32, (mi * 2) as i32, moe.bits as i32, gmt as i32))?;
                }
            }
            if !fold {
                xqlaunch!(l, "xq_had_svh_rows", ((r * 10) as u32, 1, 1), (32, 1, 1), 0,
                         (&psc.ygu_raw, &moe.svh_gu, &psc.pf_ident, &psc.row_eidx, &mut psc.ygu, r as i32, (mi * 2) as i32))?;
                xqlaunch!(l, "xq_moe_gate_mul", (((r * mi) as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                         (&mut psc.din, &psc.ygu, mi as i64, &psc.moerows))?;
                xqlaunch!(l, "xq_had_suh_rows", ((r * (mi / 128)) as u32, 1, 1), (32, 1, 1), 0,
                         (&psc.din, &moe.suh_d, &psc.pf_ident, &psc.pf_dident, &psc.row_eidx, &mut psc.xhd, r as i32, mi as i32))?;
            }
            for &(gmt, tp, np, cap) in &groups {
                if fold {
                    let (f, sm, _) = moe_pf_pick('s', gmt, pf2, moe.bits);
                    xqlaunch!(l, f, ((h as u32) / 128, cap, 1), (256, 1, 1), sm,
                             (&moe.d_tr, &moe.pf_offs_d, &psc.xhd, &mut psc.yd, tp, np,
                              &psc.offs_row, &psc.cnt, kn_d, moe.bits as i32, svh_d, 0u64))?;
                } else if pk {
                    let (f, sm, _) = moe_pf_pick('t', gmt, pf2, moe.bits);
                    xqlaunch!(l, f, ((h as u32) / 128, cap, 1), (256, 1, 1), sm,
                             (&moe.d_tr, &moe.pf_offs_d, &psc.xhd, &mut psc.yd_raw, tp, np,
                              &psc.offs_row, &psc.cnt, kn_d, moe.bits as i32, 0u64, 0u64))?;
                } else {
                    xqlaunch!(l, "exl3_hmma_gemm_wide_tiles", ((h as u32) / 128, cap, 1), (256, 1, 1), 0,
                             (&moe.d_tr, &moe.pf_offs_d, &psc.xhd, &mut psc.yd_raw, tp, np,
                              &psc.offs_row, &psc.cnt, mi as i32, h as i32, moe.bits as i32, gmt as i32))?;
                }
            }
            if !fold {
                xqlaunch!(l, "xq_had_svh_rows", ((r * (h / 128)) as u32, 1, 1), (32, 1, 1), 0,
                         (&psc.yd_raw, &moe.svh_d, &psc.pf_ident, &psc.row_eidx, &mut psc.yd, r as i32, h as i32))?;
            }
            return self.moe_prefill_tail(l, psc, moe, c, rb);
        }
        // TP-B: the host-routing A/B path (--exl3-moe-recon-min / MOE_DEVROUTE=0 / no tile lists)
        // is not expert-parallel — refuse loudly instead of computing remote experts' rows.
        anyhow::ensure!(!lt_segs_active(), "PACKPF: the host-routed MoE path is not packed (its per-expert GEMMs see the pass's row counts)");
        anyhow::ensure!(ep.is_none(), "moe_prefill: the host-routing path is not EP-wired (TP-B) — \
                         unset --exl3-moe-recon-min / --exl3-moe-devroute=0 under TP");
        // F3 (owner 2026-10-10): the host-routed arm's grouped kernels decode 3- and 4-bit experts
        // only and no shipped pack has any other width (the 3.05 pack: 3-bit; the 4.05 pack: 4-bit),
        // so say what is served and what to use instead, at the first prefill — the boot warm-up
        // runs this path, which makes it a load-time refusal — instead of leaving a 5/6-bit pack's
        // expert output unwritten (HTTP 200 with garbage).
        anyhow::ensure!((3..=4).contains(&moe.bits),
            "this pack's experts are {}-bit; this engine serves 3 and 4 on the host-routed MoE path \
(--exl3-moe-recon-min / no tile lists). Use a pack whose routed experts are 3- or 4-bit, or one with tile \
lists (the device-routed path serves 3-, 4- and 5-bit)", moe.bits);
        self.dsync()?;
        let ids_h = self.dtoh(&psc.ids)?;
        let r = c * topk;
        // counting sort by expert id; selected experts in ASCENDING id order.
        // Count ONLY the router-written prefix: psc.ids is scratch-sized
        // (c_scratch*topk) and rows >= c*topk hold stale garbage (§2.2).
        let mut cnt = vec![0i32; ne];
        for x in &ids_h[..r] {
            let ei = *x as usize;
            if ei < ne {
                cnt[ei] += 1;
            }
        }
        let uniq: Vec<i32> = (0..ne).filter(|&e| cnt[e] > 0).map(|e| e as i32).collect();
        let esel = uniq.len();
        let mut slot_of = vec![-1i32; ne];
        for (i, &e) in uniq.iter().enumerate() {
            slot_of[e as usize] = i as i32;
        }
        let mut offs_row = vec![0i32; esel + 1];
        for i in 0..esel {
            offs_row[i + 1] = offs_row[i] + cnt[uniq[i] as usize];
        }
        let mut row_tok = vec![0i32; r];
        let mut row_eidx = vec![0i32; r];
        let mut cand = vec![0i32; r];
        let mut cursor = offs_row[..esel].to_vec();
        for t in 0..c {
            for j in 0..topk {
                let e = ids_h[t * topk + j] as usize;
                let s = slot_of[e] as usize;
                let i = cursor[s] as usize;
                cursor[s] += 1;
                row_tok[i] = t as i32;
                row_eidx[i] = s as i32;
                cand[t * topk + j] = i as i32;
            }
        }
        let offs_gu: Vec<u64> = uniq.iter().map(|e| *e as u64 * moe.gu_words).collect();
        let offs_d: Vec<u64> = uniq.iter().map(|e| *e as u64 * moe.d_words).collect();
        // S-A3-f-b §2.1: these uploads MUST be synchronous — they land on the
        // null stream while the compute kernels run on the forked stream; async
        // copies raced the launches and the kernels read half-uploaded routing
        // tables (varying-layer illegal address). The stream is empty here (we
        // just synchronized for the dtoh), so sync copies cost nothing.
        psc.row_tok = self.dev.htod_sync_copy(&row_tok)?;
        psc.row_eidx = self.dev.htod_sync_copy(&row_eidx)?;
        psc.cand = self.dev.htod_sync_copy(&cand)?;
        psc.offs_row = self.dev.htod_sync_copy(&offs_row)?;
        // The kernel indexes cnt by COMPACT slot (e = blockIdx.x < esel), so it
        // must receive the per-slot counts, not the raw-id-indexed counts
        // (uploading `cnt` made n_live read another expert's count -> reads
        // past row_tok; the varying-layer OOB of 2026-09-20).
        let cnt_c: Vec<i32> = uniq.iter().map(|&e| cnt[e as usize]).collect();
        psc.cnt = self.dev.htod_sync_copy(&cnt_c)?;
        psc.idxmap = self.dev.htod_sync_copy(&uniq)?;
        psc.offs_gu = self.dev.htod_sync_copy(&offs_gu)?;
        psc.offs_d = self.dev.htod_sync_copy(&offs_d)?;
        let msup = (c + 16 * mt - 1) / (16 * mt); // per-expert cnt <= C always
        // gate*up side
        xqlaunch!(l, "xq_had_suh_rows", ((r * 20) as u32, 1, 1), (32, 1, 1), 0,
                 (&psc.x, &moe.suh_gu, &psc.idxmap, &psc.row_tok, &psc.row_eidx, &mut psc.xh_e, r as i32, 2560i32))?;
        // S-A3-f-b: the compact grouped kernel is NOT used — its ygu_raw
        // diverges (rel ~1.7 vs the C=1 reference) while xh_e is bit-exact;
        // root cause not yet found (follow-up). Run the PROVEN wide-M kernel
        // per expert on the compact row blocks instead (~esel launches/side).
        // S-A3-f-g per-expert reconstruct threshold. S-A3-q P4: the MoE side has its OWN knob,
        // default 0 = never reconstruct — every live expert runs the grouped trellis-wide GEMM
        // (the trellis is read once per 64-row super-tile; reconstruct wrote a full fp16 copy
        // of each expert, 2.1 s at 2K). --exl3-moe-recon-min=17 restores the f-g split.
        let rmin: usize = crate::opts::var(crate::opt!("exl3-moe-recon-min")).ok()
            .and_then(|v| v.parse().ok()).unwrap_or(0);
        // S-A3-q P1: the below-threshold experts' wide GEMMs run as ONE grouped launch per side
        // (bitwise == the per-expert launches; --exl3-moe-grouped=0 = the per-expert A/B).
        let grouped = tune::get(&T_PF_MOE_GROUPED, &l.cx()) != 0;
        let wide_msup: u32 = cnt_c.iter()
            .filter(|&&ce| ce > 0 && !(rmin > 0 && ce as usize >= rmin))
            .map(|&ce| ((ce as usize + 16 * mt - 1) / (16 * mt)) as u32)
            .max().unwrap_or(0);
        {
            let gu_base = *moe.gu_tr.device_ptr() as u64;
            let xh_base = *psc.xh_e.device_ptr() as u64;
            let yg_base = *psc.ygu_raw.device_ptr() as u64;
            let xh_stride = 2560u64 * 2;
            let yg_stride = 1280u64 * 2;
            for si in 0..esel {
                // S-A3-f-b: per-SLOT count (cnt[uniq[si]]), NOT cnt[si] — cnt is
                // raw-expert-id indexed; reading it by slot skipped/mis-sized
                // expert groups whenever rare experts went unselected (esel<ne)
                // and left their yd rows stale (3/10 of token-0's experts).
                let ce = cnt_c[si];
                if ce == 0 { continue; }
                let row0 = offs_row[si] as u64;
                let eb = uniq[si] as u64;
                if rmin > 0 && (ce as usize) >= rmin {
                    // S-A3-f-g: reconstruct this expert's fused gate|up ONCE
                    // ([K=2560, N=2mi] plain W_dec), then the vendor GEMM on the
                    // compact row block. Same math as the wide kernel (the staged
                    // matrix is what its mma consumes); fp32-accumulate reduction
                    // order differs — the drift class pre-authorized by the brief.
                    exl3_reconstruct(l, gu_base + eb * moe.gu_words * 2, rb.rc, rb.ws,
                                     2560, (mi * 2) as i32, moe.k2 as i32)?;
                    exl3_lt_matmul(l, rb.ws, xh_base + row0 * xh_stride,
                                   yg_base + row0 * yg_stride, 2560, (mi * 2) as usize,
                                   ce as usize, false, rb.ltws, rb.ltws_len)?;
                } else if !grouped {
                    let ms_e = ((ce as u64) + (16 * mt - 1) as u64) / (16 * mt) as u64;
                    xqlaunch!(l, "exl3_hmma_gemm_wide", (ms_e as u32, (mi as u32) * 2 / 128, 1), (256, 1, 1), 0,
                         (gu_base + eb * moe.gu_words * 2, xh_base + row0 * xh_stride,
                          yg_base + row0 * yg_stride, ce as i32, 2560i32, (mi * 2) as i32, moe.bits as i32, mt as i32))?;
                }
            }
            if grouped && wide_msup > 0 {
                xqlaunch!(l, "exl3_hmma_gemm_wide_grouped", (wide_msup, (mi as u32) * 2 / 128, esel as u32), (256, 1, 1), 0,
                         (&moe.gu_tr, &psc.offs_gu, &psc.xh_e, &mut psc.ygu_raw, &psc.offs_row, &psc.cnt,
                          2560i32, (mi * 2) as i32, moe.bits as i32, mt as i32, rmin as i32))?;
            }
        }
        xqlaunch!(l, "xq_had_svh_rows", ((r * 10) as u32, 1, 1), (32, 1, 1), 0,
                 (&psc.ygu_raw, &moe.svh_gu, &psc.idxmap, &psc.row_eidx, &mut psc.ygu, r as i32, (mi * 2) as i32))?;
        // S-A3-f-d night ROOT-CAUSE FIX (the serve-prefill blocker): this launch
        // was missing the kernel's 4th parameter. xq_moe_gate_mul is
        // (din, y_gu, row640, esel) with guard i < (*esel)*row640 — decode passes
        // &sc.esel; prefill passed only 3 args, the 4th param read garbage past
        // the param buffer and the guard killed EVERY thread: din stayed zeros,
        // so every routed-expert row (xhd/yd_raw/yd) was zero and the combine
        // emitted the shared expert only. That is the prefill divergence the
        // trace pinned (yd0 rel=1.00 vs rounding-class upstream), at every chunk
        // width, both prompt classes — junk serve text since f-b/f-g. The kernel
        // needs product(esel,row640) = total compact rows: pass moerows=r with
        // row640 = mi (guard r*mi = r*640 elements; geometry 640/1280 hardcoded
        // in the kernel, matching this pack's moe_intermediate_size).
        self.dsync()?;
        psc.moerows = self.dev.htod_sync_copy(&[r as i32])?;
        // down side (din is already compact: row_tok pass uses the IDENTITY map). NIGHT1 follow-up (PR #51): uploaded here,
        // with the stream drained, rather than after the gate-mul launch, where the blocking upload waited for that kernel
        // inside the driver.
        let ident: Vec<i32> = (0..r as i32).collect();
        let d_ident = self.dev.htod_sync_copy(&ident)?;
        xqlaunch!(l, "xq_moe_gate_mul", (((r * mi) as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                 (&mut psc.din, &psc.ygu, mi as i64, &psc.moerows))?;
        xqlaunch!(l, "xq_had_suh_rows", ((r * (mi / 128)) as u32, 1, 1), (32, 1, 1), 0,
                 (&psc.din, &moe.suh_d, &psc.idxmap, &d_ident, &psc.row_eidx, &mut psc.xhd, r as i32, mi as i32))?;
        {
            let d_base = *moe.d_tr.device_ptr() as u64;
            let xhd_base = *psc.xhd.device_ptr() as u64;
            let yd_base = *psc.yd_raw.device_ptr() as u64;
            let d_xh_stride = (mi as u64) * 2;
            let yd_stride = 2560u64 * 2;
            for si in 0..esel {
                let ce = cnt_c[si]; // per-SLOT count; see gate/up loop note above
                if ce == 0 { continue; }
                let row0 = offs_row[si] as u64;
                let eb = uniq[si] as u64;
                if rmin > 0 && (ce as usize) >= rmin {
                    // S-A3-f-g: reconstruct this expert's down projection
                    // ([K=mi, N=2560]) once, then the vendor GEMM per row block.
                    exl3_reconstruct(l, d_base + eb * moe.d_words * 2, rb.rc, rb.ws,
                                     mi as i32, 2560i32, moe.k2 as i32)?;
                    exl3_lt_matmul(l, rb.ws, xhd_base + row0 * d_xh_stride,
                                   yd_base + row0 * yd_stride, mi as usize, 2560usize,
                                   ce as usize, false, rb.ltws, rb.ltws_len)?;
                } else if !grouped {
                    let ms_e = ((ce as u64) + (16 * mt - 1) as u64) / (16 * mt) as u64;
                    xqlaunch!(l, "exl3_hmma_gemm_wide", (ms_e as u32, (h as u32) / 128, 1), (256, 1, 1), 0,
                         (d_base + eb * moe.d_words * 2, xhd_base + row0 * d_xh_stride,
                          yd_base + row0 * yd_stride, ce as i32, mi as i32, h as i32, moe.bits as i32, mt as i32))?;
                }
            }
            if grouped && wide_msup > 0 {
                xqlaunch!(l, "exl3_hmma_gemm_wide_grouped", (wide_msup, (h as u32) / 128, esel as u32), (256, 1, 1), 0,
                         (&moe.d_tr, &psc.offs_d, &psc.xhd, &mut psc.yd_raw, &psc.offs_row, &psc.cnt,
                          mi as i32, h as i32, moe.bits as i32, mt as i32, rmin as i32))?;
            }
        }
        xqlaunch!(l, "xq_had_svh_rows", ((r * (h / 128)) as u32, 1, 1), (32, 1, 1), 0,
                 (&psc.yd_raw, &moe.svh_d, &psc.idxmap, &psc.row_eidx, &mut psc.yd, r as i32, h as i32))?;
        self.moe_prefill_tail(l, psc, moe, c, rb)
    }

    /// Shared expert + combine — the common end of both prefill MoE routing paths.
    /// LEG-B (2026-10-30): fractional-k2 prefill MoE — row-block the chunk through the decode
    /// moe() (router + experts + shared + combine all inside; the exact sequence the offline gates
    /// proved on this pack). m rows per block: 8 = the widest width tonight's probe-state N-rung
    /// verified EXACT on 1.66. The tail is NOT run: moe() writes complete MoE rows to sc.moe_out,
    /// which the block loop copies back to psc.moe_out. pf_hook stays unconsumed -> the caller's
    /// hc_post takes the unfused (bitwise) path.
    fn moe_prefill_frac_bridge(&self, l: &Launcher, psc: &mut PrefillScratch, moe: &MoeLayer, c: usize) -> Result<()> {
        const BLK: usize = crate::exl3_forward::FRAC_PF_BLK;
        let h = self.cfg.hidden_size;
        {
            static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!("[exl3] LEG-B fractional prefill bridge: k2={} experts row-blocked (x{BLK}) through the decode moe() — no pf kernel decodes this width; /metrics velogb10_exl3_frac_pf_bridge_rows_total counts rows served",
                          moe.k2);
            }
        }
        crate::metrics::exl3_frac_pf_bridge_rows(c as u64);
        for t0 in (0..c).step_by(BLK) {
            let m = BLK.min(c - t0);
            let n = (m * h) as u64;
            let sc = psc.frac_sc.as_mut().expect("frac bridge without the fractional decode scratch");
            xqlaunch!(l, "xq_copy_u16", (((n as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                      (&mut sc.x, &psc.x, n as i64, 0i64, (t0 * h) as i64))?;
            self.moe(l, sc, moe, m)?;
            xqlaunch!(l, "xq_copy_u16", (((n as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                      (&mut psc.moe_out, &sc.moe_out, n as i64, (t0 * h) as i64, 0i64))?;
        }
        Ok(())
    }

    pub(super) fn moe_prefill_tail(&self, l: &Launcher, psc: &mut PrefillScratch, moe: &MoeLayer, c: usize,
                        rb: &ReconBufs) -> Result<()> {
        let h = self.cfg.hidden_size;
        let topk = self.cfg.num_experts_per_tok;
        let mi = moe.sh_down.k as usize; // TP-B: the rank's shared-expert slice (640 at TP=1)
        // FP4L --fp4l-plain exact: the chunk's shared-gate channels (the combine below / the EP combine read them)
        if let Some(sgb) = moe.sg_bf.as_ref() {
            let g = psc.fxg.as_ref().context("FP4L shared gate: the prefill scratch has no channel buffer")?;
            fp4l_fwd::sg_gate(l, *g.device_ptr() as u64, sgb, *psc.x.device_ptr() as u64, h, c)?;
        }
        // shared expert (dense wide path)
        exl3_chain_rb(l, &moe.sh_gate, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.sh1, c, &psc.chain_ws, rb)?;
        exl3_chain_rb(l, &moe.sh_up, &psc.x, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.sh2, c, &psc.chain_ws, rb)?;
        let sh1p = *psc.sh1.device_ptr() as u64;
        xqlaunch!(l, "xq_silu_mul", (((c * mi) as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                 (&mut psc.sh1, sh1p, &psc.sh2, (c * mi) as i64))?;
        exl3_chain_rb(l, &moe.sh_down, &psc.sh1, &mut psc.chain_xh, &mut psc.chain_yraw, &mut psc.ysh, c, &psc.chain_ws, rb)?;
        if let Some(ep) = moe.ep.as_ref() {
            // TP-B: this rank's FP32 partial (local picks + its shared term) -> all-reduce -> f16
            return self.ep_combine_reduce_rows(l, psc, moe, ep, c);
        }
        let gx = self.gate_x_rows(psc, moe);
        xqlaunch!(l, "xq_moe_combine_rows", (c as u32, 1, 1), (1024, 1, 1), 4096,
                 (&mut psc.moe_out, &psc.yd, &psc.ysh, &psc.cand, &psc.wts, &moe.sg, gx,
                  topk as i32, h as i32, c as i32))?;
        // TP-B xtp diagnostic (eager TP=1 reference only): the reduction-site ulp floor, prefill rows.
        if xtp::moe_ulp_armed() && self.tp.is_none() && !stream_capturing(l) {
            self.moe_ulp_perturb_buf(&mut psc.moe_out, c)?;
        }
        Ok(())
    }

    /// One chunked prefill sweep: `toks.len()` consecutive tokens of `slot`
    /// starting at absolute position `pos0`. The chunk's LAST row passes the
    /// final mixer + lm_head; `want_logits` gates that (intermediate chunks
    /// skip the vocab head entirely).
    /// WP18 rung A: the bounded selection walk for the sparse rows [t0, c) of a chunk — writes
    /// psc.qsa_sel / qsa_psel rows [t0, c) bit-identically to qsa_score_prefill_b +
    /// xq_qsa_topk_asc. `xc`: also diff every scored tile against the reference matrix
    /// (psc.qsa_scores, filled by the caller) into the XCHECK counters.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn wp18_select(&self, l: &Launcher, psc: &mut PrefillScratch, blocks_p: u64, params_p: u64,
                   pos0: usize, t0: usize, c: usize, xc: bool, r0: usize) -> Result<()> {
        if xc {
            // counters: [0] differing scores, [1] min packed (row << 16 | block) of a difference, [2] compared
            if let Some((_, _, cnt)) = psc.wp18.as_mut().and_then(|w| w.x.as_mut()) {
                self.dev.htod_sync_copy_into(&[0i32, i32::MAX, 0, 0], cnt)?;
            }
        }
        let w = psc.wp18.as_ref().expect("wp18_select without WP18 scratch");
        let cfg = &self.cfg;
        let ratio = cfg.indexer_compress_ratio;
        let qk_dim = (cfg.indexer_n_heads + 1) * cfg.indexer_head_dim;
        let sel_max = self.qsa_limit();
        let nblk_stride = (self.max_pos / ratio).max(1);
        let (slab_p, ck_p, ci_p) = (*w.slab.device_ptr() as u64, *w.ck.device_ptr() as u64, *w.ci.device_ptr() as u64);
        let (qkb_p, sel_p, psel_p) = (*psc.qsa_qkb.device_ptr() as u64, *psc.qsa_sel.device_ptr() as u64,
                                      *psc.qsa_psel.device_ptr() as u64);
        let (cnt_p, full_p) = match (&w.x, xc) {
            (Some((_, _, cnt)), true) => (*cnt.device_ptr() as u64, *psc.qsa_scores.device_ptr() as u64),
            _ => (0u64, 0u64),
        };
        for st in wp18_plan(pos0 + t0, c - t0, ratio, w.rows) {
            let row = r0 + t0 + st.g0;                     // pass row of the group's first row (PACKPF: + the segment's r0)
            if st.score_grid.0 > 0 {
                xqlaunch!(l, "xq_wp18_score", (st.score_grid.0, st.score_grid.1, 1), (256, 1, 1), 0,
                          (slab_p, qkb_p + (row * qk_dim * 2) as u64, blocks_p, params_p,
                           st.pos_first as i32, st.gn as i32, (st.kt * WP18_T) as i32, qk_dim as i32))?;
                if xc {
                    xqlaunch!(l, "xq_wp18_cmp", (st.gn as u32, 1, 1), (256, 1, 1), 0,
                              (cnt_p, slab_p, full_p, params_p, st.pos_first as i32, row as i32,
                               st.kt as i32, nblk_stride as i32))?;
                }
            }
            xqlaunch!(l, "xq_wp18_merge", (st.gn as u32, 1, 1), (WP18_MT as u32, 1, 1), 0,
                      (sel_p + (row * sel_max * 4) as u64, psel_p + (row * 4) as u64, slab_p, ck_p, ci_p,
                       params_p, st.pos_first as i32, st.kt as i32))?;
        }
        Ok(())
    }

    /// --wp18-xcheck report (eager): the WP18 lists vs the old pair's reference lists for rows
    /// [t0, c) — pos_sel and entries [0, pos_sel] — plus the scored-tile bit diff counters.
    pub(super) fn wp18_xcheck_report(&self, psc: &PrefillScratch, att_i: usize, pos0: usize, t0: usize, c: usize) -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        static ROWS: AtomicUsize = AtomicUsize::new(0);
        static BAD_ROWS: AtomicUsize = AtomicUsize::new(0);
        static SC_TOT: AtomicUsize = AtomicUsize::new(0);
        static SC_BAD: AtomicUsize = AtomicUsize::new(0);
        let (xs, xp, xcnt) = psc.wp18.as_ref().and_then(|w| w.x.as_ref()).expect("WP18 XCHECK buffers");
        let sel_max = self.qsa_limit();
        self.dsync()?;
        let (s1, p1) = (self.dtoh(&psc.qsa_sel)?, self.dtoh(&psc.qsa_psel)?);
        let (s0, p0) = (self.dtoh(xs)?, self.dtoh(xp)?);
        let cnt = self.dtoh(xcnt)?;
        let (mut bad, mut first) = (0usize, String::new());
        let mut entries = 0usize;
        for t in t0..c {
            let n = ((p0[t] + 1).max(0) as usize).min(sel_max);
            let row_ok = p0[t] == p1[t]
                && s0[t * sel_max..t * sel_max + n] == s1[t * sel_max..t * sel_max + n];
            entries += n;
            if !row_ok {
                if bad == 0 {
                    let i = (0..n).find(|&i| s0[t * sel_max + i] != s1[t * sel_max + i]);
                    first = format!(" first: row {t} (pos {}) pos_sel wp18 {} vs {}; first entry diff {:?}",
                                    pos0 + t, p1[t], p0[t],
                                    i.map(|i| (i, s1[t * sel_max + i], s0[t * sel_max + i])));
                }
                bad += 1;
            }
        }
        let rows = c - t0;
        let (sd, stot) = (cnt[0].max(0) as usize, cnt[2].max(0) as usize);
        let k = CALLS.fetch_add(1, Relaxed);
        let (cr, cb) = (ROWS.fetch_add(rows, Relaxed) + rows, BAD_ROWS.fetch_add(bad, Relaxed) + bad);
        let (ct, cs) = (SC_TOT.fetch_add(stot, Relaxed) + stot, SC_BAD.fetch_add(sd, Relaxed) + sd);
        if k < 4 || bad > 0 || sd > 0 || k.is_power_of_two() || (k + 1) % 64 == 0 {
            let sfirst = if sd > 0 { format!(" (first score diff at row {} block {})", cnt[1] >> 16, cnt[1] & 0xFFFF) } else { String::new() };
            crate::dprintln!("WP18_XCHECK #{k} layer {att_i} pos0 {pos0} rows {rows} (t0 {t0}): lists differ {bad}/{rows} rows \
                      ({entries} entries compared){first}; scores bit-differ {sd}/{stot}{sfirst} | cumulative: rows {cb}/{cr} \
                      differ, scores {cs}/{ct}{}", if bad + sd > 0 { "  <-- MISMATCH" } else { "" });
        }
        Ok(())
    }

}
