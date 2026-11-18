//! The MoE forward of one decode / verify step: `moe`, `dense_mlp`, `moe_split` and the large `moe_inner`
//! (router, routed experts, shared expert, combine). Batch-invariant by construction (AGENTS 2.4).
//! See `mod.rs` for the file map; WP20 selection is in `moe_wp20.rs`.

use super::*;

impl FwdModel {
    /// One MoE call. TP-A: an expert-parallel layer (moe.ep = Some) runs the served expert path
    /// on its LOCAL route tables (the filter inside moe_inner) and then turns the rank's routed
    /// (+ rank-0 shared) contribution into the full-rank moe_out with one FP32 all-reduce.
    pub(super) fn moe(&self, l: &Launcher, sc: &mut Scratch, moe: &MoeLayer, m: usize) -> Result<()> {
        if moe.dense { return self.dense_mlp(l, sc, moe, m); } // EXL27B
        if moe.split.is_some() { return self.moe_split(l, sc, moe, m); } // EXL27B
        self.moe_inner(l, sc, moe, m)?;
        if let Some(ep) = moe.ep.as_ref() {
            self.ep_combine_reduce(l, sc, moe, ep, m)?;
        }
        // TP-A xtp diagnostic (eager TP=1 reference only): the reduction-site ulp floor.
        if xtp::moe_ulp_armed() && moe.ep.is_none() && self.tp.is_none() && !stream_capturing(l) {
            self.moe_ulp_perturb(sc, m)?;
        }
        Ok(())
    }

    /// EXL27B: a dense pack's MLP (MoeLayer::dense) at the decode / verify widths — gate|up as one
    /// chain pair (each quad its own suh/svh/bits; split-K, batch-invariant), silu(g)*u + down, into
    /// moe_out. A copy of the shared-expert sequence of moe_inner (shared_down reused as is).
    fn dense_mlp(&self, l: &Launcher, sc: &mut Scratch, moe: &MoeLayer, m: usize) -> Result<()> {
        exl3_chain_pair(l, &moe.sh_gate, &moe.sh_up, &sc.x,
                        &mut sc.chain_xh, &mut sc.chain_yraw, &sc.chain_ws,
                        &mut sc.pair_xh, &mut sc.pair_yraw, &sc.pair_ws,
                        &mut sc.sh1, &mut sc.sh2, m)?;
        self.shared_down(l, sc, moe, m)?;
        let n = (m * self.cfg.hidden_size) as i64;
        xqlaunch!(l, "xq_copy_u16", (((n + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut sc.moe_out, &sc.ysh, n, 0i64, 0i64))?;
        Ok(())
    }

    /// EXL27B: a Qwen3.5-class MoE layer (MoeLayer::split) at the decode / verify widths — a copy of
    /// moe_inner's plain launch sequence with gate and up as two grouped GEMMs (each its own suh / svh):
    /// router GEMV (bit-exact twin) + softmax / top-k / renorm + device routing; xh_e = H(x*suh_g),
    /// yd_raw = H(x*suh_u) (dead until the down GEMM); gate -> ygu_raw -> svh -> ygu, up -> yd -> svh
    /// -> ygu_raw; din = silu(g)*u; down; shared expert; combine.
    fn moe_split(&self, l: &Launcher, sc: &mut Scratch, moe: &MoeLayer, m: usize) -> Result<()> {
        let sp = moe.split.as_ref().expect("moe_split on a non-split layer");
        let (h, ne, topk, mi) = (self.cfg.hidden_size, self.cfg.num_experts, self.cfg.num_experts_per_tok,
                                 self.cfg.moe_intermediate_size);
        let bits = moe.bits as i32;
        xqlaunch!(l, "xq_gemm_f16_f32_v", (((ne as u32) + 255) / 256, m as u32, 1), (256, 1, 1), 0,
                 (&mut sc.logits_r, &moe.router, &sc.x, m as i32, ne as i32, h as i32))?;
        xqlaunch!(l, "xq_router_topk", (m as u32, 1, 1), (256, 1, 1), ((256 + ne + 512) * 4) as u32,
                 (&mut sc.ids, &mut sc.wts, &sc.logits_r, ne as i32, topk as i32, m as i32))?;
        // TP-A: under EP the route tables are rebuilt over THIS rank's local experts (the
        // router/top-k above ran replicated; global ids/wts stay global for the combine).
        if let Some(ep) = moe.ep.as_ref() {
            xqlaunch!(l, "xq_moe_route_ep", (1, 1, 1), (256, 1, 1), 0,
                     (&sc.ids, m as i32, topk as i32, ne as i32, &ep.lut, &mut sc.slotmap, &mut sc.idxmap,
                      &mut sc.esel, &mut sc.offs_gu, &mut sc.offs_d, moe.gu_words, moe.d_words))?;
        } else {
            xqlaunch!(l, "xq_moe_route", (1, 1, 1), (256, 1, 1), 0,
                     (&sc.ids, m as i32, topk as i32, ne as i32, &mut sc.slotmap, &mut sc.idxmap,
                      &mut sc.esel, &mut sc.offs_gu, &mut sc.offs_d, moe.gu_words, moe.d_words))?;
        }
        let emax = sc.idxmap.len() as i32;
        let mi_ = mi as i32;
        let h_ = h as i32;
        let m_ = m as i32;
        xqlaunch!(l, "xq_had_suh_multi", ((emax * m_ * (h_ / 128)) as u32, 1, 1), (32, 1, 1), 0,
                 (&sc.x, &moe.suh_gu, &sc.idxmap, &mut sc.xh_e, m_, h_, &sc.esel, 0i32))?;
        xqlaunch!(l, "xq_had_suh_multi", ((emax * m_ * (h_ / 128)) as u32, 1, 1), (32, 1, 1), 0,
                 (&sc.x, &sp.suh_u, &sc.idxmap, &mut sc.yd_raw, m_, h_, &sc.esel, 0i32))?;
        let gk = grouped_fn(m, h, moe.bits);
        xqlaunch!(l, gk, ((emax * (mi_ / 128)) as u32, 1, 1), (256, 1, 1), a1_smem(m, h),
                 (&moe.gu_tr, &sc.offs_gu, &sc.xh_e, &mut sc.ygu_raw, m_, h_, mi_, bits, &sc.esel))?;
        xqlaunch!(l, gk, ((emax * (mi_ / 128)) as u32, 1, 1), (256, 1, 1), a1_smem(m, h),
                 (&sp.up_tr, &sc.offs_gu, &sc.yd_raw, &mut sc.yd, m_, h_, mi_, bits, &sc.esel))?;
        xqlaunch!(l, "xq_had_svh_multi", ((emax * m_ * (mi_ / 128)) as u32, 1, 1), (32, 1, 1), 0,
                 (&sc.ygu_raw, &moe.svh_gu, &sc.idxmap, &mut sc.ygu, m_, mi_, &sc.esel))?;
        xqlaunch!(l, "xq_had_svh_multi", ((emax * m_ * (mi_ / 128)) as u32, 1, 1), (32, 1, 1), 0,
                 (&sc.yd, &sp.svh_u, &sc.idxmap, &mut sc.ygu_raw, m_, mi_, &sc.esel))?;
        let n = (emax as usize) * m * mi;
        xqlaunch!(l, "xq_silu_mul", (((n as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                 (&mut sc.din, &sc.ygu, &sc.ygu_raw, n as i64))?;
        xqlaunch!(l, "xq_had_suh_multi", ((emax * m_ * (mi_ / 128)) as u32, 1, 1), (32, 1, 1), 0,
                 (&sc.din, &moe.suh_d, &sc.idxmap, &mut sc.xhd, m_, mi_, &sc.esel, 1i32))?;
        xqlaunch!(l, grouped_fn(m, mi, moe.bits), ((emax * (h_ / 128)) as u32, 1, 1), (256, 1, 1), a1_smem(m, mi),
                 (&moe.d_tr, &sc.offs_d, &sc.xhd, &mut sc.yd_raw, m_, mi_, h_, bits, &sc.esel))?;
        xqlaunch!(l, "xq_had_svh_multi", ((emax * m_ * (h_ / 128)) as u32, 1, 1), (32, 1, 1), 0,
                 (&sc.yd_raw, &moe.svh_d, &sc.idxmap, &mut sc.yd, m_, h_, &sc.esel))?;
        exl3_chain_pair(l, &moe.sh_gate, &moe.sh_up, &sc.x,
                        &mut sc.chain_xh, &mut sc.chain_yraw, &sc.chain_ws,
                        &mut sc.pair_xh, &mut sc.pair_yraw, &sc.pair_ws,
                        &mut sc.sh1, &mut sc.sh2, m)?;
        self.shared_down(l, sc, moe, m)?;
        if let Some(ep) = moe.ep.as_ref() {
            // TP-A: local picks + this rank's shared partial -> one FP32 all-reduce -> moe_out
            return self.ep_combine_reduce(l, sc, moe, ep, m);
        }
        xqlaunch!(l, "xq_moe_combine", (m as u32, 1, 1), (1024, 1, 1), 4096,
                 (&mut sc.moe_out, &sc.yd, &sc.ysh, &sc.ids, &sc.wts, &sc.slotmap, &moe.sg, &sc.x,
                  topk as i32, h_, m_))?;
        Ok(())
    }

    fn moe_inner(&self, l: &Launcher, sc: &mut Scratch, moe: &MoeLayer, m: usize) -> Result<()> {
        let h = self.cfg.hidden_size;
        let ne = self.cfg.num_experts;
        let topk = self.cfg.num_experts_per_tok;
        let mi = self.cfg.moe_intermediate_size;
        // WP27: fork base for the shared expert (it reads only sc.x) — while capturing, the
        // shared expert becomes a sibling branch of the routed path, joined at xq_moe_combine.
        // Inert when eager or --wp27-off=1 (then this is the linear sequence below).
        let mut wp27 = crate::exl3_wp27::Cap::open(l.stream.stream, sc.wp27_cls, crate::exl3_wp27::FORK_MOE);
        // S-A3-h item 3 rung 1: cooperative expert-stream schedule — port of the
        // reference implementation's exl3_moe_coop_a/b byte schedule (R-M3 §4; 512-thread blocks,
        // one per (expert-run, 32-col group, projection), cp.async.cg staging
        // with 6-deep pipeline groups). Math is INSTRUCTION-IDENTICAL to
        // xq_gemm_grouped_xh (see the bit-identity contract in
        // kernels/exl3_bench.cu): same exl3_dq_from decode -> same f16
        // fragments -> same mma.sync m16n8k16 K-ascending chain, staging is
        // value-transparent. Default OFF until --probe-exl3-binv / bench-mtp
        // confirm bit-equality on hardware; --moe-coop=1 enables,
        // --moe-coop=0 forces the original grouped path (A/B measurement).
        let moe_coop = tune::get(&T_MOE_COOP, &l.cx()) != 0; // TUNE S2 (legacy alias --moe-coop)
        // K35 (2026-10-31): a fractional expert width (K = 1.5 / 2.5 / 3.5, moe.bits = the EXL3_FRAC_BITS code)
        // has exactly one decode body: the HALF twin of exl3_gemm_body / exl3_gemm_body_a1 behind
        // xq_gemm_grouped_xh. The coop / WP20 / fh / W4-MOE kernels switch on integer widths with
        // `default: break` (a launch that writes nothing), so they are never asked for here:
        // wp20_select declines the code, grouped_fn returns the xh entry, fh needs the a1b3 name,
        // and --moe-coop is overridden below with one line. The round is counted on /metrics
        // (velogb10_exl3_frac_expert_rounds_total{k2}); graph replays are counted by the CudaGraph.
        let frac = exl3_is_frac_bits(moe.bits);
        let moe_coop = if frac && moe_coop {
            static COOP_ONCE: std::sync::Once = std::sync::Once::new();
            COOP_ONCE.call_once(|| eprintln!("[exl3] K35: --moe-coop asked on K={} experts (k2 {}): the coop kernels have no fractional body — serving them with the K-generic grouped chain (xq_gemm_grouped_xh, fractional body) instead", exl3_width_name(moe.bits), moe.k2));
            false
        } else if moe_coop && !moe_coop_supports_bits(moe.bits) {
            // K8EXP (2026-11-07): xq_moe_coop_a/b switch on bits 2..=6 with `default: break`; K = 7 / 8 experts
            // (k2 14 / 16, admitted with this change) take the grouped chain instead (§1b-bis.8: fall back, say so).
            static COOP_W_ONCE: std::sync::Once = std::sync::Once::new();
            COOP_W_ONCE.call_once(|| crate::reprintln!("[exl3] K8EXP: --moe-coop asked on K={} experts (k2 {}): the coop kernels decode K = 2..6 only — serving them with the grouped chain ({}) instead",
                moe.bits, moe.k2, grouped_fn(m, h, moe.bits)));
            false
        } else { moe_coop };
        if matches!(moe.k2, 14 | 16) {
            // K8EXP: the newly admitted integer widths. K = 7 runs the integer expert family (xq_gemm_grouped_a1b7 /
            // the WP20 b7 epilogues / xq_gemm_grouped_xh case 7); K = 8 has only the plain grouped chain on its own
            // entry (xq_gemm_grouped_b8: exl3_gemm_body_a1<8> / exl3_gemm_body<8>) — wp20_select, gu_chunk and
            // w4moe_plans decline bits 8. Counted like the fractional rounds (graph replays included).
            crate::metrics::exl3_frac_expert_round(moe.k2, stream_capturing(l));
            static K78_ONCE: [std::sync::Once; 2] = [const { std::sync::Once::new() }; 2];
            K78_ONCE[(moe.k2 == 16) as usize].call_once(|| crate::reprintln!("[exl3] K8EXP: K={} (k2 {}) routed experts ENGAGED — {}; /metrics velogb10_exl3_k78_expert_rounds_total{{k2=\"{}\"}} counts the rounds (replays included)",
                moe.bits, moe.k2,
                if moe.k2 == 16 { "decode/verify/draft served by the grouped chain on xq_gemm_grouped_b8 -> exl3_gemm_body_a1<8> (A-once, M*(K+8)*2 <= 48 KiB) / exl3_gemm_body<8> (no WP20 / coop / W4-MOE body at K=8)" }
                else { "decode/verify/draft served by the integer expert family at K=7 (xq_gemm_grouped_a1b7 / WP20 b7 epilogues / xq_gemm_grouped_xh case 7)" },
                moe.k2));
        }
        if frac {
            crate::metrics::exl3_frac_expert_round(moe.k2, stream_capturing(l));
            // boot-visible (not debug-only): the once-per-boot liveness line AGENTS §4 asks for
            static FRAC_ONCE: [std::sync::Once; 8] = [const { std::sync::Once::new() }; 8];
            FRAC_ONCE[moe.k2.min(7)].call_once(|| eprintln!("[exl3] K35: K={} (k2 {}) routed experts ENGAGED — decode/verify/draft served by xq_gemm_grouped_xh bits={} -> exl3_gemm_body<{},HALF> (A-once twin exl3_gemm_body_a1<{},HALF> where M*(K+8)*2 <= 48 KiB); /metrics velogb10_exl3_frac_expert_rounds_total{{k2=\"{}\"}} counts the rounds (replays included)",
                exl3_width_name(moe.bits), moe.k2, moe.bits, moe.k2 / 2, moe.k2 / 2, moe.k2));
        }
        // S-A3-f-d step 0 lever 1: router K-split. ks is a pure function of (K, ne)
        // (batch-invariant); partials + fixed ascending-ks combine. Reduction order
        // vs the monolithic kernel differs → drift-audit protocol applies.
        // --exl3-router-ks=0/1 forces the old path (A-B/diagnostics).
        let rks = router_ks(h as i32, ne as i32);
        let rw_need = rks as usize * m * ne;
        // S-A3-z lever 2: fused routing (router GEMV + K-split combine; top-k + route) — bit-identical
        // to the ks=64 path below. --exl3-router-fused=0 = the old launches; --exl3-router-xcheck=1
        // runs both and diffs logits / ids / wts / route tables bitwise (eager only).
        // S-B9-CF-R16: the 9..16-row r16 fold (xq_router_fold_c_r16) is bit-identical, so the fused
        // path serves those widths too when the flag is on. m <= 8 is unchanged.
        // S-B9-REL-FIXB K1: the router selection is decided by the PURE router_entry()
        // (above; test router_selection_table_exhaustive). The 9..16-row width can only
        // ever reach xq_router_fold_c_r16 or the legacy chain: r16 additionally requires
        // the COAL fold and a NON-EP rank, so EP ranks and non-coal runs keep the EXACT
        // pre-R16 (v0.7.2) dispatch above 8 rows. The r16 smem opt-in is hoisted here —
        // a refusal now turns r16 OFF (the width drops to the legacy chain) instead of
        // falling through to xq_router_fold_c, an M <= 8 body.
        let coal = h == 64 * 40 && tune::get(&T_MOE_ROUTER_COAL, &l.cx()) != 0;
        let r16 = m > 8 && m <= 16 && exl3_r16_on() && coal && moe.ep.is_none() && ne <= 512 && {
            static R16_SMO: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *R16_SMO.get_or_init(|| r16_smem_optin(&self.dev, "xq_router_fold_c_r16", RFC_SMEM_R16 as i32))
        };
        // FP4L --fp4l-plain exact (router_bf): the router weight holds the stored bf16 bits, which only the
        // stored-bf16 GEMM reads, so the fused / folded f16 router kernels never take it (the legacy chain below:
        // exact GEMM -> xq_router_topk -> xq_moe_route [-> the EP rebuild], every width, every rank)
        let fused_ok = rks == 64 && ne % 4 == 0 && h % 64 == 0 && (h / 64) % 8 == 0
            && topk <= 16 && tune::get(&T_ROUTER_FUSED, &l.cx()) != 0 && !moe.router_bf;
        // WP10: routing fold — xq_router_fused + xq_router_topk_route in ONE launch (the last GEMV block
        // runs softmax / top-k / renorm / route), bit-identical. Default ON; --wp10-off=1 = the
        // S-A3-z two-kernel path. With ROUTER_XCHECK the fold is diffed against the ks path;
        // --wp10-xcheck=1 diffs it against the S-A3-z pair (both eager only).
        // S-B9-CF-R16: the r16 tail's route scratch is 16*16 deep, so m*topk <= 160 at m > 8
        // (BK <= 160 spans at most 5 of the 8 prefix warps); m <= 8 keeps today's 128 gate.
        let fold_ok = wp10_fold_on() && ne <= 512;
        let topk_cap = m * topk <= if m <= 8 { 128 } else { 160 };
        // S-B9-REL-FIXB K1: the wp10-xcheck reference pair is M <= 8 hardware, so fxc
        // never applies above 8 rows (the pair would read uninitialised rows and the
        // compare would false-bail). rxc keeps the legacy chain as its reference, which
        // serves any m — the r16 fold is still cross-checked at 9..16 rows.
        let rxc_flag = crate::opts::var(crate::opt!("exl3-router-xcheck")).is_ok();
        let fxc_flag = crate::opts::var(crate::opt!("wp10-xcheck")).is_ok() && m <= 8;
        let entry = router_entry(m, moe.ep.is_some(), coal, xtp::route_fold_on(), r16,
                                 fused_ok, fold_ok, topk_cap, false, wp10_gemv_remap(),
                                 rxc_flag || fxc_flag);
        let fold = entry.is_fold();
        let fused = fused_ok && (m <= 8 || fold);
        let rxc = fused && rxc_flag;
        let fxc = fold && fxc_flag;
        if !fused || rxc {
        if m > 8 {
            // S-B9-CF-R16B: the receipt's router half must name the TRUE entry — at m > 8
            // this legacy ks chain is what serves the width whenever the fold does not take
            // (flag off / wp10 off / m*topk > 160), so record it like the fold site does.
            r16_set_router("legacy ks chain (xq_router_ks+combine+topk+moe_route)");
        }
        if moe.router_bf {
            // FP4L: the router logits from the stored bf16 weight (fp32 out, split-K fixup; batch-invariant)
            let fxs = sc.fxs.as_ref().context("FP4L stored-bf16 router: the Scratch has no FP4L buffers")?;
            crate::metrics::fp4l_twin("router");
            fp4l_fwd::bf_gemm(l, &moe.router, *sc.x.device_ptr() as u64, 0, *sc.logits_r.device_ptr() as u64, m, h, ne,
                              &fxs.ws_r, &fxs.cnt_r)?;
        } else if rks > 1 && sc.router_ws.len() >= rw_need {
            xqlaunch!(l, "xq_router_ks", (((ne as u32) + 255) / 256, m as u32, rks), (256, 1, 1), 0,
                     (&mut sc.router_ws, &moe.router, &sc.x, m as i32, ne as i32, h as i32, rks as i32))?;
            let mn = (m * ne) as u32;
            xqlaunch!(l, "xq_router_combine", ((mn + 255) / 256, 1, 1), (256, 1, 1), 0,
                     (&sc.router_ws, &mut sc.logits_r, m as i32, ne as i32, rks as i32))?;
        } else {
        // S-A3-f-f ITEM 1: router GEMV — BIT-EXACT twin (float4 loads, identical
        // ascending fp32 accumulation => identical output bits; AGENTS §7).
        xqlaunch!(l, "xq_gemm_f16_f32_v", (((ne as u32) + 255) / 256, m as u32, 1), (256, 1, 1), 0,
                 (&mut sc.logits_r, &moe.router, &sc.x, m as i32, ne as i32, h as i32))?;
        }
        if moe.router_bf {
            // FP4L: the same softmax / top-k / renorm with a faster exact selection (xf_router_topk)
            fp4l_fwd::router_topk(l, *sc.ids.device_ptr() as u64, *sc.wts.device_ptr() as u64, *sc.logits_r.device_ptr() as u64,
                                  ne, topk, m)?;
        } else {
        xqlaunch!(l, "xq_router_topk", (m as u32, 1, 1), (256, 1, 1), ((256 + ne + 512) * 4) as u32,
                 (&mut sc.ids, &mut sc.wts, &sc.logits_r, ne as i32, topk as i32, m as i32))?;
        }
        // S-A3-f Item 2: device-side routing — the dedup/slotmap/offset build the host
        // used to do after a sync+dtoh (first-seen over ids row-major, identical
        // semantics), now one kernel. No host round-trip; consumer grids are PADDED to
        // the topk cap and early-exit on the device count.
        xqlaunch!(l, "xq_moe_route", (1, 1, 1), (256, 1, 1), 0,
                 (&sc.ids, m as i32, topk as i32, ne as i32, &mut sc.slotmap, &mut sc.idxmap,
                  &mut sc.esel, &mut sc.offs_gu, &mut sc.offs_d, moe.gu_words, moe.d_words))?;
        }
        if fused && m <= 8 && (!fold || fxc) {
            let snap = if rxc {
                self.dsync()?;
                Some((self.dtoh(&sc.logits_r)?, self.dtoh(&sc.ids)?,
                      self.dtoh(&sc.wts)?, self.dtoh(&sc.slotmap)?,
                      self.dtoh(&sc.idxmap)?, self.dtoh(&sc.esel)?,
                      self.dtoh(&sc.offs_gu)?, self.dtoh(&sc.offs_d)?))
            } else { None };
            xqlaunch!(l, "xq_router_fused", ((ne / 4) as u32, 1, 1), (256, 1, 1), 0,
                     (&mut sc.logits_r, &moe.router, &sc.x, m as i32, ne as i32, h as i32))?;
            let f = xq_raw_fn("xq_router_topk_route")
                .ok_or_else(|| anyhow::anyhow!("xq_router_topk_route raw fn unavailable (stale PTX?)"))?;
            let mut a_ids = *sc.ids.device_ptr() as u64; let mut a_wts = *sc.wts.device_ptr() as u64;
            let mut a_lg = *sc.logits_r.device_ptr() as u64;
            let mut a_ne = ne as i32; let mut a_k = topk as i32; let mut a_b = m as i32;
            let mut a_cnt = *sc.route_cnt.device_ptr() as u64;
            let mut a_sm = *sc.slotmap.device_ptr() as u64; let mut a_ix = *sc.idxmap.device_ptr() as u64;
            let mut a_es = *sc.esel.device_ptr() as u64;
            let mut a_og = *sc.offs_gu.device_ptr() as u64; let mut a_od = *sc.offs_d.device_ptr() as u64;
            let mut a_gw = moe.gu_words; let mut a_dw = moe.d_words;
            let mut args: [*mut std::ffi::c_void; 14] = [
                &mut a_ids as *mut u64 as *mut _, &mut a_wts as *mut u64 as *mut _, &mut a_lg as *mut u64 as *mut _,
                &mut a_ne as *mut i32 as *mut _, &mut a_k as *mut i32 as *mut _, &mut a_b as *mut i32 as *mut _,
                &mut a_cnt as *mut u64 as *mut _, &mut a_sm as *mut u64 as *mut _, &mut a_ix as *mut u64 as *mut _,
                &mut a_es as *mut u64 as *mut _, &mut a_og as *mut u64 as *mut _, &mut a_od as *mut u64 as *mut _,
                &mut a_gw as *mut u64 as *mut _, &mut a_dw as *mut u64 as *mut _,
            ];
            let smem = ((256 + ne + 8 * 16 * 2) * 4) as u32;
            let r = unsafe {
                cudarc::driver::sys::cuLaunchKernel(f, m as u32, 1, 1, 256, 1, 1, smem, self.stream.stream,
                    args.as_mut_ptr(), std::ptr::null_mut())
            };
            anyhow::ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "xq_router_topk_route launch ({r:?})");
            if let Some((l0, i0, w0, s0, x0, e0, g0, d0)) = snap {
                self.dsync()?;
                let (l1, i1, w1) = (self.dtoh(&sc.logits_r)?, self.dtoh(&sc.ids)?,
                                    self.dtoh(&sc.wts)?);
                let (s1, x1, e1) = (self.dtoh(&sc.slotmap)?, self.dtoh(&sc.idxmap)?,
                                    self.dtoh(&sc.esel)?);
                let (g1, d1) = (self.dtoh(&sc.offs_gu)?, self.dtoh(&sc.offs_d)?);
                let nl = m * ne; let nk = m * topk; let ns = e0[0].max(0) as usize;
                let dl = (0..nl).filter(|&i| l0[i].to_bits() != l1[i].to_bits()).count();
                let di = (0..nk).filter(|&i| i0[i] != i1[i]).count();
                let dw = (0..nk).filter(|&i| w0[i].to_bits() != w1[i].to_bits()).count();
                let dr = (s0 != s1) as usize + (x0[..ns] != x1[..ns]) as usize + (e0[0] != e1[0]) as usize
                    + (g0[..ns] != g1[..ns]) as usize + (d0[..ns] != d1[..ns]) as usize;
                static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                let c = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if c < 4 || dl + di + dw + dr > 0 {
                    crate::dprintln!("ROUTER_XCHECK m={m}: logits bit-diff {dl}/{nl}, ids {di}/{nk}, wts {dw}/{nk}, route tables {dr}/5 (esel {})", e1[0]);
                }
                if dl + di + dw + dr > 0 { anyhow::bail!("fused routing diverges from the reference path"); }
            }
        }
        let mut ep_routed = false; // TP-G lever #5: the fold built this rank's EP tables
        if fold {
            // Reference for the XCHECK = whatever ran above: the ks path (ROUTER_XCHECK) or the
            // S-A3-z pair (--wp10-xcheck). Eager only (syncs + dtoh).
            let snap = if rxc || fxc {
                self.dsync()?;
                Some((self.dtoh(&sc.logits_r)?, self.dtoh(&sc.ids)?,
                      self.dtoh(&sc.wts)?, self.dtoh(&sc.slotmap)?,
                      self.dtoh(&sc.idxmap)?, self.dtoh(&sc.esel)?,
                      self.dtoh(&sc.offs_gu)?, self.dtoh(&sc.offs_d)?))
            } else { None };
            // W4S (router): the persistent twin xq_router_fold_w4 (bitwise; --w4s-off=router =
            // the fold below). The ROUTER_XCHECK / WP10_XCHECK snapshots above then check it too.
            // A5-K2 (moe.router_coal): the coalesced-weight twin xq_router_fold_c (grid ne/4 x 256,
            // K == 2560 only: the 40-wide slice chains), ahead of W4S router when on.
            // S-B9-REL-FIXB K1: fname/dsmem now come from the SAME pure router_entry()
            // decision made above — ep_fold is m <= 8 only, and the 9..16-row width can
            // only reach xq_router_fold_c_r16 here (the entry never returns an M <= 8
            // body above 8 rows). The old coal16 smem-refusal fallthrough to
            // xq_router_fold_c is gone: a refusal turns r16 off BEFORE the entry is
            // picked, so the width drops to the legacy chain, as the comment promised.
            let w4r = !coal && w4s_on(W4S_ROUTER) && self.w4s_router_fold(l, sc, moe, m)?;
            let entry = if w4r { RouterEntry::FoldW4 } else { entry };
            let fname = match entry {
                RouterEntry::FoldEp => "xq_router_fold_c_ep",
                RouterEntry::FoldR16 => "xq_router_fold_c_r16",
                RouterEntry::FoldCoal => "xq_router_fold_c",
                RouterEntry::FoldW4 => "xq_router_fold_w4",
                RouterEntry::Fold => "xq_router_fold",
                RouterEntry::FoldM0 => "xq_router_fold_m0",
                // `fold` above is exactly entry.is_fold(); reaching here is a logic error.
                RouterEntry::FusedPair | RouterEntry::Legacy => {
                    anyhow::bail!("router_entry returned {entry:?} inside the fold branch")
                }
            };
            let dsmem = match entry {
                RouterEntry::FoldR16 => RFC_SMEM_R16,
                RouterEntry::FoldCoal | RouterEntry::FoldEp => RFC_SMEM,
                _ => 0,
            };
            if coal {
                static K2_ONCE: std::sync::Once = std::sync::Once::new();
                K2_ONCE.call_once(|| crate::dprintln!("A5-K2: router fold COAL ON (xq_router_fold_c, grid {} x 256, smem {RFC_SMEM}; \
                                               --router-coal=0 = the previous fold)", ne / 4));
            }
            if !w4r {
            let f = xq_raw_fn(fname)
                .ok_or_else(|| anyhow::anyhow!("{fname} raw fn unavailable (stale PTX?)"))?;
            static WP10_ONCE: std::sync::Once = std::sync::Once::new();
            WP10_ONCE.call_once(|| crate::dprintln!("WP10: routing fold ON ({fname}; --wp10-off=1 = S-A3-z pair, \
                                              --wp10-gemv=0 = legacy GEMV lane map)"));
            if m > 8 {
                // S-B9-CF-R16B: record the router entry actually serving a 9..16-row group
                // (the liveness receipt's router half; printed by the HC site).
                r16_set_router(fname);
            }
            let mut a_lg = *sc.logits_r.device_ptr() as u64;
            let mut a_w = *moe.router.device_ptr() as u64; let mut a_x = *sc.x.device_ptr() as u64;
            let mut a_m = m as i32; let mut a_n = ne as i32; let mut a_h = h as i32;
            let mut a_cnt = *sc.fold_cnt.device_ptr() as u64;
            let mut a_ids = *sc.ids.device_ptr() as u64; let mut a_wts = *sc.wts.device_ptr() as u64;
            let mut a_k = topk as i32;
            let mut a_sm = *sc.slotmap.device_ptr() as u64; let mut a_ix = *sc.idxmap.device_ptr() as u64;
            let mut a_es = *sc.esel.device_ptr() as u64;
            let mut a_og = *sc.offs_gu.device_ptr() as u64; let mut a_od = *sc.offs_d.device_ptr() as u64;
            let mut a_gw = moe.gu_words; let mut a_dw = moe.d_words;
            let mut a_lut = moe.ep.as_ref().map_or(0u64, |ep| *ep.lut.device_ptr() as u64);
            let mut args: [*mut std::ffi::c_void; 18] = [
                &mut a_lg as *mut u64 as *mut _, &mut a_w as *mut u64 as *mut _, &mut a_x as *mut u64 as *mut _,
                &mut a_m as *mut i32 as *mut _, &mut a_n as *mut i32 as *mut _, &mut a_h as *mut i32 as *mut _,
                &mut a_cnt as *mut u64 as *mut _, &mut a_ids as *mut u64 as *mut _, &mut a_wts as *mut u64 as *mut _,
                &mut a_k as *mut i32 as *mut _,
                &mut a_sm as *mut u64 as *mut _, &mut a_ix as *mut u64 as *mut _, &mut a_es as *mut u64 as *mut _,
                &mut a_og as *mut u64 as *mut _, &mut a_od as *mut u64 as *mut _,
                &mut a_gw as *mut u64 as *mut _, &mut a_dw as *mut u64 as *mut _,
                &mut a_lut as *mut u64 as *mut _,   // read only by xq_router_fold_c_ep (extra params are ignored)
            ];
            if matches!(entry, RouterEntry::FoldEp) {
                static EPF_ONCE: std::sync::Once = std::sync::Once::new();
                EPF_ONCE.call_once(|| crate::dprintln!("TP-G: EP route folded into the router fold (xq_router_fold_c_ep; \
                                                --tp-route-fold=0 = fold + xq_moe_route_ep)"));
            }
            let grid = ((ne / 4) as u32, 1u32, 1u32);
            if pdl_on() {
                unsafe { launch_pss_raw(l.stream.stream, f, grid, (256, 1, 1), dsmem, &mut args) }
                    .with_context(|| format!("launch {fname} (pdl)"))?;
            } else {
                let r = unsafe {
                    cudarc::driver::sys::cuLaunchKernel(f, grid.0, grid.1, grid.2, 256, 1, 1, dsmem, l.stream.stream,
                        args.as_mut_ptr(), std::ptr::null_mut())
                };
                anyhow::ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "{fname} launch ({r:?})");
            }
            ep_routed = matches!(entry, RouterEntry::FoldEp);
            } // !w4r
            if let Some((l0, i0, w0, s0, x0, e0, g0, d0)) = snap {
                self.dsync()?;
                let (l1, i1, w1) = (self.dtoh(&sc.logits_r)?, self.dtoh(&sc.ids)?,
                                    self.dtoh(&sc.wts)?);
                let (s1, x1, e1) = (self.dtoh(&sc.slotmap)?, self.dtoh(&sc.idxmap)?,
                                    self.dtoh(&sc.esel)?);
                let (g1, d1) = (self.dtoh(&sc.offs_gu)?, self.dtoh(&sc.offs_d)?);
                let nl = m * ne; let nk = m * topk; let ns = e0[0].max(0) as usize;
                let dl = (0..nl).filter(|&i| l0[i].to_bits() != l1[i].to_bits()).count();
                let di = (0..nk).filter(|&i| i0[i] != i1[i]).count();
                let dw = (0..nk).filter(|&i| w0[i].to_bits() != w1[i].to_bits()).count();
                let dr = (s0 != s1) as usize + (x0[..ns] != x1[..ns]) as usize + (e0[0] != e1[0]) as usize
                    + (g0[..ns] != g1[..ns]) as usize + (d0[..ns] != d1[..ns]) as usize;
                let first = (0..nl).find(|&i| l0[i].to_bits() != l1[i].to_bits())
                    .map(|i| format!(" first logit diff @[{}][{}] {:e} vs {:e}", i / ne, i % ne, l0[i], l1[i]))
                    .unwrap_or_default();
                static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                let c = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if c < 4 || dl + di + dw + dr > 0 {
                    crate::dprintln!("WP10_XCHECK {fname} vs {} m={m}: logits bit-diff {dl}/{nl}, ids {di}/{nk}, wts {dw}/{nk}, route tables {dr}/5 (esel {}){first}",
                             if fxc { "S-A3-z pair" } else { "ks path" }, e1[0]);
                }
                if dl + di + dw + dr > 0 { anyhow::bail!("WP10 routing fold diverges from the reference path"); }
            }
        }
        // S-A3-f Item 2 selection-identity gate (diagnostic env knob): recompute the OLD
        // host grouping from a dtoh of ids and diff it against the device tables.
        let route_check = crate::opts::var(crate::opt!("exl3-route-check")).is_ok();
        if route_check {
            self.dsync()?;
            let ids_h = self.dtoh(&sc.ids)?;
            let sm_h = self.dtoh(&sc.slotmap)?;
            let ix_h = self.dtoh(&sc.idxmap)?;
            let es_h = self.dtoh(&sc.esel)?;
            let mut uniq: Vec<i32> = Vec::with_capacity(ids_h.len());
            let mut slotmap_ref = vec![-1i32; ne];
            for e in &ids_h {
                let ei = *e as usize;
                if ei < ne && slotmap_ref[ei] == -1 {
                    slotmap_ref[ei] = uniq.len() as i32;
                    uniq.push(*e);
                }
            }
            let ok = es_h.first() == Some(&(uniq.len() as i32))
                && ix_h[..uniq.len()] == uniq[..]
                && (0..ne).all(|e| sm_h[e] == slotmap_ref[e]);
            static ROUTE_ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
            use std::sync::atomic::Ordering;
            if ROUTE_ONCE.swap(false, Ordering::SeqCst) || !ok {
                crate::dprintln!("EXL3_ROUTE_CHECK: {} (esel device {:?} vs host {})",
                         if ok { "PASS" } else { "FAIL" }, es_h.first(), uniq.len());
            }
            if !ok { anyhow::bail!("device MoE routing diverges from host grouping"); }
        }
        // TP-A: expert parallelism — rebuild slotmap/idxmap/esel/offs over THIS rank's experts
        // (the router + top-k above ran replicated, so ids/wts are the same global tables on
        // every rank). Every consumer below indexes suh/svh/trellis by the LOCAL index.
        if let Some(ep) = moe.ep.as_ref().filter(|_| !ep_routed) {
            xqlaunch!(l, "xq_moe_route_ep", (1, 1, 1), (256, 1, 1), 0,
                     (&sc.ids, m as i32, topk as i32, ne as i32, &ep.lut, &mut sc.slotmap, &mut sc.idxmap,
                      &mut sc.esel, &mut sc.offs_gu, &mut sc.offs_d, moe.gu_words, moe.d_words))?;
        }
        if let Some(hb) = self.esel_hist.as_ref() {
            xqlaunch!(l, "xq_esel_hist", (1, 1, 1), (1, 1, 1), 0, (&sc.esel, hb, m as i32, ESEL_HIST_W as i32))?;
        }
        if let Some((hb, _)) = self.expert_hist.as_ref().filter(|_| m <= 16 && moe.li < self.cfg.num_layers) {
            let p = *hb.device_ptr() as u64 + (moe.li * 2 * ne * 4) as u64;
            xqlaunch_raw!(l, "xq_expert_hist", (1u32, 1, 1), (256, 1, 1), 0, (&sc.ids, m as i32, topk as i32, ne as i32, p))?;
        }
        if let Some((buf, ctr, _, _)) = self.route_log.as_ref().filter(|_| m <= 16) {
            let (pb, pc) = (*buf.device_ptr() as u64, *ctr.device_ptr() as u64);
            xqlaunch_raw!(l, "xq_route_log", (1u32, 1, 1), (256, 1, 1), 0,
                          (&sc.ids, m as i32, topk as i32, moe.li as i32, pb, ROUTE_LOG_CAP as i32, pc))?;
        }
        let emax = sc.idxmap.len() as i32; // w*topk — the padded per-step expert cap
        // A5 WP20: expert word diet + MoE glue as last-arrival epilogues (default on, bitwise):
        // xq_had_suh_multi -> xq_moe_gu_epi -> shared expert -> xq_moe_dn_epi_k<topk>. The code
        // below this block is the old launch sequence — --wp20-off=1, --wp20-epi=0, an
        // explicit --moe-fh=1/--moe-coop=1, a top-k without an instance, an ineligible
        // shape or a failed suh verification take it (every such decision is logged once per
        // width by wp20_select — no silent inertness); rung 1 (the diet twin) still runs there
        // wherever the a1b3 entry does, except on the xcheck's reference pass.
        // --exl3-moe-epi-check=1 runs the old path first (no diet: the true pre-WP20
        // reference), then the WP20 path, and diffs xhd / yd / moe_out bitwise (eager passes).
        // FP4L: NVFP4 routed experts (gate|up -> xq_moe_gate_mul -> down, into sc.yd); the WP20 /
        // coop / fh / grouped trellis sequences below are trellis-only and are skipped.
        let fx_moe = moe.fx.as_ref();
        if let Some(fx) = fx_moe { fp4l_fwd::moe_experts_decode(l, sc, fx, h, mi, m, emax)?; }
        let (wp20, rung1_sel) = if fx_moe.is_some() { (None, false) }
                                else { self.wp20_select(l, sc, m, emax as usize, moe_coop, moe.bits) };
        let wp20_chk = wp20.is_some() && wp20_check_on() && !stream_capturing(l);
        if let Some(sel) = wp20 {
            if !wp20_chk { return self.moe_wp20(l, sc, moe, m, emax, sel, &mut wp27); }
            wp20_log_once((6, 0, 0), || "WP20 XCHECK (--exl3-moe-epi-check=1): each eager MoE call runs the pre-WP20 \
                                         sequence (no diet) then the WP20 path and diffs xhd/yd/moe_out bitwise; captured \
                                         graphs record the WP20 path only".to_string());
        }
        let rung1 = rung1_sel && !wp20_chk;
        // Fused per-expert Hadamards (A5 D3 experts, bitwise): suh + a1b3 + svh in ONE launch per GEMM.
        let fh = fx_moe.is_none() && !moe_coop && moe_fh_on()
            && grouped_fn(m, 2560, moe.bits) == "xq_gemm_grouped_a1b3" && grouped_fn(m, mi, moe.bits) == "xq_gemm_grouped_a1b3"
            && 2560 % 128 == 0 && mi % 128 == 0 && (mi * 2) % 128 == 0 && h % 128 == 0;
        if fh {
            xqlaunch!(l, "xq_gemm_grouped_a1b3_fh", ((emax * ((mi * 2) as i32 / 128)) as u32, 1, 1), (256, 1, 1), a1_smem(m, 2560),
                     (&moe.gu_tr, &sc.offs_gu, &sc.x, 0i32, &moe.suh_gu, &moe.svh_gu, &sc.idxmap, &mut sc.ygu,
                      m as i32, 2560i32, (mi * 2) as i32, &sc.esel))?;
            xqlaunch!(l, "xq_moe_gate_mul", (((emax * m as i32 * mi as i32) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                     (&mut sc.din, &sc.ygu, (m * mi as usize) as i64, &sc.esel))?;
            xqlaunch!(l, "xq_gemm_grouped_a1b3_fh", ((emax * (h as i32 / 128)) as u32, 1, 1), (256, 1, 1), a1_smem(m, mi as usize),
                     (&moe.d_tr, &sc.offs_d, &sc.din, 1i32, &moe.suh_d, &moe.svh_d, &sc.idxmap, &mut sc.yd,
                      m as i32, mi as i32, h as i32, &sc.esel))?;
        } else if fx_moe.is_none() {
        xqlaunch!(l, "xq_had_suh_multi", ((emax * m as i32 * 20) as u32, 1, 1), (32, 1, 1), 0,
                 (&sc.x, &moe.suh_gu, &sc.idxmap, &mut sc.xh_e, m as i32, 2560i32, &sc.esel, 0i32))?;
        if moe_coop {
            xqlaunch!(l, "xq_moe_coop_a", ((emax * (((mi * 2) as i32) / 32)) as u32, 1, 1), (256, 1, 1), 0,
                     (&moe.gu_tr, &sc.offs_gu, &sc.xh_e, &mut sc.ygu_raw, m as i32, 2560i32, (mi * 2) as i32, moe.bits as i32, &sc.esel))?;
        } else {
            xqlaunch!(l, wp20_grouped(grouped_fn(m, 2560, moe.bits), rung1, m, 2560), ((emax * 10) as u32, 1, 1), (256, 1, 1), a1_smem(m, 2560),
                     (&moe.gu_tr, &sc.offs_gu, &sc.xh_e, &mut sc.ygu_raw, m as i32, 2560i32, (mi * 2) as i32, moe.bits as i32, &sc.esel))?;
        }
        xqlaunch!(l, "xq_had_svh_multi", ((emax * m as i32 * 10) as u32, 1, 1), (32, 1, 1), 0,
                 (&sc.ygu_raw, &moe.svh_gu, &sc.idxmap, &mut sc.ygu, m as i32, (mi * 2) as i32, &sc.esel))?;
        xqlaunch!(l, "xq_moe_gate_mul", (((emax * m as i32 * mi as i32) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
                 (&mut sc.din, &sc.ygu, (m * mi as usize) as i64, &sc.esel))?;
        xqlaunch!(l, "xq_had_suh_multi", ((emax * m as i32 * (mi as i32 / 128)) as u32, 1, 1), (32, 1, 1), 0,
                 (&sc.din, &moe.suh_d, &sc.idxmap, &mut sc.xhd, m as i32, mi as i32, &sc.esel, 1i32))?;
        if moe_coop {
            xqlaunch!(l, "xq_moe_coop_b", ((emax * (h as i32 / 32)) as u32, 1, 1), (256, 1, 1), 0,
                     (&moe.d_tr, &sc.offs_d, &sc.xhd, &mut sc.yd_raw, m as i32, mi as i32, h as i32, moe.bits as i32, &sc.esel))?;
        } else {
            xqlaunch!(l, wp20_grouped(grouped_fn(m, mi, moe.bits), rung1, m, mi), ((emax * (h as i32 / 128)) as u32, 1, 1), (256, 1, 1), a1_smem(m, mi as usize),
                     (&moe.d_tr, &sc.offs_d, &sc.xhd, &mut sc.yd_raw, m as i32, mi as i32, h as i32, moe.bits as i32, &sc.esel))?;
        }
        xqlaunch!(l, "xq_had_svh_multi", ((emax * m as i32 * (h as i32 / 128)) as u32, 1, 1), (32, 1, 1), 0,
                 (&sc.yd_raw, &moe.svh_d, &sc.idxmap, &mut sc.yd, m as i32, h as i32, &sc.esel))?;
        }
        // FP4L --fp4l-plain exact: the shared-expert gate logit from the stored bf16 gate, as the channels the
        // trunk's combine kernels pair with `sg` (xf_sg_gate; read by the combine below and, on EP layers, by
        // ep_combine_reduce). It reads only sc.x, like the shared expert, and rides the routed branch.
        if let Some(sgb) = moe.sg_bf.as_ref() {
            let fxs = sc.fxs.as_ref().context("FP4L shared gate: the Scratch has no FP4L buffers")?;
            fp4l_fwd::sg_gate(l, *fxs.xg.device_ptr() as u64, sgb, *sc.x.device_ptr() as u64, h, m)?;
        }
        // WP27: routed branch ends here (its tail joins at the combine); the shared expert
        // restarts from the fork base (the hc mixer that wrote sc.x).
        let wp27_routed = wp27.cut()?;
        wp27.side_begin()?;
        exl3_chain_pair(l, &moe.sh_gate, &moe.sh_up, &sc.x,
                        &mut sc.chain_xh, &mut sc.chain_yraw, &sc.chain_ws,
                        &mut sc.pair_xh, &mut sc.pair_yraw, &sc.pair_ws,
                        &mut sc.sh1, &mut sc.sh2, m)?;
        self.shared_down(l, sc, moe, m)?; // W4/DENSE: silu folded into the down (else xq_silu_mul + chain)
        wp27.side_end()?;
        wp27.join(&wp27_routed)?; // WP27: the combine reads yd/ids/wts/slotmap (routed) + ysh (shared)
        // FP4L: the gate's x is the channel buffer when the gate is the stored bf16 one (see above)
        let gx = self.gate_x(sc, moe);
        xqlaunch!(l, "xq_moe_combine", (m as u32, 1, 1), (1024, 1, 1), 4096,
                 (&mut sc.moe_out, &sc.yd, &sc.ysh, &sc.ids, &sc.wts, &sc.slotmap, &moe.sg, gx,
                  topk as i32, h as i32, m as i32))?;
        if let (Some(sel), true) = (wp20, wp20_chk) {
            self.moe_wp20_xcheck(l, sc, moe, m, emax, sel)?;
        }
        Ok(())
    }

}
