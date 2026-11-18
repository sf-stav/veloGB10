//! `impl FwdModel` methods for the WP20 grouped-expert path (select, launch, cross-check), the W4MOE plans and
//! the `--tp-ep-hist` style histogram dumps. See `mod.rs` for the file map.

use super::*;

impl FwdModel {
    /// A5 WP20 dispatch for ONE MoE call. (Some(sel), _) = rung 2 runs (moe_wp20: the epilogue
    /// kernels sel.gu / sel.dn); (None, rung1) = the old launch sequence, with the diet twin iff
    /// `rung1`. Rung 2 needs: mode 2, no --moe-coop / --moe-fh pre-emption, a compiled
    /// down-epilogue instance for the model's top-k (WP20_TOPKS), the served a1b3 shapes both
    /// sides (A-once smem, 3-bit, --exl3-a1b3 honoured), the epilogue smem (tile + the
    /// [m][top-k] slot/weight tables), the scratch counters/sgv/buffers sized for emax = w*top-k,
    /// the raw kernels present in the PTX, and the one-time suh verification (FwdModel::load does
    /// it eagerly; a capture never verifies). EVERY outcome is logged once per (width, reason):
    /// "WP20 ACTIVE m=.." or "WP20 FALLBACK m=..: <reason>" — the package can never be silently
    /// inert again (v1 was, for the whole top-10 model).
    pub(super) fn wp20_select(&self, l: &Launcher, sc: &Scratch, m: usize, emax: usize, moe_coop: bool, bits: usize) -> (Option<Wp20Sel>, bool) {
        if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 7) {
            // CF-P1g + P7 B3 (+ leg 2): the WP20 rung-1/2 kernels are instantiated for K = 2..=7
            // (K = 3/4 diet + top-10; K = 2/5/6/7 SHFL twins + top-10); any other width runs the K-generic chain.
            wp20_log_once((11, bits, 0), || format!("WP20 OFF for K={} experts: the epilogue kernels exist for K = 2..=7 only — \
                                                    running the plain grouped chain ({}{})", exl3_width_name(bits),
                                                    grouped_fn(m, 2560, bits),
                                                    if exl3_is_frac_bits(bits) { ", fractional body" } else { "" }));
            return (None, false);
        }
        let mode = wp20_mode();
        let diet = wp20_diet_on();
        if mode == 0 {
            wp20_log_once((0, 0, 0), || "WP20 OFF (--wp20-off=1): pre-WP20 expert path — SHFL decode, 5 glue launches per MoE layer".to_string());
            return (None, false);
        }
        let preempt = if moe_coop { Some("--moe-coop=1 (coop expert kernels) takes precedence") }
            else if moe_fh_on() { Some("--moe-fh=1 (fused-Hadamard xq_gemm_grouped_a1b3_fh) takes precedence") }
            else { None };
        if mode == 1 {
            wp20_log_once((0, 1, 0), || format!("WP20 mode 1 (--wp20-epi=0): rung 1 only — word-diet twin {} in the pre-WP20 \
                                                 launch sequence; MoE epilogues (rung 2) off by request",
                                                if diet { "on" } else { "OFF too (--wp20-diet=0): effectively the pre-WP20 path" }));
            if let Some(why) = preempt {
                wp20_log_once((5, m, 0), || format!("WP20 FALLBACK rung 1 m={m}: {why} — that path has no diet twin"));
                return (None, false);
            }
            return (None, diet);
        }
        let (h, mi, topk) = (self.cfg.hidden_size, self.cfg.moe_intermediate_size, self.cfg.num_experts_per_tok);
        let (gu, dn) = if bits == 3 {
            (wp20_gu_fn(diet), wp20_dn_fn(topk, diet))
        } else if bits == 4 {
            // K = 4: the diet entries only (no SHFL twin), top-10 only
            ("xq_moe_gu_epi_b4", if diet && topk == 10 { Some("xq_moe_dn_epi_b4_k10") } else { None })
        } else if bits == 5 {
            // P7 B3 (K = 5): SHFL twins only (the word diet is b3/b4-only by static_assert), top-10 only
            ("xq_moe_gu_epi_b5", if topk == 10 { Some("xq_moe_dn_epi_b5_k10") } else { None })
        } else {
            // P7 B3 leg 2 (K = 2/6/7): the same SHFL twin shape as K = 5, top-10 only
            let (g, d) = match bits {
                2 => ("xq_moe_gu_epi_b2", "xq_moe_dn_epi_b2_k10"),
                6 => ("xq_moe_gu_epi_b6", "xq_moe_dn_epi_b6_k10"),
                _ => ("xq_moe_gu_epi_b7", "xq_moe_dn_epi_b7_k10"),
            };
            (g, if topk == 10 { Some(d) } else { None })
        };
        let a1n = match bits { 2 => "xq_gemm_grouped_a1b2", 3 => "xq_gemm_grouped_a1b3", 4 => "xq_gemm_grouped_a1b4",
                               5 => "xq_gemm_grouped_a1b5", 6 => "xq_gemm_grouped_a1b6", _ => "xq_gemm_grouped_a1b7" };
        // CF-P1e: 10..16 rows take the K-chunked gate/up entry (smem from the chunk plan, not the A-once cap)
        let a1_ok = grouped_fn(m, h, bits) == a1n;
        let chunk = if a1_ok { None } else { gu_chunk(m, h, diet, bits) };
        let gu = chunk.map_or(gu, |c| c.0);
        let (sgu, sdn) = (chunk.map_or(a1_smem(m, h), |c| c.1) as usize, a1_smem(m, mi) as usize);
        let (need_gu, need_dn) = ((m * 256).max(4096), m * 256 + m * topk * 8);
        let why: Option<(usize, String)> = if let Some(p) = preempt {
            Some((1, p.to_string()))
        } else if dn.is_none() {
            Some((2, format!("top-k {topk} has no down-epilogue instance (compiled: {WP20_TOPKS:?})")))
        } else if m == 0 || h % 128 != 0 || mi % 128 != 0 {
            Some((3, format!("shape m={m} hidden={h} moe_intermediate={mi} (need m >= 1, both multiples of 128)")))
        } else if !a1_ok && chunk.is_none() {
            Some((4, format!("gate/up GEMM (K={h}) is off the a1b3 entry at this width: A-once smem m*(K+8)*2 = {} B \
                              (> 48 KiB?) or --exl3-a1b3=0", m * (h + 8) * 2)))
        } else if grouped_fn(m, mi, bits) != a1n {
            Some((5, format!("down GEMM (K={mi}) is off the a1b3 entry at this width: A-once smem m*(K+8)*2 = {} B \
                              (> 48 KiB?) or --exl3-a1b3=0", m * (mi + 8) * 2)))
        } else if sgu < need_gu || sdn < need_dn {
            Some((6, format!("epilogue smem: gate/up {sgu} B (need {need_gu}), down {sdn} B (need {need_dn} = tile + [m][top-k] tables)")))
        } else if sc.moe_cnt.len() < emax * (mi / 128) + h / 128 || sc.moe_sgv.len() < m
            || sc.ygu.len() < emax * m * 2 * mi || sc.xhd.len() < emax * m * mi || sc.yd.len() < emax * m * h {
            Some((7, format!("scratch undersized for emax {emax} (counters {}, sgv {}, ygu {}, xhd {}, yd {})",
                             sc.moe_cnt.len(), sc.moe_sgv.len(), sc.ygu.len(), sc.xhd.len(), sc.yd.len())))
        } else if xq_raw_fn(gu).is_none() || dn.and_then(xq_raw_fn).is_none() {
            Some((8, format!("kernel {gu} / {} missing from src/ptx/exl3_bench.ptx (stale PTX?)", dn.unwrap_or("?"))))
        } else {
            None
        };
        let conv: std::result::Result<u32, (usize, String)> = match why {
            Some(w) => Err(w),
            None => match WP20_CONV.get() {
                Some(Some(c)) => Ok(*c),
                Some(None) => Err((9, "the suh verification failed (see the WP20 suh line)".to_string())),
                None if stream_capturing(l) => Err((10, "first eligible call is inside a graph capture before the suh \
                    verification (FwdModel::load verifies eagerly — this graph records the old sequence)".to_string())),
                None => wp20_calibrated(&self.dev).ok_or((9, "the suh verification failed (see the WP20 suh line)".to_string())),
            },
        };
        match conv {
            Ok(c) => {
                let dn = dn.unwrap_or("?");
                wp20_log_once((1, m, topk * 16 + bits), || format!(
                    "WP20 ACTIVE m={m}: rungs 1+2 — xq_had_suh_multi -> {gu} -> shared expert -> {dn} (top-k {topk}, emax {emax}, \
                     suh seq 0x{c:02x} (pinned, verified), decode body {}; smem gate/up {sgu} B, down {sdn} B >= {need_dn}); removed per MoE layer: \
                     xq_had_svh_multi x2, xq_moe_gate_mul, down xq_had_suh_multi, xq_moe_combine",
                    if diet { "diet" } else { "SHFL (--wp20-diet=0)" }));
                (Some(Wp20Sel { conv: c, gu, dn, topk }), false)
            }
            Err((code, msg)) => {
                let rung1 = diet && code != 1 && !matches!(bits, 2 | 5 | 6 | 7); // SHFL-only widths have no diet twin
                wp20_log_once((2, m, code * 16 + bits), || format!(
                    "WP20 FALLBACK m={m}: MoE epilogues (rung 2) NOT active — {msg}. Running the pre-WP20 launch sequence{}",
                    if rung1 { " with rung 1 (diet twin) wherever the a1b3 entry runs" } else { " (no diet twin)" }));
                (None, rung1)
            }
        }
    }

    /// A5 WP20 MoE experts: suh(x) -> gate/up GEMM (own-tile svh, pair epilogue: silu*mul +
    /// down suh -> xhd; expert 0: shared-gate sigmoid) -> shared expert -> down GEMM (svh -> yd,
    /// last arrival per N-tile: combine -> moe_out). Same bytes as the old path (see the WP20
    /// block in kernels/exl3_bench.cu). Every per-round value is a device buffer (esel, idxmap,
    /// ids, wts, slotmap, the self re-arming counters) — capture/replay safe.
    pub(super) fn moe_wp20(&self, l: &Launcher, sc: &mut Scratch, moe: &MoeLayer, m: usize, emax: i32, sel: Wp20Sel,
                wp27: &mut crate::exl3_wp27::Cap) -> Result<()> {
        let h = self.cfg.hidden_size;
        let mi = self.cfg.moe_intermediate_size;
        // W4/MOE: the persistent grouped-expert kernels replace the WP20 bodies side by side
        // (gate/up, down) wherever planned; any side without a plan keeps its WP20 kernel (same
        // buffers, counters and sgv contract, so the two mix freely). Logged once per width.
        // p5e: DEFAULT OFF (opt-in --w4moe=1) — parts == 0 => no plan, no log, fold_live false:
        // xq_had_suh_multi + the WP20 gate/up + the shared expert + the WP20 down, the p3 sequence.
        let parts = w4moe_parts();
        let (gu_pk, dn_pk, fold_live) = self.w4moe_plans(m, emax, sel.topk, moe.bits);
        let il = if w4moe_order_il() { MPK_FLAG_IL } else { 0 };
        if parts & (MPK_GU | MPK_DN) != 0 {
            wp20_log_once((7, m, sel.topk), || {
                let f = |p: &Option<MpkPlan>, want: bool, old: &str| match p {
                    Some(p) => format!("{} G={} ({} CTA/SM, ring {} x 3 KB, smem {} B)", p.name, p.grid, p.cps, p.nst, p.smem),
                    None if want && m < w4moe_minm() => format!("{old} (--w4moe-minm={})", w4moe_minm()),
                    None if want => format!("{old} (W4/MOE FALLBACK: no persistent plan at this width)"),
                    None => format!("{old} (--w4moe-off)"),
                };
                format!("W4/MOE ACTIVE m={m}: gate/up {} | down {} | A-fold {} (xq_had_suh_multi {}), item order {}",
                        f(&gu_pk, parts & MPK_GU != 0, sel.gu), f(&dn_pk, parts & MPK_DN != 0, sel.dn),
                        fold_live as u8, if fold_live { "removed" } else { "kept" },
                        if il != 0 { "interleaved" } else { "contiguous" })
            });
        }
        // A5-K7 (moe.gu_fold): the WP20 gate/up forms its own A (xq_moe_gu_epi_f) — no xq_had_suh_multi.
        let gu_fold = gu_pk.is_none() && moe_gu_fold(l, m);
        if gu_fold {
            wp20_log_once((9, m, 0), || if sel.gu == wp20_gu_fn(true) && xq_raw_fn(WP20_GU_FOLD_FN).is_some() {
                format!("A5-K7 GU_FOLD m={m}: xq_had_suh_multi folded into the gate/up prologue ({WP20_GU_FOLD_FN};                          A = f16(H128(x * suh_e)) in-kernel, pinned suh sequence)")
            } else {
                format!("A5-K7 GU_FOLD FALLBACK m={m}: gate/up entry {} (diet off?) or {WP20_GU_FOLD_FN} missing — unfolded pair", sel.gu)
            });
        }
        let gu_fold = gu_fold && sel.gu == wp20_gu_fn(true) && xq_raw_fn(WP20_GU_FOLD_FN).is_some();
        if !fold_live && !gu_fold {
            xqlaunch!(l, "xq_had_suh_multi", ((emax * m as i32 * (h as i32 / 128)) as u32, 1, 1), (32, 1, 1), 0,
                     (&sc.x, &moe.suh_gu, &sc.idxmap, &mut sc.xh_e, m as i32, h as i32, &sc.esel, 0i32))?;
        }
        let cnt = *sc.moe_cnt.device_ptr() as u64;
        let cnt_dn = cnt + (emax as u64) * (mi as u64 / 128) * 4;
        if let Some(p) = gu_pk {
            let xp = *sc.x.device_ptr() as u64;
            mpk_launch_gu(&self.dev, l.stream.stream, &p, *moe.gu_tr.device_ptr() as u64, *sc.offs_gu.device_ptr() as u64,
                          if fold_live { xp } else { *sc.xh_e.device_ptr() as u64 }, *moe.suh_gu.device_ptr() as u64,
                          *sc.ygu.device_ptr() as u64, m, h, 2 * mi, *sc.esel.device_ptr() as u64,
                          *sc.idxmap.device_ptr() as u64, *moe.svh_gu.device_ptr() as u64, *moe.suh_d.device_ptr() as u64,
                          *sc.xhd.device_ptr() as u64, cnt, xp, *moe.sg.device_ptr() as u64,
                          *sc.moe_sgv.device_ptr() as u64, il | if fold_live { MPK_FLAG_FOLD } else { 0 })?;
        } else {
            // (gu_fold: xh = sc.x, ignored by the kernel; the 17th param is the gate/up suh table)
            let mut a = [*moe.gu_tr.device_ptr() as u64, *sc.offs_gu.device_ptr() as u64,
                         if gu_fold { *sc.x.device_ptr() as u64 } else { *sc.xh_e.device_ptr() as u64 },
                         *sc.ygu.device_ptr() as u64];
            let mut d = [m as i32, h as i32, (2 * mi) as i32];
            let mut b = [*sc.esel.device_ptr() as u64, *sc.idxmap.device_ptr() as u64, *moe.svh_gu.device_ptr() as u64,
                         *moe.suh_d.device_ptr() as u64, *sc.xhd.device_ptr() as u64, cnt, *sc.x.device_ptr() as u64,
                         *moe.sg.device_ptr() as u64, *sc.moe_sgv.device_ptr() as u64, *moe.suh_gu.device_ptr() as u64];
            let mut p: [*mut std::ffi::c_void; 17] = [
                &mut a[0] as *mut u64 as *mut _, &mut a[1] as *mut u64 as *mut _, &mut a[2] as *mut u64 as *mut _,
                &mut a[3] as *mut u64 as *mut _, &mut d[0] as *mut i32 as *mut _, &mut d[1] as *mut i32 as *mut _,
                &mut d[2] as *mut i32 as *mut _, &mut b[0] as *mut u64 as *mut _, &mut b[1] as *mut u64 as *mut _,
                &mut b[2] as *mut u64 as *mut _, &mut b[3] as *mut u64 as *mut _, &mut b[4] as *mut u64 as *mut _,
                &mut b[5] as *mut u64 as *mut _, &mut b[6] as *mut u64 as *mut _, &mut b[7] as *mut u64 as *mut _,
                &mut b[8] as *mut u64 as *mut _, &mut b[9] as *mut u64 as *mut _,
            ];
            let (gfn, np) = if gu_fold { (WP20_GU_FOLD_FN, 17) } else { (sel.gu, 16) };
            let gsm = gu_chunk(m, h, true, moe.bits).filter(|c| c.0 == gfn).map_or(a1_smem(m, h), |c| c.1);
            unsafe { wp20_launch(l.stream.stream, gfn, emax as u32 * (2 * mi as u32 / 128), gsm, &mut p[..np])?; }
        }
        // Shared expert AHEAD of the routed down (its ysh feeds the down's combine epilogue) —
        // the old path's gate|up pair + silu + down (W4/DENSE: two DENSE launches, the silu folded
        // into the down's prologue — opt-in --w4dense=1; default OFF / _SILU=0 = the three old
        // launches, the p3 sequence).
        // W4/MOE lever (c), opt-in --w4moe-shovl=1:
        // while a graph is captured, the shared expert becomes a sibling branch of the routed
        // path (routing .. gate/up) from the MoE fork base (it reads only sc.x) and joins before
        // the down kernel (its combine reads ysh). No spin wait on either branch (last-arrival
        // counters private to the gate/up / down launches; PSK: static striding) => any residency
        // is deadlock-free; same kernels, same inputs => bitwise. Default OFF: the p3c WP27 audit
        // keeps the rung-2 epilogues linear until an in-round A/B shows the overlap pays.
        // A5-K1 (moe.shovl, default ON; escape --moe-shovl=0): the same placement on the DEFAULT rung-2 path (no
        // persistent package needed). Audit (w6/SHOVL): routed branch = router fold (fold_cnt) ->
        // xq_had_suh_multi -> xq_moe_gu_epi (moe_cnt; writes xh_e, ygu, xhd, moe_sgv); shared
        // branch = sh_gate|sh_up pair + silu + sh_down (Quad.cnt of those three Quads; writes
        // chain_*/pair_* scratch, sh1, sh2, ysh; reads sc.x) — write sets disjoint from every
        // sibling read, no counter shared, no spin wait on either branch.
        let shovl = moe_shovl() || (w4moe_shovl() && parts & (MPK_GU | MPK_DN) != 0);
        if shovl && stream_capturing(l) {
            wp20_log_once((8, m, 0), || format!("A5-K1 SHOVL m={m}: shared expert on a WP27 sibling branch of the routed path \
                (router .. {}), joined before {} (captured graphs only; eager = linear)", sel.gu, sel.dn));
        }
        let routed = if shovl { Some(wp27.cut()?) } else { None };
        if shovl { wp27.side_begin()?; }
        exl3_chain_pair(l, &moe.sh_gate, &moe.sh_up, &sc.x,
                        &mut sc.chain_xh, &mut sc.chain_yraw, &sc.chain_ws,
                        &mut sc.pair_xh, &mut sc.pair_yraw, &sc.pair_ws,
                        &mut sc.sh1, &mut sc.sh2, m)?;
        self.shared_down(l, sc, moe, m)?; // W4/DENSE: silu folded into the down (else xq_silu_mul + chain)
        if let Some(t) = routed {
            wp27.side_end()?;
            wp27.join(&t)?; // W4/MOE (c): the down launch depends on the routed tail AND the shared tail
        }
        if let Some(p) = dn_pk {
            mpk_launch_dn(&self.dev, l.stream.stream, &p, *moe.d_tr.device_ptr() as u64, *sc.offs_d.device_ptr() as u64,
                          *sc.xhd.device_ptr() as u64, *sc.yd.device_ptr() as u64, m, mi, h,
                          *sc.esel.device_ptr() as u64, *sc.idxmap.device_ptr() as u64, *moe.svh_d.device_ptr() as u64,
                          cnt_dn, *sc.ysh.device_ptr() as u64, *sc.ids.device_ptr() as u64, *sc.wts.device_ptr() as u64,
                          *sc.slotmap.device_ptr() as u64, *sc.moe_sgv.device_ptr() as u64,
                          *sc.moe_out.device_ptr() as u64, sel.topk, il)?;
        } else {
            let mut a = [*moe.d_tr.device_ptr() as u64, *sc.offs_d.device_ptr() as u64, *sc.xhd.device_ptr() as u64,
                         *sc.yd.device_ptr() as u64];
            let mut d = [m as i32, mi as i32, h as i32];
            let mut b = [*sc.esel.device_ptr() as u64, *sc.idxmap.device_ptr() as u64, *moe.svh_d.device_ptr() as u64,
                         cnt_dn, *sc.ysh.device_ptr() as u64, *sc.ids.device_ptr() as u64, *sc.wts.device_ptr() as u64,
                         *sc.slotmap.device_ptr() as u64, *sc.moe_sgv.device_ptr() as u64, *sc.moe_out.device_ptr() as u64];
            let mut k = sel.topk as i32;                // the instance's TOPK (the kernel traps on a mismatch)
            let mut p: [*mut std::ffi::c_void; 18] = [
                &mut a[0] as *mut u64 as *mut _, &mut a[1] as *mut u64 as *mut _, &mut a[2] as *mut u64 as *mut _,
                &mut a[3] as *mut u64 as *mut _, &mut d[0] as *mut i32 as *mut _, &mut d[1] as *mut i32 as *mut _,
                &mut d[2] as *mut i32 as *mut _, &mut b[0] as *mut u64 as *mut _, &mut b[1] as *mut u64 as *mut _,
                &mut b[2] as *mut u64 as *mut _, &mut b[3] as *mut u64 as *mut _, &mut b[4] as *mut u64 as *mut _,
                &mut b[5] as *mut u64 as *mut _, &mut b[6] as *mut u64 as *mut _, &mut b[7] as *mut u64 as *mut _,
                &mut b[8] as *mut u64 as *mut _, &mut b[9] as *mut u64 as *mut _, &mut k as *mut i32 as *mut _,
            ];
            unsafe { wp20_launch(l.stream.stream, sel.dn, emax as u32 * (h as u32 / 128), a1_smem(m, mi), &mut p)?; }
        }
        Ok(())
    }

    /// W4/MOE plans for one MoE call at width m: (gate/up, down, A-fold live). A side without a
    /// plan (--w4moe-off part, or not launchable) keeps its WP20 rung-2 kernel.
    fn w4moe_plans(&self, m: usize, emax: i32, topk: usize, bits: usize) -> (Option<MpkPlan>, Option<MpkPlan>, bool) {
        // B10: the persistent grouped-expert kernels now cover bits 2..=7 (K=3 keeps the diet
        // entries; 2,4,5,6,7 run the width-generic SHFL bodies). Outside that range the side
        // keeps its WP20 rung-2 kernel (as before).
        if !(2..=7).contains(&bits) { return (None, None, false); }
        if bits != 3 { crate::metrics::exl3_w4moe_width_served(); }  // 1b-bis.2: the width receipt
        let (h, mi) = (self.cfg.hidden_size, self.cfg.moe_intermediate_size);
        let parts = w4moe_parts();
        let fold = parts & MPK_FOLD != 0;
        let on = m >= w4moe_minm();
        let gu = if on && parts & MPK_GU != 0 { mpk_plan(&self.dev, true, topk, m, h, 2 * mi, emax as usize, fold, bits) } else { None };
        let dn = if on && parts & MPK_DN != 0 { mpk_plan(&self.dev, false, topk, m, mi, h, emax as usize, false, bits) } else { None };
        let fold_live = fold && gu.is_some();
        (gu, dn, fold_live)
    }

    /// --exl3-moe-epi-check: the old path has just run (eager); snapshot its xhd / yd /
    /// moe_out, run the WP20 path over the same routing, diff bitwise, and check every
    /// counter re-armed to 0. Prints the first calls and any mismatch; a mismatch fails the
    /// request (the diagnostic must never be silent).
    pub(super) fn moe_wp20_xcheck(&self, l: &Launcher, sc: &mut Scratch, moe: &MoeLayer, m: usize, emax: i32, sel: Wp20Sel) -> Result<()> {
        let (h, mi) = (self.cfg.hidden_size, self.cfg.moe_intermediate_size);
        let conv = sel.conv;
        self.dsync()?;
        let es = (self.dtoh(&sc.esel)?[0].max(0) as usize).min(emax as usize);
        let (nx, ny, no) = (es * m * mi, es * m * h, m * h);
        let ng = es * m * 2 * mi;
        let x0 = self.dtoh(&sc.xhd)?;
        let y0 = self.dtoh(&sc.yd)?;
        let o0 = self.dtoh(&sc.moe_out)?;
        let g0 = self.dtoh(&sc.ygu)?;
        // W4/MOE: NaN-poison every output the new path must write (xhd, yd, moe_out, ygu, the
        // shared-gate sigmoids) — a skipped write can never compare equal on the old bytes.
        for buf in [&sc.xhd, &sc.yd, &sc.moe_out, &sc.ygu] {
            let n = buf.len();
            xqlaunch!(l, "xq_memset_u16", (((n + 255) / 256) as u32, 1, 1), (256, 1, 1), 0, (buf, n as i64, 0x7E00i32))?;
        }
        let nsg = sc.moe_sgv.len();
        self.dev.htod_sync_copy_into(&vec![f32::NAN; nsg], &mut sc.moe_sgv)?;
        self.dsync()?;
        let (gu_pk, dn_pk, fold_live) = self.w4moe_plans(m, emax, sel.topk, moe.bits);
        self.moe_wp20(l, sc, moe, m, emax, sel, &mut crate::exl3_wp27::Cap::none())?;
        self.dsync()?;
        let x1 = self.dtoh(&sc.xhd)?;
        let y1 = self.dtoh(&sc.yd)?;
        let o1 = self.dtoh(&sc.moe_out)?;
        let g1 = self.dtoh(&sc.ygu)?;
        let cnt = self.dtoh(&sc.moe_cnt)?;
        let diff = |a: &[u16], b: &[u16]| -> (usize, Option<(usize, u16, u16)>) {
            let mut n = 0;
            let mut first = None;
            for (i, (p, q)) in a.iter().zip(b.iter()).enumerate() {
                if p != q { n += 1; if first.is_none() { first = Some((i, *p, *q)); } }
            }
            (n, first)
        };
        let (dx, fx) = diff(&x0[..nx], &x1[..nx]);
        let (dy, fy) = diff(&y0[..ny], &y1[..ny]);
        let (dn, fo) = diff(&o0[..no], &o1[..no]);
        let (dg, fg) = diff(&g0[..ng], &g1[..ng]);
        let armed = cnt.iter().all(|&c| c == 0);
        let bad = dx + dy + dn + dg > 0 || !armed || es == 0;
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let c = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if c < 4 || bad {
            let first = [("ygu", fg), ("xhd", fx), ("yd", fy), ("moe_out", fo)].iter()
                .filter_map(|(n, f)| f.map(|(i, p, q)| format!(" {n}[{i}] old {p:04x} new {q:04x}")))
                .collect::<String>();
            let gu = gu_pk.map_or(sel.gu, |p| p.name);
            let dnn = dn_pk.map_or(sel.dn, |p| p.name);
            let g = |p: &Option<MpkPlan>| p.map_or(String::new(), |p| format!(" G{}", p.grid));
            let gu = if gu_pk.is_none() && moe_gu_fold(l, m) && sel.gu == wp20_gu_fn(true) {
                WP20_GU_FOLD_FN } else { gu };
            crate::dprintln!("MOE_EPI_CHECK m={m} esel={es} topk={} {gu}{}+{dnn}{} fold {} suh seq 0x{conv:02x}: bit-diff ygu {dg}/{ng}, \
                      xhd {dx}/{nx}, yd {dy}/{ny}, moe_out {dn}/{no} (outputs NaN-poisoned first); counters {}{}",
                     sel.topk, g(&gu_pk), g(&dn_pk), fold_live as u8, if armed { "re-armed" } else { "NOT re-armed" },
                     if first.is_empty() { String::new() } else { format!(" (first mismatches:{first})") });
        }
        if bad { anyhow::bail!("WP20 / W4-MOE MoE path diverges from the old expert path"); }
        Ok(())
    }

    /// One round: m lanes. Returns greedy ids per lane. States mutate in place
    /// (conv/s/KV via raw device pointers — forward_step takes &self).
    /// S-A3-i: visible-token count up to which the indexer selects EVERY block (dense
    /// attention is exact there); past it the QSA path is the model spec.
    /// S-A3-o diagnostic: print the cumulative --exl3-esel-hist table (no-op when off). Called at
    /// every request finish as well as at drop, so a server stopped by SIGTERM (no drop) still
    /// leaves the table in its log — the last block printed is the run's total.
    pub fn dump_esel_hist(&self) {
        let Some(h) = self.esel_hist.as_ref() else { return };
        match self.dtoh(h) {
            Ok(v) => {
                for m in 1..=16usize {
                    let row = &v[m * ESEL_HIST_W..(m + 1) * ESEL_HIST_W];
                    let n: i64 = row.iter().map(|&c| c as i64).sum();
                    if n == 0 { continue; }
                    let mean = row.iter().enumerate().map(|(e, &c)| e as f64 * c as f64).sum::<f64>() / n as f64;
                    let nz: Vec<String> = row.iter().enumerate().filter(|(_, &c)| c > 0)
                        .map(|(e, &c)| format!("{e}:{c}")).collect();
                    crate::dprintln!("ESEL_HIST m={m} calls={n} mean={mean:.2} [{}]", nz.join(" "));
                }
            }
            Err(e) => crate::deprintln!("[exl3] --exl3-esel-hist dtoh: {e}"),
        }
    }

    /// CF-P1e step 0 diagnostic: write this request's --exl3-route-log records to DIR/route_NNNN.bin and reset the ring.
    /// File: i32 LE [0x524C4F47, n_records, ROUTE_REC, topk] then n_records x ROUTE_REC ints
    /// ([layer, m, ids[0..m*topk], -1 padding]).
    pub fn dump_route_log(&self) {
        let Some((buf, ctr, dir, seq)) = self.route_log.as_ref() else { return };
        let res = (|| -> Result<()> {
            let n = (self.dtoh(ctr)?[0] as usize).min(ROUTE_LOG_CAP);
            let v = self.dtoh(buf)?;
            let mut out: Vec<u8> = Vec::with_capacity(16 + n * ROUTE_REC * 4);
            for x in [0x524C4F47i32, n as i32, ROUTE_REC as i32, self.cfg.num_experts_per_tok as i32] { out.extend(x.to_le_bytes()); }
            for x in &v[..n * ROUTE_REC] { out.extend(x.to_le_bytes()); }
            let k = seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            std::fs::write(format!("{dir}/route_{k:04}.bin"), out)?;
            self.dev.bind_to_thread()?;
            let rc = unsafe { cudarc::driver::sys::cuMemsetD32_v2(*ctr.device_ptr(), 0, 1) };
            anyhow::ensure!(rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "route-log counter reset ({rc:?})");
            Ok(())
        })();
        if let Err(e) = res { crate::deprintln!("[exl3] --exl3-route-log dump: {e}"); }
    }

    /// TP-I #6 diagnostic: write the cumulative --tp-ep-hist routing histogram (no-op when off).
    pub fn dump_expert_hist(&self) {
        let Some((hb, path)) = self.expert_hist.as_ref() else { return };
        let (nl, ne) = (self.cfg.num_layers, self.cfg.num_experts);
        match self.dtoh(hb) {
            Ok(v) => {
                let live: Vec<&[i32]> = (0..nl).map(|l| &v[l * 2 * ne..l * 2 * ne + ne]).collect();
                let picks: Vec<&[i32]> = (0..nl).map(|l| &v[l * 2 * ne + ne..(l + 1) * 2 * ne]).collect();
                let j = serde_json::json!({"ne": ne, "layers": nl, "topk": self.cfg.num_experts_per_tok,
                                           "live": live, "picks": picks});
                if let Err(e) = std::fs::write(path, j.to_string()) { crate::deprintln!("[exl3] --tp-ep-hist write {path}: {e}"); }
            }
            Err(e) => crate::deprintln!("[exl3] --tp-ep-hist dtoh: {e}"),
        }
    }

}
