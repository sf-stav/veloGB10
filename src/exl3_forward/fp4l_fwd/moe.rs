//! FP4L NVFP4 routed-expert dispatch: decode (m <= 16) and prefill (tile lists) for `FxMoe`. The router, top-k and combine are the trunk's own.
//! See `mod.rs` for the file map.

use super::*;

// ------------------------------------------------------------------------------------------------
// MoE dispatch
// ------------------------------------------------------------------------------------------------

pub(super) fn nv4_moe_fn() -> &'static str { if e2m1_int() { "xf_nv4_moe_i" } else { "xf_nv4_moe" } }

/// The tile-list classes the smem-tiled xf_nv4_tc serves (mt 4 and 8: 64-row sub-blocks); bit-identical to
/// xf_nv4_tiles_m4 there. Classes 1 and 2 keep xf_nv4_tiles_m1 / m2 (a 64-row CTA would idle most of its rows).
pub(super) fn nv4_tc_fn(mt: usize) -> Option<&'static str> {
    if mt >= 4 && mt % 4 == 0 { Some(if e2m1_int() { "xf_nv4_tc_i" } else { "xf_nv4_tc" }) } else { None }
}

pub(super) fn nv4_tiles_fn(mt: usize) -> (&'static str, usize) {
    let int = e2m1_int();
    // the kernel's register tile: MT 1, 2 or 4 token tiles; a list built for mt = 8 is walked in
    // two MT-4 halves (the kernel loops over sub-blocks of the list's 16*mt rows).
    match (mt.min(4), int) {
        (1, false) => ("xf_nv4_tiles_m1", 1), (2, false) => ("xf_nv4_tiles_m2", 2), (_, false) => ("xf_nv4_tiles_m4", 4),
        (1, true) => ("xf_nv4_tiles_m1_i", 1), (2, true) => ("xf_nv4_tiles_m2_i", 2), (_, true) => ("xf_nv4_tiles_m4_i", 4),
    }
}

/// Decode / verify (m <= 16) routed experts after the trunk's routing: gate|up -> sc.ygu
/// ([slot][m][2*mi]), silu(g)*u -> sc.din (xq_moe_gate_mul), down -> sc.yd ([slot][m][h]); the
/// trunk's shared expert and combine follow unchanged.
pub(crate) fn moe_experts_decode(l: &Launcher, sc: &mut Scratch, fx: &FxMoe, h: usize, mi: usize, m: usize, emax: i32) -> Result<()> {
    anyhow::ensure!((1..=16).contains(&m) && (2 * mi) % 64 == 0 && h % 64 == 0 && h % 64 == 0 && mi % 64 == 0,
                    "FP4L MoE decode: m {m}, h {h}, mi {mi} outside the xf_nv4_moe contract");
    crate::metrics::fp4l_moe(m);
    let f = nv4_moe_fn();
    xflaunch!(l, f, (((2 * mi) / 64) as u32, emax as u32, 1), (128, 1, 1), 0,
              (&fx.gu_w, &fx.gu_s, &fx.gs, &sc.x, 0i64, &mut sc.ygu, &sc.idxmap, &sc.esel,
               m as i32, h as i32, (2 * mi) as i32, mi as i32))?;
    xqlaunch!(l, "xq_moe_gate_mul", (((emax as usize * m * mi) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0,
              (&mut sc.din, &sc.ygu, (m * mi) as i64, &sc.esel))?;
    xflaunch!(l, f, ((h / 64) as u32, emax as u32, 1), (128, 1, 1), 0,
              (&fx.dn_w, &fx.dn_s, &fx.gs, &sc.din, (m * mi) as i64, &mut sc.yd, &sc.idxmap, &sc.esel,
               m as i32, mi as i32, h as i32, h as i32))
}

/// The prefill MoE layer of an FP4-Local pack, FP4L-owned: a copy of the device-routed arm of the trunk's
/// `FwdModel::moe_prefill` (base d4615ea9) with the NVFP4 expert tiles in place of the trellis kernels, so the trunk
/// function carries ONE dispatch line and its later changes neither reach this path nor conflict with it. The
/// launches are the ones the trunk function made for this pack before the split (output bit-identical): router
/// (the stored-bf16 twin by default, the trunk's f16 kernels under --fp4l-plain f16; the own rows + one all-gather
/// under --tp-seq-parallel router), device routing (count / place / EP fill; the per-expert MT classes when
/// prefill.moe_mt_pe is on), the expert tiles, then the trunk's tail (shared expert + combine). The host-routed
/// trellis arm (--exl3-moe-recon-min / --exl3-moe-devroute=0) cannot serve NVFP4 experts: those EXL3 levers are
/// not applied here (one line at the first prefill that asks for them).
pub(crate) fn moe_prefill(fm: &FwdModel, l: &Launcher, psc: &mut PrefillScratch, moe: &MoeLayer, c: usize,
                          rb: &ReconBufs) -> Result<()> {
    let fx = moe.fx.as_ref().context("FP4L prefill MoE: a stored-bf16 router on a layer without NVFP4 experts")?;
    let (h, ne, topk, mi) = (fm.cfg.hidden_size, fm.cfg.num_experts, fm.cfg.num_experts_per_tok, fm.cfg.moe_intermediate_size);
    let mt: usize = match tune::get(&T_WIDE_MT, &l.cx()) { v @ (1 | 2 | 4 | 8) => v as usize, _ => pfx1_moe_mt(c) };
    let rows_on = tune::get(&T_PF_ROUTER_ROWS, &l.cx()) != 0;
    let (lp, xp) = (*psc.logits_r.device_ptr() as u64, *psc.x.device_ptr() as u64);
    let (ip, wp) = (*psc.ids.device_ptr() as u64, *psc.wts.device_ptr() as u64);
    // router logits + softmax/top-k of rows [lo, lo + n) (every row depends only on its own x row)
    let route = |lo: usize, n: usize| -> Result<()> {
        let (lpo, xpo) = (lp + (lo * ne * 4) as u64, xp + (lo * h * 2) as u64);
        let (ipo, wpo) = (ip + (lo * topk * 4) as u64, wp + (lo * topk * 4) as u64);
        if moe.router_bf {
            crate::metrics::fp4l_twin("router");
            bf_wide(l, &moe.router, xpo, 0, lpo, n, h, ne)?;
            router_topk(l, ipo, wpo, lpo, ne, topk, n)
        } else {
            router_rows_launch(l, lpo, *moe.router.device_ptr() as u64, xpo, n, c, ne, h, rows_on)?;
            xqlaunch!(l, "xq_router_topk", (n as u32, 1, 1), (256, 1, 1), ((256 + ne + 512) * 4) as u32,
                      (ipo, wpo, lpo, ne as i32, topk as i32, n as i32))
        }
    };
    let sp_router = psc.sp_rows.filter(|_| xtp::seq_parallel_router() && (topk * 4) % 16 == 0);
    if let (Some(s), Some(tp)) = (sp_router, fm.tp.as_ref()) {
        anyhow::ensure!(s.c == c, "TP-SP router: chunk {} vs MoE c {c}", s.c);
        route(s.own.0, s.own.1 - s.own.0)?;
        tp.sp_all_gather_multi(l, &[(ip, topk * 4), (wp, topk * 4)], &s)?;
    } else {
        route(0, c)?;
    }
    let host_routing = crate::opts::var(crate::opt!("exl3-moe-recon-min")).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0) > 0
        || crate::opts::var(crate::opt!("exl3-moe-devroute")).as_deref() == Ok("0");
    if host_routing {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            crate::rprintln!("[fp4l] --exl3-moe-recon-min / --exl3-moe-devroute=0 select the EXL3 host-routed trellis arm, which has no \
NVFP4 form: the FP4-Local experts keep the device-routed tile lists");
        }
    }
    anyhow::ensure!(psc.pf_tiles.len() > 1, "FP4L prefill MoE: the prefill scratch has no expert tile lists");
    let ep = moe.ep.as_ref();
    let r = c * topk;
    // TP-B (expert parallel): count/place over THIS rank's local expert space (ids_ep: local stacked id or -1)
    let ne_eff = ep.map_or(ne, |e| e.n_local);
    if let Some(e) = ep {
        xqlaunch!(l, "xq_moe_ids_ep", (r.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0,
                  (&psc.ids, &mut psc.ids_ep, &mut psc.cand, &e.lut, r as i32, ne as i32))?;
    }
    let ids_route = if ep.is_some() { *psc.ids_ep.device_ptr() as u64 } else { ip };
    let ne = ne_eff;
    let max_tiles = (r / 16 + ne) as u32;
    // ONE launch group at the chunk-rule MT (the launches FP4L made before its tile lists moved to the pf_same line;
    // the trunk's per-expert MT classes (prefill.moe_mt_per_expert) measured 2-4 % slower TTFT on the NVFP4 tiles)
    let pe = false;
    // (mt, tiles ptr, ntiles ptr, grid-y cap) per launch group
    let tl_base = *psc.pf_tiles4.device_ptr() as u64;
    let nt_base = *psc.pf_ntiles4.device_ptr() as u64;
    let groups: Vec<(usize, u64, u64, u32)> = if pe {
        let caps = moe_pf_class_caps(r, ne);
        (0..4).filter(|&k| caps[k] > 0)
            .map(|k| (1usize << k, tl_base + (k * max_tiles as usize * 2 * 4) as u64, nt_base + (k * 4) as u64, caps[k] as u32))
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
    moe_experts_prefill(l, psc, fx, h, mi, r, &groups)?;
    fm.moe_prefill_tail(l, psc, moe, c, rb)
}

/// Prefill routed experts over the trunk's device tile lists (xq_moe_pf_count/place already ran):
/// gate|up tiles read x through row_tok -> psc.ygu ([r][2*mi], compact rows), gate_mul -> psc.din,
/// down tiles -> psc.yd ([r][h]); moe_prefill_tail combines.
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_experts_prefill(l: &Launcher, psc: &mut PrefillScratch, fx: &FxMoe, h: usize, mi: usize, r: usize,
                                  groups: &[(usize, u64, u64, u32)]) -> Result<()> {
    anyhow::ensure!(h < (1 << 15) && 2 * mi < (1 << 15) && (2 * mi) % 128 == 0 && h % 128 == 0,
                    "FP4L prefill MoE: h {h} / mi {mi} outside the xf_nv4_tiles contract");
    crate::metrics::fp4l_moe(17);
    let kn = |k: usize, n: usize| (k | (n << 16)) as i32;
    for &(gmt, tp, np, cap) in groups {
        let (f, kmt) = nv4_tiles_fn(gmt);
        anyhow::ensure!(gmt % kmt == 0, "FP4L prefill MoE: tile list mt {gmt} not a multiple of the kernel MT {kmt}");
        let f = nv4_tc_fn(gmt).unwrap_or(f);
        xflaunch!(l, f, (((2 * mi) / 128) as u32, cap, 1), (256, 1, 1), 0,
                  (&fx.gu_w, &fx.gu_s, &fx.gs, &psc.x, &psc.row_tok, &mut psc.ygu, tp, np,
                   &psc.offs_row, &psc.cnt, kn(h, 2 * mi), (mi | (gmt << 16)) as i32))?;
    }
    xqlaunch!(l, "xq_moe_gate_mul", (((r * mi) as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
              (&mut psc.din, &psc.ygu, mi as i64, &psc.moerows))?;
    for &(gmt, tp, np, cap) in groups {
        let (f, _) = nv4_tiles_fn(gmt);
        let f = nv4_tc_fn(gmt).unwrap_or(f);
        xflaunch!(l, f, ((h / 128) as u32, cap, 1), (256, 1, 1), 0,
                  (&fx.dn_w, &fx.dn_s, &fx.gs, &psc.din, 0u64, &mut psc.yd, tp, np,
                   &psc.offs_row, &psc.cnt, kn(mi, h), (h | (gmt << 16)) as i32))?;
    }
    Ok(())
}
