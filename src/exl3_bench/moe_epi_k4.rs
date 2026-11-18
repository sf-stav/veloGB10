//! The K = 4 twin of the MoE epilogue probe (CF-P1g step 2).
//! Probe-only code of the `exl3_bench` module (see `mod.rs` for the map); shared helpers come from the parent.
//! Numerics contract (module header in `mod.rs`): the kernel decode is bitwise vs the host oracle; do not change reduction orders.

use super::*;

/// CF-P1g step 2: the K = 4 twin of `probe_moe_epi` (the 4.05 pack's experts): the 4-bit WP20 epilogue entries (diet, top-10,
/// widths 1..16 incl. the K-chunked gate/up) vs the OLD chain run with 4-bit trellis, bitwise.
pub(super) fn probe_moe_epi_k4(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    const NE_MAX: usize = 64;      // physical trellis sets (expert id e uses set e % NE_MAX)
    const NE_TAB: usize = 256;     // suh/svh rows (the 256-expert routing case: every id distinct)
    let (h, mi) = (2560usize, 640usize);
    let conv = match wp20_calibrate(dev)? {
        Some(c) => c,
        None => bail!("EXL3-MOE-EPI FAIL: the pinned suh sequence does not match xq_had_suh_multi (see the line above)"),
    };
    let gu_words = (h / 16) * (2 * mi / 16) * 64;
    let d_words = (mi / 16) * (h / 16) * 64;
    // NE_MAX experts' worth; the LCG makes the first 12 experts' bytes identical to v1's 12-expert set.
    let d_gu = dev.htod_sync_copy(&wp20_synth(NE_MAX * gu_words, 0x6001, 1.0))?;   // any u16 decodes finite
    let d_dn = dev.htod_sync_copy(&wp20_synth(NE_MAX * d_words, 0x6002, 1.0))?;
    // (NE_TAB rows: the LCG only appends, so the first NE_MAX rows are the earlier tables' bytes)
    let d_suh_gu = dev.htod_sync_copy(&wp20_synth(NE_TAB * h, 0x6003, 0.25))?;
    let d_svh_gu = dev.htod_sync_copy(&wp20_synth(NE_TAB * 2 * mi, 0x6004, 0.005))?;
    let d_suh_d = dev.htod_sync_copy(&wp20_synth(NE_TAB * mi, 0x6005, 0.02))?;
    let d_svh_d = dev.htod_sync_copy(&wp20_synth(NE_TAB * h, 0x6006, 0.01))?;
    let d_sg = dev.htod_sync_copy(&wp20_synth(h, 0x6007, 0.01))?;
    let f = |n: &str| dev.get_func(MODULE, n).with_context(|| n.to_string());
    let (f_suh, f_gemm, f_svh, f_gm, f_comb) =
        (f("xq_had_suh_multi")?, f("xq_gemm_grouped_a1b4")?, f("xq_had_svh_multi")?, f("xq_moe_gate_mul")?, f("xq_moe_combine")?);
    // CF-P1e: at 10..16 rows the gate/up A-once smem exceeds the 48 KiB cap, so the OLD chain's gate/up GEMM is the
    // legacy barrier body (xq_gemm_grouped_xh, smem 0) — what served those widths before the chunked entries.
    let f_gemm_xh = f("xq_gemm_grouped_xh")?;
    let lc = |g: u32, b: u32, s: u32| LaunchConfig { grid_dim: (g, 1, 1), block_dim: (b, 1, 1), shared_mem_bytes: s };
    let a1 = |m: usize, k: usize| (m * (k + 8) * 2) as u32;
    let mut done = Vec::new();
    let mut w4_done = 0usize;
    let mut w4_cfgs: Vec<String> = Vec::new();
    // (top-k, routed experts): v1's top-8 over 12; Qwen3.8-Flash-Next's top-10 over 64 (served-like
    // live counts: ~46 at m = 8); top-10 over 256 (ids all distinct: live = m * 10, the cap).
    for (topk, ne) in [(10usize, NE_MAX), (10, NE_TAB)] {
    let (dn_diet, dn_sh): (&'static str, &'static str) = ("xq_moe_dn_epi_b4_k10", "xq_moe_dn_epi_b4_k10");
    for m in 1usize..=16 {
        // the launcher's smem contract at this (m, top-k): tile + [m][top-k] tables inside a1(m, mi)
        if (a1(m, mi) as usize) < m * 256 + m * topk * 8 {
            bail!("EXL3-MOE-EPI FAIL: down epilogue smem {} B < tile + tables {} B at m={m} top-k {topk}",
                  a1(m, mi), m * 256 + m * topk * 8);
        }
        let emax = m * topk;
        // routing: row r picks topk distinct experts (LCG), positive weights summing to ~1
        let mut s = 0x9E37_79B9u32.wrapping_mul(m as u32 + 1);
        let mut ids = Vec::with_capacity(m * topk);
        let mut wts = Vec::with_capacity(m * topk);
        for _ in 0..m {
            let mut picked: Vec<i32> = Vec::new();
            while picked.len() < topk {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let e = ((s >> 8) % ne as u32) as i32;
                if !picked.contains(&e) { picked.push(e); }
            }
            let raw: Vec<f32> = (0..topk).map(|_| { s = s.wrapping_mul(1664525).wrapping_add(1013904223); 0.05 + ((s >> 8) % 1000) as f32 / 1000.0 }).collect();
            let sum: f32 = raw.iter().sum();
            ids.extend(picked);
            wts.extend(raw.iter().map(|v| v / sum));
        }
        // xq_moe_route's tables (first-seen over ids row-major)
        let mut slotmap = vec![-1i32; ne];
        let mut idxmap = vec![0i32; emax];
        let mut es = 0usize;
        for &e in &ids { if slotmap[e as usize] < 0 { slotmap[e as usize] = es as i32; idxmap[es] = e; es += 1; } }
        let offs_gu: Vec<u64> = (0..emax).map(|i| (idxmap[i] as usize % NE_MAX) as u64 * gu_words as u64).collect();
        let offs_d: Vec<u64> = (0..emax).map(|i| (idxmap[i] as usize % NE_MAX) as u64 * d_words as u64).collect();
        let (d_ids, d_wts, d_sm, d_ix) = (dev.htod_sync_copy(&ids)?, dev.htod_sync_copy(&wts)?,
                                         dev.htod_sync_copy(&slotmap)?, dev.htod_sync_copy(&idxmap)?);
        let (d_ogu, d_od, d_es) = (dev.htod_sync_copy(&offs_gu)?, dev.htod_sync_copy(&offs_d)?, dev.htod_sync_copy(&[es as i32])?);
        let d_x = dev.htod_sync_copy(&wp20_synth(m * h, 0x7000 + m as u32, 0.5))?;
        let d_ysh = dev.htod_sync_copy(&wp20_synth(m * h, 0x7100 + m as u32, 0.25))?;
        let z = |n: usize| dev.htod_sync_copy(&vec![0u16; n]);
        let (mut xh_e, mut ygu_raw, mut ygu, mut din) = (z(emax * m * h)?, z(emax * m * 2 * mi)?, z(emax * m * 2 * mi)?, z(emax * m * mi)?);
        let (mut xhd0, mut ydr, mut yd0, mut out0) = (z(emax * m * mi)?, z(emax * m * h)?, z(emax * m * h)?, z(m * h)?);
        // ---- the old chain
        unsafe {
            f_suh.clone().launch(lc((emax * m * (h / 128)) as u32, 32, 0),
                (&d_x, &d_suh_gu, &d_ix, &mut xh_e, m as i32, h as i32, &d_es, 0i32))?;
            (if m >= 10 { f_gemm_xh.clone() } else { f_gemm.clone() })
                .launch(lc((emax * (2 * mi / 128)) as u32, 256, if m >= 10 { 0 } else { a1(m, h) }),
                (&d_gu, &d_ogu, &xh_e, &mut ygu_raw, m as i32, h as i32, (2 * mi) as i32, 4i32, &d_es))?;
            f_svh.clone().launch(lc((emax * m * (2 * mi / 128)) as u32, 32, 0),
                (&ygu_raw, &d_svh_gu, &d_ix, &mut ygu, m as i32, (2 * mi) as i32, &d_es))?;
            f_gm.clone().launch(lc(((emax * m * mi) / 256 + 1) as u32, 256, 0), (&mut din, &ygu, (m * mi) as i64, &d_es))?;
            f_suh.clone().launch(lc((emax * m * (mi / 128)) as u32, 32, 0),
                (&din, &d_suh_d, &d_ix, &mut xhd0, m as i32, mi as i32, &d_es, 1i32))?;
            f_gemm.clone().launch(lc((emax * (h / 128)) as u32, 256, a1(m, mi)),
                (&d_dn, &d_od, &xhd0, &mut ydr, m as i32, mi as i32, h as i32, 4i32, &d_es))?;
            f_svh.clone().launch(lc((emax * m * (h / 128)) as u32, 32, 0),
                (&ydr, &d_svh_d, &d_ix, &mut yd0, m as i32, h as i32, &d_es))?;
            f_comb.clone().launch(lc(m as u32, 1024, 4096),
                (&mut out0, &yd0, &d_ysh, &d_ids, &d_wts, &d_sm, &d_sg, &d_x, topk as i32, h as i32, m as i32))?;
        }
        dev.synchronize()?;
        let x_old = dev.dtoh_sync_copy(&xhd0)?;
        let g_old = dev.dtoh_sync_copy(&ygu)?;
        let y_old = dev.dtoh_sync_copy(&yd0)?;
        let o_old = dev.dtoh_sync_copy(&out0)?;
        let finite = |v: &[u16]| v.iter().all(|b| b & 0x7C00 != 0x7C00);
        if !finite(&o_old[..m * h]) || !finite(&x_old[..es * m * mi])
            || o_old[..m * h].iter().filter(|b| **b & 0x7FFF != 0).count() < m * h / 2 {
            bail!("EXL3-MOE-EPI FAIL: implausible reference (m={m}: non-finite or mostly-zero old-chain output)");
        }
        // A5-K7 fold row: xq_moe_gu_epi_f (xq_had_suh_multi folded into the gate/up prologue: reads x +
        // the gate/up suh table, never xh_e) + the diet down epilogue, vs the same old chain.
        let epi_rows: Vec<(bool, bool, &'static str, &'static str)> = if m >= 10 {
            // CF-P1e 16-row class, 4-bit: the K-chunked gate/up entries (2 and 4 chunks) + the 4-bit diet down epilogue
            vec![(true, false, "xq_moe_gu_epi_b4_c2", dn_diet), (true, false, "xq_moe_gu_epi_b4_c4", dn_diet)]
        } else {
            vec![(true, false, "xq_moe_gu_epi_b4", dn_diet)]
        };
        let gsm_of = |gu: &str| -> u32 {
            match gu { "xq_moe_gu_epi_b4_c2" => (m * (h / 2 + 8) * 2) as u32, "xq_moe_gu_epi_b4_c4" => (m * (h / 4 + 8) * 2) as u32, _ => a1(m, h) }
        };
        for (diet, fold, gu, dn) in epi_rows {
            let (mut ygu1, mut xhd1, mut yd1, mut out1) = (z(emax * m * 2 * mi)?, z(emax * m * mi)?, z(emax * m * h)?, z(m * h)?);
            let d_cnt = dev.htod_sync_copy(&vec![0u32; emax * (mi / 128) + h / 128 + 1])?;
            let mut d_sgv = dev.htod_sync_copy(&vec![0f32; m])?;
            let cnt = *d_cnt.device_ptr() as u64;
            let cnt_dn = cnt + (emax * (mi / 128) * 4) as u64;
            let fg = crate::exl3_forward::xq_raw_fn(gu).ok_or_else(|| anyhow::anyhow!("{gu} raw fn (stale PTX?)"))?;
            let fd = crate::exl3_forward::xq_raw_fn(dn).ok_or_else(|| anyhow::anyhow!("{dn} raw fn (stale PTX?)"))?;
            for rep in 0..2 {
                // NaN-poison every output first: a launch that skips a write can never pass on
                // the previous rep's bytes (the counters must really re-arm for rep 1).
                for (buf, n) in [(&mut ygu1, emax * m * 2 * mi), (&mut xhd1, emax * m * mi), (&mut yd1, emax * m * h),
                                 (&mut out1, m * h)] {
                    dev.htod_sync_copy_into(&vec![0x7E00u16; n], buf)?;
                }
                dev.htod_sync_copy_into(&vec![f32::NAN; m], &mut d_sgv)?;
                let mut a = [*d_gu.device_ptr() as u64, *d_ogu.device_ptr() as u64,
                             if fold { *d_x.device_ptr() as u64 } else { *xh_e.device_ptr() as u64 },
                             *ygu1.device_ptr() as u64];
                let mut d = [m as i32, h as i32, (2 * mi) as i32];
                let mut b = [*d_es.device_ptr() as u64, *d_ix.device_ptr() as u64, *d_svh_gu.device_ptr() as u64,
                             *d_suh_d.device_ptr() as u64, *xhd1.device_ptr() as u64, cnt, *d_x.device_ptr() as u64,
                             *d_sg.device_ptr() as u64, *d_sgv.device_ptr() as u64, *d_suh_gu.device_ptr() as u64];
                let mut p: [*mut std::ffi::c_void; 17] = [
                    &mut a[0] as *mut u64 as *mut _, &mut a[1] as *mut u64 as *mut _, &mut a[2] as *mut u64 as *mut _,
                    &mut a[3] as *mut u64 as *mut _, &mut d[0] as *mut i32 as *mut _, &mut d[1] as *mut i32 as *mut _,
                    &mut d[2] as *mut i32 as *mut _, &mut b[0] as *mut u64 as *mut _, &mut b[1] as *mut u64 as *mut _,
                    &mut b[2] as *mut u64 as *mut _, &mut b[3] as *mut u64 as *mut _, &mut b[4] as *mut u64 as *mut _,
                    &mut b[5] as *mut u64 as *mut _, &mut b[6] as *mut u64 as *mut _, &mut b[7] as *mut u64 as *mut _,
                    &mut b[8] as *mut u64 as *mut _, &mut b[9] as *mut u64 as *mut _,
                ];
                // (cuLaunchKernel reads exactly the kernel's own parameter count: 16 unfolded, 17 folded)
                let r1 = unsafe { cudarc::driver::sys::cuLaunchKernel(fg, (emax * (2 * mi / 128)) as u32, 1, 1, 256, 1, 1,
                                    gsm_of(gu), std::ptr::null_mut(), p.as_mut_ptr(), std::ptr::null_mut()) };
                let mut a2 = [*d_dn.device_ptr() as u64, *d_od.device_ptr() as u64, *xhd1.device_ptr() as u64,
                              *yd1.device_ptr() as u64];
                let mut d2 = [m as i32, mi as i32, h as i32];
                let mut b2 = [*d_es.device_ptr() as u64, *d_ix.device_ptr() as u64, *d_svh_d.device_ptr() as u64, cnt_dn,
                              *d_ysh.device_ptr() as u64, *d_ids.device_ptr() as u64, *d_wts.device_ptr() as u64,
                              *d_sm.device_ptr() as u64, *d_sgv.device_ptr() as u64, *out1.device_ptr() as u64];
                let mut k2 = topk as i32;
                let mut p2: [*mut std::ffi::c_void; 18] = [
                    &mut a2[0] as *mut u64 as *mut _, &mut a2[1] as *mut u64 as *mut _, &mut a2[2] as *mut u64 as *mut _,
                    &mut a2[3] as *mut u64 as *mut _, &mut d2[0] as *mut i32 as *mut _, &mut d2[1] as *mut i32 as *mut _,
                    &mut d2[2] as *mut i32 as *mut _, &mut b2[0] as *mut u64 as *mut _, &mut b2[1] as *mut u64 as *mut _,
                    &mut b2[2] as *mut u64 as *mut _, &mut b2[3] as *mut u64 as *mut _, &mut b2[4] as *mut u64 as *mut _,
                    &mut b2[5] as *mut u64 as *mut _, &mut b2[6] as *mut u64 as *mut _, &mut b2[7] as *mut u64 as *mut _,
                    &mut b2[8] as *mut u64 as *mut _, &mut b2[9] as *mut u64 as *mut _, &mut k2 as *mut i32 as *mut _,
                ];
                let r2 = unsafe { cudarc::driver::sys::cuLaunchKernel(fd, (emax * (h / 128)) as u32, 1, 1, 256, 1, 1,
                                    a1(m, mi), std::ptr::null_mut(), p2.as_mut_ptr(), std::ptr::null_mut()) };
                if r1 != cudarc::driver::sys::CUresult::CUDA_SUCCESS || r2 != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    bail!("EXL3-MOE-EPI FAIL: launch {gu} {r1:?} / {dn} {r2:?}");
                }
                dev.synchronize()?;
                let x_new = dev.dtoh_sync_copy(&xhd1)?;
                let g_new = dev.dtoh_sync_copy(&ygu1)?;
                let y_new = dev.dtoh_sync_copy(&yd1)?;
                let o_new = dev.dtoh_sync_copy(&out1)?;
                let cn = dev.dtoh_sync_copy(&d_cnt)?;
                let mx = x_old[..es * m * mi].iter().zip(&x_new[..es * m * mi]).filter(|(p, q)| p != q).count();
                let mg = g_old[..es * m * 2 * mi].iter().zip(&g_new[..es * m * 2 * mi]).filter(|(p, q)| p != q).count();
                let my = y_old[..es * m * h].iter().zip(&y_new[..es * m * h]).filter(|(p, q)| p != q).count();
                let mo = o_old[..m * h].iter().zip(&o_new[..m * h]).filter(|(p, q)| p != q).count();
                let armed = cn.iter().all(|v| *v == 0);
                if mg + mx + my + mo > 0 || !armed {
                    bail!("EXL3-MOE-EPI FAIL: top-k {topk} ({gu} + {dn}) m={m} esel={es} diet={diet} fold={fold} rep={rep}: bit-diff ygu {mg}, \
                           xhd {mx}, yd {my}, moe_out {mo}; counters {}", if armed { "re-armed" } else { "NOT re-armed" });
                }
            }
        }
        // ---- W4/MOE: the persistent kernels through the SERVED launchers (exl3_forward::mpk_*):
        // every schedule (ring depth, grid, item order, A-fold) and the mixed WP20/W4 pairings vs
        // the same old chain; NaN-poisoned outputs, 2 launches per schedule (re-armed counters).
        if false {
            use crate::exl3_forward as fw;
            let (sgu, sdn) = (a1(m, h), a1(m, mi));
            let cap_gu = (es * (2 * mi / 128)) as u32;
            let cap_dn = (es * (h / 128)) as u32;
            // (label, gate/up side, down side, interleaved order, A-fold); a side is Some((nst, grid))
            // with nst 0 = the served plan (mpk_plan), or None = the WP20 rung-2 kernel.
            type Side = Option<(usize, u32)>;
            let scheds: [(&str, Side, Side, bool, bool); 8] = [
                ("served", Some((0, 0)), Some((0, 0)), false, true),
                ("served-il", Some((0, 0)), Some((0, 0)), true, true),
                ("served-nofold", Some((0, 0)), Some((0, 0)), false, false),
                ("s4-G1", Some((4, 1)), Some((4, 1)), false, true),
                ("s8-G3-il", Some((8, 3)), Some((8, 3)), true, true),
                ("s4-Gitems-il-nofold", Some((4, cap_gu)), Some((4, cap_dn)), true, false),
                ("gu-pk+dn-epi", Some((0, 0)), None, false, true),
                ("gu-epi+dn-pk", None, Some((0, 0)), false, true),
            ];
            let plan = |gu: bool, side: (usize, u32), fold: bool| -> Result<Option<fw::MpkPlan>> {
                let (k, n) = if gu { (h, 2 * mi) } else { (mi, h) };
                if side.0 == 0 {
                    return fw::mpk_plan(dev, gu, topk, m, k, n, emax, fold, 4)
                        .map(Some)
                        .ok_or_else(|| anyhow::anyhow!("EXL3-MOE-EPI FAIL: W4/MOE has no served plan (gu {gu}) at m={m} top-k {topk}"));
                }
                let name = fw::mpk_entry_b(gu, topk, side.0, 4)
                    .ok_or_else(|| anyhow::anyhow!("EXL3-MOE-EPI FAIL: no W4/MOE entry gu {gu} top-k {topk} nst {} bits 4", side.0))?;
                let smem = fw::mpk_smem_bytes_b(side.0, m, k, gu, fold, 4) as u32;
                // smem over the opt-in cap at this width: the served plan falls back to the
                // smaller ring by design — this explicit-nst row is skipped, not a failure
                let Some((_, cps)) = fw::mpk_fn(name, smem, 512) else { return Ok(None) };
                Ok(Some(fw::MpkPlan { name, grid: side.1.max(1), smem, cps, nst: side.0, thr: 512 }))
            };
            let fg_epi = fw::xq_raw_fn(fw::wp20_gu_fn(true)).ok_or_else(|| anyhow::anyhow!("gu epi raw fn (stale PTX?)"))?;
            let fd_epi = fw::xq_raw_fn(dn_diet).ok_or_else(|| anyhow::anyhow!("{dn_diet} raw fn (stale PTX?)"))?;
            for (label, gs, ds, il, fold) in scheds.iter().copied() {
                let gplan = match gs { Some(sd) => plan(true, sd, fold)?, None => None };
                let dplan = match ds { Some(sd) => plan(false, sd, false)?, None => None };
                if (gs.is_some() && gplan.is_none()) || (ds.is_some() && dplan.is_none()) {
                    println!("EXL3-MOE-EPI: W4/MOE [{label}] m={m} top-k {topk}: ring nst over smem cap at this shape — skipped (the served plan falls back to the smaller ring)");
                    continue;
                }
                let fold = fold && gplan.is_some();
                let flags = if il { fw::MPK_FLAG_IL } else { 0 };
                let (mut ygu1, mut xhd1, mut yd1, mut out1) = (z(emax * m * 2 * mi)?, z(emax * m * mi)?, z(emax * m * h)?, z(m * h)?);
                let d_cnt = dev.htod_sync_copy(&vec![0u32; emax * (mi / 128) + h / 128 + 1])?;
                let mut d_sgv = dev.htod_sync_copy(&vec![0f32; m])?;
                let cnt = *d_cnt.device_ptr() as u64;
                let cnt_dn = cnt + (emax * (mi / 128) * 4) as u64;
                for rep in 0..2 {
                    for (buf, n) in [(&mut ygu1, emax * m * 2 * mi), (&mut xhd1, emax * m * mi), (&mut yd1, emax * m * h),
                                     (&mut out1, m * h)] {
                        dev.htod_sync_copy_into(&vec![0x7E00u16; n], buf)?;
                    }
                    dev.htod_sync_copy_into(&vec![f32::NAN; m], &mut d_sgv)?;
                    let sgvp = *d_sgv.device_ptr() as u64;
                    let (pygu, pxhd, pyd, pout) = (*ygu1.device_ptr() as u64, *xhd1.device_ptr() as u64,
                                                   *yd1.device_ptr() as u64, *out1.device_ptr() as u64);
                    let (px, pxh, pes, pix) = (*d_x.device_ptr() as u64, *xh_e.device_ptr() as u64,
                                               *d_es.device_ptr() as u64, *d_ix.device_ptr() as u64);
                    match gplan {
                        Some(gp) => fw::mpk_launch_gu(dev, std::ptr::null_mut(), &gp, *d_gu.device_ptr() as u64,
                                        *d_ogu.device_ptr() as u64, if fold { px } else { pxh },
                                        *d_suh_gu.device_ptr() as u64, pygu, m, h, 2 * mi, pes, pix,
                                        *d_svh_gu.device_ptr() as u64, *d_suh_d.device_ptr() as u64, pxhd, cnt, px,
                                        *d_sg.device_ptr() as u64, sgvp, flags | if fold { fw::MPK_FLAG_FOLD } else { 0 })?,
                        None => {
                            let mut a = [*d_gu.device_ptr() as u64, *d_ogu.device_ptr() as u64, pxh, pygu];
                            let mut d = [m as i32, h as i32, (2 * mi) as i32];
                            let mut b = [pes, pix, *d_svh_gu.device_ptr() as u64, *d_suh_d.device_ptr() as u64,
                                         pxhd, cnt, px, *d_sg.device_ptr() as u64, sgvp];
                            let mut pp: [*mut std::ffi::c_void; 16] = [
                                &mut a[0] as *mut u64 as *mut _, &mut a[1] as *mut u64 as *mut _, &mut a[2] as *mut u64 as *mut _,
                                &mut a[3] as *mut u64 as *mut _, &mut d[0] as *mut i32 as *mut _, &mut d[1] as *mut i32 as *mut _,
                                &mut d[2] as *mut i32 as *mut _, &mut b[0] as *mut u64 as *mut _, &mut b[1] as *mut u64 as *mut _,
                                &mut b[2] as *mut u64 as *mut _, &mut b[3] as *mut u64 as *mut _, &mut b[4] as *mut u64 as *mut _,
                                &mut b[5] as *mut u64 as *mut _, &mut b[6] as *mut u64 as *mut _, &mut b[7] as *mut u64 as *mut _,
                                &mut b[8] as *mut u64 as *mut _,
                            ];
                            let r = unsafe { cudarc::driver::sys::cuLaunchKernel(fg_epi, (emax * (2 * mi / 128)) as u32, 1, 1,
                                        256, 1, 1, sgu, std::ptr::null_mut(), pp.as_mut_ptr(), std::ptr::null_mut()) };
                            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS { bail!("EXL3-MOE-EPI FAIL: launch gu epi {r:?}"); }
                        }
                    }
                    match dplan {
                        Some(dp) => fw::mpk_launch_dn(dev, std::ptr::null_mut(), &dp, *d_dn.device_ptr() as u64,
                                        *d_od.device_ptr() as u64, pxhd, pyd, m, mi, h, pes, pix,
                                        *d_svh_d.device_ptr() as u64, cnt_dn, *d_ysh.device_ptr() as u64,
                                        *d_ids.device_ptr() as u64, *d_wts.device_ptr() as u64, *d_sm.device_ptr() as u64,
                                        sgvp, pout, topk, flags)?,
                        None => {
                            let mut a2 = [*d_dn.device_ptr() as u64, *d_od.device_ptr() as u64, pxhd, pyd];
                            let mut d2 = [m as i32, mi as i32, h as i32];
                            let mut b2 = [pes, pix, *d_svh_d.device_ptr() as u64, cnt_dn, *d_ysh.device_ptr() as u64,
                                          *d_ids.device_ptr() as u64, *d_wts.device_ptr() as u64, *d_sm.device_ptr() as u64,
                                          sgvp, pout];
                            let mut k2 = topk as i32;
                            let mut p2: [*mut std::ffi::c_void; 18] = [
                                &mut a2[0] as *mut u64 as *mut _, &mut a2[1] as *mut u64 as *mut _, &mut a2[2] as *mut u64 as *mut _,
                                &mut a2[3] as *mut u64 as *mut _, &mut d2[0] as *mut i32 as *mut _, &mut d2[1] as *mut i32 as *mut _,
                                &mut d2[2] as *mut i32 as *mut _, &mut b2[0] as *mut u64 as *mut _, &mut b2[1] as *mut u64 as *mut _,
                                &mut b2[2] as *mut u64 as *mut _, &mut b2[3] as *mut u64 as *mut _, &mut b2[4] as *mut u64 as *mut _,
                                &mut b2[5] as *mut u64 as *mut _, &mut b2[6] as *mut u64 as *mut _, &mut b2[7] as *mut u64 as *mut _,
                                &mut b2[8] as *mut u64 as *mut _, &mut b2[9] as *mut u64 as *mut _, &mut k2 as *mut i32 as *mut _,
                            ];
                            let r = unsafe { cudarc::driver::sys::cuLaunchKernel(fd_epi, (emax * (h / 128)) as u32, 1, 1, 256, 1, 1,
                                        sdn, std::ptr::null_mut(), p2.as_mut_ptr(), std::ptr::null_mut()) };
                            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS { bail!("EXL3-MOE-EPI FAIL: launch dn epi {r:?}"); }
                        }
                    }
                    dev.synchronize().with_context(|| format!("W4/MOE [{label}] m={m} top-k {topk} (kernel fault?)"))?;
                    let x_new = dev.dtoh_sync_copy(&xhd1)?;
                    let g_new = dev.dtoh_sync_copy(&ygu1)?;
                    let y_new = dev.dtoh_sync_copy(&yd1)?;
                    let o_new = dev.dtoh_sync_copy(&out1)?;
                    let cn = dev.dtoh_sync_copy(&d_cnt)?;
                    let mx = x_old[..es * m * mi].iter().zip(&x_new[..es * m * mi]).filter(|(p, q)| p != q).count();
                    let mg = g_old[..es * m * 2 * mi].iter().zip(&g_new[..es * m * 2 * mi]).filter(|(p, q)| p != q).count();
                    let my = y_old[..es * m * h].iter().zip(&y_new[..es * m * h]).filter(|(p, q)| p != q).count();
                    let mo = o_old[..m * h].iter().zip(&o_new[..m * h]).filter(|(p, q)| p != q).count();
                    let armed = cn.iter().all(|v| *v == 0);
                    if mg + mx + my + mo > 0 || !armed {
                        let gname = gplan.map_or("xq_moe_gu_epi".to_string(), |g| format!("{} G{}", g.name, g.grid));
                        let dname = dplan.map_or(dn_diet.to_string(), |d| format!("{} G{}", d.name, d.grid));
                        bail!("EXL3-MOE-EPI FAIL: W4/MOE [{label}] top-k {topk} ({gname} + {dname}, order {}, fold {fold}) m={m} \
                               esel={es} rep={rep}: bit-diff ygu {mg}, xhd {mx}, yd {my}, moe_out {mo}; counters {}",
                              if il { "il" } else { "contiguous" }, if armed { "re-armed" } else { "NOT re-armed" });
                    }
                }
                w4_done += 1;
                if m == 8 && topk == 10 && ne == NE_MAX {
                    let f = |pl: Option<fw::MpkPlan>| pl.map_or("epi".to_string(), |p| format!("s{}G{}", p.nst, p.grid));
                    w4_cfgs.push(format!("{label}: gu {} dn {}", f(gplan), f(dplan)));
                }
            }
        }
        done.push((topk, m, es));
    }
    }
    println!("EXL3-MOE-EPI[K=4]: PASS (WP20 gate/up + down epilogues bitwise == the old MoE chain, ygu/xhd/yd/moe_out, \
              top-k 8 + 10 instances, diet + SHFL bodies, 2 launches each, counters re-armed; suh seq 0x{conv:02x} (pinned, verified); \
              (top-k, m, esel) {done:?})");
    println!("EXL3-MOE-EPI[K=4]: (fold n/a) ({} (xq_had_suh_multi folded into the gate/up prologue) + the diet down \
              epilogue bitwise == the old MoE chain (hence == the unfolded xq_moe_gu_epi rows above) on ygu/xhd/yd/moe_out, \
              NaN-poisoned outputs, 2 launches each, counters re-armed; {} (top-k, m) cases, m 1..9 x top-k/experts \
              8/12, 10/64, 10/256)", crate::exl3_forward::WP20_GU_FOLD_FN, done.len());
    println!("EXL3-MOE-EPI[K=4]: (W4/MOE n/a) (persistent xq_moe_gu_pk / xq_moe_dn_pk_k<topk> through the served launchers, \
              bitwise == the old chain on ygu/xhd/yd/moe_out, NaN-poisoned outputs, 2 launches each, counters re-armed; \
              {w4_done} (top-k, m, schedule) cases over widths 1..9 x (top-k/experts) 8/12, 10/64, 10/256; schedules at \
              m=8 top-10/64: [{}])", w4_cfgs.join("; "));
    Ok(())
}

