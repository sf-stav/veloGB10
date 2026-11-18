//! WP20 epilogue synth/calibration helpers and the MoE epilogue probes (K=3, K=4 twin, bits-wide).
//! Probe-only code of the `exl3_bench` module (see `mod.rs` for the map); shared helpers come from the parent.
//! Numerics contract (module header in `mod.rs`): the kernel decode is bitwise vs the host oracle; do not change reduction orders.

use super::*;

/// Deterministic f16 vector with values in about [-4, 4] * scale (LCG; synth_x's generator).
pub(super) fn wp20_synth(n: usize, seed: u32, scale: f32) -> Vec<u16> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..n).map(|_| {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let v = ((((s >> 8) % 4001) as f32 - 2000.0) / 512.0) * scale;
        f16::from_f32(v).to_bits()
    }).collect()
}

/// A5 WP20-v1c: the id of the epilogue's pinned suh sequence (kernels/exl3_bench.cu
/// wp20_suh_had), in v1's convention-word encoding (2 bits per op: add01 sub01 add23 sub23;
/// 1 = "first product fused": FMUL q = x1*s1, FFMA x0*s0 +- q) — exactly the op sequence of
/// xq_had_suh_multi's served SASS. Logged only; the kernels hard-code the sequence.
pub(crate) const WP20_SUH_SEQ: u32 = 0x55;

/// Deterministic f16 vector spanning the f16 domain's edge cases (xorshift; never inf/NaN).
/// A Hadamard output depends on all 128 inputs of its block, so the regime is chosen PER
/// 128-BLOCK (block b: b % 4): 0 = |v| in [2^-10, 32) with +-0 / subnormals sprinkled,
/// 1 = subnormals and tiny normals (|v| < 2^-8), 2 = large (|v| < 2^10: f16 outputs can
/// overflow to inf), 3 = any finite pattern incl. +-65504 (mostly saturating outputs).
pub(super) fn wp20_synth_edge(n: usize, seed: u64) -> Vec<u16> {
    let mut s = seed | 1;
    (0..n).map(|i| {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        let (b, sm) = (s as u16, (s as u16) & 0x83FF);       // random sign + mantissa
        let e = |lo: u64, hi: u64| (((s >> 32) % (hi - lo + 1) + lo) as u16) << 10;
        match ((i / 128) % 4, (s >> 20) % 16) {
            (0, 0) => sm & 0x8000,                               // +-0
            (0, 1) => sm,                                        // +-subnormal (or 0)
            (0, _) => sm | e(5, 19),
            (1, _) => sm | e(0, 6),
            (2, _) => sm | e(10, 24),
            (_, 0) => (b & 0x8000) | 0x7BFF,                     // +-65504
            _ => if b & 0x7C00 == 0x7C00 { b & 0xBBFF } else { b },
        }
    }).collect()
}

/// A5 WP20-v1c: boot VERIFICATION of the epilogue's pinned suh sequence against the JIT'd
/// xq_had_suh_multi itself. v1/v1b searched 81 contraction conventions for a unique bitwise
/// match; x and s are f16, so x*s is exact in f32 and every convention yields the same bits —
/// all 81 matched and the search always reported "no unique match" (rung 2 OFF at boot on
/// preview p2). The epilogue now pins xq_had_suh_multi's served SASS op sequence with .rn
/// intrinsics (wp20_suh_had); this runs xq_wp20_suh_cal (xq_had_suh_multi's signature and
/// indexing, the pinned arithmetic) and the real kernel on the same inputs — per-expert input
/// form (the down transform's), 2 experts x 16 rows x 640: expert 0 = realistic activations and
/// suh (v1's synth rows), expert 1 = f16 edge cases (+-0, subnormals, +-65504, any finite;
/// f16-overflowing outputs included) — and requires every output byte equal.
/// Some(WP20_SUH_SEQ) = verified; None = a mismatch (rung 2 then stays off, loudly).
pub(crate) fn wp20_calibrate(dev: &std::sync::Arc<CudaDevice>) -> Result<Option<u32>> {
    let (e, r, k) = (2usize, 16usize, 640usize);
    let mut xs = wp20_synth(r * k, 0x20C0, 1.0);
    xs.extend(wp20_synth_edge(r * k, 0x20C1_5EED_0001));
    let mut ss = wp20_synth(k, 0x5CA1, 0.5);
    ss.extend(wp20_synth_edge(k, 0x5CA1_5EED_0002));
    let d_x = dev.htod_sync_copy(&xs)?;
    let d_s = dev.htod_sync_copy(&ss)?;
    let d_ix = dev.htod_sync_copy(&[0i32, 1])?;
    let d_es = dev.htod_sync_copy(&[e as i32])?;
    // poisoned outputs: a launch that skips a write can never compare equal by accident
    let mut d_ref = dev.htod_sync_copy(&vec![0x7E00u16; e * r * k])?;
    let mut d_out = dev.htod_sync_copy(&vec![0x7E01u16; e * r * k])?;
    let f_old = dev.get_func(MODULE, "xq_had_suh_multi").context("xq_had_suh_multi")?;
    let f_cal = dev.get_func(MODULE, "xq_wp20_suh_cal").context("xq_wp20_suh_cal (stale PTX?)")?;
    let g = (e * r * (k / 128)) as u32;
    let lc = LaunchConfig { grid_dim: (g, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
    dev.synchronize()?;
    unsafe {
        f_old.launch(lc, (&d_x, &d_s, &d_ix, &mut d_ref, r as i32, k as i32, &d_es, 1i32))?;
        f_cal.launch(lc, (&d_x, &d_s, &d_ix, &mut d_out, r as i32, k as i32, &d_es, 1i32))?;
    }
    dev.synchronize()?;
    let rf = dev.dtoh_sync_copy(&d_ref)?;
    let out = dev.dtoh_sync_copy(&d_out)?;
    // plausibility: the realistic expert is finite and mostly nonzero; nothing is a poison NaN
    let real = &rf[..r * k];
    if real.iter().filter(|v| **v & 0x7FFF != 0).count() < r * k / 2
        || real.iter().any(|v| *v & 0x7C00 == 0x7C00)
        || rf.iter().any(|v| *v & 0x7FFF > 0x7C00) {
        bail!("WP20 suh verification: implausible xq_had_suh_multi reference");
    }
    let diff = rf.iter().zip(&out).filter(|(p, q)| p != q).count();
    if diff != 0 {
        let i = rf.iter().zip(&out).position(|(p, q)| p != q).unwrap_or(0);
        println!("WP20 suh verification: pinned sequence differs from xq_had_suh_multi in {diff}/{} f16 \
                  (first [{i}] old {:04x} new {:04x})", e * r * k, rf[i], out[i]);
        return Ok(None);
    }
    Ok(Some(WP20_SUH_SEQ))
}

/// A5 WP20 gate (binv grouped section): the MoE epilogue kernels vs the OLD launch chain
/// (had_suh_multi -> a1b3 -> had_svh_multi -> gate_mul -> had_suh_multi -> a1b3 -> had_svh_multi
/// -> moe_combine) on synthetic 3-bit experts (h 2560, mi 640 — the served shapes) for EVERY
/// compiled down-epilogue top-k (WP20-v1b): top-8 over 12 experts (exactly the v1 case) and
/// top-10 over 64 experts (Qwen3.8-Flash-Next's top-k; up to ~50 live experts per launch):
/// ygu, xhd, yd and moe_out BITWISE, both decode bodies (diet / SHFL), widths {1,2,3,5,6,8,9}
/// (8 = the widened verify, 9 = the gate/up A-once cap), each run twice (the second launch rides
/// the re-armed counters), every counter back at 0. Grids padded to w*topk with the device live
/// count, as served. The per-top-k entry names come from the launcher's own table
/// (exl3_forward::wp20_dn_fn), so the gate exercises exactly what serving launches.
pub(super) fn probe_moe_epi(dev: &std::sync::Arc<CudaDevice>, bits: usize) -> Result<()> {
    const NE_MAX: usize = 64;      // physical trellis sets (expert id e uses set e % NE_MAX)
    const NE_TAB: usize = 256;     // suh/svh rows (the 256-expert routing case: every id distinct)
    let (h, mi) = (2560usize, 640usize);
    let conv = match wp20_calibrate(dev)? {
        Some(c) => c,
        None => bail!("EXL3-MOE-EPI FAIL: the pinned suh sequence does not match xq_had_suh_multi (see the line above)"),
    };
    let gu_words = (h / 16) * (2 * mi / 16) * (16 * bits);
    let d_words = (mi / 16) * (h / 16) * (16 * bits);
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
    // B10: the old chain's grouped GEMM at this width (K=3 keeps a1b3)
    let gname = match bits { 2 => "xq_gemm_grouped_a1b2", 4 => "xq_gemm_grouped_a1b4", 5 => "xq_gemm_grouped_a1b5",
                             6 => "xq_gemm_grouped_a1b6", 7 => "xq_gemm_grouped_a1b7", _ => "xq_gemm_grouped_a1b3" };
    let (f_suh, f_gemm, f_svh, f_gm, f_comb) =
        (f("xq_had_suh_multi")?, f(gname)?, f("xq_had_svh_multi")?, f("xq_moe_gate_mul")?, f("xq_moe_combine")?);
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
    for (topk, ne) in [(8usize, 12usize), (10, NE_MAX), (10, NE_TAB)] {
    let (dn_diet, dn_sh) = match (crate::exl3_forward::wp20_dn_fn(topk, true), crate::exl3_forward::wp20_dn_fn(topk, false)) {
        (Some(a), Some(b)) => (a, b),
        _ => bail!("EXL3-MOE-EPI FAIL: no down-epilogue instance for top-k {topk} in the launcher table"),
    };
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
                (&d_gu, &d_ogu, &xh_e, &mut ygu_raw, m as i32, h as i32, (2 * mi) as i32, bits as i32, &d_es))?;
            f_svh.clone().launch(lc((emax * m * (2 * mi / 128)) as u32, 32, 0),
                (&ygu_raw, &d_svh_gu, &d_ix, &mut ygu, m as i32, (2 * mi) as i32, &d_es))?;
            f_gm.clone().launch(lc(((emax * m * mi) / 256 + 1) as u32, 256, 0), (&mut din, &ygu, (m * mi) as i64, &d_es))?;
            f_suh.clone().launch(lc((emax * m * (mi / 128)) as u32, 32, 0),
                (&din, &d_suh_d, &d_ix, &mut xhd0, m as i32, mi as i32, &d_es, 1i32))?;
            f_gemm.clone().launch(lc((emax * (h / 128)) as u32, 256, a1(m, mi)),
                (&d_dn, &d_od, &xhd0, &mut ydr, m as i32, mi as i32, h as i32, bits as i32, &d_es))?;
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
        // the WP20 epi rows are K=3's entries (K=4 has its own probe); other widths: persistent only
        let epi_rows: Vec<(bool, bool, &'static str, &'static str)> = if bits != 3 { vec![] } else if m >= 10 {
            // CF-P1e 16-row class: the K-chunked gate/up entries (2 and 4 chunks) + the diet down epilogue
            vec![(true, false, "xq_moe_gu_epi_c2", dn_diet), (true, false, "xq_moe_gu_epi_c4", dn_diet)]
        } else {
            vec![(true, false, crate::exl3_forward::wp20_gu_fn(true), dn_diet),
                 (false, false, crate::exl3_forward::wp20_gu_fn(false), dn_sh),
                 (true, true, crate::exl3_forward::WP20_GU_FOLD_FN, dn_diet)]
        };
        let gsm_of = |gu: &str| -> u32 {
            match gu { "xq_moe_gu_epi_c2" => (m * (h / 2 + 8) * 2) as u32, "xq_moe_gu_epi_c4" => (m * (h / 4 + 8) * 2) as u32, _ => a1(m, h) }
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
        if m <= 9 {
            use crate::exl3_forward as fw;
            let (sgu, sdn) = (a1(m, h), a1(m, mi));
            let cap_gu = (es * (2 * mi / 128)) as u32;
            let cap_dn = (es * (h / 128)) as u32;
            // (label, gate/up side, down side, interleaved order, A-fold); a side is Some((nst, grid))
            // with nst 0 = the served plan (mpk_plan), or None = the WP20 rung-2 kernel.
            type Side = Option<(usize, u32)>;
            let mut scheds: Vec<(&str, Side, Side, bool, bool)> = vec![
                ("served", Some((0, 0)), Some((0, 0)), false, true),
                ("served-il", Some((0, 0)), Some((0, 0)), true, true),
                ("served-nofold", Some((0, 0)), Some((0, 0)), false, false),
                ("s4-G1", Some((4, 1)), Some((4, 1)), false, true),
                ("s8-G3-il", Some((8, 3)), Some((8, 3)), true, true),
                ("s4-Gitems-il-nofold", Some((4, cap_gu)), Some((4, cap_dn)), true, false),
            ];
            // the mixed WP20-pairing rows are K=3's diet entries; B10 widths gate them out
            if bits == 3 {
                scheds.push(("gu-pk+dn-epi", Some((0, 0)), None, false, true));
                scheds.push(("gu-epi+dn-pk", None, Some((0, 0)), false, true));
            }
            let plan = |gu: bool, side: (usize, u32), fold: bool| -> Result<Option<fw::MpkPlan>> {
                let (k, n) = if gu { (h, 2 * mi) } else { (mi, h) };
                let thr = if bits <= 3 { 256 } else { 512 };   // MpkGeom<BITS>::THREADS
                if side.0 == 0 {
                    return fw::mpk_plan(dev, gu, topk, m, k, n, emax, fold, bits)
                        .map(Some)
                        .ok_or_else(|| anyhow::anyhow!("EXL3-MOE-EPI FAIL: W4/MOE has no served plan (gu {gu}) at m={m} top-k {topk} bits {bits}"));
                }
                let name = fw::mpk_entry_b(gu, topk, side.0, bits)
                    .ok_or_else(|| anyhow::anyhow!("EXL3-MOE-EPI FAIL: no W4/MOE entry gu {gu} top-k {topk} nst {} bits {bits}", side.0))?;
                let smem = fw::mpk_smem_bytes_b(side.0, m, k, gu, fold, bits) as u32;
                // smem over the opt-in cap at this width (e.g. b6/b7 gu s8): the served plan
                // falls back to the smaller ring by design — this explicit-nst row is skipped
                let Some((_, cps)) = fw::mpk_fn(name, smem, thr) else { return Ok(None) };
                Ok(Some(fw::MpkPlan { name, grid: side.1.max(1), smem, cps, nst: side.0, thr }))
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
                        // B10 forensics: classify the wrong moe_out words. NaN-pattern (incl. the
                        // 0x7E00 poison) = unwritten or NaN-propagated; finite-but-different = the
                        // combine read stale producer data (ordering/visibility). Also: which sgv
                        // came back, and exactly which counter words failed to re-arm (gu vs dn).
                        let (mut c_nan, mut c_stale, mut bi, mut li) = (0usize, 0usize, usize::MAX, 0usize);
                        let mut ex: Vec<(usize, u16, u16)> = Vec::new();
                        for (i, (p, q)) in o_old[..m * h].iter().zip(&o_new[..m * h]).enumerate() {
                            if p == q { continue; }
                            if (q & 0x7C00) == 0x7C00 && (q & 0x03FF) != 0 { c_nan += 1; } else { c_stale += 1; }
                            bi = bi.min(i); li = li.max(i);
                            if ex.len() < 4 { ex.push((i, *p, *q)); }
                        }
                        let sgv_now = dev.dtoh_sync_copy(&d_sgv)?;
                        // term decomposition inputs: solve new = gg*ysh + sum_j w_j * yd[slot_j]
                        let wts_now = dev.dtoh_sync_copy(&d_wts)?;
                        let ids_now = dev.dtoh_sync_copy(&d_ids)?;
                        let sm_now = dev.dtoh_sync_copy(&d_sm)?;
                        let ysh_now = dev.dtoh_sync_copy(&d_ysh)?;
                        let yd_now = dev.dtoh_sync_copy(&yd1)?;
                        let slots: Vec<i32> = ids_now.iter().map(|i| sm_now[*i as usize]).collect();
                        let terms: Vec<String> = (0..topk).map(|j| {
                            let s = slots[j];
                            if s < 0 { return format!("j{j}:slot-1"); }
                            let y4 = (0..4).map(|t| format!("{:.4}", f16::from_bits(yd_now[((s as usize * m) * h) + t]).to_f32()))
                                           .collect::<Vec<_>>().join(",");
                            format!("j{j}: s{s} w{:.4} y[{y4}]", wts_now[j])
                        }).collect();
                        let ysh4 = (0..4).map(|t| format!("{:.4}", ysh_now[t])).collect::<Vec<_>>().join(",");
                        let gu_words = emax * (mi / 128);
                        let bad_cnt: Vec<usize> = cn.iter().enumerate().filter(|(_, v)| **v != 0).map(|(i, _)| i).collect();
                        let (ilw, arm) = (if il { "il" } else { "contiguous" },
                                          if armed { "re-armed" } else { "NOT re-armed" });
                        let gname = gplan.map_or("xq_moe_gu_epi".to_string(), |g| format!("{} G{}", g.name, g.grid));
                        let dname = dplan.map_or(dn_diet.to_string(), |d| format!("{} G{}", d.name, d.grid));
                        bail!("EXL3-MOE-EPI FAIL: W4/MOE [{label}] top-k {topk} ({gname} + {dname}, order {ilw}, fold {fold}) \
                               m={m} esel={es} rep={rep}: bit-diff ygu {mg}, xhd {mx}, yd {my}, moe_out {mo} \
                               (nan {c_nan}, finite-stale {c_stale}, span {bi}..{li}, ex {ex:?}); sgv {sgv_now:?}; \
                               bad cnt idx {bad_cnt:?} of {} words (gu 0..{gu_words}, dn {gu_words}..); counters {arm}; terms [{}]; ysh4 [{}]",
                              cn.len(), terms.join("; "), ysh4);
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
    println!("EXL3-MOE-EPI: PASS (WP20 gate/up + down epilogues bitwise == the old MoE chain, ygu/xhd/yd/moe_out, \
              top-k 8 + 10 instances, diet + SHFL bodies, 2 launches each, counters re-armed; suh seq 0x{conv:02x} (pinned, verified); \
              (top-k, m, esel) {done:?})");
    println!("EXL3-MOE-EPI: A5-K7 GU_FOLD PASS ({} (xq_had_suh_multi folded into the gate/up prologue) + the diet down \
              epilogue bitwise == the old MoE chain (hence == the unfolded xq_moe_gu_epi rows above) on ygu/xhd/yd/moe_out, \
              NaN-poisoned outputs, 2 launches each, counters re-armed; {} (top-k, m) cases, m 1..9 x top-k/experts \
              8/12, 10/64, 10/256)", crate::exl3_forward::WP20_GU_FOLD_FN, done.len());
    println!("EXL3-MOE-EPI: W4/MOE PASS (persistent xq_moe_gu_pk / xq_moe_dn_pk_k<topk> through the served launchers, \
              bitwise == the old chain on ygu/xhd/yd/moe_out, NaN-poisoned outputs, 2 launches each, counters re-armed; \
              {w4_done} (top-k, m, schedule) cases over widths 1..9 x (top-k/experts) 8/12, 10/64, 10/256; schedules at \
              m=8 top-10/64: [{}])", w4_cfgs.join("; "));
    Ok(())
}

// P7 B3 (K = 5 experts): the WP20 b5 epilogues (SHFL body — the word diet is b3/b4-only by
// static_assert in wp20_a1_mainloop) + the K-chunked b5 gate/up entries + the a1b5 grouped
// rung-1 entry, vs the OLD K-generic chain (xq_gemm_grouped_xh at bits 5) that served K = 5
// before B3. Fold and W4/MOE rows are n/a at K = 5 (no b5 fold entry; w4moe is B10).
pub(super) fn probe_moe_epi_bwd(dev: &std::sync::Arc<CudaDevice>, bits: usize) -> Result<()> {
    const NE_MAX: usize = 64;      // physical trellis sets (expert id e uses set e % NE_MAX)
    const NE_TAB: usize = 256;     // suh/svh rows (the 256-expert routing case: every id distinct)
    let (h, mi) = (2560usize, 640usize);
    let conv = match wp20_calibrate(dev)? {
        Some(c) => c,
        None => bail!("EXL3-MOE-EPI FAIL: the pinned suh sequence does not match xq_had_suh_multi (see the line above)"),
    };
    // Word geometry: 16 * bits words per (16 k) x (16 n) tile (b5 = 80, b4 = 64 — one formula).
    let gu_words = (h / 16) * (2 * mi / 16) * 16 * bits;
    let d_words = (mi / 16) * (h / 16) * 16 * bits;
    let d_gu = dev.htod_sync_copy(&wp20_synth(NE_MAX * gu_words, 0x6101, 1.0))?;   // any u16 decodes finite
    let d_dn = dev.htod_sync_copy(&wp20_synth(NE_MAX * d_words, 0x6102, 1.0))?;
    let d_suh_gu = dev.htod_sync_copy(&wp20_synth(NE_TAB * h, 0x6003, 0.25))?;
    let d_svh_gu = dev.htod_sync_copy(&wp20_synth(NE_TAB * 2 * mi, 0x6004, 0.005))?;
    let d_suh_d = dev.htod_sync_copy(&wp20_synth(NE_TAB * mi, 0x6005, 0.02))?;
    let d_svh_d = dev.htod_sync_copy(&wp20_synth(NE_TAB * h, 0x6006, 0.01))?;
    let d_sg = dev.htod_sync_copy(&wp20_synth(h, 0x6007, 0.01))?;
    let f = |n: &str| dev.get_func(MODULE, n).with_context(|| n.to_string());
    let a1n = match bits { 2 => "xq_gemm_grouped_a1b2", 6 => "xq_gemm_grouped_a1b6", 7 => "xq_gemm_grouped_a1b7", _ => "xq_gemm_grouped_a1b5" };
    let (gu_1, gu_c2, gu_c4) = match bits {
        2 => ("xq_moe_gu_epi_b2", "xq_moe_gu_epi_b2_c2", "xq_moe_gu_epi_b2_c4"),
        6 => ("xq_moe_gu_epi_b6", "xq_moe_gu_epi_b6_c2", "xq_moe_gu_epi_b6_c4"),
        7 => ("xq_moe_gu_epi_b7", "xq_moe_gu_epi_b7_c2", "xq_moe_gu_epi_b7_c4"),
        _ => ("xq_moe_gu_epi_b5", "xq_moe_gu_epi_b5_c2", "xq_moe_gu_epi_b5_c4"),
    };
    let (f_suh, f_xh, f_svh, f_gm, f_comb, f_a1b5) = (
        f("xq_had_suh_multi")?, f("xq_gemm_grouped_xh")?, f("xq_had_svh_multi")?,
        f("xq_moe_gate_mul")?, f("xq_moe_combine")?, f(a1n)?);
    let lc = |g: u32, b: u32, s: u32| LaunchConfig { grid_dim: (g, 1, 1), block_dim: (b, 1, 1), shared_mem_bytes: s };
    let a1 = |m: usize, k: usize| (m * (k + 8) * 2) as u32;
    // B3 leg 2 debug: the fatter xh (bodies 2..7) trips cuLaunchKernel INVALID_VALUE at dyn
    // 46224 (m=9). Dump the TRUE JIT static shared + opt-in state, then opt this copy in.
    {
        use cudarc::driver::sys;
        if let Some(fh) = crate::exl3_forward::xq_raw_fn("xq_gemm_grouped_xh") {
            for (attr, nm) in [(sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, "static smem"),
                               (sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, "max dyn opt-in"),
                               (sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS, "regs")] {
                let mut v: i32 = 0;
                let r = unsafe { sys::cuFuncGetAttribute(&mut v, attr, fh) };
                eprintln!("EPI-ATTR xh {nm} = {v} ({r:?})");
            }
        } else { eprintln!("EPI-ATTR xh raw fn MISSING"); }
    }
    // a1_smem() semantics: over the 48 KiB cap -> 0 (the barrier body; what served m >= 10 pre-B3)
    let a1c = |m: usize, k: usize| { let b = (m * (k + 8) * 2) as u32; if b > 48 * 1024 { 0 } else { b } };
    if let Ok(avail) = f_xh.occupancy_available_dynamic_smem_per_block(1, 256) {
        eprintln!("EPI-ATTR xh dev-copy avail dyn smem @b256 = {avail}");
    }
    let mut done = Vec::new();
    let mut r1_done = 0usize;
    // top-10 only at K = 5: the b5 down epilogue is a top-10 instantiation (mirrors b4).
    for (topk, ne) in [(10usize, NE_MAX), (10, NE_TAB)] {
    let dn_epi: &'static str = match bits { 2 => "xq_moe_dn_epi_b2_k10", 6 => "xq_moe_dn_epi_b6_k10", 7 => "xq_moe_dn_epi_b7_k10", _ => "xq_moe_dn_epi_b5_k10" };
    for m in 1usize..=16 {
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
        let (mut xhd0, mut ydr, mut yd0, mut out0) = (z(emax * m * mi)?, z(emax * m * h)?, z(emax * m * h)?, z(m * h)?);        // ---- the OLD chain (what served K = 5 before B3): xh at bits 5 for BOTH gemms
        unsafe {
            f_suh.clone().launch(lc((emax * m * (h / 128)) as u32, 32, 0),
                (&d_x, &d_suh_gu, &d_ix, &mut xh_e, m as i32, h as i32, &d_es, 0i32))?;
            f_xh.clone().launch(lc((emax * (2 * mi / 128)) as u32, 256, a1c(m, h)),
                (&d_gu, &d_ogu, &xh_e, &mut ygu_raw, m as i32, h as i32, (2 * mi) as i32, bits as i32, &d_es))?;
            f_svh.clone().launch(lc((emax * m * (2 * mi / 128)) as u32, 32, 0),
                (&ygu_raw, &d_svh_gu, &d_ix, &mut ygu, m as i32, (2 * mi) as i32, &d_es))?;
            f_gm.clone().launch(lc(((emax * m * mi) / 256 + 1) as u32, 256, 0), (&mut din, &ygu, (m * mi) as i64, &d_es))?;
            f_suh.clone().launch(lc((emax * m * (mi / 128)) as u32, 32, 0),
                (&din, &d_suh_d, &d_ix, &mut xhd0, m as i32, mi as i32, &d_es, 1i32))?;
            f_xh.clone().launch(lc((emax * (h / 128)) as u32, 256, a1(m, mi)),
                (&d_dn, &d_od, &xhd0, &mut ydr, m as i32, mi as i32, h as i32, bits as i32, &d_es))?;
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
        let graw_old = dev.dtoh_sync_copy(&ygu_raw)?;
        let ydr_old = dev.dtoh_sync_copy(&ydr)?;
        let finite = |v: &[u16]| v.iter().all(|b| b & 0x7C00 != 0x7C00);
        if !finite(&o_old[..m * h]) || !finite(&x_old[..es * m * mi])
            || o_old[..m * h].iter().filter(|b| **b & 0x7FFF != 0).count() < m * h / 2 {
            bail!("EXL3-MOE-EPI FAIL: implausible reference (m={m}: non-finite or mostly-zero old-chain output)");
        }        // ---- rung 1: the a1b5 grouped entry (the grouped_fn arm B3 adds) == xh at bits 5.
        // The gu A-once tile fits only m <= 9 at h = 2560 (a1 smem cap); the down GEMM fits all m.
        {
            let mut graw1 = z(emax * m * 2 * mi)?;
            let mut ydr1 = z(emax * m * h)?;
            unsafe {
                if m <= 9 {
                    f_a1b5.clone().launch(lc((emax * (2 * mi / 128)) as u32, 256, a1(m, h)),
                        (&d_gu, &d_ogu, &xh_e, &mut graw1, m as i32, h as i32, (2 * mi) as i32, bits as i32, &d_es))?;
                }
                f_a1b5.clone().launch(lc((emax * (h / 128)) as u32, 256, a1(m, mi)),
                    (&d_dn, &d_od, &xhd0, &mut ydr1, m as i32, mi as i32, h as i32, bits as i32, &d_es))?;
            }
            dev.synchronize()?;
            let graw_new = dev.dtoh_sync_copy(&graw1)?;
            let ydr_new = dev.dtoh_sync_copy(&ydr1)?;
            let mg = if m <= 9 { graw_old[..es * m * 2 * mi].iter().zip(&graw_new[..es * m * 2 * mi]).filter(|(p, q)| p != q).count() } else { 0 };
            let my = ydr_old[..es * m * h].iter().zip(&ydr_new[..es * m * h]).filter(|(p, q)| p != q).count();
            if mg + my > 0 {
                bail!("EXL3-MOE-EPI FAIL: rung-1 {a1n} != xh (bits {bits}) m={m} top-k {topk}: bit-diff ygu_raw {mg}, ydr {my}");
            }
            r1_done += if m <= 9 { 2 } else { 1 };
        }
        // ---- the b5 epilogue rows: SHFL body, K-chunked at m >= 10 (a1(m, h) exceeds the cap)
        let epi_rows: Vec<&'static str> = if m >= 10 {
            vec![gu_c2, gu_c4]
        } else {
            vec![gu_1]
        };
        let gsm_of = |gu: &str| -> u32 {
            match gu { g if g == gu_c2 => (m * (h / 2 + 8) * 2) as u32, g if g == gu_c4 => (m * (h / 4 + 8) * 2) as u32, _ => a1(m, h) }
        };        for gu in epi_rows {
            let (mut ygu1, mut xhd1, mut yd1, mut out1) = (z(emax * m * 2 * mi)?, z(emax * m * mi)?, z(emax * m * h)?, z(m * h)?);
            let d_cnt = dev.htod_sync_copy(&vec![0u32; emax * (mi / 128) + h / 128 + 1])?;
            let mut d_sgv = dev.htod_sync_copy(&vec![0f32; m])?;
            let cnt = *d_cnt.device_ptr() as u64;
            let cnt_dn = cnt + (emax * (mi / 128) * 4) as u64;
            let fg = crate::exl3_forward::xq_raw_fn(gu).ok_or_else(|| anyhow::anyhow!("{gu} raw fn (stale PTX?)"))?;
            let fd = crate::exl3_forward::xq_raw_fn(dn_epi).ok_or_else(|| anyhow::anyhow!("{dn_epi} raw fn (stale PTX?)"))?;
            for rep in 0..2 {                // NaN-poison every output first: a launch that skips a write can never pass on
                // the previous rep's bytes (the counters must really re-arm for rep 1).
                for (buf, n) in [(&mut ygu1, emax * m * 2 * mi), (&mut xhd1, emax * m * mi), (&mut yd1, emax * m * h),
                                 (&mut out1, m * h)] {
                    dev.htod_sync_copy_into(&vec![0x7E00u16; n], buf)?;
                }
                dev.htod_sync_copy_into(&vec![f32::NAN; m], &mut d_sgv)?;
                let mut a = [*d_gu.device_ptr() as u64, *d_ogu.device_ptr() as u64,
                             *xh_e.device_ptr() as u64, *ygu1.device_ptr() as u64];
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
                // (cuLaunchKernel reads exactly the kernel's own parameter count: 16 unfolded)
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
                    bail!("EXL3-MOE-EPI FAIL: launch {gu} {r1:?} / {dn_epi} {r2:?}");
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
                    bail!("EXL3-MOE-EPI FAIL: top-k {topk} ({gu} + {dn_epi}) m={m} esel={es} rep={rep}: bit-diff ygu {mg}, \
                           xhd {mx}, yd {my}, moe_out {mo}; counters {}", if armed { "re-armed" } else { "NOT re-armed" });
                }
            }
        }
        done.push((topk, m, es));
    }
    }
    println!("EXL3-MOE-EPI[K={bits}]: PASS (WP20 b{bits} SHFL gate/up epilogues (m 1..9) + the K-chunked b{bits} entries c2/c4 (m 10..16) \
              + the b{bits} top-10 down epilogue bitwise == the OLD K-generic xh chain (bits {bits}), on \
              ygu/xhd/yd/moe_out, 2 launches each, NaN-poisoned, counters re-armed; suh seq 0x{conv:02x} (pinned, verified); \
              (top-k, m, esel) {done:?})");
    println!("EXL3-MOE-EPI[K={bits}]: rung-1 PASS ({a1n} == xq_gemm_grouped_xh at bits {bits} on the gate/up \
              (m 1..9, the a1 smem cap at h 2560) and the down GEMM (m 1..16): {r1_done} comparisons)");
    println!("EXL3-MOE-EPI[K={bits}]: (fold n/a) (no b{bits} fold entry — the fold twin is b3-only) (W4/MOE n/a) (persistent \
              xq_moe_gu_pk / xq_moe_dn_pk are K = 3 instantiations; width-generic w4moe is B10)");
    Ok(())
}

