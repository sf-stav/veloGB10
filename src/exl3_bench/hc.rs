//! GDN tensor-core, prefill hyper-connection and W4 hyper-connection probes.
//! Probe-only code of the `exl3_bench` module (see `mod.rs` for the map); shared helpers come from the parent.
//! Numerics contract (module header in `mod.rs`): the kernel decode is bitwise vs the host oracle; do not change reduction orders.

use super::*;

// ---------------------------------------------------------------------------
// A5-L1: xq_gdn_chunk_tc2 (wavefront solve + register prefetch) must equal the served chunkwise GDN
// prefill scan xq_gdn_chunk_tc (split 7) BITWISE — core rows (f16) and the f32 recurrent state —
// at the served shape (nh 48 value heads, 16 key heads, kd = vd = 128; every head, slot 1 of 2) over
// N in {2048, 775, 96, 16} from a fresh state each, then the same four chained on one carried state.
// Core outputs NaN-poisoned first; the reference must leave no poison, must move the state and
// must write non-zero core rows. Inputs: q/k rows at spread magnitudes (incl. an all-zero q row and
// an all-zero k row: the eps path), v, beta/decay logits spanning strong decay .. g ~ 1.
// Harness-only: per-call timing at N = 2048 (x36 GDN layers = per 2048-row chunk) and the _prof
// twins' per-phase clock64 anatomy (cycles per 32-token sub-chunk, thread 0 between barriers).
pub(super) fn probe_gdn_tc2(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    use cudarc::driver::sys;
    let (nh, nk, kd, vd) = (48usize, 16usize, 128usize, 128usize);
    let stride = 2 * nk * kd + nh * vd;
    let nmax = 2048usize;
    let nslot = 2usize;
    let smem = crate::exl3_forward::GTC_SMEM;
    let get = |nm: &str| -> Result<sys::CUfunction> {
        let f = crate::exl3_forward::xq_raw_fn(nm).ok_or_else(|| anyhow::anyhow!("{nm} raw fn (stale PTX?)"))?;
        let r = unsafe { sys::cuFuncSetAttribute(f, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem) };
        if r != sys::CUresult::CUDA_SUCCESS { bail!("EXL3-GDN-TC2 FAIL: smem opt-in {nm} ({r:?})"); }
        Ok(f)
    };
    let (f_old, f_new, f_oldp, f_newp) = (get("xq_gdn_chunk_tc")?, get("xq_gdn_chunk_tc2")?,
                                          get("xq_gdn_chunk_tc_prof")?, get("xq_gdn_chunk_tc2_prof")?);
    // inputs (f16 bits)
    let mut qkv = wp20_synth(nmax * stride, 0x6D11, 0.6);
    for t in 0..nmax {
        let sc = [0.05f32, 0.3, 1.0, 4.0][t % 4];
        for i in 0..2 * nk * kd {
            let v = f16::from_bits(qkv[t * stride + i]).to_f32() * sc;
            qkv[t * stride + i] = f16::from_f32(v).to_bits();
        }
    }
    for i in 0..nk * kd { qkv[7 * stride + i] = 0; }                 // token 7: q all zero
    for i in 0..nk * kd { qkv[40 * stride + nk * kd + i] = 0; }      // token 40: k all zero
    let b_in = wp20_synth(nmax * nh, 0x6D12, 1.0);                   // beta = sigmoid(-3.9..3.9)
    let a_in: Vec<u16> = wp20_synth(nmax * nh, 0x6D13, 1.0).iter()
        .map(|&b| f16::from_f32(f16::from_bits(b).to_f32() - 1.5).to_bits()).collect();
    let a_log: Vec<f32> = (0..nh).map(|i| -2.5 + 3.5 * (i as f32) / (nh as f32)).collect();
    let dt_bias: Vec<f32> = (0..nh).map(|i| ((i * 7 % 11) as f32 - 5.0) / 5.0).collect();
    let st0: Vec<f32> = wp20_synth(nslot * nh * kd * vd, 0x6D14, 0.1).iter().map(|&b| f16::from_bits(b).to_f32()).collect();
    let d_qkv = dev.htod_sync_copy(&qkv)?;
    let d_b = dev.htod_sync_copy(&b_in)?;
    let d_a = dev.htod_sync_copy(&a_in)?;
    let d_alog = dev.htod_sync_copy(&a_log)?;
    let d_dtb = dev.htod_sync_copy(&dt_bias)?;
    let mut d_sa = dev.htod_sync_copy(&st0)?;
    let mut d_sb = dev.htod_sync_copy(&st0)?;
    let mut d_ca = dev.alloc_zeros::<u16>(nmax * nh * vd)?;
    let mut d_cb = dev.alloc_zeros::<u16>(nmax * nh * vd)?;
    let mut d_prof = dev.alloc_zeros::<u64>(nh * (vd / 32) * 10)?;
    let grid = (nh * (vd / 32)) as u32;
    let launch = |f: sys::CUfunction, core: u64, st: u64, qoff: usize, n: usize, prof: Option<u64>| -> Result<()> {
        let mut a = [core, *d_qkv.device_ptr() as u64 + (qoff * stride * 2) as u64, st,
                     *d_b.device_ptr() as u64 + (qoff * nh * 2) as u64, *d_a.device_ptr() as u64 + (qoff * nh * 2) as u64,
                     *d_alog.device_ptr() as u64, *d_dtb.device_ptr() as u64];
        let mut d = [(nh | (nk << 16)) as i32, 1i32, n as i32];
        let mut pp = prof.unwrap_or(0);
        let mut p: [*mut std::ffi::c_void; 11] = [
            &mut a[0] as *mut u64 as *mut _, &mut a[1] as *mut u64 as *mut _, &mut a[2] as *mut u64 as *mut _,
            &mut a[3] as *mut u64 as *mut _, &mut a[4] as *mut u64 as *mut _, &mut a[5] as *mut u64 as *mut _,
            &mut a[6] as *mut u64 as *mut _, &mut d[0] as *mut i32 as *mut _, &mut d[1] as *mut i32 as *mut _,
            &mut d[2] as *mut i32 as *mut _, &mut pp as *mut u64 as *mut _,
        ];
        let r = unsafe { sys::cuLaunchKernel(f, grid, 1, 1, 256, 1, 1, smem as u32, std::ptr::null_mut(),
                                             p.as_mut_ptr(), std::ptr::null_mut()) };
        if r != sys::CUresult::CUDA_SUCCESS { bail!("EXL3-GDN-TC2 FAIL: launch ({r:?})"); }
        Ok(())
    };
    let (pca, pcb, psa, psb) = (*d_ca.device_ptr() as u64, *d_cb.device_ptr() as u64,
                                *d_sa.device_ptr() as u64, *d_sb.device_ptr() as u64);
    let mut nchk = 0usize;
    let mut check = |tag: &str, n: usize, prev: &[f32], d_ca: &CudaSlice<u16>, d_cb: &CudaSlice<u16>,
                     d_sa: &CudaSlice<f32>, d_sb: &CudaSlice<f32>| -> Result<Vec<f32>> {
        let (ca, cb) = (dev.dtoh_sync_copy(d_ca)?, dev.dtoh_sync_copy(d_cb)?);
        let (sa, sb) = (dev.dtoh_sync_copy(d_sa)?, dev.dtoh_sync_copy(d_sb)?);
        let m = n * nh * vd;
        let bad_c = ca[..m].iter().zip(&cb[..m]).filter(|(p, q)| p != q).count();
        let bad_s = sa.iter().zip(&sb).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        if bad_c > 0 || bad_s > 0 {
            bail!("EXL3-GDN-TC2 FAIL {tag} N={n}: core {bad_c} of {m} f16 differ (first {:?}), state {bad_s} of {} f32 differ (first {:?})",
                  ca[..m].iter().zip(&cb[..m]).position(|(p, q)| p != q), sa.len(),
                  sa.iter().zip(&sb).position(|(p, q)| p.to_bits() != q.to_bits()));
        }
        if ca[..m].iter().any(|&v| v == 0x7E00) { bail!("EXL3-GDN-TC2 FAIL {tag} N={n}: poison left in the reference core"); }
        if sa.iter().any(|v| !v.is_finite()) { bail!("EXL3-GDN-TC2 FAIL {tag} N={n}: non-finite reference state"); }
        let moved = sa[nh * kd * vd..].iter().zip(&prev[nh * kd * vd..]).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        let slot0 = sa[..nh * kd * vd].iter().zip(&prev[..nh * kd * vd]).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        let nz = ca[..m].iter().filter(|&&v| v & 0x7FFF != 0).count();
        if moved < nh * kd * vd / 2 || slot0 != 0 || nz < m / 2 {
            bail!("EXL3-GDN-TC2 FAIL {tag} N={n}: implausible reference (state moved {moved}, slot 0 touched {slot0}, non-zero core {nz} of {m})");
        }
        nchk += 2;
        println!("  EXL3-GDN-TC2 {tag} N={n}: core ({m} f16) + state ({} f32) bitwise == xq_gdn_chunk_tc", sa.len());
        Ok(sa)
    };
    let poison = |d: &mut CudaSlice<u16>| -> Result<()> { let k = d.len(); dev.htod_sync_copy_into(&vec![0x7E00u16; k], d)?; Ok(()) };
    // fresh state per N (qkv window offsets differ so each N sees other rows)
    for (i, &n) in [2048usize, 775, 96, 16].iter().enumerate() {
        let qoff = [0usize, 1000, 1900, 2020][i].min(nmax - n);
        dev.htod_sync_copy_into(&st0, &mut d_sa)?; dev.htod_sync_copy_into(&st0, &mut d_sb)?;
        poison(&mut d_ca)?; poison(&mut d_cb)?;
        launch(f_old, pca, psa, qoff, n, None)?;
        launch(f_new, pcb, psb, qoff, n, None)?;
        dev.synchronize()?;
        check("fresh", n, &st0, &d_ca, &d_cb, &d_sa, &d_sb)?;
    }
    // chained: the four lengths back to back on one carried state (a 2048-row chunk then partial tails)
    dev.htod_sync_copy_into(&st0, &mut d_sa)?; dev.htod_sync_copy_into(&st0, &mut d_sb)?;
    let mut prev = st0.clone();
    for (i, &n) in [2048usize, 775, 96, 16].iter().enumerate() {
        let qoff = [0usize, 1273, 1952, 2032][i];
        poison(&mut d_ca)?; poison(&mut d_cb)?;
        launch(f_old, pca, psa, qoff, n, None)?;
        launch(f_new, pcb, psb, qoff, n, None)?;
        dev.synchronize()?;
        prev = check("chained", n, &prev, &d_ca, &d_cb, &d_sa, &d_sb)?;
    }
    // prof twins must compute the same bits too (they are the anatomy's instrument)
    dev.htod_sync_copy_into(&st0, &mut d_sa)?; dev.htod_sync_copy_into(&st0, &mut d_sb)?;
    poison(&mut d_ca)?; poison(&mut d_cb)?;
    launch(f_oldp, pca, psa, 0, nmax, Some(*d_prof.device_ptr() as u64))?;
    launch(f_newp, pcb, psb, 0, nmax, Some(*d_prof.device_ptr() as u64))?;
    dev.synchronize()?;
    check("prof-twins", nmax, &st0, &d_ca, &d_cb, &d_sa, &d_sb)?;
    println!("EXL3-GDN-TC2: PASS (A5-L1: xq_gdn_chunk_tc2 (wavefront solve, fused U, in-warp scan, register prefetch) bitwise == \
              xq_gdn_chunk_tc (split 7) at nh 48 / nk 16 / kd = vd = 128, slot 1 of 2; {nchk} comparisons over N in {{2048, 775, 96, 16}} \
              fresh + chained, NaN-poisoned core)");
    // ---- harness-only timing + anatomy (ranks only; the served A/B prices it)
    let reps = 20usize;
    let mut tm = |f: sys::CUfunction, core: u64, st: u64| -> Result<f64> {
        launch(f, core, st, 0, nmax, None)?; dev.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps { launch(f, core, st, 0, nmax, None)?; }
        dev.synchronize()?;
        Ok(t0.elapsed().as_secs_f64() * 1e3 / reps as f64)
    };
    let (t_old, t_new) = (tm(f_old, pca, psa)?, tm(f_new, pcb, psb)?);
    let (t_old2, t_new2) = (tm(f_old, pca, psa)?, tm(f_new, pcb, psb)?);
    println!("    EXL3-GDN-TC2 timing N=2048 (harness-only, ms/call; x36 = per chunk): chunk_tc {t_old:.3}/{t_old2:.3} -> tc2 {t_new:.3}/{t_new2:.3} \
              | per chunk {:.1} -> {:.1} ms", 36.0 * t_old.min(t_old2), 36.0 * t_new.min(t_new2));
    for (nm, f, core, st) in [("chunk_tc", f_oldp, pca, psa), ("tc2", f_newp, pcb, psb)] {
        launch(f, core, st, 0, nmax, Some(*d_prof.device_ptr() as u64))?;
        dev.synchronize()?;
        let pr = dev.dtoh_sync_copy(&d_prof)?;
        let nb = grid as usize;
        let nsub = (nmax / 32) as f64;
        let mut ph = [0f64; 8];
        let (mut cyc, mut ns, mut g0, mut g1) = (0f64, 0f64, u64::MAX, 0u64);
        for b in 0..nb {
            for i in 0..8 { ph[i] += pr[b * 10 + i] as f64 / nb as f64 / nsub; }
            cyc += (0..7).map(|i| pr[b * 10 + i] as f64).sum::<f64>();
            ns += (pr[b * 10 + 9] - pr[b * 10 + 8]) as f64;
            g0 = g0.min(pr[b * 10 + 8]); g1 = g1.max(pr[b * 10 + 9]);
        }
        let ghz = cyc / ns;
        println!("    EXL3-GDN-TC2 anatomy {nm} (cycles per 32-token sub-chunk, block-avg; {ghz:.2} GHz; block {:.1} us; span {:.3} ms): \
                  stage {:.0} | scan {:.0} | grams+W {:.0} | solve {:.0} | U {:.0} | O {:.0} | S {:.0} | total {:.0}",
                 ns / nb as f64 / 1e3, (g1 - g0) as f64 / 1e6, ph[0], ph[1], ph[2], ph[3], ph[4], ph[5], ph[6],
                 ph[..7].iter().sum::<f64>());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// A5-P2: the prefill hc elementwise twins must equal the served chain BITWISE at the served shape
// (h 2560, hc 4) over c in {2048, 775, 96, 16}, outputs NaN-poisoned first:
//   xq_pf_hc_inj_norm(y)    == xq_hc_inject + xq_hc_norm   (resid f32 AND hn f16)
//   xq_pf_hc_inj_norm(null) == xq_hc_norm                  (hn; resid untouched)
//   xq_pf_hc_mix_2560x4     == xq_hc_mix4                  (x f16 AND inj f32; with and without winj)
// Inputs: resid rows with per-row magnitudes spanning 1e-3..1e3 (+ exact zeros / subnormal-scale
// rows), the mix fed the REAL norm output. Also a harness-only timing at c = 2048 (x96 = per chunk).
pub(super) fn probe_pf_hc(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    let (h, hc) = (2560usize, 4usize);
    let n = h * hc;
    let eps = 1e-6f32;
    let f = |nm: &str| dev.get_func(MODULE, nm).with_context(|| nm.to_string());
    let lc = |g: u32, b: u32, sm: u32| LaunchConfig { grid_dim: (g, 1, 1), block_dim: (b, 1, 1), shared_mem_bytes: sm };
    let mut st = 0x9E37_79B9u32;
    let mut unif = move || { st = st.wrapping_mul(1664525).wrapping_add(1013904223); ((st >> 8) as f32 + 0.5) / 16777216.0 };
    let cmax = 2048usize;
    // resid: row r scale 10^((r % 7) - 3); every 13th row exactly zero in its first stream
    let mut resid0 = vec![0f32; cmax * n];
    for r in 0..cmax {
        let sc = 10f32.powi((r % 7) as i32 - 3);
        for i in 0..n {
            let v = (unif() * 2.0 - 1.0) * sc * (1.0 + 3.0 * (i % 97 == 0) as i32 as f32);
            resid0[r * n + i] = if r % 13 == 5 && i < h { 0.0 } else { v };
        }
    }
    let y = wp20_synth(cmax * h, 0xB0A1, 0.5);
    let injv: Vec<f32> = (0..cmax * hc).map(|_| unif() * 2.0).collect();
    let w = wp20_synth(n, 0xB0A2, 0.05);
    let uu = wp20_synth(cmax * n, 0xB0A3, 1.5);
    let winj = wp20_synth(hc * n, 0xB0A4, 0.01);
    let d_r0 = dev.htod_sync_copy(&resid0)?;
    let d_y = dev.htod_sync_copy(&y)?;
    let d_inj0 = dev.htod_sync_copy(&injv)?;
    let d_w = dev.htod_sync_copy(&w)?;
    let d_uu = dev.htod_sync_copy(&uu)?;
    let d_winj = dev.htod_sync_copy(&winj)?;
    let mut d_ra = dev.alloc_zeros::<f32>(cmax * n)?;
    let mut d_rb = dev.alloc_zeros::<f32>(cmax * n)?;
    let mut d_hna = dev.alloc_zeros::<u16>(cmax * n)?;
    let mut d_hnb = dev.alloc_zeros::<u16>(cmax * n)?;
    let mut d_xa = dev.alloc_zeros::<u16>(cmax * h)?;
    let mut d_xb = dev.alloc_zeros::<u16>(cmax * h)?;
    let mut d_ia = dev.alloc_zeros::<f32>(cmax * hc)?;
    let mut d_ib = dev.alloc_zeros::<f32>(cmax * hc)?;
    let (f_inject, f_norm, f_mix4, f_injn, f_mix) =
        (f("xq_hc_inject")?, f("xq_hc_norm")?, f("xq_hc_mix4")?, f("xq_pf_hc_inj_norm")?, f("xq_pf_hc_mix_2560x4")?);
    let yp = *d_y.device_ptr() as u64;
    let winjp = *d_winj.device_ptr() as u64;
    let pz16 = |d: &mut CudaSlice<u16>| -> Result<()> { let k = d.len(); dev.htod_sync_copy_into(&vec![0x7E00u16; k], d)?; Ok(()) };
    let pz32 = |d: &mut CudaSlice<f32>| -> Result<()> { let k = d.len(); dev.htod_sync_copy_into(&vec![f32::NAN; k], d)?; Ok(()) };
    let cmp16 = |c: usize, what: &str, a: &[u16], b: &[u16]| -> Result<()> {
        let bad = a.iter().zip(b.iter()).filter(|(p, q)| p != q).count();
        if bad > 0 || a.len() != b.len() {
            bail!("EXL3-PF-HC FAIL c={c} {what}: {bad} of {} f16 differ (first {:?})", a.len(), a.iter().zip(b.iter()).position(|(p, q)| p != q));
        }
        if a.iter().any(|&v| v == 0x7E00) { bail!("EXL3-PF-HC FAIL c={c} {what}: poisoned value left in the reference"); }
        Ok(())
    };
    let cmp32 = |c: usize, what: &str, a: &[f32], b: &[f32]| -> Result<()> {
        let bad = a.iter().zip(b.iter()).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        if bad > 0 || a.len() != b.len() {
            bail!("EXL3-PF-HC FAIL c={c} {what}: {bad} of {} f32 differ (first {:?})", a.len(),
                  a.iter().zip(b.iter()).position(|(p, q)| p.to_bits() != q.to_bits()));
        }
        if a.iter().any(|v| v.is_nan()) { bail!("EXL3-PF-HC FAIL c={c} {what}: NaN in the reference"); }
        Ok(())
    };
    let mut nchk = 0usize;
    for &c in &[2048usize, 775, 96, 16] {
        let (ci, hci, hi) = (c as i32, hc as i32, h as i32);
        // ---- inject + norm
        dev.dtod_copy(&d_r0, &mut d_ra)?; dev.dtod_copy(&d_r0, &mut d_rb)?;
        pz16(&mut d_hna)?; pz16(&mut d_hnb)?;
        unsafe {
            f_inject.clone().launch(lc(((n * c) as u32) / 256 + 1, 256, 0), (&mut d_ra, yp, &d_inj0, hi, hci, ci))?;
            f_norm.clone().launch(lc((hc * c) as u32, 1024, 4096), (&mut d_hna, &d_ra, &d_w, hi, hci, ci, eps))?;
            f_injn.clone().launch(lc((hc * c) as u32, 128, 0), (&mut d_hnb, &mut d_rb, yp, &d_inj0, &d_w, hi, hci, ci, eps))?;
        }
        dev.synchronize()?;
        cmp32(c, "inj_norm resid", &dev.dtoh_sync_copy(&d_ra)?[..c * n], &dev.dtoh_sync_copy(&d_rb)?[..c * n])?;
        let hn_ref = dev.dtoh_sync_copy(&d_hna)?;
        cmp16(c, "inj_norm hn", &hn_ref[..c * n], &dev.dtoh_sync_copy(&d_hnb)?[..c * n])?;
        nchk += 2;
        // ---- plain norm (y = null): the first hc_pre of a chunk / after the PLE flush
        dev.dtod_copy(&d_r0, &mut d_rb)?;
        pz16(&mut d_hna)?; pz16(&mut d_hnb)?;
        unsafe {
            f_norm.clone().launch(lc((hc * c) as u32, 1024, 4096), (&mut d_hna, &d_r0, &d_w, hi, hci, ci, eps))?;
            f_injn.clone().launch(lc((hc * c) as u32, 128, 0), (&mut d_hnb, &mut d_rb, 0u64, &d_inj0, &d_w, hi, hci, ci, eps))?;
        }
        dev.synchronize()?;
        cmp16(c, "norm hn", &dev.dtoh_sync_copy(&d_hna)?[..c * n], &dev.dtoh_sync_copy(&d_hnb)?[..c * n])?;
        cmp32(c, "norm resid untouched", &resid0[..c * n], &dev.dtoh_sync_copy(&d_rb)?[..c * n])?;
        nchk += 2;
        // ---- mix (on the real inject+norm output), with and without winj
        let d_hn = dev.htod_sync_copy(&hn_ref)?;
        for with_w in [true, false] {
            let wp = if with_w { winjp } else { 0u64 };
            pz16(&mut d_xa)?; pz16(&mut d_xb)?; pz32(&mut d_ia)?; pz32(&mut d_ib)?;
            unsafe {
                f_mix4.clone().launch(lc(c as u32, 1024, 4 * 1024 * 4), (&mut d_xa, &mut d_ia, &d_hn, &d_uu, wp, hi, hci, ci))?;
                f_mix.clone().launch(lc(c as u32, 128, 0), (&mut d_xb, &mut d_ib, &d_hn, &d_uu, wp, ci))?;
            }
            dev.synchronize()?;
            cmp16(c, &format!("mix x (winj {})", with_w as u8), &dev.dtoh_sync_copy(&d_xa)?[..c * h], &dev.dtoh_sync_copy(&d_xb)?[..c * h])?;
            nchk += 1;
            if with_w {
                cmp32(c, "mix inj", &dev.dtoh_sync_copy(&d_ia)?[..c * hc], &dev.dtoh_sync_copy(&d_ib)?[..c * hc])?;
                nchk += 1;
            }
        }
        println!("  EXL3-PF-HC c={c}: inject+norm, norm, mix (winj 1|0) bitwise == the served chain");
        // ---- timing (harness-only; 84 MB f32 state per call at c = 2048)
        if c == 2048 {
            let reps = 20usize;
            let mut t = |run: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
                run()?; dev.synchronize()?;
                let t0 = std::time::Instant::now();
                for _ in 0..reps { run()?; }
                dev.synchronize()?;
                Ok(t0.elapsed().as_secs_f64() * 1e3 / reps as f64)
            };
            let t_old_in = t(&mut || { unsafe {
                f_inject.clone().launch(lc(((n * c) as u32) / 256 + 1, 256, 0), (&mut d_ra, yp, &d_inj0, hi, hci, ci))?;
                f_norm.clone().launch(lc((hc * c) as u32, 1024, 4096), (&mut d_hna, &d_ra, &d_w, hi, hci, ci, eps))?; } Ok(()) })?;
            let t_new_in = t(&mut || { unsafe {
                f_injn.clone().launch(lc((hc * c) as u32, 128, 0), (&mut d_hnb, &mut d_rb, yp, &d_inj0, &d_w, hi, hci, ci, eps))?; } Ok(()) })?;
            let t_old_mix = t(&mut || { unsafe {
                f_mix4.clone().launch(lc(c as u32, 1024, 4 * 1024 * 4), (&mut d_xa, &mut d_ia, &d_hn, &d_uu, winjp, hi, hci, ci))?; } Ok(()) })?;
            let t_new_mix = t(&mut || { unsafe {
                f_mix.clone().launch(lc(c as u32, 128, 0), (&mut d_xb, &mut d_ib, &d_hn, &d_uu, winjp, ci))?; } Ok(()) })?;
            let gbs = |mb: f64, ms: f64| mb / ms;
            println!("    EXL3-PF-HC timing c=2048 (harness-only, ms/call; x96 = per chunk): inject+norm {t_old_in:.3} -> inj_norm {t_new_in:.3} \
                      ({:.0} GB/s on 220 MB) | mix4 {t_old_mix:.3} -> pf_mix {t_new_mix:.3} ({:.0} GB/s on 94 MB) | per chunk {:.1} -> {:.1} ms",
                     gbs(220.0, t_new_in), gbs(94.4, t_new_mix), 96.0 * (t_old_in + t_old_mix), 96.0 * (t_new_in + t_new_mix));
        }
    }
    println!("EXL3-PF-HC: PASS (A5-P2: xq_pf_hc_inj_norm (inject+norm and norm-only) and xq_pf_hc_mix_2560x4 (winj and final-mixer) \
              bitwise == xq_hc_inject + xq_hc_norm / xq_hc_mix4 at h 2560 hc 4; {nchk} comparisons over c in {{2048, 775, 96, 16}}, NaN-poisoned outputs)");
    Ok(())
}

// ---------------------------------------------------------------------------
// HC (decode wave 4): xq_hc_inj_norm + xq_hc_w4 must equal the p4c hc path (xq_hc_inject +
// xq_hc_norm + xq_hc_fuse_i8k (m = 1) / xq_hc_fuse_i8_rbk (m >= 2) + xq_hc_mix_rg) BITWISE on every
// output — resid, hn, dd, uu, x, inj — for m = 1..8, over grids {default, 8 (every wrap path: several
// units / gates / tiles per CTA, gates on unit CTAs), max} and flag sets (mix fused / separate, uu
// written or not, prefetch 0/1/2, look-ahead), with and without the inject gates (the final mixer),
// each launched twice (grid-barrier re-arm); outputs poisoned before every run. Batch invariance:
// row r of every width == row r of width 8. Then a harness-only, DRAM-cold (8 weight copies) timing.
// ---------------------------------------------------------------------------
pub(super) fn probe_hc_w4(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    let (h, hc, lr, mmax) = (2560usize, 4usize, 320usize, 8usize);
    let rw = h * hc;
    let eps = 1e-6f32;
    let cap = crate::exl3_forward::w4hc_capacity();
    if cap == 0 { bail!("EXL3-HC-W4 FAIL: xq_hc_w4 co-residency probe returned 0 (stale PTX / carveout pin refused)"); }
    let gdef = crate::exl3_forward::w4hc_default_grid(cap, h);
    let (gmin, gmax) = (h.div_ceil(32) as u32, (cap * 5) / 6);
    if gdef == 0 { bail!("EXL3-HC-W4 FAIL: co-residency {cap} x 5/6 below the phase-B minimum grid {gmin}"); }
    println!("  HC-W4: xq_hc_w4 co-residency {cap} blocks (256t, carveout max) -> default grid {gdef}, min {gmin}, max {gmax}");
    // ---- synthetic mixer: int8 over the FULL code range (incl. -128), fp32 per-row scales
    let mut st = 0x9E37_79B9u32;
    let mut rnd = move || { st ^= st << 13; st ^= st >> 17; st ^= st << 5; st };
    let mut mixer = |seed: u32| -> (Vec<i8>, Vec<f32>, Vec<i8>, Vec<f32>, Vec<u16>) {
        let qd: Vec<i8> = (0..lr * rw).map(|_| (rnd() >> 24) as u8 as i8).collect();
        let sd: Vec<f32> = (0..lr).map(|_| 4e-4 + (rnd() % 1000) as f32 * 1e-6).collect();
        let qu: Vec<i8> = (0..rw * lr).map(|_| (rnd() >> 24) as u8 as i8).collect();
        let su: Vec<f32> = (0..rw).map(|_| 6e-4 + (rnd() % 1000) as f32 * 1e-6).collect();
        let mut uk = vec![0i8; rw * lr];         // WP11 R3 k-major relayout (hc_quantize_i8)
        for n in 0..rw { for k in 0..lr { uk[((k / 16) * rw + n) * 16 + (k % 16)] = qu[n * lr + k]; } }
        (qd, sd, uk, su, wp20_synth(hc * rw, 0x51 ^ seed, 0.01))
    };
    let (qd, sd, qu, su, winj) = mixer(1);
    let (d_qd, d_sd, d_qu, d_su, d_winj) = (dev.htod_sync_copy(&qd)?, dev.htod_sync_copy(&sd)?,
        dev.htod_sync_copy(&qu)?, dev.htod_sync_copy(&su)?, dev.htod_sync_copy(&winj)?);
    let d_nw = dev.htod_sync_copy(&wp20_synth(hc * h, 0x77, 0.05))?;
    // per-row inputs (row r identical at every width: synth_x is row-seeded)
    let resid0: Vec<f32> = synth_x(mmax, rw, 0x4E51).iter().map(|&b| f16::from_bits(b).to_f32() * 0.5).collect();
    let y0 = synth_x(mmax, h, 0x7A11);
    let inj0: Vec<f32> = (0..mmax * hc).map(|i| 0.25 + 0.07 * i as f32).collect();
    let d_y = dev.htod_sync_copy(&y0)?;
    let mut d_resid = dev.htod_sync_copy(&resid0)?;
    let mut d_inj = dev.htod_sync_copy(&inj0)?;
    let mut d_hn = dev.htod_sync_copy(&vec![0u16; mmax * rw])?;
    let mut d_dd = dev.htod_sync_copy(&vec![0u16; mmax * lr])?;
    let mut d_uu = dev.htod_sync_copy(&vec![0u16; mmax * rw])?;
    let mut d_x = dev.htod_sync_copy(&vec![0u16; mmax * h])?;
    let d_bar = dev.htod_sync_copy(&[0i32])?;
    let f = |n: &str| dev.get_func(MODULE, n).with_context(|| format!("{n} missing"));
    let (f_inject, f_norm, f_i8k, f_rbk, f_mix, f_injn) = (f("xq_hc_inject")?, f("xq_hc_norm")?, f("xq_hc_fuse_i8k")?,
        f("xq_hc_fuse_i8_rbk")?, f("xq_hc_mix_rg")?, f("xq_hc_inj_norm")?);
    let lc = |g: (u32, u32, u32), b: u32, sm: u32| LaunchConfig { grid_dim: g, block_dim: (b, 1, 1), shared_mem_bytes: sm };
    let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
    struct Out { resid: Vec<f32>, hn: Vec<u16>, dd: Vec<u16>, uu: Vec<u16>, x: Vec<u16>, inj: Vec<f32> }
    // weights of one mixer copy, as device pointers: (qd, sd, qu, su, winj)
    type W = (u64, u64, u64, u64, u64);
    let w0: W = (dp(d_qd.device_ptr()), dp(d_sd.device_ptr()), dp(d_qu.device_ptr()), dp(d_su.device_ptr()),
                 dp(d_winj.device_ptr()));
    let hcn = hc as f32;
    // p4c fuse (+ separate mix) on hn -> dd/uu/x/inj; `gated` = inject gates present
    let old_fuse_mix = |d_dd: &mut CudaSlice<u16>, d_uu: &mut CudaSlice<u16>, d_x: &mut CudaSlice<u16>,
                        d_inj: &mut CudaSlice<f32>, d_hn: &CudaSlice<u16>, w: W, m: usize, gated: bool,
                        fuse: bool| -> Result<()> {
        if fuse {
            if m == 1 {
                unsafe { f_i8k.clone().launch(lc((40, 1, 1), 256, 0), (&mut *d_dd, &mut *d_uu, d_hn, w.0, w.1, w.2, w.3,
                    &d_bar, hcn, lr as i32, rw as i32, 1i32)) }?;
            } else {
                unsafe { f_rbk.clone().launch(lc((40, 1, 1), 256, 0), (&mut *d_dd, &mut *d_uu, d_hn, w.0, w.1, w.2, w.3,
                    &d_bar, hcn, (lr as i32) | ((m as i32) << 16), rw as i32, 17i32)) }?;
            }
        }
        let gx = (if gated { hc } else { 0 } + (h + 1023) / 1024) as u32;
        let wi = if gated { w.4 } else { 0u64 };
        unsafe { f_mix.clone().launch(lc((gx, m as u32, 1), 1024, 0), (&mut *d_x, &mut *d_inj, d_hn, &*d_uu, wi,
            h as i32, hc as i32, m as i32)) }?;
        Ok(())
    };
    let new_fuse = |w: W, m: usize, gated: bool, g: u32, flags: i32, ptrs: [u64; 5]| -> Result<()> {
        // ptrs: dd, uu, x, inj, hn
        crate::exl3_forward::w4hc_launch(std::ptr::null_mut(), g,
            [ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], w.0, w.1, w.2, w.3, if gated { w.4 } else { 0 }, dp(d_bar.device_ptr())],
            hcn, lr, m, h, hc, flags)
    };
    let poison = |d_hn: &mut CudaSlice<u16>, d_dd: &mut CudaSlice<u16>, d_uu: &mut CudaSlice<u16>, d_x: &mut CudaSlice<u16>,
                  d_resid: &mut CudaSlice<f32>, d_inj: &mut CudaSlice<f32>| -> Result<()> {
        dev.htod_copy_into(vec![0x7E00u16; mmax * rw], d_hn)?;
        dev.htod_copy_into(vec![0x7E01u16; mmax * lr], d_dd)?;
        dev.htod_copy_into(vec![0x7E02u16; mmax * rw], d_uu)?;
        dev.htod_copy_into(vec![0x7E03u16; mmax * h], d_x)?;
        dev.htod_copy_into(resid0.clone(), d_resid)?;
        dev.htod_copy_into(inj0.clone(), d_inj)?;
        Ok(())
    };
    let read = |d_resid: &CudaSlice<f32>, d_hn: &CudaSlice<u16>, d_dd: &CudaSlice<u16>, d_uu: &CudaSlice<u16>,
                d_x: &CudaSlice<u16>, d_inj: &CudaSlice<f32>| -> Result<Out> {
        dev.synchronize()?;
        Ok(Out { resid: dev.dtoh_sync_copy(d_resid)?, hn: dev.dtoh_sync_copy(d_hn)?, dd: dev.dtoh_sync_copy(d_dd)?,
                 uu: dev.dtoh_sync_copy(d_uu)?, x: dev.dtoh_sync_copy(d_x)?, inj: dev.dtoh_sync_copy(d_inj)? })
    };
    // (mismatches over the m live rows, first index) per buffer; uu skipped when `uu` is false
    let cmp = |a: &Out, b: &Out, m: usize, uu: bool| -> ([usize; 6], Option<(usize, usize)>) {
        fn d<T: PartialEq + Copy>(x: &[T], y: &[T], neq: impl Fn(T, T) -> bool) -> (usize, Option<usize>) {
            let mut k = (0usize, None);
            for (i, (&p, &q)) in x.iter().zip(y.iter()).enumerate() { if neq(p, q) { k.0 += 1; k.1.get_or_insert(i); } }
            k
        }
        let f32n = |p: f32, q: f32| p.to_bits() != q.to_bits();
        let u16n = |p: u16, q: u16| p != q;
        let r = [d(&a.resid[..m * rw], &b.resid[..m * rw], f32n), d(&a.hn[..m * rw], &b.hn[..m * rw], u16n),
                 d(&a.dd[..m * lr], &b.dd[..m * lr], u16n),
                 if uu { d(&a.uu[..m * rw], &b.uu[..m * rw], u16n) } else { (0, None) },
                 d(&a.x[..m * h], &b.x[..m * h], u16n), d(&a.inj[..m * hc], &b.inj[..m * hc], f32n)];
        let first = r.iter().enumerate().find_map(|(bi, k)| k.1.map(|i| (bi, i)));
        ([r[0].0, r[1].0, r[2].0, r[3].0, r[4].0, r[5].0], first)
    };
    const MIX: i32 = 1;
    const UU: i32 = 2;
    let cfgs: Vec<(&str, u32, i32)> = vec![
        ("default", gdef, MIX | UU | (1 << 2)),
        ("grid min (gates on unit CTAs at m >= 2), bulk pf, look-ahead 2", gmin, MIX | UU | (2 << 2) | (2 << 4)),
        ("grid max, no pf, look-ahead 1", gmax, MIX | UU | (1 << 4)),
        ("mix separate (uu only)", gdef, 1 << 2),
        ("default, uu not written", gdef, MIX | (1 << 2)),
    ];
    let (mut fails, mut checked) = (0usize, 0usize);
    let mut rows_by_m: Vec<(usize, bool, Out)> = Vec::new();
    for gated in [true, false] {
        for m in 1..=mmax {
            // ---- p4c reference
            poison(&mut d_hn, &mut d_dd, &mut d_uu, &mut d_x, &mut d_resid, &mut d_inj)?;
            unsafe { f_inject.clone().launch(lc((((h * hc * m) as u32) / 256 + 1, 1, 1), 256, 0),
                (&mut d_resid, &d_y, &d_inj, h as i32, hc as i32, m as i32)) }?;
            unsafe { f_norm.clone().launch(lc(((hc * m) as u32, 1, 1), 1024, 4096),
                (&mut d_hn, &d_resid, &d_nw, h as i32, hc as i32, m as i32, eps)) }?;
            old_fuse_mix(&mut d_dd, &mut d_uu, &mut d_x, &mut d_inj, &d_hn, w0, m, gated, true)?;
            let refo = read(&d_resid, &d_hn, &d_dd, &d_uu, &d_x, &d_inj)?;
            let nzx = refo.x[..m * h].iter().filter(|&&b| b & 0x7fff != 0 && b & 0x7c00 != 0x7c00).count();
            if nzx < m * h / 2 { bail!("EXL3-HC-W4 FAIL: implausible reference x (m={m}: {nzx} finite nonzero of {})", m * h); }
            // ---- the new path, every config, twice
            for (name, g, flags) in &cfgs {
                if *g < gmin || *g > gmax { continue; }
                for rep in 0..2 {
                    poison(&mut d_hn, &mut d_dd, &mut d_uu, &mut d_x, &mut d_resid, &mut d_inj)?;
                    unsafe { f_injn.clone().launch(lc(((hc * m) as u32, 1, 1), 1024, 0),
                        (&mut d_hn, &mut d_resid, &d_y, &d_inj, &d_nw, h as i32, hc as i32, m as i32, eps)) }?;
                    let ptrs = [dp(d_dd.device_ptr()), dp(d_uu.device_ptr()), dp(d_x.device_ptr()), dp(d_inj.device_ptr()),
                                dp(d_hn.device_ptr())];
                    new_fuse(w0, m, gated, *g, *flags, ptrs)?;
                    if flags & MIX == 0 {
                        old_fuse_mix(&mut d_dd, &mut d_uu, &mut d_x, &mut d_inj, &d_hn, w0, m, gated, false)?;
                    }
                    let o = read(&d_resid, &d_hn, &d_dd, &d_uu, &d_x, &d_inj)?;
                    let (k, first) = cmp(&refo, &o, m, flags & MIX == 0 || flags & UU != 0);
                    checked += 1;
                    if k.iter().sum::<usize>() > 0 {
                        fails += 1;
                        println!("    HC-W4 FAIL gates={gated} m={m} [{name}] G={g} rep={rep}: mismatches resid/hn/dd/uu/x/inj {k:?} \
                                  (first (buffer, idx) {first:?})");
                    }
                    if *name == "default" && rep == 0 { rows_by_m.push((m, gated, o)); }
                }
            }
        }
    }
    // ---- batch invariance: row r at width m == row r at width 8 (same gating)
    let mut binv_bad = 0usize;
    for gated in [true, false] {
        let full = rows_by_m.iter().find(|(m, g, _)| *m == mmax && *g == gated).map(|t| &t.2)
            .context("HC-W4: missing width-8 run")?;
        for (m, _, o) in rows_by_m.iter().filter(|(_, g, _)| *g == gated) {
            let (k, first) = cmp(full, o, *m, true);
            if k.iter().sum::<usize>() > 0 {
                binv_bad += 1;
                println!("    HC-W4 BINV FAIL gates={gated} width {m} vs 8: {k:?} (first {first:?})");
            }
        }
    }
    if fails + binv_bad > 0 {
        bail!("EXL3-HC-W4 FAIL ({fails} of {checked} runs differ from the p4c path, {binv_bad} width rows differ)");
    }
    println!("EXL3-HC-W4: PASS (xq_hc_inj_norm + xq_hc_w4 bitwise == xq_hc_inject + xq_hc_norm + i8k/rbk + mix_rg on \
              resid/hn/dd/uu/x/inj, M 1..=8, gated + final-mixer, {checked} runs over grids {{{gdef}, {gmin}, {gmax}}} x mix \
              fused/separate x pf 0/1/2 x look-ahead; batch-invariant rows across widths 1..8)");

    // ---- standalone timing (harness-only): 8 weight copies (52 MB > L2) rotated => DRAM-cold weights
    let nw = 8usize;
    let mut copies: Vec<(CudaSlice<i8>, CudaSlice<f32>, CudaSlice<i8>, CudaSlice<f32>, CudaSlice<u16>)> = Vec::new();
    for i in 0..nw {
        let (a, b, c, d, e) = mixer(100 + i as u32);
        copies.push((dev.htod_sync_copy(&a)?, dev.htod_sync_copy(&b)?, dev.htod_sync_copy(&c)?, dev.htod_sync_copy(&d)?,
                     dev.htod_sync_copy(&e)?));
    }
    let ws: Vec<W> = copies.iter().map(|c| (dp(c.0.device_ptr()), dp(c.1.device_ptr()), dp(c.2.device_ptr()),
                                            dp(c.3.device_ptr()), dp(c.4.device_ptr()))).collect();
    let iters = 30 * nw;
    let time = |run: &mut dyn FnMut(usize) -> Result<()>| -> Result<f64> {
        for it in 0..nw { run(it)?; }
        dev.synchronize()?;
        let t = std::time::Instant::now();
        for it in 0..iters { run(it)?; }
        dev.synchronize()?;
        Ok(t.elapsed().as_secs_f64() * 1e6 / iters as f64)
    };
    for m in [1usize, 2, 4, 6, 8] {
        let t_old = time(&mut |it| old_fuse_mix(&mut d_dd, &mut d_uu, &mut d_x, &mut d_inj, &d_hn, ws[it % nw], m, true, true))?;
        let ptrs = [dp(d_dd.device_ptr()), dp(d_uu.device_ptr()), dp(d_x.device_ptr()), dp(d_inj.device_ptr()),
                    dp(d_hn.device_ptr())];
        let t_new = time(&mut |it| new_fuse(ws[it % nw], m, true, gdef, MIX | (1 << 2), ptrs))?;
        let mut sweep = String::new();
        for g in [80u32, 88, 96, 112, 120, 144] {
            if g >= gmin && g <= gmax {
                let t = time(&mut |it| new_fuse(ws[it % nw], m, true, g, MIX | (1 << 2), ptrs))?;
                sweep += &format!(" G{g} {t:.1}");
            }
        }
        for (tag, fl) in [("pf0", MIX), ("pf2", MIX | (2 << 2)), ("la1", MIX | (1 << 2) | (1 << 4)),
                          ("la2", MIX | (1 << 2) | (2 << 4)), ("la3", MIX | (1 << 2) | (3 << 4))] {
            let t = time(&mut |it| new_fuse(ws[it % nw], m, true, gdef, fl, ptrs))?;
            sweep += &format!(" {tag} {t:.1}");
        }
        let t_in_old = time(&mut |_| {
            unsafe { f_inject.clone().launch(lc((((h * hc * m) as u32) / 256 + 1, 1, 1), 256, 0),
                (&mut d_resid, &d_y, &d_inj, h as i32, hc as i32, m as i32)) }?;
            unsafe { f_norm.clone().launch(lc(((hc * m) as u32, 1, 1), 1024, 4096),
                (&mut d_hn, &d_resid, &d_nw, h as i32, hc as i32, m as i32, eps)) }?;
            Ok(())
        })?;
        let t_in_new = time(&mut |_| {
            unsafe { f_injn.clone().launch(lc(((hc * m) as u32, 1, 1), 1024, 0),
                (&mut d_hn, &mut d_resid, &d_y, &d_inj, &d_nw, h as i32, hc as i32, m as i32, eps)) }?;
            Ok(())
        })?;
        println!("    EXL3-HC-W4 timing (standalone, harness-only, DRAM-cold weights) m={m}: p4c fuse+mix {t_old:.1} us | \
                  w4 G{gdef} {t_new:.1} us ({:+.1}) | sweep:{sweep} | inject+norm {t_in_old:.1} -> inj_norm {t_in_new:.1} us",
                 t_new - t_old);
        // TP-I3 (b): per-phase anatomy from the timestamp twin (same body), plain back-to-back launches;
        // cold = the 8 rotated weight copies (DRAM), warm = copy 0 only (L2-resident: the compute floor)
        for (arm, fl, warm) in [("cold pf1", MIX | (1 << 2), false), ("cold pf0", MIX, false), ("warm pf1", MIX | (1 << 2), true)] {
        let nl = 64usize;
        let gu = gdef as usize;
        const PS: usize = 16;
        let d_prof = dev.alloc_zeros::<u64>(nl * gu * PS)?;
        for it in 0..nl {
            let w = if warm { ws[0] } else { ws[it % nw] };
            crate::exl3_forward::w4hc_prof_launch(std::ptr::null_mut(), gdef,
                [ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], w.0, w.1, w.2, w.3, w.4, dp(d_bar.device_ptr())],
                dp(d_prof.device_ptr()) + (it * gu * PS * 8) as u64, hcn, lr, m, h, hc, fl)?;
        }
        dev.synchronize()?;
        let pr = dev.dtoh_sync_copy(&d_prof)?;
        // per launch, relative to the first CTA entry (us): [entry spread, wait end, A end (max), gates end (max),
        // barrier release (min t4), barrier last (max t4), up end (max), end (max), unit mean, silu staged (max)],
        // medians over launches nw..nl
        let mut acc: Vec<Vec<f64>> = vec![Vec::new(); 11];
        let nunits = if m >= 2 { 2 * lr.div_ceil(8) } else { lr.div_ceil(8) };
        for it in nw..nl {
            let p = &pr[it * gu * PS..(it + 1) * gu * PS];
            let col = |i: usize| (0..gu).map(move |c| p[c * PS + i]);
            let t0 = col(0).min().unwrap_or(0);
            let us = |t: u64| (t.saturating_sub(t0)) as f64 / 1e3;
            acc[0].push(us(col(0).max().unwrap()));
            acc[1].push(us(col(1).max().unwrap()));
            acc[2].push(us((0..gu.min(nunits)).map(|c| p[c * PS + 2]).max().unwrap()));
            acc[3].push(us(col(3).max().unwrap()));
            acc[4].push(us(col(4).min().unwrap()));
            acc[5].push(us(col(4).max().unwrap()));
            acc[6].push(us(col(5).max().unwrap()));
            acc[7].push(us(col(6).max().unwrap()));
            // phase-A unit duration (mean over unit CTAs, t2 - t1)
            let ua: f64 = (0..gu.min(nunits)).map(|c| (p[c * PS + 2] - p[c * PS + 1]) as f64 / 1e3).sum::<f64>()
                / gu.min(nunits) as f64;
            acc[8].push(ua);
            acc[9].push(us(col(8).max().unwrap()));
            // SM clock (MHz): per CTA clock64 cycles / globaltimer ns over its whole life, mean over CTAs
            let mhz: f64 = (0..gu).map(|c| (p[c * PS + 10].wrapping_sub(p[c * PS + 9])) as f64
                / ((p[c * PS + 6] - p[c * PS]).max(1) as f64) * 1e3).sum::<f64>() / gu as f64;
            acc[10].push(mhz);
        }
        let med = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
        let r: Vec<f64> = acc.iter_mut().map(med).collect();
        println!("    EXL3-HC-W4 anatomy m={m} G{gdef} {arm} (prof twin, median of {} launches, us from the first CTA entry): \
                  entry spread {:.1} | PDL wait end {:.1} | phase A end {:.1} (unit mean {:.1}) | gates end {:.1} | \
                  barrier release {:.1} (last {:.1}) | silu staged {:.1} | up end {:.1} | end {:.1} | SM clock {:.0} MHz",
                 nl - nw, r[0], r[1], r[2], r[8], r[3], r[4], r[5], r[9], r[6], r[7], r[10]);
        }
    }
    Ok(())
}

/// TP-I3 (b) (`--probe-exl3-hcw4`): the EXL3-HC-W4 section of --probe-exl3-binv alone (synthetic,
/// no pack): xq_hc_w4 bitwise vs the p4c hc path over m = 1..8 x grids x flags + the standalone
/// DRAM-cold timing table.
pub fn probe_hcw4() -> Result<()> {
    let dev = dev0()?;
    load_module(&dev)?;
    probe_hc_w4(&dev)
}

