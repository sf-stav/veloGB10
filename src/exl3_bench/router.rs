//! f16-rows GEMM, router fold, R16 and argmax-rows probes.
//! Probe-only code of the `exl3_bench` module (see `mod.rs` for the map); shared helpers come from the parent.
//! Numerics contract (module header in `mod.rs`): the kernel decode is bitwise vs the host oracle; do not change reduction orders.

use super::*;

// ---------------------------------------------------------------------------
// S-A3-m: xq_gemm_f16_rows (row-batched) must be BITWISE equal per row to
// xq_gemm_f16 (grid-y per row) — the PLE projection shapes, M in {1,3,6,16}.
// ---------------------------------------------------------------------------
pub(super) fn probe_gemm_f16_rows(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    let f_old = dev.get_func(MODULE, "xq_gemm_f16").context("xq_gemm_f16")?;
    let f_new = dev.get_func(MODULE, "xq_gemm_f16_rows").context("xq_gemm_f16_rows")?;
    for &(n, k) in &[(10240usize, 2560usize), (2560, 2560)] {
        let w = synth_x(n, k, 0xF16);
        let d_w = dev.htod_sync_copy(&w)?;
        for m in [1usize, 3, 6, 16] {
            let x = synth_x(m, k, 0xA11);
            let d_x = dev.htod_sync_copy(&x)?;
            let mut o1 = dev.htod_sync_copy(&vec![0u16; m * n])?;
            let mut o2 = dev.htod_sync_copy(&vec![0u16; m * n])?;
            let g = ((n + 127) / 128) as u32;
            unsafe {
                f_old.clone().launch(LaunchConfig { grid_dim: (g, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                     (&mut o1, &d_w, &d_x, m as i32, n as i32, k as i32))?;
                f_new.clone().launch(LaunchConfig { grid_dim: (g, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                     (&mut o2, &d_w, &d_x, m as i32, n as i32, k as i32))?;
            }
            dev.synchronize()?;
            let a = dev.dtoh_sync_copy(&o1)?;
            let b = dev.dtoh_sync_copy(&o2)?;
            if a.iter().filter(|v| **v != 0).count() < m * n / 2 {
                bail!("EXL3-F16ROWS FAIL: implausible reference output (m={m} n={n})");
            }
            let mism = a.iter().zip(b.iter()).filter(|(p, q)| p != q).count();
            if mism > 0 { bail!("EXL3-F16ROWS FAIL: m={m} N={n} K={k}: {mism} mismatches"); }
        }
    }
    println!("EXL3-F16ROWS: PASS (row-batched PLE GEMV bitwise == xq_gemm_f16, M in {{1,3,6,16}})");
    // WP09: the k-major twin ([K/16][N][16] weights, key+value in ONE launch — the served shape)
    // must be BITWISE equal to xq_gemm_f16 per row; plus a tail-path shape (K/16 % 4 != 0,
    // N % 128 != 0, single matrix).
    {
        let f_ref = dev.get_func(MODULE, "xq_gemm_f16").context("xq_gemm_f16")?;
        let f_km = dev.get_func(MODULE, "xq_gemm_f16_rows_km").context("xq_gemm_f16_rows_km")?;
        let km = crate::exl3_forward::relayout_km16;
        let mut done = Vec::new();
        for &(na, nb2, k) in &[(10240usize, 2560usize, 2560usize), (300, 0, 1040)] {
            let wa = synth_x(na, k, 0xF16);
            let wb = synth_x(nb2.max(1), k, 0xF17);
            let d_wa = dev.htod_sync_copy(&wa)?;
            let d_wb = dev.htod_sync_copy(&wb)?;
            let d_wa_km = dev.htod_sync_copy(&km(&wa, na, k))?;
            let d_wb_km = dev.htod_sync_copy(&km(&wb, nb2.max(1), k))?;
            let (ga, gb) = (((na + 127) / 128) as u32, ((nb2 + 127) / 128) as u32);
            for m in [1usize, 2, 3, 5, 6, 8, 9, 16] {
                let d_x = dev.htod_sync_copy(&synth_x(m, k, 0xA11 ^ m as u32))?;
                let mut ra = dev.htod_sync_copy(&vec![0u16; m * na])?;
                let mut rb = dev.htod_sync_copy(&vec![0u16; m * nb2.max(1)])?;
                let mut ka = dev.htod_sync_copy(&vec![0u16; m * na])?;
                let mut kb = dev.htod_sync_copy(&vec![0u16; m * nb2.max(1)])?;
                unsafe {
                    f_ref.clone().launch(LaunchConfig { grid_dim: (ga, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                         (&mut ra, &d_wa, &d_x, m as i32, na as i32, k as i32))?;
                    if nb2 > 0 {
                        f_ref.clone().launch(LaunchConfig { grid_dim: (gb, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                             (&mut rb, &d_wb, &d_x, m as i32, nb2 as i32, k as i32))?;
                    }
                    f_km.clone().launch(LaunchConfig { grid_dim: (ga + gb, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                        (&mut ka, &d_wa_km, na as i32, ga as i32, &mut kb, &d_wb_km, nb2 as i32,
                                         &d_x, m as i32, k as i32))?;
                }
                // W4/SMALL: the x-in-smem exact-M twin (m <= 8), 160-thread blocks, poisoned outputs
                let mut wa4 = dev.htod_sync_copy(&vec![0x7e01u16; m * na])?;
                let mut wb4 = dev.htod_sync_copy(&vec![0x7e01u16; m * nb2.max(1)])?;
                if m <= 8 {
                    let f_w4 = dev.get_func(MODULE, &format!("xq_gemm_f16_rows_km_w4m{m}"))
                        .ok_or_else(|| anyhow::anyhow!("xq_gemm_f16_rows_km_w4m{m} missing"))?;
                    let (ga4, gb4) = (na.div_ceil(160) as u32, nb2.div_ceil(160) as u32);
                    unsafe {
                        f_w4.launch(LaunchConfig { grid_dim: (ga4 + gb4, 1, 1), block_dim: (160, 1, 1),
                                                   shared_mem_bytes: (m * k * 2) as u32 },
                                    (&mut wa4, &d_wa_km, na as i32, ga4 as i32, &mut wb4, &d_wb_km, nb2 as i32,
                                     &d_x, m as i32, k as i32))?;
                    }
                }
                dev.synchronize()?;
                for (tag, r, t, t4, n) in [("a", &ra, &ka, &wa4, na), ("b", &rb, &kb, &wb4, nb2)] {
                    if n == 0 { continue; }
                    let a = dev.dtoh_sync_copy(r)?;
                    let b = dev.dtoh_sync_copy(t)?;
                    if a.iter().filter(|v| **v != 0).count() < m * n / 2 {
                        bail!("EXL3-F16ROWS-KM FAIL: implausible reference output (m={m} n={n} {tag})");
                    }
                    let mism = a.iter().zip(b.iter()).filter(|(p, q)| p != q).count();
                    if mism > 0 { bail!("EXL3-F16ROWS-KM FAIL: m={m} N={n} K={k} ({tag}): {mism} mismatches vs xq_gemm_f16"); }
                    if m <= 8 {
                        let c = dev.dtoh_sync_copy(t4)?;
                        let mism4 = a.iter().zip(c.iter()).take(m * n).filter(|(p, q)| p != q).count();
                        if mism4 > 0 {
                            bail!("EXL3-F16ROWS-KM FAIL: W4S xq_gemm_f16_rows_km_w4m{m} N={n} K={k} ({tag}): {mism4} mismatches vs xq_gemm_f16");
                        }
                    }
                }
                done.push((na, nb2, k, m));
            }
        }
        println!("EXL3-F16ROWS-KM: PASS (k-major key+value twin bitwise == xq_gemm_f16; {} (Na,Nb,K,M) cases incl. tail K/N; \
                  W4S _w4m<M> twin bitwise too at M <= 8)", done.len());
    }
    probe_argmax_rows(dev)?;
    // S-A3-u B: the prefill router twins must be BITWISE equal to xq_gemm_f16_f32 (router bits
    // decide top-k selections, AGENTS §7).
    {
        let f_ref = dev.get_func(MODULE, "xq_gemm_f16_f32").context("xq_gemm_f16_f32")?;
        let f_r32 = dev.get_func(MODULE, "xq_gemm_f16_f32_rows32").context("xq_gemm_f16_f32_rows32")?;
        let (n, k) = (512usize, 2560usize);
        let d_w = dev.htod_sync_copy(&synth_x(n, k, 0x5A3))?;
        for m in [1usize, 17, 33, 100, 2048] {
            let d_x = dev.htod_sync_copy(&synth_x(m, k, 0x77 ^ m as u32))?;
            let mut o1 = dev.htod_sync_copy(&vec![0f32; m * n])?;
            let mut o2 = dev.htod_sync_copy(&vec![0f32; m * n])?;
            unsafe {
                f_ref.clone().launch(LaunchConfig { grid_dim: ((n / 128) as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                     (&mut o1, &d_w, &d_x, m as i32, n as i32, k as i32))?;
                f_r32.clone().launch(LaunchConfig { grid_dim: ((n / 128) as u32, ((m + 31) / 32) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                     (&mut o2, &d_w, &d_x, m as i32, n as i32, k as i32))?;
            }
            dev.synchronize()?;
            let a = dev.dtoh_sync_copy(&o1)?; let b = dev.dtoh_sync_copy(&o2)?;
            if a.iter().filter(|v| **v != 0.0).count() < m * n / 2 { bail!("EXL3-ROUTER32 FAIL: implausible reference (m={m})"); }
            let mism = a.iter().zip(b.iter()).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
            if mism > 0 { bail!("EXL3-ROUTER32 FAIL: m={m}: {mism} mismatches vs xq_gemm_f16_f32"); }
        }
        println!("EXL3-ROUTER32: PASS (32-row smem router twin bitwise == xq_gemm_f16_f32, M in {{1,17,33,100,2048}})");
        // PFX1 (d): the 64-col x 8-row twin must equal BOTH the reference and the current router
        // (rows32) bit for bit at the served chunk shapes (and M = 2048, past its dispatch range).
        let f_r8 = dev.get_func(MODULE, "xq_gemm_f16_f32_rows8").context("xq_gemm_f16_f32_rows8")?;
        let ms8 = [1usize, 7, 22, 26, 72, 144, 384, 2048];
        for &m in &ms8 {
            let d_x = dev.htod_sync_copy(&synth_x(m, k, 0x88 ^ m as u32))?;
            let mut o1 = dev.htod_sync_copy(&vec![0f32; m * n])?;
            let mut o2 = dev.htod_sync_copy(&vec![0f32; m * n])?;
            // poisoned output: a row the kernel fails to write cannot pass as 0 == 0
            let mut o3 = dev.htod_sync_copy(&vec![f32::from_bits(0x7fc0_dead); m * n])?;
            unsafe {
                f_ref.clone().launch(LaunchConfig { grid_dim: ((n / 128) as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                     (&mut o1, &d_w, &d_x, m as i32, n as i32, k as i32))?;
                f_r32.clone().launch(LaunchConfig { grid_dim: ((n / 128) as u32, ((m + 31) / 32) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                     (&mut o2, &d_w, &d_x, m as i32, n as i32, k as i32))?;
                f_r8.clone().launch(LaunchConfig { grid_dim: ((n / 64) as u32, m.div_ceil(8) as u32, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 },
                                    (&mut o3, &d_w, &d_x, m as i32, n as i32, k as i32))?;
            }
            dev.synchronize()?;
            let a = dev.dtoh_sync_copy(&o1)?; let b = dev.dtoh_sync_copy(&o2)?; let c8 = dev.dtoh_sync_copy(&o3)?;
            if a.iter().filter(|v| **v != 0.0).count() < m * n / 2 { bail!("EXL3-ROUTER8 FAIL: implausible reference (m={m})"); }
            let mr = a.iter().zip(c8.iter()).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
            let m32 = b.iter().zip(c8.iter()).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
            if mr > 0 || m32 > 0 {
                bail!("EXL3-ROUTER8 FAIL: m={m}: {mr} mismatches vs xq_gemm_f16_f32, {m32} vs rows32 (of {})", m * n);
            }
            println!("EXL3-ROUTER8: m={m:>4}: 0 of {} bits differ (vs xq_gemm_f16_f32 and rows32)", m * n);
        }
        println!("EXL3-ROUTER8: PASS (64x8 register-prefetch router twin bitwise == xq_gemm_f16_f32 == rows32, M in {ms8:?})");
        // Standalone per-layer timing (harness-only: it RANKS the bodies, it does not price the
        // in-round cost — AGENTS §4). Weights rotated over 16 copies (42 MB > L2) so every
        // launch reads its router from DRAM, as the 48 per-layer routers of a prefill do.
        let nw = 16usize;
        let wall: Vec<u16> = (0..nw).flat_map(|i| synth_x(n, k, 0x5A3 ^ (i as u32 * 131))).collect();
        let d_wall = dev.htod_sync_copy(&wall)?;
        let wb0 = *d_wall.device_ptr() as u64;
        let wstride = (n * k * 2) as u64;
        // 256..448 bracket the rows8/rows32 crossover (PFX1_ROWS8_MAX, exl3_forward.rs); "served"
        // is what the prefill router dispatch picks at this m on this device (env overrides apply).
        for m in [26usize, 72, 160, 256, 320, 384, 448, 512, 736] {
            let d_x = dev.htod_sync_copy(&synth_x(m, k, 0x99 ^ m as u32))?;
            let mut o = dev.htod_sync_copy(&vec![0f32; m * n])?;
            let mut us = [0f64; 2];
            for (which, slot) in us.iter_mut().enumerate() {
                let mut run = |it: usize| -> Result<()> {
                    let wp = wb0 + (it % nw) as u64 * wstride;
                    unsafe {
                        if which == 0 {
                            f_r32.clone().launch(LaunchConfig { grid_dim: ((n / 128) as u32, ((m + 31) / 32) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                                 (&mut o, wp, &d_x, m as i32, n as i32, k as i32))?;
                        } else {
                            f_r8.clone().launch(LaunchConfig { grid_dim: ((n / 64) as u32, m.div_ceil(8) as u32, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 },
                                                (&mut o, wp, &d_x, m as i32, n as i32, k as i32))?;
                        }
                    }
                    Ok(())
                };
                for it in 0..nw { run(it)?; }
                dev.synchronize()?;
                let iters = 20 * nw;
                let t = std::time::Instant::now();
                for it in 0..iters { run(it)?; }
                dev.synchronize()?;
                *slot = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
            }
            println!("    EXL3-ROUTER8 timing (standalone, harness-only, DRAM-cold weights) m={m}: rows32 {:.1} us/layer, \
                      rows8 {:.1} us/layer ({:.2}x); x48 layers = {:.2} -> {:.2} ms per chunk; served: {}",
                     us[0], us[1], us[0] / us[1].max(1e-9), us[0] * 48e-3, us[1] * 48e-3,
                     if crate::exl3_forward::pfx1_router_rows8(m, n, k) { "rows8" } else { "rows32" });
        }
    }
    probe_router_fold(dev)?;
    probe_r16(dev)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// WP10: xq_router_fold (router GEMV + softmax / top-k / renorm / first-seen route in ONE launch)
// must be BITWISE equal to the S-A3-z pair xq_router_fused + xq_router_topk_route — logits, ids,
// wts, slotmap, esel, idxmap / offs_gu / offs_d [..esel] — for M in 1..=8 (ne 512, topk 10,
// K 2560) on random, tie-heavy (duplicated expert rows: equal logits -> equal probs -> id-asc
// order) and NaN-row data, each fold launched TWICE back-to-back (the self re-arming counter must
// read 0 after). Both GEMV lane maps (xq_router_fold, xq_router_fold_m0). Then a STANDALONE
// per-layer timing (weights rotated over 16 copies > L2): it ranks the bodies, it does not price
// the in-round cost (AGENTS §4) — harness-only numbers.
// ---------------------------------------------------------------------------
fn probe_router_fold(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    use cudarc::driver::sys;
    use std::ffi::c_void;
    let raw = |n: &str| crate::exl3_forward::xq_raw_fn(n)
        .ok_or_else(|| anyhow::anyhow!("{n} raw fn unavailable (stale PTX?)"));
    let (f_fused, f_route) = (raw("xq_router_fused")?, raw("xq_router_topk_route")?);
    let (f_fold, f_fold0) = (raw("xq_router_fold")?, raw("xq_router_fold_m0")?);
    // W4/SMALL: the persistent twin (grid #SMs x 768, x staged in smem)
    let f_w4 = raw("xq_router_fold_w4")?;
    // A5-K2: the coalesced-weight twin (grid ne/4 x 256, dynamic smem RFC_SMEM)
    let f_c = raw("xq_router_fold_c")?;
    let (ne, k, kd, mmax) = (512usize, 10usize, 2560usize, 8usize);
    let (gw, dw) = (0x0012_3457u64, 0x9abcu64);
    let g_w4 = crate::exl3_forward::w4s_router_grid(dev, ne);
    unsafe fn go(f: sys::CUfunction, grid: u32, smem: u32, args: &mut [*mut c_void]) -> Result<()> {
        go_b(f, grid, 256, smem, args)
    }
    unsafe fn go_b(f: sys::CUfunction, grid: u32, block: u32, smem: u32, args: &mut [*mut c_void]) -> Result<()> {
        let r = sys::cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, std::ptr::null_mut(),
                                    args.as_mut_ptr(), std::ptr::null_mut());
        anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "router fold probe launch ({r:?})");
        Ok(())
    }
    struct Out { lg: CudaSlice<f32>, ids: CudaSlice<i32>, wts: CudaSlice<f32>, sm: CudaSlice<i32>,
                 ix: CudaSlice<i32>, es: CudaSlice<i32>, og: CudaSlice<u64>, od: CudaSlice<u64>,
                 cnt: CudaSlice<i32> }
    let mk = || -> Result<Out> {
        Ok(Out { lg: dev.htod_sync_copy(&vec![0f32; mmax * ne])?, ids: dev.htod_sync_copy(&vec![-7i32; mmax * k])?,
                 wts: dev.htod_sync_copy(&vec![0f32; mmax * k])?, sm: dev.htod_sync_copy(&vec![-9i32; ne])?,
                 ix: dev.htod_sync_copy(&vec![-5i32; mmax * k])?, es: dev.htod_sync_copy(&[-1i32])?,
                 og: dev.htod_sync_copy(&vec![0u64; mmax * k])?, od: dev.htod_sync_copy(&vec![0u64; mmax * k])?,
                 cnt: dev.htod_sync_copy(&[0i32])? })
    };
    let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
    let pair = |o: &Out, w: u64, x: u64, m: usize| -> Result<()> {
        let (mut a_lg, mut a_w, mut a_x) = (dp(o.lg.device_ptr()), w, x);
        let (mut a_m, mut a_n, mut a_h) = (m as i32, ne as i32, kd as i32);
        let mut p1: [*mut c_void; 6] = [&mut a_lg as *mut u64 as *mut _, &mut a_w as *mut u64 as *mut _,
            &mut a_x as *mut u64 as *mut _, &mut a_m as *mut i32 as *mut _, &mut a_n as *mut i32 as *mut _,
            &mut a_h as *mut i32 as *mut _];
        unsafe { go(f_fused, (ne / 4) as u32, 0, &mut p1)? };
        let (mut a_ids, mut a_wts, mut a_lg2) = (dp(o.ids.device_ptr()), dp(o.wts.device_ptr()), dp(o.lg.device_ptr()));
        let (mut a_ne, mut a_k, mut a_b) = (ne as i32, k as i32, m as i32);
        let (mut a_cnt, mut a_sm, mut a_ix) = (dp(o.cnt.device_ptr()), dp(o.sm.device_ptr()), dp(o.ix.device_ptr()));
        let (mut a_es, mut a_og, mut a_od) = (dp(o.es.device_ptr()), dp(o.og.device_ptr()), dp(o.od.device_ptr()));
        let (mut a_gw, mut a_dw) = (gw, dw);
        let mut p2: [*mut c_void; 14] = [&mut a_ids as *mut u64 as *mut _, &mut a_wts as *mut u64 as *mut _,
            &mut a_lg2 as *mut u64 as *mut _, &mut a_ne as *mut i32 as *mut _, &mut a_k as *mut i32 as *mut _,
            &mut a_b as *mut i32 as *mut _, &mut a_cnt as *mut u64 as *mut _, &mut a_sm as *mut u64 as *mut _,
            &mut a_ix as *mut u64 as *mut _, &mut a_es as *mut u64 as *mut _, &mut a_og as *mut u64 as *mut _,
            &mut a_od as *mut u64 as *mut _, &mut a_gw as *mut u64 as *mut _, &mut a_dw as *mut u64 as *mut _];
        unsafe { go(f_route, m as u32, ((256 + ne + 8 * 16 * 2) * 4) as u32, &mut p2) }
    };
    let fold = |f: sys::CUfunction, o: &Out, w: u64, x: u64, m: usize| -> Result<()> {
        let (mut a_lg, mut a_w, mut a_x) = (dp(o.lg.device_ptr()), w, x);
        let (mut a_m, mut a_n, mut a_h) = (m as i32, ne as i32, kd as i32);
        let (mut a_cnt, mut a_ids, mut a_wts) = (dp(o.cnt.device_ptr()), dp(o.ids.device_ptr()), dp(o.wts.device_ptr()));
        let mut a_k = k as i32;
        let (mut a_sm, mut a_ix, mut a_es) = (dp(o.sm.device_ptr()), dp(o.ix.device_ptr()), dp(o.es.device_ptr()));
        let (mut a_og, mut a_od, mut a_gw, mut a_dw) = (dp(o.og.device_ptr()), dp(o.od.device_ptr()), gw, dw);
        let mut p: [*mut c_void; 17] = [&mut a_lg as *mut u64 as *mut _, &mut a_w as *mut u64 as *mut _,
            &mut a_x as *mut u64 as *mut _, &mut a_m as *mut i32 as *mut _, &mut a_n as *mut i32 as *mut _,
            &mut a_h as *mut i32 as *mut _, &mut a_cnt as *mut u64 as *mut _, &mut a_ids as *mut u64 as *mut _,
            &mut a_wts as *mut u64 as *mut _, &mut a_k as *mut i32 as *mut _, &mut a_sm as *mut u64 as *mut _,
            &mut a_ix as *mut u64 as *mut _, &mut a_es as *mut u64 as *mut _, &mut a_og as *mut u64 as *mut _,
            &mut a_od as *mut u64 as *mut _, &mut a_gw as *mut u64 as *mut _, &mut a_dw as *mut u64 as *mut _];
        if f == f_w4 {
            let smem = crate::exl3_forward::w4s_router_smem(m, kd) as u32;
            unsafe { go_b(f, g_w4, crate::exl3_forward::W4S_ROUTER_NT, smem, &mut p) }
        } else if f == f_c {
            unsafe { go(f, (ne / 4) as u32, crate::exl3_forward::RFC_SMEM, &mut p) }
        } else {
            unsafe { go(f, (ne / 4) as u32, 0, &mut p) }
        }
    };
    // bitwise diff count of b vs the reference a (rows < m); also returns the reference esel.
    let cmp = |a: &Out, b: &Out, m: usize| -> Result<(usize, i32, i32)> {
        dev.synchronize()?;
        let (la, lb) = (dev.dtoh_sync_copy(&a.lg)?, dev.dtoh_sync_copy(&b.lg)?);
        let (ia, ib) = (dev.dtoh_sync_copy(&a.ids)?, dev.dtoh_sync_copy(&b.ids)?);
        let (wa, wb) = (dev.dtoh_sync_copy(&a.wts)?, dev.dtoh_sync_copy(&b.wts)?);
        let (sa, sb) = (dev.dtoh_sync_copy(&a.sm)?, dev.dtoh_sync_copy(&b.sm)?);
        let (xa, xb) = (dev.dtoh_sync_copy(&a.ix)?, dev.dtoh_sync_copy(&b.ix)?);
        let (ea, eb) = (dev.dtoh_sync_copy(&a.es)?[0], dev.dtoh_sync_copy(&b.es)?[0]);
        let (ga, gb) = (dev.dtoh_sync_copy(&a.og)?, dev.dtoh_sync_copy(&b.og)?);
        let (da, db) = (dev.dtoh_sync_copy(&a.od)?, dev.dtoh_sync_copy(&b.od)?);
        let cnt_b = dev.dtoh_sync_copy(&b.cnt)?[0];
        let ns = (ea.max(0) as usize).min(mmax * k);
        let mut d = (0..m * ne).filter(|&i| la[i].to_bits() != lb[i].to_bits()).count();
        d += (0..m * k).filter(|&i| ia[i] != ib[i] || wa[i].to_bits() != wb[i].to_bits()).count();
        d += (sa != sb) as usize + (ea != eb) as usize + (xa[..ns] != xb[..ns]) as usize
            + (ga[..ns] != gb[..ns]) as usize + (da[..ns] != db[..ns]) as usize;
        Ok((d, ea, cnt_b))
    };
    let w_rand = synth_x(ne, kd, 0x5A3);
    let mut w_tie = w_rand.clone();          // expert e = expert e % 37: 13-14 bit-identical logits each
    for e in 37..ne { let (src, dst) = ((e % 37) * kd, e * kd); w_tie.copy_within(src..src + kd, dst); }
    let (a, b, c, e4) = (mk()?, mk()?, mk()?, mk()?);
    let mut fails = 0usize;
    for (case, wv) in [("rand", &w_rand), ("ties", &w_tie), ("nan-row", &w_rand)] {
        let d_w = dev.htod_sync_copy(wv)?;
        for m in 1..=mmax {
            let mut xv = synth_x(m, kd, 0xB0 ^ m as u32);
            if case == "nan-row" { for v in &mut xv[(m - 1) * kd..] { *v = 0x7E00; } }
            let d_x = dev.htod_sync_copy(&xv)?;
            let (wp, xp) = (dp(d_w.device_ptr()), dp(d_x.device_ptr()));
            pair(&a, wp, xp, m)?;
            for rep in 0..2 {
                fold(f_fold, &b, wp, xp, m)?;
                fold(f_fold0, &c, wp, xp, m)?;
                fold(f_w4, &e4, wp, xp, m)?;
                let (d1, es, c1) = cmp(&a, &b, m)?;
                let (d2, _, c2) = cmp(&a, &c, m)?;
                let (d3, _, c3) = cmp(&a, &e4, m)?;
                if es <= 0 || d1 + d2 + d3 > 0 || c1 != 0 || c2 != 0 || c3 != 0 {
                    fails += 1;
                    println!("    ROUTERFOLD FAIL {case} m={m} rep={rep}: fold {d1} diffs, fold_m0 {d2} diffs, \
                              fold_w4 {d3} diffs, esel {es}, counters {c1}/{c2}/{c3} (must be 0)");
                }
            }
        }
    }
    if fails > 0 { bail!("EXL3-ROUTERFOLD FAIL ({fails} cases)"); }
    println!("EXL3-ROUTERFOLD: PASS (xq_router_fold / _m0 / W4S _w4 (grid {g_w4} x {}) bitwise == xq_router_fused + \
              xq_router_topk_route: logits/ids/wts/route tables, M 1..=8, rand/ties/nan-row, x2 launches, counters re-armed)",
             crate::exl3_forward::W4S_ROUTER_NT);
    // ---- A5-K2: xq_router_fold_c (moe.router_coal) bitwise == xq_router_fold_w4 (the served fold) AND the
    // S-A3-z pair, every output NaN/garbage-poisoned before each launch, M 1..=8 (every width the fold
    // serves: m*topk <= 128, m <= 8; wider verifies take the unfused path), five data classes:
    // rand, ties37 (cross-lane ties), nan-row, ties32 (expert e = e % 32: every lane's 16 probs tie —
    // the lane-tree tie order), peaked (x * 300: most probs underflow to +0, top-k picks +0 by id asc).
    {
        let poison = |o: &Out| -> Result<()> {
            use cudarc::driver::sys;
            let fill = |p: u64, n: usize, v: u32| -> Result<()> {
                let r = unsafe { sys::cuMemsetD32_v2(p, v, n) };
                anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "poison memset ({r:?})");
                Ok(())
            };
            fill(dp(o.lg.device_ptr()), mmax * ne, 0x7FC0_0001)?;
            fill(dp(o.ids.device_ptr()), mmax * k, 0xA5A5_A5A5)?;
            fill(dp(o.wts.device_ptr()), mmax * k, 0x7FC0_0002)?;
            fill(dp(o.sm.device_ptr()), ne, 0xA5A5_A5A5)?;
            fill(dp(o.ix.device_ptr()), mmax * k, 0xA5A5_A5A5)?;
            fill(dp(o.es.device_ptr()), 1, 0xA5A5_A5A5)?;
            fill(dp(o.og.device_ptr()), mmax * k * 2, 0xA5A5_A5A5)?;
            fill(dp(o.od.device_ptr()), mmax * k * 2, 0xA5A5_A5A5)?;
            dev.synchronize()?;
            Ok(())
        };
        let mut w_t32 = w_rand.clone();
        for e in 32..ne { let (src, dst) = ((e % 32) * kd, e * kd); w_t32.copy_within(src..src + kd, dst); }
        let (r4, cc) = (mk()?, mk()?);
        let mut fails = 0usize;
        let mut ncase = 0usize;
        for (case, wv) in [("rand", &w_rand), ("ties37", &w_tie), ("nan-row", &w_rand), ("ties32", &w_t32), ("peaked", &w_rand)] {
            let d_w = dev.htod_sync_copy(wv)?;
            for m in 1..=mmax {
                let mut xv = synth_x(m, kd, 0xD0 ^ m as u32);
                if case == "nan-row" { for v in &mut xv[(m - 1) * kd..] { *v = 0x7E00; } }
                if case == "peaked" {
                    for v in &mut xv { *v = f16::from_f32(f16::from_bits(*v).to_f32() * 300.0).to_bits(); }
                }
                let d_x = dev.htod_sync_copy(&xv)?;
                let (wp, xp) = (dp(d_w.device_ptr()), dp(d_x.device_ptr()));
                pair(&a, wp, xp, m)?;
                poison(&r4)?;
                fold(f_w4, &r4, wp, xp, m)?;
                for rep in 0..2 {
                    poison(&cc)?;
                    fold(f_c, &cc, wp, xp, m)?;
                    let (d1, es, c1) = cmp(&r4, &cc, m)?;
                    let (d2, _, _) = cmp(&a, &cc, m)?;
                    ncase += 1;
                    if es <= 0 || d1 + d2 > 0 || c1 != 0 {
                        fails += 1;
                        println!("    ROUTERFOLD-C FAIL {case} m={m} rep={rep}: vs fold_w4 {d1} diffs, vs pair {d2} diffs, \
                                  esel {es}, counter {c1} (must be 0)");
                    }
                }
            }
        }
        if fails > 0 { bail!("EXL3-ROUTERFOLD-C FAIL ({fails}/{ncase} cases)"); }
        println!("EXL3-ROUTERFOLD-C: PASS (A5-K2 xq_router_fold_c (grid {} x 256, smem {}) bitwise == xq_router_fold_w4 == \
                  xq_router_fused + xq_router_topk_route: logits/ids/wts/slotmap/esel/idxmap/offs, M 1..=8, \
                  rand/ties37/nan-row/ties32/peaked, NaN-poisoned outputs, x2 launches, counter re-armed; {ncase} cases)",
                 ne / 4, crate::exl3_forward::RFC_SMEM);
    }
    // ---- standalone timing: weights rotated over 16 copies (42 MB > L2) so they stream from DRAM ----
    let nw = 16usize;
    let wbig: Vec<u16> = (0..nw).flat_map(|i| synth_x(ne, kd, 0x100 + i as u32)).collect();
    let d_wb = dev.htod_sync_copy(&wbig)?;
    let wb0 = dp(d_wb.device_ptr());
    let wstride = (ne * kd * 2) as u64;
    for m in [1usize, 2, 4, 6, 8] {
        let d_x = dev.htod_sync_copy(&synth_x(m, kd, 0xC0 ^ m as u32))?;
        let xp = dp(d_x.device_ptr());
        let mut us = [0f64; 5];
        for (which, slot) in us.iter_mut().enumerate() {
            let run = |it: usize| -> Result<()> {
                let wp = wb0 + (it % nw) as u64 * wstride;
                match which { 0 => pair(&a, wp, xp, m), 1 => fold(f_fold, &b, wp, xp, m), 2 => fold(f_fold0, &c, wp, xp, m),
                              3 => fold(f_w4, &e4, wp, xp, m), _ => fold(f_c, &e4, wp, xp, m) }
            };
            for it in 0..nw { run(it)?; }
            dev.synchronize()?;
            let iters = 30 * nw;
            let t = std::time::Instant::now();
            for it in 0..iters { run(it)?; }
            dev.synchronize()?;
            *slot = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        }
        println!("    EXL3-ROUTERFOLD timing (standalone, harness-only) m={m}: pair {:.2} us/layer, fold {:.2} ({:+.2}), \
                  fold_m0 {:.2} ({:+.2}), W4S fold_w4 {:.2} ({:+.2} vs fold), K2 fold_c {:.2} ({:+.2} vs fold_w4)", us[0], us[1],
                 us[1] - us[0], us[2], us[2] - us[0], us[3], us[3] - us[1], us[4], us[4] - us[3]);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// K9 (REL v0.7.3 review): the duplicated short banner that used to sit here was
// removed — the detailed block below is the one that stays.
// S-B9-CF-R16 (PLAN/CF_R16_REPORT.md): the r16 twins at 9..16 rows vs the paths
// today serves there, bitwise.
//   EXL3-ROUTERFOLD-R16: xq_router_fold_c_r16 vs TWO references:
//     (a) the LEGACY chain (xq_router_ks + combine + xq_router_topk +
//         xq_moe_route) -- what moe_inner launches above 8 rows today (the fused
//         pair xq_router_fused + xq_router_topk_route is M <= 8 hardware: its
//         GEMV covers 8 rows, so it is NOT a reference at m > 8);
//     (b) the m=1 fold (xq_router_fold_c) on the last row alone -- the r16 tail's
//         per-row chain must be the m<=8 tail's, bit for bit.
//   EXL3-HC-R16: xq_hc_fuse_i8_rbk_r16 == the m-grid xq_hc_fuse_i8k.
// Both are synthetic and model-free, and they poison every output before each
// launch, so a kernel that skips a row or a table cannot pass on its leftovers.
// ---------------------------------------------------------------------------
fn probe_r16(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    use cudarc::driver::sys;
    use std::ffi::c_void;
    let raw = |n: &str| crate::exl3_forward::xq_raw_fn(n)
        .ok_or_else(|| anyhow::anyhow!("{n} raw fn unavailable (stale PTX?)"));
    let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
    let memset = |ptr: u64, n: usize, v: u32| -> Result<()> {
        let r = unsafe { sys::cuMemsetD32_v2(ptr, v, n) };
        anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "probe memset ({r:?})");
        Ok(())
    };
    // cuLaunchKernel marshals every kernel param through a POINTER TO ITS VALUE.
    // Each array entry below is &mut to an owned KA field — a device address
    // placed in the array directly would be dereferenced as a host pointer (the
    // segfault the first version of this probe hit inside libcuda).
    let go = |f: sys::CUfunction, grid: (u32, u32, u32), smem: u32, args: &mut [*mut c_void]| -> Result<()> {
        let r = unsafe { sys::cuLaunchKernel(f, grid.0, grid.1, grid.2, 256, 1, 1, smem, std::ptr::null_mut(),
                                             args.as_mut_ptr(), std::ptr::null_mut()) };
        anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "r16 probe launch ({r:?})");
        dev.synchronize()?;           // a device fault lands at ITS launch, not the next sync
        Ok(())
    };
    #[derive(Default)] #[allow(non_snake_case)]
    struct KA {   // one field per kernel param this probe passes (by value or device ptr)
        out: u64, w: u64, x: u64, m: i32, N: i32, kd: i32, cnt: u64, ids: u64, wts: u64,
        k: i32, sm: u64, ix: u64, es: u64, og: u64, od: u64, gw: u64, dw: u64,
        ws: u64, ks_: i32, lr: i32, lr_m: i32, rw_: i32, fl: i32, hcn: f32,
        dd: u64, uu: u64, hn: u64, qd: u64, sdw: u64, qu: u64, su: u64, bar: u64,
    }
    macro_rules! kargs { ($ka:ident, $($f:ident),+) => { [ $(&mut $ka.$f as *mut _ as *mut c_void),+ ] } }
    // ============================================================== router
    {
        let f_c = raw("xq_router_fold_c")?;      // the m<=8 fold (per-row chain reference)
        let f_r16 = raw("xq_router_fold_c_r16")?;
        let (f_ks, f_cb) = (raw("xq_router_ks")?, raw("xq_router_combine")?);
        let (f_tk, f_rt) = (raw("xq_router_topk")?, raw("xq_moe_route")?);
        // the r16 fold opts into its dynamic smem exactly as the launcher does
        anyhow::ensure!(
            crate::exl3_forward::r16_smem_optin(dev, "xq_router_fold_c_r16",
                                               crate::exl3_forward::RFC_SMEM_R16 as i32),
            "EXL3-ROUTERFOLD-R16 FAIL: xq_router_fold_c_r16 smem opt-in refused");
        let (ne, k, kd) = (512usize, 10usize, 2560usize);
        let (ks, mmax) = (64i32, 16usize);
        let (gw, dw) = (0x0012_3457u64, 0x9abcu64);
        struct Out { lg: CudaSlice<f32>, ids: CudaSlice<i32>, wts: CudaSlice<f32>, sm: CudaSlice<i32>,
                     ix: CudaSlice<i32>, es: CudaSlice<i32>, og: CudaSlice<u64>, od: CudaSlice<u64>,
                     cnt: CudaSlice<i32> }
        let mk = || -> Result<Out> {
            Ok(Out { lg: dev.htod_sync_copy(&vec![0f32; mmax * ne])?,
                     ids: dev.htod_sync_copy(&vec![-7i32; mmax * k])?,
                     wts: dev.htod_sync_copy(&vec![0f32; mmax * k])?,
                     sm: dev.htod_sync_copy(&vec![-9i32; ne])?,
                     ix: dev.htod_sync_copy(&vec![-5i32; mmax * k])?,
                     es: dev.htod_sync_copy(&[-1i32])?,
                     og: dev.htod_sync_copy(&vec![0u64; mmax * k])?,
                     od: dev.htod_sync_copy(&vec![0u64; mmax * k])?,
                     cnt: dev.htod_sync_copy(&[0i32])? })
        };
        let poison = |o: &Out| -> Result<()> {
            memset(dp(o.lg.device_ptr()), mmax * ne, 0x7FC0_0001)?;
            memset(dp(o.ids.device_ptr()), mmax * k, 0xA5A5_A5A5)?;
            memset(dp(o.wts.device_ptr()), mmax * k, 0x7FC0_0002)?;
            memset(dp(o.sm.device_ptr()), ne, 0xA5A5_A5A5)?;
            memset(dp(o.ix.device_ptr()), mmax * k, 0xA5A5_A5A5)?;
            memset(dp(o.es.device_ptr()), 1, 0xA5A5_A5A5)?;
            memset(dp(o.og.device_ptr()), mmax * k, 0xA5A5_A5A5)?;
            memset(dp(o.od.device_ptr()), mmax * k, 0xA5A5_A5A5)?;
            dev.synchronize()?; Ok(())
        };
        // the LEGACY chain: what moe_inner launches above 8 rows today (xq_router_ks is
        // M-agnostic: row = blockIdx.y, global partials) — the reference at m > 8. The
        // fused pair (xq_router_fused + xq_router_topk_route) is M <= 8 hardware (its
        // GEMV covers 8 rows; rows 9..16 would be silently unserved) — NOT a reference.
        let legacy = |o: &Out, w: u64, x: u64, m: usize| -> Result<()> {
            let ws = dev.htod_sync_copy(&vec![0f32; ks as usize * mmax * ne])?;
            let mut ka = KA { ws: dp(ws.device_ptr()), w, x, m: m as i32, N: ne as i32,
                              kd: kd as i32, ks_: ks, out: dp(o.lg.device_ptr()),
                              ids: dp(o.ids.device_ptr()), wts: dp(o.wts.device_ptr()),
                              k: k as i32, sm: dp(o.sm.device_ptr()), ix: dp(o.ix.device_ptr()),
                              es: dp(o.es.device_ptr()), og: dp(o.og.device_ptr()),
                              od: dp(o.od.device_ptr()), gw, dw, ..Default::default() };
            let mut p1 = kargs!(ka, ws, w, x, m, N, kd, ks_);
            go(f_ks, (((ne + 255) / 256) as u32, m as u32, ks as u32), 0, &mut p1)?;
            let mut p2 = kargs!(ka, ws, out, m, N, ks_);
            go(f_cb, (((m * ne + 255) / 256) as u32, 1, 1), 0, &mut p2)?;
            let mut p3 = kargs!(ka, ids, wts, out, N, k, m);
            go(f_tk, (m as u32, 1, 1), ((256 + ne + 512) * 4) as u32, &mut p3)?;
            let mut p4 = kargs!(ka, ids, m, k, N, sm, ix, es, og, od, gw, dw);
            go(f_rt, (1, 1, 1), 0, &mut p4)
        };
        // the r16 fold, and the m<=8 fold on one row (the same code shape as its tail)
        let fold16 = |o: &Out, w: u64, x: u64, m: usize| -> Result<()> {
            let mut ka = KA { out: dp(o.lg.device_ptr()), w, x, m: m as i32, N: ne as i32,
                              kd: kd as i32, cnt: dp(o.cnt.device_ptr()), ids: dp(o.ids.device_ptr()),
                              wts: dp(o.wts.device_ptr()), k: k as i32, sm: dp(o.sm.device_ptr()),
                              ix: dp(o.ix.device_ptr()), es: dp(o.es.device_ptr()),
                              og: dp(o.og.device_ptr()), od: dp(o.od.device_ptr()), gw, dw,
                              ..Default::default() };
            let mut p = kargs!(ka, out, w, x, m, N, kd, cnt, ids, wts, k, sm, ix, es, og, od, gw, dw);
            go(f_r16, ((ne / 4) as u32, 1, 1), crate::exl3_forward::RFC_SMEM_R16, &mut p)
        };
        let fold1 = |o: &Out, w: u64, x: u64| -> Result<()> {
            let mut ka = KA { out: dp(o.lg.device_ptr()), w, x, m: 1i32, N: ne as i32,
                              kd: kd as i32, cnt: dp(o.cnt.device_ptr()), ids: dp(o.ids.device_ptr()),
                              wts: dp(o.wts.device_ptr()), k: k as i32, sm: dp(o.sm.device_ptr()),
                              ix: dp(o.ix.device_ptr()), es: dp(o.es.device_ptr()),
                              og: dp(o.og.device_ptr()), od: dp(o.od.device_ptr()), gw, dw,
                              ..Default::default() };
            let mut p = kargs!(ka, out, w, x, m, N, kd, cnt, ids, wts, k, sm, ix, es, og, od, gw, dw);
            go(f_c, ((ne / 4) as u32, 1, 1), crate::exl3_forward::RFC_SMEM, &mut p)
        };
        // per-buffer diff: logits / ids / wts / slotmap / esel / idxmap / offs_gu / offs_d
        let cmp = |a: &Out, b: &Out, m: usize| -> Result<([usize; 8], i32, i32)> {
            dev.synchronize()?;
            let (la, lb) = (dev.dtoh_sync_copy(&a.lg)?, dev.dtoh_sync_copy(&b.lg)?);
            let (ia, ib) = (dev.dtoh_sync_copy(&a.ids)?, dev.dtoh_sync_copy(&b.ids)?);
            let (wa, wb) = (dev.dtoh_sync_copy(&a.wts)?, dev.dtoh_sync_copy(&b.wts)?);
            let (sa, sb) = (dev.dtoh_sync_copy(&a.sm)?, dev.dtoh_sync_copy(&b.sm)?);
            let (xa, xb) = (dev.dtoh_sync_copy(&a.ix)?, dev.dtoh_sync_copy(&b.ix)?);
            let (ea, eb) = (dev.dtoh_sync_copy(&a.es)?[0], dev.dtoh_sync_copy(&b.es)?[0]);
            let (ga, gb) = (dev.dtoh_sync_copy(&a.og)?, dev.dtoh_sync_copy(&b.og)?);
            let (da, db) = (dev.dtoh_sync_copy(&a.od)?, dev.dtoh_sync_copy(&b.od)?);
            let (nl, nk) = (m * ne, m * k);
            let ns = ea.max(0) as usize;
            let v = [(0..nl).filter(|&i| la[i].to_bits() != lb[i].to_bits()).count(),
                     (0..nk).filter(|&i| ia[i] != ib[i]).count(),
                     (0..nk).filter(|&i| wa[i].to_bits() != wb[i].to_bits()).count(),
                     (sa != sb) as usize, (ea != eb) as usize,
                     (0..ns).filter(|&i| xa[i] != xb[i]).count(),
                     (0..ns).filter(|&i| ga[i] != gb[i]).count(),
                     (0..ns).filter(|&i| da[i] != db[i]).count()];
            let c = dev.dtoh_sync_copy(&a.cnt)?[0] + dev.dtoh_sync_copy(&b.cnt)?[0];
            Ok((v, ea, c))
        };
        let sx = |m: usize, seed: u32| -> Vec<u16> {
            (0..m * kd).map(|i| { let z = (i as u32) ^ seed;
                                  (z.wrapping_mul(1664525).wrapping_add(1013904223) >> 13) as u16 & 0x3bff })
                 .collect()
        };
        let (b, rf) = (mk()?, mk()?);
        let (mut ncase, mut fpass, mut rowdiff) = (0usize, 0usize, 0usize);
        let w_rand: Vec<u16> = (0..ne * kd)
            .map(|i| ((i as u32).wrapping_mul(2654435761).wrapping_add(7) >> 17) as u16 & 0x3bff)
            .collect();
        let mut w_tie = w_rand.clone();
        for e in 37..ne { let (src, dst) = ((e % 37) * kd, e * kd); w_tie.copy_within(src..src + kd, dst); }
        for (case, wv) in [("rand", &w_rand), ("ties37", &w_tie), ("nan-row", &w_rand)] {
            let d_w = dev.htod_sync_copy(wv)?;
            let wp = dp(d_w.device_ptr());
            for m in 9..=mmax {
                let mut xv = sx(m, 0xD0 ^ m as u32);
                if case == "nan-row" { for v in &mut xv[(m - 1) * kd..] { *v = 0x7E00; } }
                let d_x = dev.htod_sync_copy(&xv)?;
                let xp = dp(d_x.device_ptr());
                // reference: the LEGACY chain, at this width
                poison(&rf)?; legacy(&rf, wp, xp, m)?;
                // the r16 fold (x2: the counter must re-arm)
                poison(&b)?;
                for rep in 0..2 {
                    poison(&b)?; fold16(&b, wp, xp, m)?;
                    ncase += 1;
                    let (vl, esl, cl) = cmp(&rf, &b, m)?;
                    let dl: usize = vl.iter().sum();
                    // the r16 tail's per-row chain is the SAME code as the m<=8 tail:
                    // run that fold on this row alone and require the ids/wts to match
                    // bit for bit.
                    let d_x1 = dev.htod_sync_copy(&xv[(m - 1) * kd..m * kd])?;
                    let c1 = mk()?;
                    poison(&c1)?; fold1(&c1, wp, dp(d_x1.device_ptr()))?;
                    let (i1, i16) = (dev.dtoh_sync_copy(&c1.ids)?, dev.dtoh_sync_copy(&b.ids)?);
                    let (w1, w16) = (dev.dtoh_sync_copy(&c1.wts)?, dev.dtoh_sync_copy(&b.wts)?);
                    let n1 = (0..k).filter(|&j| i1[j] != i16[(m - 1) * k + j]).count()
                           + (0..k).filter(|&j| w1[j].to_bits() != w16[(m - 1) * k + j].to_bits()).count();
                    if dl == 0 && esl > 0 && cl == 0 { fpass += 1; }
                    if n1 > 0 { rowdiff += 1; }
                    println!("    ROUTERFOLD-R16 {case} m={m} rep={rep}: vs the LEGACY \
                              xq_router_ks+combine+topk+moe_route chain {dl} diffs {vl:?} \
                              (esel {esl}, counter {cl}); row {} vs the m=1 fold: {n1} id/wt diffs", m - 1);
                }
            }
        }
        if rowdiff > 0 { bail!("EXL3-ROUTERFOLD-R16 FAIL ({rowdiff} cases differ from the m=1 fold)"); }
        if fpass != ncase {
            bail!("EXL3-ROUTERFOLD-R16 FAIL ({fpass}/{ncase} bitwise vs the legacy chain)");
        }
        println!("EXL3-ROUTERFOLD-R16: PASS (xq_router_fold_c_r16 (grid {} x 256, smem {}) bitwise == the \
                  LEGACY xq_router_ks+combine+topk+moe_route chain (today's >8-row path) on \
                  logits/ids/wts/slotmap/esel/idxmap/offs_gu/offs_d, and its ids/wts bitwise == the \
                  m=1 fold's on every row, M 9..=16, rand/ties37/nan-row, every output poisoned \
                  before each launch, x2 launches, counter re-armed; {ncase} cases)",
                  ne / 4, crate::exl3_forward::RFC_SMEM_R16);
    }
    // ================================================================== HC
    {
        let f_r16 = raw("xq_hc_fuse_i8_rbk_r16")?;
        let f_k = raw("xq_hc_fuse_i8k")?;
        // the shipped shape: h = 2048, hc = 4, lr = 512 (rw = 8192, rw % 1024 == 0)
        let (h, hc, lr) = (2048usize, 4usize, 512usize);
        let (rw, mmax) = (h * hc, 16usize);
        let srand = |n: usize, seed: u32, lim: u32| -> Vec<u8> {
            (0..n).map(|i| { let z = (i as u32) ^ seed;
                             ((z.wrapping_mul(1664525).wrapping_add(1013904223) >> 11) % lim) as u8 })
                 .collect()
        };
        let d_hn = dev.htod_sync_copy(&srand(mmax * rw * 2, 1, 97))?;
        let d_qd = dev.htod_sync_copy(&srand(lr * rw, 2, 23))?;
        let d_sdw = dev.htod_sync_copy(&(0..lr).map(|i| 0.01f32 + (i % 7) as f32 * 0.003).collect::<Vec<f32>>())?;
        let d_qu = dev.htod_sync_copy(&srand(lr * rw, 3, 23))?;
        let d_su = dev.htod_sync_copy(&(0..rw).map(|i| 0.01f32 + (i % 5) as f32 * 0.002).collect::<Vec<f32>>())?;
        let d_bar = dev.htod_sync_copy(&[0i32])?;
        let mut d_dd = dev.htod_sync_copy(&vec![0u8; mmax * lr * 2])?;
        let mut d_uu = dev.htod_sync_copy(&vec![0u8; mmax * rw * 2])?;
        // 64 KB of static smem => the grid comes from THIS function's measured
        // co-residency, never more CTAs than fit (the grid barrier must complete).
        let cap = crate::exl3_forward::wp11_capacity_raw_dsmem_pub(dev, "xq_hc_fuse_i8_rbk_r16", 65536);
        let capk = crate::exl3_forward::wp11_capacity_pub(dev, "xq_hc_fuse_i8k");
        anyhow::ensure!(cap >= 4 && capk >= 1,
                        "EXL3-HC-R16 FAIL: not co-resident (rbk_r16 {cap}, i8k {capk})");
        let bx = (cap * 5 / 6).clamp(4, 40);
        let bxk = (capk * 5 / 6).clamp(1, 40);
        let (mut fails, mut ncase) = (0usize, 0usize);
        for m in 9..=mmax {
            // reference: the m-grid twin today's >8-row path runs
            memset(dp(d_dd.device_ptr()), mmax * lr * 2 / 4, 0xFFFF_FFFF)?;
            memset(dp(d_uu.device_ptr()), mmax * rw * 2 / 4, 0xFFFF_FFFF)?;
            {
                let mut ka = KA { dd: dp(d_dd.device_ptr()), uu: dp(d_uu.device_ptr()),
                                  hn: dp(d_hn.device_ptr()), qd: dp(d_qd.device_ptr()),
                                  sdw: dp(d_sdw.device_ptr()), qu: dp(d_qu.device_ptr()),
                                  su: dp(d_su.device_ptr()), bar: dp(d_bar.device_ptr()),
                                  hcn: hc as f32, lr: lr as i32, rw_: rw as i32, fl: 0i32,
                                  ..Default::default() };
                let mut q = kargs!(ka, dd, uu, hn, qd, sdw, qu, su, bar, hcn, lr, rw_, fl);
                let g = (bxk / m as u32).clamp(1, 40);
                go(f_k, (g, m as u32, 1), 0, &mut q)?;
            }
            let ra = dev.dtoh_sync_copy(&d_dd)?;
            let rb = dev.dtoh_sync_copy(&d_uu)?;
            for r4 in [0i32, 16] {
                memset(dp(d_dd.device_ptr()), mmax * lr * 2 / 4, 0xFFFF_FFFF)?;
                memset(dp(d_uu.device_ptr()), mmax * rw * 2 / 4, 0xFFFF_FFFF)?;
                let mut ka = KA { dd: dp(d_dd.device_ptr()), uu: dp(d_uu.device_ptr()),
                                  hn: dp(d_hn.device_ptr()), qd: dp(d_qd.device_ptr()),
                                  sdw: dp(d_sdw.device_ptr()), qu: dp(d_qu.device_ptr()),
                                  su: dp(d_su.device_ptr()), bar: dp(d_bar.device_ptr()),
                                  hcn: hc as f32, lr_m: (lr as i32) | ((m as i32) << 16),
                                  rw_: rw as i32, fl: r4, ..Default::default() };
                let mut q = kargs!(ka, dd, uu, hn, qd, sdw, qu, su, bar, hcn, lr_m, rw_, fl);
                let dsmem = (m * if r4 != 0 { 4096 } else { lr * 4 }) as u32;
                go(f_r16, (bx, 1, 1), dsmem, &mut q)?;
                let na = dev.dtoh_sync_copy(&d_dd)?;
                let nb = dev.dtoh_sync_copy(&d_uu)?;
                let (nu, nv) = (m * lr * 2, m * rw * 2);   // dd/uu are f16: byte counts
                let ddd = (0..nu).filter(|&i| na[i] != ra[i]).count();
                let duu = (0..nv).filter(|&i| nb[i] != rb[i]).count();
                ncase += 1;
                if ddd + duu > 0 {
                    fails += 1;
                    println!("    HC-R16 FAIL m={m} r4={r4}: dd {ddd}/{nu} uu {duu}/{nv} bit-diffs vs xq_hc_fuse_i8k");
                }
            }
        }
        if fails > 0 { bail!("EXL3-HC-R16 FAIL ({fails}/{ncase} cases)"); }
        println!("EXL3-HC-R16: PASS (xq_hc_fuse_i8_rbk_r16 (grid {bx} x 256, dynamic smem m*4096/m*lr*4, \
                  co-residency {cap} at 64 KB) bitwise == xq_hc_fuse_i8k on dd + uu, M 9..=16, R4 off + on, \
                  every output poisoned before each launch; {ncase} cases)");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// WP09: xq_argmax_rows (one launch, nb x m blocks, last-block reduce) must return the SAME ids
// as m single-block xq_argmax launches AND the host fold (fp32 compare, lowest id on ties,
// init (-inf, 0), NaN skipped) — adversarial rows: unique max, cross-block ties, all -inf,
// all NaN, NaN-sprinkled, +-0 max ties, +inf ties, constant rows; aligned (V % 8 == 0) and
// misaligned (odd V -> scalar path) row strides; empty slices (tiny V); nb in {1,7,16,32};
// a start0 offset; two back-to-back launches on the same counters (the re-arm).
// ---------------------------------------------------------------------------
fn probe_argmax_rows(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    let f_old = dev.get_func(MODULE, "xq_argmax").context("xq_argmax")?;
    let f_new = dev.get_func(MODULE, "xq_argmax_rows").context("xq_argmax_rows")?;
    let h = |x: f32| f16::from_f32(x).to_bits();
    let (ninf, pinf, nan) = (0xFC00u16, 0x7C00u16, 0x7E00u16);
    let host = |row: &[u16]| -> i32 {
        let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
        for (i, &b) in row.iter().enumerate() {
            let v = f16::from_bits(b).to_f32();
            if v > best || (v == best && i < bi) { best = v; bi = i; }
        }
        bi as i32
    };
    let mut cases = 0usize;
    for &v in &[248_320usize, 248_323, 1000, 37] {
        let m = 16usize;
        let mut lg = synth_x(m, v, 0xA46 ^ v as u32);
        for r in 0..m {
            let row = &mut lg[r * v..(r + 1) * v];
            match r % 8 {
                0 => { row[(v * 7) / 11] = h(60000.0); }                        // unique max
                1 => { for &i in &[v - 1, v / 2, 3usize.min(v - 1), v / 3] { row[i] = h(100.0); } } // cross-block ties
                2 => { row.iter_mut().for_each(|b| *b = ninf); }                 // all -inf -> 0
                3 => { row.iter_mut().for_each(|b| *b = nan); }                  // all NaN -> 0
                4 => { for i in (0..v).step_by(3) { row[i] = nan; } row[v - 2] = h(50.0); row[v / 7] = nan; }
                // max = 0 with -0 at the LOWEST id: the canonical-+0 key must keep the id order
                5 => { row.iter_mut().for_each(|b| *b = h(-1.0)); row[v - 1] = 0x0000; row[(v / 2) | 1] = 0x0000; row[(v / 3) | 1] = 0x8000; }
                6 => { row[v / 4] = pinf; row[v - 3] = pinf; row[1usize.min(v - 1)] = nan; }
                _ => { row.iter_mut().for_each(|b| *b = if r == 7 { h(1.0) } else { ninf }); row[v - 1] = h(-65504.0); }
            }
        }
        let d_lg = dev.htod_sync_copy(&lg)?;
        // reference: m single-block xq_argmax launches
        let mut ids_old = dev.htod_sync_copy(&vec![-7i32; m])?;
        for i in 0..m {
            unsafe {
                f_old.clone().launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 1024 * 12 },
                                     (&mut ids_old, i as i32, &d_lg, (i * v) as i64, v as i32))?;
            }
        }
        dev.synchronize()?;
        let old = dev.dtoh_sync_copy(&ids_old)?;
        for r in 0..m {
            let want = host(&lg[r * v..(r + 1) * v]);
            if old[r] != want { bail!("EXL3-ARGMAX-ROWS FAIL: reference xq_argmax row {r} V={v}: {} vs host {want}", old[r]); }
        }
        let mut part = dev.alloc_zeros::<u64>(m * 32)?;
        let mut cnt = dev.htod_sync_copy(&vec![0u32; m])?;
        for &nb in &[16usize, 32, 7, 1] {
            for rep in 0..2 {
                let mut ids = dev.htod_sync_copy(&vec![-9i32; m])?;
                unsafe {
                    f_new.clone().launch(LaunchConfig { grid_dim: (nb as u32, m as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                         (&mut ids, 0i32, &d_lg, 0i64, v as i64, v as i32, &mut part, &mut cnt))?;
                }
                dev.synchronize()?;
                let got = dev.dtoh_sync_copy(&ids)?;
                if got != old {
                    let r = (0..m).find(|&r| got[r] != old[r]).unwrap_or(0);
                    bail!("EXL3-ARGMAX-ROWS FAIL: V={v} nb={nb} rep={rep} row {r}: {} vs xq_argmax {}", got[r], old[r]);
                }
                cases += 1;
            }
            let c = dev.dtoh_sync_copy(&cnt)?;
            if c.iter().any(|&x| x != 0) { bail!("EXL3-ARGMAX-ROWS FAIL: counters not re-armed (V={v} nb={nb}): {c:?}"); }
        }
        // start0 offset + lane0: rows 1..m into ids[1..m]
        {
            let mut ids = dev.htod_sync_copy(&vec![-9i32; m])?;
            unsafe {
                f_new.clone().launch(LaunchConfig { grid_dim: (16, (m - 1) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                     (&mut ids, 1i32, &d_lg, v as i64, v as i64, v as i32, &mut part, &mut cnt))?;
            }
            dev.synchronize()?;
            let got = dev.dtoh_sync_copy(&ids)?;
            if got[1..] != old[1..] { bail!("EXL3-ARGMAX-ROWS FAIL: V={v} start0/lane0 offset: {:?} vs {:?}", &got[1..], &old[1..]); }
            cases += 1;
        }
    }
    println!("EXL3-ARGMAX-ROWS: PASS (one-launch argmax ids == xq_argmax == host fold; {cases} launches x 16 adversarial rows, \
              V in {{248320, 248323, 1000, 37}}, nb in {{1,7,16,32}}, counters re-armed)");
    Ok(())
}

