//! TP=4 W=4 probes: tail GEMM, router GEMM and MoE glue.
//! Probe-only code of the `exl3_bench` module (see `mod.rs` for the map); shared helpers come from the parent.
//! Numerics contract (module header in `mod.rs`): the kernel decode is bitwise vs the host oracle; do not change reduction orders.

use super::*;

/// TP-4T1 (`--probe-exl3-tailgemm`): the prefill TAIL-chunk small-M plain fp16 GEMM `xq_gemm_f16_tail`
/// against the production reference `xq_gemm_f16` (per-row launch, one thread per column) at the served
/// tail shapes (hc down/up, GDN a/b, PLE key/value) plus ragged N/K, M in {1,2,3,5,8,11,16}. PASS requires
/// the outputs to be BITWISE equal (outputs pre-poisoned, reference non-trivially nonzero), then prints the
/// per-call time table (weights rotated through >= 128 MB of copies so the read is DRAM-cold, launches
/// back-to-back). Synthetic, no model. Flag only; no env.
pub fn probe_tailgemm() -> Result<()> {
    let dev = dev0()?;
    load_module(&dev)?;
    let f_old = dev.get_func(MODULE, "xq_gemm_f16").context("xq_gemm_f16")?;
    let f_new = dev.get_func(MODULE, "xq_gemm_f16_tail").context("xq_gemm_f16_tail")?;
    // (label, N, K)
    let shapes: [(&str, usize, usize); 7] = [
        ("hc_down", 320, 10240), ("hc_up", 10240, 320), ("a_or_b", 48, 2560),
        ("ple_key", 10240, 2560), ("ple_val", 2560, 2560),
        ("ragged_N300_K1040", 300, 1040), ("ragged_N33_K136", 33, 136),
    ];
    let ms = [1usize, 2, 3, 5, 8, 11, 16];
    let mut compared = 0usize;
    let mut table: Vec<String> = Vec::new();
    for &(label, n, k) in &shapes {
        let w = synth_x(n, k, 0xF16 ^ (n as u32));
        let d_w = dev.htod_sync_copy(&w)?;
        // cold-rotation copies of the weight (same bytes)
        let wbytes = n * k * 2;
        let ncopy = ((128usize << 20) / wbytes.max(1)).clamp(2, 64);
        let mut copies = Vec::with_capacity(ncopy);
        for _ in 0..ncopy { copies.push(dev.htod_sync_copy(&w)?); }
        for &m in &ms {
            let d_x = dev.htod_sync_copy(&synth_x(m, k, 0xA11 ^ (m as u32)))?;
            let mut o_old = dev.htod_sync_copy(&vec![0x7e01u16; m * n])?;
            let mut o_new = dev.htod_sync_copy(&vec![0x7e01u16; m * n])?;
            let g_old = LaunchConfig { grid_dim: (((n + 127) / 128) as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
            let g_new = LaunchConfig { grid_dim: (((n + 31) / 32) as u32, ((m + 3) / 4) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                f_old.clone().launch(g_old, (&mut o_old, &d_w, &d_x, m as i32, n as i32, k as i32))?;
                f_new.clone().launch(g_new, (&mut o_new, &d_w, &d_x, m as i32, n as i32, k as i32))?;
            }
            dev.synchronize()?;
            let a = dev.dtoh_sync_copy(&o_old)?;
            let b = dev.dtoh_sync_copy(&o_new)?;
            if a.iter().filter(|v| **v != 0 && **v != 0x7e01).count() < m * n / 2 {
                bail!("EXL3-TAILGEMM FAIL: implausible reference output ({label} m={m} N={n} K={k})");
            }
            let mism = a.iter().zip(b.iter()).filter(|(p, q)| p != q).count();
            if mism > 0 { bail!("EXL3-TAILGEMM FAIL: {label} m={m} N={n} K={k}: {mism}/{} bytes differ", m * n); }
            compared += m * n;
            // timing (cold rotation; back-to-back launches, per-call mean)
            let iters = (ncopy * 3).max(24);
            let mut times = [0f64; 2];
            for (slot, f, cfg) in [(0usize, &f_old, g_old), (1usize, &f_new, g_new)] {
                for c in copies.iter().take(2) {
                    unsafe { f.clone().launch(cfg, (&mut o_old, c, &d_x, m as i32, n as i32, k as i32))?; }
                }
                dev.synchronize()?;
                let t0 = std::time::Instant::now();
                for i in 0..iters {
                    unsafe { f.clone().launch(cfg, (&mut o_old, &copies[i % ncopy], &d_x, m as i32, n as i32, k as i32))?; }
                }
                dev.synchronize()?;
                times[slot] = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            }
            table.push(format!("  {label:<18} M={m:<2} old {:>8.1} us   tail {:>7.1} us   x{:.1}", times[0], times[1], times[0] / times[1]));
        }
    }
    println!("EXL3-TAILGEMM timing (per call, DRAM-cold rotation, back-to-back launches, includes ~launch gap):");
    for l in &table { println!("{l}"); }
    println!("EXL3-TAILGEMM: PASS (xq_gemm_f16_tail bitwise == xq_gemm_f16; {} shapes x {} M values, {} output elements compared)",
             shapes.len(), ms.len(), compared);
    Ok(())
}

/// TP-4R1 (`--probe-exl3-routergemm`): the register-tiled prefill router GEMMs `xq_gemm_f16_f32_tile128` / `_tile64`
/// (the pair `exl3_forward::router_tile_kernel` dispatches) against the production reference `xq_gemm_f16_f32`
/// (thread per (n, m), single ascending-k chain) and the kernel they replace, `xq_gemm_f16_f32_rows32`, at the served
/// router shape (ne = 512, K = 2560) over the served chunk widths plus a second (ne, K) shape. PASS requires the fp32
/// outputs BITWISE equal (new outputs pre-poisoned with a NaN pattern, reference non-trivially nonzero) and, for the
/// widest batches, `xq_router_topk` (top-10 ids + weights) identical on both logit sets. Then a per-call time table
/// (CUDA events around 10 back-to-back launches, median of 25 reps, L2-warm — a RANKING, not an in-round price,
/// AGENTS §4). Synthetic, no model. Flag only; no env.
pub fn probe_routergemm() -> Result<()> {
    use cudarc::driver::sys;
    let dev = dev0()?;
    load_module(&dev)?;
    let g = |n: &str| dev.get_func(MODULE, n).with_context(|| n.to_string());
    let (f_ref, f_r32, f_r8) = (g("xq_gemm_f16_f32")?, g("xq_gemm_f16_f32_rows32")?, g("xq_gemm_f16_f32_rows8")?);
    let (f_t128, f_t64, f_topk) = (g("xq_gemm_f16_f32_tile128")?, g("xq_gemm_f16_f32_tile64")?, g("xq_router_topk")?);
    let st = *dev.cu_stream();
    let mut e: [sys::CUevent; 2] = [std::ptr::null_mut(); 2];
    for ev in e.iter_mut() {
        let r = unsafe { sys::cuEventCreate(ev, 0) };
        anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuEventCreate {r:?}");
    }
    let cfg_ref = |n: usize, m: usize| LaunchConfig { grid_dim: (n.div_ceil(128) as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
    let cfg_r32 = |n: usize, m: usize| LaunchConfig { grid_dim: (n.div_ceil(128) as u32, ((m + 31) / 32) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
    let cfg_r8 = |n: usize, m: usize| LaunchConfig { grid_dim: ((n / 64) as u32, m.div_ceil(8) as u32, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 };
    let cfg_t = |n: usize, m: usize, bm: usize| LaunchConfig { grid_dim: ((n / 64) as u32, m.div_ceil(bm) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
    let poison = f32::from_bits(0x7fc0_dead);
    let mut compared = 0usize;
    let mut table: Vec<String> = Vec::new();
    let mut sp: Vec<(usize, f64)> = Vec::new();
    let timed_ms = [17usize, 72, 131, 299, 384, 512, 1021, 1536, 2048, 4085];
    let cases: [(usize, usize, Vec<usize>); 2] = [
        (512, 2560, vec![1, 2, 3, 17, 33, 72, 100, 131, 299, 384, 512, 1021, 1280, 1536, 2048, 4085]),
        (192, 144, vec![1, 5, 17, 100, 200, 300]),
    ];
    for (n, k, ms) in &cases {
        let (n, k) = (*n, *k);
        let d_w = dev.htod_sync_copy(&synth_x(n, k, 0x5A3 ^ n as u32))?;
        for &m in ms {
            let d_x = dev.htod_sync_copy(&synth_x(m, k, 0x77 ^ m as u32))?;
            let mut o_ref = dev.htod_sync_copy(&vec![0f32; m * n])?;
            let mut o_r32 = dev.htod_sync_copy(&vec![poison; m * n])?;
            let mut o_t128 = dev.htod_sync_copy(&vec![poison; m * n])?;
            let mut o_t64 = dev.htod_sync_copy(&vec![poison; m * n])?;
            unsafe {
                f_ref.clone().launch(cfg_ref(n, m), (&mut o_ref, &d_w, &d_x, m as i32, n as i32, k as i32))?;
                f_r32.clone().launch(cfg_r32(n, m), (&mut o_r32, &d_w, &d_x, m as i32, n as i32, k as i32))?;
                f_t128.clone().launch(cfg_t(n, m, 128), (&mut o_t128, &d_w, &d_x, m as i32, n as i32, k as i32))?;
                f_t64.clone().launch(cfg_t(n, m, 64), (&mut o_t64, &d_w, &d_x, m as i32, n as i32, k as i32))?;
            }
            dev.synchronize()?;
            let a = dev.dtoh_sync_copy(&o_ref)?;
            if a.iter().filter(|v| **v != 0.0 && v.is_finite()).count() < m * n * 95 / 100 {
                bail!("EXL3-ROUTERGEMM FAIL: implausible reference output (m={m} ne={n} K={k})");
            }
            let outs: Vec<(&str, Vec<f32>)> = vec![("tile128", dev.dtoh_sync_copy(&o_t128)?), ("tile64", dev.dtoh_sync_copy(&o_t64)?),
                                                   ("rows32", dev.dtoh_sync_copy(&o_r32)?)];
            for (name, b) in &outs {
                let mism = a.iter().zip(b.iter()).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
                if mism > 0 { bail!("EXL3-ROUTERGEMM FAIL: {name} m={m} ne={n} K={k}: {mism}/{} fp32 outputs differ from xq_gemm_f16_f32", m * n); }
                compared += m * n;
            }
            // top-10 ids + weights on the reference logits vs the dispatch-picked kernel's logits (served ne = 512 shape)
            if n == 512 && m >= 1021 {
                let topk = 10usize;
                let (kn, bm) = crate::exl3_forward::router_tile_kernel(m, n, k).context("router_tile_kernel None at the served shape")?;
                let fp = g(kn)?;
                let mut o_pick = dev.htod_sync_copy(&vec![poison; m * n])?;
                unsafe { fp.clone().launch(cfg_t(n, m, bm), (&mut o_pick, &d_w, &d_x, m as i32, n as i32, k as i32))?; }
                let smem = ((256 + n + 512) * 4) as u32;
                let tcfg = LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem };
                let mut ids_a = dev.htod_sync_copy(&vec![-1i32; m * topk])?;
                let mut ids_b = dev.htod_sync_copy(&vec![-1i32; m * topk])?;
                let mut w_a = dev.htod_sync_copy(&vec![poison; m * topk])?;
                let mut w_b = dev.htod_sync_copy(&vec![poison; m * topk])?;
                unsafe {
                    f_topk.clone().launch(tcfg, (&mut ids_a, &mut w_a, &o_ref, n as i32, topk as i32, m as i32))?;
                    f_topk.clone().launch(tcfg, (&mut ids_b, &mut w_b, &o_pick, n as i32, topk as i32, m as i32))?;
                }
                dev.synchronize()?;
                let (ia, ib) = (dev.dtoh_sync_copy(&ids_a)?, dev.dtoh_sync_copy(&ids_b)?);
                let (wa, wb) = (dev.dtoh_sync_copy(&w_a)?, dev.dtoh_sync_copy(&w_b)?);
                if ia.iter().any(|&v| v < 0 || v as usize >= n) { bail!("EXL3-ROUTERGEMM FAIL: implausible top-k ids (m={m})"); }
                if ia != ib || wa.iter().zip(wb.iter()).any(|(p, q)| p.to_bits() != q.to_bits()) {
                    bail!("EXL3-ROUTERGEMM FAIL: top-{topk} ids/weights differ at m={m} ({kn})");
                }
                println!("EXL3-ROUTERGEMM: m={m:>4}: top-{topk} ids + weights identical on {m} rows ({kn})");
            }
            // timing (served shape only): events around 10 back-to-back launches, median of 25
            if n == 512 && timed_ms.contains(&m) {
                let mut tm = |kind: u8| -> Result<f64> {
                    let mut o = dev.htod_sync_copy(&vec![0f32; m * n])?;
                    let mut reps = Vec::with_capacity(25);
                    for rep in 0..28 {
                        let r0 = unsafe { sys::cuEventRecord(e[0], st) };
                        anyhow::ensure!(r0 == sys::CUresult::CUDA_SUCCESS, "cuEventRecord {r0:?}");
                        for _ in 0..10 {
                            unsafe {
                                match kind {
                                    0 => f_r32.clone().launch(cfg_r32(n, m), (&mut o, &d_w, &d_x, m as i32, n as i32, k as i32))?,
                                    1 => f_r8.clone().launch(cfg_r8(n, m), (&mut o, &d_w, &d_x, m as i32, n as i32, k as i32))?,
                                    2 => f_t64.clone().launch(cfg_t(n, m, 64), (&mut o, &d_w, &d_x, m as i32, n as i32, k as i32))?,
                                    _ => f_t128.clone().launch(cfg_t(n, m, 128), (&mut o, &d_w, &d_x, m as i32, n as i32, k as i32))?,
                                }
                            }
                        }
                        let r1 = unsafe { sys::cuEventRecord(e[1], st) };
                        anyhow::ensure!(r1 == sys::CUresult::CUDA_SUCCESS, "cuEventRecord {r1:?}");
                        dev.synchronize()?;
                        let mut ms: f32 = 0.0;
                        let r = unsafe { sys::cuEventElapsedTime(&mut ms, e[0], e[1]) };
                        anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuEventElapsedTime {r:?}");
                        if rep >= 3 { reps.push(ms as f64 * 1000.0 / 10.0); }
                    }
                    reps.sort_by(|p, q| p.partial_cmp(q).unwrap());
                    Ok(reps[reps.len() / 2])
                };
                let (t32, t8, t64, t128) = (tm(0)?, tm(1)?, tm(2)?, tm(3)?);
                let (kn, _) = crate::exl3_forward::router_tile_kernel(m, n, k).unwrap();
                let tp = if kn.ends_with("tile64") { t64 } else { t128 };
                let r8_on = crate::exl3_forward::pfx1_router_rows8(m, n, k);
                let old = if r8_on { t8 } else { t32 };
                sp.push((m, old / tp));
                table.push(format!("  m={m:<5} rows32 {t32:>8.1} us   rows8 {t8:>8.1} us   tile64 {t64:>8.1} us   tile128 {t128:>8.1} us   \
                                    picked {} -> x{:.2} vs today's pick ({})", kn.trim_start_matches("xq_gemm_f16_f32_"), old / tp,
                                   if r8_on { "rows8" } else { "rows32" }));
            }
        }
    }
    for ev in e { unsafe { sys::cuEventDestroy_v2(ev); } }
    println!("EXL3-ROUTERGEMM timing (ne=512 K=2560, per call, CUDA events over 10 back-to-back launches, median of 25, L2-warm, standalone ranking):");
    for l in &table { println!("{l}"); }
    println!("EXL3-ROUTERGEMM: PASS (tile128/tile64 bitwise == xq_gemm_f16_f32 (+ rows32); {} fp32 outputs compared; top-10 identical on the wide batches)", compared);
    Ok(())
}

/// TP-4M1 (`--probe-exl3-moeglue`): the two W=4 MoE prefill glue kernels at the served shapes (synthetic, no model).
///  (1) `xq_had_suh_rows_live_u{1,2,4}` vs the old `xq_had_suh_rows` (32-thread blocks, all r rows): BITWISE on every row the
///      old kernel writes (live_end = null, several grid sizes, three chunk widths) and, with live_end set, on rows
///      [0, live) with the dead rows left UNTOUCHED (poison survives).
///  (2) `xq_moe_combine_rows_ep2_k{10,16}` vs the old `xq_moe_combine_rows_ep` (1024-thread block per row): BITWISE on every
///      fp32 output, add_shared 0/1, local-pick densities 0.25 / 1.0, plus a k = 16 case.
/// An empty/degenerate compare (reference mostly zero or non-finite) is a FAILURE. Then per-call timing at the served
/// chunk (c = 2043, k = 10, ne_l = 128): L2-warm (median of 25 over 10 back-to-back launches) and L2-cold (a 192 MB memset
/// before each single launch, median of 25), with the byte floor (238 GB/s) next to it. Standalone numbers RANK a kernel,
/// they do not price it in-round (AGENTS §4). Flag only; no env.
pub fn probe_moeglue() -> Result<()> {
    use cudarc::driver::sys;
    let dev = dev0()?;
    load_module(&dev)?;
    let g = |n: &str| dev.get_func(MODULE, n).with_context(|| n.to_string());
    let f_old_h = g("xq_had_suh_rows")?;
    let f_new_h = [g("xq_had_suh_rows_live_u1")?, g("xq_had_suh_rows_live_u2")?, g("xq_had_suh_rows_live_u4")?];
    let us_of = [1usize, 2, 4];
    let f_old_c = g("xq_moe_combine_rows_ep")?;
    let (f_c10, f_c16) = (g("xq_moe_combine_rows_ep2_k10")?, g("xq_moe_combine_rows_ep2_k16")?);
    let st = *dev.cu_stream();
    let mut e: [sys::CUevent; 2] = [std::ptr::null_mut(); 2];
    for ev in e.iter_mut() {
        let r = unsafe { sys::cuEventCreate(ev, 0) };
        anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuEventCreate {r:?}");
    }
    let flush = dev.alloc_zeros::<u8>(192 << 20)?;
    let flush_ptr = *flush.device_ptr();
    // (warm us, cold us) per call
    let time_it = |launch: &mut dyn FnMut() -> Result<()>| -> Result<(f64, f64)> {
        let mut warm = Vec::new();
        for rep in 0..28 {
            unsafe { sys::cuEventRecord(e[0], st) };
            for _ in 0..10 { launch()?; }
            unsafe { sys::cuEventRecord(e[1], st) };
            dev.synchronize()?;
            let mut ms: f32 = 0.0;
            let r = unsafe { sys::cuEventElapsedTime(&mut ms, e[0], e[1]) };
            anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuEventElapsedTime {r:?}");
            if rep >= 3 { warm.push(ms as f64 * 100.0); }
        }
        let mut cold = Vec::new();
        for rep in 0..28 {
            unsafe { sys::cuMemsetD8Async(flush_ptr, 0xA5, 192 << 20, st) };
            unsafe { sys::cuEventRecord(e[0], st) };
            launch()?;
            unsafe { sys::cuEventRecord(e[1], st) };
            dev.synchronize()?;
            let mut ms: f32 = 0.0;
            let r = unsafe { sys::cuEventElapsedTime(&mut ms, e[0], e[1]) };
            anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuEventElapsedTime {r:?}");
            if rep >= 3 { cold.push(ms as f64 * 1000.0); }
        }
        warm.sort_by(|p, q| p.partial_cmp(q).unwrap());
        cold.sort_by(|p, q| p.partial_cmp(q).unwrap());
        Ok((warm[warm.len() / 2], cold[cold.len() / 2]))
    };
    let mut lcg = 0x1234_5678u32;
    let mut rnd = move || { lcg = lcg.wrapping_mul(1664525).wrapping_add(1013904223); lcg >> 8 };
    let (k, ne_l, topk) = (2560usize, 128usize, 10usize);
    let floor_us = |bytes: f64| bytes / 238e9 * 1e6;
    let mut compared = 0usize;
    let mut table: Vec<String> = Vec::new();

    // ------------------------------------------------------------------ (1) had_suh
    let d_suh = dev.htod_sync_copy(&synth_x(ne_l, k, 0x51))?;
    let idxmap: Vec<i32> = (0..ne_l).map(|x| ((x * 37 + 5) % ne_l) as i32).collect();
    let d_idx = dev.htod_sync_copy(&idxmap)?;
    for &c in &[2043usize, 777, 33] {
        let r = c * topk;
        let d_x = dev.htod_sync_copy(&synth_x(c, k, 0x77 ^ c as u32))?;
        let live = r / 4 + (c % 7);
        let mut tok = vec![0i32; r];
        let mut eidx = vec![0i32; r];
        for i in 0..live { tok[i] = (rnd() as usize % c) as i32; eidx[i] = (i * ne_l / live) as i32; }
        let d_tok = dev.htod_sync_copy(&tok)?;
        let d_eidx = dev.htod_sync_copy(&eidx)?;
        let d_live = dev.htod_sync_copy(&[live as i32])?;
        let live_p = *d_live.device_ptr() as u64;
        let mut xh_old = dev.htod_sync_copy(&vec![0xFFFFu16; r * k])?;
        let mut xh_new = dev.htod_sync_copy(&vec![0xFFFFu16; r * k])?;
        unsafe { f_old_h.clone().launch(LaunchConfig { grid_dim: ((r * 20) as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 },
                                        (&d_x, &d_suh, &d_idx, &d_tok, &d_eidx, &mut xh_old, r as i32, k as i32))?; }
        dev.synchronize()?;
        let a = dev.dtoh_sync_copy(&xh_old)?;
        if a.iter().filter(|v| **v != 0 && (**v & 0x7C00) != 0x7C00).count() < r * k * 95 / 100 {
            bail!("EXL3-MOEGLUE FAIL: implausible had_suh reference (c={c})");
        }
        for (vi, f) in f_new_h.iter().enumerate() {
            for &grid in &[5u32, 288, 4096] {
                // full coverage (live_end = null)
                dev.htod_sync_copy_into(&vec![0xFFFFu16; r * k], &mut xh_new)?;
                unsafe { f.clone().launch(LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                          (&d_x, &d_suh, &d_idx, &d_tok, &d_eidx, &mut xh_new, 0u64, r as i32, k as i32))?; }
                dev.synchronize()?;
                let b = dev.dtoh_sync_copy(&xh_new)?;
                let mism = a.iter().zip(b.iter()).filter(|(p, q)| p != q).count();
                if mism > 0 { bail!("EXL3-MOEGLUE FAIL: had_suh_live_u{} grid {grid} c={c} (full): {mism}/{} halves differ", us_of[vi], r * k); }
                compared += r * k;
                // live-only coverage: live rows bitwise, dead rows untouched
                dev.htod_sync_copy_into(&vec![0xFFFFu16; r * k], &mut xh_new)?;
                unsafe { f.clone().launch(LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                          (&d_x, &d_suh, &d_idx, &d_tok, &d_eidx, &mut xh_new, live_p, r as i32, k as i32))?; }
                dev.synchronize()?;
                let b = dev.dtoh_sync_copy(&xh_new)?;
                let mism = a[..live * k].iter().zip(b[..live * k].iter()).filter(|(p, q)| p != q).count();
                if mism > 0 { bail!("EXL3-MOEGLUE FAIL: had_suh_live_u{} grid {grid} c={c} (live {live}): {mism} live halves differ", us_of[vi]); }
                if b[live * k..].iter().any(|v| *v != 0xFFFF) { bail!("EXL3-MOEGLUE FAIL: had_suh_live_u{} wrote past live_end (c={c})", us_of[vi]); }
                compared += live * k;
            }
        }
        println!("EXL3-MOEGLUE: had_suh c={c:>4} r={r:>5} live={live:>5}: 3 variants x 3 grids bitwise == xq_had_suh_rows (full + live-only, dead rows untouched)");
        // timing at the served chunk only
        if c == 2043 {
            let cfg_old = LaunchConfig { grid_dim: ((r * 20) as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
            let (ow, oc) = time_it(&mut || { unsafe { f_old_h.clone().launch(cfg_old, (&d_x, &d_suh, &d_idx, &d_tok, &d_eidx, &mut xh_old, r as i32, k as i32))?; } Ok(()) })?;
            let fl_all = floor_us((r * k * 2 + c * k * 2) as f64);
            let fl_live = floor_us((live * k * 2 + c.min(live) * k * 2) as f64);
            table.push(format!("  had_suh OLD (all r={r} rows)       warm {ow:>7.1} us  cold {oc:>7.1} us   floor(all rows) {fl_all:>6.1} us  floor(live rows) {fl_live:>6.1} us"));
            let mut best = (f64::MAX, 0usize, 0u32);
            for (vi, f) in f_new_h.iter().enumerate() {
                for &grid in &[96u32, 144, 192, 288, 384, 576] {
                    let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    let (nw, nc) = time_it(&mut || { unsafe { f.clone().launch(cfg, (&d_x, &d_suh, &d_idx, &d_tok, &d_eidx, &mut xh_new, live_p, r as i32, k as i32))?; } Ok(()) })?;
                    table.push(format!("  had_suh LIVE u{} grid {grid:>3} (live={live})  warm {nw:>7.1} us  cold {nc:>7.1} us   x{:.2} cold / x{:.2} warm vs OLD   {:.0} GB/s live-floor-bytes (cold)",
                                       us_of[vi], oc / nc, ow / nw, (live * k * 2 + c.min(live) * k * 2) as f64 / (nc * 1e-6) / 1e9));
                    if nc < best.0 { best = (nc, us_of[vi], grid); }
                }
            }
            table.push(format!("  had_suh BEST: u{} grid {} cold {:.1} us = x{:.2} vs OLD cold {:.1} us", best.1, best.2, best.0, oc / best.0, oc));
            // live-fraction sweep with the best config
            let f = &f_new_h[us_of.iter().position(|&u| u == best.1).unwrap()];
            for &frac in &[0.10f64, 0.25, 0.40, 0.60, 1.0] {
                let lv = ((r as f64 * frac) as usize).min(r) as i32;
                let d_lv = dev.htod_sync_copy(&[lv])?;
                let lp = *d_lv.device_ptr() as u64;
                let cfg = LaunchConfig { grid_dim: (best.2, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                let (nw, nc) = time_it(&mut || { unsafe { f.clone().launch(cfg, (&d_x, &d_suh, &d_idx, &d_tok, &d_eidx, &mut xh_new, lp, r as i32, k as i32))?; } Ok(()) })?;
                table.push(format!("  had_suh LIVE best, live fraction {frac:.2} ({lv} rows)  warm {nw:>7.1} us  cold {nc:>7.1} us"));
            }
        }
    }

    // ------------------------------------------------------------------ (2) combine
    let h = 2560usize;
    for &(c, k2, p, name) in &[(2043usize, 10usize, 0.25f64, "k10"), (777, 10, 1.0, "k10"), (300, 16, 0.4, "k16"), (33, 10, 0.25, "k10")] {
        let r = c * k2;
        let mut cand = vec![-1i32; r];
        let mut wts = vec![0f32; r];
        let mut live = 0usize;
        for i in 0..r {
            wts[i] = ((rnd() % 1000) + 1) as f32 / 1000.0 * 0.3;
            if (rnd() % 10000) as f64 / 10000.0 < p { cand[i] = live as i32; live += 1; }
        }
        let d_yd = dev.htod_sync_copy(&synth_x(r, h, 0x31 ^ c as u32))?;
        let d_ysh = dev.htod_sync_copy(&synth_x(c, h, 0x32 ^ c as u32))?;
        let d_x = dev.htod_sync_copy(&synth_x(c, h, 0x33 ^ c as u32))?;
        let sg: Vec<u16> = synth_x(1, h, 0x34).iter().map(|b| f16::from_f32(f16::from_bits(*b).to_f32() / 64.0).to_bits()).collect();
        let d_sg = dev.htod_sync_copy(&sg)?;
        let d_cand = dev.htod_sync_copy(&cand)?;
        let d_wts = dev.htod_sync_copy(&wts)?;
        let poison = f32::from_bits(0x7fc0_dead);
        let mut o_old = dev.htod_sync_copy(&vec![poison; c * h])?;
        let mut o_new = dev.htod_sync_copy(&vec![poison; c * h])?;
        let f_new = if name == "k10" { &f_c10 } else { &f_c16 };
        let cfg_old = LaunchConfig { grid_dim: (c as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 4096 };
        let cfg_new = LaunchConfig { grid_dim: (c as u32, 1, 1), block_dim: (320, 1, 1), shared_mem_bytes: 0 };
        for add_sh in [1i32, 0] {
            dev.htod_sync_copy_into(&vec![poison; c * h], &mut o_old)?;
            dev.htod_sync_copy_into(&vec![poison; c * h], &mut o_new)?;
            unsafe {
                f_old_c.clone().launch(cfg_old, (&mut o_old, &d_yd, &d_ysh, &d_cand, &d_wts, &d_sg, &d_x, k2 as i32, h as i32, c as i32, add_sh))?;
                f_new.clone().launch(cfg_new, (&mut o_new, &d_yd, &d_ysh, &d_cand, &d_wts, &d_sg, &d_x, k2 as i32, h as i32, c as i32, add_sh))?;
            }
            dev.synchronize()?;
            let a = dev.dtoh_sync_copy(&o_old)?;
            let b = dev.dtoh_sync_copy(&o_new)?;
            if a.iter().filter(|v| **v != 0.0 && v.is_finite()).count() < c * h * 80 / 100 || a.iter().any(|v| !v.is_finite()) {
                bail!("EXL3-MOEGLUE FAIL: implausible combine reference (c={c} k={k2} p={p} add_shared={add_sh})");
            }
            let mism = a.iter().zip(b.iter()).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
            if mism > 0 { bail!("EXL3-MOEGLUE FAIL: combine_ep2_{name} c={c} p={p} add_shared={add_sh}: {mism}/{} fp32 outputs differ", c * h); }
            compared += c * h;
        }
        println!("EXL3-MOEGLUE: combine c={c:>4} k={k2:>2} local-pick p={p:.2} (live rows {live:>5}): ep2_{name} bitwise == xq_moe_combine_rows_ep, add_shared 1 and 0");
        if c == 2043 {
            for add_sh in [1i32, 0] {
                let (ow, oc) = time_it(&mut || { unsafe { f_old_c.clone().launch(cfg_old, (&mut o_old, &d_yd, &d_ysh, &d_cand, &d_wts, &d_sg, &d_x, k2 as i32, h as i32, c as i32, add_sh))?; } Ok(()) })?;
                let (nw, nc) = time_it(&mut || { unsafe { f_new.clone().launch(cfg_new, (&mut o_new, &d_yd, &d_ysh, &d_cand, &d_wts, &d_sg, &d_x, k2 as i32, h as i32, c as i32, add_sh))?; } Ok(()) })?;
                let bytes = (c * h * 4 + live * h * 2 + if add_sh == 1 { 2 * c * h * 2 } else { 0 }) as f64;
                table.push(format!("  combine add_shared={add_sh} (c={c}, live rows {live}):  OLD warm {ow:>7.1} us cold {oc:>7.1} us | NEW warm {nw:>7.1} us cold {nc:>7.1} us | \
                                    x{:.2} cold / x{:.2} warm | floor {:.1} us ({:.0} GB/s new cold, {:.0} GB/s old cold)",
                                   oc / nc, ow / nw, floor_us(bytes), bytes / (nc * 1e-6) / 1e9, bytes / (oc * 1e-6) / 1e9));
            }
        }
    }
    for ev in e { unsafe { sys::cuEventDestroy_v2(ev); } }
    if compared == 0 { bail!("EXL3-MOEGLUE FAIL: nothing was compared"); }
    println!("EXL3-MOEGLUE timing (served chunk c=2043, k=10, ne_l=128, h=K=2560; warm = median of 25 x 10 back-to-back, cold = 192 MB flush then one launch, median of 25; standalone ranking):");
    for l in &table { println!("{l}"); }
    println!("EXL3-MOEGLUE: PASS (had_suh_live u1/u2/u4 and combine_ep2 k10/k16 bitwise == the old kernels; {compared} elements compared)");
    Ok(())
}

