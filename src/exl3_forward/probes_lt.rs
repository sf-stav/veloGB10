//! Offline probes: `probe_sp_pinned` (pinned split-point plans) and `probe_lt_rows` (Lt algorithm rows).
//! See `cublaslt.rs` for the plan cache they exercise and `mod.rs` for the file map.

use super::*;

/// The prefill GEMM shapes a sequence-parallel chunk splits by rows (k, n; ta = plain f16 weights):
/// the hc down (rw -> lr) and hc up (lr -> rw) projections of every hc_pre. (The PLE and the router
/// stay replicated in SP-1.)
pub(crate) fn sp_split_shapes(h: usize, hcn: usize, lr: usize) -> [(usize, usize); 2] { [(h * hcn, lr), (lr, h * hcn)] }

/// TP-SP gate (`--probe-exl3-sppin`, default c = 128..=4095 = the split range; the EXL3-SP-PIN section of
/// --probe-exl3-binv over [SP_MIN_ROWS, SP_MAX_ROWS]): for every c in
/// `cs` and every split shape (the served hc down/up at h 2560, hc 4, lr 320), the reference is the
/// SERVED replicated call (exl3_lt_matmul at m = c over all rows) and the candidate is EXACTLY the SP
/// prefill's call — every rank block through exl3_lt_matmul_pinned on row-offset pointers, the plan
/// readiness decided by lt_pin_ready — at world `world`. Outputs are NaN-poisoned first; the reference
/// must be complete and plausible; every element is compared bitwise.
pub fn probe_sp_pinned(cs: &[usize], world: usize) -> Result<()> {
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    let stream = fork_blocking_stream(&dev);
    let l = Launcher { dev: &dev, stream: &stream };
    let ws_len = 32usize * 1024 * 1024; // the PrefillScratch lt_ws (every served prefill Lt call)
    let d_ws = unsafe { dev.alloc::<u8>(ws_len) }?;
    let ws = *d_ws.device_ptr() as u64;
    let cmax = cs.iter().copied().max().unwrap_or(2048);
    let mut st = 0x5EB1_7A11u32;
    let mut rnd = move || { st ^= st << 13; st ^= st >> 17; st ^= st << 5; st };
    let mut f16v = |n: usize, sc: f32| -> Vec<u16> {
        (0..n).map(|_| f16::from_f32((((rnd() >> 9) as f32 / (1u32 << 23) as f32) * 2.0 - 1.0) * sc).to_bits()).collect()
    };
    let t0 = std::time::Instant::now();
    let (mut n_cases, mut n_bad, mut n_unready) = (0usize, 0usize, 0usize);
    for (name, (k, n)) in [("hc down (rw -> lr)", sp_split_shapes(2560, 4, 320)[0]), ("hc up (lr -> rw)", sp_split_shapes(2560, 4, 320)[1])] {
        let d_a = dev.htod_sync_copy(&f16v(n * k, 0.05))?;
        let d_b = dev.htod_sync_copy(&f16v(cmax * k, 2.0))?;
        let d_ref = dev.alloc_zeros::<u16>(cmax * n)?;
        let d_sp = dev.alloc_zeros::<u16>(cmax * n)?;
        let (pa, pb) = (*d_a.device_ptr() as u64, *d_b.device_ptr() as u64);
        let (pr, ps) = (*d_ref.device_ptr() as u64, *d_sp.device_ptr() as u64);
        let (mut vr, mut vs) = (vec![0u16; cmax * n], vec![0u16; cmax * n]);
        let (mut s_bad, mut s_unready, mut s_n) = (Vec::<usize>::new(), Vec::<usize>::new(), 0usize);
        for &c in cs {
            if c < world { continue; }
            s_n += 1;
            n_cases += 1;
            let live = c * n;
            // NaN-poison the live rows of both outputs (legacy-stream memset: ordered before the blocking
            // stream's GEMMs)
            for p in [pr, ps] {
                let rc = unsafe { cudarc::driver::sys::cuMemsetD16_v2(p, 0x7E00, live) };
                anyhow::ensure!(rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "cuMemsetD16 ({rc:?})");
            }
            exl3_lt_matmul(&l, pa, pb, pr, k, n, c, true, ws, ws_len)?;
            if let Err(e) = lt_pin_ready(k, n, c, world, true, ws_len) {
                s_unready.push(c);
                n_unready += 1;
                if s_unready.len() <= 8 { crate::dprintln!("  {name} c {c:>5}: pinned plans NOT READY ({e:#})"); }
                continue;
            }
            for r in 0..world {
                let (lo, hi) = crate::tp_xport::row_block(c, r, world);
                if hi == lo { continue; }
                exl3_lt_matmul_pinned(&l, pa, pb + (lo * k * 2) as u64, ps + (lo * n * 2) as u64, k, n, hi - lo, c,
                                      true, ws, ws_len)?;
            }
            dev.synchronize()?;
            dev.dtoh_sync_copy_into(&d_ref.slice(0..live), &mut vr[..live])?;
            dev.dtoh_sync_copy_into(&d_sp.slice(0..live), &mut vs[..live])?;
            let poisoned = vr[..live].iter().filter(|&&b| b == 0x7E00).count();
            anyhow::ensure!(poisoned == 0, "EXL3-SP-PIN {name} c {c}: the reference left {poisoned} poisoned outputs");
            let nzr = vr[..live].iter().filter(|&&b| b & 0x7fff != 0).count();
            anyhow::ensure!(nzr > live / 2, "EXL3-SP-PIN {name} c {c}: implausible reference ({nzr} nonzero of {live})");
            let mut nd = 0usize;
            let mut first = None;
            if vs[..live] != vr[..live] {
                for i in 0..live { if vs[i] != vr[i] { nd += 1; first.get_or_insert(i / n); } }
            }
            if nd > 0 {
                n_bad += 1;
                s_bad.push(c);
                if s_bad.len() <= 8 { crate::dprintln!("  {name} c {c:>5}: {nd} of {live} f16 DIFFER (first row {first:?})"); }
            }
        }
        crate::dprintln!("EXL3-SP-PIN {name} (k {k} n {n}): {s_n} c values at world {world}; bitwise differ at {} (c {:?}); \
                  pinned plans not ready at {} (c {:?})", s_bad.len(), &s_bad[..s_bad.len().min(20)],
                 s_unready.len(), &s_unready[..s_unready.len().min(20)]);
    }
    // TP-SP2 EXL3-SP-ROUTER: the router extension's calls (router_rows_launch + xq_router_topk on every
    // rank block, the kernel picked for the full chunk) vs the replicated router at m = c, for both
    // prefill.router_rows values: logits, ids and weights bitwise.
    if dev.get_func(MODULE, "xq_router_topk").is_none() { crate::exl3_bench::load_module_pub(&dev)?; }
    let (h, ne, topk) = (2560usize, 256usize, 8usize);
    let d_w = dev.htod_sync_copy(&f16v(ne * h, 0.05))?;
    let d_x = dev.htod_sync_copy(&f16v(cmax * h, 2.0))?;
    let (d_lr, d_ls) = (dev.alloc_zeros::<f32>(cmax * ne)?, dev.alloc_zeros::<f32>(cmax * ne)?);
    let (d_ir, d_is) = (dev.alloc_zeros::<i32>(cmax * topk)?, dev.alloc_zeros::<i32>(cmax * topk)?);
    let (d_wr, d_ws2) = (dev.alloc_zeros::<f32>(cmax * topk)?, dev.alloc_zeros::<f32>(cmax * topk)?);
    let (pw, px) = (*d_w.device_ptr() as u64, *d_x.device_ptr() as u64);
    let p = |s: &CudaSlice<f32>| *s.device_ptr() as u64;
    let (plr, pls, pwr, pws) = (p(&d_lr), p(&d_ls), p(&d_wr), p(&d_ws2));
    let (pir, pis) = (*d_ir.device_ptr() as u64, *d_is.device_ptr() as u64);
    let smem = ((256 + ne + 512) * 4) as u32;
    let mut n_router = 0usize;
    // standalone RANKING (not an in-round price, AGENTS §4) of the router kernels at the served chunk and its
    // half: does the own-row block's launch take half the time?
    if cmax >= 2048 {
        let tm = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            for _ in 0..3 { f()?; }
            dev.synchronize()?;
            let t = std::time::Instant::now();
            for _ in 0..20 { f()?; }
            dev.synchronize()?;
            Ok(t.elapsed().as_secs_f64() * 1e6 / 20.0)
        };
        let r8 = |m: usize| -> Result<()> {
            xqlaunch!(l, "xq_gemm_f16_f32_rows8", ((ne as u32) / 64, m.div_ceil(8) as u32, 1), (64, 1, 1), 0,
                     (pls, pw, px, m as i32, ne as i32, h as i32))
        };
        let a = tm(&|| router_rows_launch(&l, plr, pw, px, 2048, 2048, ne, h, true))?;
        let b = tm(&|| router_rows_launch(&l, pls, pw, px, 1024, 2048, ne, h, true))?;
        let c8 = tm(&|| r8(1024))?;
        let d8 = tm(&|| r8(2048))?;
        let e = tm(&|| router_rows_launch(&l, plr, pw, px, 4095, 4095, ne, h, true))?;
        let f = tm(&|| router_rows_launch(&l, pls, pw, px, 2048, 4095, ne, h, true))?;
        crate::dprintln!("EXL3-SP-ROUTER timing (standalone, L2-warm, us/launch; a ranking only): rows32 m=2048 {a:.1} | rows32 m=1024 \
                  {b:.1} | rows8 m=1024 {c8:.1} | rows8 m=2048 {d8:.1} | rows32 m=4095 {e:.1} | rows32 m=2048 (of 4095) {f:.1}");
    }
    for rows_on in [true, false] {
        let mut s_bad = Vec::<usize>::new();
        let mut s_n = 0usize;
        for &c in cs {
            if c < world { continue; }
            s_n += 1;
            n_cases += 1;
            n_router += 1;
            for (q, n) in [(plr, c * ne), (pls, c * ne), (pwr, c * topk), (pws, c * topk), (pir, c * topk), (pis, c * topk)] {
                let rc = unsafe { cudarc::driver::sys::cuMemsetD32_v2(q, 0x7FC0_0000, n) };
                anyhow::ensure!(rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "cuMemsetD32 ({rc:?})");
            }
            router_rows_launch(&l, plr, pw, px, c, c, ne, h, rows_on)?;
            xqlaunch!(l, "xq_router_topk", (c as u32, 1, 1), (256, 1, 1), smem, (pir, pwr, plr, ne as i32, topk as i32, c as i32))?;
            for r in 0..world {
                let (lo, hi) = crate::tp_xport::row_block(c, r, world);
                if hi == lo { continue; }
                let m = hi - lo;
                router_rows_launch(&l, pls + (lo * ne * 4) as u64, pw, px + (lo * h * 2) as u64, m, c, ne, h, rows_on)?;
                xqlaunch!(l, "xq_router_topk", (m as u32, 1, 1), (256, 1, 1), smem,
                         (pis + (lo * topk * 4) as u64, pws + (lo * topk * 4) as u64, pls + (lo * ne * 4) as u64,
                          ne as i32, topk as i32, m as i32))?;
            }
            dev.synchronize()?;
            let mut lr = vec![0f32; c * ne]; dev.dtoh_sync_copy_into(&d_lr.slice(0..c * ne), &mut lr)?;
            let mut ls = vec![0f32; c * ne]; dev.dtoh_sync_copy_into(&d_ls.slice(0..c * ne), &mut ls)?;
            let mut ir = vec![0i32; c * topk]; dev.dtoh_sync_copy_into(&d_ir.slice(0..c * topk), &mut ir)?;
            let mut is = vec![0i32; c * topk]; dev.dtoh_sync_copy_into(&d_is.slice(0..c * topk), &mut is)?;
            let mut wr = vec![0f32; c * topk]; dev.dtoh_sync_copy_into(&d_wr.slice(0..c * topk), &mut wr)?;
            let mut ws2 = vec![0f32; c * topk]; dev.dtoh_sync_copy_into(&d_ws2.slice(0..c * topk), &mut ws2)?;
            anyhow::ensure!(lr.iter().all(|v| v.is_finite()) && ir.iter().all(|&i| (0..ne as i32).contains(&i))
                            && wr.iter().all(|v| v.is_finite() && *v > 0.0),
                            "EXL3-SP-ROUTER c {c}: incomplete / implausible reference");
            let bits = |a: &[f32], b: &[f32]| a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            let (dl, di, dw) = (bits(&lr, &ls), ir.iter().zip(&is).filter(|(x, y)| x != y).count(), bits(&wr, &ws2));
            if dl + di + dw > 0 {
                n_bad += 1;
                s_bad.push(c);
                if s_bad.len() <= 8 { crate::dprintln!("  router rows={} c {c:>5}: logits {dl} / ids {di} / weights {dw} DIFFER", rows_on as u8); }
            }
        }
        crate::dprintln!("EXL3-SP-ROUTER (x -> {ne} experts, top-{topk}, prefill.router_rows {}): {s_n} c values at world {world}; \
                  bitwise differ at {} (c {:?})", rows_on as u8, s_bad.len(), &s_bad[..s_bad.len().min(20)]);
    }
    let range = format!("c in {}..={}", cs.iter().min().copied().unwrap_or(0), cmax);
    if n_bad == 0 && n_unready == 0 {
        crate::dprintln!("EXL3-SP-PIN: PASS ({n_cases} (shape, c) cases, {range}, world {world}: every rank block through the served \
                  pinned-plan call is bit-identical to the replicated m = c call; incl. {n_router} router cases: logits / ids / \
                  weights of every block == the replicated router; {:.1} s)", t0.elapsed().as_secs_f64());
        Ok(())
    } else {
        anyhow::bail!("EXL3-SP-PIN FAIL ({n_bad} of {n_cases} cases differ, {n_unready} not ready; {range}, world {world})")
    }
}

/// One cuBLASLt algo's config attributes, printable (TP-I3 (a) rung-1 probe).
fn lt_algo_desc(algo: &ltsys::cublasLtMatmulAlgo_t) -> String {
    use ltsys::cublasLtMatmulAlgoConfigAttributes_t as A;
    let get = |a: A| -> String {
        let mut need: usize = 0;
        let r0 = unsafe { ltsys::cublasLtMatmulAlgoConfigGetAttribute(algo, a, std::ptr::null_mut(), 0, &mut need) };
        if r0 != ltsys::cublasStatus_t::CUBLAS_STATUS_SUCCESS || need == 0 || need > 8 { return "?".into(); }
        let mut v: u64 = 0;
        let mut w: usize = 0;
        let r = unsafe { ltsys::cublasLtMatmulAlgoConfigGetAttribute(algo, a, &mut v as *mut u64 as *mut _, need, &mut w) };
        if r != ltsys::cublasStatus_t::CUBLAS_STATUS_SUCCESS { return "?".into(); }
        format!("{v}")
    };
    format!("id {} tile {} splitK {} red {} swz {} custom {} stages {} inner {} cluster {}",
            get(A::CUBLASLT_ALGO_CONFIG_ID), get(A::CUBLASLT_ALGO_CONFIG_TILE_ID), get(A::CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
            get(A::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME), get(A::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING),
            get(A::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION), get(A::CUBLASLT_ALGO_CONFIG_STAGES_ID),
            get(A::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID), get(A::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID))
}

/// TP-I3 (a) rung 1 (`--probe-exl3-ltrows`, synthetic, no pack): is `exl3_lt_matmul` (cuBLASLt, one
/// heuristic plan per (k, n, m)) PER-ROW bit-identical when a chunk's rows are split between two
/// ranks (sequence-parallel prefill)? For each prefill Lt shape that a row split would touch (the hc
/// down/up projections; the PLE key/value GEMMs as the extension) and each chunk size c, the served
/// reference is the heuristic plan at m = c over all rows. Two candidates run the halves
/// [0, c0) and [c0, c) (c0 = ceil(c/2)) at m = c0 / c - c0 on row-offset pointers:
///   (H) each half with ITS OWN heuristic plan (what a naive split would do);
///   (P) each half with the reference plan's algo PINNED (layouts for the half's m; validated with
///       cublasLtMatmulAlgoCheck).
/// Every output element of both halves is compared bitwise with the reference rows. Prints the
/// algo configs and harness-only times (µs per call, L2-warm).
pub fn probe_lt_rows(cs: &[usize], shapes_sel: &str) -> Result<()> {
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    let stream = fork_blocking_stream(&dev);
    let l = Launcher { dev: &dev, stream: &stream };
    let handle = lt_handle();
    let ws_len = 32usize * 1024 * 1024; // the PrefillScratch lt_ws (every served prefill Lt call)
    let d_ws = unsafe { dev.alloc::<u8>(ws_len) }?;
    let ws = *d_ws.device_ptr() as u64;
    let cmax = cs.iter().copied().max().unwrap_or(2048);
    // (name, k, n): served prefill Lt shapes (ta = true: plain f16 weight rows [n][k])
    let shapes: [(&str, usize, usize); 4] = [("hc down (rw -> lr)", 10240, 320), ("hc up (lr -> rw)", 320, 10240),
                                            ("PLE key16 (ext)", 2560, 10240), ("PLE val (ext)", 2560, 2560)];
    let mut st = 0x2545_F491u32;
    let mut rnd = move || { st ^= st << 13; st ^= st >> 17; st ^= st << 5; st };
    let mut f16v = |n: usize, sc: f32| -> Vec<u16> {
        (0..n).map(|_| f16::from_f32((((rnd() >> 9) as f32 / (1u32 << 23) as f32) * 2.0 - 1.0) * sc).to_bits()).collect()
    };
    let (mut bad_h, mut bad_p, mut n_cases, mut p_invalid, mut bad_r) = (0usize, 0usize, 0usize, 0usize, 0usize);
    // sweep mode (many c): print only the failing cases + a per-shape summary; no timing
    let sweep = cs.len() > 40;
    for &(name, k, n) in shapes.iter().filter(|s| shapes_sel == "all" || s.0.starts_with("hc")) {
        let d_a = dev.htod_sync_copy(&f16v(n * k, 0.05))?;
        let d_b = dev.htod_sync_copy(&f16v(cmax * k, 2.0))?;
        let mut d_ref = dev.alloc_zeros::<u16>(cmax * n)?;
        let mut d_r2 = dev.alloc_zeros::<u16>(cmax * n)?;
        let mut d_h = dev.alloc_zeros::<u16>(cmax * n)?;
        let mut d_p = dev.alloc_zeros::<u16>(cmax * n)?;
        let (pa, pb) = (*d_a.device_ptr() as u64, *d_b.device_ptr() as u64);
        crate::dprintln!("LTROWS {name}: k {k} n {n}");
        let (mut s_fail_p, mut s_fail_h, mut s_fail_r, mut s_n) = (Vec::<usize>::new(), 0usize, Vec::<usize>::new(), 0usize);
        for &c in cs {
            let c0 = c.div_ceil(2);
            let c1 = c - c0;
            if c0 == 0 || c1 == 0 { continue; }
            n_cases += 1;
            s_n += 1;
            let poison = |d: &mut CudaSlice<u16>| -> Result<()> { dev.htod_copy_into(vec![0x7E00u16; cmax * n], d)?; Ok(()) };
            poison(&mut d_ref)?; poison(&mut d_h)?; poison(&mut d_p)?; poison(&mut d_r2)?;
            let pref_ = *d_ref.device_ptr() as u64;
            let pr2 = *d_r2.device_ptr() as u64;
            let (ph, pp) = (*d_h.device_ptr() as u64, *d_p.device_ptr() as u64);
            // reference: the served call (heuristic plan at m = c, cached like exl3_lt_matmul), and (R)
            // the same call again (run-to-run determinism of the served plan itself)
            exl3_lt_matmul(&l, pa, pb, pref_, k, n, c, true, ws, ws_len)?;
            exl3_lt_matmul(&l, pa, pb, pr2, k, n, c, true, ws, ws_len)?;
            // (H) naive split: each half with its own heuristic plan
            exl3_lt_matmul(&l, pa, pb, ph, k, n, c0, true, ws, ws_len)?;
            exl3_lt_matmul(&l, pa, pb + (c0 * k * 2) as u64, ph + (c0 * n * 2) as u64, k, n, c1, true, ws, ws_len)?;
            // (P) pinned: the m = c plan's algo on the halves' layouts
            let ref_algo = lt_plans().lock().unwrap().get(&(k, n, c, 1u8)).map(|p| p.algo).context("ref plan missing")?;
            let mut p_ok = true;
            let mut p_desc = String::new();
            for (m_h, boff, coff) in [(c0, 0usize, 0usize), (c1, c0 * k * 2, c0 * n * 2)] {
                let mut plan = lt_plan_build(handle, k, n, m_h, true, ws_len).map_err(|s| anyhow::anyhow!(s))?;
                plan.algo = ref_algo;
                let mut chk = std::mem::MaybeUninit::<ltsys::cublasLtMatmulHeuristicResult_t>::zeroed();
                let rc = unsafe {
                    ltsys::cublasLtMatmulAlgoCheck(handle, plan.desc, plan.a_layout, plan.b_layout, plan.c_layout,
                                                   plan.c_layout, &plan.algo, chk.as_mut_ptr())
                };
                if rc != ltsys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    p_ok = false;
                    p_desc = format!("AlgoCheck(m={m_h}) = {rc:?}");
                    lt_plan_drop(plan);
                    break;
                }
                let wsz = unsafe { chk.assume_init() }.workspaceSize;
                if wsz > ws_len { p_ok = false; p_desc = format!("workspace {wsz} > {ws_len}"); lt_plan_drop(plan); break; }
                let (alpha, beta) = (1.0f32, 0.0f32);
                let r = unsafe {
                    lt::matmul(handle, plan.desc, &alpha as *const _ as *const _, &beta as *const _ as *const _,
                        pa as *const _, plan.a_layout, (pb + boff as u64) as *const _, plan.b_layout,
                        (pp + coff as u64) as *mut _, plan.c_layout, (pp + coff as u64) as *mut _, plan.c_layout,
                        &plan.algo, ws as *mut _, ws_len, stream.stream as usize as ltsys::cudaStream_t)
                };
                lt_plan_drop(plan);
                r.with_context(|| format!("pinned matmul m={m_h}"))?;
            }
            dev.synchronize()?;
            let (vr, vh, vp) = (dev.dtoh_sync_copy(&d_ref)?, dev.dtoh_sync_copy(&d_h)?, dev.dtoh_sync_copy(&d_p)?);
            let vr2 = dev.dtoh_sync_copy(&d_r2)?;
            let live = c * n;
            let poisoned = vr[..live].iter().filter(|&&b| b == 0x7E00).count();
            anyhow::ensure!(poisoned == 0, "LTROWS {name} c {c}: the reference left {poisoned} poisoned outputs");
            let nzr = vr[..live].iter().filter(|&&b| b & 0x7fff != 0).count();
            anyhow::ensure!(nzr > live / 2, "LTROWS {name} c {c}: implausible reference ({nzr} nonzero of {live})");
            let neq = |v: &[u16]| -> (usize, Option<usize>) {
                let mut k2 = (0usize, None);
                for i in 0..live { if v[i] != vr[i] { k2.0 += 1; k2.1.get_or_insert(i / n); } }
                k2
            };
            let (nh, fh) = neq(&vh);
            let (np, fp) = if p_ok { neq(&vp) } else { (usize::MAX, None) };
            let (nr, _) = neq(&vr2);
            if nh > 0 { bad_h += 1; s_fail_h += 1; }
            if nr > 0 { bad_r += 1; s_fail_r.push(c); }
            if !p_ok { p_invalid += 1; s_fail_p.push(c); } else if np > 0 { bad_p += 1; s_fail_p.push(c); }
            if sweep {
                if nr > 0 || !p_ok || np > 0 {
                    crate::dprintln!("  c {c:>5}: (P) pinned {} | (R) ref re-run {} | ref [{}]",
                             if !p_ok { format!("INVALID ({p_desc})") } else { format!("differ {np} (first row {fp:?})") },
                             if nr == 0 { "identical".to_string() } else { format!("DIFFER {nr}") }, lt_algo_desc(&ref_algo));
                }
                continue;
            }
            let (ah, bh) = {
                let map = lt_plans().lock().unwrap();
                (map.get(&(k, n, c0, 1u8)).map(|p| lt_algo_desc(&p.algo)).unwrap_or_default(),
                 map.get(&(k, n, c1, 1u8)).map(|p| lt_algo_desc(&p.algo)).unwrap_or_default())
            };
            // harness-only times (L2-warm, 20 reps): ref at m = c, pinned halves, heuristic halves
            let time = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
                f()?; dev.synchronize()?;
                let t = std::time::Instant::now();
                for _ in 0..20 { f()?; }
                dev.synchronize()?;
                Ok(t.elapsed().as_secs_f64() * 1e6 / 20.0)
            };
            let t_ref = time(&mut || exl3_lt_matmul(&l, pa, pb, pref_, k, n, c, true, ws, ws_len))?;
            let t_h0 = time(&mut || exl3_lt_matmul(&l, pa, pb, ph, k, n, c0, true, ws, ws_len))?;
            crate::dprintln!("  c {c:>5} = {c0} + {c1}: (H) own-heuristic halves {} | (P) pinned {} | (R) ref re-run {} | ref [{}] | half0 [{ah}] \
                      | half1 [{bh}] | us: ref {t_ref:.1} half0(H) {t_h0:.1}",
                     if nh == 0 { "BIT-IDENTICAL".to_string() } else { format!("DIFFER {nh} of {live} (first row {fh:?})") },
                     if !p_ok { format!("INVALID ({p_desc})") } else if np == 0 { "BIT-IDENTICAL".to_string() }
                     else { format!("DIFFER {np} of {live} (first row {fp:?})") },
                     if nr == 0 { "identical".to_string() } else { format!("DIFFER {nr}") },
                     lt_algo_desc(&ref_algo));
        }
        crate::dprintln!("LTROWS {name} summary: {s_n} c values; own-heuristic halves differ at {s_fail_h}; pinned halves differ/invalid at {} \
                  (largest {:?}); served plan non-deterministic at {} (c {:?})",
                 s_fail_p.len(), s_fail_p.iter().max(), s_fail_r.len(), &s_fail_r[..s_fail_r.len().min(20)]);
    }
    println!("LTROWS RESULT: {n_cases} (shape, c) cases; own-heuristic halves differ in {bad_h}; pinned-plan halves differ in {bad_p} \
              ({p_invalid} pinned plans invalid); served plan re-run differs in {bad_r} -> {}",
             if bad_p == 0 && p_invalid == 0 && bad_r == 0 { "PINNED_ROW_IDENTITY_OK" } else { "PINNED_ROW_IDENTITY_FAIL" });
    Ok(())
}
