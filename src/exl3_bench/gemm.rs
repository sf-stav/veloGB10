//! EXL3 GEMM kernel-decode gate, the multi-width GEMM probe, and the fixed-cost ladder / GEMM / LMH benches (--probe-exl3-kernel, --probe-exl3-binv, --bench-exl3-gemm).
//! Probe-only code of the `exl3_bench` module (see `mod.rs` for the map); shared helpers come from the parent.
//! Numerics contract (module header in `mod.rs`): the kernel decode is bitwise vs the host oracle; do not change reduction orders.

use super::*;

// ---------------------------------------------------------------------------
// 1. G-A3-1 through the kernel: CUDA decode vs Rust oracle, bitwise.
// ---------------------------------------------------------------------------
pub fn probe_kernel_decode(dir: &str) -> Result<()> {
    let pack = Exl3Pack::open(dir)?;
    let dev = dev0()?;
    load_module(&dev)?;
    let rc = tmap_inv_table();
    let d_rc = dev.htod_sync_copy(&rc)?;
    let f_dump = dev
        .get_func(MODULE, "exl3_decode_dump")
        .ok_or_else(|| anyhow::anyhow!("fn exl3_decode_dump missing"))?;
    let stream = fork_blocking_stream(&dev);

    let modules = pick_modules(&pack);
    if modules.is_empty() {
        bail!("no gate matrices found in pack {dir}");
    }
    let mut total_blocks = 0usize;
    for m in &modules {
        let (kb, nb) = (m.k / 16, m.n / 16);
        let t0 = std::time::Instant::now();
        let raw = m.trellis.read_bytes(&pack.dir)?;
        let trellis = le_i16(&raw);
        let ref_w = decode_trellis_host(&trellis, kb, nb, m.k2);
        let t_ref = t0.elapsed();

        let d_tr = dev.htod_sync_copy(&raw)?;
        let mut d_out = unsafe { dev.alloc::<u16>(ref_w.len()) }?;
        unsafe {
            f_dump
                .clone()
                .launch_on_stream(
                    &stream,
                    LaunchConfig {
                        grid_dim: ((kb * nb) as u32, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&d_tr, &d_rc, &mut d_out, nb as i32, m.k2 as i32),
                )
                .context("launch exl3_decode_dump")?;
        }
        dev.synchronize()?;
        let mut got = dev.dtoh_sync_copy(&d_out)?;
        // S-A3-u G: the coalesced prefill reconstruct must produce the same bytes.
        if nb % 8 == 0 {
            let f8 = dev.get_func(MODULE, "exl3_decode_dump8")
                .ok_or_else(|| anyhow::anyhow!("fn exl3_decode_dump8 missing"))?;
            let mut d_out8 = unsafe { dev.alloc::<u16>(ref_w.len()) }?;
            unsafe {
                f8.launch_on_stream(&stream,
                    LaunchConfig { grid_dim: ((kb * nb / 8) as u32, 1, 1), block_dim: (256, 1, 1),
                                   shared_mem_bytes: 0 },
                    (&d_tr, &d_rc, &mut d_out8, nb as i32, m.k2 as i32))
                    .context("launch exl3_decode_dump8")?;
            }
            dev.synchronize()?;
            let got8 = dev.dtoh_sync_copy(&d_out8)?;
            let m8 = got8.iter().zip(got.iter()).filter(|(a, b)| a != b).count();
            if m8 > 0 {
                bail!("EXL3-KERNEL-DECODE FAIL: exl3_decode_dump8 differs from exl3_decode_dump in {m8} of {} on {}", got.len(), m.name);
            }
            got = got8; // and it is what gets compared to the oracle below
        }

        let mut mism = 0usize;
        let mut first: Vec<(usize, u16, u16)> = Vec::new();
        for (i, (g, r)) in got.iter().zip(ref_w.iter()).enumerate() {
            if g != r {
                mism += 1;
                if first.len() < 4 {
                    first.push((i, *g, *r));
                }
            }
        }
        total_blocks += kb * nb;
        println!(
            "  {:<28} b={} {:>5}x{:<6} {:>9} blocks  mism {}  (ref {:?}, gpu {:?})",
            m.name.rsplit('.').take(3).collect::<Vec<_>>().join("."),
            m.k2 / 2,
            m.k,
            m.n,
            kb * nb,
            mism,
            t_ref,
            t0.elapsed() - t_ref
        );
        for (i, g, r) in &first {
            println!("    first mismatch @{}: kernel {:04x} vs ref {:04x}", i, g, r);
        }
        if mism > 0 {
            bail!("EXL3-KERNEL-DECODE FAIL on {}", m.name);
        }
    }
    println!(
        "EXL3-KERNEL-DECODE: PASS (bitwise vs decode_trellis_host, {} blocks, {} matrices)",
        total_blocks,
        modules.len()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 2. G-A3-5: batch invariance + GEMM correctness vs host reference.
// ---------------------------------------------------------------------------
pub fn probe_gemm(dir: &str, widths: &[usize], slab_cols: usize) -> Result<()> {
    let pack = Exl3Pack::open(dir)?;
    let dev = dev0()?;
    load_module(&dev)?;
    // --binv-section=wp18: only the WP18 prefill-selection section (synthetic; no pack reads).
    if crate::opts::var(crate::opt!("binv-section")).as_deref() == Ok("wp18") {
        return probe_wp18_select(&dev);
    }
    // --binv-section=moe_pf: only the A5-P1 prefill MoE section (synthetic; no pack reads).
    if crate::opts::var(crate::opt!("binv-section")).as_deref() == Ok("moe_pf") {
        for bits in [2usize, 3, 5, 6, 7] { probe_moe_pf(&dev, bits)?; }
        return Ok(());
    }
    // --binv-section=pf_hc: only the A5-P2 prefill hc section (synthetic; no pack reads).
    if crate::opts::var(crate::opt!("binv-section")).as_deref() == Ok("pf_hc") {
        return probe_pf_hc(&dev);
    }
    // --binv-section=gdn_tc: only the A5-L1 chunkwise GDN scan section (synthetic; no pack reads).
    if crate::opts::var(crate::opt!("binv-section")).as_deref() == Ok("gdn_tc") {
        return probe_gdn_tc2(&dev);
    }

    for m in pick_modules(&pack) {
        // lm_head: gate on a column slab (the host reference for the full 248320-col
        // matrix would decode 1.27 GB and GEMM it — the slab carries the same per-tile
        // arithmetic; the full matrix is the BENCH's job).
        let raw = m.trellis.read_bytes(&pack.dir)?;
        let (trb, n_eff) = if m.class == TensorClass::LmHead {
            slab_trellis(&raw, m.k / 16, m.n / 16, m.k2, Some(slab_cols))
        } else {
            (raw.clone(), m.n)
        };
        let n_eff16 = n_eff;
        let suh = le_f16_bits(&m.suh.read_bytes(&pack.dir)?);
        let svh_full = le_f16_bits(&m.svh.read_bytes(&pack.dir)?);
        let svh: Vec<u16> = if n_eff16 == m.n {
            svh_full
        } else {
            svh_full[..n_eff16].to_vec()
        };

        // correctness once at width 4 (covers padded rows), then binv across widths.
        let k = m.k;
        let bits = m.k2 / 2;
        let tr_bits: Vec<u16> = trb
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();

        println!(
            "  {:<24} b={} K={:<5} N={:<6} (slab {} cols)",
            m.name.rsplit('.').take(2).collect::<Vec<_>>().join("."),
            m.k2 as f64 / 2.0,
            k,
            m.n,
            n_eff16
        );

        // ---- correctness at width 4 ----
        let w = 4usize;
        let x = synth_x(w, k, 0xC0FFEE);
        let chain = run_chain(&dev, &tr_bits, &suh, &svh, &x, w, k, n_eff16, m.k2)?; // k2-keyed (P7 B8)
        let yraw = dev.dtoh_sync_copy(&chain.d_yraw)?;
        let y = dev.dtoh_sync_copy(&chain.d_y)?;

        let xh_ref = had_scale_ref(&x, &suh, w, k, true);
        let w_dec = decode_trellis_host(
            &tr_bits.iter().map(|v| *v as i16).collect::<Vec<_>>(),
            k / 16,
            n_eff16 / 16,
            m.k2, // k2-keyed (P7 B8; was floored bits)
        );
        let yraw_ref = gemm_ref(&w_dec, &xh_ref, w, k, n_eff16);
        let (rel_raw_f32, rel_raw_16) = rel_l2(&yraw[..w * n_eff16], &yraw_ref);
        let y_ref = had_scale_ref(
            &yraw_ref.iter().map(|v| f16::from_f32(*v).to_bits()).collect::<Vec<_>>(),
            &svh,
            w,
            n_eff16,
            false,
        );
        let (rel_y_f32, rel_y_16) = rel_l2(
            &y[..w * n_eff16],
            &y_ref.iter().map(|b| f16::from_bits(*b).to_f32()).collect::<Vec<_>>(),
        );
        println!(
            "    correctness W={w}: y_raw rel-L2 (vs f32 {:.2e} | vs fp16(ref) {:.2e})   y rel-L2 (vs f32 {:.2e} | vs fp16(ref) {:.2e})   [residual = mma-vs-sequential association + fp16 output rounding; decode is bitwise-green]",
            rel_raw_f32, rel_raw_16, rel_y_f32, rel_y_16
        );
        // Thresholds calibrated 2026-09-19 on .13: fp16 OUTPUT rounding alone is ~1e-4-class
        // (ULP 9.8e-4 rel; ~5% of elements straddle a rounding boundary at the measured
        // association residual), so 1e-5 was unachievable by construction. Structural bugs
        // (transposed ref, wrong placement) measure O(1) — the gate's real target.
        if rel_raw_16 > 1e-4 || rel_y_16 > 5e-4 {
            bail!("EXL3-GEMM correctness FAIL on {}", m.name);
        }

        // ---- batch invariance: row 0 bitwise across widths ----
        let mut row0_w: Vec<Vec<u16>> = Vec::new();
        for wd in widths {
            let x = synth_x(*wd, k, 0xC0FFEE);
            // PR #27 (lesserevil): one chain per width. Relaunching the 4-row correctness chain with a new x kept
            // m_rows = 4: widths 1 and 2 read past their input and widths above 4 computed 4 rows, so this
            // "row 0 bitwise across widths" check never varied the width.
            let width_chain = run_chain(&dev, &tr_bits, &suh, &svh, &x, *wd, k, n_eff16, m.k2)?;
            let yraw = dev.dtoh_sync_copy(&width_chain.d_yraw)?;
            let y = dev.dtoh_sync_copy(&width_chain.d_y)?;
            row0_w.push(yraw[..n_eff16].to_vec());
            row0_w.push(y[..n_eff16].to_vec());
            let _ = y;
        }
        let base_raw = &row0_w[0];
        let mut mism = 0usize;
        for (wi, chunk) in row0_w.chunks(2).enumerate().skip(1) {
            for (i, (a, b)) in base_raw.iter().zip(chunk[0].iter()).enumerate() {
                if a != b {
                    mism += 1;
                    if mism < 4 {
                        println!("    binv mismatch y_raw W={}[{}]: {:04x} vs W=1 {:04x}", widths[wi], i, b, a);
                    }
                }
            }
            let _ = &chunk[1];
        }
        if mism > 0 {
            bail!("EXL3-BINV FAIL on {} ({} mismatches)", m.name, mism);
        }
        println!("    binv: PASS (y_raw row 0 bitwise identical across widths {widths:?})");
        // P7 B8: grouped-A1 leg is integer-width only — the reference drives the k2-keyed
        // m16 switch (2*bits) and the grouped kernels are bits-keyed with no half-width
        // instantiation (dense pack; expert_k2_supported has no fractional k2). Skip
        // fractional with one line instead of a false FAIL.
        // K35: k2-keyed — the fractional widths run xq_gemm_grouped_xh at the EXL3_FRAC_BITS code
        // (exl3_gemm_body_a1<KA,HALF> / exl3_gemm_body<KA,HALF>) against the k2-keyed m16 reference.
        probe_grouped_a1(&dev, &chain.d_tr, k, n_eff16, m.k2)?;
        // W3/LMH: the persistent lm_head GEMM on this matrix (the head: its column slab), every
        // m in 1..16 — every class exercises the 3/4/5-bit decode extraction.
        let all_m: Vec<usize> = (1..=16).collect();
        let nsts: &[usize] = if bits == 5 { &[4, 6, 8] } else { &[6] };
        probe_lmh(&dev, &chain.d_tr, &chain.d_svh, k, n_eff16, bits, &all_m, nsts, "matrix/slab")?;
        if m.class == TensorClass::LmHead {
            // The served shapes: the FULL head (N = vocab: 1,940 tiles over the SM grid, uneven
            // tiles per CTA) and the 65,536-id draft slice (512 tiles).
            let full: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let svh_all = le_f16_bits(&m.svh.read_bytes(&pack.dir)?);
            let d_full = dev.htod_sync_copy(&full)?;
            let d_svh_all = dev.htod_sync_copy(&svh_all)?;
            drop(full);
            probe_lmh(&dev, &d_full, &d_svh_all, k, m.n, bits, &all_m, &[6], "full head")?;
            probe_lmh(&dev, &d_full, &d_svh_all, k, m.n, bits, &[1, 6, 9, 16], &[4, 8], "full head, ring sweep")?;
            drop(d_full);
            let (trd, nd) = slab_trellis(&raw, k / 16, m.n / 16, m.k2, Some(65536));
            let trd16: Vec<u16> = trd.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let d_draft = dev.htod_sync_copy(&trd16)?;
            let d_svh_draft = dev.htod_sync_copy(&svh_all[..nd])?;
            probe_lmh(&dev, &d_draft, &d_svh_draft, k, nd, bits, &[1, 2, 6, 16], &[4, 6, 8], "draft slice")?;
        }
    }
    probe_ks_p(&dev, &pack)?;
    probe_wp21(&dev, &pack)?;
    probe_dense(&dev, &pack)?;
    println!("EXL3-GEMM: PASS (correctness + batch invariance)");
    probe_attn(&dev)?;
    probe_attn_sparse(&dev)?;
    probe_qsa_select(&dev)?;
    probe_wp18_select(&dev)?;
    probe_gemm_f16_rows(&dev)?;
    probe_moe_epi(&dev, 3)?;
    for b in [2usize, 5, 6, 7] { probe_moe_epi(&dev, b)?; }   // B10 width rows (K=4 keeps its own probe)
    probe_moe_epi_k4(&dev)?;
    for b in [5usize, 2, 6, 7] { probe_moe_epi_bwd(&dev, b)?; }
    probe_hc_w4(&dev)?;
    for bits in [2usize, 3, 5, 6, 7] { probe_moe_pf(&dev, bits)?; }
    probe_pf_hc(&dev)?;
    probe_gdn_tc2(&dev)?;
    // TP-SP1: the sequence-parallel prefill's pinned-plan gate over the whole split range (world 2)
    crate::exl3_forward::probe_sp_pinned(&(crate::exl3_forward::xtp::SP_MIN_ROWS..=crate::exl3_forward::xtp::SP_MAX_ROWS)
                                           .collect::<Vec<usize>>(), 2)?;
    Ok(())
}

/// S-C4STEP5 — the fixed-cost ablation ladder for the DENSE ks_fx shapes
/// (bench-harness only; probes `exl3_hmma_gemm_ks_fx_probe`, nothing served).
/// Per (dense module shape, K as served TP1/TP2, ks, M): time the served
/// `exl3_hmma_gemm_ks_fx` (normal) plus the two probe arms —
///   alias (MODE 1): trellis reads wrap into a w_kb window (L2-resident)  -> DRAM arm
///   stub  (MODE 2): load-preserving decode stub, bit math removed        -> decode-ALU arm
/// The deltas decompose the per-launch cost; GB/s uses the full trellis bytes.
pub fn bench_ladder(
    dir: &str,
    ms: &[usize],
    kss: &[usize],
    iters: usize,
    window_kb: usize,
    skip_n: usize,
    out_json: Option<&str>,
) -> Result<()> {
    let pack = Exl3Pack::open(dir)?;
    let dev = dev0()?;
    load_module(&dev)?;
    let stream = fork_blocking_stream(&dev);
    let gf = |n: &str| dev.get_func(MODULE, n).ok_or_else(|| anyhow::anyhow!("fn {n} missing"));
    let f_fx = gf("exl3_hmma_gemm_ks_fx")?;
    let f_probe = gf("exl3_hmma_gemm_ks_fx_probe")?;
    println!(
        "=== EXL3 fixed-cost ladder (dense ks_fx; {} iters/rep, best of 3; alias window {} kb; roofline 238 GB/s) ===",
        iters, window_kb
    );
    let mut json = String::from("[\n");
    let mut seen = std::collections::HashSet::new();
    let mut mods: Vec<&ModuleMeta> = pack
        .modules
        .iter()
        .filter(|m| matches!(m.class, TensorClass::Dense))
        .filter(|m| m.k >= 1024 && m.n >= 128 && seen.insert((m.k, m.n, m.k2 / 2)))
        .collect();
    mods.sort_by_key(|m| (m.class == TensorClass::LmHead, m.n, m.k));
    for m in &mods {
        if skip_n > 0 && m.n > skip_n {
            println!("# skip {} class={:?} k={} n={} (> --exl3-ladder-skip-n)", m.name, m.class, m.k, m.n);
            continue;
        }
        println!("# module {} class={:?} k={} n={} bits={}", m.name, m.class, m.k, m.n, m.k2 / 2);
        for k in [m.k, m.k / 2] {
            let kb = k / 16;
            if kb == 0 || k % 16 != 0 {
                continue;
            }
            // Served ks policy (exl3_forward::chain_ks_target, target 200): bench it first,
            // then the manual sweep for context.
            let ks_served = crate::exl3_forward::chain_ks_target(k as i32, m.n as i32, 200) as usize;
            let mut ks_all: Vec<usize> = vec![ks_served];
            for &kx in kss {
                if kx != ks_served && !ks_all.contains(&kx) {
                    ks_all.push(kx);
                }
            }
            let nb = m.n / 16;
            if nb == 0 {
                continue;
            }
            let tr_bytes = (kb * nb * 8 * m.k2 / 2 * 2) as f64;
            // DRAM-cold rotation: J copies of the trellis, J*tr_bytes >= ~192 MB, so every
            // rotated launch reads memory L2 no longer holds (a single tensor < L2 would make
            // the plain loop measure the L2-warm rate -- the S-C4STEP5 E1 trap).
            let j = (((192usize * 1024 * 1024) as f64 / tr_bytes).ceil() as usize)
                .max(2)
                .min(((1536usize * 1024 * 1024) as f64 / tr_bytes).floor() as usize)
                .max(2);
            // ring words are u32 (WRING = 8*BITS per row): allocate u16 x2 of the old formula
            // (the 2x-undersized buffer walked off the end on down_proj k=17408, 2.0bpw crash)
            let tru16: Vec<u16> = vec![0x6B8Du16; kb * nb * 8 * m.k2 / 2 * 2];
            let d_trs: Vec<_> = (0..j).map(|_| dev.htod_sync_copy(&tru16).map_err(anyhow::Error::from)).collect::<Result<Vec<_>>>()?;
            for &ks in &ks_all {
                if kb % ks != 0 || ks == 0 {
                    continue;
                }
                for &mm in ms {
                    let xh: Vec<u16> = vec![0x3C00; mm.max(1) * k];
                    let d_xh = dev.htod_sync_copy(&xh)?;
                    let svh: Vec<u16> = vec![0x3C00; m.n];
                    let d_svh = dev.htod_sync_copy(&svh)?;
                    let d_ws = unsafe { dev.alloc::<f32>(ks * mm.max(1) * m.n) }?;
                    let d_y = unsafe { dev.alloc::<u16>(mm.max(1) * m.n) }?;
                    let d_cnt = dev.htod_sync_copy(&vec![0u32; m.n / 128])?;
                    let cfg = LaunchConfig {
                        grid_dim: ((m.n / 128) as u32, ks as u32, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    // rotate=true walks the J trellis copies (DRAM-cold); false pins buffer 0
                    // (the L2-resident arm). The alias arm always pins (its window IS the point).
                    let time_arm = |mode: i32, wkb: i32, rotate: bool| -> Result<f64> {
                        let launch = |i: usize| -> Result<()> {
                            let d_tr = &d_trs[if rotate { i % d_trs.len() } else { 0 }];
                            unsafe {
                                if mode == 0 {
                                    f_fx.clone().launch_on_stream(
                                        &stream, cfg,
                                        (d_tr, &d_xh, &d_ws, &d_svh, &d_y, &d_cnt,
                                         mm as i32, k as i32, m.n as i32, m.k2 as i32),
                                    )?;
                                } else {
                                    f_probe.clone().launch_on_stream(
                                        &stream, cfg,
                                        (d_tr, &d_xh, &d_ws, &d_svh, &d_y, &d_cnt,
                                         mm as i32, k as i32, m.n as i32, m.k2 as i32, mode, wkb),
                                    )?;
                                }
                            }
                            Ok(())
                        };
                        for i in 0..3 { launch(i)?; }
                        dev.synchronize()?;
                        let mut best = f64::INFINITY;
                        for _ in 0..3 {
                            let t0 = std::time::Instant::now();
                            for i in 0..iters { launch(i)?; }
                            dev.synchronize()?;
                            best = best.min(t0.elapsed().as_secs_f64() / iters as f64);
                        }
                        Ok(best)
                    };
                    let jrot = if d_trs.len() > 1 { "rot" } else { "pin" };
                    let t_n = time_arm(0, 0, true)?;
                    let t_a = time_arm(1, window_kb as i32, false)?;
                    let t_s = time_arm(2, 0, true)?;
                    let gbs = tr_bytes / t_n / 1e9;
                    println!(
                        "  {:<26} {:>6} b{} k={:<5} n={:<6} ks={} gx={:<4} m={:<2} | norm {:>7.3} ms {:>6.1} GB/s | alias {:>7.3} ms ({:>5.1}%) | stub {:>7.3} ms ({:>5.1}%)",
                        m.name, format!("{:?}/{}", m.class, jrot), m.k2 / 2, k, m.n, ks, m.n / 128, mm,
                        t_n * 1e3, gbs, t_a * 1e3, 100.0 * (t_n - t_a) / t_n, t_s * 1e3, 100.0 * (t_n - t_s) / t_n
                    );
                    json.push_str(&format!(
                        "{{\"name\":\"{}\",\"class\":\"{:?}\",\"bits\":{},\"k\":{},\"n\":{},\"ks\":{},\"gx\":{},\"m\":{},\"ms_norm\":{:.6},\"gbs_norm\":{:.2},\"ms_alias\":{:.6},\"ms_stub\":{:.6},\"tr_bytes\":{:.0}}},\n",
                        m.name, m.class, m.k2 / 2, k, m.n, ks, m.n / 128, mm, t_n * 1e3, gbs, t_a * 1e3, t_s * 1e3, tr_bytes
                    ));
                }
            }
        }
    }
    json.push_str("]\n");
    if let Some(path) = out_json {
        std::fs::write(path, &json).context("write ladder json")?;
        println!("ladder json -> {path}");
    }
    Ok(())
}

pub fn bench_gemm(
    dir: &str,
    widths: &[usize],
    iters: usize,
    grouped_e: usize,
    l2_alias_mb: usize,
) -> Result<()> {
    let pack = Exl3Pack::open(dir)?;
    let dev = dev0()?;
    load_module(&dev)?;

    println!(
        "=== EXL3 weight-path GEMM bench ({} iters/rep, best of 3; roofline 238 GB/s, naive floor 79-83 GB/s) ===",
        iters
    );
    for m in pick_modules(&pack) {
        let raw = m.trellis.read_bytes(&pack.dir)?;
        let tr_bits: Vec<u16> = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let suh = le_f16_bits(&m.suh.read_bytes(&pack.dir)?);
        let svh = le_f16_bits(&m.svh.read_bytes(&pack.dir)?);
        let label = m
            .name
            .rsplit('.')
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join(".");
        for wd in widths {
            let x = synth_x(*wd, m.k, 0xBEEF);
            let mut chain =
                run_chain(&dev, &tr_bits, &suh, &svh, &x, *wd, m.k, m.n, m.k2)?;
            let ref stream = fork_blocking_stream(&dev);
            let gf = |n: &str| {
                dev.get_func(MODULE, n)
                    .ok_or_else(|| anyhow::anyhow!("fn {n} missing"))
            };
            let f_gemm = gf("exl3_hmma_gemm")?;
            let f_suh = gf("exl3_had_suh")?;
            let f_svh = gf("exl3_had_svh")?;
            let cfg_g = LaunchConfig {
                grid_dim: ((m.n / 128) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let cfg_v = LaunchConfig {
                grid_dim: ((*wd * m.k / 128) as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            let cfg_o = LaunchConfig {
                grid_dim: ((*wd * m.n / 128) as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            // warmup + best-of-3 rounds of `iters`
            for _ in 0..3 {
                bench_launch(stream, &f_suh, &f_gemm, &f_svh, &mut chain, m.k, m.n, *wd, false, cfg_g, cfg_v, cfg_o);
            }
            dev.synchronize().unwrap();
            let mut best_g = f64::INFINITY;
            let mut best_c = f64::INFINITY;
            for _ in 0..3 {
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    bench_launch(stream, &f_suh, &f_gemm, &f_svh, &mut chain, m.k, m.n, *wd, false, cfg_g, cfg_v, cfg_o);
                }
                dev.synchronize().unwrap();
                best_g = best_g.min(t0.elapsed().as_secs_f64() / iters as f64);

                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    bench_launch(stream, &f_suh, &f_gemm, &f_svh, &mut chain, m.k, m.n, *wd, true, cfg_g, cfg_v, cfg_o);
                }
                dev.synchronize().unwrap();
                best_c = best_c.min(t0.elapsed().as_secs_f64() / iters as f64);
            }
            let tr_bytes = (m.trellis.n_bytes) as f64;
            let aux_bytes = (2 * *wd * m.k + 2 * *wd * m.n) as f64; // x + xh read, yraw w + y w/r
            println!(
                "  {:<20} b={} W={:<2}  gemm {:>7.3} ms  {:>7.1} GB/s (trellis) | chain {:>7.3} ms  {:>7.1} GB/s",
                label,
                m.k2 / 2,
                wd,
                best_g * 1e3,
                tr_bytes / best_g / 1e9,
                best_c * 1e3,
                tr_bytes / best_c / 1e9
            );
            let _ = aux_bytes;
            if m.class == TensorClass::LmHead {
                // W3/LMH: the persistent kernel on the full head at this width
                bench_lmh(&dev, &chain.d_tr, &chain.d_svh, m.k, m.n, m.k2 / 2, *wd, iters, tr_bytes, "lm_head")?;
            }
        }
        if m.class == TensorClass::LmHead {
            // W3/LMH: the 65,536-id draft slice (the MTP draft pass's head)
            let (trd, nd) = slab_trellis(&raw, m.k / 16, m.n / 16, m.k2, Some(65536));
            let trd16: Vec<u16> = trd.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let d_draft = dev.htod_sync_copy(&trd16)?;
            let d_svh_draft = dev.htod_sync_copy(&svh[..nd])?;
            for wd in widths {
                bench_lmh(&dev, &d_draft, &d_svh_draft, m.k, nd, m.k2 / 2, *wd, iters,
                          (trd.len()) as f64, "draft 65536")?;
            }
        }
    }
    // ---- grouped-expert bench (the MoE serving form): E experts, one launch ----
    // All experts of a group share the shape (gate/up: 2560->640 b3, down: 640->2560 b3).
    // xh here is synthetic activation content (per-expert suh handling is S-A3-d scope —
    // noted in the bank); the bench measures the weight-streaming path: bytes / time.
    let ge = if grouped_e > 0 { grouped_e } else { 512 };
    for suffix in ["gate_proj", "down_proj"] {
        let mods: Vec<ModuleMeta> = pack
            .modules
            .iter()
            .filter(|m| m.class == TensorClass::Expert && m.k2 / 2 == 3 && m.name.ends_with(suffix))
            .take(ge)
            .cloned()
            .collect();
        if mods.len() < 2 {
            continue;
        }
        let m0 = &mods[0];
        // Single concatenated trellis buffer + per-expert ELEMENT offsets (no raw
        // pointer table; one allocation, contiguous streaming).
        let mut concat: Vec<u16> = Vec::new();
        let mut offs: Vec<u64> = Vec::with_capacity(mods.len());
        for m in &mods {
            let raw = m.trellis.read_bytes(&pack.dir)?;
            offs.push(concat.len() as u64);
            concat.extend(raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])));
        }
        let d_base = dev.htod_sync_copy(&concat)?;
        // S-A3-f-a X3 (L2-resident discriminator): --exl3-l2-alias [MB] folds the expert
        // offset table so every CTA reads one of the first alias_e trellis experts — the
        // working set (~16 MB default) fits L2, so the trellis loads stop paying DRAM
        // latency. Kernel, grid and launch config are UNCHANGED; only the offs CONTENT
        // differs. Under alias the GB/s figure is by construction NOT DRAM traffic; the
        // decision metric is the 16x16 block rate (printed on the [L2-ALIAS] line).
        let d_offs = if l2_alias_mb > 0 {
            let alias_e = ((l2_alias_mb * 1024 * 1024) / m0.trellis.n_bytes).clamp(1, mods.len());
            let offs_alias: Vec<u64> = (0..mods.len()).map(|e| offs[e % alias_e]).collect();
            println!(
                "  [L2-ALIAS] {} experts -> first {} (footprint {:.1} MB); GB/s below is NOT DRAM traffic",
                mods.len(),
                alias_e,
                alias_e as f64 * m0.trellis.n_bytes as f64 / 1e6
            );
            dev.htod_sync_copy(&offs_alias)?
        } else {
            dev.htod_sync_copy(&offs)?
        };
        let tr_bytes_total = (mods.len() * m0.trellis.n_bytes) as f64;
        println!(
            "  grouped {:>9} x{}  b={} K={:<5} N={:<6}  ({:.1} MB/launch)",
            suffix,
            mods.len(),
            m0.k2 / 2,
            m0.k,
            m0.n,
            tr_bytes_total / 1e6
        );
        for wd in widths {
            let x = synth_x(*wd, m0.k, 0xBEEF);
            let d_xh = dev.htod_sync_copy(&x)?;
            let mut d_y = unsafe { dev.alloc::<u16>(mods.len() * wd * m0.n) }?;
            let f_g = dev
                .get_func(MODULE, "exl3_hmma_gemm_grouped")
                .ok_or_else(|| anyhow::anyhow!("fn exl3_hmma_gemm_grouped missing"))?;
            let stream = fork_blocking_stream(&dev);
            let mut launch = || unsafe {
                f_g.clone()
                    .launch_on_stream(
                        &stream,
                        LaunchConfig {
                            grid_dim: ((mods.len() * m0.n / 128) as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            // S-A3-n F3: the A-once body's smem (mirrors exl3_a1_fits)
                            shared_mem_bytes: {
                                let b = *wd * (m0.k + 8) * 2;
                                if *wd <= 16 && m0.k % 8 == 0 && b <= 48 * 1024 { b as u32 } else { 0 }
                            },
                        },
                        (
                            &d_base,
                            &d_offs,
                            &d_xh,
                            &mut d_y,
                            *wd as i32,
                            m0.k as i32,
                            m0.n as i32,
                            (m0.k2 / 2) as i32,
                        ),
                    )
                    .unwrap();
            };
            for _ in 0..3 {
                launch();
            }
            dev.synchronize().unwrap();
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    launch();
                }
                dev.synchronize().unwrap();
                best = best.min(t0.elapsed().as_secs_f64() / iters as f64);
            }
            // mapping sanity: expert 0 and expert 1 row 0 must differ (different weights).
            // Skipped under L2-ALIAS: experts alias to shared weights by construction.
            let y0 = dev.dtoh_sync_copy(&d_y)?;
            let n0 = m0.n;
            let same = y0[..n0] == y0[n0 * *wd..n0 * *wd + n0];
            if l2_alias_mb > 0 {
                // The X3 decision metric: 16x16 trellis blocks decoded per second.
                let blocks = (mods.len() * m0.n * m0.k / 256) as f64;
                println!(
                    "    [L2-ALIAS] W={:<2} {:>7.3} ms  block rate {:>6.3} G blocks/s  (mapping-check skipped)",
                    wd,
                    best * 1e3,
                    blocks / best / 1e9
                );
            } else {
                println!(
                    "    W={:<2} gemm {:>7.3} ms  {:>7.1} GB/s (aggregate trellis, {} experts)  mapping-check: {}",
                    wd,
                    best * 1e3,
                    tr_bytes_total / best / 1e9,
                    mods.len(),
                    if same { "FAIL (expert rows identical!)" } else { "ok (expert rows differ)" }
                );
                if same {
                    bail!("grouped mapping check FAILED on {suffix}");
                }
            }
        }
    }
    println!("EXL3-GEMM-BENCH: done");
    Ok(())
}

/// W3/LMH bench: old exl3_hmma_gemm (and + exl3_had_svh) vs the persistent kernel (y_raw, and
/// fused -> logits, every ring depth) at one m. Back-to-back launches on one stream, 3 warmups,
/// best of 3 x `iters`. GB/s = trellis bytes / time (the 238 GB/s floor is the target).
#[allow(clippy::too_many_arguments)]
fn bench_lmh(dev: &std::sync::Arc<CudaDevice>, d_tr: &CudaSlice<u16>, d_svh: &CudaSlice<u16>,
             k: usize, n: usize, bits: usize, m: usize, iters: usize, tr_bytes: f64, label: &str)
             -> Result<()> {
    let stream = fork_blocking_stream(dev);
    let x = synth_x(m, k, 0xBEEF);
    let d_xh = dev.htod_sync_copy(&x)?;
    let mut d_yr = dev.htod_sync_copy(&vec![0u16; m * n])?;
    let mut d_y = dev.htod_sync_copy(&vec![0u16; m * n])?;
    let f_old = dev.get_func(MODULE, "exl3_hmma_gemm").context("exl3_hmma_gemm")?;
    let f_svh = dev.get_func(MODULE, "exl3_had_svh").context("exl3_had_svh")?;
    let cfg_g = LaunchConfig { grid_dim: ((n / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let cfg_o = LaunchConfig { grid_dim: ((m * n / 128) as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
    let (p_tr, p_xh, p_svh) = (*d_tr.device_ptr(), *d_xh.device_ptr(), *d_svh.device_ptr());
    let (p_yr, p_y) = (*d_yr.device_ptr(), *d_y.device_ptr());
    // variant: 0 = old gemm, 1 = old gemm + had_svh, 2.. = LMH (nst, fuse)
    let mut variants: Vec<(String, usize, bool)> = vec![("old gemm".into(), 0, false), ("old gemm+svh".into(), 0, true)];
    let nsts: &[usize] = if bits == 5 { &[4, 6, 8] } else { &[6] };
    for &nst in nsts {
        variants.push((format!("LMH s{nst}"), nst, false));
        variants.push((format!("LMH s{nst} fused"), nst, true));
    }
    let mut line = format!("  {label:<12} b={bits} W={m:<2}");
    for (name, nst, flag) in &variants {
        let mut run = |_: ()| -> Result<()> {
            if *nst == 0 {
                unsafe {
                    f_old.clone().launch_on_stream(&stream, cfg_g,
                        (d_tr, &d_xh, &mut d_yr, m as i32, k as i32, n as i32, (2 * bits) as i32))?;
                    if *flag {
                        f_svh.clone().launch_on_stream(&stream, cfg_o, (&d_yr, d_svh, &mut d_y, n as i32))?;
                    }
                }
            } else {
                let out = if *flag { p_y } else { p_yr };
                if !crate::exl3_forward::lmh_gemm_launch(dev, stream.stream, p_tr, p_xh, out, p_svh,
                                                          m, k, n, bits, *nst, *flag)? {
                    bail!("W3/LMH bench: launcher refused m={m} K={k} N={n} b={bits} nst={nst}");
                }
            }
            Ok(())
        };
        for _ in 0..3 { run(())?; }
        dev.synchronize()?;
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            for _ in 0..iters { run(())?; }
            dev.synchronize()?;
            best = best.min(t0.elapsed().as_secs_f64() / iters as f64);
        }
        let gbs = tr_bytes / best / 1e9;
        line.push_str(&format!(" | {name} {:.3} ms {:.1} GB/s ({:.1}%)", best * 1e3, gbs, gbs / 238.0 * 100.0));
    }
    let _ = p_y;
    println!("{line}");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn bench_launch(
    stream: &CudaStream,
    f_suh: &cudarc::driver::CudaFunction,
    f_gemm: &cudarc::driver::CudaFunction,
    f_svh: &cudarc::driver::CudaFunction,
    c: &mut Chain,
    k: usize,
    n: usize,
    w: usize,
    full_chain: bool,
    cfg_g: LaunchConfig,
    cfg_v: LaunchConfig,
    cfg_o: LaunchConfig,
) {
    if full_chain {
        unsafe {
            f_suh
                .clone()
                .launch_on_stream(stream, cfg_v, (&c.d_x, &c.d_suh, &mut c.d_xh, k as i32))
                .unwrap();
        }
    }
    unsafe {
        f_gemm
            .clone()
            .launch_on_stream(stream, cfg_g, (&c.d_tr, &c.d_xh, &mut c.d_yraw, w as i32, k as i32, n as i32, c.k2 as i32))
            .unwrap();
    }
    if full_chain {
        unsafe {
            f_svh
                .clone()
                .launch_on_stream(stream, cfg_o, (&c.d_yraw, &c.d_svh, &mut c.d_y, n as i32))
                .unwrap();
        }
    }
}

