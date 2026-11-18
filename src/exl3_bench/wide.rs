//! The wide-GEMM probe and bench, and the cross-chain probe.
//! Probe-only code of the `exl3_bench` module (see `mod.rs` for the map); shared helpers come from the parent.
//! Numerics contract (module header in `mod.rs`): the kernel decode is bitwise vs the host oracle; do not change reduction orders.

use super::*;

pub fn probe_wide(dir: &str, ms: &[usize], mts: &[usize], slab_cols: usize) -> Result<()> {
    let pack = Exl3Pack::open(dir)?;
    let dev = dev0()?;
    load_module(&dev)?;
    let stream = fork_blocking_stream(&dev);

    for m in pick_modules(&pack) {
        let raw = m.trellis.read_bytes(&pack.dir)?;
        let (trb, n_eff) = if m.class == TensorClass::LmHead {
            slab_trellis(&raw, m.k / 16, m.n / 16, m.k2, Some(slab_cols))
        } else {
            (raw.clone(), m.n)
        };
        let suh = le_f16_bits(&m.suh.read_bytes(&pack.dir)?);
        let k = m.k;
        let bits = m.k2 / 2;
        let tr_bits: Vec<u16> = trb
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        // one host decode per module; the wide kernel must reproduce it through
        // the mma path (rel-L2) and the m16 kernel bitwise (structural).
        let w_dec = decode_trellis_host(
            &tr_bits.iter().map(|v| *v as i16).collect::<Vec<_>>(),
            k / 16,
            n_eff / 16,
            m.k2, // k2-keyed (P7 B8: bits*2 == m.k2 only for integer widths)
        );
        println!(
            "  {:<24} b={} K={:<5} N={:<6}",
            m.name.rsplit('.').take(2).collect::<Vec<_>>().join("."),
            m.k2 as f64 / 2.0,
            k,
            n_eff
        );
        let f_wide = dev
            .get_func(MODULE, "exl3_hmma_gemm_wide")
            .ok_or_else(|| anyhow::anyhow!("fn exl3_hmma_gemm_wide missing"))?;
        let f_m16 = dev
            .get_func(MODULE, "exl3_hmma_gemm")
            .ok_or_else(|| anyhow::anyhow!("fn exl3_hmma_gemm missing"))?;
        let d_tr = dev.htod_sync_copy(&tr_bits)?;
        for &mm in ms {
            let x = synth_x(mm, k, 0xC0FFEE);
            let xh = had_scale_ref(&x, &suh, mm, k, true);
            let d_xh = dev.htod_sync_copy(&xh)?;
            let mut d_y = unsafe { dev.alloc::<u16>(mm * n_eff) }?;
            let mut first: Vec<u16> = Vec::new();
            let mut rel_report = 0.0f64;
            // (c-first) BITWISE vs the ORIGINAL m16 kernel at M=16 — run before any
            // tolerance check so a decode/indexing defect is diagnosed directly.
            if mm == 16 {
                // S-A3-f-b param-echo bisect: dump the args the kernel RECEIVES.
                {
                    let echo_in: Vec<u16> = vec![0x7777u16; 64];
                    let mut d_echo = dev.htod_sync_copy(&echo_in)?;
                    unsafe {
                        f_wide
                            .clone()
                            .launch_on_stream(
                                &stream,
                                LaunchConfig {
                                    grid_dim: (1, 1, 1),
                                    block_dim: (32, 1, 1),
                                    shared_mem_bytes: 0,
                                },
                                (&d_tr, &d_xh, &mut d_echo, 16i32, k as i32, n_eff as i32, bits as i32, -1i32),
                            )
                            .unwrap();
                    }
                    dev.synchronize().unwrap();
                    let eh: Vec<u16> = dev.dtoh_sync_copy(&d_echo)?;
                    let wd = |i: usize| eh[i] as u64;
                    let ptr = |o: usize| wd(o) | (wd(o + 1) << 16) | (wd(o + 2) << 32) | (wd(o + 3) << 48);
                    let sc = |i: usize| i32::from(eh[12 + i]) | (i32::from(eh[13 + i]) << 16);
                    println!(
                        "    echo: trellis={:#x} xh={:#x} y={:#x} M={} K={} N={} bits={} mt={}",
                        ptr(0), ptr(4), ptr(8), sc(0), sc(2), sc(4), sc(6), sc(8)
                    );
                    println!(
                        "    host: trellis={:#x} xh={:#x}",
                        *d_tr.device_ptr(),
                        *d_xh.device_ptr()
                    );
                }
                let mut d_y16 = unsafe { dev.alloc::<u16>(16 * n_eff) }?;
                unsafe {
                    f_m16
                        .clone()
                        .launch_on_stream(
                            &stream,
                            LaunchConfig {
                                grid_dim: ((n_eff / 128) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&d_tr, &d_xh, &mut d_y16, 16i32, k as i32, n_eff as i32, (bits * 2) as i32),
                        )
                        .unwrap();
                }
                dev.synchronize().unwrap();
                let y16 = dev.dtoh_sync_copy(&d_y16)?;
                // POISON: the wide target starts as 0xDEAD — if the launch writes
                // nothing, every 0xDEAD survives (allocator-stable garbage would
                // otherwise mimic a "wrong but real" kernel).
                let poison = vec![0xDEADu16; 16 * n_eff];
                let d_poison = dev.htod_sync_copy(&poison)?;
                drop(d_y);
                d_y = d_poison;
                unsafe {
                    f_wide
                        .clone()
                        .launch_on_stream(
                            &stream,
                            LaunchConfig {
                                grid_dim: (1u32, (n_eff / 128) as u32, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&d_tr, &d_xh, &mut d_y, 16i32, k as i32, n_eff as i32, bits as i32, 1i32),
                        )
                        .unwrap();
                }
                dev.synchronize().unwrap();
                let yw = dev.dtoh_sync_copy(&d_y)?;
                let survivors = yw.iter().filter(|v| **v == 0xDEAD).count();
                let mut bad = 0usize;
                for (i, (a, b)) in y16.iter().zip(yw.iter()).enumerate() {
                    if a != b {
                        bad += 1;
                        if bad <= 6 {
                            println!(
                                "    wide-vs-m16 [{i}] (row {}, col {}): m16 {:04x} wide {:04x}",
                                i / n_eff,
                                i % n_eff,
                                a,
                                b
                            );
                        }
                    }
                }
                println!(
                    "    M=16 m16-vs-wide checksum: m16 {:016x} wide {:016x} mismatches {bad}/{}  poison_survivors {survivors}",
                    y16.iter().fold(0u64, |s, v| s.wrapping_mul(31).wrapping_add(*v as u64)),
                    yw.iter().fold(0u64, |s, v| s.wrapping_mul(31).wrapping_add(*v as u64)),
                    y16.len()
                );
                if bad > 0 {
                    bail!("EXL3-WIDE FAIL on {} M=16: wide != m16 kernel bitwise ({bad})", m.name);
                }
            }
            for &mt in mts {
                let msup = (mm + 16 * mt - 1) / (16 * mt);
                unsafe {
                    f_wide
                        .clone()
                        .launch_on_stream(
                            &stream,
                            LaunchConfig {
                                grid_dim: (msup as u32, (n_eff / 128) as u32, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &d_tr,
                                &d_xh,
                                &mut d_y,
                                mm as i32,
                                k as i32,
                                n_eff as i32,
                                bits as i32,
                                mt as i32,
                            ),
                        )
                        .unwrap();
                }
                dev.synchronize().unwrap();
                let y = dev.dtoh_sync_copy(&d_y)?;
                // (b) bitwise across MT — the structural exactness check
                if first.is_empty() {
                    first = y.clone();
                } else if first[..] != y[..] {
                    let mut bad = 0usize;
                    for (i, (a, b)) in first.iter().zip(y.iter()).enumerate() {
                        if a != b {
                            bad += 1;
                            if bad < 4 {
                                println!(
                                    "    mt={mt} mismatch [{i}]: {:04x} vs mt={} {:04x}",
                                    b, mts[0], a
                                );
                            }
                        }
                    }
                    bail!(
                        "EXL3-WIDE FAIL on {} M={} mt={mt}: {bad} bitwise mismatches vs mt={}",
                        m.name,
                        mm,
                        mts[0]
                    );
                }
                // (a) rel-L2 vs fp16(host ref) on the first min(M,16) rows
                let rref = mm.min(16);
                let yref = gemm_ref(&w_dec, &xh, rref, k, n_eff);
                let yref16: Vec<f32> = yref.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
                let (_, rel16) = rel_l2(&y[..rref * n_eff], &yref16);
                rel_report = rel16;
                // Sanity threshold only (the REAL gate is the bitwise checks above):
                // single-row draws straddle the fp16(ref) rounding boundary — measured
                // 1.028e-4 on the lm_head slab b5 at M=1 with the m16 kernel itself
                // bitwise-green on the same data (S-A3-c: structural bugs are O(1)).
                let rel_cap = if rref < 4 { 2.5e-4 } else { 1e-4 };
                if rel16 > rel_cap {
                    bail!(
                        "EXL3-WIDE FAIL on {} M={} mt={mt}: y_raw rel-L2 {:.3e} > {}",
                        m.name,
                        mm,
                        rel16,
                        rel_cap
                    );
                }
            }
            // (c) M=16: bitwise vs the ORIGINAL m16 kernel (same x, same xh)
            if mm == 16 {
                let mut d_y16 = unsafe { dev.alloc::<u16>(16 * n_eff) }?;
                unsafe {
                    f_m16
                        .clone()
                        .launch_on_stream(
                            &stream,
                            LaunchConfig {
                                grid_dim: ((n_eff / 128) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&d_tr, &d_xh, &mut d_y16, 16i32, k as i32, n_eff as i32, (bits * 2) as i32),
                        )
                        .unwrap();
                }
                dev.synchronize().unwrap();
                let y16 = dev.dtoh_sync_copy(&d_y16)?;
                if y16[..] != first[..] {
                    let mut bad = 0usize;
                    for (i, (a, b)) in y16.iter().zip(first.iter()).enumerate() {
                        if a != b {
                            bad += 1;
                            if bad < 4 {
                                println!("    wide-vs-m16 mismatch [{i}]: {:04x} vs {:04x}", b, a);
                            }
                        }
                    }
                    bail!(
                        "EXL3-WIDE FAIL on {} M=16: wide != m16 kernel bitwise ({bad} mismatches)",
                        m.name
                    );
                }
            }
            println!(
                "    M={:<4} mts {:?}: PASS (bitwise across MT{}, rel-L2 vs fp16 ref {:.2e})",
                mm,
                mts,
                if mm == 16 { " + vs m16" } else { "" },
                rel_report
            );
        }
    }
    // P7 B5: synthetic K=7 leg — no real 7-bit pack exists (the loader refuses one),
    // so the 7-bit wide arm is proven on a synthesized trellis: wide (bits=7) must be
    // bitwise-identical to the m16 kernel (k2=14, case 14) and rel-L2-close to the
    // host f16 reference. Poison canary catches a silent no-op arm (the old
    // `default: break` failure class this campaign exists to kill).
    {
        const K7_BITS: usize = 7;
        const K7_K2: usize = 14;
        let k: usize = 512;
        let n_eff: usize = 1024;
        let (kb, nb) = (k / 16, n_eff / 16);
        let len_u16 = kb * nb * 8 * K7_K2; // trellis_to_ring: 8*k2 u16 per 16x16 block
        println!("  {:<24} b={} K={:<5} N={:<6} (synthetic)", "synth7", K7_BITS, k, n_eff);
        let tr_bits: Vec<u16> = (0..len_u16)
            .map(|i| ((i.wrapping_mul(2654435761)) >> 11) as u16)
            .collect();
        let w_dec = decode_trellis_host(
            &tr_bits.iter().map(|v| *v as i16).collect::<Vec<_>>(),
            kb,
            nb,
            K7_K2,
        );
        let x = synth_x(16, k, 0x7EED7);
        let suh: Vec<u16> = (0..k)
            .map(|i| f16::from_f32(0.25 + (i % 13) as f32 * 0.03125).to_bits())
            .collect();
        let xh = had_scale_ref(&x, &suh, 16, k, true);
        let f_wide = dev
            .get_func(MODULE, "exl3_hmma_gemm_wide")
            .ok_or_else(|| anyhow::anyhow!("fn exl3_hmma_gemm_wide missing"))?;
        let f_m16 = dev
            .get_func(MODULE, "exl3_hmma_gemm")
            .ok_or_else(|| anyhow::anyhow!("fn exl3_hmma_gemm missing"))?;
        let d_tr = dev.htod_sync_copy(&tr_bits)?;
        let d_xh = dev.htod_sync_copy(&xh)?;
        // poison canary: every element must be overwritten by the wide arm
        let poison: Vec<u16> = vec![0xDEADu16; 16 * n_eff];
        let mut d_y = dev.htod_sync_copy(&poison)?;
        unsafe {
            f_wide
                .clone()
                .launch_on_stream(
                    &stream,
                    LaunchConfig {
                        grid_dim: (1, (n_eff / 128) as u32, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&d_tr, &d_xh, &mut d_y, 16i32, k as i32, n_eff as i32, K7_BITS as i32, 1i32), // mt>=1: kernel loops t in 0..mt
                )
                .unwrap();
        }
        let mut d_y16 = unsafe { dev.alloc::<u16>(16 * n_eff) }?;
        unsafe {
            f_m16
                .clone()
                .launch_on_stream(
                    &stream,
                    LaunchConfig {
                        grid_dim: ((n_eff / 128) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&d_tr, &d_xh, &mut d_y16, 16i32, k as i32, n_eff as i32, K7_K2 as i32),
                )
                .unwrap();
        }
        dev.synchronize().unwrap();
        let y = dev.dtoh_sync_copy(&d_y)?;
        let y16 = dev.dtoh_sync_copy(&d_y16)?;
        let survivors = y.iter().filter(|&&v| v == 0xDEAD).count();
        if survivors != 0 {
            bail!(
                "EXL3-WIDE FAIL on synth7 K=7: wide arm left {survivors}/{} poison elements unwritten (no-op arm?)",
                16 * n_eff
            );
        }
        if y[..] != y16[..] {
            let mut bad = 0usize;
            for (i, (a, b)) in y.iter().zip(y16.iter()).enumerate() {
                if a != b {
                    bad += 1;
                }
            }
            bail!(
                "EXL3-WIDE FAIL on synth7 K=7: wide != m16 kernel bitwise ({bad} mismatches)"
            );
        }
        let yref = gemm_ref(&w_dec, &xh, 16, k, n_eff);
        let yref16: Vec<f32> = yref.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
        let (_, rel16) = rel_l2(&y[..16 * n_eff], &yref16);
        if rel16 > 1e-4 {
            bail!("EXL3-WIDE FAIL on synth7 K=7: y_raw rel-L2 {rel16:.3e} > 1e-4");
        }
        println!(
            "    K=7 synthetic: PASS (bitwise wide==m16, poison 0, rel-L2 vs fp16 ref {:.2e})",
            rel16
        );

        // PR#27-D3: synthetic K=8 wide cell — same shape as the K=7 leg above, on the
        // D2-added case-8 arms. A real 8-bit pack has never existed on any box; this cell
        // is the ONLY K=8 oracle receipt (matrix row A col 8 + rows D/F col 8).
        {
            const K8_BITS: usize = 8;
            const K8_K2: usize = 16;
            let k: usize = 512;
            let n_eff: usize = 1024;
            let (kb, nb) = (k / 16, n_eff / 16);
            let len_u16 = kb * nb * 8 * K8_K2;
            println!("  {:<24} b={} K={:<5} N={:<6} (synthetic)", "synth8", K8_BITS, k, n_eff);
            let tr_bits: Vec<u16> = (0..len_u16)
                .map(|i| ((i.wrapping_mul(2654435761)) >> 11) as u16)
                .collect();
            let w_dec = decode_trellis_host(
                &tr_bits.iter().map(|v| *v as i16).collect::<Vec<_>>(),
                kb,
                nb,
                K8_K2,
            );
            let x = synth_x(16, k, 0x7EED8);
            let suh: Vec<u16> = (0..k)
                .map(|i| f16::from_f32(0.25 + (i % 13) as f32 * 0.03125).to_bits())
                .collect();
            let xh = had_scale_ref(&x, &suh, 16, k, true);
            let f_wide = dev
                .get_func(MODULE, "exl3_hmma_gemm_wide")
                .ok_or_else(|| anyhow::anyhow!("fn exl3_hmma_gemm_wide missing"))?;
            let f_m16 = dev
                .get_func(MODULE, "exl3_hmma_gemm")
                .ok_or_else(|| anyhow::anyhow!("fn exl3_hmma_gemm missing"))?;
            let d_tr = dev.htod_sync_copy(&tr_bits)?;
            let d_xh = dev.htod_sync_copy(&xh)?;
            let poison: Vec<u16> = vec![0xDEADu16; 16 * n_eff];
            let mut d_y = dev.htod_sync_copy(&poison)?;
            unsafe {
                f_wide
                    .clone()
                    .launch_on_stream(
                        &stream,
                        LaunchConfig {
                            grid_dim: (1, (n_eff / 128) as u32, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&d_tr, &d_xh, &mut d_y, 16i32, k as i32, n_eff as i32, K8_BITS as i32, 1i32),
                    )
                    .unwrap();
            }
            let mut d_y16 = unsafe { dev.alloc::<u16>(16 * n_eff) }?;
            unsafe {
                f_m16
                    .clone()
                    .launch_on_stream(
                        &stream,
                        LaunchConfig {
                            grid_dim: ((n_eff / 128) as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&d_tr, &d_xh, &mut d_y16, 16i32, k as i32, n_eff as i32, K8_K2 as i32),
                    )
                    .unwrap();
            }
            dev.synchronize().unwrap();
            let y = dev.dtoh_sync_copy(&d_y)?;
            let y16 = dev.dtoh_sync_copy(&d_y16)?;
            let survivors = y.iter().filter(|&&v| v == 0xDEAD).count();
            if survivors != 0 {
                bail!(
                    "EXL3-BITWIDTHS FAIL synth8 K=8 wide: {survivors}/{} poison elements unwritten (no-op arm?)",
                    16 * n_eff
                );
            }
            if y[..] != y16[..] {
                bail!("EXL3-BITWIDTHS FAIL synth8 K=8: wide != m16 (case 16) kernel bitwise");
            }
            let yref = gemm_ref(&w_dec, &xh, 16, k, n_eff);
            let yref16: Vec<f32> = yref.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
            let (_, rel16v) = rel_l2(&y[..16 * n_eff], &yref16);
            if rel16v > 1e-4 {
                bail!("EXL3-BITWIDTHS FAIL synth8 K=8: y_raw rel-L2 {rel16v:.3e} > 1e-4");
            }
            println!(
                "    K=8 synthetic: PASS (bitwise wide==m16 case16, poison 0, rel-L2 vs fp16 ref {:.2e})",
                rel16v
            );
        }
        // P7 B6: synthetic K=7 CHAIN leg — exl3_hmma_gemm_ks (k2=14, case 14) + exl3_ks_combine
        // at ks=2. No real 7-bit pack exists (the loader refuses k2=14), so case 14 is proven on
        // the same synthesized trellis: NaN poison on every ws partial slot and 0xDEAD poison on
        // yraw catch a silent no-op (`default: break`, the class this campaign kills), and the
        // end-to-end chain must sit within the wide leg's rel-L2 budget vs the host reference.
        // Association note: split-K (2 chunks + ascending combine) is a different fp32
        // association than the single-pass wide/m16 fold, so this leg gates NUMERICS + liveness
        // — the bitwise identity for body<7> lives in the wide leg above.
        {
            let ks2: u32 = 2;
            let f_ks = dev
                .get_func(MODULE, "exl3_hmma_gemm_ks")
                .ok_or_else(|| anyhow::anyhow!("fn exl3_hmma_gemm_ks missing"))?;
            let f_cb = dev
                .get_func(MODULE, "exl3_ks_combine")
                .ok_or_else(|| anyhow::anyhow!("fn exl3_ks_combine missing"))?;
            let ws_len = ks2 as usize * 16 * n_eff;
            let d_ws = dev.htod_sync_copy(&vec![f32::NAN; ws_len])?;
            unsafe {
                f_ks
                    .clone()
                    .launch_on_stream(
                        &stream,
                        LaunchConfig {
                            grid_dim: ((n_eff / 128) as u32, ks2, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&d_tr, &d_xh, &d_ws, 16i32, k as i32, n_eff as i32, K7_K2 as i32),
                    )
                    .unwrap();
            }
            let mut d_yks = dev.htod_sync_copy(&vec![0xDEADu16; 16 * n_eff])?;
            unsafe {
                f_cb
                    .clone()
                    .launch_on_stream(
                        &stream,
                        LaunchConfig {
                            grid_dim: (((16 * n_eff + 255) / 256) as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&d_ws, &mut d_yks, 16i32, n_eff as i32, ks2 as i32),
                    )
                    .unwrap();
            }
            dev.synchronize().unwrap();
            let ws = dev.dtoh_sync_copy(&d_ws)?;
            let yks = dev.dtoh_sync_copy(&d_yks)?;
            let ws_surv = ws.iter().filter(|&&v| v.is_nan()).count();
            let y_surv = yks.iter().filter(|&&v| v == 0xDEAD).count();
            if ws_surv != 0 || y_surv != 0 {
                bail!(
                    "EXL3-WIDE FAIL on synth7 K=7 chain: poison survivors ws {ws_surv}/{ws_len} y {y_surv}/{} (no-op case 14?)",
                    16 * n_eff
                );
            }
            let (_, relks) = rel_l2(&yks[..16 * n_eff], &yref16[..16 * n_eff]);
            if relks > 1e-4 {
                bail!("EXL3-WIDE FAIL on synth7 K=7 chain: ks=2+combine rel-L2 {relks:.3e} > 1e-4");
            }
            println!(
                "    K=7 synthetic chain: PASS (case 14 body_ks<7> ks=2 + combine, poison 0, rel-L2 vs fp16 ref {:.2e})",
                relks
            );
        }
    }
    println!("EXL3-WIDE: PASS");
    Ok(())
}

// ---------------------------------------------------------------------------
// S-A3-f-b: wide-M GEMM bench (the chunked-prefill weight sweep). GEMM-only
// timing (Hadamards excluded, same convention as bench_gemm's gemm-only rows).
// GB/s = compressed-input (trellis bytes / time, one-way reads, roofline 238);
// rows/s = M / time is the prefill-throughput metric the TTFT table cares about.
//   ./gb10_inference --bench-exl3-wide --model-dir <pack> \
//     [--exl3-wide-ms 32,64,128,256,512,1024] [--exl3-wide-mt 4] [--exl3-iters 20]
// ---------------------------------------------------------------------------
pub fn bench_wide(dir: &str, ms: &[usize], mt: usize, iters: usize) -> Result<()> {
    let pack = Exl3Pack::open(dir)?;
    let dev = dev0()?;
    load_module(&dev)?;
    let stream = fork_blocking_stream(&dev);
    println!(
        "=== EXL3 wide-M GEMM bench (mt={mt}, {iters} iters/rep, best of 3; GB/s = compressed-input trellis bytes/time, roofline 238) ==="
    );
    for m in pick_modules(&pack) {
        let raw = m.trellis.read_bytes(&pack.dir)?;
        let tr_bits: Vec<u16> = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let suh = le_f16_bits(&m.suh.read_bytes(&pack.dir)?);
        let label = m
            .name
            .rsplit('.')
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join(".");
        let d_tr = dev.htod_sync_copy(&tr_bits)?;
        let d_suh = dev.htod_sync_copy(&suh)?;
        let f_wide = dev
            .get_func(MODULE, "exl3_hmma_gemm_wide")
            .ok_or_else(|| anyhow::anyhow!("fn exl3_hmma_gemm_wide missing"))?;
        let f_suh = dev
            .get_func(MODULE, "exl3_had_suh")
            .ok_or_else(|| anyhow::anyhow!("fn exl3_had_suh missing"))?;
        for &mm in ms {
            let x = synth_x(mm, m.k, 0xBEEF);
            let d_x = dev.htod_sync_copy(&x)?;
            let mut d_xh = unsafe { dev.alloc::<u16>(mm * m.k) }?;
            let mut d_y = unsafe { dev.alloc::<u16>(mm * m.n) }?;
            // stage xh once (excluded from timing, same as the chain benches)
            unsafe {
                f_suh
                    .clone()
                    .launch_on_stream(
                        &stream,
                        LaunchConfig {
                            grid_dim: ((mm * m.k / 128) as u32, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&d_x, &d_suh, &mut d_xh, m.k as i32),
                    )
                    .unwrap();
            }
            dev.synchronize().unwrap();
            let mut launch = || unsafe {
                f_wide
                    .clone()
                    .launch_on_stream(
                        &stream,
                        LaunchConfig {
                            grid_dim: (
                                ((mm + 16 * mt - 1) / (16 * mt)) as u32,
                                (m.n / 128) as u32,
                                1,
                            ),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &d_tr,
                            &d_xh,
                            &mut d_y,
                            mm as i32,
                            m.k as i32,
                            m.n as i32,
                            m.k2 as i32,
                            mt as i32,
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
            let gbs = m.trellis.n_bytes as f64 / best / 1e9;
            println!(
                "  {:<10} b={} K={:<5} N={:<6}  M={:<5} {:>7.1} GB/s  {:>9.0} rows/s  {:>7.2} ms",
                label,
                m.k2 / 2,
                m.k,
                m.n,
                mm,
                gbs,
                mm as f64 / best,
                best * 1e3
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 2b. Cross-chain mode: read x [M,K] fp16-u16 from --x-in, run the full chain
// (had_suh -> fused GEMM -> had_svh) on ONE module, write y [M,N] u16 to --y-out.
// The python driver (a3c_xchain.py) builds x from real embeddings on their side
// and compares rel-L2 < 1e-3 (G-A3-2 shape).
// ---------------------------------------------------------------------------
pub fn probe_xchain(dir: &str, x_path: &str, y_path: &str, module: &str) -> Result<()> {
    let pack = Exl3Pack::open(dir)?;
    let dev = dev0()?;
    load_module(&dev)?;
    let m = pack
        .modules
        .iter()
        .find(|mm| mm.name == module)
        .ok_or_else(|| anyhow::anyhow!("module {module} not in pack"))?
        .clone();
    let xbits = std::fs::read(x_path).with_context(|| format!("read {x_path}"))?;
    let x: Vec<u16> = xbits
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    if x.len() % m.k != 0 {
        bail!("x-in is {} u16, not a multiple of K={}", x.len(), m.k);
    }
    let rows = x.len() / m.k;
    let suh = le_f16_bits(&m.suh.read_bytes(&pack.dir)?);
    let svh = le_f16_bits(&m.svh.read_bytes(&pack.dir)?);
    let raw = m.trellis.read_bytes(&pack.dir)?;
    let tr_bits: Vec<u16> = raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let mut chain = run_chain(&dev, &tr_bits, &suh, &svh, &x, rows, m.k, m.n, m.k2)?;
    let y = dev.dtoh_sync_copy(&chain.d_y)?;
    let mut out = Vec::with_capacity(y.len() * 2);
    for v in &y {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(y_path, &out).with_context(|| format!("write {y_path}"))?;
    println!(
        "XCHAIN_WRITE module={module} bits={} K={} N={} rows={} y={}",
        m.k2 / 2, m.k, m.n, rows, y_path
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. Standalone bench: per-class GB/s at widths, vs 238 roofline / 79-83 naive floor.
// ---------------------------------------------------------------------------

