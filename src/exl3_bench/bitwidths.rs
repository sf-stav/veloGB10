//! The all-bitwidths gate (--probe-exl3-bitwidths) with its grouped and split-K cells.
//! Probe-only code of the `exl3_bench` module (see `mod.rs` for the map); shared helpers come from the parent.
//! Numerics contract (module header in `mod.rs`): the kernel decode is bitwise vs the host oracle; do not change reduction orders.

use super::*;

// ---------------------------------------------------------------------------
// S-A3-f-b: WIDE-M GEMM gate (chunked prefill enables the M > 16 path).
// For every module class and a ladder of M:
//   (a) y_raw rel-L2 vs fp16(gemm_ref) on the first min(M,16) rows at the
//       calibrated 1e-4 threshold (full-M host refs cost O(M·K·N) scalar FLOPs);
//   (b) y_raw BITWISE identical across MT in {1,2,4,8} at EVERY M (structural
//       exactness: any tile-boundary/indexing/padding bug is a bit diff);
//   (c) at M=16, y_raw BITWISE identical to the ORIGINAL m16 kernel
//       (row-wise equivalence of the wide body, decode included).
//   ./gb10_inference --probe-exl3-wide --model-dir <pack> \
//     [--exl3-wide-ms 1,4,16,17,32,64,256,512] [--exl3-wide-mts 1,2,4,8]
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// PR#27-D3: --probe-exl3-bitwidths — synthetic, model-independent width-matrix gate.
// Covers every width cell the loader can admit on the generic paths: dense a1 GEMM
// (all k2 in {3,4,6,8,10,12,14,16} vs the fresh inline host oracle), the wide family
// (WT/WG/WIDE at bits 3..=8, incl. the D2 case-8 arms), the split-K chain (ks, ks_p),
// the pipelined prefill MoE experts (moe_pf, bits 2..=7), the WP20 EPI grouped path
// (bits 3..=7), and the load-refusal predicates (a width the loader refuses must be
// refused, never silently served). Poison canaries catch no-op arms. No pack files
// are read — the probe runs on any box with the binary alone.
// ---------------------------------------------------------------------------
pub fn probe_bitwidths() -> Result<()> {
    let dev = dev0()?;
    load_module(&dev)?;
    let stream = fork_blocking_stream(&dev);
    println!("EXL3-BITWIDTHS: synthetic width-matrix gate (PR#27-D3; no model dir needed)");

    // ---- 1. dense a1 GEMM: every admissible k2 vs the fresh host oracle ----
    {
        let k: usize = 512;
        let n_eff: usize = 1024;
        let f_gemm = dev
            .get_func(MODULE, "exl3_hmma_gemm")
            .ok_or_else(|| anyhow::anyhow!("fn exl3_hmma_gemm missing"))?;
        for k2 in [3usize, 4, 6, 8, 10, 12, 14, 16] {
            let (kb, nb) = (k / 16, n_eff / 16);
            let len_u16 = kb * nb * 8 * k2;
            let tr_bits: Vec<u16> = (0..len_u16)
                .map(|i| ((i.wrapping_mul(2654435761)) >> 11) as u16)
                .collect();
            let w_dec = decode_trellis_host(
                &tr_bits.iter().map(|v| *v as i16).collect::<Vec<_>>(),
                kb,
                nb,
                k2,
            );
            let x = synth_x(4, k, 0xB175 + k2 as u32);
            let suh: Vec<u16> = (0..k)
                .map(|i| f16::from_f32(0.25 + (i % 13) as f32 * 0.03125).to_bits())
                .collect();
            let xh = had_scale_ref(&x, &suh, 4, k, true);
            let d_tr = dev.htod_sync_copy(&tr_bits)?;
            let d_xh = dev.htod_sync_copy(&xh)?;
            let poison: Vec<u16> = vec![0xDEADu16; 4 * n_eff];
            let mut d_y = dev.htod_sync_copy(&poison)?;
            unsafe {
                f_gemm
                    .clone()
                    .launch_on_stream(
                        &stream,
                        LaunchConfig {
                            grid_dim: ((n_eff / 128) as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&d_tr, &d_xh, &mut d_y, 4i32, k as i32, n_eff as i32, k2 as i32),
                    )
                    .unwrap();
            }
            dev.synchronize().unwrap();
            let y = dev.dtoh_sync_copy(&d_y)?;
            let survivors = y.iter().filter(|&&v| v == 0xDEAD).count();
            if survivors != 0 {
                bail!(
                    "EXL3-BITWIDTHS FAIL dense k2={k2}: {survivors}/{} poison elements unwritten (no-op arm?)",
                    4 * n_eff
                );
            }
            let yref = gemm_ref(&w_dec, &xh, 4, k, n_eff);
            let yref16: Vec<f32> = yref.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
            let (_, rel16) = rel_l2(&y[..4 * n_eff], &yref16);
            if rel16 > 1e-4 {
                bail!("EXL3-BITWIDTHS FAIL dense k2={k2}: y_raw rel-L2 {rel16:.3e} > 1e-4");
            }
            println!("    dense a1 k2={k2:<2} (K={}): PASS (poison 0, rel-L2 vs fp16 ref {:.2e})", k2 as f32 / 2.0, rel16);
        }
    }

    // ---- 2. wide family: WT/WG/WIDE arms at bits 3..=8 (incl. D2 case-8) ----
    {
        let k: usize = 512;
        let n_eff: usize = 1024;
        let f_wide = dev
            .get_func(MODULE, "exl3_hmma_gemm_wide")
            .ok_or_else(|| anyhow::anyhow!("fn exl3_hmma_gemm_wide missing"))?;
        for bits in [3usize, 4, 5, 6, 7, 8] {
            let k2 = 2 * bits;
            let (kb, nb) = (k / 16, n_eff / 16);
            let len_u16 = kb * nb * 8 * k2;
            let tr_bits: Vec<u16> = (0..len_u16)
                .map(|i| ((i.wrapping_mul(2654435761)) >> 11) as u16)
                .collect();
            let w_dec = decode_trellis_host(
                &tr_bits.iter().map(|v| *v as i16).collect::<Vec<_>>(),
                kb,
                nb,
                k2,
            );
            let x = synth_x(16, k, 0x7EED0 + bits as u32);
            let suh: Vec<u16> = (0..k)
                .map(|i| f16::from_f32(0.25 + (i % 13) as f32 * 0.03125).to_bits())
                .collect();
            let xh = had_scale_ref(&x, &suh, 16, k, true);
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
                        (&d_tr, &d_xh, &mut d_y, 16i32, k as i32, n_eff as i32, bits as i32, 1i32),
                    )
                    .unwrap();
            }
            dev.synchronize().unwrap();
            let y = dev.dtoh_sync_copy(&d_y)?;
            let survivors = y.iter().filter(|&&v| v == 0xDEAD).count();
            if survivors != 0 {
                bail!(
                    "EXL3-BITWIDTHS FAIL wide bits={bits}: {survivors}/{} poison elements unwritten (no-op arm?)",
                    16 * n_eff
                );
            }
            let yref = gemm_ref(&w_dec, &xh, 16, k, n_eff);
            let yref16: Vec<f32> = yref.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
            let (_, rel16) = rel_l2(&y[..16 * n_eff], &yref16);
            if rel16 > 1e-4 {
                bail!("EXL3-BITWIDTHS FAIL wide bits={bits}: y_raw rel-L2 {rel16:.3e} > 1e-4");
            }
            println!("    wide WIDE bits={bits}: PASS (poison 0, rel-L2 vs fp16 ref {:.2e})", rel16);
        }
    }

    // ---- 3. split-K chain: ks at k2 in {4,6,14,16}, ks_p bits 2..=6 ----
    // (reuse probe sections via the module's ks entries; poison + rel-L2 as above)
    probe_ks_bw(&dev, &stream)?;

    // ---- 4. prefill MoE experts (moe_pf) bits 2..=7 + PFHALF: the fractional K = 1.5 / 2.5 / 3.5 (codes 17 / 18 / 19) ----
    for bits in [2usize, 3, 4, 5, 6, 7, 17, 18, 19] {
        probe_moe_pf(&dev, bits)?;
    }

    // ---- 5. WP20 EPI grouped path, bits 3..=7 ----
    for bits in [3usize, 4, 5, 6, 7] {
        probe_moe_epi(&dev, bits)?;
    }

    // ---- 6. refusal predicates: a refused width must refuse ----
    {
        use crate::exl3_forward::{wide_trellis_supports_bits, moe_pf_kernel_supports_bits, quad_k2_supported};
        for b in [0usize, 1, 2, 9] {
            assert!(!wide_trellis_supports_bits(b), "wide must refuse bits={b}");
        }
        assert!(wide_trellis_supports_bits(8), "wide serves bits=8 (PR#27-D2)");
        for b in [0usize, 1, 8] {
            assert!(!moe_pf_kernel_supports_bits(b), "moe_pf must refuse bits={b}");
        }
        for k2 in [0usize, 2, 5, 7, 9, 11, 13, 15, 17, 18] {
            assert!(!quad_k2_supported(k2), "quad must refuse k2={k2}");
        }
        assert!(quad_k2_supported(16), "quad serves k2=16 (PR#27-D2)");
        // K8EXP: the expert family serves K = 7 / 8 (k2 14 / 16); the coop kernels stay 2..=6 (the host falls back)
        use crate::exl3_forward::{expert_k2_supported, moe_coop_supports_bits};
        for k2 in [3usize, 4, 5, 6, 7, 8, 10, 12, 14, 16] { assert!(expert_k2_supported(k2), "experts serve k2={k2}"); }
        for k2 in [0usize, 2, 9, 11, 13, 15, 17, 18] { assert!(!expert_k2_supported(k2), "experts must refuse k2={k2}"); }
        assert!(moe_coop_supports_bits(6) && !moe_coop_supports_bits(7) && !moe_coop_supports_bits(8), "coop serves 2..=6 only");
        println!("    refusal predicates: PASS (wide 0/1/2/9, moe_pf 0/1/8, quad odd+out-of-range refuse; 8/16 serve; experts serve k2 14/16 and refuse 2/9/11/13/15; coop 2..=6)");
    }

    // ---- 7. K8EXP: the SERVED grouped expert entry at every integer expert width K = 2..=8 ----
    probe_grouped_bw(&dev, &stream)?;

    println!("EXL3-BITWIDTHS: ALL CELLS PASS");
    Ok(())
}

// K8EXP (2026-11-07): section 7 of --probe-exl3-bitwidths. The grouped expert entry the host SERVES
// (crate::exl3_forward::grouped_fn) at every integer expert width K = 2..=8 (K = 8 = the new xq_gemm_grouped_b8):
// three experts with distinct trellis contents at distinct offsets (a 64-word gap between them, so a wrong offset
// reads wrong words) in a grid padded to four slots with esel = 3 (the fourth slot must write nothing); rows
// m in {1, 2, 4, 8, 9} take the A-once body at K = 2560 (M*(K+8)*2 <= 48 KiB) and {10, 12, 16} the barrier body;
// outputs are poisoned. Each expert's y_raw must be BITWISE equal to the k2-keyed m16 reference (exl3_hmma_gemm on
// that expert's trellis alone) AND within rel-L2 1e-3 of the unrounded f32 host oracle (decode_trellis_host + gemm_ref) — the oracle is a different implementation class, so a wrong-but-consistent body cannot pass.
fn probe_grouped_bw(dev: &std::sync::Arc<CudaDevice>, stream: &CudaStream) -> Result<()> {
    let (k, n, ne, slots, gap) = (2560usize, 256usize, 3usize, 4usize, 64usize);
    let (kb, nb) = (k / 16, n / 16);
    let f_ref = dev.get_func(MODULE, "exl3_hmma_gemm").context("exl3_hmma_gemm")?;
    for bits in 2usize..=8 {
        let k2 = 2 * bits;
        let per = kb * nb * 8 * k2;
        let tr_all: Vec<u16> = (0..(per + gap) * ne)
            .map(|i| ((i.wrapping_mul(2654435761) ^ bits.wrapping_mul(0x9E37_79B9)) >> 11) as u16)
            .collect();
        let tr_e = |e: usize| &tr_all[e * (per + gap)..e * (per + gap) + per];
        let offs: Vec<u64> = (0..slots).map(|e| (e.min(ne - 1) * (per + gap)) as u64).collect();
        let w_dec: Vec<Vec<u16>> = (0..ne)
            .map(|e| decode_trellis_host(&tr_e(e).iter().map(|v| *v as i16).collect::<Vec<_>>(), kb, nb, k2))
            .collect();
        let d_tr = dev.htod_sync_copy(&tr_all)?;
        let d_offs = dev.htod_sync_copy(&offs)?;
        let d_esel = dev.htod_sync_copy(&[ne as i32])?;
        let d_tr_e: Vec<CudaSlice<u16>> = (0..ne).map(|e| dev.htod_sync_copy(tr_e(e))).collect::<std::result::Result<_, _>>()?;
        let (mut cells, mut names, mut worst) = (Vec::new(), std::collections::BTreeSet::new(), 0f64);
        for m in [1usize, 2, 4, 8, 9, 10, 12, 16] {
            let name = crate::exl3_forward::grouped_fn(m, k, bits);
            names.insert(name);
            let f = dev.get_func(MODULE, name).with_context(|| name.to_string())?;
            let xh: Vec<u16> = (0..slots).flat_map(|e| synth_x(m, k, 0x6B38 ^ ((bits as u32) << 8) ^ (e as u32) ^ ((m as u32) << 16))).collect();
            let d_xh = dev.htod_sync_copy(&xh)?;
            let mut d_y = dev.htod_sync_copy(&vec![0xDEADu16; slots * m * n])?;
            let cfg = LaunchConfig { grid_dim: ((slots * (n / 128)) as u32, 1, 1), block_dim: (256, 1, 1),
                                     shared_mem_bytes: crate::exl3_forward::a1_smem(m, k) };
            unsafe {
                f.clone().launch_on_stream(stream, cfg, (&d_tr, &d_offs, &d_xh, &mut d_y, m as i32, k as i32, n as i32, bits as i32, &d_esel))?;
            }
            dev.synchronize()?;
            let y = dev.dtoh_sync_copy(&d_y)?;
            let touched = y[ne * m * n..].iter().filter(|v| **v != 0xDEAD).count();
            if touched > 0 {
                bail!("EXL3-BITWIDTHS FAIL grouped bits={bits} m={m} ({name}): the absent 4th slot (esel = 3) has {touched} written elements");
            }
            for e in 0..ne {
                let ye = &y[e * m * n..(e + 1) * m * n];
                let xe = &xh[e * m * k..(e + 1) * m * k];
                let d_xe = dev.htod_sync_copy(xe)?;
                let mut d_yr = dev.htod_sync_copy(&vec![0xBEEFu16; m * n])?;
                let rcfg = LaunchConfig { grid_dim: ((n / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                unsafe {
                    f_ref.clone().launch_on_stream(stream, rcfg, (&d_tr_e[e], &d_xe, &mut d_yr, m as i32, k as i32, n as i32, k2 as i32))?;
                }
                dev.synchronize()?;
                let yr = dev.dtoh_sync_copy(&d_yr)?;
                let unwritten = ye.iter().filter(|v| **v == 0xDEAD).count();
                let mism = ye.iter().zip(yr.iter()).filter(|(a, b)| a != b).count();
                if unwritten > 0 || mism > 0 {
                    bail!("EXL3-BITWIDTHS FAIL grouped bits={bits} m={m} expert {e} ({name}): {mism}/{} mismatches vs exl3_hmma_gemm k2={k2} ({unwritten} poison survivors)", m * n);
                }
                // vs the UNROUNDED f32 oracle: at K = 2560 the f16 output rounding alone is ~1e-4 rel-L2 (a first run
                // compared f16-rounded vs f16-rounded at section 1's K = 512 bound of 1e-4 and read 1.09e-4 on a cell
                // whose grouped output was bitwise == exl3_hmma_gemm). A wrong decode reads O(1); 1e-3 separates them.
                let yref: Vec<f32> = gemm_ref(&w_dec[e], xe, m, k, n);
                let (_, rel) = rel_l2(ye, &yref);
                if !(rel <= 1e-3) {
                    bail!("EXL3-BITWIDTHS FAIL grouped bits={bits} m={m} expert {e} ({name}): y_raw rel-L2 {rel:.3e} vs the f32 host oracle > 1e-3");
                }
                worst = worst.max(rel);
            }
            cells.push(m);
        }
        println!("    grouped expert K={bits} (k2 {k2}) via {:?}: PASS (3 experts + 1 absent slot, m {cells:?}: bitwise == exl3_hmma_gemm, rel-L2 vs the f32 host oracle <= {worst:.2e}, absent slot unwritten)", names);
    }
    Ok(())
}

// split-K chain cells for the bitwidths probe (ks + ks_p; poison + fresh oracle)
fn probe_ks_bw(dev: &std::sync::Arc<CudaDevice>, stream: &CudaStream) -> Result<()> {
    let k: usize = 512;
    let n_eff: usize = 1024;
    let f_ks = dev
        .get_func(MODULE, "exl3_hmma_gemm_ks")
        .ok_or_else(|| anyhow::anyhow!("fn exl3_hmma_gemm_ks missing"))?;
    for k2 in [4usize, 6, 14, 16] {
        let (kb, nb) = (k / 16, n_eff / 16);
        let len_u16 = kb * nb * 8 * k2;
        let tr_bits: Vec<u16> = (0..len_u16)
            .map(|i| ((i.wrapping_mul(2654435761)) >> 11) as u16)
            .collect();
        let w_dec = decode_trellis_host(
            &tr_bits.iter().map(|v| *v as i16).collect::<Vec<_>>(),
            kb,
            nb,
            k2,
        );
        let x = synth_x(4, k, 0x5A5 + k2 as u32);
        let suh: Vec<u16> = (0..k)
            .map(|i| f16::from_f32(0.25 + (i % 13) as f32 * 0.03125).to_bits())
            .collect();
        let xh = had_scale_ref(&x, &suh, 4, k, true);
        let d_tr = dev.htod_sync_copy(&tr_bits)?;
        let d_xh = dev.htod_sync_copy(&xh)?;
        let ks: u32 = 2;
        let words = 4 * k2; // ring words per 16x16 block
        let blocks = kb * nb;
        let ws_len = 4 * n_eff * ks as usize;
        let nan_poison: Vec<f32> = vec![f32::NAN; ws_len];
        let d_ws = dev.htod_sync_copy(&nan_poison)?;
        let poison: Vec<u16> = vec![0xDEADu16; 4 * n_eff];
        let mut d_y = dev.htod_sync_copy(&poison)?;
        unsafe {
            f_ks
                .clone()
                .launch_on_stream(
                    &stream,
                    LaunchConfig {
                        grid_dim: ((n_eff / 128) as u32, ks, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&d_tr, &d_xh, &d_ws, 4i32, k as i32, n_eff as i32, k2 as i32, 0i32, (k / ks as usize) as i32, 0i32),
                )
                .unwrap();
        }
        let _ = words;
        let _ = blocks;
        // the ks GEMM writes split-K PARTIALS into ws (it never touches y); fold them with
        // exl3_ks_combine — the same two-launch chain the serve path runs (bench :3127).
        let f_comb = dev
            .get_func(MODULE, "exl3_ks_combine")
            .ok_or_else(|| anyhow::anyhow!("fn exl3_ks_combine missing"))?;
        unsafe {
            f_comb
                .clone()
                .launch_on_stream(
                    &stream.clone(),
                    LaunchConfig {
                        grid_dim: (((4 * n_eff + 255) / 256) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&d_ws, &mut d_y, 4i32, n_eff as i32, ks as i32),
                )
                .unwrap();
        }
        dev.synchronize().unwrap();
        let y = dev.dtoh_sync_copy(&d_y)?;
        let survivors = y.iter().filter(|&&v| v == 0xDEAD).count();
        if survivors != 0 {
            bail!(
                "EXL3-BITWIDTHS FAIL ks k2={k2}: {survivors}/{} poison elements unwritten (no-op arm?)",
                4 * n_eff
            );
        }
        let yref = gemm_ref(&w_dec, &xh, 4, k, n_eff);
        // split-K fp32 association differs from the single-pass fold; gate numerics at 1e-3
        let yref_f32 = yref.clone();
        let (rel32, _) = rel_l2(&y[..4 * n_eff], &yref_f32);
        if rel32 > 1e-3 {
            bail!("EXL3-BITWIDTHS FAIL ks k2={k2}: rel-L2 {rel32:.3e} > 1e-3");
        }
        println!("    split-K ks k2={k2:<2} (K={}): PASS (poison 0, rel-L2 vs f32 ref {:.2e})", k2 as f32 / 2.0, rel32);
    }
    Ok(())
}

