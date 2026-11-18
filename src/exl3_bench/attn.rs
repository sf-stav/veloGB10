//! Dense and sparse attention probes.
//! Probe-only code of the `exl3_bench` module (see `mod.rs` for the map); shared helpers come from the parent.
//! Numerics contract (module header in `mod.rs`): the kernel decode is bitwise vs the host oracle; do not change reduction orders.

use super::*;

// ---------------------------------------------------------------------------
// S-A3-m item 5: the attention bit-contract (S-A3-l gate-coverage lesson — binv
// covered GEMM shapes only). For each verify width m and prefix length p0, the
// verify kernel `xq_attn_decode_group` (v2 warp-tree scan) must be BITWISE equal,
// in every row's output AND the KV cache it writes, to (a) the pre-S-A3-l
// per-row loop over xq_attn_softmax (`xq_attn_decode_group_ref`, the smem
// sync-tree class) and (b) m sequential single-row `xq_attn_decode` launches
// (what plain decode computes for those tokens). Synthetic Qwen3.8-Flash-Next
// shapes: 24 q-heads / 2 kv-heads, hd 256, partial rope 64.
// WP13: (c) the dense v3 chain (xq_attn_dense_prep -> _dots -> _acc4 AND _acc1, both
// accumulate variants at every width) must equal v2 bitwise too — attn rows + KV cache,
// in all four KV formats (f32 / f16 / fp8 / q8 — q8 = the p2 default: rotated staged q,
// in-kernel output un-rotation), prefixes up to the dense cap (p0 2036 + m 16 -> positions
// 0..=2051). (a)/(b) stay fatal for f32 (their original contract) and are reported for
// f16/fp8/q8. Then a small v2-vs-v3 timing table (synthetic, f32 and q8).
// ---------------------------------------------------------------------------
// TP-4G: the same gate over four shapes — 24/2 (TP=1, G=12; the full original cell set + draft
// chain), 12/1 (TP=2, G=12, one kv head), 12/2 and 6/1 (G=6: the W=4 / TP=4 shape, the new
// `_g6` kernels; 12/2 also exercises kvh > 0 at G=6). Every v3 variant must equal v2 bitwise
// at every shape; the kernels are picked by the HOST's plan (`dv3_plan` / `dv3_w4_twin`).
pub(super) fn probe_attn(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    probe_attn_shape::<24, 2>(dev, true)?;
    probe_attn_shape::<12, 1>(dev, false)?;
    probe_attn_shape::<12, 2>(dev, false)?;
    probe_attn_shape::<6, 1>(dev, false)?;
    Ok(())
}

fn probe_attn_shape<const NH: usize, const NKV: usize>(dev: &std::sync::Arc<CudaDevice>, full: bool) -> Result<()> {
    use crate::exl3_forward::{dv3_plan, dv3_w4_twin};
    let grp = crate::exl3_forward::dv3_group(NH, NKV)
        .ok_or_else(|| anyhow::anyhow!("EXL3-ATTN: {NH}/{NKV} is not a dense-v3 group"))?;
    let shape_tag = if full { String::new() } else { format!("[{NH}/{NKV}]") };
    let acc_block = dv3_plan(grp, 1).3;
    const HD: usize = 256;
    const RDIM: usize = 64;
    const MAXP: usize = 2304;   // holds the whole dense regime (positions 0..=2051)
    const TS: usize = 2304;     // XQ_DV3_TS
    const NCH: usize = 36;      // XQ_DV3_NCH
    const MMAX: usize = 16;
    let f_v2 = dev.get_func(MODULE, "xq_attn_decode_group").context("xq_attn_decode_group")?;
    let f_ref = dev.get_func(MODULE, "xq_attn_decode_group_ref").context("xq_attn_decode_group_ref")?;
    let f_dec = dev.get_func(MODULE, "xq_attn_decode").context("xq_attn_decode")?;
    let f_prep = dev.get_func(MODULE, "xq_attn_dense_prep").context("xq_attn_dense_prep")?;
    let (n_dots, n_acc1, _, _) = dv3_plan(grp, 1);
    let (_, n_acc4, _, _) = dv3_plan(grp, 2);
    let f_dots = dev.get_func(MODULE, n_dots).context(n_dots)?;
    let f_acc4 = dev.get_func(MODULE, n_acc4).context(n_acc4)?;
    let f_acc1 = dev.get_func(MODULE, n_acc1).context(n_acc1)?;
    let mut rng = 0x5eed_1234u32;
    let mut rnd = move |amp: f32| -> f32 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (((rng >> 8) % 20001) as f32 / 10000.0 - 1.0) * amp
    };
    let nh_p = (NH | (NKV << 16)) as i32;
    let hd_p = (HD | (RDIM << 16)) as i32;
    // v3 scratch (qstage, score plane, chunk maxima) sized for the widest width
    let d_qs = dev.htod_sync_copy(&vec![0f32; MMAX * NH * HD])?;
    let d_s = dev.htod_sync_copy(&vec![0f32; MMAX * NH * TS])?;
    let d_pm = dev.htod_sync_copy(&vec![0f32; MMAX * NH * NCH])?;
    let (pqs, ps, ppm) = (*d_qs.device_ptr() as u64, *d_s.device_ptr() as u64, *d_pm.device_ptr() as u64);
    // the host's launch sequence (FwdModel::dense_attn_v3); acc4 = true -> 64-dim slices
    // W4/SMALL: the 5-stage-ring accumulate twins (bitwise to acc4 / acc1) + the draft-pass chain
    let (n_acc4w, n_acc1w) = (dv3_w4_twin(n_acc4), dv3_w4_twin(n_acc1));
    let f_acc4w = dev.get_func(MODULE, n_acc4w).context(n_acc4w)?;
    let f_acc1w = dev.get_func(MODULE, n_acc1w).context(n_acc1w)?;
    let f_aprep = dev.get_func(MODULE, "xq_attn_prep").context("xq_attn_prep")?;
    let f_sk4 = dev.get_func(MODULE, "xq_attn_dense_splitk4").context("xq_attn_dense_splitk4")?;
    let f_comb = dev.get_func(MODULE, "xq_attn_sel_combine").context("xq_attn_sel_combine")?;
    let f_sk4w = dev.get_func(MODULE, "xq_attn_dense_splitk4_w4").context("xq_attn_dense_splitk4_w4")?;
    let f_combw = dev.get_func(MODULE, "xq_attn_sel_combine_w4").context("xq_attn_sel_combine_w4")?;
    const NSPL: usize = 64;   // DRAFT_ATTN_SPLITS
    let d_dpm = dev.htod_sync_copy(&vec![0f32; MMAX * NH * NSPL])?;
    let d_dpl = dev.htod_sync_copy(&vec![0f32; MMAX * NH * NSPL])?;
    let d_dpa = dev.htod_sync_copy(&vec![0f32; MMAX * NH * NSPL * HD])?;
    let (pdm, pdl, pda) = (*d_dpm.device_ptr() as u64, *d_dpl.device_ptr() as u64, *d_dpa.device_ptr() as u64);
    // the draft head's chain (FwdModel::head_launches): w4 = dense_prep + splitk4_w4 + combine_w4
    let run_draft = |w4: bool, m: usize, pa: u64, pq: u64, pk: u64, pv: u64, pc: u64, pn: u64,
                     pcs: u64, psp: u64, mpf: i32| -> Result<()> {
        let lc = |g: u32, b: u32| LaunchConfig { grid_dim: (g, 1, 1), block_dim: (b, 1, 1), shared_mem_bytes: 0 };
        let geom: i64 = (NSPL as i64) | (((mpf as u32) as i64) << 32);
        let rows_kvf = (m as i32) | (((mpf as u32 >> 28) as i32) << 28);
        unsafe {
            if w4 {
                f_prep.clone().launch(lc((m * NH) as u32, HD as u32),
                    (pqs, pc, pq, pk, pv, pn, pcs, psp, nh_p, hd_p, mpf, 1e-6f32))?;
                f_sk4w.clone().launch(lc((m * NKV * NSPL) as u32, 256), (pdm, pdl, pda, pqs, pc, psp, nh_p, geom))?;
                f_combw.clone().launch(lc((m * NH) as u32, HD as u32), (pa, pdm, pdl, pda, pq, nh_p, NSPL as i32, rows_kvf))?;
            } else {
                f_aprep.clone().launch(lc((m * NKV) as u32, HD as u32),
                    (pqs, pc, pq, pk, pv, pn, pcs, psp, nh_p, hd_p, mpf, 1e-6f32))?;
                f_sk4.clone().launch(lc((m * NKV * NSPL) as u32, 256), (pdm, pdl, pda, pqs, pc, psp, nh_p, geom))?;
                f_comb.clone().launch(lc((m * NH) as u32, HD as u32), (pa, pdm, pdl, pda, pq, nh_p, NSPL as i32, rows_kvf))?;
            }
        }
        Ok(())
    };
    let mut draft_checked = 0usize;
    let run_v3 = |w4: bool, acc4: bool, m: usize, pa: u64, pq: u64, pk: u64, pv: u64, pc: u64, pn: u64,
                  pcs: u64, psp: u64, mpf: i32| -> Result<()> {
        let lc = |g: (u32, u32, u32), b: u32| LaunchConfig { grid_dim: g, block_dim: (b, 1, 1), shared_mem_bytes: 0 };
        let nchg = (96 / (NKV * m)).clamp(1, NCH) as u32;
        let (fa, sd) = match (w4, acc4) {
            (false, true) => (&f_acc4, 64usize), (false, false) => (&f_acc1, 16usize),
            (true, true) => (&f_acc4w, 64usize), (true, false) => (&f_acc1w, 16usize),
        };
        unsafe {
            f_prep.clone().launch(lc(((m * NH) as u32, 1, 1), HD as u32),
                (pqs, pc, pq, pk, pv, pn, pcs, psp, nh_p, hd_p, mpf, 1e-6f32))?;
            f_dots.clone().launch(lc((nchg, NKV as u32, m as u32), 256),
                (ps, ppm, pqs, pc, psp, nh_p, hd_p, mpf))?;
            fa.clone().launch(lc(((HD / sd) as u32, NKV as u32, m as u32), acc_block),
                (pa, ps, ppm, pc, pq, psp, nh_p, hd_p, mpf))?;
        }
        Ok(())
    };
    let mut checked = 0usize;
    for (fmt, fname) in [(0u32, "f32"), (1, "f16"), (2, "fp8"), (3, "q8"), (5, "q4k")] {
        // q8 (WP25): hd codes + hd/32 f16 group scales padded to 16 B (= HD + 16 at hd 256)
        // q4k: hd/2 packed 4-bit codes + the same scales (144 B at hd 256)
        let rb = match fmt { 0 => 4 * HD, 1 => 2 * HD, 5 => (HD / 2 + HD / 16).next_multiple_of(16), _ => HD + 16 };
        let mpf = (MAXP as i32) | ((fmt as i32) << 28);
        let cache_bytes = 2 * NKV * MAXP * rb;
        let p0s: &[usize] = if full { &[0, 5, 63, 300, 1100, 2036] } else { &[0, 63, 1100, 2036] };
        for &p0 in p0s {
            // one prefix cache shared by every width at this p0 (positions < p0)
            let mut cache0 = vec![0u8; cache_bytes];
            for kv in 0..2 {
                for h in 0..NKV {
                    for t in 0..p0 {
                        let base = ((kv * NKV + h) * MAXP + t) * rb;
                        let amp = if kv == 0 { 1.5 } else { 1.0 };
                        match fmt {
                            0 => for d in 0..HD {
                                cache0[base + 4 * d..base + 4 * d + 4].copy_from_slice(&rnd(amp).to_le_bytes());
                            },
                            1 => for d in 0..HD {
                                cache0[base + 2 * d..base + 2 * d + 2]
                                    .copy_from_slice(&f16::from_f32(rnd(amp)).to_bits().to_le_bytes());
                            },
                            2 => {
                                for d in 0..HD {
                                    let mut b = ((rnd(1.0) + 1.0) * 127.9) as u8;
                                    if b & 0x7F == 0x7F { b ^= 1; }   // no e4m3 NaN codes
                                    cache0[base + d] = b;
                                }
                                let scl = 0.004f32 + (rnd(1.0) + 1.0) * 0.008;
                                cache0[base + HD..base + HD + 4].copy_from_slice(&scl.to_le_bytes());
                            }
                            3 => {
                                // q8: any code byte is valid; one positive f16 scale per 32-group
                                for d in 0..HD {
                                    cache0[base + d] = ((rnd(1.0) + 1.0) * 127.9) as u8;
                                }
                                for g in 0..HD / 32 {
                                    let scl = f16::from_f32(amp * (0.25 + (rnd(1.0) + 1.0) * 0.5));
                                    cache0[base + HD + 2 * g..base + HD + 2 * g + 2].copy_from_slice(&scl.to_bits().to_le_bytes());
                                }
                            }
                            5 => {
                                // q4k: a real quantized row (same bytes the served writer produces)
                                let row: Vec<f32> = (0..HD).map(|_| rnd(amp)).collect();
                                let q = crate::exl3_forward::q4k_quant_row(&row);
                                cache0[base..base + rb].copy_from_slice(&q);
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }
            for &m in &[1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16] { // WP23: + 7 (depth-6 verify width)
                let qg: Vec<u16> = (0..m * NH * HD * 2).map(|_| f16::from_f32(rnd(2.0)).to_bits()).collect();
                let kp: Vec<u16> = (0..m * NKV * HD).map(|_| f16::from_f32(rnd(2.0)).to_bits()).collect();
                let vp: Vec<u16> = (0..m * NKV * HD).map(|_| f16::from_f32(rnd(1.0)).to_bits()).collect();
                let qnkn: Vec<u16> = (0..2 * HD).map(|_| f16::from_f32(rnd(0.3)).to_bits()).collect();
                let mut cs = vec![0f32; m * 2 * RDIM];
                for r in 0..m {
                    for i in 0..RDIM {
                        let a = (p0 + r) as f32 * 0.01 * (i as f32 + 1.0);
                        cs[r * 2 * RDIM + i] = a.cos();
                        cs[r * 2 * RDIM + RDIM + i] = a.sin();
                    }
                }
                let sp: Vec<i32> = (0..m).flat_map(|r| [0i32, (p0 + r) as i32]).collect();
                let d_qg = dev.htod_sync_copy(&qg)?;
                let d_kp = dev.htod_sync_copy(&kp)?;
                let d_vp = dev.htod_sync_copy(&vp)?;
                let d_qn = dev.htod_sync_copy(&qnkn)?;
                let d_cs = dev.htod_sync_copy(&cs)?;
                let d_sp = dev.htod_sync_copy(&sp)?;
                let mut outs: Vec<(Vec<u16>, Vec<u8>)> = Vec::new();
                for which in 0..(if full { 9 } else { 7 }) {
                    let d_cache = dev.htod_sync_copy(&cache0)?;
                    let d_attn = dev.htod_sync_copy(&vec![0u16; m * NH * HD])?;
                    let pa = *d_attn.device_ptr() as u64;
                    let pq = *d_qg.device_ptr() as u64;
                    let pk = *d_kp.device_ptr() as u64;
                    let pv = *d_vp.device_ptr() as u64;
                    let pc = *d_cache.device_ptr() as u64;
                    let pn = *d_qn.device_ptr() as u64;
                    let pcs = *d_cs.device_ptr() as u64;
                    let psp = *d_sp.device_ptr() as u64;
                    let cfg = |gx: u32, gy: u32| LaunchConfig {
                        grid_dim: (gx, gy, 1),
                        block_dim: (HD as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        match which {
                            0 => f_v2.clone().launch(cfg(NH as u32, m as u32),
                                (pa, pq, pk, pv, pc, pn, pcs, psp, nh_p, hd_p, mpf, 1e-6f32))?,
                            1 => f_ref.clone().launch(cfg(NH as u32, m as u32),
                                (pa, pq, pk, pv, pc, pn, pcs, psp, nh_p, hd_p, mpf, 1e-6f32))?,
                            2 => {
                                // plain decode, one token at a time (row r sees rows < r written)
                                for r in 0..m {
                                    f_dec.clone().launch(cfg(NH as u32, 1),
                                        (pa + (r * NH * HD * 2) as u64,
                                         pq + (r * NH * HD * 2 * 2) as u64,
                                         pk + (r * NKV * HD * 2) as u64,
                                         pv + (r * NKV * HD * 2) as u64,
                                         pc, pn,
                                         pcs + (r * 2 * RDIM * 4) as u64,
                                         psp + (r * 2 * 4) as u64,
                                         nh_p, hd_p, mpf, 1e-6f32))?;
                                }
                            }
                            // WP13: the host's rule (m == 1 -> acc1, else acc4), then the other variant
                            3 => run_v3(false, m != 1, m, pa, pq, pk, pv, pc, pn, pcs, psp, mpf)?,
                            4 => run_v3(false, m == 1, m, pa, pq, pk, pv, pc, pn, pcs, psp, mpf)?,
                            // W4/SMALL: the ring twins, host rule then the other variant
                            5 => run_v3(true, m != 1, m, pa, pq, pk, pv, pc, pn, pcs, psp, mpf)?,
                            6 => run_v3(true, m == 1, m, pa, pq, pk, pv, pc, pn, pcs, psp, mpf)?,
                            // W4/SMALL draft chain: old (7) vs new (8), compared with each other
                            7 => run_draft(false, m, pa, pq, pk, pv, pc, pn, pcs, psp, mpf)?,
                            _ => run_draft(true, m, pa, pq, pk, pv, pc, pn, pcs, psp, mpf)?,
                        }
                    }
                    dev.synchronize()?;
                    outs.push((dev.dtoh_sync_copy(&d_attn)?, dev.dtoh_sync_copy(&d_cache)?));
                }
                // success signal: finite, nonzero outputs
                let nz = outs[0].0.iter().filter(|b| { let v = f16::from_bits(**b).to_f32(); v != 0.0 && v.is_finite() }).count();
                if nz < m * NH * HD / 2 {
                    bail!("EXL3-ATTN FAIL: implausible v2 output at fmt={fname} m={m} p0={p0} ({nz} finite nonzero of {})", m * NH * HD);
                }
                for (name, i) in [("ref-loop", 1usize), ("seq-decode", 2), ("v3-hostrule", 3), ("v3-other-acc", 4),
                                  ("W4S-v3-hostrule", 5), ("W4S-v3-other-acc", 6)] {
                    let am = outs[0].0.iter().zip(outs[i].0.iter()).filter(|(a, b)| a != b).count();
                    let cm = outs[0].1.iter().zip(outs[i].1.iter()).filter(|(a, b)| a != b).count();
                    if am + cm > 0 {
                        let first = outs[0].0.iter().zip(outs[i].0.iter()).position(|(a, b)| a != b);
                        let msg = format!("fmt={fname} m={m} p0={p0} vs {name}: {am} attn / {cm} cache-byte mismatches (first attn idx {first:?} = row {:?})",
                                          first.map(|f| f / (NH * HD)));
                        if i >= 3 || fmt == 0 { bail!("EXL3-ATTN FAIL {msg}"); }
                        println!("    EXL3-ATTN note (not gated for {fname}): {msg}");
                    }
                }
                // W4/SMALL draft chain: new (8) vs old (7) — attn rows + KV cache (its own class: the
                // split-K online softmax is not v2's two-pass scan, so it is gated against itself)
                if full {
                    let nzd = outs[7].0.iter().filter(|b| { let v = f16::from_bits(**b).to_f32(); v != 0.0 && v.is_finite() }).count();
                    if nzd < m * NH * HD / 2 {
                        bail!("EXL3-ATTN FAIL: implausible draft-chain output at fmt={fname} m={m} p0={p0} ({nzd} finite nonzero)");
                    }
                    let am = outs[7].0.iter().zip(outs[8].0.iter()).filter(|(a, b)| a != b).count();
                    let cm = outs[7].1.iter().zip(outs[8].1.iter()).filter(|(a, b)| a != b).count();
                    if am + cm > 0 {
                        bail!("EXL3-ATTN FAIL fmt={fname} m={m} p0={p0}: W4S draft chain (dense_prep + splitk4_w4 + combine_w4) \
                               vs (xq_attn_prep + splitk4 + combine): {am} attn / {cm} cache-byte mismatches");
                    }
                    draft_checked += 1;
                }
                checked += 1;
            }
            println!("    attn{shape_tag} {fname} p0={p0}: widths {{1..8,12,16}} v2 == v3 (acc4 + acc1) == W4S acc4_w4/acc1_w4{} \
                      (attn rows + KV cache); W4S draft chain == old draft chain",
                     if fmt == 0 { " == ref-loop == seq-decode" } else { "" });
        }
    }
    println!("EXL3-ATTN{shape_tag}: PASS ({checked} cells: v2 warp-tree scan bitwise vs xq_attn_softmax class; \
              WP13 dense v3 bitwise vs v2 in f32/f16/fp8/q8/q4k; W4S dense acc4_w4/acc1_w4 bitwise vs v2; \
              W4S draft chain bitwise vs the old draft chain in {draft_checked} cells)");

    // ---- WP13 timing (synthetic, f32 + q8, warm, 50 launches each; K/V rewrites are identical words)
    for &(fmt, m, p0) in &[(0usize, 1usize, 300usize), (0, 1, 2036), (0, 6, 300), (0, 6, 1100), (0, 6, 2036),
                           (3, 1, 2036), (3, 6, 2036)] {
        let rb = if fmt == 0 { 4 * HD } else { HD + 16 };
        let d_cache = dev.htod_sync_copy(&vec![0u8; 2 * NKV * MAXP * rb])?;
        let d_attn = dev.htod_sync_copy(&vec![0u16; m * NH * HD])?;
        let qg: Vec<u16> = (0..m * NH * HD * 2).map(|_| f16::from_f32(rnd(2.0)).to_bits()).collect();
        let kvp: Vec<u16> = (0..m * NKV * HD).map(|_| f16::from_f32(rnd(1.0)).to_bits()).collect();
        let qnkn: Vec<u16> = (0..2 * HD).map(|_| f16::from_f32(rnd(0.3)).to_bits()).collect();
        let cs = vec![0.5f32; m * 2 * RDIM];
        let sp: Vec<i32> = (0..m).flat_map(|r| [0i32, (p0 + r) as i32]).collect();
        let (d_qg, d_kv, d_qn, d_cs, d_sp) = (dev.htod_sync_copy(&qg)?, dev.htod_sync_copy(&kvp)?,
            dev.htod_sync_copy(&qnkn)?, dev.htod_sync_copy(&cs)?, dev.htod_sync_copy(&sp)?);
        let (pa, pq, pkv, pc, pn, pcs, psp) = (*d_attn.device_ptr() as u64, *d_qg.device_ptr() as u64,
            *d_kv.device_ptr() as u64, *d_cache.device_ptr() as u64, *d_qn.device_ptr() as u64,
            *d_cs.device_ptr() as u64, *d_sp.device_ptr() as u64);
        let mpf = (MAXP as i32) | ((fmt as i32) << 28);
        let cfg = |gx: u32, gy: u32| LaunchConfig { grid_dim: (gx, gy, 1), block_dim: (HD as u32, 1, 1), shared_mem_bytes: 0 };
        let mut t_old = 0f64;
        let mut t_new = 0f64;
        let mut t_w4 = 0f64;
        for pass in 0..2 {
            for which in 0..3 {
                dev.synchronize()?;
                let t0 = std::time::Instant::now();
                for _ in 0..50 {
                    if which == 0 {
                        unsafe {
                            if m == 1 {
                                f_dec.clone().launch(cfg((m * NH) as u32, 1),
                                    (pa, pq, pkv, pkv, pc, pn, pcs, psp, nh_p, hd_p, mpf, 1e-6f32))?;
                            } else {
                                f_v2.clone().launch(cfg(NH as u32, m as u32),
                                    (pa, pq, pkv, pkv, pc, pn, pcs, psp, nh_p, hd_p, mpf, 1e-6f32))?;
                            }
                        }
                    } else {
                        run_v3(which == 2, m != 1, m, pa, pq, pkv, pkv, pc, pn, pcs, psp, mpf)?;
                    }
                }
                dev.synchronize()?;
                let us = t0.elapsed().as_secs_f64() * 1e6 / 50.0;
                if pass == 1 { match which { 0 => t_old = us, 1 => t_new = us, _ => t_w4 = us } }
            }
        }
        println!("    EXL3-ATTN-V3{shape_tag} timing {} m={m} p0={p0}: old {} {t_old:.1} us, v3 {t_new:.1} us, W4S v3+acc_w4 {t_w4:.1} us \
                  per layer call ({:.1}x / {:.1}x; synthetic, launch-inclusive, harness-only)",
                 crate::exl3_forward::kv_fmt_name(fmt as u32), if m == 1 { "xq_attn_decode" } else { "xq_attn_decode_group" },
                 t_old / t_new.max(1e-9), t_old / t_w4.max(1e-9));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// W4/QSA: the SPARSE-regime attention bit-contract (probe_attn above covers the dense regime).
// For each KV format, context (~8K / 32K / 128K), selection mode and width m in {1..8, 12, 16},
// the row-shared union gather `xq_attn_sel_union` must write pm / pl / pacc BITWISE equal to the
// old served gathers — hg3 (the m >= 2 kernel it replaces), hg1 (m = 1) and splitk4 (the class
// the hg family was proven against) — and the unchanged xq_attn_sel_combine must produce the same
// attention rows from either. Synthetic Qwen3.8-Flash-Next shapes: 24 q-heads / 2 kv-heads, hd
// 256, 32 decode splits over the 2,051-entry selection pitch (split_len 65), ratio 4, top-512
// blocks; KV planes written on device through the served writer (xq_qu_synth_kv) at EVERY
// position (distinct content, so an off-by-position read cannot pass). Modes: overlap (the served
// shape: consecutive positions = one base top-512 incl. the local window + per-row block swaps +
// each row's newer blocks and tail), identical, disjoint (independent top-512 per row: the union
// approaches m x the row), two-slot (rows alternate between 2 slots: entries never shared across
// slots), fallback (a swapped pair, a duplicate and a -1: the kernel's exact verbatim path).
// Then a timing table (synthetic, isolated: RANKS the kernels; in-round cost is the ledger's job).
// ---------------------------------------------------------------------------
pub(super) fn probe_attn_sparse(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    use crate::exl3_forward::{qsa_union_fits, kv_rowbytes};
    const NH: usize = 24;
    const NKV: usize = 2;
    const HD: usize = 256;
    const NSD: usize = 32;
    const RATIO: usize = 4;
    const TOPK: usize = 512;
    const SEL_MAX: usize = 2048 + RATIO - 1;
    const LOCAL: usize = 64;                     // recent blocks every row keeps (the local window)
    let l = SEL_MAX.div_ceil(NSD);
    let fget = |n: &str| dev.get_func(MODULE, n).with_context(|| format!("{n} (stale PTX?)"));
    let (f_hg3, f_hg1, f_sk4, f_un, f_comb, f_syn) = (fget("xq_attn_sel_hg3")?, fget("xq_attn_sel_hg1")?,
        fget("xq_attn_sel_splitk4")?, fget("xq_attn_sel_union")?, fget("xq_attn_sel_combine")?, fget("xq_qu_synth_kv")?);
    let nh_p = (NH | (NKV << 16)) as i32;
    let split_hd = (l | (HD << 16)) as i32;
    let mut rs = 0x51A5_0001_2345_6789u64;
    let mut rnd = move || -> u64 { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; rs };
    // top-k block sets -> ascending token lists (block j = tokens 4j..4j+3, then the row's tail)
    let emit = |blocks: &mut Vec<usize>, pos: usize| -> Vec<i32> {
        blocks.sort_unstable();
        let nb = (pos + 1) / RATIO;
        let mut v: Vec<i32> = blocks.iter().flat_map(|&j| (0..RATIO).map(move |i| (j * RATIO + i) as i32)).collect();
        v.extend((nb * RATIO..=pos).map(|t| t as i32));
        v
    };
    let (mut cells, mut fb_cells) = (0usize, 0usize);
    let mut union_stat: Vec<String> = Vec::new();
    let widths = [1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16];
    for &(ctx, fmts) in &[(8192usize, &[0usize, 1, 2, 3, 5][..]), (32768, &[0, 1, 2, 3, 5][..]), (131072, &[0, 3][..])] {
        let modes: &[&str] = if ctx == 8192 { &["overlap", "identical", "disjoint", "two-slot", "fallback"] }
                             else { &["overlap", "identical", "disjoint"] };
        let nslot = if ctx == 8192 { 2usize } else { 1 };
        let maxp = ctx + 64;
        let p0 = ctx - 16;                       // rows at p0..p0+m-1 (QSA regime: > 2,051)
        for &fmt in fmts {
            let fname = crate::exl3_forward::kv_fmt_name(fmt as u32);   // 5 = q4k (packed row)
            let rb = kv_rowbytes(fmt as u32, HD);
            let mpf = (maxp as i32) | ((fmt as i32) << 28);
            let d_cache = dev.alloc_zeros::<u8>(nslot * 2 * NKV * maxp * rb)?;
            let pc = *d_cache.device_ptr() as u64;
            unsafe {
                f_syn.clone().launch(LaunchConfig { grid_dim: (maxp as u32, (nslot * 2 * NKV) as u32, 1),
                                                    block_dim: (HD as u32, 1, 1), shared_mem_bytes: 0 },
                                     (pc, mpf, HD as i32, NKV as i32, 0i32, 0xC0DE_0000u32 ^ (ctx as u32) ^ ((fmt as u32) << 20)))?;
            }
            dev.synchronize()?;
            for &mode in modes {
                for &m in &widths {
                    // ---- selection lists
                    let nb0 = (p0 + 1) / RATIO;
                    let mut base: Vec<usize> = (0..4).chain(nb0 - LOCAL..nb0).collect();
                    let mut inb = vec![false; nb0 + 16];
                    for &j in &base { inb[j] = true; }
                    while base.len() < TOPK {
                        let j = 4 + (rnd() as usize) % (nb0 - LOCAL - 4);
                        if !inb[j] { inb[j] = true; base.push(j); }
                    }
                    let mut sel = vec![-1i32; m * SEL_MAX];
                    let mut psel = vec![0i32; m];
                    let mut sp = vec![0i32; 2 * m];
                    let mut rows: Vec<Vec<i32>> = Vec::new();
                    for r in 0..m {
                        let pos = p0 + r;
                        let nb = (pos + 1) / RATIO;
                        let mut blocks: Vec<usize> = if mode == "disjoint" {
                            let mut s = vec![false; nb];
                            let mut b = Vec::new();
                            while b.len() < TOPK { let j = (rnd() as usize) % nb; if !s[j] { s[j] = true; b.push(j); } }
                            b
                        } else {
                            let mut b = base.clone();
                            let mut isb = inb.clone();
                            // this row's newer complete blocks join its local window
                            for j in nb0..nb { b.push(j); isb[j] = true; }
                            let swaps = if mode == "identical" { 0 } else { (rnd() % 9) as usize };
                            let mut extra = b.len() - TOPK + swaps;
                            if mode == "identical" {
                                // deterministic: every row drops the same (lowest droppable) blocks
                                b.sort_unstable();
                                b.retain(|&j| if extra > 0 && j >= 4 && j < nb0 - LOCAL { extra -= 1; false } else { true });
                            }
                            while extra > 0 {                                // drop non-local blocks
                                let i = (rnd() as usize) % b.len();
                                if b[i] >= 4 && b[i] < nb0 - LOCAL { isb[b[i]] = false; b.swap_remove(i); extra -= 1; }
                            }
                            while b.len() < TOPK {                           // add fresh ones
                                let j = 4 + (rnd() as usize) % (nb0 - LOCAL - 4);
                                if !isb[j] { isb[j] = true; b.push(j); }
                            }
                            b
                        };
                        let mut lst = emit(&mut blocks, pos);
                        if mode == "fallback" {
                            match r % 3 {
                                0 => lst.swap(100, 101),                     // unsorted pair
                                1 => lst[300] = lst[299],                    // duplicate
                                _ => lst[500] = -1,                          // a skipped entry
                            }
                        }
                        anyhow::ensure!(lst.len() <= SEL_MAX, "synthetic list {} > {SEL_MAX}", lst.len());
                        sel[r * SEL_MAX..r * SEL_MAX + lst.len()].copy_from_slice(&lst);
                        psel[r] = lst.len() as i32 - 1;
                        sp[2 * r] = if mode == "two-slot" { (r % 2) as i32 } else { 0 };
                        sp[2 * r + 1] = pos as i32;
                        rows.push(lst);
                    }
                    if mode == "overlap" && m == 8 {
                        // union entries per split vs one row's entries (what the kernel streams)
                        let (mut un, mut one) = (0usize, 0usize);
                        for s in 0..NSD {
                            let mut u: Vec<i32> = Vec::new();
                            for lst in &rows { u.extend(lst.iter().skip(s * l).take(l)); }
                            u.sort_unstable(); u.dedup();
                            un += u.len();
                            one += rows[0].iter().skip(s * l).take(l).count();
                        }
                        union_stat.push(format!("{fname}@{ctx}: {un} union entries / {one} per row ({:.2}x)", un as f64 / one as f64));
                    }
                    let qs: Vec<f32> = (0..m * NH * HD).map(|_| (rnd() % 40001) as f32 / 10000.0 - 2.0).collect();
                    let qg: Vec<u16> = (0..m * NH * HD * 2).map(|_| f16::from_f32((rnd() % 40001) as f32 / 10000.0 - 2.0).to_bits()).collect();
                    let (d_sel, d_ps, d_sp, d_qs, d_qg) = (dev.htod_sync_copy(&sel)?, dev.htod_sync_copy(&psel)?,
                        dev.htod_sync_copy(&sp)?, dev.htod_sync_copy(&qs)?, dev.htod_sync_copy(&qg)?);
                    let (psl, pps, psp, pqs, pqg) = (*d_sel.device_ptr() as u64, *d_ps.device_ptr() as u64,
                        *d_sp.device_ptr() as u64, *d_qs.device_ptr() as u64, *d_qg.device_ptr() as u64);
                    let np = m * NH * NSD;
                    let g2: u64 = (NSD as u64) | ((SEL_MAX as u64) << 20) | ((mpf as u32 as u64) << 32);
                    let poison = f32::from_bits(0x7FC0_DEAD);
                    // 0 union, 1 hg3, 2 hg1, 3 splitk4
                    let mut parts: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
                    let mut bufs = Vec::new();
                    for k in 0..4 {
                        let (bm, bl, ba) = (dev.htod_sync_copy(&vec![poison; np])?, dev.htod_sync_copy(&vec![poison; np])?,
                                            dev.htod_sync_copy(&vec![poison; np * HD])?);
                        let (pm, pl, pa) = (*bm.device_ptr() as u64, *bl.device_ptr() as u64, *ba.device_ptr() as u64);
                        let grid = ((m * NKV * NSD) as u32, 1, 1);
                        unsafe {
                            match k {
                                0 => {
                                    let smem = qsa_union_fits(fmt as u32, NH, NKV, HD, m, l)
                                        .context("union gather contract rejects the served shape")?;
                                    f_un.clone().launch(LaunchConfig { grid_dim: ((NKV * NSD * (NH / NKV / 4)) as u32, 1, 1),
                                                                       block_dim: ((32 * m) as u32, 1, 1), shared_mem_bytes: smem as u32 },
                                        (pm, pl, pa, pqs, pc, psl, pps, psp, nh_p, g2 as i64, split_hd, m as i32))?
                                }
                                1 => f_hg3.clone().launch(LaunchConfig { grid_dim: grid, block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                        (pm, pl, pa, pqs, pc, psl, pps, psp, nh_p, g2 as i64, split_hd))?,
                                2 => f_hg1.clone().launch(LaunchConfig { grid_dim: grid, block_dim: (384, 1, 1), shared_mem_bytes: 0 },
                                        (pm, pl, pa, pqs, pc, psl, pps, psp, nh_p, g2 as i64, split_hd))?,
                                _ => f_sk4.clone().launch(LaunchConfig { grid_dim: grid, block_dim: (HD as u32, 1, 1), shared_mem_bytes: 0 },
                                        (pm, pl, pa, pqs, pc, psl, pps, psp, nh_p, g2 as i64, l as i32))?,
                            }
                        }
                        dev.synchronize()?;
                        parts.push((dev.dtoh_sync_copy(&bm)?, dev.dtoh_sync_copy(&bl)?, dev.dtoh_sync_copy(&ba)?));
                        bufs.push((bm, bl, ba));
                    }
                    // success signal: the reference is real — no poison left, live splits in every row
                    let r0 = &parts[3];
                    let left = r0.0.iter().chain(&r0.1).chain(&r0.2).filter(|v| v.to_bits() == 0x7FC0_DEAD).count();
                    let live_rows = (0..m).filter(|&r| (0..NH * NSD).any(|i| r0.0[r * NH * NSD + i] > f32::NEG_INFINITY)).count();
                    let nzacc = r0.2.iter().filter(|v| **v != 0.0 && v.is_finite()).count();
                    if left > 0 || live_rows != m || nzacc < np * HD / 4 {
                        bail!("EXL3-ATTN-SPARSE FAIL: implausible splitk4 reference at {fname} ctx={ctx} {mode} m={m} \
                               (poison {left}, live rows {live_rows}/{m}, nonzero acc {nzacc}/{})", np * HD);
                    }
                    let bits = |x: &[f32], y: &[f32]| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
                    for (name, k) in [("union vs hg3", 1usize), ("union vs hg1", 2), ("union vs splitk4", 3)] {
                        let (dm, dl, da) = (bits(&parts[0].0, &parts[k].0), bits(&parts[0].1, &parts[k].1), bits(&parts[0].2, &parts[k].2));
                        if dm + dl + da > 0 {
                            let first = (0..np).find(|&i| parts[0].0[i].to_bits() != parts[k].0[i].to_bits()
                                                        || parts[0].1[i].to_bits() != parts[k].1[i].to_bits());
                            bail!("EXL3-ATTN-SPARSE FAIL {fname} ctx={ctx} {mode} m={m} {name}: pm {dm} pl {dl} pacc {da} bit-differ \
                                   of {np}/{} (first partial {first:?} = row {:?})", np * HD, first.map(|f| f / (NH * NSD)));
                        }
                    }
                    // the unchanged combine on the union partials == on hg3's (attention rows, f16 bits)
                    let mut outs = Vec::new();
                    for k in [0usize, 1] {
                        let d_attn = dev.htod_sync_copy(&vec![0x7E01u16; m * NH * HD])?;
                        let (pm, pl, pa) = (*bufs[k].0.device_ptr() as u64, *bufs[k].1.device_ptr() as u64, *bufs[k].2.device_ptr() as u64);
                        unsafe {
                            f_comb.clone().launch(LaunchConfig { grid_dim: ((m * NH) as u32, 1, 1), block_dim: (HD as u32, 1, 1), shared_mem_bytes: 0 },
                                (*d_attn.device_ptr() as u64, pm, pl, pa, pqg, nh_p, NSD as i32, (m as i32) | ((fmt as i32) << 28)))?;
                        }
                        dev.synchronize()?;
                        outs.push(dev.dtoh_sync_copy(&d_attn)?);
                    }
                    let fin = outs[1].iter().filter(|b| { let v = f16::from_bits(**b).to_f32(); v.is_finite() && v != 0.0 }).count();
                    let am = outs[0].iter().zip(&outs[1]).filter(|(a, b)| a != b).count();
                    if am > 0 || fin < m * NH * HD / 2 {
                        bail!("EXL3-ATTN-SPARSE FAIL {fname} ctx={ctx} {mode} m={m}: combine rows differ in {am} (finite nonzero {fin}/{})",
                              m * NH * HD);
                    }
                    cells += 1;
                    if mode == "fallback" { fb_cells += 1; }
                }
                println!("    attn-sparse {fname} ctx={ctx} {mode:<9}: widths {widths:?} union == hg3 == hg1 == splitk4 (pm/pl/pacc) \
                          and combine rows equal");
            }
        }
    }
    println!("EXL3-ATTN-SPARSE: PASS ({cells} cells incl. {fb_cells} fallback-path cells: W4/QSA union gather bitwise vs \
              hg3 / hg1 / splitk4 in f32/f16/fp8/q8/q4k at 8K/32K and f32/q8 at 128K)");
    println!("    union stats (overlap mode, m=8): {}", union_stat.join("; "));

    // ---- timing (synthetic, isolated, warm; 20 launches each after 3 warm-ups)
    for &(ctx, fmt, m) in &[(32768usize, 0usize, 8usize), (32768, 0, 4), (32768, 3, 8), (32768, 5, 8), (131072, 0, 8), (131072, 3, 8)] {
        let maxp = ctx + 64;
        let p0 = ctx - 16;
        let rb = kv_rowbytes(fmt as u32, HD);
        let mpf = (maxp as i32) | ((fmt as i32) << 28);
        let d_cache = dev.alloc_zeros::<u8>(2 * NKV * maxp * rb)?;
        let pc = *d_cache.device_ptr() as u64;
        unsafe {
            f_syn.clone().launch(LaunchConfig { grid_dim: (maxp as u32, (2 * NKV) as u32, 1), block_dim: (HD as u32, 1, 1), shared_mem_bytes: 0 },
                                 (pc, mpf, HD as i32, NKV as i32, 0i32, 0x7157u32))?;
        }
        let nb0 = (p0 + 1) / RATIO;
        let mut base: Vec<usize> = (0..4).chain(nb0 - LOCAL..nb0).collect();
        let mut inb = vec![false; nb0 + 16];
        for &j in &base { inb[j] = true; }
        while base.len() < TOPK { let j = 4 + (rnd() as usize) % (nb0 - LOCAL - 4); if !inb[j] { inb[j] = true; base.push(j); } }
        let mut sel = vec![-1i32; m * SEL_MAX];
        let mut psel = vec![0i32; m];
        let mut sp = vec![0i32; 2 * m];
        for r in 0..m {
            let pos = p0 + r;
            let mut b = base.clone();
            let mut isb = inb.clone();
            for _ in 0..4 {   // 4 swaps per row
                loop { let i = (rnd() as usize) % b.len(); if b[i] >= 4 && b[i] < nb0 - LOCAL { isb[b[i]] = false; b.swap_remove(i); break; } }
                loop { let j = 4 + (rnd() as usize) % (nb0 - LOCAL - 4); if !isb[j] { isb[j] = true; b.push(j); break; } }
            }
            let nb = (pos + 1) / RATIO;
            for j in nb0..nb { b.push(j); }
            while b.len() > TOPK { let i = (rnd() as usize) % b.len(); if b[i] >= 4 && b[i] < nb0 - LOCAL { b.swap_remove(i); } }
            let lst = emit(&mut b, pos);
            sel[r * SEL_MAX..r * SEL_MAX + lst.len()].copy_from_slice(&lst);
            psel[r] = lst.len() as i32 - 1;
            sp[2 * r + 1] = pos as i32;
        }
        let qs: Vec<f32> = (0..m * NH * HD).map(|_| (rnd() % 40001) as f32 / 10000.0 - 2.0).collect();
        let (d_sel, d_ps, d_sp, d_qs) = (dev.htod_sync_copy(&sel)?, dev.htod_sync_copy(&psel)?, dev.htod_sync_copy(&sp)?, dev.htod_sync_copy(&qs)?);
        let (psl, pps, psp, pqs) = (*d_sel.device_ptr() as u64, *d_ps.device_ptr() as u64, *d_sp.device_ptr() as u64, *d_qs.device_ptr() as u64);
        let np = m * NH * NSD;
        let (d_pm, d_pl, d_pa) = (dev.alloc_zeros::<f32>(np)?, dev.alloc_zeros::<f32>(np)?, dev.alloc_zeros::<f32>(np * HD)?);
        let (pm, pl, pa) = (*d_pm.device_ptr() as u64, *d_pl.device_ptr() as u64, *d_pa.device_ptr() as u64);
        let g2: u64 = (NSD as u64) | ((SEL_MAX as u64) << 20) | ((mpf as u32 as u64) << 32);
        let smem = qsa_union_fits(fmt as u32, NH, NKV, HD, m, l).context("union contract")? as u32;
        let mut t = [0f64; 2];
        for (k, tk) in t.iter_mut().enumerate() {
            let run = || -> Result<()> {
                unsafe {
                    if k == 0 {
                        f_hg3.clone().launch(LaunchConfig { grid_dim: ((m * NKV * NSD) as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                            (pm, pl, pa, pqs, pc, psl, pps, psp, nh_p, g2 as i64, split_hd))?;
                    } else {
                        f_un.clone().launch(LaunchConfig { grid_dim: ((NKV * NSD * (NH / NKV / 4)) as u32, 1, 1),
                                                           block_dim: ((32 * m) as u32, 1, 1), shared_mem_bytes: smem },
                            (pm, pl, pa, pqs, pc, psl, pps, psp, nh_p, g2 as i64, split_hd, m as i32))?;
                    }
                }
                Ok(())
            };
            for _ in 0..3 { run()?; }
            dev.synchronize()?;
            let t0 = std::time::Instant::now();
            for _ in 0..20 { run()?; }
            dev.synchronize()?;
            *tk = t0.elapsed().as_secs_f64() * 1e6 / 20.0;
        }
        println!("    EXL3-ATTN-SPARSE timing {} ctx={ctx} m={m}: hg3 {:.1} us, union {:.1} us per layer call ({:.2}x; synthetic \
                  back-to-back launches of one layer's KV — L2-warm, launch-inclusive; ranks the kernels, does not price in-round)",
                 crate::exl3_forward::kv_fmt_name(fmt as u32), t[0], t[1], t[0] / t[1].max(1e-9));
    }
    Ok(())
}

