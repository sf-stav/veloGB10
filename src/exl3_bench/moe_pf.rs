//! The prefill MoE routed-expert chain probe (A5-P1).
//! Probe-only code of the `exl3_bench` module (see `mod.rs` for the map); shared helpers come from the parent.
//! Numerics contract (module header in `mod.rs`): the kernel decode is bitwise vs the host oracle; do not change reduction orders.

use super::*;

// ---------------------------------------------------------------------------
// A5-P1 (PLAN/notes_2026-09-27/P1.md): the prefill MoE routed-expert chain.
// Reference = the served device-routed chain (xq_moe_pf_count at the PFX1 rule MT ->
// xq_moe_pf_place -> xq_had_suh_rows -> exl3_hmma_gemm_wide_tiles -> xq_had_svh_rows ->
// xq_moe_gate_mul -> xq_had_suh_rows -> exl3_hmma_gemm_wide_tiles -> xq_had_svh_rows) over 512
// synthetic experts (distinct trellis sets, 942 MB: DRAM-cold like serving) and skewed top-10
// routing (log-normal expert popularity: a hot tail with >= 128-row experts) at the served chunk
// classes c = 2048, 775 (a partial last chunk), 96 (MT 1 rule). Every candidate is compared
// BITWISE (outputs NaN-poisoned first) on every compact row:
//   * prefill.moe_kernel: xq_moe_pf_t{1,2,4,8} on both sides (y_raw) vs the reference y_raw;
//   * prefill.moe_mt_per_expert: xq_moe_pf_count4's lists (cnt / offs_row == xq_moe_pf_count's),
//     per-class launches with the served kernel AND the pipelined one;
//   * prefill.moe_glue_fold: xq_moe_pf_g{MT} (gate|up + svh + gate_mul + down suh) -> xhd, and
//     xq_moe_pf_s{MT} (down + svh) -> yd, uniform MT and per-class.
// Then a harness-only timing of the chains (ranks; the served number is the prefill A/B).
// ---------------------------------------------------------------------------
pub(super) fn probe_moe_pf(dev: &std::sync::Arc<CudaDevice>, bits: usize) -> Result<()> {
    use crate::exl3_forward::{moe_pf_class_caps, moe_pf_fn, moe_pf_smem_bytes, pfx1_moe_mt, moe_pf2_fn, moe_pf2_smem_bytes, moe_pf2h_smem_bytes};
    let (h, mi, ne, topk) = (2560usize, 640usize, 512usize, 10usize);
    // PFHALF: a fractional width code (K = KA + 0.5 as 16 + KA): the HALF ring is 4*(2*KA+1) u32 = 8*(2*KA+1) u16 per
    // 16x16 tile; the anchor is wide_tiles (EXL3_WT_CASE_H) and the pf2 twins are skipped (no HALF body there).
    let frac = crate::exl3_forward::exl3_is_frac_bits(bits);
    let tw16 = if frac { 8 * (2 * (bits - crate::exl3_forward::EXL3_FRAC_BITS_BASE) + 1) } else { 16 * bits };
    let f = |n: &str| dev.get_func(MODULE, n).with_context(|| n.to_string());
    // smem mirror
    {
        let mut d_q = dev.alloc_zeros::<i32>(8)?;
        unsafe { f("xq_moe_pf_smem_query")?.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 }, (&mut d_q,))? };
        let q = dev.dtoh_sync_copy(&d_q)?;
        for (i, (mt, nw)) in [(1, 8), (2, 8), (4, 8), (8, 8), (1, 16), (2, 16), (4, 16), (8, 16)].iter().enumerate() {
            if q[i] as u32 != moe_pf_smem_bytes(*mt, *nw) {
                bail!("EXL3-MOE-PF FAIL: smem mirror mt {mt} nw {nw}: kernel {} host {}", q[i], moe_pf_smem_bytes(*mt, *nw));
            }
            if q[i] > 48 * 1024 { bail!("EXL3-MOE-PF FAIL: mt {mt} nw {nw} needs {} B > 48 KB (no opt-in)", q[i]); }
        }
        // A5-L2: the pf2 instances, bits 3 and 4 (bits 3 = the served packs must fit 48 KB)
        let mut d_q2 = dev.alloc_zeros::<i32>(60)?;
        unsafe { f("xq_moe_pf2_smem_query")?.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 }, (&mut d_q2,))? };
        let q2 = dev.dtoh_sync_copy(&d_q2)?;
        for (i, (mt, nw)) in [(1, 8), (2, 8), (4, 8), (8, 8), (1, 16), (2, 16), (4, 16), (8, 16)].iter().enumerate() {
            for (b, bits) in [(0usize, 3usize), (1, 4)] {
                if q2[b * 8 + i] as u32 != moe_pf2_smem_bytes(*mt, *nw, bits) {
                    bail!("EXL3-MOE-PF FAIL: pf2 smem mirror mt {mt} nw {nw} bits {bits}: kernel {} host {}",
                          q2[b * 8 + i], moe_pf2_smem_bytes(*mt, *nw, bits));
                }
            }
            if q2[i] > 48 * 1024 { bail!("EXL3-MOE-PF FAIL: pf2 mt {mt} nw {nw} bits 3 needs {} B > 48 KB", q2[i]); }
        }
        for (i, (mt, bits)) in [(1usize, 3usize), (2, 3), (1, 4), (2, 4)].iter().enumerate() {
            if q2[16 + i] as u32 != moe_pf2h_smem_bytes(*mt, *bits) {
                bail!("EXL3-MOE-PF FAIL: pf2 paired smem mirror mt {mt} bits {bits}: kernel {} host {}", q2[16 + i], moe_pf2h_smem_bytes(*mt, *bits));
            }
        }
        if q2[16] > 48 * 1024 || q2[17] > 48 * 1024 { bail!("EXL3-MOE-PF FAIL: pf2 paired bits 3 over 48 KB"); }
        // P7 B4: bits-5 rows (paired h at 20/21; t/s/g in the bits-3 order at 22..29). Over 48 KB
        // is LEGAL at b5: the served picker (moe_pf_pick) falls back to P1 per instance — no bail.
        for (i, (mt5, nw5)) in [(1usize, 8usize), (2, 8), (4, 8), (8, 8), (1, 16), (2, 16), (4, 16), (8, 16)].iter().enumerate() {
            if q2[22 + i] as u32 != moe_pf2_smem_bytes(*mt5, *nw5, 5) {
                bail!("EXL3-MOE-PF FAIL: pf2 smem mirror mt {mt5} nw {nw5} bits 5: kernel {} host {}", q2[22 + i], moe_pf2_smem_bytes(*mt5, *nw5, 5));
            }
        }
        for (i, mt5) in [1usize, 2usize].iter().enumerate() {
            if q2[20 + i] as u32 != moe_pf2h_smem_bytes(*mt5, 5) {
                bail!("EXL3-MOE-PF FAIL: pf2 paired smem mirror mt {mt5} bits 5: kernel {} host {}", q2[20 + i], moe_pf2h_smem_bytes(*mt5, 5));
            }
        }
        // P7 B4 leg 2: bits 2/6/7 blocks at 30..59 (t/s +4, g +4, h +2 per width). Over 48 KB is
        // legal here: the served picker (moe_pf_pick) substitutes the P1 twin per instance.
        for (bi, wb) in [2usize, 6, 7].iter().enumerate() {
            let base = 30 + bi * 10;
            for (i, (mt, nw)) in [(1, 8), (2, 8), (4, 8), (8, 8), (1, 16), (2, 16), (4, 16), (8, 16)].iter().enumerate() {
                if q2[base + i] as u32 != moe_pf2_smem_bytes(*mt, *nw, *wb) {
                    bail!("EXL3-MOE-PF FAIL: pf2 smem mirror mt {mt} nw {nw} bits {wb}: kernel {} host {}",
                          q2[base + i], moe_pf2_smem_bytes(*mt, *nw, *wb));
                }
            }
            for (i, mt) in [1usize, 2].iter().enumerate() {
                if q2[base + 8 + i] as u32 != moe_pf2h_smem_bytes(*mt, *wb) {
                    bail!("EXL3-MOE-PF FAIL: pf2 paired smem mirror mt {mt} bits {wb}: kernel {} host {}",
                          q2[base + 8 + i], moe_pf2h_smem_bytes(*mt, *wb));
                }
            }
        }
        let over48: Vec<&str> = [("t/s", 22usize, 26usize), ("g", 26, 30), ("h", 20, 22)].iter()
            .flat_map(|&(nm, a, b)| (a..b).filter(|&i| q2[i] > 48 * 1024).map(move |i| nm)).collect();
        println!("  EXL3-MOE-PF pf2 smem bits3 t/s {:?} g {:?} h {:?} | bits5 t/s {:?} g {:?} h {:?} B (over 48KB at b5: {} instances)",
                 &q2[0..4], &q2[4..8], &q2[16..18], &q2[22..26], &q2[26..30], &q2[20..22], over48.len());
    }
    let gu_words = (h / 16) * (2 * mi / 16) * tw16;
    let d_words = (mi / 16) * (h / 16) * tw16;
    let d_gu = dev.htod_sync_copy(&wp20_synth(ne * gu_words, 0x7001, 1.0))?;
    let d_dn = dev.htod_sync_copy(&wp20_synth(ne * d_words, 0x7002, 1.0))?;
    let d_suh_gu = dev.htod_sync_copy(&wp20_synth(ne * h, 0x7003, 0.25))?;
    let d_svh_gu = dev.htod_sync_copy(&wp20_synth(ne * 2 * mi, 0x7004, 0.005))?;
    let d_suh_d = dev.htod_sync_copy(&wp20_synth(ne * mi, 0x7005, 0.02))?;
    let d_svh_d = dev.htod_sync_copy(&wp20_synth(ne * h, 0x7006, 0.01))?;
    let d_offs_gu = dev.htod_sync_copy(&(0..ne as u64).map(|e| e * gu_words as u64).collect::<Vec<u64>>())?;
    let d_offs_d = dev.htod_sync_copy(&(0..ne as u64).map(|e| e * d_words as u64).collect::<Vec<u64>>())?;
    let d_ident = dev.htod_sync_copy(&(0..ne as i32).collect::<Vec<i32>>())?;
    let (svh_gu, suh_d, svh_d) = (*d_svh_gu.device_ptr() as u64, *d_suh_d.device_ptr() as u64, *d_svh_d.device_ptr() as u64);
    // log-normal popularity (sigma 1.1), cumulative table
    let mut s = 0x1234_5678u32;
    let mut unif = || { s = s.wrapping_mul(1664525).wrapping_add(1013904223); ((s >> 8) as f64 + 0.5) / 16777216.0 };
    let mut cum = Vec::with_capacity(ne);
    let mut tot = 0.0f64;
    for _ in 0..ne {
        let z = (-2.0 * unif().ln()).sqrt() * (2.0 * std::f64::consts::PI * unif()).cos();
        tot += (1.1 * z).exp();
        cum.push(tot);
    }
    let lc = |gx: u32, gy: u32, b: u32, sm: u32| LaunchConfig { grid_dim: (gx, gy, 1), block_dim: (b, 1, 1), shared_mem_bytes: sm };
    let mut timing: Vec<String> = Vec::new();
    let mut nchk = 0usize;
    let mut pf2_subs = 0usize;   // P7 B4: pf2 launches over 48 KB at this width, run as the P1 twin (moe_pf_pick parity)
    // B4 leg 2 (K=2/6/7): wide_tiles serves 3..5 only. Outside that set the probe anchors on the
    // P1 twin — its decode class is proven by the B2 host-oracle refs (K=2/7) and the B7 b6
    // grouped-xh chain; the wide anchor returns with B5.
    let wc = if frac || matches!(bits, 3 | 4 | 5) { 'w' } else { 't' };
    let anchor = if wc == 'w' { "wide_tiles" } else { "pf P1 (wide_tiles refuses this width)" };
    let kinds_w: Vec<char> = if frac { vec!['w', 't'] } else if wc == 'w' { vec!['w', 't', 'T'] } else { vec!['t', 'T'] };
    let folds: Vec<(char, char, &str)> = if frac { vec![('g', 's', "fold"), ('P', 'P', "pf prof fold")] } else {
        vec![('g', 's', "fold"), ('G', 'S', "pf2 fold"), ('H', 'S', "pf2 paired fold"), ('P', 'P', "pf prof fold"),
             ('Q', 'Q', "pf2 prof fold"), ('R', 'Q', "pf2 paired prof fold")] };
    for &c in &[2048usize, 775, 96] {
        let r = c * topk;
        let max_tiles = r / 16 + ne;
        let caps = moe_pf_class_caps(r, ne);
        let mt_rule = pfx1_moe_mt(c);
        let mut ids: Vec<i32> = Vec::with_capacity(r);
        for _ in 0..c {
            let mut picked: Vec<i32> = Vec::new();
            while picked.len() < topk {
                let u = unif() * tot;
                let e = cum.partition_point(|&v| v < u).min(ne - 1) as i32;
                if !picked.contains(&e) { picked.push(e); }
            }
            ids.extend(picked);
        }
        let mut hist = vec![0usize; ne];
        for &e in &ids { hist[e as usize] += 1; }
        let cls_n: Vec<usize> = [(1usize, 16usize), (17, 32), (33, 127), (128, usize::MAX)].iter()
            .map(|&(a, b)| hist.iter().filter(|&&n| n >= a && n <= b).count()).collect();
        let d_ids = dev.htod_sync_copy(&ids)?;
        let d_x = dev.htod_sync_copy(&wp20_synth(c * h, 0x7010 + c as u32, 1.0))?;
        let d_dident = dev.htod_sync_copy(&(0..r as i32).collect::<Vec<i32>>())?;
        let d_moerows = dev.htod_sync_copy(&[r as i32])?;
        let mut d_cnt = dev.alloc_zeros::<i32>(ne)?;
        let mut d_offs_row = dev.alloc_zeros::<i32>(ne + 1)?;
        let mut d_tiles = dev.alloc_zeros::<i32>(2 * max_tiles)?;
        let mut d_nt = dev.alloc_zeros::<i32>(1)?;
        let mut d_tiles4 = dev.alloc_zeros::<i32>(8 * max_tiles)?;
        let mut d_nt4 = dev.alloc_zeros::<i32>(4)?;
        let mut d_row_tok = dev.alloc_zeros::<i32>(r)?;
        let mut d_row_eidx = dev.alloc_zeros::<i32>(r)?;
        let mut d_cand = dev.alloc_zeros::<i32>(r)?;
        let mut d_xh = dev.alloc_zeros::<u16>(r * h)?;
        let mut d_ygu_raw = dev.alloc_zeros::<u16>(r * 2 * mi)?;
        let mut d_ygu = dev.alloc_zeros::<u16>(r * 2 * mi)?;
        let mut d_din = dev.alloc_zeros::<u16>(r * mi)?;
        let mut d_xhd = dev.alloc_zeros::<u16>(r * mi)?;
        let mut d_xhd2 = dev.alloc_zeros::<u16>(r * mi)?;
        let mut d_ydr = dev.alloc_zeros::<u16>(r * h)?;
        let mut d_yd = dev.alloc_zeros::<u16>(r * h)?;
        let kn_gu = (h | ((2 * mi) << 16)) as i32;
        let kn_d = (mi | (h << 16)) as i32;
        let tl4 = *d_tiles4.device_ptr() as u64;
        let nt4 = *d_nt4.device_ptr() as u64;
        let tl1 = *d_tiles.device_ptr() as u64;
        let nt1 = *d_nt.device_ptr() as u64;
        let groups_pe: Vec<(usize, u64, u64, u32)> = (0..4).filter(|&k| caps[k] > 0)
            .map(|k| (1usize << k, tl4 + (k * max_tiles * 8) as u64, nt4 + (k * 4) as u64, caps[k] as u32)).collect();
        let groups_1 = |mt: usize| vec![(mt, tl1, nt1, max_tiles as u32)];
        // ---- launch helpers ----
        macro_rules! count1 { ($mt:expr) => { unsafe { f("xq_moe_pf_count")?.launch(lc(1, 1, 1024, 0),
            (&d_ids, r as i32, ne as i32, &mut d_cnt, &mut d_offs_row, &mut d_tiles, &mut d_nt, (16 * $mt) as i32))? } } }
        macro_rules! count4 { () => { unsafe { f("xq_moe_pf_count4")?.launch(lc(1, 1, 1024, 0),
            (&d_ids, r as i32, ne as i32, &mut d_cnt, &mut d_offs_row, &mut d_tiles4, &mut d_nt4, max_tiles as i32))? } } }
        // gate/up side: kind 'w' = served wide_tiles, 't' pipelined, 'g' fold (writes xhd2);
        // A5-L2: 'T' / 'G' = the pf2 twins of 't' / 'g', 'P' / 'Q' = the pf / pf2 fold `_prof` twins.
        let prof_fn = |v: usize, kind: char, mt: usize| -> String {
            format!("xq_moe_pf{}_prof_{kind}{mt}", if v == 0 { "" } else { "2" })
        };
        macro_rules! gu { ($kind:expr, $groups:expr) => { for &(gmt, tp, np, cap) in $groups.iter() {
            let mut kind = $kind;
            if kind == 'T' && moe_pf2_smem_bytes(gmt, 8, bits) > 48 * 1024 { kind = 't'; pf2_subs += 1; }
            else if (kind == 'G' || kind == 'Q') && moe_pf2_smem_bytes(gmt, 16, bits) > 48 * 1024 { kind = if kind == 'G' { 'g' } else { 'P' }; pf2_subs += 1; }
            else if kind == 'S' && moe_pf2_smem_bytes(gmt, 8, bits) > 48 * 1024 { kind = 's'; pf2_subs += 1; }
            else if (kind == 'H' || kind == 'R') && (if gmt <= 2 { moe_pf2h_smem_bytes(gmt, bits) } else { moe_pf2_smem_bytes(gmt, 16, bits) }) > 48 * 1024 { kind = if kind == 'H' { 'g' } else { 'P' }; pf2_subs += 1; }
            match kind {
                'T' => unsafe { f(moe_pf2_fn('t', gmt))?.launch(lc((2 * mi / 128) as u32, cap, 256, moe_pf2_smem_bytes(gmt, 8, bits)),
                    (&d_gu, &d_offs_gu, &d_xh, &mut d_ygu_raw, tp, np, &d_offs_row, &d_cnt, kn_gu, bits as i32, 0u64, 0u64))? },
                'G' | 'Q' => unsafe { f(&if $kind == 'G' { moe_pf2_fn('g', gmt).to_string() } else { prof_fn(1, 'g', gmt) })?
                    .launch(lc((2 * mi / 256) as u32, cap, 512, moe_pf2_smem_bytes(gmt, 16, bits)),
                    (&d_gu, &d_offs_gu, &d_xh, &mut d_xhd2, tp, np, &d_offs_row, &d_cnt, kn_gu, bits as i32, svh_gu, suh_d))? },
                'P' => unsafe { f(&prof_fn(0, 'g', gmt))?.launch(lc((2 * mi / 256) as u32, cap, 512, moe_pf_smem_bytes(gmt, 16)),
                    (&d_gu, &d_offs_gu, &d_xh, &mut d_xhd2, tp, np, &d_offs_row, &d_cnt, kn_gu, bits as i32, svh_gu, suh_d))? },
                'H' | 'R' if gmt <= 2 => unsafe { f(&format!("xq_moe_pf2_{}h{gmt}", if $kind == 'R' { "prof_" } else { "" }))?
                    .launch(lc((2 * mi / 256) as u32, cap, 256, moe_pf2h_smem_bytes(gmt, bits)),
                    (&d_gu, &d_offs_gu, &d_xh, &mut d_xhd2, tp, np, &d_offs_row, &d_cnt, kn_gu, bits as i32, svh_gu, suh_d))? },
                'H' | 'R' => unsafe { f(&if $kind == 'H' { moe_pf2_fn('g', gmt).to_string() } else { prof_fn(1, 'g', gmt) })?
                    .launch(lc((2 * mi / 256) as u32, cap, 512, moe_pf2_smem_bytes(gmt, 16, bits)),
                    (&d_gu, &d_offs_gu, &d_xh, &mut d_xhd2, tp, np, &d_offs_row, &d_cnt, kn_gu, bits as i32, svh_gu, suh_d))? },
                'w' => unsafe { f("exl3_hmma_gemm_wide_tiles")?.launch(lc((2 * mi / 128) as u32, cap, 256, 0),
                    (&d_gu, &d_offs_gu, &d_xh, &mut d_ygu_raw, tp, np, &d_offs_row, &d_cnt, h as i32, (2 * mi) as i32, bits as i32, gmt as i32))? },
                't' => unsafe { f(moe_pf_fn('t', gmt))?.launch(lc((2 * mi / 128) as u32, cap, 256, moe_pf_smem_bytes(gmt, 8)),
                    (&d_gu, &d_offs_gu, &d_xh, &mut d_ygu_raw, tp, np, &d_offs_row, &d_cnt, kn_gu, bits as i32, 0u64, 0u64))? },
                _ => unsafe { f(moe_pf_fn('g', gmt))?.launch(lc((2 * mi / 256) as u32, cap, 512, moe_pf_smem_bytes(gmt, 16)),
                    (&d_gu, &d_offs_gu, &d_xh, &mut d_xhd2, tp, np, &d_offs_row, &d_cnt, kn_gu, bits as i32, svh_gu, suh_d))? },
            } } } }
        // down side from `$xin`: 'w' / 't' -> yd_raw, 's' -> yd
        macro_rules! dn { ($kind:expr, $groups:expr, $xin:expr) => { for &(gmt, tp, np, cap) in $groups.iter() {
            let mut kind = $kind;
            if kind == 'T' && moe_pf2_smem_bytes(gmt, 8, bits) > 48 * 1024 { kind = 't'; pf2_subs += 1; }
            else if (kind == 'G' || kind == 'Q') && moe_pf2_smem_bytes(gmt, 16, bits) > 48 * 1024 { kind = if kind == 'G' { 'g' } else { 'P' }; pf2_subs += 1; }
            else if kind == 'S' && moe_pf2_smem_bytes(gmt, 8, bits) > 48 * 1024 { kind = 's'; pf2_subs += 1; }
            else if (kind == 'H' || kind == 'R') && (if gmt <= 2 { moe_pf2h_smem_bytes(gmt, bits) } else { moe_pf2_smem_bytes(gmt, 16, bits) }) > 48 * 1024 { kind = if kind == 'H' { 'g' } else { 'P' }; pf2_subs += 1; }
            match kind {
                'T' => unsafe { f(moe_pf2_fn('t', gmt))?.launch(lc((h / 128) as u32, cap, 256, moe_pf2_smem_bytes(gmt, 8, bits)),
                    (&d_dn, &d_offs_d, $xin, &mut d_ydr, tp, np, &d_offs_row, &d_cnt, kn_d, bits as i32, 0u64, 0u64))? },
                'S' | 'Q' => unsafe { f(&if $kind == 'S' { moe_pf2_fn('s', gmt).to_string() } else { prof_fn(1, 's', gmt) })?
                    .launch(lc((h / 128) as u32, cap, 256, moe_pf2_smem_bytes(gmt, 8, bits)),
                    (&d_dn, &d_offs_d, $xin, &mut d_yd, tp, np, &d_offs_row, &d_cnt, kn_d, bits as i32, svh_d, 0u64))? },
                'P' => unsafe { f(&prof_fn(0, 's', gmt))?.launch(lc((h / 128) as u32, cap, 256, moe_pf_smem_bytes(gmt, 8)),
                    (&d_dn, &d_offs_d, $xin, &mut d_yd, tp, np, &d_offs_row, &d_cnt, kn_d, bits as i32, svh_d, 0u64))? },
                'w' => unsafe { f("exl3_hmma_gemm_wide_tiles")?.launch(lc((h / 128) as u32, cap, 256, 0),
                    (&d_dn, &d_offs_d, $xin, &mut d_ydr, tp, np, &d_offs_row, &d_cnt, mi as i32, h as i32, bits as i32, gmt as i32))? },
                't' => unsafe { f(moe_pf_fn('t', gmt))?.launch(lc((h / 128) as u32, cap, 256, moe_pf_smem_bytes(gmt, 8)),
                    (&d_dn, &d_offs_d, $xin, &mut d_ydr, tp, np, &d_offs_row, &d_cnt, kn_d, bits as i32, 0u64, 0u64))? },
                _ => unsafe { f(moe_pf_fn('s', gmt))?.launch(lc((h / 128) as u32, cap, 256, moe_pf_smem_bytes(gmt, 8)),
                    (&d_dn, &d_offs_d, $xin, &mut d_yd, tp, np, &d_offs_row, &d_cnt, kn_d, bits as i32, svh_d, 0u64))? },
            } } } }
        macro_rules! place_suh { () => { unsafe {
            f("xq_moe_pf_place")?.launch(lc(ne as u32, 1, 256, 0),
                (&d_ids, r as i32, topk as i32, &d_offs_row, &mut d_row_tok, &mut d_row_eidx, &mut d_cand))?;
            f("xq_had_suh_rows")?.launch(lc((r * 20) as u32, 1, 32, 0),
                (&d_x, &d_suh_gu, &d_ident, &d_row_tok, &d_row_eidx, &mut d_xh, r as i32, h as i32))?;
        } } }
        macro_rules! glue_gu { () => { unsafe {
            f("xq_had_svh_rows")?.launch(lc((r * 10) as u32, 1, 32, 0),
                (&d_ygu_raw, &d_svh_gu, &d_ident, &d_row_eidx, &mut d_ygu, r as i32, (2 * mi) as i32))?;
            f("xq_moe_gate_mul")?.launch(lc(((r * mi) as u32 + 255) / 256, 1, 256, 0), (&mut d_din, &d_ygu, mi as i64, &d_moerows))?;
            f("xq_had_suh_rows")?.launch(lc((r * (mi / 128)) as u32, 1, 32, 0),
                (&d_din, &d_suh_d, &d_ident, &d_dident, &d_row_eidx, &mut d_xhd, r as i32, mi as i32))?;
        } } }
        macro_rules! glue_d { () => { unsafe {
            f("xq_had_svh_rows")?.launch(lc((r * (h / 128)) as u32, 1, 32, 0),
                (&d_ydr, &d_svh_d, &d_ident, &d_row_eidx, &mut d_yd, r as i32, h as i32))?;
        } } }
        let poison = |d: &mut CudaSlice<u16>| -> Result<()> { let n = d.len(); dev.htod_sync_copy_into(&vec![0xFFFFu16; n], d)?; Ok(()) };
        let cmp = |what: &str, a: &[u16], b: &[u16]| -> Result<()> {
            let bad = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
            if bad > 0 || a.len() != b.len() {
                let first = a.iter().zip(b.iter()).position(|(x, y)| x != y);
                bail!("EXL3-MOE-PF FAIL c={c} {what}: {bad} of {} f16 differ (first {:?})", a.len(), first);
            }
            Ok(())
        };
        // ---- reference: the served chain ----
        count1!(mt_rule);
        place_suh!();
        let cnt_ref = dev.dtoh_sync_copy(&d_cnt)?;
        let offs_ref = dev.dtoh_sync_copy(&d_offs_row)?;
        poison(&mut d_ygu_raw)?; poison(&mut d_xhd)?; poison(&mut d_ydr)?; poison(&mut d_yd)?;
        gu!(wc, groups_1(mt_rule));
        glue_gu!();
        dn!(wc, groups_1(mt_rule), &d_xhd);
        glue_d!();
        dev.synchronize()?;
        let ygu_ref = dev.dtoh_sync_copy(&d_ygu_raw)?;
        let xhd_ref = dev.dtoh_sync_copy(&d_xhd)?;
        let ydr_ref = dev.dtoh_sync_copy(&d_ydr)?;
        let yd_ref = dev.dtoh_sync_copy(&d_yd)?;
        // PR #32 port: the reference outputs' hashes (every candidate below must equal them bitwise in-run), so two
        // binaries that differ only in the prefill MoE kernels can be compared at EVERY width, incl. the ones whose
        // anchor is the pf kernel itself (2, 6, 7): diff the EXL3-MOE-PF-HASH lines of the two runs.
        {
            let fnv = |v: &[u16]| v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100_0000_01b3));
            println!("EXL3-MOE-PF-HASH bits={bits} c={c} anchor={wc} ygu_raw={:016x} xhd={:016x} yd_raw={:016x} yd={:016x}",
                     fnv(&ygu_ref), fnv(&xhd_ref), fnv(&ydr_ref), fnv(&yd_ref));
        }
        if ygu_ref.iter().any(|&v| v == 0xFFFF) || yd_ref.iter().any(|&v| v == 0xFFFF) || xhd_ref.iter().any(|&v| v == 0xFFFF) {
            bail!("EXL3-MOE-PF FAIL c={c}: the reference chain left poisoned rows");
        }
        // every candidate, uniform MT lists (the ref's routing tables are MT-independent)
        for mt in [1usize, 2, 4, 8] {
            count1!(mt);
            for kind in kinds_w.iter().copied() {
                poison(&mut d_ygu_raw)?; poison(&mut d_ydr)?;
                gu!(kind, groups_1(mt));
                dn!(kind, groups_1(mt), &d_xhd);   // the reference xhd as input
                dev.synchronize()?;
                cmp(&format!("{kind} mt {mt} ygu_raw"), &dev.dtoh_sync_copy(&d_ygu_raw)?, &ygu_ref)?;
                cmp(&format!("{kind} mt {mt} yd_raw"), &dev.dtoh_sync_copy(&d_ydr)?, &ydr_ref)?;
                nchk += 2;
            }
            for &(gk, dk, nm) in folds.iter() {
                poison(&mut d_xhd2)?; poison(&mut d_yd)?;
                gu!(gk, groups_1(mt));
                dn!(dk, groups_1(mt), &d_xhd);
                dev.synchronize()?;
                cmp(&format!("{nm} mt {mt} xhd"), &dev.dtoh_sync_copy(&d_xhd2)?, &xhd_ref)?;
                cmp(&format!("{nm} mt {mt} yd"), &dev.dtoh_sync_copy(&d_yd)?, &yd_ref)?;
                nchk += 2;
            }
        }
        // WP19 per-expert lists
        count4!();
        dev.synchronize()?;
        cmp("count4 cnt", &dev.dtoh_sync_copy(&d_cnt)?.iter().map(|v| *v as u16).collect::<Vec<_>>(),
            &cnt_ref.iter().map(|v| *v as u16).collect::<Vec<_>>())?;
        if dev.dtoh_sync_copy(&d_offs_row)? != offs_ref { bail!("EXL3-MOE-PF FAIL c={c}: count4 offs_row != count's"); }
        let nt4h = dev.dtoh_sync_copy(&d_nt4)?;
        for k in 0..4 { if nt4h[k] as usize > caps[k] { bail!("EXL3-MOE-PF FAIL c={c}: class {k} has {} tiles > cap {}", nt4h[k], caps[k]); } }
        for kind in kinds_w.iter().copied() {
            poison(&mut d_ygu_raw)?; poison(&mut d_ydr)?;
            gu!(kind, groups_pe);
            dn!(kind, groups_pe, &d_xhd);
            dev.synchronize()?;
            cmp(&format!("{kind} per-expert ygu_raw"), &dev.dtoh_sync_copy(&d_ygu_raw)?, &ygu_ref)?;
            cmp(&format!("{kind} per-expert yd_raw"), &dev.dtoh_sync_copy(&d_ydr)?, &ydr_ref)?;
            nchk += 2;
        }
        for &(gk, dk, nm) in folds.iter() {
            poison(&mut d_xhd2)?; poison(&mut d_yd)?;
            gu!(gk, groups_pe);
            dn!(dk, groups_pe, &d_xhd2);      // chained on the FOLDED xhd: the served fold chain end to end
            dev.synchronize()?;
            cmp(&format!("{nm} per-expert xhd"), &dev.dtoh_sync_copy(&d_xhd2)?, &xhd_ref)?;
            cmp(&format!("{nm} per-expert yd (chained)"), &dev.dtoh_sync_copy(&d_yd)?, &yd_ref)?;
            nchk += 2;
        }
        // A5-L2: every pf2 mask (mixed pf / pf2 classes in one chain) == the reference (none for a fractional width)
        for mask in if frac { vec![] } else { vec![1i32, 2, 4, 8, 3, 7, 15] } {
            poison(&mut d_xhd2)?; poison(&mut d_yd)?;
            for &g in groups_pe.iter() { let k = g.0.trailing_zeros() as i32; gu!(if mask & (1 << k) != 0 { 'H' } else { 'g' }, [g]); }
            for &g in groups_pe.iter() { let k = g.0.trailing_zeros() as i32; dn!(if mask & (1 << k) != 0 { 'S' } else { 's' }, [g], &d_xhd2); }
            dev.synchronize()?;
            cmp(&format!("pf2 mask {mask} xhd"), &dev.dtoh_sync_copy(&d_xhd2)?, &xhd_ref)?;
            cmp(&format!("pf2 mask {mask} yd (chained)"), &dev.dtoh_sync_copy(&d_yd)?, &yd_ref)?;
            nchk += 2;
        }
        println!("  EXL3-MOE-PF bits={bits} c={c}: rows {r}, experts by count [1-16 {} | 17-32 {} | 33-127 {} | >=128 {}], \
                  max {}, rule MT {mt_rule}, per-expert tiles {:?}: every candidate bitwise == the served chain",
                 cls_n[0], cls_n[1], cls_n[2], cls_n[3], hist.iter().max().unwrap(), nt4h);
        // ---- timing (harness-only; 512 distinct experts = DRAM-cold trellis) ----
        if c >= 512 || crate::opts::var(crate::opt!("p1-time-all")).is_ok() {
            let reps = 8usize;
            let mut t = |name: &str, run: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
                run()?; dev.synchronize()?;
                let t0 = std::time::Instant::now();
                for _ in 0..reps { run()?; }
                dev.synchronize()?;
                let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
                timing.push(format!("c={c} {name}: {ms:.3} ms/layer"));
                Ok(ms)
            };
            let base = t(&format!("OFF   served ({anchor} rule MT + glue)"), &mut || { count1!(mt_rule); gu!(wc, groups_1(mt_rule)); glue_gu!(); dn!(wc, groups_1(mt_rule), &d_xhd); glue_d!(); Ok(()) })?;
            let g0 = t(&format!("  gemms only: {anchor} rule MT"), &mut || { gu!(wc, groups_1(mt_rule)); dn!(wc, groups_1(mt_rule), &d_xhd); Ok(()) })?;
            let g1 = t("  gemms only: pf_t rule MT", &mut || { gu!('t', groups_1(mt_rule)); dn!('t', groups_1(mt_rule), &d_xhd); Ok(()) })?;
            for mt in [2usize, 4, 8] {
                count1!(mt);
                t(&format!("  gemms only: pf_t MT {mt}"), &mut || { gu!('t', groups_1(mt)); dn!('t', groups_1(mt), &d_xhd); Ok(()) })?;
            }
            count4!();
            let g2 = t(&format!("  gemms only: {anchor} per-expert MT"), &mut || { gu!(wc, groups_pe); dn!(wc, groups_pe, &d_xhd); Ok(()) })?;
            let g3 = t("  gemms only: pf_t per-expert MT", &mut || { gu!('t', groups_pe); dn!('t', groups_pe, &d_xhd); Ok(()) })?;
            let l1 = t("L1    pf_t rule MT + glue", &mut || { count1!(mt_rule); gu!('t', groups_1(mt_rule)); glue_gu!(); dn!('t', groups_1(mt_rule), &d_xhd); glue_d!(); Ok(()) })?;
            let l12 = t("L1+2  pf_t per-expert + glue", &mut || { count4!(); gu!('t', groups_pe); glue_gu!(); dn!('t', groups_pe, &d_xhd); glue_d!(); Ok(()) })?;
            let l2 = t(&format!("L2    {anchor} per-expert + glue"), &mut || { count4!(); gu!(wc, groups_pe); glue_gu!(); dn!(wc, groups_pe, &d_xhd); glue_d!(); Ok(()) })?;
            let all = t("ALL   fold per-expert", &mut || { count4!(); gu!('g', groups_pe); dn!('s', groups_pe, &d_xhd2); Ok(()) })?;
            let l13 = t("L1+3  fold rule MT", &mut || { count1!(mt_rule); gu!('g', groups_1(mt_rule)); dn!('s', groups_1(mt_rule), &d_xhd2); Ok(()) })?;
            let mut l2x: Vec<String> = Vec::new();
            // A5-L2: the served ALL chain with pf2 per MT-class mask, then each class alone (pf vs pf2)
            let mut mrow = Vec::new();
            for mask in if frac { vec![] } else { vec![0i32, 1, 2, 4, 8, 3, 7, 15] } {
                let ms = t(&format!("L2    ALL + pf2 mask {mask}"), &mut || {
                    count4!();
                    for &g in groups_pe.iter() { let k = g.0.trailing_zeros() as i32; gu!(if mask & (1 << k) != 0 { 'H' } else { 'g' }, [g]); }
                    for &g in groups_pe.iter() { let k = g.0.trailing_zeros() as i32; dn!(if mask & (1 << k) != 0 { 'S' } else { 's' }, [g], &d_xhd2); }
                    Ok(()) })?;
                mrow.push(format!("{mask}:{:.1}", ms * 48.0));
            }
            l2x.push(format!("c={c} L2 SUMMARY x48 (ms/chunk) ALL by pf2 mask {}", mrow.join(" | ")));
            let (gu_b, d_b) = ((gu_words * 2) as f64, (d_words * 2) as f64);
            for &g in groups_pe.iter() {
                let k = g.0.trailing_zeros() as usize;
                let ntl = nt4h[k] as f64;
                if ntl == 0.0 || frac { continue; }
                let a = t(&format!("  class MT {} g  pf ", g.0), &mut || { gu!('g', [g]); Ok(()) })?;
                let b = t(&format!("  class MT {} g  pf2", g.0), &mut || { gu!('G', [g]); Ok(()) })?;
                let bh = if g.0 <= 2 { t(&format!("  class MT {} g  pf2 paired", g.0), &mut || { gu!('H', [g]); Ok(()) })? } else { b };
                let cc = t(&format!("  class MT {} s  pf ", g.0), &mut || { dn!('s', [g], &d_xhd2); Ok(()) })?;
                let d = t(&format!("  class MT {} s  pf2", g.0), &mut || { dn!('S', [g], &d_xhd2); Ok(()) })?;
                l2x.push(format!("c={c} L2 CLASS MT {} ({} tiles): g pf {:.3} ms ({:.0} GB/s) -> pf2 {:.3} ms ({:.0} GB/s), paired {:.3} ms ({:.0} GB/s) | s pf {:.3} ms ({:.0} GB/s) -> pf2 {:.3} ms ({:.0} GB/s)  [trellis GB/s = tiles x expert bytes / time]",
                    g.0, nt4h[k], a, ntl * gu_b / a / 1e6, b, ntl * gu_b / b / 1e6, bh, ntl * gu_b / bh / 1e6, cc, ntl * d_b / cc / 1e6, d, ntl * d_b / d / 1e6));
            }
            // `_prof` anatomy (thread 0 of each live CTA: stage-top wait / decode+mma / epilogue / span)
            if !frac {
                let mut d_pr = dev.alloc_zeros::<u64>(120)?;
                unsafe { f("xq_moe_pf_prof_take")?.launch(lc(1, 1, 128, 0), (&mut d_pr,))? };
                gu!('P', groups_pe); dn!('P', groups_pe, &d_xhd2);
                gu!('Q', groups_pe); dn!('Q', groups_pe, &d_xhd2);
                gu!('R', groups_pe);
                unsafe { f("xq_moe_pf_prof_take")?.launch(lc(1, 1, 128, 0), (&mut d_pr,))? };
                dev.synchronize()?;
                let pr = dev.dtoh_sync_copy(&d_pr)?;
                let mut slots: Vec<(usize, &str, char, usize)> = Vec::new();
                for v in 0..2usize {
                    for (ki, kind) in ['s', 'g'].iter().enumerate() {
                        for (mi2, mt) in [1usize, 2, 4, 8].iter().enumerate() {
                            slots.push((v * 8 + ki * 4 + mi2, if v == 0 { "pf " } else { "pf2" }, *kind, *mt));
                        }
                    }
                }
                slots.push((16, "pf2", 'h', 1)); slots.push((17, "pf2", 'h', 2));
                {
                    {
                        for &(sl, vn, kind, mt) in slots.iter() {
                            let x = &pr[sl * 5..sl * 5 + 5];
                            if x[4] == 0 { continue; }
                            let n = x[4] as f64;
                            let span = x[3] as f64 / n;
                            l2x.push(format!("c={c} L2 PROF {} {kind}{mt}: {} CTAs, span {:.0} cyc/CTA = wait {:.1}% + decode/mma {:.1}% + epilogue {:.1}% (rest {:.1}%)",
                                vn, x[4], span, 100.0 * x[0] as f64 / n / span,
                                100.0 * x[1] as f64 / n / span, 100.0 * x[2] as f64 / n / span,
                                100.0 * (1.0 - (x[0] + x[1] + x[2]) as f64 / x[3] as f64)));
                        }
                    }
                }
            }
            timing.push(format!("c={c} SUMMARY x48 layers (ms/chunk): OFF {:.1} | L1 {:.1} | L2 {:.1} | L1+2 {:.1} | L1+3 {:.1} | ALL {:.1} \
                                 | gemms wide {:.1} -> pf {:.1} (wide pe {:.1}, pf pe {:.1})",
                                base * 48.0, l1 * 48.0, l2 * 48.0, l12 * 48.0, l13 * 48.0, all * 48.0, g0 * 48.0, g1 * 48.0, g2 * 48.0, g3 * 48.0));
            timing.extend(l2x);
        }
    }
    for l in &timing { println!("    EXL3-MOE-PF timing {l} (harness-only, serial launches, DRAM-cold trellis; ranks the paths)"); }
    if frac {
        println!("EXL3-MOE-PF bits={bits} (K = {}): PASS (PFHALF: xq_moe_pf_t{{1,2,4,8}} HALF body, per-expert MT lists, the folded xq_moe_pf_g/s chain \
                  and the pf `_prof` twin, all bitwise == the served wide_tiles HALF + glue chain on every compact row; {nchk} comparisons over \
                  c in {{2048, 775, 96}}, 512 experts, skewed top-10 routing; pf2 has no HALF body: not run)", crate::exl3_forward::exl3_width_name(bits));
        return Ok(());
    }
    println!("EXL3-MOE-PF bits={bits}: PASS (A5-P1: xq_moe_pf_t{{1,2,4,8}} (prefill.moe_kernel), per-expert MT lists (prefill.moe_mt_per_expert, \
              both kernels) and the folded xq_moe_pf_g/s chain (prefill.moe_glue_fold); A5-L2: the xq_moe_pf2 smem-ring twins \
              (prefill.moe_pf2: t/g/s every MT, per-expert, every class mask) and both `_prof` twins; all bitwise == the served {anchor} + glue chain (anchor: {anchor} at this width) \
              on every compact row; {nchk} comparisons over c in {{2048, 775, 96}}, 512 experts, skewed top-10 routing; pf2-over-48KB->P1 substitutions: {pf2_subs})");
    Ok(())
}

