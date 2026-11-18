//! QSA select and WP18 select probes.
//! Probe-only code of the `exl3_bench` module (see `mod.rs` for the map); shared helpers come from the parent.
//! Numerics contract (module header in `mod.rs`): the kernel decode is bitwise vs the host oracle; do not change reduction orders.

use super::*;

// ---------------------------------------------------------------------------
// A5-K3 (attn.qsa_select): xq_qsa_topk_asc3 must write BITWISE the same selection rows as the served
// xq_qsa_topk_asc2 — the WHOLE sel row (pre-filled -1: entries past pos_sel must stay untouched) and
// pos_sel — at 8K / 32K / 128K / 256K-sized score rows (served pitch max_pos/ratio = 65,536) plus the
// dense/sparse boundary (nb 511..513 around topk 512) and an unaligned pitch, widths m = 1..8 (rows at
// consecutive positions, as a verify), eleven key classes incl. adversarial ties: real-like, peaked,
// 64-level ties, 97% +0 (ties at T = +0), all-equal, two-valued (kp = K/2 of the lower value), raw u32
// bits (negatives / NaNs / -0 rank by u32 value in both), narrow (keys within 16 ulps: short digit),
// capflood (90% of keys in one 4,096-ulp cluster: first-digit bin over the list capacity -> a second
// global digit before the list), rising, falling; "two" at >= 32K has > 4,096 keys == T, i.e. asc3's
// global-row fallback path (d); all-equal and the boundary rows take its identity path (0). Bytes past each row's nb are poisoned with the max key (an over-read would win).
// Success signal: the asc2 reference itself == a CPU (key desc, index asc) top-K + ascending emission
// on every row of every m = 8 cell. Then a harness-only warm timing (ranks, does not price in-round).
// ---------------------------------------------------------------------------
pub(super) fn probe_qsa_select(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    const RATIO: usize = 4;
    const TOPK: usize = 512;
    const SEL_MAX: usize = 2048 + RATIO - 1;
    const MMAX: usize = 8;
    let fget = |n: &str| dev.get_func(MODULE, n).with_context(|| format!("{n} (stale PTX?)"));
    let (f2, f3) = (fget("xq_qsa_topk_asc2")?, fget("xq_qsa_topk_asc3")?);
    // QsaParams (64 B): 4 pointers (unused here) | eps | hd heads rdim ratio topk sel_max pad
    let mut pb: Vec<u8> = vec![0u8; 32];
    pb.extend_from_slice(&1e-6f32.to_le_bytes());
    for v in [128usize, 4, 64, RATIO, TOPK, SEL_MAX, 0] { pb.extend_from_slice(&(v as i32).to_le_bytes()); }
    let d_params = dev.htod_sync_copy(&pb)?;
    let pp = *d_params.device_ptr() as u64;
    let mut rs = 0x4B33_5E1E_2026_0928u64;
    let mut rnd = move || -> u64 { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; rs };
    let modes = ["real", "peaked", "ties64", "zeros", "allequal", "two", "bits", "narrow", "capflood", "rising", "falling"];
    // (name, p0 = row 0 position, pitch)
    let ctxs: [(&str, usize, usize); 6] = [("boundary", 2044, 65536), ("8K", 8192 - 16, 65536), ("8K-odd", 8192 - 16, 2053),
                                           ("32K", 32768 - 16, 65536), ("128K", 131072 - 16, 65536), ("256K", 262144 - 16, 65536)];
    let launch = |f: &cudarc::driver::CudaFunction, sel: u64, ps: u64, scp: u64, posp: u64, m: usize, pitch: usize| -> Result<()> {
        unsafe {
            f.clone().launch(LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                             (sel, ps, scp, posp, pp, pitch as i32, 0i32))?;
        }
        Ok(())
    };
    // CPU reference: (key desc, index asc) top-K blocks -> ascending token list + tail
    let cpu_ref = |keys: &[u32], pos: usize| -> Vec<i32> {
        let nb = (pos + 1) / RATIO;
        let mut idx: Vec<usize> = (0..nb).collect();
        if nb > TOPK {
            idx.sort_by(|&a, &b| keys[b].cmp(&keys[a]).then(a.cmp(&b)));
            idx.truncate(TOPK);
            idx.sort_unstable();
        }
        let mut v: Vec<i32> = idx.iter().flat_map(|&j| (0..RATIO).map(move |i| (j * RATIO + i) as i32)).collect();
        v.extend((nb * RATIO..=pos).map(|t| t as i32));
        v
    };
    let (mut cells, mut cpu_rows, mut tie_rows, mut flood_rows) = (0usize, 0usize, 0usize, 0usize);
    for &(cname, p0, pitch) in &ctxs {
        for &mode in &modes {
            let mut h: Vec<u32> = vec![0x7F7F_FFFFu32; MMAX * pitch];   // poison past nb: the max finite key
            for r in 0..MMAX {
                let nb = (p0 + r + 1) / RATIO;
                anyhow::ensure!(nb <= pitch, "row {r} nb {nb} > pitch {pitch}");
                let row = &mut h[r * pitch..r * pitch + nb];
                let u = |x: u64| (x >> 11) as f64 / (1u64 << 53) as f64;
                match mode {
                    "real" => for v in row.iter_mut() {
                        let s: f64 = (0..4).map(|_| (u(rnd()) * 2.0 - 1.3).max(0.0)).sum();
                        *v = ((s * 4.0 / 11.3137) as f32).to_bits();
                    },
                    "peaked" => for v in row.iter_mut() {
                        *v = if rnd() % 100 == 0 { (50.0 + u(rnd()) as f32).to_bits() } else { (1.0 + (rnd() % 1000) as f32 * 1e-6).to_bits() };
                    },
                    "ties64" => for v in row.iter_mut() { *v = ((rnd() % 64) as f32 * 0.125).to_bits(); },
                    "zeros" => for v in row.iter_mut() { *v = if rnd() % 100 < 97 { 0 } else { (u(rnd()) as f32 + 0.01).to_bits() }; },
                    "allequal" => row.fill(0.75f32.to_bits()),
                    "two" => {
                        row.fill(1.0f32.to_bits());
                        let mut c = 0;
                        while c < TOPK / 2 { let j = (rnd() as usize) % nb; if row[j] != 2.0f32.to_bits() { row[j] = 2.0f32.to_bits(); c += 1; } }
                    }
                    "bits" => for v in row.iter_mut() { *v = rnd() as u32; },
                    "narrow" => for v in row.iter_mut() { *v = 1.0f32.to_bits() + (rnd() % 16) as u32; },
                    "capflood" => for v in row.iter_mut() {
                        *v = if rnd() % 10 == 0 { ((u(rnd()) * 0.5) as f32).to_bits() } else { (1.0f32.to_bits() & !0xFFF) | (rnd() as u32 & 0xFFF) };
                    },
                    "rising" => for (j, v) in row.iter_mut().enumerate() { *v = (j as f32 * 0.01).to_bits(); },
                    _ => for (j, v) in row.iter_mut().enumerate() { *v = ((nb - j) as f32 * 0.01).to_bits(); },
                }
            }
            let d_sc = dev.htod_sync_copy(&h)?;
            let pos: Vec<i32> = (0..MMAX).map(|r| (p0 + r) as i32).collect();
            let d_pos = dev.htod_sync_copy(&pos)?;
            let (scp, posp) = (*d_sc.device_ptr() as u64, *d_pos.device_ptr() as u64);
            for m in 1..=MMAX {
                let d_s2 = dev.htod_sync_copy(&vec![-1i32; m * SEL_MAX])?;
                let d_s3 = dev.htod_sync_copy(&vec![-1i32; m * SEL_MAX])?;
                let d_p2 = dev.htod_sync_copy(&vec![-77i32; m])?;
                let d_p3 = dev.htod_sync_copy(&vec![-99i32; m])?;
                launch(&f2, *d_s2.device_ptr() as u64, *d_p2.device_ptr() as u64, scp, posp, m, pitch)?;
                launch(&f3, *d_s3.device_ptr() as u64, *d_p3.device_ptr() as u64, scp, posp, m, pitch)?;
                dev.synchronize()?;
                let (s2, s3) = (dev.dtoh_sync_copy(&d_s2)?, dev.dtoh_sync_copy(&d_s3)?);
                let (q2, q3) = (dev.dtoh_sync_copy(&d_p2)?, dev.dtoh_sync_copy(&d_p3)?);
                let nd = s2.iter().zip(&s3).filter(|(a, b)| a != b).count() + q2.iter().zip(&q3).filter(|(a, b)| a != b).count();
                if nd > 0 {
                    let first = (0..m * SEL_MAX).find(|&i| s2[i] != s3[i]);
                    bail!("EXL3-QSA-SELECT FAIL {cname} {mode} m={m}: asc3 vs asc2 differ in {nd} words (pos_sel asc2 {q2:?} asc3 {q3:?}; \
                           first entry {first:?} = row {:?}: asc2 {:?} asc3 {:?})", first.map(|f| f / SEL_MAX),
                          first.map(|f| s2[f]), first.map(|f| s3[f]));
                }
                if m == MMAX {
                    for r in 0..m {
                        let p = p0 + r;
                        let want = cpu_ref(&h[r * pitch..(r + 1) * pitch], p);
                        let got = &s2[r * SEL_MAX..r * SEL_MAX + want.len()];
                        if q2[r] != want.len() as i32 - 1 || got != &want[..] || s2[r * SEL_MAX + want.len()..(r + 1) * SEL_MAX].iter().any(|&x| x != -1) {
                            bail!("EXL3-QSA-SELECT FAIL {cname} {mode} row {r}: the asc2 reference != the CPU top-K (pos_sel {} vs {})",
                                  q2[r], want.len() as i32 - 1);
                        }
                        cpu_rows += 1;
                        // a tie row: the K-th largest key is shared with an unselected block
                        let nb = (p + 1) / RATIO;
                        if nb > TOPK {
                            let row = &h[r * pitch..r * pitch + nb];
                            let mut ks: Vec<u32> = row.to_vec();
                            ks.sort_unstable_by(|a, b| b.cmp(a));
                            let t = ks[TOPK - 1];
                            if ks[TOPK] == t { tie_rows += 1; }
                            // asc3's global-row fallback (d): > XQ_TK3_CAP keys equal to T (no digit's bin ever fits)
                            if ks.iter().filter(|&&k| k == t).count() > 4096 { flood_rows += 1; }
                        }
                    }
                }
                cells += 1;
            }
        }
        println!("    qsa-select {cname:<8} (p0 {p0}, pitch {pitch}): asc3 == asc2 (whole rows + pos_sel), m 1..=8 x {} key classes", modes.len());
    }
    anyhow::ensure!(tie_rows > 0 && flood_rows > 0, "EXL3-QSA-SELECT: the adversarial classes did not produce ties at T ({tie_rows}) \
                                                     or a fallback-path row ({flood_rows})");
    println!("EXL3-QSA-SELECT: PASS ({cells} cells: A5-K3 xq_qsa_topk_asc3 bitwise == xq_qsa_topk_asc2 on whole -1-prefilled sel rows + \
              pos_sel; asc2 == CPU (key desc, index asc) top-K on {cpu_rows} rows, {tie_rows} with ties at T, {flood_rows} on asc3's \
              global-row fallback (> {} keys == T))", 4096);
    // ---- harness-only warm timing (rows L2-resident as in-round after the scorer; 50 back-to-back launches)
    for &(ctx, m) in &[(32768usize, 8usize), (32768, 1), (131072, 8), (131072, 1), (262144, 8), (262144, 1)] {
        let pitch = 65536usize;
        let p0 = ctx - 16;
        let mut h = vec![0u32; m * pitch];
        for r in 0..m {
            for v in &mut h[r * pitch..r * pitch + (p0 + r + 1) / RATIO] {
                let s: f64 = (0..4).map(|_| (((rnd() >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.3).max(0.0)).sum();
                *v = ((s * 4.0 / 11.3137) as f32).to_bits();
            }
        }
        let d_sc = dev.htod_sync_copy(&h)?;
        let d_pos = dev.htod_sync_copy(&(0..m).map(|r| (p0 + r) as i32).collect::<Vec<i32>>())?;
        let d_s = dev.htod_sync_copy(&vec![-1i32; m * SEL_MAX])?;
        let d_p = dev.htod_sync_copy(&vec![0i32; m])?;
        let (scp, posp, sp, psp) = (*d_sc.device_ptr() as u64, *d_pos.device_ptr() as u64, *d_s.device_ptr() as u64, *d_p.device_ptr() as u64);
        let mut t = [0f64; 2];
        for (k, f) in [&f2, &f3].into_iter().enumerate() {
            for _ in 0..3 { launch(f, sp, psp, scp, posp, m, pitch)?; }
            dev.synchronize()?;
            let t0 = std::time::Instant::now();
            for _ in 0..50 { launch(f, sp, psp, scp, posp, m, pitch)?; }
            dev.synchronize()?;
            t[k] = t0.elapsed().as_secs_f64() * 1e6 / 50.0;
        }
        println!("    EXL3-QSA-SELECT timing ctx={ctx} m={m}: asc2 {:.1} us, asc3 {:.1} us per call ({:.2}x; synthetic real-like \
                  scores, L2-warm, launch-inclusive — ranks the kernels, does not price in-round)", t[0], t[1], t[0] / t[1].max(1e-9));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// WP18 rung A: the PREFILL selection bit-contract (EXL3-WP18). Synthetic Qwen3.8-Flash-Next
// indexer (hd 128, 4 heads, ratio 4, top-512 blocks, sel_max 2,051); chunks of the served width
// C = 2048 ending at ~8.5K / 32K / 128K (1 / 1 / 4 slab tiles), a QSA-straddle chunk (sparse rows
// start mid-chunk, t0 > 0) and a short odd chunk (77 rows, partial CTA tiles). For every sparse
// row, the WP18 walk — xq_wp18_score + xq_wp18_merge over exl3_forward::wp18_plan, the served
// loop, at slab row groups {32, 256, 2048} — must write the SAME list entries [0, pos_sel] and
// pos_sel as the old pair (qsa_score_prefill_b + xq_qsa_topk_asc), touch nothing else (poisoned
// lists), and every scored slab tile must equal the old matrix bitwise (xq_wp18_cmp; the compared
// count must be every visible (row, block) pair). Key modes: random; ties (a third of the blocks
// copy the block one tile earlier, a sixth a nearby earlier block — exact score ties across and
// within tiles, resolved by the lower index); zeros (97% zero keys: rows with < 512 positive
// blocks fill up with the LOWEST-index zero-score blocks); rising (key scale grows with the block
// index: every tile floods the carried list — the merge's worst case). Then a harness-only timing.
// ---------------------------------------------------------------------------
pub(super) fn probe_wp18_select(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    use crate::exl3_forward::{wp18_plan, WP18_MT, WP18_T};
    const HD: usize = 128;
    const NHX: usize = 4;
    const QK: usize = (NHX + 1) * HD;
    const RATIO: usize = 4;
    const BUDGET: usize = 2048;
    const TOPK: usize = BUDGET / RATIO;
    const SEL_MAX: usize = BUDGET + RATIO - 1;
    const PF_QT: usize = 8;                      // gpu_batch.cu QSA_PF_QT
    if dev.get_func("wp18_gb", "qsa_score_prefill_b").is_none() {
        let ptx = Ptx::from_src(crate::ptx_embed::GPU_BATCH);
        dev.load_ptx(ptx, "wp18_gb", &["qsa_score_prefill_b"]).context("load gpu_batch (qsa_score_prefill_b)")?;
    }
    let f_old = dev.get_func("wp18_gb", "qsa_score_prefill_b").context("qsa_score_prefill_b")?;
    let fget = |n: &str| dev.get_func(MODULE, n).with_context(|| format!("{n} (stale PTX?)"));
    let (f_topk, f_sc, f_mg, f_cmp) = (fget("xq_qsa_topk_asc")?, fget("xq_wp18_score")?, fget("xq_wp18_merge")?,
                                       fget("xq_wp18_cmp")?);
    // QsaParams (64 B): 4 pointers (unused by these kernels) | eps | hd heads rdim ratio topk sel_max pad
    let mut pb: Vec<u8> = vec![0u8; 32];
    pb.extend_from_slice(&1e-6f32.to_le_bytes());
    for v in [HD, NHX, 64, RATIO, TOPK, SEL_MAX, 0] { pb.extend_from_slice(&(v as i32).to_le_bytes()); }
    let d_params = dev.htod_sync_copy(&pb)?;
    let pp = *d_params.device_ptr() as u64;
    let mut rs = 0x5EED_0018_2026_0927u64;
    let mut rnd = move || -> u64 { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; rs };
    let bf = |x: f32| half::bf16::from_f32(x).to_bits();
    let lim = SEL_MAX;                           // qsa_limit(): rows at pos > lim are sparse
    let cases: [(&str, usize, usize); 5] = [("8.5K", 8704 - 2048, 2048), ("32K", 32768 - 2048, 2048),
                                            ("128K", 131072 - 2048, 2048), ("straddle", 1024, 2048),
                                            ("odd", 40000, 77)];
    let modes = ["random", "ties", "zeros", "rising"];
    let rgs = [32usize, 256, 2048];
    let (mut cells, mut rows_tot, mut pairs_tot, mut ties_tot, mut ztie_tot) = (0usize, 0usize, 0usize, 0usize, 0usize);
    // timing inputs kept from the random mode at 32K / 128K
    struct Keep { name: &'static str, pos0: usize, c: usize, nbs: usize, q: CudaSlice<u16>, plane: CudaSlice<u16>, pos: CudaSlice<i32> }
    let mut keep: Vec<Keep> = Vec::new();
    for &(cname, pos0, c) in &cases {
        let t0 = if pos0 > lim { 0 } else { (lim + 1 - pos0).min(c) };
        let rows = c - t0;
        anyhow::ensure!(rows > 0, "case {cname}: no sparse rows");
        let nblk = (pos0 + c) / RATIO;
        let nbs = nblk + 8;                      // matrix pitch (served: max_pos / ratio; any pitch >= nblk)
        let ntiles = nblk.div_ceil(WP18_T);
        for &mode in &modes {
            // ---- synthetic q rows (4 heads + the unused k head) and pooled keys
            let q: Vec<u16> = (0..c * QK).map(|_| bf((rnd() % 30001) as f32 / 10000.0 - 1.5)).collect();
            let mut keys: Vec<u16> = (0..nbs * HD).map(|_| bf((rnd() % 20001) as f32 / 10000.0 - 1.0)).collect();
            match mode {
                "ties" => for j in 1..nblk {
                    let r = rnd() % 6;
                    let src = if r < 2 && j >= WP18_T { Some(j - WP18_T) }
                              else if r == 2 { Some(j - 1 - (rnd() as usize) % j.min(700)) } else { None };
                    if let Some(s) = src { let row: Vec<u16> = keys[s * HD..(s + 1) * HD].to_vec(); keys[j * HD..(j + 1) * HD].copy_from_slice(&row); }
                },
                "zeros" => for j in 0..nblk { if rnd() % 100 < 97 { keys[j * HD..(j + 1) * HD].fill(0); } },
                "rising" => for j in 0..nblk {
                    let s = 0.25 + 3.0 * j as f32 / nblk as f32;
                    for v in &mut keys[j * HD..(j + 1) * HD] { *v = bf(half::bf16::from_bits(*v).to_f32() * s); }
                },
                _ => {}
            }
            let pos: Vec<i32> = (0..c).map(|i| (pos0 + i) as i32).collect();
            let (d_q, d_plane, d_pos) = (dev.htod_sync_copy(&q)?, dev.htod_sync_copy(&keys)?, dev.htod_sync_copy(&pos)?);
            let (qp, kp, posp) = (*d_q.device_ptr() as u64, *d_plane.device_ptr() as u64, *d_pos.device_ptr() as u64);
            // ---- reference: the old pair (all c rows scored, as served)
            let d_scores = dev.alloc_zeros::<f32>(c * nbs)?;
            let scp = *d_scores.device_ptr() as u64;
            let d_rsel = dev.htod_sync_copy(&vec![-9i32; c * SEL_MAX])?;
            let d_rps = dev.htod_sync_copy(&vec![-9i32; c])?;
            unsafe {
                f_old.clone().launch(LaunchConfig { grid_dim: (nblk.max(1).div_ceil(256) as u32, c.div_ceil(PF_QT) as u32, 1),
                                                    block_dim: (256, 1, 1), shared_mem_bytes: (PF_QT * NHX * HD * 4) as u32 },
                    (scp, qp, kp, pp, pos0 as i32, c as i32, nbs as i32, QK as i32))?;
                f_topk.clone().launch(LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                    (*d_rsel.device_ptr() as u64 + (t0 * SEL_MAX * 4) as u64, *d_rps.device_ptr() as u64 + (t0 * 4) as u64,
                     scp + (t0 * nbs * 4) as u64, posp + (t0 * 4) as u64, pp, nbs as i32, 0i32))?;
            }
            dev.synchronize()?;
            let (rsel, rps) = (dev.dtoh_sync_copy(&d_rsel)?, dev.dtoh_sync_copy(&d_rps)?);
            let full = dev.dtoh_sync_copy(&d_scores)?;
            // reference plausibility + tie statistics (sampled rows)
            let mut vis = 0usize;
            let (mut ties, mut zties) = (0usize, 0usize);
            for t in t0..c {
                let p = pos0 + t;
                let nb = (p + 1) / RATIO;
                vis += nb;
                let want = nb.min(TOPK) * RATIO + (p + 1 - nb * RATIO);
                if rps[t] != want as i32 - 1 {
                    bail!("EXL3-WP18 FAIL {cname} {mode}: implausible reference pos_sel {} at row {t} (want {})", rps[t], want - 1);
                }
                let l = &rsel[t * SEL_MAX..t * SEL_MAX + want];
                if l.windows(2).any(|w| w[0] >= w[1]) || l[0] < 0 {
                    bail!("EXL3-WP18 FAIL {cname} {mode}: reference list of row {t} not strictly ascending");
                }
                if (t - t0) % 8 == 0 && nb > TOPK {
                    let mut v: Vec<u32> = full[t * nbs..t * nbs + nb].iter().map(|x| x.to_bits()).collect();
                    v.select_nth_unstable_by(TOPK, |a, b| b.cmp(a));   // v[TOPK] = the (K+1)-th largest
                    let k1 = v[TOPK];
                    let kth = *v[..TOPK].iter().min().unwrap();
                    if kth == k1 { ties += 1; }
                    if kth == 0 { zties += 1; }
                }
            }
            ties_tot += ties;
            ztie_tot += zties;
            if mode == "zeros" && (cname == "8.5K" || cname == "32K") && zties == 0 {
                bail!("EXL3-WP18 FAIL {cname} zeros: no sampled row has its top-{TOPK} boundary at score 0 (tie path unexercised)");
            }
            for &rg in &rgs {
                let gr = rg.min(c);
                let d_slab = dev.alloc_zeros::<f32>(gr * WP18_T)?;
                let d_ck = dev.alloc_zeros::<u32>(gr * TOPK)?;
                let d_ci = dev.alloc_zeros::<i32>(gr * TOPK)?;
                let d_sel = dev.htod_sync_copy(&vec![-7i32; c * SEL_MAX])?;
                let d_ps = dev.htod_sync_copy(&vec![-7i32; c])?;
                let d_cnt = dev.htod_sync_copy(&[0i32, i32::MAX, 0, 0])?;
                let (slp, ckp, cip, selp, psp, cntp) = (*d_slab.device_ptr() as u64, *d_ck.device_ptr() as u64,
                    *d_ci.device_ptr() as u64, *d_sel.device_ptr() as u64, *d_ps.device_ptr() as u64, *d_cnt.device_ptr() as u64);
                let plan = wp18_plan(pos0 + t0, rows, RATIO, gr);
                for st in &plan {
                    let row = t0 + st.g0;
                    unsafe {
                        if st.score_grid.0 > 0 {
                            f_sc.clone().launch(LaunchConfig { grid_dim: (st.score_grid.0, st.score_grid.1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                (slp, qp + (row * QK * 2) as u64, kp, pp, st.pos_first as i32, st.gn as i32, (st.kt * WP18_T) as i32, QK as i32))?;
                            f_cmp.clone().launch(LaunchConfig { grid_dim: (st.gn as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                (cntp, slp, scp, pp, st.pos_first as i32, row as i32, st.kt as i32, nbs as i32))?;
                        }
                        f_mg.clone().launch(LaunchConfig { grid_dim: (st.gn as u32, 1, 1), block_dim: (WP18_MT as u32, 1, 1), shared_mem_bytes: 0 },
                            (selp + (row * SEL_MAX * 4) as u64, psp + (row * 4) as u64, slp, ckp, cip, pp, st.pos_first as i32, st.kt as i32))?;
                    }
                }
                dev.synchronize()?;
                let (sel, ps, cnt) = (dev.dtoh_sync_copy(&d_sel)?, dev.dtoh_sync_copy(&d_ps)?, dev.dtoh_sync_copy(&d_cnt)?);
                if cnt[0] != 0 {
                    bail!("EXL3-WP18 FAIL {cname} {mode} rows/group {gr}: {} slab scores bit-differ from qsa_score_prefill_b \
                           (first at chunk row {} block {})", cnt[0], cnt[1] >> 16, cnt[1] & 0xFFFF);
                }
                if cnt[2] as usize != vis {
                    bail!("EXL3-WP18 FAIL {cname} {mode} rows/group {gr}: compared {} scores, want every visible pair {vis}", cnt[2]);
                }
                for t in 0..c {
                    let r = &sel[t * SEL_MAX..(t + 1) * SEL_MAX];
                    if t < t0 {
                        if ps[t] != -7 || r.iter().any(|&x| x != -7) {
                            bail!("EXL3-WP18 FAIL {cname} {mode} rows/group {gr}: dense row {t} was written");
                        }
                        continue;
                    }
                    let n = (rps[t] + 1) as usize;
                    if ps[t] != rps[t] || r[..n] != rsel[t * SEL_MAX..t * SEL_MAX + n] || r[n..].iter().any(|&x| x != -7) {
                        let i = (0..n).find(|&i| r[i] != rsel[t * SEL_MAX + i]);
                        bail!("EXL3-WP18 FAIL {cname} {mode} rows/group {gr}: row {t} (pos {}) pos_sel {} vs {}; first entry diff {:?}; \
                               writes past pos_sel {}", pos0 + t, ps[t], rps[t],
                              i.map(|i| (i, r[i], rsel[t * SEL_MAX + i])), r[n..].iter().filter(|&&x| x != -7).count());
                    }
                }
                cells += 1;
                rows_tot += rows;
                pairs_tot += vis;
            }
            println!("    wp18 {cname:<8} {mode:<6}: rows {rows} (t0 {t0}), {nblk} blocks = {ntiles} tile(s); lists + pos_sel == \
                      old at row groups {rgs:?}; {vis} scores bitwise; sampled boundary ties {ties}, zero-score boundaries {zties}");
            if mode == "random" && (cname == "32K" || cname == "128K") {
                keep.push(Keep { name: cname, pos0, c, nbs, q: d_q, plane: d_plane, pos: d_pos });
            }
        }
    }
    if ties_tot == 0 {
        bail!("EXL3-WP18 FAIL: no sampled row had a tie at its top-{TOPK} boundary (tie-break path unexercised)");
    }
    println!("EXL3-WP18: PASS ({cells} cells: {rows_tot} sparse rows, {pairs_tot} scores — WP18 bounded walk == \
              qsa_score_prefill_b + xq_qsa_topk_asc bitwise at ~8.5K/32K/128K, straddle and odd chunks; \
              {ties_tot} sampled boundary ties, {ztie_tot} zero-score boundaries)");

    // ---- timing (harness-only: one synthetic chunk of C = 2048 rows at the window end, one layer,
    // L2-warm, launch-inclusive; ranks the paths — the served number is the prefill tok/s A/B)
    for k in &keep {
        let (pos0, c, nbs) = (k.pos0, k.c, k.nbs);
        let t0 = if pos0 > lim { 0 } else { (lim + 1 - pos0).min(c) };
        let rows = c - t0;
        let nblk = (pos0 + c) / RATIO;
        let (qp, kp, posp) = (*k.q.device_ptr() as u64, *k.plane.device_ptr() as u64, *k.pos.device_ptr() as u64);
        let d_scores = dev.alloc_zeros::<f32>(c * nbs)?;
        let d_sel = dev.alloc_zeros::<i32>(c * SEL_MAX)?;
        let d_ps = dev.alloc_zeros::<i32>(c)?;
        let gr = crate::exl3_forward::wp18_rows().min(c);
        let d_slab = dev.alloc_zeros::<f32>(gr * WP18_T)?;
        let d_ck = dev.alloc_zeros::<u32>(gr * TOPK)?;
        let d_ci = dev.alloc_zeros::<i32>(gr * TOPK)?;
        let (scp, selp, psp) = (*d_scores.device_ptr() as u64, *d_sel.device_ptr() as u64, *d_ps.device_ptr() as u64);
        let (slp, ckp, cip) = (*d_slab.device_ptr() as u64, *d_ck.device_ptr() as u64, *d_ci.device_ptr() as u64);
        let plan = wp18_plan(pos0 + t0, rows, RATIO, gr);
        let old_score = || -> Result<()> { unsafe {
            f_old.clone().launch(LaunchConfig { grid_dim: (nblk.max(1).div_ceil(256) as u32, c.div_ceil(PF_QT) as u32, 1),
                                                block_dim: (256, 1, 1), shared_mem_bytes: (PF_QT * NHX * HD * 4) as u32 },
                (scp, qp, kp, pp, pos0 as i32, c as i32, nbs as i32, QK as i32))?; } Ok(()) };
        let old_topk = || -> Result<()> { unsafe {
            f_topk.clone().launch(LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                (selp + (t0 * SEL_MAX * 4) as u64, psp + (t0 * 4) as u64, scp + (t0 * nbs * 4) as u64, posp + (t0 * 4) as u64,
                 pp, nbs as i32, 0i32))?; } Ok(()) };
        let walk = || -> Result<()> {
            for st in &plan {
                let row = t0 + st.g0;
                unsafe {
                    if st.score_grid.0 > 0 {
                        f_sc.clone().launch(LaunchConfig { grid_dim: (st.score_grid.0, st.score_grid.1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (slp, qp + (row * QK * 2) as u64, kp, pp, st.pos_first as i32, st.gn as i32, (st.kt * WP18_T) as i32, QK as i32))?;
                    }
                    f_mg.clone().launch(LaunchConfig { grid_dim: (st.gn as u32, 1, 1), block_dim: (WP18_MT as u32, 1, 1), shared_mem_bytes: 0 },
                        (selp + (row * SEL_MAX * 4) as u64, psp + (row * 4) as u64, slp, ckp, cip, pp, st.pos_first as i32, st.kt as i32))?;
                }
            }
            Ok(())
        };
        let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            for _ in 0..2 { f()?; }
            dev.synchronize()?;
            let t = std::time::Instant::now();
            for _ in 0..5 { f()?; }
            dev.synchronize()?;
            Ok(t.elapsed().as_secs_f64() * 1e3 / 5.0)
        };
        let (ts, tk, tw) = (time(&old_score)?, time(&old_topk)?, time(&walk)?);
        println!("    EXL3-WP18 timing {} (pos0 {pos0}, {rows} rows, {nblk} blocks): old {:.2} ms (qsa_score_prefill_b {ts:.2} + \
                  xq_qsa_topk_asc {tk:.2}) vs WP18 walk {tw:.2} ms at {gr} rows/group ({:.2}x) — harness-only, one layer-chunk, \
                  L2-warm; ranks the paths, the served number is the prefill A/B vs --wp18-off=1",
                 k.name, ts + tk, (ts + tk) / tw.max(1e-9));
    }
    Ok(())
}

