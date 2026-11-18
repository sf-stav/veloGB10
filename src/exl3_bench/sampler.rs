//! Sampler, flash-prefill and penalty probes with their host references.
//! Probe-only code of the `exl3_bench` module (see `mod.rs` for the map); shared helpers come from the parent.
//! Numerics contract (module header in `mod.rs`): the kernel decode is bitwise vs the host oracle; do not change reduction orders.

use super::*;

// ---------------------------------------------------------------------------
// API-parity G2 gate: `--probe-sampler`. Distribution-exactness of xq_sample_rows vs the
// host-computed target distribution (same semantics: keys -> top-k with ties -> min_p floor ->
// top-p with boundary ties -> softmax/T), on a fixed synthetic fp16 row at the real vocab.
// Chi-square over (tokens with expected >= 20) + 64 index-range buckets for the rest; hard
// fail on any sample outside the support; a second seed as the control; determinism check.
fn skey(b: u16) -> u32 { if b & 0x8000 != 0 { (!(b as u32)) & 0xFFFF } else { (b as u32) | 0x8000 } }

fn sampler_target(lg: &[u16], t: f32, top_k: u32, top_p: f32, min_p: f32) -> Vec<f64> {
    let v = lg.len();
    let keys: Vec<u32> = lg.iter().map(|&b| skey(b)).collect();
    let val = |i: usize| f16::from_bits(lg[i]).to_f32();
    let kmax = *keys.iter().max().unwrap();
    let lmax = (0..v).find(|&i| keys[i] == kmax).map(val).unwrap();
    let tau = if top_k > 0 && (top_k as usize) < v {
        let mut ks = keys.clone();
        ks.sort_unstable_by(|a, b| b.cmp(a));
        ks[top_k as usize - 1]
    } else { 0 };
    let t = t.max(1e-6);
    let mfloor = if min_p > 0.0 { lmax + t * min_p.ln() } else { f32::NEG_INFINITY };
    let cand = |i: usize| keys[i] >= tau && val(i) >= mfloor;
    let w = |i: usize| (((val(i) - lmax) / t) as f64).exp();
    let s: f64 = (0..v).filter(|&i| cand(i)).map(w).sum();
    let mut thr = tau;
    if top_p < 1.0 {
        // mass by key, descending; theta = largest key with cum mass >= top_p * S
        let mut by: std::collections::BTreeMap<u32, f64> = Default::default();
        for i in 0..v { if cand(i) { *by.entry(keys[i]).or_insert(0.0) += w(i); } }
        let target = top_p as f64 * s;
        let mut cum = 0.0;
        for (&k, &m) in by.iter().rev() {
            cum += m;
            if cum >= target { thr = thr.max(k); break; }
        }
    }
    let mut p: Vec<f64> = (0..v).map(|i| if cand(i) && keys[i] >= thr { w(i) } else { 0.0 }).collect();
    let z: f64 = p.iter().sum();
    for x in p.iter_mut() { *x /= z; }
    p
}

// ---------------------------------------------------------------------------
// w5/PFIX: the chunked-prefill attention kernels against the SWEEP xq_attn_prefill_q (one block
// per head, token loop over the decode softmax — a different kernel class, causally correct) on
// synthetic data: no model, f32 KV, 4 q heads / 2 kv heads, partial rope 64, chunk starts that put
// block rows on, across and deep past tile boundaries. Both head dims:
//   hd=128: xq_attn_prefill_flash (the scalar BT=16 shape; no host caller since 2463afc)
//           xq_attn_prefill_flash128c (--pq8-flash=causal's hd=128 twin)
//   hd=256: xq_attn_prefill_flash256 / flash256v (the default; must equal flash256 BITWISE)
//           xq_attn_prefill_flash256c (--pq8-flash=causal)
// PASS = every CAUSAL twin within f16 output rounding of the sweep (max |d| <= 4e-3 on |out| < 1)
// with every row written (outputs NaN-poisoned before each launch), and flash256v == flash256 bit
// for bit. The scalar kernels' diagonal-tile staging defect is REPORTED (max |d| vs the sweep):
// the host model (exl3_forward.rs flash256_stage_tests) predicts it far above the causal twins'.
//   ./gb10_inference --probe-exl3-flashpf
// ---------------------------------------------------------------------------
pub fn probe_flash_prefill() -> Result<()> {
    const NH: usize = 4;
    const NKV: usize = 2;
    const RDIM: usize = 64;
    const MAXP: usize = 512;
    const TOL: f32 = 4e-3;
    let dev = dev0()?;
    load_module(&dev)?;
    let mut st: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rnd = move |amp: f32| -> f32 {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        (((st >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0) * amp
    };
    let h16 = |x: f32| f16::from_f32(x).to_bits();
    let mut fail = 0usize;
    let mut worst_causal = 0f32;
    let mut worst_scalar = 0f32;
    // (hd, rows per block, scalar kernel, causal twin)
    for &(hd, bt, scalar, causal) in &[(128usize, 16usize, "xq_attn_prefill_flash", "xq_attn_prefill_flash128c"),
                                       (256, 8, "xq_attn_prefill_flash256", "xq_attn_prefill_flash256c")] {
        // slot 0 of the cache: [k|v][kvh][pos][hd] f32 (fmt bits of max_pos = 0 = f32)
        let kv: Vec<f32> = (0..2 * NKV * MAXP * hd).map(|_| rnd(1.0)).collect();
        let d_kv = dev.htod_sync_copy(&kv)?;
        let qnkn: Vec<u16> = (0..2 * hd).map(|_| h16(rnd(0.2))).collect();
        let d_qnkn = dev.htod_sync_copy(&qnkn)?;
        for &(pos0, c) in &[(0usize, 48usize), (8, 40), (13, 37), (16, 33), (21, 64), (40, 100), (256, 33), (400, 64)] {
            assert!(pos0 + c.div_ceil(bt) * bt + 32 <= MAXP);
            let qg: Vec<u16> = (0..c * NH * 2 * hd).map(|_| h16(rnd(1.0))).collect();
            let cs: Vec<f32> = (0..c).flat_map(|_| {
                let ang: Vec<f32> = (0..RDIM).map(|_| rnd(3.1)).collect();
                ang.iter().map(|a| a.cos()).chain(ang.iter().map(|a| a.sin())).collect::<Vec<f32>>()
            }).collect();
            let d_qg = dev.htod_sync_copy(&qg)?;
            let d_cs = dev.htod_sync_copy(&cs)?;
            let n = c * NH * hd;
            let nh_nkv = (NKV | (NH << 16)) as i32;
            let hd_rdim = (hd | (RDIM << 16)) as i32;
            let slot_pos0 = (pos0 << 12) as i32;
            let run = |name: &str, grid: (u32, u32, u32), block: u32| -> Result<Vec<f32>> {
                let f = dev.get_func(MODULE, name).with_context(|| format!("{name} missing"))?;
                let mut out = dev.htod_sync_copy(&vec![0xFFFFu16; n])?;   // f16 NaN poison
                unsafe {
                    f.launch(LaunchConfig { grid_dim: grid, block_dim: (block, 1, 1), shared_mem_bytes: 0 },
                             (&mut out, &d_qg, &d_kv, &d_qnkn, &d_cs, nh_nkv, hd_rdim, MAXP as i32,
                              slot_pos0, c as i32, 1e-6f32))?;
                }
                dev.synchronize()?;
                Ok(dev.dtoh_sync_copy(&out)?.iter().map(|&b| f16::from_bits(b).to_f32()).collect())
            };
            let fgrid = ((c.div_ceil(bt)) as u32, NH as u32, 1);
            let reference = run("xq_attn_prefill_q", (NH as u32, 1, 1), hd as u32)?;
            let sc = run(scalar, fgrid, (bt * 32) as u32)?;
            let ca = run(causal, fgrid, (bt * 32) as u32)?;
            let unwritten = |x: &[f32]| x.iter().filter(|v| v.is_nan()).count();
            let maxd = |x: &[f32]| x.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0f32, |m, d| if d.is_nan() { f32::INFINITY } else { m.max(d) });
            let (nr, nc) = (unwritten(&reference), unwritten(&ca));
            if nr > 0 || reference.iter().all(|&v| v == 0.0) {
                bail!("EXL3-FLASHPF FAIL: implausible sweep reference (hd {hd} pos0 {pos0} c {c}: {nr} NaN)");
            }
            let (dsc, dca) = (maxd(&sc), maxd(&ca));
            worst_causal = worst_causal.max(dca);
            worst_scalar = worst_scalar.max(dsc);
            let ok = nc == 0 && dca <= TOL;
            let mut line = format!("EXL3-FLASHPF: hd {hd} pos0 {pos0:>3} c {c:>3}: vs sweep max|d| {} {dca:.2e}{} | {} {dsc:.2e} (scalar staging)",
                                   causal.trim_start_matches("xq_attn_prefill_"),
                                   if nc > 0 { format!(" ({nc} rows' halves unwritten)") } else { String::new() },
                                   scalar.trim_start_matches("xq_attn_prefill_"));
            let mut vok = true;
            if hd == 256 {
                let vv = run("xq_attn_prefill_flash256v", fgrid, 256)?;
                let nd = vv.iter().zip(&sc).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                vok = nd == 0;
                line += &format!(" | flash256v vs flash256 differ {nd} of {n}");
            }
            println!("{line}{}", if ok && vok { "" } else { "  <-- FAIL" });
            if !(ok && vok) { fail += 1; }
        }
    }
    println!("EXL3-FLASHPF: worst causal-twin max|d| vs sweep {worst_causal:.2e} (tol {TOL:.0e}); worst scalar-staging \
              max|d| {worst_scalar:.2e}{}", if worst_scalar > 10.0 * worst_causal.max(1e-4) { " (defect visible)" }
              else { " (defect NOT visible — the host staging model disagrees with the GPU)" });
    if fail > 0 { bail!("EXL3-FLASHPF FAIL: {fail} case(s)"); }
    println!("EXL3-FLASHPF: PASS (flash128c + flash256c match the sweep within f16 rounding; flash256v == flash256 bitwise)");
    Ok(())
}

pub fn probe_sampler() -> Result<()> {
    let dev = dev0()?;
    let ptx = Ptx::from_src(crate::ptx_embed::EXL3_BENCH);
    dev.load_ptx(ptx, "exl3_sampler_probe", &["xq_argmax", "xq_sample_rows"])?;
    let f = dev.get_func("exl3_sampler_probe", "xq_sample_rows").context("xq_sample_rows missing")?;
    let v: usize = 248_320;
    // synthetic row: a Gaussian bulk (sd 2) + a 40-token head spread over [6, 11] so top-k/top-p
    // cut inside a populated region; fp16 rounding makes real ties.
    let mut st: u64 = 0x9E3779B97F4A7C15;
    let mut rnd = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; (st >> 11) as f64 / (1u64 << 53) as f64 };
    let mut row: Vec<u16> = (0..v).map(|_| {
        let (u1, u2) = (rnd().max(1e-12), rnd());
        let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        f16::from_f64(g * 2.0).to_bits()
    }).collect();
    for j in 0..40usize {
        let idx = (j * 6151 + 17) % v;
        row[idx] = f16::from_f64(6.0 + 5.0 * rnd()).to_bits();
    }
    let rows_per_launch = 512usize;
    let launches = 400usize;
    let mut lg_dev = dev.htod_sync_copy(&row.iter().cycle().take(v * rows_per_launch).cloned().collect::<Vec<u16>>())?;
    let _ = &mut lg_dev;
    let mut ids_dev = dev.alloc_zeros::<i32>(rows_per_launch)?;
    // (T, top_k, top_p, min_p)
    let cfgs: [(f32, u32, f32, f32); 5] = [
        (0.7, 20, 0.8, 0.0),   // our server default
        (0.8, 0, 1.0, 0.08),   // the reference implementation shim default (min_p)
        (1.0, 0, 1.0, 0.0),    // pure temperature, full vocab
        (0.6, 0, 0.95, 0.0),   // top-p over the full vocab
        (1.3, 50, 1.0, 0.0),   // top-k only, hot
    ];
    let mut all_ok = true;
    for &(t, k, p, mp) in cfgs.iter() {
        let target = sampler_target(&row, t, k, p, mp);
        let support = target.iter().filter(|&&x| x > 0.0).count();
        for seed in [0x1234_5678_9ABCu64, 0x0DDB_A11u64] {
            let mut counts = vec![0u64; v];
            let mut first: Vec<i32> = Vec::new();
            for l in 0..launches {
                let mut samp = vec![0u32; rows_per_launch * 8];
                for r in 0..rows_per_launch {
                    let c = (l * rows_per_launch + r) as u32;
                    samp[r * 8..r * 8 + 8].copy_from_slice(&[1, t.to_bits(), p.to_bits(), k,
                        seed as u32, (seed >> 32) as u32, c, mp.to_bits()]);
                }
                let samp_dev = dev.htod_sync_copy(&samp)?;
                unsafe {
                    f.clone().launch(LaunchConfig { grid_dim: (rows_per_launch as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                        (&mut ids_dev, &lg_dev, v as i32, &samp_dev))?;
                }
                let ids = dev.dtoh_sync_copy(&ids_dev)?;
                if l == 0 { first = ids.clone(); }
                for &id in &ids { counts[id as usize] += 1; }
            }
            // determinism: relaunch l=0 and compare
            {
                let mut samp = vec![0u32; rows_per_launch * 8];
                for r in 0..rows_per_launch {
                    samp[r * 8..r * 8 + 8].copy_from_slice(&[1, t.to_bits(), p.to_bits(), k,
                        seed as u32, (seed >> 32) as u32, r as u32, mp.to_bits()]);
                }
                let samp_dev = dev.htod_sync_copy(&samp)?;
                unsafe {
                    f.clone().launch(LaunchConfig { grid_dim: (rows_per_launch as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                        (&mut ids_dev, &lg_dev, v as i32, &samp_dev))?;
                }
                let again = dev.dtoh_sync_copy(&ids_dev)?;
                if again != first { println!("  DETERMINISM FAIL"); all_ok = false; }
            }
            let n: u64 = counts.iter().sum();
            let outside: u64 = (0..v).filter(|&i| target[i] == 0.0).map(|i| counts[i]).sum();
            // chi-square bins
            let (mut chi2, mut df) = (0.0f64, 0i64);
            let mut bucket_e = vec![0.0f64; 64];
            let mut bucket_o = vec![0u64; 64];
            for i in 0..v {
                if target[i] == 0.0 { continue; }
                let e = target[i] * n as f64;
                if e >= 20.0 {
                    let d = counts[i] as f64 - e;
                    chi2 += d * d / e; df += 1;
                } else {
                    let b = i * 64 / v;
                    bucket_e[b] += e; bucket_o[b] += counts[i];
                }
            }
            for b in 0..64 {
                if bucket_e[b] >= 20.0 {
                    let d = bucket_o[b] as f64 - bucket_e[b];
                    chi2 += d * d / bucket_e[b]; df += 1;
                }
            }
            df -= 1;
            // Wilson-Hilferty: z = ((chi2/df)^(1/3) - (1 - 2/(9df))) / sqrt(2/(9df))
            let dff = df.max(1) as f64;
            let z = ((chi2 / dff).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dff))) / (2.0 / (9.0 * dff)).sqrt();
            let tvd: f64 = 0.5 * (0..v).map(|i| (counts[i] as f64 / n as f64 - target[i]).abs()).sum::<f64>();
            let ok = outside == 0 && z < 3.5;
            all_ok &= ok;
            println!("  T={t} top_k={k} top_p={p} min_p={mp} seed={seed:#x}: n={n} support={support} chi2={chi2:.1} df={df} z={z:.2} tvd={tvd:.4} outside={outside} -> {}",
                     if ok { "PASS" } else { "FAIL" });
        }
    }
    println!("{}", if all_ok { "PROBE_SAMPLER_OK" } else { "PROBE_SAMPLER_FAIL" });
    if !all_ok { bail!("sampler probe failed"); }
    Ok(())
}

// ---------------------------------------------------------------------------
// WP15 gate: `--probe-penalties` (no model, no weights: synthetic fp16 rows at the real vocab).
// xq_pen_rows / xq_pen_draft vs a DENSE host transcription of the reference implementation's two kernels
// (exllamav3_ext/generator/rep_pen.cu: every logit rewritten, float factors, float frequency
// sum in ascending position order). Asserts:
//   (a) every logit the window did not touch is bit-identical (the sparse rewrite is exact);
//   (b) touched logits equal the reference within 1 fp16 ulp (only the frequency sum's float
//       association and FMA contraction differ) — the exact / 1-ulp split is reported;
//   (c) SPECULATION EXACTNESS: verify row r (history + b, d0..d_{r-1} in flight) is bit-identical
//       to a plain single-row launch whose committed history already holds those tokens;
//   (d) the ring commit: every penalized row wrote its input token at its position; flag-0 rows
//       touch neither their logits nor the ring;
//   (e) the draft mirror (pruned 65,536-id slice, in-chain drafts as past) against the same
//       reference.
fn pen_factors(past: &[u32], vlim: usize, s: usize, d: usize, freq: f32)
               -> (std::collections::HashMap<u32, f32>, std::collections::HashMap<u32, f32>,
                   std::collections::HashMap<u32, f32>) {
    // (rep factor max, presence factor max, frequency sum) per token — rep_pen.cu verbatim
    let (mut rf, mut pf, mut fs) = (std::collections::HashMap::new(), std::collections::HashMap::new(),
                                    std::collections::HashMap::new());
    let pl = past.len() as i64;
    for (i, &t) in past.iter().enumerate() {
        if (i as i64) <= pl - s as i64 - d as i64 { continue; }
        if t as usize >= vlim { continue; }
        let dist = (pl - i as i64) as f32;
        let (sf, df) = (s as f32, d as f32);
        let r = if dist <= sf { 1.0f32 } else {
            (if d > 0 { 1.0 - (dist - sf) / df } else { 1.0 }).clamp(0.0, 1.0)
        };
        let e = rf.entry(t).or_insert(0.0f32); if r > *e { *e = r; }
        let p = (if d > 0 { 1.0 - (dist - sf) / df } else { 1.0f32 }).clamp(0.0, 1.0);
        *fs.entry(t).or_insert(0.0f32) += p * freq;
        let e = pf.entry(t).or_insert(0.0f32); if p > *e { *e = p; }
    }
    (rf, pf, fs)
}

fn pen_ref_row(row: &mut [u16], past: &[u32], vlim: usize, rep: f32, pres: f32, freq: f32, s: usize, d: usize) {
    let (rf, pf, fs) = pen_factors(past, vlim, s, d, freq);
    for j in 0..vlim {
        let mut v = f16::from_bits(row[j]).to_f32();
        if rep != 1.0 {
            let f = rf.get(&(j as u32)).copied().unwrap_or(0.0);
            let w = if v > 0.0 { v / rep } else { v * rep };
            let fr = f + 1e-30f32;
            let f1 = (1.0f32 - fr) + 1e-30f32;
            v = v * f1 + w * fr;
        }
        if pres != 0.0 || freq != 0.0 {
            v -= fs.get(&(j as u32)).copied().unwrap_or(0.0);
            v -= pf.get(&(j as u32)).copied().unwrap_or(0.0) * pres;
        }
        if v > 65504.0 { v = 65504.0 } else if v < -65504.0 { v = -65504.0 }
        row[j] = f16::from_f32(v).to_bits();
    }
}

fn f16_ulps(a: u16, b: u16) -> u32 {
    let o = |x: u16| -> i32 { if x & 0x8000 != 0 { -((x & 0x7FFF) as i32) } else { x as i32 } };
    (o(a) - o(b)).unsigned_abs()
}

pub fn probe_penalties() -> Result<()> {
    use crate::exl3_forward::{PEN_RING, PEN_RING_LOG2};
    let dev = dev0()?;
    let ptx = Ptx::from_src(crate::ptx_embed::EXL3_BENCH);
    dev.load_ptx(ptx, "exl3_pen_probe", &["xq_pen_rows", "xq_pen_draft"])?;
    let f_rows = dev.get_func("exl3_pen_probe", "xq_pen_rows").context("xq_pen_rows missing")?;
    let f_draft = dev.get_func("exl3_pen_probe", "xq_pen_draft").context("xq_pen_draft missing")?;
    let v: usize = 248_320;
    let vd: usize = 65_536;
    let nslot = 8usize;
    let mut st: u64 = 0xD1B5_4A32_D192_ED03;
    let mut rnd = move || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
    // a skewed token stream (heavy repeats, both spans' edges, ids >= the draft slice)
    let tok = |r: &mut dyn FnMut() -> u64| -> u32 {
        let x = r();
        match x % 10 {
            0..=4 => (x >> 8) as u32 % 300,                         // hot, repeats a lot
            5 | 6 => 4090 + ((x >> 8) as u32 % 12),                 // straddles a 4096 span edge
            7 => (v as u32 - 1) - ((x >> 8) as u32 % 5),            // the last span's tail
            8 => 65_530 + ((x >> 8) as u32 % 12),                   // the draft slice edge
            _ => (x >> 8) as u32 % v as u32,
        }
    };
    let gauss_row = |r: &mut dyn FnMut() -> u64| -> Vec<u16> {
        (0..v).map(|_| {
            let u1 = ((r() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
            let u2 = (r() >> 11) as f64 / (1u64 << 53) as f64;
            f16::from_f64(3.0 * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()).to_bits()
        }).collect()
    };
    // (rep, pres, freq, sustain, decay)
    let cfgs: [(f32, f32, f32, usize, usize); 6] = [
        (1.1, 0.0, 0.0, 1024, 1024),
        (1.0, 0.5, 0.3, 1024, 1024),
        (1.05, 1.5, 0.0, 1024, 1024),
        (0.9, 0.0, 0.2, 1024, 1024),  // rep < 1: accepted (owner decision), rewards repeats
        (1.3, -0.5, 0.25, 64, 0),     // no decay tier
        (1.2, 0.4, 0.4, 16, 8),       // short windows: the decay factors are fractional
    ];
    let pack = |c: &(f32, f32, f32, usize, usize)| -> [u32; 8] {
        let flag = (c.0 != 1.0) as u32 | (((c.1 != 0.0 || c.2 != 0.0) as u32) << 1);
        [flag, c.0.to_bits(), c.1.to_bits(), c.2.to_bits(), c.3 as u32, c.4 as u32, 0, 0]
    };
    let launch_rows = |logits: &mut CudaSlice<u16>, pen: &CudaSlice<u32>, toks: &CudaSlice<i32>,
                       slots: &CudaSlice<i32>, hist: &mut CudaSlice<i32>, m: usize| -> Result<()> {
        unsafe {
            f_rows.clone().launch(LaunchConfig { grid_dim: (v.div_ceil(4096) as u32, m as u32, 1),
                                                 block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                (logits, v as i32, pen, toks, slots, hist, PEN_RING_LOG2 as i32))?;
        }
        Ok(())
    };
    let mut all_ok = true;
    let (mut n_exact, mut n_ulp1) = (0usize, 0usize);
    for (ci, c) in cfgs.iter().enumerate() {
        // ---- (a)(b)(c)(d) verify layout: slot 3, positions p..p+5 (the ring has wrapped)
        let m = 6usize;
        let slot = 3usize;
        let p = 9_000 + ci * 37;
        let seq: Vec<u32> = (0..p + m).map(|_| tok(&mut rnd)).collect(); // committed [0,p) + in-flight
        let mut ring = vec![0i32; nslot * PEN_RING];
        for (q, &t) in seq[..p].iter().enumerate() { ring[slot * PEN_RING + (q & (PEN_RING - 1))] = t as i32; }
        let rows: Vec<u16> = (0..m).flat_map(|_| gauss_row(&mut rnd)).collect();
        let mut pen = vec![0u32; m * 8];
        for r in 0..m { pen[r * 8..r * 8 + 8].copy_from_slice(&pack(c)); }
        let toks: Vec<i32> = seq[p..p + m].iter().map(|&t| t as i32).collect();
        let slots: Vec<i32> = (0..m).flat_map(|r| [slot as i32, (p + r) as i32]).collect();
        let mut lg = dev.htod_sync_copy(&rows)?;
        let pen_d = dev.htod_sync_copy(&pen)?;
        let toks_d = dev.htod_sync_copy(&toks)?;
        let slots_d = dev.htod_sync_copy(&slots)?;
        let mut hist_d = dev.htod_sync_copy(&ring)?;
        launch_rows(&mut lg, &pen_d, &toks_d, &slots_d, &mut hist_d, m)?;
        let out = dev.dtoh_sync_copy(&lg)?;
        let hist_out = dev.dtoh_sync_copy(&hist_d)?;
        let (mut untouched_bad, mut big) = (0usize, 0usize);
        for r in 0..m {
            let mut refrow = rows[r * v..(r + 1) * v].to_vec();
            pen_ref_row(&mut refrow, &seq[..=p + r], v, c.0, c.1, c.2, c.3, c.4);
            let (rf, _, _) = pen_factors(&seq[..=p + r], v, c.3, c.4, c.2);
            for j in 0..v {
                let (o, e, i0) = (out[r * v + j], refrow[j], rows[r * v + j]);
                if !rf.contains_key(&(j as u32)) {
                    if o != i0 { untouched_bad += 1; }
                } else {
                    match f16_ulps(o, e) { 0 => n_exact += 1, 1 => n_ulp1 += 1, _ => big += 1 }
                }
            }
        }
        // (d) commits: row r wrote toks[r] at p + r; nothing else in the ring moved
        let mut ring_want = ring.clone();
        for r in 0..m { ring_want[slot * PEN_RING + ((p + r) & (PEN_RING - 1))] = toks[r]; }
        let ring_ok = hist_out == ring_want;
        // (c) speculation exactness: row r == a plain single-row launch with b, d0..d_{r-1} committed
        let mut spec_bad = 0usize;
        for r in 0..m {
            let mut ring2 = ring.clone();
            for q in p..p + r { ring2[slot * PEN_RING + (q & (PEN_RING - 1))] = seq[q] as i32; }
            let mut lg1 = dev.htod_sync_copy(&rows[r * v..(r + 1) * v].to_vec())?;
            let pen1 = dev.htod_sync_copy(&pack(c).to_vec())?;
            let toks1 = dev.htod_sync_copy(&vec![toks[r]])?;
            let slots1 = dev.htod_sync_copy(&vec![slot as i32, (p + r) as i32])?;
            let mut hist1 = dev.htod_sync_copy(&ring2)?;
            launch_rows(&mut lg1, &pen1, &toks1, &slots1, &mut hist1, 1)?;
            let o1 = dev.dtoh_sync_copy(&lg1)?;
            spec_bad += o1.iter().zip(&out[r * v..(r + 1) * v]).filter(|(a, b)| a != b).count();
        }
        // flag-0 rows (a plain batch with an unpenalized lane next to a penalized one)
        let mut pen_mix = vec![0u32; 2 * 8];
        pen_mix[8..16].copy_from_slice(&pack(c));
        let rows2: Vec<u16> = (0..2).flat_map(|_| gauss_row(&mut rnd)).collect();
        let mut lg2 = dev.htod_sync_copy(&rows2)?;
        let pen2 = dev.htod_sync_copy(&pen_mix)?;
        let toks2 = dev.htod_sync_copy(&vec![seq[0] as i32, seq[1] as i32])?;
        // lane A = slot 1 at pos 5 (lo clamps to 0), lane B = slot 6 at pos 12
        let slots2 = dev.htod_sync_copy(&vec![1i32, 5, 6, 12])?;
        let mut ring3 = vec![0i32; nslot * PEN_RING];
        for q in 0..12 { ring3[6 * PEN_RING + q] = seq[q] as i32; }
        for q in 0..5 { ring3[PEN_RING + q] = seq[20 + q] as i32; }
        let mut hist2 = dev.htod_sync_copy(&ring3)?;
        launch_rows(&mut lg2, &pen2, &toks2, &slots2, &mut hist2, 2)?;
        let o2 = dev.dtoh_sync_copy(&lg2)?;
        let h2 = dev.dtoh_sync_copy(&hist2)?;
        let flag0_ok = o2[..v] == rows2[..v] && h2[PEN_RING + 5] == ring3[PEN_RING + 5];
        let mut past_b: Vec<u32> = seq[..12].to_vec();
        past_b.push(seq[1]);
        let mut ref_b = rows2[v..].to_vec();
        pen_ref_row(&mut ref_b, &past_b, v, c.0, c.1, c.2, c.3, c.4);
        let short_big = o2[v..].iter().zip(&ref_b).filter(|(a, b)| f16_ulps(**a, **b) > 1).count();
        // ---- (e) draft mirror: meta {b, p, slot}, passes 0, 2, 4 over the 65,536-id slice
        let mut draft_big = 0usize;
        for &i in &[0usize, 2, 4] {
            let row = gauss_row(&mut rnd)[..vd].to_vec();
            let mut lgd = dev.htod_sync_copy(&row)?;
            let pen_d0 = dev.htod_sync_copy(&pack(c).to_vec())?;
            let meta = dev.htod_sync_copy(&vec![seq[p] as i32, p as i32, slot as i32, 0])?;
            let drafts: Vec<i32> = seq[p + 1..p + 6].iter().map(|&t| t as i32).collect();
            let dd = dev.htod_sync_copy(&drafts)?;
            let hd = dev.htod_sync_copy(&ring)?;
            unsafe {
                f_draft.clone().launch(LaunchConfig { grid_dim: (vd.div_ceil(4096) as u32, 1, 1),
                                                      block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                    (&mut lgd, vd as i32, &pen_d0, &meta, &dd, i as i32, &hd, PEN_RING_LOG2 as i32))?;
            }
            let od = dev.dtoh_sync_copy(&lgd)?;
            let mut refd = row.clone();
            pen_ref_row(&mut refd, &seq[..=p + i], vd, c.0, c.1, c.2, c.3, c.4);
            draft_big += od.iter().zip(&refd).filter(|(a, b)| f16_ulps(**a, **b) > 1).count();
        }
        let ok = untouched_bad == 0 && big == 0 && ring_ok && spec_bad == 0 && flag0_ok
            && short_big == 0 && draft_big == 0;
        all_ok &= ok;
        println!("  rep={} pres={} freq={} range={}+{}: untouched_changed={untouched_bad} >1ulp={big} \
                  spec_row_diffs={spec_bad} ring_commit={} flag0_untouched={} short_ctx>1ulp={short_big} \
                  draft>1ulp={draft_big} -> {}",
                 c.0, c.1, c.2, c.3, c.4, if ring_ok { "ok" } else { "BAD" }, if flag0_ok { "ok" } else { "BAD" },
                 if ok { "PASS" } else { "FAIL" });
    }
    println!("  penalized logits vs the reference implementation transcription: exact={n_exact} 1-ulp={n_ulp1}");
    println!("{}", if all_ok { "PROBE_PENALTIES_OK" } else { "PROBE_PENALTIES_FAIL" });
    if !all_ok { bail!("penalty probe failed"); }
    Ok(())
}

