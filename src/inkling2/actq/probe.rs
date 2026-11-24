//! `--probe-inkling2-actq --model-dir <stock pack> [--inkling2-actq-fixture <sgl_actq.safetensors>]`
//!   1. GPU quantisers + GU epilogue vs the external reference's own outputs (fixture, bit-exact).
//!   2. Real tensors (layer 0 NVFP4, layer 20 MXFP8; x = rms(embed[id]) * mlp_norm.weight, K2's probe inputs):
//!      GPU x quantisation vs the CPU model (bit-exact); native MoE (decode M = 1) vs an f64 CPU model of the native
//!      pipeline (the reference's roundings; precise sigmoid / tanh instead of the device's approx: tolerance);
//!      batch invariance M = 4 / 8 / 16 and prefill rows == decode rows (bitwise); native vs the 16-bit path and vs
//!      the f32 reference (cpu.rs math) as the per-layer precision cost; per-layer timings, both modes.
//! `--probe-inkling2-logits --model-dir <pack> --inkling2-logits-tokens <ids.json> --inkling2-logits-out <dir>
//!   [--inkling2-expert-act native]`: teacher-forced logits of every position of every sequence (<= 2,048 tokens,
//!   one prefill chunk each), written as `<dir>/seq<i>.f32` ([len][vocab] softcapped fp32) + NLL / top-1 on stdout.

use super::cpu;
use super::{ActqScratch, ExpertAct};
use crate::inkling2::cpu::matmul;
use crate::inkling2::gpu::moe::{self, MoeLayerDev, MoeScratch, H, INTER, TOPK};
use crate::inkling2::gpu::{dptr, load_module, time_ms};
use crate::inkling2::pack::{ExpFmt, Inkling2Pack, P};
use crate::ink2_launch;
use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::CudaDevice;
use safetensors::SafeTensors;
use std::path::Path;
use std::sync::Arc;

const NT: usize = 2048;
const NREF: usize = 16;

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn softcap(x: f32, c: f32) -> f32 { (x / c).tanh() * c }
fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut n, mut d) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) { d += (*x as f64 - *y as f64).powi(2); n += (*y as f64).powi(2); }
    (d / n.max(1e-300)).sqrt()
}
fn bits_eq(a: &[f32], b: &[f32]) -> bool { a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) }
fn verdict(ok: bool) -> &'static str { if ok { "PASS" } else { "FAIL" } }

fn token_ids(vocab: usize) -> Vec<usize> {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..NT).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s >> 33) as usize) % vocab }).collect()
}
fn normed_rows(pack: &Inkling2Pack, ids: &[usize], w: &[f32]) -> Result<Vec<f32>> {
    let name = format!("{P}embed.weight");
    let mut x = vec![0f32; ids.len() * H];
    for (t, &id) in ids.iter().enumerate() {
        let r = pack.st.plain_f32_range(&name, id * H, (id + 1) * H)?;
        let ms: f64 = r.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / H as f64;
        let inv = 1.0 / ((ms as f32) + pack.cfg.eps).sqrt();
        for j in 0..H { x[t * H + j] = r[j] * inv * w[j]; }
    }
    Ok(x)
}

// ------------------------------------------------------------------------------------------------ 1. fixture
fn probe_fixture(dev: &Arc<CudaDevice>, path: &Path, gcap: f32, ucap: f32) -> Result<bool> {
    let raw = std::fs::read(path).with_context(|| format!("fixture {}", path.display()))?;
    let st = SafeTensors::deserialize(&raw)?;
    let get = |n: &str| -> Result<Vec<u8>> { Ok(st.tensor(n).with_context(|| n.to_string())?.data().to_vec()) };
    let f32v = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() };
    let r = cpu::FIX_ROWS;
    let mut ok = true;
    for (k, seed) in [(2304usize, 1u32), (1152, 2)] {
        let bits: Vec<u16> = (0..r * k).map(|i| cpu::hbits(i / k, i % k, k, seed)).collect();
        let xb = dev.htod_sync_copy(&bits)?;
        let xf = dev.htod_sync_copy(&bits.iter().map(|&b| cpu::bf16_val(b)).collect::<Vec<f32>>())?;
        let (c, s, a) = (dev.alloc_zeros::<u8>(r * k)?, dev.alloc_zeros::<u8>(r * k / 16)?, dev.alloc_zeros::<f32>(r)?);
        for (fmt, f32in) in [(ExpFmt::Nvfp4, false), (ExpFmt::Nvfp4, true), (ExpFmt::Mxfp8, false), (ExpFmt::Mxfp8, true)] {
            if f32in && k != 2304 { continue; }
            let x = if f32in { dptr(&xf, 0) } else { dptr(&xb, 0) };
            super::quant(dev, fmt, k, f32in, x, dptr(&c, 0), dptr(&s, 0), dptr(&a, 0), r, 0, 0)?;
            dev.synchronize()?;
            let (gc, gs, ga) = (dev.dtoh_sync_copy(&c)?, dev.dtoh_sync_copy(&s)?, f32v(&dev.dtoh_sync_copy(&a)?.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()));
            let (nd, what) = match fmt {
                ExpFmt::Nvfp4 => {
                    let (rc, rs, ra) = (get(&format!("{k}.nv.codes"))?, get(&format!("{k}.nv.scales"))?, f32v(&get(&format!("{k}.nv.amax"))?));
                    let d = gc[..r * k / 2].iter().zip(&rc).filter(|(x, y)| x != y).count()
                        + gs[..r * k / 16].iter().zip(&rs).filter(|(x, y)| x != y).count()
                        + ga.iter().zip(&ra).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                    (d, "nvfp4 codes + scales + amax")
                }
                ExpFmt::Mxfp8 => {
                    let (rc, re) = (get(&format!("{k}.mx.codes"))?, get(&format!("{k}.mx.exp"))?);
                    let d = gc[..r * k].iter().zip(&rc).filter(|(x, y)| x != y).count()
                        + gs[..r * k / 32].iter().zip(&re).filter(|(x, y)| x != y).count();
                    (d, "mxfp8 codes + exponents")
                }
            };
            println!("INK2_ACTQ_FIX width {k} {what} (input {}): {nd} byte mismatches vs the external reference -> {}",
                     if f32in { "fp32 of bf16 values" } else { "bf16" }, verdict(nd == 0));
            ok &= nd == 0;
        }
    }
    // GU epilogue + quantiser vs the reference's swiglu_quantize kernels
    let (n2, n) = (2304usize, 1152usize);
    let bits: Vec<u16> = (0..r * n2).map(|i| cpu::hbits(i / n2, i % n2, n2, 3)).collect();
    let acc = dev.htod_sync_copy(&bits)?;
    let aam = dev.htod_sync_copy(&f32v(&get("sw.in.acc_amax")?))?;
    let gam = dev.htod_sync_copy(&f32v(&get("sw.in.gammas")?))?;
    let wv = dev.htod_sync_copy(&f32v(&get("sw.in.w")?))?;
    let act = dev.alloc_zeros::<u16>(r * n)?;
    let (c, s, a) = (dev.alloc_zeros::<u8>(r * n)?, dev.alloc_zeros::<u8>(r * n / 16)?, dev.alloc_zeros::<f32>(r)?);
    for nv in [1i32, 0] {
        ink2_launch!(dev, "ink2q_gu_epi_test", (r as u32, 1, 1), (256, 1, 1), 0,
                     (dptr(&acc, 0), dptr(&aam, 0), dptr(&wv, 0), dptr(&gam, 0), dptr(&act, 0), n as i32, nv, gcap, ucap))?;
        let fmt = if nv == 1 { ExpFmt::Nvfp4 } else { ExpFmt::Mxfp8 };
        super::quant(dev, fmt, n, false, dptr(&act, 0), dptr(&c, 0), dptr(&s, 0), dptr(&a, 0), r, 0, 0)?;
        dev.synchronize()?;
        let (gc, gs) = (dev.dtoh_sync_copy(&c)?, dev.dtoh_sync_copy(&s)?);
        let nd = if nv == 1 {
            let ga: Vec<f32> = dev.dtoh_sync_copy(&a)?;
            let (rc, rs, ra) = (get("sw.nv.codes")?, get("sw.nv.scales")?, f32v(&get("sw.nv.amax")?));
            gc[..r * n / 2].iter().zip(&rc).filter(|(x, y)| x != y).count() + gs[..r * n / 16].iter().zip(&rs).filter(|(x, y)| x != y).count()
                + ga.iter().zip(&ra).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
        } else {
            let (rc, re) = (get("sw.mx.codes")?, get("sw.mx.exp")?);
            gc[..r * n].iter().zip(&rc).filter(|(x, y)| x != y).count() + gs[..r * n / 32].iter().zip(&re).filter(|(x, y)| x != y).count()
        };
        println!("INK2_ACTQ_FIX swiglu+quantise ({}): {nd} byte mismatches vs the external reference's swiglu_quantize -> {}",
                 if nv == 1 { "nvfp4, outer scales, gamma" } else { "mxfp8, gamma" }, verdict(nd == 0));
        ok &= nd == 0;
    }
    Ok(ok)
}

// ------------------------------------------------------------------------------------------------ 2. real tensors
/// Raw dequantised expert weight [n][k] WITHOUT the NVFP4 tensor scale (e2m1 * e4m3; MXFP8: e4m3 * 2^(e-127)),
/// checkpoint row order (w13 rows interleaved gate / up).
fn expert_raw(pack: &Inkling2Pack, l: usize, e: usize, proj: &str) -> Result<(Vec<f32>, f32, f32)> {
    let b = format!("{P}layers.{l}.mlp.experts.{proj}_weight");
    let (n, k) = if proj == "w13" { (2 * INTER, H) } else { (H, INTER) };
    match pack.exp_fmt[l] {
        ExpFmt::Nvfp4 => {
            let codes = pack.st.raw_range(&b, (e * n * k / 2) as u64, ((e + 1) * n * k / 2) as u64)?;
            let sc = pack.st.raw_range(&format!("{b}_scale"), (e * n * k / 16) as u64, ((e + 1) * n * k / 16) as u64)?;
            let amax = pack.plain(&format!("{b}_amax"))?[e];
            let den = pack.plain(&format!("{b}_outer_scale_denominator"))?[0];
            Ok(((0..n * k).map(|i| cpu::nv4_deq(&codes[(i / k) * k / 2..], &sc[(i / k) * k / 16..], i % k)).collect(), amax, den))
        }
        ExpFmt::Mxfp8 => {
            let codes = pack.st.raw_range(&b, (e * n * k) as u64, ((e + 1) * n * k) as u64)?;
            let sc = pack.st.raw_range(&format!("{b}_scale"), (e * n * k / 32) as u64, ((e + 1) * n * k / 32) as u64)?;
            Ok(((0..n * k).map(|i| cpu::mx8_deq(&codes[(i / k) * k..], &sc[(i / k) * k / 32..], i % k)).collect(), 1.0, 1.0))
        }
    }
}
fn dot64(a: &[f32], b: &[f32]) -> f64 { a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum() }
fn row_scale(xa: f32, wa: f32, den: f32) -> f32 { (xa * wa) * (1.0f32 / (1536.0f32 * den)) }

/// The native pipeline on the CPU (f64 GEMM sums, the reference's bf16 roundings, precise transcendentals).
fn ref_native(pack: &Inkling2Pack, l: usize, x: &[f32], sel: &[(usize, f32)]) -> Result<Vec<f32>> {
    let c = &pack.cfg;
    let fmt = pack.exp_fmt[l];
    let xb: Vec<f32> = x.iter().map(|&v| cpu::bf16_rn(v)).collect();
    let (xd, xamax): (Vec<f32>, f32) = match fmt {
        ExpFmt::Nvfp4 => { let (cc, s, a) = cpu::nv4_row(&xb); ((0..H).map(|j| cpu::nv4_deq(&cc, &s, j)).collect(), a) }
        ExpFmt::Mxfp8 => { let (cc, s) = cpu::mx8_row(&xb); ((0..H).map(|j| cpu::mx8_deq(&cc, &s, j)).collect(), 1.0) }
    };
    let mut out = vec![0f32; H];
    let mut sorted = sel.to_vec();
    sorted.sort_by_key(|p| p.0);
    for &(e, gamma) in &sorted {
        let (w13, a13, d13) = expert_raw(pack, l, e, "w13")?;
        let (w2, a2, d2) = expert_raw(pack, l, e, "w2")?;
        let sc = row_scale(xamax, a13, d13);
        let act: Vec<f32> = (0..INTER).map(|i| {
            let rg = dot64(&xd, &w13[(2 * i) * H..(2 * i + 1) * H]) as f32;
            let ru = dot64(&xd, &w13[(2 * i + 1) * H..(2 * i + 2) * H]) as f32;
            let (g, u) = match fmt {
                ExpFmt::Nvfp4 => (cpu::bf16_rn(cpu::bf16_rn(rg) * sc), cpu::bf16_rn(cpu::bf16_rn(ru) * sc)),
                ExpFmt::Mxfp8 => (cpu::bf16_rn(rg), cpu::bf16_rn(ru)),
            };
            cpu::bf16_rn(softcap(g, c.gate_softcap) * sigmoid(g) * softcap(u, c.up_softcap) * gamma)
        }).collect();
        let (ad, aamax): (Vec<f32>, f32) = match fmt {
            ExpFmt::Nvfp4 => { let (cc, s, a) = cpu::nv4_row(&act); ((0..INTER).map(|j| cpu::nv4_deq(&cc, &s, j)).collect(), a) }
            ExpFmt::Mxfp8 => { let (cc, s) = cpu::mx8_row(&act); ((0..INTER).map(|j| cpu::mx8_deq(&cc, &s, j)).collect(), 1.0) }
        };
        let rs = row_scale(aamax, a2, d2);
        for j in 0..H {
            let raw = dot64(&ad, &w2[j * INTER..(j + 1) * INTER]) as f32;
            let y = match fmt { ExpFmt::Nvfp4 => cpu::bf16_rn(cpu::bf16_rn(raw) * rs), ExpFmt::Mxfp8 => cpu::bf16_rn(raw) };
            out[j] += y;
        }
    }
    Ok(out)
}
/// The f32 reference (cpu.rs math: exact weights, f32 activations) for the same selection.
fn ref_f32(pack: &Inkling2Pack, l: usize, x: &[f32], sel: &[(usize, f32)]) -> Result<Vec<f32>> {
    let c = &pack.cfg;
    let mut acc = vec![0f64; H];
    for &(e, wgt) in sel {
        let (_, _, w13) = pack.expert_f32(l, e, "w13")?;
        let (_, _, w2) = pack.expert_f32(l, e, "w2")?;
        let gu = matmul(x, 1, &w13, 2 * INTER, H);
        // expert_f32 de-interleaves w13 into [gate rows; up rows] (transformers' Interleave)
        let a: Vec<f32> = (0..INTER).map(|i| softcap(gu[i], c.gate_softcap) * sigmoid(gu[i]) * softcap(gu[INTER + i], c.up_softcap)).collect();
        let y = matmul(&a, 1, &w2, H, INTER);
        for j in 0..H { acc[j] += (y[j] * wgt) as f64; }
    }
    Ok(acc.into_iter().map(|v| v as f32).collect())
}

fn probe_layer(dev: &Arc<CudaDevice>, pack: &Inkling2Pack, l: usize, ids: &[usize]) -> Result<bool> {
    let c = &pack.cfg;
    let mut ok = true;
    let t0 = std::time::Instant::now();
    let nat = super::load_layer(dev, pack, l)?;
    let a16 = MoeLayerDev::load(dev, pack, l)?;
    let fmt = nat.ex.fmt;
    println!("INK2_ACTQ L{l} ({fmt:?}): native + a16 device weights loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let w = pack.plain(&format!("{P}layers.{l}.mlp_norm.weight"))?;
    let x = normed_rows(pack, ids, &w)?;
    let xd = dev.htod_copy(x.clone())?;
    // (a) x quantisation, all NT rows, vs the CPU model
    let aq = ActqScratch::new(dev, NT, c.gate_softcap, c.up_softcap)?;
    super::quant(dev, fmt, H, true, dptr(&xd, 0), dptr(&aq.xc, 0), dptr(&aq.xs, 0), dptr(&aq.xa, 0), NT, 0, 0)?;
    dev.synchronize()?;
    let (gc, gs, ga) = (dev.dtoh_sync_copy(&aq.xc)?, dev.dtoh_sync_copy(&aq.xs)?, dev.dtoh_sync_copy(&aq.xa)?);
    let mut nd = 0usize;
    let mut n4 = 0usize;
    for t in 0..NT {
        let xb: Vec<f32> = x[t * H..(t + 1) * H].iter().map(|&v| cpu::bf16_rn(v)).collect();
        match fmt {
            ExpFmt::Nvfp4 => {
                let (cc, s, a) = cpu::nv4_row(&xb);
                nd += cc.iter().zip(&gc[t * H / 2..(t + 1) * H / 2]).filter(|(p, q)| p != q).count()
                    + s.iter().zip(&gs[t * H / 16..(t + 1) * H / 16]).filter(|(p, q)| p != q).count() + (a.to_bits() != ga[t].to_bits()) as usize;
                // blocks that chose the 4-scale
                let enc = (1536.0f32 / a).min(f32::MAX);
                for b in 0..H / 16 {
                    let bm = xb[16 * b..16 * b + 16].iter().fold(0f32, |m, v| m.max(v.abs()));
                    n4 += (cpu::e4m3_rn((bm * (enc * 0.25)).min(448.0)).0 == s[b] && cpu::e4m3_rn((bm * (enc * ((1.0f64 / 6.0) as f32))).min(448.0)).0 != s[b]) as usize;
                }
            }
            ExpFmt::Mxfp8 => {
                let (cc, s) = cpu::mx8_row(&xb);
                nd += cc.iter().zip(&gc[t * H..(t + 1) * H]).filter(|(p, q)| p != q).count()
                    + s.iter().zip(&gs[t * H / 32..(t + 1) * H / 32]).filter(|(p, q)| p != q).count();
            }
        }
    }
    let extra = if fmt == ExpFmt::Nvfp4 { format!(" (blocks on the max-4 scale: {:.1}%)", 100.0 * n4 as f64 / (NT * H / 16) as f64) } else { String::new() };
    println!("INK2_ACTQ_X L{l}: GPU x quantisation of {NT} real rows vs the CPU model: {nd} byte mismatches{extra} -> {}", verdict(nd == 0));
    ok &= nd == 0;
    // (b) native decode M = 1 vs the CPU native model; native vs a16 vs f32 reference
    let scd = MoeScratch::new(dev, 16)?;
    let aqd = ActqScratch::new(dev, 16, c.gate_softcap, c.up_softcap)?;
    let outd = dev.alloc_zeros::<f32>(16 * H)?;
    let (mut out1, mut o16, mut sels) = (Vec::new(), Vec::new(), Vec::new());
    for t in 0..NREF {
        super::forward(dev, &nat, &scd, &aqd, dptr(&xd, t * H), dptr(&outd, 0), 1)?;
        out1.push(dev.dtoh_sync_copy(&outd)?[..H].to_vec());
        let (gid, gw) = (dev.dtoh_sync_copy(&scd.ids)?, dev.dtoh_sync_copy(&scd.wts)?);
        sels.push((0..TOPK).map(|k| (gid[k] as usize, gw[k])).collect::<Vec<_>>());
        moe::forward(dev, &a16, &scd, dptr(&xd, t * H), dptr(&outd, 0), 1)?;
        o16.push(dev.dtoh_sync_copy(&outd)?[..H].to_vec());
    }
    let (mut wn, mut mn, mut m_na16, mut m_nf, mut m_af) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let refs: Vec<Result<(Vec<f32>, Vec<f32>)>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..NREF).map(|t| {
            let (xr, sel) = (&x[t * H..(t + 1) * H], &sels[t]);
            s.spawn(move || -> Result<(Vec<f32>, Vec<f32>)> { Ok((ref_native(pack, l, xr, sel)?, ref_f32(pack, l, xr, sel)?)) })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (t, rr) in refs.into_iter().enumerate() {
        let (rn, rf) = rr?;
        let r = rel_l2(&out1[t], &rn);
        wn = wn.max(r);
        mn += r / NREF as f64;
        m_na16 += rel_l2(&out1[t], &o16[t]) / NREF as f64;
        m_nf += rel_l2(&out1[t], &rf) / NREF as f64;
        m_af += rel_l2(&o16[t], &rf) / NREF as f64;
    }
    let tol = 2e-2;
    println!("INK2_ACTQ_ACC L{l} native decode M=1 vs the CPU native model ({NREF} tokens): rel-L2 mean {mn:.2e} worst {wn:.2e} \
(tol {tol:.0e}: fp32 vs f64 sums and approx vs precise transcendentals before bf16 roundings / fp4 re-quantisation) -> {}", verdict(wn < tol));
    ok &= wn < tol;
    println!("INK2_ACTQ_COST L{l} ({fmt:?}) MoE output rel-L2 vs the f32 reference (cpu.rs math, same routing): native {m_nf:.3e}, \
a16 {m_af:.3e}; native vs a16 {m_na16:.3e}");
    for m in [4usize, 8, 16] {
        let mut bad = 0;
        for g in 0..NREF / m {
            super::forward(dev, &nat, &scd, &aqd, dptr(&xd, g * m * H), dptr(&outd, 0), m)?;
            let o = dev.dtoh_sync_copy(&outd)?;
            for r in 0..m { if !bits_eq(&o[r * H..(r + 1) * H], &out1[g * m + r]) { bad += 1; } }
        }
        println!("INK2_ACTQ_BINV L{l} M={m}: {}/{NREF} rows bit-identical to their M=1 run -> {}", NREF - bad, verdict(bad == 0));
        ok &= bad == 0;
    }
    // (c) prefill rows == decode rows
    let scp = MoeScratch::new(dev, NT)?;
    let outp = dev.alloc_zeros::<f32>(NT * H)?;
    super::PF_TILE128.store(1, std::sync::atomic::Ordering::Relaxed);
    super::forward(dev, &nat, &scp, &aq, dptr(&xd, 0), dptr(&outp, 0), NT)?;
    let op = dev.dtoh_sync_copy(&outp)?;
    super::PF_TILE128.store(0, std::sync::atomic::Ordering::Relaxed);
    super::forward(dev, &nat, &scp, &aq, dptr(&xd, 0), dptr(&outp, 0), NT)?;
    let op64 = dev.dtoh_sync_copy(&outp)?;
    let same = bits_eq(&op, &op64);
    println!("INK2_ACTQ_PF L{l} prefill 128-row tiles vs 64-row tiles: all {NT} rows bit-identical -> {}", verdict(same));
    ok &= same;
    let mut bad = 0;
    for t in 0..NREF { if !bits_eq(&op[t * H..(t + 1) * H], &out1[t]) { bad += 1; } }
    let tail0 = NT - 16;
    super::forward(dev, &nat, &scd, &aqd, dptr(&xd, tail0 * H), dptr(&outd, 0), 16)?;
    let ot = dev.dtoh_sync_copy(&outd)?;
    for r in 0..16 { if !bits_eq(&op[(tail0 + r) * H..(tail0 + r + 1) * H], &ot[r * H..(r + 1) * H]) { bad += 1; } }
    println!("INK2_ACTQ_PF L{l} prefill M={NT}: {}/32 checked rows bit-identical to decode -> {}", 32 - bad, verdict(bad == 0));
    ok &= bad == 0;
    // (d) timings: whole MoE block, both modes
    let sets = 16usize;
    for m in [1usize, 4, 8, 16] {
        let scs: Vec<MoeScratch> = (0..sets).map(|_| MoeScratch::new(dev, 16)).collect::<Result<_>>()?;
        let aqs: Vec<ActqScratch> = (0..sets).map(|_| ActqScratch::new(dev, 16, c.gate_softcap, c.up_softcap)).collect::<Result<_>>()?;
        let t_a = time_ms(dev, 2, 15, &mut || { for (s, sc) in scs.iter().enumerate() { moe::forward(dev, &a16, sc, dptr(&xd, 64 * s * H), dptr(&outd, 0), m)?; } Ok(()) })? / sets as f64;
        let t_n = time_ms(dev, 2, 15, &mut || { for (s, sc) in scs.iter().enumerate() { super::forward(dev, &nat, sc, &aqs[s], dptr(&xd, 64 * s * H), dptr(&outd, 0), m)?; } Ok(()) })? / sets as f64;
        println!("INK2_ACTQ_TIME L{l} {fmt:?} decode M={m:>2}: whole MoE a16 {:.1} us | native {:.1} us ({:+.1}%)", t_a * 1e3, t_n * 1e3, 100.0 * (t_n / t_a - 1.0));
    }
    let t_a = time_ms(dev, 2, 10, &mut || moe::forward(dev, &a16, &scp, dptr(&xd, 0), dptr(&outp, 0), NT))?;
    use std::sync::atomic::Ordering;
    let mut tn = [0f64; 2];
    for (i, t128) in [0u8, 1].into_iter().enumerate() {
        super::PF_TILE128.store(t128, Ordering::Relaxed);
        tn[i] = time_ms(dev, 2, 10, &mut || super::forward(dev, &nat, &scp, &aq, dptr(&xd, 0), dptr(&outp, 0), NT))?;
    }
    super::PF_TILE128.store(0, Ordering::Relaxed);
    println!("INK2_ACTQ_TIME L{l} {fmt:?} prefill M={NT}: whole MoE a16 {:.2} ms | native 64-row tiles {:.2} ms ({:.2}x) | native 128-row tiles {:.2} ms ({:.2}x)",
             t_a, tn[0], t_a / tn[0], tn[1], t_a / tn[1]);
    Ok(ok)
}

pub fn probe_actq(dir: &Path, args: &[String]) -> Result<()> {
    let pack = Inkling2Pack::open(dir)?;
    let dev = CudaDevice::new(0)?;
    load_module(&dev)?;
    let mut ok = true;
    let fx = args.iter().position(|a| a == "--inkling2-actq-fixture").and_then(|i| args.get(i + 1)).cloned()
        .unwrap_or_else(|| "src/inkling2/actq/fixtures/sgl_actq.safetensors".to_string());
    if Path::new(&fx).exists() {
        ok &= probe_fixture(&dev, Path::new(&fx), pack.cfg.gate_softcap, pack.cfg.up_softcap)?;
    } else {
        println!("INK2_ACTQ_FIX skipped: no fixture at {fx} (--inkling2-actq-fixture)");
    }
    let ids = token_ids(pack.cfg.unpadded_vocab);
    let nv = (0..pack.cfg.layers).find(|&l| pack.exp_fmt[l] == ExpFmt::Nvfp4);
    let mx = (0..pack.cfg.layers).find(|&l| pack.exp_fmt[l] == ExpFmt::Mxfp8);
    ensure!(nv.is_some() && mx.is_some(), "the pack lacks an NVFP4 or an MXFP8 expert layer");
    for l in [nv, mx].into_iter().flatten() { ok &= probe_layer(&dev, &pack, l, &ids)?; }
    if !ok { bail!("INK2_ACTQ_PROBE_FAIL (see the FAIL lines)"); }
    println!("INK2_ACTQ_PROBE_OK");
    Ok(())
}

// ------------------------------------------------------------------------------------------------ logits dump
pub fn probe_logits(dir: &Path, args: &[String]) -> Result<()> {
    use crate::inkling2::model::{GpuModel, Seg, CHUNK};
    let arg = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let tf = arg("--inkling2-logits-tokens").context("--inkling2-logits-tokens <ids.json> ({\"seqs\": [[ids...], ...]})")?;
    let out = arg("--inkling2-logits-out").context("--inkling2-logits-out <dir>")?;
    let mode = super::set_mode_from_args(args)?;
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&tf)?)?;
    let seqs: Vec<Vec<u32>> = v["seqs"].as_array().context("seqs")?.iter()
        .map(|s| s.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()).collect();
    std::fs::create_dir_all(&out)?;
    let dev = CudaDevice::new(0)?;
    let m = GpuModel::load(dir, &dev)?;
    let mut ws = m.new_work()?;
    let vocab = m.vocab();
    println!("INK2_LOGITS mode {mode:?}: {} sequences, vocab {vocab}", seqs.len());
    let (mut nll, mut ntok, mut top1) = (0f64, 0usize, 0usize);
    let t0 = std::time::Instant::now();
    for (i, ids) in seqs.iter().enumerate() {
        ensure!(ids.len() >= 2 && ids.len() <= CHUNK, "sequence {i}: {} tokens (2..={CHUNK})", ids.len());
        let mut lanes = vec![m.new_lane(ids.len())?];
        {
            let refs: Vec<_> = lanes.iter().collect();
            m.forward(&mut ws, &refs, &[Seg { lane: 0, p0: 0, m: ids.len() }], ids, &[])?;
        }
        lanes[0].pos = ids.len();
        let mut buf: Vec<u8> = Vec::with_capacity(ids.len() * vocab * 4);
        let (mut snll, mut sn) = (0f64, 0usize);
        for r0 in (0..ids.len()).step_by(16) {
            let n = 16.min(ids.len() - r0);
            m.logits_rows(&mut ws, r0, n)?;
            let lg = m.logits_host(&ws, n)?;
            for r in 0..n {
                let row = &lg[r * vocab..(r + 1) * vocab];
                buf.extend(row.iter().flat_map(|x| x.to_le_bytes()));
                let p = r0 + r;
                if p + 1 < ids.len() {
                    let mx = row.iter().fold(f32::MIN, |a, &b| a.max(b)) as f64;
                    let lse = mx + row.iter().map(|&x| (x as f64 - mx).exp()).sum::<f64>().ln();
                    let tgt = ids[p + 1] as usize;
                    snll += lse - row[tgt] as f64;
                    sn += 1;
                    let am = row.iter().enumerate().fold((0usize, f32::MIN), |a, (j, &x)| if x > a.1 { (j, x) } else { a }).0;
                    top1 += (am == tgt) as usize;
                }
            }
        }
        std::fs::write(Path::new(&out).join(format!("seq{i}.f32")), &buf)?;
        nll += snll;
        ntok += sn;
        println!("INK2_LOGITS seq {i}: {} tokens, NLL {:.4} nats/token ({:.1} s elapsed)", ids.len(), snll / sn as f64, t0.elapsed().as_secs_f64());
    }
    println!("INK2_LOGITS_DONE mode {mode:?}: {ntok} predicted tokens, NLL {:.5} nats/token, ppl {:.4}, next-token top-1 {:.2}%",
             nll / ntok as f64, (nll / ntok as f64).exp(), 100.0 * top1 as f64 / ntok as f64);
    let _ = ExpertAct::A16;
    Ok(())
}
