//! `--probe-inkling2-moe --model-dir <stock pack>`: the MoE + LM-head kernels vs the CPU reference on REAL checkpoint
//! tensors (layer 0 = NVFP4 experts, layer 20 = MXFP8 experts, the real routers, the real unembed), batch invariance,
//! prefill == decode row identity, and per-kernel timings against the 238 GB/s roofline.
//!
//! Activations: x_t = rms(embed[id_t]) * mlp_norm.weight (the MoE input is an RMSNorm output: unit-RMS direction times
//! the layer's real norm weight) for 2,048 token ids from a fixed LCG; the head reads rms(embed[id]) * norm.weight.
//! The reference math mirrors `cpu.rs` (`CpuModel::moe`, `CpuModel::logits`) line for line: f64-accumulated f32 GEMVs,
//! sigmoid + bias selection sorted by (biased score desc, id asc), weights from unbiased scores / (sum + 1e-20) * 8,
//! softcap(g,4)*sigmoid(g)*softcap(u,25), y*w summed in f64; the experts dequantized by `pack::expert_f32`
//! (bit-exact to the transformers loader, S0).

use super::head::{self, HeadDev, HeadScratch};
use super::moe::{self, MoeLayerDev, MoeScratch, E, H, INTER, TOPK};
use super::{dptr, load_module, time_ms};
use crate::inkling2::cpu::matmul;
use crate::inkling2::pack::{ExpFmt, Inkling2Pack, P};
use anyhow::{bail, ensure, Result};
use cudarc::driver::CudaDevice;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

const NT: usize = 2048;
const NREF: usize = 16;
const ROOF: f64 = 238.0;

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn softcap(x: f32, c: f32) -> f32 { (x / c).tanh() * c }

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut n, mut d) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) { d += (*x as f64 - *y as f64).powi(2); n += (*y as f64).powi(2); }
    (d / n.max(1e-300)).sqrt()
}

fn token_ids(vocab: usize) -> Vec<usize> {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..NT).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s >> 33) as usize) % vocab }).collect()
}

/// rms(embed[id]) * w for every id (eps from the config).
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

/// cpu.rs's router selection: sorted by (biased score desc, id asc), weights from unbiased scores.
fn ref_select(logits: &[f32], bias: &[f32], t: usize) -> (Vec<(usize, f32)>, f32) {
    let sc: Vec<f32> = logits[t * E..(t + 1) * E].iter().map(|&v| sigmoid(v)).collect();
    let mut idx: Vec<usize> = (0..E).collect();
    idx.sort_by(|&a, &b| (sc[b] + bias[b]).partial_cmp(&(sc[a] + bias[a])).unwrap().then(a.cmp(&b)));
    let top = &idx[..TOPK];
    let s: f32 = top.iter().map(|&e| sc[e]).sum();
    let gap = (sc[idx[TOPK - 1]] + bias[idx[TOPK - 1]]) - (sc[idx[TOPK]] + bias[idx[TOPK]]);
    (top.iter().map(|&e| (e, sc[e] / (s + 1e-20) * 8.0)).collect(), gap)
}

/// cpu.rs's expert evaluation for the given rows and selections.
fn ref_experts(pack: &Inkling2Pack, l: usize, x: &[f32], rows: &[usize], sel: &[Vec<(usize, f32)>]) -> Result<Vec<Vec<f32>>> {
    let c = &pack.cfg;
    let mut uniq: Vec<usize> = rows.iter().flat_map(|&t| sel[t].iter().map(|p| p.0)).collect();
    uniq.sort();
    uniq.dedup();
    let nt = 16usize;
    let deq: Vec<Result<(usize, Vec<f32>, Vec<f32>)>> = std::thread::scope(|s| {
        let chunks: Vec<Vec<usize>> = (0..nt).map(|i| uniq.iter().cloned().skip(i).step_by(nt).collect()).collect();
        let hs: Vec<_> = chunks.into_iter().map(|ch| s.spawn(move || ch.into_iter().map(|e| {
            let (_, _, w13) = pack.expert_f32(l, e, "w13")?;
            let (_, _, w2) = pack.expert_f32(l, e, "w2")?;
            Ok((e, w13, w2))
        }).collect::<Vec<_>>())).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let mut ex: BTreeMap<usize, (Vec<f32>, Vec<f32>)> = BTreeMap::new();
    for r in deq { let (e, a, b) = r?; ex.insert(e, (a, b)); }
    let ii = INTER;
    let mut outs = Vec::new();
    for &t in rows {
        let xr = &x[t * H..(t + 1) * H];
        let mut acc = vec![0f64; H];
        for &(e, wgt) in &sel[t] {
            let (w13, w2) = &ex[&e];
            let gu = matmul(xr, 1, w13, 2 * ii, H);
            let a: Vec<f32> = (0..ii).map(|i| softcap(gu[i], c.gate_softcap) * sigmoid(gu[i]) * softcap(gu[ii + i], c.up_softcap)).collect();
            let y = matmul(&a, 1, w2, H, ii);
            for j in 0..H { acc[j] += (y[j] * wgt) as f64; }
        }
        outs.push(acc.into_iter().map(|v| v as f32).collect());
    }
    Ok(outs)
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool { a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) }

fn probe_layer(dev: &Arc<CudaDevice>, pack: &Inkling2Pack, l: usize, ids: &[usize]) -> Result<bool> {
    let mut ok = true;
    let t0 = std::time::Instant::now();
    let lay = MoeLayerDev::load(dev, pack, l)?;
    let fmt = lay.ex.fmt;
    println!("INK2_MOE L{l} ({fmt:?}): device weights loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let w = pack.plain(&format!("{P}layers.{l}.mlp_norm.weight"))?;
    let x = normed_rows(pack, ids, &w)?;
    let xd = dev.htod_copy(x.clone())?;
    // ---------------- router vs reference (all NT tokens)
    let rw = pack.plain(&format!("{P}layers.{l}.mlp.gate.weight"))?;
    let rb = pack.plain(&format!("{P}layers.{l}.mlp.gate.bias"))?;
    let lref = matmul(&x, NT, &rw, E, H);
    let scp = MoeScratch::new(dev, NT)?;
    let outp = dev.alloc_zeros::<f32>(NT * H)?;
    moe::forward(dev, &lay, &scp, dptr(&xd, 0), dptr(&outp, 0), NT)?;
    dev.synchronize()?;
    let lg = dev.dtoh_sync_copy(&scp.logits)?;
    let gid = dev.dtoh_sync_copy(&scp.ids)?;
    let gw = dev.dtoh_sync_copy(&scp.wts)?;
    let mut maxd = 0f32;
    for (a, b) in lg.iter().zip(&lref) { maxd = maxd.max((a - b).abs()); }
    let (mut sel_bad, mut near_tie, mut wmax) = (0usize, 0usize, 0f32);
    let mut sel = Vec::with_capacity(NT);
    for t in 0..NT {
        let (s, gap) = ref_select(&lref, &rb, t);
        let mut srt = s.clone();
        srt.sort_by_key(|p| p.0);
        let same = (0..TOPK).all(|k| gid[t * TOPK + k] as usize == srt[k].0);
        if !same { if gap.abs() < 1e-5 { near_tie += 1; } else { sel_bad += 1; } } else {
            for k in 0..TOPK { wmax = wmax.max((gw[t * TOPK + k] - srt[k].1).abs()); }
        }
        sel.push(s);
    }
    let rok = maxd < 1e-4 && sel_bad == 0 && wmax < 1e-5;
    println!("INK2_ROUTER L{l}: {NT} tokens: logits max|d| {maxd:.2e} (tol 1e-4: fp32 fma vs the f64-accumulated oracle), \
selections {} / {NT} identical ({sel_bad} differ, {near_tie} near-ties |gap| < 1e-5), weights max|d| {wmax:.2e} (tol 1e-5) -> {}",
             NT - sel_bad - near_tie, if rok { "PASS" } else { "FAIL" });
    ok &= rok;
    // ---------------- decode M = 1 / 4 / 8 / 16 on the first 16 tokens; batch invariance
    let scd = MoeScratch::new(dev, 16)?;
    let outd = dev.alloc_zeros::<f32>(16 * H)?;
    let mut out1: Vec<Vec<f32>> = Vec::new();
    for t in 0..NREF {
        moe::forward(dev, &lay, &scd, dptr(&xd, t * H), dptr(&outd, 0), 1)?;
        out1.push(dev.dtoh_sync_copy(&outd)?[..H].to_vec());
    }
    let refs = ref_experts(pack, l, &x, &(0..NREF).collect::<Vec<_>>(), &sel)?;
    let (mut worst, mut mean) = (0f64, 0f64);
    for t in 0..NREF { let r = rel_l2(&out1[t], &refs[t]); worst = worst.max(r); mean += r / NREF as f64; }
    let tol = 2e-3;
    let aok = worst < tol;
    println!("INK2_MOE_ACC L{l} decode M=1 vs cpu.rs ({NREF} tokens): rel-L2 mean {mean:.2e} worst {worst:.2e} (tol {tol:.0e}: \
f16 expert input + f16 SwiGLU output, ~2^-11 each) -> {}", if aok { "PASS" } else { "FAIL" });
    ok &= aok;
    for m in [4usize, 8, 16] {
        let mut bad = 0;
        for g in 0..NREF / m {
            moe::forward(dev, &lay, &scd, dptr(&xd, g * m * H), dptr(&outd, 0), m)?;
            let o = dev.dtoh_sync_copy(&outd)?;
            for r in 0..m { if !bits_eq(&o[r * H..(r + 1) * H], &out1[g * m + r]) { bad += 1; } }
        }
        let bok = bad == 0;
        println!("INK2_MOE_BINV L{l} M={m}: {}/{NREF} rows bit-identical to their M=1 run -> {}", NREF - bad, if bok { "PASS" } else { "FAIL" });
        ok &= bok;
    }
    // ---------------- prefill (M = NT) rows vs decode rows (bitwise) and vs the reference
    let op = dev.dtoh_sync_copy(&outp)?;
    let mut bad = 0;
    for t in 0..NREF { if !bits_eq(&op[t * H..(t + 1) * H], &out1[t]) { bad += 1; } }
    let tail0 = NT - 16;
    moe::forward(dev, &lay, &scd, dptr(&xd, tail0 * H), dptr(&outd, 0), 16)?;
    let ot = dev.dtoh_sync_copy(&outd)?;
    for r in 0..16 { if !bits_eq(&op[(tail0 + r) * H..(tail0 + r + 1) * H], &ot[r * H..(r + 1) * H]) { bad += 1; } }
    let pok = bad == 0;
    println!("INK2_MOE_PF L{l} prefill M={NT}: {}/32 checked rows (0..16 vs M=1, {tail0}..{NT} vs M=16) bit-identical to decode -> {}",
             32 - bad, if pok { "PASS" } else { "FAIL" });
    ok &= pok;
    // ---------------- timing
    let (b13, b2) = lay.ex.bytes_per_expert();
    let rbytes = E * H * 2;
    let sets = 16usize;
    for m in [1usize, 4, 8, 16] {
        let scs: Vec<MoeScratch> = (0..sets).map(|_| MoeScratch::new(dev, 16)).collect::<Result<_>>()?;
        let mut uni = 0usize;
        for (s, sc) in scs.iter().enumerate() {
            moe::forward(dev, &lay, sc, dptr(&xd, 64 * s * H), dptr(&outd, 0), m)?;
            uni += dev.dtoh_sync_copy(&sc.esel)?[0] as usize;
        }
        let per = |f: &mut dyn FnMut(&MoeScratch, usize) -> Result<()>| -> Result<f64> {
            Ok(time_ms(dev, 2, 15, &mut || { for (s, sc) in scs.iter().enumerate() { f(sc, s)?; } Ok(()) })? / sets as f64)
        };
        let t_r = per(&mut |sc, s| moe::router(dev, &lay, sc, dptr(&xd, 64 * s * H), m))?;
        let t_g = per(&mut |sc, _| moe::experts_decode(dev, &lay, sc, m, 1))?;
        let t_d = per(&mut |sc, _| moe::experts_decode(dev, &lay, sc, m, 2))?;
        let t_c = per(&mut |sc, _| moe::combine(dev, sc, dptr(&outd, 0), m))?;
        let t_f = per(&mut |sc, s| moe::forward(dev, &lay, sc, dptr(&xd, 64 * s * H), dptr(&outd, 0), m))?;
        let u = uni as f64 / sets as f64;
        let gbs = |bytes: f64, ms: f64| bytes / ms / 1e6;
        let fb = rbytes as f64 + u * (b13 + b2) as f64;
        println!("INK2_MOE_TIME L{l} {fmt:?} decode M={m:>2}: union {u:5.1} experts | router {:.1} us | gate/up {:.1} us ({:.0} GB/s) | \
down {:.1} us ({:.0} GB/s) | combine {:.1} us | whole MoE {:.1} us = {:.0} GB/s ({:.0}% of {ROOF:.0})",
                 t_r * 1e3, t_g * 1e3, gbs(u * b13 as f64, t_g), t_d * 1e3, gbs(u * b2 as f64, t_d), t_c * 1e3,
                 t_f * 1e3, gbs(fb, t_f), 100.0 * gbs(fb, t_f) / ROOF);
    }
    {
        let m = NT;
        let t_r = time_ms(dev, 2, 10, &mut || moe::router(dev, &lay, &scp, dptr(&xd, 0), m))?;
        let t_p = time_ms(dev, 2, 10, &mut || moe::route_prefill(dev, &scp, m))?;
        let t_g = time_ms(dev, 2, 10, &mut || moe::experts_prefill(dev, &lay, &scp, m, 1))?;
        let t_d = time_ms(dev, 2, 10, &mut || moe::experts_prefill(dev, &lay, &scp, m, 2))?;
        let t_c = time_ms(dev, 2, 10, &mut || moe::combine(dev, &scp, dptr(&outp, 0), m))?;
        let t_f = time_ms(dev, 2, 10, &mut || moe::forward(dev, &lay, &scp, dptr(&xd, 0), dptr(&outp, 0), m))?;
        let r = (m * TOPK) as f64;
        let fl_g = 2.0 * r * (2 * INTER * H) as f64;
        let fl_d = 2.0 * r * (H * INTER) as f64;
        println!("INK2_MOE_TIME L{l} {fmt:?} prefill M={m}: router {:.0} us | route {:.0} us | gate/up {:.0} us ({:.1} TFLOP/s) | \
down {:.0} us ({:.1} TFLOP/s) | combine {:.0} us | whole MoE {:.2} ms ({:.1} TFLOP/s, {:.0} tok/s per layer visit; all {E} experts' bytes {:.0} GB/s)",
                 t_r * 1e3, t_p * 1e3, t_g * 1e3, fl_g / t_g / 1e9, t_d * 1e3, fl_d / t_d / 1e9, t_c * 1e3, t_f,
                 (fl_g + fl_d) / t_f / 1e9, m as f64 / t_f * 1e3, (E * (b13 + b2)) as f64 / t_f / 1e6);
    }
    Ok(ok)
}

fn probe_head(dev: &Arc<CudaDevice>, pack: &Inkling2Pack, ids: &[usize]) -> Result<bool> {
    let c = &pack.cfg;
    let mut ok = true;
    let t0 = std::time::Instant::now();
    let h = HeadDev::load(dev, pack, 0, 0)?;
    let hs = HeadScratch::new(dev, &h)?;
    println!("INK2_HEAD: {} rows x {} bf16 (B32) loaded in {:.1} s", h.n, h.k, t0.elapsed().as_secs_f64());
    let w = pack.plain(&format!("{P}norm.weight"))?;
    let x = normed_rows(pack, &ids[..16], &w)?;
    let xd = dev.htod_copy(x.clone())?;
    let n = h.n;
    let lg = dev.alloc_zeros::<f32>(16 * n)?;
    // M = 1 per row, then M = 4 / 16 (bitwise)
    let mut l1 = Vec::new();
    let mut a1 = Vec::new();
    for t in 0..16 {
        head::forward(dev, &h, &hs, dptr(&xd, t * H), dptr(&lg, 0), n, 1, true)?;
        l1.push(dev.dtoh_sync_copy(&lg)?[..n].to_vec());
        a1.push((dev.dtoh_sync_copy(&hs.out_idx)?[0], dev.dtoh_sync_copy(&hs.out_val)?[0]));
    }
    for m in [4usize, 16] {
        let mut bad = 0;
        for g in 0..16 / m {
            head::forward(dev, &h, &hs, dptr(&xd, g * m * H), dptr(&lg, 0), n, m, true)?;
            let o = dev.dtoh_sync_copy(&lg)?;
            let ai = dev.dtoh_sync_copy(&hs.out_idx)?;
            for r in 0..m {
                if !bits_eq(&o[r * n..(r + 1) * n], &l1[g * m + r]) || ai[r] != a1[g * m + r].0 { bad += 1; }
            }
        }
        let bok = bad == 0;
        println!("INK2_HEAD_BINV M={m}: {}/16 rows (logits + argmax) bit-identical to M=1 -> {}", 16 - bad, if bok { "PASS" } else { "FAIL" });
        ok &= bok;
    }
    // reference (4 rows): fp32 unembed, f64 accumulation, softcap 30
    let nr = 4usize;
    let un = pack.st.plain_f32_range(&format!("{P}unembed.weight"), 0, c.unpadded_vocab * H)?;
    let rl = matmul(&x[..nr * H], nr, &un, c.unpadded_vocab, H);
    drop(un);
    let (mut worst, mut amx, mut argok) = (0f64, 0f32, 0usize);
    for t in 0..nr {
        let r: Vec<f32> = rl[t * n..(t + 1) * n].iter().map(|&v| softcap(v, c.logit_softcap)).collect();
        worst = worst.max(rel_l2(&l1[t], &r));
        for (a, b) in l1[t].iter().zip(&r) { amx = amx.max((a - b).abs()); }
        let (mut bi, mut bv) = (0usize, f32::MIN);
        for (i, &v) in r.iter().enumerate() { if v > bv { bv = v; bi = i; } }
        if a1[t].0 as usize == bi { argok += 1; }
    }
    let hok = worst < 1e-5 && argok == nr;
    println!("INK2_HEAD_ACC vs cpu.rs ({nr} rows x {n}): logits rel-L2 worst {worst:.2e}, max|d| {amx:.2e} (tol 1e-5: x split into \
two bf16 halves, |x - hi - lo| <= 2^-17|x|), argmax {argok}/{nr} identical -> {}", if hok { "PASS" } else { "FAIL" });
    ok &= hok;
    for m in [1usize, 4, 8, 16] {
        let t = time_ms(dev, 2, 10, &mut || head::forward(dev, &h, &hs, dptr(&xd, 0), dptr(&lg, 0), n, m, true))?;
        let gbs = h.bytes() as f64 / t / 1e6;
        println!("INK2_HEAD_TIME M={m:>2}: {:.0} us (incl. argmax) = {gbs:.0} GB/s ({:.0}% of {ROOF:.0})", t * 1e3, 100.0 * gbs / ROOF);
    }
    Ok(ok)
}

pub fn probe_moe(dir: &Path) -> Result<()> {
    let pack = Inkling2Pack::open(dir)?;
    let dev = CudaDevice::new(0)?;
    load_module(&dev)?;
    let ids = token_ids(pack.cfg.unpadded_vocab);
    let mut ok = true;
    let nv = (0..pack.cfg.layers).find(|&l| pack.exp_fmt[l] == ExpFmt::Nvfp4);
    let mx = (0..pack.cfg.layers).find(|&l| pack.exp_fmt[l] == ExpFmt::Mxfp8);
    for l in [nv, mx].into_iter().flatten() {
        ok &= probe_layer(&dev, &pack, l, &ids)?;
    }
    ok &= probe_head(&dev, &pack, &ids)?;
    ensure!(nv.is_some() && mx.is_some(), "the pack lacks an NVFP4 or an MXFP8 expert layer");
    if !ok { bail!("INK2_MOE_PROBE_FAIL (see the FAIL lines)"); }
    println!("INK2_MOE_PROBE_OK");
    Ok(())
}
