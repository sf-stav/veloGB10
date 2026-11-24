//! `--probe-inkling2-kernels --model-dir <pack> [--hist N] [--long-hist N] [--reps N]`: every K1 kernel
//! against the cpu.rs oracle on REAL layer tensors with random-but-realistic activations (bf16-valued unit-RMS
//! linear inputs; the projection outputs the GPU computed from them feed both sides of the downstream gates,
//! so each gate isolates one kernel). Per kernel: rel-L2 vs the oracle (bound 1e-3), batch invariance (every
//! row of an M-row call, M = 1..16, bit-identical to the same row of the M = 16 call: in particular row 0 ==
//! the M = 1 call), back-to-back time at M = 1, 4, 8, 16 and a 2,048-row prefill, and achieved GB/s
//! (bytes the kernel must move / time) against the 238 GB/s measured roofline (AGENTS §6).
//!
//! Prints `INK2_K <kernel> ...` lines and `INK2_K_OK` / `INK2_K_FAIL` at the end.

use super::k1::*;
use super::{dp, Ink2Kernels};
use crate::inkling2::cpu::{self, CpuModel};
use crate::inkling2::pack::P;
use anyhow::{bail, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice};
use std::path::Path;

const TOL: f64 = 1e-3;
const ROOF: f64 = 238.0;

struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 >> 11) as f64 / (1u64 << 53) as f64 }
    fn g(&mut self) -> f32 { let (a, b) = (self.u().max(1e-300), self.u()); ((-2.0 * a.ln()).sqrt() * (6.283185307179586 * b).cos()) as f32 }
    fn vec(&mut self, n: usize, s: f32) -> Vec<f32> { (0..n).map(|_| self.g() * s).collect() }
    fn vec_bf(&mut self, n: usize, s: f32) -> Vec<f32> { (0..n).map(|_| bf16_to_f32(f32_to_bf16(self.g() * s))).collect() }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut n, mut d) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) { n += (*x as f64 - *y as f64).powi(2); d += (*y as f64).powi(2); }
    (n / d.max(1e-300)).sqrt()
}
fn bits_eq(a: &[f32], b: &[f32]) -> bool { a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) }
fn bf(v: &[u16]) -> Vec<f32> { v.iter().map(|&b| bf16_to_f32(b)).collect() }
fn to_bf(v: &[f32]) -> Vec<u16> { v.iter().map(|&x| f32_to_bf16(x)).collect() }
/// columns [c0, c0+w) of rows [r0, r1) of a row-major [*][ld] matrix
fn cols(y: &[f32], ld: usize, r0: usize, r1: usize, c0: usize, w: usize) -> Vec<f32> {
    let mut o = Vec::with_capacity((r1 - r0) * w);
    for r in r0..r1 { o.extend_from_slice(&y[r * ld + c0..r * ld + c0 + w]); }
    o
}

struct Ctx<'a> { k: &'a Ink2Kernels, dev: std::sync::Arc<CudaDevice>, reps: usize, fails: usize }
impl<'a> Ctx<'a> {
    fn up(&self, v: &[f32]) -> Result<CudaSlice<f32>> { Ok(self.dev.htod_sync_copy(v)?) }
    fn up16(&self, v: &[u16]) -> Result<CudaSlice<u16>> { Ok(self.dev.htod_sync_copy(v)?) }
    fn zf(&self, n: usize) -> Result<CudaSlice<f32>> { Ok(self.dev.htod_sync_copy(&vec![0f32; n.max(1)])?) }
    fn z16(&self, n: usize) -> Result<CudaSlice<u16>> { Ok(self.dev.htod_sync_copy(&vec![0u16; n.max(1)])?) }
    fn down(&self, s: &CudaSlice<f32>) -> Result<Vec<f32>> { self.k.sync()?; Ok(self.dev.dtoh_sync_copy(s)?) }
    fn down16(&self, s: &CudaSlice<u16>) -> Result<Vec<u16>> { self.k.sync()?; Ok(self.dev.dtoh_sync_copy(s)?) }
    /// GPU time per call (µs): `reps` calls captured into a CUDA graph and replayed (Ink2Kernels::graph_time).
    fn time(&self, f: impl FnMut() -> Result<()>) -> Result<f64> { self.k.graph_time(self.reps, f) }
    fn check(&mut self, name: &str, err: f64, inv: Option<bool>) {
        let ok = err <= TOL && inv != Some(false);
        if !ok { self.fails += 1; }
        println!("INK2_K {name:<34} rel-L2 {err:.2e} (<= {TOL:.0e}) invariance {} {}", match inv { None => "-", Some(true) => "M=1..16 bit-identical", Some(false) => "BROKEN" },
                 if ok { "PASS" } else { "FAIL" });
    }
    fn timing(&self, name: &str, cells: &[(String, f64, f64)]) {
        let s: Vec<String> = cells.iter().map(|(l, us, b)| if *b > 0.0 { format!("{l} {us:.1}us {:.0}GB/s ({:.0}%)", b / us / 1e3, b / us / 1e3 / ROOF * 100.0) } else { format!("{l} {us:.1}us") }).collect();
        println!("INK2_T {name:<34} {}", s.join(" | "));
    }
}

pub fn probe(dir: &Path, hist: usize, long_hist: usize, reps: usize, only: &str) -> Result<()> {
    let on = |sec: &str| only.is_empty() || only.split(',').any(|x| x == sec);
    let t0 = std::time::Instant::now();
    let m = CpuModel::load(dir)?;
    let cfg = m.cfg.clone();
    println!("INK2_K loaded oracle in {:.1}s", t0.elapsed().as_secs_f64());
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    let k = Ink2Kernels::global(&dev)?;
    let mut c = Ctx { k: &k, dev: dev.clone(), reps, fails: 0 };
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let pk = m.pack();
    let g = |n: &str| -> Result<Vec<f32>> { pk.plain(&format!("{P}{n}")) };
    let d = cfg.hidden;

    // ------------------------------------------------------------------------------------------
    // 1. projection GEMM (fused attention projections of a sliding and a full layer, wo, MessageGate down)
    // ------------------------------------------------------------------------------------------
    for &l in if on("proj") { &[0usize, 5][..] } else { &[][..] } {
        let p = format!("layers.{l}.attn.");
        let full = !cfg.is_sliding[l];
        let mut segs: Vec<Vec<f32>> = vec![g(&format!("{p}wq_du.weight"))?, g(&format!("{p}gate_attn.weight"))?, g(&format!("{p}wr_du.weight"))?,
                                           g(&format!("{p}attn_sink_proj.weight"))?, g(&format!("{p}wk_dv.weight"))?];
        if full { segs.push(g(&format!("{p}k_sconv.weight_proj"))?); }
        segs.push(g(&format!("{p}wv_dv.weight"))?);
        if full { segs.push(g(&format!("{p}v_sconv.weight_proj"))?); }
        let w = ProjW::new(&k, &segs.iter().map(|v| v.as_slice()).collect::<Vec<_>>(), d)?;
        let (t1, t2) = if full { (w.offs[4] / 16, w.offs[6] / 16) } else { (1 << 30, 1 << 30) };
        let seg_x = |si: usize| if !full { 0 } else if si < 4 { 0 } else if si < 6 { 1 } else { 2 };
        let xs: Vec<Vec<f32>> = (0..3).map(|_| rng.vec_bf(16 * d, 1.0)).collect();
        let xd: Vec<CudaSlice<u16>> = xs.iter().map(|x| c.up16(&to_bf(x))).collect::<Result<_>>()?;
        let ldy = w.n_pad;
        let y = c.zf(16 * ldy)?;
        let name = format!("proj L{l} {} N={} K={d}", if full { "full 3-seg" } else { "sliding" }, w.n_pad);
        let mut outs: Vec<Vec<f32>> = Vec::new();
        for mm in 1..=16 {
            k.proj(&w, dp(&xd[0]), dp(&xd[1]), dp(&xd[2]), t1, t2, dp(&y), ldy, mm, 0)?;
            outs.push(c.down(&y)?[..mm * ldy].to_vec());
        }
        let inv = (0..16).all(|i| bits_eq(&outs[i], &outs[15][..(i + 1) * ldy]));
        let mut err = 0f64;
        for (si, s) in segs.iter().enumerate() {
            let n = s.len() / d;
            let r = cpu::matmul(&xs[seg_x(si)], 16, s, n, d);
            err = err.max(rel_l2(&cols(&outs[15], ldy, 0, 16, w.offs[si], n), &r));
        }
        c.check(&name, err, Some(inv));
        let mut cells = Vec::new();
        for &mm in &[1usize, 4, 8, 16] {
            for ks in [1usize, 2, 4, 8] {
                if (d / 32) % ks != 0 { continue; }
                let us = c.time(|| k.proj(&w, dp(&xd[0]), dp(&xd[1]), dp(&xd[2]), t1, t2, dp(&y), ldy, mm, ks))?;
                cells.push((format!("M{mm}/ks{ks}"), us, w.bytes() as f64));
            }
        }
        c.timing(&name, &cells);
        // prefill
        let mp = 2048;
        let xw: Vec<Vec<f32>> = (0..3).map(|_| rng.vec_bf(mp * d, 1.0)).collect();
        let xwd: Vec<CudaSlice<u16>> = xw.iter().map(|x| c.up16(&to_bf(x))).collect::<Result<_>>()?;
        let yw = c.zf(mp * ldy)?;
        k.proj(&w, dp(&xwd[0]), dp(&xwd[1]), dp(&xwd[2]), t1, t2, dp(&yw), ldy, mp, 0)?;
        let ywh = c.down(&yw)?;
        let mut errw = 0f64;
        let rows: Vec<usize> = (0..mp).step_by(67).collect();
        for (si, s) in segs.iter().enumerate() {
            let n = s.len() / d;
            let xr: Vec<f32> = rows.iter().flat_map(|&r| xw[seg_x(si)][r * d..(r + 1) * d].to_vec()).collect();
            let rf = cpu::matmul(&xr, rows.len(), s, n, d);
            let gpu: Vec<f32> = rows.iter().flat_map(|&r| ywh[r * ldy + w.offs[si]..r * ldy + w.offs[si] + n].to_vec()).collect();
            errw = errw.max(rel_l2(&gpu, &rf));
        }
        c.check(&format!("proj_wide L{l} M=2048 ({} rows sampled)", rows.len()), errw, None);
        let us = c.time(|| k.proj(&w, dp(&xwd[0]), dp(&xwd[1]), dp(&xwd[2]), t1, t2, dp(&yw), ldy, mp, 0))?;
        let tf = 2.0 * mp as f64 * w.n_pad as f64 * d as f64 / us / 1e6;
        println!("INK2_T {:<34} M2048 {us:.0}us = {:.0} tok/s per call, {tf:.1} TFLOP/s", format!("proj_wide L{l}"), mp as f64 / us * 1e6);
    }
    // wo [2304][3072] and the MessageGate down [64][4608]
    for (name, wn, n, kk) in if !on("proj") { vec![] } else { vec![("proj wo L0 N=2304 K=3072", "layers.0.attn.wo_ud.weight", d, 3072usize), ("proj ctrl-down L6 N=64 K=4608", "layers.6.controllers.attn.down_weight", 64, 2 * d)] } {
        let wv = g(wn)?;
        let w = ProjW::new(&k, &[&wv], kk)?;
        let x = rng.vec_bf(16 * kk, 1.0);
        let xd = c.up16(&to_bf(&x))?;
        let y = c.zf(16 * n)?;
        let mut outs = Vec::new();
        for mm in 1..=16 { k.proj(&w, dp(&xd), dp(&xd), dp(&xd), 1 << 30, 1 << 30, dp(&y), n, mm, 0)?; outs.push(c.down(&y)?[..mm * n].to_vec()); }
        let inv = (0..16).all(|i| bits_eq(&outs[i], &outs[15][..(i + 1) * n]));
        c.check(name, rel_l2(&outs[15], &cpu::matmul(&x, 16, &wv, n, kk)), Some(inv));
        // weights smaller than L2 would be timed L2-resident back to back: rotate over copies (>= 96 MB)
        let ncopy = (96usize << 20).div_ceil(w.bytes()).min(64);
        let copies: Vec<ProjW> = (0..ncopy).map(|_| ProjW::new(&k, &[&wv], kk)).collect::<Result<_>>()?;
        let mut cells = Vec::new();
        for &mm in &[1usize, 4, 8, 16] {
            for ks in [1usize, 2, 4, 8] {
                if (kk / 32) % ks != 0 { continue; }
                let mut it = 0usize;
                let us = c.time(|| { it += 1; k.proj(&copies[it % ncopy], dp(&xd), dp(&xd), dp(&xd), 1 << 30, 1 << 30, dp(&y), n, mm, ks) })?;
                cells.push((format!("M{mm}/ks{ks}"), us, w.bytes() as f64));
            }
        }
        c.timing(name, &cells);
    }

    // ------------------------------------------------------------------------------------------
    // 2. attention: prep (Q/R clip + log scale, K/V conv + clip/RMS + cache) + attention (+ combine)
    // ------------------------------------------------------------------------------------------
    for &(l, h, cap_swa) in &[(0usize, hist, 0usize), (5, hist, 0), (0, long_hist, 4096), (5, long_hist, 0)] {
        if h == 0 || !on("attn") { continue; }
        attn_case(&mut c, &m, &mut rng, l, h, cap_swa)?;
    }

    // ------------------------------------------------------------------------------------------
    // 3. static short conv + residual (attn_sconv of layer 0), prefill ring path + decode
    // ------------------------------------------------------------------------------------------
    if on("conv") {
        let wc = g("layers.0.attn_sconv.weight")?; // [C][1][4], tap 3 = current
        let mut wl = vec![0f32; 4 * d];
        for ch in 0..d { for lag in 0..4 { wl[lag * d + ch] = wc[ch * 4 + 3 - lag]; } }
        let wld = c.up(&wl)?;
        let h0 = 40usize;
        let xs = rng.vec(( h0 + 16) * d, 1.0);
        let base = rng.vec((h0 + 16) * d, 1.0);
        let ring = c.zf(RING * d)?;
        let xd = c.up(&xs)?;
        let bd = c.up(&base)?;
        let out = c.zf((h0 + 16) * d)?;
        k.sconv(dp(&xd), d, dp(&ring), dp(&wld), dp(&bd), dp(&out), d, 0, h0)?; // M = 40 > R - 3: ring_store path
        let pre = c.down(&out)?[..h0 * d].to_vec();
        let rpre: Vec<f32> = CpuModel::conv_static(&wc, &xs[..h0 * d], d, 4, h0).iter().zip(&base).map(|(a, b)| a + b).collect();
        let e_pre = rel_l2(&pre, &rpre);
        let mut outs = Vec::new();
        let xdec = c.up(&xs[h0 * d..])?;
        let bdec = c.up(&base[h0 * d..])?;
        let odec = c.zf(16 * d)?;
        for mm in 1..=16 { k.sconv(dp(&xdec), d, dp(&ring), dp(&wld), dp(&bdec), dp(&odec), d, h0, mm)?; outs.push(c.down(&odec)?[..mm * d].to_vec()); }
        let inv = (0..16).all(|i| bits_eq(&outs[i], &outs[15][..(i + 1) * d]));
        let rdec: Vec<f32> = CpuModel::conv_static(&wc, &xs, d, 4, 16).iter().zip(&base[h0 * d..]).map(|(a, b)| a + b).collect();
        c.check("sconv prefill M=40 (ring_store)", e_pre, None);
        c.check("sconv decode p0=40 (ring)", rel_l2(&outs[15], &rdec), Some(inv));
        let mut cells = Vec::new();
        for &mm in &[1usize, 4, 8, 16] {
            let us = c.time(|| k.sconv(dp(&xdec), d, dp(&ring), dp(&wld), dp(&bdec), dp(&odec), d, h0, mm))?;
            cells.push((format!("M{mm}"), us, (mm * d * 12 + 4 * d * 4 + 4 * d * 4) as f64));
        }
        let xb = c.zf(2048 * d)?; let ob = c.zf(2048 * d)?;
        let us = c.time(|| k.sconv(dp(&xb), d, dp(&ring), dp(&wld), dp(&xb), dp(&ob), d, 64, 2048))?;
        cells.push(("M2048".into(), us, (2048 * d * 12) as f64));
        c.timing("sconv C=2304", &cells);
    }

    // ------------------------------------------------------------------------------------------
    // 4. AttnRes (1, 3 and 4 queries; 1..6 values)
    // ------------------------------------------------------------------------------------------
    if on("attnres") {
        let qsets: Vec<(String, Vec<f32>)> = vec![
            ("mlp_q L3 (1 q)".into(), m.mlp_queries(3).to_vec()),
            ("attn_q L5 (3 q)".into(), m.attn_queries(5).to_vec()),
            ("msg_q k0 (4 q)".into(), m.msg_queries(0)),
        ];
        let vals: Vec<Vec<f32>> = (0..6).map(|j| rng.vec(16 * d, [0.7f32, 2.0, 1.0, 4.0, 0.5, 1.5][j])).collect();
        let vd: Vec<CudaSlice<f32>> = vals.iter().map(|v| c.up(v)).collect::<Result<_>>()?;
        let outs_d: Vec<CudaSlice<f32>> = (0..4).map(|_| c.zf(2048 * d)).collect::<Result<_>>()?;
        for (qn, q) in &qsets {
            let nq = q.len() / d;
            let qd = c.up(q)?;
            for nv in [1usize, 3, 6] {
                let blocks: Vec<Vec<f32>> = vals[..nv - 1].to_vec();
                let r = m.attn_res(&blocks, &vals[nv - 1], q, 16);
                let mut got: Vec<Vec<Vec<f32>>> = Vec::new();
                let vp: Vec<u64> = vd[..nv].iter().map(dp).collect();
                let op: Vec<u64> = outs_d[..nq].iter().map(dp).collect();
                for mm in 1..=16 {
                    k.attnres(&vp, dp(&qd), nq, &op, d, mm)?;
                    got.push(outs_d[..nq].iter().map(|o| c.down(o).map(|v| v[..mm * d].to_vec())).collect::<Result<_>>()?);
                }
                let inv = (0..16).all(|i| (0..nq).all(|qi| bits_eq(&got[i][qi], &got[15][qi][..(i + 1) * d])));
                let err = (0..nq).map(|qi| rel_l2(&got[15][qi], &r[qi])).fold(0f64, f64::max);
                c.check(&format!("attnres {qn} nv={nv}"), err, Some(inv));
            }
            let vp: Vec<u64> = vd.iter().map(dp).collect();
            let op: Vec<u64> = outs_d[..nq].iter().map(dp).collect();
            let mut cells = Vec::new();
            for &mm in &[1usize, 4, 8, 16] {
                let us = c.time(|| k.attnres(&vp, dp(&qd), nq, &op, d, mm))?;
                cells.push((format!("M{mm}"), us, ((2 * 6 + nq) * mm * d * 4) as f64));
            }
            let big: Vec<CudaSlice<f32>> = (0..6).map(|_| c.zf(2048 * d)).collect::<Result<_>>()?;
            let bp: Vec<u64> = big.iter().map(dp).collect();
            let us = c.time(|| k.attnres(&bp, dp(&qd), nq, &op, d, 2048))?;
            cells.push(("M2048".into(), us, ((2 * 6 + nq) * 2048 * d * 4) as f64));
            c.timing(&format!("attnres {qn} nv=6"), &cells);
        }
    }

    // ------------------------------------------------------------------------------------------
    // 5. MessageGate (layer 6 controllers, all four)
    // ------------------------------------------------------------------------------------------
    if on("msg") {
        for (gi, gn) in ["attn", "k", "v", "mlp"].iter().enumerate() {
            let gt = m.gate(6, gi).context("layer 6 gates")?;
            let down = ProjW::new(&k, &[&gt.down], 2 * d)?;
            let wf = c.up16(&gt.forget.iter().map(|&x| bf16_bits(x)).collect::<Vec<_>>())?;
            let wg = c.up16(&gt.gain.iter().map(|&x| bf16_bits(x)).collect::<Vec<_>>())?;
            let hh = rng.vec(16 * d, 1.0);
            let mg = rng.vec(16 * d, 1.0);
            let r = m.message_gate(gt, &hh, &mg, 16);
            let (hd_, md) = (c.up(&hh)?, c.up(&mg)?);
            let cat = c.z16(2048 * 2 * d)?;
            let z = c.zf(2048 * 64 * MSG_ZS)?;
            let out = c.zf(2048 * d)?;
            let mut got = Vec::new();
            for mm in 1..=16 {
                k.msggate(&down, dp(&wf), dp(&wg), dp(&hd_), dp(&md), dp(&cat), dp(&z), dp(&out), d, mm, cfg.eps)?;
                got.push(c.down(&out)?[..mm * d].to_vec());
            }
            let inv = (0..16).all(|i| bits_eq(&got[i], &got[15][..(i + 1) * d]));
            // D9: the bf16 reference feeds `down` a bf16 cast of the concat; the GPU path does the same. Gate vs the
            // oracle with that cast (cpu.rs helpers), the pure-f32 oracle error as info.
            let rb = {
                let (hn, mn) = (cpu::rms_rows(&hh, d, cfg.eps), cpu::rms_rows(&mg, d, cfg.eps));
                let mut cat = Vec::with_capacity(16 * 2 * d);
                for t in 0..16 { cat.extend_from_slice(&hn[t * d..(t + 1) * d]); cat.extend_from_slice(&mn[t * d..(t + 1) * d]); }
                let cat: Vec<f32> = cat.iter().map(|&x| bf16_to_f32(f32_to_bf16(x))).collect();
                let zz = cpu::matmul(&cat, 16, &gt.down, 64, 2 * d);
                let (f, gg) = (cpu::matmul(&zz, 16, &gt.forget, d, 64), cpu::matmul(&zz, 16, &gt.gain, d, 64));
                (0..16 * d).map(|i| (1.0 - f[i].tanh()) * hh[i] + gg[i].tanh() * mg[i]).collect::<Vec<f32>>()
            };
            c.check(&format!("msggate L6.{gn} (bf16 cat, D9)"), rel_l2(&got[15], &rb), Some(inv));
            println!("INK2_K {:<34} vs the pure-f32 oracle {:.2e} (the D9 bf16 cast, info)", format!("msggate L6.{gn}"), rel_l2(&got[15], &r));
            if gi == 0 {
                let mut cells = Vec::new();
                let wbytes = (down.bytes() + 2 * d * 64 * 2) as f64;
                for &mm in &[1usize, 4, 8, 16] {
                    let us = c.time(|| k.msggate(&down, dp(&wf), dp(&wg), dp(&hd_), dp(&md), dp(&cat), dp(&z), dp(&out), d, mm, cfg.eps))?;
                    cells.push((format!("M{mm}"), us, wbytes));
                }
                let hb = c.zf(2048 * d)?;
                let us = c.time(|| k.msggate(&down, dp(&wf), dp(&wg), dp(&hb), dp(&hb), dp(&cat), dp(&z), dp(&out), d, 2048, cfg.eps))?;
                cells.push(("M2048".into(), us, wbytes + (2048 * d * 4 * 5) as f64));
                c.timing("msggate (3 launches)", &cells);
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // 6. norms + embed
    // ------------------------------------------------------------------------------------------
    if on("norm") {
        let wn = g("layers.0.attn_norm.weight")?;
        let wnd = c.up(&wn)?;
        let x: Vec<f32> = (0..16).flat_map(|r| { let s = 0.3 + r as f32 * 0.4; rng.vec(d, s) }).collect();
        let xd = c.up(&x)?;
        let (ob, of) = (c.z16(2048 * d)?, c.zf(2048 * d)?);
        for (name, w, rf) in [("rmsnorm weighted (attn_norm L0)", dp(&wnd), cpu::rmsw_rows(&x, &wn, cfg.eps)), ("rmsnorm unweighted", 0u64, cpu::rms_rows(&x, d, cfg.eps))] {
            let mut got = Vec::new();
            for mm in 1..=16 { k.rmsnorm(dp(&xd), w, dp(&ob), dp(&of), d, mm, cfg.eps)?; got.push(c.down(&of)?[..mm * d].to_vec()); }
            let inv = (0..16).all(|i| bits_eq(&got[i], &got[15][..(i + 1) * d]));
            c.check(name, rel_l2(&got[15], &rf), Some(inv));
            let gb = bf(&c.down16(&ob)?[..16 * d]);
            println!("INK2_K {:<34} bf16 output vs oracle rel-L2 {:.2e} (bf16 rounding, info)", name, rel_l2(&gb, &rf));
        }
        let mut cells = Vec::new();
        for &mm in &[1usize, 4, 8, 16] {
            let us = c.time(|| k.rmsnorm(dp(&xd), dp(&wnd), dp(&ob), 0, d, mm, cfg.eps))?;
            cells.push((format!("M{mm}"), us, (mm * d * 6) as f64));
        }
        let xb = c.zf(2048 * d)?;
        let us = c.time(|| k.rmsnorm(dp(&xb), dp(&wnd), dp(&ob), 0, d, 2048, cfg.eps))?;
        cells.push(("M2048".into(), us, (2048 * d * 6) as f64));
        c.timing("rmsnorm -> bf16", &cells);
        // embed + embed_norm on a sub-table of the real rows (ids remapped)
        let ids: Vec<u32> = vec![0, 1, 13, 371, 152717, 200001, 200006, 150409, 68014, 32, 5, 99999, 123456, 200057, 7, 1000];
        let mut sub = Vec::with_capacity(ids.len() * d);
        for &t in &ids { sub.extend(pk.st.plain_f32_range(&format!("{P}embed.weight"), t as usize * d, (t as usize + 1) * d)?); }
        let subd = c.up16(&sub.iter().map(|&x| bf16_bits(x)).collect::<Vec<_>>())?;
        let en = c.up(&g("embed_norm.weight")?)?;
        let idd = dev.htod_sync_copy(&(0..ids.len() as i32).collect::<Vec<i32>>())?;
        let out = c.zf(16 * d)?;
        let mut got = Vec::new();
        for mm in 1..=16 { k.embed_norm(dp(&idd), dp(&subd), dp(&en), dp(&out), d, mm, cfg.eps)?; got.push(c.down(&out)?[..mm * d].to_vec()); }
        let inv = (0..16).all(|i| bits_eq(&got[i], &got[15][..(i + 1) * d]));
        c.check("embed + embed_norm (16 real ids)", rel_l2(&got[15], &m.embed(&ids)?), Some(inv));
    }

    if c.fails > 0 { bail!("INK2_K_FAIL: {} gate(s) failed", c.fails); }
    println!("INK2_K_OK every K1 kernel gate passed ({:.0}s)", t0.elapsed().as_secs_f64());
    Ok(())
}

/// Attention for layer `l` with `h` history positions (prefilled through the GPU prefill path in chunks of
/// 2,048) then decode / verify calls of M = 1..16 rows at p0 = h. `cap_swa` > 0: the sliding layer's cache is a
/// ring of that many rows (exercises wrap-around). The oracle is cpu.rs `attn_qr` / `kv_append` / `attend` fed
/// the GPU's own projection outputs.
fn attn_case(c: &mut Ctx, m: &CpuModel, rng: &mut Rng, l: usize, h: usize, cap_swa: usize) -> Result<()> {
    let k = c.k;
    let cfg = &m.cfg;
    let d = cfg.hidden;
    let pk = m.pack();
    let g = |n: &str| -> Result<Vec<f32>> { pk.plain(&format!("{P}{n}")) };
    let p = format!("layers.{l}.attn.");
    let full = !cfg.is_sliding[l];
    let mut segs: Vec<Vec<f32>> = vec![g(&format!("{p}wq_du.weight"))?, g(&format!("{p}gate_attn.weight"))?, g(&format!("{p}wr_du.weight"))?,
                                       g(&format!("{p}attn_sink_proj.weight"))?, g(&format!("{p}wk_dv.weight"))?];
    if full { segs.push(g(&format!("{p}k_sconv.weight_proj"))?); }
    segs.push(g(&format!("{p}wv_dv.weight"))?);
    if full { segs.push(g(&format!("{p}v_sconv.weight_proj"))?); }
    let w = ProjW::new(k, &segs.iter().map(|v| v.as_slice()).collect::<Vec<_>>(), d)?;
    let o = &w.offs;
    let (oq, og, or_, os, okk) = (o[0], o[1], o[2], o[3], o[4]);
    let (okd, ov, ovd) = if full { (o[5] as i32, o[6], o[7] as i32) } else { (-1, o[5], -1) };
    let (t1, t2) = if full { (o[4] / 16, o[6] / 16) } else { (1 << 30, 1 << 30) };
    let ldy = w.n_pad;
    let n = h + 16;
    // inputs (sliding: one stream; full: q / k / v streams)
    let xs: Vec<Vec<f32>> = (0..if full { 3 } else { 1 }).map(|_| rng.vec_bf(n * d, 1.0)).collect();
    let xd: Vec<CudaSlice<u16>> = xs.iter().map(|x| c.up16(&to_bf(x))).collect::<Result<_>>()?;
    let xi = |i: usize| if full { i } else { 0 };
    let y = c.zf(n * ldy)?;
    k.proj(&w, dp(&xd[0]), dp(&xd[xi(1)]), dp(&xd[xi(2)]), t1, t2, dp(&y), ldy, n, 0)?;
    let yh = c.down(&y)?;
    // conv taps [4 lag][1024]
    let conv_w = |s: &str| -> Result<Vec<f32>> {
        if full { g(&format!("{p}{s}_sconv.static_w")) } else {
            let wc = g(&format!("{p}{s}_sconv.weight"))?;
            let mut o = vec![0f32; 4 * 1024];
            for ch in 0..1024 { for lag in 0..4 { o[lag * 1024 + ch] = wc[ch * 4 + 3 - lag]; } }
            Ok(o)
        }
    };
    let (cwk, cwv) = (c.up(&conv_w("k")?)?, c.up(&conv_w("v")?)?);
    let (qs, rs) = (c.up(&g(&format!("{p}q_l2_clip.scale"))?)?, c.up(&g(&format!("{p}rel_l2_clip.scale"))?)?);
    let ext = cfg.rel_ext(l);
    let proj = c.up(&g(&format!("{p}rel_logits_proj.proj"))?)?;
    let cap = if !full && cap_swa > 0 { cap_swa } else { n.next_power_of_two() };
    let (kc, vc) = (dev_zero16(c, cap * 1024)?, dev_zero16(c, cap * 1024)?);
    let (rk, rv) = (c.zf(RING * 1024)?, c.zf(RING * 1024)?);
    let qf = c.zf(2048 * NH * HD)?;
    let rf = c.zf(2048 * NH * DREL)?;
    let out = c.z16(2048 * NH * HD)?;
    let maxsplit = attn_split(h, 16, if full { 0 } else { cfg.sliding_window }).nsplit + 2;
    let (pm, pl, pacc) = (c.zf(16 * NH * maxsplit)?, c.zf(16 * NH * maxsplit)?, c.zf(16 * NH * maxsplit * HD)?);
    let prep = |p0: usize, mm: usize, rw: bool| PrepArgs {
        y: dp(&y) + (p0 * ldy * 4) as u64, ldy: ldy as i32, oq: oq as i32, orr: or_ as i32, ok: okk as i32, ov: ov as i32, okd, ovd,
        q_scale: dp(&qs), r_scale: dp(&rs), qk_clip: cfg.qk_clip, rel_clip: cfg.rel_clip,
        full: full as i32, log_alpha: cfg.log_alpha, log_floor: cfg.log_floor,
        cw_k: dp(&cwk), cw_v: dp(&cwv), conv_div: if full { 1.0 / 3.0 } else { 1.0 / 5f32.sqrt() },
        ring_k: dp(&rk), ring_v: dp(&rv), ring_mask: (RING - 1) as i32, ring_write: rw as i32,
        kc: dp(&kc), vc: dp(&vc), cap_mask: (cap - 1) as u32, eps: cfg.eps, qf: dp(&qf), rf: dp(&rf), p0: p0 as i32, m: mm as i32,
    };
    let att = |p0: usize, mm: usize| AttnArgs {
        qf: dp(&qf), rf: dp(&rf), kc: dp(&kc), vc: dp(&vc), cap_mask: (cap - 1) as u32, proj: dp(&proj), ext: ext as i32,
        window: if full { 0 } else { cfg.sliding_window as i32 }, p0: p0 as i32, m: mm as i32, chunk: 0, c0: 0,
        y: dp(&y) + (p0 * ldy * 4) as u64, ldy: ldy as i32, os: os as i32, og: og as i32, out: dp(&out),
        pm: dp(&pm), pl: dp(&pl), pacc: dp(&pacc),
    };
    // GPU history: prefill chunks of <= 2048 rows
    let compare_hist = h <= 2048;
    let mut hist_out: [Vec<f32>; 2] = [Vec::new(), Vec::new()];   // [CUDA-core, tensor-core]
    let mut p0 = 0;
    let mut pf_time = None;
    while p0 < h {
        let mm = (h - p0).min(2048);
        // time one full 2,048-row chunk with a full window behind it: prep is re-run BEFORE this chunk's ring store
        // (it reads the ring rows p0-3..p0-1), attention after (it only reads the cache): both idempotent there
        let timed = mm == 2048 && pf_time.is_none() && p0 >= 1024;
        k.attn_prep(&prep(p0, mm, false))?;
        let us_p = if timed { c.time(|| k.attn_prep(&prep(p0, mm, false)))? } else { 0.0 };
        k.ring_store(dp(&y) + (p0 * ldy * 4) as u64, ldy, okk, dp(&rk), 1024, p0, mm)?;
        k.ring_store(dp(&y) + (p0 * ldy * 4) as u64, ldy, ov, dp(&rv), 1024, p0, mm)?;
        for tc in [false, true] {
            k.attn(&att(p0, mm), true, tc)?;
            if compare_hist { hist_out[tc as usize] = bf(&c.down16(&out)?[..mm * NH * HD]); }
        }
        if timed {
            let us_a = c.time(|| k.attn(&att(p0, mm), true, false).map(|_| ()))?;
            let us_t = c.time(|| k.attn(&att(p0, mm), true, true).map(|_| ()))?;
            pf_time = Some((p0, us_p, us_a, us_t));
        }
        p0 += mm;
    }
    // decode / verify rows at p0 = h
    let mut gots: [Vec<Vec<f32>>; 2] = [Vec::new(), Vec::new()];
    let mut sp = None;
    for mm in 1..=16 {
        k.attn_prep(&prep(h, mm, true))?;
        for tc in [false, true] {
            sp = Some(k.attn(&att(h, mm), false, tc)?);
            gots[tc as usize].push(bf(&c.down16(&out)?[..mm * NH * HD]));
        }
    }
    // oracle: Q / R and the K / V cache of all n rows
    let col = |r0: usize, r1: usize, off: usize, wd: usize| cols(&yh, ldy, r0, r1, off, wd);
    let slot = cfg.cache_slot(l, 0);
    let mut st = m.new_state();
    let (qh, rh) = m.attn_qr(l, col(0, h, oq, NH * HD), col(0, h, or_, NH * DREL), 0, h);
    m.kv_append(l, &mut st, slot, &col(0, h, okk, 1024), &col(0, h, ov, 1024), &xs[xi(1)][..h * d], &xs[xi(2)][..h * d], h);
    let (q16, r16) = m.attn_qr(l, col(h, n, oq, NH * HD), col(h, n, or_, NH * DREL), h, 16);
    m.kv_append(l, &mut st, slot, &col(h, n, okk, 1024), &col(h, n, ov, 1024), &xs[xi(1)][h * d..], &xs[xi(2)][h * d..], 16);
    let tag = format!("L{l} {} hist={h}{}", if full { "full" } else { "swa" }, if cap_swa > 0 && !full { format!(" ring cap {cap_swa}") } else { String::new() });
    // prep gate (M = 16 rows): Q, R and the f16 K / V cache rows the call wrote
    let qg = c.down(&qf)?[..16 * NH * HD].to_vec();
    let rg = c.down(&rf)?[..16 * NH * DREL].to_vec();
    let kcache: Vec<f32> = c.dev.dtoh_sync_copy(&kc)?.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
    let vcache: Vec<f32> = c.dev.dtoh_sync_copy(&vc)?.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
    let rows_of = |cache: &[f32], r0: usize, r1: usize| -> Vec<f32> { (r0..r1).flat_map(|pp| cache[(pp & (cap - 1)) * 1024..(pp & (cap - 1)) * 1024 + 1024].to_vec()).collect() };
    let e_prep = rel_l2(&qg, &q16).max(rel_l2(&rg, &r16)).max(rel_l2(&rows_of(&kcache, h, n), &st.k[slot][h * 1024..]))
        .max(rel_l2(&rows_of(&vcache, h, n), &st.v[slot][h * 1024..]));
    let lo = n.saturating_sub(cap); // rows still resident in a ring cache
    let e_kv_all = rel_l2(&rows_of(&kcache, lo, n), &st.k[slot][lo * 1024..]).max(rel_l2(&rows_of(&vcache, lo, n), &st.v[slot][lo * 1024..]));
    c.check(&format!("attn_prep {tag} (Q,R,K,V)"), e_prep, None);
    println!("INK2_K {:<34} every resident cache row ({lo}..{n}) vs oracle K/V rel-L2 {e_kv_all:.2e} (f16 storage, info)", format!("attn_prep {tag}"));
    // the attention KERNEL gate: the oracle attends over the GPU's own f16 cache (storage precision is a
    // separate, reported choice); plus the f32-cache and bf16-cache (the bf16 reference's dtype) errors as info
    let mut st16 = m.new_state();
    st16.k[slot] = st.k[slot].clone();
    st16.v[slot] = st.v[slot].clone();
    st16.k[slot][lo * 1024..].copy_from_slice(&rows_of(&kcache, lo, n));
    st16.v[slot][lo * 1024..].copy_from_slice(&rows_of(&vcache, lo, n));
    let bfr = |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| bf16_to_f32(f32_to_bf16(x))).collect() };
    let mut stb = m.new_state();
    stb.k[slot] = bfr(&st.k[slot]);
    stb.v[slot] = bfr(&st.v[slot]);
    if compare_hist {
        let r16_ = m.attend(l, &st16, slot, &qh, &rh, &col(0, h, os, NH), &col(0, h, og, NH * HD), 0, h);
        let r32 = m.attend(l, &st, slot, &qh, &rh, &col(0, h, os, NH), &col(0, h, og, NH * HD), 0, h);
        for tc in [false, true] {
            let nm = if tc { "attn_tc" } else { "attn" };
            c.check(&format!("{nm} prefill {tag}"), rel_l2(&hist_out[tc as usize], &bfr(&r16_)), None);
            println!("INK2_K {:<34} vs the f32-cache oracle unrounded {:.2e} (bf16 output + f16 cache, info)", format!("{nm} prefill {tag}"), rel_l2(&hist_out[tc as usize], &r32));
        }
    }
    let args = (col(h, n, os, NH), col(h, n, og, NH * HD));
    let r = m.attend(l, &st16, slot, &q16, &r16, &args.0, &args.1, h, 16);
    let r32 = m.attend(l, &st, slot, &q16, &r16, &args.0, &args.1, h, 16);
    let rb = m.attend(l, &stb, slot, &q16, &r16, &args.0, &args.1, h, 16);
    let sp = sp.unwrap();
    for tc in [false, true] {
        let got = &gots[tc as usize];
        let nm = if tc { "attn_tc" } else { "attn" };
        let inv = (0..16).all(|i| bits_eq(&got[i], &got[15][..(i + 1) * NH * HD]));
        println!("INK2_K {:<34} vs the f32-cache oracle unrounded {:.2e} (bf16 output + f16 cache); cache dtype alone (oracle vs oracle): f16 {:.2e}, bf16 {:.2e} (info)",
                 format!("{nm} decode {tag}"), rel_l2(&got[15], &r32), rel_l2(&r, &r32), rel_l2(&rb, &r32));
        c.check(&format!("{nm} decode {tag} split {}x{}", sp.nsplit, sp.chunk), rel_l2(&got[15], &bfr(&r)), Some(inv));
    }
    // timing
    let mut cp = Vec::new();
    let mut ca = Vec::new();
    let mut ct = Vec::new();
    for &mm in &[1usize, 4, 8, 16] {
        let us = c.time(|| k.attn_prep(&prep(h, mm, true)))?;
        cp.push((format!("M{mm}"), us, (mm * ldy * 4 + mm * 2048 * 4) as f64));
        let span = if full { h + mm } else { (h + mm).min(cfg.sliding_window + mm) };
        let us = c.time(|| k.attn(&att(h, mm), false, false).map(|_| ()))?;
        ca.push((format!("M{mm}"), us, (span * 1024 * 2 * 2) as f64));
        let us = c.time(|| k.attn(&att(h, mm), false, true).map(|_| ()))?;
        ct.push((format!("M{mm}"), us, (span * 1024 * 2 * 2) as f64));
    }
    if let Some((pp, up, ua, ut)) = pf_time {
        cp.push((format!("M2048@{pp}"), up, (2048 * ldy * 4) as f64));
        ca.push((format!("M2048@{pp}"), ua, 0.0));
        ct.push((format!("M2048@{pp}"), ut, 0.0));
    }
    c.timing(&format!("attn_prep {tag}"), &cp);
    c.timing(&format!("attn(+combine) {tag}"), &ca);
    c.timing(&format!("attn_tc(+combine) {tag}"), &ct);
    Ok(())
}

fn dev_zero16(c: &Ctx, n: usize) -> Result<CudaSlice<u16>> { c.z16(n) }
