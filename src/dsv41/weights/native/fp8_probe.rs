//! `--probe-dsv41-kernel --dsv41-kernel-set fp8` (leg KN1): the native FP8 dense path on the GPU against a host
//! reference, through the same `dense()` entry the forward uses, on REAL weights read from the native checkpoint
//! (`--model-dir`). It checks, and asserts the success signal of each: (1) `act_quant` bitwise equal to the
//! reference formula (host re-implementation, independent e4m3 conversion); (2) dense outputs for 1..64 rows
//! within fp32-summation noise of the reference's chunked scheme (fp64 dot per 32-K block, f32 blocks in order);
//! (3) batch invariance: row 0..r of every call equals the 1-row / 64-row call bit for bit, inside the split-K
//! family (rows <= 32) and inside the GEMM family (rows > 32); (4) the (n0, n_count, k0b, nkb) slice interface and
//! the accumulate mode. Timing tables are the standalone harness' job (`tools/dsv41_kn1/fp8_harness.cu`).

use super::{dense, fp8};
use crate::dsv41::census::classify::{Role, TensorClass};
use crate::dsv41::census::LinearFormat;
use crate::dsv41::gpu::abi::{e4m3, e8m0};
use crate::dsv41::st;
use crate::dsv41::weights::{Accum, Act, ActDtype, DenseWeight, DevPtr, DevTensor, Launch, Scratch, Shard};
use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::{CudaDevice, DevicePtr};
use std::collections::BTreeMap;
use std::path::Path;

fn bf16_round(x: f32) -> f32 { let u = x.to_bits(); f32::from_bits(((u.wrapping_add(0x7FFF + ((u >> 16) & 1))) >> 16) << 16) }
fn bf16_bits(x: f32) -> u16 { (bf16_round(x).to_bits() >> 16) as u16 }
fn bf16_val(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 >> 40) as f32 / (1u64 << 24) as f32 }
    fn normal(&mut self) -> f32 { (0..12).map(|_| self.next()).sum::<f32>() - 6.0 }
}
/// Activations with heavy-tailed channels, tiny groups (< the 1e-4 amax floor) and quiet groups.
fn make_x(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    let mut x = vec![0f32; rows * k];
    for row in 0..rows { for c in 0..k {
        let mut v = r.normal() * if c % 61 == 0 { 20.0 } else { 1.0 } * if (c / 32) % 7 == 3 { 0.01 } else { 1.0 };
        if (c / 32) % 11 == 5 { v *= 1e-6; }
        x[row * k + c] = bf16_round(v);
    } }
    x
}
/// nearest E4M3 code, ties to the even code (independent of the device converter)
fn cvt_e4m3(x: f32) -> u8 {
    let sgn = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a >= 448.0 { return sgn | 126; }
    let (mut lo, mut hi) = (0usize, 126usize);
    while lo < hi { let mid = (lo + hi + 1) / 2; if e4m3(mid as u8) <= a { lo = mid } else { hi = mid - 1 } }
    let mut c = lo;
    if c < 126 { let (dl, dh) = (a - e4m3(c as u8), e4m3(c as u8 + 1) - a); if dh < dl || (dh == dl && c & 1 == 1) { c += 1; } }
    sgn | c as u8
}
fn host_act_quant(x: &[f32], rows: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let kb = k / 32;
    let (mut q, mut s) = (vec![0u8; rows * k], vec![0u8; rows * kb]);
    for r in 0..rows { for b in 0..kb {
        let g = &x[r * k + b * 32..r * k + b * 32 + 32];
        let amax = g.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-4);
        let bits = (amax * 0.0022321429569274187f32).to_bits();
        let e = ((bits >> 23) & 0xFF) + u32::from(bits & 0x7FFFFF != 0);
        let sc = f32::from_bits(e << 23);
        s[r * kb + b] = e as u8;
        for (i, v) in g.iter().enumerate() { q[r * k + b * 32 + i] = cvt_e4m3((v / sc).clamp(-448.0, 448.0)); }
    } }
    (q, s)
}
/// The reference kernel's scheme on one output: per 32-K block an fp64 dot, blocks accumulated in f32, in order.
#[allow(clippy::too_many_arguments)]
fn host_dot(w: &[u8], ws: &[u8], k_full: usize, n: usize, k0b: usize, nkb: usize, xq: &[u8], xs: &[u8]) -> f32 {
    let kbf = k_full / 32;
    let mut acc = 0f32;
    for b in 0..nkb {
        let mut d = 0f64;
        for i in 0..32 { d += e4m3(w[n * k_full + (k0b + b) * 32 + i]) as f64 * e4m3(xq[b * 32 + i]) as f64; }
        acc += d as f32 * (e8m0(ws[(n / 32) * kbf + k0b + b]) * e8m0(xs[b]));
    }
    acc
}

struct Lin { key: &'static str, class: TensorClass, n: usize, k: usize, w: Vec<u8>, ws: Vec<u8>, gw: cudarc::driver::CudaSlice<u8>, gs: cudarc::driver::CudaSlice<u8> }

fn load_lin(dev: &std::sync::Arc<CudaDevice>, dir: &Path, idx: &serde_json::Value, key: &'static str, class: TensorClass) -> Result<Lin> {
    let rd = |name: &str| -> Result<(Vec<u8>, Vec<u64>)> {
        let file = idx["weight_map"][name].as_str().with_context(|| format!("{name} not in the index"))?;
        let h = st::read_header(&dir.join(file))?;
        let shape = h.tensors.iter().find(|t| t.name == name).with_context(|| format!("{name} not in {file}"))?.shape.clone();
        Ok((st::read_tensor(&h, name)?, shape))
    };
    let (w, sh) = rd(&format!("{key}.weight"))?;
    let (ws, ssh) = rd(&format!("{key}.scale"))?;
    ensure!(sh.len() == 2 && ssh == [sh[0].div_ceil(32), sh[1] / 32], "{key}: unexpected shapes {sh:?} / {ssh:?}");
    Ok(Lin { key, class, n: sh[0] as usize, k: sh[1] as usize, gw: dev.htod_sync_copy(&w)?, gs: dev.htod_sync_copy(&ws)?, w, ws })
}
fn dense_weight(l: &Lin) -> DenseWeight {
    DenseWeight { key: l.key.to_string(), class: l.class, layer: Some(3), sub: None,
        format: LinearFormat::Fp8Block { bo: 32, bi: 32, scale: "F8_E8M0".into() }, out: l.n as u64, inn: l.k as u64, shard: Shard::Whole,
        tensors: vec![
            DevTensor { role: Role::Weight, ptr: DevPtr(*l.gw.device_ptr() as u64), dtype: "F8_E4M3".into(), shape: vec![l.n as u64, l.k as u64], bytes: (l.n * l.k) as u64 },
            DevTensor { role: Role::Scale, ptr: DevPtr(*l.gs.device_ptr() as u64), dtype: "F8_E8M0".into(), shape: vec![l.n.div_ceil(32) as u64, (l.k / 32) as u64], bytes: (l.n.div_ceil(32) * l.k / 32) as u64 },
        ] }
}

pub fn run(dir: &Path) -> Result<Vec<String>> {
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    crate::dsv41::gpu::load_module(&dev)?;
    let stream = *dev.cu_stream() as u64;
    let l = Launch { stream, pdl: false };
    let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json")).context("model.safetensors.index.json (use the NATIVE checkpoint dir)")?)?;
    let mut out = vec![format!("fp8 probe: {} entry kernels in band 20, native dir {}", fp8::FNS.len(), dir.display())];

    // (1) act_quant bitwise
    for k in [32usize, 1280, 5120] {
        let rows = 37;
        let x = make_x(rows, k, 7 + k as u64);
        let xb: Vec<u16> = x.iter().map(|v| bf16_bits(*v)).collect();
        let gx = dev.htod_sync_copy(&xb)?;
        let mut gq = dev.alloc_zeros::<u8>(rows * k)?;
        let mut gsc = dev.alloc_zeros::<u8>(rows * k / 32)?;
        fp8::quantize(&l, &Act { ptr: DevPtr(*gx.device_ptr() as u64), rows, cols: k, ld: k, dtype: ActDtype::Bf16 }, DevPtr(*gq.device_ptr() as u64), DevPtr(*gsc.device_ptr() as u64))?;
        dev.synchronize()?;
        let (hq, hs) = host_act_quant(&x, rows, k);
        let (dq, ds) = (dev.dtoh_sync_copy(&gq)?, dev.dtoh_sync_copy(&gsc)?);
        let (bq, bs) = (dq.iter().zip(&hq).filter(|(a, b)| a != b).count(), ds.iter().zip(&hs).filter(|(a, b)| a != b).count());
        ensure!(bq == 0 && bs == 0, "act_quant K={k}: {bq} value bytes and {bs} scale bytes differ from the reference formula");
        out.push(format!("act_quant K={k} rows={rows}: bitwise equal to the reference formula ({} value bytes, {} scales)", hq.len(), hs.len()));
    }

    // (2)-(4) dense on real weights
    let lins = [load_lin(&dev, dir, &idx, "layers.3.attn.wq_a", TensorClass::AttnWqA)?, load_lin(&dev, dir, &idx, "layers.3.ffn.shared_experts.w1", TensorClass::SharedW1)?,
                load_lin(&dev, dir, &idx, "layers.3.attn.wkv", TensorClass::AttnWkv)?];
    let mut checked = 0usize;
    for lin in &lins {
        let dw = dense_weight(lin);
        let (n, k) = (lin.n, lin.k);
        let max_rows = 64;
        let x = make_x(max_rows, k, 99 + n as u64);
        let xb: Vec<u16> = x.iter().map(|v| bf16_bits(*v)).collect();
        let gx = dev.htod_sync_copy(&xb)?;
        let (hq, hs) = host_act_quant(&x, max_rows, k);
        let step = (n / 96).max(1);
        let mut by_rows: BTreeMap<usize, Vec<u16>> = BTreeMap::new();
        for rows in [1usize, 2, 3, 6, 8, 12, 16, 17, 24, 32, 33, 64] {
            let sbytes = dense::scratch_bytes(&dw, rows);
            let gscr = dev.alloc_zeros::<u8>(sbytes)?;
            let mut gy = dev.alloc_zeros::<u16>(rows * n)?;
            let cls = dense::dense(&l, &dw, &Act { ptr: DevPtr(*gx.device_ptr() as u64), rows, cols: k, ld: k, dtype: ActDtype::Bf16 },
                &Act { ptr: DevPtr(*gy.device_ptr() as u64), rows, cols: n, ld: n, dtype: ActDtype::Bf16 }, Accum::Overwrite,
                Scratch { ptr: DevPtr(*gscr.device_ptr() as u64), bytes: sbytes })?;
            dev.synchronize()?;
            ensure!(cls == fp8::served_class(rows), "class {cls:?} for {rows} rows");
            let y = dev.dtoh_sync_copy(&gy)?;
            // reference on sampled columns
            let mut worst = 0f64; let mut rms = 0f64; let mut cnt = 0usize;
            for r in 0..rows { for c in (0..n).step_by(step) {
                let rf = host_dot(&lin.w, &lin.ws, k, c, 0, k / 32, &hq[r * k..(r + 1) * k], &hs[r * (k / 32)..(r + 1) * (k / 32)]);
                rms += (rf as f64).powi(2); cnt += 1;
                worst = worst.max((bf16_val(y[r * n + c]) as f64 - rf as f64).abs() - 0.00391 * (rf as f64).abs());   // bf16 half-ulp is 2^-8 relative
            } }
            rms = (rms / cnt as f64).sqrt();
            // what is left after allowing the one bf16 rounding is fp32 summation noise
            ensure!(worst / (rms + 1e-30) < 1e-5, "{} rows={rows}: max |diff| {worst:.3e} vs rms {rms:.3e}", lin.key);
            checked += cnt;
            by_rows.insert(rows, y);
            if rows == 1 || rows == 24 || rows == 64 { out.push(format!("{:<34} rows={rows:<3} class={:<7} max|diff|/rms(bf16 out) = {:.2e} over {cnt} sampled outputs", lin.key, cls.name(), worst / (rms + 1e-30))); }
        }
        // batch invariance: every call's rows are the leading rows of the largest call of its family, bit for bit
        for (family, top) in [(1..=32usize, 32usize), (33..=64, 64)] {
            let t = &by_rows[&top];
            for (rows, y) in by_rows.range(family) { ensure!(y[..] == t[..rows * n], "{} rows={rows}: differs from the {top}-row call", lin.key); }
        }
        // slices + accumulate through the quantised interface (f32 out)
        let rows = 6usize;
        let kb = k / 32;
        let (k0b, nkb, n0, nc) = (kb / 8, (kb / 4) | 1, 16usize, (n / 2) / 16 * 16);
        let slice_k = nkb * 32;
        let xs_local: Vec<f32> = (0..rows).flat_map(|r| x[r * k + k0b * 32..r * k + k0b * 32 + slice_k].to_vec()).collect();
        let (lq, ls) = host_act_quant(&xs_local, rows, slice_k);
        let (gq, gsc) = (dev.htod_sync_copy(&lq)?, dev.htod_sync_copy(&ls)?);
        let mut gy = dev.alloc_zeros::<f32>(rows * nc)?;
        for pass in 0..2 {
            fp8::gemm_quantised(&l, DevPtr(*lin.gw.device_ptr() as u64), DevPtr(*lin.gs.device_ptr() as u64), k, kb, DevPtr(*gq.device_ptr() as u64), DevPtr(*gsc.device_ptr() as u64),
                rows, n0, nc, k0b, nkb, &Act { ptr: DevPtr(*gy.device_ptr() as u64), rows, cols: nc, ld: nc, dtype: ActDtype::F32 }, if pass == 0 { Accum::Overwrite } else { Accum::Add })?;
            dev.synchronize()?;
        }
        let y = dev.dtoh_sync_copy(&gy)?;
        let mut worst = 0f64; let mut rms = 0f64; let mut cnt = 0;
        for r in 0..rows { for c in (0..nc).step_by(7) {
            let rf = host_dot(&lin.w, &lin.ws, k, n0 + c, k0b, nkb, &lq[r * slice_k..(r + 1) * slice_k], &ls[r * nkb..(r + 1) * nkb]);
            rms += (rf as f64).powi(2); cnt += 1; worst = worst.max((y[r * nc + c] as f64 - 2.0 * rf as f64).abs());
        } }
        rms = (rms / cnt as f64).sqrt();
        ensure!(worst / (2.0 * rms + 1e-30) < 5e-5, "{} slice: max |diff| {worst:.3e} vs rms {rms:.3e}", lin.key);
        out.push(format!("{:<34} slice n0={n0} n={nc} k0b={k0b} nkb={nkb} + accumulate (x2): max|diff|/rms = {:.2e}", lin.key, worst / (2.0 * rms + 1e-30)));
        out.push(format!("{:<34} batch invariance: every row of the 1,2,3,6,8,12,16,17,24,32-row calls equals the 32-row call bitwise (split-K family); 33 and 64 rows likewise (GEMM family)", lin.key));
    }
    if checked == 0 { bail!("no outputs were checked"); }
    out.push(format!("{checked} sampled outputs checked against the reference scheme"));
    out.push("DSV41_KN1_FP8_OK".into());
    Ok(out)
}

