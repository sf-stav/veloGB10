//! KE: `--probe-dsv41-kernel --dsv41-kernel-set trellis [--model-dir <fixture dir>]` — every kernel of band 10 against the
//! host trellis decode + an f64 reference chain, slices on both sides, batch invariance at widths 1..24 / 32 and the wide
//! class, the plain linears, then the byte-floor table. The fixture dir holds `fixture.safetensors` (real EAQ v1 tensors cut
//! by PLAN/DSV41_KE_REPORT.md's script); without it the real-tensor cases are skipped and SAY so.

use super::launch::bits_code;
use crate::dsv41::census::classify::{Role, TensorClass};
use crate::dsv41::census::LinearFormat;
use crate::dsv41::weights::{Accum, Act, ActDtype, DenseWeight, DevPtr, DevTensor, Dsv41Weights, Launch, MoeWeight, Route, Scratch, Shard};
use crate::dsv41::weights::{exl3::Exl3Weights, plain::PlainWeights};
use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::{sys, CudaDevice, CudaSlice, DevicePtr, DeviceRepr, ValidAsZeroBits};
use std::sync::Arc;

pub(super) struct Rng(pub u64);
impl Rng {
    pub fn u(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    pub fn f(&mut self) -> f32 { ((self.u() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 }
    pub fn below(&mut self, n: usize) -> usize { (self.u() % n as u64) as usize }
}

pub(super) struct Cx { pub dev: Arc<CudaDevice>, pub stream: u64 }
pub(super) fn dp<T>(s: &CudaSlice<T>) -> u64 { *s.device_ptr() as u64 }
impl Cx {
    pub fn new() -> Result<Cx> {
        let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
        crate::dsv41::gpu::load_module(&dev)?;
        let mut st: sys::CUstream = std::ptr::null_mut();
        ensure!(unsafe { sys::cuStreamCreate(&mut st, 0) } == sys::CUresult::CUDA_SUCCESS, "cuStreamCreate");
        Ok(Cx { dev, stream: st as u64 })
    }
    pub fn l(&self) -> Launch { Launch { stream: self.stream, pdl: false } }
    pub fn up<T: DeviceRepr + Unpin>(&self, v: &[T]) -> Result<CudaSlice<T>> { Ok(self.dev.htod_sync_copy(v)?) }
    pub fn zeros<T: DeviceRepr + ValidAsZeroBits + Unpin>(&self, n: usize) -> Result<CudaSlice<T>> { let z = self.dev.alloc_zeros::<T>(n.max(1))?; Ok(z) }
    pub fn sync(&self) -> Result<()> { ensure!(unsafe { sys::cuStreamSynchronize(self.stream as sys::CUstream) } == sys::CUresult::CUDA_SUCCESS, "stream sync (a kernel failed)"); Ok(()) }
    pub fn dl<T: DeviceRepr + Unpin>(&self, s: &CudaSlice<T>) -> Result<Vec<T>> { self.sync()?; Ok(self.dev.dtoh_sync_copy(s)?) }
}
pub(super) fn h2f(b: u16) -> f32 { half::f16::from_bits(b).to_f32() }
pub(super) fn f2h(v: f64) -> u16 { half::f16::from_f64(v).to_bits() }
pub(super) fn bf2f(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }
pub(super) fn f2bf(v: f32) -> f32 { half::bf16::from_f32(v).to_f32() }

pub(super) fn had(v: &mut [f64]) {
    let n = v.len(); let mut h = 1;
    while h < n { for i in (0..n).step_by(2 * h) { for j in i..i + h { let (a, b) = (v[j], v[j + h]); v[j] = a + b; v[j + h] = a - b; } } h *= 2; }
    let s = 1.0 / (n as f64).sqrt(); for x in v.iter_mut() { *x *= s; }
}
/// fp16(H128(x * suh)) over a K slice (x: rows x kloc, suh: kloc)
pub(super) fn ref_prep(x: &[f32], rows: usize, kloc: usize, suh: &[u16]) -> Vec<u16> {
    let mut o = vec![0u16; rows * kloc];
    for m in 0..rows { for b in 0..kloc / 128 {
        let mut v: Vec<f64> = (0..128).map(|j| x[m * kloc + b * 128 + j] as f64 * h2f(suh[b * 128 + j]) as f64).collect();
        had(&mut v);
        for j in 0..128 { o[m * kloc + b * 128 + j] = f2h(v[j]); }
    } }
    o
}
/// out[m][n - n_lo] = H128-block( sum_k xh[m][k] * W[k_lo + k][n] ) * svh[n], n in [n_lo, n_hi)
pub(super) fn ref_gemm_post(xh: &[u16], rows: usize, kloc: usize, w: &[f32], out_full: usize, k_lo: usize, n_lo: usize, n_hi: usize, svh: &[u16]) -> Vec<f64> {
    let (b0, b1) = (n_lo / 128, (n_hi + 127) / 128);
    let mut res = vec![0f64; rows * (n_hi - n_lo)];
    for m in 0..rows {
        let xr: Vec<f64> = (0..kloc).map(|k| h2f(xh[m * kloc + k]) as f64).collect();
        for b in b0..b1 {
            let mut raw = vec![0f64; 128];
            for k in 0..kloc { let x = xr[k]; if x == 0.0 { continue; } let row = &w[(k_lo + k) * out_full + b * 128..(k_lo + k) * out_full + b * 128 + 128]; for j in 0..128 { raw[j] += x * row[j] as f64; } }
            had(&mut raw);
            for j in 0..128 { let n = b * 128 + j; if n >= n_lo && n < n_hi { res[m * (n_hi - n_lo) + n - n_lo] = raw[j] * h2f(svh[n]) as f64; } }
        }
    }
    res
}
pub(super) fn rel_l2(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut n, mut d) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) { n += (*x as f64 - y).powi(2); d += y * y; }
    (n / d.max(1e-300)).sqrt()
}
fn decode_w(tr: &[i16], kb: usize, nb: usize, k2: usize) -> Vec<f32> {
    crate::exl3::decode_trellis_host(tr, kb, nb, k2).into_iter().map(h2f).collect()
}

/// A linear on the host and the device.
pub(super) struct Lin { pub k2: usize, pub inn: usize, pub out: usize, pub tr: Vec<i16>, pub suh: Vec<u16>, pub svh: Vec<u16>, pub w: Vec<f32>,
             pub d_tr: CudaSlice<i16>, pub d_suh: CudaSlice<u16>, pub d_svh: CudaSlice<u16> }
impl Lin {
    pub fn from_host(cx: &Cx, k2: usize, inn: usize, out: usize, tr: Vec<i16>, suh: Vec<u16>, svh: Vec<u16>, decode: bool) -> Result<Lin> {
        let w = if decode { decode_w(&tr, inn / 16, out / 16, k2) } else { vec![] };
        Ok(Lin { k2, inn, out, d_tr: cx.up(&tr)?, d_suh: cx.up(&suh)?, d_svh: cx.up(&svh)?, tr, suh, svh, w })
    }
    pub fn synth(cx: &Cx, rng: &mut Rng, k2: usize, inn: usize, out: usize) -> Result<Lin> {
        let tr: Vec<i16> = (0..(inn / 16) * (out / 16) * 8 * k2).map(|_| rng.u() as i16).collect();
        let suh: Vec<u16> = (0..inn).map(|_| f2h(if rng.u() & 1 == 0 { 1.0 } else { -1.0 } * (0.7 + 0.3 * rng.f().abs() as f64))).collect();
        let svh: Vec<u16> = (0..out).map(|_| f2h((if rng.u() & 1 == 0 { 1.0 } else { -1.0 }) * (0.5 + 0.5 * rng.f().abs() as f64))).collect();
        Lin::from_host(cx, k2, inn, out, tr, suh, svh, true)
    }
    /// the resident slice (block-aligned) of this linear, as its own tensors
    pub fn cut(&self, cx: &Cx, n_lo: usize, n_hi: usize, k_lo: usize, k_hi: usize) -> Result<Lin> {
        let (kb, nb, wd) = (self.inn / 16, self.out / 16, 8 * self.k2);
        let mut tr = Vec::new();
        for k in k_lo / 16..k_hi / 16 { for n in n_lo / 16..n_hi / 16 { let o = (k * nb + n) * wd; tr.extend_from_slice(&self.tr[o..o + wd]); } }
        let _ = kb;
        Lin::from_host(cx, self.k2, k_hi - k_lo, n_hi - n_lo, tr, self.suh[k_lo..k_hi].to_vec(), self.svh[n_lo..n_hi].to_vec(), false)
    }
    pub fn dense(&self, key: &str, logical_out: usize, logical_in: usize, shard: Shard) -> DenseWeight {
        let t = |role, ptr: u64, dt: &str, shape: Vec<u64>, bytes: usize| DevTensor { role, ptr: DevPtr(ptr), dtype: dt.into(), shape, bytes: bytes as u64 };
        DenseWeight { key: key.into(), class: TensorClass::AttnWqA, layer: Some(0), sub: None,
            format: LinearFormat::Exl3 { k: (self.k2 / 2) as u8, mul1: true, suh: true, svh: true }, out: logical_out as u64, inn: logical_in as u64, shard,
            tensors: vec![t(Role::Trellis, dp(&self.d_tr), "I16", vec![(self.inn / 16) as u64, (self.out / 16) as u64, (8 * self.k2) as u64], self.tr.len() * 2),
                          t(Role::Suh, dp(&self.d_suh), "F16", vec![self.inn as u64], self.inn * 2), t(Role::Svh, dp(&self.d_svh), "F16", vec![self.out as u64], self.out * 2)] }
    }
}

struct Out { y: Vec<f32>, xh: Vec<u16> }
/// one dense call: x rows x kloc (f32), returns y [rows, ncols] and the device's xh for the first `keep` rows
fn run_dense(cx: &Cx, w: &DenseWeight, x: &[f32], rows: usize, keep: usize) -> Result<Out> {
    let b = Exl3Weights;
    let (n_lo, n_hi, k_lo, k_hi) = match &w.shard { Shard::Whole => (0, w.out as usize, 0, w.inn as usize), Shard::Out(r) => (r.start as usize, r.end as usize, 0, w.inn as usize), Shard::In(r) => (0, w.out as usize, r.start as usize, r.end as usize) };
    let kloc = k_hi - k_lo; let ncols = n_hi - n_lo;
    let xs: Vec<f32> = x.chunks(w.inn as usize).take(rows).flat_map(|r| r[k_lo..k_hi].to_vec()).collect();
    let dx = cx.up(&xs)?;
    let dy = cx.zeros::<f32>(rows * ncols)?;
    let sb = b.scratch_bytes(w, rows);
    let ds = cx.zeros::<u8>(sb)?;
    let xa = Act { ptr: DevPtr(dp(&dx)), rows, cols: kloc, ld: kloc, dtype: ActDtype::F32 };
    let ya = Act { ptr: DevPtr(dp(&dy)), rows, cols: ncols, ld: ncols, dtype: ActDtype::F32 };
    b.dense(&cx.l(), w, &xa, &ya, Accum::Overwrite, Scratch { ptr: DevPtr(dp(&ds)), bytes: sb })?;
    let y = cx.dl(&dy)?;
    let xhb = cx.dl(&ds)?;
    let xh: Vec<u16> = xhb[..keep * kloc * 2].chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    Ok(Out { y, xh })
}

pub(super) fn check(name: &str, err: f64, tol: f64, fails: &mut Vec<String>, lines: &mut Vec<String>) {
    let ok = err <= tol;
    lines.push(format!("  {} {name}: rel-L2 {err:.2e} (tol {tol:.0e})", if ok { "ok  " } else { "FAIL" }));
    if !ok { fails.push(name.to_string()); }
}

fn dense_battery(cx: &Cx, rng: &mut Rng, fails: &mut Vec<String>, lines: &mut Vec<String>) -> Result<()> {
    let (inn, out) = (1280usize, 2048usize);
    let xall: Vec<f32> = (0..200 * inn).map(|_| rng.f()).collect();
    for &k2 in &[3usize, 4, 5, 6, 7, 8, 10, 12, 14, 16] {
        let lin = Lin::synth(cx, rng, k2, inn, out)?;
        ensure!(bits_code(8 * k2 as u64).is_ok(), "bits_code");
        let nm = |s: &str| format!("dense K{}{} {s}", k2 / 2, if k2 % 2 == 1 { ".5" } else { "" });
        let full = lin.dense("lin", out, inn, Shard::Whole);
        // whole matrix, every row class, first min(rows, 8) rows vs the f64 chain on the device's own xh
        let mut row0: Option<Vec<f32>> = None;
        for &rows in &[1usize, 2, 5, 16, 17, 24, 32, 33, 40, 64, 65, 200] {
            let o = run_dense(cx, &full, &xall, rows, rows.min(8))?;
            let kr = rows.min(8);
            let r = ref_gemm_post(&o.xh, kr, inn, &lin.w, out, 0, 0, out, &lin.svh);
            check(&nm(&format!("whole rows={rows}")), rel_l2(&o.y[..kr * out], &r), 1e-5, fails, lines);
            if rows == 1 {
                let px = ref_prep(&xall, 1, inn, &lin.suh);
                let dif = px.iter().zip(&o.xh).map(|(a, b)| (h2f(*a) - h2f(*b)).abs() as f64).fold(0.0, f64::max);
                check(&nm("prep fp16 max abs diff vs f64"), dif, 2e-3, fails, lines);
            }
            // batch invariance within the class: row 0 bit-equal to the 1-row call (rows <= 32) / to the 33-row call (wide)
            if rows == 1 { row0 = Some(o.y[..out].to_vec()); }
            if rows == 33 { row0 = Some(o.y[..out].to_vec()); }
            if rows <= 32 || rows > 33 {
                let base = row0.as_ref().unwrap();
                let same = o.y[..out].iter().zip(base).all(|(a, b)| a.to_bits() == b.to_bits());
                lines.push(format!("  {} {} row 0 of rows={rows} bit-equal to its class's 1st call", if same { "ok  " } else { "FAIL" }, nm("batch invariance")));
                if !same { fails.push(nm(&format!("invariance rows={rows}"))); }
            }
        }
        // all rows of a 24-row call equal the 1-row calls (decode class + coop + MT2)
        {
            let o24 = run_dense(cx, &full, &xall, 24, 0)?;
            let mut bad = 0;
            for r in 0..24 { let o1 = run_dense(cx, &full, &xall[r * inn..], 1, 0)?; if !o1.y.iter().zip(&o24.y[r * out..(r + 1) * out]).all(|(a, b)| a.to_bits() == b.to_bits()) { bad += 1; } }
            lines.push(format!("  {} {} 24 rows each bit-equal to its 1-row call ({bad} differ)", if bad == 0 { "ok  " } else { "FAIL" }, nm("batch invariance")));
            if bad > 0 { fails.push(nm("invariance 24")); }
        }
        // PK4: every width 1..32: each row of the N-row call bit-equal to its 1-row call (the decode class's invariance)
        {
            let one: Vec<Vec<f32>> = (0..32).map(|r| run_dense(cx, &full, &xall[r * inn..], 1, 0).map(|o| o.y)).collect::<Result<_>>()?;
            let mut bad = vec![];
            for rows in 2..=32usize {
                let o = run_dense(cx, &full, &xall, rows, 0)?;
                if (0..rows).any(|r| !o.y[r * out..(r + 1) * out].iter().zip(&one[r]).all(|(a, b)| a.to_bits() == b.to_bits())) { bad.push(rows); }
            }
            lines.push(format!("  {} {} every row of N = 2..32 bit-equal to its 1-row call (widths differing: {bad:?})", if bad.is_empty() { "ok  " } else { "FAIL" }, nm("batch invariance")));
            if !bad.is_empty() { fails.push(nm("invariance 1..32")); }
        }
        // out slices (aligned, un-aligned), in slices, resident slices
        for (n_lo, n_hi) in [(256usize, 1024usize), (100, 1500), (0, 128), (1920, 2048)] {
            let w = lin.dense("lin", out, inn, Shard::Out(n_lo as u64..n_hi as u64));
            let o = run_dense(cx, &w, &xall, 3, 3)?;
            let r = ref_gemm_post(&o.xh, 3, inn, &lin.w, out, 0, n_lo, n_hi, &lin.svh);
            check(&nm(&format!("out slice {n_lo}..{n_hi}")), rel_l2(&o.y, &r), 1e-5, fails, lines);
        }
        {
            let (a, b) = (256usize, 1024usize);
            let w = lin.dense("lin", out, inn, Shard::In(a as u64..b as u64));
            let o = run_dense(cx, &w, &xall, 3, 3)?;
            let r = ref_gemm_post(&o.xh, 3, b - a, &lin.w, out, a, 0, out, &lin.svh);
            check(&nm("in slice 256..1024"), rel_l2(&o.y, &r), 1e-5, fails, lines);
            let w1 = lin.dense("lin", out, inn, Shard::In(0..640)); let w2 = lin.dense("lin", out, inn, Shard::In(640..1280));
            let (o1, o2) = (run_dense(cx, &w1, &xall, 3, 3)?, run_dense(cx, &w2, &xall, 3, 3)?);
            let sum: Vec<f32> = o1.y.iter().zip(&o2.y).map(|(a, b)| a + b).collect();
            let wf = run_dense(cx, &full, &xall, 3, 3)?;
            let refv: Vec<f64> = wf.y.iter().map(|v| *v as f64).collect();
            check(&nm("two In halves summed vs the whole"), rel_l2(&sum, &refv), 1e-5, fails, lines);
            // resident-slice twins are bit-equal to whole-resident with the same shard
            let rs = lin.cut(cx, 0, out, a, b)?;
            let wr = rs.dense("lin", out, inn, Shard::In(a as u64..b as u64));
            let orr = run_dense(cx, &wr, &xall, 3, 3)?;
            let same = orr.y.iter().zip(&o.y).all(|(x, y)| x.to_bits() == y.to_bits());
            lines.push(format!("  {} {} resident In-slice == whole-resident, bits", if same { "ok  " } else { "FAIL" }, nm("residency")));
            if !same { fails.push(nm("resident in")); }
            let rs = lin.cut(cx, 256, 1024, 0, inn)?;
            let wr = rs.dense("lin", out, inn, Shard::Out(256..1024));
            let wo = lin.dense("lin", out, inn, Shard::Out(256..1024));
            let (a1, a2) = (run_dense(cx, &wr, &xall, 3, 3)?, run_dense(cx, &wo, &xall, 3, 3)?);
            let same = a1.y.iter().zip(&a2.y).all(|(x, y)| x.to_bits() == y.to_bits());
            lines.push(format!("  {} {} resident Out-slice == whole-resident, bits", if same { "ok  " } else { "FAIL" }, nm("residency")));
            if !same { fails.push(nm("resident out")); }
        }
    }
    Ok(())
}

/// PK4: `dense_dec::run_group` (several linears in one launch pair): each segment vs the f64 chain, vs its single call
/// (PD1: bit for bit: each segment is cut into its own partitions),
/// batch invariance of the grouped call at every width 1..32, an In-slice segment, an accumulating segment, mixed widths
/// (falls back to single calls), and an output dtype mix.
fn dense_group_battery(cx: &Cx, rng: &mut Rng, fails: &mut Vec<String>, lines: &mut Vec<String>) -> Result<()> {
    use super::dense_dec::{run_group, scratch_bytes_for_group, Call};
    let shapes = [(10usize, 1024usize, 1024usize, Shard::Whole), (10, 1024, 512, Shard::Whole), (10, 2048, 768, Shard::In(512..1536)), (10, 1024, 1280, Shard::Out(128..1152))];
    let lins: Vec<Lin> = shapes.iter().map(|(k2, inn, out, _)| Lin::synth(cx, rng, *k2, *inn, *out)).collect::<Result<_>>()?;
    let ws: Vec<DenseWeight> = lins.iter().zip(&shapes).map(|(l, (_, inn, out, sh))| l.dense("grp", *out, *inn, sh.clone())).collect();
    let geo: Vec<(usize, usize, usize, usize)> = ws.iter().map(|w| match &w.shard { Shard::Whole => (0, w.out as usize, 0, w.inn as usize), Shard::Out(r) => (r.start as usize, r.end as usize, 0, w.inn as usize), Shard::In(r) => (0, w.out as usize, r.start as usize, r.end as usize) }).collect();
    let xin: Vec<f32> = (0..32 * 2048).map(|_| rng.f()).collect();
    let run = |rows: usize, ys: &mut Vec<Vec<f32>>, xh_out: &mut Vec<Vec<u16>>, single: bool, acc_last: bool| -> Result<()> {
        let dxs: Vec<CudaSlice<f32>> = geo.iter().map(|&(_, _, k_lo, k_hi)| cx.up(&xin.chunks(2048).take(rows).flat_map(|r| r[k_lo..k_hi].to_vec()).collect::<Vec<f32>>())).collect::<Result<_>>()?;
        let dys: Vec<CudaSlice<f32>> = geo.iter().enumerate().map(|(g, &(n_lo, n_hi, _, _))| if acc_last && g + 1 == geo.len() { cx.up(&vec![0.5f32; rows * (n_hi - n_lo)]) } else { cx.zeros::<f32>(rows * (n_hi - n_lo)) }).collect::<Result<_>>()?;
        let xs: Vec<Act> = geo.iter().zip(&dxs).map(|(&(_, _, k_lo, k_hi), d)| Act { ptr: DevPtr(dp(d)), rows, cols: k_hi - k_lo, ld: k_hi - k_lo, dtype: ActDtype::F32 }).collect();
        let yv: Vec<Act> = geo.iter().zip(&dys).map(|(&(n_lo, n_hi, _, _), d)| Act { ptr: DevPtr(dp(d)), rows, cols: n_hi - n_lo, ld: n_hi - n_lo, dtype: ActDtype::F32 }).collect();
        let calls: Vec<Call> = (0..ws.len()).map(|g| Call::new(&ws[g], &xs[g], &yv[g], if acc_last && g + 1 == ws.len() { Accum::Add } else { Accum::Overwrite })).collect();
        let refs: Vec<&DenseWeight> = ws.iter().collect();
        let sb = scratch_bytes_for_group(&refs, rows);
        let ds = cx.zeros::<u8>(sb)?;
        let sc = Scratch { ptr: DevPtr(dp(&ds)), bytes: sb };
        if single { for c in &calls { Exl3Weights.dense(&cx.l(), c.w, c.x, c.y, c.acc, sc)?; } } else { run_group(&cx.l(), &calls, sc)?; }
        *ys = dys.iter().map(|d| cx.dl(d)).collect::<Result<_>>()?;
        if !single {
            let sbytes = cx.dl(&ds)?;
            *xh_out = super::dense_dec::xh_offsets(&refs, rows).iter().zip(&geo).map(|(&o, &(_, _, k_lo, k_hi))|
                sbytes[o..o + rows * (k_hi - k_lo) * 2].chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()).collect();
        }
        Ok(())
    };
    let (mut grp, mut sgl, mut gxh, mut nxh) = (vec![], vec![], vec![], vec![]);
    run(3, &mut grp, &mut gxh, false, true)?;
    run(3, &mut sgl, &mut nxh, true, true)?;
    for (g, (lin, &(n_lo, n_hi, k_lo, k_hi))) in lins.iter().zip(&geo).enumerate() {
        // the f64 chain on the device's own prep output for this segment (as the single-call battery does)
        let mut r = ref_gemm_post(&gxh[g], 3, k_hi - k_lo, &lin.w, lin.out, k_lo, n_lo, n_hi, &lin.svh);
        if g + 1 == geo.len() { for v in r.iter_mut() { *v += 0.5; } }
        check(&format!("grouped segment {g} ({}x{} {:?}) vs f64", lin.inn, lin.out, ws[g].shard), rel_l2(&grp[g], &r), 1e-5, fails, lines);
        // PD1: each segment is cut into its own partitions (the 1-segment cut), so a grouped call equals the single calls bit for bit
        let same = grp[g].iter().zip(&sgl[g]).all(|(a, b)| a.to_bits() == b.to_bits());
        lines.push(format!("  {} grouped segment {g} bit-equal to its single call (PD1 per-segment partitions)", if same { "ok  " } else { "FAIL" }));
        if !same { fails.push(format!("grouped segment {g} != single")); }
    }
    let mut one: Vec<Vec<Vec<f32>>> = vec![];
    for r in 0..32 {
        let (mut y, mut xh) = (vec![], vec![]);
        run(r + 1, &mut y, &mut xh, false, false)?;
        one.push(geo.iter().enumerate().map(|(g, &(n_lo, n_hi, _, _))| y[g][r * (n_hi - n_lo)..(r + 1) * (n_hi - n_lo)].to_vec()).collect());
    }
    // `one[r]` is row r of an (r+1)-row call; invariance: row r equal in every N > r call, and to the (r+1)-row call
    let mut bad = vec![];
    let (mut y32, mut xh32) = (vec![], vec![]);
    run(32, &mut y32, &mut xh32, false, false)?;
    for r in 0..32 {
        for (g, &(n_lo, n_hi, _, _)) in geo.iter().enumerate() {
            let w = n_hi - n_lo;
            if !y32[g][r * w..(r + 1) * w].iter().zip(&one[r][g]).all(|(a, b)| a.to_bits() == b.to_bits()) { bad.push((r, g)); }
        }
    }
    lines.push(format!("  {} grouped call: row r of the 32-row call bit-equal to row r of the (r+1)-row call, r = 0..31, every segment ({} differ)", if bad.is_empty() { "ok  " } else { "FAIL" }, bad.len()));
    if !bad.is_empty() { fails.push("grouped invariance".into()); }
    Ok(())
}

// ---------------------------------------------------------------- fixture (real tensors) ---------------------------------------
pub(super) struct Fx { pub h: crate::dsv41::st::StHeader }
impl Fx {
    pub fn open(dir: &str) -> Result<Fx> { Ok(Fx { h: crate::dsv41::st::read_header(&std::path::Path::new(dir).join("fixture.safetensors"))? }) }
    pub fn shape(&self, n: &str) -> Result<Vec<u64>> { Ok(self.h.tensors.iter().find(|t| t.name == n).with_context(|| format!("fixture lacks {n}"))?.shape.clone()) }
    pub fn lin(&self, cx: &Cx, base: &str, decode: bool) -> Result<Lin> {
        let tr = self.shape(&format!("{base}.trellis"))?;
        let rd = |p: &str| crate::dsv41::st::read_tensor(&self.h, &format!("{base}.{p}"));
        let tb = rd("trellis")?;
        let trv: Vec<i16> = tb.chunks(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        let u16s = |b: Vec<u8>| -> Vec<u16> { b.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() };
        Lin::from_host(cx, (tr[2] / 8) as usize, tr[0] as usize * 16, tr[1] as usize * 16, trv, u16s(rd("suh")?), u16s(rd("svh")?), decode)
    }
}

fn real_dense(cx: &Cx, fx: &Fx, rng: &mut Rng, fails: &mut Vec<String>, lines: &mut Vec<String>) -> Result<()> {
    let cases: [(&str, &str, Shard); 5] = [
        ("layers.24.attn.wq_a", "wq_a K5 whole", Shard::Whole),
        ("layers.24.attn.wq_b", "wq_b K5 heads 4..8 of 64 (out 2048..4096)", Shard::Out(2048..4096)),
        ("layers.24.attn.wo_a.slice.0", "wo_a slice 0 whole", Shard::Whole),
        ("layers.24.attn.wo_b", "wo_b K5 In 2048..4096", Shard::In(2048..4096)),
        ("layers.24.ffn.shared_experts.w1", "shared w1 K4 Out 1152..2304", Shard::Out(1152..2304)),
    ];
    for (base, name, shard) in cases {
        let lin = fx.lin(cx, base, true)?;
        let x: Vec<f32> = (0..2 * lin.inn).map(|_| rng.f()).collect();
        let w = lin.dense(base, lin.out, lin.inn, shard.clone());
        let (n_lo, n_hi, k_lo, k_hi) = match &shard { Shard::Whole => (0, lin.out, 0, lin.inn), Shard::Out(r) => (r.start as usize, r.end as usize, 0, lin.inn), Shard::In(r) => (0, lin.out, r.start as usize, r.end as usize) };
        let o = run_dense(cx, &w, &x, 2, 2)?;
        let r = ref_gemm_post(&o.xh, 2, k_hi - k_lo, &lin.w, lin.out, k_lo, n_lo, n_hi, &lin.svh);
        let (m, s) = { let v: Vec<f64> = lin.w.iter().map(|x| *x as f64).collect(); let m = v.iter().sum::<f64>() / v.len() as f64; (m, (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()) };
        ensure!(lin.w.iter().all(|x| x.is_finite()) && s > 0.0, "{name}: decoded weights not finite / constant");
        check(&format!("real {name} (decoded W mean {m:.1e} std {s:.3})"), rel_l2(&o.y, &r), 1e-5, fails, lines);
    }
    // the head's resident slice: out 129280 logical, first 1024 columns resident
    let h = fx.lin(cx, "head.slice", true)?;
    let hs = crate::dsv41::st::read_tensor(&fx.h, "head.suh")?;
    let suh: Vec<u16> = hs.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let hl = Lin { suh: suh.clone(), d_suh: cx.up(&suh)?, ..h };
    let x: Vec<f32> = (0..3 * 5120).map(|_| rng.f()).collect();
    let w = hl.dense("head", 129_280, 5120, Shard::Out(0..1024));
    let o = run_dense(cx, &w, &x, 3, 3)?;
    let r = ref_gemm_post(&o.xh, 3, 5120, &hl.w, 1024, 0, 0, 1024, &hl.svh);
    check("real head K6 vocab slice 0..1024 (resident slice, f32 logits)", rel_l2(&o.y, &r), 1e-5, fails, lines);
    Ok(())
}

// ---------------------------------------------------------------- MoE -------------------------------------------------------
pub(super) struct Ex { pub l: [Lin; 3] }  // w1, w2, w3
pub(super) fn load_experts(cx: &Cx, fx: &Fx, layer: &str, n: usize) -> Result<Vec<Ex>> {
    (0..n).map(|e| Ok(Ex { l: [fx.lin(cx, &format!("layers.{layer}.ffn.experts.{e}.w1"), true)?, fx.lin(cx, &format!("layers.{layer}.ffn.experts.{e}.w2"), true)?, fx.lin(cx, &format!("layers.{layer}.ffn.experts.{e}.w3"), true)?] })).collect()
}
pub(super) fn moe_weight(ex: &[Ex], layer: u16, top_k: usize, rank_slice: &dyn Fn(usize) -> (usize, usize)) -> MoeWeight {
    let experts = ex.iter().enumerate().map(|(e, x)| {
        let (lo, hi) = rank_slice(e);
        let (lo, hi) = (lo as u64 * 128, hi as u64 * 128);
        let sh13 = if lo == 0 && hi == 2304 { Shard::Whole } else { Shard::Out(lo..hi) };
        let sh2 = if lo == 0 && hi == 2304 { Shard::Whole } else { Shard::In(lo..hi) };
        [x.l[0].dense("w1", 2304, 5120, sh13.clone()), x.l[1].dense("w2", 5120, 2304, sh2), x.l[2].dense("w3", 2304, 5120, sh13)]
    }).collect::<Vec<_>>();
    MoeWeight { layer, n_experts: ex.len(), local: 0..ex.len(), top_k, dim: 5120, inter: 2304, swiglu_limit: 10.0, experts }
}
/// host reference of one expert on one row (f64 chain mirroring the kernels' fp16 / bf16 rounding points), f32 output [5120]
pub(super) fn ref_expert(x: &[f32], e: &Ex, wgt: f32, limit: f32) -> Vec<f64> {
    let xh1 = |m: usize| ref_prep(x, 1, 5120, &e.l[m].suh);
    let (xa, xb) = (xh1(0), xh1(2));
    let g = ref_gemm_post(&xa, 1, 5120, &e.l[0].w, 2304, 0, 0, 2304, &e.l[0].svh);
    let u = ref_gemm_post(&xb, 1, 5120, &e.l[2].w, 2304, 0, 0, 2304, &e.l[2].svh);
    let a: Vec<f32> = (0..2304).map(|i| {
        let (mut gg, mut uu) = (g[i] as f32, u[i] as f32);
        if limit > 0.0 { uu = uu.clamp(-limit, limit); gg = gg.min(limit); }
        f2bf((gg / (1.0 + (-gg).exp())) * uu * wgt)
    }).collect();
    let xh2 = ref_prep(&a, 1, 2304, &e.l[1].suh);
    ref_gemm_post(&xh2, 1, 2304, &e.l[1].w, 5120, 0, 0, 5120, &e.l[1].svh)
}

fn run_moe(cx: &Cx, mw: &MoeWeight, x: &[f32], ids: &[i32], wts: &[f32], rows: usize) -> Result<Vec<f32>> {
    let b = Exl3Weights;
    super::moe::clear_tables();
    let t = super::moe::prepare_moe(mw)?;
    let sb = super::moe::scratch_bytes(rows, mw.top_k, mw.dim, t.nblk_max);
    let ds = cx.zeros::<u8>(sb)?;
    let (dx, di, dw) = (cx.up(&x[..rows * 5120])?, cx.up(&ids[..rows * mw.top_k])?, cx.up(&wts[..rows * mw.top_k])?);
    let dy = cx.zeros::<f32>(rows * 5120)?;
    // the kernel ADDS: start from zero
    unsafe { sys::cuMemsetD8_v2(dp(&dy), 0, rows * 5120 * 4) };
    let xa = Act { ptr: DevPtr(dp(&dx)), rows, cols: 5120, ld: 5120, dtype: ActDtype::F32 };
    let ya = Act { ptr: DevPtr(dp(&dy)), rows, cols: 5120, ld: 5120, dtype: ActDtype::F32 };
    b.moe(&cx.l(), mw, &xa, &Route { ids: DevPtr(dp(&di)), weights: DevPtr(dp(&dw)), top_k: mw.top_k }, &ya, Scratch { ptr: DevPtr(dp(&ds)), bytes: sb })?;
    cx.dl(&dy)
}
fn slice_pat(tp: usize, e: usize) -> (usize, usize) {   // 128-block range [lo, hi) of expert e at TP `tp`, rank given by closure users
    let _ = (tp, e); (0, 18)
}
pub(super) fn tp_bounds(tp: usize, e: usize) -> Vec<usize> {
    let w: Vec<usize> = match tp {
        1 => vec![18], 2 => vec![9, 9],
        4 => if e % 2 == 0 { vec![5, 4, 5, 4] } else { vec![4, 5, 4, 5] },
        _ => { let mut b = vec![3, 2, 2, 2, 3, 2, 2, 2]; b.rotate_left(e % 4); b }
    };
    let mut v = vec![0]; for x in w { v.push(v.last().unwrap() + x); } v
}

pub(super) fn moe_battery(cx: &Cx, fx: &Fx, rng: &mut Rng, fails: &mut Vec<String>, lines: &mut Vec<String>) -> Result<()> {
    let _ = slice_pat;
    for (layer, nex, label) in [("24", 10usize, "routed K3"), ("20", 6, "routed K2")] {
        let ex = load_experts(cx, fx, layer, nex)?;
        let top_k = 6.min(nex);
        let mut mk = |rows: usize| {
            let x: Vec<f32> = (0..rows * 5120).map(|_| rng.f()).collect();
            let mut ids = vec![]; let mut wts = vec![];
            for _ in 0..rows { let mut pool: Vec<usize> = (0..nex).collect(); for _ in 0..top_k { let i = rng.below(pool.len()); ids.push(pool.remove(i) as i32); wts.push(0.1 + 0.4 * rng.f().abs()); } }
            (x, ids, wts)
        };
        let full = moe_weight(&ex, 24, top_k, &|_| (0, 18));
        let (x, ids, wts) = mk(24);
        // reference, 4 rows
        let o_full = run_moe(cx, &full, &x, &ids, &wts, 4)?;
        let mut refy = vec![0f64; 4 * 5120];
        for r in 0..4 { for j in 0..top_k { let e = ids[r * top_k + j] as usize; let v = ref_expert(&x[r * 5120..(r + 1) * 5120], &ex[e], wts[r * top_k + j], 10.0); for c in 0..5120 { refy[r * 5120 + c] += v[c]; } } }
        check(&format!("moe {label} rows=4 full slices vs f64 chain"), rel_l2(&o_full, &refy), 2e-4, fails, lines);
        // TP splits: sum of the ranks' partials equals the full result
        for tp in [2usize, 4, 8] {
            let mut sum = vec![0f32; 4 * 5120];
            for rank in 0..tp {
                let mw = moe_weight(&ex, 24, top_k, &|e| { let b = tp_bounds(tp, e); (b[rank], b[rank + 1]) });
                let o = run_moe(cx, &mw, &x, &ids, &wts, 4)?;
                for (s, v) in sum.iter_mut().zip(&o) { *s += v; }
            }
            let rf: Vec<f64> = o_full.iter().map(|v| *v as f64).collect();
            check(&format!("moe {label} TP{tp} inner-slice partials summed vs full"), rel_l2(&sum, &rf), 2e-5, fails, lines);
        }
        // batch invariance: every row of the N-row call equals its 1-row call (24 rows: experts with >16 rows use the MT2 path; 40 rows: chunked)
        for rows in [24usize, 40] {
            let (x, ids, wts) = mk(rows);
            let big = run_moe(cx, &full, &x, &ids, &wts, rows)?;
            let mut bad = 0;
            for r in 0..rows { let one = run_moe(cx, &full, &x[r * 5120..], &ids[r * top_k..], &wts[r * top_k..], 1)?; if !one.iter().zip(&big[r * 5120..(r + 1) * 5120]).all(|(a, b)| a.to_bits() == b.to_bits()) { bad += 1; } }
            lines.push(format!("  {} moe {label} batch invariance: each of {rows} rows bit-equal to its 1-row call ({bad} differ)", if bad == 0 { "ok  " } else { "FAIL" }));
            if bad > 0 { fails.push(format!("moe {label} invariance {rows}")); }
        }
    }
    // the shared expert as a one-expert table, ids 0 weight 1.0
    let sh = Ex { l: [fx.lin(cx, "layers.24.ffn.shared_experts.w1", true)?, fx.lin(cx, "layers.24.ffn.shared_experts.w2", true)?, fx.lin(cx, "layers.24.ffn.shared_experts.w3", true)?] };
    let mw = moe_weight(std::slice::from_ref(&sh), 24, 1, &|_| (0, 18));
    let x: Vec<f32> = (0..3 * 5120).map(|_| rng.f()).collect();
    let o = run_moe(cx, &mw, &x, &[0, 0, 0], &[1.0, 1.0, 1.0], 3)?;
    let mut refy = vec![0f64; 3 * 5120];
    for r in 0..3 { let v = ref_expert(&x[r * 5120..(r + 1) * 5120], &sh, 1.0, 10.0); refy[r * 5120..(r + 1) * 5120].copy_from_slice(&v); }
    check("moe shared expert K4 as a 1-expert table, rows=3 vs f64 chain", rel_l2(&o, &refy), 2e-4, fails, lines);
    Ok(())
}

// ---------------------------------------------------------------- plain -----------------------------------------------------
fn plain_battery(cx: &Cx, rng: &mut Rng, fails: &mut Vec<String>, lines: &mut Vec<String>) -> Result<()> {
    let (n, k) = (512usize, 1024usize);
    for (dt, code) in [("BF16", 0u8), ("F16", 1u8)] {
        let wv: Vec<u16> = (0..n * k).map(|_| if code == 0 { half::bf16::from_f32(rng.f()).to_bits() } else { half::f16::from_f32(rng.f()).to_bits() }).collect();
        let wf: Vec<f32> = wv.iter().map(|b| if code == 0 { bf2f(*b) } else { h2f(*b) }).collect();
        let dw = cx.up(&wv)?;
        let mk = |shard: Shard| DenseWeight { key: "plain".into(), class: TensorClass::GateWeight, layer: Some(0), sub: None, format: LinearFormat::Plain(dt.into()), out: n as u64, inn: k as u64, shard,
            tensors: vec![DevTensor { role: Role::Weight, ptr: DevPtr(dp(&dw)), dtype: dt.into(), shape: vec![n as u64, k as u64], bytes: (n * k * 2) as u64 }] };
        let x: Vec<f32> = (0..13 * k).map(|_| rng.f()).collect();
        let mut first: Option<Vec<f32>> = None;
        for rows in [1usize, 5, 13] {
            let dx = cx.up(&x[..rows * k])?; let dy = cx.zeros::<f32>(rows * n)?;
            PlainWeights.dense(&cx.l(), &mk(Shard::Whole), &Act { ptr: DevPtr(dp(&dx)), rows, cols: k, ld: k, dtype: ActDtype::F32 },
                &Act { ptr: DevPtr(dp(&dy)), rows, cols: n, ld: n, dtype: ActDtype::F32 }, Accum::Overwrite, Scratch::default())?;
            let y = cx.dl(&dy)?;
            let r: Vec<f64> = (0..rows * n).map(|i| { let (m, j) = (i / n, i % n); (0..k).map(|c| x[m * k + c] as f64 * wf[j * k + c] as f64).sum() }).collect();
            check(&format!("plain {dt} rows={rows}"), rel_l2(&y, &r), 1e-5, fails, lines);
            if rows == 1 { first = Some(y[..n].to_vec()); } else {
                let same = y[..n].iter().zip(first.as_ref().unwrap()).all(|(a, b)| a.to_bits() == b.to_bits());
                lines.push(format!("  {} plain {dt} batch invariance row 0 of rows={rows}", if same { "ok  " } else { "FAIL" })); if !same { fails.push(format!("plain inv {dt}")); }
            }
        }
        // out slice
        let dx = cx.up(&x[..2 * k])?; let dy = cx.zeros::<f32>(2 * 128)?;
        PlainWeights.dense(&cx.l(), &mk(Shard::Out(128..256)), &Act { ptr: DevPtr(dp(&dx)), rows: 2, cols: k, ld: k, dtype: ActDtype::F32 },
            &Act { ptr: DevPtr(dp(&dy)), rows: 2, cols: 128, ld: 128, dtype: ActDtype::F32 }, Accum::Overwrite, Scratch::default())?;
        let y = cx.dl(&dy)?;
        let r: Vec<f64> = (0..256).map(|i| { let (m, j) = (i / 128, 128 + i % 128); (0..k).map(|c| x[m * k + c] as f64 * wf[j * k + c] as f64).sum() }).collect();
        check(&format!("plain {dt} out slice 128..256"), rel_l2(&y, &r), 1e-5, fails, lines);
    }
    // gather (vocab slice 100..300 of a 512-row table)
    let (v, d) = (512usize, 256usize);
    let tab: Vec<u16> = (0..v * d).map(|_| half::bf16::from_f32(rng.f()).to_bits()).collect();
    let dt = cx.up(&tab)?;
    let w = DenseWeight { key: "embed".into(), class: TensorClass::Embed, layer: None, sub: None, format: LinearFormat::Plain("BF16".into()), out: v as u64, inn: d as u64, shard: Shard::Out(100..300),
        tensors: vec![DevTensor { role: Role::Weight, ptr: DevPtr(dp(&dt)), dtype: "BF16".into(), shape: vec![v as u64, d as u64], bytes: (v * d * 2) as u64 }] };
    let ids = vec![5i32, 100, 250, 299, 300, 450];
    let di = cx.up(&ids)?; let dy = cx.zeros::<u16>(6 * d)?;
    PlainWeights.gather(&cx.l(), &w, DevPtr(dp(&di)), 6, &Act { ptr: DevPtr(dp(&dy)), rows: 6, cols: d, ld: d, dtype: ActDtype::Bf16 })?;
    let y = cx.dl(&dy)?;
    let ok = ids.iter().enumerate().all(|(r, &id)| (0..d).all(|c| y[r * d + c] == if (100..300).contains(&id) { tab[id as usize * d + c] } else { 0 }));
    lines.push(format!("  {} plain vocab-parallel gather (ids in / out of 100..300)", if ok { "ok  " } else { "FAIL" })); if !ok { fails.push("gather".into()); }
    Ok(())
}

// ---------------------------------------------------------------- byte-floor bench ------------------------------------------------
pub(super) fn timed(cx: &Cx, iters: usize, mut f: impl FnMut(usize) -> Result<()>) -> Result<f64> {
    for i in 0..3 { f(i)?; }
    cx.sync()?;
    let (mut a, mut b): (sys::CUevent, sys::CUevent) = (std::ptr::null_mut(), std::ptr::null_mut());
    unsafe { sys::cuEventCreate(&mut a, 0); sys::cuEventCreate(&mut b, 0); sys::cuEventRecord(a, cx.stream as sys::CUstream); }
    for i in 0..iters { f(i)?; }
    let mut ms = 0f32;
    unsafe { sys::cuEventRecord(b, cx.stream as sys::CUstream); sys::cuEventSynchronize(b); sys::cuEventElapsedTime(&mut ms, a, b); sys::cuEventDestroy_v2(a); sys::cuEventDestroy_v2(b); }
    Ok(ms as f64 * 1000.0 / iters as f64)
}

pub fn run(args: &[String]) -> Result<()> {
    // PK3: with a pack and a REFb fixture, the numeric probe of --dsv41-prefill-bf16-partials instead of this battery
    if crate::dsv41::weights::partials::probe::wanted(args) { return crate::dsv41::weights::partials::probe::run(args).map(|_| ()); }
    let cx = Cx::new()?;
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let (mut fails, mut lines) = (vec![], vec![]);
    lines.push(format!("gpu_dsv41 kernel build {} loaded; stream {:#x}", env!("KERNEL_BUILD_ID"), cx.stream));
    let nocheck = crate::arg_value(args, "--dsv41-kernel-timing").map_or(false, |v| v.split(',').any(|x| x.trim() == "nocheck"));
    if nocheck { lines.push("NOTE: nocheck: every correctness battery SKIPPED (timing only)".into()); }
    if !nocheck {
    lines.push("== dense EXL3 (synthetic trellis, K 1.5..8, host decode + f64 reference chain) ==".into());
    dense_battery(&cx, &mut rng, &mut fails, &mut lines)?;
    lines.push("== PK4 dense decode: grouped calls (segments), f64, invariance ==".into());
    // its own random stream: the later batteries keep the inputs (and margins) they had before this battery existed
    dense_group_battery(&cx, &mut Rng(0xD41D_D0EC_5EED_0001), &mut fails, &mut lines)?;
    lines.push("== PD1 producer-side rotation (rot_out, pre-rotated inputs, normrot), mixed-width groups, the paced L2 prefetch ==".into());
    super::probe_pd1::battery(&cx, &mut Rng(0x9D1_0000_D41D_0001), &mut fails, &mut lines)?;
    lines.push("== KE2 dense wide (> 32 rows): trellis-source vs reconstruct arm, slices, f64 ==".into());
    super::probe_moe::dense_wide_battery(&cx, &mut rng, &mut fails, &mut lines)?;
    lines.push("== plain BF16 / F16 + vocab gather ==".into());
    plain_battery(&cx, &mut rng, &mut fails, &mut lines)?;
    lines.push("== SX1 MoE experts with a different EXL3 width per matrix (synthetic, every served path, f64) ==".into());
    // its own random stream: the later batteries keep their inputs
    super::probe_moe::mixed_width_battery(&cx, &mut Rng(0x5A1_0000_D41_0029), &mut fails, &mut lines)?;
    match crate::arg_value(args, "--model-dir") {
        Some(d) => {
            let fx = Fx::open(d)?;
            lines.push("== real EAQ v1 tensors (layers 20 / 24, head) ==".into());
            real_dense(&cx, &fx, &mut rng, &mut fails, &mut lines)?;
            moe_battery(&cx, &fx, &mut rng, &mut fails, &mut lines)?;
            lines.push("== KE2 served MoE paths (decode kernels, fused shared expert) on real experts ==".into());
            super::probe_moe::battery(&cx, &fx, &mut rng, &mut fails, &mut lines)?;
        }
        None => lines.push("NOTE: no --model-dir fixture: the REAL-tensor dense + MoE cases were SKIPPED (the synthetic dense cases and plain ran)".into()),
    }
    }
    lines.push("== timing (--dsv41-kernel-timing) ==".into());
    let mut out = |l: String| println!("{l}");      // the probe's one line sink (correctness lines, then timing lines as they come)
    for l in lines.drain(..) { out(l); }
    super::probe_bench::run(&cx, &mut rng, &mut out, &super::probe_bench::Sel::parse(crate::arg_value(args, "--dsv41-kernel-timing")))?;
    if !fails.is_empty() { bail!("{} check(s) failed: {}", fails.len(), fails.join("; ")); }
    println!("DSV41_KE_OK");
    Ok(())
}
