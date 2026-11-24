//! DSV4.1 native MXFP4 experts (leg KN2): `--probe-dsv41-kernel --dsv41-kernel-set mxfp4`.
//!
//! 1. **Exactness** on synthetic layers (384 x top-6 trunk and 128 x top-3 DSpark, whole experts and
//!    inner slices of a non-zero rank) through the served entry (`moe_view`, and `moe` through a real
//!    `MoeWeight` for the loader contract): act_quant codes, gate / up, h codes, every route's w2
//!    output and the combined y are compared with `reference` BIT FOR BIT. The h codes may differ
//!    only where host and CUDA `expf` differ by an ulp across a bf16 boundary (counted, bounded).
//! 2. **Batch invariance**: every row of an N-row call equals the same row run alone, bit for bit
//!    (this covers all three row classes: 1-2 / 3-16 / 17-32 rows).
//! 3. **Byte floor**: the served launch sequence timed at 1 / 2 / 6 / 12 / 24 rows at the slice
//!    widths of TP1 / TP2 / TP4 / TP8 (host-timed over many calls, no graph: an upper bound).
//!
//! The last line is `DSV41_KERNEL_MXFP4_OK` only when every check holds.

use super::launch::{self, Debug, PackedView};
use super::pack::ExpertGeom;
use super::reference as rf;
use crate::dsv41::weights::{Act, ActDtype, DevPtr, Launch, Route, Scratch};
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use std::sync::Arc;
use std::time::Instant;

const DIM: usize = 5120;
const INTER: usize = 2304;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn normal(&mut self) -> f64 { let (u1, u2) = (self.unit().max(1e-12), self.unit()); (-2.0 * u1.ln()).sqrt() * (6.283185307179586 * u2).cos() }
}

/// A synthetic expert in checkpoint layout (random codes, realistic ue8m0 scales 2^-15..2^-1).
struct OwnedExpert { w1: Vec<u8>, s1: Vec<u8>, w3: Vec<u8>, s3: Vec<u8>, w2: Vec<u8>, s2: Vec<u8> }
impl OwnedExpert {
    fn gen(r: &mut Rng) -> Self {
        let mut b = |n: usize| (0..n).map(|_| r.next() as u8).collect::<Vec<u8>>();
        let (w1, w3, w2) = (b(INTER * DIM / 2), b(INTER * DIM / 2), b(DIM * INTER / 2));
        let mut s = |n: usize| (0..n).map(|_| (112 + r.below(15)) as u8).collect::<Vec<u8>>();
        OwnedExpert { w1, w3, w2, s1: s(INTER * DIM / 32), s3: s(INTER * DIM / 32), s2: s(DIM * INTER / 32) }
    }
    fn raw(&self) -> rf::RawExpert<'_> { rf::RawExpert { w1: &self.w1, s1: &self.s1, w3: &self.w3, s3: &self.s3, w2: &self.w2, s2: &self.s2 } }
}

fn dptr<T>(s: &CudaSlice<T>) -> u64 { *s.device_ptr() }

struct DevLayer { _w13: CudaSlice<u8>, _w2: CudaSlice<u8>, view: PackedView }

/// Pack `n_alloc` experts (generated expert `e % gen.len()` at id e) at one slice and upload.
fn upload(dev: &Arc<CudaDevice>, gen: &[OwnedExpert], n_alloc: usize, geom: &ExpertGeom) -> Result<DevLayer> {
    let raws: Vec<rf::RawExpert> = gen.iter().map(|e| e.raw()).collect();
    let (b13, b2) = super::pack_layer(&raws, geom)?;
    let (e13, e2) = (geom.w13_bytes(), geom.w2_bytes());
    let mut h13 = vec![0u8; e13 * n_alloc];
    let mut h2 = vec![0u8; e2 * n_alloc];
    for e in 0..n_alloc {
        let g = e % gen.len();
        h13[e * e13..(e + 1) * e13].copy_from_slice(&b13[g * e13..(g + 1) * e13]);
        h2[e * e2..(e + 1) * e2].copy_from_slice(&b2[g * e2..(g + 1) * e2]);
    }
    let w13 = dev.htod_sync_copy(&h13)?;
    let w2 = dev.htod_sync_copy(&h2)?;
    let view = PackedView { w13: DevPtr(dptr(&w13)), e13: e13 as u64, mt13: geom.w13_mtile() as u64, w2: DevPtr(dptr(&w2)), e2: e2 as u64,
                            mt2: geom.w2_mtile() as u64, n_local: n_alloc, e_first: 0, n_experts: n_alloc, dim: DIM, i0: geom.i0, ns: geom.ns,
                            round_bf16: geom.is_whole(), wide_bf16: geom.is_whole(), limit: 10.0 };
    Ok(DevLayer { _w13: w13, _w2: w2, view })
}

fn gen_x(r: &mut Rng, rows: usize) -> Vec<f32> {
    (0..rows * DIM).map(|_| { let mut v = r.normal(); if r.below(64) == 0 { v *= 30.0; } rf::bf16r(v as f32) }).collect()
}
/// Distinct top-k per row; renormalized weights x 1.5 (the Gate's output).
fn gen_routes(r: &mut Rng, rows: usize, top_k: usize, n_exp: usize) -> (Vec<i32>, Vec<f32>) {
    let (mut ids, mut ws) = (Vec::new(), Vec::new());
    for _ in 0..rows {
        let mut pick: Vec<i32> = Vec::new();
        while pick.len() < top_k { let e = r.below(n_exp) as i32; if !pick.contains(&e) { pick.push(e); } }
        let w: Vec<f32> = (0..top_k).map(|_| 0.05 + 0.95 * r.unit() as f32).collect();
        let s: f32 = w.iter().sum();
        ids.extend(pick);
        ws.extend(w.iter().map(|v| v / (s + 1e-20) * 1.5));
    }
    (ids, ws)
}

/// A blocking stream (AGENTS §2.1): CU_STREAM_DEFAULT synchronizes with the legacy stream cudarc copies on.
fn blocking_stream() -> Result<u64> {
    use cudarc::driver::sys;
    let mut s: sys::CUstream = std::ptr::null_mut();
    let r = unsafe { sys::cuStreamCreate(&mut s, sys::CUstream_flags::CU_STREAM_DEFAULT as u32) };
    ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuStreamCreate: {r:?}");
    Ok(s as u64)
}
fn sync(stream: u64) -> Result<()> {
    let r = unsafe { cudarc::driver::sys::cuStreamSynchronize(stream as cudarc::driver::sys::CUstream) };
    ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "stream sync: {r:?}");
    Ok(())
}

struct Bufs { x: CudaSlice<u16>, ids: CudaSlice<i32>, rw: CudaSlice<f32>, y: CudaSlice<f32>, scr: CudaSlice<u8>, gu: CudaSlice<u16>, ro: CudaSlice<f32> }

/// Copy `src` into the front of `dst` (the rest zero / default).
fn put<T: cudarc::driver::DeviceRepr + Default + Copy>(dev: &Arc<CudaDevice>, src: &[T], dst: &mut CudaSlice<T>) -> Result<()> {
    let mut v = src.to_vec();
    v.resize(dst.len(), T::default());
    dev.htod_sync_copy_into(&v, dst)?;
    Ok(())
}

fn call(stream: u64, v: &PackedView, top_k: usize, b: &Bufs, rows: usize, dbg: bool) -> Result<crate::dsv41::weights::RowClass> {
    call_ids(stream, v, top_k, b, rows, dbg, dptr(&b.ids), dptr(&b.rw))
}
#[allow(clippy::too_many_arguments)]
fn call_ids(stream: u64, v: &PackedView, top_k: usize, b: &Bufs, rows: usize, dbg: bool, ids: u64, rw: u64) -> Result<crate::dsv41::weights::RowClass> {
    let l = Launch { stream, pdl: false };
    let x = Act { ptr: DevPtr(dptr(&b.x)), rows, cols: DIM, ld: DIM, dtype: ActDtype::Bf16 };
    let y = Act { ptr: DevPtr(dptr(&b.y)), rows, cols: DIM, ld: DIM, dtype: ActDtype::F32 };
    let r = Route { ids: DevPtr(ids), weights: DevPtr(rw), top_k };
    let d = if dbg { Debug { gate_up: dptr(&b.gu), route_out: dptr(&b.ro) } } else { Debug::default() };
    super::moe_view(&l, v, top_k, &x, &r, &y, false, Scratch { ptr: DevPtr(dptr(&b.scr)), bytes: b.scr.len() }, d)
}

fn bf16_bits(v: &[f32]) -> Vec<u16> { v.iter().map(|&f| half::bf16::from_f32(f).to_bits()).collect() }

/// Exactness + batch invariance of one (layer, slice) configuration. Returns (lines, failures).
fn check_config(dev: &Arc<CudaDevice>, stream: u64, gen: &[OwnedExpert], n_exp: usize, top_k: usize, geom: &ExpertGeom,
                rows_list: &[usize], r: &mut Rng) -> Result<(Vec<String>, usize)> {
    let layer = upload(dev, gen, n_exp, geom)?;
    let v = &layer.view;
    let mr = launch::GROUPED_MAX_ROWS;
    let mut b = Bufs {
        x: dev.htod_sync_copy(&vec![0u16; mr * DIM])?, ids: dev.htod_sync_copy(&vec![0i32; mr * top_k])?, rw: dev.htod_sync_copy(&vec![0f32; mr * top_k])?,
        y: dev.htod_sync_copy(&vec![0f32; mr * DIM])?, scr: dev.htod_sync_copy(&vec![0u8; launch::ScratchPlan::of(mr, top_k, DIM, geom.ns).bytes])?,
        gu: dev.htod_sync_copy(&vec![0u16; mr * top_k * 2 * geom.ns])?, ro: dev.htod_sync_copy(&vec![0f32; mr * top_k * DIM])?,
    };
    let (mut lines, mut fails) = (Vec::new(), 0);
    for &rows in rows_list {
        let sp = launch::ScratchPlan::of(rows, top_k, DIM, geom.ns);   // the offsets the call used
        let x = gen_x(r, rows);
        let (mut ids, rw) = gen_routes(r, rows, top_k, n_exp);
        if rows >= 17 { for row in 1..rows { for k in 0..top_k { ids[row * top_k + k] = ids[k]; } } }   // one expert routed by every row
        put(dev, &bf16_bits(&x), &mut b.x)?;
        put(dev, &ids, &mut b.ids)?;
        put(dev, &rw, &mut b.rw)?;
        let class = call(stream, v, top_k, &b, rows, true)?;
        sync(stream)?;
        let nr = rows * top_k;
        let scr = dev.dtoh_sync_copy(&b.scr)?;
        let y = dev.dtoh_sync_copy(&b.y)?;
        let gu = dev.dtoh_sync_copy(&b.gu)?;
        let ro = dev.dtoh_sync_copy(&b.ro)?;
        let (xq, xs) = (&scr[sp.xq..], &scr[sp.xs..]);
        // act_quant
        let mut bad_aq = 0usize;
        let (mut cq, mut cs) = (vec![0u8; DIM], vec![0u8; DIM / 32]);
        for row in 0..rows {
            rf::act_quant(&x[row * DIM..(row + 1) * DIM], &mut cq, &mut cs);
            bad_aq += (0..DIM).filter(|&i| cq[i] != xq[row * DIM + i]).count() + (0..DIM / 32).filter(|&i| cs[i] != xs[row * sp.ldxs + i]).count();
        }
        // per route: gate / up and h codes vs the host (FC1), w2 output from the GPU's own h (FC2)
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let per = nr.div_ceil(threads);
        let ns = geom.ns;
        let results: Vec<(usize, usize, usize)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads).map(|t| {
                let (gen, ids, rw, gu, ro, scr, xq, xs, sp) = (gen, &ids, &rw, &gu, &ro, &scr, xq, xs, &sp);
                sc.spawn(move || {
                    let (mut bgu, mut bh, mut bro) = (0, 0, 0);
                    for route in t * per..((t + 1) * per).min(nr) {
                        let row = route / top_k;
                        let ex = gen[ids[route] as usize % gen.len()].raw();
                        let rr = rf::fc1(&ex, DIM, geom.i0, ns, &xq[row * DIM..], &xs[row * sp.ldxs..], rw[route], 10.0);
                        for i in 0..ns {
                            bgu += (half::bf16::from_f32(rr.gate[i]).to_bits() != gu[route * 2 * ns + i]) as usize;
                            bgu += (half::bf16::from_f32(rr.up[i]).to_bits() != gu[route * 2 * ns + ns + i]) as usize;
                        }
                        let (ghq, ghs) = (&scr[sp.hq + route * ns..sp.hq + (route + 1) * ns], &scr[sp.hs + route * sp.ldhs..sp.hs + route * sp.ldhs + ns / 32]);
                        bh += (0..ns).filter(|&i| rr.hq[i] != ghq[i]).count() + (0..ns / 32).filter(|&i| rr.hs[i] != ghs[i]).count();
                        let out = rf::fc2(&ex, DIM, INTER, geom.i0, ns, ghq, ghs, geom.is_whole());
                        bro += (0..DIM).filter(|&o| out[o].to_bits() != ro[route * DIM + o].to_bits()).count();
                    }
                    (bgu, bh, bro)
                })
            }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let (bgu, bh, bro) = results.iter().fold((0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        // the ordered combine from the (checked) route outputs
        let mut bad_y = 0usize;
        for row in 0..rows {
            let outs: Vec<&[f32]> = (0..top_k).map(|k| &ro[(row * top_k + k) * DIM..(row * top_k + k + 1) * DIM]).collect();
            let yy = rf::combine(&ids[row * top_k..(row + 1) * top_k], &outs, DIM);
            bad_y += (0..DIM).filter(|&o| yy[o].to_bits() != y[row * DIM + o].to_bits()).count();
        }
        // h codes may differ only through the host / CUDA expf ulp (bounded: <= 1 per 2 routes)
        let ok = bad_aq == 0 && bgu == 0 && bro == 0 && bad_y == 0 && bh * 2 <= nr;
        fails += !ok as usize;
        lines.push(format!("{n_exp}x{top_k} slice {}+{} rows {rows:2} ({}): act_quant {bad_aq}, gate/up {bgu} of {}, h codes {bh} of {}, route out {bro} of {}, y {bad_y} of {}  {}",
                           geom.i0, ns, class.name(), nr * 2 * ns, nr * (ns + ns / 32), nr * DIM, rows * DIM, if ok { "OK" } else { "FAIL" }));
        // batch invariance: each row alone
        if rows > 1 {
            let mut bad_bi = 0usize;
            for row in 0..rows {
                put(dev, &bf16_bits(&x[row * DIM..(row + 1) * DIM]), &mut b.x)?;
                put(dev, &ids[row * top_k..(row + 1) * top_k], &mut b.ids)?;
                put(dev, &rw[row * top_k..(row + 1) * top_k], &mut b.rw)?;
                call(stream, v, top_k, &b, 1, false)?;
                sync(stream)?;
                let y1 = dev.dtoh_sync_copy(&b.y)?;
                bad_bi += (0..DIM).filter(|&o| y1[o].to_bits() != y[row * DIM + o].to_bits()).count();
            }
            fails += (bad_bi != 0) as usize;
            lines.push(format!("    batch invariance, {rows} rows vs each row alone: {bad_bi} of {} differ  {}", rows * DIM, if bad_bi == 0 { "OK" } else { "FAIL" }));
        }
    }
    Ok((lines, fails))
}

/// The loader contract: a `MoeWeight` built the documented way serves through the trait.
fn check_trait(dev: &Arc<CudaDevice>, stream: u64, gen: &[OwnedExpert], r: &mut Rng) -> Result<(String, usize)> {
    use crate::dsv41::census::classify::{Role, TensorClass};
    use crate::dsv41::census::LinearFormat;
    use crate::dsv41::weights::{DenseWeight, DevTensor, Dsv41Weights, MoeWeight, Shard};
    let geom = ExpertGeom::slice(DIM, INTER, 1, 4)?;
    let n = 16;
    let layer = upload(dev, gen, n, &geom)?;
    let v = layer.view;
    let fmt = LinearFormat::Fp4 { block: 32, scale: "F8_E8M0".into(), scale2: false };
    let dw = |e: usize, which: usize| -> DenseWeight {
        let (class, out, inn) = match which { 0 => (TensorClass::ExpertW1, INTER, DIM), 1 => (TensorClass::ExpertW2, DIM, INTER), _ => (TensorClass::ExpertW3, INTER, DIM) };
        let rng = geom.i0 as u64..(geom.i0 + geom.ns) as u64;
        let shard = if which == 1 { Shard::In(rng) } else { Shard::Out(rng) };
        let tensors = match which {
            0 => vec![DevTensor { role: Role::Weight, ptr: DevPtr(v.w13.0 + e as u64 * v.e13), dtype: super::DT_W13.into(), shape: vec![v.e13], bytes: v.e13 }],
            1 => vec![DevTensor { role: Role::Weight, ptr: DevPtr(v.w2.0 + e as u64 * v.e2), dtype: super::DT_W2.into(), shape: vec![v.e2], bytes: v.e2 }],
            _ => vec![],
        };
        DenseWeight { key: format!("layers.1.ffn.experts.{e}.w{}", [1, 2, 3][which]), class, layer: Some(1), sub: Some(e as u16), format: fmt.clone(),
                      out: out as u64, inn: inn as u64, shard, tensors }
    };
    let mw = MoeWeight { layer: 1, n_experts: n, local: 0..n, top_k: 6, dim: DIM, inter: INTER, swiglu_limit: 10.0,
                         experts: (0..n).map(|e| [dw(e, 0), dw(e, 1), dw(e, 2)]).collect() };
    // PK2: the trait's wide path (rows > 32: act_quant_p, plan, FC1 / FC2 on 128-route tiles, combine) against the same
    // trait called in <= 32-row pieces (the grouped kernels), bit for bit: exercises the Rust launch sequence of `wide`.
    let wide = {
        let rows = 300;
        let x = gen_x(r, rows);
        let (ids, rw) = gen_routes(r, rows, 6, n);
        let bx = dev.htod_sync_copy(&bf16_bits(&x))?;
        let bids = dev.htod_sync_copy(&ids)?;
        let brw = dev.htod_sync_copy(&rw)?;
        let y_init: Vec<f32> = (0..rows * DIM).map(|i| (i % 5) as f32 * 0.5).collect();
        let (bw, bg) = (dev.htod_sync_copy(&y_init)?, dev.htod_sync_copy(&y_init)?);
        let scr = dev.htod_sync_copy(&vec![0u8; super::scratch_bytes(&mw, rows)])?;
        let be = crate::dsv41::weights::native::NativeWeights;
        let call = |r0: usize, n: usize, y: u64| be.moe(&Launch { stream, pdl: true }, &mw,
            &Act { ptr: DevPtr(dptr(&bx) + (r0 * DIM * 2) as u64), rows: n, cols: DIM, ld: DIM, dtype: ActDtype::Bf16 },
            &Route { ids: DevPtr(dptr(&bids) + (r0 * 6 * 4) as u64), weights: DevPtr(dptr(&brw) + (r0 * 6 * 4) as u64), top_k: 6 },
            &Act { ptr: DevPtr(y + (r0 * DIM * 4) as u64), rows: n, cols: DIM, ld: DIM, dtype: ActDtype::F32 }, Scratch { ptr: DevPtr(dptr(&scr)), bytes: scr.len() });
        let cw = call(0, rows, dptr(&bw))?;
        let mut r0 = 0;
        while r0 < rows { let n = (rows - r0).min(32); call(r0, n, dptr(&bg))?; r0 += n; }
        sync(stream)?;
        let (yw, yg) = (dev.dtoh_sync_copy(&bw)?, dev.dtoh_sync_copy(&bg)?);
        let bad = yw.iter().zip(&yg).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        (format!("trait entry wide (300 rows, class {}) == the trait in 32-row grouped pieces: {bad} of {} differ  {}", cw.name(), rows * DIM, if bad == 0 { "OK" } else { "FAIL" }), (bad != 0) as usize)
    };
    let rows = 6;
    let x = gen_x(r, rows);
    let (ids, rw) = gen_routes(r, rows, 6, n);
    let bx = dev.htod_sync_copy(&bf16_bits(&x))?;
    let bids = dev.htod_sync_copy(&ids)?;
    let brw = dev.htod_sync_copy(&rw)?;
    let y_init: Vec<f32> = (0..rows * DIM).map(|i| (i % 7) as f32 * 0.25).collect();   // the trait ADDS into y
    let by = dev.htod_sync_copy(&y_init)?;
    let scr = dev.htod_sync_copy(&vec![0u8; super::scratch_bytes(&mw, rows)])?;
    let be = crate::dsv41::weights::native::NativeWeights;
    let class = be.moe(&Launch { stream, pdl: true }, &mw, &Act { ptr: DevPtr(dptr(&bx)), rows, cols: DIM, ld: DIM, dtype: ActDtype::Bf16 },
                       &Route { ids: DevPtr(dptr(&bids)), weights: DevPtr(dptr(&brw)), top_k: 6 },
                       &Act { ptr: DevPtr(dptr(&by)), rows, cols: DIM, ld: DIM, dtype: ActDtype::F32 }, Scratch { ptr: DevPtr(dptr(&scr)), bytes: scr.len() })?;
    sync(stream)?;
    let y = dev.dtoh_sync_copy(&by)?;
    // host: y_init + ordered sum of the routes (host h; an expf ulp would show here as a few elements)
    let mut bad = 0usize;
    for row in 0..rows {
        let (mut q, mut s) = (vec![0u8; DIM], vec![0u8; DIM / 32]);
        rf::act_quant(&x[row * DIM..(row + 1) * DIM], &mut q, &mut s);
        let outs: Vec<Vec<f32>> = (0..6).map(|k| {
            let ex = gen[ids[row * 6 + k] as usize % gen.len()].raw();
            let rr = rf::fc1(&ex, DIM, geom.i0, geom.ns, &q, &s, rw[row * 6 + k], 10.0);
            rf::fc2(&ex, DIM, INTER, geom.i0, geom.ns, &rr.hq, &rr.hs, false)
        }).collect();
        let refs: Vec<&[f32]> = outs.iter().map(|o| o.as_slice()).collect();
        let yy = rf::combine(&ids[row * 6..(row + 1) * 6], &refs, DIM);
        bad += (0..DIM).filter(|&o| (y_init[row * DIM + o] + yy[o]).to_bits() != y[row * DIM + o].to_bits()).count();
    }
    let ok = bad <= DIM / 1000;
    Ok((format!("trait entry (MoeWeight with mx1 tensors, inner slice 576+576, accumulate into y, PDL launch): class {}, {bad} of {} differ from the host  {}\n{}",
                class.name(), rows * DIM, if ok { "OK" } else { "FAIL" }, wide.0), (!ok) as usize + wide.1))
}

/// The byte-floor table: GB/s of the served launch sequence over the distinct experts' bytes.
fn bench(dev: &Arc<CudaDevice>, stream: u64, gen: &[OwnedExpert], n_exp: usize, top_k: usize, r: &mut Rng) -> Result<Vec<String>> {
    let mut lines = vec![format!("| experts x top-k | slice (TP) | rows | distinct experts | MB read | us / call | GB/s | of 238 |"),
                         "|---|---|---:|---:|---:|---:|---:|---:|".to_string()];
    for (n, tp) in [(1, "TP1"), (2, "TP2"), (4, "TP4"), (8, "TP8")] {
        let geom = ExpertGeom::slice(DIM, INTER, 0, n)?;
        let layer = upload(dev, gen, n_exp, &geom)?;
        let mr = launch::GROUPED_MAX_ROWS;
        let sets = 64;
        let b = Bufs {
            x: dev.htod_sync_copy(&bf16_bits(&gen_x(r, mr)))?, ids: dev.htod_sync_copy(&vec![0i32; mr * top_k])?, rw: dev.htod_sync_copy(&vec![0f32; mr * top_k])?,
            y: dev.htod_sync_copy(&vec![0f32; mr * DIM])?, scr: dev.htod_sync_copy(&vec![0u8; launch::ScratchPlan::of(mr, top_k, DIM, geom.ns).bytes])?,
            gu: dev.htod_sync_copy(&vec![0u16; 1])?, ro: dev.htod_sync_copy(&vec![0f32; 1])?,
        };
        for rows in [1usize, 2, 6, 12, 24] {
            // the verify rows of one lane share half their experts with the previous row (T0 §5.3)
            let mut all_ids = Vec::new();
            let mut all_w = Vec::new();
            let mut distinct = 0usize;
            for _ in 0..sets {
                let (mut ids, rw) = gen_routes(r, rows, top_k, n_exp);
                for row in 1..rows {
                    if row % 6 == 0 { continue; }
                    let mut rowv: Vec<i32> = ids[(row - 1) * top_k + top_k / 2..row * top_k].to_vec();
                    while rowv.len() < top_k { let e = r.below(n_exp) as i32; if !rowv.contains(&e) { rowv.push(e); } }
                    ids[row * top_k..(row + 1) * top_k].copy_from_slice(&rowv);
                }
                let mut u = ids.clone(); u.sort(); u.dedup(); distinct += u.len();
                all_ids.push(ids); all_w.push(rw);
            }
            let dsets: Vec<(CudaSlice<i32>, CudaSlice<f32>)> = all_ids.iter().zip(&all_w).map(|(i, w)| Ok((dev.htod_sync_copy(i)?, dev.htod_sync_copy(w)?))).collect::<Result<_>>()?;
            let reps = 200;
            let run = |i: usize| -> Result<()> {
                let (di, dw) = &dsets[i % sets];
                call_ids(stream, &layer.view, top_k, &b, rows, false, dptr(di), dptr(dw)).map(|_| ())
            };
            for i in 0..8 { run(i)?; }
            sync(stream)?;
            let t0 = Instant::now();
            for i in 0..reps { run(i)?; }
            sync(stream)?;
            let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
            let dist = distinct as f64 / sets as f64;
            let mb = dist * geom.raw_bytes() as f64 / 1e6;
            let gbs = mb / us * 1e3;
            lines.push(format!("| {n_exp} x {top_k} | {} ({tp}) | {rows} | {dist:.1} | {mb:.2} | {us:.1} | {gbs:.1} | {:.2} |", geom.ns, gbs / 238.0));
        }
    }
    Ok(lines)
}

/// The host model of the shared expert (FP8 E4M3 weights, one ue8m0 per 32 x 32 block; KN1's
/// format): act_quant, per k-block dot (rounded toward zero past 24 bits, as GB10's e4m3 x e4m3
/// tensor core does), FP32 k-block sum, bf16; SwiGLU without a routing weight. Returns bf16 values.
fn shared_ref(x: &[f32], w: &[(Vec<u8>, Vec<u8>); 3], dim: usize, inter: usize) -> Vec<f32> {
    let fp8_dot = |wr: &[u8], ws_row: &[u8], q: &[u8], s: &[u8], nkb: usize| -> f32 {
        let mut acc = 0f32;
        for kb in 0..nkb {
            let mut sum = 0i64;
            for k in 0..32 { sum += rf::e4m3_int(wr[kb * 32 + k]) * rf::e4m3_int(q[kb * 32 + k]); }
            // units of 2^-18: reuse dot_rz (units 2^-10) with the exponent shifted by -8
            acc += rf::dot_rz(sum, ws_row[kb], (s[kb] as i32 - 8) as u8);
        }
        acc
    };
    let (mut q, mut sx) = (vec![0u8; dim], vec![0u8; dim / 32]);
    rf::act_quant(x, &mut q, &mut sx);
    let mut h = vec![0f32; inter];
    for i in 0..inter {
        let srow: Vec<u8> = (0..dim / 32).map(|kb| w[0].1[(i / 32) * (dim / 32) + kb]).collect();
        let g = rf::bf16r(fp8_dot(&w[0].0[i * dim..], &srow, &q, &sx, dim / 32));
        let srow3: Vec<u8> = (0..dim / 32).map(|kb| w[2].1[(i / 32) * (dim / 32) + kb]).collect();
        let u = rf::bf16r(fp8_dot(&w[2].0[i * dim..], &srow3, &q, &sx, dim / 32));
        h[i] = rf::swiglu_h(g, u, 1.0, 10.0);
    }
    let (mut hq, mut hs) = (vec![0u8; inter], vec![0u8; inter / 32]);
    rf::act_quant(&h, &mut hq, &mut hs);
    (0..dim).map(|o| {
        let srow: Vec<u8> = (0..inter / 32).map(|kb| w[1].1[(o / 32) * (inter / 32) + kb]).collect();
        rf::bf16r(fp8_dot(&w[1].0[o * inter..], &srow, &hq, &hs, inter / 32))
    }).collect()
}

/// Real weights against DeepSeek's own forward: a REFb layer dump's MoE rows (ffn_in, router,
/// ffn_out) with the checkpoint's experts (`tools/dsv41_kn2/moe_fixture.py`). Routed experts on the
/// GPU (whole experts = the reference's per-expert bf16), the shared expert on the host model, one
/// bf16 rounding; compared with the dumped ffn_out.
fn check_fixture(dev: &Arc<CudaDevice>, stream: u64, path: &std::path::Path) -> Result<(Vec<String>, usize)> {
    use crate::dsv41::st;
    let h = st::read_header(path)?;
    let shape = |n: &str| h.tensors.iter().find(|t| t.name == n).map(|t| t.shape.clone()).with_context(|| format!("{}: no {n}", path.display()));
    let (p, top_k) = { let sh = shape("ids")?; (sh[0] as usize, sh[1] as usize) };
    let xb = st::read_tensor(&h, "x")?;
    let x16: Vec<u16> = xb.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let x: Vec<f32> = x16.iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect();
    let ids: Vec<i32> = st::read_tensor(&h, "ids")?.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    let rw: Vec<f32> = st::read_tensor(&h, "w")?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let yref: Vec<u16> = st::read_tensor(&h, "y_ref")?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    ensure!(x.len() == p * DIM && rw.len() == p * top_k && yref.len() == p * DIM, "{}: shapes", path.display());
    // the routed experts, renumbered by rank (keeps the ascending-id order of the combine)
    let mut distinct = ids.clone(); distinct.sort(); distinct.dedup();
    let local: Vec<i32> = ids.iter().map(|e| distinct.binary_search(e).unwrap() as i32).collect();
    let owned: Vec<OwnedExpert> = distinct.iter().map(|e| -> Result<OwnedExpert> {
        let g = |m: &str, k: &str| st::read_tensor(&h, &format!("experts.{e}.{m}.{k}"));
        Ok(OwnedExpert { w1: g("w1", "weight")?, s1: g("w1", "scale")?, w3: g("w3", "weight")?, s3: g("w3", "scale")?, w2: g("w2", "weight")?, s2: g("w2", "scale")? })
    }).collect::<Result<_>>()?;
    let geom = ExpertGeom::whole(DIM, INTER);
    let layer = upload(dev, &owned, owned.len(), &geom)?;
    let mr = p.max(1);
    let b = Bufs {
        x: dev.htod_sync_copy(&x16)?, ids: dev.htod_sync_copy(&local)?, rw: dev.htod_sync_copy(&rw)?, y: dev.htod_sync_copy(&vec![0f32; mr * DIM])?,
        scr: dev.htod_sync_copy(&vec![0u8; super::scratch_bytes_for(p, top_k, DIM, INTER, owned.len())])?,
        gu: dev.htod_sync_copy(&vec![0u16; 1])?, ro: dev.htod_sync_copy(&vec![0f32; 1])?,
    };
    let class = call(stream, &layer.view, top_k, &b, p, false)?;
    sync(stream)?;
    let y = dev.dtoh_sync_copy(&b.y)?;
    let sw: [(Vec<u8>, Vec<u8>); 3] = [("w1"), ("w2"), ("w3")].map(|m| (st::read_tensor(&h, &format!("shared.{m}.weight")).unwrap_or_default(),
                                                                         st::read_tensor(&h, &format!("shared.{m}.scale")).unwrap_or_default()));
    ensure!(sw.iter().all(|(w, s)| !w.is_empty() && !s.is_empty()), "{}: shared expert tensors missing", path.display());
    let sh: Vec<Vec<f32>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..p).map(|row| { let (x, sw) = (&x, &sw); sc.spawn(move || shared_ref(&x[row * DIM..(row + 1) * DIM], sw, DIM, INTER)) }).collect();
        hs.into_iter().map(|j| j.join().unwrap()).collect()
    });
    let (mut equal, mut l2n, mut l2d, mut maxd) = (0usize, 0f64, 0f64, 0f32);
    for row in 0..p {
        for o in 0..DIM {
            let v = half::bf16::from_f32(y[row * DIM + o] + sh[row][o]);
            let r = half::bf16::from_bits(yref[row * DIM + o]);
            equal += (v.to_bits() == r.to_bits()) as usize;
            let d = (v.to_f32() - r.to_f32()) as f64;
            l2n += d * d; l2d += (r.to_f32() as f64).powi(2);
            maxd = maxd.max(d.abs() as f32);
        }
    }
    let n = p * DIM;
    let rel = (l2n / l2d.max(1e-30)).sqrt();
    // gate: the routed experts are the reference's arithmetic; the shared expert is a host model of
    // KN1's FP8 path; together they must equal DeepSeek's dumped output on >= 99.9 % of elements
    let ok = equal * 1000 >= n * 999;
    let layer_id = h.metadata.get("layer").cloned().unwrap_or_default();
    Ok((vec![format!("REFb real weights, layer {layer_id} ({}): {p} rows, {} distinct experts, class {}: ffn_out bf16 equal {equal} of {n} ({:.4} %), rel-L2 {rel:.3e}, max |diff| {maxd:.3e}  {}",
                     path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default(), distinct.len(), class.name(),
                     100.0 * equal as f64 / n as f64, if ok { "OK" } else { "FAIL" })], (!ok) as usize))
}

/// The probe entry (`dsv41::cli`). `--model-dir DIR`: also check every `kn2_moe_fixture_*.safetensors` in DIR.
pub fn run(args: &[String]) -> Result<Vec<String>> {
    // PK3: with a pack and a REFb fixture, the numeric probe of --dsv41-prefill-bf16-partials instead of this battery (it prints as it goes)
    if crate::dsv41::weights::partials::probe::wanted(args) { return crate::dsv41::weights::partials::probe::run(args).map(|_| Vec::new()); }
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    crate::dsv41::gpu::load_module(&dev)?;
    let stream = blocking_stream()?;
    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    let gen: Vec<OwnedExpert> = (0..12).map(|_| OwnedExpert::gen(&mut r)).collect();
    let mut lines = vec![format!("gpu_dsv41 kernel build {} (handshake ok); band 30: {} entries", env!("KERNEL_BUILD_ID"), launch::FNS.len())];
    let mut fails = 0;
    let configs: [(usize, usize, ExpertGeom, &[usize]); 4] = [
        (384, 6, ExpertGeom::whole(DIM, INTER), &[1, 2, 3, 6, 16, 17, 24, 32]),
        (384, 6, ExpertGeom::slice(DIM, INTER, 1, 4)?, &[1, 2, 6, 12, 24]),
        (384, 6, ExpertGeom::slice(DIM, INTER, 7, 8)?, &[1, 6, 24]),
        (128, 3, ExpertGeom::whole(DIM, INTER), &[1, 2, 6, 12, 24]),
    ];
    for (n_exp, top_k, geom, rows) in configs.iter() {
        let (l, f) = check_config(&dev, stream, &gen, *n_exp, *top_k, geom, rows, &mut r)?;
        lines.extend(l);
        fails += f;
    }
    let (l, f) = check_trait(&dev, stream, &gen, &mut r)?;
    lines.push(l);
    fails += f;
    if let Some(dir) = crate::arg_value(args, "--model-dir") {
        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir).with_context(|| format!("read {dir}"))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.file_name().and_then(|f| f.to_str()).is_some_and(|f| f.starts_with("kn2_moe_fixture_") && f.ends_with(".safetensors")))
            .collect();
        files.sort();
        ensure!(!files.is_empty(), "--model-dir {dir}: no kn2_moe_fixture_*.safetensors (tools/dsv41_kn2/moe_fixture.py writes them)");
        for f in &files {
            let (l, fl) = check_fixture(&dev, stream, f)?;
            lines.extend(l);
            fails += fl;
        }
    }
    lines.extend(bench(&dev, stream, &gen, 384, 6, &mut r)?);
    lines.extend(bench(&dev, stream, &gen, 128, 3, &mut r)?);
    ensure!(fails == 0, "{fails} MXFP4 checks failed:\n{}", lines.join("\n"));
    lines.push("DSV41_KERNEL_MXFP4_OK".into());
    Ok(lines)
}
