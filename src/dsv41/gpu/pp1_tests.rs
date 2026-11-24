//! Leg PP1 GPU tests (`#[ignore]`, run under `flock /tmp/dsv41_gpu.lock`):
//! `cargo test --release --lib dsv41_pp1 -- --ignored --test-threads 1 --nocapture`.
//! The prefill paths must give the SAME BITS as the paths they replace: the wide router entries against the 32-row r* launches,
//! the row-split plain GEMV against one block row. Timings are printed (2,048 rows, the prefill chunk).

use super::mix_ref::*;
use super::*;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;

const DIM: usize = 5120;

fn dev() -> Arc<CudaDevice> {
    let d = CudaDevice::new(0).expect("GPU");
    crate::dsv41::gpu::load_module(&d).expect("gpu_dsv41 module");
    d
}
fn p<T>(s: &CudaSlice<T>) -> u64 { *s.device_ptr() as u64 }

fn time_ms(d: &Arc<CudaDevice>, reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    d.synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..reps { f(); }
    d.synchronize().unwrap();
    t.elapsed().as_secs_f64() * 1e3 / reps as f64
}

#[test]
#[ignore = "GPU"]
fn dsv41_pp1_router_wide_is_bit_identical_to_the_32_row_launches() {
    let d = dev();
    let mut rng = Rng(0x5EED_0001);
    let rmax = 2048usize;
    let x: Vec<u16> = (0..rmax * DIM).map(|_| bf_bits(bf(rng.normal()))).collect();
    let xd = d.htod_sync_copy(&x).unwrap();
    for (e, fmt) in [(384usize, 2u32), (384, 1), (384, 3), (128, 2), (100, 1)] {
        let wv: Vec<f32> = (0..e * DIM).map(|_| rng.normal() * 0.02).collect();
        let w16: Vec<u16> = wv.iter().map(|&v| if fmt == 2 { half::f16::from_f32(v).to_bits() } else { bf_bits(v) }).collect();
        let (wd16, wd32) = (d.htod_sync_copy(&w16).unwrap(), d.htod_sync_copy(&wv).unwrap());
        let wptr = if fmt == 3 { p(&wd32) } else { p(&wd16) };
        for rows in [33usize, 47, 100, 2000, 2048] {
            let a0 = RouterScoresArgs { x: p(&xd), x_ld: DIM as i64, w: wptr, w_ld: DIM as i64, scores: 0, s_ld: e as i64, w_fmt: fmt, dim: DIM as i32, e: e as i32, rows: rows as i32 };
            let rf = d.alloc_zeros::<f32>(rows * e).unwrap();
            router_scores_mode(&d, &RouterScoresArgs { scores: p(&rf), ..a0 }, false).unwrap();
            let want = d.dtoh_sync_copy(&rf).unwrap();
            for v in 0..ROUTER_WIDE.len() {
                let o = d.htod_sync_copy(&vec![f32::NAN; rows * e]).unwrap();
                router_scores_wide(&d, &RouterScoresArgs { scores: p(&o), ..a0 }, v).unwrap();
                let got = d.dtoh_sync_copy(&o).unwrap();
                let bad = got.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                assert_eq!(bad, 0, "{} E {e} fmt {fmt} rows {rows}: {bad} scores differ from the r* launches", ROUTER_WIDE[v].0);
            }
        }
        println!("router wide: E {e} fmt {fmt}: rows 33 / 47 / 100 / 2000 / 2048 bit-identical to the r* launches ({} entries)", ROUTER_WIDE.len());
    }
    // timing at the prefill chunk (E 384, f16 gate: the EXL3 pack)
    let wv: Vec<u16> = (0..384 * DIM).map(|_| half::f16::from_f32(rng.normal() * 0.02).to_bits()).collect();
    let wd = d.htod_sync_copy(&wv).unwrap();
    let sc = d.alloc_zeros::<f32>(rmax * 384).unwrap();
    let a = RouterScoresArgs { x: p(&xd), x_ld: DIM as i64, w: p(&wd), w_ld: DIM as i64, scores: p(&sc), s_ld: 384, w_fmt: 2, dim: DIM as i32, e: 384, rows: rmax as i32 };
    let t0 = time_ms(&d, 10, || router_scores_mode(&d, &a, false).unwrap());
    print!("router timing, 2,048 rows x 384 experts (f16 gate): r* launches of 32 rows {t0:.3} ms");
    for v in 0..ROUTER_WIDE.len() {
        let t = time_ms(&d, 10, || router_scores_wide(&d, &a, v).unwrap());
        print!("; {} {t:.3} ms", ROUTER_WIDE[v].0);
    }
    println!();
}

#[test]
#[ignore = "GPU"]
fn dsv41_pp1_plain_gemv_row_split_is_bit_identical() {
    use crate::dsv41::census::classify::{Role, TensorClass};
    use crate::dsv41::census::LinearFormat;
    use crate::dsv41::weights::{plain::PlainWeights, Accum, Act, ActDtype, DenseWeight, DevPtr, DevTensor, Dsv41Weights, Launch, Scratch, Shard};
    use cudarc::driver::sys;
    let d = dev();
    let mut st: sys::CUstream = std::ptr::null_mut();
    assert!(unsafe { sys::cuStreamCreate(&mut st, 0) } == sys::CUresult::CUDA_SUCCESS);
    let l = Launch { stream: st as u64, pdl: false };
    let mut rng = Rng(77);
    let id = crate::opts::lookup("dsv41-pf-gemv-rows").unwrap();
    for (n, k, dt) in [(32usize, 5120usize, "F16"), (32, 5120, "BF16"), (512, 1024, "F16")] {
        let wv: Vec<u16> = (0..n * k).map(|_| if dt == "F16" { half::f16::from_f32(rng.normal() * 0.02).to_bits() } else { bf_bits(rng.normal() * 0.02) }).collect();
        let dw = d.htod_sync_copy(&wv).unwrap();
        let w = DenseWeight { key: "plain".into(), class: TensorClass::GateWeight, layer: Some(0), sub: None, format: LinearFormat::Plain(dt.into()), out: n as u64, inn: k as u64,
            shard: Shard::Whole, tensors: vec![DevTensor { role: Role::Weight, ptr: DevPtr(p(&dw)), dtype: dt.into(), shape: vec![n as u64, k as u64], bytes: (n * k * 2) as u64 }] };
        for rows in [33usize, 100, 2048] {
            let x: Vec<u16> = (0..rows * k).map(|_| bf_bits(bf(rng.normal()))).collect();
            let dx = d.htod_sync_copy(&x).unwrap();
            let mut outs = Vec::new();
            let mut ts = Vec::new();
            for on in ["off", "on"] {
                crate::opts::set(id, on);
                let dy = d.htod_sync_copy(&vec![f32::NAN; rows * n]).unwrap();
                let xa = Act { ptr: DevPtr(p(&dx)), rows, cols: k, ld: k, dtype: ActDtype::Bf16 };
                let ya = Act { ptr: DevPtr(p(&dy)), rows, cols: n, ld: n, dtype: ActDtype::F32 };
                let run = || { PlainWeights.dense(&l, &w, &xa, &ya, Accum::Overwrite, Scratch::default()).unwrap(); };
                run();
                unsafe { sys::cuStreamSynchronize(st) };
                let t = std::time::Instant::now();
                for _ in 0..5 { run(); }
                unsafe { sys::cuStreamSynchronize(st) };
                ts.push(t.elapsed().as_secs_f64() * 1e3 / 5.0);
                outs.push(d.dtoh_sync_copy(&dy).unwrap());
            }
            let bad = outs[0].iter().zip(&outs[1]).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            println!("plain gemv {dt} N {n} K {k} rows {rows}: one block row {:.3} ms, split {:.3} ms, {bad} outputs differ", ts[0], ts[1]);
            assert_eq!(bad, 0);
            assert!(outs[1].iter().all(|v| v.is_finite()), "every output written");
        }
    }
    crate::opts::set(id, "on");
}

/// Leg P4: the one-warp-per-block plain GEMV (`--dsv41-pf-gemv-small`) gives the SAME BITS as `d41x_plain_gemv` for every row count
/// 1..=32, both dtypes, overwrite and add, narrow weights (32 / 100 / 256 columns); timings go to /tmp/dsv41_p4_gemv_small.txt.
#[test]
#[ignore = "GPU"]
fn dsv41_p4_plain_gemv_small_is_bit_identical() {
    use crate::dsv41::census::classify::{Role, TensorClass};
    use crate::dsv41::census::LinearFormat;
    use crate::dsv41::weights::{plain::PlainWeights, Accum, Act, ActDtype, DenseWeight, DevPtr, DevTensor, Dsv41Weights, Launch, Scratch, Shard};
    use cudarc::driver::sys;
    let d = dev();
    let mut st: sys::CUstream = std::ptr::null_mut();
    assert!(unsafe { sys::cuStreamCreate(&mut st, 0) } == sys::CUresult::CUDA_SUCCESS);
    let l = Launch { stream: st as u64, pdl: false };
    let mut rng = Rng(4242);
    let id = crate::opts::lookup("dsv41-pf-gemv-small").unwrap();
    let mut rep = String::new();
    for (n, k, dt) in [(32usize, 5120usize, "F16"), (32, 5120, "BF16"), (100, 5120, "F16"), (256, 5120, "F16"), (256, 1024, "BF16")] {
        let wv: Vec<u16> = (0..n * k).map(|_| if dt == "F16" { half::f16::from_f32(rng.normal() * 0.02).to_bits() } else { bf_bits(rng.normal() * 0.02) }).collect();
        let dw = d.htod_sync_copy(&wv).unwrap();
        let w = DenseWeight { key: "plain".into(), class: TensorClass::GateWeight, layer: Some(0), sub: None, format: LinearFormat::Plain(dt.into()), out: n as u64, inn: k as u64,
            shard: Shard::Whole, tensors: vec![DevTensor { role: Role::Weight, ptr: DevPtr(p(&dw)), dtype: dt.into(), shape: vec![n as u64, k as u64], bytes: (n * k * 2) as u64 }] };
        for rows in 1..=32usize {
            let x: Vec<u16> = (0..rows * k).map(|_| bf_bits(bf(rng.normal()))).collect();
            let dx = d.htod_sync_copy(&x).unwrap();
            let y0: Vec<f32> = (0..rows * n).map(|_| rng.normal()).collect();
            for acc in [Accum::Overwrite, Accum::Add] {
                let mut outs = Vec::new();
                let mut ts = Vec::new();
                for on in ["off", "on"] {
                    crate::opts::set(id, on);
                    let dy = d.htod_sync_copy(&if acc == Accum::Add { y0.clone() } else { vec![f32::NAN; rows * n] }).unwrap();
                    let xa = Act { ptr: DevPtr(p(&dx)), rows, cols: k, ld: k, dtype: ActDtype::Bf16 };
                    let ya = Act { ptr: DevPtr(p(&dy)), rows, cols: n, ld: n, dtype: ActDtype::F32 };
                    let run = || { PlainWeights.dense(&l, &w, &xa, &ya, acc, Scratch::default()).unwrap(); };
                    run();
                    unsafe { sys::cuStreamSynchronize(st) };
                    let t = std::time::Instant::now();
                    for _ in 0..20 { run(); }
                    unsafe { sys::cuStreamSynchronize(st) };
                    ts.push(t.elapsed().as_secs_f64() * 1e6 / 20.0);
                    // dense() for an accumulating call adds again on every run: re-run once on a fresh buffer for the bits
                    let dy2 = d.htod_sync_copy(&if acc == Accum::Add { y0.clone() } else { vec![f32::NAN; rows * n] }).unwrap();
                    let ya2 = Act { ptr: DevPtr(p(&dy2)), rows, cols: n, ld: n, dtype: ActDtype::F32 };
                    PlainWeights.dense(&l, &w, &xa, &ya2, acc, Scratch::default()).unwrap();
                    unsafe { sys::cuStreamSynchronize(st) };
                    outs.push(d.dtoh_sync_copy(&dy2).unwrap());
                }
                let bad = outs[0].iter().zip(&outs[1]).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                assert_eq!(bad, 0, "{dt} N {n} K {k} rows {rows} {acc:?}");
                assert!(outs[1].iter().all(|v| v.is_finite()), "every output written");
                if acc == Accum::Overwrite && (rows == 1 || rows == 6 || rows == 24 || rows == 32) {
                    rep.push_str(&format!("{dt} N {n} K {k} rows {rows}: generic {:.2} us, small {:.2} us\n", ts[0], ts[1]));
                }
            }
        }
    }
    crate::opts::set(id, "on");
    std::fs::write("/tmp/dsv41_p4_gemv_small.txt", rep).unwrap();
}
