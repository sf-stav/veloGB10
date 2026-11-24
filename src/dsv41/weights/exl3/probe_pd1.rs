//! Leg PD1: the kernel probe's battery for the producer-side rotation and the paced L2 prefetch
//! (`--probe-dsv41-kernel --dsv41-kernel-set trellis`, after PK4's grouped-call battery). Every check is BIT-FOR-BIT
//! against the path the forward used before PD1 (prep launch + 1-linear calls):
//! - `rot_out`: linear A rotates its stored output into linear B's input; B then runs on it without a prep: A's and
//!   B's outputs equal A then B with B's own prep, at rows 1..32 (sampled) and for a bf16 and an f16 A output;
//! - `normrot`: the RMSNorm that rotates for two linears: y equals `d41_rmsnorm`'s, and each linear on the rotated rows
//!   equals the linear on y with its own prep;
//! - mixed widths in one `run_group`: each linear equals its single call (the group splits by width);
//! - the paced prefetch: 16 / 8 MiB launches take the paced time (1 MiB per wave_ns after the delay) and finish.

use super::dense::resolve;
use super::dense_dec::{self, normrot, run, run_group, Call, Pre, RotTo};
use super::probe::{dp, Cx, Lin, Rng};
use crate::dsv41::weights::{Accum, Act, ActDtype, DevPtr, Scratch, Shard};
use anyhow::{ensure, Result};
use cudarc::driver::{sys, CudaSlice};

fn act(p: &CudaSlice<u16>, rows: usize, cols: usize, dt: ActDtype) -> Act { Act { ptr: DevPtr(dp(p)), rows, cols, ld: cols, dtype: dt } }
fn actf(p: &CudaSlice<f32>, rows: usize, cols: usize) -> Act { Act { ptr: DevPtr(dp(p)), rows, cols, ld: cols, dtype: ActDtype::F32 } }
fn bits_eq<T: Copy + PartialEq>(a: &[T], b: &[T]) -> bool { a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x == y) }
fn f32_bits(v: &[f32]) -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() }

pub(super) fn battery(cx: &Cx, rng: &mut Rng, fails: &mut Vec<String>, lines: &mut Vec<String>) -> Result<()> {
    dense_dec::geometry()?;
    let sb = 64usize << 20;
    let ds = cx.zeros::<u8>(sb)?;
    let sc = Scratch { ptr: DevPtr(dp(&ds)), bytes: sb };
    let l = cx.l();
    // ---- rot_out + pre: A (1024 -> 1024, K5) then B (1024 -> 512, K5) ----
    let la = Lin::synth(cx, rng, 10, 1024, 1024)?;
    let lb = Lin::synth(cx, rng, 10, 1024, 512)?;
    let (wa, wb) = (la.dense("pd1.a", 1024, 1024, Shard::Whole), lb.dense("pd1.b", 512, 1024, Shard::Whole));
    let (sub, _) = dense_dec::suh_of(&wb)?;
    let mut bad = vec![];
    for &rows in &[1usize, 2, 3, 6, 12, 17, 18, 20, 23, 24, 32] {
        for dt in [ActDtype::Bf16, ActDtype::F16] {
            let x: Vec<f32> = (0..rows * 1024).map(|_| rng.f()).collect();
            let dx = cx.up(&x)?;
            // reference: A, then B with its own prep
            let (ya, yb) = (cx.zeros::<u16>(rows * 1024)?, cx.zeros::<f32>(rows * 512)?);
            run(&l, &[Call::new(&wa, &actf(&dx, rows, 1024), &act(&ya, rows, 1024, dt), Accum::Overwrite)], sc)?;
            run(&l, &[Call::new(&wb, &act(&ya, rows, 1024, dt), &actf(&yb, rows, 512), Accum::Overwrite)], sc)?;
            // PD1: A rotates into B's input, B runs on it without a prep
            let (ya2, yb2, xh) = (cx.zeros::<u16>(rows * 1024)?, cx.zeros::<f32>(rows * 512)?, cx.zeros::<u16>(rows * 1032)?);
            let (xa, y2a) = (actf(&dx, rows, 1024), act(&ya2, rows, 1024, dt));
            let mut ca = Call::new(&wa, &xa, &y2a, Accum::Overwrite);
            ca.rot = Some(RotTo { suh: sub, xh: dp(&xh), ld: 1032 });
            run(&l, &[ca], sc)?;
            let (a2, yb2a) = (act(&ya2, rows, 1024, dt), actf(&yb2, rows, 512));
            let mut cb = Call::new(&wb, &a2, &yb2a, Accum::Overwrite);
            cb.pre = Some(Pre { xh: dp(&xh), ld: 1032 });
            run(&l, &[cb], sc)?;
            let ok = bits_eq(&cx.dl(&ya)?, &cx.dl(&ya2)?) && bits_eq(&f32_bits(&cx.dl(&yb)?), &f32_bits(&cx.dl(&yb2)?));
            if !ok { bad.push((rows, dt)); }
        }
    }
    lines.push(format!("  {} rot_out: A (rotating into B's input) then B (no prep) == A then B (own prep), bits, rows 1..32 x bf16 / f16 A out ({} differ)",
                       if bad.is_empty() { "ok  " } else { "FAIL" }, bad.len()));
    if !bad.is_empty() { fails.push(format!("PD1 rot_out {bad:?}")); }

    // ---- normrot: RMSNorm over 1280 (f32 weight) rotating for D (1280 -> 512) and E (1280 -> 256) ----
    let ld_ = Lin::synth(cx, rng, 10, 1280, 512)?;
    let le = Lin::synth(cx, rng, 10, 1280, 256)?;
    let (wd, we) = (ld_.dense("pd1.d", 512, 1280, Shard::Whole), le.dense("pd1.e", 256, 1280, Shard::Whole));
    let nw: Vec<f32> = (0..1280).map(|_| 0.5 + rng.f().abs()).collect();
    let dnw = cx.up(&nw)?;
    let mut nbad = 0usize;
    for &rows in &[1usize, 3, 6, 24] {
        let xb: Vec<u16> = (0..rows * 1280).map(|_| half::bf16::from_f32(rng.f() * 3.0).to_bits()).collect();
        let dxb = cx.up(&xb)?;
        // reference y: d41_rmsnorm
        let y0 = cx.zeros::<u16>(rows * 1280)?;
        let a = crate::dsv41::gpu::mixers::RmsNormArgs { x: dp(&dxb), x_ld: 1280, w: dp(&dnw), y: dp(&y0), y_ld: 1280, w_fmt: 3u32,
                                                         dim: 1280, rows: rows as i32, eps: 1e-6, inv_dim: 1.0 / 1280.0 };
        let f = crate::dsv41::gpu::raw_fn("d41_rmsnorm").ok_or_else(|| anyhow::anyhow!("d41_rmsnorm missing"))?;
        let mut p = [&a as *const _ as *mut std::ffi::c_void];
        ensure!(unsafe { sys::cuLaunchKernel(f, rows as u32, 1, 1, 256, 1, 1, 0, cx.stream as sys::CUstream, p.as_mut_ptr(), std::ptr::null_mut()) } == sys::CUresult::CUDA_SUCCESS, "d41_rmsnorm launch");
        let (yd0, ye0) = (cx.zeros::<f32>(rows * 512)?, cx.zeros::<f32>(rows * 256)?);
        run(&l, &[Call::new(&wd, &act(&y0, rows, 1280, ActDtype::Bf16), &actf(&yd0, rows, 512), Accum::Overwrite)], sc)?;
        run(&l, &[Call::new(&we, &act(&y0, rows, 1280, ActDtype::Bf16), &actf(&ye0, rows, 256), Accum::Overwrite)], sc)?;
        // PD1: normrot writes y and both rotated inputs; D and E run without a prep (one grouped launch: same width)
        let (y1, xd, xe) = (cx.zeros::<u16>(rows * 1280)?, cx.zeros::<u16>(rows * 1280)?, cx.zeros::<u16>(rows * 1280)?);
        let (sd, _) = dense_dec::suh_of(&wd)?;
        let (se, _) = dense_dec::suh_of(&we)?;
        normrot(&l, dp(&dxb), 1280, dp(&dnw), 3u32, dp(&y1), 1280, 1280, rows, 1e-6, &[(sd, dp(&xd), 1280), (se, dp(&xe), 1280)])?;
        let (yd1, ye1) = (cx.zeros::<f32>(rows * 512)?, cx.zeros::<f32>(rows * 256)?);
        let ya = act(&y1, rows, 1280, ActDtype::Bf16);
        let (oa, ob) = (actf(&yd1, rows, 512), actf(&ye1, rows, 256));
        let mut c1 = Call::new(&wd, &ya, &oa, Accum::Overwrite); c1.pre = Some(Pre { xh: dp(&xd), ld: 1280 });
        let mut c2 = Call::new(&we, &ya, &ob, Accum::Overwrite); c2.pre = Some(Pre { xh: dp(&xe), ld: 1280 });
        run_group(&l, &[c1, c2], sc)?;
        let ok = bits_eq(&cx.dl(&y0)?, &cx.dl(&y1)?) && bits_eq(&f32_bits(&cx.dl(&yd0)?), &f32_bits(&cx.dl(&yd1)?)) && bits_eq(&f32_bits(&cx.dl(&ye0)?), &f32_bits(&cx.dl(&ye1)?));
        if !ok { nbad += 1; }
    }
    lines.push(format!("  {} normrot: y == d41_rmsnorm's y, and two linears on its rotated rows (one grouped launch, no prep) == each on y with its own prep, bits, rows 1 / 3 / 6 / 24",
                       if nbad == 0 { "ok  " } else { "FAIL" }));
    if nbad > 0 { fails.push("PD1 normrot".into()); }

    // ---- mixed widths in one run_group (K5 + K6 + K5 on one input): each == its single call ----
    let lc = Lin::synth(cx, rng, 12, 1024, 256)?;
    let wc = lc.dense("pd1.c", 256, 1024, Shard::Whole);
    ensure!(resolve(&wc, 1)?.bits == 6 && resolve(&wa, 1)?.bits == 5, "synthetic widths");
    let mut mbad = 0usize;
    for &rows in &[1usize, 6, 24] {
        let x: Vec<f32> = (0..rows * 1024).map(|_| rng.f()).collect();
        let dx = cx.up(&x)?;
        let xa = actf(&dx, rows, 1024);
        let ys: Vec<CudaSlice<f32>> = [1024usize, 256, 512].iter().map(|&n| cx.zeros::<f32>(rows * n)).collect::<Result<_>>()?;
        let yg: Vec<CudaSlice<f32>> = [1024usize, 256, 512].iter().map(|&n| cx.zeros::<f32>(rows * n)).collect::<Result<_>>()?;
        let ws = [&wa, &wc, &wb];
        let ya: Vec<Act> = ys.iter().zip([1024usize, 256, 512]).map(|(y, n)| actf(y, rows, n)).collect();
        let ga: Vec<Act> = yg.iter().zip([1024usize, 256, 512]).map(|(y, n)| actf(y, rows, n)).collect();
        for i in 0..3 { run(&l, &[Call::new(ws[i], &xa, &ya[i], Accum::Overwrite)], sc)?; }
        let calls: Vec<Call> = (0..3).map(|i| Call::new(ws[i], &xa, &ga[i], Accum::Overwrite)).collect();
        run_group(&l, &calls, sc)?;
        for i in 0..3 { if !bits_eq(&f32_bits(&cx.dl(&ys[i])?), &f32_bits(&cx.dl(&yg[i])?)) { mbad += 1; } }
    }
    lines.push(format!("  {} mixed widths (K5, K6, K5 on one input) in one run_group == single calls, bits, rows 1 / 6 / 24 (one launch per width)",
                       if mbad == 0 { "ok  " } else { "FAIL" }));
    if mbad > 0 { fails.push("PD1 mixed widths".into()); }

    // ---- the paced L2 prefetch: duration = delay + waves x wave_ns (+ the last wave's issue) ----
    let buf = cx.zeros::<u8>(40 << 20)?;
    let base = (dp(&buf) + 255) & !255;
    // one warm launch first (the first launch of a function pays its lazy load: 148 us for a 71 us pacing floor on .12)
    dense_dec::l2_paced(cx.stream, &[(base, 1 << 20, 1 << 20, 1)], 1 << 20, 0, 4559)?;
    cx.sync()?;
    for &(mib, delay, wave) in &[(16u64, 3000u64, 4559u64), (8, 3000, 4559), (16, 3000, 9118)] {
        let mut ev = [std::ptr::null_mut(); 2];
        for e in ev.iter_mut() { ensure!(unsafe { sys::cuEventCreate(e, 0) } == sys::CUresult::CUDA_SUCCESS, "event"); }
        cx.sync()?;
        unsafe { sys::cuEventRecord(ev[0], cx.stream as sys::CUstream); }
        let bytes = dense_dec::l2_paced(cx.stream, &[(base, 24 << 20, 24 << 20, 1)], mib << 20, delay, wave)?;
        unsafe { sys::cuEventRecord(ev[1], cx.stream as sys::CUstream); }
        cx.sync()?;
        let mut ms = 0f32;
        unsafe { sys::cuEventElapsedTime(&mut ms, ev[0], ev[1]); for e in ev { sys::cuEventDestroy_v2(e); } }
        let us = ms as f64 * 1000.0;
        let floor = (delay + (mib - 1) * wave) as f64 / 1000.0;
        let ok = bytes == mib << 20 && us >= floor && us < floor + 60.0;
        lines.push(format!("  {} paced L2 prefetch {mib} MiB (delay {delay} ns, {wave} ns a MiB): {us:.1} us (pacing floor {floor:.1} us), {bytes} B asked",
                           if ok { "ok  " } else { "FAIL" }));
        if !ok { fails.push(format!("PD1 paced prefetch {mib} MiB: {us:.1} us")); }
    }
    Ok(())
}
