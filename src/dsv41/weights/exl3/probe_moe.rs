//! KE2: correctness battery of the served MoE paths (decode <= 32 rows, prefill above) on the real EAQ v1 experts of the
//! fixture: bitwise against the generic path, the fused shared expert against two generic calls, batch invariance at
//! every width 1..32 (and decode rows against prefill rows), TP inner-slice partials, the f64 chain, PDL and graph replay.

use super::moe::{self, MoePath};
use super::probe::{check, dp, ref_expert, tp_bounds, Cx, Ex, Fx, Rng};
use crate::dsv41::weights::{Act, ActDtype, DenseWeight, DevPtr, Launch, MoeWeight, Route, Scratch, Shard};
use anyhow::{ensure, Result};
use cudarc::driver::sys;

pub(super) fn moe_weight_tp(ex: &[Ex], layer: u16, top_k: usize, tp: usize, rank: usize) -> MoeWeight {
    let experts = ex.iter().enumerate().map(|(e, x)| {
        let b = tp_bounds(tp, e);
        let (lo, hi) = (b[rank] as u64 * 128, b[rank + 1] as u64 * 128);
        let sh13 = if lo == 0 && hi == 2304 { Shard::Whole } else { Shard::Out(lo..hi) };
        let sh2 = if lo == 0 && hi == 2304 { Shard::Whole } else { Shard::In(lo..hi) };
        [x.l[0].dense("w1", 2304, 5120, sh13.clone()), x.l[1].dense("w2", 5120, 2304, sh2), x.l[2].dense("w3", 2304, 5120, sh13)]
    }).collect::<Vec<[DenseWeight; 3]>>();
    MoeWeight { layer, n_experts: ex.len(), local: 0..ex.len(), top_k, dim: 5120, inter: 2304, swiglu_limit: 10.0, experts }
}

pub(super) struct Call<'a> { pub mw: &'a MoeWeight, pub sh: Option<&'a MoeWeight>, pub path: MoePath, pub pdl: bool, pub graph: bool, pub bf16: bool }

/// One MoE call: y starts as y0 (accumulate), returns y.
pub(super) fn run(cx: &Cx, c: &Call, x: &[f32], ids: &[i32], wts: &[f32], y0: &[f32], rows: usize) -> Result<Vec<f32>> {
    let sb = moe::scratch_for(c.mw, c.sh, rows, c.path)?;
    let ds = cx.zeros::<u8>(sb.max(256))?;
    let dx8: cudarc::driver::CudaSlice<u8> = if c.bf16 {
        let v: Vec<u8> = x[..rows * 5120].iter().flat_map(|f| half::bf16::from_f32(*f).to_bits().to_le_bytes()).collect(); cx.up(&v)?
    } else { let v: Vec<u8> = x[..rows * 5120].iter().flat_map(|f| f.to_le_bytes()).collect(); cx.up(&v)? };
    let (di, dw, dy) = (cx.up(&ids[..rows * c.mw.top_k])?, cx.up(&wts[..rows * c.mw.top_k])?, cx.up(&y0[..rows * 5120])?);
    let xa = Act { ptr: DevPtr(dp(&dx8)), rows, cols: 5120, ld: 5120, dtype: if c.bf16 { ActDtype::Bf16 } else { ActDtype::F32 } };
    let ya = Act { ptr: DevPtr(dp(&dy)), rows, cols: 5120, ld: 5120, dtype: ActDtype::F32 };
    let l = Launch { stream: cx.stream, pdl: c.pdl };
    let rt = Route { ids: DevPtr(dp(&di)), weights: DevPtr(dp(&dw)), top_k: c.mw.top_k };
    let sc = Scratch { ptr: DevPtr(dp(&ds)), bytes: sb };
    if !c.graph {
        moe::moe_ex(&l, c.mw, c.sh, &xa, &rt, &ya, sc, c.path)?;
        return cx.dl(&dy);
    }
    // graph: capture one call on a fresh y, replay it once
    let st = cx.stream as sys::CUstream;
    let (mut g, mut ge): (sys::CUgraph, sys::CUgraphExec) = (std::ptr::null_mut(), std::ptr::null_mut());
    unsafe {
        ensure!(sys::cuStreamBeginCapture_v2(st, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) == sys::CUresult::CUDA_SUCCESS, "begin capture");
        let r = moe::moe_ex(&l, c.mw, c.sh, &xa, &rt, &ya, sc, c.path);
        let ec = sys::cuStreamEndCapture(st, &mut g);
        r?;
        ensure!(ec == sys::CUresult::CUDA_SUCCESS, "end capture {ec:?}");
        ensure!(sys::cuGraphInstantiateWithFlags(&mut ge, g, 0) == sys::CUresult::CUDA_SUCCESS, "instantiate");
        ensure!(sys::cuGraphLaunch(ge, st) == sys::CUresult::CUDA_SUCCESS, "graph launch");
    }
    let out = cx.dl(&dy)?;
    unsafe { sys::cuGraphExecDestroy(ge); sys::cuGraphDestroy(g); }
    Ok(out)
}

/// leg PF1: one OVERWRITE call (`moe_ex_ow`) on a y pre-filled with garbage (NaN bits), eager or captured + replayed; the
/// result must equal an accumulating call on a zeroed y bit for bit.
fn run_ow(cx: &Cx, c: &Call, x: &[f32], ids: &[i32], wts: &[f32], rows: usize) -> Result<Vec<f32>> {
    let sb = moe::scratch_for(c.mw, c.sh, rows, c.path)?;
    let ds = cx.zeros::<u8>(sb.max(256))?;
    let dx = cx.up(&x[..rows * 5120])?;
    let (di, dw, dy) = (cx.up(&ids[..rows * c.mw.top_k])?, cx.up(&wts[..rows * c.mw.top_k])?, cx.up(&vec![f32::from_bits(0x7fc0_dead); rows * 5120])?);
    let xa = Act { ptr: DevPtr(dp(&dx)), rows, cols: 5120, ld: 5120, dtype: ActDtype::F32 };
    let ya = Act { ptr: DevPtr(dp(&dy)), rows, cols: 5120, ld: 5120, dtype: ActDtype::F32 };
    let l = Launch { stream: cx.stream, pdl: c.pdl };
    let rt = Route { ids: DevPtr(dp(&di)), weights: DevPtr(dp(&dw)), top_k: c.mw.top_k };
    let sc = Scratch { ptr: DevPtr(dp(&ds)), bytes: sb };
    if !c.graph {
        moe::moe_ex_ow(&l, c.mw, c.sh, &xa, &rt, &ya, sc, c.path, true)?;
        return cx.dl(&dy);
    }
    let st = cx.stream as sys::CUstream;
    let (mut g, mut ge): (sys::CUgraph, sys::CUgraphExec) = (std::ptr::null_mut(), std::ptr::null_mut());
    unsafe {
        ensure!(sys::cuStreamBeginCapture_v2(st, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) == sys::CUresult::CUDA_SUCCESS, "begin capture");
        let r = moe::moe_ex_ow(&l, c.mw, c.sh, &xa, &rt, &ya, sc, c.path, true);
        let ec = sys::cuStreamEndCapture(st, &mut g);
        r?;
        ensure!(ec == sys::CUresult::CUDA_SUCCESS, "end capture {ec:?}");
        ensure!(sys::cuGraphInstantiateWithFlags(&mut ge, g, 0) == sys::CUresult::CUDA_SUCCESS, "instantiate");
        // replayed twice: the second replay starts from the first one's output (overwrite must not read it)
        ensure!(sys::cuGraphLaunch(ge, st) == sys::CUresult::CUDA_SUCCESS, "graph launch");
        ensure!(sys::cuGraphLaunch(ge, st) == sys::CUresult::CUDA_SUCCESS, "graph launch");
    }
    let out = cx.dl(&dy)?;
    unsafe { sys::cuGraphExecDestroy(ge); sys::cuGraphDestroy(g); }
    Ok(out)
}

fn bits_eq(a: &[f32], b: &[f32]) -> usize { a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count() }
fn line(ok: bool, s: String, fails: &mut Vec<String>, lines: &mut Vec<String>) { lines.push(format!("  {} {s}", if ok { "ok  " } else { "FAIL" })); if !ok { fails.push(s); } }

pub(super) fn battery(cx: &Cx, fx: &Fx, rng: &mut Rng, fails: &mut Vec<String>, lines: &mut Vec<String>) -> Result<()> {
    let sh = Ex { l: [fx.lin(cx, "layers.24.ffn.shared_experts.w1", true)?, fx.lin(cx, "layers.24.ffn.shared_experts.w2", true)?, fx.lin(cx, "layers.24.ffn.shared_experts.w3", true)?] };
    for (layer, nex, label) in [("24", 10usize, "K3"), ("20", 6, "K2")] {
        let ex = super::probe::load_experts(cx, fx, layer, nex)?;
        let top_k = 6.min(nex);
        let mk = |rng: &mut Rng, rows: usize| {
            let x: Vec<f32> = (0..rows * 5120).map(|_| rng.f()).collect();
            let y0: Vec<f32> = (0..rows * 5120).map(|_| 0.25 * rng.f()).collect();
            let mut ids = vec![]; let mut wts = vec![];
            for _ in 0..rows { let mut pool: Vec<usize> = (0..nex).collect(); for _ in 0..top_k { let i = rng.below(pool.len()); ids.push(pool.remove(i) as i32); wts.push(0.1 + 0.4 * rng.f().abs()); } }
            (x, ids, wts, y0)
        };
        for tp in [1usize, 2, 4, 8] {
            let ranks: Vec<usize> = if tp == 1 { vec![0] } else { vec![0, 1] };
            for rank in ranks {
                moe::clear_tables();
                let mw = moe_weight_tp(&ex, 24, top_k, tp, rank);
                let shw = moe_weight_tp(std::slice::from_ref(&sh), 24, 1, tp, rank);
                let tag = format!("moe {label} TP{tp} r{rank}");
                // served (decode kernels) vs generic, bitwise, with and without the shared expert, f32 and bf16 x; leg PM1:
                // the dispatched structure (auto), then each decode MoE structure forced at every row count
                for (fm, sname) in [(1u8, "auto"), (2, "short-CTA (19_*)"), (3, "persistent (15_*)")] {
                    super::moe_cp::force(fm);
                    let mut bad = vec![];
                    for &rows in &[1usize, 2, 3, 5, 6, 8, 9, 16, 17, 24, 32] {
                        let (x, ids, wts, y0) = mk(rng, rows);
                        for (shx, bf) in [(false, false), (true, false), (true, true)] {
                            let s = if shx { Some(&shw) } else { None };
                            let g = run(cx, &Call { mw: &mw, sh: s, path: MoePath::Generic, pdl: false, graph: false, bf16: bf }, &x, &ids, &wts, &y0, rows)?;
                            let a = run(cx, &Call { mw: &mw, sh: s, path: MoePath::Auto, pdl: false, graph: false, bf16: bf }, &x, &ids, &wts, &y0, rows)?;
                            let d = bits_eq(&a, &g);
                            if d > 0 { bad.push(format!("rows {rows}{}{}: {d} differ", if shx { " +shared" } else { "" }, if bf { " bf16" } else { "" })); }
                        }
                    }
                    super::moe_cp::force(0);
                    let what = if fm == 1 { String::new() } else { format!(", {sname} forced") };
                    line(bad.is_empty(), format!("{tag}: served decode == generic path bit for bit (rows 1..32, +shared, f32 / bf16 x{what}) {}", bad.join("; ")), fails, lines);
                }
                // batch invariance at every width: each row of a 32-row call == its 1-row call (and the 24-row call's rows)
                if rank == 0 {
                    let (x, ids, wts, y0) = mk(rng, 32);
                    let c = Call { mw: &mw, sh: Some(&shw), path: MoePath::Auto, pdl: false, graph: false, bf16: false };
                    let big = run(cx, &c, &x, &ids, &wts, &y0, 32)?;
                    let mut nb = 0;
                    for w in [1usize, 2, 3, 6, 7, 12, 16, 17, 24, 31] {
                        let o = run(cx, &c, &x, &ids, &wts, &y0, w)?;
                        nb += bits_eq(&o, &big[..w * 5120]);
                    }
                    for r in 0..32 {
                        let o = run(cx, &c, &x[r * 5120..], &ids[r * top_k..], &wts[r * top_k..], &y0[r * 5120..], 1)?;
                        nb += bits_eq(&o, &big[r * 5120..(r + 1) * 5120]);
                    }
                    line(nb == 0, format!("{tag}: batch invariance, rows of widths 1..31 and each of 32 rows == the 32-row call ({nb} values differ)"), fails, lines);
                    // PDL attribute and graph replay give the same bits
                    let p = run(cx, &Call { pdl: true, ..c }, &x, &ids, &wts, &y0, 32)?;
                    let gr = run(cx, &Call { pdl: true, graph: true, ..c }, &x, &ids, &wts, &y0, 32)?;
                    line(bits_eq(&p, &big) == 0 && bits_eq(&gr, &big) == 0, format!("{tag}: PDL launch and graph replay (PDL edges) == eager, bits"), fails, lines);
                    // leg PF1: the overwrite combine (no memset before the call) == accumulate on a zeroed y, every width
                    let z = vec![0f32; 32 * 5120];
                    let mut no = 0;
                    for w in [1usize, 2, 6, 12, 24, 32] {
                        let acc = run(cx, &c, &x, &ids, &wts, &z, w)?;
                        let e = run_ow(cx, &Call { pdl: true, ..c }, &x, &ids, &wts, w)?;
                        let gw = run_ow(cx, &Call { pdl: true, graph: true, ..c }, &x, &ids, &wts, w)?;
                        no += bits_eq(&e, &acc) + bits_eq(&gw, &acc);
                    }
                    line(no == 0, format!("{tag}: overwrite y (0 + sum, PDL eager and graph x2) == accumulate on zeroed y, rows 1..32 ({no} values differ)"), fails, lines);
                }
            }
        }
        // PREFILL (> 32 rows): every row bit-equal to the decode numerics (the generic path runs them as 32-row chunks);
        // 600 rows = 3 plan blocks and up to 3 row tiles per expert
        for (tp, rank) in [(1usize, 0usize), (4, 1), (8, 0)] {
            moe::clear_tables();
            let mw = moe_weight_tp(&ex, 24, top_k, tp, rank);
            let shw = moe_weight_tp(std::slice::from_ref(&sh), 24, 1, tp, rank);
            let mut bad = vec![];
            let mut cls = String::new();
            for &rows in &[33usize, 200, 600] {
                let (x, ids, wts, y0) = mk(rng, rows);
                for (shx, bf) in [(false, false), (true, false), (true, true)] {
                    let s = if shx { Some(&shw) } else { None };
                    let g = run(cx, &Call { mw: &mw, sh: s, path: MoePath::Generic, pdl: false, graph: false, bf16: bf }, &x, &ids, &wts, &y0, rows)?;
                    let a = run(cx, &Call { mw: &mw, sh: s, path: MoePath::Auto, pdl: false, graph: false, bf16: bf }, &x, &ids, &wts, &y0, rows)?;
                    let d = bits_eq(&a, &g);
                    if d > 0 { bad.push(format!("rows {rows}{}{}: {d} differ", if shx { " +shared" } else { "" }, if bf { " bf16" } else { "" })); }
                }
                if rows == 600 {
                    let (x, ids, wts, y0) = mk(rng, rows);
                    let a = run(cx, &Call { mw: &mw, sh: Some(&shw), path: MoePath::Auto, pdl: true, graph: true, bf16: true }, &x, &ids, &wts, &y0, rows)?;
                    let g = run(cx, &Call { mw: &mw, sh: Some(&shw), path: MoePath::Generic, pdl: false, graph: false, bf16: true }, &x, &ids, &wts, &y0, rows)?;
                    cls = format!("; graph + PDL replay at 600 rows: {} differ", bits_eq(&a, &g));
                    if bits_eq(&a, &g) > 0 { bad.push("graph replay".into()); }
                }
            }
            line(bad.is_empty(), format!("moe {label} TP{tp} r{rank}: served PREFILL rows == decode-numerics rows bit for bit (33 / 200 / 600 rows, +shared, f32 / bf16 x){cls} {}", bad.join("; ")), fails, lines);
        }
        // f64 chain (whole experts + shared, 4 rows) and TP partials summed vs whole
        moe::clear_tables();
        let full = moe_weight_tp(&ex, 24, top_k, 1, 0);
        let shf = moe_weight_tp(std::slice::from_ref(&sh), 24, 1, 1, 0);
        let (x, ids, wts, _) = mk(rng, 4);
        let z = vec![0f32; 4 * 5120];
        let c = Call { mw: &full, sh: Some(&shf), path: MoePath::Auto, pdl: false, graph: false, bf16: false };
        let o = run(cx, &c, &x, &ids, &wts, &z, 4)?;
        let mut refy = vec![0f64; 4 * 5120];
        for r in 0..4 {
            let mut rr = vec![0f64; 5120];
            for j in 0..top_k { let e = ids[r * top_k + j] as usize; let v = ref_expert(&x[r * 5120..(r + 1) * 5120], &ex[e], wts[r * top_k + j], 10.0); for k in 0..5120 { rr[k] += v[k]; } }
            let v = ref_expert(&x[r * 5120..(r + 1) * 5120], &sh, 1.0, 10.0);
            for k in 0..5120 { refy[r * 5120 + k] = rr[k] + v[k]; }
        }
        // Tolerance 4e-4: the error floor of this check is the reference's bf16 rounding point of the activation: wherever
        // the device's f32 value and the f64 value straddle a bf16 boundary the element differs by ~2^-9 (draw-dependent:
        // 1.4e-4 .. 2.6e-4 seen on the same kernels). The bitwise gates above are the strong ones.
        check(&format!("moe {label} served decode, rows=4, whole experts + shared vs f64 chain"), super::probe::rel_l2(&o, &refy), 4e-4, fails, lines);
        for tp in [2usize, 4, 8] {
            let mut sum = vec![0f32; 4 * 5120];
            for rank in 0..tp {
                moe::clear_tables();
                let mw = moe_weight_tp(&ex, 24, top_k, tp, rank);
                let shw = moe_weight_tp(std::slice::from_ref(&sh), 24, 1, tp, rank);
                let o = run(cx, &Call { mw: &mw, sh: Some(&shw), ..c }, &x, &ids, &wts, &z, 4)?;
                for (s, v) in sum.iter_mut().zip(&o) { *s += v; }
            }
            let rf: Vec<f64> = o.iter().map(|v| *v as f64).collect();
            check(&format!("moe {label} served decode TP{tp} inner-slice partials (+shared) summed vs whole"), super::probe::rel_l2(&sum, &rf), 2e-5, fails, lines);
        }
        moe::clear_tables();
    }
    Ok(())
}

/// Dense wide (> 32 rows): the trellis-source and reconstruct arms give the same bits (integer K); both vs the f64
/// chain; Out slices (aligned / un-aligned), In slices, bf16 / f32 out, accumulate; several row tiles.
pub(super) fn dense_wide_battery(cx: &Cx, rng: &mut Rng, fails: &mut Vec<String>, lines: &mut Vec<String>) -> Result<()> {
    use super::dense_wide::{dense_wide, WideSrc};
    use super::probe::{ref_gemm_post, ref_prep, rel_l2, Lin};
    use crate::dsv41::weights::Accum;
    let (inn, out) = (1280usize, 2048usize);
    for &k2 in &[4usize, 6, 8, 10, 16, 5] {
        let lin = Lin::synth(cx, rng, k2, inn, out)?;
        let nm = format!("dense wide K{}{}", k2 / 2, if k2 % 2 == 1 { ".5" } else { "" });
        for (shard, lab) in [(Shard::Whole, "whole"), (Shard::Out(256..1024), "out 256..1024"), (Shard::Out(100..1500), "out 100..1500"), (Shard::In(256..1024), "in 256..1024")] {
            let w = lin.dense("lin", out, inn, shard.clone());
            let (n_lo, n_hi, k_lo, k_hi) = match &shard { Shard::Whole => (0, out, 0, inn), Shard::Out(r) => (r.start as usize, r.end as usize, 0, inn), Shard::In(r) => (0, out, r.start as usize, r.end as usize) };
            let (kl, nc) = (k_hi - k_lo, n_hi - n_lo);
            for &rows in &[33usize, 300] {
                let x: Vec<f32> = (0..rows * kl).map(|_| rng.f()).collect();
                let y0: Vec<f32> = (0..rows * nc).map(|_| 0.1 * rng.f()).collect();
                let sb = super::dense_wide::scratch_bytes(&w, rows, WideSrc::Recon)?;
                let ds = cx.zeros::<u8>(sb)?; let dx = cx.up(&x)?;
                let mut outs = vec![];
                for src in [WideSrc::Auto, WideSrc::Recon] {
                    for accm in [Accum::Overwrite, Accum::Add] {
                        let dy = cx.up(&y0)?;
                        let xa = Act { ptr: DevPtr(dp(&dx)), rows, cols: kl, ld: kl, dtype: ActDtype::F32 };
                        let ya = Act { ptr: DevPtr(dp(&dy)), rows, cols: nc, ld: nc, dtype: ActDtype::F32 };
                        dense_wide(&cx.l(), &w, &xa, &ya, accm, Scratch { ptr: DevPtr(dp(&ds)), bytes: sb }, src)?;
                        outs.push(cx.dl(&dy)?);
                    }
                }
                if k2 % 2 == 0 {
                    let d = bits_eq(&outs[0], &outs[2]) + bits_eq(&outs[1], &outs[3]);
                    line(d == 0, format!("{nm} {lab} rows={rows}: trellis-source == reconstruct arm, bits ({d} differ)"), fails, lines);
                }
                if rows == 33 {
                    // the f64 GEMM chain on the device's OWN fp16 activations (the prep is checked separately by the KE battery)
                    let xb = cx.dl(&ds)?;
                    let xh: Vec<u16> = xb[..4 * kl * 2].chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                    let hp = ref_prep(&x, 4, kl, &lin.suh[k_lo..k_hi]);
                    let dif = hp.iter().zip(&xh).map(|(a, b)| (super::probe::h2f(*a) - super::probe::h2f(*b)).abs() as f64).fold(0.0, f64::max);
                    check(&format!("{nm} {lab} wide prep fp16 max abs diff vs f64"), dif, 2e-3, fails, lines);
                    let r = ref_gemm_post(&xh, 4, kl, &lin.w, out, k_lo, n_lo, n_hi, &lin.svh);
                    check(&format!("{nm} {lab} rows=33 (first 4) vs f64 chain"), rel_l2(&outs[0][..4 * nc], &r), 1e-5, fails, lines);
                    let acc: Vec<f32> = outs[0].iter().zip(&y0).map(|(a, b)| a + b).collect();
                    let d = outs[1].iter().zip(&acc).filter(|(a, b)| (*a - *b).abs() > 1e-6 * (1.0 + b.abs())).count();
                    line(d == 0, format!("{nm} {lab}: Accum::Add == overwrite + y0 ({d} differ beyond 1e-6)"), fails, lines);
                }
            }
        }
    }
    Ok(())
}

/// Leg SX1: experts whose three matrices have different EXL3 widths (synthetic trellis, host decode; no fixture needed):
/// a shared expert like EAQ v1 2.9 bpw layer 29's (w1 / w3 K4, w2 K5) and routed experts mixing w1 != w3 != w2 and widths
/// across experts (the SAGE 3.30 bpw pattern). Every served path (decode rows <= 32, prefill rows > 32) and the generic path
/// vs the f64 chain of the host-decoded weights (each matrix at its own width), and served vs generic bit for bit. Before
/// SX1 the kernels read every matrix of a table at ONE width: this battery fails there (rel-L2 O(1)).
pub(super) fn mixed_width_battery(cx: &Cx, rng: &mut Rng, fails: &mut Vec<String>, lines: &mut Vec<String>) -> Result<()> {
    use super::probe::Lin;
    let mk_ex = |rng: &mut Rng, k1: usize, k3: usize, k2: usize| -> Result<Ex> {
        Ok(Ex { l: [Lin::synth(cx, rng, 2 * k1, 5120, 2304)?, Lin::synth(cx, rng, 2 * k2, 2304, 5120)?, Lin::synth(cx, rng, 2 * k3, 5120, 2304)?] })
    };
    let sh = mk_ex(rng, 4, 4, 5)?;
    let widths = [(3usize, 3usize, 3usize), (4, 5, 4), (2, 3, 5), (5, 6, 7), (3, 3, 4), (8, 7, 6)];
    let ex: Vec<Ex> = widths.iter().map(|&(a, b, c)| mk_ex(rng, a, b, c)).collect::<Result<_>>()?;
    let (nex, top_k) = (ex.len(), 2usize);
    let mk = |rng: &mut Rng, rows: usize| {
        let x: Vec<f32> = (0..rows * 5120).map(|_| rng.f()).collect();
        let mut ids = vec![]; let mut wts = vec![];
        for _ in 0..rows { let mut pool: Vec<usize> = (0..nex).collect(); for _ in 0..top_k { let i = rng.below(pool.len()); ids.push(pool.remove(i) as i32); wts.push(0.1 + 0.4 * rng.f().abs()); } }
        (x, ids, wts)
    };
    let refy = |x: &[f32], ids: &[i32], wts: &[f32], rows: usize, with_sh: bool, only_sh: bool| -> Vec<f64> {
        let mut out = vec![0f64; rows * 5120];
        for r in 0..rows {
            let xr = &x[r * 5120..(r + 1) * 5120];
            if !only_sh { for j in 0..top_k { let v = ref_expert(xr, &ex[ids[r * top_k + j] as usize], wts[r * top_k + j], 10.0); for k in 0..5120 { out[r * 5120 + k] += v[k]; } } }
            if with_sh || only_sh { let v = ref_expert(xr, &sh, 1.0, 10.0); for k in 0..5120 { out[r * 5120 + k] += v[k]; } }
        }
        out
    };
    let row_set = [1usize, 5, 17, 32, 33, 200];
    let inputs: Vec<(Vec<f32>, Vec<i32>, Vec<f32>)> = row_set.iter().map(|&r| mk(rng, r)).collect();
    for tp in [1usize, 2] {
        let ranks: Vec<usize> = if tp == 1 { vec![0] } else { vec![0, 1] };
        let mut sum_ref: Vec<Vec<f32>> = Vec::new();
        for rank in ranks {
            moe::clear_tables();
            let mw = moe_weight_tp(&ex, 29, top_k, tp, rank);
            let shw = moe_weight_tp(std::slice::from_ref(&sh), 29, 1, tp, rank);
            let tag = format!("SX1 mixed widths TP{tp} r{rank}");
            let mut bad = vec![];
            for (&rows, (x, ids, wts)) in row_set.iter().zip(&inputs) {
                let z = vec![0f32; rows * 5120];
                for (shx, bf) in [(false, false), (true, false), (true, true)] {
                    let s = if shx { Some(&shw) } else { None };
                    let g = run(cx, &Call { mw: &mw, sh: s, path: MoePath::Generic, pdl: false, graph: false, bf16: bf }, x, ids, wts, &z, rows)?;
                    let a = run(cx, &Call { mw: &mw, sh: s, path: MoePath::Auto, pdl: false, graph: false, bf16: bf }, x, ids, wts, &z, rows)?;
                    let d = bits_eq(&a, &g);
                    if d > 0 { bad.push(format!("rows {rows}{}{}: {d} differ", if shx { " +shared" } else { "" }, if bf { " bf16" } else { "" })); }
                    // leg PM1: each decode MoE structure forced (per-matrix widths inside one launch: w1 / w3 / w2 differ)
                    if rows <= 32 {
                        for (fm, sname) in [(2u8, "short-CTA"), (3, "persistent")] {
                            super::moe_cp::force(fm);
                            let f = run(cx, &Call { mw: &mw, sh: s, path: MoePath::Auto, pdl: false, graph: false, bf16: bf }, x, ids, wts, &z, rows);
                            super::moe_cp::force(0);
                            let d = bits_eq(&f?, &g);
                            if d > 0 { bad.push(format!("rows {rows}{}{} {sname} forced: {d} differ", if shx { " +shared" } else { "" }, if bf { " bf16" } else { "" })); }
                        }
                    }
                    if tp == 1 && !bf && rows <= 33 {
                        let rf = refy(x, ids, wts, rows, shx, false);
                        check(&format!("{tag}: served rows={rows}{} vs f64 chain (each matrix at its own width)", if shx { " +shared (w1/w3 K4, w2 K5)" } else { "" }), super::probe::rel_l2(&a, &rf), 4e-4, fails, lines);
                    }
                }
                if tp == 2 && rows == 5 {
                    let o = run(cx, &Call { mw: &mw, sh: Some(&shw), path: MoePath::Auto, pdl: false, graph: false, bf16: false }, x, ids, wts, &z, rows)?;
                    sum_ref.push(o);
                    if sum_ref.len() == 2 {
                        let s: Vec<f32> = sum_ref[0].iter().zip(&sum_ref[1]).map(|(a, b)| a + b).collect();
                        check("SX1 mixed widths TP2 inner-slice partials (+shared) summed vs f64 chain", super::probe::rel_l2(&s, &refy(x, ids, wts, rows, true, false)), 4e-4, fails, lines);
                    }
                }
            }
            line(bad.is_empty(), format!("{tag}: served (decode / prefill) == generic path bit for bit (rows 1..200, +shared, f32 / bf16 x) {}", bad.join("; ")), fails, lines);
        }
    }
    moe::clear_tables();
    Ok(())
}
