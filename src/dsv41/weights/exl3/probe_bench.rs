//! KE / KE2: the timing sections of `--probe-dsv41-kernel --dsv41-kernel-set trellis`, chosen by
//! `--dsv41-kernel-timing <comma list>`:
//!   ddec     PK4: the dense decode byte-floor table only (served / graph+pdl / old split-K, the TP forward's shapes)
//!   decode   dense byte-floor table + routed MoE decode byte floor (K3 and K2, TP1/2/4/8 slices, 1..24 rows)
//!   shared   routed + shared expert in one call (the fused launch), TP4 / TP2 slices
//!   prefill  routed MoE prefill (384 experts, top-6) at M = 2K / 8K / 16K rows per rank: TFLOP/s
//!   dense    dense wide (> 32 rows) linears at M = 2K / 8K / 16K: TFLOP/s
//!   moe      routed MoE decode byte floor only (no dense byte-floor table)
//!   old      also time the generic (pre-KE2) MoE path and KE's dense MT 4/8 wide path next to the served ones
//!   recon    also time the dense wide reconstruct arm
//!   pdl      also time the served MoE decode with the PDL launch attribute
//!   graph    also time the served MoE decode replayed from a CUDA graph (PDL edges)
//!   none     no timing
//!   k3 / k2, tp1 / tp2 / tp4 / tp8, rows1 / rows6 / ...: restrict the MoE decode table to that width / slice / row count
//! Default (flag absent): decode,shared,prefill,dense.

use super::moe::{self, MoePath};
use super::probe::{dp, f2h, timed, Cx, Lin, Rng};
use super::Exl3Weights;
use crate::dsv41::weights::{Accum, Act, ActDtype, DenseWeight, DevPtr, Dsv41Weights, Launch, MoeWeight, Route, Scratch, Shard};
use anyhow::{ensure, Result};
use cudarc::driver::{sys, CudaSlice};

pub(super) struct Sel { pub dense_dec: bool, pub decode: bool, pub shared: bool, pub prefill: bool, pub dense: bool, pub old: bool, pub pdl: bool, pub graph: bool, pub recon: bool, pub moe: bool,
                        pub only_k: Option<usize>, pub only_tp: Option<usize>, pub only_rows: Option<usize> }
impl Sel {
    pub fn parse(v: Option<&str>) -> Sel {
        let v = v.unwrap_or("decode,shared,prefill,dense");
        let has = |k: &str| v.split(',').any(|x| x.trim() == k);
        let num = |p: &str| v.split(',').find_map(|x| x.trim().strip_prefix(p).and_then(|n| n.parse::<usize>().ok()));
        Sel { dense_dec: has("ddec"), decode: has("decode"), shared: has("shared"), prefill: has("prefill"), dense: has("dense"), old: has("old"), pdl: has("pdl"), graph: has("graph"), recon: has("recon"), moe: has("moe"),
              only_k: num("k"), only_tp: num("tp"), only_rows: num("rows") }
    }
}

pub(super) fn run(cx: &Cx, rng: &mut Rng, lines: &mut dyn FnMut(String), sel: &Sel) -> Result<()> {
    if sel.decode || sel.dense_dec { bench_dense(cx, rng, lines, sel)?; }
    if sel.decode || sel.moe { bench_moe_decode(cx, rng, lines, sel)?; }
    if sel.shared { bench_moe_shared(cx, rng, lines, sel)?; }
    if sel.prefill { bench_moe_prefill(cx, rng, lines, sel)?; }
    if sel.dense { bench_dense_wide(cx, rng, lines, sel)?; }
    Ok(())
}

/// PK4: the dense byte floor at the TP forward's shapes. Served path = `Exl3Weights::dense` (rows <= 32: the stream-K decode
/// class, 18_*); "graph" = the same calls replayed from one CUDA graph with PDL edges (how a captured round runs); "old" =
/// KE's split-K class (`dense_split`, 12_*), eager / graph. Grouped rows ("wo_a x N groups", "wq_a + wkv") run the N linears
/// as ONE call (`dense_dec::run_group`) next to N separate calls.
fn bench_dense(cx: &Cx, rng: &mut Rng, lines: &mut dyn FnMut(String), sel: &Sel) -> Result<()> {
    lines("BYTEFLOOR dense (resident slices, rotating copies > L2; time = every launch of the call; GB/s = trellis slice bytes / time; roofline 238 GB/s)".into());
    // (name, k2, inn, out, k range divisor (In slice), n range divisor (Out slice), linears in the call)
    let shapes: [(&str, usize, usize, usize, usize, usize, usize); 15] = [
        ("wq_a whole", 10, 5120, 1280, 1, 1, 1), ("wkv whole", 10, 5120, 512, 1, 1, 1), ("wq_a + wkv (1 call)", 10, 5120, 1280, 1, 1, 2),
        ("wq_b Out 1/2", 10, 1280, 32768, 1, 2, 1), ("wq_b Out 1/4", 10, 1280, 32768, 1, 4, 1), ("wq_b Out 1/8", 10, 1280, 32768, 1, 8, 1),
        ("wo_a 1 group", 10, 4096, 1024, 1, 1, 1), ("wo_a 2 groups", 10, 4096, 1024, 1, 1, 2), ("wo_a 4 groups", 10, 4096, 1024, 1, 1, 4),
        ("wo_b In 1/2", 10, 8192, 5120, 2, 1, 1), ("wo_b In 1/4", 10, 8192, 5120, 4, 1, 1), ("wo_b In 1/8", 10, 8192, 5120, 8, 1, 1),
        ("head Out 1/2", 12, 5120, 129_280, 1, 2, 1), ("head Out 1/4", 12, 5120, 129_280, 1, 4, 1), ("head Out 1/8", 12, 5120, 129_280, 1, 8, 1),
    ];
    for (name, k2, inn, out, kdiv, ndiv, ng) in shapes {
        let (kl, nl) = (inn / kdiv, ((out / ndiv) / 128).max(1) * 128);
        let pair = name.starts_with("wq_a + wkv");
        let one = |nl: usize| (kl / 16) * (nl / 16) * 8 * k2 * 2;
        let bytes = if pair { one(nl) + one(512) } else { one(nl) * ng };
        let copies = ((160usize << 20) / bytes).clamp(2, 32);
        // copies x linears, each its own resident slice tensor
        let mut lins: Vec<Vec<Lin>> = vec![];
        for _ in 0..copies {
            let mut v = vec![];
            for g in 0..ng {
                let n = if pair && g == 1 { 512 } else { nl };
                let tr: Vec<i16> = (0..(kl / 16) * (n / 16) * 8 * k2).map(|_| rng.u() as i16).collect();
                v.push(Lin::from_host(cx, k2, kl, n, tr, vec![f2h(1.0); kl], vec![f2h(1.0); n], false)?);
            }
            lins.push(v);
        }
        let ws: Vec<Vec<DenseWeight>> = lins.iter().map(|v| v.iter().map(|l| l.dense(name, l.out, kl, Shard::Whole)).collect()).collect();
        let ncols: Vec<usize> = ws[0].iter().map(|w| w.out as usize).collect();
        let ytot: usize = ncols.iter().sum();
        for rows in [1usize, 2, 6, 12, 24, 32] {
            if sel.only_rows.map_or(false, |r| r != rows) { continue; }
            let dx = cx.zeros::<f32>(rows * kl * ng)?; let dy = cx.zeros::<f32>(rows * ytot)?;
            let refs: Vec<&DenseWeight> = ws[0].iter().collect();
            let sb = super::dense_dec::scratch_bytes_for_group(&refs, rows).max(ws[0].iter().map(|w| super::dense::resolve(w, rows).map(|r| r.plan.scratch_bytes()).unwrap_or(0)).max().unwrap_or(0));
            let ds = cx.zeros::<u8>(sb)?;
            let sc = Scratch { ptr: DevPtr(dp(&ds)), bytes: sb };
            // linear g reads x columns [g * kl, (g + 1) * kl) (wq_a + wkv: the same x) and writes y columns at its offset
            let acts = |g: usize| -> (Act, Act) {
                let xo = if pair { 0 } else { g * kl };
                let yo: usize = ncols[..g].iter().sum();
                (Act { ptr: DevPtr(dp(&dx) + (xo * 4) as u64), rows, cols: kl, ld: if pair { kl } else { kl * ng }, dtype: ActDtype::F32 },
                 Act { ptr: DevPtr(dp(&dy) + (yo * 4) as u64), rows, cols: ncols[g], ld: ytot, dtype: ActDtype::F32 })
            };
            let served = |l: Launch, i: usize| -> Result<()> {
                let w = &ws[i % copies];
                if ng == 1 { let (x, y) = acts(0); return Exl3Weights.dense(&l, &w[0], &x, &y, Accum::Overwrite, sc).map(|_| ()); }
                let av: Vec<(Act, Act)> = (0..ng).map(acts).collect();
                let calls: Vec<super::dense_dec::Call> = (0..ng).map(|g| super::dense_dec::Call::new(&w[g], &av[g].0, &av[g].1, Accum::Overwrite)).collect();
                if rows < super::dense::WIDE_MIN_ROWS { super::dense_dec::run_group(&l, &calls, sc) } else { for c in &calls { Exl3Weights.dense(&l, c.w, c.x, c.y, c.acc, sc)?; } Ok(()) }
            };
            let separate = |l: Launch, i: usize| -> Result<()> {
                let w = &ws[i % copies];
                for g in 0..ng { let (x, y) = acts(g); Exl3Weights.dense(&l, &w[g], &x, &y, Accum::Overwrite, sc)?; }
                Ok(())
            };
            let old = |l: Launch, i: usize| -> Result<()> {
                let w = &ws[i % copies];
                for g in 0..ng { let (x, y) = acts(g); super::dense::dense_split(&l, &w[g], &x, &y, Accum::Overwrite, sc)?; }
                Ok(())
            };
            let n_it = ((400usize << 20) / bytes).clamp(16, 200);
            let eager = Launch { stream: cx.stream, pdl: false };
            let pdl = Launch { stream: cx.stream, pdl: true };
            let mut cells = vec![];
            let us = timed(cx, n_it, |i| served(eager, i))?; cells.push(format!("served {us:>8.1} us {:>6.1} GB/s", bytes as f64 / us / 1e3));
            if sel.graph { let us = time_graph(cx, n_it, |i| served(pdl, i))?; cells.push(format!("graph+pdl {us:>8.1} us {:>6.1} GB/s", bytes as f64 / us / 1e3)); }
            if ng > 1 && rows < super::dense::WIDE_MIN_ROWS {
                if sel.graph { let us = time_graph(cx, n_it, |i| separate(pdl, i))?; cells.push(format!("{ng} calls graph+pdl {us:>8.1} us {:>6.1} GB/s", bytes as f64 / us / 1e3)); }
                else { let us = timed(cx, n_it, |i| separate(eager, i))?; cells.push(format!("{ng} calls {us:>8.1} us {:>6.1} GB/s", bytes as f64 / us / 1e3)); }
            }
            if sel.old && rows < super::dense::WIDE_MIN_ROWS {
                let us = timed(cx, n_it, |i| old(eager, i))?; cells.push(format!("old {us:>8.1} us {:>6.1} GB/s", bytes as f64 / us / 1e3));
                if sel.graph { let us = time_graph(cx, n_it, |i| old(eager, i))?; cells.push(format!("old graph {us:>8.1} us {:>6.1} GB/s", bytes as f64 / us / 1e3)); }
            }
            lines(format!("  {name:<20} K{} {kl:>5}x{nl:>6} {:>7.2} MB rows={rows:<2} {}", k2 / 2, bytes as f64 / 1e6, cells.join(" | ")));
        }
    }
    Ok(())
}

/// Time `n_it` calls captured once into a CUDA graph (PDL edges where the calls ask for them), best of 3 replays, per call.
pub(super) fn time_graph(cx: &Cx, n_it: usize, mut call: impl FnMut(usize) -> Result<()>) -> Result<f64> {
    call(0)?; cx.sync()?;
    let st = cx.stream as sys::CUstream;
    let mut g: sys::CUgraph = std::ptr::null_mut();
    let mut ge: sys::CUgraphExec = std::ptr::null_mut();
    unsafe {
        ensure!(sys::cuStreamBeginCapture_v2(st, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) == sys::CUresult::CUDA_SUCCESS, "begin capture");
        let mut r = Ok(());
        for i in 0..n_it { if let Err(e) = call(i) { r = Err(e); break; } }
        let ec = sys::cuStreamEndCapture(st, &mut g);
        r?;
        ensure!(ec == sys::CUresult::CUDA_SUCCESS, "end capture: {ec:?}");
        let ie = sys::cuGraphInstantiateWithFlags(&mut ge, g, 0);
        ensure!(ie == sys::CUresult::CUDA_SUCCESS, "graph instantiate: {ie:?}");
    }
    let mut best = f64::MAX;
    for _ in 0..3 {
        let us = timed(cx, 1, |_| { ensure!(unsafe { sys::cuGraphLaunch(ge, st) } == sys::CUresult::CUDA_SUCCESS, "graph launch"); Ok(()) })?;
        best = best.min(us);
    }
    unsafe { sys::cuGraphExecDestroy(ge); sys::cuGraphDestroy(g); }
    Ok(best / n_it as f64)
}

// ------------------------------------------------------------------ MoE pools ---------------------------------------------------
/// A pool of `n` synthetic experts with the inner slice of TP `tp` (expert e holds `nb_of(tp, e)` 128-blocks), width k2/2.
pub(super) struct Pool { pub lins: Vec<[Lin; 3]>, pub mw: MoeWeight, pub tp: usize, pub k2: usize }
pub(super) fn nb_of(tp: usize, e: usize) -> usize { match tp { 1 => 18, 2 => 9, 4 => if e % 2 == 0 { 5 } else { 4 }, _ => if e % 4 == 0 { 3 } else { 2 } } }
/// Trellis bytes of one expert slice (w1 + w3 + w2).
pub(super) fn expert_bytes(tp: usize, k2: usize, e: usize, dim: usize) -> f64 { 3.0 * ((dim / 16) * nb_of(tp, e) * 8 * 8 * k2 * 2) as f64 }

pub(super) fn pool(cx: &Cx, rng: &mut Rng, n: usize, tp: usize, k2: usize, top_k: usize, layer: u16) -> Result<Pool> {
    let mut lins: Vec<[Lin; 3]> = vec![];
    let mut cache: std::collections::HashMap<usize, (Vec<i16>, Vec<i16>)> = Default::default();
    for e in 0..n {
        let nb = nb_of(tp, e);
        let c = cache.entry(nb).or_insert_with(|| {
            let t13: Vec<i16> = (0..320 * nb * 8 * 8 * k2).map(|_| rng.u() as i16).collect();
            let t2: Vec<i16> = (0..nb * 8 * 320 * 8 * k2).map(|_| rng.u() as i16).collect();
            (t13, t2)
        });
        let one = vec![f2h(1.0); 5120]; let sl = vec![f2h(1.0); nb * 128];
        lins.push([Lin::from_host(cx, k2, 5120, nb * 128, c.0.clone(), one.clone(), sl.clone(), false)?,
                   Lin::from_host(cx, k2, nb * 128, 5120, c.1.clone(), sl.clone(), one.clone(), false)?,
                   Lin::from_host(cx, k2, 5120, nb * 128, c.0.clone(), one, sl, false)?]);
    }
    let experts: Vec<[DenseWeight; 3]> = lins.iter().enumerate().map(|(e, l)| { let nb = nb_of(tp, e) as u64;
        let mut d = [l[0].dense("w1", nb as usize * 128, 5120, Shard::Whole), l[1].dense("w2", 5120, nb as usize * 128, Shard::Whole), l[2].dense("w3", nb as usize * 128, 5120, Shard::Whole)];
        // slice-resident tensors with the logical (full) shape: the shard names the slice; logical inter stays 2304
        d[0].out = 2304; d[2].out = 2304; d[1].inn = 2304;
        d[0].shard = Shard::Out(0..nb * 128); d[2].shard = Shard::Out(0..nb * 128); d[1].shard = Shard::In(0..nb * 128); d }).collect();
    let mw = MoeWeight { layer, n_experts: n, local: 0..n, top_k, dim: 5120, inter: 2304, swiglu_limit: 10.0, experts };
    Ok(Pool { lins, mw, tp, k2 })
}

fn random_ids(rng: &mut Rng, rows: usize, n_ex: usize, top_k: usize) -> Vec<i32> {
    let mut ids = vec![];
    for _ in 0..rows { let mut pool: Vec<usize> = (0..n_ex).collect(); for _ in 0..top_k { let i = rng.below(pool.len()); ids.push(pool.remove(i) as i32); } }
    ids
}

/// Time `iters` back-to-back MoE calls (ids cycling through the given sets) on `path`; eager launches, PDL, or one CUDA graph.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Mode { Eager, Pdl, Graph }
fn time_moe(cx: &Cx, mw: &MoeWeight, shared: Option<&MoeWeight>, rows: usize, idsets: &[CudaSlice<i32>], dw: &CudaSlice<f32>,
            xa: &Act, ya: &Act, ds: &CudaSlice<u8>, sb: usize, path: MoePath, mode: Mode) -> Result<f64> {
    let n_it = idsets.len();
    let l = Launch { stream: cx.stream, pdl: mode != Mode::Eager };
    let call = |i: usize| moe::moe_ex(&l, mw, shared, xa, &Route { ids: DevPtr(dp(&idsets[i % n_it])), weights: DevPtr(dp(dw)), top_k: mw.top_k }, ya,
                                      Scratch { ptr: DevPtr(dp(ds)), bytes: sb }, path).map(|_| ());
    let _ = rows;
    if mode != Mode::Graph { return timed(cx, n_it, call); }
    // graph: capture n_it calls once, replay it a few times
    call(0)?; cx.sync()?;
    let st = cx.stream as sys::CUstream;
    let mut g: sys::CUgraph = std::ptr::null_mut();
    let mut ge: sys::CUgraphExec = std::ptr::null_mut();
    unsafe {
        ensure!(sys::cuStreamBeginCapture_v2(st, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) == sys::CUresult::CUDA_SUCCESS, "begin capture");
        let mut r = Ok(());
        for i in 0..n_it { if let Err(e) = call(i) { r = Err(e); break; } }
        let ec = sys::cuStreamEndCapture(st, &mut g);
        r?;
        ensure!(ec == sys::CUresult::CUDA_SUCCESS, "end capture: {ec:?}");
        let ie = sys::cuGraphInstantiateWithFlags(&mut ge, g, 0);
        ensure!(ie == sys::CUresult::CUDA_SUCCESS, "graph instantiate: {ie:?}");
    }
    let reps = 3usize;
    let us = timed(cx, reps, |_| { ensure!(unsafe { sys::cuGraphLaunch(ge, st) } == sys::CUresult::CUDA_SUCCESS, "graph launch"); Ok(()) })?;
    unsafe { sys::cuGraphExecDestroy(ge); sys::cuGraphDestroy(g); }
    Ok(us / n_it as f64)
}

fn bench_moe_decode(cx: &Cx, rng: &mut Rng, lines: &mut dyn FnMut(String), sel: &Sel) -> Result<()> {
    let n_ex = 48usize;
    for (k2, kname) in [(6usize, "K3"), (4, "K2")] {
        if sel.only_k.map_or(false, |k| k * 2 != k2) { continue; }
        lines(format!("BYTEFLOOR routed MoE decode, {kname}, {n_ex} experts resident as slices, random top-6 per row each iteration; bytes = distinct experts x (w1+w3+w2 slice); the whole call (all launches) per layer"));
        for (label, tp) in [("full 18/18 blocks", 1usize), ("TP2 9 blocks", 2), ("TP4 5/4 blocks", 4), ("TP8 3/2 blocks", 8)] {
            if sel.only_tp.map_or(false, |t| t != tp) { continue; }
            let p = pool(cx, rng, n_ex, tp, k2, 6, 24)?;
            moe::clear_tables();
            for rows in [1usize, 2, 6, 12, 24] {
                if sel.only_rows.map_or(false, |r| r != rows) { continue; }
                let n_it = 20usize;
                let mut idsets: Vec<Vec<i32>> = vec![]; let mut bytes_tot = 0f64;
                for _ in 0..n_it {
                    let ids = random_ids(rng, rows, n_ex, 6);
                    let seen: std::collections::BTreeSet<i32> = ids.iter().copied().collect();
                    bytes_tot += seen.iter().map(|&e| expert_bytes(tp, k2, e as usize, 5120)).sum::<f64>(); idsets.push(ids);
                }
                let dids: Vec<CudaSlice<i32>> = idsets.iter().map(|v| cx.up(v)).collect::<Result<_>>()?;
                let dw = cx.up(&vec![0.2f32; rows * 6])?; let dx = cx.zeros::<f32>(rows * 5120)?; let dy = cx.zeros::<f32>(rows * 5120)?;
                let sb = moe::scratch_for(&p.mw, None, rows, MoePath::Auto)?.max(moe::scratch_for(&p.mw, None, rows, MoePath::Generic)?);
                let ds = cx.zeros::<u8>(sb)?;
                let xa = Act { ptr: DevPtr(dp(&dx)), rows, cols: 5120, ld: 5120, dtype: ActDtype::F32 }; let ya = Act { ptr: DevPtr(dp(&dy)), rows, cols: 5120, ld: 5120, dtype: ActDtype::F32 };
                let per = bytes_tot / n_it as f64;
                let mut cells = vec![];
                let mut modes = vec![(MoePath::Auto, Mode::Eager, "served")];
                if sel.pdl { modes.push((MoePath::Auto, Mode::Pdl, "served+pdl")); }
                if sel.graph { modes.push((MoePath::Auto, Mode::Graph, "served graph+pdl")); }
                if sel.old { modes.push((MoePath::Generic, Mode::Eager, "generic")); }
                for (path, mode, nm) in modes {
                    let us = time_moe(cx, &p.mw, None, rows, &dids, &dw, &xa, &ya, &ds, sb, path, mode)?;
                    cells.push(format!("{nm} {us:>7.1} us {:>6.1} GB/s", per / us / 1e3));
                }
                lines(format!("  {kname} {label:<18} rows={rows:<2} {:>6.2} MB/iter  {}", per / 1e6, cells.join(" | ")));
            }
            drop(p);
            moe::clear_tables();
        }
    }
    Ok(())
}

/// Routed experts + the shared expert (K4, inner slice of the same TP) in one MoE call.
fn bench_moe_shared(cx: &Cx, rng: &mut Rng, lines: &mut dyn FnMut(String), sel: &Sel) -> Result<()> {
    let n_ex = 48usize;
    lines("BYTEFLOOR routed (K3, 48 experts) + shared expert (K4, its inner slice) in one call; bytes = distinct routed slices + the shared slice".into());
    for (label, tp) in [("TP2 9 blocks", 2usize), ("TP4 5/4 blocks", 4)] {
        let p = pool(cx, rng, n_ex, tp, 6, 6, 24)?;
        let sh = pool(cx, rng, 1, tp, 8, 1, 24)?;
        moe::clear_tables();
        for rows in [1usize, 2, 6, 12, 24] {
            let n_it = 20usize;
            let mut idsets: Vec<Vec<i32>> = vec![]; let mut bytes_tot = 0f64;
            for _ in 0..n_it {
                let ids = random_ids(rng, rows, n_ex, 6);
                let seen: std::collections::BTreeSet<i32> = ids.iter().copied().collect();
                bytes_tot += seen.iter().map(|&e| expert_bytes(tp, 6, e as usize, 5120)).sum::<f64>() + expert_bytes(tp, 8, 0, 5120); idsets.push(ids);
            }
            let dids: Vec<CudaSlice<i32>> = idsets.iter().map(|v| cx.up(v)).collect::<Result<_>>()?;
            let dw = cx.up(&vec![0.2f32; rows * 6])?; let dx = cx.zeros::<f32>(rows * 5120)?; let dy = cx.zeros::<f32>(rows * 5120)?;
            let sb = moe::scratch_for(&p.mw, Some(&sh.mw), rows, MoePath::Auto)?.max(moe::scratch_for(&p.mw, Some(&sh.mw), rows, MoePath::Generic)?);
            let ds = cx.zeros::<u8>(sb)?;
            let xa = Act { ptr: DevPtr(dp(&dx)), rows, cols: 5120, ld: 5120, dtype: ActDtype::F32 }; let ya = Act { ptr: DevPtr(dp(&dy)), rows, cols: 5120, ld: 5120, dtype: ActDtype::F32 };
            let per = bytes_tot / n_it as f64;
            let mut cells = vec![];
            let mut modes = vec![(MoePath::Auto, Mode::Eager, "served")];
            if sel.pdl { modes.push((MoePath::Auto, Mode::Pdl, "served+pdl")); }
            if sel.graph { modes.push((MoePath::Auto, Mode::Graph, "served graph+pdl")); }
            if sel.old { modes.push((MoePath::Generic, Mode::Eager, "generic x2 calls")); }
            for (path, mode, nm) in modes {
                let us = time_moe(cx, &p.mw, Some(&sh.mw), rows, &dids, &dw, &xa, &ya, &ds, sb, path, mode)?;
                cells.push(format!("{nm} {us:>7.1} us {:>6.1} GB/s", per / us / 1e3));
            }
            lines(format!("  K3+K4sh {label:<15} rows={rows:<2} {:>6.2} MB/iter  {}", per / 1e6, cells.join(" | ")));
        }
        moe::clear_tables();
    }
    Ok(())
}

/// Prefill: 384 experts, top-6 uniform random (distinct per row), bf16 input rows, M rows per rank.
fn bench_moe_prefill(cx: &Cx, rng: &mut Rng, lines: &mut dyn FnMut(String), sel: &Sel) -> Result<()> {
    lines("PREFILL routed MoE (384 experts, K3, top-6 uniform random, bf16 x), one call per M rows; TFLOP/s = sum over pairs of 3 x 2 x 5120 x slice / time (expert GEMM FLOPs; the whole call incl. plan / transforms / combine)".into());
    for (label, tp) in [("TP4 5/4 blocks", 4usize), ("TP2 9 blocks", 2)] {
        let p = pool(cx, rng, 384, tp, 6, 6, 24)?;
        moe::clear_tables();
        for m in [2048usize, 8192, 16384] {
            let ids = random_ids(rng, m, 384, 6);
            let flop: f64 = ids.iter().map(|&e| 3.0 * 2.0 * 5120.0 * (nb_of(tp, e as usize) * 128) as f64).sum();
            let di = cx.up(&ids)?; let dw = cx.up(&(0..m * 6).map(|_| 0.05 + 0.2 * rng.f().abs()).collect::<Vec<f32>>())?;
            let xb: Vec<u16> = (0..m * 5120).map(|_| half::bf16::from_f32(rng.f()).to_bits()).collect();
            let dx = cx.up(&xb)?; let dy = cx.zeros::<f32>(m * 5120)?;
            let xa = Act { ptr: DevPtr(dp(&dx)), rows: m, cols: 5120, ld: 5120, dtype: ActDtype::Bf16 }; let ya = Act { ptr: DevPtr(dp(&dy)), rows: m, cols: 5120, ld: 5120, dtype: ActDtype::F32 };
            let mut cells = vec![];
            let mut paths = vec![(MoePath::Auto, "served")];
            if sel.old { paths.push((MoePath::Generic, "generic (32-row chunks)")); }
            for (path, nm) in paths {
                let sb = moe::scratch_for(&p.mw, None, m, path)?;
                let ds = cx.zeros::<u8>(sb)?;
                let reps = if path == MoePath::Generic { 1 } else { 3 };
                let us = timed(cx, reps, |_| moe::moe_ex(&cx.l(), &p.mw, None, &xa, &Route { ids: DevPtr(dp(&di)), weights: DevPtr(dp(&dw)), top_k: 6 }, &ya,
                                                          Scratch { ptr: DevPtr(dp(&ds)), bytes: sb }, path).map(|_| ()))?;
                cells.push(format!("{nm} {:>9.2} ms {:>6.1} TFLOP/s", us / 1e3, flop / us / 1e6));
            }
            lines(format!("  {label:<15} M={m:<5} {:>7.1} GFLOP  {}", flop / 1e9, cells.join(" | ")));
        }
        drop(p);
        moe::clear_tables();
    }
    Ok(())
}

/// Dense wide (> 32 rows): the attention linears at prefill shapes.
fn bench_dense_wide(cx: &Cx, rng: &mut Rng, lines: &mut dyn FnMut(String), sel: &Sel) -> Result<()> {
    lines("PREFILL dense wide (rows > 32), bf16 x, f32 y; TFLOP/s = 2 M K N / time (the whole call)".into());
    // (name, k2, K, N)
    let shapes: [(&str, usize, usize, usize); 5] = [("wq_a K5", 10, 5120, 1280), ("wq_b K5 1/4", 10, 1280, 8192), ("wo_a K5 group", 10, 4096, 1024), ("wo_b K5 In 1/4", 10, 2048, 5120), ("wkv K5", 10, 5120, 512)];
    for (name, k2, kl, nl) in shapes {
        let tr: Vec<i16> = (0..(kl / 16) * (nl / 16) * 8 * k2).map(|_| rng.u() as i16).collect();
        let lin = Lin::from_host(cx, k2, kl, nl, tr, vec![f2h(1.0); kl], vec![f2h(1.0); nl], false)?;
        let w = lin.dense(name, nl, kl, Shard::Whole);
        for m in [2048usize, 8192, 16384] {
            let xb: Vec<u16> = (0..m * kl).map(|_| half::bf16::from_f32(rng.f()).to_bits()).collect();
            let dx = cx.up(&xb)?; let dy = cx.zeros::<f32>(m * nl)?;
            // the served bound and the Recon arm's (run below at wide rows) in one scratch
            let sb = Exl3Weights.scratch_bytes(&w, m).max(if m >= super::dense::WIDE_MIN_ROWS { super::dense_wide::scratch_bytes(&w, m, super::dense_wide::WideSrc::Recon)? } else { 0 });
            let ds = cx.zeros::<u8>(sb)?;
            let xa = Act { ptr: DevPtr(dp(&dx)), rows: m, cols: kl, ld: kl, dtype: ActDtype::Bf16 }; let ya = Act { ptr: DevPtr(dp(&dy)), rows: m, cols: nl, ld: nl, dtype: ActDtype::F32 };
            let fl = 2.0 * m as f64 * kl as f64 * nl as f64;
            let mut cells = vec![];
            let us = timed(cx, 3, |_| Exl3Weights.dense(&cx.l(), &w, &xa, &ya, Accum::Overwrite, Scratch { ptr: DevPtr(dp(&ds)), bytes: sb }).map(|_| ()))?;
            cells.push(format!("served {:>8.3} ms {:>6.1} TFLOP/s", us / 1e3, fl / us / 1e6));
            if sel.old || sel.recon {
                let us = timed(cx, 3, |_| super::dense_wide::dense_wide(&cx.l(), &w, &xa, &ya, Accum::Overwrite, Scratch { ptr: DevPtr(dp(&ds)), bytes: sb }, super::dense_wide::WideSrc::Recon).map(|_| ()))?;
                cells.push(format!("reconstruct arm {:>8.3} ms {:>6.1}", us / 1e3, fl / us / 1e6));
            }
            if sel.old {
                let so = Exl3Weights.scratch_bytes(&w, 32).max(super::launch::DensePlan::new(m, kl, 0, nl, 0).scratch_bytes()); let dso = cx.zeros::<u8>(so)?;
                let us = timed(cx, 3, |_| super::dense::dense_split(&cx.l(), &w, &xa, &ya, Accum::Overwrite, Scratch { ptr: DevPtr(dp(&dso)), bytes: so }).map(|_| ()))?;
                cells.push(format!("KE MT4/8 {:>8.3} ms {:>6.1}", us / 1e3, fl / us / 1e6));
            }
            lines(format!("  {name:<16} {kl:>5}x{nl:<5} M={m:<5} {}", cells.join(" | ")));
        }
    }
    Ok(())
}
