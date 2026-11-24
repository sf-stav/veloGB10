//! S7 probe: `--probe-inkling2-full --model-dir <stock pack> --inkling2-probe-only s7[,<opts>]` (the existing flags;
//! the `s7` spec is read from the command line here, so no new flag is needed). Real-text lanes (prose + code from
//! this repository, tokenized with the pack's tokenizer), then:
//!   * GATE verify: every row of `verify_lanes` (1x9, 2x3, 4x5, 4x9 rows, eager and graph) vs a 1-row `decode_lanes`
//!     of the same token at the same position, bitwise; graph == eager; decode graph == eager.
//!   * TIMING (untraced, CUDA graph, greedy, ~2.1K ctx): the 1-row step, 4-lane step and the verify-width table
//!     (1 lane x 1..9, 4 lanes x 1..9 rows), PDL off then on; the S4 verify path (`forward`, eager) alongside.
//!   * prefill timing per 2,048-row chunk up to `pf=<ctx>` (default 32768) in one lane.
//! Options: `nopdl` (timing with PDL off only), `ledger=dec|<L>x<M>|pf<ctx>` (wrap 20 replays of that shape, or the
//! last prefill chunk before <ctx>, in cuProfilerStart / Stop for `nsys --capture-range=cudaProfilerApi`),
//! `notime`, `nogate`, `pf=<ctx>`.

use super::{GpuModel, Lane, Seg, Work};
use anyhow::{bail, Context, Result};
use cudarc::driver::CudaDevice;
use std::path::Path;

/// Real prose for the probe's long prompt: two repo docs, read when the probe runs (not compiled in, so the binary's
/// source stamp does not depend on docs).
fn prose() -> anyhow::Result<String> {
    let d = std::path::Path::new("docs");
    Ok(std::fs::read_to_string(d.join("FAILURE_REPORT_2026-08-27.md"))? + &std::fs::read_to_string(d.join("RIVAL_ANALYSIS_STANDARD.md"))?)
}
const CODE: &str = include_str!("../cpu.rs");

fn prefill(g: &GpuModel, ws: &mut Work, lanes: &mut [Lane], li: usize, ids: &[u32]) -> Result<()> {
    let mut i = 0;
    while i < ids.len() {
        let m = (ids.len() - i).min(super::CHUNK);
        let p0 = lanes[li].pos;
        { let refs: Vec<&Lane> = lanes.iter().collect(); g.forward(ws, &refs, &[Seg { lane: li, p0, m }], &ids[i..i + m], &[])?; }
        lanes[li].pos += m;
        i += m;
    }
    Ok(())
}

fn med(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }

pub fn run(dir: &Path, spec: &str) -> Result<()> {
    let t0 = std::time::Instant::now();
    let opt = |k: &str| spec.split(',').any(|s| s == k);
    let val = |k: &str| spec.split(',').find_map(|s| s.strip_prefix(&format!("{k}=")).map(|x| x.to_string()));
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    let mut g = GpuModel::load(dir, &dev)?;
    let mut ws = g.new_work()?;
    let v = g.vocab();
    let tok = crate::tokenizer::QwenTokenizer::from_file(dir.join("tokenizer.json").to_str().context("path")?)?;
    let (pr, co) = (tok.encode(&prose()?, false)?, tok.encode(CODE, false)?);
    ensure_len(&pr, 3000, "prose")?;
    ensure_len(&co, 3000, "code")?;
    // lane l: 2,048 prompt tokens + continuation (prose for even lanes, code for odd), offsets differ per lane
    let text = |l: usize| -> Vec<u32> { let src = if l % 2 == 0 { &pr } else { &co }; let o = (l / 2) * 400; src[o..o + 2600].to_vec() };
    let mut lanes: Vec<Lane> = Vec::new();
    for l in 0..4 {
        lanes.push(g.new_lane(8192)?);
        prefill(&g, &mut ws, &mut lanes, l, &text(l)[..2048])?;
    }
    g.bind_lanes(&mut ws, &lanes)?;
    g.k.sync()?;
    println!("INK2_S7 up in {:.1}s: 4 lanes x 2,048 real-text tokens (prose {} / code {} tokens available)", t0.elapsed().as_secs_f64(), pr.len(), co.len());
    let base: Vec<usize> = lanes.iter().map(|l| l.pos).collect();
    let cont = |l: usize, i: usize| text(l)[2048 + i];
    let mut fails = 0usize;
    // ---------------------------------------------------------------- gates
    if !opt("nogate") {
        for &(nl, m) in &[(1usize, 9usize), (2, 3), (4, 5), (4, 9), (3, 16)] {
            for (l, ln) in lanes.iter_mut().enumerate() { ln.pos = base[l]; }
            let toks: Vec<Vec<u32>> = (0..nl).map(|l| (0..m).map(|i| cont(l, i)).collect()).collect();
            let segs: Vec<(usize, &[u32])> = toks.iter().enumerate().map(|(l, t)| (l, &t[..])).collect();
            let mut res = Vec::new();
            for gr in [false, true] {
                g.verify_lanes(&mut ws, &lanes, &segs, gr)?;
                res.push(g.logits_host(&ws, nl * m)?);
            }
            // reference: 1-row decode steps (all nl lanes per step, graph), row (l, t) at position base + t
            let mut refl = vec![0f32; nl * m * v];
            for t in 0..m {
                let rows: Vec<(usize, u32)> = (0..nl).map(|l| (l, toks[l][t])).collect();
                g.decode_lanes(&mut ws, &lanes, &rows, true)?;
                let lg = g.logits_host(&ws, nl)?;
                for l in 0..nl { refl[(l * m + t) * v..(l * m + t + 1) * v].copy_from_slice(&lg[l * v..(l + 1) * v]); }
                for ln in lanes.iter_mut().take(nl) { ln.pos += 1; }
            }
            let d = |a: &[f32], b: &[f32]| a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            let (de, dg) = (d(&res[0], &refl), d(&res[1], &res[0]));
            println!("INK2_S7 gate verify {nl} lane(s) x {m} rows: every row vs 1-row decode {de} of {} logits differ; graph vs eager {dg} differ (bitwise)", refl.len());
            if de + dg != 0 { fails += 1; }
        }
        for (l, ln) in lanes.iter_mut().enumerate() { ln.pos = base[l]; }
    }
    // ---------------------------------------------------------------- timing
    let reps = 16;
    let time_verify = |g: &GpuModel, ws: &mut Work, lanes: &mut [Lane], nl: usize, m: usize, graph: bool| -> Result<f64> {
        let toks: Vec<Vec<u32>> = (0..nl).map(|l| (0..m).map(|i| cont(l, 100 + i)).collect()).collect();
        let segs: Vec<(usize, &[u32])> = toks.iter().enumerate().map(|(l, t)| (l, &t[..])).collect();
        let mut tt = Vec::new();
        for i in 0..reps + 3 {
            let t = std::time::Instant::now();
            g.verify_lanes(ws, lanes, &segs, graph)?;
            g.k.sync()?;
            if i >= 3 { tt.push(t.elapsed().as_secs_f64() * 1e3); }
        }
        Ok(med(tt))
    };
    if let Some(ld) = val("ledger") {
        if opt("nopdl") { g.k.pdl.set(false); g.bind_lanes(&mut ws, &lanes)?; }
        if let Some(c) = ld.strip_prefix("pf") {
            let ctx: usize = c.parse()?;
            let mut ln = vec![g.new_lane(ctx + 16)?];
            let mut ids = Vec::new();
            while ids.len() < ctx { ids.extend_from_slice(if ids.len() % 2 == 0 { &pr } else { &co }); }
            ids.truncate(ctx);
            prefill(&g, &mut ws, &mut ln, 0, &ids[..ctx - super::CHUNK])?;
            g.k.sync()?;
            unsafe { cudarc::driver::sys::cuProfilerStart(); }
            prefill(&g, &mut ws, &mut ln, 0, &ids[ctx - super::CHUNK..])?;
            g.k.sync()?;
            unsafe { cudarc::driver::sys::cuProfilerStop(); }
            println!("INK2_S7 ledger: one 2,048-row prefill chunk at {ctx} captured");
        } else {
            let (nl, m) = if ld == "dec" { (1, 1) } else {
                let (a, b) = ld.split_once('x').context("ledger=<L>x<M>")?;
                (a.parse()?, b.parse()?)
            };
            time_verify(&g, &mut ws, &mut lanes, nl, m, true)?;
            let toks: Vec<Vec<u32>> = (0..nl).map(|l| (0..m).map(|i| cont(l, 100 + i)).collect()).collect();
            let segs: Vec<(usize, &[u32])> = toks.iter().enumerate().map(|(l, t)| (l, &t[..])).collect();
            unsafe { cudarc::driver::sys::cuProfilerStart(); }
            for _ in 0..20 { g.verify_lanes(&mut ws, &lanes, &segs, true)?; g.k.sync()?; }
            unsafe { cudarc::driver::sys::cuProfilerStop(); }
            println!("INK2_S7 ledger: 20 graph replays of {nl} lane(s) x {m} rows captured");
        }
    }
    // ---------------------------------------------------------------- A/B of the verify switches, interleaved
    if opt("ab") {
        use super::super::gpu::{k1::ATTN_MR, moe::{ROUTER_SMALL, VERIFY_KERNELS}};
        let set = |c: usize| { ATTN_MR.with(|x| x.set(c & 1 != 0)); ROUTER_SMALL.with(|x| x.set(c & 2 != 0)); VERIFY_KERNELS.with(|x| x.set(c & 4 != 0)); };
        let cfgs = [(7usize, "all on"), (6, "mr attention off"), (5, "small router tiles off"), (3, "expert verify tiles off")];
        for &(nl, m) in &[(1usize, 2usize), (1, 5), (1, 9), (4, 2), (4, 5), (4, 9)] {
            for (l, ln) in lanes.iter_mut().enumerate() { ln.pos = base[l]; }
            let toks: Vec<Vec<u32>> = (0..nl).map(|l| (0..m).map(|i| cont(l, 100 + i)).collect()).collect();
            let segs: Vec<(usize, &[u32])> = toks.iter().enumerate().map(|(l, t)| (l, &t[..])).collect();
            let mut tt: Vec<Vec<f64>> = vec![Vec::new(); cfgs.len()];
            for rep in 0..14 {
                for (ci, &(c, _)) in cfgs.iter().enumerate() {
                    set(c);
                    let t = std::time::Instant::now();
                    g.verify_lanes(&mut ws, &lanes, &segs, true)?;
                    g.k.sync()?;
                    if rep >= 2 { tt[ci].push(t.elapsed().as_secs_f64() * 1e3); }
                }
            }
            let mut line = format!("INK2_S7 A/B verify {nl} lane(s) x {m} rows (graph, interleaved, median of 12):");
            for (ci, &(_, name)) in cfgs.iter().enumerate() { line += &format!(" {name} {:.2} ms;", med(tt[ci].clone())); }
            println!("{line}");
        }
        set(7);
        for (l, ln) in lanes.iter_mut().enumerate() { ln.pos = base[l]; }
    }
    // long-context verify: one lane prefilled to `abctx`, 1 x {1, 5, 9} rows, mr attention on / off (interleaved)
    if let Some(cx) = val("abctx") {
        use super::super::gpu::k1::ATTN_MR;
        let ctx: usize = cx.parse()?;
        let mut ids = Vec::new();
        while ids.len() < ctx + 64 { ids.extend_from_slice(if (ids.len() / 1000) % 2 == 0 { &pr } else { &co }); }
        let mut ll = vec![g.new_lane(ctx + 64)?];
        prefill(&g, &mut ws, &mut ll, 0, &ids[..ctx])?;
        g.bind_lanes(&mut ws, &ll)?;
        for m in [1usize, 5, 9] {
            let toks: Vec<u32> = ids[ctx..ctx + m].to_vec();
            let mut tt = vec![Vec::new(), Vec::new()];
            for rep in 0..10 {
                for (ci, on) in [true, false].into_iter().enumerate() {
                    ATTN_MR.with(|x| x.set(on));
                    let t = std::time::Instant::now();
                    g.verify_lanes(&mut ws, &ll, &[(0, &toks[..])], true)?;
                    g.k.sync()?;
                    if rep >= 2 { tt[ci].push(t.elapsed().as_secs_f64() * 1e3); }
                }
            }
            ATTN_MR.with(|x| x.set(true));
            println!("INK2_S7 A/B verify at {ctx} ctx, 1 lane x {m} rows (graph, interleaved): mr attention on {:.2} ms, off {:.2} ms", med(tt[0].clone()), med(tt[1].clone()));
        }
        drop(ll);
        g.bind_lanes(&mut ws, &lanes)?;
    }
    if !opt("notime") {
        let modes: Vec<bool> = if opt("nopdl") { vec![false] } else { vec![false, true] };
        for &pdl in &modes {
            g.k.pdl.set(pdl);
            g.bind_lanes(&mut ws, &lanes)?; // drop the captured graphs: recapture with this launch mode
            // the step a server runs: graph replay + greedy sample (host read of the token), 64 steps
            let mut tok1 = cont(0, 0);
            let mut st = Vec::new();
            for i in 0..72 {
                let t = std::time::Instant::now();
                g.decode_lanes(&mut ws, &lanes, &[(0, tok1)], true)?;
                tok1 = g.sample(&mut ws, &[0.0], &[0])?[0];
                if i >= 8 { st.push(t.elapsed().as_secs_f64() * 1e3); }
            }
            let s1 = med(st);
            let mut st = Vec::new();
            for i in 0..40 {
                let rows: Vec<(usize, u32)> = (0..4).map(|l| (l, cont(l, i % 50))).collect();
                let t = std::time::Instant::now();
                g.decode_lanes(&mut ws, &lanes, &rows, true)?;
                g.sample(&mut ws, &[0.0; 4], &[0; 4])?;
                if i >= 8 { st.push(t.elapsed().as_secs_f64() * 1e3); }
            }
            let s4 = med(st);
            println!("INK2_S7 timing PDL {}: decode step 1 lane {s1:.2} ms ({:.1} tok/s), 4 lanes {s4:.2} ms ({:.1} tok/s aggregate) [graph + greedy sample, ~2.1K ctx, real text]",
                     if pdl { "on" } else { "off" }, 1e3 / s1, 4e3 / s4);
            for nl in [1usize, 4] {
                let mut line = format!("INK2_S7 timing PDL {}: verify graph {nl} lane(s) x m=1..9:", if pdl { "on" } else { "off" });
                for m in 1..=9 { line += &format!(" {:.2}", time_verify(&g, &mut ws, &mut lanes, nl, m, true)?); }
                println!("{line} ms");
            }
            for (l, ln) in lanes.iter_mut().enumerate() { ln.pos = base[l]; }
        }
        // the S4 verify path (forward, Segs, eager), PDL as set last
        for nl in [1usize, 4] {
            let mut line = format!("INK2_S7 timing: S4-path verify (forward, eager) {nl} lane(s) x m=1..9:");
            for m in 1..=9 {
                let segs: Vec<Seg> = (0..nl).map(|l| Seg { lane: l, p0: base[l], m }).collect();
                let tk: Vec<u32> = (0..nl).flat_map(|l| (0..m).map(move |i| (l, i))).map(|(l, i)| cont(l, 100 + i)).collect();
                let outr: Vec<usize> = (0..nl * m).collect();
                let mut tt = Vec::new();
                for i in 0..reps + 3 {
                    let t = std::time::Instant::now();
                    { let refs: Vec<&Lane> = lanes.iter().collect(); g.forward(&mut ws, &refs, &segs, &tk, &outr)?; }
                    g.k.sync()?;
                    if i >= 3 { tt.push(t.elapsed().as_secs_f64() * 1e3); }
                }
                line += &format!(" {:.2}", med(tt));
            }
            println!("{line} ms");
        }
        // prefill per chunk
        let pfctx: usize = val("pf").map(|x| x.parse()).transpose()?.unwrap_or(32768);
        if pfctx >= super::CHUNK {
            drop(lanes);
            let mut ln = vec![g.new_lane(pfctx + 16)?];
            let mut ids = Vec::new();
            while ids.len() < pfctx { ids.extend_from_slice(if (ids.len() / 1000) % 2 == 0 { &pr } else { &co }); }
            ids.truncate(pfctx / super::CHUNK * super::CHUNK);
            let mut tot = 0f64;
            let mut line = String::from("INK2_S7 timing: prefill per 2,048-row chunk (ms) at ctx");
            for (ci, ch) in ids.chunks(super::CHUNK).enumerate() {
                let t = std::time::Instant::now();
                prefill(&g, &mut ws, &mut ln, 0, ch)?;
                g.k.sync()?;
                let dt = t.elapsed().as_secs_f64();
                tot += dt;
                let end = (ci + 1) * super::CHUNK;
                if end.is_power_of_two() || end == ids.len() {
                    line += &format!(" {}K: chunk {:.0} / cum {:.0} tok/s;", end / 1024, dt * 1e3, end as f64 / tot);
                }
            }
            println!("{line}");
        }
    }
    // ---------------------------------------------------------------- opt-in single-pass attention vs exact
    if let Some(kl) = val("kl") {
        for ctx in kl.split('+').map(|x| x.parse::<usize>()).collect::<std::result::Result<Vec<_>, _>>()? {
            let mut ids = Vec::new();
            while ids.len() < ctx { ids.extend_from_slice(if (ids.len() / 1000) % 2 == 0 { &pr } else { &co }); }
            ids.truncate(ctx);
            let nr = 16usize;
            let mut res: Vec<Vec<f32>> = Vec::new();
            let mut times = Vec::new();
            for sp in [false, true] {
                g.attn_single_pass = sp;
                let mut ln = vec![g.new_lane(ctx + 16)?];
                let mut all = Vec::new();
                let t = std::time::Instant::now();
                let mut i = 0;
                while i < ids.len() {
                    let m = (ids.len() - i).min(super::CHUNK);
                    let outr: Vec<usize> = (0..nr).map(|j| (j * m) / nr + (m / nr) - 1).collect();
                    { let refs: Vec<&Lane> = ln.iter().collect(); g.forward(&mut ws, &refs, &[Seg { lane: 0, p0: i, m }], &ids[i..i + m], &outr)?; }
                    all.extend(g.logits_host(&ws, nr)?);
                    ln[0].pos += m;
                    i += m;
                }
                times.push(t.elapsed().as_secs_f64());
                res.push(all);
            }
            g.attn_single_pass = false;
            let (mut kls, mut agree, mut nrow) = (Vec::new(), 0usize, 0usize);
            for (a, b) in res[0].chunks(v).zip(res[1].chunks(v)) {
                let lse = |x: &[f32]| { let m = x.iter().cloned().fold(f32::MIN, f32::max) as f64; m + x.iter().map(|&y| (y as f64 - m).exp()).sum::<f64>().ln() };
                let (la, lb) = (lse(a), lse(b));
                let kl: f64 = a.iter().zip(b).map(|(&x, &y)| { let pa = (x as f64 - la).exp(); pa * ((x as f64 - la) - (y as f64 - lb)) }).sum();
                kls.push(kl);
                let am = |x: &[f32]| { let mut bi = 0; for i in 1..x.len() { if x[i] > x[bi] { bi = i; } } bi };
                if am(a) == am(b) { agree += 1; }
                nrow += 1;
            }
            let mean = kls.iter().sum::<f64>() / kls.len() as f64;
            kls.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("INK2_S7 single-pass vs exact attention, {ctx}-token real-text prompt ({nrow} rows sampled across all chunks): KL(exact||sp) mean {mean:.2e} p50 {:.2e} max {:.2e}; top-1 agree {agree}/{nrow}; prefill wall exact {:.2} s ({:.0} tok/s), single-pass {:.2} s ({:.0} tok/s)",
                     kls[kls.len() / 2], kls[kls.len() - 1], times[0], ctx as f64 / times[0], times[1], ctx as f64 / times[1]);
        }
    }
    if fails > 0 { bail!("INK2_S7_FAIL: {fails} gate(s) failed"); }
    println!("INK2_S7_OK ({:.0}s)", t0.elapsed().as_secs_f64());
    Ok(())
}

fn ensure_len(v: &[u32], n: usize, what: &str) -> Result<()> {
    if v.len() < n { bail!("{what} text has {} tokens (< {n})", v.len()); }
    Ok(())
}
