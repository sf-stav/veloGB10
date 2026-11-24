//! `--probe-inkling2-full --model-dir <stock pack> --oracle-dir <ref with cases.json>`: the full 48-visit GPU forward
//! vs the cpu.rs oracle on real prompts (prefill, then greedy decode steps fed the ORACLE's argmax to both sides so
//! one disagreement cannot cascade), plus GPU-only path checks:
//!   * chunked prefill (chunk 2,048 vs 512 vs 2,048 + single-token decode of the tail) on a > 2K prompt that crosses
//!     the sliding window: the three paths must pick the same next token, logits rel-L2 reported;
//!   * decode timing (eager, one lane) and prefill timing.
//! Gate: argmax identical on every compared step, logits rel-L2 <= 5e-2. The GPU feeds every linear a bf16 input (as
//! the bf16 reference does), the experts f16 activations and keeps an f16 K/V cache; vs the f32 oracle that is
//! ~1e-2..4e-2 on the final logits after 48 visits (measured, first S3 run); the values are printed per step.

use super::{GpuModel, Lane, Seg, Work};
use crate::inkling2::cpu::CpuModel;
use anyhow::{bail, Context, Result};
use cudarc::driver::CudaDevice;
use std::path::Path;

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut n, mut d) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) { n += (*x as f64 - *y as f64).powi(2); d += (*y as f64).powi(2); }
    (n / d.max(1e-300)).sqrt()
}
fn argmax(v: &[f32]) -> usize {
    let mut b = 0;
    for i in 1..v.len() { if v[i] > v[b] { b = i; } }
    b
}

/// One decode step of one lane: logits of the new token land in `ws.logits[0]`.
pub fn step1(g: &GpuModel, ws: &mut Work, lanes: &mut [Lane], li: usize, tok: u32) -> Result<()> {
    let p0 = lanes[li].pos;
    {
        let refs: Vec<&Lane> = lanes.iter().collect();
        g.forward(ws, &refs, &[Seg { lane: li, p0, m: 1 }], &[tok], &[0])?;
    }
    lanes[li].pos += 1;
    Ok(())
}

fn prefill_chunked(g: &GpuModel, ws: &mut Work, lanes: &mut [Lane], li: usize, ids: &[u32], chunk: usize) -> Result<()> {
    let mut i = 0;
    while i < ids.len() {
        let m = (ids.len() - i).min(chunk);
        let last = i + m == ids.len();
        let p0 = lanes[li].pos;
        let outr = [m - 1];
        {
            let refs: Vec<&Lane> = lanes.iter().collect();
            g.forward(ws, &refs, &[Seg { lane: li, p0, m }], &ids[i..i + m], if last { &outr[..] } else { &[][..] })?;
        }
        lanes[li].pos += m;
        i += m;
    }
    Ok(())
}

pub fn probe_full(dir: &Path, ref_dir: Option<&str>, steps: usize, decode_tc: bool, cpu: bool) -> Result<()> {
    // S7: `--inkling2-probe-only s7[,opts]` runs the S7 verify / timing probe instead (probe_s7.rs)
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--inkling2-probe-only") {
        if let Some(sp) = args.get(i + 1).filter(|s| s.starts_with("s7")) { return super::probe_s7::run(dir, sp); }
    }
    let t0 = std::time::Instant::now();
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    let mut g = GpuModel::load(dir, &dev)?;
    g.decode_tc = decode_tc;
    let mut ws = g.new_work()?;
    let v = g.vocab();
    let mut lanes = vec![g.new_lane(8192)?];
    println!("INK2_FULL GPU model up in {:.1}s (decode attention: {})", t0.elapsed().as_secs_f64(), if decode_tc { "ink_attn_tc" } else { "ink_attn" });
    let mut fails = 0usize;
    let mut worst = 0f64;
    // ---- 1. vs the CPU oracle
    if cpu {
        let rd = ref_dir.context("--oracle-dir <dir with cases.json> (the S1 real_ref set) is needed for the oracle comparison")?;
        let cases: serde_json::Value = serde_json::from_slice(&std::fs::read(Path::new(rd).join("cases.json"))?)?;
        let t1 = std::time::Instant::now();
        let m = CpuModel::load(dir)?;
        println!("INK2_FULL oracle loaded in {:.1}s", t1.elapsed().as_secs_f64());
        for case in cases.as_array().context("cases.json")? {
            let name = case["name"].as_str().unwrap_or("?");
            let prompt: Vec<u32> = case["steps"][0].as_array().context("steps")?.iter().map(|x| x.as_u64().unwrap() as u32).collect();
            let mut st = m.new_state();
            lanes[0].pos = 0;
            let tc = std::time::Instant::now();
            let f = m.forward(&mut st, &prompt, None)?;
            let mut lc = m.logits(&f[(prompt.len() - 1) * m.cfg.hidden..]);
            prefill_chunked(&g, &mut ws, &mut lanes, 0, &prompt, super::CHUNK)?;
            let mut lg = g.logits_host(&ws, 1)?;
            let mut agree = 0usize;
            let mut errs = Vec::new();
            let mut toks = Vec::new();
            for s in 0..=steps {
                let (ac, ag) = (argmax(&lc), argmax(&lg[..v]));
                let e = rel_l2(&lg[..v], &lc);
                errs.push(e);
                worst = worst.max(e);
                if ac == ag { agree += 1; } else {
                    println!("  {name} step {s}: ARGMAX MISMATCH cpu {ac} ({:.4}) gpu {ag} ({:.4}); cpu logit of gpu pick {:.4}", lc[ac], lg[ag], lc[ag]);
                    fails += 1;
                }
                if e > 5e-2 { fails += 1; println!("  {name} step {s}: logits rel-L2 {e:.3e} > 5e-2"); }
                toks.push(ac as u32);
                if s == steps { break; }
                let f = m.forward(&mut st, &[ac as u32], None)?;
                lc = m.logits(&f);
                step1(&g, &mut ws, &mut lanes, 0, ac as u32)?;
                lg = g.logits_host(&ws, 1)?;
            }
            let mean = errs.iter().sum::<f64>() / errs.len() as f64;
            println!("INK2_FULL {name}: prompt {} tok + {steps} decode steps: argmax {agree}/{} identical; logits rel-L2 mean {mean:.2e} max {:.2e} ({:.0}s)",
                     prompt.len(), steps + 1, errs.iter().cloned().fold(0f64, f64::max), tc.elapsed().as_secs_f64());
            println!("  tokens: {toks:?}");
        }
    }
    // ---- 2. GPU path consistency on a long prompt (crosses the 1,024 window and a 2,048 chunk boundary)
    let n = 3000usize;
    let mut s = 0x1234_5678u64;
    let ids: Vec<u32> = (0..n).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; 1000 + (s % 150_000) as u32 }).collect();
    let mut outs = Vec::new();
    for (tag, chunk, tail) in [("chunk2048", 2048usize, 0usize), ("chunk512", 512, 0), ("chunk2048+decode24", 2048, 24)] {
        lanes[0].pos = 0;
        prefill_chunked(&g, &mut ws, &mut lanes, 0, &ids[..n - tail], chunk)?;
        for i in n - tail..n { step1(&g, &mut ws, &mut lanes, 0, ids[i])?; }
        let lg = g.logits_host(&ws, 1)?[..v].to_vec();
        println!("INK2_FULL path {tag}: next {} ({:.4})", argmax(&lg), lg[argmax(&lg)]);
        outs.push(lg);
    }
    // S4: the 2-tile prefill attention (ink_attn_tc2) vs the 1-tile one, whole 3,000-token prefill, bitwise
    {
        super::super::gpu::k1::ATTN_TC2.with(|c| c.set(false));
        lanes[0].pos = 0;
        prefill_chunked(&g, &mut ws, &mut lanes, 0, &ids, 2048)?;
        let l1 = g.logits_host(&ws, 1)?[..v].to_vec();
        super::super::gpu::k1::ATTN_TC2.with(|c| c.set(true));
        let diff = l1.iter().zip(&outs[0]).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        println!("INK2_FULL attn_tc2 vs attn_tc (3,000-token prefill): {diff} of {v} logits differ (bitwise)");
        if diff != 0 { fails += 1; }
        for (name, on) in [("attn_tc", false), ("attn_tc2", true)] {
            super::super::gpu::k1::ATTN_TC2.with(|c| c.set(on));
            lanes[0].pos = 0;
            let t = std::time::Instant::now();
            prefill_chunked(&g, &mut ws, &mut lanes, 0, &ids[..2048], 2048)?;
            prefill_chunked(&g, &mut ws, &mut lanes, 0, &ids[..2048], 2048)?;
            g.k.sync()?;
            println!("INK2_FULL timing: prefill 2 x 2048 (second chunk at 2K) with {name}: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
        }
        super::super::gpu::k1::ATTN_TC2.with(|c| c.set(true));
    }
    for i in 1..outs.len() {
        let e = rel_l2(&outs[i], &outs[0]);
        let same = argmax(&outs[i]) == argmax(&outs[0]);
        println!("INK2_FULL path {} vs chunk2048: logits rel-L2 {e:.2e}, argmax {}", i, if same { "same" } else { "DIFFERENT" });
        if !same || e > 5e-2 { fails += 1; }
    }
    // ---- 2b. S4: the tiled prefill GEMM (ink_proj_tile) vs the K1 one (ink_proj_wide), bitwise, + timing at 2,048 rows
    {
        let full_l = (0..g.cfg.layers).find(|&l| !g.cfg.is_sliding[l]).unwrap_or(0);
        let mut rs = 0x9e37u64;
        for (mrows, l) in [(1000usize, 0usize), (2048, full_l), (37, full_l)] {
            let xv: Vec<u16> = (0..3 * mrows * super::D).map(|_| { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; super::super::gpu::k1::f32_to_bf16(((rs % 2001) as f32 - 1000.0) / 700.0) }).collect();
            let xd = g.dev.htod_sync_copy(&xv)?;
            let x = super::super::gpu::dp(&xd);
            let xs = (mrows * super::D * 2) as u64;
            let lw = &g.layers[l];
            let (t1, t2) = if lw.sliding { (1usize << 30, 1usize << 30) } else { (lw.t1, lw.t2) };
            let ldy = lw.proj.n_pad;
            let ya = g.dev.alloc_zeros::<f32>(mrows * ldy)?;
            let yb = g.dev.alloc_zeros::<f32>(mrows * ldy)?;
            g.k.proj(&lw.proj, x, x + xs, x + 2 * xs, t1, t2, super::super::gpu::dp(&ya), ldy, mrows, 0)?;
            g.k.proj_wide_ref(&lw.proj, x, x + xs, x + 2 * xs, t1, t2, super::super::gpu::dp(&yb), ldy, mrows)?;
            let (ha, hb) = (g.dev.dtoh_sync_copy(&ya)?, g.dev.dtoh_sync_copy(&yb)?);
            let diff = ha.iter().zip(&hb).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let mut tst = String::new();
            for st in [2usize, 3, 4] {
                super::super::gpu::k1::PROJ_TILE_ST.with(|c| c.set(st));
                let t = super::super::gpu::time_ms(&g.dev, 2, 9, &mut || g.k.proj(&lw.proj, x, x + xs, x + 2 * xs, t1, t2, super::super::gpu::dp(&ya), ldy, mrows, 0))?;
                let hc = g.dev.dtoh_sync_copy(&ya)?;
                let d2 = hc.iter().zip(&hb).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                if d2 != 0 { fails += 1; }
                tst += &format!(" st{st} {t:.3} ms (diff {d2})");
            }
            super::super::gpu::k1::PROJ_TILE_ST.with(|c| c.set(2));
            println!("INK2_FULL proj_tile stages:{tst}");
            let t_tile = super::super::gpu::time_ms(&g.dev, 2, 9, &mut || g.k.proj(&lw.proj, x, x + xs, x + 2 * xs, t1, t2, super::super::gpu::dp(&ya), ldy, mrows, 0))?;
            let t_wide = super::super::gpu::time_ms(&g.dev, 2, 9, &mut || g.k.proj_wide_ref(&lw.proj, x, x + xs, x + 2 * xs, t1, t2, super::super::gpu::dp(&yb), ldy, mrows))?;
            let fl = 2.0 * mrows as f64 * ldy as f64 * super::D as f64;
            println!("INK2_FULL proj_tile vs proj_wide L{l} M={mrows} N={ldy}: {diff} of {} outputs differ (bitwise); tile {:.3} ms ({:.1} TFLOP/s), wide {:.3} ms ({:.1} TFLOP/s)",
                     ha.len(), t_tile, fl / t_tile / 1e9, t_wide, fl / t_wide / 1e9);
            if diff != 0 { fails += 1; }
        }
    }
    // ---- 3. timing (eager launches, one lane, greedy)
    lanes[0].pos = 0;
    let tp = std::time::Instant::now();
    prefill_chunked(&g, &mut ws, &mut lanes, 0, &ids[..2048], 2048)?;
    g.k.sync()?;
    let pf = tp.elapsed().as_secs_f64();
    let mut tok = argmax(&g.logits_host(&ws, 1)?[..v]) as u32;
    for _ in 0..4 { step1(&g, &mut ws, &mut lanes, 0, tok)?; tok = g.sample(&mut ws, &[0.0], &[0])?[0]; }
    let td = std::time::Instant::now();
    let nd = 64;
    // the decode loop is the profiler capture range (nsys --capture-range=cudaProfilerApi); a no-op otherwise
    unsafe { cudarc::driver::sys::cuProfilerStart(); }
    for _ in 0..nd { step1(&g, &mut ws, &mut lanes, 0, tok)?; tok = g.sample(&mut ws, &[0.0], &[0])?[0]; }
    unsafe { cudarc::driver::sys::cuProfilerStop(); }
    let dt = td.elapsed().as_secs_f64() / nd as f64;
    println!("INK2_FULL timing: prefill 2048 tok {:.0} ms ({:.0} tok/s); decode step at ~2.1K ctx {:.2} ms ({:.1} tok/s), eager launches, greedy",
             pf * 1e3, 2048.0 / pf, dt * 1e3, 1.0 / dt);
    // ---- 4. S4 lane-batched decode (device-side positions): per-lane eager == lanes eager == lanes graph, bitwise,
    //         1 and 2 rows; then graphed decode timing
    if !decode_tc {
        lanes.push(g.new_lane(8192)?);
        g.bind_lanes(&mut ws, &lanes)?;
        lanes[1].pos = 0;
        prefill_chunked(&g, &mut ws, &mut lanes, 1, &ids[500..1700], 2048)?;
        let base = [lanes[0].pos, lanes[1].pos];
        let toks0: Vec<u32> = (0..8).map(|i| ids[2100 + i]).collect();
        let toks1: Vec<u32> = (0..8).map(|i| ids[2200 + i]).collect();
        let mut runs: Vec<Vec<f32>> = Vec::new();
        for mode in 0..3 {
            lanes[0].pos = base[0];
            lanes[1].pos = base[1];
            let mut all = Vec::new();
            for st in 0..8 {
                let rows = if st % 3 == 2 { vec![(0usize, toks0[st])] } else { vec![(0usize, toks0[st]), (1usize, toks1[st])] };
                match mode {
                    0 => {
                        let segs: Vec<Seg> = rows.iter().map(|&(l, _)| Seg { lane: l, p0: lanes[l].pos, m: 1 }).collect();
                        let tk: Vec<u32> = rows.iter().map(|r| r.1).collect();
                        let outr: Vec<usize> = (0..rows.len()).collect();
                        let refs: Vec<&Lane> = lanes.iter().collect();
                        g.forward(&mut ws, &refs, &segs, &tk, &outr)?;
                    }
                    1 => g.decode_lanes(&mut ws, &lanes, &rows, false)?,
                    _ => g.decode_lanes(&mut ws, &lanes, &rows, true)?,
                }
                all.extend(g.logits_host(&ws, rows.len())?);
                for &(l, _) in &rows { lanes[l].pos += 1; }
            }
            runs.push(all);
        }
        let names = ["per-lane eager", "lanes eager", "lanes graph"];
        for i in 1..3 {
            let diff = runs[i].iter().zip(&runs[0]).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            println!("INK2_FULL decode {} vs {}: {} of {} logits differ (bitwise)", names[i], names[0], diff, runs[0].len());
            if diff != 0 { fails += 1; }
        }
        lanes[0].pos = 2048 + 4;
        lanes.truncate(1);
        g.bind_lanes(&mut ws, &lanes)?;
        for gr in [false, true] {
            let mut tok = 1000u32;
            for _ in 0..4 { g.decode_lanes(&mut ws, &lanes, &[(0, tok)], gr)?; lanes[0].pos += 1; tok = g.sample(&mut ws, &[0.0], &[0])?[0]; }
            let td = std::time::Instant::now();
            for _ in 0..nd { g.decode_lanes(&mut ws, &lanes, &[(0, tok)], gr)?; lanes[0].pos += 1; tok = g.sample(&mut ws, &[0.0], &[0])?[0]; }
            let dt = td.elapsed().as_secs_f64() / nd as f64;
            println!("INK2_FULL timing: decode step (lanes, {}) at ~2.1K ctx {:.2} ms ({:.1} tok/s), greedy", if gr { "CUDA graph" } else { "eager" }, dt * 1e3, 1.0 / dt);
        }
        // verify-width costs (eager launches: compare with the eager 1-row step above): one lane x m rows, 4 lanes x m rows
        for nl in [1usize, 4] {
            while lanes.len() < nl {
                let li = lanes.len();
                lanes.push(g.new_lane(8192)?);
                prefill_chunked(&g, &mut ws, &mut lanes, li, &ids[li * 100..li * 100 + 2048], 2048)?;
            }
            for m in [1usize, 2, 3, 5, 9] {
                let base: Vec<usize> = lanes.iter().map(|l| l.pos).collect();
                let segs: Vec<Seg> = (0..nl).map(|l| Seg { lane: l, p0: base[l], m }).collect();
                let tk: Vec<u32> = (0..nl * m).map(|i| ids[2400 + i]).collect();
                let outr: Vec<usize> = (0..(nl * m).min(super::MAX_OUT)).collect();
                let reps = 12;
                let mut tt = Vec::new();
                for _ in 0..reps + 2 {
                    let t = std::time::Instant::now();
                    { let refs: Vec<&Lane> = lanes.iter().collect(); g.forward(&mut ws, &refs, &segs, &tk, &outr)?; }
                    g.k.sync()?;
                    tt.push(t.elapsed().as_secs_f64());
                }
                tt.drain(..2);
                tt.sort_by(|a, b| a.partial_cmp(b).unwrap());
                println!("INK2_FULL timing: verify-shaped forward {nl} lane(s) x {m} rows ({} rows) {:.2} ms (median of {reps}, eager)", nl * m, tt[reps / 2] * 1e3);
            }
        }
    }
    if fails > 0 { bail!("INK2_FULL_FAIL: {fails} comparison(s) failed (worst logits rel-L2 {worst:.2e})"); }
    println!("INK2_FULL_OK (worst logits rel-L2 {worst:.2e}, {:.0}s)", t0.elapsed().as_secs_f64());
    Ok(())
}
