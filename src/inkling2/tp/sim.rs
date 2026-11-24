//! `--probe-inkling2-tp-sim --model-dir <stock pack> [--tp 2|4] [--inkling2-sim-steps N]`: the TP exchange simulation
//! on ONE GPU. W ranks run as W threads, each with its own slice of the model (`GpuModel::load_rank`) and a loopback
//! transport (the partials cross host channels with the link's association: p0 + p1 at W = 2, (p0 + p2) + (p1 + p3)
//! at W = 4), through the same model code the served TP path runs (every reduce, the vocab-parallel sampler, the
//! logits gather, the K / V widths of a rank). One scenario per configuration: two lanes, a 2,600-token prompt
//! (a 2,048 chunk + a 552 chunk, past the 1,024 sliding window) and a 300-token prompt, then N decode steps of both
//! lanes (lane-batched decode, eager) fed TP1's greedy tokens, greedy and sampled draws.
//!
//! Gates:
//!   1. identity: TP1 with d0's NQ = 24 kernel copies == TP1 (bitwise logits, every row);
//!   2. TP-W vs TP1: logits rel-L2 per row, argmax agreement, the GPU sampler's tokens (greedy and T = 1 with fixed
//!      seeds) — the reduce order differs from TP1 (two partial sums per visit), so the bytes are not expected equal.

use super::shard::Rank;
use super::xport::Xport;
use super::TpRank;
use crate::inkling2::model::{GpuModel, Lane, Seg};
use anyhow::{Context, Result};
use cudarc::driver::CudaDevice;
use std::path::Path;
use std::sync::Arc;

#[derive(Default, Clone)]
struct Trace { logits: Vec<Vec<f32>>, greedy: Vec<u32>, sampled: Vec<u32>, ms: f64, vbad: usize, vrows: usize }

const TEXT: &str = "The lighthouse keeper kept a careful log of every ship that passed the headland, noting the weather, the \
 direction of the wind, the colour of the sails and the hour. Over forty years the log grew into eleven volumes. ";

fn prompts(dir: &Path) -> Result<(Vec<u32>, Vec<u32>)> {
    let tok = crate::tokenizer::QwenTokenizer::from_file(&dir.join("tokenizer.json").to_string_lossy()).context("tokenizer")?;
    let mut a = Vec::new();
    let mut i = 0;
    while a.len() < 2600 { a.extend(tok.encode(&format!("{i}. {TEXT}"), false)?); i += 1; }
    a.truncate(2600);
    let mut b = Vec::new();
    while b.len() < 300 { b.extend(tok.encode(&format!("Question {i}: why does the tide rise twice a day? "), false)?); i += 1; }
    b.truncate(300);
    Ok((a, b))
}

/// One configuration's scenario on `g` (feed = TP1's greedy tokens to teacher-force, or None = its own).
fn scenario(g: &GpuModel, a: &[u32], b: &[u32], steps: usize, feed: Option<&[u32]>) -> Result<Trace> {
    let mut ws = g.new_work()?;
    let mut lanes: Vec<Lane> = (0..2).map(|_| g.new_lane(4096)).collect::<Result<_>>()?;
    g.bind_lanes(&mut ws, &lanes)?;
    let t0 = std::time::Instant::now();
    let mut tr = Trace::default();
    for (li, ids) in [(0usize, a), (1, b)] {
        let mut i = 0;
        while i < ids.len() {
            let m = (ids.len() - i).min(2048);
            let last = i + m == ids.len();
            let p0 = lanes[li].pos;
            let outr = [m - 1];
            {
                let refs: Vec<&Lane> = lanes.iter().collect();
                g.forward(&mut ws, &refs, &[Seg { lane: li, p0, m }], &ids[i..i + m], if last { &outr[..] } else { &[][..] })?;
            }
            lanes[li].pos += m;
            i += m;
        }
        tr.logits.push(g.logits_host(&ws, 1)?);
        tr.greedy.push(g.sample(&mut ws, &[0.0], &[7])?[0]);
    }
    for s in 0..steps {
        let fed: Vec<u32> = match feed { Some(f) => vec![f[2 * s], f[2 * s + 1]], None => vec![tr.greedy[2 * s], tr.greedy[2 * s + 1]] };
        g.decode_lanes(&mut ws, &lanes, &[(0, fed[0]), (1, fed[1])], false)?;
        for l in lanes.iter_mut() { l.pos += 1; }
        let lg = g.logits_host(&ws, 2)?;
        let v = g.vocab();
        tr.logits.push(lg[..v].to_vec());
        tr.logits.push(lg[v..2 * v].to_vec());
        tr.sampled.extend(g.sample(&mut ws, &[1.0, 1.0], &[1000 + s as u64, 2000 + s as u64])?);
        tr.greedy.extend(g.sample(&mut ws, &[0.0, 0.0], &[0, 0])?);
    }
    // PV1: verify rows == 1-row decodes at this rank count (the speculative verify path): lane 0 carries a 5-row run,
    // lane 1 a 3-row run, one verify_lanes call; then each row again as its own 1-row decode at its position
    let v = g.vocab();
    let (p0, p1) = (lanes[0].pos, lanes[1].pos);
    let t0r: Vec<u32> = vec![1000, 2345, 777, 9000, 31337];
    let t1r: Vec<u32> = vec![4242, 17, 60000];
    g.verify_lanes(&mut ws, &lanes, &[(0, &t0r[..]), (1, &t1r[..])], false)?;
    let vr = g.logits_host(&ws, 8)?;
    let mut bad = 0;
    for (j, (li, t, p)) in t0r.iter().enumerate().map(|(j, &t)| (0usize, t, p0 + j))
        .chain(t1r.iter().enumerate().map(|(j, &t)| (1usize, t, p1 + j))).enumerate() {
        lanes[li].pos = p;
        g.decode_lanes(&mut ws, &lanes, &[(li, t)], false)?;
        let d = g.logits_host(&ws, 1)?;
        if vr[j * v..(j + 1) * v].iter().zip(&d[..v]).any(|(a, b)| a.to_bits() != b.to_bits()) { bad += 1; }
    }
    tr.vbad = bad;
    tr.vrows = 8;
    g.k.sync()?;
    tr.ms = t0.elapsed().as_secs_f64() * 1e3;
    Ok(tr)
}

/// The cpu.rs f32 oracle on the S1 real cases (`--oracle-dir`): per case the prompt, the oracle's logits rows (the
/// prompt's last row, then `steps` greedy decode steps) and the tokens it fed.
type Oracle = Vec<(Vec<u32>, Vec<Vec<f32>>, Vec<u32>)>;
fn oracle(dir: &Path, rd: &str, steps: usize) -> Result<Oracle> {
    let cases: serde_json::Value = serde_json::from_slice(&std::fs::read(Path::new(rd).join("cases.json"))?)?;
    let m = crate::inkling2::cpu::CpuModel::load(dir)?;
    let mut out = Vec::new();
    for case in cases.as_array().context("cases.json")? {
        let prompt: Vec<u32> = case["steps"][0].as_array().context("steps")?.iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let mut st = m.new_state();
        let f = m.forward(&mut st, &prompt, None)?;
        let mut rows = vec![m.logits(&f[(prompt.len() - 1) * m.cfg.hidden..])];
        let mut toks = Vec::new();
        for _ in 0..steps {
            let t = argmax(rows.last().unwrap()) as u32;
            toks.push(t);
            let f = m.forward(&mut st, &[t], None)?;
            rows.push(m.logits(&f));
        }
        out.push((prompt, rows, toks));
    }
    Ok(out)
}
/// The GPU logits of the oracle's cases (fed the oracle's tokens), one lane, one row per step.
fn oracle_gpu(g: &GpuModel, o: &Oracle) -> Result<Vec<Vec<f32>>> {
    let mut ws = g.new_work()?;
    let mut lanes = vec![g.new_lane(4096)?];
    g.bind_lanes(&mut ws, &lanes)?;
    let v = g.vocab();
    let mut rows = Vec::new();
    for (prompt, _, toks) in o {
        lanes[0].pos = 0;
        let refs: Vec<&Lane> = lanes.iter().collect();
        g.forward(&mut ws, &refs, &[Seg { lane: 0, p0: 0, m: prompt.len() }], prompt, &[prompt.len() - 1])?;
        lanes[0].pos = prompt.len();
        rows.push(g.logits_host(&ws, 1)?[..v].to_vec());
        for &t in toks {
            let refs: Vec<&Lane> = lanes.iter().collect();
            g.forward(&mut ws, &refs, &[Seg { lane: 0, p0: lanes[0].pos, m: 1 }], &[t], &[0])?;
            lanes[0].pos += 1;
            rows.push(g.logits_host(&ws, 1)?[..v].to_vec());
        }
    }
    Ok(rows)
}
fn vs_oracle(rows: &[Vec<f32>], o: &Oracle) -> (f64, f64, usize, usize) {
    let cpu: Vec<&Vec<f32>> = o.iter().flat_map(|c| c.1.iter()).collect();
    let e: Vec<f64> = rows.iter().zip(&cpu).map(|(a, b)| rel_l2(a, b)).collect();
    let am = rows.iter().zip(&cpu).filter(|(a, b)| argmax(a) == argmax(b)).count();
    (e.iter().sum::<f64>() / e.len().max(1) as f64, e.iter().cloned().fold(0f64, f64::max), am, e.len())
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut n, mut d) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) { n += (*x as f64 - *y as f64).powi(2); d += (*y as f64).powi(2); }
    (n / d.max(1e-300)).sqrt()
}
fn argmax(v: &[f32]) -> usize { let mut b = 0; for i in 1..v.len() { if v[i] > v[b] { b = i; } } b }
/// KL(p || q) of softmax(a) vs softmax(b) (nats): the distribution difference a sampled request sees.
fn kl(a: &[f32], b: &[f32]) -> f64 {
    let lse = |v: &[f32]| { let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64; m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln() };
    let (la, lb) = (lse(a), lse(b));
    a.iter().zip(b).map(|(&x, &y)| { let lp = x as f64 - la; lp.exp() * (lp - (y as f64 - lb)) }).sum()
}

pub fn probe(dir: &Path, worlds: &[usize], steps: usize, oracle_dir: Option<&str>) -> Result<bool> {
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    let (a, b) = prompts(dir)?;
    let mut ok = true;
    let orc = match oracle_dir { Some(rd) => Some(oracle(dir, rd, 4)?), None => None };
    // TP1, then TP1 on the d0 copies (NQ = 24)
    let (r1, r24) = {
        let g = GpuModel::load(dir, &dev)?;
        if let Some(o) = &orc {
            let (mean, max, am, n) = vs_oracle(&oracle_gpu(&g, o)?, o);
            println!("INK2_TP_SIM oracle: TP1 vs the cpu.rs f32 oracle ({} cases, prompt + 4 steps): logits rel-L2 mean {mean:.2e} max {max:.2e}, argmax {am}/{n}", o.len());
        }
        let r1 = scenario(&g, &a, &b, steps, None)?;
        super::kern::FORCE_TP24.with(|c| c.set(true));
        let r24 = scenario(&g, &a, &b, steps, Some(&r1.greedy));
        super::kern::FORCE_TP24.with(|c| c.set(false));
        (r1, r24?)
    };
    dev.synchronize()?;
    let same = r1.logits.iter().zip(&r24.logits).all(|(x, y)| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()));
    println!("INK2_TP_SIM identity: d0 NQ=24 copies vs the TP1 kernels: {} logits rows {} (bitwise), greedy {} sampled {}; TP1 scenario {:.0} ms",
             r1.logits.len(), if same { "IDENTICAL" } else { "DIFFER" }, if r1.greedy == r24.greedy { "equal" } else { "DIFFER" },
             if r1.sampled == r24.sampled { "equal" } else { "DIFFER" }, r1.ms);
    ok &= same && r1.greedy == r24.greedy;
    println!("INK2_TP_SIM verify: TP1 verify rows vs 1-row decodes: {} of {} rows differ (bitwise); NQ=24 copies: {} of {}", r1.vbad, r1.vrows, r24.vbad, r24.vrows);
    ok &= r1.vbad == 0 && r24.vbad == 0;
    for &w in worlds {
        let xs = Xport::loopback_group(w);
        let cfg = crate::inkling2::config::Inkling2Config::load(dir)?;
        let feed = r1.greedy.clone();
        let res: Vec<Result<(Trace, Option<Vec<Vec<f32>>>)>> = std::thread::scope(|s| {
            let hs: Vec<_> = xs.into_iter().enumerate().map(|(r, x)| {
                let (dev, a, b, feed, cfg, orc): (Arc<CudaDevice>, &[u32], &[u32], &[u32], _, &Option<Oracle>) = (dev.clone(), &a, &b, &feed, &cfg, &orc);
                s.spawn(move || -> Result<(Trace, Option<Vec<Vec<f32>>>)> {
                    dev.bind_to_thread()?;
                    let rank = Rank::new(r, w, cfg.inter, cfg.unpadded_vocab)?;
                    let mut g = GpuModel::load_rank(dir, &dev, Some(rank))?;
                    g.tp = Some(TpRank::new(&dev, x, None)?);
                    let tr = scenario(&g, a, b, steps, Some(feed))?;
                    Ok((tr, match orc { Some(o) => Some(oracle_gpu(&g, o)?), None => None }))
                })
            }).collect();
            hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("rank thread panicked")))).collect()
        });
        let mut traces = Vec::new();
        let mut orows = None;
        for (r, t) in res.into_iter().enumerate() {
            let (t, o) = t.with_context(|| format!("TP{w} sim rank {r}"))?;
            if r == 0 { orows = o; }
            traces.push(t);
        }
        if let (Some(rows), Some(o)) = (&orows, &orc) {
            let (mean, max, am, n) = vs_oracle(rows, o);
            println!("INK2_TP_SIM oracle: TP{w} vs the cpu.rs f32 oracle: logits rel-L2 mean {mean:.2e} max {max:.2e}, argmax {am}/{n}");
        }
        let t0 = &traces[0];
        let ranks_agree = traces.iter().all(|t| t.greedy == t0.greedy && t.sampled == t0.sampled
                                         && t.logits.iter().zip(&t0.logits).all(|(x, y)| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())));
        let mut worst = 0f64;
        let mut am = 0;
        let (mut kls, mut rls) = (Vec::new(), Vec::new());
        for (x, y) in t0.logits.iter().zip(&r1.logits) {
            worst = worst.max(rel_l2(x, y));
            rls.push(rel_l2(x, y));
            kls.push(kl(y, x));
            if argmax(x) == argmax(y) { am += 1; }
        }
        let slices: Vec<String> = (0..w).map(|r| {
            let (v0, n) = super::shard::vocab_range(cfg.unpadded_vocab, w, r);
            format!("[{v0}..{}) {:.1e}", v0 + n, rel_l2(&t0.logits[0][v0..v0 + n], &r1.logits[0][v0..v0 + n]))
        }).collect();
        println!("INK2_TP_SIM TP{w}: row 0 rel-L2 by vocabulary slice: {}", slices.join(", "));
        let kmax = kls.iter().cloned().fold(0f64, f64::max);
        let kmean = kls.iter().sum::<f64>() / kls.len().max(1) as f64;
        println!("INK2_TP_SIM TP{w}: per row rel-L2 {}", rls.iter().map(|x| format!("{x:.1e}")).collect::<Vec<_>>().join(" "));
        println!("INK2_TP_SIM TP{w}: per row KL(TP1 || TP{w}) nats {}  (mean {kmean:.2e}, max {kmax:.2e})", kls.iter().map(|x| format!("{x:.1e}")).collect::<Vec<_>>().join(" "));
        let g_eq = t0.greedy.iter().zip(&r1.greedy).filter(|(x, y)| x == y).count();
        let s_eq = t0.sampled.iter().zip(&r1.sampled).filter(|(x, y)| x == y).count();
        println!("INK2_TP_SIM TP{w}: ranks bitwise in agreement: {}; logits vs TP1: worst rel-L2 {worst:.2e}, argmax equal {am}/{}; \
                  GPU greedy tokens equal {g_eq}/{}, sampled (T=1, same seeds) equal {s_eq}/{}; rank-0 scenario {:.0} ms (loopback exchanges, host round trips)",
                 if ranks_agree { "YES" } else { "NO" }, t0.logits.len(), t0.greedy.len(), t0.sampled.len(), t0.ms);
        // the gate: every rank identical, argmax equal on >= all but one row, KL(TP1 || TPW) <= 2e-2 nats on every row
        println!("INK2_TP_SIM TP{w}: verify rows vs 1-row decodes (rank 0): {} of {} rows differ (bitwise)", t0.vbad, t0.vrows);
        let pass = ranks_agree && am + 1 >= t0.logits.len() && kmax <= 2e-2 && t0.vbad == 0;
        println!("INK2_TP_SIM TP{w}: {}", if pass { "PASS" } else { "FAIL" });
        ok &= pass;
    }
    Ok(ok)
}
