//! `--probe-inkling2-mtp --model-dir <stock pack> [--inkling2-mtp-tokens N] [--inkling2-mtp-oracle]
//!  [--inkling2-mtp-dump DIR]`: the S6 gates on one box.
//!   1. drafter vs the f32 CPU oracle (`mtp/cpu.rs`) on the real target taps of a real prompt (rel-L2 of the final
//!      hidden, base and step logits; argmax agreement of every step); `--inkling2-mtp-dump DIR` also writes the taps,
//!      anchor, drafts and the GPU step logits for the external-reference script (scripts/inkling2/mtp_ref.py);
//!   2. greedy speculative == plain greedy, token for token, 3 prompts x N tokens (gate: identical);
//!   3. acceptance per draft position and speed: plain vs speculative, greedy and sampled (temperature 1.0, the
//!      pack's default), one lane.

use super::cpu::MtpCpu;
use super::policy::Policy;
use super::spec::Spec;
use super::super::chat::request::{render, Defaults};
use super::super::model::{GpuModel, Lane, Work, D};
use super::super::pack::{Inkling2Pack, P};
use crate::tokenizer::QwenTokenizer;
use anyhow::{bail, Context, Result};
use cudarc::driver::CudaDevice;
use std::path::Path;

const PROMPTS: [(&str, &str); 3] = [
    ("prose", "Write a warm, vivid short story (about 600 words) about a lighthouse keeper who finds a message in a bottle."),
    ("code", "Write a complete Python module implementing an LRU cache class with get, put, delete and resize methods, full type hints, docstrings and a set of unittest test cases."),
    ("reason", "A train leaves city A at 9:40 travelling at 84 km/h towards city B, 377 km away. Another leaves B at 10:15 at 96 km/h towards A. When and where do they meet? Explain step by step."),
];

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut n, mut d) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) { n += (*x as f64 - *y as f64).powi(2); d += (*y as f64).powi(2); }
    (n / d.max(1e-300)).sqrt()
}
fn argmax(v: &[f32]) -> usize { let mut b = 0; for i in 1..v.len() { if v[i] > v[b] { b = i; } } b }

fn plain_step(g: &GpuModel, ws: &mut Work, lanes: &mut [Lane], x: u32) -> Result<()> {
    let p0 = lanes[0].pos;
    // the served plain decode entry (PV1: lanes path, device positions)
    let _ = p0;
    g.decode_lanes(ws, lanes, &[(0, x)], false)?;
    lanes[0].pos += 1;
    Ok(())
}

pub fn probe(dir: &Path, n_tok: usize, oracle: bool, dump: Option<&str>, shift: i64) -> Result<()> {
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    let g = GpuModel::load(dir, &dev)?;
    let pack = Inkling2Pack::open(dir)?;
    let mut ws = g.new_work()?;
    let mut lanes = vec![g.new_lane(8192)?];
    g.bind_lanes(&mut ws, &lanes)?;
    let mut sp = Spec::new(&g, &pack, &mut ws, 1, 8192)?;
    sp.dev.pos_shift = shift;
    let tok = QwenTokenizer::from_file(&dir.join("tokenizer.json").to_string_lossy())?;
    let defaults = Defaults { effort: None, max_seq_len: 8192 };
    let mut fails = 0;
    let mut prompts = Vec::new();
    for (name, text) in PROMPTS {
        let body = serde_json::json!({"messages": [{"role": "user", "content": text}]});
        let r = render(&body, &defaults, &tok).map_err(|e| anyhow::anyhow!("render: {}", e.message))?;
        prompts.push((name, r.prompt_ids));
    }
    // ---------------------------------------------------------------- 1. oracle
    if oracle || dump.is_some() {
        let (_, ids) = &prompts[0];
        lanes[0].pos = 0;
        sp.prefill(&g, &mut ws, &mut lanes, 0, ids)?;
        let x = g.sample(&mut ws, &[0.0], &[0])?[0];
        let p = lanes[0].pos;
        // the taps of the prompt (one chunk)
        let taps_b = g.dev.dtoh_sync_copy(ws.taps.as_ref().unwrap())?;
        let taps: Vec<f32> = taps_b[..p * 4 * D].iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
        let d = sp.draft(&g, &lanes, 0, x, 0.0, 1)?;
        let (gfin, gbase, gstep) = sp.dev.dump(&g, &sp.w[0])?;
        let n = sp.dev.n;
        println!("INK2_MTP oracle prompt {} tokens, anchor {x}, drafts {:?}, conf {:?}", p, d.toks, d.conf.iter().map(|c| (c * 1000.0).round() / 1000.0).collect::<Vec<_>>());
        if let Some(dd) = dump {
            let dd = Path::new(dd);
            std::fs::create_dir_all(dd)?;
            let w16 = |f: &str, v: &[u16]| std::fs::write(dd.join(f), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
            let w32 = |f: &str, v: &[f32]| std::fs::write(dd.join(f), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
            w16("taps.bf16", &taps_b[..p * 4 * D])?;
            w32("gpu_fin.f32", &gfin)?;
            w32("gpu_step.f32", &gstep[..7 * n])?;
            w32("gpu_base.f32", &gbase[..7 * n])?;
            std::fs::write(dd.join("meta.json"), serde_json::to_vec_pretty(&serde_json::json!({
                "p": p, "anchor": x, "drafts": d.toks, "conf": d.conf, "n": n, "prompt": ids }))?)?;
            println!("INK2_MTP dump written to {}", dd.display());
        }
        if oracle {
            let t0 = std::time::Instant::now();
            let cpu = MtpCpu::load(&pack)?;
            let unembed = pack.st.plain_f32_range(&format!("{P}unembed.weight"), 0, n * D)?;
            let emb = pack.st.plain_f32_range(&format!("{P}embed.weight"), x as usize * D, (x as usize + 1) * D)?;
            let ctx = cpu.context(&taps, p);
            let mut prev = vec![x];
            prev.extend_from_slice(&d.toks);
            let o = cpu.block(&ctx, &emb, &prev, &unembed);
            println!("INK2_MTP oracle ran in {:.1}s", t0.elapsed().as_secs_f64());
            let rf = rel_l2(&gfin, &o.fin);
            let rb = rel_l2(&gbase[..7 * n], &o.base);
            let rs = rel_l2(&gstep[..7 * n], &o.step);
            let mut agree = 0;
            for j in 0..7 {
                let (ga, oa) = (argmax(&gstep[j * n..(j + 1) * n]), argmax(&o.step[j * n..(j + 1) * n]));
                println!("  step {j}: gpu argmax {ga} (draft {}), oracle argmax {oa}, step rel-L2 {:.2e}", d.toks[j],
                         rel_l2(&gstep[j * n..(j + 1) * n], &o.step[j * n..(j + 1) * n]));
                if ga == oa { agree += 1; }
            }
            println!("INK2_MTP oracle: final hidden rel-L2 {rf:.2e}, base logits {rb:.2e}, step logits {rs:.2e}, argmax agree {agree}/7");
            if rs > 3e-2 || agree < 6 { fails += 1; println!("INK2_MTP oracle FAIL"); }
        }
    }
    // ---------------------------------------------------------------- 0. teacher-forced row accuracy (alignment check)
    if std::env::args().any(|a| a == "--inkling2-mtp-align") {
        for (name, ids) in prompts.iter().take(2) {
            lanes[0].pos = 0;
            sp.prefill(&g, &mut ws, &mut lanes, 0, ids)?;
            let mut x = g.sample(&mut ws, &[0.0], &[0])?[0];
            let mut seq = ids.clone();
            for _ in 0..300 { seq.push(x); plain_step(&g, &mut ws, &mut lanes, x)?; x = g.sample(&mut ws, &[0.0], &[0])?[0]; }
            lanes[0].pos = 0;
            sp.prefill(&g, &mut ws, &mut lanes, 0, &seq)?;
            let n = sp.dev.n;
            let taps = super::super::gpu::dp(ws.taps.as_ref().unwrap());
            if let Some(dd) = dump {
                // every visit's entering residual (+ the final-norm output) of the whole sequence, for tap-convention checks
                let tl = seq.len();
                let mut all = vec![0u16; 49 * tl * D];
                let keep = ws.tap_visits;
                for v0 in (0..48).step_by(3) {
                    ws.tap_visits = [v0, v0 + 1, v0 + 2, 48];
                    lanes[0].pos = 0;
                    sp.prefill(&g, &mut ws, &mut lanes, 0, &seq)?;
                    let tb = g.dev.dtoh_sync_copy(ws.taps.as_ref().unwrap())?;
                    for t in 0..tl { for sl in 0..4 {
                        let v = if sl == 3 { 48 } else { v0 + sl };
                        all[(v * tl + t) * D..(v * tl + t + 1) * D].copy_from_slice(&tb[(t * 4 + sl) * D..(t * 4 + sl + 1) * D]);
                    } }
                }
                ws.tap_visits = keep;
                let dd = Path::new(dd).join(*name);
                std::fs::create_dir_all(&dd)?;
                std::fs::write(dd.join("visits.bf16"), all.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                std::fs::write(dd.join("seq.json"), serde_json::to_vec(&serde_json::json!({"seq": seq, "prompt_len": ids.len()}))?)?;
                lanes[0].pos = 0;
                sp.prefill(&g, &mut ws, &mut lanes, 0, &seq)?;
            }
            for (shift, lead) in [(0i64, 0usize), (0, 1), (-1, 1), (1, 1)] {
                sp.dev.pos_shift = shift;
                // lead 1: context position t carries the taps of row t + 1 (diagnostic of the feature convention)
                sp.dev.commit_ctx(g.k, &mut sp.w[0], &sp.lanes[0], taps + (lead * 4 * D * 2) as u64, seq.len() - lead, 0)?;
                let (mut hit_s, mut hit_b, mut cnt) = ([0usize; 7], [0usize; 7], 0usize);
                let mut pp = ids.len() + 5;
                while pp + 8 < seq.len() {
                    sp.dev.set_frnn(g.k, &sp.lanes[0], taps + ((pp - 1 + lead) * 4 * D * 2) as u64)?;
                    lanes[0].pos = pp;
                    let d = sp.draft(&g, &lanes, 0, seq[pp], 0.0, 1)?;
                    let (fin, b, stp) = sp.dev.dump(&g, &sp.w[0])?;
                    if let (Some(dd), 0, 0) = (dump, cnt, lead) {
                        if shift == 0 {
                            let dd = Path::new(dd).join(format!("{name}_blk"));
                            std::fs::create_dir_all(&dd)?;
                            let tb = g.dev.dtoh_sync_copy(ws.taps.as_ref().unwrap())?;
                            std::fs::write(dd.join("taps.bf16"), tb[..pp * 4 * D].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                            std::fs::write(dd.join("gpu_fin.f32"), fin.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                            std::fs::write(dd.join("gpu_base.f32"), b[..7 * n].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                            std::fs::write(dd.join("gpu_step.f32"), stp[..7 * n].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                            std::fs::write(dd.join("meta.json"), serde_json::to_vec(&serde_json::json!({"p": pp, "anchor": seq[pp], "drafts": d.toks, "n": n}))?)?;
                        }
                    }
                    for j in 0..7 {
                        if d.toks[j] == seq[pp + 1 + j] { hit_s[j] += 1; }
                        if argmax(&b[j * n..(j + 1) * n]) as u32 == seq[pp + 1 + j] { hit_b[j] += 1; }
                    }
                    cnt += 1;
                    pp += 9;
                }
                println!("INK2_MTP align {name} shift {shift} lead {lead}: step top-1 by row {:?}, base top-1 {:?} ({cnt} blocks)",
                         hit_s.iter().map(|&h| (h as f64 / cnt as f64 * 100.0).round() / 100.0).collect::<Vec<_>>(),
                         hit_b.iter().map(|&h| (h as f64 / cnt as f64 * 100.0).round() / 100.0).collect::<Vec<_>>());
            }
            sp.dev.pos_shift = shift;
        }
        return Ok(());
    }
    // ---------------------------------------------------------------- 1b. verify rows == decode rows, bitwise
    {
        let (_, ids) = &prompts[1];
        let n = g.vocab();
        lanes[0].pos = 0;
        sp.prefill(&g, &mut ws, &mut lanes, 0, ids)?;
        let p0 = lanes[0].pos;
        let mut x = g.sample(&mut ws, &[0.0], &[0])?[0];
        let mut toks = vec![x];
        let mut rows = Vec::new();
        for _ in 0..8 {
            plain_step(&g, &mut ws, &mut lanes, x)?;
            rows.push(g.logits_host(&ws, 1)?);
            x = g.sample(&mut ws, &[0.0], &[0])?[0];
            toks.push(x);
        }
        for m in [2usize, 5, 8] {
            lanes[0].pos = p0;
            let outr: Vec<usize> = (0..m).collect();
            let _ = &outr;
            g.verify_lanes(&mut ws, &lanes, &[(0, &toks[..m])], false)?;
            let l = g.logits_host(&ws, m)?;
            let mut bad = Vec::new();
            for r in 0..m {
                let d = l[r * n..(r + 1) * n].iter().zip(&rows[r]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                if d != 0.0 { bad.push((r, d)); }
            }
            println!("INK2_MTP binv: verify of {m} rows vs {m} decode steps: {} (mismatching rows {bad:?})", if bad.is_empty() { "bit-identical" } else { "DIFFERENT" });
            if !bad.is_empty() { fails += 1; }
        }
    }
    // ---------------------------------------------------------------- 1b'. row 0 of a verify with unrelated co-rows == decode
    {
        let (_, ids) = &prompts[0];
        let n = g.vocab();
        lanes[0].pos = 0;
        sp.prefill(&g, &mut ws, &mut lanes, 0, ids)?;
        let mut x = g.sample(&mut ws, &[0.0], &[0])?[0];
        let mut bad = 0;
        let mut rng = 12345u64;
        for step in 0..80 {
            let p0 = lanes[0].pos;
            for m in [3usize, 8] {
                let mut tk = vec![x];
                for _ in 1..m { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); tk.push(((rng >> 33) % 150000) as u32 + 1000); }
                let outr: Vec<usize> = (0..m).collect();
                let _ = &outr;
                g.verify_lanes(&mut ws, &lanes, &[(0, &tk[..])], false)?;
                let l = g.logits_host(&ws, 1)?;
                lanes[0].pos = p0;
                plain_step(&g, &mut ws, &mut lanes, x)?;
                let r = g.logits_host(&ws, 1)?;
                lanes[0].pos = p0;
                let d = l.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                if d != 0.0 {
                    bad += 1;
                    if bad < 5 { println!("INK2_MTP binv2 step {step} m {m}: row 0 differs by {d:.3e} (argmax {} vs {})", argmax(&l[..n]), argmax(&r[..n])); }
                    if bad == 1 {
                        // locate the first visit whose entering residual differs (row 0, m-row call vs 1-row call)
                        let keep = ws.tap_visits;
                        let mut first = None;
                        for v0 in (0..48).step_by(3) {
                            ws.tap_visits = [v0, v0 + 1, v0 + 2, 48];
                            let mut got = Vec::new();
                            for mm in [m, 1] {
                                g.verify_lanes(&mut ws, &lanes, &[(0, &tk[..mm])], false)?;
                                lanes[0].pos = p0;
                                let tb = g.dev.dtoh_sync_copy(ws.taps.as_ref().unwrap())?;
                                got.push(tb[..4 * D].to_vec());
                            }
                            for sl in 0..4 {
                                let dd = got[0][sl * D..(sl + 1) * D].iter().zip(&got[1][sl * D..(sl + 1) * D]).filter(|(a, b)| a != b).count();
                                if dd > 0 && first.is_none() { first = Some((if sl == 3 { 48 } else { v0 + sl }, dd)); }
                            }
                            if first.is_some() { break; }
                        }
                        ws.tap_visits = keep;
                        println!("INK2_MTP binv2: first differing visit entering state (row 0): {first:?}");
                    }
                }
            }
            plain_step(&g, &mut ws, &mut lanes, x)?;
            x = g.sample(&mut ws, &[0.0], &[0])?[0];
        }
        println!("INK2_MTP binv2: row 0 of verifies with random co-rows vs decode over 80 positions: {bad} mismatches");
        if bad > 0 { fails += 1; }
    }
    // ---------------------------------------------------------------- 1c. drafter state after rounds == after a fresh prefill
    {
        let (_, ids) = &prompts[0];
        let n = sp.dev.n;
        lanes[0].pos = 0;
        sp.prefill(&g, &mut ws, &mut lanes, 0, ids)?;
        let mut x = g.sample(&mut ws, &[0.0], &[0])?[0];
        let mut seq = ids.clone();
        let mut pol = Policy::new(31.0, 2.5, 9.0, 7);
        for r in 0..12 {
            seq.push(x);
            let rd = sp.round(&g, &mut ws, &mut lanes, 0, x, 0.0, 77 + r, &mut pol, 7)?;
            seq.extend_from_slice(&rd.toks[..rd.toks.len() - 1]);
            x = *rd.toks.last().unwrap();
        }
        let da = sp.draft(&g, &lanes, 0, x, 0.0, 5)?;
        let (_, ba, sa) = sp.dev.dump(&g, &sp.w[0])?;
        println!("INK2_MTP anchor {x}: base argmax {:?} step argmax {:?}", (0..7).map(|j| argmax(&ba[j * n..(j + 1) * n])).collect::<Vec<_>>(),
                 (0..7).map(|j| argmax(&sa[j * n..(j + 1) * n])).collect::<Vec<_>>());
        let pa = lanes[0].pos;
        lanes[0].pos = 0;
        sp.prefill(&g, &mut ws, &mut lanes, 0, &seq)?;
        let db = sp.draft(&g, &lanes, 0, x, 0.0, 5)?;
        let (_, _, sb) = sp.dev.dump(&g, &sp.w[0])?;
        println!("INK2_MTP drafter after 12 rounds (pos {pa}) vs fresh prefill (pos {}): step rel-L2 {:.2e}, drafts {:?} vs {:?}",
                 lanes[0].pos, rel_l2(&sa[..7 * n], &sb[..7 * n]), da.toks, db.toks);
    }
    // ---------------------------------------------------------------- 2./3. greedy identity, acceptance, speed
    let mut summary = Vec::new();
    for (name, ids) in &prompts {
        // plain greedy
        lanes[0].pos = 0;
        sp.prefill(&g, &mut ws, &mut lanes, 0, ids)?;
        let mut x = g.sample(&mut ws, &[0.0], &[0])?[0];
        let mut plain = vec![x];
        g.k.sync()?;
        let t0 = std::time::Instant::now();
        while plain.len() < n_tok {
            plain_step(&g, &mut ws, &mut lanes, x)?;
            x = g.sample(&mut ws, &[0.0], &[0])?[0];
            plain.push(x);
        }
        if *name == "prose" { println!("  plain prose: {:?}", &plain[..40.min(plain.len())]); }
        let plain_ms = t0.elapsed().as_secs_f64() * 1e3 / (n_tok - 1) as f64;
        // speculative greedy and sampled
        for (mode, t) in [("greedy", 0.0f32), ("sampled", 1.0f32)] {
            let mut pol = Policy::new(plain_ms, 2.5, 9.0, 7);
            lanes[0].pos = 0;
            sp.prefill(&g, &mut ws, &mut lanes, 0, ids)?;
            let mut x = g.sample(&mut ws, &[t], &[7])?[0];
            let mut out = vec![x];
            let (mut rounds, mut acc_pos, mut tried_pos) = (0usize, [0usize; 8], [0usize; 8]);
            let (mut cs, mut as_) = (0usize, 0usize);
            g.k.sync()?;
            let t0 = std::time::Instant::now();
            let mut seed = 1000u64;
            while out.len() < n_tok {
                seed += 1;
                let r = sp.round(&g, &mut ws, &mut lanes, 0, x, t, seed, &mut pol, 7)?;
                if rounds < 12 && *name == "prose" { println!("  round {rounds}: c {} a {} drafts {:?} emitted {:?} conf {:?}", r.c, r.a, r.drafts, r.toks, r.conf.iter().map(|c| (c * 100.0).round() / 100.0).collect::<Vec<_>>()); }
                rounds += 1;
                cs += r.c;
                as_ += r.a;
                for j in 0..r.c { tried_pos[j] += 1; if j < r.a { acc_pos[j] += 1; } }
                out.extend_from_slice(&r.toks);
                x = *r.toks.last().unwrap();
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let ntoks = out.len() - 1;
            let tps = ntoks as f64 / ms * 1e3;
            let per: Vec<String> = (0..7).map(|j| if tried_pos[j] > 0 { format!("{:.2}", acc_pos[j] as f64 / tried_pos[j] as f64) } else { "-".into() }).collect();
            let mut line = format!("INK2_MTP {name} {mode}: {ntoks} tokens in {rounds} rounds, {:.2} tokens/round, verified {:.2} accepted {:.2} per round, \
                                    accept by position {per:?}, {tps:.1} tok/s (plain {:.1} tok/s, {plain_ms:.2} ms/step) x{:.2}; draft {:.2} ms",
                                   ntoks as f64 / rounds as f64, cs as f64 / rounds as f64, as_ as f64 / rounds as f64, 1e3 / plain_ms, tps * plain_ms / 1e3, pol.draft_ms);
            if t == 0.0 {
                let k = out.len().min(plain.len()).min(n_tok);
                let first_diff = (0..k).find(|&i| out[i] != plain[i]);
                match first_diff {
                    None => line.push_str(&format!("; greedy == plain on all {k} tokens")),
                    Some(i) => { fails += 1; line.push_str(&format!("; GREEDY MISMATCH at token {i}: spec {} vs plain {}", out[i], plain[i])); }
                }
            }
            println!("{line}");
            summary.push(line);
        }
    }
    if fails > 0 { bail!("{fails} S6 gate failure(s)"); }
    println!("INK2_MTP_OK");
    Ok(())
}
