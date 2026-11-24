//! `--probe-inkling2-vision --model-dir <pack> --oracle-dir <ref>`: the vision path against transformers.
//!
//! `<ref>` comes from `tools/inkling2/ref_vision.py tower` (+ `ids`, `text`): per image the processor's pixel_values and
//! the f32 tower's taps (embed, ln_pre, 23 layers, merger), plus the same with the model in bf16 (the reference's own
//! bf16 noise floor). Checks, each printed as one `INK2_VIS` line:
//!   1. preprocessing: our pixels vs the processor's (grid equal; max |diff|; PNG must be exact);
//!   2. CPU oracle (f32) vs transformers f32, every block, on the reference's pixels: rel-L2 <= 1e-3 (`--inkling2-vision-cpu-max-tokens`
//!      bounds the image size the CPU runs, default 400 tokens);
//!   3. GPU tower vs transformers f32, every block, and vs the CPU oracle where it ran; the bf16 reference's distance is
//!      printed beside it (the GPU keeps an f32 residual stream and must stay well under that floor);
//!   4. GPU tower time per image (CUDA events, median of 5, no taps).

use super::cpu;
use super::gpu::{GpuTaps, VisionGpu};
use super::preproc::{self, PreImage};
use super::{Grid, VisionHost};
use crate::inkling2::pack::open_single;
use anyhow::{bail, Context, Result};
use std::path::Path;

pub fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) { num += (*x as f64 - *y as f64).powi(2); den += (*y as f64).powi(2); }
    if a.len() != b.len() { return f64::INFINITY; }
    (num / den.max(1e-300)).sqrt()
}

pub fn probe_vision(dir: &Path, ref_dir: &str, cpu_max_tokens: usize, gpu: bool, blas: bool) -> Result<()> {
    let t0 = std::time::Instant::now();
    let host = VisionHost::load(dir)?;
    println!("INK2_VIS load: {} layers, hidden {}, merger {} -> {} ({:.1} s)", host.cfg.layers, host.cfg.hidden, host.cfg.proj_hidden,
             host.cfg.text_hidden, t0.elapsed().as_secs_f64());
    let cases: serde_json::Value = serde_json::from_slice(&std::fs::read(Path::new(ref_dir).join("tower_cases.json")).context("tower_cases.json")?)?;
    let mut vg = if gpu {
        let dev = cudarc::driver::CudaDevice::new(0).map_err(|e| anyhow::anyhow!("CudaDevice::new(0): {e:?}"))?;
        crate::inkling2::gpu::load_module(&dev)?;
        let v = VisionGpu::new(&dev, &host, blas)?;
        println!("INK2_VIS gpu: tower weights {:.1} MB on the device; GEMMs {}", v.weight_bytes as f64 / 1e6, if blas { "cuBLAS" } else { "K1 ink_proj_wide" });
        Some((dev, v))
    } else { None };
    let mut fails: Vec<String> = Vec::new();
    for c in cases.as_array().context("cases")? {
        let name = c["name"].as_str().unwrap_or("?");
        let path = c["path"].as_str().unwrap_or("");
        let gref: Vec<usize> = c["grid"].as_array().context("grid")?.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
        let r = open_single(&Path::new(ref_dir).join(format!("{name}.safetensors")))?;
        let rpx = r.plain_f32("pixel_values")?;
        // 1. preprocessing
        let bytes = std::fs::read(path).with_context(|| format!("read {path}"))?;
        let ours = preproc::preprocess(&bytes, 40, usize::MAX)?;
        let grid_ok = ours.grid.rows == gref[1] && ours.grid.cols == gref[2];
        let (mut mx, mut nd) = (0f32, 0usize);
        if grid_ok {
            for (a, b) in ours.pixels.iter().zip(&rpx) { let d = (a - b).abs(); if d > 0.0 { nd += 1; } mx = mx.max(d); }
        }
        let png = path.to_ascii_lowercase().ends_with(".png");
        println!("INK2_VIS {name} preproc: {}x{} alpha={} grid {}x{} (ref {}x{}) pixels differing {nd}/{} max|d| {mx:.3e} ({})",
                 ours.src.0, ours.src.1, ours.had_alpha, ours.grid.rows, ours.grid.cols, gref[1], gref[2], rpx.len(),
                 if png { "png: must be exact" } else { "jpeg: decoder rounding allowed" });
        if !grid_ok || (png && nd > 0) { fails.push(format!("{name} preproc")); }
        // the tower gates run on the reference's own pixels
        let g = Grid { rows: gref[1], cols: gref[2] };
        let pre = PreImage { grid: g, pixels: rpx.clone(), hash: 0, src: ours.src, resized: None, had_alpha: ours.had_alpha };
        let blocks: Vec<String> = ["embed", "ln_pre"].iter().map(|s| s.to_string())
            .chain((0..host.cfg.layers).map(|i| format!("layer{i}"))).chain(std::iter::once("merger".to_string())).collect();
        let rf: Vec<Vec<f32>> = blocks.iter().map(|b| r.plain_f32(&format!("f32.{b}"))).collect::<Result<_>>()?;
        let rb: Vec<Vec<f32>> = blocks.iter().map(|b| r.plain_f32(&format!("bf16.{b}"))).collect::<Result<_>>()?;
        let floor: Vec<f64> = rb.iter().zip(&rf).map(|(a, b)| rel_l2(a, b)).collect();
        // 2. CPU oracle
        let mut cpu_t: Option<Vec<Vec<f32>>> = None;
        if g.tokens() <= cpu_max_tokens {
            let t = std::time::Instant::now();
            let tp = cpu::forward(&host, &pre, true);
            let secs = t.elapsed().as_secs_f64();
            let mut v = vec![tp.embed, tp.ln_pre];
            v.extend(tp.layers);
            v.push(tp.merger);
            let errs: Vec<f64> = v.iter().zip(&rf).map(|(a, b)| rel_l2(a, b)).collect();
            let worst = errs.iter().cloned().fold(0.0, f64::max);
            println!("INK2_VIS {name} cpu-vs-ref(f32): embed {:.2e} ln_pre {:.2e} layer0 {:.2e} layer11 {:.2e} layer22 {:.2e} merger {:.2e} worst {:.2e} ({:.1} s) {}",
                     errs[0], errs[1], errs[2], errs[13], errs[24], errs[25], worst, secs, if worst <= 1e-3 { "PASS" } else { "FAIL" });
            if worst > 1e-3 { fails.push(format!("{name} cpu")); }
            cpu_t = Some(v);
        } else {
            println!("INK2_VIS {name} cpu: skipped ({} tokens > --inkling2-vision-cpu-max-tokens {cpu_max_tokens})", g.tokens());
        }
        // 3. GPU
        if let Some((dev, vgpu)) = vg.as_mut() {
            let t = g.tokens();
            let out = dev.alloc_zeros::<f32>(t * host.cfg.text_hidden)?;
            let mut taps = GpuTaps::default();
            vgpu.forward(&pre, crate::inkling2::gpu::dp(&out), Some(&mut taps))?;
            let merger = dev.dtoh_sync_copy(&out)?;
            let mut v = vec![taps.embed, taps.ln_pre];
            v.extend(taps.layers);
            v.push(merger);
            let errs: Vec<f64> = v.iter().zip(&rf).map(|(a, b)| rel_l2(a, b)).collect();
            let worst_ratio = errs.iter().zip(&floor).skip(2).map(|(e, f)| e / f).fold(0.0, f64::max);
            println!("INK2_VIS {name} gpu-vs-ref(f32): embed {:.2e} ln_pre {:.2e} layer0 {:.2e} layer11 {:.2e} layer22 {:.2e} merger {:.2e} | bf16-ref floor: layer0 {:.2e} layer22 {:.2e} merger {:.2e} | worst gpu/floor {:.3}",
                     errs[0], errs[1], errs[2], errs[13], errs[24], errs[25], floor[2], floor[24], floor[25], worst_ratio);
            let per: Vec<String> = errs.iter().map(|e| format!("{e:.1e}")).collect();
            println!("INK2_VIS {name} gpu per-block rel-L2: [{}]", per.join(" "));
            if let Some(cv) = &cpu_t {
                let e2: Vec<f64> = v.iter().zip(cv).map(|(a, b)| rel_l2(a, b)).collect();
                println!("INK2_VIS {name} gpu-vs-cpu: layer0 {:.2e} layer22 {:.2e} merger {:.2e}", e2[2], e2[24], e2[25]);
            }
            if worst_ratio > 0.5 || !errs.iter().all(|e| e.is_finite()) { fails.push(format!("{name} gpu (worst gpu/floor {worst_ratio:.2})")); }
            if nd > 0 && grid_ok {
                // the served path decodes the image itself: how far do its pixels move the features?
                vgpu.forward(&ours, crate::inkling2::gpu::dp(&out), None)?;
                let m2 = dev.dtoh_sync_copy(&out)?;
                println!("INK2_VIS {name} gpu on OUR decoded pixels: merger rel-L2 vs ref(f32) {:.2e} (on the reference's pixels {:.2e})",
                         rel_l2(&m2, &rf[rf.len() - 1]), errs[errs.len() - 1]);
            }
            // 4. timing
            let k = crate::inkling2::gpu::Ink2Kernels::get()?;
            let _ = k;
            let ms = crate::inkling2::gpu::time_ms(&*dev, 2, 5, &mut || vgpu.forward(&pre, crate::inkling2::gpu::dp(&out), None))?;
            println!("INK2_VIS {name} gpu tower time: {ms:.2} ms for {t} tokens ({} patches; {:.1} ms per 1000 tokens)", g.patches(), ms * 1000.0 / t as f64);
        }
    }
    if fails.is_empty() { println!("INK2_VIS_OK"); Ok(()) } else { bail!("INK2_VIS_FAIL: {}", fails.join(", ")) }
}

fn argmax(v: &[f32]) -> usize {
    let mut b = 0;
    for i in 1..v.len() { if v[i] > v[b] { b = i; } }
    b
}

/// `--probe-inkling2-vision --inkling2-vision-text`: image prompts through the TEXT model vs the transformers dumps of
/// `ref_vision.py text` (the processor's ids, the reference's f32 features spliced at the placeholders):
///   * the spliced input rows (ours vs `step0.inputs`): the splice position and that the features are NOT normed;
///   * CPU oracle (cpu.rs, f32) from those inputs: first-visit hidden `v0` and the last-position logits;
///   * GPU (the served path: tower on the reference pixels, `Work::img` splice in the chunked prefill): last-position
///     logits and argmax, then greedy steps fed the reference's own tokens (argmax per step, logits rel-L2).
pub fn probe_text(dir: &Path, ref_dir: &str, cpu: bool, gpu: bool) -> Result<()> {
    use crate::inkling2::vision::IMAGE_TOKEN;
    let cases: serde_json::Value = serde_json::from_slice(&std::fs::read(Path::new(ref_dir).join("ids_cases.json")).context("ids_cases.json")?)?;
    let cm = if cpu { Some(crate::inkling2::cpu::CpuModel::load(dir)?) } else { None };
    let mut gm = if gpu {
        let dev = cudarc::driver::CudaDevice::new(0).map_err(|e| anyhow::anyhow!("CudaDevice::new(0): {e:?}"))?;
        let g = crate::inkling2::model::GpuModel::load(dir, &dev)?;
        let ws = g.new_work()?;
        let lanes = vec![g.new_lane(8192)?];
        let host = VisionHost::load(dir)?;
        let v = VisionGpu::new(&dev, &host, true)?;
        Some((dev, g, ws, lanes, v))
    } else { None };
    let mut fails = Vec::new();
    for c in cases.as_array().context("cases")? {
        let name = c["name"].as_str().unwrap_or("?");
        let Some(greedy) = c.get("greedy").and_then(|g| g.as_array()) else { continue };
        let greedy: Vec<u32> = greedy.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect();
        let ids: Vec<u32> = c["ids"].as_array().context("ids")?.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect();
        let rt = open_single(&Path::new(ref_dir).join(format!("{name}.text.safetensors")))?;
        let rv = open_single(&Path::new(ref_dir).join(format!("{name}.safetensors")))?;
        let feats = rv.plain_f32("f32.merger")?;
        let d = 2304;
        let rows: Vec<usize> = ids.iter().enumerate().filter(|(_, &t)| t == IMAGE_TOKEN).map(|(i, _)| i).collect();
        let ref_logits0 = rt.plain_f32("step0.logits")?;
        println!("INK2_VIS_TEXT {name}: {} ids, {} image rows, reference greedy {:?}", ids.len(), rows.len(), &greedy[..greedy.len().min(12)]);
        if let Some(cm) = &cm {
            let mut x = cm.embed(&ids)?;
            for (k, &r) in rows.iter().enumerate() { x[r * d..(r + 1) * d].copy_from_slice(&feats[k * d..(k + 1) * d]); }
            let e_in = rel_l2(&x, &rt.plain_f32("step0.inputs")?);
            let mut st = cm.new_state();
            let mut taps = crate::inkling2::cpu::Taps::new();
            let f = cm.forward_x(&mut st, x, ids.len(), Some(&mut taps))?;
            let lg = cm.logits(&f[(ids.len() - 1) * d..]);
            let e_v0 = rel_l2(&taps["v0"], &rt.plain_f32("step0.v0")?);
            let n = lg.len().min(ref_logits0.len());
            let e_lg = rel_l2(&lg[..n], &ref_logits0[..n]);
            let am = argmax(&lg[..n]) as u32;
            println!("INK2_VIS_TEXT {name} cpu: spliced inputs rel-L2 {e_in:.2e}, first-visit hidden v0 {e_v0:.2e}, last logits {e_lg:.2e}, argmax {am} (ref {})", greedy[0]);
            if e_in > 1e-6 || e_v0 > 1e-3 || am != greedy[0] { fails.push(format!("{name} cpu")); }
        }
        if let Some((dev, g, ws, lanes, v)) = gm.as_mut() {
            // tower on the reference pixels, features spliced exactly as the server does
            let pv = rv.plain_f32("pixel_values")?;
            let grid: Vec<usize> = c["grid"].as_array().context("grid")?.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
            let pre = PreImage { grid: Grid { rows: grid[1], cols: grid[2] }, pixels: pv, hash: 0, src: (0, 0), resized: None, had_alpha: false };
            let buf = dev.alloc_zeros::<f32>(pre.grid.tokens() * d)?;
            v.forward(&pre, crate::inkling2::gpu::dp(&buf), None)?;
            let ef = rel_l2(&dev.dtoh_sync_copy(&buf)?, &feats);
            let rbuf = dev.htod_sync_copy(&feats)?;
            for (src, label) in [(crate::inkling2::gpu::dp(&buf), "gpu tower features"), (crate::inkling2::gpu::dp(&rbuf), "reference f32 features")] {
            ws.img = vec![(rows[0], rows.len(), src)];
            lanes[0].pos = 0;
            g.prefill(ws, lanes, 0, &ids)?;
            ws.img.clear();
            let lg = g.logits_host(ws, 1)?;
            let n = lg.len().min(ref_logits0.len());
            let e0 = rel_l2(&lg[..n], &ref_logits0[..n]);
            let mut agree = (argmax(&lg[..n]) as u32 == greedy[0]) as usize;
            let (mut worst, mut wstep) = (e0, 0usize);
            let mut ours = vec![argmax(&lg[..n]) as u32];
            for k in 1..greedy.len() {
                crate::inkling2::model::probe::step1(g, ws, lanes, 0, greedy[k - 1])?;
                let lg = g.logits_host(ws, 1)?;
                let rl = rt.plain_f32(&format!("step{k}.logits"))?;
                let n = lg.len().min(rl.len());
                let e = rel_l2(&lg[..n], &rl[..n]);
                if e > worst { worst = e; wstep = k; }
                let am = argmax(&lg[..n]) as u32;
                ours.push(am);
                agree += (am == greedy[k]) as usize;
            }
            println!("INK2_VIS_TEXT {name} gpu [{label}]: features vs ref {ef:.2e}; last-prompt logits rel-L2 {e0:.2e}; greedy argmax agree {agree}/{} (fed the reference tokens), worst step logits rel-L2 {worst:.2e} (step {wstep}); ours {:?}",
                     greedy.len(), &ours[..ours.len().min(12)]);
            if agree + 1 < greedy.len() || ours[0] != greedy[0] { fails.push(format!("{name} gpu ({agree}/{})", greedy.len())); }
            }
        }
    }
    if fails.is_empty() { println!("INK2_VIS_TEXT_OK"); Ok(()) } else { bail!("INK2_VIS_TEXT_FAIL: {}", fails.join(", ")) }
}

/// `--probe-inkling2-vision --inkling2-vision-time <img>...`: tower time per image (no reference needed): our decode +
/// preprocessing (CPU, timed), then the GPU tower (CUDA events, median of 7 after 2 warm-ups), several sizes.
pub fn probe_time(dir: &Path, images: &[String], max_tokens: usize) -> Result<()> {
    let host = VisionHost::load(dir)?;
    let dev = cudarc::driver::CudaDevice::new(0).map_err(|e| anyhow::anyhow!("CudaDevice::new(0): {e:?}"))?;
    crate::inkling2::gpu::load_module(&dev)?;
    let mut v = VisionGpu::new(&dev, &host, true)?;
    for p in images {
        let bytes = std::fs::read(p).with_context(|| format!("read {p}"))?;
        let t0 = std::time::Instant::now();
        let pre = preproc::preprocess(&bytes, 40, max_tokens)?;
        let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
        let out = dev.alloc_zeros::<f32>(pre.grid.tokens() * host.cfg.text_hidden)?;
        let ms = crate::inkling2::gpu::time_ms(&dev, 2, 7, &mut || v.forward(&pre, crate::inkling2::gpu::dp(&out), None))?;
        println!("INK2_VIS_TIME {p}: {}x{} -> {} tokens ({} patches): decode+preprocess {cpu_ms:.1} ms (CPU), tower {ms:.2} ms (GPU){}",
                 pre.src.0, pre.src.1, pre.grid.tokens(), pre.grid.patches(),
                 pre.resized.map_or(String::new(), |(w, h)| format!(", downscaled to {w}x{h}")));
    }
    Ok(())
}
