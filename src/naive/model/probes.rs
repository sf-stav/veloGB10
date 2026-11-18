//! The `--probe-naive-*` gate entry points (fuse, slice, chunk-vs-CPU, full / full-dump) and their small helpers.
//! Siblings: `mod.rs` has the map; the probes drive `GpuModel` exactly as the server does.

use super::*;

/// `--probe-naive-fuse --model-dir <pack> [--layers N]`: the F13 fused split-K combine against the
/// two-launch form, eager, m in {1, 4, 8, 16}: logits and every layer tap must be BITWISE identical
/// (the fused epilogue runs nv_ks_combine_had's arithmetic in the same order; only the launch count changes).
pub fn probe_fuse(dir: &str, n_layers: usize) -> Result<()> {
    let pack = Arc::new(NaivePack::open(dir)?);
    println!("NAIVE FUSE PROBE {} layers {n_layers}", pack.dir);
    let gpu = GpuModel::load_slice(&pack, n_layers, 4096, 1024, 1)?;
    for &m in &[1usize, 4, 8, 16] {
        let ids: Vec<u32> = (0..m).map(|i| (1000 + (i as u32 * 7919) % 100_000) as u32).collect();
        let pos: Vec<usize> = (0..m).collect();
        super::gpu::set_fuse(false);
        let (lg0, taps0) = gpu.forward_rows(&ids, &pos)?;
        super::gpu::set_fuse(true);
        let (lg1, taps1) = gpu.forward_rows(&ids, &pos)?;
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        let bad_l = bits(&lg0).iter().zip(bits(&lg1)).filter(|(a, b)| **a != *b).count();
        let mut bad_t = 0usize;
        for l in 0..n_layers { bad_t += bits(&taps0[l]).iter().zip(bits(&taps1[l])).filter(|(a, b)| **a != *b).count(); }
        println!("  m {m:2}: logits {bad_l} of {} differ, taps {bad_t} differ", lg0.len());
        ensure!(bad_l == 0 && bad_t == 0, "NAIVE_FUSE_FAIL: the fused combine is not bitwise at m {m}");
    }
    println!("NAIVE_FUSE_OK");
    Ok(())
}

/// `--probe-naive-slice --model-dir <pack> --layers N --tokens T`: GPU slice vs CPU twin.
pub fn probe_slice(dir: &str, n_layers: usize, tokens: usize) -> Result<()> {
    let pack = Arc::new(NaivePack::open(dir)?);
    ensure!(tokens >= 1 && tokens <= MAX_ROWS, "tokens 1..16");
    println!("NAIVE SLICE PROBE {} layers {n_layers} tokens {tokens}", pack.dir);
    let t0 = std::time::Instant::now();
    let gpu = GpuModel::load_slice(&pack, n_layers, 4096, 1024, 1)?;
    println!("  gpu slice loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let t1 = std::time::Instant::now();
    let cpu = super::cpu::CpuModel::load_pack_slice(pack.clone(), n_layers)?;
    println!("  cpu twin loaded (lazy experts) in {:.1} s", t1.elapsed().as_secs_f64());
    // a deterministic prompt from real vocabulary ids
    let ids: Vec<u32> = (0..tokens).map(|i| (1000 + (i as u32 * 7919) % 100_000) as u32).collect();
    let pos: Vec<usize> = (0..tokens).collect();
    let t2 = std::time::Instant::now();
    let (lg, taps_g) = gpu.forward_rows(&ids, &pos)?;
    println!("  gpu forward {:.2} s", t2.elapsed().as_secs_f64());
    let t3 = std::time::Instant::now();
    let (lc, taps_c) = cpu.forward(&ids)?;
    println!("  cpu forward {:.1} s", t3.elapsed().as_secs_f64());
    let mut worst = 0f32;
    for l in 0..n_layers {
        let r = super::cpu::rel_l2(&taps_g[l], &taps_c.layer_out[l]);
        worst = worst.max(r);
        println!("  layer {l:2} ({:?}, {}): residual rel-L2 {:.3e}", gpu.cfg.attn[l], if gpu.cfg.moe[l] { "moe" } else { "dense" }, r);
    }
    let r = super::cpu::rel_l2(&lg, &lc);
    let v = gpu.cfg.vocab_size;
    let am = |x: &[f32]| x.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &y)| if y > b.1 { (i, y) } else { b }).0;
    let agree = (0..tokens).filter(|&t| am(&lg[t * v..(t + 1) * v]) == am(&lc[t * v..(t + 1) * v])).count();
    println!("  logits rel-L2 {:.3e}; argmax agreement {agree}/{tokens}", r);
    // f16 activations drift ~x1.5 per layer against the f32 twin (the served precision class);
    // argmax ties on a TRUNCATED model's near-flat logits are not a defect: >= 90% agreement.
    let ok = worst < 5e-2 && r < 5e-2 && agree * 10 >= tokens * 9;
    println!("NAIVE_SLICE_{}", if ok { "OK" } else { "FAIL" });
    if !ok { bail!("slice mismatch"); }
    Ok(())
}

/// index of the max element of one logit row (first wins ties, deterministic)
fn argmax_row(x: &[f32]) -> usize {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &y) in x.iter().enumerate() {
        if y > bv {
            bi = i;
            bv = y;
        }
    }
    bi
}

/// `--probe-naive-chunkxcpu --model-dir <pack> [--layers N] [--tokens T]` — the 1c gate, done as
/// a CROSS-CLASS check.
///
/// The over-claim it replaces: that "a 4200-token prompt returned 200 with plausible text" proved
/// `chunked == unchunked`. It did not. Worse, the comparison that IS available on the GPU is
/// same-class — two arms of the same forward — and per the 2026-09-22 lesson (`--probe-exl3-prefillx`
/// is the pattern) two paths of the same class can both be wrong in the same way and still call it
/// zero drift. So the reference side here is the CPU twin (`CpuModel::load_pack_slice`, dequantized
/// f32 weights, the model's own reference op order), a DIFFERENT class from the CUDA kernels, and
/// BOTH GPU arms are scored against it. A same-class A-vs-B row is printed as a diagnostic and is
/// explicitly NOT the proof.
///
/// The chunked arm is forced with a cap BELOW the row count: at the serving cap of 2048 a 16-row
/// prompt is a single pass, so the default arm would compare a call with itself. Engagement is
/// asserted (chunk count > 1) because bit-identity between two arms passes equally when the
/// chunked path never runs.
pub fn probe_chunk_xcpu(dir: &str, n_layers: usize, tokens: usize) -> Result<()> {
    let pack = Arc::new(NaivePack::open(dir)?);
    ensure!(tokens >= 2 && tokens <= MAX_ROWS, "tokens 2..16");
    let cap = (tokens / 2).max(1);
    let n_chunks = chunk_ranges(tokens, cap).len();
    ensure!(n_chunks > 1, "cap {cap} on {tokens} rows must actually chunk — a single pass proves nothing");
    println!("NAIVE CHUNK-XCPU PROBE {} layers {n_layers} tokens {tokens} cap {cap} ({n_chunks} chunks vs 1)", pack.dir);
    let t0 = std::time::Instant::now();
    let gpu = GpuModel::load_slice(&pack, n_layers, 4096, 1024, 1)?;
    println!("  gpu slice loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let t1 = std::time::Instant::now();
    let cpu = super::cpu::CpuModel::load_pack_slice(pack.clone(), n_layers)?;
    println!("  cpu twin loaded (lazy experts) in {:.1} s", t1.elapsed().as_secs_f64());
    // a deterministic prompt from real vocabulary ids (same construction as probe_slice)
    let ids: Vec<u32> = (0..tokens).map(|i| (1000 + (i as u32 * 7919) % 100_000) as u32).collect();
    let pos: Vec<usize> = (0..tokens).collect();

    let t2 = std::time::Instant::now();
    let (lg_one, taps_one) = gpu.forward_rows_cap(&ids, &pos, MAX_PREFILL_ROWS)?;
    let t_one = t2.elapsed().as_secs_f64();
    let t3 = std::time::Instant::now();
    let (lg_chunk, taps_chunk) = gpu.forward_rows_cap(&ids, &pos, cap)?;
    let t_chunk = t3.elapsed().as_secs_f64();
    let t4 = std::time::Instant::now();
    let (lg_cpu, taps_cpu) = cpu.forward(&ids)?;
    let t_cpu = t4.elapsed().as_secs_f64();
    println!("  gpu 1-pass {:.3} s | gpu {n_chunks}-chunk {:.3} s | cpu reference {:.1} s", t_one, t_chunk, t_cpu);

    let v = gpu.cfg.vocab_size;
    let am = |x: &[f32]| argmax_row(x);
    let agree = |a: &[f32], b: &[f32]| (0..tokens).filter(|&t| am(&a[t * v..(t + 1) * v]) == am(&b[t * v..(t + 1) * v])).count();
    // The residual column compares the A side's taps against the B side's taps, so BOTH are passed.
    // The first version captured `taps_cpu` and ignored the taps argument; the same-class row then
    // printed a CROSS-class residual (chunked-vs-cpu) under a same-class label. Reporting defect.
    let rows = |name: &str, a: &[f32], b: &[f32], ta: &[Vec<f32>], tb: &[Vec<f32>]| {
        let worst = (0..n_layers).map(|l| super::cpu::rel_l2(&ta[l], &tb[l])).fold(0f32, f32::max);
        let r = super::cpu::rel_l2(a, b);
        let ag = agree(a, b);
        println!("  {name}: worst layer residual rel-L2 {worst:.3e} | logits rel-L2 {r:.3e} | argmax {ag}/{tokens}");
        (worst, r, ag)
    };
    // CROSS-CLASS (the proof): each GPU arm against the CPU reference of a different class
    let (w_chunk, r_chunk, a_chunk) = rows("gpu CHUNKED  vs cpu", &lg_chunk, &lg_cpu, &taps_chunk, &taps_cpu.layer_out);
    let (w_one, r_one, a_one) = rows("gpu 1-PASS   vs cpu", &lg_one, &lg_cpu, &taps_one, &taps_cpu.layer_out);
    // SAME-CLASS diagnostic only — printed so a chunking-specific divergence is visible, but it
    // cannot establish correctness on its own
    let (w_ab, r_ab, a_ab) = rows("gpu chunked vs gpu 1-pass [SAME-CLASS, diagnostic only]", &lg_chunk, &lg_one, &taps_chunk, &taps_one);
    let ok = w_chunk < 5e-2 && r_chunk < 5e-2 && a_chunk * 10 >= tokens * 9
        && w_one < 5e-2 && r_one < 5e-2 && a_one * 10 >= tokens * 9;
    println!("  chunk-vs-cpu vs one-pass-vs-cpu delta: residual {:.3e}, logits {:.3e}, argmax {}",
             (w_chunk - w_one).abs(), (r_chunk - r_one).abs(), a_chunk.abs_diff(a_one));
    println!("NAIVE_CHUNK_XCPU_{}", if ok { "OK" } else { "FAIL" });
    if !ok {
        bail!("chunked GPU prefill disagrees with the CPU reference (chunked {w_chunk:.3e}/{r_chunk:.3e}/{a_chunk}, one-pass {w_one:.3e}/{r_one:.3e}/{a_one}; same-class {w_ab:.3e}/{r_ab:.3e}/{a_ab})");
    }
    Ok(())
}

/// `--probe-naive-full --model-dir <pack> [--prompt S] [--gen N]`: the WHOLE model on one GPU with one
/// layer resident at a time (137 GB does not fit beside the desk server), greedy decode of `gen`
/// tokens by re-running the prompt + generated rows every step (M <= 16 rows; the K/V caches do
/// not survive a layer's drop). Coherence of the complete GPU path on real weights: the CPU
/// reference gave " Paris" p = 0.946 for the default prompt.
/// A minimal safetensors writer (header length u64 LE + JSON header + raw little-endian data) for the per-layer
/// state dumps of `probe_full --dump-states` (read by scripts/naive/requant/compare_states.py).
fn write_safetensors(path: &str, tensors: &[(String, &str, Vec<usize>, Vec<u8>)]) -> Result<()> {
    let mut header = String::from("{");
    let mut off = 0usize;
    for (i, (name, dtype, shape, data)) in tensors.iter().enumerate() {
        if i > 0 { header.push(','); }
        header.push_str(&format!("\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[{off},{}]}}", off + data.len()));
        off += data.len();
    }
    header.push('}');
    while header.len() % 8 != 0 { header.push(' '); }
    let mut out = Vec::with_capacity(8 + header.len() + off);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for (_, _, _, data) in tensors { out.extend_from_slice(data); }
    std::fs::write(path, out).with_context(|| format!("write {path}"))?;
    Ok(())
}

pub fn probe_full(dir: &str, prompt: &str, gen: usize, rows_chunk: usize, nll: bool) -> Result<()> {
    probe_full_dump(dir, prompt, gen, rows_chunk, nll, None)
}

/// As `probe_full`; with `dump` = Some(path) the residual stream after the embedding and after every layer of
/// the prompt pass (f32 [m][h], keys `000.embed`, `001.layer0`, ...) plus the ids and the last row's logits are
/// written to `path` as safetensors, for a layer-by-layer comparison against the HF reference / exllamav3.
pub fn probe_full_dump(dir: &str, prompt: &str, gen: usize, rows_chunk: usize, nll: bool, dump: Option<&str>) -> Result<()> {
    let pack = Arc::new(NaivePack::open(dir)?);
    let tok = crate::tokenizer::QwenTokenizer::from_file(&format!("{}/tokenizer.json", pack.dir)).context("tokenizer")?;
    let mut ids = tok.encode(prompt, false)?;
    ensure!(!ids.is_empty() && ids.len() + gen <= MAX_PREFILL_ROWS, "prompt + gen must fit {MAX_PREFILL_ROWS} rows");
    println!("NAIVE FULL PROBE {} prompt {:?} -> {} tokens, gen {gen}", pack.dir, prompt, ids.len());
    let t0 = std::time::Instant::now();
    let gpu = GpuModel::load_slice(&pack, 0, 4096, 1024, 1)?;
    println!("  trunk (embed, norm, lm_head) loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let c = gpu.cfg.clone();
    let (h, v, nl) = (c.hidden_size, c.vocab_size, c.num_layers);
    let dev = &gpu.dev;
    let mut out = String::new();
    for step in 0..=gen {
        let m = ids.len();
        let pos: Vec<usize> = (0..m).collect();
        let ids_i: Vec<i32> = ids.iter().map(|x| *x as i32).collect();
        let pos_i: Vec<i32> = pos.iter().map(|x| *x as i32).collect();
        let ids_d = dev.htod_sync_copy(&ids_i)?;
        let pos_d = dev.htod_sync_copy(&pos_i)?;
        // a chunk of <= MAX_ROWS rows takes the DECODE-class kernels (64-way DSA splits, the small-M MoE path), whose
        // per-chunk scratch is sized for m = 16 regardless of the prompt: give the probe that headroom
        let decode_extra = if rows_chunk != 0 && rows_chunk <= MAX_ROWS { 768usize << 20 } else { 0 };
        let mut ws = Ws::new(dev, ws_bytes(&c, m, 4096, 0) + decode_extra)?;
        let x = ws.alloc::<f32>(m * h)?;
        let hbuf = ws.alloc::<u16>(m * h)?;
        gpu.l("nv_embed", LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, (x, &gpu.embed, dptr(&ids_d), h as i32))?;
        let mut dumped: Vec<(String, &str, Vec<usize>, Vec<u8>)> = Vec::new();
        if dump.is_some() && step == 0 {
            gpu.sync()?;
            let xs = dtoh::<f32>(x, m * h)?;
            dumped.push(("000.embed".to_string(), "F32", vec![m, h], xs.iter().flat_map(|f| f.to_le_bytes()).collect()));
        }
        let t1 = std::time::Instant::now();
        let (mut tl, mut tf) = (0f64, 0f64);
        for li in 0..nl {
            let a = std::time::Instant::now();
            let layer = GpuModel::load_layer(dev, &pack, li, 4096, 1024, 1, &gpu.shard)?;
            tl += a.elapsed().as_secs_f64();
            let b = std::time::Instant::now();
            // rows_chunk < m: the layer sees the rows in chunks (the small-row kernels when chunk <= MAX_ROWS);
            // the layer's KV cache carries the earlier chunks, as a chunked prefill does
            let step_rows = if rows_chunk == 0 { m } else { rows_chunk };
            let mut c0 = 0usize;
            while c0 < m {
                let mc = (m - c0).min(step_rows);
                let mark = ws.mark();
                gpu.layer_forward_ws(&layer, li, x + (c0 * h * 4) as u64, hbuf + (c0 * h * 2) as u64, dptr(&pos_d) + (c0 * 4) as u64, mc, gpu.sids, &mut ws)?;
                ws.release(mark);
                c0 += mc;
            }
            gpu.sync()?;
            tf += b.elapsed().as_secs_f64();
            if dump.is_some() && step == 0 {
                let xs = dtoh::<f32>(x, m * h)?;
                dumped.push((format!("{:03}.layer{li}", li + 1), "F32", vec![m, h], xs.iter().flat_map(|f| f.to_le_bytes()).collect()));
            }
            if step == 0 {
                let xs = dtoh::<f32>(x, m * h)?;
                let rms = (xs.iter().map(|a| (*a as f64) * (*a as f64)).sum::<f64>() / xs.len() as f64).sqrt();
                let nan = xs.iter().filter(|a| !a.is_finite()).count();
                println!("  layer {li:2} ({:?}, {}): residual rms {rms:.4} nonfinite {nan}  [load {:.1} s fwd {:.3} s]",
                         c.attn[li], if c.moe[li] { "moe" } else { "dense" }, a.elapsed().as_secs_f64() - (b.elapsed().as_secs_f64()), b.elapsed().as_secs_f64());
            }
            drop(layer);
        }
        if nll && step == 0 {
            // teacher-forced mean NLL of the prompt (the recipe's bench.py `nll` suite: vLLM prompt_logprobs over
            // every token but the first): the head on every row, in MAX_ROWS groups, fp32 log-softmax
            let mut sum = 0f64; let mut cnt = 0usize; let mut worst: Vec<(f64, usize)> = Vec::new();
            let mut g0 = 0usize;
            while g0 < m {
                let gm = (m - g0).min(MAX_ROWS);
                let mark = ws.mark();
                let lg = gpu.head_logits_ws(x + (g0 * h * 4) as u64, hbuf + (g0 * h * 2) as u64, gm, &mut ws)?;
                gpu.sync()?;
                let rows: Vec<f32> = dtoh::<u16>(lg, gm * v)?.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect();
                for r in 0..gm {
                    let ti = g0 + r;
                    if ti + 1 >= m { break; }
                    let row = &rows[r * v..(r + 1) * v];
                    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let z: f64 = row.iter().map(|l| ((l - mx) as f64).exp()).sum();
                    let nl = -(((row[ids[ti + 1] as usize] - mx) as f64) - z.ln());
                    sum += nl; cnt += 1; worst.push((nl, ti));
                }
                ws.release(mark);
                g0 += gm;
            }
            worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            println!("  NLL: {} tokens scored, mean {:.5} nats/token (ppl {:.3})", cnt, sum / cnt as f64, (sum / cnt as f64).exp());
            println!("  worst 5 positions: {:?}", worst.iter().take(5).map(|(n, i)| (format!("{n:.2}"), *i, tok.decode(&[ids[*i + 1]], false).unwrap_or_default())).collect::<Vec<_>>());
        }
        // the head on the LAST row only (a wide prompt's rows share the trunk; the head is 1 row)
        let lg = gpu.head_logits_ws(x + ((m - 1) * h * 4) as u64, hbuf + ((m - 1) * h * 2) as u64, 1, &mut ws)?;
        gpu.sync()?;
        let logits: Vec<f32> = dtoh::<u16>(lg, v)?.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect();
        if let (Some(path), 0) = (dump, step) {
            dumped.push(("ids".to_string(), "I64", vec![m], ids.iter().flat_map(|t| (*t as i64).to_le_bytes()).collect()));
            dumped.push(("logits_last".to_string(), "F32", vec![v], logits.iter().flat_map(|f| f.to_le_bytes()).collect()));
            write_safetensors(path, &dumped)?;
            println!("  dumped {} tensors to {path}", dumped.len());
        }
        let last = &logits[..v];
        let am = last.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0;
        if step == 0 {
            let mx = last.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let z: f64 = last.iter().map(|l| ((l - mx) as f64).exp()).sum();
            let mut idx: Vec<usize> = (0..v).collect();
            idx.sort_by(|a, b| last[*b].partial_cmp(&last[*a]).unwrap_or(std::cmp::Ordering::Equal));
            println!("  top-5 after the prompt:");
            for &i in &idx[..5] { println!("    {i:7} p={:.3} {:?}", ((last[i] - mx) as f64).exp() / z, tok.decode(&[i as u32], false).unwrap_or_default()); }
            println!("  top-5 ids: {:?}", &idx[..5]);
        }
        println!("  step {step}: {m} rows, {:.1} s (layer loads {tl:.1} s, forwards {tf:.2} s) -> token {am} {:?}",
                 t1.elapsed().as_secs_f64(), tok.decode(&[am as u32], false).unwrap_or_default());
        out += &tok.decode(&[am as u32], false).unwrap_or_default();
        ids.push(am as u32);
    }
    println!("  greedy continuation: {:?}", out);
    println!("NAIVE_FULL_DONE");
    Ok(())
}

