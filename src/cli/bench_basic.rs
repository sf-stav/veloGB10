//! The basic decode/prefill/batch/tree/lane/accept/tau benches (the interactive CLI, `run_cli`, is in `args.rs`).
//! Bench harnesses here run the shipped decode path; acceptance numbers are the check on the draft side
//! (a lossless gate alone proves nothing about the draft).

use super::*;

/// Compact acceptance discriminator, shared by --bench-accept and the TP=2 variant
/// (--tp-accept in tp_serve): overall rate, the target-confidence buckets that separate
/// HARD TEXT (target unsure where the head misses) from a WEAK HEAD (misses even when the
/// target is confident), and top-2/3 fork coverage by draft depth.
pub(super) fn bench_accept_report(depth: usize, s: &[gb10_inference::gpu::AcceptSample]) {
    if s.is_empty() { println!("bench_accept: NO samples"); return; }
    let acc = s.iter().filter(|x| x.accepted).count() as f32 / s.len() as f32;
    println!("\n=== {} draft positions on the correct prefix, acceptance {:.1}%", s.len(), 100.0 * acc);
    println!("  acceptance bucketed by the TARGET's own top-1 probability:");
    for (lo, hi) in [(0.0f32, 0.3f32), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)] {
        let b: Vec<_> = s.iter().filter(|x| x.target_top1_p >= lo && x.target_top1_p < hi).collect();
        if b.is_empty() { continue; }
        let a = b.iter().filter(|x| x.accepted).count() as f32 / b.len() as f32;
        println!("    {lo:.2}-{hi:.2}      {:5}     {:5.1}%", b.len(), 100.0 * a);
    }
    let confident: Vec<_> = s.iter().filter(|x| x.target_top1_p >= 0.9).collect();
    let uncertain: Vec<_> = s.iter().filter(|x| x.target_top1_p < 0.5).collect();
    if !confident.is_empty() {
        println!("  CONFIDENT target (p>=0.9): {:5} positions, {:.1}% accepted", confident.len(),
                 100.0 * confident.iter().filter(|x| x.accepted).count() as f32 / confident.len() as f32);
    }
    if !uncertain.is_empty() {
        println!("  UNSURE target    (p< 0.5): {:5} positions, {:.1}% accepted ({:.0}% of all)", uncertain.len(),
                 100.0 * uncertain.iter().filter(|x| x.accepted).count() as f32 / uncertain.len() as f32,
                 100.0 * uncertain.len() as f32 / s.len() as f32);
    }
    println!("  HARD TEXT => target unsure a lot, acceptance high where sure.  WEAK HEAD => poor even when sure.");
    println!("  fork coverage (target argmax in head top-k) by draft depth:");
    for d in 1..depth {
        let b: Vec<_> = s.iter().filter(|x| x.depth_idx == d).collect();
        if b.is_empty() { continue; }
        let n = b.len() as f32;
        let (t1, t2, t3) = (b.iter().filter(|x| x.accepted).count() as f32 / n,
                            b.iter().filter(|x| x.covered_top2).count() as f32 / n,
                            b.iter().filter(|x| x.covered_top3).count() as f32 / n);
        println!("    pos {d:<2}  {:5}  top1 {:5.1}%  top2 {:5.1}%  top3 {:5.1}%  (fork rescue +{:4.1}%)",
                 b.len(), 100.0*t1, 100.0*t2, 100.0*t3, 100.0*(t2-t1));
    }
}

pub(crate) fn run_bench_prefill(args: &[String]) {
    let dir = parse_arg(args, "--model-dir").expect("--bench-prefill requires --model-dir");
    let seq_len: usize = parse_arg(args, "--seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);
    let reps: usize = parse_arg(args, "--reps").and_then(|s| s.parse().ok()).unwrap_or(5);
    let (gpu, _) = load_model_gpu(dir, None, 1);
    // synthetic prompt of seq_len tokens (content irrelevant to prefill cost)
    let prompt: Vec<u32> = (0..seq_len).map(|i| ((i * 2654435761usize) % 30000 + 5) as u32).collect();
    let max_seq_len = (seq_len + 128).next_power_of_two();
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    let mut state = gpu.new_batch_state(1, 1, max_seq_len);
    let kv_stride = max_seq_len;
    // warmup
    gpu.prefill_batch(&mut pool, &prompt, &mut state, 0, kv_stride, 0);
    gpu.sync_stream();
    let t0 = std::time::Instant::now();
    for _ in 0..reps { gpu.prefill_batch(&mut pool, &prompt, &mut state, 0, kv_stride, 0); }
    gpu.sync_stream();
    let ms = t0.elapsed().as_secs_f32() * 1e3 / reps as f32;
    println!("prefill N={seq_len}  {ms:.1} ms  ({:.0} tok/s)", seq_len as f32 / ms * 1e3);
}

pub(crate) fn run_probe_ppsplit(args: &[String]) {
    let dir = parse_arg(args, "--model-dir").expect("--probe-ppsplit requires --model-dir");
    let seq_len: usize = parse_arg(args, "--seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);
    let split: usize = parse_arg(args, "--split").and_then(|s| s.parse().ok()).unwrap_or(32);
    let (gpu, _) = load_model_gpu(dir, None, 1);
    let prompt: Vec<u32> = (0..seq_len).map(|i| ((i * 2654435761usize) % 30000 + 5) as u32).collect();
    let max_seq_len = (seq_len + 128).next_power_of_two();
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    let state_slots = 2usize;
    let mut state = gpu.new_batch_state(state_slots, state_slots, max_seq_len);
    let kv_stride = max_seq_len;
    let ok = gpu.probe_ppsplit(&mut pool, &mut state, &prompt, kv_stride, split, state_slots);
    std::process::exit(if ok { 0 } else { 1 });
}

/// DSV4 single-process isolated prefill bench (R2.2). Loads the full trunk (43 layers)
/// single-process and times `Dsv4GpuModel::forward` (auto-chunked at PREFILL_CHUNK=4096) over a
/// synthetic prompt of `--seq-len` tokens. Reports per-rep and best prefill tok/s. When
/// --prefill-trace is set, the per-chunk timers + attention-path logs (batched vs
/// SEQ-per-token) fire too — the audit's "is the batched chunk-prefill path engaged for chunks
/// 2+?" probe. Single-process (rank 0/1, no TP) — measures the prefill path in isolation, not the
/// TP=2 serve path. `--dsv4-bench-prefill --model-dir <bundle> --seq-len N [--reps R]`.
pub(crate) fn run_dsv4_bench_prefill(args: &[String]) {
    use gb10_inference::dsv4_load;
    use gb10_inference::dsv4_model::{Dsv4GpuModel, PREFILL_CHUNK};
    use std::path::Path;
    let bundle = parse_arg(args, "--model-dir").expect("--dsv4-bench-prefill requires --model-dir <bundle>");
    let seq_len: usize = parse_arg(args, "--seq-len").and_then(|s| s.parse().ok()).unwrap_or(8192);
    let reps: usize = parse_arg(args, "--reps").and_then(|s| s.parse().ok()).unwrap_or(3);
    let cfg = dsv4_load::load_config(Path::new(bundle)).expect("load_config");
    let dev = cudarc::driver::CudaDevice::new(0).expect("CUDA device 0");
    // s_max caps the prefill scratch at ONE chunk (the serving configuration) — forward auto-chunks
    // the prompt internally, so all scratch is chunk-sized regardless of seq_len.
    let s_max = (PREFILL_CHUNK + 16).max(256);
    let max_seq_len = (seq_len + 256).max(2048);
    eprintln!("[dsv4-bench-prefill] loading {n} trunk layers single-process (max_seq_len={max_seq_len}, s_max={s_max}) ...",
        n = cfg.n_layers);
    let t_load = std::time::Instant::now();
    let mut m = Dsv4GpuModel::load(&dev, Path::new(bundle), &cfg, max_seq_len, s_max, cfg.n_layers)
        .expect("Dsv4GpuModel::load");
    let load_secs = t_load.elapsed().as_secs_f64();
    eprintln!("[dsv4-bench-prefill] shard load: {load_secs:.1}s");
    // synthetic prompt (content irrelevant to prefill cost)
    let prompt: Vec<i32> = (0..seq_len).map(|i| ((7 + i as i64 * 9973) % cfg.vocab_size as i64) as i32).collect();
    // warmup rep (excluded) — primes allocator pools / JIT
    let _ = m.forward(&prompt, 0).expect("warmup forward");
    m.rt.dev.synchronize().ok();
    let mut best_ms = f64::INFINITY;
    let mut per_rep = Vec::with_capacity(reps);
    for r in 0..reps {
        m.reset_states().expect("reset_states");
        let t0 = std::time::Instant::now();
        let _logits = m.forward(&prompt, 0).expect("forward");
        m.rt.dev.synchronize().ok();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let tps = seq_len as f64 / ms * 1e3;
        eprintln!("[dsv4-bench-prefill] rep {r}: {ms:.0} ms  ({tps:.1} tok/s)");
        per_rep.push((ms, tps));
        if ms < best_ms { best_ms = ms; }
    }
    let best_tps = seq_len as f64 / best_ms * 1e3;
    println!("=== DSV4 prefill (single-process, {n} layers, N={seq_len}) ===", n = cfg.n_layers);
    println!("  load: {load_secs:.1}s   reps: {reps}");
    for (r, (ms, tps)) in per_rep.iter().enumerate() {
        println!("  rep {r}: {ms:.0} ms  ({tps:.1} tok/s)");
    }
    println!("  BEST: {best_ms:.0} ms  ({best_tps:.1} tok/s)   [chunk={PREFILL_CHUNK}, chunks={}]", (seq_len + PREFILL_CHUNK - 1) / PREFILL_CHUNK);
}

/// DSV4 single-process isolated decode bench (R3.0). Loads the full trunk single-process,
/// prefills a synthetic `--prompt-len` prompt, then greedy-decodes `--max-new-tokens` tokens,
/// timing the FULL serving step per token (forward → dtoh logits → host argmax — the same
/// shape the TP=2 serve loop pays). One warmup token (JIT/allocator) is excluded. Reports
/// per-token min/median/mean ms + tok/s. This is the per-lever decode iteration harness and
/// the ncu steady-state target: it isolates the decode path from TP transport (the rulers
/// stay the TP=2 --head ladder + the tool-eval harness). Single-process reads ALL 256 experts
/// (~10.1 GB/token computed) — scale expectations accordingly vs the TP=2 per-node ~8 GB.
/// `--dsv4-bench-decode --model-dir <bundle> [--max-new-tokens N] [--prompt-len P]`.
pub(crate) fn run_dsv4_bench_decode(args: &[String]) {
    use gb10_inference::dsv4_load;
    use gb10_inference::dsv4_model::{Dsv4GpuModel, PREFILL_CHUNK};
    use std::path::Path;
    let bundle = parse_arg(args, "--model-dir").expect("--dsv4-bench-decode requires --model-dir <bundle>");
    let max_new: usize = parse_arg(args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(64);
    let prompt_len: usize = parse_arg(args, "--prompt-len").and_then(|s| s.parse().ok()).unwrap_or(128);
    // --shard rank,world: load the TP=2 rank shard (load_converted on the sharded bundle). The
    // FULL 43-layer single-process load is ~130 GB — over .11's 121 GB (earlyoom SIGTERM at
    // layer ~28, measured twice). The rank-0 shard (~103 GB) is the TP per-node footprint and
    // the ncu/iteration target: same per-node kernel mix minus the TP comm kernels. Outputs are
    // numerically PARTIAL (remote experts contribute zero — no all-reduce partner); this mode
    // is a PERFORMANCE proxy, never a correctness gate.
    let shard: Option<(usize, usize)> = parse_arg(args, "--shard").map(|s| {
        let mut it = s.split(',');
        (it.next().expect("--shard rank,world").parse().expect("rank"),
         it.next().expect("--shard rank,world").parse().expect("world"))
    });
    let cfg = dsv4_load::load_config(Path::new(bundle)).expect("load_config");
    let dev = cudarc::driver::CudaDevice::new(0).expect("CUDA device 0");
    let max_seq_len = (prompt_len + max_new + 256).max(2048);
    let s_max = (prompt_len + 16).max(256).min(PREFILL_CHUNK);
    eprintln!("[dsv4-bench-decode] loading {n} trunk layers (max_seq_len={max_seq_len}, s_max={s_max}, shard={shard:?}) ...",
        n = cfg.n_layers);
    let t_load = std::time::Instant::now();
    let dev = std::sync::Arc::new(dev);
    let mut m = match shard {
        Some((rank, world)) => {
            eprintln!("[dsv4-bench-decode] SHARD MODE rank {rank}/{world} — PERF PROXY ONLY (outputs numerically partial)");
            Dsv4GpuModel::load_converted(&dev, Path::new(bundle), &cfg, max_seq_len, s_max, cfg.n_layers, rank, world)
                .expect("Dsv4GpuModel::load_converted (shard)")
        }
        None => Dsv4GpuModel::load(&dev, Path::new(bundle), &cfg, max_seq_len, s_max, cfg.n_layers)
            .expect("Dsv4GpuModel::load"),
    };
    eprintln!("[dsv4-bench-decode] shard load: {:.1}s", t_load.elapsed().as_secs_f64());
    let prompt: Vec<i32> = (0..prompt_len).map(|i| ((7 + i as i64 * 9973) % cfg.vocab_size as i64) as i32).collect();
    let mut logits = {
        let t0 = std::time::Instant::now();
        let l = m.forward(&prompt, 0).expect("prefill forward");
        m.rt.dev.synchronize().ok();
        eprintln!("[bench-decode] prefill {} tok: {:.1} ms ({:.1} tok/s)", prompt_len, t0.elapsed().as_secs_f64() * 1e3, prompt_len as f64 / t0.elapsed().as_secs_f64());
        l
    };
    let mut pos = prompt_len;
    // warmup token (excluded): primes JIT + allocator pools on the decode shapes
    let mut tok = dsv4_argmax(&m.rt.dev.dtoh_sync_copy(&logits).expect("dtoh logits")) as i32;
    logits = m.forward(&[tok], pos).expect("warmup decode forward");
    m.rt.dev.synchronize().ok();
    tok = dsv4_argmax(&m.rt.dev.dtoh_sync_copy(&logits).expect("dtoh logits")) as i32;
    pos += 1;
    let mut per_tok: Vec<f64> = Vec::with_capacity(max_new);
    for _ in 0..max_new {
        let t0 = std::time::Instant::now();
        logits = m.forward(&[tok], pos).expect("decode forward");
        m.rt.dev.synchronize().ok();
        tok = dsv4_argmax(&m.rt.dev.dtoh_sync_copy(&logits).expect("dtoh logits")) as i32;
        per_tok.push(t0.elapsed().as_secs_f64() * 1e3);
        pos += 1;
    }
    let mut sorted = per_tok.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (min, med) = (sorted[0], sorted[sorted.len() / 2]);
    let mean = per_tok.iter().sum::<f64>() / per_tok.len() as f64;
    println!("=== DSV4 decode (single-process, {n} layers, prompt={prompt_len}, {max_new} tokens) ===", n = cfg.n_layers);
    println!("  per-token ms: min {min:.1}  median {med:.1}  mean {mean:.1}");
    println!("  tok/s:        max {:.2}  median {:.2}  mean {:.2}", 1e3 / min, 1e3 / med, 1e3 / mean);
    // --bench-logits-hash: serving-level A/B fingerprint (e.g. --packed-cache on vs
    // off) — FNV-1a over the final logits' f32 bits + the emitted argmax chain. Identical
    // hashes across arms = bitwise-identical serving decode at this depth.
    if gb10_inference::opts::var(gb10_inference::opt!("bench-logits-hash")).is_ok() {
        let lg: Vec<f32> = m.rt.dev.dtoh_sync_copy(&logits).expect("dtoh final logits");
        let mut h: u64 = 0xcbf29ce484222325;
        for v in &lg {
            for b in v.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        println!("  logits-fnv1a: {h:016x}  last-tok: {tok}");
    }
}

pub(crate) fn run_bench_batch(args: &[String]) {
    let (model_path, tokenizer_path) = if let Some(dir) = parse_arg(args, "--model-dir") {
        // Always pass the directory — Model::load handles directory vs file
        (dir.to_string(), format!("{}/tokenizer.json", dir.trim_end_matches('/')))
    } else {
        (parse_arg(args, "--model").unwrap_or("model/model.safetensors").to_string(),
         parse_arg(args, "--tokenizer").unwrap_or("model/tokenizer.json").to_string())
    };
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("The capital of France is");
    let m: usize = parse_arg(args, "--batch").and_then(|s| s.parse().ok()).unwrap_or(4);
    let max_new: usize = parse_arg(args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(32);
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);

    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");
    println!("Batched benchmark: M={} seqs, prompt={} tokens, decode={} tokens", m, prompt.len(), max_new);

    let gpu = if std::path::Path::new(&model_path).is_dir() {
        let (gpu, _) = load_model_gpu(&model_path, None, 1);
        gpu
    } else {
        let host = gb10_inference::qwen::Model::load(&model_path).expect("load model");
        gb10_inference::gpu::GpuModel::new(&host).expect("gpu init")
    };
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    let mut state = gpu.new_batch_state(m, m, max_seq_len);

    let start = std::time::Instant::now();
    let (tokens, agg) = gpu.bench_batch(&mut pool, &mut state, &prompt, m, max_new, max_seq_len);
    let total = start.elapsed();
    let text = tokenizer.decode(&tokens, true).unwrap_or_default();

    println!("Aggregate decode throughput: {:.1} tok/s  ({} seqs)", agg, m);
    println!("Wall time (prefill+decode):  {:.2?}", total);
    println!("Slot-0 output: {}", text);
    println!("Slot-0 tokens: {:?}", &tokens[..tokens.len().min(16)]);
    // Full sequence, machine-readable: the TP=2 divergence gate diffs this against the sharded run.
    // 16 tokens is an argmax-stability sample, not a bound — reassociation drift from the K/2 split
    // shows up at hundreds of tokens, so the gate needs the whole sequence.
    println!("GATE_TOKENS {}", tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
}

/// MTP verify lossless probe: confirm the K-token causal-append forward (verify_forward) produces
/// predictions identical to sequential greedy decoding. Greedy MTP output must equal greedy non-MTP
/// output token-for-token; this isolates and validates the append/verify primitive before any MTP
/// scheduling is wired in.
/// `--bench-accept --model-dir <d> --prompt <p> [--depth N] [--max-new-tokens N]`
///
/// Buckets MTP acceptance by the TARGET's own confidence. See `GpuModel::bench_accept`.
/// `--bench-tree --model-dir <d> [--prompt <p>] [--depth N]` — twin-chain planted-tree byte gate.
pub(crate) fn run_bench_tree(args: &[String]) {
    let dir = parse_arg(args, "--model-dir").expect("--bench-tree requires --model-dir");
    let tok_path = format!("{}/tokenizer.json", dir.trim_end_matches('/'));
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("The quick brown fox jumps over the lazy dog near the river bank at dawn.");
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(8192);
    let tokenizer = QwenTokenizer::from_file(&tok_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");
    let (gpu, _) = load_model_gpu(dir, None, 1);
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    // 1 KV lane + (2 + MAX_VERIFY) GDN state slots: slot 0 the lane, 1 the MTP snapshot, 2.. the
    // per-column tree checkpoints the parent reload reads.
    let mut state = gpu.new_batch_state(1, 2 + gb10_inference::gpu::MAX_VERIFY, max_seq_len);
    gpu.dev().synchronize().unwrap();

    println!("Twin-chain planted-tree byte gate: prompt={} tokens", prompt.len());
    let mut total_mism = 0usize;
    // Sweep depths (twin width = 2d-1 must be <= MAX_VERIFY=16, so d <= 8) and prompt offsets that
    // straddle the 256 split boundary (via prompt length).
    for depth in [2usize, 3, 4, 6, 8] {
        for take in [64usize, 255, 256, 300, prompt.len().min(511)] {
            if take < 2 || take > prompt.len() { continue; }
            let (cmp, mism) = gpu.bench_tree_twin(&mut pool, &mut state, &prompt[..take], max_seq_len, depth);
            total_mism += mism;
            println!("  depth {depth} ctx {take:4}: {cmp} twin pairs, {mism} bit-mismatch{}",
                     if mism == 0 { "  OK" } else { "  <-- FAIL" });
        }
    }
    // PATH-ORACLE: random trees, each column vs its ancestor-path chain (absolute ground truth).
    println!("\n  path-oracle fuzz (random trees vs per-column chains):");
    let take = prompt.len().min(600);
    for trial in 0..12u64 {
        let width = 4 + (trial as usize % 12);   // 4..15 columns
        let (cols, mism) = gpu.bench_tree_oracle(&mut pool, &mut state, &prompt[..take],
                                                 max_seq_len, width.min(gb10_inference::gpu::MAX_VERIFY),
                                                 0xA53F ^ trial.wrapping_mul(0x9E3779B1));
        total_mism += mism;
        println!("    trial {trial:2} width {width:2}: {cols} columns, {mism} bit-mismatch{}",
                 if mism == 0 { "  OK" } else { "  <-- FAIL" });
    }

    // ACCEPT + COMPACT end-to-end (Step 3a): planted fork, second branch = target greedy.
    println!("\n  accept-walk + KV compaction (planted fork, second branch = target greedy):");
    let mut accept_fail = 0usize;
    for depth in [2usize, 3, 4, 6] {
        let (emit_ok, kv_ok) = gpu.bench_tree_accept(&mut pool, &mut state, &prompt[..take.min(prompt.len())], max_seq_len, depth);
        if !emit_ok || !kv_ok { accept_fail += 1; }
        println!("    depth {depth}: emitted==greedy {}  kv_compacted {}",
                 if emit_ok {"OK"} else {"FAIL"}, if kv_ok {"OK"} else {"FAIL"});
    }

    if total_mism == 0 && accept_fail == 0 {
        println!("\nRESULT: TREE_OK (verify ancestor-pure; accept-walk emits the greedy sequence; KV \
                  compaction moves the accepted path to contiguous slots)");
    } else {
        println!("\nRESULT: TREE_MISMATCH ({total_mism} diverged) — the tree verify is NOT ancestor-pure");
        std::process::exit(1);
    }
}

/// `--bench-lanes` — the batched-verify-across-lanes byte gate (LANES design Step 3a). Packs two
/// independent draft chains (one per lane, each rooted in its own committed slot state) into ONE verify
/// and asserts each lane's per-column logits are bit-equal to running that lane alone. Lanes share
/// committed length here (shared pos_start). Prints RESULT: LANES_OK or LANES_MISMATCH (exit 1).
/// P11 (BATCHED_MTP P1): additionally PACKS p=2..=5 lanes (the production take widths) at the
/// production allocator depths — the 2-lane sweep stays as the regression anchor.
pub(crate) fn run_bench_lanes(args: &[String]) {
    let dir = parse_arg(args, "--model-dir").expect("--bench-lanes requires --model-dir");
    let tok_path = format!("{}/tokenizer.json", dir.trim_end_matches('/'));
    let prompt_text = parse_arg(args, "--prompt").unwrap_or(
        "The quick brown fox jumps over the lazy dog near the river bank at dawn while the sun rises \
         slowly over distant hills and a light wind carries the smell of rain across the wide valley. \
         Later that morning a surveyor arrived with maps and instruments, measured the river's width at \
         three separate fords, noted the reed beds on the near bank, and recorded every observation in a \
         leather notebook whose pages had already begun to curl from the damp. She had walked the valley \
         for eleven days by then, sleeping in barns and occasionally in the open, and had filled most of \
         the notebook with sketches of bridges, fords, mill races, and the old stone weir two miles \
         downstream from the mill, where her grandfather had once worked as a keeper and where, if the \
         deeds could be believed, a right of way still existed across the water meadow to the church. \
         The fox, unbothered by any of this, had long since gone to ground.");
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(8192);
    let tokenizer = QwenTokenizer::from_file(&tok_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");
    let (gpu, _) = load_model_gpu(dir, None, 1);
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    // Slots: 0,1 the two lanes; 2,3 their post-prefill GDN snapshots. 2 KV slots.
    let mut state = gpu.new_batch_state(2, 4, max_seq_len);
    gpu.dev().synchronize().unwrap();

    // Split the prompt into two equal halves -> two lanes with DIFFERENT committed state, EQUAL length.
    let half = prompt.len() / 2;
    assert!(half >= 8, "prompt too short to split into two lanes");
    println!("Forest byte gate (batched verify across 2 lanes): {half} tokens/lane, distinct prefixes");
    let mut total_mism = 0usize;
    let mut total_cols = 0usize;
    // (len_a, len_b): equal AND unequal lane lengths, several straddling the 256 split boundary. Unequal
    // pairs exercise the per-column pos_start (Step 3b) — the attention must split each lane's prefix at
    // ITS OWN committed length, not a shared one.
    for depth in [2usize, 3, 4, 6, 8] {
        for (la, lb) in [(16usize, 16usize), (200, 200), (254, 254), (256, 256),
                         (300, 96), (64, 290), (256, 17), (17, 256)] {
            if la < 4 || lb < 4 || la > half || lb > half { continue; }
            let lane_a = &prompt[..la];
            let lane_b = &prompt[half..half + lb];
            let (cols, mism) = gpu.bench_lanes(&mut pool, &mut state, lane_a, lane_b, max_seq_len, depth);
            total_mism += mism; total_cols += cols;
            let eq = if la == lb { "eq " } else { "NEQ" };
            println!("  depth {depth} ctx a={la:4} b={lb:4} [{eq}]: {cols} cols, {mism} bit-mismatch{}",
                     if mism == 0 { "  OK" } else { "  <-- FAIL" });
        }
    }
    // P11 (BATCHED_MTP P1): the production pack widths. take=5 ships; the pairs sweep above cannot
    // see p=3..5 (AGENTS §3's self-consistency blind spot). Same byte oracle, production allocator
    // (per-lane depth = MAX_VERIFY/p - 1 capped by the policy depth; caps 4 and 8 both exercised —
    // auto re-picks depth every window, so both caps are production-reachable).
    println!("Forest pack gate (production take widths, distinct prefixes per lane):");
    let mut pack_mism = 0usize;
    let mut pack_cols = 0usize;
    for cap in [4usize, 8] {
        for pack in [2usize, 3, 4, 5] {
            let depth = (16usize / pack).saturating_sub(1).clamp(1, cap);
            let per = prompt.len() / pack;
            if per < 8 {
                // Degrade, don't panic: a short custom --prompt can still exercise the widths it
                // fits; the skipped width says so in the log (P1 needs PASS, not a crash).
                println!("  pack p={pack} cap={cap}: SKIPPED (prompt too short — {per} tokens/lane)");
                continue;
            }
            let lanes: Vec<Vec<u32>> =
                (0..pack).map(|i| prompt[i * per..(i + 1) * per].to_vec()).collect();
            let refs: Vec<&[u32]> = lanes.iter().map(|v| v.as_slice()).collect();
            // p lanes + p GDN snapshot slots.
            let mut state = gpu.new_batch_state(pack, 2 * pack, max_seq_len);
            gpu.dev().synchronize().unwrap();
            let (cols, mism) = gpu.bench_lanes_pack(&mut pool, &mut state, &refs, max_seq_len, cap);
            pack_mism += mism;
            pack_cols += cols;
            println!("  pack p={pack} cap={cap} d={depth}: {cols} cols, {mism} bit-mismatch{}",
                     if mism == 0 { "  OK" } else { "  <-- FAIL" });
        }
    }
    if total_mism + pack_mism == 0 {
        println!("\nRESULT: LANES_OK (pairs {total_cols} + packs {pack_cols} lane-columns, packs p=2..5; \
                  every lane's logits bit-identical packed vs alone -- GDN forest scan is lane-independent)");
    } else {
        println!("\nRESULT: LANES_MISMATCH ({total_mism} pair + {pack_mism} pack diverged) -- a lane's \
                  logits depend on its neighbours; the forest verify is NOT lane-pure");
        std::process::exit(1);
    }
}

pub(crate) fn run_bench_accept(args: &[String]) {
    let dir = parse_arg(args, "--model-dir").expect("--bench-accept requires --model-dir");
    let tok_path = format!("{}/tokenizer.json", dir.trim_end_matches('/'));
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("Write a short essay about the sea.");
    let depth: usize = parse_arg(args, "--depth").and_then(|s| s.parse().ok()).unwrap_or(8);
    let max_new: usize = parse_arg(args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(400);
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(8192);
    let label = parse_arg(args, "--label").unwrap_or("workload");
    let ngram: usize = parse_arg(args, "--ngram").and_then(|s| s.parse().ok()).unwrap_or(0);

    let tokenizer = QwenTokenizer::from_file(&tok_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");
    let (gpu, _) = load_model_gpu(dir, None, 1);
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    let mut state = gpu.new_batch_state(1, 2 + depth, max_seq_len);
    gpu.dev().synchronize().unwrap();

    let (s, generated) = match gpu.bench_accept(&mut pool, &mut state, &prompt, max_seq_len, depth, max_new, ngram) {
        Ok(x) => x,
        Err(e) => { eprintln!("bench_accept FAILED (no acceptance number from an aborted run): {e:#}"); std::process::exit(1); }
    };
    assert!(!s.is_empty(), "bench_accept produced NO samples — it measured nothing, which is not a result");
    assert!(generated.len() > 8, "bench_accept generated almost nothing ({} tokens)", generated.len());

    let acc = s.iter().filter(|x| x.accepted).count() as f32 / s.len() as f32;
    println!("\n=== {label}: {} draft positions on the correct prefix, acceptance {:.1}%",
             s.len(), 100.0 * acc);

    // THE DISCRIMINATOR. If the target is CONFIDENT where the head misses -> the head is weak.
    // If the head only misses where the target is itself near-tied -> the text is hard, and no head
    // can fix it.
    println!("\n  acceptance bucketed by the TARGET's own top-1 probability:");
    println!("    target p(top1)     n      accepted   <- if acceptance is high here, the head CAN track it");
    let buckets = [(0.0f32, 0.3f32), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)];
    for (lo, hi) in buckets {
        let b: Vec<_> = s.iter().filter(|x| x.target_top1_p >= lo && x.target_top1_p < hi).collect();
        if b.is_empty() { continue; }
        let a = b.iter().filter(|x| x.accepted).count() as f32 / b.len() as f32;
        let bar = "#".repeat((a * 40.0) as usize);
        println!("    {lo:.2}-{hi:.2}      {:5}     {:5.1}%  {bar}", b.len(), 100.0 * a);
    }

    let confident: Vec<_> = s.iter().filter(|x| x.target_top1_p >= 0.9).collect();
    let uncertain: Vec<_> = s.iter().filter(|x| x.target_top1_p < 0.5).collect();
    let ca = if confident.is_empty() { f32::NAN }
             else { confident.iter().filter(|x| x.accepted).count() as f32 / confident.len() as f32 };
    let ua = if uncertain.is_empty() { f32::NAN }
             else { uncertain.iter().filter(|x| x.accepted).count() as f32 / uncertain.len() as f32 };
    println!("\n  VERDICT INPUTS");
    println!("    where the target is CONFIDENT (p>=0.90): {:5} positions, {:.1}% accepted",
             confident.len(), 100.0 * ca);
    println!("    where the target is UNSURE    (p< 0.50): {:5} positions, {:.1}% accepted",
             uncertain.len(), 100.0 * ua);
    println!("    share of positions where the target is unsure: {:.1}%",
             100.0 * uncertain.len() as f32 / s.len() as f32);
    println!("\n    HARD TEXT  => the target is unsure a lot, and acceptance is high where it IS sure.");
    println!("    WEAK HEAD  => acceptance is poor EVEN where the target is confident.");

    println!("\n  acceptance by draft depth (does the chain decay?):");
    for d in 1..depth {
        let b: Vec<_> = s.iter().filter(|x| x.depth_idx == d).collect();
        if b.is_empty() { continue; }
        let a = b.iter().filter(|x| x.accepted).count() as f32 / b.len() as f32;
        println!("    depth {d}: {:5} positions, {:5.1}% accepted", b.len(), 100.0 * a);
    }

    // ---- FORK COVERAGE: what a top-2/top-3 fork would rescue (Step 0.5 of tree drafting).
    //
    // `accepted` = target argmax ∈ head top-1. `covered_top2/3` = target argmax ∈ head top-2/3. The GAP
    // (covered_top2 − accepted) is the fraction of positions a k=2 fork would newly rescue. The review's
    // yield arithmetic keys the whole payoff on POSITION-1 coverage (c₁), because a wrong position-1
    // guess gates the entire chain — so that row is the one that sets the Step-3 gate.
    println!("\n  FORK COVERAGE — target argmax in the head's top-k (this sets the yield target):");
    println!("    position    n     top1(=accept)   top2       top3      fork rescue (top2−top1)");
    let cov = |v: &[&gb10_inference::gpu::AcceptSample]| {
        let n = v.len().max(1) as f32;
        (v.iter().filter(|x| x.accepted).count() as f32 / n,
         v.iter().filter(|x| x.covered_top2).count() as f32 / n,
         v.iter().filter(|x| x.covered_top3).count() as f32 / n)
    };
    for d in 1..depth {
        let b: Vec<_> = s.iter().filter(|x| x.depth_idx == d).collect();
        if b.is_empty() { continue; }
        let (t1, t2, t3) = cov(&b);
        let mark = if d == 1 { "  <- c₁ (sets the gate)" } else { "" };
        println!("    pos {d:<2}    {:5}    {:5.1}%       {:5.1}%    {:5.1}%      +{:4.1}%{mark}",
                 b.len(), 100.0*t1, 100.0*t2, 100.0*t3, 100.0*(t2-t1));
    }
    let all: Vec<_> = s.iter().collect();
    let (t1, t2, t3) = cov(&all);
    println!("    ALL      {:5}    {:5.1}%       {:5.1}%    {:5.1}%      +{:4.1}%",
             all.len(), 100.0*t1, 100.0*t2, 100.0*t3, 100.0*(t2-t1));
    // Yield ceiling of a k=2 fork at position 1 (review §5.2): 1 + (c₁/p₁)·A, A = mean accepted drafts.
    let p1 = if t1 > 0.0 {
        let pos1: Vec<_> = s.iter().filter(|x| x.depth_idx == 1).collect();
        cov(&pos1).0
    } else { 0.0 };
    let c1 = {
        let pos1: Vec<_> = s.iter().filter(|x| x.depth_idx == 1).collect();
        cov(&pos1).1
    };
    // A = mean accepted drafts per step = (# accepted draft samples) / (# steps). Every step contributes
    // exactly one position-1 sample, so #steps = #(depth_idx == 1). Chain yield Y = 1 + A.
    let n_steps = s.iter().filter(|x| x.depth_idx == 1).count().max(1) as f32;
    let a_drafts = s.iter().filter(|x| x.accepted).count() as f32 / n_steps;
    if p1 > 0.0 {
        let ceiling = 1.0 + (c1 / p1) * a_drafts;
        println!("\n    chain yield Y = 1 + A = 1 + {:.2} = {:.2} tok/fwd (measured here)", a_drafts, 1.0 + a_drafts);
        println!("    fork@1 yield CEILING (perfect top-2 rescue) ≈ 1 + (c₁/p₁)·A = 1 + ({:.2}/{:.2})·{:.2} = {:.2} tok/fwd",
                 c1, p1, a_drafts, ceiling);
        println!("    (review §5.2: realistic is BELOW this — rescues are conditioned on the head having just missed)");
    }

    // ---- Could an N-GRAM LOOKUP have drafted these tokens, for free?
    //
    // The draft head is ONE LAYER. Exact copying is an induction task, which one layer does badly --
    // and structured tool output is mostly copying: tool names, argument keys, JSON scaffolding, all
    // lifted from the prompt or repeated from earlier in the generation. An n-gram matcher copies
    // PERFECTLY and costs zero GPU (it is a host-side string search over tokens we already have).
    //
    // So: replay the generation and ask, at each position, whether the last N tokens appeared earlier
    // in (prompt + generated-so-far), and if so whether the token that FOLLOWED that earlier match is
    // the token the target actually chose. That is exactly what prompt-lookup decoding would propose.
    println!("\n  could a free N-GRAM LOOKUP have drafted these? (prompt-lookup decoding)");
    println!("    ngram   proposals   correct    hit rate   coverage");
    let seq: Vec<u32> = prompt.clone();
    for n in [2usize, 3, 4] {
        let (mut proposals, mut correct) = (0usize, 0usize);
        let mut ctx = seq.clone();
        for w in generated.windows(2) {
            let next = w[1];
            ctx.push(w[0]);
            if ctx.len() <= n { continue; }
            let tail = &ctx[ctx.len() - n..];
            let mut found: Option<u32> = None;
            for i in (0..ctx.len().saturating_sub(n)).rev() {
                if &ctx[i..i + n] == tail {
                    if i + n < ctx.len() { found = Some(ctx[i + n]); }
                    break;
                }
            }
            if let Some(p) = found {
                proposals += 1;
                if p == next { correct += 1; }
            }
        }
        let total = generated.len().saturating_sub(1).max(1);
        println!("    {n}-gram  {proposals:8}   {correct:7}    {:6.1}%    {:6.1}% of positions",
                 if proposals > 0 { 100.0 * correct as f32 / proposals as f32 } else { 0.0 },
                 100.0 * proposals as f32 / total as f32);
    }

    // N-GRAM RUN LENGTH (review §5.4): once a 3-gram match fires, how many CONSECUTIVE tokens does the
    // copy stay correct? A copy fires long. If the median run ≥ 3, the n-gram branch should get MORE
    // depth than the head branch (asymmetric tree: head chain d_a + n-gram chain m, 1 + (d_a−1) + m ≤ 16).
    let mut runs: Vec<usize> = Vec::new();
    {
        let n = 3usize;
        let mut ctx = seq.clone();
        let mut run = 0usize;
        for w in generated.windows(2) {
            let next = w[1];
            ctx.push(w[0]);
            let mut hit = false;
            if ctx.len() > n {
                let tail = &ctx[ctx.len() - n..];
                for i in (0..ctx.len().saturating_sub(n)).rev() {
                    if &ctx[i..i + n] == tail {
                        if i + n < ctx.len() && ctx[i + n] == next { hit = true; }
                        break;
                    }
                }
            }
            if hit { run += 1; } else { if run > 0 { runs.push(run); } run = 0; }
        }
        if run > 0 { runs.push(run); }
    }
    if !runs.is_empty() {
        runs.sort_unstable();
        let median = runs[runs.len() / 2];
        let mean = runs.iter().sum::<usize>() as f32 / runs.len() as f32;
        let max = *runs.last().unwrap();
        let ge3 = runs.iter().filter(|&&r| r >= 3).count();
        println!("\n  3-gram RUN LENGTH (consecutive correct copies): {} runs, median {}, mean {:.1}, max {}",
                 runs.len(), median, mean, max);
        println!("    runs of length ≥3: {}/{} ({:.0}%)  →  {} the n-gram branch deeper than the head branch",
                 ge3, runs.len(), 100.0 * ge3 as f32 / runs.len() as f32,
                 if median >= 3 { "MAKE" } else { "do NOT make" });
    }
}

/// B8/G2 — the τ (acceptance-length) harness: per domain x context bucket, measured against the
/// RUNTIME TARGET (the loaded NVFP4 model) with the live MTP head as the reference drafter.
///
/// This is the DSpark go/no-go instrument (PLAN/08 validation item 1; B8 brief G2). It runs the
/// engine's own bench_mtp per (domain, ctx) cell and reports:
///   tau        = mean tokens committed per verify forward (emitted / verify_fwds) — the number
///                the parity thresholds compare against;
///   alpha      = per-draft acceptance (accepted / offered);
///   tok_s      = MTP wall throughput (the engine's own target measurement);
///   and the parity verdicts: tau > 3.55 (BF16 drafter parity vs MTP g=3), tau > 2.78 (FP8
///   drafter), tau > 1.78 (BF16 break-even vs no-spec), tau > 1.39 (FP8 break-even).
///
/// Domains are prompt-text classes (chat / math / code / agentic); the context bucket is set by
/// padding the prompt with domain-relevant filler to the target token count (the acceptance
/// regime depends on ctx length, which is what the ladder varies). The SAME code path the
/// DSpark session will run for its end-to-end A/B (bench_mtp) produces the numbers.
pub(crate) fn run_bench_tau(args: &[String]) {
    let dir = parse_arg(args, "--model-dir").expect("--bench-tau requires --model-dir <DIR>");
    let domains: Vec<String> = parse_arg(args, "--domains")
        .map(|d| d.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
        .unwrap_or_else(|| vec!["chat".into(), "math".into()]);
    let ctxs: Vec<usize> = parse_arg(args, "--ctxs")
        .map(|c| c.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![4096, 16384]);
    let depth: usize = parse_arg(args, "--depth").and_then(|s| s.parse().ok()).unwrap_or(7);
    let max_new: usize = parse_arg(args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(64);
    let reps: usize = parse_arg(args, "--reps").and_then(|s| s.parse().ok()).unwrap_or(3);
    // KV headroom: the prompt fills the ctx bucket and generation appends max_new + depth draft
    // positions — a stride == ctx OOBs the MTP KV (asserted at mtp_draft_step). Default with
    // headroom; honor an explicit --max-seq-len only if it already has room.
    let needed = ctxs.iter().max().copied().unwrap_or(4096) + max_new + depth + 8;
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok())
        .unwrap_or(needed).max(needed);

    let tokenizer_path = format!("{}/tokenizer.json", dir.trim_end_matches('/'));
    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");

    // Domain seeds: representative openings. Padding repeats a domain-specific filler sentence so
    // the context is domain-typical prose/code/math, not repeated identical tokens.
    let seed_text = |dom: &str| -> String {
        match dom {
            "math" => "Solve step by step. Let x be the unknown. We compute the partial sum, then divide by n.                        The equation reduces after substitution. Verify the units cancel.                        The integral converges absolutely. Round to four decimals. ".to_string(),
            "code" => "fn process(input: &[u8]) -> Vec<u8> { let mut out = Vec::new(); for b in input {                        if b.is_ascii_graphic() { out.push(*b); } } out } // review: allocation-free variant? ".to_string(),
            "agentic" => "You are an autonomous agent. Observe the tool result, update the plan, and act.                           Step 3 failed; retry with a narrower query. Log the state transition. ".to_string(),
            _ => "The quick history of computing: machines grew smaller as the years went by, and software                   followed. People talked about it at length, in parlors and later on forums. ".to_string(),
        }
    };

    println!("B8/G2 tau harness: model={dir} depth={depth} reps={reps} domains={:?} ctxs={:?}", domains, ctxs);
    println!("parity thresholds (PLAN/08): tau>3.55 (DSpark-BF16 ~ MTP g=3) | >2.78 (FP8 drafter) | >1.78 (BF16 vs no-spec) | >1.39 (FP8 vs no-spec)");
    let (gpu, _) = load_model_gpu(&dir, None, 1);
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    let n_slots = 2 + depth.saturating_sub(1).max(1);

    println!("\n| domain | ctx | tau (best-of-reps) | alpha | MTP tok/s | SEQ tok/s | BF16-parity | FP8-parity | no-spec BE |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for dom in &domains {
        for &ctx in ctxs.iter().filter(|c| **c <= max_seq_len).collect::<Vec<&usize>>() {
            // Build a domain prompt of ~ctx tokens by repeating the seed to the bucket size.
            // VARIED filler: verbatim repetition makes the greedy continuation pattern-lock and
            // the tau estimate degenerate (measured 6.4 vs 1.07 tokens/step on the SAME prompt).
            // Each block gets a numbered heading so the text never repeats exactly.
            let seed = seed_text(dom);
            let mut text = String::new();
            let mut i = 0usize;
            while tokenizer.encode(&text, true).map(|t| t.len()).unwrap_or(0) < ctx + 64 {
                text.push_str(&format!("({i}) {seed}"));
                i += 1;
            }
            let base = tokenizer.encode(&text, true).expect("encode");
            let mut prompt: Vec<u32> = Vec::with_capacity(ctx);
            prompt.extend_from_slice(&base);
            prompt.truncate(ctx);
            let mut best_tau = 0.0f32; let mut a5 = 0.0f32; let mut m5 = 0.0f32; let mut s5 = 0.0f32;
            for _r in 0..reps {
                let mut state = gpu.new_batch_state(n_slots, n_slots, max_seq_len);
                // COUNTED tau (assert-the-signal rule): bench_mtp_steps returns the loop's verify-forward
                // count; tau = emitted tokens / verify forwards. Never a closed-form approximation.
                let (mt, _st, mtok, stok, acc, n_steps, _acc_k, _dr_k) =
                    gpu.bench_mtp_steps(&mut pool, &mut state, &prompt, max_seq_len, depth, max_new);
                assert!(n_steps > 0, "bench-tau: zero steps measured — harness failure, not a pass");
                let tau = mt.len() as f32 / n_steps as f32;
                if tau > best_tau { best_tau = tau; }
                a5 = acc; m5 = mtok; s5 = stok;
            }
            // ASSERT THE SUCCESS SIGNAL (AGENTS §3): a cell that ran every rep prints a green line;
            // the runner greps for it. Absence of errors is not a pass.
            println!("CELL OK ({reps}/{reps} reps): domain={dom} ctx={ctx} tau={best_tau:.3} alpha={a5:.3}");
            let bf16_p = if best_tau > 3.55 { "CLEAR" } else { "below" };
            let fp8_p  = if best_tau > 2.78 { "CLEAR" } else { "below" };
            let be     = if best_tau > 1.78 { "pays" } else { "LOSES" };
            println!("| {dom} | {ctx} | {best_tau:.3} | {:.3} | {m5:.1} | {s5:.1} | {bf16_p} | {fp8_p} | {be} |", a5);
        }
    }
    println!("\nNOTE: tau here is MTP-head tau on the NVFP4 runtime target — the BASELINE the DSpark drafter must beat. DSpark-vs-MTP parity uses the same harness with --drafter dspark once K-DSP lands.");
}
