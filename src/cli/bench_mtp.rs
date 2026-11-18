//! MTP-related benches and probes: verify == sequential decode, MTP profiling, sampled and file-driven
//! MTP benches, and the MTP calibration cache helpers (`mtp_calib_*`).
//! MTP semantics (single-slot head KV, pre-final-norm hidden, depth policy) are those of AGENTS.md section 2.

use super::*;

pub(crate) fn run_bench_verify(args: &[String]) {
    let (model_path, tokenizer_path) = if let Some(dir) = parse_arg(args, "--model-dir") {
        (dir.to_string(), format!("{}/tokenizer.json", dir.trim_end_matches('/')))
    } else {
        (parse_arg(args, "--model").unwrap_or("model/model.safetensors").to_string(),
         parse_arg(args, "--tokenizer").unwrap_or("model/tokenizer.json").to_string())
    };
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("The capital of France is");
    let depth: usize = parse_arg(args, "--depth").and_then(|s| s.parse().ok()).unwrap_or(4);
    let offset: usize = parse_arg(args, "--offset").and_then(|s| s.parse().ok()).unwrap_or(0);
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);

    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");
    println!("MTP verify lossless probe: prompt={} tokens, offset={}, depth={}", prompt.len(), offset, depth);

    let gpu = if std::path::Path::new(&model_path).is_dir() {
        let (gpu, _) = load_model_gpu(&model_path, None, 1);
        gpu
    } else {
        let host = gb10_inference::qwen::Model::load(&model_path).expect("load model");
        gb10_inference::gpu::GpuModel::new(&host).expect("gpu init")
    };
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    // batch=2: slot 0 = sequential ground truth, slot 1 = verify_forward.
    let mut state = gpu.new_batch_state(2, 2, max_seq_len);
    // kv_stride MUST match the state's allocation stride (max_seq_len), not cfg.max_position_embeddings.
    let kv_stride = max_seq_len;

    // `--draws N`: run N RANDOMIZED (ctx, offset, depth) draws in ONE process.
    //
    // A gate must be a loop, not a single run — the split-K bug passed most runs, because a 1-ulp
    // difference rarely flips an argmax (AGENTS.md §4.13). But one process per draw meant reloading a
    // 6 GB artifact 13 times: ~11 s of model load to run ~1 s of gate. Loading once turns a 3-minute
    // gate into a 20-second one, which is the difference between a gate you run and a gate you skip.
    //
    // Offsets straddle the 256-key split-K boundary on purpose: that is exactly where the shipped bug
    // lived, and no fixed context ever reached it.
    // `--time-widths 1,2,4,8,16 [--reps R]`: PLAN/25 Phase 0 — the verify-width cost curve
    // (bench-only): per width, the median of ONE verify_forward(W columns) vs the per-token
    // median of sequential decode, alternating within each rep on fresh identical prefill
    // states. This is the flat-in-N extension to W=16 (= MAX_VERIFY, the engine's hard verify
    // ceiling: wider silently falls to the prefill dequant path — verify_forward_core asserts).
    let tw = parse_arg(args, "--time-widths");
    if let Some(tw) = tw {
        let widths: Vec<usize> = tw.split(',').map(|s| s.trim().parse()
            .expect("--time-widths: comma-separated integers")).collect();
        assert!(!widths.is_empty() && widths.iter().all(|&w| w >= 1 && w <= 16),
                "--time-widths: widths must be in 1..=16 (MAX_VERIFY — wider is not a chain verify)");
        let reps: usize = parse_arg(args, "--reps").and_then(|s| s.parse().ok()).unwrap_or(7);
        let max_seq_len_tw = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);
        let prompt_tw = tokenizer.encode(
            parse_arg(args, "--prompt").unwrap_or("The capital of France is"), true)
            .expect("encode");
        let (gpu_tw, _) = load_model_gpu(&model_path, None, 1);
        let mut pool_tw = gb10_inference::gpu::Pool::new(gpu_tw.dev().clone());
        let maxw = *widths.iter().max().unwrap();
        let mut state_tw = gpu_tw.new_batch_state(2, 2 + maxw, max_seq_len_tw);
        println!("verify-width cost curve: widths={widths:?} reps={reps} plen={} seq={max_seq_len_tw}",
                 prompt_tw.len());
        let rows = gpu_tw.time_verify_widths(&mut pool_tw, &mut state_tw, &prompt_tw,
                                             max_seq_len_tw, &widths, reps)
            .expect("time_verify_widths");
        let dec0 = rows.iter().find(|&&(w, _, _)| w == 1).map(|&(_, d, _)| d);
        println!("  width  verify_ms  decode_ms/tok  verify/decode  verify/verify1");
        for (i, &(w, d, v)) in rows.iter().enumerate() {
            let v1 = rows[0].2;
            let _ = i;
            println!("  {:>5}  {:9.3}  {:13.3}  {:13.3}x  {:13.3}x",
                     w, v, d, v / d, v / v1);
        }
        if let Some(d1) = dec0 {
            println!("  decode baseline (W=1 row's decode_ms/tok): {d1:.3} ms");
        }
        println!("RESULT: VERIFY_WIDTH_CURVE_OK ({} widths x {reps} reps, alternating legs)", rows.len());
        return;
    }

    let draws: usize = parse_arg(args, "--draws").and_then(|s| s.parse().ok()).unwrap_or(0);
    if draws > 0 {
        let seed: u64 = parse_arg(args, "--seed").and_then(|s| s.parse().ok()).unwrap_or(0x9E3779B9);
        let mut rng = seed;
        let mut next = |n: u64| { // xorshift; deterministic per seed so a failure is reproducible
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; (rng % n) as usize
        };
        const DEPTHS: [usize; 5] = [2, 3, 4, 6, 8];
        const OFFSETS: [usize; 9] = [0, 250, 254, 255, 256, 510, 511, 1022, 1023];
        let mut failed = 0usize;
        for i in 0..draws {
            let take = 64 + next((prompt.len().saturating_sub(64).max(1)) as u64);
            let d = DEPTHS[next(DEPTHS.len() as u64)];
            let off = OFFSETS[next(OFFSETS.len() as u64)];
            let p = &prompt[..take.min(prompt.len())];
            // offset must leave room for the verify block inside the prompt
            let off = if off + d + 2 >= p.len() { 0 } else { off };
            let (_, _, m) = if off == 0 {
                gpu.bench_verify(&mut pool, &mut state, p, kv_stride, d)
            } else {
                gpu.bench_verify_at_offset(&mut pool, &mut state, p, kv_stride, off, d)
            };
            let ok = m.iter().all(|&b| b);
            if !ok { failed += 1; }
            println!("  draw {:2}/{}  ctx={:<5} offset={:<5} depth={}  {}",
                     i + 1, draws, p.len(), off, d,
                     if ok { "LOSSLESS_OK" } else { "MISMATCH" });
            if !ok { break; }   // SPRT: one failure is enough
        }
        if failed > 0 {
            println!("RESULT: MISMATCH ({} draw(s) diverged) — MTP verify is NOT lossless", failed);
            std::process::exit(1);
        }
        println!("RESULT: LOSSLESS_OK ({} randomized draws, seed {})", draws, seed);
        return;
    }

    let (seq_tokens, preds, matches) = if offset == 0 {
        gpu.bench_verify(&mut pool, &mut state, &prompt, kv_stride, depth)
    } else {
        gpu.bench_verify_at_offset(&mut pool, &mut state, &prompt, kv_stride, offset, depth)
    };

    let seq_text = tokenizer.decode(&seq_tokens, true).unwrap_or_default();
    let all_ok = matches.iter().all(|&b| b);
    println!("ground-truth (seq decode slot 0): {:?}", &seq_tokens);
    println!("verify preds   (verify_forward  slot 1): {:?}", &preds);
    println!("expected preds (gt shifted by 1): {:?}", &seq_tokens[1..=depth]);
    println!("per-position match: {:?}", matches);
    println!("ground-truth text: {:?}", seq_text);
    if all_ok {
        println!("RESULT: LOSSLESS_OK (verify_forward == sequential greedy for all {} positions)", depth);
    } else {
        let nmismatch = matches.iter().filter(|&&b| !b).count();
        println!("RESULT: MISMATCH ({} of {} positions diverged) — MTP verify is NOT lossless", nmismatch, depth);
        std::process::exit(1);
    }
}

/// `--profile-mtp`: attribute the cost of one stochastic-MTP step.
pub(crate) fn run_profile_mtp(args: &[String]) {
    let (model_path, tokenizer_path) = if let Some(dir) = parse_arg(args, "--model-dir") {
        (dir.to_string(), format!("{}/tokenizer.json", dir.trim_end_matches('/')))
    } else {
        (parse_arg(args, "--model").unwrap_or("model/model.safetensors").to_string(),
         parse_arg(args, "--tokenizer").unwrap_or("model/tokenizer.json").to_string())
    };
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("The capital of France is");
    let iters: usize = parse_arg(args, "--iters").and_then(|s| s.parse().ok()).unwrap_or(20);
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);

    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");
    let gpu = if std::path::Path::new(&model_path).is_dir() {
        let (gpu, _) = load_model_gpu(&model_path, None, 1);
        gpu
    } else {
        let host = gb10_inference::qwen::Model::load(&model_path).expect("load model");
        gb10_inference::gpu::GpuModel::new(&host).expect("gpu init")
    };
    if !gpu.mtp_present() { println!("No MTP head."); std::process::exit(1); }
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    let mut state = gpu.new_batch_state(2 + gb10_inference::gpu::PROFILE_MAX_N, 2 + gb10_inference::gpu::PROFILE_MAX_N, max_seq_len);

    let rows = gpu.profile_mtp(&mut pool, &mut state, &prompt, max_seq_len, iters);
    let base = rows.iter().find(|r| r.0.starts_with("decode step")).map(|r| r.1).unwrap_or(1.0);
    println!("\nPer-phase cost of one MTP step ({} iters), vs a plain decode step:\n", iters);
    for (name, ms) in &rows {
        println!("  {:<44} {:8.2} ms   {:5.2}x decode", name, ms, ms / base);
    }

    // A depth-2 stochastic step = draft(MTP fwd + LM-head argmax) + verify_sample + rollback +
    // re-prime(<=2 MTP fwd). Sum it and compare against what its ~1.76 tokens should have cost.
    let get = |k: &str| rows.iter().find(|r| r.0.trim().starts_with(k)).map(|r| r.1).unwrap_or(0.0);
    let (draft, argmax, pen, vsample, roll, reprime) =
        (get("mtp_draft_step"), get("argmax_hidden"), get("penalty upload"),
         get("verify_forward_sample"), get("copy_gdn_slot"), get("mtp_reprime"));
    let step = draft + argmax + pen + vsample + roll + reprime;
    println!("\n  modelled step = draft+argmax+penalty+verify_sample+rollback+reprime(batched)");
    println!("  {:<44} {:8.2} ms   {:5.2}x decode", "modelled MTP step", step, step / base);
    println!("  {:<44} {:8.2} ms", "  ...of which verify_forward_core", get("verify_forward_core"));
    println!("\n  At 1.76 tok/step, break-even needs step < 1.76x decode.");
    println!("  Projected speedup at this step cost: {:.2}x", 1.76 / (step / base));
}

/// `--bench-mtp-sample` — the distribution-exactness gate for stochastic MTP.
///
/// Greedy MTP is bitwise-lossless and `--bench-mtp` proves it by direct comparison. Stochastic MTP is
/// only *distribution*-exact, so it is gated statistically instead: hold the prefix fixed and draw
/// many emissions through the real kernels, then compare the emitted-token histogram against the
/// distribution the plain sampler is defined to produce. The plain sampler's own histogram is drawn
/// alongside as a control — it fixes the sampling-noise floor at this trial count, which is the bar
/// the MTP path has to reach (being merely "small" is not enough).
pub(crate) fn run_bench_mtp_sample(args: &[String]) {
    let (model_path, tokenizer_path) = if let Some(dir) = parse_arg(args, "--model-dir") {
        (dir.to_string(), format!("{}/tokenizer.json", dir.trim_end_matches('/')))
    } else {
        (parse_arg(args, "--model").unwrap_or("model/model.safetensors").to_string(),
         parse_arg(args, "--tokenizer").unwrap_or("model/tokenizer.json").to_string())
    };
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("The capital of France is");
    let trials: usize = parse_arg(args, "--trials").and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let top_k: usize = parse_arg(args, "--top-k").and_then(|s| s.parse().ok()).unwrap_or(20);
    let top_p: f32 = parse_arg(args, "--top-p").and_then(|s| s.parse().ok()).unwrap_or(0.8);
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);
    let temps: Vec<f32> = match parse_arg(args, "--temp") {
        Some(s) => vec![s.parse().expect("--temp")],
        None => vec![0.3, 0.7, 1.0],
    };

    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");

    let gpu = if std::path::Path::new(&model_path).is_dir() {
        let (gpu, _) = load_model_gpu(&model_path, None, 1);
        gpu
    } else {
        let host = gb10_inference::qwen::Model::load(&model_path).expect("load model");
        gb10_inference::gpu::GpuModel::new(&host).expect("gpu init")
    };
    if !gpu.mtp_present() {
        println!("No MTP head — cannot run the stochastic-MTP gate.");
        std::process::exit(1);
    }
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    let mut state = gpu.new_batch_state(3, 3, max_seq_len);

    println!("Stochastic-MTP distribution gate: prompt={} tokens, trials={}, top_k={}, top_p={}",
             prompt.len(), trials, top_k, top_p);
    println!();

    // The gate is a TWO-SAMPLE test of stochastic MTP against the plain sampler, at |z| < 4.
    //
    // Two things this deliberately does NOT do. It does not use an absolute TVD bar: the TVD noise
    // floor grows with the nucleus size and shrinks with the trial count, so on a 47-token nucleus at
    // 100k trials even a flawless sampler sits near 0.009 and would fail a "TVD < 0.01" rule. And it
    // does not gate against a host-computed analytic nucleus: the top-p cut is float-sensitive deep in
    // the tail, so the host and the kernels disagree on the cutoff by a token or two and BOTH paths
    // "fail" the analytic reference in lockstep. The claim we need is reference-free — MTP emits from
    // the same law as the plain sampler — so we test the two empirical histograms against each other.
    // The vs-analytic numbers are still printed, as diagnostics.
    const ZBAR: f32 = 4.0;
    let mut all_pass = true;
    for (ti, &temp) in temps.iter().enumerate() {
        // Independent RNG stream per temperature, so a single unlucky draw cannot masquerade as a
        // systematic bias across rows.
        let base = 0xA5A5_1234_0000_0000u64 ^ ((ti as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let s = gpu.bench_mtp_sample(&mut pool, &mut state, &prompt, max_seq_len,
                                     temp, top_k, top_p, trials, base);
        let p_agree = (s.p_draft_analytic - s.p_draft_device).abs();
        let pass = s.mtp_vs_sampler.z.abs() < ZBAR       // MTP draws from the sampler's law
            && s.bonus_vs_sampler.z.abs() < ZBAR         // so does the all-accepted bonus column
            && s.accept_z.abs() < ZBAR                   // accept rate equals p(x_draft)
            && p_agree < 1e-3;                           // device and host agree on p(x_draft)
        all_pass &= pass;

        println!("temp={:.2}  draft={} nucleus={} p(draft)={:.4} (device {:.4}, Δ={:.1e})",
                 temp, s.x_draft, s.nucleus_size, s.p_draft_analytic, s.p_draft_device, p_agree);
        println!("   accept rate   {:.4}  vs p(draft) {:.4}   z={:+.2}",
                 s.accept_rate, s.p_draft_analytic, s.accept_z);
        println!("   GATE  MTP   vs sampler : z={:+.2}  chi2/df={:.3}  TVD={:.5}  bins={}",
                 s.mtp_vs_sampler.z, s.mtp_vs_sampler.chi2_over_df,
                 s.mtp_vs_sampler.tvd, s.mtp_vs_sampler.bins);
        println!("   GATE  bonus vs sampler : z={:+.2}  chi2/df={:.3}  TVD={:.5}  bins={}",
                 s.bonus_vs_sampler.z, s.bonus_vs_sampler.chi2_over_df,
                 s.bonus_vs_sampler.tvd, s.bonus_vs_sampler.bins);
        println!("   (diag vs analytic p)   : sampler z={:+.2}  MTP z={:+.2}  bonus z={:+.2}",
                 s.sampler.z, s.mtp.z, s.bonus.z);
        println!("   [{} draft trials, {} bonus trials]   => {}",
                 s.trials, s.bonus_trials, if pass { "PASS" } else { "FAIL" });
        println!();
    }

    if all_pass {
        println!("RESULT: DISTRIBUTION_OK (stochastic MTP is distribution-exact vs the plain sampler)");
    } else {
        println!("RESULT: DISTRIBUTION_MISMATCH");
        std::process::exit(1);
    }
}

pub(crate) fn run_bench_mtp(args: &[String]) {
    if parse_arg(args, "--bench-file").is_some() { run_bench_mtp_file(args); return; }
    // S-A3-f-e: an EXL3 pack cannot go through the generic GpuModel loader (the
    // container is EXL3-native; the generic load panicked at "Shard 1/9"). Run the
    // gate on the EXL3 engine instead — same LOSSLESS_OK contract, exl3 semantics.
    if let Some(dir) = parse_arg(args, "--model-dir") {
        if gb10_inference::exl3_serve::is_exl3_pack(dir) {
            let depth: usize = parse_arg(args, "--depth").or_else(|| parse_arg(args, "--mtp-depth"))
                .and_then(|s| s.parse().ok()).unwrap_or(2);
            let a = gb10_inference::exl3_forward::FwdArgs {
                dir: dir.to_string(),
                prompt: parse_arg(args, "--prompt").map(|s| s.to_string()),
                ids_file: parse_arg(args, "--prompt-ids").map(|s| s.to_string()),
                max_new: parse_arg(args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(24),
                width: 1,
                ladder_dir: None,
                ladder_layers: Vec::new(),
                ladder_step: None,
                max_pos: gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(8192),
                bench_steps: 0,
                widths: Vec::new(),
                prefill_chunk: parse_arg(&args, "--prefill-chunk").and_then(|s| s.parse().ok()),
            };
            gb10_inference::exl3_forward::bench_mtp_exl3(&a, depth).unwrap_or_else(|e| {
                eprintln!("BENCH_MTP_EXL3_FAIL: {e:#}");
                std::process::exit(1);
            });
            return;
        }
    }
    let (model_path, tokenizer_path) = if let Some(dir) = parse_arg(args, "--model-dir") {
        (dir.to_string(), format!("{}/tokenizer.json", dir.trim_end_matches('/')))
    } else {
        (parse_arg(args, "--model").unwrap_or("model/model.safetensors").to_string(),
         parse_arg(args, "--tokenizer").unwrap_or("model/tokenizer.json").to_string())
    };
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("The capital of France is");
    let depth: usize = parse_arg(args, "--depth").and_then(|s| s.parse().ok()).unwrap_or(4);
    let max_new: usize = parse_arg(args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(64);
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);

    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");
    // KV headroom floor (B8 blocker B): the MTP draft/verify/re-prime step writes up to
    // plen + max_new + depth + 8. The bench paths admit requests directly (no server gate), so an
    // explicit --max-seq-len that lacks the headroom would OOB — floor it like run_bench_tau does.
    let max_seq_len = max_seq_len.max(prompt.len() + max_new + depth + 8);
    println!("MTP end-to-end probe: prompt={} tokens, depth={}, max_new={}", prompt.len(), depth, max_new);

    let gpu = if std::path::Path::new(&model_path).is_dir() {
        let (gpu, _) = load_model_gpu(&model_path, None, 1);
        gpu
    } else {
        let host = gb10_inference::qwen::Model::load(&model_path).expect("load model");
        gb10_inference::gpu::GpuModel::new(&host).expect("gpu init")
    };
    if gpu.mtp_present() {
        println!("MTP head loaded.");
    } else {
        println!("No MTP head — cannot run MTP probe.");
        std::process::exit(1);
    }
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    // slot 0 = MTP lane, slot 1 = sequential ground truth, slots 2.. = one GDN checkpoint per verify
    // column we might roll back to (nacc ranges 0..depth-2, so depth-1 of them).
    let mut state = gpu.new_batch_state(2 + depth.saturating_sub(1).max(1), 2 + depth.saturating_sub(1).max(1), max_seq_len);

    let (mtp_tokens, seq_tokens, mtp_tok_s, seq_tok_s, accept_rate) =
        gpu.bench_mtp(&mut pool, &mut state, &prompt, max_seq_len, depth, max_new);

    let lossless = mtp_tokens == seq_tokens;
    let mtp_text = tokenizer.decode(&mtp_tokens, true).unwrap_or_default();
    println!("MTP tokens : {:?}", &mtp_tokens[..mtp_tokens.len().min(20)]);
    println!("SEQ tokens : {:?}", &seq_tokens[..seq_tokens.len().min(20)]);
    println!("MTP output : {:?}", mtp_text);
    println!("acceptance rate: {:.1}%  (drafts accepted / total drafts)", accept_rate * 100.0);
    println!("throughput: MTP {:.1} tok/s   sequential {:.1} tok/s   speedup {:.2}x",
             mtp_tok_s, seq_tok_s, mtp_tok_s / seq_tok_s.max(1e-6));
    if lossless {
        println!("RESULT: LOSSLESS_OK (MTP output == sequential greedy, {} tokens)", mtp_tokens.len());
    } else {
        // Find first divergence for diagnostics; print a window of context around it.
        let div = mtp_tokens.iter().zip(seq_tokens.iter()).position(|(a, b)| a != b);
        println!("RESULT: MISMATCH at token {:?} — MTP is NOT lossless", div);
        if let Some(d) = div {
            let lo = d.saturating_sub(6);
            let hi = (d + 6).min(mtp_tokens.len()).min(seq_tokens.len());
            println!("  ctx MTP[{}..{}]: {:?}", lo, hi, &mtp_tokens[lo..hi]);
            println!("  ctx SEQ[{}..{}]: {:?}", lo, hi, &seq_tokens[lo..hi]);
        }
        std::process::exit(1);
    }
}

/// S-B4 (MTP truth): multi-prompt acceptance battery. Each JSONL line of `--bench-file <path>`:
///   {"text": "...", "regime": "prose", "depth": 2, "max_new": 64}
/// Default renders `text` through the model's OWN chat template with thinking OFF and encodes the
/// result WITHOUT add_special_tokens — the exact serving prompt (`--no-chat` for raw completions,
/// `--thinking-on` to prime reasoning). One `bench_mtp_steps` run per line: the MTP draft chain
/// PLUS the interleaved greedy ground truth, so every line carries its own losslessness gate —
/// a MISMATCH is a finding (chain/re-prime corruption), never retried away. Aggregates per
/// (regime, depth) with per-position accept@k (accepted_at_k[k]/drafts_at_k[k]).
fn run_bench_mtp_file(args: &[String]) {
    use gb10_inference::tokenizer::ThinkingMode;
    let file = parse_arg(args, "--bench-file").unwrap().to_string();
    // B30: read the battery BEFORE the multi-minute model load and NAME the path on failure.
    // A missing battery file used to surface only after the full load as a bare os-error-2 —
    // identical for an existing and a nonexistent file — which made the flag undiagnosable.
    let raw = std::fs::read_to_string(&file).unwrap_or_else(|e| {
        eprintln!("[bench-file] ERROR: cannot read battery file '{file}': {e}");
        std::process::exit(1);
    });
    let model_path = parse_arg(args, "--model-dir").expect("--bench-file needs --model-dir").to_string();
    let max_new_default: usize = parse_arg(args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(64);
    let chat = !args.iter().any(|a| a == "--no-chat");
    let thinking = if args.iter().any(|a| a == "--thinking-on") { ThinkingMode::On } else { ThinkingMode::Off };
    let tokenizer = QwenTokenizer::from_file(
        &format!("{}/tokenizer.json", model_path.trim_end_matches('/'))).expect("tokenizer");
    println!("[bench-file] {} chat={} thinking={:?}", file, chat, thinking);
    let gpu = { let (gpu, _) = load_model_gpu(&model_path, None, 1); gpu };
    if !gpu.mtp_present() { println!("No MTP head — cannot run MTP bench."); std::process::exit(1); }
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());

    #[derive(serde::Deserialize)]
    struct Ent { text: String, regime: String, depth: usize, max_new: Option<usize> }
    let ents: Vec<Ent> = raw.lines().filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("bad jsonl line in --bench-file")).collect();

    struct Agg { steps: usize, drafts: usize, accepted: usize, mt_n: usize,
                 mtp_tok_s: f32, seq_tok_s: f32, lossless: usize, runs: usize,
                 acc_k: Vec<usize>, dr_k: Vec<usize> }
    let mut agg: std::collections::HashMap<(String, usize), Agg> = Default::default();

    for (i, e) in ents.iter().enumerate() {
        let ptoks = if chat {
            let msgs = vec![gb10_inference::tokenizer::ChatMessage {
                role: "user".to_string(), content: Some(e.text.clone()), images: vec![], layout: vec![], unsupported_parts: vec![],
                tool_calls: None, tool_call_id: None, name: None, reasoning_content: None,
            }];
            let s = tokenizer.apply_chat_template(&msgs, None, None, None, thinking)
                .expect("chat template render");
            tokenizer.encode(&s, false).expect("encode chat prompt")
        } else {
            tokenizer.encode(&e.text, true).expect("encode prompt")
        };
        let depth = e.depth.max(1);
        let max_new = e.max_new.unwrap_or(max_new_default);
        let kv_stride = (ptoks.len() + max_new + depth + 8).max(512);
        let n_slots = 2 + depth.saturating_sub(1).max(1);
        let mut state = gpu.new_batch_state(n_slots, n_slots, kv_stride);
        let (mt, st, mtok, stok, acc, n_steps, acc_k, dr_k) =
            gpu.bench_mtp_steps(&mut pool, &mut state, &ptoks, kv_stride, depth, max_new);
        let lossless = mt == st;
        let ak: Vec<String> = acc_k.iter().zip(dr_k.iter()).enumerate()
            .map(|(i, (a, d))| format!("k{}={:.1}%", i + 1, if *d > 0 { *a as f32 / *d as f32 * 100.0 } else { 0.0 }))
            .collect();
        println!("[bench-file] {:2}/{:2} regime={:9} depth={} plen={:5} accept {:5.1}% accept@k [{}] tok/step {:.2} mtp {:5.1} seq {:5.1} tok/s {}",
            i + 1, ents.len(), e.regime, depth, ptoks.len(), acc * 100.0, ak.join(" "),
            if n_steps > 0 { mt.len() as f32 / n_steps as f32 } else { 0.0 }, mtok, stok,
            if lossless { "LOSSLESS_OK" } else { "MISMATCH" });
        let v = agg.entry((e.regime.clone(), depth)).or_insert(Agg {
            steps: 0, drafts: 0, accepted: 0, mt_n: 0, mtp_tok_s: 0.0, seq_tok_s: 0.0,
            lossless: 0, runs: 0, acc_k: vec![0; depth.saturating_sub(1)], dr_k: vec![0; depth.saturating_sub(1)] });
        v.steps += n_steps; v.drafts += dr_k.iter().sum::<usize>(); v.accepted += acc_k.iter().sum::<usize>();
        v.mt_n += mt.len(); v.mtp_tok_s += mtok; v.seq_tok_s += stok;
        if lossless { v.lossless += 1; } v.runs += 1;
        for (x, a) in v.acc_k.iter_mut().enumerate() { *a += acc_k.get(x).copied().unwrap_or(0); }
        for (x, d) in v.dr_k.iter_mut().enumerate() { *d += dr_k.get(x).copied().unwrap_or(0); }
    }

    println!("[bench-file] SUMMARY — accept@k = accepted_at_k[k]/drafts_at_k[k] (per-position); accept = all positions pooled:");
    let mut keys: Vec<_> = agg.keys().cloned().collect();
    keys.sort();
    for (regime, depth) in keys {
        let v = &agg[&(regime.clone(), depth)];
        let ak: Vec<String> = v.acc_k.iter().zip(v.dr_k.iter()).enumerate()
            .map(|(i, (a, d))| format!("k{}={:.1}%", i + 1, if *d > 0 { *a as f32 / *d as f32 * 100.0 } else { 0.0 }))
            .collect();
        let rate = if v.drafts > 0 { v.accepted as f32 / v.drafts as f32 * 100.0 } else { 0.0 };
        let tps = if v.steps > 0 { v.mt_n as f32 / v.steps as f32 } else { 0.0 };
        println!("[bench-file]   {:9} d={} runs={} lossless={}/{} accept {:5.1}% [{}] tok/step {:.2} mtp {:5.1} seq {:5.1} tok/s (means)",
            regime, depth, v.runs, v.lossless, v.runs, rate, ak.join(" "), tps,
            v.mtp_tok_s / v.runs as f32, v.seq_tok_s / v.runs as f32);
    }
}

pub(super) fn mtp_calib_cache_path(model_path: &str) -> Option<std::path::PathBuf> {
    // <binary_dir>/mtp_calib/<model-basename>.json  — a subdir next to the running executable.
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?.join("mtp_calib");
    let base = std::path::Path::new(model_path.trim_end_matches('/'))
        .file_name()?.to_string_lossy().replace(['/', ' '], "_");
    // --draft-vocab changes r(d) (the draft chain reads a smaller head): an explicit setting gets its
    // own file, so switching it never thrashes one cache; the unflagged default keeps the plain name.
    let dv = mtp_calib_draft_vocab_key().map(|k| format!(".dv-{}", k.replace(['/', ' ', '%', '.'], "_"))).unwrap_or_default();
    Some(dir.join(format!("{base}{dv}.json")))
}

/// The --draft-vocab / --draft-vocab-file setting that shapes the draft chain's cost, when one was
/// GIVEN (None = the model's default, which is per family: quant::DraftVocab::default_for). An explicit
/// `off` is a key of its own: the Qwen default is on, so off must never share the default's file.
fn mtp_calib_draft_vocab_key() -> Option<String> {
    let dv = gb10_inference::opts::var(gb10_inference::opt!("draft-vocab")).ok();
    let file = gb10_inference::opts::var(gb10_inference::opt!("draft-vocab-file")).ok();
    match (dv, file) {
        (None, None) => None,
        (d, f) => Some(format!("{}{}", d.unwrap_or_default(), f.map(|f| format!("+file:{f}")).unwrap_or_default())),
    }
}

pub(super) fn mtp_calib_stamp(model_path: &str) -> String {
    // Invalidate on a rebuild (binary mtime changes = new kernels), a different model, or a different
    // --draft-vocab (a calibration taken with the full-vocab drafter overprices a reduced one).
    let bin_mtime = std::env::current_exe().ok()
        .and_then(|e| std::fs::metadata(e).ok())
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs()).unwrap_or(0);
    let dv = mtp_calib_draft_vocab_key().map(|k| format!(";draft_vocab={k}")).unwrap_or_default();
    format!("bin_mtime={bin_mtime};model={}{dv}", model_path.trim_end_matches('/'))
}

pub(super) fn read_mtp_calib(path: &Option<std::path::PathBuf>, stamp: &str) -> Option<Vec<(usize, Vec<(usize, f32)>)>> {
    let txt = std::fs::read_to_string(path.as_ref()?).ok()?;
    let v: serde_json::Value = serde_json::from_str(&txt).ok()?;
    if v.get("stamp")?.as_str()? != stamp { return None; }
    // fmt 2 (E17): r(d) per context bucket. fmt-1 caches (one global table) are ignored — a
    // short-context table is exactly the mis-calibration E17 fixes, so reuse would re-import it.
    if v.get("fmt")?.as_u64()? != 2 { return None; }
    let mut out: Vec<(usize, Vec<(usize, f32)>)> = Vec::new();
    for b in v.get("buckets")?.as_array()? {
        let ctx = b.get(0)?.as_u64()? as usize;
        let table: Vec<(usize, f32)> = b.get(1)?.as_array()?.iter()
            .filter_map(|e| Some((e.get(0)?.as_u64()? as usize, e.get(1)?.as_f64()? as f32)))
            .collect();
        out.push((ctx, table));
    }
    if out.is_empty() { None } else { Some(out) }
}

pub(super) fn write_mtp_calib(path: &Option<std::path::PathBuf>, stamp: &str, buckets: &[(usize, Vec<(usize, f32)>)]) {
    let Some(p) = path else { return };
    if let Some(dir) = p.parent() { let _ = std::fs::create_dir_all(dir); }
    let bs: Vec<serde_json::Value> = buckets.iter()
        .map(|&(ctx, ref t)| serde_json::json!([ctx, t.iter().map(|&(d, r)| serde_json::json!([d, r])).collect::<Vec<_>>()]))
        .collect();
    if let Ok(s) = serde_json::to_string_pretty(&serde_json::json!({ "stamp": stamp, "fmt": 2, "buckets": bs })) {
        if std::fs::write(p, s).is_ok() && gb10_inference::boot_report::debug() { println!("MTP cost/depth: cached to {}", p.display()); }
    }
}
