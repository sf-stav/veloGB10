//! Small single-model probes: Q4 and GDN-state checks, verify-width and reject probes, GEMM probe and
//! sweep, MoE probe, perplexity, and the layer-capture / argmax dump tools.
//! The state probe expects GDN state to match exactly (0.0); batch-invariance probes live in `exl3_bench`.

use super::*;

/// GDN state-divergence probe: compare the recurrent s_state after ONE verify_forward(N tokens)
/// call vs N individual forward_decode calls. A zero diff means verify_forward and forward_decode
/// are numerically identical in their GDN state update; a nonzero diff at N=1 means a real kernel
/// bug, while nonzero only at N>=2 implicates cuBLAS batch-size-dependent bf16 rounding in the
/// projection GEMMs (decode uses N=1, verify uses N=K).
/// `--probe-q4 --model-dir <d> [--prompt <text>] [--max-new-tokens N] [--max-seq-len N] [--chat]`
/// The smallest end-to-end generation on the GpuModel path: tokenize, prefill, greedy decode,
/// print. Family-agnostic (used to bring up qwen4_exp, works on any GpuModel model). `--chat`
/// renders the prompt through the model's chat template. Ported verbatim from PR #4 except the
/// chat-template call, which grew tools/effort/tpl_kwargs/thinking arguments in dev's P13.
pub(crate) fn run_probe_q4(args: &[String]) {
    let dir = parse_arg(args, "--model-dir").expect("--probe-q4 requires --model-dir <DIR>");
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("The capital of France is");
    let max_new: usize = parse_arg(args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(32);
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(2048);
    let chat = args.iter().any(|a| a == "--chat");
    let t0 = std::time::Instant::now();
    let (gpu, cfg) = gb10_inference::gpu::GpuModel::load_from_dir(dir).expect("gpu load");
    eprintln!("[probe-q4] loaded in {:.1}s (family {:?}, layers {}, rw {})", t0.elapsed().as_secs_f32(), cfg.family, cfg.num_layers, cfg.resid_width());
    // `--tokens 1,2,3` feeds raw ids (synthetic models have no tokenizer).
    let raw_tokens: Option<Vec<u32>> = parse_arg(args, "--tokens").map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
    let tok = if raw_tokens.is_some() { None } else {
        Some(gb10_inference::tokenizer::QwenTokenizer::from_file(&format!("{}/tokenizer.json", dir.trim_end_matches('/'))).expect("tokenizer"))
    };
    let prompt: Vec<u32> = if let Some(t) = raw_tokens { t } else {
        let text = if chat {
            let msgs = vec![gb10_inference::tokenizer::ChatMessage::user(prompt_text)];
            tok.as_ref().unwrap().apply_chat_template(&msgs, None, None, None,
                gb10_inference::tokenizer::ThinkingMode::Auto).expect("chat template")
        } else { prompt_text.to_string() };
        tok.as_ref().unwrap().encode(&text, false).expect("encode")
    };
    eprintln!("[probe-q4] prompt {} tokens: {:?}", prompt.len(), &prompt[..prompt.len().min(24)]);
    assert!(prompt.len() + max_new + 1 <= max_seq_len, "prompt+gen exceeds --max-seq-len");
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    let mut state = gpu.new_batch_state(1, 1, max_seq_len);
    let kv_stride = max_seq_len;
    let mut bufs = gpu.new_decode_buffers(1);
    let tp = std::time::Instant::now();
    let (first, hout) = gpu.prefill_batch(&mut pool, &prompt, &mut state, 0, kv_stride, 0);
    gpu.sync_stream();
    pool.release_bf16(hout, cfg.resid_width() * prompt.len());
    eprintln!("[probe-q4] prefill {} tok in {:.0} ms ({:.0} tok/s)", prompt.len(), tp.elapsed().as_secs_f32() * 1e3, prompt.len() as f32 / tp.elapsed().as_secs_f32());
    // `--dump-logits <dir>`: write prompt/generated tokens (JSON) and the per-step f32 logits
    // (prefill last column + every decode step) for the reference-oracle comparison.
    let dump: Option<String> = parse_arg(args, "--dump-logits").map(|s| s.to_string());
    let mut logit_steps: Vec<Vec<f32>> = Vec::new();
    if dump.is_some() {
        // Re-run the prefill on a fresh slot to read the last column's logits (the first one
        // consumed hout above).
        gpu.zero_slot_state(&mut state, 0, kv_stride);
        let (_, hout2) = gpu.prefill_batch(&mut pool, &prompt, &mut state, 0, kv_stride, 0);
        logit_steps.push(gpu.probe_logits_of_hidden_col(&mut pool, &hout2, prompt.len() - 1));
        pool.release_bf16(hout2, cfg.resid_width() * prompt.len());
    }
    let mut out = vec![first];
    let mut pos = prompt.len();
    let td = std::time::Instant::now();
    for _ in 1..max_new {
        if *out.last().unwrap() == cfg.eos_token_id && dump.is_none() { break; }
        let toks_i32 = vec![*out.last().unwrap() as i32];
        let pos_i32 = vec![pos as i32];
        gpu.dev().htod_sync_copy_into(&toks_i32, &mut bufs.tokens_dev).unwrap();
        gpu.dev().htod_sync_copy_into(&pos_i32, &mut bufs.pos_dev).unwrap();
        gpu.dev().synchronize().unwrap();
        let next = if dump.is_some() {
            let (lg, t) = gpu.probe_decode_logits(&mut pool, &mut bufs, &mut state, kv_stride, pos + 1);
            logit_steps.push(lg);
            t
        } else {
            gpu.forward_decode(&mut pool, &mut bufs, &mut state, kv_stride, pos + 1, 1)[0]
        };
        out.push(next);
        pos += 1;
    }
    let dt = td.elapsed().as_secs_f32();
    eprintln!("[probe-q4] decode {} tok in {:.2}s ({:.1} tok/s)", out.len() - 1, dt, (out.len() - 1) as f32 / dt.max(1e-6));
    println!("TOKENS: {:?}", out);
    if let Some(t) = &tok { println!("TEXT: {}", t.decode(&out, false).unwrap_or_default()); }
    if let Some(dir) = dump {
        std::fs::create_dir_all(&dir).expect("dump dir");
        let meta = serde_json::json!({ "prompt": prompt, "generated": out, "vocab": cfg.vocab_size, "steps": logit_steps.len() });
        std::fs::write(format!("{dir}/tokens.json"), serde_json::to_string(&meta).unwrap()).unwrap();
        let mut raw: Vec<u8> = Vec::with_capacity(logit_steps.len() * cfg.vocab_size * 4);
        for st in &logit_steps { for x in st { raw.extend_from_slice(&x.to_le_bytes()); } }
        std::fs::write(format!("{dir}/logits.f32"), raw).unwrap();
        eprintln!("[probe-q4] dumped {} logit steps to {dir}", logit_steps.len());
    }
}

pub(crate) fn run_probe_state(args: &[String]) {
    let (model_path, tokenizer_path) = if let Some(dir) = parse_arg(args, "--model-dir") {
        // S-A3-f-e: EXL3 packs route to the exl3 engine (the generic loader cannot
        // read this container — see the note in run_bench_mtp).
        if gb10_inference::exl3_serve::is_exl3_pack(dir) {
            let a = gb10_inference::exl3_forward::FwdArgs {
                dir: dir.to_string(),
                prompt: parse_arg(args, "--prompt").map(|s| s.to_string()),
                ids_file: parse_arg(args, "--prompt-ids").map(|s| s.to_string()),
                max_new: parse_arg(args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(16),
                width: 1,
                ladder_dir: None,
                ladder_layers: Vec::new(),
                ladder_step: None,
                max_pos: gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(8192),
                bench_steps: 0,
                widths: Vec::new(),
                prefill_chunk: None,
            };
            gb10_inference::exl3_forward::probe_state_exl3(&a).unwrap_or_else(|e| {
                eprintln!("PROBE_STATE_EXL3_FAIL: {e:#}");
                std::process::exit(1);
            });
            return;
        }
        (dir.to_string(), format!("{}/tokenizer.json", dir.trim_end_matches('/')))
    } else {
        (parse_arg(args, "--model").unwrap_or("model/model.safetensors").to_string(),
         parse_arg(args, "--tokenizer").unwrap_or("model/tokenizer.json").to_string())
    };
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("The capital of France is");
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);

    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");
    println!("GDN state-divergence probe: prompt={} tokens", prompt.len());

    let gpu = if std::path::Path::new(&model_path).is_dir() {
        let (gpu, _) = load_model_gpu(&model_path, None, 1);
        gpu
    } else {
        let host = gb10_inference::qwen::Model::load(&model_path).expect("load model");
        gb10_inference::gpu::GpuModel::new(&host).expect("gpu init")
    };
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    let mut state = gpu.new_batch_state(2, 2, max_seq_len);
    let kv_stride = max_seq_len;

    // B8/G1 ladder item 7: N=1..8 (was 1..=2). verify_state_diff re-zeros + re-prefills per N.
    let max_n: usize = parse_arg(args, "--max-n").and_then(|s| s.parse().ok())
        .unwrap_or(8).min(gb10_inference::gpu::MAX_VERIFY);
    let mut all_exact = true;
    for n in 1..=max_n {
        // Re-zero both slots and re-prefill fresh each iteration (verify_state_diff advances state).
        let diff = gpu.verify_state_diff(&mut pool, &mut state, &prompt, kv_stride, n);
        let exact = diff == 0.0;
        all_exact &= exact;
        println!("N={}: max |s_state(verify) - s_state(decode)| = {:.7}  {}",
                 n, diff, if exact { "EXACT MATCH" } else { "DIVERGES" });
    }
    println!("RESULT: {} (verify_state_diff N=1..={max_n})",
             if all_exact { "EXACT" } else { "DIVERGES" });
}

/// B8/G1: `--probe-verify-m8` — the verify-bucket bit-exactness gate at M<=8 (padded buckets
/// {2,4,6,8} vs exact width vs sequential decode, on real activations), followed by the width-8
/// reject-path probe. Prints per-width mismatches and a single RESULT line.
#[allow(clippy::too_many_lines)]
pub(crate) fn run_probe_verify_m8(args: &[String]) {
    let dir = parse_arg(args, "--model-dir").expect("--probe-verify-m8 requires --model-dir <DIR>");
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("The capital of France is");
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);
    let max_n: usize = parse_arg(args, "--max-n").and_then(|s| s.parse().ok()).unwrap_or(8).min(gb10_inference::gpu::MAX_VERIFY);

    let tokenizer_path = format!("{}/tokenizer.json", dir.trim_end_matches('/'));
    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");
    println!("verify-M8 bucket probe: prompt={} tokens, widths 1..={max_n}, buckets {{2,4,6,8}}", prompt.len());

    let (gpu, _) = load_model_gpu(&dir, None, 1);
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    // 2 + max_n state slots: verify lane, decode lane, per-column GDN checkpoints.
    let mut state = gpu.new_batch_state(2 + max_n, 2 + max_n, max_seq_len);
    let kv_stride = max_seq_len;

    let res = gpu.verify_bucket_bitexact(&mut pool, &mut state, &prompt, kv_stride, max_n);
    let mut bad = 0usize;
    for (n, mism) in &res {
        println!("  width {n}: exact-vs-bucket bit mismatches = {mism}");
        bad += *mism;
    }
    let (b, c, _d) = gpu.probe_reject_path_w8(&mut pool, &mut state, &prompt, kv_stride);
    println!("  reject@w8: restored-vs-checkpoint max |diff| = {b}, restored-vs-decode-ref max |diff| = {c}");
    let reject_ok = b == 0.0 && c == 0.0;
    println!("RESULT: {} (bucket bit-exactness mismatches={bad}, reject@w8 {})",
             if bad == 0 && reject_ok { "PASS" } else { "FAIL" },
             if reject_ok { "EXACT" } else { "DIVERGES" });
    if bad > 0 || !reject_ok { std::process::exit(1); }
}

/// Reject-path checkpoint/rollback three-way probe. Forces a rejection and checks whether the MTP
/// ping-pong snapshot (S1) and its D2D restore are bit-exact vs a single decode of the committed
/// token. Run with:  --probe-reject --model-dir 9b
pub(crate) fn run_probe_reject(args: &[String]) {
    let (model_path, tokenizer_path) = if let Some(dir) = parse_arg(args, "--model-dir") {
        (dir.to_string(), format!("{}/tokenizer.json", dir.trim_end_matches('/')))
    } else {
        (parse_arg(args, "--model").unwrap_or("model/model.safetensors").to_string(),
         parse_arg(args, "--tokenizer").unwrap_or("model/tokenizer.json").to_string())
    };
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("The capital of France is");
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);

    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");
    println!("Reject-path probe: prompt={} tokens", prompt.len());

    let gpu = if std::path::Path::new(&model_path).is_dir() {
        let (gpu, _) = load_model_gpu(&model_path, None, 1);
        gpu
    } else {
        let host = gb10_inference::qwen::Model::load(&model_path).expect("load model");
        gb10_inference::gpu::GpuModel::new(&host).expect("gpu init")
    };
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    // 3 slots: 0 = MTP slot A, 1 = decode reference B, 2 = checkpoint snapshot.
    let mut state = gpu.new_batch_state(3, 3, max_seq_len);
    let kv_stride = max_seq_len;
    gpu.probe_reject_path(&mut pool, &mut state, &prompt, kv_stride);
}

/// GEMM batch-invariance probe: directly measures whether cuBLAS bf16 GEMM gives identical results
/// for the first column at N=1 vs N=2, for the model's key GEMM shapes. Run with:
///   --probe-gemm --model-dir 9b          (tests the model's hidden/intermediate/conv_dim shapes)
pub(crate) fn run_probe_gemm(args: &[String]) {
    let model_dir = parse_arg(args, "--model-dir").map(|s| s.to_string());
    let gpu: gb10_inference::gpu::GpuModel = if let Some(dir) = model_dir {
        let (g, _) = load_model_gpu(&dir, None, 1);
        g
    } else {
        eprintln!("--probe-gemm requires --model-dir <DIR>");
        std::process::exit(1);
    };
    let cfg = gpu.cfg().clone();
    let h = cfg.hidden_size;
    let conv_dim = cfg.key_dim() * 2 + cfg.value_dim();
    let intermediate = cfg.intermediate_size;
    let value_dim = cfg.value_dim();
    // Test the dominant GEMM shapes (W^T @ X): (outn=M, inn=K=hidden or value_dim).
    println!("=== GEMM batch-invariance (N=1 vs N=2) for {} layers ===", cfg.num_layers);
    gpu.probe_gemm(conv_dim, h);          // GDN in_proj_qkv
    gpu.probe_gemm(value_dim, h);         // GDN in_proj_z
    gpu.probe_gemm(h, value_dim);         // GDN out_proj
    gpu.probe_gemm(intermediate, h);      // MLP gate/up
    gpu.probe_gemm(h, intermediate);      // MLP down
}

/// cuBLAS algo sweep: find a batch-invariant algo (N=1 == N=2) for the problematic GEMM shape.
/// Usage: --sweep-gemm --model-dir 9b   (sweeps the GDN in_proj_qkv shape, the dominant diverger)
pub(crate) fn run_sweep_gemm(args: &[String]) {
    let model_dir = parse_arg(args, "--model-dir").expect("--sweep-gemm requires --model-dir <DIR>");
    let (gpu, _) = load_model_gpu(model_dir, None, 1);
    let cfg = gpu.cfg().clone();
    let conv_dim = cfg.key_dim() * 2 + cfg.value_dim();
    // Sweep the in_proj_qkv shape (the one that diverged 0.5 on 9B).
    gpu.probe_gemm_sweep(conv_dim, cfg.hidden_size);
    // And the cuBLASLt SplitK-off variant (the candidate fix).
    println!();
    gpu.probe_gemm_lt(conv_dim, cfg.hidden_size);
    // And the custom batch-invariant kernel (the guaranteed fix).
    gpu.probe_gemm_binv(conv_dim, cfg.hidden_size);
}

/// `--profile-mtp` — attribute the cost of one stochastic-MTP step.
/// DEBUG PROBE: `--probe-moe --model-dir <D> --batch B --out-x <f> --out <f>` — run the first MoE
/// layer's `moe_batch` on a deterministic input and write input+output (f32, space-separated) so a
/// numpy reference over that layer's checkpoint weights can validate the block numerically.
pub(crate) fn run_probe_moe(args: &[String]) {
    use half::bf16;
    let dir = parse_arg(args, "--model-dir").expect("--probe-moe requires --model-dir").to_string();
    let batch: usize = parse_arg(args, "--batch").and_then(|s| s.parse().ok()).unwrap_or(4);
    let out_x = parse_arg(args, "--out-x").unwrap_or("/tmp/moe_x.txt").to_string();
    let out_y = parse_arg(args, "--out").unwrap_or("/tmp/moe_y.txt").to_string();

    let (gpu, cfg) = load_model_gpu(&dir, None, 1);
    let h = cfg.hidden_size;
    // Input: --in-x <file> (whitespace floats, token-major [batch, h]) or the deterministic
    // reproducible default in [-0.5, 0.5], col-major [h, batch].
    let mut xh = vec![bf16::from_f32(0.0); h * batch];
    if let Some(in_x) = parse_arg(args, "--in-x") {
        let vals: Vec<f32> = std::fs::read_to_string(in_x).expect("read --in-x")
            .split_whitespace().map(|t| t.parse().expect("float")).collect();
        assert_eq!(vals.len(), h * batch, "--in-x must hold hidden*batch floats (token-major)");
        for (i, v) in vals.iter().enumerate() { xh[i] = bf16::from_f32(*v); }
    } else {
        for b in 0..batch { for j in 0..h {
            let v = ((j * 7 + b * 131) % 211) as f32 / 211.0 - 0.5; // in [-0.5, 0.5)
            xh[j + b * h] = bf16::from_f32(v);
        }}
    }
    let (li, out) = gpu.probe_moe(&xh, batch);
    eprintln!("probe-moe: first MoE layer = {}, batch = {}, hidden = {}", li, batch, h);
    let fmt = |v: &[bf16]| v.iter().map(|x| format!("{:.6}", x.to_f32())).collect::<Vec<_>>().join(" ");
    std::fs::write(&out_x, fmt(&xh)).expect("write x");
    std::fs::write(&out_y, fmt(&out)).expect("write y");
    eprintln!("probe-moe: wrote input->{} output->{} (layer {})", out_x, out_y, li);
    println!("MOE_PROBE_LAYER={}", li);
}

pub(crate) fn run_perplexity(args: &[String]) {
    // S-B9-REL-PPL: an EXL3 pack runs on the EXL3 engine — the served chunked prefill with every row
    // of each window through the lm_head (the generic loader cannot read this container: on the
    // v0.7.3 binary it panicked on a misaligned tensor and then hung).
    if let Some(dir) = parse_arg(args, "--model-dir") {
        if gb10_inference::exl3_serve::is_exl3_pack(dir) {
            let a = gb10_inference::exl3_forward::PplArgs::from_args(args).unwrap_or_else(|e| {
                eprintln!("PERPLEXITY_EXL3_FAIL: {e:#}");
                std::process::exit(2);
            });
            gb10_inference::exl3_forward::perplexity_exl3(&a).unwrap_or_else(|e| {
                eprintln!("PERPLEXITY_EXL3_FAIL: {e:#}");
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
    let text_path = gb10_inference::opt_or_arg(args, "--text").expect("--text <file> required");
    let window: usize = gb10_inference::opt_or_arg(args, "--window").and_then(|s| s.parse().ok()).unwrap_or(1024);
    let max_windows: usize = gb10_inference::opt_or_arg(args, "--max-windows").and_then(|s| s.parse().ok()).unwrap_or(8);
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);
    if window < 2 {
        // a scored position needs two tokens; this used to run and print `mean_nll=NaN`
        eprintln!("--window must be >= 2 (one scored position needs two tokens), got {window}");
        std::process::exit(2);
    }

    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let text = std::fs::read_to_string(&text_path).expect("read --text file");
    // No chat template, no BOS games: score raw text.
    let toks = tokenizer.encode(&text, false).expect("encode");
    println!("Perplexity: {} tokens from {}, window={}, max_windows={}",
             toks.len(), text_path, window, max_windows);

    let gpu = if std::path::Path::new(&model_path).is_dir() {
        let (gpu, _) = load_model_gpu(&model_path, None, 1);
        gpu
    } else {
        let host = gb10_inference::qwen::Model::load(&model_path).expect("load model");
        gb10_inference::gpu::GpuModel::new(&host).expect("gpu init")
    };
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    let mut state = gpu.new_batch_state(2, 2, max_seq_len);

    let mut total_nll = 0.0f64;
    let mut total_tok = 0usize;
    let mut w = 0usize;
    for chunk in toks.chunks(window) {
        if chunk.len() < 2 || w >= max_windows { break; }
        let (nll, n) = gpu.window_nll(&mut pool, &mut state, chunk, max_seq_len);
        total_nll += nll;
        total_tok += n;
        w += 1;
        println!("  window {:2}: {:5} tok   nll/tok {:.4}   ppl {:.3}",
                 w, n, nll / n as f64, (nll / n as f64).exp());
    }
    let mean = total_nll / total_tok as f64;
    println!();
    println!("RESULT: tokens={}  mean_nll={:.5}  PERPLEXITY={:.4}", total_tok, mean, mean.exp());
}

/// qwen4_exp per-layer capture: the residual is rw-wide (hc streams × h), so the per-layer
/// boundaries are captured with PREFIX runs of the PP layer-range prefill instead of
/// `capture_prefill` (which is FullAttention/hy_v3-only): `prefill_batch_range(0..i)` returns the
/// residual leaving layer i-1, value-exactly (that is the PP hand-off contract). Each prefix
/// re-prefills from a zeroed slot — cheap on a slice (a handful of prefills). i=0 dumps the
/// expanded embed output (layer.00.in). Shared by `--capture-layers` (TP=1) and the
/// `--tp-capture` SPMD branch (TP=2: both ranks run the identical prefix ladder, the
/// all-reduces fire inside, and the two dumps must be bit-identical).
pub(super) fn capture_layers_q4(gpu: &mut gb10_inference::gpu::GpuModel,
                     pool: &mut gb10_inference::gpu::Pool,
                     ids: &[u32]) -> Vec<(String, Vec<half::bf16>)> {
    let n = ids.len();
    let rw = gpu.cfg().resid_width();
    let nlayers = gpu.cfg().num_layers;
    let kv_stride = n.max(16);
    let mut state = gpu.new_batch_state(1, 1, kv_stride);
    let mut named: Vec<(String, Vec<half::bf16>)> = Vec::with_capacity(nlayers + 1);
    for i in 0..=nlayers {
        gpu.zero_slot_state(&mut state, 0, kv_stride);
        let (_, hout) = gpu.prefill_batch_range(pool, ids, &mut state, 0, kv_stride, 0, 0, i, None);
        let v: Vec<half::bf16> = gpu.dev().dtoh_sync_copy(&hout).unwrap();
        pool.release_bf16(hout, rw * n);
        assert_eq!(v.len(), rw * n, "q4 capture: expected rw-wide residual");
        let name = if i == 0 { "layer.00.in".to_string() } else { format!("layer.{:02}.out", i - 1) };
        named.push((name, v));
    }
    named
}

/// DEBUG CAPTURE: `--capture-layers --model-dir <D> [--ids <f> | --prompt <text>] --out <f>` —
/// teacher-force one prompt through the prefill path and dump the hidden state at every layer
/// boundary as a safetensors file of bf16 [seq, width] tensors: `layer.00.in` (embed out),
/// `layer.NN.out` (residual after layer NN), `final_norm`. The prompt is RAW token ids
/// (whitespace-separated, NOT re-tokenized text) or `--prompt` text tokenized with the model dir's
/// own tokenizer — the same encode the TP bench head broadcasts, so a TP=1 and a TP=2 capture of
/// the same `--prompt` see bit-identical ids. This is the engine side of the Hy3 oracle comparison
/// (scripts/compare_hy3_oracle.py consumes these dumps) and the B15 TP=2-vs-TP=1 localization.
pub(crate) fn run_capture_layers(args: &[String]) {
    use safetensors::{Dtype, tensor::TensorView};
    let dir = parse_arg(args, "--model-dir").expect("--capture-layers requires --model-dir").to_string();
    let out_path = parse_arg(args, "--out").expect("--capture-layers requires --out <file>").to_string();

    let ids: Vec<u32> = match parse_arg(args, "--prompt") {
        Some(text) => {
            let tok_path = format!("{}/tokenizer.json", dir.trim_end_matches('/'));
            let tokenizer = QwenTokenizer::from_file(&tok_path).expect("tokenizer.json");
            let ids = tokenizer.encode(&text, true).expect("tokenize --prompt");
            println!("capture-layers: --prompt {:?} -> {} tokens", text, ids.len());
            ids
        }
        None => {
            let ids_path = parse_arg(args, "--ids")
                .expect("--capture-layers requires --ids <file> or --prompt <text>").to_string();
            let ids_txt = std::fs::read_to_string(&ids_path).expect("read --ids file");
            ids_txt.split_whitespace()
                .map(|t| t.parse().expect("token id")).collect()
        }
    };
    assert!(!ids.is_empty(), "empty --ids file");
    let n = ids.len();

    let (mut gpu, cfg) = load_model_gpu(&dir, None, 1);
    let h = cfg.hidden_size;
    let nlayers = gpu.cfg().num_layers;
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    let kv_stride = n.max(16);
    let mut state = gpu.new_batch_state(1, 1, kv_stride);

    let (named, width) = if cfg.is_q4() {
        (capture_layers_q4(&mut gpu, &mut pool, &ids), cfg.resid_width())
    } else {
        let dumps = gpu.capture_prefill(&mut pool, &ids, &mut state, kv_stride);
        let debug = gb10_inference::opts::var(gb10_inference::opt!("cap-debug")).is_ok();
        if !debug {
            assert_eq!(dumps.len(), nlayers + 2, "capture count: embed + L layers + final_norm");
        }

        // Name per the oracle convention and serialize as bf16 [seq, hidden] (token-major — the
        // engine's [h, n] column-major activations are byte-identical to [n, h] row-major rows).
        let mut named: Vec<(String, Vec<half::bf16>)> = Vec::with_capacity(dumps.len());
        for (i, dmp) in dumps.into_iter().enumerate() {
            let name = if debug { format!("dump.{i:03}") } else if i == 0 { "layer.00.in".to_string() }
                       else if i <= nlayers { format!("layer.{:02}.out", i - 1) }
                       else { "final_norm".to_string() };
            named.push((name, dmp));
        }
        (named, h)
    };
    let views: Vec<(String, TensorView)> = named.iter()
        .map(|(name, dmp)| {
            let bytes: &[u8] = bytemuck::cast_slice(&dmp[..]);
            (name.clone(), TensorView::new(Dtype::BF16, vec![n, width], bytes).expect("view"))
        }).collect();
    safetensors::serialize_to_file(views, None, std::path::Path::new(&out_path)).expect("write safetensors");
    eprintln!("capture-layers: {} tokens x {} layers -> {} ({} tensors)", n, nlayers, out_path, named.len());
    println!("CAPTURE_OK {}", out_path);
}

pub(crate) fn run_dump_argmax(args: &[String]) {
    let model_path = parse_arg(args, "--model-dir").expect("--dump-argmax requires --model-dir <DIR>").to_string();
    let tokenizer_path = format!("{}/tokenizer.json", model_path.trim_end_matches('/'));
    let text_path = gb10_inference::opt_or_arg(args, "--text").expect("--text <file> required");
    let out_path = parse_arg(args, "--out").expect("--out <file> required");
    let window: usize = gb10_inference::opt_or_arg(args, "--window").and_then(|s| s.parse().ok()).unwrap_or(512);
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);

    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let text = std::fs::read_to_string(&text_path).expect("read --text file");
    let toks = tokenizer.encode(&text, false).expect("encode");
    eprintln!("dump-argmax: {} tokens from {}, window={}", toks.len(), text_path, window);

    let (gpu, _) = load_model_gpu(&model_path, None, 1);
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    let mut state = gpu.new_batch_state(2, 2, max_seq_len);

    // For each fresh window: pred[i] = argmax at position i (the model's greedy token for position i+1).
    // Emit `actual_next_token \t model_argmax` for i in 0..len-1; a blank line marks each window
    // boundary so the offline comparator never pairs predictions across a state reset.
    let mut lines: Vec<String> = Vec::with_capacity(toks.len());
    for chunk in toks.chunks(window) {
        if chunk.len() < 2 { continue; }
        let pred = gpu.window_argmax(&mut pool, &mut state, chunk, max_seq_len);
        for i in 0..chunk.len().saturating_sub(1) {
            lines.push(format!("{}\t{}", chunk[i + 1], pred[i]));
        }
        lines.push(String::new());
    }
    std::fs::write(&out_path, lines.join("\n")).expect("write --out");
    eprintln!("dump-argmax: wrote {} rows to {}", lines.iter().filter(|l| !l.is_empty()).count(), out_path);
}
