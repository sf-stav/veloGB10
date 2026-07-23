use gb10_inference::{engine::GB10InferenceEngine, server::{create_router, AppState}, tokenizer::QwenTokenizer};
use std::env;
use std::sync::Arc;

fn print_help() {
    let arg0 = std::env::args().next().unwrap_or_default();
    let prog = std::path::Path::new(&arg0)
        .file_name().unwrap_or_default().to_string_lossy();

    println!("gb10_inference â from-scratch Rust+CUDA inference for Qwen3.5/3.6 hybrid models on GB10");
    println!();
    println!("USAGE:");
    println!("  {prog} <MODE> [OPTIONS]");
    println!();
    println!("MODES:");
    println!("  --server            OpenAI-compatible HTTP server (this is the one you want)");
    println!("  --quantize          Offline NVFP4/FP8 quantizer (--model-dir <in> --out <dir> --recipe <r>)");
    println!("  --perplexity        Perplexity on held-out text (--text <file> --window N --max-windows N)");
    println!("  --bench-mtp         End-to-end MTP probe: proves greedy is bitwise lossless, reports tok/s");
    println!("  --bench-verify      MTP verify == sequential decode, bitwise (add --draws N to fuzz)");
    println!("  --bench-accept      Diagnose acceptance: coverage by target confidence, n-gram run-length");
    println!("  --probe-binv        Batch-invariance probe (column 0 bit-identical for every N)");
    println!("  --bench-batch       Batched-decode throughput");
    println!("  (default)           Interactive CLI: load model, generate from a prompt");
    println!("  --help, -h          Show this help");
    println!();
    println!("âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ");
    println!("  SERVER MODE (--server) â OpenAI-compatible, continuous batching");
    println!("âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ");
    println!();
    println!("  Endpoints:  POST /v1/chat/completions   GET /v1/models[/:id]   GET /health");
    println!();
    println!("  MODEL");
    println!("    --model-dir <DIR>          Model directory (config.json + *.safetensors + tokenizer).");
    println!("                               THIS is the normal way to load.                   [required]");
    println!("    --model-name <NAME>        Name reported by /v1/models    [derived from the dir name]");
    println!("    --model <FILE>             Legacy: single .safetensors file (use --model-dir instead)");
    println!("    --tokenizer <FILE>         Legacy: tokenizer.json path (implied by --model-dir)");
    println!();
    println!("  SERVING");
    println!("    --port <N>                 Listen port                                          [8000]");
    println!("    --max-batch <N>            Max concurrent sequences (lanes)                        [8]");
    println!("    --max-tokens <N>           Default generation cap when a request omits max_tokens [8192]");
    println!();
    println!("  CONTEXT LENGTH  (this is how you set context size â the KV cache is sized to it)");
    println!("    --max-seq-len <N>          KV cache depth in tokens = the max context (prompt+gen) a");
    println!("                               request may use. The KV cache is allocated to exactly this");
    println!("                               size. Clamped to the model's max_position_embeddings (256K");
    println!("                               for this family). A prompt longer than this is rejected; a");
    println!("                               request whose prompt+max_tokens would exceed it has its");
    println!("                               generation clamped (logged as '[req] max_tokens clamped').");
    println!("                               MEMORY: KV â (full-attn layers Ã kv_heads Ã head_dim Ã 4B) per");
    println!("                               token per lane â ~64 KB/token on 27B. So 256K Ã batch-2 â 34");
    println!("                               GB (fine on 128 GB, but a 256K prefill is slow). Above ~12K,");
    println!("                               CUDA graphs are skipped (measured zero cost on GB10).   [4096]");
    println!();
    println!("  SPECULATION  (auto-tuned; you normally set none of these)");
    println!("    --mtp <auto|on|off>        Multi-token (MTP) speculative decoding. 'auto' measures");
    println!("                               whether it pays and self-tunes depth from live acceptance.");
    println!("                               Greedy verify is bitwise lossless; temp>0 is distribution-");
    println!("                               exact. on/off force it (benchmarking).                 [auto]");
    println!("    --mtp-depth <N>            Pin the draft depth instead of auto-picking (benchmarking).");
    println!("    --ngram-draft <N>          EXPERIMENTAL prompt-lookup drafting, n-gram order N (0=off).");
    println!("                               Lossless but measured net-negative as a plain replacement.   [0]");
    println!();
    println!("  PREFIX CACHE");
    println!("    --prefix-cache <on|off>    Reuse a conversation's cached prefix instead of re-prefilling");
    println!("                               it â ~3x faster follow-up turns on multi-turn/agent traffic.");
    println!("                               NOT bit-exact: reuse re-chunks the prefill and cuBLAS picks a");
    println!("                               kernel per shape, so a cached turn can word an answer");
    println!("                               differently than a cold one. Greedy MTP stays lossless.  [off]");
    println!();
    println!("  SAMPLING DEFAULTS  (server-level; every request may override in its JSON body)");
    println!("    --default-repetition-penalty <F>  Repetition penalty, 1.0 = off                    [1.0]");
    println!("    --default-presence-penalty <F>    Presence penalty       [2.0 on 2B, else 1.5]");
    println!("    --default-frequency-penalty <F>   Frequency penalty                                [0.0]");
    println!("    (temperature / top_p / top_k / seed are per-REQUEST only â defaults 0.7 / 0.8 / 20)");
    println!();
    println!("  EXAMPLES");
    println!("    {prog} --server --model-dir /models/3.6-27b-nvfp4-full \\");
    println!("        --port 9000 --max-seq-len 32768 --max-batch 2 --prefix-cache on");
    println!("    {prog} --server --model-dir <dir> --max-seq-len 262144   # full 256K context");
    println!();
    println!("âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ");
    println!("  BENCH-BATCH MODE (--bench-batch)");
    println!("âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ");
    println!();
    println!("  Runs M identical prompts through batched decode and reports");
    println!("  aggregate throughput. Verifies token-exact correctness.");
    println!();
    println!("  Options:");
    println!("    --model <PATH>          Model weights (safetensors)   [model/model.safetensors]");
    println!("    --tokenizer <PATH>      Tokenizer JSON               [model/tokenizer.json]");
    println!("    --prompt <TEXT>         Prompt text                   [\"The capital of France is\"]");
    println!("    --batch <N>             Number of parallel sequences  [4]");
    println!("    --max-new-tokens <N>    Tokens to decode per sequence  [32]");
    println!("    --max-seq-len <N>       KV cache positions             [4096]");
    println!();
    println!("  Example:");
    println!("    {prog} --bench-batch --batch 16 --max-new-tokens 64");
    println!();
    println!("âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ");
    println!("  CLI MODE (default)");
    println!("âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ");
    println!();
    println!("  Loads the model, encodes a prompt, and generates tokens.");
    println!();
    println!("  Options:");
    println!("    --model <PATH>          Model weights (packed or safetensors)  [qwen3.5-0.8b-packed]");
    println!("    --tokenizer <PATH>      Tokenizer JSON                       [model/tokenizer.json]");
    println!("    --prompt <TEXT>         Prompt text                          [\"The capital of France is\"]");
    println!("    --max-seq-len <N>       KV cache positions                   [4096]");
    println!("    --max-new-tokens <N>    Tokens to generate                   [16]");
    println!("    --temperature <F>       Sampling temperature (0 = greedy)    [0.0]");
    println!();
    println!("  Example:");
    println!("    {prog} --prompt \"Explain gravity\" --max-new-tokens 128 --temperature 0.7");
    println!();
    println!("════════════════════════════════════════════════════════════════════════════════");
    println!("  TP=2 SERVING (--server --tp) — two-box tensor parallelism over RoCE");
    println!("════════════════════════════════════════════════════════════════════════════════");
    println!();
    println!("  One model served by TWO GB10 boxes: the HEAD runs the OpenAI server and drives");
    println!("  SPMD decode; the NODE runs a mirror scheduler with ZERO configuration (model,");
    println!("  config, cost table and stop tokens all ship from the head at sync). Output is");
    println!("  bitwise in lockstep; a per-step agree() guard + liveness watchdog abort both");
    println!("  sides LOUDLY on any divergence.");
    println!();
    println!("  Node (on the peer, resident — re-arms itself between head sessions):");
    println!("    gb10_inference --node [--port 29500] [--rdma-dev <ib0[,ib1]>] [--once]");
    println!();
    println!("  Head (this box):");
    println!("    gb10_inference --server --model-dir <DIR> --tp --nodes <ip>[:port] \\");
    println!("        [--max-seq-len N] [--max-batch N] [--prefix-cache on] [usual server flags]");
    println!();
    println!("  TP FLAGS (head)");
    println!("    --tp                     Enable TP=2 on --server (sync + RDMA bring-up first)");
    println!("    --nodes <ip[:port],...>  Explicit node address(es); skips UDP discovery");
    println!("    --discover-wait <S>      Discovery broadcast window instead of --nodes      [3]");
    println!("    --rdma-dev <d1[,d2]>     RoCE devices if the defaults (rocep1s0f1, roceP2p1s0f1)");
    println!("                             don't match the platform (also GB10_RDMA_DEV)");
    println!("    --no-shard-mixers        Escape hatch: DON'T shard attention/GDN mixers + KV");
    println!("                             (default is ON under --tp/--head — it's where the speed is)");
    println!("    --tp-graph               CUDA-graph the TP decode (bench path)");
    println!("    --tp-fp32-partials       FP32 all-reduce partials (kills the bf16-partial dip)");
    println!("    --tp-trace               Per-barrier timing histograms at exit");
    println!("    --tp-cache <dir>         Node's model blob cache        [~/.cache/gb10_tp]");
    println!("    --head --model-dir <DIR> One-shot bench/generate head (tp_serve proof path;");
    println!("                             use --server --tp for serving)");
    println!();
    println!("  TP ENV-VAR ALIASES (the CLI flags above are preferred; env stays for back-compat.");
    println!("  Set on the HEAD only — the sync ships the config to the node, which needs nothing):");
    println!("    GB10_TP_SHARD_MIXERS=1 ↔ default-on under --tp/--head (--no-shard-mixers turns off)");
    println!("    GB10_TP_GRAPH=1 ↔ --tp-graph    GB10_TP_FP32_PARTIALS=1 ↔ --tp-fp32-partials");
    println!("    GB10_TP_TRACE=1 ↔ --tp-trace    GB10_TP_CACHE=<dir> ↔ --tp-cache <dir>");
    println!("    GB10_TP_MTP=1 ↔ --mtp=on (bench rig)   GB10_TP_MTP_DEPTH=N ↔ --mtp-depth N");
    println!("    GB10_TP_TAIL_DRILL=1     TEST: invert commit/payload every 4096th epoch");
    println!("    GB10_TP_AGREE_DRILL=N    TEST: corrupt this rank's agree hash at step N");
    println!();
    println!("  TYPICAL");
    println!("    peer: ./gb10_inference --node --port 29500");
    println!("    head: ./gb10_inference --server --model-dir /models/3.5-122b-nvfp4-gdn4 --tp \\");
    println!("          --nodes <peer-ip>:29500 --port 9000 --max-seq-len 32768");
    println!("  See TP2_SERVING_RUNBOOK.md for the full runbook + troubleshooting table.");
    println!();
    println!("════════════════════════════════════════════════════════════════════════════════");
    println!("  PROBES & BENCHES (correctness gates and diagnostics)");
    println!("════════════════════════════════════════════════════════════════════════════════");
    println!();
    println!("  --bench-mtp              MTP vs sequential greedy: LOSSLESS_OK + acceptance + tok/s");
    println!("                           [--model-dir <d> --depth N --max-new-tokens N --max-seq-len N]");
    println!("  --bench-verify           Verify == sequential, bitwise [--draws N to fuzz]");
    println!("  --bench-accept           Acceptance diagnosis by target confidence / n-gram run");
    println!("  --bench-mtp-sample       Distribution gate for stochastic MTP (chi-square)");
    println!("  --bench-tree             Tree-verify byte gate (RESULT: TREE_OK)");
    println!("  --bench-lanes            Batched-verify-across-lanes byte gate (RESULT: LANES_OK)");
    println!("  --bench-prefill          Pure prefill timing (TTFT proxy) [--seq-len N]");
    println!("  --probe-binv             Batch-invariance: col 0 bit-identical N=1..16 (prints PASS)");
    println!("  --probe-state            GDN recurrent-state divergence probe (must be 0.0)");
    println!("  --probe-reject           Reject-path checkpoint/rollback 3-way probe");
    println!("  --probe-gemm             cuBLAS bf16 GEMM per-shape batch-invariance audit");
    println!("  --probe-bandwidth        STREAM-style roofline (idle GB10 ≈ 255 GB/s; <245 = contended)");
    println!("  --probe-bandwidth-sustained [--seconds N]   thermal derating under load");
    println!("  --tp-barrier-bench       Doorbell transport adversarial gates (no model needed)");
    println!("  --net-test               Transport + FP32-partial audit, 2 procs (--rank 0|1 --peer)");
    println!("  --sweep-gemm             GEMM shape sweep");
    println!("  --perplexity             PPL on held-out text (--text <file> --window N --max-windows N)");
    println!("  --quantize               bf16 dir -> NVFP4/FP8 artifact (--model-dir <in> --out <dir> --recipe <r>)");
    println!("  --capture-layers         Dump per-layer hidden states for raw token ids (--ids <f> --out <f>)");
    println!();
    println!("════════════════════════════════════════════════════════════════════════════════");
    println!("  SESSION BEHAVIOR FLAGS (env aliases in parentheses, back-compat; CLI wins)");
    println!("════════════════════════════════════════════════════════════════════════════════");
    println!();
    println!("  --kv-cache bf16|q4       KV cache format (q4 = 4-bit;  GB10_KV_QUANT)");
    println!("  --reasoning-effort <e>   hy_v3 reasoning in the chat template: no_think|low|high");
    println!("                           (default no_think; request field `reasoning_effort` overrides)");
    println!("  --no-decode-graphs       Disable decode CUDA graphs    (GB10_NO_DECODE_GRAPHS)");
    println!("  --no-gqpack              Per-head q4 attention fallback  (GB10_NO_GQPACK)");
    println!("  --fuse-residual          Fused residual+norm epilogue    (GB10_FUSE_RESIDUAL)");
    println!("  --cpu-sample             Sample on CPU instead of GPU    (RUST_INFER_CPU_SAMPLE)");
    println!("  --prefill-scalar         Scalar (non-tiled) attn prefill (RUST_INFER_PREFILL_SCALAR)");
    println!("  --zero-kv                Restore cold-admit KV zeroing   (RUST_INFER_ZERO_KV)");
    println!("  --draft-vocab N          FR-Spec draft vocab subset; 0=off (RUST_INFER_DRAFT_VOCAB)");
    println!("  GB10_RDMA_DEV=<d1[,d2]>  RoCE device override (see --rdma-dev)");
    println!();
    println!("Note: there are NO MTP on/off env vars — speculation is auto-tuned per request");
    println!("(greedy => argmax verify, bitwise lossless; temp>0 => speculative rejection");
    println!("sampling, distribution-exact). --mtp=on|off and --mtp-depth exist for benches.");
    println!();
}

fn main() {
    let args: Vec<String> = env::args().collect();
    cli_env_bridge(&args);

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }

    // Check for server mode
    if args.iter().any(|a| a == "--server") {
        run_server(&args);
        return;
    }

    // Batched benchmark mode
    if args.iter().any(|a| a == "--bench-batch") {
        run_bench_batch(&args);
        return;
    }

    // TP=2 doorbell barrier microbench — the adversarial gate the protocol must pass BEFORE the model
    // depends on it (tp_doorbell_ref/BENCH_PLAN.md). Runs on the real transport with no model.
    if args.iter().any(|a| a == "--tp-barrier-bench") {
        run_tp_barrier_bench(&args);
        return;
    }

    // Pure prefill timing (TTFT proxy) at a given sequence length. Profile with nsys for the kernel
    // breakdown. `--bench-prefill --model-dir <d> --seq-len N`.
    if args.iter().any(|a| a == "--bench-prefill") {
        run_bench_prefill(&args);
        return;
    }

    // MTP verify lossless probe
    if args.iter().any(|a| a == "--bench-verify") {
        run_bench_verify(&args);
        return;
    }

    // GDN state-divergence probe (verify_forward vs forward_decode recurrent state)
    if args.iter().any(|a| a == "--probe-state") {
        run_probe_state(&args);
        return;
    }
    // Reject-path checkpoint/rollback three-way probe
    if args.iter().any(|a| a == "--probe-reject") {
        run_probe_reject(&args);
        return;
    }

    // GEMM batch-invariance: per-shape divergence + cuBLAS algo sweep
    if args.iter().any(|a| a == "--probe-gemm") {
        run_probe_gemm(&args);
        return;
    }
    // What IS the roofline? Every other number is measured against it.
    if args.iter().any(|a| a == "--probe-bandwidth") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-bandwidth requires --model-dir <DIR>");
        let (gpu, _) = gb10_inference::gpu::GpuModel::load_from_dir(dir).expect("gpu load");
        gpu.probe_bandwidth();
        return;
    }
    // §2.3 audit: sustained bandwidth over `--seconds N` — reveals LPDDR5x thermal derating under load.
    if args.iter().any(|a| a == "--probe-bandwidth-sustained") {
        let dir = parse_arg(&args, "--model-dir").expect("requires --model-dir <DIR>");
        let secs: u64 = parse_arg(&args, "--seconds").and_then(|s| s.parse().ok()).unwrap_or(180);
        let (gpu, _) = gb10_inference::gpu::GpuModel::load_from_dir(dir).expect("gpu load");
        gpu.probe_bandwidth_sustained(secs);
        return;
    }
    // The gate the whole speculative path rests on: col 0 of an N-wide verify == a N=1 decode, BITWISE.
    if args.iter().any(|a| a == "--probe-binv") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-binv requires --model-dir <DIR>");
        let (gpu, _) = gb10_inference::gpu::GpuModel::load_from_dir(dir).expect("gpu load");
        if !gpu.probe_binv() { std::process::exit(1); }
        return;
    }
    // TP=2 half-width GEMV: does the per-node sharded decode GEMV still hit ~80% roofline? The whole
    // ~1.85x TP=2 projection hinges on it. Pass any --model-dir just for the GPU context (synthetic
    // buffers; weights unused). 27B dims: H=5120, I=17408, Q=24x256, KV=4x256.
    if args.iter().any(|a| a == "--probe-tp-gemv") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-tp-gemv requires --model-dir <DIR>");
        let (gpu, _) = gb10_inference::gpu::GpuModel::load_from_dir(dir).expect("gpu load");
        println!("=== TP=2 half-width GEMV probe (gemm_mma_fp4_b, N=1) — 27B linears, FULL vs TP=2-half ===");
        println!("  roofline reference: ~245 GB/s sustained (probe-bandwidth for the live number)");
        println!("--- FFN gate/up (column-parallel: M 17408 -> 8704) ---");
        gpu.probe_tp_gemv(17408, 5120, "gate/up  FULL");
        gpu.probe_tp_gemv(8704, 5120, "gate/up  TP2-half");
        println!("--- FFN down (row-parallel: K 17408 -> 8704, halves the reduction) ---");
        gpu.probe_tp_gemv(5120, 17408, "down     FULL");
        gpu.probe_tp_gemv(5120, 8704, "down     TP2-half");
        println!("--- attn Q proj (column-parallel: M 6144 -> 3072) ---");
        gpu.probe_tp_gemv(6144, 5120, "q_proj   FULL");
        gpu.probe_tp_gemv(3072, 5120, "q_proj   TP2-half");
        println!("--- attn O proj (row-parallel: K 6144 -> 3072) ---");
        gpu.probe_tp_gemv(5120, 6144, "o_proj   FULL");
        gpu.probe_tp_gemv(5120, 3072, "o_proj   TP2-half");
        return;
    }
    // G-A: TP=2 transport + FP32-partial numerical audit. rank 0 (head) here, rank 1 (node) on peer:
    //   ./gb10_inference --net-test --rank 0 --port 23470
    //   ./gb10_inference --net-test --rank 1 --peer <peer-ip> --port 23470
    if args.iter().any(|a| a == "--net-test") {
        run_net_test(&args);
        return;
    }
    // TP=2 cluster orchestration. Node: launch and wait for the head to sync the model.
    //   ./gb10_inference --node [--port 29500]
    // Head: discover nodes (or --nodes ip[:port],...) and push the model (content-addressed cache).
    //   ./gb10_inference --head --model-dir <DIR> [--nodes <peer-ip>] [--discover-wait 3]
    if args.iter().any(|a| a == "--node") {
        run_cluster_node(&args);
        return;
    }
    if args.iter().any(|a| a == "--head") {
        run_cluster_head(&args);
        return;
    }
    if args.iter().any(|a| a == "--sweep-gemm") {
        run_sweep_gemm(&args);
        return;
    }

    // Offline quantizer: bf16 model dir -> NVFP4/FP8 compressed-tensors artifact.
    if args.iter().any(|a| a == "--quantize") {
        run_quantize(&args);
        return;
    }

    // Derive a gdn4 (GDN-nvfp4) artifact FROM an existing mixed (GDN-fp8) one, WITHOUT the bf16:
    // copies every non-GDN tensor verbatim and re-quantizes only the fp8 GDN in/out-projs to nvfp4.
    //   --requant-gdn --from <mixed-dir> --out <gdn4-dir>
    if args.iter().any(|a| a == "--requant-gdn") {
        run_requant_gdn(&args);
        return;
    }

    // Perplexity on held-out text â the quality gate for quantization.
    if args.iter().any(|a| a == "--perplexity") {
        run_perplexity(&args);
        return;
    }

    // DEBUG PROBE: dump per-position greedy argmax over a corpus (cross-model spec-decode acceptance).
    if args.iter().any(|a| a == "--dump-argmax") {
        run_dump_argmax(&args);
        return;
    }

    // DEBUG PROBE: MoE block correctness oracle (run moe_batch on a fixed input, dump in/out).
    if args.iter().any(|a| a == "--probe-moe") {
        run_probe_moe(&args);
        return;
    }

    // DEBUG CAPTURE: per-layer hidden states for one prompt of raw token ids (oracle validation).
    if args.iter().any(|a| a == "--capture-layers") {
        run_capture_layers(&args);
        return;
    }

    // Per-phase timing of one stochastic-MTP step (where the acceptance win is being lost).
    if args.iter().any(|a| a == "--profile-mtp") {
        run_profile_mtp(&args);
        return;
    }

    // Stochastic-MTP distribution-exactness gate (temp>0). Exact match, so it does not shadow
    // --bench-mtp below.
    if args.iter().any(|a| a == "--bench-mtp-sample") {
        run_bench_mtp_sample(&args);
        return;
    }

    // Tree-drafting Step-2.9 byte gate: twin-chain planted tree, branches must be bit-equal.
    if args.iter().any(|a| a == "--bench-tree") {
        run_bench_tree(&args);
        return;
    }

    // Batched-verify-across-lanes byte gate (LANES design Step 3a): pack independent lane chains into
    // one verify; each lane's logits must be bit-equal to running it alone.
    if args.iter().any(|a| a == "--bench-lanes") {
        run_bench_lanes(&args);
        return;
    }

    // Why is acceptance 39.5% on tool traffic and ~80% on prose? Weak head, or hard text?
    if args.iter().any(|a| a == "--bench-accept") {
        run_bench_accept(&args);
        return;
    }

    // MTP end-to-end speculative-decoding probe
    if args.iter().any(|a| a == "--bench-mtp") {
        run_bench_mtp(&args);
        return;
    }

    // CLI mode
    run_cli(&args);
}

/// Maps operator-facing CLI flags onto the env vars the engine reads internally (the
/// `--rdma-dev` pattern). Every session option is a CLI flag; the env vars remain as
/// back-compat aliases, and an explicit CLI flag always wins (it sets/removes the env
/// before anything reads it). Dev/debug probes stay env-only by design. Under TP the
/// head ships the resulting config to the node (TpConfig), so flags only need to exist
/// on the head/serve path and the node stays zero-config.
fn cli_env_bridge(args: &[String]) {
    fn set(args: &[String], flag: &str, var: &str) {
        if args.iter().any(|a| a == flag) { std::env::set_var(var, "1"); }
    }
    // TP feature flags.
    set(args, "--tp-fp32-partials", "GB10_TP_FP32_PARTIALS");
    set(args, "--tp-graph", "GB10_TP_GRAPH");
    set(args, "--tp-trace", "GB10_TP_TRACE");
    // Serving behavior.
    set(args, "--no-decode-graphs", "GB10_NO_DECODE_GRAPHS");
    set(args, "--no-gqpack", "GB10_NO_GQPACK");
    set(args, "--fuse-residual", "GB10_FUSE_RESIDUAL");
    set(args, "--cpu-sample", "RUST_INFER_CPU_SAMPLE");
    set(args, "--prefill-scalar", "RUST_INFER_PREFILL_SCALAR");
    set(args, "--zero-kv", "RUST_INFER_ZERO_KV");
    // KV cache format.
    if let Some(v) = parse_arg(args, "--kv-cache") {
        match v {
            "q4" => std::env::set_var("GB10_KV_QUANT", "1"),
            "bf16" => std::env::remove_var("GB10_KV_QUANT"),
            other => { eprintln!("--kv-cache must be bf16|q4 (got '{other}')"); std::process::exit(1); }
        }
    }
    // MTP: unify the bench-path GB10_TP_MTP env with the server CLI (`--mtp=on|off`, `--mtp-depth`).
    if let Some(v) = parse_arg(args, "--mtp") {
        match v {
            "on" => std::env::set_var("GB10_TP_MTP", "1"),
            "off" => std::env::remove_var("GB10_TP_MTP"),
            _ => {}   // auto: nothing to translate
        }
    }
    if let Some(d) = parse_arg(args, "--mtp-depth") { std::env::set_var("GB10_TP_MTP_DEPTH", d); }
    // FR-Spec draft vocabulary subset size (0 = full-vocab draft).
    if let Some(n) = parse_arg(args, "--draft-vocab") { std::env::set_var("RUST_INFER_DRAFT_VOCAB", n); }
    // Node blob cache dir.
    if let Some(d) = parse_arg(args, "--tp-cache") { std::env::set_var("GB10_TP_CACHE", d); }
    // Mixer sharding: DEFAULT ON under --tp/--head — Hy3 requires it, and it is where TP's speed
    // comes from on every model (halved mixer bytes + halved KV). --no-shard-mixers is the escape
    // hatch (and also wins over an inherited env, so a stale shell cannot surprise a bench).
    if args.iter().any(|a| a == "--no-shard-mixers") {
        std::env::remove_var("GB10_TP_SHARD_MIXERS");
    } else if args.iter().any(|a| a == "--tp" || a == "--head") {
        std::env::set_var("GB10_TP_SHARD_MIXERS", "1");
    }
}

fn parse_arg<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag || a.starts_with(&format!("{}=", flag)))
        .and_then(|i| {
            let arg = &args[i];
            if let Some(val) = arg.strip_prefix(&format!("{}=", flag)) {
                Some(val)
            } else {
                args.get(i + 1).map(|s| s.as_str())
            }
        })
}

fn run_cli(args: &[String]) {
    let model_path = parse_arg(args, "--model").unwrap_or("qwen3.5-0.8b-packed");
    let tokenizer_path = parse_arg(args, "--tokenizer").unwrap_or("model/tokenizer.json");
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("The capital of France is");
    let max_seq_len = parse_arg(args, "--max-seq-len").and_then(|s| s.parse::<usize>().ok()).unwrap_or(4096);
    let max_new_tokens = parse_arg(args, "--max-new-tokens").and_then(|s| s.parse::<usize>().ok()).unwrap_or(16);
    let temperature = parse_arg(args, "--temperature").and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0);

    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");

    let result = rt.block_on(async {
        let mut engine = GB10InferenceEngine::new(model_path, max_seq_len).await?;
        engine.set_sampler(gb10_inference::sampler::Sampler::new(temperature, 0.9, 50));

        let tokenizer = gb10_inference::tokenizer::QwenTokenizer::from_file(&tokenizer_path)?;

        println!("Engine initialized. Model: {}", model_path);
        println!("Prompt: {:?}", prompt_text);

        let prompt_tokens = tokenizer.encode(&prompt_text, true)?;
        println!("Prompt tokens ({}): {:?}", prompt_tokens.len(), &prompt_tokens);

        println!("Generating {} tokens...", max_new_tokens);
        let start = std::time::Instant::now();
        let output = engine.generate(&prompt_tokens, max_new_tokens);
        let elapsed = start.elapsed();

        let new_tokens = &output[prompt_tokens.len()..];
        println!("Generated {} tokens in {:?}", new_tokens.len(), elapsed);
        if elapsed.as_secs_f32() > 0.0 {
            println!("Throughput: {:.1} tok/s", new_tokens.len() as f32 / elapsed.as_secs_f32());
        }
        let text = tokenizer.decode(new_tokens, true).unwrap_or_default();
        println!("Output: {}", text);
        println!("Output token IDs: {:?}", new_tokens);

        Ok::<_, anyhow::Error>(())
    });

    match result {
        Ok(_) => println!("Done"),
        Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
    }
}

fn run_bench_prefill(args: &[String]) {
    let dir = parse_arg(args, "--model-dir").expect("--bench-prefill requires --model-dir");
    let seq_len: usize = parse_arg(args, "--seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);
    let reps: usize = parse_arg(args, "--reps").and_then(|s| s.parse().ok()).unwrap_or(5);
    let (gpu, _) = gb10_inference::gpu::GpuModel::load_from_dir(dir).expect("gpu load");
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

fn run_bench_batch(args: &[String]) {
    let (model_path, tokenizer_path) = if let Some(dir) = parse_arg(args, "--model-dir") {
        // Always pass the directory â Model::load handles directory vs file
        (dir.to_string(), format!("{}/tokenizer.json", dir.trim_end_matches('/')))
    } else {
        (parse_arg(args, "--model").unwrap_or("model/model.safetensors").to_string(),
         parse_arg(args, "--tokenizer").unwrap_or("model/tokenizer.json").to_string())
    };
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("The capital of France is");
    let m: usize = parse_arg(args, "--batch").and_then(|s| s.parse().ok()).unwrap_or(4);
    let max_new: usize = parse_arg(args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(32);
    let max_seq_len: usize = parse_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);

    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");
    println!("Batched benchmark: M={} seqs, prompt={} tokens, decode={} tokens", m, prompt.len(), max_new);

    let gpu = if std::path::Path::new(&model_path).is_dir() {
        let (gpu, _) = gb10_inference::gpu::GpuModel::load_from_dir(&model_path).expect("gpu load");
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
/// `--bench-tree --model-dir <d> [--prompt <p>] [--depth N]` â twin-chain planted-tree byte gate.
fn run_bench_tree(args: &[String]) {
    let dir = parse_arg(args, "--model-dir").expect("--bench-tree requires --model-dir");
    let tok_path = format!("{}/tokenizer.json", dir.trim_end_matches('/'));
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("The quick brown fox jumps over the lazy dog near the river bank at dawn.");
    let max_seq_len: usize = parse_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(8192);
    let tokenizer = QwenTokenizer::from_file(&tok_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");
    let (gpu, _) = gb10_inference::gpu::GpuModel::load_from_dir(dir).expect("gpu load");
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
        println!("\nRESULT: TREE_MISMATCH ({total_mism} diverged) â the tree verify is NOT ancestor-pure");
        std::process::exit(1);
    }
}

/// `--bench-lanes` — the batched-verify-across-lanes byte gate (LANES design Step 3a). Packs two
/// independent draft chains (one per lane, each rooted in its own committed slot state) into ONE verify
/// and asserts each lane's per-column logits are bit-equal to running that lane alone. Lanes share
/// committed length here (shared pos_start). Prints RESULT: LANES_OK or LANES_MISMATCH (exit 1).
fn run_bench_lanes(args: &[String]) {
    let dir = parse_arg(args, "--model-dir").expect("--bench-lanes requires --model-dir");
    let tok_path = format!("{}/tokenizer.json", dir.trim_end_matches('/'));
    let prompt_text = parse_arg(args, "--prompt").unwrap_or(
        "The quick brown fox jumps over the lazy dog near the river bank at dawn while the sun rises \
         slowly over distant hills and a light wind carries the smell of rain across the wide valley.");
    let max_seq_len: usize = parse_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(8192);
    let tokenizer = QwenTokenizer::from_file(&tok_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");
    let (gpu, _) = gb10_inference::gpu::GpuModel::load_from_dir(dir).expect("gpu load");
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
    if total_mism == 0 {
        println!("\nRESULT: LANES_OK ({total_cols} lane-columns; every lane's logits bit-identical packed \
                  vs alone -- GDN forest scan is lane-independent)");
    } else {
        println!("\nRESULT: LANES_MISMATCH ({total_mism} diverged) -- a lane's logits depend on its \
                  neighbours; the forest verify is NOT lane-pure");
        std::process::exit(1);
    }
}

fn run_bench_accept(args: &[String]) {
    let dir = parse_arg(args, "--model-dir").expect("--bench-accept requires --model-dir");
    let tok_path = format!("{}/tokenizer.json", dir.trim_end_matches('/'));
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("Write a short essay about the sea.");
    let depth: usize = parse_arg(args, "--depth").and_then(|s| s.parse().ok()).unwrap_or(8);
    let max_new: usize = parse_arg(args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(400);
    let max_seq_len: usize = parse_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(8192);
    let label = parse_arg(args, "--label").unwrap_or("workload");
    let ngram: usize = parse_arg(args, "--ngram").and_then(|s| s.parse().ok()).unwrap_or(0);

    let tokenizer = QwenTokenizer::from_file(&tok_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");
    let (gpu, _) = gb10_inference::gpu::GpuModel::load_from_dir(dir).expect("gpu load");
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    let mut state = gpu.new_batch_state(1, 2 + depth, max_seq_len);
    gpu.dev().synchronize().unwrap();

    let (s, generated) = gpu.bench_accept(&mut pool, &mut state, &prompt, max_seq_len, depth, max_new, ngram);
    assert!(!s.is_empty(), "bench_accept produced NO samples â it measured nothing, which is not a result");
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
    // `accepted` = target argmax â head top-1. `covered_top2/3` = target argmax â head top-2/3. The GAP
    // (covered_top2 â accepted) is the fraction of positions a k=2 fork would newly rescue. The review's
    // yield arithmetic keys the whole payoff on POSITION-1 coverage (câ), because a wrong position-1
    // guess gates the entire chain â so that row is the one that sets the Step-3 gate.
    println!("\n  FORK COVERAGE â target argmax in the head's top-k (this sets the yield target):");
    println!("    position    n     top1(=accept)   top2       top3      fork rescue (top2âtop1)");
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
        let mark = if d == 1 { "  <- câ (sets the gate)" } else { "" };
        println!("    pos {d:<2}    {:5}    {:5.1}%       {:5.1}%    {:5.1}%      +{:4.1}%{mark}",
                 b.len(), 100.0*t1, 100.0*t2, 100.0*t3, 100.0*(t2-t1));
    }
    let all: Vec<_> = s.iter().collect();
    let (t1, t2, t3) = cov(&all);
    println!("    ALL      {:5}    {:5.1}%       {:5.1}%    {:5.1}%      +{:4.1}%",
             all.len(), 100.0*t1, 100.0*t2, 100.0*t3, 100.0*(t2-t1));
    // Yield ceiling of a k=2 fork at position 1 (review Â§5.2): 1 + (câ/pâ)Â·A, A = mean accepted drafts.
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
        println!("    fork@1 yield CEILING (perfect top-2 rescue) â 1 + (câ/pâ)Â·A = 1 + ({:.2}/{:.2})Â·{:.2} = {:.2} tok/fwd",
                 c1, p1, a_drafts, ceiling);
        println!("    (review Â§5.2: realistic is BELOW this â rescues are conditioned on the head having just missed)");
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

    // N-GRAM RUN LENGTH (review Â§5.4): once a 3-gram match fires, how many CONSECUTIVE tokens does the
    // copy stay correct? A copy fires long. If the median run â¥ 3, the n-gram branch should get MORE
    // depth than the head branch (asymmetric tree: head chain d_a + n-gram chain m, 1 + (d_aâ1) + m â¤ 16).
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
        println!("    runs of length â¥3: {}/{} ({:.0}%)  â  {} the n-gram branch deeper than the head branch",
                 ge3, runs.len(), 100.0 * ge3 as f32 / runs.len() as f32,
                 if median >= 3 { "MAKE" } else { "do NOT make" });
    }
}

/// Compact acceptance discriminator, shared by --bench-accept and the TP=2 variant
/// (GB10_TP_ACCEPT in tp_serve): overall rate, the target-confidence buckets that separate
/// HARD TEXT (target unsure where the head misses) from a WEAK HEAD (misses even when the
/// target is confident), and top-2/3 fork coverage by draft depth.
fn bench_accept_report(depth: usize, s: &[gb10_inference::gpu::AcceptSample]) {
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

fn run_bench_verify(args: &[String]) {
    let (model_path, tokenizer_path) = if let Some(dir) = parse_arg(args, "--model-dir") {
        (dir.to_string(), format!("{}/tokenizer.json", dir.trim_end_matches('/')))
    } else {
        (parse_arg(args, "--model").unwrap_or("model/model.safetensors").to_string(),
         parse_arg(args, "--tokenizer").unwrap_or("model/tokenizer.json").to_string())
    };
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("The capital of France is");
    let depth: usize = parse_arg(args, "--depth").and_then(|s| s.parse().ok()).unwrap_or(4);
    let offset: usize = parse_arg(args, "--offset").and_then(|s| s.parse().ok()).unwrap_or(0);
    let max_seq_len: usize = parse_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);

    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");
    println!("MTP verify lossless probe: prompt={} tokens, offset={}, depth={}", prompt.len(), offset, depth);

    let gpu = if std::path::Path::new(&model_path).is_dir() {
        let (gpu, _) = gb10_inference::gpu::GpuModel::load_from_dir(&model_path).expect("gpu load");
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
    // A gate must be a loop, not a single run â the split-K bug passed most runs, because a 1-ulp
    // difference rarely flips an argmax (AGENTS.md Â§4.13). But one process per draw meant reloading a
    // 6 GB artifact 13 times: ~11 s of model load to run ~1 s of gate. Loading once turns a 3-minute
    // gate into a 20-second one, which is the difference between a gate you run and a gate you skip.
    //
    // Offsets straddle the 256-key split-K boundary on purpose: that is exactly where the shipped bug
    // lived, and no fixed context ever reached it.
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
            println!("RESULT: MISMATCH ({} draw(s) diverged) â MTP verify is NOT lossless", failed);
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
        println!("RESULT: MISMATCH ({} of {} positions diverged) â MTP verify is NOT lossless", nmismatch, depth);
        std::process::exit(1);
    }
}

/// GDN state-divergence probe: compare the recurrent s_state after ONE verify_forward(N tokens)
/// call vs N individual forward_decode calls. A zero diff means verify_forward and forward_decode
/// are numerically identical in their GDN state update; a nonzero diff at N=1 means a real kernel
/// bug, while nonzero only at N>=2 implicates cuBLAS batch-size-dependent bf16 rounding in the
/// projection GEMMs (decode uses N=1, verify uses N=K).
fn run_probe_state(args: &[String]) {
    let (model_path, tokenizer_path) = if let Some(dir) = parse_arg(args, "--model-dir") {
        (dir.to_string(), format!("{}/tokenizer.json", dir.trim_end_matches('/')))
    } else {
        (parse_arg(args, "--model").unwrap_or("model/model.safetensors").to_string(),
         parse_arg(args, "--tokenizer").unwrap_or("model/tokenizer.json").to_string())
    };
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("The capital of France is");
    let max_seq_len: usize = parse_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);

    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");
    println!("GDN state-divergence probe: prompt={} tokens", prompt.len());

    let gpu = if std::path::Path::new(&model_path).is_dir() {
        let (gpu, _) = gb10_inference::gpu::GpuModel::load_from_dir(&model_path).expect("gpu load");
        gpu
    } else {
        let host = gb10_inference::qwen::Model::load(&model_path).expect("load model");
        gb10_inference::gpu::GpuModel::new(&host).expect("gpu init")
    };
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    let mut state = gpu.new_batch_state(2, 2, max_seq_len);
    let kv_stride = max_seq_len;

    for n in 1..=2 {
        // Re-zero both slots and re-prefill fresh each iteration (verify_state_diff advances state).
        let diff = gpu.verify_state_diff(&mut pool, &mut state, &prompt, kv_stride, n);
        println!("N={}: max |s_state(verify) - s_state(decode)| = {:.7}  {}",
                 n, diff, if diff == 0.0 { "EXACT MATCH" } else { "DIVERGES" });
    }
}

/// Reject-path checkpoint/rollback three-way probe. Forces a rejection and checks whether the MTP
/// ping-pong snapshot (S1) and its D2D restore are bit-exact vs a single decode of the committed
/// token. Run with:  --probe-reject --model-dir 9b
fn run_probe_reject(args: &[String]) {
    let (model_path, tokenizer_path) = if let Some(dir) = parse_arg(args, "--model-dir") {
        (dir.to_string(), format!("{}/tokenizer.json", dir.trim_end_matches('/')))
    } else {
        (parse_arg(args, "--model").unwrap_or("model/model.safetensors").to_string(),
         parse_arg(args, "--tokenizer").unwrap_or("model/tokenizer.json").to_string())
    };
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("The capital of France is");
    let max_seq_len: usize = parse_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);

    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");
    println!("Reject-path probe: prompt={} tokens", prompt.len());

    let gpu = if std::path::Path::new(&model_path).is_dir() {
        let (gpu, _) = gb10_inference::gpu::GpuModel::load_from_dir(&model_path).expect("gpu load");
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

/// `--tp-barrier-bench` — adversarial proof of the doorbell all-reduce on the real transport, no model.
/// Run the SAME command on both boxes with `--rank 0` (head, listens) and `--rank 1 --peer <head-ip>`;
/// the barrier count and every mode flag must match, since the two ranks rendezvous per barrier.
fn run_tp_barrier_bench(args: &[String]) {
    let g = |k: &str, d: u64| parse_arg(args, k).and_then(|s| s.parse().ok()).unwrap_or(d);
    let a = gb10_inference::tp_bench::BenchArgs {
        rank: parse_arg(args, "--rank").and_then(|s| s.parse().ok())
            .expect("--tp-barrier-bench needs --rank 0|1"),
        peer: parse_arg(args, "--peer").unwrap_or("").to_string(),
        port: g("--port", 29600) as u16,
        dev: parse_arg(args, "--dev").unwrap_or("rocep1s0f1").to_string(),
        gid: g("--gid", 3) as i32,
        barriers: g("--barriers", 1_000_000),
        payload_bytes: g("--payload-bytes", 10240) as usize,   // 5120 bf16 = the 27B hidden vector
        spacing_us: g("--spacing-us", 0),
        inject_delay_us_max: g("--inject-delay-us-max", 0) as u32,
        poison: args.iter().any(|x| x == "--poison"),
        stall_every: g("--stall-consumer-every", 0) as u32,
        stall_us: g("--stall-us", 0),
        window: g("--window", 1024),
        proxy_core: g("--proxy-core", 19) as i32,
        main_core: g("--main-core", 9) as i32,
        cq_hold: g("--cq-hold", 0) as u32,
        cq_hold_us: g("--cq-hold-us", 50) as u32,
    };
    if let Err(e) = gb10_inference::tp_bench::run(a) {
        eprintln!("\n[tp-barrier-bench] FAILED: {e:#}");
        std::process::exit(1);
    }
}

/// G-A: TP=2 transport + FP32-partial numerical audit (design §2/§4, build step 1). Both ranks generate
/// the SAME deterministic contributions, each sums its K-half into an FP32 partial, exchanges over RDMA,
/// then rank 0 checks: (1) transport byte-exact, (2) FP32-partial reduce == single-node full-K reduce
/// (lossless), and how much WORSE a bf16-partial reduce is (the §4 hole FP32 partials close), (3) latency.
fn run_net_test(args: &[String]) {
    let rank: i32 = parse_arg(args, "--rank").and_then(|s| s.parse().ok())
        .expect("--net-test needs --rank 0|1");
    let peer = parse_arg(args, "--peer").unwrap_or("").to_string();
    let port: u16 = parse_arg(args, "--port").and_then(|s| s.parse().ok()).unwrap_or(23470);
    let dev = parse_arg(args, "--dev").unwrap_or("rocep1s0f1").to_string();
    let gid: i32 = parse_arg(args, "--gid").and_then(|s| s.parse().ok()).unwrap_or(3);

    const M: usize = 5120;   // 27B hidden -> payload 20480 B, the real all-reduce size
    const K: usize = 4096;   // contributions per output element, split K/2 per rank
    let half = K / 2;
    let k0 = (rank as usize) * half;   // this rank's K-range [k0, k0+half)

    // deterministic contribution in [-1,1), identical on both nodes (splitmix64-style hash)
    #[inline] fn contrib(i: usize, k: usize) -> f32 {
        let mut x = (i as u64).wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add((k as u64).wrapping_mul(0xD1B54A32D192ED03));
        x ^= x >> 29; x = x.wrapping_mul(0xBF58476D1CE4E5B9); x ^= x >> 32;
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    let slot = M * 4;   // FP32 payload bytes
    let mut link = gb10_inference::net::TpLink::connect(rank, &peer, port, &dev, gid, slot)
        .expect("TpLink::connect");

    let mut partial = vec![0f32; M];
    for i in 0..M { let mut s = 0f32; for k in k0..k0 + half { s += contrib(i, k); } partial[i] = s; }
    link.send_host_mut::<f32>(M).copy_from_slice(&partial);
    link.exchange(slot).expect("exchange");
    let peer_partial: Vec<f32> = link.recv_host::<f32>(M).to_vec();

    if rank != 0 {
        for _ in 0..2000 { link.exchange(slot).expect("exchange"); }   // responder for the latency loop
        println!("[node] rank 1 done.");
        return;
    }

    // (1) transport byte-exact: recv must equal the peer's independently-recomputable partial
    let mut peer_ref = vec![0f32; M];
    for i in 0..M { let mut s = 0f32; for k in half..K { s += contrib(i, k); } peer_ref[i] = s; }
    let tx_ok = peer_partial.iter().zip(&peer_ref).all(|(a, b)| a.to_bits() == b.to_bits());

    // (2) FP32-partial vs single-node full-K reference; and the bf16-partial (lossy) path
    let b16 = |x: f32| half::bf16::from_f32(x).to_f32();
    let (mut fp32_mm, mut bf16_mm) = (0usize, 0usize);
    let (mut fp32_max, mut bf16_max) = (0f32, 0f32);
    for i in 0..M {
        let mut refi = 0f32; for k in 0..K { refi += contrib(i, k); }   // single-node full-K FP32
        let refb   = b16(refi);
        let fp32tp = b16(partial[i] + peer_partial[i]);              // FP32 partials -> one round
        let bf16tp = b16(b16(partial[i]) + b16(peer_partial[i]));    // bf16 partials -> extra rounding
        if fp32tp.to_bits() != refb.to_bits() { fp32_mm += 1; }
        if bf16tp.to_bits() != refb.to_bits() { bf16_mm += 1; }
        fp32_max = fp32_max.max((fp32tp - refb).abs());
        bf16_max = bf16_max.max((bf16tp - refb).abs());
    }

    // (3) exchange latency
    let n = 2000usize;
    let t0 = std::time::Instant::now();
    for _ in 0..n { link.exchange(slot).expect("exchange"); }
    let us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;

    println!("=== G-A: TP=2 transport + FP32-partial audit (M={M}, payload {slot} B, K={K}) ===");
    println!("  (1) transport byte-exact (recv == peer partial): {}", if tx_ok { "YES" } else { "NO  ***FAIL***" });
    println!("  (2) FP32-partial reduce vs single-node: {fp32_mm}/{M} bf16 mismatches, max|Δ|={fp32_max:.3e}");
    println!("      bf16-partial reduce vs single-node: {bf16_mm}/{M} bf16 mismatches, max|Δ|={bf16_max:.3e}  <- the hole FP32 partials close");
    println!("  (3) exchange latency: {us:.2} us/op   (x128/token = {:.2} ms)", us * 128.0 / 1000.0);
}

/// TP=2 cluster NODE: answer discovery + accept one head sync, bring up the RDMA link, then join the
/// SPMD masked-replicated decode (Proof v0). The node loads the synced model, receives the prompt from
/// the head over the link, and runs the IDENTICAL greedy generate loop — the per-layer FFN all-reduces
/// keep it bit-for-bit in step with the head, which owns the printing.
fn run_cluster_node(args: &[String]) {
    if let Some(d) = parse_arg(args, "--rdma-dev") { std::env::set_var("GB10_RDMA_DEV", d); }
    let port: u16 = parse_arg(args, "--port").and_then(|s| s.parse().ok()).unwrap_or(29500);

    // RESIDENT by default (TP item B): supervise one-shot sessions, re-arming after each so the node
    // survives a head restart with zero manual intervention. One process per session is the whole
    // isolation story: the process-global TP config, the mem::forget'd link + proxy thread, and the
    // freshly-sharded weights (attach_tp shards in place — a model cannot be re-attached) all die
    // with the child, so the next head sync starts from a provably clean process. `--once` (or the
    // GB10_NODE_CHILD marker the supervisor sets on its children) runs a single session, for
    // debugging. A graceful per-request serve loop (no reload between requests) is item A's server
    // mode; until then a session = one sync + one SPMD run.
    let once = args.iter().any(|a| a == "--once") || std::env::var("GB10_NODE_CHILD").is_ok();
    if once { run_cluster_node_once(port); return; }

    let exe = std::env::current_exe().expect("current_exe");
    eprintln!("[node-resident] supervisor up on port {port} — one process per head session; \
               kill this process to stop the node");
    loop {
        let t0 = std::time::Instant::now();
        let status = std::process::Command::new(&exe)
            .args(["--node", "--once", "--port", &port.to_string()])
            .env("GB10_NODE_CHILD", "1")
            .status();
        match status {
            Ok(s) => eprintln!("[node-resident] session ended ({s}) after {:.1}s — re-arming for the next head",
                               t0.elapsed().as_secs_f64()),
            Err(e) => eprintln!("[node-resident] failed to spawn a session process: {e} — retrying"),
        }
        // A session that dies instantly (bind failure, RDMA down) must not spin-fork.
        if t0.elapsed().as_secs() < 5 { std::thread::sleep(std::time::Duration::from_secs(1)); }
    }
}

/// One head session: sync (model + config) -> bring up the RDMA link -> serve one SPMD run -> exit.
fn run_cluster_node_once(port: u16) {
    let (dir, head_ip, tpc, stream) = match gb10_inference::cluster::run_node(port) {
        Ok(x) => x,
        Err(e) => { eprintln!("node error: {e:#}"); std::process::exit(1); }
    };
    // The head's TP config, shipped during the sync — install BEFORE any TP consumer reads a setting,
    // so the node reproduces the head's behavior with ZERO GB10_TP_* env vars.
    gb10_inference::tp::set_tp_config(tpc.clone());
    println!("NODE SYNCED — model ready at {}", dir.display());
    let r = if tpc.mode_serve {
        // Serving session: mirror the head's OpenAI-server BatchScheduler in SPMD lockstep over the
        // retained control stream (TP item A). Returning Ok ends the session; the resident
        // supervisor re-arms for the next head.
        node_serve_tp(&dir, head_ip, stream, &tpc)
    } else {
        // Bench session: one-shot SPMD bench/generate (unchanged). The retained stream is dropped.
        drop(stream);
        tp_serve(&dir.to_string_lossy(), gb10_inference::tp::TpContext::bring_up_node(head_ip), None)
    };
    if let Err(e) = r {
        eprintln!("node tp serve error: {e:#}"); std::process::exit(1);
    }
}

/// TP=2 serving NODE (TP item A): bring up the RDMA link, load the synced model, attach TP, then
/// mirror the head's BatchScheduler. All admissions/cancels arrive as per-step events on the
/// retained sync stream; the node runs identical scheduler state and discards its TokEvents.
fn node_serve_tp(dir: &std::path::Path, head_ip: std::net::IpAddr, mut stream: std::net::TcpStream,
                 tpc: &gb10_inference::tp::TpConfig) -> anyhow::Result<()> {
    use gb10_inference::tp_serve::{recv_serving, send_serving, ServingMsg};
    let mut ctx = gb10_inference::tp::TpContext::bring_up_node(head_ip)?;
    ctx.sanity()?;
    println!("NODE (rank 1/2) — TP LINK UP (serving mode)");

    // These are read as ENV at model LOAD (GB10_KV_QUANT selects the 4-bit KV cache layout and the
    // q4 attention path at GpuModel::load_from_dir_tp) and inside BatchScheduler::new (graph capture,
    // gpu-sample probes). The head ships its values in the config and the node installs them
    // process-wide BEFORE THE LOAD — this used to sit below load_from_dir_tp, so a serve-mode node
    // silently built a bf16 KV cache + bf16 per-head attention while the head ran q4: the node
    // became the straggler (the "32K anomaly" — 34.6 GB/token of bf16 per-head re-reads at 26K)
    // and every serve-mode "q4" number was really a mixed q4-head/bf16-node number.
    if tpc.no_decode_graphs { std::env::set_var("GB10_NO_DECODE_GRAPHS", "1"); }
    if tpc.cpu_sample { std::env::set_var("RUST_INFER_CPU_SAMPLE", "1"); }
    // The 4-bit KV cache must match on BOTH ranks (the caches are all-reduced-consistent).
    if tpc.kv_quant { std::env::set_var("GB10_KV_QUANT", "1"); }

    let (mut gpu, _cfg) = gb10_inference::gpu::GpuModel::load_from_dir_tp(&dir.to_string_lossy(), ctx.rank)?;
    let (rank, world, link) = ctx.into_parts();
    gpu.attach_tp(rank, world, link);

    // Pin the decode/launch thread to a big X925 core AFTER CUDA init — same rule as tp_serve: an
    // unpinned launch thread presents exactly like a protocol stall, so a pin failure is loud.
    if world == 2 && !gb10_inference::net::pin_thread(9) {
        panic!("FATAL: launch thread failed to pin to core 9 — TP refuses to run unpinned");
    }

    // SPMD calibration. The node MUST execute the identical forward sequence — the all-reduces are
    // barriers the head waits on — but DISCARDS its table: both ranks drive MtpPolicy from the
    // head's numbers (shipped next as CalibTable), so the policy state cannot diverge. Skipped
    // exactly when the head skips it (same model head-presence, same --mtp force in the config).
    if gpu.mtp_present() && tpc.mtp_force != Some(false) {
        println!("NODE — SPMD MTP calibration (table discarded; head's is shipped)...");
        let mut cpool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
        let calib_seq = 1024usize.min(tpc.max_seq_len);
        // Same sizing as the head's calibration state (main.rs run_server): slots 2..=PROFILE_MAX_N
        // receive one GDN checkpoint per verify column.
        let mut cstate = gpu.new_batch_state(2 + gb10_inference::gpu::PROFILE_MAX_N,
                                             2 + gb10_inference::gpu::PROFILE_MAX_N, calib_seq);
        let _ = gpu.calibrate_mtp_r(&mut cpool, &mut cstate, &tpc.calib_prompt, calib_seq);
    }
    let head_r: Vec<(usize, f32)> = match recv_serving(&mut stream)? {
        ServingMsg::CalibTable { r } => r.into_iter().map(|(d, r)| (d as usize, r)).collect(),
        other => anyhow::bail!("expected CalibTable from head, got {other:?}"),
    };
    println!("NODE — MTP cost table from head ({} depths)", head_r.len());
    let policy = gb10_inference::batch::MtpPolicy::new(
        gpu.mtp_present(), tpc.mtp_force, tpc.mtp_depth_pin, head_r);
    let (_stx, srx) = tokio::sync::mpsc::unbounded_channel::<gb10_inference::batch::BatchRequest>();
    let scheduler = gb10_inference::batch::BatchScheduler::new(
        gpu, tpc.max_batch, tpc.max_seq_len, tpc.eos.clone(), srx, policy,
        tpc.prefix_cache, tpc.ngram_draft, tpc.tree_draft, tpc.mtp_lanes);
    send_serving(&mut stream, &ServingMsg::Ready)?;
    println!("NODE — READY; entering SPMD mirror loop");
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    rt.block_on(scheduler.run_tp_mirror(stream))
}

/// TP=2 cluster HEAD: discover node(s) (or explicit --nodes) and push the model with content-addressed
/// caching (a node that already has the artifacts transfers nothing), then drive the SPMD masked-
/// replicated decode (Proof v0): broadcast the prompt to the node, run the identical generate loop, and
/// print the coherent output. `--prompt` / `--max-new-tokens` set the request.
fn run_cluster_head(args: &[String]) {
    if let Some(d) = parse_arg(args, "--rdma-dev") { std::env::set_var("GB10_RDMA_DEV", d); }
    let model_dir = parse_arg(args, "--model-dir").expect("--head requires --model-dir <DIR>").to_string();
    let prompt_text = match parse_arg(args, "--prompt-file") {
        Some(f) => std::fs::read_to_string(f).expect("read --prompt-file"),   // verbatim — `$(cat)` strips trailing newlines and changes the last token
        None => parse_arg(args, "--prompt").unwrap_or("The capital of France is").to_string(),
    };
    let max_new: usize = parse_arg(args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(64);
    let explicit = parse_arg(args, "--nodes").map(|s| {
        s.split(',').map(|p| {
            let p = p.trim();
            if p.contains(':') { p.parse::<std::net::SocketAddr>().expect("bad --nodes addr (ip:port)") }
            else { std::net::SocketAddr::new(p.parse::<std::net::IpAddr>().expect("bad --nodes ip"), 29500) }
        }).collect::<Vec<_>>()
    });
    let wait = std::time::Duration::from_secs(
        parse_arg(args, "--discover-wait").and_then(|s| s.parse().ok()).unwrap_or(3));
    // Snapshot our GB10_TP_* env as THE config, install it process-globally, and ship it to every
    // node during the sync (nodes run with zero TP env and reproduce this behavior).
    let tpc = gb10_inference::tp::TpConfig::from_env();
    gb10_inference::tp::set_tp_config(tpc.clone());
    match gb10_inference::cluster::run_head(std::path::Path::new(&model_dir), explicit, wait, &tpc) {
        Ok(nodes) => println!("HEAD SYNCED {} node(s)", nodes.len()),
        Err(e) => { eprintln!("head error: {e:#}"); std::process::exit(1); }
    }
    if let Err(e) = tp_serve(&model_dir, gb10_inference::tp::TpContext::bring_up_head(),
                             Some((prompt_text, max_new))) {
        eprintln!("head tp serve error: {e:#}"); std::process::exit(1);
    }
}

/// Shared TP=2 Proof-v0 serve path for both roles: sanity-check the link, broadcast/receive the
/// prompt, load the model + attach the TP link, run the SPMD `tp_generate`, and (head only) decode+print.
fn tp_serve(model_dir: &str, ctx: anyhow::Result<gb10_inference::tp::TpContext>,
            head_req: Option<(String, usize)>) -> anyhow::Result<()> {
    let is_head = head_req.is_some();
    let role = if is_head { "HEAD (rank 0/2)" } else { "NODE (rank 1/2)" };
    let mut ctx = ctx?;
    ctx.sanity()?;
    println!("{role} — TP LINK UP");

    // Head encodes the prompt; both ranks agree on (prompt ids, max_new) via the link broadcast.
    let tok_path = format!("{}/tokenizer.json", model_dir.trim_end_matches('/'));
    let head_payload = match &head_req {
        Some((text, max_new)) => {
            let tokenizer = QwenTokenizer::from_file(&tok_path)?;
            let ids = tokenizer.encode(text, true)?;
            println!("{role} — prompt {text:?} → {} tokens", ids.len());
            Some((ids, *max_new))
        }
        None => None,
    };
    let (prompt, max_new) = ctx.broadcast_prompt(
        head_payload.as_ref().map(|(ids, m)| (ids.as_slice(), *m)))?;
    println!("{role} — SPMD decode: {} prompt tokens, max_new {max_new}", prompt.len());

    // Node-side TP env that must agree with the head before load (the 4-bit KV cache changes the
    // cache layout on BOTH ranks; the head ships it in the TpConfig).
    if gb10_inference::tp::tp_config().map(|c| c.kv_quant).unwrap_or(false) {
        std::env::set_var("GB10_KV_QUANT", "1");
    }

    // Load the (whole, replicated) model and attach the TP link → the forward runs the sharded FFN
    // all-reduce (world==2). Same binary/model on both boxes, so the compute is identical. (hy_v3:
    // the loader shards to ctx.rank host-side — the full model does not fit one node.)
    let (mut gpu, _cfg) = gb10_inference::gpu::GpuModel::load_from_dir_tp(model_dir, ctx.rank)?;
    let (rank, world, link) = ctx.into_parts();
    gpu.attach_tp(rank, world, link);

    // Pin the decode/launch thread to a big X925 core (GB10 big.LITTLE) AFTER CUDA init, so CUDA's
    // helper threads keep their own affinity (pinning before init would make them inherit this mask and
    // contend). ~320 launches/token; a launch thread parked on an A725 drains the GPU stream mid-token.
    // Proxy is on core (ncpu-1)=19; keep the launch thread on a different big core (9).
    if world == 2 && !gb10_inference::net::pin_thread(9) {
        // Pinning is the measurement, not a preference (proxy pin was 9.0 -> 15.1 tok/s; an unpinned
        // launch thread presents exactly like a protocol stall). Fail loudly, same rule as the proxy.
        panic!("FATAL: launch thread failed to pin to core 9 — TP refuses to run unpinned");
    }

    // TP-aware per-layer capture (hy_v3 oracle localization): BOTH ranks run the identical batched
    // prefill in SPMD lockstep (the all-reduces fire inside), and each writes its own dump — the two
    // files must be bit-identical to each other (SPMD check) and comparable to the oracle per layer
    // per position (scripts/compare_hy3_oracle.py). `GB10_TP_CAPTURE=<out.safetensors>`.
    if let Ok(cap_out) = std::env::var("GB10_TP_CAPTURE") {
        use safetensors::{Dtype, tensor::TensorView};
        let n = prompt.len();
        let h = gpu.cfg().hidden_size;
        let nlayers = gpu.cfg().num_layers;
        let kv_stride = n.max(16);
        let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
        let mut state = gpu.new_batch_state(1, 1, kv_stride);
        let dumps = gpu.capture_prefill(&mut pool, &prompt, &mut state, kv_stride);
        // GB10_CAP_DEBUG interleaves mixer/mlp_out dumps per layer (count check only without it).
        if std::env::var("GB10_CAP_DEBUG").is_err() {
            assert_eq!(dumps.len(), nlayers + 2, "capture count: embed + L layers + final_norm");
        }
        let mut named: Vec<(String, Vec<half::bf16>)> = Vec::with_capacity(dumps.len());
        let dbg = std::env::var("GB10_CAP_DEBUG").is_ok();
        for (i, dmp) in dumps.into_iter().enumerate() {
            // Debug mode: mixer/mlp_out dumps are interleaved per layer — name by index (unique).
            let name = if dbg { format!("dump.{i:03}") }
                       else if i == 0 { "layer.00.in".to_string() }
                       else if i <= nlayers { format!("layer.{:02}.out", i - 1) }
                       else { "final_norm".to_string() };
            named.push((name, dmp));
        }
        let views: Vec<(String, TensorView)> = named.iter()
            .map(|(name, dmp)| {
                let bytes: &[u8] = bytemuck::cast_slice(&dmp[..]);
                (name.clone(), TensorView::new(Dtype::BF16, vec![n, h], bytes).expect("view"))
            }).collect();
        safetensors::serialize_to_file(views, None, std::path::Path::new(&cap_out)).expect("write safetensors");
        println!("{role} — TP capture: {n} tokens x {nlayers} layers -> {cap_out}");
        return Ok(());
    }

    let max_seq_len = (prompt.len() + max_new + 16).next_power_of_two().max(256);

    // TP=2 acceptance diagnosis (GB10_TP_ACCEPT=<depth>): both ranks run bench_accept in SPMD
    // (the main forwards barrier; the replicated drafter cannot diverge), rank 0 prints the
    // discriminator report. This is the instrument for "is the draft head weak or is the text hard".
    if let Ok(depth_s) = std::env::var("GB10_TP_ACCEPT") {
        let depth: usize = depth_s.parse().unwrap_or(2);
        let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
        let mut state = gpu.new_batch_state(1, 2 + depth, max_seq_len);
        let (s, generated) = gpu.bench_accept(&mut pool, &mut state, &prompt, max_seq_len, depth, max_new, 0);
        if is_head {
            assert!(!s.is_empty(), "bench_accept produced NO samples");
            assert!(generated.len() > 8, "bench_accept generated almost nothing");
            bench_accept_report(depth, &s);
        }
        return Ok(());
    }

    // Q6 probe: does "a batch-N forward costs ~= a batch-1 forward" survive TP? Runs INSTEAD of decode.
    // v1 MTP under TP: run the real speculative loop on the sharded model.
    // Every TP bench route resolves env-first (override), then the installed TP config (shipped by the
    // head during the sync), then the same default as the no-env behavior.
    let tpc = gb10_inference::tp::tp_config();
    if std::env::var("GB10_TP_MTP").is_ok() || tpc.map(|c| c.mtp).unwrap_or(false) {
        let depth: usize = std::env::var("GB10_TP_MTP_DEPTH").ok()
            .and_then(|v| v.parse().ok())
            .or(tpc.and_then(|c| c.mtp_depth)).unwrap_or(4);
        let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
        let slots = 2 + depth.saturating_sub(1).max(1);
        let mut st = gpu.new_batch_state(slots, slots, max_seq_len);
        let (mtp_toks, seq_toks, mtp_tps, seq_tps, acc) =
            gpu.bench_mtp(&mut pool, &mut st, &prompt, max_seq_len, depth, max_new);
        let lossless = mtp_toks == seq_toks;
        println!("{role} — TP+MTP depth {depth}: {:.1} tok/s (sequential {:.1}), acceptance {:.1}%, {}",
                 mtp_tps, seq_tps, acc * 100.0,
                 if lossless { "LOSSLESS_OK" } else { "DIVERGED vs sequential" });
        if is_head {
            println!("GATE_TOKENS {}", mtp_toks.iter().map(|t| t.to_string())
                     .collect::<Vec<_>>().join(","));
        }
        return Ok(());
    }

    // Probe 2: measure a synthetic MTP step under TP directly.
    let step_probe = match std::env::var("GB10_TP_STEP_PROBE") {
        Ok(d) => Some(d.parse().unwrap_or(4)),
        Err(_) => tpc.and_then(|c| c.step_probe),
    };
    if let Some(d) = step_probe {
        gpu.tp_synthetic_step_probe(d, 20, max_seq_len);
        return Ok(());
    }

    let batch_probe = match std::env::var("GB10_TP_BATCH_PROBE") {
        Ok(n) => Some(n.parse().unwrap_or(1)),
        Err(_) => tpc.and_then(|c| c.batch_probe),
    };
    if let Some(n) = batch_probe {
        gpu.tp_batch_probe(n, 30, max_seq_len);
        return Ok(());
    }
    let t0 = std::time::Instant::now();
    let out = gpu.tp_generate(&prompt, max_new, max_seq_len);
    let dt = t0.elapsed();
    gpu.tp_trace_dump(role);

    if is_head {
        let tokenizer = QwenTokenizer::from_file(&tok_path)?;
        let text = tokenizer.decode(&out, true).unwrap_or_default();
        let tps = if dt.as_secs_f32() > 0.0 { out.len() as f32 / dt.as_secs_f32() } else { 0.0 };
        println!("\n===== TP=2 PROOF v0 OUTPUT ({} tokens, {:.1} tok/s) =====", out.len(), tps);
        println!("{text}");
        println!("===== token ids: {:?}", out);
        println!("GATE_TOKENS {}", out.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
    } else {
        println!("{role} — generated {} tokens in lockstep (head prints)", out.len());
    }
    Ok(())
}

/// GEMM batch-invariance probe: directly measures whether cuBLAS bf16 GEMM gives identical results
/// for the first column at N=1 vs N=2, for the model's key GEMM shapes. Run with:
///   --probe-gemm --model-dir 9b          (tests the model's hidden/intermediate/conv_dim shapes)
fn run_probe_gemm(args: &[String]) {
    let model_dir = parse_arg(args, "--model-dir").map(|s| s.to_string());
    let gpu: gb10_inference::gpu::GpuModel = if let Some(dir) = model_dir {
        let (g, _) = gb10_inference::gpu::GpuModel::load_from_dir(&dir).expect("gpu load");
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
fn run_sweep_gemm(args: &[String]) {
    let model_dir = parse_arg(args, "--model-dir").expect("--sweep-gemm requires --model-dir <DIR>");
    let (gpu, _) = gb10_inference::gpu::GpuModel::load_from_dir(model_dir).expect("gpu load");
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

/// MTP end-to-end probe: runs full speculative decoding (draft â verify â accept â rollback) and
/// checks the output is token-for-token identical to sequential greedy (lossless), while reporting
/// the acceptance rate and speedup vs sequential.
/// `--quantize --model-dir <in> --out <dir> --recipe <spec>` â the offline quantizer.
///
/// Emits **compressed-tensors** layout, byte-compatible with HF (so our artifacts and theirs are
/// mutually loadable â and so the format is one we already validated against a real checkpoint):
///
/// ```text
///   NVFP4:  {name}.weight_packed        U8       [M, K/2]   nibble-packed E2M1
///           {name}.weight_scale         F8_E4M3  [M, K/16]  block scales
///           {name}.weight_global_scale  F32      [1]        (6*448)/amax â DIVIDE on dequant
///   FP8:    {name}.weight               F8_E4M3  [M, K]
///           {name}.weight_scale         F32      [M]        one per output row
///   else:   {name}.weight               copied through unchanged (norms, conv1d, A_log, dt_biasâ¦)
/// ```
///
/// Recipe is the same syntax as the fake-quant knob, e.g. `all`, `all,gdn:fp8`, `all:fp8`.
/// Measured on 9B (held-out prose+code, bf16 PPL 7.622): `all` â 8.332 (+9.3%),
/// `all,gdn:fp8` â 8.036 (+5.4%), `all:fp8` â 7.673 (+0.7%).
fn run_quantize(args: &[String]) {
    use safetensors::{SafeTensors, Dtype, tensor::TensorView};
    use gb10_inference::quant::{self, Fmt};

    let in_dir = parse_arg(args, "--model-dir").expect("--model-dir <in> required");
    let out_dir = parse_arg(args, "--out").expect("--out <dir> required");
    let recipe_s = parse_arg(args, "--recipe").unwrap_or("all,gdn:fp8");
    let recipe = quant::parse_recipe(recipe_s).expect("empty recipe");

    let ind = std::path::Path::new(in_dir);
    let outd = std::path::Path::new(out_dir);
    std::fs::create_dir_all(outd).expect("create --out dir");

    println!("Quantizing {} -> {}", in_dir, out_dir);
    println!("  recipe: {}", recipe_s);
    for (g, f) in &recipe {
        println!("    {:<8} -> {}", quant::group_name(*g), quant::fmt_name(*f));
    }

    // Shards, in index order when there is an index.
    let index_path = ind.join("model.safetensors.index.json");
    let shards: Vec<std::path::PathBuf> = if index_path.exists() {
        let raw = std::fs::read_to_string(&index_path).expect("read index");
        let idx: serde_json::Value = serde_json::from_str(&raw).expect("parse index");
        idx["weight_map"].as_object().unwrap().values()
            .filter_map(|v| v.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter().map(|s| ind.join(s)).collect()
    } else {
        std::fs::read_dir(ind).unwrap().filter_map(|e| {
            let e = e.ok()?;
            let n = e.file_name().to_string_lossy().to_string();
            if n.ends_with(".safetensors") { Some(e.path()) } else { None }
        }).collect()
    };

    // Owned output buffers (TensorView borrows, so these must outlive serialization).
    struct Out { name: String, dtype: Dtype, shape: Vec<usize>, data: Vec<u8> }
    let mut outs: Vec<Out> = Vec::new();
    let (mut n_q4, mut n_q8, mut n_copy) = (0usize, 0usize, 0usize);
    let (mut bytes_in, mut bytes_out) = (0u64, 0u64);
    let t0 = std::time::Instant::now();

    // Quantize-or-copy ONE tensor into `outs`, honoring the recipe. Used both inline in the shard loop
    // and for the SYNTHESIZED fused MTP-expert stacks below, so both paths are byte-identical.
    // 3D experts [E,M,K] flatten to [E*M,K] (the per-16-block E4M3 scales absorb per-expert magnitude
    // variation; M is 16-aligned so a block never straddles two experts). Vision tower stays bf16.
    let emit = |name: String, dtype: Dtype, shape: Vec<usize>, data: &[u8],
                outs: &mut Vec<Out>, n_q4: &mut usize, n_q8: &mut usize, n_copy: &mut usize,
                bytes_out: &mut u64| {
        let fmt = quant::fmt_for(&recipe, &name);
        let last = *shape.last().unwrap_or(&0);
        let m_eff = if shape.len() == 3 { shape[0] * shape[1] }
                    else if shape.len() == 2 { shape[0] } else { 0 };
        let quantizable = fmt != Fmt::Bf16
            && (shape.len() == 2 || shape.len() == 3)
            && dtype == Dtype::BF16
            && last % quant::BLOCK == 0
            && m_eff % 16 == 0
            && !name.contains(".visual.");
        if !quantizable {
            // Copy through unchanged: norms, conv1d, A_log, dt_bias, router (if bf16), vision, bf16 head.
            outs.push(Out { name, dtype, shape, data: data.to_vec() });
            *n_copy += 1;
            *bytes_out += data.len() as u64;
            return;
        }
        let (m, k) = (m_eff, last);
        let w: &[half::bf16] = bytemuck::cast_slice(data);
        let stem = name.strip_suffix(".weight").unwrap_or(&name).to_string();
        match fmt {
            Fmt::Nvfp4 => {
                let q = quant::quantize_nvfp4(w, m, k);
                *bytes_out += (q.qweight.len() + q.scales.len() + 4) as u64;
                outs.push(Out { name: format!("{}.weight_packed", stem), dtype: Dtype::U8,
                                shape: vec![m, k / 2], data: q.qweight });
                outs.push(Out { name: format!("{}.weight_scale", stem), dtype: Dtype::F8_E4M3,
                                shape: vec![m, k / quant::BLOCK], data: q.scales });
                outs.push(Out { name: format!("{}.weight_global_scale", stem), dtype: Dtype::F32,
                                shape: vec![1], data: q.global_scale.to_le_bytes().to_vec() });
                *n_q4 += 1;
            }
            Fmt::Fp8 => {
                let q = quant::quantize_fp8(w, m, k);
                let sc: Vec<u8> = q.row_scale.iter().flat_map(|f| f.to_le_bytes()).collect();
                *bytes_out += (q.qweight.len() + sc.len()) as u64;
                outs.push(Out { name: format!("{}.weight", stem), dtype: Dtype::F8_E4M3,
                                shape: vec![m, k], data: q.qweight });
                outs.push(Out { name: format!("{}.weight_scale", stem), dtype: Dtype::F32,
                                shape: vec![m], data: sc });
                *n_q8 += 1;
            }
            Fmt::Bf16 => unreachable!(),
        }
    };

    // Some checkpoints store routed experts UN-FUSED and per-expert (`...mlp.experts.<i>.{gate,up,down}_proj.weight`):
    // the 122B's MTP head, and Hy3 (hy_v3) for EVERY MoE layer (79 + the layer-80 MTP block). Our
    // loader/kernels only ingest the fused layout (`...mlp.experts.gate_up_proj` / `.down_proj`), so FUSE
    // per-expert tensors before quantizing: stash keyed by (base, proj) with a BTreeMap over the INTEGER
    // expert index (a lexical sort would permute 0,1,10,100,… → silent wrong experts). Fusion happens
    // ON-COMPLETE (gate/up/down each holding a contiguous 0..E set) so a 295B model's stash never exceeds
    // ~1 layer of experts in RAM; anything left unfused at the end is a loud error, not silent garbage.
    type ExpertMap = std::collections::BTreeMap<usize, (Vec<usize>, Vec<u8>)>;
    type Pending = std::collections::BTreeMap<String, std::collections::BTreeMap<String, ExpertMap>>;
    let mut pending: Pending = std::collections::BTreeMap::new();
    let n_experts: Option<usize> = std::fs::read_to_string(ind.join("config.json")).ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|c| c.get("num_experts").or_else(|| c.get("n_routed_experts")).and_then(|v| v.as_u64()))
        .map(|v| v as usize);
    let parse_expert = |name: &str| -> Option<(String, String, usize)> {
        let stem = name.strip_suffix(".weight")?;
        let (base_prefix, tail) = stem.split_once(".experts.")?;   // "...layers.L.mlp" , "<i>.<proj>"
        let (idx_s, proj) = tail.split_once('.')?;
        if proj != "gate_proj" && proj != "up_proj" && proj != "down_proj" { return None; }
        let idx: usize = idx_s.parse().ok()?;
        Some((format!("{}.experts", base_prefix), proj.to_string(), idx))
    };
    let complete = |projs: &std::collections::BTreeMap<String, ExpertMap>, e: usize| -> bool {
        ["gate_proj", "up_proj", "down_proj"].iter().all(|p| projs.get(*p)
            .map(|m| m.len() == e && (0..e).all(|i| m.contains_key(&i))).unwrap_or(false))
    };
    // Fuse one base's per-expert tensors into the stacked layout the loader expects, then run it through
    // the SAME `emit` (quantize or copy-through per recipe). Ordering is load-bearing in BOTH dims:
    // gate_up = concat([gate, up], dim=0); experts stacked in INTEGER order.
    let fuse_base = |base: &String, projs: &std::collections::BTreeMap<String, ExpertMap>,
                     outs: &mut Vec<Out>, n_q4: &mut usize, n_q8: &mut usize, n_copy: &mut usize,
                     bytes_out: &mut u64| {
        let gate = projs.get("gate_proj").unwrap_or_else(|| panic!("{base}: no gate_proj experts"));
        let up   = projs.get("up_proj").unwrap_or_else(|| panic!("{base}: no up_proj experts"));
        let down = projs.get("down_proj").unwrap_or_else(|| panic!("{base}: no down_proj experts"));
        let e = gate.len();
        // Every projection must have the SAME contiguous 0..E expert set (assert one per expert).
        for (nm, m) in [("gate", gate), ("up", up), ("down", down)] {
            assert_eq!(m.len(), e, "{base}: {nm} has {} experts, gate has {e}", m.len());
            for i in 0..e { assert!(m.contains_key(&i), "{base}: {nm} missing expert {i}"); }
        }
        let (inter, hidden) = { let s = &gate[&0].0; assert_eq!(s.len(), 2, "gate not 2-D"); (s[0], s[1]) };
        // Shape guard — catches a transposed/permuted export before it becomes silent garbage.
        for i in 0..e {
            assert_eq!(gate[&i].0, vec![inter, hidden], "{base}: gate[{i}] shape");
            assert_eq!(up[&i].0,   vec![inter, hidden], "{base}: up[{i}] shape");
            assert_eq!(down[&i].0, vec![hidden, inter], "{base}: down[{i}] shape");
        }
        // gate_up[e] = concat([gate[e], up[e]], dim=0) → [2·inter, hidden]; stack over e → [E,2·inter,hidden].
        let mut gu = Vec::<u8>::with_capacity(e * 2 * inter * hidden * 2);
        for i in 0..e { gu.extend_from_slice(&gate[&i].1); gu.extend_from_slice(&up[&i].1); }
        emit(format!("{base}.gate_up_proj"), Dtype::BF16, vec![e, 2 * inter, hidden], &gu,
             outs, n_q4, n_q8, n_copy, bytes_out);
        let mut dn = Vec::<u8>::with_capacity(e * hidden * inter * 2);
        for i in 0..e { dn.extend_from_slice(&down[&i].1); }
        emit(format!("{base}.down_proj"), Dtype::BF16, vec![e, hidden, inter], &dn,
             outs, n_q4, n_q8, n_copy, bytes_out);
        println!("  fused experts: {base} → gate_up_proj [{e},{},{hidden}] + down_proj [{e},{hidden},{inter}]", 2 * inter);
    };

    // STREAM the output to disk in ~12 GB shards as we go, so peak RAM is ~one shard + one input shard.
    // A 397B's ~220 GB of quantized output can't be buffered in 128 GB host RAM (the old accumulate-then-
    // shard path OOM'd past ~100 GB). Small models still collapse to one `model.safetensors` at the end.
    const SHARD_BYTES: usize = 12 * 1024 * 1024 * 1024;
    let meta = std::collections::HashMap::from([
        ("format".to_string(), "pt".to_string()),
        ("quant_recipe".to_string(), recipe_s.to_string()),
    ]);
    let mut weight_map = serde_json::Map::new();
    let mut shard_idx = 0usize;
    let outs_bytes = |outs: &Vec<Out>| -> usize { outs.iter().map(|o| o.data.len()).sum() };
    let write_shard = |outs: &mut Vec<Out>,
                       weight_map: &mut serde_json::Map<String, serde_json::Value>,
                       shard_idx: &mut usize| {
        if outs.is_empty() { return; }
        let fname = format!("model-{:05}.safetensors", *shard_idx + 1);
        let views: Vec<(String, TensorView)> = outs.iter()
            .map(|o| (o.name.clone(), TensorView::new(o.dtype, o.shape.clone(), &o.data).expect("view")))
            .collect();
        safetensors::serialize_to_file(views, Some(meta.clone()), &outd.join(&fname)).expect("write shard");
        for o in outs.iter() { weight_map.insert(o.name.clone(), serde_json::json!(fname.clone())); }
        *shard_idx += 1;
        println!("  wrote {} ({} tensors)", fname, outs.len());
        outs.clear();
    };

    for (i, sf) in shards.iter().enumerate() {
        println!("  shard {}/{}: {}", i + 1, shards.len(),
                 sf.file_name().unwrap_or_default().to_string_lossy());
        let raw = std::fs::read(sf).expect("read shard");
        let st = SafeTensors::deserialize(&raw).expect("parse shard");
        for (name, view) in st.tensors() {
            let data = view.data();
            bytes_in += data.len() as u64;
            // Un-fused expert weight? stash for fusion, don't emit yet. Fuse ON-COMPLETE (a full
            // contiguous 0..E set for all three projections) so the stash stays ~1 layer deep in RAM.
            if let Some((base, proj, idx)) = parse_expert(&name) {
                pending.entry(base.clone()).or_default().entry(proj).or_default()
                    .insert(idx, (view.shape().to_vec(), data.to_vec()));
                if let Some(e) = n_experts {
                    if complete(pending.get(&base).unwrap(), e) {
                        let projs = pending.remove(&base).unwrap();
                        fuse_base(&base, &projs, &mut outs, &mut n_q4, &mut n_q8, &mut n_copy, &mut bytes_out);
                        if outs_bytes(&outs) >= SHARD_BYTES { write_shard(&mut outs, &mut weight_map, &mut shard_idx); }
                    }
                }
                continue;
            }
            emit(name.clone(), view.dtype(), view.shape().to_vec(), data,
                 &mut outs, &mut n_q4, &mut n_q8, &mut n_copy, &mut bytes_out);
            if outs_bytes(&outs) >= SHARD_BYTES { write_shard(&mut outs, &mut weight_map, &mut shard_idx); }
        }
        // shard bytes dropped here â peak memory is one shard + the (much smaller) output
    }

    // Drain any bases still pending at the end (e.g. the 122B's single MTP block, whose fusion triggers
    // here when `n_experts` was absent or the set only completed on the last shard). They MUST be
    // complete — an incomplete set means a broken checkpoint (missing expert), and emitting anything
    // would produce a wrong-expert model. Loud, not silent.
    for (base, projs) in pending.iter() {
        let e = n_experts.unwrap_or_else(|| projs["gate_proj"].len());
        assert!(complete(projs, e),
                "{base}: incomplete expert set ({e} expected) — cannot fuse; the checkpoint is broken");
        fuse_base(base, projs, &mut outs, &mut n_q4, &mut n_q8, &mut n_copy, &mut bytes_out);
        if outs_bytes(&outs) >= SHARD_BYTES { write_shard(&mut outs, &mut weight_map, &mut shard_idx); }
    }

    // FINALIZE. If nothing was flushed mid-run (small model < SHARD_BYTES), collapse to one
    // `model.safetensors` (no index) — preserves the existing small-model layout. Otherwise flush the
    // tail shard and write the index over all streamed shards.
    if shard_idx == 0 {
        let views: Vec<(String, TensorView)> = outs.iter()
            .map(|o| (o.name.clone(), TensorView::new(o.dtype, o.shape.clone(), &o.data).expect("view"))).collect();
        safetensors::serialize_to_file(views, Some(meta.clone()), &outd.join("model.safetensors")).expect("write safetensors");
    } else {
        write_shard(&mut outs, &mut weight_map, &mut shard_idx);
        let index = serde_json::json!({ "metadata": { "total_size": bytes_out }, "weight_map": weight_map });
        std::fs::write(outd.join("model.safetensors.index.json"),
                       serde_json::to_string_pretty(&index).unwrap()).expect("write index");
        println!("  wrote {} shards + index (streamed to disk; peak RAM ~one shard)", shard_idx);
    }

    // Carry the sidecars across, and record the recipe in config.json so the loader can self-detect.
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
              "chat_template.jinja", "merges.txt", "vocab.json", "preprocessor_config.json"] {
        let src = ind.join(f);
        if src.exists() { let _ = std::fs::copy(&src, outd.join(f)); }
    }
    let cfg_path = outd.join("config.json");
    if let Ok(raw) = std::fs::read_to_string(&cfg_path) {
        if let Ok(mut cfg) = serde_json::from_str::<serde_json::Value>(&raw) {
            cfg["quantization_config"] = serde_json::json!({
                "quant_method": "compressed-tensors",
                "format": "nvfp4-pack-quantized",
                "recipe": recipe_s,
            });
            let _ = std::fs::write(&cfg_path, serde_json::to_string_pretty(&cfg).unwrap());
        }
    }

    let gi = bytes_in as f64 / 1e9;
    let go = bytes_out as f64 / 1e9;
    println!();
    println!("  tensors: {} NVFP4, {} FP8, {} copied through", n_q4, n_q8, n_copy);
    println!("  size:    {:.2} GB -> {:.2} GB   ({:.2}x smaller)", gi, go, gi / go.max(1e-9));
    println!("  wrote {} in {:.0}s", outd.display(), t0.elapsed().as_secs_f32());
}

/// `--perplexity --model-dir <d> --text <file>` â perplexity on held-out text.
///
/// The quality gate for quantization. Combine with `RUST_INFER_FAKE_QUANT=<groups>` to measure what
/// 4 bits actually costs, per tensor group, in the real engine â without needing any 4-bit kernel.
// Derive a gdn4 (GDN-nvfp4) artifact from a mixed (GDN-fp8) one — no bf16 needed. Only the fp8 GDN
// in/out-proj tensors are re-quantized (dequant fp8 → bf16 → nvfp4); everything else (already nvfp4 or
// bf16) is copied byte-for-byte. Output is streamed in ~12 GB shards (bounded RAM, works for the 397B).
fn run_requant_gdn(args: &[String]) {
    use safetensors::{SafeTensors, Dtype, tensor::TensorView};
    use gb10_inference::quant;
    let from = parse_arg(args, "--from").expect("--requant-gdn requires --from <mixed-dir>");
    let out  = parse_arg(args, "--out").expect("--requant-gdn requires --out <gdn4-dir>");
    let ind = std::path::Path::new(from);
    let outd = std::path::Path::new(out);
    std::fs::create_dir_all(outd).expect("create --out dir");
    println!("Deriving gdn4 (GDN nvfp4) from {} -> {}", from, out);

    let index_path = ind.join("model.safetensors.index.json");
    let shards: Vec<std::path::PathBuf> = if index_path.exists() {
        let raw = std::fs::read_to_string(&index_path).expect("read index");
        let idx: serde_json::Value = serde_json::from_str(&raw).expect("parse index");
        idx["weight_map"].as_object().unwrap().values().filter_map(|v| v.as_str())
            .collect::<std::collections::BTreeSet<_>>().into_iter().map(|s| ind.join(s)).collect()
    } else { vec![ind.join("model.safetensors")] };

    struct Out { name: String, dtype: Dtype, shape: Vec<usize>, data: Vec<u8> }
    const SHARD_BYTES: usize = 12 * 1024 * 1024 * 1024;
    let meta = std::collections::HashMap::from([
        ("format".to_string(), "pt".to_string()),
        ("quant_recipe".to_string(), "all,-router (gdn4 derived from mixed via requant-gdn)".to_string()),
    ]);
    let mut weight_map = serde_json::Map::new();
    let mut shard_idx = 0usize;
    let mut bytes_out = 0u64;
    let outs_bytes = |o: &Vec<Out>| -> usize { o.iter().map(|x| x.data.len()).sum() };
    let write_shard = |outs: &mut Vec<Out>, wm: &mut serde_json::Map<String, serde_json::Value>, si: &mut usize| {
        if outs.is_empty() { return; }
        let fname = format!("model-{:05}.safetensors", *si + 1);
        let views: Vec<(String, TensorView)> = outs.iter()
            .map(|o| (o.name.clone(), TensorView::new(o.dtype, o.shape.clone(), &o.data).expect("view"))).collect();
        safetensors::serialize_to_file(views, Some(meta.clone()), &outd.join(&fname)).expect("write shard");
        for o in outs.iter() { wm.insert(o.name.clone(), serde_json::json!(fname.clone())); }
        *si += 1; println!("  wrote {} ({} tensors)", fname, outs.len()); outs.clear();
    };
    let is_gdn = |n: &str| n.contains(".linear_attn.") && (n.contains("in_proj") || n.contains("out_proj"));

    let mut outs: Vec<Out> = Vec::new();
    let (mut n_req, mut n_copy) = (0usize, 0usize);
    for (si, sf) in shards.iter().enumerate() {
        println!("  shard {}/{}: {}", si + 1, shards.len(), sf.file_name().unwrap_or_default().to_string_lossy());
        let raw = std::fs::read(sf).expect("read shard");
        let st = SafeTensors::deserialize(&raw).expect("parse shard");
        let tvec: Vec<(String, TensorView)> = st.tensors();
        // Group the shard's tensors by STEM so a tensor's whole family (nvfp4 triple
        // weight_packed/weight_scale/weight_global_scale, or the fp8 weight+weight_scale) is emitted
        // together and never split across output shards — the loader pairs the triple WITHIN one shard.
        let mut by_stem: std::collections::BTreeMap<&str, Vec<&(String, TensorView)>> =
            std::collections::BTreeMap::new();
        for t in &tvec {
            let n = t.0.as_str();
            let stem = &n[..n.rfind(".weight").unwrap_or(n.len())];
            by_stem.entry(stem).or_default().push(t);
        }
        for (stem, parts) in by_stem {
            // A GDN in/out-proj stored as fp8 = a `.weight` (F8) + `.weight_scale` (F32). Re-quantize it.
            let fp8_w = parts.iter().find(|(n, v)| n.ends_with(".weight") && v.dtype() == Dtype::F8_E4M3);
            if is_gdn(stem) && fp8_w.is_some() {
                let (_, wv) = fp8_w.unwrap();
                let (_, sv) = parts.iter().find(|(n, _)| n.ends_with(".weight_scale"))
                    .expect("gdn fp8 weight_scale missing");
                let (m, k) = (wv.shape()[0], wv.shape()[1]);
                let fp8 = quant::Fp8Tensor {
                    qweight: wv.data().to_vec(),
                    row_scale: bytemuck::cast_slice::<u8, f32>(sv.data()).to_vec(),
                    m, k,
                };
                let bf = quant::dequantize_fp8(&fp8);
                let q = quant::quantize_nvfp4(&bf, m, k);
                bytes_out += (q.qweight.len() + q.scales.len() + 4) as u64;
                outs.push(Out { name: format!("{stem}.weight_packed"), dtype: Dtype::U8, shape: vec![m, k/2], data: q.qweight });
                outs.push(Out { name: format!("{stem}.weight_scale"), dtype: Dtype::F8_E4M3, shape: vec![m, k/quant::BLOCK], data: q.scales });
                outs.push(Out { name: format!("{stem}.weight_global_scale"), dtype: Dtype::F32, shape: vec![1], data: q.global_scale.to_le_bytes().to_vec() });
                n_req += 1;
            } else {
                for (n, v) in parts {
                    bytes_out += v.data().len() as u64;
                    outs.push(Out { name: n.clone(), dtype: v.dtype(), shape: v.shape().to_vec(), data: v.data().to_vec() });
                    n_copy += 1;
                }
            }
            if outs_bytes(&outs) >= SHARD_BYTES { write_shard(&mut outs, &mut weight_map, &mut shard_idx); }
        }
    }
    if shard_idx == 0 {
        let views: Vec<(String, TensorView)> = outs.iter()
            .map(|o| (o.name.clone(), TensorView::new(o.dtype, o.shape.clone(), &o.data).expect("view"))).collect();
        safetensors::serialize_to_file(views, Some(meta.clone()), &outd.join("model.safetensors")).expect("write");
    } else {
        write_shard(&mut outs, &mut weight_map, &mut shard_idx);
        let index = serde_json::json!({ "metadata": { "total_size": bytes_out }, "weight_map": weight_map });
        std::fs::write(outd.join("model.safetensors.index.json"), serde_json::to_string_pretty(&index).unwrap()).expect("write index");
    }
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
              "chat_template.jinja", "merges.txt", "vocab.json", "preprocessor_config.json"] {
        let src = ind.join(f);
        if src.exists() { let _ = std::fs::copy(&src, outd.join(f)); }
    }
    println!("  requant-gdn done: {} GDN tensors re-quantized fp8->nvfp4, {} copied verbatim, {} shards, {:.1} GB",
             n_req, n_copy, shard_idx.max(1), bytes_out as f64 / 1e9);
}

fn mtp_calib_cache_path(model_path: &str) -> Option<std::path::PathBuf> {
    // <binary_dir>/mtp_calib/<model-basename>.json  — a subdir next to the running executable.
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?.join("mtp_calib");
    let base = std::path::Path::new(model_path.trim_end_matches('/'))
        .file_name()?.to_string_lossy().replace(['/', ' '], "_");
    Some(dir.join(format!("{base}.json")))
}
fn mtp_calib_stamp(model_path: &str) -> String {
    // Invalidate on a rebuild (binary mtime changes = new kernels) or a different model.
    let bin_mtime = std::env::current_exe().ok()
        .and_then(|e| std::fs::metadata(e).ok())
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs()).unwrap_or(0);
    format!("bin_mtime={bin_mtime};model={}", model_path.trim_end_matches('/'))
}
fn read_mtp_calib(path: &Option<std::path::PathBuf>, stamp: &str) -> Option<Vec<(usize, f32)>> {
    let txt = std::fs::read_to_string(path.as_ref()?).ok()?;
    let v: serde_json::Value = serde_json::from_str(&txt).ok()?;
    if v.get("stamp")?.as_str()? != stamp { return None; }
    let out: Vec<(usize, f32)> = v.get("r")?.as_array()?.iter()
        .filter_map(|e| Some((e.get(0)?.as_u64()? as usize, e.get(1)?.as_f64()? as f32)))
        .collect();
    if out.is_empty() { None } else { Some(out) }
}
fn write_mtp_calib(path: &Option<std::path::PathBuf>, stamp: &str, rs: &[(usize, f32)]) {
    let Some(p) = path else { return };
    if let Some(dir) = p.parent() { let _ = std::fs::create_dir_all(dir); }
    let r: Vec<serde_json::Value> = rs.iter().map(|&(d, r)| serde_json::json!([d, r])).collect();
    if let Ok(s) = serde_json::to_string_pretty(&serde_json::json!({ "stamp": stamp, "r": r })) {
        if std::fs::write(p, s).is_ok() { println!("MTP cost/depth: cached to {}", p.display()); }
    }
}

fn run_perplexity(args: &[String]) {
    let (model_path, tokenizer_path) = if let Some(dir) = parse_arg(args, "--model-dir") {
        (dir.to_string(), format!("{}/tokenizer.json", dir.trim_end_matches('/')))
    } else {
        (parse_arg(args, "--model").unwrap_or("model/model.safetensors").to_string(),
         parse_arg(args, "--tokenizer").unwrap_or("model/tokenizer.json").to_string())
    };
    let text_path = parse_arg(args, "--text").expect("--text <file> required");
    let window: usize = parse_arg(args, "--window").and_then(|s| s.parse().ok()).unwrap_or(1024);
    let max_windows: usize = parse_arg(args, "--max-windows").and_then(|s| s.parse().ok()).unwrap_or(8);
    let max_seq_len: usize = parse_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);

    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let text = std::fs::read_to_string(&text_path).expect("read --text file");
    // No chat template, no BOS games: score raw text.
    let toks = tokenizer.encode(&text, false).expect("encode");
    println!("Perplexity: {} tokens from {}, window={}, max_windows={}",
             toks.len(), text_path, window, max_windows);

    let gpu = if std::path::Path::new(&model_path).is_dir() {
        let (gpu, _) = gb10_inference::gpu::GpuModel::load_from_dir(&model_path).expect("gpu load");
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

/// `--profile-mtp` â attribute the cost of one stochastic-MTP step.
/// DEBUG PROBE: `--probe-moe --model-dir <D> --batch B --out-x <f> --out <f>` — run the first MoE
/// layer's `moe_batch` on a deterministic input and write input+output (f32, space-separated) so a
/// numpy reference over that layer's checkpoint weights can validate the block numerically.
fn run_probe_moe(args: &[String]) {
    use half::bf16;
    let dir = parse_arg(args, "--model-dir").expect("--probe-moe requires --model-dir").to_string();
    let batch: usize = parse_arg(args, "--batch").and_then(|s| s.parse().ok()).unwrap_or(4);
    let out_x = parse_arg(args, "--out-x").unwrap_or("/tmp/moe_x.txt").to_string();
    let out_y = parse_arg(args, "--out").unwrap_or("/tmp/moe_y.txt").to_string();

    let (gpu, cfg) = gb10_inference::gpu::GpuModel::load_from_dir(&dir).expect("gpu load");
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

/// DEBUG CAPTURE: `--capture-layers --model-dir <D> --ids <f> --out <f>` — teacher-force one prompt
/// of RAW token ids (whitespace-separated, NOT re-tokenized text) through the prefill path and dump
/// the hidden state at every layer boundary as a safetensors file of bf16 [seq, hidden] tensors:
/// `layer.00.in` (embed out), `layer.NN.out` (residual after layer NN), `final_norm`. This is the
/// engine side of the Hy3 oracle comparison (scripts/compare_hy3_oracle.py consumes these dumps).
fn run_capture_layers(args: &[String]) {
    use safetensors::{Dtype, tensor::TensorView};
    let dir = parse_arg(args, "--model-dir").expect("--capture-layers requires --model-dir").to_string();
    let ids_path = parse_arg(args, "--ids").expect("--capture-layers requires --ids <file>").to_string();
    let out_path = parse_arg(args, "--out").expect("--capture-layers requires --out <file>").to_string();

    let ids_txt = std::fs::read_to_string(&ids_path).expect("read --ids file");
    let ids: Vec<u32> = ids_txt.split_whitespace()
        .map(|t| t.parse().expect("token id")).collect();
    assert!(!ids.is_empty(), "empty --ids file");
    let n = ids.len();

    let (gpu, cfg) = gb10_inference::gpu::GpuModel::load_from_dir(&dir).expect("gpu load");
    let h = cfg.hidden_size;
    let nlayers = gpu.cfg().num_layers;
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    let kv_stride = n.max(16);
    let mut state = gpu.new_batch_state(1, 1, kv_stride);

    let dumps = gpu.capture_prefill(&mut pool, &ids, &mut state, kv_stride);
    let debug = std::env::var("GB10_CAP_DEBUG").is_ok();
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
    let views: Vec<(String, TensorView)> = named.iter()
        .map(|(name, dmp)| {
            let bytes: &[u8] = bytemuck::cast_slice(&dmp[..]);
            (name.clone(), TensorView::new(Dtype::BF16, vec![n, h], bytes).expect("view"))
        }).collect();
    safetensors::serialize_to_file(views, None, std::path::Path::new(&out_path)).expect("write safetensors");
    eprintln!("capture-layers: {} tokens x {} layers -> {} ({} tensors)", n, nlayers, out_path, named.len());
    println!("CAPTURE_OK {}", out_path);
}

fn run_dump_argmax(args: &[String]) {
    let model_path = parse_arg(args, "--model-dir").expect("--dump-argmax requires --model-dir <DIR>").to_string();
    let tokenizer_path = format!("{}/tokenizer.json", model_path.trim_end_matches('/'));
    let text_path = parse_arg(args, "--text").expect("--text <file> required");
    let out_path = parse_arg(args, "--out").expect("--out <file> required");
    let window: usize = parse_arg(args, "--window").and_then(|s| s.parse().ok()).unwrap_or(512);
    let max_seq_len: usize = parse_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);

    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let text = std::fs::read_to_string(&text_path).expect("read --text file");
    let toks = tokenizer.encode(&text, false).expect("encode");
    eprintln!("dump-argmax: {} tokens from {}, window={}", toks.len(), text_path, window);

    let (gpu, _) = gb10_inference::gpu::GpuModel::load_from_dir(&model_path).expect("gpu load");
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

/// `--profile-mtp`: attribute the cost of one stochastic-MTP step.
fn run_profile_mtp(args: &[String]) {
    let (model_path, tokenizer_path) = if let Some(dir) = parse_arg(args, "--model-dir") {
        (dir.to_string(), format!("{}/tokenizer.json", dir.trim_end_matches('/')))
    } else {
        (parse_arg(args, "--model").unwrap_or("model/model.safetensors").to_string(),
         parse_arg(args, "--tokenizer").unwrap_or("model/tokenizer.json").to_string())
    };
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("The capital of France is");
    let iters: usize = parse_arg(args, "--iters").and_then(|s| s.parse().ok()).unwrap_or(20);
    let max_seq_len: usize = parse_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);

    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");
    let gpu = if std::path::Path::new(&model_path).is_dir() {
        let (gpu, _) = gb10_inference::gpu::GpuModel::load_from_dir(&model_path).expect("gpu load");
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

/// `--bench-mtp-sample` â the distribution-exactness gate for stochastic MTP.
///
/// Greedy MTP is bitwise-lossless and `--bench-mtp` proves it by direct comparison. Stochastic MTP is
/// only *distribution*-exact, so it is gated statistically instead: hold the prefix fixed and draw
/// many emissions through the real kernels, then compare the emitted-token histogram against the
/// distribution the plain sampler is defined to produce. The plain sampler's own histogram is drawn
/// alongside as a control â it fixes the sampling-noise floor at this trial count, which is the bar
/// the MTP path has to reach (being merely "small" is not enough).
fn run_bench_mtp_sample(args: &[String]) {
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
    let max_seq_len: usize = parse_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);
    let temps: Vec<f32> = match parse_arg(args, "--temp") {
        Some(s) => vec![s.parse().expect("--temp")],
        None => vec![0.3, 0.7, 1.0],
    };

    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");

    let gpu = if std::path::Path::new(&model_path).is_dir() {
        let (gpu, _) = gb10_inference::gpu::GpuModel::load_from_dir(&model_path).expect("gpu load");
        gpu
    } else {
        let host = gb10_inference::qwen::Model::load(&model_path).expect("load model");
        gb10_inference::gpu::GpuModel::new(&host).expect("gpu init")
    };
    if !gpu.mtp_present() {
        println!("No MTP head â cannot run the stochastic-MTP gate.");
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
    // "fail" the analytic reference in lockstep. The claim we need is reference-free â MTP emits from
    // the same law as the plain sampler â so we test the two empirical histograms against each other.
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

        println!("temp={:.2}  draft={} nucleus={} p(draft)={:.4} (device {:.4}, Î={:.1e})",
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

fn run_bench_mtp(args: &[String]) {
    let (model_path, tokenizer_path) = if let Some(dir) = parse_arg(args, "--model-dir") {
        (dir.to_string(), format!("{}/tokenizer.json", dir.trim_end_matches('/')))
    } else {
        (parse_arg(args, "--model").unwrap_or("model/model.safetensors").to_string(),
         parse_arg(args, "--tokenizer").unwrap_or("model/tokenizer.json").to_string())
    };
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("The capital of France is");
    let depth: usize = parse_arg(args, "--depth").and_then(|s| s.parse().ok()).unwrap_or(4);
    let max_new: usize = parse_arg(args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(64);
    let max_seq_len: usize = parse_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);

    let tokenizer = QwenTokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");
    println!("MTP end-to-end probe: prompt={} tokens, depth={}, max_new={}", prompt.len(), depth, max_new);

    let gpu = if std::path::Path::new(&model_path).is_dir() {
        let (gpu, _) = gb10_inference::gpu::GpuModel::load_from_dir(&model_path).expect("gpu load");
        gpu
    } else {
        let host = gb10_inference::qwen::Model::load(&model_path).expect("load model");
        gb10_inference::gpu::GpuModel::new(&host).expect("gpu init")
    };
    if gpu.mtp_present() {
        println!("MTP head loaded.");
    } else {
        println!("No MTP head â cannot run MTP probe.");
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
        println!("RESULT: MISMATCH at token {:?} â MTP is NOT lossless", div);
        if let Some(d) = div {
            let lo = d.saturating_sub(6);
            let hi = (d + 6).min(mtp_tokens.len()).min(seq_tokens.len());
            println!("  ctx MTP[{}..{}]: {:?}", lo, hi, &mtp_tokens[lo..hi]);
            println!("  ctx SEQ[{}..{}]: {:?}", lo, hi, &seq_tokens[lo..hi]);
        }
        std::process::exit(1);
    }
}

fn run_server(args: &[String]) {
    // Support both --model-dir <DIR> and legacy --model <FILE> + --tokenizer <FILE>
    let (model_path, tokenizer_path) = if let Some(dir) = parse_arg(args, "--model-dir") {
        (dir.to_string(), format!("{}/tokenizer.json", dir.trim_end_matches('/')))
    } else {
        let model = parse_arg(args, "--model").unwrap_or("model/model.safetensors");
        let tokenizer = parse_arg(args, "--tokenizer").unwrap_or("model/tokenizer.json");
        (model.to_string(), tokenizer.to_string())
    };

    let port = parse_arg(args, "--port").and_then(|s| s.parse::<u16>().ok()).unwrap_or(8000);
    let max_seq_len = parse_arg(args, "--max-seq-len").and_then(|s| s.parse::<usize>().ok()).unwrap_or(4096);
    let max_batch = parse_arg(args, "--max-batch").and_then(|s| s.parse::<usize>().ok()).unwrap_or(8);
    // TP=2 serving (TP item A): sync the model + config to ONE --node, bring up the RDMA link, and
    // run this same server with its BatchScheduler in SPMD lockstep with the node's mirror.
    let tp = args.iter().any(|a| a == "--tp");
    // Auto-detect the model name from the directory name ("2b" â "qwen3.5-2b").
    //
    // If the directory ALREADY carries a family version ("3.6-27b-nvfp4-full"), do not staple another
    // one in front of it â that produced "qwen3.5-3.6-27b-nvfp4-full". The directory name is the only
    // signal we have: config.json reports model_type "qwen3_5" for Qwen3.6 too, so it cannot tell the
    // families apart.
    //
    // The version must be digits/dots followed by '-', which is what separates "3.6-27b" (a version)
    // from "0.8b-bf16" (a size that merely starts with a digit, and must stay qwen3.5-0.8b-bf16).
    fn has_leading_version(s: &str) -> bool {
        let mut parts = s.splitn(2, '-');
        let head = parts.next().unwrap_or("");
        parts.next().is_some()
            && head.contains('.')
            && head.chars().all(|c| c.is_ascii_digit() || c == '.')
    }
    let model_name = parse_arg(args, "--model-name").map(|s| s.to_string()).unwrap_or_else(|| {
        let dir_name = std::path::Path::new(model_path.trim_end_matches('/'))
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        // The family comes from the model's OWN config.json, not the directory name: only
        // qwen-family models get the "qwen" prefix. hy_v3's model_type is "hy_v3", so Hy3 must
        // not become "qwen3.5-hy3-nvfp4" (its dir carries no version number and fell into the
        // prepend branch). Unreadable/missing config keeps the old behavior (qwen).
        let model_type = std::fs::read_to_string(format!("{}/config.json", model_path.trim_end_matches('/')))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| v.get("model_type").and_then(|m| m.as_str()).map(|s| s.to_string()));
        let is_qwen = model_type.as_deref().map(|m| m.starts_with("qwen")).unwrap_or(true);
        if !is_qwen { dir_name }
        else if has_leading_version(&dir_name) { format!("qwen{}", dir_name) }
        else { format!("qwen3.5-{}", dir_name) }
    });
    let default_max_tokens = parse_arg(args, "--max-tokens").and_then(|s| s.parse::<usize>().ok()).unwrap_or(8192);
    // Model-card presence-penalty default varies by model size (2B: 2.0, 4B+: 1.5). Temperature
    // and top_p defaults are applied per-request via serde defaults in server.rs.
    let is_2b = model_path.contains("2b");
    let default_presence_penalty = if is_2b { 2.0 } else { 1.5 };

    let default_rep_penalty = parse_arg(args, "--default-repetition-penalty").and_then(|s| s.parse::<f32>().ok()).unwrap_or(1.0);
    let default_presence_penalty = parse_arg(args, "--default-presence-penalty").and_then(|s| s.parse::<f32>().ok()).unwrap_or(default_presence_penalty);
    let default_frequency_penalty = parse_arg(args, "--default-frequency-penalty").and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0);

    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");

    rt.block_on(async {
        // Use streaming loader for all models (reads bf16 directly, no f32 intermediate)
        let is_dir = std::path::Path::new(&model_path).is_dir();

        // TP=2 PARALLEL LOAD (head serve path). Everything the head session needs is derivable
        // from config.json + the tokenizer (KBs, no weights): the --max-seq-len clamp needs
        // max_position_embeddings, tpc.eos/calib_prompt need the tokenizer. So: pre-read, build
        // + install the TpConfig, run the session (manifest + blob transfer + Config ship —
        // pure TCP/UDP/fs) on a thread, and — critically — bring up the RDMA link BEFORE the
        // heavy load. The node handshakes its QP immediately after the sync, ahead of ITS load
        // (net_shim retries the connect for ~20 s), so a post-load bring-up either serializes
        // both loads (the old order: node idles through the whole head load) or kills the node
        // outright (head load > 20 s retry window → connection refused, verified). With the
        // handshake up front BOTH ranks load CONCURRENTLY; the first all-reduce (calibration)
        // remains the rendezvous the watchdog already tolerates multi-minute load skew on.
        // Warm-cache bring-up becomes max(head_load, node_load) instead of head_load + node_load.
        let mut pre_tokenizer: Option<QwenTokenizer> = None;
        // (max_position_embeddings, eos_token_id) as the pre-read saw them — the post-load
        // guardrail asserts the loaded cfg agrees (the reorder must not change any value).
        let mut pre_cfg_check: Option<(usize, u32)> = None;
        // Parallel-load path only: the completed QP handshake + the retained control stream,
        // both established BEFORE the weight load.
        let mut pre_tp: Option<(gb10_inference::tp::TpContext, std::net::TcpStream)> = None;
        if tp && is_dir {
            let pre = gb10_inference::qwen::Config::from_config_json(
                &format!("{}/config.json", model_path.trim_end_matches('/')))
                .expect("pre-read config.json");
            let max_seq_pre = if max_seq_len > pre.max_position_embeddings {
                pre.max_position_embeddings
            } else { max_seq_len };
            println!("Loading tokenizer from {}...", tokenizer_path);
            let tok_pre = QwenTokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer");
            // Same parses as their post-load counterparts below (idempotent; needed here for tpc).
            let mtp_depth_pre = parse_arg(args, "--mtp-depth").and_then(|s| s.parse::<usize>().ok());
            let mtp_force_pre = match parse_arg(args, "--mtp").unwrap_or("auto") {
                "on"  | "1" | "true"  => Some(true),
                "off" | "0" | "false" => Some(false),
                "auto" => None,
                other => { eprintln!("--mtp must be auto|on|off (got {:?})", other); std::process::exit(1); }
            };
            let explicit = parse_arg(args, "--nodes").map(|s| {
                s.split(',').map(|p| {
                    let p = p.trim();
                    if p.contains(':') { p.parse::<std::net::SocketAddr>().expect("bad --nodes addr (ip:port)") }
                    else { std::net::SocketAddr::new(p.parse::<std::net::IpAddr>().expect("bad --nodes ip"), 29500) }
                }).collect::<Vec<_>>()
            });
            let wait = std::time::Duration::from_secs(
                parse_arg(args, "--discover-wait").and_then(|s| s.parse().ok()).unwrap_or(3));
            let mut tpc = gb10_inference::tp::TpConfig::from_env();
            tpc.mode_serve = true;
            tpc.max_seq_len = max_seq_pre;
            tpc.max_batch = max_batch;
            tpc.prefix_cache = matches!(parse_arg(args, "--prefix-cache").unwrap_or("off"),
                                        "on" | "true" | "1" | "yes");
            tpc.ngram_draft = parse_arg(args, "--ngram-draft").and_then(|s| s.parse().ok()).unwrap_or(0);
            tpc.tree_draft = matches!(parse_arg(args, "--tree-draft").unwrap_or("off"), "on"|"true"|"1"|"yes");
            tpc.mtp_lanes = matches!(parse_arg(args, "--mtp-lanes").unwrap_or("off"), "on"|"true"|"1"|"yes");
            tpc.mtp_force = mtp_force_pre;
            tpc.mtp_depth_pin = mtp_depth_pre;
            tpc.no_decode_graphs = std::env::var("GB10_NO_DECODE_GRAPHS").is_ok();
            tpc.cpu_sample = std::env::var("RUST_INFER_CPU_SAMPLE").is_ok();
            tpc.eos = tok_pre.stop_token_ids(pre.eos_token_id);
            tpc.calib_prompt = tok_pre.encode("The capital of France is", true)
                .expect("probe encode");
            tpc.batch_probe = Some(max_batch);
            // Installing before the load is behavior-neutral: the only loader consumer
            // (gpu.rs shard-at-load shard_mixers) ORs tp_config with the very env var
            // TpConfig::from_env mirrors, so it reads the same value either way.
            gb10_inference::tp::set_tp_config(tpc.clone());
            // The session overlaps our pre-load CPU work (config/tokenizer above); the transfer
            // itself is 0 bytes warm. On a cold first sync it serializes ahead of the load —
            // the old order was just as serialized cold (documented out of scope in the plan).
            let path_for_session = model_path.clone();
            let session = std::thread::spawn(move || {
                gb10_inference::cluster::run_head_session(
                    std::path::Path::new(&path_for_session), explicit, wait, &tpc)
            });
            // Join + handshake NOW (see the big comment above). A session error (node
            // unreachable) surfaces here with the same fatality as before — just EARLIER.
            let (_nodes, stream) = session.join()
                .expect("TP head session thread panicked")
                .expect("TP head session sync");
            let mut ctx = gb10_inference::tp::TpContext::bring_up_head().expect("TP bring-up");
            ctx.sanity().expect("TP sanity");
            pre_tp = Some((ctx, stream));
            pre_cfg_check = Some((pre.max_position_embeddings, pre.eos_token_id));
            pre_tokenizer = Some(tok_pre);
        }

        let (mut gpu, cfg) = if is_dir {
            println!("Loading model from {} (streaming bf16)...", model_path);
            if tp {
                // TP=2 head = rank 0. hy_v3 shards host-side in the loader (full model > one node);
                // qwen loads whole and shards at attach_tp, unchanged.
                gb10_inference::gpu::GpuModel::load_from_dir_tp(&model_path, 0).expect("gpu load")
            } else {
                gb10_inference::gpu::GpuModel::load_from_dir(&model_path).expect("gpu load")
            }
        } else {
            println!("Loading model from {} ...", model_path);
            let host = gb10_inference::qwen::Model::load(&model_path).expect("load model");
            let gpu = gb10_inference::gpu::GpuModel::new(&host).expect("gpu init");
            (gpu, host.config)
        };
        // Guardrail on the TP parallel-load reorder: the pre-read values drove the TpConfig shipped
        // to the node (and our own clamp); the loaded cfg MUST agree, or head and node would run
        // with different context limits / stop tokens.
        if let Some((mpe, eos_id)) = pre_cfg_check {
            assert_eq!(cfg.max_position_embeddings, mpe,
                       "TP parallel load: pre-read max_position_embeddings {mpe} != loaded cfg's {}",
                       cfg.max_position_embeddings);
            assert_eq!(cfg.eos_token_id, eos_id,
                       "TP parallel load: pre-read eos_token_id {eos_id} != loaded cfg's {}",
                       cfg.eos_token_id);
        }
        let config_eos = cfg.eos_token_id;

        // Clamp --max-seq-len to what the model actually supports. The RoPE cos/sin tables are sized to
        // `max_position_embeddings` (262144 for this family), so a KV cache bigger than that would ask
        // for rotations past the end of the tables. Going UP to the model max is fully supported (KV is
        // ~64 KB/token â 256K/batch-2 â 34 GB, fine on 128 GB, just slow to prefill); going beyond is not.
        let model_max = cfg.max_position_embeddings;
        let max_seq_len = if max_seq_len > model_max {
            eprintln!("[warn] --max-seq-len {} exceeds the model max_position_embeddings {} â clamping to {}.",
                      max_seq_len, model_max, model_max);
            model_max
        } else { max_seq_len };
        println!("Context: --max-seq-len {} (model max {}). KV cache ~{:.1} GB at batch {}.",
                 max_seq_len, model_max,
                 {
                     let nfull = cfg.layer_types.iter().filter(|t| matches!(t, gb10_inference::qwen::LayerType::FullAttention)).count();
                     (nfull * cfg.num_kv_heads * cfg.head_dim * 2 * 2 * max_seq_len * max_batch) as f64 / 1e9
                 }, max_batch);

        // MTP speculative decoding. There is no on/off flag and no env var: the engine measures
        // whether MTP pays and decides for itself.
        //
        //   a step emits (1 + accepted) tokens and costs r decode-steps  =>  MTP pays iff tok/step > r
        //
        // `r` is a pure cost ratio (it depends on the model's shape, chiefly the LM-head fraction,
        // because drafting must read the LM head a second time), so it is calibrated once here.
        // Acceptance is workload-dependent, so the scheduler tracks it live and revisits the decision.
        // Greedy vs stochastic verify is decided per REQUEST by temperature, never by configuration.
        //
        // `--mtp=on|off` forces the decision (for benchmarking); the default is `auto`.
        // `--mtp-depth` PINS the depth (benchmarking); by default the policy picks it from the
        // measured r(d) and the live acceptance, and re-picks as the workload changes.
        let mtp_depth = parse_arg(args, "--mtp-depth").and_then(|s| s.parse::<usize>().ok());
        let mtp_force = match parse_arg(args, "--mtp").unwrap_or("auto") {
            "on"  | "1" | "true"  => Some(true),
            "off" | "0" | "false" => Some(false),
            "auto" => None,
            other => { eprintln!("--mtp must be auto|on|off (got {:?})", other); std::process::exit(1); }
        };
        for stale in ["RUST_INFER_MTP", "RUST_INFER_MTP_STOCHASTIC", "RUST_INFER_GPU_SAMPLE"] {
            if std::env::var(stale).is_ok() {
                println!("note: {} is obsolete and ignored (MTP is auto-tuned; GPU sampling is the \
                          default). Use --mtp=on|off to force MTP.", stale);
            }
        }

        let tokenizer = match pre_tokenizer {
            // The TP parallel-load path already loaded it (and printed) before the weight load.
            Some(t) => t,
            None => {
                println!("Loading tokenizer from {}...", tokenizer_path);
                QwenTokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer")
            }
        };

        // STOP ON EVERY TURN TERMINATOR, not just the one config.json advertises. Qwen3.5 declares
        // eos_token_id = <|endoftext|> (248044), but a CHAT turn ends with <|im_end|> (248046) â which
        // is what the model actually emits. Stopping only on the advertised id let the assistant run
        // past the end of its own turn and hallucinate the next one: a fabricated `user` message, a new
        // `<think>` block, sometimes a second conflicting tool call. See QwenTokenizer::stop_token_ids.
        let eos = tokenizer.stop_token_ids(config_eos);
        println!("Stop tokens: {:?}  (config.json advertises {})", eos, config_eos);

        // Serving-option values, parsed once here so the TP config (below) and BatchScheduler::new
        // (further down) agree. The explanatory prints stay at their original spots.
        let prefix_cache = matches!(parse_arg(args, "--prefix-cache").unwrap_or("off"),
                                    "on" | "true" | "1" | "yes");
        let ngram_draft: usize = parse_arg(args, "--ngram-draft").and_then(|s| s.parse().ok()).unwrap_or(0);
        let tree_draft = matches!(parse_arg(args, "--tree-draft").unwrap_or("off"), "on"|"true"|"1"|"yes");
        let mtp_lanes = matches!(parse_arg(args, "--mtp-lanes").unwrap_or("off"), "on"|"true"|"1"|"yes");

        // TP=2 serving bring-up (TP item A). Order matters and mirrors the node's `node_serve_tp`:
        // ship the model + config, bring up the RDMA link, attach TP — from here on EVERY forward
        // (calibration, graph capture, decode) runs SPMD in lockstep with the node. The retained
        // sync stream becomes the serving control plane (CalibTable / Ready / Step / Shutdown).
        let mut tp_stream: Option<std::net::TcpStream> = None;
        if tp {
            if prefix_cache {
                eprintln!("[tp] WARNING: --prefix-cache on under TP — a known TP issue is open on \
                           prefix reuse; v1 serving configs keep it off");
            }
            // The session + RDMA handshake either already happened BEFORE the weight load (the
            // --model-dir parallel-load path above — the node has been loading concurrently with
            // us since then) or — the legacy --model <file> path — happen here, serialized as before.
            let (stream, ctx) = match pre_tp {
                Some((ctx, stream)) => (stream, ctx),
                None => {
                    let explicit = parse_arg(args, "--nodes").map(|s| {
                        s.split(',').map(|p| {
                            let p = p.trim();
                            if p.contains(':') { p.parse::<std::net::SocketAddr>().expect("bad --nodes addr (ip:port)") }
                            else { std::net::SocketAddr::new(p.parse::<std::net::IpAddr>().expect("bad --nodes ip"), 29500) }
                        }).collect::<Vec<_>>()
                    });
                    let wait = std::time::Duration::from_secs(
                        parse_arg(args, "--discover-wait").and_then(|s| s.parse().ok()).unwrap_or(3));
                    // TpConfig v2: env snapshot for the bench knobs, serving fields from the server args.
                    // batch_probe = max_batch so attach_tp sizes the all-reduce payload for batched decode.
                    let mut tpc = gb10_inference::tp::TpConfig::from_env();
                    tpc.mode_serve = true;
                    tpc.max_seq_len = max_seq_len;
                    tpc.max_batch = max_batch;
                    tpc.prefix_cache = prefix_cache;
                    tpc.ngram_draft = ngram_draft;
                    tpc.tree_draft = tree_draft;
                    tpc.mtp_lanes = mtp_lanes;
                    tpc.mtp_force = mtp_force;
                    tpc.mtp_depth_pin = mtp_depth;
                    tpc.no_decode_graphs = std::env::var("GB10_NO_DECODE_GRAPHS").is_ok();
                    tpc.cpu_sample = std::env::var("RUST_INFER_CPU_SAMPLE").is_ok();
                    tpc.eos = eos.clone();
                    tpc.calib_prompt = tokenizer.encode("The capital of France is", true)
                        .expect("probe encode");
                    tpc.batch_probe = Some(max_batch);
                    gb10_inference::tp::set_tp_config(tpc.clone());
                    let (_nodes, stream) = gb10_inference::cluster::run_head_session(
                        std::path::Path::new(&model_path), explicit, wait, &tpc)
                        .expect("TP head session sync");
                    let mut ctx = gb10_inference::tp::TpContext::bring_up_head().expect("TP bring-up");
                    ctx.sanity().expect("TP sanity");
                    (stream, ctx)
                }
            };
            println!("HEAD (rank 0/2) — TP LINK UP (serving mode)");
            let (rank, world, link) = ctx.into_parts();
            gpu.attach_tp(rank, world, link);
            tp_stream = Some(stream);
        }

        let mtp_r = if gpu.mtp_present() && mtp_force != Some(false) {
            // The per-depth cost ratios are a stable function of (kernels, GPU, model) — independent of
            // the conversation — so cache them under <binary_dir>/mtp_calib/<model>.json and skip the
            // recalibration on subsequent launches. Keyed by model path + the binary's mtime, so a
            // rebuild (new kernels) transparently invalidates the cache and recalibrates.
            let calib_path = mtp_calib_cache_path(&model_path);
            let calib_stamp = mtp_calib_stamp(&model_path);
            // TP serving: bypass the cache READ. A hit skips the calibration forwards entirely, but
            // the node cannot know that (its model path/cache can never match the head's) and MUST
            // run the same SPMD forward sequence the head runs — an unshared skip deadlocks the
            // all-reduce barriers. Both ranks therefore always calibrate in lockstep; the cache
            // WRITE below still lands, so single-node launches keep the fast path.
            let cached = if tp { None } else { read_mtp_calib(&calib_path, &calib_stamp) };
            if let Some(rs) = cached {
                println!("MTP cost/depth: loaded from cache ({} depths) -> {}",
                         rs.len(), calib_path.as_ref().map(|p| p.display().to_string()).unwrap_or_default());
                for &(d, r) in &rs {
                    println!("    depth {}: a step costs {:.2}x a decode  (cached)", d, r);
                }
                rs
            } else {
            // Calibrate on a short canned prompt. Costs a few forward passes per candidate depth.
            println!("Calibrating MTP cost per depth...");
            let mut cpool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
            // profile_mtp probes the verify at N up to PROFILE_MAX_N with a checkpoint at slot 2, and
            // the GDN kernels write ONE checkpoint slot per verify column -- so slots 2..=(N) get
            // written. A 3-slot state here silently corrupted the heap past the buffer (the model then
            // decoded pure garbage), which is exactly the trap the per-column checkpoint introduces.
            // The calibration state stride is CAPPED, not max_seq_len. It probes a ~6-token prompt with
            // depth <= PROFILE_MAX_N, so it needs ~probe_len+depth positions -- a few dozen. Sizing it
            // to max_seq_len allocated a 10-SLOT KV cache at the full context length: 10 x 16.8 GB =
            // 168 GB at 256K, which OOM'd the box AT LOAD (the model was already resident). The main
            // scheduler cache below is correctly sized to max_seq_len; this transient probe is not.
            let calib_seq = 1024usize.min(max_seq_len);
            let mut cstate = gpu.new_batch_state(2 + gb10_inference::gpu::PROFILE_MAX_N, 2 + gb10_inference::gpu::PROFILE_MAX_N, calib_seq);
            let probe = tokenizer.encode("The capital of France is", true).expect("probe encode");
            let rs = gpu.calibrate_mtp_r(&mut cpool, &mut cstate, &probe, calib_seq);
            for &(d, r) in &rs {
                println!("    depth {}: a step costs {:.2}x a decode  (pays if it emits > {:.2} tok)",
                         d, r, r);
            }
            write_mtp_calib(&calib_path, &calib_stamp, &rs);
            rs
            }
        } else {
            vec![]
        };
        // TP serving: ship the head's cost table. The node has already run the identical SPMD
        // calibration forwards (and discarded its own table); both ranks' MtpPolicy must be built
        // from ONE identical set of numbers or the live depth decisions would diverge.
        if let Some(s) = tp_stream.as_mut() {
            let r32: Vec<(u32, f32)> = mtp_r.iter().map(|&(d, r)| (d as u32, r)).collect();
            gb10_inference::tp_serve::send_serving(s, &gb10_inference::tp_serve::ServingMsg::CalibTable { r: r32 })
                .expect("ship MTP calib table to node");
            println!("TP — MTP calib table shipped to node");
        }
        let policy = gb10_inference::batch::MtpPolicy::new(
            gpu.mtp_present(), mtp_force, mtp_depth, mtp_r);
        if !gpu.mtp_present() {
            println!("MTP: model has no MTP head â plain decode.");
        } else {
            println!("MTP: {}; depth {}; greedy requests verify by argmax (bitwise lossless), \
                      temp>0 requests by rejection sampling (distribution-exact).",
                     match mtp_force {
                         Some(true) => "FORCED ON".to_string(),
                         Some(false) => "FORCED OFF".to_string(),
                         None => "auto (disables itself if no depth beats plain decode)".to_string(),
                     },
                     match mtp_depth {
                         Some(d) => format!("PINNED at {}", d),
                         None => format!("starts at {}, re-picked from live acceptance", policy.depth()),
                     });
        }

        // Prefix caching is OPT-IN. It skips re-prefilling a conversation's history â on a 5-turn tool
        // conversation that is 97% of the prefill, and follow-up turns get ~3x faster. The price is that
        // reusing a prefix re-chunks the prefill, and prefill runs on cuBLAS, which picks a different
        // kernel per shape: a cached turn is NOT bit-identical to a cold one, so the same conversation
        // can word an answer slightly differently depending on cache state. That is a trade the operator
        // makes, not one we make for them. (Value parsed above, next to the TP config build.)
        if prefix_cache {
            println!("Prefix cache: ON â a conversation's history is reused instead of re-prefilled \
                      (~3x faster follow-up turns). Cached turns are NOT bit-identical to cold ones: \
                      reuse re-chunks the prefill and cuBLAS picks a kernel per shape.");
        } else {
            println!("Prefix cache: off â every request prefills its whole prompt (bit-exact, and \
                      slow on multi-turn agents: ~88% of prefill is recomputed). Enable: --prefix-cache on");
        }

        // Prompt-lookup n-gram drafting: EXPERIMENTAL, default OFF. Lossless (the verify checks every
        // draft, so output is byte-identical either way -- confirmed), but as a naive REPLACEMENT of the
        // MTP draft it is a net LOSS on the serving path: with real auto-depth the MTP acceptance
        // baseline is high (~74% on tool text), so a spurious short n-gram match replaces a GOOD draft
        // more often than it rescues a bad one (74% -> 66%, ~3 tok/s slower, measured). The right design
        // proposes the n-gram token as an ADDITIONAL candidate (tree verify, backlog #7), not a
        // replacement. Kept behind a flag for that work. `--ngram-draft 3` to experiment.
        if ngram_draft > 0 {
            println!("Prompt-lookup drafting: ON (order {ngram_draft}) â EXPERIMENTAL, measured net-negative vs MTP");
        }

        if tree_draft { println!("Tree drafting: ON (k=2 fork-then-chain) — EXPERIMENTAL, lossless, gated on yield"); }
        let (stx, srx) = tokio::sync::mpsc::unbounded_channel::<gb10_inference::batch::BatchRequest>();
        if mtp_lanes { println!("Batched MTP verify across lanes: ON -- EXPERIMENTAL, lossless (LANES_OK), packs concurrent greedy lanes into one verify"); }
        let scheduler = gb10_inference::batch::BatchScheduler::new(
            gpu, max_batch, max_seq_len, eos, srx, policy, prefix_cache, ngram_draft, tree_draft, mtp_lanes);
        // If the scheduler dies, the server must DIE WITH IT. It used to be a bare tokio::spawn: a panic
        // inside (an OOM, say) killed the task silently, and the HTTP layer went on accepting requests
        // and answering every one of them with ZERO TOKENS, forever. A loud crash is recoverable; a
        // zombie that looks healthy is not.
        match tp_stream {
            None => {
                tokio::spawn(async move {
                    if let Err(e) = tokio::spawn(scheduler.run()).await {
                        eprintln!("\n*** FATAL: the scheduler task died ({e}). The server cannot serve without \
                                   it and will not pretend to. Exiting. ***\n");
                        std::process::exit(70);
                    }
                    eprintln!("\n*** FATAL: the scheduler loop returned unexpectedly. Exiting. ***\n");
                    std::process::exit(70);
                });
            }
            Some(mut s) => {
                // Bind HTTP only after the node's mirror is armed: a client request admitted before
                // the mirror's first Step recv would desync the lockstep.
                match gb10_inference::tp_serve::recv_serving(&mut s).expect("node Ready") {
                    gb10_inference::tp_serve::ServingMsg::Ready =>
                        println!("TP -- node READY (mirror scheduler armed); binding HTTP"),
                    other => panic!("expected Ready from node, got {other:?}"),
                }
                // TP wants the launch thread PINNED (an unpinned launch thread presents exactly like
                // a protocol stall -- the 9.0->15.1 tok/s lesson), so the scheduler gets a dedicated
                // pinned thread with a current-thread runtime, not a tokio worker (a task can migrate
                // between workers at every await). Same die-with-it rule as the single-node spawn.
                std::thread::spawn(move || {
                    if !gb10_inference::net::pin_thread(9) {
                        eprintln!("\n*** FATAL: TP head scheduler failed to pin to core 9 -- TP refuses \
                                   to run unpinned. Exiting. ***\n");
                        std::process::exit(70);
                    }
                    let rt = tokio::runtime::Builder::new_current_thread().enable_all()
                        .build().expect("scheduler runtime");
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                        || rt.block_on(scheduler.run_tp_head(s))));
                    match res {
                        Ok(Ok(())) => eprintln!("\n*** FATAL: the TP scheduler loop returned unexpectedly. Exiting. ***\n"),
                        Ok(Err(e)) => eprintln!("\n*** FATAL: the TP scheduler loop failed: {e:#}. Exiting. ***\n"),
                        Err(_) => eprintln!("\n*** FATAL: the TP scheduler task panicked. Exiting. ***\n"),
                    }
                    std::process::exit(70);
                });
            }
        }

        let state = AppState {
            scheduler: stx,
            tokenizer: Arc::new(tokenizer),
            model_name: model_name.clone(),
            default_max_tokens,
            default_rep_penalty,
            default_presence_penalty,
            default_frequency_penalty,
            reasoning_effort: parse_arg(args, "--reasoning-effort").map(|s| s.to_string())
                .map(|e| match e.as_str() {
                    "no_think" | "low" | "high" => e,
                    other => {
                        eprintln!("--reasoning-effort must be no_think|low|high (got '{other}')");
                        std::process::exit(1);
                    }
                })
                .unwrap_or_else(|| "no_think".to_string()),
            max_seq_len,
        };

        let app = create_router(state);
        let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await.unwrap();
        println!("OpenAI-compatible server running on http://0.0.0.0:{}", port);
        println!("Serving model: {}  (GET /v1/models)", model_name);
        println!("POST /v1/chat/completions   max_batch={}  default max_tokens={}", max_batch, default_max_tokens);
        axum::serve(listener, app).await.unwrap();
    });
}
