//! `tp_serve` (the trunk TP serve driver) and the DSV4 bundle detection / load / TP serve entry points.
//! The DSV4 and DSpark TP servers are in `dsv4_serve.rs`.

use super::*;

/// Shared TP=2 Proof-v0 serve path for both roles: sanity-check the link, broadcast/receive the
/// prompt, load the model + attach the TP link, run the SPMD `tp_generate`, and (head only) decode+print.
pub(super) fn tp_serve(model_dir: &str, ctx: anyhow::Result<gb10_inference::tp::TpContext>,
            head_req: Option<(String, usize)>) -> anyhow::Result<()> {
    let is_head = head_req.is_some();
    let mut ctx = ctx?;
    let role = format!("{} (rank {}/{})", if is_head { "HEAD" } else { "NODE" }, ctx.rank, ctx.world);
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
    let (prompt, max_new, _) = ctx.broadcast_prompt(
        head_payload.as_ref().map(|(ids, m)| (ids.as_slice(), *m, 0)))?;
    println!("{role} — SPMD decode: {} prompt tokens, max_new {max_new}", prompt.len());

    // Node-side TP env that must agree with the head before load (the 4-bit KV cache changes the
    // cache layout on BOTH ranks; the head ships it in the TpConfig).
    if gb10_inference::tp::tp_config().map(|c| c.kv_quant).unwrap_or(false) {
        gb10_inference::opts::set(gb10_inference::opt!("kv-quant"), "1");
    }
    // TurboQuant KV (E4): same SPMD rule as kv_quant — the cache layout must match both ranks.
    let tpc_tq = gb10_inference::tp::tp_config();
    if tpc_tq.map(|c| c.kv_tq).unwrap_or(false) {
        let v = if tpc_tq.map(|c| c.kv_tq_b3).unwrap_or(false) { "3" } else { "1" };
        gb10_inference::opts::set(gb10_inference::opt!("kv-tq"), v);
    }
    // k8v4 KV (int8-K + q4-V): same SPMD rule; mutually exclusive with kv_quant/kv_tq.
    if gb10_inference::tp::tp_config().map(|c| c.kv_k8v4).unwrap_or(false) {
        gb10_inference::opts::set(gb10_inference::opt!("kv-k8v4"), "1");
    }
    // Same class for the verify-graph escape (eager verify on BOTH ranks — the serving install at
    // node_serve_tp covers that path; the bench path (tp_serve) needs its own here).
    if gb10_inference::tp::tp_config().map(|c| c.no_verify_graph).unwrap_or(false) {
        gb10_inference::opts::set(gb10_inference::opt!("no-verify-graph"), "1");
    }
    // MXFP4-native mode (--mxfp4=on) + the MTP-head allowlist escape hatch — SPMD-relevant, same
    // rule as kv_quant: the loader reads --mxfp4 as env on BOTH ranks; the bench node installs
    // the head's shipped decision here (the serving path installs at node_serve_tp).
    if gb10_inference::tp::tp_config().map(|c| c.mxfp4).unwrap_or(false) {
        gb10_inference::opts::set(gb10_inference::opt!("mxfp4"), "1");
    }
    if gb10_inference::tp::tp_config().map(|c| c.mxfp4_mtp_native).unwrap_or(false) {
        gb10_inference::opts::set(gb10_inference::opt!("mxfp4-mtp-native"), "1");
    }

    // SPMD branch resolution: WHICH program both ranks run. Every gate resolves env-first
    // (override), then the installed TpConfig (shipped by the head during the sync), then the
    // no-env default. An env-ONLY gate would put the zero-config node on a different program than
    // the head — that was the --tp-accept split-brain: two ranks, one link, deterministically
    // mismatched all-reduce epochs (silent garbage, healthy-looking output). Resolve in ONE place,
    // then prove agreement over the link before any kernel runs (branch_check).
    use gb10_inference::tp::TpBranch;
    let tpc = gb10_inference::tp::tp_config();
    let branch = {
        let capture = gb10_inference::opts::var(gb10_inference::opt!("tp-capture")).ok()
            .or_else(|| tpc.and_then(|c| c.capture.clone()));
        let accept = gb10_inference::opts::var(gb10_inference::opt!("tp-accept")).ok().map(|v| v.parse().unwrap_or(2))
            .or(tpc.and_then(|c| c.accept));
        if let Some(cap_out) = capture { TpBranch::Capture(cap_out) }
        else if let Some(d) = accept { TpBranch::Accept(d) }
        else if gb10_inference::opts::var(gb10_inference::opt!("tp-mtp")).is_ok() || tpc.map(|c| c.mtp).unwrap_or(false) {
            TpBranch::Mtp
        }
        // Phase-7 F9 gate: the TP batch-invariance probe (SPMD, both ranks).
        else if gb10_inference::opts::var(gb10_inference::opt!("tp-binv-tp")).is_ok()
            || tpc.map(|c| c.binv_tp).unwrap_or(false) {
            TpBranch::BinvTp
        }
        else if gb10_inference::opts::var(gb10_inference::opt!("tp-state-tp")).is_ok()
            || tpc.map(|c| c.state_tp).unwrap_or(false) {
            TpBranch::StateTp
        } else {
            let step_probe = match gb10_inference::opts::var(gb10_inference::opt!("tp-step-probe")) {
                Ok(d) => Some(d.parse().unwrap_or(4)),
                Err(_) => tpc.and_then(|c| c.step_probe),
            };
            let batch_probe = match gb10_inference::opts::var(gb10_inference::opt!("tp-batch-probe")) {
                Ok(n) => Some(n.parse().unwrap_or(1)),
                Err(_) => tpc.and_then(|c| c.batch_probe),
            };
            // P0-2: --tp-decode-ctx rode env-ONLY here while every other gate resolved
            // env-first-then-shipped-config — a head-only env put the zero-config node on
            // TpBranch::Generate against the head's DecodeCtx (the observed "rank 1 selector
            // 0x00, head 0x60", 2026-08-15). Same resolution ladder as the rest now.
            let decode_ctx = match gb10_inference::opts::var(gb10_inference::opt!("tp-decode-ctx")) {
                Ok(c) => Some(c.parse().unwrap_or(2048)),
                Err(_) => tpc.and_then(|c| c.decode_ctx),
            };
            if let Some(d) = step_probe { TpBranch::StepProbe(d) }
            else if let Some(n) = batch_probe { TpBranch::BatchProbe(n) }
            else if let Some(c) = decode_ctx { TpBranch::DecodeCtx(c) }
            else { TpBranch::Generate }
        }
    };
    ctx.branch_check(&branch)?;

    // Load the (whole, replicated) model and attach the TP link → the forward runs the sharded FFN
    // all-reduce (world==2). Same binary/model on both boxes, so the compute is identical. (hy_v3:
    // the loader shards to ctx.rank host-side — the full model does not fit one node.)
    let (mut gpu, _cfg) = gb10_inference::gpu::GpuModel::load_from_dir_tp(model_dir, ctx.rank, ctx.world)?;
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

    let max_seq_len = (prompt.len() + max_new + 16).next_power_of_two().max(256);

    match branch {
        // TP-aware per-layer capture (hy_v3 oracle localization): BOTH ranks run the identical
        // batched prefill in SPMD lockstep (the all-reduces fire inside), and each writes its own
        // dump — the two files must be bit-identical to each other (SPMD check) and comparable to
        // the oracle per layer per position (scripts/compare_hy3_oracle.py).
        TpBranch::Capture(cap_out) => {
            use safetensors::{Dtype, tensor::TensorView};
            let n = prompt.len();
            let h = gpu.cfg().hidden_size;
            let nlayers = gpu.cfg().num_layers;
            if gpu.cfg().is_q4() {
                // qwen4_exp (B15): the residual is rw-wide (hc streams × h) — capture_prefill
                // cannot see it. Run the SAME PP-prefix ladder as --capture-layers' q4 branch on
                // BOTH ranks in SPMD lockstep (the all-reduces fire inside); each rank writes its
                // own dump, the two files must be bit-identical (SPMD check), and both are
                // layer-comparable to the TP=1 capture (the sharding localization).
                let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
                let named = capture_layers_q4(&mut gpu, &mut pool, &prompt);
                let rw = gpu.cfg().resid_width();
                let views: Vec<(String, TensorView)> = named.iter()
                    .map(|(name, dmp)| {
                        let bytes: &[u8] = bytemuck::cast_slice(&dmp[..]);
                        (name.clone(), TensorView::new(Dtype::BF16, vec![n, rw], bytes).expect("view"))
                    }).collect();
                safetensors::serialize_to_file(views, None, std::path::Path::new(&cap_out))
                    .expect("write safetensors");
                println!("{role} — TP q4 capture: {n} tokens x {rw} wide x {} boundaries -> {cap_out}",
                         nlayers + 1);
                println!("CAPTURE_OK {cap_out}");
                return Ok(());
            }
            let kv_stride = n.max(16);
            let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
            let mut state = gpu.new_batch_state(1, 1, kv_stride);
            let dumps = gpu.capture_prefill(&mut pool, &prompt, &mut state, kv_stride);
            // --cap-debug interleaves mixer/mlp_out dumps per layer (count check only without it).
            if gb10_inference::opts::var(gb10_inference::opt!("cap-debug")).is_err() {
                assert_eq!(dumps.len(), nlayers + 2, "capture count: embed + L layers + final_norm");
            }
            let mut named: Vec<(String, Vec<half::bf16>)> = Vec::with_capacity(dumps.len());
            let dbg = gb10_inference::opts::var(gb10_inference::opt!("cap-debug")).is_ok();
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
        // TP=2 acceptance diagnosis (--tp-accept=<depth>): both ranks run bench_accept in SPMD
        // (the main forwards barrier; the replicated drafter cannot diverge), rank 0 prints the
        // discriminator report. This is the instrument for "is the draft head weak or is the text hard".
        TpBranch::Accept(depth) => {
            let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
            let mut state = gpu.new_batch_state(1, 2 + depth, max_seq_len);
            let (s, generated) = gpu.bench_accept(&mut pool, &mut state, &prompt, max_seq_len, depth, max_new, 0)?;
            if is_head {
                assert!(!s.is_empty(), "bench_accept produced NO samples");
                assert!(generated.len() > 8, "bench_accept generated almost nothing");
                bench_accept_report(depth, &s);
            }
            return Ok(());
        }
        // v1 MTP under TP: run the real speculative loop on the sharded model.
        TpBranch::Mtp => {
            let depth: usize = gb10_inference::opts::var(gb10_inference::opt!("tp-mtp-depth")).ok()
                .and_then(|v| v.parse().ok())
                .or(tpc.and_then(|c| c.mtp_depth)).unwrap_or(4);
            let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
            let slots = 2 + depth.saturating_sub(1).max(1);
            let mut st = gpu.new_batch_state(slots, slots, max_seq_len);
            // E13: the load-time MTP-calibration forwards pollute the capture sink — reset right
            // before the bench so a PRFX/FOLDX dump holds exactly this run's prefill + steps.
            if gb10_inference::opts::var(gb10_inference::opt!("fold-xchain-dump")).is_ok() {
                gb10_inference::gpu::xchain_capture_reset();
            }
            let (mtp_toks, seq_toks, mtp_tps, seq_tps, acc) =
                gpu.bench_mtp(&mut pool, &mut st, &prompt, max_seq_len, depth, max_new);
            let lossless = mtp_toks == seq_toks;
            println!("{role} — TP+MTP depth {depth}: {:.1} tok/s (sequential {:.1}), acceptance {:.1}%, {}",
                     mtp_tps, seq_tps, acc * 100.0,
                     if lossless { "LOSSLESS_OK" } else { "DIVERGED vs sequential" });
            if is_head {
                println!("GATE_TOKENS {}", mtp_toks.iter().map(|t| t.to_string())
                         .collect::<Vec<_>>().join(","));
                // E13: wide-prefill dump (--fold-xchain-dump=<path> +
                // --mxfp4-xchain-capture=1) — the batched prefill's per-layer final-row
                // hiddens. The fold fires only at batch > MAX_VERIFY (the wide prefill) and is
                // NONDETERMINISTIC there — two fold-on runs of the same prompt differ. The dump
                // localizes which layers race (compare two runs of the same config).
                if let Ok(path) = gb10_inference::opts::var(gb10_inference::opt!("fold-xchain-dump")) {
                    let caps = gb10_inference::gpu::xchain_capture_take();
                    let mut chain = prompt.clone();
                    chain.extend_from_slice(&mtp_toks);
                    gb10_inference::gpu::prfx_write(&path, &caps, &chain, prompt.len())
                        .map_err(|e| anyhow::anyhow!("prfx dump: {e}"))?;
                    println!("PRFX -> {}", path);
                }
            }
            return Ok(());
        }
        // Phase-7 F9 gate: verify-width sweep through the REAL sharded verify path. Both
        // ranks run it in SPMD lockstep (the all-reduces pair up); the head prints the
        // verdict. Failure bails loudly on BOTH ranks.
        TpBranch::BinvTp => {
            let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
            let _ = &mut pool;
            let ok = gpu.probe_binv_tp(&prompt, max_seq_len, gb10_inference::gpu::MAX_VERIFY);
            if !ok {
                anyhow::bail!("PROBE-BINV-TP FAIL — col 0 of a wide verify is not bit-identical to N=1");
            }
            return Ok(());
        }
        // Phase-7: the GDN decode-vs-verify state probe on the SHARDED path (the
        // single-process --probe-state gate cannot see the TP-only residual).
        TpBranch::StateTp => {
            let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
            let ok = gpu.probe_state_tp(&mut pool, &prompt, max_seq_len, gb10_inference::gpu::MAX_VERIFY);
            if !ok {
                anyhow::bail!("PROBE-STATE-TP FAIL — verify and decode GDN state differ on the sharded path");
            }
            return Ok(());
        }
        // Probe 2: measure a synthetic MTP step under TP directly.
        TpBranch::StepProbe(d) => {
            gpu.tp_synthetic_step_probe(d, 20, max_seq_len);
            return Ok(());
        }
        // Q6 probe: does "a batch-N forward costs ~= a batch-1 forward" survive TP? Runs INSTEAD of decode.
        TpBranch::BatchProbe(n) => {
            gpu.tp_batch_probe(n, 30, max_seq_len);
            return Ok(());
        }
        // E11: decode/MTP step p50 at context c (SPMD lockstep — the all-reduces pair up; the
        // zero-KV timing is value-independent). The head prints the per-phase table.
        TpBranch::DecodeCtx(c) => {
            let kv_stride = (c + 64).next_power_of_two().max(2048);
            let rows = gpu.bench_decode_at_ctx(c, kv_stride, 3);
            if is_head {
                println!("=== E11 decode/MTP step p50 at ctx={c} (best-of-3, zeros KV, TP=2) ===");
                for (name, ms) in &rows {
                    println!("  {name}: {ms:.2} ms");
                }
            }
            return Ok(());
        }
        TpBranch::Generate => {}
        TpBranch::Exl3Xtp => anyhow::bail!("the EXL3 xtp program runs through exl3_forward::xtp::run_tp_xtp, never tp_serve"),
        TpBranch::Naive => anyhow::bail!("the Naive-N0.5 program runs through naive::serve, never tp_serve"),
    }
    let t0 = std::time::Instant::now();
    // E29-B2: the capture sink accumulates MTP-calibration steps at load (batch==1 too) — reset
    // right before generation so the DFCTX dump holds exactly this run's prefill + decode steps.
    if gb10_inference::opts::var(gb10_inference::opt!("xchain-ctx-dump")).is_ok() || gb10_inference::opts::var(gb10_inference::opt!("fold-xchain-dump")).is_ok() {
        gb10_inference::gpu::xchain_capture_reset();
    }
    let out = gpu.tp_generate(&prompt, max_new, max_seq_len);
    let dt = t0.elapsed();
    gpu.tp_trace_dump(&role);

    if is_head {
        let tokenizer = QwenTokenizer::from_file(&tok_path)?;
        let text = tokenizer.decode(&out, true).unwrap_or_default();
        let tps = if dt.as_secs_f32() > 0.0 { out.len() as f32 / dt.as_secs_f32() } else { 0.0 };
        println!("\n===== TP=2 PROOF v0 OUTPUT ({} tokens, {:.1} tok/s) =====", out.len(), tps);
        println!("{text}");
        println!("===== token ids: {:?}", out);
        println!("GATE_TOKENS {}", out.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
        // E29-B2: DFlash context-chain dump — --xchain-ctx-dump=<path> arms the xchain
        // capture (--mxfp4-xchain-capture=1 must also be set) and writes the drafter's
        // conditioning hiddens (layers 1/20/39/58/77) per step + the full token chain.
        if let Ok(path) = gb10_inference::opts::var(gb10_inference::opt!("xchain-ctx-dump")) {
            let caps = gb10_inference::gpu::xchain_capture_take();
            anyhow::ensure!(caps.len() - 1 == out.len(),
                "xchain ctx dump: {} captures vs {} generated tokens (sink must be [final prefill step, decode steps...])",
                caps.len(), out.len());
            let mut chain = prompt.clone();
            chain.extend_from_slice(&out);
            gb10_inference::gpu::xchain_ctx_write(&path, &caps, &chain, prompt.len())
                .map_err(|e| anyhow::anyhow!("xchain ctx dump: {e}"))?;
            println!("XCHAIN_CTX {} decode steps -> {}", out.len(), path);
        }
        // E13: fold-vs-nofold cross-chain dump — --fold-xchain-dump=<path> (+
        // --mxfp4-xchain-capture=1) writes ALL layers' per-step outputs + the token chain.
        // Run the same prompt once with --moe-no-fold=1 (reference) and once with
        // --moe-fold=1, then compare with --probe-fold-xchain-compare.
        if let Ok(path) = gb10_inference::opts::var(gb10_inference::opt!("fold-xchain-dump")) {
            let caps = gb10_inference::gpu::xchain_capture_take();
            anyhow::ensure!(caps.len() == out.len(),
                "fold xchain dump: {} captures vs {} generated tokens", caps.len(), out.len());
            let mut chain = prompt.clone();
            chain.extend_from_slice(&out);
            gb10_inference::gpu::xchain_fold_write(&path, &caps, &chain, prompt.len())
                .map_err(|e| anyhow::anyhow!("fold xchain dump: {e}"))?;
            println!("FOLD_XCHAIN {} decode steps -> {}", out.len(), path);
        }
    } else {
        println!("{role} — generated {} tokens in lockstep (head prints)", out.len());
    }
    Ok(())
}

/// Detect a DSV4 bundle (config.json `model_type == "deepseek_v4"`) — the cluster-arm dispatch.
pub(super) fn is_dsv4_bundle(model_dir: &std::path::Path) -> bool {
    // The HF `model_type: deepseek_v4` lives in the ROOT config.json (both the bundle and a
    // converted artifact — the converter copies it there). inference/config.json is the reference's
    // own ModelArgs format and carries no model_type.
    let cfg_path = model_dir.join("config.json");
    std::fs::read_to_string(&cfg_path)
        .map(|s| s.contains("\"deepseek_v4\""))
        .unwrap_or(false)
}

/// Load the DSV4 trunk for serving — converted (fast) when a `manifest.json` artifact is present,
/// else the streaming `load_tp`. TP=2: head + node call this concurrently, each loading its ~84 GB
/// shard in parallel (the wall-clock load time is one node's load, not both summed). Returns the
/// model + the per-node load seconds.
pub(super) fn dsv4_load_for_serve(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    model_dir: &std::path::Path,
    cfg: &gb10_inference::dsv4_load::Dsv4Config,
    max_seq_len: usize,
    s_max: usize,
    rank: usize,
    world: usize,
) -> anyhow::Result<(gb10_inference::dsv4_model::Dsv4GpuModel, f64)> {
    let t0 = std::time::Instant::now();
    // Converted (fast) when ANY artifact manifest is present: the flat `manifest.json` OR the
    // per-rank `rank{rank}/manifest.json` (load_converted itself detects + loads the rank subdir).
    let m = if model_dir.join(format!("rank{rank}")).join("manifest.json").exists()
        || model_dir.join("manifest.json").exists()
    {
        eprintln!("[dsv4] artifact manifest detected → load_converted (fast path, rank {rank}/{world})");
        gb10_inference::dsv4_model::Dsv4GpuModel::load_converted(dev, model_dir, cfg, max_seq_len, s_max, cfg.n_layers, rank, world)?
    } else {
        eprintln!("[dsv4] no manifest → streaming load_tp (rank {rank}/{world})");
        gb10_inference::dsv4_model::Dsv4GpuModel::load_tp(dev, model_dir, cfg, max_seq_len, s_max, cfg.n_layers, rank, world)?
    };
    Ok((m, t0.elapsed().as_secs_f64()))
}

/// DSV4 TP=2 SPMD serve (the minimal first-light path — no BatchScheduler, no verify graphs, no MTP;
/// §6b "faster first-light option"). Both ranks run the IDENTICAL greedy loop; the per-layer routed
/// all-reduces (inside `block_forward`'s TP path) keep the node bit-for-bit in step with the head.
/// Head encodes the prompt and prints the decoded response; the node mirrors silently. The trunk-top
/// head runs FULL-vocab + replicated here (proven by tp-sim-full) — vocab-parallel is a later memory
/// optimization. Stop at EOS (generation_config `eos_token_id = 1`).
pub(super) fn dsv4_tp_serve(
    model_dir: &str,
    ctx_result: anyhow::Result<gb10_inference::tp::TpContext>,
    head_req: Option<(String, usize)>,
) -> anyhow::Result<()> {
    use gb10_inference::dsv4_load;
    use gb10_inference::dsv4_model::Dsv4GpuModel;
    use cudarc::driver::CudaDevice;

    let is_head = head_req.is_some();
    let role = if is_head { "HEAD (rank 0/2)" } else { "NODE (rank 1/2)" };
    let mut ctx = ctx_result?;
    ctx.sanity()?;
    println!("{role} — DSV4 TP LINK UP");

    // Head encodes; both ranks agree on (prompt ids, max_new) via the link broadcast.
    let tok_path = format!("{}/tokenizer.json", model_dir.trim_end_matches('/'));
    let head_payload = match &head_req {
        Some((text, max_new)) => {
            let tok = QwenTokenizer::from_file(&tok_path)?;
            let mut ids = tok.encode(text, true)?;
            // --bisect-len=N: pad/truncate to exactly N tokens (deterministic valid ids) — drives
            // the TP prefill-length bisect with precise token counts (the prompt text is just a seed pool).
            if let Some(len_str) = gb10_inference::opts::var(gb10_inference::opt!("bisect-len")).ok() {
                let len: usize = len_str.parse().expect("--bisect-len N");
                ids.truncate(len.min(ids.len()));
                while ids.len() < len {
                    // deterministic valid non-EOS ids (mirrors the single-process --prompt-len bisect)
                    ids.push((((7 + ids.len() as i64 * 9973) % 129040) as u32).max(2));
                }
                eprintln!("[dsv4-bisect] prompt → {} tokens (--bisect-len={len})", ids.len());
            } else {
                println!("{role} — prompt {text:?} → {} tokens", ids.len());
            }
            Some((ids, *max_new))
        }
        None => None,
    };
    let (prompt, max_new, _) = ctx.broadcast_prompt(
        head_payload.as_ref().map(|(ids, m)| (ids.as_slice(), *m, 0)))?;
    let (rank, world, link) = ctx.into_parts();
    println!("{role} — SPMD DSV4 decode: {max_new} new tokens, prompt {} tok", prompt.len());

    // Load the FULL trunk (43 layers), sharded per rank. s_max covers ONE prefill chunk, not the
    // prompt: Dsv4GpuModel::forward processes prompts > PREFILL_CHUNK (4096) in 128-aligned chunks
    // (§12.B.5 chunked prefill — bitwise-identical to one-shot), so all prefill scratch is
    // chunk-sized and 200K+ prompts fit the memory budget (DSV4_LONG_CONTEXT_1M §3).
    let cfg = dsv4_load::load_config(std::path::Path::new(model_dir))
        .map_err(|e| anyhow::anyhow!("load_config {model_dir}: {e:#}"))?;
    let prompt_len = prompt.len();
    // --max-seq-len overrides the derived cache depth (memory-budget probes: 1M-sized
    // caches with a short prompt). Default: prompt + generation headroom.
    let max_seq_len = gb10_inference::opts::var(gb10_inference::opt!("max-seq-len")).ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or((prompt_len + max_new + 256).max(2048));
    let s_max = (prompt_len + 16).max(256).min(gb10_inference::dsv4_model::PREFILL_CHUNK);
    let dev = CudaDevice::new(0)?;
    let (mut m, load_secs) = dsv4_load_for_serve(&dev, std::path::Path::new(model_dir), &cfg,
                                                  max_seq_len, s_max, rank as usize, world as usize)
        .map_err(|e| anyhow::anyhow!("dsv4_load_for_serve {model_dir} rank {rank}: {e:#}"))?;
    println!("{role} — shard load: {load_secs:.1}s");
    m.attach_tp(rank, world, link);

    // SPMD greedy: prefill (one-shot) → decode loop. R3.2: under TP the next token comes from the
    // vocab-parallel maxloc head (bitwise == full-vocab argmax); single-process keeps full logits
    // + host argmax. Both ranks compute the same winner (logits match post-reduce / the maxloc
    // total order), so the loops stay in lockstep.
    let prompt_i32: Vec<i32> = prompt.iter().map(|&v| v as i32).collect();
    let maxloc = m.rt.tp_ctx_dptr != 0 && m.tp_rank >= 0;
    let eos = 1u32; // generation_config.json eos_token_id
    let mut generated: Vec<u32> = Vec::with_capacity(max_new);
    let mut pos = prompt_len;
    let mut next: u32 = if maxloc {
        m.forward_next(&prompt_i32, 0)?.expect("maxloc head under TP")
    } else {
        let l = m.forward(&prompt_i32, 0)?;
        dsv4_argmax(&dev.dtoh_sync_copy(&l)?) as u32
    };
    let t0 = std::time::Instant::now();
    for step in 0..max_new {
        if next == eos {
            println!("{role} — EOS at step {step}; stopping.");
            break;
        }
        generated.push(next);
        next = if maxloc {
            m.forward_next(std::slice::from_ref(&(next as i32)), pos)?.expect("maxloc head under TP")
        } else {
            let l = m.forward(std::slice::from_ref(&(next as i32)), pos)?;
            dsv4_argmax(&dev.dtoh_sync_copy(&l)?) as u32
        };
        pos += 1;
    }
    let dt = t0.elapsed().as_secs_f64();
    if is_head {
        let tok = QwenTokenizer::from_file(&tok_path)?;
        let text = tok.decode(&generated, true)?;
        let n = generated.len();
        println!("\n=== DSV4 TP=2 response ({n} tokens in {dt:.2}s, {:.1} tok/s) ===", n as f64 / dt.max(1e-9));
        println!("    prompt: {:?}", head_req.as_ref().unwrap().0);
        println!("    response: {text}");
        println!("    raw ids: {generated:?}");
    }
     Ok(())
}
