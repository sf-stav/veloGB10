//! `run_server`: the OpenAI-compatible server entry (trunk NVFP4/bf16 path), one function kept whole.
//! Pinned by source-text tests (batch cap wiring, DSV4 option checks); the TP and DSV4 serve drivers it
//! dispatches to live in `tp_serve_trunk.rs`, `tp_heads.rs`, `dsv4_serve.rs`.

use super::*;

pub(crate) fn run_server(args: &[String]) {
    // Ported from sf-stav/veloGB10 PR#5 (Morxi): honor --rdma-dev → --rdma-dev here. The TP
    // data-plane (tp::rdma_dev / cluster run_head_ session) reads ONLY the env var; on the
    // bench/`serve` paths the flag was already folded in, but the `--server` path never set it,
    // so it silently fell back to the DEFAULT_RDMA_DEV (rocep1s0f1). On a GB10 whose RoCE rail
    // is the P2P1 port, bringing the QP up on the wrong device makes the RTR ibv_modify_qp fail
    // ("[net_shim] RTR" → "TP bring-up: net_init failed").
    // (--rdma-dev is a registry option: parsed at startup, node-local, never shipped)
    // Validate the draft-dir rule FIRST — an explicit drafter --spec-source without a valid
    // --draft-dir must stop HERE, before any model load or GPU work (resolve_draft_dir is pure:
    // exits 2 on the violation, returns the dir otherwise; every downstream consumer re-resolves
    // the identical value from the same flag).
    let _ = resolve_draft_dir(args, Some(resolve_spec_source(args)));
    // PLAN/25 Phase 0: the coverage-trace dump (--df2-step-dump <dir>).
    // Diagnostic-only: with it unset the server is byte-identical to the standing path. A dump
    // dir we cannot create is an operator error — refuse loudly rather than serve silently
    // undumped (the FAILS-OPEN rule).
    let cov_dump_dir: Option<String> = parse_arg(args, "--df2-step-dump").map(|s| s.to_string());
    let cov_dump = match cov_dump_dir.as_deref()
    {
        Some(dir) => match gb10_inference::dflash2::stepdump::StepDump::new(&dir) {
            Ok(d) => {
                eprintln!("[df2-dump] coverage trace ON → {dir} (records: steps.jsonl; the trace flag rides TpConfig, so node mirrors run the identical op sequence)");
                Some(d)
            }
            Err(e) => {
                eprintln!("[df2-dump] cannot create {dir}: {e} — exiting (fix the path; the server will not pretend to dump)");
                std::process::exit(2);
            }
        },
        None => None,
    };
    // Support both --model-dir <DIR> and legacy --model <FILE> + --tokenizer <FILE>
    let (model_path, tokenizer_path) = if let Some(dir) = parse_arg(args, "--model-dir") {
        (dir.to_string(), format!("{}/tokenizer.json", dir.trim_end_matches('/')))
    } else {
        let model = parse_arg(args, "--model").unwrap_or("model/model.safetensors");
        let tokenizer = parse_arg(args, "--tokenizer").unwrap_or("model/tokenizer.json");
        (model.to_string(), tokenizer.to_string())
    };

    // S-A3-e: an EXL3 pack routes to the dedicated serve backend (SAME HTTP surface via the
    // shared BatchRequest protocol; TP=1 - EXL3 TP is S-A3-g). Detection: quantization_config
    // manifest + a .trellis-carrying index.
    if parse_arg(args, "--model-dir").is_some() && !std::path::Path::new(&model_path).is_dir() {
        eprintln!("error: --model-dir {model_path} does not exist on this box (TP nodes do not need it; the head does)");
        std::process::exit(2);
    }
    // WIDEVERIFY1: --copy-span is built into the EXL3 speculative round only; any other pack's server runs
    // unchanged and says so once (velogb10_copy_span_state 3). Never a refusal (AGENTS 1b-bis item 8).
    // COPY27B: the trunk server (`BatchScheduler`, NVFP4/bf16 packs) resolves the flag itself when it is
    // built; only the bundles served by other servers (DeepSeek V4, Naive) take the family fallback.
    if !gb10_inference::exl3_serve::is_exl3_pack(&model_path)
        && (is_dsv4_bundle(std::path::Path::new(&model_path))
            || gb10_inference::naive::config::NaiveConfig::is_naive_dir(&model_path)) {
        gb10_inference::exl3_serve::copy_span_family_fallback();
    }
    // NAIVE S5: a Naive-N0.5 pack serves TP-only (137 GB; no single GB10 holds it): this box is the head
    if gb10_inference::naive::config::NaiveConfig::is_naive_dir(&model_path) {
        match parse_tp_world(args) {
            Some(w) if w >= 2 => { run_naive_tp_head(args, &model_path, "serve"); return; }
            _ => { eprintln!("Naive-N0.5: the pack needs --tp 2 (two GB10s); start the node with `--node` on the other box"); std::process::exit(2); }
        }
    }
    if gb10_inference::exl3_serve::is_exl3_pack(&model_path) {
        // TP-D: --server --tp 2 on an EXL3 pack = the TP=2 served head (rank 0; node = --node)
        if parse_tp_world(args).map_or(false, |w| w >= 2) {
            run_exl3_tp_serve_head(args, &model_path);
            return;
        }
        if let Err(e) = gb10_inference::exl3_serve::run(args, &model_path) {
            gb10_inference::boot_report::error(&format!("exl3 serve: {e:#}"));
            std::process::exit(1);
        }
        return;
    }
    let port = parse_arg(args, "--port").and_then(|s| s.parse::<u16>().ok()).unwrap_or(8000);
    // E5 (512K mandate): --rope-yarn-factor <f> — trunk YaRN override, mirrored into the env
    // BEFORE the model load (rope tables build at load; single-box has no TpConfig channel).
    // Under --tp the value also rides TpConfig to the node (SPMD — both ranks' tables match).
    if let Some(f) = gb10_inference::opt_or_arg(args, "--rope-yarn-factor").and_then(|v| v.parse::<f32>().ok()) {
        if f >= 1.0 { gb10_inference::opts::set(gb10_inference::opt!("rope-yarn-factor"), f.to_string()); }
    }
    let max_seq_len = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse::<usize>().ok()).unwrap_or(4096);
    // PREHARD1 [ISSUES 104] (owner ruling 2026-10-10 §1b-bis#8; MAXBATCH1's EXL3 pattern ported
    // to the trunk NVFP4 serve path): at most 16 lanes serve at once. A larger --max-batch BOOTS
    // ANYWAY at the cap — one clear line says what was asked and what runs instead, further
    // requests wait in the queue, and /metrics velogb10_max_batch{,_requested} carry both numbers.
    // Every rank parses the same argv, so the clamp is SPMD-identical before TpConfig ships it.
    // (EXL3's boot-anyway memory clamp — fewer lanes when the asked --max-seq-len does not fit —
    // stays EXL3: the trunk has no fit planner. Nothing refuses; nothing silently cuts context.)
    let max_batch_raw = parse_arg(args, "--max-batch").and_then(|s| s.parse::<usize>().ok()).unwrap_or(8);
    let max_batch = max_batch_raw.min(gb10_inference::batch::SERVE_MAX_LANES);
    gb10_inference::metrics::set_max_batch_requested(max_batch_raw);
    if max_batch_raw > max_batch {
        gb10_inference::boot_report::notice(&format!(
            "--max-batch {max_batch_raw}: this server serves at most {} lanes at once (the same \
             cap as the EXL3 server); booting with {max_batch} — further requests wait in the queue",
            gb10_inference::batch::SERVE_MAX_LANES));
    }
    // W3 (Phase 13): fail FAST on an unsupported KV-mode x lane-width combination — before the
    // model load and before any node contact. Used to surface as a panic inside the attention
    // dispatch after a full load (owner's TP4 k8v8 repro).
    {
        let tp_for_check = parse_tp_world(args).unwrap_or(1) as usize;
        if let gb10_inference::kv_cache::KvLaneVerdict::Reject(msg) =
            gb10_inference::kv_cache::kv_lane_check(gb10_inference::opt_or_arg(args, "--kv-cache").as_deref(), max_batch, tp_for_check)
        {
            eprintln!("error: {msg}");
            std::process::exit(2);
        }
    }
    // TP serving (TP item A): sync the model + config to ONE --node, bring up the RDMA link, and
    // run this same server with its BatchScheduler in SPMD lockstep with the node's mirror.
    // `--tp [N]` is the single authority for the rank count (bare --tp = 2; absent = no TP run).
    let tp_world: Option<u32> = parse_tp_world(args);
    let tp = tp_world.is_some();
    // S9F (TP-DF2 leg): set [df2-tp] BEFORE the model load — the shard-at-load path's Q4
    // assembly reads it in the worker (the round's full-lm_head capture); attach_tp's
    // tp_shard_weights reads it later for the non-shard-at-load path. The node installs the
    // same env from the shipped config before ITS load. WI1: the DSPARK source needs the SAME
    // arming — its round borrows the identical full-vocab head/embed (the round's head GEMM
    // reads all v rows); without it both ranks borrow the E7 vocab shard (the node's
    // two-stage head quantize then hits the full-m vs shard-len shape assert — TP=2 boot
    // crash 2026-09-06; before the two-stage landing the same miss was the acceptance-0.0%
    // class). DSpark is NOT in is_df2_src (that's the DFlash2 family), so it is added here.
    {
        let src = resolve_spec_source(args);
        if tp && (gb10_inference::batch::is_df2_src(src)
                  || src == gb10_inference::batch::SpecSource::Dspark) {
            gb10_inference::opts::set(gb10_inference::opt!("df2-tp"), "1");
        }
    }

    // DSV4 (DeepSeek-V4) serving rides the SAME interface as every other model:
    //   --server --model-dir <bundle> --tp --nodes <peer[:29500]> --port <N> [--max-seq-len <N>]
    // The bundle is TP=2-sharded by construction — there is no single-node serve of it, so --tp is
    // required. The node stays zero-config (resident supervisor + shipped TpConfig, as for qwen).
    if is_dsv4_bundle(std::path::Path::new(&model_path)) {
        if !tp {
            gb10_inference::boot_report::error("[dsv4] this bundle is TP=2-sharded: serving requires --tp --nodes <peer:29500>");
            std::process::exit(1);
        }
        run_dsv4_server_tp(args, &model_path, port);
        return;
    }
    // Model name for /v1/models. Public model id (v0.7.3, issue #8.3): --model-name >
    // --served-model-name (alias) > the model card's `model_name:` > its `base_model:` (e.g.
    // Qwen/Qwen3.8-27B) > dir name — see server::resolve_model_name.
    let model_name = gb10_inference::server::resolve_model_name(
        parse_arg(args, "--model-name"), parse_arg(args, "--served-model-name"), &model_path);
    let default_max_tokens = parse_arg(args, "--max-tokens").and_then(|s| s.parse::<usize>().ok()).unwrap_or(8192);
    // Model-card presence-penalty default varies by model size (2B: 2.0, 4B+: 1.5). Temperature,
    // top_p and top_k defaults are resolved per request in server.rs (AppState::sampling_defaults
    // on the chat path, default_temperature()/.. on the legacy completions path).
    // LOOPFIX (2026-11-06): by the model's IDENTITY (config.json), not its folder name — a "2b"
    // substring also matched "122b" and missed the 0.8B, and the velogb10 runner's folder is
    // "current_model". Qwen3.5 0.8B / 2B cards: 2.0; every other model 1.5, as before.
    let default_presence_penalty = gb10_inference::server::qwen35_card(&model_path).map(|(_, p)| p).unwrap_or(1.5);

    let default_rep_penalty = parse_arg(args, "--default-repetition-penalty").and_then(|s| s.parse::<f32>().ok()).unwrap_or(1.0);
    let default_presence_penalty = parse_arg(args, "--default-presence-penalty").and_then(|s| s.parse::<f32>().ok()).unwrap_or(default_presence_penalty);
    let default_frequency_penalty = parse_arg(args, "--default-frequency-penalty").and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0);

    // OTel generation-telemetry emitter: ONE arg enables it (--otel-endpoint); ABSENT = OFF =
    // zero cost (AppState.otel is None and every telemetry hook in the SSE path compiles to a
    // single predictable branch). See gb10_inference::otel — the design IS the interference
    // gate: lock-free ring push (no alloc/lock/syscall/wake), drop-on-full (the client can
    // never back-pressure decode), ONE timer-polled sender task off the compute stream.
    // v0.7.3 ("tel"): the SAME shared parser as the EXL3 server (otel::config_from_args) —
    // one flag vocabulary, one behaviour, every engine.
    let otel_cfg = gb10_inference::otel::config_from_args(|flag| {
        parse_arg(args, &format!("--{flag}")).map(str::to_string)
    });
    // Refuse an UNUSABLE endpoint before the multi-minute model load. At RUNTIME the emitter is
    // best-effort (ring-full drops rows, failed POSTs drop the batch); a misconfigured one must
    // not boot pretending to stream.
    if let Some(cfg) = &otel_cfg {
        if let Err(e) = cfg.hostport() {
            eprintln!("[otel] {e}");
            std::process::exit(1);
        }
    }

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
        // Parallel-load path only: the completed QP handshake + the retained control streams,
        // both established BEFORE the weight load.
        let mut pre_tp: Option<(gb10_inference::tp::TpContext, Vec<std::net::TcpStream>)> = None;
        if tp && is_dir {
            let pre = gb10_inference::qwen::Config::from_config_json(
                &format!("{}/config.json", model_path.trim_end_matches('/')))
                .expect("pre-read config.json");
            let max_seq_pre = if max_seq_len > pre.max_position_embeddings {
                pre.max_position_embeddings
            } else { max_seq_len };
            let st_tok_pre = gb10_inference::boot_report::Stage::begin("tokenizer");
            if gb10_inference::boot_report::debug() {
                println!("Loading tokenizer from {}...", tokenizer_path);
            }
            // TP2BIT5 #26: a missing/corrupt tokenizer is a REFUSAL with a fix, not a panic (the
            // bogus-pack boot test hit exactly this .expect on its way out).
            let tok_pre = QwenTokenizer::from_file(&tokenizer_path).unwrap_or_else(|e| {
                eprintln!("FATAL: cannot load the tokenizer at {tokenizer_path} ({e:#}); the pack is incomplete — \
                           re-download it or point --model-dir at a complete snapshot");
                std::process::exit(2);
            });
            st_tok_pre.ok();
            // Same parses as their post-load counterparts below (idempotent; needed here for tpc).
            let mtp_depth_pre = parse_arg(args, "--mtp-depth").and_then(|s| s.parse::<usize>().ok());
            let mtp_force_pre = match gb10_inference::opt_or_arg(args, "--mtp").as_deref().unwrap_or("auto") {
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
            let mut tpc = gb10_inference::tp::TpConfig::from_opts();
            tpc.world = tp_world.unwrap_or(2);   // --tp [N] is the single authority (bare = 2)
            tpc.mode_serve = true;
            tpc.max_seq_len = max_seq_pre;
            tpc.max_batch = max_batch;
            tpc.prefix_cache = matches!(parse_arg(args, "--prefix-cache").unwrap_or("off"),
                                        "on" | "true" | "1" | "yes");
            tpc.ngram_draft = parse_arg(args, "--ngram-draft").and_then(|s| s.parse().ok()).unwrap_or(0);
            tpc.tree_draft = matches!(parse_arg(args, "--tree-draft").unwrap_or("off"), "on"|"true"|"1"|"yes");
            tpc.mtp_lanes = matches!(gb10_inference::opt_or_arg(args, "--mtp-lanes").as_deref().unwrap_or("off"), "on"|"true"|"1"|"yes");
            tpc.mtp_force = mtp_force_pre;
            tpc.mtp_depth_pin = mtp_depth_pre;
            tpc.no_decode_graphs = gb10_inference::opts::var(gb10_inference::opt!("no-decode-graphs")).is_ok();
                    tpc.fp8_prefill = gb10_inference::opts::var(gb10_inference::opt!("fp8-prefill")).is_ok_and(|v| v != "0");
        tpc.fp8_prefill = gb10_inference::opts::var(gb10_inference::opt!("fp8-prefill")).is_ok_and(|v| v != "0");
            tpc.cpu_sample = gb10_inference::opts::var(gb10_inference::opt!("cpu-sample")).is_ok();
            tpc.eos = tok_pre.stop_token_ids(pre.eos_token_id);
            tpc.calib_prompt = tok_pre.encode("The capital of France is", true)
                .expect("probe encode");
            tpc.batch_probe = Some(max_batch);
            // S9F (TP-DF2 leg): ship the resolved speculation source + draft dir so the node
            // builds the IDENTICAL MtpPolicy and loads the SAME drafter artifact (bit-identical
            // round state is the SPMD contract).
            tpc.spec_source = resolve_spec_source(args).cli_name().to_string();
            // MANDATORY user-supplied path (owner rule: no default, no fallback constant; a bad
            // path stops the app). The head's resolved dir ships on the config AND the artifact
            // bytes ride the sync (cluster.rs DraftManifest) into the node's blob cache.
            tpc.df2_draft_dir = resolve_draft_dir(args, Some(resolve_spec_source(args)))
                .unwrap_or_default();
            // S9F+ (2026-08-29): ship the --sha256 artifact-pin override (None = published
            // REAL_SHA256) so the node loads the same artifact under the same pin — a one-sided
            // pin would be a round-load mismatch between ranks.
            tpc.df2_sha_pin = parse_arg(args, "--sha256").map(str::to_string);
            // P2: the round-sharding toggle (CLI flag per AGENTS §7; rides TpConfig — no env
            // side channel). DEFAULT OFF until the Phase D quad truth flips it.
            tpc.df2_round_shard = matches!(parse_arg(args, "--df2-round-shard").unwrap_or("on"),
                                           "on" | "true" | "1" | "yes");
            // DF2 block-16: ship the resolved block so every rank sizes the identical round and
            // verify (SPMD — a one-sided block is a collective mismatch).
            tpc.df2_block = gb10_inference::dflash2::block();
            // DFLASH2-GEN: ship the resolved mask-row override (SPMD, same resolution as block).
            tpc.df2_mask_override = gb10_inference::dflash2::mask_override();
            // DF2_CARRY: ship the prefix-cache carry toggle (SPMD-critical — a one-sided carry
            // would desync the verify all-reduce; every rank must make the same admit decision).
            tpc.df2_carry = matches!(parse_arg(args, "--df2-carry").unwrap_or("off"),
                                     "on" | "true" | "1" | "yes");
            // P3(b) L1: prose-lane routing (SPMD-critical — the node runs the identical decode_step).
            tpc.df2_prose_lane_greedy = matches!(parse_arg(args, "--df2-prose-lane").unwrap_or("greedy-drafts"),
                                                 "greedy-drafts" | "greedy" | "argmax");
            // PLAN/25 Phase 0: coverage-trace flag rides the config here TOO (this is the live
            // session-sync arm — the flag must be in the tpc the head actually ships).
            tpc.df2_step_dump = cov_dump.is_some();
            // Installing before the load is behavior-neutral: the only loader consumer
            // (gpu.rs shard-at-load shard_mixers) ORs tp_config with the very env var
            // TpConfig::from_opts mirrors, so it reads the same value either way.
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
            let (_nodes, streams) = session.join()
                .expect("TP head session thread panicked")
                .expect("TP head session sync");
            let mut ctx = gb10_inference::tp::TpContext::bring_up_head(tp_world.unwrap_or(2) as i32).expect("TP bring-up");
            ctx.sanity().expect("TP sanity");
            pre_tp = Some((ctx, streams));
            pre_cfg_check = Some((pre.max_position_embeddings, pre.eos_token_id));
            pre_tokenizer = Some(tok_pre);
        }

        if is_dir {
            // OUT1 (§1b-bis 8): a single-file EXL3 pack (no model.safetensors.index.json) is not
            // detected by is_exl3_pack and used to fall into the NVFP4 loader, dying mid-load with
            // a raw `missing tensor: lm_head.weight` panic. Refuse at the door, with the way out.
            if let Ok(qc) = std::fs::read_to_string(format!("{}/quantization_config.json", model_path.trim_end_matches('/'))) {
                if qc.contains("exl3") && !gb10_inference::exl3_serve::is_exl3_pack(&model_path) {
                    // EXL27 (§1b-bis 8, owner scope add 2026-10-16): single-file packs are served, not
                    // refused. Best effort: synthesize the one-shard index from the safetensors header;
                    // the refusal below stays only for packs we truly cannot read (no safetensors at all).
                    match gb10_inference::exl3_serve::ensure_singlefile_index(&model_path) {
                        Ok(true) => gb10_inference::rprintln!("[boot] single-file EXL3 pack: synthesized model.safetensors.index.json (1 shard) — serving via the standard indexed path"),
                        Ok(false) => {},
                        Err(e) => gb10_inference::reprintln!("[boot] single-file EXL3 index synthesis failed: {e}"),
                    }
                    if !gb10_inference::exl3_serve::is_exl3_pack(&model_path) {
                        gb10_inference::boot_report::error("[exl3] this EXL3 bundle has no readable tensors (no index, no model.safetensors): re-download the pack. Refusing.");
                        std::process::exit(1);
                    }
                }
            }
            // OUT1: the config stage covers the config read + the mem-budget forecast (debug).
            let st_config = gb10_inference::boot_report::Stage::begin("config");
            // Config pre-read for the mem budget. A corrupt config.json is a load error, not a
            // panic: surface the same graceful message and exit cleanly (owner directive 2026-08-27).
            let cfg_pre = match gb10_inference::qwen::Config::from_config_json(
                &format!("{}/config.json", model_path.trim_end_matches('/')))
            {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Error loading model from {}: config.json does not parse: {e:#}",
                              model_path);
                    eprintln!("  The model checkpoint appears to be corrupted.");
                    eprintln!("  Fix: re-download the FULL set cleanly (all shards + the index + config), \
                               verify with");
                    eprintln!("  sha256sum that the shards match the source, and confirm the directory \
                               is the engine's");
                    eprintln!("  CONVERTED NVFP4 model (not a raw or differently-quantized checkpoint). \
                               See the README.");
                    std::process::exit(1);
                }
            };
            let km = if matches!(gb10_inference::opts::var(gb10_inference::opt!("kv-tq")).ok().as_deref(), Some("1") | Some("3")) { gb10_inference::gpu::KVCacheMode::Tq }
                     else if gb10_inference::opts::var(gb10_inference::opt!("kv-k8v8")).ok().as_deref() == Some("1") { gb10_inference::gpu::KVCacheMode::K8v8 }
                     else if gb10_inference::opts::var(gb10_inference::opt!("kv-k8v4")).ok().as_deref() == Some("1") { gb10_inference::gpu::KVCacheMode::K8v4 }
                     else if gb10_inference::opts::var(gb10_inference::opt!("kv-quant")).is_ok() { gb10_inference::gpu::KVCacheMode::Q4 }
                     else { gb10_inference::gpu::KVCacheMode::Bf16 };
            mem_budget_report(&model_path, &cfg_pre, tp, max_seq_len, max_batch, km);
            st_config.ok();
        }
        let st_weights = gb10_inference::boot_report::Stage::begin("weights");
        let (mut gpu, cfg) = if is_dir {
            if gb10_inference::boot_report::debug() {
                println!("Loading model from {} (streaming bf16)...", model_path);
            }
            if tp {
                // TP=2 head = rank 0. hy_v3 shards host-side in the loader (full model > one node);
                // qwen loads whole and shards at attach_tp, unchanged.
                load_model_gpu(&model_path, Some(0), tp_world.unwrap_or(2) as i32)
            } else {
                load_model_gpu(&model_path, None, 1)
            }
        } else {
            if gb10_inference::boot_report::debug() {
                println!("Loading model from {} ...", model_path);
            }
            let host = gb10_inference::qwen::Model::load(&model_path).expect("load model");
            let gpu = gb10_inference::gpu::GpuModel::new(&host).expect("gpu init");
            (gpu, host.config)
        };
        st_weights.ok();
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
        // ~64 KB/token — 256K/batch-2 ≈ 34 GB, fine on 128 GB, just slow to prefill); going beyond is not.
        let model_max = cfg.max_position_embeddings;
        // E5/YARN1: the trunk YaRN factor (--rope-yarn-factor > shipped TpConfig > the pack's
        // rope_scaling; one rule: qwen::pick_yarn) extends the rope ceiling to native x factor,
        // and the launch-argument bit-packings cap it in turn — the same two-ceiling clamp the
        // EXL3 serve uses (the packings are 20 bits on this path too). Over-length boots on
        // the nearest workable value with one line, never a silent refusal.
        let (yarn, _yarn_src) = gb10_inference::qwen::resolved_yarn(&cfg);
        let (eff_max, why) = gb10_inference::exl3_serve::clamp_ctx(
            max_seq_len, Some(cfg.yarn_native_window), yarn as f64);
        if let Some(why) = why {
            gb10_inference::boot_report::notice(&gb10_inference::exl3_serve::clamp_ctx_message(
                max_seq_len, eff_max, why, Some(cfg.yarn_native_window), yarn as f64));
        }
        let max_seq_len = eff_max;
        if gb10_inference::boot_report::debug() {
        println!("Context: --max-seq-len {} (model max {}). KV cache ~{:.1} GB at batch {}.",
                 max_seq_len, model_max,
                 {
                     let nfull = cfg.layer_types.iter().filter(|t| matches!(t, gb10_inference::qwen::LayerType::FullAttention)).count();
                     (nfull * cfg.num_kv_heads * cfg.head_dim * 2 * 2 * max_seq_len * max_batch) as f64 / 1e9
                 }, max_batch);
        }

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
        let mtp_force = match gb10_inference::opt_or_arg(args, "--mtp").as_deref().unwrap_or("auto") {
            "on"  | "1" | "true"  => Some(true),
            "off" | "0" | "false" => Some(false),
            "auto" => None,
            other => { eprintln!("--mtp must be auto|on|off (got {:?})", other); std::process::exit(1); }
        };
        // S8F (S6F adjudication): `--spec-source {mtp,dflash2,dflash2-rq,dflash2-auto,none}` —
        // the selectable speculation source. Default is now `dflash2-auto` (DFlash2 with the
        // per-request lane split: greedy on code, real-q on math/chat/prose). DFlash2 serves via
        // the S4F round when the artifact is resident (absent/failed → MTP fallback, the standing
        // directive); `--spec-source mtp` is unchanged and permanently selectable; none = plain
        // decode. Rides TpConfig (SPMD) under TP.
        let spec_source = resolve_spec_source(args);
        // ([df2-tp] is set EARLY — right after the `tp` determination, BEFORE the model
        // load — because the shard-at-load Q4 assembly reads it in the worker.)
        // (CLI-1: the obsolete RUST_INFER_MTP / RUST_INFER_MTP_STOCHASTIC / RUST_INFER_GPU_SAMPLE vars
        // are refused at startup by opts::reject_env, naming --mtp on|off.)

        let tokenizer = match pre_tokenizer {
            // The TP parallel-load path already loaded it (and printed) before the weight load.
            Some(t) => t,
            None => {
                let st_tok = gb10_inference::boot_report::Stage::begin("tokenizer");
                if gb10_inference::boot_report::debug() {
                    println!("Loading tokenizer from {}...", tokenizer_path);
                }
                let tok = QwenTokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer");
                st_tok.ok();
                tok
            }
        };

        // STOP ON EVERY TURN TERMINATOR, not just the one config.json advertises. Qwen3.5 declares
        // eos_token_id = <|endoftext|> (248044), but a CHAT turn ends with <|im_end|> (248046) — which
        // is what the model actually emits. Stopping only on the advertised id let the assistant run
        // past the end of its own turn and hallucinate the next one: a fabricated `user` message, a new
        // `<think>` block, sometimes a second conflicting tool call. See QwenTokenizer::stop_token_ids.
        // COPY27B accounting: the think-tag ids for the per-request reasoning/answer split ([copy-acct])
        if let (Ok(o), Ok(c)) = (tokenizer.encode("<think>", false), tokenizer.encode("</think>", false)) {
            if o.len() == 1 && c.len() == 1 { gb10_inference::batch::set_think_ids(o[0], c[0]); }
        }
        let eos = tokenizer.stop_token_ids(config_eos);
        // A3: AppState needs its own copy of the stop set (the original `eos` is moved into the
        // TP config below).
        let stop_ids_for_state = eos.clone();
        if gb10_inference::boot_report::debug() {
            println!("Stop tokens: {:?}  (config.json advertises {})", eos, config_eos);
        }

        // Serving-option values, parsed once here so the TP config (below) and BatchScheduler::new
        // (further down) agree. The explanatory prints stay at their original spots.
        let prefix_cache = matches!(parse_arg(args, "--prefix-cache").unwrap_or("off"),
                                    "on" | "true" | "1" | "yes");
        let ngram_draft: usize = parse_arg(args, "--ngram-draft").and_then(|s| s.parse().ok()).unwrap_or(0);
        let tree_draft = matches!(parse_arg(args, "--tree-draft").unwrap_or("off"), "on"|"true"|"1"|"yes");
        let mtp_lanes = matches!(gb10_inference::opt_or_arg(args, "--mtp-lanes").as_deref().unwrap_or("off"), "on"|"true"|"1"|"yes");
        // P11 W5a: --prefill-sched <inline|cursor> (default cursor = the scheduler default-ON).
        let pf_sched_inline = gb10_inference::opt_or_arg(args, "--prefill-sched").map(|s| s == "inline").unwrap_or(false);

        // TP=2 serving bring-up (TP item A). Order matters and mirrors the node's `node_serve_tp`:
        // ship the model + config, bring up the RDMA link, attach TP — from here on EVERY forward
        // (calibration, graph capture, decode) runs SPMD in lockstep with the node. The retained
        // sync stream becomes the serving control plane (CalibTable / Ready / Step / Shutdown).
        let mut tp_streams: Option<Vec<std::net::TcpStream>> = None;
        if tp {
            // The session + RDMA handshake either already happened BEFORE the weight load (the
            // --model-dir parallel-load path above — the node has been loading concurrently with
            // us since then) or — the legacy --model <file> path — happen here, serialized as before.
            let (streams, ctx) = match pre_tp {
                Some((ctx, streams)) => (streams, ctx),
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
                    let mut tpc = gb10_inference::tp::TpConfig::from_opts();
                    tpc.world = tp_world.unwrap_or(2);   // --tp [N] is the single authority (bare = 2)
                    tpc.mode_serve = true;
                    tpc.max_seq_len = max_seq_len;
                    tpc.max_batch = max_batch;
                    tpc.prefix_cache = prefix_cache;
                    // §4.1 MTP-block sharding: --tp-shard-mtp on|off is a registry option (default on),
                    // already resolved by from_opts (CLI-1: `--tp-shard-mtp off` must not be ORed back on).
                    tpc.ngram_draft = ngram_draft;
                    tpc.tree_draft = tree_draft;
                    tpc.mtp_lanes = mtp_lanes;
                    tpc.pf_sched_inline = pf_sched_inline;
                    tpc.mtp_force = mtp_force;
                    tpc.mtp_depth_pin = mtp_depth;
                    tpc.no_decode_graphs = gb10_inference::opts::var(gb10_inference::opt!("no-decode-graphs")).is_ok();
                    tpc.fp8_prefill = gb10_inference::opts::var(gb10_inference::opt!("fp8-prefill")).is_ok_and(|v| v != "0");
                    tpc.cpu_sample = gb10_inference::opts::var(gb10_inference::opt!("cpu-sample")).is_ok();
                    tpc.no_verify_graph = gb10_inference::opts::var(gb10_inference::opt!("no-verify-graph")).is_ok();
                    // --device-loop (device-resident token loop): OFF by default until gated.
                    tpc.device_loop = matches!(gb10_inference::opt_or_arg(args, "--device-loop").as_deref().unwrap_or("off"),
                                                "on" | "true" | "1" | "yes");
                    tpc.eos = eos.clone();
                    tpc.calib_prompt = tokenizer.encode("The capital of France is", true)
                        .expect("probe encode");
                    tpc.batch_probe = Some(max_batch);
                    // S9F (TP-DF2 leg): same as the pre-TP path — ship the resolved source +
                    // draft dir so the node's policy and round load match the head's.
                    tpc.spec_source = resolve_spec_source(args).cli_name().to_string();
                    // MANDATORY user-supplied path (same rule as the pre-TP fill above). The
                    // WI1 DSpark source resolves through its own resolver (mandatory when
                    // --spec-source dspark); both ride df2_draft_dir (the shipped-artifact slot).
                    // ONE flag for every source (dflash2 | dspark | dflash): the resolved dir is
                    // the generic shipped-artifact slot the node loads from its blob cache.
                    tpc.df2_draft_dir = resolve_draft_dir(args, Some(resolve_spec_source(args)))
                        .unwrap_or_default();
                    // S9F+ (2026-08-29): ship the --sha256 artifact-pin override (same as the
                    // pre-TP fill — the node must load the same artifact under the same pin).
                    tpc.df2_sha_pin = parse_arg(args, "--sha256").map(str::to_string);
                    // P2: the round-sharding toggle (same resolution as the pre-TP fill above).
                    tpc.df2_round_shard = matches!(parse_arg(args, "--df2-round-shard").unwrap_or("on"),
                                                   "on" | "true" | "1" | "yes");
                    // DF2 block-16: same as the pre-TP fill above (SPMD).
                    tpc.df2_block = gb10_inference::dflash2::block();
                    // DFLASH2-GEN: same as the pre-TP fill above (SPMD).
                    tpc.df2_mask_override = gb10_inference::dflash2::mask_override();
                    // DF2_CARRY: same as the pre-TP fill above (SPMD).
                    tpc.df2_carry = matches!(parse_arg(args, "--df2-carry").unwrap_or("off"),
                                             "on" | "true" | "1" | "yes");
                    // P3(b) L1: prose-lane routing (SPMD-critical — the node runs the identical decode_step).
                    tpc.df2_prose_lane_greedy = matches!(parse_arg(args, "--df2-prose-lane").unwrap_or("greedy-drafts"),
                                                         "greedy-drafts" | "greedy" | "argmax");
                    // PLAN/25 Phase 0: the coverage-trace flag rides the config (SPMD — the node
                    // mirror must run the identical trace op sequence; only the head keeps the
                    // dump handle and writes records).
                    tpc.df2_step_dump = cov_dump.is_some();
                    gb10_inference::tp::set_tp_config(tpc.clone());
                    let (_nodes, streams) = gb10_inference::cluster::run_head_session(
                        std::path::Path::new(&model_path), explicit, wait, &tpc)
                        .expect("TP head session sync");
                    let mut ctx = gb10_inference::tp::TpContext::bring_up_head(tpc.world as i32).expect("TP bring-up");
                    ctx.sanity().expect("TP sanity");
                    (streams, ctx)
                }
            };
            println!("HEAD (rank {}/{}) — TP LINK UP (serving mode)", ctx.rank, ctx.world);
            let (rank, world, link) = ctx.into_parts();
            gpu.attach_tp(rank, world, link);
            tp_streams = Some(streams);
        }

        let mtp_r = if gpu.mtp_present() && mtp_force != Some(false) {
            // The per-depth cost ratios are a stable function of (kernels, GPU, model) — independent of
            // the conversation — so cache them under <binary_dir>/mtp_calib/<model>.json and skip the
            // recalibration on subsequent launches. Keyed by model path + the binary's mtime, so a
            // rebuild (new kernels) transparently invalidates the cache and recalibrates. E17: the
            // table is per CONTEXT BUCKET (fmt 2); the verify's KV bytes grow ∝ context.
            let calib_path = mtp_calib_cache_path(&model_path);
            let calib_stamp = mtp_calib_stamp(&model_path);
            // TP serving: bypass the cache READ. A hit skips the calibration forwards entirely, but
            // the node cannot know that (its model path/cache can never match the head's) and MUST
            // run the same SPMD forward sequence the head runs — an unshared skip deadlocks the
            // all-reduce barriers. Both ranks therefore always calibrate in lockstep; the cache
            // WRITE below still lands, so single-node launches keep the fast path.
            let st_cal = gb10_inference::boot_report::Stage::begin("calibrate");
            let cached = if tp { None } else { read_mtp_calib(&calib_path, &calib_stamp) };
            if let Some(bs) = cached {
                if gb10_inference::boot_report::debug() {
                    println!("MTP cost/depth: loaded from cache ({} ctx buckets) -> {}",
                             bs.len(), calib_path.as_ref().map(|p| p.display().to_string()).unwrap_or_default());
                }
                for (ctx, t) in &bs {
                    let r2 = t.iter().find(|&&(d, _)| d == 2).map(|&(_, r)| r).unwrap_or(0.0);
                    if gb10_inference::boot_report::debug() {
                        println!("    ctx {:>6}: depth 2 costs {:.2}x a decode  (cached)", ctx, r2);
                    }
                }
                bs
            } else {
            // E17: calibrate per context bucket. The harness runs NO prefill — every phase cost is a
            // pure function of pc (loop bounds derive from positions, never values) — so a bucket
            // costs the phases, not a long prompt. The state needs ONE live KV slot (the verify
            // writes only slot 0) at the top bucket's stride; the GDN checkpoint slots (2..=N) live
            // in the separately-sized state half, so the long stride does not multiply slots.
            if gb10_inference::boot_report::debug() {
                println!("Calibrating MTP cost per depth, per context bucket...");
            }
            let mut cpool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
            let calib_points = gb10_inference::gpu::mtp_calib_ctx_points(max_seq_len);
            let calib_seq = *calib_points.last().unwrap();
            let mut cstate = gpu.new_batch_state(1, 2 + gb10_inference::gpu::PROFILE_MAX_N, calib_seq);
            let probe = tokenizer.encode("The capital of France is", true).expect("probe encode");
            // TP boot barrier (the node's twin is before its SPMD calibration): ranks finish their own setup at
            // different times; the calibration's all-reduces must not start under the 10 s per-step watchdog.
            if tp { gb10_inference::net::boot_barrier(0xCA1B).expect("TP boot barrier before the MTP calibration"); }
            let bs = gpu.calibrate_mtp_r(&mut cpool, &mut cstate, &probe, calib_seq);
            for (ctx, t) in &bs {
                for &(d, r) in t {
                    if gb10_inference::boot_report::debug() {
                        println!("    ctx {:>6} depth {}: a step costs {:.2}x a decode  (pays if it emits > {:.2} tok)", ctx, d, r, r);
                    }
                }
            }
            write_mtp_calib(&calib_path, &calib_stamp, &bs);
            st_cal.ok();
            bs
            }
        } else {
            vec![]
        };
        // S8F: the DFlash2 round (the selectable/default speculation source). Loaded ONLY when
        // requested; an absent/failed artifact degrades to the MTP fallback (never a hard failure).
        let df2 = if gb10_inference::batch::is_df2_src(spec_source) {
            // Mandatory only for an EXPLICIT DF2 --spec-source; the resolved default falls back
            // to MTP when no --draft-dir was supplied (resolve_draft_dir -> None).
            let args = std::env::args().collect::<Vec<_>>();
            let sha_pin = parse_arg(&args, "--sha256");
            match resolve_draft_dir(&args, Some(spec_source)) {
                Some(d) => load_df2_round_dir(&mut gpu, max_seq_len, &d, sha_pin),
                None => None,
            }
        } else { None };
        if df2.is_some() {
            if gb10_inference::boot_report::debug() {
                println!("[df2] DFlash2 round RESIDENT (spec-source={}) — serving via the S4F \
                      integrated round (b==1 lanes); MTP remains the fallback (standing directive)",
                     spec_source.cli_name());
            }
        }
        // WI1: the DSpark drafter (--spec-source dspark). Same fail-soft load contract; the
        // residency line is the startup-log proof the acceptance gate greps for.
        let dspark = if matches!(spec_source, gb10_inference::batch::SpecSource::Dspark) {
            let args = std::env::args().collect::<Vec<_>>();
            match resolve_draft_dir(&args, Some(spec_source)) {
                Some(d) => {
                    let sha_pin = parse_arg(&args, "--sha256");
                    load_dspark_round_dir(&mut gpu, max_seq_len, &d, sha_pin)
                }
                None => None,
            }
        } else { None };
        if dspark.is_some() {
            println!("[dspark] DSpark round RESIDENT (spec-source=dspark) — serving via the WI1 \
                      Qwen3.8-27B-DSpark round (b==1 lanes); MTP remains the fallback (standing \
                      directive)");
        }
        // TP serving: ship the head's cost tables to EVERY node. Each node has already run the
        // identical SPMD calibration forwards (and discarded its own tables); every rank's MtpPolicy
        // must be built from ONE identical set of numbers or the live depth decisions would diverge.
        // S9F (TP-DF2 leg): the CalibTable also carries the round's LOAD OUTCOME (df2_round) — the
        // node loads the round iff the head did (a one-sided round is a lane-branch mismatch).
        if let Some(streams) = tp_streams.as_mut() {
            let ctx_r: Vec<(u32, Vec<(u32, f32)>)> = mtp_r.iter()
                .map(|&(c, ref t)| (c as u32, t.iter().map(|&(d, r)| (d as u32, r)).collect()))
                .collect();
            // WI1: df2_round is the GENERIC round-residency bit — a resident DSpark round
            // means the node must hold one too (a one-sided round is a lane-branch mismatch).
            let calib = gb10_inference::tp_serve::ServingMsg::CalibTable {
                ctx_r,
                df2_round: df2.is_some() || dspark.is_some(),
            };
            for s in streams.iter_mut() {
                gb10_inference::tp_serve::send_serving(s, &calib)
                    .expect("ship MTP calib table to node");
            }
            if gb10_inference::boot_report::debug() {
                println!("TP — MTP calib table + df2_round={} shipped to {} node(s)",
                         df2.is_some(), streams.len());
            }
        }
        let policy = gb10_inference::batch::MtpPolicy::with_source(
            gpu.mtp_present(), mtp_force, mtp_depth, mtp_r, spec_source);
        // Captured before `policy` moves into the scheduler (PR #4: the HTTP clamp mirror).
        let mtp_active_at_start = policy.active();
        if !gpu.mtp_present() {
            if gb10_inference::boot_report::debug() {
                println!("MTP: model has no MTP head — plain decode.");
            }
        } else {
            if gb10_inference::boot_report::debug() {
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
        }

        // Prefix caching is OPT-IN. It skips re-prefilling a conversation's history — on a 5-turn tool
        // conversation that is 97% of the prefill, and follow-up turns get ~3x faster. The price is that
        // reusing a prefix re-chunks the prefill, and prefill runs on cuBLAS, which picks a different
        // kernel per shape: a cached turn is NOT bit-identical to a cold one, so the same conversation
        // can word an answer slightly differently depending on cache state. That is a trade the operator
        // makes, not one we make for them. (Value parsed above, next to the TP config build.)
        if prefix_cache {
            if gb10_inference::boot_report::debug() {
                println!("Prefix cache: ON — a conversation's history is reused instead of re-prefilled \
                      (~3x faster follow-up turns). Cached turns are NOT bit-identical to cold ones: \
                      reuse re-chunks the prefill and cuBLAS picks a kernel per shape.");
            }
        } else {
            if gb10_inference::boot_report::debug() {
                println!("Prefix cache: off — every request prefills its whole prompt (bit-exact, and \
                      slow on multi-turn agents: ~88% of prefill is recomputed). Enable: --prefix-cache on");
            }
        }

        // Prompt-lookup n-gram drafting: EXPERIMENTAL, default OFF. Lossless (the verify checks every
        // draft, so output is byte-identical either way -- confirmed), but as a naive REPLACEMENT of the
        // MTP draft it is a net LOSS on the serving path: with real auto-depth the MTP acceptance
        // baseline is high (~74% on tool text), so a spurious short n-gram match replaces a GOOD draft
        // more often than it rescues a bad one (74% -> 66%, ~3 tok/s slower, measured). The right design
        // proposes the n-gram token as an ADDITIONAL candidate (tree verify, backlog #7), not a
        // replacement. Kept behind a flag for that work. `--ngram-draft 3` to experiment.
        if ngram_draft > 0 {
            gb10_inference::boot_report::notice(&format!("Prompt-lookup drafting: ON (order {ngram_draft}) — EXPERIMENTAL, measured net-negative vs MTP"));
        }

        if tree_draft { gb10_inference::boot_report::notice("Tree drafting: ON (k=2 fork-then-chain) — EXPERIMENTAL, lossless, gated on yield"); }
        let (stx, srx) = tokio::sync::mpsc::unbounded_channel::<gb10_inference::batch::BatchRequest>();
        if mtp_lanes { gb10_inference::boot_report::notice("Batched MTP verify across lanes: ON -- EXPERIMENTAL, lossless (LANES_OK), packs concurrent greedy lanes into one verify"); }
        let (df2_round, df2_sink, df2_prime) = match df2 {
            Some((r, s, p)) => (Some(r), Some(s), Some(p)),
            None => (None, None, None),
        };
        let (dspark_round, dspark_sink, dspark_prime) = match dspark {
            Some((r, s2, p2)) => (Some(r), Some(s2), Some(p2)),
            None => (None, None, None),
        };
        let st_graphs = gb10_inference::boot_report::Stage::begin("graphs");
        let mut scheduler = gb10_inference::batch::BatchScheduler::with_df2(
            gpu, max_batch, max_seq_len, eos, srx, policy, prefix_cache, ngram_draft, tree_draft, mtp_lanes,
            df2_round, df2_sink, df2_prime, dspark_round, dspark_sink, dspark_prime, cov_dump);
        st_graphs.ok();
        // P14: the DFlash v1 lane (--spec-source dflash). The artifact is REQUIRED and the install is
        // loud: a missing dir or a failed load refuses to start rather than silently serving MTP.
        if matches!(spec_source, gb10_inference::batch::SpecSource::DFlash) {
            // Same single resolver every other source uses; for an explicit `dflash` source it is
            // MANDATORY (it exits 2 with the named remedy rather than serving MTP silently).
            let d = resolve_draft_dir(&args, Some(spec_source))
                .expect("resolve_draft_dir is mandatory for --spec-source dflash");
            if let Err(e) = scheduler.install_dflash_lane(&d) {
                eprintln!("FATAL: DFlash v1 lane install failed for {d}: {e:#}");
                std::process::exit(2);
            }
            if gb10_inference::boot_report::debug() {
                println!("[dflash] v1 lane RESIDENT (spec-source=dflash) — greedy b==1 lanes; unprimed \
                      requests (prefix-cache reuse) are served by MTP and logged");
            }
        }
        // P11 W5a: the prefill-schedule escape (default = cursor, the owner-blessed default-ON).
        scheduler.set_pf_inline(pf_sched_inline);
        // P3(b) L1: prose-lane routing (default rq = sampled real-q selector; greedy-drafts =
        // argmax drafts + the existing sampled-verify path). Affects the DFlash2Auto General domain.
        scheduler.set_prose_lane_greedy(
            matches!(parse_arg(args, "--df2-prose-lane").unwrap_or("greedy-drafts"),
                     "greedy-drafts" | "greedy" | "argmax"));
        // DF2_CARRY: keep DFlash2 in the draft seat across a prefix-cache hit (default off).
        // NOTE this must be set on the SERVING scheduler — the bench path builds its own.
        scheduler.set_df2_carry(
            matches!(parse_arg(args, "--df2-carry").unwrap_or("off"),
                     "on" | "true" | "1" | "yes"));
        // S-B9-REL-VISIONFIX2 V2d: the vision tower is built HERE, before the scheduler is spawned, not
        // after. `GpuVisualTower::new_with_drill` uploads its weights through the legacy (NULL) stream
        // (332 `htod_sync_copy(..).unwrap()`, a cuBLAS handle, a module load): a call that is refused,
        // and that invalidates another thread's open graph capture, if the scheduler task is capturing
        // at that moment (tests/vision_capture_race_test.rs). The scheduler's own boot captures ran
        // inside `BatchScheduler::with_df2` on this thread, so before this point nothing can be capturing and
        // nothing else touches the device; the spawn below is the first moment it can.
        // Vision tower load, geometry-driven. The whole Qwen3.5/3.8 VL family is supported:
        // dimensions come from `config.json` `vision_config` (TowerDims) and the `nvfp4-*` dirs'
        // packed MLP weights are dequantized at load. Any model that declares `vision_config`
        // but fails the strict load gets a visible notice and serves text-only (image traffic →
        // clean BAD_REQUEST, never a crash). Text-only models (no `vision_config`) load nothing.
        let vision_tower = match gb10_inference::vision_tower::vision_geometry(&model_path) {
            Ok(Some(_)) => match gb10_inference::vision_tower::VisualTower::load(&model_path) {
                Ok(t) => Some(std::sync::Arc::new(t)),
                Err(e) => {
                    gb10_inference::boot_report::warn(&format!("[vision] tower load failed ({e}); serving text-only (image requests are rejected)"));
                    None
                }
            },
            Ok(None) => None,
            Err(e) => {
                gb10_inference::boot_report::warn(&format!("[vision] vision geometry probe failed ({e}); serving text-only"));
                None
            }
        };
        // S-B9-REL-VISIONFIX2 V5: `--vision-cpu` is a registry flag. `parse_arg` is the VALUE reader: as the LAST
        // argument it was silently ignored (the GPU tower served), and `--vision-cpu=0` forced the CPU tower.
        let vision_cpu = gb10_inference::opts::on(gb10_inference::opt!("vision-cpu"));
        // PR #16 drill: fail the FIRST GPU-tower GEMM once (then back to the GPU path).
        // S-B9-REL-PRPORT3 (review R7): registered in opts_table (Cat::Diag -> --help-diag) and
        // consumed through the registry — the hand parse silently dropped the flag when it was
        // the LAST argv token, and `=0` armed it.
        let vision_fail_gemm = gb10_inference::opts::on(gb10_inference::opt!("vision-fail-gemm"));
        gb10_inference::memwatch::phase("srv:post-vision-cpu-flag");
        // GPU vision fast path: build on the shared device (same primary context as the serving model).
        // Soft-fail — a GPU-less / PTX-less box keeps the CPU tower (or text-only behavior).
        let vision_gpu = if vision_cpu { None } else {
            vision_tower.as_ref().and_then(|t| {
                let dev = match cudarc::driver::CudaDevice::new(0) {
                    Ok(d) => d,
                    Err(e) => { gb10_inference::boot_report::warn(&format!("[vision] no CUDA device ({e}); image requests use the CPU path.")); return None; }
                };
                match gb10_inference::vision_gpu::GpuVisualTower::new_with_drill(dev, t, vision_fail_gemm) {
                    Ok(v) => Some(std::sync::Arc::new(std::sync::Mutex::new(v))),
                    Err(e) => { eprintln!("[vision] GPU tower unavailable ({e}); image requests use the CPU path."); None }
                }
            })
        };

        // If the scheduler dies, the server must DIE WITH IT. It used to be a bare tokio::spawn: a panic
        // inside (an OOM, say) killed the task silently, and the HTTP layer went on accepting requests
        // and answering every one of them with ZERO TOKENS, forever. A loud crash is recoverable; a
        // zombie that looks healthy is not.
        match tp_streams {
            None => {
                tokio::spawn(async move {
                    if let Err(e) = tokio::spawn(scheduler.run()).await {
                        eprintln!("\n*** FATAL: the scheduler task died ({e}). The server cannot serve without \
                                   it and will not pretend to. Exiting. ***\n");
                        gb10_inference::logq::flush_and_exit(std::time::Duration::from_millis(300), 70); // H7
                    }
                    eprintln!("\n*** FATAL: the scheduler loop returned unexpectedly. Exiting. ***\n");
                    gb10_inference::logq::flush_and_exit(std::time::Duration::from_millis(300), 70); // H7
                });
            }
            Some(mut streams) => {
                // Bind HTTP only after EVERY node's mirror is armed: a client request admitted before
                // any mirror's first Step recv would desync the lockstep.
                for (i, s) in streams.iter_mut().enumerate() {
                    match gb10_inference::tp_serve::recv_serving(s).expect("node Ready") {
                        gb10_inference::tp_serve::ServingMsg::Ready =>
                            println!("TP -- node rank {} READY (mirror scheduler armed)", i + 1),
                        other => panic!("expected Ready from node rank {}, got {other:?}", i + 1),
                    }
                }
                println!("TP -- all {} node(s) READY; binding HTTP", streams.len());
                // TP wants the launch thread PINNED (an unpinned launch thread presents exactly like
                // a protocol stall -- the 9.0->15.1 tok/s lesson), so the scheduler gets a dedicated
                // pinned thread with a current-thread runtime, not a tokio worker (a task can migrate
                // between workers at every await). Same die-with-it rule as the single-node spawn.
                std::thread::spawn(move || {
                    if !gb10_inference::net::pin_thread(9) {
                        eprintln!("\n*** FATAL: TP head scheduler failed to pin to core 9 -- TP refuses \
                                   to run unpinned. Exiting. ***\n");
                        gb10_inference::logq::flush_and_exit(std::time::Duration::from_millis(300), 70); // H7
                    }
                    let rt = tokio::runtime::Builder::new_current_thread().enable_all()
                        .build().expect("scheduler runtime");
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                        || rt.block_on(scheduler.run_tp_head(streams))));
                    match res {
                        Ok(Ok(())) => eprintln!("\n*** FATAL: the TP scheduler loop returned unexpectedly. Exiting. ***\n"),
                        Ok(Err(e)) => eprintln!("\n*** FATAL: the TP scheduler loop failed: {e:#}. Exiting. ***\n"),
                        Err(_) => eprintln!("\n*** FATAL: the TP scheduler task panicked. Exiting. ***\n"),
                    }
                    gb10_inference::logq::flush_and_exit(std::time::Duration::from_millis(300), 70); // H7
                });
            }
        }
        gb10_inference::memwatch::phase("srv:post-scheduler");

        // OTel emitter: build the sink and spawn THE sender (one timer-polled task; never wakes
        // per token, never runs on a GPU thread, does no GPU work). `None` when --otel-endpoint
        // is absent. model.id / topology auto-derive from the serve config + --tp world unless
        // --otel-model-id / --otel-topology override.
        let otel_sink = match otel_cfg.clone() {
            Some(cfg) => {
                let sink = gb10_inference::otel::OtelSink::new(
                    cfg, &model_name, &gb10_inference::otel::topology_from_world(tp_world));
                tokio::spawn(gb10_inference::otel::run_sender(std::sync::Arc::clone(&sink)));
                Some(sink)
            }
            None => None,
        };
        gb10_inference::memwatch::phase("srv:post-otel");

        // --tool-call-format <xml|json>: Froggeric-class templates accept a `tool_call_format`
        // kwarg; 'json' selects the model's NATIVE Qwen tool-call syntax (what our server's
        // tool parser handles). Templates that ignore the variable are unaffected.
        let mut tokenizer = tokenizer;
        if let Some(fmt) = parse_arg(args, "--tool-call-format") {
            match fmt {
                "xml" | "json" => tokenizer.tool_call_format = Some(fmt.to_string()),
                other => {
                    eprintln!("--tool-call-format must be xml|json (got '{other}')");
                    std::process::exit(1);
                }
            }
        }
        // Froggeric-class agentic guard: thinking auto-disables when the request carries tools.
        if args.iter().any(|a| a == "--auto-disable-thinking-with-tools") {
            tokenizer.auto_disable_think_tools = true;
        }
        // Keep the HTTP clamp aligned with the scheduler reserve (PR #4). Auto may re-enable
        // later, so retain the startup value for the server lifetime; the scheduler's own
        // `admit` re-checks per request (its clamp is authoritative).
        let decode_headroom = gb10_inference::batch::decode_headroom(mtp_active_at_start);
        gb10_inference::metrics::set_max_batch(max_batch);
        // FLASHNEXT-DF2-B (ISSUES work[113]): the pack's generation_config.json decides the
        // unsent-field sampler defaults (LEGACY 0.7/0.8/20 only when the pack has none).
        // LOOPFIX (2026-11-06): a pack WITHOUT generation_config.json falls back to its Qwen3.5 / 3.6 /
        // 3.8 model card (qwen35_card: per-mode, per-size), not LEGACY 0.7 / 0.8 / 20 for both modes; a
        // pack WITH the file is unchanged (LEGACY fallback => its triple for both modes, as before).
        let has_gen = std::path::Path::new(&format!("{}/generation_config.json", model_path.trim_end_matches('/'))).exists();
        let card = if has_gen { None } else { gb10_inference::server::qwen35_card(&model_path).map(|(d, _)| d) };
        let (sampling_defaults, sd_src) = match card {
            Some(d) => (d, "the Qwen3.5 model card (the pack has no generation_config.json)"),
            None => gb10_inference::server::SamplingDefaults::from_pack(
                &model_path, gb10_inference::server::SamplingDefaults::LEGACY),
        };
        println!("[serve] {}", sampling_defaults.describe(sd_src));
        let state = AppState {
            sampling_defaults,
            scheduler: stx,
            tokenizer: Arc::new(tokenizer),
            model_name: model_name.clone(),
            default_max_tokens,
            default_rep_penalty,
            default_presence_penalty,
            default_frequency_penalty,
            // None = unspecified: each model family's OWN template picks its baked-in default
            // (Qwen -> xhigh, hy_v3 -> low). An explicit --reasoning-effort sets a server-wide
            // override. Accept the union of both families' vocabularies.
            reasoning_effort: parse_arg(args, "--reasoning-effort").map(|s| s.to_string())
                .map(|e| match e.as_str() {
                    "no_think" | "low" | "high" | "medium" | "xhigh" => e,
                    other => {
                        eprintln!("--reasoning-effort must be no_think|low|high|medium|xhigh (got '{other}')");
                        std::process::exit(1);
                    }
                }),
            // W1 (Phase 13): `--thinking auto|on|off`. Auto is the default and passes NOTHING to
            // the chat template, so the model's own chat_template.jinja decides — this is the fix
            // that lets a customer-edited template (thinking defaulted OFF) actually take effect.
            // A request's `chat_template_kwargs.enable_thinking` overrides this per call.
            thinking: parse_arg(args, "--thinking").map(|s| {
                gb10_inference::tokenizer::ThinkingMode::parse(s).unwrap_or_else(|| {
                    eprintln!("--thinking must be auto|on|off (got '{s}')");
                    std::process::exit(1);
                })
            }).unwrap_or(gb10_inference::tokenizer::ThinkingMode::Auto),
            // excerpt; explicit numeric arg overrides the cap.
            output_prompts: args.iter().position(|a| a == "--output-prompts")
                .map(|i| args.get(i + 1).and_then(|v| v.parse::<usize>().ok()).unwrap_or(6000))
                .unwrap_or(0),
            max_seq_len,
            decode_headroom,
            prefix_cache,
            // this scheduler applies a schema from the first token
            schema_defer: false,
            // PR #30 port: this scheduler computes no log probabilities (served with logprobs null, counted);
            // W2: it applies response_format masks itself (greedy MTP verify included), the TP head too
            logprobs_ok: false,
            schema_ok: true,
            // Vision tower(s) (v3): load the 333 model.visual.* tensors, and build the GPU fast path
            // on the shared device. Fails softly — a text-only / GPU-less server keeps its exact
            // behavior. --vision-cpu forces the CPU reference tower (diagnostic escape hatch).
            vision_tower: vision_tower.clone(),
            vision_gpu: vision_gpu.clone(),
            vision_cpu,
            otel: otel_sink,
            stop_ids: stop_ids_for_state,
        };

        // Issue #34: a --chat-template override is a configuration the owner must SEE, not one
        // he has to dig out of debug logs. One line, always, naming the file that won and its
        // hash (template_meta.origin is the flag's path when the override engaged).
        if let Some((origin, sha, nbytes)) = state.tokenizer.template_meta.clone() {
            let argv: Vec<String> = std::env::args().collect();
            if gb10_inference::opt_or_arg(&argv, "--chat-template").is_some() {
                println!("[tokenizer] chat template OVERRIDE in effect: {} (sha256 {} {}, {} bytes)", origin, &sha[..12], nbytes, nbytes);
            }
        }

        // Phase-2 A3: ONE boot line that pins the rendering/penalty configuration this process
        // actually runs under. Every Phase-1 report had to infer these from campaign notes, and
        // the ledger's config fingerprint could not tell two legs apart without them (te_hard2
        // vs te_hard3). Log line only — nothing here changes behaviour.
        if gb10_inference::boot_report::debug() {
            let (tpl_origin, tpl_sha, tpl_bytes) = state.tokenizer.template_meta.clone()
                .unwrap_or_else(|| ("<none: legacy hand-rolled template>".to_string(),
                                    "<none>".to_string(), 0));
            // P10 W3a (PLAN/PRODCONFIG_TABLE.md §2): the fields above are tokenizer/serving state.
            // The record block a PERFORMANCE leg needs also carries the LANE, TOPOLOGY and DRAFT
            // knobs — without them an FP8 trunk cannot be told from an NVFP4 one, TP2 from TP4,
            // depth 4 from depth 8, block 8 from block 16 (the doc's §0.2 finding: rule 11's
            // "config of record = its [boot] line" was UNSATISFIABLE as written). These are the
            // RESOLVED values where a resolver exists (`--mtp=auto` prints what is configured),
            // and `<default:...>` marks a flag that was not passed, so a default is never silently
            // read as an explicit choice (the `--df2-round-shard` default-on trap of K2 §W0').
            let tp_world_boot = parse_tp_world(args).unwrap_or(1);
            let nodes_boot = parse_arg(args, "--nodes").unwrap_or("<none>");
            let kvb_s = gb10_inference::opt_or_arg(args, "--kv-cache");
            let kvb = kvb_s.as_deref().unwrap_or("<default:bf16>");
            let mtp_boot_s = gb10_inference::opt_or_arg(args, "--mtp");
            let mtp_boot = mtp_boot_s.as_deref().unwrap_or("<default:auto>");
            let mtp_depth_boot = parse_arg(args, "--mtp-depth").unwrap_or("<default:auto>");
            let spec_src_boot = parse_arg(args, "--spec-source").unwrap_or("<default:mtp>");
            let df2_block_boot = parse_arg(args, "--df2-block")
                .map(|v| v.to_string())
                .unwrap_or_else(|| format!("<default:{}>", gb10_inference::dflash2::block()));
            let df2_carry_boot = parse_arg(args, "--df2-carry").unwrap_or("<default:off>");
            let df2_shard_boot = parse_arg(args, "--df2-round-shard").unwrap_or("<default:on>");
            let mtp_lanes_boot = matches!(gb10_inference::opt_or_arg(args, "--mtp-lanes").as_deref().unwrap_or("off"),
                                          "on" | "true" | "1" | "yes");
            let pf_sched_s = gb10_inference::opt_or_arg(args, "--prefill-sched");
            let pf_sched_boot = pf_sched_s.as_deref().unwrap_or("cursor");
            let fp8_pf_s = gb10_inference::opt_or_arg(args, "--fp8-prefill");
            let fp8_pf_boot = fp8_pf_s.as_deref().unwrap_or("<default:off>");
            let draft_s = gb10_inference::opt_or_arg(args, "--draft-dir");
            let draft_boot = draft_s.as_deref().unwrap_or("<none>");
            eprintln!("[boot] model_dir={} template={} template_sha256={} template_bytes={} \
                       tool_call_format={} reasoning_effort_default={} thinking={} \
                       auto_disable_thinking_with_tools={} presence_penalty_default={} \
                       rep_penalty_default={} freq_penalty_default={} max_seq_len={} \
                       prefix_cache={} max_batch={} stop_ids={:?} \
                       trunk_dir={} tp_world={} nodes={} kv_cache={} mtp={} mtp_depth={} \
                       spec_source={} df2_block={} df2_carry={} df2_round_shard={} mtp_lanes={} \
                       prefill_sched={} fp8_prefill={} draft_dir={} tripwire={:?} pool_census={}",
                parse_arg(args, "--model-dir").unwrap_or_else(|| "<none>".into()), tpl_origin, tpl_sha, tpl_bytes,
                state.tokenizer.tool_call_format.as_deref().unwrap_or("<template-default:xml>"),
                state.reasoning_effort.as_deref().unwrap_or("<template-default:medium>"),
                state.thinking.as_str(),
                state.tokenizer.auto_disable_think_tools,
                state.default_presence_penalty, state.default_rep_penalty,
                state.default_frequency_penalty, state.max_seq_len, state.prefix_cache,
                max_batch, state.stop_ids,
                parse_arg(args, "--model-dir").unwrap_or("<none>"),
                tp_world_boot, nodes_boot, kvb, mtp_boot, mtp_depth_boot,
                spec_src_boot, df2_block_boot, df2_carry_boot, df2_shard_boot, mtp_lanes_boot,
                pf_sched_boot, fp8_pf_boot, draft_boot,
                gb10_inference::gpu::tripwire_get(), gb10_inference::gpu::census_on());
        }
        gb10_inference::memwatch::phase("serve:pre-bind");
        let prefix_on = state.prefix_cache; // OUT1: read before `state` moves into the router.
        let app = create_router(state);
        let listener = gb10_inference::server::http_listen(port).await;
        let url_boot = listener.local_addr().map_or_else(|_| format!("0.0.0.0:{port}"), |a| a.to_string());
        if gb10_inference::boot_report::debug() {
            println!("OpenAI-compatible server running on http://{}", url_boot);
            println!("Serving model: {}  (GET /v1/models)", model_name);
            println!("POST /v1/chat/completions   max_batch={}  default max_tokens={}", max_batch, default_max_tokens);
            println!("POST /v1/tokenize           vLLM-compatible: {{tokens, count, max_model_len}} — pure tokenizer, no forward (prompt | messages)");
            println!("POST /v1/detokenize         vLLM-compatible: {{model, prompt}} — decode half of the pair (exact-N prompt building)");
        }
        {
            // OUT1: THE one idle line. Field values are the RESOLVED posture (defaults shown resolved).
            let kvb_s = gb10_inference::opt_or_arg(args, "--kv-cache");
            let kvb = kvb_s.as_deref().unwrap_or("bf16");
            let mtp_b = if mtp_force == Some(false) { "off" }
                        else if !mtp_active_at_start { "none" }
                        else if mtp_force.is_some() { "on" } else { "auto" };
            gb10_inference::boot_report::serving(&gb10_inference::boot_report::ServingLine {
                family: "nvfp4", model: &model_name, tp: parse_tp_world(args).unwrap_or(1),
                ctx: max_seq_len, batch: max_batch, kv: kvb,
                prefix: if prefix_on { "on" } else { "off" },
                mtp: mtp_b, spec: spec_source.cli_name(), port,
                url: format!("http://{url_boot}"),
            });
        }
        // WP01: TCP_NODELAY on accepted sockets — SSE token chunks are small writes that Nagle +
        // delayed-ACK can hold for tens of ms. Bytes on the wire are unchanged, only when they
        // leave. --wp01-off=1 = legacy (OS default, Nagle on).
        let wp01_off = gb10_inference::opts::var(gb10_inference::opt!("wp01-off")).map_or(false, |v| !v.is_empty() && v != "0");
        let srv = axum::serve(listener, app);
        let srv = if wp01_off { srv } else { srv.tcp_nodelay(true) };
        srv.await.unwrap();
    });
}
