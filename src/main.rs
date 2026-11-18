#[cfg(test)] #[macro_use] #[path = "src_text.rs"] mod src_text;
use gb10_inference::{engine::GB10InferenceEngine, server::{create_router, AppState}, tokenizer::QwenTokenizer};
use gb10_inference::dsv4_sse::{dsv4_partial_overlap, dsv4_sse_chunk, Dsv4SseStreamer};
use std::env;
use std::sync::Arc;

// Entry point of the binary: `fn main` parses the mode flags and dispatches. Every mode driver, probe, bench
// and the server entry lives in `src/cli/` (see its `mod.rs` map); items called from here are `pub(crate)` there.
mod cli;
use cli::*;

fn main() {
    let args: Vec<String> = env::args().collect();
    // BASICS (owner 2026-11-02): `--version` prints the version line and exits; every other run
    // prints it as its FIRST line (stderr under --print-config, whose stdout stays byte-identical;
    // --help prints it as the first line of the help text).
    if args.iter().skip(1).any(|a| a == "--version") {
        println!("{}", gb10_inference::startup::version_line());
        return;
    }
    // --update: self-update to the latest GitHub release (installed apps only; never a TP
    // node process). Explicit only — the boot path below merely NOTIFIES.
    if args.iter().skip(1).any(|a| a == "--update") {
        let tp_world = parse_tp_world(&args).unwrap_or(1);
        if tp_world >= 2 && args.iter().any(|a| a == "--node") {
            eprintln!("refused: a TP node must never replace itself out from under a head — run --update on the head, and install the same version on every node");
            std::process::exit(2);
        }
        std::process::exit(gb10_inference::self_update::update(false));
    }
    gb10_inference::startup::print_banner(&args);
    // Boot-time version notification (never a replacement): one colored line when a newer
    // GitHub release exists. SILENT on no-network / rate-limit / curl-missing; skip entirely
    // under --no-update-check or on a TP node (the head is the only rank that should talk).
    if !args.iter().skip(1).any(|a| a == "--no-update-check") {
        let tp_world = parse_tp_world(&args).unwrap_or(1);
        let is_node = args.iter().skip(1).any(|a| a == "--node");
        if tp_world < 2 || !is_node {
            gb10_inference::self_update::boot_check();
        }
    }
    // Issue #33: a --model-dir that looks like a Hugging Face repo id (org/name, no path
    // separators of its own) resolves to the newest LOCAL snapshot under the HF cache before
    // any mode reads it. No network: an uncached id fails loudly here, naming the roots
    // searched and the download command.
    let mut args: Vec<String> = args;
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == "--model-dir" || (args[i].starts_with("--model-dir=") && false) {
            let v = &args[i + 1];
            if gb10_inference::hf_cache::looks_like_repo_id(v) {
                match gb10_inference::hf_cache::resolve(v, None) {
                    Ok(p) => {
                        let ps = p.display().to_string();
                        eprintln!("[hf] --model-dir {v} -> local snapshot {ps}");
                        args[i + 1] = ps;
                    }
                    Err(e) => { eprintln!("FATAL: {e}"); std::process::exit(2); }
                }
            }
        }
    }
    // CLI-1 (AGENTS §7): every engine option is a command-line flag. An environment variable the
    // engine used to read is refused here, loudly, naming its flag; then the registry is filled
    // from argv once, before any worker / GPU thread starts.
    gb10_inference::opts::reject_env();
    // (then the HTTP-layer options whose values the registry cannot range-check by type, e.g.
    // --inject-stall ROUND:SECONDS: refused here, in milliseconds, not after the model has loaded)
    if let Err(e) = gb10_inference::opts::parse_cli(&args).and_then(|_| gb10_inference::server::check_serve_options(&args)) {
        eprintln!("FATAL: {e}");
        std::process::exit(2);
    }
    // NIGHT1: the --gpu-wait policy goes into the CUDA primary context's scheduling flags BEFORE any context
    // exists (block = every wait sleeps until the GPU signals). A TP node applies the head's value again after
    // install_head. Help / version output never touches the driver.
    if !args.iter().any(|a| a == "--help" || a == "-h" || a == "--version") {
        gb10_inference::host_wait::apply_context_policy();
    }
    // OUT1: the boot report starts here — every family's boot prints through it from the first
    // line on (the level comes from the store; a TP node refreshes it after install_head).
    gb10_inference::boot_report::init_from_args(&args);
    // F9 (owner decision A, 2026-10-11): the hand-read value flags refuse the same swallow the
    // registry refuses (F7) - one pass over argv before any warning, --help, --print-config,
    // mode dispatch, model load or TP shipping; nodes included (their own --node/--port line).
    if let Some(msg) = gb10_inference::opts::refuse_hand_flag_swallows(&args) {
        eprintln!("FATAL: {msg}");
        std::process::exit(2);
    }
    // BASICS: the run's configuration right after the version line (not for --help / --print-config).
    gb10_inference::startup::print_config(&args);
    // F9b: one info line when --tp is bare (stderr only; --print-config stdout stays byte-identical)
    if let Some(hint) = gb10_inference::opts::tp_bare_hint(&args) {
        eprintln!("{hint}");
    }
    // F7b (owner ruling 2026-10-11): a "--"-token that is no option at all gets ONE warning line
    // and is ignored exactly as before — warn, never refuse. Printed here on the head only: a TP
    // node's options arrive resolved on TpConfig, so a node never repeats the line.
    if !args.iter().any(|a| a == "--node") {
        let unknown = gb10_inference::opts::unknown_warnings(&args);
        for w in &unknown { gb10_inference::reprintln!("[opts] {w}"); }
        if !unknown.is_empty() { gb10_inference::metrics::set_unknown_options(unknown.len() as u64); }
    }
    // DF2 block-16: resolve the draft block FIRST, before any DFlash2 buffer or round is built.
    // It is a process global read by every sizing/launch site in src/dflash2/, so it must be
    // installed once and never changed; block 8 reproduces today's allocations and launches
    // exactly, which is what makes flag-off identity structural rather than re-verified.
    // (On a TP node the head's value arrives on TpConfig and is installed at the node install
    // site below — a node is never given --df2-block by hand.)
    if let Some(v) = parse_arg(&args, "--df2-block") {
        let b: usize = v.parse().unwrap_or_else(|_| {
            panic!("--df2-block must be 8 or 16 (got '{v}')")
        });
        gb10_inference::dflash2::set_block(b).unwrap_or_else(|e| panic!("--df2-block: {e}"));
    }
    // DFLASH2-GEN: the mask-row override lever (default ON; the OFF arm is the A/B cell).
    // Rides TpConfig to the nodes (SPMD); single-box reads the global directly.
    if let Some(v) = parse_arg(&args, "--df2-mask-override") {
        let on = matches!(v, "on" | "true" | "1" | "yes");
        gb10_inference::dflash2::set_mask_override(on);
        if !on {
            println!("[df2] mask-row override DISABLED by --df2-mask-override off — a DAMAGED \
                      quantized mask row will serve as-is (logged at round load)");
        }
    }
    cli_opts_bridge(&args);

    // PR veloGB10#4 (memwatch): host-memory watchdog for the unified-memory box — a process
    // that EXITS under pressure frees everything; the kernel OOM path can hang the machine.
    // --mem-watchdog-gb=<floor> (default 5, 0 = off).
    gb10_inference::memwatch::start();
    // S-B14: the 1 Hz unified-memory timeline (--mem-trace=1) — starts at boot so the
    // serve-prep sink window is fully covered by the trace, not just the phase probes.
    gb10_inference::memwatch::trace_loop();

    // ---- PLAN/TRIPWIRE_SPEC.md (P10 W2): the permanent host-side OOB / alloc tripwire --------
    // `--tripwire <off|observe|assert>` sets the posture; `--pool-census` turns on the per-alloc
    // census (never for a timing run); `--tripwire-selftest` is the negative control and exits.
    {
        use gb10_inference::gpu::{census_set, tripwire_set, Tripwire};
        match parse_arg(&args, "--tripwire") {
            None | Some("off") => tripwire_set(Tripwire::Off),
            Some("observe") => tripwire_set(Tripwire::Observe),
            Some("assert") => {
                // Spec §5.8: under TP an asymmetric abort is WORSE than the bug it reports (the
                // FIX #4 rank-asymmetry amplifier killed a node and tripped the TP watchdog). The
                // downgrade is deterministic, so both ranks agree on the posture.
                let world = parse_tp_world(&args).unwrap_or(1);
                if world > 1 {
                    eprintln!("[tripwire] --tripwire assert DOWNGRADED to observe: this process is a \
                               TP rank (world={world}); an asymmetric abort is worse than the bug");
                    tripwire_set(Tripwire::Observe);
                } else {
                    tripwire_set(Tripwire::Assert);
                }
            }
            Some(other) => {
                eprintln!("--tripwire must be off|observe|assert (got '{other}')");
                std::process::exit(1);
            }
        }
        census_set(args.iter().any(|a| a == "--pool-census"));
        if args.iter().any(|a| a == "--tripwire-selftest") {
            std::process::exit(gb10_inference::gpu::tripwire_selftest());
        }
    }

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        print!("{}", gb10_inference::opts::help_day());
        return;
    }
    // CLIPRUNE1: everything the binary accepts (diagnostics/drills/hand-read flags included).
    if args.iter().any(|a| a == "--help-advanced") {
        print_help();
        print!("{}", gb10_inference::opts::help_advanced());
        return;
    }
    // CLI-1: the diagnostics / test-drill options (kept out of --help so it stays readable).
    if args.iter().any(|a| a == "--help-diag") {
        print!("{}", gb10_inference::opts::help_text(true));
        return;
    }
    // CLI-1: every registry option with its resolved value (+ the resolvers' view), then exit. Pass
    // it the same command line as a run to see that run's effective configuration.
    if args.iter().any(|a| a == "--print-config") {
        print!("{}", gb10_inference::opts::print_config());
        print!("{}", gb10_inference::opts::print_resolved());
        return;
    }

    // WIDEVERIFY1: the copy-span proposer replayed along recorded outputs (host only: no model,
    // no GPU). --probe-copy-span-replay --ids-file <json> [--copy-span-min-n N] [--copy-span-max-w W]
    if gb10_inference::opts::on(gb10_inference::opt!("probe-copy-span-replay")) {
        std::process::exit(gb10_inference::copy_span::probe_replay_cli());
    }

    // TUNE T1 (PLAN/AUTOTUNE_DESIGN.md §5.7): the autotuner's measurement core + self-test.
    //   ./gb10_inference --autotune --selftest --model-dir <EXL3 pack> [--max-seq-len 8192]
    //       [--max-new-tokens 384] [--autotune-li-steps 64] [--autotune-out tune/receipts]
    //       [--autotune-corpus prompts.json] [--autotune-seed N] [--tune-table off|auto|<path>]
    // Refuses unless this is an idle .13/.14 (AGENTS §1.0, design §12 G-T8). The search / table
    // writer (plain --autotune) is T2.
    // TUNE T2: plain --autotune = the search + table writer; --autotune-report <table> = the report.
    if args.iter().any(|a| a == "--autotune" || a == "--autotune-report") {
        std::process::exit(gb10_inference::exl3_forward::autotune::cli(&args));
    }

    // Check for server mode
    if args.iter().any(|a| a == "--server" || a == "server") {
        // 2026-10-30 (P7 B6): the bare `server` positional is accepted as `--server`.
        // Before: it was silently ignored, fell through to the default-generation path,
        // and the trunk qwen loader panicked ("missing tensor: lm_head.weight") on every
        // EXL3 pack — a confusing crash instead of serve (the §1b-bis.8 posture).
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
    if args.iter().any(|a| a == "--tp-reduce-bench") {
        run_tp_reduce_bench(&args);
        return;
    }
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

    // PP-prefill (PLAN/14) two-box roles: layer-split pipeline prefill across 2 GB10s.
    if args.iter().any(|a| a == "--pp-node") {
        let dir = parse_arg(&args, "--model-dir").expect("--pp-node requires --model-dir");
        gb10_inference::pp::pp_node(&dir).expect("pp-node");
        return;
    }
    if args.iter().any(|a| a == "--pp-bench-prefill") {
        let dir = parse_arg(&args, "--model-dir").expect("--pp-bench-prefill requires --model-dir");
        let seq_len: usize = parse_arg(&args, "--seq-len").and_then(|s| s.parse().ok()).unwrap_or(32768);
        let reps: usize = parse_arg(&args, "--reps").and_then(|s| s.parse().ok()).unwrap_or(3);
        let split: usize = parse_arg(&args, "--split").and_then(|s| s.parse().ok()).unwrap_or(0);
        let verify = args.iter().any(|a| a == "--verify");
        // Window-size sweep, e.g. --pp-chunk-list 8192,4096,2048 (default: PREFILL_CHUNK).
        let chunks: Vec<usize> = parse_arg(&args, "--pp-chunk-list")
            .map(|v| v.split(',').filter_map(|t| t.trim().parse::<usize>().ok()).collect())
            .unwrap_or_default();
        // split == 0 = "half", resolved INSIDE pp_bench_head after its model load — do NOT
        // pre-load here just to count layers (that delays the listener by a whole model load).
        gb10_inference::pp::pp_bench_head(&dir, seq_len, reps, split, verify, &chunks).expect("pp-bench-head");
        return;
    }

    // PP-prefill (PLAN/14) on-box split validation: layers [0,split) then [split,64) continuing
    // from the returned residual must be BYTE-IDENTICAL to the monolithic prefill (token, final
    // hidden, and full post-prefill KV/GDN/conv state). The exactness that licenses the 2-box
    // pipeline (the only cross-box artifact is the bf16 residual itself).
    if args.iter().any(|a| a == "--probe-ppsplit") {
        run_probe_ppsplit(&args);
        return;
    }

    // DSV4 single-process isolated prefill bench (R2.2): loads the full trunk single-process and
    // times `forward` (auto-chunked at PREFILL_CHUNK) over a synthetic prompt. This is the
    // isolated-prefill measurement the audit lacks (the --head path is wall-derived). Set
    // --prefill-trace so the per-chunk + path timers fire too. `--dsv4-bench-prefill
    // --model-dir <bundle> --seq-len N [--reps R]`.
    if args.iter().any(|a| a == "--dsv4-bench-prefill") {
        run_dsv4_bench_prefill(&args);
        return;
    }

    // DSV4 single-process isolated DECODE bench (R3.0): the per-lever decode iteration harness
    // and the ncu steady-state target. `--dsv4-bench-decode --model-dir <bundle>
    // [--max-new-tokens N] [--prompt-len P]`.
    if args.iter().any(|a| a == "--dsv4-bench-decode") {
        run_dsv4_bench_decode(&args);
        return;
    }

    // MTP verify lossless probe
    if args.iter().any(|a| a == "--bench-verify") {
        run_bench_verify(&args);
        return;
    }

    // qwen4_exp (Qwen3.8-Flash-Next): `--ple-offload ssd` keeps the ~31 GB PLE n-gram table on
    // disk and reads rows per forward; `none`/absent = device-resident. (The env var is internal
    // transport only — the user interface is this CLI flag; read by GpuModel::ple_offload_ssd.)
    // S-H0.4a fix: this MUST run before any dispatch that loads a model (--probe-q4's footprint
    // check reads it) — it previously sat BELOW the --probe-q4 dispatch, so the flag was dead on
    // the probe path and the single-node check refused with the table counted as resident.
    // (CLI-1: --ple-offload ssd|none is a registry option, parsed and validated at startup.)
    // GDN state-divergence probe (verify_forward vs forward_decode recurrent state)
    if args.iter().any(|a| a == "--probe-state") {
        run_probe_state(&args);
        return;
    }
    // qwen4_exp bring-up probe (PR #4): smallest end-to-end generation on the GpuModel path —
    // tokenize, prefill, greedy decode, print. Family-agnostic; `--dump-logits <dir>` writes the
    // reference-oracle logits.
    if args.iter().any(|a| a == "--probe-q4") {
        run_probe_q4(&args);
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
        // No model needed (--model-dir is accepted and ignored, for old scripts).
        if let Err(e) = gb10_inference::gpu::probe_bandwidth(None) { eprintln!("--probe-bandwidth: {e:#}"); std::process::exit(1); }
        return;
    }
    // VIS-1: vision-tower oracle probe (no text model). Loads the tower from --model-dir (raw bf16, NVFP4 or
    // the EXL3 pack), reports the load time, and optionally (a) --dump-weights DIR: writes the dense f32
    // [out,in] weights of six probe matrices for a weight-level check against the reference's own
    // reconstruction; (b) --pixels FILE --grid H,W --out-dir DIR: runs the GPU tower on those f32
    // pixel_values (row-major [H*W, wpv], t = 1) and writes pre_blocks, block_0..N-1 and merger as raw f32.
    // The comparison against the HF reference lives in scripts/vis1/ (class-different chain, AGENTS §3).
    // B32: the per-rank shard plan, checked without a GPU (coverage, sparse shard dirs opened by the real
    // reader, byte round trip). --probe-shard-plan --model-dir M [--plan-worlds 2,4] [--plan-deals interleave,contig,freq] [--tmp DIR]
    if args.iter().any(|a| a == "--probe-shard-plan") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-shard-plan requires --model-dir <DIR>");
        let worlds: Vec<usize> = parse_arg(&args, "--plan-worlds").unwrap_or("2,4").split(',').filter_map(|w| w.trim().parse().ok()).collect();
        let deals: Vec<String> = parse_arg(&args, "--plan-deals").unwrap_or("interleave,contig,freq").split(',').map(|s| s.trim().to_string()).collect();
        let tmp = std::path::PathBuf::from(parse_arg(&args, "--tmp").unwrap_or("/tmp/b32_probe"));
        std::fs::create_dir_all(&tmp).expect("--tmp dir");
        if let Err(e) = gb10_inference::shard_plan::probe(std::path::Path::new(dir), &worlds, &deals, &tmp) {
            eprintln!("--probe-shard-plan: {e:#}"); std::process::exit(1);
        }
        return;
    }
    if args.iter().any(|a| a == "--probe-gpu-wait") {
        if let Err(e) = gb10_inference::host_wait::probe(&args) { eprintln!("--probe-gpu-wait: {e:#}"); std::process::exit(1); }
        return;
    }
    if args.iter().any(|a| a == "--probe-vision-tower") {
        if let Err(e) = probe_vision_tower(&args) { eprintln!("--probe-vision-tower: {e:#}"); std::process::exit(1); }
        return;
    }
    // §2.3 audit: sustained bandwidth over `--seconds N` — reveals LPDDR5x thermal derating under load.
    if args.iter().any(|a| a == "--probe-bandwidth-sustained") {
        let secs: u64 = parse_arg(&args, "--seconds").and_then(|s| s.parse().ok()).unwrap_or(180);
        if let Err(e) = gb10_inference::gpu::probe_bandwidth(Some(secs)) { eprintln!("--probe-bandwidth-sustained: {e:#}"); std::process::exit(1); }
        return;
    }
    // The gate the whole speculative path rests on: col 0 of an N-wide verify == a N=1 decode, BITWISE.
    if let Some(dir) = parse_arg(&args, "--probe-dspark-bind") {
        let _ = dir;
        run_probe_dspark_bind(&args);
        return;
    }
    // S2: P8 reference oracle synthetic artifact + probe (CPU-only; no CUDA, no trunk).
    if args.iter().any(|a| a == "--gen-dspark-synth" || a.starts_with("--gen-dspark-synth=")) {
        let dir = parse_arg(&args, "--gen-dspark-synth")
            .map(str::to_string)
            .unwrap_or_else(gb10_inference::dspark::synth::default_dir);
        run_gen_dspark_synth(&args, &dir);
        return;
    }
    if args.iter().any(|a| a == "--probe-dspark-synth" || a.starts_with("--probe-dspark-synth=")) {
        let dir = parse_arg(&args, "--probe-dspark-synth")
            .map(str::to_string)
            .unwrap_or_else(gb10_inference::dspark::synth::default_dir);
        run_probe_dspark_synth(&args, &dir);
        return;
    }
    // S2F: DFlash2 reference oracle (CPU-only; no CUDA, no trunk).
    if args.iter().any(|a| a == "--gen-dflash2-synth" || a.starts_with("--gen-dflash2-synth=")) {
        let dir = parse_arg(&args, "--gen-dflash2-synth")
            .map(str::to_string)
            .unwrap_or_else(gb10_inference::dflash2::synth::default_dir);
        run_gen_dflash2_synth(&args, &dir);
        return;
    }
    if args.iter().any(|a| a == "--replay-df2-round") {
        run_replay_df2_round(&args);
        return;
    }
    if args.iter().any(|a| a == "--replay-df2-bisect") {
        run_replay_df2_bisect(&args);
        return;
    }
    if args.iter().any(|a| a == "--probe-dflash2" || a.starts_with("--probe-dflash2=")) {
        let dir = required_dir_arg(&args, "--draft-dir", "the DFlash2 draft artifact for --probe-dflash2");
        run_probe_dflash2(&args, &dir);
        return;
    }
    if args.iter().any(|a| a == "--probe-dspark-round" || a.starts_with("--probe-dspark-round=")) {
        // value on the flag itself, else --draft-dir (the same convention as --probe-df2-round)
        let dir = parse_arg(&args, "--probe-dspark-round").filter(|v| !v.starts_with("--"))
            .map(str::to_string)
            .or_else(|| gb10_inference::opt_or_arg(&args, "--draft-dir"))
            .unwrap_or_else(|| panic!("--probe-dspark-round needs <draft-dir> (on the flag or --draft-dir)"));
        run_probe_dspark_round(&args, &dir);
        return;
    }
    if args.iter().any(|a| a == "--probe-df2-draft" || a.starts_with("--probe-df2-draft=")) {
        let dir = required_dir_arg(&args, "--draft-dir", "the DFlash2 draft artifact for --probe-df2-draft");
        run_probe_df2_draft(&args, &dir);
        return;
    }
    if args.iter().any(|a| a == "--probe-df2-round" || a.starts_with("--probe-df2-round=")) {
        // bare `--probe-df2-round --model-dir X`: parse_arg would eat `--model-dir` as the
        // value — only accept a value that isn't itself a flag.
        let dir = match parse_arg(&args, "--probe-df2-round").filter(|v| !v.starts_with("--")) {
            Some(v) => v.to_string(),
            None => required_dir_arg(&args, "--draft-dir", "the DFlash2 draft artifact for --probe-df2-round"),
        };
        run_probe_df2_round(&args, &dir);
        return;
    }
    // S5F: the draft-round CUDA-graph gate.
    if args.iter().any(|a| a == "--probe-df2-graph") {
        run_probe_df2_graph(&args);
        return;
    }
    // S5F: the prompt-prime correctness gate.
    if args.iter().any(|a| a == "--probe-df2-prime") {
        run_probe_df2_prime(&args);
        return;
    }
    // S5F2: dump the trunk's captured tap hiddens for the S3T3 chat1 prompt (dtype-generic —
    // works on the NVFP4 serving trunk AND the plain-BF16 trunk class). The output [plen,25600]
    // f32 feeds the S5F2 L0 tap-cleanliness check (vs the BF16 text model's reference taps) and
    // the L1c calibrated-correction fit (nvfp4-tap -> bf16-tap pairs).
    if args.iter().any(|a| a == "--probe-df2-tapcap") {
        run_probe_df2_tapcap(&args);
        return;
    }
    // S5F: the greedy bit-identity gate (DFlash2-on == off == MTP at temp 0).
    if args.iter().any(|a| a == "--probe-df2-lossless") {
        run_probe_df2_lossless(&args);
        return;
    }
    // S5F: the DFlash2-source distribution-exactness gate (chi-square).
    if args.iter().any(|a| a == "--bench-df2-sample") {
        run_bench_df2_sample(&args);
        return;
    }
    // S5F2 L2: the REAL-q distribution-exactness gate (sampled selector path, u*q < p accept,
    // exact relu(p-q) residual) — the L2 chi-square gate.
    if args.iter().any(|a| a == "--bench-df2-sample-realq") {
        run_bench_df2_sample_realq(&args);
        return;
    }
    // S5F: one cell-group of the on-engine τ matrix (source × regime × domains).
    if args.iter().any(|a| a == "--bench-df2-matrix") {
        run_bench_df2_matrix(&args);
        return;
    }
    if args.iter().any(|a| a == "--probe-verify-m8") {
        run_probe_verify_m8(&args);
        return;
    }
    if args.iter().any(|a| a == "--probe-binv") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-binv requires --model-dir <DIR>");
        let (gpu, _) = load_model_gpu(dir, None, 1);
        if !gpu.probe_binv() { std::process::exit(1); }
        return;
    }
    // DISPATCH_ASSERT (PLAN/DISPATCH_ASSERT_SPEC.md): the AGENTS §6 rule as a machine-checked test.
    // Asserts BOTH directions of the dispatch invariant and requires the cliff to be observable.
    if args.iter().any(|a| a == "--probe-dispatch") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-dispatch requires --model-dir <DIR>");
        gb10_inference::dispatch_assert::set_posture(gb10_inference::dispatch_assert::POSTURE_ASSERT);
        let (gpu, _) = load_model_gpu(dir, None, 1);
        if !gpu.probe_dispatch() { std::process::exit(1); }
        return;
    }
    // TP4-campaign Step-0 audit: per-rank weight accounting. `--rank R --world W` reproduces the
    // exact TP weight layout offline (shard-at-load + in-place shard, NO RDMA attach — the
    // transport never touches weights). Remember to export the same env ([tp-shard-mixers]…)
    // as the live configuration being audited.
    if args.iter().any(|a| a == "--dump-rank-weights") {
        let dir = parse_arg(&args, "--model-dir").expect("--dump-rank-weights requires --model-dir <DIR>");
        let rank: i32 = parse_arg(&args, "--rank").and_then(|s| s.parse().ok()).unwrap_or(0);
        let world: i32 = parse_arg(&args, "--world").and_then(|s| s.parse().ok()).unwrap_or(1);
        assert!(world == 1 || (world > 0 && world & (world - 1) == 0), "--world must be 1 or a power of two");
        let (mut gpu, _) = if world > 1 {
            load_model_gpu(&dir, Some(rank), world)
        } else {
            load_model_gpu(&dir, None, 1)
        };
        gpu.prepare_tp_weight_layout(rank, world);
        gpu.dump_rank_weights();
        return;
    }
    // DSV4 G3 integration probe: head-only gate (hc_head + norm + LM head vs dsv4_head.npz) by
    // default; `--layers N` loads the first N trunk layers and runs a short-prompt forward diff
    // vs the dsv4_cpu reference (the full-trunk glue check). `--oracle <dir>` overrides the npz path.
    if args.iter().any(|a| a == "--probe-dsv4") {
        if args.iter().any(|a| a == "--tp-sim-full") {
            run_probe_dsv4_tp_sim_full(&args);
        } else if args.iter().any(|a| a == "--tp-sim") {
            run_probe_dsv4_tp_sim(&args);
        } else if args.iter().any(|a| a == "--prefix") {
            run_probe_dsv4_prefix(&args);
        } else {
            run_probe_dsv4(&args);
        }
        return;
    }
    // DSV4 Phase-5 DSpark draft-mechanics probe (single-process): load the 3 DSpark stages,
    // feed the oracle's warm+draft main_hidden, validate the warm path / 133-entry index list /
    // Markov chain structure vs dsv4_cpu (chaos amendment: mechanics, never value equality).
    if args.iter().any(|a| a == "--probe-dspark") {
        run_probe_dspark(&args);
        return;
    }
    // DSV4 Phase-5 verify-rollback gate (single-process, trunk slice): snapshot the per-layer
    // attention state, run a verify forward, restore, re-verify → the two verify logits MUST be
    // bitwise-identical (proving the restore fully rewinds KV + compressor + indexer). The
    // forced-mismatch control (verify twice WITHOUT restore) must DIFFER (state really advanced).
    if args.iter().any(|a| a == "--probe-dspark-rollback") {
        run_probe_dspark_rollback(&args);
        return;
    }
    // E29-B1 DFlash drafter probe: load the Hy3-DFlash-B8 draft model and run the 8-token block
    // forward on recorded ctx features, printing per-position top-1 (+ acceptance vs the target
    // chain) and dumping the logits for the torch-golden comparison.
    //   ./gb10_inference --probe-dflash <ctx-file> [tokens.json] --model-dir <dflash-dir>
    // ctx-file: DFCTX (recorder format, tokens embedded) or DFCT (plain golden format).
    if args.iter().any(|a| a == "--probe-dflash") {
        run_probe_dflash(&args);
        return;
    }
    // DSV4 Phase-5: extract the DSpark stages from the bundle into {model-dir}/rank{0,1}/
    // dspark.safetensors (replicated; the cluster ships rank1/ to the node). One-time offline.
    if args.iter().any(|a| a == "--extract-dspark") {
        run_extract_dspark(&args);
        return;
    }
    // TP=2 half-width GEMV: does the per-node sharded decode GEMV still hit ~80% roofline? The whole
    // ~1.85x TP=2 projection hinges on it. Pass any --model-dir just for the GPU context (synthetic
    // buffers; weights unused). 27B dims: H=5120, I=17408, Q=24x256, KV=4x256.
    if args.iter().any(|a| a == "--probe-tp-gemv") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-tp-gemv requires --model-dir <DIR>");
        let (gpu, _) = load_model_gpu(dir, None, 1);
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
    // Split-K revival A/B: interleaved split-count sweep per shape (MxK,...) at N=1 on synthetic
    // NVFP4 buffers. Diagnostic-only (no model weights needed). Shapes default to the family set.
    //   --probe-splitk --model-dir 27b                 (auto family shapes)
    //   --probe-splitk --shapes 5120x17408,4096x13312
    if args.iter().any(|a| a == "--probe-splitk") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-splitk requires --model-dir <DIR>");
        let (gpu, _) = load_model_gpu(dir, None, 1);
        let rounds = parse_arg(&args, "--rounds").and_then(|v| v.parse::<u32>().ok()).unwrap_or(3);
        let shapes: Vec<(usize, usize, String)> = if let Some(sh) = parse_arg(&args, "--shapes") {
            sh.split(',').map(|s| {
                let (m, k) = s.split_once('x').expect("--shapes MxK,MxK,...");
                (m.parse().unwrap(), k.parse().unwrap(), format!("{m}x{k}"))
            }).collect()
        } else {
            vec![
                (5120, 17408, "qwen27 down FULL".to_string()),
                (5120, 8704,  "qwen27 down TP-half".to_string()),
                (17408, 5120, "qwen27 gate/up".to_string()),
                (4096, 12288, "qwen9  down".to_string()),
                (4096, 13312, "hy3    down".to_string()),
                (13312, 4096, "hy3    gate/up".to_string()),
                (2560, 9216,  "qwen4  down".to_string()),
                (2048, 6144,  "qwen2  down".to_string()),
                (1024, 3584,  "qwen08 down".to_string()),
            ]
        };
        println!("=== split-K A/B (N=1, synthetic NVFP4, interleaved S sweep, best-of-{rounds}) ===");
        println!("  roofline reference: ~245 GB/s sustained (probe-bandwidth for the live number)");
        gpu.probe_splitk(&shapes, rounds);
        return;
    }
    // E11 MoE expert-GEMM A/B: synthetic weights at the fold decode shapes (default Hy3's
    // slot-fold geometry — no model load), interleaved best-of-N across the E11 variants
    // (plain/u4/x2/rast/lb5/lb4/pdl) + the PDL overlap mechanism test.
    //   ./gb10_inference --probe-moe-gemm [--shapes MxK,MxK --rounds N]
    if args.iter().any(|a| a == "--probe-moe-gemm") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-moe-gemm requires --model-dir <DIR>");
        let (gpu, _) = load_model_gpu(dir, None, 1);
        let rounds = parse_arg(&args, "--rounds").and_then(|v| v.parse::<u32>().ok()).unwrap_or(3);
        let shapes: Vec<(usize, usize, String)> = if let Some(sh) = parse_arg(&args, "--shapes") {
            sh.split(',').map(|s| {
                let (m, k) = s.split_once('x').expect("--shapes MxK,MxK,...");
                (m.parse().unwrap(), k.parse().unwrap(), format!("{m}x{k}"))
            }).collect()
        } else {
            vec![
                (3072, 4096, "hy3 gate_up".to_string()),
                (4096, 1536, "hy3 down".to_string()),
                (4096, 13312, "hy3 gate_up FULL".to_string()),
            ]
        };
        println!("=== E11 MoE slot-fold A/B (N=1, synthetic NVFP4, interleaved, best-of-{rounds}) ===");
        gpu.probe_moe_gemm(&shapes, rounds);
        return;
    }
    // E11 decode p50 harness: per-phase p50 of one decode/MTP step at context ctx (default 2048).
    //   ./gb10_inference --bench-decode-ctx --model-dir <DIR> [--ctx N] [--runs N]
    if args.iter().any(|a| a == "--bench-decode-ctx") {
        let dir = parse_arg(&args, "--model-dir").expect("--bench-decode-ctx requires --model-dir <DIR>");
        let (gpu, _) = load_model_gpu(dir, None, 1);
        let ctx: usize = parse_arg(&args, "--ctx").and_then(|v| v.parse().ok()).unwrap_or(2048);
        let runs: usize = parse_arg(&args, "--runs").and_then(|v| v.parse().ok()).unwrap_or(3);
        let kv_stride = (ctx + 64).next_power_of_two().max(2048);
        let rows = gpu.bench_decode_at_ctx(ctx, kv_stride, runs);
        println!("=== E11 decode/MTP step p50 at ctx={ctx} (best-of-{runs}, zeros KV) ===");
        for (name, ms) in &rows {
            println!("  {name}: {ms:.2} ms");
        }
        return;
    }
    // MXFP4-native Phase 0 (QWEN_MXFP4_NATIVE_DESIGN.md §7): standalone OMMA microbenchmark on
    // the real model's fp4 tensors vs the production bf16 chain. GO/NO-GO at ~215 GB/s sustained.
    //   ./gb10_inference --probe-mxfp4 --model-dir /path/to/27b-nvfp4-full
    if args.iter().any(|a| a == "--probe-mxfp4") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-mxfp4 requires --model-dir <DIR>");
        let (gpu, _) = load_model_gpu(dir, None, 1);
        gpu.probe_mxfp4();
        return;
    }
    // Fused activation-quant byte-identity probe (EXPERT_FUSED_QUANT_RESPONSE.md §11 P1):
    // synthetic weights — byte-compares the fused quant+GEMM kernels' C AND their Bp/SFB
    // fragments against today's quant(+silu)+GEMM chain, {dense, slot, grouped} x {XSILU 0,1}
    // x N in {1,2,4,8,16} x K in {1536,4096,13312} + the edge sweep. Prints FUSED_BIT_IDENTITY
    // OK when all byte-compares pass. Native mode must be on at load.
    //   ./gb10_inference --probe-mxfp4-fused --model-dir <DIR>   (--mxfp4=1 implied)
    if args.iter().any(|a| a == "--probe-mxfp4-fused") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-mxfp4-fused requires --model-dir <DIR>");
        gb10_inference::opts::set(gb10_inference::opt!("mxfp4"), "1");   // native mode at load (OMMA modules resident)
        let (gpu, _) = load_model_gpu(dir, None, 1);
        gpu.probe_mxfp4_fused();
        return;
    }
    // S-A3-b Phase 1 (track A3): EXL3 pack census — directory parse + quadruple
    // completeness + manifest compare, NO model load. See PLAN/A3_EXL3_FORMAT_SPEC.md.
    //   ./gb10_inference --probe-exl3 --model-dir <EXL3-pack-dir>
    // API-parity G2 gate: device sampler distribution-exactness (no model load).
    //   ./gb10_inference --probe-sampler
    if args.iter().any(|a| a == "--probe-sampler") {
        gb10_inference::exl3_bench::probe_sampler().unwrap_or_else(|e| {
            eprintln!("PROBE_SAMPLER_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // WP15 gate: device penalties vs a dense transcription of the reference implementation's rep_pen.cu, verify-row
    // == plain-row exactness, ring commits, draft mirror. Synthetic rows; no model.
    //   ./gb10_inference --probe-penalties
    if args.iter().any(|a| a == "--probe-penalties") {
        gb10_inference::exl3_bench::probe_penalties().unwrap_or_else(|e| {
            eprintln!("PROBE_PENALTIES_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // WP24 gate: real-q speculative sampling distribution-exactness (reference-free two-sample
    // chi-square, 4 configs T 0.7/1.0 x top_p 0.8/0.95, top_k 20, >= 2 seeds). Stage A: kernel
    // level on synthetic rows (no model; plain-vs-plain control, negative control). Stage B with
    // --model-dir: the served scheduler, ratio MTP vs plain decode with a match-MTP control.
    //   ./gb10_inference --probe-spec-sampling [--model-dir <pack>] [--launches 200]
    //       [--trials-per-launch 512] [--seeds 2] [--top-k 20] [--draft-temperature T]
    //       [--trials 600] [--positions 6] [--max-batch 4] [--prompt ..] [--kv-cache ..] [--skip-kernel]
    //       [--quick (stage B: T 1.0 / top_p 0.95 only)]
    if args.iter().any(|a| a == "--probe-spec-sampling") {
        gb10_inference::wp24::probe_spec_sampling(&args).unwrap_or_else(|e| {
            eprintln!("PROBE_SPEC_SAMPLING_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // TP-A (EXL3 TP bring-up, PLAN/TP2_DESIGN_EXL3_2026-09-28.md §6 T1):
    //   --probe-exl3-slice --model-dir <pack> [--slice-experts N] [--gpu]   G-T1-0 slice oracle
    //   --exl3-pack-manifest --model-dir <pack>                            the D-T0-6 pack hash
    //   --probe-exl3-xtp --model-dir <pack> [--prompt s|--prompt-ids f] [--xtp-gen N]
    //       [--max-seq-len N] --xtp-out <dir>        TP=1 reference + bit-0 ulp floor (single box)
    //   --head --tp 2 --probe-exl3-xtp --model-dir <pack> --xtp-ref <dir> [--nodes ip] [--tp-trace]
    //       + `--node` on the peer                    the TP=2 run: G-T1-b + cross-TP tier
    if args.iter().any(|a| a == "--probe-exl3-slice") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-exl3-slice requires --model-dir <DIR>");
        let ne: usize = parse_arg(&args, "--slice-experts").and_then(|s| s.parse().ok()).unwrap_or(4);
        let gpu = args.iter().any(|a| a == "--gpu");
        gb10_inference::exl3_forward::xtp::probe_slice(dir, ne, gpu).unwrap_or_else(|e| {
            eprintln!("PROBE_SLICE_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    if args.iter().any(|a| a == "--exl3-pack-manifest") {
        let dir = parse_arg(&args, "--model-dir").expect("--exl3-pack-manifest requires --model-dir <DIR>");
        let t0 = std::time::Instant::now();
        match gb10_inference::cluster::pack_manifest(std::path::Path::new(dir)) {
            Ok(m) => {
                for (r, h, s) in &m.files { println!("EXL3_PACK_FILE {h} {s:>12} {r}"); }
                println!("EXL3_PACK_IGNORED (not read by the engine): {}", m.ignored.join(", "));
                let dt = t0.elapsed().as_secs_f64();
                println!("EXL3_PACK_MANIFEST {} files={} bytes={} dir={dir} ({dt:.1}s; hashed now {:.2} GB{})",
                         m.hash, m.files.len(), m.bytes, m.hashed_now as f64 / 1e9,
                         if m.hashed_now > 0 && dt > 0.0 { format!(" = {:.2} GB/s", m.hashed_now as f64 / 1e9 / dt) } else { String::new() });
            }
            Err(e) => { eprintln!("EXL3_PACK_MANIFEST_FAIL: {e:#}"); std::process::exit(1); }
        }
        return;
    }
    // TP-C: the spec program (the served scheduler offline; G-T1-e / G-T1-g / speed / G-T1-h) at TP=1:
    //   --probe-exl3-tpspec --model-dir <pack> --spec-prompts name=ids.json,... [--spec-gen N]
    //       [--spec-reps N] [--spec-depths 1,2,..] [--spec-dds 0|1] [--max-batch N] [--spec-prefix 0|1]
    //       [--spec-samp 0|quick|full] [--spec-trials N] [--spec-ident 0|1] [--max-seq-len N]
    //   (TP=2: the same flags with --head --tp 2 --nodes ip:port; the node runs it from TpConfig)
    if args.iter().any(|a| a == "--probe-exl3-tpspec") && !args.iter().any(|a| a == "--head") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-exl3-tpspec requires --model-dir <DIR>");
        let max_pos: usize = gb10_inference::opt_or_arg(&args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(8192);
        let opts = exl3_spec_opts(&args);
        let prompts = exl3_spec_prompts(&args);
        gb10_inference::exl3_serve::run_spec(dir, None, Some(prompts), max_pos, &opts).unwrap_or_else(|e| {
            eprintln!("PROBE_TPSPEC_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    if args.iter().any(|a| a == "--probe-exl3-xtp") && !args.iter().any(|a| a == "--head") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-exl3-xtp requires --model-dir <DIR>");
        let out = parse_arg(&args, "--xtp-out").expect("--probe-exl3-xtp (TP=1 reference) requires --xtp-out <DIR>");
        let gen: usize = parse_arg(&args, "--xtp-gen").and_then(|s| s.parse().ok()).unwrap_or(64);
        let max_pos: usize = gb10_inference::opt_or_arg(&args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(8192);
        let ids = exl3_xtp_prompt_ids(&args, dir);
        let opts = exl3_xtp_opts(&args);
        gb10_inference::exl3_forward::xtp::probe_xtp_ref(dir, &ids, gen, max_pos, out, &opts).unwrap_or_else(|e| {
            eprintln!("PROBE_XTP_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // NAIVE S0 gate: our decode vs pinned ext.reconstruct dumps, bitwise (CPU; dumps from a GPU burst).
    //   ./gb10_inference --probe-naive-oracle --model-dir <pack> --oracle-dir <ref>
    if args.iter().any(|a| a == "--probe-naive-oracle") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-oracle requires --model-dir <DIR>");
        let ref_dir = parse_arg(&args, "--oracle-dir").expect("--probe-naive-oracle requires --oracle-dir <REF_DIR>");
        gb10_inference::naive::frac::probe_oracle(&dir, &ref_dir).unwrap_or_else(|e| {
            eprintln!("NAIVE_ORACLE_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // NAIVE S4: the GPU slice model vs its CPU twin (first N layers, T tokens). GPU: ask first.
    //   ./gb10_inference --probe-naive-slice --model-dir <pack> [--layers N] [--tokens T]
    if args.iter().any(|a| a == "--probe-naive-fuse") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-fuse requires --model-dir <DIR>");
        let layers: usize = parse_arg(&args, "--layers").and_then(|s| s.parse().ok()).unwrap_or(2);
        gb10_inference::naive::model::probe_fuse(&dir, layers).unwrap_or_else(|e| { eprintln!("NAIVE_FUSE_FAIL: {e:#}"); std::process::exit(1); });
        return;
    }
    if args.iter().any(|a| a == "--probe-naive-slice") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-slice requires --model-dir <DIR>");
        let layers: usize = parse_arg(&args, "--layers").map(|v| v.parse().expect("--layers takes an integer")).unwrap_or(2);
        let tokens: usize = parse_arg(&args, "--tokens").map(|v| v.parse().expect("--tokens takes an integer")).unwrap_or(8);
        gb10_inference::naive::model::probe_slice(&dir, layers, tokens).unwrap_or_else(|e| {
            eprintln!("NAIVE_SLICE_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // NAIVE S4b: the whole model, one layer resident at a time, greedy decode (GPU coherence).
    //   ./gb10_inference --probe-naive-full --model-dir <pack> [--prompt S|--prompt-file F] [--gen N] [--nll] [--dump-states OUT.safetensors]
    if args.iter().any(|a| a == "--probe-naive-full") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-full requires --model-dir <DIR>");
        let from_file = parse_arg(&args, "--prompt-file").map(|f| std::fs::read_to_string(f).expect("--prompt-file"));
        let prompt = from_file.as_deref().unwrap_or(parse_arg(&args, "--prompt").unwrap_or("The capital of France is"));
        let gen: usize = parse_arg(&args, "--gen").and_then(|s| s.parse().ok()).unwrap_or(4);
        let rows_chunk: usize = parse_arg(&args, "--rows-chunk").and_then(|s| s.parse().ok()).unwrap_or(0);
        let dump = parse_arg(&args, "--dump-states");
        gb10_inference::naive::model::probe_full_dump(&dir, prompt, gen, rows_chunk, args.iter().any(|a| a == "--nll"), dump).unwrap_or_else(|e| {
            eprintln!("NAIVE_FULL_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // NAIVE S6: GEMM variant bench on the real shapes.
    if args.iter().any(|a| a == "--probe-naive-gemm-bench") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-gemm-bench requires --model-dir <DIR>");
        let m: usize = parse_arg(&args, "--gemm-m").and_then(|s| s.parse().ok()).unwrap_or(1);
        gb10_inference::naive::gpu::probe_gemm_bench(&dir, m).unwrap_or_else(|e| { eprintln!("NAIVE_GEMM_BENCH_FAIL: {e:#}"); std::process::exit(1); });
        return;
    }
    // NAIVE S8: the DSpark drafter round on the GPU vs the CPU reference (fixture), then 20 timed rounds
    // of the real drafter with the pack's embedding + lm_head.
    //   ./gb10_inference --probe-naive-drafter --model-dir <pack> [--draft-dir <dir>]
    if args.iter().any(|a| a == "--probe-naive-drafter") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-drafter requires --model-dir <DIR>");
        let draft = parse_arg(&args, "--draft-dir").unwrap_or("/mnt/models/NaiveAI/Naive-N0.5-Flash-FP8-Draft");
        gb10_inference::naive::drafter_gpu::probe_drafter(&dir, draft).unwrap_or_else(|e| { eprintln!("NAIVE_DRAFTER_FAIL: {e:#}"); std::process::exit(1); });
        return;
    }
    // NAIVE S6: the GPU sampler vs a CPU reference distribution.
    if args.iter().any(|a| a == "--probe-naive-sampler") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-sampler requires --model-dir <DIR>");
        gb10_inference::naive::decode::probe_sampler(&dir).unwrap_or_else(|e| { eprintln!("NAIVE_SAMPLER_FAIL: {e:#}"); std::process::exit(1); });
        return;
    }
    // option A: the per-request KV slot dimension — 2 sequences in 2 slots == 2 sequential forwards.
    if args.iter().any(|a| a == "--probe-naive-slots") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-slots requires --model-dir <DIR>");
        gb10_inference::naive::decode::probe_slots(&dir).unwrap_or_else(|e| { eprintln!("NAIVE_SLOTS_FAIL: {e:#}"); std::process::exit(1); });
        return;
    }
    // G1: lane isolation — m requests in one DecodeStep share no sampler state.
    if args.iter().any(|a| a == "--probe-naive-lanes") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-lanes requires --model-dir <DIR>");
        gb10_inference::naive::decode::probe_lanes(&dir).unwrap_or_else(|e| { eprintln!("NAIVE_LANES_FAIL: {e:#}"); std::process::exit(1); });
        return;
    }
    // NAIVE S6: the graphed decode step vs the eager step on one GPU.
    if args.iter().any(|a| a == "--probe-naive-graph") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-graph requires --model-dir <DIR>");
        let layers: usize = parse_arg(&args, "--layers").and_then(|s| s.parse().ok()).unwrap_or(2);
        let bench_only = args.iter().any(|a| a == "--bench-only");
        gb10_inference::naive::gpu::set_pdl(gb10_inference::naive::serve::pdl_flag(&args));
        gb10_inference::naive::decode::probe_graph(&dir, layers, bench_only).unwrap_or_else(|e| { eprintln!("NAIVE_GRAPH_FAIL: {e:#}"); std::process::exit(1); });
        return;
    }
    // NAIVE S5: the loopback TP gate on one GPU (two shards, two threads, a host channel).
    if args.iter().any(|a| a == "--probe-naive-loop") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-loop requires --model-dir <DIR>");
        let layers: usize = parse_arg(&args, "--layers").and_then(|s| s.parse().ok()).unwrap_or(2);
        let tokens: usize = parse_arg(&args, "--tokens").and_then(|s| s.parse().ok()).unwrap_or(8);
        let world: usize = parse_tp_world(&args).unwrap_or(2) as usize;
        gb10_inference::naive::serve::run_loop_probe(&dir, layers, tokens, world).unwrap_or_else(|e| { eprintln!("NAIVE_LOOP_FAIL: {e:#}"); std::process::exit(1); });
        return;
    }
    // NAIVE S6b: the prefill attention kernels (group-shared K/V) against the per-head kernels, bitwise.
    if args.iter().any(|a| a == "--probe-naive-attn-prefill") {
        gb10_inference::naive::gpu::probe_attn_prefill().unwrap_or_else(|e| { eprintln!("NAIVE_ATTN_PREFILL_FAIL: {e:#}"); std::process::exit(1); });
        return;
    }
    // NAIVE S6b: the wide prefill path vs the 16-row chunked path (bitwise), one GPU.
    if args.iter().any(|a| a == "--probe-naive-prefill") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-prefill requires --model-dir <DIR>");
        let layers: usize = parse_arg(&args, "--layers").and_then(|s| s.parse().ok()).unwrap_or(2);
        let tokens: usize = parse_arg(&args, "--tokens").and_then(|s| s.parse().ok()).unwrap_or(300);
        gb10_inference::naive::serve::run_prefill_probe(&dir, layers, tokens).unwrap_or_else(|e| { eprintln!("NAIVE_PREFILL_FAIL: {e:#}"); std::process::exit(1); });
        return;
    }
    // NAIVE S5b: a shipped shard dir vs the full pack sliced at load (bitwise), one GPU.
    if args.iter().any(|a| a == "--probe-naive-shard-dir") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-shard-dir requires --model-dir <FULL PACK>");
        let sd = parse_arg(&args, "--shard-dir").expect("--probe-naive-shard-dir requires --shard-dir <SHIPPED SHARD DIR>");
        let layers: usize = parse_arg(&args, "--layers").and_then(|s| s.parse().ok()).unwrap_or(2);
        let tokens: usize = parse_arg(&args, "--tokens").and_then(|s| s.parse().ok()).unwrap_or(8);
        gb10_inference::naive::serve::run_shard_dir_probe(&dir, &sd, layers, tokens).unwrap_or_else(|e| { eprintln!("NAIVE_SHARD_DIR_FAIL: {e:#}"); std::process::exit(1); });
        return;
    }
    // NAIVE S8: the speculative engine path vs plain greedy on one GPU.
    if args.iter().any(|a| a == "--probe-naive-spec") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-spec requires --model-dir <DIR>");
        let dd = parse_arg(&args, "--draft-dir").unwrap_or("/mnt/models/NaiveAI/Naive-N0.5-Flash-FP8-Draft");
        let layers: usize = parse_arg(&args, "--layers").and_then(|s| s.parse().ok()).unwrap_or(2);
        let block: usize = parse_arg(&args, "--spec-block").and_then(|s| s.parse().ok()).unwrap_or(4);
        let tokens: usize = parse_arg(&args, "--tokens").and_then(|s| s.parse().ok()).unwrap_or(24);
        let mode = parse_arg(&args, "--spec-mode").unwrap_or("drafter");
        let world: usize = parse_arg(&args, "--tp").and_then(|s| s.parse().ok()).unwrap_or(1);
        let sampler = parse_arg(&args, "--spec-sampler").unwrap_or("auto");
        gb10_inference::naive::serve::run_spec_probe(&dir, dd, layers, block, tokens, mode, world, sampler).unwrap_or_else(|e| { eprintln!("NAIVE_SPEC_FAIL: {e:#}"); std::process::exit(1); });
        return;
    }
    // NAIVE S5: the quad-slicing gate on one GPU (no transport).
    if args.iter().any(|a| a == "--probe-naive-shard") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-shard requires --model-dir <DIR>");
        gb10_inference::naive::shard::probe_shard(&dir).unwrap_or_else(|e| { eprintln!("NAIVE_SHARD_FAIL: {e:#}"); std::process::exit(1); });
        return;
    }
    // NAIVE S5: the two-rank identity probe (this box = head; the other runs `--node`).
    //   ./gb10_inference --probe-naive-tp --model-dir <pack> --tp 2 [--layers N] [--tokens T] [--nodes ip]
    if args.iter().any(|a| a == "--probe-naive-tp") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-tp requires --model-dir <DIR>");
        run_naive_tp_head(&args, &dir, "probe");
        return;
    }
    // NAIVE prefill: tensor-core indexer scorer vs the scalar kernel (synthetic planes).
    //   ./gb10_inference --probe-naive-idx-prefill --model-dir <pack>
    if args.iter().any(|a| a == "--probe-naive-idx-prefill") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-idx-prefill requires --model-dir <DIR>");
        gb10_inference::naive::gpu::probe_idx_prefill(&dir).unwrap_or_else(|e| {
            eprintln!("NAIVE_IDX_PREFILL_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // NAIVE S3: indexer scorer + exact top-k at long positions (synthetic, GPU vs CPU).
    //   ./gb10_inference --probe-naive-topk --model-dir <pack>
    if args.iter().any(|a| a == "--probe-naive-topk") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-topk requires --model-dir <DIR>");
        gb10_inference::naive::gpu::probe_topk(&dir).unwrap_or_else(|e| {
            eprintln!("NAIVE_TOPK_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // NAIVE T0-c: the staged body's BITWISE A/B against the per-step body, on real quads.
    // Engagement was proven by the nsys census in PLAN/NAIVE_RUN1_TP4_RESULTS.md; this proves
    // EQUALITY, which that census cannot. Same inputs, same surrounding passes, only the GEMM body.
    //   ./gb10_inference --probe-naive-a1-ab --model-dir <pack>
    if args.iter().any(|a| a == "--probe-naive-a1-ab") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-a1-ab requires --model-dir <DIR>");
        gb10_inference::naive::gpu::probe_a1_ab(&dir).unwrap_or_else(|e| {
            eprintln!("NAIVE_A1_AB_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // NAIVE S3: GPU kernel gates on real quads (decode bitwise, chain vs CPU dequant, batch invariance,
    // grouped == dense). GPU: ask the owner for the box first.
    //   ./gb10_inference --probe-naive-gemm --model-dir <pack>
    if args.iter().any(|a| a == "--probe-naive-gemm") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-gemm requires --model-dir <DIR>");
        gb10_inference::naive::gpu::probe_gemm(&dir).unwrap_or_else(|e| {
            eprintln!("NAIVE_GEMM_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // NAIVE S2c: the real pack through the first N layers on the CPU (coherence check, no GPU).
    //   ./gb10_inference --probe-naive-cpu --model-dir <pack> [--layers N] [--prompt TEXT]
    if args.iter().any(|a| a == "--probe-naive-cpu") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-cpu requires --model-dir <DIR>");
        let layers: usize = parse_arg(&args, "--layers").map(|v| v.parse().expect("--layers takes an integer")).unwrap_or(2);
        let prompt = parse_arg(&args, "--prompt").unwrap_or("The capital of France is");
        gb10_inference::naive::cpu::probe_cpu(&dir, layers, &prompt).unwrap_or_else(|e| {
            eprintln!("NAIVE_CPU_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // NAIVE 1c: chunked GPU prefill vs the CPU reference (CROSS-CLASS, the real equality check).
    //   ./gb10_inference --probe-naive-chunkxcpu --model-dir <pack> [--layers N] [--tokens T]
    if args.iter().any(|a| a == "--probe-naive-chunkxcpu") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-chunkxcpu requires --model-dir <DIR>");
        let layers: usize = parse_arg(&args, "--layers").map(|v| v.parse().expect("--layers takes an integer")).unwrap_or(2);
        let tokens: usize = parse_arg(&args, "--tokens").map(|v| v.parse().expect("--tokens takes an integer")).unwrap_or(16);
        gb10_inference::naive::model::probe_chunk_xcpu(&dir, layers, tokens).unwrap_or_else(|e| {
            eprintln!("NAIVE_CHUNK_XCPU_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // NAIVE S2b: host dequant of one layer's non-expert quads + the padded-region falsifier.
    //   ./gb10_inference --probe-naive-dequant --model-dir <pack> [--layer N]
    if args.iter().any(|a| a == "--probe-naive-dequant") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-dequant requires --model-dir <DIR>");
        let layer: usize = parse_arg(&args, "--layer").map(|v| v.parse().expect("--layer takes an integer")).unwrap_or(0);
        gb10_inference::naive::dequant::probe(&dir, layer).unwrap_or_else(|e| {
            eprintln!("NAIVE_DEQUANT_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // NAIVE S1 (PLAN/NAIVE_N05_PORT.md): pack census + inventory check, headers only.
    //   ./gb10_inference --probe-naive-pack --model-dir <Naive-N0.5-Flash EXL3 pack>
    if args.iter().any(|a| a == "--probe-naive-pack") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-naive-pack requires --model-dir <DIR>");
        gb10_inference::naive::pack::probe(&dir).unwrap_or_else(|e| {
            eprintln!("NAIVE_PACK_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    if args.iter().any(|a| a == "--probe-exl3") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-exl3 requires --model-dir <DIR>");
        gb10_inference::exl3::probe_census(&dir).unwrap_or_else(|e| {
            eprintln!("EXL3_CENSUS_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // S-A3-b Phase 2 (track A3): G-A3-1 decode oracle — our Rust decode vs pinned
    // ext.reconstruct dumps, BITWISE (mxfp4 lesson: external reference, never
    // self-consistency). Ref dir from scripts/a3b/a3b_gen_ref.py on .13.
    //   ./gb10_inference --probe-exl3-oracle --model-dir <EXL3-pack> --oracle-dir <REF_DIR>
    if args.iter().any(|a| a == "--probe-exl3-oracle") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-exl3-oracle requires --model-dir <DIR>");
        let ref_dir = parse_arg(&args, "--oracle-dir").expect("--probe-exl3-oracle requires --oracle-dir <REF_DIR>");
        gb10_inference::exl3::probe_oracle(&dir, &ref_dir).unwrap_or_else(|e| {
            eprintln!("ORACLE_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // S-A3-c (track A3): G-A3-1 through the KERNEL — the CUDA decode path (funnel shift +
    // magic-mul + dp4a + fp16 HFMA) dumped via exl3_decode_dump, bitwise vs the Rust oracle
    // (which is G-A3-1-green vs ext.reconstruct). Probe-only; loads src/ptx/exl3_bench.ptx.
    //   ./gb10_inference --probe-exl3-kernel --model-dir <EXL3-pack>
    if args.iter().any(|a| a == "--probe-exl3-kernel") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-exl3-kernel requires --model-dir <DIR>");
        gb10_inference::exl3_bench::probe_kernel_decode(&dir).unwrap_or_else(|e| {
            eprintln!("EXL3_KERNEL_DECODE_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // w5/PFIX: the chunked-prefill flash kernels (hd 128 + 256, scalar / bitwise twin / CAUSAL twin)
    // vs the sweep kernel on synthetic data (no model).
    //   ./gb10_inference --probe-exl3-flashpf
    if args.iter().any(|a| a == "--probe-exl3-flashpf") {
        gb10_inference::exl3_bench::probe_flash_prefill().unwrap_or_else(|e| {
            eprintln!("EXL3_FLASHPF_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // WP25: KV cache formats through the SERVED device helpers vs the q8 host reference (no model).
    //   ./gb10_inference --probe-exl3-kvq
    if args.iter().any(|a| a == "--probe-exl3-kvq") {
        gb10_inference::exl3_forward::probe_kv_q8_selftest().unwrap_or_else(|e| {
            eprintln!("PROBE_KVQ_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // WP25 rung 1: one KV-format arm of the quality gate (format = --kv-cache / [kv-cache];
    // q4x = the broken positive control via [kv-cache]=q4x). scripts/kvnll_arms.sh runs them all.
    //   ./gb10_inference --probe-exl3-kvnll --model-dir <EXL3-pack> --kvnll-text corpus.txt \
    //     [--kvnll-windows 8192,32768,131072] [--kvnll-tf 2048] [--kvnll-dump-ref f32.bin | --kvnll-ref f32.bin]
    //     [--kvnll-csv arm.csv] [--kvnll-needles on|off] [--kvnll-accept 256] [--kvnll-accept-ctx 0]
    //     [--kvnll-label name] [--prefill-chunk 2048] [--max-seq-len 262144]
    if args.iter().any(|a| a == "--probe-exl3-kvnll") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-exl3-kvnll requires --model-dir <DIR>");
        let num = |k: &str, d: usize| gb10_inference::opt_or_arg(&args, k).and_then(|s| s.parse::<usize>().ok()).unwrap_or(d);
        let a = gb10_inference::exl3_forward::KvNllArgs {
            dir: dir.to_string(),
            max_pos: num("--max-seq-len", 262144),
            chunk: num("--prefill-chunk", 2048),
            text: parse_arg(&args, "--kvnll-text").map(|s| s.to_string()),
            ids: parse_arg(&args, "--kvnll-ids").map(|s| s.to_string()),
            windows: parse_arg(&args, "--kvnll-windows").unwrap_or("8192,32768,131072")
                .split(',').filter_map(|s| s.trim().parse().ok()).collect(),
            tf: num("--kvnll-tf", 2048),
            dump_ref: parse_arg(&args, "--kvnll-dump-ref").map(|s| s.to_string()),
            ref_path: parse_arg(&args, "--kvnll-ref").map(|s| s.to_string()),
            out_csv: parse_arg(&args, "--kvnll-csv").map(|s| s.to_string()),
            needles: parse_arg(&args, "--kvnll-needles") != Some("off"),
            accept_new: num("--kvnll-accept", 256),
            accept_ctx: num("--kvnll-accept-ctx", 0),
            label: parse_arg(&args, "--kvnll-label").unwrap_or("arm").to_string(),
        };
        gb10_inference::exl3_forward::probe_kv_nll(&a).unwrap_or_else(|e| {
            eprintln!("PROBE_KVNLL_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // S-B9-REL-QSAREGIME: a short row's logits alone vs inside a decode step that also carries a row
    // past the sparse-attention switch (teacher-forced; served forward_step, graphs included).
    //   ./gb10_inference --probe-exl3-regime --model-dir <EXL3-pack> --regime-runner r.txt --regime-joiner j.txt
    //     [--regime-steps 1600] [--regime-csv out.csv] [--regime-order short-first|long-first] [--regime-time 0]
    //     [--prefill-chunk 2048] [--max-seq-len 32768]
    if args.iter().any(|a| a == "--probe-exl3-regime") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-exl3-regime requires --model-dir <DIR>");
        let num = |k: &str, d: usize| gb10_inference::opt_or_arg(&args, k).and_then(|s| s.parse::<usize>().ok()).unwrap_or(d);
        let a = gb10_inference::exl3_forward::RegimeArgs {
            dir: dir.to_string(),
            max_pos: num("--max-seq-len", 32768),
            chunk: num("--prefill-chunk", 2048),
            runner: gb10_inference::opt_or_arg(&args, "--regime-runner").expect("--probe-exl3-regime requires --regime-runner <file>").to_string(),
            joiner: gb10_inference::opt_or_arg(&args, "--regime-joiner").expect("--probe-exl3-regime requires --regime-joiner <file>").to_string(),
            steps: num("--regime-steps", 1600),
            out_csv: gb10_inference::opt_or_arg(&args, "--regime-csv").map(|s| s.to_string()),
            long_first: gb10_inference::opt_or_arg(&args, "--regime-order").as_deref() == Some("long-first"),
            time_steps: num("--regime-time", 0),
        };
        gb10_inference::exl3_forward::probe_regime(&a).unwrap_or_else(|e| {
            eprintln!("PROBE_REGIME_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // S-A3-f-d: chunked-prefill vs token-by-token equivalence on the EXL3 engine —
    // the serve-text correctness gate the prose smoke could not see (chat prompts
    // derailed through prefill_chunk while bench token-by-token was exact).
    //   ./gb10_inference --probe-exl3-prefillx --model-dir <EXL3-pack> [--prompt-ids <file>]
    // The absolute verdict (RESULT: EXACT | DIVERGES) can never be EXACT on the served posture
    // (Q1: ulp-amplification floor + the decode-side int8 HC mixer). L10 re-baseline:
    //   always   PREFILLX_SIG <12 hex> = md5 of the reference / chunk C= / absolute RESULT lines
    //            (== the scripts' historical grep | md5sum | cut -c1-12; served default, ids16:
    //            12baeb3946b5)
    //   --prefillx-expect-sig=<hex>  bitwise tier (S-class changes):
    //            RESULT: PREFILLX_SIG_OK | PREFILLX_SIG_MISMATCH
    //   --prefillx-tier=1            tolerance tier (N-class / output-changing prefill work;
    //            implies --prefillx-rel=1 and the bit-0 --prefillx-ulp floor): every arm's
    //            S and KV relL2 median <= 3x the run's own one-f32-ulp floor AND the chunked first
    //            token at reference rank <= 5 within 1.0 logit -> RESULT: PREFILLX_TIER_OK | _FAIL
    //   Exit code: legacy (1 on DIVERGES) unless EXPECT_SIG or TIER is set — then 0 iff every
    //   requested check passed. Diagnostics: --prefillx-rel=1|2, --prefillx-ulp=none,0,13,
    //   --prefillx-widths=2048,256 (served widths), --prefillx-qsa-keys=1.
    //   TP-4X1: a width entry g<C> / g<C>a (e.g. g2048,g4095a) runs this prompt's SERVED chunk grid
    //   (plan_grid, tail checkpoint at n-14; trailing 'a' = the removed --prefill-absorb-tail) and prints its rows.
    if args.iter().any(|a| a == "--probe-exl3-prefillx") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-exl3-prefillx requires --model-dir <DIR>");
        let a = gb10_inference::exl3_forward::FwdArgs {
            dir: dir.to_string(),
            prompt: parse_arg(&args, "--prompt").map(|s| s.to_string()),
            ids_file: parse_arg(&args, "--prompt-ids").map(|s| s.to_string()),
            max_new: 16,
            width: 1,
            ladder_dir: None,
            ladder_layers: Vec::new(),
            ladder_step: None,
            max_pos: gb10_inference::opt_or_arg(&args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(8192),
            bench_steps: 0,
            widths: Vec::new(),
            prefill_chunk: None,
        };
        gb10_inference::exl3_forward::probe_prefill_x(&a).unwrap_or_else(|e| {            eprintln!("PROBE_PREFILLX_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // COPYPF1: the --copy-span-prefill step on a known copy (GPU, one box). The copy = the prompt's own greedy
    // decode continuation of N tokens. RESULT: COPY_PREFILL_EXACT (the step's choices and committed state are bit
    // for bit those of the same tokens read as prompt) | COPY_PREFILL_MISMATCH (exit 1); also prints how many
    // positions the prompt-reading path chooses differently from the decoding path, with their logit margins.
    //   ./gb10_inference --probe-copy-prefill --model-dir <EXL3-pack> [--prompt-ids <file>] [--probe-copy-prefill-n 256]
    //       [--probe-copy-prefill-break K] [--prefill-chunk 2048] [--max-seq-len 16384]
    // PACKPF: ./gb10_inference --probe-exl3-packpf --model-dir <EXL3-pack> [--packpf-lens a,b,c] [--packpf-chunk C]
    //   [--packpf-reps N] [--max-seq-len N] — packed multi-prompt prefill vs per-prompt prefill (bitwise + time)
    if gb10_inference::opts::on(gb10_inference::opt!("probe-exl3-packpf")) {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-exl3-packpf requires --model-dir <DIR>");
        let a = gb10_inference::exl3_forward::FwdArgs {
            dir: dir.to_string(), prompt: None, ids_file: None, max_new: 16, width: 1,
            ladder_dir: None, ladder_layers: Vec::new(), ladder_step: None,
            max_pos: gb10_inference::opt_or_arg(&args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(8192),
            bench_steps: 0, widths: Vec::new(), prefill_chunk: None,
        };
        match gb10_inference::exl3_forward::probe_pack_pf(&a) {
            Ok(true) => return,
            Ok(false) => std::process::exit(1),
            Err(e) => { eprintln!("PROBE_PACKPF_FAIL: {e:#}"); std::process::exit(1); }
        }
    }
    // LANEWIDE: ./gb10_inference --probe-lane-wide-xclass --model-dir <EXL3-pack> --prompt-ids <file> [--probe-lane-wide-rows S]
    if gb10_inference::opts::on(gb10_inference::opt!("probe-lane-wide-xclass")) {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-lane-wide-xclass requires --model-dir <DIR>");
        let a = gb10_inference::exl3_forward::FwdArgs {
            dir: dir.to_string(), prompt: None, ids_file: parse_arg(&args, "--prompt-ids").map(|s| s.to_string()),
            max_new: 16, width: 1, ladder_dir: None, ladder_layers: Vec::new(), ladder_step: None,
            max_pos: gb10_inference::opt_or_arg(&args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096),
            bench_steps: 0, widths: Vec::new(), prefill_chunk: None,
        };
        let steps = gb10_inference::opts::get(gb10_inference::opt!("probe-lane-wide-rows")).and_then(|s| s.trim().parse().ok()).unwrap_or(12);
        match gb10_inference::exl3_forward::probe_lane_wide_xclass(&a, steps) {
            Ok(_) => return,
            Err(e) => { eprintln!("PROBE_LANE_WIDE_XCLASS_FAIL: {e:#}"); std::process::exit(1); }
        }
    }
    // LANEWIDE-EXACT: ./gb10_inference --probe-lane-wide-pin --model-dir <EXL3-pack> [--probe-lane-wide-reps N]
    if gb10_inference::opts::on(gb10_inference::opt!("probe-lane-wide-pin")) {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-lane-wide-pin requires --model-dir <DIR>");
        let reps = gb10_inference::opts::ranged(gb10_inference::opt!("probe-lane-wide-reps")).map(|v| v as usize).unwrap_or(7);
        match gb10_inference::exl3_forward::probe_lane_wide_pin(dir, reps) {
            Ok(true) => return,
            Ok(false) => std::process::exit(1),
            Err(e) => { eprintln!("PROBE_LANE_WIDE_PIN_FAIL: {e:#}"); std::process::exit(1); }
        }
    }
    // LANEWIDE-MID: ./gb10_inference --probe-mid-gemm --model-dir <EXL3-pack> [--probe-lane-wide-rows ..]
    if gb10_inference::opts::on(gb10_inference::opt!("probe-mid-gemm")) {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-mid-gemm requires --model-dir <DIR>");
        let rows: Vec<usize> = gb10_inference::opts::get(gb10_inference::opt!("probe-lane-wide-rows"))
            .unwrap_or_else(|| "16,24,32,48,64,96,128,176,256".to_string()).split(',').filter_map(|x| x.trim().parse().ok()).collect();
        let reps = gb10_inference::opts::ranged(gb10_inference::opt!("probe-lane-wide-reps")).map(|v| v as usize).unwrap_or(7);
        match gb10_inference::exl3_forward::dres::probe_mid_gemm(dir, &rows, reps) {
            Ok(true) => return,
            Ok(false) => std::process::exit(1),
            Err(e) => { eprintln!("PROBE_MID_GEMM_FAIL: {e:#}"); std::process::exit(1); }
        }
    }
    // LANEWIDE-DRES: ./gb10_inference --probe-dense-resident --model-dir <EXL3-pack> --prompt-ids <file> [--prefill-chunk C]
    if gb10_inference::opts::on(gb10_inference::opt!("probe-dense-resident")) {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-dense-resident requires --model-dir <DIR>");
        let a = gb10_inference::exl3_forward::FwdArgs {
            dir: dir.to_string(),
            prompt: parse_arg(&args, "--prompt").map(|s| s.to_string()),
            ids_file: parse_arg(&args, "--prompt-ids").map(|s| s.to_string()),
            max_new: 16, width: 1, ladder_dir: None, ladder_layers: Vec::new(), ladder_step: None,
            max_pos: gb10_inference::opt_or_arg(&args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(16384),
            bench_steps: 0, widths: Vec::new(),
            prefill_chunk: parse_arg(&args, "--prefill-chunk").and_then(|s| s.parse().ok()),
        };
        match gb10_inference::exl3_forward::probe_dense_resident(&a) {
            Ok(true) => return,
            Ok(false) => std::process::exit(1),
            Err(e) => { eprintln!("PROBE_DENSE_RESIDENT_FAIL: {e:#}"); std::process::exit(1); }
        }
    }
    // LANEWIDE step 1: ./gb10_inference --probe-lane-wide-cost --model-dir <EXL3-pack> --prompt-ids <file> [--probe-lane-wide-rows ..]
    if gb10_inference::opts::on(gb10_inference::opt!("probe-lane-wide-cost")) {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-lane-wide-cost requires --model-dir <DIR>");
        let a = gb10_inference::exl3_forward::FwdArgs {
            dir: dir.to_string(),
            prompt: parse_arg(&args, "--prompt").map(|s| s.to_string()),
            ids_file: parse_arg(&args, "--prompt-ids").map(|s| s.to_string()),
            max_new: 16,
            width: 1,
            ladder_dir: None,
            ladder_layers: Vec::new(),
            ladder_step: None,
            max_pos: gb10_inference::opt_or_arg(&args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(8192),
            bench_steps: 0,
            widths: Vec::new(),
            prefill_chunk: parse_arg(&args, "--prefill-chunk").and_then(|s| s.parse().ok()),
        };
        let rows: Vec<usize> = gb10_inference::opts::get(gb10_inference::opt!("probe-lane-wide-rows"))
            .unwrap_or_else(|| "16,32,64,96,128,176".to_string()).split(',').filter_map(|x| x.trim().parse().ok()).collect();
        let reps = gb10_inference::opts::ranged(gb10_inference::opt!("probe-lane-wide-reps")).map(|v| v as usize).unwrap_or(7);
        if let Err(e) = gb10_inference::exl3_forward::probe_lane_wide_cost(&a, &rows, reps) {
            eprintln!("PROBE_LANE_WIDE_COST_FAIL: {e:#}");
            std::process::exit(1);
        }
        return;
    }
    if gb10_inference::opts::on(gb10_inference::opt!("probe-copy-prefill")) {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-copy-prefill requires --model-dir <DIR>");
        let a = gb10_inference::exl3_forward::FwdArgs {
            dir: dir.to_string(),
            prompt: parse_arg(&args, "--prompt").map(|s| s.to_string()),
            ids_file: parse_arg(&args, "--prompt-ids").map(|s| s.to_string()),
            max_new: 16,
            width: 1,
            ladder_dir: None,
            ladder_layers: Vec::new(),
            ladder_step: None,
            max_pos: gb10_inference::opt_or_arg(&args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(16384),
            bench_steps: 0,
            widths: Vec::new(),
            prefill_chunk: parse_arg(&args, "--prefill-chunk").and_then(|s| s.parse().ok()),
        };
        let n = gb10_inference::opts::ranged(gb10_inference::opt!("probe-copy-prefill-n")).map(|v| v as usize).unwrap_or(256);
        let brk = gb10_inference::opts::ranged(gb10_inference::opt!("probe-copy-prefill-break")).map(|v| v as usize).unwrap_or(0);
        match gb10_inference::exl3_forward::probe_copy_prefill(&a, n, brk) {
            Ok(true) => return,
            Ok(false) => std::process::exit(1),
            Err(e) => {
                eprintln!("PROBE_COPY_PREFILL_FAIL: {e:#}");
                std::process::exit(1);
            }
        }
    }
    // WP16: intermediate recurrent checkpoints — resumed == fresh, bitwise (single process).
    //   ./gb10_inference --probe-exl3-wp16 --model-dir <EXL3-pack> --prompt-ids <A> \
    //       [--wp16-b-ids <B> | --wp16-split S] [--prefill-chunk 2048] [--max-seq-len N] [--max-new-tokens 16]
    if args.iter().any(|a| a == "--probe-exl3-wp16") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-exl3-wp16 requires --model-dir <DIR>");
        let p = gb10_inference::exl3_forward::Wp16ProbeArgs {
            base: gb10_inference::exl3_forward::FwdArgs {
                dir: dir.to_string(),
                prompt: parse_arg(&args, "--prompt").map(|s| s.to_string()),
                ids_file: parse_arg(&args, "--prompt-ids").map(|s| s.to_string()),
                max_new: parse_arg(&args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(16),
                width: 2,
                ladder_dir: None,
                ladder_layers: Vec::new(),
                ladder_step: None,
                max_pos: gb10_inference::opt_or_arg(&args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(40960),
                bench_steps: 0,
                widths: Vec::new(),
                prefill_chunk: parse_arg(&args, "--prefill-chunk").and_then(|s| s.parse().ok()),
            },
            split: parse_arg(&args, "--wp16-split").and_then(|s| s.parse().ok()),
            b_ids: parse_arg(&args, "--wp16-b-ids").map(|s| s.to_string()),
        };
        gb10_inference::exl3_forward::probe_wp16(&p).unwrap_or_else(|e| {
            eprintln!("PROBE_WP16_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // S-A3-f-d night: per-stage trace diff (forward_step reference vs
    // prefill_chunk C=1) + first-divergent-token sweep — the family-isolation
    // instrument for the serve-prefill blocker.
    //   ./gb10_inference --probe-exl3-pfxtrace --model-dir <EXL3-pack> [--prompt-ids <file>]
    if args.iter().any(|a| a == "--probe-exl3-pfxtrace") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-exl3-pfxtrace requires --model-dir <DIR>");
        let a = gb10_inference::exl3_forward::FwdArgs {
            dir: dir.to_string(),
            prompt: parse_arg(&args, "--prompt").map(|s| s.to_string()),
            ids_file: parse_arg(&args, "--prompt-ids").map(|s| s.to_string()),
            max_new: 16,
            width: 1,
            ladder_dir: None,
            ladder_layers: Vec::new(),
            ladder_step: None,
            max_pos: gb10_inference::opt_or_arg(&args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(8192),
            bench_steps: 0,
            widths: Vec::new(),
            prefill_chunk: None,
        };
        gb10_inference::exl3_forward::probe_pfx_trace(&a).unwrap_or_else(|e| {
            eprintln!("PROBE_PFXTRACE_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // PR#27-D3: synthetic, model-independent EXL3 width-matrix gate (no --model-dir).
    //   ./gb10_inference --probe-exl3-bitwidths
    if args.iter().any(|a| a == "--probe-exl3-bitwidths") {
        gb10_inference::exl3_bench::probe_bitwidths().unwrap_or_else(|e| {
            eprintln!("EXL3-BITWIDTHS FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // S-A3-c: G-A3-5 — batch invariance (y_raw/y row 0 bitwise across verify widths) +
    // GEMM correctness vs the host reference (kernel-mirrored Hadamards, fp32 GEMM).
    //   ./gb10_inference --probe-exl3-binv --model-dir <EXL3-pack> [--exl3-widths 1,2,4,8]
    if args.iter().any(|a| a == "--probe-exl3-binv") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-exl3-binv requires --model-dir <DIR>");
        let widths: Vec<usize> = parse_arg(&args, "--exl3-widths")
            .unwrap_or("1,2,4,8")
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let slab: usize = parse_arg(&args, "--exl3-slab-cols")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1280);
        gb10_inference::exl3_bench::probe_gemm(&dir, &widths, slab).unwrap_or_else(|e| {
            eprintln!("EXL3_BINV_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }

    // PACK1 LEG 1 (S-B9-CF-PACK1): the eager 2-request packed-verify probe.
    if args.iter().any(|a| a == "--probe-exl3-pack2") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-exl3-pack2 requires --model-dir <DIR>");
        let max_pos: usize = gb10_inference::opt_or_arg(&args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(32768);
        gb10_inference::exl3_forward::pack2_probe_exl3(&dir, max_pos).unwrap_or_else(|e| {
            eprintln!("EXL3_PACK2_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }

    // PACK2 LEG 2 (S-B9-CF-PACK2): the packed-ROUND correctness probe.
    if args.iter().any(|a| a == "--probe-exl3-packstate") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-exl3-packstate requires --model-dir <DIR>");
        let max_pos: usize = gb10_inference::opt_or_arg(&args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(32768);
        gb10_inference::exl3_forward::packstate_probe_exl3(&dir, max_pos).unwrap_or_else(|e| {
            eprintln!("EXL3_PACKSTATE_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    if args.iter().any(|a| a == "--probe-exl3-pack3") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-exl3-pack3 requires --model-dir <DIR>");
        let max_pos: usize = gb10_inference::opt_or_arg(&args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(32768);
        gb10_inference::exl3_forward::pack3_probe_exl3(&dir, max_pos).unwrap_or_else(|e| {
            eprintln!("EXL3_PACK3_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // FP4L: the FP4-Local probes (--probe-fp4l CPU receipts, --probe-fp4l-gpu kernel truth; src/fp4l/probe.rs)
    if gb10_inference::fp4l::probe::dispatch(&args) { return; }
    // TP-4T1: the prefill tail-chunk small-M GEMM (synthetic; bitwise vs xq_gemm_f16 + per-call timing).
    if args.iter().any(|a| a == "--probe-exl3-tailgemm") {
        gb10_inference::exl3_bench::probe_tailgemm().unwrap_or_else(|e| {
            eprintln!("EXL3_TAILGEMM_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // TP-4R1: the register-tiled prefill router GEMM (synthetic; bitwise vs xq_gemm_f16_f32 + rows32, top-10, timing).
    if args.iter().any(|a| a == "--probe-exl3-routergemm") {
        gb10_inference::exl3_bench::probe_routergemm().unwrap_or_else(|e| {
            eprintln!("EXL3_ROUTERGEMM_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // TP-4M1: the MoE prefill glue kernels (synthetic; bitwise vs the old kernels, timing).
    if args.iter().any(|a| a == "--probe-exl3-moeglue") {
        gb10_inference::exl3_bench::probe_moeglue().unwrap_or_else(|e| {
            eprintln!("EXL3_MOEGLUE_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // TP-I3 (b): the EXL3-HC-W4 section alone (synthetic; bitwise gate + standalone timing table).
    if args.iter().any(|a| a == "--probe-exl3-hcw4") {
        gb10_inference::exl3_bench::probe_hcw4().unwrap_or_else(|e| {
            eprintln!("EXL3_HCW4_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // TP-SP1: the sequence-parallel prefill's pinned-plan gate (synthetic, the served calls).
    if args.iter().any(|a| a == "--probe-exl3-sppin") {
        let spec = parse_arg(&args, "--sppin-c").unwrap_or("128-4095");
        let cs: Vec<usize> = if let Some((a, b)) = spec.split_once('-') {
            let (a, b): (usize, usize) = (a.trim().parse().unwrap_or(128), b.trim().parse().unwrap_or(2048));
            (a..=b).collect()
        } else {
            spec.split(',').filter_map(|s| s.trim().parse().ok()).collect()
        };
        let world: usize = parse_arg(&args, "--sppin-world").and_then(|s| s.parse().ok()).unwrap_or(2);
        gb10_inference::exl3_forward::probe_sp_pinned(&cs, world).unwrap_or_else(|e| {
            eprintln!("EXL3_SPPIN_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // TP-I3 (a) rung 1: per-row identity of the prefill Lt GEMMs under a row split (synthetic).
    if args.iter().any(|a| a == "--probe-exl3-ltrows") {
        // a list "2048,1024,..." or a range "A-B" (sweep: every c in [A, B], failing cases only)
        let spec = parse_arg(&args, "--ltrows-c").unwrap_or("2048,2047,1782,1024,775,513,351,96,64,35,34");
        let cs: Vec<usize> = if let Some((a, b)) = spec.split_once('-') {
            let (a, b): (usize, usize) = (a.trim().parse().unwrap_or(2), b.trim().parse().unwrap_or(2048));
            (a..=b).collect()
        } else {
            spec.split(',').filter_map(|s| s.trim().parse().ok()).collect()
        };
        let shapes = parse_arg(&args, "--ltrows-shapes").unwrap_or("all");
        gb10_inference::exl3_forward::probe_lt_rows(&cs, shapes).unwrap_or_else(|e| {
            eprintln!("EXL3_LTROWS_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // S-A3-f-b: WIDE-M GEMM gate — bitwise across MT at every M + bitwise vs the
    // m16 kernel at M=16 + rel-L2 vs the host reference. Gates the chunked-prefill
    // wide path before serving use.
    //   ./gb10_inference --probe-exl3-wide --model-dir <EXL3-pack> \
    //     [--exl3-wide-ms 1,4,16,17,32,64,256,512] [--exl3-wide-mts 1,2,4,8]
    if args.iter().any(|a| a == "--probe-exl3-wide") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-exl3-wide requires --model-dir <DIR>");
        let ms: Vec<usize> = parse_arg(&args, "--exl3-wide-ms")
            .unwrap_or("1,4,16,17,32,64,256,512")
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let mts: Vec<usize> = parse_arg(&args, "--exl3-wide-mts")
            .unwrap_or("1,2,4,8")
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let slab: usize = parse_arg(&args, "--exl3-slab-cols")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1280);
        gb10_inference::exl3_bench::probe_wide(&dir, &ms, &mts, slab).unwrap_or_else(|e| {
            eprintln!("EXL3_WIDE_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // S-A3-f-b: wide-M GEMM bench — the chunked-prefill weight sweep (rows/s).
    //   ./gb10_inference --bench-exl3-wide --model-dir <EXL3-pack> \
    //     [--exl3-wide-ms 32,64,128,256,512,1024] [--exl3-wide-mt 4] [--exl3-iters 20]
    if args.iter().any(|a| a == "--bench-exl3-wide") {
        let dir = parse_arg(&args, "--model-dir").expect("--bench-exl3-wide requires --model-dir <DIR>");
        let ms: Vec<usize> = parse_arg(&args, "--exl3-wide-ms")
            .unwrap_or("32,64,128,256,512,1024")
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let mt: usize = gb10_inference::opt_or_arg(&args, "--exl3-wide-mt")
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        let iters: usize = parse_arg(&args, "--exl3-iters")
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);
        gb10_inference::exl3_bench::bench_wide(&dir, &ms, mt, iters).unwrap_or_else(|e| {
            eprintln!("EXL3_WIDE_BENCH_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // S-A3-c cross-chain mode: x [M,K] u16 in -> full kernel chain -> y [M,N] u16 out.
    // Driven by scripts/a3c/a3c_xchain.py (their ext chain = reference; rel-L2 < 1e-3).
    if args.iter().any(|a| a == "--probe-exl3-xchain") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-exl3-xchain requires --model-dir <DIR>");
        let x_in = parse_arg(&args, "--x-in").expect("--probe-exl3-xchain requires --x-in <FILE>");
        let y_out = parse_arg(&args, "--y-out").expect("--probe-exl3-xchain requires --y-out <FILE>");
        let module = parse_arg(&args, "--module")
            .unwrap_or("model.language_model.layers.0.linear_attn.in_proj_qkv");
        gb10_inference::exl3_bench::probe_xchain(&dir, &x_in, &y_out, &module).unwrap_or_else(|e| {
            eprintln!("EXL3_XCHAIN_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // S-A3-c: standalone weight-path GEMM bench — per-class GB/s at verify widths,
    // vs the 238 GB/s roofline and the 79-83 GB/s naive decode floor (spec §10.4).
    //   ./gb10_inference --bench-exl3-gemm --model-dir <EXL3-pack> [--exl3-widths 1,2,4,8] [--exl3-iters 20]
    if args.iter().any(|a| a == "--bench-exl3-ladder") {
        let dir = parse_arg(&args, "--model-dir").expect("--bench-exl3-ladder requires --model-dir <DIR>");
        let ms: Vec<usize> = parse_arg(&args, "--exl3-widths")
            .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![1, 4, 8]);
        let kss: Vec<usize> = parse_arg(&args, "--exl3-ks")
            .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![4, 5, 8]);
        let iters: usize = parse_arg(&args, "--exl3-iters").and_then(|s| s.parse().ok()).unwrap_or(50);
        let win: usize = parse_arg(&args, "--exl3-ladder-window").and_then(|s| s.parse().ok()).unwrap_or(16);
        let skip_n: usize = parse_arg(&args, "--exl3-ladder-skip-n").and_then(|s| s.parse().ok()).unwrap_or(0);
        let out = parse_arg(&args, "--exl3-ladder-out");
        gb10_inference::exl3_bench::bench_ladder(&dir, &ms, &kss, iters, win, skip_n, out.as_deref())
            .unwrap_or_else(|e| {
                eprintln!("EXL3_LADDER_FAIL: {e:#}");
                std::process::exit(1);
            });
        return;
    }
    if args.iter().any(|a| a == "--bench-exl3-gemm") {
        let dir = parse_arg(&args, "--model-dir").expect("--bench-exl3-gemm requires --model-dir <DIR>");
        let widths: Vec<usize> = parse_arg(&args, "--exl3-widths")
            .unwrap_or("1,2,4,8")
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let iters: usize = parse_arg(&args, "--exl3-iters")
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);
        let grouped_e: usize = parse_arg(&args, "--exl3-grouped-e")
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);
        // S-A3-f-a X3: alias the grouped expert offset table so the whole run reads an
        // L2-resident trellis footprint (default 16 MB) — the H-MEM/H-SM discriminator.
        // Off = normal streaming bench.
        let l2_alias_mb: usize = if args.iter().any(|a| a == "--exl3-l2-alias") {
            parse_arg(&args, "--exl3-l2-alias")
                .and_then(|s| s.parse().ok())
                .unwrap_or(16)
        } else {
            0
        };
        gb10_inference::exl3_bench::bench_gemm(&dir, &widths, iters, grouped_e, l2_alias_mb)
            .unwrap_or_else(|e| {
            eprintln!("EXL3_BENCH_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // S-A3-d: whole-model OFFLINE forward on the EXL3 pack (probe-only; no server,
    // no scheduler, no MTP, no TP; PLE lands in Phase 0b).
    //   ./gb10_inference --bench-exl3-forward --model-dir <EXL3-pack> --prompt "..." \
    //       [--prompt-ids ids.json] [--max-new-tokens 64] [--exl3-widths 1,4,8] \
    //       [--ladder-dir /tmp/a3d_ladder --ladder-layers 0,8,16,24,32,40,47] \
    //       [--max-seq-len 8192] [--exl3-bench-steps 32]
    if args.iter().any(|a| a == "--bench-exl3-forward") {
        let dir = parse_arg(&args, "--model-dir")
            .expect("--bench-exl3-forward requires --model-dir <DIR>")
            .to_string();
        let widths: Vec<usize> = parse_arg(&args, "--exl3-widths")
            .unwrap_or("")
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let ladder_layers: Vec<usize> = parse_arg(&args, "--ladder-layers")
            .unwrap_or("0,8,16,24,32,40,47")
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let a = gb10_inference::exl3_forward::FwdArgs {
            dir,
            prompt: parse_arg(&args, "--prompt").map(|s| s.to_string()),
            ids_file: parse_arg(&args, "--prompt-ids").map(|s| s.to_string()),
            max_new: parse_arg(&args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(32),
            width: widths.first().copied().unwrap_or(1),
            ladder_dir: parse_arg(&args, "--ladder-dir").map(|s| s.to_string()),
            ladder_layers,
            ladder_step: parse_arg(&args, "--ladder-step").and_then(|s| s.parse().ok()),
            max_pos: gb10_inference::opt_or_arg(&args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(8192),
            bench_steps: parse_arg(&args, "--exl3-bench-steps").and_then(|s| s.parse().ok()).unwrap_or(0),
            widths,
            prefill_chunk: parse_arg(&args, "--prefill-chunk").and_then(|s| s.parse().ok()),
        };
        gb10_inference::exl3_forward::run(&a).unwrap_or_else(|e| {
            eprintln!("EXL3_FORWARD_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    //   S-A3-e: M1 draft-quality harness (the G-A3-4 instrument).
    //   ./gb10_inference --bench-exl3-mtp --model-dir <EXL3-pack> --prompt "..." [--mtp-depth N]
    if args.iter().any(|a| a == "--bench-exl3-mtp") {
        let dir = parse_arg(&args, "--model-dir")
            .expect("--bench-exl3-mtp requires --model-dir <DIR>")
            .to_string();
        let depth: usize = parse_arg(&args, "--mtp-depth").and_then(|s| s.parse().ok()).unwrap_or(4);
        let a = gb10_inference::exl3_forward::FwdArgs {
            dir,
            prompt: parse_arg(&args, "--prompt").map(|s| s.to_string()),
            ids_file: parse_arg(&args, "--prompt-ids").map(|s| s.to_string()),
            max_new: parse_arg(&args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(64),
            width: 1,
            ladder_dir: None,
            ladder_layers: Vec::new(),
            ladder_step: None,
            max_pos: gb10_inference::opt_or_arg(&args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(8192),
            bench_steps: 0,
            widths: Vec::new(),
            prefill_chunk: parse_arg(&args, "--prefill-chunk").and_then(|s| s.parse().ok()),
        };
        gb10_inference::exl3_forward::run_mtp_bench(&a, depth).unwrap_or_else(|e| {
            eprintln!("EXL3_MTP_BENCH_FAIL: {e:#}");
            std::process::exit(1);
        });
        return;
    }
    // P14 DFlash v1 local lane probe (single process, greedy):
    //   ./gb10_inference --probe-dflash-local /tmp/p14_ctx --model-dir <35B> \
    //        --dflash-dir <DFlash artifact> --prompt "..." --max-new-tokens N --max-seq-len M
    // Runs the v1 draft/verify/accept loop and writes the real-tap DFCT + tokens.json (the torch
    // oracle input) at <prefix>.dfct / <prefix>.tokens.json.
    if args.iter().any(|a| a == "--probe-dflash-local") {
        let i = args.iter().position(|a| a == "--probe-dflash-local").unwrap();
        let out_prefix = args.get(i + 1).filter(|s| !s.starts_with("--"))
            .cloned().unwrap_or_else(|| "/tmp/p14_ctx".to_string());
        let dir = parse_arg(&args, "--model-dir")
            .expect("--probe-dflash-local requires --model-dir <target DIR>").to_string();
        let ddir = resolve_draft_dir(&args, Some(gb10_inference::batch::SpecSource::DFlash))
            .expect("--probe-dflash-local requires --draft-dir <DIR>");
        // the trunk's tap sink arms at load (it reads the internal [draft-dir] option)
        gb10_inference::opts::set(gb10_inference::opt!("draft-dir"), &ddir);
        let (gpu, _) = gb10_inference::gpu::GpuModel::load_from_dir(&dir).expect("model load");
        let tokenizer = QwenTokenizer::from_file(&format!("{}/tokenizer.json", dir.trim_end_matches('/')))
            .expect("tokenizer");
        let prompt_text = parse_arg(&args, "--prompt").unwrap_or(
            "The invention of the printing press in the fifteenth century changed Europe forever.");
        let max_new: usize = parse_arg(&args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(32);
        let max_seq_len: usize = gb10_inference::opt_or_arg(&args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(2048);
        let prompt = tokenizer.encode(prompt_text, true).expect("encode");
        println!("probe-dflash-local: prompt {} tokens, max_new={max_new}, max_seq_len={max_seq_len}, drafter {ddir}",
                 prompt.len());
        let out = gpu.dflash_generate_local(&ddir, &prompt, max_new, max_seq_len, Some(&out_prefix));
        println!("TOKENS: {}", out.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
        println!("TEXT: {}", tokenizer.decode(&out, true).unwrap_or_default());
        return;
    }
    // MXFP4-native Phase A (quality fix): CROSS-CHAIN quality instrument — the same real prompt
    // through the native (OMMA) chain and the bf16 chain in one process, per-layer rel-L2 curve +
    // per-GEMM-input quant roundtrip on REAL hiddens + logit/token agreement + both texts.
    //   ./gb10_inference --probe-mxfp4-xchain --model-dir /path/to/3.6-27b-nvfp4-full
    if args.iter().any(|a| a == "--probe-mxfp4-xchain") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-mxfp4-xchain requires --model-dir <DIR>");
        gb10_inference::opts::set(gb10_inference::opt!("mxfp4"), "1");   // native mode at load (both layouts resident)
        // The xchain capture hooks cannot run inside the verify-graph stream capture (sync/dtoh
        // during capture invalidates it) — probes force the EAGER verify path.
        gb10_inference::opts::set(gb10_inference::opt!("no-verify-graph"), "1");
        let (gpu, _) = load_model_gpu(dir, None, 1);
        let tokenizer = QwenTokenizer::from_file(&format!("{}/tokenizer.json", dir.trim_end_matches('/')))
            .expect("tokenizer");
        let prompt_text = parse_arg(&args, "--prompt").unwrap_or(
            "The invention of the printing press in the fifteenth century changed Europe forever. \
             Before it, books were copied by hand, one at a time, and only the wealthiest \
             institutions could own a library. After it, ideas could travel faster than armies. \
             The scientific revolution, the Reformation, and the rise of the modern university \
             all depended on the ability to put the same words in front of many readers at once. \
             The parallel to our own century is obvious: the computer network has done for \
             information what the printing press did for ink. But there is an important \
             difference, which historians of technology are still arguing about.");
        let max_new: usize = parse_arg(&args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(16);
        let max_seq_len: usize = gb10_inference::opt_or_arg(&args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);
        let prompt = tokenizer.encode(prompt_text, true).expect("encode");
        println!("probe-mxfp4-xchain: prompt '{}…' ({} tokens), max_new={}", &prompt_text[..prompt_text.len().min(60)], prompt.len(), max_new);
        let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
        let (nat, bf16) = gpu.probe_mxfp4_xchain(&mut pool, &prompt, max_new, max_seq_len);
        let tnat = tokenizer.decode(&nat, true).unwrap_or_default();
        let tbf16 = tokenizer.decode(&bf16, true).unwrap_or_default();
        println!("\nTEXT native: {}", tnat);
        println!("TEXT bf16  : {}", tbf16);
        return;
    }
    // Two-LOAD cross-chain probe (the economy escape hatch --probe-mxfp4-xchain refuses):
    //   ./gb10_inference --probe-mxfp4-xchain2 /tmp/nat.bin --model-dir <D> --mxfp4=on
    //   ./gb10_inference --probe-mxfp4-xchain2 /tmp/ref.bin --model-dir <D>            (--mxfp4=off)
    //   ./gb10_inference --probe-mxfp4-xchain2-compare /tmp/nat.bin,/tmp/ref.bin --model-dir <D>
    // hy3 runs economy (fp4 > dual-layout budget), so its cross-chain validation is two loads:
    // the native OMMA chain dump vs the standard-chain dump, compared by the same report as the
    // in-process probe (prefill col, per-layer rel-L2, final logits, tokens, texts).
    if let Some(path) = parse_arg(&args, "--probe-mxfp4-xchain2") {
        let dir = parse_arg(&args, "--model-dir").expect("--probe-mxfp4-xchain2 requires --model-dir <DIR>");
        gb10_inference::opts::set(gb10_inference::opt!("no-verify-graph"), "1");   // eager verify: capture hooks vs verify-graph capture are incompatible
        let (gpu, _) = load_model_gpu(dir, None, 1);
        let tokenizer = QwenTokenizer::from_file(&format!("{}/tokenizer.json", dir.trim_end_matches('/')))
            .expect("tokenizer");
        let prompt_text = parse_arg(&args, "--prompt").unwrap_or(
            "The invention of the printing press in the fifteenth century changed Europe forever. \
             Before it, books were copied by hand, one at a time, and only the wealthiest \
             institutions could own a library. After it, ideas could travel faster than armies. \
             The scientific revolution, the Reformation, and the rise of the modern university \
             all depended on the ability to put the same words in front of many readers at once. \
             The parallel to our own century is obvious: the computer network has done for \
             information what the printing press did for ink. But there is an important \
             difference, which historians of technology are still arguing about.");
        let max_new: usize = parse_arg(&args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(16);
        let max_seq_len: usize = gb10_inference::opt_or_arg(&args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);
        let chain = if args.iter().any(|a| a == "--mxfp4=on") { "native" } else { "ref" };
        let prompt = tokenizer.encode(prompt_text, true).expect("encode");
        println!("probe-mxfp4-xchain2: chain '{chain}', prompt '{}…' ({} tokens), max_new={} -> {}",
                 &prompt_text[..prompt_text.len().min(60)], prompt.len(), max_new, path);
        let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
        let dump = gpu.probe_mxfp4_xchain2_dump(&mut pool, &prompt, max_new, max_seq_len);
        gb10_inference::gpu::xchain2_write(path, &dump).expect("write dump");
        let text = tokenizer.decode(&dump.tokens, true).unwrap_or_default();
        println!("  a0={} tokens={:?}", dump.a0, &dump.tokens[..dump.tokens.len().min(12)]);
        println!("  text: {}", text);
        return;
    }
    // E13: fold-vs-nofold cross-chain compare (two FOLDX dumps from the TP head — see the
    // --fold-xchain-dump hook in tp_serve). Per-step first-divergent layer + full per-layer
    // rel-L2 curve at step 1, token-chain divergence position.
    //   ./gb10_inference --probe-fold-xchain-compare /tmp/fold_off.bin,/tmp/fold_on.bin
    if let Some(pair) = parse_arg(&args, "--probe-fold-xchain-compare") {
        let (pa, pb) = pair.split_once(',').expect("--probe-fold-xchain-compare <a.bin>,<b.bin>");
        let is_mtpx = |p: &str| -> bool {
            std::fs::read(p).map(|b| b.len() >= 4 && &b[0..4] == b"MTPX").unwrap_or(false)
        };
        let is_prfx = |p: &str| -> bool {
            std::fs::read(p).map(|b| b.len() >= 4 && &b[0..4] == b"PRFX").unwrap_or(false)
        };
        if is_prfx(pa) {
            let a = gb10_inference::gpu::prfx_read(pa).expect("read a");
            let b = gb10_inference::gpu::prfx_read(pb).expect("read b");
            assert_eq!((a.plen, a.nsteps, a.h, a.nlayers), (b.plen, b.nsteps, b.h, b.nlayers),
                       "prfx dumps shapes differ");
            let rel = |x: &[f32], y: &[f32]| -> f64 {
                let (mut num, mut den) = (0.0f64, 0.0f64);
                for i in 0..x.len().min(y.len()) {
                    let d = x[i] as f64 - y[i] as f64;
                    num += d * d; den += y[i] as f64 * y[i] as f64;
                }
                (num / den.max(1e-30)).sqrt()
            };
            let tdiv = a.tokens.iter().zip(&b.tokens).position(|(x, y)| x != y);
            println!("=== probe-fold-xchain-compare (PRFX): {} vs {} ===", pa, pb);
            println!("prompt {} tokens, gen {}; first divergent token: {:?}",
                     a.plen, a.nsteps, tdiv);
            println!("per-layer rel-L2 (prefill final row):");
            let mut ndiv = 0;
            for li in 0..a.nlayers {
                let r = rel(&a.outputs[li], &b.outputs[li]);
                if r > 1e-9 { ndiv += 1; }
                println!("  layer {li:3}: {:.3e}", r);
            }
            println!("divergent layers: {ndiv}/{}", a.nlayers);
            return;
        }
        if is_mtpx(pa) {
            let a = gb10_inference::gpu::mtpx_read(pa).expect("read a");
            let b = gb10_inference::gpu::mtpx_read(pb).expect("read b");
            assert_eq!((a.plen, a.nsteps, a.h), (b.plen, b.nsteps, b.h), "mtpx dumps shapes differ");
            let rel = |x: &[f32], y: &[f32]| -> f64 {
                let (mut num, mut den) = (0.0f64, 0.0f64);
                for i in 0..x.len().min(y.len()) {
                    let d = x[i] as f64 - y[i] as f64;
                    num += d * d; den += y[i] as f64 * y[i] as f64;
                }
                (num / den.max(1e-30)).sqrt()
            };
            let tdiv = a.tokens.iter().zip(&b.tokens).position(|(x, y)| x != y);
            println!("=== probe-fold-xchain-compare (MTPX): {} vs {} ===", pa, pb);
            println!("generated: {}; captures: {} vs {}; first divergent token: {:?}",
                     a.nsteps, a.ncap, b.ncap, tdiv);
            let n = a.ncap.min(b.ncap).min(12);
            println!("per-draft-step rel-L2 (input, output):");
            for s in 0..n {
                println!("  step {s:2}: in {:.3e}   out {:.3e}",
                         rel(&a.io[s * 2], &b.io[s * 2]), rel(&a.io[s * 2 + 1], &b.io[s * 2 + 1]));
            }
            return;
        }
        let a = gb10_inference::gpu::fold_xchain_read(pa).expect("read a");
        let b = gb10_inference::gpu::fold_xchain_read(pb).expect("read b");
        assert_eq!((a.nlayers, a.h, a.plen, a.nsteps), (b.nlayers, b.h, b.plen, b.nsteps),
                   "fold dumps shapes differ (same prompt/model required)");
        let rel = |x: &[f32], y: &[f32]| -> f64 {
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for i in 0..x.len().min(y.len()) {
                let d = x[i] as f64 - y[i] as f64;
                num += d * d; den += y[i] as f64 * y[i] as f64;
            }
            (num / den.max(1e-30)).sqrt()
        };
        let tdiv = a.tokens.iter().zip(&b.tokens).position(|(x, y)| x != y);
        println!("=== probe-fold-xchain-compare: {} vs {} ===", pa, pb);
        println!("tokens: {} (prompt {} + gen {}); first divergent token: {:?}",
                 a.tokens.len(), a.plen, a.nsteps, tdiv);
        if let Some(dir) = parse_arg(&args, "--model-dir") {
            let tokenizer = QwenTokenizer::from_file(&format!("{}/tokenizer.json", dir.trim_end_matches('/')))
                .expect("tokenizer");
            let ta = tokenizer.decode(&a.tokens[a.plen..], true).unwrap_or_default();
            let tb = tokenizer.decode(&b.tokens[b.plen..], true).unwrap_or_default();
            println!("  a text: {ta}");
            println!("  b text: {tb}");
        }
        if let Some(d) = tdiv { println!("  a[{}..{}]={:?}", d, d + 8, &a.tokens[d..(d + 8).min(a.tokens.len())]); }
        if let Some(d) = tdiv { println!("  b[{}..{}]={:?}", d, d + 8, &b.tokens[d..(d + 8).min(b.tokens.len())]); }
        let nsteps_shown = a.nsteps.min(8);
        println!("per-step first-divergent layer (prefill feature = step 0):");
        for step in 0..=nsteps_shown {
            let mut first: Option<(usize, f64)> = None;
            let mut worst: (usize, f64) = (0, 0.0);
            for li in 0..a.nlayers {
                let r = rel(&a.outputs[step * a.nlayers + li], &b.outputs[step * a.nlayers + li]);
                if r > 1e-9 && first.is_none() { first = Some((li, r)); }
                if r > worst.1 { worst = (li, r); }
            }
            println!("  step {step:2}: first-divergent layer {:?}   max rel-L2 layer {} ({:.3e})",
                     first.map(|(l, r)| (l, r)), worst.0, worst.1);
        }
        // The full per-layer curve at step 1 (the first decode step) — the localization map.
        if a.nsteps >= 1 {
            println!("per-layer rel-L2 at step 1 (first decode):");
            for li in 0..a.nlayers {
                let r = rel(&a.outputs[1 * a.nlayers + li], &b.outputs[1 * a.nlayers + li]);
                if r > 1e-9 {
                    println!("  layer {li:3}: {:.3e}", r);
                }
            }
        }
        return;
    }
    if let Some(pair) = parse_arg(&args, "--probe-mxfp4-xchain2-compare") {
        let (pa, pb) = pair.split_once(',').expect("--probe-mxfp4-xchain2-compare <a.bin>,<b.bin>");
        let a = gb10_inference::gpu::xchain2_read(pa).expect("read a");
        let b = gb10_inference::gpu::xchain2_read(pb).expect("read b");
        assert_eq!((a.nlayers, a.h, a.vocab, a.plen, a.max_new),
                   (b.nlayers, b.h, b.vocab, b.plen, b.max_new),
                   "dump shapes differ (same model required)");
        let dir = parse_arg(&args, "--model-dir").expect("compare requires --model-dir (tokenizer)");
        let tokenizer = QwenTokenizer::from_file(&format!("{}/tokenizer.json", dir.trim_end_matches('/')))
            .expect("tokenizer");
        let rel = |x: &[f32], y: &[f32]| -> f64 {
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for i in 0..x.len().min(y.len()) {
                let d = x[i] as f64 - y[i] as f64;
                num += d * d; den += y[i] as f64 * y[i] as f64;
            }
            (num / den.max(1e-30)).sqrt()
        };
        let argmax = |l: &[f32]| -> u32 {
            let mut b = 0usize;
            for (i, v) in l.iter().enumerate() { if *v > l[b] { b = i; } }
            b as u32
        };
        println!("=== probe-mxfp4-xchain2-compare: {} vs {} ===", pa, pb);
        println!("  {} layers, h={}, vocab={}, plen={}, max_new={}",
                 a.nlayers, a.h, a.vocab, a.plen, a.max_new);
        println!("  [prefill] hidden col {}/{} rel-L2: {:.4}", a.plen / 2, a.plen, rel(&a.hcol, &b.hcol));
        println!("  [step 1]  first token a0: {} vs {}", a.a0, b.a0);
        println!("\n  per-layer hidden rel-L2 (a vs b), layer 0 = first layer:");
        println!("  {:>4}  {:>10}  {:>10}    {:>12}", "layer", "step1", "stepLast", "growth");
        for li in 0..a.nlayers {
            let r1 = rel(&a.step1_outputs[li], &b.step1_outputs[li]);
            let rl = rel(&a.step_last_outputs[li], &b.step_last_outputs[li]);
            if li % 4 == 0 || li + 1 == a.nlayers {
                println!("  {:>4}  {:>10.4}  {:>10.4}    {:>12.1}", li, r1, rl, rl / r1.max(1e-9));
            }
        }
        let lrel = rel(&a.final_logits, &b.final_logits);
        let (ta, tb) = (argmax(&a.final_logits), argmax(&b.final_logits));
        println!("\n  final logits (step {}): rel-L2 {:.4}   top-1 a {} b {}   agree {}",
                 a.max_new, lrel, ta, tb, ta == tb);
        let top10 = |l: &[f32]| -> Vec<u32> {
            let mut idx: Vec<u32> = (0..l.len() as u32).collect();
            idx.sort_by(|&x, &y| l[y as usize].partial_cmp(&l[x as usize]).unwrap());
            idx.truncate(10);
            idx
        };
        let (t10a, t10b) = (top10(&a.final_logits), top10(&b.final_logits));
        println!("  top-10 agreement: {} (a {:?} vs b {:?})",
                 t10a.iter().zip(&t10b).filter(|(x, y)| x == y).count(), &t10a[..5], &t10b[..5]);
        let div = a.tokens.iter().zip(&b.tokens).position(|(x, y)| x != y);
        println!("\n  token stream a: {:?}", &a.tokens[..a.tokens.len().min(24)]);
        println!("  token stream b: {:?}", &b.tokens[..b.tokens.len().min(24)]);
        match div {
            Some(d) => println!("  first token divergence at step {d} (a {} vs b {})", a.tokens[d], b.tokens[d]),
            None => println!("  token streams IDENTICAL over {} tokens", a.max_new),
        }
        println!("\nTEXT a: {}", tokenizer.decode(&a.tokens, true).unwrap_or_default());
        println!("TEXT b: {}", tokenizer.decode(&b.tokens, true).unwrap_or_default());
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
    if args.iter().any(|a| a == "--df2-bake-nvfp4") {
        run_df2_bake(&args, false);
        return;
    }
    if args.iter().any(|a| a == "--df2-bake-fp8") {
        run_df2_bake(&args, true);
        return;
    }
    if args.iter().any(|a| a == "--df2-quant-fidelity") {
        run_df2_quant_fidelity(&args);
        return;
    }
    if args.iter().any(|a| a == "--quantize") {
        run_quantize(&args);
        return;
    }

    // Quantization-error simulator: bake a target format's error INTO an NVFP4 artifact's
    // routed-expert tensors (dequant -> simulated round-trip -> requant NVFP4), so quality is
    // measured on the real serving path before the real kernels exist. All quant tooling is RUST
    // (no Python implementations).   --requant-sim --model-dir <packed-dir> --out <dir> [--scope all|gate_up]
    // (`--requant-q2fake` is the legacy alias, kept so old scripts don't die.)
    if args.iter().any(|a| a == "--requant-sim" || a == "--requant-q2fake") {
        run_requant_sim(&args);
        return;
    }

    // SQ campaign: STQ1_0/ternary-2bit/3-bit values-only probe bake over an NVFP4 artifact
    // (dequant -> SQ round-trip -> requant NVFP4; engine serves the result unmodified).
    // Rust-only per the standing directive; imatrix comes in as DATA (safetensors of per-channel
    // importance).   --stq-bake --model-dir <packed-dir> --out <dir> --arm a|b [--imatrix <file>]
    //                 [--classes gateup,down,attn] [--shard-start N] [--shard-end N] [--limit N]
    //                 [--check]
    if args.iter().any(|a| a == "--stq-bake") {
        run_stq_bake(&args);
        return;
    }

    // Derive a gdn4 (GDN-nvfp4) artifact FROM an existing mixed (GDN-fp8) one, WITHOUT the bf16:
    // copies every non-GDN tensor verbatim and re-quantizes only the fp8 GDN in/out-projs to nvfp4.
    //   --requant-gdn --from <mixed-dir> --out <gdn4-dir>
    if args.iter().any(|a| a == "--requant-gdn") {
        run_requant_gdn(&args);
        return;
    }

    // TP cached-model ops (node-side; pure local file ops on --tp-cache / ~/.cache/gb10_tp).
    // New MODEL-centric names; the old blob-centric names are deprecated aliases (warn once).
    if args.iter().any(|a| a == "--cached-models-list" || a == "--list-model-blobs") {
        let old = args.iter().any(|a| a == "--list-model-blobs");
        let r = if old { gb10_inference::cluster::list_model_blobs() }
                else { gb10_inference::cluster::list_cached_models() };
        if let Err(e) = r { eprintln!("error: {e:#}"); std::process::exit(1); }
        return;
    }
    if let Some(pos) = args.iter().position(|a| a == "--cached-models-remove" || a == "--remove-model-blob") {
        let id = args.get(pos + 1).unwrap_or_else(|| {
            eprintln!("{} requires <model-name|unique-prefix>", args[pos]); std::process::exit(2);
        });
        let old = args.iter().any(|a| a == "--remove-model-blob");
        let r = if old { gb10_inference::cluster::remove_model_blob(id) }
                else { gb10_inference::cluster::remove_cached_model(id) };
        if let Err(e) = r { eprintln!("error: {e:#}"); std::process::exit(1); }
        return;
    }
    if args.iter().any(|a| a == "--cached-models-remove-all" || a == "--clear-model-blobs") {
        let old = args.iter().any(|a| a == "--clear-model-blobs");
        let r = if old { gb10_inference::cluster::clear_model_blobs() }
                else { gb10_inference::cluster::remove_all_cached_models() };
        if let Err(e) = r { eprintln!("error: {e:#}"); std::process::exit(1); }
        return;
    }

    // Perplexity on held-out text — the quality gate for quantization.
    if gb10_inference::opts::on(gb10_inference::opt!("perplexity")) {
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

    // TurboQuant KV kernel validation (E4): the engine's TQ kernels vs the reference goldens
    // (/tmp/tq_ref2/goldens — packed rows, scores_tq/pv_tq, the constants). No model load —
    // a bare CUDA device + the gpu_batch.ptx kernels. Single-node GPU use only.
    if args.iter().any(|a| a == "--probe-tq") {
        run_probe_tq(&args);
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
    if args.iter().any(|a| a == "--bench-tau") {
        run_bench_tau(&args);
        return;
    }
    if args.iter().any(|a| a == "--bench-accept") {
        run_bench_accept(&args);
        return;
    }

    // MTP end-to-end speculative-decoding probe. B30: this used to dispatch ONLY on
    // --bench-mtp, so a bare `--bench-file <path>` (what scripts/r6a/bench_file_leg.sh runs)
    // never reached run_bench_mtp's --bench-file handling and fell through to run_cli — which
    // opened the CWD-relative default model 'qwen3.5-0.8b-packed' and died with a bare
    // "Error: No such file or directory (os error 2)", IDENTICAL for an existing and a
    // nonexistent battery file. Dispatch on either flag.
    if args.iter().any(|a| a == "--bench-mtp" || a == "--bench-file") {
        run_bench_mtp(&args);
        return;
    }

    // B29 gate: exercise the REAL serving path at a PINNED depth -- the exact blind spot that
    // let the recorded serving abort (r6a d2 arm, B29) past every existing gate (the bench/probe
    // paths never touched the server route). Boots this same binary as a server three times
    // (pinned d2, auto, pinned d4), sends ONE greedy chat request to each over HTTP, and asserts:
    //   (a) clean completion (an empty response is a FAIL, not a pass -- AGENTS §3),
    //   (b) MTP actually ENGAGED on the pinned arms (/health telemetry drafts > 0),
    //   (c) byte-identical greedy output across all three arms (pinned depth changes SPEED,
    //       never OUTPUT -- AGENTS §2.7).
    // scripts/r6a/gate_serve_pinned_controls.sh adds the negative controls (boot-fail RED,
    // mid-serve abort RED -- the recorded failure class).
    if args.iter().any(|a| a == "--probe-serve-pinned") {
        let code = run_probe_serve_pinned(&args);
        std::process::exit(code);
    }

    // CLI mode
    run_cli(&args);
}
