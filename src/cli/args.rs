//! Small shared helpers of the binary: argv lookup (`parse_arg`, `required_dir_arg`), the CLI option
//! bridge into the registry, draft-source resolution, TP world parsing, the model loader used by probes and the
//! interactive CLI mode (`run_cli`).
//! Called from `fn main` in `main.rs` and from the sibling mode files in `src/cli/`.

use super::*;

/// Load the serving GPU model. On any load error, print the graceful user-facing message (path +
/// cause + fix — see `GpuModel::load_from_dir_impl`) and exit cleanly with code 1, instead of a
/// `.expect` panic or an OOM/core-dump from garbage checkpoint metadata (owner directive 2026-08-27).
pub(crate) fn load_model_gpu(dir: &str, tp_rank: Option<i32>, world: i32)
    -> (gb10_inference::gpu::GpuModel, gb10_inference::qwen::Config)
{
    let res = match tp_rank {
        Some(r) => gb10_inference::gpu::GpuModel::load_from_dir_tp(dir, r, world),
        None => gb10_inference::gpu::GpuModel::load_from_dir(dir),
    };
    match res {
        Ok(x) => x,
        Err(e) => {
            eprintln!("{e:#}");
            std::process::exit(1);
        }
    }
}

/// CLI-1: the flags whose value WRITES other registry options (the old env bridge, now registry
/// writes — no env anywhere). Every one-to-one flag (`--tp-trace`, `--no-decode-graphs`, `--mxfp4`,
/// `--fp8-prefill`, `--exact-gemm`, `--hc-fp16`, `--splitk-gemm`, `--draft-vocab`, `--tp-cache`, …) is
/// a registry option parsed by `opts::parse_cli`; what is left here maps one user-facing flag onto
/// several internal options. Under TP the head ships the resolved registry (TpConfig.opts), so the
/// node never parses these flags itself.
pub(crate) fn cli_opts_bridge(args: &[String]) {
    use gb10_inference::{opt, opts};
    // EXL27 (§1b-bis 8, owner scope add 2026-10-16): single-file EXL3 packs are served, not refused.
    // Synthesize the one-shard index BEFORE any is_exl3_pack dispatch (server / probes / TP) so
    // every later check sees the standard indexed layout. The pack dir gains exactly one metadata
    // file, atomically (tmp + rename); weight files are untouched.
    if let Some(mp) = parse_arg(args, "--model-dir") {
        if let Ok(qc) = std::fs::read_to_string(format!("{}/quantization_config.json", mp.trim_end_matches('/'))) {
            if qc.contains("exl3") && !gb10_inference::exl3_serve::is_exl3_pack(&mp) {
                match gb10_inference::exl3_serve::ensure_singlefile_index(&mp) {
                    Ok(true) => gb10_inference::rprintln!("[boot] single-file EXL3 pack: synthesized model.safetensors.index.json (1 shard) — serving via the standard indexed path"),
                    Ok(false) => {}
                    Err(e) => gb10_inference::reprintln!("[boot] single-file EXL3 index synthesis failed: {e}"),
                }
            }
        }
    }
    // KV cache format (NVFP4 formats write the kv-* internals; EXL3 formats write kv-cache).
    if let Some(v) = gb10_inference::opt_or_arg(args, "--kv-cache").as_deref() {
        let (q4, tq, k8v4, k8v8) = (opt!("kv-quant"), opt!("kv-tq"), opt!("kv-k8v4"), opt!("kv-k8v8"));
        match v {
            "q4" => { opts::set(q4, "1"); opts::unset(tq); opts::unset(k8v4); }
            "tq" => { opts::set(tq, "1"); opts::unset(q4); opts::unset(k8v4); }
            "tq3" => { opts::set(tq, "3"); opts::unset(q4); opts::unset(k8v4); }
            "k8v4" => { opts::set(k8v4, "1"); opts::unset(q4); opts::unset(tq); opts::unset(k8v8); }
            "k8v8" => { opts::set(k8v8, "1"); opts::unset(q4); opts::unset(tq); opts::unset(k8v4); }
            "bf16" => { opts::unset(q4); opts::unset(tq); opts::unset(k8v4); }
            // EXL3 engine formats (exl3_forward::kv_fmt_from_opts; WP25 q8 = the reference implementation's scheme).
            // This bridge runs for every mode, so without these arms `--kv-cache fp8` on an EXL3
            // server/probe exited here before exl3_serve ever parsed it. An NVFP4 model never reads
            // [kv-cache], so these values are refused there rather than silently serving bf16.
            // q4x = the WP25 kvnll probe's deliberately broken positive control (never served:
            // exl3_serve refuses it)
            "f32" | "f16" | "fp8" | "q8" | "q4x" | "q4k" => {
                if !parse_arg(args, "--model-dir").map_or(false, gb10_inference::exl3_serve::is_exl3_pack) {
                    eprintln!("--kv-cache {v} is an EXL3-pack format (needs an EXL3 --model-dir); NVFP4 formats are bf16|q4|tq|tq3|k8v4|k8v8");
                    std::process::exit(1);
                }
                opts::set(opt!("kv-cache"), v);
            }
            other => { eprintln!("--kv-cache must be bf16|q4|tq|tq3|k8v4|k8v8 (NVFP4) or f32|f16|fp8|q8|q4k (EXL3) (got '{other}')"); std::process::exit(1); }
        }
    }
    // MTP: the bench-path [tp-mtp] / [tp-mtp-depth] internals from the server CLI (`--mtp=on|off`, `--mtp-depth`).
    if let Some(v) = gb10_inference::opt_or_arg(args, "--mtp").as_deref() {
        match v {
            "on" => opts::set(opt!("tp-mtp"), "1"),
            "off" => opts::unset(opt!("tp-mtp")),
            _ => {}   // auto: nothing to translate
        }
    }
    if let Some(d) = parse_arg(args, "--mtp-depth") { opts::set(opt!("tp-mtp-depth"), d); }
    // Mixer sharding: DEFAULT ON under --tp/--head — Hy3 requires it, and it is where TP's speed
    // comes from on every model (halved mixer bytes + halved KV). --no-shard-mixers is the escape.
    if args.iter().any(|a| a == "--no-shard-mixers") {
        opts::unset(opt!("tp-shard-mixers"));
    } else if parse_tp_world(args).is_some() || args.iter().any(|a| a == "--head") {
        opts::set(opt!("tp-shard-mixers"), "1");
    }
}

/// S1 (REL_V0_7_3): the NVFP4-side twin now delegates to the ONE shared `=`-aware parser
/// (`gb10_inference::arg_value`) — body moved there verbatim so both servers parse
/// `--flag value` and `--flag=value` identically.
pub(crate) fn parse_arg<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    gb10_inference::arg_value(args, flag)
}

/// Mandatory directory argument (owner rule 2026-08-23): NO default, NO fallback constant —
/// the user supplies a valid path (`/abs/path`, or `~/...` which the SHELL expands before we
/// see it) or the app STOPS right here. `--node` mode is exempt by design: it loads model
/// shards, whole files AND the DFlash2 draft artifact from its blob cache, never from paths.
pub(crate) fn required_dir_arg(args: &[String], flag: &str, what: &str) -> String {
    let d = match gb10_inference::opt_or_arg(args, flag) {
        Some(v) => v.to_string(),
        None => {
            eprintln!("FATAL: {flag} <dir> is REQUIRED ({what}) — there is no default; pass an explicit path");
            std::process::exit(2);
        }
    };
    if !std::path::Path::new(&d).is_dir() {
        eprintln!("FATAL: {flag} directory does not exist: {d}");
        std::process::exit(2);
    }
    d
}

pub(crate) fn run_cli(args: &[String]) {
    // PREHARD1 [ISSUES 84]: the bare CLI path (no subcommand) used to IGNORE --model-dir —
    // `gb10_inference --model-dir X` fell back to the CWD-relative default 'qwen3.5-0.8b-packed'
    // and died with a bare "No such file or directory (os error 2)" that named neither the path
    // it tried nor the flag it ignored (found while diagnosing the owner's boot failure; the
    // --server path was already correct). Now: --model-dir stands in for --model when --model is
    // absent (a directory of config.json + *.safetensors is exactly what Model::load takes), the
    // tokenizer defaults INSIDE that dir, and a model path that does not exist is ONE clear line.
    let model_path = parse_arg(args, "--model")
        .or_else(|| parse_arg(args, "--model-dir"))
        .unwrap_or("qwen3.5-0.8b-packed");
    let from_model_dir = parse_arg(args, "--model-dir").is_some() && parse_arg(args, "--model").is_none();
    let tokenizer_path: String = match parse_arg(args, "--tokenizer") {
        Some(t) => t.to_string(),
        None => if from_model_dir {
            format!("{}/tokenizer.json", parse_arg(args, "--model-dir").unwrap())
        } else {
            "model/tokenizer.json".to_string()
        },
    };
    if !std::path::Path::new(model_path).exists() {
        eprintln!(
            "model path not found: '{}'{} — pass --model-dir <dir> (a directory with config.json \
             and *.safetensors) or --model <packed-model path>",
            model_path,
            if from_model_dir { " (from --model-dir)" } else { "" }
        );
        std::process::exit(2);
    }
    let prompt_text = parse_arg(args, "--prompt").unwrap_or("The capital of France is");
    let max_seq_len = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse::<usize>().ok()).unwrap_or(4096);
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

/// SERVE-path --draft-dir resolution (owner refinement 2026-08-23): the artifact is MANDATORY
/// only when `--spec-source` EXPLICITLY names a DFlash2 source (dflash2 | dflash2-rq |
/// dflash2-auto) — the user asked for that drafter, so a missing/bad dir stops the app. With
/// any other source (or none given — the default dflash2-auto resolves to the MTP fallback
/// when no artifact is supplied) the flag is OPTIONAL: None = serve via MTP, no hard failure.
/// One-time deprecation notice for a superseded spelling (AGENTS §7: a family-prefixed name lives
/// on only as a back-compat alias that WARNS — silent aliases keep the old spelling alive forever).
fn warn_deprecated_alias(old: &str, new: &str) {
    use std::sync::Once;
    static FLAG: Once = Once::new();
    FLAG.call_once(|| eprintln!("[warn] {old} is deprecated — use {new} (still honored for now)"));
}

/// Does `--spec-source` name a drafter that CANNOT run without an artifact? An explicit drafter
/// source makes `--draft-dir` mandatory (the user asked for that lane: a missing or bad dir stops
/// the app, never a silent MTP downgrade); `dflash2-auto`/`mtp`/`none` may serve without one.
fn draft_source_requires_artifact(src: Option<gb10_inference::batch::SpecSource>) -> bool {
    use gb10_inference::batch::SpecSource as S;
    matches!(src, Some(S::DFlash) | Some(S::DFlash2) | Some(S::DFlash2Rq)
                  | Some(S::DFlash2Tree) | Some(S::Dspark))
}

/// THE drafter-artifact resolver — ONE flag, `--draft-dir <dir>`, for every speculative source
/// (`--spec-source` is the only selector; AGENTS §7 "one knob, one generic name"). Precedence:
/// `--draft-dir` > `--dflash-dir` (deprecated, v1 source only). CLI-1: the GB10_DRAFT_DIR /
/// GB10_DFLASH_DIR env fallbacks are gone (refused at startup, naming --draft-dir).
///
/// The deprecated spelling is P14's first interface for the v1 block drafter and is honored ONLY
/// for that source — a `--dflash-dir` can never silently stand in for a DF2/DSpark artifact.
///
/// A flag that IS provided but points nowhere is always a hard stop, whatever the source.
pub(crate) fn resolve_draft_dir(args: &[String], src: Option<gb10_inference::batch::SpecSource>) -> Option<String> {
    let is_v1 = matches!(src, Some(gb10_inference::batch::SpecSource::DFlash));
    let from_flag = gb10_inference::opt_or_arg(args, "--draft-dir").as_deref().map(str::to_string).or_else(|| {
        if !is_v1 { return None; }
        parse_arg(args, "--dflash-dir").map(|d| {
            warn_deprecated_alias("--dflash-dir", "--draft-dir");
            d.to_string()
        })
    });
    // CLI-1: the old GB10_DRAFT_DIR / GB10_DFLASH_DIR env fallbacks are gone (the startup guard
    // names --draft-dir); the flag is the only source.
    let dir = from_flag;
    match dir {
        Some(d) => {
            if !std::path::Path::new(&d).is_dir() {
                eprintln!("FATAL: drafter artifact directory does not exist: {d}");
                std::process::exit(2);
            }
            Some(d)
        }
        None if draft_source_requires_artifact(src) => {
            eprintln!("FATAL: --spec-source {} requires --draft-dir <artifact dir>; there is no default path",
                      src.map(|s| s.cli_name()).unwrap_or("?"));
            std::process::exit(2);
        }
        None => None,
    }
}

/// Parse the `--tp [N]` flag — the single authority for the TP rank count on a TP run.
/// `--tp N` → N, bare `--tp` → 2, `--tp=N` → N, no `--tp` → None (no TP run). A non-numeric
/// value rejects loudly. `N` is expected to be a power of two (the dynamic-TP ladder).
pub(crate) fn parse_tp_world(args: &[String]) -> Option<u32> {
    let valid = |n: u32| matches!(n, 1 | 2 | 3 | 4 | 6 | 8 | 16);
    let reject = |s: &str| -> u32 {
        eprintln!("--tp must be a supported topology size (1, 2, 3, 4, 6, 8; got '{s}')");
        std::process::exit(1);
    };
    for (i, a) in args.iter().enumerate() {
        if let Some(v) = a.strip_prefix("--tp=") {
            let n = v.parse::<u32>().unwrap_or_else(|_| reject(v));
            if !valid(n) { reject(v); }
            return Some(n);
        }
        if a == "--tp" {
            // An optional value follows only if the next token is not another flag.
            if let Some(next) = args.get(i + 1) {
                if !next.starts_with('-') {
                    let n = next.parse::<u32>().unwrap_or_else(|_| reject(next));
                    if !valid(n) { reject(next); }
                    return Some(n);
                }
            }
            return Some(2);
        }
    }
    None
}

/// S9F: resolve `--spec-source` (default `dflash2-auto`, the S8F routing flip) — ONE parse shared
/// by the serving path and the pre-load TP config build (the shipped source must equal the head's
/// resolved source or the zero-config node takes a different lane branch and the verify
/// all-reduces desync).
pub(super) fn resolve_spec_source(args: &[String]) -> gb10_inference::batch::SpecSource {
    use gb10_inference::batch::SpecSource;
    match parse_arg(args, "--spec-source").map(str::to_lowercase) {
        None => SpecSource::DFlash2Auto,
        Some(s) => match SpecSource::from_cli(&s) {
            Some(src) => src,
            None => { eprintln!("--spec-source must be mtp|dspark|dflash2|dflash2-rq|dflash2-auto|dflash2-tree|none (got {s:?})"); std::process::exit(1); }
        },
    }
}

/// The model-dir's config.json `eos_token_id` (the seed for `QwenTokenizer::stop_token_ids`).
pub(super) fn config_eos(model_dir: &str) -> u32 {
    gb10_inference::qwen::Config::from_config_json(&format!("{model_dir}/config.json"))
        .expect("config.json").eos_token_id
}
