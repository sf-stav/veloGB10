/// CLI-1: the options registry (every engine option is a command-line flag; AGENTS §7).
pub mod opts;
pub mod hf_cache;
pub mod self_update; // --update + boot version check (owner 2026-11-13) // issue #33: --model-dir accepts a HF repo id (local snapshot resolution, no network)
pub mod memory;
pub mod model;
pub mod qwen;
pub mod vision_preproc;
pub mod vision_tower;
pub mod vision_encoder;
pub mod vision_gpu;
pub mod gpu;
pub mod ptx_embed; // EMB1: every PTX module embedded at compile time (one-file deploy)
/// S-B9-REL-PPL: alignment-safe reads of file-derived bytes (safetensors offsets are not
/// guaranteed aligned; a bare `bytemuck::cast_slice` panics on them).
pub mod podread;
/// S1 (REL_V0_7_3 review): THE one `=`-aware command-line flag reader, shared by BOTH
/// servers' hand-rolled arg loops (EXL3's `arg()` and the NVFP4 path's `parse_arg()`).
/// Accepts `--flag value` AND `--flag=value`, exactly like the option registry
/// (opts::parse) — a server-side reader that matched only the space form silently IGNORED
/// `--lane-order=fcfs` / `--model-name=X` / `--otel-endpoint=...` while `--print-config`
/// showed them as set.
/// CLIPRUNE1 (one flag = one registry row): single-parse read for the mixed call sites.
/// A flag WITH a registry row is read from the store `opts::parse_cli` filled (main calls it
/// before any mode dispatch), so the registry is the one parse path; a flag WITHOUT a row keeps
/// the shared `=`-aware argv scan below (hand flags unchanged).
pub fn opt_or_arg(args: &[String], name: &str) -> Option<String> {
    match opts::lookup(name.trim_start_matches("--")) {
        // Internal rows are never CLI-parsed (another flag's writer sets them), so the
        // store holds nothing at read time: keep reading argv for those.
        Some(id) if !matches!(id.def().ty, opts::Ty::Internal(_)) => opts::var(id).ok(),
        _ => arg_value(args, name).map(|s| s.to_string()),
    }
}

pub fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag || a.starts_with(&format!("{flag}=")))
        .and_then(|i| {
            let arg = &args[i];
            if let Some(val) = arg.strip_prefix(&format!("{flag}=")) {
                Some(val)
            } else {
                args.get(i + 1).map(|s| s.as_str())
            }
        })
}

pub mod quant;
pub mod mxfp4;
pub mod exl3;
pub mod naive; // NAIVE: Naive-N0.5-Flash family (PLAN/NAIVE_N05_PORT.md) — own tree, copies + one-line dispatches
pub mod shard_plan;
pub mod exl3_bench;
pub mod exl3_forward;
pub mod exl3_ngram_embed;
pub mod fp4l; // FP4L: nvidia/Qwen3.8-Flash-Next-FP4-Local (ModelOpt MXFP8 + NVFP4 csf) on the EXL3 trunk
// TUNE (PLAN/AUTOTUNE_DESIGN.md): tunable registry + frozen table (re-exported as exl3_forward::tune)
pub mod exl3_tune;
pub mod exl3_serve;
pub mod exl3_wp27;
pub mod logq;
pub mod loop_detect;
pub mod startup; // BASICS (owner 2026-11-02): line 1 = app name + version (--version), then the run's configuration
pub mod boot_report; // OUT1: one vocabulary for every family's boot output (stages, notices, the SERVING line)
pub mod wp24; // WP24: real-q speculative sampling (host replicas, dump, cross-check, gate)
pub mod batch;
pub mod tel;
pub mod dispatch_assert;
pub mod sampler;
pub mod think_budget;
pub mod tool_trace;
pub mod tool_scrub;
pub mod tools;
pub mod kv_cache;
pub mod engine;
pub mod tokenizer;
pub mod server;
pub mod metrics;
pub mod mtp_reprobe;
pub mod lane_pack; // S-B9-CF-PACK4: --lane-pack (pure decision; the arm lives in exl3_serve)
pub mod lane_reserve; // LANERES1: --lane-reserve (pure admission policy; the arm lives in exl3_serve)
pub mod kv_spill; // LANERES2: --kv-spill (pure spill policy + disk store; the arm lives in exl3_serve/exl3_forward)
pub mod prefix_pool; // PFXPOOL: prompt states kept in memory across lanes (policy; the arm lives in exl3_serve/exl3_forward)
pub mod otel;
pub mod net;
// HOST / RO-7: big-core detection + pinning of the serving engine's critical host threads.
pub mod cpu_affinity;
pub mod host_wait; // CPUIDLE: --gpu-wait spin|hybrid (how the host waits for the GPU)
pub mod pp;
pub mod cluster;
pub mod tp;
pub mod tp_serve;
pub mod tp_lockstep; // TP-4D: world>2 head-hub lockstep frames + verdicts (pure; tested over net::hub_mock)
pub mod tp_bench;
pub mod tp_xport;
pub mod tp_vpgather;
pub mod dsv4_load;
pub mod dsv4_cpu;
pub mod dsv4_moe;
pub mod dsv4_gpu;
pub mod dsv4_attn;
pub mod dsv4_comp;
pub mod dsv4_graph;
pub mod dsv4_model;
pub mod dsv4_convert;
pub mod dsv4_chat;
pub mod dsv4_sse;
pub mod dsv4_dspark;
pub mod dflash;
pub mod dspark;
pub mod dflash2;
pub mod json_schema;
// PR veloGB10#4 (qwen4_exp): new modules. gptq is DEFERRED — it needs the
// gpu.rs GptqTap/GptqHess/IGS surface that only exists after the gpu.rs merge.
pub mod ple;
pub mod memwatch;
pub mod w4a4;

use serde::Serialize;

/// Standard OpenAI usage object.
#[derive(Serialize, Clone, Debug)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// v0.7.3 (issue #8.2): OpenAI-compatible prompt-usage detail.
    pub prompt_tokens_details: PromptTokensDetails,
}

/// The `usage.prompt_tokens_details` object (OpenAI / vLLM convention).
/// `cached_tokens` counts prompt tokens served from the prefix cache (prefill
/// skipped); 0 on a fresh prefill or an engine without a prefix cache.
#[derive(Serialize, Clone, Debug, Default)]
pub struct PromptTokensDetails {
    pub cached_tokens: usize,
}

/// llama.cpp-compatible timing block, emitted as a top-level extension next to `usage`.
/// `prompt_*` spans request-submit -> first token (TTFT); `predicted_*` spans first-token -> end.
#[derive(Serialize, Clone, Debug)]
pub struct Timings {
    pub prompt_ms: f64,
    pub predicted_ms: f64,
    pub prompt_per_second: f64,
    pub predicted_per_second: f64,
}

/// Compute timings from request-start to now, using optional first-token time.
pub fn make_timings(
    t0: std::time::Instant,
    first_tok: Option<std::time::Instant>,
    prompt_len: usize,
    n: usize,
) -> Timings {
    let end = std::time::Instant::now();
    let (prompt_ms, predicted_ms) = match first_tok {
        Some(ft) => (
            ft.duration_since(t0).as_secs_f64() * 1e3,
            end.duration_since(ft).as_secs_f64() * 1e3,
        ),
        None => (end.duration_since(t0).as_secs_f64() * 1e3, 0.0),
    };
    Timings {
        prompt_ms,
        predicted_ms,
        prompt_per_second: if prompt_ms > 0.0 { prompt_len as f64 * 1e3 / prompt_ms } else { 0.0 },
        predicted_per_second: if predicted_ms > 0.0 { n as f64 * 1e3 / predicted_ms } else { 0.0 },
    }
}

/// The drafter-artifact directory the binary resolved (`--draft-dir`, written into the internal
/// [draft-dir] option by the paths that need it after `resolve_draft_dir`; CLI-1: no env alias).
pub fn draft_dir_env() -> Option<String> {
    crate::opts::var(crate::opt!("draft-dir")).ok()
}

/// S-B9-REL-SMALLFIXES item 4: `scripts/run_tp_server.sh` derives `--tp` from the number of
/// `--node` flags (it used to default to 2 while documenting "must equal 1 + number of --node",
/// unchecked: three `--node` started a TP=2 engine). The REAL script, run with `--dry-run` from a
/// scratch deployed dir (a stub engine next to it, a stub model dir).
#[cfg(test)]
mod tp_launcher_tests {
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;

    const PEERS: [&str; 3] = ["192.0.2.12:29500", "192.0.2.13:29500", "192.0.2.14:29500"];

    /// (exit code, stdout, stderr) of `run_tp_server.sh --model-dir model <--node x n_nodes> <extra> --dry-run`
    fn dry_run(tag: &str, n_nodes: usize, extra: &[&str]) -> (i32, String, String) {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let dir = std::env::temp_dir().join(format!("gb10_tp_launcher_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("model")).unwrap();
        std::fs::copy(root.join("scripts/run_tp_server.sh"), dir.join("run_tp_server.sh")).unwrap();
        std::fs::write(dir.join("model/config.json"), "{}").unwrap();
        let stub = dir.join("gb10_inference");
        std::fs::write(&stub, "#!/bin/sh\necho STUB_ENGINE_RAN \"$@\"\n").unwrap();
        std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut c = Command::new("bash");
        c.arg(dir.join("run_tp_server.sh")).args(["--model-dir", "model"]);
        for p in &PEERS[..n_nodes] { c.args(["--node", p]); }
        let out = c.args(extra).arg("--dry-run").output().expect("run bash");
        let _ = std::fs::remove_dir_all(&dir);
        (out.status.code().unwrap_or(-1), String::from_utf8_lossy(&out.stdout).into_owned(),
         String::from_utf8_lossy(&out.stderr).into_owned())
    }

    #[test]
    fn tp_defaults_to_one_plus_the_node_count() {
        let (code, out, err) = dry_run("n1", 1, &[]);
        assert_eq!(code, 0, "one --node: {err}");
        assert!(out.contains("--tp 2 --nodes 192.0.2.12:29500"), "one --node must give --tp 2: {out}");
        // the first v0.7.3 packet's TP=4 command: three --node and no --tp
        let (code, out, err) = dry_run("n3", 3, &[]);
        assert_eq!(code, 0, "three --node: {err}");
        assert!(out.contains("--tp 4 --nodes"), "three --node must give --tp 4 (was --tp 2): {out}");
        assert!(!out.contains("--tp 2"), "{out}");
        assert!(!out.contains("STUB_ENGINE_RAN"), "--dry-run must not start the engine: {out}");
    }

    #[test]
    fn an_explicit_tp_must_match_the_node_count() {
        let (code, out, err) = dry_run("ok", 3, &["--tp", "4"]);
        assert_eq!(code, 0, "an explicit matching --tp is accepted: {err}");
        assert!(out.contains("--tp 4 --nodes"), "{out}");
        let (code, out, err) = dry_run("bad", 3, &["--tp", "2"]);
        assert_eq!(code, 2, "a --tp that disagrees with the --node count is exit 2 (stdout {out:?})");
        assert!(err.contains("--tp 2 does not match the 3 peer(s) given with --node") && err.contains("--tp 4"), "{err}");
        assert!(out.is_empty(), "a refused command prints no engine line: {out}");
        let (code, _, err) = dry_run("nan", 1, &["--tp", "two"]);
        assert_eq!(code, 2);
        assert!(err.contains("--tp needs a whole number"), "{err}");
    }

    #[test]
    fn a_world_the_engine_refuses_is_refused_by_the_launcher() {
        // two peers = world 3: the engine takes a power of two only, and would blame a --tp the
        // user never typed
        let (code, out, err) = dry_run("n2", 2, &[]);
        assert_eq!(code, 2, "stdout {out:?}");
        assert!(err.contains("2 peer(s) given with --node make a world of 3") && err.contains("power-of-two"), "{err}");
    }

    /// S-B9-REL-SURFACE2 item 3: one `--node` holding a comma list is that many PEERS (it used to
    /// count as one peer: `--node a,b,c` started `--tp 2 --nodes a,b,c`), and `--tp 02` no longer
    /// reaches the engine. The full case list is the shell test, which this runs so that
    /// `cargo test` covers it.
    #[test]
    fn a_comma_list_in_one_node_flag_is_that_many_peers() {
        let list = PEERS.join(",");
        let (code, out, err) = dry_run("csv", 0, &["--node", &list]);
        assert_eq!(code, 0, "{err}");
        assert!(out.replace("\\,", ",").contains(&format!("--tp 4 --nodes {list}")), "three peers in one flag = TP=4: {out}");
        let (code, _, err) = dry_run("csv_bad", 0, &["--node", "192.0.2.12:29500,,192.0.2.13:29500"]);
        assert_eq!(code, 2);
        assert!(err.contains("has an empty peer"), "{err}");
        let (code, out, err) = dry_run("tp02", 1, &["--tp", "02"]);
        assert_eq!(code, 2, "stdout {out:?}");
        assert!(err.contains("without leading zeros (got '02')"), "{err}");
        // The shell test is a file of the development tree. A tree that carries the launchers
        // without it (the public sync copies the two launchers only: RELEASE_PROTOCOLS.md A1)
        // cannot run its 27 checks: say so where nobody can miss it, not in captured output,
        // and do not fail on a file that was never shipped. The three checks above ran.
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/test_run_tp_server.sh");
        if !script.is_file() {
            use std::io::Write;
            let _ = writeln!(std::io::stderr(), "\n*** SKIPPED: scripts/test_run_tp_server.sh is not in this tree, so the launcher's \
                                                 27-case shell test did NOT run (3 launcher checks did). ***\n");
            return;
        }
        let sh = Command::new("bash").arg(&script).output().expect("run bash");
        let text = String::from_utf8_lossy(&sh.stdout);
        assert!(sh.status.success() && text.contains("RUN_TP_SERVER TEST PASS") && !text.contains("FAIL "), "{text}");
    }

    #[test]
    fn help_prints_the_header_block_only() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        // --help needs a binary next to the script too: reuse the dry-run scaffold through a tiny copy
        let dir = std::env::temp_dir().join(format!("gb10_tp_launcher_{}_help", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(root.join("scripts/run_tp_server.sh"), dir.join("run_tp_server.sh")).unwrap();
        let stub = dir.join("gb10_inference");
        std::fs::write(&stub, "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();
        let out = Command::new("bash").arg(dir.join("run_tp_server.sh")).arg("--help").output().expect("run bash");
        let _ = std::fs::remove_dir_all(&dir);
        let text = String::from_utf8_lossy(&out.stdout);
        assert_eq!(out.status.code(), Some(0));
        assert!(text.contains("default: 1 + the number of peers given"), "{text}");
        assert!(text.contains("separated by commas"), "the comma form is documented: {text}");
        assert!(text.trim_end().ends_with("--max-tokens 4096)"), "--help must end with the header block: {text}");
    }
}
