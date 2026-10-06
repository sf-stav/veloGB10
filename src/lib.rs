/// CLI-1: the options registry (every engine option is a command-line flag; AGENTS §7).
pub mod opts;
pub mod memory;
pub mod model;
pub mod qwen;
pub mod vision_preproc;
pub mod vision_tower;
pub mod vision_encoder;
pub mod vision_gpu;
pub mod gpu;
/// S1 (REL_V0_7_3 review): THE one `=`-aware command-line flag reader, shared by BOTH
/// servers' hand-rolled arg loops (EXL3's `arg()` and the NVFP4 path's `parse_arg()`).
/// Accepts `--flag value` AND `--flag=value`, exactly like the option registry
/// (opts::parse) — a server-side reader that matched only the space form silently IGNORED
/// `--lane-order=fcfs` / `--model-name=X` / `--otel-endpoint=...` while `--print-config`
/// showed them as set.
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
pub mod shard_plan;
pub mod exl3_bench;
pub mod exl3_forward;
// TUNE (PLAN/AUTOTUNE_DESIGN.md): tunable registry + frozen table (re-exported as exl3_forward::tune)
pub mod exl3_tune;
pub mod exl3_serve;
pub mod exl3_wp27;
pub mod logq;
pub mod loop_detect;
pub mod wp24; // WP24: real-q speculative sampling (host replicas, dump, cross-check, gate)
pub mod batch;
pub mod tel;
pub mod dispatch_assert;
pub mod kernels;
pub mod sampler;
pub mod tools;
pub mod kv_cache;
pub mod engine;
pub mod tokenizer;
pub mod server;
pub mod metrics;
pub mod otel;
pub mod net;
// HOST / RO-7: big-core detection + pinning of the serving engine's critical host threads.
pub mod cpu_affinity;
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
