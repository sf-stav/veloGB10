pub mod memory;
pub mod model;
pub mod qwen;
pub mod vision_preproc;
pub mod vision_tower;
pub mod vision_encoder;
pub mod vision_gpu;
pub mod gpu;
pub mod quant;
pub mod ple;
pub mod memwatch;
pub mod gptq;
pub mod mxfp4;
pub mod batch;
pub mod kernels;
pub mod sampler;
pub mod tools;
pub mod kv_cache;
pub mod engine;
pub mod tokenizer;
pub mod server;
pub mod net;
pub mod pp;
pub mod cluster;
pub mod tp;
pub mod tp_serve;
pub mod tp_bench;
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
pub mod w4a4;

use serde::Serialize;

/// Standard OpenAI usage object.
#[derive(Serialize, Clone, Debug)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
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

/// Resolve a generic (family-agnostic) env knob, honoring a deprecated family-prefixed alias.
/// AGENTS.md §7: one generic name per knob (`GB10_*` / `RUST_INFER_*`), honored by all families;
/// family-prefixed names (`DSV4_*`) survive only as documented back-compat aliases. The generic
/// name wins when both are set; a set alias logs a one-time-per-process deprecation warning.
pub fn env_knob(generic: &'static str, deprecated_alias: &'static str) -> Option<String> {
    if let Ok(v) = std::env::var(generic) {
        return Some(v);
    }
    if let Ok(v) = std::env::var(deprecated_alias) {
        static WARNED: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<&'static str>>> =
            std::sync::OnceLock::new();
        let warned = WARNED.get_or_init(Default::default);
        if warned.lock().unwrap().insert(deprecated_alias) {
            eprintln!("[deprecated] env {deprecated_alias} is a back-compat alias — use {generic} (AGENTS.md §7)");
        }
        return Some(v);
    }
    None
}
