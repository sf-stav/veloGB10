//! Inkling-2 serving: the serve shell skeleton (leg T1). The chat surface is real (the pack's template through the shared tokenizer and
//! jinja, reasoning effort as a name or a number, image splice points, token-level reasoning and tool-call parsing with the engine's
//! catalogue hardening, OpenAI response shapes, `/metrics`); the model behind it is not: `forward::NotBuilt` answers every chat
//! request with a clear error. The integration leg implements `forward::Forward` on the GPU model and passes it to `serve_with`.
//!
//! Flags: the same as the other families', read from the command line (`--model-dir`, `--host`, `--port`, `--max-seq-len`,
//! `--max-batch`, `--max-tokens`, `--prefix-cache`, `--mtp`, `--reasoning-effort`, `--model-name`, `--tp`); no new option.
//! An option this skeleton cannot serve as asked boots anyway on the nearest thing that works, with one notice line
//! (AGENTS 1b-bis item 8): lanes above the cap, `--tp` above 1, `--mtp` other than off.
//!
//! The mode hook: `src/cli/serve.rs` routes an Inkling-2 pack here (`is_inkling2_pack`) BEFORE `exl3_serve::is_exl3_pack`, which would
//! otherwise claim an EXL3 Inkling-2 pack for the Flash-Next trunk.
//!
//! Map: `forward.rs` the seam to the model, `output.rs` token ids to OpenAI shapes, `http.rs` the router, `metrics.rs` the `/metrics` rows.

pub mod forward;
pub mod http;
pub mod metrics;
pub mod output;
#[cfg(test)]
mod tests;

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};

use self::forward::{Forward, NotBuilt};
use super::config::Inkling2Config;
use crate::tokenizer::QwenTokenizer;

/// Default concurrent lanes when `--max-batch` is omitted (THE POSTURE: 4 users), and the ceiling this shell accepts.
pub const DEFAULT_LANES: usize = 4;
pub const MAX_LANES: usize = 16;
/// The context served when `--max-seq-len` is omitted (THE POSTURE), capped by the model's trained window.
pub const DEFAULT_CTX: usize = 262_144;

pub fn is_inkling2_pack(dir: &str) -> bool { Inkling2Config::is_inkling2_dir(Path::new(dir)) }

/// `--prefix-cache`: ON unless the operator says otherwise.
pub fn prefix_cache_on(v: Option<&str>) -> bool { !matches!(v.map(|s| s.trim().to_ascii_lowercase()).as_deref(), Some("off" | "0" | "false" | "no")) }

/// The trained window: `max_position_embeddings` at the top of `config.json` or under `text_config`.
pub fn trained_window(dir: &Path) -> Option<usize> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
    v.get("max_position_embeddings").or_else(|| v.pointer("/text_config/max_position_embeddings")).and_then(|x| x.as_u64()).map(|x| x as usize)
}

/// Sampler defaults from the pack's `generation_config.json` (temperature 1.0 on the stock pack; top_p 1.0 / top_k 0 when absent).
pub fn default_sampling(dir: &Path) -> (f32, f32, usize) {
    let v: Option<serde_json::Value> = std::fs::read_to_string(dir.join("generation_config.json")).ok().and_then(|t| serde_json::from_str(&t).ok());
    let g = |k: &str| v.as_ref().and_then(|v| v.get(k)).and_then(|x| x.as_f64());
    (g("temperature").unwrap_or(1.0) as f32, g("top_p").unwrap_or(1.0) as f32, g("top_k").unwrap_or(0.0) as usize)
}

pub struct Setup {
    pub settings: http::Settings,
    pub lanes: usize,
    pub lanes_requested: usize,
    pub prefix_on: bool,
    pub port: u16,
    pub mtp: String,
    pub tp: usize,
    /// one line per option that was mapped to something that works (printed by `run` as notices)
    pub notices: Vec<String>,
}

pub fn setup(args: &[String], model_dir: &str) -> Result<Setup> {
    let arg = |f: &str| crate::arg_value(args, f).map(str::to_string);
    let dir = Path::new(model_dir);
    let mut notices = Vec::new();
    let lanes_requested: usize = arg("--max-batch").map(|s| s.parse()).transpose().context("--max-batch")?.unwrap_or(DEFAULT_LANES);
    let lanes = lanes_requested.clamp(1, MAX_LANES);
    if lanes != lanes_requested {
        notices.push(format!("--max-batch {lanes_requested}: the Inkling-2 server runs at most {MAX_LANES} requests at once; booting with {lanes} lanes, further requests wait in the queue (none is rejected)"));
    }
    let window = trained_window(dir).unwrap_or(DEFAULT_CTX);
    let asked_ctx: usize = arg("--max-seq-len").map(|s| s.parse()).transpose().context("--max-seq-len")?.unwrap_or(DEFAULT_CTX);
    let max_seq_len = asked_ctx.min(window);
    if max_seq_len != asked_ctx { notices.push(format!("--max-seq-len {asked_ctx}: the model's trained window is {window}; serving {max_seq_len}")); }
    let tp_asked: usize = arg("--tp").and_then(|s| s.parse().ok()).unwrap_or(1);
    // S-INK2-TP2 + Velo-Fabric TP3/TP6/TP8: TP1 / TP2 / TP3 / TP4 / TP6 / TP8; another count boots on the nearest one below it (AGENTS 1b-bis item 8)
    let tp = match tp_asked { 0 | 1 => 1, 2 => 2, 3 => 3, 4 | 5 => 4, 6 | 7 => 6, _ => 8 };
    if tp != tp_asked.max(1) { notices.push(format!("--tp {tp_asked}: the Inkling-2 model splits over 1, 2, 3, 4, 6 or 8 GB10s (24 query / 8 KV heads, 1152 MoE channels); serving TP={tp}")); }
    let mtp = crate::opts::lookup("mtp").and_then(crate::opts::get).or_else(|| arg("--mtp")).unwrap_or_else(|| "auto".into());
    if !matches!(mtp.as_str(), "off" | "0" | "false" | "none") && !dir.join("mtp.safetensors").exists() {
        notices.push(format!("--mtp {mtp}: this pack has no mtp.safetensors (the bundled drafter); decoding is plain (no speculation)"));
    }
    let (temperature, top_p, top_k) = default_sampling(dir);
    let settings = http::Settings {
        model_name: crate::server::resolve_model_name(arg("--model-name").as_deref(), arg("--served-model-name").as_deref(), model_dir),
        max_seq_len,
        default_max_tokens: arg("--max-tokens").and_then(|s| s.parse().ok()),
        temperature, top_p, top_k,
        effort: arg("--reasoning-effort"),
        image_max_tokens: arg("--image-max-tokens").and_then(|s| s.parse().ok()).filter(|&n: &usize| n > 0).unwrap_or(crate::inkling2::vision::preproc::DEFAULT_MAX_TOKENS),
    };
    if arg("--image-max-edge").is_some() {
        notices.push("--image-max-edge: the Inkling-2 server does not resize by edge (its reference preprocessing has no resize); it caps the LM tokens per image with --image-max-tokens instead".into());
    }
    Ok(Setup { settings, lanes, lanes_requested, prefix_on: prefix_cache_on(arg("--prefix-cache").as_deref()), port: arg("--port").and_then(|s| s.parse().ok()).unwrap_or(9000), mtp, tp, notices })
}

/// Serve `app` on `listener` until the server stops. (Tests call this with their own listener.)
pub async fn serve_on(listener: tokio::net::TcpListener, app: Arc<http::App>) -> Result<()> {
    axum::serve(listener, http::router(app)).tcp_nodelay(true).await.context("axum serve")
}

/// The entry of `--server --model-dir <Inkling-2 pack>` (one dispatch line in `src/cli/serve.rs`): the skeleton, no model.
pub fn run(args: &[String], model_dir: &str) -> Result<()> {
    let su = setup(args, model_dir)?;
    // S3: the stock (NVFP4 / MXFP8) pack runs the GPU model; any other pack format still gets the skeleton's clear error.
    let pack_ok = super::pack::Inkling2Pack::open(Path::new(model_dir)).is_ok();
    if !pack_ok {
        crate::boot_report::notice("this Inkling-2 pack is not the stock NVFP4/MXFP8 layout the GPU model reads (EXL3 packs come in a later stage): serving the chat surface with the not-built error");
        return serve_with(args, model_dir, Arc::new(NotBuilt));
    }
    let tc = args.iter().any(|a| a == "--inkling2-decode-attn-tc");
    let act = crate::inkling2::actq::set_mode_from_args(args)?;
    if act == crate::inkling2::actq::ExpertAct::Native { metrics::set(&metrics::EXPERT_ACT_NATIVE, 1); }
    if crate::inkling2::model::set_ulysses_wire_from_args(args)? {
        metrics::set(&metrics::ULYSSES_WIRE_BF16, 1);
        crate::boot_report::notice("--inkling2-ulysses-wire bf16: DeepSpeed-Ulysses SP attention ships Q/K/V and ungated attention outputs in BF16 (r/s/kd/vd stay FP32; -47% Ulysses wire, deterministic cross-rank)");
    }
    // S7 opt-in: single-pass f16 prefill attention (changes outputs; default off: the exact two-pass kernel)
    let sp = args.iter().any(|a| a == "--inkling2-attn-single-pass");
    if sp {
        crate::boot_report::notice("--inkling2-attn-single-pass: prefill chunks use the single-pass f16 attention (Q and P rounded once to f16: faster long-context prefill, outputs differ slightly from the exact default)");
    }
    let spec = spec_opts(args, &su.mtp, Path::new(model_dir));
    let fwd = if su.tp >= 2 {
        // S-INK2-TP2: this box is the head (rank 0); the nodes run `--node` (zero-config)
        let boot = crate::inkling2::tp::launch::boot_head(args, Path::new(model_dir), su.tp)?;
        crate::boot_report::notice(&format!("inkling2 TP={}: {}", boot.world, boot.describe));
        crate::inkling2::model::serve::GpuForward::start_model(boot.model, Path::new(model_dir), su.lanes, su.settings.max_seq_len, tc, su.prefix_on, sp, spec, su.settings.image_max_tokens)?
    } else {
        crate::inkling2::model::serve::GpuForward::start(Path::new(model_dir), su.lanes, su.settings.max_seq_len, tc, su.prefix_on, sp, spec, su.settings.image_max_tokens)?
    };
    serve_with(args, model_dir, Arc::new(fwd))
}

/// S6: speculation options from `--mtp` (auto / on: the bundled drafter when the pack has `mtp.safetensors`; off: plain)
/// and `--inkling2-mtp-max-drafts N` (1..7, default 7).
pub fn spec_opts(args: &[String], mtp: &str, dir: &Path) -> Option<crate::inkling2::mtp::serve::SpecOpts> {
    if matches!(mtp, "off" | "0" | "false" | "none") || !dir.join("mtp.safetensors").exists() { return None; }
    let md = crate::arg_value(args, "--inkling2-mtp-max-drafts").and_then(|s| s.parse::<usize>().ok()).unwrap_or(7).clamp(1, 7);
    Some(crate::inkling2::mtp::serve::SpecOpts { force: matches!(mtp, "on" | "1" | "true"), max_drafts: md })
}

/// The same with a forward plugged in (the integration leg's entry).
pub fn serve_with(args: &[String], model_dir: &str, forward: Arc<dyn Forward>) -> Result<()> {
    let dir = Path::new(model_dir);
    let tok = Arc::new(QwenTokenizer::from_file(&dir.join("tokenizer.json").to_string_lossy()).context("tokenizer.json")?);
    let su = setup(args, model_dir)?;
    for n in &su.notices { crate::boot_report::notice(n); }
    if forward.name() == "not-built" {
        crate::boot_report::notice("the Inkling-2 forward is not built yet: this server answers the chat surface (template, reasoning and tool-call parsing, /metrics) but every chat request gets a clear error until the model is plugged in");
        metrics::set(&metrics::NOT_BUILT, 1);
    }
    if tok.template_meta.is_none() {
        crate::boot_report::notice("the pack's chat template did not compile: requests would render with the legacy fallback, which is wrong for this model; fix the template (see the WARNING above)");
    }
    metrics::mark_start();
    metrics::set(&metrics::SERVING, 1);
    metrics::set(&metrics::LANES, su.lanes as u64);
    metrics::set(&metrics::LANES_REQUESTED, su.lanes_requested as u64);
    metrics::set(&metrics::PREFIX_ON, su.prefix_on as u64);
    crate::metrics::set_max_batch_requested(su.lanes_requested);
    let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64);
    let (lanes, ctx, model, port, prefix_on, mtp, tp) = (su.lanes, su.settings.max_seq_len, su.settings.model_name.clone(), su.port, su.prefix_on, su.mtp.clone(), su.tp);
    let app = Arc::new(http::App { settings: su.settings, tok, forward, created });
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().context("tokio runtime")?;
    rt.block_on(async move {
        let listener = crate::server::http_listen(port).await;
        let url = listener.local_addr().map_or_else(|_| format!("http://0.0.0.0:{port}"), |a| format!("http://{a}"));
        crate::boot_report::serving(&crate::boot_report::ServingLine {
            family: "inkling2", model: &model, tp: tp.max(1) as u32, ctx, batch: lanes, kv: "none", prefix: if prefix_on { "on" } else { "off" },
            mtp: if spec_opts(args, &mtp, dir).is_some() { "on" } else if mtp == "off" { "off" } else { "none" },
            spec: if spec_opts(args, &mtp, dir).is_some() { "dspark (bundled drafter)" } else { "plain" }, port, url,
        });
        serve_on(listener, app).await
    })
}
