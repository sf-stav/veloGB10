//! DSV4.1 serving: the serve shell (leg SV). The engine loop on the shared `server` / `metrics` / `logq` /
//! `boot_report` / `tokenizer` code, lanes 1-4, the sampler at the model's defaults, prefix state, `/metrics` +
//! `owner_check.py` rows. Parity with the other families (PLAN/DSV41_ADDENDUM_UX_2026-11-18.md) comes from serving
//! THROUGH the shared modules, never around them: `startup` prints the configuration block (the dsv41 rows come from
//! `crate::dsv41::boot::config_rows`), `boot_report` the notices and the one SERVING line (family=dsv41), `logq` the
//! per-request lines, `metrics` the generic rows (ours are appended).
//!
//! The mode hook: `src/cli/serve.rs` routes a dsv41 pack here (`is_dsv41_pack`) BEFORE `exl3_serve::is_exl3_pack` can
//! claim an EXL3 dsv41 pack for the Flash-Next trunk.
//!
//! Map of this directory:
//! - `mod.rs` (this file): the entry (`run`), the flags read at boot (`setup`), the routing predicate.
//! - `forward.rs`: the `Forward` trait the lane loop drives, and the CPU `StubForward`.
//! - `engine.rs`: the lane loop (admission, chunked prefill, the shared decode step, sink handling, prefix snapshots).
//! - `sampler.rs`: the host sampler (DeepSeek's defaults, joint top-p/top-k, counter RNG, penalties, seedless seed).
//! - `sink.rs`: opt-in attention-sink detection and draft-zero rejection (`--dsv41-sink-reject`).
//! - `prefix.rs`: the prefix pool and hit policy (128-token blocks, cap `L - max(keep, 134)`, LRU under a byte budget).
//! - `api.rs`: one chat request to a prompt and a job (ENC's encoder, tools, thinking, sampler fields).
//! - `output.rs`: the reply's bytes through ENC's DSML parser and the stop strings; OpenAI JSON builders.
//! - `http.rs`: the router (`/v1/chat/completions`, `/v1/completions`, `/v1/models[/id]`, `/v1/tokenize`, `/v1/detokenize`, `/health`, `/metrics`) and the SSE writer.
//! - `metrics.rs`: the `velogb10_dsv41_*` rows.
//! - `tests.rs`: the end-to-end tests (a real HTTP server on a socket, the stub forward).

pub mod api;
pub mod engine;
pub mod forward;
pub mod http;
pub mod metrics;
pub mod output;
pub mod prefix;
pub mod sampler;
pub mod sink;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod rounds_tests;
#[cfg(test)]
mod resume_tests;

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};

use crate::dsv41::chat::special::{stop_token_ids, THINK_END_ID, THINK_START_ID};
use crate::dsv41::config::Dsv41Config;
use crate::opts;
use crate::tokenizer::QwenTokenizer;

use self::api::{effort_from_flag, ServeSettings};
use self::engine::EngineCfg;
use self::forward::{Forward, StubForward};
use self::sampler::Sampling;
use self::sink::SinkCfg;

/// Default concurrent lanes when `--max-batch` is omitted (`THE POSTURE` 4-user default), and the multi-user ceiling (`--max-batch 8`).
pub const DEFAULT_LANES: usize = 4;
pub const MAX_LANES: usize = 8;

/// Is `dir` a DeepSeek-V4.1 pack of ANY weight format (native or EXL3)? The routing predicate: it must be asked BEFORE
/// `exl3_serve::is_exl3_pack`, which would otherwise claim an EXL3 dsv41 pack for the Flash-Next trunk.
pub fn is_dsv41_pack(dir: &str) -> bool {
    if crate::dsv41::config::is_dsv41_dir(dir) { return true; }
    std::fs::read_to_string(Path::new(dir).join("config.json")).ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .map_or(false, |v| {
            let mt = |x: Option<&serde_json::Value>| x.and_then(|m| m.as_str()).map_or(false, |m| m == "deepseek_v41" || m == "deepseek_v41_text");
            mt(v.get("model_type")) || mt(v.get("text_config").and_then(|t| t.get("model_type")))
        })
}

fn opt_str(name: &str) -> Option<String> { opts::lookup(name).and_then(opts::get) }
fn opt_ranged(name: &str, default: i64) -> i64 { opts::lookup(name).and_then(|o| opts::ranged(o).ok()).unwrap_or(default) }
fn opt_on(name: &str) -> bool { opts::lookup(name).map_or(false, opts::on) }

/// `--prefix-cache`: ON unless the operator says otherwise (D-DSV41-2 (3): the default for DeepSeek-V4.1, as for the EXL3 server).
/// `off` / `0` / `false` / `no` turn it off; `on` / `1` / `true` and an absent flag leave it on.
pub fn prefix_cache_on(v: Option<&str>) -> bool {
    !matches!(v.map(|s| s.trim().to_ascii_lowercase()).as_deref(), Some("off" | "0" | "false" | "no"))
}

/// The sampler defaults: the pack's `generation_config.json` (temperature / top_p / top_k) when it has them, else DeepSeek's 1.0 / 1.0.
pub fn default_sampling(dir: &Path) -> Sampling {
    let mut s = Sampling::default();
    if let Some(v) = std::fs::read_to_string(dir.join("generation_config.json")).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()) {
        if let Some(t) = v.get("temperature").and_then(|x| x.as_f64()) { s.temperature = t as f32; }
        if let Some(p) = v.get("top_p").and_then(|x| x.as_f64()) { s.top_p = p as f32; }
        if let Some(k) = v.get("top_k").and_then(|x| x.as_u64()) { s.top_k = k as usize; }
    }
    s
}

/// Everything the boot resolved from the flags.
pub struct Setup {
    pub settings: ServeSettings,
    pub engine: EngineCfg,
    pub http: http::HttpOpts,
    pub lanes_requested: usize,
    pub lanes: usize,
    pub port: u16,
}

/// Resolve the serve flags against `args` (and the registry store) for a model directory.
pub fn setup(args: &[String], model_dir: &str, tok: &QwenTokenizer, cfg: &Dsv41Config) -> Result<Setup> {
    let arg = |f: &str| crate::arg_value(args, f).map(str::to_string);
    let lanes_requested: usize = arg("--max-batch").map(|s| s.parse()).transpose().context("--max-batch")?.unwrap_or(DEFAULT_LANES);
    let lanes = lanes_requested.clamp(1, MAX_LANES);
    let max_seq_len: usize = arg("--max-seq-len").map(|s| s.parse()).transpose().context("--max-seq-len")?.unwrap_or(cfg.max_pos).min(cfg.max_pos);
    let dir = Path::new(model_dir);
    let stop_ids = { let s = stop_token_ids(dir); if s.is_empty() { vec![cfg.eos] } else { s } };
    let thinking = match arg("--thinking").as_deref() { Some("off") => Some(false), Some("on") => Some(true), _ => None };
    let model_name = crate::server::resolve_model_name(arg("--model-name").as_deref(), arg("--served-model-name").as_deref(), model_dir);
    let prefix_on = prefix_cache_on(arg("--prefix-cache").as_deref());
    let pool_gb = opt_ranged("prefix-pool-gb", 32);
    let reasoning_end = crate::think_budget::reasoning_cfg().end_str.clone()
        .and_then(|s| tok.encode(&s, false).ok()).filter(|v| v.contains(&THINK_END_ID)).unwrap_or_else(|| vec![THINK_END_ID]);
    let settings = ServeSettings {
        model_name, max_seq_len,
        default_max_tokens: arg("--max-tokens").and_then(|s| s.parse().ok()),
        default_sampling: default_sampling(dir),
        server_thinking: thinking,
        server_effort: arg("--reasoning-effort").map(|e| effort_from_flag(&e)),
        keep_tools_when_none: opt_on("keep-tools-when-tool-choice-none"),
        seedless_mode: opt_str("seedless-seed").unwrap_or_else(|| "random".into()),
        server_think_budget: None,
        include_usage_default: true,
        vision: cfg.vision.clone(),
        image_token_id: cfg.image_token_id,
        vision_on: crate::dsv41::vision::vision_enabled(cfg),
        max_images: crate::dsv41::vision::max_images_from_registry(),
    };
    let world = tp_world(args).unwrap_or(1);
    let default_chunk = if world >= 4 { 8192 } else { 2048 };
    let engine = EngineCfg {
        lanes, stop_ids, sink: SinkCfg::from_opts(),
        prefix: prefix_on.then(|| (pool_gb.max(0) as u64 * (1 << 30), prefix::MIN_KEEP)),
        grid_snaps: opt_ranged("dsv41-prefix-grid-snaps", 4) as usize,
        prefill_chunk: arg("--prefill-chunk").and_then(|s| s.parse().ok()).unwrap_or(default_chunk),
        force_think_end: reasoning_end,
        near_end: opt_str("dsv41-prefix-near-end").as_deref() == Some("on"),
        tail_merge: opt_ranged("dsv41-prefill-tail-merge", 512) as usize,
    };
    let http = http::HttpOpts {
        max_waiting: opt_ranged("max-waiting", 256) as usize,
        backlog: opt_ranged("stream-backlog-events", 65_536) as usize,
        keepalive_secs: opt_ranged("sse-keepalive", 10) as u64,
        keepalive_data: opt_str("sse-keepalive-frame").as_deref() == Some("data"),
    };
    let port = arg("--port").and_then(|s| s.parse().ok()).unwrap_or(9000);
    Ok(Setup { settings, engine, http, lanes_requested, lanes, port })
}

/// The canned reply of the CLI stub forward: a one-line answer (with a short reasoning block in thinking mode).
pub fn canned_reply(tok: Arc<QwenTokenizer>) -> forward::ReplyFn {
    Box::new(move |prompt: &[u32]| {
        let enc = |s: &str| tok.encode(s, false).unwrap_or_default();
        let mut out = Vec::new();
        if prompt.last() == Some(&THINK_START_ID) {
            out.extend(enc("The stub forward has no model to reason with."));
            out.push(THINK_END_ID);
            out.extend(enc("\n\n"));
        }
        out.extend(enc("This is the DeepSeek-V4.1 serve shell answering from the CPU stub forward: no model is loaded in this process."));
        out
    })
}

/// Serve `app` on `listener` until the server stops. (Tests call this with their own listener.)
pub async fn serve_on(listener: tokio::net::TcpListener, app: Arc<http::App>) -> Result<()> {
    axum::serve(listener, http::router(app)).tcp_nodelay(true).await.context("axum serve")
}

/// LS2: the memory-fit check of the boot (UX row 7: graceful caps, never a refusal). Rank 0's weights come from the
/// shard plan at `--tp`; a posture that does not fit what this box can give (leg MEM1: its MemAvailable and reclaimable
/// swap, less the memory watchdog's floor) is capped (Engram cache, prefill chunk, lanes, context, in that order) with
/// ONE notice line and the `velogb10_dsv41_load_fit_*` rows; the Engram cap is applied (it rides `TpConfig` to the
/// nodes). The prefix pool is sized at run time by the governor (`tp::mem`). A pack the plan cannot read, or one whose
/// weights alone exceed the box, is one line too: the stub forward loads nothing, so it goes on; a real forward has
/// nothing to load into and says so.
pub fn apply_memory_fit(args: &[String], dir: &Path, su: &mut Setup, stub: bool) {
    let arg = |f: &str| crate::arg_value(args, f).map(str::to_string);
    let world: usize = tp_world(args).unwrap_or(1);
    // the TP forward's chunk is the engine's (`run_tp`: `--prefill-chunk`, default 2048), not T0's planning figure
    let tm: usize = if world >= 2 { su.engine.tail_merge } else { 0 };   // leg P4: the buffers hold chunk + tail merge rows
    let chunk: usize = if world >= 2 { su.engine.prefill_chunk.clamp(64, 16384) + tm } else { arg("--prefill-chunk").and_then(|s| s.parse().ok()).unwrap_or_else(|| crate::dsv41::load::fit::default_chunk(world)) };
    let explicit_cache = arg("--engram-row-cache-mib").and_then(|s| s.parse::<u64>().ok())
        .or_else(|| opt_str("engram-row-cache-mib").and_then(|s| s.parse::<u64>().ok()));
    let ram_gib = opt_str("dsv41-engram-ram-gib");
    let cache = crate::dsv41::load::fit::resolve_engram_cache_mib(dir, world, su.lanes, su.settings.max_seq_len, chunk, explicit_cache, ram_gib.as_deref());
    if explicit_cache.is_none() {
        if let Some(o) = opts::lookup("engram-row-cache-mib") { opts::set(o, cache.to_string()); }
    }
    let pool = su.engine.prefix.map_or(0, |p| p.0);
    match crate::dsv41::load::fit::boot_check(dir, world, su.lanes, su.settings.max_seq_len, chunk, cache, pool) {
        Ok((f, inp, c)) => {
            if let Some(n) = f.notice(&inp) { crate::boot_report::notice(&n); }
            crate::dprintln!("memory fit: {}", f.line(&c, &inp));
            su.lanes = f.lanes;
            su.engine.lanes = f.lanes;
            su.settings.max_seq_len = f.max_seq_len;
            if f.chunk < su.engine.prefill_chunk + tm { su.engine.prefill_chunk = f.chunk.saturating_sub(tm).max(64); }
            if explicit_cache.is_none() || f.engram_cache_mib != cache {
                if let Some(o) = opts::lookup("engram-row-cache-mib") { opts::set(o, f.engram_cache_mib.to_string()); }
            }
        }
        Err(e) => crate::boot_report::notice(&format!("memory fit not applied at --tp {world}: {e:#}{}", if stub { " (the stub forward loads no weights, so the boot goes on)" } else { "" })),
    }
}

/// `--tp N` from the command line (a bare `--tp` means 2, as everywhere else).
fn tp_world(args: &[String]) -> Option<usize> {
    for (i, a) in args.iter().enumerate() {
        if let Some(v) = a.strip_prefix("--tp=") { return v.parse().ok(); }
        if a == "--tp" { return Some(match args.get(i + 1) { Some(n) if !n.starts_with('-') => n.parse().ok()?, _ => 2 }); }
    }
    None
}

/// The TP serve (leg SE): rank 0 is this process. The head ships the shards, brings the link up, loads the model + the DSpark
/// stages and warms up (`tp::serve::boot_rank`), then HTTP runs on its own thread and the lane loop on this one, driving
/// `TpForward` (every call an op broadcast to the nodes, which `follow`).
fn run_tp(args: &[String], model_dir: &str, cfg: Dsv41Config, tok: Arc<QwenTokenizer>, su: Setup, world: usize) -> Result<()> {
    use crate::dsv41::tp::serve::TpForward;
    let set = |flag: &str, v: usize| { if let Some(o) = opts::lookup(flag) { opts::set(o, v.to_string()); } };
    // what the nodes must build identically rides the registry snapshot (TpConfig) the head ships
    let tp_chunk = su.engine.prefill_chunk.clamp(64, 16384);
    set("dsv41-tp-lanes", su.lanes);
    set("max-seq-len", su.settings.max_seq_len);
    set("dsv41-tp-chunk", tp_chunk);
    if su.lanes_requested != su.lanes {
        crate::boot_report::notice(&format!("--max-batch {}: the DeepSeek-V4.1 server runs at most {MAX_LANES} requests at once; booting with {} lanes, further requests wait in the queue (none is rejected)",
                                            su.lanes_requested, su.lanes));
    }
    if su.engine.sink.enabled {
        crate::boot_report::notice("--dsv41-sink-reject is not available with speculative rounds (a token cannot be held for a sink verdict): serving without it");
    }
    let mut su = su;
    if let Some((bytes, _)) = su.engine.prefix.as_mut() {
        let cap = if world <= 2 { 6u64 << 30 } else { 16u64 << 30 };
        if *bytes > cap {
            crate::boot_report::notice(&format!("--prefix-pool-gb: the TP{world} head holds its rank's weights in most of the box's memory; the prefix pool is capped at {} GiB (a kept state is ~50 MB at 100K tokens)", cap >> 30));
            *bytes = cap;
        }
    }
    // leg SE2: the lane loop hands the TP prefill `--prefill-chunk` rows per call (CE's exact CED, one schedule for the whole prompt: the
    // forward's `begin_prefill` / `Op::Prefill { total }`), so the other lanes' decode rounds run between the calls of a long prompt;
    // the grid snapshot below keeps its place (a snapshot of a prefix is a Full state: any resume is exact)
    su.engine.prefill_chunk = tp_chunk.max(256);
    su.engine.grid_snaps = 1;
    metrics::mark_start();
    metrics::set(&metrics::SERVING, 1);
    metrics::set(&metrics::LANES, su.lanes as u64);
    metrics::set(&metrics::LANES_REQUESTED, su.lanes_requested as u64);
    metrics::set(&metrics::PREFIX_ON, su.engine.prefix.is_some() as u64);
    metrics::set(&metrics::SINK_ON, 0);
    crate::metrics::kv_pool_init(su.lanes, su.settings.max_seq_len);
    scrub_notice();
    crate::metrics::set_max_batch_requested(su.lanes_requested);
    let mut slot = Some(su);
    let (vocab, dsv) = (cfg.vocab, Path::new(model_dir).to_path_buf());
    let mut head = |s: &mut crate::dsv41::tp::serve::TpServe, say: &mut dyn FnMut(String)| -> Result<()> {
        let su = slot.take().context("the head program ran twice")?;
        let (handle, parts) = engine::channel();
        let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64);
        let (prefix_on, lanes, ctx, model, port) = (su.engine.prefix.is_some(), su.lanes, su.settings.max_seq_len, su.settings.model_name.clone(), su.port);
        let spec = format!("{:?}", s.sp.policy.mode).to_lowercase();
        // UXF F5: speculation off by --mtp (or --dsv41-draft-len 0) says so once, in the boot notices
        if let Some(why) = crate::dsv41::boot::spec_off_reason(opt_str("mtp").as_deref(), opt_str("dsv41-draft-len").as_deref()) {
            crate::boot_report::notice(&format!("{why}: speculation is OFF on this server (plain decoding, one token per step; no DSpark drafts are made or verified){}",
                if why == "--mtp off" { "; same effect as --dsv41-draft-len 0" } else { "" }));
        }
        let pack = pack_name(&dsv);
        let spec_desc = if spec.contains("off") { "off".to_string() } else { format!("DSpark, draft length {spec}") };
        let app = Arc::new(http::App { max_model_len: ctx, settings: su.settings, tok: tok.clone(), engine: handle, http: su.http, created });
        let (tx_up, rx_up) = std::sync::mpsc::channel::<Result<()>>();
        std::thread::Builder::new().name("dsv41-http".into()).spawn(move || {
            let r = (|| -> Result<()> {
                let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().context("tokio runtime")?;
                rt.block_on(async move {
                    let listener = crate::server::http_listen(port).await;
                    let url = listener.local_addr().map_or_else(|_| format!("http://0.0.0.0:{port}"), |a| format!("http://{a}"));
                    // the SERVING line's shape is frozen (scripts grep it): its model= is the served id; the pack it was read from is one line before it
                    crate::logq::send(false, format!("[boot] serving pack {pack} (model id {model}; speculation: {spec_desc})"));
                    crate::boot_report::serving(&crate::boot_report::ServingLine {
                        family: "dsv41", model: &model, tp: world as u32, ctx, batch: lanes, kv: "fp4+fp8", prefix: if prefix_on { "on" } else { "off" },
                        mtp: "none", spec: if spec.contains("off") { "plain" } else { "dspark" }, port, url,
                    });
                    let _ = tx_up.send(Ok(()));
                    serve_on(listener, app).await
                })
            })();
            if let Err(e) = r { eprintln!("dsv41 http server stopped: {e:#}"); std::process::exit(1); }
        }).context("spawn the http thread")?;
        let _ = rx_up.recv_timeout(std::time::Duration::from_secs(60));
        say(format!("serving: {lanes} lanes, context {ctx}, prefix cache {}", if prefix_on { "on" } else { "off" }));
        // F4: the once-per-boot block naming the kernel entries that served each path and width (debug-only; still exported in /metrics)
        for l in metrics::boot_receipt_lines() { crate::dprintln!("[head] {l}"); }
        let mut fwd = TpForward::new(s, vocab, ctx);
        if let Some(l) = su.engine.prefix.and_then(|(b, _)| fwd.pool_line(b)) { crate::boot_report::notice(&l); }
        engine::run_loop(&mut fwd, su.engine, parts);
        bail!("the lane loop ended (a TP op failed, or every handle closed)")
    };
    let _ = &dsv;
    crate::dsv41::tp::launch::serve_head(args, Path::new(model_dir), world, &mut head)
}

/// The pack's directory name (what a user sees on disk), for the line next to SERVING.
pub fn pack_name(dir: &Path) -> String {
    dir.components().rev().find_map(|c| match c { std::path::Component::Normal(n) => Some(n.to_string_lossy().to_string()), _ => None }).unwrap_or_else(|| dir.display().to_string())
}

/// `--scrub-tool-secrets` is a served-text redaction: say which mode runs (one line, off included, as the other servers do).
fn scrub_notice() {
    let m = crate::tool_scrub::Mode::from_opt();
    let l = crate::tool_scrub::boot_line(m);
    if m == crate::tool_scrub::Mode::Off { crate::dprintln!("{l}"); } else { crate::boot_report::notice(&l); }
}

/// The entry of `--server --model-dir <dsv41 pack>` (one dispatch line in `src/cli/serve.rs`).
pub fn run(args: &[String], model_dir: &str) -> Result<()> {
    let dir = Path::new(model_dir);
    let cfg = Dsv41Config::load(dir)?;
    let tok = Arc::new(QwenTokenizer::from_file(&dir.join("tokenizer.json").to_string_lossy()).context("tokenizer.json")?);
    let mut su = setup(args, model_dir, &tok, &cfg)?;
    if crate::arg_value(args, "--chat-template").is_some() {
        crate::boot_report::notice("--chat-template does not apply to DeepSeek-V4.1: its chat format is DeepSeek's own encoder (encoding/encoding.py, ported as the dsv41 chat encoder), not a Jinja template; serving with the encoder");
    }
    if cfg.vision.is_some() && !su.settings.vision_on {
        crate::boot_report::notice("--dsv41-vision off: Rank-0 vision tower and aligner are NOT loaded; multimodal requests will be refused (400 images_unsupported)");
    }
    apply_memory_fit(args, dir, &mut su, opt_on("dsv41-stub-forward"));
    if !opt_on("dsv41-stub-forward") {
        let world = tp_world(args).unwrap_or(1);
        if world < 2 {
            bail!("DeepSeek-V4.1-Flash does not fit one GB10 (the EXL3 2.9 bpw pack alone is ~200 GB): start with --tp 2 --nodes <ip> (or --tp 4 with three nodes). \
                   To exercise the serve shell without a model, start with --dsv41-stub-forward");
        }
        return run_tp(args, model_dir, cfg, tok, su, world);
    }
    crate::boot_report::notice("--dsv41-stub-forward: serving from the CPU STUB forward (a canned reply; no model compute). Smoke and test only: this is not a speed or quality receipt");
    if su.lanes_requested != su.lanes {
        crate::boot_report::notice(&format!("--max-batch {}: the DeepSeek-V4.1 server runs at most {MAX_LANES} requests at once; booting with {} lanes, further requests wait in the queue (none is rejected)",
                                            su.lanes_requested, su.lanes));
    }
    if su.engine.sink.enabled {
        crate::boot_report::notice(&format!("--dsv41-sink-reject ON (opt-in, changes output bytes): sink threshold {} on residual column {}, up to {} alternates; every token is emitted one step later",
                                            su.engine.sink.threshold, opt_str("dsv41-sink-column").unwrap_or_else(|| "2455".into()), su.engine.sink.retries));
    }
    metrics::mark_start();
    metrics::set(&metrics::SERVING, 1);
    metrics::set(&metrics::STUB, 1);
    crate::metrics::kv_pool_init(su.lanes, su.settings.max_seq_len);
    scrub_notice();
    metrics::set(&metrics::LANES, su.lanes as u64);
    metrics::set(&metrics::LANES_REQUESTED, su.lanes_requested as u64);
    metrics::set(&metrics::PREFIX_ON, su.engine.prefix.is_some() as u64);
    metrics::set(&metrics::SINK_ON, su.engine.sink.enabled as u64);
    crate::metrics::set_max_batch_requested(su.lanes_requested);
    let fwd: Box<dyn Forward + Send> = Box::new(StubForward::new(cfg.vocab, cfg.eos, su.lanes, su.settings.max_seq_len, canned_reply(tok.clone())));
    let handle = engine::spawn(fwd, su.engine.clone());
    let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64);
    let (prefix_on, lanes, ctx, model, port) = (su.engine.prefix.is_some(), su.lanes, su.settings.max_seq_len, su.settings.model_name.clone(), su.port);
    let app = Arc::new(http::App { max_model_len: ctx, settings: su.settings, tok, engine: handle, http: su.http, created });
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().context("tokio runtime")?;
    rt.block_on(async move {
        let listener = crate::server::http_listen(port).await;
        let url = listener.local_addr().map_or_else(|_| format!("http://0.0.0.0:{port}"), |a| format!("http://{a}"));
        crate::boot_report::serving(&crate::boot_report::ServingLine {
            family: "dsv41", model: &model, tp: 1, ctx, batch: lanes, kv: "fp4+fp8", prefix: if prefix_on { "on" } else { "off" },
            mtp: "none", spec: "plain", port, url,
        });
        serve_on(listener, app).await
    })
}
