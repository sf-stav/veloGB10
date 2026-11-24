//! Memory fit and context clamp for the EXL3 server: `FitParts` / `fit_plan` / `fit_max_ctx` / `fit_max_lanes` (the pre-load footprint estimate) and
//! `clamp_ctx` / `resolve_max_seq_len` (the trained-window clamp). No GPU and no weight reads; `builder.rs` and the TP head entry point call it at boot.

use super::*;

/// CF-P1d pre-load fit check: the parts of a serving footprint, in bytes (estimates; see `fit_plan`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitParts {
    pub weights: u64,  // device weights on this rank (model shards / world + the vision tower on the head)
    pub table: u64,    // the PLE n-gram table (host RAM when RAM-resident)
    pub per_lane: u64, // one lane: KV + indexer keys + MTP-head KV over max_pos, + its recurrent state (x1.15)
    pub fixed: u64,    // scratch, prefill scratch, graphs, vendor workspaces, prefix-checkpoint cap
    pub floor: u64,    // the memory watchdog's floor (--mem-watchdog-gb)
}

/// What a configuration needs and whether it fits: Ok(table_in_ram_possible) or Err(the refusal text).
pub fn fit_plan(p: FitParts, lanes: usize, avail: u64, mode: &str, max_pos: usize) -> Result<bool, String> {
    let gb = |b: u64| b as f64 / 1e9;
    // KVPOOL (fork): with a lazy shared pool of a set size the lanes cost the POOL, not lanes x a full window
    // (+ ~0.3 GB a lane: its floor rows and the pages that straddle segment edges). Unset = the stock plan.
    let pool = match (crate::kv_vmm::mode_from_opts(), crate::kv_vmm::pool_bytes_from_opts()) {
        (Some(crate::kv_vmm::Mode::Lazy), Some(b)) => Some(b as u64 + 300_000_000 * lanes as u64),
        _ => None,
    };
    let lanes_bytes = pool.unwrap_or(p.per_lane * lanes as u64);
    let base = p.weights + p.fixed + lanes_bytes + p.floor;
    let with_table = base + p.table;
    let line = format!("model weights {:.1} GB + {lanes} lane(s) x {:.2} GB (KV for {max_pos} tokens{}) + {:.1} GB working memory \
                        + {:.1} GB safety floor{} = {:.1} GB; this machine has {:.1} GB available",
                       gb(p.weights), gb(lanes_bytes / lanes.max(1) as u64),
                       if pool.is_some() { "; shared KV pool" } else { "" }, gb(p.fixed), gb(p.floor),
                       if mode == "ram" { format!(" + n-gram table {:.1} GB in RAM", gb(p.table)) } else { String::new() },
                       gb(if mode == "ram" { with_table } else { base }), gb(avail));
    let fits_ssd = base <= avail;
    if mode == "ram" && with_table > avail {
        let mut msg = format!("this configuration does not fit in memory: {line}.");
        if fits_ssd {
            msg += " It fits with the n-gram table on SSD: use --ple-ram ssd (or leave --ple-ram at auto) — identical output, decode ~1-2% slower.";
        }
        return Err(msg + &suggest(p, lanes, avail, max_pos));
    }
    if !fits_ssd {
        return Err(format!("this configuration does not fit in memory, even with the n-gram table on SSD: {line}.{}",
                           suggest(p, lanes, avail, max_pos)));
    }
    Ok(with_table <= avail)
}

/// CTXCLAMP (owner 2026-11-07): the longest `--max-seq-len` below `asked` — a multiple of 1,024, at least 4,096 — at
/// which ONE lane's memory plan fits (`fit_plan` Ok, as `fit_max_lanes` reads it). `parts_at(max_pos)` gives the parts
/// at a context (the per-lane KV grows with it; nothing else does). None: not even 4,096 tokens fit (the refusal).
pub fn fit_max_ctx(asked: usize, headroom: usize, avail: u64, mode: &str,
                   parts_at: &dyn Fn(usize) -> Option<FitParts>) -> Option<usize> {
    let ok = |seq: usize| parts_at(seq + headroom).map_or(false, |p| fit_plan(p, 1, avail, mode, seq + headroom).is_ok());
    const MIN_K: usize = 4; // 4,096 tokens
    let mut hi = asked / 1024;                 // candidates are k * 1024 for k in MIN_K..=hi, below `asked`
    if hi * 1024 >= asked { hi = hi.saturating_sub(1); }
    if hi < MIN_K || !ok(MIN_K * 1024) { return None; }
    if ok(hi * 1024) { return Some(hi * 1024); }
    let mut lo = MIN_K;                        // invariant: ok(lo * 1024), !ok(hi * 1024)
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if ok(mid * 1024) { lo = mid } else { hi = mid }
    }
    Some(lo * 1024)
}

/// MAXBATCH1: the largest lane count `<= width` whose memory plan fits — `fit_plan` Ok (the auto
/// arm accepts the n-gram table landing on SSD, exactly like an un-fitted boot). None when not
/// even one lane fits (the refusal case: nothing can be served on this box at this context).
pub fn fit_max_lanes(p: FitParts, width: usize, avail: u64, mode: &str, max_pos: usize) -> Option<usize> {
    (1..=width).rev().find(|&n| fit_plan(p, n, avail, mode, max_pos).is_ok())
}

fn suggest(p: FitParts, lanes: usize, avail: u64, max_pos: usize) -> String {
    let room = avail.saturating_sub(p.weights + p.fixed + p.floor);
    let mut out = String::new();
    if p.per_lane > 0 && room / p.per_lane < lanes as u64 {
        let n = room / p.per_lane;
        if n >= 1 { out += &format!(" At this context length, at most --max-batch {n} fits."); }
        let per_tok = p.per_lane as f64 / max_pos.max(1) as f64;
        let ctx = (room as f64 / lanes.max(1) as f64 / per_tok) as u64 / 1024 * 1024;
        if ctx >= 4096 { out += &format!(" With --max-batch {lanes}, at most --max-seq-len {ctx} fits."); }
    }
    if room < p.per_lane {
        out += " The model alone leaves too little memory on one box: run it over two boxes (--tp 2).";
    }
    out
}

/// Which limit trimmed a `--max-seq-len` request (public PR #21).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CtxLimit {
    /// The model's trained window (`max_position_embeddings`) times the rope extension factor.
    Rope,
    /// The launch-argument bit-packings: `max_pos` rides in 20 bits (`exl3_forward::MAX_POS_PACKED`).
    Packing,
}

/// The YaRN factor the EXL3 serve honours for `model_dir`: `--rope-yarn-factor` wins, then
/// the shipped TpConfig (TP node), then the pack config's `rope_scaling` block (YARN1). One
/// precedence rule with the Config-based `qwen::resolved_yarn` (shared `qwen::pick_yarn`).
pub fn yarn_factor(model_dir: &str) -> f64 {
    let (f, _) = crate::qwen::pick_yarn(
        crate::opts::var(crate::opt!("rope-yarn-factor")).ok().and_then(|v| v.parse::<f32>().ok()),
        crate::tp::tp_config().map(|c| c.rope_yarn_factor),
        pack_yarn_from_config(model_dir),
    );
    f as f64
}

/// The pack config's own YaRN factor — `rope_scaling {type: "yarn", factor}` from the same
/// config.json `native_window` opens (None = the pack ships no YaRN block).
fn pack_yarn_from_config(model_dir: &str) -> Option<f32> {
    let raw = std::fs::read(std::path::Path::new(model_dir).join("config.json")).ok()?;
    let cfg: serde_json::Value = serde_json::from_slice(&raw).ok()?;
    let t = cfg.get("text_config").unwrap_or(&cfg);
    let rs = t.get("rope_scaling")?;
    let is_yarn = rs.get("type").and_then(|v| v.as_str())
        .map_or(false, |s| s.eq_ignore_ascii_case("yarn"));
    if !is_yarn { return None; }
    rs.get("factor").and_then(|v| v.as_f64())
        .map(|f| f as f32).filter(|f| f.is_finite() && *f >= 1.0)
}

/// The widest `--max-seq-len` the EXL3 packings carry: `max_pos = max_seq_len + decode_headroom`
/// must stay below `MAX_POS_PACKED`.
pub fn packable_max_seq_len() -> usize {
    (crate::exl3_forward::MAX_POS_PACKED - 1).saturating_sub(crate::batch::decode_headroom(false))
}

/// The context a `--max-seq-len` request may use, and the limit that trimmed it (None = honoured as
/// asked). `native` is the pack's trained window when its config could be read; the packing ceiling
/// is a constant and applies either way, so an unreadable config never disables the guard.
pub fn clamp_ctx(max_seq_len: usize, native: Option<usize>, rope_factor: f64) -> (usize, Option<CtxLimit>) {
    let pack_eff = packable_max_seq_len();
    let (eff, why) = match native {
        Some(n) => {
            let rope_eff = ((n as f64) * rope_factor.max(1.0)).ceil() as usize;
            (rope_eff.min(pack_eff), if pack_eff < rope_eff { CtxLimit::Packing } else { CtxLimit::Rope })
        }
        None => (pack_eff, CtxLimit::Packing),
    };
    if max_seq_len > eff { (eff, Some(why)) } else { (max_seq_len, None) }
}

/// The pack's trained window from its config.json (`text_config` when nested), as `fit_parts` reads it.
pub(super) fn native_window(model_dir: &str) -> Option<usize> {
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::path::Path::new(model_dir).join("config.json")).ok()?).ok()?;
    let t = cfg.get("text_config").unwrap_or(&cfg);
    t.get("max_position_embeddings").and_then(|v| v.as_u64()).map(|v| v as usize)
}

/// One line naming the limit that trimmed `asked` to `eff` (pure: the text is unit-tested).
pub fn clamp_ctx_message(asked: usize, eff: usize, why: CtxLimit, native: Option<usize>, rope_factor: f64) -> String {
    let limit = match why {
        CtxLimit::Rope if rope_factor > 1.0 =>
            format!("the model's trained window {} x rope factor {rope_factor}", native.unwrap_or(0)),
        CtxLimit::Rope => format!("the model's trained window {} (max_position_embeddings)", native.unwrap_or(0)),
        CtxLimit::Packing => format!("the launch-argument bit-packings (max_pos must stay below {} = 2^20)",
                                     crate::exl3_forward::MAX_POS_PACKED),
    };
    format!("[warn] --max-seq-len {asked} exceeds {limit} - clamping to {eff}.")
}

/// The context this EXL3 serve honours for `model_dir`: `--max-seq-len` trimmed to the rope and
/// packing ceilings, with one log line naming the limit when it was trimmed. Every rank of a TP serve
/// resolves it from the same argv and the same pack config (`build_serve`), and the head ships the
/// trimmed value in `TpConfig.max_seq_len`.
pub fn clamp_max_seq_len(model_dir: &str, max_seq_len: usize) -> usize {
    let (eff, msg) = resolve_max_seq_len(model_dir, max_seq_len);
    if let Some(m) = msg { crate::drprintln!("{m}"); }
    eff
}

/// `clamp_max_seq_len` without the log line: (window, the line to print when it was trimmed).
pub fn resolve_max_seq_len(model_dir: &str, max_seq_len: usize) -> (usize, Option<String>) {
    let native = native_window(model_dir);
    let yarn = yarn_factor(model_dir);
    let (eff, why) = clamp_ctx(max_seq_len, native, yarn);
    (eff, why.map(|w| clamp_ctx_message(max_seq_len, eff, w, native, yarn)))
}

/// Estimate the footprint of an EXL3 pack from its files and config (no GPU, no weight reads).
pub fn fit_parts(model_dir: &str, max_pos: usize, world: usize, head: bool, kv_bytes: f64, ckpt_gb: f64) -> Option<FitParts> {
    let dir = std::path::Path::new(model_dir);
    let size = |n: &str| std::fs::metadata(dir.join(n)).map(|m| m.len()).unwrap_or(0);
    let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json")).ok()?).ok()?;
    let all: std::collections::BTreeSet<String> = idx["weight_map"].as_object()?.values()
        .filter_map(|v| v.as_str().map(str::to_string)).collect();
    // the n-gram table is counted as `table`, not device weights (in a shipped shard dir it sits under seg/)
    let (ngram, files): (std::collections::BTreeSet<String>, std::collections::BTreeSet<String>) =
        all.into_iter().partition(|f| f.contains("ngram"));
    let shards: u64 = files.iter().map(|f| size(f)).sum();
    // A shipped per-rank shard dir (B32, `gb10_shard.json`) already holds this rank's share; a full pack is split by world.
    let share = if dir.join("gb10_shard.json").is_file() { 1 } else { world.max(1) as u64 };
    // FP4L: the n-gram table and the tower live INSIDE the shards; size them from the tensor headers
    let fp4l = crate::fp4l::fit_bytes(dir);
    let mut weights = match fp4l { Some((d, _, _)) => d / share, None => shards / share };
    if head { weights += match fp4l { Some((_, _, tw)) => tw, None => size("vision_tower_bf16.safetensors") }; }
    let mut table: u64 = match fp4l { Some((_, t, _)) => t, None => ngram.iter().map(|f| size(f)).sum() };
    if table == 0 {
        table = std::fs::read_dir(dir).ok()?.flatten()
            .filter(|e| e.file_name().to_string_lossy().contains("ngram"))
            .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0)).sum();
    }
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
    let t = cfg.get("text_config").unwrap_or(&cfg);
    let g = |k: &str| t[k].as_u64().unwrap_or(0);
    let types = t["layer_types"].as_array()?;
    let n_full = types.iter().filter(|v| v.as_str() == Some("full_attention")).count() as u64;
    let n_lin = types.len() as u64 - n_full;
    let kvh_hd = g("num_key_value_heads") * g("head_dim");
    // per token: trunk K+V (kv-cache format) + indexer keys (bf16) + the MTP head's own K+V (f16)
    // trunk KV heads are split across ranks up to their count (2 KV heads: TP=2 and TP=4 hold one each)
    let kv_split = (world.max(1) as u64).min(g("num_key_value_heads").max(1)) as f64;
    let per_tok = n_full as f64 * 2.0 * kvh_hd as f64 * kv_bytes / kv_split
        + (n_full * g("indexer_kv_heads") * g("indexer_head_dim") * 2) as f64
        + (g("mtp_num_hidden_layers").max(1) * 2 * kvh_hd * 2) as f64;
    // per lane fixed: GDN recurrent state (f32) + conv ring
    let state = n_lin * g("linear_num_value_heads") * g("linear_key_head_dim") * g("linear_value_head_dim") * 4
        + n_lin * (2 * g("linear_num_key_heads") * g("linear_key_head_dim") + g("linear_num_value_heads") * g("linear_value_head_dim"))
            * g("linear_conv_kernel_dim").max(1) * 2;
    let per_lane = ((per_tok * max_pos as f64 + state as f64) * 1.15) as u64;
    // Working memory beyond weights + table + lanes, measured on .14 2026-10-02 (Flash-Next 3.05 TP=1, 131K): 14.3 GB
    // at 1 lane and 14.8 GB at 8 lanes after the boot (scratch, prefill scratch, graphs, vendor workspaces, staging),
    // + the prefix-checkpoint pool, which grows to its cap while serving.
    let fixed = 15_000_000_000 + (ckpt_gb * (1u64 << 30) as f64) as u64;
    // MAXBATCH1 (2026-10-21, .12, 27B EXL3 3.05, TP=1, posture ctx 262144, 8-lane clamp): serving-time memory
    // grew ~6.4 GB BEYOND this model under one 32-concurrency cell (CUDA-graph captures at mixed widths,
    // prefix/snapshot churn, the queue) and MemAvailable fell to 4.4 GB — the memwatch emergency exit fired and
    // the server died mid-traffic. The 5 GB watchdog floor is the EMERGENCY bar (box protection), not a serving
    // plan. The serve fit reserves the owner's §5 serving-headroom bar instead: ~20 GB free per box on a serving
    // posture ("lower is good"); the watchdog keeps its own floor as the last resort below that.
    let watch_gb: f64 = crate::opts::var(crate::opt!("mem-watchdog-gb")).ok().and_then(|v| v.parse().ok()).unwrap_or(5.0);
    let floor_gb = watch_gb.max(crate::opts::var(crate::opt!("fit-floor-gb")).ok().and_then(|v| v.parse().ok()).unwrap_or(20.0));
    Some(FitParts { weights, table, per_lane, fixed, floor: (floor_gb * 1e9) as u64 })
}
