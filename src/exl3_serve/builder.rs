//! `build_serve`: the boot sequence of the EXL3 server (flag resolution, memory fit, model load, scratch and graph precapture,
//! scheduler construction). One item, kept whole in its own file; it returns the `ServeParts` that `serve.rs` turns into a running server.
//! Boot lines printed here are pinned by the source-text tests (guard line before each print); keep the lines and their order as they are.

use super::*;

/// TP-D: resolve every serve knob from `args`, load (TP: this rank's shard) and build the
/// scheduler + HTTP state. At TP the head and the node run this with the SAME args (the head ships
/// its argv in TpConfig.exl3_mode), so both resolve every knob identically; `tp_boot_agree` then
/// proves the resolved scheduler config equal across ranks before a request is served.
pub(super) fn build_serve(args: &[String], model_dir: &str, tp: Option<crate::exl3_forward::xtp::TpAttach>) -> Result<ServeParts> {
    let tp_rank: Option<i32> = tp.as_ref().map(|a| a.rank);
    let tp_world: u32 = tp.as_ref().map_or(1, |a| a.world as u32);
    let who = match tp_rank { Some(r) => format!("TP={tp_world} rank {r}"), None => "TP=1".to_string() };
    // V0710: family/TP-scoped resolved defaults (--copy-span, --copy-span-max-w, --exl3-dense-resident, --exl3-mid-gemm,
    // --spec-arm-meter), BEFORE any of them is read. Pure in (family, world, set values): every TP rank resolves alike.
    crate::family_defaults::resolve_exl3(model_dir, tp_world as usize);
    // KVPOOL (veloGB10 #43): a TP rank serves the stock per-lane KV allocation even when --kv-pool is asked (said at bind)
    if tp_world > 1 { crate::kv_vmm::note_tp_role(); }
    // v0.7.3 ("tel"): --otel-* on the EXL3 server, the SAME shared parser as the NVFP4 path
    // (otel::config_from_args). Parsed + refused on EVERY rank BEFORE the model load (a bad
    // endpoint aborts the boot); the SINK is built later, only where the HTTP API lives.
    let otel_cfg = crate::otel::config_from_args(
        |f| arg(args, &format!("--{f}")).map(|s| s.to_string()));
    if let Some(cfg) = &otel_cfg {
        if let Err(e) = cfg.hostport() { crate::deprintln!("[otel] {e}"); crate::logq::flush_and_exit(std::time::Duration::from_millis(300), 1); } // H7
    }
    // S-B9-CF-PACK4: --lane-pack, parsed before the model load. F10 (owner 2026-10-12): the registry default is
    // ON, so the mode is resolved from the TYPED value (this head's argv; the node re-parses the head's argv) and
    // otherwise from the registry row: both ranks of one binary resolve the same mode with no flag typed.
    // Only a TYPED `--lane-pack on` above the largest world is refused (as before); a defaulted-on boot that
    // cannot pack, for that or any other reason, falls back with one line (AGENTS 1b-bis item 8).
    let lane_pack_typed = arg(args, "--lane-pack");
    let mut lane_pack = match lane_pack_typed {
        Some(v) => crate::lane_pack::LanePack::parse(Some(v))?,
        None => crate::lane_pack::LanePack::from_opts()?,
    };
    if lane_pack_typed.is_some() { crate::lane_pack::tp_refusal(lane_pack, tp_world)?; }
    let port: u16 = arg(args, "--port").and_then(|s| s.parse().ok()).unwrap_or(8000);
    // Public PR #21: trim --max-seq-len to what the engine can address (the trained window, then the
    // 20-bit packings) and say which limit applied; an unclamped wider window used to boot.
    // YARN1: set the trunk YaRN factor BEFORE both — the load (the EXL3 tables build at load)
    // and the clamp (its rope ceiling is native x factor). The EXL3 TP node re-parses this
    // same argv and the value also ships in TpConfig, so both ranks resolve identically.
    if let Some(f) = arg(args, "--rope-yarn-factor") {
        let f: f32 = f.parse().map_err(|_| anyhow::anyhow!(
            "--rope-yarn-factor must be a finite number >= 1.0 (got {f:?})"))?;
        if !f.is_finite() || f < 1.0 {
            anyhow::bail!("--rope-yarn-factor must be a finite number >= 1.0 (got {f})");
        }
        crate::opts::set(crate::opt!("rope-yarn-factor"), f.to_string());
    }
    let st_config = crate::boot_report::Stage::begin("config");
    // CTXCLAMP (owner 2026-11-07): `mut` — at TP = 1 the memory fit below may boot a context that does not fit even
    // one lane at the longest context that does (one boot line + /metrics), instead of refusing.
    let mut max_seq_len: usize = clamp_max_seq_len(model_dir, arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096));
    // F2 (owner 2026-10-10): at most 16 requests are served at once; a larger --max-batch boots
    // with 16 and further requests wait in the queue (--max-waiting semantics unchanged).
    // MAXBATCH1: `width` is `mut` — the memory fit below may boot it down to the largest lane
    // count that fits (§1b-bis#8 boot-anyway; the context length is never silently cut).
    let (mut width, raw_width) = resolved_max_batch(args);
    crate::metrics::set_max_batch_requested(raw_width);
    if raw_width > width {
        crate::boot_report::notice(&format!("[exl3-serve] --max-batch {raw_width}: this server serves at most {EXL3_MAX_LANES} \
                           requests at the same time; using {width}; further requests wait in the queue"));
    }
    // (backstop: the TP entry points resolve the same cap before any network step,
    // `tp_lane_width_refusal`; 16 always fits the pairwise agree width)
    tp_lane_width_fits(tp_world, width).map_err(|e| anyhow::anyhow!(e))?;
    // S-B9-CF-AGREE16B (W3): at TP=1 there is no TpSync to say what a lockstep drill flag does: say it here
    if tp_rank.is_none() {
        for l in crate::net::drill_plan(crate::opts::get(crate::opt!("tp-agree-drill")).as_deref(),
                                        crate::opts::get(crate::opt!("tp-width-drill")).as_deref(), true, None).lines {
            crate::boot_report::notice(&l);
        }
    }
    let mut max_pos = max_seq_len + crate::batch::decode_headroom(false);
    // --prefix-cache on|off (default on, as the NVFP4 server): per-slot recurrent-state snapshots
    // at the message boundary so a multi-turn re-send prefills only its new suffix.
    let prefix_on = !matches!(arg(args, "--prefix-cache"), Some("off"));
    if prefix_on { crate::opts::set(crate::opt!("exl3-prefix"), "1"); }
    // WP16: --prefix-ckpt-mem-gb <GiB> — the LRU byte cap of the intermediate recurrent checkpoints
    // (allocated lazily up to it). Default 4 (placeholder: the owner sets the final default);
    // 0 = no checkpoints (today's behaviour, as --wp16-off=1).
    let wp16_gb: f64 = match arg(args, "--prefix-ckpt-mem-gb") {
        None => 4.0,
        Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--prefix-ckpt-mem-gb must be a number (GiB)"))?,
    };
    anyhow::ensure!(wp16_gb.is_finite() && wp16_gb >= 0.0, "--prefix-ckpt-mem-gb must be >= 0");
    // --ple-ram on|off|auto: where the ~30 GiB PLE n-gram table lives (RAM vs per-token pread
    // through the page cache). Output-identical either way; transported to the loader via env.
    if let Some(v) = arg(args, "--ple-ram") {
        anyhow::ensure!(matches!(v, "auto" | "ram" | "ssd" | "on" | "off"), "--ple-ram must be auto, ram or ssd (on/off are the old spellings of ram/ssd)");
        crate::opts::set(crate::opt!("ple-ram"), v);
    }
    // --kv-cache f32|f16|fp8|q8: attention KV storage format (PLAN/KV_CACHE_FORMATS.md). Default
    // q8 (owner decision 2026-09-26, after the WP25 quality gate); f32 is exact; f16 halves, fp8 (e4m3 + per-row f32 scale) and q8 (the reference implementation's H32-rotated int8 +
    // f16 per-32 scale, WP25) quarter the long-context KV bytes. Transported to the loader via
    // [kv-cache] (an explicit env value wins when the flag is absent).
    if let Some(v) = arg(args, "--kv-cache") {
        let v = exl3_kv_cache_value(v)?;   // TPSHIP: bf16 -> f16 with one line (no refusal)
        crate::opts::set(crate::opt!("kv-cache"), v);
    }
    // --spec-source <s> (ISSUES[94]): the mirror of the trunk resolve_spec_source (main.rs) —
    // same value set (mtp|dspark|dflash2|dflash2-rq|dflash2-auto|dflash2-tree|none), same
    // default (dflash2-auto), same error text — parsed HERE so the EXL3 serve path honors what
    // it was asked for instead of silently ignoring the DF2 family. CLI flags only, never env.
    let df2_src = resolve_spec_source_exl3(args);
    // --draft-dir <dir>: the DFlash2 artifact directory, the same flag the NVFP4 serving path
    // takes. (main's pre-dispatch resolve_draft_dir already exits on an explicit
    // artifact-REQUIRING source with no dir; dflash2-auto — the default — without a dir is the
    // silent-ignore case this gate fixes, with a loud FALLBACK line.)
    // FLASHNEXT-DF2-C (ISSUES 116): under an EXL3 TP serve the SHIPPED config decides the draft
    // dir (the head's absolute path on rank 0, the blob-cache dir the sync assembled on a node,
    // empty = the head shipped no drafter); a node never opens the head's filesystem path.
    let df2_draft_dir = df2_draft_dir_for_boot(args, crate::tp::tp_config());
    // --df2-block <n>: the artifact's block size (default: dflash2::block(), 8; 16 on the wide
    // selector artifact). The round's chain-verify depth is k = block - 1.
    let mut df2_block = parse_df2_block(arg(args, "--df2-block"))?;
    if args.iter().any(|a| a == "--df2-trace") {
        crate::exl3_forward::DF2_TRACE.store(true, std::sync::atomic::Ordering::Relaxed);
        println!("[exl3-serve] df2: trace on (--df2-trace: stage syncs, rounds slower)");
    }
    // --df2-round-shard on|off (default off): the TP>2-only round sharding — parsed for flag
    // parity and shown on the DF2 boot line; it never applies on this TP=1 path.
    let df2_shard = parse_df2_round_shard(arg(args, "--df2-round-shard"))?;
    let kv_fmt = crate::exl3_forward::kv_fmt_from_opts();
    anyhow::ensure!(kv_fmt != crate::exl3_forward::KV_Q4X,
                    "[kv-cache]=q4x is the WP25 probe's deliberately broken control; it is never served");
    let kv_name = crate::exl3_forward::kv_fmt_name(kv_fmt);
    // PACK4-REVIEW B3, now F1 (owner 2026-10-10: "support things properly or gracefully fall back"):
    // the postures --lane-pack on was never gated under no longer REFUSE the boot. The reasons are
    // collected here — before the model load, so nothing packed is ever built for a boot that will
    // not pack — and the boot decides just before the packed scratch: ONE plain line, packed rounds
    // OFF for this boot, and the decision rides /metrics and the TP boot agree (a rank-local
    // difference still dies loudly there).
    let mut pack_blockers: Vec<String> = Vec::new();
    if lane_pack.on() && tp_world > crate::lane_pack::LANE_PACK_MAX_WORLD {
        pack_blockers.push(format!("TP={tp_world}: the packed arm has a lockstep frame for at most {} machines",
                                   crate::lane_pack::LANE_PACK_MAX_WORLD));
    }
    if lane_pack.on() {
        pack_blockers.extend(crate::lane_pack::posture_fallbacks(&crate::lane_pack::Posture {
            kv: kv_name,
            mtp_k: crate::exl3_forward::mtp_depth_default(),
            row_regime: crate::exl3_forward::row_regime_on(),
            a_ctrl_old: crate::exl3_forward::a_flags().ctrl == 2,
            fcfs: LaneOrder::parse(crate::opt_or_arg(args, "--lane-order").as_deref().map(str::trim).filter(|v| !v.is_empty()))? == LaneOrder::Fcfs,
            dense_v3: crate::exl3_forward::dense_v3_on(),
            kv_any: crate::lane_pack::kv_any(),
        }));
        if crate::lane_pack::kv_any() && !matches!(kv_name, "q8" | "q4k") {
            crate::rprintln!("[exl3-serve] lane packing runs on --kv-cache {kv_name} (--lane-pack-kv any): UNGATED — no gate chain has run \
                              packed verifies on a {kv_name} cache yet; --lane-pack-kv q8 serves that cache without packed rounds");
        }
    }
    // --reasoning-effort / --thinking: the server-wide thinking defaults, as on the NVFP4 server.
    // Unset = the model's own chat template decides (Qwen3.8: thinking on at xhigh, the model
    // card's default). A request's reasoning_effort / chat_template_kwargs.enable_thinking wins.
    let reasoning_effort = match arg(args, "--reasoning-effort") {
        None => None,
        Some(e @ ("none" | "no_think" | "low" | "medium" | "high" | "xhigh")) => Some(e.to_string()),
        Some(other) => anyhow::bail!("--reasoning-effort must be none|low|medium|high|xhigh (got '{other}')"),
    };
    let thinking = match arg(args, "--thinking") {
        None => crate::tokenizer::ThinkingMode::Auto,
        Some(v) => crate::tokenizer::ThinkingMode::parse(v)
            .ok_or_else(|| anyhow::anyhow!("--thinking must be auto|on|off (got '{v}')"))?,
    };
    // HOST / RO-7: --cpu-affinity auto|off (default auto; src/cpu_affinity.rs). Resolved here so
    // a bad value fails before the load; applied after it (the load's threads keep every core).
    let cpu_aff = crate::cpu_affinity::resolve(
        crate::cpu_affinity::AffinityMode::parse(arg(args, "--cpu-affinity"))?)?;
    // MAXBATCH1 (owner ruling 2026-10-10 §1b-bis#8: "support it properly or fall back gracefully —
    // no refusals"): resolve the memory fit BEFORE the tune table is selected (the table is keyed
    // by the lane count that will actually serve). An asked lane count that does not fit memory
    // BOOTS ANYWAY on the nearest thing that works — the largest lane count that fits at the
    // asked context — with ONE clear line: what was asked, what it does instead, why. Lanes are
    // cut before the context (full-context serving is the READY bar; a lane waits instead).
    // CTXCLAMP (owner 2026-11-07): when not even ONE lane fits, a TP = 1 boot serves one lane at the
    // longest context that fits (one line + /metrics) instead of refusing; TP > 1 and a pack whose
    // weights alone overflow keep CF-P1d's refusal, with the arithmetic and what to change.
    {
        // Bytes per element per K/V channel: q4k's row is 144 B at hd 256, q8's 272.
        let kvb = match kv_name {
            "f32" => 4.0,
            "f16" => 2.0,
            "fp8" => 1.0,
            "q4k" => 0.5625,
            _ => 1.0625,
        };
        let mode = match crate::opts::var(crate::opt!("ple-ram")).as_deref() { Ok("on") | Ok("ram") => "ram", Ok("off") | Ok("ssd") => "ssd", _ => "auto" };
        let world = tp_world.max(1) as usize;
        if let Some(parts) = fit_parts(model_dir, max_pos, world, tp_rank.map_or(true, |r| r == 0), kvb, wp16_gb) {
            let avail = std::fs::read_to_string("/proc/meminfo").ok()
                .and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable:"))
                    .and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok()))
                .unwrap_or(0) * 1024;
            if avail > 0 {
                if let Err(ref msg) = fit_plan(parts, width, avail, mode, max_pos) {
                    if let Some(n) = fit_max_lanes(parts, width, avail, mode, max_pos) {
                        let gb = |b: u64| b as f64 / 1e9;
                        crate::boot_report::notice(&format!(
                            "[exl3-serve] --max-batch {raw_width}: {width} lanes do not fit memory at this context length on this box \
                             ({avail_g:.1} GB available; asked-for plan: weights {w_g:.1} GB + {width} x {pl_g:.2} GB per-lane KV + {fx_g:.1} GB \
                             working memory + {fl_g:.1} GB serving headroom); booting with {n} lanes — the most that fit (the context length is \
                             kept); further requests wait in the queue",
                            avail_g = gb(avail), w_g = gb(parts.weights), pl_g = gb(parts.per_lane),
                            fx_g = gb(parts.fixed), fl_g = gb(parts.floor)));
                        width = n;
                    } else {
                        // CTXCLAMP (owner 2026-11-07, "probably yes"): not even ONE lane fits at the asked context. At
                        // TP = 1 boot anyway with one lane at the longest context that fits (a multiple of 1,024, at
                        // least 4,096), one line saying what was asked and what serves, and /metrics
                        // velogb10_exl3_ctx_clamped{asked,served}. TP > 1 keeps the refusal: every rank runs this fit
                        // on its own memory and ranks must not disagree on the context.
                        let headroom = crate::batch::decode_headroom(false);
                        let parts_at = |mp: usize| fit_parts(model_dir, mp, world, true, kvb, wp16_gb);
                        let clamp = if tp_world <= 1 { fit_max_ctx(max_seq_len, headroom, avail, mode, &parts_at) } else { None };
                        let Some(ctx) = clamp else {
                            eprintln!("error: {msg}");
                            std::process::exit(2);
                        };
                        let gb = |b: u64| b as f64 / 1e9;
                        crate::boot_report::notice(&format!(
                            "[exl3-serve] --max-seq-len {asked}: not even one lane fits memory at this context length on this \
                             box ({avail_g:.1} GB available; one lane needs weights {w_g:.1} GB + {pl_g:.2} GB KV + {fx_g:.1} GB \
                             working memory + {fl_g:.1} GB serving headroom); booting with 1 lane at --max-seq-len {ctx} — the \
                             longest context that fits (--tp 2 serves the full context)",
                            asked = max_seq_len, avail_g = gb(avail), w_g = gb(parts.weights), pl_g = gb(parts.per_lane),
                            fx_g = gb(parts.fixed), fl_g = gb(parts.floor)));
                        crate::metrics::set_exl3_ctx_clamped(max_seq_len, ctx);
                        max_seq_len = ctx;
                        max_pos = ctx + headroom;
                        width = 1;
                    }
                }
            }
        }
    }
    // TUNE T0 (PLAN/AUTOTUNE_DESIGN.md §7): resolve + freeze the tune table BEFORE the load (Load-scope
    // knobs are read there). --tune-table auto (default: a matching table from ./tune/, S section) |
    // off (built-in defaults) | <path>; --tune-draft on also applies section D (draft-only);
    // --tune-profile <MHz> picks the clock profile (default: detected, else the 2400 MHz stock one).
    // A missing or mismatched table is a loud FALLBACK to the defaults (== today, bitwise).
    {
        use crate::exl3_tune as tune;
        let sel = tune::TableSel::parse(arg(args, "--tune-table"))?;
        let draft_on = match arg(args, "--tune-draft") {
            None | Some("off") => false,
            Some("on") => true,
            Some(o) => anyhow::bail!("--tune-draft must be on or off (got '{o}')"),
        };
        let profile_mhz = match arg(args, "--tune-profile") {
            None => None,
            Some(v) => Some(v.parse::<u32>().map_err(|_| anyhow::anyhow!("--tune-profile must be MHz (got '{v}')"))?),
        };
        let posture = tune::Posture {
            lanes: width,
            mtp_max_k: crate::exl3_forward::mtp_depth_default(),
            dds: arg(args, "--draft-confidence").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.4) > 0.0,
            kv_fmt: crate::exl3_forward::kv_fmt_from_opts(),
            max_pos,
            prefill_chunk: arg(args, "--prefill-chunk").and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_PREFILL_CHUNK),
        };
        tune::boot(&tune::BootReq { sel, model_dir, posture, profile_mhz, draft_on,
                                    tp: tp_world });
    }
    // (CF-P1d + MAXBATCH1: the memory fit ran above, before the tune table — boot-anyway clamp.)
    st_config.ok();
    let st_weights = crate::boot_report::Stage::begin("weights");
    if crate::boot_report::debug() {
        crate::dprintln!("[exl3-serve] {who}: loading EXL3 pack {model_dir} (width {width}, max_pos {max_pos}, kv-cache {kv_name})");
    }
    let mut model = FwdModel::load_tp(model_dir, width, max_pos, tp)?;
    st_weights.ok();
    // ---- ISSUES[94]: the DFlash2 (DF2) boot gate + round load. Exactly ONE line per boot —
    // ENGAGED or FALLBACK with the reason — and a load failure is a FALLBACK, never a dead
    // boot (the pack MTP this server already has serves every request unchanged).
    let mut df2: Option<crate::dflash2::round::Df2Round> = None;
    match df2_gate_reason(df2_src, df2_draft_dir.as_deref(), model.cfg.hidden_size,
                          model.cfg.vocab_size, model.cfg.num_layers, max_seq_len,
                          tp_world as usize) {
        Some(reason) => {
            println!("{}", df2_boot_line_fallback(&reason));
            // FLASHNEXT-DF2-B (TP=2 boot 2026-10-23): a node re-parses the head's argv and reads
            // --draft-dir from ITS OWN disk (the EXL3 TP sync ships the pack, not the drafter). A
            // node without the artifact fell to the 27B default geometry, reported a "dims
            // mismatch" and served MTP while the head ENGAGED — the pair diverged at the first
            // round. Say so, next to the reason.
            if let Some(dir) = df2_draft_dir.as_deref() {
                if !std::path::Path::new(dir).join("config.json").is_file() {
                    println!("[exl3-serve] df2: note — {dir}/config.json is not readable on this box, so the dims above assume the 27B default geometry; under TP every rank needs the artifact at this path (rank {})", tp_rank.map_or(0, |r| r as usize));
                }
            }
        }
        None => {
            // EXL27C: the unsharded round kernel's ring band is structurally 8 rows
            // (write_kv_b ring rows [RING, RING+8)); block 16 is the SHARDED (TP) shape.
            // Unsharded TP=1 boots clamp to 8 with one clear line (sanitizer: gqa_attn_band_ring_b
            // OOB __shared__ writes at block 16 unsharded, 2026-10-16).
            if !df2_shard && df2_block > 8 {
                crate::dflash2::set_block(8).ok();
                println!("[exl3-serve] df2: --df2-block {df2_block} without --df2-round-shard runs block 8 unsharded (the round kernel's band width)");
                df2_block = 8;
            }
            // sha_pin: None = the built-in published pin (load_tp_pinned applies it), the
            // trunk's --sha256 handling exactly (Some("off") disables, Some(hex) pins).
            let pin = arg(args, "--sha256");
            let dir = df2_draft_dir.clone().unwrap_or_default();
            match df2_load_round(&mut model, &dir, max_seq_len, pin) {
                Ok(mut r) => {
                    // S5F parity (batch.rs:1682): capture the draft-round CUDA graph once at boot.
                    // The eager EXL3 chain measured 149 ms/round (df2-trace, 2026-10-18, .12);
                    // the graph replay collapses launch/sync overhead. Opt-out: --no-df2-graph.
                    if crate::opts::var(crate::opt!("no-df2-graph")).is_err() {
                        if r.capture_round_graph() {
                            println!("[exl3-serve] df2: draft-round CUDA graph captured (eager fallback via --no-df2-graph)");
                        } else {
                            println!("[exl3-serve] df2: draft-round graph capture unsupported — staying eager");
                        }
                    }
                    let sha = crate::dflash2::load::last_load_sha().unwrap_or("unverified");
                    println!("{}", df2_boot_line_engaged(&dir, sha, df2_block, df2_shard, tp_world as usize));
                    df2_mark_engaged();
                    df2 = Some(r);
                }
                Err(e) => println!("{}", df2_boot_line_fallback(&format!("load failed: {e:#}"))),
            }
        }
    }
    // FLASHNEXT-DF2-B: `--df2-k <n>` caps the DF2 verify width (the walk still drafts block-1
    // tokens; only the first n are verified). Default = block-1. Applied and shown, never a
    // refusal (AGENTS §1b-bis.8): out-of-range values clamp into 1..=block-1.
    let df2_k_max = (df2_block - 1).min(crate::exl3_forward::MTP_MAX_K);
    let df2_k = match arg(args, "--df2-k") {
        None => df2_k_max,
        Some(v) => {
            let want: usize = v.parse().unwrap_or(df2_k_max);
            let k = want.clamp(1, df2_k_max);
            if df2.is_some() {
                println!("[exl3-serve] df2: verify width k={k} (--df2-k {v}{}; default {df2_k_max} = block-1; the chain still drafts {} and verifies the first k)",
                         if k != want { ", clamped" } else { "" }, df2_block - 1);
            }
            k
        }
    };
    // SURFACE2 item 4: the boot lines that describe HOW this server speculates print only when it
    // does. With --mtp off (or the hatch) the WP23 DDS line, the lane-order line, the shared-step
    // line and the spec-sampling line described rounds that never run.
    let spec_live = spec_live(model.mtp.is_some(), crate::exl3_forward::mtp_off_by_opt());
    if let Some(line) = head_loaded_line(model.mtp.is_some(), crate::exl3_forward::mtp_off_by_opt()) {
        if crate::boot_report::debug() {
            crate::dprintln!("{line}");
        }
    }
    let st_tok = crate::boot_report::Stage::begin("tokenizer");
    let tok = QwenTokenizer::from_file(&format!("{}/tokenizer.json", model_dir.trim_end_matches('/')))?;
    st_tok.ok();
    // Stop tokens = the §3 union: generation_config eos (int or list) + tokenizer/config eos.
    let cfg_eos: u32 = {
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/config.json", model_dir.trim_end_matches('/')))?,
        )?;
        cfg.get("eos_token_id").and_then(|e| e.as_u64()).unwrap_or(248046) as u32
    };
    let eos = tok.stop_token_ids(cfg_eos);
    crate::dprintln!("[exl3-serve] stop_ids={eos:?}");

    let mtp_k = crate::exl3_forward::mtp_depth_default();
    // API-parity DDS: dynamic draft stop (the reference implementation's -dds -dc 0.6). CLI --draft-confidence <p>;
    // 0 = fixed-depth chain. Output-neutral (greedy lossless, sampled distribution-exact).
    // Default 0.4, not the reference implementation's 0.6: our draft pass is cheaper relative to a verify row, so
    // longer drafts pay (sweep 2026-09-25, k5 greedy 1024 tok: 0.4 -> ansic 87.9 / python 77.6 /
    // prose 55.8; 0.6 -> 85.2 / 74.2 / 55.6; 0.2-0.7 all within +-2).
    let draft_conf: f64 = arg(args, "--draft-confidence").and_then(|s| s.parse().ok()).unwrap_or(0.4);
    anyhow::ensure!((0.0..1.0).contains(&draft_conf), "--draft-confidence must be in [0, 1)");
    // WP08 (owner decision 2026-09-26): the reference implementation's streaming loop detector, ON by default at its
    // launcher setting stop_on_loop = (300, 3) (chat.py -lw 300 -lmr 3): a response whose last
    // 300 tokens are one repeating sequence of period <= 100 ends as a normal stop
    // (stop_reason "loop_detected"). --loop-detect off disables; --loop-window / --loop-min-reps
    // tune it (the reference implementation's asserts: window > 1, 1 < reps < window). It only changes output when
    // the model is already looping.
    let loop_cfg = match arg(args, "--loop-detect") {
        None | Some("on") => {
            let w: usize = match arg(args, "--loop-window") {
                None => 300,
                Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--loop-window must be an integer"))?,
            };
            let r: usize = match arg(args, "--loop-min-reps") {
                None => 3,
                Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--loop-min-reps must be an integer"))?,
            };
            anyhow::ensure!(w > 1, "--loop-window must be > 1");
            anyhow::ensure!(r > 1 && r < w, "--loop-min-reps must be > 1 and < --loop-window");
            Some((w, r))
        }
        Some("off") => None,
        Some(v) => anyhow::bail!("--loop-detect must be on or off (got '{v}')"),
    };
    // WP15: penalty window, the reference implementation's -penr (penalty_range, default 1024) used as both its
    // full-strength (sustain) and fading (decay) spans. Penalties themselves are per request.
    let pen_range: usize = match arg(args, "--penalty-range") {
        None => 1024,
        Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--penalty-range must be an integer"))?,
    };
    anyhow::ensure!(pen_range <= crate::exl3_forward::PEN_RANGE_MAX,
                    "--penalty-range must be <= {}", crate::exl3_forward::PEN_RANGE_MAX);
    // WP24 (owner-consent class: seeded sampled bytes change; default match = today bit for bit).
    let (spec_ratio, draft_temp) = parse_spec_sampling(args)?;
    // Server-wide penalty defaults for requests that send none (the shared server flags; the
    // reference implementation's defaults 1.0 / 0 / 0 when absent). Same validity rules as a request's values.
    let dflt = |name: &str, d: f32| -> Result<f32> {
        match arg(args, name) {
            None => Ok(d),
            Some(v) => v.parse::<f32>().map_err(|_| anyhow::anyhow!("{name} must be a number")),
        }
    };
    let default_rep_penalty = dflt("--default-repetition-penalty", 1.0)?;
    let default_presence_penalty = dflt("--default-presence-penalty", 0.0)?;
    let default_frequency_penalty = dflt("--default-frequency-penalty", 0.0)?;
    crate::server::validate_penalties(Some(default_rep_penalty), Some(default_presence_penalty),
                                      Some(default_frequency_penalty), None)
        .map_err(|e| anyhow::anyhow!("server penalty defaults: {e}"))?;
    crate::dprintln!("[exl3-serve] loop detector {} ; penalty window {pen_range} + {pen_range} tokens \
              (defaults rep {default_rep_penalty} presence {default_presence_penalty} frequency {default_frequency_penalty})",
             match loop_cfg { Some((w, r)) => format!("ON (window {w}, min reps {r})"), None => "OFF".into() });
    // S-B9-REL-INTEG (AGENTS 1b-bis): a penalized request never joins a packed verify, and a request that sends no
    // penalty field INHERITS these server defaults. Say so at boot, where the operator chose the flags.
    if lane_pack.on() {
        if let Some(w) = lane_pack_penalty_warning(default_rep_penalty, default_presence_penalty, default_frequency_penalty) {
            crate::drprintln!("{w}");
        }
    }
    let tok = Arc::new(tok);
    // BUG-5 FIX (width half): mtp_round verifies at m=k+1 rows — the scratch must
    // hold the verify width even when the lane count is 1 (w1 serve panicked on
    // sc.slots[2] with a 2-element buffer). Model width (state slots) stays = lanes.
    // WIDEVERIFY1 refinement 2: the wide copy verify (--copy-span on with --copy-span-max-w > MTP_MAX_K, armed
    // on this world) needs a 16-row scratch whose ring / taps planes and accept words hold 16 verify rows;
    // otherwise MTP_MAX_K + 1 rows: the scratch is exactly the pre-refinement one.
    let vrows = copy_span_scratch_vrows(crate::opts::on(crate::opt!("copy-span")),
                                        if tp_rank.is_some() { (tp_world as usize).max(2) } else { 1 }, spec_live,
                                        crate::opts::ranged(crate::opt!("copy-span-max-w")).map_or(0, |v| v as usize));
    let mut sc = FwdModel::scratch_vrows(model.dev(), &model.cfg, width.max(mtp_k + 1).max(vrows), model.max_pos(), vrows)?;
    // S-A3-f-b: chunked prefill width (CLI --prefill-chunk; user-facing knob).
    // S-A3-f-d Item 2: default 512 -> 2048 — the matched-C table (7838 toks,
    // fixed flash256 kernel) shows 112 tok/s @ C=512 vs 234 @ C=2048 (the KV
    // re-scan term ~N^2/2C); serve == bench at matched C, so the chunk width is
    // the whole serve-side lever. Scratch cost is ~4x of a few hundred MB.
    // v0.7.1 (owner 2026-10-02, D-TP4-6): default 2048 -> 4095 (TP-4X1: W=4 prefill -3% at 8K, -5.4% at 32K).
    let chunk: usize = arg(args, "--prefill-chunk").and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_PREFILL_CHUNK);
    if tp_rank.is_some() {
        anyhow::ensure!(psc_rows(chunk.max(1), true) <= crate::exl3_forward::xtp::TP_MAX_ROWS,
                        "TP: --prefill-chunk {chunk} gives prefill chunks up to {} rows, above the TP all-reduce \
                         partial buffer ({} rows; C1 tail checkpoints merge up to min(2C-1, {}) rows)",
                        psc_rows(chunk.max(1), true), crate::exl3_forward::xtp::TP_MAX_ROWS,
                        crate::exl3_forward::xtp::TP_MAX_ROWS - 1);
    }
    // S-A3-u: build the prefill scratch at startup and warm cuBLASLt (lazy kernel loads + plans
    // ~0.4 s) so request 1 does not pay it. --exl3-no-prefill-warmup skips (diagnostics).
    let mut psc0 = model.prefill_scratch(psc_alloc_rows(chunk.max(1), tp_rank.is_some()))?;
    crate::exl3_forward::dres::init_from_opts();
    crate::exl3_forward::dres::mid_init_from_opts();
    if crate::opts::var(crate::opt!("exl3-no-prefill-warmup")).is_err() {
        model.prefill_warmup(&mut psc0, 0, &crate::exl3_forward::FwdModel::prefill_warmup_widths(chunk.max(1)))?;
    }
    // LANEWIDE-DRES: the warm-up ran every prefill width class, so the kept dense matrices are in place now
    if let Some(line) = crate::exl3_forward::dres::boot_line() { crate::rprintln!("{line}"); }
    // WP02 --exit-on-fatal (default ON since v0.7.3, owner decision 2026-10-06; was opt-in):
    // a sticky CUDA error or a scheduler panic exits 70 so a supervisor (serve_ours.sh SUPERVISE=1)
    // restarts the server. --exit-on-fatal off keeps the old behaviour: the engine goes DEAD and
    // /health + every request answer 503 until restarted. Registered Bool: bare flag = on.
    let exit_on_fatal = crate::opts::var(crate::opt!("exit-on-fatal")).map_or(true, |v| v != "0");
    let inject_panic: Option<usize> = crate::opts::var(crate::opt!("inject-panic-steps")).ok().and_then(|v| v.parse().ok());
    if let Some(n) = inject_panic {
        crate::deprintln!("[exl3-serve] DIAGNOSTIC: --inject-panic-steps={n} — injected scheduler fault at decode step {n}");
    }
    let _ = crate::exl3_forward::a_flags(); // log any GB10_A_* diagnostic escape at boot
    // WP04: --graph-precapture on|off (default on) — capture every decode graph the served config
    // reaches (step, verify widths, draft passes / chain, both attention regimes, every lane slot)
    // BEFORE listening, against the scheduler's own scratch, so no request pays a lazy capture.
    // Capture-only (WP04-v2): no kernel runs, so every device byte stays as an `off` boot has it.
    let precapture = match arg(args, "--graph-precapture") {
        None | Some("on") => true,
        Some("off") => false,
        Some(other) => anyhow::bail!("--graph-precapture must be on or off (got '{other}')"),
    };
    if precapture {
        model.precapture_decode_graphs(&mut sc, width, mtp_k, draft_conf > 0.0)?;
    } else {
        crate::dprintln!("[exl3-serve] graph precapture OFF (--graph-precapture off): decode graphs capture lazily");
    }
    // S-B9-CF-PACK4 `--lane-pack on`: the packed verify's own 16-row scratch and its graph family
    // (every exact width R = 4..16, each reachable regime), captured NOW — the served arm never
    // captures a packed graph while serving. With the flag off nothing here runs: no scratch,
    // no graph, and every device byte is master's (the host still prints one boot line and renders
    // the lane-pack /metrics series, all zero).
    // F1 (owner 2026-10-10): `--lane-pack on` never refuses a boot. Every reason it cannot pack on
    // this boot has been collected into `pack_blockers` (the posture fallbacks, `--max-batch 1`,
    // `--graph-precapture off`) plus `pack_supported`'s kernel/model postures here. A blocked boot
    // flips the flag OFF before anything packed runs — no packed scratch, no packed graph, the
    // served path is master's bit for bit — prints ONE plain line, and records the decision for
    // /metrics and the TP boot agree.
    if lane_pack.on() {
        if width < 2 {
            pack_blockers.push(format!("--max-batch {width}: a pack steps two requests per round (needs --max-batch >= 2)"));
        }
        if !precapture {
            pack_blockers.push("--graph-precapture off: the packed graphs are captured at boot, never while serving".to_string());
        }
        if let Err(e) = model.pack_supported() {
            pack_blockers.push(e.to_string());
        }
        if !pack_blockers.is_empty() {
            let why = pack_blockers.join("; ");
            crate::drprintln!("[exl3-serve] lane packing (--lane-pack, on by default) is NOT ACTIVE on this boot: {why}; requests are served without packed rounds (--lane-pack off silences this line)");
            crate::lane_pack::record_boot_pack(Some(why.clone()));
            lane_pack = crate::lane_pack::LanePack::Off;
        }
    }
    // F10: building the packed scratch and its graph family is the only allocation the default adds to a boot. It
    // must never abort one: any failure here (device memory, a capture) is a FALLBACK reason, the half-built
    // scratch is dropped, and the decision is recorded AFTER the build (the TP boot agree hashes it).
    let (scp, pack_ready, pack_sampled) = if lane_pack.on() {
        // Do sampled lanes pack on this boot? TP=1: yes (PACK4S). Under TP: `--lane-pack-sampled-tp`
        // (default on; off = a sampled lane runs its own lone round, counted `skip sampled`).
        let sampled_ok = tp_rank.is_none() || pack_sampled_under_tp();
        let fams = crate::lane_pack::fams(spec_ratio, sampled_ok);
        let free = || cudarc::driver::result::mem_get_info().map(|x| x.0).unwrap_or(0) as f64;
        let f0 = free();
        let built = crate::lane_pack::build_or_fall_back(|| {
            let mut scp = model.pack_scratch()?;
            model.dev().synchronize()?;
            let f1 = free();
            // S-B9-CF-PACK4S: under --spec-sampling ratio (the default) a sampled lane's verify rows
            // are real-q rows, so the boot also holds the rq family (the same widths and regimes);
            // under the match rule the greedy family serves sampled rows at TP=1 (a family of its own
            // under TP: a different verify tail).
            let tcap = std::time::Instant::now();
            let (n, graph_mb, note) = model.pack_precapture(&mut scp, &fams)?;
            let cap_s = tcap.elapsed().as_secs_f64();
            let fam_ready = |sparse: bool| fams.iter().all(|&f| model.pack_family_ready(&scp, sparse, f));
            let ready = [fam_ready(false), fam_ready(true)];
            anyhow::ensure!(ready[0], "the dense packed graph family is incomplete after the boot capture");
            Ok((scp, ready, n, graph_mb, note, cap_s, f1))
        });
        match built {
            Err(why) => {
                crate::drprintln!("[exl3-serve] lane packing (--lane-pack, on by default) is NOT ACTIVE on this boot: {why}; requests are served without packed rounds (--lane-pack off silences this line)");
                crate::lane_pack::record_boot_pack(Some(why));
                lane_pack = crate::lane_pack::LanePack::Off;
                (None, [false, false], false)
            }
            Ok((scp, ready, n, graph_mb, note, cap_s, f1)) => {
                crate::lane_pack::record_boot_pack(Some(String::new()));
                if tp_rank.is_some() {
                    crate::drprintln!("[exl3-serve] lane-pack under TP: sampled requests {} (--lane-pack-sampled-tp {})",
                                     if sampled_ok { "JOIN packed rounds (the pack ran them at TP=2 on one pair, 2026-10-10; TP=4 was exercised by hand, no gate chain)" } else { "run alone; only greedy requests pack" },
                                     if sampled_ok { "on" } else { "off" });
                }
                let f2 = free();
                let fam_names: Vec<&str> = fams.iter().map(|f| f.label()).collect();
                crate::drprintln!("[exl3-serve] lane-pack ON (the default; --lane-pack off turns it off){}: two requests share ONE packed speculative verify per round \
(three busy = one pair + one rotating lone round); {}; penalized requests and requests with images do not; \
draft chain: {}; packed scratch {:.0} MB + {n} packed verify graphs captured in {cap_s:.2} s, {graph_mb:+.0} MB \
(exact widths R = {}..{}, families: {}; {note}), {:.2} GiB of device memory in all (the graph delta is cuMemGetInfo-noisy; \
the free memory that matters is the 'after the boot' figure of the PLE line below)",
                         if tp_rank.is_some() { format!(" at TP={tp_world} rank {}", tp_rank.unwrap_or(0)) } else { String::new() },
                         if !sampled_ok { "GREEDY requests pack; SAMPLED requests run their own lone round (the sampled tail of a packed verify is not gated under TP)".to_string() }
                         else if crate::wp24::dump_on() { "greedy and sampled requests pack, in any combination (--wp24-dump is set: sampled requests stay on the lone path)".to_string() }
                         else if spec_ratio { "greedy and SAMPLED requests pack, in any combination (sampled rows verify by the real-q ratio rule, per-row sampler blocks)".to_string() }
                         else { "greedy and SAMPLED requests pack, in any combination (sampled rows verify by the match rule, per-row sampler blocks)".to_string() },
                         if draft_conf > 0.0 { "dynamic draft stop" } else { "FIXED depth (--draft-confidence 0)" },
                         (f0 - f1) / 1e6, crate::lane_pack::PACK_WIDTHS.start(), crate::lane_pack::PACK_WIDTHS.end(),
                         fam_names.join(" + "), (f0 - f2) / (1u64 << 30) as f64);
                // S-B9-CF-PACK5: the packed graph set, named on EVERY rank (head and node print this line);
                // the TP boot agree compares its count and digest (a rank holding a different set refuses
                // to boot instead of letting `pack_plan` decide rank-locally)
                let (gn, gh, gdesc) = model.pack_graph_set(&scp, &fams);
                crate::drprintln!("[exl3-serve] lane-pack packed graph set{}: {gn} graphs [{gdesc}] digest {gh:08x}; dense ready {}, sparse ready {}",
                         if tp_rank.is_some() { format!(" (TP={tp_world} rank {})", tp_rank.unwrap_or(0)) } else { String::new() },
                         ready[0], ready[1]);
                (Some(scp), ready, sampled_ok)
            }
        }
    } else {
        (None, [false, false], false)
    };
    // PFX1 (e): after the boot captures — the cuBLASLt plans of every prefill tail length are
    // built in the background (host-only work; requests are served meanwhile, lazily as before).
    let _ = model.lt_prewarm_start();
    // WP16: intermediate recurrent checkpoints — memory accounting at boot (allocated lazily).
    let wp16 = {
        let lay = model.ckpt_layout();
        let per = lay.bytes() as u64;
        let cap_bytes = (wp16_gb * (1u64 << 30) as f64) as u64;
        let cap = if per > 0 { (cap_bytes / per) as usize } else { 0 };
        if !prefix_on {
            crate::dprintln!("[exl3-serve] WP16 prefix checkpoints OFF (--prefix-cache off)");
            None
        } else if crate::exl3_forward::wp16_off() {
            crate::dprintln!("[exl3-serve] WP16 prefix checkpoints OFF (--wp16-off=1 — diagnostic escape: the prompt-end \
                      snapshot only, today's behaviour)");
            None
        } else if cap == 0 {
            crate::dprintln!("[exl3-serve] WP16 prefix checkpoints OFF (--prefix-ckpt-mem-gb {wp16_gb} holds no {:.1} MB checkpoint)",
                     per as f64 / 1e6);
            None
        } else {
            crate::dprintln!("[exl3-serve] WP16 prefix checkpoints ON: {}; cap {wp16_gb} GiB (--prefix-ckpt-mem-gb; default 4 is a \
                      placeholder — the owner sets the final default) = {cap} checkpoint(s) = {:.2} GiB at most, \
                      allocated lazily, LRU across slots, kept across a conversation's cache-hit turns. Taken at chunk \
                      ends of aligned prefills (grid from 0 or from an aligned checkpoint; never after a prompt-end \
                      prefix hit): end % {} == 0, plus every C={chunk} boundary in a prefill's last 2 chunks; keyed \
                      by C. Resumed on a prompt-end miss from the deepest checkpoint <= min(LCP, plen-1) \
                      (a 128K document holds ~18 = ~{:.1} GiB)",
                     lay.describe(), (cap as u64 * per) as f64 / (1u64 << 30) as f64,
                     crate::exl3_forward::WP16_STRIDE, 18.0 * per as f64 / (1u64 << 30) as f64);
            Some(crate::exl3_forward::CkptStore::new(width, cap))
        }
    };
    // C1: tail (message-boundary) checkpoints — opt-in, output-changing (registry prefix.tail_ckpt, N).
    let tail_ckpt_boot = wp16.is_some() && crate::exl3_forward::tail_ckpt_on();
    if tail_ckpt_boot {
        crate::dprintln!("[exl3-serve] C1 tail checkpoints ON (prefix.tail_ckpt = {} / --prefix-tail-ckpt=1|2: {}; OUTPUT-CHANGING, \
                  owner consent): every prefill also checkpoints at its message boundary (the prompt without its \
                  generation prompt) minus {} tokens when that saves >= {} tokens over the C grid point below it; runs \
                  are realigned to the C grid; the next chat turn resumes there",
                 if crate::exl3_forward::tail_ckpt_extra_chunk() { 2 } else { 1 },
                 if crate::exl3_forward::tail_ckpt_extra_chunk() {
                     "also splits costing one extra chunk when they promise >= 1024 tokens"
                 } else { "only splits absorbed into an existing chunk boundary" },
                 crate::exl3_forward::TAIL_BACKOFF, crate::exl3_forward::TAIL_MIN_GAIN);
    } else if crate::exl3_forward::tail_ckpt_on() {
        crate::dprintln!("[exl3-serve] C1 tail checkpoints requested but WP16 checkpoints are off — ignored");
    }
    let wp16_xcheck = wp16.is_some() && crate::exl3_forward::wp16_xcheck_on();
    if wp16_xcheck {
        crate::dprintln!("[exl3-serve] WP16 XCHECK ON (--wp16-xcheck=1, diagnostic): every checkpoint resume also prefills \
                  fresh from 0 into a scratch slot ({}) and bit-compares GDN S / conv / PLE ring + hist / head-fill \
                  carry and the first-token logits ([wp16-xcheck] lines)",
                 if width > 1 { "a free lane's slot" } else { "--max-batch 1: the resumed slot itself, then the resumed state is restored" });
    }
    // VIS-2: the vision tower (TP=1; TP > 1 serves images from VIS-4). Loaded when config.json declares
    // a vision_config and the text config an interleaved mrope; a load failure serves text-only with a
    // visible notice (image requests then answer 400).
    // VIS-4: the head (TP=1, or rank 0) runs the tower; a node only needs the mrope section (it splices
    // the rows the head ships).
    let (vision_tower, vision_gpu, mrope_section) = if tp_rank.map_or(true, |r| r == 0) {
        // PR #16 drill flag: --vision-fail-gemm (one-shot GEMM fault; see vision_gemm_drill).
        vision_boot(model_dir, vision_fail_gemm_armed())
    } else {
        let sec = mrope_section_of(model_dir);
        crate::dprintln!("[exl3-serve] vision (node): mrope section {sec:?}; image rows arrive from the head");
        (None, None, sec)
    };
    // CF-P1d: every lane's KV, the scratch, graphs and the vision tower now exist — `--ple-ram auto` decides RAM vs
    // SSD for the n-gram table on the memory that is actually left (each rank for itself; identical output either way).
    model.ple_promote_auto()?;
    let spec_policy = SpecLanes::parse(crate::opt_or_arg(args, "--spec-lanes-max").as_deref())?;
    // CF-FCFS: --lane-order reaches every rank through the SPMD option registry (Scope::Spmd), so
    // the head's value is installed on the nodes before the scheduler is built. The `arg()` read is
    // the TP=1 path's own value; under TP both ranks see the head's shipped value (identical).
    let lane_order = LaneOrder::parse(crate::opt_or_arg(args, "--lane-order").as_deref()
                                      .map(str::trim)
                                      .filter(|v| !v.is_empty()))?;
    let lane_quantum = match crate::opt_or_arg(args, "--lane-quantum").as_deref() {
        None => 256,
        Some(v) if v.trim().is_empty() => 256,
        Some(v) => v.trim().parse::<usize>().map_err(|e| anyhow::anyhow!("--lane-quantum: {e}"))?,
    };
    anyhow::ensure!(lane_quantum > 0, "--lane-quantum must be > 0");
    anyhow::ensure!(!(lane_pack.on() && lane_order == LaneOrder::Fcfs),
                    "--lane-pack on needs --lane-order rr: fcfs steps ONE lane per scheduler step, a pack steps two");
    // --prefill-interleave N (public PR #23; default 0 = off = no code path change). Every TP rank
    // parses the same argv and resolves 0, so the node needs nothing and the lockstep is untouched.
    let (interleave, il_line) = resolve_prefill_interleave(
        // PERFFIX1: from the registry (SPMD: a node gets the head's value; its own argv never carries the flag)
        // owner 2026-10-24: default 1 — an unset flag takes the REGISTRY default (never a hard-coded 0 that would drift)
        parse_prefill_interleave(Some(crate::opts::var(crate::opt!("prefill-interleave")).ok()
            .unwrap_or_else(|| crate::opt!("prefill-interleave").def().def.to_string()).as_str()))?, tp_rank.is_some(), chunk);
    if let Some(line) = il_line { crate::drprintln!("{line}"); }
    if width > 1 && spec_live {
        crate::dprintln!("[exl3-serve] load-adaptive batching (--spec-lanes-max): {} (--max-batch {width})", match spec_policy {
            SpecLanes::Auto => "auto — each round picks serial speculation or ONE shared plain step by estimated aggregate tokens/s".to_string(),
            SpecLanes::Over(k) => format!("one shared plain step with more than {k} busy lanes, serial speculation otherwise"),
            SpecLanes::Never => "off — always serial speculative rounds".to_string(),
        });
    }
    // S-B9-REL-MTPREPROBE: --mtp-reprobe=off restores the previous one-way behaviour;
    // --mtp-reprobe-drill shrinks the schedule for identity/stress runs.
    let reprobe_on = crate::opts::var(crate::opt!("mtp-reprobe")).map_or(true, |v| v != "off");
    let reprobe_cfg = if crate::opts::var(crate::opt!("mtp-reprobe-drill")).is_ok() {
        crate::mtp_reprobe::ReProbeCfg::drill()
    } else {
        crate::mtp_reprobe::ReProbeCfg::default()
    };
    // LANERES1: --lane-reserve R --lane-reserve-patience P --lane-reserve-swap S. LANERES3:
    // the policy is now head-authoritative at every topology — under TP the head runs
    // `lane_reserve::plan` in its admit sweep and ships each admit's scoping on the wire
    // (`WireRequest::reserve`), so the node restricts its lane pick identically and both
    // ranks hold the same slot tables. Default R=0 = today's admission, byte-for-byte.
    let lr_r = crate::opts::ranged(crate::opt!("lane-reserve")).map_err(|e| anyhow::anyhow!("{e}"))? as u64;
    let lr_p = crate::opts::ranged(crate::opt!("lane-reserve-patience")).map_err(|e| anyhow::anyhow!("{e}"))? as u64;
    let lr_s = crate::opts::ranged(crate::opt!("lane-reserve-swap")).map_err(|e| anyhow::anyhow!("{e}"))? as u64;
    let reserve_cfg = if lr_r > 0 {
        Some(crate::lane_reserve::Cfg::on(lr_r, lr_p, lr_s))
    } else {
        None
    };
    // LANERES2: --kv-spill <auto|off|PATH> --kv-spill-gb N. An SSD spill tier for evicted lane
    // snapshots: a hard eviction is captured to NVMe (async writer thread) and restored on the
    // conversation's next arrival when the read beats the re-prefill (kv_spill::Policy::lookup).
    // Default OFF since 2026-10-23 (owner: a snapshot copies the whole lane, so it cost more than it saved on typical
    // prompts; ISSUES 123); auto = /tmp/gb10_kv_spill; --kv-spill-gb 0 or "off" = no tier.
    // LANERES3: active at TP>=2 as well — every rank runs the same capture/restore hooks in
    // lockstep, writes ITS rank's shard to ITS OWN local SSD (the file name carries the rank;
    // snapshot bytes never cross the wire), and the outcome is agreed across ranks with one
    // SpillFlag rendezvous, so the replicated policy — and with it every later restore
    // decision — can never diverge on a disk fault (a key any rank cannot back is dropped
    // everywhere and counted).
    // owner 2026-10-23: default OFF (the registry default; read from the registry so the two cannot drift apart)
    let ks_mode = crate::opts::var(crate::opt!("kv-spill")).map(|v| v.to_string())
        .unwrap_or_else(|_| crate::opt!("kv-spill").def().def.to_string());
    let ks_gb = crate::opts::ranged(crate::opt!("kv-spill-gb")).map_err(|e| anyhow::anyhow!("{e}"))? as u64;
    let (spill_on, spill_dir) = if ks_mode == "off" || ks_gb == 0 {
        (false, String::new())
    } else {
        let d = if ks_mode == "auto" { "/tmp/gb10_kv_spill".to_string() } else { ks_mode.clone() };
        (true, d)
    };
    let (spill_store, mut spill_pol) = if spill_on {
        let st = crate::kv_spill::Store::spawn_with(std::path::PathBuf::from(&spill_dir),
            std::cmp::min(std::cmp::max((ks_gb << 30) / 4, 4 << 30), 6 << 30))
            .map_err(|e| anyhow::anyhow!("kv-spill: cannot create {spill_dir}: {e}"))?;
        (Some(st), crate::kv_spill::Policy::new(ks_gb << 30))
    } else {
        (None, crate::kv_spill::Policy::new(0))
    };
    crate::rprintln!("{}", if spill_on {
        let tp_note = if tp_rank.is_some() {
            "; TP: per-rank local shards, capture/restore agreed across ranks in-round"
        } else { "" };
        format!("[exl3-serve] kv spill: on, cap {ks_gb} GiB, dir {spill_dir} (rank {}; text and image prompts; waits briefly for a capture still in the writer queue; restores when the NVMe read beats the re-prefill it replaces){tp_note}", tp_rank.unwrap_or(0))
    } else {
        "[exl3-serve] kv spill: off (the default; --kv-spill auto or a directory turns it on)".to_string()
    });
    // SPILLGC: self-clean the spill directory at boot so snapshots from previous boots cannot accumulate past
    // `--kv-spill-gb`. On single-box (TP=1), adopt valid v6 snapshots into the policy, sweep garbage/temps, and
    // cap-evict in SPILLDUR2 order. Under TP, sweep all prior files for this rank without adopting: adopting local
    // files independently per rank would fork the replicated policy if one node's `/tmp/gb10_kv_spill` was wiped
    // on reboot while a peer's survived, causing a `spill_agree` deadlock on the first divergent lookup.
    if let Some(st) = &spill_store {
        const GIB: f64 = (1u64 << 30) as f64;
        let rk = tp_rank.unwrap_or(0) as usize;
        let rep = if tp_rank.is_some() {
            st.boot_sweep_all(rk)
        } else {
            let scan = st.boot_scan(rk, model.max_pos() as u64, model.ckpt_bytes());
            st.boot_finish(&mut spill_pol, rk, scan)
        };
        if rep.adopted + rep.swept + rep.evicted > 0 {
            crate::rprintln!("[exl3-serve] kv spill: boot sweep (rank {rk}): adopted {} snapshots ({:.1} GiB), swept {} stale/garbage files ({:.1} GiB), cap-evicted {} ({:.1} GiB)",
                rep.adopted, rep.adopted_bytes as f64 / GIB,
                rep.swept, rep.swept_bytes as f64 / GIB, rep.evicted, rep.evicted_bytes as f64 / GIB);
        }
    }
    // PFXPOOL: --prefix-pool-gb N (Spmd: identical on every rank) — prompt states kept in device memory across lanes;
    // --prefix-pool-floor-gb F: the MemAvailable every rank keeps (a capture that would go below it evicts, then is
    // dropped). Off with --prefix-cache off (no snapshots to keep) or N = 0.
    let pool_gb = crate::opts::ranged(crate::opt!("prefix-pool-gb")).map_err(|e| anyhow::anyhow!("{e}"))? as u64;
    let pool_floor_gb = crate::opts::ranged(crate::opt!("prefix-pool-floor-gb")).map_err(|e| anyhow::anyhow!("{e}"))? as u64;
    let pool_on = prefix_on && model.prefix_snapshots() && pool_gb > 0;
    let pool = crate::prefix_pool::Pool::<PoolBuf>::new(if pool_on { pool_gb << 30 } else { 0 });
    let pool_floor = pool_floor_gb << 30;
    crate::metrics::pfx_pool_state(pool.cap_bytes(), 0, 0);
    crate::rprintln!("{}", if pool_on {
        let free = crate::memwatch::mem_available_bytes().map_or("?".to_string(), |b| format!("{:.1}", b as f64 / (1u64 << 30) as f64));
        let per_k = model.pool_entry_bytes(1024).map(|b| b as f64 / (1u64 << 20) as f64).unwrap_or(0.0);
        format!("[exl3-serve] prefix pool: on, up to {pool_gb} GiB of prompt states shared by all lanes (keeps {pool_floor_gb} GiB \
                 free; {free} GiB free now; ~{per_k:.0} MiB per 1K-token prompt on this rank; text and image prompts; --prefix-pool-gb 0 = off)")
    } else if !prefix_on {
        "[exl3-serve] prefix pool: off (--prefix-cache off)".to_string()
    } else {
        "[exl3-serve] prefix pool: off (--prefix-pool-gb 0)".to_string()
    });
    // WIDEVERIFY1: --copy-span (one boot line; under TP / without speculation it runs off, never refuses)
    let copy_arm = copy_span_resolve(if tp_rank.is_some() { (tp_world as usize).max(2) } else { 1 }, spec_live)?;
    // COPYPF1: --copy-span-prefill (default off; one boot line; under TP / without the copy arm / with the DFlash2
    // chain it runs off, counted, never refuses)
    let cpv_st = crate::copy_prefill::resolve(crate::opts::on(crate::opt!("copy-span-prefill")), copy_arm.is_some(),
                                              if tp_rank.is_some() { (tp_world as usize).max(2) } else { 1 }, df2.is_some());
    crate::copy_prefill::set_state(cpv_st);
    crate::copy_prefill::set_max_rows(crate::opts::ranged(crate::opt!("copy-span-prefill-max-rows"))
        .map_or(crate::copy_prefill::CPV_MAX_ROWS_DEFAULT, |v| v as usize));
    crate::copy_prefill::set_min_run(crate::opts::ranged(crate::opt!("copy-span-prefill-min-run")).map_or(0, |v| v as usize));
    crate::copy_prefill::set_priced(!matches!(crate::opts::var(crate::opt!("copy-span-prefill-price")).as_deref(), Ok("0")));
    if let Some(line) = crate::copy_prefill::boot_line(cpv_st, crate::copy_prefill::schedule_top(chunk.max(1), crate::copy_prefill::max_rows()),
                                                        if tp_rank.is_some() { (tp_world as usize).max(2) } else { 1 }) {
        crate::rprintln!("{line}");
    }
    let cpv_on = cpv_st == crate::copy_prefill::State::On;
    // COPYLANES: --copy-span-lanes (default off; one boot line; without the copy arm, with --lane-order fcfs or one lane it
    // runs off, counted, never refuses)
    let cl_st = crate::copy_lanes::resolve(crate::opts::on(crate::opt!("copy-span-lanes")), copy_arm.is_some(),
                                           lane_order == LaneOrder::Fcfs, width);
    let cl_priced = !matches!(crate::opts::var(crate::opt!("copy-span-lanes-price")).as_deref(), Ok("0"));
    crate::copy_lanes::set_state(cl_st, cl_priced);
    if let Some(line) = crate::copy_lanes::boot_line(cl_st, if tp_rank.is_some() { (tp_world as usize).max(2) } else { 1 }, cl_priced) {
        crate::rprintln!("{line}");
    }
    let cl_on = cl_st == crate::copy_lanes::State::On;
    // LANEWIDE: --lane-wide-verify (default off; one boot line; under TP / without MTP / on a pack it is not wired for it
    // runs off, counted, never refuses)
    let wide_st = crate::lane_wide::resolve(crate::lane_wide::Mode::from_opts(), if tp_rank.is_some() { (tp_world as usize).max(2) } else { 1 },
                                            model.wide_unsupported(), df2.is_some());
    crate::lane_wide::set_state(wide_st);
    let wide_rows = crate::opts::ranged(crate::opt!("lane-wide-rows-max")).map_or(128, |v| v as usize);
    crate::lane_wide::set_draft_mode(crate::opts::choice(crate::opt!("lane-wide-draft")) == "serial");
    if let Some(line) = crate::lane_wide::boot_line(wide_st, wide_rows) { crate::rprintln!("{line}"); }
    let wide_sched = crate::lane_wide::WideSched { st: wide_st, rows_max: wide_rows, ..Default::default() };
    // COPYREC1: --copy-span-record (default off; needs the copy arm; one boot line with the record's state and size)
    let record = copy_record_resolve(copy_arm.is_some(), model_dir, tok.vocab_size(),
                                     if tp_rank.is_some() { (tp_world as usize).max(2) } else { 1 });
    // TURNRESUME: --turn-resume / --next-turn-prefill (both default off; one boot line; never a refusal)
    let turn_arm = turn_resolve(&model, width, prefix_on, tp_rank.is_some(), df2.is_some(), pool_floor);
    let sched = Exl3Scheduler {
        model: model.clone(), sc, width,
        lanes: (0..width).map(|_| None).collect(), eos: eos.clone(), chunk, interleave, psc: Some(psc0),
        df2, df2_k, copy_arm, record, cpv_on, cpv: None, cpv_price: Default::default(), wide: wide_sched,
        cl_on, cl_priced, cl_price: Default::default(),
        mtp_k, spec_policy, lane_order, lane_quantum, admit_ctr: 0, fcfs_rots: 0,
        lane_pack, scp, pack_ready, pack_sampled, pack_rot: 0, pack_logs: 0, in_prefill: false, pack_stats: PackStats::default(), tr: SchedTrace::default(),
        reprobe_on, reprobe_cfg,
        reserve: reserve_cfg, reserve_st: crate::lane_reserve::State::new(), rel: (0..width).map(|_| None).collect(), lru: vec![0; width],
        spill: spill_store, spill_pol, spill_rank: tp_rank.map(|r| r as usize).unwrap_or(0),
        pool, pool_floor, pool_spare: Vec::new(), slot_segs: Vec::new(), row_gen: 0, pool_emerg_at: None,
        shared_scale: 1.0, shared_last: false, shared_streak: 0, shared_probe_ctr: 0, spec_win_tok: [0.0; 2], spec_win_ms: [0.0; 2], spec_win_n: [0; 2], spec_win_acc: [0; 2], spec_arm_rate: [0.0; 2], spec_arm_n: [0; 2], spec_arm_fold: [0; 2], spec_rate_n: vec![[(0.0, 0); 2]; SPEC_RATE_NMAX + 1], spec_rate_folds: vec![[0; 2]; SPEC_RATE_NMAX + 1], meter_settle: 0, meter_last_n: 0, probe_fold: [false; 2], spec_probe: None, spec_probe_ctr: 0, spec_est_pick: None, spec_margin: 0.0, spec_margin_n: 0, policy_logs: 0, ctrl_round: 0, ema_plain: 0.0, draft_conf, cal: DraftCal::new(),
        prefix_on, cache: (0..width).map(|_| None).collect(),
        wp16, wp16_xcheck,
        loop_cfg, pen_range, spec_ratio, draft_temp, gate: None, tok: tok.clone(),
        tp: tp_rank.map(|r| TpSync::new(r, tp_world as i32)),
        exit_on_fatal, fatal: None, inject_panic, steps: 0, last_resume: (0, 0), pf_done: Default::default(), pf_jobs: Vec::new(), vmm_wait: None, mrope_section, vis_seam: None, kv_pf: None, turn: turn_arm,
    };
    if width > 1 && spec_live {
        crate::dprintln!("[exl3-serve] serial lane order (--lane-order): {}{}", lane_order.name(),
                 if lane_order == LaneOrder::Fcfs {
                     format!(" — run to completion, one lane at a time, FCFS by admission order, \
                              quantum {} generated tokens (--lane-quantum)", lane_quantum)
                 } else {
                     " — every busy lane one round per scheduler step, in slot order (unchanged)".to_string()
                 });
        // after the lane-order line: `speculation_boot_lines_follow_the_switch` pins that line directly behind its guard
        crate::rprintln!("[exl3-serve] packed speculative verify (--lane-pack): {}", lane_pack.name());
    }
    // LANERES1: one line for the lane-reservation posture — active at every topology since
    // LANERES3 (head-authoritative under TP), or off.
    match reserve_cfg {
        Some(c) => crate::rprintln!("[exl3-serve] lane reservation: R {} s, patience {} s, swap {} s (--lane-reserve: keep a conversation's lane and its prefix cache; andrei.gnezdilov's design{})",
            c.reserve.as_secs(), c.patience.as_secs(), c.swap.as_secs(),
            if tp_rank.is_some() { "; TP: head-authoritative, the node applies the shipped admit scoping" } else { "" }),
        None => crate::rprintln!("[exl3-serve] lane reservation: off (--lane-reserve 0: today's admission, byte-for-byte)"),
    }
    println!("[exl3-serve] liveness: sticky CUDA error / scheduler thread crash -> {}",
             if exit_on_fatal { "exit 70 (--exit-on-fatal, default)" } else { "engine DEAD, /health 503 (--exit-on-fatal off)" });
    crate::dprintln!("[exl3-serve] prefill chunk width = {chunk} tokens/sweep (wide-M)");
    if let Some(line) = mtp_banner(model.mtp.is_some(), crate::exl3_forward::mtp_off_by_opt(), mtp_k, draft_conf) {
        crate::dprintln!("{line}");
    }
    // WP23: depth 6/7 posture (both parts default on iff k > 5; --exl3-mtp-k=5 = the K5 build).
    if spec_live && draft_conf > 0.0 {
        crate::dprintln!("[exl3-serve] WP23 DDS: depth ceiling k={mtp_k} (max {}), depth-keyed bins {}, cost guard {} \
                  (target = max({draft_conf}, clamp(c_hat*tok/ms, 0.15, 0.8)) after 16 rounds, c_hat seeded 6.0 ms/draft; \
                  --exl3-dds-depth-bins / --exl3-dds-guard = 0|1 force)",
                 crate::exl3_forward::MTP_MAX_K,
                 if wp23_on(crate::opt!("exl3-dds-depth-bins"), mtp_k) { "ON" } else { "off" },
                 if wp23_guard_on(mtp_k) { "ON" } else { "off" });
    }

    let (stx, srx) = tokio::sync::mpsc::unbounded_channel::<crate::batch::BatchRequest>();
    // v0.7.3 (#8.3): --model-name (alias --served-model-name) wins verbatim; the default
    // is the model card's `base_model:` line, else the directory name — the same
    // resolution order as the NVFP4 server (server::resolve_model_name).
    let model_name = crate::server::resolve_model_name(
        arg(args, "--model-name"), arg(args, "--served-model-name"), model_dir);
    crate::metrics::set_max_batch(width);
    // Public issue #24: the KV occupancy gauges — `width` lanes of the SERVED window (after the
    // clamp above). The model load registered the byte counts.
    crate::metrics::kv_pool_init(width, max_seq_len);
    // v0.7.3 ("tel"): the sink lives ONLY where the HTTP API lives — TP=1, or TP rank 0
    // (the head). A node never builds one: it has no HTTP hooks to fill the ring and no
    // sender to drain it, so under TP only the head emits. `--otel-model-id` /
    // `--otel-topology` override the auto values inside OtelSink::new.
    let otel = match (&otel_cfg, tp_rank) {
        (Some(cfg), None) | (Some(cfg), Some(0)) => Some(crate::otel::OtelSink::new(
            cfg.clone(), &model_name,
            // tp_rank (not tp — that was moved into the load) still tells single vs TP.
            &crate::otel::topology_from_world(if tp_rank.is_some() { Some(tp_world) } else { None }))),
        _ => None,
    };
    // FLASHNEXT-DF2-B (ISSUES work[113]): the pack's generation_config.json decides the
    // unsent-field sampler defaults (the Qwen3.8 card values only when the pack has none; a
    // Qwen3.8 pack's file holds exactly the card's thinking triple, so this is identity there).
    let (sampling_defaults, sd_src) = crate::server::SamplingDefaults::from_pack(
        model_dir, crate::server::SamplingDefaults::QWEN38_CARD);
    println!("[exl3-serve] {}", sampling_defaults.describe(sd_src));
    let state = AppState {
        sampling_defaults,
        scheduler: stx,
        tokenizer: tok,
        model_name: model_name.clone(),
        // No output cap in the request = generate until a stop token or the end of the context
        // window (the vLLM contract; was a silent 1024). --max-tokens N is the DEFAULT for a chat
        // request that omits max_tokens, not a cap: server.rs takes
        // `req.max_tokens.unwrap_or(default_max_tokens)` and clamps only to the context room.
        default_max_tokens: arg(args, "--max-tokens").and_then(|s| s.parse().ok()).unwrap_or(usize::MAX),
        default_rep_penalty,
        default_presence_penalty,
        default_frequency_penalty,
        reasoning_effort,
        thinking,
        output_prompts: 0,
        max_seq_len,
        decode_headroom: crate::batch::decode_headroom(false),
        // exl3 snapshots at the prompt end: no ckpt_at render needed — except C1 tail checkpoints,
        // which take one at the message boundary (ckpt_at) of every prompt
        prefix_cache: prefix_on && tail_ckpt_boot,
        // the EXL3 scheduler arms a schema after the think-close token (schema_feed)
        schema_defer: true,
        // PR #30 / #31 ports: logprobs at TP=1 and (owner 2026-11-02) TP=2 / 4 (the full-rows tail is the row
        // gather); the vocabulary mask at TP=1 only (the TP head serves it unconstrained, said once, counted)
        logprobs_ok: tp_rank.is_none() || matches!(tp_world, 2 | 4),
        schema_ok: tp_rank.is_none(),
        vision_tower,
        vision_gpu,
        vision_cpu: false,
        stop_ids: eos,
        otel,
    };
    let df2_engaged = sched.df2.is_some();
    Ok(ServeParts { sched, state, port, cpu_aff, exit_on_fatal, rx: srx,
                    boot_facts: (tp_world, max_seq_len, width, kv_name.to_string(), prefix_on,
                                 model.mtp.is_some(), crate::exl3_forward::mtp_off_by_opt().is_some(),
                                 df2_engaged) })
}
