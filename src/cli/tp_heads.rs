//! TP head and node drivers per family: the EXL3 head/node and serve head, the naive head/node, the DSV4
//! TP server entry, node-side `node_serve_tp`, and their option helpers. The node is zero-config: the
//! head ships model, config, calibration and stop tokens; do not bypass that.

use super::*;

pub(super) fn node_serve_tp(dir: &std::path::Path, head_ip: std::net::IpAddr, mut stream: std::net::TcpStream,
                 tpc: &gb10_inference::tp::TpConfig) -> anyhow::Result<()> {
    use gb10_inference::tp_serve::{recv_serving, send_serving, ServingMsg};
    // ISSUES 153: the bring-up dial/accept loops abort the moment THIS stream dies (head gone).
    gb10_inference::net::set_head_liveness(&stream);
    let mut ctx = gb10_inference::tp::TpContext::bring_up_node(head_ip, tpc.node_rank, tpc.world as i32)?;
    ctx.sanity()?;
    println!("NODE (rank {}/{}) — TP LINK UP (serving mode)", ctx.rank, ctx.world);

    // These are read as ENV at model LOAD ([kv-quant] selects the 4-bit KV cache layout and the
    // q4 attention path at GpuModel::load_from_dir_tp) and inside BatchScheduler::new (graph capture,
    // gpu-sample probes). The head ships its values in the config and the node installs them
    // process-wide BEFORE THE LOAD — this used to sit below load_from_dir_tp, so a serve-mode node
    // silently built a bf16 KV cache + bf16 per-head attention while the head ran q4: the node
    // became the straggler (the "32K anomaly" — 34.6 GB/token of bf16 per-head re-reads at 26K)
    // and every serve-mode "q4" number was really a mixed q4-head/bf16-node number.
    if tpc.no_decode_graphs { gb10_inference::opts::set(gb10_inference::opt!("no-decode-graphs"), "1"); }
    if tpc.fp8_prefill { gb10_inference::opts::set(gb10_inference::opt!("fp8-prefill"), "1"); }
    if tpc.cpu_sample { gb10_inference::opts::set(gb10_inference::opt!("cpu-sample"), "1"); }
    if tpc.no_verify_graph { gb10_inference::opts::set(gb10_inference::opt!("no-verify-graph"), "1"); }
    // The 4-bit KV cache must match on BOTH ranks (the caches are all-reduced-consistent).
    if tpc.kv_quant { gb10_inference::opts::set(gb10_inference::opt!("kv-quant"), "1"); }
    // TurboQuant KV (E4) — same SPMD rule (the cache layout must match on both ranks). Value-
    // based like the loader's [kv-tq] read: "1" = b=2 K, "3" = b=3 K.
    if tpc.kv_tq { gb10_inference::opts::set(gb10_inference::opt!("kv-tq"), if tpc.kv_tq_b3 { "3" } else { "1" }); }
    // k8v4 KV (int8-K + q4-V) — same SPMD rule; mutually exclusive with the two above.
    if tpc.kv_k8v4 { gb10_inference::opts::set(gb10_inference::opt!("kv-k8v4"), "1"); }
    if tpc.kv_k8v8 { gb10_inference::opts::set(gb10_inference::opt!("kv-k8v8"), "1"); }
    // FFN-epilogue fusion (E14) must match on BOTH ranks — one-sided fusion rounds the residual
    // differently and the all-reduce mixes divergent hiddens. None = default ON (both sides).
    match tpc.fuse_residual {
        Some(true) => gb10_inference::opts::set(gb10_inference::opt!("fuse-residual"), "1"),
        Some(false) => gb10_inference::opts::set(gb10_inference::opt!("fuse-residual"), "0"),
        None => {}
    }
    // Device-resident token loop (--device-loop): SPMD-relevant — the mirror replays identical
    // decode_steps, so both ranks must run the same (resident or host-bookkeeping) sequence.
    if tpc.device_loop { gb10_inference::opts::set(gb10_inference::opt!("device-loop"), "1"); }
    // v2 GPU-direct all-reduce receive (--tp-gpu-recv): SPMD-relevant — both ranks must pick the
    // same K2 variant or the barrier gates diverge (K2 on cpu_done vs K2' on the payload tail).
    match tpc.gpu_recv {
        Some(true) => gb10_inference::opts::set(gb10_inference::opt!("tp-gpu-recv"), "1"),
        Some(false) => gb10_inference::opts::set(gb10_inference::opt!("tp-gpu-recv"), "0"),
        None => {}
    }
    // AR landing 2 fused reduce+residual+norm epilogue (--tp-reduce-fuse): SPMD-relevant — the
    // fused kernel replaces the K2 + norm two-launch chain at the mixer/FFN epilogue sites, so both
    // ranks must fuse the same launches or the barrier sequences diverge.
    match tpc.reduce_fuse {
        Some(true) => gb10_inference::opts::set(gb10_inference::opt!("tp-reduce-fuse"), "1"),
        Some(false) => gb10_inference::opts::set(gb10_inference::opt!("tp-reduce-fuse"), "0"),
        None => {}
    }
    // MXFP4-native mode (--mxfp4=on) + the MTP-head allowlist escape hatch: SPMD-relevant — both
    // ranks must build the same OMMA repacks / make the same allowlist decision or the decode and
    // verify chains diverge across the all-reduce. The head's env snapshot ships in the config.
    if tpc.mxfp4 { gb10_inference::opts::set(gb10_inference::opt!("mxfp4"), "1"); }
    if tpc.mxfp4_mtp_native { gb10_inference::opts::set(gb10_inference::opt!("mxfp4-mtp-native"), "1"); }
    // S4F DFlash2 tap capture (--df2-capture): both ranks capture or neither — the sink feeds
    // the drafter's fc; config-shipped beats env drift.
    if tpc.df2_capture { gb10_inference::opts::set(gb10_inference::opt!("df2-capture"), "1"); }
    // DF2 block-16: install the HEAD's block on this node BEFORE the early round load below.
    // SPMD-critical: the round's MASK row count and the verify column count are collective, so a
    // one-sided block desyncs the verify all-reduce (and the round would size different buffers).
    // An absent field on the wire means an older head => the serde default 8.
    // TP2BIT5 #26: clean exit, never a panic! (same class as the EXL3 site above)
    if let Err(e) = gb10_inference::dflash2::set_block(tpc.df2_block) {
        eprintln!("node tp serve error: shipped df2_block {}: {e}", tpc.df2_block);
        std::process::exit(1);
    }
    // DF2_CARRY: the node applies the head's value (never resolved locally — SPMD). The
    // scheduler install below is the only consumer, so there is nothing to install here; the
    // value is read from `tpc` at the node's own `set_df2_carry` call site.
    println!("NODE — DF2 block = {} (from the head's config)", gb10_inference::dflash2::block());
    // S9F (TP-DF2 leg): the head's resolved spec source ships on the config; when it is a DF2
    // variant the node installs [df2-tp]=1 BEFORE its load so attach_tp keeps the FULL
    // lm_head for the round (the rank-local half is vocab-sharded). SPMD-relevant: a one-sided
    // capture is a weight-handle mismatch that drafts different tokens on the node.
    // WI1: DSPARK borrows the same full-vocab head — same arming (see the head-side comment).
    {
        let src = gb10_inference::batch::SpecSource::from_cli(&tpc.spec_source)
            .unwrap_or(gb10_inference::batch::SpecSource::Mtp);
        if gb10_inference::batch::is_df2_src(src)
            || src == gb10_inference::batch::SpecSource::Dspark {
            gb10_inference::opts::set(gb10_inference::opt!("df2-tp"), "1");
        }
    }
    // E12/E8/E9 escapes: SPMD-relevant — the fold changes the MoE launch sequence, the E8 shard
    // changes the WEIGHT LAYOUT at load, and E9 changes the launch attributes; a one-sided escape
    // desyncs the all-reduce epochs or mismatches the sharded tensors. The loader reads these as
    // env, so the node installs them BEFORE the load (the interim launcher exports were head-only).
    if tpc.moe_fold { gb10_inference::opts::set(gb10_inference::opt!("moe-fold"), "1"); }
    else { gb10_inference::opts::set(gb10_inference::opt!("moe-no-fold"), "1"); }
    if !tpc.e8_shard { gb10_inference::opts::set(gb10_inference::opt!("e8-no-shard"), "1"); }
    if !tpc.e9_fold { gb10_inference::opts::set(gb10_inference::opt!("e9-no-fold"), "1"); }
    // P3-1 one-shot push: SPMD-critical ring-layout selector. The transport reads this env at
    // init on BOTH ranks (head: its own env; node: installed here from the shipped config) — a
    // one-sided setting is a ring-layout mismatch, which dies loudly at the first barrier.
    if tpc.oneshot { gb10_inference::opts::set(gb10_inference::opt!("tp-oneshot"), "1"); }

    {
        let cfg = gb10_inference::qwen::Config::from_config_json(
            &format!("{}/config.json", dir.to_string_lossy().trim_end_matches('/')))?;
        let km = if tpc.kv_tq { gb10_inference::gpu::KVCacheMode::Tq }
                 else if tpc.kv_k8v8 { gb10_inference::gpu::KVCacheMode::K8v8 }
                 else if tpc.kv_k8v4 { gb10_inference::gpu::KVCacheMode::K8v4 }
                 else if tpc.kv_quant { gb10_inference::gpu::KVCacheMode::Q4 }
                 else { gb10_inference::gpu::KVCacheMode::Bf16 };
        mem_budget_report(&dir.to_string_lossy(), &cfg, true, tpc.max_seq_len, tpc.max_batch, km);
    }
    let (mut gpu, _cfg) = gb10_inference::gpu::GpuModel::load_from_dir_tp(&dir.to_string_lossy(), ctx.rank, ctx.world)?;
    let (rank, world, link) = ctx.into_parts();
    gpu.attach_tp(rank, world, link);

    // Pin the decode/launch thread to a big X925 core AFTER CUDA init — same rule as tp_serve: an
    // unpinned launch thread presents exactly like a protocol stall, so a pin failure is loud.
    if world == 2 && !gb10_inference::net::pin_thread(9) {
        panic!("FATAL: launch thread failed to pin to core 9 — TP refuses to run unpinned");
    }

    // S9F (TP-DF2 leg): the node's policy must carry the SAME speculation source as the head's
    // (shipped on the config) or the lane decisions diverge — head=DFlash2/node=MTP would take
    // different Phase-A branches and desync the verify all-reduces.
    let node_src = gb10_inference::batch::SpecSource::from_cli(&tpc.spec_source)
        .unwrap_or(gb10_inference::batch::SpecSource::Mtp);
    // S9F (TP-DF2 leg): the node loads the DFlash2 round EARLY (rank-local, right after attach,
    // BEFORE the SPMD calibration). Why early: the round load is a ~3.6 GB drafter read, and if
    // it sat between the CalibTable recv and the node's scheduler build (the decode-graph
    // capture warmup), the head's capture warmup would starve on the node's late barriers and
    // the 10 s transport watchdog would abort a HEALTHY bring-up. Early-loading overlaps the
    // head's own post-calib round load + calib ship; the CalibTable's df2_round outcome below
    // decides whether the round is KEPT (head's load succeeded) or DROPPED (a one-sided round
    // is a lane-branch mismatch — if the head's load failed, both ranks must serve MTP).
    let early_round: Option<(gb10_inference::dflash2::round::Df2Round,
                             std::sync::Arc<gb10_inference::dflash2::capture::Df2TapSink>,
                             std::sync::Arc<gb10_inference::dflash2::capture::Df2PrimeSink>)> =
        if gb10_inference::batch::is_df2_src(node_src) {
            // The node NEVER resolves a local draft path (owner rule 2026-08-23): the artifact
            // was shipped through the cluster sync's blob cache and the config's draft dir was
            // rewritten to the cache path. Empty means the head shipped none — the head's own
            // round load failed the same way, so CalibTable's df2_round=false keeps both ranks
            // consistently on MTP.
            if tpc.df2_draft_dir.is_empty() {
                eprintln!("NODE — no draft artifact was shipped by the head (df2_draft_dir empty) — \
                           matching the head's MTP fallback via the CalibTable outcome");
            }
            let draft_dir = tpc.df2_draft_dir.clone();
            match load_df2_round_dir(&mut gpu, tpc.max_seq_len, &draft_dir,
                                     tpc.df2_sha_pin.as_deref()) {
                Some(x) => {
                    println!("NODE — DFlash2 round RESIDENT (spec-source={}, draft-dir={}) — SPMD with the head",
                             node_src.cli_name(), draft_dir);
                    Some(x)
                }
                None => {
                    eprintln!("NODE — DFlash2 round load FAILED (draft-dir {draft_dir}) — will match the \
                               head's CalibTable outcome below (a head-side df2_round=true would then be a \
                               loud bail)");
                    None
                }
            }
        } else { None };
    // WI1 (TP-DSpark leg): the node loads the DSpark round EARLY under the same contract as the
    // DF2 early load — the shipped blob-cache dir, the head's CalibTable outcome decides keep/drop.
    let early_dspark: Option<(gb10_inference::dspark::round::DsparkRound,
                              std::sync::Arc<gb10_inference::dflash2::capture::Df2TapSink>,
                              std::sync::Arc<gb10_inference::dflash2::capture::Df2PrimeSink>)> =
        if matches!(node_src, gb10_inference::batch::SpecSource::Dspark) {
            if tpc.df2_draft_dir.is_empty() {
                eprintln!("NODE — no DSpark draft artifact was shipped (df2_draft_dir empty) — \
                           matching the head's MTP fallback via the CalibTable outcome");
            }
            match load_dspark_round_dir(&mut gpu, tpc.max_seq_len, &tpc.df2_draft_dir,
                                        tpc.df2_sha_pin.as_deref()) {
                Some(r) => {
                    println!("NODE — DSpark round RESIDENT (spec-source=dspark, draft-dir={}) — \
                              SPMD with the head", tpc.df2_draft_dir);
                    Some(r)
                }
                None => {
                    eprintln!("NODE — DSpark round load FAILED (draft-dir {}) — will match the \
                               head's CalibTable outcome below", tpc.df2_draft_dir);
                    None
                }
            }
        } else { None };

    // SPMD calibration. The node MUST execute the identical forward sequence — the all-reduces are
    // barriers the head waits on — but DISCARDS its tables: both ranks drive MtpPolicy from the
    // head's numbers (shipped next as CalibTable), so the policy state cannot diverge. Skipped
    // exactly when the head skips it (same model head-presence, same --mtp force in the config).
    if gpu.mtp_present() && tpc.mtp_force != Some(false) {
        println!("NODE — SPMD MTP calibration (tables discarded; head's are shipped)...");
        // TP boot barrier: every rank arrives before the calibration's first all-reduce (boot deadline, not the 10 s watchdog)
        gb10_inference::net::boot_barrier(0xCA1B).expect("TP boot barrier before the MTP calibration");
        let mut cpool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
        // Same sizing as the head's calibration state (main.rs run_server): ONE live KV slot at the
        // top context bucket's stride; GDN checkpoint slots (2..=N) are separately sized.
        let calib_points = gb10_inference::gpu::mtp_calib_ctx_points(tpc.max_seq_len);
        let calib_seq = *calib_points.last().unwrap();
        let mut cstate = gpu.new_batch_state(1, 2 + gb10_inference::gpu::PROFILE_MAX_N, calib_seq);
        let _ = gpu.calibrate_mtp_r(&mut cpool, &mut cstate, &tpc.calib_prompt, calib_seq);
    }
    let (head_r, df2_round): (Vec<(usize, Vec<(usize, f32)>)>, bool) = match recv_serving(&mut stream)? {
        ServingMsg::CalibTable { ctx_r, df2_round } => (
            ctx_r.into_iter()
                .map(|(c, t)| (c as usize, t.into_iter().map(|(d, r)| (d as usize, r)).collect()))
                .collect(),
            df2_round,
        ),
        other => anyhow::bail!("expected CalibTable from head, got {other:?}"),
    };
    println!("NODE — MTP cost tables from head ({} ctx buckets)", head_r.len());
    let policy = gb10_inference::batch::MtpPolicy::with_source(
        gpu.mtp_present(), tpc.mtp_force, tpc.mtp_depth_pin, head_r, node_src);
    let (_stx, srx) = tokio::sync::mpsc::unbounded_channel::<gb10_inference::batch::BatchRequest>();
    // S9F (TP-DF2 leg): reconcile the EARLY round load with the head's CalibTable outcome.
    // df2_round=true + loaded → keep (SPMD with the head). df2_round=true + not loaded → the
    // node's artifact is broken while the head's is fine — a one-sided round would desync the
    // verify all-reduces; refuse loudly. df2_round=false → drop the speculative load + disarm
    // the tap capture (both ranks serve MTP — the head's fallback, lane-branch consistent).
    // WI1: reconcile the early DSpark round with the head's (generic) round bit. The bit means
    // "SOME speculative round is resident on the head" — for a DSpark serve it refers to the
    // DSpark round (bail if ours failed to load: one-sided round = verify-all-reduce desync),
    // and for a DF2-family serve it refers to the DF2 round (the DF2 reconciliation below owns
    // that case — None here is the EXPECTED state, not a failure). The unguarded (true, None)
    // bail broke every DF2 TP serve after the WI1 landing (the node refused, the head's
    // watchdog killed the session) — regression class: path-under-change tested, sibling
    // path not; both arms now gated on their own source.
    let dspark_round = match (df2_round, early_dspark) {
        (true, Some(x)) => Some(x),
        (true, None) if matches!(node_src, gb10_inference::batch::SpecSource::Dspark) =>
            anyhow::bail!(
                "NODE — the head ships the speculative round as resident but the DSpark round \
                 FAILED to load here — a one-sided round would desync the verify all-reduces; \
                 refusing to serve"),
        (true, None) => None,   // DF2-family serve: the head's bit refers to ITS round
        (false, Some(mut x)) => {
            x.0.reset();
            eprintln!("NODE — head's speculative round did NOT load — dropping the DSpark round; \
                       serving the MTP fallback in lockstep with the head");
            None
        }
        (false, None) => None,
    };
    let (df2_round, df2_sink, df2_prime) = match (df2_round, early_round) {
        (true, Some(x)) => (Some(x.0), Some(x.1), Some(x.2)),
        (true, None) if gb10_inference::batch::is_df2_src(node_src) => anyhow::bail!(
            "NODE — the head ships df2_round=true but the DFlash2 round FAILED to load here \
             — a one-sided round would desync the verify all-reduces; refusing to serve \
             (fix the artifact on this node)"),
        // WI1: the head's round bit is GENERIC — "some speculative round is resident on the
        // head". A non-DF2 source (dspark) never attempts a DFlash2 load, so None here is
        // the EXPECTED state (the dspark reconciliation above handled that source), not a
        // one-sided-round failure.
        (true, None) => (None, None, None),
        (false, Some(x)) => {
            drop(x);
            gpu.set_df2_capture_off();
            eprintln!("NODE — head's DFlash2 round did NOT load — dropping the speculative round; \
                       serving the MTP fallback in lockstep with the head");
            (None, None, None)
        }
        (false, None) => (None, None, None),
    };
    let (dr, ds, dp) = match dspark_round {
        Some(x) => (Some(x.0), Some(x.1), Some(x.2)),
        None => (None, None, None),
    };
    let mut scheduler = gb10_inference::batch::BatchScheduler::with_df2(
        gpu, tpc.max_batch, tpc.max_seq_len, tpc.eos.clone(), srx, policy,
        tpc.prefix_cache, tpc.ngram_draft, tpc.tree_draft, tpc.mtp_lanes,
        df2_round, df2_sink, df2_prime, dr, ds, dp, None);
    // P14 (TP-v1 leg): the DFlash v1 BLOCK lane must exist on EVERY rank. The node replays the
    // head's identical `decode_step` (SPMD mirror), and `dflash_lane_step` is part of that step:
    // a node without the lane would take the MTP branch instead, drafting different tokens AND
    // issuing a different collective sequence. The artifact arrives through the cluster sync's
    // blob cache (the head ships `df2_draft_dir`, the config's path is rewritten to the cache) —
    // the same slot the DF2 round and DSpark use, so no local copy is needed on the node.
    if matches!(node_src, gb10_inference::batch::SpecSource::DFlash) {
        if tpc.df2_draft_dir.is_empty() {
            anyhow::bail!("NODE — the head resolved --spec-source dflash but shipped NO drafter \
                           artifact (df2_draft_dir empty). A node without the v1 lane would \
                           desync every verify all-reduce — refusing to serve");
        }
        if let Err(e) = scheduler.install_dflash_lane(&tpc.df2_draft_dir) {
            anyhow::bail!("NODE — DFlash v1 lane install failed for {} ({e:#}); a one-sided lane \
                           would desync every verify all-reduce — refusing to serve",
                          tpc.df2_draft_dir);
        }
        println!("NODE — DFlash v1 lane RESIDENT (spec-source=dflash, draft-dir={}) — SPMD with the head",
                 tpc.df2_draft_dir);
    }
    // P3(b) L1: mirror the head's prose-lane routing (SPMD — the node runs the identical decode_step).
    scheduler.set_prose_lane_greedy(tpc.df2_prose_lane_greedy);
    scheduler.set_df2_carry(tpc.df2_carry);
    // P11 W5a: mirror the head's prefill schedule (SPMD — the op order must match the head's).
    scheduler.set_pf_inline(tpc.pf_sched_inline);
    // PLAN/25 Phase 0: run the coverage-trace op sequence in lockstep with the head when the
    // session was launched with --df2-step-dump; the node's step_dump stays None (no records).
    scheduler.set_cov_trace(tpc.df2_step_dump);
    send_serving(&mut stream, &ServingMsg::Ready)?;
    println!("NODE — READY; entering SPMD mirror loop");
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    rt.block_on(scheduler.run_tp_mirror(stream))
}

/// TP=2 cluster HEAD: discover node(s) (or explicit --nodes) and push the model with content-addressed
/// caching (a node that already has the artifacts transfers nothing), then drive the SPMD masked-
/// replicated decode (Proof v0): broadcast the prompt to the node, run the identical generate loop, and
/// print the coherent output. `--prompt` / `--max-new-tokens` set the request.
/// DSV4 TP=2 serving through the STANDARD `--server` interface (identical flags to qwen TP
/// serving: --server --model-dir <bundle> --tp --nodes <peer[:29500]> --port <N> [--max-seq-len N]
/// [--max-tokens N]). Installs the serve-mode TpConfig (max_seq_len ships zero-config to the node), runs the cluster
/// sync, then enters the persistent DSV4 serve loop. The node routes itself (mode_serve ships).
/// `--max-tokens` (default 4096) is the generation cap when a request OMITS `max_tokens` — matches
/// the main engine's `--max-tokens` knob (the main engine defaults 8192; 4096 here is conservative
/// for TP=2 where a runaway 8K-token decode is a long stall; real chat turns send max_tokens).
pub(super) fn run_dsv4_server_tp(args: &[String], model_dir: &str, port: u16) {
    // (--rdma-dev is a registry option: parsed at startup, node-local, never shipped)
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
    if gb10_inference::opts::var(gb10_inference::opt!("tpc-debug")).is_ok() {
        eprintln!("[tpc-debug] serve from_opts: kv_k8v8={} kv_k8v4={} env={:?}",
                  tpc.kv_k8v8, tpc.kv_k8v4, gb10_inference::opts::var(gb10_inference::opt!("kv-k8v8")).ok());
    }
    tpc.mode_serve = true;
    // Same flag as qwen serving; the serve loops resolve env --max-seq-len > this > 4096.
    tpc.max_seq_len = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse::<usize>().ok()).unwrap_or(4096);
    // item 2.3: --prefix-cache <on|off> (AGENTS §7 CLI flag; rides TpConfig to the node).
    // DEFAULT ON for the dsv4 server (the pre-flag always-on behavior); the A/B uses off.
    tpc.prefix_cache = matches!(parse_arg(args, "--prefix-cache").unwrap_or("on"),
                                "on" | "true" | "1" | "yes");
    // item 3.4: --server-dspark <on|off> — DSpark speculation in the persistent server path.
    // DEFAULT ON (user decision 2026-08-05, after the 3.4 VERIFIED verdict — pass off for the
    // greedy server); rides TpConfig so the zero-config node takes the same decode branch
    // (SPMD). Requires the DSpark stages in the sharded dir (rank0/rank1/dspark_stage*.safetensors).
    tpc.server_dspark = matches!(parse_arg(args, "--server-dspark").unwrap_or("on"),
                                 "on" | "true" | "1" | "yes");
    // item 1.7(i) / T3: --dspark-fp8-head <on|off> — fp8_bsb draft LM head + Markov W2 (halve
    // the draft head reads). DEFAULT ON (user decision 2026-08-05; pass off for the bf16 draft
    // head); rides TpConfig so the zero-config node builds the SAME fp8 arms (SPMD — a
    // draft-logits divergence would desync the acceptance SPMD sequence). LOSSLESS preserved.
    tpc.dspark_fp8_head = matches!(parse_arg(args, "--dspark-fp8-head").unwrap_or("on"),
                                    "on" | "true" | "1" | "yes");
    // F8/P1: --fp8-prefill <on|off> — native e4m3 W8A8 tensor-core prefill GEMM for FP8
    // trunks (1.35x TP2 / 1.63x TP1 prefill at 1500 MHz; engine cross-check 2.5e-3..1.8e-2
    // rel-L2). DEFAULT OFF until the owner's release gate; --fp8-prefill stays as the
    // diagnostics/back-compat alias. Requires --mxfp4=on (allow_pf4), batch >= 256, %128
    // geometry. Rides TpConfig so the zero-config node dispatches identically (SPMD).
    if let Some(v) = gb10_inference::opt_or_arg(args, "--fp8-prefill") {
        tpc.fp8_prefill = matches!(&v[..], "on" | "true" | "1" | "yes");
    }
    if tpc.fp8_prefill {
        gb10_inference::opts::set(gb10_inference::opt!("fp8-prefill"), "1");
    }
    // E5 (512K mandate): --rope-yarn-factor <f> — TRUNK YaRN rope override (default 1.0 =
    // legacy tables byte-identical). f=2 on the 256K-native trunk extends the rope tables to
    // 512K rows with the vLLM-convention per-dim ramp + mscale — the same math the dspark
    // oracle validated (yaarn_freqs). Rides TpConfig so the node builds the same tables.
    tpc.rope_yarn_factor = gb10_inference::opt_or_arg(args, "--rope-yarn-factor").and_then(|v| v.parse::<f32>().ok())
        .filter(|f| *f >= 1.0).unwrap_or(1.0);
    // Single-box serves have no TpConfig (that's the --tp channel), so mirror the parsed flag
    // into the env — build_rope_tables resolves env-first. Startup-time set_var, before any
    // threads that could race it; under --tp the value ALSO rides TpConfig to the node.
    if tpc.rope_yarn_factor > 1.0 {
        gb10_inference::opts::set(gb10_inference::opt!("rope-yarn-factor"), tpc.rope_yarn_factor.to_string());
    }
    // item 3.3/3.4: --dspark-depth N pins the drafted-row count (None = ADAPTIVE). The pin is
    // the adaptive-depth DISABLE path — N=block reproduces the pre-3.3 fixed-depth behavior
    // bit-identically. Rides TpConfig so both ranks draft the same width (SPMD).
    tpc.dspark_depth = parse_arg(args, "--dspark-depth").and_then(|s| s.parse::<u32>().ok());
    if let Some(d) = tpc.dspark_depth {
        if !(1..=5).contains(&d) {
            eprintln!("--dspark-depth must be 1..=5 (the DSpark block size), got {d}");
            std::process::exit(1);
        }
    }
    gb10_inference::tp::set_tp_config(tpc.clone());
    // The DSpark server keeps the control stream RETAINED: the head ships its measured r(D)
    // table over it once per process (item 3.4 — see dsv4_tp_serve_server).
    let st_weights = gb10_inference::boot_report::Stage::begin("weights");
    let ctl = if tpc.server_dspark {
        match gb10_inference::cluster::run_head_session(std::path::Path::new(model_dir), explicit, wait, &tpc) {
            Ok((nodes, mut streams)) => {
                if gb10_inference::boot_report::debug() {
                    println!("HEAD SYNCED {} node(s) (control stream retained)", nodes.len());
                }
                // DSV4 serving is still TP=2 (N-way deferred): exactly one retained stream.
                Some(streams.pop().expect("world==2 requires exactly one retained control stream"))
            }
            Err(e) => { eprintln!("head error: {e:#}"); std::process::exit(1); }
        }
    } else {
        match gb10_inference::cluster::run_head(std::path::Path::new(model_dir), explicit, wait, &tpc) {
            Ok(nodes) => if gb10_inference::boot_report::debug() { println!("HEAD SYNCED {} node(s)", nodes.len()) },
            Err(e) => { eprintln!("head error: {e:#}"); std::process::exit(1); }
        }
        None
    };
    st_weights.ok();
    gb10_inference::boot_report::notice(&format!("[head] deepseek_v4 detected → Dsv4GpuModel TP=2 persistent server (standard --server --tp interface{})", if tpc.server_dspark { ", DSpark speculation" } else { "" }));
    let default_max_tokens = parse_arg(args, "--max-tokens").and_then(|s| s.parse::<usize>().ok()).unwrap_or(4096);
    // P4: DSV4 N-way deferred (still TP=2).
    if let Err(e) = dsv4_tp_serve_server(model_dir, gb10_inference::tp::TpContext::bring_up_head(2), port, default_max_tokens, ctl, &tpc) {
        eprintln!("head tp serve error: {e:#}"); std::process::exit(1);
    }
}

/// TP-A: the xtp probe's prompt ids (--prompt-ids <json|ws file> | --prompt <text> | the gate
/// sentence), tokenized with the pack's own tokenizer.
pub(crate) fn exl3_xtp_prompt_ids(args: &[String], dir: &str) -> Vec<u32> {
    if let Some(f) = parse_arg(args, "--prompt-ids") {
        let s = std::fs::read_to_string(f).expect("read --prompt-ids");
        return serde_json::from_str(&s).unwrap_or_else(|_| s.split_whitespace().map(|x| x.parse::<u32>().expect("bad id")).collect());
    }
    let text = parse_arg(args, "--prompt").unwrap_or(
        "The theory of general relativity states that gravity is the curvature of spacetime itself").to_string();
    let tok = gb10_inference::tokenizer::QwenTokenizer::from_file(&format!("{dir}/tokenizer.json")).expect("tokenizer.json");
    tok.encode(&text, false).expect("encode")
}

/// TP-B: the xtp program options from the CLI (the TP=1 reference and the TP=2 head take the
/// same flags; the head ships them to the node inside TpConfig.exl3_mode):
///   --xtp-pf-chunk N   prefill the prompt through the served chunked prefill (chunk N)
///   --xtp-gates N      G-T1-c / G-T1-d for verify widths 1..=N
///   --xtp-ident-only   no TP=1 reference: rank identity + greedy agreement only (head: --prompt-ids + --xtp-gen)
///   --xtp-time N       an untraced greedy timing pass of N tokens (not a perf claim)
pub(crate) fn exl3_xtp_opts(args: &[String]) -> gb10_inference::exl3_forward::xtp::XtpOpts {
    gb10_inference::exl3_forward::xtp::XtpOpts {
        pf_chunk: parse_arg(args, "--xtp-pf-chunk").map(|s| s.parse().expect("--xtp-pf-chunk N")),
        gates: parse_arg(args, "--xtp-gates").map(|s| s.parse().expect("--xtp-gates N")).unwrap_or(0),
        ident_only: args.iter().any(|a| a == "--xtp-ident-only"),
        time_tokens: parse_arg(args, "--xtp-time").map(|s| s.parse().expect("--xtp-time N")).unwrap_or(0),
        // TP-SP2: the G-T1-b fault injection (probe only): --xtp-sp-fault L[:R] (rank R, default 1)
        sp_fault: parse_arg(args, "--xtp-sp-fault").map(|s| {
            let (l, r) = s.split_once(':').unwrap_or((s, "1"));
            (l.trim().parse().expect("--xtp-sp-fault L[:R]"), r.trim().parse().expect("--xtp-sp-fault L[:R]"))
        }),
    }
}

/// TP-C: the spec program's options from the CLI (shipped to the node in TpConfig.exl3_mode).
pub(crate) fn exl3_spec_opts(args: &[String]) -> gb10_inference::exl3_serve::SpecOpts {
    let mut o = gb10_inference::exl3_serve::SpecOpts::default();
    let num = |n: &str| gb10_inference::opt_or_arg(args, n).map(|s| s.parse::<usize>().unwrap_or_else(|_| panic!("{n} N")));
    if let Some(v) = num("--spec-gen") { o.gen = v; }
    if let Some(v) = num("--spec-reps") { o.reps = v; }
    if let Some(v) = num("--max-batch") { o.width = v; }
    if let Some(v) = num("--spec-trials") { o.trials = v; }
    if let Some(v) = parse_arg(args, "--spec-depths") {
        o.depths = if v.is_empty() || v == "none" { Vec::new() } else { v.split(',').map(|x| x.parse().expect("--spec-depths a,b")).collect() };
    }
    if let Some(v) = parse_arg(args, "--spec-dds") { o.dds = v == "1"; }
    if let Some(v) = parse_arg(args, "--spec-prefix") { o.prefix = v == "1"; }
    if let Some(v) = parse_arg(args, "--spec-ident") { o.ident = v == "1"; }
    if let Some(v) = parse_arg(args, "--spec-samp") {
        o.samp = match v { "0" | "off" => 0, "quick" | "1" => 1, "full" | "2" => 2, _ => panic!("--spec-samp 0|quick|full") };
    }
    // validate through the wire format (the node parses exactly this)
    gb10_inference::exl3_serve::SpecOpts::parse(&o.to_mode()).expect("spec options")
}

/// TP-C: --spec-prompts name=ids.json,... (JSON id arrays; names <= 16 bytes).
pub(crate) fn exl3_spec_prompts(args: &[String]) -> Vec<(String, Vec<u32>)> {
    let v = parse_arg(args, "--spec-prompts").expect("--probe-exl3-tpspec requires --spec-prompts name=ids.json,...");
    v.split(',').map(|kv| {
        let (n, f) = kv.split_once('=').expect("--spec-prompts name=file");
        assert!(n.len() <= 16, "prompt name '{n}' longer than 16 bytes");
        let s = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("read {f}: {e}"));
        (n.to_string(), serde_json::from_str::<Vec<u32>>(&s).unwrap_or_else(|e| panic!("{f}: {e}")))
    }).collect()
}

/// CLI-1: the node supervisor's session child: `--node --once --port N` plus the supervisor's own
/// node-local registry options (`--rdma-dev`, `--tp-cache`, `--mem-watchdog-gb`, `--mem-trace`) —
/// flags, never an inherited env (every other option arrives from the head in TpConfig.opts).
pub(super) fn node_child_args(args: &[String], port: u16, udp: u16) -> Vec<String> {
    let _ = args;
    let mut v: Vec<String> = vec!["--node".into(), "--once".into(), "--port".into(), port.to_string(),
                                  "--discovery-port".into(), udp.to_string()];
    for (i, o) in gb10_inference::opts::REG.iter().enumerate() {
        if o.scope != gb10_inference::opts::Scope::Local { continue; }
        if let Ok(val) = gb10_inference::opts::var(gb10_inference::opts::OptId(i as u16)) {
            v.push(format!("--{}={val}", o.flag));
        }
    }
    v
}

/// TP-A: the EXL3 TP head (rank 0). Rung 3 program: `--probe-exl3-xtp`.
pub(super) fn run_exl3_tp_head(args: &[String], model_dir: &str) {
    gb10_inference::dprintln!("{}", gb10_inference::exl3_forward::xtp::identity_line("head rank=0"));
    let world = parse_tp_world(args).unwrap_or(2);
    if world != 2 && world != 4 { eprintln!("EXL3 TP: --tp 2 or --tp 4 only (got {world})"); std::process::exit(2); }
    if let Err(e) = gb10_inference::exl3_forward::xtp::world_gt2_flag_refusals(world as i32) { eprintln!("{e}"); std::process::exit(2); }
    // TP-4D: --probe-exl3-tpspec runs at world 4 too (the lockstep is the head-hub protocol of tp_lockstep)
    let spec = args.iter().any(|a| a == "--probe-exl3-tpspec");
    if !spec && !args.iter().any(|a| a == "--probe-exl3-xtp") {
        eprintln!("EXL3 TP: the programs wired are --probe-exl3-xtp and --probe-exl3-tpspec (serving is rung 5)");
        std::process::exit(2);
    }
    let spec_opts = if spec { Some(exl3_spec_opts(args)) } else { None };
    let spec_prompts = if spec { Some(exl3_spec_prompts(args)) } else { None };
    // the spec program's prefix cache allocates its snapshots at load: the node must see the same env
    if spec_opts.as_ref().map_or(false, |o| o.prefix) { gb10_inference::opts::set(gb10_inference::opt!("exl3-prefix"), "1"); }
    let opts = if spec { Default::default() } else { exl3_xtp_opts(args) };
    if let Err(e) = gb10_inference::exl3_forward::xtp::world_gt2_opts_refusals(world as i32, &opts) { eprintln!("{e}"); std::process::exit(2); }
    let ref_dir = if opts.ident_only || spec { parse_arg(args, "--xtp-ref").unwrap_or("").to_string() } else {
        parse_arg(args, "--xtp-ref").expect("--head --probe-exl3-xtp requires --xtp-ref <DIR> (the TP=1 reference) or --xtp-ident-only").to_string()
    };
    let head_prompt = if opts.ident_only {
        let gen: usize = parse_arg(args, "--xtp-gen").and_then(|s| s.parse().ok()).unwrap_or(16);
        Some((exl3_xtp_prompt_ids(args, model_dir), gen))
    } else { None };
    let max_pos: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(8192);
    let dir_abs = std::fs::canonicalize(model_dir).map(|p| p.to_string_lossy().to_string()).unwrap_or_else(|_| model_dir.to_string());
    let explicit = parse_arg(args, "--nodes").map(|s| {
        s.split(',').map(|p| {
            let p = p.trim();
            if p.contains(':') { p.parse::<std::net::SocketAddr>().expect("bad --nodes addr (ip:port)") }
            else { std::net::SocketAddr::new(p.parse::<std::net::IpAddr>().expect("bad --nodes ip"), 29500) }
        }).collect::<Vec<_>>()
    });
    let wait = std::time::Duration::from_secs(parse_arg(args, "--discover-wait").and_then(|s| s.parse().ok()).unwrap_or(3));
    let mut tpc = gb10_inference::tp::TpConfig::from_opts();
    tpc.world = world;
    tpc.exl3 = true;
    tpc.df2_block = gb10_inference::dflash2::block();   // DF2 block-16: the node installs the head's block (SPMD)
    tpc.exl3_pack_dir = dir_abs.clone();
    // --ep-deal interleave|contig|freq: a registry option (validated at startup; CLI-1 merged the
    // old GB10_TP_EP_DEAL=freq env override into it)
    tpc.exl3_ep_deal = gb10_inference::opts::get(gb10_inference::opt!("ep-deal")).unwrap_or_else(|| "interleave".into());
    tpc.exl3_mode = match &spec_opts { Some(o) => o.to_mode(), None => opts.to_mode() };
    tpc.max_seq_len = max_pos;
    if args.iter().any(|a| a == "--tp-trace") { tpc.trace = true; }
    exl3_tp_cli_knobs(args, &mut tpc);
    println!("[head] EXL3 TP config: deal {} max_pos {max_pos} trace {} options shipped {:?}",
             tpc.exl3_ep_deal, tpc.trace, gb10_inference::opts::snapshot());
    gb10_inference::tp::set_tp_config(tpc.clone());
    // PACK-FIX: the sync streams are retained through the RDMA bring-up so a node that refuses (or
    // dies) before the link is reported here instead of the head blocking in the QP handshake.
    let streams = match gb10_inference::cluster::run_head_retain(std::path::Path::new(&dir_abs), explicit, wait, &tpc) {
        Ok((_, s)) => s,
        Err(e) => { eprintln!("head error: {e:#}"); std::process::exit(1); }
    };
    let ctx = exl3_head_bring_up_watched(&streams, world as i32);
    drop(streams);   // the programs have no control plane past the link (pre-PACK-FIX: dropped at sync)
    let r = ctx.and_then(|ctx| match &spec_opts {
        Some(o) => gb10_inference::exl3_serve::run_spec(&dir_abs, Some(ctx), spec_prompts.clone(), max_pos, o),
        None => gb10_inference::exl3_forward::xtp::run_tp_xtp(&dir_abs, ctx,
                    if ref_dir.is_empty() { None } else { Some(&ref_dir) }, head_prompt, max_pos, &opts),
    });
    if let Err(e) = r { eprintln!("head exl3 tp error: {e:#}"); std::process::exit(1); }
}

/// TP-D: the EXL3 TP=2 SERVED head — `--server --tp 2 --model-dir <exl3 pack> [--nodes ip:port]`
/// plus every TP=1 serve flag. The head ships its whole argv to the node (TpConfig.exl3_mode =
/// "serve:<json argv>"), so the node resolves every serve knob exactly as the head does (AGENTS §7:
/// knobs ride TpConfig), then both ranks load their shard, agree the resolved config, and the head
/// binds HTTP once the node's mirror is armed.
/// NAIVE S5: the head of a Naive-N0.5 TP session (`mode` = "serve" | "probe"): ship the config (the pack
/// path, the argv, the program) to the nodes through the cluster sync, bring up the RDMA link, run.
pub(crate) fn run_naive_tp_head(args: &[String], model_dir: &str, mode: &str) {
    let world = parse_tp_world(args).unwrap_or(2);
    if world != 2 && world != 4 { eprintln!("Naive-N0.5 TP: --tp 2 or --tp 4 (got {world})"); std::process::exit(2); }
    let dir_abs = std::fs::canonicalize(model_dir).map(|p| p.to_string_lossy().to_string()).unwrap_or_else(|_| model_dir.to_string());
    let explicit = parse_arg(args, "--nodes").map(|s| {
        s.split(',').map(|p| {
            let p = p.trim();
            if p.contains(':') { p.parse::<std::net::SocketAddr>().expect("bad --nodes addr (ip:port)") }
            else { std::net::SocketAddr::new(p.parse::<std::net::IpAddr>().expect("bad --nodes ip"), 29500) }
        }).collect::<Vec<_>>()
    });
    let wait = std::time::Duration::from_secs(parse_arg(args, "--discover-wait").and_then(|s| s.parse().ok()).unwrap_or(3));
    let layers: usize = parse_arg(args, "--layers").and_then(|s| s.parse().ok()).unwrap_or(2);
    let tokens: usize = parse_arg(args, "--tokens").and_then(|s| s.parse().ok()).unwrap_or(8);
    let max_pos: usize = parse_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(32768);
    let mut tpc = gb10_inference::tp::TpConfig::from_opts();
    tpc.world = world;
    tpc.mode_serve = true;
    tpc.exl3 = true;              // the cluster sync ships NO artifacts: every rank reads its own pack copy
    tpc.df2_block = gb10_inference::dflash2::block();   // DF2 block-16: the node installs the head's block (SPMD)
    tpc.exl3_pack_dir = dir_abs.clone();
    tpc.max_seq_len = max_pos;
    let mut node_args: Vec<String> = args.to_vec();
    if let Some(i) = node_args.iter().position(|a| a == "--model-dir") { if i + 1 < node_args.len() { node_args[i + 1] = dir_abs.clone(); } }
    tpc.exl3_mode = match mode {
        "serve" => format!("naive-serve:{}", serde_json::to_string(&node_args).expect("argv json")),
        _ => format!("naive-probe:{}", serde_json::to_string(&(layers, tokens, args.iter().any(|a| a == "--probe-naive-tp-engine"), parse_arg(args, "--prefill-chunk").and_then(|s| s.parse::<usize>().ok()).unwrap_or(gb10_inference::naive::serve::PREFILL_CHUNK), node_args.clone())).expect("probe json")),
    };
    // ISSUES 98: the sharded drafter loads its shard on EVERY rank, so the head ships the drafter dir through the
    // cluster sync's draft round (content-addressed blob cache: a node that has it re-fetches nothing); the node
    // rewrites its --draft-dir to the cache path (run_naive_tp_node). No hand copy on the nodes any more.
    if mode == "serve" {
        if let Some(d) = gb10_inference::naive::serve::head_draft_dir(args, &dir_abs) {
            tpc.df2_draft_dir = std::fs::canonicalize(&d).map(|p| p.to_string_lossy().to_string()).unwrap_or(d);
            println!("[head] Naive drafter {} ships to every node (cluster sync, blob cache)", tpc.df2_draft_dir);
        }
    }
    gb10_inference::tp::set_tp_config(tpc.clone());
    println!("[head] Naive-N0.5 TP={world} {mode}: pack {dir_abs}, max_pos {max_pos}");
    let streams = match gb10_inference::cluster::run_head_retain(std::path::Path::new(&dir_abs), explicit, wait, &tpc) {
        Ok((_, s)) => s,
        Err(e) => { eprintln!("head error: {e:#}"); std::process::exit(1); }
    };
    let r = gb10_inference::cluster::watch_nodes_during_bring_up(&streams).and_then(|watch| {
        let ctx = gb10_inference::tp::TpContext::bring_up_head(world as i32)?;
        if mode == "serve" {
            gb10_inference::naive::serve::run_head(args, &dir_abs, ctx, streams, watch)
        } else {
            watch.disarm(&streams);
            let chunk: usize = parse_arg(args, "--prefill-chunk").and_then(|s| s.parse().ok()).unwrap_or(gb10_inference::naive::serve::PREFILL_CHUNK);
            // the node streams ride along: the full-depth probe's load barrier runs over them
            gb10_inference::naive::serve::run_probe(&dir_abs, ctx, layers, tokens, None, args.iter().any(|a| a == "--probe-naive-tp-engine"), chunk, args, streams)
        }
    });
    if let Err(e) = r { eprintln!("head naive tp error: {e:#}"); std::process::exit(1); }
}

/// NAIVE S5b: the node of a Naive-N0.5 TP session loads the SHIPPED shard dir (B32: the head ships this
/// rank's experts and pre-sliced row quads through the blob cache; `naive::shipplan`). No local copy, no
/// path override (owner rule 2026-10-17: every model ships its shards).
pub(super) fn run_naive_tp_node(tpc: &gb10_inference::tp::TpConfig, shard_dir: &std::path::Path, head_ip: std::net::IpAddr, mut stream: std::net::TcpStream) -> anyhow::Result<()> {
    // ISSUES 153: liveness for the RDMA bring-up retry loops — the cold-cache head ships the
    // other ranks for many minutes AFTER this node is READY; the dial must wait, not expire.
    gb10_inference::net::set_head_liveness(&stream);
    let dir = shard_dir.to_string_lossy().to_string();
    if !tpc.ship_shards || !gb10_inference::naive::config::NaiveConfig::is_naive_dir(&dir) {
        let why = format!("Naive-N0.5 node: expected this rank's SHIPPED shard dir (ship_shards {}), got {dir}", tpc.ship_shards);
        gb10_inference::cluster::node_report_failure(&mut stream, &why);
        anyhow::bail!(why);
    }
    let ctx = match gb10_inference::tp::TpContext::bring_up_node(head_ip, tpc.node_rank, tpc.world as i32) {
        Ok(c) => c,
        Err(e) => {
            let e = e.context("the node's RDMA data-plane link bring-up failed");
            gb10_inference::cluster::node_report_failure(&mut stream, &format!("{e:#}"));
            return Err(e);
        }
    };
    if let Some(j) = tpc.exl3_mode.strip_prefix("naive-probe:") {
        let (layers, tokens, engine, chunk, argv): (usize, usize, bool, usize, Vec<String>) = serde_json::from_str(j)?;
        return gb10_inference::naive::serve::run_probe(&dir, ctx, layers, tokens, Some(stream), engine, chunk, &argv, Vec::new());
    }
    // the head's argv (shipped) carries the serve knobs both ranks must agree on (--layers)
    let mut head_argv: Vec<String> = tpc.exl3_mode.strip_prefix("naive-serve:").and_then(|j| serde_json::from_str(j).ok()).unwrap_or_default();
    // ISSUES 98: the head shipped its drafter through the sync (tpc.df2_draft_dir = the assembled cache dir here):
    // point this rank's --draft-dir at it, so the node never opens the head's filesystem path
    if !tpc.df2_draft_dir.is_empty() {
        match head_argv.iter().position(|a| a == "--draft-dir") {
            Some(i) if i + 1 < head_argv.len() => head_argv[i + 1] = tpc.df2_draft_dir.clone(),
            _ => { head_argv.push("--draft-dir".into()); head_argv.push(tpc.df2_draft_dir.clone()); }
        }
        println!("[node] Naive drafter from the head's sync: {}", tpc.df2_draft_dir);
    }
    let layers = parse_arg(&head_argv, "--layers").and_then(|s| s.parse().ok());
    let chunk = parse_arg(&head_argv, "--prefill-chunk").and_then(|s| s.parse().ok()).unwrap_or(gb10_inference::naive::serve::PREFILL_CHUNK);
    gb10_inference::naive::serve::run_node(&dir, tpc.max_seq_len, layers, chunk, &head_argv, ctx, stream)
}

pub(super) fn run_exl3_tp_serve_head(args: &[String], model_dir: &str) {
    gb10_inference::fp4l::apply_pack_defaults(model_dir); // FP4L: the pack's quality defaults (ride TpConfig as set options)
    // (--rdma-dev is a registry option: parsed at startup, node-local, never shipped)
    gb10_inference::dprintln!("{}", gb10_inference::exl3_forward::xtp::identity_line("head rank=0 (serve)"));
    let world = parse_tp_world(args).unwrap_or(2);
    // TP-4D: world 2 (the pairwise lockstep, unchanged) or world 4 (the head-hub lockstep); anything else is refused
    // BEFORE the model load, and so is every flag the world-4 engine cannot honour (--tp-oneshot, --tp-dec-grecv != 0,
    // --ep-deal freq outside world 4: world_gt2_flag_refusals, the same list the probe head applies). The serve head
    // accepts `--tp-prefill-xport dual` at W=4 (TP-4E); the launcher's own default stays serial (PREFILL_XPORT=0)
    // until the TP-4E hardware gates pass. World 2 passes untouched.
    if world != 2 && world != 4 {
        eprintln!("EXL3 TP serve: --tp 2 or --tp 4 only (got {world})");
        std::process::exit(2);
    }
    // S-B9-CF-AGREE16B (W4): + the TP=2 `--max-batch` bound of the pairwise lockstep token, here, before discovery
    // and the pack sync (it used to be refused inside build_serve, after the sync and the RDMA attach)
    if let Err(e) = gb10_inference::exl3_forward::xtp::world_gt2_flag_refusals(world as i32)
        .and_then(|_| gb10_inference::exl3_serve::tp_lane_width_refusal(args, world as u32)) { eprintln!("{e}"); std::process::exit(2); }
    let dir_abs = std::fs::canonicalize(model_dir).map(|p| p.to_string_lossy().to_string()).unwrap_or_else(|_| model_dir.to_string());
    let explicit = parse_arg(args, "--nodes").map(|s| {
        s.split(',').map(|p| {
            let p = p.trim();
            if p.contains(':') { p.parse::<std::net::SocketAddr>().expect("bad --nodes addr (ip:port)") }
            else { std::net::SocketAddr::new(p.parse::<std::net::IpAddr>().expect("bad --nodes ip"), 29500) }
        }).collect::<Vec<_>>()
    });
    let wait = std::time::Duration::from_secs(parse_arg(args, "--discover-wait").and_then(|s| s.parse().ok()).unwrap_or(3));
    let mut tpc = gb10_inference::tp::TpConfig::from_opts();
    tpc.world = world;
    tpc.mode_serve = true;
    tpc.exl3 = true;
    tpc.df2_block = gb10_inference::dflash2::block();   // DF2 block-16: the node installs the head's block (SPMD)
    tpc.exl3_pack_dir = dir_abs.clone();
    // --ep-deal interleave|contig|freq: a registry option (validated at startup; CLI-1 merged the
    // old GB10_TP_EP_DEAL=freq env override into it)
    tpc.exl3_ep_deal = gb10_inference::opts::get(gb10_inference::opt!("ep-deal")).unwrap_or_else(|| "interleave".into());
    // the node re-parses exactly this argv (the model dir is the head's canonical path)
    let mut node_args: Vec<String> = args.to_vec();
    if let Some(i) = node_args.iter().position(|a| a == "--model-dir") {
        if i + 1 < node_args.len() { node_args[i + 1] = dir_abs.clone(); }
    }
    tpc.exl3_mode = format!("serve:{}", serde_json::to_string(&node_args).expect("argv json"));
    // Public PR #21: the shipped window is the clamped one. Every rank's build_serve resolves the same
    // clamp from the same argv and pack config, and logs the line naming the limit.
    tpc.max_seq_len = gb10_inference::exl3_serve::resolve_max_seq_len(
        &dir_abs, gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096)).0;
    if args.iter().any(|a| a == "--tp-trace") { tpc.trace = true; }
    exl3_tp_cli_knobs(args, &mut tpc);
    // FLASHNEXT-DF2-C (ISSUES 116): ship the DFlash2 drafter to the nodes exactly as the nvfp4 head
    // does — cluster.rs's DraftManifest reads `spec_source` + `df2_draft_dir`; the node assembles the
    // artifact in its blob cache and loads it from there (no local copy, never the head's path).
    tpc.spec_source = gb10_inference::exl3_serve::resolve_spec_source_exl3(args).cli_name().to_string();
    tpc.df2_draft_dir = gb10_inference::exl3_serve::tp_head_draft_dir(args).unwrap_or_default();
    if !tpc.df2_draft_dir.is_empty() {
        gb10_inference::dprintln!("[head] EXL3 TP serve: DFlash2 drafter {} ships to every node through the sync", tpc.df2_draft_dir);
    }
    gb10_inference::dprintln!("[head] EXL3 TP serve config: deal {} options shipped {:?}; node argv {:?}", tpc.exl3_ep_deal,
             gb10_inference::opts::snapshot(), node_args);
    // TP2BIT5 #26: head-side width preflight -- the pack's module widths decide bootability, so
    // check them BEFORE discovery and the first sync (the node refusal used to arrive only after
    // 57.5 GiB had shipped). Same predicate as the loader's (width_supported); MTP-head-only
    // unsupported widths stay a Fix-4 skip (serve without speculation), not a refusal.
    if let Err(e) = gb10_inference::exl3_serve::tp_pack_width_preflight(&dir_abs) {
        eprintln!("EXL3 TP head: {e:#}"); std::process::exit(2);
    }
    gb10_inference::tp::set_tp_config(tpc.clone());
    let streams = match gb10_inference::cluster::run_head_retain(std::path::Path::new(&dir_abs), explicit, wait, &tpc) {
        Ok((nodes, s)) => {
            // TP-4D: the world > 2 rank/host map (rank 0 = this head; `nodes[i]` is rank i + 1, the order the
            // control streams are indexed in) — printed at once and again at the API startup line
            if world > 2 {
                let map: Vec<String> = nodes.iter().enumerate().map(|(i, n)| format!("rank {} = {} ({})", i + 1, n.hostname, n.addr)).collect();
                println!("[head] TP={world} rank/host map: rank 0 = this head; {}", map.join("; "));
                gb10_inference::exl3_serve::set_tp_rank_map(map);
            }
            s
        }
        Err(e) => { eprintln!("head error: {e:#}"); std::process::exit(1); }
    };
    // PACK-FIX: every node's control stream is watched from here through the head's own load + boot
    // agree (run_tp_head_serve disarms it right before it reads the node's Ready): a node that fails
    // its link bring-up, load or boot — or dies — makes the head print the node's reason and exit 1
    // within ~0.1 s, instead of blocking in the QP-handshake accept() or loading for minutes first.
    let r = gb10_inference::cluster::watch_nodes_during_bring_up(&streams).and_then(|watch|
        gb10_inference::tp::TpContext::bring_up_head(world as i32)
            .and_then(|ctx| gb10_inference::exl3_serve::run_tp_head_serve(args, &dir_abs, ctx, streams, watch)));
    if let Err(e) = r { eprintln!("head exl3 tp serve error: {e:#}"); std::process::exit(1); }
}

/// PACK-FIX: the head's RDMA data-plane bring-up with every node's control stream watched — a node
/// that refuses (pack mismatch, option/argv error, its own link failure) or exits before the link
/// makes the head print the node's reason and exit 1 (cluster::watch_nodes_during_bring_up) instead
/// of blocking forever in the QP handshake accept(). The probe programs' variant: the watch ends
/// (blocking reads restored) once the link is up — a program node that finishes first must not read
/// as a failure; their post-link node deaths are the liveness probe's job (net_shim, abort code 10).
/// The served head keeps the watch armed through its load + boot (run_exl3_tp_serve_head).
fn exl3_head_bring_up_watched(streams: &[std::net::TcpStream], world: i32) -> anyhow::Result<gb10_inference::tp::TpContext> {
    let watch = gb10_inference::cluster::watch_nodes_during_bring_up(streams)?;
    let ctx = gb10_inference::tp::TpContext::bring_up_head(world);
    watch.disarm(streams);
    ctx
}

/// TP-A: the EXL3 TP node (rank >= 1) — zero-config: knobs from the shipped snapshot; the pack is the local
/// copy at the head's path (legacy, unchecked since 074c669) or (B32, always for Qwen3.8-Flash-Next) the rank's
/// shard dir assembled in the blob cache.
/// PACK-FIX: every failure BEFORE the RDMA data-plane link (pack mismatch, a missing/unreadable
/// pack file, an unparsable program from the head, the node's own link bring-up) is reported to the
/// head over the control stream before this returns Err — the head is blocked in its link bring-up
/// and prints the reason and exits instead of hanging (cluster::watch_nodes_during_bring_up). The
/// stream stays open for the whole session in every mode (a close before the link is exactly what
/// the head's watch treats as "the node quit").
pub(super) fn run_exl3_tp_node(tpc: &gb10_inference::tp::TpConfig, head_ip: std::net::IpAddr,
                    mut stream: std::net::TcpStream) -> anyhow::Result<()> {
    // ISSUES 153: liveness for the RDMA bring-up retry loops (see net.rs).
    gb10_inference::net::set_head_liveness(&stream);
    let program = match exl3_tp_node_prelink(tpc) {
        Ok(p) => p,
        Err(e) => { gb10_inference::cluster::node_report_failure(&mut stream, &format!("{e:#}")); return Err(e); }
    };
    let ctx = match gb10_inference::tp::TpContext::bring_up_node(head_ip, tpc.node_rank, tpc.world as i32) {
        Ok(c) => c,
        Err(e) => {
            let e = e.context("the node's RDMA data-plane link bring-up failed");
            gb10_inference::cluster::node_report_failure(&mut stream, &format!("{e:#}"));
            return Err(e);
        }
    };
    match program {
        // TP-D: the served mirror — the head's argv, re-parsed here by the same serve boot
        Exl3NodeProgram::Serve(argv) =>
            gb10_inference::exl3_serve::run_tp_node_serve(&argv, &tpc.exl3_pack_dir, ctx, stream),
        Exl3NodeProgram::Spec(o) => {
            let _ctl = stream;   // held to the end (the head's watch reads a close as a refusal)
            gb10_inference::exl3_serve::run_spec(&tpc.exl3_pack_dir, Some(ctx), None, tpc.max_seq_len, &o)
        }
        Exl3NodeProgram::Xtp(opts) => {
            let _ctl = stream;
            gb10_inference::exl3_forward::xtp::run_tp_xtp(&tpc.exl3_pack_dir, ctx, None, None, tpc.max_seq_len, &opts)
        }
    }
}

/// The EXL3 TP node's program, resolved from the head's `exl3_mode` before the link comes up.
enum Exl3NodeProgram {
    Serve(Vec<String>),
    Spec(gb10_inference::exl3_serve::SpecOpts),
    Xtp(gb10_inference::exl3_forward::xtp::XtpOpts),
}

/// The EXL3 TP node's pre-link phase: install the head's knobs and resolve the program (the pack itself is
/// no longer compared with the head's — the manifest check was removed by the owner in 074c669). Any Err here
/// is reported to the head by the caller.
fn exl3_tp_node_prelink(tpc: &gb10_inference::tp::TpConfig) -> anyhow::Result<Exl3NodeProgram> {
    // (CLI-1: the head's option registry was installed at the sync — run_cluster_node_once)
    // TP-4B: a world the EXL3 engine cannot run, or a world > 2 flag it refuses, is reported to the head
    // here, before the link, exactly as the head's own pre-load refusal would (world 2 passes untouched)
    anyhow::ensure!(tpc.world == 2 || tpc.world == 4, "EXL3 TP node: --tp 2 or --tp 4 only (the head shipped world {})", tpc.world);
    gb10_inference::exl3_forward::xtp::world_gt2_flag_refusals(tpc.world as i32)?;
    // TP-I2: the head's CLI-flag knobs (TpConfig v19) — the node never parses its own
    gb10_inference::exl3_forward::xtp::set_pf_overlap(tpc.exl3_pf_overlap.unwrap_or(0));
    // TP-H2 (v21): None (a pre-v21 head) = off, the pre-TP-H2 replicated sampled tail
    gb10_inference::exl3_forward::xtp::set_vp_sampled(tpc.exl3_vp_sampled.unwrap_or(0));
    // TP-SP1 (v22): None (a pre-v22 head) = off, the replicated prefill
    gb10_inference::exl3_forward::xtp::set_seq_parallel(tpc.exl3_seq_parallel.unwrap_or(0));
    gb10_inference::dprintln!("[node] EXL3 TP CLI knobs from the head: --tp-prefill-overlap {} --tp-vp-sampled {}{}",
             gb10_inference::exl3_forward::xtp::pf_overlap_desc(tpc.exl3_pf_overlap.unwrap_or(0)),
             gb10_inference::exl3_forward::xtp::vp_sampled_desc(tpc.exl3_vp_sampled.unwrap_or(0)),
             if tpc.exl3_seq_parallel.unwrap_or(0) != 0 {
                 format!(" --tp-seq-parallel {}", gb10_inference::exl3_forward::xtp::seq_parallel_desc(tpc.exl3_seq_parallel.unwrap_or(0)))
             } else { String::new() });
    if let Some(j) = tpc.exl3_mode.strip_prefix("serve:") {
        let argv: Vec<String> = serde_json::from_str(j).map_err(|e| anyhow::anyhow!("serve argv from the head: {e}"))?;
        gb10_inference::dprintln!("[node] EXL3 TP serve: head argv {argv:?}");
        // S-B9-CF-AGREE16B (W4): the same pre-network refusal as on the head, from the head's argv, before the link
        gb10_inference::exl3_serve::tp_lane_width_refusal(&argv, tpc.world as u32)?;
        return Ok(Exl3NodeProgram::Serve(argv));
    }
    if tpc.exl3_mode.starts_with("spec") {
        return Ok(Exl3NodeProgram::Spec(gb10_inference::exl3_serve::SpecOpts::parse(&tpc.exl3_mode)?));
    }
    let opts = gb10_inference::exl3_forward::xtp::XtpOpts::parse(&tpc.exl3_mode)?;
    gb10_inference::exl3_forward::xtp::world_gt2_opts_refusals(tpc.world as i32, &opts)?;
    Ok(Exl3NodeProgram::Xtp(opts))
}

/// TP-I2: resolve the head's CLI-flag knobs of the EXL3 TP engine, install them in this process, and
/// put them into the TpConfig the node receives (owner rule: CLI flags ride TpConfig, never env).
fn exl3_tp_cli_knobs(args: &[String], tpc: &mut gb10_inference::tp::TpConfig) {
    use gb10_inference::exl3_forward::xtp;
    let _ = args;
    // CLI-1: the three are registry options (parsed at startup; --print-config shows them)
    let pfo_v = gb10_inference::opts::get(gb10_inference::opt!("tp-prefill-overlap"));
    let pfo = xtp::parse_pf_overlap(pfo_v.as_deref())
        .unwrap_or_else(|e| { eprintln!("{e:#}"); std::process::exit(2); });
    // TP-4E: the fold consumer is a world-2 schedule; refuse it at the head, before any load (the nodes re-check in attach)
    if let Err(e) = xtp::world_gt2_overlap_refusal(tpc.world as i32, pfo) { eprintln!("{e}"); std::process::exit(2); }
    xtp::set_pf_overlap(pfo);
    tpc.exl3_pf_overlap = Some(pfo);
    // TP-H2: the vocab-parallel lm_head for sampled / penalized / ratio-rule rows (TpConfig v21)
    let vps_v = gb10_inference::opts::get(gb10_inference::opt!("tp-vp-sampled"));
    // TP-4H2: world 2 and 4 resolve the default ON (world 4 = the recursive-doubling row gather); any other
    // world > 2 resolves it OFF and an explicit `on` is refused
    let (vps, vps_note) = xtp::resolve_vp_sampled(tpc.world, vps_v.as_deref())
        .unwrap_or_else(|e| { eprintln!("{e:#}"); std::process::exit(2); });
    if let Some(n) = vps_note { println!("[head] {n}"); }
    xtp::set_vp_sampled(vps);
    tpc.exl3_vp_sampled = Some(vps);
    // TP-SP1: sequence-parallel prefill (TpConfig v22)
    let spv_v = gb10_inference::opts::get(gb10_inference::opt!("tp-seq-parallel"));
    let spv = spv_v.as_deref();
    let mut spp = xtp::parse_seq_parallel(spv)
        .unwrap_or_else(|e| { eprintln!("{e:#}"); std::process::exit(2); });
    if spp != 0 && xtp::pf_overlap_fold() {
        if spv.is_some() {
            eprintln!("--tp-seq-parallel {} is not composed with --tp-prefill-overlap fold (TP-SP1: use on|dual|inline|off)",
                      spv.unwrap_or(""));
            std::process::exit(2);
        }
        // TP-SP2: the default (ON) yields to an explicit fold — the two are not composed
        println!("[head] --tp-prefill-overlap fold: sequence-parallel prefill OFF (not composed with fold; the default yields)");
        spp = 0;
    }
    // TP-4E / TP-4S: SP exists at world 2 and world 4; an EXPLICIT --tp-seq-parallel above world 4 would silently not run:
    // refuse it loudly; the default (flag absent) stays ON in the config and yields at runtime, as TP-4B validated
    if let Err(e) = xtp::world_gt2_sp_refusal(tpc.world as i32, spv.is_some(), spp) { eprintln!("{e}"); std::process::exit(2); }
    xtp::set_seq_parallel(spp);
    tpc.exl3_seq_parallel = Some(spp);
}
