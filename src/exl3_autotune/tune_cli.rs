//! `pair_obs` (paired observations of every candidate arm against the base) and `tune_cli`, the `--autotune` search driver.
//! Part of the `--autotune` T2 code (see `mod.rs` for the file map).

use super::*;

/// Paired observations of every candidate arm (1..arms) against the base (arm 0) in one phase:
/// per round, (m', t_arm - t_base) for the verify unit, (0, draft delta) for the draft unit, or
/// (m', dv + (passes per round / s) * dd) for a Round target. Only digest-clean samples pair; the
/// warm-ups (mode "warm") never do. v2: each time has its cell's fitted position offset removed
/// first when `offs` holds the cell (an empty table = the plain paired difference).
pub(super) fn pair_obs(samples: &[Sample], offs: &PosTable, phase: usize, target: Target, arms: usize, k: usize, regime: u8)
    -> Vec<Vec<tune::Obs>> {

    let mut by_round: HashMap<usize, (HashMap<usize, (usize, f64)>, HashMap<usize, f64>)> = HashMap::new();
    for s in samples.iter().filter(|s| s.phase == phase && s.mode == "sr" && s.arm < arms && s.digest_ok && s.regime == regime) {
        let e = by_round.entry(s.round).or_default();
        let ms = adj_ms(s, offs);
        match s.unit {
            "verify_dry" => { e.0.insert(s.arm, (s.m, ms)); }
            "draft" => { e.1.insert(s.arm, ms); }
            _ => {}
        }
    }
    let w = DRAFT_PASSES_PER_ROUND / k.max(1) as f64;
    let mut out: Vec<Vec<tune::Obs>> = vec![Vec::new(); arms];
    let mut rounds: Vec<usize> = by_round.keys().copied().collect();
    rounds.sort_unstable();
    for r in rounds {
        let (v, d) = &by_round[&r];
        for a in 1..arms {
            let dv = match (v.get(&0), v.get(&a)) { (Some(b), Some(x)) => Some((x.0, x.1 - b.1)), _ => None };
            let dd = match (d.get(&0), d.get(&a)) { (Some(b), Some(x)) => Some(x - b), _ => None };
            let o = match target {
                Target::Verify => dv.map(|(m, x)| (m as u16, x)),
                Target::Draft => dd.map(|x| (0u16, x)),
                Target::Round => match (dv, dd) { (Some((m, x)), Some(y)) => Some((m as u16, x + w * y)), _ => None },
            };
            if let Some(o) = o { out[a].push(o); }
        }
    }
    out
}

/// `gb10_inference --autotune` (T2). Ok(true) = a table was written; Ok(false) = the gate rejected it.
pub(super) fn tune_cli(args: &[String]) -> Result<bool> {
    // the box rule first (G-T8)
    if let Err(why) = preflight() {
        println!("AUTOTUNE REFUSED: {why}");
        bail!("preflight refused: {why}");
    }
    let dir = arg(args, "--model-dir").context("--autotune requires --model-dir <EXL3 pack>")?.to_string();
    if !crate::exl3_serve::is_exl3_pack(&dir) { bail!("{dir} is not an EXL3 pack"); }
    let num = |k: &str, d: usize| -> Result<usize> {
        match arg(args, k) { None => Ok(d), Some(v) => v.parse().map_err(|_| anyhow::anyhow!("{k} must be an integer")) }
    };
    // ---- the SERVED posture (the server's own flags and defaults: exl3_serve::run)
    let max_seq_len = num("--max-seq-len", 4096)?;
    let lanes = num("--max-batch", 8)?.max(1);
    let max_pos = max_seq_len + crate::batch::decode_headroom(false);
    let chunk = num("--prefill-chunk", crate::exl3_serve::DEFAULT_PREFILL_CHUNK)?.max(1);
    let draft_conf: f64 = match arg(args, "--draft-confidence") {
        None => 0.4,
        Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--draft-confidence must be a number"))?,
    };
    anyhow::ensure!((0.0..1.0).contains(&draft_conf), "--draft-confidence must be in [0, 1)");
    if let Some(v) = arg(args, "--kv-cache") {
        anyhow::ensure!(matches!(v, "f32" | "f16" | "fp8" | "q8" | "q4k"), "--kv-cache must be f32, f16, fp8, q8 or q4k");
        crate::opts::set(crate::opt!("kv-cache"), v);
    }
    if !matches!(arg(args, "--prefix-cache"), Some("off")) { crate::opts::set(crate::opt!("exl3-prefix"), "1"); }
    let k = mtp_depth_default();
    let posture = tune::Posture { lanes, mtp_max_k: k, dds: draft_conf > 0.0, kv_fmt: kv_fmt_from_opts(), max_pos,
                                  prefill_chunk: chunk };
    // ---- the run's own knobs
    let budget_min: f64 = match arg(args, "--autotune-budget") {
        None => 20.0,
        Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--autotune-budget must be minutes"))?,
    };
    let max_new = num("--autotune-max-new", 512)?.max(32);
    let li_rounds = num("--autotune-li-rounds", 120)?.max(16);
    let hash_tokens = num("--autotune-hash-tokens", 1024)?.max(64);
    let max_sweeps = num("--autotune-sweeps", 3)?.max(1);
    let out_dir = arg(args, "--autotune-out").unwrap_or("tune/receipts").to_string();
    let origin = arg(args, "--autotune-origin").unwrap_or("user").to_string();
    anyhow::ensure!(origin == "user" || origin == "shipped", "--autotune-origin must be user or shipped");
    let only: Option<Vec<String>> = arg(args, "--autotune-families").map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
    let skip_sparse = args.iter().any(|a| a == "--autotune-skip-sparse");
    // the sparse context point: default just past the dense/sparse switch (cheap: ~2K-token prompt);
    // --autotune-sparse-ctx <N> puts it deeper (e.g. 32768: ~33 s per prompt restart, labelled)
    let sparse_ctx: Option<usize> = match arg(args, "--autotune-sparse-ctx") {
        None => None,
        Some(v) => Some(v.parse().map_err(|_| anyhow::anyhow!("--autotune-sparse-ctx must be tokens"))?),
    };
    let skip_prefill = args.iter().any(|a| a == "--autotune-skip-prefill");
    let skip_gate = args.iter().any(|a| a == "--autotune-skip-gate");
    let seed: u64 = match arg(args, "--autotune-seed") {
        Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--autotune-seed must be an integer"))?,
        None => std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1),
    };
    let profile = match arg(args, "--tune-profile") {
        Some(v) => v.parse::<u32>().map_err(|_| anyhow::anyhow!("--tune-profile must be MHz"))?,
        None => tune::detect_clock_profile().unwrap_or(tune::STOCK_PROFILE_MHZ),
    };
    let (warmup, stratify) = protocol_args(args)?;
    // the measured config = the built-in defaults (a new table is tuned from scratch)
    tune::arm_tuner();
    // §13 instrumentation: bind the device-0 primary context up front (allocation-free) so
    // mem_get_info works at every probe point below (same ctx FwdModel::load reuses).
    // Three-point attribution receipt: pre-boot / post-boot / after-load (the check line below).
    let _ctx_guard = cudarc::driver::CudaDevice::new(0).ok();
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let rss_bytes = || -> Option<u64> {
        std::fs::read_to_string("/proc/self/status").ok()?.lines().find(|l| l.starts_with("VmRSS:"))?
            .split_whitespace().nth(1)?.parse::<u64>().ok().map(|kb| kb * 1024)
    };
    let mem_probe = |tag: &str| {
        let fmt = |o: Option<u64>| o.map(|b| format!("{:.1} GiB", gib(b))).unwrap_or_else(|| "n/a".into());
        let cu = cudarc::driver::result::mem_get_info().ok();
        println!("AUTOTUNE: mem {tag}: CUDA free {} of {}; MemAvailable {}; process RSS {}",
                 fmt(cu.map(|c| c.0 as u64)), fmt(cu.map(|c| c.1 as u64)),
                 fmt(crate::memwatch::mem_available_bytes()), fmt(rss_bytes()));
    };
    mem_probe("pre-boot");
    tune::boot(&tune::BootReq { sel: tune::TableSel::Off, model_dir: &dir, posture: posture.clone(), profile_mhz: Some(profile),
                                draft_on: false, tp: 1 });
    mem_probe("after tune::boot");
    let overridden = tune::env_overrides();
    std::fs::create_dir_all(&out_dir).with_context(|| format!("mkdir {out_dir}"))?;
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let jsonl_path = format!("{out_dir}/autotune-{stamp}.jsonl");
    let jsonl = std::io::BufWriter::new(std::fs::File::create(&jsonl_path).with_context(|| format!("create {jsonl_path}"))?);
    let bin_sha = std::fs::read("/proc/self/exe").map(|b| tune::sha256_hex(&b)).unwrap_or_else(|_| "unknown".into());

    // §13 baseline BEFORE the load: on GB10 unified memory cuMemGetInfo's `free` reads ~0
    // whenever a large model is resident (a healthy serve box measures 0.98 by 1-free/total),
    // so the absolute formula can never pass after weights load. The footprint the ceiling
    // means is OUR delta from this post-boot, pre-load baseline (ctx bound above).
    let mem_base = cudarc::driver::result::mem_get_info().ok();
    let avail_base = crate::memwatch::mem_available_bytes();
    let t_load = std::time::Instant::now();
    let model = FwdModel::load(&dir, lanes, max_pos)?;
    let head = model.mtp.as_ref().context("the pack has no mtp.* draft head — the tuner measures MTP rounds")?;
    // §13: refuse above the 80% memory ceiling (unified LPDDR5x is shared with the OS).
    // What the load's footprint IS (2026-09-27 attribution, TUNE_ceiling_bug_RESOLVED.md): the
    // device-side weights/state (what nvidia-smi shows) PLUS the RAM-resident PLE n-gram table —
    // ordinary host Vec pages (~30 GiB on Flash-Next) that nvidia-smi never shows but that lower
    // cuMemGetInfo's free exactly like a device allocation on the unified pool. Serve holds the
    // same table, so this footprint equals serve's. CUDA free is MemFree: it excludes the
    // reclaimable page cache the load just filled (the safetensors), so it reads ~1 GiB on a
    // healthy box — the survivability floor is Linux MemAvailable, which counts that cache.
    let mem_after = cudarc::driver::result::mem_get_info().ok();
    let avail_after = crate::memwatch::mem_available_bytes();
    model.ple_promote_auto()?; // the served residency (CF-P1d: auto decides after the load)
    let (ple_b, ple_mode) = (model.ple_ram_bytes(), model.ple_residency());
    let avail_s = match (avail_base, avail_after) {
        (Some(a0), Some(a1)) => format!("MemAvailable {:.0} -> {:.0} GiB", gib(a0), gib(a1)),
        _ => "MemAvailable n/a".into(),
    };
    match (mem_base, mem_after) {
        (Some((free0, total)), Some((free1, _))) => {
            let used_b = free0.saturating_sub(free1) as u64;
            let used = used_b as f64 / total.max(1) as f64;
            // real consumption = MemAvailable Δ: the CUDA-free Δ (MemFree) also counts the page
            // cache the load's file reads grew, so footprint − PLE overstates the device side
            // (.14 v4 smoke: 61.0 vs nvidia-smi 54.1 GiB; MemAvailable Δ − PLE = 53.2 GiB)
            let real = match (avail_base, avail_after) { (Some(a0), Some(a1)) => Some(a0.saturating_sub(a1)), _ => None };
            let (dev_b, dev_how) = match real {
                Some(r) => (r.saturating_sub(ple_b), "MemAvailable Δ − PLE"),
                None => (used_b.saturating_sub(ple_b), "footprint − PLE"),
            };
            println!("AUTOTUNE: memory after load: footprint {:.1}% of {:.0} GiB (CUDA free {:.0} -> {:.0} GiB; {avail_s}; \
                      PLE RAM table {:.1} GiB [{ple_mode}]; device-side ≈ {dev_how} ≈ {:.1} GiB)",
                     used * 100.0, gib(total as u64), gib(free0 as u64), gib(free1 as u64), gib(ple_b), gib(dev_b));
            if let Some(r) = real {
                let cache = used_b as i64 - r as i64;
                println!("AUTOTUNE: memory attribution: MemAvailable Δ {:.1} GiB = real consumption; the footprint's CUDA-free Δ \
                          ({:.1} GiB) also counts {:+.1} GiB of reclaimable page-cache growth; process RSS {} (host pages: \
                          PLE table + host buffers; nvidia-smi shows only the device side)",
                         gib(r), gib(used_b), cache as f64 / (1u64 << 30) as f64,
                         rss_bytes().map(|b| format!("{:.1} GiB", gib(b))).unwrap_or_else(|| "n/a".into()));
            }
            if used > 0.80 {
                println!("AUTOTUNE REFUSED: the load consumed {:.0}% of device memory (> 80% ceiling, §13)", used * 100.0);
                bail!("memory ceiling");
            }
        }
        _ => println!("AUTOTUNE: memory ceiling check SKIPPED ({} mem_get_info unavailable)",
                      if mem_base.is_none() { "pre-load" } else { "after-load" }),
    }
    // survivability floor: what the OS can still hand out (free + reclaimable cache), not MemFree
    match avail_after {
        Some(a1) if a1 < (8u64 << 30) => {
            println!("AUTOTUNE REFUSED: only {:.1} GiB MemAvailable after the load (< 8 GiB survivability floor, §13; \
                      {avail_s}, PLE RAM table {:.1} GiB [{ple_mode}])", gib(a1), gib(ple_b));
            bail!("memory ceiling");
        }
        Some(_) => {}
        None => println!("AUTOTUNE: survivability floor SKIPPED (/proc/meminfo MemAvailable unavailable)"),
    }
    let sc = FwdModel::scratch(model.dev(), &model.cfg, lanes.max(k + 1), model.max_pos())?;
    let mut psc0 = model.prefill_scratch(chunk)?;
    model.prefill_warmup(&mut psc0, 0, &FwdModel::prefill_warmup_widths(chunk))?;
    let tok = crate::tokenizer::QwenTokenizer::from_file(&format!("{}/tokenizer.json", dir.trim_end_matches('/')))?;
    let corpus: Vec<Vec<u32>> = match arg(args, "--autotune-corpus") {
        Some(f) => {
            let texts: Vec<String> = serde_json::from_str(&std::fs::read_to_string(f)?)
                .context("--autotune-corpus must be a JSON array of prompt strings")?;
            texts.iter().map(|t| tok.encode(t, false)).collect::<Result<_>>()?
        }
        None => CORPUS.iter().chain(std::iter::once(&PY_PROMPT)).map(|t| tok.encode(t, false)).collect::<Result<_>>()?,
    };
    anyhow::ensure!(corpus.iter().all(|c| c.len() >= 2), "every corpus prompt needs >= 2 tokens");
    let corpus_sha = tune::sha256_hex(serde_json::to_string(&corpus)?.as_bytes());
    let fp = tune::current_fingerprint(&dir, 1, profile);

    // the whole run fits the budget: the search gets it minus the prefill point and the final gate
    // (LI 3 classes x 2 x li_rounds rounds + 3 prompts x 2 modes x 2 arms x hash_tokens tokens)
    let reserve_s = if skip_prefill { 0.0 } else { 120.0 }
        + if skip_gate { 0.0 } else { 3.0 * 2.0 * li_rounds as f64 * 0.07 + 12.0 * hash_tokens as f64 / 90.0 + 30.0 };
    let rep: Vec<usize> = [3usize, 5, 6, 8].iter().copied().filter(|&m| m <= k + 1).collect();
    let all: Vec<usize> = (2..=k + 1).collect();
    let mut session = Session::new(&model, SrCfg { verify_on: true, draft_on: true, li_on: false, verify_widths: all.clone(),
                                                   draft_s: k, mem_cap: 512 << 20 }, vec![Arm::base()], seed)?;
    session.set_warmup(warmup)?;
    SESSION.with(|c| *c.borrow_mut() = Some(session));
    let mut t = Tuner {
        model: &model, head, sc, psc: Some(psc0), chunk, k, draft_conf, max_new, corpus: corpus.clone(), corpus_label: "dense",
        gen_next: 0, gen: None, round_ms: Vec::new(), widths_rep: rep, widths_all: all,
        base_ms: HashMap::new(), base_mad: HashMap::new(), decisions: Vec::new(), families: serde_json::Map::new(),
        jsonl, t0: std::time::Instant::now(), budget_s: (budget_min * 60.0 - reserve_s).max(60.0), rcfg: tune::RaceCfg::default(), regime: 0,
        gpu_log: Vec::new(), throttle_events: 0, mem_cap: 512 << 20,
        warmup, stratify, pf_order: tune::BalancedOrder::new(seed ^ 0x9F), known_answer: None,
    };
    let _ = writeln!(t.jsonl, "{}", serde_json::json!({
        "receipt": "gb10-exl3-autotune/1", "fingerprint": serde_json::to_value(&fp)?, "posture": serde_json::to_value(&posture)?,
        "binary_sha256": bin_sha, "commit": env!("TUNE_GIT_COMMIT"), "dirty": env!("TUNE_GIT_DIRTY"),
        "kernel_build_id": env!("KERNEL_BUILD_ID"), "tune_build_id": env!("TUNE_BUILD_ID"), "model_dir": dir, "seed": seed,
        "date_unix": stamp, "budget_min": budget_min, "overridden_env": format!("{overridden:?}"),
        "protocol": protocol_json(warmup, stratify) }));
    t.say(format!("AUTOTUNE: measurement protocol v3 — {warmup} untimed base warm-up replay(s) per block, balanced (Williams) \
                   arm order per cell, analysis {}", if stratify { "position-stratified" } else { "plain paired" }));
    t.say(format!("AUTOTUNE: model {dir} loaded in {:.1} s | posture lanes {lanes} max_pos {max_pos} k {k} dds {} kv {} chunk {chunk} \
                   | profile {profile} MHz tp 1 | budget {budget_min} min (search {:.1} min, gate + prefill reserve {:.1} min) \
                   | seed {seed} | receipts {jsonl_path}",
                  t_load.elapsed().as_secs_f64(), draft_conf > 0.0, kv_fmt_name(kv_fmt_from_opts()),
                  t.budget_s / 60.0, reserve_s / 60.0));
    for (id, env, v) in &overridden {
        t.say(format!("AUTOTUNE: {env} is set -> {id}={v} is OVERRIDDEN(flag) and will not be searched"));
    }
    let e0 = t.env_snapshot("start");
    let run = (|| -> Result<()> {
        // ---- dense: probe, plan, search
        t.probe(0)?;
        let mut plan = t.build_plan(0);
        if let Some(o) = &only { plan.retain(|(f, _)| o.iter().any(|x| f.name.starts_with(x.as_str()))); }
        let est: f64 = plan.iter().map(|(f, tt)| t.est_s(f, *tt)).sum();
        let order: Vec<String> = plan.iter().map(|(f, tt)| format!("{} [{}] ({} cand)", f.name, tt.name(), f.cands.len())).collect();
        t.say(format!("PLAN (dense): {} families, est {:.1} min, budget {budget_min} min — order: {}",
                      plan.len(), est / 60.0, order.join(", ")));
        t.search(plan, max_sweeps)?;
        // ---- sparse point (a real long prompt past the dense/sparse switch)
        let sparse_len = sparse_ctx.unwrap_or(0).max(model.qsa_limit() + 32);
        let tiled: Vec<u32> = {
            let src: Vec<u32> = corpus.iter().flatten().copied().collect();
            let want = sparse_len.max(p_len_for_prefill(chunk));
            src.iter().cycle().take(want.min(max_pos.saturating_sub(max_new + 2 * (MTP_MAX_K + 2) + 8))).copied().collect()
        };
        if skip_sparse {
            t.families.insert("sparse point".into(), serde_json::json!({"status": "SKIPPED (--autotune-skip-sparse)"}));
        } else if !model.cfg.has_indexer() || tiled.len() < sparse_len {
            t.families.insert("sparse point".into(), serde_json::json!({"status": format!(
                "UNTUNED (posture max_pos {max_pos} leaves no sparse positions past the switch at {})", model.qsa_limit())}));
        } else {
            t.corpus = vec![tiled[..sparse_len].to_vec()];
            t.corpus_label = "sparse (tiled corpus past the dense/sparse switch — an approximation)";
            t.gen = None;
            t.gen_next = 0;
            t.regime = 1;
            t.probe(1)?;
            let mut plan = t.build_plan(1);
            if let Some(o) = &only { plan.retain(|(f, _)| o.iter().any(|x| f.name.starts_with(x.as_str()))); }
            t.say(format!("PLAN (sparse, ctx {sparse_len} tokens, {}): {} families", t.corpus_label, plan.len()));
            t.search(plan, 1)?;
            t.corpus = corpus.clone();
            t.gen = None;
            t.regime = 0;
        }
        // ---- prefill point (the B3 known answer)
        if skip_prefill {
            t.families.insert("prefill.wide_mt".into(), serde_json::json!({"status": "SKIPPED (--autotune-skip-prefill)"}));
        } else {
            let tiled_pf: Vec<u32> = corpus.iter().flatten().copied().cycle().take(64 + chunk).collect();
            t.prefill_phase(&tiled_pf)?;
        }
        Ok(())
    })();
    // the session's tuner graphs + events go before anything else touches the stream
    let session = SESSION.with(|c| c.borrow_mut().take());
    let mut position_effect = serde_json::Value::Array(Vec::new());
    if let Some(s) = session.as_ref() {
        for smp in &s.samples {
            let variant = if smp.arm == usize::MAX { serde_json::json!("canonical") }
                          else if smp.arm == WARMUP_ARM { serde_json::json!("warmup") } else { serde_json::json!(smp.arm) };
            let _ = writeln!(t.jsonl, "{}", serde_json::json!({
                "case": format!("{}:m{}", smp.unit, smp.m), "variant": variant, "phase": smp.phase, "trial": smp.round,
                "time_us": smp.ms * 1e3, "correct": smp.digest_ok, "contract": format!("{}:{}:graph", &fp.tune_build_id, smp.mode),
                "round": smp.round, "m": smp.m, "regime": if smp.regime == 1 { "sparse" } else { "dense" },
                "pos": smp.pos, "prompt": smp.prompt, "mode": smp.mode,
                "arm_pos": if smp.apos == NO_APOS { serde_json::Value::Null } else { serde_json::json!(smp.apos) },
                "n_pos": smp.npos }));
        }
        // §5.2 v2: the run's replay-position effect (table provenance -> --autotune-report)
        let offs = pos_table(&s.samples, &|_| true);
        position_effect = position_effect_json(&s.samples, &offs);
        let _ = writeln!(t.jsonl, "{}", serde_json::json!({"position_effect": position_effect.clone()}));
    }
    drop(session);
    run?;
    // ---- the final gate
    let prompts: Vec<(&'static str, Vec<u32>, bool)> = vec![
        ("ansic", tok.encode(CORPUS[0], false)?, false),
        ("python", tok.encode(PY_PROMPT, false)?, false),
        ("prose", tok.encode(CORPUS[1], false)?, false),
        ("scifi", tok.encode(SCIFI_PROMPT, false)?, true),
    ];
    let (pass, confirm) = if skip_gate {
        t.say("FINAL GATE: SKIPPED (--autotune-skip-gate) — the table is written UNGATED (diagnostic runs only)".into());
        (true, serde_json::json!({"skipped": "--autotune-skip-gate", "pass": false}))
    } else {
        t.final_gate(li_rounds, hash_tokens, &prompts)?
    };
    let e1 = t.env_snapshot("end");
    // ---- the table (§7.1)
    let clocks: Vec<f64> = t.gpu_log.iter().filter_map(sm_clock).collect();
    let temps: Vec<f64> = t.gpu_log.iter().filter_map(|e| e.get("temperature.gpu").and_then(|v| v.as_str())
        .and_then(|s| s.trim().parse::<f64>().ok())).collect();
    let mm = |v: &[f64]| if v.is_empty() { serde_json::Value::Null } else {
        serde_json::json!([v.iter().cloned().fold(f64::INFINITY, f64::min), v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)]) };
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default().trim().to_string();
    let ips: Vec<String> = local_ipv4s();
    let uuid = tune::run_timeout("nvidia-smi", &["--query-gpu=uuid", "--format=csv,noheader"], std::time::Duration::from_secs(5))
        .map(|s| s.trim().to_string()).unwrap_or_default();
    let noise: serde_json::Map<String, serde_json::Value> = t.base_mad.iter()
        .map(|((u, m, r), v)| (format!("{u}_m{m}{}", if *r == 1 { "_sparse" } else { "" }), serde_json::json!(v))).collect();
    let decisions: Vec<tune::Decision> = t.table_decisions();
    // the not-searched list (every registered entry is accounted for)
    for def in tune::REGISTRY.iter() {
        if let Some(why) = t.prune_reason(def) {
            if !t.families.keys().any(|k| k.starts_with(def.id)) {
                t.families.insert(def.id.to_string(), serde_json::json!({"status": format!("NOT SEARCHED: {why}")}));
            }
        }
    }
    let mut table = tune::Table {
        format: tune::TABLE_FORMAT.into(), origin: origin.clone(), fingerprint: fp.clone(), posture: posture.clone(),
        provenance: serde_json::json!({
            "binary_sha256": bin_sha, "kernel_build_id": env!("KERNEL_BUILD_ID"), "tune_build_id": env!("TUNE_BUILD_ID"),
            "source_build_id": option_env!("SOURCE_BUILD_ID").unwrap_or(""), "commit": env!("TUNE_GIT_COMMIT"),
            "dirty": env!("TUNE_GIT_DIRTY"), "box": format!("{host} ({})", ips.join(",")), "gpu_uuid": uuid,
            "date_unix": stamp, "clocks": {"sm_mhz": mm(&clocks), "temp_c": mm(&temps), "throttle_events": t.throttle_events},
            "resident_apps": [], "corpus_sha256": corpus_sha, "mode": "full", "classes": ["S"],
            "budget_min": budget_min, "elapsed_min": t.elapsed_s() / 60.0, "noise_floor_ms": noise, "seed": seed,
            "receipts": jsonl_path, "env_start": e0, "env_end": e1,
            "protocol": protocol_json(warmup, stratify), "position_effect": position_effect,
        }),
        decisions,
        families: serde_json::Value::Object(t.families.clone()),
        confirm,
        table_sha256: String::new(),
    };
    let path = match arg(args, "--autotune-table") { Some(p) => p.to_string(), None => tune::table_file_name(&table) };
    // a rejected or ungated table never lands where `--tune-table auto` looks, and carries
    // confirm.pass = false (boot's auto selection refuses it anyway); load it only by explicit path
    let fname = path.rsplit('/').next().unwrap_or("table.json").to_string();
    let (dest, written) = if skip_gate {
        (format!("{out_dir}/ungated-{stamp}.{fname}"), true)
    } else if pass {
        (path.clone(), true)
    } else {
        (format!("{out_dir}/rejected-{stamp}.{fname}"), false)
    };
    let sha = tune::write_table(&mut table, &dest)?;
    let raw = std::fs::read(&dest)?;
    print!("{}", tune::report(&table, &raw));
    t.say(format!("AUTOTUNE RESULT: {} {dest} (file sha256 {sha}, {} decision(s), {:.1} min; commit {} dirty {}, binary {bin_sha}, \
                   tune_build_id {}, model {dir})",
                  if skip_gate { "UNGATED TABLE (diagnostic; auto never loads it):" }
                  else if written { "TABLE WRITTEN" } else { "REJECTED by the final gate — kept out of ./tune:" },
                  table.decisions.len(), t.elapsed_s() / 60.0, env!("TUNE_GIT_COMMIT"), env!("TUNE_GIT_DIRTY"), env!("TUNE_BUILD_ID")));
    Ok(written)
}

/// Prefill point: the tiled corpus must hold the prefix + the largest chunk.
fn p_len_for_prefill(chunk: usize) -> usize { 64 + chunk }
