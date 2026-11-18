//! Position-effect analysis, protocol arguments and the `--autotune` command-line entry (`cli`, `report_cli`, `cli_inner`, the self-test driver).
//! Part of the `--autotune` core (see `mod.rs` for the file map).

use super::*;

/// The position model's cell: (unit, width, regime, timed replays in the block).
pub(super) type PosCell = (&'static str, usize, u8, u8);
/// Fitted position offsets per cell (ms, indexed by `apos`); an empty table = no stratification.
pub(super) type PosTable = HashMap<PosCell, Vec<f64>>;

pub(super) fn pos_cell(s: &Sample) -> PosCell { (s.unit, s.m, s.regime, s.npos) }

/// §5.2 v2: fit the position model (tune::position_offsets) for every SR cell `want` accepts, over
/// ALL timed replays of the session in that cell (the effect belongs to the protocol, not to one arm
/// set: every phase's arms enter with their own arm effect).
pub(super) fn pos_table(samples: &[Sample], want: &dyn Fn(&PosCell) -> bool) -> PosTable {
    let mut by: HashMap<PosCell, Vec<tune::PosObs>> = HashMap::new();
    for s in samples.iter().filter(|s| s.mode == "sr" && s.digest_ok && s.apos != NO_APOS && s.npos >= 2) {
        let cell = pos_cell(s);
        if !want(&cell) { continue; }
        by.entry(cell).or_default().push(tune::PosObs { round: s.round as u64, arm: ((s.phase as u64) << 16) | s.arm as u64,
                                                        pos: s.apos as usize, ms: s.ms });
    }
    by.into_iter().map(|(c, v)| (c, tune::position_offsets(&v, c.3 as usize))).collect()
}

/// A timed replay's time with its fitted position offset removed (the raw time when no offsets).
pub(super) fn adj_ms(s: &Sample, offs: &PosTable) -> f64 {
    s.ms - offs.get(&pos_cell(s)).and_then(|f| f.get(s.apos as usize)).copied().unwrap_or(0.0)
}

/// Paired deltas (ms) of `arm` over the base arm for (unit, m): per (prompt, round) pair where both
/// ran and both digests held; position-stratified when `offs` holds the cell (v2).
pub(super) fn paired(samples: &[Sample], unit: &str, m: usize, arm: usize, offs: &PosTable) -> Vec<f64> {
    let base: HashMap<(usize, usize), f64> = samples.iter()
        .filter(|s| s.mode == "sr" && s.unit == unit && s.m == m && s.arm == 0 && s.digest_ok)
        .map(|s| ((s.prompt, s.round), adj_ms(s, offs))).collect();
    samples.iter()
        .filter(|s| s.mode == "sr" && s.unit == unit && s.m == m && s.arm == arm && s.digest_ok)
        .filter_map(|s| base.get(&(s.prompt, s.round)).map(|b| adj_ms(s, offs) - b))
        .collect()
}

/// §5.7 test 1 (A/A null), v3 (2026-09-27, design §5.7 amendment): tau_selftest = max(abs, pct % of
/// the cell's base median) — defaults 0.02 ms and 0.05 %, `--autotune-selftest-tau <ms>` /
/// `--autotune-selftest-tau-pct <pct>`. PASS when |HL| < tau AND (the 95% CI covers 0 OR it lies
/// wholly inside +-tau). v1/v2 required the CI to cover 0, which a tight CI (large n) around a
/// practically-null offset fails by chance (3 cells at 95% ~ 14% family-wise) — receipt
/// selftest-1790486005: HL +0.0188 ms, CI95 [+0.0005, +0.0368] at verify m = 3.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct AaRule {
    pub(super) abs_ms: f64,
    pub(super) pct: f64,
}
impl AaRule {
    pub(super) const DEFAULT: AaRule = AaRule { abs_ms: 0.02, pct: 0.05 };
    /// tau_selftest of a cell whose base replays have median `cell_median_ms`.
    pub(super) fn tau(&self, cell_median_ms: f64) -> f64 {
        let rel = if cell_median_ms.is_finite() { self.pct / 100.0 * cell_median_ms.abs() } else { 0.0 };
        self.abs_ms.max(rel)
    }
    fn json(&self) -> serde_json::Value {
        serde_json::json!({"rule": "|HL| < tau AND (CI95 covers 0 OR CI95 within +-tau)",
                           "tau": format!("max({} ms, {} % of the cell's base median)", self.abs_ms, self.pct)})
    }
}
/// The A/A verdict at tau: (pass, the clause that decided).
pub(super) fn aa_verdict(e: &tune::Est, tau: f64) -> (bool, &'static str) {
    if !(e.est.abs() < tau) { return (false, "|HL| >= tau"); }
    if e.lo <= 0.0 && 0.0 <= e.hi { return (true, "CI covers 0"); }
    if e.lo >= -tau && e.hi <= tau { return (true, "CI within +-tau"); }
    (false, "CI excludes 0 and leaves +-tau")
}
pub(super) fn aa_ok(e: &tune::Est, tau: f64) -> bool { aa_verdict(e, tau).0 }
/// `--autotune-selftest-tau <ms>` (> 0, default 0.02) and `--autotune-selftest-tau-pct <pct>` (>= 0,
/// default 0.05): the A/A threshold of the self-test.
pub(super) fn aa_args(args: &[String]) -> Result<AaRule> {
    let mut r = AaRule::DEFAULT;
    if let Some(v) = arg(args, "--autotune-selftest-tau") {
        r.abs_ms = v.parse::<f64>().ok().filter(|x| x.is_finite() && *x > 0.0)
            .with_context(|| format!("--autotune-selftest-tau must be a positive number of ms (got {v})"))?;
    }
    if let Some(v) = arg(args, "--autotune-selftest-tau-pct") {
        r.pct = v.parse::<f64>().ok().filter(|x| x.is_finite() && *x >= 0.0 && *x <= 10.0)
            .with_context(|| format!("--autotune-selftest-tau-pct must be a percentage in [0, 10] (got {v})"))?;
    }
    Ok(r)
}
/// Median of the base arm's timed SR replays in one (unit, m) cell (ms).
fn cell_median(samples: &[Sample], unit: &str, m: usize) -> f64 {
    let v: Vec<f64> = samples.iter().filter(|s| s.mode == "sr" && s.unit == unit && s.m == m && s.arm == 0 && s.digest_ok)
        .map(|s| s.ms).collect();
    tune::median(&v).unwrap_or(f64::NAN)
}
/// §5.7 test 2 (known positive): the spin read within +-0.03 ms of its value (unchanged by v2).
pub(super) fn spin_ok(e: &tune::Est, spin_ms: f64) -> bool { (e.est - spin_ms).abs() <= 0.03 }

/// Per-position view of one (unit, m) cell for the self-test / receipts: (warm-up median, timed
/// median by position, per-arm position counts). Times in ms.
fn position_view(samples: &[Sample], unit: &str, m: usize, arms: usize)
    -> (Option<f64>, Vec<Option<f64>>, Vec<Vec<usize>>, usize) {
    let sel = |s: &&Sample| s.unit == unit && s.m == m && s.digest_ok;
    let warm: Vec<f64> = samples.iter().filter(sel).filter(|s| s.mode == "warm").map(|s| s.ms).collect();
    let timed: Vec<&Sample> = samples.iter().filter(sel).filter(|s| s.mode == "sr" && s.apos != NO_APOS).collect();
    let np = timed.iter().map(|s| s.apos as usize + 1).max().unwrap_or(0);
    let by_pos: Vec<Option<f64>> = (0..np).map(|p| tune::median(&timed.iter().filter(|s| s.apos as usize == p)
        .map(|s| s.ms).collect::<Vec<_>>())).collect();
    let mut counts = vec![vec![0usize; np]; arms];
    for s in &timed { if s.arm < arms { counts[s.arm][s.apos as usize] += 1; } }
    let rounds = timed.iter().map(|s| (s.prompt, s.round)).collect::<std::collections::HashSet<_>>().len();
    (tune::median(&warm), by_pos, counts, rounds)
}

/// The run's position effect per cell (unit x width x regime x timed replays, >= 8 rounds): the
/// warm-up median, the timed median by position and the fitted offsets (table provenance, receipts,
/// `--autotune-report`).
pub(super) fn position_effect_json(samples: &[Sample], offs: &PosTable) -> serde_json::Value {
    let mut cells: Vec<PosCell> = samples.iter().filter(|s| s.mode == "sr" && s.apos != NO_APOS && s.npos >= 2)
        .map(pos_cell).collect::<std::collections::HashSet<_>>().into_iter().collect();
    cells.sort();
    let r3 = |x: f64| (x * 1e3).round() / 1e3;
    let mut out = Vec::new();
    for c in cells {
        let in_cell = |s: &&Sample| pos_cell(s) == c && s.digest_ok;
        let timed: Vec<&Sample> = samples.iter().filter(in_cell).filter(|s| s.mode == "sr" && s.apos != NO_APOS).collect();
        let rounds = timed.iter().map(|s| s.round).collect::<std::collections::HashSet<_>>().len();
        if rounds < 8 { continue; }
        let warm: Vec<f64> = samples.iter().filter(in_cell).filter(|s| s.mode == "warm").map(|s| s.ms).collect();
        let by_pos: Vec<Option<f64>> = (0..c.3 as usize).map(|p| tune::median(&timed.iter()
            .filter(|s| s.apos as usize == p).map(|s| s.ms).collect::<Vec<_>>())).collect();
        let seen: Vec<f64> = by_pos.iter().flatten().copied().collect();
        let spread = if seen.is_empty() { None } else {
            Some(seen.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - seen.iter().cloned().fold(f64::INFINITY, f64::min))
        };
        out.push(serde_json::json!({
            "unit": c.0, "m": c.1, "regime": if c.2 == 1 { "sparse" } else { "dense" }, "n_pos": c.3, "rounds": rounds,
            "warmup_median_ms": tune::median(&warm).map(r3),
            "median_ms_by_pos": by_pos.iter().map(|x| x.map(r3)).collect::<Vec<_>>(),
            "offsets_ms": offs.get(&c).map(|f| f.iter().map(|x| r3(*x)).collect::<Vec<_>>()),
            "spread_ms": spread.map(r3),
        }));
    }
    serde_json::Value::Array(out)
}

/// `--autotune-warmup N` (0..=4, default 2 since v3) and `--autotune-stratify on|off` (default on).
pub(super) fn protocol_args(args: &[String]) -> Result<(usize, bool)> {
    let warmup = match arg(args, "--autotune-warmup") {
        None => DEFAULT_WARMUP,
        Some(v) => v.parse::<usize>().ok().filter(|&w| w <= MAX_WARMUP)
            .with_context(|| format!("--autotune-warmup must be 0..={MAX_WARMUP}"))?,
    };
    let stratify = match arg(args, "--autotune-stratify") {
        None | Some("on") => true,
        Some("off") => false,
        Some(v) => bail!("--autotune-stratify must be on or off (got {v})"),
    };
    Ok((warmup, stratify))
}

pub(super) fn protocol_json(warmup: usize, stratify: bool) -> serde_json::Value {
    serde_json::json!({"version": 3, "warmup_replays": warmup, "order": "williams (balanced per unit x width x regime x arms)",
                       "analysis": if stratify { "position-stratified (median-polish offsets)" } else { "plain paired" },
                       "li": "williams schedule; first round(s) of every generation / segment untimed"})
}

/// `gb10_inference --autotune [--selftest]` / `--autotune-report <table>`. Returns the exit code:
/// 0 = PASS / table written, 1 = FAIL / table rejected by its gate, 2 = refused or error.
pub fn cli(args: &[String]) -> i32 {
    let r = if let Some(path) = arg(args, "--autotune-report") {
        report_cli(path)
    } else if args.iter().any(|a| a == "--selftest") {
        cli_inner(args)
    } else {
        tune_cli(args)
    };
    match r {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(e) => {
            eprintln!("AUTOTUNE_FAIL: {e:#}");
            2
        }
    }
}

/// `--autotune-report <table>`: parse + print (no GPU, no box rule: it only reads a file).
fn report_cli(path: &str) -> Result<bool> {
    let raw = std::fs::read(path).with_context(|| format!("read {path}"))?;
    let t: tune::Table = serde_json::from_slice(&raw).with_context(|| format!("parse {path}"))?;
    print!("{}", tune::report(&t, &raw));
    Ok(t.seal_ok())
}

fn cli_inner(args: &[String]) -> Result<bool> {
    // the box rule first (G-T8): on .11/.12 the refusal is the signal, before any pack check
    if let Err(why) = preflight() {
        println!("AUTOTUNE REFUSED: {why}");
        bail!("preflight refused: {why}");
    }
    let dir = arg(args, "--model-dir").context("--autotune --selftest requires --model-dir <EXL3 pack>")?.to_string();
    if !crate::exl3_serve::is_exl3_pack(&dir) { bail!("{dir} is not an EXL3 pack"); }
    let num = |k: &str, d: usize| -> Result<usize> {
        match arg(args, k) { None => Ok(d), Some(v) => v.parse().map_err(|_| anyhow::anyhow!("{k} must be an integer")) }
    };
    let max_pos = num("--max-seq-len", 8192)?;
    let chunk = num("--prefill-chunk", crate::exl3_serve::DEFAULT_PREFILL_CHUNK)?.max(1);
    let max_new = num("--max-new-tokens", 384)?.max(8);
    let li_steps = num("--autotune-li-steps", 64)?;
    let out_dir = arg(args, "--autotune-out").unwrap_or("tune/receipts").to_string();
    let seed: u64 = match arg(args, "--autotune-seed") {
        Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--autotune-seed must be an integer"))?,
        None => std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1),
    };
    let (warmup, stratify) = protocol_args(args)?;
    let aa_rule = aa_args(args)?;
    let spin_ns = 200_000i32;

    // the measured config = the served defaults unless a table is named (--tune-table, default off)
    tune::arm_tuner();
    let sel = tune::TableSel::parse(Some(arg(args, "--tune-table").unwrap_or("off")))?;
    let posture = tune::Posture { lanes: 1, mtp_max_k: MTP_MAX_K, dds: false, kv_fmt: kv_fmt_from_opts(),
                                  max_pos, prefill_chunk: chunk };
    tune::boot(&tune::BootReq { sel, model_dir: &dir, posture: posture.clone(), profile_mhz: None,
                                draft_on: false, tp: 1 });

    std::fs::create_dir_all(&out_dir).with_context(|| format!("mkdir {out_dir}"))?;
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let jsonl_path = format!("{out_dir}/selftest-{stamp}.jsonl");
    let mut jsonl = std::io::BufWriter::new(std::fs::File::create(&jsonl_path).with_context(|| format!("create {jsonl_path}"))?);
    let bin_sha = std::fs::read("/proc/self/exe").map(|b| tune::sha256_hex(&b)).unwrap_or_else(|_| "unknown".into());

    let t_load = std::time::Instant::now();
    let model = FwdModel::load(&dir, 1, max_pos)?;
    let head = model.mtp.as_ref().context("the pack has no mtp.* draft head — the self-test needs MTP rounds")?;
    let mut sc = FwdModel::scratch(model.dev(), &model.cfg, MTP_MAX_K + 1, model.max_pos())?;
    let mut psc: Option<PrefillScratch> = None;
    let fp = tune::current_fingerprint(&dir, 1, tune::STOCK_PROFILE_MHZ);
    let fp_json = serde_json::to_value(&fp)?;
    let contract = tune::sha256_hex(fp_json.to_string().as_bytes())[..8].to_string();
    let tok = crate::tokenizer::QwenTokenizer::from_file(&format!("{}/tokenizer.json", dir.trim_end_matches('/')))?;
    let corpus: Vec<Vec<u32>> = match arg(args, "--autotune-corpus") {
        Some(f) => {
            let texts: Vec<String> = serde_json::from_str(&std::fs::read_to_string(f)?)
                .context("--autotune-corpus must be a JSON array of prompt strings")?;
            texts.iter().map(|t| tok.encode(t, false)).collect::<Result<_>>()?
        }
        None => CORPUS.iter().map(|t| tok.encode(t, false)).collect::<Result<_>>()?,
    };
    anyhow::ensure!(corpus.iter().all(|c| c.len() >= 2), "every corpus prompt needs >= 2 tokens");
    println!("AUTOTUNE SELFTEST: model {dir}, load {:.1} s, corpus {} prompts {:?} tokens, max_new {max_new}, \
              k {}, verify widths [3, 6], draft s {}, LI steps {li_steps}, seed {seed}, receipts {jsonl_path}",
             t_load.elapsed().as_secs_f64(), corpus.len(), corpus.iter().map(|c| c.len()).collect::<Vec<_>>(),
             MTP_MAX_K, MTP_MAX_K);
    println!("AUTOTUNE SELFTEST: protocol v3 — {} untimed base warm-up replay(s) per block, balanced (Williams) \
              arm order per cell, analysis {}; A/A: |HL| < tau AND (CI95 covers 0 OR CI95 within +-tau), \
              tau = max({} ms, {} % of the cell's base median)", warmup,
             if stratify { "position-stratified (--autotune-stratify off = plain paired)" } else { "plain paired (not stratified)" },
             aa_rule.abs_ms, aa_rule.pct);
    writeln!(jsonl, "{}", serde_json::json!({
        "receipt": "gb10-exl3-tune-selftest/3", "fingerprint": fp_json, "posture": serde_json::to_value(&posture)?,
        "binary_sha256": bin_sha, "commit": env!("TUNE_GIT_COMMIT"), "dirty": env!("TUNE_GIT_DIRTY"),
        "kernel_build_id": env!("KERNEL_BUILD_ID"), "model_dir": dir, "seed": seed, "contract": contract,
        "date_unix": stamp, "tune_status": format!("{:?}", tune::status().map(|s| (s.state, s.reason))),
        "protocol": protocol_json(warmup, stratify), "aa_rule": aa_rule.json(),
    }))?;
    writeln!(jsonl, "{}", serde_json::json!({"env": gpu_env("start")}))?;

    // base2 = the base config under a distinct signature (a separately captured A/A twin graph)
    let arms = vec![
        Arm::base(),
        Arm { name: "base2".into(), assign: vec![(&tune::T_SELFTEST_SPIN, None, 0)] },
        Arm { name: "spin".into(), assign: vec![(&tune::T_SELFTEST_SPIN, None, spin_ns)] },
    ];
    let cfg = SrCfg { verify_on: true, draft_on: true, li_on: false, verify_widths: vec![3, 6],
                      draft_s: MTP_MAX_K, mem_cap: 512 << 20 };
    let mut session = Session::new(&model, cfg.clone(), arms.clone(), seed)?;
    session.set_warmup(warmup)?;
    SESSION.with(|c| *c.borrow_mut() = Some(session));

    let mut traj_ok = true;
    let mut li_tok_ok = true;
    let mut traj_lines = Vec::new();
    let run = (|| -> Result<()> {
        for (pi, ids) in corpus.iter().enumerate() {
            // pass A: the served path, harness disarmed
            let a = run_mtp_pass(&model, &mut sc, head, ids, max_new, chunk, &mut psc)?;
            writeln!(jsonl, "{}", serde_json::json!({"env": gpu_env(&format!("prompt{pi} after pass A"))}))?;
            // re-check tenants at the family boundary (§6.4.3)
            if let Some(ap) = resident_apps() { if !ap.is_empty() { bail!("a tenant appeared: {ap:?}"); } }
            // pass B: the same generation with SR armed
            with_session(|s| { s.prompt = pi; s.cfg = SrCfg { li_on: false, ..cfg.clone() }; Ok(()) })?;
            ARMED.store(true, Ordering::SeqCst);
            let b = run_mtp_pass(&model, &mut sc, head, ids, max_new, chunk, &mut psc);
            ARMED.store(false, Ordering::SeqCst);
            let b = b?;
            with_session(|s| s.finish(&model))?;
            writeln!(jsonl, "{}", serde_json::json!({"env": gpu_env(&format!("prompt{pi} after pass B"))}))?;
            let same = a.tokens == b.tokens && a.rounds == b.rounds && a.drafted == b.drafted
                && a.accepted == b.accepted && a.accepts == b.accepts;
            traj_ok &= same;
            traj_lines.push(format!(
                "prompt {pi}: served tokens {} hash {:016x} rounds {} drafted {} accepted {} | SR-on tokens {} hash {:016x} \
                 rounds {} drafted {} accepted {} -> {}",
                a.tokens.len(), fnv64(&a.tokens), a.rounds, a.drafted, a.accepted,
                b.tokens.len(), fnv64(&b.tokens), b.rounds, b.drafted, b.accepted, if same { "IDENTICAL" } else { "DIFFERENT" }));
            writeln!(jsonl, "{}", serde_json::json!({"trajectory": {
                "prompt": pi, "served": {"tokens": a.tokens.len(), "hash": format!("{:016x}", fnv64(&a.tokens)),
                                          "rounds": a.rounds, "drafted": a.drafted, "accepted": a.accepted},
                "sr_on": {"tokens": b.tokens.len(), "hash": format!("{:016x}", fnv64(&b.tokens)),
                          "rounds": b.rounds, "drafted": b.drafted, "accepted": b.accepted},
                "identical": same }}))?;
            // LI segment: plain steps through the served forward_step, arms from the LI schedule
            if li_steps > 1 {
                with_session(|s| { s.prompt = pi; s.cfg = SrCfg { verify_on: false, draft_on: false, li_on: true, ..cfg.clone() }; Ok(()) })?;
                let plain = run_plain_pass(&model, &mut sc, ids, li_steps, chunk, &mut psc, true)?;
                with_session(|s| s.finish(&model))?;
                let n = plain.len().min(a.tokens.len());
                let ok = plain[..n] == a.tokens[..n];
                li_tok_ok &= ok;
                traj_lines.push(format!("prompt {pi}: LI plain greedy {} tokens == MTP served prefix: {}", n, if ok { "YES" } else { "NO" }));
            }
        }
        Ok(())
    })();
    let session = SESSION.with(|c| c.borrow_mut().take()).context("session vanished")?;
    run?;
    writeln!(jsonl, "{}", serde_json::json!({"env": gpu_env("end")}))?;

    // ---- receipts: every sample (v2: + the arm's place in its timed block and the block size)
    let arm_name = |a: usize| if a == usize::MAX { "canonical" } else if a == WARMUP_ARM { "warmup" }
                              else { arms.get(a).map_or("?", |x| x.name.as_str()) };
    for s in &session.samples {
        writeln!(jsonl, "{}", serde_json::json!({
            "case": format!("{}:m{}", s.unit, s.m), "variant": arm_name(s.arm), "trial": s.round,
            "time_us": s.ms * 1e3, "correct": s.digest_ok, "contract": format!("{contract}:{}:graph", s.mode),
            "round": s.round, "m": s.m, "fam": s.unit, "regime": if s.regime == 1 { "sparse" } else { "dense" },
            "pos": s.pos, "prompt": s.prompt, "mode": s.mode,
            "arm_pos": if s.apos == NO_APOS { serde_json::Value::Null } else { serde_json::json!(s.apos) },
            "n_pos": s.npos }))?;
    }

    // ---- §5.7 tests (v2: the verdict follows --autotune-stratify; both analyses are printed)
    let mut pass = true;
    let mut lines: Vec<String> = Vec::new();
    let cells: [(&str, usize); 3] = [("verify_dry", 3), ("verify_dry", 6), ("draft", MTP_MAX_K)];
    let fmt = |e: &tune::Est| format!("n={} HL={:+.4} ms CI95=[{:+.4}, {:+.4}]", e.n, e.est, e.lo, e.hi);
    let offs = pos_table(&session.samples, &|_| true);
    let none = PosTable::new();
    let (used, other, used_name, other_name) = if stratify { (&offs, &none, "position-stratified", "plain paired") }
                                               else { (&none, &offs, "plain paired", "position-stratified") };
    // (4) the per-position medians: the bias the protocol removes, visible
    for (unit, m) in cells {
        let (warm, by_pos, counts, rounds) = position_view(&session.samples, unit, m, arms.len());
        let f = offs.iter().filter(|(c, _)| c.0 == unit && c.1 == m).max_by_key(|(c, _)| c.3).map(|(_, f)| f.clone());
        let ms = |x: Option<f64>| x.map_or("-".to_string(), |v| format!("{v:.3}"));
        let bal = counts.iter().all(|c| c.iter().max().unwrap_or(&0) - c.iter().min().unwrap_or(&0) <= 1);
        lines.push(format!(
            "SELFTEST POSITION {unit} m={m}: rounds {rounds} | warm-up median {} ms | timed median by position [{}] ms | \
             fitted offsets [{}] ms | arm x position counts {} -> {}",
            ms(warm), by_pos.iter().map(|x| ms(*x)).collect::<Vec<_>>().join(", "),
            f.map_or("-".into(), |f| f.iter().map(|x| format!("{x:+.3}")).collect::<Vec<_>>().join(", ")),
            counts.iter().enumerate().map(|(a, c)| format!("{}{:?}", arm_name(a), c)).collect::<Vec<_>>().join(" "),
            if bal { "balanced" } else { "UNBALANCED" }));
    }
    for (unit, m) in cells {
        let d = paired(&session.samples, unit, m, 1, used);
        let alt = tune::hl_paired(&paired(&session.samples, unit, m, 1, other));
        let alt_s = alt.map_or("n/a".into(), |e| fmt(&e));
        match tune::hl_paired(&d) {
            Some(e) if e.n >= 20 => {
                let med = cell_median(&session.samples, unit, m);
                let tau = aa_rule.tau(med);
                let (ok, why) = aa_verdict(&e, tau);
                pass &= ok;
                lines.push(format!("SELFTEST 1 A/A {unit} m={m}: {} ({used_name}) | tau {tau:.4} ms (max({}, {} % x base median \
                                    {med:.3})) -> {} ({why}) | {other_name}: {alt_s}",
                                   fmt(&e), aa_rule.abs_ms, aa_rule.pct, if ok { "PASS" } else { "FAIL" }));
            }
            other => {
                pass = false;
                lines.push(format!("SELFTEST 1 A/A {unit} m={m}: INSUFFICIENT samples ({})", other.map_or(0, |e| e.n)));
            }
        }
    }
    let spin_ms = spin_ns as f64 / 1e6;
    for (unit, m) in cells {
        let d = paired(&session.samples, unit, m, 2, used);
        let alt = tune::hl_paired(&paired(&session.samples, unit, m, 2, other));
        let alt_s = alt.map_or("n/a".into(), |e| fmt(&e));
        match tune::hl_paired(&d) {
            Some(e) if e.n >= 20 => {
                let ok = spin_ok(&e, spin_ms);
                pass &= ok;
                lines.push(format!("SELFTEST 2 known-positive spin {unit} m={m}: {} ({used_name}; target {spin_ms:+.3} +- 0.030) \
                                    -> {} | {other_name}: {alt_s}", fmt(&e), if ok { "PASS" } else { "FAIL" }));
            }
            other => {
                pass = false;
                lines.push(format!("SELFTEST 2 known-positive spin {unit} m={m}: INSUFFICIENT samples ({})", other.map_or(0, |e| e.n)));
            }
        }
    }
    let xok = session.xfail.is_empty() && session.canon_checks.0 > 0;
    pass &= traj_ok && xok;
    for l in &traj_lines { lines.push(format!("SELFTEST 3 trajectory {l}")); }
    lines.push(format!("SELFTEST 3 trajectory invariance (greedy tokens + [mtp-stats] rounds/drafted/accepted, SR on vs off): {}",
                       if traj_ok { "PASS" } else { "FAIL" }));
    lines.push(format!("SELFTEST digests: base + warm-up replays == canonical {}/{}; arms == base {}/{} -> {}",
                       session.canon_checks.0 - session.canon_checks.1, session.canon_checks.0,
                       session.arm_checks.0 - session.arm_checks.1, session.arm_checks.0,
                       if xok { "PASS" } else { "FAIL" }));
    for f in session.xfail.iter().take(20) { lines.push(f.clone()); }
    // LI (informational: unpaired, served conditions; the segment warm-ups excluded)
    let li_of = |arm: usize| -> Vec<f64> { session.samples.iter().filter(|s| s.mode == "li" && s.arm == arm).map(|s| s.ms).collect() };
    {
        let li_warm: Vec<f64> = session.samples.iter().filter(|s| s.unit == "step" && s.mode == "warm").map(|s| s.ms).collect();
        let li_pos: Vec<String> = (0..arms.len()).map(|p| tune::median(&session.samples.iter()
            .filter(|s| s.mode == "li" && s.apos as usize == p).map(|s| s.ms).collect::<Vec<_>>())
            .map_or("-".into(), |x| format!("{x:.3}"))).collect();
        lines.push(format!("SELFTEST INFO LI step position: warm-up median {} ms (n {}) | median by in-block position [{}] ms",
                           tune::median(&li_warm).map_or("-".into(), |x| format!("{x:.3}")), li_warm.len(), li_pos.join(", ")));
    }
    for (arm, name) in [(1usize, "A/A base2"), (2, "spin")] {
        if let Some(e) = tune::hl_two_sample(&li_of(0), &li_of(arm)) {
            lines.push(format!("SELFTEST INFO LI step {name} vs base: {}", fmt(&e)));
        }
    }
    lines.push(format!("SELFTEST INFO LI plain greedy == served MTP greedy prefix: {}", if li_tok_ok { "YES" } else { "NO" }));
    let canon: Vec<f64> = session.samples.iter().filter(|s| s.arm == usize::MAX).map(|s| s.ms).collect();
    if !canon.is_empty() {
        let mut c = canon.clone();
        c.sort_by(|a, b| a.partial_cmp(b).unwrap());
        lines.push(format!("SELFTEST INFO canonical verify m={}: median {:.3} ms over {} rounds", MTP_MAX_K + 1, c[c.len() / 2], c.len()));
    }
    lines.push(format!("SELFTEST INFO tuner graphs captured {} (mem cap hit: {})", session.captures, session.mem_cap_hit));
    let verdict = format!("SELFTEST RESULT: {} (commit {} dirty {}, binary sha256 {}, kernel_build_id {}, tune_build_id {}, \
                           model {dir}, date_unix {stamp}, receipts {jsonl_path})",
                          if pass { "PASS" } else { "FAIL" }, env!("TUNE_GIT_COMMIT"), env!("TUNE_GIT_DIRTY"), bin_sha,
                          env!("KERNEL_BUILD_ID"), env!("TUNE_BUILD_ID"));
    lines.push(verdict);
    for l in &lines { println!("{l}"); }
    writeln!(jsonl, "{}", serde_json::json!({"position_effect": position_effect_json(&session.samples, &offs),
                                             "protocol": protocol_json(warmup, stratify)}))?;
    writeln!(jsonl, "{}", serde_json::json!({"summary": lines, "pass": pass}))?;
    jsonl.flush()?;
    drop(session); // tuner graphs + events before the model's context goes away
    Ok(pass)
}
