//! WP24 gate, stage B (served level): `wp24_served_gate` and its arms / conditions, driven on the real scheduler. Gate tooling, not a serving path.

use super::*;

// ---- WP24 gate, stage B (served level): see src/wp24.rs for stage A and the statistics ----

/// The gate's three arms: ratio MTP (under test), match MTP (the control: today's rule, known
/// exact), plain sampled decode with MTP off (the reference law).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GateArm {
    Plain = 0,
    Ratio = 1,
    Match = 2,
}

struct GateArmOut {
    seqs: Vec<Vec<u32>>,
    stats: GateStats,
    secs: f64,
}

pub(super) const WP24_GATE_PROMPT: &str = "Write a short story about a lighthouse keeper who finds a message in a bottle.\n\nThe lighthouse keeper";

/// One arm of one condition: `trials` seeded requests of `prompt` through the REAL scheduler
/// (admit -> prefix-cache restore -> seam step -> prime -> step() rounds), `positions` tokens
/// each (ignore_eos, so every trial yields exactly that many), in waves of `width` lanes. The
/// plain arm clears each lane's MTP state after admit (the per-request §6 fallback's exact
/// hand-off: the lane then decodes through step_plain). Trial seeds: distinct per (condition,
/// arm, trial) — independent RNG streams everywhere (AGENTS §3).
#[allow(clippy::too_many_arguments)]
fn wp24_gate_arm(sched: &mut Exl3Scheduler, arm: GateArm, prompt: &[u32], t: f32, top_p: f32,
                 top_k: usize, trials: usize, positions: usize, base: u64) -> Result<GateArmOut> {
    sched.spec_ratio = arm == GateArm::Ratio;
    sched.cal = DraftCal::new(); // each arm learns its own DDS map (never another arm's labels)
    sched.gate = Some(GateStats::default());
    let t0 = std::time::Instant::now();
    let mut seqs: Vec<Vec<u32>> = Vec::with_capacity(trials);
    let mut done = 0usize;
    while done < trials {
        let nb = (trials - done).min(sched.width);
        let mut rxs = Vec::with_capacity(nb);
        for j in 0..nb {
            let (tx, rx, bl) = crate::server::tok_channel(); // LR-5 (H6): drained counted below
            let seed = crate::wp24::mix64(base ^ ((arm as u64 + 1) << 56) ^ (done + j) as u64);
            sched.admit(BatchRequest {
                prompt: prompt.to_vec(),
                max_new: positions,
                temperature: t,
                top_p,
                top_k,
                rep_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                min_p: 0.0,
                min_new: 0,
                ignore_eos: true,
                tx,
                seed: Some(seed),
                ckpt_at: None,
                domain: crate::batch::Domain::General,
                received_at: std::time::Instant::now(),
                image_embeds: None,
                image_spans: Vec::new(),
                schema: None,
                thinking_budget: None,
                logprobs: None,
            });
            if let Some(f) = sched.fatal.as_ref() { anyhow::bail!("WP24 gate: engine fatal at admit: {f}"); }
            rxs.push((rx, bl));
        }
        if arm == GateArm::Plain {
            for l in sched.lanes.iter_mut().flatten() { l.mtp = None; }
        }
        let mut steps = 0usize;
        while sched.lanes.iter().any(|l| l.is_some()) {
            sched.step()?;
            if let Some(f) = sched.fatal.as_ref() { anyhow::bail!("WP24 gate: engine fatal: {f}"); }
            steps += 1;
            anyhow::ensure!(steps <= 4 * positions + 64, "WP24 gate: lanes still live after {steps} steps");
        }
        for (mut rx, bl) in rxs {
            let (mut toks, mut reason) = (Vec::with_capacity(positions), None);
            while let Some(ev) = crate::server::try_recv_counted(&mut rx, &bl) { // LR-5 (H6)
                match ev {
                    TokEvent::Tok(x) => toks.push(x),
                    TokEvent::Finish { reason: r } => reason = Some(r),
                    // #8.2: admission telemetry — no action for a gate drain.
                    TokEvent::Admitted { .. } => {}
                    TokEvent::StopInfo { .. } => {}
                    TokEvent::Logprob { .. } => {} // not requested by this drain
                }
            }
            anyhow::ensure!(reason.as_deref() == Some("length") && toks.len() == positions,
                            "WP24 gate: a {arm:?} trial ended {reason:?} after {} tokens (want {positions}, \"length\")",
                            toks.len());
            seqs.push(toks);
        }
        done += nb;
    }
    let stats = sched.gate.take().unwrap_or_default();
    Ok(GateArmOut { seqs, stats, secs: t0.elapsed().as_secs_f64() })
}

/// `--probe-spec-sampling --model-dir <EXL3 pack>` stage B: the served-level distribution gate.
/// Per condition (T in {0.7, 1.0} x top_p in {0.8, 0.95}, top_k 20, >= 2 seeds): `--trials`
/// requests per arm of one raw prompt (`--prompt`, default a story opening), each generating
/// `--positions` tokens (default 6: x1 = the seam step, identical code in every arm — a built-in
/// control; x2.. = MTP rounds; the remaining-budget clamp lets round 1 draft up to positions-2).
/// Tests (reference-free two-sample chi-square, wp24::gof2 — |z| < 4 passes):
///   GATE    ratio-vs-plain per position and on the joint (x2, x3);
///   CONTROL match-vs-plain, the same tests (today's rule is exact: a failure = the harness is
///           broken, and the condition FAILS);
///   NEG     the ratio arm's first-round drafts d_0 vs plain x2 (d_0 ~ q' != p: must be REJECTED
///           in at least one condition, else the gate had no power on this prompt).
/// Success signals asserted, not just the absence of errors: every trial returns exactly
/// `positions` tokens with finish "length"; the ratio arm ran real-q rounds (>0) and the match
/// arm none; both MTP arms drafted. Output: one line per condition, then RESULT. `--quick` = the
/// owner's config (T 1.0, top_p 0.95) only, as the first rung of the ladder.
pub fn wp24_served_gate(args: &[String], model_dir: &str) -> Result<bool> {
    let num = |name: &str, d: usize| -> Result<usize> {
        match crate::opt_or_arg(args, name) {
            None => Ok(d),
            Some(v) => v.parse().map_err(|_| anyhow::anyhow!("{name} must be an integer")),
        }
    };
    let trials = num("--trials", 600)?.max(50);
    let positions = num("--positions", 6)?.clamp(3, 32);
    let width = num("--max-batch", 4)?.clamp(1, 16);
    let n_seeds = num("--seeds", 2)?.max(2);
    let top_k = num("--top-k", 20)?;
    let max_seq_len = num("--max-seq-len", 4096)?;
    let draft_conf: f64 = match arg(args, "--draft-confidence") {
        None => 0.4,
        Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--draft-confidence must be a number"))?,
    };
    anyhow::ensure!((0.0..1.0).contains(&draft_conf), "--draft-confidence must be in [0, 1)");
    let (_, draft_temp) = parse_spec_sampling(args)?; // --draft-temperature applies to the ratio arm
    let text = arg(args, "--prompt").unwrap_or(WP24_GATE_PROMPT).to_string();
    // the served posture: prefix snapshots on (run() sets this for --prefix-cache on), KV format
    // from --kv-cache / [kv-cache] / the default, graphs precaptured like the server boot
    crate::opts::set(crate::opt!("exl3-prefix"), "1");
    if let Some(v) = arg(args, "--kv-cache") {
        let v = exl3_kv_cache_value(v)?;   // TPSHIP: bf16 -> f16 with one line (no refusal)
        crate::opts::set(crate::opt!("kv-cache"), v);
    }
    if let Some(off) = crate::exl3_forward::mtp_off_by_opt() {
        anyhow::bail!("WP24 gate: {} is set — the MTP arms need the head", off.flag());
    }
    let max_pos = max_seq_len + crate::batch::decode_headroom(false);
    crate::dprintln!("WP24 served gate: loading {model_dir} (width {width}, max_pos {max_pos}, kv-cache {})",
             crate::exl3_forward::kv_fmt_name(crate::exl3_forward::kv_fmt_from_opts()));
    let model = FwdModel::load(model_dir, width, max_pos)?;
    anyhow::ensure!(model.mtp.is_some(), "WP24 gate: the pack has no MTP draft head");
    let tok = QwenTokenizer::from_file(&format!("{}/tokenizer.json", model_dir.trim_end_matches('/')))?;
    let cfg_eos: u32 = {
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/config.json", model_dir.trim_end_matches('/')))?)?;
        cfg.get("eos_token_id").and_then(|e| e.as_u64()).unwrap_or(248046) as u32
    };
    let eos = tok.stop_token_ids(cfg_eos);
    let prompt = tok.encode(&text, false)?;
    let mtp_k = crate::exl3_forward::mtp_depth_default();
    anyhow::ensure!(prompt.len() >= 2 && prompt.len() + positions + mtp_k + 4 < model.max_pos(),
                    "WP24 gate: prompt of {} tokens does not fit the window", prompt.len());
    let mut sc = FwdModel::scratch(model.dev(), &model.cfg, width.max(mtp_k + 1), model.max_pos())?;
    let chunk = 2048usize;
    let psc0 = model.prefill_scratch(chunk)?;
    model.precapture_decode_graphs(&mut sc, width, mtp_k, draft_conf > 0.0)?;
    let tok = Arc::new(tok);
    // S-B9-REL-MTPREPROBE policy flags (bench/probe paths construct the same scheduler).
    let reprobe_on = crate::opts::var(crate::opt!("mtp-reprobe")).map_or(true, |v| v != "off");
    let reprobe_cfg = if crate::opts::var(crate::opt!("mtp-reprobe-drill")).is_ok() {
        crate::mtp_reprobe::ReProbeCfg::drill()
    } else {
        crate::mtp_reprobe::ReProbeCfg::default()
    };
    let mut sched = Exl3Scheduler {
        spill: None, spill_pol: crate::kv_spill::Policy::new(0), spill_rank: 0,
        pool: crate::prefix_pool::Pool::new(0), pool_floor: 0, pool_spare: Vec::new(), slot_segs: Vec::new(), row_gen: 0, pool_emerg_at: None,
        model: model.clone(), sc, width,
        lanes: (0..width).map(|_| None).collect(), eos, chunk, interleave: 0, psc: Some(psc0),
        df2: None, df2_k: crate::dflash2::block() - 1, // run_spec probe path: no DF2 artifact
        copy_arm: None, // WIDEVERIFY1: the served-gate harness never copies
        record: None, // COPYREC1: nor keeps a record
        wide: Default::default(), cpv_price: Default::default(), cpv_on: false, cpv: None, // COPYPF1: nor runs a prefill-width copy verify
        cl_on: false, cl_priced: true, cl_price: Default::default(), // COPYLANES: nor copy work in busy steps
        mtp_k, spec_policy: SpecLanes::Never, lane_order: LaneOrder::Rr, lane_quantum: 256, admit_ctr: 0, fcfs_rots: 0,
        lane_pack: crate::lane_pack::LanePack::Off, scp: None, pack_ready: [false, false], pack_sampled: false, pack_rot: 0, pack_logs: 0, in_prefill: false, pack_stats: PackStats::default(), tr: SchedTrace::default(),
        reprobe_on, reprobe_cfg, shared_scale: 1.0, shared_last: false, shared_streak: 0, shared_probe_ctr: 0, spec_win_tok: [0.0; 2], spec_win_ms: [0.0; 2], spec_win_n: [0; 2], spec_win_acc: [0; 2], spec_arm_rate: [0.0; 2], spec_arm_n: [0; 2], spec_arm_fold: [0; 2], spec_rate_n: vec![[(0.0, 0); 2]; SPEC_RATE_NMAX + 1], spec_rate_folds: vec![[0; 2]; SPEC_RATE_NMAX + 1], meter_settle: 0, meter_last_n: 0, probe_fold: [false; 2], spec_probe: None, spec_probe_ctr: 0, spec_est_pick: None, spec_margin: 0.0, spec_margin_n: 0, policy_logs: 0, ctrl_round: 0, ema_plain: 0.0, draft_conf, cal: DraftCal::new(),
        reserve: None, reserve_st: crate::lane_reserve::State::new(), rel: (0..width).map(|_| None).collect(),
        prefix_on: true, cache: (0..width).map(|_| None).collect(),
        wp16: None, wp16_xcheck: false,
        loop_cfg: None, pen_range: 1024, spec_ratio: false, draft_temp, gate: None, tok: tok.clone(), tp: None,
        exit_on_fatal: false, fatal: None, inject_panic: None, steps: 0, last_resume: (0, 0), pf_done: Default::default(), pf_jobs: Vec::new(), mrope_section: None, vis_seam: None, kv_pf: None, turn: None,
    };
    let quick = args.iter().any(|a| a == "--quick");
    wp24_gate_conditions(&mut sched, &prompt, &text, trials, positions, n_seeds, top_k, draft_conf, draft_temp, quick)
}

/// The WP24 served gate's condition loop on a constructed scheduler (TP-C: shared by the TP=1
/// gate above and the TP=2 spec program, where every rank runs it in lockstep).
#[allow(clippy::too_many_arguments)]
pub(super) fn wp24_gate_conditions(sched: &mut Exl3Scheduler, prompt: &[u32], text: &str, trials: usize, positions: usize,
                        n_seeds: usize, top_k: usize, draft_conf: f64, draft_temp: Option<f32>, quick: bool) -> Result<bool> {
    let mtp_k = sched.mtp_k;
    sched.draft_conf = draft_conf;
    sched.draft_temp = draft_temp;
    crate::dprintln!("WP24 served gate: prompt {} tokens {:?}; {trials} trials x 3 arms (plain | ratio | match) per condition, \
              {positions} positions, top_k {top_k}, {n_seeds} seeds, DDS {draft_conf}, depth {mtp_k}, draft temperature {}",
             prompt.len(), text.chars().take(60).collect::<String>(),
             draft_temp.map_or("= T".to_string(), |x| x.to_string()));
    type Hist = std::collections::HashMap<u64, u64>;
    let hist = |seqs: &[Vec<u32>], j: usize| -> Hist {
        let mut h = Hist::new();
        for s in seqs { *h.entry(s[j] as u64).or_insert(0) += 1; }
        h
    };
    let pair = |seqs: &[Vec<u32>], j: usize| -> Hist {
        let mut h = Hist::new();
        for s in seqs { *h.entry(((s[j] as u64) << 32) | s[j + 1] as u64).or_insert(0) += 1; }
        h
    };
    let t_all = std::time::Instant::now();
    let (mut all_ok, mut neg_rejected, mut conds) = (true, false, 0usize);
    // --quick: the owner's daily config only (T 1.0, top_p 0.95) — the ladder's first rung; the
    // full gate is all four configs
    let configs: &[(f32, f32)] = if quick { &[(1.0, 0.95)] } else { &[(0.7, 0.8), (0.7, 0.95), (1.0, 0.8), (1.0, 0.95)] };
    let total = configs.len() * n_seeds;
    for &(t, top_p) in configs {
        for si in 0..n_seeds {
            let base = crate::wp24::mix64(0x5EED_2400_0000_0000u64
                ^ (((t.to_bits() as u64) << 32) | top_p.to_bits() as u64)
                ^ (si as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let plain = wp24_gate_arm(sched, GateArm::Plain, prompt, t, top_p, top_k, trials, positions, base)?;
            let ratio = wp24_gate_arm(sched, GateArm::Ratio, prompt, t, top_p, top_k, trials, positions, base)?;
            let matc = wp24_gate_arm(sched, GateArm::Match, prompt, t, top_p, top_k, trials, positions, base)?;
            let (mut zr, mut zm) = (Vec::with_capacity(positions + 1), Vec::with_capacity(positions + 1));
            for j in 0..positions {
                zr.push(crate::wp24::gof2_maps(&hist(&ratio.seqs, j), &hist(&plain.seqs, j)).1);
                zm.push(crate::wp24::gof2_maps(&hist(&matc.seqs, j), &hist(&plain.seqs, j)).1);
            }
            zr.push(crate::wp24::gof2_maps(&pair(&ratio.seqs, 1), &pair(&plain.seqs, 1)).1);
            zm.push(crate::wp24::gof2_maps(&pair(&matc.seqs, 1), &pair(&plain.seqs, 1)).1);
            let mut d0h = Hist::new();
            for &d in &ratio.stats.d0 { *d0h.entry(d as u64).or_insert(0) += 1; }
            let neg = crate::wp24::gof2_maps(&d0h, &hist(&plain.seqs, 1));
            let signals = ratio.stats.ratio_rounds > 0 && ratio.stats.drafted > 0
                && matc.stats.ratio_rounds == 0 && matc.stats.drafted > 0;
            let gate_ok = zr.iter().all(|z| z.abs() < 4.0);
            let ctrl_ok = zm.iter().all(|z| z.abs() < 4.0);
            let ok = gate_ok && ctrl_ok && signals;
            all_ok &= ok;
            neg_rejected |= neg.1 >= 4.0 && !ratio.stats.d0.is_empty();
            conds += 1;
            let fz = |v: &[f64]| v.iter().map(|z| format!("{z:+.2}")).collect::<Vec<_>>().join(" ");
            let arm = |a: &GateArmOut| {
                let s = &a.stats;
                format!("rounds {} ratio-rounds {} accept {:.1}% tok/round {:.2} ({:.0} s)", s.rounds, s.ratio_rounds,
                        100.0 * s.accepted as f64 / s.drafted.max(1) as f64,
                        (s.accepted + s.rounds) as f64 / s.rounds.max(1) as f64, a.secs)
            };
            crate::dprintln!("  [served] T={t} top_p={top_p} seed#{si}: n={trials}/arm | GATE ratio-vs-plain z[pos 1..{positions}, pair 2-3] = {} | \
                      CONTROL match-vs-plain z = {} | NEG d0-vs-plain-x2 z={:+.1} ({} d0) | ratio: {} | match: {} | plain {:.0} s{} -> {}",
                     fz(&zr), fz(&zm), neg.1, ratio.stats.d0.len(), arm(&ratio), arm(&matc), plain.secs,
                     if signals { "" } else { " | SUCCESS SIGNALS MISSING (ratio rounds / drafts)" },
                     if ok { "PASS" } else if !ctrl_ok { "FAIL (control failed: harness suspect)" } else { "FAIL" });
            if conds == 1 {
                let per = t_all.elapsed().as_secs_f64();
                crate::dprintln!("  [served] projected total ~{:.0} min ({total} conditions at {:.0} s each)", per * total as f64 / 60.0, per);
            }
        }
    }
    if !neg_rejected {
        crate::dprintln!("  [served] negative control never rejected: the gate had no power to see q' != p on this prompt");
        crate::dprintln!("RESULT: SERVED_GATE_NO_POWER");
        return Ok(false);
    }
    crate::dprintln!("RESULT: {} ({conds} conditions{}, {:.1} min)",
             if all_ok { "SERVED_DISTRIBUTION_OK (ratio-mode MTP emits the plain sampler's law at every tested position)" }
             else { "SERVED_DISTRIBUTION_MISMATCH" },
             if quick { ", --quick: T 1.0 / top_p 0.95 only — run without --quick for the full gate" } else { "" },
             t_all.elapsed().as_secs_f64() / 60.0);
    Ok(all_ok)
}
