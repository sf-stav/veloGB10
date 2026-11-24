//! The offline spec program (`run_spec`, the tpspec probe): the SERVED scheduler driven without HTTP, at TP=1 or as one rank of a TP pair,
//! with its option parser, arms and cross-rank fold. A bring-up and gate tool, not a serving path; it constructs the same `Exl3Scheduler` the server runs.

use super::*;

// ---------------------------------------------------------------------------------------------
// TP-C: the spec program (`--probe-exl3-tpspec`) — the SERVED scheduler, offline (no HTTP), at
// TP=1 (single process) or as one rank of TP=2 (SPMD, every rank runs this function). Every
// request goes through Exl3Scheduler::admit (prefix cache -> chunked prefill -> seam step ->
// head prime) and step() (graphed DDS rounds / plain steps / control rounds), exactly the
// serve's code; at TP=2 each step is a lockstep point (TpSync). Gates:
//   G-T1-e  greedy: every spec arm (DDS at the served depth, fixed chains at each --spec-depths)
//           emits exactly the plain arm's tokens, per prompt and rep (and plain is rep-stable);
//   G-T1-g  acceptance: tok/round of the DDS arm per prompt class (compare TP=2 vs the TP=1 run
//           of this same program);
//   speed   DDS ms/round and tok/s, plain ms/token (bring-up readings, not perf claims);
//   G-T1-b  (graphed) with ident on: logits rows / residual / taps digests identical across ranks
//           at every lockstep point of rep 0;
//   G-T1-h  the WP24 served distribution gate (ratio + match vs plain, chi-square) on the same
//           scheduler (--spec-samp quick|full).
// ---------------------------------------------------------------------------------------------

/// Options of the spec program; ride TpConfig.exl3_mode as
/// "spec:gen=N:reps=N:depths=1,2:dds=0|1:width=N:prefix=0|1:samp=0|1|2:trials=N:ident=0|1".
#[derive(Clone, Debug, PartialEq)]
pub struct SpecOpts {
    pub gen: usize,
    pub reps: usize,
    pub depths: Vec<usize>,
    pub dds: bool,
    pub width: usize,
    pub prefix: bool,
    /// 0 = no sampled gate, 1 = WP24 --quick, 2 = the full ladder
    pub samp: u8,
    pub trials: usize,
    pub ident: bool,
}

impl Default for SpecOpts {
    fn default() -> Self {
        SpecOpts { gen: 1024, reps: 3, depths: (1..=crate::exl3_forward::MTP_MAX_K).collect(), dds: true, width: 1,
                   prefix: true, samp: 0, trials: 600, ident: true }
    }
}

impl SpecOpts {
    pub fn parse(mode: &str) -> Result<Self> {
        let mut it = mode.split(':');
        anyhow::ensure!(it.next() == Some("spec"), "EXL3 TP: '{mode}' is not a spec program");
        let mut o = SpecOpts::default();
        for kv in it {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            let b = |v: &str| v == "1";
            match k {
                "gen" => o.gen = v.parse().context("spec gen=N")?,
                "reps" => o.reps = v.parse().context("spec reps=N")?,
                "depths" => o.depths = if v.is_empty() { Vec::new() } else {
                    v.split(',').map(|x| x.parse::<usize>().context("spec depths=a,b")).collect::<Result<_>>()?
                },
                "dds" => o.dds = b(v),
                "width" => o.width = v.parse().context("spec width=N")?,
                "prefix" => o.prefix = b(v),
                "samp" => o.samp = v.parse().context("spec samp=0|1|2")?,
                "trials" => o.trials = v.parse().context("spec trials=N")?,
                "ident" => o.ident = b(v),
                _ => anyhow::bail!("EXL3 TP: unknown spec option '{kv}' in '{mode}'"),
            }
        }
        anyhow::ensure!(o.depths.iter().all(|&d| (1..=crate::exl3_forward::MTP_MAX_K).contains(&d)),
                        "spec depths must be in 1..={}", crate::exl3_forward::MTP_MAX_K);
        anyhow::ensure!(o.width >= 1 && o.reps >= 1 && o.gen >= 2, "spec: width/reps >= 1, gen >= 2");
        Ok(o)
    }

    pub fn to_mode(&self) -> String {
        format!("spec:gen={}:reps={}:depths={}:dds={}:width={}:prefix={}:samp={}:trials={}:ident={}",
                self.gen, self.reps, self.depths.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","),
                self.dds as u8, self.width, self.prefix as u8, self.samp, self.trials, self.ident as u8)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SpecArm {
    Plain,
    Dds,
    Fixed(usize),
}

impl SpecArm {
    fn name(&self) -> String {
        match self { SpecArm::Plain => "plain".into(), SpecArm::Dds => "dds".into(), SpecArm::Fixed(d) => format!("d{d}") }
    }
}

struct SpecArmOut {
    toks: Vec<u32>,
    st: GateStats,
    secs: f64,
    reason: String,
}

/// One greedy request of `prompt` through the real scheduler under `arm` (plain = the §6 per-request
/// fallback's exact hand-off: the lane's MTP state cleared after admit, then step_plain).
fn spec_arm(sched: &mut Exl3Scheduler, arm: SpecArm, prompt: &[u32], gen: usize) -> Result<SpecArmOut> {
    let kdef = crate::exl3_forward::mtp_depth_default();
    match arm {
        SpecArm::Plain | SpecArm::Dds => { sched.mtp_k = kdef; sched.draft_conf = 0.4; }
        SpecArm::Fixed(d) => { sched.mtp_k = d; sched.draft_conf = 0.0; }
    }
    sched.cal = DraftCal::new();
    sched.gate = Some(GateStats::default());
    let (tx, mut rx, bl) = crate::server::tok_channel(); // LR-5 (H6): drained counted below
    let t0 = std::time::Instant::now();
    sched.admit(BatchRequest {
        prompt: prompt.to_vec(), max_new: gen, temperature: 0.0, top_p: 1.0, top_k: 0,
        rep_penalty: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0, min_p: 0.0, min_new: 0,
        ignore_eos: false, tx, seed: Some(0x7C0D_E5EE_D000_0001), ckpt_at: None,
        domain: crate::batch::Domain::General, received_at: std::time::Instant::now(),
        image_embeds: None, image_spans: Vec::new(), schema: None, thinking_budget: None, logprobs: None,
    });
    if let Some(f) = sched.fatal.as_ref() { anyhow::bail!("spec: engine fatal at admit: {f}"); }
    if arm == SpecArm::Plain {
        for l in sched.lanes.iter_mut().flatten() { l.mtp = None; }
    }
    let mut steps = 0usize;
    while sched.lanes.iter().any(|l| l.is_some()) {
        sched.step()?;
        if let Some(f) = sched.fatal.as_ref() { anyhow::bail!("spec: engine fatal: {f}"); }
        steps += 1;
        anyhow::ensure!(steps <= 4 * gen + 64, "spec: lane still live after {steps} steps");
    }
    let secs = t0.elapsed().as_secs_f64();
    let (mut toks, mut reason) = (Vec::with_capacity(gen), String::new());
    while let Some(ev) = crate::server::try_recv_counted(&mut rx, &bl) { // LR-5 (H6)
        match ev {
            TokEvent::Tok(x) => toks.push(x),
            TokEvent::Finish { reason: r } => reason = r,
            // #8.2: admission telemetry — no action for a spec gate drain.
            TokEvent::Admitted { .. } => {}
            TokEvent::StopInfo { .. } => {}
            TokEvent::Logprob { .. } => {} // not requested by this drain
        }
    }
    anyhow::ensure!(!reason.starts_with("error"), "spec: {} request failed: {reason}", arm.name());
    Ok(SpecArmOut { toks, st: sched.gate.take().unwrap_or_default(), secs, reason })
}

fn spec_pack(prompts: &[(String, Vec<u32>)]) -> Vec<u32> {
    let mut v = vec![prompts.len() as u32];
    for (n, ids) in prompts {
        let mut b = n.as_bytes().to_vec();
        b.resize(16, 0);
        v.extend(b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])));
        v.push(ids.len() as u32);
        v.extend_from_slice(ids);
    }
    v
}

fn spec_unpack(v: &[u32]) -> Result<Vec<(String, Vec<u32>)>> {
    let n = *v.first().context("spec: empty prompt pack")? as usize;
    let mut i = 1;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        anyhow::ensure!(i + 5 <= v.len(), "spec: truncated prompt pack");
        let b: Vec<u8> = v[i..i + 4].iter().flat_map(|w| w.to_le_bytes()).take_while(|&c| c != 0).collect();
        let len = v[i + 4] as usize;
        i += 5;
        anyhow::ensure!(i + len <= v.len(), "spec: truncated prompt pack");
        out.push((String::from_utf8_lossy(&b).to_string(), v[i..i + len].to_vec()));
        i += len;
    }
    Ok(out)
}

/// TP-4D: fold the world > 2 end-of-program hub result: `all[r] = [rank r's verdict (1 = OK), rank r's agree count]`.
/// Ok only when EVERY rank reports OK and every rank's agree count equals this rank's (the same lockstep points were
/// crossed everywhere). The line names each rank, so one failing rank of four is visible in every rank's log.
pub(super) fn tpspec_fold(all: &[Vec<u32>], my_rank: usize, agrees: u64) -> (bool, String) {
    let mut ok = true;
    let mut parts = Vec::new();
    for (r, f) in all.iter().enumerate() {
        let good = f[0] != 0 && f[1] as u64 == agrees;
        ok &= good;
        parts.push(format!("rank {r}{} {} (agrees {})", if r == my_rank { " (this)" } else { "" },
                           if f[0] != 0 { "OK" } else { "FAIL" }, f[1]));
    }
    (ok, format!("{} | this rank agrees {agrees}", parts.join(" | ")))
}

/// The spec program (see the section comment). `ctx` = Some on both TP ranks (the head passes
/// `prompts`, the node None — they arrive over the link); None = TP=1 (prompts required).
pub fn run_spec(model_dir: &str, ctx: Option<crate::tp::TpContext>, prompts: Option<Vec<(String, Vec<u32>)>>,
                max_pos: usize, o: &SpecOpts) -> Result<()> {
    use sha2::{Digest, Sha256};
    let (prompts, attach, rank, world) = match ctx {
        Some(mut ctx) => {
            ctx.sanity()?;
            ctx.branch_check(&crate::tp::TpBranch::Exl3Xtp)?;
            let packed = prompts.as_ref().map(|p| spec_pack(p));
            let (seq, _, _) = match &packed {
                Some(v) => ctx.broadcast_prompt(Some((v, 0, 0)))?,
                None => ctx.broadcast_prompt(None)?,
            };
            let prompts = spec_unpack(&seq)?;
            let (rank, world, link) = ctx.into_parts();
            anyhow::ensure!(crate::net::pin_thread(9), "launch thread failed to pin to core 9 — TP refuses to run unpinned");
            (prompts, Some(crate::exl3_forward::xtp::TpAttach { rank, world, link }), rank, world)
        }
        None => (prompts.context("spec (TP=1): no prompts")?, None, 0, 1),
    };
    let tp_on = attach.is_some();
    // "TP=2 rank N" at world 2 (the label the gate scripts grep), "TP=4 rank N" at world 4
    let who = if tp_on { format!("TP={world} rank {rank}") } else { "TP=1".to_string() };
    if o.prefix { crate::opts::set(crate::opt!("exl3-prefix"), "1"); }
    crate::dprintln!("[tpspec] {who}: program {} — {} prompt(s) {:?}; loading {model_dir} (width {}, max_pos {max_pos}, kv-cache {})",
             o.to_mode(), prompts.len(), prompts.iter().map(|(n, p)| format!("{n}:{}", p.len())).collect::<Vec<_>>(),
             o.width, crate::exl3_forward::kv_fmt_name(crate::exl3_forward::kv_fmt_from_opts()));
    let t_load = std::time::Instant::now();
    let model = FwdModel::load_tp(model_dir, o.width, max_pos, attach)?;
    anyhow::ensure!(model.mtp.is_some(), "spec: the pack has no MTP draft head");
    crate::dprintln!("[tpspec] {who}: loaded in {:.1}s (trunk per rank: {} q / {} kv heads, GDN {} k / {} v heads)",
             t_load.elapsed().as_secs_f32(), model.tc.num_heads, model.tc.num_kv_heads,
             model.tc.lin_num_k_heads, model.tc.lin_num_v_heads);
    let tok = QwenTokenizer::from_file(&format!("{}/tokenizer.json", model_dir.trim_end_matches('/')))?;
    let cfg_eos: u32 = {
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/config.json", model_dir.trim_end_matches('/')))?)?;
        cfg.get("eos_token_id").and_then(|e| e.as_u64()).unwrap_or(248046) as u32
    };
    let eos = tok.stop_token_ids(cfg_eos);
    let mtp_k = crate::exl3_forward::mtp_depth_default();
    let width = o.width;
    let mut sc = FwdModel::scratch(model.dev(), &model.cfg, width.max(mtp_k + 1), model.max_pos())?;
    let chunk = 2048usize;
    let psc0 = model.prefill_scratch(chunk)?;
    model.precapture_decode_graphs(&mut sc, width, mtp_k, true)?;
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
        lanes: (0..width).map(|_| None).collect(), eos: eos.clone(), chunk, interleave: 0, psc: Some(psc0),
        df2: None, df2_k: crate::dflash2::block() - 1, // this path loads no DF2 artifact
        copy_arm: None, // WIDEVERIFY1: the TP-C harness never copies
        record: None, // COPYREC1: nor keeps a record
        wide: Default::default(), cpv_price: Default::default(), cpv_on: false, cpv: None, // COPYPF1: nor runs a prefill-width copy verify
        cl_on: false, cl_priced: true, cl_price: Default::default(), // COPYLANES: nor copy work in busy steps
        mtp_k, spec_policy: SpecLanes::Never, lane_order: LaneOrder::Rr, lane_quantum: 256, admit_ctr: 0, fcfs_rots: 0,
        lane_pack: crate::lane_pack::LanePack::Off, scp: None, pack_ready: [false, false], pack_sampled: false, pack_rot: 0, pack_logs: 0, in_prefill: false, pack_stats: PackStats::default(), tr: SchedTrace::default(),
        reprobe_on, reprobe_cfg, shared_scale: 1.0, shared_last: false, shared_streak: 0, shared_probe_ctr: 0, spec_win_tok: [0.0; 2], spec_win_ms: [0.0; 2], spec_win_n: [0; 2], spec_win_acc: [0; 2], spec_arm_rate: [0.0; 2], spec_arm_n: [0; 2], spec_arm_fold: [0; 2], spec_rate_n: vec![[(0.0, 0); 2]; SPEC_RATE_NMAX + 1], spec_rate_folds: vec![[0; 2]; SPEC_RATE_NMAX + 1], meter_settle: 0, meter_last_n: 0, probe_fold: [false; 2], spec_probe: None, spec_probe_ctr: 0, spec_est_pick: None, spec_margin: 0.0, spec_margin_n: 0, policy_logs: 0, ctrl_round: 0, ema_plain: 0.0, draft_conf: 0.4, cal: DraftCal::new(),
        reserve: None, reserve_st: crate::lane_reserve::State::new(), rel: (0..width).map(|_| None).collect(), lru: vec![0; width],
        prefix_on: o.prefix, cache: (0..width).map(|_| None).collect(),
        wp16: None, wp16_xcheck: false,
        loop_cfg: Some((300, 3)), pen_range: 1024, spec_ratio: true, draft_temp: None, gate: None, tok: tok.clone(),
        exit_on_fatal: false, fatal: None, inject_panic: None, steps: 0, last_resume: (0, 0), pf_done: Default::default(), pf_jobs: Vec::new(), vmm_wait: None, mrope_section: None, vis_seam: None, kv_pf: None, turn: None,
        tp: if tp_on { Some(TpSync::new(rank, world)) } else { None },
    };
    let mut arms = vec![SpecArm::Plain];
    if o.dds { arms.push(SpecArm::Dds); }
    arms.extend(o.depths.iter().map(|&d| SpecArm::Fixed(d)));
    crate::dprintln!("[tpspec] {who}: arms {:?} x {} reps, gen {} greedy (thinking-off chat ids), served depth {mtp_k}, DDS 0.4, \
              prefix cache {}, loop detect (300, 3), precapture done ({} graphs)",
             arms.iter().map(|a| a.name()).collect::<Vec<_>>(), o.reps, o.gen, if o.prefix { "on" } else { "off" },
             model.graph_count());
    let fnv64 = |t: &[u32]| -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &x in t { h ^= x as u64; h = h.wrapping_mul(0x0000_0100_0000_01b3); }
        h
    };
    let text_sha = |t: &[u32]| -> String {
        let body: Vec<u32> = t.iter().copied().filter(|x| !eos.contains(x)).collect();
        let s = tok.decode(&body, true).unwrap_or_default();
        Sha256::digest(s.as_bytes()).iter().take(8).map(|b| format!("{b:02x}")).collect()
    };
    let t_all = std::time::Instant::now();
    let mut e_ok = true;
    let (mut n_cmp, mut n_bad) = (0usize, 0usize);
    let mut plain_ref: std::collections::BTreeMap<String, Vec<u32>> = Default::default();
    // per (class, arm): (tok/round, ms/round, tok/s) per rep; plain: ms/token
    let mut acc: std::collections::BTreeMap<(String, String), Vec<(f64, f64, f64)>> = Default::default();
    for rep in 0..o.reps {
        if let Some(t) = sched.tp.as_mut() { t.ident = o.ident && rep == 0; }
        for (name, ids) in &prompts {
            let mut rf: Option<Vec<u32>> = None;
            for &arm in &arms {
                let r = spec_arm(&mut sched, arm, ids, o.gen)?;
                let st = &r.st;
                let same = match arm {
                    SpecArm::Plain => {
                        let prev = plain_ref.entry(name.clone()).or_insert_with(|| r.toks.clone());
                        rf = Some(r.toks.clone());
                        *prev == r.toks
                    }
                    _ => rf.as_ref().map_or(false, |p| *p == r.toks),
                };
                n_cmp += 1;
                if !same { n_bad += 1; e_ok = false; }
                let div = rf.as_ref().and_then(|p| p.iter().zip(&r.toks).position(|(a, b)| a != b));
                let n = r.toks.len();
                let (tpr, mpr) = if st.rounds > 0 {
                    ((st.accepted + st.rounds) as f64 / st.rounds as f64, st.ms / st.rounds as f64)
                } else { (0.0, 0.0) };
                let tps = if r.secs > 0.0 { n as f64 / r.secs } else { 0.0 };
                let key = (name.clone(), arm.name());
                acc.entry(key).or_default().push(if arm == SpecArm::Plain { (0.0, r.secs * 1e3 / n.max(1) as f64, tps) } else { (tpr, mpr, tps) });
                if arm == SpecArm::Plain && rep == 0 && rank == 0 {
                    let body: Vec<u32> = r.toks.iter().copied().filter(|x| !eos.contains(x)).collect();
                    let t = tok.decode(&body, true).unwrap_or_default();
                    let (hd, tl): (String, String) = (t.chars().take(160).collect(), t.chars().rev().take(120).collect::<Vec<_>>().into_iter().rev().collect());
                    crate::dprintln!("TPSPEC-TEXT {who} {name}: head {hd:?} ... tail {tl:?}");
                }
                crate::dprintln!("TPSPEC {who} rep {rep} {name} {:<5} n={n} ({}) ids {:016x} text {} | {} | rounds {} drafted {} accepted {} \
                          tok/round {tpr:.3} ms/round {mpr:.2} ctrl {} plain {} | {tps:.1} tok/s ({:.1} s)",
                         arm.name(), r.reason, fnv64(&r.toks), text_sha(&r.toks),
                         if arm == SpecArm::Plain { if same { "plain (rep-stable)".to_string() } else { "PLAIN DIFFERS FROM REP 0".to_string() } }
                         else if same { "== plain".to_string() } else { format!("MISMATCH vs plain at {:?}", div) },
                         st.rounds, st.drafted, st.accepted, st.ctrl, st.plain, r.secs);
            }
        }
        if let Some(t) = sched.tp.as_ref() {
            crate::dprintln!("[tpspec] {who} rep {rep}: lockstep agrees {} ok (agree_ext tripwire armed; proxy watchdog 10 s){}",
                     t.agrees, if t.ident_bufs > 0 { format!("; G-T1-b graphed digests {} compared, {} differ", t.ident_bufs, t.ident_bad) } else { String::new() });
        }
    }
    crate::dprintln!("TPSPEC {who} SUMMARY (mean over {} reps; DDS/fixed: tok/round | ms/round | tok/s; plain: ms/token | tok/s):", o.reps);
    for ((name, arm), v) in &acc {
        let k = v.len() as f64;
        let m = |f: fn(&(f64, f64, f64)) -> f64| v.iter().map(f).sum::<f64>() / k;
        let mn = |f: fn(&(f64, f64, f64)) -> f64| v.iter().map(f).fold(f64::INFINITY, f64::min);
        let mx = |f: fn(&(f64, f64, f64)) -> f64| v.iter().map(f).fold(f64::NEG_INFINITY, f64::max);
        if arm == "plain" {
            crate::dprintln!("  {name:<7} {arm:<5} ms/token {:.2} (min {:.2} max {:.2}) | {:.1} tok/s", m(|x| x.1), mn(|x| x.1), mx(|x| x.1), m(|x| x.2));
        } else {
            crate::dprintln!("  {name:<7} {arm:<5} tok/round {:.3} (min {:.3} max {:.3}) | ms/round {:.2} (min {:.2} max {:.2}) | {:.1} tok/s",
                     m(|x| x.0), mn(|x| x.0), mx(|x| x.0), m(|x| x.1), mn(|x| x.1), mx(|x| x.1), m(|x| x.2));
        }
    }
    let (ident_bufs, ident_bad, agrees) = sched.tp.as_ref().map_or((0, 0, 0), |t| (t.ident_bufs, t.ident_bad, t.agrees));
    let (sp_ok, sp_bad) = (crate::exl3_forward::xtp::SPINE_OK.load(std::sync::atomic::Ordering::Relaxed),
                           crate::exl3_forward::xtp::SPINE_BAD.load(std::sync::atomic::Ordering::Relaxed));
    if tp_on {
        crate::dprintln!("[tpspec] {who}: graphs captured {} ; TP-SPINE audit {sp_ok} graphs OK, {sp_bad} discarded (a discarded shape runs eager)",
                 model.graph_count());
    }
    crate::dprintln!("G-T1-e {who}: {} — {n_cmp} requests compared ({n_bad} differ) over {} prompt(s) x {} arm(s) x {} rep(s), gen {} ({:.1} min)",
             if e_ok { "LOSSLESS_OK (every spec arm == plain, byte-identical; plain rep-stable)" } else { "MISMATCH" },
             prompts.len(), arms.len(), o.reps, o.gen, t_all.elapsed().as_secs_f64() / 60.0);
    let mut samp_ok = true;
    if o.samp > 0 {
        let text = WP24_GATE_PROMPT;
        let p = tok.encode(text, false)?;
        sched.mtp_k = mtp_k;
        let ok = wp24_gate_conditions(&mut sched, &p, text, o.trials.max(50), 6, 2, 20, 0.4, None, o.samp == 1)?;
        samp_ok = ok;
        println!("G-T1-h {who}: {} (WP24 served gate on this scheduler, {}; trials {}/arm, width {width})",
                 if ok { "SERVED_DISTRIBUTION_OK" } else { "FAIL / NO_POWER" }, if o.samp == 1 { "--quick" } else { "full ladder" }, o.trials);
    }
    // (re-read after the sampled gate: its rounds are lockstep points too)
    let (ident_bufs, ident_bad, agrees) = sched.tp.as_ref().map_or((0, 0, 0), |t| (t.ident_bufs, t.ident_bad, t.agrees));
    let mut all_ok = e_ok && samp_ok && ident_bad == 0;
    if let Some(t) = sched.tp.as_ref() {
        if t.world > 2 {
            // TP-4D: world > 2 — every rank's verdict + agree count through the head hub (unbounded: the ranks
            // finish the gate's last rep at different times); every rank prints the same per-rank table
            let all = crate::net::exchange_u32s_all(&[all_ok as u32, t.agrees as u32], 4)?;
            let (ok, line) = tpspec_fold(&all, t.rank as usize, agrees);
            crate::dprintln!("[tpspec] {who}: {line} (graphed digests {ident_bufs} compared, {ident_bad} differ)");
            all_ok &= ok;
        } else {
            let peer = crate::net::exchange_u32s(&[all_ok as u32, t.agrees as u32], 4)?;
            println!("[tpspec] {who}: this rank {} | peer {} (peer agrees {}, this rank {agrees}; graphed digests {ident_bufs} compared, {ident_bad} differ)",
                     if all_ok { "OK" } else { "FAIL" }, if peer[0] != 0 { "OK" } else { "FAIL" }, peer[1]);
            all_ok &= peer[0] != 0 && peer[1] as u64 == agrees;
        }
    }
    println!("RESULT: {} ({who}; G-T1-e {}{}{})", if all_ok { "TPSPEC_OK" } else { "TPSPEC_FAIL" },
             if e_ok { "OK" } else { "MISMATCH" },
             if o.samp > 0 { if samp_ok { " ; G-T1-h OK" } else { " ; G-T1-h FAIL" } } else { "" },
             if tp_on { if ident_bad == 0 { " ; RANK_IDENTITY_OK (graphed)" } else { " ; RANK_IDENTITY_FAIL" } } else { "" });
    anyhow::ensure!(all_ok, "spec program gates failed ({who})");
    Ok(())
}
