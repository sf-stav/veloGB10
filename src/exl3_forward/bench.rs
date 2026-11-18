//! Offline (probe / bench) entry points: `FwdArgs`, `run`, `run_mtp_bench`, `probe_state_exl3`, `bench_mtp_exl3`
//! and their snapshot / diff helpers. Not on the served path. See `mod.rs` for the file map.

use super::*;

// ---------------------------------------------------------------------------
// Drivers: generate / ladder / bench
// ---------------------------------------------------------------------------

pub struct FwdArgs {
    pub dir: String,
    pub prompt: Option<String>,
    pub ids_file: Option<String>,
    pub max_new: usize,
    pub width: usize,
    pub ladder_dir: Option<String>,
    pub ladder_layers: Vec<usize>,
    pub ladder_step: Option<usize>,
    pub max_pos: usize,
    pub bench_steps: usize,
    pub widths: Vec<usize>,
    pub prefill_chunk: Option<usize>,
}

pub(super) const EOS_IDS: [i32; 2] = [248044, 248046];

pub fn run(a: &FwdArgs) -> Result<()> {
    let model = FwdModel::load(&a.dir, a.width.max(a.widths.iter().copied().max().unwrap_or(1)), a.max_pos)?;
    let mut sc = Scratch::new(&model.dev, &model.cfg, a.width.max(a.widths.iter().copied().max().unwrap_or(1)), model.cfg.rotary_dim, model.max_pos)?;

    // prompt ids: explicit file wins; else tokenizer encode
    let mut prompt_ids: Vec<u32> = match &a.ids_file {
        Some(f) => {
            let s = std::fs::read_to_string(f)?;
            let v: Vec<u32> = serde_json::from_str(&s).or_else(|_| {
                s.split_whitespace().map(|x| x.parse::<u32>()).collect::<std::result::Result<_, _>>()
            })?;
            v
        }
        None => {
            let text = a.prompt.clone().context("--prompt or --prompt-ids required")?;
            let tok = crate::tokenizer::QwenTokenizer::from_file(&format!("{}/tokenizer.json", a.dir))?;
            tok.encode(&text, false)?
        }
    };
    crate::dprintln!("prompt: {} tokens (first 8: {:?})", prompt_ids.len(), &prompt_ids[..prompt_ids.len().min(8)]);

    let mut ladder = a.ladder_dir.as_ref().map(|d| {
        std::fs::create_dir_all(d).ok();
        LadderCtx { dir: d.clone(), step: 0, layers: a.ladder_layers.clone(), only_step: a.ladder_step, mem: None, digest: false, sp: None }
    });

    // prefill: chunked sweeps (S-A3-f-b, --prefill-chunk N) or sequential decode-step
    let mut t0 = std::time::Instant::now();
    let w = a.width.max(1);
    let mut pos = 0usize;
    let mut last = prompt_ids[0] as i32;
    if let Some(cw) = a.prefill_chunk.filter(|&c| c > 0) {
        // chunked: process prompt[..len-1] in C-token sweeps (the final token
        // enters via the first decode step, same as the sequential path).
        let n = prompt_ids.len().saturating_sub(1);
        let mut psc = model.prefill_scratch(cw)?;
        // S-A3-u: --prefill-warmup=1 = the serve's startup warm-up before the timed prefill;
        // --prefill-reps=k = k timed prefills in one process (rep 1 = cold-process, rep 2+ = warm).
        if crate::opts::var(crate::opt!("prefill-warmup")).is_ok() {
            model.prefill_warmup(&mut psc, 0, &FwdModel::prefill_warmup_widths(cw))?;
            // PFX1 (e): the bench measures the serve's post-boot posture — wait for the plans
            if let Some(h) = model.lt_prewarm_start() { let _ = h.join(); }
        }
        let reps: usize = crate::opts::var(crate::opt!("prefill-reps")).ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
        for rep in 1..reps {
            let tr = std::time::Instant::now();
            let mut p2 = 0usize;
            while p2 < n {
                let end = (p2 + cw).min(n);
                let toks: Vec<i32> = prompt_ids[p2..end].iter().map(|&t| t as i32).collect();
                model.prefill_chunk(&mut psc, &toks, p2, 0, false, None)?;
                p2 = end;
            }
            let dt = tr.elapsed().as_secs_f32();
            crate::dprintln!("prefill rep {rep}: {} tokens in {:.2}s ({:.0} tok/s)", prompt_ids.len(), dt, prompt_ids.len() as f32 / dt);
            model.reset_slot(0)?;
        }
        if reps > 1 || crate::opts::var(crate::opt!("prefill-warmup")).is_ok() {
            t0 = std::time::Instant::now(); // the reported prefill excludes warm-up / earlier reps
        }
        while pos < n {
            let end = (pos + cw).min(n);
            let toks: Vec<i32> = prompt_ids[pos..end].iter().map(|&t| t as i32).collect();
            model.prefill_chunk(&mut psc, &toks, pos, 0, false, None)?;
            pos = end;
        }
    } else {
        for (i, t) in prompt_ids.iter().enumerate() {
            last = *t as i32;
            if i + 1 < prompt_ids.len() {
                let toks_w = vec![last; w];
                let pos_w: Vec<_> = (0..w).map(|_| pos).collect();
                let slots_w: Vec<usize> = (0..w).collect();
                model.forward_step(&mut sc, &toks_w, &pos_w, &slots_w, ladder.as_mut())?;
                pos += 1;
            }
        }
    }
    last = *prompt_ids.last().context("empty prompt")? as i32;
    let prefill_s = t0.elapsed().as_secs_f32();
    if a.prefill_chunk.is_some() {
        crate::dprintln!("prefill: {} tokens in {:.1}s ({:.0} tok/s, chunked wide-M prefill, C={})",
                 prompt_ids.len(), prefill_s, prompt_ids.len() as f32 / prefill_s.max(1e-9),
                 a.prefill_chunk.unwrap_or(0));
    } else {
        crate::dprintln!("prefill: {} tokens in {:.1}s ({:.0} tok/s, sequential decode-step prefill)",
                 prompt_ids.len(), prefill_s, prompt_ids.len() as f32 / prefill_s.max(1e-9));
    }

    // greedy decode
    let mut out_ids: Vec<u32> = Vec::with_capacity(a.max_new);
    let mut round_ms: Vec<f64> = Vec::new();
    for step in 0..a.max_new {
        let ts = std::time::Instant::now();
        let toks_w = vec![last; w];
        let pos_w: Vec<_> = (0..w).map(|_| pos).collect();
        let slots_w: Vec<usize> = (0..w).collect();
        let ids = model.forward_step(&mut sc, &toks_w, &pos_w, &slots_w, ladder.as_mut())?;
        for lane in 1..w {
            anyhow::ensure!(ids[lane] == ids[0], "lane {lane} diverged at step {step}: assembled batch-invariance broken");
        }
        round_ms.push(ts.elapsed().as_secs_f64() * 1000.0);
        // Gated per-step top-k dump (S-A3-d Phase 1b margin evidence; diagnostics-only env).
        if let Ok(tk_dir) = crate::opts::var(crate::opt!("step-topk-dir")) {
            let lg = model.dev.dtoh_sync_copy(&sc.logits)?;
            let kv: Vec<f32> = lg.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
            let mut idx: Vec<u32> = (0..lg.len() as u32).collect();
            idx.sort_by(|&a, &b| kv[b as usize].partial_cmp(&kv[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal));
            let top: Vec<String> = idx.iter().take(8)
                .map(|&i| format!("[{},{}]", i, kv[i as usize])).collect();
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
                .open(format!("{tk_dir}/topk.jsonl")) {
                let _ = writeln!(f, "{{\"step\":{},\"argmax\":{},\"top\":[{}]}}",
                    step, ids[0], top.join(","));
            }
        }
        let id = ids[0];
        pos += 1;
        if EOS_IDS.contains(&id) {
            crate::dprintln!("EOS at step {step} (id {id})");
            break;
        }
        out_ids.push(id as u32);
        last = id;
    }

    // report
    let tok = crate::tokenizer::QwenTokenizer::from_file(&format!("{}/tokenizer.json", a.dir))?;
    let text = tok.decode(&out_ids, true)?;
    round_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = round_ms.get(round_ms.len() / 2).copied().unwrap_or(0.0);
    crate::dprintln!("generated {} tokens; round ms: median {:.1} p10 {:.1} p90 {:.1}",
             out_ids.len(), med,
             round_ms.get(round_ms.len() / 10).copied().unwrap_or(0.0),
             round_ms.get(round_ms.len() * 9 / 10).copied().unwrap_or(0.0));
    crate::dprintln!("effective GB/s (byte model / median round): {:.1}",
             model.bytes_per_round as f64 / 1e9 / (med / 1000.0));
    crate::dprintln!("---- TEXT ----");
    crate::dprintln!("{text}");
    crate::dprintln!("---- IDS ----");
    crate::dprintln!("{out_ids:?}");

    // width sweep (bench mode): flatness through the assembled model
    if a.widths.len() > 1 || a.bench_steps > 0 {
        let widths = if a.widths.is_empty() { vec![a.width] } else { a.widths.clone() };
        for w in widths {
            // states reset per width via a fresh load (the honest offline reset)
            let m2 = FwdModel::load(&a.dir, w, a.max_pos)?;
            let mut sc2 = Scratch::new(&m2.dev, &m2.cfg, w, m2.cfg.rotary_dim, m2.max_pos)?;
            let mut pos2 = 0usize;
            let mut last2 = prompt_ids[0] as i32;
            for (i, t) in prompt_ids.iter().enumerate() {
                last2 = *t as i32;
                if i + 1 < prompt_ids.len() {
                    m2.forward_step(&mut sc2, &[last2], &[pos2], &[0], None)?;
                    pos2 += 1;
                }
            }
            let lanes: Vec<i32> = vec![last2; w];
            let mut poss: Vec<usize> = (0..w).map(|_| pos2).collect();
            let mut ms = Vec::new();
            for _ in 0..a.bench_steps.max(1) {
                let ts = std::time::Instant::now();
                let slots_w: Vec<usize> = (0..w).collect();
                m2.forward_step(&mut sc2, &lanes, &poss, &slots_w, None)?;
                ms.push(ts.elapsed().as_secs_f64() * 1000.0);
                for p in poss.iter_mut() {
                    *p += 1;
                }
            }
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = ms.get(ms.len() / 2).copied().unwrap_or(0.0);
            crate::dprintln!("W={w}: median round {:.1} ms over {} steps (byte model {:.2} GB/round → {:.1} GB/s)",
                     med, ms.len(), m2.bytes_per_round as f64 / 1e9,
                     m2.bytes_per_round as f64 / 1e9 / (med / 1000.0));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// S-A3-e — M1 draft-quality harness (the G-A3-4 instrument)
//
// Plain greedy decode is the GROUND TRUTH; the M1 pack head drafts `depth` tokens per step
// from the live tap; every draft is scored against the trunk's own revealed token (exact,
// no approximations). The decode itself is NOT speculated: the exl3 stack has no parallel
// chain-verify (GDN recurrence + per-column KV checkpoints are NVFP4-path machinery), and
// sequential verification degenerates to plain decode. What this measures honestly is the
// DECISION-RELEVANT quantity — M1 draft acceptance and head cost — plus the projected
// economics IF the parallel-verify kernel is ported (the enablement item).
// ---------------------------------------------------------------------------

/// Round record for exact post-hoc scoring.
struct DraftRound {
    drafts: Vec<i32>,
    accepted: usize, // device-computed a for this round (out.a)
}

pub fn run_mtp_bench(a: &FwdArgs, depth: usize) -> Result<()> {
    let depth = depth.max(1);
    let model = FwdModel::load(&a.dir, 1, a.max_pos)?;
    let head = match model.mtp.as_ref() {
        Some(h) => h,
        None => bail!("pack has no mtp.* draft head — M1 unavailable"),
    };
    // S-A3-o: the verify round needs width depth+1 (was 1 -> "scratch width 1 < verify width").
    let mut sc = Scratch::new(&model.dev, &model.cfg, (depth + 1).max(2), model.cfg.rotary_dim, model.max_pos)?;

    let prompt_ids: Vec<u32> = match &a.ids_file {
        Some(f) => {
            let s = std::fs::read_to_string(f)?;
            serde_json::from_str(&s).or_else(|_| {
                s.split_whitespace().map(|x| x.parse::<u32>()).collect::<std::result::Result<_, _>>()
            })?
        }
        None => {
            let text = a.prompt.clone().context("--prompt or --prompt-ids required")?;
            let tok = crate::tokenizer::QwenTokenizer::from_file(&format!("{}/tokenizer.json", a.dir))?;
            tok.encode(&text, false)?
        }
    };
    let plen = prompt_ids.len();
    crate::dprintln!("MTP-BENCH depth={depth} prompt={plen} tokens max_new={} model={}", a.max_new, a.dir);

    // ---- prefill (trunk only; the SERVE primes the head ONCE — the chain-verify
    // round mirrors that; the S-A3-e per-position priming is the OLD arm's) ----
    let mut pos = 0usize;
    let mut last = prompt_ids[0] as i32;
    let t0 = std::time::Instant::now();
    let mut psc_keep: Option<PrefillScratch> = None; // WP03: the seam prime reads its head-fill carry
    if let Some(cw) = a.prefill_chunk.filter(|&c| c > 0) {
        // S-A3-r: the SERVED posture — chunked sweeps over prompt[..len-1] (exl3_serve / run()),
        // then the last prompt token through forward_step (its argmax + tap feed round 1).
        let n = plen.saturating_sub(1);
        let psc = psc_keep.insert(model.prefill_scratch(cw)?);
        while pos < n {
            let end = (pos + cw).min(n);
            let toks: Vec<i32> = prompt_ids[pos..end].iter().map(|&t| t as i32).collect();
            model.prefill_chunk(psc, &toks, pos, 0, false, None)?;
            pos = end;
        }
        last = *prompt_ids.last().unwrap() as i32;
        model.forward_step(&mut sc, &[last], &[pos], &[0], None)?;
        pos += 1;
        crate::dprintln!("prefill: {:.1} s (chunked C={cw}, served posture)", t0.elapsed().as_secs_f32());
    } else {
    for (i, t) in prompt_ids.iter().enumerate() {
        last = *t as i32;
        model.forward_step(&mut sc, &[last], &[pos], &[0], None)?;
        pos += 1;
    }
    crate::dprintln!("prefill: {:.1} s", t0.elapsed().as_secs_f32());
    }

    // The trunk's argmax after the last prompt token IS t_plen — the first generated token
    // comes free (no duplicated-token re-consume).
    let mut actual: Vec<u32> = Vec::with_capacity(a.max_new);
    let pred0 = model.dev.dtoh_sync_copy(&sc.argmax)?[0];
    actual.push(pred0 as u32);

    // ---- S-A3-f-d CHAIN-VERIFY arm (the wired serve path): one head prime from
    // the tap that predicted the first bonus, then mtp_round per step.
    model.mtp_seam_prime(&mut sc, psc_keep.as_ref(), head, last, plen, 0)?; // WP03 (the serve's prime)
    let mut rounds: Vec<DraftRound> = Vec::new();
    let mut round_ms: Vec<f64> = Vec::new();
    let mut drafted_n = 0usize; // drafts actually run (a round clamped at the sparse switch drafts fewer than `depth`)
    {
        let mut b = pred0 as i32;
        let mut p = plen;
        let mut tap_row = 0usize;
        // S-A3-t: a first token that is EOS ends the request (the serve stops there; the plain arm
        // does too) — the MTP arm used to keep drafting past it.
        let first_eos = EOS_IDS.contains(&(pred0 as i32));
        while !first_eos && actual.len() < a.max_new {
            let tr = std::time::Instant::now();
            // clamped at the sparse switch like the serve (S-B9-REL-SMALLFIXES item 2)
            let out = model.mtp_round_clamped(&mut sc, head, 0, depth, b, p, tap_row)?;
            round_ms.push(tr.elapsed().as_secs_f64() * 1e3);
            drafted_n += out.drafts.len();
            for &e in out.emitted.iter() {
                actual.push(e as u32);
            }
            rounds.push(DraftRound { drafts: out.emitted[..out.a].to_vec(), accepted: out.a });
            b = out.last_tok;
            p = out.pos_next;
            tap_row = out.tap_row;
            if EOS_IDS.contains(&b) {
                break;
            }
        }
    }

    let med = |v: &mut Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.get(v.len() / 2).copied().unwrap_or(0.0)
    };
    // S-A3-f-d: acceptance is now DEVICE-COMPUTED per round (the accept kernel's
    // `a`), not post-hoc. drafted = the drafts each round ran (depth, fewer for a round clamped at
    // the sparse switch); accepted = sum of round a's.
    let drafted = drafted_n;
    let accepted: usize = rounds.iter().map(|r| r.drafts.len()).sum();
    let mean_prefix = if rounds.is_empty() { 0.0 } else {
        accepted as f64 / rounds.len() as f64
    };
    let rms = med(&mut round_ms);
    let acc_pct = if drafted > 0 { 100.0 * accepted as f64 / drafted as f64 } else { 0.0 };
    let e_per_round = if rounds.is_empty() { 0.0 } else { actual.len() as f64 / rounds.len() as f64 };
    let mtp_decode_s = round_ms.iter().sum::<f64>() / 1e3;
    crate::dprintln!("MTP-BENCH RESULT depth={depth} (chain-verify, the wired serve path)");
    crate::dprintln!("  generated {} tokens vs plain greedy ground truth", actual.len());
    crate::dprintln!("  acceptance {accepted}/{drafted} = {acc_pct:.2}%   mean accepted prefix {mean_prefix:.3}   emitted/round {e_per_round:.2}");
    crate::dprintln!("  round (median) {rms:.1} ms   decode-only tok/s {:.1}",
             if mtp_decode_s > 0.0 { (actual.len() - 1) as f64 / mtp_decode_s } else { 0.0 });
    let plain = if rms > 0.0 { 1e3 * e_per_round / rms } else { 0.0 };
    crate::dprintln!("  effective served tok/s (emitted/round / round) {plain:.1}");
    Ok(())
}

// ===========================================================================
// S-A3-f-e: EXL3 packs in the GENERIC gate chain. `--bench-mtp` and
// `--probe-state` dispatch here (main.rs) when --model-dir is an EXL3 pack:
// the generic GpuModel loader cannot read this container (it panicked with a
// bytemuck alignment fault at "Shard 1/9" — the pack's index legitimately
// references NINE safetensors files: 7 model shards + the M1 mixer patch +
// the PLE ngram table — and the generic loader's bf16/f32 casts hit a
// misaligned packed tensor). Both gates run the EXL3 engine (FwdModel).
// ===========================================================================

pub(super) fn gate_prompt_ids(a: &FwdArgs) -> Result<Vec<u32>> {
    match &a.ids_file {
        Some(f) => {
            let s = std::fs::read_to_string(f)?;
            serde_json::from_str(&s).or_else(|_| {
                s.split_whitespace().map(|x| x.parse::<u32>()).collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(anyhow::Error::from)
            })
        }
        None => {
            // Bare gate invocations (`--bench-mtp --model-dir <m> --depth 2`) must run
            // without flags: default to the S-A3-f corpus sentence (the f round-bench
            // prompt, 16 tokens).
            let text = a.prompt.clone().unwrap_or_else(|| {
                "The theory of general relativity states that gravity is the curvature of spacetime itself".to_string()
            });
            let tok = crate::tokenizer::QwenTokenizer::from_file(&format!("{}/tokenizer.json", a.dir))?;
            tok.encode(&text, false)
        }
    }
}

/// Slot-0 state windows (conv + recurrent) of every GDN layer, as raw bits.
/// Layout is [width slots × per_layer]; slot 0 occupies [0, n).
pub(super) fn snap_slot0_state(model: &FwdModel) -> Result<(Vec<Vec<u32>>, Vec<Vec<u32>>)> {
    let cfg = &model.tc; // TP-B: this rank's trunk state
    let per_conv = (cfg.key_dim() * 2 + cfg.value_dim()) * cfg.conv_kernel;
    let per_s = cfg.lin_num_v_heads * cfg.lin_k_dim * cfg.lin_v_dim;
    let mut conv = Vec::with_capacity(model.conv_state.len());
    for cs in &model.conv_state {
        let h: Vec<f32> = model.dev.dtoh_sync_copy(cs)?;
        conv.push(h[..per_conv].iter().map(|f| f.to_bits()).collect());
    }
    let mut s = Vec::with_capacity(model.s_state.len());
    for ss in &model.s_state {
        let h: Vec<f32> = model.dev.dtoh_sync_copy(ss)?;
        s.push(h[..per_s].iter().map(|f| f.to_bits()).collect());
    }
    Ok((conv, s))
}

pub(super) fn state_diff(
    r: &(Vec<Vec<u32>>, Vec<Vec<u32>>),
    c: &(Vec<Vec<u32>>, Vec<Vec<u32>>),
) -> (f64, usize) {
    let mut maxd = 0.0f64;
    let mut neq = 0usize;
    for (lr, lc) in r.0.iter().zip(c.0.iter()).chain(r.1.iter().zip(c.1.iter())) {
        for (a, b) in lr.iter().zip(lc.iter()) {
            if a != b { neq += 1; }
            let (fa, fb) = (f32::from_bits(*a) as f64, f32::from_bits(*b) as f64);
            maxd = maxd.max((fa - fb).abs());
        }
    }
    (maxd, neq)
}

/// `--probe-state` on an EXL3 pack: lane-count state-equivalence probe.
/// For N = 2..=8: slot 0 consumes the SAME prompt+continuation token stream in an
/// N-lane forward_step sweep (lanes 1..N run deterministic distractor prompts),
/// and slot 0's conv/GDN state and argmax stream must be BIT-IDENTICAL to the
/// N=1 reference (the G-A3-5 batch-invariance contract, end-to-end on real
/// tokens and recurrent state — the property the f-d chain-verify lanes need).
pub fn probe_state_exl3(a: &FwdArgs) -> Result<()> {
    let w = 8usize;
    let k_steps = a.max_new.clamp(4, 64);
    let model = FwdModel::load(&a.dir, w, a.max_pos)?;
    let mut sc = Scratch::new(&model.dev, &model.cfg, w, model.cfg.rotary_dim, model.max_pos)?;
    let ids = gate_prompt_ids(a)?;
    let v = model.cfg.vocab_size as i32;
    crate::dprintln!("EXL3 GDN state-equivalence probe: prompt={} tokens, K={} continuations, N=1..={w}",
             ids.len(), k_steps);

    // ---- N=1 reference: slot 0 alone, m=1 sweeps ----
    model.reset_slot(0)?;
    let mut pos = 0usize;
    for t in &ids {
        model.forward_step(&mut sc, &[*t as i32], &[pos], &[0], None)?;
        pos += 1;
    }
    let mut ref_toks: Vec<u32> = Vec::with_capacity(k_steps);
    let mut cur = model.dev.dtoh_sync_copy(&sc.argmax)?[0];
    for _ in 0..k_steps {
        ref_toks.push(cur as u32);
        model.forward_step(&mut sc, &[cur], &[pos], &[0], None)?;
        pos += 1;
        cur = model.dev.dtoh_sync_copy(&sc.argmax)?[0];
    }
    let ref_state = snap_slot0_state(&model)?;

    let mut all_exact = true;
    let mut first_bad: Option<usize> = None;
    let v = model.cfg.vocab_size as i32;
    let junk = |step: usize, slot: usize| -> i32 {
        ((step * 131 + slot * 7919 + 17) % ((v - 8) as usize) + 4) as i32
    };
    for n in 2..=w {
        for s in 0..n { model.reset_slot(s)?; }
        let mut posn = vec![0usize; n];
        // prefill sweeps: lane 0 walks the real prompt
        for (i, t) in ids.iter().enumerate() {
            let toks: Vec<i32> = (0..n).map(|s| if s == 0 { *t as i32 } else { junk(i, s) }).collect();
            let slots: Vec<usize> = (0..n).collect();
            model.forward_step(&mut sc, &toks, &posn, &slots, None)?;
            for p in posn.iter_mut() { *p += 1; }
        }
        // continuation sweeps: lane 0 consumes the REFERENCE tokens. The argmax read
        // AFTER sweep j predicts position plen+j+1, i.e. ref_toks[j+1] — the stream
        // under comparison therefore starts with the POST-PREFILL argmax (ref_toks[0]).
        let mut lane0: Vec<u32> = Vec::with_capacity(k_steps + 1);
        lane0.push(model.dev.dtoh_sync_copy(&sc.argmax)?[0] as u32);
        for (i, &rt) in ref_toks.iter().enumerate() {
            let toks: Vec<i32> = (0..n).map(|s| if s == 0 { rt as i32 } else { junk(i, s) }).collect();
            let slots: Vec<usize> = (0..n).collect();
            model.forward_step(&mut sc, &toks, &posn, &slots, None)?;
            for p in posn.iter_mut() { *p += 1; }
            lane0.push(model.dev.dtoh_sync_copy(&sc.argmax)?[0] as u32);
        }
        let st = snap_slot0_state(&model)?;
        let (maxd, neq) = state_diff(&ref_state, &st);
        // lane0[j] (argmax after consuming ref_toks[j-1]) must equal ref_toks[j]
        let amiss = lane0.iter().zip(ref_toks.iter()).filter(|(a, b)| a != b).count();
        let exact = maxd == 0.0 && neq == 0 && amiss == 0;
        all_exact &= exact;
        crate::dprintln!("N={n}: max |s_state(lane0@m={n}) - s_state(@m=1)| = {maxd:.7}  \
                  bit-neq={neq}  argmax-mismatch={amiss}  {}",
                 if exact { "EXACT MATCH" } else { "DIVERGES" });
        if !exact { first_bad = first_bad.or(Some(n)); }
    }
    // ---- mechanism datum: compare LOGITS row 0 (f16 bits) m=1 vs first-bad N ----
    if let (Some(nbad), false) = (first_bad, all_exact) {
        // replay the reference logits for the first continuation step, then the N-lane ones
        let lg = |model: &FwdModel, sc: &Scratch| -> Vec<u16> { model.logits_host(sc).unwrap() };
        for s in 0..=nbad { model.reset_slot(s).ok(); }
        let mut posn = vec![0usize; nbad];
        for (i, t) in ids.iter().enumerate() {
            let toks: Vec<i32> = (0..nbad).map(|s| if s == 0 { *t as i32 } else { junk(i, s) }).collect();
            let slots: Vec<usize> = (0..nbad).collect();
            model.forward_step(&mut sc, &toks, &posn, &slots, None)?;
            for p in posn.iter_mut() { *p += 1; }
        }
        let mut nref: Vec<u16>;
        {
            // m=1 reference logits at the same position
            for s in 0..1 { model.reset_slot(s).ok(); }
            let mut p0 = 0usize;
            for t in ids.iter() {
                model.forward_step(&mut sc, &[*t as i32], &[p0], &[0], None)?;
                p0 += 1;
            }
            // continue one step at m=1, read logits AFTER it (the compared position)
            model.forward_step(&mut sc, &[ref_toks[0] as i32], &[p0], &[0], None)?;
            nref = lg(&model, &sc);
        }
        for s in 0..nbad { model.reset_slot(s).ok(); }
        let mut posn = vec![0usize; nbad];
        for (i, t) in ids.iter().enumerate() {
            let toks: Vec<i32> = (0..nbad).map(|s| if s == 0 { *t as i32 } else { junk(i, s) }).collect();
            let slots: Vec<usize> = (0..nbad).collect();
            model.forward_step(&mut sc, &toks, &posn, &slots, None)?;
            for p in posn.iter_mut() { *p += 1; }
        }
        model.forward_step(&mut sc, &[ref_toks[0] as i32], &posn.clone(), &(0..nbad).collect::<Vec<_>>(), None)?;
        let nbad_lg = lg(&model, &sc);
        let v = model.cfg.vocab_size;
        let mut ndiff = 0usize;
        let mut max_ulp = 0i32;
        let mut arg_ref = 0usize;
        let mut arg_bad = 0usize;
        for i in 0..v {
            let (a, b) = (nref[i], nbad_lg[i]);
            let (fa, fb) = (half::f16::from_bits(a).to_f32(), half::f16::from_bits(b).to_f32());
            if fa > half::f16::from_bits(nref[arg_ref]).to_f32() { arg_ref = i; }
            if fb > half::f16::from_bits(nbad_lg[arg_bad]).to_f32() { arg_bad = i; }
            if a != b {
                ndiff += 1;
                max_ulp = max_ulp.max((b as i32) - (a as i32)).max((a as i32) - (b as i32));
            }
        }
        crate::dprintln!("LOGITS row0 @m={nbad} vs @m=1 (first cont step): {ndiff}/{v} entries differ, \
                  max ulp {max_ulp}, argmax m=1={arg_ref} (p={:.5}) vs m={nbad}={arg_bad} (p={:.5})",
                 half::f16::from_bits(nref[arg_ref]).to_f32(),
                 half::f16::from_bits(nbad_lg[arg_bad]).to_f32());
    }
    crate::dprintln!("RESULT: {} (exl3 probe_state N=2..={w}, K={k_steps})",
             if all_exact { "EXACT" } else { "DIVERGES" });
    if !all_exact { std::process::exit(1); }
    Ok(())
}

/// Q1 diagnostic (--prefillx-rel=1): slot-0 K/V cache rows of every attention layer for
/// positions < n, dequantized to f32 (q8: the rotated basis — L2 is invariant under the
/// orthonormal H32 rotation), K plane then V plane. Empty for formats without a host decoder.
pub(super) fn snap_slot0_kv(model: &FwdModel, n: usize) -> Result<Vec<Vec<f32>>> {
    let cfg = &model.tc; // TP-B: this rank's KV heads
    let (nkv, hd, mp) = (cfg.num_kv_heads, cfg.head_dim, model.max_pos);
    let rb = kv_rowbytes(model.kv_fmt, hd);
    let mut out = Vec::with_capacity(model.k_cache.len());
    for kc in &model.k_cache {
        let b: Vec<u8> = model.dev.dtoh_sync_copy(kc)?;
        let mut v = Vec::with_capacity(2 * nkv * n * hd);
        for plane in 0..2 {
            for h in 0..nkv {
                for p in 0..n.min(mp) {
                    let off = ((plane * nkv + h) * mp + p) * rb;
                    let row = &b[off..off + rb];
                    match model.kv_fmt {
                        KV_Q8 => v.extend(kvnll::q8_dequant_row(row, hd)),
                        KV_F32 => v.extend(row.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))),
                        KV_F16 => v.extend(row.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())),
                        _ => {}
                    }
                }
            }
        }
        out.push(v);
    }
    Ok(out)
}

/// Q1: (relL2 = ||r-c||/||r||, max|r-c|, max|r|) of one plane.
fn rel_stats(r: &[f32], c: &[f32]) -> (f64, f64, f64) {
    let (mut d2, mut r2, mut md, mut mr) = (0f64, 0f64, 0f64, 0f64);
    for (&a, &b) in r.iter().zip(c.iter()) {
        let (a, b) = (a as f64, b as f64);
        d2 += (a - b) * (a - b); r2 += a * a; md = md.max((a - b).abs()); mr = mr.max(a.abs());
    }
    ((d2 / r2.max(1e-300)).sqrt(), md, mr)
}

/// Q1 magnitude report (--prefillx-rel=1|2): per GDN layer conv + recurrent S, per attention
/// layer K/V cache — relative L2 and max-abs vs the reference; one PFX_RELSUM summary line
/// (=2 also prints every layer).
/// Returns (S relL2 median, KV relL2 median — None when the KV format has no host decoder).
pub(super) fn prefillx_rel_report(tag: &str, sr: &(Vec<Vec<u32>>, Vec<Vec<u32>>), sc: &(Vec<Vec<u32>>, Vec<Vec<u32>>),
                       kr: &[Vec<f32>], kc: &[Vec<f32>]) -> (f64, Option<f64>) {
    let f = |v: &Vec<u32>| -> Vec<f32> { v.iter().map(|&b| f32::from_bits(b)).collect() };
    let verbose = crate::opts::var(crate::opt!("prefillx-rel")).as_deref() == Ok("2");
    let (mut s_rel, mut c_rel, mut kv_rel) = (Vec::new(), Vec::new(), Vec::new());
    let (mut s_md, mut s_mr, mut s_md_over_mr) = (0f64, 0f64, 0f64);
    for (i, ((cr, cc), (ssr, ssc))) in sr.0.iter().zip(sc.0.iter()).zip(sr.1.iter().zip(sc.1.iter())).enumerate() {
        let (a, _, _) = rel_stats(&f(cr), &f(cc));
        let (b, md, mr) = rel_stats(&f(ssr), &f(ssc));
        if verbose { crate::dprintln!("PFX_REL[{tag}] gdn {i:>2}: conv relL2 {a:.3e} | S relL2 {b:.3e} max|d| {md:.4e} max|ref| {mr:.4e}"); }
        c_rel.push(a); s_rel.push(b);
        s_md = s_md.max(md); s_mr = s_mr.max(mr); s_md_over_mr = s_md_over_mr.max(md / mr.max(1e-30));
    }
    for (i, (a, b)) in kr.iter().zip(kc.iter()).enumerate() {
        let (r, md, mr) = rel_stats(a, b);
        if verbose { crate::dprintln!("PFX_REL[{tag}] attn {i:>2}: K|V relL2 {r:.3e} max|d| {md:.4e} max|ref| {mr:.4e}"); }
        kv_rel.push(r);
    }
    let summ = |v: &mut Vec<f64>| -> String {
        if v.is_empty() { return "n/a".into(); }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        format!("med {:.3e} max {:.3e}", v[v.len() / 2], v[v.len() - 1])
    };
    crate::dprintln!("PFX_RELSUM[{tag}] S relL2 {} (layer-max |d|/max|ref| {:.3e}; max|d| {:.4e} of max|ref| {:.4e}) | conv relL2 {} | KV relL2 {}",
             summ(&mut s_rel), s_md_over_mr, s_md, s_mr, summ(&mut c_rel), summ(&mut kv_rel));
    // summ sorted both in place
    let med = |v: &Vec<f64>| if v.is_empty() { None } else { Some(v[v.len() / 2]) };
    (med(&s_rel).unwrap_or(f64::NAN), med(&kv_rel))
}

/// Minimal MD5 (RFC 1321) for the prefillx pinned signature — the gate scripts' historical
/// `grep -E "^chunk C=|^RESULT|^reference" log | md5sum | cut -c1-12`, computed in-process.
pub(super) fn md5_hex(msg: &[u8]) -> String {
    const S: [u32; 64] = [7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                          5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
                          4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                          6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21];
    let k: Vec<u32> = (0..64).map(|i| ((i as f64 + 1.0).sin().abs() * 4294967296.0) as u32).collect();
    let mut h: [u32; 4] = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476];
    let mut m = msg.to_vec();
    let bits = (msg.len() as u64).wrapping_mul(8);
    m.push(0x80);
    while m.len() % 64 != 56 { m.push(0); }
    m.extend_from_slice(&bits.to_le_bytes());
    for blk in m.chunks_exact(64) {
        let w: Vec<u32> = blk.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let (mut a, mut b, mut c, mut d) = (h[0], h[1], h[2], h[3]);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let t = d;
            d = c;
            c = b;
            b = b.wrapping_add(a.wrapping_add(f).wrapping_add(k[i]).wrapping_add(w[g]).rotate_left(S[i]));
            a = t;
        }
        h[0] = h[0].wrapping_add(a); h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c); h[3] = h[3].wrapping_add(d);
    }
    h.iter().flat_map(|x| x.to_le_bytes()).map(|b| format!("{b:02x}")).collect()
}

/// L10 tolerance tier (Q1 recommendation): one chunked arm passes iff its S and KV relL2 medians
/// are <= 3x the same run's one-f32-ulp floor (bit 0) and its first token sits in the reference's
/// near-tie (rank <= 5, gap to top-1 <= 1.0 logit). Returns the failed criteria (empty = pass).
pub(super) fn prefillx_tier_fails(arm_s: f64, arm_kv: Option<f64>, floor_s: f64, floor_kv: Option<f64>,
                       rank: usize, gap: f32) -> Vec<String> {
    let mut f = Vec::new();
    if !(arm_s <= 3.0 * floor_s) { f.push(format!("S relL2 {arm_s:.3e} > 3x floor {floor_s:.3e}")); }
    match (arm_kv, floor_kv) {
        (Some(a), Some(fl)) => if !(a <= 3.0 * fl) { f.push(format!("KV relL2 {a:.3e} > 3x floor {fl:.3e}")); },
        (None, None) => {}
        _ => f.push("KV relL2 missing on one side".into()),
    }
    if rank > 5 { f.push(format!("first-token rank {rank} > 5")); }
    if !(gap <= 1.0) { f.push(format!("first-token gap {gap:.4} > 1.0 logit")); }
    f
}


/// K9 (REL v0.7.3 review): the doc block below had wandered onto PACK_SEG_MAX.
/// `--bench-mtp` on an EXL3 pack: REAL chain-verify losslessness + draft
/// acceptance. Arm M (slot 0) prefills token-by-token, primes the head ONCE
/// (tap_snapshot + head_prime — the serve's prime), then decodes with
/// `mtp_round` (k+1-width verify, on-device accept, shadow commit). Arm S
/// (slot 0 after a state reset) runs the same generation with the head DEAD
/// and plain sequential decode. The trunk token streams must be BIT-IDENTICAL —
/// the verify/accept/commit machinery must reproduce sequential greedy exactly.
pub fn bench_mtp_exl3(a: &FwdArgs, depth: usize) -> Result<()> {
    let depth = depth.max(1);
    let model = FwdModel::load(&a.dir, 1, a.max_pos)?;
    let head = match model.mtp.as_ref() {
        Some(h) => h,
        None => bail!("pack has no mtp.* draft head — MTP probe unavailable"),
    };
    // scratch holds the VERIFY width m=k+1 (mtp_round asserts it); arm S's m=1
    // steps ride the same buffer fine.
    let mut sc = Scratch::new(&model.dev, &model.cfg, (depth + 1).max(2), model.cfg.rotary_dim, model.max_pos)?;
    let prompt_ids = gate_prompt_ids(a)?;
    let plen = prompt_ids.len();
    crate::dprintln!("MTP end-to-end probe (exl3): prompt={plen} tokens, depth={depth}, max_new={} model={}",
             a.max_new, a.dir);

    // ---- Arm M: REAL chain-verify rounds (mtp_round) — the path the serve runs.
    // Token-by-token prefill writes sc (the true tap), one prime, then k+1-width
    // verify rounds. The lossless contract is actual == seq_tokens (arm S):
    // verify/accept/commit must reproduce sequential greedy bit-for-bit.
    // (The OLD arm M ran plain sequential decode with the head drafting alongside
    // and scored drafts post-hoc — it never executed the chain-verify round, so
    // "LOSSLESS_OK" proved only that priming does not perturb the trunk. That
    // same-class equivalence is exactly the §3 blind spot; bug 5 lived behind it.)
    // S-A3-x --mtp-cover=<L>: plain greedy FIRST (the true tokens), then per-round teacher-forced
    // head ranks at levels 1..L before each real round; CSV to --mtp-cover-out (default /tmp/mtp_cover.csv).
    let cover_l: usize = crate::opts::var(crate::opt!("mtp-cover")).ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut truth: Vec<u32> = Vec::new();
    let mut cover_rows: Vec<(usize, Vec<usize>, usize)> = Vec::new();
    if cover_l > 0 {
        let mut psc_t: Option<PrefillScratch> = None;
        model.reset_slot(0)?;
        model.prefill_prompt(&mut sc, &prompt_ids, a.prefill_chunk, &mut psc_t)?;
        truth.push(model.dev.dtoh_sync_copy(&sc.argmax)?[0] as u32);
        for k in 0..(a.max_new + cover_l + depth + 2) {
            let tok = truth[k] as i32;
            if EOS_IDS.contains(&tok) { break; }
            model.forward_step(&mut sc, &[tok], &[plen + k], &[0], None)?;
            truth.push(model.dev.dtoh_sync_copy(&sc.argmax)?[0] as u32);
        }
        model.reset_slot(0)?;
        crate::dprintln!("MTP_COVER: truth {} tokens, levels 1..={cover_l}", truth.len());
    }
    let mut actual: Vec<u32> = Vec::with_capacity(a.max_new);
    let t0 = std::time::Instant::now();
    // S-A3-t: --prefill-chunk C = the SERVED posture (chunked prompt[..n-1] + the last token
    // through forward_step) for BOTH arms — the only posture that exercises the head-KV fill.
    let mut psc_opt: Option<PrefillScratch> = None;
    model.prefill_prompt(&mut sc, &prompt_ids, a.prefill_chunk, &mut psc_opt)?;
    let pred0 = model.dev.dtoh_sync_copy(&sc.argmax)?[0];
    actual.push(pred0 as u32);
    // DIAG (--bmtp-diag=plain): run ONE plain forward_step consuming pred0 and
    // print its argmax — the col-0 ground truth — then restore state via reset+reprefill.
    if crate::opts::var(crate::opt!("bmtp-diag")).as_deref() == Ok("plain") {
        model.forward_step(&mut sc, &[pred0 as i32], &[plen], &[0], None)?;
        crate::dprintln!("DIAG plain-step argmax after pred0: {}",
                 model.dev.dtoh_sync_copy(&sc.argmax)?[0]);
        model.reset_slot(0)?;
        let mut pos = 0usize;
        for t in prompt_ids.iter() {
            model.forward_step(&mut sc, &[*t as i32], &[pos], &[0], None)?;
            pos += 1;
        }
    }
    // the serve's prime: snapshot the tap that predicted pred0, feed the head the
    // last prompt token with it (round 1's draft-extend reuses taps_keep[0]).
    if crate::opts::var(crate::opt!("bmtp-diag")).as_deref() != Ok("noprime") {
        let last_prompt = *prompt_ids.last().unwrap() as i32;
        model.mtp_seam_prime(&mut sc, psc_opt.as_ref(), head, last_prompt, plen, 0)?; // WP03
    } else {
        model.tap_snapshot(&mut sc, 0)?;
        crate::dprintln!("DIAG prime SKIPPED (noprime)");
    }
    let mut rounds: Vec<DraftRound> = Vec::new();
    let mut trunk_ms: Vec<f64> = Vec::new();
    let mut head_ms: Vec<f64> = Vec::new();
    {
        let mut b = pred0 as i32;
        let mut p = plen;
        let mut tap_row = 0usize;
        // S-A3-t: a first token that is EOS ends the request (the serve stops there; the plain arm
        // does too) — the MTP arm used to keep drafting past it.
        let first_eos = EOS_IDS.contains(&(pred0 as i32));
        while !first_eos && actual.len() < a.max_new {
            if cover_l > 0 {
                let off = p - plen;   // truth[off] = b (the token entering at p)
                if off + cover_l < truth.len() && truth[off] as i32 == b {
                    let tg: Vec<i32> = truth[off + 1..=off + cover_l].iter().map(|&x| x as i32).collect();
                    let r = model.mtp_cover(&mut sc, head, b, p, tap_row, &tg)?;
                    cover_rows.push((p, r, 0));
                }
            }
            let tt = std::time::Instant::now();
            // clamped at the sparse switch like the serve (S-B9-REL-SMALLFIXES item 2): a LOSSLESS
            // run crossing it no longer verifies with a straddling round
            let out = model.mtp_round_clamped(&mut sc, head, 0, depth, b, p, tap_row)?;
            if let Some(last) = cover_rows.last_mut() { if last.0 == p { last.2 = out.a; } }
            trunk_ms.push(tt.elapsed().as_secs_f64() * 1e3);
            for &e in out.emitted.iter() {
                actual.push(e as u32);
            }
            rounds.push(DraftRound { drafts: out.drafts, accepted: out.a });
            b = out.last_tok;
            p = out.pos_next;
            tap_row = out.tap_row;
            if EOS_IDS.contains(&b) { break; }
        }
    }
    // S-A3-f-d Item 1b FIX-3 companion: a round emits a+1 tokens (accepted +
    // bonus), so the chain can overshoot max_new. Trim to the requested horizon
    // — the lossless compare is exact-length by design.
    actual.truncate(a.max_new);
    let mtp_s = t0.elapsed().as_secs_f64();

    // ---- Arm S: same generation, head DEAD (state fully reset first) ----
    model.reset_slot(0)?;
    let mut seq_tokens: Vec<u32> = Vec::with_capacity(a.max_new);
    let t1 = std::time::Instant::now();
    model.prefill_prompt(&mut sc, &prompt_ids, a.prefill_chunk, &mut psc_opt)?;
    seq_tokens.push(model.dev.dtoh_sync_copy(&sc.argmax)?[0] as u32);
    for k in 0..a.max_new.saturating_sub(1) {
        let tok = seq_tokens[k] as i32;
        if EOS_IDS.contains(&tok) { break; }
        model.forward_step(&mut sc, &[tok], &[plen + k], &[0], None)?;
        seq_tokens.push(model.dev.dtoh_sync_copy(&sc.argmax)?[0] as u32);
    }
    let seq_s = t1.elapsed().as_secs_f64();

    // ---- acceptance scoring from the DEVICE accept count (out.a per round).
    // The chain round emits a variable number of tokens, so the old post-hoc
    // stream-index scoring (k+1+j) no longer applies; out.a is authoritative.
    // (drafts actually run: a round clamped at the sparse switch drafts fewer than `depth`)
    let drafted: usize = rounds.iter().map(|r| r.drafts.len()).sum();
    let accepted: usize = rounds.iter().map(|r| r.accepted).sum();
    let prefix_sum: usize = rounds.iter().map(|r| r.accepted).sum();
    let med = |v: &mut Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.get(v.len() / 2).copied().unwrap_or(0.0)
    };
    let (tms, hms) = (med(&mut trunk_ms), med(&mut head_ms));
    let acc_pct = if drafted > 0 { 100.0 * accepted as f64 / drafted as f64 } else { 0.0 };
    let mean_prefix = if rounds.is_empty() { 0.0 } else { prefix_sum as f64 / rounds.len() as f64 };
    let mtp_tok_s = if mtp_s > 0.0 { actual.len() as f64 / mtp_s } else { 0.0 };
    let seq_tok_s = if seq_s > 0.0 { seq_tokens.len() as f64 / seq_s } else { 0.0 };
    crate::dprintln!("MTP tokens : {:?}", &actual[..actual.len().min(20)]);
    crate::dprintln!("SEQ tokens : {:?}", &seq_tokens[..seq_tokens.len().min(20)]);
    crate::dprintln!("acceptance rate: {:.1}%  (drafts accepted / total drafts, 1-ahead)  mean matched prefix {:.3}",
             acc_pct, mean_prefix);
    crate::dprintln!("trunk round (median) {tms:.1} ms   head draft x{depth} (median) {hms:.1} ms");
    crate::dprintln!("throughput: MTP {:.1} tok/s   sequential {:.1} tok/s", mtp_tok_s, seq_tok_s);

    if cover_l > 0 && !cover_rows.is_empty() {
        let path = crate::opts::var(crate::opt!("mtp-cover-out")).unwrap_or_else(|_| "/tmp/mtp_cover.csv".into());
        let mut csv = String::from("pos,accepted");
        for i in 1..=cover_l { csv += &format!(",r{i}"); }
        for i in 1..=cover_l { csv += &format!(",c{i}"); }
        csv.push('\n');
        for (p, r, acc) in &cover_rows {
            csv += &format!("{p},{acc}");
            for x in r { csv += &format!(",{}", if *x == usize::MAX { -1i64 } else { *x as i64 }); }
            csv.push('\n');
        }
        std::fs::write(&path, csv)?;
        let n = cover_rows.len() as f64;
        for lv in 0..cover_l {
            let hit = |j: usize| cover_rows.iter().filter(|(_, r, _)| r[lv] != usize::MAX && r[lv] < j).count() as f64 / n;
            let chain = cover_rows.iter().filter(|(_, r, _)| r[..=lv].iter().all(|&x| x == 0)).count() as f64 / n;
            crate::dprintln!("MTP_COVER level {}: top1 {:.3} top2 {:.3} top3 {:.3} top4 {:.3} | chain-prefix(all top1 to here) {:.3}",
                     lv + 1, hit(1), hit(2), hit(3), hit(4), chain);
        }
        crate::dprintln!("MTP_COVER: {} rounds -> {path}", cover_rows.len());
    }
    let lossless = actual == seq_tokens;
    if lossless {
        crate::dprintln!("RESULT: LOSSLESS_OK (MTP-primed output == plain greedy, {} tokens; exl3 path)",
                 actual.len());
        Ok(())
    } else {
        let div = actual.iter().zip(seq_tokens.iter()).position(|(a, b)| a != b);
        crate::dprintln!("RESULT: MISMATCH at token {:?} — head priming PERTURBS the trunk", div);
        if let Some(d) = div {
            let lo = d.saturating_sub(6);
            let hi = (d + 6).min(actual.len()).min(seq_tokens.len());
            crate::dprintln!("  ctx MTP[{}..{}]: {:?}", lo, hi, &actual[lo..hi]);
            crate::dprintln!("  ctx SEQ[{}..{}]: {:?}", lo, hi, &seq_tokens[lo..hi]);
        }
        std::process::exit(1);
    }
}
