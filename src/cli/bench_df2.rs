//! DFlash2 sampling benches: sampled run, real-question run and the matrix bench.
//! Speed numbers are taken at production sampling (AGENTS.md section 5); greedy is for correctness only.

use super::*;

/// S5F — `--bench-df2-sample`: the distribution-exactness gate for DFlash2-source sampled
/// decoding (workdoc §3.2, BINDING). Extends the `--bench-mtp-sample` two-sample chi-square
/// pattern to the DFlash2 source: the round's greedy draft (q=1) into `spec_verify_b`, emitted
/// distribution vs the plain sampler's (the control arm), ≥3 seeds, RNG streams varied across
/// conditions. The sampler claim is draft-source-independent (see bench_rejection_gate) — the
/// gate proves the DFlash2 lane's rejection-sampling emission matches plain sampling.
pub(crate) fn run_bench_df2_sample(args: &[String]) {
    let trunk_dir = parse_arg(args, "--model-dir")
        .expect("--bench-df2-sample needs --model-dir <trunk>").to_string();
    let prompt_text = parse_arg(args, "--prompt")
        .unwrap_or("Explain how a transformer's self-attention mechanism works, in three short sentences.");
    let trials: usize = parse_arg(args, "--trials").and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let top_k: usize = parse_arg(args, "--top-k").and_then(|s| s.parse().ok()).unwrap_or(20);
    let top_p: f32 = parse_arg(args, "--top-p").and_then(|s| s.parse().ok()).unwrap_or(0.8);
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);
    let temps: Vec<f32> = match parse_arg(args, "--temp") {
        Some(s) => vec![s.parse().expect("--temp")],
        None => vec![0.7, 1.0],
    };
    // ≥3 seeds: the caller varies the RNG base (the gate varies it per temperature below).
    let n_seeds: usize = parse_arg(args, "--seeds").and_then(|s| s.parse().ok()).unwrap_or(3);

    let tokenizer = QwenTokenizer::from_file(&format!("{}/tokenizer.json", trunk_dir.trim_end_matches('/')))
        .expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");

    let (mut gpu, _) = load_model_gpu(&trunk_dir, None, 1);
    let df2 = load_df2_round(&mut gpu, max_seq_len);
    let (mut round, sink, prime) = match df2 {
        Some(x) => x,
        None => { println!("RESULT: DISTRIBUTION_MISMATCH (round not loadable)"); std::process::exit(1); }
    };
    // Prime the round with the REAL prompt taps (prefill capture) and draw ONE greedy draft —
    // the gate's fixed (committed, draft) pair; the sampler claim is source-independent, but the
    // round must be exercised with real taps to be the honest engine path.
    gpu.set_df2_prime_sink(prime.clone());
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    let mut state = gpu.new_batch_state(3, 3, max_seq_len);
    let plen = prompt.len();
    gpu.zero_slot_state(&mut state, 0, max_seq_len);
    let (a0, hout) = gpu.prefill_batch(&mut pool, &prompt, &mut state, 0, max_seq_len, 0);
    pool.release_bf16(hout, 5120 * plen);
    round.reset();
    round.prime_window(&prime.taps, plen, 0).expect("df2 prime");
    gpu.set_df2_prime_off();
    round.refresh_block_pos().expect("df2 refresh");
    let drafts = round.draft_round_dev(a0).expect("df2 draft");
    let x_draft = drafts[0];
    println!("DFlash2 gate: prompt={} tokens, round draft[0]={x_draft}, trials={trials}, top_k={top_k}, top_p={top_p}",
             plen);

    const ZBAR: f32 = 4.0;
    let mut all_pass = true;
    for (ti, &temp) in temps.iter().enumerate() {
        for si in 0..n_seeds {
            // Independent RNG stream per (temperature, seed) condition (the AGENTS §3 rule).
            let base = 0xA5A5_1234_0000_0000u64
                ^ ((ti as u64 * 31 + si as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let s = gpu.bench_rejection_gate(&mut pool, &mut state, &prompt, max_seq_len,
                                             a0, x_draft, temp, top_k, top_p, trials, base);
            let p_agree = (s.p_draft_analytic - s.p_draft_device).abs();
            let pass = s.mtp_vs_sampler.z.abs() < ZBAR
                && s.bonus_vs_sampler.z.abs() < ZBAR
                && s.accept_z.abs() < ZBAR
                && p_agree < 1e-3;
            all_pass &= pass;
            println!("temp={temp:.2} seed={si}: p(draft)={:.4} (dev {:.4}, Δ={:.1e}) | accept {:.4} (z={:+.2}) | \
                      GATE DF2-vs-sampler z={:+.2} chi2/df={:.3} | bonus z={:+.2} | {}",
                     s.p_draft_analytic, s.p_draft_device, p_agree,
                     s.accept_rate, s.accept_z, s.mtp_vs_sampler.z,
                     s.mtp_vs_sampler.chi2_over_df, s.bonus_vs_sampler.z,
                     if pass { "PASS" } else { "FAIL" });
        }
    }
    if all_pass {
        println!("RESULT: DISTRIBUTION_OK (DFlash2-source sampled decoding is distribution-exact \
                  vs the plain sampler)");
    } else {
        println!("RESULT: DISTRIBUTION_MISMATCH");
        std::process::exit(1);
    }
}

/// S5F2 L2 — `--bench-df2-sample-realq`: the REAL-q distribution-exactness gate. Same two-sample
/// chi-square structure as `--bench-df2-sample` (≥3 seeds, control arm, varied RNG streams), but
/// the draft is drawn from the SAMPLED selector path (`Df2Round::draft_round_dev_sample` — the
/// SGLang `sample_path` semantics) and the verify uses the real-q kernel + host accept `u·q < p`
/// with the exact relu(p−q) residual — the L2 chi-square gate. For each (temperature, seed)
/// condition a FRESH sampled draft + candidate table is drawn (the draft is part of the
/// condition; the RNG stream varies across conditions per the AGENTS §3 rule).
pub(crate) fn run_bench_df2_sample_realq(args: &[String]) {
    use gb10_inference::batch::{rng_u32, RNG_DOM_DF2_SEL};
    let trunk_dir = parse_arg(args, "--model-dir")
        .expect("--bench-df2-sample-realq needs --model-dir <trunk>").to_string();
    let prompt_text = parse_arg(args, "--prompt")
        .unwrap_or("Explain how a transformer's self-attention mechanism works, in three short sentences.");
    let trials: usize = parse_arg(args, "--trials").and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let top_k: usize = parse_arg(args, "--top-k").and_then(|s| s.parse().ok()).unwrap_or(20);
    let top_p: f32 = parse_arg(args, "--top-p").and_then(|s| s.parse().ok()).unwrap_or(0.8);
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);
    let temps: Vec<f32> = match parse_arg(args, "--temp") {
        Some(s) => vec![s.parse().expect("--temp")],
        None => vec![0.7, 1.0],
    };
    let n_seeds: usize = parse_arg(args, "--seeds").and_then(|s| s.parse().ok()).unwrap_or(3);

    let tokenizer = QwenTokenizer::from_file(&format!("{}/tokenizer.json", trunk_dir.trim_end_matches('/')))
        .expect("tokenizer");
    let prompt = tokenizer.encode(prompt_text, true).expect("encode");

    let (mut gpu, _) = load_model_gpu(&trunk_dir, None, 1);
    let df2 = load_df2_round(&mut gpu, max_seq_len);
    let (mut round, sink, prime) = match df2 {
        Some(x) => x,
        None => { println!("RESULT: DISTRIBUTION_MISMATCH (round not loadable)"); std::process::exit(1); }
    };
    gpu.set_df2_prime_sink(prime.clone());
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    let mut state = gpu.new_batch_state(3, 3, max_seq_len);
    let plen = prompt.len();
    round.reset();
    round.prime_window(&prime.taps, plen, 0).expect("df2 prime");
    gpu.set_df2_prime_off();

    const ZBAR: f32 = 4.0;
    let mut all_pass = true;
    for (ti, &temp) in temps.iter().enumerate() {
        for si in 0..n_seeds {
            // Independent RNG stream per (temperature, seed) condition (the AGENTS §3 rule).
            let base = 0xBEEF_1234_0000_0000u64
                ^ ((ti as u64 * 31 + si as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            // FRESH trunk state per condition: the verify ADVANCES the GDN state, so a shared
            // state would verify column 0 against a stale recurrent context from the previous
            // condition (self-consistent but wrong logits — the gate's cross-condition bug).
            gpu.zero_slot_state(&mut state, 0, max_seq_len);
            let (a0, hout) = gpu.prefill_batch(&mut pool, &prompt, &mut state, 0, max_seq_len, 0);
            pool.release_bf16(hout, 5120 * plen);
            round.refresh_block_pos().expect("df2 refresh");
            // A FRESH sampled draft per condition: the proposal distribution is part of the
            // condition under test (varied RNG streams across conditions).
            let sel_seeds: Vec<u32> = (0..gb10_inference::dflash2::levels()).map(|j| rng_u32(base, RNG_DOM_DF2_SEL, j)).collect();
            let dout = round.draft_round_dev_sample(a0, &sel_seeds, temp).expect("df2 sampled draft");
            let x_draft = dout.tokens[0];
            let q_draft = dout.q_rows[0];
            let cand_tok = &dout.cand_tok[..16];
            let cand_q = &dout.cand_q[..16];
            let s = gpu.bench_rejection_gate_rq(&mut pool, &mut state, &prompt, max_seq_len,
                                                a0, x_draft, q_draft, cand_tok, cand_q,
                                                temp, top_k, top_p, trials, base);
            // p-consistency: per-trial mean |device p(draft) - host p(draft)| (the single-draft
            // p_analytic-vs-device comparison is meaningless here — the per-trial drafts vary).
            let pass = s.mtp_vs_sampler.z.abs() < ZBAR
                && s.bonus_vs_sampler.z.abs() < ZBAR
                && s.accept_z.abs() < ZBAR
                && s.p_agree_mean < 1e-3;
            all_pass &= pass;
            println!("realq temp={temp:.2} seed={si}: draft={x_draft} q={q_draft:.4} | p(draft)={:.4} (dev {:.4}) | \
                      accept {:.4} (z={:+.2}, E={:.4}) p-Δ={:.1e} | GATE realq-vs-sampler z={:+.2} chi2/df={:.3} | bonus z={:+.2} | {}",
                     s.p_draft_analytic, s.p_draft_device,
                     s.accept_rate, s.accept_z, s.accept_e, s.p_agree_mean,
                     s.mtp_vs_sampler.z,
                     s.mtp_vs_sampler.chi2_over_df, s.bonus_vs_sampler.z,
                     if pass { "PASS" } else { "FAIL" });
        }
    }
    if all_pass {
        println!("RESULT: DISTRIBUTION_OK (DFlash2 real-q sampled decoding is distribution-exact \
                  vs the plain sampler)");
    } else {
        println!("RESULT: DISTRIBUTION_MISMATCH");
        std::process::exit(1);
    }
}

/// S5F — `--bench-df2-matrix`: ONE cell-group of the on-engine τ matrix (workdoc §3.5): the
/// given (spec-source, regime) over the S3T3 domain prompt sets, ≥`--reps` reps/cell, ≥40
/// tokens/gen, pinned MTP depth. Per cell (domain × rep): τ (mean emitted incl. bonus over full
/// steps) + spread, tok/s, the per-step breakdown (round / verify / step). The orchestrator
/// fans out 6 cell-groups (3 sources × 2 regimes) across `.11`/`.12`.
pub(crate) fn run_bench_df2_matrix(args: &[String]) {
    use gb10_inference::batch::{SpecBenchJob, SpecSource};
    let trunk_dir = parse_arg(args, "--model-dir")
        .expect("--bench-df2-matrix needs --model-dir <trunk>").to_string();
    let spec_source = match parse_arg(args, "--spec-source").map(str::to_lowercase) {
        Some(s) => SpecSource::from_cli(&s)
            .unwrap_or_else(|| panic!("--spec-source must be mtp|dspark|dflash2|dflash2-rq|dflash2-auto|dflash2-tree|none (got {s})")),
        None => panic!("--bench-df2-matrix needs --spec-source {{mtp,dflash2,dflash2-rq,dflash2-auto,none}}"),
    };
    let temp: f32 = parse_arg(args, "--temp").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let max_new: usize = parse_arg(args, "--max-new-tokens")
        .and_then(|s| s.parse().ok()).unwrap_or(40);
    let reps: usize = parse_arg(args, "--reps").and_then(|s| s.parse().ok()).unwrap_or(3);
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len")
        .and_then(|s| s.parse().ok()).unwrap_or(4096);
    let mtp_depth: usize = parse_arg(args, "--mtp-depth")
        .and_then(|s| s.parse().ok()).unwrap_or(4);
    let out_path = parse_arg(args, "--out").map(str::to_string)
        .unwrap_or_else(|| format!("/tmp/s5f/matrix_{}_{}.json", spec_source.cli_name(),
                                   if temp < 1e-6 { "greedy" } else { "samp" }));
    let domains: Vec<&str> = parse_arg(args, "--domains")
        .map(|s| s.split(',').collect())
        .unwrap_or_else(|| vec!["math", "code", "chat"]);
    // S3R protocol-parity mode (orchestrator addendum 2026-08-20): the credited SGLang recipe —
    // chat template + enable_thinking + reasoning_effort, temp 1.0 / top-p 0.95 / top-k 20,
    // long generations (>=1024). --parity forces exactly that (thinking=xhigh, math domain only);
    // --thinking <effort> applies the thinking template without forcing the other knobs.
    // (S-B9-REL-VISIONFIX2 V5: a registry flag; `parse_arg` is the value reader and ignored it as the last argument)
    let parity = gb10_inference::opts::on(gb10_inference::opt!("parity"));
    let thinking = if parity { Some("xhigh".to_string()) } else { parse_arg(args, "--thinking").map(str::to_string) };
    let (temp, top_p, top_k, max_new) = if parity {
        let mn: usize = parse_arg(args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(1024);
        (1.0, 0.95, 20usize, mn)
    } else {
        (temp, 0.95, 20usize, max_new)
    };
    let domains: Vec<&str> = if parity { vec!["math"] } else { domains };

    // S9F (L5): `--prompt-file <json>` — replace the S3T3 matrix template with an explicit
    // prompt set ([{"id": .., "domain": .., "text": ..}, ...]) so the τ headline can run on
    // the EXACT S3R/GSM8K prompt set (the A0 seed-42 slice) instead of the hand-written
    // S3T3 gsm1–5 prompts. The parity protocol (chat template + xhigh thinking + T1.0/p0.95/k20)
    // applies to `math`-domain entries exactly as it does to S3T3_PROMPTS.
    let prompt_set: Vec<(String, String, String)> = match parse_arg(args, "--prompt-file") {
        Some(f) => {
            let raw = std::fs::read_to_string(&f).expect("read --prompt-file");
            let v: serde_json::Value = serde_json::from_str(&raw).expect("--prompt-file must be a JSON array");
            let arr = v.as_array().expect("--prompt-file must be a JSON array");
            arr.iter().map(|e| {
                let id = e.get("id").and_then(|x| x.as_str()).unwrap_or("?").to_string();
                let dom = e.get("domain").and_then(|x| x.as_str()).unwrap_or("math").to_string();
                let text = e.get("text").and_then(|x| x.as_str())
                    .unwrap_or_else(|| panic!("--prompt-file entry {id} has no text")).to_string();
                (id, dom, text)
            }).collect()
        }
        None => S3T3_PROMPTS.iter().map(|(p, d, t)| (p.to_string(), d.to_string(), t.to_string())).collect(),
    };

    let tokenizer = QwenTokenizer::from_file(&format!("{}/tokenizer.json", trunk_dir.trim_end_matches('/')))
        .expect("tokenizer");
    let eos = tokenizer.stop_token_ids(config_eos(trunk_dir.trim_end_matches('/')));
    let (mut gpu, _) = load_model_gpu(&trunk_dir, None, 1);
    let (round, sink, prime) = if gb10_inference::batch::is_df2_src(spec_source) {
        match load_df2_round(&mut gpu, max_seq_len) {
            Some(x) => (Some(x.0), Some(x.1), Some(x.2)),
            None => {
                println!("RESULT: DF2_UNAVAILABLE — serving the DFlash2 cells via the MTP fallback \
                          would NOT be a DFlash2 measurement; refusing the cell-group. (See the \
                          [df2] WARN above for the load failure.)");
                std::process::exit(2);
            }
        }
    } else { (None, None, None) };
    // WI1: the DSpark source needs its own round wired in (before WI1 the scheduler got the
    // dspark triple as None and the cells silently ran MTP — a NOT-a-dspark measurement).
    // Same refuse-don't-fallback discipline as the DF2 arm: an unloadable drafter exits loudly.
    let (ds_round, ds_sink, ds_prime) = if spec_source == gb10_inference::batch::SpecSource::Dspark {
        let draft_dir = gb10_inference::opt_or_arg(args, "--draft-dir")
            .unwrap_or_else(|| panic!("--bench-df2-matrix --spec-source dspark needs --draft-dir \
                                      <Qwen3.8-27B-DSpark artifact>"));
        match load_dspark_round_dir(&mut gpu, max_seq_len, &draft_dir, None) {
            Some(x) => (Some(x.0), Some(x.1), Some(x.2)),
            None => {
                println!("RESULT: DSPARK_UNAVAILABLE — serving the DSpark cells via the MTP \
                          fallback would NOT be a DSpark measurement; refusing the cell-group. \
                          (See the [dspark] WARN above for the load failure.)");
                std::process::exit(2);
            }
        }
    } else { (None, None, None) };

    // One job per (domain prompt × rep). Seeds deterministic per (prompt, rep).
    // With --thinking/--parity, the MATH prompts are rendered through the model's chat template
    // with enable_thinking + reasoning_effort (the SGLang chat_template_kwargs the S3R recipe
    // credits) — the generation prompt ends inside the primed <think> block.
    let mut jobs: Vec<SpecBenchJob> = Vec::new();
    let mut job_meta: Vec<(String, String, usize, Vec<u32>)> = Vec::new();   // (prompt, domain, rep, ptoks)
    for (pname, dom, text) in prompt_set.iter() {
        if !domains.contains(&dom.as_str()) { continue; }
        let ptoks = if thinking.is_some() && dom == "math" {
            let msgs = vec![gb10_inference::tokenizer::ChatMessage {
                role: "user".to_string(), content: Some(text.to_string()), images: vec![], layout: vec![], unsupported_parts: vec![],
                tool_calls: None, tool_call_id: None, name: None, reasoning_content: None,
            }];
            tokenizer.apply_chat_template(&msgs, None, thinking.as_deref(), None, gb10_inference::tokenizer::ThinkingMode::On)
                .map(|s| tokenizer.encode(&s, false).expect("encode chat prompt"))
                .expect("chat template render")
        } else {
            tokenizer.encode(text, true).expect("encode prompt")
        };
        for r in 0..reps {
            jobs.push(SpecBenchJob {
                prompt: ptoks.clone(), max_new, temperature: temp, top_p, top_k,
                seed: 0x5EED_0000u64 ^ ((*dom.as_bytes().first().unwrap() as u64) << 32)
                    ^ (r as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                source: spec_source,
                domain: gb10_inference::batch::classify_domain(text),
            });
            job_meta.push((pname.to_string(), dom.to_string(), r, ptoks.clone()));
        }
    }
    println!("S5F matrix cell-group: source={} temp={} top_p={top_p} top_k={top_k} domains={:?} reps={} \
             max_new={} thinking={:?} ({} jobs)",
             spec_source.cli_name(), temp, domains, reps, max_new, thinking, jobs.len());
    // S5F3: the draft-parity step dump (dump-only; `--df2-step-dump <dir>`; CLI-1: the old env
    // alias is gone). Built BEFORE the scheduler — the dump is attached to the run (job_start/job_end
    // + the lane-step records).
    let dump_dir = parse_arg(args, "--df2-step-dump")
        .map(str::to_string);
    let mut step_dump = match &dump_dir {
        Some(dir) => {
            let d = gb10_inference::dflash2::stepdump::StepDump::new(dir)
                .expect("df2 step dump dir");
            println!("[df2-dump] step dump ON -> {dir}");
            Some(d)
        }
        None => None,
    };
    // P3(a) close (quad sweep 2026-08-23): prose-lane routing DEFAULT FLIPPED rq -> greedy-drafts,
    // TEMP-CONDITIONAL: greedy (temp-0) General requests take the greedy-draft lane, sampled
    // (temp>0) General keeps the real-q walk (df2_effective_src gates on the lane's own greedy
    // flag). Quad temp-0 sweep (both routings, 3 passes x 5 reps): prose tau +8.7/+7.8/+19.1/
    // +8.6% per cell (step-weighted +10.5%), step 61.3 -> 58.0 ms, code control bit-identical
    // (tau 5.5393 both arms). The blanket flip regressed chat_t1_off @T1.0 (37.4 -> 32.1 tok/s,
    // 1.010x -> 0.865x vs MTP) — the conditional keeps that cell on rq. `--df2-prose-lane rq`
    // restores the unconditional walk for every temp.
    // DF2_CARRY: keep DFlash2 in the draft seat across a prefix-cache hit (default OFF until
    // PLAN/DF2_CARRY_SPEC.md §7 is green). CLI flag per AGENTS §7; rides TpConfig (no side channel).
    let df2_carry = matches!(parse_arg(args, "--df2-carry").unwrap_or("off"),
                             "on" | "true" | "1" | "yes");
    let prose_lane_greedy = matches!(parse_arg(args, "--df2-prose-lane").unwrap_or("greedy-drafts"),
                                     "greedy-drafts" | "greedy" | "argmax");
    let mut scheduler = build_spec_scheduler(gpu, max_seq_len, eos, spec_source, mtp_depth,
                                             round, sink, prime, ds_round, ds_sink, ds_prime,
                                             step_dump);
    scheduler.set_prose_lane_greedy(prose_lane_greedy);
    scheduler.set_df2_carry(df2_carry);
    let dump_armed = dump_dir.is_some();
    let dump_tags: Vec<String> = job_meta.iter()
        .map(|(pname, dom, r, _)| format!("{pname}_{dom}_r{r}")).collect();
    let rt = tokio::runtime::Builder::new_current_thread().enable_all()
        .build().expect("runtime");
    let (streams, step_recs, walls) = rt.block_on(scheduler.run_spec_bench(jobs, &dump_tags));

    // Aggregate per (domain): τ over full steps (emitted == nacc+1 — the offline method's
    // full-block rule), spread (stdev), tok/s, step breakdown.
    #[derive(Default)]
    struct CellAgg {
        n: usize, tau_sum: f64, tau_sq: f64,
        tok: usize, wall: f64,
        round_ms: f64, verify_ms: f64, step_ms: f64,
        accepted: u64, offered: u64,
    }
    use std::collections::BTreeMap;
    let mut cells: BTreeMap<String, CellAgg> = BTreeMap::new();
    // S3R protocol-parity cuts: per-position acceptance by think-block class + position band.
    // The think markers + starts-inside come from the tokenizer's think_tags(); a step at
    // absolute position p is INSIDE <think> iff the emitted stream before p has more opens than
    // closes (plus the template's prime). Bands: [0,256), [256,1024), [1024,∞).
    struct ThinkCut { inside_n: u64, inside_acc: u64, inside_off: u64, outside_n: u64, outside_acc: u64, outside_off: u64 }
    let mut cuts: BTreeMap<String, ThinkCut> = BTreeMap::new();
    let mut bands: BTreeMap<String, [u64; 6]> = BTreeMap::new();  // per band: [n, acc, off] x2
    let mut think_ids: Option<(u32, u32, bool)> = None;
    if thinking.is_some() {
        let (open, close, starts_inside) = tokenizer.think_tags();
        let oi = tokenizer.encode(open, false).ok().and_then(|v| v.first().copied());
        let ci = tokenizer.encode(close, false).ok().and_then(|v| v.first().copied());
        if let (Some(oi), Some(ci)) = (oi, ci) { think_ids = Some((oi, ci, starts_inside)); }
    }
    for (k, ((stream, recs), wall)) in streams.iter().zip(step_recs.iter()).zip(walls.iter()).enumerate() {
        let (_, dom, _, prompt) = &job_meta[k];
        let c = cells.entry(dom.clone()).or_default();
        c.tok += stream.len();
        c.wall += *wall as f64;
        if let Some((oi, ci, _)) = think_ids {
            // think-state at each step's absolute position: count opens/closes over the PROMPT
            // (the chat template primes <think>) + the emitted stream up to that position.
            let prompt = prompt;
            let mut opens = prompt.iter().filter(|&&t| t == oi).count() as u64;
            let mut closes = prompt.iter().filter(|&&t| t == ci).count() as u64;
            let mut si = 0usize;
            for r in recs {
                while si < (r.pos as usize).saturating_sub(prompt.len()) {
                    if stream.get(si) == Some(&oi) { opens += 1; }
                    if stream.get(si) == Some(&ci) { closes += 1; }
                    si += 1;
                }
                let inside = opens > closes;
                let cut = cuts.entry(dom.clone()).or_insert(ThinkCut { inside_n: 0, inside_acc: 0, inside_off: 0, outside_n: 0, outside_acc: 0, outside_off: 0 });
                if inside {
                    cut.inside_n += 1; cut.inside_off += r.drafts as u64; cut.inside_acc += r.nacc as u64;
                } else {
                    cut.outside_n += 1; cut.outside_off += r.drafts as u64; cut.outside_acc += r.nacc as u64;
                }
            }
        }
        let b = bands.entry(dom.clone()).or_insert([0u64; 6]);
        for r in recs {
            // position-band marginal acceptance: [0,256), [256,1024), [1024,∞)
            let band = if r.pos < 256 { 0usize } else if r.pos < 1024 { 1usize } else { 2usize };
            b[band * 2] += r.drafts as u64; b[band * 2 + 1] += r.nacc as u64;
        }
        for r in recs {
            // full steps only (the offline full-block rule: emitted == nacc+1).
            if r.emitted as u32 == r.nacc + 1 {
                c.n += 1;
                let tau = (r.nacc + 1) as f64;
                c.tau_sum += tau; c.tau_sq += tau * tau;
                c.round_ms += r.round_ms as f64;
                c.verify_ms += r.verify_ms as f64;
                c.step_ms += r.step_ms as f64;
            }
            c.accepted += r.nacc as u64;
            c.offered += r.drafts as u64;
        }
    }
    let mut lines = Vec::new();
    for (dom, c) in &cells {
        // A "none" (plain-decode) cell has NO speculation steps — its τ is 1.0 BY DEFINITION
        // (one emitted token per step); the tok/s IS the sequential baseline.
        let tau = if c.n == 0 && spec_source == SpecSource::Plain { 1.0 } else { c.tau_sum / c.n.max(1) as f64 };
        let var = (c.tau_sq / c.n.max(1) as f64 - tau * tau).max(0.0);
        let spread = var.sqrt();
        let tps = c.tok as f64 / c.wall.max(1e-6);
        let acc = if c.offered > 0 { c.accepted as f64 / c.offered as f64 } else { f64::NAN };
        let line = format!(
            "  {dom:5}: τ {tau:.3} ± {spread:.3} (n={}) | {tps:.2} tok/s ({:.2} s, {} tok) | \
             acc {acc:.3} | step {:.1} ms = round {:.1} + verify {:.1}",
            c.n, c.wall, c.tok, c.step_ms / c.n.max(1) as f64,
            c.round_ms / c.n.max(1) as f64, c.verify_ms / c.n.max(1) as f64);
        println!("{line}");
        if let Some(cut) = cuts.get(dom) {
            let a = |n: u64, off: u64| -> f64 { if off > 0 { n as f64 / off as f64 } else { f64::NAN } };
            println!("    [think-cut] inside <think>: acc {:.3} ({}/{} drafts, {} steps) | outside: acc {:.3} ({}/{} drafts, {} steps)",
                     a(cut.inside_acc, cut.inside_off), cut.inside_acc, cut.inside_off, cut.inside_n,
                     a(cut.outside_acc, cut.outside_off), cut.outside_acc, cut.outside_off, cut.outside_n);
        }
        if let Some(b) = bands.get(dom) {
            let a2 = |off: u64, acc: u64| -> f64 { if off > 0 { acc as f64 / off as f64 } else { f64::NAN } };
            println!("    [band-cut] pos<256 acc {:.3} | 256-1024 acc {:.3} | >1024 acc {:.3}",
                     a2(b[0], b[1]), a2(b[2], b[3]), a2(b[4], b[5]));
        }
        lines.push(format!("{}|{:.6}|{:.6}|{}|{:.4}|{:.2}|{:.4}|{:.4}|{:.4}",
            dom, tau, spread, c.n, tps, c.wall, c.step_ms / c.n.max(1) as f64,
            c.round_ms / c.n.max(1) as f64, c.verify_ms / c.n.max(1) as f64));
    }
    let mut json = String::from("{\"source\":\"");
    json.push_str(spec_source.cli_name());
    json.push_str("\",\"temp\":");
    json.push_str(&temp.to_string());
    json.push_str(",\"thinking\":");
    json.push_str(if thinking.is_some() { "true" } else { "false" });
    json.push_str(",\"mtp_depth\":");
    json.push_str(&mtp_depth.to_string());
    json.push_str(",\"cells\":{");
    for (i, (dom, c)) in cells.iter().enumerate() {
        if i > 0 { json.push(','); }
        let tau = if c.n == 0 && spec_source == SpecSource::Plain { 1.0 } else { c.tau_sum / c.n.max(1) as f64 };
        let tps = c.tok as f64 / c.wall.max(1e-6);
        let a = |n: u64, off: u64| -> f64 { if off > 0 { n as f64 / off as f64 } else { f64::NAN } };
        let cut_json = cuts.get(dom).map(|cut| format!(
            ",\"think\":{{\"inside_acc\":{:.4},\"inside_off\":{},\"inside_n\":{},\
             \"outside_acc\":{:.4},\"outside_off\":{},\"outside_n\":{}}}",
            a(cut.inside_acc, cut.inside_off), cut.inside_off, cut.inside_n,
            a(cut.outside_acc, cut.outside_off), cut.outside_off, cut.outside_n))
            .unwrap_or_default();
        let band_json = bands.get(dom).map(|b| format!(
            ",\"bands\":[{{\"lo\":0,\"hi\":256,\"acc\":{:.4}}},{{\"lo\":256,\"hi\":1024,\"acc\":{:.4}}},{{\"lo\":1024,\"hi\":null,\"acc\":{:.4}}}]",
            a(b[1], b[0]), a(b[3], b[2]), a(b[5], b[4]))).unwrap_or_default();
        json.push_str(&format!(
            "\"{dom}\":{{\"tau\":{tau:.6},\"n\":{},\"tok_per_s\":{tps:.4},\"wall_s\":{:.4},\
             \"step_ms\":{:.4},\"round_ms\":{:.4},\"verify_ms\":{:.4},\"accepted\":{},\"offered\":{}{}{}}}",
            c.n, c.wall, c.step_ms / c.n.max(1) as f64, c.round_ms / c.n.max(1) as f64,
            c.verify_ms / c.n.max(1) as f64, c.accepted, c.offered, cut_json, band_json));
    }
    json.push_str("}}");
    std::fs::write(&out_path, &json).expect("write matrix out");
    println!("[matrix] wrote {out_path}");
    println!("RESULT: MATRIX_CELLS_DONE");
}
