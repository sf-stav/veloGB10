//! Child of `dsv4_serve.rs` (so it may touch the private fields of the depth policy and serve state):
//! `dsv4_tp_dspark_serve`, the DSpark TP serve driver, and `dsv4_tp_serve_node_loop`, the node-side loop.

use super::*;

/// DSV4 TP=2 SPMD DSpark speculation serve (Phase 5). Both ranks load the trunk (TP-sharded) AND
/// the DSpark stages (replicated `dspark.safetensors`). The draft is deterministic → both ranks
/// compute identical drafts → SPMD preserved (no broadcast needed). Per step: forward the carry →
/// draft (rooted at carry_pos) → verify [r, d1..d5] → accept k → rollback/re-prime. On rejection
/// (k<block): full state restore + re-forward the committed prefix (correct; a fast selective
/// rollback is the follow-up). The draft ring is re-primed with the verify's real main_hidden
/// (§2.6 "re-prime with real verify hiddens") so it stays contiguous. Head prints the tokens +
/// acceptance + tok/s; the node mirrors silently.
pub(crate) fn dsv4_tp_dspark_serve(
    model_dir: &str,
    ctx_result: anyhow::Result<gb10_inference::tp::TpContext>,
    head_req: Option<(String, usize)>,
    mut ctl: Option<std::net::TcpStream>,
) -> anyhow::Result<()> {
    use gb10_inference::dsv4_dspark::Dsv4DSpark;
    use gb10_inference::dsv4_load;
    use gb10_inference::dsv4_model::Dsv4GpuModel;
    use cudarc::driver::CudaDevice;

    let is_head = head_req.is_some();
    let role = if is_head { "HEAD (rank 0/2)" } else { "NODE (rank 1/2)" };
    let mut ctx = ctx_result?;
    ctx.sanity()?;
    println!("{role} — DSV4 TP LINK UP (DSpark speculation)");

    let tok_path = format!("{}/tokenizer.json", model_dir.trim_end_matches('/'));
    let head_payload = match &head_req {
        Some((text, max_new)) => {
            let tok = QwenTokenizer::from_file(&tok_path)?;
            let ids = tok.encode(text, true)?;
            println!("{role} — prompt {text:?} → {} tokens", ids.len());
            Some((ids, *max_new))
        }
        None => None,
    };
    let (prompt, max_new, _) = ctx.broadcast_prompt(
        head_payload.as_ref().map(|(ids, m)| (ids.as_slice(), *m, 0)))?;
    let (rank, world, link) = ctx.into_parts();
    println!("{role} — SPMD DSpark decode: {max_new} new tokens, prompt {} tok", prompt.len());

    let cfg = dsv4_load::load_config(std::path::Path::new(model_dir))?;
    let prompt_len = prompt.len();
    let max_seq_len = gb10_inference::opts::var(gb10_inference::opt!("max-seq-len")).ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or((prompt_len + max_new + 256).max(2048));
    // DSpark warm/gates use forward_capture_main which is one-shot — keep the prompt ≤ chunk.
    let s_max = (prompt_len + 16).max(256).min(gb10_inference::dsv4_model::PREFILL_CHUNK);
    let dev = CudaDevice::new(0)?;
    let (mut m, load_secs) = dsv4_load_for_serve(&dev, std::path::Path::new(model_dir), &cfg,
                                                   max_seq_len, s_max, rank as usize, world as usize)?;
    println!("{role} — trunk shard load: {load_secs:.1}s");
    m.attach_tp(rank, world, link);

    // DSpark stages (replicated; both ranks load from their rank subdir's dspark_stage*.safetensors).
    let rank_dir = std::path::Path::new(model_dir).join(format!("rank{rank}"));
    let dspark_path = rank_dir.join("dspark_stage0.safetensors");
    anyhow::ensure!(dspark_path.exists(),
        "dspark_stage0.safetensors missing at {} — run --extract-dspark --bundle <b> --out <model-dir> first", rank_dir.display());
    let embed = m.embed.clone();
    let head_w = m.head.clone();
    let mut ds = Dsv4DSpark::load_from_artifact(&dev, &rank_dir, &cfg, max_seq_len, embed, head_w)?;
    println!("{role} — DSpark stages loaded (3 stages, replicated)");

    let prompt_i32: Vec<i32> = prompt.iter().map(|&v| v as i32).collect();
    // Prefill + capture main_hidden (TP — both ranks compute the same).
    let (logits_dev, main_hidden_pre) = m.forward_capture_main(&prompt_i32, 0)?;
    anyhow::ensure!(main_hidden_pre.is_some(), "forward_capture_main returned no main_hidden (need layers 40/41/42)");
    ds.warm(main_hidden_pre.as_ref().unwrap(), prompt_len)?;
    println!("{role} — prefill + DSpark warm done ({prompt_len} positions)");

    let eos = 1u32;
    let block = cfg.dspark_block_size;
    let vocab = cfg.vocab_size;
    let three_d = 3 * cfg.dim;
    // Item 3.3 adaptive draft depth: the depth policy (--dspark-depth N pins; None = auto, the
    // 128-step re-pick). Rides TpConfig so the zero-config node runs the SAME policy inputs.
    let depth_pin = gb10_inference::tp::tp_config()
        .map(|c| c.dspark_depth.map(|d| d as usize)).unwrap_or(None);
    let graph_armed = gb10_inference::opts::var(gb10_inference::opt!("dspark-graph")).is_ok();
    let pin_eff = if graph_armed && depth_pin.is_none() {
        // The drafter graph bakes width=block into its capture; adaptive width needs the eager
        // draft. Pin at block so the graph path keeps its current behavior (loud, once).
        eprintln!("{role} — [dspark-depth] --dspark-graph armed → adaptive depth disabled \
                   (the graph bakes width {block}); pin --dspark-depth N for the graph path");
        Some(block)
    } else {
        depth_pin
    };
    let mut policy = DsparkDepthPolicy::new(block, pin_eff, Vec::new());
    match pin_eff {
        Some(p) => println!("{role} — DSpark depth: PINNED at {} (fixed-width draft)", p),
        None => println!("{role} — DSpark depth: ADAPTIVE (r(D) calibrated at start, re-picked every {} steps)",
                         DSPARK_EVAL_WINDOW),
    }
    let mut generated: Vec<u32> = Vec::with_capacity(max_new);
    let mut logits_host: Vec<f32> = dev.dtoh_sync_copy(&logits_dev)?;
    let mut carry = dsv4_argmax(&logits_host) as u32;
    let mut carry_pos = prompt_len;
    let mut n_steps = 0u64;
    let mut n_offered = 0u64;
    let mut n_accepted = 0u64;
    let mut n_fast = 0u64;        // k == depth steps (no rollback)
    let mut reforward_toks = 0u64; // committed-prefix re-forwards on rejection
    // per-phase timing (r(d) measurement — the MTP step cost breakdown)
    let mut t_carryfwd = 0.0f64;  // forward_capture_main([carry]) — the plain-decode baseline cost
    let mut t_draft = 0.0f64;     // DSpark draft forward (3 stages + Markov)
    let mut t_verify = 0.0f64;    // trunk verify forward (depth+1 tokens)
    let mut t_reforward = 0.0f64; // rejection re-forward (committed prefix)
    // --dspark-phase-ms: synchronize around each phase so the timers measure GPU time,
    // not host launch time (the default async timers under-report ~0.1-0.3 ms phases).
    let phase_sync = gb10_inference::opts::var(gb10_inference::opt!("dspark-phase-ms")).is_ok();
    // --dspark-graph=1 arms the graphed drafter; a capture/classify failure falls back
    // to the eager draft ONCE (loud) and stays eager for the rest of the run.
    let mut dspark_graph_ok = true;
    let t0 = std::time::Instant::now();
    while generated.len() < max_new {
        // forward the carry → logits predict carry_pos+1, main_hidden@carry_pos.
        if phase_sync { unsafe { cudarc::driver::result::stream::synchronize(m.rt.stream.stream).ok(); cudarc::driver::result::stream::synchronize(ds.rt.stream.stream).ok(); } }
        let _tc = std::time::Instant::now();
        let (lc, mh_carry) = m.forward_capture_main(std::slice::from_ref(&(carry as i32)), carry_pos)?;
        if phase_sync { unsafe { cudarc::driver::result::stream::synchronize(m.rt.stream.stream).ok(); cudarc::driver::result::stream::synchronize(ds.rt.stream.stream).ok(); } }
        t_carryfwd += _tc.elapsed().as_secs_f64();
        logits_host = dev.dtoh_sync_copy(&lc)?;
        generated.push(carry);
        if carry == eos { println!("{role} — EOS (carry); stopping."); break; }
        if generated.len() >= max_new { break; }

        let r = dsv4_argmax(&logits_host) as i32; // greedy token at carry_pos+1 (ALWAYS committed)
        // Item 3.3: measure r(D) once on step 0 (adaptive only), on the real state; ship the
        // head's table over the retained control stream so both ranks drive ONE table (the node
        // runs the identical SPMD calibration forwards — the verify all-reduces are barriers —
        // and discards its own timings).
        if n_steps == 0 && pin_eff.is_none() {
            let rtab = dspark_calibrate_rd(&mut m, &mut ds, mh_carry.as_ref().unwrap(), r, carry_pos, block)?;
            // The calibration's draft calls accumulated into the phase split timers — reset so
            // the printed per-step chain/Markov split reflects the serving steps only.
            ds.t_chain = 0.0;
            ds.t_markov = 0.0;
            policy.set_r(rtab.clone());
            if is_head {
                let table: Vec<(u32, f32)> = rtab.iter().map(|&(d, r)| (d as u32, r)).collect();
                if let Some(mut s) = ctl.as_ref() {
                    gb10_inference::tp_serve::send_serving(&mut s, &gb10_inference::tp_serve::ServingMsg::DsparkRd { table })
                        .expect("ship DSpark r(D) table to node");
                    println!("{role} — DSpark r(D) table shipped to node");
                } else {
                    anyhow::bail!("[dspark-depth] adaptive depth needs the retained control stream (head)");
                }
            } else {
                let got = gb10_inference::tp_serve::recv_serving(ctl.as_mut()
                    .expect("adaptive depth needs the retained control stream (node)"))?;
                match got {
                    gb10_inference::tp_serve::ServingMsg::DsparkRd { table } => {
                        policy.set_r(table.iter().map(|&(d, r)| (d as usize, r)).collect());
                    }
                    other => anyhow::bail!("[dspark-depth] expected DsparkRd from head, got {other:?}"),
                }
            }
            println!("{role} — r(D) table (carry + draft(D) + verify(D+1)) / carry:");
            for &(d, r) in &rtab {
                println!("    D={}: {:.2}x decode — pays if yield > {:.2} tok/step", d, r, r);
            }
        }
        if phase_sync { unsafe { cudarc::driver::result::stream::synchronize(m.rt.stream.stream).ok(); cudarc::driver::result::stream::synchronize(ds.rt.stream.stream).ok(); } }
        let depth = policy.depth();
        let _td = std::time::Instant::now();
        let draft_out = if graph_armed && dspark_graph_ok {
            match ds.draft_graphed(mh_carry.as_ref().unwrap(), r, carry_pos) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("{role} — [dspark-graph] eager fallback at sp={carry_pos} (stays eager): {e:#}");
                    dspark_graph_ok = false;
                    ds.draft_n(mh_carry.as_ref().unwrap(), r, carry_pos, policy.depth())?
                }
            }
        } else {
            ds.draft_n(mh_carry.as_ref().unwrap(), r, carry_pos, depth)?
        };
        if phase_sync { unsafe { cudarc::driver::result::stream::synchronize(m.rt.stream.stream).ok(); cudarc::driver::result::stream::synchronize(ds.rt.stream.stream).ok(); } }
        t_draft += _td.elapsed().as_secs_f64();
        // verify [r, d1..dD] at carry_pos+1 (depth+1 rows)
        let verify_ids: Vec<i32> = std::iter::once(r).chain(draft_out.drafts.iter().copied()).collect();
        let snap = m.snapshot_verify_state()?;
        if phase_sync { unsafe { cudarc::driver::result::stream::synchronize(m.rt.stream.stream).ok(); cudarc::driver::result::stream::synchronize(ds.rt.stream.stream).ok(); } }
        let _tv = std::time::Instant::now();
        let (vl, vmh) = m.forward_verify_capture_main(&verify_ids, carry_pos + 1)?;
        if phase_sync { unsafe { cudarc::driver::result::stream::synchronize(m.rt.stream.stream).ok(); cudarc::driver::result::stream::synchronize(ds.rt.stream.stream).ok(); } }
        let vh: Vec<f32> = dev.dtoh_sync_copy(&vl)?;
        t_verify += _tv.elapsed().as_secs_f64();
        // acceptance: row j-1 (at carry_pos+j) predicts carry_pos+j+1; argmax vs draft_out.drafts[j-1].
        let mut k = 0usize;
        for j in 1..=depth {
            let row_arg = dsv4_argmax(&vh[(j - 1) * vocab..j * vocab]);
            if row_arg == draft_out.drafts[j - 1] as usize { k = j; } else { break; }
        }
        n_steps += 1;
        n_offered += depth as u64;
        n_accepted += k as u64;
        policy.record_step(depth, k, 1 + k as u64);
        policy.tick(n_steps);
        // r is the greedy token at carry_pos+1 — always correct (the trunk's own prediction,
        // confirmed by the verify's row-0 forward). Emit it, then the accepted drafts.
        generated.push(r as u32);
        if r as u32 == eos { println!("{role} — EOS (verify real token); stopping."); break; }
        // commit d1..dk
        let mut hit_eos = false;
        for j in 0..k {
            let t = draft_out.drafts[j] as u32;
            generated.push(t);
            if t == eos { hit_eos = true; break; }
        }
        if hit_eos { println!("{role} — EOS (accepted draft); stopping."); break; }
        if generated.len() >= max_new { break; }

        if k < depth {
            // rejection: rollback trunk + re-apply committed prefix SELECTIVELY (R4) — the
            // verify's ring writes for committed positions are valid, so only the KV ring rows,
            // compressor and indexer state need re-application from the verify's captured
            // activations (no full re-forward; final state identical, gates prove equivalence).
            let corrected = dsv4_argmax(&vh[k * vocab..(k + 1) * vocab]) as u32; // row k → carry_pos+k+2
            m.restore_verify_state(&snap)?;
            if phase_sync { unsafe { cudarc::driver::result::stream::synchronize(m.rt.stream.stream).ok(); cudarc::driver::result::stream::synchronize(ds.rt.stream.stream).ok(); } }
            let _tr = std::time::Instant::now();
            m.readvance_committed(carry_pos + 1, k + 1)?;
            if phase_sync { unsafe { cudarc::driver::result::stream::synchronize(m.rt.stream.stream).ok(); cudarc::driver::result::stream::synchronize(ds.rt.stream.stream).ok(); } }
            reforward_toks += (k + 1) as u64;
            t_reforward += _tr.elapsed().as_secs_f64();
            // re-prime the draft ring for the committed positions (real verify hiddens).
            ds.warm_range(vmh.as_ref().unwrap(), k + 1, carry_pos + 1)?;
            carry = corrected;
            carry_pos = carry_pos + k + 2;
        } else {
            // all accepted: bonus = row depth (the (depth+1)th verify row) → carry_pos + depth + 2.
            let bonus = dsv4_argmax(&vh[depth * vocab..(depth + 1) * vocab]) as u32;
            n_fast += 1;
            ds.warm_range(vmh.as_ref().unwrap(), depth + 1, carry_pos + 1)?;
            carry = bonus;
            carry_pos = carry_pos + depth + 2;
        }
        let _ = three_d;
    }
    let dt = t0.elapsed().as_secs_f64();
    let accept_rate = if n_offered > 0 { 100.0 * n_accepted as f64 / n_offered as f64 } else { 0.0 };
    // E[advance] = mean tokens committed per step (r is the carry, counted in the next step; the
    // emitted-per-step = accepted drafts + corrected/bonus). Use generated/step as the headline.
    let tok_per_step = if n_steps > 0 { generated.len() as f64 / n_steps as f64 } else { 0.0 };
    if is_head {
        let tok = QwenTokenizer::from_file(&tok_path)?;
        // trim a trailing EOS if present for decode display
        let display: Vec<u32> = generated.iter().copied().filter(|&t| t != eos).collect();
        let text = tok.decode(&display, true).unwrap_or_else(|e| format!("<decode err: {e}>"));
        let n = generated.len();
        println!("\n=== DSV4 TP=2 DSPARK ({n} tokens in {dt:.2}s, {:.1} tok/s) ===", n as f64 / dt.max(1e-9));
        println!("    prompt: {:?}", head_req.as_ref().unwrap().0);
        println!("    response: {text}");
        println!("    raw ids: {generated:?}");
        println!("--- DSpark stats ---");
        println!("    steps: {n_steps}, drafts offered: {n_offered}, accepted: {n_accepted} ({accept_rate:.1}%)");
        println!("    fast-path (k=depth) steps: {n_fast}/{n_steps}, re-forward tokens (rejection): {reforward_toks}");
        println!("    tokens/step (mean committed): {tok_per_step:.2}");
        let hz = policy.hazard_counts();
        print!("    yield curve (P(draft j accepted | earlier accepted)):");
        for (j, &(a, o)) in hz.iter().enumerate() {
            let p = if o > 0 { 100.0 * a as f64 / o as f64 } else { 0.0 };
            print!(" d{}={:.0}%({})", j + 1, p, o);
        }
        println!();
        let cum = policy.cumulative_yield();
        if !cum.is_empty() {
            print!("    cumulative yield (P(k>=j), the docs' reference format):");
            for (j, &p) in cum.iter().enumerate() { print!(" d{}=:{:.0}%", j + 1, p * 100.0); }
            println!();
        }
        if !policy.windows.is_empty() {
            let hist: Vec<String> = policy.windows.iter()
                .map(|&(s, d, x)| format!("@{}:d{}:{x:.2}x", s, d)).collect();
            println!("    adaptive depth history (step → depth → score): {}", hist.join(" "));
        } else {
            println!("    adaptive depth: none (pinned at {depth_pin:?} or run < 128 steps)");
        }
        // r(d): per-phase cost breakdown (the MTP step cost). decode = plain per-token cost;
        // DSpark step = carry-fwd (1× decode) + draft + verify + (rejection) re-forward.
        let n = n_steps.max(1) as f64;
        let t_decode = t_carryfwd / n;          // ≈ one plain decode
        let t_d = t_draft / n;
        let t_v = t_verify / n;
        let t_r = t_reforward / n;
        let rd_with = t_decode + t_d + t_v + t_r; // current (re-forward on rejection)
        let rd_fast = t_decode + t_d + t_v;       // projected (fast selective rollback — no re-forward)
        println!("--- r(d) cost breakdown (per step, {n_steps} steps{}) ---", if phase_sync { ", GPU-synced" } else { ", HOST-LAUNCH ONLY (set --dspark-phase-ms for GPU time)" });
        println!("    carry-fwd (1 decode): {t_decode:.4}s   draft: {t_d:.4}s   verify(mean {:.0} tok): {t_v:.4}s   re-forward: {t_r:.4}s",
                 n_offered as f64 / n_steps.max(1) as f64 + 1.0);
        if phase_sync {
            println!("    draft split: device chain {:.4}s   Markov tail {:.4}s", ds.t_chain / n, ds.t_markov / n);
        }
        println!("    r(d) current (w/ re-forward): {rd_with:.4}s → {:.2}x decode", rd_with / t_decode.max(1e-9));
        println!("    r(d) fast-rollback (projected): {rd_fast:.4}s → {:.2}x decode", rd_fast / t_decode.max(1e-9));
        println!("    E[advance]={tok_per_step:.2} tok/step → speedup current {:.2}x / fast-rollback projected {:.2}x",
            tok_per_step / (rd_with / t_decode.max(1e-9)).max(1e-9),
            tok_per_step / (rd_fast / t_decode.max(1e-9)).max(1e-9));
    }
    Ok(())
}

/// DSV4 TP=2 persistent NODE loop (server mode). Loads its shard IMMEDIATELY (concurrent with the
/// head's load), parks in `broadcast_prompt(None)` for the first prompt, attaches TP without
/// consuming the context, then loops: broadcast_prompt(None) → reset → forward → decode.
/// Item 3.4 (--server-dspark on, shipped via TpConfig): the per-request decode runs the SAME
/// DSpark SPMD sequence as the head ([`DsparkServeState`]) and the retained control stream is
/// kept for the head's once-per-process r(D) table.
pub(crate) fn dsv4_tp_serve_node_loop(
    model_dir: &str,
    head_ip: std::net::IpAddr,
    ctl: Option<std::net::TcpStream>,
) -> anyhow::Result<()> {
    use gb10_inference::dsv4_load;
    use cudarc::driver::CudaDevice;

    let server_dspark = gb10_inference::tp::tp_config().map(|c| c.server_dspark).unwrap_or(false);
    let mut ctl_mut = if !server_dspark {
        // Greedy-only server: the control stream has no consumer — close it now (the head's
        // run_head has already finished its sync traffic; keeping it open is harmless but the
        // node must not hold it when the head expects the session to end cleanly).
        drop(ctl);
        None
    } else {
        ctl
    };

    // P4: DSV4 N-way deferred (still TP=2).
    let mut ctx = gb10_inference::tp::TpContext::bring_up_node(head_ip, 1, 2)?;
    ctx.sanity()?;
    println!("NODE (rank 1/2) — DSV4 TP LINK UP (server mode{})", if server_dspark { ", DSpark speculation" } else { "" });

    // WORK 2: the shard loads IMMEDIATELY and concurrently with the head's load. The previous
    // load-on-first-request order was a workaround for a "QP goes stale during a ~100 s load
    // gap" theory that turned out to be three separate exchange bugs (no dead-peer detection
    // in net_exchange, the proxy stealing the exchange's send CQE, and a generation-clobber /
    // visibility race on the startup channel) — all fixed 2026-07-27, see DSV4_SESSION_REPORT.md.
    // The post-load park in `broadcast_prompt` below now survives arbitrary gaps (proven by the
    // 310-s idle survival test); a dead head during the park is detected ≤10 s by the 5-s
    // liveness probes (abort code 10) and the node supervisor re-arms. Bringing the load forward
    // means the pair is fully ready when the head starts listening, so first-request TTFT is
    // decode-only (no ~90 s load pause on the first curl).
    let cfg = dsv4_load::load_config(std::path::Path::new(model_dir))
        .map_err(|e| anyhow::anyhow!("load_config: {e:#}"))?;
    let dev = CudaDevice::new(0)?;
    // max_seq_len: env override (--max-seq-len) > shipped TpConfig (--max-seq-len from the head) > 4096.
    let max_seq_len = gb10_inference::opts::var(gb10_inference::opt!("max-seq-len")).ok()
        .and_then(|v| v.parse::<usize>().ok())
        .or_else(|| gb10_inference::tp::tp_config().map(|c| c.max_seq_len).filter(|&v| v > 0))
        .unwrap_or(4096);
    let s_max = gb10_inference::dsv4_model::PREFILL_CHUNK;
    let (m0, load_secs) = dsv4_load_for_serve(&dev, std::path::Path::new(model_dir), &cfg,
                                                   max_seq_len, s_max, 1, 2)?;
    println!("NODE (rank 1/2) — shard load: {load_secs:.1}s (max_seq_len={max_seq_len})");
    // Item 3.4: the DSpark state (trunk moves in; the greedy path keeps `m0`).
    let mut m: Option<gb10_inference::dsv4_model::Dsv4GpuModel> = Some(m0);
    let mut dspark_state: Option<DsparkServeState> = if server_dspark {
        Some(DsparkServeState::new(&dev, std::path::Path::new(model_dir), &cfg, max_seq_len,
                                   m.take().expect("fresh model"), 1, false)?)
    } else {
        None
    };

    // Park for the first prompt AFTER the load (the head loads concurrently during this gap;
    // the park is safe for arbitrary durations per the 2026-07-27 exchange fixes above).
    let (prompt, max_new, priming_len) = ctx.broadcast_prompt(None)?;
    let prompt_len = prompt.len();
    eprintln!("[dsv4-node] received first prompt ({prompt_len} tok, max_new={max_new}, priming={priming_len}) — decoding");

    // Attach TP without consuming the context. Kept AFTER the first broadcast_prompt — the head
    // attaches after its first broadcast too; both ranks must attach at the same protocol point
    // so the proxy's first epoch lines up on both sides. Do NOT attach before the park.
    let mm = if let Some(st) = dspark_state.as_mut() { &mut st.m } else { m.as_mut().unwrap() };
    let nbytes = mm.cfg.dim * 2;
    ctx.link.set_payload(nbytes, false)?;
    mm.rt.tp_ctx_dptr = ctx.link.ctx_device_ptr();
    mm.tp_rank = 1; // R3.2: enables the vocab-parallel maxloc head (rank 1 = second vocab half)
    let ctx_addr = ctx.link.ctx_addr();
    gb10_inference::net::spawn_proxy(ctx_addr, 19);
    eprintln!("[dsv4-tp] rank 1/2 — RDMA proxy up");

    // R2.3 prefix-cache (+ item 2.3 LRU): same snapshot/restore logic as the head. Both ranks
    // see the same prompt_ids + priming_len and do the same deterministic split → SPMD
    // preserved. Gated by the head's shipped --prefix-cache flag (default ON for dsv4).
    let prefix_cache_on = gb10_inference::tp::tp_config().map(|c| c.prefix_cache).unwrap_or(true);
    let mut prefix_cache = gb10_inference::dsv4_model::PrefixCache::new(8);
    let mut prompt = prompt;
    let mut max_new = max_new;
    let mut priming_len = priming_len;

    loop {
        let prompt_i32: Vec<i32> = prompt.iter().map(|&v| v as i32).collect();
        let eos = 1u32;
        // Item 3.4: DSpark mode — run the SAME SPMD draft/verify/rollback sequence as the head
        // (the verify all-reduces keep both ranks bit-for-bit in step; the head's r(D) table
        // arrives over the retained control stream at the first request's calibration point).
        if let Some(st) = dspark_state.as_mut() {
            let _gen = st.decode_request(&prompt_i32, priming_len, max_new, &mut prefix_cache,
                                         prefix_cache_on, ctl_mut.as_mut(), false, None)?;
            eprintln!("[dsv4-node] DSpark decode complete ({} tok prompt) — waiting for next request", prompt_i32.len());
            let (prompt_next, max_new_next, priming_next) = ctx.broadcast_prompt(None)?;
            prompt = prompt_next;
            max_new = max_new_next;
            priming_len = priming_next;
            eprintln!("[dsv4-node] received prompt ({} tok, max_new={max_new}, priming={priming_len}) — decoding", prompt.len());
            continue;
        }
        // Greedy path: shadow the Option with the owned model.
        let m = m.as_mut().expect("greedy node owns the model");
        // R3.2: same maxloc branch as the head (bitwise == full-vocab argmax, SPMD lockstep).
        let maxloc = m.rt.tp_ctx_dptr != 0 && m.tp_rank >= 0;
        let mut next: u32 = if maxloc {
            if prefix_cache_on {
                m.forward_prefix_cached_next(&prompt_i32, priming_len, &mut prefix_cache)?
                    .expect("maxloc head under TP")
            } else {
                m.reset_states()?;
                // chunked full prefill (forward_streams processes <= s_max rows per call —
                // the RUN-4 crash class; forward_prefill_chunked is bitwise one-shot).
                let (xt, tl) = if prompt_i32.len() > gb10_inference::dsv4_model::PREFILL_CHUNK {
                    m.forward_prefill_chunked(&prompt_i32, gb10_inference::dsv4_model::PREFILL_CHUNK)?
                } else {
                    (m.forward_streams(&prompt_i32, 0)?, prompt_i32.len())
                };
                m.forward_head_next(&xt, tl)?.expect("maxloc head under TP")
            }
        } else {
            let l = if prefix_cache_on {
                m.forward_prefix_cached(&prompt_i32, priming_len, &mut prefix_cache)?
            } else {
                m.reset_states()?;
                let (xt, tl) = if prompt_i32.len() > gb10_inference::dsv4_model::PREFILL_CHUNK {
                    m.forward_prefill_chunked(&prompt_i32, gb10_inference::dsv4_model::PREFILL_CHUNK)?
                } else {
                    (m.forward_streams(&prompt_i32, 0)?, prompt_i32.len())
                };
                let (_, l) = m.forward_head(&xt, tl)?;
                l
            };
            dsv4_argmax(&dev.dtoh_sync_copy(&l)?) as u32
        };
        let mut pos = prompt_i32.len();
        for step in 0..max_new {
            if next == eos { eprintln!("[dsv4-node] EOS at step {step}"); break; }
            next = if maxloc {
                m.forward_next(std::slice::from_ref(&(next as i32)), pos)?.expect("maxloc head under TP")
            } else {
                let l = m.forward(std::slice::from_ref(&(next as i32)), pos)?;
                dsv4_argmax(&dev.dtoh_sync_copy(&l)?) as u32
            };
            pos += 1;
        }
        eprintln!("[dsv4-node] decode complete ({} tokens) — waiting for next request", pos - prompt_i32.len());

        // Subsequent requests: receive the next prompt + priming_len.
        let (prompt_next, max_new_next, priming_next) = ctx.broadcast_prompt(None)?;
        prompt = prompt_next;
        max_new = max_new_next;
        priming_len = priming_next;
        eprintln!("[dsv4-node] received prompt ({} tok, max_new={max_new}, priming={priming_len}) — decoding", prompt.len());
    }
}
