//! `run_head` / `run_node`: cluster bring-up (TP attach, shard load, Ready handshake) around the serving loops.
//! Siblings: `loops.rs` (the loops), `flags.rs` (argv parsing); `mod.rs` has the map.

use super::*;

/// The head's serve: attach TP, load the shard, wait for the node, run HTTP + the one-at-a-time loop.
pub fn run_head(args: &[String], model_dir: &str, ctx: crate::tp::TpContext, mut streams: Vec<std::net::TcpStream>,
                watch: crate::cluster::NodeWatch) -> Result<()> {
    let world = ctx.world as usize;
    let pack = Arc::new(NaivePack::open(model_dir)?);
    let c = &pack.cfg;
    let port: u16 = crate::arg_value(args, "--port").and_then(|s| s.parse().ok()).unwrap_or(9000);
    let max_pos: usize = crate::arg_value(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(32768);
    ensure!(max_pos <= c.max_position_embeddings, "--max-seq-len {max_pos} exceeds the model's {}", c.max_position_embeddings);
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    super::gpu::load_module(&dev)?;
    super::gpu::set_pdl(pdl_flag(args));
    super::gpu::set_fuse(fuse_flag(args));
    let n_layers: usize = crate::arg_value(args, "--layers").and_then(|s| s.parse().ok()).unwrap_or(c.num_layers);
    let chunk: usize = crate::arg_value(args, "--prefill-chunk").and_then(|s| s.parse().ok()).unwrap_or(PREFILL_CHUNK);
    let spec = spec_cfg(args, model_dir);
    let tp = NaiveTp::attach(&dev, ctx, c.hidden_size, chunk.max(MAX_ROWS))?;
    let (lanes, asked) = lanes_asked(args);
    if asked > lanes {
        println!("[naive-serve] asked for {asked} requests at once (--max-batch {asked}): the Naive server shares a step among at most \
                  {MAX_LANES}, so it serves {lanes} at a time and further requests wait in the queue (none is rejected)");
    }
    let mut eng = Engine::load_spec_lanes(dev, &pack, Shard { rank: 0, world }, max_pos, n_layers, true, chunk, spec.as_ref(), lanes)?;
    crate::metrics::set_max_batch_requested(asked);
    if lanes > 1 {
        println!("[naive-serve] G1: up to {lanes} requests at once (--max-batch): one row per request in a shared step, {lanes} KV slots; {}",
                 if eng.per_lane_ctx() { "every lane keeps its own drafter context, so any request left alone speculates (--naive-lane-ctx on)" }
                 else { "one drafter context: only a request admitted into an idle engine speculates when alone (--naive-lane-ctx off, or no drafter)" });
    }
    eng.model.tp = Some(tp);
    let prefix = prefix_flag(args);
    if prefix {
        eng.enable_prefix()?;
        println!("[naive-serve] prefix reuse ON (--prefix-cache on): a prompt that repeats an earlier request's prompt up to a \
                  prefill chunk boundary resumes its prefill there (one conversation per KV slot: {lanes}; same chunks, same state)");
    }
    crate::metrics::NAIVE_PREFIX_ON.store(if eng.pfx_on() { 1 } else { 0 }, std::sync::atomic::Ordering::Relaxed);
    let pfx_on = eng.pfx_on();
    watch.disarm(&streams);
    for (i, s) in streams.iter_mut().enumerate() {
        match crate::tp_serve::recv_serving(s).context("node Ready")? {
            crate::tp_serve::ServingMsg::Ready => crate::dprintln!("[naive-serve] node rank {} READY", i + 1),
            other => bail!("expected Ready from node rank {}, got {other:?}", i + 1),
        }
    }
    let tok = Arc::new(QwenTokenizer::from_file(&format!("{}/tokenizer.json", pack.dir)).context("tokenizer")?);
    let stop_ids = tok.stop_token_ids(c.eos_token_id);
    let model_name = crate::server::resolve_model_name(crate::arg_value(args, "--model-name"), crate::arg_value(args, "--served-model-name"), model_dir);
    let (stx, mut srx) = tokio::sync::mpsc::unbounded_channel::<BatchRequest>();
    let state = crate::server::AppState {
        // the pack's generation_config: temperature 1.0, top_p 0.95, no top-k
        sampling_defaults: crate::server::SamplingDefaults { think: (1.0, 0.95, 0), no_think: (1.0, 0.95, 0) },
        scheduler: stx,
        tokenizer: tok.clone(),
        model_name: model_name.clone(),
        default_max_tokens: crate::arg_value(args, "--max-tokens").and_then(|s| s.parse().ok()).unwrap_or(usize::MAX),
        default_rep_penalty: 1.0,
        default_presence_penalty: 0.0,
        default_frequency_penalty: 0.0,
        reasoning_effort: None,
        thinking: crate::tokenizer::ThinkingMode::Auto,
        output_prompts: 0,
        max_seq_len: max_pos,
        decode_headroom: 0,
        // the message boundary (`ckpt_at`) places the Naive prefix snapshot
        prefix_cache: pfx_on,
        // no deferred schema start on this path
        schema_defer: false,
        // PR #30 / #31 ports: the Naive scheduler computes no log probabilities and applies no schema mask
        logprobs_ok: false,
        schema_ok: false,
        vision_tower: None,
        vision_gpu: None,
        vision_cpu: false,
        stop_ids: stop_ids.clone(),
        otel: None,
    };
    println!("[naive-serve] TP={world} serving: single stream, head-authoritative tokens over the control stream, \
              fp32 doorbell all-reduces (2 per layer), lm_head replicated (head only), graphed decode step with GPU sampling; stop_ids {stop_ids:?}");
    std::thread::spawn(move || {
        if let Err(e) = eng.model.dev.bind_to_thread() { eprintln!("\n*** FATAL: CUDA context bind: {e:?} ***\n"); std::process::exit(70); }
        // the node captures its graph after Ready; the head captures here, before the first request (its
        // graph holds the reduces, which are barriers: both ranks must capture at the same program point)
        if let Err(e) = eng.arm() { eprintln!("\n*** FATAL: decode graph capture failed: {e:#} ***\n"); std::process::exit(70); }
        if eng.lanes > 1 {
            if let Err(e) = serve_lanes(&mut eng, &mut srx, &stop_ids, &mut streams) {
                eprintln!("\n*** FATAL: the Naive TP head failed in the lanes loop: {e:#}. Exiting (70). ***\n");
                std::thread::sleep(std::time::Duration::from_millis(300));
                std::process::exit(70);
            }
        }
        while let Some(req) = srx.blocking_recv() {
            if let Err(e) = serve_one(&mut eng, req, &stop_ids, &mut streams) {
                eprintln!("\n*** FATAL: the Naive TP head failed a request: {e:#}. Exiting (70). ***\n");
                std::thread::sleep(std::time::Duration::from_millis(300));
                std::process::exit(70);
            }
        }
        for n in streams.iter_mut() { let _ = crate::cluster::send_json(n, &Ctl::Shutdown); }
        eprintln!("[naive-serve] request channel closed — session over");
        std::process::exit(0);
    });
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().context("tokio runtime")?;
    rt.block_on(async move {
        let app = crate::server::create_router(state);
        let listener = crate::server::http_listen(port).await;
        let url_l = listener.local_addr().map_or_else(|_| format!("port {port}"), |a| a.to_string());
        // report the RESOLVED spec config, not a literal. This line used to hardcode `plain`, so it
        // read `plain` even when `--spec-source dspark` had resolved a live drafter — a status
        // field that cannot be wrong about engagement is exactly the untrustworthy-receipt class
        // (an engaged-but-unlabelled path reads as dead, and a dead one reads as engaged).
        let spec_label: String = match &spec {
            None => "off".to_string(),
            Some(sc) => format!(
                "{}:block{}",
                crate::arg_value(args, "--spec-source").unwrap_or("dspark"),
                sc.block
            ),
        };
        crate::boot_report::serving(&crate::boot_report::ServingLine {
            family: "naive", model: &model_name, tp: world as u32, ctx: max_pos, batch: lanes, kv: "f16", prefix: if pfx_on { "on" } else { "off" }, mtp: "none",
            spec: &spec_label, port, url: if url_l.starts_with("port ") { format!("http://0.0.0.0:{port}") } else { format!("http://{url_l}") },
        });
        axum::serve(listener, app).tcp_nodelay(true).await.unwrap();
    });
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The node
// ---------------------------------------------------------------------------------------------

/// The node's serve: attach TP, load the shard, Ready, then mirror Admit / Tok forever.
pub fn run_node(model_dir: &str, max_pos: usize, n_layers: Option<usize>, chunk: usize, head_argv: &[String], ctx: crate::tp::TpContext, mut stream: std::net::TcpStream) -> Result<()> {
    let (rank, world) = (ctx.rank as usize, ctx.world as usize);
    let pack = Arc::new(NaivePack::open(model_dir)?);
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    super::gpu::load_module(&dev)?;
    super::gpu::set_pdl(pdl_flag(head_argv));
    super::gpu::set_fuse(fuse_flag(head_argv));
    let spec = spec_cfg(head_argv, model_dir); // the node has no drafter: only the block size matters here
    let tp = NaiveTp::attach(&dev, ctx, pack.cfg.hidden_size, chunk.max(MAX_ROWS))?;
    let mut eng = Engine::load_spec_lanes(dev, &pack, Shard { rank, world }, max_pos, n_layers.unwrap_or(pack.cfg.num_layers), false, chunk, spec.as_ref(), lanes_flag(head_argv))?;
    eng.model.tp = Some(tp);
    if prefix_flag(head_argv) { eng.enable_prefix()?; } // the head's decision rides every Admit; this rank keeps its own shard's snapshot
    let tok = QwenTokenizer::from_file(&format!("{}/tokenizer.json", pack.dir)).context("tokenizer")?;
    let stop_ids = tok.stop_token_ids(pack.cfg.eos_token_id);
    crate::tp_serve::send_serving(&mut stream, &crate::tp_serve::ServingMsg::Ready)?;
    eng.arm()?;
    crate::dprintln!("[naive-serve] node rank {rank}: Ready sent, graph armed — mirroring the head");
    loop {
        match crate::cluster::recv_json::<Ctl>(&mut stream).context("control stream (head gone?)")? {
            Ctl::Admit { prompt, max_new, seed, sp, min_new, ignore_eos, reuse, snap } => {
                // the mirror: the same prefill, the same sampler (every rank derives the same token and the
                // same stop decision from the exchanged values); the head's Tok messages are read one step
                // LATE as a lockstep check, so the TCP hop is never on this rank's critical path
                eng.begin_request(&sp, seed, &prompt)?;
                let row = eng.prefill_resume(0, &prompt, reuse, snap, true)?.context("node prefill without logits")?;
                let (mut t, mut cancel) = eng.dec.sample_row(&eng.model, row)?;
                let mut pos = prompt.len();
                let mut gen = 0usize;
                if eng.ver.is_some() {
                    // S8 mirror: the same rounds (drafts arrive in the exchange), the same stop rule; the head's
                    // Round message is checked one round late
                    let mut pending: Option<(Vec<u32>, bool)> = None;
                    let mut stop0 = false;
                    {
                        let is_stop = stop_ids.contains(&t) && !ignore_eos && gen + 1 > min_new;
                        if is_stop { stop0 = true; } else { gen += 1; if gen >= max_new { stop0 = true; } }
                        pending = Some((vec![t], stop0));
                        if !stop0 { eng.note_commit_all(&[t], false)?; }
                    }
                    let mut guard = SpecGuard::with_every(eng.guard.0, eng.guard.1, eng.guard.2);
                    while !stop0 {
                        let mut from_plain = false;
                        let (committed, a, c) = if guard.speculate() {
                            if eng.sharded {
                                // work item B: this rank runs its shard of the drafter in lockstep with the head (the
                                // exchanges inside); its drafts are checked against the head's in the slot exchange
                                let (d, cf) = eng.draft(t, pos)?;
                                let k = eng.pick_width(&cf);
                                eng.node_drafts = Some((d, k));
                            }
                            let r = eng.spec_round(None, t, pos)?;
                            guard.observe(r.1);
                            r
                        } else {
                            let (nt, c) = eng.plain_step_spec(t, pos)?;
                            from_plain = true;
                            (vec![nt], 0, c)
                        };
                        let toks_out: Vec<u32> = committed.clone();
                        let mut stop = false; let mut emitted = Vec::new();
                        for &tok in &toks_out {
                            let is_stop = stop_ids.contains(&tok) && !ignore_eos && gen + 1 > min_new;
                            if is_stop { stop = true; break; }
                            gen += 1; emitted.push(tok);
                            if gen >= max_new { stop = true; break; }
                        }
                        if c || cancel { stop = true; }
                        if let Some((pt, ps)) = pending.take() {
                            match crate::cluster::recv_json::<Ctl>(&mut stream)? {
                                Ctl::Round { toks: ht, stop: hs } => ensure!(ht == pt && hs == ps, "NAIVE TP LOCKSTEP BROKEN (round): node {pt:?}/{ps}, head {ht:?}/{hs}"),
                                Ctl::Shutdown => return Ok(()),
                                other => bail!("unexpected {other:?} inside a request"),
                            }
                        }
                        pending = Some((toks_out, stop));
                        eng.note_commit_all(&emitted, from_plain || eng.last_round_plain())?;
                        if stop { break; }
                        pos += a + 1;
                        t = *committed.last().unwrap();
                        cancel = c;
                        if pos + eng.block > eng.max_pos { break; }
                    }
                    if let Some((pt, ps)) = pending.take() {
                        match crate::cluster::recv_json::<Ctl>(&mut stream)? {
                            Ctl::Round { toks: ht, stop: hs } => ensure!(ht == pt && hs == ps, "NAIVE TP LOCKSTEP BROKEN at the end (round): node {pt:?}/{ps}, head {ht:?}/{hs}"),
                            Ctl::Shutdown => return Ok(()),
                            other => bail!("unexpected {other:?} at the end of a request"),
                        }
                    }
                    continue;
                }
                let mut pending: Option<(u32, bool)> = None; // the previous step's (token, stop) awaiting the head's word
                for i in 0..max_new {
                    let is_stop = stop_ids.contains(&t) && !ignore_eos && gen + 1 > min_new;
                    let last = is_stop || i + 1 == max_new || pos + 1 >= eng.max_pos || cancel;
                    if let Some((pt, ps)) = pending.take() {
                        match crate::cluster::recv_json::<Ctl>(&mut stream)? {
                            Ctl::Tok { t: ht, stop: hs } => ensure!(ht == pt && hs == ps, "NAIVE TP LOCKSTEP BROKEN: node token {pt} stop {ps}, head token {ht} stop {hs}"),
                            Ctl::Shutdown => return Ok(()),
                            other => bail!("unexpected {other:?} inside a request"),
                        }
                    }
                    pending = Some((t, last));
                    if !is_stop { gen += 1; }
                    if last { break; }
                    (t, cancel) = eng.decode_step(t, pos)?;
                    pos += 1;
                }
                if let Some((pt, ps)) = pending.take() {
                    match crate::cluster::recv_json::<Ctl>(&mut stream)? {
                        Ctl::Tok { t: ht, stop: hs } => ensure!(ht == pt && hs == ps, "NAIVE TP LOCKSTEP BROKEN at the end: node {pt}/{ps}, head {ht}/{hs}"),
                        Ctl::Shutdown => return Ok(()),
                        other => bail!("unexpected {other:?} at the end of a request"),
                    }
                }
            }
            m @ Ctl::LIter { .. } => return node_lanes(&mut eng, &mut stream, m),
            Ctl::Shutdown => return Ok(()),
            other => bail!("unexpected {other:?} between requests"),
        }
    }
}

