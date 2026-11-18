//! `--probe-naive-*` entry points: TP slice, loop, prefill, shard-dir and speculative probes.
//! Siblings: `mod.rs` has the map; the probes drive `Engine` as the server does.

use super::*;

// ---------------------------------------------------------------------------------------------
// The TP identity probe: `--probe-naive-tp --model-dir <pack> --tp 2 --layers N --tokens T`
// ---------------------------------------------------------------------------------------------

/// Both ranks: N sharded layers + the transport, forward T fixed tokens; the head compares its
/// logits and residual taps with an UNSHARDED N-layer slice it loads alongside (small N).
pub fn run_probe(model_dir: &str, ctx: crate::tp::TpContext, n_layers: usize, tokens: usize, stream: Option<std::net::TcpStream>,
                 engine: bool, chunk: usize, argv: &[String], peers: Vec<std::net::TcpStream>) -> Result<()> {
    // FULL-DEPTH mode (n_layers == 0): the real prompt "The capital of France is" through every layer of the
    // sharded model (the serve configuration), rank 0 reporting the argmax of its vocabulary half —
    // " Paris" (12095) lives in rank 0's half at every world. No whole-model reference (it does not fit).
    if n_layers == 0 { return run_probe_full(model_dir, ctx, stream, engine, chunk, argv, peers); }
    let (rank, world) = (ctx.rank as usize, ctx.world as usize);
    let pack = Arc::new(NaivePack::open(model_dir)?);
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    super::gpu::load_module(&dev)?;
    let tp = NaiveTp::attach(&dev, ctx, pack.cfg.hidden_size, MAX_ROWS)?;
    let test_stream = dev.fork_default_stream()?;
    for n in [4096usize, 16 * 4096, 2048 * 4096] {
        let bad = tp.self_test(&dev, &test_stream, n)?;
        println!("[naive-tp] rank {rank}: transport self-test {n} floats: {bad} wrong");
        ensure!(bad == 0, "NAIVE_TP_FAIL: transport self-test");
    }
    // the reduce latency: 500 back-to-back 16 KB all-reduces (the decode shape), stream time per reduce
    {
        let buf = dev.alloc_zeros::<f32>(4096)?;
        dev.synchronize()?;
        for _ in 0..20 { tp.all_reduce(&test_stream, &buf, 4096)?; }
        unsafe { cudarc::driver::result::stream::synchronize(test_stream.stream) }?;
        let t0 = std::time::Instant::now();
        let n = 500;
        for _ in 0..n { tp.all_reduce(&test_stream, &buf, 4096)?; }
        unsafe { cudarc::driver::result::stream::synchronize(test_stream.stream) }?;
        println!("[naive-tp] rank {rank}: 16 KB all-reduce: {:.1} us each ({n} back to back)", t0.elapsed().as_secs_f64() / n as f64 * 1e6);
    }
    let t0 = std::time::Instant::now();
    let mut sh = GpuModel::load_slice_on(dev.clone(), &pack, n_layers, 4096, SWA_RING, 1, Shard { rank, world })?;
    sh.tp = Some(tp);
    println!("[naive-tp] rank {rank}/{world}: {n_layers} sharded layers loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let ids: Vec<u32> = (0..tokens).map(|i| (1000 + (i as u32 * 7919) % 100_000) as u32).collect();
    let pos: Vec<usize> = (0..tokens).collect();
    let _ctl = stream; // the node holds its control stream to the end (the head's watch reads a close as a failure)
    // round 1: the sharded forward on both ranks (the reduces are the barriers)
    let t1 = std::time::Instant::now();
    let (lg_sh, taps_sh) = sh.forward_rows(&ids, &pos)?;
    let fwd = t1.elapsed().as_secs_f64();
    println!("[naive-tp] rank {rank}: sharded forward {fwd:.3} s ({} reduces)", 2 * n_layers);
    // round 2: again — the second pass must be bit-identical to the first (transport determinism)
    let (lg_sh2, _) = sh.forward_rows(&ids, &pos)?;
    let same = lg_sh == lg_sh2;
    println!("[naive-tp] rank {rank}: second sharded pass bitwise identical to the first: {same}");
    if rank != 0 { ensure!(same, "NAIVE_TP_FAIL: rank {rank} not deterministic"); println!("NAIVE_TP_NODE_DONE"); return Ok(()); }
    // the unsharded reference on the head's GPU
    let r = GpuModel::load_slice_on(dev, &pack, n_layers, 4096, SWA_RING, 1, Shard::single())?;
    let (lg_r, taps_r) = r.forward_rows(&ids, &pos)?;
    let mut worst = 0f32;
    for l in 0..n_layers {
        let rel = super::cpu::rel_l2(&taps_sh[l], &taps_r[l]);
        worst = worst.max(rel);
        println!("  layer {l:2}: residual rel-L2 sharded vs whole {rel:.3e}");
    }
    // the sharded head holds rank 0's vocabulary slice: compare it with the same columns of the whole
    let (v, vl) = (pack.cfg.vocab_size, sh.vl);
    let mut lg_rs = Vec::with_capacity(tokens * vl);
    for t in 0..tokens { lg_rs.extend_from_slice(&lg_r[t * v..t * v + vl]); }
    let rel = super::cpu::rel_l2(&lg_sh, &lg_rs);
    let agree = (0..tokens).filter(|t| argmax(&lg_sh[t * vl..(t + 1) * vl]) == argmax(&lg_rs[t * vl..(t + 1) * vl])).count();
    println!("  logits (rank-0 vocabulary slice, {vl} of {v}) rel-L2 {rel:.3e}; argmax agreement {agree}/{tokens}; determinism {same}");
    // the sharded path differs from the whole-layer path only by fp32 summation order of the two
    // partials (and the f16 rounding of the halves' Hadamard outputs): 1e-3-class, not 1e-2
    if worst > 5e-3 || rel > 5e-3 || agree * 10 < tokens * 9 || !same { bail!("NAIVE_TP_FAIL"); }
    println!("NAIVE_TP_OK");
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The loopback TP gate: `--probe-naive-loop --model-dir <pack> [--layers N] [--tokens T]`
// ---------------------------------------------------------------------------------------------

/// Two shards on ONE GPU, two threads, the partials crossing a host channel: the sharded math
/// without a link. The head thread compares against the whole-layer slice.
pub fn run_loop_probe(model_dir: &str, n_layers: usize, tokens: usize, world: usize) -> Result<()> {
    ensure!(world == 2 || world == 4, "--tp 2 or 4");
    let pack = Arc::new(NaivePack::open(model_dir)?);
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    super::gpu::load_module(&dev)?;
    let mut tps = NaiveTp::loopback_group(&dev, pack.cfg.hidden_size, tokens.max(MAX_ROWS), world)?;
    let ids: Vec<u32> = (0..tokens).map(|i| (1000 + (i as u32 * 7919) % 100_000) as u32).collect();
    let pos: Vec<usize> = (0..tokens).collect();
    fn run(dev: Arc<CudaDevice>, pack: Arc<NaivePack>, ids: Vec<u32>, pos: Vec<usize>, n_layers: usize, rank: usize, world: usize, tp: NaiveTp)
           -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        dev.bind_to_thread()?;
        let mut m = GpuModel::load_slice_on(dev, &pack, n_layers, 4096, SWA_RING, 1, Shard { rank, world })?;
        m.tp = Some(tp);
        m.forward_rows(&ids, &pos)
    }
    let tp0 = tps.remove(0);
    let mut handles = Vec::new();
    for (i, tp) in tps.into_iter().enumerate() {
        let (d, p, iv, q) = (dev.clone(), pack.clone(), ids.clone(), pos.clone());
        handles.push(std::thread::spawn(move || run(d, p, iv, q, n_layers, i + 1, world, tp)));
    }
    let (lg0, taps0) = run(dev.clone(), pack.clone(), ids.clone(), pos.clone(), n_layers, 0, world, tp0)?;
    let mut all_same = true;
    for (i, h) in handles.into_iter().enumerate() {
        let (_lg, taps) = h.join().map_err(|_| anyhow::anyhow!("rank {} thread panicked", i + 1))??;
        all_same &= taps.last() == taps0.last();
    }
    println!("[naive-loop] world {world}: every rank's final residual bitwise identical to rank 0's: {all_same}");
    let r = GpuModel::load_slice_on(dev.clone(), &pack, n_layers, 4096, SWA_RING, 1, Shard::single())?;
    let (lg_r, taps_r) = r.forward_rows(&ids, &pos)?;
    let mut worst = 0f32;
    for l in 0..n_layers {
        let rel = super::cpu::rel_l2(&taps0[l], &taps_r[l]);
        worst = worst.max(rel);
        println!("  layer {l:2}: residual rel-L2 sharded(loopback) vs whole {rel:.3e}");
    }
    let (v, vl) = (pack.cfg.vocab_size, lg0.len() / tokens);
    let mut lg_rs = Vec::with_capacity(tokens * vl);
    for t in 0..tokens { lg_rs.extend_from_slice(&lg_r[t * v..t * v + vl]); }
    let rel = super::cpu::rel_l2(&lg0, &lg_rs);
    let agree = (0..tokens).filter(|t| argmax(&lg0[t * vl..(t + 1) * vl]) == argmax(&lg_rs[t * vl..(t + 1) * vl])).count();
    println!("  logits (rank-0 vocabulary slice, {vl} of {v}) rel-L2 {rel:.3e}; argmax agreement {agree}/{tokens}");
    if worst > 5e-3 || rel > 5e-3 || agree * 10 < tokens * 9 { bail!("NAIVE_LOOP_FAIL"); }
    println!("NAIVE_LOOP_OK");
    Ok(())
}

/// `--probe-naive-prefill --model-dir <pack> [--layers N] [--tokens T]`: the wide prefill path (one chunk of T
/// rows) against the 16-row chunked path on the same tokens: the residual after every layer and the last
/// row's logits must be BITWISE identical (every kernel in the wide path is row-invariant by construction).
pub fn run_prefill_probe(model_dir: &str, n_layers: usize, tokens: usize) -> Result<()> {
    let pack = Arc::new(NaivePack::open(model_dir)?);
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    super::gpu::load_module(&dev)?;
    let model = GpuModel::load_slice_on(dev.clone(), &pack, n_layers, 8192, SWA_RING, 1, Shard::single())?;
    let ids: Vec<u32> = (0..tokens).map(|i| (1000 + (i as u32 * 7919) % 100_000) as u32).collect();
    let (h, vl) = (model.cfg.hidden_size, model.vl);
    // wide: one chunk
    let mut ws = Ws::new(&dev, ws_bytes(&model.cfg, tokens, 8192, n_layers))?;
    let ids_d = dev.htod_sync_copy(&ids.iter().map(|x| *x as i32).collect::<Vec<i32>>())?;
    let pos_d = dev.htod_sync_copy(&(0..tokens as i32).collect::<Vec<i32>>())?;
    let mut taps_w = Vec::new();
    let t0 = std::time::Instant::now();
    let lg_w = model.step_kernels(&mut ws, dptr(&ids_d), dptr(&pos_d), tokens, true, Some(&mut taps_w))?.unwrap();
    model.sync()?;
    let dt_w = t0.elapsed().as_secs_f64();
    let taps_w: Vec<Vec<f32>> = taps_w.iter().map(|t| super::model::dtoh::<f32>(*t, tokens * h)).collect::<Result<_>>()?;
    let lg_w: Vec<u16> = super::model::dtoh::<u16>(lg_w, vl)?;
    // chunked: 16 rows at a time (the caches are rewritten position by position identically)
    let mut taps_c: Vec<Vec<f32>> = vec![vec![0f32; tokens * h]; n_layers];
    let mut lg_c = Vec::new();
    let t1 = std::time::Instant::now();
    let mut i = 0;
    while i < tokens {
        let m = (tokens - i).min(MAX_ROWS);
        let mut ws2 = Ws::new(&dev, ws_bytes(&model.cfg, m, 8192, n_layers))?;
        let id2 = dev.htod_sync_copy(&ids[i..i + m].iter().map(|x| *x as i32).collect::<Vec<i32>>())?;
        let po2 = dev.htod_sync_copy(&(i as i32..(i + m) as i32).collect::<Vec<i32>>())?;
        let mut tp2 = Vec::new();
        let lg = model.step_kernels(&mut ws2, dptr(&id2), dptr(&po2), m, i + m == tokens, Some(&mut tp2))?;
        model.sync()?;
        for (l, t) in tp2.iter().enumerate() {
            let v = super::model::dtoh::<f32>(*t, m * h)?;
            taps_c[l][i * h..(i + m) * h].copy_from_slice(&v);
        }
        if let Some(p) = lg { lg_c = super::model::dtoh::<u16>(p + ((m - 1) * vl * 2) as u64, vl)?; }
        i += m;
    }
    let dt_c = t1.elapsed().as_secs_f64();
    // the decode path sums split-K fp32 partials, the wide kernel accumulates K in one pass: the two are
    // not bitwise (2e-4-class rounding differences, the f16 noise floor); the gate is the tolerance + argmax
    let mut ok = true;
    // the f16 activation drift grows ~1.5x per layer against any differently-rounded path (the slice gates
    // show the same trend); the bound follows it
    for l in 0..n_layers {
        let rel = super::cpu::rel_l2(&taps_w[l], &taps_c[l]);
        let bound = 1e-3 * 1.5f32.powi(l as i32 + 1);
        println!("  layer {l:2}: residual wide vs chunked-16: rel-L2 {rel:.2e} (bound {bound:.1e})");
        if rel > bound { ok = false; }
    }
    let f = |v: &[u16]| v.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect::<Vec<f32>>();
    let (lw, lc) = (f(&lg_w), f(&lg_c));
    let rel = super::cpu::rel_l2(&lw, &lc);
    let same_argmax = argmax(&lw) == argmax(&lc);
    println!("  last-row logits: rel-L2 {rel:.2e}, argmax equal {same_argmax}");
    println!("  wide {tokens} rows: {:.1} ms ({:.0} tok/s, {n_layers} layers); chunked-16: {:.1} ms ({:.0} tok/s)", dt_w * 1e3, tokens as f64 / dt_w, dt_c * 1e3, tokens as f64 / dt_c);
    if rel > 1e-2 || !same_argmax { ok = false; }
    if !ok { bail!("NAIVE_PREFILL_FAIL"); }
    println!("NAIVE_PREFILL_OK");
    Ok(())
}

/// `--probe-naive-shard-dir --model-dir <full pack> --shard-dir <shipped shard dir> --tp W --layers N`: the shipped
/// shard must load into the SAME rank model as the full pack sliced at load: rank-local partial forward (no
/// transport) on both, residual taps and local logits BITWISE identical.
pub fn run_shard_dir_probe(full_dir: &str, shard_dir: &str, n_layers: usize, tokens: usize) -> Result<()> {
    let full = Arc::new(NaivePack::open(full_dir)?);
    let shard_pack = Arc::new(NaivePack::open(shard_dir)?);
    let (world, rank) = shard_pack.shard.context("the shard dir has no gb10_shard.json")?;
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    super::gpu::load_module(&dev)?;
    let ids: Vec<u32> = (0..tokens).map(|i| (1000 + (i as u32 * 7919) % 100_000) as u32).collect();
    let pos: Vec<usize> = (0..tokens).collect();
    let sh = Shard { rank, world };
    let a = GpuModel::load_slice_on(dev.clone(), &full, n_layers, 4096, SWA_RING, 1, sh)?;
    let (lg_a, taps_a) = a.forward_rows(&ids, &pos)?;
    drop(a);
    let b = GpuModel::load_slice_on(dev.clone(), &shard_pack, n_layers, 4096, SWA_RING, 1, sh)?;
    let (lg_b, taps_b) = b.forward_rows(&ids, &pos)?;
    let mut ok = true;
    for l in 0..n_layers {
        let d = taps_a[l].iter().zip(&taps_b[l]).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        println!("  layer {l:2}: residual (rank {rank} partial path) full-pack vs shipped shard: {d} of {} values differ", taps_a[l].len());
        if d > 0 { ok = false; }
    }
    let d = lg_a.iter().zip(&lg_b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    println!("  local logits: {d} of {} differ", lg_a.len());
    if d > 0 { ok = false; }
    if !ok { bail!("NAIVE_SHARD_DIR_FAIL"); }
    println!("NAIVE_SHARD_DIR_OK");
    Ok(())
}

/// The probe's load barrier over the control streams: every node says "loaded", the head waits for all of them, then
/// says "go". The ranks load at very different speeds (the head slices the whole pack, a node reads its shipped shard),
/// and the first reduce of a rank that is >10 s ahead trips the transport watchdog, which aborts the link: every later
/// reduce then returns a partial sum without an error (2026-10-21: the NLL probe's 12.7 nats/token).
fn probe_barrier(stream: &mut Option<std::net::TcpStream>, peers: &mut [std::net::TcpStream]) -> Result<()> {
    match stream.as_mut() {
        Some(s) => { crate::cluster::send_json(s, &1u8)?; let _: u8 = crate::cluster::recv_json(s).context("probe barrier: head's go")?; }
        None => {
            for p in peers.iter_mut() { let _: u8 = crate::cluster::recv_json(p).context("probe barrier: a node's loaded")?; }
            for p in peers.iter_mut() { crate::cluster::send_json(p, &2u8)?; }
        }
    }
    Ok(())
}

fn run_probe_full(model_dir: &str, ctx: crate::tp::TpContext, mut stream: Option<std::net::TcpStream>, engine: bool, chunk: usize, argv: &[String],
                  mut peers: Vec<std::net::TcpStream>) -> Result<()> {
    let (rank, world) = (ctx.rank as usize, ctx.world as usize);
    let pack = Arc::new(NaivePack::open(model_dir)?);
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    super::gpu::load_module(&dev)?;
    // ENGINE mode (--probe-naive-tp-engine): the serve's own path — Engine prefill (chunked workspace) + the
    // graphed decode step with the GPU sampler — greedy tokens printed on every rank
    if engine {
        // the head's argv decides speculation (`--spec-source`, `--spec-block`, `--draft-dir` on the head)
        let spec = spec_cfg(argv, model_dir).filter(|_| rank == 0 || true);
        let tp = NaiveTp::attach(&dev, ctx, pack.cfg.hidden_size, chunk.max(MAX_ROWS))?;
        let mut eng = Engine::load_spec(dev.clone(), &pack, Shard { rank, world }, 32768, pack.cfg.num_layers, rank == 0, chunk, spec.as_ref())?;
        eng.model.tp = Some(tp);
        eng.arm()?;
        let _ctl = stream;
        let ids: Vec<u32> = vec![785, 6722, 315, 9625, 374];
        eng.begin_request(&SampleParams::GREEDY, 1, &ids)?;
        let row = eng.prefill(&ids, 0, true)?.unwrap();
        let (mut t, _) = eng.dec.sample_row(&eng.model, row)?;
        let mut out = vec![t];
        let mut pos = ids.len();
        if eng.ver.is_some() {
            // speculative rounds (the head drafts; the node reads the exchange): greedy must match the plain path
            for r in 0..4 {
                let drafts: Option<(Vec<u32>, usize)> = if rank == 0 { let (d, c) = eng.draft(t, pos)?; let k = eng.pick_width(&c); Some((d, k)) } else { None };
                let (committed, a, _) = eng.spec_round(drafts.as_ref().map(|(d, k)| (d.as_slice(), *k)), t, pos).with_context(|| format!("spec round {r}"))?;
                println!("[naive-tp] rank {rank}: round {r}: drafts {:?} accepted {a} committed {:?}", drafts, committed);
                out.extend_from_slice(&committed); pos += a + 1; t = *committed.last().unwrap();
            }
        } else {
            for _ in 0..5 { let (nt, _) = eng.decode_step(t, pos)?; out.push(nt); t = nt; pos += 1; }
        }
        println!("[naive-tp] rank {rank}: ENGINE greedy tokens {out:?} (want [12095, 382, 59604, 374, ...] = ' Paris.\n\nParis is')");
        if rank == 0 { ensure!(out[0] == 12095, "NAIVE_TP_FAIL: engine path"); println!("NAIVE_TP_FULL_OK"); } else { println!("NAIVE_TP_NODE_DONE"); }
        return Ok(());
    }
    // Teacher-forced NLL on the SHARDED model (2026-10-20, the expert-TP quality gate): `--nll-file F --nll-out P`
    // (F and P paths on every rank). It runs the SERVED engine's prefill in MAX_ROWS-row chunks (a chunk that narrow
    // returns every row's logits): each rank writes, per row i, its vocabulary slice's (max, sum exp(l - max)) and the
    // target logit of token i+1 when the target is in its slice; the per-row NLL is combined offline (logsumexp over the
    // ranks' slices minus the target logit). (The first version forwarded all rows through `GpuModel::forward_rows`,
    // a probe-only path: under TP at 2,047 rows it gave ~12.7 nats per token on both expert deals, i.e. garbage.)
    if let (Some(f), Some(out)) = (crate::arg_value(argv, "--nll-file"), crate::arg_value(argv, "--nll-out")) {
        let text = std::fs::read_to_string(f).with_context(|| format!("--nll-file {f}"))?;
        let tok = QwenTokenizer::from_file(&format!("{}/tokenizer.json", pack.dir)).context("tokenizer")?;
        let mut ids = tok.encode(&text, false)?;
        let n_max: usize = crate::arg_value(argv, "--nll-tokens").and_then(|s| s.parse().ok()).unwrap_or(1024);
        ids.truncate(n_max.min(4000));
        ensure!(ids.len() >= 2, "--nll-file: {} tokens", ids.len());
        // `--nll-rows R` (1..=16, default 1): rows per prefill call (1 = every row is the LAST row of its call, the
        // layout the serve reads; wider calls read a multi-row logits block the serve never reads row by row)
        let nll_rows: usize = crate::arg_value(argv, "--nll-rows").and_then(|s| s.parse().ok()).unwrap_or(1usize).clamp(1, MAX_ROWS);
        let tp = NaiveTp::attach(&dev, ctx, pack.cfg.hidden_size, MAX_ROWS)?;
        let mut eng = Engine::load(dev.clone(), &pack, Shard { rank, world }, 4096, pack.cfg.num_layers, rank == 0, MAX_ROWS)?;
        eng.model.tp = Some(tp);
        probe_barrier(&mut stream, &mut peers)?;
        let _ctl = stream;
        let (vl, v0) = (eng.model.vl, eng.model.v0);
        let mut lines = String::new();
        let mut i0 = 0usize;
        while i0 < ids.len() - 1 {
            let m = (ids.len() - 1 - i0).min(nll_rows);
            let last = eng.prefill_feed(&ids[i0..i0 + m], i0, true, false)?.context("NLL chunk without logits")?;
            eng.model.sync()?;
            let base = last - ((m - 1) * vl * 2) as u64;
            let lg: Vec<f32> = super::model::dtoh::<u16>(base, m * vl)?.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect();
            for r in 0..m {
                let i = i0 + r;
                let row = &lg[r * vl..(r + 1) * vl];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let se: f64 = row.iter().map(|x| ((*x as f64) - mx).exp()).sum();
                let t = ids[i + 1] as usize;
                let tl = if t >= v0 && t < v0 + vl { format!("{:.9}", row[t - v0]) } else { "null".into() };
                lines += &format!("{{\"i\":{i},\"m\":{mx:.9},\"s\":{se:.9},\"t\":{tl}}}\n");
            }
            i0 += m;
        }
        if let Some(tp) = &eng.model.tp { tp.check_link("at the end of the NLL pass")?; }
        let path = format!("{out}.rank{rank}.jsonl");
        std::fs::write(&path, lines)?;
        println!("[naive-tp] rank {rank}: NLL partials of {} rows (engine prefill) written to {path} ({nll_rows} row(s) per call)", ids.len() - 1);
        println!("{}", if rank == 0 { "NAIVE_TP_NLL_OK" } else { "NAIVE_TP_NODE_DONE" });
        return Ok(());
    }
    let tp = NaiveTp::attach(&dev, ctx, pack.cfg.hidden_size, MAX_ROWS)?;
    let t0 = std::time::Instant::now();
    let mut sh = GpuModel::load_slice_on(dev.clone(), &pack, pack.cfg.num_layers, 4096, SWA_RING, 1, Shard { rank, world })?;
    sh.tp = Some(tp);
    probe_barrier(&mut stream, &mut peers)?;
    println!("[naive-tp] rank {rank}/{world}: {} sharded layers loaded in {:.1} s", pack.cfg.num_layers, t0.elapsed().as_secs_f64());
    let _ctl = stream;
    let ids: Vec<u32> = vec![785, 6722, 315, 9625, 374];
    let pos: Vec<usize> = (0..ids.len()).collect();
    let (lg, taps) = sh.forward_rows(&ids, &pos)?;
    let vl = sh.vl;
    let last = &lg[(ids.len() - 1) * vl..ids.len() * vl];
    let mut idx: Vec<usize> = (0..vl).collect();
    idx.sort_by(|a, b| last[*b].partial_cmp(&last[*a]).unwrap_or(std::cmp::Ordering::Equal));
    let rms = |v: &[f32]| (v.iter().map(|a| (*a as f64) * (*a as f64)).sum::<f64>() / v.len() as f64).sqrt();
    println!("[naive-tp] rank {rank}: residual rms after layer 0 {:.4}, 20 {:.4}, 47 {:.4}", rms(&taps[0]), rms(&taps[20]), rms(&taps[taps.len() - 1]));
    println!("[naive-tp] rank {rank}: top-5 of the local vocabulary half [{}..{}): {:?}", sh.v0, sh.v0 + vl,
             idx[..5].iter().map(|i| (sh.v0 + i, last[*i])).collect::<Vec<_>>());
    if rank == 0 {
        ensure!(sh.v0 + idx[0] == 12095, "NAIVE_TP_FAIL: rank 0 argmax {} is not ' Paris' (12095)", sh.v0 + idx[0]);
        println!("NAIVE_TP_FULL_OK");
    } else { println!("NAIVE_TP_NODE_DONE"); }
    Ok(())
}

/// `--probe-naive-spec --model-dir <pack> --draft-dir <dir> [--layers N] [--spec-block B] [--tokens T] [--spec-mode M] [--tp W]`:
/// the speculative engine path (prefill with taps, rounds, block verify, match acceptance) must produce EXACTLY
/// the plain greedy continuation. Modes: `drafter` (the DSpark drafter), `oracle` (drafts = the plain continuation:
/// every row of the verify block is accepted and checked every round), `miss` (oracle drafts with row (round mod n)
/// wrong: every partial-acceptance path), `mixed` (oracle, with a plain step between rounds: the guard's shape).
/// `--tp 2|4` runs W ranks as threads on ONE GPU over the loopback transport (eager kernels, no graph).
pub fn run_spec_probe(model_dir: &str, draft_dir: &str, n_layers: usize, block: usize, tokens: usize, mode: &str, world: usize, sampler: &str) -> Result<()> {
    let pack = Arc::new(NaivePack::open(model_dir)?);
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    super::gpu::load_module(&dev)?;
    let prompt: Vec<u32> = vec![785, 6722, 315, 9625, 374];
    let n = block - 1;
    // plain greedy (one GPU, whole model slice): the reference, tokens + block extra for the oracle drafts
    let mut plain = Engine::load(dev.clone(), &pack, Shard::single(), 4096, n_layers, true, 64)?;
    plain.arm()?;
    plain.begin_request(&SampleParams::GREEDY, 1, &prompt)?;
    let row = plain.prefill(&prompt, 0, true)?.unwrap();
    let (mut t, _) = plain.dec.sample_row(&plain.model, row)?;
    let mut ref_toks = vec![t];
    let mut pos = prompt.len();
    while ref_toks.len() < tokens + block + 1 { let (nt, _) = plain.decode_step(t, pos)?; ref_toks.push(nt); t = nt; pos += 1; }
    drop(plain);
    let ref_toks = Arc::new(ref_toks);
    let sc = SpecCfg { draft_dir: Some(draft_dir.to_string()), block, conf: if mode == "drafter" || mode == "sampled" || mode == "chi" { SPEC_CONF } else { 0.0 }, guard_off: SPEC_GUARD_OFF, guard_on: SPEC_GUARD_ON, probe_every: SPEC_PROBE_EVERY, draft_temp: SPEC_DRAFT_TEMP, drafter: SPEC_DRAFTER.to_string(), sampler: sampler.to_string(), lane_ctx: true };
    // ---- gate (b) of work item A: distribution exactness, two-sample and reference-free -----------
    // The plain step and the FIRST token of a speculative round are both p's draw for the same
    // position: they differ only in the RNG stream and in the accept/residual machinery. So a
    // two-sample chi-square over the 50 most frequent ids plus an "other" bin rejects iff the spec
    // path is not drawing from p. A plain-vs-plain pair on disjoint seed ranges is run as the
    // control: it must NOT reject, and it prices the test's own false-positive rate.
    if mode == "chi" {
        fn gammq(a: f64, x: f64) -> f64 {
            // regularised upper incomplete gamma Q(a, x): series below a+1, continued fraction above
            if x <= 0.0 { return 1.0; }
            let gln = libm_lgamma(a);
            if x < a + 1.0 {
                let (mut ap, mut sum, mut del) = (a, 1.0 / a, 1.0 / a);
                for _ in 0..1000 { ap += 1.0; del *= x / ap; sum += del; if del.abs() < sum.abs() * 1e-15 { break; } }
                1.0 - sum * (-x + a * x.ln() - gln).exp()
            } else {
                let (tiny, mut b, mut c, mut d) = (1e-300f64, x + 1.0 - a, 1.0 / 1e-300f64, 1.0 / (x + 1.0 - a));
                let mut h = d;
                for i in 1..1000 {
                    let an = -(i as f64) * (i as f64 - a);
                    b += 2.0;
                    d = an * d + b; if d.abs() < tiny { d = tiny; }
                    c = b + an / c; if c.abs() < tiny { c = tiny; }
                    d = 1.0 / d;
                    let del = d * c;
                    h *= del;
                    if (del - 1.0).abs() < 1e-15 { break; }
                }
                (-x + a * x.ln() - gln).exp() * h
            }
        }
        fn libm_lgamma(x: f64) -> f64 {
            // Lanczos (g=7, n=9): enough for df/2 in [0.5, 60]
            const C: [f64; 9] = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
                                 -176.61502916214059, 12.507343278686905, -0.13857109526572012,
                                 9.9843695780195716e-6, 1.5056327351493116e-7];
            if x < 0.5 { return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - libm_lgamma(1.0 - x); }
            let x = x - 1.0;
            let mut a = C[0];
            let t = x + 7.5;
            for (i, c) in C.iter().enumerate().skip(1) { a += c / (x + i as f64); }
            0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
        }
        /// two-sample chi-square over the `top` most frequent ids (union) plus one "other" bin
        fn chi2_two(a: &[u32], b: &[u32], top: usize) -> (f64, usize, Vec<(u32, usize, usize)>) {
            use std::collections::HashMap;
            let mut ca: HashMap<u32, usize> = HashMap::new();
            let mut cb: HashMap<u32, usize> = HashMap::new();
            for &t in a { *ca.entry(t).or_insert(0) += 1; }
            for &t in b { *cb.entry(t).or_insert(0) += 1; }
            let mut ids: Vec<u32> = ca.keys().chain(cb.keys()).copied().collect();
            ids.sort_unstable(); ids.dedup();
            ids.sort_by(|x, y| {
                let (ox, oy) = (ca.get(x).copied().unwrap_or(0) + cb.get(x).copied().unwrap_or(0),
                                ca.get(y).copied().unwrap_or(0) + cb.get(y).copied().unwrap_or(0));
                oy.cmp(&ox).then(x.cmp(y))
            });
            ids.truncate(top);
            let (na, nb) = (a.len() as f64, b.len() as f64);
            let mut x2 = 0.0f64;
            let mut rows = Vec::new();
            let (mut oa_o, mut ob_o) = (0usize, 0usize);
            for &id in &ids {
                let (oa, ob) = (ca.get(&id).copied().unwrap_or(0), cb.get(&id).copied().unwrap_or(0));
                oa_o += oa; ob_o += ob;
                if oa + ob < 5 { continue; }
                let (ea, eb) = ((oa + ob) as f64 * na / (na + nb), (oa + ob) as f64 * nb / (na + nb));
                x2 += (oa as f64 - ea).powi(2) / ea + (ob as f64 - eb).powi(2) / eb;
                rows.push((id, oa, ob));
            }
            // the "other" bin absorbs everything outside the listed ids
            let (oa, ob) = (a.len() - oa_o, b.len() - ob_o);
            let (ea, eb) = ((oa + ob) as f64 * na / (na + nb), (oa + ob) as f64 * nb / (na + nb));
            if ea > 0.0 && eb > 0.0 { x2 += (oa as f64 - ea).powi(2) / ea + (ob as f64 - eb).powi(2) / eb; }
            rows.push((u32::MAX, oa, ob));
            (x2, rows.len() - 1, rows)
        }
        let sp = SampleParams { temperature: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0, rep_penalty: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0 };
        let prefixes: [Vec<u32>; 3] = [vec![785, 6722, 315, 9625, 374], vec![9707, 11, 1879, 311, 279], vec![3838, 374, 220, 16, 13]];
        let n = tokens.max(4000);
        let width = block - 1;   // the verify width: a full block of drafts
        let mut eng = Engine::load_spec(dev.clone(), &pack, Shard::single(), 4096, n_layers, true, 64, Some(&sc))?;
        eng.arm()?;
        fn one(eng: &mut Engine, sp: &SampleParams, pre: &[u32], seed: u64, spec: bool, k: usize) -> Result<u32> {
            eng.begin_request(sp, seed, pre)?;
            let row = eng.prefill(pre, 0, true)?.unwrap();
            let (t0, _) = eng.dec.sample_row(&eng.model, row)?;
            let pos = pre.len();
            if !spec { let (nt, _) = eng.decode_step(t0, pos)?; return Ok(nt); }
            let (d, _c) = eng.draft(t0, pos)?;
            let (c, _, _) = eng.spec_round(Some((d.as_slice(), k)), t0, pos)?;
            Ok(c[0])
        }
        let (mut worst_p, mut fails) = (1.0f64, 0usize);
        println!("gate (b) distribution exactness: two-sample chi-square, top-50 + other, N = {n} draws/path");
        for (pi, pre) in prefixes.iter().enumerate() {
            let mut pl = Vec::with_capacity(n); let mut spc = Vec::with_capacity(n); let mut pl2 = Vec::with_capacity(n);
            for i in 0..n {
                let seed = 1000 + i as u64;
                pl.push(one(&mut eng, &sp, pre, seed, false, width)?);
                spc.push(one(&mut eng, &sp, pre, seed, true, width)?);
                pl2.push(one(&mut eng, &sp, pre, 1_000_000 + i as u64, false, width)?);
            }
            let (x2s, dfs, _) = chi2_two(&pl, &spc, 50);
            let (x2c, dfc, _) = chi2_two(&pl, &pl2, 50);
            let ps = gammq(dfs as f64 / 2.0, x2s / 2.0);
            let pc = gammq(dfc as f64 / 2.0, x2c / 2.0);
            println!("  prefix {pi}: spec vs plain chi2 {x2s:.1} df {dfs} p {ps:.4}  |  control plain vs plain chi2 {x2c:.1} df {dfc} p {pc:.4}");
            ensure!(pc >= 0.01, "NAIVE_CHI_CONTROL_FAIL: the plain-vs-plain control rejected at p = {pc:.4} (prefix {pi}) — the test itself is broken");
            if ps < 0.01 { println!("    SPEC REJECTS at p<0.01 (prefix {pi})"); fails += 1; worst_p = worst_p.min(ps); }
            else { worst_p = worst_p.min(ps); }
        }
        ensure!(fails == 0, "NAIVE_CHI_FAIL: the speculative path rejected distribution exactness on {fails}/3 prefixes (worst p {worst_p:.4})");
        println!("NAIVE_CHI_OK (spec indistinguishable from plain on 3 prefixes at p>=0.01; control clean)");
        return Ok(());
    }
    // one rank's run: returns its committed token stream and the acceptance stats
    fn run(dev: Arc<CudaDevice>, pack: Arc<NaivePack>, prompt: Vec<u32>, refs: Arc<Vec<u32>>, sc: SpecCfg, n_layers: usize,
           tokens: usize, mode: String, rank: usize, world: usize, tp: Option<NaiveTp>) -> Result<(Vec<u32>, usize, usize, f64)> {
        dev.bind_to_thread()?;
        let block = sc.block; let n = block - 1;
        let head = rank == 0;
        // work item A: `sampled` runs the request at the PRODUCTION sampler settings (the pack's
        // generation_config defaults: T 1.0 / top_p 0.95 / top_k 20) so the exact-acceptance path is
        // the one under test; every other mode stays greedy (its gates are token-exactness).
        let sp = if mode == "sampled" {
            SampleParams { temperature: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0, rep_penalty: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0 }
        } else { SampleParams::GREEDY };
        let mut eng = Engine::load_spec(dev.clone(), &pack, if world == 1 { Shard::single() } else { Shard { rank, world } }, 4096, n_layers, head, 64, Some(&sc))?;
        eng.model.tp = tp;
        if world == 1 { eng.arm()?; }
        eng.begin_request(&sp, 1, &prompt)?;
        let row = eng.prefill(&prompt, 0, true)?.unwrap();
        let (mut t, _) = eng.dec.sample_row(&eng.model, row)?;
        let mut out = vec![t];
        let mut pos = prompt.len();
        let (mut rounds, mut acc) = (0usize, 0usize);
        let mut hist = vec![0usize; n + 1];
        let t0 = std::time::Instant::now();
        let mut r = 0usize;
        while out.len() < tokens {
            if mode == "mixed" && r % 2 == 1 {
                let (nt, _) = eng.decode_step(t, pos)?; out.push(nt); pos += 1; t = nt; r += 1; continue;
            }
            // (drafts, verify width): `adapt` cycles the width 0..=n over oracle drafts (every verify step,
            // including the width-0 plain step, must give the plain continuation); `drafter` uses the
            // confidence head's width; the other modes verify the whole block.
            let drafts: Option<(Vec<u32>, usize)> = if !head { None } else if mode == "drafter" || mode == "sampled" {
                let (d, c) = eng.draft(t, pos)?; let k = eng.pick_width(&c); Some((d, k))
            } else {
                let mut d: Vec<u32> = refs[out.len()..out.len() + n].to_vec();
                if mode == "miss" { let j = r % n; d[j] = if d[j] == 0 { 1 } else { d[j] - 1 }; }
                let k = if mode == "adapt" { r % (n + 1) } else { n };
                Some((d, k))
            };
            let (committed, a, _) = eng.spec_round(drafts.as_ref().map(|(d, k)| (d.as_slice(), *k)), t, pos)?;
            if let Some((_, k)) = drafts.as_ref() {
                // work item A invariants, every round: never accept more rows than were verified,
                // and a round commits exactly a+1 tokens (the a drafts plus the one the target itself
                // drew — the residual when row a rejected, the bonus row when every row accepted).
                ensure!(a <= *k, "NAIVE_SPEC_FAIL: accepted {a} > verified rows {k}");
                ensure!(committed.len() == a + 1, "NAIVE_SPEC_FAIL: committed {} tokens for {a} accepted of {k}", committed.len());
                if mode == "sampled" { hist[a] += 1; }
            }
            rounds += 1; acc += a;
            out.extend_from_slice(&committed);
            pos += a + 1;
            t = *committed.last().unwrap();
            r += 1;
        }
        out.truncate(tokens);
        if head && mode == "sampled" {
            let full = *hist.last().unwrap_or(&0);
            println!("  accept histogram (accepted rows -> rounds): {hist:?}  ({full}/{rounds} rounds accepted every verified row)");
        }
        Ok((out, rounds, acc, t0.elapsed().as_secs_f64()))
    }
    let (spec_toks, rounds, acc, dt) = if world == 1 {
        run(dev.clone(), pack.clone(), prompt.clone(), ref_toks.clone(), sc.clone(), n_layers, tokens, mode.to_string(), 0, 1, None)?
    } else {
        let mut tps = NaiveTp::loopback_group(&dev, pack.cfg.hidden_size, 64.max(MAX_ROWS), world)?;
        let tp0 = tps.remove(0);
        let mut handles = Vec::new();
        for (i, tp) in tps.into_iter().enumerate() {
            let (d, p, pr, rf, s, m) = (dev.clone(), pack.clone(), prompt.clone(), ref_toks.clone(), sc.clone(), mode.to_string());
            handles.push(std::thread::spawn(move || run(d, p, pr, rf, s, n_layers, tokens, m, i + 1, world, Some(tp))));
        }
        let r0 = run(dev.clone(), pack.clone(), prompt.clone(), ref_toks.clone(), sc.clone(), n_layers, tokens, mode.to_string(), 0, world, Some(tp0))?;
        for (i, h) in handles.into_iter().enumerate() {
            let (o, _, _, _) = h.join().map_err(|_| anyhow::anyhow!("rank {} thread panicked", i + 1))??;
            println!("  rank {}: tokens identical to rank 0: {}", i + 1, o == r0.0);
            ensure!(o == r0.0, "NAIVE_SPEC_FAIL: rank {} committed a different stream", i + 1);
        }
        r0
    };
    println!("  plain greedy {:?}", &ref_toks[..tokens.min(16)]);
    println!("  spec  greedy {:?}", &spec_toks[..tokens.min(16)]);
    println!("  mode {mode} world {world}: {rounds} rounds, {:.2} accepted/round, {:.1} ms/round (block {block}, {n_layers} layers)", acc as f64 / rounds.max(1) as f64, dt / rounds.max(1) as f64 * 1e3);
    if mode == "sampled" {
        // A sampled round draws a token the plain path would not draw (different RNG streams), so
        // token-equality is NOT the assertion here. What this mode proves: the chain's structural
        // invariants held every round (checked in `run`), every committed id is in vocabulary, and the
        // path is deterministic for a fixed seed (a second run reproduces the stream). Exactness of
        // the DISTRIBUTION is gate (b): scripts/naive/spec_exact.py.
        ensure!(spec_toks.iter().all(|&t2| (t2 as usize) < pack.cfg.vocab_size), "NAIVE_SPEC_FAIL: sampled stream has an out-of-vocabulary id");
        // work item C's gate: the same seed under `--spec-sampler rows` and `batched` must print the same hash
        let h = spec_toks.iter().fold(0xcbf29ce484222325u64, |h, &t2| (h ^ t2 as u64).wrapping_mul(0x100000001b3));
        println!("  sampled stream hash {h:016x} ({} tokens, {rounds} rounds, {acc} accepted; --spec-sampler {sampler})", spec_toks.len());
        if world == 1 {
            let (b2, r2, a2, _) = run(dev.clone(), pack.clone(), prompt.clone(), ref_toks.clone(), sc.clone(), n_layers, tokens, mode.to_string(), 0, 1, None)?;
            ensure!(b2 == spec_toks && r2 == rounds && a2 == acc,
                    "NAIVE_SPEC_FAIL: sampled path is not deterministic for a fixed seed (rounds {r2} vs {rounds}, accepted {a2} vs {acc})");
            println!("  determinism: a second run with the same seed reproduced the stream ({rounds} rounds, {acc} accepted)");
        }
        println!("NAIVE_SPEC_OK (sampled: invariants + vocabulary + determinism; distribution exactness = spec_exact.py)");
        return Ok(());
    }
    if spec_toks != ref_toks[..tokens] {
        let first = spec_toks.iter().zip(ref_toks.iter()).position(|(a, b)| a != b).unwrap_or(tokens);
        bail!("NAIVE_SPEC_FAIL: speculative greedy differs from plain greedy at token {first} (spec {:?} vs plain {:?})",
              &spec_toks[first.saturating_sub(2)..(first + 3).min(tokens)], &ref_toks[first.saturating_sub(2)..(first + 3).min(tokens)]);
    }
    println!("NAIVE_SPEC_OK");
    Ok(())
}

