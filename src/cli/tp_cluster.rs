//! TP transport benches (reduce / barrier), the network test, the cluster node and head entry points and
//! the memory-budget report. TP=2 `agree()` guards and the watchdog are load-bearing (AGENTS.md section 2).

use super::*;

/// TP-F `--tp-reduce-bench` — prefill-sized fp32 all-reduce (serial vs pipelined schedule, 1..2 rails)
/// on the real transport, every reduce validated bitwise. Same command on both boxes (--rank 0 / --rank 1
/// --peer <head-ip>), identical flags.
/// TP-4C: `--tp-reduce-bench --mode decw|decwf32|decwserial|decwkeys|decwmix` — the decode-path reduce at
/// world 2 or 4 (separate bench over `connect_nway`; every other mode is the unchanged pair bench below).
fn run_tp_reduce_bench_world(args: &[String]) {
    let g = |k: &str, d: u64| gb10_inference::opt_or_arg(args, k).and_then(|s| s.parse().ok()).unwrap_or(d);
    let a = gb10_inference::tp_bench::DecWorldArgs {
        rank: parse_arg(args, "--rank").and_then(|s| s.parse().ok()).expect("--tp-reduce-bench --mode decw* needs --rank 0..world-1"),
        world: g("--world", 4) as u32,
        peer_ips: parse_arg(args, "--peer-ips")
            .map(|s| s.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect())
            .unwrap_or_default(),
        port: g("--port", 29700) as u16,
        dev: parse_arg(args, "--dev").unwrap_or("rocep1s0f1").to_string(),
        gid: g("--gid", 3) as i32,
        floats: g("--floats", 10240) as usize,
        reduces: g("--reduces", 20) as usize,
        mode: parse_arg(args, "--mode").unwrap_or("decw").to_string(),
        blocks: g("--blocks", 8) as u32,
        proxy_core: g("--proxy-core", 19) as i32,
        main_core: g("--main-core", 9) as i32,
    };
    if let Err(e) = gb10_inference::tp_bench::run_reduce_dec_nway(a) {
        eprintln!("\n[tp-reduce-bench] FAILED: {e:#}");
        std::process::exit(1);
    }
}

pub(crate) fn run_tp_reduce_bench(args: &[String]) {
    if parse_arg(args, "--mode").map_or(false, |m| m.starts_with("decw")) {
        return run_tp_reduce_bench_world(args);
    }
    let g = |k: &str, d: u64| gb10_inference::opt_or_arg(args, k).and_then(|s| s.parse().ok()).unwrap_or(d);
    if g("--world", 2) > 2 { return run_tp_reduce_bench_w4(args); }   // TP-4E: the world > 2 prefill sweep
    let a = gb10_inference::tp_bench::ReduceArgs {
        rank: parse_arg(args, "--rank").and_then(|s| s.parse().ok()).expect("--tp-reduce-bench needs --rank 0|1"),
        peer: parse_arg(args, "--peer").unwrap_or("").to_string(),
        port: g("--port", 29700) as u16,
        dev: parse_arg(args, "--dev").unwrap_or("rocep1s0f1").to_string(),
        gid: g("--gid", 3) as i32,
        dev2: parse_arg(args, "--dev2").unwrap_or("").to_string(),
        floats: g("--floats", 2048 * 2560) as usize,
        reduces: g("--reduces", 200) as usize,
        mode: parse_arg(args, "--mode").unwrap_or("pipe").to_string(),
        lookahead: g("--lookahead", 4) as usize,
        blocks: g("--blocks", 16) as u32,
        mix: g("--mix", 0) as usize,
        io: parse_arg(args, "--io").unwrap_or("f32").to_string(),
        proxy_core: g("--proxy-core", 19) as i32,
        proxy_core2: g("--proxy-core2", 18) as i32,
        main_core: g("--main-core", 9) as i32,
    };
    if let Err(e) = gb10_inference::tp_bench::run_reduce(a) {
        eprintln!("\n[tp-reduce-bench] FAILED: {e:#}");
        std::process::exit(1);
    }
}

/// TP-4E `--tp-reduce-bench --world 4 --rank R --peer-ips IP0,IP1,IP2,IP3 --xport 0|1|2 [--io f32|f32to16|f16|f16w]
/// [--sizes-mb 2,8,21,42] [--reduces N] [--warmup N] [--dev D1] [--dev2 D2] [--port P]` — the prefill-sized
/// all-reduce at world > 2 on the real transport (no model), every reduce bitwise-checked against the host
/// recursive-doubling reference; per-reduce microseconds and GB/s per direction. Same command on all four boxes.
fn run_tp_reduce_bench_w4(args: &[String]) {
    let g = |k: &str, d: u64| gb10_inference::opt_or_arg(args, k).and_then(|s| s.parse().ok()).unwrap_or(d);
    fn fail(e: anyhow::Error) -> ! { eprintln!("\n[tp-reduce-bench --world>2] FAILED: {e:#}"); std::process::exit(1) }
    let sizes = gb10_inference::tp_bench::w4_parse_sizes(parse_arg(args, "--sizes-mb").unwrap_or("2,8,21,42")).unwrap_or_else(|e| fail(e));
    let a = gb10_inference::tp_bench::ReduceW4Args {
        rank: parse_arg(args, "--rank").and_then(|s| s.parse().ok()).expect("--tp-reduce-bench --world N needs --rank"),
        world: g("--world", 4) as i32,
        peer_ips: parse_arg(args, "--peer-ips")
            .map(|s| s.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect())
            .unwrap_or_default(),
        port: g("--port", 29700) as u16,
        dev: parse_arg(args, "--dev").unwrap_or("rocep1s0f1").to_string(),
        dev2: parse_arg(args, "--dev2").unwrap_or("").to_string(),
        gid: g("--gid", 3) as i32,
        xport: g("--xport", 2) as u8,
        sizes_mb: sizes,
        reduces: g("--reduces", 20) as usize,
        warmup: g("--warmup", 2) as usize,
        io: parse_arg(args, "--io").unwrap_or("f32").to_string(),
        lookahead: g("--lookahead", 4) as usize,
        blocks: g("--blocks", 16) as u32,
        proxy_core: g("--proxy-core", 19) as i32,
        proxy_core2: g("--proxy-core2", 18) as i32,
        main_core: g("--main-core", 9) as i32,
    };
    if let Err(e) = gb10_inference::tp_bench::run_reduce_w4(a) { fail(e); }
}

/// `--tp-barrier-bench` — adversarial proof of the doorbell all-reduce on the real transport, no model.
/// Run the SAME command on both boxes with `--rank 0` (head, listens) and `--rank 1 --peer <head-ip>`;
/// the barrier count and every mode flag must match, since the two ranks rendezvous per barrier.
pub(crate) fn run_tp_barrier_bench(args: &[String]) {
    let g = |k: &str, d: u64| gb10_inference::opt_or_arg(args, k).and_then(|s| s.parse().ok()).unwrap_or(d);
    let a = gb10_inference::tp_bench::BenchArgs {
        rank: parse_arg(args, "--rank").and_then(|s| s.parse().ok())
            .expect("--tp-barrier-bench needs --rank 0|1"),
        peer: parse_arg(args, "--peer").unwrap_or("").to_string(),
        port: g("--port", 29600) as u16,
        dev: parse_arg(args, "--dev").unwrap_or("rocep1s0f1").to_string(),
        gid: g("--gid", 3) as i32,
        world: g("--world", 2) as u32,
        // P6: rank-indexed peer topology (comma-separated, peer_ips[rank] = that rank's RoCE IP).
        // Empty = derive the world==2 single peer from --peer (and reject world>2 below in run()).
        peer_ips: parse_arg(args, "--peer-ips")
            .map(|s| s.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect())
            .unwrap_or_default(),
        barriers: g("--barriers", 1_000_000),
        payload_bytes: g("--payload-bytes", 10240) as usize,   // 5120 bf16 = the 27B hidden vector
        spacing_us: g("--spacing-us", 0),
        inject_delay_us_max: g("--inject-delay-us-max", 0) as u32,
        poison: args.iter().any(|x| x == "--poison"),
        stall_every: g("--stall-consumer-every", 0) as u32,
        stall_us: g("--stall-us", 0),
        window: g("--window", 1024),
        proxy_core: g("--proxy-core", 19) as i32,
        main_core: g("--main-core", 9) as i32,
        cq_hold: g("--cq-hold", 0) as u32,
        cq_hold_us: g("--cq-hold-us", 50) as u32,
    };
    if let Err(e) = gb10_inference::tp_bench::run(a) {
        eprintln!("\n[tp-barrier-bench] FAILED: {e:#}");
        std::process::exit(1);
    }
}

/// G-A: TP=2 transport + FP32-partial numerical audit (design §2/§4, build step 1). Both ranks generate
/// the SAME deterministic contributions, each sums its K-half into an FP32 partial, exchanges over RDMA,
/// then rank 0 checks: (1) transport byte-exact, (2) FP32-partial reduce == single-node full-K reduce
/// (lossless), and how much WORSE a bf16-partial reduce is (the §4 hole FP32 partials close), (3) latency.
pub(crate) fn run_net_test(args: &[String]) {
    let rank: i32 = parse_arg(args, "--rank").and_then(|s| s.parse().ok())
        .expect("--net-test needs --rank 0|1");
    let peer = parse_arg(args, "--peer").unwrap_or("").to_string();
    let port: u16 = parse_arg(args, "--port").and_then(|s| s.parse().ok()).unwrap_or(23470);
    let dev = parse_arg(args, "--dev").unwrap_or("rocep1s0f1").to_string();
    let gid: i32 = parse_arg(args, "--gid").and_then(|s| s.parse().ok()).unwrap_or(3);

    const M: usize = 5120;   // 27B hidden -> payload 20480 B, the real all-reduce size
    const K: usize = 4096;   // contributions per output element, split K/2 per rank
    let half = K / 2;
    let k0 = (rank as usize) * half;   // this rank's K-range [k0, k0+half)

    // deterministic contribution in [-1,1), identical on both nodes (splitmix64-style hash)
    #[inline] fn contrib(i: usize, k: usize) -> f32 {
        let mut x = (i as u64).wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add((k as u64).wrapping_mul(0xD1B54A32D192ED03));
        x ^= x >> 29; x = x.wrapping_mul(0xBF58476D1CE4E5B9); x ^= x >> 32;
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    // +8: net_exchange rides a generation tag in the payload's last 8 bytes (the recv-side placement
    // proof — see net_shim.c). Pad the slot so the tag lands past the audited floats.
    let slot = M * 4 + 8;   // FP32 payload bytes + exchange trailer
    let mut link = gb10_inference::net::TpLink::connect(rank, &peer, port, &dev, gid, slot)
        .expect("TpLink::connect");

    let mut partial = vec![0f32; M];
    for i in 0..M { let mut s = 0f32; for k in k0..k0 + half { s += contrib(i, k); } partial[i] = s; }
    link.send_host_mut::<f32>(M).copy_from_slice(&partial);
    link.exchange(slot).expect("exchange");
    let peer_partial: Vec<f32> = link.recv_host::<f32>(M).to_vec();

    if rank != 0 {
        for _ in 0..2000 { link.exchange(slot).expect("exchange"); }   // responder for the latency loop
        println!("[node] rank 1 done.");
        return;
    }

    // (1) transport byte-exact: recv must equal the peer's independently-recomputable partial
    let mut peer_ref = vec![0f32; M];
    for i in 0..M { let mut s = 0f32; for k in half..K { s += contrib(i, k); } peer_ref[i] = s; }
    let tx_ok = peer_partial.iter().zip(&peer_ref).all(|(a, b)| a.to_bits() == b.to_bits());

    // (2) FP32-partial vs single-node full-K reference; and the bf16-partial (lossy) path
    let b16 = |x: f32| half::bf16::from_f32(x).to_f32();
    let (mut fp32_mm, mut bf16_mm) = (0usize, 0usize);
    let (mut fp32_max, mut bf16_max) = (0f32, 0f32);
    for i in 0..M {
        let mut refi = 0f32; for k in 0..K { refi += contrib(i, k); }   // single-node full-K FP32
        let refb   = b16(refi);
        let fp32tp = b16(partial[i] + peer_partial[i]);              // FP32 partials -> one round
        let bf16tp = b16(b16(partial[i]) + b16(peer_partial[i]));    // bf16 partials -> extra rounding
        if fp32tp.to_bits() != refb.to_bits() { fp32_mm += 1; }
        if bf16tp.to_bits() != refb.to_bits() { bf16_mm += 1; }
        fp32_max = fp32_max.max((fp32tp - refb).abs());
        bf16_max = bf16_max.max((bf16tp - refb).abs());
    }

    // (3) exchange latency
    let n = 2000usize;
    let t0 = std::time::Instant::now();
    for _ in 0..n { link.exchange(slot).expect("exchange"); }
    let us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;

    println!("=== G-A: TP=2 transport + FP32-partial audit (M={M}, payload {slot} B, K={K}) ===");
    println!("  (1) transport byte-exact (recv == peer partial): {}", if tx_ok { "YES" } else { "NO  ***FAIL***" });
    println!("  (2) FP32-partial reduce vs single-node: {fp32_mm}/{M} bf16 mismatches, max|Δ|={fp32_max:.3e}");
    println!("      bf16-partial reduce vs single-node: {bf16_mm}/{M} bf16 mismatches, max|Δ|={bf16_max:.3e}  <- the hole FP32 partials close");
    println!("  (3) exchange latency: {us:.2} us/op   (x128/token = {:.2} ms)", us * 128.0 / 1000.0);
}

/// TP=2 cluster NODE: answer discovery + accept one head sync, bring up the RDMA link, then join the
/// SPMD masked-replicated decode (Proof v0). The node loads the synced model, receives the prompt from
/// the head over the link, and runs the IDENTICAL greedy generate loop — the per-layer FFN all-reduces
/// keep it bit-for-bit in step with the head, which owns the printing.
pub(crate) fn run_cluster_node(args: &[String]) {
    // (--rdma-dev is a registry option: parsed at startup, node-local, never shipped)
    let port: u16 = parse_arg(args, "--port").and_then(|s| s.parse().ok()).unwrap_or(29500);
    // OUT2: the node's UDP discovery bind is 29499 by default; a resident supervisor (or any other
    // process) may already hold it on a shared box -- this flag lets a second node coexist without
    // touching the first. The head is unaffected (explicit --nodes dials TCP; no UDP involved).
    let udp: u16 = parse_arg(args, "--discovery-port").and_then(|s| s.parse().ok()).unwrap_or(29499);
    // resolve + print the RDMA rails at supervisor start: a box with two active port pairs exits HERE
    // (naming them), not when a head connects
    let _ = gb10_inference::tp::rdma_devs();

    // RESIDENT by default (TP item B): supervise one-shot sessions, re-arming after each so the node
    // survives a head restart with zero manual intervention. One process per session is the whole
    // isolation story: the process-global TP config, the mem::forget'd link + proxy thread, and the
    // freshly-sharded weights (attach_tp shards in place — a model cannot be re-attached) all die
    // with the child, so the next head sync starts from a provably clean process. `--once` (or the
    // `--once` the supervisor passes its children — CLI-1: no env marker) runs a single session, for
    // debugging. A graceful per-request serve loop (no reload between requests) is item A's server
    // mode; until then a session = one sync + one SPMD run.
    let once = args.iter().any(|a| a == "--once");
    if once { run_cluster_node_once(port, udp); return; }

    let exe = std::env::current_exe().expect("current_exe");
    eprintln!("[node-resident] supervisor up on port {port} — one process per head session; \
               kill this process to stop the node");
    loop {
        let t0 = std::time::Instant::now();
        let status = std::process::Command::new(&exe)
            .args(node_child_args(&args, port, udp))
            .status();
        match status {
            Ok(s) => eprintln!("[node-resident] session ended ({s}) after {:.1}s — re-arming for the next head",
                               t0.elapsed().as_secs_f64()),
            Err(e) => eprintln!("[node-resident] failed to spawn a session process: {e} — retrying"),
        }
        // A session that dies instantly (bind failure, RDMA down) must not spin-fork.
        if t0.elapsed().as_secs() < 5 { std::thread::sleep(std::time::Duration::from_secs(1)); }
    }
}

/// One head session: sync (model + config) -> bring up the RDMA link -> serve one SPMD run -> exit.
fn run_cluster_node_once(port: u16, udp: u16) {
    gb10_inference::dprintln!("{}", gb10_inference::exl3_forward::xtp::identity_line("node"));
    let (dir, draft_dir, head_ip, mut tpc, stream) = match gb10_inference::cluster::run_node(port, udp) {
        Ok(x) => x,
        Err(e) => { eprintln!("node error: {e:#}"); std::process::exit(1); }
    };
    // v5 gap fix (2026-08-23): when the head shipped the DFlash2 drafter, its bytes are in OUR
    // blob cache and `draft_dir` is the assembled cache path — rewrite the config's draft dir to
    // it so node_serve_tp loads the drafter FROM THE CACHE. When the head shipped NO drafter,
    // CLEAR the path: a node must never open the head's filesystem path (a stale local dir at
    // the same location would silently stand in for a failed sync — the round load then fails
    // cleanly and the CalibTable df2_round outcome keeps both sides consistent instead).
    match &draft_dir {
        Some(d) => {
            println!("NODE — draft artifact synced via blob cache: {} (no local copy needed)", d.display());
            tpc.df2_draft_dir = d.to_string_lossy().to_string();
        }
        None => { tpc.df2_draft_dir.clear(); }
    }
    // CLI-1: the head's resolved option registry (TpConfig v23 `opts`) — installed FIRST, before any
    // consumer reads an option: the node's own command line carries only node-local options.
    if let Err(e) = gb10_inference::opts::install_head(&tpc.opts) {
        // PACK-FIX: tell the head (blocked in its link bring-up) why, then exit
        let mut stream = stream;
        gb10_inference::cluster::node_report_failure(&mut stream, &format!("head options refused: {e}"));
        eprintln!("node error: {e}"); std::process::exit(1);
    }
    gb10_inference::dprintln!("[node] options from the head: {}", gb10_inference::opts::set_summary());
    // NIGHT1: the head's --gpu-wait into this node's context scheduling flags (the node's own command line
    // resolved the default before the head's options arrived).
    gb10_inference::host_wait::apply_context_policy();
    // The head's TP config, shipped during the sync — install BEFORE any TP consumer reads a setting,
    // so the node reproduces the head's behavior with zero local configuration.
    gb10_inference::tp::set_tp_config(tpc.clone());
    let node_rank = tpc.node_rank;
    let world = tpc.world as i32;
    if tpc.exl3_mode.starts_with("naive-") {
        // NAIVE S5b: the Naive-N0.5 node loads the shipped shard dir
        if let Err(e) = run_naive_tp_node(&tpc, &dir, head_ip, stream) {
            eprintln!("node naive tp error: {e:#}"); std::process::exit(1);
        }
        return;
    }
    if tpc.exl3 {
        // TP-A: the EXL3 engine, zero-config knobs. B32 (always, for Qwen3.8-Flash-Next): the rank's shard dir the head
        // just shipped through the blob cache; TPSHIP (2026-11-07): every other EXL3 pack ships whole the same way
        // (ship_shards is set for both), so a node never opens the head's filesystem path. TP-D: the serve mode keeps the control stream (the Step /
        // HeadFlag plane); the programs drop it.
        let mut tpc = tpc;
        if tpc.ship_shards {
            gb10_inference::dprintln!("NODE — EXL3 pack shipped through the blob cache: {} (the head's path is not read)", dir.display());
            tpc.exl3_pack_dir = dir.to_string_lossy().to_string();
            gb10_inference::tp::set_tp_config(tpc.clone());
        }
        // DF2 block-16 (2026-11-08, the runner's 27B EXL3 DF2 TP=2/TP=4 boots): install the HEAD's draft block before
        // the EXL3 load, as the NVFP4 node does below. This branch returned before that site, so an EXL3 node kept the
        // built-in block 8 while a --df2-block 16 head ran 16 — the TP boot agree refused every such boot (its DF2 word
        // carries the global block). SPMD-critical: MASK rows and verify columns are collective.
        // TP2BIT5 #26: clean exit, never a panic! — a panic in the node boot path unwinds
        // through partial CUDA state and can double-abort in CudaSlice::drop (the issue's
        // teardown double-panic class).
        if let Err(e) = gb10_inference::dflash2::set_block(tpc.df2_block) {
            eprintln!("node exl3 tp error: shipped df2_block {}: {e}", tpc.df2_block);
            std::process::exit(1);
        }
        gb10_inference::dprintln!("NODE — DF2 block = {} (from the head's config)", gb10_inference::dflash2::block());
        if let Err(e) = run_exl3_tp_node(&tpc, head_ip, stream) {
            eprintln!("node exl3 tp error: {e:#}"); std::process::exit(1);
        }
        return;
    }
    println!("NODE SYNCED (rank {node_rank}/{world}) — model ready at {}", dir.display());
    // ISSUES 153: every branch below either brings the link up with this stream still alive
    // (the dial/accept loops abort if it dies) or drops it (cleared first).
    gb10_inference::net::set_head_liveness(&stream);
    let r = if tpc.mode_serve && is_dsv4_bundle(&dir) {
        // DSV4 persistent server: load once, loop over broadcast_prompt → decode. The control
        // stream stays RETAINED when the head runs --server-dspark (item 3.4: the head ships
        // its r(D) table over it once per process); greedy mode has no consumer and drops it.
        dsv4_tp_serve_node_loop(&dir.to_string_lossy(), head_ip, Some(stream))
    } else if tpc.mode_serve {
        // Serving session: mirror the head's OpenAI-server BatchScheduler in SPMD lockstep over the
        // retained control stream (TP item A). Returning Ok ends the session; the resident
        // supervisor re-arms for the next head.
        node_serve_tp(&dir, head_ip, stream, &tpc)
    } else if tpc.dspark && is_dsv4_bundle(&dir) {
        // P4: DSV4 N-way deferred (still TP=2). DSpark speculation session: the minimal
        // Dsv4GpuModel SPMD DSpark path (no head req — the head ships the prompt via
        // broadcast_prompt; the node mirrors the identical draft+verify). The retained control
        // stream stays open: the head ships its measured r(D) table over it (item 3.3 adaptive
        // depth — the node runs the identical SPMD calibration forwards and discards its timings).
        dsv4_tp_dspark_serve(&dir.to_string_lossy(), gb10_inference::tp::TpContext::bring_up_node(head_ip, 1, 2), None, Some(stream))
    } else if is_dsv4_bundle(&dir) {
        // P4: DSV4 N-way deferred (still TP=2). DSV4 bench/first-light session: the minimal
        // Dsv4GpuModel SPMD greedy path (no head req).
        gb10_inference::net::clear_head_liveness();
        drop(stream);
        dsv4_tp_serve(&dir.to_string_lossy(), gb10_inference::tp::TpContext::bring_up_node(head_ip, 1, 2), None)
    } else {
        // Bench session: one-shot SPMD bench/generate (unchanged). The retained stream is dropped.
        gb10_inference::net::clear_head_liveness();
        drop(stream);
        tp_serve(&dir.to_string_lossy(), gb10_inference::tp::TpContext::bring_up_node(head_ip, node_rank, world), None)
    };
    if let Err(e) = r {
        eprintln!("node tp serve error: {e:#}"); std::process::exit(1);
    }
}

/// TP=2 serving NODE (TP item A): bring up the RDMA link, load the synced model, attach TP, then
/// mirror the head's BatchScheduler. All admissions/cancels arrive as per-step events on the
/// retained sync stream; the node runs identical scheduler state and discards its TokEvents.

/// Startup memory-budget estimate for this configuration, printed BEFORE the model load.
/// **INFORM ONLY — no action is taken.** On the unified-DRAM GB10 (CPU and GPU share one pool),
/// a configuration that exceeds physical memory does not swap — it dies by earlyoom SIGTERM
/// (the daemon's whole job). The estimate is deliberately pessimistic-brief: weights from the
/// shard bytes, KV preallocated at max_seq_len, the startup calibration transient (a single-slot
/// KV cache at the top context bucket, freed after timing), the q4 mirror budget, and flat pools.
pub(super) fn mem_budget_report(model_dir: &str, cfg: &gb10_inference::qwen::Config,
                     tp: bool, max_seq_len: usize, max_batch: usize, kv_mode: gb10_inference::gpu::KVCacheMode) {
    let gb = 1u64 << 30;
    let dir = std::path::Path::new(model_dir);
    let mut total = 0u64;
    // B32: a shard dir keeps its segments under seg/ and is ALREADY this rank's share
    let shard_dir = dir.join(gb10_inference::shard_plan::SHARD_FILE).is_file();
    for d in [dir.to_path_buf(), dir.join("seg")] {
        if let Ok(rd) = std::fs::read_dir(&d) {
            for e in rd.flatten() {
                if e.file_name().to_string_lossy().ends_with(".safetensors") {
                    total += e.metadata().map(|m| m.len()).unwrap_or(0);
                }
            }
        }
    }
    // Per-rank weights: TP shard-at-load splits experts/attention (and the E7 bf16 lm_head) 50/50;
    // embed + router + shared_mlp + norms are replicated. replicated ≈ embed + lm_head(E7 half) +
    // shared_mlp (nvfp4) + small extras.
    let (v, h) = (cfg.vocab_size as u64, cfg.hidden_size as u64);
    let embed = v * h * 2;
    let lm_head_r = embed / 2;
    let shared_b = (cfg.num_layers.saturating_sub(1)) as u64 * 3
        * cfg.shared_expert_intermediate_size as u64 * h * 9 / 16; // ~0.5625 B/elem incl scales
    let replicated = embed + lm_head_r + shared_b + (256 << 20);
    let weights = if tp && !shard_dir && total > replicated { (total - replicated) / 2 + replicated } else { total };

    // KV cache: FULL-ATTENTION layers only (GDN layers carry no KV) × 2 (K,V) × TP-local kv heads
    // × hd × elem bytes × max_seq_len × slots.
    let fa_layers = cfg.layer_types.iter()
        .filter(|t| **t == gb10_inference::qwen::LayerType::FullAttention).count() as u64;
    let nkv = if tp { (cfg.num_kv_heads as u64 / 2).max(1) } else { cfg.num_kv_heads as u64 };
    let hd = cfg.head_dim as u64;
    use gb10_inference::gpu::KVCacheMode as KM;
    // Bytes per element per K/V channel: bf16 = 2; q4 = 12 B/16 elems = 0.75;
    // k8v4 = (20 B K + 12 B V per 16 elems) / 2 channels / 16 = 32/32 = 1.0 for every hd;
    // TQ = (52 B K + 50 B V) rows / 2 / hd = 102/256 = 0.3984 (the E4 layout); the b=3 K
    // variant = (68 B K + 50 B V) / 2 / hd = 118/256 = 0.4609 ([kv-tq]=3).
    let (elem_b, quantized) = match kv_mode {
        KM::Bf16 => (2.0, false),
        KM::Q4 => (0.75, true),
        KM::K8v4 => ((hd / 16) as f64 * 32.0 / (2.0 * hd as f64), true),
        // k8v8: 20 B per 16 elems on BOTH channels = 40/32 = 1.25 B/elem (vs bf16 2.0).
        KM::K8v8 => (1.25, true),
        KM::Tq => {
            let row = if gb10_inference::opts::var(gb10_inference::opt!("kv-tq")).ok().as_deref() == Some("3") { 118.0 } else { 102.0 };
            (row / (2.0 * hd as f64), true)
        }
    };
    let kv_per_slot = fa_layers as f64 * 2.0 * nkv as f64 * hd as f64 * elem_b;
    let slots = (max_batch as u64 + 1).max(1);
    let kv = kv_per_slot * max_seq_len as f64 * slots as f64;

    // Startup calibration transient: one KV slot at the top context bucket (freed after timing),
    // same dtype as the serving cache.
    let top = gb10_inference::gpu::mtp_calib_ctx_points(max_seq_len).last().copied().unwrap_or(2048);
    let calib = kv_per_slot * top as f64;

    // q4/TQ mirror (incremental dequant, budgeted ≤32K positions): bf16 K/V mirror per full-attn layer.
    let mirror = if quantized {
        (max_seq_len.min(32768) as f64) * fa_layers as f64 * 2.0 * nkv as f64 * hd as f64 * 2.0
    } else { 0.0 };
    let pools = 3.0 * gb as f64;
    let (gf) = gb as f64;
    let steady = weights as f64 + kv + mirror + pools;
    let peak = steady + calib;
    let phys = std::fs::read_to_string("/proc/meminfo").ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("MemTotal:"))
            .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()))
        .map(|kb| kb * 1024).unwrap_or(0);
    let avail = std::fs::read_to_string("/proc/meminfo").ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("MemAvailable:"))
            .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()))
        .map(|kb| kb * 1024).unwrap_or(0);
    let fmt = |b: f64| format!("{b:.1}");
    let kv_label = match kv_mode { KM::Bf16 => "bf16", KM::Q4 => "q4", KM::Tq => "tq", KM::K8v4 => "k8v4", KM::K8v8 => "k8v8" };
    let dbg_mb = gb10_inference::boot_report::debug();
    if dbg_mb {
    eprintln!("[mem-budget] {} ({}), max-seq-len {max_seq_len}, batch {max_batch}, kv-cache {}",
              std::path::Path::new(model_dir).file_name().unwrap_or_default().to_string_lossy(),
              if tp { "TP=2, rank-local" } else { "single node" }, kv_label);
    }
    if dbg_mb { eprintln!("  weights (per rank)     ~{} GB", fmt(weights as f64 / gf)); }
    if total == 0 {
        eprintln!("  *** WARNING: no .safetensors shards found in {} — the directory may be empty, a",
                  std::path::Path::new(model_dir).display());
        eprintln!("  *** partial/stale download, or the wrong format (see the load error below).");
    }
    if dbg_mb { eprintln!("  KV cache (~{slots} slots)  ~{} GB", fmt(kv / gf)); }
    if dbg_mb { eprintln!("  calibration transient  ~{} GB (startup only, freed)", fmt(calib / gf)); }
    if quantized && dbg_mb { eprintln!("  packed-KV mirror (<=32K) ~{} GB", fmt(mirror / gf)); }
    if dbg_mb {
    eprintln!("  pools/workspaces (est) ~{} GB", fmt(pools / gf));
    eprintln!("  ----------------------------------------------");
    eprintln!("  steady-state estimate  ~{} GB of {} GB physical", fmt(steady / gf), fmt(phys as f64 / gf));
    eprintln!("  startup peak estimate  ~{} GB of {} GB physical", fmt(peak / gf), fmt(phys as f64 / gf));
    }
    if avail > 0 {
        if dbg_mb { eprintln!("  available NOW          ~{} GB (other load on this box already eats the rest)", fmt(avail as f64 / gf)); }
    }
    let headroom = phys as f64 - steady;
    let headroom_now = if avail > 0 { avail as f64 - steady } else { headroom };
    if phys > 0 && (peak > phys as f64 * 0.98 || headroom < 8.0 * gf || headroom_now < 8.0 * gf) {
        eprintln!("  *** WARNING: this configuration is estimated to need ~{} GB steady-state",
                  fmt(steady / gf));
        eprintln!("  *** (startup peak ~{} GB), leaving ~{} GB headroom NOW.",
                  fmt(peak / gf), fmt(headroom_now / gf));
        eprintln!("  *** The earlyoom daemon on this box SIGTERMs the largest processes under");
        eprintln!("  *** memory pressure — THIS CONFIG MAY EXHAUST MEMORY. Inform only — NO action");
        eprintln!("  *** taken. Consider a smaller --max-seq-len, --kv-cache q4, or clearing other");
        eprintln!("  *** large processes from this box first.");
    }
}

pub(crate) fn run_cluster_head(args: &[String]) {
    // (--rdma-dev is a registry option: parsed at startup, node-local, never shipped)
    let model_dir = parse_arg(args, "--model-dir").expect("--head requires --model-dir <DIR>").to_string();
    if gb10_inference::exl3::is_exl3_dir(&model_dir) || gb10_inference::fp4l::pack::is_fp4l_dir(std::path::Path::new(&model_dir)) {
        gb10_inference::fp4l::apply_pack_defaults(&model_dir);
        run_exl3_tp_head(args, &model_dir);
        return;
    }
    let prompt_text = match parse_arg(args, "--prompt-file") {
        Some(f) => std::fs::read_to_string(f).expect("read --prompt-file"),   // verbatim — `$(cat)` strips trailing newlines and changes the last token
        None => parse_arg(args, "--prompt").unwrap_or("The capital of France is").to_string(),
    };
    let max_new: usize = parse_arg(args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(64);
    let explicit = parse_arg(args, "--nodes").map(|s| {
        s.split(',').map(|p| {
            let p = p.trim();
            if p.contains(':') { p.parse::<std::net::SocketAddr>().expect("bad --nodes addr (ip:port)") }
            else { std::net::SocketAddr::new(p.parse::<std::net::IpAddr>().expect("bad --nodes ip"), 29500) }
        }).collect::<Vec<_>>()
    });
    let wait = std::time::Duration::from_secs(
        parse_arg(args, "--discover-wait").and_then(|s| s.parse().ok()).unwrap_or(3));
    // Snapshot our GB10_TP_* env as THE config, install it process-globally, and ship it to every
    // node during the sync (nodes run with zero TP env and reproduce this behavior).
    let mut tpc = gb10_inference::tp::TpConfig::from_opts();
    tpc.world = parse_tp_world(args).unwrap_or(2);   // --tp [N] is the single authority (bare = 2)
    tpc.dspark = args.iter().any(|a| a == "--bench-dspark" || a == "--dspark");
    // S4F DFlash2 tap capture: DEFAULT OFF (capture is a dead branch until the round turns on).
    tpc.df2_capture = matches!(parse_arg(args, "--df2-capture").unwrap_or("off"),
                               "on" | "true" | "1" | "yes");
    // DSV4 persistent server mode: --server-port → mode_serve (the node routes to the
    // persistent loop instead of the one-shot serve path). LEGACY ALIAS kept for compatibility —
    // the canonical interface is the standard `--server --tp --nodes --port` (run_dsv4_server_tp).
    // item 2.3: --prefix-cache <on|off> (AGENTS §7; rides TpConfig). Default ON for the dsv4
    // server (pre-flag behavior), OFF for qwen (unchanged).
    tpc.prefix_cache = matches!(parse_arg(args, "--prefix-cache")
        .unwrap_or(if is_dsv4_bundle(std::path::Path::new(&model_dir)) { "on" } else { "off" }),
        "on" | "true" | "1" | "yes");
    if parse_arg(args, "--server-port").is_some() {
        tpc.mode_serve = true;
        tpc.max_seq_len = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse::<usize>().ok()).unwrap_or(4096);
    }
    // item 3.4: --server-dspark <on|off> — DSpark speculation in the persistent --server-port
    // server (DEFAULT ON — user decision 2026-08-05; pass off for greedy; rides TpConfig to
    // the node).
    tpc.server_dspark = matches!(parse_arg(args, "--server-dspark").unwrap_or("on"),
                                 "on" | "true" | "1" | "yes");
    // item 1.7(i) / T3: --dspark-fp8-head <on|off> — the one-shot --gear/--dspark drafter's fp8
    // LM head + Markov W2 (DEFAULT ON — user decision 2026-08-05; pass off for bf16; rides
    // TpConfig for SPMD).
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
    let dspark = args.iter().any(|a| a == "--bench-dspark" || a == "--dspark");
    // item 3.3: --dspark-depth N pins the drafted-row count (None = adaptive). Rides TpConfig so
    // the zero-config node drafts the same width as the head (SPMD — a width mismatch diverges
    // the verify all-reduces). N=block is the adaptive-depth DISABLE path.
    tpc.dspark_depth = parse_arg(args, "--dspark-depth").and_then(|s| s.parse::<u32>().ok());
    if let Some(d) = tpc.dspark_depth {
        if !(1..=5).contains(&d) {
            eprintln!("--dspark-depth must be 1..=5 (the DSpark block size), got {d}");
            std::process::exit(1);
        }
    }
    gb10_inference::tp::set_tp_config(tpc.clone());
    // The DSpark SESSION keeps the control stream RETAINED (the head ships its measured r(D)
    // table over it — item 3.3; the --server-dspark server likewise — item 3.4).
    let retain_ctl = dspark || (tpc.mode_serve && tpc.server_dspark);
    let ctl_stream = if retain_ctl {
        match gb10_inference::cluster::run_head_session(std::path::Path::new(&model_dir), explicit, wait, &tpc) {
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
        match gb10_inference::cluster::run_head(std::path::Path::new(&model_dir), explicit, wait, &tpc) {
            Ok(nodes) => if gb10_inference::boot_report::debug() { println!("HEAD SYNCED {} node(s)", nodes.len()) },
            Err(e) => { eprintln!("head error: {e:#}"); std::process::exit(1); }
        }
        None
    };
    let serve_res = if is_dsv4_bundle(std::path::Path::new(&model_dir)) {
        eprintln!("[head] deepseek_v4 detected → Dsv4GpuModel TP=2 path");
        if dspark {
            // P4: DSV4 N-way deferred (still TP=2). Phase-5 DSpark speculation serve (both ranks
            // load the replicated dspark.safetensors).
            dsv4_tp_dspark_serve(&model_dir, gb10_inference::tp::TpContext::bring_up_head(2),
                                 Some((prompt_text, max_new)), ctl_stream)
        } else if let Some(port) = parse_arg(args, "--server-port") {
            // P4: DSV4 N-way deferred (still TP=2). Minimal /v1/chat/completions server (first-light,
            // §6b bypass): load + attach once, then serve one request over raw HTTP → broadcast →
            // SPMD greedy (or DSpark when --server-dspark on) → OpenAI JSON → exit.
            let port: u16 = port.parse().expect("--server-port N");
            let default_max_tokens = parse_arg(args, "--max-tokens").and_then(|s| s.parse::<usize>().ok()).unwrap_or(4096);
            dsv4_tp_serve_server(&model_dir, gb10_inference::tp::TpContext::bring_up_head(2), port, default_max_tokens, ctl_stream, &tpc)
        } else {
            // P4: DSV4 N-way deferred (still TP=2).
            dsv4_tp_serve(&model_dir, gb10_inference::tp::TpContext::bring_up_head(2),
                          Some((prompt_text, max_new)))
        }
    } else {
        tp_serve(&model_dir, gb10_inference::tp::TpContext::bring_up_head(tpc.world as i32),
                 Some((prompt_text, max_new)))
    };
    if let Err(e) = serve_res {
        eprintln!("head tp serve error: {e:#}"); std::process::exit(1);
    }
}
