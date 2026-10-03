//! `--tp-barrier-bench` — adversarial harness for the TP=2 doorbell all-reduce.
//!
//! Spec: `tp_doorbell_ref/BENCH_PLAN.md`. This runs on the REAL transport (same proxy, same K1/K2, no
//! model compute) and exists to prove the protocol correct under worst-case timing BEFORE the model
//! depends on it — a racy protocol hidden behind model noise is exactly how the previous attempt got to
//! "token-identical but 3× slower and occasionally wrong".
//!
//! Acceptance gates (all must hold before touching the model):
//!   * zero-spacing + poison + inject-delay(100 µs): **0 validation errors** over >= 1e6 barriers, both
//!     nodes. Zero-spacing is strictly denser than the 2-barrier attn cluster, so it covers the model.
//!   * `--stall-consumer-every` drives the ring to full: no deadlock, the reuse gate opens on a CQE.
//!     Nothing else in the bench reaches ring depth, so without this the I3/I4 gate ships UNTESTED.
//!   * tail-epoch guard fire count == **0** — the runtime proof that RC/PCIe placement ordering held,
//!     which is what makes the CPU-bounce receive sound given `CAN_FLUSH_REMOTE_WRITES = 0`.
//!
//! Percentiles, never means: the failure mode we are hunting (scheduler/IRQ jitter) is a tail.

use crate::net::{self, TpLink};
use cudarc::driver::{CudaDevice, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

pub struct BenchArgs {
    pub rank: i32,
    pub peer: String,
    pub port: u16,
    pub dev: String,
    pub gid: i32,
    /// TP rank count (power of two). 2 = the byte-identical legacy pair mode; >2 drives the
    /// recursive-doubling round schedule (transport invariants only — numerical equality is P5/P6).
    pub world: u32,
    /// Peer-IP list indexed by PEER RANK (P3: placeholder; P6 supplies the real topology). Empty means
    /// "derive a placeholder" (head IP for rank 0, loopback for the rest).
    pub peer_ips: Vec<String>,
    pub barriers: u64,
    pub payload_bytes: usize,
    pub spacing_us: u64,
    pub inject_delay_us_max: u32,
    pub poison: bool,
    pub stall_every: u32,
    pub stall_us: u64,
    pub window: u64,
    pub proxy_core: i32,
    pub main_core: i32,
    /// Withhold CQ retirement credit until this many epochs are outstanding. The ONLY mode that makes
    /// the I3 reuse gate bind: the bidirectional rendezvous bounds inter-node skew to ~1 barrier, so a
    /// consumer stall just slows both ranks symmetrically and never reaches ring depth.
    pub cq_hold: u32,
    /// How long to keep withholding once the threshold is reached, so the gate binds every cycle.
    pub cq_hold_us: u32,
}

/// Exact percentiles from the raw samples (we keep every sample — 8 B each, 1e6 barriers is 8 MB).
fn pct(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() { return 0; }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}
fn report(name: &str, samples: &mut Vec<u64>) {
    if samples.is_empty() { println!("  {name:<34} (no samples)"); return; }
    samples.sort_unstable();
    println!("  {name:<34} n={:<8} p50={:>8.2} p90={:>8.2} p99={:>8.2} p999={:>8.2} max={:>9.2}  (µs)",
             samples.len(),
             pct(samples, 0.50) as f64 / 1000.0, pct(samples, 0.90) as f64 / 1000.0,
             pct(samples, 0.99) as f64 / 1000.0, pct(samples, 0.999) as f64 / 1000.0,
             pct(samples, 1.0) as f64 / 1000.0);
}

/// Dump the per-barrier histograms collected during a MODEL run (--tp-trace=1), in the same shape
/// the microbench reports so the two are directly comparable. The rings hold the last GTS_EPOCHS
/// barriers, which is what we summarise.
/// Per-layer-TYPE cost split, from the barrier trace. The barriers tile the forward pass, so the gap
/// between consecutive barriers IS the compute between them — which makes the existing trace a free
/// per-layer-type profiler with no nsys and no GPU counters.
///
/// Barrier order within a forward is `[mixer, FFN]` per layer when the mixers are sharded, `[FFN]` only
/// otherwise. So the gap AFTER a mixer barrier is that layer's FFN compute, and the gap after an FFN
/// barrier is the NEXT layer's mixer compute — whose type (GDN vs full attention) we know from the config.
///
/// This exists to size the prize before anyone writes a fused GDN kernel: if the GDN chain is 100 µs/layer
/// the fusion is worth ~3-4 ms/token, and if it is 20 µs it is not worth the risk.
pub fn trace_layer_split(label: &str, layer_is_gdn: &[bool], mixer_sharded: bool) {
    let Some((gts, _cts, _c, _gw, dev_epoch)) = net::trace_data() else { return; };
    let nlayer = layer_is_gdn.len();
    let sites = if mixer_sharded { 2 * nlayer } else { nlayer };
    if sites == 0 || dev_epoch < (sites as u64) * 2 { return; }
    let n = (net::GTS_EPOCHS - sites).min(dev_epoch as usize - 1);
    let lo = dev_epoch - n as u64 + 1;

    let (mut ffn, mut gdn_mix, mut attn_mix) = (Vec::new(), Vec::new(), Vec::new());
    let mut prev: Option<(u64, usize)> = None;
    for ep in lo..=dev_epoch {
        let gi = (ep as usize % net::GTS_EPOCHS) * net::GTS_STRIDE;
        let k1_in = gts[gi + net::GTS_K1_IN];
        let idx = ((ep - 1) % sites as u64) as usize;
        if let Some((pt, pidx)) = prev {
            if k1_in > pt && idx == (pidx + 1) % sites {
                let gap = k1_in - pt;
                if !mixer_sharded {
                    ffn.push(gap);                       // FFN->FFN spans one whole layer
                } else if pidx % 2 == 0 {
                    ffn.push(gap);                       // after a mixer barrier => this layer's FFN
                } else {
                    let next_layer = ((pidx / 2) + 1) % nlayer;
                    if layer_is_gdn[next_layer] { gdn_mix.push(gap) } else { attn_mix.push(gap) }
                }
            }
        }
        prev = Some((k1_in, idx));
    }
    println!("\n=== [tp-trace] {label} — per-layer-type cost (gap between barriers = the compute between them) ===");
    let mut tot = 0f64;
    for (name, v, count) in [("FFN (all 64 layers)", &mut ffn, nlayer),
                             ("GDN mixer", &mut gdn_mix, layer_is_gdn.iter().filter(|x| **x).count()),
                             ("full-attn mixer", &mut attn_mix, layer_is_gdn.iter().filter(|x| !**x).count())] {
        if v.is_empty() { continue; }
        v.sort_unstable();
        let p50 = v[v.len() / 2] as f64 / 1000.0;
        let per_token = p50 * count as f64 / 1000.0;
        tot += per_token;
        println!("  {name:<22} p50 {p50:7.2} µs/layer × {count:2} layers = {per_token:6.2} ms/token   (n={})", v.len());
    }
    println!("  {:<22} {tot:29.2} ms/token accounted", "TOTAL");
}

pub fn trace_dump(label: &str) {
    let Some((gts, cts, (posted, retired, released, tail_fires), gate_waits, dev_epoch)) =
        net::trace_data() else { return; };
    let n = (dev_epoch.min(net::GTS_EPOCHS as u64 - 1)) as usize;
    if n < 8 { println!("[tp-trace] only {n} barriers — nothing to summarise"); return; }
    // Walk the most recent `n` epochs, skipping the one currently being overwritten.
    let lo = dev_epoch.saturating_sub(n as u64) + 1;
    let (mut s_signal, mut s_bounce, mut s_wait, mut s_barrier, mut s_gap) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut prev_k1in = 0u64;
    for ep in lo..=dev_epoch {
        let gi = (ep as usize % net::GTS_EPOCHS) * net::GTS_STRIDE;
        let ci = (ep as usize % net::GTS_EPOCHS) * net::CTS_STRIDE;
        let (k1_in, k1_out) = (gts[gi + net::GTS_K1_IN], gts[gi + net::GTS_K1_OUT]);
        let (k2_in, k2_go) = (gts[gi + net::GTS_K2_IN], gts[gi + net::GTS_K2_GO]);
        let (seen, rel) = (cts[ci + net::CTS_PEERSEEN], cts[ci + net::CTS_RELEASED]);
        if k1_out >= k1_in && k1_in > 0 { s_signal.push(k1_out - k1_in); }
        if rel >= seen && seen > 0 { s_bounce.push(rel - seen); }
        if k2_go >= k2_in && k2_in > 0 { s_wait.push(k2_go - k2_in); }
        if k2_go >= k1_in && k1_in > 0 { s_barrier.push(k2_go - k1_in); }
        if prev_k1in > 0 && k1_in > prev_k1in { s_gap.push(k1_in - prev_k1in); }
        prev_k1in = k1_in;
    }
    println!("\n=== [tp-trace] {label} — last {} barriers of {dev_epoch} ===", s_barrier.len());
    println!("  proxy: posted {posted} retired {retired} released {released} | tail fires {tail_fires} | reuse-gate binds {gate_waits}");
    report("K1 duration (copy+signal)", &mut s_signal);
    report("receive bounce (peer->cpu_done)", &mut s_bounce);
    report("K2 wait on cpu_done", &mut s_wait);
    report("whole barrier (K1in->K2go)", &mut s_barrier);
    report("barrier-to-barrier gap", &mut s_gap);
    let tot: u64 = s_barrier.iter().sum();
    let span: u64 = s_gap.iter().sum();
    if span > 0 {
        println!("  => barriers account for {:.1}% of wall time in the traced span \
                  ({:.2} ms of barrier per {:.2} ms)", 100.0 * tot as f64 / span as f64,
                 tot as f64 / 1e6, span as f64 / 1e6);
    }
}

/// P3 placeholder peer-IP list for the N-way bench (P6 supplies the real topology). Indexed by PEER
/// RANK: rank 0 = the head IP, everything else = loopback (the world>2 attach is still gated behind
/// `attach_tp`'s panic, so this path only exercises the QP bootstrap + round schedule transport).
fn bench_peer_ips(world: u32, head_ip: &str) -> Vec<std::net::IpAddr> {
    let mut v = Vec::with_capacity(world as usize);
    for r in 0..world as i32 {
        let ip = if r == 0 {
            head_ip.parse().unwrap_or_else(|_| "127.0.0.1".parse().unwrap())
        } else {
            "127.0.0.1".parse().unwrap()
        };
        v.push(ip);
    }
    v
}

pub fn run(a: BenchArgs) -> anyhow::Result<()> {
    println!("=== --tp-barrier-bench  rank {} ===", a.rank);
    println!("    barriers {}  payload {} B  window {}", a.barriers, a.payload_bytes, a.window);
    println!("    spacing {} µs  inject-delay-max {} µs  poison {}  stall every {} for {} µs",
             a.spacing_us, a.inject_delay_us_max, a.poison, a.stall_every, a.stall_us);

    let dev = CudaDevice::new(0)?;
    let ptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_batch.ptx")?);
    let names = ["tp_gate_copy_signal", "tp_wait_add", "tp_wait_add_4way", "tp_bench_fill", "tp_bench_validate",
                 "tp_bench_stall", "tp_bench_now", "kernel_build_id"];
    dev.load_ptx(ptx, "tpb", &names)?;
    let f = |n: &str| dev.get_func("tpb", n).ok_or_else(|| anyhow::anyhow!("missing kernel {n}"));
    let k1 = f("tp_gate_copy_signal")?;
    let (kfill, kval, kstall, know) = (f("tp_bench_fill")?, f("tp_bench_validate")?,
                                       f("tp_bench_stall")?, f("tp_bench_now")?);

    // Pin the launch thread to a big X925 core BEFORE the hot loop. A failure to pin invalidates the
    // measurement (jitter reads exactly like a protocol stall), so it is fatal, not a warning.
    if a.main_core >= 0 && !net::pin_thread(a.main_core) {
        anyhow::bail!("could not pin the bench thread to core {} — refusing to report numbers", a.main_core);
    }

    let link = if a.world > 2 {
        // P6 supplies the real peer-IP topology via `--peer`-rank list; P3 falls back to a placeholder
        // (head IP for rank 0, loopback for the rest) so the N-way QP bootstrap + round schedule run.
        let ips: Vec<std::net::IpAddr> = if a.peer_ips.len() >= a.world as usize {
            a.peer_ips[..a.world as usize].iter().map(|s| s.parse()).collect::<Result<_, _>>()?
        } else {
            bench_peer_ips(a.world, &a.peer)
        };
        TpLink::connect_nway(a.rank, a.world as i32, &ips, a.port, &a.dev, a.gid, crate::tp::TP_SLOT_BYTES)?
    } else {
        TpLink::connect(a.rank, &a.peer, a.port, &a.dev, a.gid, crate::tp::TP_SLOT_BYTES)?
    };
    let mut link = link;
    link.set_payload(a.payload_bytes, false)?;
    // P3-1: when the transport ctx selected the one-shot push (--tp-oneshot, world==4), K2 MUST
    // be tp_wait_add_4way (sender-indexed rings) — the v1 tp_wait_add would wait on round-keyed
    // slots the proxy never wrote. The ctx is the single source of truth; read it AFTER connect.
    let k2 = if link.oneshot_on() {
        // One-shot REQUIRES the v2 GPU-direct receive: the CPU proxy's RECV stage validates
        // round-keyed slots and would tail-guard-abort on sender-indexed traffic (observed:
        // TAIL-EPOCH GUARD FIRED on ranks 2/3 at epoch 1). The serving path pairs one-shot with
        // GPU-direct receive for the same reason.
        link.set_recv_mode(true)?;
        f("tp_wait_add_4way")?
    } else {
        f("tp_wait_add")?
    };
    link.bench_config(a.inject_delay_us_max, true);
    if a.cq_hold > 0 { link.bench_cq_hold(a.cq_hold, a.cq_hold_us)?; }
    println!("[bench] link up (world {}); proxy → core {}", a.world, a.proxy_core);

    // GPU<->CPU clock offset: %globaltimer and CLOCK_MONOTONIC_RAW have different epochs, so any
    // cross-domain stage delta is meaningless without this. Bracket a timestamp kernel and take the
    // tightest of several samples; the residual uncertainty is the bracket width, reported below.
    let now_buf = dev.alloc_zeros::<u64>(1)?;
    let (mut offset, mut best_span) = (0i64, u64::MAX);
    for _ in 0..32 {
        let t0 = std::time::Instant::now();
        let c0 = mono_ns();
        unsafe { know.clone().launch(LaunchConfig { grid_dim: (1,1,1), block_dim: (1,1,1), shared_mem_bytes: 0 }, (&now_buf,))?; }
        dev.synchronize()?;
        let c1 = mono_ns();
        let g = dev.dtoh_sync_copy(&now_buf)?[0];
        let span = c1 - c0;
        if span < best_span { best_span = span; offset = ((c0 + c1) / 2) as i64 - g as i64; }
        let _ = t0;
    }
    println!("[bench] GPU→CPU clock offset {offset} ns (bracket ±{} ns)", best_span / 2);

    let ctx_d = link.ctx_device_ptr();
    let n_elems = a.payload_bytes / 2;                       // bf16 elements
    let src = dev.alloc_zeros::<u8>(a.payload_bytes)?;
    let out = dev.alloc_zeros::<u8>(a.payload_bytes)?;
    let err = dev.alloc_zeros::<u64>(4)?;
    let cfg1 = LaunchConfig { grid_dim: (1,1,1), block_dim: (512,1,1), shared_mem_bytes: 0 };

    // The proxy owns the transport from here (I8: nothing else mutates protocol state). ManuallyDrop
    // rather than mem::forget so we keep a usable handle for the counters — TpLink::drop would
    // net_shutdown the ctx out from under the running proxy thread.
    let link = std::mem::ManuallyDrop::new(link);
    net::spawn_proxy(link.ctx_addr(), a.proxy_core);
    std::thread::sleep(std::time::Duration::from_millis(200));   // let both proxies settle

    let (mut s_signal, mut s_post, mut s_bounce, mut s_wait, mut s_barrier, mut s_gap) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());

    let t_start = std::time::Instant::now();
    let mut done: u64 = 0;
    while done < a.barriers {
        let win = a.window.min(a.barriers - done);
        for _ in 0..win {
            unsafe {
                // fill the payload for the NEXT epoch (self-describing: epoch + LFSR + checksum)
                kfill.clone().launch(cfg1, (ctx_d, &src))?;
                // consumer stall — the only thing that drives the ring to full (I3/I4 coverage)
                if a.stall_every > 0 {
                    kstall.clone().launch(cfg1, (ctx_d, a.stall_every, a.stall_us * 1000))?;
                }
                k1.clone().launch(cfg1, (ctx_d, &src, a.payload_bytes as u32))?;
                k2.clone().launch(cfg1, (ctx_d, &out, &src, n_elems as i32, 0i32))?;
                kval.clone().launch(cfg1, (ctx_d, &err, a.poison as i32))?;
                if a.spacing_us > 0 {
                    kstall.clone().launch(cfg1, (ctx_d, 1u32, a.spacing_us * 1000))?;
                }
            }
        }
        dev.synchronize()?;

        // fail fast: a validation error or a cooperative abort means the run is over, and the state
        // dump is the whole point of catching it here rather than at the end
        let e = dev.dtoh_sync_copy(&err)?;
        if e[0] != 0 {
            anyhow::bail!("VALIDATION FAILED after {} barriers: {} errors; first bad epoch {} word {:#x} got {:#x}",
                          done + win, e[0], e[1], e[2], e[3]);
        }
        if link.abort_status() != 0 {
            anyhow::bail!("protocol ABORT (status {}) after {} barriers; tail-guard fires {}",
                          link.abort_status(), done + win, link.tail_fires());
        }

        // Harvest the per-epoch timestamp rings for this window (window <= ring/2 so nothing wrapped).
        if let (Some(cts), Some(gts)) = (link.cpu_ts(), link.gpu_ts()) {
            let mut prev_k1in = 0u64;
            for i in 0..win {
                let ep = done + i + 1;                       // epochs are 1-based
                let ci = (ep as usize % net::GTS_EPOCHS) * net::CTS_STRIDE;
                let gi = (ep as usize % net::GTS_EPOCHS) * net::GTS_STRIDE;
                let (k1_in, k1_out) = (gts[gi + net::GTS_K1_IN], gts[gi + net::GTS_K1_OUT]);
                let (k2_in, k2_go) = (gts[gi + net::GTS_K2_IN], gts[gi + net::GTS_K2_GO]);
                let (ready, posted) = (cts[ci + net::CTS_READY], cts[ci + net::CTS_POSTED]);
                let (seen, rel) = (cts[ci + net::CTS_PEERSEEN], cts[ci + net::CTS_RELEASED]);

                // (a) GPU published the watermark -> local proxy observed it. Cross-domain: needs the
                //     offset. This is the "CPU scheduling / poll latency" stage.
                if k1_out > 0 && ready > 0 {
                    let g = k1_out as i64 + offset;
                    if ready as i64 > g { s_signal.push((ready as i64 - g) as u64); }
                }
                // (b) proxy observed -> ibv_post_send returned (intra-CPU)
                if posted >= ready && ready > 0 { s_post.push(posted - ready); }
                // (d) peer epoch observed -> cpu_done released (intra-CPU): the receive bounce
                if rel >= seen && seen > 0 { s_bounce.push(rel - seen); }
                // (e) GPU wait on cpu_done (intra-GPU): tracks wire RTT, must NOT track a sleep quantum
                if k2_go >= k2_in && k2_in > 0 { s_wait.push(k2_go - k2_in); }
                // (f) whole barrier, and (g) barrier-to-barrier spacing (intra-GPU)
                if k2_go >= k1_in && k1_in > 0 { s_barrier.push(k2_go - k1_in); }
                if prev_k1in > 0 && k1_in > prev_k1in { s_gap.push(k1_in - prev_k1in); }
                prev_k1in = k1_in;
            }
        }
        done += win;
        if done % (a.window * 64) == 0 {
            let el = t_start.elapsed().as_secs_f64();
            println!("[bench] {done}/{} barriers  {:.0} barriers/s  tail-guard fires {}",
                     a.barriers, done as f64 / el, link.tail_fires());
        }
    }

    let elapsed = t_start.elapsed();
    let (posted, retired, released, tail_fires) = link.counters();
    let (dev_epoch, gpu_ready) = (link.device_epoch(), link.gpu_ready());

    println!("\n=== results (rank {}) ===", a.rank);
    println!("  barriers {}  in {:.2}s  = {:.0} barriers/s ({:.2} µs/barrier mean wall)",
             done, elapsed.as_secs_f64(), done as f64 / elapsed.as_secs_f64(),
             elapsed.as_secs_f64() * 1e6 / done as f64);
    println!("  proxy: posted {posted}  retired {retired}  released {released}");
    println!("  device epoch {dev_epoch}  gpu_ready watermark {gpu_ready}");
    println!("  I3 reuse-gate binds: {} (cq-hold {})", link.gate_waits(), a.cq_hold);
    println!("\n  stage histograms (percentiles, not means):");
    report("(a) gpu_ready -> proxy saw it", &mut s_signal);
    report("(b) proxy saw -> post returned", &mut s_post);
    report("(d) peer epoch -> cpu_done", &mut s_bounce);
    report("(e) GPU wait on cpu_done", &mut s_wait);
    report("(f) whole barrier (K1in->K2go)", &mut s_barrier);
    report("(g) barrier-to-barrier gap", &mut s_gap);

    println!("\n  === GATES ===");
    let val_ok = dev.dtoh_sync_copy(&err)?[0] == 0;
    println!("  validation errors      : {}", if val_ok { "0            PASS" } else { "NONZERO      FAIL" });
    println!("  tail-epoch guard fires : {tail_fires}{}", if tail_fires == 0 { "            PASS" } else { "            FAIL" });
    println!("  abort status           : {}{}", link.abort_status(),
             if link.abort_status() == 0 { "            PASS" } else { "            FAIL" });
    println!("  epoch == gpu_ready     : {dev_epoch} == {gpu_ready}{}",
             if dev_epoch == gpu_ready { "   PASS" } else { "   FAIL (I8 tripwire)" });
    let ok = val_ok && tail_fires == 0 && link.abort_status() == 0 && dev_epoch == gpu_ready;
    println!("  OVERALL                : {}", if ok { "PASS" } else { "FAIL" });
    if !ok { anyhow::bail!("bench gates FAILED"); }
    Ok(())
}

fn mono_ns() -> u64 { net::now_ns() }

// ---------------------------------------------------------------------------------------------
// TP-F: `--tp-reduce-bench` — the prefill-sized fp32 all-reduce on the REAL transport (no model):
// times the serial (TP-B..E) and pipelined (TP-F) schedules of crate::tp_xport — the exact code the
// EXL3 engine launches — and validates EVERY reduce bitwise against the CPU's lower + upper fp32 sum.
// ---------------------------------------------------------------------------------------------

pub struct ReduceArgs {
    pub rank: i32,
    pub peer: String,
    pub port: u16,
    pub dev: String,
    pub gid: i32,
    /// second rail (TP-F dual rail): RoCE device for rail 1 ("" = single rail)
    pub dev2: String,
    pub floats: usize,
    pub reduces: usize,
    /// "serial" | "pipe"
    pub mode: String,
    pub lookahead: usize,
    pub blocks: u32,
    /// decode-sized serial reduces interleaved after every big one (mixed-sequence coverage)
    pub mix: usize,
    /// "f32" (in place) | "f16" (f16 partial in, f16 sum out, in place) | "f32to16" (fp32 in, f16 out)
    pub io: String,
    pub proxy_core: i32,
    pub proxy_core2: i32,
    pub main_core: i32,
}

fn fill_val(r: i32, i: usize, it: usize) -> f32 {
    let h = (i as u64).wrapping_mul(2654435761).wrapping_add((it as u64) * 40503 + (r as u64) * 7919) % 2_000_003;
    let v = (h as f32 - 1_000_001.0) * 1.37e-4;
    if (i + it) % 97 == 0 { v * 1e-6 } else { v }
}

pub fn run_reduce(a: ReduceArgs) -> anyhow::Result<()> {
    println!("=== --tp-reduce-bench rank {} mode {} floats {} ({:.1} MiB) reduces {} lookahead {} blocks {} mix {} rails {} ===",
             a.rank, a.mode, a.floats, a.floats as f64 * 4.0 / 1048576.0, a.reduces, a.lookahead, a.blocks, a.mix,
             if a.dev2.is_empty() { 1 } else { 2 });
    let dev = CudaDevice::new(0)?;
    let kern = crate::tp_xport::Kernels::load(&dev, "tprb")?;
    if a.main_core >= 0 && !net::pin_thread(a.main_core) {
        anyhow::bail!("could not pin the bench thread to core {}", a.main_core);
    }
    let mut link = TpLink::connect(a.rank, &a.peer, a.port, &a.dev, a.gid, crate::tp::TP_SLOT_BYTES)?;
    link.set_payload(2560 * 4, true)?;
    let ctx = link.ctx_device_ptr();
    let link = std::mem::ManuallyDrop::new(link);
    net::spawn_proxy(link.ctx_addr(), a.proxy_core);
    let mut rails = vec![ctx];
    let mut link2: Option<std::mem::ManuallyDrop<TpLink>> = None;
    if !a.dev2.is_empty() {
        let mut l2 = TpLink::connect(a.rank, &a.peer, a.port + 10, &a.dev2, a.gid, crate::tp::TP_SLOT_BYTES)?;
        l2.set_payload(2560 * 4, true)?;
        rails.push(l2.ctx_device_ptr());
        let l2 = std::mem::ManuallyDrop::new(l2);
        net::spawn_proxy_aux(l2.ctx_addr(), a.proxy_core2);
        link2 = Some(l2);
    }
    std::thread::sleep(std::time::Duration::from_millis(300));
    if a.mode.starts_with("dec") {
        return run_reduce_dec(&a, &dev, &kern, ctx, &link);
    }
    let pipe = crate::tp_xport::Pipe::new(&dev, rails, 2, a.lookahead, a.blocks)?;
    let stream = crate::gpu::fork_blocking_stream(&dev);   // AGENTS §2.1: blocking compute stream
    let n = a.floats;
    let mut buf = dev.alloc_zeros::<f32>(n)?;
    let dn = 16 * 2560;
    let mut small = dev.alloc_zeros::<f32>(dn)?;
    let mut host = vec![0f32; n];
    let mut hs = vec![0f32; dn];
    let (mut times, mut bad) = (Vec::with_capacity(a.reduces), 0usize);
    let blocking = &stream;
    let mut h16 = dev.alloc_zeros::<u16>(n)?;
    let mut hh = vec![0u16; n];
    let f16of = |v: f32| half::f16::from_f32(v);
    for it in 0..a.reduces {
        let t0;
        let got: Vec<f32>;
        let want = |i: usize| -> f32 {
            match a.io.as_str() {
                "f16" | "f16w" => f16of(f16of(fill_val(0, i, it)).to_f32() + f16of(fill_val(1, i, it)).to_f32()).to_f32(),
                "f32to16" => f16of(fill_val(0, i, it) + fill_val(1, i, it)).to_f32(),
                _ => fill_val(0, i, it) + fill_val(1, i, it),
            }
        };
        match a.io.as_str() {
            "f16" | "f16w" => {
                for (i, v) in hh.iter_mut().enumerate() { *v = f16of(fill_val(a.rank, i, it)).to_bits(); }
                dev.htod_sync_copy_into(&hh, &mut h16)?;
                dev.synchronize()?;
                let p16 = *h16.device_ptr() as u64;
                t0 = std::time::Instant::now();
                crate::tp_xport::reduce_pipe(&kern, blocking, &pipe, crate::tp_xport::Io { src: p16, src_f16: true, out: p16, out_f16: true, wire_f16: a.io == "f16w", mid: 0 }, n)?;
                dev.synchronize()?;
                times.push(t0.elapsed().as_nanos() as u64);
                got = dev.dtoh_sync_copy(&h16)?.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
            }
            "f32to16" => {
                for (i, v) in host.iter_mut().enumerate() { *v = fill_val(a.rank, i, it); }
                dev.htod_sync_copy_into(&host, &mut buf)?;
                dev.synchronize()?;
                let (p, p16) = (*buf.device_ptr() as u64, *h16.device_ptr() as u64);
                t0 = std::time::Instant::now();
                crate::tp_xport::reduce_pipe(&kern, blocking, &pipe, crate::tp_xport::Io { src: p, src_f16: false, out: p16, out_f16: true, wire_f16: false, mid: 0 }, n)?;
                dev.synchronize()?;
                times.push(t0.elapsed().as_nanos() as u64);
                got = dev.dtoh_sync_copy(&h16)?.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
            }
            _ => {
                for (i, v) in host.iter_mut().enumerate() { *v = fill_val(a.rank, i, it); }
                dev.htod_sync_copy_into(&host, &mut buf)?;
                dev.synchronize()?;
                let p = *buf.device_ptr() as u64;
                t0 = std::time::Instant::now();
                if a.mode == "pipe" && crate::tp_xport::pipe_ok(n) {
                    crate::tp_xport::reduce_pipe(&kern, blocking, &pipe, crate::tp_xport::Io::f32_inplace(p), n)?;
                } else {
                    crate::tp_xport::reduce_serial(&kern, blocking, ctx, p, n, 1)?;
                }
                dev.synchronize()?;
                times.push(t0.elapsed().as_nanos() as u64);
                got = dev.dtoh_sync_copy(&buf)?;
            }
        }
        for i in 0..n {
            let w = want(i);
            if got[i].to_bits() != w.to_bits() {
                if bad < 5 { println!("  MISMATCH reduce {it} elem {i}: got {:e} want {:e}", got[i], w); }
                bad += 1;
            }
        }
        for m in 0..a.mix {
            let w = 2560 * (1 + (it + m) % 16);
            for (i, v) in hs[..w].iter_mut().enumerate() { *v = fill_val(a.rank, i, it * 31 + m); }
            dev.htod_sync_copy_into(&hs, &mut small)?;
            dev.synchronize()?;
            crate::tp_xport::reduce_serial(&kern, blocking, ctx, *small.device_ptr() as u64, w, 1)?;
            dev.synchronize()?;
            let g = dev.dtoh_sync_copy(&small)?;
            for i in 0..w {
                let want = fill_val(0, i, it * 31 + m) + fill_val(1, i, it * 31 + m);
                if g[i].to_bits() != want.to_bits() { bad += 1; }
            }
        }
        if link.abort_status() != 0 { anyhow::bail!("transport ABORT status {} at reduce {it}", link.abort_status()); }
        if let Some(l2) = &link2 { if l2.abort_status() != 0 { anyhow::bail!("rail-2 ABORT status {} at reduce {it}", l2.abort_status()); } }
    }
    let mut s = times.clone();
    s.sort_unstable();
    let p50 = pct(&s, 0.5) as f64 / 1e3;
    println!("  reduce {:.1} MiB: p10 {:.0} p50 {:.0} p90 {:.0} max {:.0} us  => {:.2} GB/s per direction (p50), {:.1} ms per 96-reduce chunk",
             n as f64 * 4.0 / 1048576.0, pct(&s, 0.1) as f64 / 1e3, p50, pct(&s, 0.9) as f64 / 1e3, pct(&s, 1.0) as f64 / 1e3,
             n as f64 * 4.0 / (p50 * 1e3), p50 * 96.0 / 1e3);
    println!("  tail fires {}  gate binds {}  abort {}  device epoch {} gpu_ready {}",
             link.tail_fires(), link.gate_waits(), link.abort_status(), link.device_epoch(), link.gpu_ready());
    if let Some(l2) = &link2 {
        println!("  rail2: tail fires {}  abort {}  device epoch {} gpu_ready {}", l2.tail_fires(), l2.abort_status(), l2.device_epoch(), l2.gpu_ready());
    }
    println!("  VALIDATION: {} mismatching elements over {} reduces (+{} mixed small) => {}", bad, a.reduces, a.reduces * a.mix,
             if bad == 0 { "BITWISE_OK" } else { "FAIL" });
    if bad != 0 { anyhow::bail!("reduce bench validation FAILED"); }
    Ok(())
}

// TP-G: decode-sized all-reduces, 96 back-to-back per timed batch (one verify's barrier count), each
// with its own buffers; every result checked bitwise against f16(lower + upper).
//   mode "decold"  : xq_ks_combine_f32 (KS 1) -> serial K1/K2 (tp_xport::reduce_serial) -> xq_cvt_f32_f16
//                    (the TP-B..F decode sequence)
//   mode "decl"    : xq_ks_combine_f32_k1l (K1 folded into the last block) -> xq_tp_wait_add_dec -> f16
//   mode "declg"   : the same with the GPU-side receive (io bit 3)
//   "decmix1" old producer + old K1 + new K2; "decmix2" folded K1 + old K2 + cvt; "decmix3" old producer
//   + multi-block K1m + new K2 (the attribution ladder, TP-G.md §2)
fn run_reduce_dec(a: &ReduceArgs, dev: &std::sync::Arc<CudaDevice>, kern: &crate::tp_xport::Kernels, ctx: u64,
                  link: &TpLink) -> anyhow::Result<()> {
    const NB: usize = 96;
    let names = ["xq_ks_combine_f32", "xq_ks_combine_f32_k1l", "xq_tp_wait_add_dec", "xq_cvt_f32_f16"];
    let ptx = Ptx::from_src(std::fs::read_to_string(crate::exl3_bench::bench_ptx_path())?);
    dev.load_ptx(ptx, "tpgx", &names)?;
    let f = |n: &str| dev.get_func("tpgx", n).ok_or_else(|| anyhow::anyhow!("{n} missing"));
    let (kc, kcl, k2d, kcv) = (f(names[0])?, f(names[1])?, f(names[2])?, f(names[3])?);
    let k1m = f_k1m(dev)?;
    let stream = crate::gpu::fork_blocking_stream(dev);
    let n = a.floats;
    anyhow::ensure!(n % 4 == 0 && n * 4 <= 256 * 1024, "decode bench: floats must be a multiple of 4 and <= 64K");
    let mut src = dev.alloc_zeros::<f32>(NB * n)?;
    let part = dev.alloc_zeros::<f32>(NB * n)?;
    let out = dev.alloc_zeros::<u16>(NB * n)?;
    let mut arrive = dev.alloc_zeros::<u32>(1)?;
    dev.memset_zeros(&mut arrive)?;
    dev.synchronize()?;
    let arr = *arrive.device_ptr() as u64;
    let (ps, pp, po) = (*src.device_ptr() as u64, *part.device_ptr() as u64, *out.device_ptr() as u64);
    let io: i32 = 2 | if a.mode == "declg" { 8 } else { 0 };
    let g = ((n as u32).div_ceil(256), 1, 1);
    let mut host = vec![0f32; NB * n];
    let (mut times, mut bad) = (Vec::new(), 0usize);
    let reps = a.reduces.max(1);
    for it in 0..reps {
        for j in 0..NB { for i in 0..n { host[j * n + i] = fill_val(a.rank, i, it * NB + j); } }
        dev.htod_sync_copy_into(&host, &mut src)?;
        dev.synchronize()?;
        let t0 = std::time::Instant::now();
        for j in 0..NB {
            let (s, p, o) = (ps + (j * n * 4) as u64, pp + (j * n * 4) as u64, po + (j * n * 2) as u64);
            let c = LaunchConfig { grid_dim: g, block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                if a.mode == "decold" {
                    kc.clone().launch_on_stream(&stream, c, (s, p, 1i32, n as i32, 1i32))?;
                    crate::tp_xport::reduce_serial(kern, &stream, ctx, p, n, 1)?;
                    kcv.clone().launch_on_stream(&stream, c, (o, p, n as i64))?;
                } else if a.mode == "decmix1" {
                    // old producer + old single-block K1, new K2 (isolates the K2 side)
                    kc.clone().launch_on_stream(&stream, c, (s, p, 1i32, n as i32, 1i32))?;
                    crate::tp_xport::k1_serial(kern, &stream, ctx, p, n)?;
                    let c2 = LaunchConfig { grid_dim: (a.blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    k2d.clone().launch_on_stream(&stream, c2, (ctx, o, p, n as i32, io, 0u64, 0u32, 0u64, 0u32))?;
                } else if a.mode == "decmix3" {
                    // old producer + multi-block K1m (a.blocks), new K2
                    kc.clone().launch_on_stream(&stream, c, (s, p, 1i32, n as i32, 1i32))?;
                    let c2 = LaunchConfig { grid_dim: (a.blocks, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
                    k1m.clone().launch_on_stream(&stream, c2, (ctx, p, (n * 4) as u32, arr, 0i32))?;
                    let c3 = LaunchConfig { grid_dim: (a.blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    k2d.clone().launch_on_stream(&stream, c3, (ctx, o, p, n as i32, io, 0u64, 0u32, 0u64, 0u32))?;
                } else if a.mode == "decl" || a.mode == "declg" {
                    // v2 fold: producer + last-block K1, new K2
                    kcl.clone().launch_on_stream(&stream, c, (ctx, s, p, 1i32, n as i32, 1i32, arr))?;
                    let c2 = LaunchConfig { grid_dim: (a.blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    k2d.clone().launch_on_stream(&stream, c2, (ctx, o, p, n as i32, io, 0u64, 0u32, 0u64, 0u32))?;
                } else if a.mode == "decmix2" {
                    // folded producer+K1 (last block), old single-block K2 + cvt (isolates the K1 side)
                    kcl.clone().launch_on_stream(&stream, c, (ctx, s, p, 1i32, n as i32, 1i32, arr))?;
                    crate::tp_xport::k2_serial(kern, &stream, ctx, p, n)?;
                    kcv.clone().launch_on_stream(&stream, c, (o, p, n as i64))?;
                } else {
                    anyhow::bail!("decode bench mode {} unknown (decold | decl | declg | decmix1 | decmix2 | decmix3)", a.mode);
                }
            }
        }
        dev.synchronize()?;
        times.push(t0.elapsed().as_nanos() as u64 / NB as u64);
        let got = dev.dtoh_sync_copy(&out)?;
        for j in 0..NB {
            for i in 0..n {
                let w = half::f16::from_f32(fill_val(0, i, it * NB + j) + fill_val(1, i, it * NB + j)).to_bits();
                if got[j * n + i] != w {
                    if bad < 5 { println!("  MISMATCH rep {it} reduce {j} elem {i}: got {:04x} want {:04x}", got[j * n + i], w); }
                    bad += 1;
                }
            }
        }
        if link.abort_status() != 0 { anyhow::bail!("transport ABORT status {} at rep {it}", link.abort_status()); }
    }
    let mut s = times.clone();
    s.sort_unstable();
    println!("  decode reduce {} floats ({:.0} KiB) mode {}: per reduce p10 {:.2} p50 {:.2} p90 {:.2} us  => {:.2} ms per 96-barrier verify (p50)",
             n, n as f64 * 4.0 / 1024.0, a.mode, pct(&s, 0.1) as f64 / 1e3, pct(&s, 0.5) as f64 / 1e3, pct(&s, 0.9) as f64 / 1e3,
             pct(&s, 0.5) as f64 * 96.0 / 1e6);
    println!("  tail fires {}  gpu_rx_skips {}  abort {}  device epoch {} gpu_ready {}", link.tail_fires(),
             crate::net::traced_gpu_rx_skips(), link.abort_status(), link.device_epoch(), link.gpu_ready());
    println!("  VALIDATION: {} mismatching elements over {} x {} reduces => {}", bad, reps, NB, if bad == 0 { "BITWISE_OK" } else { "FAIL" });
    if bad != 0 { anyhow::bail!("decode reduce bench validation FAILED"); }
    Ok(())
}

fn f_k1m(dev: &std::sync::Arc<CudaDevice>) -> anyhow::Result<cudarc::driver::CudaFunction> {
    dev.get_func("tprb", "tp_gate_copy_signal_mb").ok_or_else(|| anyhow::anyhow!("tp_gate_copy_signal_mb missing"))
}

// ---------------------------------------------------------------------------------------------
// TP-4C: `--tp-reduce-bench --mode decw|decwf32|decwserial|decwkeys|decwmix` — the DECODE-path
// all-reduce at world 2 or 4 (rounds = log2(world) exchanges per logical reduce) on the REAL
// transport, no model. A SEPARATE function over `connect_nway` (the TP-G `run_reduce_dec` above is
// untouched and stays the world-2 pair bench). It launches exactly the sequence the engine launches
// (`tp_xport::dec_exchanges` / `dec_exchange_args` / `dec_intermediate`, the same kernels) and checks
// every result BITWISE against the host reference (`tp_xport::rd_tree_reference`: the recursive-doubling
// association the epoch phase fixes; keys: the max of the ranks' keys). The device epoch counter is read
// back after every timed batch and must have advanced by EXACTLY `rounds` per logical reduce — the
// hardware proof of the R10 epoch-phase discipline that the CPU model in tp_xport.rs only assumes.
//   decw       : producer with folded K1 (xq_ks_combine_f32_k1l) -> rounds x K2dec, fp32 intermediate in the
//                partial buffer, f16 out on the last round   (the mixer out-proj / MoE combine arm)
//   decwf32    : the same with an fp32 `out` that differs from the zero-free `local` (the draft-head screen
//                arm: the intermediate is `out`, `local` must survive unchanged)
//   decwserial : producer + reduce_serial(rounds) + f16 cvt (the TP-B..F serial sequence; the A/B for decw)
//   decwkeys   : vocab-parallel key merge — keys K1 (plain, every round) + xq_tp_wait_keys per round,
//                merged ids vs the host max (ties broken to the lowest id, the total order)
//   decwmix    : all four arms interleaved in a rotating order, every reduce a different shape: any arm
//                consuming a wrong number of epochs flips the round association of a LATER arm and fails
// ---------------------------------------------------------------------------------------------

pub struct DecWorldArgs {
    pub rank: i32,
    pub world: u32,
    /// peer IPs indexed by RANK (comma list on the CLI; entry `[rank]` is this node's own, unused)
    pub peer_ips: Vec<String>,
    pub port: u16,
    pub dev: String,
    pub gid: i32,
    pub floats: usize,
    /// timed batches (each 96 logical reduces per arm; the first batch is validated but excluded from the percentiles)
    pub reduces: usize,
    pub mode: String,
    pub blocks: u32,
    pub proxy_core: i32,
    pub main_core: i32,
}

/// A vocab-parallel argmax key (value in the high 32 bits, `0xFFFFFFFF - global_id` in the low 32): only a few
/// distinct values, so equal-value ties across ranks are frequent and the lowest-id tie-break is exercised.
fn dec_key(rank: i32, row: usize, it: usize) -> u64 {
    let mut z = (((it as u64) << 32) | row as u64).wrapping_add((rank as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let value = 1 + (z >> 32) % 5;
    let id = rank as u64 * 62_080 + (z & 0xFFFF_FFFF) % 62_080;
    (value << 32) | (0xFFFF_FFFFu64 - id)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DecArm { Dec16, Dec32, Serial, Keys }

pub fn run_reduce_dec_nway(a: DecWorldArgs) -> anyhow::Result<()> {
    if a.mode == "decwvpg" { return run_vp_gather_nway(a); }   // TP-4H2: the vocab-parallel row all-gather arm
    if a.mode == "decws" { return run_reduce_single_nway(a); } // TP-4F2: the single-stage vs rd decode reduce A/B
    const NB: usize = 96;
    const MKEYS: usize = 16;
    let active: Vec<DecArm> = match a.mode.as_str() {
        "decw" => vec![DecArm::Dec16],
        "decwf32" => vec![DecArm::Dec32],
        "decwserial" => vec![DecArm::Serial],
        "decwkeys" => vec![DecArm::Keys],
        "decwmix" => vec![DecArm::Dec16, DecArm::Dec32, DecArm::Serial, DecArm::Keys],
        m => anyhow::bail!("decode world bench mode {m} unknown (decw | decwf32 | decwserial | decwkeys | decwmix | decwvpg)"),
    };
    let world = a.world as usize;
    let rounds = crate::tp_xport::rounds_of(world)?;
    anyhow::ensure!(a.rank >= 0 && (a.rank as usize) < world, "--rank {} is outside world {world}", a.rank);
    anyhow::ensure!(a.peer_ips.len() >= world, "--peer-ips needs {world} rank-indexed IPs (got {})", a.peer_ips.len());
    let n = a.floats;
    anyhow::ensure!(n >= 4 && n % 4 == 0 && n * 4 <= 256 * 1024, "decode bench: floats must be a multiple of 4 and <= 64K");
    println!("=== --tp-reduce-bench rank {} world {} ({} exchange(s) per logical reduce) mode {} floats {} ({:.0} KiB) batches {} K2 blocks {} ===",
             a.rank, world, rounds, a.mode, n, n as f64 * 4.0 / 1024.0, a.reduces, a.blocks);
    let dev = CudaDevice::new(0)?;
    let kern = crate::tp_xport::Kernels::load(&dev, "tprb")?;
    let names = ["xq_ks_combine_f32", "xq_ks_combine_f32_k1l", "xq_tp_wait_add_dec", "xq_cvt_f32_f16", "xq_tp_wait_keys"];
    let ptx = Ptx::from_src(std::fs::read_to_string(crate::exl3_bench::bench_ptx_path())?);
    dev.load_ptx(ptx, "tpgw", &names)?;
    let f = |nm: &str| dev.get_func("tpgw", nm).ok_or_else(|| anyhow::anyhow!("{nm} missing"));
    let (kc, kcl, k2d, kcv, kkeys) = (f(names[0])?, f(names[1])?, f(names[2])?, f(names[3])?, f(names[4])?);
    if a.main_core >= 0 && !net::pin_thread(a.main_core) {
        anyhow::bail!("could not pin the bench thread to core {} — refusing to report numbers", a.main_core);
    }
    let ips: Vec<std::net::IpAddr> = a.peer_ips[..world].iter().map(|s| s.parse()).collect::<Result<_, _>>()?;
    let mut link = TpLink::connect_nway(a.rank, world as i32, &ips, a.port, &a.dev, a.gid, crate::tp::TP_SLOT_BYTES)?;
    link.set_payload(2560 * 4, true)?;
    let ctx = link.ctx_device_ptr();
    let link = std::mem::ManuallyDrop::new(link);
    net::spawn_proxy(link.ctx_addr(), a.proxy_core);
    std::thread::sleep(std::time::Duration::from_millis(300));
    let stream = crate::gpu::fork_blocking_stream(&dev);   // AGENTS §2.1: blocking compute stream

    // R10, exactly as the engine's attach does it: pad the device epoch counter to a multiple of `rounds`
    // with tiny serial exchanges (a fresh ctx is at 0, so the pad is 0 in practice).
    let mut zero4 = dev.alloc_zeros::<f32>(4)?;
    dev.memset_zeros(&mut zero4)?;
    dev.synchronize()?;
    let e_boot = link.device_epoch();
    let pad = ((rounds as u64 - e_boot % rounds as u64) % rounds as u64) as usize;
    for _ in 0..pad {
        crate::tp_xport::k1_serial(&kern, &stream, ctx, *zero4.device_ptr() as u64, 4)?;
        crate::tp_xport::k2_serial(&kern, &stream, ctx, *zero4.device_ptr() as u64, 4)?;
    }
    dev.synchronize()?;
    anyhow::ensure!(link.device_epoch() % rounds as u64 == 0, "round phase not aligned after padding (epoch {})", link.device_epoch());
    println!("[bench] link up (world {world}); device epoch {} -> {} ({pad} pad exchange(s)); first reduce runs round {} first",
             e_boot, link.device_epoch(), (link.device_epoch() + 1) % rounds as u64);

    // buffers (one slot per reduce of a batch: every reduce has its own operands)
    let mut s1 = dev.alloc_zeros::<f32>(NB * n)?;  let p1 = dev.alloc_zeros::<f32>(NB * n)?;  let o1 = dev.alloc_zeros::<u16>(NB * n)?;
    let mut s2 = dev.alloc_zeros::<f32>(NB * n)?;  let p2 = dev.alloc_zeros::<f32>(NB * n)?;  let o2 = dev.alloc_zeros::<f32>(NB * n)?;
    let mut s3 = dev.alloc_zeros::<f32>(NB * n)?;  let p3 = dev.alloc_zeros::<f32>(NB * n)?;  let o3 = dev.alloc_zeros::<u16>(NB * n)?;
    let mut kb = dev.alloc_zeros::<u64>(NB * MKEYS)?;
    let ki = dev.alloc_zeros::<i32>(NB * MKEYS)?;
    let mut arrive = dev.alloc_zeros::<u32>(1)?;
    dev.memset_zeros(&mut arrive)?;
    dev.synchronize()?;
    let arr = *arrive.device_ptr() as u64;
    let (ps1, pp1, po1) = (*s1.device_ptr() as u64, *p1.device_ptr() as u64, *o1.device_ptr() as u64);
    let (ps2, pp2, po2) = (*s2.device_ptr() as u64, *p2.device_ptr() as u64, *o2.device_ptr() as u64);
    let (ps3, pp3, po3) = (*s3.device_ptr() as u64, *p3.device_ptr() as u64, *o3.device_ptr() as u64);
    let (pkb, pki) = (*kb.device_ptr() as u64, *ki.device_ptr() as u64);
    // per-reduce shapes (the mix varies them; the single-arm modes use n / 16 rows throughout)
    let mix = a.mode == "decwmix";
    let ser_w = |j: usize| -> usize { if mix { (4 * (1 + (j * 131) % 1024)).min(n) } else { n } };
    let key_m = |j: usize| -> usize { if mix || a.mode == "decwkeys" { 1 + j % MKEYS } else { MKEYS } };
    let k2cfg = LaunchConfig { grid_dim: (a.blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let gcfg = |w: usize| LaunchConfig { grid_dim: ((w as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };

    // One decode-path logical reduce of arm `arm`, reduce slot `j`: the engine's launch sequence.
    let launch_arm = |arm: DecArm, j: usize| -> anyhow::Result<()> {
        match arm {
            DecArm::Dec16 | DecArm::Dec32 => {
                let f16o = arm == DecArm::Dec16;
                let (s, p, o) = if f16o { (ps1, pp1, po1) } else { (ps2, pp2, po2) };
                let (s, p, o) = (s + (j * n * 4) as u64, p + (j * n * 4) as u64, o + (j * n * if f16o { 2 } else { 4 }) as u64);
                unsafe { kcl.clone().launch_on_stream(&stream, gcfg(n), (ctx, s, p, 1i32, n as i32, 1i32, arr))?; }
                let inter = crate::tp_xport::dec_intermediate(p, o, f16o);
                for x in crate::tp_xport::dec_exchanges(rounds) {
                    if !x.folded_k1 { crate::tp_xport::k1_serial(&kern, &stream, ctx, inter, n)?; }
                    let k = crate::tp_xport::dec_exchange_args(&x, p, false, o, f16o);
                    let io = (k.local_f16 as i32) | ((k.out_f16 as i32) << 1);
                    unsafe { k2d.clone().launch_on_stream(&stream, k2cfg, (ctx, k.out, k.local, n as i32, io, 0u64, 0u32, 0u64, 0u32))?; }
                }
            }
            DecArm::Serial => {
                let w = ser_w(j);
                let (s, p, o) = (ps3 + (j * n * 4) as u64, pp3 + (j * n * 4) as u64, po3 + (j * n * 2) as u64);
                unsafe { kc.clone().launch_on_stream(&stream, gcfg(w), (s, p, 1i32, w as i32, 1i32))?; }
                crate::tp_xport::reduce_serial(&kern, &stream, ctx, p, w, rounds)?;
                unsafe { kcv.clone().launch_on_stream(&stream, gcfg(w), (o, p, w as i64))?; }
            }
            DecArm::Keys => {
                let m = key_m(j);
                let mpad = m + (m & 1);
                let nbytes = 8 * mpad;
                let (kp, ip) = (pkb + (j * MKEYS * 8) as u64, pki + (j * MKEYS * 4) as u64);
                for _x in crate::tp_xport::dec_exchanges(rounds) {
                    crate::tp_xport::k1_serial(&kern, &stream, ctx, kp, nbytes / 4)?;
                    let c = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
                    unsafe { kkeys.clone().launch_on_stream(&stream, c, (ctx, kp, ip, m as i32, nbytes as i32, 0i32))?; }
                }
            }
        }
        Ok(())
    };

    let f16b = |v: f32| half::f16::from_f32(v).to_bits();
    let reps = a.reduces.max(1);
    let (mut times, mut bad, mut checked) = (Vec::new(), 0usize, 0usize);
    let mut host = vec![0f32; NB * n];
    let mut hkeys = vec![0u64; NB * MKEYS];
    for it in 0..reps {
        let salt = |arm: usize, j: usize| it * NB + j + arm * 1_000_000;
        // ---- fill (host, untimed) ----
        for (arm, buf, sl) in [(0usize, &mut s1, DecArm::Dec16), (1, &mut s2, DecArm::Dec32), (2, &mut s3, DecArm::Serial)] {
            if !active.contains(&sl) { continue; }
            for j in 0..NB { for i in 0..n { host[j * n + i] = fill_val(a.rank, i, salt(arm, j)); } }
            dev.htod_sync_copy_into(&host, buf)?;
        }
        if active.contains(&DecArm::Keys) {
            hkeys.iter_mut().for_each(|k| *k = 0);
            for j in 0..NB { for r in 0..key_m(j) { hkeys[j * MKEYS + r] = dec_key(a.rank, r, salt(3, j)); } }
            dev.htod_sync_copy_into(&hkeys, &mut kb)?;
        }
        dev.synchronize()?;
        let e_start = link.device_epoch();
        anyhow::ensure!(e_start % rounds as u64 == 0, "round phase lost before batch {it}: epoch {e_start}");
        // ---- timed batch: NB logical reduces per active arm, rotating arm order ----
        let mut e0 = vec![vec![0u64; NB]; 4];
        let mut seq = 0u64;
        let t0 = std::time::Instant::now();
        for j in 0..NB {
            let mut order = active.clone();
            let rot = j % order.len();
            order.rotate_left(rot);
            for &arm in &order {
                e0[arm as usize][j] = e_start + seq;
                launch_arm(arm, j)?;
                seq += rounds as u64;
            }
        }
        dev.synchronize()?;
        times.push(t0.elapsed().as_nanos() as u64 / NB as u64);
        let e_end = link.device_epoch();
        anyhow::ensure!(e_end == e_start + seq,
            "EPOCH ACCOUNTING: batch {it} consumed {} epochs, expected {seq} ({} logical reduces x {rounds} rounds)",
            e_end - e_start, seq / rounds as u64);
        if link.abort_status() != 0 { anyhow::bail!("transport ABORT status {} at batch {it}", link.abort_status()); }
        // ---- validate every reduce bitwise against the host reference ----
        let parts = |arm: usize, j: usize, w: usize| -> Vec<Vec<f32>> {
            (0..world).map(|r| (0..w).map(|i| fill_val(r as i32, i, salt(arm, j))).collect()).collect()
        };
        if active.contains(&DecArm::Dec16) {
            let got = dev.dtoh_sync_copy(&o1)?;
            for j in 0..NB {
                let want = crate::tp_xport::rd_tree_reference(world, e0[0][j], &parts(0, j, n))?;
                for i in 0..n {
                    checked += 1;
                    if got[j * n + i] != f16b(want[i]) {
                        if bad < 5 { println!("  MISMATCH decw batch {it} reduce {j} elem {i}: got {:04x} want {:04x}", got[j * n + i], f16b(want[i])); }
                        bad += 1;
                    }
                }
            }
        }
        if active.contains(&DecArm::Dec32) {
            let got = dev.dtoh_sync_copy(&o2)?;
            let local = dev.dtoh_sync_copy(&p2)?;
            for j in 0..NB {
                let want = crate::tp_xport::rd_tree_reference(world, e0[1][j], &parts(1, j, n))?;
                for i in 0..n {
                    checked += 1;
                    if got[j * n + i].to_bits() != want[i].to_bits() {
                        if bad < 5 { println!("  MISMATCH decwf32 batch {it} reduce {j} elem {i}: got {:e} want {:e}", got[j * n + i], want[i]); }
                        bad += 1;
                    }
                    // the screen invariant: `local` (the producer's partial) is not the intermediate and survives
                    checked += 1;
                    if local[j * n + i].to_bits() != fill_val(a.rank, i, salt(1, j)).to_bits() {
                        if bad < 5 { println!("  MISMATCH decwf32 batch {it} reduce {j} elem {i}: LOCAL partial was overwritten"); }
                        bad += 1;
                    }
                }
            }
        }
        if active.contains(&DecArm::Serial) {
            let got = dev.dtoh_sync_copy(&o3)?;
            for j in 0..NB {
                let w = ser_w(j);
                let want = crate::tp_xport::rd_tree_reference(world, e0[2][j], &parts(2, j, w))?;
                for i in 0..w {
                    checked += 1;
                    if got[j * n + i] != f16b(want[i]) {
                        if bad < 5 { println!("  MISMATCH decwserial batch {it} reduce {j} elem {i}: got {:04x} want {:04x}", got[j * n + i], f16b(want[i])); }
                        bad += 1;
                    }
                }
            }
        }
        if active.contains(&DecArm::Keys) {
            let ids = dev.dtoh_sync_copy(&ki)?;
            let merged = dev.dtoh_sync_copy(&kb)?;
            for j in 0..NB {
                for r in 0..key_m(j) {
                    let mx = (0..world as i32).map(|rk| dec_key(rk, r, salt(3, j))).max().unwrap();
                    let want_id = (0xFFFF_FFFFu32 - (mx & 0xFFFF_FFFF) as u32) as i32;
                    checked += 1;
                    if ids[j * MKEYS + r] != want_id || merged[j * MKEYS + r] != mx {
                        if bad < 5 { println!("  MISMATCH decwkeys batch {it} reduce {j} row {r}: id {} want {want_id}, key {:016x} want {mx:016x}", ids[j * MKEYS + r], merged[j * MKEYS + r]); }
                        bad += 1;
                    }
                }
            }
        }
    }
    let mut s = if times.len() > 1 { times[1..].to_vec() } else { times.clone() };   // batch 0 = cold start, validated only
    s.sort_unstable();
    let per = |p: f64| pct(&s, p) as f64 / 1e3;
    let unit = if active.len() > 1 { format!("mixed round of {} arms", active.len()) } else { "logical reduce".to_string() };
    println!("  {} per {unit}: p10 {:.2} p50 {:.2} p90 {:.2} us  ({} exchange(s) each => {:.2} us per exchange at p50)",
             a.mode, per(0.1), per(0.5), per(0.9), rounds * active.len(), per(0.5) / (rounds * active.len()) as f64);
    println!("  tail fires {}  abort {}  device epoch {} gpu_ready {}", link.tail_fires(), link.abort_status(), link.device_epoch(), link.gpu_ready());
    println!("  EPOCH ACCOUNTING: every batch consumed exactly {rounds} epoch(s) per logical reduce => OK");
    println!("  VALIDATION: {bad} mismatching elements of {checked} checked over {reps} x {NB} x {} arm(s) => {}",
             active.len(), if bad == 0 && checked > 0 { "BITWISE_OK" } else { "FAIL" });
    // quiesce: one tiny serial reduce after the verdict so no rank leaves while a peer still waits on its last send
    for _ in 0..rounds {
        crate::tp_xport::k1_serial(&kern, &stream, ctx, *zero4.device_ptr() as u64, 4)?;
        crate::tp_xport::k2_serial(&kern, &stream, ctx, *zero4.device_ptr() as u64, 4)?;
    }
    dev.synchronize()?;
    std::thread::sleep(std::time::Duration::from_millis(300));
    if bad != 0 || checked == 0 { anyhow::bail!("decode world bench validation FAILED"); }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// TP-4F2: `--tp-reduce-bench --mode decws --world 4` — the SINGLE-STAGE decode reduce (`--tp-reduce single`) against
// the two-stage rd reduce, on the REAL transports, no model: ONE process per rank brings up the rd link (`--port`,
// proxy core `--proxy-core`) AND the dedicated uniform one-shot ctx (`--port` + 40, proxy core `--proxy-core` - 2,
// slot capacity 256 KiB as the engine builds it). Every iteration reduces the SAME per-rank partials through both
// arms (the engine's launch sequences: the producer-folded K1 + rounds x xq_tp_wait_add_dec, vs the folded K1 on the
// single ctx + ONE xq_tp_wait_add_dec_single), batches of 96 logical reduces timed per arm in an ABBA alternation
// (the first batch is validated, not timed). Checked on EVERY rank, bitwise: rd output == f16(rd_tree_reference),
// single output == f16(single_tree_reference), and single == rd element for element. The epoch counters must advance
// by exactly `rounds` per rd reduce and exactly 1 per single reduce (the single ctx has its own phase-free counter).
// ---------------------------------------------------------------------------------------------
fn run_reduce_single_nway(a: DecWorldArgs) -> anyhow::Result<()> {
    const NB: usize = 96;
    let world = a.world as usize;
    anyhow::ensure!(world == 4, "--mode decws is the world-4 single-stage bench (got --world {world})");
    let rounds = crate::tp_xport::rounds_of(world)?;
    anyhow::ensure!(a.rank >= 0 && (a.rank as usize) < world, "--rank {} is outside world {world}", a.rank);
    anyhow::ensure!(a.peer_ips.len() >= world, "--peer-ips needs {world} rank-indexed IPs (got {})", a.peer_ips.len());
    let n = a.floats;
    anyhow::ensure!(n >= 4 && n % 4 == 0 && n * 4 <= 256 * 1024, "decws bench: floats must be a multiple of 4 and <= 64K");
    println!("=== --tp-reduce-bench rank {} world {world} mode decws (rd: {rounds} exchanges; single: 1) floats {n} ({:.0} KiB) batches {} K2 blocks {} ===",
             a.rank, n as f64 * 4.0 / 1024.0, a.reduces, a.blocks);
    let dev = CudaDevice::new(0)?;
    let kern = crate::tp_xport::Kernels::load(&dev, "tprb")?;
    let names = ["xq_ks_combine_f32_k1l", "xq_tp_wait_add_dec", "xq_tp_wait_add_dec_single"];
    let ptx = Ptx::from_src(std::fs::read_to_string(crate::exl3_bench::bench_ptx_path())?);
    dev.load_ptx(ptx, "tpsw", &names)?;
    let f = |nm: &str| dev.get_func("tpsw", nm).ok_or_else(|| anyhow::anyhow!("{nm} missing"));
    let (kcl, k2d, k2s) = (f(names[0])?, f(names[1])?, f(names[2])?);
    if a.main_core >= 0 && !net::pin_thread(a.main_core) {
        anyhow::bail!("could not pin the bench thread to core {} — refusing to report numbers", a.main_core);
    }
    let ips: Vec<std::net::IpAddr> = a.peer_ips[..world].iter().map(|s| s.parse()).collect::<Result<_, _>>()?;
    // rail-1-style rd link first (every rank does the same sequence, so the handshakes pair), then the uniform ctx
    let mut link = TpLink::connect_nway(a.rank, world as i32, &ips, a.port, &a.dev, a.gid, crate::tp::TP_SLOT_BYTES)?;
    link.set_payload(2560 * 4, true)?;
    let ctx = link.ctx_device_ptr();
    let link = std::mem::ManuallyDrop::new(link);
    net::spawn_proxy(link.ctx_addr(), a.proxy_core);
    std::thread::sleep(std::time::Duration::from_millis(300));
    let sport = a.port + 40;
    let mut slink = TpLink::connect_nway_uniform(a.rank, world as i32, &ips, sport, &a.dev, a.gid, 256 * 1024)?;
    slink.set_payload(2560 * 4, true)?;
    let sctx = slink.ctx_device_ptr();
    let slink = std::mem::ManuallyDrop::new(slink);
    net::spawn_proxy_aux(slink.ctx_addr(), a.proxy_core - 2);
    std::thread::sleep(std::time::Duration::from_millis(300));
    let stream = crate::gpu::fork_blocking_stream(&dev);   // AGENTS §2.1: blocking compute stream

    // R10 on the rd link exactly as the engine's attach does it (a fresh ctx is at epoch 0: pad 0)
    let mut zero4 = dev.alloc_zeros::<f32>(4)?;
    dev.memset_zeros(&mut zero4)?;
    dev.synchronize()?;
    let e_boot = link.device_epoch();
    let pad = ((rounds as u64 - e_boot % rounds as u64) % rounds as u64) as usize;
    for _ in 0..pad {
        crate::tp_xport::k1_serial(&kern, &stream, ctx, *zero4.device_ptr() as u64, 4)?;
        crate::tp_xport::k2_serial(&kern, &stream, ctx, *zero4.device_ptr() as u64, 4)?;
    }
    dev.synchronize()?;
    anyhow::ensure!(link.device_epoch() % rounds as u64 == 0, "round phase not aligned after padding (epoch {})", link.device_epoch());
    println!("[bench] rd link + single-stage ctx up (world {world}); rd device epoch {} (pad {pad}), single ctx epoch {}",
             link.device_epoch(), slink.device_epoch());

    let mut ws = dev.alloc_zeros::<f32>(NB * n)?;
    let pa = dev.alloc_zeros::<f32>(NB * n)?;  let oa = dev.alloc_zeros::<u16>(NB * n)?;   // rd arm: partial, f16 out
    let ps = dev.alloc_zeros::<f32>(NB * n)?;  let os = dev.alloc_zeros::<u16>(NB * n)?;   // single arm
    let mut arrive = dev.alloc_zeros::<u32>(1)?;
    dev.memset_zeros(&mut arrive)?;
    dev.synchronize()?;
    let arr = *arrive.device_ptr() as u64;
    let (pws, ppa, poa, pps, pos) = (*ws.device_ptr() as u64, *pa.device_ptr() as u64, *oa.device_ptr() as u64,
                                     *ps.device_ptr() as u64, *os.device_ptr() as u64);
    let k2cfg = LaunchConfig { grid_dim: (a.blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let gcfg = LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };

    let launch_rd = |j: usize| -> anyhow::Result<()> {
        let (s, p, o) = (pws + (j * n * 4) as u64, ppa + (j * n * 4) as u64, poa + (j * n * 2) as u64);
        unsafe { kcl.clone().launch_on_stream(&stream, gcfg, (ctx, s, p, 1i32, n as i32, 1i32, arr))?; }
        let inter = crate::tp_xport::dec_intermediate(p, o, true);
        for x in crate::tp_xport::dec_exchanges(rounds) {
            if !x.folded_k1 { crate::tp_xport::k1_serial(&kern, &stream, ctx, inter, n)?; }
            let k = crate::tp_xport::dec_exchange_args(&x, p, false, o, true);
            let io = (k.local_f16 as i32) | ((k.out_f16 as i32) << 1);
            unsafe { k2d.clone().launch_on_stream(&stream, k2cfg, (ctx, k.out, k.local, n as i32, io, 0u64, 0u32, 0u64, 0u32))?; }
        }
        Ok(())
    };
    let launch_single = |j: usize| -> anyhow::Result<()> {
        let (s, p, o) = (pws + (j * n * 4) as u64, pps + (j * n * 4) as u64, pos + (j * n * 2) as u64);
        unsafe { kcl.clone().launch_on_stream(&stream, gcfg, (sctx, s, p, 1i32, n as i32, 1i32, arr))?; }
        unsafe { k2s.clone().launch_on_stream(&stream, k2cfg, (sctx, o, p, n as i32, 0u64, 0u32, 0u64, 0u32))?; }
        Ok(())
    };

    let f16b = |v: f32| half::f16::from_f32(v).to_bits();
    let reps = a.reduces.max(2);
    let (mut t_rd, mut t_si) = (Vec::new(), Vec::new());
    let (mut bad, mut checked, mut cross) = (0usize, 0usize, 0usize);
    let mut host = vec![0f32; NB * n];
    for it in 0..reps {
        let salt = |j: usize| it * NB + j;
        for j in 0..NB { for i in 0..n { host[j * n + i] = fill_val(a.rank, i, salt(j)); } }
        dev.htod_sync_copy_into(&host, &mut ws)?;
        dev.synchronize()?;
        let (e_rd0, e_si0) = (link.device_epoch(), slink.device_epoch());
        anyhow::ensure!(e_rd0 % rounds as u64 == 0, "rd round phase lost before batch {it}: epoch {e_rd0}");
        let rd_first = it % 2 == 0;      // ABBA: alternate which arm runs first in a batch pair
        let run_arm = |single: bool| -> anyhow::Result<u64> {
            let t0 = std::time::Instant::now();
            for j in 0..NB { if single { launch_single(j)?; } else { launch_rd(j)?; } }
            dev.synchronize()?;
            Ok(t0.elapsed().as_nanos() as u64 / NB as u64)
        };
        if rd_first { t_rd.push(run_arm(false)?); t_si.push(run_arm(true)?); }
        else { t_si.push(run_arm(true)?); t_rd.push(run_arm(false)?); }
        let (e_rd1, e_si1) = (link.device_epoch(), slink.device_epoch());
        anyhow::ensure!(e_rd1 == e_rd0 + (NB * rounds) as u64, "EPOCH ACCOUNTING (rd): batch {it} consumed {} epochs, expected {}", e_rd1 - e_rd0, NB * rounds);
        anyhow::ensure!(e_si1 == e_si0 + NB as u64, "EPOCH ACCOUNTING (single): batch {it} consumed {} epochs, expected {NB}", e_si1 - e_si0);
        if link.abort_status() != 0 || slink.abort_status() != 0 {
            anyhow::bail!("transport ABORT (rd {}, single {}) at batch {it}", link.abort_status(), slink.abort_status());
        }
        // ---- validate every reduce bitwise (on this rank) ----
        let got_rd = dev.dtoh_sync_copy(&oa)?;
        let got_si = dev.dtoh_sync_copy(&os)?;
        for j in 0..NB {
            let parts: Vec<Vec<f32>> = (0..world).map(|r| (0..n).map(|i| fill_val(r as i32, i, salt(j))).collect()).collect();
            // the rd arm of reduce j ran as the (j)th rd reduce of the batch: start epoch e_rd0 + j * rounds (even)
            let want_rd = crate::tp_xport::rd_tree_reference(world, e_rd0 + (j * rounds) as u64, &parts)?;
            let want_si = crate::tp_xport::single_tree_reference(&parts)?;
            for i in 0..n {
                checked += 2;
                if got_rd[j * n + i] != f16b(want_rd[i]) {
                    if bad < 5 { println!("  MISMATCH rd batch {it} reduce {j} elem {i}: got {:04x} want {:04x}", got_rd[j * n + i], f16b(want_rd[i])); }
                    bad += 1;
                }
                if got_si[j * n + i] != f16b(want_si[i]) {
                    if bad < 5 { println!("  MISMATCH single batch {it} reduce {j} elem {i}: got {:04x} want {:04x}", got_si[j * n + i], f16b(want_si[i])); }
                    bad += 1;
                }
                if got_si[j * n + i] != got_rd[j * n + i] { cross += 1; }
            }
        }
    }
    let stat = |t: &Vec<u64>| { let mut s = if t.len() > 1 { t[1..].to_vec() } else { t.clone() }; s.sort_unstable(); s };
    let (sr, ss) = (stat(&t_rd), stat(&t_si));
    let per = |s: &Vec<u64>, p: f64| pct(s, p) as f64 / 1e3;
    println!("  rd     per logical reduce ({rounds} exchanges): p10 {:.2} p50 {:.2} p90 {:.2} us", per(&sr, 0.1), per(&sr, 0.5), per(&sr, 0.9));
    println!("  single per logical reduce (1 exchange)  : p10 {:.2} p50 {:.2} p90 {:.2} us   => delta p50 {:+.2} us per reduce ({:+.2} ms per 96-reduce round)",
             per(&ss, 0.1), per(&ss, 0.5), per(&ss, 0.9), per(&ss, 0.5) - per(&sr, 0.5), (per(&ss, 0.5) - per(&sr, 0.5)) * 96.0 / 1e3);
    println!("  rd link: tail fires {} abort {} epoch {}   single ctx: tail fires {} abort {} epoch {} gpu_ready {}",
             link.tail_fires(), link.abort_status(), link.device_epoch(), slink.tail_fires(), slink.abort_status(), slink.device_epoch(), slink.gpu_ready());
    println!("  EPOCH ACCOUNTING: rd consumed exactly {rounds} epochs and single exactly 1 per logical reduce => OK");
    println!("  VALIDATION: {bad} mismatching elements of {checked} checked (rd vs rd_tree_reference, single vs single_tree_reference) over {reps} x {NB} reduces; \
              single-vs-rd cross mismatches {cross} => {}", if bad == 0 && cross == 0 && checked > 0 { "BITWISE_OK" } else { "FAIL" });
    // quiesce so no rank leaves while a peer still waits on its last send
    for _ in 0..rounds {
        crate::tp_xport::k1_serial(&kern, &stream, ctx, *zero4.device_ptr() as u64, 4)?;
        crate::tp_xport::k2_serial(&kern, &stream, ctx, *zero4.device_ptr() as u64, 4)?;
    }
    dev.synchronize()?;
    std::thread::sleep(std::time::Duration::from_millis(300));
    if bad != 0 || cross != 0 || checked == 0 { anyhow::bail!("single-stage decode bench validation FAILED"); }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// TP-4H2: `--tp-reduce-bench --mode decwvpg --world 4` — the WORLD-4 vocab-parallel row all-gather
// (`xq_vp_gather4_k1/k2`, the sampled / penalized / ratio-rule tail of `--tp-vp-sampled`) on the REAL
// transport, no model. It launches exactly the engine's sequence (`tp_vpgather::gather4_plan`, the same
// kernels, the same arguments as `vp_gather_rows_w4`) over synthetic, POSITION-DEPENDENT shard rows
// (`tp_vpgather::ref_half`, salted per gather so a stale ring slot can never match) and checks, on EVERY
// rank, that every assembled logits row is bit for bit the four shards in place. No arithmetic is involved
// (ks = 1: the gather is a bit copy), so the reference is exact equality. The device epoch counter is read
// back after every timed batch and must have advanced by EXACTLY `epochs_per_gather(m)` per gather (plus
// `rounds` per interleaved serial exchange in batch 0) — the hardware proof of the R10 phase discipline for
// this arm. Batch 0 of every width is validated but excluded from the percentiles; it also interleaves tiny
// serial exchanges between gathers (the decode arms' phase) so a gather that consumed a wrong number of
// epochs corrupts a LATER gather's partner schedule and fails.
// Widths: 1 2 3 4 5 8 9 12 13 16 (every group count and both partial-group shapes). Per gather: NG = 8
// gathers per timed batch, each with its own shard and logits operands.
// ---------------------------------------------------------------------------------------------
pub fn run_vp_gather_nway(a: DecWorldArgs) -> anyhow::Result<()> {
    use crate::tp_vpgather as g;
    const NSH: usize = 62_080;
    const V: usize = NSH * g::WORLD;
    const NG: usize = 8;
    const WIDTHS: [usize; 10] = [1, 2, 3, 4, 5, 8, 9, 12, 13, 16];
    let world = a.world as usize;
    anyhow::ensure!(world == g::WORLD, "--mode decwvpg is the world-4 row all-gather bench (got --world {world})");
    let rounds = crate::tp_xport::rounds_of(world)?;
    anyhow::ensure!(rounds == g::ROUNDS, "rounds {rounds} != {}", g::ROUNDS);
    anyhow::ensure!(a.rank >= 0 && (a.rank as usize) < world, "--rank {} is outside world {world}", a.rank);
    anyhow::ensure!(a.peer_ips.len() >= world, "--peer-ips needs {world} rank-indexed IPs (got {})", a.peer_ips.len());
    anyhow::ensure!(g::slot_fits(g::GROUP_ROWS, NSH, crate::tp::TP_SLOT_BYTES), "a 4-row group does not fit one ring slot");
    let rank = a.rank as usize;
    println!("=== --tp-reduce-bench rank {} world {} ({} exchange(s) per 4-row group) mode {} n_sh {} V {} gathers/batch {} batches {} ===",
             a.rank, world, rounds, a.mode, NSH, V, NG, a.reduces);
    let dev = CudaDevice::new(0)?;
    let kern = crate::tp_xport::Kernels::load(&dev, "tprb")?;
    let names = ["xq_vp_gather4_k1", "xq_vp_gather4_k2"];
    let ptx = Ptx::from_src(std::fs::read_to_string(crate::exl3_bench::bench_ptx_path())?);
    dev.load_ptx(ptx, "tpvg", &names)?;
    let f = |nm: &str| dev.get_func("tpvg", nm).ok_or_else(|| anyhow::anyhow!("{nm} missing"));
    let (k1, k2) = (f(names[0])?, f(names[1])?);
    if a.main_core >= 0 && !net::pin_thread(a.main_core) {
        anyhow::bail!("could not pin the bench thread to core {} — refusing to report numbers", a.main_core);
    }
    let ips: Vec<std::net::IpAddr> = a.peer_ips[..world].iter().map(|s| s.parse()).collect::<Result<_, _>>()?;
    let mut link = TpLink::connect_nway(a.rank, world as i32, &ips, a.port, &a.dev, a.gid, crate::tp::TP_SLOT_BYTES)?;
    link.set_payload(2560 * 4, true)?;
    let ctx = link.ctx_device_ptr();
    let link = std::mem::ManuallyDrop::new(link);
    net::spawn_proxy(link.ctx_addr(), a.proxy_core);
    std::thread::sleep(std::time::Duration::from_millis(300));
    let stream = crate::gpu::fork_blocking_stream(&dev);   // AGENTS §2.1: blocking compute stream

    // R10, exactly as the engine's attach does it (a fresh ctx is at 0, so the pad is 0 in practice)
    let mut zero4 = dev.alloc_zeros::<f32>(4)?;
    dev.memset_zeros(&mut zero4)?;
    dev.synchronize()?;
    let zp = *zero4.device_ptr() as u64;
    let e_boot = link.device_epoch();
    let pad = ((rounds as u64 - e_boot % rounds as u64) % rounds as u64) as usize;
    for _ in 0..pad {
        crate::tp_xport::k1_serial(&kern, &stream, ctx, zp, 4)?;
        crate::tp_xport::k2_serial(&kern, &stream, ctx, zp, 4)?;
    }
    dev.synchronize()?;
    anyhow::ensure!(link.device_epoch() % rounds as u64 == 0, "round phase not aligned after padding (epoch {})", link.device_epoch());
    println!("[bench] link up (world {world}); device epoch {} -> {} ({pad} pad exchange(s)); first gather runs round {} first",
             e_boot, link.device_epoch(), (link.device_epoch() + 1) % rounds as u64);
    let mut arrive = dev.alloc_zeros::<u32>(1)?;
    dev.memset_zeros(&mut arrive)?;
    dev.synchronize()?;
    let arr = *arrive.device_ptr() as u64;

    let reps = a.reduces.max(2);
    let (mut bad, mut checked) = (0usize, 0usize);
    let mut summary: Vec<(usize, usize, f64, f64, f64)> = Vec::new();
    for (mi, &m) in WIDTHS.iter().enumerate() {
        let eps = g::epochs_per_gather(m);
        let plan = g::gather4_plan(m, NSH);
        let mut shard = dev.alloc_zeros::<u16>(NG * m * NSH)?;
        let mut logits = dev.alloc_zeros::<u16>(NG * m * V)?;
        let (psh, plg) = (*shard.device_ptr() as u64, *logits.device_ptr() as u64);
        let poison = vec![0x7E7Eu16; NG * m * V];
        let mut hsh = vec![0u16; NG * m * NSH];
        let mut times: Vec<u64> = Vec::new();
        // the synthetic row id of (batch it, gather j, row r): unique across every width and batch
        let rowid = |it: usize, j: usize, r: usize| ((mi * reps + it) * NG + j) * g::MAX_ROWS + r;
        for it in 0..reps {
            // ---- fill (host, untimed): this rank's shard rows; poison every logits column ----
            for j in 0..NG {
                for r in 0..m {
                    let row = rowid(it, j, r);
                    let o = (j * m + r) * NSH;
                    for c in 0..NSH { hsh[o + c] = g::ref_half(rank, row, rank * NSH + c); }
                }
            }
            dev.htod_sync_copy_into(&hsh, &mut shard)?;
            dev.htod_sync_copy_into(&poison, &mut logits)?;
            dev.synchronize()?;
            let e_start = link.device_epoch();
            anyhow::ensure!(e_start % rounds as u64 == 0, "round phase lost before width {m} batch {it}: epoch {e_start}");
            // ---- timed batch: NG gathers back to back (batch 0 interleaves serial exchanges: untimed) ----
            let inter = it == 0;
            let mut extra = 0u64;
            let t0 = std::time::Instant::now();
            for j in 0..NG {
                let (sb, lb) = (psh + (j * m * NSH * 2) as u64, plg + (j * m * V * 2) as u64);
                for la in &plan {
                    let cfg = LaunchConfig { grid_dim: (la.blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    match la.half {
                        g::Half::K1 => unsafe {
                            k1.clone().launch_on_stream(&stream, cfg,
                                (ctx, sb, lb, V as i64, NSH as i32, la.r0 as i32, la.nr as i32, la.stage as i32, arr))?;
                        },
                        g::Half::K2 => unsafe {
                            k2.clone().launch_on_stream(&stream, cfg,
                                (ctx, lb, V as i64, NSH as i32, la.r0 as i32, la.nr as i32, la.stage as i32, la.lag, 0i32))?;
                        },
                    }
                }
                if inter && j % 2 == 1 {
                    for _ in 0..rounds {
                        crate::tp_xport::k1_serial(&kern, &stream, ctx, zp, 4)?;
                        crate::tp_xport::k2_serial(&kern, &stream, ctx, zp, 4)?;
                    }
                    extra += rounds as u64;
                }
            }
            dev.synchronize()?;
            if !inter { times.push(t0.elapsed().as_nanos() as u64 / NG as u64); }
            let want = (NG * eps) as u64 + extra;
            let e_end = link.device_epoch();
            anyhow::ensure!(e_end == e_start + want,
                "EPOCH ACCOUNTING: width {m} batch {it} consumed {} epochs, expected {want} ({NG} gathers x {eps} + {extra} interleaved)",
                e_end - e_start);
            if link.abort_status() != 0 { anyhow::bail!("transport ABORT status {} at width {m} batch {it}", link.abort_status()); }
            // ---- validate: every assembled row is the four shards in place, bit for bit ----
            let got = dev.dtoh_sync_copy(&logits)?;
            let mut row_bad = 0usize;
            for j in 0..NG {
                for r in 0..m {
                    let row = rowid(it, j, r);
                    let o = (j * m + r) * V;
                    for s in 0..world {
                        for c in 0..NSH {
                            checked += 1;
                            let w = g::ref_half(s, row, s * NSH + c);
                            if got[o + s * NSH + c] != w {
                                if bad + row_bad < 5 {
                                    println!("  MISMATCH decwvpg width {m} batch {it} gather {j} row {r} shard {s} col {c}: got {:04x} want {w:04x}",
                                             got[o + s * NSH + c]);
                                }
                                row_bad += 1;
                            }
                        }
                    }
                }
            }
            bad += row_bad;
        }
        let mut s = times.clone();   // batch 0 (cold, interleaved) is validated only
        s.sort_unstable();
        let per = |p: f64| pct(&s, p) as f64 / 1e3;
        println!("  decwvpg m={m:<2}: {} group(s), {eps} epoch(s) per gather, payload {} B (+8 tail) per last-stage epoch: \
                  p10 {:.1} p50 {:.1} p90 {:.1} us per gather ({:.1} us per epoch at p50)",
                 m.div_ceil(g::GROUP_ROWS), g::payload_bytes(g::GROUP_ROWS.min(m), NSH, 1), per(0.1), per(0.5), per(0.9), per(0.5) / eps as f64);
        summary.push((m, eps, per(0.1), per(0.5), per(0.9)));
    }
    println!("  VPG_US (p50 us per gather): {}",
             summary.iter().map(|&(m, _, _, p50, _)| format!("m={m}:{p50:.1}")).collect::<Vec<_>>().join(" "));
    println!("  tail fires {}  abort {}  device epoch {} gpu_ready {}", link.tail_fires(), link.abort_status(), link.device_epoch(), link.gpu_ready());
    println!("  EPOCH ACCOUNTING: every batch consumed exactly epochs_per_gather(m) x gathers (+ rounds per interleaved serial exchange) => OK");
    println!("  VALIDATION: {bad} mismatching elements of {checked} checked over {reps} batches x {NG} gathers x widths {:?} => {}",
             WIDTHS, if bad == 0 && checked > 0 { "BITWISE_OK" } else { "FAIL" });
    // quiesce: one tiny serial reduce after the verdict so no rank leaves while a peer still waits on its last send
    for _ in 0..rounds {
        crate::tp_xport::k1_serial(&kern, &stream, ctx, zp, 4)?;
        crate::tp_xport::k2_serial(&kern, &stream, ctx, zp, 4)?;
    }
    dev.synchronize()?;
    std::thread::sleep(std::time::Duration::from_millis(300));
    if bad != 0 || checked == 0 { anyhow::bail!("world-4 row all-gather bench validation FAILED"); }
    Ok(())
}

#[cfg(test)]
mod dec_world_tests {
    use super::*;

    /// The key generator the W=4 key-merge bench validates against must actually exercise the tie-break
    /// (equal values on several ranks, lowest id wins) and never hand two ranks the same key.
    #[test]
    fn dec_key_ties_are_frequent_and_keys_distinct() {
        let (mut ties, mut rows) = (0usize, 0usize);
        for it in 0..50 {
            for row in 0..64 {
                let ks: Vec<u64> = (0..4).map(|r| dec_key(r, row, it)).collect();
                for i in 0..4 { for j in i + 1..4 { assert_ne!(ks[i], ks[j], "ranks {i},{j} row {row} it {it}"); } }
                let top = ks.iter().map(|k| k >> 32).max().unwrap();
                if ks.iter().filter(|k| **k >> 32 == top).count() > 1 { ties += 1; }
                // the decoded id of the max key is a valid global id (< 4 * 62080)
                let mx = *ks.iter().max().unwrap();
                assert!(0xFFFF_FFFFu32 - (mx & 0xFFFF_FFFF) as u32 <= 4 * 62_080);
                rows += 1;
            }
        }
        assert!(ties * 4 >= rows, "only {ties}/{rows} rows had a top-value tie: the lowest-id tie-break is not exercised");
        assert_eq!(dec_key(1, 7, 3), dec_key(1, 7, 3), "deterministic");
    }

    /// The max of the keys is the lowest-id holder of the top value (the total order the engine relies on).
    #[test]
    fn dec_key_max_is_value_desc_then_lowest_id() {
        for it in 0..20 {
            for row in 0..64 {
                let ks: Vec<(u64, u32)> = (0..4).map(|r| {
                    let k = dec_key(r, row, it);
                    (k >> 32, 0xFFFF_FFFFu32 - (k & 0xFFFF_FFFF) as u32)
                }).collect();
                let top = ks.iter().map(|k| k.0).max().unwrap();
                let lowest = ks.iter().filter(|k| k.0 == top).map(|k| k.1).min().unwrap();
                let mx = (0..4).map(|r| dec_key(r, row, it)).max().unwrap();
                assert_eq!(0xFFFF_FFFFu32 - (mx & 0xFFFF_FFFF) as u32, lowest, "row {row} it {it}");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// TP-4E: `--tp-reduce-bench --world 4` — the PREFILL-sized fp32 all-reduce at world > 2 on the REAL
// transport (no model), the same `crate::tp_xport` code the EXL3 engine launches. Additive: nothing above
// this line changed, and world 2 never reaches it (`run_tp_reduce_bench` dispatches on --world > 2).
//
// One process per rank, the same command on every box, `--peer-ips` rank-indexed. Per payload of the sweep
// (2 / 8 / 21 / 42 MB of fp32, whole 2560-float rows; 21 MB is the 2,048-row prefill chunk) it runs
// `--warmup` + `--reduces` reduces on transport `--xport 0|1|2` (0 serial single-block, 1 pipelined on one
// rail, 2 pipelined on both rails, rail 2 being a full N-way link), validates EVERY reduce bitwise against a
// host reference (the recursive-doubling tree at the R10-aligned phase), and reports per-reduce microseconds
// and GB/s per direction. Timing starts after a small serial all-reduce that re-aligns the ranks (host
// checks between reduces skew them), so a slow peer's host work is not billed to the transport.
//
// What it does NOT prove: a thermally/contention-loaded or model-coexisting rate (no model is resident), the
// in-model overlap hooks (no consumer), or serving (no sequence-parallel, no MoE). A throughput taken here is
// a transport number, not a prefill tok/s.
// ---------------------------------------------------------------------------------------------

pub struct ReduceW4Args {
    pub rank: i32,
    pub world: i32,
    /// rank-indexed RoCE/management IPs of all `world` ranks (both rails' TCP handshakes use them)
    pub peer_ips: Vec<String>,
    /// rail-1 control port; rail 2 takes `xtp::rail2_nway_base_for(world, port)`
    pub port: u16,
    pub dev: String,
    /// rail-2 device ("" = the engine default roceP2p1s0f1); used only with `xport == 2`
    pub dev2: String,
    pub gid: i32,
    /// 0 = serial single-block pairs, 1 = pipelined one rail, 2 = pipelined both rails
    pub xport: u8,
    pub sizes_mb: Vec<u64>,
    pub reduces: usize,
    pub warmup: usize,
    /// "f32" (in place) | "f32to16" (fp32 in, f16 out, the stage is the consumed src) |
    /// "f16" (f16 in/out in place, separate fp32 stage) | "f16w" (the same with the f16 first-round wire)
    pub io: String,
    pub lookahead: usize,
    pub blocks: u32,
    pub proxy_core: i32,
    pub proxy_core2: i32,
    pub main_core: i32,
}

/// Reduce sites per 2,048-row prefill chunk: 96 (48 layers x 2; PLAN/TP2_DESIGN_EXL3_2026-09-28.md section on prefill,
/// PLAN/notes_2026-09-28/TP-B.md, and the world-2 bench's own '96-reduce chunk' line). Used only to print an
/// order-of-magnitude extrapolation next to the per-reduce time; it is not a measured chunk time.
pub const W4_CHUNK_REDUCES: usize = 96;

/// The row width the payload sweep is rounded to (the Flash-Next hidden size, one fp32 row = 10,240 B).
pub const W4_ROW_FLOATS: usize = 2560;

/// fp32 floats of a `mb`-MB (decimal) payload, rounded DOWN to whole rows: 21 MB -> 2,050 rows, which is
/// the 2,048-row prefill chunk's 20 MiB (20.97 MB) within a row pair.
pub fn w4_sweep_floats(mb: u64) -> anyhow::Result<usize> {
    anyhow::ensure!((1..=512).contains(&mb), "sweep payload {mb} MB is outside 1..=512");
    let rows = (mb as usize * 1_000_000) / (W4_ROW_FLOATS * 4);
    anyhow::ensure!(rows >= 1, "sweep payload {mb} MB is under one row");
    Ok(rows * W4_ROW_FLOATS)
}

/// `--sizes-mb 2,8,21,42` -> [2, 8, 21, 42]; an empty or malformed list is an error, never a silent default.
pub fn w4_parse_sizes(s: &str) -> anyhow::Result<Vec<u64>> {
    let v: Vec<u64> = s.split(',').map(|t| t.trim()).filter(|t| !t.is_empty())
        .map(|t| t.parse::<u64>().map_err(|e| anyhow::anyhow!("--sizes-mb entry '{t}': {e}"))).collect::<Result<_, _>>()?;
    anyhow::ensure!(!v.is_empty(), "--sizes-mb needs at least one payload size (MB)");
    for &m in &v { w4_sweep_floats(m)?; }
    Ok(v)
}

/// Bytes ONE rank sends per reduce on the wire (equal to what it receives): a recursive-doubling reduce runs
/// `rounds` exchanges of the whole payload; only the FIRST may be f16 (half the bytes), every later exchange
/// carries the fp32 running sum (`tp_xport::Io` docs). World 2 is the one-round case.
pub fn w4_wire_bytes(n_floats: usize, rounds: usize, wire_f16: bool) -> u64 {
    let f32b = n_floats as u64 * 4;
    let first = if wire_f16 { f32b / 2 } else { f32b };
    first + f32b * (rounds.saturating_sub(1)) as u64
}

/// The combinations the bench accepts. Transport 0 is the in-place fp32 serial pair (the serving path
/// converts around it), so any f16 IO there would be a different experiment, refused rather than silently
/// upgraded to the pipelined transport.
pub fn w4_check_combo(xport: u8, io: &str) -> anyhow::Result<usize> {
    anyhow::ensure!(xport <= 2, "--xport {xport}: 0 (serial) | 1 (pipelined, one rail) | 2 (pipelined, both rails)");
    anyhow::ensure!(matches!(io, "f32" | "f32to16" | "f16" | "f16w"), "--io '{io}': f32 | f32to16 | f16 | f16w");
    anyhow::ensure!(xport > 0 || io == "f32",
        "--xport 0 (serial K1/K2 pairs) carries only --io f32 in place; '{io}' needs the pipelined transport (--xport 1 or 2)");
    Ok(if xport == 2 { 2 } else { 1 })
}

/// The recursive-doubling sums of one element across `world` ranks, every rank's final value. Exchange k pairs
/// ranks across bit `(k + shift) % rounds`; `shift` is the R10 phase: the engine aligns both rails to an even
/// device epoch, so the first exchange of a chunk runs round 1 (partner ^ 2) and the world-4 result is
/// (p0+p2)+(p1+p3), i.e. `shift` 1. fp32 addition is commutative, so the two operands' order is immaterial.
fn w4_tree_all(vals: &[f32; 8], world: usize, shift: usize) -> [f32; 8] {
    let rounds = world.trailing_zeros() as usize;
    let mut v = *vals;
    for k in 0..rounds {
        let bit = 1usize << ((k + shift) % rounds);
        let mut nv = v;
        for r in 0..world { nv[r] = v[r & !bit] + v[r | bit]; }
        v = nv;
    }
    v
}

/// The W=4 bench fill: `fill_val` everywhere except every 16th element, which is a CANCELLATION quad that makes the
/// two round-phase associations differ in EVERY io mode: rank 0 = +2048, rank 2 = -2048 and ranks 1 / 3 = tiny f16-exact
/// values (2^-14 (1 + k/1024)), far below half an fp32 ulp of 2048. At the aligned phase ((p0+p2)+(p1+p3)) the big pair
/// cancels exactly and the result is t1+t3 (about 2^-13); at the other phase ((p0+p1)+(p2+p3)) both tiny values are
/// absorbed and the result is 0. Without it, f16-exact partials of similar magnitude sum exactly in fp32 in either order
/// and the f16 modes could not detect a phase shift at all (found by `checker_detects_a_flipped_bit_and_a_phase_shift`).
/// Only valid for world 4 (ranks 0..3); other ranks fall back to `fill_val`.
pub fn w4_fill(r: i32, i: usize, it: usize) -> f32 {
    if i % 16 != 0 || !(0..4).contains(&r) { return fill_val(r, i, it); }
    let k = (((i / 16) + it) * 7 % 1024) as f32;
    let k3 = ((((i / 16) + it) * 7 + 512) % 1024) as f32;
    let tiny = |k: f32| (1.0 + k / 1024.0) * 2f32.powi(-14);
    match r { 0 => 2048.0, 2 => -2048.0, 1 => tiny(k), _ => tiny(k3) }
}

/// The expected output bits of element `i` of reduce `it` for `io`: fp32 bits for "f32", f16 bits for the
/// three f16-out IOs. "f16" / "f16w" start from f16-exact partials (the rank's fill rounded to f16 first).
pub fn w4_expected_bits(io: &str, world: usize, i: usize, it: usize, shift: usize) -> u32 {
    let mut v = [0f32; 8];
    for r in 0..world {
        let x = w4_fill(r as i32, i, it);
        v[r] = if matches!(io, "f16" | "f16w") { half::f16::from_f32(x).to_f32() } else { x };
    }
    let s = w4_tree_all(&v, world, shift)[0];
    if io == "f32" { s.to_bits() } else { half::f16::from_f32(s).to_bits() as u32 }
}

/// Count the elements of `got` (output bits, see `w4_expected_bits`) that differ from the host reference,
/// and return the first mismatch (index, got, want). Multithreaded: a 42 MB payload is 10.5 M elements x
/// `world` hashes.
pub fn w4_count_bad(io: &str, world: usize, it: usize, shift: usize, n: usize,
                    got: &(dyn Fn(usize) -> u32 + Sync)) -> (usize, Option<(usize, u32, u32)>) {
    let threads = 8usize.min(n.max(1));
    let per = n.div_ceil(threads);
    let parts: Vec<(usize, Option<(usize, u32, u32)>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads).map(|t| sc.spawn(move || {
            let (lo, hi) = (t * per, ((t + 1) * per).min(n));
            let (mut bad, mut first) = (0usize, None);
            for i in lo..hi {
                let (g, w) = (got(i), w4_expected_bits(io, world, i, it, shift));
                if g != w { bad += 1; if first.is_none() { first = Some((i, g, w)); } }
            }
            (bad, first)
        })).collect();
        hs.into_iter().map(|h| h.join().expect("w4 checker thread")).collect()
    });
    let bad = parts.iter().map(|p| p.0).sum();
    (bad, parts.iter().filter_map(|p| p.1).next())
}

pub fn run_reduce_w4(a: ReduceW4Args) -> anyhow::Result<()> {
    use crate::exl3_forward::xtp;
    let nr = w4_check_combo(a.xport, &a.io)?;
    anyhow::ensure!(a.world > 2 && (a.world as u32).is_power_of_two(), "--world {} (this harness is world > 2, a power of two; world 2 is the TP-F bench)", a.world);
    anyhow::ensure!(a.rank >= 0 && a.rank < a.world, "--rank {} outside 0..{}", a.rank, a.world);
    anyhow::ensure!(a.peer_ips.len() >= a.world as usize, "--peer-ips needs {} rank-indexed addresses (got {})", a.world, a.peer_ips.len());
    anyhow::ensure!(a.reduces >= 1, "--reduces must be >= 1");
    let rounds = crate::tp_xport::rounds_of(a.world as usize)?;
    let shift = 1 % rounds;   // R10: an even aligned start runs round (0 + 1) % rounds first
    let ips: Vec<std::net::IpAddr> = a.peer_ips[..a.world as usize].iter().map(|s| s.parse())
        .collect::<Result<_, _>>().map_err(|e| anyhow::anyhow!("--peer-ips: {e}"))?;
    let dev2 = if a.dev2.is_empty() { "roceP2p1s0f1".to_string() } else { a.dev2.clone() };
    println!("=== --tp-reduce-bench --world {} rank {} xport {} ({} rail(s)) io {} sizes {:?} MB reduces {} (+{} warmup) lookahead {} blocks {} ===",
             a.world, a.rank, a.xport, nr, a.io, a.sizes_mb, a.reduces, a.warmup, a.lookahead, a.blocks);
    let dev = CudaDevice::new(0)?;
    let kern = crate::tp_xport::Kernels::load(&dev, "tprb")?;
    if a.main_core >= 0 && !net::pin_thread(a.main_core) {
        anyhow::bail!("could not pin the bench thread to core {}", a.main_core);
    }
    // rail 1: the N-way link, exactly as tp::bring_up builds it; the proxy owns it from here
    let mut link = TpLink::connect_nway(a.rank, a.world, &ips, a.port, &a.dev, a.gid, crate::tp::TP_SLOT_BYTES)?;
    link.set_payload(W4_ROW_FLOATS * 4, true)?;
    let ctx = link.ctx_device_ptr();
    let link = std::mem::ManuallyDrop::new(link);
    net::spawn_proxy(link.ctx_addr(), a.proxy_core);
    let mut rails = vec![ctx];
    let mut link2: Option<std::mem::ManuallyDrop<TpLink>> = None;
    if nr == 2 {
        // a box whose second device is unusable must fail by name here: its three peers are about to block in accept()
        xtp::rail_local_preflight(&dev2, a.gid)?;
        let base2 = xtp::rail2_nway_base_for(a.world, a.port)?;
        let mut l2 = TpLink::connect_nway(a.rank, a.world, &ips, base2, &dev2, a.gid, crate::tp::TP_SLOT_BYTES)?;
        l2.set_payload(W4_ROW_FLOATS * 4, true)?;
        rails.push(l2.ctx_device_ptr());
        let l2 = std::mem::ManuallyDrop::new(l2);
        net::spawn_proxy_aux(l2.ctx_addr(), a.proxy_core2);
        println!("[w4] rank {} rail 2 UP: {dev2} (N-way, TCP base {base2}), proxy core {}", a.rank, a.proxy_core2);
        link2 = Some(l2);
    }
    std::thread::sleep(std::time::Duration::from_millis(300));
    // a fresh process: both rails start at epoch 0. A nonzero, non-multiple start would change the association
    // the reference assumes, so it is refused instead of "adjusted" (the engine pads; a bench has no reason to).
    let e1 = net::ctx_device_epoch(link.ctx_addr());
    let e2 = link2.as_ref().map(|l| net::ctx_device_epoch(l.ctx_addr()));
    anyhow::ensure!(e1 % rounds as u64 == 0 && e2.map_or(true, |e| e % rounds as u64 == 0),
                    "rail epochs {e1}/{e2:?} are not multiples of {rounds} on a fresh link");
    let pipe = if a.xport > 0 { Some(crate::tp_xport::Pipe::new(&dev, rails, a.world as usize, a.lookahead, a.blocks)?) } else { None };
    let stream = crate::gpu::fork_blocking_stream(&dev);   // AGENTS §2.1: blocking compute stream
    let mut sync_buf = dev.alloc_zeros::<f32>(W4_ROW_FLOATS)?;
    dev.memset_zeros(&mut sync_buf)?;   // AGENTS §2.2: alloc_zeros does not zero
    dev.synchronize()?;
    let sync_p = *sync_buf.device_ptr() as u64;
    let wire_f16 = a.io == "f16w";
    let (mut total_bad, mut total_reduces) = (0usize, 0usize);
    let mut rows_out: Vec<String> = Vec::new();
    for &mb in &a.sizes_mb {
        let n = w4_sweep_floats(mb)?;
        anyhow::ensure!(crate::tp_xport::pipe_ok(n), "payload {n} floats is not 16 B-aligned for the pipelined transport");
        let mut buf32 = dev.alloc_zeros::<f32>(if a.io == "f16" || a.io == "f16w" { 4 } else { n })?;
        let mut out16 = dev.alloc_zeros::<u16>(if a.io == "f32" { 4 } else { n })?;
        dev.memset_zeros(&mut out16)?;   // AGENTS §2.2: alloc_zeros does not zero
        dev.synchronize()?;
        let mid32 = dev.alloc_zeros::<f32>(if a.io == "f16" || a.io == "f16w" { n } else { 4 })?;
        let mut host32 = vec![0f32; if a.io == "f32" || a.io == "f32to16" { n } else { 0 }];
        let mut host16 = vec![0u16; if a.io == "f16" || a.io == "f16w" { n } else { 0 }];
        let (mut times, mut bad_here) = (Vec::with_capacity(a.reduces), 0usize);
        for it in 0..a.warmup + a.reduces {
            // fill this rank's partial (uploaded and synced BEFORE the timed region)
            match a.io.as_str() {
                "f16" | "f16w" => {
                    for (i, v) in host16.iter_mut().enumerate() { *v = half::f16::from_f32(w4_fill(a.rank, i, it)).to_bits(); }
                    dev.htod_sync_copy_into(&host16, &mut out16)?;
                }
                _ => {
                    for (i, v) in host32.iter_mut().enumerate() { *v = w4_fill(a.rank, i, it); }
                    dev.htod_sync_copy_into(&host32, &mut buf32)?;
                }
            }
            dev.synchronize()?;
            // re-align the ranks: a small serial all-reduce on rail 1 (rounds epochs, phase preserved)
            crate::tp_xport::reduce_serial(&kern, &stream, ctx, sync_p, W4_ROW_FLOATS, rounds)?;
            dev.synchronize()?;
            let (p32, p16, pm) = (*buf32.device_ptr() as u64, *out16.device_ptr() as u64, *mid32.device_ptr() as u64);
            let t0 = std::time::Instant::now();
            match (&pipe, a.io.as_str()) {
                (None, _) => crate::tp_xport::reduce_serial(&kern, &stream, ctx, p32, n, rounds)?,
                (Some(x), "f32") => crate::tp_xport::reduce_pipe(&kern, &stream, x, crate::tp_xport::Io::f32_inplace(p32), n)?,
                (Some(x), "f32to16") => crate::tp_xport::reduce_pipe(&kern, &stream, x,
                    crate::tp_xport::Io { src: p32, src_f16: false, out: p16, out_f16: true, wire_f16: false, mid: p32 }, n)?,
                (Some(x), _) => crate::tp_xport::reduce_pipe(&kern, &stream, x,
                    crate::tp_xport::Io { src: p16, src_f16: true, out: p16, out_f16: true, wire_f16, mid: pm }, n)?,
            }
            dev.synchronize()?;
            let dt = t0.elapsed().as_nanos() as u64;
            if it >= a.warmup { times.push(dt); }
            // validate every reduce, warmup included
            let (bad, first, alt) = if a.io == "f32" {
                let got = dev.dtoh_sync_copy(&buf32)?;
                let g = |i: usize| got[i].to_bits();
                let (b, f) = w4_count_bad(&a.io, a.world as usize, it, shift, n, &g);
                let alt = if b > 0 { w4_count_bad(&a.io, a.world as usize, it, (shift + 1) % rounds, n, &g).0 } else { usize::MAX };
                (b, f, alt)
            } else {
                let got = dev.dtoh_sync_copy(&out16)?;
                let g = |i: usize| got[i] as u32;
                let (b, f) = w4_count_bad(&a.io, a.world as usize, it, shift, n, &g);
                let alt = if b > 0 { w4_count_bad(&a.io, a.world as usize, it, (shift + 1) % rounds, n, &g).0 } else { usize::MAX };
                (b, f, alt)
            };
            if bad > 0 {
                if let Some((i, g, w)) = first { println!("  MISMATCH {mb} MB reduce {it} elem {i}: got {g:#010x} want {w:#010x} ({bad} of {n} differ)"); }
                if alt == 0 { println!("  NOTE: the output equals the OTHER round-phase association exactly (an R10 phase shift, not corruption)"); }
            }
            bad_here += bad;
            total_reduces += 1;
            if link.abort_status() != 0 { anyhow::bail!("rail-1 transport ABORT status {} at {mb} MB reduce {it}", link.abort_status()); }
            if let Some(l2) = &link2 { if l2.abort_status() != 0 { anyhow::bail!("rail-2 ABORT status {} at {mb} MB reduce {it}", l2.abort_status()); } }
        }
        total_bad += bad_here;
        let mut s = times.clone();
        s.sort_unstable();
        let p50_us = pct(&s, 0.5) as f64 / 1e3;
        let wire = w4_wire_bytes(n, rounds, wire_f16);
        let algo_gbs = n as f64 * 4.0 / (p50_us * 1e3);
        let wire_gbs = wire as f64 / (p50_us * 1e3);
        println!("  {mb:>3} MB ({:.2} MiB, {} rows) xport {} io {}: p10 {:.0} p50 {:.0} p90 {:.0} max {:.0} us | payload {:.2} GB/s | wire {:.2} GB/s per direction \
                  ({:.2} per rail, {} rail(s), {:.1} MB sent/reduce) | x{} reduce sites = {:.1} ms per chunk (extrapolation) | {}",
                 n as f64 * 4.0 / 1048576.0, n / W4_ROW_FLOATS, a.xport, a.io, pct(&s, 0.1) as f64 / 1e3, p50_us, pct(&s, 0.9) as f64 / 1e3,
                 pct(&s, 1.0) as f64 / 1e3, algo_gbs, wire_gbs, wire_gbs / nr as f64, nr, wire as f64 / 1e6, W4_CHUNK_REDUCES, p50_us * W4_CHUNK_REDUCES as f64 / 1e3,
                 if bad_here == 0 { "BITWISE_OK" } else { "FAIL" });
        rows_out.push(format!("W4SWEEP rank {} world {} xport {} io {} mb {} p50_us {:.0} payload_gbs {:.2} wire_gbs_per_dir {:.2} ms_x{} {:.1} bitwise {}",
                              a.rank, a.world, a.xport, a.io, mb, p50_us, algo_gbs, wire_gbs, W4_CHUNK_REDUCES, p50_us * W4_CHUNK_REDUCES as f64 / 1e3,
                              if bad_here == 0 { "OK" } else { "FAIL" }));
    }
    let (f1, g1) = (net::ctx_device_epoch(link.ctx_addr()), link.gpu_ready());
    println!("  rail 1: tail fires {}  gate binds {}  abort {}  device epoch {f1} gpu_ready {g1}", link.tail_fires(), link.gate_waits(), link.abort_status());
    let mut phase_ok = f1 % rounds as u64 == 0;
    if let Some(l2) = &link2 {
        let f2 = net::ctx_device_epoch(l2.ctx_addr());
        println!("  rail 2: tail fires {}  abort {}  device epoch {f2} gpu_ready {}", l2.tail_fires(), l2.abort_status(), l2.gpu_ready());
        phase_ok &= f2 % rounds as u64 == 0;
    }
    println!("  round phase after the run: {} (every rail's device epoch must stay a multiple of {rounds})", if phase_ok { "INTACT" } else { "DRIFTED" });
    for r in &rows_out { println!("{r}"); }
    println!("  VALIDATION: {total_bad} mismatching elements over {total_reduces} reduces => {}",
             if total_bad == 0 && phase_ok { "BITWISE_OK" } else { "FAIL" });
    if total_bad != 0 || !phase_ok { anyhow::bail!("world-{} reduce bench validation FAILED", a.world); }
    Ok(())
}

#[cfg(test)]
mod w4_tests {
    use super::*;

    #[test]
    fn sweep_payloads_are_whole_rows() {
        let want_rows = [(2u64, 195usize), (8, 781), (21, 2050), (42, 4101)];
        for (mb, rows) in want_rows {
            let n = w4_sweep_floats(mb).unwrap();
            assert_eq!(n, rows * W4_ROW_FLOATS, "{mb} MB");
            assert!(crate::tp_xport::pipe_ok(n));
            assert!(n * 4 <= mb as usize * 1_000_000 && n * 4 + W4_ROW_FLOATS * 4 > mb as usize * 1_000_000, "{mb} MB rounds DOWN by under one row");
        }
        assert!(w4_sweep_floats(0).is_err() && w4_sweep_floats(513).is_err());
        assert_eq!(w4_parse_sizes("2, 8,21,42").unwrap(), vec![2, 8, 21, 42]);
        for bad in ["", " , ", "2,x", "0", "600"] { assert!(w4_parse_sizes(bad).is_err(), "{bad:?}"); }
    }

    #[test]
    fn wire_bytes_count_only_the_first_round_as_f16() {
        let n = 1_000_000usize;
        assert_eq!(w4_wire_bytes(n, 2, false), 8_000_000, "world 4 fp32: two full exchanges");
        assert_eq!(w4_wire_bytes(n, 2, true), 6_000_000, "world 4 f16 first round: 0.5 + 1.0 payloads");
        assert_eq!(w4_wire_bytes(n, 1, false), 4_000_000, "world 2 fp32");
        assert_eq!(w4_wire_bytes(n, 1, true), 2_000_000, "world 2 f16 wire: the TP-F dual-rail f16 payload");
        assert_eq!(w4_wire_bytes(n, 3, true), 10_000_000, "world 8");
    }

    #[test]
    fn combos_refuse_what_the_serial_pair_cannot_carry() {
        for io in ["f32", "f32to16", "f16", "f16w"] {
            assert_eq!(w4_check_combo(1, io).unwrap(), 1, "{io}");
            assert_eq!(w4_check_combo(2, io).unwrap(), 2, "{io}");
        }
        assert_eq!(w4_check_combo(0, "f32").unwrap(), 1);
        for io in ["f32to16", "f16", "f16w"] {
            let e = w4_check_combo(0, io).unwrap_err().to_string();
            assert!(e.contains("--xport 0") && e.contains("pipelined"), "{io}: {e}");
        }
        assert!(w4_check_combo(3, "f32").is_err());
        assert!(w4_check_combo(1, "bf16").is_err());
    }

    /// The reference is not vacuous: it is rank-identical, equals the closed forms at both phases, the data
    /// tells the two world-4 associations apart, and the f16 IOs differ from each other on real data.
    #[test]
    fn reference_tree_is_rank_identical_and_phase_sensitive() {
        let n = 4000;
        let mut differ = 0;
        for i in 0..n {
            let mut v = [0f32; 8];
            for r in 0..4 { v[r] = w4_fill(r as i32, i, 3); }
            let all = w4_tree_all(&v, 4, 1);
            assert!((1..4).all(|r| all[r].to_bits() == all[0].to_bits()), "elem {i}: ranks disagree");
            let a02 = (v[0] + v[2]) + (v[1] + v[3]);
            let a01 = (v[0] + v[1]) + (v[2] + v[3]);
            assert_eq!(all[0].to_bits(), a02.to_bits(), "shift 1 is (p0+p2)+(p1+p3)");
            assert_eq!(w4_tree_all(&v, 4, 0)[0].to_bits(), a01.to_bits(), "shift 0 is (p0+p1)+(p2+p3)");
            if a02.to_bits() != a01.to_bits() { differ += 1; }
            // world 2 has one round: the phase shift cannot matter
            assert_eq!(w4_tree_all(&v, 2, 0)[0].to_bits(), w4_tree_all(&v, 2, 1)[0].to_bits());
            assert_eq!(w4_tree_all(&v, 2, 0)[0].to_bits(), (v[0] + v[1]).to_bits());
        }
        assert!(differ > 50, "the fill data must distinguish the two associations (got {differ})");
        // world 8: three rounds, every rank agrees
        let v = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let all = w4_tree_all(&v, 8, 1);
        assert!((1..8).all(|r| all[r].to_bits() == all[0].to_bits()));
    }

    #[test]
    fn expected_bits_follow_the_io_rounding_points() {
        let (mut f32to16_vs_f16, mut f32_vs_f16) = (0, 0);
        for i in 0..4000 {
            let mut f = [0f32; 8];
            let mut h = [0f32; 8];
            for r in 0..4 { f[r] = w4_fill(r as i32, i, 1); h[r] = half::f16::from_f32(f[r]).to_f32(); }
            let s32 = w4_tree_all(&f, 4, 1)[0];
            let s16 = w4_tree_all(&h, 4, 1)[0];
            assert_eq!(w4_expected_bits("f32", 4, i, 1, 1), s32.to_bits());
            assert_eq!(w4_expected_bits("f32to16", 4, i, 1, 1), half::f16::from_f32(s32).to_bits() as u32);
            assert_eq!(w4_expected_bits("f16", 4, i, 1, 1), half::f16::from_f32(s16).to_bits() as u32);
            assert_eq!(w4_expected_bits("f16w", 4, i, 1, 1), w4_expected_bits("f16", 4, i, 1, 1), "the f16 wire is bitwise the same sum");
            if w4_expected_bits("f32to16", 4, i, 1, 1) != w4_expected_bits("f16", 4, i, 1, 1) { f32to16_vs_f16 += 1; }
            if half::f16::from_f32(s32).to_f32().to_bits() != s32.to_bits() { f32_vs_f16 += 1; }
        }
        assert!(f32to16_vs_f16 > 0, "f16-rounded inputs and fp32 inputs must be distinguishable by the reference");
        assert!(f32_vs_f16 > 2000, "the f16 output rounding must be visible in the reference");
    }

    /// The checker detects one flipped bit, reports it, and names the other association exactly.
    #[test]
    fn checker_detects_a_flipped_bit_and_a_phase_shift() {
        let (world, it, n) = (4usize, 2usize, 3000usize);
        for io in ["f32", "f16w"] {
            let exact: Vec<u32> = (0..n).map(|i| w4_expected_bits(io, world, i, it, 1)).collect();
            let (bad, first) = w4_count_bad(io, world, it, 1, n, &|i| exact[i]);
            assert_eq!((bad, first), (0, None), "{io}: exact input must pass");
            let mut flipped = exact.clone();
            flipped[1234] ^= 1;
            let (bad, first) = w4_count_bad(io, world, it, 1, n, &|i| flipped[i]);
            assert_eq!(bad, 1, "{io}");
            assert_eq!(first, Some((1234, flipped[1234], exact[1234])));
            let other: Vec<u32> = (0..n).map(|i| w4_expected_bits(io, world, i, it, 0)).collect();
            let (bad_vs_mine, _) = w4_count_bad(io, world, it, 1, n, &|i| other[i]);
            let (bad_vs_other, _) = w4_count_bad(io, world, it, 0, n, &|i| other[i]);
            assert!(bad_vs_mine >= n / 16, "{io}: the shifted association must fail the aligned reference on every cancellation-quad element (got {bad_vs_mine}; else the check is vacuous)");
            assert_eq!(bad_vs_other, 0, "{io}: and pass its own");
        }
    }
}
