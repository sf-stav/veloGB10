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
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

pub struct BenchArgs {
    pub rank: i32,
    pub peer: String,
    pub port: u16,
    pub dev: String,
    pub gid: i32,
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

/// Dump the per-barrier histograms collected during a MODEL run (GB10_TP_TRACE=1), in the same shape
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

pub fn run(a: BenchArgs) -> anyhow::Result<()> {
    println!("=== --tp-barrier-bench  rank {} ===", a.rank);
    println!("    barriers {}  payload {} B  window {}", a.barriers, a.payload_bytes, a.window);
    println!("    spacing {} µs  inject-delay-max {} µs  poison {}  stall every {} for {} µs",
             a.spacing_us, a.inject_delay_us_max, a.poison, a.stall_every, a.stall_us);

    let dev = CudaDevice::new(0)?;
    let ptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_batch.ptx")?);
    let names = ["tp_gate_copy_signal", "tp_wait_add", "tp_bench_fill", "tp_bench_validate",
                 "tp_bench_stall", "tp_bench_now", "kernel_build_id"];
    dev.load_ptx(ptx, "tpb", &names)?;
    let f = |n: &str| dev.get_func("tpb", n).ok_or_else(|| anyhow::anyhow!("missing kernel {n}"));
    let (k1, k2) = (f("tp_gate_copy_signal")?, f("tp_wait_add")?);
    let (kfill, kval, kstall, know) = (f("tp_bench_fill")?, f("tp_bench_validate")?,
                                       f("tp_bench_stall")?, f("tp_bench_now")?);

    // Pin the launch thread to a big X925 core BEFORE the hot loop. A failure to pin invalidates the
    // measurement (jitter reads exactly like a protocol stall), so it is fatal, not a warning.
    if a.main_core >= 0 && !net::pin_thread(a.main_core) {
        anyhow::bail!("could not pin the bench thread to core {} — refusing to report numbers", a.main_core);
    }

    let mut link = TpLink::connect(a.rank, &a.peer, a.port, &a.dev, a.gid, crate::tp::TP_SLOT_BYTES)?;
    link.set_payload(a.payload_bytes, false)?;
    link.bench_config(a.inject_delay_us_max, true);
    if a.cq_hold > 0 { link.bench_cq_hold(a.cq_hold, a.cq_hold_us)?; }
    println!("[bench] link up; proxy → core {}", a.proxy_core);

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
