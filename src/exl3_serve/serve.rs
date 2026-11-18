//! Entry glue of the EXL3 server: `run`, pack detection (`is_exl3_pack`, single-file index), `turn_resolve`, vision boot, the worker mask,
//! `serve_http` (router, listener, scheduler thread) and the `FatalGuard` drop. `build_serve` (the boot sequence) is in `builder.rs`.

use super::*;

impl Drop for FatalGuard {
    fn drop(&mut self) {
        if std::thread::panicking() {
            crate::server::set_engine_dead("scheduler thread panicked");
            if self.exit_on_fatal {
                eprintln!("[exl3-serve] FATAL: scheduler thread panicked — exit 70 (--exit-on-fatal, default)");
                std::thread::sleep(std::time::Duration::from_millis(300));
                crate::logq::flush_and_exit(std::time::Duration::from_millis(300), 70); // H7
            }
        }
    }
}

/// The served `--prefill-chunk` default (owner 2026-10-02, v0.7.1: steady 4,095-row chunks, TP-4X1).
pub const DEFAULT_PREFILL_CHUNK: usize = 4095;

/// EXL27 (owner scope add 2026-10-16): single-file EXL3 packs (one model.safetensors, no
/// model.safetensors.index.json) are SUPPORTED, not refused (§1b-bis 8). We synthesize the
/// one-shard index from the safetensors header — exactly what HF emits for a 1-shard model —
/// so the standard indexed-pack path serves them unchanged. Returns true if an index was written.
pub fn ensure_singlefile_index(dir: &str) -> std::io::Result<bool> {
    use std::io::Read;
    let p = std::path::Path::new(dir.trim_end_matches('/'));
    let idx = p.join("model.safetensors.index.json");
    let st = p.join("model.safetensors");
    if idx.exists() || !st.exists() {
        return Ok(false);
    }
    let mut f = std::fs::File::open(&st)?;
    let mut hlen = [0u8; 8];
    f.read_exact(&mut hlen)?;
    let n = u64::from_le_bytes(hlen) as usize;
    if n == 0 || n > 128_000_000 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad safetensors header length"));
    }
    let mut hdr = vec![0u8; n];
    f.read_exact(&mut hdr)?;
    let v: serde_json::Value = serde_json::from_slice(&hdr)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("header JSON: {e}")))?;
    let mut wm = serde_json::Map::new();
    for k in v.as_object().map(|o| o.keys()).into_iter().flatten() {
        if k == "__metadata__" {
            continue;
        }
        wm.insert(k.clone(), serde_json::Value::String("model.safetensors".to_string()));
    }
    if wm.is_empty() {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "no tensors in header"));
    }
    let total = std::fs::metadata(&st)?.len();
    let out = serde_json::json!({ "metadata": { "total_size": total }, "weight_map": wm });
    let tmp = p.join(".model.safetensors.index.json.tmp-exl27");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&out)?)?;
    std::fs::rename(&tmp, &idx)?; // atomic: no partial index is ever observed
    Ok(true)
}

/// Detection: a dir is an EXL3 pack iff quantization_config.json exists and the safetensors
/// index carries `.trellis` entries (the A3 spec's quadruple manifest).
pub fn is_exl3_pack(dir: &str) -> bool {
    let dir = dir.trim_end_matches('/');
    // FP4L: NVIDIA's ModelOpt mixed-precision Qwen3.8-Flash-Next is served by this trunk too
    // (exl3_forward/fp4l_fwd.rs); every EXL3 entry point (serve, TP head/node, bench, ppl) accepts it.
    if crate::fp4l::pack::is_fp4l_dir(std::path::Path::new(dir)) { return true; }
    let qc = std::path::Path::new(dir).join("quantization_config.json");
    if !qc.exists() {
        return false;
    }
    let idx = std::path::Path::new(dir).join("model.safetensors.index.json");
    if !idx.exists() {
        return false;
    }
    std::fs::read_to_string(&idx).map(|s| s.contains(".trellis")).unwrap_or(false)
}

/// Boot the EXL3 HTTP server (TP=1). Mirrors run_server's serving surface.
/// TURNRESUME: resolve --turn-resume / --next-turn-prefill at boot. One line either way when asked for; the kept states
/// (one recurrent state per lane) are allocated here, not at first use; no room, TP, no prefix cache or the DFlash2
/// ring = off with one line saying why (counted as mode 2 on /metrics), never a refusal.
pub(super) fn turn_resolve(model: &Arc<FwdModel>, width: usize, prefix_on: bool, tp: bool, df2: bool, floor: u64) -> Option<Box<TurnArm>> {
    use crate::turn_resume as tr;
    let nt = tr::NtMode::parse(&crate::opts::choice(crate::opt!("next-turn-prefill")));
    let asked = crate::opts::on(crate::opt!("turn-resume")) || nt != tr::NtMode::Off;
    tr::set_accepting(false);
    if !asked { return None; }
    let fall = |why: &str| -> Option<Box<TurnArm>> {
        tr::MODE_RESUME.store(2, std::sync::atomic::Ordering::Relaxed);
        tr::MODE_NTP.store(if nt != tr::NtMode::Off { 2 } else { 0 }, std::sync::atomic::Ordering::Relaxed);
        crate::rprintln!("[exl3-serve] turn resume: asked for (--turn-resume on / --next-turn-prefill {}), running OFF: {why}", nt.name());
        None
    };
    if !prefix_on || !model.prefix_snapshots() { return fall("it needs --prefix-cache on (the prompt-end snapshot is its pre-turn state)"); }
    if tp { return fall("not driven by head-shipped events under TP in this build (TP=1 only)"); }
    if df2 { return fall("the DFlash2 draft ring is primed by every prefill chunk; background passes would run on it (MTP head packs only)"); }
    let per = model.ckpt_layout().bytes() as u64;
    let total = per * width as u64;
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    if let Some(avail) = crate::memwatch::mem_available_bytes() {
        if avail < total.saturating_add(floor) {
            return fall(&format!("{width} lane state(s) need {:.2} GiB and {:.1} GiB are free (the prefix pool keeps {:.0} GiB free)",
                                 gib(total), gib(avail), gib(floor)));
        }
    }
    let mut bufs = Vec::with_capacity(width);
    for _ in 0..width {
        match model.ckpt_alloc() {
            Ok(b) => bufs.push(b),
            Err(e) => { drop(bufs); return fall(&format!("allocating {width} x {:.0} MB failed ({e:#})", per as f64 / 1e6)); }
        }
    }
    let chunk = crate::opts::ranged(crate::opt!("next-turn-chunk")).map(|v| v as usize).unwrap_or(512).max(1);
    let delay_ms = crate::opts::ranged(crate::opt!("next-turn-delay-ms")).map(|v| v as u64).unwrap_or(150);
    tr::MODE_RESUME.store(1, std::sync::atomic::Ordering::Relaxed);
    tr::MODE_NTP.store(if nt != tr::NtMode::Off { 1 } else { 0 }, std::sync::atomic::Ordering::Relaxed);
    tr::STATE_BYTES.store(total, std::sync::atomic::Ordering::Relaxed);
    tr::STATE_LANES.store(width as u64, std::sync::atomic::Ordering::Relaxed);
    tr::set_accepting(nt != tr::NtMode::Off);
    crate::rprintln!("[exl3-serve] turn resume: ON — one kept next-turn state per lane, {width} x {:.0} MB = {:.2} GiB of device memory; \
                      next-turn prefill {}{} (OUTPUT-CHANGING, like every prefix-cache resume: a different chunk grid)",
                     per as f64 / 1e6, gib(total), nt.name(),
                     match nt {
                         tr::NtMode::On => format!(": idle-time passes of {chunk}-row chunks, started {delay_ms} ms after a reply; a request waits for at most the chunk in flight"),
                         tr::NtMode::Inline => format!(": DIAGNOSTIC, the pass runs at the next admission ({chunk}-row chunks)"),
                         tr::NtMode::Off => " (nothing fills the kept states: --next-turn-prefill on does)".to_string(),
                     });
    Some(Box::new(TurnArm {
        nt, chunk, bufs, toks: (0..width).map(|_| None).collect(), pending: (0..width).map(|_| None).collect(),
        pending_at: (0..width).map(|_| std::time::Instant::now()).collect(), delay: std::time::Duration::from_millis(delay_ms),
        last_upkeep: std::time::Instant::now(),
    }))
}

pub fn run(args: &[String], model_dir: &str) -> Result<()> {
    crate::fp4l::apply_pack_defaults(model_dir);
    let parts = build_serve(args, model_dir, None)?;
    serve_http(parts, None)
}

/// VIS-2: the text config's interleaved mrope_section (None = the model declares no interleaved mrope).
pub(super) fn mrope_section_of(model_dir: &str) -> Option<[usize; 3]> {
    crate::vision_tower::vision_geometry(model_dir).ok()??;
    let raw = std::fs::read_to_string(std::path::Path::new(model_dir).join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let rp = v.get("text_config").unwrap_or(&v).get("rope_parameters")?;
    if rp.get("mrope_interleaved").and_then(|x| x.as_bool()) != Some(true) { return None; }
    let a = rp.get("mrope_section")?.as_array()?;
    if a.len() != 3 { return None; }
    Some([a[0].as_u64()? as usize, a[1].as_u64()? as usize, a[2].as_u64()? as usize])
}

/// VIS-2: load the vision tower for the EXL3 server (CPU weights + the f32 GPU tower) and the text
/// config's interleaved mrope_section. Any failure = text-only with a printed reason.
#[allow(clippy::type_complexity)]
pub(super) fn vision_boot(model_dir: &str, fail_first_gemm: bool) -> (Option<Arc<crate::vision_tower::VisualTower>>,
                                    Option<Arc<std::sync::Mutex<crate::vision_gpu::GpuVisualTower>>>,
                                    Option<[usize; 3]>) {
    // S-B9-REL-VISIONFIX2 (review M3): `--vision-cpu` is a registry flag, but only the NVFP4 server honours it.
    // This server always builds the GPU tower (the CPU tower runs only as the fallback after a GPU failure):
    // say so instead of ignoring the flag silently.
    if crate::opts::on(crate::opt!("vision-cpu")) {
        crate::rprintln!("[exl3-serve] vision: --vision-cpu is honoured by the NVFP4 server only; this server always builds the GPU tower \
(the CPU tower serves only after a GPU-tower failure)");
    }
    match crate::vision_tower::vision_geometry(model_dir) {
        Ok(Some(_)) => {}
        Ok(None) => return (None, None, None),
        Err(e) => { println!("[exl3-serve] vision: geometry probe failed ({e:#}) — text-only"); return (None, None, None); }
    }
    let Some(sec) = mrope_section_of(model_dir) else {
        crate::dprintln!("[exl3-serve] vision: the text config declares no interleaved mrope_section — text-only");
        return (None, None, None);
    };
    let t0 = std::time::Instant::now();
    let tower = match crate::vision_tower::VisualTower::load(model_dir) {
        Ok(t) => Arc::new(t),
        Err(e) => { println!("[exl3-serve] vision: tower load failed ({e:#}) — text-only"); return (None, None, None); }
    };
    let gpu = match cudarc::driver::CudaDevice::new(0).map_err(anyhow::Error::from)
        .and_then(|d| crate::vision_gpu::GpuVisualTower::new_with_drill(d, &tower, fail_first_gemm)) {
        Ok(g) => g,
        Err(e) => { crate::dprintln!("[exl3-serve] vision: GPU tower unavailable ({e:#}) — text-only"); return (None, None, None); }
    };
    let d = tower.dims;
    crate::dprintln!("[exl3-serve] vision ON: tower from {}: {} blocks x {} wide -> {} ({:.1} s); images resized to a longer side <= {} px \
              (--image-max-edge), mrope section {sec:?}; QSA indexer pooled keys follow the image positions (VIS-3)",
             tower.source, d.depth, d.hidden, d.out_hidden, t0.elapsed().as_secs_f64(),
             if tower.preproc.max_edge == 0 { "unlimited".to_string() } else { tower.preproc.max_edge.to_string() });
    (Some(tower), Some(Arc::new(std::sync::Mutex::new(gpu))), Some(sec))
}

/// TP-D: the host-thread mask at TP — the big cores minus the launch core (9, the pinned scheduler
/// thread) and the RDMA proxy core (19). TP=1: the resolved mask unchanged.
pub(super) fn tp_worker_mask(m: &Option<Vec<usize>>, tp: bool, world: i32) -> Option<Vec<usize>> {
    let m = m.clone()?;
    if !tp { return Some(m); }
    // TP-F: the dual-rail prefill transport pins its second proxy on TP_AUX_PROXY_CORE (18).
    let aux = if crate::exl3_forward::xtp::prefill_xport_mode() == 2 { crate::exl3_forward::xtp::TP_AUX_PROXY_CORE as usize } else { usize::MAX };
    // TP-4F2: `--tp-reduce single` pins a third proxy (the single-stage reduce ctx) on TP_SINGLE_PROXY_CORE (17).
    let single = if crate::exl3_forward::xtp::tp_reduce_single_for(world) { crate::exl3_forward::xtp::TP_SINGLE_PROXY_CORE as usize } else { usize::MAX };
    let f: Vec<usize> = m.into_iter().filter(|&c| c != 9 && c != 19 && c != aux && c != single).collect();
    if f.is_empty() { None } else { Some(f) }
}

/// TP-D: threads the pinned (single-core) TP scheduler thread spawns — the prefill PLE row workers —
/// would inherit its one-core mask and serialize on core 9; they re-pin to this mask instead.
pub(super) fn set_worker_mask(m: Option<Vec<usize>>) {
    if let Some(m) = m {
        crate::dprintln!("[exl3-serve] TP: scheduler pinned to core 9; spawned host workers use cpus {}",
                 crate::cpu_affinity::format_cpu_list(&m));
        crate::cpu_affinity::set_worker_mask(m);
    }
}

/// Spawn the scheduler thread and serve HTTP on this thread (never returns on success).
/// `tp_ctl` Some = the TP=2 head: the scheduler runs `run_tp_head` over the node control streams,
/// on a thread pinned to core 9 (the TP launch-thread rule), and the HTTP/worker threads keep the
/// big-core mask minus the launch (9) and proxy (19) cores.
pub(super) fn serve_http(parts: ServeParts, tp_ctl: Option<Vec<std::net::TcpStream>>) -> Result<()> {
    let ServeParts { sched, state, port, cpu_aff, exit_on_fatal, rx: srx, boot_facts } = parts;
    let model_name = state.model_name.clone();
    // v0.7.3 ("tel"): the ONE sender task rides THIS runtime (serve_http is head/TP=1 only).
    let otel_sink = state.otel.clone();
    // THINKBUDGET1: parse --reasoning-config now, so a value that does not parse is reported at boot
    // (with its fallback) rather than at the first budgeted request.
    let _ = crate::think_budget::reasoning_cfg();
    // HOST / RO-7: pin THIS thread (it runs the HTTP runtime below) to the big cores BEFORE the
    // scheduler thread is spawned — a new thread inherits its creator's mask (Linux), and so do
    // the threads either of them spawns later (prefill PLE workers, tokio's blocking pool).
    crate::dprintln!("[exl3-serve] {}", cpu_aff.1);
    let mask = tp_worker_mask(&cpu_aff.0, tp_ctl.is_some(), tp_ctl.as_ref().map_or(1, |c| c.len() as i32 + 1));
    if let Some(cpus) = mask.as_ref() {
        if let Err(e) = crate::cpu_affinity::pin_current_thread(cpus) {
            println!("[exl3-serve] cpu affinity: pinning FAILED ({e:#}) — threads left to the OS scheduler");
        }
    }
    match tp_ctl {
        None => {
            std::thread::spawn(move || {
                let _guard = FatalGuard { exit_on_fatal };
                sched.run(srx)
            });
        }
        Some(streams) => {
            std::thread::spawn(move || {
                // any exit of the TP scheduler thread ends the process (a TP head cannot serve on
                // without its lockstep): exit 70, the NVFP4 TP head's contract
                let _guard = FatalGuard { exit_on_fatal: true };
                if !crate::net::pin_thread(9) {
                    eprintln!("\n*** FATAL: TP head scheduler failed to pin to core 9 — TP refuses to run unpinned. Exiting. ***\n");
                    crate::logq::flush_and_exit(std::time::Duration::from_millis(300), 70); // H7
                }
                set_worker_mask(mask);
                let r = sched.run_tp_head(srx, streams);
                match r {
                    Ok(()) => eprintln!("[exl3-serve] TP head: request channel closed — session over"),
                    Err(e) => eprintln!("\n*** FATAL: the TP scheduler failed: {e:#}. Exiting (70). ***\n"),
                }
                std::thread::sleep(std::time::Duration::from_millis(300)); // let the error events flush
                crate::logq::flush_and_exit(std::time::Duration::from_millis(300), 70); // H7 + LR-1
            });
        }
    }
    let rt = tokio::runtime::Builder::new_current_thread().enable_all()
        .build().context("tokio runtime")?;
    rt.block_on(async move {
        // v0.7.3 ("tel"): run_sender is fully async (timed connect/write/read) — a slow or
        // absent receiver delays only itself; decode and SSE serving continue (the design
        // contract: ring, drop-on-full, one task off the compute path, zero cost when off).
        if let Some(sink) = otel_sink { tokio::spawn(crate::otel::run_sender(sink)); }
        let app = create_router(state);
        let listener = crate::server::http_listen(port).await;
        let url_l = listener.local_addr().map_or_else(|_| format!("port {port}"), |a| a.to_string());
        if crate::boot_report::debug() {
            println!("[exl3-serve] listening on {url_l} (model id: {model_name})");
        }
        {
            let (tp_w, ctx, batch, kv_name, prefix_on, mtp_head, mtp_off, df2_on) = &boot_facts;
            crate::boot_report::serving(&crate::boot_report::ServingLine {
                family: "exl3", model: &model_name, tp: *tp_w, ctx: *ctx, batch: *batch,
                kv: kv_name, prefix: if *prefix_on { "on" } else { "off" },
                mtp: if !*mtp_head { "none" } else if *mtp_off { "off" } else { "auto" },
                // ISSUES[94]: the DF2 chain is the engaged speculator — the SERVING line says
                // so (dflash2), the pack MTP otherwise.
                spec: if *df2_on { "dflash2" } else if *mtp_head && !*mtp_off { "mtp" } else { "plain" },
                port,
                url: if url_l.starts_with("port ") { format!("http://0.0.0.0:{port}") } else { format!("http://{url_l}") },
            });
        }
        // WP02: TCP_NODELAY — an SSE token event is a few dozen bytes; Nagle held them for the ACK
        axum::serve(listener, app).tcp_nodelay(true).await.unwrap();
    });
    Ok(())
}
