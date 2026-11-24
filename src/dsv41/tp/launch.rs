//! DSV4.1 TP (leg TPC): the dsv41 head / node launch. `--probe-dsv41-tp --model-dir <V4.1 pack> --tp N
//! [--nodes ip:port,..]` on the head (`.11`); every other box runs a plain zero-config `--node --port 29500`.
//!
//! The head (rank 0): the cluster sync ships every node ITS shard dir through the content-addressed blob cache
//! (`shard_plan` -> `ShardFormat::Dsv41` -> leg LS's per-rank plan, Engram heads included: a node that already
//! holds a head by manifest sha256 is sent nothing, a head the head box does not hold is packed from the native
//! checkpoint while it is sent). The build-id handshake of the sync refuses a head and a node from different
//! builds, as for every family. The head's options ride `TpConfig` (the registry snapshot `opts`; every dsv41 TP
//! option is `Scope::Spmd`), and the program the nodes run is named in `TpConfig::exl3_mode` (`dsv41-tp-smoke`).
//! Then: the RDMA link (`TpContext`), the branch agree (`TpBranch::Dsv41`), the comm layer (`TpComm::attach`)
//! and the program, with the per-step agree guard and the transport watchdog (a dead or slow peer aborts the
//! link: every rank stops, nothing partial is used).

use super::comm::TpComm;
use super::frame::Shape;
use super::smoke::{Smoke, SmokeOpts};
use anyhow::{bail, ensure, Context, Result};
use std::path::Path;

/// The program name the nodes dispatch on (`tp_cluster` node arm: `exl3_mode` prefix `dsv41-`).
pub const SMOKE_MODE: &str = "dsv41-tp-smoke";
/// Leg TP: the full-model program (`--dsv41-tp-program model`, `prog.rs`).
pub const MODEL_MODE: &str = "dsv41-tp-model";

fn program_is_model() -> bool { crate::opts::lookup("dsv41-tp-program").and_then(crate::opts::get).as_deref() == Some("model") }

fn world_of(args: &[String]) -> Result<Option<usize>> {
    for (i, a) in args.iter().enumerate() {
        if let Some(v) = a.strip_prefix("--tp=") { return Ok(Some(v.parse().context("--tp")?)); }
        if a == "--tp" {
            return Ok(Some(match args.get(i + 1) { Some(n) if !n.starts_with('-') => n.parse().context("--tp")?, _ => 2 }));
        }
    }
    Ok(None)
}

fn nodes_of(args: &[String]) -> Result<Option<Vec<std::net::SocketAddr>>> {
    let Some(s) = crate::arg_value(args, "--nodes") else { return Ok(None) };
    let mut v = Vec::new();
    for p in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        v.push(if p.contains(':') { p.parse().with_context(|| format!("--nodes {p}"))? }
               else { std::net::SocketAddr::new(p.parse().with_context(|| format!("--nodes {p}"))?, 29500) });
    }
    Ok(Some(v))
}

/// The engagement receipt of the shipped shard dir on a node: its plan, Engram heads and bytes.
fn shard_receipt(dir: &Path) -> Result<String> {
    let m = crate::dsv41::load::ship::read_manifest(dir).context("this rank's shard manifest (dsv41_shard.json)")?;
    let mut bytes = 0u64;
    let mut files = 0usize;
    for d in [dir.to_path_buf(), dir.join("seg")] {
        if let Ok(rd) = std::fs::read_dir(&d) {
            for e in rd.flatten() {
                if let Ok(md) = std::fs::metadata(e.path()) { if md.is_file() { bytes += md.len(); files += 1; } }
            }
        }
    }
    let mut heads = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir.join("engram")) {
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            if n.ends_with(".bin") { heads.push(format!("{n} ({:.1} GB)", std::fs::metadata(e.path()).map(|m| m.len()).unwrap_or(0) as f64 / 1e9)); }
        }
    }
    heads.sort();
    Ok(format!("shard dir {}: {files} files, {:.2} GB (weights; symlinks into the blob cache), manifest {:?}; Engram heads: {}",
               dir.display(), bytes as f64 / 1e9, m, if heads.is_empty() { "none".into() } else { heads.join(", ") }))
}

fn device() -> Result<std::sync::Arc<cudarc::driver::CudaDevice>> {
    cudarc::driver::CudaDevice::new(0).context("CUDA device 0")
}

/// `--probe-dsv41-tp`: with `--tp N` the TP head; without it the world-1 smoke on this box (no link).
pub fn probe_cli(args: &[String]) -> Result<()> {
    let dir = crate::arg_value(args, "--model-dir").map(std::path::PathBuf::from);
    let shape = match &dir { Some(d) => Shape::of(&crate::dsv41::config::Dsv41Config::load(d)?), None => Shape::v41_flash() };
    let model = program_is_model();
    match world_of(args)? {
        Some(w) if w >= 2 => run_head(args, dir.as_deref().context("--tp needs --model-dir <V4.1 pack> (the head ships its shards)")?, w, shape),
        _ if model => {
            // world 1 on this box (a slice pack holds some layers: the gate runs on those)
            let dir = dir.context("--dsv41-tp-program model needs --model-dir")?;
            let dev = device()?;
            let comm = TpComm::solo(&dev, shape)?;
            super::metrics::attached(1);
            let o = super::prog::ModelOpts::from_registry()?;
            let src = super::wload::RankSrc::pack(&dir, 1, 0, &crate::dsv41::load::ship::deal())?;
            let edir = crate::dsv41::load::engram::engram_dir(None);
            let ok = super::prog::run(&dev, &comm, &src, &dir, &edir, &o, &mut |l| println!("{l}"))?;
            println!("{}", if ok { "DSV41_TP_MODEL_OK" } else { "DSV41_TP_MODEL_FAIL" });
            Ok(())
        }
        _ => {
            let dev = device()?;
            let comm = TpComm::solo(&dev, shape)?;
            super::metrics::attached(1);
            let lines = Smoke::new(&dev, &comm, SmokeOpts::from_registry()?).run()?;
            for l in lines { println!("{l}"); }
            Ok(())
        }
    }
}

/// Leg PFX2: resolve `--engram-row-cache-mib` on the head before `TpConfig` snapshots `opts` to the nodes:
/// when neither `--engram-row-cache-mib` nor `--dsv41-engram-ram-gib` was explicitly set, TP2 defaults to
/// 1024 MiB (+1.16 GiB Rank 0 UMA margin) and other worlds keep 2048 MiB.
pub(crate) fn resolve_head_engram_cache(dir: &Path, world: usize) {
    let Some(o) = crate::opts::lookup("engram-row-cache-mib") else { return };
    if crate::opts::get(o).is_some() { return; }
    let num = |n: &str, d: usize| crate::opts::lookup(n).and_then(crate::opts::get).and_then(|v| v.parse().ok()).unwrap_or(d);
    let lanes = num("dsv41-tp-lanes", 4).clamp(1, super::fwd::MAX_LANES);
    let max_seq = num("max-seq-len", 16384);
    let chunk = num("dsv41-tp-chunk", 1024).max(1) + num("dsv41-prefill-tail-merge", 512);
    let ram = crate::opts::lookup("dsv41-engram-ram-gib").and_then(crate::opts::get);
    let mib = crate::dsv41::load::fit::resolve_engram_cache_mib(dir, world, lanes, max_seq, chunk, None, ram.as_deref());
    crate::opts::set(o, mib.to_string());
}

fn run_head(args: &[String], model_dir: &Path, world: usize, shape: Shape) -> Result<()> {
    ensure!(matches!(world, 2 | 3 | 4 | 6 | 8), "dsv41 TP: --tp 2, 3, 4, 6, or 8 (got {world})");
    let dir_abs = std::fs::canonicalize(model_dir).with_context(|| format!("--model-dir {}", model_dir.display()))?;
    let explicit = nodes_of(args)?;
    let wait = std::time::Duration::from_secs(crate::arg_value(args, "--discover-wait").and_then(|s| s.parse().ok()).unwrap_or(3));
    let model = program_is_model();
    let opts = SmokeOpts::from_registry()?;   // fail on a bad option before shipping anything
    if model { resolve_head_engram_cache(&dir_abs, world); }
    let mopts = if model { Some(super::prog::ModelOpts::from_registry_world(world)?) } else { None };
    let mut tpc = crate::tp::TpConfig::from_opts();
    tpc.world = world as u32;
    tpc.mode_serve = false;
    tpc.exl3_mode = if model { MODEL_MODE } else { SMOKE_MODE }.to_string();
    crate::tp::set_tp_config(tpc.clone());
    println!("[head] dsv41 TP={world} smoke: pack {} (shards + Engram heads ship through the blob cache)", dir_abs.display());
    let t0 = std::time::Instant::now();
    let (nodes, streams) = crate::cluster::run_head_retain(&dir_abs, explicit, wait, &tpc)?;
    println!("[head] {} node(s) synced in {:.1}s: {}", nodes.len(), t0.elapsed().as_secs_f64(),
             nodes.iter().map(|n| n.addr.to_string()).collect::<Vec<_>>().join(", "));
    // the model program: rank 0's slices come from the fast-reboot stage (built once under the tp cache, BEFORE the link
    // comes up: a cold stage takes minutes and the nodes wait in their bring-up dial, not in a timed collective)
    let staged = if mopts.is_some() { Some(stage_rank0(&dir_abs, world)?) } else { None };
    let watch = crate::cluster::watch_nodes_during_bring_up(&streams)?;
    let ctx = crate::tp::TpContext::bring_up_head(world as i32)?;
    watch.disarm(&streams);
    let dev = device()?;
    let comm = TpComm::attach(&dev, shape, ctx)?;
    super::metrics::attached(world);
    if let Some(o) = mopts {
        let deal = crate::dsv41::load::ship::deal();
        let src = match staged.flatten() { Some(d) => super::wload::RankSrc::shard_dir(&d)?, None => super::wload::RankSrc::pack(&dir_abs, world, 0, &deal)? };
        let edir = crate::dsv41::load::engram::engram_dir(None);
        let r = super::prog::run(&dev, &comm, &src, &dir_abs, &edir, &o, &mut |l| println!("[head] {l}"));
        drop(streams);
        return match r {
            Ok(ok) => { println!("{}", if ok { "DSV41_TP_MODEL_OK" } else { "DSV41_TP_MODEL_FAIL" }); Ok(()) }
            Err(e) => bail!("{e:#}"),
        };
    }
    let r = Smoke::new(&dev, &comm, opts).run();
    drop(streams);
    match r {
        Ok(lines) => { for l in lines { println!("{l}"); } Ok(()) }
        Err(e) => bail!("{e:#}"),
    }
}

/// Rank 0's shard as the head's fast-reboot stage (`load::stage`): its dir, or None when it was skipped (no room,
/// `--dsv41-load-cache off`): the caller then reads the slices straight out of the pack.
fn stage_rank0(dir_abs: &Path, world: usize) -> Result<Option<std::path::PathBuf>> {
    {
        let deal = crate::dsv41::load::ship::deal();
        let c = crate::dsv41::load::ship::census_of(dir_abs)?;
        let po = crate::dsv41::load::ship::plan_opts(&c, world, &deal)?;
        let sel = crate::dsv41::load::plan::select(&c, &po, 0)?;
        let (files, _) = crate::dsv41::load::ship::rank_files(&c, dir_abs, &po, &sel)?;
        let source = dir_abs.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "pack".into());
        let mode = crate::dsv41::load::stage::Mode::parse(crate::opts::lookup("dsv41-load-cache").and_then(crate::opts::get).as_deref());
        let t = std::time::Instant::now();
        let st = crate::dsv41::load::stage::stage_or_skip(&files, &source, world, 0, mode, (0, world));
        println!("[head] rank 0 stage: {:?} {} ({:.1} GB, {:.0} s)", st.outcome, st.dir.as_ref().map_or("-".into(), |d| d.display().to_string()), st.bytes as f64 / 1e9, t.elapsed().as_secs_f64());
        Ok(st.dir)
    }
}

/// `--server --model-dir <dsv41 pack> --tp N` (leg SE): the head of the serving program. Ships like `run_head`, brings the link
/// up, loads rank 0 (model + DSpark stages + warm-up), then hands the built `TpServe` to `head_main`, which runs the HTTP
/// server and the lane loop on this thread. The nodes run `serve::TpServe::follow` (`run_node`).
pub fn serve_head(args: &[String], model_dir: &Path, world: usize,
                  head_main: &mut dyn FnMut(&mut super::serve::TpServe, &mut dyn FnMut(String)) -> Result<()>) -> Result<()> {
    ensure!(matches!(world, 2 | 3 | 4 | 6 | 8), "dsv41 TP: --tp 2, 3, 4, 6, or 8 (got {world})");
    let dir_abs = std::fs::canonicalize(model_dir).with_context(|| format!("--model-dir {}", model_dir.display()))?;
    let shape = Shape::of(&crate::dsv41::config::Dsv41Config::load(&dir_abs)?);
    let explicit = nodes_of(args)?;
    let wait = std::time::Duration::from_secs(crate::arg_value(args, "--discover-wait").and_then(|s| s.parse().ok()).unwrap_or(3));
    resolve_head_engram_cache(&dir_abs, world);
    let so = super::serve::ServeOpts::from_registry_world(world);
    let mut tpc = crate::tp::TpConfig::from_opts();
    tpc.world = world as u32;
    tpc.mode_serve = false;
    tpc.exl3_mode = super::serve::SERVE_MODE.to_string();
    crate::tp::set_tp_config(tpc.clone());
    println!("[head] dsv41 TP={world} serve: pack {} (shards + Engram heads ship through the blob cache)", dir_abs.display());
    let t0 = std::time::Instant::now();
    let (nodes, streams) = crate::cluster::run_head_retain(&dir_abs, explicit, wait, &tpc)?;
    println!("[head] {} node(s) synced in {:.1}s: {}", nodes.len(), t0.elapsed().as_secs_f64(), nodes.iter().map(|n| n.addr.to_string()).collect::<Vec<_>>().join(", "));
    let staged = stage_rank0(&dir_abs, world)?;
    let watch = crate::cluster::watch_nodes_during_bring_up(&streams)?;
    let dev = device()?;
    let _ = crate::dsv41::gpu::raw_fn("d41x_xc2_stage");
    let ctx = crate::tp::TpContext::bring_up_head(world as i32)?;
    watch.disarm(&streams);
    let comm = TpComm::attach(&dev, shape, ctx)?;
    super::metrics::attached(world);
    let deal = crate::dsv41::load::ship::deal();
    let src = match staged { Some(d) => super::wload::RankSrc::shard_dir(&d)?, None => super::wload::RankSrc::pack(&dir_abs, world, 0, &deal)? };
    let edir = crate::dsv41::load::engram::engram_dir(None);
    // leg HT1: the idle tier wakes parked nodes over the session's control streams (idle during serving)
    // (every node's stream, in order, or no serving: a node the head cannot wake would strand in its park)
    let link = super::idle::Link::Head(streams.iter().map(|s| s.try_clone()).collect::<std::io::Result<Vec<_>>>()
        .context("leg HT1: the idle tier needs a handle on every node's control connection")?);
    let r = super::serve::run(&dev, &comm, &src, &dir_abs, &edir, &so, &mut |l| println!("[head] {l}"), link, Some(head_main));
    drop(streams);
    r
}

/// The node side (called by `tp_cluster`'s node arm for an `exl3_mode` starting `dsv41-`).
pub fn run_node(tpc: &crate::tp::TpConfig, dir: &Path, head_ip: std::net::IpAddr, mut stream: std::net::TcpStream) -> Result<()> {
    crate::net::set_head_liveness(&stream);
    let fail = |stream: &mut std::net::TcpStream, e: anyhow::Error| -> anyhow::Error {
        crate::cluster::node_report_failure(stream, &format!("{e:#}"));
        e
    };
    if tpc.exl3_mode != SMOKE_MODE && tpc.exl3_mode != MODEL_MODE && tpc.exl3_mode != super::serve::SERVE_MODE {
        return Err(fail(&mut stream, anyhow::anyhow!("dsv41 node: unknown program '{}' (this build runs '{SMOKE_MODE}', '{MODEL_MODE}' and '{}')", tpc.exl3_mode, super::serve::SERVE_MODE)));
    }
    let receipt = match shard_receipt(dir) { Ok(r) => r, Err(e) => return Err(fail(&mut stream, e)) };
    println!("[node] dsv41 rank {}/{}: {receipt}", tpc.node_rank, tpc.world);
    let shape = match crate::dsv41::config::Dsv41Config::load(dir) { Ok(c) => Shape::of(&c), Err(e) => return Err(fail(&mut stream, e)) };
    let opts = match SmokeOpts::from_registry() { Ok(o) => o, Err(e) => return Err(fail(&mut stream, e)) };
    let dev = match device() { Ok(d) => d, Err(e) => return Err(fail(&mut stream, e)) };
    let _ = crate::dsv41::gpu::raw_fn("d41x_xc2_stage");
    let ctx = match crate::tp::TpContext::bring_up_node(head_ip, tpc.node_rank, tpc.world as i32) {
        Ok(c) => c,
        Err(e) => return Err(fail(&mut stream, e.context("the node's RDMA data-plane link bring-up failed"))),
    };
    let comm = TpComm::attach(&dev, shape, ctx)?;
    super::metrics::attached(tpc.world as usize);
    if tpc.exl3_mode == super::serve::SERVE_MODE {
        let so = super::serve::ServeOpts::from_registry_world(tpc.world as usize);
        let src = super::wload::RankSrc::shard_dir(dir)?;
        // a node without its control connection could not leave a park: the session refuses to start rather than strand there
        let link = super::idle::Link::Node(stream.try_clone().context("leg HT1: the idle tier needs a handle on the head's control connection")?);
        let r = super::serve::run(&dev, &comm, &src, dir, &dir.join("engram"), &so, &mut |l| println!("[node] {l}"), link, None);
        drop(stream);
        return r;
    }
    if tpc.exl3_mode == MODEL_MODE {
        let o = super::prog::ModelOpts::from_registry_world(tpc.world as usize)?;
        let src = super::wload::RankSrc::shard_dir(dir)?;
        let ok = super::prog::run(&dev, &comm, &src, dir, &dir.join("engram"), &o, &mut |l| println!("[node] {l}"))?;
        println!("[node] {}", if ok { "DSV41_TP_MODEL_OK" } else { "DSV41_TP_MODEL_FAIL" });
        drop(stream);
        return Ok(());
    }
    let lines = Smoke::new(&dev, &comm, opts).run()?;
    for l in lines { println!("[node] {l}"); }
    drop(stream);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv41_pfx2_tp2_head_engram_cache_propagates_1024_and_tp4_keeps_2048() {
        let dir = Path::new("/nonexistent");
        let o_mib = crate::opts::lookup("engram-row-cache-mib").unwrap();
        let o_ram = crate::opts::lookup("dsv41-engram-ram-gib").unwrap();
        let saved_mib = crate::opts::get(o_mib);
        let saved_ram = crate::opts::get(o_ram);

        // Neither flag set: TP2 resolves 1024 MiB into the Spmd registry snapshot; TP4 resolves 2048 MiB
        crate::opts::unset(o_mib);
        crate::opts::unset(o_ram);
        resolve_head_engram_cache(dir, 2);
        assert_eq!(crate::opts::get(o_mib).as_deref(), Some("1024"));
        assert!(crate::opts::snapshot().iter().any(|(k, v)| k == "engram-row-cache-mib" && v == "1024"));
        assert_eq!(super::super::serve::ServeOpts::from_registry_world(2).cache_mib, 1024);

        crate::opts::unset(o_mib);
        resolve_head_engram_cache(dir, 4);
        assert_eq!(crate::opts::get(o_mib).as_deref(), Some("2048"));
        assert!(crate::opts::snapshot().iter().any(|(k, v)| k == "engram-row-cache-mib" && v == "2048"));
        assert_eq!(super::super::serve::ServeOpts::from_registry_world(4).cache_mib, 2048);

        // Explicit --engram-row-cache-mib on TP2 is preserved
        crate::opts::set(o_mib, "3072");
        resolve_head_engram_cache(dir, 2);
        assert_eq!(crate::opts::get(o_mib).as_deref(), Some("3072"));

        // Explicit --dsv41-engram-ram-gib on TP2 is honored when --engram-row-cache-mib is unset
        crate::opts::unset(o_mib);
        crate::opts::set(o_ram, "4");
        resolve_head_engram_cache(dir, 2);
        assert_eq!(crate::opts::get(o_mib).as_deref(), Some("4096"));

        match saved_mib { Some(v) => crate::opts::set(o_mib, v), None => crate::opts::unset(o_mib) }
        match saved_ram { Some(v) => crate::opts::set(o_ram, v), None => crate::opts::unset(o_ram) }
    }
}
