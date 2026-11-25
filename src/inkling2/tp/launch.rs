//! Head / node bring-up of an Inkling-2 TP session and the node's replay loop.
//!
//! Head: `--server --model-dir <stock pack> --tp 2|4 [--nodes ip[:port],..]` (`.11` by default, AGENTS §1.0); every
//! other box runs the plain zero-config `--node --port 29500`. The pack ships through the cluster sync's
//! content-addressed blob cache (a node that holds it by hash is sent nothing); the head's options ride `TpConfig`
//! (the option registry snapshot) and the serve argv rides `exl3_mode` (`inkling2-serve:<json argv>`), so the node
//! sizes its lanes / context exactly as the head does. The build-id handshake of the sync refuses a head and a node
//! from different builds, and `TpBranch::Inkling2` refuses a node that would run another family's program.

use super::shard::Rank;
use super::xport::Xport;
use super::{Op, TpRank};
use crate::inkling2::model::{GpuModel, Lane, Seg};
use anyhow::{bail, ensure, Context, Result};
use std::path::Path;

pub const SERVE_PREFIX: &str = "inkling2-serve:";

fn nodes_of(args: &[String]) -> Result<Option<Vec<std::net::SocketAddr>>> {
    let Some(s) = crate::arg_value(args, "--nodes") else { return Ok(None) };
    let mut v = Vec::new();
    for p in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        v.push(if p.contains(':') { p.parse().with_context(|| format!("--nodes {p}"))? }
               else { std::net::SocketAddr::new(p.parse().with_context(|| format!("--nodes {p}"))?, 29500) });
    }
    Ok(Some(v))
}

/// The pieces the serve worker needs on the head: the model (rank 0, its `tp` attached) on device 0.
pub struct HeadBoot { pub model: GpuModel, pub world: usize, pub describe: String }

/// The head of an Inkling-2 TP session: ship, link, load rank 0, wait for every node's Ready.
pub fn boot_head(args: &[String], model_dir: &Path, world: usize) -> Result<HeadBoot> {
    ensure!(matches!(world, 2 | 3 | 4 | 6 | 8), "inkling2 TP: --tp 2, 3, 4, 6, or 8 (got {world})");
    let dir_abs = std::fs::canonicalize(model_dir).with_context(|| format!("--model-dir {}", model_dir.display()))?;
    let explicit = nodes_of(args)?;
    let wait = std::time::Duration::from_secs(crate::arg_value(args, "--discover-wait").and_then(|s| s.parse().ok()).unwrap_or(3));
    let mut tpc = crate::tp::TpConfig::from_opts();
    tpc.world = world as u32;
    tpc.mode_serve = false;
    tpc.exl3_mode = format!("{SERVE_PREFIX}{}", serde_json::to_string(args).context("argv json")?);
    crate::tp::set_tp_config(tpc.clone());
    crate::boot_report::notice(&format!("inkling2 TP={world}: shipping the pack {} to the node(s) through the blob cache (a node that holds it is sent nothing)", dir_abs.display()));
    let t0 = std::time::Instant::now();
    let (nodes, mut streams) = crate::cluster::run_head_retain(&dir_abs, explicit, wait, &tpc)?;
    crate::boot_report::notice(&format!("inkling2 TP={world}: {} node(s) synced in {:.1} s: {}", nodes.len(), t0.elapsed().as_secs_f64(),
                                        nodes.iter().map(|n| n.addr.to_string()).collect::<Vec<_>>().join(", ")));
    let watch = crate::cluster::watch_nodes_during_bring_up(&streams)?;
    let dev = cudarc::driver::CudaDevice::new(0).context("CUDA device 0")?;
    // Preload both CUDA PTX modules BEFORE RDMA TCP handshake so cold-JIT peers finish compiling before QPs go live.
    let _ = crate::inkling2::gpu::Ink2Kernels::global(&dev)?;
    let _ = crate::dsv41::gpu::raw_fn("d41x_xc2_stage");
    let ctx = crate::tp::TpContext::bring_up_head(world as i32)?;
    let x = Xport::attach(&dev, ctx)?;
    watch.disarm(&streams);
    let cfg = crate::inkling2::config::Inkling2Config::load(&dir_abs)?;
    let rank = Rank::new(0, world, cfg.inter, cfg.unpadded_vocab)?;
    let describe = format!("{}; {}", rank.describe(), x.describe);
    let mut g = GpuModel::load_rank(&dir_abs, &dev, Some(rank))?;
    for (i, s) in streams.iter_mut().enumerate() {
        match crate::tp_serve::recv_serving(s).with_context(|| format!("node rank {} Ready", i + 1))? {
            crate::tp_serve::ServingMsg::Ready => {}
            other => bail!("expected Ready from node rank {}, got {other:?}", i + 1),
        }
    }
    g.tp = Some(TpRank::new(&dev, x, Some(streams))?);
    Ok(HeadBoot { model: g, world, describe })
}

/// The node side (called by `tp_cluster`'s node arm for an `exl3_mode` starting `inkling2-`).
pub fn run_node(tpc: &crate::tp::TpConfig, dir: &Path, head_ip: std::net::IpAddr, mut stream: std::net::TcpStream) -> Result<()> {
    crate::net::set_head_liveness(&stream);
    let fail = |stream: &mut std::net::TcpStream, e: anyhow::Error| -> anyhow::Error { crate::cluster::node_report_failure(stream, &format!("{e:#}")); e };
    let Some(j) = tpc.exl3_mode.strip_prefix(SERVE_PREFIX) else {
        return Err(fail(&mut stream, anyhow::anyhow!("inkling2 node: unknown program '{}' (this build runs '{SERVE_PREFIX}')", tpc.exl3_mode)));
    };
    let argv: Vec<String> = match serde_json::from_str(j) { Ok(v) => v, Err(e) => return Err(fail(&mut stream, e.into())) };
    let dir_s = dir.to_string_lossy().to_string();
    let su = match crate::inkling2::serve::setup(&argv, &dir_s) { Ok(s) => s, Err(e) => return Err(fail(&mut stream, e)) };
    if let Err(e) = crate::inkling2::actq::set_mode_from_args(&argv) {
        return Err(fail(&mut stream, e));
    }
    if let Err(e) = crate::inkling2::model::set_ulysses_wire_from_args(&argv) {
        return Err(fail(&mut stream, e));
    }
    let dev = match cudarc::driver::CudaDevice::new(0).context("CUDA device 0") {
        Ok(d) => d,
        Err(e) => return Err(fail(&mut stream, e)),
    };
    // Preload both CUDA PTX modules BEFORE RDMA TCP handshake so cold JIT never stalls RDMA self-tests.
    if let Err(e) = crate::inkling2::gpu::Ink2Kernels::global(&dev) {
        return Err(fail(&mut stream, e.context("node CUDA PTX preload failed")));
    }
    let _ = crate::dsv41::gpu::raw_fn("d41x_xc2_stage");
    let ctx = match crate::tp::TpContext::bring_up_node(head_ip, tpc.node_rank, tpc.world as i32) {
        Ok(c) => c,
        Err(e) => return Err(fail(&mut stream, e.context("the node's RDMA data-plane link bring-up failed"))),
    };
    let init = (|| -> Result<(GpuModel, crate::inkling2::model::Work, Vec<Lane>, Option<crate::inkling2::mtp::spec::Spec>)> {
        let x = Xport::attach(&dev, ctx)?;
        let cfg = crate::inkling2::config::Inkling2Config::load(dir)?;
        let rank = Rank::new(x.rank, x.world, cfg.inter, cfg.unpadded_vocab)?;
        crate::dprintln!("[inkling2-node] {} ; pack {}", rank.describe(), dir.display());
        let mut g = GpuModel::load_rank(dir, &dev, Some(rank))?;
        let spec_on = crate::inkling2::serve::spec_opts(&argv, &su.mtp, dir).is_some();
        g.decode_tc = argv.iter().any(|a| a == "--inkling2-decode-attn-tc") && !spec_on;
        g.attn_single_pass = argv.iter().any(|a| a == "--inkling2-attn-single-pass");
        g.tp = Some(TpRank::new(&dev, x, None)?);
        let mut ws = g.new_work()?;
        let mut lanes: Vec<Lane> = Vec::new();
        for _ in 0..su.lanes { lanes.push(g.new_lane(su.settings.max_seq_len)?); }
        g.bind_lanes(&mut ws, &lanes)?;
        let spec = if spec_on {
            let pack = crate::inkling2::pack::Inkling2Pack::open(dir)?;
            Some(crate::inkling2::mtp::spec::Spec::new(&g, &pack, &mut ws, su.lanes, su.settings.max_seq_len)?)
        } else { None };
        Ok((g, ws, lanes, spec))
    })();
    let (g, mut ws, mut lanes, mut spec) = match init {
        Ok(v) => v,
        Err(e) => return Err(fail(&mut stream, e)),
    };
    crate::tp_serve::send_serving(&mut stream, &crate::tp_serve::ServingMsg::Ready)?;
    let _ = stream.set_nodelay(true);
    crate::dprintln!("[inkling2-node] rank {}: Ready, {} lanes x {} positions (spec: {}) — replaying the head",
                     g.rank.rank, su.lanes, su.settings.max_seq_len, if spec.is_some() { "vocab-parallel" } else { "off" });
    let r = node_loop(&g, &mut ws, &mut lanes, spec.as_mut(), &mut stream);
    if let Err(e) = &r { eprintln!("[inkling2-node] rank {}: stopping: {e:#}", g.rank.rank); }
    r
}

/// Replay the head's calls until Shutdown (or the head goes away).
pub fn node_loop(g: &GpuModel, ws: &mut crate::inkling2::model::Work, lanes: &mut [Lane],
                 mut spec: Option<&mut crate::inkling2::mtp::spec::Spec>, stream: &mut std::net::TcpStream) -> Result<()> {
    use crate::inkling2::gpu::dp;
    use crate::inkling2::model::D;
    let mut snaps: Vec<Option<crate::inkling2::model::prefix::Snap>> = (0..lanes.len()).map(|_| None).collect();
    let mut mtp_snaps: Vec<Option<crate::inkling2::mtp::gpu::MtpSnap>> = (0..lanes.len()).map(|_| None).collect();
    let mut prev: Vec<u32> = Vec::new();
    let tp = g.tp.as_ref().context("node without TP state")?;
    loop {
        let op: Op = match crate::cluster::recv_json(stream) {
            Ok(o) => o,
            Err(e) => return Err(e.context("the control stream from the head closed (head gone)")),
        };
        match op {
            Op::Forward { segs, ids, out, img } => {
                let s: Vec<Seg> = segs.iter().map(|&(lane, p0, m)| Seg { lane, p0, m }).collect();
                // the head's image feature rows for this chunk (kept alive until the forward's launches are done)
                let mut bufs = Vec::new();
                let mut spl = Vec::new();
                for (r0, f) in &img {
                    let b = g.dev.htod_sync_copy(f)?;
                    spl.push((*r0, f.len() / D, dp(&b)));
                    bufs.push(b);
                }
                ws.set_img_chunk(&spl, 0, ids.len());
                let refs: Vec<&Lane> = lanes.iter().collect();
                g.forward(ws, &refs, &s, &ids, &out)?;
                if !bufs.is_empty() { g.k.sync()?; }
                drop(bufs);
            }
            Op::Verify { segs, graph } => {
                for (l, _, p) in &segs { lanes[*l].pos = *p; }
                let sv: Vec<(usize, &[u32])> = segs.iter().map(|(l, t, _)| (*l, &t[..])).collect();
                g.verify_lanes(ws, lanes, &sv, graph)?;
            }
            Op::GatherLogits { n } => { g.logits_full(ws, n)?; }
            Op::Sample { t, seeds, check } => {
                ensure!(check == prev, "INKLING2 TP LOCKSTEP BROKEN: the head drew {check:?} at the previous step, this rank {prev:?}");
                prev = g.sample(ws, &t, &seeds)?;
            }
            Op::Logits { n } => { g.logits_host(ws, n)?; }
            Op::DraftMulti { reqs } => {
                let sp = spec.as_deref_mut().context("node received DraftMulti without drafter loaded")?;
                let rq: Vec<crate::inkling2::mtp::gpu::DraftReq> = reqs.into_iter()
                    .map(|(lane, anchor, p, t, seed, cap)| crate::inkling2::mtp::gpu::DraftReq { lane, anchor, p, t, seed, cap })
                    .collect();
                sp.dev.draft_multi_node(g, &mut sp.batch, &mut sp.w, &sp.lanes, &rq, stream)?;
            }
            Op::MtpAccept { lane, r0, c, t, seed, p0 } => {
                let sp = spec.as_deref_mut().context("node received MtpAccept without drafter loaded")?;
                let vn = sp.dev.n;
                let (acc, _nt) = sp.dev.accept_tp(g, &mut sp.w[lane], dp(&ws.logits) + (r0 * vn * 4) as u64, c, t, seed)?;
                let taps = dp(ws.taps.as_ref().context("node MtpAccept without taps")?);
                sp.dev.commit_ctx(g.k, &mut sp.w[lane], &sp.lanes[lane], taps + (r0 * 4 * D * 2) as u64, acc + 1, p0)?;
                lanes[lane].pos = p0 + acc + 1;
            }
            Op::MtpCommit { lane, tap_off, rows, p0 } => {
                let sp = spec.as_deref_mut().context("node received MtpCommit without drafter loaded")?;
                let taps = dp(ws.taps.as_ref().context("node MtpCommit without taps")?);
                sp.dev.commit_ctx(g.k, &mut sp.w[lane], &sp.lanes[lane], taps + (tap_off * 4 * D * 2) as u64, rows, p0)?;
            }
            Op::MtpGatherStep { lane, c } => {
                let sp = spec.as_deref_mut().context("node received MtpGatherStep without drafter loaded")?;
                let _ = g.gather_vocab_rows(dp(&sp.w[lane].step), c)?;
            }
            Op::SnapSave { lane, pos } => {
                lanes[lane].pos = pos;
                if snaps[lane].is_none() { snaps[lane] = Some(g.new_snap(&lanes[lane])?); }
                g.snap_save(&lanes[lane], snaps[lane].as_mut().unwrap())?;
                if let Some(sp) = spec.as_deref() {
                    if mtp_snaps[lane].is_none() { mtp_snaps[lane] = Some(sp.dev.new_snap(&g.dev, &sp.lanes[lane])?); }
                    sp.dev.snap_copy(&sp.lanes[lane], mtp_snaps[lane].as_mut().unwrap(), true)?;
                }
            }
            Op::SnapRestore { lane, p } => {
                let s = snaps[lane].as_ref().context("prefix restore of a lane this rank never saved")?;
                g.snap_restore(&lanes[lane], s, p)?;
                if let Some(sp) = spec.as_deref() {
                    let ms = mtp_snaps[lane].as_ref().context("prefix restore of drafter lane never saved")?;
                    sp.dev.snap_copy(&sp.lanes[lane], ms, false)?;
                }
            }
            Op::Shutdown => return Ok(()),
        }
        tp.x.check("in the node's replay loop")?;
    }
}
