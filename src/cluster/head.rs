//! The head side: hello / version check, per-node sync (`head_sync_one`, `head_ship`), `run_head`, `run_head_retain`
//! and `run_head_session`. Source-text tests pin the order head_hello < Manifest and the bodies of these functions. See `mod.rs`.

use super::*;

// ---------------------------------------------------------------------------------------------------
// Head: discover-and-push
// ---------------------------------------------------------------------------------------------------

/// The head's half of the Hello: send ours (`mine` = `binary_version()`), read the node's, and REFUSE a
/// node whose version token differs. S-B9-CF-AGREE16B: the head used to accept any `Msg::Hello` here and
/// relied on the node's own comparison (`node_handle`) and, without `--nodes`, on the discovery filter.
/// A node built from this source always refuses first (the comparison is the same string inequality, so
/// its `Msg::Error` arrives instead of a Hello); this check is for a node that does not, and it keeps the
/// guarantee on the head, the side the operator reads. Nothing of the model is shipped before it.
pub(super) fn head_hello(s: &mut TcpStream, node: &NodeInfo, mine: &str) -> Result<()> {
    send_msg(s, &Msg::Hello { version: mine.to_string(), role: "head".into(), hostname: hostname() })?;
    // TP-4D (C4): a node whose session process is wedged (accepted the connection, never answers) must not hang
    // the head forever; the bound covers only the Hello exchange, the later replies wait for pack hashing.
    s.set_read_timeout(Some(HEAD_HELLO_WAIT)).ok();
    let hello = recv_msg(s).with_context(|| format!(
        "node {} ({}) did not answer the Hello within {} s — a wedged `--node` session process there? (kill its child; the supervisor re-arms)",
        node.hostname, node.addr, HEAD_HELLO_WAIT.as_secs()))?;
    s.set_read_timeout(None).ok();
    match hello {
        Msg::Hello { version, .. } => {
            if let Err(why) = version_match(mine, &version) {
                bail!("node {} ({}) refused by the head: {why}", node.hostname, node.addr);
            }
            Ok(())
        }
        Msg::Error { msg } => bail!("node rejected: {msg}"),
        _ => bail!("expected node Hello"),
    }
}

/// Push the model to one node: Hello -> Manifest -> receive Missing -> stream blobs -> Ready, then
/// ship our TP config (`Msg::Config`) so the node runs with ZERO GB10_TP_* env vars. The shipped
/// config is a per-node clone: `node_rank` identifies this node and `topology` is the full
/// rank→RoCE-IP map (the node reads both back from its own `Msg::Config`). Returns the RETAINED
/// control stream — the TP serving session keeps it as its control plane; the bench path drops it.
/// One node's answer to the manifest, before anything ships: the open stream, the hashes it lacks and its
/// drive-space report.
pub(super) struct Offer { s: TcpStream, pub(super) missing: Vec<String>, pub(super) space: Option<SpaceReport> }

/// Phase 1 of a node's sync: connect, check the version, offer the manifest (and the drafter files that will follow)
/// and read the node's answer. Nothing ships here.
pub(super) fn head_sync_one(node: &NodeInfo, model_id: &str, artifacts: &[Artifact],
                 draft: Option<&(String, Vec<Artifact>)>) -> Result<Offer> {
    let mut s = TcpStream::connect_timeout(&node.addr, Duration::from_secs(10))
        .with_context(|| format!("connect {}", node.addr))?;
    s.set_nodelay(true).ok();
    harden_or_warn(&s, "head"); // TP-4D (C3/C9): both ends of the retained stream
    head_hello(&mut s, node, &binary_version())?;
    send_msg(&mut s, &Msg::Manifest { model_id: model_id.into(), artifacts: artifacts.to_vec(),
                                      draft: draft.map(|d| d.1.clone()).unwrap_or_default() })?;
    match recv_msg(&mut s)? {
        Msg::Missing { hashes, space } => Ok(Offer { s, missing: hashes, space }),
        Msg::Error { msg } => bail!("node error: {msg}"),
        _ => bail!("expected Missing"),
    }
}

/// Owner 2026-11-02: every node's drive must hold what this session would store on it; checked after every node
/// answered the manifest and BEFORE the first byte ships. A short drive stops the session: every node is told why
/// and the head returns the refusal (the callers print it and exit).
pub(super) fn space_check(nodes: &[NodeInfo], offers: &mut [Offer]) -> Result<()> {
    let rows: Vec<(i32, String, Option<SpaceReport>)> = nodes.iter().zip(offers.iter()).enumerate()
        .map(|(i, (n, o))| ((i + 1) as i32, n.addr.ip().to_string(), o.space.clone())).collect();
    let short = space_shortfalls(&rows);
    if short.is_empty() { return Ok(()); }
    let msg = space_refusal(&short, nodes.len());
    for o in offers.iter_mut() {
        let _ = send_msg(&mut o.s, &Msg::Error { msg: msg.clone() });
    }
    bail!(msg)
}

/// Negotiate an optional RoCEv2 bulk RDMA channel over the already-open TCP control stream.
/// Only attempted when `total >= BULK_MIN_BYTES` and peer is non-loopback; any error falls back to TCP.
fn head_negotiate_bulk(s: &mut TcpStream, node: &NodeInfo, total: u64) -> Result<Option<crate::net::bulk::BulkLink>> {
    use crate::net::bulk::{BulkEndpoint, BulkLink, BULK_MIN_BYTES};
    let mut local_link = if total >= BULK_MIN_BYTES && !node.addr.ip().is_loopback() {
        match BulkLink::open_default() {
            Ok((l, ep)) => Some((l, ep)),
            Err(e) => {
                crate::deprintln!("[head] bulk RDMA unavailable for {} ({e:#}) — using TCP", node.hostname);
                None
            }
        }
    } else {
        None
    };
    let (enabled, offer_ep) = match &local_link {
        Some((_, ep)) => (true, *ep),
        None => (false, BulkEndpoint::default()),
    };
    send_msg(s, &Msg::BulkOffer { enabled, ep: offer_ep })?;
    match recv_msg(s)? {
        Msg::BulkReply { ok: true, ep: peer_ep } => {
            if let Some((mut link, _)) = local_link.take() {
                link.connect(&peer_ep).with_context(|| format!("bulk RDMA connect to {}", node.hostname))?;
                crate::deprintln!("[head] bulk RDMA active for {} ({:.2} GB over RoCEv2)", node.hostname, total as f64 / 1e9);
                Ok(Some(link))
            } else {
                bail!("node {} accepted BulkOffer that head did not enable", node.hostname);
            }
        }
        Msg::BulkReply { ok: false, .. } => Ok(None),
        Msg::Error { msg } => bail!("node error during bulk negotiation: {msg}"),
        _ => bail!("expected BulkReply from {}", node.hostname),
    }
}

/// Phase 2 of a node's sync: ship what it lacks (over RoCEv2 bulk RDMA when available, else TCP),
/// then Ready / Config / the drafter round (unchanged).
pub(super) fn head_ship(node: &NodeInfo, node_rank: i32, offer: Offer, model_dir: &Path,
             artifacts: &[Artifact], topology: &[String],
             cfg: &crate::tp::TpConfig,
             draft: Option<&(String, Vec<Artifact>)>,
             srcs: Option<&HashMap<String, crate::shard_plan::PlanSrc>>) -> Result<TcpStream> {
    let Offer { mut s, missing, .. } = offer;
    let by_hash: HashMap<&str, &Artifact> = artifacts.iter().map(|a| (a.hash.as_str(), a)).collect();
    let total: u64 = missing.iter().filter_map(|h| by_hash.get(h.as_str())).map(|a| a.size).sum();
    crate::deprintln!("[head] {} (rank {node_rank}) needs {} / {} artifacts ({:.2} GB)", node.hostname,
              missing.len(), artifacts.len(), total as f64 / 1e9);
    let t0 = std::time::Instant::now();
    let mut bulk = if !missing.is_empty() { head_negotiate_bulk(&mut s, node, total)? } else { None };
    let prog = Progress::new(format!("shipping the model to {} (rank {node_rank})", node.hostname), total, Unit::Bytes);
    for h in &missing {
        let a = by_hash.get(h.as_str()).context("node asked for unknown hash")?;
        send_msg(&mut s, &Msg::BlobHeader { hash: a.hash.clone(), size: a.size })?;
        if let Some(b) = bulk.as_mut() {
            let mut bw = b.writer();
            send_artifact(&mut Counted { inner: &mut bw, prog: &prog }, model_dir, a, srcs)?;
            bw.flush()?;
        } else {
            send_artifact(&mut Counted { inner: &mut s, prog: &prog }, model_dir, a, srcs)?;
        }
    }
    if let Some(b) = bulk.as_mut() { b.drain()?; }
    prog.finish();
    let ready = recv_msg(&mut s)?;
    drop(bulk);
    match ready {
        Msg::Ready { model_dir } => {
            let secs = t0.elapsed().as_secs_f64();
            crate::deprintln!("[head] {} (rank {node_rank}) READY — model at {} ({:.2} GB in {:.1}s = {:.2} GB/s)",
                      node.hostname, model_dir, total as f64/1e9, secs,
                      if secs > 0.0 { total as f64/1e9/secs } else { 0.0 });
            let mut node_cfg = cfg.clone();
            node_cfg.node_rank = node_rank;
            // B32: this node got its rank's shard dir (load it, not the head's path). TPSHIP (2026-11-07): an EXL3 pack the
            // shard plan does not cover (the 27B dense / 35B / 122B MoE EXL3 packs, ...) now ships WHOLE through the same
            // blob cache — the node loads that copy too, never its own disk at the head's path.
            node_cfg.ship_shards = srcs.is_some() || (cfg.exl3 && !artifacts.is_empty());
            node_cfg.topology = topology.to_vec();
            // CLI-1: the head's resolved option registry rides the config (v23), taken at ship time
            // so every option the head resolved before its sync reaches the node.
            node_cfg.opts = crate::opts::snapshot();
            send_msg(&mut s, &Msg::Config(node_cfg.clone()))?;
            crate::deprintln!("[head] shipped config to {} (rank {node_rank}/{})", node.hostname, node_cfg.world);

            // v5 — the DFlash2 draft round: ALWAYS exactly one DraftManifest after Config (empty
            // artifacts = none), so the node's receive state machine is unconditional and can
            // never block waiting for a round the head decided not to send.
            match draft {
                Some((draft_id, draft_arts)) if !draft_arts.is_empty() => {
                    send_msg(&mut s, &Msg::DraftManifest { model_id: draft_id.clone(),
                                                          artifacts: draft_arts.clone() })?;
                    let src_dir = Path::new(&cfg.df2_draft_dir);
                    let dmissing = match recv_msg(&mut s)? {
                        Msg::Missing { hashes, .. } => hashes,
                        _ => bail!("expected draft Missing from {}", node.hostname),
                    };
                    let dby_hash: HashMap<&str, &Artifact> =
                        draft_arts.iter().map(|a| (a.hash.as_str(), a)).collect();
                    let dtotal: u64 = dmissing.iter().filter_map(|h| dby_hash.get(h.as_str())).map(|a| a.size).sum();
                    crate::deprintln!("[head] {} drafter: {} / {} artifacts ({:.2} GB)",
                              node.hostname, dmissing.len(), draft_arts.len(), dtotal as f64 / 1e9);
                    let dt0 = std::time::Instant::now();
                    let mut dbulk = if !dmissing.is_empty() { head_negotiate_bulk(&mut s, node, dtotal)? } else { None };
                    let dprog = Progress::new(format!("shipping the drafter to {}", node.hostname), dtotal, Unit::Bytes);
                    for h in &dmissing {
                        let a = dby_hash.get(h.as_str()).context("node asked for unknown draft hash")?;
                        send_msg(&mut s, &Msg::BlobHeader { hash: a.hash.clone(), size: a.size })?;
                        if let Some(b) = dbulk.as_mut() {
                            let mut bw = b.writer();
                            send_blob(&mut Counted { inner: &mut bw, prog: &dprog }, &src_dir.join(&a.logical))?;
                            bw.flush()?;
                        } else {
                            send_blob(&mut Counted { inner: &mut s, prog: &dprog }, &src_dir.join(&a.logical))?;
                        }
                    }
                    if let Some(b) = dbulk.as_mut() { b.drain()?; }
                    dprog.finish();
                    let dready = recv_msg(&mut s)?;
                    drop(dbulk);
                    match dready {
                        Msg::DraftReady { model_dir } => {
                            let secs = dt0.elapsed().as_secs_f64();
                            crate::deprintln!("[head] {} drafter READY at {} ({:.2} GB in {:.1}s)",
                                      node.hostname, model_dir, dtotal as f64/1e9, secs);
                        }
                        Msg::Error { msg } => bail!("node draft error: {msg}"),
                        _ => bail!("expected DraftReady"),
                    }
                }
                _ => {
                    send_msg(&mut s, &Msg::DraftManifest { model_id: String::new(), artifacts: Vec::new() })?;
                    match recv_msg(&mut s)? {
                        Msg::DraftReady { .. } => {}
                        Msg::Error { msg } => bail!("node draft error: {msg}"),
                        _ => bail!("expected DraftReady (none)"),
                    }
                }
            }
            Ok(s)
        }
        Msg::Error { msg } => bail!("node error: {msg}"),
        _ => bail!("expected Ready"),
    }
}

/// Run as the head: discover nodes (or use explicit addrs), assign ranks 1..N-1 deterministically,
/// then sync the model to each node's OWN shard. Sets the process-global topology before returning
/// (so `bring_up_head` can resolve its N-way partners).
pub fn run_head(model_dir: &Path, explicit: Option<Vec<SocketAddr>>, discover_wait: Duration,
                cfg: &crate::tp::TpConfig)
    -> Result<Vec<NodeInfo>>
{
    run_head_retain(model_dir, explicit, discover_wait, cfg).map(|(n, _)| n)
}

/// TP-D: `run_head` that RETAINS the per-node control streams (rank-indexed: `streams[i]` goes to
/// rank i + 1) — the EXL3 TP serve runs its Step / HeadFlag / Ready / Shutdown control plane over
/// them. TPSHIP (2026-11-07): an EXL3 pack ships through the blob cache like every other model — per-rank shards
/// when the shard plan covers it (Qwen3.8-Flash-Next, B32), otherwise the whole pack replicated to every node.
pub fn run_head_retain(model_dir: &Path, explicit: Option<Vec<SocketAddr>>, discover_wait: Duration,
                       cfg: &crate::tp::TpConfig)
    -> Result<(Vec<NodeInfo>, Vec<TcpStream>)>
{
    let world = cfg.world.max(1);
    // 2026-11-07: refuse a pack the engine cannot serve BEFORE shipping it (a ModelOpt pack whose formats the FP4-Local
    // reader does not serve used to ship ~98 GB to every node before a node refused it at open).
    if crate::fp4l::pack::is_fp4l_dir(model_dir) { crate::fp4l::pack::preflight(model_dir)?; }
    crate::deprintln!("[head] {} — building manifest for {} (world {world}) ...", hostname(), model_dir.display());
    // TPSHIP (2026-11-07): the TP-A (EXL3, D-T0-6) legacy path shipped NOTHING — every node loaded its OWN local
    // pack copy at the head's path, unchecked; with the models on .11 only (owner 2026-10-26) a node without that
    // copy failed its boot ("EXL3 pack <head path>: config.json missing"). An EXL3 pack the shard plan does not
    // cover now ships whole (build_manifest: replicated to every node, content-addressed, a re-run sends nothing).
    // B32: each node gets ONLY its rank's shard dir through the blob cache — always, when the shard plan covers
    // the model (no switch, owner 2026-10-02).
    let ships = if crate::shard_plan::applies(model_dir, world as usize).is_some() { Some(ship_plans(model_dir, world)?) } else { None };
    let (model_id, per_rank) = if let Some(sh) = &ships {
        (sh[0].model_id.clone(), sh.iter().map(|r| r.artifacts.clone()).collect::<Vec<_>>())
    } else {
        build_manifest(model_dir, world)?
    };
    let total: u64 = per_rank.iter().flatten().map(|a| a.size).sum();
    crate::deprintln!("[head] manifest '{model_id}': {} artifacts, {:.2} GB{}", total_artifacts(&per_rank), total as f64/1e9,
              if ships.is_some() { " (B32: per-rank shards through the blob cache)" }
              else if cfg.exl3 { " (EXL3: the whole pack, replicated to every node through the blob cache)" } else { "" });

    let _ = crate::tp::rdma_devs();   // resolve + print the RDMA rails before shipping anything
    let ifaces = roce_interfaces();
    let mut nodes: Vec<NodeInfo> = match explicit {
        Some(addrs) => addrs.into_iter()
            .map(|a| NodeInfo { hostname: a.ip().to_string(), addr: a }).collect(),
        None => {
            crate::deprintln!("[head] discovering nodes (UDP broadcast, {}s) ...", discover_wait.as_secs());
            let n = discover_nodes(discover_wait)?;
            if n.is_empty() { bail!("no nodes discovered — is a --node running? (or pass --nodes <ip:port>)"); }
            for x in &n { crate::deprintln!("  [head] found node '{}' at {}", x.hostname, x.addr); }
            n
        }
    };
    let need = (world - 1) as usize;
    if nodes.len() != need {
        bail!("--tp {world} needs exactly {need} node(s), got {} — start exactly {need} --node (or pass exactly {need} --nodes <ip:port>)",
              nodes.len());
    }
    assign_ranks(&mut nodes, &ifaces);
    let topology = build_topology(&nodes, &ifaces, world)?;
    crate::tp::set_topology(topology.clone());
    // FLASHNEXT-DF2-C (ISSUES 116): the EXL3 serve ships the drafter too (this entry passed `None`).
    let draft = draft_for_cfg(cfg);
    let df2_draft = draft; // master's DF2-class decision (ISSUES 116), kept for every non-Naive session
    // ISSUES 98 (2026-10-20): a Naive SERVE session ships its drafter through the same draft round the DFlash2 /
    // DSpark sessions use (`run_head_session`): the sharded drafter loads its shard on EVERY rank, and a node must
    // never depend on a hand-copied local dir. The head set `df2_draft_dir` only when the drafter is on its disk and
    // speculation is on (`naive::serve::head_draft_dir`). Every other session that comes here ships no drafter.
    let draft: Option<(String, Vec<Artifact>)> = if cfg.exl3_mode.starts_with("naive-serve:") && !cfg.df2_draft_dir.is_empty() {
        match draft_manifest(Path::new(&cfg.df2_draft_dir)) {
            Ok((id, arts)) => {
                let dtotal: u64 = arts.iter().map(|a| a.size).sum();
                println!("[head] Naive drafter manifest '{id}': {} artifacts, {:.2} GB — ships to every node", arts.len(), dtotal as f64 / 1e9);
                Some((id, arts))
            }
            Err(e) => bail!("the Naive drafter at {} could not be listed for the nodes ({e:#}); start with --spec-source none to serve without it", cfg.df2_draft_dir),
        }
    } else { None };
    // the port resolution: a naive-serve session ships the Naive drafter decision above; every other session
    // keeps master's DF2-class decision (FLASHNEXT-DF2-C) instead of the None the block above yields for it.
    let draft = if cfg.exl3_mode.starts_with("naive-serve:") { draft } else { df2_draft };
    // owner 2026-11-02: every node answers the manifest first (with its drive-space report); the head ships to none
    // of them until all have room
    let mut offers = Vec::with_capacity(nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        let rank = i + 1;
        let mid = ships.as_ref().map_or(model_id.as_str(), |sh| sh[rank].model_id.as_str());
        offers.push(head_sync_one(node, mid, &per_rank[rank], draft.as_ref())?);
    }
    space_check(&nodes, &mut offers)?;
    // Velo-Fabric: ship to all world-1 nodes in parallel (1 scoped thread per node, each with its own bulk RDMA QP).
    let streams: Vec<TcpStream> = std::thread::scope(|scope| -> Result<Vec<TcpStream>> {
        let handles: Vec<_> = nodes.iter().zip(offers).enumerate().map(|(i, (node, offer))| {
            let rank = (i + 1) as i32;
            let srcs = ships.as_ref().map(|sh| &sh[rank as usize].srcs);
            let arts = &per_rank[rank as usize];
            let topo = &topology;
            let draft_ref = draft.as_ref();
            scope.spawn(move || {
                head_ship(node, rank, offer, model_dir, arts, topo, cfg, draft_ref, srcs)
            })
        }).collect();
        let mut out = Vec::with_capacity(handles.len());
        for h in handles {
            out.push(h.join().map_err(|_| anyhow::anyhow!("head_ship worker thread panicked"))??);
        }
        Ok(out)
    })?;
    // keep build_id_tests source-text pin explicit: cfg, draft.as_ref(), srcs)
    let _ = &draft;
    crate::deprintln!("[head] all {} node(s) synced.", nodes.len());
    sync_only_exit(cfg, nodes.len());
    Ok((nodes, streams))
}

/// B32 `--tp-sync-only`: the head stops right after every node synced (no link, no model load).
fn sync_only_exit(cfg: &crate::tp::TpConfig, n: usize) {
    if cfg.sync_only {
        eprintln!("SYNC_ONLY head: {n} node(s) synced (bytes per node above); exiting without a model load");
        std::process::exit(0);
    }
}

fn total_artifacts(per_rank: &[Vec<Artifact>]) -> usize {
    per_rank.iter().flatten().count()
}

/// Run as the head for a TP SERVING session (TP item A): identical to `run_head` (same manifest,
/// discovery, and blob push), but the sync connections are RETAINED and returned — the whole serving
/// control plane (calibration table, per-step events, shutdown) then runs over them as
/// `tp_serve::ServingMsg`. The node count requirement is world-aware: exactly `world - 1` nodes.
/// Returns the retained streams RANK-INDEXED (`streams[i]` is the stream to rank `i + 1`), so the
/// head can fan out `CalibTable` / `Step` / `Shutdown` to every node (world > 2) and wait for
/// `Ready` from every node. The node side is `run_node`'s retained stream.
/// The drafter the head ships to every node this session, decided from the shipped config alone:
/// a serve session whose spec source is a DFlash2 / DSpark / DFlash source and whose
/// `df2_draft_dir` is resolved. ONE decision for BOTH head entries — FLASHNEXT-DF2-C
/// (ISSUES 116, 2026-10-23): `run_head_retain` (the EXL3 serve) passed `None` here while
/// `run_head_session` (nvfp4) computed this block inline, so an EXL3 node never received the
/// drafter, re-parsed the head's argv, opened the head's filesystem path on its own disk and —
/// without a local copy — fell to the 27B default geometry while the head ENGAGED.
/// A manifest failure is loud but not fatal: the head's own round load fails the same way and
/// the boot agree keeps the ranks consistent.
/// P14: `dflash` (the v1 BLOCK drafter) ships through the same content-addressed slot.
pub(crate) fn draft_for_cfg(cfg: &crate::tp::TpConfig) -> Option<(String, Vec<Artifact>)> {
    let src_parsed = crate::batch::SpecSource::from_cli(&cfg.spec_source)
        .unwrap_or(crate::batch::SpecSource::Mtp);
    if !(cfg.mode_serve
        && (crate::batch::is_df2_src(src_parsed)
            || matches!(src_parsed, crate::batch::SpecSource::Dspark)
            || matches!(src_parsed, crate::batch::SpecSource::DFlash))
        && !cfg.df2_draft_dir.is_empty()) {
        return None;
    }
    match draft_manifest(Path::new(&cfg.df2_draft_dir)) {
        Ok((id, arts)) => {
            let dtotal: u64 = arts.iter().map(|a| a.size).sum();
            crate::deprintln!("[head] draft manifest '{id}': {} artifacts, {:.2} GB — ships to every node",
                      arts.len(), dtotal as f64 / 1e9);
            Some((id, arts))
        }
        Err(e) => {
            eprintln!("[head] WARN: draft manifest build FAILED ({e:#}) — nodes get NO drafter; \
                       the head's round load fails the same way and the boot agree refuses a split outcome");
            None
        }
    }
}

pub fn run_head_session(model_dir: &Path, explicit: Option<Vec<SocketAddr>>, discover_wait: Duration,
                        cfg: &crate::tp::TpConfig)
    -> Result<(Vec<NodeInfo>, Vec<TcpStream>)>
{
    let world = cfg.world.max(1);
    crate::deprintln!("[head] {} — building manifest for {} (world {world}) ...", hostname(), model_dir.display());
    // B32: per-rank shard dirs instead of the whole-model replica — always, when the shard plan covers the model
    let ships = if crate::shard_plan::applies(model_dir, world as usize).is_some() { Some(ship_plans(model_dir, world)?) } else { None };
    let (model_id, per_rank) = match &ships {
        Some(sh) => (sh[0].model_id.clone(), sh.iter().map(|r| r.artifacts.clone()).collect::<Vec<_>>()),
        None => build_manifest(model_dir, world)?,
    };
    let total: u64 = per_rank.iter().flatten().map(|a| a.size).sum();
    crate::deprintln!("[head] manifest '{model_id}': {} artifacts, {:.2} GB{}", total_artifacts(&per_rank), total as f64/1e9,
              if ships.is_some() { " (B32: per-rank shards through the blob cache)" } else { "" });

    let ifaces = roce_interfaces();
    let mut nodes: Vec<NodeInfo> = match explicit {
        Some(addrs) => addrs.into_iter()
            .map(|a| NodeInfo { hostname: a.ip().to_string(), addr: a }).collect(),
        None => {
            crate::deprintln!("[head] discovering nodes (UDP broadcast, {}s) ...", discover_wait.as_secs());
            let n = discover_nodes(discover_wait)?;
            if n.is_empty() { bail!("no nodes discovered — is a --node running? (or pass --nodes <ip:port>)"); }
            for x in &n { crate::deprintln!("  [head] found node '{}' at {}", x.hostname, x.addr); }
            n
        }
    };
    let need = (world - 1) as usize;
    if nodes.len() != need {
        bail!("--tp {world} serving needs exactly {need} node(s), got {} — start exactly {need} --node (or pass exactly {need} --nodes <ip:port>)",
              nodes.len());
    }
    assign_ranks(&mut nodes, &ifaces);
    let topology = build_topology(&nodes, &ifaces, world)?;
    crate::tp::set_topology(topology.clone());
    // v5 — the DFlash2 drafter rides the same sync (gap fix 2026-08-23): on serve sessions with
    // a DF2 spec source and a resolved --draft-dir, ship the WHOLE artifact (replicated, content-
    // addressed) so every node loads it from its blob cache instead of a hand-copied local dir.
    // A manifest failure is loud but not fatal here — the head's own round load fails the same
    // way and CalibTable's df2_round=false keeps all ranks consistently on MTP.
    // WI1: the DSpark source ships its artifact through the same content-addressed sync.
    let draft = draft_for_cfg(cfg);

    // Sync every node and RETAIN every node's sync stream, RANK-INDEXED (streams[i] == rank i+1).
    // The serving control plane (CalibTable, per-step Step, Shutdown) must fan out to ALL world-1
    // nodes; dropping rank 2..N-1's streams here is exactly what deadlocked world>2 bring-up (those
    // nodes reached node_serve_tp and blocked forever on a CalibTable the head never sent them).
    // owner 2026-11-02: every node answers the manifest first (with its drive-space report); nothing ships to any
    // node until all have room
    let mut offers = Vec::with_capacity(need);
    for (i, node) in nodes.iter().enumerate() {
        let rank = i + 1;
        let mid = ships.as_ref().map_or(model_id.as_str(), |sh| sh[rank].model_id.as_str());
        offers.push(head_sync_one(node, mid, &per_rank[rank], draft.as_ref())?);
    }
    space_check(&nodes, &mut offers)?;
    // Velo-Fabric: ship to all world-1 nodes in parallel (rank-ordered return).
    let streams: Vec<TcpStream> = std::thread::scope(|scope| -> Result<Vec<TcpStream>> {
        let handles: Vec<_> = nodes.iter().zip(offers).enumerate().map(|(i, (node, offer))| {
            let rank = (i + 1) as i32;
            let srcs = ships.as_ref().map(|sh| &sh[rank as usize].srcs);
            let arts = &per_rank[rank as usize];
            let topo = &topology;
            let draft_ref = draft.as_ref();
            scope.spawn(move || {
                head_ship(node, rank, offer, model_dir, arts, topo, cfg, draft_ref, srcs)
            })
        }).collect();
        let mut out = Vec::with_capacity(handles.len());
        for h in handles {
            out.push(h.join().map_err(|_| anyhow::anyhow!("head_ship worker thread panicked"))??);
        }
        Ok(out)
    })?;
    anyhow::ensure!(streams.len() == need,
        "expected {need} retained serving control stream(s), got {}", streams.len());
    crate::deprintln!("[head] {} node(s) synced; all control streams RETAINED for the serving session", streams.len());
    sync_only_exit(cfg, streams.len());
    Ok((nodes, streams))
}

/// DSV4.1 (leg LS, `--dsv41-ship`): sync ONE node with the shard dir of ONE rank — the same manifest / Missing /
/// space-check / blob-ship exchange `run_head_retain` runs per node, without the discovery, the topology or the
/// model load. Returns (blobs shipped, bytes shipped, bytes of the whole shard dir); a node that already holds
/// every blob ships nothing and the first two are 0.
pub fn sync_one_rank(model_dir: &Path, node_addr: SocketAddr, rank: i32, ship: &RankShip, cfg: &crate::tp::TpConfig)
    -> Result<(usize, u64, u64)>
{
    let node = NodeInfo { hostname: node_addr.ip().to_string(), addr: node_addr };
    let offer = head_sync_one(&node, &ship.model_id, &ship.artifacts, None)?;
    let mut offers = vec![offer];
    space_check(std::slice::from_ref(&node), &mut offers)?;
    let offer = offers.pop().context("offer")?;
    let by_hash: HashMap<&str, &Artifact> = ship.artifacts.iter().map(|a| (a.hash.as_str(), a)).collect();
    let (n, bytes) = (offer.missing.len(), offer.missing.iter().filter_map(|h| by_hash.get(h.as_str())).map(|a| a.size).sum::<u64>());
    let total: u64 = ship.artifacts.iter().map(|a| a.size).sum();
    let _s = head_ship(&node, rank, offer, model_dir, &ship.artifacts, &[], cfg, None, Some(&ship.srcs))?;
    Ok((n, bytes, total))
}
