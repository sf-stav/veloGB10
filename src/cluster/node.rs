//! Blob streaming and the node side of the bring-up: `send_blob` / `recv_blob`, `assemble_model_dir`, `node_handle`
//! (the version check comes before any manifest) and `run_node`. See `mod.rs` for the map.

use super::*;

// ---------------------------------------------------------------------------------------------------
// Blob streaming
// ---------------------------------------------------------------------------------------------------

pub(super) fn send_blob(w: &mut impl Write, path: &Path) -> Result<()> {
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; 4 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        w.write_all(&buf[..n])?;
    }
    w.flush()?;
    Ok(())
}

/// Node half of Velo-Fabric bulk RDMA negotiation: if head offered RDMA, try opening/connecting a
/// `BulkLink` on the primary RoCEv2 device; any failure cleanly replies `ok: false` to fall back to TCP.
fn node_negotiate_bulk(s: &mut TcpStream) -> Result<Option<crate::net::bulk::BulkLink>> {
    use crate::net::bulk::{BulkEndpoint, BulkLink};
    match recv_msg(s)? {
        Msg::BulkOffer { enabled: true, ep: head_ep } => {
            match BulkLink::open_default().and_then(|(mut l, ep)| { l.connect(&head_ep)?; Ok((l, ep)) }) {
                Ok((link, local_ep)) => {
                    send_msg(s, &Msg::BulkReply { ok: true, ep: local_ep })?;
                    crate::deprintln!("  [node] bulk RDMA active (RoCEv2 8x4 MiB ring + parallel HW SHA-256/write)");
                    Ok(Some(link))
                }
                Err(e) => {
                    crate::deprintln!("  [node] bulk RDMA unavailable ({e:#}) — falling back to TCP");
                    send_msg(s, &Msg::BulkReply { ok: false, ep: BulkEndpoint::default() })?;
                    Ok(None)
                }
            }
        }
        Msg::BulkOffer { enabled: false, .. } => {
            send_msg(s, &Msg::BulkReply { ok: false, ep: BulkEndpoint::default() })?;
            Ok(None)
        }
        Msg::Error { msg } => bail!("the head stopped the sync: {msg}"),
        other => bail!("expected BulkOffer, got {other:?}"),
    }
}

/// Receive one blob over RDMA into the exact same `blobs/tmp.<pid>.<hash>` staging path used by `recv_blob`.
fn recv_blob_rdma(bulk: &mut crate::net::bulk::BulkLink, hash: &str, size: u64, prog: &Progress) -> Result<PathBuf> {
    let tmp = cache_root().join("blobs").join(format!("tmp.{}.{}", std::process::id(), hash));
    bulk.recv_blob_to_file(&tmp, size, hash, |n| prog.add(n as u64))?;
    Ok(tmp)
}

/// Receive `size` bytes into a temp file, hashing as we go; verify against `hash`; return the temp path.
fn recv_blob(r: &mut impl Read, hash: &str, size: u64) -> Result<PathBuf> {
    let tmp = cache_root().join("blobs").join(format!("tmp.{}.{}", std::process::id(), hash));
    std::fs::create_dir_all(tmp.parent().unwrap())?;
    let mut f = std::fs::File::create(&tmp)?;
    let mut hasher = crate::net::bulk::FastSha256::new();
    let mut left = size;
    let mut buf = vec![0u8; 4 << 20];
    while left > 0 {
        let want = left.min(buf.len() as u64) as usize;
        r.read_exact(&mut buf[..want])?;
        hasher.update(&buf[..want]);
        f.write_all(&buf[..want])?;
        left -= want as u64;
    }
    f.flush()?;
    let got = hasher.finalize_hex();
    if got != hash {
        let _ = std::fs::remove_file(&tmp);
        bail!("blob hash mismatch: expected {hash}, got {got}");
    }
    Ok(tmp)
}

// ---------------------------------------------------------------------------------------------------
// Node: receive-and-assemble
// ---------------------------------------------------------------------------------------------------

/// Assemble a model dir under the cache: each logical name -> symlink to blobs/<hash>.
pub(super) fn assemble_model_dir(model_id: &str, artifacts: &[Artifact]) -> Result<PathBuf> {
    let dir = cache_root().join("models").join(model_id);
    // B32: an empty manifest (the legacy EXL3 path ships nothing) caches nothing — no empty placeholder dir
    // (`--cached-models-list` used to show it as "0.00 GiB 0 blob(s)").
    if artifacts.is_empty() { return Ok(dir); }
    std::fs::create_dir_all(&dir)?;
    for a in artifacts {
        let link = dir.join(&a.logical);
        if let Some(parent) = link.parent() { std::fs::create_dir_all(parent)?; }
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(blob_path(&a.hash), &link)
            .with_context(|| format!("symlink {}", a.logical))?;
    }
    // B32: the dir mirrors THIS manifest — a link left from an earlier sync of the same id (another plan
    // version, a re-quantized pack) is removed, so the loader and the cache ops see exactly what shipped.
    let want: std::collections::HashSet<PathBuf> = artifacts.iter().map(|a| dir.join(&a.logical)).collect();
    let mut stack = vec![dir.clone()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for ent in rd.flatten() {
            let p = ent.path();
            match ent.file_type() {
                Ok(ft) if ft.is_dir() => stack.push(p),
                Ok(ft) if ft.is_symlink() && !want.contains(&p) => { let _ = std::fs::remove_file(&p); }
                _ => {}
            }
        }
    }
    Ok(dir)
}

/// DSV4.1 (leg LS2): an Engram head file this box already holds (pre-staged in an Engram dir, manifest sha256 and
/// size equal to the artifact's) becomes the blob by a hard link (a symlink across drives), so the head ships nothing
/// for it. Only artifacts named `engram/engram_head<h>.bin` are looked at; every other family's sync is unchanged.
fn adopt_staged(artifacts: &[Artifact]) -> usize {
    let mut n = 0;
    for a in artifacts {
        if !a.logical.starts_with("engram/engram_head") || !a.logical.ends_with(".bin") || have_blob(&a.hash) { continue; }
        let Some(p) = crate::dsv41::load::engram::find_staged(&a.logical, &a.hash, a.size) else { continue };
        let dst = blob_path(&a.hash);
        if let Some(parent) = dst.parent() { let _ = std::fs::create_dir_all(parent); }
        let _ = std::fs::remove_file(&dst); // a dangling link from an earlier adoption
        if std::fs::hard_link(&p, &dst).or_else(|_| std::os::unix::fs::symlink(&p, &dst)).is_ok() { n += 1; }
    }
    n
}

/// Handle one head connection: Hello -> Manifest -> Missing -> blobs -> assemble -> Ready ->
/// Config -> DraftManifest -> (draft blobs) -> DraftReady. Returns the assembled model dir +
/// the assembled DRAFT cache dir (None when the head shipped no drafter) + the head's TP config
/// (so the node needs ZERO GB10_TP_* env, ZERO model paths, ZERO draft paths) + the RETAINED
/// control stream (the TP serving session runs over it; the bench path just drops it).
pub(super) fn node_handle(mut s: TcpStream) -> Result<(PathBuf, Option<PathBuf>, crate::tp::TpConfig, TcpStream)> {
    match recv_msg(&mut s)? {
        Msg::Hello { version, hostname, .. } => {
            if let Err(why) = version_match(&version, &binary_version()) {
                send_msg(&mut s, &Msg::Error { msg: why.clone() })?;
                bail!("head/node {why}");
            }
            crate::deprintln!("  [node] head '{hostname}' connected (binary {version})");
        }
        _ => bail!("expected Hello"),
    }
    send_msg(&mut s, &Msg::Hello { version: binary_version(), role: "node".into(), hostname: hostname() })?;

    let (model_id, artifacts, draft_offer) = match recv_msg(&mut s)? {
        Msg::Manifest { model_id, artifacts, draft } => (model_id, artifacts, draft),
        _ => bail!("expected Manifest"),
    };
    let adopted = adopt_staged(&artifacts);
    if adopted > 0 { crate::deprintln!("  [node] {adopted} Engram head file(s) already staged on this box (manifest sha256 and size match): adopted into the cache, not received"); }
    let missing: Vec<String> = artifacts.iter().map(|a| a.hash.clone())
        .filter(|h| !have_blob(h)).collect();
    let have = artifacts.len() - missing.len();
    crate::deprintln!("  [node] manifest '{model_id}': {} artifacts, {have} cached, {} to fetch",
              artifacts.len(), missing.len());
    let fetch_bytes: u64 = artifacts.iter().filter(|a| missing.contains(&a.hash)).map(|a| a.size).sum();
    // owner 2026-11-02: report what this session would store here and the drive's free space; the head checks EVERY
    // node before it ships to any (and stops the session, telling us why, when a drive is short)
    let draft_need: u64 = draft_offer.iter().filter(|a| !have_blob(&a.hash) && !missing.contains(&a.hash)).map(|a| a.size).sum();
    let root = cache_root();
    let space = free_bytes(&root).map(|free| SpaceReport { need: fetch_bytes + draft_need, free,
                                                           dir: root.display().to_string(), host: hostname() });
    let t_fetch = std::time::Instant::now();
    send_msg(&mut s, &Msg::Missing { hashes: missing.clone(), space })?;

    let mut bulk = if !missing.is_empty() { node_negotiate_bulk(&mut s)? } else { None };
    let prog = Progress::new("receiving the model from the head", fetch_bytes, Unit::Bytes);
    for _ in 0..missing.len() {
        match recv_msg(&mut s)? {
            Msg::BlobHeader { hash, size } => {
                let tmp = if let Some(b) = bulk.as_mut() {
                    recv_blob_rdma(b, &hash, size, &prog)?
                } else {
                    recv_blob(&mut Counted { inner: &mut s, prog: &prog }, &hash, size)?
                };
                publish_blob(&hash, &tmp)?;
                crate::deprintln!("  [node] cached {} ({:.1} MB)", &hash[..12], size as f64 / 1e6);
            }
            Msg::Error { msg } => bail!("the head stopped the sync: {msg}"),
            _ => bail!("expected BlobHeader"),
        }
    }
    drop(bulk);
    prog.finish();
    let dir = assemble_model_dir(&model_id, &artifacts)?;
    send_msg(&mut s, &Msg::Ready { model_dir: dir.to_string_lossy().to_string() })?;
    let cfg = match recv_msg(&mut s)? {
        Msg::Config(c) => c,
        Msg::Error { msg } => bail!("the head stopped the sync: {msg}"),
        _ => bail!("expected Config"),
    };
    crate::deprintln!("  [node] config from head: shard_mixers={} graph={} fp32_partials={} mtp={} depth={:?} mode_serve={}",
              cfg.shard_mixers, cfg.graph, cfg.fp32_partials, cfg.mtp, cfg.mtp_depth, cfg.mode_serve);


    // v5 — the DFlash2 draft round (always exactly one DraftManifest after Config). The bytes
    // land in the SAME content-addressed blob store as the model shards; the assembled dir lives
    // under models/<draft-id>/ exactly like a trunk model. A node therefore NEVER needs a local
    // draft copy — the caller rewrites the config's df2_draft_dir to the returned cache path.
    let draft_dir: Option<PathBuf> = match recv_msg(&mut s)? {
        Msg::DraftManifest { model_id, artifacts } if artifacts.is_empty() => {
            send_msg(&mut s, &Msg::DraftReady { model_dir: String::new() })?;
            crate::deprintln!("  [node] no draft artifact this session (head sent an empty manifest)");
            None
        }
        Msg::DraftManifest { model_id, artifacts } => {
            let missing: Vec<String> = artifacts.iter().map(|a| a.hash.clone())
                .filter(|h| !have_blob(h)).collect();
            let have = artifacts.len() - missing.len();
            crate::deprintln!("  [node] draft manifest '{model_id}': {} artifacts, {have} cached, {} to fetch",
                      artifacts.len(), missing.len());
            send_msg(&mut s, &Msg::Missing { hashes: missing.clone(), space: None })?;
            let mut dbulk = if !missing.is_empty() { node_negotiate_bulk(&mut s)? } else { None };
            let dbytes: u64 = artifacts.iter().filter(|a| missing.contains(&a.hash)).map(|a| a.size).sum();
            let dprog = Progress::new("receiving the drafter from the head", dbytes, Unit::Bytes);
            for _ in 0..missing.len() {
                match recv_msg(&mut s)? {
                    Msg::BlobHeader { hash, size } => {
                        let tmp = if let Some(b) = dbulk.as_mut() {
                            recv_blob_rdma(b, &hash, size, &dprog)?
                        } else {
                            recv_blob(&mut Counted { inner: &mut s, prog: &dprog }, &hash, size)?
                        };
                        publish_blob(&hash, &tmp)?;
                        crate::deprintln!("  [node] cached draft blob {} ({:.1} MB)", &hash[..12], size as f64 / 1e6);
                    }
                    _ => bail!("expected draft BlobHeader"),
                }
            }
            drop(dbulk);
            dprog.finish();
            let ddir = assemble_model_dir(&model_id, &artifacts)?;
            send_msg(&mut s, &Msg::DraftReady { model_dir: ddir.to_string_lossy().to_string() })?;
            crate::deprintln!("  [node] DRAFT READY — drafter assembled at {} (loads from the cache, no local copy)",
                      ddir.display());
            Some(ddir)
        }
        _ => bail!("expected DraftManifest after Config"),
    };
    if cfg.sync_only {
        // B32 `--tp-sync-only`: report and stop here (the head stops after its sync too; the supervisor re-arms)
        let total: u64 = artifacts.iter().map(|a| a.size).sum();
        let mut uniq: Vec<&str> = artifacts.iter().map(|a| a.hash.as_str()).collect();
        uniq.sort(); uniq.dedup();
        eprintln!("SYNC_ONLY node '{}': model '{model_id}' {} artifacts ({} blobs) = {:.2} GiB; fetched {:.2} GiB in {:.1} s \
                   ({} already cached); assembled at {}",
                  hostname(), artifacts.len(), uniq.len(), total as f64 / (1u64 << 30) as f64,
                  fetch_bytes as f64 / (1u64 << 30) as f64, t_fetch.elapsed().as_secs_f64(), have, dir.display());
        std::process::exit(0);
    }
    Ok((dir, draft_dir, cfg, s))
}

/// Run as a node: answer discovery, accept ONE head sync, return the assembled model dir + the head's
/// IP (its RoCE address, used to bring up the RDMA data-plane link back to it) + the head's TP config
/// + the retained control stream (dropped by bench sessions, kept by serving ones).
pub fn run_node(tcp_port: u16, udp_port: u16)
    -> Result<(PathBuf, Option<PathBuf>, IpAddr, crate::tp::TpConfig, TcpStream)> {
    let _ = crate::tp::rdma_devs();   // resolve + print the RDMA rails now (an ambiguous box exits here, not mid-boot)
    spawn_discovery_responder(tcp_port, udp_port)?;
    let listener = TcpListener::bind((Ipv4Addr::UNSPECIFIED, tcp_port))
        .with_context(|| format!("bind TCP {tcp_port}"))?;
    eprintln!("[node] {} ready: discovery on UDP {udp_port}, control on TCP {tcp_port}, cache {}",
              hostname(), cache_root().display());
    // TP-4D (C4/C9): hardened stream + stray connections dropped (see `accept_head`)
    let (s, from) = accept_head(&listener, NODE_FIRST_FRAME_WAIT)?;
    crate::deprintln!("[node] head connected from {from}");
    let (dir, draft_dir, cfg, s) = node_handle(s)?;
    eprintln!("[node] SYNCED — model ready at {}{}",
              dir.display(),
              draft_dir.as_ref().map(|d| format!(", drafter at {}", d.display())).unwrap_or_default());
    Ok((dir, draft_dir, from.ip(), cfg, s))
}
