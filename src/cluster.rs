//! `--head` / `--node` cluster orchestration for 2-node TP=2 (Stage 3).
//!
//! Goal (user): launch one binary per box, zero manual config or model copy. The **head** owns the
//! model; it auto-discovers **node**s, checks binary compatibility, and ships whatever artifacts the
//! node is missing. A **content-addressed cache** means a node that already has the files (or a re-run)
//! transfers nothing.
//!
//! Control plane = normal network (UDP discovery + TCP sync). RDMA (`net.rs`) is reserved for the
//! inference data plane only — bootstrap never depends on verbs, so recovery stays simple.
//!
//! MVP scope: whole-model distribution (per-rank shard distribution is a later optimization once the
//! G-D weight sharding exists). After sync the node has an assembled model dir ready for the TP run.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket, SocketAddr, IpAddr, Ipv4Addr};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const PROTOCOL_VERSION: u32 = 3;
const DISCOVERY_PORT: u16 = 29499;          // UDP; TCP control plane defaults to 29500 (--port)
const DISCOVERY_MAGIC: &str = "GB10-TP-DISCOVER";
/// Binary-compat token: same compiled kernels + same Rust-side sources + protocol => same wire
/// behavior. Cheaper than hashing the 15 MB executable each launch, and it is exactly what must
/// match across boxes. `-k` covers the kernels (KERNEL_BUILD_ID), `-r` the Rust sources + C shim
/// (SOURCE_BUILD_ID): the sharders/protocol/scheduler change behavior without touching a .cu.
fn binary_version() -> String {
    format!("v{}-k{}-r{}", PROTOCOL_VERSION, env!("KERNEL_BUILD_ID"), env!("SOURCE_BUILD_ID"))
}

// ---------------------------------------------------------------------------------------------------
// Wire protocol
// ---------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Artifact {
    pub logical: String,   // path relative to the model dir, e.g. "config.json"
    pub hash: String,      // sha256 hex
    pub size: u64,
}

#[derive(Serialize, Deserialize, Debug)]
enum Msg {
    Hello { version: String, role: String, hostname: String },
    Manifest { model_id: String, artifacts: Vec<Artifact> },
    Missing { hashes: Vec<String> },
    BlobHeader { hash: String, size: u64 },
    Ready { model_dir: String },
    Config(crate::tp::TpConfig),
    Error { msg: String },
}

/// Length-prefixed JSON framing, shared by the sync protocol (`Msg`) and, once a session is
/// retained, by the TP serving control plane (`tp_serve::ServingMsg`).
pub(crate) fn send_json<T: Serialize>(w: &mut impl Write, m: &T) -> Result<()> {
    let b = serde_json::to_vec(m)?;
    w.write_all(&(b.len() as u32).to_be_bytes())?;
    w.write_all(&b)?;
    w.flush()?;
    Ok(())
}
pub(crate) fn recv_json<T: serde::de::DeserializeOwned>(r: &mut impl Read) -> Result<T> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len)?;
    let n = u32::from_be_bytes(len) as usize;
    if n > 64 * 1024 * 1024 { bail!("control message too large ({n} B)"); }
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    Ok(serde_json::from_slice(&b)?)
}

fn send_msg(w: &mut impl Write, m: &Msg) -> Result<()> { send_json(w, m) }
fn recv_msg(r: &mut impl Read) -> Result<Msg> { recv_json(r) }

// ---------------------------------------------------------------------------------------------------
// Content-addressed cache
// ---------------------------------------------------------------------------------------------------

fn cache_root() -> PathBuf {
    std::env::var("GB10_TP_CACHE").map(PathBuf::from).unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(home).join(".cache/gb10_tp")
    })
}
fn blob_path(hash: &str) -> PathBuf { cache_root().join("blobs").join(hash) }
fn have_blob(hash: &str) -> bool { blob_path(hash).exists() }

/// Atomically publish `tmp` (already hash-verified) into the content store as `blobs/<hash>`.
fn publish_blob(hash: &str, tmp: &Path) -> Result<()> {
    let dst = blob_path(hash);
    std::fs::create_dir_all(dst.parent().unwrap())?;
    std::fs::rename(tmp, &dst).with_context(|| format!("publish blob {hash}"))?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<(String, u64)> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        h.update(&buf[..n]);
        total += n as u64;
    }
    Ok((hex(&h.finalize()), total))
}
fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{x:02x}")).collect() }

/// Per-file (path, mtime, size) -> hash cache so re-launches don't re-hash a 15 GB model.
fn hash_cache_load() -> HashMap<String, String> {
    let p = cache_root().join("hashcache.json");
    std::fs::read(&p).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}
fn hash_cache_save(m: &HashMap<String, String>) {
    let p = cache_root().join("hashcache.json");
    let _ = std::fs::create_dir_all(p.parent().unwrap());
    if let Ok(b) = serde_json::to_vec(m) { let _ = std::fs::write(&p, b); }
}
fn cached_key(path: &Path) -> Result<String> {
    let md = std::fs::metadata(path)?;
    let mtime = md.modified()?.duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    Ok(format!("{}|{}|{}", path.display(), mtime, md.len()))
}

// ---------------------------------------------------------------------------------------------------
// Manifest (head side)
// ---------------------------------------------------------------------------------------------------

/// Files a node needs to serve a model. Follows a symlinked model dir to the real files.
fn model_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
        let e = entry?;
        let p = e.path();
        let meta = std::fs::metadata(&p)?;   // follows symlinks
        if meta.is_file() {
            // skip editor/OS cruft and our own sidecars
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') || name.ends_with(".tmp") { continue; }
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}

pub fn build_manifest(model_dir: &Path) -> Result<(String, Vec<Artifact>)> {
    let mut cache = hash_cache_load();
    let mut dirty = false;
    let mut artifacts = Vec::new();
    for path in model_files(model_dir)? {
        let key = cached_key(&path)?;
        let (hash, size) = if let Some(h) = cache.get(&key) {
            (h.clone(), std::fs::metadata(&path)?.len())
        } else {
            let (h, sz) = sha256_file(&path)?;
            cache.insert(key, h.clone());
            dirty = true;
            (h, sz)
        };
        let logical = path.strip_prefix(model_dir).unwrap_or(&path).to_string_lossy().to_string();
        artifacts.push(Artifact { logical, hash, size });
    }
    if dirty { hash_cache_save(&cache); }
    let model_id = model_dir.file_name().map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".into());
    Ok((model_id, artifacts))
}

// ---------------------------------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct NodeInfo { pub hostname: String, pub addr: SocketAddr }

#[derive(Serialize, Deserialize)]
struct DiscoverProbe { magic: String, version: String }
#[derive(Serialize, Deserialize)]
struct DiscoverReply { hostname: String, tcp_port: u16, version: String }

/// Node side: answer discovery probes with our on-path IP + TCP port. Runs until the process exits.
pub fn spawn_discovery_responder(tcp_port: u16) -> Result<()> {
    let sock = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, DISCOVERY_PORT))
        .context("bind UDP discovery port")?;
    let hostname = hostname();
    std::thread::spawn(move || {
        let mut buf = [0u8; 2048];
        loop {
            let (n, from) = match sock.recv_from(&mut buf) { Ok(x) => x, Err(_) => continue };
            let probe: DiscoverProbe = match serde_json::from_slice(&buf[..n]) { Ok(p) => p, Err(_) => continue };
            if probe.magic != DISCOVERY_MAGIC { continue; }
            // Reply via the same socket: the OS picks the source IP by the route back to the head, so
            // the head sees our on-path (RoCE) IP as the datagram source — exactly the TCP address to use.
            let reply = DiscoverReply { hostname: hostname.clone(), tcp_port, version: binary_version() };
            if let Ok(b) = serde_json::to_vec(&reply) { let _ = sock.send_to(&b, from); }
        }
    });
    Ok(())
}

/// Head side: broadcast a probe on the RoCE subnets (+ global broadcast) and collect responders.
pub fn discover_nodes(wait: Duration) -> Result<Vec<NodeInfo>> {
    let sock = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))?;
    sock.set_broadcast(true)?;
    sock.set_read_timeout(Some(Duration::from_millis(200)))?;
    let probe = serde_json::to_vec(&DiscoverProbe {
        magic: DISCOVERY_MAGIC.into(), version: binary_version() })?;
    // GB10 boxes always expose the same RoCE interface NAMES; resolve their live IP/broadcast at
    // runtime (IPs are config, names/types are fixed) and broadcast there + a global catch-all.
    let ifaces = roce_interfaces();
    for f in &ifaces {
        eprintln!("  [discover] RoCE rail {} {} ({}) ip {} bcast {}", f.rail, f.ib, f.netdev, f.ip, f.bcast);
    }
    if ifaces.is_empty() {
        eprintln!("  [discover] no RoCE interface resolved — broadcasting globally only");
    }
    let mut targets: Vec<Ipv4Addr> = ifaces.iter().map(|f| f.bcast).collect();
    targets.push(Ipv4Addr::BROADCAST);
    for t in &targets { let _ = sock.send_to(&probe, (*t, DISCOVERY_PORT)); }
    let deadline = std::time::Instant::now() + wait;
    // A node replies once per subnet the probe reached it on (RoCE rail 1/2 + mgmt), all same hostname.
    // Keep the RoCE-preferred source IP: that is exactly the address the RDMA data plane must use.
    let mut nodes: HashMap<String, (u8, NodeInfo)> = HashMap::new();
    let mut buf = [0u8; 2048];
    while std::time::Instant::now() < deadline {
        match sock.recv_from(&mut buf) {
            Ok((n, from)) => {
                if let Ok(r) = serde_json::from_slice::<DiscoverReply>(&buf[..n]) {
                    if r.version != binary_version() {
                        eprintln!("  [discover] {} at {} has MISMATCHED binary ({} vs {}) — skipping",
                                  r.hostname, from.ip(), r.version, binary_version());
                        continue;
                    }
                    let rank = ip_rank(from.ip(), &ifaces);
                    let ni = NodeInfo { hostname: r.hostname.clone(), addr: SocketAddr::new(from.ip(), r.tcp_port) };
                    match nodes.get(&r.hostname) {
                        Some((existing, _)) if *existing >= rank => {}
                        _ => { nodes.insert(r.hostname, (rank, ni)); }
                    }
                }
            }
            Err(_) => {}   // read timeout; keep polling until the deadline
        }
    }
    Ok(nodes.into_values().map(|(_, ni)| ni).collect())
}

/// A live ConnectX-7 RoCE interface, resolved from its (fixed) IB device name to its current IPv4.
struct Roce { ib: String, netdev: String, ip: Ipv4Addr, bcast: Ipv4Addr, mask: u32, rail: u8 }

/// Resolve the GB10 RoCE rails by their fixed IB device names → netdev (via /sys) → IPv4 (via `ip`).
/// Names/types are constant across GB10 boxes (per the hardware); only the IPs are configuration.
fn roce_interfaces() -> Vec<Roce> {
    // Default = the fixed GB10 rail names (identical across DGX Spark + every OEM clone: same SoC,
    // hard-wired PCIe topology, systemd predictable naming). Manual fallback for any platform that
    // breaks that: GB10_RDMA_DEV=dev1[,dev2] (rail order), set via --rdma-dev.
    let devs: Vec<(String, u8)> = match std::env::var("GB10_RDMA_DEV") {
        Ok(s) if !s.trim().is_empty() =>
            s.split(',').enumerate().map(|(i, d)| (d.trim().to_string(), (i + 1) as u8)).collect(),
        _ => vec![("rocep1s0f1".into(), 1), ("roceP2p1s0f1".into(), 2)],
    };
    let mut out = Vec::new();
    for (ib, rail) in &devs {
        let netdir = format!("/sys/class/infiniband/{ib}/device/net");
        let netdev = std::fs::read_dir(&netdir).ok()
            .and_then(|mut d| d.next()).and_then(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string());
        let Some(netdev) = netdev else {
            eprintln!("  [discover] RoCE device '{ib}' not found — override with --rdma-dev / GB10_RDMA_DEV, \
                       or use --nodes <ip> to skip discovery");
            continue;
        };
        if let Some((ip, prefix, bcast)) = ipv4_of(&netdev) {
            let mask = if prefix == 0 { 0 } else { u32::MAX << (32 - prefix) };
            out.push(Roce { ib: ib.clone(), netdev, ip, bcast, mask, rail: *rail });
        }
    }
    out
}

/// Parse `ip -o -4 addr show dev <netdev>` → (addr, prefix_len, broadcast).
fn ipv4_of(netdev: &str) -> Option<(Ipv4Addr, u8, Ipv4Addr)> {
    let out = std::process::Command::new("ip")
        .args(["-o", "-4", "addr", "show", "dev", netdev]).output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let toks: Vec<&str> = s.split_whitespace().collect();
    let (mut ip, mut prefix, mut brd) = (None, None, None);
    let mut i = 0;
    while i + 1 < toks.len() {
        match toks[i] {
            "inet" => { let mut it = toks[i + 1].split('/');
                ip = it.next().and_then(|x| x.parse().ok());
                prefix = it.next().and_then(|x| x.parse().ok()); }
            "brd" => brd = toks[i + 1].parse().ok(),
            _ => {}
        }
        i += 1;
    }
    Some((ip?, prefix?, brd?))
}

/// Rank a reply's source IP: on RoCE rail 1 (3) > rail 2 (2) > any other link (1).
fn ip_rank(ip: std::net::IpAddr, ifaces: &[Roce]) -> u8 {
    if let std::net::IpAddr::V4(v4) = ip {
        let x = u32::from(v4);
        for f in ifaces {
            if x & f.mask == u32::from(f.ip) & f.mask {
                return if f.rail == 1 { 3 } else { 2 };
            }
        }
    }
    1
}

fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname").map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "node".into())
}

// ---------------------------------------------------------------------------------------------------
// Blob streaming
// ---------------------------------------------------------------------------------------------------

fn send_blob(w: &mut impl Write, path: &Path) -> Result<()> {
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        w.write_all(&buf[..n])?;
    }
    w.flush()?;
    Ok(())
}

/// Receive `size` bytes into a temp file, hashing as we go; verify against `hash`; return the temp path.
fn recv_blob(r: &mut impl Read, hash: &str, size: u64) -> Result<PathBuf> {
    let tmp = cache_root().join("blobs").join(format!("tmp.{}.{}", std::process::id(), hash));
    std::fs::create_dir_all(tmp.parent().unwrap())?;
    let mut f = std::fs::File::create(&tmp)?;
    let mut hasher = Sha256::new();
    let mut left = size;
    let mut buf = vec![0u8; 1 << 20];
    while left > 0 {
        let want = left.min(buf.len() as u64) as usize;
        r.read_exact(&mut buf[..want])?;
        hasher.update(&buf[..want]);
        f.write_all(&buf[..want])?;
        left -= want as u64;
    }
    f.flush()?;
    let got = hex(&hasher.finalize());
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
fn assemble_model_dir(model_id: &str, artifacts: &[Artifact]) -> Result<PathBuf> {
    let dir = cache_root().join("models").join(model_id);
    std::fs::create_dir_all(&dir)?;
    for a in artifacts {
        let link = dir.join(&a.logical);
        if let Some(parent) = link.parent() { std::fs::create_dir_all(parent)?; }
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(blob_path(&a.hash), &link)
            .with_context(|| format!("symlink {}", a.logical))?;
    }
    Ok(dir)
}

/// Handle one head connection: Hello -> Manifest -> Missing -> blobs -> assemble -> Ready -> Config.
/// Returns the assembled model dir + the head's TP config (so the node needs ZERO GB10_TP_* env) +
/// the RETAINED control stream (the TP serving session runs over it; the bench path just drops it).
fn node_handle(mut s: TcpStream) -> Result<(PathBuf, crate::tp::TpConfig, TcpStream)> {
    match recv_msg(&mut s)? {
        Msg::Hello { version, hostname, .. } => {
            if version != binary_version() {
                send_msg(&mut s, &Msg::Error { msg: format!("binary mismatch: head {version} vs node {}", binary_version()) })?;
                bail!("head/node binary mismatch");
            }
            eprintln!("  [node] head '{hostname}' connected (binary {version})");
        }
        _ => bail!("expected Hello"),
    }
    send_msg(&mut s, &Msg::Hello { version: binary_version(), role: "node".into(), hostname: hostname() })?;

    let (model_id, artifacts) = match recv_msg(&mut s)? {
        Msg::Manifest { model_id, artifacts } => (model_id, artifacts),
        _ => bail!("expected Manifest"),
    };
    let missing: Vec<String> = artifacts.iter().map(|a| a.hash.clone())
        .filter(|h| !have_blob(h)).collect();
    let have = artifacts.len() - missing.len();
    eprintln!("  [node] manifest '{model_id}': {} artifacts, {have} cached, {} to fetch",
              artifacts.len(), missing.len());
    send_msg(&mut s, &Msg::Missing { hashes: missing.clone() })?;

    for _ in 0..missing.len() {
        match recv_msg(&mut s)? {
            Msg::BlobHeader { hash, size } => {
                let tmp = recv_blob(&mut s, &hash, size)?;
                publish_blob(&hash, &tmp)?;
                eprintln!("  [node] cached {} ({:.1} MB)", &hash[..12], size as f64 / 1e6);
            }
            _ => bail!("expected BlobHeader"),
        }
    }
    let dir = assemble_model_dir(&model_id, &artifacts)?;
    send_msg(&mut s, &Msg::Ready { model_dir: dir.to_string_lossy().to_string() })?;
    let cfg = match recv_msg(&mut s)? {
        Msg::Config(c) => c,
        _ => bail!("expected Config"),
    };
    eprintln!("  [node] config from head: shard_mixers={} graph={} fp32_partials={} mtp={} depth={:?} mode_serve={}",
              cfg.shard_mixers, cfg.graph, cfg.fp32_partials, cfg.mtp, cfg.mtp_depth, cfg.mode_serve);
    Ok((dir, cfg, s))
}

/// Run as a node: answer discovery, accept ONE head sync, return the assembled model dir + the head's
/// IP (its RoCE address, used to bring up the RDMA data-plane link back to it) + the head's TP config
/// + the retained control stream (dropped by bench sessions, kept by serving ones).
pub fn run_node(tcp_port: u16) -> Result<(PathBuf, IpAddr, crate::tp::TpConfig, TcpStream)> {
    spawn_discovery_responder(tcp_port)?;
    let listener = TcpListener::bind((Ipv4Addr::UNSPECIFIED, tcp_port))
        .with_context(|| format!("bind TCP {tcp_port}"))?;
    eprintln!("[node] {} ready: discovery on UDP {DISCOVERY_PORT}, control on TCP {tcp_port}, cache {}",
              hostname(), cache_root().display());
    let (s, from) = listener.accept()?;
    eprintln!("[node] head connected from {from}");
    s.set_nodelay(true).ok();
    let (dir, cfg, s) = node_handle(s)?;
    eprintln!("[node] SYNCED — model ready at {}", dir.display());
    Ok((dir, from.ip(), cfg, s))
}

// ---------------------------------------------------------------------------------------------------
// Head: discover-and-push
// ---------------------------------------------------------------------------------------------------

/// Push the model to one node: Hello -> Manifest -> receive Missing -> stream blobs -> Ready, then
/// ship our TP config (`Msg::Config`) so the node runs with ZERO GB10_TP_* env vars. Returns the
/// RETAINED control stream — the TP serving session keeps it as its control plane; the bench path
/// (`run_head`) drops it, ending the session exactly as before.
fn head_sync_one(node: &NodeInfo, model_dir: &Path, model_id: &str, artifacts: &[Artifact],
                 cfg: &crate::tp::TpConfig) -> Result<TcpStream> {
    let mut s = TcpStream::connect_timeout(&node.addr, Duration::from_secs(10))
        .with_context(|| format!("connect {}", node.addr))?;
    s.set_nodelay(true).ok();
    send_msg(&mut s, &Msg::Hello { version: binary_version(), role: "head".into(), hostname: hostname() })?;
    match recv_msg(&mut s)? {
        Msg::Hello { .. } => {}
        Msg::Error { msg } => bail!("node rejected: {msg}"),
        _ => bail!("expected node Hello"),
    }
    send_msg(&mut s, &Msg::Manifest { model_id: model_id.into(), artifacts: artifacts.to_vec() })?;
    let missing = match recv_msg(&mut s)? {
        Msg::Missing { hashes } => hashes,
        _ => bail!("expected Missing"),
    };
    let by_hash: HashMap<&str, &Artifact> = artifacts.iter().map(|a| (a.hash.as_str(), a)).collect();
    let total: u64 = missing.iter().filter_map(|h| by_hash.get(h.as_str())).map(|a| a.size).sum();
    eprintln!("[head] {} needs {} / {} artifacts ({:.2} GB)", node.hostname, missing.len(),
              artifacts.len(), total as f64 / 1e9);
    let t0 = std::time::Instant::now();
    for h in &missing {
        let a = by_hash.get(h.as_str()).context("node asked for unknown hash")?;
        send_msg(&mut s, &Msg::BlobHeader { hash: a.hash.clone(), size: a.size })?;
        send_blob(&mut s, &model_dir.join(&a.logical))?;
    }
    match recv_msg(&mut s)? {
        Msg::Ready { model_dir } => {
            let secs = t0.elapsed().as_secs_f64();
            eprintln!("[head] {} READY — model at {} ({:.2} GB in {:.1}s = {:.2} GB/s)",
                      node.hostname, model_dir, total as f64/1e9, secs,
                      if secs > 0.0 { total as f64/1e9/secs } else { 0.0 });
            send_msg(&mut s, &Msg::Config(cfg.clone()))?;
            eprintln!("[head] shipped config to {}: shard_mixers={} graph={} fp32_partials={} mtp={} depth={:?}",
                      node.hostname, cfg.shard_mixers, cfg.graph, cfg.fp32_partials, cfg.mtp, cfg.mtp_depth);
            Ok(s)
        }
        Msg::Error { msg } => bail!("node error: {msg}"),
        _ => bail!("expected Ready"),
    }
}

/// Run as the head: discover nodes (or use explicit addrs), then sync the model to each.
pub fn run_head(model_dir: &Path, explicit: Option<Vec<SocketAddr>>, discover_wait: Duration,
                cfg: &crate::tp::TpConfig)
    -> Result<Vec<NodeInfo>>
{
    eprintln!("[head] {} — building manifest for {} ...", hostname(), model_dir.display());
    let (model_id, artifacts) = build_manifest(model_dir)?;
    let total: u64 = artifacts.iter().map(|a| a.size).sum();
    eprintln!("[head] manifest '{model_id}': {} artifacts, {:.2} GB", artifacts.len(), total as f64/1e9);

    let nodes: Vec<NodeInfo> = match explicit {
        Some(addrs) => addrs.into_iter()
            .map(|a| NodeInfo { hostname: a.ip().to_string(), addr: a }).collect(),
        None => {
            eprintln!("[head] discovering nodes (UDP broadcast, {}s) ...", discover_wait.as_secs());
            let n = discover_nodes(discover_wait)?;
            if n.is_empty() { bail!("no nodes discovered — is a --node running? (or pass --nodes <ip:port>)"); }
            for x in &n { eprintln!("  [head] found node '{}' at {}", x.hostname, x.addr); }
            n
        }
    };
    for node in &nodes { head_sync_one(node, model_dir, &model_id, &artifacts, cfg)?; }
    eprintln!("[head] all {} node(s) synced.", nodes.len());
    Ok(nodes)
}

/// Run as the head for a TP SERVING session (TP item A): identical to `run_head` (same manifest,
/// discovery, and blob push), but TP=2 means EXACTLY one node, and the sync connection is RETAINED
/// and returned — the whole serving control plane (calibration table, per-step events, shutdown)
/// then runs over it as `tp_serve::ServingMsg`. The node side is `run_node`'s retained stream.
pub fn run_head_session(model_dir: &Path, explicit: Option<Vec<SocketAddr>>, discover_wait: Duration,
                        cfg: &crate::tp::TpConfig)
    -> Result<(Vec<NodeInfo>, TcpStream)>
{
    eprintln!("[head] {} — building manifest for {} ...", hostname(), model_dir.display());
    let (model_id, artifacts) = build_manifest(model_dir)?;
    let total: u64 = artifacts.iter().map(|a| a.size).sum();
    eprintln!("[head] manifest '{model_id}': {} artifacts, {:.2} GB", artifacts.len(), total as f64/1e9);

    let nodes: Vec<NodeInfo> = match explicit {
        Some(addrs) => addrs.into_iter()
            .map(|a| NodeInfo { hostname: a.ip().to_string(), addr: a }).collect(),
        None => {
            eprintln!("[head] discovering nodes (UDP broadcast, {}s) ...", discover_wait.as_secs());
            let n = discover_nodes(discover_wait)?;
            if n.is_empty() { bail!("no nodes discovered — is a --node running? (or pass --nodes <ip:port>)"); }
            for x in &n { eprintln!("  [head] found node '{}' at {}", x.hostname, x.addr); }
            n
        }
    };
    if nodes.len() != 1 {
        bail!("TP serving is TP=2: exactly one node required, got {} — start exactly one --node \
               (or pass exactly one --nodes <ip:port>)", nodes.len());
    }
    let stream = head_sync_one(&nodes[0], model_dir, &model_id, &artifacts, cfg)?;
    eprintln!("[head] node synced; control stream RETAINED for the serving session");
    Ok((nodes, stream))
}

// ---------------------------------------------------------------------------------------------------
// Blob cache management (ops CLI: --list-model-blobs / --remove-model-blob / --clear-model-blobs)
// ---------------------------------------------------------------------------------------------------

/// Walk every symlink under `cache/models/<model_id>/`, calling `f(model_id, link_path, target)`.
fn walk_model_links(mut f: impl FnMut(&str, &Path, &Path)) {
    let mdir = cache_root().join("models");
    let Ok(models) = std::fs::read_dir(&mdir) else { return };
    for e in models.flatten() {
        let mid = e.file_name().to_string_lossy().to_string();
        let mut stack = vec![e.path()];
        while let Some(d) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for ent in rd.flatten() {
                let p = ent.path();
                let Ok(ft) = ent.file_type() else { continue };
                if ft.is_dir() { stack.push(p); continue; }
                if ft.is_symlink() {
                    if let Ok(t) = std::fs::read_link(&p) { f(&mid, &p, &t); }
                }
            }
        }
    }
}

fn fmt_gib(b: u64) -> String { format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64) }

/// `--list-model-blobs`: every blob in the cache — its id (sha256 = the blob's file name), size,
/// and which assembled model dir(s) reference it (ORPHAN = none). `tmp.*` rows are interrupted
/// fetch partials, safe to reclaim via --clear-model-blobs.
pub fn list_model_blobs() -> Result<()> {
    let dir = cache_root().join("blobs");
    let mut refs: HashMap<String, Vec<String>> = HashMap::new();
    walk_model_links(|mid, _p, t| {
        if let Some(h) = t.file_name() { refs.entry(h.to_string_lossy().to_string()).or_default().push(mid.to_string()); }
    });
    for v in refs.values_mut() { v.sort(); v.dedup(); }

    let mut rows: Vec<(String, u64)> = Vec::new();
    let mut tmp: Vec<(String, u64)> = Vec::new();
    for e in std::fs::read_dir(&dir).with_context(|| format!("read_dir {}", dir.display()))?.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let size = e.metadata().map(|m| m.len()).unwrap_or(0);
        if name.starts_with("tmp.") { tmp.push((name, size)); } else { rows.push((name, size)); }
    }
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    let total: u64 = rows.iter().map(|r| r.1).sum();
    println!("cache {} — {} blobs, {}", dir.display(), rows.len(), fmt_gib(total));
    for (h, sz) in &rows {
        let users = refs.get(h).cloned().unwrap_or_default();
        let tag = if users.is_empty() { "ORPHAN".to_string() } else { users.join(",") };
        println!("{h}  {:>12}  {tag}", fmt_gib(*sz));
    }
    if !tmp.is_empty() {
        let t: u64 = tmp.iter().map(|x| x.1).sum();
        println!("-- {} interrupted-fetch partial(s), {} reclaimable (tmp.*)", tmp.len(), fmt_gib(t));
    }
    Ok(())
}

/// `--remove-model-blob <hash|unique-prefix>`: delete ONE blob + the assembled-dir symlinks that
/// reference it (so `models/` stays consistent). A model that loses files re-fetches the blob on
/// its next head sync — removal never corrupts a later run. If every match is a `tmp.*`
/// interrupted-fetch partial, ALL of them are deleted (they are junk by definition).
pub fn remove_model_blob(id: &str) -> Result<()> {
    if id.len() < 4 { bail!("refusing to match an id shorter than 4 characters"); }
    let dir = cache_root().join("blobs");
    let mut matches: Vec<String> = Vec::new();
    for e in std::fs::read_dir(&dir).with_context(|| format!("read_dir {}", dir.display()))?.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if name == id || name.starts_with(id) { matches.push(name); }
    }
    if matches.is_empty() { bail!("no blob matching '{id}' in {}", dir.display()); }
    if matches.iter().all(|m| m.starts_with("tmp.")) {
        let mut freed = 0u64;
        for m in &matches {
            freed += std::fs::metadata(dir.join(m)).map(|x| x.len()).unwrap_or(0);
            let _ = std::fs::remove_file(dir.join(m));
        }
        println!("removed {} interrupted-fetch partial(s), {}", matches.len(), fmt_gib(freed));
        return Ok(());
    }
    let hash = match matches.len() {
        1 => matches.pop().unwrap(),
        n => bail!("'{id}' matches {n} blobs — give a longer prefix"),
    };
    let size = std::fs::metadata(blob_path(&hash)).map(|m| m.len()).unwrap_or(0);
    std::fs::remove_file(blob_path(&hash)).with_context(|| format!("remove blob {hash}"))?;
    let mut pruned = 0usize;
    let mut affected: Vec<String> = Vec::new();
    walk_model_links(|mid, p, t| {
        if t.file_name().map(|h| h.to_string_lossy() == hash).unwrap_or(false) {
            if std::fs::remove_file(p).is_ok() { pruned += 1; affected.push(mid.to_string()); }
        }
    });
    affected.sort(); affected.dedup();
    println!("removed blob {hash} ({}) — pruned {pruned} link(s) in [{}]",
             fmt_gib(size), affected.join(","));
    println!("note: those models now have missing files; a head re-sync re-fetches this blob on next use.");
    Ok(())
}

/// `--clear-model-blobs`: delete ALL cached blobs (including interrupted tmp.* partials) and the
/// assembled model dirs (symlink trees). The next head run re-syncs from scratch.
pub fn clear_model_blobs() -> Result<()> {
    let dir = cache_root().join("blobs");
    let mut total = 0u64;
    let mut n = 0usize;
    for e in std::fs::read_dir(&dir).with_context(|| format!("read_dir {}", dir.display()))?.flatten() {
        total += e.metadata().map(|m| m.len()).unwrap_or(0);
        std::fs::remove_file(e.path()).with_context(|| format!("remove {}", e.path().display()))?;
        n += 1;
    }
    let mdir = cache_root().join("models");
    if mdir.exists() { std::fs::remove_dir_all(&mdir).context("remove assembled model dirs")?; }
    println!("cleared {n} blob(s), {} — assembled model dirs removed; next head run re-syncs from scratch",
             fmt_gib(total));
    Ok(())
}
