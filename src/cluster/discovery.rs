//! Discovery and topology: the UDP responder / prober, RoCE interface scan, rank assignment by link rail, and the
//! per-rank topology strings. See `mod.rs` for the map.

use super::*;

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
pub fn spawn_discovery_responder(tcp_port: u16, udp_port: u16) -> Result<()> {
    let sock = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, udp_port))
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
        crate::deprintln!("  [discover] RoCE rail {} {} ({}) ip {} bcast {}", f.rail, f.ib, f.netdev, f.ip, f.bcast);
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
pub(super) struct Roce { pub(super) ib: String, pub(super) netdev: String, pub(super) ip: Ipv4Addr, pub(super) bcast: Ipv4Addr, pub(super) mask: u32, pub(super) rail: u8 }

/// Resolve the GB10 RoCE rails by their fixed IB device names → netdev (via /sys) → IPv4 (via `ip`).
/// Names/types are constant across GB10 boxes (per the hardware); only the IPs are configuration.
pub(super) fn roce_interfaces() -> Vec<Roce> {
    // The rails in order: explicit --rdma-dev=dev1[,dev2], else auto-detected (tp::rdma_devs: the only
    // active RoCE port pair; the ASUS GX10s here cable f1, the DGX Sparks f0).
    let devs: Vec<(String, u8)> = crate::tp::rdma_devs().iter().enumerate()
        .map(|(i, d)| (d.clone(), (i + 1) as u8)).collect();
    let mut out = Vec::new();
    for (ib, rail) in &devs {
        let netdir = format!("/sys/class/infiniband/{ib}/device/net");
        let netdev = std::fs::read_dir(&netdir).ok()
            .and_then(|mut d| d.next()).and_then(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string());
        let Some(netdev) = netdev else {
            eprintln!("  [discover] RoCE device '{ib}' not found — override with --rdma-dev / --rdma-dev, \
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

/// Parse `ip -o -4 addr show dev <netdev>` output → (addr, prefix_len, broadcast).
pub(super) fn ipv4_of_str(s: &str) -> Option<(Ipv4Addr, u8, Ipv4Addr)> {
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
    // Ported from sf-stav/veloGB10 PR#5 (Morxi): `brd` is absent for point-to-point prefixes
    // (/31, /30, ...) on some kernels: the runtime only needs the address itself, while the
    // bcast/mask serve discovery broadcast + rail ranking. So do NOT hard-require `brd`;
    // synthesize it from ip | ~mask when the kernel omits it.
    let ip = ip?;
    let prefix = prefix?;
    let mask = if prefix == 0 { 0 } else { u32::MAX << (32 - prefix) };
    let bcast = brd.unwrap_or_else(|| Ipv4Addr::from(u32::from(ip) | !mask));
    Some((ip, prefix, bcast))
}

/// Run `ip -o -4 addr show dev <netdev>` and parse it (the pure parser above is unit-tested).
fn ipv4_of(netdev: &str) -> Option<(Ipv4Addr, u8, Ipv4Addr)> {
    let out = std::process::Command::new("ip")
        .args(["-o", "-4", "addr", "show", "dev", netdev]).output().ok()?;
    ipv4_of_str(&String::from_utf8_lossy(&out.stdout))
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

/// Deterministic total order over a node's `SocketAddr`. Higher RoCE-rail preference sorts FIRST,
/// then ascending IPv4, then ascending port. The IPv4 tie-break is what makes the order a *total*
/// order (hostnames may collide across boxes or be unset in explicit `--nodes` mode).
fn node_order_key(addr: &SocketAddr, ifaces: &[Roce]) -> (u8, u128, u16) {
    let rail = ip_rank(addr.ip(), ifaces);
    let ip_key = match addr.ip() {
        IpAddr::V4(v4) => u32::from(v4) as u128,
        IpAddr::V6(v6) => u128::from(v6),
    };
    // rail desc => invert; ip/port asc.
    (255u8.wrapping_sub(rail), ip_key, addr.port())
}

/// Assign node ranks 1..N-1 in a stable, reproducible order. The head is ALWAYS rank 0; the
/// discovered nodes (sorted by `node_order_key`) become ranks 1,2,... in that order. The sort is
/// decoupled from `discover_nodes` (which only picks the best source IP per hostname) so explicit
/// `--nodes` and discovery take the same path.
pub(super) fn assign_ranks(nodes: &mut Vec<NodeInfo>, ifaces: &[Roce]) {
    nodes.sort_by(|a, b| node_order_key(&a.addr, ifaces).cmp(&node_order_key(&b.addr, ifaces)));
}

/// Build the full rank→RoCE-IP topology (`Vec<String>` indexed by rank, size `world`). The head is
/// rank 0 at its own RoCE rail-1 IP (or the first resolved RoCE IP); `ranked_nodes` must already be
/// sorted by `assign_ranks` (ranks 1..N-1 in order). `topology[self_rank]` is the rank's own IP and
/// is unused by the N-way transport.
pub(super) fn build_topology(ranked_nodes: &[NodeInfo], ifaces: &[Roce], world: u32) -> Result<Vec<String>> {
    anyhow::ensure!(
        ranked_nodes.len() as u32 == world.saturating_sub(1),
        "build_topology: got {} nodes for world {world} (need {})",
        ranked_nodes.len(),
        world.saturating_sub(1)
    );
    let mut topo: Vec<String> = vec![String::new(); world as usize];
    // rank 0 = this process; use its own RoCE rail-1 IP (fall back to any resolved RoCE IP). This
    // entry is the address every node uses to dial the control QP, so it MUST be non-empty.
    let head_ip = ifaces.iter().find(|f| f.rail == 1).map(|f| f.ip)
        .or_else(|| ifaces.first().map(|f| f.ip))
        .context("no RoCE interface resolved — cannot build the rank->IP topology")?;
    topo[0] = head_ip.to_string();
    for (i, node) in ranked_nodes.iter().enumerate() {
        let rank = (i + 1) as usize;
        topo[rank] = node.addr.ip().to_string();
    }
    Ok(topo)
}

pub(super) fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname").map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "node".into())
}
