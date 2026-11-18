//! The control-channel exchange: `exchange_u32s` (world 2) and the world>2 hub (`HubXport`, `CtxHub`, `hub_all`, `exchange_u32s_all*`).
//! Startup / audit traffic only: it never touches the hot-path rings or the GPU epoch chain. See `mod.rs` for the map.

use super::*;

/// Ship a small u32 payload to the peer over the startup/audit channel (the `TpLink::exchange`
/// path, using the process-registered ctx — works while the RDMA proxy runs: the exchange's send
/// CQE is handed over via `xchg_send_done`). S-B9-REL-TP2RACE: at world 2 the frame stages and
/// lands in the DEDICATED control slots (net_shim.c ctrl region), disjoint from the hot-path
/// doorbell rings — before that fix it landed in recv slot `gen & 7` OF those rings, the #10 race.
/// Both ranks call it in the same SPMD order; rank 0 fills `mine` with its payload, rank 1 with
/// zeros, and BOTH read the peer's payload from the received slot. Returns the peer's words.
///
/// The wire frame's LAST 8 bytes are the generation tag and are clobbered (see net_exchange), so
/// `wire_u32s` must leave 8 bytes of headroom: usable payload = (wire_u32s*4 - 8) bytes, and only
/// `mine.len()` leading words are meaningful on the receive side.
pub fn exchange_u32s(mine: &[u32], wire_u32s: usize) -> anyhow::Result<Vec<u32>> {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { anyhow::bail!("exchange_u32s: no registered TP link (single-node?)"); }
    if mine.len() > wire_u32s { anyhow::bail!("exchange_u32s: payload {} > wire words {wire_u32s}", mine.len()); }
    let nbytes = wire_u32s * 4;
    if nbytes < 16 || nbytes > crate::tp::TP_SLOT_BYTES {
        anyhow::bail!("exchange_u32s: {nbytes} bytes outside the exchange envelope (16..={})", crate::tp::TP_SLOT_BYTES);
    }
    unsafe {
        let ctx = c as *mut NetCtx;
        if net_world(ctx) > 2 {
            anyhow::bail!("exchange_u32s: the pairwise channel of a world-{} link reaches rank 1 only; use exchange_u32s_all", net_world(ctx));
        }
        let send = std::slice::from_raw_parts_mut(net_send_hptr(ctx) as *mut u32, wire_u32s);
        for x in send.iter_mut() { *x = 0; }
        send[..mine.len()].copy_from_slice(mine);
        let rc = net_exchange(ctx, nbytes as c_int);
        if rc != 0 { anyhow::bail!("net_exchange failed rc={rc}"); }
        let recv = std::slice::from_raw_parts(net_recv_hptr(ctx) as *const u32, wire_u32s);
        Ok(recv[..mine.len()].to_vec())
    }
}

/// Place rank k's payload at word `k*wire` of the round-2 hub frame; a rank that failed to deliver stays zero.
pub(super) fn hub_pack(all: &[Option<Vec<u32>>], wire: usize, dst: &mut [u32]) {
    dst.iter_mut().for_each(|x| *x = 0);
    for (k, p) in all.iter().enumerate() {
        if let Some(p) = p { dst[k * wire..k * wire + p.len()].copy_from_slice(p); }
    }
}

pub(super) fn hub_unpack(src: &[u32], wire: usize, len: usize, world: usize) -> Vec<Vec<u32>> {
    (0..world).map(|k| src[k * wire..k * wire + len].to_vec()).collect()
}

/// The transport surface of the world>2 head hub: the dedicated control send staging slot, the stable
/// per-sender "last received" slots and the symmetric `net_exchange_one`. `hub_all` is written against this so
/// the REAL hub algorithm (round structure, failed-gather fan-out, pack/unpack, echo check) runs unchanged over
/// the in-memory transport of the CPU tests (`hub_mock`) and over the RDMA link (`CtxHub`).
pub(crate) trait HubXport {
    fn world(&self) -> usize;
    fn rank(&self) -> usize;
    /// The control SEND staging slot, `words` u32 long.
    fn send_buf(&mut self, words: usize) -> &mut [u32];
    /// A copy of the first `words` u32 of the stable slot holding the last frame received from `src`.
    fn recv_copy(&self, src: usize, words: usize) -> Vec<u32>;
    /// `net_exchange_one`: post the first `nbytes` of the staging slot to `peer` and wait for the peer's frame
    /// of the same size. 0 = ok; -2 = aborted; -3 = peer dead / link error; -4 = lockstep deadline expired.
    fn exchange_one(&mut self, peer: usize, nbytes: usize) -> i32;
}

/// The registered link as a `HubXport`. `deadline_ns` 0 = unbounded (boot paths); non-zero = the per-wait lockstep
/// deadline of `net_exchange_one_dl` (audit C5).
pub(crate) struct CtxHub { ctx: *mut NetCtx, world: usize, rank: usize, deadline_ns: u64 }

/// The registered link as a world>2 hub transport with a per-wait lockstep deadline (`deadline_ms` 0 =
/// unbounded). Err on a single-node run and at world <= 2 (the pairwise channel has no hub; the W=2 lockstep
/// stays on `exchange_u32s` + `agree_ext`).
pub(crate) fn link_hub(deadline_ms: u64) -> anyhow::Result<CtxHub> {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { anyhow::bail!("link_hub: no registered TP link (single-node?)"); }
    let ctx = c as *mut NetCtx;
    let (world, rank) = unsafe { (net_world(ctx) as usize, net_rank(ctx) as usize) };
    if world <= 2 { anyhow::bail!("link_hub: world {world} uses the pairwise channel, not the hub"); }
    Ok(CtxHub { ctx, world, rank, deadline_ns: deadline_ms.saturating_mul(1_000_000) })
}

impl HubXport for CtxHub {
    fn world(&self) -> usize { self.world }
    fn rank(&self) -> usize { self.rank }
    fn send_buf(&mut self, words: usize) -> &mut [u32] {
        unsafe { std::slice::from_raw_parts_mut(net_ctrl_send_hptr(self.ctx) as *mut u32, words) }
    }
    fn recv_copy(&self, src: usize, words: usize) -> Vec<u32> {
        unsafe { std::slice::from_raw_parts(net_ctrl_recv_hptr(self.ctx, src as c_int) as *const u32, words) }.to_vec()
    }
    fn exchange_one(&mut self, peer: usize, nbytes: usize) -> i32 {
        unsafe { net_exchange_one_dl(self.ctx, peer as c_int, nbytes as c_int, self.deadline_ns) }
    }
}

/// The world>2 hub exchange over any `HubXport` (see `exchange_u32s_all` for the contract). `mine` needs 8 bytes
/// of headroom inside its `wire` frame (the tail tag overwrites the last 8).
pub(crate) fn hub_all<X: HubXport>(x: &mut X, mine: &[u32], wire: usize) -> anyhow::Result<Vec<Vec<u32>>> {
    const TAIL_BYTES: usize = 8;
    let (world, rank) = (x.world(), x.rank());
    let (n1, n2) = (wire * 4, world * wire * 4);
    if n1 < 16 || mine.len() * 4 + TAIL_BYTES > n1 || n2 > crate::tp::TP_SLOT_BYTES {
        anyhow::bail!("exchange_u32s_all: {} words in a {wire}-word frame at world {world} is outside the hub envelope \
                       (frame >= 16 B with 8 B tail headroom, world * frame <= {} B)", mine.len(), crate::tp::TP_SLOT_BYTES);
    }
    {
        let send = x.send_buf(n2 / 4);
        send.iter_mut().for_each(|w| *w = 0);
        send[..mine.len()].copy_from_slice(mine);
    }
    if rank != 0 {
        let rc = x.exchange_one(0, n1);
        if rc != 0 { anyhow::bail!("exchange_u32s_all: rank {rank} round 1 rc={rc}{}", hub_rc_note(rc)); }
        x.send_buf(n2 / 4).iter_mut().for_each(|w| *w = 0);
        let rc = x.exchange_one(0, n2);
        if rc != 0 { anyhow::bail!("exchange_u32s_all: rank {rank} round 2 rc={rc}{}", hub_rc_note(rc)); }
        let out = hub_unpack(&x.recv_copy(0, n2 / 4), wire, mine.len(), world);
        anyhow::ensure!(out[rank] == mine, "exchange_u32s_all: rank {rank}'s payload came back altered from the head");
        return Ok(out);
    }
    // A failed gather must not return early: nodes already in round 1 would park in round 2 (the agree_ext rule).
    let mut all: Vec<Option<Vec<u32>>> = vec![None; world];
    all[0] = Some(mine.to_vec());
    // Every failed exchange is kept: once one link dies the abort fans out to the whole transport, so the
    // FIRST rank in loop order usually reports rc=-2 (abort already set) while the dead node is another
    // rank. A rc=-3/-4 failure names the peer that actually failed; the -2s are only echoes of it.
    let mut failed: Vec<(u8, usize, i32)> = Vec::new();
    for r in 1..world {
        let rc = x.exchange_one(r, n1);
        if rc != 0 { failed.push((1, r, rc)); continue; }
        all[r] = Some(x.recv_copy(r, mine.len()));
    }
    hub_pack(&all, wire, x.send_buf(n2 / 4));
    for r in 1..world {
        let rc = x.exchange_one(r, n2);
        if rc != 0 { failed.push((2, r, rc)); }
    }
    if !failed.is_empty() { anyhow::bail!("exchange_u32s_all: {}", hub_failure_summary(&failed)); }
    Ok(all.into_iter().map(Option::unwrap).collect())
}

/// The head hub's failure line: the culprit first (a rc=-3/-4 link, which names the failed peer), then
/// every exchange that failed. When every failure is rc=-2 the abort arrived from elsewhere before the
/// head reached any live link, and the line says the rank numbers do not identify the failed node.
fn hub_failure_summary(failed: &[(u8, usize, i32)]) -> String {
    let all: Vec<String> = failed.iter().map(|&(rd, r, rc)| format!("round {rd} rank {r} rc={rc}")).collect();
    match failed.iter().find(|&&(_, _, rc)| rc != -2) {
        Some(&(rd, r, rc)) => format!("head round {rd} with rank {r} rc={rc}{} — failed node: rank {r} [all failed exchanges: {}]",
                                      hub_rc_note(rc), all.join(", ")),
        None => format!("transport already aborted before the head reached a live link (rc=-2 on every exchange: {}) — \
                         these ranks are NOT necessarily the failed node; find it from the nodes (the one whose session \
                         process is gone or whose log shows the first fault)", all.join(", ")),
    }
}

fn hub_rc_note(rc: i32) -> &'static str {
    match rc {
        -2 => " (link already aborted)",
        -3 => " (peer dead or link error; the abort status word names the code)",
        -4 => " (lockstep deadline expired: the peer is alive but never reached this point; abort code 12)",
        _ => "",
    }
}

/// World-general `exchange_u32s`: every rank's payload, indexed by rank (`out[rank] == mine`); payload
/// lengths equal on all ranks (SPMD). world <= 2 is exactly `exchange_u32s` (the pairwise startup channel,
/// same wire bytes). world > 2 is a head hub over the DEDICATED control slots (`net_exchange_one`): round 1
/// every node sends its frame and the head collects; round 2 the head sends every node the concatenation of
/// all payloads, and the node's own block comes back as an echo check. The head is also a compute rank, so
/// nothing here assumes it is quiet: the hub touches neither the hot-path rings nor the GPU epoch counter.
/// `mine` needs 8 bytes of headroom inside its `wire_u32s` frame (the tail tag overwrites the last 8).
/// No lockstep deadline (boot / probe paths, where load skew is legitimate); see `exchange_u32s_all_dl`.
pub fn exchange_u32s_all(mine: &[u32], wire_u32s: usize) -> anyhow::Result<Vec<Vec<u32>>> {
    exchange_u32s_all_dl(mine, wire_u32s, 0)
}

/// `exchange_u32s_all` with a per-wait lockstep deadline (audit C5) at world > 2: a peer that is alive (the
/// dead-peer probe passes) but never reaches this lockstep point makes the wait fail with abort code 12 after
/// `deadline_ms` instead of holding every rank forever. `deadline_ms == 0` = unbounded. At world <= 2 the pairwise
/// channel is used unchanged and the deadline is IGNORED (the W=2 wire protocol and timing path stay as shipped).
pub fn exchange_u32s_all_dl(mine: &[u32], wire_u32s: usize, deadline_ms: u64) -> anyhow::Result<Vec<Vec<u32>>> {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { anyhow::bail!("exchange_u32s_all: no registered TP link (single-node?)"); }
    let ctx = c as *mut NetCtx;
    let (world, rank) = unsafe { (net_world(ctx) as usize, net_rank(ctx) as usize) };
    if world <= 2 {
        let peer = exchange_u32s(mine, wire_u32s)?;
        return Ok(if rank == 0 { vec![mine.to_vec(), peer] } else { vec![peer, mine.to_vec()] });
    }
    let mut x = CtxHub { ctx, world, rank, deadline_ns: deadline_ms.saturating_mul(1_000_000) };
    hub_all(&mut x, mine, wire_u32s)
}
