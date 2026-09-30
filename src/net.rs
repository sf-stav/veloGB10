//! TP=2 comm transport — thin safe-ish Rust wrapper over `native/net_shim.c` (libibverbs +
//! cudaHostAlloc). GB10 has NO GPUDirect, so comm buffers are `cudaHostAlloc` + `ibv_reg_mr`
//! (coherent to GPU+CPU+NIC); the GPU reduction reads/writes the same buffers via device pointers.
//!
//! The hot path is the doorbell all-reduce: a global epoch ring (R slots, S-signaled), a proxy that
//! owns the posted epoch and ships it INLINE, and a CPU-bounced receive (GB10 reports
//! `CAN_FLUSH_REMOTE_WRITES = 0`, so the GPU may not consume NIC-written payload directly). The
//! invariants live in `native/tp_doorbell.h`; the rationale in `tp_doorbell_ref/`.

use std::ffi::CString;
use std::net::IpAddr;
use std::os::raw::{c_char, c_int, c_void};

#[repr(C)]
pub struct NetCtx {
    _private: [u8; 0],
}

extern "C" {
    /// CLI-1: the transport's options (the shim reads no env) — call before every net_init.
    fn net_set_opts(spin_us: u64, tail_drill: c_int, oneshot: c_int, tp_diag: c_int);
    fn net_init(rank: c_int, world: c_int, peer_ips: *const *const c_char, n_peers: c_int,
                tcp_port: c_int, dev_name: *const c_char,
                gid_idx: c_int, fp32_capacity_bytes: c_int, payload_bytes: c_int) -> *mut NetCtx;
    fn net_set_payload(c: *mut NetCtx, payload_bytes: c_int, fp32: c_int) -> c_int;
    fn net_oneshot_on(c: *mut NetCtx) -> c_int;   // P3-1: the ctx's one-shot selector (single source of truth)
    fn net_set_recv_mode(c: *mut NetCtx, gpu: c_int) -> c_int;
    fn net_rx_done(c: *mut NetCtx) -> u64;
    fn net_ctx_dptr(c: *mut NetCtx) -> *mut c_void;
    fn net_peer_ip(c: *mut NetCtx) -> *const c_char;
    fn net_flags_dptr(c: *mut NetCtx) -> *mut c_void;
    fn net_send_dptr(c: *mut NetCtx) -> *mut c_void;
    fn net_recv_dptr(c: *mut NetCtx) -> *mut c_void;
    fn net_send_hptr(c: *mut NetCtx) -> *mut c_void;
    fn net_recv_hptr(c: *mut NetCtx) -> *mut c_void;
    fn net_device_epoch(c: *mut NetCtx) -> u64;
    fn net_gate_waits(c: *mut NetCtx) -> u64;
    fn net_wait_sleeps() -> u64;
    fn net_bench_cq_hold(c: *mut NetCtx, hold: u32, hold_us: u32) -> c_int;
    fn net_gpu_ready(c: *mut NetCtx) -> u64;
    fn net_tail_fires(c: *mut NetCtx) -> u64;
    fn net_gpu_rx_skips(c: *mut NetCtx) -> u64;
    fn net_abort_status(c: *mut NetCtx) -> u64;
    fn net_exchange(c: *mut NetCtx, nbytes: c_int) -> c_int;
    fn net_flush(c: *mut NetCtx) -> c_int;
    fn net_proxy_loop(c: *mut NetCtx, core: c_int);
    fn net_pin_thread(core: c_int) -> c_int;
    fn net_bench_config(c: *mut NetCtx, inject_delay_us_max: u32, ts_on: c_int);
    fn net_now_ns() -> u64;
    fn net_cpu_ts(c: *mut NetCtx) -> *mut u64;
    fn net_gpu_ts(c: *mut NetCtx) -> *mut u64;
    fn net_counters(c: *mut NetCtx, posted: *mut u64, retired: *mut u64,
                    released: *mut u64, tail_fires: *mut u64);
    fn net_agree(c: *mut NetCtx, val: u64, step_mask: u64, step_val: u64) -> u64;
    fn net_exchange_one(c: *mut NetCtx, peer_rank: c_int, nbytes: c_int) -> c_int;
    fn net_ctrl_recv_hptr(c: *mut NetCtx, src: c_int) -> *mut c_void;
    fn net_ctrl_send_hptr(c: *mut NetCtx) -> *mut c_void;
    fn net_world(c: *mut NetCtx) -> c_int;
    fn net_rank(c: *mut NetCtx) -> c_int;
    fn net_abort(c: *mut NetCtx);
    fn net_shutdown(c: *mut NetCtx);
    // R9 DIAGNOSTIC (world>2, --tp-diag=1) — per-epoch payload checksum rings.
    fn net_diag_send_xor(c: *mut NetCtx) -> u64;
    fn net_diag_recv_xor(c: *mut NetCtx) -> u64;
    fn net_diag_send_hptr(c: *mut NetCtx) -> *mut c_void;
    fn net_diag_recv_hptr(c: *mut NetCtx) -> *mut c_void;
    fn net_diag_send_idx(c: *mut NetCtx) -> u64;
    fn net_diag_recv_idx(c: *mut NetCtx) -> u64;
    fn net_diag_folds(c: *mut NetCtx, send_fold: *mut u64, recv_fold: *mut u64);
    fn net_diag_dump(c: *mut NetCtx, path: *const c_char);
}

/// Ring geometry mirror (TP_DIAG_RING_EPOCHS in net_shim.c). Entries are [epoch, partner, fnv64].
pub const DIAG_RING_EPOCHS: usize = 65536;

/// This process's TP rank from the registered link (0 = head). 0 when no link (single-node).
pub fn diag_rank() -> i32 {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { 0 } else { unsafe { net_rank(c as *mut NetCtx) } }
}

/// Per-epoch CPU timestamp ring stride and slot indices (mirrors `net_shim.c`).
pub const CTS_STRIDE: usize = 5;
pub const CTS_READY: usize = 0;
pub const CTS_POSTED: usize = 1;
pub const CTS_CQE: usize = 2;
pub const CTS_PEERSEEN: usize = 3;
pub const CTS_RELEASED: usize = 4;
/// Per-epoch GPU timestamp ring (mirrors `native/tp_doorbell.h`).
pub const GTS_EPOCHS: usize = 4096;
pub const GTS_STRIDE: usize = 4;
pub const GTS_K1_IN: usize = 0;
pub const GTS_K1_OUT: usize = 1;
pub const GTS_K2_IN: usize = 2;
pub const GTS_K2_GO: usize = 3;

/// TP-E: 1 ms sleeps taken so far by the host rendezvous waits (net_agree / net_exchange / the proxy's
/// placement waits), process-wide. Mid-request this should stay 0 under the TP-E spin budget.
pub fn wait_sleeps() -> u64 { unsafe { net_wait_sleeps() } }

/// Pin the CALLING thread to `core` and VERIFY the affinity read back (GB10 is big.LITTLE; a launch or
/// poll thread parked on a little A725 balloons latency and drains the GPU stream mid-token). Returns
/// false if the mask did not take — treat that as a measurement-invalidating fault, not a warning:
/// scheduling jitter is indistinguishable from a protocol stall in the numbers.
pub fn pin_thread(core: i32) -> bool { unsafe { net_pin_thread(core as c_int) == 0 } }

/// `CLOCK_MONOTONIC_RAW` in ns — the exact clock the proxy stamps its per-epoch timestamps with, so
/// bench deltas against them are meaningful (`Instant` is `CLOCK_MONOTONIC` and drifts from `_RAW`).
pub fn now_ns() -> u64 { unsafe { net_now_ns() } }

// ---- trace hook: lets the MODEL run report the same per-barrier histograms as the microbench ----
// The link is handed to the proxy thread and never returned, so the ctx address is stashed here for the
// post-run dump. Without this the only barrier numbers we have come from the bench, and "the bench is
// fast but the model is slow" is precisely the question that needs data rather than reasoning.
static TRACE_CTX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Enable per-epoch timestamping on a link and register it for `trace_dump`. Call BEFORE the proxy
/// starts. No-op cost when never called: the proxy checks one flag per stamp.
pub fn trace_enable(link: &mut TpLink) {
    link.bench_config(0, true);
    TRACE_CTX.store(link.ctx_addr(), std::sync::atomic::Ordering::Relaxed);
}
/// `(gpu_ts, cpu_ts, counters, gate_waits, tail_fires)` for the traced link, if tracing was enabled.
pub fn trace_data() -> Option<(&'static [u64], &'static [u64], (u64, u64, u64, u64), u64, u64)> {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { return None; }
    let c = c as *mut NetCtx;
    unsafe {
        let (g, cp) = (net_gpu_ts(c), net_cpu_ts(c));
        if g.is_null() || cp.is_null() { return None; }
        let (mut p, mut r, mut rel, mut tf) = (0u64, 0u64, 0u64, 0u64);
        net_counters(c, &mut p, &mut r, &mut rel, &mut tf);
        Some((std::slice::from_raw_parts(g, GTS_EPOCHS * GTS_STRIDE),
              std::slice::from_raw_parts(cp, GTS_EPOCHS * CTS_STRIDE),
              (p, r, rel, tf), net_gate_waits(c), net_device_epoch(c)))
    }
}

/// Lockstep agreement for MTP under TP: publish this rank's `(step, accept_count, hash)` token and block
/// until the peer's token for the SAME step arrives. Returns None if the link aborted.
///
/// This exists because acceptance divergence is silent and permanent: if the two ranks ever accept a
/// different number of drafted tokens they execute different barrier sequences forever after. Count alone
/// is not enough — same count with different token ids desyncs the KV and recurrent state just as badly —
/// so the token carries a hash of the accepted ids too.
///
/// world==2 keeps the proven pairwise `net_agree` (proxy inline-ships the token over the single QP) byte
/// for byte. world>2 uses a head-hub gather+broadcast over `net_exchange_one`: every rank sends its token
/// to the head, the head verifies all tokens are equal, then broadcasts the consensus (or a mismatch
/// sentinel) back — every rank returns `Some(consensus)` on full agreement, `None` on any divergence.
pub fn agree(step: u64, accept_count: u8, hash: u32) -> Option<(u8, u32)> {
    agree_ext(step, accept_count, 0, hash)
}

/// B8/G1 EXTENDED determinism token: `(step | k_verify | accept_count | hash_of_ids)`.
/// `k_verify` is this step's VERIFY WIDTH (the confidence-truncated bucket width under DSpark; the
/// plain chain width under MTP). Ranks that disagree on k_verify execute different barrier
/// sequences — the I9 silent-desync class — so it joins the per-step agreement token. The wire
/// shape stays the 64-bit token (no protocol change): the 24-bit step field keeps [40..64),
/// accept_count [32..40), and k_verify is folded into the hash word at bits [27..31) (4 bits,
/// MAX_VERIFY=16 fits exactly), which every caller already compares in full. MTP callers can keep
/// the 3-arg `agree` (k_verify=0); speculative callers that choose a width MUST use this one.
pub fn agree_ext(step: u64, accept_count: u8, k_verify: u8, hash: u32) -> Option<(u8, u32)> {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { return None; }
    let h_ext = (hash ^ ((k_verify as u32 & 0xF) << 27)) as u32;
    let val = ((step & 0xFF_FFFF) << 40) | ((accept_count as u64) << 32) | h_ext as u64;
    let ctx = c as *mut NetCtx;
    let world = unsafe { net_world(ctx) };
    if world <= 2 {
        // R1: pairwise path unchanged.
        let got = unsafe { net_agree(ctx, val, 0xFF_FFFF << 40, (step & 0xFF_FFFF) << 40) };
        if got == 0 { return None; }
        return Some((((got >> 32) & 0xFF) as u8, (got & 0xFFFF_FFFF) as u32));
    }
    // ---- world > 2: head-hub gather + broadcast over the dedicated control slots. ----
    // Control frame layout (nbytes = 32): [0..8) token u64, [8..16) status u64 (0 = ok, 1 = mismatch),
    // [16..24) the device epoch probe (this rank's barrier counter at agree time), [24..32) the tail
    // tag (written by net_exchange_one). The epoch probe is diagnostic (the (k) epoch-divergence
    // hypothesis): if any rank is one barrier ahead/behind, the head sees it immediately.
    const AGREE_NBYTES: usize = 32;
    const STATUS_OK: u64 = 0;
    const STATUS_MISMATCH: u64 = 1;
    let diag = crate::opts::var(crate::opt!("tp-diag")).is_ok();
    let rank = unsafe { net_rank(ctx) };
    let epoch = unsafe { net_device_epoch(ctx) };
    let send = unsafe { net_ctrl_send_hptr(ctx) as *mut u64 };
    unsafe {
        // Round 1: stage this rank's token + ok status + the device-epoch probe.
        std::ptr::write_unaligned(send, val);
        // R9 DIAGNOSTIC: status word carries the send-ring (epoch,partner) XOR — an O(1) proof that
        // every rank SAW the same barrier schedule. Frame stays 32 B (byte-identical wire shape).
        std::ptr::write_unaligned(send.add(1),
            if diag { net_diag_send_xor(ctx) } else { STATUS_OK });
        std::ptr::write_unaligned(send.add(2), epoch);
        if rank == 0 {
            // Head: gather every node's token, then compute consensus.
            let mut all_equal = true;
            let mut diag_sched_ok = true;   // R9: schedule-ambiguity check across the gathered tokens
            let my_sx = if diag { net_diag_send_xor(ctx) } else { 0 };
            // B8 §1.5-6: a round-1 failure must NOT early-return — that parks nodes r+1.. in their
            // round-1 placement wait. Collect the failed peer, keep gathering, then fan round 2 (the
            // mismatch sentinel) so every node unblocks before we abort.
            let mut round1_failed = false;
            for r in 1..world {
                let rc = net_exchange_one(ctx, r, AGREE_NBYTES as c_int);
                if rc != 0 {
                    // B8 §1.5-1: echo the abort code + watermarks — rc=-2 means the abort flag was
                    // ALREADY set (a checkpoint, not a fault); the code classifies the real cause.
                    eprintln!("[tp-agree] head round-1 exchange_one({r}) rc={rc} abort_status={:#018x} device_epoch={} gpu_ready={} rx_done={} — fanning round 2 anyway",
                              net_abort_status(ctx), net_device_epoch(ctx), net_gpu_ready(ctx), net_rx_done(ctx));
                    round1_failed = true;
                    continue;
                }
                let slot = net_ctrl_recv_hptr(ctx, r) as *const u64;
                let peer_val = std::ptr::read_unaligned(slot);
                let peer_sx = std::ptr::read_unaligned(slot.add(1));
                let peer_epoch = std::ptr::read_unaligned(slot.add(2));
                if peer_epoch != epoch {
                    eprintln!("[tp-agree] EPOCH-PROBE: rank {r} device epoch {peer_epoch} != head {epoch} (delta {})",
                              peer_epoch as i64 - epoch as i64);
                }
                if diag && peer_sx != my_sx {
                    eprintln!("[tp-agree] DIAG-SCHED: rank {r} send-ring XOR {peer_sx:#018x} != head {my_sx:#018x} — barrier SCHEDULE diverged (not a content race)");
                    diag_sched_ok = false;
                }
                if peer_val != val {
                    eprintln!("[tp-agree] head MISMATCH: rank {r} val {peer_val:#018x} != head {val:#018x} (hash {peer_val:08x} vs {val:08x})");
                    all_equal = false;
                }
            }
            // R9 DIAGNOSTIC: on the FIRST mismatch, freeze and localize. Every rank dumps its own
            // checksum rings (nodes do it on receipt of the mismatch sentinel below); the head
            // prints its own per-partner folds here so the divergent (epoch, partner) can be read
            // straight off the logs even before the offline ring diff.
            if diag && !all_equal {
                eprintln!("[tp-agree] DIAG: schedule-XOR {} — dumping per-epoch checksum rings",
                          if diag_sched_ok { "MATCH (content race, not a schedule bug)" } else { "MISMATCH" });
                let mut sf = [0u64; 16]; let mut rf = [0u64; 16];
                net_diag_folds(ctx, sf.as_mut_ptr(), rf.as_mut_ptr());
                for p in 0..world as usize {
                    eprintln!("[tp-agree] DIAG head folds: partner {p} send={:#018x} recv={:#018x}", sf[p], rf[p]);
                }
                let path = CString::new("/tmp/tp_diag_head").unwrap();
                net_diag_dump(ctx, path.as_ptr());
                // R9 layer localizer: dump the xchain capture sink (per-layer residuals + logits).
                eprintln!("[tp-agree] R9 head xchain dump: sink_entries={}", crate::gpu::xchain_sink_len());
                let _ = crate::gpu::xchain_rank_dump("/tmp/tp_xchain", rank);
                // R9 DECISIVE: the GDN recurrent state (never on the wire) — the divergence source.
                crate::batch::r9_dump_gdn_state(rank);
            }
            // Round 2: fan the consensus (or mismatch/abort sentinel) back out — ALWAYS complete round 2
            // (even on a round-1 failure) so every node's exchange_one unblocks before any abort.
            let (out_val, out_status) = if all_equal && !round1_failed { (val, STATUS_OK) }
                                        else { (0u64, STATUS_MISMATCH) };
            std::ptr::write_unaligned(send, out_val);
            std::ptr::write_unaligned(send.add(1), out_status);
            for r in 1..world {
                let rc = net_exchange_one(ctx, r, AGREE_NBYTES as c_int);
                if rc != 0 {
                    eprintln!("[tp-agree] head round-2 exchange_one({r}) rc={rc} abort_status={:#018x} device_epoch={} gpu_ready={} rx_done={}",
                              net_abort_status(ctx), net_device_epoch(ctx), net_gpu_ready(ctx), net_rx_done(ctx));
                    round1_failed = true;
                }
            }
            if round1_failed || !all_equal { return None; }
            Some((((val >> 32) & 0xFF) as u8, (val & 0xFFFF_FFFF) as u32))
        } else {
            // Node: one round-1 exchange (send my token), one round-2 exchange (receive consensus).
            let rc1 = net_exchange_one(ctx, 0, AGREE_NBYTES as c_int);
            if rc1 != 0 {
                eprintln!("[tp-agree] node rank {rank} round-1 exchange_one(0) rc={rc1} abort_status={:#018x} device_epoch={} gpu_ready={} rx_done={} — returning None",
                          net_abort_status(ctx), net_device_epoch(ctx), net_gpu_ready(ctx), net_rx_done(ctx));
                return None;
            }
            let rc2 = net_exchange_one(ctx, 0, AGREE_NBYTES as c_int);
            if rc2 != 0 {
                eprintln!("[tp-agree] node rank {rank} round-2 exchange_one(0) rc={rc2} abort_status={:#018x} device_epoch={} gpu_ready={} rx_done={} — returning None",
                          net_abort_status(ctx), net_device_epoch(ctx), net_gpu_ready(ctx), net_rx_done(ctx));
                return None;
            }
            let slot = net_ctrl_recv_hptr(ctx, 0) as *const u64;
            let status = std::ptr::read_unaligned(slot.add(1));
            let out = std::ptr::read_unaligned(slot);
            if status != STATUS_OK {
                eprintln!("[tp-agree] node rank {rank}: head sent MISMATCH sentinel (my val {val:#018x}, out {out:#018x})");
                // R9 DIAGNOSTIC: dump this rank's checksum rings so the head-side diff has all 4.
                if diag {
                    let mut sf = [0u64; 16]; let mut rf = [0u64; 16];
                    net_diag_folds(ctx, sf.as_mut_ptr(), rf.as_mut_ptr());
                    for p in 0..world as usize {
                        eprintln!("[tp-agree] DIAG node rank {rank} folds: partner {p} send={:#018x} recv={:#018x}", sf[p], rf[p]);
                    }
                    let path = CString::new("/tmp/tp_diag_node").unwrap();
                    net_diag_dump(ctx, path.as_ptr());
                    // R9 layer localizer: the node's xchain capture sink.
                    eprintln!("[tp-agree] R9 node rank {rank} xchain dump: sink_entries={}", crate::gpu::xchain_sink_len());
                    let _ = crate::gpu::xchain_rank_dump("/tmp/tp_xchain", rank);
                    // R9 DECISIVE: the GDN recurrent state.
                    crate::batch::r9_dump_gdn_state(rank);
                }
                return None;
            }
            if out == 0 { return None; }
            Some((((out >> 32) & 0xFF) as u8, (out & 0xFFFF_FFFF) as u32))
        }
    }
}

/// Ship a small u32 payload to the peer over the startup/audit channel (the `TpLink::exchange`
/// path, using the process-registered ctx — works while the RDMA proxy runs: the exchange's send
/// CQE is handed over via `xchg_send_done`, and the recv ring is separate from the hot-path rings).
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
        let send = std::slice::from_raw_parts_mut(net_send_hptr(ctx) as *mut u32, wire_u32s);
        for x in send.iter_mut() { *x = 0; }
        send[..mine.len()].copy_from_slice(mine);
        let rc = net_exchange(ctx, nbytes as c_int);
        if rc != 0 { anyhow::bail!("net_exchange failed rc={rc}"); }
        let recv = std::slice::from_raw_parts(net_recv_hptr(ctx) as *const u32, wire_u32s);
        Ok(recv[..mine.len()].to_vec())
    }
}

/// Abort the registered TP link (cooperative stop: the abort STATUS word makes in-flight kernels
/// no-op through the stream rather than trapping, I9). No-op on a single-node run. Used by the
/// per-step agreement guard (TP item D) to take both ranks down together on a proven divergence.
pub fn abort_link() {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c != 0 { unsafe { net_abort(c as *mut NetCtx) } }
}

/// Device epoch / published watermark of the traced link — the I8 tripwire for graph instantiation.
/// Returns 0 when no link is registered (single-node), which makes the assert vacuous there.
pub fn traced_device_epoch() -> u64 {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { 0 } else { unsafe { net_device_epoch(c as *mut NetCtx) } }
}
pub fn traced_gpu_ready() -> u64 {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { 0 } else { unsafe { net_gpu_ready(c as *mut NetCtx) } }
}
/// The GPU's receive watermark (TP_F_RX_DONE) — the v2-receive watchdog's debt signal and the
/// graph-instantiation tripwire sibling (rx_done == device_epoch at quiesce). 0 when no link.
pub fn traced_rx_done() -> u64 {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { 0 } else { unsafe { net_rx_done(c as *mut NetCtx) } }
}

/// The link's cooperative abort status word (0 = healthy) for the CURRENT registered ctx, or 0 when
/// no TP link is attached. Mirrors `traced_rx_done` — used by the acceptance gates so an aborted
/// run FAILS LOUDLY instead of reporting a number computed on no-op'd kernels (I9).
/// TP-F: an auxiliary link's cooperative abort status (the dual-rail prefill transport's second rail).
pub fn ctx_abort_status(ctx_addr: usize) -> u64 { unsafe { net_abort_status(ctx_addr as *mut NetCtx) } }

pub fn traced_abort_status() -> u64 {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { 0 } else { unsafe { net_abort_status(c as *mut NetCtx) } }
}

/// The link's tail-epoch guard fire count (MUST stay 0; a fire means RC/PCIe placement ordering
/// failed). Same traced pattern as `traced_abort_status`.
/// TP-I: wait until the proxy has posted AND retired (send CQE) every hot-path epoch the device has
/// published — after a device synchronize, this means no doorbell payload of ours is still being read
/// out of the send ring and none of the peer's (consumed by our K2s) is still landing in the recv ring.
/// The host exchanges stage in send slot 0 and receive in recv slot g&7 of the SAME rings, so a host
/// exchange must not overlap an in-flight epoch. Bounded (the proxy retires in microseconds); Err past
/// `timeout`. No-op without a registered link.
pub fn drain_sends(timeout: std::time::Duration) -> anyhow::Result<()> {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { return Ok(()); }
    let c = c as *mut NetCtx;
    let t0 = std::time::Instant::now();
    loop {
        let e = unsafe { net_device_epoch(c) };
        let (mut p, mut r) = (0u64, 0u64);
        unsafe { net_counters(c, &mut p, &mut r, std::ptr::null_mut(), std::ptr::null_mut()) };
        if p >= e && r >= e { return Ok(()); }
        if t0.elapsed() > timeout {
            anyhow::bail!("TP drain: the proxy has not retired the device's epochs (device {e}, posted {p}, retired {r}) within {:?}", timeout);
        }
        std::hint::spin_loop();
    }
}

/// TP-I (C7): the I3 reuse-gate bind counter of the registered link's device ctx (cumulative; the
/// folded decode K1 now counts on whichever block runs the gate).
pub fn traced_gate_waits() -> u64 {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { 0 } else { unsafe { net_gate_waits(c as *mut NetCtx) } }
}

pub fn traced_tail_fires() -> u64 {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { 0 } else { unsafe { net_tail_fires(c as *mut NetCtx) } }
}

/// TP-G: epochs the proxy did NOT validate because the GPU decode K2 proved them first (RX_DONE).
pub fn traced_gpu_rx_skips() -> u64 {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { 0 } else { unsafe { net_gpu_rx_skips(c as *mut NetCtx) } }
}


/// Spawn the persistent proxy loop for a TP link on its own thread, pinned to `core`. `ctx_addr` is a
/// raw `*mut NetCtx` (from `TpLink::ctx_addr`); the caller must keep the ctx alive for the run (the
/// proxy owns the transport from here, so the main thread `mem::forget`s the TpLink).
pub fn spawn_proxy(ctx_addr: usize, core: i32) -> std::thread::JoinHandle<()> {
    // Register the ctx for the whole process: `agree()` and the trace accessors need it on EVERY TP run,
    // not just traced ones.
    TRACE_CTX.store(ctx_addr, std::sync::atomic::Ordering::Relaxed);
    std::thread::spawn(move || {
        let ctx = ctx_addr as *mut NetCtx;
        unsafe { net_proxy_loop(ctx, core as c_int); }
    })
}

/// TP-F: spawn the proxy of an AUXILIARY link (the dual-rail prefill transport's second rail).
/// Unlike `spawn_proxy` it does NOT register the ctx as the process's agree/trace link.
pub fn spawn_proxy_aux(ctx_addr: usize, core: i32) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let ctx = ctx_addr as *mut NetCtx;
        unsafe { net_proxy_loop(ctx, core as c_int); }
    })
}

/// A 2-node tensor-parallel link (one RC QP, RoCEv2). `rank` 0 = head (listens), 1 = node (connects).
pub struct TpLink {
    ctx: *mut NetCtx,
    slot_bytes: usize,
}

/// CLI-1: hand the registry's transport options to the C shim (the old getenv reads, same meanings:
/// --tp-spin-us N>=1 overrides the 20 ms spin budget; --tp-tail-drill / --tp-diag presence;
/// --tp-oneshot any value not starting with '0'). Called before every net_init.
fn push_net_opts() {
    use crate::{opt, opts};
    let spin: u64 = opts::var(opt!("tp-spin-us")).ok()
        .and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0);
    let tail = opts::var(opt!("tp-tail-drill")).is_ok() as c_int;
    let oneshot = opts::var(opt!("tp-oneshot")).map_or(false, |v| !v.starts_with('0')) as c_int;
    let diag = opts::var(opt!("tp-diag")).is_ok() as c_int;
    unsafe { net_set_opts(spin, tail, oneshot, diag) }
}

impl TpLink {
    /// `slot_bytes` is the ring-slot CAPACITY — size it for the FP32 payload (and the startup prompt
    /// frame) so switching precision later never re-addresses the rings, which would invalidate a
    /// captured graph. The active hot-path payload is set separately by `set_payload`.
    /// P3-1: does this transport ctx run the one-shot all-peers push? (--tp-oneshot + world==4,
    /// resolved at net_init — the same field the proxy and K1 read. Single source of truth.)
    pub fn oneshot_on(&self) -> bool { unsafe { net_oneshot_on(self.ctx) != 0 } }

    pub fn connect(rank: i32, peer_ip: &str, tcp_port: u16, dev: &str, gid_idx: i32,
                   slot_bytes: usize) -> anyhow::Result<Self> {
        push_net_opts();
        // world==2 legacy call: a synthetic 2-entry peer-IP list indexed by rank (peer_ips[1-rank]
        // is the peer — identical to the old single peer_ip). net_init dispatches to the unchanged
        // single-QP path. (rank 0 may pass "" — it listens and never dials.)
        let peer_ips: [&str; 2] = if rank == 0 { ["0.0.0.0", peer_ip] } else { [peer_ip, "0.0.0.0"] };
        let dev_c = CString::new(dev)?;
        let cstrs: Vec<CString> = peer_ips.iter().map(|s| CString::new(*s)).collect::<Result<_, _>>()?;
        let ptrs: Vec<*const c_char> = cstrs.iter().map(|s| s.as_ptr()).collect();
        let ctx = unsafe {
            net_init(rank, 2, ptrs.as_ptr(), 1, tcp_port as c_int, dev_c.as_ptr(), gid_idx,
                     slot_bytes as c_int, 4)
        };
        if ctx.is_null() {
            anyhow::bail!("net_init failed (see [net_shim] logs above)");
        }
        Ok(TpLink { ctx, slot_bytes })
    }

    /// N-way bring-up: `world` ranks, `peer_ips` indexed by PEER RANK (entry `[rank]` is unused).
    /// world==2 takes the exact single-QP fast path; world>2 builds world-1 per-peer QPs. In P3 the
    /// world>2 peer list is a PLACEHOLDER (P4 fills the real topology) — see bring_up_head/node.
    pub fn connect_nway(rank: i32, world: i32, peer_ips: &[IpAddr], tcp_port: u16, dev: &str,
                        gid_idx: i32, slot_bytes: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(world >= 2, "connect_nway: world must be >= 2");
        push_net_opts();
        anyhow::ensure!(peer_ips.len() >= world as usize, "connect_nway: peer_ips too short");
        let dev_c = CString::new(dev)?;
        let cstrs: Vec<CString> = peer_ips
            .iter()
            .map(|ip| CString::new(ip.to_string()))
            .collect::<Result<_, _>>()?;
        let ptrs: Vec<*const c_char> = cstrs.iter().map(|s| s.as_ptr()).collect();
        // placeholder active payload; the model config sets the real one at attach time
        let ctx = unsafe {
            net_init(rank, world as c_int, ptrs.as_ptr(), (world - 1) as c_int,
                     tcp_port as c_int, dev_c.as_ptr(), gid_idx,
                     slot_bytes as c_int, 4)
        };
        if ctx.is_null() {
            anyhow::bail!("net_init failed (see [net_shim] logs above)");
        }
        Ok(TpLink { ctx, slot_bytes })
    }

    /// Set the DEFAULT hot-path payload (the K1 nbytes==0 path — decode/bench barriers; chunked
    /// prefill barriers carry a per-call length). MUST be called before the proxy thread starts: both
    /// the proxy and K1/K2 read it, and I8 forbids mutating protocol state under a running system.
    pub fn set_payload(&mut self, payload_bytes: usize, fp32: bool) -> anyhow::Result<()> {
        let rc = unsafe { net_set_payload(self.ctx, payload_bytes as c_int, fp32 as c_int) };
        if rc != 0 { anyhow::bail!("net_set_payload({payload_bytes}, fp32={fp32}) failed"); }
        Ok(())
    }

    /// v2 receive mode (EXPERT_GPU_ALLREDUCE §8): with `gpu=true` the GPU kernels validate the
    /// NIC-written payload tail directly and the proxy skips its RECV stage. MUST be called before
    /// the proxy thread starts (same discipline as set_payload). Default off (v1 CPU bounce).
    pub fn set_recv_mode(&mut self, gpu: bool) -> anyhow::Result<()> {
        let rc = unsafe { net_set_recv_mode(self.ctx, gpu as c_int) };
        if rc != 0 { anyhow::bail!("net_set_recv_mode(gpu={gpu}) failed"); }
        Ok(())
    }

    /// Host view of ring slot 0 — used by the startup/audit channel (`exchange`), never the hot path.
    pub fn send_host_mut<T: Copy>(&mut self, n: usize) -> &mut [T] {
        assert!(n * std::mem::size_of::<T>() <= self.slot_bytes, "send slot overflow");
        unsafe { std::slice::from_raw_parts_mut(net_send_hptr(self.ctx) as *mut T, n) }
    }
    pub fn recv_host<T: Copy>(&self, n: usize) -> &[T] {
        assert!(n * std::mem::size_of::<T>() <= self.slot_bytes, "recv slot overflow");
        unsafe { std::slice::from_raw_parts(net_recv_hptr(self.ctx) as *const T, n) }
    }

    /// Device pointer to the `tp_dev_ctx` — the ONLY argument K1/K2 take. Everything the protocol needs
    /// (epoch, ring bases, stride, rank, precision) is derived from it on-device, which is what makes
    /// CUDA-graph capture a no-op instead of a rewrite (round-3 capture-hygiene rule).
    pub fn ctx_device_ptr(&self) -> u64 { unsafe { net_ctx_dptr(self.ctx) as u64 } }
    pub fn flags_device_ptr(&self) -> u64 { unsafe { net_flags_dptr(self.ctx) as u64 } }
    pub fn send_device_ptr(&self) -> u64 { unsafe { net_send_dptr(self.ctx) as u64 } }
    pub fn recv_device_ptr(&self) -> u64 { unsafe { net_recv_dptr(self.ctx) as u64 } }
    pub fn ctx_addr(&self) -> usize { self.ctx as usize }
    /// The peer's IP as this link's TCP handshake saw it (rank 0: learned from accept(); rank 1: dialed).
    pub fn peer_ip(&self) -> String {
        unsafe { std::ffi::CStr::from_ptr(net_peer_ip(self.ctx)).to_string_lossy().into_owned() }
    }

    /// Device-side barrier counter (source of truth) and the published watermark. Equal at quiesce —
    /// assert that at graph instantiation (I8/Q4 tripwire).
    pub fn device_epoch(&self) -> u64 { unsafe { net_device_epoch(self.ctx) } }
    pub fn gpu_ready(&self) -> u64 { unsafe { net_gpu_ready(self.ctx) } }
    /// Tail-epoch guard fire count. MUST be 0 — a nonzero value is an RC/PCIe ordering violation that
    /// reached us, and the empirical closure on the `CAN_FLUSH_REMOTE_WRITES=0` question.
    pub fn tail_fires(&self) -> u64 { unsafe { net_tail_fires(self.ctx) } }
    pub fn abort_status(&self) -> u64 { unsafe { net_abort_status(self.ctx) } }

    pub fn bench_config(&mut self, inject_delay_us_max: u32, ts_on: bool) {
        unsafe { net_bench_config(self.ctx, inject_delay_us_max, ts_on as c_int) }
    }
    /// Number of times K1 actually blocked on the I3 reuse gate — the proof it was exercised.
    pub fn gate_waits(&self) -> u64 { unsafe { net_gate_waits(self.ctx) } }
    /// Withhold CQ retirement credit until `hold` epochs are outstanding, forcing the reuse gate to
    /// bind. Must be <= R+1, else it deadlocks by construction rather than testing anything.
    pub fn bench_cq_hold(&mut self, hold: u32, hold_us: u32) -> anyhow::Result<()> {
        if unsafe { net_bench_cq_hold(self.ctx, hold, hold_us) } != 0 {
            anyhow::bail!("cq_hold {hold} exceeds R — that deadlocks by construction (max is R)");
        }
        Ok(())
    }
    pub fn cpu_ts(&self) -> Option<&[u64]> {
        let p = unsafe { net_cpu_ts(self.ctx) };
        if p.is_null() { None } else { Some(unsafe { std::slice::from_raw_parts(p, GTS_EPOCHS * CTS_STRIDE) }) }
    }
    pub fn gpu_ts(&self) -> Option<&[u64]> {
        let p = unsafe { net_gpu_ts(self.ctx) };
        if p.is_null() { None } else { Some(unsafe { std::slice::from_raw_parts(p, GTS_EPOCHS * GTS_STRIDE) }) }
    }
    /// `(posted, retired, released, tail_fires)` from the proxy.
    pub fn counters(&self) -> (u64, u64, u64, u64) {
        let (mut p, mut r, mut rel, mut tf) = (0u64, 0u64, 0u64, 0u64);
        unsafe { net_counters(self.ctx, &mut p, &mut r, &mut rel, &mut tf) };
        (p, r, rel, tf)
    }

    /// Forced signaled flush — post one signaled WR and drain, so every outstanding unsignaled WR
    /// becomes observably retired. For quiesce / finite-bench end (round-3 R3b).
    pub fn flush(&mut self) -> anyhow::Result<()> {
        match unsafe { net_flush(self.ctx) } {
            0 => Ok(()),
            -2 => anyhow::bail!("flush aborted"),
            e => anyhow::bail!("flush error {e}"),
        }
    }

    /// One all-reduce EXCHANGE over the retained WITH_IMM startup channel (slot 0). Off the hot path —
    /// the numerical audit (`--net-test`), the prompt broadcast, and the out-of-band re-init channel.
    pub fn exchange(&mut self, nbytes: usize) -> anyhow::Result<()> {
        assert!(nbytes <= self.slot_bytes, "exchange nbytes > slot");
        match unsafe { net_exchange(self.ctx, nbytes as c_int) } {
            0 => Ok(()),
            -2 => anyhow::bail!("exchange aborted"),
            e => anyhow::bail!("exchange error {e}"),
        }
    }

    /// P5 world>2: one bidirectional control-plane rendezvous with a SPECIFIC peer (head<->node) over
    /// the dedicated per-rank control slots. The caller stages its payload into `send_host_mut` first;
    /// on return the peer's payload for this exchange is in `ctrl_recv(peer_rank)`. world==2 must use
    /// `exchange` (this panics as an invariant guard — it is never routed for world==2).
    pub fn exchange_one(&mut self, peer_rank: i32, nbytes: usize) -> anyhow::Result<()> {
        assert!(self.world() > 2, "exchange_one is world>2 only");
        assert!(nbytes <= self.slot_bytes, "exchange_one nbytes > slot");
        match unsafe { net_exchange_one(self.ctx, peer_rank as c_int, nbytes as c_int) } {
            0 => Ok(()),
            -2 => anyhow::bail!("exchange_one aborted"),
            e => anyhow::bail!("exchange_one error {e}"),
        }
    }

    /// Host view of the world>2 control receive slot for sender rank `src` (read the peer's reply after
    /// `exchange_one`).
    pub fn ctrl_recv<T: Copy>(&self, src: i32, n: usize) -> &[T] {
        unsafe { std::slice::from_raw_parts(net_ctrl_recv_hptr(self.ctx, src as c_int) as *const T, n) }
    }

    /// Mutable host view of the world>2 dedicated control SEND staging slot (stage `exchange_one`'s
    /// outgoing payload here — separate from the hot-path send ring and the world==2 `net_send_hptr`).
    pub fn ctrl_send_mut<T: Copy>(&mut self, n: usize) -> &mut [T] {
        assert!(n * std::mem::size_of::<T>() <= self.slot_bytes, "ctrl send slot overflow");
        unsafe { std::slice::from_raw_parts_mut(net_ctrl_send_hptr(self.ctx) as *mut T, n) }
    }

    pub fn world(&self) -> i32 { unsafe { net_world(self.ctx) } }

    /// Release a blocked exchange / stop the proxy (dead-peer / shutdown path). Cooperative: it sets
    /// the abort STATUS word, so in-flight kernels no-op through the stream rather than trapping (I9).
    pub fn abort(&self) { unsafe { net_abort(self.ctx) } }
}

impl Drop for TpLink {
    fn drop(&mut self) { unsafe { net_shutdown(self.ctx) } }
}

unsafe impl Send for TpLink {}
