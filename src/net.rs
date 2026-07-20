//! TP=2 comm transport — thin safe-ish Rust wrapper over `native/net_shim.c` (libibverbs +
//! cudaHostAlloc). GB10 has NO GPUDirect, so comm buffers are `cudaHostAlloc` + `ibv_reg_mr`
//! (coherent to GPU+CPU+NIC); the GPU reduction reads/writes the same buffers via device pointers.
//!
//! The hot path is the doorbell all-reduce: a global epoch ring (R slots, S-signaled), a proxy that
//! owns the posted epoch and ships it INLINE, and a CPU-bounced receive (GB10 reports
//! `CAN_FLUSH_REMOTE_WRITES = 0`, so the GPU may not consume NIC-written payload directly). The
//! invariants live in `native/tp_doorbell.h`; the rationale in `tp_doorbell_ref/`.

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};

#[repr(C)]
pub struct NetCtx {
    _private: [u8; 0],
}

extern "C" {
    fn net_init(rank: c_int, peer_ip: *const c_char, tcp_port: c_int, dev_name: *const c_char,
                gid_idx: c_int, fp32_capacity_bytes: c_int, payload_bytes: c_int) -> *mut NetCtx;
    fn net_set_payload(c: *mut NetCtx, payload_bytes: c_int, fp32: c_int) -> c_int;
    fn net_ctx_dptr(c: *mut NetCtx) -> *mut c_void;
    fn net_flags_dptr(c: *mut NetCtx) -> *mut c_void;
    fn net_send_dptr(c: *mut NetCtx) -> *mut c_void;
    fn net_recv_dptr(c: *mut NetCtx) -> *mut c_void;
    fn net_send_hptr(c: *mut NetCtx) -> *mut c_void;
    fn net_recv_hptr(c: *mut NetCtx) -> *mut c_void;
    fn net_device_epoch(c: *mut NetCtx) -> u64;
    fn net_gate_waits(c: *mut NetCtx) -> u64;
    fn net_bench_cq_hold(c: *mut NetCtx, hold: u32, hold_us: u32) -> c_int;
    fn net_gpu_ready(c: *mut NetCtx) -> u64;
    fn net_tail_fires(c: *mut NetCtx) -> u64;
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
    fn net_abort(c: *mut NetCtx);
    fn net_shutdown(c: *mut NetCtx);
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
pub fn agree(step: u64, accept_count: u8, hash: u32) -> Option<(u8, u32)> {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { return None; }
    let val = ((step & 0xFF_FFFF) << 40) | ((accept_count as u64) << 32) | hash as u64;
    let got = unsafe { net_agree(c as *mut NetCtx, val, 0xFF_FFFF << 40, (step & 0xFF_FFFF) << 40) };
    if got == 0 { return None; }
    Some((((got >> 32) & 0xFF) as u8, (got & 0xFFFF_FFFF) as u32))
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

/// A 2-node tensor-parallel link (one RC QP, RoCEv2). `rank` 0 = head (listens), 1 = node (connects).
pub struct TpLink {
    ctx: *mut NetCtx,
    slot_bytes: usize,
}

impl TpLink {
    /// `slot_bytes` is the ring-slot CAPACITY — size it for the FP32 payload (and the startup prompt
    /// frame) so switching precision later never re-addresses the rings, which would invalidate a
    /// captured graph. The active hot-path payload is set separately by `set_payload`.
    pub fn connect(rank: i32, peer_ip: &str, tcp_port: u16, dev: &str, gid_idx: i32,
                   slot_bytes: usize) -> anyhow::Result<Self> {
        let peer = CString::new(peer_ip)?;
        let dev_c = CString::new(dev)?;
        // placeholder active payload; the model config sets the real one at attach time
        let ctx = unsafe {
            net_init(rank, peer.as_ptr(), tcp_port as c_int, dev_c.as_ptr(), gid_idx,
                     slot_bytes as c_int, 4)
        };
        if ctx.is_null() {
            anyhow::bail!("net_init failed (see [net_shim] logs above)");
        }
        Ok(TpLink { ctx, slot_bytes })
    }

    /// Set the active hot-path payload. MUST be called before the proxy thread starts: both the proxy
    /// and K1/K2 read it, and I8 forbids mutating protocol state under a running system.
    pub fn set_payload(&mut self, payload_bytes: usize, fp32: bool) -> anyhow::Result<()> {
        let rc = unsafe { net_set_payload(self.ctx, payload_bytes as c_int, fp32 as c_int) };
        if rc != 0 { anyhow::bail!("net_set_payload({payload_bytes}, fp32={fp32}) failed"); }
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

    /// Release a blocked exchange / stop the proxy (dead-peer / shutdown path). Cooperative: it sets
    /// the abort STATUS word, so in-flight kernels no-op through the stream rather than trapping (I9).
    pub fn abort(&self) { unsafe { net_abort(self.ctx) } }
}

impl Drop for TpLink {
    fn drop(&mut self) { unsafe { net_shutdown(self.ctx) } }
}

unsafe impl Send for TpLink {}
