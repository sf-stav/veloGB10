//! TP=2 comm transport — thin safe-ish Rust wrapper over `native/net_shim.c` (libibverbs +
//! cudaHostAlloc). GB10 has NO GPUDirect, so comm buffers are `cudaHostAlloc` + `ibv_reg_mr`
//! (coherent to GPU+CPU+NIC); the GPU reduction reads/writes the same buffers via device pointers.
//!
//! The hot path is the doorbell all-reduce: a global epoch ring (R slots, S-signaled), a proxy that
//! owns the posted epoch and ships it INLINE, and a CPU-bounced receive (GB10 reports
//! `CAN_FLUSH_REMOTE_WRITES = 0`, so the GPU may not consume NIC-written payload directly). The
//! invariants live in `native/tp_doorbell.h`; the rationale in `tp_doorbell_ref/`.
//!
//! File map (the former single `net.rs`; every item keeps its `crate::net::` path): this file holds the FFI wrapper, link,
//! agree / exchange protocols and liveness code; the former inline `#[cfg(test)]` modules live in files named after them:
//! `hub.rs` (the world>2 hub exchange: `exchange_u32s*`, `HubXport`, `CtxHub`); tests: `hub_mock.rs` (in-memory `HubXport`), `hub_tests.rs`, `tp2race_tests.rs`, `agree16b_tests.rs`, `nway_liveness_tests.rs`.


pub mod bulk;

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
    fn net_set_probe(hold_ms: u32);
    /// ISSUES 153: the node registers its head control stream fd so the C bring-up retry loops
    /// abort the instant the head goes away (MSG_PEEK inside the shim — protocol bytes are
    /// never consumed). net_abort_check is the test/diagnostic probe of that liveness.
    fn net_set_abort_fd(fd: c_int);
    fn net_abort_check() -> c_int;
    fn net_init(rank: c_int, world: c_int, peer_ips: *const *const c_char, n_peers: c_int,
                tcp_port: c_int, dev_name: *const c_char,
                gid_idx: c_int, fp32_capacity_bytes: c_int, payload_bytes: c_int) -> *mut NetCtx;
    fn net_set_payload(c: *mut NetCtx, payload_bytes: c_int, fp32: c_int) -> c_int;
    /// TP-4F2: the NEXT net_init (world 4) builds a uniform one-shot ctx (consumed by that init).
    fn net_set_oneshot_uniform_once(on: c_int);
    /// TP-4F2: tie this ctx's abort to the primary link's (both ways, inside its proxy loop).
    fn net_set_parent(c: *mut NetCtx, parent: *mut NetCtx);
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
    /// S-B9-REL-TP2RACE (issue #10): the exchange generation counter, a monotone bump for the race
    /// probe's slot alignment, and the probe's withheld-slot clobber counter (must stay 0).
    fn net_xchg_gen(c: *mut NetCtx) -> u64;
    fn net_gen_add(c: *mut NetCtx, delta: u64) -> u64;
    fn net_probe_hits(c: *mut NetCtx) -> u64;
    fn net_device_epoch(c: *mut NetCtx) -> u64;
    fn net_gate_waits(c: *mut NetCtx) -> u64;
    fn net_wait_sleeps() -> u64;
    /// CPUIDLE: the proxy idle tier (--proxy-idle-us) and its counters (all proxies, process-wide).
    fn net_set_proxy_idle(us: u64);
    /// --proxy-owed-wait: 1 = park the owed wait in WFET (FEAT_WFxT), 0 = spin (the default).
    fn net_set_proxy_owed(wfet: c_int);
    fn net_proxy_parks() -> u64;
    fn net_proxy_park_ns() -> u64;
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
    fn net_exchange_one_dl(c: *mut NetCtx, peer_rank: c_int, nbytes: c_int, deadline_ns: u64) -> c_int;
    fn net_exchange_all_peers(c: *mut NetCtx, nbytes: c_int, deadline_ns: u64) -> c_int;
    fn net_agree_dl(c: *mut NetCtx, val: u64, step_mask: u64, step_val: u64, deadline_ns: u64) -> u64;
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

/// CPUIDLE: times a TP proxy entered its idle tier (WFE wait after --proxy-idle-us with nothing owed),
/// and the seconds it spent there. Process-wide; 0 parks with an idle server up = the tier never ran.
/// NIGHT1: the proxy idle tier's threshold when `--proxy-idle-us` is not given (us).
pub const PROXY_IDLE_US_DEFAULT: u64 = 300;
pub fn proxy_parks() -> u64 { unsafe { net_proxy_parks() } }
pub fn proxy_parked_secs() -> f64 { (unsafe { net_proxy_park_ns() }) as f64 / 1e9 }

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

/// The widest verify the lockstep token can name (its width field is five bits).
pub const AGREE_WIDTH_MAX: usize = 31;

/// A verify width as the token carries it, or None when it does not fit. S-B9-CF-AGREE16B (W5): this is
/// the ONLY narrowing of a width on its way into a lockstep word. The NVFP4 and DFlash call sites used to
/// write `width as u8` and leave the range check to `agree_ext`, which then saw the wrapped value: 256 + k
/// would have arrived as k and passed. Unreachable with today's widths (at most 16); structurally
/// impossible now, because the check sees the whole `usize` before any cast.
pub fn agree_width(k_verify: usize) -> Option<u8> {
    if k_verify <= AGREE_WIDTH_MAX { Some(k_verify as u8) } else { None }
}
const AGREE_DIGEST_BITS: u32 = 27;
// What this proves: the NVFP4 / DFlash verify widths (rows 1..=MAX_VERIFY) fit the field. The EXL3 widths
// are `drafts + 1 <= MTP_MAX_K + 1` and, for a shared plain step, the number of busy lanes: that one is
// bounded by `--max-batch`, which the EXL3 server checks against AGREE_WIDTH_MAX at boot (TP=2 only: the
// world > 2 frames carry whole u32 words). A wider verify needs a wider field, not a clamp.
const _: () = assert!(crate::gpu::MAX_VERIFY + 1 <= AGREE_WIDTH_MAX);

/// What the two lockstep fault drills do on THIS process (S-B9-CF-AGREE16B, W3). Pure: the callers pass
/// the stored option values, the pack kind and the TP position, and print `lines` at boot.
///
/// * `--tp-width-drill N` (EXL3 packs): ONE-SHOT. The first N-row round is reported as width
///   `N - 1 - rank` in the post-round lockstep word, then the rank tells the truth. It used to lie on
///   EVERY N-row round and report `N - 1` on every rank: a launcher that passed the flag to both boxes
///   made both say 15 for 16, which agrees, and a real 15-against-16 disagreement was masked for the
///   whole boot. The rank-dependent lie cannot cancel between ranks (two armed ranks report different
///   widths, so the round fails), and one shot bounds what a forgotten flag can hide to one round.
/// * `--tp-agree-drill S` (NVFP4 packs): this rank corrupts its agree hash at scheduler step S (one
///   step by nature); `drill_hash_flip` makes that corruption rank-dependent for the same reason.
///
/// A flag that cannot act where it was given (the other pack kind, TP=1, N <= rank) arms nothing and
/// says so: an INERT line, never silence.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct DrillPlan {
    /// EXL3: the row count whose first round is misreported (None = not armed)
    pub width: Option<usize>,
    /// NVFP4: the scheduler step whose hash is corrupted (None = not armed)
    pub hash_step: Option<u64>,
    /// one line per flag that is set: `DRILL ARMED ..` or `.. INERT ..`
    pub lines: Vec<String>,
}

/// `tp` = Some((rank, world)) under TP, None at TP=1. See `DrillPlan`.
pub fn drill_plan(agree_drill: Option<&str>, width_drill: Option<&str>, exl3: bool, tp: Option<(i32, i32)>) -> DrillPlan {
    let mut p = DrillPlan::default();
    let pack = if exl3 { "EXL3" } else { "NVFP4" };
    if let Some(v) = width_drill {
        let n = v.trim().parse::<usize>().ok().filter(|n| (1..=AGREE_WIDTH_MAX).contains(n));
        p.lines.push(match (n, tp) {
            (None, _) => format!("[tp-agree] --tp-width-drill '{v}' is INERT: not an integer from 1 to {AGREE_WIDTH_MAX}"),
            (Some(n), _) if !exl3 => format!("[tp-agree] --tp-width-drill {n} is INERT on an NVFP4 pack: it drills the EXL3 \
                                             lockstep word (the NVFP4 drill is --tp-agree-drill <step>)"),
            (Some(n), None) => format!("[tp-agree] --tp-width-drill {n} is INERT at TP=1: there is no lockstep to drill"),
            (Some(n), Some((rank, _))) if rank as usize >= n => format!(
                "[tp-agree] --tp-width-drill {n} is INERT on rank {rank}: this rank would report width N - 1 - rank, so it needs \
                 N > {rank}"),
            (Some(n), Some((rank, world))) => {
                p.width = Some(n);
                format!("[tp-agree] DRILL ARMED (--tp-width-drill {n}, rank {rank} of TP={world}): the FIRST {n}-row round on this \
                         rank will be REPORTED as width {} in its lockstep word, once; the guard must abort the link there. \
                         This boot is a fault drill, not a serving boot. It fires only if a {n}-row round occurs ({n} busy \
                         lanes sharing a plain step, or a {n}-row verify)", n - 1 - rank as usize)
            }
        });
    }
    if let Some(v) = agree_drill {
        let s = v.trim().parse::<u64>().ok();
        p.lines.push(match (s, tp) {
            (None, _) => format!("[tp-agree] --tp-agree-drill '{v}' is INERT: not a step number (an integer >= 0)"),
            (Some(s), _) if exl3 => format!("[tp-agree] --tp-agree-drill {s} is INERT on an EXL3 pack: it drills the NVFP4 agree \
                                            hash (the EXL3 drill is --tp-width-drill <rows>)"),
            (Some(s), None) => format!("[tp-agree] --tp-agree-drill {s} is INERT at TP=1: there is no lockstep to drill"),
            (Some(s), Some((rank, world))) => {
                p.hash_step = Some(s);
                format!("[tp-agree] DRILL ARMED (--tp-agree-drill {s}, rank {rank} of TP={world}, {pack} pack): this rank will \
                         corrupt its agree hash at scheduler step {s}, once; the guard must abort the link there. This boot \
                         is a fault drill, not a serving boot")
            }
        });
    }
    p
}

/// The NVFP4 drill's corruption of a rank's agree hash: different on every rank (bits 16.. carry
/// `rank + 1`), so two ranks that both carry `--tp-agree-drill S` still publish different hashes.
pub fn drill_hash_flip(rank: i32) -> u32 { 0xDEAD ^ ((((rank as u32) & 0xFF) + 1) << 16) }

/// The 32-bit low word of the lockstep token (S-B9-CF-AGREE16):
///
/// ```text
///    31        27 26                                   0
///   +------------+--------------------------------------+
///   | width  (5) |            digest27  (27)            |
///   +------------+--------------------------------------+
///   width    = the step's verify width, 0..=31, in bits of its own
///   digest27 = (hash ^ (hash >> 27)) & 0x07FF_FFFF: the 32-bit digest folded to 27 bits (its top five
///              bits land on bits 4..0, so every digest bit still reaches the word)
/// ```
///
/// Before this the word was `hash ^ ((width & 0xF) << 27)`: four bits, so 16 folded to 0 (and the EXL3
/// call sites saturated at 15 first: a 15-row and a 16-row round made the same width contribution, and
/// only the digests' own difference could tell them apart), and XOR-ed into bits the digest also sets,
/// so it was never a field: a digest difference in bits 27..30 could cancel a width difference. Now two
/// ranks whose widths differ ALWAYS send different words, whatever their digests hold; the price is a
/// 27-bit digest instead of a 32-bit one for a same-width content divergence (one chance in 134 million
/// per divergent round instead of one in 4 billion, and the next round tests the diverged state again).
/// Callers must pass a DIGEST as `hash` (the fold maps its top five bits onto bits 4..0): a structured
/// word whose top bits carry meaning has to be hashed first (`exl3_serve::go_fence_word`).
///
/// The token stays 64 bits and every frame keeps its size. A width above 31 is never clamped into the
/// field: `agree_ext` and `exl3_serve::tp_agree_fields` refuse it before a word is built, and this
/// function panics on one (in release builds too) instead of masking it.
pub fn agree_word(k_verify: u8, hash: u32) -> u32 {
    assert!(k_verify as usize <= AGREE_WIDTH_MAX, "verify width {k_verify} does not fit the token's 5-bit width field");
    ((k_verify as u32) << AGREE_DIGEST_BITS) | ((hash ^ (hash >> AGREE_DIGEST_BITS)) & ((1 << AGREE_DIGEST_BITS) - 1))
}

/// The whole 64-bit pairwise token: `[63..40] step field | [39..32] accept count | [31..0] agree_word`.
pub fn agree_token(step: u64, accept_count: u8, k_verify: u8, hash: u32) -> u64 {
    ((step & 0xFF_FFFF) << 40) | ((accept_count as u64) << 32) | agree_word(k_verify, hash) as u64
}

/// What a rank reads back from the peer's token: (accept count, agree_word).
pub fn agree_token_reply(token: u64) -> (u8, u32) {
    (((token >> 32) & 0xFF) as u8, (token & 0xFFFF_FFFF) as u32)
}

/// TP-D: the BOOT-time world-2 agree. The boot agree is the FIRST cross-rank rendezvous of a
/// session, immediately after two INDEPENDENT model loads (raw pack on the head, 50-400 s;
/// converted cache on a warm node, ~60-90 s) — legitimate skew the 10 s per-step watchdog
/// (TP_WDOG_NS) cannot cover: the 2026-10-30 Lygodactylus-4bpw and 2.05bpw TP boots aborted
/// as "peer token did not arrive within 10s" (code 7) while the head was still loading
/// weights, on a node whose cache was already warm. The deadline here is therefore 10 minutes;
/// world>2's hub boot agree is already unbounded for the same reason (see `exchange_u32s_all`).
/// A DEAD peer (vs merely slow) is still caught inside the wait by the liveness probe / abort
/// flag, so the longer deadline cannot hang a session whose peer is gone.
pub fn agree_boot(step: u64, hash: u32) -> Option<(u8, u32)> {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { return None; }
    const BOOT_AGREE_DEADLINE_NS: u64 = 600_000_000_000;
    let val = agree_token(step, 0, 0, hash);
    let ctx = c as *mut NetCtx;
    let got = unsafe {
        net_agree_dl(ctx, val, 0xFF_FFFF << 40, (step & 0xFF_FFFF) << 40, BOOT_AGREE_DEADLINE_NS)
    };
    if got == 0 { return None; }
    Some(agree_token_reply(got))
}

/// TP boot barrier (2026-11-09, runner Final_Run): every rank waits here, under the BOOT deadline, before the first
/// collective of a boot phase that follows independent per-rank work. The trunk's "measuring the speculation cost"
/// (MTP calibration) ran straight after per-rank setup whose length differs by box (the DFlash2 borrowed-head bf16 ->
/// NVFP4 quantization took 2.2 s on two ranks and 21-24 s on the others), so its first all-reduce sat under the 10 s
/// per-step watchdog and the fast ranks declared the slow ones dead (Qwen3.8-27B-FP8 + DF2 TP=4, Qwen3.5-122B-MIXED +
/// DF2 TP=4: link abort, first request 503). World 2: `agree_boot` (10-minute deadline); world > 2: the head hub,
/// unbounded (a DEAD peer is still caught by the liveness probe). No-op without a TP link.
pub fn boot_barrier(tag: u32) -> anyhow::Result<()> {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { return Ok(()); }
    let world = unsafe { net_world(c as *mut NetCtx) };
    if world <= 2 {
        match agree_boot(0xBB_0000 | (tag as u64 & 0xFFFF), tag) {
            Some((_, w)) if w == agree_word(0, tag) => Ok(()),
            Some((_, w)) => anyhow::bail!("TP boot barrier {tag:#x}: peer answered {w:#x}"),
            None => anyhow::bail!("TP boot barrier {tag:#x}: link aborted or the peer did not arrive within the boot deadline"),
        }
    } else {
        let all = exchange_u32s_all(&[0xB007_BA77, tag], 4)?;
        anyhow::ensure!(all.iter().all(|p| p.len() >= 2 && p[1] == tag), "TP boot barrier {tag:#x}: ranks at different barriers");
        Ok(())
    }
}

/// B8/G1 EXTENDED determinism token: `(step | k_verify | accept_count | hash_of_ids)`.
/// `k_verify` is this step's VERIFY WIDTH (the confidence-truncated bucket width under DSpark; the
/// plain chain width under MTP). Ranks that disagree on k_verify execute different barrier
/// sequences — the I9 silent-desync class — so it joins the per-step agreement token. The wire
/// shape stays the 64-bit token: see `agree_token` / `agree_word` for the layout (width 0..=31 in a field
/// of its own; a wider one is refused, never clamped).
/// MTP callers can keep the 3-arg `agree` (k_verify=0); speculative callers that choose a width MUST
/// use this one, and compare the returned word against `agree_word(k_verify, hash)`.
pub fn agree_ext(step: u64, accept_count: u8, k_verify: u8, hash: u32) -> Option<(u8, u32)> {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { return None; }
    if k_verify as usize > AGREE_WIDTH_MAX {
        // Never clamp a width into the field: two different widths would share one word. This rank aborts
        // its link here and returns None (every caller's "link aborted" arm). No token is published, so at
        // world 2 the peer leaves its own agree at the 10 s deadline (or at once, if it refused the same
        // width); at world > 2 the abort is fanned out to the other ranks by the proxy.
        eprintln!("[tp-agree] REFUSED at step field {step:#x}: verify width {k_verify} does not fit the token's 5-bit width \
                   field (max {AGREE_WIDTH_MAX}) — aborting instead of sending a width the peer could mistake for another");
        abort_link();
        return None;
    }
    let val = agree_token(step, accept_count, k_verify, hash);
    let ctx = c as *mut NetCtx;
    let world = unsafe { net_world(ctx) };
    if world <= 2 {
        // R1: pairwise path unchanged.
        let got = unsafe { net_agree(ctx, val, 0xFF_FFFF << 40, (step & 0xFF_FFFF) << 40) };
        if got == 0 { return None; }
        return Some(agree_token_reply(got));
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
            // Phase 2B: gather all peers in 1 parallel RTT instead of (world-1) serial RTTs.
            let rc1 = net_exchange_all_peers(ctx, AGREE_NBYTES as c_int, 0);
            if rc1 != 0 {
                eprintln!("[tp-agree] head round-1 exchange_all_peers rc={rc1} abort_status={:#018x} device_epoch={} gpu_ready={} rx_done={} — fanning round 2 anyway",
                          net_abort_status(ctx), net_device_epoch(ctx), net_gpu_ready(ctx), net_rx_done(ctx));
                round1_failed = true;
            } else {
                for r in 1..world {
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
            let rc2 = net_exchange_all_peers(ctx, AGREE_NBYTES as c_int, 0);
            if rc2 != 0 {
                eprintln!("[tp-agree] head round-2 exchange_all_peers rc={rc2} abort_status={:#018x} device_epoch={} gpu_ready={} rx_done={}",
                          net_abort_status(ctx), net_device_epoch(ctx), net_gpu_ready(ctx), net_rx_done(ctx));
                round1_failed = true;
            }
            if round1_failed || !all_equal { return None; }
            Some(agree_token_reply(val))
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
            Some(agree_token_reply(out))
        }
    }
}


/// Abort the registered TP link (cooperative stop: the abort STATUS word makes in-flight kernels
/// no-op through the stream rather than trapping, I9). No-op on a single-node run. Used by the
/// per-step agreement guard (TP item D) to take both ranks down together on a proven divergence.
pub fn abort_link() {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c != 0 { unsafe { net_abort(c as *mut NetCtx) } }
}

/// `abort_link`, except that an abort the link already recorded keeps its code: `net_abort` stores code 1
/// unconditionally, which would overwrite the diagnostic 10 (peer dead) / 12 (lockstep deadline) that a failed
/// hub exchange has just set. Used by the world > 2 lockstep paths only.
pub fn abort_link_keep_code() {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c != 0 && unsafe { net_abort_status(c as *mut NetCtx) } == 0 { unsafe { net_abort(c as *mut NetCtx) } }
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

// ---------------------------------------------------------------------------------
// S-B9-REL-TP2RACE (public issue #10): the world-2 pre-verify race probe + diagnostics.
//
// At world 2 the host exchanges now ride DEDICATED control slots (net_shim.c xchg_* helpers), so a
// control frame can no longer land on a doorbell recv slot that holds a validated-but-unconsumed
// epoch payload (the reporter's hypothesis; four v0.7.x head exits "TP pre-verify FAILED" with only
// the drafts hash differing). --tp-race-probe <ms> forces the window deterministically: the node's
// proxy holds every epoch release by <ms> (its K2 then consumes nothing), the head aligns its next
// exchange generation onto the withheld epoch's recv slot, and the proxy CLOBBER-detects any change
// of the withheld payload. The probe must fire on the old hot-ring transport and stay silent (and
// the pre-verify hashes agree) on the dedicated-slot one.

/// `--tp-race-probe <ms>` — 0/unset = off. Spmd scope: both ranks resolve it identically.
pub fn race_probe_hold_ms() -> u32 {
    static G: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        crate::opts::var(crate::opt!("tp-race-probe")).ok()
            .and_then(|v| v.trim().parse::<u32>().ok()).unwrap_or(0)
    });
    *G
}

/// The smallest delta >= 0 with `((gen + delta + 1) & 7) == want_next_slot` — the probe's
/// generation alignment. Pure: unit-tested. Both ranks compute the same delta at the same
/// lockstep point (same device epoch, same gen), so the exchange generations stay paired.
pub fn probe_gen_delta(gen: u64, want_next_slot: u64) -> u64 {
    (want_next_slot.wrapping_add(8).wrapping_sub((gen.wrapping_add(1)) & 7)) & 7
}

/// The registered link's exchange generation (for the pre-verify bail diagnostic: `gen & 7` next to
/// `device_epoch & 7`). 0 when no link.
pub fn traced_xchg_gen() -> u64 {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { 0 } else { unsafe { net_xchg_gen(c as *mut NetCtx) } }
}

/// Monotone bump of the link's exchange generation by `delta` (returns the new gen). Probe-only.
pub fn traced_gen_add(delta: u64) -> u64 {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { 0 } else { unsafe { net_gen_add(c as *mut NetCtx, delta) } }
}

/// The probe's withheld-slot clobber count (0 = clean; any hit = the race fired).
pub fn traced_probe_hits() -> u64 {
    let c = TRACE_CTX.load(std::sync::atomic::Ordering::Relaxed);
    if c == 0 { 0 } else { unsafe { net_probe_hits(c as *mut NetCtx) } }
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

/// TP-4E: the device epoch of an AUXILIARY link's ctx (the rail-2 twin of `traced_device_epoch`, which reads
/// only the registered agree/trace link). Same reader `TpLink::device_epoch` uses, by raw ctx address.
pub fn ctx_device_epoch(ctx_addr: usize) -> u64 { unsafe { net_device_epoch(ctx_addr as *mut NetCtx) } }

/// TP-4E: every TCP port an N-way bring-up with control port `tcp_port` may bind or dial at `world`:
/// the liveness responder (`tcp_port + 1`) and the per-pair handshake ports. A mirror of the C shim's
/// `nway_pair_port` (native/net_shim.c: `tcp_port + 2 + lo * world + hi`, lo < hi) — keep the two in step;
/// `nway_pair_port_mirrors_the_shim_formula` pins the Rust side to the documented formula, and the
/// hardware gate (TP-4E_GATES.md) checks the live listeners.
pub fn nway_link_ports(tcp_port: u16, world: i32) -> Vec<u32> {
    let mut v = vec![tcp_port as u32 + 1];
    for lo in 0..world {
        for hi in lo + 1..world {
            v.push(tcp_port as u32 + 2 + (lo * world + hi) as u32);
        }
    }
    v
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
    let probe: u32 = opts::var(opt!("tp-race-probe")).ok()
        .and_then(|v| v.trim().parse::<u32>().ok()).unwrap_or(0);
    // CPUIDLE: --proxy-idle-us. NIGHT1 (2026-11-16, owner's overheating plan + jetspark's report that the proxy held its
    // core through the whole generation): default 300 us, so the proxy also parks in the gaps INSIDE a request, not only
    // on an idle server. TP=4 Flash-Next 3.05bpw greedy decode, 4 boots ABAB: 300 us 109.3 / 110.2 tok/s vs 20000 us
    // 110.3 / 110.3 (no cost); ~10,400 parks per run instead of ~180. 0 = the debt-free idle tier is off (the pre-CPUIDLE behaviour; --proxy-owed-wait is separate).
    let idle_us: u64 = opts::var(opt!("proxy-idle-us")).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(PROXY_IDLE_US_DEFAULT);
    unsafe { net_set_proxy_idle(idle_us) }
    // --proxy-owed-wait (opt-in): park the proxy while a posted epoch is unmatched (see native/net_shim, TP_OWED_SPIN_NS).
    let owed_wfet = opts::var(opt!("proxy-owed-wait")).map_or(false, |v| v.trim() == "wfet") as c_int;
    unsafe { net_set_proxy_owed(owed_wfet) }
    unsafe { net_set_opts(spin, tail, oneshot, diag); net_set_probe(probe);
              net_set_abort_fd(HEAD_LIVENESS_FD.load(std::sync::atomic::Ordering::Relaxed)) }
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

    /// TP-4F2 (`--tp-reduce single`): the dedicated single-stage reduce link. An ordinary world-4 N-way bring-up
    /// whose ctx is a UNIFORM one-shot ctx — every epoch is the all-peers push into sender-indexed rings, the proxy
    /// releases all three peers together, K1's reuse gate waits on all three QPs. It has its own epoch counter, so
    /// the primary link's R10 round phase never sees it. `slot_bytes` = the ring-slot capacity (decode-sized
    /// reduces only: XPORT_MIN_BYTES). World 4 only (the shim ignores the request elsewhere, which is refused here).
    pub fn connect_nway_uniform(rank: i32, world: i32, peer_ips: &[IpAddr], tcp_port: u16, dev: &str,
                                gid_idx: i32, slot_bytes: usize) -> anyhow::Result<Self> {
        anyhow::ensure!((3..=8).contains(&world), "the single-stage reduce ctx requires world in 3..=8 (got {world})");
        unsafe { net_set_oneshot_uniform_once(1) };
        let r = Self::connect_nway(rank, world, peer_ips, tcp_port, dev, gid_idx, slot_bytes);
        unsafe { net_set_oneshot_uniform_once(0) };     // never leak the request past this bring-up
        let l = r?;
        anyhow::ensure!(l.oneshot_on(), "the single-stage reduce ctx did not come up one-shot (shim/option mismatch)");
        Ok(l)
    }

    /// TP-4F2: mirror this link's abort into `parent` (the primary link's NetCtx address) and the parent's into
    /// this one, inside this link's proxy loop. MUST be called before the proxy thread starts.
    pub fn set_parent(&mut self, parent_ctx_addr: usize) {
        unsafe { net_set_parent(self.ctx, parent_ctx_addr as *mut NetCtx) }
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

mod hub;
pub use self::hub::*;

/// In-memory `HubXport` for the CPU tests: one thread per rank over a shared mailbox that mirrors
/// `net_exchange_one` exactly where it matters — the per-generation tail tag `(g << 8) | sender` stamped in the
/// last 8 bytes of the frame, per-(sender, receiver) ring slots of depth `TP_CTRL_RING` written in place by the
/// post, the wait that compares ONE tag word at `nbytes - 8` (tag check first, abort second, exactly the C
/// order), the copy to a stable "last received" slot, the optional retirement of the consumed frame
/// (`clear_frame`, the TP-4D net_shim.c change), a per-rank abort status (each process has its own ctx), the
/// dead-peer probe (a rank whose thread has ended counts as a dead process after `probe_after` of silence: abort
/// code 10) and the lockstep deadline (abort code 12). It does NOT model RDMA placement ordering, NIC retries,
/// the proxy, the hot-path rings or the GPU — those are hardware questions this model cannot answer.
#[cfg(test)]
pub(crate) mod hub_mock;

#[cfg(test)]
mod hub_tests;

#[cfg(test)]
mod tp2race_tests;

/// S-B9-CF-AGREE16B: the checked width narrowing (W5) and the drill plan (W3).
#[cfg(test)]
mod agree16b_tests;

// ---------------------------------------------------------------------------
// ISSUES 153 (2026-10-27): liveness-tied TP bring-up.
//
// A cold-cache head ships its world-1 ranks SEQUENTIALLY (~0.10 GB/s, minutes per rank) and
// opens its per-pair RDMA listeners only afterwards, so the shim's old fixed retry budgets
// (300 s dial / 600 s accept) expired mid-ship and every node died with
// "nway connect: Connection refused" on the FIRST cold boot (a warm blob cache hides it: the
// whole sync takes seconds). The dial/accept loops now retry while the head's control stream
// is alive and give up the moment it closes/resets, with a 1 h hard cap.
// ---------------------------------------------------------------------------

/// fd of the head's control stream on a node process, peeked by the C bring-up loops. -1 on
/// the head and on bench paths that drop the stream before bring-up: no abort source, the
/// retry loops fall back to the cap alone.
static HEAD_LIVENESS_FD: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

/// Node serve paths call this BEFORE `bring_up_node` — the stream must outlive the bring-up
/// (it always does: it is the session's control channel).
pub fn set_head_liveness(stream: &std::net::TcpStream) {
    use std::os::unix::io::AsRawFd;
    let fd = stream.as_raw_fd();
    HEAD_LIVENESS_FD.store(fd, std::sync::atomic::Ordering::Relaxed);
    // Push now (the loops can start before the next push_net_opts) — idempotent with it.
    unsafe { net_set_abort_fd(fd) }
}

/// Clear a registration when the owning stream is about to be dropped (bench paths), so a
/// recycled fd number can never be mistaken for the head's liveness.
pub fn clear_head_liveness() {
    HEAD_LIVENESS_FD.store(-1, std::sync::atomic::Ordering::Relaxed);
    unsafe { net_set_abort_fd(-1) }
}

#[cfg(test)]
mod nway_liveness_tests;
