//! TP-F — the world==2 fp32 all-reduce LAUNCH SCHEDULES over the doorbell transport
//! (native/tp_doorbell.h, kernels/gpu_batch.cu), shared by the EXL3 TP engine (exl3_forward/xtp.rs)
//! and the transport bench (`--tp-reduce-bench`, tp_bench.rs) so the bench measures the served code.
//!
//! * `reduce_serial`  — the TP-B..E path: single-block K1 `tp_gate_copy_signal` + K2 `tp_wait_add`
//!                      (mode 2), one 1 MiB epoch at a time, K1(e) -> K2(e) -> K1(e+1).
//! * `reduce_pipe`    — TP-F: multi-block K1m `tp_gate_copy_signal_mb` + K2m `tp_wait_add_mb`, up to
//!                      `lookahead` epochs in flight PER RAIL, epochs dealt round-robin over 1..2 rails
//!                      (each rail = its own NetCtx/QP/ring/proxy on its own ConnectX-7 port).
//!
//! Both produce, per element, lower-rank partial + upper-rank partial in fp32 (one IEEE add): the
//! result is bitwise independent of the schedule, the chunking and the rail deal.

use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// FP32 elements per epoch: one 1 MiB ring slot minus the tail guard, rounded down to 256.
pub const EPOCH_FLOATS: usize = ((crate::tp::TP_SLOT_BYTES - 64) / 4) / 256 * 256;
/// native/tp_doorbell.h TP_RING_SLOTS (R). The pipelined lookahead per rail is capped at R/2.
pub const RING_SLOTS: usize = 8;

pub struct Kernels {
    k1: CudaFunction,
    k2: CudaFunction,
    k1m: CudaFunction,
    k2m: CudaFunction,
    /// TP-SP: the all-gather's receive side (the K2m gate + a plain copy, no arithmetic)
    k2c: CudaFunction,
}

pub const KERNEL_NAMES: [&str; 5] = ["tp_gate_copy_signal", "tp_wait_add", "tp_gate_copy_signal_mb", "tp_wait_add_mb",
                                     "tp_wait_copy_mb"];

impl Kernels {
    /// Load (once per module name) the five doorbell kernels from src/ptx/gpu_batch.ptx.
    pub fn load(dev: &Arc<CudaDevice>, module: &str) -> Result<Self> {
        if dev.get_func(module, "tp_gate_copy_signal_mb").is_none() {
            let ptx = cudarc::nvrtc::Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_batch.ptx")
                .context("src/ptx/gpu_batch.ptx missing — the doorbell kernels live there")?);
            dev.load_ptx(ptx, module, &KERNEL_NAMES).context("load the TP doorbell kernels (gpu_batch.ptx)")?;
        }
        let f = |n: &str| dev.get_func(module, n).with_context(|| format!("doorbell kernel {n} missing"));
        Ok(Kernels { k1: f(KERNEL_NAMES[0])?, k2: f(KERNEL_NAMES[1])?, k1m: f(KERNEL_NAMES[2])?, k2m: f(KERNEL_NAMES[3])?,
                     k2c: f(KERNEL_NAMES[4])? })
    }
}

/// The pipelined transport's state: the rails' device ctx pointers (rail 0 = the primary link that
/// also carries decode barriers and agrees) + K1m's arrive counter + geometry + the bubble scratch.
pub struct Pipe {
    pub rails: Vec<u64>,
    arrive: CudaSlice<u32>,
    /// the recursive-doubling round count log2(world) (1 at world 2)
    pub rounds: usize,
    /// epochs in flight per rail (G): <= R/2, and a multiple of `rounds` (the world>2 slot rule)
    pub lookahead: usize,
    pub blocks: u32,
    /// world>2 pipeline-fill bubbles: a 16 B send/receive scratch (4 floats)
    bubble: CudaSlice<f32>,
    /// TP-I2 item 1 v2: the transport's own BLOCKING stream (AGENTS §2.1: never a NonBlocking fork)
    /// + the two handoff events, for the dual-stream rows-landed hook (`RowHook::side`)
    side: Side,
}

/// TP-I2 item 1 v2: a second blocking stream for the K1m/K2m schedule and its handoff events
/// (`ev_in`: compute -> side, the producer is done; `ev_out`: side -> compute, rows are final).
/// cuStreamWaitEvent snapshots the event's latest record, so each event is re-recorded freely.
struct Side {
    stream: CudaStream,
    ev_in: u64,
    ev_out: u64,
}
// the raw CUevent handles are process-global driver objects; every use is on the launching thread
unsafe impl Send for Side {}
unsafe impl Sync for Side {}

impl Side {
    fn new(dev: &Arc<CudaDevice>) -> Result<Self> {
        let stream = crate::gpu::fork_blocking_stream(dev);
        let mk = || -> Result<u64> {
            let mut ev: cudarc::driver::sys::CUevent = std::ptr::null_mut();
            // CU_EVENT_DISABLE_TIMING (2): a pure ordering event
            let r = unsafe { cudarc::driver::sys::cuEventCreate(&mut ev, 2) };
            anyhow::ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS && !ev.is_null(), "cuEventCreate ({r:?})");
            Ok(ev as u64)
        };
        Ok(Side { stream, ev_in: mk()?, ev_out: mk()? })
    }
}

/// `a` (stream) waits for everything enqueued on `b` so far (record `ev` on b, a waits on it).
fn stream_after(a: &CudaStream, b: &CudaStream, ev: u64) -> Result<()> {
    use cudarc::driver::sys;
    let e = ev as sys::CUevent;
    let r = unsafe { sys::cuEventRecord(e, b.stream) };
    anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuEventRecord ({r:?})");
    let r = unsafe { sys::cuStreamWaitEvent(a.stream, e, 0) };
    anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuStreamWaitEvent ({r:?})");
    Ok(())
}

impl Pipe {
    pub fn new(dev: &Arc<CudaDevice>, rails: Vec<u64>, world: usize, lookahead: usize, blocks: u32) -> Result<Self> {
        anyhow::ensure!(!rails.is_empty() && rails.len() <= 2, "1..2 rails (got {})", rails.len());
        anyhow::ensure!(world >= 2 && world.is_power_of_two(), "world {world} must be a power of two >= 2");
        let rounds = world.trailing_zeros() as usize;
        let mut arrive = dev.alloc_zeros::<u32>(1)?;
        dev.memset_zeros(&mut arrive)?;   // AGENTS §2.2: alloc_zeros does not zero
        let mut bubble = dev.alloc_zeros::<f32>(4)?;
        dev.memset_zeros(&mut bubble)?;
        dev.synchronize()?;
        // G <= R/2 (recv-slot rule) and a multiple of rounds (world>2: round(x+R-G) == round(x)).
        let mut g = lookahead.clamp(1, RING_SLOTS / 2);
        g = (g / rounds).max(1) * rounds;
        anyhow::ensure!(g <= RING_SLOTS / 2, "world {world}: no lookahead satisfies G <= R/2 with G % rounds == 0");
        let side = Side::new(dev)?;
        Ok(Pipe { rails, arrive, rounds, lookahead: g, blocks: blocks.clamp(1, 128), bubble, side })
    }
}

/// Split n floats into ring-slot epochs: (float offset, len).
fn epochs(n: usize) -> Vec<(usize, usize)> { epochs_of(n, EPOCH_FLOATS) }

fn epochs_of(n: usize, per: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::with_capacity(n / per + 1);
    let mut off = 0usize;
    while off < n {
        let len = (n - off).min(per);
        v.push((off, len));
        off += len;
    }
    v
}

/// In-place fp32 sum all-reduce of `n` floats at device address `p`, serial single-block pair.
pub fn reduce_serial(k: &Kernels, stream: &CudaStream, ctx: u64, p: u64, n: usize) -> Result<()> {
    let c1 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
    for (off, len) in epochs(n) {
        anyhow::ensure!(len * 4 + 64 <= crate::tp::TP_SLOT_BYTES, "all-reduce epoch {len} floats exceeds one ring slot");
        let q = p + (off * 4) as u64;
        unsafe {
            k.k1.clone().launch_on_stream(stream, c1, (ctx, q, (len * 4) as u32)).context("K1 launch")?;
            k.k2.clone().launch_on_stream(stream, c1, (ctx, q, q, len as i32, 2i32)).context("K2 launch")?;
        }
    }
    Ok(())
}

/// TP-G bench helpers: one single-block K1 / K2 (mode 2, in place) of a <= one-slot reduce.
pub fn k1_serial(k: &Kernels, stream: &CudaStream, ctx: u64, p: u64, n: usize) -> Result<()> {
    let c1 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
    anyhow::ensure!(n <= EPOCH_FLOATS, "k1_serial: one epoch only");
    unsafe { k.k1.clone().launch_on_stream(stream, c1, (ctx, p, (n * 4) as u32)) }.context("K1 launch")
}
pub fn k2_serial(k: &Kernels, stream: &CudaStream, ctx: u64, p: u64, n: usize) -> Result<()> {
    let c1 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
    anyhow::ensure!(n <= EPOCH_FLOATS, "k2_serial: one epoch only");
    unsafe { k.k2.clone().launch_on_stream(stream, c1, (ctx, p, p, n as i32, 2i32)) }.context("K2 launch")
}

/// True when `reduce_pipe` can carry this payload (every epoch a 16 B multiple).
pub fn pipe_ok(n: usize) -> bool { n % 4 == 0 && EPOCH_FLOATS % 4 == 0 }

/// One epoch of a rail's pipelined sequence: a chunk (float offset, len) in round k, or a fill
/// bubble (world>2 only: every device epoch has a fixed round = e % rounds, so a round slot with no
/// chunk ready still ships a minimal 16 B epoch to keep the round schedule aligned).
#[derive(Clone, Copy)]
enum Ep { Chunk(usize, usize), Bubble }

/// A rail's epoch sequence for its chunks under recursive doubling with `rounds` rounds: at step s,
/// epoch s*rounds + k carries round k of chunk s - k*d (bubble when out of range). Round k of a chunk
/// reads round k-1's in-place result, so its K1m must follow round k-1's K2m in stream order; with
/// K1m(j) issued right after K2m(j-G) that holds iff d*rounds + 1 >= G. rounds == 1 (world 2): the
/// chunks in order, no bubbles.
fn rail_seq(chunks: &[(usize, usize)], rounds: usize, g: usize) -> Vec<Ep> {
    if rounds == 1 { return chunks.iter().map(|&(o, l)| Ep::Chunk(o, l)).collect(); }
    let d = (g.saturating_sub(1) + rounds - 1) / rounds;
    let c = chunks.len();
    let steps = c + (rounds - 1) * d;
    let mut v = Vec::with_capacity(steps * rounds);
    for st in 0..steps {
        for k in 0..rounds {
            let ci = st as isize - (k * d) as isize;
            v.push(if ci >= 0 && (ci as usize) < c { let (o, l) = chunks[ci as usize]; Ep::Chunk(o, l) } else { Ep::Bubble });
        }
    }
    v
}

/// Where a pipelined reduce reads its local partial and writes the sum. The wire is always fp32.
/// `src_f16`: the local partial is f16 (K1m widens it on the copy, K2m on the add — exact);
/// `out_f16`: the sum is rounded once to f16 (__float2half_rn, as xq_cvt_f32_f16). src == out is
/// allowed (same-thread read-then-write in K2m).
/// `wire_f16` (requires `src_f16`): the partial is already f16-exact, so the wire carries f16 —
/// half the bytes, bitwise the same fp32 sum (the peer widens exactly).
#[derive(Clone, Copy)]
pub struct Io { pub src: u64, pub src_f16: bool, pub out: u64, pub out_f16: bool, pub wire_f16: bool }

impl Io {
    /// The plain in-place fp32 reduce.
    pub fn f32_inplace(p: u64) -> Io { Io { src: p, src_f16: false, out: p, out_f16: false, wire_f16: false } }
    fn wire16(&self) -> bool { self.wire_f16 && self.src_f16 }
    fn bits(&self) -> i32 { (self.src_f16 as i32) | ((self.out_f16 as i32) << 1) | ((self.wire16() as i32) << 2) }
}

/// The pipelined, rail-dealt schedule: chunk c rides rail c % nr; each rail runs its own epoch
/// sequence (`rail_seq`) with G = lookahead epochs in flight. Launch program, tick t = 0, 1, ...:
/// for each rail, K2m(t-G) [lag = that rail's K1m launched after it] then K1m(t). So on every rail
/// K1m(j) is stream-ordered behind K2m(j-G) (the recv-slot rule, gpu_batch.cu), each rail keeps its
/// own device epoch counter, and every chunk's value is the canonical fp32 sum.
pub fn reduce_pipe(k: &Kernels, stream: &CudaStream, x: &Pipe, io: Io, n: usize) -> Result<()> {
    reduce_pipe_hooked(k, stream, x, io, n, None)
}

/// TP-I2 item 1: a consumer of the reduced rows, launched on the SAME stream as rows become final.
/// `row_len` elements per row (n % row_len == 0); `min_rows` = the hook granularity; `f(r0, r1)`
/// launches per-row consumer kernels over rows [r0, r1). It may read only those rows of the reduce
/// output and must not write the reduce's source or output buffers.
pub struct RowHook<'a> {
    pub row_len: usize,
    pub min_rows: usize,
    pub f: &'a mut dyn FnMut(usize, usize) -> Result<()>,
    /// v2 (dual stream): the K1m/K2m schedule runs on the Pipe's own blocking stream, ordered after
    /// everything already on the caller's stream (the producer); each hook range is handed to the
    /// caller's stream by an event recorded after the K2m's that finish it, and the caller's stream
    /// joins the whole schedule at the end. The hook's kernels then run CONCURRENTLY with the K1m/K2m
    /// of later epochs (the in-model reduce is bound by K1m/K2m execution, not by wire wait — the v1
    /// same-stream hook could only fill the wait). false = v1: everything on the caller's stream.
    pub side: bool,
    /// TP-I3 (c1) FOLD: every epoch carries whole rows (the epoch length is rounded down to a multiple of
    /// `row_len`), and each epoch's K2m is REPLACED by `fold(ctx, lag, r0, r1)` — a fused consumer launched
    /// at K2m's stream position with K2m's (ctx, lag), which gates on the epoch, forms the canonical sum of
    /// rows [r0, r1) itself and consumes it (bitwise the K2m + consumer pair). Single stream (`side` and `f`
    /// are not used). World 2 only (rounds == 1).
    pub fold: Option<&'a mut dyn FnMut(u64, u32, usize, usize) -> Result<()>>,
}

/// `reduce_pipe` with an optional rows-landed hook (TP-I2 item 1: comm/compute overlap by row
/// sub-chunks). The schedule and every K1m/K2m launch are exactly `reduce_pipe`'s. The hook's
/// launches go in at the END of a tick (after both rails' K2m(t-G) and K1m(t)), once the K2m's
/// launched so far have summed at least `min_rows` new whole rows. At that point G epochs per rail
/// are published and in flight, so the wire moves them while the GPU runs the hook instead of
/// spinning in the next K2m. A row is handed over only when every epoch covering it has had its K2m
/// launched earlier in stream order, so every value the hook reads is the finished canonical fp32
/// sum: the reduce output and the consumer's output are bitwise `reduce_pipe` + the consumer over
/// all rows. World > 2 (rounds > 1): a chunk is final only after its last round, so the hook runs
/// once over all rows after the whole schedule (no overlap, same bits).
pub fn reduce_pipe_hooked(k: &Kernels, stream: &CudaStream, x: &Pipe, io: Io, n: usize,
                          mut hook: Option<RowHook<'_>>) -> Result<()> {
    let caller = stream;
    anyhow::ensure!(pipe_ok(n), "pipelined all-reduce of {n} floats: not a 16 B multiple");
    if let Some(h) = hook.as_ref() {
        anyhow::ensure!(h.row_len > 0 && n % h.row_len == 0 && h.min_rows > 0,
            "reduce hook: n {n} is not a whole number of {}-element rows (min_rows {})", h.row_len, h.min_rows);
    }
    // f16 in/out is single-exchange only: with rounds > 1 an f16 in-place buffer would round the
    // intermediate round sums (output-changing). Refuse rather than change numerics.
    anyhow::ensure!(x.rounds == 1 || !(io.src_f16 || io.out_f16 || io.wire_f16),
        "f16 reduce IO needs an fp32 intermediate at world > 2 (rounds {})", x.rounds);
    // K1m copies the f16 wire in 16 B vectors: an f16 wire needs every epoch a multiple of 8 elements
    let io = if io.wire16() && n % 8 != 0 { Io { wire_f16: false, ..io } } else { io };
    // an f16-wire epoch carries twice the elements in the same slot bytes
    let per = if io.wire16() { 2 * EPOCH_FLOATS } else { EPOCH_FLOATS };
    // TP-I3 (c1) fold: whole rows per epoch (the canonical per-element sum does not depend on the chunking)
    let fold_on = hook.as_ref().map_or(false, |h| h.fold.is_some());
    let per = if fold_on {
        let h = hook.as_ref().unwrap();
        anyhow::ensure!(x.rounds == 1, "prefill fold: world 2 only (rounds {})", x.rounds);
        anyhow::ensure!(h.row_len <= per && h.row_len % 8 == 0, "prefill fold: row_len {} vs epoch {per}", h.row_len);
        fold_per(per, h.row_len)
    } else { per };
    let ep = epochs_of(n, per);
    let nr = x.rails.len().min(ep.len().max(1));
    let g = x.lookahead;
    let seqs: Vec<Vec<Ep>> = (0..nr).map(|r| {
        let mine: Vec<(usize, usize)> = ep.iter().enumerate().filter(|(i, _)| i % nr == r).map(|(_, &c)| c).collect();
        rail_seq(&mine, x.rounds, g)
    }).collect();
    let cfg = LaunchConfig { grid_dim: (x.blocks, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
    let arrive = *x.arrive.device_ptr() as u64;
    let bub = *x.bubble.device_ptr() as u64;
    let (se, oe) = (if io.src_f16 { 2u64 } else { 4 }, if io.out_f16 { 2u64 } else { 4 });
    // (src, out, len, io bits) of an epoch
    let span = |e: &Ep| -> (u64, u64, usize, i32) {
        match *e {
            Ep::Chunk(o, l) => (io.src + o as u64 * se, io.out + o as u64 * oe, l, io.bits()),
            Ep::Bubble => (bub, bub, 4, 0),
        }
    };
    let longest = seqs.iter().map(|s| s.len()).max().unwrap_or(0);
    // TP-I2 item 1: rows handed to the hook so far (rounds == 1: every epoch is final after its K2m)
    let mut rows_done = 0usize;
    let per_tick = x.rounds == 1;
    // v2: the schedule on the side stream, after the producer (everything already on `stream`)
    let dual = !fold_on && hook.as_ref().map_or(false, |h| h.side);
    let comm: &CudaStream = if dual { &x.side.stream } else { stream };
    if dual { stream_after(comm, stream, x.side.ev_in)?; }
    let stream = comm;
    for t in 0..longest + g {
        for (r, sq) in seqs.iter().enumerate() {
            let ctx = x.rails[r];
            if t >= g && t - g < sq.len() {
                let j = t - g;
                let (sp, op, len, bits) = span(&sq[j]);
                let lag = (t.min(sq.len()) - 1 - j) as u32;       // this rail's K1m launched after j
                if fold_on {
                    // TP-I3 (c1): the fused consumer of this epoch's whole rows, at K2m's position
                    let Ep::Chunk(o, l) = sq[j] else { anyhow::bail!("prefill fold: bubble epoch (world > 2)") };
                    let rl = hook.as_ref().unwrap().row_len;
                    let f = hook.as_mut().unwrap().fold.as_mut().unwrap();
                    f(ctx, lag, o / rl, (o + l) / rl)?;
                } else {
                    unsafe { k.k2m.clone().launch_on_stream(stream, cfg, (ctx, op, sp, len as i32, lag, bits)) }
                        .context("K2m launch")?;
                }
            }
            if t < sq.len() {
                let (sp, _, len, bits) = span(&sq[t]);
                let wb = if bits & 4 != 0 { 2 } else { 4 };      // wire bytes per element
                unsafe { k.k1m.clone().launch_on_stream(stream, cfg, (ctx, sp, (len * wb) as u32, arrive, bits & 5)) }
                    .context("K1m launch")?;
            }
        }
        if let (true, Some(h)) = (per_tick && t >= g && !fold_on, hook.as_mut()) {
            if let Some(rows) = hook_rows_after_tick(&ep, n, nr, g, t, h.row_len, h.min_rows, rows_done) {
                // v2: the caller's stream waits for the K2m's launched so far (rows [.., rows) final)
                if dual { stream_after(caller, comm, x.side.ev_out)?; }
                (h.f)(rows_done, rows)?;
                rows_done = rows;
            }
        }
    }
    // v2: the caller's stream joins the whole schedule (every K1m/K2m) before anything after the reduce
    if dual { stream_after(caller, comm, x.side.ev_out)?; }
    if let Some(h) = hook.as_mut() {
        let rows = n / h.row_len;
        if rows > rows_done && !fold_on { (h.f)(rows_done, rows)?; }
    }
    Ok(())
}

/// TP-I3 (c1): the fold schedule's epoch length — the transport epoch rounded down to whole rows.
fn fold_per(per: usize, row_len: usize) -> usize { (per / row_len) * row_len }

/// TP-I2 item 1: the rows a world-2 (rounds == 1) pipelined reduce may hand to its hook after tick
/// `t` (Some(r1): hand over [rows_done, r1)). Global epoch i rides rail i % nr at sequence index
/// i / nr and tick t launches every rail's K2m of index t - g, so after tick t every global epoch
/// < nr * (t - g + 1) has had its K2m launched; the finished elements are that prefix, the finished
/// rows its whole rows. A range is handed over once it holds >= min_rows rows, or at the end.
#[allow(clippy::too_many_arguments)]
fn hook_rows_after_tick(ep: &[(usize, usize)], n: usize, nr: usize, g: usize, t: usize, row_len: usize,
                        min_rows: usize, rows_done: usize) -> Option<usize> {
    if t < g { return None; }
    let fin = (nr * (t - g + 1)).min(ep.len());
    let elems = if fin == ep.len() { n } else { ep[fin].0 };
    let rows = elems / row_len;
    (rows > rows_done && (rows - rows_done >= min_rows || elems == n)).then_some(rows)
}

// =================================================================================================
// TP-SP (sequence-parallel prefill; PLAN/notes_2026-09-28/TP-I3.md §3.2, TP-SP1): row-block
// REDUCE-SCATTER and ALL-GATHER over the same doorbell protocol (same K1m, same rings, same len/tail
// tags, same proxies, same recv-slot rule), world 2 built. Rank r owns rows row_block(c, r, W) of a
// chunk; every exchange is a list of whole-row epochs:
//   reduce-scatter: epoch j ships PIECE j OF THE PEER'S ROWS of this rank's partial (K1m) and consumes
//                   the peer's epoch j = piece j of THIS rank's rows of the peer's partial (K2m: the
//                   canonical lower + upper fp32 sum into the own rows, bitwise the all-reduce's value
//                   for those rows: the per-element sum does not depend on the chunking);
//   all-gather:     epoch j ships piece j of the OWN rows (K1m, a plain byte copy) and the copy-K2
//                   (tp_wait_copy_mb: the K2m gate + a plain copy, NO add — a K2m with a zero local
//                   partial would turn -0.0 into +0.0) lands the peer's piece j in the peer's rows.
// Both ranks split each block into the SAME number E of pieces (E = the larger block's epoch count,
// pieces as even as whole rows allow), so rank a's send piece j IS rank b's receive piece j (SPMD:
// a pure function of (c, W)). A rank's per-epoch LENGTHS may differ from its peer's (odd c): at world
// 2 the proxy sizes each RDMA write from K1m's len tag and K2m never reads the peer's length.
// =================================================================================================

/// The row block [lo, hi) of `rank` in a chunk of `c` rows at world `world`: blocks of ceil(c / W)
/// rows (rank 0 first). World-general; the schedules below are built for world 2.
pub fn row_block(c: usize, rank: usize, world: usize) -> (usize, usize) {
    let cb = c.div_ceil(world.max(1));
    ((rank * cb).min(c), ((rank + 1) * cb).min(c))
}

/// Whole rows of `row_bytes` bytes that fit one ring slot's payload (EPOCH_FLOATS * 4 bytes).
fn rows_per_epoch(row_bytes: usize) -> usize { (EPOCH_FLOATS * 4) / row_bytes.max(1) }

/// The `e` whole-row pieces of rows [lo, hi): piece i = [lo + L*i/e, lo + L*(i+1)/e), in order.
fn row_pieces(lo: usize, hi: usize, e: usize) -> Vec<(usize, usize)> {
    let l = hi - lo;
    (0..e).map(|i| (lo + l * i / e, lo + l * (i + 1) / e)).collect()
}

/// The epoch count both ranks use to exchange `blocks` at `rpe` rows per epoch (>= 1).
fn pieces_count(blocks: &[(usize, usize)], rpe: usize) -> usize {
    blocks.iter().map(|&(a, b)| (b - a).div_ceil(rpe.max(1))).max().unwrap_or(1).max(1)
}

/// One epoch of a row exchange: what K1m ships and what the receive kernel does with the peer's epoch.
#[derive(Clone, Copy, Debug)]
enum Rx {
    /// K2m: out[0..n) = canonical sum of `local[0..n)` and the peer's payload (io bits)
    Add { out: u64, local: u64, n: usize, bits: i32 },
    /// copy-K2: out[0..bytes) = the peer's payload bytes
    Copy { out: u64, bytes: usize },
}
#[derive(Clone, Copy, Debug)]
struct XEp {
    src: u64,
    bytes: usize,
    k1bits: i32,
    rx: Rx,
    /// own rows this epoch's receive finishes (the reduce-scatter hook's unit)
    rows: usize,
}

/// A rows-landed hook of a reduce-scatter (TP-I2's dual overlap on the OWN rows): `f(r0, r1)` launches
/// the per-row consumer over ABSOLUTE rows [r0, r1) of the own block; `side` = the K1m/K2m schedule on
/// the Pipe's own blocking stream (the served dual:ROWS posture), false = on the caller's stream.
pub struct SpHook<'a> {
    pub min_rows: usize,
    pub side: bool,
    pub f: &'a mut dyn FnMut(usize, usize) -> Result<()>,
}

/// TP-SP world-2 REDUCE-SCATTER of a `c`-row partial (row_len elements per row; io as reduce_pipe:
/// io.src = the partial's row 0, io.out = the output's row 0): this rank ends with the canonical sum
/// in its own rows of io.out ONLY (the peer's rows of io.out are not written). `own` / `peer` = the
/// row blocks. Bitwise the all-reduce's values on the own rows.
#[allow(clippy::too_many_arguments)]
pub fn reduce_scatter_rows(k: &Kernels, stream: &CudaStream, x: &Pipe, io: Io, row_len: usize,
                           own: (usize, usize), peer: (usize, usize), hook: Option<SpHook<'_>>) -> Result<()> {
    anyhow::ensure!(x.rounds == 1, "TP-SP reduce-scatter: world 2 only (rounds {})", x.rounds);
    anyhow::ensure!(row_len % 8 == 0 && own.1 > own.0 && peer.1 > peer.0,
                    "TP-SP reduce-scatter: row_len {row_len} own {own:?} peer {peer:?}");
    let wire16 = io.wire16();
    let (se, oe, wb) = (if io.src_f16 { 2usize } else { 4 }, if io.out_f16 { 2usize } else { 4 }, if wire16 { 2usize } else { 4 });
    let rpe = rows_per_epoch(row_len * wb);
    anyhow::ensure!(rpe >= 1, "TP-SP reduce-scatter: a {row_len}-element row exceeds one ring slot");
    let e = pieces_count(&[own, peer], rpe);
    let (sp, rp) = (row_pieces(peer.0, peer.1, e), row_pieces(own.0, own.1, e));
    let eps: Vec<XEp> = (0..e).map(|j| {
        let ((s0, s1), (r0, r1)) = (sp[j], rp[j]);
        XEp {
            src: io.src + (s0 * row_len * se) as u64,
            bytes: (s1 - s0) * row_len * wb,
            k1bits: io.bits() & 5,
            rx: Rx::Add { out: io.out + (r0 * row_len * oe) as u64, local: io.src + (r0 * row_len * se) as u64,
                          n: (r1 - r0) * row_len, bits: io.bits() },
            rows: r1 - r0,
        }
    }).collect();
    run_xchg(k, stream, x, &eps, own.0, hook)
}

/// TP-SP world-2 ALL-GATHER of `row_bytes`-byte rows at `base`: this rank's own rows go to the peer
/// (K1m, a plain byte copy) and the peer's own rows land here (copy-K2), bit for bit. Afterwards every
/// row of [min(own, peer), max(own, peer)) is the owner's bytes on both ranks.
pub fn all_gather_rows(k: &Kernels, stream: &CudaStream, x: &Pipe, base: u64, row_bytes: usize,
                       own: (usize, usize), peer: (usize, usize)) -> Result<()> {
    all_gather_rows_multi(k, stream, x, &[(base, row_bytes)], own, peer)
}

/// The all-gather epochs of ONE row buffer (`base`, `row_bytes`): piece j of the own rows out, piece
/// j of the peer's rows in (the SPMD pairing above).
fn gather_epochs(base: u64, row_bytes: usize, own: (usize, usize), peer: (usize, usize)) -> Result<Vec<XEp>> {
    anyhow::ensure!(row_bytes % 16 == 0 && own.1 > own.0 && peer.1 > peer.0,
                    "TP-SP all-gather: row_bytes {row_bytes} own {own:?} peer {peer:?}");
    let rpe = rows_per_epoch(row_bytes);
    anyhow::ensure!(rpe >= 1, "TP-SP all-gather: a {row_bytes}-byte row exceeds one ring slot");
    let e = pieces_count(&[own, peer], rpe);
    let (op, pp) = (row_pieces(own.0, own.1, e), row_pieces(peer.0, peer.1, e));
    Ok((0..e).map(|j| {
        let ((s0, s1), (r0, r1)) = (op[j], pp[j]);
        XEp {
            src: base + (s0 * row_bytes) as u64,
            bytes: (s1 - s0) * row_bytes,
            k1bits: 0,
            rx: Rx::Copy { out: base + (r0 * row_bytes) as u64, bytes: (r1 - r0) * row_bytes },
            rows: 0,
        }
    }).collect())
}

/// TP-SP2: the all-gather of SEVERAL row buffers over the same row blocks as ONE exchange program:
/// buffer b's epochs follow buffer b-1's in one list, which both ranks build identically (a pure
/// function of the row lengths and (c, W)). The router's ids and weights (two 32-B rows per token)
/// thus ride two epochs dealt to the two rails in one launch program instead of two exchanges.
pub fn all_gather_rows_multi(k: &Kernels, stream: &CudaStream, x: &Pipe, bufs: &[(u64, usize)],
                             own: (usize, usize), peer: (usize, usize)) -> Result<()> {
    anyhow::ensure!(x.rounds == 1, "TP-SP all-gather: world 2 only (rounds {})", x.rounds);
    anyhow::ensure!(!bufs.is_empty(), "TP-SP all-gather: no buffers");
    let mut eps: Vec<XEp> = Vec::new();
    for &(base, row_bytes) in bufs { eps.extend(gather_epochs(base, row_bytes, own, peer)?); }
    run_xchg(k, stream, x, &eps, own.0, None)
}

/// The TP-SP exchange program: epoch j rides rail j % nr at index j / nr; per rail the TP-F launch
/// program (tick t: the receive of index t - G [lag = this rail's K1m launched after it], then K1m(t)),
/// so every rail keeps the TP-F recv-slot rule (R >= 2G) and its own device epoch counter. The hook
/// (reduce-scatter only) gets the own rows whose every epoch has had its receive launched, once >=
/// min_rows accumulate, then the rest at the end — reduce_pipe_hooked's rule on whole-row pieces.
fn run_xchg(k: &Kernels, stream: &CudaStream, x: &Pipe, eps: &[XEp], own0: usize,
            mut hook: Option<SpHook<'_>>) -> Result<()> {
    for ep in eps {
        anyhow::ensure!(ep.bytes % 16 == 0 && ep.bytes + 64 <= crate::tp::TP_SLOT_BYTES && ep.bytes > 0,
                        "TP-SP epoch of {} bytes: not a 16 B multiple within one ring slot", ep.bytes);
    }
    let caller = stream;
    let nr = x.rails.len().min(eps.len().max(1));
    let g = x.lookahead;
    let seqs: Vec<Vec<usize>> = (0..nr).map(|r| (0..eps.len()).filter(|i| i % nr == r).collect()).collect();
    let cfg = LaunchConfig { grid_dim: (x.blocks, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
    let arrive = *x.arrive.device_ptr() as u64;
    let dual = hook.as_ref().map_or(false, |h| h.side);
    let comm: &CudaStream = if dual { &x.side.stream } else { stream };
    if dual { stream_after(comm, caller, x.side.ev_in)?; }
    let longest = seqs.iter().map(|s| s.len()).max().unwrap_or(0);
    let total_rows: usize = eps.iter().map(|e| e.rows).sum();
    let mut rows_done = 0usize;
    for t in 0..longest + g {
        for (r, sq) in seqs.iter().enumerate() {
            let ctx = x.rails[r];
            if t >= g && t - g < sq.len() {
                let j = t - g;
                let lag = (t.min(sq.len()) - 1 - j) as u32;
                match eps[sq[j]].rx {
                    Rx::Add { out, local, n, bits } => unsafe {
                        k.k2m.clone().launch_on_stream(comm, cfg, (ctx, out, local, n as i32, lag, bits))
                    }.context("TP-SP K2m launch")?,
                    Rx::Copy { out, bytes } => unsafe {
                        k.k2c.clone().launch_on_stream(comm, cfg, (ctx, out, bytes as u32, lag))
                    }.context("TP-SP copy-K2 launch")?,
                }
            }
            if t < sq.len() {
                let ep = &eps[sq[t]];
                unsafe { k.k1m.clone().launch_on_stream(comm, cfg, (ctx, ep.src, ep.bytes as u32, arrive, ep.k1bits)) }
                    .context("TP-SP K1m launch")?;
            }
        }
        if let (true, Some(h)) = (t >= g, hook.as_mut()) {
            let fin = (nr * (t - g + 1)).min(eps.len());
            let rows: usize = eps[..fin].iter().map(|e| e.rows).sum();
            if rows > rows_done && (rows - rows_done >= h.min_rows || rows == total_rows) {
                if dual { stream_after(caller, comm, x.side.ev_out)?; }
                (h.f)(own0 + rows_done, own0 + rows)?;
                rows_done = rows;
            }
        }
    }
    if dual { stream_after(caller, comm, x.side.ev_out)?; }
    if let Some(h) = hook.as_mut() {
        if total_rows > rows_done { (h.f)(own0 + rows_done, own0 + total_rows)?; }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TP-SP: for every chunk size the prefill can split (2..4096 rows), every row length the schedules
    /// carry (the f16-wire / fp32-wire partial, the f16 x row, the f32 stream row) and world 2: the two
    /// blocks tile [0, c); each rank's E pieces tile its block in order, are non-empty and fit a slot;
    /// both ranks compute the SAME E, and rank a's SEND piece j (piece j of the peer's block) is exactly
    /// rank b's RECEIVE piece j (piece j of its own block) — the SPMD pairing the doorbell relies on.
    #[test]
    fn sp_pieces_pair_and_tile() {
        for c in 2..=4096usize {
            let (b0, b1) = (row_block(c, 0, 2), row_block(c, 1, 2));
            assert_eq!((b0.0, b0.1, b1.1), (0, b1.0, c), "c {c}: blocks do not tile");
            if b1.1 == b1.0 { continue; }
            for &row_bytes in &[2560usize * 2, 2560 * 4, 10240 * 4, 8 * 4] {
                let rpe = rows_per_epoch(row_bytes);
                assert!(rpe >= 1);
                let e0 = pieces_count(&[b0, b1], rpe);
                let e1 = pieces_count(&[b1, b0], rpe);
                assert_eq!(e0, e1, "c {c}: ranks disagree on E");
                for &(lo, hi) in &[b0, b1] {
                    let p = row_pieces(lo, hi, e0);
                    let mut at = lo;
                    for &(a, b) in &p {
                        assert_eq!(a, at);
                        assert!(b > a || hi - lo < e0, "c {c} rb {row_bytes}: empty piece");
                        assert!((b - a) * row_bytes + 64 <= crate::tp::TP_SLOT_BYTES, "c {c}: piece exceeds a slot");
                        at = b;
                    }
                    assert_eq!(at, hi);
                }
                // rank 0 sends pieces of b1 = rank 1 receives pieces of its own b1 (and vice versa)
                assert_eq!(row_pieces(b1.0, b1.1, e0), row_pieces(b1.0, b1.1, e1));
                assert_eq!(row_pieces(b0.0, b0.1, e1), row_pieces(b0.0, b0.1, e0));
                if c >= 128 { assert!((b1.1 - b1.0) >= e0, "c {c} rb {row_bytes}: fewer rows than epochs"); }
            }
        }
    }

    /// TP-SP2: the multi-buffer all-gather (the router's ids + weights, or x alone): both ranks build
    /// lists of the SAME length, and epoch j of rank a sends exactly the bytes epoch j of rank b
    /// receives (same buffer, same rows: src range of a == Copy.out range of b), for every split c.
    #[test]
    fn sp_multi_gather_pairs() {
        let bufs: [(u64, usize); 2] = [(0x1000_0000, 32), (0x2000_0000, 32)];
        for c in 128..=4095usize {
            let (b0, b1) = (row_block(c, 0, 2), row_block(c, 1, 2));
            for set in [&bufs[..], &bufs[..1], &[(0x3000_0000u64, 2560usize * 2)][..]] {
                let mk = |own: (usize, usize), peer: (usize, usize)| -> Vec<XEp> {
                    set.iter().flat_map(|&(b, rb)| gather_epochs(b, rb, own, peer).unwrap()).collect()
                };
                let (r0, r1) = (mk(b0, b1), mk(b1, b0));
                assert_eq!(r0.len(), r1.len(), "c {c}: ranks build different epoch counts");
                for (a, b) in r0.iter().zip(&r1).chain(r1.iter().zip(&r0)) {
                    let Rx::Copy { out, bytes } = b.rx else { panic!("c {c}: not a copy epoch") };
                    assert_eq!((a.src, a.bytes), (out, bytes), "c {c}: send piece != the peer's receive piece");
                }
            }
        }
    }

    /// TP-I2 item 1: the hook's row ranges tile [0, rows) exactly, in order, and every row handed over
    /// at tick t lies entirely inside epochs whose K2m was launched at or before tick t (simulated with
    /// the schedule's own rail deal), for both wire widths, 1 and 2 rails, every lookahead.
    #[test]
    fn hook_rows_are_finished_rows() {
        for &per in &[EPOCH_FLOATS, 2 * EPOCH_FLOATS] {
            for &row_len in &[2560usize, 2048, 1000] {
                for &c in &[17usize, 26, 100, 1024, 2047, 2048] {
                    let n = c * row_len;
                    let ep = epochs_of(n, per);
                    for nr in 1..=2usize.min(ep.len()) {
                        for g in 1..=RING_SLOTS / 2 {
                            for &min_rows in &[1usize, 64, 128, 4096] {
                                let lens: Vec<usize> = (0..nr).map(|r| (0..ep.len()).filter(|i| i % nr == r).count()).collect();
                                let longest = *lens.iter().max().unwrap();
                                let mut done_ep = vec![false; ep.len()];
                                let mut rows_done = 0usize;
                                for t in 0..longest + g {
                                    for r in 0..nr {
                                        if t >= g && t - g < lens[r] { done_ep[(t - g) * nr + r] = true; }
                                    }
                                    if let Some(r1) = hook_rows_after_tick(&ep, n, nr, g, t, row_len, min_rows, rows_done) {
                                        assert!(r1 > rows_done);
                                        let last = r1 * row_len - 1; // the range's last element
                                        let e_last = ep.iter().position(|&(o, l)| last >= o && last < o + l).unwrap();
                                        assert!(done_ep[..=e_last].iter().all(|&d| d),
                                                "per {per} row_len {row_len} c {c} nr {nr} g {g} t {t}: rows < {r1} not final");
                                        rows_done = r1;
                                    }
                                }
                                assert_eq!(rows_done, c, "per {per} row_len {row_len} c {c} nr {nr} g {g} min {min_rows}");
                            }
                        }
                    }
                }
            }
        }
    }
    /// TP-I3 (c1): the fold schedule's epochs carry whole rows, fit a ring slot, and tile [0, n) in order,
    /// for both wire widths and the served row lengths.
    #[test]
    fn fold_epochs_are_whole_rows() {
        for &per in &[EPOCH_FLOATS, 2 * EPOCH_FLOATS] {
            for &row_len in &[2560usize, 2048, 4096] {
                for &c in &[1usize, 17, 101, 102, 103, 204, 205, 1024, 1775, 2048] {
                    let n = c * row_len;
                    let fp = fold_per(per, row_len);
                    assert!(fp > 0 && fp <= per && fp % row_len == 0);
                    let ep = epochs_of(n, fp);
                    let mut off = 0usize;
                    for &(o, l) in &ep {
                        assert_eq!(o, off);
                        assert!(o % row_len == 0 && l % row_len == 0 && l <= per, "per {per} row_len {row_len} c {c}");
                        off += l;
                    }
                    assert_eq!(off, n);
                }
            }
        }
    }
    /// The world>2 schedule invariants that TP-F cannot test on hardware (2 boxes): for every chunk
    /// and round k >= 1, round k-1's epoch j' of the same chunk precedes round k's epoch j by >= G
    /// (K1m(j) is launched after K2m(j-G), so round k copies round k-1's finished sum); every epoch's
    /// round is its index % rounds; every chunk appears exactly once per round.
    #[test]
    fn rail_seq_round_dependencies() {
        for rounds in [1usize, 2, 3, 4] {
            for g in 1..=RING_SLOTS / 2 {
                if g % rounds != 0 && rounds > 1 { continue; }
                for c in 1..30usize {
                    let chunks: Vec<(usize, usize)> = (0..c).map(|i| (i * 16, 16)).collect();
                    let sq = rail_seq(&chunks, rounds, g);
                    assert_eq!(sq.len() % rounds, 0);
                    let mut pos = vec![vec![usize::MAX; rounds]; c];
                    for (j, e) in sq.iter().enumerate() {
                        if let Ep::Chunk(o, _) = e {
                            let ci = o / 16;
                            let k = j % rounds;
                            assert_eq!(pos[ci][k], usize::MAX, "chunk {ci} round {k} twice");
                            pos[ci][k] = j;
                        }
                    }
                    for ci in 0..c {
                        for k in 0..rounds {
                            assert!(pos[ci][k] != usize::MAX, "chunk {ci} round {k} missing");
                            if k > 0 { assert!(pos[ci][k] >= pos[ci][k - 1] + g, "rounds {rounds} g {g} chunk {ci} round {k}"); }
                        }
                    }
                }
            }
        }
    }
}
