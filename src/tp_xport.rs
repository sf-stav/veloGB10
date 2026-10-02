//! TP-F — the fp32 all-reduce LAUNCH SCHEDULES (world 2: one exchange; world 4: two recursive-doubling
//! rounds, TP-4B) over the doorbell transport
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
//! result is bitwise independent of the schedule, the chunking and the rail deal. At world > 2 the
//! per-round adds telescope in a fixed tree whose association depends on the device-epoch phase
//! (see `serial_model` in the tests); every rank ends bit-identical.

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
        let g = effective_lookahead(lookahead, rounds).with_context(|| format!("world {world}"))?;
        let side = Side::new(dev)?;
        Ok(Pipe { rails, arrive, rounds, lookahead: g, blocks: blocks.clamp(1, 128), bubble, side })
    }
}

/// The epochs in flight per rail for a requested `lookahead`: G <= R/2 (recv-slot rule) and a multiple of
/// `rounds` (world>2: round(x+R-G) == round(x)); also `rounds` must divide R, or epoch x and x+R (the same ring
/// slot) would sit on different rounds (world 8: rounds 3 does not divide R = 8, refused here). A pure function so the
/// CPU model tests run the real clamp.
pub fn effective_lookahead(lookahead: usize, rounds: usize) -> Result<usize> {
    anyhow::ensure!(rounds >= 1 && RING_SLOTS % rounds == 0,
        "rounds {rounds} does not divide the {RING_SLOTS}-slot recv ring: epoch x and x + {RING_SLOTS} share a slot but not a round \
         (world 8 has rounds 3), so the pipelined transport refuses it");
    let mut g = lookahead.clamp(1, RING_SLOTS / 2);
    g = (g / rounds).max(1) * rounds;
    anyhow::ensure!(g <= RING_SLOTS / 2, "no lookahead satisfies G <= R/2 with G % rounds == 0 (rounds {rounds})");
    Ok(g)
}

/// Split n floats into ring-slot epochs: (float offset, len).
#[cfg(test)]
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

/// log2(world): the recursive-doubling round count (1 at world 2). Only power-of-two worlds >= 2.
pub fn rounds_of(world: usize) -> Result<usize> {
    anyhow::ensure!(world >= 2 && world.is_power_of_two(), "world {world} must be a power of two >= 2");
    Ok(world.trailing_zeros() as usize)
}

/// One K1/K2 pair of the serial all-reduce: floats [off, off + len) of the buffer, round `round` of that chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SerialPair { pub off: usize, pub len: usize, pub round: usize }

impl SerialPair {
    /// (address, K1 byte count, K2 element count) of this pair over a buffer that starts at `base`.
    fn args(&self, base: u64) -> (u64, u32, i32) { (base + (self.off * 4) as u64, (self.len * 4) as u32, self.len as i32) }
}

/// The serial all-reduce's launch plan, in stream order: every chunk of `per` floats runs `rounds`
/// consecutive K1/K2 pairs. `reduce_serial` launches exactly this list and the CPU schedule model in the
/// tests consumes the same list, so a change to the real loop's pairs-per-chunk or chunking fails a test.
fn serial_plan_of(n: usize, per: usize, rounds: usize) -> Vec<SerialPair> {
    let mut v = Vec::new();
    for (off, len) in epochs_of(n, per) {
        for round in 0..rounds { v.push(SerialPair { off, len, round }); }
    }
    v
}

pub fn serial_plan(n: usize, rounds: usize) -> Vec<SerialPair> { serial_plan_of(n, EPOCH_FLOATS, rounds) }

/// In-place fp32 sum all-reduce of `n` floats at device address `p`, serial single-block pairs.
/// `rounds` = log2(world): each chunk runs `rounds` consecutive K1/K2 pairs on the same fp32 buffer
/// (round k exchanges the round k-1 SUM with partner rank ^ (1 << (epoch % rounds)); K2 mode 2 adds in
/// place, no rounding between rounds). A chunk consumes exactly `rounds` device epochs, so the round
/// phase is invariant across reduces. rounds == 1 (world 2) is one pair per chunk, the TP-B..E launch.
pub fn reduce_serial(k: &Kernels, stream: &CudaStream, ctx: u64, p: u64, n: usize, rounds: usize) -> Result<()> {
    anyhow::ensure!(rounds >= 1, "reduce_serial: rounds {rounds}");
    let c1 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
    for sp in serial_plan(n, rounds) {
        anyhow::ensure!(sp.len * 4 + 64 <= crate::tp::TP_SLOT_BYTES, "all-reduce epoch {} floats exceeds one ring slot", sp.len);
        let (q, bytes, len) = sp.args(p);
        unsafe {
            k.k1.clone().launch_on_stream(stream, c1, (ctx, q, bytes)).context("K1 launch")?;
            k.k2.clone().launch_on_stream(stream, c1, (ctx, q, q, len, 2i32)).context("K2 launch")?;
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

/// TP-4C: one K1 + K2 exchange of a DECODE-path logical reduce (the folded single-exchange path of TP-G,
/// generalised to `rounds` = log2(world) exchanges). Every decode arm — the mixer out-projection reduce,
/// the EP MoE combine reduce, the vocab-parallel key merge and the sharded draft-head screen — runs
/// exactly `dec_exchanges(rounds)`: EXACTLY `rounds` device epochs (each K1 pre-increments the epoch
/// counter once), so the R10 round phase (`align_round_phase`: the counter is a multiple of `rounds`
/// between logical reduces) is invariant across arms and across any interleaving of them with each other
/// and with the prefill `reduce_serial` (also `rounds` per chunk).
///   round 0 : K1 is the PRODUCER's folded last-block K1 (xq_*_k1l / xq_argmax_rows_vp / xq_dh_screen_k1);
///             K2 reads the caller's `local` partial; at rounds > 1 it writes the fp32 intermediate.
///   round k : K1 is the plain single-block K1 (tp_gate_copy_signal, `k1_serial`) of the fp32
///             intermediate (the round k-1 sum: an f16 buffer would round between rounds, so only the
///             LAST round may write f16); K2 reads the intermediate, the last round writes `out`.
/// rounds == 1 (world 2) is the single exchange of today's folded path: local -> out, folded K1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecExchange {
    /// the exchange's index within the logical reduce (its K1 is epoch e0 + 1 + round; the device arm's
    /// round is (epoch % rounds), which is what picks the partner bit)
    pub round: usize,
    /// K1 is the producer's folded last-block K1 (round 0 only); otherwise the host launches the plain K1
    pub folded_k1: bool,
    /// K2's local operand is the caller's `local` buffer (round 0) rather than the fp32 intermediate
    pub local_is_input: bool,
    /// K2 writes the caller's final `out` (the last round) rather than the fp32 intermediate
    pub writes_final: bool,
}

/// The exchange list of one decode-path logical reduce at `rounds` rounds (see `DecExchange`).
pub fn dec_exchanges(rounds: usize) -> Vec<DecExchange> {
    let r = rounds.max(1);
    (0..r).map(|k| DecExchange { round: k, folded_k1: k == 0, local_is_input: k == 0, writes_final: k + 1 == r }).collect()
}

/// The K2 operands of one exchange over the caller's buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecK2Args { pub out: u64, pub local: u64, pub local_f16: bool, pub out_f16: bool }

/// The fp32 intermediate of a multi-round decode reduce: `out` itself when it is fp32 (the draft-head
/// screen: its `local` is a zero-padded block-maxima buffer that must keep its zeros), else `local`
/// (mixer / MoE: the fp32 partial buffer, fully rewritten by its producer before every reduce).
pub fn dec_intermediate(local: u64, out: u64, out_f16: bool) -> u64 { if out_f16 { local } else { out } }

/// K2 operands of exchange `x`. At rounds == 1 this is exactly (out, local, local_f16, out_f16) — the
/// TP-G launch — so world 2 is unchanged; at rounds > 1 round 0 reads the caller's local (never f16:
/// the producers write fp32) and writes the fp32 intermediate, later rounds read and write the fp32
/// intermediate, and only the last round writes the caller's `out` (f16 when `out_f16`).
pub fn dec_exchange_args(x: &DecExchange, local: u64, local_f16: bool, out: u64, out_f16: bool) -> DecK2Args {
    let inter = dec_intermediate(local, out, out_f16);
    DecK2Args {
        local: if x.local_is_input { local } else { inter },
        local_f16: x.local_is_input && local_f16,
        out: if x.writes_final { out } else { inter },
        out_f16: x.writes_final && out_f16,
    }
}

/// The recursive-doubling reference of the fixed-order all-reduce, as a tree over the `world` ranks'
/// partials: before a reduce the device epoch counter is `e0` (K1 pre-increments, so the first exchange
/// is epoch e0 + 1 and runs round (e0 + 1) % rounds first); after round j a rank holds
/// tree(lower half) + tree(upper half) over the bit just exchanged. Every rank ends with this value,
/// bitwise. Only `e0 % rounds` matters, which R10 pins at attach. Pure host code shared by the W=4
/// decode-mode reduce bench and the schedule-model tests.
pub fn rd_tree_reference(world: usize, e0: u64, parts: &[Vec<f32>]) -> Result<Vec<f32>> {
    let rounds = rounds_of(world)?;
    anyhow::ensure!(parts.len() == world && parts.iter().all(|p| p.len() == parts[0].len()),
                    "rd_tree_reference: {} partials for world {world} (or ragged)", parts.len());
    let first = ((e0 + 1) % rounds as u64) as usize;
    let order: Vec<usize> = (0..rounds).map(|j| (first + j) % rounds).collect();
    fn tree(r: usize, j: usize, order: &[usize], parts: &[Vec<f32>], i: usize) -> f32 {
        if j == 0 { return parts[r][i]; }
        let bit = 1usize << order[j - 1];
        tree(r & !bit, j - 1, order, parts, i) + tree(r | bit, j - 1, order, parts, i)
    }
    Ok((0..parts[0].len()).map(|i| tree(0, rounds, &order, parts, i)).collect())
}

/// TP-4F2 (`--tp-reduce single`): the SINGLE-STAGE world-4 reduce's summation, per element, from the four ranks'
/// fp32 partials indexed by RANK: `(p0 + p2) + (p1 + p3)` — the rank-independent association every rank computes
/// after ONE all-peers exchange (`xq_tp_wait_add_dec_single`). It is exactly the recursive-doubling tree of the
/// R10-aligned reduce (`rd_tree_reference(4, even e0, ..)`: stage 0 pairs rank^2, stage 1 pairs rank^1; fp32 `+` is
/// commutative, so the order inside a pair — which differs per rank — cannot change a bit). Pure host code: the
/// bit-identity argument of the flag, unit-tested against the rd reference and an explicit per-rank rd dataflow.
pub fn single_tree_reference(parts: &[Vec<f32>]) -> Result<Vec<f32>> {
    anyhow::ensure!(parts.len() == 4 && parts.iter().all(|p| p.len() == parts[0].len()),
                    "single_tree_reference: {} partials (world 4 needs exactly 4, equal length)", parts.len());
    Ok((0..parts[0].len()).map(|i| (parts[0][i] + parts[2][i]) + (parts[1][i] + parts[3][i])).collect())
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

/// Where a pipelined reduce reads its local partial and writes the sum.
/// `src_f16`: the local partial is f16 (K1m widens it on the copy, K2m on the add — exact);
/// `out_f16`: the sum is rounded once to f16 (__float2half_rn, as xq_cvt_f32_f16). src == out is
/// allowed (same-thread read-then-write in K2m).
/// `wire_f16` (requires `src_f16`): the partial is already f16-exact, so the wire carries f16 —
/// half the bytes, bitwise the same fp32 sum (the peer widens exactly). At world > 2 only the FIRST
/// exchange of a chunk is f16 (its payload is the f16-exact partial); every later exchange carries an
/// fp32 running sum, because rounding that sum to f16 between rounds would change the output.
/// `mid` (world > 2 only): the fp32 staging buffer of `n` floats holding a chunk's running sum between
/// exchanges. Round k > 0 must exchange the round k-1 SUM, never the unsummed partial, and the sum is
/// fp32 (an f16 buffer would round between rounds), so an IO whose src/out is f16 or differs needs
/// somewhere to keep it. 0 = none: only the plain fp32 in-place IO (src == out, no f16) is legal then
/// (it stages in place). `mid == src` (an fp32 src the caller lets the reduce consume) and
/// `mid == out` (an fp32 out) are legal; otherwise `mid` must not overlap src or out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Io { pub src: u64, pub src_f16: bool, pub out: u64, pub out_f16: bool, pub wire_f16: bool, pub mid: u64 }

/// world > 2 (rounds > 1): where the running sum lives between exchanges. Exchange k of a chunk reads
/// its local operand from `src` (k == 0) or the stage (k > 0) and writes its sum to the stage (k < rounds-1)
/// or `out` (the last exchange), so each round exchanges the previous round's SUM, in fp32, and the one
/// rounding to f16 (if `out_f16`) happens once, at the end — bitwise the old cvt -> fp32 all-reduce -> cvt
/// chain, without the two conversion passes. Returns the stage base address. Anything that cannot be staged
/// is refused (the unsummed-partial trap: each rank would silently keep a pair sum).
fn stage_for(io: &Io, n: usize) -> Result<u64> {
    let inplace_f32 = io.src == io.out && !io.src_f16 && !io.out_f16;
    if inplace_f32 { return Ok(io.src); }
    anyhow::ensure!(io.mid != 0 && io.mid % 16 == 0,
        "reduce IO at world > 2 needs a 16 B-aligned fp32 staging buffer `mid` (src {:#x} out {:#x}, f16 src/out {}/{}, mid {:#x}): \
         each round must exchange the previous round's fp32 sum",
        io.src, io.out, io.src_f16, io.out_f16, io.mid);
    let (se, oe) = (if io.src_f16 { 2u64 } else { 4 }, if io.out_f16 { 2u64 } else { 4 });
    let span = |p: u64, w: u64| (p, p + w * n as u64);
    let disjoint = |a: (u64, u64), b: (u64, u64)| a.1 <= b.0 || b.1 <= a.0;
    anyhow::ensure!((io.src == io.out && io.src_f16 == io.out_f16) || disjoint(span(io.src, se), span(io.out, oe)),
        "reduce IO at world > 2: src {:#x} ({} B/elem) and out {:#x} ({} B/elem) overlap without being the identical span \
         (K2m reads src[i] and writes out[i] in one thread; a partial overlap lets another thread overwrite an element before it is read)",
        io.src, se, io.out, oe);
    let mb = span(io.mid, 4);
    anyhow::ensure!((io.mid == io.src && !io.src_f16) || disjoint(mb, span(io.src, se)),
        "reduce IO at world > 2: staging buffer {:#x} overlaps src {:#x} (only an fp32 src may be consumed in place)", io.mid, io.src);
    anyhow::ensure!((io.mid == io.out && !io.out_f16) || disjoint(mb, span(io.out, oe)),
        "reduce IO at world > 2: staging buffer {:#x} overlaps out {:#x} (only an fp32 out may be the stage)", io.mid, io.out);
    Ok(io.mid)
}

/// The launch arguments of exchange `k` of a chunk (see `Io::xchg`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Xchg {
    /// K1m's `src` and K2m's `local` (the same operand)
    local: u64,
    /// K2m's `out`
    out: u64,
    /// K1m's bit 0 (local f16) and bit 2 (f16 wire); K2m adds bit 1 (out f16) on the last exchange
    k1_bits: i32,
    k2_bits: i32,
    /// the wire of this exchange is f16 (2 bytes per element instead of 4)
    wire_f16: bool,
}

impl Io {
    /// The plain in-place fp32 reduce.
    pub fn f32_inplace(p: u64) -> Io { Io { src: p, src_f16: false, out: p, out_f16: false, wire_f16: false, mid: 0 } }
    fn wire16(&self) -> bool { self.wire_f16 && self.src_f16 }
    fn bits(&self) -> i32 { (self.src_f16 as i32) | ((self.out_f16 as i32) << 1) | ((self.wire16() as i32) << 2) }

    /// Exchange `k` (0-based; k < rounds) of the chunk at float offset `o`; `stage` from `stage_for` (unused at
    /// rounds == 1). Exchange 0 reads the source (f16 or fp32) and may ship f16; later exchanges read the fp32
    /// stage and ship fp32; the last writes `out` (f16 if `out_f16`), the others the stage. At rounds == 1 the
    /// only exchange is first AND last, which is exactly the world-2 launch (src/out/bits as `Io::bits`).
    fn xchg(&self, stage: u64, rounds: usize, o: usize, k: usize) -> Xchg {
        let (se, oe) = (if self.src_f16 { 2u64 } else { 4 }, if self.out_f16 { 2u64 } else { 4 });
        let (first, last) = (k == 0, k + 1 == rounds);
        let w16 = first && self.wire16();
        let k1_bits = ((first && self.src_f16) as i32) | ((w16 as i32) << 2);
        Xchg {
            local: if first { self.src + o as u64 * se } else { stage + o as u64 * 4 },
            out: if last { self.out + o as u64 * oe } else { stage + o as u64 * 4 },
            k1_bits,
            k2_bits: k1_bits | (((last && self.out_f16) as i32) << 1),
            wire_f16: w16,
        }
    }
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

/// One launch of a pipelined reduce, in stream order. The program is a pure function of the geometry
/// (`build_program`), so the CPU model in the tests runs the very list the executor launches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    /// `tp_gate_copy_signal_mb(ctx[rail], src, nbytes, arrive, bits)`
    K1m { rail: usize, src: u64, nbytes: u32, bits: i32 },
    /// `tp_wait_add_mb(ctx[rail], out, local, n, lag, bits)`
    K2m { rail: usize, out: u64, local: u64, n: i32, lag: u32, bits: i32 },
    /// `tp_wait_copy_mb(ctx[rail], out, bytes, lag)` — the TP-SP all-gather's plain-copy receive (TP-4S programs only;
    /// `build_program` never emits it)
    K2c { rail: usize, out: u64, bytes: u32, lag: u32 },
    /// TP-I3 fold consumer of rows [r0, r1) at K2m's position (rounds == 1 only)
    Fold { rail: usize, lag: u32, r0: usize, r1: usize },
    /// the rows-landed hook over [r0, r1); `tail`: the trailing hook after the whole schedule
    Hook { r0: usize, r1: usize, tail: bool },
}

#[derive(Clone, Copy, Debug)]
struct HookSpec { row_len: usize, min_rows: usize }

#[derive(Clone, Copy, Debug)]
struct ProgSpec {
    io: Io,
    rounds: usize,
    /// rails requested (the program uses min(nrails, #epochs))
    nrails: usize,
    /// epochs in flight per rail
    g: usize,
    /// the bubble scratch (16 B)
    bub: u64,
    hook: Option<HookSpec>,
    fold: bool,
}

/// The K1m/K2m/hook launch list of a pipelined reduce of `n` floats cut into the epochs `ep`. Tick t: per
/// rail, K2m(t-G) (if any) then K1m(t) (if any); after the tick the hook gets every whole-row range whose
/// chunks have ALL had their final-exchange K2m launched (rounds == 1: the chunk's only K2m; rounds > 1: its
/// round `rounds-1` K2m, which is where the chunk's output is final). A range is handed over once it holds
/// >= min_rows rows, or at the end; whatever is left goes to the trailing hook.
fn build_program(spec: &ProgSpec, ep: &[(usize, usize)], n: usize) -> Result<Vec<Step>> {
    let ProgSpec { io, rounds, nrails, g, bub, hook, fold } = *spec;
    anyhow::ensure!(!fold || (rounds == 1 && hook.is_some()), "prefill fold: world 2 only with a hook (rounds {rounds})");
    let stage = if rounds > 1 { stage_for(&io, n)? } else { 0 };
    let nr = nrails.min(ep.len().max(1));
    let seqs: Vec<Vec<Ep>> = (0..nr).map(|r| {
        let mine: Vec<(usize, usize)> = ep.iter().enumerate().filter(|(i, _)| i % nr == r).map(|(_, &c)| c).collect();
        rail_seq(&mine, rounds, g)
    }).collect();
    // seq index of each chunk's LAST exchange (its output is final after that K2m)
    let mut fin_j = vec![usize::MAX; ep.len()];
    for sq in &seqs {
        for (j, e) in sq.iter().enumerate() {
            if let Ep::Chunk(o, _) = *e {
                if j % rounds == rounds - 1 {
                    let i = ep.binary_search_by_key(&o, |c| c.0).map_err(|_| anyhow::anyhow!("chunk offset {o} not an epoch"))?;
                    fin_j[i] = j;
                }
            }
        }
    }
    anyhow::ensure!(fin_j.iter().all(|&j| j != usize::MAX), "a chunk never reaches its last exchange");
    let longest = seqs.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut prog: Vec<Step> = Vec::with_capacity(seqs.iter().map(|s| 2 * s.len()).sum::<usize>() + 8);
    let (mut rows_done, mut fin) = (0usize, 0usize);
    for t in 0..longest + g {
        for (r, sq) in seqs.iter().enumerate() {
            if t >= g && t - g < sq.len() {
                let j = t - g;
                let lag = (t.min(sq.len()) - 1 - j) as u32;       // this rail's K1m launched after j
                match sq[j] {
                    Ep::Chunk(o, l) => {
                        if fold {
                            // TP-I3 (c1): the fused consumer of this epoch's whole rows, at K2m's position
                            let rl = hook.unwrap().row_len;
                            prog.push(Step::Fold { rail: r, lag, r0: o / rl, r1: (o + l) / rl });
                        } else {
                            let a = io.xchg(stage, rounds, o, j % rounds);
                            prog.push(Step::K2m { rail: r, out: a.out, local: a.local, n: l as i32, lag, bits: a.k2_bits });
                        }
                    }
                    Ep::Bubble => prog.push(Step::K2m { rail: r, out: bub, local: bub, n: 4, lag, bits: 0 }),
                }
            }
            if t < sq.len() {
                match sq[t] {
                    Ep::Chunk(o, l) => {
                        let a = io.xchg(stage, rounds, o, t % rounds);
                        let wb = if a.wire_f16 { 2 } else { 4 };      // wire bytes per element
                        prog.push(Step::K1m { rail: r, src: a.local, nbytes: (l * wb) as u32, bits: a.k1_bits });
                    }
                    Ep::Bubble => prog.push(Step::K1m { rail: r, src: bub, nbytes: 16, bits: 0 }),
                }
            }
        }
        if let (true, Some(h)) = (!fold, hook) {
            while fin < ep.len() && fin_j[fin] + g <= t { fin += 1; }
            let elems = if fin == ep.len() { n } else { ep[fin].0 };
            let rows = elems / h.row_len;
            if rows > rows_done && (rows - rows_done >= h.min_rows || elems == n) {
                prog.push(Step::Hook { r0: rows_done, r1: rows, tail: false });
                rows_done = rows;
            }
        }
    }
    if let (false, Some(h)) = (fold, hook) {
        let rows = n / h.row_len;
        if rows > rows_done { prog.push(Step::Hook { r0: rows_done, r1: rows, tail: true }); }
    }
    Ok(prog)
}

/// `reduce_pipe` with an optional rows-landed hook (TP-I2 item 1: comm/compute overlap by row
/// sub-chunks). The schedule and every K1m/K2m launch are `build_program`'s. The hook's launches go in at
/// the END of a tick (after both rails' K2m(t-G) and K1m(t)), once the K2m's launched so far have FINISHED at
/// least `min_rows` new whole rows. At that point G epochs per rail are published and in flight, so the wire
/// moves them while the GPU runs the hook instead of spinning in the next K2m. A row is handed over only when
/// every epoch covering it has had its LAST exchange's K2m launched earlier in stream order, so every value the
/// hook reads is the finished canonical fp32 sum: the reduce output and the consumer's output are bitwise
/// `reduce_pipe` + the consumer over all rows. World > 2 (rounds > 1): a chunk is final after its round
/// `rounds-1` K2m (the pipeline delays it by `d*rounds` epochs), and the same per-tick hand-over applies; fold
/// stays world 2 only.
pub fn reduce_pipe_hooked(k: &Kernels, stream: &CudaStream, x: &Pipe, io: Io, n: usize,
                          mut hook: Option<RowHook<'_>>) -> Result<()> {
    let caller = stream;
    anyhow::ensure!(pipe_ok(n), "pipelined all-reduce of {n} floats: not a 16 B multiple");
    if let Some(h) = hook.as_ref() {
        anyhow::ensure!(h.row_len > 0 && n % h.row_len == 0 && h.min_rows > 0,
            "reduce hook: n {n} is not a whole number of {}-element rows (min_rows {})", h.row_len, h.min_rows);
    }
    // K1m copies the f16 wire in 16 B vectors: an f16 wire needs every epoch a multiple of 8 elements
    let io = if io.wire16() && n % 8 != 0 { Io { wire_f16: false, ..io } } else { io };
    // an f16-wire epoch carries twice the elements in the same slot bytes; at world > 2 only the first
    // exchange is f16 and the later ones are fp32, so the epoch is sized for the fp32 exchanges
    let per = if io.wire16() && x.rounds == 1 { 2 * EPOCH_FLOATS } else { EPOCH_FLOATS };
    // TP-I3 (c1) fold: whole rows per epoch (the canonical per-element sum does not depend on the chunking)
    let fold_on = hook.as_ref().map_or(false, |h| h.fold.is_some());
    let per = if fold_on {
        let h = hook.as_ref().unwrap();
        anyhow::ensure!(x.rounds == 1, "prefill fold: world 2 only (rounds {})", x.rounds);
        anyhow::ensure!(h.row_len <= per && h.row_len % 8 == 0, "prefill fold: row_len {} vs epoch {per}", h.row_len);
        fold_per(per, h.row_len)
    } else { per };
    let ep = epochs_of(n, per);
    let bub = *x.bubble.device_ptr() as u64;
    let spec = ProgSpec { io, rounds: x.rounds, nrails: x.rails.len(), g: x.lookahead, bub, fold: fold_on,
                          hook: hook.as_ref().map(|h| HookSpec { row_len: h.row_len, min_rows: h.min_rows }) };
    let prog = build_program(&spec, &ep, n)?;
    let cfg = LaunchConfig { grid_dim: (x.blocks, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
    let arrive = *x.arrive.device_ptr() as u64;
    // v2: the schedule on the side stream, after the producer (everything already on `stream`)
    let dual = !fold_on && hook.as_ref().map_or(false, |h| h.side);
    let comm: &CudaStream = if dual { &x.side.stream } else { stream };
    if dual { stream_after(comm, stream, x.side.ev_in)?; }
    let stream = comm;
    for st in &prog {
        match *st {
            Step::K1m { rail, src, nbytes, bits } => {
                unsafe { k.k1m.clone().launch_on_stream(stream, cfg, (x.rails[rail], src, nbytes, arrive, bits)) }
                    .context("K1m launch")?;
            }
            Step::K2m { rail, out, local, n, lag, bits } => {
                unsafe { k.k2m.clone().launch_on_stream(stream, cfg, (x.rails[rail], out, local, n, lag, bits)) }
                    .context("K2m launch")?;
            }
            Step::K2c { .. } => anyhow::bail!("K2c is not part of an all-reduce program"),
            Step::Fold { rail, lag, r0, r1 } => {
                let f = hook.as_mut().unwrap().fold.as_mut().unwrap();
                f(x.rails[rail], lag, r0, r1)?;
            }
            Step::Hook { r0, r1, tail } => {
                let h = hook.as_mut().unwrap();
                if tail {
                    // v2: the caller's stream joins the whole schedule (every K1m/K2m) before anything after the reduce
                    if dual { stream_after(caller, comm, x.side.ev_out)?; }
                } else if dual {
                    // v2: the caller's stream waits for the K2m's launched so far (rows [.., r1) final)
                    stream_after(caller, comm, x.side.ev_out)?;
                }
                (h.f)(r0, r1)?;
            }
        }
    }
    // v2: the caller's stream joins the whole schedule (every K1m/K2m) before anything after the reduce
    // (idempotent with the tail hook's join above; also covers a reduce with no tail hook)
    if dual && !prog.iter().any(|s| matches!(s, Step::Hook { tail: true, .. })) { stream_after(caller, comm, x.side.ev_out)?; }
    Ok(())
}

/// TP-I3 (c1): the fold schedule's epoch length — the transport epoch rounded down to whole rows.
fn fold_per(per: usize, row_len: usize) -> usize { (per / row_len) * row_len }

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

/// TP-4S: the world-2 reduce-scatter's epoch list, moved out of `reduce_scatter_rows` VERBATIM so a test can
/// pin it against a frozen copy of the original inline block (the launched program is unchanged).
fn rs2_eps(io: &Io, row_len: usize, own: (usize, usize), peer: (usize, usize), rpe: usize) -> Vec<XEp> {
    let wire16 = io.wire16();
    let (se, oe, wb) = (if io.src_f16 { 2usize } else { 4 }, if io.out_f16 { 2usize } else { 4 }, if wire16 { 2usize } else { 4 });
    let e = pieces_count(&[own, peer], rpe);
    let (sp, rp) = (row_pieces(peer.0, peer.1, e), row_pieces(own.0, own.1, e));
    (0..e).map(|j| {
        let ((s0, s1), (r0, r1)) = (sp[j], rp[j]);
        XEp {
            src: io.src + (s0 * row_len * se) as u64,
            bytes: (s1 - s0) * row_len * wb,
            k1bits: io.bits() & 5,
            rx: Rx::Add { out: io.out + (r0 * row_len * oe) as u64, local: io.src + (r0 * row_len * se) as u64,
                          n: (r1 - r0) * row_len, bits: io.bits() },
            rows: r1 - r0,
        }
    }).collect()
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
    let wb = if wire16 { 2usize } else { 4 };
    let rpe = rows_per_epoch(row_len * wb);
    anyhow::ensure!(rpe >= 1, "TP-SP reduce-scatter: a {row_len}-element row exceeds one ring slot");
    let eps = rs2_eps(&io, row_len, own, peer, rpe);
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

// =================================================================================================
// TP-4S: sequence-parallel prefill at WORLD 4 (PLAN/TP-4S_DESIGN.md). Rank r owns QUARTER r of a chunk
// (row_block(c, r, 4)). The exchanges are the world-4 recursive-doubling ring's own two rounds used as a
// recursive-HALVING reduce-scatter and a recursive-DOUBLING all-gather, on the same doorbell protocol (same
// K1m, K2m, tp_wait_copy_mb, rings, tags, proxies): the partner of a device epoch e is rank ^ (1 << (e % 2)), so
// on a phase-aligned rail (R10) slot 0, 2, 4, .. of a program is an "A" exchange with rank ^ 2 and slot 1, 3, ..
// a "B" exchange with rank ^ 1.
//   reduce-scatter: A ships the two quarters of the rank^2 half (single contiguous source) and K2m adds the
//                   partner's partial into OUR half's fp32 stage (absolute-row offsets); B ships the stage's
//                   quarter rank^1 (fp32) and K2m adds the partner's A-sum of OUR quarter into the output.
//                   Per element that is (p0 + p2) + (p1 + p3): bitwise the all-reduce tree (a single f16 round
//                   at the end), because the canonical tree visits the bits in the same order (round 1 first).
//   all-gather:     A sends the own quarter to rank^2; B sends the own quarter AND the landed rank^2 quarter
//                   to rank^1 (two epochs, so no pack copy); every receive is tp_wait_copy_mb (bytes verbatim).
// A B epoch may read/consume only what its A epochs have landed, so the builder places B epochs at the first
// odd slot at least G + 1 slots after every A it depends on (bubbles elsewhere: a 16 B K1m + K2m(n = 4) keeps the
// round phase aligned), and ERRORS on any stream-order violation. At world > 2 the receiver validates the slot
// tail at its OWN expected length, so what rank r sends to p per epoch must equal what p expects from r: the
// pieces are a pure function of (c, widths), and the receiver's expected quarter is the sender's own quarter.
// =================================================================================================

/// The four ranks' row blocks of a `c`-row chunk.
pub type Blocks4 = [(usize, usize); 4];
pub fn row_blocks4(c: usize) -> Blocks4 { [row_block(c, 0, 4), row_block(c, 1, 4), row_block(c, 2, 4), row_block(c, 3, 4)] }

/// One epoch of a world-4 SP exchange: XEp plus the A epochs (indices into the A list) that must have been
/// RECEIVED (their K2 launched earlier in stream order) before this B epoch's K1m and K2m.
#[derive(Clone, Debug)]
struct PEp { x: XEp, deps: Vec<usize> }

/// Sub-piece `s` (0 = the floor-half first, 1 = the rest) of a piece.
fn split_half(p: (usize, usize), s: usize) -> (usize, usize) {
    let m = p.0 + (p.1 - p.0) / 2;
    if s == 0 { (p.0, m) } else { (m, p.1) }
}

/// Rows per epoch of a world-4 reduce-scatter whose A wire is `wb` bytes/element: the piece fits one slot at the A
/// width, and its half (a B epoch: fp32) fits one slot too.
fn rs4_rows_per_epoch(payload: usize, row_len: usize, wb: usize) -> usize {
    (payload / (row_len * 4).max(1)) * (4 / wb)
}

/// The reduce-scatter's A and B epoch lists for `rank` (see the section header). `stage` = the fp32 staging
/// buffer indexed by ABSOLUTE row (== io.src for an fp32 in-place source). `payload` = one slot's payload bytes.
fn rs4_epochs(io: &Io, row_len: usize, blocks: &Blocks4, rank: usize, stage: u64, payload: usize)
              -> Result<(Vec<PEp>, Vec<PEp>)> {
    anyhow::ensure!(rank < 4 && row_len % 8 == 0 && blocks.iter().all(|b| b.1 > b.0),
                    "TP-4S reduce-scatter: row_len {row_len} rank {rank} blocks {blocks:?}");
    let (se, oe, wb) = (if io.src_f16 { 2usize } else { 4 }, if io.out_f16 { 2usize } else { 4 }, if io.wire16() { 2usize } else { 4 });
    let rpe = rs4_rows_per_epoch(payload, row_len, wb);
    anyhow::ensure!(rpe >= 2, "TP-4S reduce-scatter: a {row_len}-element row is too wide for a half-slot fp32 piece (rpe {rpe})");
    let e = pieces_count(blocks, rpe);
    let pcs: Vec<Vec<(usize, usize)>> = blocks.iter().map(|&(a, b)| row_pieces(a, b, e)).collect();
    let (h, h2) = (rank >> 1, (rank ^ 2) >> 1);
    let (mut a, mut b) = (Vec::with_capacity(2 * e), Vec::with_capacity(2 * e));
    for p in 0..e {
        for pos in 0..2 {
            let ((s0, s1), (r0, r1)) = (pcs[2 * h2 + pos][p], pcs[2 * h + pos][p]);
            anyhow::ensure!(s1 > s0 && r1 > r0, "TP-4S reduce-scatter: empty A piece {p} (chunk too small for {e} pieces)");
            a.push(PEp { x: XEp {
                src: io.src + (s0 * row_len * se) as u64,
                bytes: (s1 - s0) * row_len * wb,
                k1bits: io.bits() & 5,
                rx: Rx::Add { out: stage + (r0 * row_len * 4) as u64, local: io.src + (r0 * row_len * se) as u64,
                              n: (r1 - r0) * row_len, bits: io.bits() & 5 },
                rows: 0,
            }, deps: vec![] });
        }
    }
    for p in 0..e {
        for s in 0..2 {
            let ((s0, s1), (r0, r1)) = (split_half(pcs[rank ^ 1][p], s), split_half(pcs[rank][p], s));
            anyhow::ensure!(s1 > s0 && r1 > r0, "TP-4S reduce-scatter: empty B piece {p}.{s}");
            b.push(PEp { x: XEp {
                src: stage + (s0 * row_len * 4) as u64,
                bytes: (s1 - s0) * row_len * 4,
                k1bits: 0,
                rx: Rx::Add { out: io.out + (r0 * row_len * oe) as u64, local: stage + (r0 * row_len * 4) as u64,
                              n: (r1 - r0) * row_len, bits: io.bits() & 2 },
                rows: r1 - r0,
            }, deps: vec![2 * p, 2 * p + 1] });
        }
    }
    Ok((a, b))
}

/// The all-gather's A and B epoch lists of ONE row buffer for `rank`.
fn ag4_epochs(base: u64, row_bytes: usize, blocks: &Blocks4, rank: usize, payload: usize) -> Result<(Vec<PEp>, Vec<PEp>)> {
    anyhow::ensure!(rank < 4 && row_bytes % 16 == 0 && blocks.iter().all(|b| b.1 > b.0),
                    "TP-4S all-gather: row_bytes {row_bytes} rank {rank} blocks {blocks:?}");
    let rpe = payload / row_bytes.max(1);
    anyhow::ensure!(rpe >= 1, "TP-4S all-gather: a {row_bytes}-byte row exceeds one ring slot");
    let e = pieces_count(blocks, rpe);
    let pcs: Vec<Vec<(usize, usize)>> = blocks.iter().map(|&(a, b)| row_pieces(a, b, e)).collect();
    let ep = |send: (usize, usize), land: (usize, usize), deps: Vec<usize>| -> Result<PEp> {
        anyhow::ensure!(send.1 > send.0 && land.1 > land.0, "TP-4S all-gather: empty piece (chunk too small for {e} pieces)");
        Ok(PEp { x: XEp {
            src: base + (send.0 * row_bytes) as u64,
            bytes: (send.1 - send.0) * row_bytes,
            k1bits: 0,
            rx: Rx::Copy { out: base + (land.0 * row_bytes) as u64, bytes: (land.1 - land.0) * row_bytes },
            rows: 0,
        }, deps })
    };
    let (mut a, mut b) = (Vec::with_capacity(e), Vec::with_capacity(2 * e));
    for p in 0..e { a.push(ep(pcs[rank][p], pcs[rank ^ 2][p], vec![])?); }
    for p in 0..e {
        b.push(ep(pcs[rank][p], pcs[rank ^ 1][p], vec![])?);
        b.push(ep(pcs[rank ^ 2][p], pcs[rank ^ 3][p], vec![p])?);
    }
    Ok((a, b))
}

/// Several row buffers as ONE exchange: A lists and B lists concatenated, B dependencies re-based.
fn concat_sp4(parts: Vec<(Vec<PEp>, Vec<PEp>)>) -> (Vec<PEp>, Vec<PEp>) {
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (pa, pb) in parts {
        let off = a.len();
        a.extend(pa);
        b.extend(pb.into_iter().map(|mut e| { for d in e.deps.iter_mut() { *d += off; } e }));
    }
    (a, b)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cell4 { A(usize), B(usize), Bubble }

/// Deal A epoch i to rail i % nr at slot 2 * (i / nr) and B epoch j to rail j % nr at the first ODD slot that is
/// >= G + 1 slots after each of its A dependencies (and after the rail's previous B). Every rail is padded to an
/// even length (R10: a multiple of `rounds`). Pure function of (#A, B deps, nr, g): identical on every rank.
fn place_sp4(a: &[PEp], b: &[PEp], nr: usize, g: usize) -> Result<Vec<Vec<Cell4>>> {
    anyhow::ensure!(nr >= 1 && g >= 2 && g % 2 == 0, "TP-4S placer: nr {nr} g {g}");
    let slot_a = |i: usize| 2 * (i / nr);
    let mut rails: Vec<Vec<Cell4>> = (0..nr).map(|r| (0..a.len()).filter(|i| i % nr == r).flat_map(|i| [Cell4::A(i), Cell4::Bubble]).collect()).collect();
    let mut cursor = vec![1usize; nr];
    for (j, eb) in b.iter().enumerate() {
        let r = j % nr;
        let mut s = cursor[r];
        for &d in &eb.deps {
            anyhow::ensure!(d < a.len(), "TP-4S placer: B epoch {j} depends on A epoch {d} of {}", a.len());
            s = s.max(slot_a(d) + g + 1);
        }
        if s % 2 == 0 { s += 1; }
        while rails[r].len() <= s { rails[r].push(Cell4::Bubble); }
        rails[r][s] = Cell4::B(j);
        cursor[r] = s + 2;
    }
    for rl in rails.iter_mut() { if rl.len() % 2 == 1 { rl.push(Cell4::Bubble); } }
    Ok(rails)
}

/// The launch list of a world-4 SP exchange, tick loop as `run_xchg`/`build_program` (tick t: per rail, the
/// receive of index t - G [lag = this rail's K1m launched after it], then K1m(t)). `hook_min`: the rows-landed
/// hook over the own rows (a reduce-scatter's B epochs, contiguous and increasing), handed over once >= min rows
/// have had their K2m launched, then the rest at the end. Errors unless every B epoch's K1m and K2m come AFTER
/// the K2 of each A epoch it depends on, every epoch fits one slot, and every rail keeps the round phase.
#[allow(clippy::too_many_arguments)]
fn build_sp4_program(a: &[PEp], b: &[PEp], nr: usize, g: usize, bub: u64, hook_min: Option<usize>, payload: usize)
                     -> Result<Vec<Step>> {
    for e in a.iter().chain(b.iter()) {
        anyhow::ensure!(e.x.bytes % 16 == 0 && e.x.bytes > 0 && e.x.bytes <= payload,
                        "TP-4S epoch of {} bytes: not a 16 B multiple within one ring slot payload ({payload})", e.x.bytes);
    }
    let rails = place_sp4(a, b, nr, g)?;
    emit_sp4_program(&rails, a, b, g, bub, hook_min)
}

/// `build_sp4_program`'s emitter over an explicit placement (a test can hand it an adversarial one: the
/// dependency check below is what makes a bad placement an error rather than a silent race).
fn emit_sp4_program(rails: &[Vec<Cell4>], a: &[PEp], b: &[PEp], g: usize, bub: u64, hook_min: Option<usize>)
                    -> Result<Vec<Step>> {
    anyhow::ensure!(rails.iter().all(|r| r.len() % 2 == 0), "TP-4S: a rail would break the round phase");
    let ep_of = |c: Cell4| -> Option<&XEp> { match c { Cell4::A(i) => Some(&a[i].x), Cell4::B(j) => Some(&b[j].x), Cell4::Bubble => None } };
    let mut k2_tick = vec![usize::MAX; b.len()];
    for sq in rails { for (s, c) in sq.iter().enumerate() { if let Cell4::B(j) = *c { k2_tick[j] = s + g; } } }
    anyhow::ensure!(k2_tick.iter().all(|&t| t != usize::MAX), "TP-4S: a B epoch was never placed");
    let total_rows: usize = b.iter().map(|e| e.x.rows).sum();
    let longest = rails.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut prog: Vec<Step> = Vec::new();
    let (mut pos_k2a, mut pos_k1b, mut pos_k2b) = (vec![usize::MAX; a.len()], vec![usize::MAX; b.len()], vec![usize::MAX; b.len()]);
    let (mut m, mut rows_fin, mut rows_done) = (0usize, 0usize, 0usize);
    for t in 0..longest + g {
        for (r, sq) in rails.iter().enumerate() {
            if t >= g && t - g < sq.len() {
                let j = t - g;
                let lag = (t.min(sq.len()) - 1 - j) as u32;
                match ep_of(sq[j]) {
                    Some(xe) => prog.push(match xe.rx {
                        Rx::Add { out, local, n, bits } => Step::K2m { rail: r, out, local, n: n as i32, lag, bits },
                        Rx::Copy { out, bytes } => Step::K2c { rail: r, out, bytes: bytes as u32, lag },
                    }),
                    None => prog.push(Step::K2m { rail: r, out: bub, local: bub, n: 4, lag, bits: 0 }),
                }
                match sq[j] {
                    Cell4::A(i) => pos_k2a[i] = prog.len() - 1,
                    Cell4::B(jj) => pos_k2b[jj] = prog.len() - 1,
                    Cell4::Bubble => {}
                }
            }
            if t < sq.len() {
                match ep_of(sq[t]) {
                    Some(xe) => prog.push(Step::K1m { rail: r, src: xe.src, nbytes: xe.bytes as u32, bits: xe.k1bits }),
                    None => prog.push(Step::K1m { rail: r, src: bub, nbytes: 16, bits: 0 }),
                }
                if let Cell4::B(jj) = sq[t] { pos_k1b[jj] = prog.len() - 1; }
            }
        }
        if let (true, Some(min_rows)) = (t >= g, hook_min) {
            while m < b.len() && k2_tick[m] <= t { rows_fin += b[m].x.rows; m += 1; }
            if rows_fin > rows_done && (rows_fin - rows_done >= min_rows || rows_fin == total_rows) {
                prog.push(Step::Hook { r0: rows_done, r1: rows_fin, tail: false });
                rows_done = rows_fin;
            }
        }
    }
    if hook_min.is_some() && total_rows > rows_done { prog.push(Step::Hook { r0: rows_done, r1: total_rows, tail: true }); }
    for (j, eb) in b.iter().enumerate() {
        for &d in &eb.deps {
            anyhow::ensure!(pos_k2a[d] != usize::MAX && pos_k2a[d] < pos_k1b[j] && pos_k2a[d] < pos_k2b[j],
                            "TP-4S: B epoch {j} would run before A epoch {d} has landed (stream order)");
        }
    }
    Ok(prog)
}

/// The world-4 reduce-scatter's launch program for `rank` (pure; `reduce_scatter_rows_w4` launches exactly this).
/// Hook rows are ABSOLUTE rows of the own quarter.
#[allow(clippy::too_many_arguments)]
fn sp4_rs_program(io: &Io, row_len: usize, blocks: &Blocks4, rank: usize, nr: usize, g: usize, bub: u64,
                  hook_min: Option<usize>, payload: usize) -> Result<Vec<Step>> {
    let c = blocks[3].1;
    let stage = stage_for(io, c * row_len)?;
    let (a, b) = rs4_epochs(io, row_len, blocks, rank, stage, payload)?;
    let mut prog = build_sp4_program(&a, &b, nr, g, bub, hook_min, payload)?;
    for st in prog.iter_mut() {
        if let Step::Hook { r0, r1, .. } = st { *r0 += blocks[rank].0; *r1 += blocks[rank].0; }
    }
    Ok(prog)
}

/// The world-4 all-gather's launch program of several row buffers (pure).
fn sp4_ag_program(bufs: &[(u64, usize)], blocks: &Blocks4, rank: usize, nr: usize, g: usize, bub: u64, payload: usize)
                  -> Result<Vec<Step>> {
    anyhow::ensure!(!bufs.is_empty(), "TP-4S all-gather: no buffers");
    let parts = bufs.iter().map(|&(base, rb)| ag4_epochs(base, rb, blocks, rank, payload)).collect::<Result<Vec<_>>>()?;
    let (a, b) = concat_sp4(parts);
    build_sp4_program(&a, &b, nr, g, bub, None, payload)
}

/// Launch a world-4 SP program (K1m / K2m / K2c / Hook). Dual stream exactly as `run_xchg`: with a `side` hook the
/// schedule runs on the Pipe's own blocking stream after everything already on the caller's stream; each hook range
/// is handed to the caller's stream by an event recorded after the K2's launched so far; the caller joins the
/// whole schedule at the end.
fn exec_sp4(k: &Kernels, stream: &CudaStream, x: &Pipe, prog: &[Step], mut hook: Option<SpHook<'_>>) -> Result<()> {
    let caller = stream;
    let dual = hook.as_ref().map_or(false, |h| h.side);
    let comm: &CudaStream = if dual { &x.side.stream } else { stream };
    if dual { stream_after(comm, caller, x.side.ev_in)?; }
    let cfg = LaunchConfig { grid_dim: (x.blocks, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
    let arrive = *x.arrive.device_ptr() as u64;
    for st in prog {
        match *st {
            Step::K1m { rail, src, nbytes, bits } => {
                unsafe { k.k1m.clone().launch_on_stream(comm, cfg, (x.rails[rail], src, nbytes, arrive, bits)) }
                    .context("TP-4S K1m launch")?;
            }
            Step::K2m { rail, out, local, n, lag, bits } => {
                unsafe { k.k2m.clone().launch_on_stream(comm, cfg, (x.rails[rail], out, local, n, lag, bits)) }
                    .context("TP-4S K2m launch")?;
            }
            Step::K2c { rail, out, bytes, lag } => {
                unsafe { k.k2c.clone().launch_on_stream(comm, cfg, (x.rails[rail], out, bytes, lag)) }
                    .context("TP-4S copy-K2 launch")?;
            }
            Step::Hook { r0, r1, .. } => {
                let h = hook.as_mut().ok_or_else(|| anyhow::anyhow!("TP-4S: a hook step without a hook"))?;
                if dual { stream_after(caller, comm, x.side.ev_out)?; }
                (h.f)(r0, r1)?;
            }
            Step::Fold { .. } => anyhow::bail!("TP-4S programs have no fold step"),
        }
    }
    if dual { stream_after(caller, comm, x.side.ev_out)?; }
    Ok(())
}

/// TP-4S world-4 REDUCE-SCATTER of a `c`-row partial: this rank ends with the canonical sum in its own QUARTER
/// of io.out only (the other rows of io.out are not written), bitwise the all-reduce's value on those rows.
/// `io.mid` must be an fp32 staging buffer indexed by absolute row (or io.src itself for an fp32 source).
#[allow(clippy::too_many_arguments)]
pub fn reduce_scatter_rows_w4(k: &Kernels, stream: &CudaStream, x: &Pipe, io: Io, row_len: usize, blocks: &Blocks4,
                              rank: usize, hook: Option<SpHook<'_>>) -> Result<()> {
    anyhow::ensure!(x.rounds == 2, "TP-4S reduce-scatter: world 4 only (rounds {})", x.rounds);
    let bub = *x.bubble.device_ptr() as u64;
    let prog = sp4_rs_program(&io, row_len, blocks, rank, x.rails.len(), x.lookahead, bub,
                              hook.as_ref().map(|h| h.min_rows), EPOCH_FLOATS * 4)?;
    exec_sp4(k, stream, x, &prog, hook)
}

/// TP-4S world-4 ALL-GATHER of `row_bytes`-byte rows at `base` (every rank's own quarter goes to the other three,
/// bit for bit).
pub fn all_gather_rows_w4(k: &Kernels, stream: &CudaStream, x: &Pipe, base: u64, row_bytes: usize, blocks: &Blocks4,
                          rank: usize) -> Result<()> {
    all_gather_rows_multi_w4(k, stream, x, &[(base, row_bytes)], blocks, rank)
}

/// TP-4S: the world-4 all-gather of several row buffers as ONE exchange program.
pub fn all_gather_rows_multi_w4(k: &Kernels, stream: &CudaStream, x: &Pipe, bufs: &[(u64, usize)], blocks: &Blocks4,
                                rank: usize) -> Result<()> {
    anyhow::ensure!(x.rounds == 2, "TP-4S all-gather: world 4 only (rounds {})", x.rounds);
    let bub = *x.bubble.device_ptr() as u64;
    let prog = sp4_ag_program(bufs, blocks, rank, x.rails.len(), x.lookahead, bub, EPOCH_FLOATS * 4)?;
    exec_sp4(k, stream, x, &prog, None)
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

    /// CPU model of the `reduce_serial` device schedule over `world` ranks in lockstep. `e0` = the device
    /// epoch counter before the reduce (K1 pre-increments, so the first pair is epoch e0 + 1); epoch e runs
    /// round e % rounds with partner rank ^ (1 << round); K2 mode 2 computes (lower half's value) +
    /// (upper half's value) in fp32, lower = the rank whose bit `round` is 0, in place (no rounding
    /// between rounds). `pairs` = K1/K2 pairs per chunk (`rounds` = the real schedule; 1 models the
    /// pre-TP-4B single pair). Returns each rank's buffer and the counter after the reduce.
    fn serial_model(world: usize, e0: u64, per: usize, pairs: usize, parts: &[Vec<f32>]) -> (Vec<Vec<f32>>, u64) {
        serial_model_plan(world, e0, &serial_plan_of(parts[0].len(), per, pairs), parts)
    }

    fn serial_model_plan(world: usize, e0: u64, plan: &[SerialPair], parts: &[Vec<f32>]) -> (Vec<Vec<f32>>, u64) {
        let rounds = rounds_of(world).unwrap() as u64;
        let mut cur: Vec<Vec<f32>> = parts.to_vec();
        let mut e = e0;
        for &SerialPair { off, len, .. } in plan {
            e += 1;
            let k = (e % rounds) as usize;
            let prev: Vec<Vec<f32>> = cur.iter().map(|v| v[off..off + len].to_vec()).collect();
            for r in 0..world {
                let peer = r ^ (1 << k);
                let lower = r & (1 << k) == 0;
                for i in 0..len {
                    let (a, b) = if lower { (prev[r][i], prev[peer][i]) } else { (prev[peer][i], prev[r][i]) };
                    cur[r][off + i] = a + b;
                }
            }
        }
        (cur, e)
    }

    /// The fixed-order reference, written as a tree rather than a schedule: after round j (0-based, in
    /// the order the epoch phase visits the bits) a rank holds tree(lower half) + tree(upper half) over
    /// the bit just exchanged. The bit order starts at (e0 + 1) % rounds.
    fn tree_reference(world: usize, e0: u64, parts: &[Vec<f32>]) -> Vec<f32> {
        let rounds = rounds_of(world).unwrap();
        let first = ((e0 + 1) % rounds as u64) as usize;
        let order: Vec<usize> = (0..rounds).map(|j| (first + j) % rounds).collect();
        fn tree(r: usize, j: usize, order: &[usize], parts: &[Vec<f32>], i: usize) -> f32 {
            if j == 0 { return parts[r][i]; }
            let bit = 1usize << order[j - 1];
            tree(r & !bit, j - 1, order, parts, i) + tree(r | bit, j - 1, order, parts, i)
        }
        (0..parts[0].len()).map(|i| tree(0, rounds, &order, parts, i)).collect()
    }

    /// Partials with mixed magnitudes and cancellation, so different associations differ in bits.
    fn parts_for(world: usize, n: usize) -> Vec<Vec<f32>> {
        (0..world).map(|r| (0..n).map(|i| {
            let h = (i as u64).wrapping_mul(2654435761).wrapping_add(r as u64 * 40503 + 12345) % 2_000_003;
            let v = (h as f32 - 1_000_001.0) * 1.37e-4;
            match (i + r) % 5 { 0 => v * 1e6, 1 => -v * 1e6 + 0.3, 2 => v * 1e-6, _ => v }
        }).collect()).collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() }

    #[test]
    fn rounds_of_powers_of_two_only() {
        assert_eq!(rounds_of(2).unwrap(), 1);
        assert_eq!(rounds_of(4).unwrap(), 2);
        assert_eq!(rounds_of(8).unwrap(), 3);
        for w in [0usize, 1, 3, 5, 6, 12] { assert!(rounds_of(w).is_err(), "world {w}"); }
    }

    /// World 2: rounds == 1, the single pair is rank0 + rank1 on both ranks, at every epoch phase, every
    /// chunking — the exact result of today's reduce_serial.
    #[test]
    fn serial_w2_is_the_canonical_pair_sum() {
        let parts = parts_for(2, 1000);
        let want: Vec<f32> = (0..1000).map(|i| parts[0][i] + parts[1][i]).collect();
        for e0 in 0..12u64 {
            for per in [1usize, 7, 256, 1000] {
                let (out, e) = serial_model(2, e0, per, 1, &parts);
                assert_eq!(e, e0 + epochs_of(1000, per).len() as u64, "one epoch per chunk at world 2");
                assert_eq!(bits(&out[0]), bits(&want), "e0 {e0} per {per} rank 0");
                assert_eq!(bits(&out[1]), bits(&want), "e0 {e0} per {per} rank 1");
            }
        }
    }

    /// World 4: every rank ends bit-identical, equal to the tree reference for the epoch phase, for every
    /// start epoch and chunking; a chunk consumes exactly `rounds` epochs so the phase never drifts. The
    /// realised association is (p0+p2)+(p1+p3) when the counter is even (first pair = odd epoch = round 1)
    /// and (p0+p1)+(p2+p3) when it is odd.
    #[test]
    fn serial_w4_bit_identical_fixed_order() {
        let n = 1500;
        let parts = parts_for(4, n);
        let a_01_23: Vec<f32> = (0..n).map(|i| (parts[0][i] + parts[1][i]) + (parts[2][i] + parts[3][i])).collect();
        let a_02_13: Vec<f32> = (0..n).map(|i| (parts[0][i] + parts[2][i]) + (parts[1][i] + parts[3][i])).collect();
        assert_ne!(bits(&a_01_23), bits(&a_02_13), "test data cannot tell the two associations apart");
        for e0 in 0..10u64 {
            for per in [1usize, 7, 256, n] {
                let (out, e) = serial_model(4, e0, per, 2, &parts);
                assert_eq!(e, e0 + 2 * epochs_of(n, per).len() as u64, "two epochs per chunk at world 4");
                assert_eq!(e % 2, e0 % 2, "the round phase drifted");
                let want = tree_reference(4, e0, &parts);
                for r in 0..4 { assert_eq!(bits(&out[r]), bits(&want), "e0 {e0} per {per} rank {r} != the tree reference"); }
                let closed = if e0 % 2 == 0 { &a_02_13 } else { &a_01_23 };
                assert_eq!(bits(&want), bits(closed), "e0 {e0}: the tree reference is not the closed form");
            }
        }
    }

    /// The silent-wrong-bytes trap, pinned: the pre-TP-4B schedule (ONE pair per chunk) at world 4 leaves
    /// each rank with only its round partner's pair sum — wrong, and different across ranks. rounds-aware
    /// pairs fix it.
    #[test]
    fn serial_w4_single_pair_is_the_trap() {
        let n = 64;
        let parts = parts_for(4, n);
        let (one, _) = serial_model(4, 0, n, 1, &parts);
        let want = tree_reference(4, 0, &parts);
        assert_ne!(bits(&one[0]), bits(&want));
        assert_ne!(bits(&one[0]), bits(&one[1]), "ranks 0 and 1 must disagree after a single round");
        let (two, _) = serial_model(4, 0, n, 2, &parts);
        for r in 0..4 { assert_eq!(bits(&two[r]), bits(&want)); }
    }

    /// The plan `reduce_serial` really launches: every chunk appears once per round, rounds consecutive,
    /// chunks tile [0, n) in order, each within one ring slot; the launch arguments address the chunk.
    #[test]
    fn serial_plan_shape_and_args() {
        for rounds in [1usize, 2, 3] {
            for n in [1usize, 4, EPOCH_FLOATS - 4, EPOCH_FLOATS, EPOCH_FLOATS + 4, 3 * EPOCH_FLOATS + 28] {
                let plan = serial_plan(n, rounds);
                let chunks = epochs(n).len();
                assert_eq!(plan.len(), chunks * rounds, "n {n} rounds {rounds}");
                let mut at = 0usize;
                for (i, sp) in plan.iter().enumerate() {
                    assert_eq!(sp.round, i % rounds);
                    assert_eq!(sp.off, at, "chunk offset");
                    assert!(sp.len > 0 && sp.len * 4 + 64 <= crate::tp::TP_SLOT_BYTES);
                    assert_eq!(sp.args(0x1000), (0x1000 + (sp.off * 4) as u64, (sp.len * 4) as u32, sp.len as i32));
                    if sp.round == rounds - 1 { at += sp.len; }
                }
                assert_eq!(at, n);
            }
        }
    }

    /// The plan the real loop iterates (production chunking, several chunks + a ragged tail), run through
    /// the CPU schedule model at world 2 and 4 from an even and an odd start epoch: all ranks bit-identical,
    /// equal to the tree reference, phase preserved. A regression that drops the per-chunk rounds in
    /// `serial_plan` fails here (the world-4 single-pair trap).
    #[test]
    fn real_serial_plan_reduces_correctly() {
        let n = 2 * EPOCH_FLOATS + 1012;
        for world in [2usize, 4] {
            let rounds = rounds_of(world).unwrap();
            let parts = parts_for(world, n);
            for e0 in [0u64, 1] {
                let plan = serial_plan(n, rounds);
                let (out, e) = serial_model_plan(world, e0, &plan, &parts);
                assert_eq!(e, e0 + (rounds * epochs(n).len()) as u64);
                let want = tree_reference(world, e0, &parts);
                for r in 0..world { assert_eq!(bits(&out[r]), bits(&want), "world {world} e0 {e0} rank {r}"); }
            }
        }
    }

    /// The serial schedule's recv-slot safety at world 2 and 4, from the happens-before DAG: per rank the
    /// stream order K1(e) -> K2(e) -> K1(e+1); K2(e) on rank r needs K1(e) of partner(e). A recv slot is
    /// (round, e & (R-1)) at world > 2 (e & (R-1) at world 2), so epochs e and e - R share a slot; the
    /// write of epoch e into rank r's slot (partner's K1(e)) must be ordered AFTER rank r's K2(e - R) read.
    #[test]
    fn serial_schedule_recv_slot_safety() {
        for world in [2usize, 4] {
            let rounds = rounds_of(world).unwrap();
            let last = 6 * RING_SLOTS as u64;
            let id = |r: usize, e: u64, k2: bool| ((e as usize) * world + r) * 2 + k2 as usize;
            let nodes = (last as usize + 1) * world * 2;
            let partner = |r: usize, e: u64| r ^ (1usize << (e as usize % rounds));
            let mut succ: Vec<Vec<usize>> = vec![Vec::new(); nodes];
            for r in 0..world {
                for e in 1..=last {
                    succ[id(r, e, false)].push(id(r, e, true));
                    if e < last { succ[id(r, e, true)].push(id(r, e + 1, false)); }
                    succ[id(partner(r, e), e, false)].push(id(r, e, true));
                }
            }
            let reaches = |from: usize, to: usize| {
                let mut seen = vec![false; nodes];
                let mut st = vec![from];
                seen[from] = true;
                while let Some(x) = st.pop() {
                    if x == to { return true; }
                    for &y in &succ[x] { if !seen[y] { seen[y] = true; st.push(y); } }
                }
                false
            };
            for r in 0..world {
                for e in (RING_SLOTS as u64 + 1)..=last {
                    let writer = id(partner(r, e), e, false);
                    let reader_prev = id(r, e - RING_SLOTS as u64, true);
                    assert!(reaches(reader_prev, writer),
                            "world {world}: rank {r} epoch {e}: the slot write is not ordered after the read of epoch {}",
                            e - RING_SLOTS as u64);
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // TP-4C: the decode-path arms (folded mixer / MoE reduce, vocab-parallel key merge, sharded
    // draft-head screen) as a CPU model of the device protocol, over the SAME `dec_exchanges` /
    // `dec_exchange_args` the xtp.rs launch code iterates. What the model PROVES: for every
    // interleaving of the arms with each other and with the prefill serial reduce, every rank consumes
    // the same number of epochs, the R10 phase never drifts, every rank ends bit-identical, and the
    // value is the expected tree (sums) / the global max (keys) / the exact union (screen). What it
    // does NOT prove: the device kernels' own reads of c->epoch / recv-slot addressing, PDL ordering
    // between the plain K1 and the PDL-launched K2, the proxy and the wire, or any hardware behaviour
    // — those are the hardware gates in PLAN/TP-4C_GATES.md.
    // ---------------------------------------------------------------------------------------------
    use std::collections::HashMap;
    type Mem = HashMap<u64, Vec<f32>>;

    const PART: u64 = 0x1000; // the mixer / MoE fp32 partial buffer (producer rewrites it whole)
    const YP: u64 = 0x2000;   // the f16 output buffer (y_raw / moe_out)
    const DHL: u64 = 0x3000;  // the draft-head screen's zero-padded block-maxima buffer (persistent)
    const DHO: u64 = 0x4000;  // the screen's fp32 output

    fn f16r(v: f32) -> f32 { half::f16::from_f32(v).to_f32() }

    /// One Sum decode arm over `world` ranks in lockstep, executing the caller's exchange list exactly as
    /// xtp.rs does: K1 publishes `local` (round 0, the producer's folded K1) or the fp32 intermediate
    /// (plain K1), K2 combines lower + upper per `dec_exchange_args`, rounding to f16 only when the
    /// resolved K2 says so. The published snapshot is taken before any rank's K2 writes (the recv slot).
    fn model_dec_sum(world: usize, e: &mut u64, xs: &[DecExchange], mem: &mut [Mem], local: u64, local_f16: bool,
                     out: u64, out_f16: bool) {
        let rounds = rounds_of(world).unwrap() as u64;
        let inter = dec_intermediate(local, out, out_f16);
        for x in xs {
            *e += 1;
            let k = (*e % rounds) as usize;
            let pub_addr = if x.folded_k1 { local } else { inter };
            let published: Vec<Vec<f32>> = (0..world).map(|r| mem[r][&pub_addr].clone()).collect();
            let a = dec_exchange_args(x, local, local_f16, out, out_f16);
            for r in 0..world {
                let (peer, lower) = (r ^ (1 << k), r & (1 << k) == 0);
                let mine = mem[r][&a.local].clone();
                let res: Vec<f32> = (0..mine.len()).map(|i| {
                    let (p, q) = if lower { (mine[i], published[peer][i]) } else { (published[peer][i], mine[i]) };
                    let s = p + q;
                    if a.out_f16 { f16r(s) } else { s }
                }).collect();
                mem[r].insert(a.out, res);
            }
        }
    }

    /// The vocab-parallel key merge: every exchange is xq_tp_wait_keys (keys[r] = max(own, peer) in place,
    /// ids = id(merged key)), the plain K1 of rounds > 0 republishes the in-place merged keys.
    fn model_dec_keys(world: usize, e: &mut u64, xs: &[DecExchange], keys: &mut [Vec<u64>]) -> Vec<Vec<u32>> {
        let rounds = rounds_of(world).unwrap() as u64;
        let mut ids: Vec<Vec<u32>> = vec![Vec::new(); world];
        for _ in xs {
            *e += 1;
            let k = (*e % rounds) as usize;
            let published: Vec<Vec<u64>> = keys.to_vec();
            for r in 0..world {
                let peer = r ^ (1 << k);
                for i in 0..keys[r].len() { keys[r][i] = keys[r][i].max(published[peer][i]); }
                ids[r] = keys[r].iter().map(|&key| 0xFFFF_FFFFu32 - (key & 0xFFFF_FFFF) as u32).collect();
            }
        }
        ids
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Op { Mixer(usize), Moe(usize), Vp(usize), Dh, Prefill(usize) }

    fn salted(world: usize, n: usize, salt: usize) -> Vec<Vec<f32>> {
        parts_for(world, n).into_iter().map(|v| v.into_iter().map(|x| x * (1.0 + salt as f32 * 0.37)).collect()).collect()
    }

    const NBLK: usize = 512;

    /// Run `ops` from device epoch `e0` at `world` with the decode exchange list chosen by `exch`. Checks,
    /// after every op: the epoch phase (e % rounds == e0 % rounds), rank identity and the expected value.
    /// Returns (final epoch, number of ops verified) or the first violation.
    fn run_ops(world: usize, e0: u64, ops: &[Op], exch: &dyn Fn(usize) -> Vec<DecExchange>) -> Result<(u64, usize), String> {
        let rounds = rounds_of(world).unwrap();
        let mut e = e0;
        let mut mem: Vec<Mem> = (0..world).map(|_| Mem::new()).collect();
        for r in 0..world { mem[r].insert(DHL, vec![0.0f32; NBLK]); }
        let grid = NBLK / world;
        let mut verified = 0usize;
        for (idx, op) in ops.iter().enumerate() {
            let e_before = e;
            match *op {
                Op::Mixer(n) | Op::Moe(n) => {
                    let parts = salted(world, n, idx);
                    for r in 0..world { mem[r].insert(PART, parts[r].clone()); }
                    model_dec_sum(world, &mut e, &exch(rounds), &mut mem, PART, false, YP, true);
                    let want: Vec<u32> = rd_tree_reference(world, e_before, &parts).unwrap().iter().map(|&x| f16r(x).to_bits()).collect();
                    for r in 0..world {
                        if bits(&mem[r][&YP]) != want { return Err(format!("op {idx} {op:?} rank {r}: f16 sum != the tree reference")); }
                    }
                }
                Op::Dh => {
                    // each rank refills ONLY its own span (the screen kernel), then the reduce
                    let mut truth = vec![0.0f32; NBLK];
                    for r in 0..world {
                        let buf = mem[r].get_mut(&DHL).unwrap();
                        for b in r * grid..(r + 1) * grid {
                            let v = ((b as u32).wrapping_mul(2654435761).wrapping_add(idx as u32 * 97) % 10007) as f32 * 0.013;
                            buf[b] = v;
                            truth[b] = v;
                        }
                    }
                    model_dec_sum(world, &mut e, &exch(rounds), &mut mem, DHL, false, DHO, false);
                    for r in 0..world {
                        if bits(&mem[r][&DHO]) != bits(&truth) { return Err(format!("op {idx} Dh rank {r}: the screen != the exact union")); }
                        let own_only = mem[r][&DHL].iter().enumerate().all(|(b, &v)| b / grid == r || v == 0.0);
                        if !own_only { return Err(format!("op {idx} Dh rank {r}: the zero-padded screen buffer was polluted")); }
                    }
                }
                Op::Vp(m) => {
                    let mut keys: Vec<Vec<u64>> = (0..world).map(|r| (0..m).map(|i| {
                        let h = ((i * 31 + r * 7 + idx * 13 + 5) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                        (h & 0x7FFF_FFFF_0000_0000) | (0xFFFF_FFFFu64 - ((r * 62080 + i * 3 + idx) as u64 & 0xFFFF))
                    }).collect()).collect();
                    let want: Vec<u64> = (0..m).map(|i| (0..world).map(|r| keys[r][i]).max().unwrap()).collect();
                    let want_ids: Vec<u32> = want.iter().map(|&k| 0xFFFF_FFFFu32 - (k & 0xFFFF_FFFF) as u32).collect();
                    let ids = model_dec_keys(world, &mut e, &exch(rounds), &mut keys);
                    for r in 0..world {
                        if keys[r] != want || ids[r] != want_ids { return Err(format!("op {idx} Vp rank {r}: merged key != the global max")); }
                    }
                }
                Op::Prefill(n) => {
                    let parts = salted(world, n, idx);
                    let (out, e2) = serial_model_plan(world, e, &serial_plan(n, rounds), &parts);
                    e = e2;
                    let want = rd_tree_reference(world, e_before, &parts).unwrap();
                    for r in 0..world {
                        if bits(&out[r]) != bits(&want) { return Err(format!("op {idx} Prefill({n}) rank {r}: != the tree reference")); }
                    }
                }
            }
            if e % rounds as u64 != e0 % rounds as u64 {
                return Err(format!("op {idx} {op:?}: the round phase drifted (epoch {e0} -> {e}, rounds {rounds})"));
            }
            verified += 1;
        }
        Ok((e, verified))
    }

    fn real_exch(rounds: usize) -> Vec<DecExchange> { dec_exchanges(rounds) }

    /// The exchange list shape: rounds exchanges, only the first is the producer's folded K1 and reads
    /// the caller's local, only the last writes the caller's out; one K1 and one K2 per exchange, so the
    /// logical reduce consumes EXACTLY `rounds` epochs. rounds == 1 is the single TP-G exchange.
    #[test]
    fn dec_exchanges_shape() {
        for rounds in 1..=4usize {
            let xs = dec_exchanges(rounds);
            assert_eq!(xs.len(), rounds);
            for (k, x) in xs.iter().enumerate() {
                assert_eq!(x.round, k);
                assert_eq!(x.folded_k1, k == 0);
                assert_eq!(x.local_is_input, k == 0);
                assert_eq!(x.writes_final, k + 1 == rounds);
            }
        }
        assert_eq!(dec_exchanges(1), vec![DecExchange { round: 0, folded_k1: true, local_is_input: true, writes_final: true }]);
    }

    /// World 2 no-op proof: at rounds == 1 the single exchange's K2 operands are exactly the TP-G launch
    /// (out, local, local_f16, out_f16) for every flag combination and any addresses — the multi-round
    /// helper changes nothing at world 2 (and world 1 never attaches).
    #[test]
    fn dec_exchange_args_w2_is_the_tpg_launch() {
        let xs = dec_exchanges(1);
        assert_eq!(xs.len(), 1);
        for (lf, of) in [(false, false), (false, true), (true, false), (true, true)] {
            let a = dec_exchange_args(&xs[0], 0xAAA0, lf, 0xBBB0, of);
            assert_eq!(a, DecK2Args { out: 0xBBB0, local: 0xAAA0, local_f16: lf, out_f16: of }, "local_f16 {lf} out_f16 {of}");
        }
    }

    /// Multi-round operand rules: the intermediate between rounds is always fp32 and lives in `out` when
    /// `out` is fp32 (the screen keeps its zero pad) else in `local` (mixer / MoE partial); only round 0
    /// reads the caller's local and only the last round writes the caller's out.
    #[test]
    fn dec_exchange_args_multi_round_rules() {
        for rounds in [2usize, 3] {
            let xs = dec_exchanges(rounds);
            for out_f16 in [false, true] {
                let (local, out) = (0x1000u64, 0x2000u64);
                let inter = dec_intermediate(local, out, out_f16);
                assert_eq!(inter, if out_f16 { local } else { out });
                for x in &xs {
                    let a = dec_exchange_args(x, local, false, out, out_f16);
                    assert_eq!(a.local, if x.round == 0 { local } else { inter });
                    assert!(!a.local_f16, "an f16 operand between rounds would round the partial sums");
                    if x.round + 1 == rounds {
                        assert_eq!((a.out, a.out_f16), (out, out_f16), "the last round writes the caller's out");
                    } else {
                        assert_eq!((a.out, a.out_f16), (inter, false), "an intermediate round writes the fp32 intermediate");
                    }
                }
            }
        }
    }

    /// The public tree reference against the closed forms and the test-local `tree_reference` (world 8
    /// included): the association is (p0+p2)+(p1+p3) at an even epoch and (p0+p1)+(p2+p3) at an odd one.
    #[test]
    fn rd_tree_reference_closed_forms() {
        let n = 400;
        let p4 = parts_for(4, n);
        let c02: Vec<f32> = (0..n).map(|i| (p4[0][i] + p4[2][i]) + (p4[1][i] + p4[3][i])).collect();
        let c01: Vec<f32> = (0..n).map(|i| (p4[0][i] + p4[1][i]) + (p4[2][i] + p4[3][i])).collect();
        assert_ne!(bits(&c02), bits(&c01));
        assert_eq!(bits(&rd_tree_reference(4, 0, &p4).unwrap()), bits(&c02));
        assert_eq!(bits(&rd_tree_reference(4, 1, &p4).unwrap()), bits(&c01));
        assert_eq!(bits(&rd_tree_reference(4, 6, &p4).unwrap()), bits(&c02));
        let p2 = parts_for(2, n);
        let c: Vec<f32> = (0..n).map(|i| p2[0][i] + p2[1][i]).collect();
        for e0 in 0..4u64 { assert_eq!(bits(&rd_tree_reference(2, e0, &p2).unwrap()), bits(&c)); }
        let p8 = parts_for(8, n);
        for e0 in 0..6u64 { assert_eq!(bits(&rd_tree_reference(8, e0, &p8).unwrap()), bits(&tree_reference(8, e0, &p8))); }
        assert!(rd_tree_reference(4, 0, &p2).is_err(), "world/partials mismatch must be refused");
        assert!(rd_tree_reference(3, 0, &p2).is_err());
    }

    /// TP-4F2: a deterministic fp32 stream spanning many magnitudes, signs, exact zeros and DENORMALS (never NaN/inf):
    /// the values a single-stage vs two-stage sum could differ on if the association were wrong.
    fn tp4f2_partials(n: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut st = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut nxt = move || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
        (0..4).map(|_| (0..n).map(|_| {
            let r = nxt();
            let m = ((r >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0;             // [-1, 1)
            match (r >> 3) & 15 {
                0 => 0.0,
                1 => f32::from_bits(((r >> 20) as u32) & 0x007f_ffff),                // a denormal (or zero)
                2 => -f32::from_bits(((r >> 20) as u32) & 0x007f_ffff),
                3 => m * 1.0e-30,
                4 => m * 1.0e4,
                5 => m * 65504.0,                                                       // f16 max-ish: the final rounding bites
                _ => m * (1u32 << ((r >> 40) as u32 % 20)) as f32 * 1.0e-3,
            }
        }).collect()).collect()
    }

    /// TP-4F2 (F1, the CPU-only bit-identity proof): the single-stage grouping equals recursive doubling bit for bit.
    /// Three views of rd are checked against `single_tree_reference` for random fp32 partials incl. denormals —
    /// (a) `rd_tree_reference` at every EVEN start epoch (the R10-aligned reduce), (b) an explicit PER-RANK rd
    /// dataflow (stage 0 `own + partner(rank^2)`, stage 1 `own + partner(rank^1)`, each rank adding its OWN operand
    /// on the left, as the kernel does), (c) the single-stage per-rank evaluation (rank r takes its own partial as
    /// operand r, `(v0+v2)+(v1+v3)`) — for ALL 24 assignments of the four partials to the four ranks, every rank, and
    /// the final single f16 rounding. A negative control: the (p0+p1)+(p2+p3) grouping DOES differ (so the test can fail).
    #[test]
    fn single_stage_grouping_equals_rd_bit_for_bit() {
        let n = 4096;
        let mut perms: Vec<[usize; 4]> = Vec::new();
        for a in 0..4 { for b in 0..4 { for c in 0..4 { for d in 0..4 {
            let v = [a, b, c, d];
            if (0..4).all(|k| v.contains(&k)) { perms.push(v); }
        }}}}
        assert_eq!(perms.len(), 24);
        let mut differing_ctl = 0usize;
        for seed in 1..=5u64 {
            let base = tp4f2_partials(n, seed);
            for perm in &perms {
                let parts: Vec<Vec<f32>> = (0..4).map(|r| base[perm[r]].clone()).collect();   // rank r holds partial perm[r]
                let want = single_tree_reference(&parts).unwrap();
                for e0 in [0u64, 2, 4, 6, 100] {
                    assert_eq!(bits(&rd_tree_reference(4, e0, &parts).unwrap()), bits(&want), "rd tree vs single, e0 {e0}");
                }
                for rank in 0..4usize {
                    for i in 0..n {
                        // explicit rd dataflow of ONE rank, own operand first
                        let s0 = |r: usize| parts[r][i] + parts[r ^ 2][i];
                        let rd = s0(rank) + s0(rank ^ 1);
                        // the single-stage kernel: operands by rank index, own = local
                        let v: Vec<f32> = (0..4).map(|q| if q == rank { parts[rank][i] } else { parts[q][i] }).collect();
                        let single = (v[0] + v[2]) + (v[1] + v[3]);
                        assert_eq!(rd.to_bits(), single.to_bits(), "perm {perm:?} rank {rank} elem {i}");
                        assert_eq!(single.to_bits(), want[i].to_bits());
                        assert_eq!(f16r(rd).to_bits(), f16r(single).to_bits(), "the final f16 rounding");
                        let other = (v[0] + v[1]) + (v[2] + v[3]);
                        if other.to_bits() != single.to_bits() { differing_ctl += 1; }
                    }
                }
            }
        }
        assert!(differing_ctl > 0, "negative control: the other grouping must differ somewhere or the data is too tame");
        assert!(single_tree_reference(&tp4f2_partials(8, 1)[..3]).is_err(), "world/partials mismatch must be refused");
        assert!(tp4f2_partials(n, 1).iter().flatten().all(|x| x.is_finite()));
        assert!(tp4f2_partials(n, 1).iter().flatten().any(|x| x.abs() > 0.0 && x.abs() < f32::MIN_POSITIVE), "denormals present");
    }

    /// Each arm alone at world 4, from an even and an odd start epoch (even = the R10-aligned case; odd
    /// is the un-aligned case the tree reference still describes): rank-identical, expected value, phase
    /// preserved, and the verified-op count equals the op count (a success signal, not absence of error).
    #[test]
    fn dec_arms_single_ops_w4() {
        for e0 in [0u64, 1, 2, 3] {
            for op in [Op::Mixer(256), Op::Moe(1024), Op::Vp(1), Op::Vp(5), Op::Vp(16), Op::Dh, Op::Prefill(3 * EPOCH_FLOATS + 28)] {
                let (e, v) = run_ops(4, e0, &[op], &real_exch).unwrap_or_else(|m| panic!("e0 {e0} {op:?}: {m}"));
                assert_eq!(v, 1, "{op:?}");
                assert!(e > e0, "{op:?} consumed no epochs");
                assert_eq!((e - e0) % 2, 0, "{op:?}: a logical reduce is a whole number of rounds");
            }
        }
    }

    /// World 2 through the same model (the exchange list at rounds == 1): every arm still exact.
    #[test]
    fn dec_arms_single_ops_w2() {
        for e0 in [0u64, 1, 2, 5] {
            for op in [Op::Mixer(256), Op::Moe(1024), Op::Vp(5), Op::Dh, Op::Prefill(EPOCH_FLOATS + 4)] {
                let (e, v) = run_ops(2, e0, &[op], &real_exch).unwrap_or_else(|m| panic!("e0 {e0} {op:?}: {m}"));
                assert_eq!(v, 1);
                let want = if matches!(op, Op::Prefill(_)) { epochs(EPOCH_FLOATS + 4).len() as u64 } else { 1 };
                assert_eq!(e - e0, want, "{op:?}: one epoch per logical reduce at world 2");
            }
        }
    }

    /// The decode spine's real interleaving: per layer a mixer reduce then an EP-MoE reduce (x40), then
    /// the draft passes' screens, the vp key merge, a prefill chunk reduce in the middle of it all — at
    /// world 2 and 4, from aligned epochs.
    #[test]
    fn dec_arms_layer_stack_interleaving() {
        for world in [2usize, 4] {
            let rounds = rounds_of(world).unwrap();
            let mut ops = Vec::new();
            for _ in 0..40 { ops.push(Op::Mixer(256)); ops.push(Op::Moe(512)); }
            ops.extend([Op::Dh, Op::Dh, Op::Vp(3), Op::Prefill(EPOCH_FLOATS + 12), Op::Mixer(64), Op::Dh, Op::Vp(16)]);
            for e0 in [0u64, 2 * rounds as u64] {
                let (e, v) = run_ops(world, e0, &ops, &real_exch).unwrap_or_else(|m| panic!("world {world} e0 {e0}: {m}"));
                assert_eq!(v, ops.len());
                let want = ops.iter().map(|op| match op {
                    Op::Prefill(n) => (rounds * epochs(*n).len()) as u64,
                    _ => rounds as u64,
                }).sum::<u64>();
                assert_eq!(e - e0, want, "world {world}: the stack consumed {} epochs, expected {want}", e - e0);
                assert_eq!(e % rounds as u64, e0 % rounds as u64);
            }
        }
    }

    /// Randomised interleavings (200 sequences, fixed LCG): any order of the five arms at world 4 (and 2)
    /// keeps rank identity, the expected values and the phase.
    #[test]
    fn dec_arms_random_interleavings() {
        let mut s = 0x1234_5678_9ABC_DEF1u64;
        let mut next = move |m: u64| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 33) % m };
        let mut total = 0usize;
        for seq in 0..200 {
            let world = if seq % 4 == 3 { 2 } else { 4 };
            let len = 1 + next(24) as usize;
            let ops: Vec<Op> = (0..len).map(|_| match next(10) {
                0..=2 => Op::Mixer(4 * (1 + next(200) as usize)),
                3..=4 => Op::Moe(4 * (1 + next(300) as usize)),
                5..=6 => Op::Vp(1 + next(16) as usize),
                7..=8 => Op::Dh,
                _ => Op::Prefill(1 + next(2 * EPOCH_FLOATS as u64) as usize),
            }).collect();
            let e0 = 2 * next(4);
            let (_, v) = run_ops(world, e0, &ops, &real_exch).unwrap_or_else(|m| panic!("seq {seq} world {world} e0 {e0} {ops:?}: {m}"));
            assert_eq!(v, ops.len());
            total += v;
        }
        assert!(total > 1000, "the randomised run verified only {total} ops");
    }

    /// Negative controls — the model must FAIL on the wrong schedules it exists to catch.
    ///  (a) the world-2 single-exchange fold applied at world 4: one epoch per logical reduce leaves each
    ///      rank with only its round partner's sum (ranks disagree) and flips the round phase;
    ///  (b) an arm that consumes one epoch too many (rounds + 1): the extra round double-counts and the
    ///      phase flips;
    ///  (c) the screen wired in place (out == local): the zero pad is polluted by the intermediate sums, so
    ///      the SECOND screen is wrong — the reason the multi-round screen keeps its intermediate in `out`.
    #[test]
    fn dec_arms_negative_controls() {
        let one = |_r: usize| dec_exchanges(1);
        let too_many = |r: usize| dec_exchanges(r + 1);
        for op in [Op::Mixer(256), Op::Moe(512), Op::Vp(5), Op::Dh] {
            let ea = run_ops(4, 0, &[op], &one).expect_err(&format!("(a) {op:?}: a one-epoch arm at world 4 must be caught"));
            assert!(ea.contains("tree reference") || ea.contains("global max") || ea.contains("exact union") || ea.contains("phase"), "(a) {op:?}: {ea}");
            let eb = run_ops(4, 0, &[op], &too_many).expect_err(&format!("(b) {op:?}: a three-epoch arm at world 4 must be caught"));
            assert!(eb.contains("tree reference") || eb.contains("global max") || eb.contains("exact union") || eb.contains("phase"), "(b) {op:?}: {eb}");
        }
        // the phase check alone (the cheap epoch-count assertion) catches both shapes even where the
        // values happen to survive (keys: max is idempotent, so a wrong round count can still merge)
        let e = run_ops(4, 0, &[Op::Vp(5), Op::Vp(5)], &one);
        assert!(e.is_err(), "two one-epoch key merges at world 4 must not pass");
        // (c) the in-place screen: drive model_dec_sum directly with out == local
        let world = 4usize;
        let grid = NBLK / world;
        let mut mem: Vec<Mem> = (0..world).map(|_| { let mut m = Mem::new(); m.insert(DHL, vec![0.0f32; NBLK]); m }).collect();
        let mut e = 0u64;
        let mut polluted = false;
        for call in 0..2u32 {
            let mut truth = vec![0.0f32; NBLK];
            for r in 0..world {
                let buf = mem[r].get_mut(&DHL).unwrap();
                for b in r * grid..(r + 1) * grid { buf[b] = (b as u32 + 1 + call * 1000) as f32; truth[b] = buf[b]; }
            }
            model_dec_sum(world, &mut e, &dec_exchanges(2), &mut mem, DHL, false, DHL, false);
            if call == 1 { polluted = (0..world).any(|r| bits(&mem[r][&DHL]) != bits(&truth)); }
        }
        assert!(polluted, "(c) the in-place screen must be caught by the second call");
    }

    /// The decode arms are the same K1 -> K2 -> K1 stream order as the serial schedule, so the recv-slot
    /// safety DAG of `serial_schedule_recv_slot_safety` applies unchanged: every exchange is exactly one
    /// K1 then one K2, consecutive exchanges of consecutive epochs (the plain K1 of round k is stream-
    /// ordered after the K2 of round k-1 — it has no PDL attribute).
    #[test]
    fn dec_exchange_is_one_k1_then_one_k2() {
        for rounds in [1usize, 2, 3] {
            let xs = dec_exchanges(rounds);
            let k1s = xs.iter().filter(|x| x.folded_k1).count() + xs.iter().filter(|x| !x.folded_k1).count();
            assert_eq!(k1s, rounds, "one K1 (folded or plain) per exchange");
            assert_eq!(xs.iter().filter(|x| x.folded_k1).count(), 1, "exactly one folded producer K1 per logical reduce");
            assert_eq!(xs.iter().filter(|x| x.writes_final).count(), 1);
        }
    }

    // =============================================================================================
    // TP-4E: the pipelined K1m/K2m PROGRAM (`build_program`) at every world, run through a CPU device model.
    //
    // What the model is: `world` ranks, each executing the SAME Step list in stream order; per (rank, rail) a
    // device epoch counter and, per round, an R-slot receive ring; the K1m/K2m kernels' gating, per-round
    // partner (rank ^ (1 << (e % rounds))), canonical add order, io bits (typed f16/fp32 reads and writes,
    // f16 wire vs fp32 wire), lag semantics and the proxy (an asynchronous per-QP FIFO that delivers a K1m's
    // payload into the partner's recv slot whenever the scheduler says so). A randomised scheduler (several
    // fairness modes, several seeds) interleaves ranks and deliveries; the model fails on: a recv slot
    // overwritten before its K2m consumed it (clobber), a wire format / length mismatch between sender and
    // receiver, an untyped memory access, a K2m for an epoch outside the reduce, a hook over rows that are
    // not final, and a deadlock (no rank can advance and nothing is in flight).
    //
    // What it does NOT prove: the kernels' real arithmetic and memory ordering (fences, tails, the
    // generation-tagged tail check), the RDMA proxy's real timing or QP-level ordering beyond per-QP FIFO,
    // the dual-stream event ordering, the fold kernels, or any throughput number. It proves the LAUNCH
    // PROGRAM and the per-round IO design are consistent, deadlock-free, clobber-free, rank-identical and
    // equal to the fixed reduction tree — the hardware gates then check the real kernels against the same tree.
    // =============================================================================================

    use half::f16 as H16;

    const SRC: u64 = 0x1000_0000;
    const OUT: u64 = 0x2000_0000;
    const MID: u64 = 0x3000_0000;
    const BUB: u64 = 0x4000_0000;
    const NAN32: u32 = 0xFFC0_0001;

    /// The four served reduce IOs plus two shape variants (name, io).
    fn sites() -> Vec<(&'static str, Io)> {
        vec![
            ("f32_inplace", Io::f32_inplace(SRC)),
            ("mixer_wire", Io { src: SRC, src_f16: true, out: SRC, out_f16: true, wire_f16: true, mid: MID }),
            ("mixer_nowire", Io { src: SRC, src_f16: true, out: SRC, out_f16: true, wire_f16: false, mid: MID }),
            ("moe_consume_src", Io { src: SRC, src_f16: false, out: OUT, out_f16: true, wire_f16: false, mid: SRC }),
            ("moe_separate_mid", Io { src: SRC, src_f16: false, out: OUT, out_f16: true, wire_f16: false, mid: MID }),
            ("f16src_f32out_wire", Io { src: SRC, src_f16: true, out: OUT, out_f16: false, wire_f16: true, mid: MID }),
        ]
    }

    // ---- the frozen pre-TP-4E launch loop (world 2 / rounds == 1), the reference for the refactor ----

    #[allow(clippy::too_many_arguments)]
    fn legacy_hook_rows_after_tick(ep: &[(usize, usize)], n: usize, nr: usize, g: usize, t: usize, row_len: usize,
                                   min_rows: usize, rows_done: usize) -> Option<usize> {
        if t < g { return None; }
        let fin = (nr * (t - g + 1)).min(ep.len());
        let elems = if fin == ep.len() { n } else { ep[fin].0 };
        let rows = elems / row_len;
        (rows > rows_done && (rows - rows_done >= min_rows || elems == n)).then_some(rows)
    }

    /// `reduce_pipe_hooked`'s tick loop as it was before TP-4E, transcribed into Steps.
    #[allow(clippy::too_many_arguments)]
    fn legacy_program(io: &Io, nrails: usize, g: usize, hook: Option<HookSpec>, fold: bool, ep: &[(usize, usize)],
                      n: usize) -> Vec<Step> {
        let nr = nrails.min(ep.len().max(1));
        let seqs: Vec<Vec<Ep>> = (0..nr).map(|r| {
            let mine: Vec<(usize, usize)> = ep.iter().enumerate().filter(|(i, _)| i % nr == r).map(|(_, &c)| c).collect();
            rail_seq(&mine, 1, g)
        }).collect();
        let (se, oe) = (if io.src_f16 { 2u64 } else { 4 }, if io.out_f16 { 2u64 } else { 4 });
        let span = |e: &Ep| -> (u64, u64, usize, i32) {
            match *e {
                Ep::Chunk(o, l) => (io.src + o as u64 * se, io.out + o as u64 * oe, l, io.bits()),
                Ep::Bubble => (BUB, BUB, 4, 0),
            }
        };
        let longest = seqs.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut rows_done = 0usize;
        let mut prog = Vec::new();
        for t in 0..longest + g {
            for (r, sq) in seqs.iter().enumerate() {
                if t >= g && t - g < sq.len() {
                    let j = t - g;
                    let (sp, op, len, bits) = span(&sq[j]);
                    let lag = (t.min(sq.len()) - 1 - j) as u32;
                    if fold {
                        let Ep::Chunk(o, l) = sq[j] else { panic!("bubble in a fold program") };
                        let rl = hook.unwrap().row_len;
                        prog.push(Step::Fold { rail: r, lag, r0: o / rl, r1: (o + l) / rl });
                    } else {
                        prog.push(Step::K2m { rail: r, out: op, local: sp, n: len as i32, lag, bits });
                    }
                }
                if t < sq.len() {
                    let (sp, _, len, bits) = span(&sq[t]);
                    let wb = if bits & 4 != 0 { 2 } else { 4 };
                    prog.push(Step::K1m { rail: r, src: sp, nbytes: (len * wb) as u32, bits: bits & 5 });
                }
            }
            if let (true, Some(h)) = (!fold, hook) {
                if let Some(rows) = legacy_hook_rows_after_tick(ep, n, nr, g, t, h.row_len, h.min_rows, rows_done) {
                    prog.push(Step::Hook { r0: rows_done, r1: rows, tail: false });
                    rows_done = rows;
                }
            }
        }
        if let (false, Some(h)) = (fold, hook) {
            let rows = n / h.row_len;
            if rows > rows_done { prog.push(Step::Hook { r0: rows_done, r1: rows, tail: true }); }
        }
        prog
    }

    /// World 2 (rounds == 1) is a no-op refactor: `build_program` emits EXACTLY the frozen pre-TP-4E launch
    /// list (every K1m/K2m argument, lag, hook range and tail hook, and the fold list) for every served IO,
    /// 1 and 2 rails, every lookahead, ragged and exact chunkings, with and without a hook, hook granularities
    /// from 1 row to "never until the end". Success signal: a nonzero count of compared programs that include
    /// non-tail hooks and fold steps.
    #[test]
    fn program_world2_equals_frozen_legacy() {
        let (mut compared, mut with_hooks, mut with_folds) = (0usize, 0usize, 0usize);
        for (name, io) in sites() {
            for nrails in 1..=2usize {
                for g in 1..=RING_SLOTS / 2 {
                    for &row_len in &[8usize, 16, 40] {
                        for &c in &[1usize, 2, 3, 7, 10, 13, 25] {
                            for &per in &[24usize, 64, 256] {
                                let n = c * row_len;
                                for hook in [None, Some(HookSpec { row_len, min_rows: 1 }), Some(HookSpec { row_len, min_rows: 3 }),
                                             Some(HookSpec { row_len, min_rows: 1000 })] {
                                    for fold in [false, true] {
                                        if fold && (hook.is_none() || per < row_len) { continue; }
                                        let ep = if fold { epochs_of(n, fold_per(per, row_len)) } else { epochs_of(n, per) };
                                        let spec = ProgSpec { io, rounds: 1, nrails, g, bub: BUB, hook, fold };
                                        let got = build_program(&spec, &ep, n).unwrap();
                                        let want = legacy_program(&io, nrails, g, hook, fold, &ep, n);
                                        assert_eq!(got, want, "{name} nrails {nrails} g {g} row_len {row_len} c {c} per {per} hook {hook:?} fold {fold}");
                                        assert!(!got.is_empty());
                                        compared += 1;
                                        with_hooks += got.iter().any(|s| matches!(s, Step::Hook { tail: false, .. })) as usize;
                                        with_folds += got.iter().any(|s| matches!(s, Step::Fold { .. })) as usize;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(compared > 5000 && with_hooks > 500 && with_folds > 500, "compared {compared} hooks {with_hooks} folds {with_folds}");
    }

    /// The exchange arguments per round: exchange 0 reads the source (f16 when the source is) and is the only
    /// exchange that may ship f16; later exchanges read the fp32 stage and ship fp32; the last writes `out`
    /// (with the f16 rounding bit), the others write the stage (fp32, no rounding). rounds == 1 is the
    /// pre-TP-4E launch (`Io::bits`). Also the looked-up lookahead clamp.
    #[test]
    fn xchg_shapes_per_round() {
        for (name, io) in sites() {
            let (se, oe) = (if io.src_f16 { 2u64 } else { 4 }, if io.out_f16 { 2u64 } else { 4 });
            let a = io.xchg(0, 1, 8, 0);
            assert_eq!((a.local, a.out, a.k1_bits, a.k2_bits, a.wire_f16), (io.src + 8 * se, io.out + 8 * oe, io.bits() & 5, io.bits(), io.wire16()), "{name} rounds 1");
            for rounds in [2usize, 3] {
                let stage = 0x7000_0000u64;
                for k in 0..rounds {
                    let a = io.xchg(stage, rounds, 8, k);
                    let (first, last) = (k == 0, k == rounds - 1);
                    assert_eq!(a.local, if first { io.src + 8 * se } else { stage + 32 }, "{name} r{rounds} k{k} local");
                    assert_eq!(a.out, if last { io.out + 8 * oe } else { stage + 32 }, "{name} r{rounds} k{k} out");
                    assert_eq!(a.k1_bits & 1 != 0, first && io.src_f16, "{name} r{rounds} k{k}: only exchange 0 reads f16");
                    assert_eq!(a.k1_bits & 4 != 0, first && io.wire16(), "{name} r{rounds} k{k}: only exchange 0 ships f16");
                    assert_eq!(a.wire_f16, a.k1_bits & 4 != 0);
                    assert_eq!(a.k2_bits & 2 != 0, last && io.out_f16, "{name} r{rounds} k{k}: the one f16 rounding is the last write");
                    assert_eq!(a.k2_bits & 5, a.k1_bits, "{name} r{rounds} k{k}: K2m reads the peer's wire as K1m wrote it");
                }
            }
        }
        for (req, rounds, want) in [(1usize, 1usize, 1usize), (2, 1, 2), (4, 1, 4), (9, 1, 4), (1, 2, 2), (3, 2, 2), (4, 2, 4), (8, 2, 4),
                                    (4, 4, 4), (2, 4, 4)] {
            assert_eq!(effective_lookahead(req, rounds).unwrap(), want, "lookahead {req} rounds {rounds}");
        }
        assert!(effective_lookahead(4, 5).is_err() && effective_lookahead(4, 8).is_err(), "rounds > R/2 cannot be pipelined");
        // W4P-1: rounds must divide the ring (epoch x and x + R share a slot): world 8 (rounds 3) and rounds 5, 6, 7 are refused
        // for every requested lookahead, and the message names the ring; rounds 0 is an error, not a division by zero
        for rounds in [0usize, 3, 5, 6, 7] {
            for req in [1usize, 2, 3, 4, 8] {
                assert!(effective_lookahead(req, rounds).is_err(), "rounds {rounds} lookahead {req} must be refused");
            }
        }
        let why = format!("{:#}", effective_lookahead(4, 3).unwrap_err());
        assert!(why.contains("does not divide") && why.contains("recv ring"), "{why}");
    }

    /// rounds > 1 needs somewhere to keep the running sum: the plain fp32 in-place IO stages in place; any f16
    /// or split IO needs a 16 B-aligned fp32 `mid` that does not alias src/out (except an fp32 src consumed in
    /// place or an fp32 out used as the stage); rounds == 1 needs nothing (the stage is unused).
    #[test]
    fn io_rounds_guard() {
        let n = 1024usize;
        let ok = Io::f32_inplace(0x1000);
        assert_eq!(stage_for(&ok, n).unwrap(), 0x1000, "fp32 in place stages in place");
        let with_mid = Io { mid: 0x9000_0000, ..ok };
        assert_eq!(stage_for(&with_mid, n).unwrap(), 0x1000, "an in-place fp32 IO ignores mid");
        let mixer = Io { src: 0x10_0000, src_f16: true, out: 0x10_0000, out_f16: true, wire_f16: true, mid: 0 };
        let moe = Io { src: 0x30_0000, src_f16: false, out: 0x40_0000, out_f16: true, wire_f16: false, mid: 0 };
        let split = Io { src: 0x50_0000, src_f16: false, out: 0x60_0000, out_f16: false, wire_f16: false, mid: 0 };
        for (name, io) in [("mixer", mixer), ("moe", moe), ("split f32", split)] {
            assert!(stage_for(&io, n).is_err(), "{name}: no staging buffer must be refused");
            assert!(stage_for(&Io { mid: 0x7000_0008, ..io }, n).is_err(), "{name}: a misaligned mid must be refused");
            assert_eq!(stage_for(&Io { mid: 0x7000_0000, ..io }, n).unwrap(), 0x7000_0000, "{name}: a disjoint mid is the stage");
            assert!(stage_for(&Io { mid: io.out + 16, ..io }, n).is_err(), "{name}: a mid inside out must be refused");
        }
        // aliasing: an fp32 src may be the stage (consumed), an f16 src never; an fp32 out may be the stage
        assert_eq!(stage_for(&Io { mid: moe.src, ..moe }, n).unwrap(), moe.src);
        assert!(stage_for(&Io { mid: mixer.src, ..mixer }, n).is_err(), "an f16 src cannot be the fp32 stage");
        assert!(stage_for(&Io { mid: 0x10_0000 + 512, ..mixer }, n).is_err(), "a mid inside the f16 src must be refused");
        assert_eq!(stage_for(&Io { mid: split.out, ..split }, n).unwrap(), split.out, "an fp32 out may be the stage");
        assert!(stage_for(&Io { mid: moe.out, ..moe }, n).is_err(), "an f16 out cannot be the fp32 stage");
        // W4P-2: src and out must be the identical span (same base AND width) or disjoint; K2m reads src[i] and writes out[i] in one
        // thread, so a partial/width-mismatched overlap lets another thread overwrite an element before it is read
        let stage = 0x7000_0000u64;
        let (base, f16_bytes) = (0x10_0000u64, 2 * n as u64);
        for (name, src_f16, out_f16, out) in [
            ("f16 src, fp32 out, same base", true, false, base),
            ("f16 src, fp32 out at src+512", true, false, base + 512),
            ("fp32 src, f16 out, same base", false, true, base),
            ("fp32 src, f16 out at src+512", false, true, base + 512),
            ("fp32 src, fp32 out at src+64", false, false, base + 64),
            ("f16 src, f16 out at src+2", true, true, base + 2),
        ] {
            let io = Io { src: base, src_f16, out, out_f16, wire_f16: false, mid: stage };
            let why = format!("{:#}", stage_for(&io, n).expect_err(name));
            assert!(why.contains("overlap") && why.contains("identical span"), "{name}: {why}");
        }
        // controls: the identical f16 span, a disjoint pair and an exactly-adjacent pair are still accepted
        assert_eq!(stage_for(&Io { mid: stage, ..mixer }, n).unwrap(), stage, "identical f16 in-place span");
        let adjacent = Io { src: base, src_f16: true, out: base + f16_bytes, out_f16: false, wire_f16: false, mid: stage };
        assert_eq!(stage_for(&adjacent, n).unwrap(), stage, "adjacent (touching) src/out spans are disjoint");
        let after = Io { src: base + 4 * n as u64, src_f16: false, out: base, out_f16: true, wire_f16: false, mid: stage };
        assert_eq!(stage_for(&after, n).unwrap(), stage, "an f16 out below an fp32 src is disjoint");
        // the guard applies at rounds > 1 only: world 2 (rounds 1) never reaches it
        let overlapped = Io { src: base, src_f16: true, out: base, out_f16: false, wire_f16: false, mid: stage };
        let ep1 = epochs_of(64, 32);
        let spec1 = |rounds| ProgSpec { io: overlapped, rounds, nrails: 1, g: 2, bub: BUB, hook: None, fold: false };
        assert!(build_program(&spec1(1), &ep1, 64).is_ok(), "rounds 1 does not stage, so the overlap check is not reached");
        assert!(build_program(&spec1(2), &ep1, 64).is_err(), "rounds 2 refuses the overlapping src/out");
        // the program builder applies the guard at rounds > 1 only
        let ep = epochs_of(64, 32);
        for io in [mixer, moe, split] {
            let spec = |rounds| ProgSpec { io, rounds, nrails: 1, g: 2, bub: BUB, hook: None, fold: false };
            assert!(build_program(&spec(1), &ep, 64).is_ok(), "rounds 1 needs no stage");
            assert!(build_program(&spec(2), &ep, 64).is_err(), "rounds 2 without a stage must be refused");
        }
        assert!(build_program(&ProgSpec { io: ok, rounds: 2, nrails: 2, g: 2, bub: BUB, hook: None, fold: false }, &ep, 64).is_ok());
        let fold = ProgSpec { io: ok, rounds: 2, nrails: 1, g: 2, bub: BUB, hook: Some(HookSpec { row_len: 8, min_rows: 1 }), fold: true };
        assert!(build_program(&fold, &ep, 64).is_err(), "fold stays world 2 only");
    }

    // ---- the CPU device model ----

    #[derive(Clone)]
    enum Buf { F32(Vec<f32>), F16(Vec<u16>) }
    #[derive(Clone)]
    struct Region { base: u64, buf: Buf }
    #[derive(Clone, PartialEq, Debug)]
    enum Wire { F32(Vec<f32>), F16(Vec<u16>) }
    struct Slot { epoch: u64, consumed: u64, wire: Wire }
    struct Pend { from: usize, to: usize, rail: usize, epoch: u64, wire: Wire }
    struct MRank {
        regions: Vec<Region>,
        ctr: Vec<u64>,
        base: Vec<u64>,
        recv: Vec<Vec<Vec<Option<Slot>>>>,
        pc: usize,
        hooks: Vec<(usize, usize, bool)>,
    }
    struct Expect { out: u64, f16: bool, bits: Vec<u32>, row_len: usize }

    fn region_at(regions: &[Region], addr: u64, bytes: u64) -> Result<usize, String> {
        regions.iter().position(|r| {
            let len = match &r.buf { Buf::F32(v) => v.len() as u64 * 4, Buf::F16(v) => v.len() as u64 * 2 };
            addr >= r.base && addr + bytes <= r.base + len
        }).ok_or_else(|| format!("access [{addr:#x}, +{bytes}) is outside every region"))
    }
    fn rd_raw(regions: &[Region], addr: u64, n: usize, f16: bool) -> Result<Buf, String> {
        let es = if f16 { 2usize } else { 4 };
        let i = region_at(regions, addr, (n * es) as u64)?;
        let r = &regions[i];
        let off = (addr - r.base) as usize;
        if off % es != 0 { return Err(format!("misaligned access at {addr:#x}")); }
        let at = off / es;
        match (&r.buf, f16) {
            (Buf::F16(v), true) => Ok(Buf::F16(v[at..at + n].to_vec())),
            (Buf::F32(v), false) => Ok(Buf::F32(v[at..at + n].to_vec())),
            _ => Err(format!("typed access mismatch at {addr:#x}: the region is {} but the launch treats it as {}",
                             if matches!(r.buf, Buf::F16(_)) { "f16" } else { "fp32" }, if f16 { "f16" } else { "fp32" })),
        }
    }
    fn rd_f32(regions: &[Region], addr: u64, n: usize, f16: bool) -> Result<Vec<f32>, String> {
        Ok(match rd_raw(regions, addr, n, f16)? {
            Buf::F16(v) => v.iter().map(|&b| H16::from_bits(b).to_f32()).collect(),
            Buf::F32(v) => v,
        })
    }
    fn rd_bits(regions: &[Region], addr: u64, n: usize, f16: bool) -> Result<Vec<u32>, String> {
        Ok(match rd_raw(regions, addr, n, f16)? {
            Buf::F16(v) => v.iter().map(|&b| b as u32).collect(),
            Buf::F32(v) => v.iter().map(|x| x.to_bits()).collect(),
        })
    }
    fn wr(regions: &mut [Region], addr: u64, vals: &[f32], f16: bool) -> Result<(), String> {
        let es = if f16 { 2usize } else { 4 };
        let i = region_at(regions, addr, (vals.len() * es) as u64)?;
        let r = &mut regions[i];
        let off = (addr - r.base) as usize;
        if off % es != 0 { return Err(format!("misaligned write at {addr:#x}")); }
        let at = off / es;
        match (&mut r.buf, f16) {
            (Buf::F16(v), true) => { for (k, &x) in vals.iter().enumerate() { v[at + k] = H16::from_f32(x).to_bits(); } Ok(()) }
            (Buf::F32(v), false) => { v[at..at + vals.len()].copy_from_slice(vals); Ok(()) }
            _ => Err(format!("typed write mismatch at {addr:#x}")),
        }
    }

    struct Model<'a> { world: usize, rounds: usize, progs: Vec<&'a [Step]>, ranks: Vec<MRank>, pend: Vec<Pend>, expect: Option<&'a Expect> }

    impl<'a> Model<'a> {
        fn retired(&self, r: usize, rail: usize, e: u64) -> bool {
            // K1m(e) waits for the QP of partner(e - R) to have retired e - R (per-QP FIFO: nothing to that
            // partner at or below e - R is still in flight)
            if e <= RING_SLOTS as u64 { return true; }
            let tgt = e - RING_SLOTS as u64;
            let to = r ^ (1usize << (tgt % self.rounds as u64) as usize);
            !self.pend.iter().any(|p| p.from == r && p.rail == rail && p.to == to && p.epoch <= tgt)
        }
        fn can_run(&self, r: usize) -> bool {
            let rk = &self.ranks[r];
            match self.progs[r][rk.pc] {
                Step::K1m { rail, .. } => self.retired(r, rail, rk.ctr[rail] + 1),
                Step::K2m { rail, lag, .. } | Step::K2c { rail, lag, .. } => match rk.ctr[rail].checked_sub(lag as u64) {
                    None => true,                  // exec reports it
                    Some(e) => matches!(&rk.recv[rail][(e % self.rounds as u64) as usize][(e % RING_SLOTS as u64) as usize],
                                        Some(s) if s.epoch == e),
                },
                Step::Fold { .. } | Step::Hook { .. } => true,
            }
        }
        fn exec(&mut self, r: usize) -> Result<(), String> {
            let st = self.progs[r][self.ranks[r].pc];
            let (world, rounds) = (self.world, self.rounds as u64);
            match st {
                Step::K1m { rail, src, nbytes, bits } => {
                    let rk = &mut self.ranks[r];
                    let e = rk.ctr[rail] + 1;
                    let wire = if bits & 4 != 0 {
                        if bits & 1 == 0 { return Err("K1m f16 wire without an f16 source".into()); }
                        if nbytes % 2 != 0 { return Err("f16 wire bytes".into()); }
                        let Buf::F16(v) = rd_raw(&rk.regions, src, nbytes as usize / 2, true)? else { unreachable!() };
                        Wire::F16(v)
                    } else {
                        if nbytes % 4 != 0 { return Err("fp32 wire bytes".into()); }
                        Wire::F32(rd_f32(&rk.regions, src, nbytes as usize / 4, bits & 1 != 0)?)
                    };
                    if nbytes % 16 != 0 { return Err(format!("K1m nbytes {nbytes} is not a 16 B multiple")); }
                    rk.ctr[rail] = e;
                    let to = r ^ (1usize << (e % rounds) as usize);
                    if to >= world { return Err("partner outside the world".into()); }
                    self.pend.push(Pend { from: r, to, rail, epoch: e, wire });
                }
                Step::K2m { rail, out, local, n, lag, bits } => {
                    let rk = &mut self.ranks[r];
                    let e = rk.ctr[rail].checked_sub(lag as u64).filter(|&e| e > rk.base[rail])
                        .ok_or_else(|| format!("K2m lag {lag} reaches before the reduce (ctr {}, base {})", rk.ctr[rail], rk.base[rail]))?;
                    let round = (e % rounds) as usize;
                    let n = n as usize;
                    let (peer, slot_epoch) = {
                        let s = rk.recv[rail][round][(e % RING_SLOTS as u64) as usize].as_ref().ok_or("K2m on an empty slot")?;
                        (s.wire.clone(), s.epoch)
                    };
                    if slot_epoch != e { return Err(format!("K2m epoch {e} found epoch {slot_epoch} in its slot")); }
                    let peer_f32: Vec<f32> = match (&peer, bits & 4 != 0) {
                        (Wire::F16(v), true) => v.iter().map(|&b| H16::from_bits(b).to_f32()).collect(),
                        (Wire::F32(v), false) => v.clone(),
                        _ => return Err(format!("wire format mismatch at epoch {e}: sender {} receiver expects {}",
                                                if matches!(peer, Wire::F16(_)) { "f16" } else { "fp32" },
                                                if bits & 4 != 0 { "f16" } else { "fp32" })),
                    };
                    if peer_f32.len() != n { return Err(format!("wire length mismatch at epoch {e}: {} vs n {n}", peer_f32.len())); }
                    let loc = rd_f32(&rk.regions, local, n, bits & 1 != 0)?;
                    let lower = r & (1usize << round) == 0;
                    let sum: Vec<f32> = (0..n).map(|i| if lower { loc[i] + peer_f32[i] } else { peer_f32[i] + loc[i] }).collect();
                    wr(&mut rk.regions, out, &sum, bits & 2 != 0)?;
                    rk.recv[rail][round][(e % RING_SLOTS as u64) as usize].as_mut().unwrap().consumed = e;
                }
                Step::K2c { rail, out, bytes, lag } => {
                    let rk = &mut self.ranks[r];
                    let e = rk.ctr[rail].checked_sub(lag as u64).filter(|&e| e > rk.base[rail])
                        .ok_or_else(|| format!("K2c lag {lag} reaches before the exchange (ctr {}, base {})", rk.ctr[rail], rk.base[rail]))?;
                    let round = (e % rounds) as usize;
                    let (peer, slot_epoch) = {
                        let s = rk.recv[rail][round][(e % RING_SLOTS as u64) as usize].as_ref().ok_or("K2c on an empty slot")?;
                        (s.wire.clone(), s.epoch)
                    };
                    if slot_epoch != e { return Err(format!("K2c epoch {e} found epoch {slot_epoch} in its slot")); }
                    // the world > 2 receiver validates the slot tail at ITS expected length: bytes must equal the sender's
                    let Wire::F32(v) = &peer else { return Err("the model's K2c lands fp32-typed bytes only".into()) };
                    if v.len() * 4 != bytes as usize {
                        return Err(format!("K2c length mismatch at epoch {e}: sender {} B vs expected {bytes} B (tail timeout on hardware)", v.len() * 4));
                    }
                    wr(&mut rk.regions, out, v, false)?;
                    rk.recv[rail][round][(e % RING_SLOTS as u64) as usize].as_mut().unwrap().consumed = e;
                }
                Step::Fold { .. } => return Err("the model does not run fold programs".into()),
                Step::Hook { r0, r1, tail } => {
                    let rk = &mut self.ranks[r];
                    if let Some(x) = self.expect {
                        let es = if x.f16 { 2u64 } else { 4 };
                        let got = rd_bits(&rk.regions, x.out + (r0 * x.row_len) as u64 * es, (r1 - r0) * x.row_len, x.f16)?;
                        if got != x.bits[r0 * x.row_len..r1 * x.row_len] {
                            return Err(format!("rank {r}: hook over rows [{r0}, {r1}) read rows that are not final"));
                        }
                    }
                    rk.hooks.push((r0, r1, tail));
                }
            }
            self.ranks[r].pc += 1;
            Ok(())
        }
        fn deliver(&mut self, i: usize) -> Result<(), String> {
            let p = self.pend.remove(i);
            let round = (p.epoch % self.rounds as u64) as usize;
            let slot = &mut self.ranks[p.to].recv[p.rail][round][(p.epoch % RING_SLOTS as u64) as usize];
            if let Some(old) = slot {
                if old.consumed < old.epoch {
                    return Err(format!("CLOBBER: epoch {} from rank {} overwrote rank {}'s recv slot (rail {}, round {round}) still holding \
                                        unconsumed epoch {}", p.epoch, p.from, p.to, p.rail, old.epoch));
                }
            }
            *slot = Some(Slot { epoch: p.epoch, consumed: 0, wire: p.wire });
            Ok(())
        }
    }

    /// Run `prog` on `world` ranks. `mode`: 0 random, 1 slow proxy (deliveries only when no rank can step),
    /// 2 fast proxy (deliveries first), 3 one starved rank (`seed % world` steps only when nobody else can).
    #[allow(clippy::too_many_arguments)]
    fn simulate(world: usize, prog: &[Step], regs: Vec<Vec<Region>>, e0: &[Vec<u64>], nrails: usize, seed: u64, mode: usize,
                expect: Option<&Expect>) -> Result<Vec<MRank>, String> {
        simulate_ranks(world, &vec![prog; world], regs, e0, nrails, seed, mode, expect)
    }

    /// `simulate` with a PROGRAM PER RANK (the TP-4S reduce-scatter / all-gather: every rank's pointers differ).
    #[allow(clippy::too_many_arguments)]
    fn simulate_ranks(world: usize, progs: &[&[Step]], regs: Vec<Vec<Region>>, e0: &[Vec<u64>], nrails: usize, seed: u64, mode: usize,
                      expect: Option<&Expect>) -> Result<Vec<MRank>, String> {
        let rounds = rounds_of(world).unwrap();
        let ranks = (0..world).map(|r| MRank {
            regions: regs[r].clone(),
            ctr: e0[r].clone(),
            base: e0[r].clone(),
            recv: (0..nrails).map(|_| (0..rounds).map(|_| (0..RING_SLOTS).map(|_| None).collect()).collect()).collect(),
            pc: 0,
            hooks: Vec::new(),
        }).collect();
        let mut m = Model { world, rounds, progs: progs.to_vec(), ranks, pend: Vec::new(), expect };
        let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
        let starved = (seed % world as u64) as usize;
        loop {
            let runnable: Vec<usize> = (0..world).filter(|&r| m.ranks[r].pc < progs[r].len() && m.can_run(r)).collect();
            // per-QP FIFO: a delivery is ready when no older pending entry shares its (from, rail, to)
            let ready: Vec<usize> = (0..m.pend.len()).filter(|&i| {
                let p = &m.pend[i];
                !m.pend[..i].iter().any(|q| q.from == p.from && q.rail == p.rail && q.to == p.to)
            }).collect();
            if runnable.is_empty() && ready.is_empty() {
                if m.ranks.iter().enumerate().all(|(r, k)| k.pc == progs[r].len()) && m.pend.is_empty() { break; }
                return Err(format!("DEADLOCK: pcs {:?} of {:?}, {} deliveries pending (none ready)",
                                   m.ranks.iter().map(|k| k.pc).collect::<Vec<_>>(), progs.iter().map(|p| p.len()).collect::<Vec<_>>(), m.pend.len()));
            }
            let pick_step = |rn: u64, cands: &[usize]| cands[(rn % cands.len() as u64) as usize];
            let rn = next();
            let do_step = match mode {
                1 => !runnable.is_empty(),
                2 => ready.is_empty(),
                _ => !runnable.is_empty() && (ready.is_empty() || rn & 1 == 0),
            };
            if do_step {
                let cands: Vec<usize> = if mode == 3 && runnable.iter().any(|&r| r != starved) {
                    runnable.iter().copied().filter(|&r| r != starved).collect()
                } else { runnable.clone() };
                m.exec(pick_step(rn >> 1, &cands))?;
            } else {
                m.deliver(ready[((rn >> 1) % ready.len() as u64) as usize])?;
            }
        }
        Ok(m.ranks)
    }

    /// Partials with mixed magnitudes and exact cancellation across ranks (so different associations, and an
    /// f16 running sum, differ in bits). `exact16`: every value is f16-exact (the mixer's y_raw); values are
    /// bounded so a world-wide sum stays inside f16's range.
    fn model_parts(world: usize, n: usize, exact16: bool) -> Vec<Vec<f32>> {
        let amp = 50000.0f32 / world as f32;
        (0..world).map(|r| (0..n).map(|i| {
            let hsh = |salt: u64| -> f32 {
                let h = (i as u64 + 1).wrapping_mul(2654435761).wrapping_add(salt.wrapping_mul(40503) + 12345).wrapping_mul(0x9E37_79B9) >> 7;
                ((h % 2001) as f32 - 1000.0) / 1000.0
            };
            let big = hsh(1000 + (r / 4) as u64) * amp;
            let small = hsh(r as u64 + 7) * 0.013 + hsh(r as u64 + 99) * 0.000173;
            let v = match i % 3 {
                0 => match r % 4 { 0 => big, 1 => -big, _ => small },
                1 => hsh(r as u64 + 31) * 3.1,
                _ => if r % 2 == 0 { hsh(r as u64 + 5) * amp * 0.5 } else { small * 7.0 },
            };
            if exact16 { H16::from_f32(v).to_f32() } else { v }
        }).collect()).collect()
    }

    fn regions_for(io: &Io, n: usize, src_vals: &[f32]) -> Vec<Region> {
        let mut v = vec![Region { base: io.src, buf: if io.src_f16 { Buf::F16(src_vals.iter().map(|&x| H16::from_f32(x).to_bits()).collect()) }
                                                     else { Buf::F32(src_vals.to_vec()) } }];
        let marker = |f16_: bool| if f16_ { Buf::F16(vec![0xFFFF; n]) } else { Buf::F32(vec![f32::from_bits(NAN32); n]) };
        if io.out != io.src { v.push(Region { base: io.out, buf: marker(io.out_f16) }); }
        if io.mid != 0 && io.mid != io.src && io.mid != io.out { v.push(Region { base: io.mid, buf: marker(false) }); }
        v.push(Region { base: BUB, buf: Buf::F32(vec![0.0; 4]) });
        v
    }

    /// Expected output bits (as stored): per chunk the fixed tree over the widened partials at the chunk's RAIL
    /// phase, then the single f16 rounding when `out_f16`.
    fn expected_bits(world: usize, ep: &[(usize, usize)], nr: usize, e0_rail: &[u64], parts: &[Vec<f32>], out_f16: bool) -> Vec<u32> {
        let n = parts[0].len();
        let mut out = vec![0u32; n];
        for (i, &(o, l)) in ep.iter().enumerate() {
            let sl: Vec<Vec<f32>> = parts.iter().map(|p| p[o..o + l].to_vec()).collect();
            let t = tree_reference(world, e0_rail[i % nr], &sl);
            for (k, &v) in t.iter().enumerate() { out[o + k] = if out_f16 { H16::from_f32(v).to_bits() as u32 } else { v.to_bits() }; }
        }
        out
    }

    fn out_of(rk: &MRank, io: &Io, n: usize) -> Vec<u32> { rd_bits(&rk.regions, io.out, n, io.out_f16).unwrap() }

    /// THE matrix: world 2 and 4 (world 8 must be refused by the real clamp, W4P-1); every IO; 1 and 2 rails; lookahead 1..4 (clamped by the real rule); ragged
    /// chunkings down to one chunk; an even and a large start epoch (slot index and gate phase not at zero);
    /// with and without the rows-landed hook (granularity 1 and 4 rows); a fair scheduler, slow-proxy,
    /// fast-proxy and starved-rank schedules over several seeds. Every run must finish (no deadlock), never
    /// overwrite an unconsumed recv slot, and leave EVERY rank with out == the fixed reduction tree (per-chunk
    /// rail phase) bit for bit, the hooks over final rows only, and each rail's counter advanced by exactly its
    /// program length (a multiple of rounds: the phase cannot drift).
    #[test]
    fn pipelined_program_model_matches_the_tree_on_every_rank() {
        let (mut runs, mut hooked_runs, mut staged_runs, mut wire_f16_runs) = (0usize, 0usize, 0usize, 0usize);
        for &world in &[2usize, 4, 8] {
            let rounds = rounds_of(world).unwrap();
            if world == 8 {
                // W4P-1: world 8 (rounds 3) cannot be pipelined on the 8-slot ring: the real clamp refuses it, so no program exists to model
                assert!((1..=4).all(|l| effective_lookahead(l, rounds).is_err()), "world 8 must be refused for every lookahead");
                continue;
            }
            let mut gs: Vec<usize> = (1..=4).map(|l| effective_lookahead(l, rounds).unwrap()).collect();
            gs.dedup();
            for (name, io) in sites() {
                for &nrails in &[1usize, 2] {
                    for &g in &gs {
                        for &(per, row_len, c) in &[(64usize, 16usize, 3usize), (64, 24, 9), (96, 40, 17), (64, 16, 40)] {
                            let n = c * row_len;
                            let ep = epochs_of(n, per);
                            let nr = nrails.min(ep.len());
                            let parts = model_parts(world, n, io.src_f16);
                            for hook in [None, Some(HookSpec { row_len, min_rows: 1 }), Some(HookSpec { row_len, min_rows: 4 })] {
                                let spec = ProgSpec { io, rounds, nrails, g, bub: BUB, hook, fold: false };
                                let prog = build_program(&spec, &ep, n).unwrap();
                                for &e0v in &[0u64, 5 * rounds as u64] {
                                    let e0: Vec<Vec<u64>> = vec![vec![e0v; nrails]; world];
                                    let want = expected_bits(world, &ep, nr, &vec![e0v; nrails], &parts, io.out_f16);
                                    let exp = Expect { out: io.out, f16: io.out_f16, bits: want.clone(), row_len };
                                    for (mode, seed) in [(0usize, 1u64), (0, 2), (1, 3), (2, 4), (3, 5), (3, 6)] {
                                        let regs: Vec<Vec<Region>> = (0..world).map(|r| regions_for(&io, n, &parts[r])).collect();
                                        let ranks = simulate(world, &prog, regs, &e0, nrails, seed, mode, hook.map(|_| &exp))
                                            .unwrap_or_else(|e| panic!("{name} world {world} rails {nrails} g {g} n {n} per {per} hook {hook:?} e0 {e0v} \
                                                                        mode {mode} seed {seed}: {e}"));
                                        for (r, rk) in ranks.iter().enumerate() {
                                            assert_eq!(out_of(rk, &io, n), want, "{name} world {world} rails {nrails} g {g} n {n} per {per} hook {hook:?} \
                                                                                   e0 {e0v} mode {mode} seed {seed}: rank {r} != the tree");
                                            for rail in 0..nr {
                                                let len: usize = prog.iter().filter(|s| matches!(s, Step::K1m { rail: q, .. } if *q == rail)).count();
                                                assert_eq!(rk.ctr[rail], e0v + len as u64, "rail {rail} counter");
                                                assert_eq!(len % rounds, 0, "rail {rail}: program length must keep the round phase");
                                            }
                                            if let Some(h) = hook {
                                                let rows = n / h.row_len;
                                                let mut at = 0usize;
                                                for &(a, b, tail) in &rk.hooks {
                                                    assert_eq!(a, at, "hook ranges must tile the rows in order");
                                                    assert!(b > a && (tail || b - a >= h.min_rows || b == rows));
                                                    at = b;
                                                }
                                                assert_eq!(at, rows, "hooks must cover every row");
                                                assert!(rk.hooks.iter().filter(|x| x.2).count() <= 1 && rk.hooks.last().map_or(true, |x| x.2 || x.1 == rows));
                                            }
                                        }
                                        runs += 1;
                                        hooked_runs += hook.is_some() as usize;
                                        staged_runs += (rounds > 1 && io.mid != 0) as usize;
                                        wire_f16_runs += (io.wire16() && n % 8 == 0) as usize;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // world 8 is no longer modelled (W4P-1: the real clamp refuses it), so the floors are the world 2 + 4 counts
        // (10368 runs / 6912 hooked / 2880 staged / 3456 f16-wire after the change; the old staged floor 3000 included world 8)
        assert!(runs > 10_000 && hooked_runs > 5_000 && staged_runs > 2_500 && wire_f16_runs > 2_000,
                "runs {runs} hooked {hooked_runs} staged {staged_runs} f16-wire {wire_f16_runs}");
    }

    /// The model is not vacuous: (1) the data tells the two world-4 associations apart; (2) an IO variant that
    /// rounds the running sum to f16 between rounds (what an in-place f16 buffer would do) completes, is
    /// rank-identical, and is NOT the tree — the reason the staging buffer is fp32; (3) the real staged program
    /// on the same data IS the tree on every rank.
    #[test]
    fn f16_intermediate_is_output_changing_and_the_data_can_tell() {
        let world = 4usize;
        let (n, per) = (640usize, 64usize);
        let parts = model_parts(world, n, true);
        let a: Vec<f32> = (0..n).map(|i| (parts[0][i] + parts[1][i]) + (parts[2][i] + parts[3][i])).collect();
        let b: Vec<f32> = (0..n).map(|i| (parts[0][i] + parts[2][i]) + (parts[1][i] + parts[3][i])).collect();
        assert_ne!(bits(&a), bits(&b), "the test data cannot tell the two associations apart");
        let ep = epochs_of(n, per);
        let io16 = Io { src: SRC, src_f16: true, out: SRC, out_f16: true, wire_f16: true, mid: MID };
        let want = expected_bits(world, &ep, 2, &[0, 0], &parts, true);
        // the legal in-place fp32 program, rewritten as an all-f16 program over an f16 buffer
        let legal = build_program(&ProgSpec { io: Io::f32_inplace(SRC), rounds: 2, nrails: 2, g: 2, bub: BUB, hook: None, fold: false }, &ep, n).unwrap();
        let half = |a: u64| if a >= SRC && a < OUT { SRC + (a - SRC) / 2 } else { a };
        let bad: Vec<Step> = legal.iter().map(|s| match *s {
            Step::K1m { rail, src, nbytes, bits: _ } if src != BUB => Step::K1m { rail, src: half(src), nbytes: nbytes / 2, bits: 5 },
            Step::K2m { rail, out, local, n, lag, bits: _ } if out != BUB => Step::K2m { rail, out: half(out), local: half(local), n, lag, bits: 7 },
            other => other,
        }).collect();
        let regs: Vec<Vec<Region>> = (0..world).map(|r| regions_for(&io16, n, &parts[r])).collect();
        let e0 = vec![vec![0u64; 2]; world];
        let ranks = simulate(world, &bad, regs, &e0, 2, 9, 0, None).unwrap();
        let outs: Vec<Vec<u32>> = ranks.iter().map(|rk| out_of(rk, &io16, n)).collect();
        for r in 1..world { assert_eq!(outs[r], outs[0], "the f16-intermediate variant is still deterministic across ranks"); }
        let differing = outs[0].iter().zip(&want).filter(|(x, y)| x != y).count();
        assert!(differing > n / 20, "an f16 running sum must change the output (differs in {differing} of {n} elements)");
        // the real staged program, same data, IS the tree on every rank
        let spec = ProgSpec { io: io16, rounds: 2, nrails: 2, g: 2, bub: BUB, hook: None, fold: false };
        let good = build_program(&spec, &ep, n).unwrap();
        let regs: Vec<Vec<Region>> = (0..world).map(|r| regions_for(&io16, n, &parts[r])).collect();
        let ranks = simulate(world, &good, regs, &e0, 2, 9, 0, None).unwrap();
        for rk in &ranks { assert_eq!(out_of(rk, &io16, n), want); }
    }

    /// R10 alignment, pinned by controls. (1) Both rails at the SAME phase (every rank) -> one association for
    /// every chunk, so the result does not depend on how chunks are dealt to rails (1 rail vs 2 rails vs any
    /// payload size). (2) Rails at DIFFERENT phases (identical on every rank) -> still rank-identical and equal
    /// to the per-rail tree, but it differs from the aligned result and from the 1-rail result: the output would
    /// depend on the rail deal — why rail 2 must run the same phase alignment as rail 1. (3) One RANK at a
    /// different phase -> the model refuses (deadlock / wire mismatch) or the result is wrong; never silently right.
    #[test]
    fn rail_and_rank_phase_controls() {
        let world = 4usize;
        let (n, per) = (640usize, 64usize);
        let io = Io { src: SRC, src_f16: true, out: SRC, out_f16: true, wire_f16: true, mid: MID };
        let parts = model_parts(world, n, true);
        let ep = epochs_of(n, per);
        let run = |e0: Vec<Vec<u64>>, nrails: usize| -> Result<Vec<Vec<u32>>, String> {
            let spec = ProgSpec { io, rounds: 2, nrails, g: 2, bub: BUB, hook: None, fold: false };
            let prog = build_program(&spec, &ep, n).unwrap();
            let regs: Vec<Vec<Region>> = (0..world).map(|r| regions_for(&io, n, &parts[r])).collect();
            simulate(world, &prog, regs, &e0, nrails, 11, 0, None).map(|rs| rs.iter().map(|rk| out_of(rk, &io, n)).collect())
        };
        // (1) aligned rails
        let one = run(vec![vec![0]; world], 1).unwrap();
        let two = run(vec![vec![0, 0]; world], 2).unwrap();
        let two_shifted = run(vec![vec![4, 10]; world], 2).unwrap();
        assert_eq!(one, two, "aligned rails: the result must not depend on the rail deal");
        assert_eq!(one, two_shifted, "aligned rails at any (even) start epoch: same association");
        for r in 1..world { assert_eq!(one[r], one[0]); }
        // (2) rails at different phases
        let skew = run(vec![vec![0, 1]; world], 2).unwrap();
        for r in 1..world { assert_eq!(skew[r], skew[0], "a rail-phase skew is still rank-identical"); }
        assert_eq!(skew[0], expected_bits(world, &ep, 2, &[0, 1], &parts, true), "a rail-phase skew is the per-rail tree");
        assert_ne!(skew[0], one[0], "a rail-phase skew changes the output (payload-size dependent bits)");
        // (3) one rank skewed
        let mut e0 = vec![vec![0u64, 0]; world];
        e0[3] = vec![1, 1];
        let bad = run(e0, 2);
        let want = expected_bits(world, &ep, 2, &[0, 0], &parts, true);
        match bad {
            Err(_) => {}        // refused (deadlock / wire mismatch / wrong epoch): the model saw the misalignment
            Ok(outs) => assert!(outs.iter().any(|o| *o != want), "a misaligned rank must not silently produce the tree"),
        }
    }

    /// Recv-slot safety of the REAL pipelined program, from the happens-before DAG. Per rank the stream order
    /// is the Step list; K2m(e) of rank p (rail q) needs the K1m(e) of partner(p, e) on the same rail; a recv
    /// slot (rail, round, e % R) is written by the partner's K1m(e) (plus proxy latency) and read by p's K2m(e),
    /// so the write of epoch e must be ordered AFTER p's K2m(e - lcm(R, rounds)) (the slot's previous
    /// occupant). Checked for every epoch of every program at world 2 and 4 (world 8 is refused, W4P-1), 1 and 2 rails, every
    /// lookahead the real clamp allows. CONTROL: lookahead 6 (> R/2, which `effective_lookahead` refuses)
    /// fails the same check AND the model reports a clobber on some schedule — the check and the model can see
    /// the hazard.
    #[test]
    fn pipelined_recv_slot_safety() {
        fn epochs_of_prog(prog: &[Step], nrails: usize, e0: u64) -> Vec<(u64 /*epoch*/, bool /*is k1m*/, usize /*rail*/)> {
            let mut ctr = vec![e0; nrails];
            prog.iter().map(|s| match *s {
                Step::K1m { rail, .. } => { ctr[rail] += 1; (ctr[rail], true, rail) }
                Step::K2m { rail, lag, .. } => (ctr[rail] - lag as u64, false, rail),
                _ => (0, false, usize::MAX),
            }).collect()
        }
        fn gcd(a: u64, b: u64) -> u64 { if b == 0 { a } else { gcd(b, a % b) } }
        // true when every recv-slot write is ordered after the read of the slot's previous occupant
        fn safe(world: usize, prog: &[Step], nrails: usize) -> (bool, usize) {
            let rounds = rounds_of(world).unwrap() as u64;
            let info = epochs_of_prog(prog, nrails, 0);
            let np = prog.len();
            let id = |r: usize, i: usize| r * np + i;
            let mut succ: Vec<Vec<usize>> = vec![Vec::new(); world * np];
            let (mut k1, mut k2) = (std::collections::HashMap::new(), std::collections::HashMap::new());
            for r in 0..world {
                for i in 0..np {
                    if i + 1 < np { succ[id(r, i)].push(id(r, i + 1)); }
                    match info[i] { (e, true, q) => { k1.insert((r, q, e), i); } (e, false, q) if q != usize::MAX => { k2.insert((r, q, e), i); } _ => {} }
                }
            }
            for r in 0..world {
                for (&(rr, q, e), &i) in k1.iter().filter(|(k, _)| k.0 == r) {
                    let _ = rr;
                    let p = r ^ (1usize << (e % rounds));
                    if let Some(&j) = k2.get(&(p, q, e)) { succ[id(r, i)].push(id(p, j)); }
                }
            }
            let reaches = |from: usize, to: usize| {
                let mut seen = vec![false; world * np];
                let mut st = vec![from];
                seen[from] = true;
                while let Some(x) = st.pop() {
                    if x == to { return true; }
                    for &y in &succ[x] { if !seen[y] { seen[y] = true; st.push(y); } }
                }
                false
            };
            let lcm = RING_SLOTS as u64 * rounds / gcd(RING_SLOTS as u64, rounds);
            let mut checked = 0usize;
            for r in 0..world {
                for (&(_, q, e), &i) in k1.iter().filter(|(k, _)| k.0 == r && k.2 > lcm) {
                    let p = r ^ (1usize << (e % rounds));
                    let prev = e - lcm;
                    let Some(&j) = k2.get(&(p, q, prev)) else { return (false, checked) };
                    checked += 1;
                    if !reaches(id(p, j), id(r, i)) { return (false, checked); }
                }
            }
            (true, checked)
        }
        let mut total = 0usize;
        for &world in &[2usize, 4, 8] {
            let rounds = rounds_of(world).unwrap();
            if world == 8 {
                assert!((1..=4).all(|l| effective_lookahead(l, rounds).is_err()), "world 8 must be refused for every lookahead");
                continue;
            }
            for &nrails in &[1usize, 2] {
                for l in 1..=4usize {
                    let g = effective_lookahead(l, rounds).unwrap();
                    let n = 40 * 64;
                    let ep = epochs_of(n, 64);
                    let spec = ProgSpec { io: Io::f32_inplace(SRC), rounds, nrails, g, bub: BUB, hook: None, fold: false };
                    let prog = build_program(&spec, &ep, n).unwrap();
                    let (ok, checked) = safe(world, &prog, nrails);
                    assert!(ok, "world {world} rails {nrails} g {g}: a recv-slot write is not ordered after the previous occupant's read");
                    assert!(checked > 20, "world {world} rails {nrails} g {g}: only {checked} slot reuses were checked");
                    total += checked;
                }
            }
        }
        assert!(total > 1000, "checked {total} slot reuses");
        // CONTROL: G = 6 > R/2 at world 4
        let (n, world, rounds) = (40 * 64usize, 4usize, 2usize);
        let ep = epochs_of(n, 64);
        let spec = ProgSpec { io: Io::f32_inplace(SRC), rounds, nrails: 1, g: 6, bub: BUB, hook: None, fold: false };
        let prog = build_program(&spec, &ep, n).unwrap();
        assert!(!safe(world, &prog, 1).0, "control: G=6 must fail the happens-before check");
        let parts = model_parts(world, n, false);
        let io = Io::f32_inplace(SRC);
        let mut clobbered = false;
        for seed in 1..=40u64 {
            let mode = (seed % 4) as usize;
            let regs: Vec<Vec<Region>> = (0..world).map(|r| regions_for(&io, n, &parts[r])).collect();
            let e0 = vec![vec![0u64]; world];
            if let Err(m) = simulate(world, &prog, regs, &e0, 1, seed, mode, None) { clobbered |= m.contains("CLOBBER"); }
        }
        assert!(clobbered, "control: the model must catch a recv-slot clobber at G=6 on some schedule");
    }

    /// The hook overlaps at world 4 (the row sub-chunk mechanism is world-general): with a many-chunk payload
    /// the first non-tail hook comes BEFORE the last K1m (rows are handed over while later chunks are still
    /// on the wire), at rounds 1 and 2 (rounds 3 is refused, W4P-1), and each handed range is whole rows >= min_rows.
    #[test]
    fn hook_overlaps_at_world_4() {
        assert!(effective_lookahead(4, 3).is_err(), "rounds 3 (world 8) is refused by the real clamp (W4P-1)");
        for rounds in [1usize, 2] {
            let (row_len, c, per) = (16usize, 160usize, 64usize);
            let n = c * row_len;
            let ep = epochs_of(n, per);
            let g = effective_lookahead(4, rounds).unwrap();
            for nrails in [1usize, 2] {
                let spec = ProgSpec { io: Io::f32_inplace(SRC), rounds, nrails, g, bub: BUB, hook: Some(HookSpec { row_len, min_rows: 4 }), fold: false };
                let prog = build_program(&spec, &ep, n).unwrap();
                let first_hook = prog.iter().position(|s| matches!(s, Step::Hook { tail: false, .. })).expect("no overlapped hook at all");
                let last_k1 = prog.iter().rposition(|s| matches!(s, Step::K1m { .. })).unwrap();
                assert!(first_hook < last_k1, "rounds {rounds} rails {nrails}: the first hook must precede the last K1m");
                let hooks: Vec<(usize, usize)> = prog.iter().filter_map(|s| match *s { Step::Hook { r0, r1, .. } => Some((r0, r1)), _ => None }).collect();
                assert!(hooks.len() >= 3, "rounds {rounds} rails {nrails}: {} hooks", hooks.len());
                assert_eq!(hooks[0].0, 0);
                assert_eq!(hooks.last().unwrap().1, c);
                for w in hooks.windows(2) { assert_eq!(w[0].1, w[1].0); }
            }
        }
    }

    /// TP-4E x TP-4C: the epoch-phase invariant across BOTH teams' arms on the SAME counters. The pipelined
    /// (1- and 2-rail) prefill reduce of this leg and the TP-4C decode arms (mixer / MoE / vp key merge / DH
    /// screen / the serial prefill pair) are chained in one history: the rail-1 counter a pipelined reduce
    /// leaves is the start epoch `run_ops` (TP-4C's model) runs its decode block from, and the counter that
    /// block leaves is where the next pipelined reduce starts. At every hand-off the counter must be a multiple
    /// of `rounds` (every logical reduce of every arm consumes whole rounds; R10 at attach), each decode op is
    /// checked by `run_ops` against the aligned tree, and every pipelined output must equal the tree at the
    /// ALIGNED phase and be bit-identical to the same reduce run from epoch 0 — the output cannot depend on
    /// what ran before it. Rail 2 is prefill-only (decode never touches it): it must also come back aligned.
    /// Proves the schedule arithmetic only (see the section header for what the model does not prove).
    #[test]
    fn pipelined_prefill_interleaves_with_decode_arms_w4() {
        let world = 4usize;
        let rounds = rounds_of(world).unwrap();
        let (n, per) = (640usize, 64usize);
        let ep = epochs_of(n, per);
        let decode_blocks: [&[Op]; 3] = [
            &[Op::Mixer(256), Op::Moe(512), Op::Vp(5), Op::Dh],
            &[Op::Dh, Op::Mixer(64), Op::Vp(16), Op::Prefill(EPOCH_FLOATS + 12), Op::Moe(1024)],
            &[Op::Vp(1), Op::Moe(256)],
        ];
        let (mut pipelined, mut decode_ops) = (0usize, 0usize);
        for (name, io) in sites() {
            for &nrails in &[1usize, 2] {
                let parts = model_parts(world, n, io.src_f16);
                let spec = ProgSpec { io, rounds, nrails, g: 2, bub: BUB, hook: None, fold: false };
                let prog = build_program(&spec, &ep, n).unwrap();
                let nr = nrails.min(ep.len());
                let reference = expected_bits(world, &ep, nr, &vec![0; nrails], &parts, io.out_f16);
                for start in [0u64, 4 * rounds as u64] {
                    let mut ctr: Vec<u64> = vec![start; nrails];
                    for round_trip in 0..2 {
                        for block in decode_blocks {
                            // decode arms on rail 1's counter (rail 2 is prefill-only)
                            assert_eq!(ctr[0] % rounds as u64, 0, "{name} rails {nrails}: rail 1 is unaligned before a decode block");
                            let (e, v) = run_ops(world, ctr[0], block, &real_exch)
                                .unwrap_or_else(|m| panic!("{name} rails {nrails} start {start} trip {round_trip}: decode block {block:?}: {m}"));
                            assert_eq!(v, block.len(), "every decode op must have been verified");
                            assert_eq!(e % rounds as u64, 0, "{name} rails {nrails}: a decode block left rail 1 unaligned");
                            ctr[0] = e;
                            decode_ops += v;
                            // the pipelined prefill reduce from the counters the decode history left
                            let e0: Vec<Vec<u64>> = vec![ctr.clone(); world];
                            let want = expected_bits(world, &ep, nr, &ctr, &parts, io.out_f16);
                            assert_eq!(want, reference, "{name}: the aligned tree must not depend on the start epoch");
                            let regs: Vec<Vec<Region>> = (0..world).map(|r| regions_for(&io, n, &parts[r])).collect();
                            let ranks = simulate(world, &prog, regs, &e0, nrails, 7 + round_trip as u64, round_trip, None)
                                .unwrap_or_else(|m| panic!("{name} rails {nrails} start {start} ctr {ctr:?}: {m}"));
                            for (r, rk) in ranks.iter().enumerate() {
                                assert_eq!(out_of(rk, &io, n), reference, "{name} rails {nrails} ctr {ctr:?}: rank {r} != the aligned tree");
                                assert_eq!(rk.ctr, ranks[0].ctr, "rank {r}: counters disagree across ranks");
                                for (rail, &c) in rk.ctr.iter().enumerate() {
                                    assert!(c >= ctr[rail], "rail {rail} counter went backwards");
                                    assert_eq!(c % rounds as u64, 0, "{name} rails {nrails}: the pipelined reduce left rail {rail} unaligned ({c})");
                                }
                            }
                            ctr = ranks[0].ctr.clone();
                            pipelined += 1;
                        }
                    }
                }
            }
        }
        assert!(pipelined >= 6 * 2 * 2 * 2 * 3 && decode_ops >= 6 * 2 * 2 * 2 * 11, "pipelined {pipelined}, decode ops {decode_ops}");
    }

    // =============================================================================================
    // TP-4S: the world-4 sequence-parallel reduce-scatter / all-gather (CPU model; no hardware)
    // =============================================================================================

    /// A test slot payload: `rows` fp32 rows of `row_len` (so the pieces are small and numerous).
    fn small_payload(row_len: usize, rows: usize) -> usize { rows * row_len * 4 }

    /// The frozen pre-TP-4S inline epoch builder of `reduce_scatter_rows` (world 2), copied verbatim before the
    /// extraction into `rs2_eps`: the extraction must produce the identical epoch list for every input.
    fn frozen_rs2_inline(io: &Io, row_len: usize, own: (usize, usize), peer: (usize, usize), rpe: usize) -> Vec<XEp> {
        let wire16 = io.wire16();
        let (se, oe, wb) = (if io.src_f16 { 2usize } else { 4 }, if io.out_f16 { 2usize } else { 4 }, if wire16 { 2usize } else { 4 });
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
        eps
    }

    /// World 2 is byte-identical: the extracted `rs2_eps` equals the frozen inline block (Debug-compared: every
    /// pointer, length, bits and rows field) over every site, both ranks' blocks, odd/even chunk sizes and several
    /// rows-per-epoch limits. The world-2 executor (`run_xchg`) and the world-2 gather builder are untouched.
    #[test]
    fn world2_reduce_scatter_epochs_are_byte_identical_to_the_frozen_block() {
        let mut n = 0usize;
        for (_, io) in sites() {
            for &row_len in &[8usize, 16, 4096] {
                for &c in &[128usize, 129, 131, 2048, 2047, 4095] {
                    for rank in 0..2usize {
                        let (own, peer) = (row_block(c, rank, 2), row_block(c, 1 - rank, 2));
                        for &rpe in &[1usize, 3, 8, 63, 126, 4096] {
                            let got = format!("{:?}", rs2_eps(&io, row_len, own, peer, rpe));
                            let want = format!("{:?}", frozen_rs2_inline(&io, row_len, own, peer, rpe));
                            assert_eq!(got, want, "io {io:?} row_len {row_len} c {c} rank {rank} rpe {rpe}");
                            n += 1;
                        }
                    }
                }
            }
        }
        assert!(n >= 6 * 3 * 6 * 2 * 6);
    }

    fn rs_ios() -> Vec<(&'static str, Io)> { sites() }

    fn wire_bytes(io: &Io) -> usize { if io.wire16() { 2 } else { 4 } }

    /// Ownership and length pairing at the production widths for EVERY chunk size the SP path can see
    /// (128..=4095 rows): per direction per epoch, what a rank sends to its partner is exactly what the partner
    /// expects (the world > 2 tail check is by equality), the A and B lists have the same shape on every rank
    /// (so the placer deals them identically), every A receive lands one of THIS rank's half's pieces exactly once,
    /// every B receive finishes the own quarter exactly once in row order, and the all-gather delivers every other
    /// rank's quarter exactly once.
    #[test]
    fn sp4_ownership_and_lengths_pair_up_for_every_chunk_size() {
        let payload = EPOCH_FLOATS * 4;
        let row_len = 4096usize;
        let stage = 0x7000_0000u64;
        let ios = [Io { src: 0x1000, src_f16: true, out: 0x1000, out_f16: true, wire_f16: true, mid: stage },
                   Io { src: 0x1000, src_f16: false, out: 0x9000_0000, out_f16: true, wire_f16: false, mid: 0x1000 }];
        for io in ios {
            let wb = wire_bytes(&io);
            for c in 128usize..=4095 {
                let blocks = row_blocks4(c);
                let per: Vec<(Vec<PEp>, Vec<PEp>)> = (0..4).map(|r| rs4_epochs(&io, row_len, &blocks, r, stage.max(io.mid), payload)
                    .unwrap_or_else(|e| panic!("c {c} rank {r}: {e:#}"))).collect();
                for r in 0..4usize {
                    assert_eq!(per[r].0.len(), per[0].0.len());
                    assert_eq!(per[r].1.len(), per[0].1.len());
                    for (i, (xa, xp)) in per[r].0.iter().zip(per[r ^ 2].0.iter()).enumerate() {
                        let Rx::Add { n, .. } = xp.x.rx else { panic!() };
                        assert_eq!(xa.x.bytes, n * wb, "c {c} A epoch {i}: rank {r} sends {} B but rank {} expects {} B", xa.x.bytes, r ^ 2, n * wb);
                        assert!(xa.x.bytes <= payload && xa.x.bytes % 16 == 0 && xa.deps.is_empty());
                    }
                    for (j, (xa, xp)) in per[r].1.iter().zip(per[r ^ 1].1.iter()).enumerate() {
                        let Rx::Add { n, .. } = xp.x.rx else { panic!() };
                        assert_eq!(xa.x.bytes, n * 4, "c {c} B epoch {j}: rank {r} sends {} B but rank {} expects {} B", xa.x.bytes, r ^ 1, n * 4);
                        assert!(xa.x.bytes <= payload && xa.x.bytes % 16 == 0);
                        assert_eq!(xa.deps, vec![2 * (j / 2), 2 * (j / 2) + 1]);
                    }
                    // B receives tile the own quarter in order; A receives tile the rank's half (stage offsets)
                    let own = blocks[r];
                    let mut at = own.0;
                    for xb in &per[r].1 {
                        let Rx::Add { out, n, .. } = xb.x.rx else { panic!() };
                        let oe = if io.out_f16 { 2 } else { 4 };
                        assert_eq!(out, io.out + (at * row_len * oe) as u64, "c {c} rank {r}: B receive not contiguous at row {at}");
                        assert_eq!(n % row_len, 0);
                        assert_eq!(xb.x.rows * row_len, n);
                        at += xb.x.rows;
                    }
                    assert_eq!(at, own.1, "c {c} rank {r}: B epochs do not finish exactly the own quarter");
                    let half = (blocks[r & !1].0, blocks[r | 1].1);
                    let mut landed = vec![0u8; c];
                    for xa in &per[r].0 {
                        let Rx::Add { out, n, .. } = xa.x.rx else { panic!() };
                        let row0 = ((out - stage.max(io.mid)) as usize) / (row_len * 4);
                        for row in row0..row0 + n / row_len { landed[row] += 1; }
                    }
                    for (row, &cnt) in landed.iter().enumerate() {
                        assert_eq!(cnt, (row >= half.0 && row < half.1) as u8, "c {c} rank {r}: A landing of row {row}");
                    }
                }
                // all-gather: pairing and exact-once coverage
                let rb = row_len * 2;
                let ag: Vec<(Vec<PEp>, Vec<PEp>)> = (0..4).map(|r| ag4_epochs(0x5000, rb, &blocks, r, payload).unwrap()).collect();
                for r in 0..4usize {
                    let mut cover = vec![0u8; c];
                    for (q, (xa, xp)) in ag[r].0.iter().zip(ag[r ^ 2].0.iter()).enumerate() {
                        let Rx::Copy { bytes, out } = xp.x.rx else { panic!() };
                        assert_eq!(xa.x.bytes, bytes, "c {c} AG A {q}");
                        let Rx::Copy { bytes: mine, out: mine_out } = xa.x.rx else { panic!() };
                        let row0 = ((mine_out - 0x5000) as usize) / rb;
                        for row in row0..row0 + mine / rb { cover[row] += 1; }
                        let _ = out;
                    }
                    for (j, (xa, xp)) in ag[r].1.iter().zip(ag[r ^ 1].1.iter()).enumerate() {
                        let Rx::Copy { bytes, .. } = xp.x.rx else { panic!() };
                        assert_eq!(xa.x.bytes, bytes, "c {c} AG B {j}");
                        let Rx::Copy { bytes: mine, out: mine_out } = xa.x.rx else { panic!() };
                        let row0 = ((mine_out - 0x5000) as usize) / rb;
                        for row in row0..row0 + mine / rb { cover[row] += 1; }
                    }
                    for (row, &cnt) in cover.iter().enumerate() {
                        let own = blocks[r];
                        assert_eq!(cnt, !(row >= own.0 && row < own.1) as u8, "c {c} rank {r}: AG landing of row {row}");
                    }
                }
            }
        }
    }

    /// The full programs build for every rank at production widths (both rails counts, both lookaheads the W=4 clamp
    /// yields), keep every rail a multiple of `rounds` long, and contain only real-length epochs.
    #[test]
    fn sp4_programs_build_at_production_shapes_and_keep_the_phase() {
        let payload = EPOCH_FLOATS * 4;
        let row_len = 4096usize;
        let stage = 0x7000_0000u64;
        let io = Io { src: 0x1000, src_f16: true, out: 0x1000, out_f16: true, wire_f16: true, mid: stage };
        let gs: Vec<usize> = { let mut v: Vec<usize> = (1..=4).map(|l| effective_lookahead(l, 2).unwrap()).collect(); v.dedup(); v };
        assert_eq!(gs, vec![2, 4]);
        for &c in &[128usize, 129, 255, 513, 1000, 2048, 4095] {
            let blocks = row_blocks4(c);
            for &nr in &[1usize, 2] {
                for &g in &gs {
                    for hook in [None, Some(4usize), Some(32)] {
                        let lens: Vec<Vec<usize>> = (0..4).map(|r| {
                            let prog = sp4_rs_program(&io, row_len, &blocks, r, nr, g, BUB, hook, payload).unwrap();
                            (0..nr).map(|rail| prog.iter().filter(|s| matches!(s, Step::K1m { rail: q, .. } if *q == rail)).count()).collect()
                        }).collect();
                        for l in &lens {
                            assert_eq!(l, &lens[0], "c {c} nr {nr} g {g}: per-rail program lengths differ across ranks");
                            assert!(l.iter().all(|x| x % 2 == 0));
                        }
                    }
                    let ag: Vec<Vec<Step>> = (0..4).map(|r| sp4_ag_program(&[(0x5000, row_len * 2), (0x9000_0000, 64)], &blocks, r, nr, g, BUB, payload).unwrap()).collect();
                    for r in 0..4 {
                        for rail in 0..nr {
                            let n_k1 = |p: &Vec<Step>| p.iter().filter(|s| matches!(s, Step::K1m { rail: q, .. } if *q == rail)).count();
                            assert_eq!(n_k1(&ag[r]), n_k1(&ag[0]));
                            assert_eq!(n_k1(&ag[r]) % 2, 0);
                        }
                    }
                }
            }
        }
    }

    /// Expected output bits of the reduce-scatter over ALL rows: the aligned canonical tree per element (the
    /// same `tree_reference` the all-reduce tests use), then the single f16 rounding when `out_f16`.
    fn rs_expected(parts: &[Vec<f32>], out_f16: bool) -> Vec<u32> {
        tree_reference(4, 0, parts).iter().map(|&v| if out_f16 { H16::from_f32(v).to_bits() as u32 } else { v.to_bits() }).collect()
    }

    /// THE reduce-scatter matrix at world 4 on the CPU device model: every IO; 1 and 2 rails; G 2 and 4; ragged and
    /// odd chunk sizes (down to one-row-per-B-half quarters); small slot payloads so every rank runs many A and B
    /// epochs; with and without the rows-landed hook; a fair scheduler, slow proxy, fast proxy and starved-rank
    /// schedules. Each rank runs ITS OWN program. Every run must finish (no deadlock), never overwrite an
    /// unconsumed recv slot, pass the model's per-epoch length/format check (the tail-equality analogue), leave
    /// each rank's OWN quarter bit-identical to the all-reduce tree (rd_tree_reference order: partner rank^2 then
    /// rank^1), leave every foreign row of a separate `out` untouched, keep each rail's counter a multiple of
    /// `rounds` and advanced by exactly its program length, and hand the hook only final own rows, in order.
    #[test]
    fn sp4_reduce_scatter_model_is_bitwise_the_all_reduce_tree() {
        let (mut runs, mut hooked) = (0usize, 0usize);
        for (name, io) in rs_ios() {
            for &row_len in &[16usize] {
                for &c in &[16usize, 23, 37, 64, 131] {
                    let n = c * row_len;
                    let blocks = row_blocks4(c);
                    let parts = model_parts(4, n, io.src_f16);
                    let want = rs_expected(&parts, io.out_f16);
                    for &rows_cap in &[2usize, 4] {
                        let payload = small_payload(row_len, rows_cap);
                        for &nrails in &[1usize, 2] {
                            for &g in &[2usize, 4] {
                                for hook_min in [None, Some(1usize), Some(5)] {
                                    let progs: Vec<Vec<Step>> = match (0..4).map(|r| sp4_rs_program(&io, row_len, &blocks, r, nrails, g, BUB, hook_min, payload)).collect::<Result<Vec<_>>>() {
                                        Ok(p) => p,
                                        Err(e) => {
                                            // only the documented "a piece cannot split into two non-empty halves" refusal is allowed
                                            assert!(format!("{e:#}").contains("empty"), "{name} c {c} cap {rows_cap}: {e:#}");
                                            continue;
                                        }
                                    };
                                    let refs: Vec<&[Step]> = progs.iter().map(|p| p.as_slice()).collect();
                                    for &e0v in &[0u64, 10] {
                                        let e0: Vec<Vec<u64>> = vec![vec![e0v; nrails]; 4];
                                        let exp = Expect { out: io.out, f16: io.out_f16, bits: want.clone(), row_len };
                                        for (mode, seed) in [(0usize, 1u64), (0, 2), (1, 3), (2, 4), (3, 5), (3, 6)] {
                                            let regs: Vec<Vec<Region>> = (0..4).map(|r| regions_for(&io, n, &parts[r])).collect();
                                            let ranks = simulate_ranks(4, &refs, regs, &e0, nrails, seed, mode, hook_min.map(|_| &exp))
                                                .unwrap_or_else(|m| panic!("{name} c {c} cap {rows_cap} rails {nrails} g {g} hook {hook_min:?} e0 {e0v} mode {mode} seed {seed}: {m}"));
                                            runs += 1;
                                            for (r, rk) in ranks.iter().enumerate() {
                                                let own = blocks[r];
                                                let got = out_of(rk, &io, n);
                                                assert_eq!(&got[own.0 * row_len..own.1 * row_len], &want[own.0 * row_len..own.1 * row_len],
                                                           "{name} c {c} cap {rows_cap} rails {nrails} g {g} hook {hook_min:?} e0 {e0v} mode {mode} seed {seed}: rank {r} own quarter != the all-reduce tree");
                                                if io.out != io.src {
                                                    let marker = if io.out_f16 { 0xFFFFu32 } else { NAN32 };
                                                    for row in (0..c).filter(|&x| x < own.0 || x >= own.1) {
                                                        assert!(got[row * row_len..(row + 1) * row_len].iter().all(|&b| b == marker),
                                                                "{name} c {c}: rank {r} wrote foreign row {row}");
                                                    }
                                                }
                                                for rail in 0..nrails {
                                                    let len = progs[r].iter().filter(|s| matches!(s, Step::K1m { rail: q, .. } if *q == rail)).count();
                                                    assert_eq!(rk.ctr[rail], e0v + len as u64, "rail {rail} counter");
                                                    assert_eq!(len % 2, 0, "rail {rail}: program length must keep the round phase");
                                                }
                                                if let Some(min_rows) = hook_min {
                                                    let mut at = own.0;
                                                    for &(a, b, tail) in &rk.hooks {
                                                        assert_eq!(a, at, "hook ranges must tile the own rows in order");
                                                        assert!(b > a && (tail || b - a >= min_rows || b == own.1));
                                                        at = b;
                                                    }
                                                    assert_eq!(at, own.1, "hooks must cover the whole own quarter");
                                                    hooked += 1;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(runs >= 1500 && hooked >= 500, "runs {runs}, hooked {hooked}");
    }

    /// Bit patterns an add would corrupt: -0.0, NaN payloads, +-inf, denormals, plus ordinary values. Per-rank
    /// distinct so a misrouted row is caught.
    fn ag_pattern(rank: usize, i: usize) -> f32 {
        match i % 7 {
            0 => -0.0,
            1 => f32::from_bits(0x7FC0_0000 | ((rank as u32) << 4) | (i as u32 & 0xF)),   // quiet NaNs with payloads
            2 => if rank % 2 == 0 { f32::INFINITY } else { f32::NEG_INFINITY },
            3 => f32::from_bits(1 + (i as u32 % 5)),                                       // denormals
            _ => ((rank * 1000 + i) as f32) * 0.37 - 11.0,
        }
    }

    /// THE all-gather matrix at world 4: several row buffers as one exchange, plain byte copies — every rank ends
    /// with every owner's quarter bit-for-bit (-0.0 / NaN payload / inf / denormal patterns survive: no add),
    /// under every schedule, with per-rank programs.
    #[test]
    fn sp4_all_gather_model_copies_every_quarter_bit_for_bit() {
        let mut runs = 0usize;
        for &c in &[16usize, 23, 37, 64, 131] {
            let blocks = row_blocks4(c);
            // two buffers with different row widths (floats per row): like x (f16 rows) and the router's rows
            let rowf: [usize; 2] = [16, 8];
            let bases: [u64; 2] = [0x5000, 0x8000_0000];
            for &cap_rows in &[2usize, 5] {
                let payload = cap_rows * 16 * 4;                          // the wide buffer gets `cap_rows` rows per epoch
                for &nrails in &[1usize, 2] {
                    for &g in &[2usize, 4] {
                        let bufs: Vec<(u64, usize)> = (0..2).map(|b| (bases[b], rowf[b] * 4)).collect();
                        let progs: Vec<Vec<Step>> = (0..4).map(|r| sp4_ag_program(&bufs, &blocks, r, nrails, g, BUB, payload).unwrap()).collect();
                        let refs: Vec<&[Step]> = progs.iter().map(|p| p.as_slice()).collect();
                        let truth: Vec<Vec<f32>> = (0..2).map(|b| (0..c * rowf[b]).map(|i| {
                            let row = i / rowf[b];
                            let owner = (0..4).find(|&r| row >= blocks[r].0 && row < blocks[r].1).unwrap();
                            ag_pattern(owner, i)
                        }).collect()).collect();
                        for &e0v in &[0u64, 6] {
                            let e0: Vec<Vec<u64>> = vec![vec![e0v; nrails]; 4];
                            for (mode, seed) in [(0usize, 1u64), (0, 2), (1, 3), (2, 4), (3, 5), (3, 6)] {
                                let regs: Vec<Vec<Region>> = (0..4).map(|r| {
                                    let mut v: Vec<Region> = (0..2).map(|b| Region { base: bases[b], buf: Buf::F32((0..c * rowf[b]).map(|i| {
                                        let row = i / rowf[b];
                                        if row >= blocks[r].0 && row < blocks[r].1 { truth[b][i] } else { f32::from_bits(NAN32) }
                                    }).collect()) }).collect();
                                    v.push(Region { base: BUB, buf: Buf::F32(vec![0.0; 4]) });
                                    v
                                }).collect();
                                let ranks = simulate_ranks(4, &refs, regs, &e0, nrails, seed, mode, None)
                                    .unwrap_or_else(|m| panic!("c {c} cap {cap_rows} rails {nrails} g {g} e0 {e0v} mode {mode} seed {seed}: {m}"));
                                runs += 1;
                                for (r, rk) in ranks.iter().enumerate() {
                                    for b in 0..2 {
                                        let got = rd_bits(&rk.regions, bases[b], c * rowf[b], false).unwrap();
                                        let want: Vec<u32> = truth[b].iter().map(|x| x.to_bits()).collect();
                                        assert_eq!(got, want, "c {c} cap {cap_rows} rails {nrails} g {g} mode {mode}: rank {r} buffer {b} is not the owners' bytes");
                                    }
                                    for rail in 0..nrails {
                                        let len = progs[r].iter().filter(|s| matches!(s, Step::K1m { rail: q, .. } if *q == rail)).count();
                                        assert_eq!(rk.ctr[rail], e0v + len as u64);
                                        assert_eq!(len % 2, 0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(runs >= 200, "runs {runs}");
    }

    /// The dependency checks are real: an adversarial placement that puts a B epoch's K1m before the A epochs it
    /// reads have landed is REFUSED by the emitter, the real placer never produces one, every placement keeps each
    /// rail an even number of slots, and a wrong-length epoch is caught by the model's tail-equality analogue.
    #[test]
    fn sp4_builder_refuses_a_violated_dependency_and_the_model_catches_a_wrong_length() {
        let payload = small_payload(16, 4);
        let io = Io { src: SRC, src_f16: true, out: SRC, out_f16: true, wire_f16: true, mid: MID };
        let (c, row_len) = (64usize, 16usize);
        let blocks = row_blocks4(c);
        let stage = stage_for(&io, c * row_len).unwrap();
        let (a, b) = rs4_epochs(&io, row_len, &blocks, 1, stage, payload).unwrap();
        for &nr in &[1usize, 2] {
            for &g in &[2usize, 4] {
                let rails = place_sp4(&a, &b, nr, g).unwrap();
                assert!(rails.iter().all(|r| r.len() % 2 == 0));
                assert!(emit_sp4_program(&rails, &a, &b, g, BUB, None).is_ok());
                // every item placed exactly once; A on even slots, B on odd slots
                let (mut na, mut nb) = (0, 0);
                for sq in &rails { for (s, cell) in sq.iter().enumerate() { match cell { Cell4::A(_) => { na += 1; assert_eq!(s % 2, 0); }
                    Cell4::B(_) => { nb += 1; assert_eq!(s % 2, 1); } Cell4::Bubble => {} } } }
                assert_eq!((na, nb), (a.len(), b.len()));
                // adversarial: B(0) at slot 1 of its rail, before A's K2 (needs slot >= 2*(d/nr) + g + 1)
                let mut bad = rails.clone();
                let r0 = 0usize;
                let cur = bad[r0].iter().position(|c| *c == Cell4::B(0)).unwrap();
                bad[r0][cur] = Cell4::Bubble;
                bad[r0][1] = Cell4::B(0);
                let why = format!("{:#}", emit_sp4_program(&bad, &a, &b, g, BUB, None).err().expect("violated dependency must be refused"));
                assert!(why.contains("before A epoch"), "{why}");
            }
        }
        // an out-of-range dependency is refused by the placer
        let mut b2 = b.clone();
        b2[0].deps = vec![a.len()];
        assert!(place_sp4(&a, &b2, 2, 2).is_err());
        // a wrong-length epoch: rank 0's A epoch 0 sends one row too many -> the receiver's length check fires
        let ios = io;
        let parts = model_parts(4, c * row_len, true);
        let mut progs: Vec<Vec<Step>> = (0..4).map(|r| sp4_rs_program(&ios, row_len, &blocks, r, 1, 2, BUB, None, payload).unwrap()).collect();
        let k = progs[0].iter().position(|s| matches!(s, Step::K1m { nbytes, .. } if *nbytes > 16)).unwrap();
        if let Step::K1m { nbytes, .. } = &mut progs[0][k] { *nbytes += 16; }
        let refs: Vec<&[Step]> = progs.iter().map(|p| p.as_slice()).collect();
        let regs: Vec<Vec<Region>> = (0..4).map(|r| regions_for(&ios, c * row_len, &parts[r])).collect();
        let err = simulate_ranks(4, &refs, regs, &vec![vec![0u64]; 4], 1, 1, 0, None).err().expect("a wrong-length epoch must be caught");
        assert!(err.contains("length mismatch") || err.contains("outside every region"), "{err}");
    }
}
