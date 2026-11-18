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

//!
//! File map (the former single `tp_xport.rs`; every item keeps its `crate::tp_xport::` path):
//! - `mod.rs`: the launch schedules (serial, pipelined, hooked, sequence-parallel w2 / w4) and their pure planners
//! - `tests.rs`: the former inline `#[cfg(test)] mod tests` (the CPU device model, the frozen pre-TP-4E loop, schedule proofs); one module, kept whole.
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
            let ptx = cudarc::nvrtc::Ptx::from_src(crate::ptx_embed::GPU_BATCH);
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
mod tests;
