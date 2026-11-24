//! DSV4.1 TP (leg TPC): the fp32 sum all-reduce over the engine's RDMA doorbell transport (`tp_xport`, reused as
//! is). Copied from the Naive family's attach (`src/naive/tp.rs`, read only) and trimmed to what a dsv41 round
//! needs:
//! - **world 2**: the serial K1 / K2 pair on the primary link (one exchange: `p0 + p1`), the multi-block K1 (16 B
//!   stores over up to 32 blocks) for decode-width frames;
//! - **world 4**: every frame up to `SINGLE_CAP_BYTES` takes ONE all-peers stage on a dedicated uniform one-shot ctx
//!   (`tp_wait_add_4way_f32`: `(p0 + p2) + (p1 + p3)`, bitwise the aligned two-stage tree), the round phase of the
//!   primary link aligned at attach (R10);
//! - **wider frames** (24 rows with the Engram rider: 2.95 MB) take the pipelined multi-block transport
//!   (`tp_xport::reduce_pipe`, lookahead 4, 16 blocks), whose per-element sums are the serial schedule's.
//!
//! Everything is a kernel launch on the caller's stream (no host sync): graph-capturable. A transport abort is
//! cooperative (the kernels no-op), so callers check `abort_code` after a round (`Dsv41Comm::check`).

use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchAsync, LaunchConfig};
use std::sync::Arc;

const TP_MODULE: &str = "dsv41_tp";
const SINGLE_MODULE: &str = "dsv41_tp_single";
const SINGLE_K2: &str = "tp_wait_add_4way_f32";
const SINGLE_K2_GPU: &str = "tp_wait_add_4way_f32g";
const SINGLE_K2_8: &str = "tp_wait_add_8way_f32";
const SINGLE_K1_MB: &str = "tp_gate_copy_signal_os_mb";
/// the primary link's proxy core and the single-stage ctx's (the engine's pinned-proxy rule)
const TP_PROXY_CORE: i32 = 19;
const TP_SINGLE_PROXY_CORE: i32 = 17;
/// leg TX1: the rail-2 twin's proxy core (the engine's aux rule: 18 at world 2; the single-stage twin takes 16)
const TP_DUAL_PROXY_CORE: i32 = 16;
const TP_DUAL2_PROXY_CORE: i32 = 18;
/// leg TX1: rail-2 TCP port offset for world 2 (same as EXL3 second rail: TP_PORT + 10).
/// World 4/8 N-way bases are computed dynamically by `xtp::dual_single_nway_base(world)` / `xtp::rail2_nway_base(world)`.
const TP_DUAL2_PORT_OFF: u16 = 10;
/// default `--dsv41-dual-rail-min-kib` (smoke sweep on the Sparks, fused dual kernels: a 1-row frame, 20 KiB, gains 14 % split):
/// frames (or ring epochs) of at least this many KiB are split over both rails
pub const DUAL_MIN_BYTES: usize = 16 * 1024;
/// frames above this take the pipelined transport by default (`--dsv41-tp-pipe-min-kib`). Measured at TP2 (.11 + .14,
/// 2026-11-19): the pair is faster for a 1-row frame (20.6 KB: 22.5 vs 24.6 us p50), the pipe from 4 rows on (123 KB:
/// 32.8 vs 45.1 us); the crossover interpolates to ~35 KB, so 2-row frames (41 KB) take the pipe. The EXL3 / Naive
/// threshold is 256 KiB (at 12 rows that costs 69.6 against 45.1 us per layer reduce).
pub const PIPE_MIN_BYTES: usize = 32 * 1024;
/// leg PX1: the default threshold with the dsv41 exchange pair: one ring slot's epoch (`tp_xport::EPOCH_FLOATS`), so
/// every decode / verify frame is one exchange and only frames wider than a slot (the 24-row Engram frames) pipe.
pub const PIPE_MIN_BYTES_XCHG: usize = crate::tp_xport::EPOCH_FLOATS * 4;
/// the single-stage ctx's ring-slot payload bound (world 4) with the shared pair
pub const SINGLE_CAP_BYTES: usize = 256 * 1024;
/// leg PX2: the bound with the dsv41 world-4 pair (`--dsv41-tp4-single-kib`): 768 KiB holds every decode / verify frame
/// up to 24 rows (24 x 20,608 B = 494.6 KB) and the 6-row Engram-rider frame (6 x 123,008 B = 738.0 KB) in ONE stage;
/// only the 12- and 24-row Engram-rider frames (1.48 / 2.95 MB) are wider and take the pipe.
pub const SINGLE_CAP_BYTES_XCHG4: usize = 768 * 1024;

/// The multi-block K1's grid for an `n`-float payload (one 16 B store per thread per block of 256, at most 32 blocks).
pub fn k1_mb_blocks(n: usize) -> u32 { ((n / 4).div_ceil(256)).clamp(1, 32) as u32 }

/// Which path a frame of `bytes` takes at `world` (the routing rule, pure).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path { Pipe, Single, Serial }
pub fn route(bytes: usize, world: usize, single_on: bool, pipe_on: bool, pipe_min: usize) -> Path {
    route_cap(bytes, world, single_on, pipe_on, pipe_min, SINGLE_CAP_BYTES)
}
/// `route` with the single-stage ctx's payload bound `single_cap` (leg PX2: larger with the dsv41 world-4 pair).
pub fn route_cap(bytes: usize, world: usize, single_on: bool, pipe_on: bool, pipe_min: usize, single_cap: usize) -> Path {
    if pipe_on && bytes > pipe_min && (bytes / 4) % 4 == 0 { return Path::Pipe; }
    if single_on && world >= 3 && bytes <= single_cap { return Path::Single; }
    Path::Serial
}

/// leg PX2 / TX1: the default `--dsv41-tp4-rsag-min-kib`: world-4 frames from this size take reduce-scatter + all-gather.
/// PX2 predicted 192 KiB; SPT4's A/B on the Sparks found 600 KiB (with the pipe from 384 KiB) wins at 24 rows (-10.8 %).
pub const RSAG_MIN_BYTES: usize = 600 * 1024;
/// Phase 1D: with dual-rail RS+AG active (`pctx` up), 0.75F/rail beats dual-rail single-stage (1.5F/rail) from 384 KiB
/// (A/B on .15-.18: 24-row 94-graph 10,346 vs 10,776 us, -430 us/round; 12-row 7,377 vs 7,488 us).
pub const RSAG_MIN_BYTES_DUAL: usize = 384 * 1024;
/// leg TX1: the default `--dsv41-tp4-pipe-min-kib` with the dsv41 world-4 pair (SPT4: 24-row frames, 494,592 B, go to the pipe;
/// 12-row frames and smaller stay on one stage / reduce-scatter).
pub const PIPE_MIN_BYTES_XCHG4: usize = 384 * 1024;
/// Phase A: default `--dsv41-tp8-rsag-min-kib` (world 8: 7.0F -> 1.75F, 4x wire reduction).
pub const RSAG8_MIN_BYTES: usize = 256 * 1024;
pub const RSAG8_MIN_BYTES_DUAL: usize = 128 * 1024;

/// Phase A: whether an `n`-float world-8 frame takes reduce-scatter + all-gather (single-rail).
pub fn use_rsag8(n: usize, rsag_min: usize, single_cap: usize) -> bool {
    rsag_min > 0 && n * 4 >= rsag_min && n % 32 == 0 && (n / 2) * 4 + 64 <= crate::tp::TP_SLOT_BYTES && (n / 8) * 4 <= single_cap
}
/// Phase A: world-8 dual-rail RS+AG splits `n` into two 128 B-aligned halves `(ha, hb)`.
pub fn use_rsag8_dual(n: usize, rsag_min: usize, single_cap: usize) -> bool {
    if rsag_min == 0 || n * 4 < rsag_min { return false; }
    let Some((ha, hb)) = super::xchg::dual_rsag8_split(n) else { return false; };
    let hi = ha.max(hb);
    (hi / 2) * 4 + 64 <= crate::tp::TP_SLOT_BYTES && (hi / 8) * 4 <= single_cap
}

/// Phase A: resolve zero-CPU-bounce GPU-validated collect for the dsv41 PDL pair (TP2 / TP4 / TP8).
/// Default ON (`auto`/unset/`on`/`1`/`true`); explicit `off`/`0`/`false` falls back to v1 CPU bounce.
pub fn gpu_recv_resolve(raw: Option<&str>) -> bool {
    !matches!(raw.map(|v| v.trim().to_ascii_lowercase()).as_deref(), Some("0" | "off" | "false"))
}

/// Phase A: resolve opt-in BF16 wire compression (`--dsv41-wire-bf16 on|off`, default off).
pub fn wire_bf16_resolve(raw: Option<&str>) -> bool {
    matches!(raw.map(|v| v.trim().to_ascii_lowercase()).as_deref(), Some("1" | "on" | "true"))
}

/// Whether an `n`-float world-4 frame takes reduce-scatter + all-gather: at least `rsag_min` bytes (0 = never), a 64 B
/// multiple (quarters of whole float4s), its half within one primary ring slot and its quarter within the single-stage
/// bound.
pub fn use_rsag(n: usize, rsag_min: usize, single_cap: usize) -> bool {
    rsag_min > 0 && n * 4 >= rsag_min && n % 16 == 0 && (n / 2) * 4 + 64 <= crate::tp::TP_SLOT_BYTES && (n / 4) * 4 <= single_cap
}
/// Phase 1D: dual-rail RS+AG splits `n` into two 64 B-aligned halves `(ha, hb)` so each rail carries half as much
/// (`max(ha,hb)/2` <= 1 MiB primary slot, `max(ha,hb)/4` <= `single_cap`), doubling max RS+AG frame size to ~4 MB.
pub fn use_rsag_dual(n: usize, rsag_min: usize, single_cap: usize) -> bool {
    if rsag_min == 0 || n * 4 < rsag_min { return false; }
    let Some((ha, hb)) = super::xchg::dual_rsag_split(n) else { return false; };
    let hi = ha.max(hb);
    (hi / 2) * 4 + 64 <= crate::tp::TP_SLOT_BYTES && (hi / 4) * 4 <= single_cap
}

/// leg PX2: the pipe threshold and the single-stage bound at `world` (bytes): `tp4_pipe` = `--dsv41-tp4-pipe-min-kib`,
/// `tp_pipe` = `--dsv41-tp-pipe-min-kib` (world 2, and world 4 when the TP4 one is not given), `cap` =
/// `--dsv41-tp4-single-kib`; `pair2` / `pair4` = the dsv41 pair serves that world. Pure (unit-tested).
pub fn thresholds(world: usize, pair2: bool, pair_n: bool, tp_pipe: Option<usize>, tp4_pipe: Option<usize>, cap: Option<usize>) -> (usize, usize) {
    let kib = |k: usize| k * 1024;
    let single_cap = cap.map(kib).unwrap_or(if pair_n { SINGLE_CAP_BYTES_XCHG4 } else { SINGLE_CAP_BYTES });
    let pipe_min = if world >= 3 {
        tp4_pipe.or(tp_pipe).map(kib).unwrap_or(if pair_n { single_cap.min(PIPE_MIN_BYTES_XCHG4) } else { PIPE_MIN_BYTES })
    } else {
        tp_pipe.map(kib).unwrap_or(if pair2 { PIPE_MIN_BYTES_XCHG } else { PIPE_MIN_BYTES })
    };
    (pipe_min, single_cap)
}

/// leg TX1 + Phase 1D: the rail-2 twin ctx(s) of the exchange links (world 4: single-stage twin `ctx` + primary tree twin `pctx`;
/// world 2: primary twin `ctx`, `pctx = None`).
struct Dual { ctx: u64, pctx: Option<u64>, min_bytes: usize, dev: String }

/// `--dsv41-dual-rail on|off|auto` (default auto).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DualMode { On, Off, Auto }
pub fn dual_mode_of(raw: Option<&str>) -> DualMode {
    match raw.map(|v| v.trim().to_ascii_lowercase()).as_deref() { Some("on") => DualMode::On, Some("off") => DualMode::Off, _ => DualMode::Auto }
}
const DUAL_MAGIC: u32 = 0xD41D_0A10;

/// Agree over the primary link's control plane whether every rank can bring up rail 2. Returns the device and a
/// one-line verdict; `None` + why = run single-rail (never a refusal: owner rule 8).
fn dual_agree(rank: usize, _world: usize, pair_ok: bool) -> (Option<String>, String) {
    let mode = dual_mode_of(crate::opts::get(crate::opt!("dsv41-dual-rail")).as_deref());
    if mode == DualMode::Off { return (None, "--dsv41-dual-rail off".into()); }
    let devs = crate::tp::rdma_devs();
    let dev2 = devs.get(1).cloned();
    let mut local = if !pair_ok { Err("the dsv41 PDL exchange pair is not serving (--dsv41-xchg legacy or --tp-gpu-recv on)".to_string()) }
                    else if let Some(d) = &dev2 {
        crate::exl3_forward::xtp::rail_local_preflight(d, crate::tp::resolve_gid_idx(d)).map_err(|e| format!("{e:#}"))
    } else { Err(format!("only one RDMA device ({}) on this box", devs.join(","))) };
    if let Err(e) = &mut local { *e = format!("rank {rank}: {e}"); }
    let word = DUAL_MAGIC + local.is_ok() as u32;
    let all = crate::net::exchange_u32s_all(&[word, rank as u32], 4);
    match all {
        Ok(all) if all.iter().all(|f| f[0] == DUAL_MAGIC + 1) && local.is_ok() => (dev2, format!("rail 2 = {} on every rank", devs[1])),
        Ok(all) => {
            let bad: Vec<String> = all.iter().enumerate().filter(|(_, f)| f[0] != DUAL_MAGIC + 1).map(|(r, _)| r.to_string()).collect();
            (None, format!("no usable second rail on rank(s) {} ({}){}", bad.join(","), local.err().unwrap_or_else(|| "this rank is fine".into()),
                           if mode == DualMode::On { "; --dsv41-dual-rail on asked for it: running on rail 1 only" } else { "" }))
        }
        Err(e) => (None, format!("the rail-2 agreement failed ({e:#}): running on rail 1 only")),
    }
}

struct Single {
    ctx: u64,
    k2: CudaFunction,
    k1m: (CudaFunction, CudaSlice<u32>),
    gpu: bool,
}

pub struct RdmaXport {
    pub rank: usize,
    pub world: usize,
    ctx: u64,
    k1: CudaFunction,
    k2: CudaFunction,
    k1m: (CudaFunction, CudaSlice<u32>),
    single: Option<Single>,
    /// leg TX1: the rail-2 twin (`--dsv41-dual-rail`)
    dual: Option<Dual>,
    pipe: Option<(crate::tp_xport::Kernels, crate::tp_xport::Pipe)>,
    /// frames above this many bytes take the pipe
    pipe_min: usize,
    /// leg PX2: the single-stage ctx's payload bound (world 4)
    single_cap: usize,
    /// leg PX2: world-4 frames from this many bytes take reduce-scatter + all-gather (0 = never)
    rsag_min: usize,
    /// host addresses of the link ctxs (their abort words)
    links: Vec<usize>,
    /// the once-per-boot description of the path (engagement receipt)
    pub describe: String,
    /// leg PX1: the dsv41 PDL exchange pair (world 2, `--dsv41-xchg pdl`, the default: `pair`) and / or its trace; None or `!pair` = the shared K1m / K2
    xchg: Option<super::xchg::Xchg>,
}

/// `--tp-reduce auto|rd|single` at `world`: (single stage on, why). World 2 reduces in one stage already.
pub fn single_resolve(raw: Option<&str>, world: usize) -> (bool, String) {
    if world >= 3 && !world.is_power_of_two() {
        return (true, format!("world {world}: direct all-peers one-shot + 2-RTT direct RS/AG"));
    }
    match (raw.map(|v| v.trim().to_ascii_lowercase()).as_deref(), world) {
        (Some("rd"), _) => (false, "--tp-reduce rd: the recursive-doubling tree".into()),
        (_, 4 | 8) => (true, format!("--tp-reduce auto|single = one all-peers stage at world {world}")),
        (Some("single"), w) => (false, format!("--tp-reduce single asked, world {w} reduces in ONE stage already (the pair runs)")),
        (_, w) => (false, format!("world {w}: one exchange per reduce")),
    }
}

static VF_WORLD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static VF_DUAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static VF_GPU_RECV: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static VF_SP_READY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Telemetry snapshot for `/metrics` (`velogb10_velo_fabric_*`): `(world, dual_rail, gpu_recv, sp_ready)`.
pub fn fabric_telemetry() -> (u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (VF_WORLD.load(Relaxed), VF_DUAL.load(Relaxed), VF_GPU_RECV.load(Relaxed), VF_SP_READY.load(Relaxed))
}

impl RdmaXport {
    /// Attach the brought-up link: sanity + branch agree, the doorbell kernels, the proxy thread(s), R10 alignment,
    /// the world-4 single-stage ctx, the pipelined wide path, and a self-test of every path.
    pub fn attach(dev: &Arc<CudaDevice>, ctx: crate::tp::TpContext, row_bytes: usize) -> Result<Self> {
        Self::attach_with_branch(dev, ctx, row_bytes, &crate::tp::TpBranch::Dsv41)
    }

    /// Reusable branch-parameterized attach (used by `crate::dist::FabricXport` so any model family — e.g. Inkling,
    /// Naive, Qwen, DSV4.1 — gets the full dual-rail PDL / single-CTA C2C / 2-WR / lock-free verbs / dual-rail RS+AG
    /// transport without duplicating RDMA bring-up).
    pub fn attach_with_branch(dev: &Arc<CudaDevice>, mut ctx: crate::tp::TpContext, row_bytes: usize, branch: &crate::tp::TpBranch) -> Result<Self> {
        ensure!(matches!(ctx.world, 2 | 3 | 4 | 6 | 8), "velo-fabric TP transport: world {} (2, 3, 4, 6, or 8)", ctx.world);
        ensure!(!(ctx.world > 2 && (ctx.world as usize).is_power_of_two() && ctx.link.oneshot_on()), "dsv41 TP world {}: the one-shot primary link is not taken", ctx.world);
        ctx.sanity()?;
        ctx.branch_check(branch)?;
        let (rank, world, mut link) = ctx.into_parts();
        let kern = crate::tp_xport::Kernels::load(dev, TP_MODULE)?;
        let f = |m: &str, n: &str| dev.get_func(m, n).with_context(|| format!("doorbell kernel {n} missing"));
        let (k1, k2) = (f(TP_MODULE, "tp_gate_copy_signal")?, f(TP_MODULE, "tp_wait_add")?);
        let k1mf = f(TP_MODULE, "tp_gate_copy_signal_mb")?;
        link.set_payload(row_bytes, true).context("net_set_payload")?;
        // leg PX1: the dsv41 exchange pair at world 2 (`--dsv41-xchg`), and the trace (`--dsv41-xchg-trace`): the proxy's
        // per-epoch stamps must be switched on before it starts
        let trace_dir = crate::opts::get(crate::opt!("dsv41-xchg-trace")).filter(|d| !d.trim().is_empty());
        if trace_dir.is_some() { link.bench_config(0, true); }
        let log_tag = if matches!(branch, crate::tp::TpBranch::Dsv41) { "dsv41" } else { "velo-fabric" };
        let gpu_recv = gpu_recv_resolve(crate::opts::var(crate::opt!("tp-gpu-recv")).ok().as_deref());
        let wire_bf16 = wire_bf16_resolve(crate::opts::get(crate::opt!("dsv41-wire-bf16")).as_deref());
        let pair = world == 2 && super::xchg::mode() == super::xchg::Mode::Pdl;
        let mut xchg = if pair || (world == 2 && trace_dir.is_some()) {
            Some(super::xchg::Xchg::new_with_tag(dev, 2, rank as usize, pair, trace_dir.is_some(), gpu_recv, wire_bf16, log_tag).context("velo-fabric xchg")?)
        } else { None };
        let cptr = link.ctx_device_ptr();
        let addr = link.ctx_addr();
        let peer_ip = link.peer_ip();
        std::mem::forget(link);   // the proxy owns the transport from here; the OS reclaims at exit
        crate::net::spawn_proxy(addr, TP_PROXY_CORE);
        if let (Some(x), Some(d)) = (&xchg, &trace_dir) { x.start_dump(dev, d, rank as usize, addr)?; }
        let mut arrive = dev.alloc_zeros::<u32>(1)?;
        dev.memset_zeros(&mut arrive)?;   // AGENTS §2.2
        let scratch = dev.alloc_zeros::<f32>(16)?;
        dev.synchronize()?;
        let (rank, world) = (rank as usize, world as usize);
        if world > 2 && world.is_power_of_two() {
            // R10: a logical reduce at world 4/8 is log2(W) epochs (round = epoch % rounds); the counter must be a
            // multiple of rounds before the first reduce on every rank (non-power-of-two worlds have no XOR tree).
            let rounds = world.trailing_zeros() as u64;
            let e0 = crate::net::traced_device_epoch();
            let pad = ((rounds - e0 % rounds) % rounds) as usize;
            if pad > 0 {
                let stream = crate::gpu::fork_blocking_stream(dev);
                let c1 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
                let p = *scratch.device_ptr();
                for _ in 0..pad {
                    unsafe { k1.clone().launch_on_stream(&stream, c1, (cptr, p, 16u32)) }?;
                    unsafe { k2.clone().launch_on_stream(&stream, c1, (cptr, p, p, 4i32, 2i32)) }?;
                }
                unsafe { cudarc::driver::result::stream::synchronize(stream.stream) }?;
            }
            let e1 = crate::net::traced_device_epoch();
            ensure!(e1 % rounds == 0, "dsv41 TP: device epoch {e1} not a multiple of {rounds} after alignment");
        }
        let (single_on, why) = single_resolve(crate::opts::var(crate::opt!("tp-reduce")).ok().as_deref(), world);
        // Phase A + Arbitrary-W: velo-fabric PDL pair at world >= 3 (3, 4, 6, 8) supports zero-CPU-bounce GPU-validated collect!
        let pair_n = world >= 3 && single_on && super::xchg::mode() == super::xchg::Mode::Pdl;
        let kib = |o: crate::opts::OptId| crate::opts::get(o).and_then(|v| v.trim().parse::<usize>().ok());
        let (mut pipe_min, single_cap) = thresholds(world, pair, pair_n, kib(crate::opt!("dsv41-tp-pipe-min-kib")),
                                                kib(crate::opt!("dsv41-tp4-pipe-min-kib")), kib(crate::opt!("dsv41-tp4-single-kib")));
        let mut rsag_min = if world == 4 && pair_n {
            kib(crate::opt!("dsv41-tp4-rsag-min-kib")).map_or(RSAG_MIN_BYTES, |k| k * 1024)
        } else if world == 8 && pair_n {
            kib(crate::opt!("dsv41-tp8-rsag-min-kib")).map_or(RSAG8_MIN_BYTES, |k| k * 1024)
        } else { 0 };
        let mut single_addr = 0usize;
        let single = if single_on {
            let base = crate::exl3_forward::xtp::single_nway_base(world as i32)?;
            let ips = crate::tp::resolve_topology(world as i32)?;
            let dev1 = crate::tp::rdma_dev();
            let mut l3 = crate::net::TpLink::connect_nway_uniform(rank as i32, world as i32, &ips, base, &dev1,
                                                                  crate::tp::resolve_gid_idx(&dev1), single_cap)
                .with_context(|| format!("dsv41 TP: single-stage ctx ({dev1}, TCP base {base}, world {world})"))?;
            l3.set_payload(single_cap, true).context("single-stage ctx net_set_payload")?;
            // Enable v2 GPU-validated receive on the dedicated oneshot_uniform ctx when PDL pair uses GPU-recv
            // (or legacy world-4 SINGLE_K2_GPU). Legacy world-8 uses CPU-bounce SINGLE_K2_8.
            let gpu = gpu_recv && (pair_n || world == 4);
            if gpu { l3.set_recv_mode(true).context("single-stage ctx net_set_recv_mode")?; }
            l3.set_parent(addr);
            let ctx3 = l3.ctx_device_ptr();
            single_addr = l3.ctx_addr();
            std::mem::forget(l3);
            crate::net::spawn_proxy_aux(single_addr, TP_SINGLE_PROXY_CORE);
            if pair_n || trace_dir.is_some() {
                let x = super::xchg::Xchg::new_with_tag(dev, world, rank, pair_n, trace_dir.is_some(), gpu, wire_bf16, log_tag)
                    .with_context(|| format!("{log_tag} xchg (world {world})"))?;
                if let Some(d) = &trace_dir { x.start_dump(dev, d, rank, single_addr)?; }
                xchg = Some(x);
            }
            if dev.get_func(SINGLE_MODULE, SINGLE_K2).is_none() {
                dev.load_ptx(cudarc::nvrtc::Ptx::from_src(crate::ptx_embed::GPU_BATCH), SINGLE_MODULE, &[SINGLE_K2, SINGLE_K2_GPU, SINGLE_K2_8, SINGLE_K1_MB])
                    .context("load the single-stage kernels (gpu_batch.ptx)")?;
            }
            let k2_name = if world == 8 { SINGLE_K2_8 } else if gpu { SINGLE_K2_GPU } else { SINGLE_K2 };
            let k2s = f(SINGLE_MODULE, k2_name)?;
            let mut arr = dev.alloc_zeros::<u32>(1)?;
            dev.memset_zeros(&mut arr)?;
            Some(Single { ctx: ctx3, k2: k2s, k1m: (f(SINGLE_MODULE, SINGLE_K1_MB)?, arr), gpu })
        } else { None };
        // leg TX1 (`--dsv41-dual-rail`): the rail-2 twin of the exchange ctx. Agreed over the primary link's control plane
        // (every rank must have a usable second device), brought up like the EXL3 second rail; a rank that cannot
        // does not refuse, every rank runs on rail 1 and the boot line says why.
        let pair_ok = xchg.as_ref().map_or(false, |x| x.pair);
        let (dual, dual_addr, dual_p2_addr, dual_line) = {
            let (d2, why) = dual_agree(rank, world, pair_ok);
            match d2 {
                None => (None, 0usize, 0usize, format!("dual rail OFF: {why}")),
                Some(dev2) => {
                    let gid2 = crate::tp::resolve_gid_idx(&dev2);
                    let min_bytes = kib(crate::opt!("dsv41-dual-rail-min-kib")).map_or(DUAL_MIN_BYTES, |k| k * 1024);
                    let tcp = |r: Result<crate::net::TpLink>| r.with_context(|| format!("dsv41 dual rail: rail-2 ctx on {dev2}"));
                    let (mut l2, how, core, p2_link) = if world >= 3 {
                        let base = crate::exl3_forward::xtp::dual_single_nway_base(world as i32)?;
                        let ips = crate::tp::resolve_topology(world as i32)?;
                        let l = tcp(crate::net::TpLink::connect_nway_uniform(rank as i32, world as i32, &ips, base, &dev2, gid2, single_cap))?;
                        if world.is_power_of_two() {
                            // Phase 1D + Phase 2: also bring up the rail-2 primary tree twin so world-4/8 Pipe (wide/prefill)
                            // and world-4/8 dual-rail RS+AG use BOTH RoCE rails (dedicated proxy on core TP_DUAL2_PROXY_CORE = 18).
                            let p2_base = crate::exl3_forward::xtp::rail2_nway_base(world as i32)?;
                            let mut lp2 = crate::net::TpLink::connect_nway(rank as i32, world as i32, &ips, p2_base, &dev2, gid2, crate::tp::TP_SLOT_BYTES)
                                .with_context(|| format!("dsv41 dual rail: rail-2 primary tree twin on {dev2} (TCP base {p2_base})"))?;
                            lp2.set_payload(row_bytes, true).context("dual rail primary twin net_set_payload")?;
                            lp2.set_parent(addr);
                            let p2_c = lp2.ctx_device_ptr();
                            let p2_a = lp2.ctx_addr();
                            std::mem::forget(lp2);
                            crate::net::spawn_proxy_aux(p2_a, TP_DUAL2_PROXY_CORE);
                            let e_p2 = crate::net::ctx_device_epoch(p2_a);
                            ensure!(e_p2 % (world.trailing_zeros() as u64) == 0,
                                    "dsv41 dual rail: rail-2 primary twin epoch {e_p2} not aligned to rounds");
                            (l, format!("single-stage twin (TCP base {base}, core {TP_DUAL_PROXY_CORE}) + primary tree twin (TCP base {p2_base}, core {TP_DUAL2_PROXY_CORE})"),
                             TP_DUAL_PROXY_CORE, Some((p2_c, p2_a)))
                        } else {
                            (l, format!("single-stage twin (TCP base {base}, core {TP_DUAL_PROXY_CORE})"),
                             TP_DUAL_PROXY_CORE, None)
                        }
                    } else {
                        let port = crate::tp::TP_PORT + TP_DUAL2_PORT_OFF;
                        if rank != 0 { std::thread::sleep(std::time::Duration::from_millis(300)); }
                        let l = tcp(crate::net::TpLink::connect(rank as i32, if rank == 0 { "" } else { &peer_ip }, port, &dev2, gid2, crate::tp::TP_SLOT_BYTES))?;
                        (l, format!("primary twin, TCP {port}"), TP_DUAL2_PROXY_CORE, None)
                    };
                    l2.set_payload(if world >= 3 { single_cap } else { row_bytes }, true).context("dual rail net_set_payload")?;
                    // Phase A: match rail-1 oneshot_uniform GPU-recv mode on rail-2 single-stage twin (world >= 3).
                    // At world 2, rail-2 primary twin keeps v1 so Pipe works on it while d41x_xc_collect_g runs hybrid gpu_rx_skips!
                    if world >= 3 && single.as_ref().map_or(false, |s| s.gpu) {
                        l2.set_recv_mode(true).context("dual rail single-stage twin net_set_recv_mode")?;
                    }
                    l2.set_parent(addr);
                    let c2 = l2.ctx_device_ptr();
                    let a2 = l2.ctx_addr();
                    std::mem::forget(l2);
                    crate::net::spawn_proxy_aux(a2, core);
                    dev.synchronize()?;
                    let (p2_c, p2_a) = match p2_link { Some((pc, pa)) => (Some(pc), pa), None => (None, 0usize) };
                    (Some(Dual { ctx: c2, pctx: p2_c, min_bytes, dev: dev2.clone() }), a2, p2_a,
                     format!("dual rail ON: {why}; {how}; exchanges of >= {} KiB are split in two 16 B-aligned halves, one per rail", min_bytes / 1024))
                }
            }
        };
        // leg TX1: with both rails up the one-stage exchange beats the pipe at 24 rows too (smoke: 12.34 vs 13.37 ms a round), so the
        // default pipe threshold rises to the single-stage bound; an explicit --dsv41-tp4-pipe-min-kib / --dsv41-tp-pipe-min-kib stays
        if dual.is_some() && world >= 3 && pair_n && kib(crate::opt!("dsv41-tp4-pipe-min-kib")).or(kib(crate::opt!("dsv41-tp-pipe-min-kib"))).is_none() {
            pipe_min = single_cap;
        }
        if dual.as_ref().map_or(false, |d| d.pctx.is_some()) && world == 4 && pair_n && kib(crate::opt!("dsv41-tp4-rsag-min-kib")).is_none() {
            rsag_min = RSAG_MIN_BYTES_DUAL;
        }
        if dual.as_ref().map_or(false, |d| d.pctx.is_some()) && world == 8 && pair_n && kib(crate::opt!("dsv41-tp8-rsag-min-kib")).is_none() {
            rsag_min = RSAG8_MIN_BYTES_DUAL;
        }
        crate::reprintln!("[{log_tag} tx1] rank {rank}/{world}: {dual_line}");
        // leg P7SP + Phase 1D + Phase 2: power-of-two worlds use pipelined multi-block Pipe across 1..2 rails;
        // non-power-of-two worlds (TP3 / TP6) use chunked dual-rail 1-RTT direct RS / AG / All-to-All on `single` + `dual`.
        let mut pipe_rails = vec![cptr];
        if let Some(d) = &dual {
            if world == 2 { pipe_rails.push(d.ctx); }
            else if let Some(p2) = d.pctx { pipe_rails.push(p2); }
        }
        let n_pipe_rails = pipe_rails.len();
        let pipe_lookahead = if world == 8 { 3 } else { 4 };
        let pipe = if world.is_power_of_two() {
            Some((kern, crate::tp_xport::Pipe::new(dev, pipe_rails, world, pipe_lookahead, 16).context("pipelined transport")?))
        } else {
            None
        };
        // legs PX1 / PX2: with the dsv41 pair every frame up to one ring slot (world 2) / the single-stage bound (world >= 3)
        // takes ONE exchange; wider frames take Pipe (pow2) or 2-RTT direct RS+AG (TP3 / TP6).
        dev.synchronize()?;
        let describe = format!("dsv41 TP rank {rank}/{world}: fp32 sum all-reduce over the RDMA doorbell link (tp_xport). Frames <= {} KiB: {}; \
                                wider frames: {}. Association {}; proxy on core {TP_PROXY_CORE}{}. {why}",
                               pipe_min / 1024,
                               match (&single, pair) { (Some(s), _) => format!("ONE all-peers stage on a one-shot ctx (receive {}, wire {}, payload bound {} KiB){}",
                                                                               if s.gpu { "GPU-validated" } else { "CPU bounce" },
                                                                               if wire_bf16 { "BF16" } else { "FP32" },
                                                                               single_cap / 1024,
                                                                               if pair_n { format!(", on the dsv41 PDL pair (--dsv41-xchg legacy = the shared pair){}", if rsag_min > 0 { format!("; frames from {} KiB: reduce-scatter (primary tree, {} epochs) + all-gather (one all-peers epoch), same sums", rsag_min / 1024, world.trailing_zeros()) } else { String::new() }) } else { String::new() }),
                                               (None, true) => format!("the dsv41 PDL exchange pair (receive {}, wire {}; --dsv41-xchg legacy = the shared pair)", if gpu_recv { "GPU-validated" } else { "CPU bounce" }, if wire_bf16 { "BF16" } else { "FP32" }),
                                               (None, false) => "the K1 / K2 pair (multi-block K1)".into() },
                               if pipe.is_some() { format!("the pipelined multi-block transport ({n_pipe_rails} rail(s), lookahead {pipe_lookahead}, 16 blocks)") }
                               else { format!("2-RTT direct RS+AG ({} rail(s))", if dual.is_some() { 2 } else { 1 }) },
                               match world {
                                   2 => "p0 + p1",
                                   4 => "(p0 + p2) + (p1 + p3)",
                                   8 => "((p0 + p2) + (p4 + p6)) + ((p1 + p3) + (p5 + p7))",
                                   3 => "(p0 + p1) + p2",
                                   6 => "((((p0 + p1) + p2) + p3) + p4) + p5",
                                   _ => "canonical 0..W",
                               },
                               if single.is_some() { format!(", single-stage proxy on core {TP_SINGLE_PROXY_CORE}") } else { String::new() });
        let describe = format!("{describe} | {dual_line}{}", dual.as_ref().map_or(String::new(), |d| format!(" (rail-2 device {})", d.dev)));
        let links = [addr, single_addr, dual_addr, dual_p2_addr].into_iter().filter(|a| *a != 0).collect();
        let out = RdmaXport { rank, world, ctx: cptr, k1, k2, k1m: (k1mf, arrive), single, dual, pipe, pipe_min, single_cap, rsag_min, links, describe, xchg };
        VF_WORLD.store(world as u64, std::sync::atomic::Ordering::Relaxed);
        VF_DUAL.store(u64::from(out.dual_on()), std::sync::atomic::Ordering::Relaxed);
        VF_GPU_RECV.store(u64::from(out.xchg.as_ref().is_some_and(|x| x.gpu_recv)), std::sync::atomic::Ordering::Relaxed);
        VF_SP_READY.store(u64::from(out.sp_ready()), std::sync::atomic::Ordering::Relaxed);
        Ok(out)
    }

    /// The path a frame of `bytes` takes on this transport.
    pub fn path(&self, bytes: usize) -> Path { route_cap(bytes, self.world, self.single.is_some(), self.pipe.is_some(), self.pipe_min, self.single_cap) }

    /// True when the rail-2 twin ctx is active on this transport.
    pub fn dual_on(&self) -> bool { self.dual.is_some() }

    /// True when sequence-parallel prefill collectives (`sp_reduce_scatter` / `sp_all_gather_multi`) are supported.
    pub fn sp_ready(&self) -> bool {
        if matches!(self.world, 2 | 4 | 8) {
            self.pipe.as_ref().map_or(false, |(_, x)| x.rounds == self.world.trailing_zeros() as usize)
        } else {
            matches!(self.world, 3 | 6) && self.xchg.as_ref().is_some_and(|x| x.pair) && self.single.is_some()
        }
    }

    /// True when 1-RTT All-to-All (DeepSpeed-Ulysses sequence-parallel attention, arXiv:2309.14509) is ready.
    pub fn a2a_ready(&self) -> bool {
        self.xchg.as_ref().is_some_and(|x| x.pair) && (self.world == 2 || self.single.is_some())
    }

    /// 1-RTT chunked All-to-All across all ranks (`world in {2, 3, 4, 6, 8}`, dual-rail when active).
    /// `src` holds `world` contiguous slices of `slice_bytes` (16 B multiple); `dst` receives one slice per peer.
    pub fn all_to_all(&self, stream: &CudaStream, src: u64, dst: u64, slice_bytes: usize) -> Result<()> {
        let x = self.xchg.as_ref().filter(|x| x.pair).context("velo-fabric all_to_all requires PDL xchg pair")?;
        let (ctx1, slot_cap) = if self.world == 2 {
            (self.ctx, crate::tp::TP_SLOT_BYTES)
        } else {
            (self.single.as_ref().context("velo-fabric all_to_all requires single-stage ctx")?.ctx, self.single_cap)
        };
        let (ctx2, dual_min) = self.dual.as_ref().map_or((None, usize::MAX), |d| (Some(d.ctx), d.min_bytes));
        x.all_to_all(stream.stream as u64, ctx1, ctx2, src, dst, slice_bytes, slot_cap, dual_min)
    }

    /// Leg P7SP + Phase A + Arbitrary-W: row-block in-place FP32 reduce-scatter across `blocks` (`blocks.len() == self.world`).
    /// After completion, `base + blocks[self.rank].0 * row_len * 4` holds the canonical FP32 sum across all ranks.
    pub fn sp_reduce_scatter(&self, stream: &CudaStream, base: u64, row_len: usize, blocks: &[(usize, usize)]) -> Result<usize> {
        ensure!(blocks.len() == self.world, "dsv41 SP reduce-scatter: {} blocks for world {}", blocks.len(), self.world);
        if !self.world.is_power_of_two() {
            let x = self.xchg.as_ref().filter(|x| x.pair).context("velo-fabric direct SP RS requires PDL xchg pair")?;
            let s = self.single.as_ref().context("velo-fabric direct SP RS requires single-stage ctx")?;
            let (ctx2, dual_min) = self.dual.as_ref().map_or((None, usize::MAX), |d| (Some(d.ctx), d.min_bytes));
            return x.direct_rs_rows(stream.stream as u64, s.ctx, ctx2, base, row_len, blocks, self.single_cap, dual_min);
        }
        let (kern, x) = self.pipe.as_ref().context("dsv41 SP reduce-scatter without pipelined transport")?;
        let io = crate::tp_xport::Io::f32_inplace(base);
        if self.world == 8 {
            let b8: crate::tp_xport::Blocks8 = std::array::from_fn(|i| blocks[i]);
            crate::tp_xport::reduce_scatter_rows_w8(kern, stream, x, io, row_len, &b8, self.rank, None)
                .context("dsv41 world-8 SP reduce-scatter")?;
            let a_rows: usize = (0..8).filter(|&i| (i & 2) != (self.rank & 2)).map(|i| b8[i].1 - b8[i].0).sum();
            let b_rows: usize = (0..2).map(|k| { let i = ((self.rank ^ 4) & 6) | k; b8[i].1 - b8[i].0 }).sum();
            let c_rows = b8[self.rank ^ 1].1 - b8[self.rank ^ 1].0;
            Ok((a_rows + b_rows + c_rows) * row_len * 4)
        } else if self.world == 4 {
            let b4: crate::tp_xport::Blocks4 = [blocks[0], blocks[1], blocks[2], blocks[3]];
            crate::tp_xport::reduce_scatter_rows_w4(kern, stream, x, io, row_len, &b4, self.rank, None)
                .context("dsv41 world-4 SP reduce-scatter")?;
            let h2 = (self.rank ^ 2) >> 1;
            let a_rows = (b4[2 * h2].1 - b4[2 * h2].0) + (b4[2 * h2 + 1].1 - b4[2 * h2 + 1].0);
            let b_rows = b4[self.rank ^ 1].1 - b4[self.rank ^ 1].0;
            Ok((a_rows + b_rows) * row_len * 4)
        } else {
            ensure!(self.world == 2, "dsv41 SP reduce-scatter: world {}", self.world);
            let (own, peer) = (blocks[self.rank], blocks[1 - self.rank]);
            crate::tp_xport::reduce_scatter_rows(kern, stream, x, io, row_len, own, peer, None)
                .context("dsv41 world-2 SP reduce-scatter")?;
            Ok((peer.1 - peer.0) * row_len * 4)
        }
    }

    /// Leg P7SP + Phase A + Arbitrary-W: row-block byte-exact all-gather of one or more row buffers (`(base, row_bytes)`) across `blocks`.
    pub fn sp_all_gather_multi(&self, stream: &CudaStream, bufs: &[(u64, usize)], blocks: &[(usize, usize)]) -> Result<usize> {
        ensure!(blocks.len() == self.world, "dsv41 SP all-gather: {} blocks for world {}", blocks.len(), self.world);
        if !self.world.is_power_of_two() {
            let x = self.xchg.as_ref().filter(|x| x.pair).context("velo-fabric direct SP AG requires PDL xchg pair")?;
            let s = self.single.as_ref().context("velo-fabric direct SP AG requires single-stage ctx")?;
            let (ctx2, dual_min) = self.dual.as_ref().map_or((None, usize::MAX), |d| (Some(d.ctx), d.min_bytes));
            return x.direct_ag_rows_multi(stream.stream as u64, s.ctx, ctx2, bufs, blocks, self.single_cap, dual_min);
        }
        let (kern, x) = self.pipe.as_ref().context("dsv41 SP all-gather without pipelined transport")?;
        let rb_sum: usize = bufs.iter().map(|b| b.1).sum();
        if self.world == 8 {
            let b8: crate::tp_xport::Blocks8 = std::array::from_fn(|i| blocks[i]);
            crate::tp_xport::all_gather_rows_multi_w8(kern, stream, x, bufs, &b8, self.rank)
                .context("dsv41 world-8 SP all-gather")?;
            let r_own = b8[self.rank].1 - b8[self.rank].0;
            let r2 = b8[self.rank ^ 2].1 - b8[self.rank ^ 2].0;
            let r4 = b8[self.rank ^ 4].1 - b8[self.rank ^ 4].0;
            let r6 = b8[self.rank ^ 6].1 - b8[self.rank ^ 6].0;
            Ok((3 * r_own + 2 * r2 + r4 + r6) * rb_sum)
        } else if self.world == 4 {
            let b4: crate::tp_xport::Blocks4 = [blocks[0], blocks[1], blocks[2], blocks[3]];
            crate::tp_xport::all_gather_rows_multi_w4(kern, stream, x, bufs, &b4, self.rank)
                .context("dsv41 world-4 SP all-gather")?;
            let own_rows = b4[self.rank].1 - b4[self.rank].0;
            let r2_rows = b4[self.rank ^ 2].1 - b4[self.rank ^ 2].0;
            Ok((2 * own_rows + r2_rows) * rb_sum)
        } else {
            ensure!(self.world == 2, "dsv41 SP all-gather: world {}", self.world);
            let (own, peer) = (blocks[self.rank], blocks[1 - self.rank]);
            crate::tp_xport::all_gather_rows_multi(kern, stream, x, bufs, own, peer)
                .context("dsv41 world-2 SP all-gather")?;
            Ok((own.1 - own.0) * rb_sum)
        }
    }

    /// The transport's abort word (0 = healthy).
    pub fn abort_code(&self) -> u64 {
        self.links.iter().map(|a| crate::net::ctx_abort_status(*a)).find(|c| *c != 0).unwrap_or(0)
    }

    /// In-place exact fp32 sum all-reduce of `n` floats at device address `base` (NEVER BF16-compressed).
    pub fn all_reduce(&self, stream: &CudaStream, base: u64, n: usize) -> Result<()> {
        self.all_reduce_impl(stream, base, n, false)
    }

    /// In-place neural activation/residual partial sum all-reduce (uses BF16 wire when `--dsv41-wire-bf16 on`).
    pub fn all_reduce_act(&self, stream: &CudaStream, base: u64, n: usize) -> Result<()> {
        self.all_reduce_impl(stream, base, n, true)
    }

    /// In-place exact fp32 sum all-reduce on an explicit raw `CUstream` (for callers like Inkling-2 whose
    /// CUDA graph capture stream is a raw `sys::CUstream`). Wide `Pipe` / non-4-aligned launches use
    /// `fallback_stream` (a blocking stream ordered against the caller's NULL stream; never hit during decode-width
    /// CUDA graph capture because `single_cap` = 768 KiB). When `wide_row_len` is `Some(rl)` dividing `n`, wide
    /// world-4 / world-8 frames (`> 4 MiB` / past `rsag_dual`) run as 2-rail pipelined reduce-scatter + all-gather
    /// (`0.75F`/rail at world 4, `0.875F`/rail at world 8 — bit-exact to `reduce_pipe`).
    pub fn all_reduce_raw(
        &self,
        raw_stream: cudarc::driver::sys::CUstream,
        fallback_stream: &CudaStream,
        base: u64,
        n: usize,
        wide_row_len: Option<usize>,
    ) -> Result<()> {
        let capturing = !raw_stream.is_null();
        let st_u64 = if capturing { raw_stream as u64 } else { fallback_stream.stream as u64 };
        // Arbitrary non-power-of-two world (TP3 / TP6): 1-stage PDL up to single_cap (handles any positive float
        // count including e.g. 6 floats = 24 B for TP3 rows=1 sampler reduce, even inside CUDA graph capture),
        // and 2-RTT direct RS+AG above single_cap.
        if !self.world.is_power_of_two() {
            let x = self.xchg.as_ref().filter(|x| x.pair).context("velo-fabric TP3/TP6 requires PDL xchg pair")?;
            let s = self.single.as_ref().context("velo-fabric TP3/TP6 requires single-stage ctx")?;
            if n * 4 <= self.single_cap {
                if n % 4 == 0 {
                    if let Some(d) = self.dual.as_ref().filter(|d| n * 4 >= d.min_bytes) {
                        return x.exchange_dual(st_u64, s.ctx, d.ctx, base, n)
                            .with_context(|| format!("velo-fabric world-{} dual-rail pair", x.world));
                    }
                }
                return x.exchange(st_u64, s.ctx, base, n)
                    .with_context(|| format!("velo-fabric world-{} pair", x.world));
            }
            ensure!(n % 4 == 0, "velo-fabric: wide world-{} all-reduce ({n} floats) must be 16 B aligned", self.world);
            ensure!(!capturing, "velo-fabric: wide world-{} all-reduce ({} B > {} B) cannot run inside CUDA graph capture", self.world, n * 4, self.single_cap);
            let (ctx2, dual_min) = self.dual.as_ref().map_or((None, usize::MAX), |d| (Some(d.ctx), d.min_bytes));
            return x.direct_wide_allreduce(st_u64, s.ctx, ctx2, base, n, self.single_cap, dual_min);
        }
        if n % 4 == 0 {
            // World-4 / world-8 PDL dual-rail / single-rail RS+AG (graph-capturable on `st_u64`, up to ~4 MiB)
            if let (Some(x), Some(s)) = (self.xchg.as_ref().filter(|x| x.pair), self.single.as_ref()) {
                if x.world == 4 {
                    if let Some(d) = self.dual.as_ref().filter(|d| d.pctx.is_some() && n * 4 >= d.min_bytes) {
                        if use_rsag_dual(n, self.rsag_min, self.single_cap) {
                            return x.rsag_dual(st_u64, self.ctx, d.pctx.unwrap(), s.ctx, d.ctx, base, n)
                                .context("velo-fabric world-4 dual-rail RS+AG");
                        }
                    }
                    if use_rsag(n, self.rsag_min, self.single_cap) {
                        return x.rsag(st_u64, self.ctx, s.ctx, base, n)
                            .context("velo-fabric world-4 RS+AG");
                    }
                } else if x.world == 8 {
                    if let Some(d) = self.dual.as_ref().filter(|d| d.pctx.is_some() && n * 4 >= d.min_bytes) {
                        if use_rsag8_dual(n, self.rsag_min, self.single_cap) {
                            return x.rsag8_dual(st_u64, self.ctx, d.pctx.unwrap(), s.ctx, d.ctx, base, n)
                                .context("velo-fabric world-8 dual-rail RS+AG");
                        }
                    }
                    if use_rsag8(n, self.rsag_min, self.single_cap) {
                        return x.rsag8(st_u64, self.ctx, s.ctx, base, n)
                            .context("velo-fabric world-8 RS+AG");
                    }
                }
            }
            let path = if capturing && n * 4 <= self.single_cap && self.single.is_some() {
                Path::Single
            } else {
                self.path(n * 4)
            };
            match path {
                Path::Pipe => {
                    ensure!(!capturing, "velo-fabric: wide all-reduce ({} B > {} B) cannot run inside CUDA graph capture", n * 4, self.pipe_min);
                    if let Some(rl) = wide_row_len {
                        if matches!(self.world, 4 | 8) && self.sp_ready() && rl % 8 == 0 && n % rl == 0 {
                            let rows = n / rl;
                            if rows >= self.world * 8 {
                                let blocks: Vec<(usize, usize)> = (0..self.world)
                                    .map(|r| crate::tp_xport::row_block(rows, r, self.world))
                                    .collect();
                                self.sp_reduce_scatter(fallback_stream, base, rl, &blocks)
                                    .context("velo-fabric wide prefill RS")?;
                                self.sp_all_gather_multi(fallback_stream, &[(base, rl * 4)], &blocks)
                                    .context("velo-fabric wide prefill AG")?;
                                return Ok(());
                            }
                        }
                    }
                    let (kern, x) = self.pipe.as_ref().unwrap();
                    return crate::tp_xport::reduce_pipe(kern, fallback_stream, x, crate::tp_xport::Io::f32_inplace(base), n)
                        .context("velo-fabric pipelined all-reduce");
                }
                Path::Single => {
                    let s = self.single.as_ref().unwrap();
                    if let Some(x) = self.xchg.as_ref().filter(|x| x.pair && matches!(x.world, 4 | 8)) {
                        if let Some(d) = self.dual.as_ref().filter(|d| n * 4 >= d.min_bytes) {
                            return x.exchange_dual(st_u64, s.ctx, d.ctx, base, n)
                                .with_context(|| format!("velo-fabric world-{} dual-rail pair", x.world));
                        }
                        return x.exchange(st_u64, s.ctx, base, n)
                            .with_context(|| format!("velo-fabric world-{} pair", x.world));
                    }
                    ensure!(!capturing, "velo-fabric: legacy single-stage fallback requires PDL pair for raw stream capture");
                    let c1 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
                    let cm = LaunchConfig { grid_dim: (k1_mb_blocks(n), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    unsafe {
                        s.k1m.0.clone().launch_on_stream(fallback_stream, cm, (s.ctx, base, (n * 4) as u32, &s.k1m.1)).context("single-stage K1m")?;
                        s.k2.clone().launch_on_stream(fallback_stream, c1, (s.ctx, base, base, n as i32)).context("single-stage K2")?;
                    }
                    return Ok(());
                }
                Path::Serial => {
                    let rounds = self.world.trailing_zeros() as usize;
                    for sp in crate::tp_xport::serial_plan(n, rounds) {
                        ensure!(sp.len * 4 + 64 <= crate::tp::TP_SLOT_BYTES, "all-reduce epoch {} floats exceeds one ring slot", sp.len);
                        let q = base + (sp.off * 4) as u64;
                        if let (Some(x), true) = (self.xchg.as_ref().filter(|x| x.pair && x.world == 2), sp.len % 4 == 0) {
                            if let Some(d) = self.dual.as_ref().filter(|d| sp.len * 4 >= d.min_bytes) {
                                x.exchange_dual(st_u64, self.ctx, d.ctx, q, sp.len)?;
                                continue;
                            }
                            x.exchange(st_u64, self.ctx, q, sp.len)?;
                            continue;
                        }
                        ensure!(!capturing, "velo-fabric: legacy serial fallback requires PDL pair for raw stream capture");
                        let c1 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
                        unsafe {
                            if sp.len % 4 == 0 {
                                let cm = LaunchConfig { grid_dim: (k1_mb_blocks(sp.len), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                                self.k1m.0.clone().launch_on_stream(fallback_stream, cm, (self.ctx, q, (sp.len * 4) as u32, &self.k1m.1, 0i32)).context("K1m")?;
                            } else {
                                self.k1.clone().launch_on_stream(fallback_stream, c1, (self.ctx, q, (sp.len * 4) as u32)).context("K1")?;
                            }
                            self.k2.clone().launch_on_stream(fallback_stream, c1, (self.ctx, q, q, sp.len as i32, 2i32)).context("K2")?;
                        }
                    }
                    return Ok(());
                }
            }
        }
        // Defensive fallback for n % 4 != 0 (e.g. odd float count)
        ensure!(!capturing, "velo-fabric: unaligned all-reduce ({n} floats) cannot run inside CUDA graph capture");
        let rounds = self.world.trailing_zeros() as usize;
        let c1 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        for sp in crate::tp_xport::serial_plan(n, rounds) {
            ensure!(sp.len * 4 + 64 <= crate::tp::TP_SLOT_BYTES, "all-reduce epoch {} floats exceeds one ring slot", sp.len);
            let q = base + (sp.off * 4) as u64;
            unsafe {
                self.k1.clone().launch_on_stream(fallback_stream, c1, (self.ctx, q, (sp.len * 4) as u32)).context("K1")?;
                self.k2.clone().launch_on_stream(fallback_stream, c1, (self.ctx, q, q, sp.len as i32, 2i32)).context("K2")?;
            }
        }
        Ok(())
    }

    fn all_reduce_impl(&self, stream: &CudaStream, base: u64, n: usize, allow_bf16: bool) -> Result<()> {
        let use_bf16 = allow_bf16 && self.xchg.as_ref().map_or(false, |x| x.pair && x.wire_bf16);
        // leg PX2 + Phase 1D + Phase A: wide world-4 / world-8 frames as reduce-scatter + all-gather.
        // At world 4, opt-in BF16 1-stage sends 1.5F in 1 RTT (same bytes as FP32 rsag in 3 RTTs), so skip rsag.
        // At world 8, FP32 rsag8 sends 1.75F vs 3.5F for 1-stage BF16 (2x less wire), so keep rsag8 even with wire_bf16 on.
        if let (Some(x), Some(s)) = (self.xchg.as_ref().filter(|x| x.pair && !(use_bf16 && x.world == 4)), self.single.as_ref()) {
            if x.world == 4 {
                if let Some(d) = self.dual.as_ref().filter(|d| d.pctx.is_some() && n * 4 >= d.min_bytes) {
                    if use_rsag_dual(n, self.rsag_min, self.single_cap) {
                        return x.rsag_dual(stream.stream as u64, self.ctx, d.pctx.unwrap(), s.ctx, d.ctx, base, n)
                            .context("dsv41 world-4 dual-rail reduce-scatter + all-gather");
                    }
                }
                if use_rsag(n, self.rsag_min, self.single_cap) {
                    return x.rsag(stream.stream as u64, self.ctx, s.ctx, base, n).context("dsv41 world-4 reduce-scatter + all-gather");
                }
            } else if x.world == 8 {
                if let Some(d) = self.dual.as_ref().filter(|d| d.pctx.is_some() && n * 4 >= d.min_bytes) {
                    if use_rsag8_dual(n, self.rsag_min, self.single_cap) {
                        return x.rsag8_dual(stream.stream as u64, self.ctx, d.pctx.unwrap(), s.ctx, d.ctx, base, n)
                            .context("dsv41 world-8 dual-rail reduce-scatter + all-gather");
                    }
                }
                if use_rsag8(n, self.rsag_min, self.single_cap) {
                    return x.rsag8(stream.stream as u64, self.ctx, s.ctx, base, n).context("dsv41 world-8 reduce-scatter + all-gather");
                }
            }
        }
        if !self.world.is_power_of_two() {
            let x = self.xchg.as_ref().filter(|x| x.pair).context("velo-fabric TP3/TP6 requires PDL xchg pair")?;
            let s = self.single.as_ref().context("velo-fabric TP3/TP6 requires single-stage ctx")?;
            if n * 4 <= self.single_cap {
                if n % 4 == 0 {
                    if let Some(d) = self.dual.as_ref().filter(|d| n * 4 >= d.min_bytes) {
                        return if use_bf16 { x.exchange_dual_act(stream.stream as u64, s.ctx, d.ctx, base, n) }
                               else { x.exchange_dual(stream.stream as u64, s.ctx, d.ctx, base, n) }
                            .with_context(|| format!("velo-fabric world-{} dual-rail pair", x.world));
                    }
                    if use_bf16 {
                        return x.exchange_act(stream.stream as u64, s.ctx, base, n)
                            .with_context(|| format!("velo-fabric world-{} bf16 pair", x.world));
                    }
                }
                return x.exchange(stream.stream as u64, s.ctx, base, n)
                    .with_context(|| format!("velo-fabric world-{} pair", x.world));
            }
            ensure!(n % 4 == 0, "velo-fabric TP all-reduce: {n} floats is not a 16 B multiple");
            let (ctx2, dual_min) = self.dual.as_ref().map_or((None, usize::MAX), |d| (Some(d.ctx), d.min_bytes));
            return x.direct_wide_allreduce(stream.stream as u64, s.ctx, ctx2, base, n, self.single_cap, dual_min);
        }
        ensure!(n % 4 == 0, "dsv41 TP all-reduce: {n} floats is not a 16 B multiple (frames are padded to it)");
        let c1 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        match self.path(n * 4) {
            Path::Pipe => {
                let (kern, x) = self.pipe.as_ref().unwrap();
                crate::tp_xport::reduce_pipe(kern, stream, x, crate::tp_xport::Io::f32_inplace(base), n).context("pipelined all-reduce")
            }
            Path::Single => {
                let s = self.single.as_ref().unwrap();
                if let Some(x) = self.xchg.as_ref().filter(|x| x.pair && matches!(x.world, 4 | 8)) {
                    if let Some(d) = self.dual.as_ref().filter(|d| n * 4 >= d.min_bytes) {
                        return if use_bf16 { x.exchange_dual_act(stream.stream as u64, s.ctx, d.ctx, base, n) }
                               else { x.exchange_dual(stream.stream as u64, s.ctx, d.ctx, base, n) }
                            .with_context(|| format!("dsv41 world-{} dual-rail pair", x.world));
                    }
                    return if use_bf16 { x.exchange_act(stream.stream as u64, s.ctx, base, n) }
                           else { x.exchange(stream.stream as u64, s.ctx, base, n) }
                        .with_context(|| format!("dsv41 world-{} pair", x.world));
                }
                let cm = LaunchConfig { grid_dim: (k1_mb_blocks(n), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                unsafe {
                    s.k1m.0.clone().launch_on_stream(stream, cm, (s.ctx, base, (n * 4) as u32, &s.k1m.1)).context("single-stage K1m")?;
                    s.k2.clone().launch_on_stream(stream, c1, (s.ctx, base, base, n as i32)).context("single-stage K2")?;
                }
                Ok(())
            }
            Path::Serial => {
                let rounds = self.world.trailing_zeros() as usize;
                for sp in crate::tp_xport::serial_plan(n, rounds) {
                    ensure!(sp.len * 4 + 64 <= crate::tp::TP_SLOT_BYTES, "all-reduce epoch {} floats exceeds one ring slot", sp.len);
                    let q = base + (sp.off * 4) as u64;
                    if let (Some(x), true) = (self.xchg.as_ref().filter(|x| x.pair && x.world == 2), sp.len % 4 == 0) {
                        if let Some(d) = self.dual.as_ref().filter(|d| sp.len * 4 >= d.min_bytes) {
                            if use_bf16 { x.exchange_dual_act(stream.stream as u64, self.ctx, d.ctx, q, sp.len)?; }
                            else { x.exchange_dual(stream.stream as u64, self.ctx, d.ctx, q, sp.len)?; }
                            continue;
                        }
                        if use_bf16 { x.exchange_act(stream.stream as u64, self.ctx, q, sp.len)?; }
                        else { x.exchange(stream.stream as u64, self.ctx, q, sp.len)?; }
                        continue;
                    }
                    unsafe {
                        if sp.len % 4 == 0 {
                            let cm = LaunchConfig { grid_dim: (k1_mb_blocks(sp.len), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                            self.k1m.0.clone().launch_on_stream(stream, cm, (self.ctx, q, (sp.len * 4) as u32, &self.k1m.1, 0i32)).context("K1m")?;
                        } else {
                            self.k1.clone().launch_on_stream(stream, c1, (self.ctx, q, (sp.len * 4) as u32)).context("K1")?;
                        }
                        self.k2.clone().launch_on_stream(stream, c1, (self.ctx, q, q, sp.len as i32, 2i32)).context("K2")?;
                    }
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tx1_tests {
    use super::*;
    #[test]
    fn dual_mode_parses_and_defaults_to_auto() {
        assert_eq!(dual_mode_of(None), DualMode::Auto);
        assert_eq!(dual_mode_of(Some("on")), DualMode::On);
        assert_eq!(dual_mode_of(Some(" OFF ")), DualMode::Off);
        assert_eq!(dual_mode_of(Some("auto")), DualMode::Auto);
    }
    #[test]
    fn a_one_row_layer_frame_splits_and_the_small_head_exchanges_do_not_by_default() {
        let row = 5152usize * 4;   // 20,608 B
        assert!(row >= DUAL_MIN_BYTES);
        assert!(6144 < DUAL_MIN_BYTES && 2560 < DUAL_MIN_BYTES);   // the head / candidate exchanges stay on rail 1
    }
    #[test]
    fn world8_routing_ports_and_canonical_tree_equivalence() {
        assert!(single_resolve(None, 8).0);
        assert_eq!(route_cap(20_608, 8, true, true, PIPE_MIN_BYTES_XCHG4, SINGLE_CAP_BYTES_XCHG4), Path::Single);
        assert_eq!(route_cap(1024 * 1024, 8, true, true, PIPE_MIN_BYTES_XCHG4, SINGLE_CAP_BYTES_XCHG4), Path::Pipe);
        // Port blocks at world 8 must be mutually disjoint across all 4 N-way ctxs (primary r1/r2 + single r1/r2)
        let b1 = crate::tp::TP_PORT;
        let b2 = crate::exl3_forward::xtp::rail2_nway_base(8).unwrap();
        let b3 = crate::exl3_forward::xtp::single_nway_base(8).unwrap();
        let b4 = crate::exl3_forward::xtp::dual_single_nway_base(8).unwrap();
        assert_eq!((b2 - b1, b3 - b1, b4 - b1), (70, 140, 210));
        // World-8 xc8 canonical sum ((p0+p2)+(p4+p6)) + ((p1+p3)+(p5+p7)) == aligned 3-round RD tree bit-for-bit!
        let parts: Vec<Vec<f32>> = (0..8)
            .map(|r| (0..64).map(|i| (((r * 97 + i * 31) % 251) as f32 - 125.5) * 0.0137).collect())
            .collect();
        let rd = crate::tp_xport::rd_tree_reference(8, 0, &parts).unwrap();
        for i in 0..64 {
            let xc8 = ((parts[0][i] + parts[2][i]) + (parts[4][i] + parts[6][i]))
                    + ((parts[1][i] + parts[3][i]) + (parts[5][i] + parts[7][i]));
            assert_eq!(xc8.to_bits(), rd[i].to_bits(), "bit mismatch at elem {i}");
        }
    }
}
