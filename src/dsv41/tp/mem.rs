//! DSV4.1 TP (leg MEM1): the memory of a serving rank.
//!
//! On a GB10 the device allocations come out of the same unified memory as the host, and the memory watchdog
//! (`--mem-watchdog-gb`, default 5 GB of MemAvailable) exits the process before the box runs dry. At TP2 the head
//! holds ~99 GiB of weights, so everything else this rank keeps decides whether a long session survives. This file:
//!
//! * the formulas of what the forward allocates (`lane_bytes`, `arena_bytes`, `fixed_bytes`, `snap_bytes`, the MoE
//!   scratch estimate): `TpModel::new` and the snapshot use them, and the memory-fit check of the boot
//!   (`load::fit`) predicts with the same ones;
//! * the SCRATCH need: every linear / MoE call checks its scratch against the backend's own bound (`glue::linear`,
//!   `moe_ex`), so the scratch is the largest bound over every weight the forward calls, at the row classes it calls
//!   them at (decode / verify <= 32 rows, prefill up to the chunk), instead of a fixed 1 GiB;
//! * the LEDGER: what this rank holds by part and the box's MemAvailable at each boot phase (one boot line, one
//!   breakdown line at `--log-level debug`, `velogb10_dsv41_mem_*` on `/metrics`);
//! * TRIMS: freed device memory back to the box (`cuMemPoolTrimTo` on the default pool after a sync), freed host
//!   memory too (`malloc_trim`);
//! * the PRESSURE GOVERNOR of the head's lane loop: the prefix pool may hold its states plus what is free above
//!   `floor + reserve`; below that line it gives states back (least recently used first; every rank frees them with
//!   the next op) and the trims run. `reserve` = 1.5 GiB (the box's other processes, the transients of a call) plus one
//!   prefix state of the longest context (the snapshot is copied before the pool decides whether it keeps it).

use super::fwd::{TpLayer, TpModel, HEAD_MAX};
use crate::dsv41::config::Dsv41Config;
use crate::dsv41::weights::{BackendKind, DenseWeight};
use anyhow::Result;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub const GIB: u64 = 1 << 30;
pub const MIB: u64 = 1 << 20;

/// Captured decode / verify / draft-step graphs one rank keeps at most (the least recently used one is destroyed
/// above it): with 4 lanes the keys are (active lanes, draft length, context class), more than a session needs at once.
pub const GRAPH_CAP: usize = 64;
/// A captured graph's device bytes before the boot measures one (the fit's estimate).
pub const GRAPH_EST: u64 = 4 * MIB;
/// The CUDA context, the loaded modules and the process's own host memory after the trims (tokenizer, census, the
/// lane loop): the fit's allowance (the boot ledger measures the real figure: `velogb10_dsv41_mem_bytes{part="process"}`).
pub const PROCESS_EST: u64 = 1280 * MIB;
/// The DSpark round's buffers and the penalty counts (the fit's estimate; the ledger has the real figure).
pub const SPEC_EST: u64 = 48 * MIB;
/// What the governor keeps free above the memwatch floor besides one prefix state.
pub const RESERVE: u64 = 3 * GIB / 2;
/// The Engram row cache's smallest size the fit caps it to.
pub const CACHE_FLOOR_MIB: u64 = 512;

fn b256(n: usize) -> u64 { n.max(256) as u64 }
fn a256(n: usize) -> usize { (n + 255) / 256 * 256 }

/// The two activation arenas' size each (layer l's outputs stay valid while layer l + 1 runs): 512 KiB per row of the
/// largest call, at least 256 MiB.
pub fn arena_bytes(max_rows: usize) -> usize { (256usize << 20).max(max_rows << 19) }

/// Device bytes of one TP lane (`TpModel::new_lane`, buffer for buffer): per layer the SWA ring and the call's linear
/// rows, per KV source the caches for `max_seq` positions, the frontier ring and the latent rows.
pub fn lane_bytes(cfg: &Dsv41Config, max_rows: usize, max_seq: usize) -> u64 {
    let hd = cfg.head_dim;
    let (slots, lin) = (cfg.swa_slots(), max_rows + cfg.window);
    let win = b256(slots * hd) + b256(slots * hd / 32) + b256(lin * hd) + b256(lin * hd / 32);
    let mut t = win * cfg.layers as u64;
    for &s in &cfg.kv_sources {
        let ratio = cfg.ratio(s);
        let cap = max_seq / ratio.max(1) as usize + 2;
        t += b256(cap * 288) + b256(cap * 64) + b256(cap * 4) + b256((max_rows + 1) * hd * 2);
        if ratio >= 2 { t += 2 * b256(cfg.frontier_rows(ratio) * hd * 4); }
    }
    t
}

/// The pieces of one prefix state of a lane at `pos` positions, in the order `TpModel::snapshot` copies them, as
/// (offset, bytes) inside its one allocation: per layer the ring (q, s), per KV source (ascending) its main KV,
/// index-K q / s planes up to its groups and (ratio >= 2) its two frontier planes.
pub fn snap_layout(cfg: &Dsv41Config, pos: usize) -> (Vec<(usize, usize)>, usize) {
    let hd = cfg.head_dim;
    let slots = cfg.swa_slots();
    let mut v = Vec::new();
    let mut off = 0usize;
    let mut put = |n: usize, v: &mut Vec<(usize, usize)>| { v.push((off, n)); off += a256(n); };
    for _ in 0..cfg.layers { put(slots * hd, &mut v); put(slots * hd / 32, &mut v); }
    let mut srcs = cfg.kv_sources.clone();
    srcs.sort_unstable();
    for s in srcs {
        let ratio = cfg.ratio(s);
        let n = pos / ratio.max(1) as usize;
        put(n * 288, &mut v); put(n * 64, &mut v); put(n * 4, &mut v);
        if ratio >= 2 { let rr = cfg.frontier_rows(ratio) * hd * 4; put(rr, &mut v); put(rr, &mut v); }
    }
    (v, off.max(256))
}

/// Device bytes of one prefix state at `pos` positions (`snap_layout`'s total).
pub fn snap_bytes(cfg: &Dsv41Config, pos: usize) -> u64 { snap_layout(cfg, pos).1 as u64 }

/// The forward's fixed buffers besides the arenas, the scratch and the lanes (`TpModel::finish_new`): the two frames
/// (hidden + mix rider + Engram rider per row), the embedding broadcast, the Engram output, the per-index-source
/// selections, the candidate union, the attention decode scratch, the head rows, positions and ids.
pub fn fixed_bytes(cfg: &Dsv41Config, max_rows: usize, n_own: usize) -> u64 {
    use super::frame::MIX_STRIDE;
    let h = cfg.hidden;
    let frame = b256(max_rows * (h + MIX_STRIDE + (cfg.hc + 1) * h) * 4);
    let (pp, cc) = crate::dsv41::gpu::attn::DecodeScratch::bytes(HEAD_MAX, cfg.heads, crate::dsv41::gpu::attn::MAX_SPLITS);
    let rows = HEAD_MAX.max(max_rows);
    2 * frame + b256(max_rows * h * 4) + b256(2 * max_rows.max(1) * n_own.max(1) * 256 * 2) + b256(rows * 4) + b256(pp) + b256(cc)
        + b256(rows * std::mem::size_of::<super::comm::TpRow>()) + b256(rows * std::mem::size_of::<super::comm::TpOut>()) + 2 * b256(max_rows * 4)
        + 4 * 256 + b256(super::fwd::MAX_ROW_BLOCKS * 8 * crate::dsv41::gpu::indexer::CAND_BLOCKS * 4) + 256
        + cfg.index_sources.len() as u64 * b256(max_rows * 512 * 4)
        + b256(max_rows)
}

/// The MoE scratch the prefill rows need before the weights are loaded (the fit; KE2 / PK2's wide layout: the routed
/// pairs plus the shared expert's rows, their gate / up f32, the activation fp16 and the down f32 at the rank's inner width).
pub fn moe_scratch_est(cfg: &Dsv41Config, world: usize, rows: usize) -> u64 {
    let rows = rows.min(16384);
    let p = rows * (cfg.top_k + 1);
    let inter_ld = (cfg.moe_inter / world.max(1)).div_ceil(128) * 128;
    (p * 2 * inter_ld * 4 + p * inter_ld * 2 + p * cfg.hidden * 4) as u64 + 16 * MIB
}

/// The Engram runtime's (pinned cache + slot tables, host index) bytes for a `cache_mib` cache (EIO's `RowCache`:
/// per slot key / refs / clock / free list, an open-addressed index of twice the slots).
pub fn engram_bytes(cache_mib: u64) -> (u64, u64) {
    let slots = (cache_mib * MIB / crate::dsv41::gpu::engram::SLOT_STRIDE as u64).max(1);
    (slots * crate::dsv41::gpu::engram::SLOT_STRIDE as u64 + 8 * MIB, slots * 17 + (2 * slots).next_power_of_two() * 12)
}

// ---------------------------------------------------------------------------------------------------------------------
// the scratch need
// ---------------------------------------------------------------------------------------------------------------------

fn lin_need(w: &(DenseWeight, BackendKind), rows: usize) -> usize { crate::dsv41::fwd::glue::backend(w.1).scratch_bytes(&w.0, rows) }

/// The scratch every linear / MoE call of `layers` needs at the rows the forward calls them at: trunk layers (index
/// < `n_trunk`) at 1..=32 rows (decode / verify) and 33..=`max_rows` (prefill, the wide family), DSpark stages at
/// <= 32 rows, the head at <= `HEAD_MAX` rows and at 33 (a prefill's head runs padded to the wide family).
pub fn scratch_need(layers: &[Option<TpLayer>], head: Option<&(DenseWeight, BackendKind)>, n_trunk: usize, max_rows: usize) -> Result<usize> {
    use crate::dsv41::weights::exl3::moe::{scratch_for, MoePath};
    let dec = [1usize, 2, 4, 8, 16, 24, 32];
    let mut need = 0usize;
    for (l, tl) in layers.iter().enumerate() {
        let Some(tl) = tl else { continue };
        let mut rows: Vec<usize> = dec.to_vec();
        if l < n_trunk && max_rows > 32 { rows.push(33); rows.push(max_rows); }
        for &r in &rows {
            if let Some((mw, BackendKind::Exl3)) = tl.w.moe.as_ref() { need = need.max(scratch_for(mw, Some(&tl.shared), r, MoePath::Auto)?); }
            for w in tl.w.lin.values() { need = need.max(lin_need(w, r)); }
            need = need.max(lin_need(&tl.wq_b, r)).max(lin_need(&tl.wo_b, r));
            for e in &tl.engram { need = need.max(lin_need(&e.w, r)); }
        }
    }
    if let Some(h) = head { for r in (1..=HEAD_MAX).chain([33]) { need = need.max(lin_need(h, r)); } }
    Ok(need.div_ceil(1 << 20) << 20)
}

// ---------------------------------------------------------------------------------------------------------------------
// the box
// ---------------------------------------------------------------------------------------------------------------------

/// `/proc/meminfo` MemAvailable in bytes.
pub fn avail() -> Option<u64> { crate::memwatch::mem_available_bytes() }

/// One `/proc/meminfo` field in bytes.
pub fn meminfo(key: &str) -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let l = s.lines().find(|l| l.starts_with(key) && l[key.len()..].starts_with(':'))?;
    l.split_whitespace().nth(1)?.parse::<u64>().ok().map(|kb| kb * 1024)
}

/// The memwatch floor in bytes (`--mem-watchdog-gb`, default 5; 0 = the watchdog is off: no floor).
pub fn floor() -> u64 {
    let gb: f64 = crate::opts::var(crate::opt!("mem-watchdog-gb")).ok().and_then(|v| v.parse().ok()).unwrap_or(5.0);
    (gb.max(0.0) * 1e9) as u64
}

/// What the kernel can give a new process besides MemAvailable: half of the box's anonymous and shared pages, as far as
/// the free swap holds them (other processes' idle pages move to swap when the weights load; GR's boot moved ~6 GB).
pub fn swap_reclaimable() -> u64 {
    let (sf, an, sh) = (meminfo("SwapFree").unwrap_or(0), meminfo("AnonPages").unwrap_or(0), meminfo("Shmem").unwrap_or(0));
    sf.min((an + sh) / 2)
}

/// This process's resident host memory: (anonymous, shared) bytes (`/proc/self/status`).
pub fn rss() -> (u64, u64) {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let f = |k: &str| s.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0) * 1024;
    (f("RssAnon:"), f("RssShmem:"))
}

/// The default memory pool of the device: (reserved, used) bytes (`cuMemPoolGetAttribute`).
pub fn pool_mem(dev: &cudarc::driver::CudaDevice) -> (u64, u64) {
    use cudarc::driver::sys;
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
    unsafe {
        if sys::cuDeviceGetDefaultMemPool(&mut pool, *dev.cu_device()) != sys::CUresult::CUDA_SUCCESS { return (0, 0); }
        let (mut r, mut u) = (0u64, 0u64);
        sys::cuMemPoolGetAttribute(pool, sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT, &mut r as *mut u64 as *mut std::ffi::c_void);
        sys::cuMemPoolGetAttribute(pool, sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT, &mut u as *mut u64 as *mut std::ffi::c_void);
        (r, u)
    }
}

/// Freed memory back to the box: the device synchronised, the default pool trimmed to what is in use, the host heap's
/// free pages returned (`malloc_trim`). Never fails the caller (the trims are an optimisation).
pub fn trim(dev: &cudarc::driver::CudaDevice) {
    use cudarc::driver::sys;
    if dev.bind_to_thread().is_err() { return; }
    unsafe {
        sys::cuCtxSynchronize();
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        if sys::cuDeviceGetDefaultMemPool(&mut pool, *dev.cu_device()) == sys::CUresult::CUDA_SUCCESS { sys::cuMemPoolTrimTo(pool, 0); }
        libc::malloc_trim(0);
    }
    TRIMS.fetch_add(1, Relaxed);
}

// ---------------------------------------------------------------------------------------------------------------------
// the ledger
// ---------------------------------------------------------------------------------------------------------------------

static PHASES: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());
static PARTS: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());
static TRIMS: AtomicU64 = AtomicU64::new(0);
static FLOOR: AtomicU64 = AtomicU64::new(0);
static AVAIL: AtomicU64 = AtomicU64::new(0);
static ALLOWANCE: AtomicU64 = AtomicU64::new(0);
static PRESSURE_EVICTIONS: AtomicU64 = AtomicU64::new(0);
static PRESSURE_EVENTS: AtomicU64 = AtomicU64::new(0);
static ARENA_PEAK: AtomicU64 = AtomicU64::new(0);
static PREDICTED: AtomicU64 = AtomicU64::new(0);
static ON: AtomicU64 = AtomicU64::new(0);

/// Record the box's MemAvailable at a boot phase (the ledger's "free after ..." column).
pub fn phase(name: &'static str) {
    if let Some(a) = avail() { if let Ok(mut p) = PHASES.lock() { p.push((name, a)); } }
}

/// The fit's prediction of this rank's resident bytes (weights + serving state, the pool's budget excluded).
pub fn set_predicted(bytes: u64) { PREDICTED.store(bytes, Relaxed); }

/// Prefix states the governor gave back (`n`), one pressure event.
pub fn pressure_evicted(n: u64) { PRESSURE_EVENTS.fetch_add(1, Relaxed); PRESSURE_EVICTIONS.fetch_add(n, Relaxed); }

/// The forward's fixed device buffers (`fixed_bytes`, from the live objects).
fn fixed_dev(m: &TpModel) -> u64 {
    let mut t = (m.frames[0].bytes + m.frames[1].bytes + m.bcast.bytes + m.eout.bytes + m.ds.0.bytes + m.ds.1.bytes + m.rows_p.bytes + m.outs.bytes
                 + m.pos_buf.bytes + m.ids_buf.bytes + m.lane_buf.bytes + m.tiny.bytes + m.tick.bytes + m.kc_freq[0].bytes + m.kc_freq[1].bytes
                 + m.vl_mask_buf.bytes) as u64;
    if let Some((a, b)) = m.uni_bufs.as_ref() { t += (a.bytes + b.bytes) as u64; }
    t + m.sel_bufs.values().map(|b| b.bytes as u64).sum::<u64>()
}

/// The largest use either activation arena has seen.
pub fn arena_peak(m: &TpModel) -> usize { m.f.b.peak.max(m.spare.peak) }

/// The parts this rank holds on the device (bytes), from the live objects.
pub fn device_parts(m: &TpModel) -> Vec<(&'static str, u64)> {
    let mut v: Vec<(&'static str, u64)> = Vec::new();
    v.push(("weights", m.weights_bytes));
    v.push(("scratch", m.scr_buf.bytes as u64));
    v.push(("arenas", (m.f.b.base.bytes + m.spare.base.bytes) as u64));
    v.push(("fixed buffers", fixed_dev(m)));
    let lanes: u64 = m.lanes.iter().map(|ln| {
        ln.win.iter().map(|w| (w.ring_q.bytes + w.ring_s.bytes + w.lin_q.bytes + w.lin_s.bytes) as u64).sum::<u64>()
            + ln.src.values().map(|s| (s.kv_main.bytes + s.iq.bytes + s.is.bytes + s.latent.bytes) as u64 + s.front.as_ref().map_or(0, |(a, b)| (a.bytes + b.bytes) as u64)).sum::<u64>()
    }).sum();
    v.push(("lanes", lanes));
    if let Some(g) = m.graphs.as_ref() { v.push(("graphs", g.bytes())); }
    v
}

/// The boot ledger of this rank: the parts (device, Engram, prefix states, host), the box's MemAvailable at each
/// phase, the floor. Returns (one summary line, one breakdown line) and publishes the `/metrics` gauges.
pub fn ledger(dev: &cudarc::driver::CudaDevice, m: &TpModel, spec_bytes: u64, snaps: u64, rank: usize) -> (String, String) {
    let mut parts = device_parts(m);
    parts.push(("dspark round", spec_bytes));
    if let Some(io) = m.eio.as_ref() { let (p, h) = io.mem_bytes(); parts.push(("engram cache (pinned)", p)); parts.push(("engram index (host)", h)); }
    parts.push(("prefix states", snaps));
    let (pr, pu) = pool_mem(dev);
    parts.push(("pool slack", pr.saturating_sub(pu)));
    let (an, sh) = rss();
    parts.push(("process host", an + sh));
    let fl = floor();
    let av = avail().unwrap_or(0);
    FLOOR.store(fl, Relaxed);
    AVAIL.store(av, Relaxed);
    ON.store(1, Relaxed);
    ARENA_PEAK.store(arena_peak(m) as u64, Relaxed);
    let gib = |b: u64| b as f64 / GIB as f64;
    let w = parts.iter().find(|p| p.0 == "weights").map_or(0, |p| p.1);
    let rest: u64 = parts.iter().filter(|p| p.0 != "weights").map(|p| p.1).sum();
    let phases = PHASES.lock().map(|p| p.clone()).unwrap_or_default();
    let summary = format!("rank {rank}: memory: weights {:.2} GiB + serving state {:.2} GiB; the box has {:.1} GB free, {:.1} GB above the {:.1} GB memory-watchdog floor",
                          gib(w), gib(rest), av as f64 / 1e9, (av as f64 - fl as f64) / 1e9, fl as f64 / 1e9);
    let detail = format!("rank {rank}: memory ledger: {}; free at {}; fit predicted {:.2} GiB; arena peak {:.0} MiB",
                         parts.iter().map(|(n, b)| format!("{n} {:.3} GiB", gib(*b))).collect::<Vec<_>>().join(", "),
                         phases.iter().map(|(n, a)| format!("{n} {:.1} GB", *a as f64 / 1e9)).collect::<Vec<_>>().join(" / "),
                         gib(PREDICTED.load(Relaxed)), ARENA_PEAK.load(Relaxed) as f64 / MIB as f64);
    if let Ok(mut p) = PARTS.lock() { *p = parts; }
    (summary, detail)
}

/// Refresh the parts that change while serving (graphs, prefix states, pool slack, process host, arena peak).
pub fn refresh(dev: &cudarc::driver::CudaDevice, m: &TpModel, snaps: u64) {
    if ON.load(Relaxed) == 0 { return; }
    let (pr, pu) = pool_mem(dev);
    let (an, sh) = rss();
    let graphs = m.graphs.as_ref().map_or(0, |g| g.bytes());
    ARENA_PEAK.store(arena_peak(m) as u64, Relaxed);
    if let Ok(mut p) = PARTS.lock() {
        for e in p.iter_mut() {
            match e.0 { "graphs" => e.1 = graphs, "prefix states" => e.1 = snaps, "pool slack" => e.1 = pr.saturating_sub(pu), "process host" => e.1 = an + sh, _ => {} }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// the pressure governor
// ---------------------------------------------------------------------------------------------------------------------

/// What the prefix pool may hold now: its current bytes plus what is free above `floor + reserve`, never above `max`.
pub fn pool_allowance(pool_bytes: u64, avail: u64, floor: u64, reserve: u64, max: u64) -> u64 {
    let free = avail as i128 - floor.saturating_add(reserve) as i128;
    (pool_bytes as i128 + free).clamp(0, max as i128) as u64
}

/// The head's governor: samples MemAvailable at most every `every` and returns the pool's allowance.
pub struct Governor { pub floor: u64, pub reserve: u64, pub max: u64, every: Duration, last: Option<Instant>, pub last_allowance: Option<u64>, trimmed: Option<Instant> }

impl Governor {
    /// `max` = the pool's asked budget; `snap_max` = one prefix state of the longest context.
    pub fn new(max: u64, snap_max: u64) -> Governor {
        Governor { floor: floor(), reserve: RESERVE + snap_max, max, every: Duration::from_millis(250), last: None, last_allowance: None, trimmed: None }
    }

    /// The pool's allowance now (None between samples, or when the floor is off and nothing limits the pool).
    pub fn poll(&mut self, pool_bytes: u64) -> Option<u64> {
        if self.last.map_or(false, |t| t.elapsed() < self.every) { return None; }
        self.last = Some(Instant::now());
        let a = avail()?;
        AVAIL.store(a, Relaxed);
        FLOOR.store(self.floor, Relaxed);
        let al = if self.floor == 0 { self.max } else { pool_allowance(pool_bytes, a, self.floor, self.reserve, self.max) };
        ALLOWANCE.store(al, Relaxed);
        self.last_allowance = Some(al);
        Some(al)
    }

    /// A trim is due after states were given back (at most once a second), or after `idle_every` without one.
    pub fn trim_due(&mut self, evicted: bool, idle_every: Option<Duration>) -> bool {
        let since = self.trimmed.map_or(Duration::MAX, |t| t.elapsed());
        let due = (evicted && since >= Duration::from_secs(1)) || idle_every.map_or(false, |e| since >= e);
        if due { self.trimmed = Some(Instant::now()); }
        due
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// /metrics
// ---------------------------------------------------------------------------------------------------------------------

pub fn append(o: &mut String) {
    if ON.load(Relaxed) == 0 { return; }
    use std::fmt::Write as _;
    let _ = writeln!(o, "# HELP velogb10_dsv41_mem_bytes this rank's memory by part (device, Engram pinned / host, prefix states, host process)\n# TYPE velogb10_dsv41_mem_bytes gauge");
    if let Ok(p) = PARTS.lock() { for (n, b) in p.iter() { let _ = writeln!(o, "velogb10_dsv41_mem_bytes{{part=\"{n}\"}} {b}"); } }
    let g: [(&str, &str, &str, u64); 9] = [
        ("velogb10_dsv41_mem_floor_bytes", "gauge", "the memory watchdog's floor of MemAvailable (--mem-watchdog-gb)", FLOOR.load(Relaxed)),
        ("velogb10_dsv41_mem_available_bytes", "gauge", "the box's MemAvailable at the governor's last sample", AVAIL.load(Relaxed)),
        ("velogb10_dsv41_mem_pool_allowance_bytes", "gauge", "what the prefix pool may hold now (its states + free memory above floor + reserve, at most its budget)", ALLOWANCE.load(Relaxed)),
        ("velogb10_dsv41_mem_pressure_events_total", "counter", "times the prefix pool had to give states back because the box's free memory neared the floor", PRESSURE_EVENTS.load(Relaxed)),
        ("velogb10_dsv41_mem_pressure_evictions_total", "counter", "prefix states given back under memory pressure", PRESSURE_EVICTIONS.load(Relaxed)),
        ("velogb10_dsv41_mem_trims_total", "counter", "device pool + host heap trims (boot, after pressure, idle)", TRIMS.load(Relaxed)),
        ("velogb10_dsv41_mem_arena_peak_bytes", "gauge", "the largest activation-arena use seen (per arena)", ARENA_PEAK.load(Relaxed)),
        ("velogb10_dsv41_mem_fit_predicted_bytes", "gauge", "the boot's memory-fit prediction of this rank's resident bytes (pool excluded)", PREDICTED.load(Relaxed)),
        ("velogb10_dsv41_mem_graphs_live", "gauge", "captured CUDA graphs this rank keeps", super::graph::live()),
    ];
    for (n, t, h, v) in g { let _ = writeln!(o, "# HELP {n} {h}\n# TYPE {n} {t}\n{n} {v}"); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv41::config::{test_config_json, Dsv41Config};

    fn cfg() -> Dsv41Config { Dsv41Config::from_value(&test_config_json(), None).unwrap() }

    #[test]
    fn dsv41_mem1_the_pool_gets_what_is_free_above_floor_plus_reserve() {
        let g = GIB;
        // 9 GB free, floor 5, reserve 1.5 -> 2.5 GB more than the pool holds now
        assert_eq!(pool_allowance(g, 9 * g, 5 * g, 3 * g / 2, 6 * g), g + 5 * g / 2);
        // below the line: the pool gives back exactly the shortfall
        assert_eq!(pool_allowance(2 * g, 6 * g, 5 * g, 3 * g / 2, 6 * g), 2 * g - g / 2);
        // never negative, never above its budget
        assert_eq!(pool_allowance(g, 4 * g, 5 * g, g, 6 * g), 0);
        assert_eq!(pool_allowance(0, 100 * g, 5 * g, g, 2 * g), 2 * g);
    }

    #[test]
    fn dsv41_mem1_lane_and_state_formulas_scale_with_context_and_rows() {
        let c = cfg();
        let (a, b) = (lane_bytes(&c, 1024, 16384), lane_bytes(&c, 1024, 262144));
        assert!(b > a, "{a} {b}");
        assert!(lane_bytes(&c, 2048, 16384) > a);
        // a state holds every layer's ring and the groups of its position: monotone in pos, 256-B aligned pieces
        let (l0, t0) = snap_layout(&c, 0);
        let (l1, t1) = snap_layout(&c, 100_000);
        assert_eq!(l0.len(), l1.len());
        assert!(t1 > t0 && t1 % 256 == 0);
        for w in l1.windows(2) { assert!(w[1].0 >= w[0].0 + w[0].1 && w[1].0 % 256 == 0); }
        assert_eq!(snap_bytes(&c, 100_000), t1 as u64);
        assert_eq!(arena_bytes(64), 256 << 20);
        assert_eq!(arena_bytes(2048), 1 << 30);
    }

    #[test]
    fn dsv41_mem1_the_engram_cache_and_its_index_shrink_together() {
        let (p2, h2) = engram_bytes(2048);
        let (p5, h5) = engram_bytes(512);
        assert!(p2 > 3 * p5 && h2 > 3 * h5, "{p2} {h2} {p5} {h5}");
        // 2 GiB: ~7.9 M slots; the host index ~335 MB
        assert!(h2 > 300 * MIB && h2 < 360 * MIB, "{h2}");
    }

    #[test]
    fn dsv41_mem1_a_trim_is_due_after_evictions_or_idle_time() {
        let mut g = Governor::new(GIB, 0);
        assert!(g.trim_due(true, None));
        assert!(!g.trim_due(true, None), "at most once a second");
        assert!(!g.trim_due(false, Some(Duration::from_secs(120))));
    }
}
