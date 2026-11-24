//! KE2: host side of the decode MoE (kernels/gpu_dsv41/15_exl3_moe_dec.cuh): <= 32 rows, routed + shared expert in one
//! call, three launches (prep, gate/up, down), persistent grids sized from the occupancy API at the real dynamic smem
//! (two CTAs per SM at every row count when the ring fits: PK4),
//! the PDL launch attribute when `Launch::pdl`. Numerics equal the generic path bit for bit (the probe gates it).
//! Eligible when every width code is an integer K in 2..=8 and the table fits the in-CTA routing (<= 1024 experts,
//! top-k <= 8); anything else runs the generic path (counted by the caller as `RowClass::Generic`).

use super::launch::{act_code, kernel, Kernel};
use super::moe::{sm_count, MoeTable};
use crate::dsv41::weights::{Act, ActDtype, Launch, MoeWeight, Route, Scratch};
use anyhow::{ensure, Context, Result};
use cudarc::driver::sys;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

/// D41X_MD_MAXS: routing slots (256 distinct routed experts + the shared expert).
pub const MAXS: usize = 257;
pub const MAX_EXPERTS: usize = 1024;

/// kernels/gpu_dsv41/15_exl3_moe_dec.cuh `D41xMdArgs` (size checked against the device at first use).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct MdArgs {
    pub ex: u64, pub sh: u64, pub ids: u64, pub wts: u64,
    pub x: u64, pub xcode: i32, pub ldx: i32,
    pub xh1: u64, pub graw: u64, pub xh2: u64, pub h2: u64,
    pub out: u64, pub ldo: i32,
    pub cnt: u64,
    pub limit: f32, pub limit_sh: f32,
    pub n_experts: i32, pub rows: i32, pub topk: i32, pub dim: i32, pub inter_ld: i32, pub ncb: i32, pub kmax: i32, pub kmin: i32,
    pub slot: i32, pub ns: i32, pub wreg: i32, pub sms: i32,
    /// PV1: per-k-block chains of gate/up K parts [dim / 128][pair][2][inter_ld] f32, 0 = one pass over K (no K parts)
    pub chain: u64,
    /// leg PF1: 1 = the combine OVERWRITES y (writes 0 + sum: a zeroed y's bits without a memset before the call)
    pub ow: i32,
    pub pad: i32,
}

/// The device's view of the geometry: the routing-header size (the block is header + 6 B per expert, 128-aligned).
pub struct Geometry { pub route_hdr: usize, pub kps: usize }
static GEOM: OnceLock<Geometry> = OnceLock::new();

/// Read the kernels' geometry once (a tiny kernel + sync: call before any graph capture; prepare_moe does).
pub fn geometry() -> Result<&'static Geometry> {
    if let Some(g) = GEOM.get() { return Ok(g); }
    let k = kernel("d41x_md_sizes")?;
    let mut d: sys::CUdeviceptr = 0;
    let mut h = [0i32; 16];
    unsafe {
        ensure!(sys::cuMemAlloc_v2(&mut d, 64) == sys::CUresult::CUDA_SUCCESS, "cuMemAlloc (md sizes)");
        let mut p = d as u64;
        let mut params = [&mut p as *mut u64 as *mut c_void];
        let mut st: sys::CUstream = std::ptr::null_mut();
        ensure!(sys::cuStreamCreate(&mut st, 0) == sys::CUresult::CUDA_SUCCESS, "cuStreamCreate (md sizes)");
        let r = sys::cuLaunchKernel(k.0, 1, 1, 1, 32, 1, 1, 0, st, params.as_mut_ptr(), std::ptr::null_mut());
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "launch d41x_md_sizes: {r:?}");
        ensure!(sys::cuStreamSynchronize(st) == sys::CUresult::CUDA_SUCCESS, "sync (md sizes)");
        sys::cuStreamDestroy_v2(st);
        ensure!(sys::cuMemcpyDtoH_v2(h.as_mut_ptr() as *mut c_void, d, 64) == sys::CUresult::CUDA_SUCCESS, "read md sizes");
        sys::cuMemFree_v2(d);
    }
    ensure!(h[0] as usize == std::mem::size_of::<MdArgs>(), "D41xMdArgs is {} B on the device, {} B in Rust (stale PTX or a layout change)", h[0], std::mem::size_of::<MdArgs>());
    ensure!(h[5] as usize == std::mem::size_of::<super::launch::ExpertEntry>(), "D41xExpert is {} B on the device, {} B in Rust (stale PTX or a layout change)", h[5], std::mem::size_of::<super::launch::ExpertEntry>());
    ensure!(h[2] as usize == MAXS, "D41X_MD_MAXS {} != {MAXS}", h[2]);
    let g = Geometry { route_hdr: h[1] as usize, kps: h[3] as usize };
    ensure!(route_bytes(&g, 384) == h[4] as usize, "routing block size mismatch: {} vs {}", route_bytes(&g, 384), h[4]);
    let _ = GEOM.set(g);
    Ok(GEOM.get().unwrap())
}
fn route_bytes(g: &Geometry, e: usize) -> usize { (g.route_hdr + 6 * e + 127) / 128 * 128 }

/// Per-CTA dynamic smem that lets two CTAs share an SM (GB10: 100 KB per SM, 1 KB reserved per CTA), and the 1-CTA cap.
const SMEM_2CTA: usize = 48 * 1024;
const SMEM_1CTA: usize = 101_376;
const NS_MAX: usize = 16;

/// Ring geometry of a call: (slot bytes, slots, trellis bytes per slot, dynamic smem) for gate/up (gu) or down.
fn ring(g: &Geometry, rows: usize, kmax: i32, e: usize, gu: bool) -> Result<(usize, usize, usize, usize)> {
    let wreg = g.kps * 8 * 32 * kmax as usize;
    let abytes = g.kps * (if gu { 2 } else { 1 }) * rows * 32;
    let slot = (wreg + abytes + 127) / 128 * 128;
    let stage = if gu { 0 } else { rows.max(16).next_power_of_two().max(16) * 132 * 4 };
    let fixed = route_bytes(g, e) + stage;
    // PK4: two CTAs per SM at every row count when >= 3 slots fit (the 17-32-row instances compile spill-free at 128
    // registers); one CTA per SM only when the slots are too wide (e.g. a K8 table at 32 rows)
    let cap = SMEM_2CTA;
    let mut ns = cap.saturating_sub(fixed) / slot;
    if ns < 3 { ns = SMEM_1CTA.saturating_sub(fixed) / slot; }
    ensure!(ns >= 3, "decode MoE ring: slot {slot} B leaves fewer than 3 slots");
    let ns = ns.min(NS_MAX);
    Ok((slot, ns, wreg, fixed + ns * slot))
}


/// Can the decode kernels serve this table (+ shared table)?
pub fn eligible(t: &MoeTable, ts: Option<&MoeTable>, w: &MoeWeight, sh: Option<&MoeWeight>) -> bool {
    t.int_k() && w.n_experts <= MAX_EXPERTS && w.top_k <= 8 && w.dim % 128 == 0 && w.inter % 128 == 0
        && ts.map_or(true, |s| s.int_k() && s.dim == t.dim && s.nblk_max <= t.ncb)
        && sh.map_or(true, |s| s.n_experts == 1 && s.top_k == 1 && s.dim == w.dim)
}

fn a256(b: usize) -> usize { (b + 255) & !255 }
/// PV1: rows up to which a call gets the chain scratch that lets the gate/up kernel split K (15_*: the block count picks
/// the K parts on the device; the result is the same bits either way)
pub const CHAIN_ROWS: usize = 4;
struct Lay { xh1: usize, graw: usize, xh2: usize, h2: usize, chain: usize, total: usize, inter_ld: usize }
fn layout(t: &MoeTable, ts: Option<&MoeTable>, rows: usize, topk: usize, with_chain: bool) -> Lay {
    let p = rows * topk + if ts.is_some() { rows } else { 0 };
    let inter_ld = t.nblk_max.max(ts.map_or(0, |s| s.nblk_max)) * 128;
    let dim = t.dim;
    let xh1 = 0;
    let graw = xh1 + a256(p * 2 * dim * 2);
    let xh2 = graw + a256(p * 2 * inter_ld * 4);
    let h2 = xh2 + a256(p * inter_ld * 2);
    let chain = h2 + a256(p * dim * 4);
    let total = if with_chain { chain + a256((dim / 128) * p * 2 * inter_ld * 4) } else { chain };
    Lay { xh1, graw, xh2, h2, chain, total, inter_ld }
}
pub fn scratch_bytes(t: &MoeTable, ts: Option<&MoeTable>, rows: usize, topk: usize) -> usize {
    // PM1: the short-CTA path (moe_cp.rs) serves most decode shapes and needs its chain planes + routing table
    layout(t, ts, rows, topk, rows <= CHAIN_ROWS).total.max(super::moe_cp::scratch_bytes(t, ts, rows, topk))
}

/// PV1 engagement receipts (for /metrics): decode MoE calls run, and calls given the chain scratch (gate/up may split K)
static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CHAIN_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn calls() -> u64 { CALLS.load(std::sync::atomic::Ordering::Relaxed) }
pub fn chain_calls() -> u64 { CHAIN_CALLS.load(std::sync::atomic::Ordering::Relaxed) }
/// F4 (RV1): calls that were owed the chain scratch (rows <= CHAIN_ROWS) and ran the one-pass-over-K path because the caller's
/// scratch was short. Same bits, but the PV1 K-split gain is not delivered on those calls: counted, and ONE line per process.
static SHORT_SCRATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn short_scratch_calls() -> u64 { SHORT_SCRATCH.load(std::sync::atomic::Ordering::Relaxed) }
fn note_short_scratch(rows: usize, have: usize, need: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    if SHORT_SCRATCH.fetch_add(1, Relaxed) == 0 {
        crate::boot_report::notice(&format!("dsv41 decode MoE: the caller's scratch ({have} B) is short of the chain scratch ({need} B, {rows} row(s)): these calls run the one-pass-over-K path (same bits, without the PV1 K-split gain); counted in velogb10_dsv41_moe_dec_short_scratch_total"));
    }
}
/// `/metrics` rows of PV1's decode MoE path (the counts are launches the host built: a graph replay re-runs a captured launch without counting).
pub fn render(o: &mut String) {
    for (n, h, v) in [
        ("velogb10_dsv41_moe_dec_calls_total", "Decode MoE launches built by the host (graph replays re-run captured launches without counting)", calls()),
        ("velogb10_dsv41_moe_dec_chain_calls_total", "Decode MoE launches given the chain scratch (the gate/up kernel may split K into parts)", chain_calls()),
        ("velogb10_dsv41_moe_dec_short_scratch_total", "Decode MoE launches owed the chain scratch that ran one pass over K because the caller's scratch was short", short_scratch_calls()),
    ] { o.push_str(&format!("# HELP {n} {h}\n# TYPE {n} counter\n{n} {v}\n")); }
    super::moe_cp::render(o);
}

/// The kernel instance for a row count and the widest K of the call: (gu, dn) entry names.
fn instance(rows: usize, kmax: i32) -> (&'static str, &'static str) {
    let a = if rows <= 8 { 0 } else if rows <= 16 { 1 } else { 2 };
    const GU: [&str; 6] = ["d41x_md_gu_a8_k5", "d41x_md_gu_a16_k5", "d41x_md_gu_a32_k5", "d41x_md_gu_a8_k8", "d41x_md_gu_a16_k8", "d41x_md_gu_a32_k8"];
    const DN: [&str; 6] = ["d41x_md_dn_a8_k5", "d41x_md_dn_a16_k5", "d41x_md_dn_a32_k5", "d41x_md_dn_a8_k8", "d41x_md_dn_a16_k8", "d41x_md_dn_a32_k8"];
    let i = a + if kmax > 5 { 3 } else { 0 };
    (GU[i], DN[i])
}

/// Per (function, smem): the > 48 KB smem opt-in (the attribute is one value per function: raised, never lowered) and the
/// persistent grid (SMs x resident CTAs at that smem).
pub(super) fn grid_for(name: &str, smem: u32) -> Result<(Kernel, u32)> {
    static G: OnceLock<Mutex<(HashMap<String, u32>, HashMap<(String, u32), u32>)>> = OnceLock::new();
    let k = kernel(name)?;
    let m = G.get_or_init(|| Mutex::new((HashMap::new(), HashMap::new())));
    let mut g = m.lock().unwrap();
    if let Some(n) = g.1.get(&(name.to_string(), smem)) { return Ok((k, *n)); }
    unsafe {
        let cur = g.0.get(name).copied().unwrap_or(0);   // SX1: opt in from the first request: a kernel with static smem cannot take 48 KB dynamic without it
        if smem > cur {
            let r = sys::cuFuncSetAttribute(k.0, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32);
            ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuFuncSetAttribute({name}, smem {smem}): {r:?}");
            g.0.insert(name.to_string(), smem);
        }
        let mut n = 0i32;
        let r = sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut n, k.0, 256, smem as usize);
        ensure!(r == sys::CUresult::CUDA_SUCCESS && n > 0, "occupancy of {name} at smem {smem}: {r:?} ({n} CTAs / SM)");
        let grid = (n * sm_count()) as u32;
        g.1.insert((name.to_string(), smem), grid);
        Ok((k, grid))
    }
}

/// `cuLaunchKernelEx` with the PDL attribute when asked (the kernels wait before reading their predecessor's output).
pub fn launch_ex(k: &Kernel, grid: (u32, u32, u32), block: u32, smem: u32, l: &Launch, params: &mut [*mut c_void]) -> Result<()> {
    ensure!(l.stream != 0, "dsv41 kernels never run on the legacy default stream (AGENTS 2.1)");
    let mut attr = sys::CUlaunchAttribute {
        id: if l.pdl { sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION } else { sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_IGNORE },
        pad: [0; 4],
        value: if l.pdl { sys::CUlaunchAttributeValue { programmaticStreamSerializationAllowed: 1 } } else { sys::CUlaunchAttributeValue_union { pad: [0; 64] } },
    };
    let mut cfg = sys::CUlaunchConfig {
        gridDimX: grid.0, gridDimY: grid.1, gridDimZ: grid.2, blockDimX: block, blockDimY: 1, blockDimZ: 1,
        sharedMemBytes: smem, hStream: l.stream as sys::CUstream, attrs: &mut attr, numAttrs: if l.pdl { 1 } else { 0 },
    };
    let r = unsafe { sys::cuLaunchKernelEx(&mut cfg, k.0, params.as_mut_ptr(), std::ptr::null_mut()) };
    if r != sys::CUresult::CUDA_SUCCESS {
        let q = |a: sys::CUfunction_attribute| { let mut v = -1i32; unsafe { sys::cuFuncGetAttribute(&mut v, a, k.0) }; v };
        use sys::CUfunction_attribute as A;
        anyhow::bail!("cuLaunchKernelEx: {r:?} (grid {grid:?} block {block} dyn smem {smem}; function: regs {}, static smem {}, max threads {}, max dyn smem attr {})",
            q(A::CU_FUNC_ATTRIBUTE_NUM_REGS), q(A::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES), q(A::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK), q(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES));
    }
    Ok(())
}

/// The decode MoE: y [rows, dim] f32 += routed (+ shared). Rows <= 32; `eligible` must hold.
#[allow(clippy::too_many_arguments)]
pub fn run(l: &Launch, t: &MoeTable, ts: Option<&MoeTable>, w: &MoeWeight, sh: Option<&MoeWeight>, x: &Act, route: &Route, y: &Act, s: Scratch) -> Result<()> {
    run_ow(l, t, ts, w, sh, x, route, y, s, false)
}

/// `run` with `overwrite` (leg PF1): y = routed (+ shared), written by the combine as `0 + sum` (needs the shared expert:
/// then every output block of every row has a slot and is written).
///
/// Launch order (leg PF1, the structure of the external reference's TP2 decode MoE): prep, gate/up, down, all PDL. The
/// down grid does NOT wait for the whole gate/up grid: gate/up publishes each slot's readiness (the epoch of this call, set
/// by prep) when the last of the slot's 128-blocks has written its SwiGLU rows, and a down CTA waits for its own item's
/// slot only (slot 0 = the shared expert first), after issuing the trellis of its first ring slots.
#[allow(clippy::too_many_arguments)]
pub fn run_ow(l: &Launch, t: &MoeTable, ts: Option<&MoeTable>, w: &MoeWeight, sh: Option<&MoeWeight>, x: &Act, route: &Route, y: &Act, s: Scratch,
              overwrite: bool) -> Result<()> {
    ensure!(!overwrite || sh.is_some(), "moe {}: overwrite needs the shared expert (every block of every row written)", w.layer);
    let rows = x.rows;
    ensure!(rows >= 1 && rows <= 32, "decode MoE: {rows} rows");
    ensure!(y.dtype == ActDtype::F32 && x.cols == w.dim && y.cols == w.dim && y.rows == rows, "moe {}: y must be f32 [rows, dim]", w.layer);
    ensure!(route.top_k == w.top_k, "moe {}: route top_k {} != {}", w.layer, route.top_k, w.top_k);
    // the chain scratch when the caller's scratch holds it (rows <= CHAIN_ROWS); otherwise one pass over K, same bits
    let mut lay = layout(t, ts, rows, w.top_k, rows <= CHAIN_ROWS);
    if s.bytes < lay.total {
        lay = layout(t, ts, rows, w.top_k, false);
        if rows <= CHAIN_ROWS { note_short_scratch(rows, s.bytes, scratch_bytes(t, ts, rows, w.top_k)); }
    }
    ensure!(s.bytes >= lay.total, "moe {}: scratch {} < {}", w.layer, s.bytes, lay.total);
    let with_chain = rows <= CHAIN_ROWS && lay.total > lay.chain;
    let g = geometry()?;
    // SX1: the ring and the KMAX instance cover the widest matrix of either table (w2 included): an item is consumed by
    // the switch on its own K, which has a case for every K <= the instance's KMAX
    let kmax = t.kmax().max(ts.map_or(0, |s| s.kmax()));
    let kmin = t.kmin().min(ts.map_or(i32::MAX, |s| s.kmin()));
    ensure!((2..=8).contains(&kmin) && kmax <= 8, "moe {}: decode kernels need integer K 2..8, the call spans {kmin}..{kmax}", w.layer);
    let (gu, dn) = instance(rows, kmax);
    let (slot_gu, ns_gu, wreg, sm_gu) = ring(g, rows, kmax, w.n_experts, true)?;
    let (slot_dn, ns_dn, _, sm_dn) = ring(g, rows, kmax, w.n_experts, false)?;
    let b = s.ptr.0;
    let mut a = MdArgs {
        ex: t.dev, sh: ts.map_or(0, |s| s.dev), ids: route.ids.0, wts: route.weights.0,
        x: x.ptr.0, xcode: act_code(x.dtype)?, ldx: x.ld as i32,
        xh1: b + lay.xh1 as u64, graw: b + lay.graw as u64, xh2: b + lay.xh2 as u64, h2: b + lay.h2 as u64,
        out: y.ptr.0, ldo: y.ld as i32, cnt: t.cnt,
        limit: w.swiglu_limit, limit_sh: sh.map_or(0.0, |s| s.swiglu_limit),
        n_experts: w.n_experts as i32, rows: rows as i32, topk: w.top_k as i32, dim: w.dim as i32, inter_ld: lay.inter_ld as i32,
        ncb: t.ncb as i32, kmax, kmin,
        slot: slot_gu as i32, ns: ns_gu as i32, wreg: wreg as i32, sms: sm_count(),
        chain: if with_chain { b + lay.chain as u64 } else { 0 },
        ow: overwrite as i32, pad: 0,
    };
    CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // PM1: the short-CTA structure (19_*) for the shapes it serves (measured dispatch; same bits)
    if super::moe_cp::try_run(l, t, ts, rows, w.top_k, (s.ptr.0, s.bytes), a)? { return Ok(()); }
    super::moe_cp::note_persistent(rows, t, kmin, kmax);
    if with_chain { CHAIN_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
    let pairs = rows * w.top_k + if ts.is_some() { rows } else { 0 };
    let mut pp = [&mut a as *mut MdArgs as *mut c_void];
    let kp = kernel("d41x_md_prep")?;
    launch_ex(&kp, ((w.dim / 128).div_ceil(4) as u32, pairs as u32, 1), 128, 0, l, &mut pp)?;
    let (kg, gg) = grid_for(gu, sm_gu as u32)?;
    launch_ex(&kg, (gg, 1, 1), 256, sm_gu as u32, l, &mut pp).context("decode MoE gate/up launch")?;
    a.slot = slot_dn as i32; a.ns = ns_dn as i32;
    let mut pp = [&mut a as *mut MdArgs as *mut c_void];
    let (kd, gd) = grid_for(dn, sm_dn as u32)?;
    launch_ex(&kd, (gd, 1, 1), 256, sm_dn as u32, l, &mut pp).context("decode MoE down launch")?;
    Ok(())
}
