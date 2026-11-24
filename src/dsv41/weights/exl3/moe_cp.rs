//! PM1: host side of the short-CTA decode MoE (kernels/gpu_dsv41/19_exl3_moe_cp.cuh): the external reference's TP2 decode
//! MoE structure (PLAN/DSV41_RX3_REPORT.md B.3: prep, gate/up CTAs of (slot, 128-column block, matrix, K part) with a ring per
//! warp, down CTAs of (slot, output block) waiting on their own slot's ready flag, the combine in down's epilogue, the shared
//! expert first) with OUR arithmetic: the output equals the persistent kernels of 15_* (moe_dec.rs) bit for bit at every row
//! count (the KE probe and tools/dsv41_pm1/pm_bench.cu gate it). `moe_dec::run_ow` calls `try_run` first; the shapes it serves
//! come from `serves` (a table measured in the harness: PLAN/DSV41_PM1_REPORT.md), `--dsv41-moe-cta` overrides it.

use super::launch::kernel;
use super::moe::MoeTable;
use super::moe_dec::{launch_ex, MdArgs, MAXS};
use crate::dsv41::weights::Launch;
use anyhow::{ensure, Context, Result};
use cudarc::driver::sys;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// 19_*'s `D41xMcArgs` (size checked against the device at first use).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct McArgs { pub a: MdArgs, pub route: u64, pub sk: i32, pub nsl: i32, pub nby: i32, pub nmt: i32 }

/// Gate/up K parts (each CTA streams dim / 128 / SK k-blocks; part 0 folds in registers, the others store chains). 5 measured
/// best at TP2 / TP4 one row (4: 504 CTAs, 5: 630, 8: 1,008; pm_bench, `.12`).
pub const SK: usize = 5;
/// Rows a call may have on this path (instances a8 / a16 / a32).
pub const MAX_ROWS: usize = 32;

struct Geo { args: usize, gu_ring_k: usize, dn_ring_k: usize, route_hdr: usize }
static GEO: OnceLock<Geo> = OnceLock::new();
/// Read the kernels' geometry once (a tiny kernel + sync: before any graph capture; `exl3::prepare` runs it).
pub fn geometry() -> Result<()> {
    if GEO.get().is_some() { return Ok(()); }
    let k = kernel("d41x_mc_sizes")?;
    let mut d: sys::CUdeviceptr = 0;
    let mut h = [0i32; 16];
    unsafe {
        ensure!(sys::cuMemAlloc_v2(&mut d, 64) == sys::CUresult::CUDA_SUCCESS, "cuMemAlloc (mc sizes)");
        let mut p = d as u64;
        let mut params = [&mut p as *mut u64 as *mut c_void];
        let mut st: sys::CUstream = std::ptr::null_mut();
        ensure!(sys::cuStreamCreate(&mut st, 0) == sys::CUresult::CUDA_SUCCESS, "cuStreamCreate (mc sizes)");
        let r = sys::cuLaunchKernel(k.0, 1, 1, 1, 32, 1, 1, 0, st, params.as_mut_ptr(), std::ptr::null_mut());
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "launch d41x_mc_sizes: {r:?}");
        ensure!(sys::cuStreamSynchronize(st) == sys::CUresult::CUDA_SUCCESS, "sync (mc sizes)");
        sys::cuStreamDestroy_v2(st);
        ensure!(sys::cuMemcpyDtoH_v2(h.as_mut_ptr() as *mut c_void, d, 64) == sys::CUresult::CUDA_SUCCESS, "read mc sizes");
        sys::cuMemFree_v2(d);
    }
    let g = Geo { args: h[0] as usize, gu_ring_k: h[1] as usize, dn_ring_k: h[2] as usize, route_hdr: h[3] as usize };
    ensure!(g.args == std::mem::size_of::<McArgs>(), "D41xMcArgs is {} B on the device, {} B in Rust (stale PTX or a layout change)", g.args, std::mem::size_of::<McArgs>());
    ensure!(g.gu_ring_k > 0 && g.dn_ring_k > 0 && g.route_hdr > 0, "d41x_mc_sizes: {h:?}");
    let _ = GEO.set(g);
    Ok(())
}
fn route_bytes(hdr: usize, e: usize) -> usize { (hdr + 6 * e + 127) / 128 * 128 }
fn a256(b: usize) -> usize { (b + 255) & !255 }
/// Persistent kernels' instance (a8 / a16 / a32); the short-CTA path (KM1) never needs a32: above 16 rows it runs the a16
/// instance on m tiles of 16 rows (`mtiles`), a slot with more rows getting one CTA set per tile.
fn ar_of(rows: usize) -> usize { if rows <= 8 { 8 } else if rows <= 16 { 16 } else { 32 } }
fn mc_ar_of(rows: usize) -> usize { ar_of(rows).min(16) }
fn mtiles(rows: usize) -> usize { rows.div_ceil(16).max(1) }

/// P7K1: per-(K, AR) dynamic smem of `d41x_mc_gu_a{ar}` and `d41x_mc_dn_a{ar}` (mirrors `d41x_mc_sum_bytes` / `d41x_mc_ring_bytes`).
fn mc_ring_bytes(k: i32, ar: usize, dn: bool) -> usize {
    let kps = if k == 2 && ar <= 16 { 8 } else { 4 };
    let stg = if k == 3 && ar == 16 { if dn { 4 } else { 5 } } else { 3 };
    4 * stg * kps * 2 * 32 * k as usize
}
fn smem_for(ar: usize, kmin: i32, kmax: i32) -> (usize, usize) {
    let sums = if ar <= 8 { 4 * 4 * 32 * 8 } else { 4 * (if ar > 16 { 8 } else { 4 }) * 32 * 16 };
    let gu_ring = (kmin..=kmax).map(|k| mc_ring_bytes(k, ar, false)).max().unwrap_or(0);
    let dn_ring = (kmin..=kmax).map(|k| mc_ring_bytes(k, ar, true)).max().unwrap_or(0);
    (sums + gu_ring, sums + dn_ring.max(ar * 132 * 4))
}

/// The A/B switch `--dsv41-moe-cta auto|short|persistent` (Spmd: rides TpConfig); the KE probe forces each structure in
/// turn (`force`: 0 = the option, 1 = auto, 2 = short, 3 = persistent; probe only).
static FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
pub fn force(m: u8) { FORCE.store(m, Ordering::Relaxed); }
fn mode() -> u8 {
    match FORCE.load(Ordering::Relaxed) { 1 => return 0, 2 => return 1, 3 => return 2, _ => {} }
    match crate::opts::get(crate::opt!("dsv41-moe-cta")).as_deref() { Some("short") => 1, Some("persistent") => 2, _ => 0 }
}

/// The measured dispatch (pm_bench on `.12`, graph + PDL, 384 experts, disjoint ids, router predecessor; old = 15_*'s
/// persistent kernels, new = 19_*): the short-CTA call is faster at 1..16 rows at every TP slice and width class measured
/// (K3 and K2 routed, shared K4 / K5 / mixed 4-5) except TP1 one row (-1.3 %) and K3 TP8 three rows (-2.6 %); above 16 rows (KM1: the a16 instance on m
/// tiles of 16 rows, `mtiles`; PM1's a32 instance lost 7..18 %) the call is faster than the persistent kernels at TP2 / TP4 / TP8
/// (+2..+69 % at 17..32 rows, PLAN/DSV41_KM1_REPORT.md) and slower at TP1 24 / 32 rows (kept persistent). `nblk` = the routed table's widest slice (18 / 9 /
/// 4-5 / 2-3 at TP1 / 2 / 4 / 8), `kmax_routed` its widest code.
pub fn serves_measured(rows: usize, nblk: usize, kmax_routed: i32) -> bool {
    if rows == 0 || rows > MAX_ROWS { return false; }
    if nblk >= 18 && (rows == 1 || rows > 16) { return false; }        // TP1: one row and 17+ rows measured slower (-1.3 % / -0.8..-4.9 %)
    if nblk <= 3 && rows == 3 && kmax_routed >= 3 { return false; }
    true
}
pub fn serves(rows: usize, t: &MoeTable) -> bool {
    if t.dim % (128 * SK) != 0 { return false; }          // the K parts need dim / 128 to split in SK (DSV4.1: 40 k-blocks)
    match mode() { 1 => rows >= 1 && rows <= MAX_ROWS, 2 => false, _ => serves_measured(rows, t.nblk_max, t.kmax()) }
}

/// Scratch layout of the short-CTA call: 15_*'s regions (xh1, graw, xh2, h2), the chain planes (always: K parts) and the
/// routing table. Returns (chain offset, route offset, total).
pub fn layout(t: &MoeTable, ts: Option<&MoeTable>, rows: usize, topk: usize, route_hdr: usize) -> (usize, usize, usize) {
    let p = rows * topk + if ts.is_some() { rows } else { 0 };
    let inter_ld = t.nblk_max.max(ts.map_or(0, |s| s.nblk_max)) * 128;
    let dim = t.dim;
    let chain = a256(p * 2 * dim * 2) + a256(p * 2 * inter_ld * 4) + a256(p * inter_ld * 2) + a256(p * dim * 4);
    let route = chain + a256((dim / 128) * p * 2 * inter_ld * 4);
    (chain, route, route + a256(route_bytes(route_hdr, t.n_experts)))
}
/// Scratch bytes this path needs for a call it would serve (0 when it would not serve the shape).
pub fn scratch_bytes(t: &MoeTable, ts: Option<&MoeTable>, rows: usize, topk: usize) -> usize {
    if !serves(rows, t) { return 0; }
    let hdr = GEO.get().map_or(4096, |g| g.route_hdr);    // before the geometry read: an upper bound of the header
    layout(t, ts, rows, topk, hdr).2
}

/// Engagement receipts: launches built per entry (a CUDA-graph replay re-runs captured launches without counting).
const ENTRIES: [&str; 6] = ["d41x_mc_gu_a8", "d41x_mc_gu_a16", "d41x_mc_gu_a32", "d41x_md_gu_a8", "d41x_md_gu_a16", "d41x_md_gu_a32"];
static CALLS: [AtomicU64; 6] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static SHORT_SCRATCH: AtomicU64 = AtomicU64::new(0);
static SHORT_SAID: AtomicBool = AtomicBool::new(false);
pub fn calls() -> u64 { CALLS[..3].iter().map(|c| c.load(Ordering::Relaxed)).sum() }
pub fn persistent_calls() -> u64 { CALLS[3..].iter().map(|c| c.load(Ordering::Relaxed)).sum() }
/// moe_dec counts its own (persistent) launches here too, so `/metrics` names the entry of every decode MoE launch.
pub fn count_persistent(rows: usize) { CALLS[3 + [8, 16, 32].iter().position(|&a| a == ar_of(rows)).unwrap()].fetch_add(1, Ordering::Relaxed); }

/// Once per shape (rows, routed slice width, width range, entry): which decode MoE structure serves it (debug log level, as
/// every technical boot line: OUT2; the `/metrics` counters below are always on).
fn say_shape(rows: usize, t: &MoeTable, kmin: i32, kmax: i32, short: bool) {
    static SEEN: OnceLock<Mutex<std::collections::HashSet<(usize, usize, i32, i32, bool)>>> = OnceLock::new();
    let m = SEEN.get_or_init(|| Mutex::new(Default::default()));
    if !m.lock().map(|mut s| s.insert((rows, t.nblk_max, kmin, kmax, short))).unwrap_or(false) { return; }
    if !crate::boot_report::debug() { return; }
    let ar = if short { mc_ar_of(rows) } else { ar_of(rows) };
    let what = if short { format!("d41x_mc_prep + d41x_mc_gu_a{ar} + d41x_mc_dn_a{ar} (short CTAs, ready flags, K parts {SK})") }
               else { format!("d41x_md_prep + d41x_md_gu_a{ar}_k* + d41x_md_dn_a{ar}_k* (persistent)") };
    crate::logq::send(false, format!("[dsv41] EXL3 decode MoE: {rows} row(s), routed slice {} blocks, K {kmin}..{kmax}: {what}", t.nblk_max));
}
pub fn note_persistent(rows: usize, t: &MoeTable, kmin: i32, kmax: i32) { count_persistent(rows); say_shape(rows, t, kmin, kmax, false); }

/// `/metrics` rows.
pub fn render(o: &mut String) {
    o.push_str("# HELP velogb10_dsv41_moe_dec_entry_calls_total Decode MoE launches built, by gate/up entry (d41x_mc_*: PM1's short-CTA structure; d41x_md_*: the persistent kernels)\n");
    o.push_str("# TYPE velogb10_dsv41_moe_dec_entry_calls_total counter\n");
    for (i, e) in ENTRIES.iter().enumerate() {
        o.push_str(&format!("velogb10_dsv41_moe_dec_entry_calls_total{{entry=\"{e}\"}} {}\n", CALLS[i].load(Ordering::Relaxed)));
    }
    o.push_str("# HELP velogb10_dsv41_moe_cp_short_scratch_total Decode MoE calls the short-CTA path would serve that ran the persistent kernels because the caller's scratch was short\n");
    o.push_str("# TYPE velogb10_dsv41_moe_cp_short_scratch_total counter\n");
    o.push_str(&format!("velogb10_dsv41_moe_cp_short_scratch_total {}\n", SHORT_SCRATCH.load(Ordering::Relaxed)));
}

/// Run the call on the short-CTA kernels when `serves` says so and the scratch holds the layout; `a` = moe_dec's filled
/// arguments (its ring fields are unused here). Ok(false): not served (the caller runs the persistent kernels).
#[allow(clippy::too_many_arguments)]
pub fn try_run(l: &Launch, t: &MoeTable, ts: Option<&MoeTable>, rows: usize, topk: usize, scratch: (u64, usize), mut a: MdArgs) -> Result<bool> {
    if !serves(rows, t) { return Ok(false); }
    geometry()?;
    let g = GEO.get().unwrap();
    let (chain, route, total) = layout(t, ts, rows, topk, g.route_hdr);
    if scratch.1 < total {
        SHORT_SCRATCH.fetch_add(1, Ordering::Relaxed);
        if !SHORT_SAID.swap(true, Ordering::Relaxed) {
            crate::boot_report::notice(&format!("dsv41 decode MoE: the caller's scratch ({} B) is short of the short-CTA layout ({total} B, {rows} row(s)): these calls run the persistent kernels (same bits); counted in velogb10_dsv41_moe_cp_short_scratch_total", scratch.1));
        }
        return Ok(false);
    }
    let kmax = a.kmax;
    ensure!((2..=8).contains(&a.kmin) && kmax <= 8, "short-CTA decode MoE: K {}..{kmax}", a.kmin);
    a.chain = scratch.0 + chain as u64;
    let n_ex = a.n_experts as usize;
    let nsl = (rows * topk + usize::from(ts.is_some())).min(MAXS);
    let nby = t.nblk_max.max(ts.map_or(0, |s| s.nblk_max));
    let kb = a.dim as usize / 128;
    let nmt = mtiles(rows);
    let max_vt = (nsl + if ts.is_some() && nmt > 1 { nmt - 1 } else { 0 }).min(MAXS + 1);
    let mut m = McArgs { a, route: scratch.0 + route as u64, sk: SK as i32, nsl: nsl as i32, nby: nby as i32, nmt: nmt as i32 };
    let ar = mc_ar_of(rows);
    let (gu_smem, dn_smem) = smem_for(ar, a.kmin, kmax);
    ensure!(gu_smem <= 48 * 1024 && dn_smem <= 48 * 1024, "short-CTA decode MoE: smem {gu_smem} / {dn_smem} B");
    let pairs = rows * topk + if ts.is_some() { rows } else { 0 };
    let prep_blocks = if rows <= 4 { 1 + (pairs * kb * 2).div_ceil(8) } else { 1 + (pairs * kb.div_ceil(8)).min(256) };
    let mut pp = [&mut m as *mut McArgs as *mut c_void];
    launch_ex(&kernel("d41x_mc_prep")?, (prep_blocks as u32, 1, 1), 256, route_bytes(g.route_hdr, n_ex) as u32, l, &mut pp).context("short-CTA MoE prep")?;
    let gu = format!("d41x_mc_gu_a{ar}");
    launch_ex(&kernel(&gu)?, ((max_vt * nby * 2 * SK) as u32, 1, 1), 128, gu_smem as u32, l, &mut pp).context("short-CTA MoE gate/up")?;
    launch_ex(&kernel(&format!("d41x_mc_dn_a{ar}"))?, ((max_vt * kb) as u32, 1, 1), 128, dn_smem as u32, l, &mut pp).context("short-CTA MoE down")?;
    CALLS[[8, 16, 32].iter().position(|&x| x == ar).unwrap()].fetch_add(1, Ordering::Relaxed);
    say_shape(rows, t, a.kmin, kmax, true);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dsv41_pm1_dispatch_table() {
        // TP2 / TP4 / TP8 (9 / 5 / 3 blocks), 1..16 rows: short CTAs, except K3 TP8 three rows; TP1 one row persistent
        for rows in 1..=16 { assert!(serves_measured(rows, 9, 3), "{rows}"); assert!(serves_measured(rows, 5, 3)); assert!(serves_measured(rows, 9, 2)); }
        assert!(!serves_measured(3, 3, 3) && serves_measured(3, 3, 2) && serves_measured(1, 3, 3) && serves_measured(6, 3, 3));
        assert!(!serves_measured(1, 18, 3) && serves_measured(2, 18, 3));
        for rows in [0, 33, 64] { assert!(!serves_measured(rows, 9, 3), "{rows}"); }
        for rows in [17, 24, 32] { assert!(serves_measured(rows, 9, 3) && serves_measured(rows, 5, 3) && !serves_measured(rows, 18, 3), "{rows}"); }
        assert_eq!(mtiles(16), 1); assert_eq!(mtiles(17), 2); assert_eq!(mtiles(32), 2); assert_eq!(mc_ar_of(32), 16); assert_eq!(mc_ar_of(5), 8);
        assert_eq!(ar_of(1), 8); assert_eq!(ar_of(9), 16); assert_eq!(ar_of(17), 32);
        // a table whose dim does not split into SK K parts of 128-blocks is never served here (the persistent kernels run it)
        let t = MoeTable { dev: 0, n_experts: 8, nblk_max: 2, k13: (3, 3), k2: (3, 3), dim: 512, inter: 256, cnt: 0, ncb: 2 };
        assert!(!serves(1, &t));
    }
    #[test]
    fn dsv41_pm1_layout_regions() {
        let t = MoeTable { dev: 0, n_experts: 384, nblk_max: 9, k13: (3, 3), k2: (3, 3), dim: 5120, inter: 2304, cnt: 0, ncb: 18 };
        let ts = MoeTable { n_experts: 1, k13: (4, 4), k2: (4, 4), ..t };
        let (c, r, tot) = layout(&t, Some(&ts), 1, 6, 3000);
        // chain = after xh1 [7][2][5120] f16, graw [7][2][1152] f32, xh2 [7][1152] f16, h2 [7][5120] f32; chain planes [40][7][2][1152] f32
        assert_eq!(c, a256(7 * 2 * 5120 * 2) + a256(7 * 2 * 1152 * 4) + a256(7 * 1152 * 2) + a256(7 * 5120 * 4));
        assert_eq!(r, c + 40 * 7 * 2 * 1152 * 4);
        assert_eq!(tot, r + a256(route_bytes(3000, 384)));
        assert_eq!(std::mem::size_of::<McArgs>(), std::mem::size_of::<MdArgs>() + 24);
        // P7K1: every served (AR = 8 / 16, K = 2..5) instance fits 4 CTAs/SM on GB10 (<= 24,064 B dynamic smem)
        for ar in [8usize, 16] {
            for (kmin, kmax) in [(2, 4), (2, 5), (3, 4), (3, 5), (2, 2), (3, 3)] {
                let (gu, dn) = smem_for(ar, kmin, kmax);
                assert!(gu <= 24_064 && dn <= 24_064, "ar {ar} K{kmin}..{kmax}: gu {gu} dn {dn}");
            }
        }
    }
}
