//! MPK persistent MoE kernel family: smem sizing, entry tables, `MpkPlan`, the gate/up and down launches,
//! and the W4MOE option parsing, announce and prewarm. See `mod.rs` for the file map.

use super::*;

// ---- W4/MOE (decode wave 4): persistent grouped-expert kernels (kernels/exl3_bench.cu "W4/MOE") ----
// The LMH byte schedule (persistent grid, cp.async ring + evict_first, A staged per expert run)
// under the WP20 rung-2 epilogues, and the gate/up input transform folded in. Every knob is a
// schedule knob of a bitwise package (the kernel contract).
// p5e DEFAULT = OFF (the p3 WP20 rung-2 launches: xq_had_suh_multi + xq_moe_gu_epi +
// xq_moe_dn_epi_k<topk>). Bitwise, but SLOWER IN-ROUND (p5c on .13, 2026-09-27 03:17-05:02 UTC,
// hashes exact, every XCHECK zero): +2.4..+4.5 ms/round on every class; nsys per call
// xq_moe_gu_pk_s8 361.7 vs xq_moe_gu_epi 295.2 us, xq_moe_dn_pk 182.0 vs 150.6 us.
//   (unset)                     OFF (default): the WP20 rung-2 launches exactly, as in p3
//   --w4moe=1                opt in (any value but ""/0/off): every part (gu, dn, fold)
//   --w4moe-off=1|all        escape, unchanged; wins over --w4moe=1
//   --w4moe-off=<parts>      with --w4moe=1: comma list gu (gate/up keeps xq_moe_gu_epi),
//                               dn (down keeps xq_moe_dn_epi_k<topk>), fold (persistent gate/up
//                               reads xh_e from xq_had_suh_multi instead of forming A in-kernel)
//   --w4moe-order=il         interleaved item order (the LMH order, A per item); default
//                               contiguous item ranges (A per expert run)
//   --w4moe-nst=4|8          ring depth (default: the occupancy rule in mpk_plan)
//   --w4moe-g=<n>            persistent grid for both kernels (clamped to [1, items cap])
//   --w4moe-minm=<m>         persistent kernels only at widths >= m (default 1)
//   --w4moe-shovl=1          lever (c), opt-in: shared expert on a WP27 sibling branch
// The ORDER / NST / G / MINM / SHOVL knobs act only while the package is ON (the OFF boot line
// lists any that are set as ignored).
// XCHECK: --exl3-moe-epi-check=1 (eager) diffs this path against the pre-WP20 chain (ygu/xhd/
// yd/moe_out bitwise + counters re-armed). Probe: --probe-exl3-binv EXL3-MOE-EPI (W4/MOE rows).
pub(crate) const MPK_GU: u8 = 1;
pub(crate) const MPK_DN: u8 = 2;
pub(crate) const MPK_FOLD: u8 = 4;
/// Kernel flag bits (mirror MPK_IL / MPK_FOLD in kernels/exl3_bench.cu).
pub(crate) const MPK_FLAG_IL: i32 = 1;
pub(crate) const MPK_FLAG_FOLD: i32 = 2;
/// Mirror of MPK_STB (ring slot bytes) / MPK_EMAX (live-expert table cap) in exl3_bench.cu.
const MPK_STB: usize = 4 * 768;
pub(crate) const MPK_EMAX: usize = 256;
/// Mirror of MPK_RTMAX: the down kernel's combine tables hold m * topk entries.
const MPK_RTMAX: usize = 256;

/// --w4moe opt-in (p5e: the package is DEFAULT OFF): set to anything but "", "0" or "off".
fn w4moe_opt_in() -> bool {
    crate::opts::var(crate::opt!("w4moe")).is_ok_and(|v| !matches!(v.trim(), "" | "0" | "off" | "OFF"))
}
/// Live W4/MOE parts (MPK_GU | MPK_DN | MPK_FOLD). p5e: 0 (the WP20 rung-2 kernels) unless
/// --w4moe=1; then --w4moe-off (1|all|<parts>) applied on top, as before.
/// TUNE T0c (p5e): the parts mask is the registry entry w4moe.parts (per width; every mask bitwise),
/// so a full tune re-evaluates the package in-round per (family, m); the aliases decide exactly as
/// before whenever one is set (w4moe_parts_env).
pub(crate) fn w4moe_parts() -> u8 {
    let p = tune::get_cur(&T_W4MOE_PARTS).clamp(0, 7) as u8;
    if p & MPK_GU == 0 { p & !MPK_FOLD } else { p }
}
/// The aliases' parts mask (the pre-registry parse, verbatim; its warnings once); None when
/// neither --w4moe nor --w4moe-off is set.
fn w4moe_parts_env() -> Option<u8> {
    static V: std::sync::OnceLock<Option<u8>> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let (opt, off) = (crate::opts::var(crate::opt!("w4moe")).ok(), crate::opts::var(crate::opt!("w4moe-off")).ok());
        if opt.is_none() && off.is_none() { return None; }
        Some(w4moe_parts_of(opt.as_deref(), off.as_deref()))
    })
}
/// The --w4moe / --w4moe-off values -> the live parts mask (the pre-registry w4moe_parts
/// body, verbatim: w4moe_opt_in's test, then the escape's parts on top, fold only with gu).
pub(crate) fn w4moe_parts_of(opt_in: Option<&str>, off: Option<&str>) -> u8 {
    let mut p = if opt_in.is_some_and(|v| !matches!(v.trim(), "" | "0" | "off" | "OFF")) {
        MPK_GU | MPK_DN | MPK_FOLD
    } else { 0 };
    if let Some(v) = off {
        for t in v.split(',').map(|t| t.trim().to_ascii_lowercase()) {
            match t.as_str() {
                "1" | "all" => p = 0,
                "gu" => p &= !MPK_GU,
                "dn" => p &= !MPK_DN,
                "fold" => p &= !MPK_FOLD,
                "" | "0" => {}
                other => crate::dprintln!("W4/MOE: unknown --w4moe-off part '{other}' ignored"),
            }
        }
    }
    if p & MPK_GU == 0 { p &= !MPK_FOLD; }
    p
}
/// W4/MOE lever (c): --w4moe-shovl=1 = the shared expert on a WP27 sibling branch of the routed
/// gate/up path in captured graphs (opt-in; see moe_wp20).
/// --w4moe-minm=<m>: the persistent kernels only at widths >= m (narrower widths — e.g. the
/// m = 1 draft pass — keep the WP20 rung-2 kernels). Default 1 (every width). A/B + bisect knob
/// (TUNE: env-only — the per-width w4moe.parts subsumes it).
pub(super) fn w4moe_minm() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::var(crate::opt!("w4moe-minm")).ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(1).max(1))
}
pub(super) fn w4moe_shovl() -> bool {
    tune::get_cur(&T_W4MOE_SHOVL) != 0
}
/// A5-K1 (w6/SHOVL): the shared expert on a WP27 sibling branch of the routed path on the DEFAULT
/// WP20 rung-2 MoE path (xq_router_fold[_w4] -> xq_had_suh_multi -> xq_moe_gu_epi ‖ shared expert,
/// joined before xq_moe_dn_epi_k<topk>) — the W4/MOE (c) placement without the persistent package.
/// Registry entry moe.shovl (S class, per width); alias --moe-shovl=0 = escape (linear MoE window,
/// the pre-K1 graphs exactly), any other value = on. DEFAULT ON since the A5-K1 in-round A/B on .14
/// (6 boots interleaved, 12/12 pairs faster: -0.87..-1.23 ms/round per class, trajectories and
/// served hashes identical) — PLAN/notes_2026-09-27/K1_SHOVL.md.
pub(super) fn moe_shovl() -> bool {
    tune::get_cur(&T_MOE_SHOVL) != 0
}
pub(super) fn w4moe_order_il() -> bool {
    tune::get_cur(&T_W4MOE_ORDER) != 0
}
/// The ring-depth pin (None = mpk_plan's occupancy rule): w4moe.nst 4 | 8, 0 = the rule.
fn w4moe_nst_env() -> Option<usize> {
    match tune::get_cur(&T_W4MOE_NST) { 4 => Some(4), 8 => Some(8), _ => None }
}
/// The persistent grid pin (None = the rule): w4moe.g > 0.
fn w4moe_g_env() -> Option<u32> {
    let g = tune::get_cur(&T_W4MOE_G);
    (g > 0).then_some(g as u32)
}
// TUNE T0c (p5e): the W4/MOE registry entries (class S, per width; every value bitwise by the kernel
// contract). --w4moe-minm stays an env-only diagnostic.
const W4MOE_XCHECK: &str = "--probe-exl3-binv EXL3-MOE-EPI (W4/MOE rows) + --exl3-moe-epi-check";
pub(crate) const T_W4MOE_PARTS: tune::TunableDef = tune::TunableDef {
    slot: 72, id: "w4moe.parts", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    // MPK_GU 1 | MPK_DN 2 | MPK_FOLD 4 (fold needs gu): every reachable mask
    domain: &[0, 1, 2, 3, 5, 7], default: 0, valid: tune::valid_any, xcheck: W4MOE_XCHECK,
    env: "--w4moe|--w4moe-off", env_parse: || w4moe_parts_env().map(|p| p as i32), rev: 1, wp: "W4/MOE",
};
pub(crate) const T_W4MOE_ORDER: tune::TunableDef = tune::TunableDef {
    slot: 73, id: "w4moe.order", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 1], default: 0, valid: tune::valid_any, xcheck: W4MOE_XCHECK,
    env: "--w4moe-order",
    env_parse: || tune::env_str(crate::opt!("w4moe-order")).map(|v| matches!(v.as_str(), "il" | "IL" | "1") as i32),
    rev: 1, wp: "W4/MOE",
};
pub(crate) const T_W4MOE_NST: tune::TunableDef = tune::TunableDef {
    slot: 74, id: "w4moe.nst", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 4, 8], default: 0, valid: tune::valid_any, xcheck: W4MOE_XCHECK,
    env: "--w4moe-nst",
    env_parse: || tune::env_str(crate::opt!("w4moe-nst")).map(|v| match v.as_str() { "4" => 4, "8" => 8, _ => 0 }),
    rev: 1, wp: "W4/MOE",
};
pub(crate) const T_W4MOE_G: tune::TunableDef = tune::TunableDef {
    slot: 75, id: "w4moe.g", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 48, 96, 144, 192], default: 0, valid: tune::valid_any, xcheck: W4MOE_XCHECK,
    env: "--w4moe-g",
    env_parse: || tune::env_str(crate::opt!("w4moe-g")).map(|v| v.trim().parse::<u32>().ok().filter(|&g| g > 0)
        .map_or(0, |g| g.min(i32::MAX as u32) as i32)),
    rev: 1, wp: "W4/MOE",
};
pub(crate) const T_W4MOE_SHOVL: tune::TunableDef = tune::TunableDef {
    slot: 76, id: "w4moe.shovl", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 1], default: 0, valid: tune::valid_any, xcheck: W4MOE_XCHECK,
    env: "--w4moe-shovl", env_parse: || tune::env_on_if_1(crate::opt!("w4moe-shovl")), rev: 1, wp: "W4/MOE",
};
/// A5-K1: shared-expert overlap on the default WP20 rung-2 path (see moe_shovl). Same kernels, same
/// inputs, same launch shapes; only the captured dependency edges differ (bitwise class S).
pub(crate) const T_MOE_SHOVL: tune::TunableDef = tune::TunableDef {
    slot: 85, id: "moe.shovl", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 1], default: 1, valid: tune::valid_any,
    xcheck: "WP27 capture-DAG audit (shared expert reads sc.x only; private last-arrival counters) + --probe-exl3-binv",
    env: "--moe-shovl", env_parse: || tune::env_on_unless_0(crate::opt!("moe-shovl")), rev: 1, wp: "A5-K1",
};

/// A5-K7 (ledger #1(a)): xq_had_suh_multi folded into the DEFAULT WP20 rung-2 gate/up prologue —
/// xq_moe_gu_epi_f forms A = f16(H128(x * suh_e)) in-kernel (the pinned, boot-verified sequence;
/// same bytes as xq_had_suh_multi -> xq_moe_gu_epi) and the separate launch goes. Diet body only
/// (--wp20-diet=0 keeps the unfolded pair); inert under a W4/MOE persistent gate/up. Class S
/// (bitwise; --probe-exl3-binv EXL3-MOE-EPI "fold" rows + --exl3-moe-epi-check=1 eager xcheck).
/// VALUE = the widest MoE call that folds: at width m the fold runs iff m <= value (0 = off). v1
/// (fold at every width) measured the fold's cost growing with m (each of an expert's 10 CTAs forms
/// all m x 20 transforms) against a ~constant saved launch: fitted verify delta -1.12 + 0.167 m
/// ms/verify, crossover m ~ 6.7 — so v2's candidate is 6 (draft m = 1 and the narrow verifies fold,
/// m = 7..8 keep the separate launch). Alias --moe-gu-fold=<max width> (0..16; any width-scoped
/// table value works the same way per width). Default 0 (OFF) until integrated
/// (PLAN/notes_2026-09-27/K7.md).
pub(crate) const T_MOE_GU_FOLD: tune::TunableDef = tune::TunableDef {
    slot: 90, id: "moe.gu_fold", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 1, 4, 5, 6, 7, 9], default: 6, valid: tune::valid_any,
    xcheck: "--probe-exl3-binv EXL3-MOE-EPI (fold rows) + --exl3-moe-epi-check=1 (eager)",
    env: "--moe-gu-fold", env_parse: || tune::env_str(crate::opt!("moe-gu-fold"))
        .map(|v| v.trim().parse::<i32>().unwrap_or(0).clamp(0, 16)), rev: 1, wp: "A5-K7",
};
/// A5-K7: does the WP20 gate/up at width m fold xq_had_suh_multi (moe.gu_fold = max folded width)?
pub(super) fn moe_gu_fold(l: &Launcher, m: usize) -> bool {
    let v = tune::get(&T_MOE_GU_FOLD, &l.cx());
    v > 0 && m <= v as usize
}
/// The folded gate/up entry (A5-K7).
pub(crate) const WP20_GU_FOLD_FN: &str = "xq_moe_gu_epi_f";

/// Mirror of mpk_smem_bytes() in kernels/exl3_bench.cu (the kernel traps if launched short).
/// B10: width-keyed smem (ring bytes scale with the trellis width; MpkGeom<BITS>::STB =
/// 1024*bits — bits=3 reproduces mpk_smem_bytes exactly). Mirrors the CUDA mpk_smem_bytes_b.
pub(crate) fn mpk_smem_bytes_b(nst: usize, m: usize, k: usize, gu: bool, fold: bool, bits: usize) -> usize {
    let epi = if gu { (m * 256).max(4096) } else { m * 256 };
    nst * (1024 * bits) + m * (k + 8) * 2 + if gu && fold { k * 2 } else { 0 } + epi
}

pub(crate) fn mpk_smem_bytes(nst: usize, m: usize, k: usize, gu: bool, fold: bool) -> usize {
    let epi = if gu { (m * 256).max(4096) } else { m * 256 };
    nst * MPK_STB + m * (k + 8) * 2 + if gu && fold { k * 2 } else { 0 } + epi
}

/// B10: width-keyed entry table — bits 2,4,5,6,7 (generic SHFL bodies, 256 threads at
/// bits<=3 else 512); K=3 keeps mpk_entry's diet entries byte-identical. None = not compiled.
pub(crate) fn mpk_entry_b(gu: bool, topk: usize, nst: usize, bits: usize) -> Option<&'static str> {
    if bits == 3 { return mpk_entry(gu, topk, nst); }
    Some(match (gu, topk, nst, bits) {
        (true, _, 4, 2) => "xq_moe_gu_pk_b2_s4",
        (true, _, 4, 4) => "xq_moe_gu_pk_b4_s4",
        (true, _, 4, 5) => "xq_moe_gu_pk_b5_s4",
        (true, _, 4, 6) => "xq_moe_gu_pk_b6_s4",
        (true, _, 4, 7) => "xq_moe_gu_pk_b7_s4",
        (true, _, 8, 2) => "xq_moe_gu_pk_b2_s8",
        (true, _, 8, 4) => "xq_moe_gu_pk_b4_s8",
        (true, _, 8, 5) => "xq_moe_gu_pk_b5_s8",
        (true, _, 8, 6) => "xq_moe_gu_pk_b6_s8",
        (true, _, 8, 7) => "xq_moe_gu_pk_b7_s8",
        (false, 8, 4, 2) => "xq_moe_dn_pk_b2_k8_s4",
        (false, 8, 4, 4) => "xq_moe_dn_pk_b4_k8_s4",
        (false, 8, 4, 5) => "xq_moe_dn_pk_b5_k8_s4",
        (false, 8, 4, 6) => "xq_moe_dn_pk_b6_k8_s4",
        (false, 8, 4, 7) => "xq_moe_dn_pk_b7_k8_s4",
        (false, 8, 8, 2) => "xq_moe_dn_pk_b2_k8_s8",
        (false, 8, 8, 4) => "xq_moe_dn_pk_b4_k8_s8",
        (false, 8, 8, 5) => "xq_moe_dn_pk_b5_k8_s8",
        (false, 8, 8, 6) => "xq_moe_dn_pk_b6_k8_s8",
        (false, 8, 8, 7) => "xq_moe_dn_pk_b7_k8_s8",
        (false, 10, 4, 2) => "xq_moe_dn_pk_b2_k10_s4",
        (false, 10, 4, 4) => "xq_moe_dn_pk_b4_k10_s4",
        (false, 10, 4, 5) => "xq_moe_dn_pk_b5_k10_s4",
        (false, 10, 4, 6) => "xq_moe_dn_pk_b6_k10_s4",
        (false, 10, 4, 7) => "xq_moe_dn_pk_b7_k10_s4",
        (false, 10, 8, 2) => "xq_moe_dn_pk_b2_k10_s8",
        (false, 10, 8, 4) => "xq_moe_dn_pk_b4_k10_s8",
        (false, 10, 8, 5) => "xq_moe_dn_pk_b5_k10_s8",
        (false, 10, 8, 6) => "xq_moe_dn_pk_b6_k10_s8",
        (false, 10, 8, 7) => "xq_moe_dn_pk_b7_k10_s8",
        _ => return None,
    })
}

/// Entry name for (side, top-k, ring depth); None = not compiled.
pub(crate) fn mpk_entry(gu: bool, topk: usize, nst: usize) -> Option<&'static str> {
    Some(match (gu, topk, nst) {
        (true, _, 4) => "xq_moe_gu_pk_s4",
        (true, _, 8) => "xq_moe_gu_pk_s8",
        (false, 8, 4) => "xq_moe_dn_pk_k8_s4",
        (false, 8, 8) => "xq_moe_dn_pk_k8_s8",
        (false, 10, 4) => "xq_moe_dn_pk_k10_s4",
        (false, 10, 8) => "xq_moe_dn_pk_k10_s8",
        _ => return None,
    })
}

/// Raw handle of one W4/MOE entry opted in to `smem` dynamic bytes, with its co-resident CTAs
/// per SM at that smem (occupancy API, 256 threads). Cached per (entry, smem); None = unusable
/// (stale PTX, over the opt-in cap, or zero occupancy) — the caller keeps the WP20 kernel.
pub(crate) fn mpk_fn(name: &'static str, smem: u32, threads: u32) -> Option<(cudarc::driver::sys::CUfunction, u32)> {
    use cudarc::driver::sys;
    #[derive(Clone, Copy)]
    struct FnH(sys::CUfunction);
    unsafe impl Send for FnH {}
    unsafe impl Sync for FnH {}
    static C: std::sync::Mutex<Vec<(&'static str, u32, FnH, u32, u32)>> = std::sync::Mutex::new(Vec::new());
    let mut c = C.lock().ok()?;
    if let Some(e) = c.iter().find(|e| e.0 == name && e.1 == smem) { return Some((e.2 .0, e.3)); }
    // B10: failed opt-ins are remembered (name, smem) — the ring fallback re-plans per request and
    // would otherwise retry + re-log the same failure on every planning call.
    static FAILS: std::sync::Mutex<Vec<(&'static str, u32)>> = std::sync::Mutex::new(Vec::new());
    if FAILS.lock().unwrap().iter().any(|e| e.0 == name && e.1 == smem) { return None; }
    let f = xq_raw_fn(name)?;
    if smem as usize > crate::gpu::SMEM_OPTIN_CAP as usize { return None; }
    // the opt-in only ever grows (a later smaller request must not lower it below a cached one)
    let optin = c.iter().filter(|e| e.0 == name).map(|e| e.4).max().unwrap_or(0).max(smem);
    let r = unsafe { sys::cuFuncSetAttribute(f,
        sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, optin as i32) };
    if r != sys::CUresult::CUDA_SUCCESS {
        let mut fl = FAILS.lock().unwrap();
        if !fl.iter().any(|e| e.0 == name && e.1 == smem) {
            fl.push((name, smem));
            eprintln!("[w4moe] {name}: smem opt-in {optin} B failed ({r:?}) — ring falls back (once per entry)");
        }
        return None;
    }
    let mut nb: i32 = 0;
    let r = unsafe { sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut nb, f, threads as i32, smem as usize) };
    if r != sys::CUresult::CUDA_SUCCESS || nb < 1 {
        crate::deprintln!("[w4moe] {name}: not resident at {smem} B smem threads {threads} (occupancy {nb}, {r:?})");
        return None;
    }
    for e in c.iter_mut().filter(|e| e.0 == name) { e.4 = optin; }
    c.push((name, smem, FnH(f), nb as u32, optin));
    Some((f, nb as u32))
}

/// One W4/MOE launch plan: entry, grid, dynamic smem, CTAs/SM, ring depth.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MpkPlan {
    pub thr: u32,   // B10: block threads (256 at bits<=3, 512 above)
    pub name: &'static str,
    pub grid: u32,
    pub smem: u32,
    pub cps: u32,
    pub nst: usize,
}

/// Plan one side (gate/up or down) at width m: items cap = min(emax, m*topk) * N/128 (the live
/// expert count is device-side; m*topk bounds it). Rule: the ring depth giving the most
/// co-resident CTAs (occupancy x SMs, capped at the items cap), ties -> the deeper ring; env
/// --w4moe-nst / --w4moe-g override. None = not launchable (the caller keeps WP20).
pub(crate) fn mpk_plan(dev: &Arc<CudaDevice>, gu: bool, topk: usize, m: usize, k: usize, n: usize,
                       emax: usize, fold: bool, bits: usize) -> Option<MpkPlan> {
    if m == 0 || m > 16 || n % 128 != 0 || k % 128 != 0 || (gu && n % 256 != 0) { return None; }
    if (!gu && m * topk > MPK_RTMAX) || emax.min(m * topk) > MPK_EMAX { return None; }
    let cap = (emax.min(m * topk).max(1) * (n / 128)) as u32;
    let sms = lmh_sm_count(dev);
    let nsts: &[usize] = match w4moe_nst_env() { Some(4) => &[4], Some(8) => &[8], _ => &[8, 4] };
    let mut best: Option<MpkPlan> = None;
    for &nst in nsts {
        let Some(name) = mpk_entry_b(gu, topk, nst, bits) else { continue };
        let smem = mpk_smem_bytes_b(nst, m, k, gu, fold, bits) as u32;
        let thr = if bits <= 3 { 256 } else { 512 };   // MpkGeom<BITS>::THREADS
        let Some((_, cps)) = mpk_fn(name, smem, thr) else { continue };
        let grid = (cps * sms).min(cap).max(1);
        if best.map_or(true, |b| grid > b.grid) { best = Some(MpkPlan { name, grid, smem, cps, nst, thr }); }
    }
    let mut p = best?;
    if let Some(g) = w4moe_g_env() { p.grid = g.min(cap).max(1); }
    Some(p)
}

/// W4/MOE gate/up launch (the SERVED launcher; also the binv probe's). ain = x (fold) or xh_e;
/// every pointer is a device address. Plain stream launch (graph-capturable), PDL-aware.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mpk_launch_gu(dev: &Arc<CudaDevice>, stream: cudarc::driver::sys::CUstream, plan: &MpkPlan,
                            base: u64, offs: u64, ain: u64, suh_gu: u64, ygu: u64, m: usize, k: usize, n: usize,
                            esel: u64, idxmap: u64, svh: u64, suh_d: u64, xhd: u64, cnt: u64, x: u64, sg: u64,
                            sgv: u64, flags: i32) -> Result<()> {
    let _ = dev;
    let mut a = [base, offs, ain, suh_gu, ygu];
    let mut d = [m as i32, k as i32, n as i32];
    let mut b = [esel, idxmap, svh, suh_d, xhd, cnt, x, sg, sgv];
    let mut fl = flags;
    let mut p: [*mut std::ffi::c_void; 18] = [
        &mut a[0] as *mut u64 as *mut _, &mut a[1] as *mut u64 as *mut _, &mut a[2] as *mut u64 as *mut _,
        &mut a[3] as *mut u64 as *mut _, &mut a[4] as *mut u64 as *mut _,
        &mut d[0] as *mut i32 as *mut _, &mut d[1] as *mut i32 as *mut _, &mut d[2] as *mut i32 as *mut _,
        &mut b[0] as *mut u64 as *mut _, &mut b[1] as *mut u64 as *mut _, &mut b[2] as *mut u64 as *mut _,
        &mut b[3] as *mut u64 as *mut _, &mut b[4] as *mut u64 as *mut _, &mut b[5] as *mut u64 as *mut _,
        &mut b[6] as *mut u64 as *mut _, &mut b[7] as *mut u64 as *mut _, &mut b[8] as *mut u64 as *mut _,
        &mut fl as *mut i32 as *mut _,
    ];
    mpk_raw_launch(stream, plan, &mut p)
}

/// W4/MOE down launch (the SERVED launcher; also the binv probe's).
#[allow(clippy::too_many_arguments)]
pub(crate) fn mpk_launch_dn(dev: &Arc<CudaDevice>, stream: cudarc::driver::sys::CUstream, plan: &MpkPlan,
                            base: u64, offs: u64, xhd: u64, yd: u64, m: usize, k: usize, n: usize,
                            esel: u64, idxmap: u64, svh: u64, cnt_dn: u64, ysh: u64, ids: u64, wts: u64,
                            slotmap: u64, sgv: u64, out: u64, topk: usize, flags: i32) -> Result<()> {
    let _ = dev;
    let mut a = [base, offs, xhd, yd];
    let mut d = [m as i32, k as i32, n as i32];
    let mut b = [esel, idxmap, svh, cnt_dn, ysh, ids, wts, slotmap, sgv, out];
    let (mut tk, mut fl) = (topk as i32, flags);
    let mut p: [*mut std::ffi::c_void; 19] = [
        &mut a[0] as *mut u64 as *mut _, &mut a[1] as *mut u64 as *mut _, &mut a[2] as *mut u64 as *mut _,
        &mut a[3] as *mut u64 as *mut _,
        &mut d[0] as *mut i32 as *mut _, &mut d[1] as *mut i32 as *mut _, &mut d[2] as *mut i32 as *mut _,
        &mut b[0] as *mut u64 as *mut _, &mut b[1] as *mut u64 as *mut _, &mut b[2] as *mut u64 as *mut _,
        &mut b[3] as *mut u64 as *mut _, &mut b[4] as *mut u64 as *mut _, &mut b[5] as *mut u64 as *mut _,
        &mut b[6] as *mut u64 as *mut _, &mut b[7] as *mut u64 as *mut _, &mut b[8] as *mut u64 as *mut _,
        &mut b[9] as *mut u64 as *mut _, &mut tk as *mut i32 as *mut _, &mut fl as *mut i32 as *mut _,
    ];
    mpk_raw_launch(stream, plan, &mut p)
}

fn mpk_raw_launch(stream: cudarc::driver::sys::CUstream, plan: &MpkPlan, p: &mut [*mut std::ffi::c_void]) -> Result<()> {
    use cudarc::driver::sys;
    let (f, _) = mpk_fn(plan.name, plan.smem, plan.thr)
        .ok_or_else(|| anyhow::anyhow!("W4/MOE: {} unavailable at {} B smem (stale PTX?)", plan.name, plan.smem))?;
    if pdl_on() {
        return unsafe { launch_pss_raw(stream, f, (plan.grid, 1, 1), (plan.thr, 1, 1), plan.smem, p) }
            .with_context(|| format!("launch {} (pdl)", plan.name));
    }
    let r = unsafe { sys::cuLaunchKernel(f, plan.grid, 1, 1, plan.thr, 1, 1, plan.smem, stream, p.as_mut_ptr(), std::ptr::null_mut()) };
    anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "launch {} ({r:?})", plan.name);
    Ok(())
}

/// FwdModel::load: resolve + opt in every W4/MOE entry the widths 1..=width can take (outside any
/// graph capture), and print the feature marker with the per-width plan.
pub(super) fn w4moe_announce(dev: &Arc<CudaDevice>, cfg: &crate::qwen::Config, width: usize) {
    if cfg.num_experts == 0 || wp20_mode() != 2 { return; }
    let parts = w4moe_parts();
    if parts & (MPK_GU | MPK_DN) == 0 {
        let esc = crate::opts::var(crate::opt!("w4moe-off")).ok().filter(|v| !matches!(v.trim(), "" | "0"));
        let why = match (w4moe_opt_in(), esc) {
            (false, None) => "default".to_string(),
            (false, Some(v)) => format!("--w4moe-off={v}"),
            (true, Some(v)) => format!("--w4moe-off={v} overrides --w4moe=1"),
            (true, None) => "no live part".to_string(),
        };
        let knobs: Vec<String> = [crate::opt!("w4moe-order"), crate::opt!("w4moe-nst"), crate::opt!("w4moe-g"), crate::opt!("w4moe-minm"), crate::opt!("w4moe-shovl")]
            .into_iter().filter(|k| crate::opts::var_os(*k).is_some()).map(|k| k.to_string()).collect();
        crate::dprintln!("W4/MOE: OFF ({why}) — WP20 rung-2 kernels (xq_had_suh_multi + xq_moe_gu_epi + xq_moe_dn_epi); \
                  --w4moe=1 = persistent grouped experts{}",
                 if knobs.is_empty() { String::new() } else { format!(" ({} ignored while OFF)", knobs.join("/")) });
        w4moe_prewarm(dev, cfg, width, parts);
        return;
    }
    let (h, mi, topk) = (cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts_per_tok);
    let fold = parts & MPK_FOLD != 0;
    let mut rows = Vec::new();
    for m in 1..=width.clamp(1, 16) {
        if a1_smem(m, h) == 0 || a1_smem(m, mi) == 0 { continue; } // WP20 rung 2's envelope
        let emax = width * topk;
        let on = m >= w4moe_minm();
        // K=3 rows only (the boot log; per-layer widths plan lazily at serve — B10)
        let gu = if on && parts & MPK_GU != 0 { mpk_plan(dev, true, topk, m, h, 2 * mi, emax, fold, 3) } else { None };
        let dn = if on && parts & MPK_DN != 0 { mpk_plan(dev, false, topk, m, mi, h, emax, false, 3) } else { None };
        let f = |p: Option<MpkPlan>| p.map_or("WP20".to_string(), |p| format!("s{}xG{}({}/SM)", p.nst, p.grid, p.cps));
        rows.push(format!("m{m} gu {} dn {}", f(gu), f(dn)));
    }
    eprintln!("W4/MOE: persistent grouped experts ON (opt-in --w4moe=1) — parts gu {} dn {} fold {}, order {}, \
              shared-expert overlap {}, top-k {topk}; plans [{}] (default OFF / --w4moe-off=1 = WP20 rung-2 kernels)",
             (parts & MPK_GU != 0) as u8, (parts & MPK_DN != 0) as u8, fold as u8,
             if w4moe_order_il() { "interleaved" } else { "contiguous" },
             if w4moe_shovl() { "ON (--w4moe-shovl=1)" } else { "off" }, rows.join(", "));
    w4moe_prewarm(dev, cfg, width, parts);
}

/// TUNE T0c: plan (and so opt in) every W4/MOE entry the OTHER reachable parts masks can take — a
/// table value or the --autotune overlay can turn the package on (or change the fold) at some width —
/// here, outside any graph capture, silently. Nothing reachable beyond `planned`: a no-op (the
/// default posture plans exactly what it did before).
fn w4moe_prewarm(dev: &Arc<CudaDevice>, cfg: &crate::qwen::Config, width: usize, planned: u8) {
    let (h, mi, topk) = (cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts_per_tok);
    for p in tune::reachable(&T_W4MOE_PARTS) {
        let p = { let p = p.clamp(0, 7) as u8; if p & MPK_GU == 0 { p & !MPK_FOLD } else { p } };
        if p == planned || p & (MPK_GU | MPK_DN) == 0 { continue; }
        for m in 1..=width.clamp(1, 16) {
            if a1_smem(m, h) == 0 || a1_smem(m, mi) == 0 { continue; }
            let emax = width * topk;
            if p & MPK_GU != 0 { let _ = mpk_plan(dev, true, topk, m, h, 2 * mi, emax, p & MPK_FOLD != 0, 3); }
            if p & MPK_DN != 0 { let _ = mpk_plan(dev, false, topk, m, mi, h, emax, false, 3); }
        }
    }
}
