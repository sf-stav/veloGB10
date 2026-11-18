//! Router entry selection (R16 liveness record and `RouterEntry`), the dynamic-smem opt-in helpers, the WP10 /
//! PDL tunables and `launch_pss_raw`. A new dispatch path ships a once-per-boot liveness line (AGENTS 4).
//! See `mod.rs` for the file map.

use super::*;

/// WP11: co-resident block capacity (blocks/SM at 256 threads x SMs) of a grid-barrier kernel
/// of the loaded exl3_bench module (cudarc occupancy on the same function handle the launcher
/// uses). 0 on any failure => that twin is never launched.
pub(super) fn wp11_capacity(dev: &Arc<CudaDevice>, name: &str) -> u32 {
    use cudarc::driver::sys;
    let Some(f) = dev.get_func(MODULE, name) else { return 0; };
    let Ok(bps) = f.occupancy_max_active_blocks_per_multiprocessor(256, 0, None) else { return 0; };
    let Ok(sms) = dev.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) else { return 0; };
    bps.saturating_mul(sms.max(0) as u32)
}

/// WP11: true while the launcher's stream is recording a graph (or the query fails) — the
/// eager-only diagnostics must not synchronize inside a capture.
pub(super) fn stream_capturing(l: &Launcher) -> bool {
    use cudarc::driver::sys;
    let mut st = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    let r = unsafe { sys::cuStreamIsCapturing(l.stream.stream, &mut st) };
    r != sys::CUresult::CUDA_SUCCESS || st != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
}

/// S-B9-CF-R16 (PLAN/CF_R16_REPORT.md): serve 9..16-row forwards on the 9..16-row twins of the
/// routing fold (xq_router_fold_c_r16) and the row-batched HC mixer (xq_hc_fuse_i8_rbk_r16 /
/// xq_hc_w4). Both are bit-identical to the paths today's >8 rows take, so the escape restores
/// today's dispatch exactly: --exl3-r16=0 => the 4-kernel legacy router chain + the m-grid
/// xq_hc_fuse_i8k. m <= 8 is never affected by the flag.
pub(super) fn exl3_r16_on() -> bool {
    // S-B9-CF-R16B FIX: `opts::var` only sees EXPLICITLY set flags, so the R16 leg's
    // `.is_ok()` gate kept the WHOLE r16 path dormant at default flags (the liveness
    // receipt caught it: "fallback hc=xq_hc_fuse_i8k (reason: --exl3-r16=0)" on a boot
    // with no --exl3-r16 flag). Unset now reads the table default (opts_table: def "on");
    // `--exl3-r16=0/off` still restores today's pre-R16 dispatch exactly.
    // S-B9-REL-FIXB K9: cached once — the flag cannot change after boot and this is
    // consulted at every HC launch (it used to allocate a String per call).
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let v = crate::opts::var(crate::opt!("exl3-r16"))
            .unwrap_or_else(|_| crate::opt!("exl3-r16").def().def.to_string());
        !matches!(v.trim(), "" | "0" | "off" | "false" | "OFF")
    })
}
/// S-B9-CF-R16B: the once-per-boot liveness receipt (TASK 3) — which router fold and which
/// HC mixer entry served (or were skipped for) the first 9..16-row forward at the CURRENT
/// flags. An identity-hash gate proves nothing about liveness: a path that never runs also
/// gives identical hashes; this line is the receipt future sessions grep for. The HC mixer
/// runs BEFORE the router fold inside a forward, so the line normally prints from the
/// router-fold site; if the fold never takes at m > 8, the second HC width emits it with
/// "(router fold not reached)" so the receipt always exists.
static R16_ROUTER: std::sync::OnceLock<String> = std::sync::OnceLock::new();
static R16_HC: std::sync::OnceLock<(&'static str, Option<&'static str>)> = std::sync::OnceLock::new();
static R16_HC_HITS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static R16_PRINTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn r16_set_router(name: &str) {
    let _ = R16_ROUTER.set(name.to_string());
    r16_emit(false);
}
pub(crate) fn r16_liveness(hc: &'static str, reason: Option<&'static str>) {
    let n = R16_HC_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n == 0 { let _ = R16_HC.set((hc, reason)); }
    // The MTP draft chain runs the head's HC mixer k(=7) times per round BEFORE the
    // verify's first router fold, so a force at the second hit would print before any
    // m > 8 routing ran. Force only as a last resort (no-MoE model): hit 12.
    r16_emit(n >= 11);
}
fn r16_emit(force: bool) {
    let Some((hc, reason)) = R16_HC.get() else { return };
    if !(force || R16_ROUTER.get().is_some()) { return; }
    if R16_PRINTED.swap(true, std::sync::atomic::Ordering::Relaxed) { return; }
    let r = R16_ROUTER.get().map(String::as_str).unwrap_or("(router fold not reached)");
    match reason {
        Some(rr) => crate::dprintln!("EXL3-R16: fallback router={r} hc={hc} (reason: {rr})"),
        None => crate::dprintln!("EXL3-R16: live router={r} hc={hc}"),
    }
}

/// S-B9-REL-FIXB K1: which router entry serves a forward of m rows. PURE (no tune
/// lookups, no side effects) so the selection is exhaustively testable
/// (`router_selection_table_exhaustive`). The caller resolves every flag; `moe_inner`
/// maps each variant to its launch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RouterEntry {
    /// xq_router_ks + combine (or xq_gemm_f16_f32_v) + xq_router_topk + xq_moe_route — any m.
    Legacy,
    /// xq_router_fused + xq_router_topk_route — M <= 8 hardware only.
    FusedPair,
    /// xq_router_fold_c_ep — M <= 8 (XQ_RF_MAXM 8).
    FoldEp,
    /// xq_router_fold_c_r16 — the ONLY 9..16-row entry (coal fold, non-EP rank).
    FoldR16,
    /// xq_router_fold_c — M <= 8.
    FoldCoal,
    /// xq_router_fold_w4 — M <= 8.
    FoldW4,
    /// xq_router_fold — M <= 8.
    Fold,
    /// xq_router_fold_m0 — M <= 8.
    FoldM0,
}

impl RouterEntry {
    /// True for every fold-family entry (the one-launch routing fold).
    pub(crate) fn is_fold(self) -> bool {
        matches!(self, RouterEntry::FoldEp | RouterEntry::FoldR16 | RouterEntry::FoldCoal
                     | RouterEntry::FoldW4 | RouterEntry::Fold | RouterEntry::FoldM0)
    }
}

/// The invariant this function exists to enforce (REL v0.7.3 review K1): for m > 8 the
/// only possible entries are `FoldR16` and `Legacy` — no M <= 8 body (xq_router_fold_c_ep,
/// _c, _w4, xq_router_fold, _m0, the fused pair) can ever be selected, whatever the flags.
/// At m <= 8 and with r16 off the selection is EXACTLY the pre-R16 (v0.7.2) dispatch.
pub(crate) fn router_entry(
    m: usize,
    ep: bool,          // moe.ep.is_some() (this rank holds an EP shard)
    coal: bool,        // router-coal twin eligible (h == 2560 && T_MOE_ROUTER_COAL)
    route_fold: bool,  // xtp::route_fold_on()
    r16_hw: bool,      // --exl3-r16 on AND the r16 smem opt-in succeeded
    fused_ok: bool,    // m-free fused gates (rks == 64 && shape gates && T_ROUTER_FUSED)
    fold_ok: bool,     // m-free fold gates (wp10_fold_on() && ne <= 512)
    topk_cap: bool,    // m * topk <= (m <= 8 ? 128 : 160)
    w4r: bool,         // the W4S twin took the fold (it launches itself)
    gemv_remap: bool,  // wp10_gemv_remap()
    xcheck: bool,      // an eager XCHECK snapshot precedes the fold (rxc || fxc)
) -> RouterEntry {
    // K1: the r16 twin exists ONLY as the COAL fold on a NON-EP rank. EP ranks and
    // non-coal runs keep the pre-R16 dispatch (legacy chain) at 9..16 rows.
    let r16 = m > 8 && m <= 16 && r16_hw && coal && !ep;
    let fold = fused_ok && fold_ok && topk_cap && (m <= 8 || r16);
    if !fold {
        // The non-fold fused pair is M <= 8 hardware (its GEMV covers 8 rows) — K1
        // corollary: it never serves above 8 rows, not even as an xcheck reference.
        return if fused_ok && m <= 8 { RouterEntry::FusedPair } else { RouterEntry::Legacy };
    }
    if coal && m <= 8 && !xcheck && ep && route_fold {
        return RouterEntry::FoldEp;
    }
    if m > 8 {
        // fold at m > 8 requires r16 by construction; FoldR16 is the safe answer even
        // if a future edit breaks that — it is the only 9..16-row body.
        return RouterEntry::FoldR16;
    }
    if coal {
        return RouterEntry::FoldCoal;
    }
    if w4r {
        return RouterEntry::FoldW4;
    }
    if gemv_remap {
        RouterEntry::Fold
    } else {
        RouterEntry::FoldM0
    }
}

/// Dynamic smem of xq_router_fold_c_r16 (8 warps x 4 experts x 40 float4 == XQ_RFC_SMEM_R16 —
/// the m<=8 fold's exact 20 KB carve-out; the 256-thread geometry never changed).
pub(crate) const RFC_SMEM_R16: u32 = 8 * 4 * 40 * 16;
/// S-B9-CF-R16: ONE-TIME smem-cap opt-in on xq_router_fold_c_r16. 20 KB is under the 48 KB
/// default, so this is belt-and-braces (it also proves the entry is in the loaded module);
/// a refusal drops m > 8 back to today's path instead of aborting a launch. The call site
/// caches the result in a OnceLock — never a per-launch driver call.
/// (xq_hc_fuse_i8_rbk_r16's 64 KB is STATIC smem: part of the launch config, no opt-in.)
pub(crate) fn r16_smem_optin(dev: &Arc<CudaDevice>, name: &str, bytes: i32) -> bool {
    use cudarc::driver::sys;
    // xq_raw_fn already hands back the raw CUfunction cuFuncSetAttribute wants.
    let Some(f) = xq_raw_fn(name) else {
        crate::deprintln!("EXL3-R16: {name} not in the module (stale PTX?)");
        return false;
    };
    let r = unsafe {
        sys::cuFuncSetAttribute(f, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes)
    };
    if r != sys::CUresult::CUDA_SUCCESS {
        eprintln!("EXL3-R16: {name} smem opt-in of {bytes} B failed ({r:?})");
        return false;
    }
    true
}
/// S-B9-CF-R16: xq_hc_fuse_i8_rbk_r16 takes its smem carve-out DYNAMICALLY (R4 hn staging
/// [2][m][1024] halves = m*4096 B; phase B [m][lr] f32 <= m*2048 B at lr = 512). The m = 16
/// R4 case asks 64 KB — over the 48 KB default — so the function must opt in once before it
/// launches or is measured (a STATIC 64 KB array compiles but cuOccupancy* reports 0 resident
/// CTAs for it on this arch; dynamic + opt-in is the supported path). 65536 B covers this
/// family (lr = 512); the dispatch gate keeps widths needing more on today's m-grid path.
pub(crate) fn r16_hc_smem_optin() -> bool {
    use cudarc::driver::sys;
    let Some(f) = xq_raw_fn("xq_hc_fuse_i8_rbk_r16") else {
        crate::deprintln!("EXL3-R16: xq_hc_fuse_i8_rbk_r16 not in the module (stale PTX?)");
        return false;
    };
    for (attr, val) in [
        (sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 65536),
        (sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100),
    ] {
        let r = unsafe { sys::cuFuncSetAttribute(f, attr, val) };
        if r != sys::CUresult::CUDA_SUCCESS {
            eprintln!("EXL3-R16: xq_hc_fuse_i8_rbk_r16 smem opt-in ({val}) failed ({r:?})");
            return false;
        }
    }
    true
}

/// Occupancy of the RAW-module handle (exl3_bench.ptx is loaded TWICE: xq_raw_fn's copy,
/// which the PDL launches run, and cudarc's get_func copy, whose opt-in state is
/// unreachable — cu_function is pub(crate)). The r16 HC twin's > 48 KB dynamic carve-out
/// can only be opted in on the raw copy (xq_raw_fn_smem does it), so its co-residency is
/// measured on THAT handle, at the carve-out its launches actually use.
pub(super) fn wp11_capacity_raw_dsmem(dev: &Arc<CudaDevice>, name: &str, dsmem: u32) -> u32 {
    use cudarc::driver::sys;
    let Some(f) = xq_raw_fn_smem(name, dsmem) else { return 0; };
    let mut b: std::ffi::c_int = 0;
    let r = unsafe { sys::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &mut b, f, 256, dsmem as usize, 0u32) };   // 0 = CU_OCCUPANCY_DEFAULT
    if r != sys::CUresult::CUDA_SUCCESS || b <= 0 { return 0; }
    let Ok(sms) = dev.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) else { return 0; };
    (b as u32).saturating_mul(sms.max(0) as u32)
}
/// wp11_capacity as pub (the r16 probe sizes the HC r16 grid the way the launcher does).
pub(crate) fn wp11_capacity_pub(dev: &Arc<CudaDevice>, name: &str) -> u32 {
    wp11_capacity(dev, name)
}
/// wp11_capacity_raw_dsmem as pub (the r16 HC probe measures at the 64 KB carve-out,
/// on the same raw-module handle its launches use).
pub(crate) fn wp11_capacity_raw_dsmem_pub(dev: &Arc<CudaDevice>, name: &str, dsmem: u32) -> u32 {
    wp11_capacity_raw_dsmem(dev, name, dsmem)
}
/// WP10 routing fold (xq_router_fold: router GEMV + top-k + route in one launch, bitwise) — default
/// on; --wp10-off=1 = the S-A3-z xq_router_fused + xq_router_topk_route pair.
pub(super) fn wp10_fold_on() -> bool {
    tune::get_cur(&T_WP10_FOLD) != 0
}
pub(crate) const T_WP10_FOLD: tune::TunableDef = tune::TunableDef {
    slot: 12, id: "wp10.fold", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--wp10-xcheck",
    // --wp10-off=1 = fold OFF (value 0); any other value keeps it on
    env: "--wp10-off", env_parse: || tune::env_str(crate::opt!("wp10-off")).map(|v| (v != "1") as i32),
    rev: 1, wp: "WP10",
};
/// A5-K2 (ledger #4): the routing fold with COALESCED weight staging (xq_router_fold_c). The 80-B
/// per-thread weight read of xq_router_fold[_w4] streams the 2.62 MB router at ~146 GB/s; _c stages
/// each warp's expert chunks with lane-contiguous cp.async into smem (221 GB/s standalone) and picks
/// the top-k by a lane tree + redux.sync. Same per-(row, expert) FFMA chains, same 64-deep ascending
/// fold, same pick order: bitwise (class S; --probe-exl3-binv EXL3-ROUTERFOLD-C). Takes precedence
/// over W4S router when on. Default ON after the in-round A/B (PLAN/notes_2026-09-27/K2.md: GO);
/// escape --router-coal=0 (= the previous fold, W4S router).
pub(crate) const T_MOE_ROUTER_COAL: tune::TunableDef = tune::TunableDef {
    slot: 86, id: "moe.router_coal", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 1], default: 1, valid: tune::valid_any,
    xcheck: "--probe-exl3-binv EXL3-ROUTERFOLD-C + --wp10-xcheck / --exl3-router-xcheck (eager)",
    env: "--router-coal", env_parse: || tune::env_on_unless_0(crate::opt!("router-coal")), rev: 1, wp: "A5-K2",
};
/// Dynamic smem of xq_router_fold_c (== XQ_RFC_SMEM: 8 warps x 4 experts x 40 float4).
pub(crate) const RFC_SMEM: u32 = 8 * 4 * 40 * 16;

/// WP10 (b): the fold's GEMV lane remap + conflict-free partial layout (values untouched) — default
/// on; --wp10-gemv=0 = xq_router_fused's lane map/layout inside the fold (xq_router_fold_m0).
pub(super) fn wp10_gemv_remap() -> bool {
    tune::get_cur(&T_WP10_GEMV) != 0
}
pub(crate) const T_WP10_GEMV: tune::TunableDef = tune::TunableDef {
    slot: 13, id: "wp10.gemv", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_DECODE,
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--wp10-xcheck",
    env: "--wp10-gemv", env_parse: || tune::env_on_unless_0(crate::opt!("wp10-gemv")), rev: 1, wp: "WP10",
};

/// A5 D14: --exl3-pdl=1 launches every exl3_bench kernel with programmatic stream
/// serialization (graph capture records programmatic edges). Numerics untouched (ordering only).
/// TUNE T0: a per-graph-family registry value (S21); the lookup keys on the current family scope.
pub(super) fn pdl_on() -> bool {
    tune::get_cur(&T_PDL) != 0
}
pub(crate) const T_PDL: tune::TunableDef = tune::TunableDef {
    slot: 0, id: "pdl", class: tune::Class::S, scope: tune::Scope::Fam,
    fams: &[tune::Fam::Util, tune::Fam::Step, tune::Fam::Verify, tune::Fam::DraftPass, tune::Fam::DraftChain,
            tune::Fam::Reprime, tune::Fam::Prefill],
    // ordering only (programmatic edges); no bit-diff harness exists yet -> DOMAIN UNPROVEN for T2
    domain: &[0, 1], default: 0, valid: tune::valid_any, xcheck: "",
    env: "--exl3-pdl", env_parse: || tune::env_on_if_1(crate::opt!("exl3-pdl")), rev: 1, wp: "A5-D14",
};

/// Raw handle for a PDL launch; opts the raw function into dynamic smem > 48 KB when needed
/// (the cudarc module's handles carry their own opt-ins; this second module's do not).
pub(super) fn xq_raw_fn_smem(name: &str, smem: u32) -> Option<cudarc::driver::sys::CUfunction> {
    let f = xq_raw_fn(name)?;
    if smem > 48 * 1024 {
        use cudarc::driver::sys;
        let r = unsafe { sys::cuFuncSetAttribute(f,
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32) };
        if r != sys::CUresult::CUDA_SUCCESS { return None; }
    }
    Some(f)
}

/// cuLaunchKernelEx with CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION.
pub(super) unsafe fn launch_pss_raw(stream: cudarc::driver::sys::CUstream, f: cudarc::driver::sys::CUfunction,
                         grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32,
                         params: &mut [*mut std::ffi::c_void]) -> Result<()> {
    use cudarc::driver::sys::*;
    let mut attr = CUlaunchAttribute {
        id: CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
        pad: [0; 4],
        value: CUlaunchAttributeValue { programmaticStreamSerializationAllowed: 1 },
    };
    let mut cfg = CUlaunchConfig {
        gridDimX: grid.0, gridDimY: grid.1, gridDimZ: grid.2,
        blockDimX: block.0, blockDimY: block.1, blockDimZ: block.2,
        sharedMemBytes: smem,
        hStream: stream,
        attrs: &mut attr,
        numAttrs: 1,
    };
    let r = cuLaunchKernelEx(&mut cfg, f, params.as_mut_ptr(), std::ptr::null_mut());
    anyhow::ensure!(r == CUresult::CUDA_SUCCESS, "PDL launch failed: {r:?}");
    Ok(())
}
