//! cuBLASLt plan cache (`LtPlan`, handle, pinned plans) and the Lt matmul entry points used by the reconstruct
//! arm of the dense chain, plus `exl3_reconstruct` and `a1_smem`. See `mod.rs` for the file map.

use super::*;

// The descriptors are opaque host objects: created on one thread (the prefill thread, or the
// PFX1 pre-population thread) and afterwards only READ, under the PLANS mutex, by the prefill
// thread's matmul; same contract as `unsafe impl Send for FwdModel`.
unsafe impl Send for LtPlan {}

/// Plan-cache key: (k, n, m, ta). The workspace preference is NOT in the key — every caller
/// passes the PrefillScratch's 32 MB lt_ws, and PFX1's pre-population replays the recorded one.
pub(super) type LtKey = (usize, usize, usize, u8);

/// The process-wide cuBLASLt handle (created on first use; cuBLASLt handles are thread-safe —
/// PFX1's pre-population thread issues heuristic queries on it, never matmuls).
pub(super) fn lt_handle() -> ltsys::cublasLtHandle_t {
    static HANDLE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    // cublasLtHandle_t is an opaque pointer: stash as usize (Copy) and cast back.
    *HANDLE.get_or_init(|| lt::create_handle().expect("cublasLtCreate") as usize) as ltsys::cublasLtHandle_t
}

pub(super) fn lt_plans() -> &'static std::sync::Mutex<std::collections::HashMap<LtKey, LtPlan>> {
    static PLANS: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<LtKey, LtPlan>>> =
        std::sync::OnceLock::new();
    PLANS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Build the plan of one (k, n, m, ta) shape: descriptors + the heuristic's top algo. A pure
/// function of (k, n, m, ta, ltws_len) — no timing, no pointers — so a plan built by PFX1's
/// background thread is the plan the lazy path would have built (same algo => same bits).
pub(super) fn lt_plan_build(handle: ltsys::cublasLtHandle_t, k: usize, n: usize, m: usize, ta: bool,
                 ltws_len: usize) -> std::result::Result<LtPlan, String> {
    let e = |what: &str, r: lt::CublasError| format!("cublasLt {what} (k={k} n={n} m={m} ta={ta}): {r:?}");
    let bf = ltsys::cudaDataType_t::CUDA_R_16F;
    let desc = lt::create_matmul_desc(
        ltsys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        ltsys::cudaDataType_t::CUDA_R_32F)
        .map_err(|r| e("create_matmul_desc", r))?;
    // ta=true: A is a plain f16 weight row-major [N, K] → col-major [K, N] ld=K,
    // transposed (the xq_gemm_f16 a_w/b_w convention). ta=false: A is Wdec
    // row-major [K, N] → col-major [N, K] ld=N, not transposed.
    let ta_v: u32 = if ta { 1 } else { 0 };
    let tb: u32 = 0; // opB=N: B row-major [m, K] → col-major [K, m] ld=K
    unsafe {
        lt::set_matmul_desc_attribute(desc,
            ltsys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &ta_v as *const _ as *const _, 4).map_err(|r| e("set TRANSA", r))?;
        lt::set_matmul_desc_attribute(desc,
            ltsys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            &tb as *const _ as *const _, 4).map_err(|r| e("set TRANSB", r))?;
    }
    let pref = lt::create_matmul_pref().map_err(|r| e("create_matmul_pref", r))?;
    let ws_size: usize = ltws_len;
    unsafe {
        lt::set_matmul_pref_attribute(pref,
            ltsys::cublasLtMatmulPreferenceAttributes_t::
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &ws_size as *const _ as *const _, 8).map_err(|r| e("set MAX_WORKSPACE", r))?;
    }
    // opA=T: A col-major [K, N] ld=K; opA=N: A col-major [N, K] ld=N.
    let a_layout = if ta {
        lt::create_matrix_layout(bf, k as u64, n as u64, k as i64)
    } else {
        lt::create_matrix_layout(bf, n as u64, k as u64, n as i64)
    }.map_err(|r| e("a_layout", r))?;
    let b_layout = lt::create_matrix_layout(bf, k as u64, m as u64, k as i64)
        .map_err(|r| e("b_layout", r))?;
    let c_layout = lt::create_matrix_layout(bf, n as u64, m as u64, n as i64)
        .map_err(|r| e("c_layout", r))?;
    let heur = lt::get_matmul_algo_heuristic(handle, desc,
            a_layout, b_layout, c_layout, c_layout, pref)
        .map_err(|r| e("heuristic (f16 wide-M)", r));
    // the preference only feeds the heuristic query (its result is a value copy)
    unsafe { let _ = lt::destroy_matmul_pref(pref); }
    Ok(LtPlan { algo: heur?.algo, desc, a_layout, b_layout, c_layout })
}

/// Release a plan's descriptors (PFX1: a pre-populated duplicate that lost the insert race).
pub(super) fn lt_plan_drop(p: LtPlan) {
    unsafe {
        let _ = lt::destroy_matrix_layout(p.a_layout);
        let _ = lt::destroy_matrix_layout(p.b_layout);
        let _ = lt::destroy_matrix_layout(p.c_layout);
        let _ = lt::destroy_matmul_desc(p.desc);
    }
}

/// PFX1 (e): while set, every exl3_lt_matmul records its (k, n, m, ta, ltws_len) into LT_REC
/// (armed by prefill_warmup around its full-width chunk only).
pub(super) static LT_REC_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub(super) static LT_REC: std::sync::Mutex<Vec<(usize, usize, usize, bool, usize)>> = std::sync::Mutex::new(Vec::new());
/// PFX1 (e): the Lt shape families of the served prefill at chunk width C — (k, n, ta, m/C,
/// ltws_len) — plus C, derived from the warm-up's full-width chunk; consumed by lt_prewarm_start.
#[allow(clippy::type_complexity)]
pub(super) static LT_FAMILIES: std::sync::Mutex<Option<(Vec<(usize, usize, bool, usize, usize)>, usize)>> =
    std::sync::Mutex::new(None);

/// One wide-M GEMM through cuBLASLt (SM121 nvjet kernels) on a reconstructed f16
/// weight. Layouts (all f16, fp32 accumulate): A = Wdec row-major [K,N] →
/// col-major view [N,K], ld=N, opA=N; B = xh row-major [m,K] → col-major [K,m],
/// ld=K, opB=N (row offset applied by the caller via pointer); C row-major
/// [m,N] → col-major [N,m], ld=N. Heuristics per (k,n,m), cached process-wide
/// (the brief's "heuristics first"; the winning algo is pinned by the cache).
pub(super) fn exl3_lt_matmul(l: &Launcher, a: u64, b: u64, c: u64,
                  k: usize, n: usize, m: usize, ta: bool, ltws: u64, ltws_len: usize) -> Result<()> {
    if let Some(segs) = lt_seg_split(m) {
        // PACKPF: one call per packed prompt at that prompt's own row count (row-offset B / C)
        let mut r0 = 0usize;
        for mi in segs {
            exl3_lt_matmul_one(l, a, b + (r0 * k * 2) as u64, c + (r0 * n * 2) as u64, k, n, mi, ta, ltws, ltws_len)?;
            r0 += mi;
        }
        return Ok(());
    }
    exl3_lt_matmul_one(l, a, b, c, k, n, m, ta, ltws, ltws_len)
}

// PACKPF (idea and first implementation: GitHub andersc, veloGB10 fork, PACK-PF): inside a packed
// prefill pass (prefill_segs with 2+ segments) LT_SEGS holds the segments' row counts in row order.
// A cuBLASLt GEMM over all the pass's token rows (m == f * their sum, f = rows per token: 1, or hc
// for the head-fill fc chains) runs as one call per segment at that segment's own m, so each prompt
// gets the algorithm cuBLASLt picks for its own row count — the one its unpacked prefill runs.
// cuBLASLt picks its algorithm per m; every other launch of the pass is row-independent.
std::thread_local! {
    pub(super) static LT_SEGS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// PACKPF: the per-segment row counts a packed pass splits an m-row GEMM into, or None.
fn lt_seg_split(m: usize) -> Option<Vec<usize>> {
    LT_SEGS.with(|v| {
        let v = v.borrow();
        if v.len() < 2 { return None; }
        let sum: usize = v.iter().sum();
        if sum == 0 || m % sum != 0 { return None; }
        let f = m / sum;
        Some(v.iter().map(|&x| x * f).collect())
    })
}

/// PACKPF: is a packed (multi-segment) prefill pass running on this thread?
pub(super) fn lt_segs_active() -> bool { LT_SEGS.with(|v| v.borrow().len() > 1) }

fn exl3_lt_matmul_one(l: &Launcher, a: u64, b: u64, c: u64,
                      k: usize, n: usize, m: usize, ta: bool, ltws: u64, ltws_len: usize) -> Result<()> {
    if LT_REC_ON.load(std::sync::atomic::Ordering::Relaxed) {
        let mut rec = LT_REC.lock().unwrap();
        if !rec.contains(&(k, n, m, ta, ltws_len)) { rec.push((k, n, m, ta, ltws_len)); }
    }
    let handle = lt_handle();
    let mut map = lt_plans().lock().unwrap();
    let plan = map.entry((k, n, m, ta as u8)).or_insert_with(|| {
        lt_plan_build(handle, k, n, m, ta, ltws_len).unwrap_or_else(|s| panic!("{s}"))
    });
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let r = unsafe {
        lt::matmul(handle, plan.desc,
            &alpha as *const _ as *const core::ffi::c_void,
            &beta as *const _ as *const core::ffi::c_void,
            a as *const core::ffi::c_void, plan.a_layout,
            b as *const core::ffi::c_void, plan.b_layout,
            c as *mut core::ffi::c_void, plan.c_layout,
            c as *mut core::ffi::c_void, plan.c_layout,
            &plan.algo, ltws as *mut core::ffi::c_void, ltws_len,
            l.stream.stream as usize as ltsys::cudaStream_t)
    };
    r.with_context(|| format!("cublasLt matmul k={k} n={n} m={m}"))
}

/// FP4L: exl3_lt_matmul with a caller alpha (the exact power-of-two output scale of a pre-scaled f16
/// weight). Same plan cache, same layouts; alpha only scales the fp32 result before the f16 store.
#[allow(clippy::too_many_arguments)]
pub(super) fn exl3_lt_matmul_alpha(l: &Launcher, a: u64, b: u64, c: u64,
                        k: usize, n: usize, m: usize, ta: bool, ltws: u64, ltws_len: usize, alpha: f32) -> Result<()> {
    if let Some(segs) = lt_seg_split(m) {
        let mut r0 = 0usize;
        for mi in segs {
            exl3_lt_matmul_alpha(l, a, b + (r0 * k * 2) as u64, c + (r0 * n * 2) as u64, k, n, mi, ta, ltws, ltws_len, alpha)?;
            r0 += mi;
        }
        return Ok(());
    }
    let handle = lt_handle();
    let mut map = lt_plans().lock().unwrap();
    let plan = map.entry((k, n, m, ta as u8)).or_insert_with(|| {
        lt_plan_build(handle, k, n, m, ta, ltws_len).unwrap_or_else(|s| panic!("{s}"))
    });
    let beta: f32 = 0.0;
    let r = unsafe {
        lt::matmul(handle, plan.desc,
            &alpha as *const _ as *const core::ffi::c_void,
            &beta as *const _ as *const core::ffi::c_void,
            a as *const core::ffi::c_void, plan.a_layout,
            b as *const core::ffi::c_void, plan.b_layout,
            c as *mut core::ffi::c_void, plan.c_layout,
            c as *mut core::ffi::c_void, plan.c_layout,
            &plan.algo, ltws as *mut core::ffi::c_void, ltws_len,
            l.stream.stream as usize as ltsys::cudaStream_t)
    };
    r.with_context(|| format!("cublasLt matmul (alpha {alpha}) k={k} n={n} m={m}"))
}

// ---------------------------------------------------------------------------------------------
// TP-SP (sequence-parallel prefill): PINNED-PLAN mode. A rank that owns rows [lo, hi) of a c-row
// chunk runs the chunk's per-row GEMMs on its rows only, with the (k, n, c) plan's ALGO pinned on the
// m = hi - lo layouts (row-offset pointers) — TP-I3 §3.1: then every output row is bit-identical to
// the m = c call for c >= 97 (hc down; hc up at every c), while each half's OWN heuristic plan differs
// at 1,971 of 2,047 sizes (split-K 2/3/4/12 vs 1/2/3/4). The gate is `--probe-exl3-sppin` (also the
// EXL3-SP-PIN section of --probe-exl3-binv): every c in [T, SP_MAX] for every split shape, through
// exactly the functions below.
// ---------------------------------------------------------------------------------------------

/// Pinned-plan cache key: (k, n, m, ta, pin_m) — the plan of (k, n, pin_m)'s algo on m-row layouts.
type LtPinKey = (usize, usize, usize, u8, usize);

fn lt_pinned() -> &'static std::sync::Mutex<std::collections::HashMap<LtPinKey, LtPlan>> {
    static P: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<LtPinKey, LtPlan>>> = std::sync::OnceLock::new();
    P.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Pinned keys whose algo does not run on the m-row layouts (AlgoCheck refused / workspace too big):
/// cached so the refusal is decided once, identically on every rank.
fn lt_pin_bad() -> &'static std::sync::Mutex<std::collections::HashMap<LtPinKey, String>> {
    static B: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<LtPinKey, String>>> = std::sync::OnceLock::new();
    B.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// The algo of the SERVED (k, n, m, ta) plan — the cached plan exl3_lt_matmul runs at that m (built the
/// same way, lazily, when absent: a pure function of (k, n, m, ta, ltws_len)).
fn lt_ref_algo(k: usize, n: usize, m: usize, ta: bool, ltws_len: usize) -> Result<ltsys::cublasLtMatmulAlgo_t> {
    let handle = lt_handle();
    let mut map = lt_plans().lock().unwrap();
    if let std::collections::hash_map::Entry::Vacant(v) = map.entry((k, n, m, ta as u8)) {
        v.insert(lt_plan_build(handle, k, n, m, ta, ltws_len).map_err(|s| anyhow::anyhow!(s))?);
    }
    Ok(map[&(k, n, m, ta as u8)].algo)
}

/// A plan whose algo is GIVEN (no heuristic): lt_plan_build's descriptors / layouts for (k, n, m, ta),
/// `algo` pinned, validated with cublasLtMatmulAlgoCheck and a workspace need <= ltws_len.
fn lt_plan_build_pinned(handle: ltsys::cublasLtHandle_t, k: usize, n: usize, m: usize, ta: bool, ltws_len: usize,
                        algo: ltsys::cublasLtMatmulAlgo_t) -> std::result::Result<LtPlan, String> {
    let e = |what: &str, r: lt::CublasError| format!("cublasLt {what} (pinned k={k} n={n} m={m} ta={ta}): {r:?}");
    let bf = ltsys::cudaDataType_t::CUDA_R_16F;
    let desc = lt::create_matmul_desc(ltsys::cublasComputeType_t::CUBLAS_COMPUTE_32F, ltsys::cudaDataType_t::CUDA_R_32F)
        .map_err(|r| e("create_matmul_desc", r))?;
    let ta_v: u32 = if ta { 1 } else { 0 };
    let tb: u32 = 0;
    unsafe {
        lt::set_matmul_desc_attribute(desc, ltsys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &ta_v as *const _ as *const _, 4).map_err(|r| e("set TRANSA", r))?;
        lt::set_matmul_desc_attribute(desc, ltsys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            &tb as *const _ as *const _, 4).map_err(|r| e("set TRANSB", r))?;
    }
    let a_layout = if ta { lt::create_matrix_layout(bf, k as u64, n as u64, k as i64) }
                   else { lt::create_matrix_layout(bf, n as u64, k as u64, n as i64) }.map_err(|r| e("a_layout", r))?;
    let b_layout = lt::create_matrix_layout(bf, k as u64, m as u64, k as i64).map_err(|r| e("b_layout", r))?;
    let c_layout = lt::create_matrix_layout(bf, n as u64, m as u64, n as i64).map_err(|r| e("c_layout", r))?;
    let plan = LtPlan { algo, desc, a_layout, b_layout, c_layout };
    let mut chk = std::mem::MaybeUninit::<ltsys::cublasLtMatmulHeuristicResult_t>::zeroed();
    let rc = unsafe {
        ltsys::cublasLtMatmulAlgoCheck(handle, plan.desc, plan.a_layout, plan.b_layout, plan.c_layout, plan.c_layout,
                                       &plan.algo, chk.as_mut_ptr())
    };
    if rc != ltsys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        lt_plan_drop(plan);
        return Err(format!("AlgoCheck(k={k} n={n} m={m}) = {rc:?}"));
    }
    let wsz = unsafe { chk.assume_init() }.workspaceSize;
    if wsz > ltws_len {
        lt_plan_drop(plan);
        return Err(format!("pinned workspace {wsz} > {ltws_len} (k={k} n={n} m={m})"));
    }
    Ok(plan)
}

/// Build (once) the pinned plan of (k, n, m, ta) under the (k, n, pin_m) algo; Err = it cannot run.
fn lt_pin_plan_ensure(k: usize, n: usize, m: usize, pin_m: usize, ta: bool, ltws_len: usize) -> Result<()> {
    let key: LtPinKey = (k, n, m, ta as u8, pin_m);
    if lt_pinned().lock().unwrap().contains_key(&key) { return Ok(()); }
    if let Some(why) = lt_pin_bad().lock().unwrap().get(&key) { anyhow::bail!("{why}"); }
    let algo = lt_ref_algo(k, n, pin_m, ta, ltws_len)?;
    match lt_plan_build_pinned(lt_handle(), k, n, m, ta, ltws_len, algo) {
        Ok(p) => {
            let mut map = lt_pinned().lock().unwrap();
            if let std::collections::hash_map::Entry::Vacant(v) = map.entry(key) { v.insert(p); } else { drop(map); lt_plan_drop(p); }
            Ok(())
        }
        Err(s) => { lt_pin_bad().lock().unwrap().insert(key, s.clone()); anyhow::bail!("{s}") }
    }
}

/// TP-SP: are the pinned plans of EVERY rank's row block of a c-row chunk runnable for (k, n)? Every
/// rank evaluates every block (not only its own), so the verdict is the same on all ranks (SPMD).
pub(crate) fn lt_pin_ready(k: usize, n: usize, c: usize, world: usize, ta: bool, ltws_len: usize) -> Result<()> {
    for r in 0..world {
        let (lo, hi) = crate::tp_xport::row_block(c, r, world);
        if hi > lo && hi - lo != c { lt_pin_plan_ensure(k, n, hi - lo, c, ta, ltws_len)?; }
    }
    Ok(())
}

/// TP-SP: rows [0, m) of `b` / `c` (row-offset pointers of a pin_m-row chunk) through the (k, n, pin_m)
/// plan's algo pinned on the m-row layouts — bitwise those rows of exl3_lt_matmul at m = pin_m wherever
/// the gate proved it (--probe-exl3-sppin). m == pin_m is exactly exl3_lt_matmul.
#[allow(clippy::too_many_arguments)]
pub(super) fn exl3_lt_matmul_pinned(l: &Launcher, a: u64, b: u64, c: u64, k: usize, n: usize, m: usize, pin_m: usize,
                         ta: bool, ltws: u64, ltws_len: usize) -> Result<()> {
    if m == pin_m { return exl3_lt_matmul(l, a, b, c, k, n, m, ta, ltws, ltws_len); }
    if LT_REC_ON.load(std::sync::atomic::Ordering::Relaxed) {
        // PFX1 (e): the reference family (m = pin_m) is what the background pre-population must build
        let mut rec = LT_REC.lock().unwrap();
        if !rec.contains(&(k, n, pin_m, ta, ltws_len)) { rec.push((k, n, pin_m, ta, ltws_len)); }
    }
    lt_pin_plan_ensure(k, n, m, pin_m, ta, ltws_len)?;
    let handle = lt_handle();
    let map = lt_pinned().lock().unwrap();
    let plan = &map[&(k, n, m, ta as u8, pin_m)];
    let (alpha, beta) = (1.0f32, 0.0f32);
    let r = unsafe {
        lt::matmul(handle, plan.desc,
            &alpha as *const _ as *const core::ffi::c_void, &beta as *const _ as *const core::ffi::c_void,
            a as *const core::ffi::c_void, plan.a_layout, b as *const core::ffi::c_void, plan.b_layout,
            c as *mut core::ffi::c_void, plan.c_layout, c as *mut core::ffi::c_void, plan.c_layout,
            &plan.algo, ltws as *mut core::ffi::c_void, ltws_len, l.stream.stream as usize as ltsys::cudaStream_t)
    };
    r.with_context(|| format!("cublasLt pinned matmul k={k} n={n} m={m} (pin m={pin_m})"))
}

/// Decode ONE trellis matrix into `ws` as plain W_dec row-major [K, N] f16 —
/// the same values the fused mma consumes. `tr` may be a Quad base (u64) or an
/// expert slice offset (same in-tree u64-pointer launch convention as
/// exl3_hmma_gemm_wide's MoE calls).
pub(super) fn exl3_reconstruct(l: &Launcher, tr: u64, rc: u64,
                    ws: u64, k: i32, n: i32, k2: i32) -> Result<()> {
    let kb = (k as u32) / 16;
    let nb = (n as u32) / 16;
    if nb % 8 == 0 && k2 <= 16 && crate::opts::var(crate::opt!("exl3-dump8")).map_or(true, |v| v != "0") {
        // S-A3-u G: coalesced strip reconstruct — same bytes (gated in --probe-exl3-kernel).
        return xqlaunch!(l, "exl3_decode_dump8", (kb * nb / 8, 1, 1), (256, 1, 1), 0,
                         (tr, rc, ws, nb as i32, k2));
    }
    xqlaunch!(l, "exl3_decode_dump", (kb * nb, 1, 1), (32, 1, 1), 0,
             (tr, rc, ws, nb as i32, k2))
}

/// S-A3-n F3: dynamic smem for the A-once GEMM body (`exl3_gemm_body_a1`, taken by
/// the grouped expert kernels when it fits). Must mirror `exl3_a1_fits` in
/// kernels/exl3_bench.cu; 0 = the kernel keeps the barrier body.
pub(crate) fn a1_smem(m: usize, k: usize) -> u32 {
    let bytes = m * (k + 8) * 2;
    if m >= 1 && m <= 16 && k % 8 == 0 && bytes <= 48 * 1024 { bytes as u32 } else { 0 }
}
