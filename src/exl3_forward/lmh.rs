//! LM-head GEMM (LMH) lever: tunables, kernel entry selection by width, prewarm, `lmh_gemm_launch` and the
//! cross-check run. The LM head is vocab-parallel under TP. See `mod.rs` for the file map.

use super::*;

// ---- W3/LMH (2026-09-27): persistent lm_head GEMM (kernels/exl3_bench.cu "W3/LMH") ----
// Every path below is bitwise-identical by the kernel's contract; the knobs are A/B + bisect
// tools only (AGENTS §1b — the new path is the default):
//   --lmh-off=1      the old exl3_hmma_gemm + exl3_had_svh pair
//   --lmh-had=0      new GEMM writing y_raw + the separate exl3_had_svh (fused epilogue off)
//   --lmh-nst=4|6|8  cp.async ring depth of the 5-bit head (default 6; 3/4-bit always 6)
//   --lmh-xcheck=1   eager bit-diff of every NON-captured lm_head chain vs the old pair
//                       (use --exl3-no-graph=1 for coverage; prints LMH_XCHECK lines)
pub(super) fn lmh_off() -> bool {
    tune::get_cur(&T_LMH_OFF) != 0
}
pub(super) fn lmh_had() -> bool {
    tune::get_cur(&T_LMH_HAD) != 0
}
pub(super) fn lmh_nst() -> usize {
    tune::get_cur(&T_LMH_NST).max(1) as usize
}
// TUNE T0 (p3c): the W3/LMH switches (bitwise twins: LMH_XCHECK + the binv ring sweep).
pub(crate) const T_LMH_OFF: tune::TunableDef = tune::TunableDef {
    slot: 28, id: "lmh.off", class: tune::Class::S, scope: tune::Scope::Global, fams: FAMS_LMH,
    domain: &[0, 1], default: 0, valid: tune::valid_any, xcheck: "--probe-exl3-binv W3/LMH + --lmh-xcheck",
    env: "--lmh-off", env_parse: || tune::env_on_if_1(crate::opt!("lmh-off")), rev: 1, wp: "W3/LMH",
};
pub(crate) const T_LMH_HAD: tune::TunableDef = tune::TunableDef {
    slot: 29, id: "lmh.had", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_LMH,
    domain: &[0, 1], default: 1, valid: tune::valid_any, xcheck: "--probe-exl3-binv W3/LMH (fused svh) + --lmh-xcheck",
    env: "--lmh-had", env_parse: || tune::env_on_unless_0(crate::opt!("lmh-had")), rev: 1, wp: "W3/LMH",
};
pub(crate) const T_LMH_NST: tune::TunableDef = tune::TunableDef {
    slot: 30, id: "lmh.nst", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_LMH,
    // cp.async ring depth of the 5-bit head (3/4-bit heads always run 6: lmh_nst_for — inert there)
    domain: &[4, 6, 8], default: 6, valid: tune::valid_any, xcheck: "--probe-exl3-binv [full head, ring sweep]",
    env: "--lmh-nst",
    env_parse: || tune::env_str(crate::opt!("lmh-nst")).map(|v| match v.as_str() { "4" => 4, "8" => 8, _ => 6 }),
    rev: 1, wp: "W3/LMH",
};
/// Families whose graphs run an lm_head-class chain (verify / step full head, draft slice).
pub(crate) const FAMS_LMH: &[tune::Fam] = &[tune::Fam::Step, tune::Fam::Verify, tune::Fam::DraftPass,
                                            tune::Fam::DraftChain, tune::Fam::Util];
pub(super) fn lmh_xcheck() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::var(crate::opt!("lmh-xcheck")).is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// k16 steps per ring slot — mirror of LMH_KSTEP (kernels/exl3_bench.cu).
pub(crate) const LMH_KSTEP: usize = 4;

/// Mirror of lmh_smem_bytes() in kernels/exl3_bench.cu (the kernel traps if launched short).
pub(crate) fn lmh_smem_bytes(bits: usize, mrows: usize, nst: usize, k: usize) -> usize {
    nst * LMH_KSTEP * 8 * 32 * bits + mrows * 136 * 2  // W9b: m8 A-once term removed (kernel reads A via __ldg)
}

fn lmh_entry(bits: usize, mrows: usize, nst: usize) -> Option<&'static str> {
    Some(match (bits, mrows, nst) {
        (5, 8, 4) => "exl3_lmh_b5_m8_s4",
        (5, 8, 6) => "exl3_lmh_b5_m8_s6",
        (5, 8, 8) => "exl3_lmh_b5_m8_s8",
        (5, 16, 4) => "exl3_lmh_b5_m16_s4",
        (5, 16, 6) => "exl3_lmh_b5_m16_s6",
        (5, 16, 8) => "exl3_lmh_b5_m16_s8",
        (4, 8, 6) => "exl3_lmh_b4_m8_s6",
        (4, 16, 6) => "exl3_lmh_b4_m16_s6",
        (3, 8, 6) => "exl3_lmh_b3_m8_s6",
        (3, 16, 6) => "exl3_lmh_b3_m16_s6",
        (6, 8, 6) => "exl3_lmh_b6_m8_s6",
        (6, 16, 6) => "exl3_lmh_b6_m16_s6",
        _ => return None,
    })
}

/// Raw handle of one exl3_lmh_* entry (the xq-raw module), opted in to `smem` dynamic bytes and
/// checked to fit >= 1 CTA/SM at 256 threads. Cached per entry; the opt-in is raised when a
/// later call needs more. None = unusable (the caller keeps the old kernel).
fn lmh_fn(name: &'static str, smem: u32) -> Option<cudarc::driver::sys::CUfunction> {
    use cudarc::driver::sys;
    #[derive(Clone, Copy)]
    struct FnH(sys::CUfunction);
    unsafe impl Send for FnH {}
    unsafe impl Sync for FnH {}
    static C: std::sync::Mutex<Vec<(&'static str, FnH, u32)>> = std::sync::Mutex::new(Vec::new());
    let mut c = C.lock().ok()?;
    if let Some(e) = c.iter().find(|e| e.0 == name) {
        if smem <= e.2 { return Some(e.1 .0); }
    }
    let f = xq_raw_fn(name)?;
    if smem as usize > crate::gpu::SMEM_OPTIN_CAP as usize {
        crate::deprintln!("[lmh] {name}: {smem} B over the {} B opt-in cap", crate::gpu::SMEM_OPTIN_CAP);
        return None;
    }
    let r = unsafe { sys::cuFuncSetAttribute(f,
        sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32) };
    if r != sys::CUresult::CUDA_SUCCESS { eprintln!("[lmh] {name}: smem opt-in {smem} B failed ({r:?})"); return None; }
    let mut nb: i32 = 0;
    let r = unsafe { sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut nb, f, 256, smem as usize) };
    if r != sys::CUresult::CUDA_SUCCESS || nb < 1 {
        crate::deprintln!("[lmh] {name}: not resident at {smem} B smem (occupancy {nb}, {r:?})");
        return None;
    }
    crate::dprintln!("W3/LMH: {name} ready (dynamic smem {smem} B, {nb} CTA/SM; --lmh-off=1 = old exl3_hmma_gemm)");
    c.retain(|e| e.0 != name);
    c.push((name, FnH(f), smem));
    Some(f)
}

/// Ring depths instantiated per rate: 5-bit 4/6/8, 3/4/6-bit 6 (kernels/exl3_bench.cu LMH_ENTRY).
fn lmh_nst_for(bits: usize, nst: usize) -> usize {
    if bits == 5 && matches!(nst, 4 | 6 | 8) { nst } else { 6 }
}

/// Resolve + opt in both lm_head entries (m <= 8, m 9..16) at LOAD, outside any graph capture
/// (the lazy first use could otherwise land inside a verify capture). Prints the feature marker.
pub(super) fn lmh_prewarm(bits: usize, k: usize) {
    if lmh_off() {
        crate::dprintln!("W3/LMH: OFF (--lmh-off=1) — lm_head runs exl3_hmma_gemm + exl3_had_svh");
        return;
    }
    let nst = lmh_nst_for(bits, lmh_nst());
    for mrows in [8usize, 16] {
        match lmh_entry(bits, mrows, nst) {
            Some(name) if lmh_fn(name, lmh_smem_bytes(bits, mrows, nst, k) as u32).is_some() => {}
            _ => crate::dprintln!("W3/LMH: no usable entry for bits {bits} mrows {mrows} nst {nst} K {k} — \
                           those widths keep exl3_hmma_gemm"),
        }
    }
    // TUNE: every other ring depth the frozen table (or the --autotune overlay) can select is
    // opted in here too, so no lazy cuFuncSetAttribute ever lands inside a capture.
    for alt in tune::reachable(&T_LMH_NST) {
        let alt = lmh_nst_for(bits, alt.max(1) as usize);
        if alt == nst { continue; }
        for mrows in [8usize, 16] {
            if let Some(name) = lmh_entry(bits, mrows, alt) {
                let _ = lmh_fn(name, lmh_smem_bytes(bits, mrows, alt, k) as u32);
            }
        }
    }
    crate::dprintln!("W3/LMH: lm_head persistent GEMM ON (nst {nst}, fused had_svh {}, xcheck {})",
             lmh_had() as u8, lmh_xcheck() as u8);
}

pub(crate) fn lmh_sm_count(dev: &Arc<CudaDevice>) -> u32 {
    use cudarc::driver::sys;
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| dev.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .ok().filter(|&v| v > 0).map(|v| v as u32).unwrap_or(48))
}

/// W3/LMH launch on `stream`: out = logits (fuse_had: Hadamard + svh fused) or y_raw. Ok(false)
/// = shape/resources not eligible (the caller runs exl3_hmma_gemm). grid = min(#SMs, N/128),
/// 256 threads, 1 CTA/SM. PDL-aware (--exl3-pdl=1), graph-capturable (plain stream launch).
#[allow(clippy::too_many_arguments)]
pub(crate) fn lmh_gemm_launch(dev: &Arc<CudaDevice>, stream: cudarc::driver::sys::CUstream,
                              tr: u64, xh: u64, out: u64, svh: u64, m: usize, k: usize, n: usize,
                              bits: usize, nst: usize, fuse_had: bool) -> Result<bool> {
    use cudarc::driver::sys;
    if m == 0 || m > 16 || n == 0 || n % 128 != 0 || k == 0 || k % (16 * LMH_KSTEP) != 0 || k % 128 != 0 {
        return Ok(false);
    }
    let nst = lmh_nst_for(bits, nst);
    let mrows = if m <= 8 { 8 } else { 16 };
    let Some(name) = lmh_entry(bits, mrows, nst) else { return Ok(false); };
    let smem = lmh_smem_bytes(bits, mrows, nst, k);
    let Some(f) = lmh_fn(name, smem as u32) else { return Ok(false); };
    let grid = lmh_sm_count(dev).min((n / 128) as u32).max(1);
    let (mut p_tr, mut p_xh, mut p_out, mut p_svh) = (tr, xh, out, svh);
    let (mut p_m, mut p_k, mut p_n, mut p_f) = (m as i32, k as i32, n as i32, fuse_had as i32);
    let mut params: [*mut std::ffi::c_void; 8] = [
        &mut p_tr as *mut u64 as *mut _, &mut p_xh as *mut u64 as *mut _,
        &mut p_out as *mut u64 as *mut _, &mut p_svh as *mut u64 as *mut _,
        &mut p_m as *mut i32 as *mut _, &mut p_k as *mut i32 as *mut _,
        &mut p_n as *mut i32 as *mut _, &mut p_f as *mut i32 as *mut _,
    ];
    if pdl_on() {
        unsafe { launch_pss_raw(stream, f, (grid, 1, 1), (256, 1, 1), smem as u32, &mut params) }
            .with_context(|| format!("launch {name} (pdl)"))?;
    } else {
        let r = unsafe {
            sys::cuLaunchKernel(f, grid, 1, 1, 256, 1, 1, smem as u32, stream,
                                params.as_mut_ptr(), std::ptr::null_mut())
        };
        anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "launch {name} failed: {r:?}");
    }
    Ok(true)
}

/// --lmh-xcheck: the old pair (exl3_hmma_gemm [+ exl3_had_svh]) on side buffers from the SAME
/// xh, bit-diffed against the new path's output rows (`got`: logits if fused, else y_raw).
pub(super) fn lmh_xcheck_run(l: &Launcher, q: &Quad, xh: &CudaSlice<u16>, got: &CudaSlice<u16>, m: usize,
                  fused: bool) -> Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    static BAD: AtomicUsize = AtomicUsize::new(0);
    let n = q.n as usize;
    let mut yr = l.dev.alloc_zeros::<u16>(m * n)?;
    xqlaunch!(l, "exl3_hmma_gemm", ((q.n as u32) / 128, 1, 1), (256, 1, 1), 0,
              (&q.tr, xh, &mut yr, m as i32, q.k, q.n, q.k2))?;
    let refv = if fused {
        let mut yo = l.dev.alloc_zeros::<u16>(m * n)?;
        xqlaunch!(l, "exl3_had_svh", (m as u32 * (q.n as u32 / 128), 1, 1), (32, 1, 1), 0,
                  (&yr, &q.svh, &mut yo, q.n))?;
        l.dev.synchronize()?;
        l.dev.dtoh_sync_copy(&yo)?
    } else {
        l.dev.synchronize()?;
        l.dev.dtoh_sync_copy(&yr)?
    };
    let newv = l.dev.dtoh_sync_copy(got)?;
    let (mut mism, mut maxd, mut first) = (0usize, 0f32, None);
    for i in 0..m * n {
        if newv[i] != refv[i] {
            mism += 1;
            if first.is_none() { first = Some((i / n, i % n, newv[i], refv[i])); }
            let d = (f16::from_bits(newv[i]).to_f32() - f16::from_bits(refv[i]).to_f32()).abs();
            if d.is_nan() || d > maxd { maxd = if d.is_nan() { f32::INFINITY } else { d }; }
        }
    }
    let nz = refv[..m * n].iter().filter(|&&b| b != 0).count();
    let c = N.fetch_add(1, Ordering::SeqCst) + 1;
    if mism > 0 { BAD.fetch_add(1, Ordering::SeqCst); }
    if mism > 0 || c == 1 || c % 100 == 0 {
        crate::dprintln!("LMH_XCHECK: checks {c} bad {} | this: m {m} n {n} fused {} mism {mism} max|d| {maxd:e} \
                  ref_nonzero {nz} first(row,col,new,old) {first:?}",
                 BAD.load(Ordering::SeqCst), fused as u8);
    }
    Ok(())
}
