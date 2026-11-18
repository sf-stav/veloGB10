//! WP21 fused two-quad dense chain: enable / occupancy gates, boot log, the old-vs-new cross-checks and
//! sweeps, `wp21_fx_x2_raw` and `exl3_chain`. See `mod.rs` for the file map.

use super::*;

/// WP21: split-K fixup epilogue in exl3_hmma_gemm_ks(_x2) — DEFAULT ON. The last-arriving
/// CTA per N-tile does the ascending fp32 combine + H128 svh in-launch, so a ks > 1 chain is
/// had_suh + ONE GEMM launch (was + exl3_ks_combine + exl3_had_svh). Bitwise the same y.
/// Sites: exl3_chain_impl (single), exl3_chain_pair (x2), and REPRIME's wp22r1 row slabs (one
/// fused launch per <= 16-row slab; the slab loop's trailing exl3_had_svh is gone too). Opt-in
/// paths keep their own tails: W4/DENSE (--w4dense=1, its own in-kernel fixup) runs first,
/// PSK (--psk=1) keeps combine + svh; LMH (ks = 1) never reaches a split-K site.
/// --wp21-off=1 = the 3-launch chain tail (A/B + bisect escape).
/// --wp21-ks-off=<ks,ks,...> (diagnostics): the listed split counts keep the 3-launch tail —
/// a per-shape A/B without a rebuild (ks is a pure function of (K, N), so every width of a shape
/// takes the same path; both paths are bitwise equal anyway). Parsed once; malformed entries are
/// ignored and the parsed set is printed on first use.
///
/// Concurrency (WP27 capture-DAG forks + the other in-flight classes). The fixup's counter set is
/// `Quad.cnt` (one self re-arming u32 per N-tile, per weight matrix; shared only with W4/DENSE's
/// opt-in exl3_dks_* on the same Quad, which is a different launch of the SAME site, never both).
/// A counter set is therefore only ever hit by launches of ONE Quad, and no graph ever has two
/// launches of one Quad on parallel branches:
///  - GDN fork (verify + step graphs): in_qkv runs from the fork base, in_z from in_qkv's tail
///    (so the two chains are ordered, never concurrent; distinct Quads, and in_z has its own
///    z_xh / z_yraw / z_ws); the branches beside them (a|b GEMV + saves, conv, scan) hold no
///    counter. out_proj runs after the join.
///  - MoE fork (the pre-WP20 fallback sequence) and W4/MOE SHOVL (opt-in): the shared-expert
///    branch holds sh_gate|sh_up (x2: two counter slices, cnt0 != cnt1 enforced) then sh_down;
///    the routed branch holds only the router fold's private fold_cnt and sc.moe_cnt — no Quad.
///  - PLE fork: dropped in p3c. Draft graphs: no forks.
///  - HOST speculative pass, REPRIME graph, multi-lane rounds: stream-ordered on the ONE compute
///    stream (consecutive launches of a Quad complete — counters back at 0 — before the next
///    starts; under --exl3-pdl=1 XQ_PDL_ENTRY waits on the whole previous grid first).
/// A pair launch with both members on the same counter set (q0.cnt == q1.cnt) would double-count
/// arrivals; wp21_pair_ok refuses it (the 3-launch x2 tail runs instead).
fn wp21_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("wp21-off")).as_deref() != Ok("1"))
}

pub(super) fn wp21_on(ks: u32) -> bool {
    static KS_OFF: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    if !wp21_enabled() { return false; }
    let off = KS_OFF.get_or_init(|| {
        let v: Vec<u32> = crate::opts::var(crate::opt!("wp21-ks-off")).ok()
            .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
            .unwrap_or_default();
        if !v.is_empty() { crate::dprintln!("WP21: --wp21-ks-off -> 3-launch tail for ks in {v:?}"); }
        v
    });
    !off.contains(&ks)
}

/// WP21 eligibility of one member: one counter per 128-column tile of N (load-time sizing).
pub(super) fn wp21_cnt_ok(q: &Quad) -> bool {
    q.n > 0 && q.n % 128 == 0 && q.cnt.len() * 128 >= q.n as usize
}

/// WP21 eligibility of a same-input pair: both members' counters fit AND are distinct slices
/// (a shared set would take 2 * ks arrivals per tile — see the concurrency note at wp21_on).
pub(super) fn wp21_pair_ok(q0: &Quad, q1: &Quad, n: i32) -> bool {
    q0.n == n && q1.n == n && wp21_cnt_ok(q0) && wp21_cnt_ok(q1)
        && *q0.cnt.device_ptr() != *q1.cnt.device_ptr()
}

/// PCKG-ALL (2026-10-30, prime-batch xcheck root cause): the fused bodies (EXL3_FX_MEMBER,
/// exl3_hmma_gemm_ks_fx / _fx_x2) switch on k2 with cases 3..=14 — NO case 16. The 4.18bpw
/// pack's 8-bit MTP-head quad (k2 16) made every fused launch return without computing while
/// the per-lane path ran the plain kernel (case 16 there since D3): the first batched prime
/// of a 2-lane boot diverged from per-lane, the prime-batch xcheck FAILED and the engine
/// died (exit 70, owner's 4.18 TP=2 log). Refuse the fused arm at every serve site; the
/// plain 3-launch tail serves it (bitwise the same contract). Counted per k2 on /metrics
/// plus one boot WARN line — the psk_width_fallback pattern.
pub(super) fn wp21_fx_width_ok(q: &Quad) -> bool {
    const FX_K2: [i32; 7] = [3, 4, 6, 8, 10, 12, 14];
    if FX_K2.contains(&q.k2) {
        return true;
    }
    crate::metrics::exl3_wp21_width_fallback(q.k2);
    static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    if WARNED.set(()).is_ok() {
        crate::dprintln!("WP21: fused split-K refused at k2 {} (no fx body — the 8-bit-head class): \
                  serving the plain 3-launch tail (case 16 there; bitwise the same contract)", q.k2);
    }
    false
}

/// Load-time feature marker: which split-K chain tail this process runs (AGENTS §1a.4).
pub(super) fn wp21_log_boot() {
    if !wp21_enabled() {
        crate::dprintln!("EXL3-FWD WP21: OFF (--wp21-off=1) — split-K chains run the 3-launch tail \
                  (exl3_hmma_gemm_ks + exl3_ks_combine + exl3_had_svh)");
        return;
    }
    let mut notes: Vec<&str> = Vec::new();
    if dense::dense_on() { notes.push("W4/DENSE ON: its chains take the DENSE fixup, WP21 serves the rest"); }
    if !psk_off() { notes.push("PSK ON: PSK grids keep the combine + svh tail"); }
    if crate::opts::var_os(crate::opt!("wp21-ks-off")).is_some() { notes.push("--wp21-ks-off set: listed ks keep the 3-launch tail"); }
    if wp21_xcheck() { notes.push("--wp21-xcheck: eager fused calls diffed vs the 3-launch tail"); }
    crate::dprintln!("EXL3-FWD WP21: split-K fixup epilogue ON — ks > 1 chains (single / pair / REPRIME slabs) = had_suh + ONE \
              exl3_hmma_gemm_ks_fx(_x2) launch (last CTA per N-tile: ascending combine + H128 svh){}; \
              --wp21-off=1 = the 3-launch tail",
             if notes.is_empty() { String::new() } else { format!(" [{}]", notes.join("; ")) });
}

/// WP21 v2 receipt, printed once from the first EAGER fused call (skipped while capturing).
/// The fused kernels only pay at 4 CTAs/SM (64 regs, like exl3_hmma_gemm_ks): v2 carries
/// `.maxntid 256` WITHOUT a min-blocks cap (the (256,4) cap made ptxas degrade the k-loop, the
/// v1 regression), so the register count is the driver JIT's own choice — offline ptxas 13.0
/// picks 64. This line proves what the JIT actually built on the box.
pub(super) fn wp21_occupancy_once(l: &Launcher) {
    static DONE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    if DONE.get().is_some() || stream_capturing(l) { return; }
    DONE.get_or_init(|| {
        use cudarc::driver::sys;
        for name in ["exl3_hmma_gemm_ks_fx", "exl3_hmma_gemm_ks_fx_x2", "exl3_hmma_gemm_ks"] {
            let Some(f) = xq_raw_fn(name) else {
                crate::dprintln!("WP21_OCC: {name} raw fn unavailable");
                continue;
            };
            let (mut regs, mut bps): (std::ffi::c_int, std::ffi::c_int) = (-1, -1);
            unsafe {
                let _ = sys::cuFuncGetAttribute(&mut regs, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS, f);
                let _ = sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut bps, f, 256, 0);
            }
            println!("WP21_OCC: {name} regs {regs} blocks/SM {bps} -> {}",
                     if bps >= 4 { "OK" } else { "WARNING: < 4 CTAs/SM (fused path loses occupancy vs the 3-launch tail)" });
        }
    });
}

/// WP21 eager bitwise cross-check (--wp21-xcheck=1; add --exl3-no-graph=1 for full
/// coverage — calls issued while the stream is capturing are skipped, never synced).
/// Every fused call: fused y vs the 3-launch y on the live activations, counters back at 0
/// (WP21_XCHECK lines). First use of each chain / pair shape: a synthetic m = 1..=16 sweep —
/// fused vs 3-launch at every width, row-0 batch invariance, counters (WP21_SWEEP lines).
pub(super) fn wp21_xcheck() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("wp21-xcheck")).is_ok())
}

/// (mismatches, first mismatch index, max |a - b| as f16 values) over the first `n` elements.
fn wp21_diff(a: &[u16], b: &[u16], n: usize) -> (usize, Option<usize>, f32) {
    let (mut bad, mut first, mut maxd) = (0usize, None, 0f32);
    for i in 0..n {
        if a[i] != b[i] {
            bad += 1;
            if first.is_none() { first = Some(i); }
            let d = (f16::from_bits(a[i]).to_f32() - f16::from_bits(b[i]).to_f32()).abs();
            if !(d <= maxd) { maxd = d; } // NaN-propagating
        }
    }
    (bad, first, maxd)
}

fn wp21_report(kind: &str, m: usize, k: i32, n: i32, d: (usize, Option<usize>, f32), cnt_nz: usize) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    static BAD: AtomicUsize = AtomicUsize::new(0);
    let c = CALLS.fetch_add(1, Ordering::SeqCst) + 1;
    let fail = d.0 > 0 || cnt_nz > 0;
    if fail { BAD.fetch_add(1, Ordering::SeqCst); }
    if fail || c == 1 || c % 500 == 0 {
        crate::dprintln!("WP21_XCHECK: calls {c} bad {} (this: {kind} m {m} K {k} N {n} mism {} first {:?} maxdiff {:e} cnt_nonzero {cnt_nz})",
                 BAD.load(Ordering::SeqCst), d.0, d.1, d.2);
    }
}

/// Deterministic synthetic f16 activations in [-1, 1) for the WP21 width sweeps.
fn wp21_synth(len: usize, seed: u64) -> Vec<u16> {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    (0..len).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((s >> 40) as f32) / ((1u64 << 24) as f32);
        f16::from_f32(u * 2.0 - 1.0).to_bits()
    }).collect()
}

/// The pre-WP21 single-chain tail (xh already holds had_suh(x)): ks GEMM -> combine -> svh.
fn wp21_old_single(l: &Launcher, q: &Quad, xh: &CudaSlice<u16>, yraw: &mut CudaSlice<u16>,
                   y: &mut CudaSlice<u16>, m: usize, ws: &CudaSlice<f32>) -> Result<()> {
    xqlaunch!(l, "exl3_hmma_gemm_ks", ((q.n as u32) / 128, q.ks, 1), (256, 1, 1), 0,
             (&q.tr, xh, ws, m as i32, q.k, q.n, q.k2))?;
    let mn = (m * q.n as usize) as u32;
    xqlaunch!(l, "exl3_ks_combine", ((mn + 255) / 256, 1, 1), (256, 1, 1), 0,
             (ws, &mut *yraw, m as i32, q.n, q.ks as i32))?;
    xqlaunch!(l, "exl3_had_svh", (m as u32 * (q.n as u32 / 128), 1, 1), (32, 1, 1), 0,
             (&*yraw, &q.svh, &mut *y, q.n))
}

/// The pre-WP21 pair tail: ks GEMM x2 -> combine x2 -> svh x2.
#[allow(clippy::too_many_arguments)]
fn wp21_old_pair(l: &Launcher, q0: &Quad, q1: &Quad, xh0: &CudaSlice<u16>, xh1: &CudaSlice<u16>,
                 yraw0: &mut CudaSlice<u16>, yraw1: &mut CudaSlice<u16>,
                 ws0: &CudaSlice<f32>, ws1: &CudaSlice<f32>,
                 y0: &mut CudaSlice<u16>, y1: &mut CudaSlice<u16>, m: usize) -> Result<()> {
    let (k, n) = (q0.k, q0.n);
    xqlaunch!(l, "exl3_hmma_gemm_ks_x2", ((n as u32) / 128, q0.ks, 2), (256, 1, 1), 0,
             (&q0.tr, &q1.tr, xh0, xh1, ws0, ws1, m as i32, k, n, q0.k2))?;
    let mn = (m * n as usize) as u32;
    xqlaunch!(l, "exl3_ks_combine_x2", ((mn + 255) / 256, 1, 2), (256, 1, 1), 0,
             (ws0, ws1, &mut *yraw0, &mut *yraw1, m as i32, n, q0.ks as i32))?;
    xqlaunch!(l, "exl3_had_svh_x2", (m as u32 * (n as u32 / 128), 1, 2), (32, 1, 1), 0,
             (&*yraw0, &*yraw1, &q0.svh, &q1.svh, &mut *y0, &mut *y1, n))
}

/// WP21: exl3_hmma_gemm_ks_fx_x2 takes 16 parameters (cudarc's tuple launch stops at 12), so it
/// launches raw through the xq raw module — the xq_router_topk_route precedent — honoring the
/// same --exl3-pdl switch as xqlaunch!. Captures into graphs like any stream launch.
/// `p` = device addresses [tr0, tr1, xh0, xh1, ws0, ws1, svh0, svh1, y0, y1, cnt0, cnt1]; grid
/// (N/128, ks, 2). The served pair chain AND the binv probe (EXL3-WP21) launch through here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn wp21_fx_x2_raw(stream: cudarc::driver::sys::CUstream, mut p: [u64; 12], m: usize, k: i32, n: i32,
                             bits: i32, ks: u32) -> Result<()> {
    use std::ffi::c_void;
    static F: std::sync::OnceLock<usize> = std::sync::OnceLock::new(); // CUfunction, cached
    let fu = match F.get() {
        Some(&v) => v,
        None => {
            let f = xq_raw_fn("exl3_hmma_gemm_ks_fx_x2")
                .ok_or_else(|| anyhow::anyhow!("exl3_hmma_gemm_ks_fx_x2 raw fn unavailable (stale PTX?)"))?;
            *F.get_or_init(|| f as usize)
        }
    };
    anyhow::ensure!(p[10] != p[11], "WP21 x2: both members on one counter set");
    let f = fu as cudarc::driver::sys::CUfunction;
    let mut iv: [i32; 4] = [m as i32, k, n, bits];
    let mut args: [*mut c_void; 16] = [std::ptr::null_mut(); 16];
    for (i, v) in p.iter_mut().enumerate() { args[i] = v as *mut u64 as *mut c_void; }
    for (i, v) in iv.iter_mut().enumerate() { args[12 + i] = v as *mut i32 as *mut c_void; }
    let grid = ((n as u32) / 128, ks, 2u32);
    if pdl_on() {
        return unsafe { launch_pss_raw(stream, f, grid, (256, 1, 1), 0, &mut args) }
            .context("launch exl3_hmma_gemm_ks_fx_x2 (pdl)");
    }
    let r = unsafe {
        cudarc::driver::sys::cuLaunchKernel(f, grid.0, grid.1, grid.2, 256, 1, 1, 0, stream,
                                            args.as_mut_ptr(), std::ptr::null_mut())
    };
    anyhow::ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "exl3_hmma_gemm_ks_fx_x2 launch ({r:?})");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn wp21_launch_fx_x2(l: &Launcher, q0: &Quad, q1: &Quad, xh0: &CudaSlice<u16>, xh1: &CudaSlice<u16>,
                     ws0: &CudaSlice<f32>, ws1: &CudaSlice<f32>,
                     y0: &mut CudaSlice<u16>, y1: &mut CudaSlice<u16>, m: usize) -> Result<()> {
    let p: [u64; 12] = [
        *q0.tr.device_ptr() as u64, *q1.tr.device_ptr() as u64,
        *xh0.device_ptr() as u64, *xh1.device_ptr() as u64,
        *ws0.device_ptr() as u64, *ws1.device_ptr() as u64,
        *q0.svh.device_ptr() as u64, *q1.svh.device_ptr() as u64,
        *y0.device_ptr() as u64, *y1.device_ptr() as u64,
        *q0.cnt.device_ptr() as u64, *q1.cnt.device_ptr() as u64,
    ];
    wp21_fx_x2_raw(l.stream.stream, p, m, q0.k, q0.n, q0.k2, q0.ks)
}

fn wp21_cnt_nonzero(l: &Launcher, q: &Quad) -> Result<usize> {
    Ok(l.dev.dtoh_sync_copy(&q.cnt)?.iter().filter(|&&c| c != 0).count())
}

/// --wp21-xcheck, single chain: the fused launch just ran; diff it against the 3-launch
/// tail on the same xh (y is left holding the 3-launch result — equal when the check passes).
pub(super) fn wp21_check_single(l: &Launcher, q: &Quad, xh: &CudaSlice<u16>, yraw: &mut CudaSlice<u16>,
                     y: &mut CudaSlice<u16>, m: usize, ws: &CudaSlice<f32>) -> Result<()> {
    if stream_capturing(l) { return Ok(()); }
    l.dev.synchronize()?;
    let fx = l.dev.dtoh_sync_copy(&*y)?;
    let cnt_nz = wp21_cnt_nonzero(l, q)?;
    wp21_old_single(l, q, xh, yraw, y, m, ws)?;
    l.dev.synchronize()?;
    let old = l.dev.dtoh_sync_copy(&*y)?;
    wp21_report("chain", m, q.k, q.n, wp21_diff(&fx, &old, m * q.n as usize), cnt_nz);
    wp21_sweep_single(l, q)
}

/// --wp21-xcheck, REPRIME wp22r1 slabs (m > 16 rows): the fused per-slab launches just ran;
/// diff y against the pre-WP21 slab loop on the same xh — per slab ks GEMM + combine into its
/// yraw rows, then ONE exl3_had_svh over all m rows (y is left holding that result).
#[allow(clippy::too_many_arguments)]
pub(super) fn wp21_check_slabs(l: &Launcher, q: &Quad, xh: &CudaSlice<u16>, yraw: &mut CudaSlice<u16>,
                    y: &mut CudaSlice<u16>, m: usize, ws: &CudaSlice<f32>) -> Result<()> {
    if stream_capturing(l) { return Ok(()); }
    let (k, n) = (q.k as usize, q.n as usize);
    l.dev.synchronize()?;
    let fx = l.dev.dtoh_sync_copy(&*y)?;
    let cnt_nz = wp21_cnt_nonzero(l, q)?;
    let (xh_p, yr_p) = (*xh.device_ptr() as u64, *yraw.device_ptr() as u64);
    for (r0, ms) in wp22r1_slabs(m) {
        xqlaunch!(l, "exl3_hmma_gemm_ks", ((q.n as u32) / 128, q.ks, 1), (256, 1, 1), 0,
                 (&q.tr, xh_p + (r0 * k * 2) as u64, ws, ms as i32, q.k, q.n, q.k2))?;
        xqlaunch!(l, "exl3_ks_combine", (((ms * n) as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                 (ws, yr_p + (r0 * n * 2) as u64, ms as i32, q.n, q.ks as i32))?;
    }
    xqlaunch!(l, "exl3_had_svh", ((m * (n / 128)) as u32, 1, 1), (32, 1, 1), 0, (&*yraw, &q.svh, &mut *y, q.n))?;
    l.dev.synchronize()?;
    let old = l.dev.dtoh_sync_copy(&*y)?;
    wp21_report("slabs", m, q.k, q.n, wp21_diff(&fx, &old, m * n), cnt_nz);
    wp21_sweep_single(l, q)
}

/// --wp21-xcheck, pair: fused x2 launch vs the 3-launch x2 tail (both members).
#[allow(clippy::too_many_arguments)]
pub(super) fn wp21_check_pair(l: &Launcher, q0: &Quad, q1: &Quad,
                   xh0: &CudaSlice<u16>, yraw0: &mut CudaSlice<u16>, ws0: &CudaSlice<f32>,
                   xh1: &CudaSlice<u16>, yraw1: &mut CudaSlice<u16>, ws1: &CudaSlice<f32>,
                   y0: &mut CudaSlice<u16>, y1: &mut CudaSlice<u16>, m: usize) -> Result<()> {
    if stream_capturing(l) { return Ok(()); }
    let mn = m * q0.n as usize;
    l.dev.synchronize()?;
    let (f0, f1) = (l.dev.dtoh_sync_copy(&*y0)?, l.dev.dtoh_sync_copy(&*y1)?);
    let cnt_nz = wp21_cnt_nonzero(l, q0)? + wp21_cnt_nonzero(l, q1)?;
    wp21_old_pair(l, q0, q1, xh0, xh1, yraw0, yraw1, ws0, ws1, y0, y1, m)?;
    l.dev.synchronize()?;
    let (o0, o1) = (l.dev.dtoh_sync_copy(&*y0)?, l.dev.dtoh_sync_copy(&*y1)?);
    let (d0, d1) = (wp21_diff(&f0, &o0, mn), wp21_diff(&f1, &o1, mn));
    let d = (d0.0 + d1.0, d0.1.or(d1.1.map(|i| i + mn)), if d0.2 >= d1.2 { d0.2 } else { d1.2 });
    wp21_report("pair", m, q0.k, q0.n, d, cnt_nz);
    wp21_sweep_pair(l, q0, q1)
}

/// First use per (k, n, bits, ks): synthetic m = 1..=16 sweep, fused vs 3-launch + row-0
/// batch invariance + counters re-armed.
fn wp21_sweep_single(l: &Launcher, q: &Quad) -> Result<()> {
    static SEEN: std::sync::Mutex<Vec<(i32, i32, i32, u32)>> = std::sync::Mutex::new(Vec::new());
    let key = (q.k, q.n, q.bits, q.ks);
    {
        let mut s = SEEN.lock().unwrap();
        if s.contains(&key) { return Ok(()); }
        s.push(key);
    }
    let (k, n) = (q.k as usize, q.n as usize);
    let x = l.dev.htod_sync_copy(&wp21_synth(16 * k, ((k as u64) << 32) ^ n as u64))?;
    let mut xh = l.dev.htod_sync_copy(&vec![0u16; 16 * k])?;
    let mut yraw = l.dev.htod_sync_copy(&vec![0u16; 16 * n])?;
    let mut ya = l.dev.htod_sync_copy(&vec![0u16; 16 * n])?;
    let mut yb = l.dev.htod_sync_copy(&vec![0u16; 16 * n])?;
    let ws = l.dev.htod_sync_copy(&vec![0f32; q.ks as usize * 16 * n])?;
    let (mut bad, mut binv_bad, mut cnt_bad) = (0usize, 0usize, 0usize);
    let mut row0: Option<Vec<u16>> = None;
    for m in 1..=16usize {
        xqlaunch!(l, "exl3_had_suh", (m as u32 * (q.k as u32 / 128), 1, 1), (32, 1, 1), 0,
                 (&x, &q.suh, &mut xh, q.k))?;
        xqlaunch!(l, "exl3_hmma_gemm_ks_fx", ((q.n as u32) / 128, q.ks, 1), (256, 1, 1), 0,
                 (&q.tr, &xh, &ws, &q.svh, &mut ya, &q.cnt, m as i32, q.k, q.n, q.k2))?;
        l.dev.synchronize()?;
        cnt_bad += wp21_cnt_nonzero(l, q)?;
        wp21_old_single(l, q, &xh, &mut yraw, &mut yb, m, &ws)?;
        l.dev.synchronize()?;
        let (a, b) = (l.dev.dtoh_sync_copy(&ya)?, l.dev.dtoh_sync_copy(&yb)?);
        bad += wp21_diff(&a, &b, m * n).0;
        match &row0 {
            None => row0 = Some(a[..n].to_vec()),
            Some(r) => if r[..] != a[..n] { binv_bad += 1; },
        }
    }
    let ok = bad == 0 && binv_bad == 0 && cnt_bad == 0;
    println!("WP21_SWEEP: chain K {k} N {n} bits {} ks {} m=1..16 mism {bad} binv_bad {binv_bad} cnt_nonzero {cnt_bad} -> {}",
             q.bits, q.ks, if ok { "PASS" } else { "FAIL" });
    Ok(())
}

/// Pair twin of wp21_sweep_single (x2 kernels, both members).
fn wp21_sweep_pair(l: &Launcher, q0: &Quad, q1: &Quad) -> Result<()> {
    static SEEN: std::sync::Mutex<Vec<(i32, i32, i32, u32)>> = std::sync::Mutex::new(Vec::new());
    let key = (q0.k, q0.n, q0.bits, q0.ks);
    {
        let mut s = SEEN.lock().unwrap();
        if s.contains(&key) { return Ok(()); }
        s.push(key);
    }
    let (k, n) = (q0.k as usize, q0.n as usize);
    let zb = |len: usize| l.dev.htod_sync_copy(&vec![0u16; len]);
    let x = l.dev.htod_sync_copy(&wp21_synth(16 * k, ((n as u64) << 32) ^ k as u64))?;
    let (mut xh0, mut xh1) = (zb(16 * k)?, zb(16 * k)?);
    let (mut yr0, mut yr1) = (zb(16 * n)?, zb(16 * n)?);
    let (mut ya0, mut ya1) = (zb(16 * n)?, zb(16 * n)?);
    let (mut yb0, mut yb1) = (zb(16 * n)?, zb(16 * n)?);
    let ws0 = l.dev.htod_sync_copy(&vec![0f32; q0.ks as usize * 16 * n])?;
    let ws1 = l.dev.htod_sync_copy(&vec![0f32; q0.ks as usize * 16 * n])?;
    let (mut bad, mut binv_bad, mut cnt_bad) = (0usize, 0usize, 0usize);
    let mut row0: Option<(Vec<u16>, Vec<u16>)> = None;
    for m in 1..=16usize {
        xqlaunch!(l, "exl3_had_suh_x2", (m as u32 * (q0.k as u32 / 128), 1, 2), (32, 1, 1), 0,
                 (&x, &q0.suh, &q1.suh, &mut xh0, &mut xh1, q0.k))?;
        wp21_launch_fx_x2(l, q0, q1, &xh0, &xh1, &ws0, &ws1, &mut ya0, &mut ya1, m)?;
        l.dev.synchronize()?;
        cnt_bad += wp21_cnt_nonzero(l, q0)? + wp21_cnt_nonzero(l, q1)?;
        wp21_old_pair(l, q0, q1, &xh0, &xh1, &mut yr0, &mut yr1, &ws0, &ws1, &mut yb0, &mut yb1, m)?;
        l.dev.synchronize()?;
        let (a0, a1) = (l.dev.dtoh_sync_copy(&ya0)?, l.dev.dtoh_sync_copy(&ya1)?);
        let (b0, b1) = (l.dev.dtoh_sync_copy(&yb0)?, l.dev.dtoh_sync_copy(&yb1)?);
        bad += wp21_diff(&a0, &b0, m * n).0 + wp21_diff(&a1, &b1, m * n).0;
        match &row0 {
            None => row0 = Some((a0[..n].to_vec(), a1[..n].to_vec())),
            Some((r0, r1)) => if r0[..] != a0[..n] || r1[..] != a1[..n] { binv_bad += 1; },
        }
    }
    let ok = bad == 0 && binv_bad == 0 && cnt_bad == 0;
    println!("WP21_SWEEP: pair K {k} N {n} bits {} ks {} m=1..16 mism {bad} binv_bad {binv_bad} cnt_nonzero {cnt_bad} -> {}",
             q0.bits, q0.ks, if ok { "PASS" } else { "FAIL" });
    Ok(())
}

pub(super) fn exl3_chain(l: &Launcher, q: &Quad, x: &CudaSlice<u16>, xh: &mut CudaSlice<u16>,
              yraw: &mut CudaSlice<u16>, y: &mut CudaSlice<u16>, m: usize,
              ws: &CudaSlice<f32>) -> Result<()> {
    exl3_chain_impl(l, q, x, xh, yraw, y, m, ws, None)
}
