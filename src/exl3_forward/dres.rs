//! LANEWIDE-DRES (2026-11-12): `--exl3-dense-resident on|off` (default off).
//!
//! A prefill-class pass (a chain of m >= --exl3-recon-min rows, 17 by default) serves every dense trellis module
//! through the reconstruct + cuBLASLt arm (`chain_wide_gemm`): `exl3_reconstruct` decodes the whole matrix into
//! the prefill scratch's fp16 staging (`exl3_decode_dump8`), then the vendor GEMM reads it. That decode is the
//! same work on every pass (~31 ms per pass on Flash-Next 3.05bpw at TP=1, measured 2026-11-12), whatever the row
//! count. With the flag on, each such matrix is decoded ONCE into its own device buffer (first use: the boot's
//! prefill warm-up runs every width class, so in serving this happens at boot) and every later pass hands that
//! buffer to the same GEMM instead of decoding again. The GEMM's inputs are the same bytes (the same decode
//! kernel wrote them) at a buffer of the same alignment, so the output is the per-pass path's bit for bit
//! (`--probe-dense-resident` checks a fixed prompt's state and first token both ways).
//!
//! Memory: a matrix is kept only while the device keeps `--exl3-dense-resident-reserve-gib` (default 2) GiB free after it; past that (or inside a graph
//! capture) the module keeps today's per-pass decode, with one line and the
//! `velogb10_exl3_dense_resident_fallback_total{reason}` count. Never a refusal. Only the dense chains: the
//! expert reconstruct (an expert with >= 17 rows in one chunk) keeps the per-pass decode.

use super::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

/// Free device memory a kept matrix must leave behind (GiB): `--exl3-dense-resident-reserve-gib`.
fn reserve_gib() -> u64 {
    crate::opts::ranged(crate::opt!("exl3-dense-resident-reserve-gib")).map(|v| v as u64).unwrap_or(2)
}

static ON: AtomicBool = AtomicBool::new(false);
static FULL: AtomicBool = AtomicBool::new(false);
static BYTES: AtomicU64 = AtomicU64::new(0);
static MATS: AtomicU64 = AtomicU64::new(0);
static HITS: AtomicU64 = AtomicU64::new(0);
static DECODES: AtomicU64 = AtomicU64::new(0);
static FB_MEM: AtomicU64 = AtomicU64::new(0);
static FB_CAPTURE: AtomicU64 = AtomicU64::new(0);
static FB_ALLOC: AtomicU64 = AtomicU64::new(0);

struct Kept(CudaSlice<u16>);
// The buffers are written once on the compute stream and afterwards only read by kernels on it; the map is
// only touched under its mutex (same contract as `unsafe impl Send for FwdModel`).
unsafe impl Send for Kept {}

fn map() -> &'static Mutex<HashMap<u64, Kept>> {
    static M: OnceLock<Mutex<HashMap<u64, Kept>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Boot: the registry value of `--exl3-dense-resident` (call once, before the first prefill).
pub fn init_from_opts() {
    ON.store(crate::opts::on(crate::opt!("exl3-dense-resident")), Relaxed);
}

pub fn on() -> bool { ON.load(Relaxed) }

/// Test / probe hook: switch the arm without the registry.
pub fn set_on(v: bool) { ON.store(v, Relaxed); }

/// The fp16 [K, N] buffer `chain_wide_gemm` hands to the GEMM for `q`: the kept one (decoded at its first use),
/// or None = decode into the staging as today (flag off, no room, a graph capture).
pub(super) fn kept_ptr(l: &Launcher, q: &Quad, rc: u64) -> Result<Option<u64>> {
    if !on() { return Ok(None); }
    let key = *q.tr.device_ptr() as u64;
    let mut m = map().lock().unwrap();
    if let Some(k) = m.get(&key) {
        HITS.fetch_add(1, Relaxed);
        return Ok(Some(*k.0.device_ptr() as u64));
    }
    if stream_capturing(l) { FB_CAPTURE.fetch_add(1, Relaxed); return Ok(None); }
    if FULL.load(Relaxed) { FB_MEM.fetch_add(1, Relaxed); return Ok(None); }
    let n_el = (q.k as usize) * (q.n as usize);
    let need = (n_el * 2) as u64;
    let rg = reserve_gib();
    let free = cudarc::driver::result::mem_get_info().map(|x| x.0 as u64).unwrap_or(0);
    if free < need + (rg << 30) {
        FULL.store(true, Relaxed);
        FB_MEM.fetch_add(1, Relaxed);
        crate::rprintln!("[exl3] dense-resident: device memory exhausted after {} kept matrices ({:.2} GiB; {:.2} GiB free, {rg} GiB \
reserve --exl3-dense-resident-reserve-gib): the remaining dense modules keep the per-pass decode (velogb10_exl3_dense_resident_fallback_total{{reason=\"memory\"}})",
                         MATS.load(Relaxed), BYTES.load(Relaxed) as f64 / (1u64 << 30) as f64, free as f64 / (1u64 << 30) as f64);
        return Ok(None);
    }
    let buf = match l.dev.alloc_zeros::<u16>(n_el) {
        Ok(b) => b,
        Err(e) => {
            FULL.store(true, Relaxed);
            FB_ALLOC.fetch_add(1, Relaxed);
            crate::rprintln!("[exl3] dense-resident: allocation of {need} B failed ({e:?}): the remaining dense modules keep the per-pass decode");
            return Ok(None);
        }
    };
    let p = *buf.device_ptr() as u64;
    exl3_reconstruct(l, key, rc, p, q.k, q.n, q.k2)?;
    DECODES.fetch_add(1, Relaxed);
    m.insert(key, Kept(buf));
    MATS.fetch_add(1, Relaxed);
    BYTES.fetch_add(need, Relaxed);
    Ok(Some(p))
}

/// (matrices kept, bytes kept)
pub fn kept() -> (u64, u64) { (MATS.load(Relaxed), BYTES.load(Relaxed)) }

/// The boot line (after the prefill warm-up): None when the flag is off.
pub fn boot_line() -> Option<String> {
    if !on() { return None; }
    let (n, b) = kept();
    let fb = FB_MEM.load(Relaxed) + FB_ALLOC.load(Relaxed);
    Some(if n == 0 && fb > 0 {
        format!("[exl3] dense-resident: asked (--exl3-dense-resident on), runs OFF: no room for one decoded dense matrix with {} GiB \
left free (--exl3-dense-resident-reserve-gib); prefill-class passes keep the per-pass decode (velogb10_exl3_dense_resident_fallback_total)", reserve_gib())
    } else if n == 0 {
        // ULTRA: nothing to keep (an FP4-Local pack serves its MXFP8 dense weights as stored: no EXL3 trellis matrix goes
        // through the reconstruct path), or the warm-up that keeps them was skipped: say so instead of "ON: 0 matrices"
        "[exl3] dense-resident: asked (--exl3-dense-resident on), nothing kept at boot: this pack has no EXL3 dense trellis matrix \
that prefill-class passes decode (an FP4-Local pack serves its MXFP8 dense weights as stored), or the boot's prefill warm-up was skipped \
(--exl3-no-prefill-warmup: matrices are then kept on first use). Output and speed are those of the flag off".to_string()
    } else {
        format!("[exl3] dense-resident: ON (--exl3-dense-resident on): {n} dense matrices decoded once to fp16 and kept ({:.2} GiB device \
memory){}; prefill-class passes (>= --exl3-recon-min rows) skip the per-pass decode, bit-identical output. Counters velogb10_exl3_dense_resident_*",
                b as f64 / (1u64 << 30) as f64,
                if fb > 0 { format!(", {fb} more could not be kept (memory reserve {} GiB, --exl3-dense-resident-reserve-gib): those keep the per-pass decode", reserve_gib()) } else { String::new() })
    })
}

pub fn metrics_append(o: &mut String) {
    let (n, b) = kept();
    o.push_str("# HELP velogb10_exl3_dense_resident_state --exl3-dense-resident on this boot: 0 = off, 1 = on.\n# TYPE velogb10_exl3_dense_resident_state gauge\n");
    o.push_str(&format!("velogb10_exl3_dense_resident_state {}\n", on() as u8));
    o.push_str("# HELP velogb10_exl3_dense_resident_matrices Dense trellis matrices kept decoded (fp16) on the device.\n# TYPE velogb10_exl3_dense_resident_matrices gauge\n");
    o.push_str(&format!("velogb10_exl3_dense_resident_matrices {n}\n"));
    o.push_str("# HELP velogb10_exl3_dense_resident_bytes Device memory of the kept fp16 dense matrices.\n# TYPE velogb10_exl3_dense_resident_bytes gauge\n");
    o.push_str(&format!("velogb10_exl3_dense_resident_bytes {b}\n"));
    o.push_str("# HELP velogb10_exl3_dense_resident_hits_total Prefill-class dense GEMMs served from a kept matrix (the per-pass decode skipped).\n# TYPE velogb10_exl3_dense_resident_hits_total counter\n");
    o.push_str(&format!("velogb10_exl3_dense_resident_hits_total {}\n", HITS.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_dense_resident_fallback_total Prefill-class dense GEMMs that decoded per pass although the flag is on, by reason (memory = the reserve would be crossed, alloc = the allocation failed, capture = inside a graph capture).\n# TYPE velogb10_exl3_dense_resident_fallback_total counter\n");
    for (r, v) in [("memory", &FB_MEM), ("alloc", &FB_ALLOC), ("capture", &FB_CAPTURE)] {
        o.push_str(&format!("velogb10_exl3_dense_resident_fallback_total{{reason=\"{r}\"}} {}\n", v.load(Relaxed)));
    }
}

/// `--probe-dense-resident --model-dir <pack> --prompt-ids <file> [--prefill-chunk C]`: the prompt is prefilled in
/// chunks of C (default 2048) on slot 0 twice from a reset slot, per-pass decode then kept matrices; the
/// recurrent state (GDN, conv, PLE ring, head-fill carry), trunk KV rows and the first token (greedy argmax of
/// the last row) must be bit-identical. Then the per-pass arm once more (kept buffers in place) as a control.
pub fn probe_dense_resident(a: &FwdArgs) -> Result<bool> {
    let model = FwdModel::load(&a.dir, 1, a.max_pos)?;
    anyhow::ensure!(model.tp.is_none(), "--probe-dense-resident runs on one box");
    let mut sc = Scratch::new(&model.dev, &model.cfg, 16, model.cfg.rotary_dim, model.max_pos)?;
    let ids = gate_prompt_ids(a)?;
    let n = ids.len();
    anyhow::ensure!(n >= 32, "--probe-dense-resident: the prompt needs at least 32 tokens");
    let cw = a.prefill_chunk.unwrap_or(2048);
    let mut psc = model.prefill_scratch(cw)?;
    let mut bufs = CpvBufs::new();
    model.set_sampling(&mut sc, &[None])?;
    model.set_penalties(&mut sc, &[None])?;
    let mut run = |kept: bool| -> Result<(RecurCkpt, Vec<Vec<f32>>, i32, f64)> {
        set_on(kept);
        model.reset_slot(0)?;
        model.dev.synchronize()?;
        let t0 = std::time::Instant::now();
        let mut pos = 0usize;
        while pos < n {
            let end = (pos + cw).min(n);
            let toks: Vec<i32> = ids[pos..end].iter().map(|&t| t as i32).collect();
            model.cpv_chunk(&mut psc, &mut bufs, &toks, pos, 0, true)?;
            pos = end;
        }
        model.dev.synchronize()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let last = (n - 1) % cw;
        let xs = &bufs.stash.as_ref().unwrap().xs;
        let id = model.cpv_head_group(&mut sc, xs, last, 1, &[None])?[0];
        let mut ck = model.ckpt_alloc()?;
        model.ckpt_save(0, &psc, &mut ck)?;
        let kv = snap_slot0_kv(&model, n)?;
        Ok((ck, kv, id, ms))
    };
    let (ck0, kv0, id0, ms0) = run(false)?;
    let (ck1, kv1, id1, ms1) = run(true)?;
    let (n_k, b_k) = kept();
    let (ck2, kv2, id2, ms2) = run(true)?;
    let (ck3, _kv3, id3, ms3) = run(false)?;
    set_on(false);
    let kv_eq = |x: &Vec<Vec<f32>>, y: &Vec<Vec<f32>>| x.len() == y.len()
        && x.iter().zip(y).all(|(p, q)| p.len() == q.len() && p.iter().zip(q).all(|(a, b)| a.to_bits() == b.to_bits()));
    let d1 = model.ckpt_diff(&ck0, &ck1)?;
    let d2 = model.ckpt_diff(&ck0, &ck2)?;
    let d3 = model.ckpt_diff(&ck0, &ck3)?;
    let ok = d1.exact() && d2.exact() && d3.exact() && kv_eq(&kv0, &kv1) && kv_eq(&kv0, &kv2) && id0 == id1 && id0 == id2 && id0 == id3;
    crate::dprintln!("DENSE_RESIDENT probe: prompt {n} tokens in chunks of {cw}; kept {n_k} matrices ({:.2} GiB)", b_k as f64 / (1u64 << 30) as f64);
    crate::dprintln!("  per-pass decode: first token {id0}, {ms0:.1} ms");
    crate::dprintln!("  kept (first use, decodes): first token {id1}, {ms1:.1} ms, state {}, KV {}", d1.line(), if kv_eq(&kv0, &kv1) { "EQUAL" } else { "DIFFER" });
    crate::dprintln!("  kept (served from kept):   first token {id2}, {ms2:.1} ms, state {}, KV {}", d2.line(), if kv_eq(&kv0, &kv2) { "EQUAL" } else { "DIFFER" });
    crate::dprintln!("  per-pass decode again:     first token {id3}, {ms3:.1} ms, state {}", d3.line());
    crate::dprintln!("RESULT: {}", if ok { "DENSE_RESIDENT_EXACT" } else { "DENSE_RESIDENT_MISMATCH" });
    Ok(ok)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_is_registered_cli_only_off_by_default_and_spmd() {
        // V0710: the row's default is "auto" (on for the Flash-Next family only, src/family_defaults.rs);
        // an unset value still reads as off wherever the resolver did not run
        let o = crate::opt!("exl3-dense-resident").def();
        assert_eq!((o.def, o.env), ("auto", ""));
        assert!(matches!(o.ty, crate::opts::Ty::Bool));
        assert!(matches!(o.scope, crate::opts::Scope::Spmd), "rides the TP config to the nodes");
        let p = crate::opt!("probe-dense-resident").def();
        assert_eq!((p.def, p.env), ("off", ""));
    }

    #[test]
    fn boot_line_and_metrics() {
        set_on(false);
        assert!(boot_line().is_none());
        set_on(true);
        // (nothing kept in a unit test: 82f14978's "nothing kept at boot" line, not "ON: 0 dense matrices")
        assert!(boot_line().unwrap().contains("dense-resident: asked (--exl3-dense-resident on), nothing kept at boot"));
        let mut o = String::new();
        metrics_append(&mut o);
        for s in ["velogb10_exl3_dense_resident_state 1", "velogb10_exl3_dense_resident_matrices ", "velogb10_exl3_dense_resident_bytes ",
                  "velogb10_exl3_dense_resident_hits_total ", "reason=\"memory\"", "reason=\"alloc\"", "reason=\"capture\""] {
            assert!(o.contains(s), "missing {s}");
        }
        set_on(false);
    }
}

// =============================================================================================
// LANEWIDE-MID (2026-11-13): `--exl3-mid-gemm on|off` (default off). A dense chain of 17..=MID_MAX rows takes the
// split-K trellis body with MT m16 tiles per CTA (`exl3_hmma_gemm_ks_mid`): the packed EXL3 weights are read and
// decoded ONCE per pass for every row (no fp16 reconstruct, no kept fp16 copy), and each row's arithmetic is the
// decode-class (m <= 16) chain's: the same ks split, the same ascending k16 mma sequence, the same ascending combine
// (`exl3_ks_combine`) and svh. A row's result is therefore bit-identical to the decode-class chain's for that row and
// independent of how many rows share the pass (`--probe-mid-gemm` checks both on every dense trunk matrix).

/// The largest chain the mid body takes (rows); above it the reconstruct / kept arm serves (measured: at 96 and 128
/// rows the mid body cost 40-49 ms per pass against 31-32 ms for the kept fp16 matrices).
pub const MID_MAX: usize = 64;

static MID_ON: AtomicBool = AtomicBool::new(false);
static MID_HITS: AtomicU64 = AtomicU64::new(0);
static MID_ROWS: AtomicU64 = AtomicU64::new(0);

pub fn mid_init_from_opts() {
    MID_ON.store(crate::opts::on(crate::opt!("exl3-mid-gemm")), Relaxed);
    EXACT_ON.store(crate::opts::on(crate::opt!("lane-wide-exact")), Relaxed);
}

// LANEWIDE-EXACT (2026-11-13): `--lane-wide-exact on|off`. In a wide verify the small fp16 projections (the hc
// down / up of every mixing site, GDN a / b, the PLE key / value) pick their kernel by the pass's row count: below
// --exl3-recon-min rows the fixed-sequence tail GEMM, from it cuBLASLt with the heuristic's plan for that row count
// (split-K and tiling vary with M). A lane's rows therefore changed with the other lanes sharing its pass
// (--probe-lane-wide-xclass, XCLASS_TAPS: the first differing kernel is GDN b at layer 0, then the hc down / up mix).
// With it on, a wide pass runs those projections through cuBLASLt with ONE algo per shape at every row count (the
// algo of a fixed row count's plan, `lwv::wide_pin`, run on the pass's m-row layouts): --probe-lane-wide-pin checks
// every row bit for bit at m 1..144 and row offsets 0..16 against the pinned 160-row call. The tail GEMM is also
// row-independent but cost up to 4 ms per pass on the PLE key at 64 rows. Prompt prefill is untouched (its scratch
// never sets the switch). Together with --exl3-mid-gemm on (the dense chains) the probe's lane rows are bit-identical
// inside any pass.
static EXACT_ON: AtomicBool = AtomicBool::new(false);
static EXACT_PASSES: AtomicU64 = AtomicU64::new(0);
pub fn exact_on() -> bool { EXACT_ON.load(Relaxed) }
pub fn set_exact(v: bool) { EXACT_ON.store(v, Relaxed); }
pub(super) fn exact_pass() { EXACT_PASSES.fetch_add(1, Relaxed); }
pub fn mid_on() -> bool { MID_ON.load(Relaxed) }
pub fn set_mid(v: bool) { MID_ON.store(v, Relaxed); }

/// MT (m16 tiles per CTA) for a chain of m rows: always one m-supertile (the trellis is decoded once per pass).
pub fn mid_mt(m: usize) -> usize { if m <= 32 { 2 } else { 4 } }

pub(super) fn mid_fits(q: &Quad, m: usize) -> bool {
    mid_on() && m > 16 && m <= MID_MAX && q.fx.is_none() && q.ks >= 1 && q.n % 128 == 0 && q.k % 16 == 0
        && ((q.k as u32) / 16) % q.ks == 0 && matches!(q.k2, 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 | 14 | 16)
}

/// Launch the mid body + the ascending combine into `yraw` (pre-svh). Ok(false) = the staging cannot hold the
/// partials (the caller takes the reconstruct arm).
pub(super) fn mid_gemm(l: &Launcher, q: &Quad, xh: &CudaSlice<u16>, yraw: &mut CudaSlice<u16>, m: usize, r: &ReconBufs) -> Result<bool> {
    let (k, n, ks) = (q.k as usize, q.n as usize, q.ks as usize);
    let need_f32 = ks * m * n;
    if need_f32 * 2 > r.ws_len || m * n > yraw.len() { return Ok(false); }
    let mt = mid_mt(m);
    let msup = m.div_ceil(16 * mt);
    xqlaunch!(l, "exl3_hmma_gemm_ks_mid", ((n / 128) as u32, ks as u32, msup as u32), (256, 1, 1), 0,
              (&q.tr, xh, r.ws, m as i32, k as i32, n as i32, q.k2, mt as i32))?;
    let mn = (m * n) as u32;
    xqlaunch!(l, "exl3_ks_combine", ((mn + 255) / 256, 1, 1), (256, 1, 1), 0,
              (r.ws, &mut *yraw, m as i32, n as i32, ks as i32))?;
    MID_HITS.fetch_add(1, Relaxed);
    MID_ROWS.fetch_add(m as u64, Relaxed);
    Ok(true)
}

pub fn mid_metrics_append(o: &mut String) {
    o.push_str("# HELP velogb10_exl3_mid_gemm_state --exl3-mid-gemm on this boot: 0 = off, 1 = on.\n# TYPE velogb10_exl3_mid_gemm_state gauge\n");
    o.push_str(&format!("velogb10_exl3_mid_gemm_state {}\n", mid_on() as u8));
    o.push_str("# HELP velogb10_exl3_mid_gemm_chains_total Dense chains of 17..64 rows served by the packed-weight mid body.\n# TYPE velogb10_exl3_mid_gemm_chains_total counter\n");
    o.push_str(&format!("velogb10_exl3_mid_gemm_chains_total {}\n", MID_HITS.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_mid_gemm_rows_total Rows of those chains.\n# TYPE velogb10_exl3_mid_gemm_rows_total counter\n");
    o.push_str(&format!("velogb10_exl3_mid_gemm_rows_total {}\n", MID_ROWS.load(Relaxed)));
    o.push_str("# HELP velogb10_lane_wide_exact_state --lane-wide-exact on this boot: 0 = off, 1 = on.\n# TYPE velogb10_lane_wide_exact_state gauge\n");
    o.push_str(&format!("velogb10_lane_wide_exact_state {}\n", exact_on() as u8));
    o.push_str("# HELP velogb10_lane_wide_exact_passes_total Wide verify passes whose small projections ran the pinned per-shape cuBLASLt algos (row-independent).\n# TYPE velogb10_lane_wide_exact_passes_total counter\n");
    o.push_str(&format!("velogb10_lane_wide_exact_passes_total {}\n", EXACT_PASSES.load(Relaxed)));
    // POSTURE1 --exl3-sel-tc: the prefill sparse gather on tensor cores
    o.push_str("# HELP velogb10_exl3_sel_tc_state --exl3-sel-tc on this boot: 0 = off, 1 = on.\n# TYPE velogb10_exl3_sel_tc_state gauge\n");
    o.push_str(&format!("velogb10_exl3_sel_tc_state {}\n", super::sel_tc_on() as u8));
    o.push_str("# HELP velogb10_exl3_sel_tc_calls_total Prefill sparse-gather launches served by the tensor-core kernel (one per attention layer per chunk).\n# TYPE velogb10_exl3_sel_tc_calls_total counter\n");
    o.push_str(&format!("velogb10_exl3_sel_tc_calls_total {}\n", super::SEL_TC_CALLS.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_sel_tc8_calls_total Of those, the launches served by xq_attn_sel_tcu8 (6 query heads per key head: Flash-Next at TP=4, 8 prompt tokens per block).\n# TYPE velogb10_exl3_sel_tc8_calls_total counter\n");
    o.push_str(&format!("velogb10_exl3_sel_tc8_calls_total {}\n", super::SEL_TC8_CALLS.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_sel_tc_rows_total Prompt rows (summed over layers) whose sparse gather ran on the tensor cores.\n# TYPE velogb10_exl3_sel_tc_rows_total counter\n");
    o.push_str(&format!("velogb10_exl3_sel_tc_rows_total {}\n", super::SEL_TC_ROWS.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_sel_tc_fallback 1 = --exl3-sel-tc on was asked but this boot cannot serve it (KV cache not 8-bit or a head shape it does not take; the boot line says why): today's gather runs.\n# TYPE velogb10_exl3_sel_tc_fallback gauge\n");
    o.push_str(&format!("velogb10_exl3_sel_tc_fallback {}\n", super::SEL_TC_FALLBACK.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_sel_tc_fallback_calls_total Prefill gather launches that ran today's kernel although --exl3-sel-tc on was asked.\n# TYPE velogb10_exl3_sel_tc_fallback_calls_total counter\n");
    o.push_str(&format!("velogb10_exl3_sel_tc_fallback_calls_total {}\n", super::SEL_TC_FALLBACK_CALLS.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_dense_pf_tc_qsa_state --exl3-dense-pf-tc-qsa on this boot: 0 = off, 1 = on.\n# TYPE velogb10_exl3_dense_pf_tc_qsa_state gauge\n");
    o.push_str(&format!("velogb10_exl3_dense_pf_tc_qsa_state {}\n", super::pf_tc_qsa_on() as u8));
    o.push_str("# HELP velogb10_exl3_dense_pf_tc_qsa_rows_total Dense prefill rows (summed over attention layers, sparse-attention packs) served by the tensor-core dense kernels.\n# TYPE velogb10_exl3_dense_pf_tc_qsa_rows_total counter\n");
    o.push_str(&format!("velogb10_exl3_dense_pf_tc_qsa_rows_total {}\n", super::PF_TC_QSA_ROWS.load(Relaxed)));
}

/// `--probe-mid-gemm --model-dir <pack> [--probe-lane-wide-rows 24,32,48,64,96,128] [--probe-lane-wide-reps N]`
/// (GPU, one box): every dense trunk matrix (GDN in_qkv / in_z / out_proj, attention q / k / v / o, indexer qk,
/// shared-expert gate / up / down) with synthetic rows:
///  1. bit-identity: the mid body at 48 and 128 rows vs the decode-class chain (m <= 16 groups) row by row;
///  2. row-independence: the same rows at an offset of 7 inside a 7-row-longer pass;
///  3. the per-pass dense time (every matrix once, had_suh + GEMM + svh) at each R: mid vs kept fp16 (cuBLASLt on the
///     kept matrices) vs reconstruct + cuBLASLt, median of the reps.
pub fn probe_mid_gemm(dir: &str, rows: &[usize], reps: usize) -> Result<bool> {
    let model = FwdModel::load(dir, 1, 4096)?;
    anyhow::ensure!(model.tp.is_none(), "--probe-mid-gemm runs on one box");
    let l = Launcher { dev: &model.dev, stream: &model.stream };
    let rmax = rows.iter().copied().max().unwrap_or(128).max(135);
    let psc = model.prefill_scratch(rmax)?;
    let rb = model.recon_bufs(&psc);
    let mut quads: Vec<(&str, &Quad)> = Vec::new();
    for layer in model.layers.iter() {
        match &layer.mixer {
            Mixer::Gdn(g) => { quads.push(("gdn.in_qkv", &g.in_qkv)); quads.push(("gdn.in_z", &g.in_z)); quads.push(("gdn.out_proj", &g.out_proj)); }
            Mixer::Attn(a) => {
                quads.push(("attn.q", &a.q_proj)); quads.push(("attn.k", &a.k_proj)); quads.push(("attn.v", &a.v_proj)); quads.push(("attn.o", &a.o_proj));
                if let Some(ix) = a.idx.as_ref() { quads.push(("attn.idx_qk", &ix.qk)); }
            }
        }
        if !layer.moe.dense && layer.moe.split.is_none() && layer.moe.fx.is_none() {
            quads.push(("shared.gate", &layer.moe.sh_gate)); quads.push(("shared.up", &layer.moe.sh_up)); quads.push(("shared.down", &layer.moe.sh_down));
        }
    }
    let kmax = quads.iter().map(|q| q.1.k as usize).max().unwrap_or(1);
    let nmax = quads.iter().map(|q| q.1.n as usize).max().unwrap_or(1);
    let dev = &model.dev;
    // deterministic synthetic rows (|x| < ~2, f16)
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rnd = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; ((seed >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0 };
    let host: Vec<u16> = (0..rmax * kmax).map(|_| half::f16::from_f32(rnd()).to_bits()).collect();
    let mut x = dev.htod_sync_copy(&host)?;
    let mut xh = dev.alloc_zeros::<u16>(rmax * kmax.max(12288))?;
    let mut yraw = dev.alloc_zeros::<u16>(rmax * nmax)?;
    let mut y = dev.alloc_zeros::<u16>(rmax * nmax)?;
    let ws = dev.alloc_zeros::<f32>(2_000_000)?;
    let mut x16 = dev.alloc_zeros::<u16>(16 * kmax)?;
    let mut y16 = dev.alloc_zeros::<u16>(16 * nmax)?;
    let mut ok = true;
    let shapes: std::collections::BTreeSet<(usize, usize, u32, i32)> = quads.iter().map(|q| (q.1.k as usize, q.1.n as usize, q.1.ks, q.1.k2)).collect();
    crate::dprintln!("MID_GEMM probe: {} dense trunk matrices; shapes (K, N, ks, k2): {:?}", quads.len(), shapes);
    // 1 + 2: bit-identity and row-independence, one matrix per distinct shape
    let mut seen = std::collections::BTreeSet::new();
    for (name, q) in quads.iter() {
        let key = (q.k, q.n, q.k2);
        if !seen.insert(key) { continue; }
        let (k, n) = (q.k as usize, q.n as usize);
        for &r in &[24usize, 48, 57] {
            set_mid(true);
            exl3_chain_impl(&l, q, &x, &mut xh, &mut yraw, &mut y, r, &ws, Some(&rb))?;
            dev.synchronize()?;
            let ym: Vec<u16> = dev.dtoh_sync_copy(&y)?[..r * n].to_vec();
            // decode-class reference, 16-row groups
            let mut yd = vec![0u16; r * n];
            let mut r0 = 0;
            while r0 < r {
                let m = 16.min(r - r0);
                let n16 = (m * k) as i64;
                xqlaunch!(l, "xq_copy_u16", ((((n16 + 255) / 256) as u32), 1, 1), (256, 1, 1), 0, (&mut x16, &x, n16, 0i64, (r0 * k) as i64))?;
                exl3_chain(&l, q, &x16, &mut xh, &mut yraw, &mut y16, m, &ws)?;
                dev.synchronize()?;
                yd[r0 * n..(r0 + m) * n].copy_from_slice(&dev.dtoh_sync_copy(&y16)?[..m * n]);
                r0 += m;
            }
            let diff_dec = ym.iter().zip(&yd).filter(|(a, b)| a != b).count();
            // row-independence: 7 other rows first, then the same rows
            let off = 7usize;
            let mut xs = dev.alloc_zeros::<u16>((r + off) * k)?;
            let other: Vec<u16> = (0..off * k).map(|i| host[(i * 7919) % host.len()]).collect();
            dev.htod_sync_copy_into(&other, &mut xs.slice_mut(0..off * k))?;
            let nn = (r * k) as i64;
            xqlaunch!(l, "xq_copy_u16", ((((nn + 255) / 256) as u32), 1, 1), (256, 1, 1), 0, (&mut xs, &x, nn, (off * k) as i64, 0i64))?;
            exl3_chain_impl(&l, q, &xs, &mut xh, &mut yraw, &mut y, r + off, &ws, Some(&rb))?;
            dev.synchronize()?;
            let ys: Vec<u16> = dev.dtoh_sync_copy(&y)?[off * n..(off + r) * n].to_vec();
            let diff_off = ym.iter().zip(&ys).filter(|(a, b)| a != b).count();
            // the kept-fp16 arm (today's) vs the decode class, for scale
            set_mid(false);
            exl3_chain_impl(&l, q, &x, &mut xh, &mut yraw, &mut y, r, &ws, Some(&rb))?;
            dev.synchronize()?;
            let yk: Vec<u16> = dev.dtoh_sync_copy(&y)?[..r * n].to_vec();
            let diff_kept = yk.iter().zip(&yd).filter(|(a, b)| a != b).count();
            let pass = diff_dec == 0 && diff_off == 0;
            ok &= pass;
            crate::dprintln!("MID_GEMM {name} K {k} N {n} ks {} k2 {} R {r}: mid vs decode-class differ {diff_dec} of {}, mid at offset 7 \
in R+7 differ {diff_off}; (today's fp16 arm vs decode-class differ {diff_kept}) {}", q.ks, q.k2, r * n, if pass { "OK" } else { "MISMATCH" });
        }
    }
    // 3: per-pass dense time
    let time_all = |r: usize, mid: bool, kept: bool| -> Result<f64> {
        set_mid(mid);
        set_on(kept);
        let (mut xh2, mut yr2, mut y2) = (xh.clone(), yraw.clone(), y.clone());
        let mut t = Vec::new();
        for it in 0..reps + 2 {
            dev.synchronize()?;
            let t0 = std::time::Instant::now();
            for (_, q) in quads.iter() {
                exl3_chain_impl(&l, q, &x, &mut xh2, &mut yr2, &mut y2, r, &ws, Some(&rb))?;
            }
            dev.synchronize()?;
            if it >= 2 { t.push(t0.elapsed().as_secs_f64() * 1e3); }
        }
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(t[t.len() / 2])
    };
    for &r in rows {
        let r = r.min(rmax);
        let tm = if r > 16 && r <= MID_MAX { time_all(r, true, false)? } else { f64::NAN };
        let tk = time_all(r, false, true)?;
        let tr = time_all(r, false, false)?;
        crate::dprintln!("MID_GEMM per-pass dense R={r:4}: mid {tm:7.2} ms | kept fp16 {tk:7.2} ms | reconstruct {tr:7.2} ms");
    }
    set_mid(false);
    set_on(false);
    crate::dprintln!("RESULT: {}", if ok { "MID_GEMM_EXACT" } else { "MID_GEMM_MISMATCH" });
    Ok(ok)
}
