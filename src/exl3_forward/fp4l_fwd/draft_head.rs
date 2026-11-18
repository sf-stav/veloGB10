//! The pruned draft lm_head (`--fp4l-draft-head head|mxfp8`) and the MXFP8 RTN quantizer. Draft-only: the verify keeps the stored bf16 head,
//! so served tokens are unchanged. See `mod.rs` for the file map.

use super::*;

// ------------------------------------------------------------------------------------------------
// Draft head (draft passes only; the verify keeps the stored bf16 head)
// ------------------------------------------------------------------------------------------------

/// `--fp4l-draft-head head|mxfp8` (default head = the verify head's rows in its own format, --fp4l-lm-head):
/// the pruned draft lm_head. mxfp8 (opt-in) = an MXFP8 RTN copy of the stored bf16 rows (E4M3 with a UE8M0
/// scale per 32 — the format's own rule), about half the bytes of a bf16 draft pass. Draft-only either way:
/// the verify scores with the verify head, so the served tokens (greedy) and distribution (sampled) are
/// unchanged; only acceptance can move.
pub(crate) fn draft_head_mode() -> &'static str {
    static V: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    // default screen (owner 2026-10-30: the draft-only levers are defaults; draft-only = served output unchanged)
    *V.get_or_init(|| match crate::opts::var(crate::opt!("fp4l-draft-head")).as_deref() {
        Ok("mxfp8") => "mxfp8",
        Ok("head") => "head",
        _ => "screen",
    })
}

/// The candidate blocks of the draft-head screen (== XF_DHS_T in gpu_fp4l.cu).
pub(crate) const DHS_T: usize = 16;

/// `--fp4l-draft-head screen` (DRAFT-ONLY): an int4 per-row-scaled copy of the draft head's rows picks the DHS_T best
/// 128-row blocks; those rows are rescored exactly (the full GEMV's bits) and every other row reads -inf. See
/// xf_dhs_* in gpu_fp4l.cu.
pub(crate) struct FxDhs {
    /// int4 [n2][k/2] (low nibble = even k), per-row fp32 scale
    pub q: CudaSlice<u8>,
    pub s: CudaSlice<f32>,
    pub bmax: CudaSlice<f32>,
    pub cand: CudaSlice<i32>,
    pub n2: usize,
    pub k: usize,
}

/// Build the screen from the stored bf16 rows [0, n2) of the lm_head (host quantize: s = max|v| / 7, q = RN(v / s)
/// clamped to [-7, 7]; parallel over rows).
pub(crate) fn dhs_build(dev: &Arc<CudaDevice>, fp: &Fp4lPack, n2: usize) -> Result<FxDhs> {
    let (n, k) = fp.shape2("lm_head.weight")?;
    anyhow::ensure!(n2 <= n && n2 % 128 == 0 && n2 / 128 <= 512 && k % 64 == 0 && k <= 4096,
                    "FP4L draft-head screen: {n2} rows of [{n}, {k}] outside the kernels' contract");
    let raw = fp.raw_range("lm_head.weight", 0, (n2 * k * 2) as u64)?;
    let mut q = vec![0u8; n2 * k / 2];
    let mut sc = vec![0f32; n2];
    let th = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).clamp(1, 16);
    let per = n2.div_ceil(th);
    std::thread::scope(|sco| {
        for ((rc, qc), scc) in raw.chunks(per * k * 2).zip(q.chunks_mut(per * k / 2)).zip(sc.chunks_mut(per)) {
            sco.spawn(move || {
                for (r, (row, qrow)) in rc.chunks_exact(k * 2).zip(qc.chunks_exact_mut(k / 2)).enumerate() {
                    let v: Vec<f32> = row.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
                    let amax = v.iter().fold(0f32, |a, &x| a.max(x.abs()));
                    let s = if amax > 0.0 { amax / 7.0 } else { 1.0 };
                    scc[r] = s;
                    let qv = |x: f32| -> u8 { ((x / s).round().clamp(-7.0, 7.0) as i8 as u8) & 0xF };
                    for (j, b) in qrow.iter_mut().enumerate() { *b = qv(v[2 * j]) | (qv(v[2 * j + 1]) << 4); }
                }
            });
        }
    });
    Ok(FxDhs {
        q: dev.htod_sync_copy(&q)?, s: dev.htod_sync_copy(&sc)?,
        bmax: dev.alloc_zeros::<f32>(n2 / 128)?, cand: dev.htod_sync_copy(&vec![0i32; DHS_T])?, n2, k,
    })
}

/// One screened draft-head pass (m = 1, graph-capturable): `x` the normed hidden [k] f16, `head` the draft head
/// (B32 rows, FxW::Bf16), `logits` [n2] f16 out.
pub(crate) fn dhs_logits(l: &Launcher, dhs: &FxDhs, head: &Quad, x: u64, logits: u64) -> Result<()> {
    let Some(FxW::Bf16 { w }) = head.fx.as_deref() else { bail!("FP4L draft-head screen: the draft head is not stored bf16") };
    let nblk = dhs.n2 / 128;
    crate::metrics::fp4l_gemm("bf16", 1);
    xflaunch!(l, "xf_dhs_screen", (nblk as u32, 1, 1), (256, 1, 1), 0, (&dhs.q, &dhs.s, x, &dhs.bmax, logits, dhs.k as i32))?;
    xflaunch!(l, "xf_dhs_select", (1, 1, 1), (512, 1, 1), 0, (&dhs.bmax, &dhs.cand, nblk as i32))?;
    xflaunch!(l, "xf_dhs_rescore", (DHS_T as u32, 1, 1), (256, 1, 1), 0, (w, x, logits, &dhs.cand, dhs.k as i32))
}

/// OCP MX RTN of one 32-value group: shared exponent = floor(log2(amax)) - 8 (E4M3 emax), codes RNE.
pub(crate) fn mx8_quant_group(v: &[f32], codes: &mut [u8]) -> u8 {
    let amax = v.iter().fold(0f32, |a, &x| a.max(x.abs()));
    if amax == 0.0 { codes.fill(0); return 127; }
    let e = amax.log2().floor() as i32 - 8;
    let e = e.clamp(-127, 127);
    let s = 2f32.powi(e);
    for (c, &x) in codes.iter_mut().zip(v) { *c = f32_to_e4m3_rne(x / s); }
    (e + 127) as u8
}

/// f32 -> OCP E4M3 (fn), round to nearest even, saturating at 448.
pub(crate) fn f32_to_e4m3_rne(x: f32) -> u8 {
    if x.is_nan() { return 0x7F; }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs().min(448.0);
    // enumerate the 127 non-negative finite codes: exact search of the nearest (ties to even code)
    let mut best = 0u8;
    let mut bd = f32::INFINITY;
    for c in 0u8..=0x7E {
        let v = crate::fp4l::pack::e4m3(c);
        let d = (v - a).abs();
        if d < bd || (d == bd && c % 2 == 0) { bd = d; best = c; }
    }
    sign | best
}

/// Build the MXFP8 draft copy of rows [0, n2) of the stored bf16 lm_head (host RTN, then upload).
pub(crate) fn draft_head_mx8(dev: &Arc<CudaDevice>, fp: &Fp4lPack, n2: usize) -> Result<Quad> {
    let (n, k) = fp.shape2("lm_head.weight")?;
    anyhow::ensure!(n2 <= n && k % 32 == 0, "FP4L draft head: {n2} rows of [{n}, {k}]");
    let raw = fp.raw_range("lm_head.weight", 0, (n2 * k * 2) as u64)?;
    let vals: Vec<f32> = raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
    let (codes, scales) = mx8_rtn(&vals, n2, k);
    fx_quad_wrap(dev, mx8_dev(dev, &codes, &scales, n2, k)?, k, n2, false)
}

/// OCP MX RTN of a row-major [n][k] f32 matrix (k % 32 == 0): E4M3 codes + one UE8M0 scale per 32 (shared
/// exponent floor(log2 amax) - 8, codes RNE, saturating at 448). Host, parallel over row blocks.
pub(crate) fn mx8_rtn(vals: &[f32], n2: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let mut codes = vec![0u8; n2 * k];
    let mut scales = vec![0u8; n2 * k / 32];
    // precomputed e4m3 grid for a fast RNE search (127 values, monotone)
    let grid: Vec<f32> = (0u8..=0x7E).map(crate::fp4l::pack::e4m3).collect();
    let q = |x: f32| -> u8 {
        let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
        let a = x.abs().min(448.0);
        let i = grid.partition_point(|&g| g < a);
        let c = if i == 0 { 0 } else if i >= grid.len() { grid.len() - 1 } else {
            let (lo, hi) = (grid[i - 1], grid[i]);
            if a - lo < hi - a { i - 1 } else if hi - a < a - lo { i } else if (i - 1) % 2 == 0 { i - 1 } else { i }
        };
        sign | c as u8
    };
    let nt = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).clamp(1, 16);
    let rows_per = n2.div_ceil(nt);
    std::thread::scope(|s| {
        for ((vc, cc), sc) in vals.chunks(rows_per * k).zip(codes.chunks_mut(rows_per * k)).zip(scales.chunks_mut(rows_per * k / 32)) {
            let q = &q;
            s.spawn(move || {
                for (g, (v, c)) in vc.chunks(32).zip(cc.chunks_mut(32)).enumerate() {
                    let amax = v.iter().fold(0f32, |a, &x| a.max(x.abs()));
                    if amax == 0.0 { c.fill(0); sc[g] = 127; continue; }
                    let e = (amax.log2().floor() as i32 - 8).clamp(-127, 127);
                    let s = 2f32.powi(e);
                    for (ci, &x) in c.iter_mut().zip(v) { *ci = q(x / s); }
                    sc[g] = (e + 127) as u8;
                }
            });
        }
    });
    (codes, scales)
}
