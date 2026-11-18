//! FP4L dense dispatch: the batch-invariant m <= 16 chains, the pair / rowpar variants, the MTP sidecar and the m > 16 wide (dequantize + cuBLASLt) path.
//! Every chain dispatcher branches here FIRST for an FP4L Quad (see `mod.rs`).

use super::*;

// ------------------------------------------------------------------------------------------------
// Dense dispatch
// ------------------------------------------------------------------------------------------------

/// The FP4L chain: y[m, n] = x[m, k] · W[n, k]^T (f16 out). m <= 16: the batch-invariant MMA GEMM
/// (split-K fixup in-kernel). m > 16: dequant (MXFP8) + cuBLASLt (opA = T on the [n, k] weight), or —
/// without recon buffers — the m <= 16 kernel over 16-row blocks.
pub(crate) fn fx_chain(l: &Launcher, q: &Quad, fx: &FxW, x: &CudaSlice<u16>, y: &mut CudaSlice<u16>, m: usize,
                       ws: &CudaSlice<f32>, rb: Option<&ReconBufs>) -> Result<()> {
    let (k, n) = (q.k as usize, q.n as usize);
    anyhow::ensure!(m * k <= x.len() && m * n <= y.len(), "FP4L chain m={m} K={k} N={n}: x {} / y {} too small", x.len(), y.len());
    crate::metrics::fp4l_gemm(fx.kind(), m);
    if m <= 16 {
        return fx_small(l, q, fx, *x.device_ptr() as u64, *y.device_ptr() as u64, 0, ws, m);
    }
    match rb {
        Some(r) => fx_wide(l, q, fx, *x.device_ptr() as u64, *y.device_ptr() as u64, m, r),
        None => {
            // no recon staging (an m > 16 chain outside prefill): 16-row blocks of the small kernel
            let mut r0 = 0;
            while r0 < m {
                let mb = (m - r0).min(16);
                fx_small(l, q, fx, *x.device_ptr() as u64 + (r0 * k * 2) as u64, *y.device_ptr() as u64 + (r0 * n * 2) as u64,
                         0, ws, mb)?;
                r0 += mb;
            }
            Ok(())
        }
    }
}

/// m <= 16: one launch (fixup) — `y` f16, or, when `yf != 0`, the fp32 result (rowpar partial).
/// `yf` mode never uses the fixup: ks > 1 writes the fp32 slabs to `ws` for the caller's combine.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fx_small(l: &Launcher, q: &Quad, fx: &FxW, x: u64, y: u64, yf: u64, ws: &CudaSlice<f32>, m: usize) -> Result<()> {
    let (k, n, ks) = (q.k as usize, q.n as usize, q.ks.max(1));
    anyhow::ensure!((1..=16).contains(&m), "fx_small: m {m} outside 1..=16");
    if let FxW::Trellis { tr, suh, svh, bits, ks: tks, scr, .. } = fx {
        // --fp4l-mtp-sidecar: H128(x * suh) -> trellis GEMM (fp32 K-slabs) -> ascending combine -> H128(.) * svh
        anyhow::ensure!(yf == 0, "FP4L MTP sidecar: a trellis linear has no fp32 partial-output (row-parallel) form");
        let (part, xh, yraw) = tr_scratch(scr, k, n, *tks)?;
        xqlaunch!(l, "exl3_had_suh", ((m * (k / 128)) as u32, 1, 1), (32, 1, 1), 0, (x, suh, xh, k as i32))?;
        xflaunch!(l, "xf_tr_gemm", ((n / 128) as u32, *tks as u32, 1), (256, 1, 1), 0,
                  (tr, xh, part, m as i32, k as i32, n as i32, *bits as i32, *tks as i32))?;
        xflaunch!(l, "xf_tr_combine", (((m * n) as u32).div_ceil(256), 1, 1), (256, 1, 1), 0, (part, yraw, (m * n) as i32, *tks as i32))?;
        return xqlaunch!(l, "exl3_had_svh", ((m * (n / 128)) as u32, 1, 1), (32, 1, 1), 0, (yraw, svh, y, n as i32));
    }
    anyhow::ensure!(k % (128 * ks as usize) == 0, "FP4L chain: ks {ks} does not split K {k} into 128-multiples");
    if ks > 1 { anyhow::ensure!(ks as usize * m * n <= ws.len(), "FP4L chain_ws too small: need {} have {}", ks as usize * m * n, ws.len()); }
    let tiles = n.div_ceil(128) as u32;
    // fixup when producing f16 with ks > 1; slab-only when the caller combines (yf mode)
    let cnt: u64 = if ks > 1 && yf == 0 { *q.cnt.device_ptr() as u64 } else { 0 };
    let (yp, yfp) = if yf != 0 { (0u64, if ks > 1 { 0u64 } else { yf }) } else { (y, 0u64) };
    let wsp = *ws.device_ptr() as u64;
    match fx {
        FxW::Mx8 { w, s } => xflaunch!(l, "xf_mx8_gemm", (tiles, ks, 1), (256, 1, 1), 0,
            (w, s, x, yp, yfp, wsp, cnt, m as i32, k as i32, n as i32, ks as i32)),
        FxW::F16 { w, osc } => xflaunch!(l, "xf_f16w_gemm", (tiles, ks, 1), (256, 1, 1), 0,
            (w, x, yp, yfp, wsp, cnt, m as i32, k as i32, n as i32, ks as i32, *osc)),
        FxW::Bf16 { w } => xflaunch!(l, "xf_bf16w_gemm", (tiles, ks, 1), (256, 1, 1), 0,
            (w, x, yp, yfp, wsp, cnt, m as i32, k as i32, n as i32, ks as i32)),
        FxW::Trellis { .. } => unreachable!(),
    }
}


/// Two m <= 16 FP4L GEMMs that read the same `x` (the GDN in_proj_a | in_proj_b pair, the shared expert's
/// gate | up) as ONE xf_mx8_gemm_x2 launch: taken when both are MXFP8 with the same (K, N, ks) and each member's
/// fp32 slab space (`ws0` / `ws1`, raw pointer + length in floats: the members run at the same time) holds its
/// ks x m x N slabs. Each output is bitwise its own xf_mx8_gemm result (same body, its own slabs and fixup
/// counters). Ok(false) = not taken, nothing launched: the caller runs the two chains as before.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fx_pair(l: &Launcher, q0: &Quad, q1: &Quad, x: u64, y0: u64, y1: u64, ws0: (u64, usize), ws1: (u64, usize),
                      m: usize) -> Result<bool> {
    let (Some(FxW::Mx8 { w: w0, s: s0 }), Some(FxW::Mx8 { w: w1, s: s1 })) = (q0.fx.as_deref(), q1.fx.as_deref()) else {
        return Ok(false);
    };
    let (k, n, ks) = (q0.k as usize, q0.n as usize, q0.ks.max(1));
    if !(1..=16).contains(&m) || q1.k as usize != k || q1.n as usize != n || q1.ks.max(1) != ks
        || k % (128 * ks as usize) != 0 || (ks > 1 && (ks as usize * m * n > ws0.1 || ks as usize * m * n > ws1.1)) {
        return Ok(false);
    }
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        crate::drprintln!("[fp4l] paired MXFP8 GEMMs (GDN a|b, shared gate|up) run as ONE xf_mx8_gemm_x2 launch (bitwise per member)");
    }
    crate::metrics::fp4l_gemm("mxfp8", m);
    crate::metrics::fp4l_gemm("mxfp8", m);
    let cnt = |q: &Quad| if ks > 1 { *q.cnt.device_ptr() as u64 } else { 0 };
    let p = XfPair {
        w: [*w0.device_ptr() as u64, *w1.device_ptr() as u64],
        s: [*s0.device_ptr() as u64, *s1.device_ptr() as u64],
        y: [y0, y1],
        ws: [ws0.0, ws1.0],
        cnt: [cnt(q0), cnt(q1)],
    };
    xflaunch!(l, "xf_mx8_gemm_x2", (n.div_ceil(128) as u32, ks, 2), (256, 1, 1), 0,
              (p, x, m as i32, k as i32, n as i32, ks as i32))?;
    Ok(true)
}

/// The trellis linear's decode-path scratch (fp32 units): [ks][16][n] partials, then xh [16][k] f16, then y_raw
/// [16][n] f16. Returns (partials, xh, y_raw) device pointers.
fn tr_scratch(scr: &CudaSlice<f32>, k: usize, n: usize, ks: usize) -> Result<(u64, u64, u64)> {
    let need = tr_scratch_len(k, n, ks);
    anyhow::ensure!(scr.len() >= need, "FP4L MTP sidecar scratch {} < {need} floats", scr.len());
    let base = *scr.device_ptr() as u64;
    let part = base;
    let xh = base + (ks * 16 * n * 4) as u64;
    let yraw = xh + (16 * k * 2) as u64;
    Ok((part, xh, yraw))
}
fn tr_scratch_len(k: usize, n: usize, ks: usize) -> usize { ks * 16 * n + 8 * k + 8 * n }

/// K-slabs of a sidecar trellis linear at m <= 16: a pure shape function — the smallest divisor d of K/16 putting
/// >= 96 CTAs on the device (2 waves of 48 SMs), else the largest d leaving >= 4 k16 steps per slab.
fn tr_ks(k: usize, n: usize) -> usize {
    let (kb, tiles) = (k / 16, n / 128);
    let divs: Vec<usize> = (1..=kb).filter(|d| kb % d == 0 && kb / d >= 4).collect();
    for &d in &divs { if tiles * d >= 96 { return d; } }
    *divs.last().unwrap_or(&1)
}

/// The pack directory's EXL3 MTP-head sidecar (`--fp4l-mtp-sidecar auto`, draft-only): the file's header, by tensor name.
pub(crate) struct MtpSidecar {
    pub path: String,
    pub hdr: std::collections::BTreeMap<String, crate::exl3::HdrEntry>,
}

/// `--fp4l-mtp-sidecar auto|off` resolved against the model directory `dir`: auto (default, owner 2026-10-30) = the
/// `mtp-*.safetensors` in `dir` (the lexicographically last of several), if any — under TP the head ships that file to every node (shard
/// plan), so every rank resolves the same sidecar in its own model directory; off = none. No external-path form (owner 2026-11-01:
/// the pack decides — a sidecar is used exactly when the pack directory ships one).
pub(crate) fn mtp_sidecar_path(dir: &str) -> Option<String> {
    match crate::opts::var(crate::opt!("fp4l-mtp-sidecar")).ok().as_deref().map(str::trim) {
        Some("off") | Some("none") | Some("0") => None,
        _ => crate::fp4l::mtp_sidecar_in_dir(std::path::Path::new(dir)).map(|p| p.to_string_lossy().into_owned()),
    }
}

pub(crate) fn mtp_sidecar_open(path: &str) -> Result<MtpSidecar> {
    let hdr = crate::exl3::shard_header(path).with_context(|| format!("--fp4l-mtp-sidecar {path}"))?;
    anyhow::ensure!(hdr.keys().any(|k| k.starts_with("mtp.") && k.ends_with(".trellis")),
                    "--fp4l-mtp-sidecar {path}: no mtp.*.trellis tensors (not an EXL3 MTP-head sidecar)");
    Ok(MtpSidecar { path: path.to_string(), hdr })
}

/// The sidecar's trellis linear `stem` as an FP4L Quad (FxW::Trellis), or None when the sidecar has no such stem.
/// Integer K = 2..8 only (the K = 1.5 / 2.5 / 3.5 half widths are refused); mul1 codebook only.
pub(crate) fn mtp_sidecar_quad(dev: &Arc<CudaDevice>, sc: &MtpSidecar, stem: &str) -> Result<Option<Quad>> {
    let get = |suffix: &str| sc.hdr.get(&format!("{stem}.{suffix}"));
    let (Some(tm), Some(su), Some(sv), Some(m1)) = (get("trellis"), get("suh"), get("svh"), get("mul1")) else { return Ok(None) };
    use std::os::unix::fs::FileExt;
    let f = std::fs::File::open(&sc.path).with_context(|| format!("open {}", sc.path))?;
    let read = |e: &crate::exl3::HdrEntry| -> Result<Vec<u8>> {
        let mut b = vec![0u8; e.n_bytes];
        f.read_exact_at(&mut b, e.offset as u64).with_context(|| format!("read {stem} from {}", sc.path))?;
        Ok(b)
    };
    let mul1 = u32::from_le_bytes(read(m1)?[..4].try_into().unwrap());
    anyhow::ensure!(mul1 == 0x83DC_D12D, "{stem}: codebook flag {mul1:#x} is not mul1 (0x83dcd12d)");
    anyhow::ensure!(tm.dtype == "I16" && tm.shape.len() == 3 && su.dtype == "F16" && sv.dtype == "F16", "{stem}: tensor types");
    let (kb, nb, w) = (tm.shape[0], tm.shape[1], tm.shape[2]);
    let k2 = crate::exl3::k2_from_words(w).map_err(|e| anyhow::anyhow!("{stem}: {e}"))?;
    anyhow::ensure!(k2 % 2 == 0 && (4..=16).contains(&k2), "{stem}: K = {} — the sidecar path serves integer K = 2..8", k2 as f64 / 2.0);
    let (k, n, bits) = (kb * 16, nb * 16, k2 / 2);
    anyhow::ensure!(k % 128 == 0 && n % 128 == 0 && su.shape == [k] && sv.shape == [n], "{stem}: [{k}, {n}] not 128-aligned or suh/svh shapes");
    let trb = read(tm)?;
    let tr16: Vec<i16> = trb.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
    let suh: Vec<u16> = read(su)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let svh: Vec<u16> = read(sv)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let weff = tr_effective_weight(&tr16, &suh, &svh, kb, nb, k2);
    let ks = tr_ks(k, n);
    let fx = FxW::Trellis {
        tr: dev.htod_sync_copy(&tr16.iter().map(|&v| v as u16).collect::<Vec<u16>>())?,
        suh: dev.htod_sync_copy(&suh)?, svh: dev.htod_sync_copy(&svh)?, bits, ks,
        scr: dev.alloc_zeros::<f32>(tr_scratch_len(k, n, ks))?,
        weff: dev.htod_sync_copy(&weff)?,
    };
    Ok(Some(fx_quad_wrap(dev, fx, k, n, false)?))
}

/// In-place orthonormal Walsh-Hadamard transform of a 128-vector (Sylvester order: exl3_had128's matrix).
fn fwht128(v: &mut [f32]) {
    let mut len = 1;
    while len < 128 {
        for i in (0..128).step_by(2 * len) {
            for j in i..i + len {
                let (a, b) = (v[j], v[j + len]);
                v[j] = a + b;
                v[j + len] = a - b;
            }
        }
        len <<= 1;
    }
    let s = 0.088_388_35_f32; // 1/sqrt(128)
    for x in v.iter_mut() { *x *= s; }
}

/// The effective weight of an EXL3 linear, transposed for the prefill cuBLASLt arm: f16 [n][k] of
/// diag(suh) H Wq H diag(svh) (Wq = the decoded trellis [k][n]; H = block-diagonal orthonormal H128), fp32 math,
/// one RN to f16. y = x W_eff equals had_svh(had_suh(x) Wq) up to the f16 roundings of the decode path.
fn tr_effective_weight(tr: &[i16], suh: &[u16], svh: &[u16], kb: usize, nb: usize, k2: usize) -> Vec<u16> {
    let (k, n) = (kb * 16, nb * 16);
    let wq = crate::exl3::decode_trellis_host(tr, kb, nb, k2); // [k][n] f16 bits
    let mut w: Vec<f32> = wq.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
    let suf: Vec<f32> = suh.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
    let svf: Vec<f32> = svh.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
    // along N (each row k): H over every 128-block, then * svh
    for row in w.chunks_exact_mut(n) {
        for blk in row.chunks_exact_mut(128) { fwht128(blk); }
        for (x, s) in row.iter_mut().zip(&svf) { *x *= *s; }
    }
    // along K (each column n): H over every 128-block of rows, then * suh (row scale)
    let mut col = vec![0f32; 128];
    for c in 0..n {
        for b in 0..k / 128 {
            for i in 0..128 { col[i] = w[(b * 128 + i) * n + c]; }
            fwht128(&mut col);
            for i in 0..128 { w[(b * 128 + i) * n + c] = col[i] * suf[b * 128 + i]; }
        }
    }
    // transpose to [n][k] f16
    let mut out = vec![0u16; n * k];
    for kk in 0..k {
        for c in 0..n { out[c * k + kk] = half::f16::from_f32(w[kk * n + c]).to_bits(); }
    }
    out
}

/// `--fp4l-prefill-dense exact|lt` (default exact): the m > 16 MXFP8 GEMMs as the exact W8A16 kernel
/// (xf_mx8_tc: per-group fp32 scaling, the decode kernels' numerics) or as an f16 dequant into the recon
/// staging + cuBLASLt (faster; rounds the 56,179 MXFP8 weights of this pack that lie below f16's exact range).
pub(crate) fn prefill_dense_lt() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::var(crate::opt!("fp4l-prefill-dense")).as_deref() == Ok("lt"))
}

/// m > 16 (prefill, head fill). MXFP8: the exact smem-tiled xf_mx8_tc (default) or dequant + cuBLASLt (opt-in).
/// F16: cuBLASLt straight on the stored (pre-scaled) weight with alpha = osc (exact power of two).
/// y = x · W^T, opA = T on the row-major [n, k] weight (the xq_gemm_f16 convention). `y` is f16.
pub(crate) fn fx_wide(l: &Launcher, q: &Quad, fx: &FxW, x: u64, y: u64, m: usize, r: &ReconBufs) -> Result<()> {
    let (k, n) = (q.k as usize, q.n as usize);
    match fx {
        FxW::Mx8 { w, s } if !prefill_dense_lt() => {
            // xf_mx8_tc: the smem-tiled exact GEMM (bit-identical to xf_mx8_wide), 1-D grouped grid
            let ctas = n.div_ceil(128) * m.div_ceil(128);
            xflaunch!(l, "xf_mx8_tc", (ctas as u32, 1, 1), (256, 1, 1), 0,
                      (w, s, x, y, 0u64, m as i32, k as i32, n as i32))
        }
        FxW::Mx8 { w, s } => {
            anyhow::ensure!(n * k <= r.ws_len, "FP4L recon staging too small: need {} have {}", n * k, r.ws_len);
            let ne = (n * k) as i64;
            xflaunch!(l, "xf_mx8_dequant", (((ne / 8) as u32).div_ceil(256), 1, 1), (256, 1, 1), 0,
                      (w, s, r.ws, ne, k as i32))?;
            exl3_lt_matmul(l, r.ws, x, y, k, n, m, true, r.ltws, r.ltws_len)
        }
        FxW::F16 { w, osc } => exl3_lt_matmul_alpha(l, *w.device_ptr() as u64, x, y, k, n, m, true, r.ltws, r.ltws_len, *osc),
        // --fp4l-mtp-sidecar: the load-time effective f16 weight through cuBLASLt (the MTP prime over a prompt chunk)
        FxW::Trellis { weff, .. } => exl3_lt_matmul_alpha(l, *weff.device_ptr() as u64, x, y, k, n, m, true, r.ltws, r.ltws_len, 1.0),
        FxW::Bf16 { w } => {
            let _ = r;
            xflaunch!(l, "xf_bf16_tc", ((n.div_ceil(128) * m.div_ceil(128)) as u32, 1, 1), (256, 1, 1), 0,
                      (w, x, y, 0u64, m as i32, k as i32, n as i32))
        }
    }
}

/// The FP4L DFlash2 head: the pre-scaled f16 lm_head back to its stored bf16 values (exact).
pub(crate) fn f16_to_bf16_scaled(l: &Launcher, out: u64, w: &CudaSlice<u16>, scale: f32) -> Result<()> {
    let n = w.len() as i64;
    xflaunch!(l, "xf_f16_to_bf16_scaled", (((n + 255) / 256).min(65535) as u32, 1, 1), (256, 1, 1), 0, (out, w, n, scale))
}

/// A u16 bit copy of `n` elements on the compute stream (src / dst raw device pointers).
pub(crate) fn copy_u16(l: &Launcher, dst: u64, src: u64, n: usize) -> Result<()> {
    let n = n as i64;
    xflaunch!(l, "xf_copy_u16", (((n + 255) / 256).min(65535) as u32, 1, 1), (256, 1, 1), 0, (dst, src, n))
}

/// The verify head as bf16 [n][k] at `out` (the DFlash2 drafter's head view): the stored bits (Bf16), the exact
/// unshifted cast (F16), or the exact bf16 dequant of the MXFP8 copy (Mx8).
pub(crate) fn head_to_bf16(l: &Launcher, out: u64, fx: &FxW, n: usize, k: usize) -> Result<()> {
    // the REAL rows (F64 / B32 pad N to a multiple of 16; the padding is never copied out)
    let ne = (n * k) as i64;
    anyhow::ensure!(ne % 8 == 0 && k % 32 == 0, "FP4L head_to_bf16: {n} x {k}");
    match fx {
        FxW::Trellis { .. } => bail!("FP4L head_to_bf16: the head is never a sidecar trellis linear"),
        FxW::F16 { w, osc } => f16_to_bf16_scaled(l, out, w, *osc),
        FxW::Bf16 { w } => {
            anyhow::ensure!(w.len() >= n * k, "FP4L head_to_bf16: B32 head {} < {n} x {k}", w.len());
            xflaunch!(l, "xf_b32_to_rm", (((ne / 8) as u32).div_ceil(256), 1, 1), (256, 1, 1), 0, (out, w, ne, k as i32))
        }
        FxW::Mx8 { w, s } => {
            anyhow::ensure!(w.len() >= n * k, "FP4L head_to_bf16: F64 head {} < {n} x {k}", w.len());
            xflaunch!(l, "xf_mx8_dequant_bf16", (((ne / 8) as u32).div_ceil(256), 1, 1), (256, 1, 1), 0, (w, s, out, ne, k as i32))
        }
    }
}

/// TP row-parallel partial (m <= 16): this rank's fp32 result over its K-slice as ks fp32 slabs in `ws`
/// (`[ks][m][n]`; ks == 1 = one slab), for the caller's ascending xq_ks_combine_f32* into the TP
/// partial buffer — the exl3_hmma_gemm_ks + combine contract of the trellis row-parallel chain.
pub(crate) fn fx_rowpar_small(l: &Launcher, q: &Quad, fx: &FxW, x: &CudaSlice<u16>, ws: &CudaSlice<f32>, m: usize) -> Result<()> {
    crate::metrics::fp4l_gemm(fx.kind(), m);
    anyhow::ensure!(q.ks.max(1) as usize * m * q.n as usize <= ws.len(), "FP4L row-parallel chain_ws too small");
    fx_small(l, q, fx, *x.device_ptr() as u64, 0, *ws.device_ptr() as u64, ws, m)
}
