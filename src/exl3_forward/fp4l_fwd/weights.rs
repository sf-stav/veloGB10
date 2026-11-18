//! Loading the FP4-Local pack into FP4L weight kinds: MXFP8 and bf16 fragment layouts, `fx_quad` and row slices, GDN a/b, experts,
//! the PLE n-gram table and the byte model. Weights are served AS STORED. See `mod.rs` for the file map.

use super::*;

// ------------------------------------------------------------------------------------------------
// Load
// ------------------------------------------------------------------------------------------------

/// `Some(pack)` when `dir` is an FP4-Local directory (the loader then reads every weight here).
pub(crate) fn open_if_fp4l(dir: &str) -> Result<Option<Arc<Fp4lPack>>> {
    let p = std::path::Path::new(dir);
    if !crate::fp4l::pack::is_fp4l_dir(p) { return Ok(None); }
    crate::metrics::fp4l_pack_loaded();
    let t0 = std::time::Instant::now();
    let pk = Fp4lPack::open(p).with_context(|| format!("FP4L: open {dir}"))?;
    crate::rprintln!("[fp4l] {dir}: ModelOpt MIXED_PRECISION pack, {} tensors in {} files ({:.1}s); MXFP8 dense + NVFP4 csf experts \
                      + bf16 head/MTP served as stored (W8A16 / W4A16, f16 activations)", pk.t.len(), pk.files.len(), t0.elapsed().as_secs_f32());
    Ok(Some(Arc::new(pk)))
}

/// bf16 bytes -> (f16 bits of v * 2^k, osc = 2^-k), exact for every value in this pack (receipt:
/// --probe-fp4l FP4L_PLAIN_F16_SHIFTED); any value that still does not fit is counted and printed.
pub(super) fn bf16_bytes_to_f16(raw: &[u8], what: &str) -> (Vec<u16>, f32) {
    let vals: Vec<f32> = raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
    let k = crate::fp4l::pack::exact_f16_shift(&vals);
    let sc = 2f32.powi(k);
    let mut n_round = 0u64;
    let out: Vec<u16> = vals.iter().map(|&v| {
        let h = half::f16::from_f32(v * sc);
        if h.to_f32() != v * sc && v.is_finite() { n_round += 1; }
        h.to_bits()
    }).collect();
    if n_round > 0 {
        crate::rprintln!("[fp4l] {what}: {n_round} bf16 values do not fit f16 even with the 2^{k} shift (rounded; range wider than f16's)");
    }
    (out, 2f32.powi(-k))
}

/// Row-major MXFP8 ([n][k] E4M3 codes, [n][k/32] UE8M0 scales) -> the fragment-native F64 layout the
/// xf_mx8_* kernels read (gpu_fp4l.cu header). A pure byte permutation: every code and scale byte is kept,
/// rows are padded to a multiple of 16 with zero codes / scale 127 (2^0; the padded rows are never stored).
/// Codes W'[t][b][h][lane][16]: lane = gid*4 + tig holds row 16t+8h+gid, k = 64b + 32gs + 8tig + j at byte
/// gs*8 + j. Scales S'[t][b][ts][8]: byte h*4 + r*2 + gs holds row 16t+8h+2ts+r, group 2b+gs.
pub(crate) fn mx8_frag(codes: &[u8], scales: &[u8], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    assert!(k % 64 == 0 && codes.len() == n * k && scales.len() == n * k / 32, "mx8_frag: [{n}][{k}] shape");
    let nt = n.div_ceil(16);
    let kb = k / 64;
    let mut w = vec![0u8; nt * 16 * k];
    let mut sc = vec![127u8; nt * k / 2];
    let th = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).clamp(1, 16);
    let per = nt.div_ceil(th).max(1);
    std::thread::scope(|sco| {
        for (ci, (wc, scc)) in w.chunks_mut(per * 16 * k).zip(sc.chunks_mut(per * k / 2)).enumerate() {
            sco.spawn(move || {
                for tl in 0..wc.len() / (16 * k) {
                    let t = ci * per + tl;
                    for rr in 0..16 {
                        let row = 16 * t + rr;
                        if row >= n { break; }
                        let (h, gid) = (rr >> 3, rr & 7);
                        let src = &codes[row * k..(row + 1) * k];
                        let ssrc = &scales[row * (k / 32)..(row + 1) * (k / 32)];
                        for b in 0..kb {
                            for gs in 0..2 {
                                for tig in 0..4 {
                                    let o = ((tl * kb + b) * 2 + h) * 512 + (gid * 4 + tig) * 16 + gs * 8;
                                    let i = 64 * b + 32 * gs + 8 * tig;
                                    wc[o..o + 8].copy_from_slice(&src[i..i + 8]);
                                }
                                scc[(tl * kb + b) * 32 + (gid >> 1) * 8 + h * 4 + (gid & 1) * 2 + gs] = ssrc[2 * b + gs];
                            }
                        }
                    }
                }
            });
        }
    });
    (w, sc)
}

/// Row-major stored bf16 [n][k] -> the fragment-native B32 layout the stored-bf16 GEMMs read (gpu_fp4l.cu): a pure
/// u16 permutation (no value changes), rows padded to a multiple of 16 with zeros. W'[t][g][h][lane][8]: lane =
/// gid*4 + tig holds row 16t+8h+gid, k = 32g + 8tig + j at u16 j.
pub(crate) fn bf16_frag(bits: &[u16], n: usize, k: usize) -> Vec<u16> {
    assert!(k % 32 == 0 && bits.len() == n * k, "bf16_frag: [{n}][{k}] shape ({} values)", bits.len());
    let (nt, g) = (n.div_ceil(16), k / 32);
    let mut w = vec![0u16; nt * 16 * k];
    let th = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).clamp(1, 16);
    let per = nt.div_ceil(th).max(1);
    std::thread::scope(|sco| {
        for (ci, wc) in w.chunks_mut(per * 16 * k).enumerate() {
            sco.spawn(move || {
                for tl in 0..wc.len() / (16 * k) {
                    for rr in 0..16 {
                        let row = 16 * (ci * per + tl) + rr;
                        if row >= n { break; }
                        let (h, gid) = (rr >> 3, rr & 7);
                        let src = &bits[row * k..(row + 1) * k];
                        for gg in 0..g {
                            for tig in 0..4 {
                                let o = ((tl * g + gg) * 2 + h) * 256 + (gid * 4 + tig) * 8;
                                let i = 32 * gg + 8 * tig;
                                wc[o..o + 8].copy_from_slice(&src[i..i + 8]);
                            }
                        }
                    }
                }
            });
        }
    });
    w
}

/// The inverse of `bf16_frag` (the first n rows, row-major) — host readers of a B32 device copy (--hc-int8).
pub(crate) fn bf16_unfrag(w: &[u16], n: usize, k: usize) -> Vec<u16> {
    assert!(k % 32 == 0 && w.len() >= n * k, "bf16_unfrag: [{n}][{k}] shape ({} values)", w.len());
    let g = k / 32;
    let mut out = vec![0u16; n * k];
    for row in 0..n {
        let (t, rr) = (row / 16, row % 16);
        let (h, gid) = (rr >> 3, rr & 7);
        for gg in 0..g {
            for tig in 0..4 {
                let o = ((t * g + gg) * 2 + h) * 256 + (gid * 4 + tig) * 8;
                let i = row * k + 32 * gg + 8 * tig;
                out[i..i + 8].copy_from_slice(&w[o..o + 8]);
            }
        }
    }
    out
}

/// Upload a row-major MXFP8 matrix in the F64 layout.
pub(super) fn mx8_dev(dev: &Arc<CudaDevice>, codes: &[u8], scales: &[u8], n: usize, k: usize) -> Result<FxW> {
    let (w, s) = mx8_frag(codes, scales, n, k);
    Ok(FxW::Mx8 { w: dev.htod_sync_copy(&w)?, s: dev.htod_sync_copy(&s)? })
}

/// Concatenate the [r0, r1) element ranges of `v`, each range scaled by `unit` elements.
fn take_ranges<T: Copy>(v: &[T], ranges: &[(usize, usize)], unit: usize) -> Vec<T> {
    ranges.iter().flat_map(|&(a, b)| v[a * unit..b * unit].iter().copied()).collect()
}

/// An FP4L dense linear as a Quad. `cols` = output (N) ranges, `rows` = input (K) ranges, in
/// ELEMENTS (the EXL3 get_quad_sl convention; 128-aligned by the TP planner). `rowpar` marks the
/// K-sliced partial class (xtp::chain_rowpar).
pub(crate) fn fx_quad(dev: &Arc<CudaDevice>, fp: &Fp4lPack, stem: &str, rows: Option<&[(usize, usize)]>,
                      cols: Option<&[(usize, usize)]>, rowpar: bool) -> Result<Quad> {
    let kind = fp.lin_kind(stem)?;
    let (n_full, k_full) = match kind {
        LinKind::Mxfp8 | LinKind::Bf16 => fp.shape2(&format!("{stem}.weight"))?,
        other => bail!("FP4L {stem}: {other:?} is not a dense linear kind"),
    };
    let nr: Vec<(usize, usize)> = cols.map(|c| c.to_vec()).unwrap_or_else(|| vec![(0, n_full)]);
    let kr: Vec<(usize, usize)> = rows.map(|r| r.to_vec()).unwrap_or_else(|| vec![(0, k_full)]);
    for &(a, b) in &nr { anyhow::ensure!(a <= b && b <= n_full, "FP4L {stem}: output range [{a}, {b}) outside N {n_full}"); }
    for &(a, b) in &kr {
        anyhow::ensure!(a <= b && b <= k_full && a % 32 == 0 && b % 32 == 0,
                        "FP4L {stem}: input range [{a}, {b}) outside K {k_full} or not 32-aligned");
    }
    let n: usize = nr.iter().map(|(a, b)| b - a).sum();
    let k: usize = kr.iter().map(|(a, b)| b - a).sum();
    check_dense_shape(k, n, stem)?;
    let whole_k = kr.len() == 1 && kr[0] == (0, k_full);
    let fx = match kind {
        LinKind::Mxfp8 => {
            let m = fp.mxfp8(stem)?;
            let mut codes = Vec::with_capacity(n * k);
            let mut scales = Vec::with_capacity(n * k / 32);
            for &(a, b) in &nr {
                for r in a..b {
                    let row = &m.codes[r * k_full..(r + 1) * k_full];
                    let srow = &m.scales[r * (k_full / 32)..(r + 1) * (k_full / 32)];
                    if whole_k { codes.extend_from_slice(row); scales.extend_from_slice(srow); }
                    else {
                        codes.extend(take_ranges(row, &kr, 1));
                        scales.extend(kr.iter().flat_map(|&(x, y)| srow[x / 32..y / 32].iter().copied()));
                    }
                }
            }
            anyhow::ensure!(!scales.contains(&0xFF), "FP4L {stem}: a UE8M0 scale is NaN (0xFF) — checkpoint corruption");
            mx8_dev(dev, &codes, &scales, n, k)?
        }
        LinKind::Bf16 => {
            // the lm_head follows --fp4l-lm-head; every other bf16 linear (the MTP layer) --fp4l-plain
            let mode = if stem == "lm_head" { head_prec() } else if plain_exact() { HeadPrec::Bf16 } else { HeadPrec::F16 };
            let raw = fp.raw(&format!("{stem}.weight"))?;
            let bits: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let gather = |full: &[u16]| -> Vec<u16> {
                let mut w = Vec::with_capacity(n * k);
                for &(a, b) in &nr {
                    for r in a..b {
                        let row = &full[r * k_full..(r + 1) * k_full];
                        if whole_k { w.extend_from_slice(row); } else { w.extend(take_ranges(row, &kr, 1)); }
                    }
                }
                w
            };
            match mode {
                HeadPrec::Bf16 => FxW::Bf16 { w: dev.htod_sync_copy(&bf16_frag(&gather(&bits), n, k))? },
                HeadPrec::F16 => {
                    let (full, osc) = bf16_bytes_to_f16(&raw, stem);
                    FxW::F16 { w: dev.htod_sync_copy(&gather(&full))?, osc }
                }
                HeadPrec::Mx8 => {
                    // opt-in speed lever: an MXFP8 RTN copy of the stored bf16 rows (changes the logits)
                    let vals: Vec<f32> = gather(&bits).iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
                    let (codes, scales) = mx8_rtn(&vals, n, k);
                    crate::rprintln!("[fp4l] {stem}: served as an MXFP8 RTN copy (--fp4l-lm-head mxfp8: {:.0} MB instead of {:.0} MB; \
                                      changes the logits)", (codes.len() + scales.len()) as f64 / 1e6, (2 * n * k) as f64 / 1e6);
                    mx8_dev(dev, &codes, &scales, n, k)?
                }
            }
        }
        _ => unreachable!(),
    };
    fx_quad_wrap(dev, fx, k, n, rowpar)
}

/// Wrap a device FxW into a Quad (placeholder trellis fields, zeroed fixup counters).
pub(crate) fn fx_quad_wrap(dev: &Arc<CudaDevice>, fx: FxW, k: usize, n: usize, rowpar: bool) -> Result<Quad> {
    let one = || -> Result<CudaSlice<u16>> { Ok(dev.htod_sync_copy(&[0u16])?) };
    Ok(Quad {
        tr: one()?, suh: one()?, svh: one()?, bits: 0, k2: 0, k: k as i32, n: n as i32,
        ks: fx_ks(k, n), lmh: false,
        // one u32 per 128-row N-tile, explicitly zeroed (AGENTS §2.2); the fixup epilogue re-arms it
        cnt: dev.htod_sync_copy(&vec![0u32; n.div_ceil(128).max(1)])?,
        rowpar,
        fx: Some(Box::new(fx)),
    })
}

/// Rows [r0, r1) of an FP4L f16 Quad as a new Quad (the pruned draft head prefix, the vocab shard).
pub(crate) fn fx_rows_of(dev: &Arc<CudaDevice>, stream: &CudaStream, q: &Quad, r0: usize, r1: usize) -> Result<Quad> {
    let Some(fx) = q.fx.as_deref() else { bail!("fx_rows_of on a trellis Quad") };
    let k = q.k as usize;
    anyhow::ensure!(r0 < r1 && r1 <= q.n as usize, "fx_rows_of [{r0}, {r1}) outside N {}", q.n);
    let _ = stream;
    let fx2 = match fx {
        FxW::Trellis { .. } => bail!("fx_rows_of on a sidecar trellis linear (only heads are row-sliced)"),
        FxW::F16 { w, osc } => {
            let mut d = dev.alloc_zeros::<u16>((r1 - r0) * k)?;
            dev.dtod_copy(&w.slice(r0 * k..r1 * k), &mut d)?;
            FxW::F16 { w: d, osc: *osc }
        }
        FxW::Bf16 { w } => {
            // B32 is tiled by 16 rows (as F64): whole tiles
            anyhow::ensure!(r0 % 16 == 0 && (r1 % 16 == 0 || r1 == q.n as usize),
                            "fx_rows_of [{r0}, {r1}) on a bf16 Quad must be 16-row aligned (the B32 tile)");
            let (t0, t1) = (r0 / 16, r1.div_ceil(16));
            let mut d = dev.alloc_zeros::<u16>((t1 - t0) * 16 * k)?;
            dev.dtod_copy(&w.slice(t0 * 16 * k..t1 * 16 * k), &mut d)?;
            FxW::Bf16 { w: d }
        }
        FxW::Mx8 { w, s } => {
            // F64 is tiled by 16 rows: slice whole tiles (a tail past r1 inside the last tile is zero-free
            // padding the epilogue never writes, since the new Quad's N is r1 - r0)
            anyhow::ensure!(r0 % 16 == 0 && (r1 % 16 == 0 || r1 == q.n as usize),
                            "fx_rows_of [{r0}, {r1}) on an MXFP8 Quad must be 16-row aligned (the F64 tile)");
            let (t0, t1) = (r0 / 16, r1.div_ceil(16));
            let mut dw = dev.alloc_zeros::<u8>((t1 - t0) * 16 * k)?;
            let mut ds = dev.alloc_zeros::<u8>((t1 - t0) * k / 2)?;
            dev.dtod_copy(&w.slice(t0 * 16 * k..t1 * 16 * k), &mut dw)?;
            dev.dtod_copy(&s.slice(t0 * k / 2..t1 * k / 2), &mut ds)?;
            FxW::Mx8 { w: dw, s: ds }
        }
    };
    dev.synchronize()?;
    let mut out = fx_quad_wrap(dev, fx2, k, r1 - r0, false)?;
    out.lmh = q.lmh;
    Ok(out)
}

/// A plain f16 view of an MXFP8 tensor (the tiny GDN a/b projections the trunk keeps as plain f16
/// [n][k] weights). Rounding of sub-2^-14 values is counted and printed (never silent).
pub(crate) fn mx8_plain_f16(fp: &Fp4lPack, weight_name: &str) -> Result<Vec<u16>> {
    let stem = weight_name.strip_suffix(".weight").unwrap_or(weight_name);
    let m = fp.mxfp8(stem)?;
    let mut row = vec![0f32; m.k];
    let mut out = Vec::with_capacity(m.n * m.k);
    let mut n_round = 0u64;
    for r in 0..m.n {
        m.row_f32(r, &mut row);
        for &v in &row {
            let h = half::f16::from_f32(v);
            if h.to_f32() != v { n_round += 1; }
            out.push(h.to_bits());
        }
    }
    if n_round > 0 {
        crate::rprintln!("[fp4l] {stem}: {n_round} MXFP8 values below f16's exact range rounded to the nearest f16 (plain f16 projection)");
    }
    Ok(out)
}

/// FP4L: the GDN in_proj_a / in_proj_b as exact MXFP8 Quads (rows = value heads [h0, h1) of this rank).
pub(crate) fn fx_gdn_ab(dev: &Arc<CudaDevice>, fp: &Fp4lPack, p: &str, heads: Option<(usize, usize)>) -> Result<(Quad, Quad)> {
    let r: Option<Vec<(usize, usize)>> = heads.map(|h| vec![h]);
    Ok((fx_quad(dev, fp, &format!("{p}.linear_attn.in_proj_a"), None, r.as_deref(), false)?,
        fx_quad(dev, fp, &format!("{p}.linear_attn.in_proj_b"), None, r.as_deref(), false)?))
}

/// The a|b pair's private split-K workspace: two halves of fp32 slabs (one per member — xf_mx8_gemm_x2 runs them at
/// the same time; the two-launch path and the prefill tails use the first half).
pub(crate) fn ab_ws_len(a: &Quad, b: &Quad) -> usize {
    2 * ab_ws_half(a, b)
}
fn ab_ws_half(a: &Quad, b: &Quad) -> usize {
    16 * (a.ks.max(1) as usize * a.n as usize).max(b.ks.max(1) as usize * b.n as usize).max(1)
}

/// FP4L GDN a|b at m <= 16 (decode / verify): the exact MXFP8 GEMM, private workspace.
pub(crate) fn fx_ab(l: &Launcher, ab: &(Quad, Quad, CudaSlice<f32>), x: &CudaSlice<u16>, a_out: &mut CudaSlice<u16>,
                    b_out: &mut CudaSlice<u16>, m: usize) -> Result<()> {
    let (qa, qb, ws) = ab;
    let half = ab_ws_half(qa, qb);
    anyhow::ensure!(m * qa.k as usize <= x.len() && m * qa.n as usize <= a_out.len() && m * qb.n as usize <= b_out.len(),
                    "FP4L GDN a|b m={m}: buffers too small");
    let wsp = *ws.device_ptr() as u64;
    if fx_pair(l, qa, qb, *x.device_ptr() as u64, *a_out.device_ptr() as u64, *b_out.device_ptr() as u64,
               (wsp, half.min(ws.len())), (wsp + (half * 4) as u64, ws.len().saturating_sub(half)), m)? {
        return Ok(());
    }
    fx_chain(l, qa, qa.fx.as_deref().unwrap(), x, a_out, m, ws, None)?;
    fx_chain(l, qb, qb.fx.as_deref().unwrap(), x, b_out, m, ws, None)
}

/// FP4L GDN a|b at a prefill chunk's c rows (the exact wide kernel; c <= 16 = the small one).
pub(crate) fn fx_ab_rows(l: &Launcher, ab: &(Quad, Quad, CudaSlice<f32>), x: &CudaSlice<u16>, a_out: &mut CudaSlice<u16>,
                         b_out: &mut CudaSlice<u16>, c: usize, rb: &ReconBufs) -> Result<()> {
    let (qa, qb, ws) = ab;
    fx_chain(l, qa, qa.fx.as_deref().unwrap(), x, a_out, c, ws, Some(rb))?;
    fx_chain(l, qb, qb.fx.as_deref().unwrap(), x, b_out, c, ws, Some(rb))
}

/// The routed experts of `prefix` (`...layers.L.mlp` or `mtp.layers.0.mlp`) for the experts in
/// `local` (stacking order = local index). gate and up rows are fused per expert: [gate; up].
pub(crate) fn fx_experts(dev: &Arc<CudaDevice>, fp: &Fp4lPack, prefix: &str, local: &[usize], h: usize, mi: usize) -> Result<FxMoe> {
    let ne = local.len();
    let (gu_w_e, gu_s_e) = (2 * mi * h / 2, 2 * mi * h / 16);
    let (dn_w_e, dn_s_e) = (h * mi / 2, h * mi / 16);
    let mut gu_w = vec![0u8; ne * gu_w_e];
    let mut gu_s = vec![0u8; ne * gu_s_e];
    let mut dn_w = vec![0u8; ne * dn_w_e];
    let mut dn_s = vec![0u8; ne * dn_s_e];
    let mut gs = vec![0f32; ne * 3];
    // parallel over experts: csf decode + reads (the load is I/O + integer adds)
    let nt = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).clamp(1, 16);
    let chunk = ne.div_ceil(nt).max(1);
    let res: Vec<Result<()>> = std::thread::scope(|s| {
        let mut hs = Vec::new();
        for (ci, ((((gw, gsc), dw), dsc), gsv)) in gu_w.chunks_mut(chunk * gu_w_e).zip(gu_s.chunks_mut(chunk * gu_s_e))
            .zip(dn_w.chunks_mut(chunk * dn_w_e)).zip(dn_s.chunks_mut(chunk * dn_s_e)).zip(gs.chunks_mut(chunk * 3)).enumerate() {
            let local = &local[ci * chunk..(ci * chunk + gw.len() / gu_w_e)];
            hs.push(s.spawn(move || -> Result<()> {
                for (j, &e) in local.iter().enumerate() {
                    let g = fp.nvfp4(&format!("{prefix}.experts.{e}.gate_proj"))?;
                    let u = fp.nvfp4(&format!("{prefix}.experts.{e}.up_proj"))?;
                    let d = fp.nvfp4(&format!("{prefix}.experts.{e}.down_proj"))?;
                    anyhow::ensure!(g.n == mi && g.k == h && u.n == mi && u.k == h && d.n == h && d.k == mi,
                                    "FP4L {prefix} expert {e}: shapes gate {}x{} up {}x{} down {}x{} vs h {h} mi {mi}",
                                    g.n, g.k, u.n, u.k, d.n, d.k);
                    let gu_rm: Vec<u8> = g.packed.iter().chain(&u.packed).copied().collect();
                    let gus_rm: Vec<u8> = g.scales.iter().chain(&u.scales).copied().collect();
                    nv4_frag_into(&gu_rm, &gus_rm, 2 * mi, h, &mut gw[j * gu_w_e..(j + 1) * gu_w_e],
                                  &mut gsc[j * gu_s_e..(j + 1) * gu_s_e]);
                    nv4_frag_into(&d.packed, &d.scales, h, mi, &mut dw[j * dn_w_e..(j + 1) * dn_w_e],
                                  &mut dsc[j * dn_s_e..(j + 1) * dn_s_e]);
                    gsv[3 * j] = g.gs; gsv[3 * j + 1] = u.gs; gsv[3 * j + 2] = d.gs;
                }
                Ok(())
            }));
        }
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("FP4L expert load thread panicked")))).collect()
    });
    for r in res { r?; }
    let bad = gu_s.iter().chain(dn_s.iter()).filter(|&&b| b & 0x80 != 0 || b & 0x7F == 0x7F).count();
    anyhow::ensure!(bad == 0, "FP4L {prefix}: {bad} expert block scales are negative or NaN — refusing to serve them");
    Ok(FxMoe {
        gu_w: dev.htod_sync_copy(&gu_w)?, gu_s: dev.htod_sync_copy(&gu_s)?,
        dn_w: dev.htod_sync_copy(&dn_w)?, dn_s: dev.htod_sync_copy(&dn_s)?,
        gs: dev.htod_sync_copy(&gs)?, n_local: ne,
    })
}

/// Row-major NVFP4 ([n][k/2] nibbles, [n][k/16] E4M3 scales; n % 16 == 0, k % 128 == 0) -> the fragment-native
/// F128 layout the xf_nv4_* expert kernels read (gpu_fp4l.cu). A pure byte permutation (same sizes).
/// Codes W'[t][b][h][lane][16]: lane = gid*4 + tig holds row 16t+8h+gid, bytes s*8 + j = row byte 64b + 32s + 8tig + j.
/// Scales S'[t][b][lane][4]: byte 2h + s = row 16t+8h+gid's scale 8b + 4s + tig.
pub(crate) fn nv4_frag_into(codes: &[u8], scales: &[u8], n: usize, k: usize, w: &mut [u8], sc: &mut [u8]) {
    assert!(n % 16 == 0 && k % 128 == 0 && codes.len() == n * k / 2 && scales.len() == n * k / 16
            && w.len() == codes.len() && sc.len() == scales.len(), "nv4_frag: [{n}][{k}] shape");
    let (kb, k2, k16) = (k / 128, k / 2, k / 16);
    for t in 0..n / 16 {
        for h in 0..2 {
            for gid in 0..8 {
                let row = 16 * t + 8 * h + gid;
                for b in 0..kb {
                    for tig in 0..4 {
                        let lane = gid * 4 + tig;
                        for s2 in 0..2 {
                            let o = ((t * kb + b) * 2 + h) * 512 + lane * 16 + s2 * 8;
                            let i = row * k2 + 64 * b + 32 * s2 + 8 * tig;
                            w[o..o + 8].copy_from_slice(&codes[i..i + 8]);
                            sc[(t * kb + b) * 128 + lane * 4 + 2 * h + s2] = scales[row * k16 + 8 * b + 4 * s2 + tig];
                        }
                    }
                }
            }
        }
    }
}

/// Bytes one decode token streams through the FP4L routed experts (10 experts): the byte model.
pub(crate) fn fx_expert_bytes_per_token(h: usize, mi: usize, topk: usize) -> u64 {
    let per = 3 * h * mi / 2 + 3 * h * mi / 16;
    (topk * per) as u64
}

/// FP4L byte model of one plain decode step (W = 1): every dense FP4L weight once, the hyper-connection
/// mixers as loaded (f16, or int8 with --hc-int8), the router, top-k experts' NVFP4 bytes per layer, the head.
pub(crate) fn byte_model(layers: &[LayerDev], hc_mixer: &HcDev, lm_head: &Quad, topk: usize, h: usize, mi: usize) -> u64 {
    let qb = |q: &Quad| q.fx.as_deref().map_or(0, |f| f.bytes()) as u64;
    let hcb = |hc: &HcDev| match hc.q8.as_ref() {
        Some(q) => (q.down_q.len() + q.up_q.len()) as u64,
        None => 2 * (hc.down.len() + hc.up.len()) as u64,
    };
    let mut b = qb(lm_head) + hcb(hc_mixer);
    for l in layers {
        b += hcb(&l.hc_attn) + hcb(&l.hc_mlp);
        b += match &l.mixer {
            Mixer::Gdn(g) => qb(&g.in_qkv) + qb(&g.in_z) + qb(&g.out_proj)
                + g.ab_fx.as_deref().map_or(2 * (g.a_w.len() + g.b_w.len()) as u64, |ab| qb(&ab.0) + qb(&ab.1)),
            Mixer::Attn(a) => qb(&a.q_proj) + qb(&a.k_proj) + qb(&a.v_proj) + qb(&a.o_proj)
                + a.idx.as_ref().map_or(0, |i| qb(&i.qk)),
        };
        b += qb(&l.moe.sh_gate) + qb(&l.moe.sh_up) + qb(&l.moe.sh_down) + 2 * l.moe.router.len() as u64;
        if l.moe.fx.is_some() { b += fx_expert_bytes_per_token(h, mi, topk); }
    }
    b
}

/// The NVFP4 PLE table's shard geometry: (files, nibble offsets, scale offsets, rows/shard, gs).
pub(crate) fn fx_ple_table(fp: &Fp4lPack, ple_key: &str) -> Result<(Vec<std::fs::File>, PleFx, i64)> {
    let pre = format!("{ple_key}.ple_embedding.ngram_embedding");
    let gs = fp.scalar_f32(&format!("{pre}.weight_scale_2"))?;
    anyhow::ensure!(gs.is_finite() && gs > 0.0, "FP4L PLE weight_scale_2 {gs}");
    let mut files = Vec::new();
    let (mut w_off, mut s_off) = (Vec::new(), Vec::new());
    let mut rows = 0i64;
    for s in 0..128usize {
        let w = fp.get(&format!("{pre}.shard_{s}.weight"))?;
        let sc = fp.get(&format!("{pre}.shard_{s}.weight_scale"))?;
        anyhow::ensure!(w.shape.len() == 2 && w.shape[1] == 80 && sc.shape.len() == 2 && sc.shape[1] == 10 && w.shape[0] == sc.shape[0],
                        "FP4L PLE shard {s}: shapes {:?} / {:?} (want [rows, 80] / [rows, 10])", w.shape, sc.shape);
        anyhow::ensure!(w.file == sc.file, "FP4L PLE shard {s}: nibbles and scales live in different files (unsupported)");
        if s == 0 { rows = w.shape[0] as i64; }
        anyhow::ensure!(w.shape[0] as i64 == rows, "FP4L PLE shard {s}: {} rows, shard 0 has {rows}", w.shape[0]);
        files.push(fp.files[w.file as usize].try_clone()?);
        w_off.push(w.off);
        s_off.push(sc.off);
    }
    Ok((files, PleFx { w_off, s_off, gs }, rows))
}

/// Decode one FP4L PLE row (90 B: nibbles then scales) into 160 f16 bits: e2m1 * e4m3 * gs, one RN.
pub(crate) fn ple_fx_decode(raw: &[u8], gs: f32, out: &mut [u16]) {
    use crate::fp4l::pack::{e4m3, E2M1};
    for b in 0..10 {
        let s = e4m3(raw[80 + b]) * gs;
        for j in 0..8 {
            let byte = raw[b * 8 + j];
            out[b * 16 + 2 * j] = half::f16::from_f32(E2M1[(byte & 0xF) as usize] * s).to_bits();
            out[b * 16 + 2 * j + 1] = half::f16::from_f32(E2M1[(byte >> 4) as usize] * s).to_bits();
        }
    }
}

/// `--ple-ram ram` for the FP4L table: every shard's rows interleaved as 90-B records in RAM.
pub(crate) fn ple_fx_ram(files: &[std::fs::File], fx: &PleFx, rows: i64) -> Result<Vec<Vec<u8>>> {
    use std::os::unix::fs::FileExt;
    let rows = rows as usize;
    // BASICS (owner 2026-11-02): the table read as a percentage at the normal level
    let prog = crate::boot_report::Progress::new("loading the n-gram table into RAM", (files.len() * rows * PLE_FX_ROW) as u64,
                                                 crate::boot_report::Unit::Bytes);
    let prog = &prog;
    let res: Vec<Result<Vec<u8>>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..files.len()).map(|si| {
            let f = &files[si];
            let (wo, so) = (fx.w_off[si], fx.s_off[si]);
            s.spawn(move || -> Result<Vec<u8>> {
                let mut w = vec![0u8; rows * 80];
                let mut sc = vec![0u8; rows * 10];
                f.read_exact_at(&mut w, wo)?;
                f.read_exact_at(&mut sc, so)?;
                let mut out = vec![0u8; rows * PLE_FX_ROW];
                for r in 0..rows {
                    out[r * PLE_FX_ROW..r * PLE_FX_ROW + 80].copy_from_slice(&w[r * 80..(r + 1) * 80]);
                    out[r * PLE_FX_ROW + 80..(r + 1) * PLE_FX_ROW].copy_from_slice(&sc[r * 10..(r + 1) * 10]);
                }
                prog.add((rows * PLE_FX_ROW) as u64);
                Ok(out)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("PLE RAM thread panicked")))).collect()
    });
    prog.finish();
    res.into_iter().collect()
}
