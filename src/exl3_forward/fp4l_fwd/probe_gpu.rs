//! `--probe-fp4l-gpu`: the first GPU check of the FP4L kernels (real pack tensors, exact host reference). Probe path only; exit status 1 on any FAIL line.
//! See `mod.rs` for the file map.

use super::*;

// ================================================================================================
// --probe-fp4l-gpu: the FIRST GPU check of the FP4L kernels (PLAN/FP4L_GPU_RUNBOOK.md rung 1).
// Real pack tensors, deterministic activations, an exact host reference (f64 sums of the exactly
// dequantized weights times the f16 activations), every row count 1..=16:
//   FP4L_GPU_E2M1   cvt vs int E2M1 decode (every byte pair, both nibble orders) bitwise + vs the table
//   FP4L_GPU_MX8    xf_mx8_gemm: max rel err vs host per m; row 0 BITWISE equal at every m (batch invariance)
//   FP4L_GPU_F16W   xf_f16w_gemm (lm_head rows, MTP fc): same
//   FP4L_GPU_MOE    xf_nv4_moe (gate|up with per-projection gs, down): vs host, row 0 bitwise across m
//   FP4L_GPU_TILES  xf_nv4_tiles_m{1,2,4} (prefill): vs host on a synthetic tile list
//   FP4L_GPU_DEQ    xf_mx8_dequant / xf_nv4_dequant vs host bits
//   FP4L_GPU_TIME   per-kernel µs and achieved GB/s at m = 1 (cold-ish: 200 launches, rotating nothing)
// Exit status 1 on any FAIL line. No model load; ~3 GB of device memory.
// ================================================================================================

fn lcg_f16(seed: &mut u64, n: usize, scale: f32) -> Vec<u16> {
    (0..n).map(|_| {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((*seed >> 40) as f32) / ((1u64 << 24) as f32) - 0.5;
        half::f16::from_f32(u * 2.0 * scale).to_bits()
    }).collect()
}

/// host y[t][r] = sum_k w(r, k) * x[t][k] in f64 over exactly-dequantized weights.
fn host_gemm(wrow: &dyn Fn(usize, &mut [f32]), x: &[u16], m: usize, n: usize, k: usize) -> Vec<f64> {
    let xf: Vec<f32> = x.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
    let mut out = vec![0f64; m * n];
    let mut row = vec![0f32; k];
    for r in 0..n {
        wrow(r, &mut row);
        for t in 0..m {
            let xs = &xf[t * k..(t + 1) * k];
            let mut s = 0f64;
            for c in 0..k { s += row[c] as f64 * xs[c] as f64; }
            out[t * n + r] = s;
        }
    }
    out
}

fn rel_err(got: &[u16], want: &[f64]) -> (f64, f64) {
    let (mut d2, mut w2, mut mx) = (0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let g = half::f16::from_bits(*g).to_f32() as f64;
        d2 += (g - w) * (g - w); w2 += w * w;
        mx = mx.max((g - w).abs() / w.abs().max(1e-3));
    }
    ((d2 / w2.max(1e-300)).sqrt(), mx)
}

/// Probe only: rows [0, n) of a bf16 linear in the given representation (the lm_head lever's three ends).
fn probe_quad_bf(dev: &Arc<CudaDevice>, fp: &Fp4lPack, stem: &str, n: usize, mode: HeadPrec) -> Result<Quad> {
    let (n_full, k) = fp.shape2(&format!("{stem}.weight"))?;
    let n = n.min(n_full);
    let raw = fp.raw_range(&format!("{stem}.weight"), 0, (n * k * 2) as u64)?;
    let bits: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let fx = match mode {
        HeadPrec::Bf16 => FxW::Bf16 { w: dev.htod_sync_copy(&bf16_frag(&bits, n, k))? },
        HeadPrec::F16 => { let (f, osc) = bf16_bytes_to_f16(&raw, stem); FxW::F16 { w: dev.htod_sync_copy(&f)?, osc } }
        HeadPrec::Mx8 => {
            let vals: Vec<f32> = bits.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
            let (c, s) = mx8_rtn(&vals, n, k);
            mx8_dev(dev, &c, &s, n, k)?
        }
    };
    fx_quad_wrap(dev, fx, k, n, false)
}

/// The stored bf16 bits of `name` with every value below 2^-17 in magnitude set to +-0: such a tensor is
/// exactly representable in f16 too, so the trunk kernel (its f16 copy) and the twin (bf16) see the same values.
fn f16_safe_bf16(fp: &Fp4lPack, name: &str) -> Result<(Vec<u16>, Vec<u16>)> {
    let bits: Vec<u16> = fp.raw(name)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let bf: Vec<u16> = bits.iter().map(|&b| { let v = f32::from_bits((b as u32) << 16); if v.abs() < 2f32.powi(-17) { b & 0x8000 } else { b } }).collect();
    let f16: Vec<u16> = bf.iter().map(|&b| half::f16::from_f32(f32::from_bits((b as u32) << 16)).to_bits()).collect();
    for (a, b) in bf.iter().zip(&f16) {
        anyhow::ensure!(f32::from_bits((*a as u32) << 16) == half::f16::from_bits(*b).to_f32(), "{name}: a bf16 value is not f16-exact");
    }
    Ok((bf, f16))
}

/// An fp32 norm rounded to f16 and widened back: (f32 copy, f16 copy) of the same values.
fn f16_safe_norm(fp: &Fp4lPack, name: &str) -> Result<(Vec<f32>, Vec<u16>)> {
    let v = fp.plain_f32(name)?;
    let h: Vec<u16> = v.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
    Ok((h.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect(), h))
}

fn count_diff<T: PartialEq>(a: &[T], b: &[T]) -> usize { a.iter().zip(b).filter(|(x, y)| x != y).count() }

/// FP4L_GPU_TWIN: each precision twin against the trunk kernel it copies, bit for bit, on f16-exact inputs.
fn probe_twins(dev: &Arc<CudaDevice>, l: &Launcher, fp: &Fp4lPack, seed: &mut u64, fails: &mut usize) -> Result<()> {
    let (h, hcn, lr) = (2560usize, 4usize, 320usize);
    let rw = h * hcn;
    let mut line = |name: &str, bad: usize, of: usize| {
        if bad != 0 { *fails += 1; }
        println!("FP4L_GPU_TWIN {} {name}: {bad} bit mismatches vs the trunk kernel (of {of})", if bad == 0 { "OK" } else { "FAIL" });
    };
    let lay = "model.language_model.layers.0.attn_hyper_connection";
    let (nf32, n16) = f16_safe_norm(fp, &format!("{lay}.hc_norm.weight"))?;
    let (dn_bf, dn_16) = f16_safe_bf16(fp, &format!("{lay}.input_mix_weight_down.weight"))?;
    let (up_bf, up_16) = f16_safe_bf16(fp, &format!("{lay}.input_mix_weight_up.weight"))?;
    let (wi_bf, wi_16) = f16_safe_bf16(fp, &format!("{lay}.block_inject_weight.weight"))?;
    let (nf, nh) = (dev.htod_sync_copy(&nf32)?, dev.htod_sync_copy(&n16)?);
    let (dnb, dnh) = (dev.htod_sync_copy(&dn_bf)?, dev.htod_sync_copy(&dn_16)?);
    let (upb, uph) = (dev.htod_sync_copy(&up_bf)?, dev.htod_sync_copy(&up_16)?);
    let (wib, wih) = (dev.htod_sync_copy(&wi_bf)?, dev.htod_sync_copy(&wi_16)?);
    let bar = dev.htod_sync_copy(&[0i32; 2])?;
    let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
    // the served mixer's B32 copies of the same weights (xf_hc_fuse_b32)
    let (dnb32, upb32) = (dev.htod_sync_copy(&bf16_frag(&dn_bf, lr, rw))?, dev.htod_sync_copy(&bf16_frag(&up_bf, rw, lr))?);
    let hcf_cap = xf_co_residency("xf_hc_fuse_b32");
    let wsb = dev.alloc_zeros::<f32>(HCF_WS_PER_LR * lr)?;
    let mut hcf_fails = 0usize;
    let mut hcf_row0: Option<Vec<u16>> = None;
    for m in [1usize, 5, 16] {
        // f32 stream state + an f16 sublayer output + gates
        let resid0: Vec<f32> = lcg_f16(seed, m * rw, 4.0).iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
        let y: Vec<u16> = lcg_f16(seed, m * h, 2.0);
        let inj: Vec<f32> = (0..m * hcn).map(|i| 0.5 + 0.1 * i as f32).collect();
        let (yd, injd) = (dev.htod_sync_copy(&y)?, dev.htod_sync_copy(&inj)?);
        // norm (+ fused inject) : trunk vs twin, each on its own copy of the state
        for fused in [false, true] {
            for pf in [false, true] {
                let (mut ra, mut rb) = (dev.htod_sync_copy(&resid0)?, dev.htod_sync_copy(&resid0)?);
                let (mut ha, mut hb) = (dev.alloc_zeros::<u16>(m * rw)?, dev.alloc_zeros::<u16>(m * rw)?);
                let yp = if fused { dp(yd.device_ptr()) } else { 0 };
                if pf {
                    xqlaunch!(l, "xq_pf_hc_inj_norm", ((hcn * m) as u32, 1, 1), (128, 1, 1), 0,
                              (&mut ha, &mut ra, yp, &injd, &nh, h as i32, hcn as i32, m as i32, 1e-6f32))?;
                    pf_hc_inj_norm_f(l, dp(hb.device_ptr()), dp(rb.device_ptr()), yp, dp(injd.device_ptr()), dp(nf.device_ptr()), h, hcn, m, 1e-6)?;
                } else if fused {
                    xqlaunch!(l, "xq_hc_inj_norm", ((hcn * m) as u32, 1, 1), (1024, 1, 1), 0,
                              (&mut ha, &mut ra, yp, &injd, &nh, h as i32, hcn as i32, m as i32, 1e-6f32))?;
                    hc_inj_norm_f(l, dp(hb.device_ptr()), dp(rb.device_ptr()), yp, dp(injd.device_ptr()), &nf, h, hcn, m, 1e-6)?;
                } else {
                    xqlaunch!(l, "xq_hc_norm", ((hcn * m) as u32, 1, 1), (1024, 1, 1), 4096,
                              (&mut ha, &ra, &nh, h as i32, hcn as i32, m as i32, 1e-6f32))?;
                    hc_norm_f(l, dp(hb.device_ptr()), dp(rb.device_ptr()), &nf, h, hcn, m, 1e-6)?;
                }
                dev.synchronize()?;
                let (a, b) = (dev.dtoh_sync_copy(&ha)?, dev.dtoh_sync_copy(&hb)?);
                let (sa, sb) = (dev.dtoh_sync_copy(&ra)?, dev.dtoh_sync_copy(&rb)?);
                let bad = count_diff(&a, &b) + sa.iter().zip(&sb).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
                line(&format!("{} m={m}", match (pf, fused) { (true, _) => "xf_pf_hc_inj_norm_f", (false, true) => "xf_hc_inj_norm_f", _ => "xf_hc_norm_f" }),
                     bad, 2 * m * rw);
            }
        }
        // mixer (fused) + mix: trunk xq_hc_fuse + xq_hc_mix_rg vs the twins, same hn
        let hn: Vec<u16> = lcg_f16(seed, m * rw, 1.0);
        let hnd = dev.htod_sync_copy(&hn)?;
        let bx = 8u32;
        let (mut dda, mut uua) = (dev.alloc_zeros::<u16>(m * lr)?, dev.alloc_zeros::<u16>(m * rw)?);
        let (ddb, uub) = (dev.alloc_zeros::<u16>(m * lr)?, dev.alloc_zeros::<u16>(m * rw)?);
        xqlaunch!(l, "xq_hc_fuse", (bx, m as u32, 1), (256, 1, 1), 0,
                  (&mut dda, &mut uua, &hnd, &dnh, &uph, &bar, hcn as f32, lr as i32, rw as i32))?;
        hc_fuse_bf(l, dp(ddb.device_ptr()), dp(uub.device_ptr()), dp(hnd.device_ptr()), &dnb, &upb, dp(bar.device_ptr()), hcn as f32, lr, rw, bx, m)?;
        dev.synchronize()?;
        let bad = count_diff(&dev.dtoh_sync_copy(&dda)?, &dev.dtoh_sync_copy(&ddb)?) + count_diff(&dev.dtoh_sync_copy(&uua)?, &dev.dtoh_sync_copy(&uub)?);
        line(&format!("xf_hc_fuse_bf m={m}"), bad, m * (lr + rw));
        // the SERVED mixer xf_hc_fuse_b32 (tensor cores, B32): a different summation order than the scalar twin, so
        // rel-L2 vs the twin; bitwise across grid sizes (80 / 24 CTAs) and, on row 0, across m (batch invariance)
        {
            let g = hc_fuse_grid(hcf_cap * 5 / 6);
            anyhow::ensure!(g > 0, "xf_hc_fuse_b32 co-residency {hcf_cap} too small for the probe");
            let mut outs = Vec::new();
            for gg in [g, 24u32.min(g)] {
                let (ddc, uuc) = (dev.alloc_zeros::<u16>(m * lr)?, dev.alloc_zeros::<u16>(m * rw)?);
                hc_fuse_b32(l, dp(ddc.device_ptr()), dp(uuc.device_ptr()), dp(hnd.device_ptr()), &dnb32, &upb32, &wsb,
                            dp(bar.device_ptr()), hcn as f32, lr, rw, gg, m)?;
                dev.synchronize()?;
                outs.push(dev.dtoh_sync_copy(&uuc)?);
            }
            let twin: Vec<f64> = dev.dtoh_sync_copy(&uub)?.iter().map(|&b| half::f16::from_bits(b).to_f32() as f64).collect();
            let (rel, _) = rel_err(&outs[0], &twin);
            let gbad = count_diff(&outs[0], &outs[1]);
            // row 0 of every m sees a different hn (fresh per m): batch invariance is checked by re-running row 0 alone
            let (ddc, uuc) = (dev.alloc_zeros::<u16>(lr)?, dev.alloc_zeros::<u16>(rw)?);
            hc_fuse_b32(l, dp(ddc.device_ptr()), dp(uuc.device_ptr()), dp(hnd.device_ptr()), &dnb32, &upb32, &wsb,
                        dp(bar.device_ptr()), hcn as f32, lr, rw, g, 1)?;
            dev.synchronize()?;
            let bi = count_diff(&dev.dtoh_sync_copy(&uuc)?, &outs[0][..rw]);
            let ok = rel < 2e-3 && gbad == 0 && bi == 0;
            if !ok { hcf_fails += 1; }
            if m == 1 { hcf_row0 = Some(outs[0].clone()); }
            println!("FP4L_GPU_HCF {} xf_hc_fuse_b32 m={m} grid {g}: rel-L2 vs the scalar twin {rel:.2e}; grid {g} vs 24 bit-diffs {gbad}; row 0 alone vs in m={m} bit-diffs {bi}", if ok { "OK" } else { "FAIL" });
        }
        for pf in [false, true] {
            let (mut xa, xb) = (dev.alloc_zeros::<u16>(m * h)?, dev.alloc_zeros::<u16>(m * h)?);
            let (mut ia, ib) = (dev.alloc_zeros::<f32>(m * hcn)?, dev.alloc_zeros::<f32>(m * hcn)?);
            if pf {
                xqlaunch!(l, "xq_pf_hc_mix_2560x4", (m as u32, 1, 1), (128, 1, 1), 0, (&mut xa, &mut ia, &hnd, &uua, &wih, m as i32))?;
                pf_hc_mix_bf(l, dp(xb.device_ptr()), dp(ib.device_ptr()), dp(hnd.device_ptr()), dp(uua.device_ptr()), dp(wib.device_ptr()), m)?;
            } else {
                let gx = (hcn + h.div_ceil(1024)) as u32;
                xqlaunch!(l, "xq_hc_mix_rg", (gx, m as u32, 1), (1024, 1, 1), 0, (&mut xa, &mut ia, &hnd, &uua, &wih, h as i32, hcn as i32, m as i32))?;
                hc_mix_bf(l, dp(xb.device_ptr()), dp(ib.device_ptr()), dp(hnd.device_ptr()), dp(uua.device_ptr()), dp(wib.device_ptr()), h, hcn, m)?;
            }
            dev.synchronize()?;
            let (fa, fb) = (dev.dtoh_sync_copy(&ia)?, dev.dtoh_sync_copy(&ib)?);
            let bad = count_diff(&dev.dtoh_sync_copy(&xa)?, &dev.dtoh_sync_copy(&xb)?) + fa.iter().zip(&fb).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
            line(&format!("{} m={m}", if pf { "xf_pf_hc_mix_bf" } else { "xf_hc_mix_bf" }), bad, m * (h + hcn));
        }
    }
    // embedding gather: 16 token rows of the real table
    {
        let name = "model.language_model.embed_tokens.weight";
        let (n_full, _) = fp.shape2(name)?;
        let toks: Vec<i32> = (0..16).map(|i| ((i as u64 * 15_485_863) % n_full as u64) as i32).collect();
        let rows: Vec<u8> = toks.iter().flat_map(|&t| fp.raw_range(name, (t as usize * h * 2) as u64, ((t as usize + 1) * h * 2) as u64).unwrap_or_default()).collect();
        anyhow::ensure!(rows.len() == 16 * h * 2, "embedding rows read");
        let bf: Vec<u16> = rows.chunks_exact(2).map(|c| { let b = u16::from_le_bytes([c[0], c[1]]); if f32::from_bits((b as u32) << 16).abs() < 2f32.powi(-17) { b & 0x8000 } else { b } }).collect();
        let f16: Vec<u16> = bf.iter().map(|&b| half::f16::from_f32(f32::from_bits((b as u32) << 16)).to_bits()).collect();
        // a 16-row table (row i = token i's row) and tokens 0..16
        let (tb, th) = (dev.htod_sync_copy(&bf)?, dev.htod_sync_copy(&f16)?);
        let td = dev.htod_sync_copy(&(0..16).collect::<Vec<i32>>())?;
        let (mut ra, rb) = (dev.alloc_zeros::<f32>(16 * rw)?, dev.alloc_zeros::<f32>(16 * rw)?);
        xqlaunch!(l, "xq_embed_resid", (((rw * 16) as u32) / 256 + 1, 1, 1), (256, 1, 1), 0, (&mut ra, &th, &td, h as i32, hcn as i32, 16i32))?;
        embed_resid_bf(l, dp(rb.device_ptr()), &tb, dp(td.device_ptr()), h, hcn, 16)?;
        dev.synchronize()?;
        let (a, b) = (dev.dtoh_sync_copy(&ra)?, dev.dtoh_sync_copy(&rb)?);
        line("xf_embed_resid_bf m=16", a.iter().zip(&b).filter(|(p, q)| p.to_bits() != q.to_bits()).count(), 16 * rw);
    }
    // n-gram norms (fp32) and conv (fp32 taps) — the trunk's layer-1 PLE tensors
    {
        let pk = "model.language_model.layers.1.ple";
        let m = 5usize;
        for (which, nm) in [(0u8, "norm_key"), (1, "norm_query"), (2, "norm_conv")] {
            let (wf, wh) = f16_safe_norm(fp, &format!("{pk}.{nm}.weight"))?;
            let (wfd, whd) = (dev.htod_sync_copy(&wf)?, dev.htod_sync_copy(&wh)?);
            let x16 = dev.htod_sync_copy(&lcg_f16(seed, m * 4 * 2560, 2.0))?;
            let x32: Vec<f32> = lcg_f16(seed, m * 4 * 2560, 2.0).iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
            let x32d = dev.htod_sync_copy(&x32)?;
            let bad = match which {
                0 | 1 => {
                    let (mut oa, ob) = (dev.alloc_zeros::<f32>(m * 4 * 2560)?, dev.alloc_zeros::<f32>(m * 4 * 2560)?);
                    if which == 0 {
                        xqlaunch!(l, "xq_ple_norm_f16", ((m * 4) as u32, 1, 1), (256, 1, 1), 0, (&x16, &whd, &mut oa, 1e-6f32))?;
                        ple_norm_f(l, 0, dp(x16.device_ptr()), &wfd, dp(ob.device_ptr()), m, 1e-6)?;
                    } else {
                        xqlaunch!(l, "xq_ple_norm_f32", ((m * 4) as u32, 1, 1), (256, 1, 1), 0, (&x32d, &whd, &mut oa, 1e-6f32))?;
                        ple_norm_f(l, 1, dp(x32d.device_ptr()), &wfd, dp(ob.device_ptr()), m, 1e-6)?;
                    }
                    dev.synchronize()?;
                    let (a, b) = (dev.dtoh_sync_copy(&oa)?, dev.dtoh_sync_copy(&ob)?);
                    a.iter().zip(&b).filter(|(p, q)| p.to_bits() != q.to_bits()).count()
                }
                _ => {
                    let (mut oa, ob) = (dev.alloc_zeros::<u16>(m * 4 * 2560)?, dev.alloc_zeros::<u16>(m * 4 * 2560)?);
                    xqlaunch!(l, "xq_ple_norm_hh", ((m * 4) as u32, 1, 1), (256, 1, 1), 0, (&x16, &whd, &mut oa, 1e-6f32))?;
                    ple_norm_f(l, 2, dp(x16.device_ptr()), &wfd, dp(ob.device_ptr()), m, 1e-6)?;
                    dev.synchronize()?;
                    count_diff(&dev.dtoh_sync_copy(&oa)?, &dev.dtoh_sync_copy(&ob)?)
                }
            };
            line(&format!("xf_ple_{nm}_f m={m}"), bad, m * 4 * 2560);
        }
        // conv: decode step and chunk (f16-safe taps as f16 / as f32)
        let (cb, c16) = f16_safe_bf16(fp, &format!("{pk}.conv1d.weight"))?;
        let cf: Vec<f32> = cb.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
        let (cfd, chd) = (dev.htod_sync_copy(&cf)?, dev.htod_sync_copy(&c16)?);
        let gated = dev.htod_sync_copy(&lcg_f16(seed, m * 10240, 1.0))?;
        let normed = dev.htod_sync_copy(&lcg_f16(seed, m * 10240, 1.0))?;
        let state0 = lcg_f16(seed, 2 * 10240 * 9, 1.0);
        let slots = dev.htod_sync_copy(&(0..m).flat_map(|q| [(q % 2) as i32, 0]).collect::<Vec<i32>>())?;
        let r0: Vec<f32> = lcg_f16(seed, m * 10240, 3.0).iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
        for chunk in [false, true] {
            let (mut ra, rb) = (dev.htod_sync_copy(&r0)?, dev.htod_sync_copy(&r0)?);
            let (mut sa, sb) = (dev.htod_sync_copy(&state0)?, dev.htod_sync_copy(&state0)?);
            if chunk {
                xqlaunch!(l, "xq_ple_conv_chunk", ((10240u32 + 255) / 256, 1, 1), (256, 1, 1), 0, (&mut ra, &gated, &normed, &mut sa, &chd, 1i32, m as i32))?;
                ple_conv_chunk_f(l, dp(rb.device_ptr()), dp(gated.device_ptr()), dp(normed.device_ptr()), dp(sb.device_ptr()), &cfd, 1, m)?;
            } else {
                xqlaunch!(l, "xq_ple_conv", (((m * 10240) as u32 + 255) / 256, 1, 1), (256, 1, 1), 0, (&mut ra, &gated, &normed, &sa, &chd, m as i32, &slots))?;
                ple_conv_f(l, dp(rb.device_ptr()), dp(gated.device_ptr()), dp(normed.device_ptr()), dp(sb.device_ptr()), &cfd, m, dp(slots.device_ptr()))?;
            }
            dev.synchronize()?;
            let (a, b) = (dev.dtoh_sync_copy(&ra)?, dev.dtoh_sync_copy(&rb)?);
            let bad = a.iter().zip(&b).filter(|(p, q)| p.to_bits() != q.to_bits()).count()
                + count_diff(&dev.dtoh_sync_copy(&sa)?, &dev.dtoh_sync_copy(&sb)?);
            line(&format!("{} m={m}", if chunk { "xf_ple_conv_chunk_f" } else { "xf_ple_conv_f" }), bad, m * 10240);
        }
    }
    let _ = hcf_row0;
    *fails += hcf_fails;
    Ok(())
}

/// FP4L_GPU_SG: xf_sg_gate's channels through the trunk's combine kernels (decode xq_moe_combine, prefill
/// xq_moe_combine_rows, the TP vectorised tree xq_moe_combine_rows_ep2_k10) against the same combines fed an
/// f16 gate whose single product IS the logit (sg = s * e0 with s a power of two, so L = s * x0 is exact in
/// both paths): the outputs must agree bit for bit (the encoding + every tree rebuild L exactly).
fn probe_sg_gate(dev: &Arc<CudaDevice>, l: &Launcher, seed: &mut u64, fails: &mut usize) -> Result<()> {
    let h = 2560usize;
    let m = 16usize;
    let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
    let sgc = dev.htod_sync_copy(&sg_channel_constants(h))?;
    let mut x: Vec<u16> = lcg_f16(seed, m * h, 30.0);
    // row 0: zero logit; row 1: a huge one (clamped path); row 2: tiny
    x[0] = 0; x[h] = half::f16::from_f32(65504.0).to_bits(); x[2 * h] = half::f16::from_f32(3e-7).to_bits();
    let xd = dev.htod_sync_copy(&x)?;
    let ysh = dev.htod_sync_copy(&vec![half::f16::from_f32(1.0).to_bits(); m * h])?;
    let ids = dev.htod_sync_copy(&vec![0i32; m])?;            // k = 1, expert 0 ...
    let slotmap = dev.htod_sync_copy(&vec![-1i32; 512])?;     // ... absent: only the shared term
    let wts = dev.htod_sync_copy(&vec![0f32; m])?;
    let cand = dev.htod_sync_copy(&vec![-1i32; m])?;
    let yd = dev.alloc_zeros::<u16>(m * h)?;
    let mut total_bad = 0usize;
    for s_exp in [0i32, -6, -12, -20, 1] {
        let s = 2f32.powi(s_exp);
        let mut sg_bf = vec![0u16; h];
        sg_bf[0] = (s.to_bits() >> 16) as u16;
        let mut sg16 = vec![0u16; h];
        sg16[0] = half::f16::from_f32(s).to_bits();
        let (sgb, sgh) = (dev.htod_sync_copy(&sg_bf)?, dev.htod_sync_copy(&sg16)?);
        let xg = dev.htod_sync_copy(&vec![0u16; m * h])?;
        sg_gate(l, dp(xg.device_ptr()), &sgb, dp(xd.device_ptr()), h, m)?;
        let run = |name: &str, sgp: u64, xp: u64| -> Result<Vec<u32>> {
            match name {
                "xq_moe_combine" => {
                    let mut o = dev.alloc_zeros::<u16>(m * h)?;
                    xqlaunch!(l, "xq_moe_combine", (m as u32, 1, 1), (1024, 1, 1), 4096, (&mut o, &yd, &ysh, &ids, &wts, &slotmap, sgp, xp, 1i32, h as i32, m as i32))?;
                    dev.synchronize()?;
                    Ok(dev.dtoh_sync_copy(&o)?.iter().map(|&v| v as u32).collect())
                }
                "xq_moe_combine_rows" => {
                    let mut o = dev.alloc_zeros::<u16>(m * h)?;
                    xqlaunch!(l, "xq_moe_combine_rows", (m as u32, 1, 1), (1024, 1, 1), 4096, (&mut o, &yd, &ysh, &cand, &wts, sgp, xp, 1i32, h as i32, m as i32))?;
                    dev.synchronize()?;
                    Ok(dev.dtoh_sync_copy(&o)?.iter().map(|&v| v as u32).collect())
                }
                _ => {
                    let mut o = dev.alloc_zeros::<f32>(m * h)?;
                    xqlaunch!(l, "xq_moe_combine_rows_ep2_k10", (m as u32, 1, 1), (320, 1, 1), 0, (&mut o, &yd, &ysh, &cand, &wts, sgp, xp, 1i32, h as i32, m as i32, 1i32))?;
                    dev.synchronize()?;
                    Ok(dev.dtoh_sync_copy(&o)?.iter().map(|v| v.to_bits()).collect())
                }
            }
        };
        for name in ["xq_moe_combine", "xq_moe_combine_rows", "xq_moe_combine_rows_ep2_k10"] {
            let a = run(name, dp(sgh.device_ptr()), dp(xd.device_ptr()))?;
            let b = run(name, dp(sgc.device_ptr()), dp(xg.device_ptr()))?;
            total_bad += count_diff(&a, &b);
        }
        // the channels themselves vs the host model of the encoding (L = s * x0 exactly)
        let got = dev.dtoh_sync_copy(&xg)?;
        for r in 0..m {
            let want = sg_encode(s * half::f16::from_bits(x[r * h]).to_f32());
            total_bad += count_diff(&got[r * h..r * h + 6], &want);
            total_bad += got[r * h + 6..(r + 1) * h].iter().filter(|&&v| v != 0).count();
        }
    }
    if total_bad != 0 { *fails += 1; }
    println!("FP4L_GPU_SG {} shared-gate channels: {total_bad} mismatches over 5 gate scales x 16 rows x 3 combine kernels + the \
              encoding vs its host model (0 = every combine rebuilds the logit bit for bit)", if total_bad == 0 { "OK" } else { "FAIL" });
    Ok(())
}

pub fn probe_gpu(dir: &str) -> Result<()> {
    let fp = Fp4lPack::open(std::path::Path::new(dir))?;
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    crate::exl3_bench::load_module_pub(&dev)?;
    load_module(&dev)?;
    let s = crate::gpu::fork_blocking_stream(&dev);
    let l = Launcher { dev: &dev, stream: &s };
    let mut fails = 0usize;
    let mut seed = 0x5EED_F4F4u64;

    // ---- E2M1: every 16-bit word pattern through both decoders (8 nibbles per u32 word)
    {
        let n = 1usize << 16;
        let words: Vec<u32> = (0..n as u32).map(|v| v | (v.rotate_left(7) << 16)).collect();
        let w = dev.htod_sync_copy(&words)?;
        let mut oc = dev.alloc_zeros::<u32>(4 * n)?;
        let mut oi = dev.alloc_zeros::<u32>(4 * n)?;
        xflaunch!(l, "xf_e2m1_selftest", ((n as u32).div_ceil(256), 1, 1), (256, 1, 1), 0, (&w, &mut oc, &mut oi, n as i32))?;
        dev.synchronize()?;
        let (hc, hi) = (dev.dtoh_sync_copy(&oc)?, dev.dtoh_sync_copy(&oi)?);
        let mut bad_ci = 0usize;
        let mut bad_tab = 0usize;
        for i in 0..n {
            for j in 0..4 {
                if hc[4 * i + j] != hi[4 * i + j] { bad_ci += 1; }
                let byte = (words[i] >> (8 * j)) & 0xFF;
                let want_lo = half::f16::from_f32(crate::fp4l::pack::E2M1[(byte & 0xF) as usize]).to_bits() as u32;
                let want_hi = half::f16::from_f32(crate::fp4l::pack::E2M1[(byte >> 4) as usize]).to_bits() as u32;
                if hc[4 * i + j] != (want_lo | (want_hi << 16)) { bad_tab += 1; }
            }
        }
        let ok = bad_ci == 0 && bad_tab == 0;
        if !ok { fails += 1; }
        println!("FP4L_GPU_E2M1 {} cvt-vs-int mismatches {bad_ci}, cvt-vs-table mismatches {bad_tab} (of {})",
                 if ok { "OK" } else { "FAIL" }, 4 * n);
    }

    // ---- dense MXFP8 / f16 GEMMs on real tensors
    let ws = dev.alloc_zeros::<f32>(2_000_000)?; // == Scratch::chain_ws
    // `mode` (bf16 linears only): build the stated representation (the lm_head lever's three ends) instead of the
    // loader's default; Mx8 is lossy by design, so its line reports the error without a pass bound.
    let mut dense = |stem: &str, label: &str, rows_cap: Option<usize>, mode: Option<HeadPrec>| -> Result<()> {
        let kind = fp.lin_kind(stem)?;
        let (n_full, kk) = fp.shape2(&format!("{stem}.weight"))?;
        let n = rows_cap.map_or(n_full, |c| c.min(n_full));
        let q = match mode { Some(md) => probe_quad_bf(&dev, &fp, stem, n, md)?, None => fx_quad(&dev, &fp, stem, None, Some(&[(0, n)]), false)? };
        let lossy = mode == Some(HeadPrec::Mx8);
        let k = q.k as usize;
        let mx = if kind == LinKind::Mxfp8 { Some(fp.mxfp8_rows(stem, 0, n)?) } else { None };
        let bf = if kind == LinKind::Bf16 { Some(fp.plain_f32_range(&format!("{stem}.weight"), 0, n * kk)?) } else { None };
        let wrow = |r: usize, out: &mut [f32]| {
            if let Some(m) = mx.as_ref() { m.row_f32(r, out); }
            else if let Some(b) = bf.as_ref() {
                out[..k].copy_from_slice(&b[r * k..(r + 1) * k]); // the STORED bf16 values: the f16 copy must be exact
            }
        };
        let x_host = lcg_f16(&mut seed, 16 * k, 1.0);
        let want = host_gemm(&wrow, &x_host, 16, n, k);
        let x = dev.htod_sync_copy(&x_host)?;
        let mut row0: Option<Vec<u16>> = None;
        let (mut worst_rel, mut worst_mx, mut bitdiff) = (0f64, 0f64, 0usize);
        for m in 1..=16usize {
            let mut y = dev.alloc_zeros::<u16>(m * n)?;
            fx_chain(&l, &q, q.fx.as_deref().unwrap(), &x, &mut y, m, &ws, None)?;
            dev.synchronize()?;
            let got = dev.dtoh_sync_copy(&y)?;
            let (r, mxe) = rel_err(&got, &want[..m * n]);
            worst_rel = worst_rel.max(r); worst_mx = worst_mx.max(mxe);
            match &row0 { None => row0 = Some(got[..n].to_vec()), Some(r0) => bitdiff += r0.iter().zip(&got[..n]).filter(|(a, b)| a != b).count() }
        }
        // prefill row counts: the exact wide kernel (MXFP8) / cuBLASLt with alpha (f16)
        let (mut wide_rel, wide_ms) = (0f64, [17usize, 48, 100]);
        let mut tc_vs_wide: Option<usize> = None;
        {
            let xw_host = lcg_f16(&mut seed, 100 * k, 1.0);
            let want_w = host_gemm(&wrow, &xw_host, 100, n, k);
            let xw = dev.htod_sync_copy(&xw_host)?;
            let rws = dev.alloc_zeros::<u16>(n * k)?;
            let ltws = dev.alloc_zeros::<u8>(32 << 20)?;
            let rc = dev.alloc_zeros::<u8>(256)?;
            let rb = ReconBufs { ws: *rws.device_ptr() as u64, ws_len: rws.len(), rc: *rc.device_ptr() as u64,
                                 ltws: *ltws.device_ptr() as u64, ltws_len: ltws.len() };
            for &m in &wide_ms {
                let mut y = dev.alloc_zeros::<u16>(m * n)?;
                fx_chain(&l, &q, q.fx.as_deref().unwrap(), &xw, &mut y, m, &ws, Some(&rb))?;
                dev.synchronize()?;
                let got = dev.dtoh_sync_copy(&y)?;
                wide_rel = wide_rel.max(rel_err(&got, &want_w[..m * n]).0);
                // MXFP8: the served tiled kernel (xf_mx8_tc) must equal the simple wide kernel bit for bit
                if let (Some(FxW::Mx8 { w, s }), true) = (q.fx.as_deref(), !prefill_dense_lt()) {
                    let mut yw = dev.alloc_zeros::<u16>(m * n)?;
                    xflaunch!(l, "xf_mx8_wide", (n.div_ceil(128) as u32, m.div_ceil(64) as u32, 1), (256, 1, 1), 0,
                              (w, s, &xw, &mut yw, 0u64, m as i32, k as i32, n as i32))?;
                    dev.synchronize()?;
                    *tc_vs_wide.get_or_insert(0) += count_diff(&got, &dev.dtoh_sync_copy(&yw)?);
                }
            }
        }
        // timing at m = 1
        let mut y = dev.alloc_zeros::<u16>(n)?;
        dev.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..200 { fx_chain(&l, &q, q.fx.as_deref().unwrap(), &x, &mut y, 1, &ws, None)?; }
        dev.synchronize()?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / 200.0;
        let bytes = q.fx.as_deref().unwrap().bytes() as f64;
        let ok = lossy || (worst_rel < 2e-3 && bitdiff == 0 && wide_rel < 2e-3 && tc_vs_wide.unwrap_or(0) == 0);
        if !ok { fails += 1; }
        println!("{label} {} {stem} as {} [{n}x{k}] ks {}: rel-L2 vs host (stored values) <= {worst_rel:.2e} (max elem {worst_mx:.2e}); \
                  row 0 bit-diffs across m=1..16: {bitdiff}; prefill m {wide_ms:?} rel-L2 <= {wide_rel:.2e}{}; m=1 {us:.1} us = {:.0} GB/s{}",
                 if ok { "OK" } else { "FAIL" }, q.fx.as_deref().unwrap().kind(), q.ks,
                 tc_vs_wide.map_or(String::new(), |d| format!(" (xf_mx8_tc vs xf_mx8_wide bit-diffs {d})")), bytes / us / 1e3,
                 if lossy { " (lossy by design: no pass bound)" } else { "" });
        Ok(())
    };
    dense("model.language_model.layers.0.linear_attn.in_proj_qkv", "FP4L_GPU_MX8", None, None)?;
    dense("model.language_model.layers.0.linear_attn.in_proj_z", "FP4L_GPU_MX8", None, None)?;
    dense("model.language_model.layers.0.linear_attn.out_proj", "FP4L_GPU_MX8", None, None)?;
    dense("model.language_model.layers.3.self_attn.q_proj", "FP4L_GPU_MX8", None, None)?;
    dense("model.language_model.layers.3.self_attn.k_proj", "FP4L_GPU_MX8", None, None)?;
    dense("model.language_model.layers.3.self_attn.o_proj", "FP4L_GPU_MX8", None, None)?;
    dense("model.language_model.layers.0.mlp.shared_expert.down_proj", "FP4L_GPU_MX8", None, None)?;
    // the lm_head lever's three ends on the first 16384 rows (bf16 = stored, f16 = the shift, mxfp8 = RTN), MTP linears
    dense("lm_head", "FP4L_GPU_HEAD", Some(16384), Some(HeadPrec::Bf16))?;
    dense("lm_head", "FP4L_GPU_HEAD", Some(16384), Some(HeadPrec::F16))?;
    dense("lm_head", "FP4L_GPU_HEAD", Some(16384), Some(HeadPrec::Mx8))?;
    dense("mtp.fc_hidden", "FP4L_GPU_HEAD", None, Some(HeadPrec::Bf16))?;
    dense("mtp.fc_hidden", "FP4L_GPU_HEAD", None, Some(HeadPrec::F16))?;

    // ---- paired MXFP8 GEMMs (xf_mx8_gemm_x2): each member bitwise its own xf_mx8_gemm, m = 1..16
    for (p0, p1) in [("model.language_model.layers.0.linear_attn.in_proj_a", "model.language_model.layers.0.linear_attn.in_proj_b"),
                     ("model.language_model.layers.0.mlp.shared_expert.gate_proj", "model.language_model.layers.0.mlp.shared_expert.up_proj"),
                     ("model.language_model.layers.3.self_attn.k_proj", "model.language_model.layers.3.self_attn.v_proj")] {
        let q0 = fx_quad(&dev, &fp, p0, None, None, false)?;
        let q1 = fx_quad(&dev, &fp, p1, None, None, false)?;
        let (k, n) = (q0.k as usize, q0.n as usize);
        let x = dev.htod_sync_copy(&lcg_f16(&mut seed, 16 * k, 1.0))?;
        let half = 16 * q0.ks.max(1) as usize * n;
        let pws = dev.alloc_zeros::<f32>(2 * half)?;
        let (mut bad, mut taken) = (0usize, 0usize);
        for m in 1..=16usize {
            let (mut a0, mut a1) = (dev.alloc_zeros::<u16>(m * n)?, dev.alloc_zeros::<u16>(m * q1.n as usize)?);
            let (mut b0, mut b1) = (dev.alloc_zeros::<u16>(m * n)?, dev.alloc_zeros::<u16>(m * q1.n as usize)?);
            fx_chain(&l, &q0, q0.fx.as_deref().unwrap(), &x, &mut a0, m, &ws, None)?;
            fx_chain(&l, &q1, q1.fx.as_deref().unwrap(), &x, &mut a1, m, &ws, None)?;
            let wsp = *pws.device_ptr() as u64;
            if fx_pair(&l, &q0, &q1, *x.device_ptr() as u64, *b0.device_ptr() as u64, *b1.device_ptr() as u64,
                       (wsp, half), (wsp + (half * 4) as u64, half), m)? { taken += 1; }
            dev.synchronize()?;
            bad += count_diff(&dev.dtoh_sync_copy(&a0)?, &dev.dtoh_sync_copy(&b0)?);
            bad += count_diff(&dev.dtoh_sync_copy(&a1)?, &dev.dtoh_sync_copy(&b1)?);
        }
        let ok = bad == 0 && taken == 16;
        if !ok { fails += 1; }
        println!("FP4L_GPU_PAIR {} {p0} | {p1} [{n}x{k}] ks {}: xf_mx8_gemm_x2 taken at {taken}/16 widths; bit-diffs vs two \
                  xf_mx8_gemm launches over m = 1..16: {bad}", if ok { "OK" } else { "FAIL" }, q0.ks);
    }

    // ---- the stored-bf16 plain GEMMs (router fp32, n-gram projections, hc mixers) vs the host; batch invariance
    {
        let fxs = FxScratch::with_dims(&dev, 2560, 4, 320, 512, 16)?;   // the Flash-Next geometry
        let mut plain = |name: &str, k: usize, n: usize, f32out: bool| -> Result<()> {
            let bits: Vec<u16> = fp.raw(name)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            anyhow::ensure!(bits.len() == n * k, "{name}: {} values != {n} x {k}", bits.len());
            let w = dev.htod_sync_copy(&bf16_frag(&bits, n, k))?;
            let vals: Vec<f32> = bits.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
            let wrow = |r: usize, o: &mut [f32]| o[..k].copy_from_slice(&vals[r * k..(r + 1) * k]);
            let x_host = lcg_f16(&mut seed, 100 * k, 1.0);
            let want = host_gemm(&wrow, &x_host, 100, n, k);
            let x = dev.htod_sync_copy(&x_host)?;
            let (ws, cnt) = if n == 512 { (&fxs.ws_r, &fxs.cnt_r) } else if k == 2560 { (&fxs.ws_p, &fxs.cnt_p) } else { (&fxs.ws_h, &fxs.cnt_h) };
            let (mut worst, mut bitdiff, mut tcdiff) = (0f64, 0usize, 0usize);
            let mut row0: Option<Vec<u32>> = None;
            for m in (1..=16usize).chain([17, 48, 100]) {
                let (y16, yf) = (dev.alloc_zeros::<u16>(m * n)?, dev.alloc_zeros::<f32>(m * n)?);
                let (yp, yfp) = if f32out { (0u64, *yf.device_ptr() as u64) } else { (*y16.device_ptr() as u64, 0u64) };
                bf_gemm(&l, &w, *x.device_ptr() as u64, yp, yfp, m, k, n, ws, cnt)?;
                dev.synchronize()?;
                let got: Vec<f32> = if f32out { dev.dtoh_sync_copy(&yf)? }
                                    else { dev.dtoh_sync_copy(&y16)?.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect() };
                let (mut d2, mut w2) = (0f64, 0f64);
                for (g, w) in got.iter().zip(&want[..m * n]) { d2 += (*g as f64 - w).powi(2); w2 += w * w; }
                worst = worst.max((d2 / w2.max(1e-300)).sqrt());
                if m <= 16 {
                    let r0: Vec<u32> = got[..n].iter().map(|v| v.to_bits()).collect();
                    match &row0 { None => row0 = Some(r0), Some(a) => bitdiff += a.iter().zip(&r0).filter(|(p, q)| p != q).count() }
                } else {
                    // m > 16: the served tiled xf_bf16_tc must equal the simple xf_bf16_wide bit for bit
                    let (y16b, yfb) = (dev.alloc_zeros::<u16>(m * n)?, dev.alloc_zeros::<f32>(m * n)?);
                    let (yp2, yfp2) = if f32out { (0u64, *yfb.device_ptr() as u64) } else { (*y16b.device_ptr() as u64, 0u64) };
                    xflaunch!(l, "xf_bf16_wide", (n.div_ceil(128) as u32, m.div_ceil(64) as u32, 1), (256, 1, 1), 0,
                              (&w, *x.device_ptr() as u64, yp2, yfp2, m as i32, k as i32, n as i32))?;
                    dev.synchronize()?;
                    let other: Vec<f32> = if f32out { dev.dtoh_sync_copy(&yfb)? }
                                          else { dev.dtoh_sync_copy(&y16b)?.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect() };
                    tcdiff += got.iter().zip(&other).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
                }
            }
            // a pure-f32 output keeps fp32 accumulation error only (~1e-6); f16 output adds its own rounding (~3e-4)
            let bound = if f32out { 1e-5 } else { 2e-3 };
            let ok = worst < bound && bitdiff == 0 && tcdiff == 0;
            if !ok { fails += 1; }
            println!("FP4L_GPU_BF16 {} {name} [{n}x{k}] ks {} ({} out): rel-L2 vs host (stored bf16) <= {worst:.2e} at m = 1..16, 17, 48, 100; \
                      row 0 bit-diffs across m = 1..16: {bitdiff}; xf_bf16_tc vs xf_bf16_wide bit-diffs {tcdiff}",
                     if ok { "OK" } else { "FAIL" }, bf_ks(k, n), if f32out { "fp32" } else { "f16" });
            Ok(())
        };
        plain("model.language_model.layers.0.mlp.gate.weight", 2560, 512, true)?;
        plain("model.language_model.layers.1.ple.key_proj.weight", 2560, 10240, false)?;
        plain("model.language_model.layers.1.ple.value_proj.weight", 2560, 2560, false)?;
        plain("model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight", 10240, 320, false)?;
        plain("model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight", 320, 10240, false)?;
    }

    // ---- the TWINS: on weights exactly representable in f16 (the stored values with |v| < 2^-17 set to 0, and
    // the fp32 norms rounded to f16 and widened back) every twin must return the trunk kernel's bits
    probe_twins(&dev, &l, &fp, &mut seed, &mut fails)?;
    // ---- the shared-gate channels through the trunk's own combine kernels (bitwise vs an f16-expressible gate)
    probe_sg_gate(&dev, &l, &mut seed, &mut fails)?;

    // ---- NVFP4 experts (decode / verify kernel)
    {
        let (h, mi) = (2560usize, 640usize);
        let local: Vec<usize> = (0..512).collect();
        let fx = fx_experts(&dev, &fp, "model.language_model.layers.0.mlp", &local, h, mi)?;
        let picks: Vec<usize> = vec![0, 7, 255, 511, 100, 300, 42, 9, 77, 400];
        let hosts: Vec<(crate::fp4l::pack::Nv4Host, crate::fp4l::pack::Nv4Host, crate::fp4l::pack::Nv4Host)> = picks.iter().map(|&e| -> Result<_> {
            let p = format!("model.language_model.layers.0.mlp.experts.{e}");
            Ok((fp.nvfp4(&format!("{p}.gate_proj"))?, fp.nvfp4(&format!("{p}.up_proj"))?, fp.nvfp4(&format!("{p}.down_proj"))?))
        }).collect::<Result<_>>()?;
        let x_host = lcg_f16(&mut seed, 16 * h, 1.0);
        let din_host = lcg_f16(&mut seed, picks.len() * 16 * mi, 0.5);
        let x = dev.htod_sync_copy(&x_host)?;
        let idx: Vec<i32> = picks.iter().map(|&e| e as i32).collect();
        let idxmap = dev.htod_sync_copy(&idx)?;
        let esel = dev.htod_sync_copy(&[picks.len() as i32])?;
        let f = nv4_moe_fn();
        let (mut worst, mut bitdiff) = (0f64, 0usize);
        let mut row0: Option<Vec<u16>> = None;
        for m in 1..=16usize {
            let mut y = dev.alloc_zeros::<u16>(picks.len() * m * 2 * mi)?;
            xflaunch!(l, f, (((2 * mi) / 64) as u32, picks.len() as u32, 1), (128, 1, 1), 0,
                      (&fx.gu_w, &fx.gu_s, &fx.gs, &x, 0i64, &mut y, &idxmap, &esel, m as i32, h as i32, (2 * mi) as i32, mi as i32))?;
            // down on a fixed din [slot][m][mi] (rows of the 16-row host tensor)
            let din_m: Vec<u16> = (0..picks.len()).flat_map(|s| din_host[s * 16 * mi..s * 16 * mi + m * mi].to_vec()).collect();
            let din = dev.htod_sync_copy(&din_m)?;
            let mut yd = dev.alloc_zeros::<u16>(picks.len() * m * h)?;
            xflaunch!(l, f, ((h / 64) as u32, picks.len() as u32, 1), (128, 1, 1), 0,
                      (&fx.dn_w, &fx.dn_s, &fx.gs, &din, (m * mi) as i64, &mut yd, &idxmap, &esel, m as i32, mi as i32, h as i32, h as i32))?;
            dev.synchronize()?;
            let (gy, gd) = (dev.dtoh_sync_copy(&y)?, dev.dtoh_sync_copy(&yd)?);
            for (si, (g, u, d)) in hosts.iter().enumerate() {
                let wg = |r: usize, o: &mut [f32]| if r < mi { g.row_f32(r, o) } else { u.row_f32(r - mi, o) };
                let want = host_gemm(&wg, &x_host[..m * h], m, 2 * mi, h);
                let (r1, _) = rel_err(&gy[si * m * 2 * mi..(si + 1) * m * 2 * mi], &want);
                let wd = |r: usize, o: &mut [f32]| d.row_f32(r, o);
                let want_d = host_gemm(&wd, &din_m[si * m * mi..(si + 1) * m * mi], m, h, mi);
                let (r2, _) = rel_err(&gd[si * m * h..(si + 1) * m * h], &want_d);
                worst = worst.max(r1).max(r2);
            }
            match &row0 { None => row0 = Some(gy[..2 * mi].to_vec()), Some(r0) => bitdiff += r0.iter().zip(&gy[..2 * mi]).filter(|(a, b)| a != b).count() }
        }
        let ok = worst < 2e-3 && bitdiff == 0;
        if !ok { fails += 1; }
        println!("FP4L_GPU_MOE {} layer 0, {} expert slots, gate|up + down: rel-L2 vs host <= {worst:.2e}; slot 0 row 0 bit-diffs across m: {bitdiff}",
                 if ok { "OK" } else { "FAIL" }, picks.len());
        // timing: m = 1, 10 slots (a decode token's experts)
        let mut y = dev.alloc_zeros::<u16>(picks.len() * 2 * mi)?;
        dev.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..200 {
            xflaunch!(l, f, (((2 * mi) / 64) as u32, picks.len() as u32, 1), (128, 1, 1), 0,
                      (&fx.gu_w, &fx.gu_s, &fx.gs, &x, 0i64, &mut y, &idxmap, &esel, 1i32, h as i32, (2 * mi) as i32, mi as i32))?;
        }
        dev.synchronize()?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / 200.0;
        let bytes = (picks.len() * 2 * mi * (h / 2 + h / 16)) as f64;
        println!("FP4L_GPU_TIME xf_nv4_moe gate|up m=1 x 10 experts: {us:.1} us = {:.0} GB/s", bytes / us / 1e3);

        // ---- prefill tiles: 3 experts with 5, 37, 130 compact rows; every MT
        let counts = [5usize, 37, 130];
        let pe: [usize; 3] = [3, 400, 511];
        let rows: usize = counts.iter().sum();
        let row_tok: Vec<i32> = (0..rows).map(|i| ((i * 7) % 16) as i32).collect();
        let xs = lcg_f16(&mut seed, 16 * h, 1.0);
        let xdev = dev.htod_sync_copy(&xs)?;
        let rt = dev.htod_sync_copy(&row_tok)?;
        let mut offs = vec![0i32; 512];
        let mut cnt = vec![0i32; 512];
        let mut acc = 0i32;
        for (j, &e) in pe.iter().enumerate() { offs[e] = acc; cnt[e] = counts[j] as i32; acc += counts[j] as i32; }
        let (offs_d, cnt_d) = (dev.htod_sync_copy(&offs)?, dev.htod_sync_copy(&cnt)?);
        let ph: Vec<(crate::fp4l::pack::Nv4Host, crate::fp4l::pack::Nv4Host)> = pe.iter().map(|&e| -> Result<_> {
            let p = format!("model.language_model.layers.0.mlp.experts.{e}");
            Ok((fp.nvfp4(&format!("{p}.gate_proj"))?, fp.nvfp4(&format!("{p}.up_proj"))?))
        }).collect::<Result<_>>()?;
        for mt in [1usize, 2, 4, 8] {
            let mut tl: Vec<i32> = Vec::new();
            for (j, &e) in pe.iter().enumerate() { for sup in 0..counts[j].div_ceil(16 * mt) { tl.push(e as i32); tl.push(sup as i32); } }
            let ntl = (tl.len() / 2) as i32;
            let (tld, ntd) = (dev.htod_sync_copy(&tl)?, dev.htod_sync_copy(&[ntl])?);
            let mut y = dev.alloc_zeros::<u16>(rows * 2 * mi)?;
            let (fname, _) = nv4_tiles_fn(mt);
            xflaunch!(l, fname, (((2 * mi) / 128) as u32, ntl as u32, 1), (256, 1, 1), 0,
                      (&fx.gu_w, &fx.gu_s, &fx.gs, &xdev, &rt, &mut y, &tld, &ntd, &offs_d, &cnt_d,
                       (h | (2 * mi) << 16) as i32, (mi | (mt << 16)) as i32))?;
            dev.synchronize()?;
            let gy = dev.dtoh_sync_copy(&y)?;
            let mut worst = 0f64;
            for (j, &e) in pe.iter().enumerate() {
                let (g, u) = &ph[j];
                let base = offs[e] as usize;
                let xin: Vec<u16> = (0..counts[j]).flat_map(|i| { let t = row_tok[base + i] as usize; xs[t * h..(t + 1) * h].to_vec() }).collect();
                let wg = |r: usize, o: &mut [f32]| if r < mi { g.row_f32(r, o) } else { u.row_f32(r - mi, o) };
                let want = host_gemm(&wg, &xin, counts[j], 2 * mi, h);
                let (r1, _) = rel_err(&gy[base * 2 * mi..(base + counts[j]) * 2 * mi], &want);
                worst = worst.max(r1);
            }
            // mt 4 / 8: the served tiled kernel must equal xf_nv4_tiles_m4 bit for bit
            let tc_diff = if let Some(tf) = nv4_tc_fn(mt) {
                let mut y2 = dev.alloc_zeros::<u16>(rows * 2 * mi)?;
                xflaunch!(l, tf, (((2 * mi) / 128) as u32, ntl as u32, 1), (256, 1, 1), 0,
                          (&fx.gu_w, &fx.gu_s, &fx.gs, &xdev, &rt, &mut y2, &tld, &ntd, &offs_d, &cnt_d,
                           (h | (2 * mi) << 16) as i32, (mi | (mt << 16)) as i32))?;
                dev.synchronize()?;
                Some(count_diff(&gy, &dev.dtoh_sync_copy(&y2)?))
            } else { None };
            let ok = worst < 2e-3 && tc_diff.unwrap_or(0) == 0;
            if !ok { fails += 1; }
            println!("FP4L_GPU_TILES {} {fname} (list mt {mt}): rel-L2 vs host <= {worst:.2e} over {rows} compact rows{}", if ok { "OK" } else { "FAIL" },
                     tc_diff.map_or(String::new(), |d| format!("; xf_nv4_tc vs it bit-diffs {d}")));
        }
    }

    // ---- --fp4l-mtp-sidecar: every dense sidecar linear through the decode path (m = 1, 4, 16) and the prefill arm (m = 48)
    //      vs the host chain on the effective weight; row 0 bitwise across m (batch invariance of the decode path)
    if let Some(sp) = mtp_sidecar_path(dir) {
        let sc = mtp_sidecar_open(&sp)?;
        let stems: Vec<String> = sc.hdr.keys().filter_map(|k| k.strip_suffix(".trellis")).filter(|s| !s.contains(".experts.")).map(str::to_string).collect();
        let ws = dev.alloc_zeros::<f32>(2_000_000)?;
        let rws = dev.alloc_zeros::<u16>(1)?;
        let ltws = dev.alloc_zeros::<u8>(32 << 20)?;
        let rc = dev.alloc_zeros::<u8>(256)?;
        let rb = ReconBufs { ws: *rws.device_ptr() as u64, ws_len: 0, rc: *rc.device_ptr() as u64,
                             ltws: *ltws.device_ptr() as u64, ltws_len: ltws.len() };
        for stem in &stems {
            let Some(q) = mtp_sidecar_quad(&dev, &sc, stem)? else { continue };
            let (k, n) = (q.k as usize, q.n as usize);
            let Some(FxW::Trellis { weff, bits, ks, .. }) = q.fx.as_deref() else { unreachable!() };
            let wh = dev.dtoh_sync_copy(weff)?; // [n][k] f16
            let wrow = |r: usize, o: &mut [f32]| for (c, v) in o[..k].iter_mut().enumerate() { *v = half::f16::from_bits(wh[r * k + c]).to_f32(); };
            let x_host = lcg_f16(&mut seed, 48 * k, 1.0);
            let want = host_gemm(&wrow, &x_host, 48, n, k);
            let x = dev.htod_sync_copy(&x_host)?;
            let (mut worst, mut bitdiff) = (0f64, 0usize);
            let mut row0: Option<Vec<u16>> = None;
            for m in [1usize, 4, 16, 48] {
                let mut y = dev.alloc_zeros::<u16>(m * n)?;
                fx_chain(&l, &q, q.fx.as_deref().unwrap(), &x, &mut y, m, &ws, Some(&rb))?;
                dev.synchronize()?;
                let got = dev.dtoh_sync_copy(&y)?;
                worst = worst.max(rel_err(&got, &want[..m * n]).0);
                if m <= 16 {
                    match &row0 { None => row0 = Some(got[..n].to_vec()), Some(r0) => bitdiff += count_diff(r0, &got[..n]) }
                }
            }
            let ok = worst < 3e-3 && bitdiff == 0;
            if !ok { fails += 1; }
            println!("FP4L_GPU_SIDECAR {} {stem} K={bits} [{k}x{n}] ks {ks}: rel-L2 vs host (effective weight) <= {worst:.2e} at m = 1, 4, 16 \
                      (trellis decode path) and 48 (cuBLASLt arm); row 0 bit-diffs across m = 1..16: {bitdiff}", if ok { "OK" } else { "FAIL" });
        }
    }

    // ---- dequantizers
    {
        let stem = "model.language_model.layers.0.linear_attn.in_proj_z";
        let m = fp.mxfp8_rows(stem, 0, 256)?;
        let (fc, fs) = mx8_frag(&m.codes, &m.scales, m.n, m.k);
        let (w, sc) = (dev.htod_sync_copy(&fc)?, dev.htod_sync_copy(&fs)?);
        let mut out = dev.alloc_zeros::<u16>(m.n * m.k)?;
        let ne = (m.n * m.k) as i64;
        xflaunch!(l, "xf_mx8_dequant", (((ne / 8) as u32).div_ceil(256), 1, 1), (256, 1, 1), 0, (&w, &sc, &mut out, ne, m.k as i32))?;
        dev.synchronize()?;
        let got = dev.dtoh_sync_copy(&out)?;
        let mut row = vec![0f32; m.k];
        let mut bad = 0usize;
        for r in 0..m.n { m.row_f32(r, &mut row); for c in 0..m.k { if got[r * m.k + c] != half::f16::from_f32(row[c]).to_bits() { bad += 1; } } }
        if bad != 0 { fails += 1; }
        println!("FP4L_GPU_DEQ {} xf_mx8_dequant {stem} rows 0..256: {bad} f16 bit mismatches vs host RN", if bad == 0 { "OK" } else { "FAIL" });
    }
    println!("FP4L_GPU {} ({fails} failing checks)", if fails == 0 { "PASS" } else { "FAIL" });
    if fails > 0 { bail!("{fails} FP4L GPU checks failed"); }
    Ok(())
}
