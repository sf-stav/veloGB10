//! The bf16-staged mirror of the DFlash2 oracle — the K-DF2-1 diff reference.
//!
//! The oracle (`oracle.rs`) is pure f32 and is the DEFINITION. The device kernels accumulate in
//! fp32 but store bf16 at every stage boundary (GEMM out, norm out, RoPE out, conv out, attention
//! out, residual add out, silu out). Diffing a bf16 device chain directly against a pure-f32
//! oracle would measure the ~1.1e-3 RMS bf16 *quantization* noise, not the kernel arithmetic
//! (the P-B trap: "mirror the device's bf16 output store in any reference or you measure
//! quantization noise instead of arithmetic").
//!
//! So this module re-runs the oracle's dataflow with `round_bf16` applied at exactly the device's
//! stage boundaries. The remaining device↔mirror residual is then fp32 accumulation-order noise
//! only (tree/strided reduce vs ascending sum), which is ~1e-5-class — comfortably under the
//! 1e-3 per-piece / 5e-3 per-layer gates.
//!
//! # Parity discipline
//!
//! Every primitive here is a LINE-EXACT copy of the oracle's f32 arithmetic (same iteration
//! orders, same formulas — `linear`, `rms_norm_rows`, `rms_norm_heads`, `rope_apply`, `convolve`,
//! `silu`, the RoPE inv_freq/tables). The ONLY additions are the `round_bf16` boundaries. The
//! norm weights are applied as PLAIN `w·x` here (the device uploads `w−1` and its `(1+w')`
//! kernels reconstruct `w` to <1 ulp — far below the bf16 store the kernel applies).

use half::bf16;

use crate::dflash2::oracle::{ConvWeights, Dflash2Config, LayerWeights};
use crate::dflash2::synth::SyntheticTables;

#[inline]
pub fn rb_clone(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| bf16::from_f32(v).to_f32()).collect()
}

/// CPU courtesy: the mirror's parallelism cap (default 8 of 20 cores; env-overridable,
/// diagnostics-only knob — the mirror is probe-only code).
pub fn mirror_threads() -> usize {
    crate::opts::var(crate::opt!("df2-mirror-threads")).ok()
        .and_then(|v| v.parse::<usize>().ok()).filter(|&n| (1..=16).contains(&n)).unwrap_or(8)
}

/// Round ONE f32 through bf16 (the device's f2b-then-b2f at a store boundary).
#[inline]
pub fn rb(v: f32) -> f32 {
    bf16::from_f32(v).to_f32()
}

// ---- primitives (line-exact oracle copies) ----------------------------------

pub fn inv_freq(cfg: &Dflash2Config) -> Vec<f32> {
    let mut inv = vec![0.0f32; cfg.head_dim / 2];
    for i in 0..cfg.head_dim / 2 {
        let e = (2 * i) as f64 / cfg.head_dim as f64;
        inv[i] = (1.0f64 / (cfg.rope_theta as f64).powf(e)) as f32;
    }
    inv
}

/// cos/sin tables `[max_pos, head_dim/2]` — the ORACLE's layout (what the mirror's `rope_apply`
/// reads, `cos[p*half + j]`).
pub fn rope_tables_half(cfg: &Dflash2Config, inv: &[f32], max_pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = cfg.head_dim / 2;
    let mut cos = vec![0.0f32; max_pos * half];
    let mut sin = vec![0.0f32; max_pos * half];
    for p in 0..max_pos {
        let pf = p as f32;
        for i in 0..half {
            let ang = pf * inv[i];
            cos[p * half + i] = ang.cos();
            sin[p * half + i] = ang.sin();
        }
    }
    (cos, sin)
}

/// cos/sin tables `[max_pos, rdim]` (rdim = head_dim), the duplicated-freqs convention the
/// DEVICE's gather_rope_b/rope_b expect (first `head_dim/2` entries = the oracle's cos/sin;
/// second half duplicated — unused by rope_b, which reads `pair < half`).
pub fn rope_tables(cfg: &Dflash2Config, inv: &[f32], max_pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = cfg.head_dim / 2;
    let rdim = cfg.head_dim;
    let mut cos = vec![0.0f32; max_pos * rdim];
    let mut sin = vec![0.0f32; max_pos * rdim];
    for p in 0..max_pos {
        let pf = p as f32;
        for i in 0..half {
            let ang = pf * inv[i];
            let c = ang.cos();
            let s = ang.sin();
            cos[p * rdim + i] = c;
            sin[p * rdim + i] = s;
            cos[p * rdim + i + half] = c;
            sin[p * rdim + i + half] = s;
        }
    }
    (cos, sin)
}

pub fn linear(w: &[f32], x: &[f32], outn: usize, inn: usize, rows: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), outn * inn);
    debug_assert_eq!(x.len(), rows * inn);
    let mut out = vec![0.0f32; rows * outn];
    // Parallelize over rows (each output's ascending-i reduction order is unchanged, so the
    // result is bit-identical to the single-threaded oracle; the fc/k/v ctx GEMMs are the hot spot).
    if rows >= 4 && inn * outn >= 1_000_000 {
        std::thread::scope(|scope| {
            let nthreads = crate::dflash2::mirror::mirror_threads();
        let chunk = (rows + nthreads - 1) / nthreads;
            for (ci, or_chunk) in out.chunks_mut(chunk * outn).enumerate() {
                let w = w;
                let x = x;
                scope.spawn(move || {
                    let base = ci * chunk;
                    for rr in 0..or_chunk.len() / outn {
                        let r = base + rr;
                        let xr = &x[r * inn..(r + 1) * inn];
                        let or = &mut or_chunk[rr * outn..(rr + 1) * outn];
                        linear_row(w, xr, or, outn, inn);
                    }
                });
            }
        });
    } else {
        for r in 0..rows {
            let xr = &x[r * inn..(r + 1) * inn];
            linear_row(w, xr, &mut out[r * outn..(r + 1) * outn], outn, inn);
        }
    }
    out
}

/// The block-GEMM mirror: the EXACT fp32 accumulation ORDER the device's `gemm_dsp_b` uses —
/// 256 "threads" strided-sum k in blocks of 8 (bf16 inputs make every product exact in f32, so
/// `+= w*x` here is bit-identical to the device's `fmaf`), then the device's shuffle-DOWN tree
/// (32-lane warp tree, then an 8-warp cross tree). This isolates REAL wiring bugs from fp32
/// reassociation noise.
pub fn linear_gemm_dsp(w: &[f32], x: &[f32], outn: usize, inn: usize, rows: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), outn * inn);
    debug_assert_eq!(x.len(), rows * inn);
    debug_assert_eq!(inn % 8, 0, "gemm_dsp_b requires inn % 8 == 0");
    let nvec = inn / 8;
    let mut out = vec![0.0f32; rows * outn];
    for r in 0..rows {
        let xr = &x[r * inn..(r + 1) * inn];
        let or = &mut out[r * outn..(r + 1) * outn];
        for o in 0..outn {
            let wr = &w[o * inn..(o + 1) * inn];
            let mut partials = [0.0f32; 256];
            for t in 0..256 {
                let mut acc = 0.0f32;
                let mut i = t;
                while i < nvec {
                    let base = i << 3;
                    for j in 0..8 {
                        acc = wr[base + j].mul_add(xr[base + j], acc);
                    }
                    i += 256;
                }
                partials[t] = acc;
            }
            // shuffle-down tree: 8 warps of 32 lanes, then an 8-lane cross-warp tree.
            let mut warp = [0.0f32; 8];
            for w in 0..8 {
                let mut p = [0.0f32; 32];
                for l in 0..32 {
                    p[l] = partials[w * 32 + l];
                }
                for off in [16, 8, 4, 2, 1] {
                    for l in 0..(32 - off) {
                        p[l] += p[l + off];
                    }
                }
                warp[w] = p[0];
            }
            for off in [4, 2, 1] {
                for l in 0..(8 - off) {
                    warp[l] += warp[l + off];
                }
            }
            or[o] = warp[0];
        }
    }
    out
}

#[inline]
fn linear_row(w: &[f32], xr: &[f32], or: &mut [f32], outn: usize, inn: usize) {
    let mut o = 0usize;
    while o + 8 <= outn {
        let mut acc = [0.0f32; 8];
        for i in 0..inn {
            let xv = xr[i];
            for u in 0..8 {
                acc[u] += w[(o + u) * inn + i] * xv;
            }
        }
        for u in 0..8 {
            or[o + u] = acc[u];
        }
        o += 8;
    }
    while o < outn {
        let wr = &w[o * inn..(o + 1) * inn];
        let mut acc = 0.0f32;
        for i in 0..inn {
            acc += wr[i] * xr[i];
        }
        or[o] = acc;
        o += 1;
    }
}

/// The device's rmsnorm sum-of-squares order: `block` threads strided-sum `v^2` (ascending within
/// each thread's stride), then a halving tree over the `block` partials (gpu_batch.cu rmsnorm_b /
/// rmsnorm_perhead_b). Matching this order in the mirror removes the per-column scaling
/// reassociation that otherwise propagates through every downstream piece.
fn sum_sq_tree(v: &[f32], n: usize, block: usize) -> f32 {
    // the device's `sum_sq += v * v` compiles to fma.rn.f32 (fused) — mirror it exactly
    let mut s = vec![0.0f32; block];
    for t in 0..block {
        let mut acc = 0.0f32;
        let mut i = t;
        while i < n {
            acc = v[i].mul_add(v[i], acc);
            i += block;
        }
        s[t] = acc;
    }
    let mut s2 = block / 2;
    while s2 > 0 {
        for t in 0..s2 {
            s[t] += s[t + s2];
        }
        s2 >>= 1;
    }
    s[0]
}

/// The exact `inv` the mirror's rms_norm_rows computes for row 0 (probe candidate scan).
pub fn rms_norm_rows_inv_exact(x: &[f32], w: &[f32], n: usize, eps: f32) -> f32 {
    let sum_sq = sum_sq_tree(x, n, n.min(1024));
    1.0f32 / (sum_sq / n as f32 + eps).sqrt()
}

pub fn rms_norm_rows(x: &[f32], w: &[f32], rows: usize, n: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        let xr = &x[r * n..(r + 1) * n];
        let sum_sq = sum_sq_tree(xr, n, n.min(1024));
        let inv = 1.0f32 / (sum_sq / n as f32 + eps).sqrt();
        let or = &mut out[r * n..(r + 1) * n];
        for (i, &v) in xr.iter().enumerate() {
            or[i] = v * inv * w[i];
        }
    }
    out
}

pub fn rms_norm_heads(x: &mut [f32], rows: usize, heads: usize, w: &[f32], hd: usize, eps: f32) {
    for r in 0..rows {
        for h in 0..heads {
            let base = (r * heads + h) * hd;
            let sum_sq = sum_sq_tree(&x[base..base + hd], hd, hd);
            let inv = 1.0f32 / (sum_sq / hd as f32 + eps).sqrt();
            for d in 0..hd {
                x[base + d] *= inv * w[d];
            }
        }
    }
}

/// RoPE apply in the DEVICE's exact arithmetical shape (`rope_b`): the product operands are
/// bf16-rounded BEFORE the f32 multiply-add (bf16 inputs, f32 math, bf16 store). The old shape
/// read f32 operands from the table convention and missed the pre-rounding — a <=1-ulp-per-input
/// divergence that the k gate sees exactly at large |angle| positions.
pub fn rope_apply(
    x: &mut [f32],
    rows: usize,
    heads: usize,
    positions: &[usize],
    cos: &[f32],
    sin: &[f32],
    hd: usize,
) {
    let half = hd / 2;
    for r in 0..rows {
        let p = positions[r];
        for h in 0..heads {
            let base = (r * heads + h) * hd;
            for j in 0..half {
                let c = cos[p * half + j];
                let s = sin[p * half + j];
                let re = x[base + j];
                let im = x[base + half + j];
                let re_b = rb(re);
                let im_b = rb(im);
                x[base + j] = rb(re_b * c - im_b * s);
                x[base + half + j] = rb(im_b * c + re_b * s);
            }
        }
    }
}

pub fn convolve(x: &[f32], dyn_taps: &[f32], base: &[f32], n: usize, hidden: usize, k: usize, groups: usize, gs: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * hidden];
    for r in 0..n {
        for o in 0..k {
            if r < o {
                continue;
            }
            let src = &x[(r - o) * hidden..(r - o + 1) * hidden];
            let drow = &dyn_taps[(r * k + o) * groups..(r * k + o + 1) * groups];
            let brow = &base[o * hidden..(o + 1) * hidden];
            let orow = &mut out[r * hidden..(r + 1) * hidden];
            for g in 0..groups {
                let d = drow[g];
                for c in g * gs..(g + 1) * gs {
                    orow[c] += (brow[c] + d) * src[c];
                }
            }
        }
    }
    out
}

pub fn silu(x: f32) -> f32 {
    if x >= 0.0 {
        x / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        x * e / (1.0 + e)
    }
}

// ---- the bf16-staged dataflow (device stage boundaries only) -----------------

/// `conv_prepare` mirror: kernel_projection GEMM (bf16 out) → split dyn0/dyn1 → convolve side 0
/// (bf16 out). Returns (x_conv, dyn_hold) both bf16-staged, plus the raw dyn_all for the diff.
pub fn conv_prepare_mirror(cfg: &Dflash2Config, conv: &ConvWeights, x: &[f32], n: usize)
    -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hidden = cfg.hidden;
    let k = cfg.conv_kernel;
    let groups = hidden / cfg.conv_group;
    let gs = cfg.conv_group;
    let dyn_all = rb_clone(&linear_gemm_dsp(&conv.kernel_projection, x, 2 * k * groups, hidden, n));
    let mut dyn0 = vec![0.0f32; n * k * groups];
    let mut dyn1 = vec![0.0f32; n * k * groups];
    for r in 0..n {
        for o in 0..k {
            for g in 0..groups {
                dyn0[(r * k + o) * groups + g] = dyn_all[(r * 2 * k + o) * groups + g];
                dyn1[(r * k + o) * groups + g] = dyn_all[(r * 2 * k + k + o) * groups + g];
            }
        }
    }
    let base0 = &conv.base_kernel[0..k * hidden];
    let x_conv = rb_clone(&convolve(x, &dyn0, base0, n, hidden, k, groups, gs));
    (x_conv, dyn1, dyn_all)
}

pub fn conv_finish_mirror(cfg: &Dflash2Config, conv: &ConvWeights, y: &[f32], dyn_hold: &[f32], n: usize) -> Vec<f32> {
    let hidden = cfg.hidden;
    let k = cfg.conv_kernel;
    let groups = hidden / cfg.conv_group;
    let gs = cfg.conv_group;
    let base1 = &conv.base_kernel[k * hidden..2 * k * hidden];
    rb_clone(&convolve(y, dyn_hold, base1, n, hidden, k, groups, gs))
}

/// The attention piece (pre-o_proj) mirror, bf16-staged at every device boundary. Returns
/// (attn bf16, o bf16, q bf16, kb bf16, vb bf16).
#[allow(clippy::too_many_arguments)]
pub fn attn_mirror(
    cfg: &Dflash2Config,
    l: &LayerWeights,
    x_conv: &[f32],
    ctx_k: &[f32],
    ctx_v: &[f32],
    block_pos: &[usize],
    cos: &[f32],
    sin: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let hidden = cfg.hidden;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let block = cfg.block;
    let ctx_len = ctx_k.len() / (nkv * hd);
    let ntot = ctx_len + block;

    // The device rounds q/k to bf16 after EVERY stage (q_proj store, then the per-head-norm store,
    // then the RoPE store). Mirror that exactly: norm in f32, round to bf16, then RoPE.
    let mut q = rb_clone(&linear_gemm_dsp(&l.q_proj, x_conv, nh * hd, hidden, block));
    rms_norm_heads(&mut q, block, nh, &l.q_norm, hd, cfg.rms_eps);
    let mut q = rb_clone(&q);
    rope_apply(&mut q, block, nh, block_pos, cos, sin, hd);
    let q = rb_clone(&q);

    let mut kb = rb_clone(&linear_gemm_dsp(&l.k_proj, x_conv, nkv * hd, hidden, block));
    rms_norm_heads(&mut kb, block, nkv, &l.k_norm, hd, cfg.rms_eps);
    let mut kb = rb_clone(&kb);
    rope_apply(&mut kb, block, nkv, block_pos, cos, sin, hd);
    let kb = rb_clone(&kb);
    let vb = rb_clone(&linear_gemm_dsp(&l.v_proj, x_conv, nkv * hd, hidden, block));

    let mut k = Vec::with_capacity(ntot * nkv * hd);
    let mut v = Vec::with_capacity(ntot * nkv * hd);
    k.extend_from_slice(ctx_k);
    k.extend_from_slice(&kb);
    v.extend_from_slice(ctx_v);
    v.extend_from_slice(&vb);

    let scale = 1.0f32 / (hd as f32).sqrt();
    let group = nh / nkv;
    let mut attn = vec![0.0f32; block * nh * hd];
    let mut scores = vec![0.0f32; ntot];
    for r in 0..block {
        let qp = ctx_len + r;
        for h in 0..nh {
            let kvh = h / group;
            let qrow = &q[(r * nh + h) * hd..(r * nh + h + 1) * hd];
            let mut m = f32::NEG_INFINITY;
            for j in 0..ntot {
                if !visible(cfg, qp, j) {
                    scores[j] = f32::NEG_INFINITY;
                    continue;
                }
                let krow = &k[(j * nkv + kvh) * hd..(j * nkv + kvh + 1) * hd];
                let mut s = 0.0f32;
                for d in 0..hd {
                    s += qrow[d] * krow[d];
                }
                s *= scale;
                scores[j] = s;
                if s > m {
                    m = s;
                }
            }
            // device-matching softmax: the gqa_attn_band_b pass-2 sum is 128 "threads" strided-sum
            // over [lo, ntot), then an XOR shuffle tree (4 warps of 32, then a 4-lane cross tree).
            // Multiply-by-reciprocal + fused-multiply-add PV match the device's pass 3.
            let lo = (qp as i64 - (cfg.sliding_window as i64 - 1)).max(0) as usize;
            for j in 0..ntot {
                let e = if scores[j] == f32::NEG_INFINITY { 0.0 } else { (scores[j] - m).exp() };
                scores[j] = e;
            }
            let mut partial = [0.0f32; 128];
            for d in 0..128 {
                let mut acc = 0.0f32;
                let mut j = lo + d;
                while j < ntot {
                    acc += scores[j];
                    j += 128;
                }
                partial[d] = acc;
            }
            let mut warp = [0.0f32; 4];
            for w in 0..4 {
                let mut p = [0.0f32; 32];
                for l in 0..32 {
                    p[l] = partial[w * 32 + l];
                }
                for off in [16, 8, 4, 2, 1] {
                    let old = p;
                    for l in 0..32 {
                        p[l] = old[l] + old[l ^ off];
                    }
                }
                warp[w] = p[0];
            }
            for off in [2, 1] {
                let old = warp;
                for l in 0..4 {
                    warp[l] = old[l] + old[l ^ off];
                }
            }
            let sum = warp[0];
            let inv = 1.0f32 / sum;
            let o = &mut attn[(r * nh + h) * hd..(r * nh + h + 1) * hd];
            for d in 0..hd {
                o[d] = 0.0;
            }
            for j in 0..ntot {
                if scores[j] == 0.0 {
                    continue;
                }
                let w = scores[j] * inv;
                let vrow = &v[(j * nkv + kvh) * hd..(j * nkv + kvh + 1) * hd];
                for d in 0..hd {
                    o[d] = w.mul_add(vrow[d], o[d]);
                }
            }
        }
    }
    let attn_bf16 = rb_clone(&attn);
    // The device stores attention to bf16 BEFORE the o_proj GEMM; mirror that stage boundary.
    let o = rb_clone(&linear_gemm_dsp(&l.o_proj, &attn_bf16, hidden, nh * hd, block));
    (attn_bf16, o, q, kb, vb)
}

fn visible(cfg: &Dflash2Config, q_pos: usize, k_pos: usize) -> bool {
    let mut v = true;
    if cfg.is_causal {
        v &= k_pos <= q_pos;
    }
    let w = cfg.sliding_window;
    v &= (q_pos as i64 - k_pos as i64) < w as i64;
    if !cfg.is_causal {
        v &= (k_pos as i64 - q_pos as i64) < w as i64;
    }
    v
}

/// The SwiGLU MLP piece (pre-finish-conv) mirror, bf16-staged. Returns (gate, up, ffn, down).
pub fn mlp_mirror(cfg: &Dflash2Config, l: &LayerWeights, x_conv: &[f32])
    -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let hidden = cfg.hidden;
    let inter = cfg.inter;
    let block = cfg.block;
    let gate = rb_clone(&linear_gemm_dsp(&l.gate_proj, x_conv, inter, hidden, block));
    let up = rb_clone(&linear_gemm_dsp(&l.up_proj, x_conv, inter, hidden, block));
    let mut ffn = vec![0.0f32; block * inter];
    for i in 0..block * inter {
        ffn[i] = silu(gate[i]) * up[i];
    }
    let ffn = rb_clone(&ffn);
    let down = rb_clone(&linear_gemm_dsp(&l.down_proj, &ffn, hidden, inter, block));
    (gate, up, ffn, down)
}

/// `tap_project` mirror: th_raw = fc(taps), th = hidden_norm(th_raw). Returns (th_raw, th).
pub fn tap_project_mirror(cfg: &Dflash2Config, fc: &[f32], hidden_norm: &[f32], taps: &[f32], m: usize)
    -> (Vec<f32>, Vec<f32>) {
    let hidden = cfg.hidden;
    let tap_dim = cfg.n_taps * hidden;
    let th_raw = rb_clone(&linear(fc, taps, hidden, tap_dim, m));
    let th = rb_clone(&rms_norm_rows(&th_raw, hidden_norm, m, hidden, cfg.rms_eps));
    (th_raw, th)
}

/// `draft_kv_write` mirror for ONE layer: (k_norm + RoPE on k, raw v). Returns (k_ctx, v_ctx).
pub fn draft_kv_mirror(cfg: &Dflash2Config, l: &LayerWeights, th: &[f32], m: usize, pos_start: usize,
    cos: &[f32], sin: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let hidden = cfg.hidden;
    let nkv = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let positions: Vec<usize> = (pos_start..pos_start + m).collect();
    let mut k = rb_clone(&linear(&l.k_proj, th, nkv * hd, hidden, m));
    rms_norm_heads(&mut k, m, nkv, &l.k_norm, hd, cfg.rms_eps);
    let mut k = rb_clone(&k);
    rope_apply(&mut k, m, nkv, &positions, cos, sin, hd);
    let k = rb_clone(&k);
    let v = rb_clone(&linear(&l.v_proj, th, nkv * hd, hidden, m));
    (k, v)
}

/// One layer's bf16-staged forward, producing every piece the device emits (for the per-piece
/// diff) plus the post-layer hidden.
#[allow(clippy::too_many_arguments)]
pub fn mirror_layer_forward(
    cfg: &Dflash2Config,
    l: &LayerWeights,
    h: &[f32],
    ctx_k: &[f32],
    ctx_v: &[f32],
    block_pos: &[usize],
    cos: &[f32],
    sin: &[f32],
) -> MirrorLayerOut {
    let hidden = cfg.hidden;
    let block = cfg.block;

    let input_ln_out = rb_clone(&rms_norm_rows(h, &l.input_ln, block, hidden, cfg.rms_eps));
    let (x_conv, dyn_hold, dyn_all) = conv_prepare_mirror(cfg, &l.attention_conv, &input_ln_out, block);
    let (attn, o, q, k, v) = attn_mirror(cfg, l, &x_conv, ctx_k, ctx_v, block_pos, cos, sin);
    let fin = conv_finish_mirror(cfg, &l.attention_conv, &o, &dyn_hold, block);

    let mut h2 = vec![0.0f32; block * hidden];
    for i in 0..block * hidden {
        h2[i] = h[i] + fin[i];
    }
    let h2 = rb_clone(&h2);

    let post_ln_out = rb_clone(&rms_norm_rows(&h2, &l.post_ln, block, hidden, cfg.rms_eps));
    let (x_conv2, dyn_hold2, dyn_all2) = conv_prepare_mirror(cfg, &l.mlp_conv, &post_ln_out, block);
    let (gate, up, ffn, down) = mlp_mirror(cfg, l, &x_conv2);
    let fin2 = conv_finish_mirror(cfg, &l.mlp_conv, &down, &dyn_hold2, block);

    let mut h3 = vec![0.0f32; block * hidden];
    for i in 0..block * hidden {
        h3[i] = h2[i] + fin2[i];
    }
    let h3 = rb_clone(&h3);

    MirrorLayerOut {
        input_ln_out,
        x_conv,
        dyn_attn: dyn_all,
        q,
        k,
        v,
        attn,
        o,
        fin,
        h2,
        post_ln_out,
        x_conv2,
        dyn_mlp: dyn_all2,
        gate,
        up,
        ffn,
        down,
        fin2,
        h3,
    }
}

/// All the pieces one device layer forward emits, plus the post-layer hidden.
#[derive(Clone)]
pub struct MirrorLayerOut {
    pub input_ln_out: Vec<f32>,
    pub x_conv: Vec<f32>,
    pub dyn_attn: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn: Vec<f32>,
    pub o: Vec<f32>,
    pub fin: Vec<f32>,
    pub h2: Vec<f32>,
    pub post_ln_out: Vec<f32>,
    pub x_conv2: Vec<f32>,
    pub dyn_mlp: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub ffn: Vec<f32>,
    pub down: Vec<f32>,
    pub fin2: Vec<f32>,
    pub h3: Vec<f32>,
}

/// The block input embeddings (anchor + 7× MASK), bf16-staged (DECISION O).
pub fn block_emb_mirror(cfg: &Dflash2Config, synth: &SyntheticTables, anchor: u32) -> Vec<f32> {
    let hidden = cfg.hidden;
    let scale = 1.0f32 / (hidden as f32).sqrt();
    let mut emb = Vec::with_capacity(cfg.block * hidden);
    emb.extend_from_slice(&synth.row(SyntheticTables::TABLE_EMBED, anchor, hidden, scale));
    for _ in 1..cfg.block {
        emb.extend_from_slice(&synth.row(SyntheticTables::TABLE_EMBED, cfg.mask_token_id, hidden, scale));
    }
    rb_clone(&emb)
}

// ===========================================================================================
// S4F — the ROUND mirror (K-DF2-2/3): incremental injection (gemm_dsp order), the borrowed
// NVFP4 head/embed (mma dequant semantics), top-16, and the fused selector walk. Every new
// piece mirrors the DEVICE's exact rounding + reduction order; the oracle stays untouched.
// ===========================================================================================

/// Row-parallel `linear_gemm_dsp` (rows are independent — per-element order unchanged, so the
/// result is bit-identical to the serial one; the fc/head mirrors are the hot spot).
pub fn linear_gemm_dsp_par(w: &[f32], x: &[f32], outn: usize, inn: usize, rows: usize) -> Vec<f32> {
    if rows < 4 || inn * outn < 1_000_000 {
        return linear_gemm_dsp(w, x, outn, inn, rows);
    }
    let mut out = vec![0.0f32; rows * outn];
    // Disjoint mutable row-chunks of `out`, precomputed with split_at_mut so the spawned
    // closures each own a unique slice (no double borrow).
    let mut chunks_out: Vec<&mut [f32]> = Vec::new();
    {
        let nthreads = crate::dflash2::mirror::mirror_threads();
        let chunk = (rows + nthreads - 1) / nthreads;
        let mut rest: &mut [f32] = out.as_mut_slice();
        let mut r0 = 0usize;
        while r0 < rows {
            let r1 = (r0 + chunk).min(rows);
            let n = (r1 - r0) * outn;
            let (head, tail) = rest.split_at_mut(n);
            chunks_out.push(head);
            rest = tail;
            r0 = r1;
        }
    }
    // `chunks_out` is consumed by value (into an iterator of owned &mut slices).
    std::thread::scope(|scope| {
        let mut spawns = Vec::new();
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        {
            let nthreads = crate::dflash2::mirror::mirror_threads();
        let chunk = (rows + nthreads - 1) / nthreads;
            let mut r0 = 0usize;
            while r0 < rows {
                let r1 = (r0 + chunk).min(rows);
                ranges.push((r0, r1));
                r0 = r1;
            }
        }
        for (os, &(r0, r1)) in chunks_out.into_iter().zip(&ranges) {
            let (w, xs) = (w, &x[r0 * inn..r1 * inn]);
            spawns.push(scope.spawn(move || {
                let sub = linear_gemm_dsp(w, xs, outn, inn, r1 - r0);
                os.copy_from_slice(&sub);
            }));
        }
        for s in spawns { let _ = s.join(); }
    });
    out
}

/// The injection fc + hidden_norm mirror in the DEVICE's gemm_dsp order (workdoc §3.2 — the
/// production M<=8 path; S3F's `tap_project_mirror` mirrors the probe's gemm_tiled path).
/// `taps` = the chunk's tap columns, row-major `[m, 25600]` (one row per committed position).
pub fn round_tap_project_dsp(cfg: &Dflash2Config, fc: &[f32], hidden_norm: &[f32], taps: &[f32], m: usize)
    -> (Vec<f32>, Vec<f32>) {
    let hidden = cfg.hidden;
    let tap_dim = cfg.n_taps * hidden;
    let th_raw = rb_clone(&linear_gemm_dsp_par(fc, taps, hidden, tap_dim, m));
    let th = rb_clone(&rms_norm_rows(&th_raw, hidden_norm, m, hidden, cfg.rms_eps));
    (th_raw, th)
}

/// The injection chunk's k/v mirror (gemm_dsp order): k = RoPE(k_norm(k_proj(th))) at ABSOLUTE
/// positions `[pos_start, pos_start+m)`, v raw. Returns the chunk rows.
pub fn round_draft_kv_dsp(cfg: &Dflash2Config, l: &LayerWeights, th: &[f32], m: usize, pos_start: usize,
    cos: &[f32], sin: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let hidden = cfg.hidden;
    let nkv = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let positions: Vec<usize> = (pos_start..pos_start + m).collect();
    let mut k = rb_clone(&linear_gemm_dsp_par(&l.k_proj, th, nkv * hd, hidden, m));
    rms_norm_heads(&mut k, m, nkv, &l.k_norm, hd, cfg.rms_eps);
    let mut k = rb_clone(&k);
    rope_apply(&mut k, m, nkv, &positions, cos, sin, hd);
    let k = rb_clone(&k);
    let v = rb_clone(&linear_gemm_dsp_par(&l.v_proj, th, nkv * hd, hidden, m));
    (k, v)
}

// ---- the borrowed NVFP4 head/embed (mma semantics) ---------------------------
//
// The device's `gemm_mma_fp4_b` dequant: the A-fragment is bf16(e2m1(code) * e4m3(block_scale))
// — the global scale is applied at the EPILOGUE in f32 (o = v * gs[mt] = v * 1/global_scale).
// The trunk's `embed_gather_fp4_tiled_b` instead folds the global scale into the element BEFORE
// the single bf16 round: bf16(e2m1 * e4m3 * (1/global_scale)). Both are mirrored exactly.

/// Debug-exposed decodes (probe printouts).
pub fn dbg_e2m1(code: u8) -> f32 { e2m1(code) }
pub fn dbg_e4m3(b: u8) -> f32 { e4m3(b) }

/// E2M1 decode (the device's `e2m1_f` — magnitudes {0,.5,1,1.5,2,3,4,6}, sign bit 3).
fn e2m1(code: u8) -> f32 {
    const T: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = T[(code & 0x7) as usize];
    if code & 0x8 != 0 { -v } else { v }
}

/// E4M3 -> f32 by bit surgery (the device's `e4m3_f`).
fn e4m3(b: u8) -> f32 {
    let sign = ((b & 0x80) as u32) << 24;
    let e = ((b >> 3) & 0x0F) as i32;
    let m = ((b & 0x07) as u32) << 20;
    if e == 0 {
        let v = (b & 0x07) as f32 * 0.001953125;      // 2^-9 subnormals
        return if b & 0x80 != 0 { -v } else { v };
    }
    f32::from_bits(sign | (((e - 7 + 127) as u32) << 23) | m)
}

/// One element of the ORIGINAL safetensors layout: packed `[m, k/2]` u8 (low nibble = even
/// column), scales `[m, k/16]` e4m3.
fn nvfp4_elem(packed: &[u8], scales: &[u8], k: usize, row: usize, col: usize) -> f32 {
    let byte = packed[row * (k / 2) + col / 2];
    let code = if col % 2 == 0 { byte & 0x0F } else { byte >> 4 };
    e2m1(code) * e4m3(scales[row * (k / 16) + col / 16])
}

/// The HEAD's dequantized weight row (mma semantics): bf16(e2m1·e4m3) — NO global scale.
pub fn head_row_mma(packed: &[u8], scales: &[u8], k: usize, row: usize, out: &mut [bf16]) {
    for (c, o) in out.iter_mut().enumerate() {
        *o = bf16::from_f32(nvfp4_elem(packed, scales, k, row, c));
    }
}

/// The EMBED's dequantized row (gather semantics): bf16(e2m1·e4m3·(1/global_scale)).
pub fn embed_row_mma(packed: &[u8], scales: &[u8], k: usize, row: usize, global_scale: f32, out: &mut [bf16]) {
    let inv = 1.0f32 / global_scale;
    for (c, o) in out.iter_mut().enumerate() {
        *o = bf16::from_f32(nvfp4_elem(packed, scales, k, row, c) * inv);
    }
}

/// The borrowed-head logits mirror: logits[n][m] = bf16((Σ_k w[m,k]·x[n,k] in gemm_dsp order)
/// × (1/global_scale)) — the epilogue scale in f32, then the single bf16 store. `x` = h_sel
/// rows (`[rows, 5120]` bf16-as-f32). NOTE: the device's mma internal accumulation order is
/// hardware-defined and NOT mirrored — the probe gates this surface at rel-L2 (~1e-6 expected),
/// and the EXACT selector gates below feed BOTH paths the SAME bf16 logits.
pub fn head_logits_mirror(w_flat: &[bf16], x: &[f32], rows: usize, hidden: usize,
                          global_scale: f32, vocab: usize) -> Vec<f32> {
    let inv = 1.0f32 / global_scale;
    let mut out = vec![0.0f32; rows * vocab];
    // gemm_dsp order per output element, streaming the bf16 weight rows (no f32 row copy).
    let nvec = hidden / 8;
    let mut on_idx = 0usize;
    for n in 0..rows {
        let xn = &x[n * hidden..(n + 1) * hidden];
        for o in 0..vocab {
            let wrow = &w_flat[o * hidden..(o + 1) * hidden];
            let mut partials = [0.0f32; 256];
            for t in 0..256 {
                let mut acc = 0.0f32;
                let mut i = t;
                while i < nvec {
                    let base = i << 3;
                    for j in 0..8 {
                        acc += wrow[base + j].to_f32() * xn[base + j];
                    }
                    i += 256;
                }
                partials[t] = acc;
            }
            let mut warp = [0.0f32; 8];
            for w in 0..8 {
                let mut p = [0.0f32; 32];
                for l in 0..32 { p[l] = partials[w * 32 + l]; }
                for off in [16, 8, 4, 2, 1] {
                    for l in 0..(32 - off) { p[l] += p[l + off]; }
                }
                warp[w] = p[0];
            }
            for off in [4, 2, 1] {
                for l in 0..(8 - off) { warp[l] += warp[l + off]; }
            }
            out[on_idx] = bf16::from_f32(warp[0] * inv).to_f32();
            on_idx += 1;
        }
    }
    out
}

/// The selector walk mirror — the DEVICE's exact order (`df2_sel_walk_b`): per (p, k) the
/// 256 partials (one product per lane) reduce via the 32-lane shuffle-down tree, then a SERIAL
/// ASCENDING 8-warp sum, then s = unary + w. Identical bf16 inputs → bitwise-equal scores.
/// `hp` = [7][256] (bf16 values as f32), `cand`/`unary` = [7][16], codebooks row-major bf16.
pub fn round_walk_mirror(hp: &[f32], cand: &[u32], unary: &[f32], anchor: u32,
                         pred_cb: &[bf16], succ_cb: &[bf16], rank: usize, nlevels: usize)
    -> (Vec<u32>, Vec<f32>) {
    // DF2 block-16: `nlevels` is the live chain length (block-1: 7 or 15). The device walk's
    // trip count is the same parameter, so the mirror must be told it rather than assume 7.
    let mut tokens = vec![0u32; nlevels];
    let mut scores = vec![0.0f32; nlevels * 16];
    let mut prev = anchor as usize;
    for p in 0..nlevels {
        let prow = &pred_cb[prev * rank..(prev + 1) * rank];
        let hpr = &hp[p * rank..(p + 1) * rank];
        let a: Vec<f32> = (0..rank).map(|r| prow[r].to_f32() * hpr[r]).collect();
        for k in 0..16 {
            let tok = cand[p * 16 + k] as usize;
            let srow = &succ_cb[tok * rank..(tok + 1) * rank];
            let partials: Vec<f32> = (0..rank).map(|r| a[r] * srow[r].to_f32()).collect();
            let mut warp = [0.0f32; 8];
            for w in 0..8 {
                let mut pp = [0.0f32; 32];
                for l in 0..32 { pp[l] = partials[w * 32 + l]; }
                for off in [16, 8, 4, 2, 1] {
                    for l in 0..(32 - off) { pp[l] += pp[l + off]; }
                }
                warp[w] = pp[0];
            }
            let mut w_total = 0.0f32;
            for w in 0..8 { w_total += warp[w]; }        // serial ascending (the kernel's tid-0 sum)
            scores[p * 16 + k] = unary[p * 16 + k] + w_total;
        }
        let mut best = scores[p * 16];
        let mut bi = 0usize;
        for k in 1..16 {
            if scores[p * 16 + k] > best { best = scores[p * 16 + k]; bi = k; }
        }
        tokens[p] = cand[p * 16 + bi];
        prev = tokens[p] as usize;
    }
    (tokens, scores)
}
