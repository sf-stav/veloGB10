//! Per-op math of the CPU reference: RoPE, Hadamard rotation, sparse attention, RMSNorm, hyper-connection
//! (Sinkhorn) math, deterministic top-k, router / expert forward, weight-take helpers and index-list helpers.
//! Siblings: `numerics.rs` (primitives), `compressor.rs` / `attn_moe.rs` (compositions); `mod.rs` has the map.

use super::*;

// ---------------------------------------------------------------------------
// RoPE (model.py:205-250; §B.1.3)
// ---------------------------------------------------------------------------

/// Complex rotation table, fp32, [positions, rd/2] — adjacent-pair convention.
pub struct RopeTable {
    pub cos: Vec<f32>, // [positions, rd/2]
    pub sin: Vec<f32>,
    pub rd: usize,
}

/// `precompute_freqs_cis` (model.py:205-235) exactly: plain θ table, or the
/// YaRN correction when `original_seq_len > 0` (compress layers only; SWA layers
/// force-disable YaRN, §B.1.3). Correction-range math is python f64; the ramp
/// and the frequency blend are torch f32 op order.
pub fn rope_table(
    rd: usize,
    positions: usize,
    original_seq_len: usize,
    base: f32,
    factor: f32,
    beta_fast: u32,
    beta_slow: u32,
) -> RopeTable {
    let half = rd / 2;
    let dim = rd as f64;
    let mut freqs = vec![0.0f32; half];
    for i in 0..half {
        let e = (2 * i) as f32 / rd as f32;
        freqs[i] = 1.0f32 / base.powf(e);
    }
    if original_seq_len > 0 {
        // find_correction_dim in python f64 (math.log), then floor/ceil.
        let find = |num_rot: f64| -> f64 {
            dim * ((original_seq_len as f64) / (num_rot * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * (base as f64).ln())
        };
        let low = find(beta_fast as f64).floor().max(0.0) as i32;
        let high = find(beta_slow as f64).ceil().min(dim - 1.0) as i32;
        // linear_ramp_factor(low, high, half): clamp((i-low)/(high-low), 0, 1), f32
        let (minv, maxv) = if low == high { (low as f32, low as f32 + 0.001) } else { (low as f32, high as f32) };
        for (i, f) in freqs.iter_mut().enumerate() {
            let ramp = ((i as f32 - minv) / (maxv - minv)).clamp(0.0, 1.0);
            let smooth = 1.0f32 - ramp;
            *f = (*f / factor) * (1.0f32 - smooth) + (*f) * smooth;
        }
    }
    let mut cos = vec![0.0f32; positions * half];
    let mut sin = vec![0.0f32; positions * half];
    for t in 0..positions {
        for (i, &f) in freqs.iter().enumerate() {
            let ang = (t as f32) * f; // torch.polar(1, ang)
            cos[t * half + i] = ang.cos();
            sin[t * half + i] = ang.sin();
        }
    }
    RopeTable { cos, sin, rd }
}

/// `apply_rotary_emb` on the last `rd` dims of each row, in place (bf16 round
/// on write-back). `row_pos[i]` is the absolute position of row i. Complex mul
/// in f32: (re+im·i)(c+s·i) = (re·c − im·s) + (re·s + im·c)i.
pub fn apply_rope(x: &mut [f32], rows: usize, dim: usize, table: &RopeTable, row_pos: &[usize], inverse: bool) {
    let rd = table.rd;
    let half = rd / 2;
    assert_eq!(x.len(), rows * dim);
    for (i, &p) in row_pos.iter().enumerate().take(rows) {
        let off = dim - rd;
        let row = &mut x[i * dim..(i + 1) * dim];
        for j in 0..half {
            let re = row[off + 2 * j];
            let im = row[off + 2 * j + 1];
            let c = table.cos[p * half + j];
            let s = if inverse { -table.sin[p * half + j] } else { table.sin[p * half + j] };
            row[off + 2 * j] = bf(re * c - im * s);
            row[off + 2 * j + 1] = bf(re * s + im * c);
        }
    }
}

// ---------------------------------------------------------------------------
// Hadamard rotation (oracle emu: exact fp32 Walsh-Hadamard matmul, scale d^-0.5)
// ---------------------------------------------------------------------------

/// Sylvester ±1 Hadamard matrix × d^-0.5, f32, row-major [d,d]. d must be a
/// power of two. Mirrors dsv4_ref.py's `hadamard_matrix(d) * (d ** -0.5)`.
pub fn hadamard_scaled(d: usize) -> Vec<f32> {
    assert!(d.is_power_of_two());
    let scale = (d as f64).powf(-0.5) as f32;
    let mut h = vec![1.0f32];
    let mut n = 1;
    while n < d {
        let mut nh = vec![0.0f32; 4 * n * n];
        for r in 0..n {
            for c in 0..n {
                let v = h[r * n + c];
                nh[r * 2 * n + c] = v;
                nh[r * 2 * n + c + n] = v;
                nh[(r + n) * 2 * n + c] = v;
                nh[(r + n) * 2 * n + c + n] = -v;
            }
        }
        h = nh;
        n *= 2;
    }
    for v in h.iter_mut() {
        *v *= scale;
    }
    h
}

/// `rotate_activation`: y = bf16(x_f32 @ H·d^-0.5) on the last d dims of each row.
pub fn rotate_activation(x: &mut [f32], rows: usize, dim: usize, d: usize, h_scaled: &[f32]) {
    assert_eq!(x.len(), rows * dim);
    assert_eq!(h_scaled.len(), d * d);
    let off = dim - d;
    for r in 0..rows {
        let row = &mut x[r * dim..(r + 1) * dim];
        let src: Vec<f32> = row[off..].to_vec();
        for c in 0..d {
            let mut acc = 0.0f32;
            for j in 0..d {
                acc += src[j] * h_scaled[j * d + c];
            }
            row[off + c] = bf(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// sparse_attn (§B.7 — the oracle's pure-torch emulation of kernel.py:276-368)
// ---------------------------------------------------------------------------

/// fp32 scores with a single global max per (row, head); probabilities rounded
/// to bf16 for the P·V numerator only; fp32 denominator; sink added to the
/// DENOMINATOR only; −1 index = masked (zero KV row, −inf score).
/// q [m,h,d] / kv [n,d] bf16-valued f32; idxs [m,t] (i64, −1 masked).
/// Returns o [m,h,d] bf16-rounded.
pub fn sparse_attn(
    q: &[f32],
    m: usize,
    h: usize,
    d: usize,
    kv: &[f32],
    n: usize,
    sink: &[f32],
    idxs: &[i64],
    t: usize,
    scale: f32,
) -> Vec<f32> {
    assert_eq!(q.len(), m * h * d);
    assert_eq!(kv.len(), n * d);
    assert_eq!(sink.len(), h);
    assert_eq!(idxs.len(), m * t);
    let mut out = vec![0.0f32; m * h * d];
    par_chunks(&mut out, h * d, |mi, orow| {
        let ids = &idxs[mi * t..(mi + 1) * t];
        for hh in 0..h {
            let qrow = &q[(mi * h + hh) * d..(mi * h + hh + 1) * d];
            // scores (masked = -inf), global row max
            let mut scores = vec![f32::NEG_INFINITY; t];
            let mut row_max = f32::NEG_INFINITY;
            for (tt, &ix) in ids.iter().enumerate() {
                if ix < 0 {
                    continue;
                }
                let kvrow = &kv[ix as usize * d..(ix as usize + 1) * d];
                let s = dot8(qrow, kvrow) * scale;
                scores[tt] = s;
                if s > row_max {
                    row_max = s;
                }
            }
            // p = exp(s - max); denominator fp32 incl. denominator-only sink
            let mut denom = 0.0f32;
            let mut p = vec![0.0f32; t];
            for tt in 0..t {
                if scores[tt].is_finite() {
                    let e = (scores[tt] - row_max).exp();
                    p[tt] = e;
                    denom += e;
                }
            }
            denom += (sink[hh] - row_max).exp();
            // numerator uses bf16-rounded probabilities (acc_s_cast), fp32 accum
            let orow = &mut orow[hh * d..(hh + 1) * d];
            for (tt, &ix) in ids.iter().enumerate() {
                if ix < 0 {
                    continue;
                }
                let pbf = bf(p[tt]);
                let kvrow = &kv[ix as usize * d..(ix as usize + 1) * d];
                for dd in 0..d {
                    orow[dd] += pbf * kvrow[dd];
                }
            }
            for dd in 0..d {
                orow[dd] = bf(orow[dd] / denom);
            }
        }
    });
    out
}

// ---------------------------------------------------------------------------
// RMSNorm (model.py:189-202): fp32 math, bf16 result, eps = norm_eps (1e-6)
// ---------------------------------------------------------------------------

pub fn rms_norm(x: &[f32], rows: usize, dim: usize, weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), rows * dim);
    assert_eq!(weight.len(), dim);
    let mut out = vec![0.0f32; rows * dim];
    par_chunks(&mut out, dim, |r, orow| {
        let row = &x[r * dim..(r + 1) * dim];
        let ss = sumsq_tree(row);
        let var = ss / dim as f32;
        let inv = (var + eps).sqrt().recip();
        for d in 0..dim {
            orow[d] = bf(weight[d] * (row[d] * inv));
        }
    });
    out
}

/// In-place RMSNorm variant over one row slice (used where rows share scratch).
pub fn rms_norm_row(x: &mut [f32], weight: &[f32], eps: f32) {
    let ss = sumsq_tree(x);
    let inv = (ss / x.len() as f32 + eps).sqrt().recip();
    for d in 0..x.len() {
        x[d] = bf(weight[d] * (x[d] * inv));
    }
}

// ---------------------------------------------------------------------------
// mHC (§B.8; kernel.py:371-438 exact iteration order)
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x).exp())
}

/// `hc_split_sinkhorn` for one token's 24 mixes (hc=4). Returns (pre[4], post[4], comb[16]).
/// Exact sequence, all fp32, eps = hc_eps (1e-6) in every denominator:
/// pre = sigmoid+eps; post = 2·sigmoid; comb: row-softmax+eps → col-norm →
/// 19×(row-norm, col-norm).
pub fn hc_split_sinkhorn(mixes: &[f32], scale: &[f32; 3], base: &[f32], hc: usize, iters: usize, eps: f32) -> ([f32; 4], [f32; 4], [f32; 16]) {
    assert_eq!(mixes.len(), (2 + hc) * hc);
    assert_eq!(base.len(), (2 + hc) * hc);
    let mut pre = [0.0f32; 4];
    let mut post = [0.0f32; 4];
    let mut comb = [0.0f32; 16];
    for j in 0..hc {
        pre[j] = sigmoid(mixes[j] * scale[0] + base[j]) + eps;
    }
    for j in 0..hc {
        post[j] = 2.0 * sigmoid(mixes[hc + j] * scale[1] + base[hc + j]);
    }
    for j in 0..hc {
        for k in 0..hc {
            comb[j * hc + k] = mixes[hc * 2 + j * hc + k] * scale[2] + base[hc * 2 + j * hc + k];
        }
    }
    // comb = softmax_rows(comb) + eps  (max-subtracted exp / row sum, then + eps)
    for j in 0..hc {
        let mut mx = f32::NEG_INFINITY;
        for k in 0..hc {
            mx = mx.max(comb[j * hc + k]);
        }
        let mut rs = 0.0f32;
        for k in 0..hc {
            let e = (comb[j * hc + k] - mx).exp();
            comb[j * hc + k] = e;
            rs += e;
        }
        for k in 0..hc {
            comb[j * hc + k] = comb[j * hc + k] / rs + eps;
        }
    }
    // comb /= (col_sum + eps)
    let mut col_norm = |comb: &mut [f32; 16]| {
        for k in 0..hc {
            let mut cs = 0.0f32;
            for j in 0..hc {
                cs += comb[j * hc + k];
            }
            for j in 0..hc {
                comb[j * hc + k] /= cs + eps;
            }
        }
    };
    col_norm(&mut comb);
    for _ in 0..iters - 1 {
        // row-norm
        for j in 0..hc {
            let mut rs = 0.0f32;
            for k in 0..hc {
                rs += comb[j * hc + k];
            }
            for k in 0..hc {
                comb[j * hc + k] /= rs + eps;
            }
        }
        col_norm(&mut comb);
    }
    (pre, post, comb)
}

/// mHC mixing parameters for one sublayer (all fp32).
#[derive(Debug, Clone)]
pub struct HcParams {
    pub hc_fn: Vec<f32>,   // [24, hc*dim]
    pub hc_base: Vec<f32>, // [24]
    pub hc_scale: [f32; 3],
}

/// `hc_pre` for one token: xf [hc*dim] (bf16-valued f32 streams flattened) →
/// (y [dim] bf16-rounded, post [4], comb [16]).
pub fn hc_pre_token(xf: &[f32], hc: usize, dim: usize, p: &HcParams, norm_eps: f32, iters: usize, hc_eps: f32) -> (Vec<f32>, [f32; 4], [f32; 16]) {
    let hcd = hc * dim;
    assert_eq!(xf.len(), hcd);
    let ss = sumsq_tree(xf);
    let rsqrt = (ss / hcd as f32 + norm_eps).sqrt().recip();
    let nmix = (2 + hc) * hc;
    let mut mixes = vec![0.0f32; nmix];
    for m in 0..nmix {
        mixes[m] = dot_tree(&p.hc_fn[m * hcd..(m + 1) * hcd], xf) * rsqrt;
    }
    let (pre, post, comb) = hc_split_sinkhorn(&mixes, &p.hc_scale, &p.hc_base, hc, iters, hc_eps);
    // y = sum_h pre[h] * x[h] (fp32), then bf16
    let mut y = vec![0.0f32; dim];
    for h in 0..hc {
        for d in 0..dim {
            y[d] += pre[h] * xf[h * dim + d];
        }
    }
    round_bf16(&mut y);
    (y, post, comb)
}

/// `hc_post` for one token: out[k,d] = post[k]·x[d] + Σ_j comb[j,k]·res[j,d],
/// result bf16-rounded (streams update).
pub fn hc_post_token(x: &[f32], residual: &[f32], post: &[f32; 4], comb: &[f32; 16], hc: usize, dim: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), dim);
    debug_assert_eq!(residual.len(), hc * dim);
    debug_assert_eq!(out.len(), hc * dim);
    for k in 0..hc {
        for d in 0..dim {
            let mut acc = 0.0f32;
            for j in 0..hc {
                acc += comb[j * hc + k] * residual[j * dim + d];
            }
            out[k * dim + d] = bf(post[k] * x[d] + acc);
        }
    }
}

/// `hc_head` (final collapse, sigmoid-only — no post/comb/Sinkhorn; model.py:709-716).
/// xf [hc*dim] flattened streams → y [dim] bf16-rounded.
pub fn hc_head_token(xf: &[f32], hc: usize, dim: usize, hc_fn: &[f32], hc_base: &[f32], hc_scale: f32, norm_eps: f32, hc_eps: f32) -> Vec<f32> {
    let hcd = hc * dim;
    assert_eq!(xf.len(), hcd);
    assert_eq!(hc_fn.len(), hc * hcd);
    let ss = sumsq_tree(xf);
    let rsqrt = (ss / hcd as f32 + norm_eps).sqrt().recip();
    let mut pre = [0.0f32; 4];
    for h in 0..hc {
        let mixes = dot_tree(&hc_fn[h * hcd..(h + 1) * hcd], xf) * rsqrt;
        pre[h] = sigmoid(mixes * hc_scale + hc_base[h]) + hc_eps;
    }
    let mut y = vec![0.0f32; dim];
    for h in 0..hc {
        for d in 0..dim {
            y[d] += pre[h] * xf[h * dim + d];
        }
    }
    round_bf16(&mut y);
    y
}

// ---------------------------------------------------------------------------
// Deterministic top-k (§12.B.2: value desc, index asc; -inf entries sort last)
// ---------------------------------------------------------------------------

/// Indices of the k largest values, sorted value-desc with index-asc tie-break.
/// This is the CSA batch-invariance contract — selection is a pure function of
/// the scores, independent of batch width and platform (torch.topk tie order is
/// implementation-defined; near-ties in identical inputs still agree exactly).
pub fn topk_deterministic(scores: &[f32], k: usize) -> Vec<i64> {
    let k = k.min(scores.len());
    let mut idx: Vec<i64> = (0..scores.len() as i64).collect();
    // Full sort is O(n log n) with n ≤ 262144 for the indexer — fine on CPU.
    idx.sort_by(|&a, &b| {
        let (sa, sb) = (scores[a as usize], scores[b as usize]);
        // desc by value; NaN/-inf naturally last via partial_cmp reversal; ties by index asc
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

// ---------------------------------------------------------------------------
// Router + experts (§B.9)
// ---------------------------------------------------------------------------

/// PyTorch softplus (threshold 20): log1p(exp(s)) for s ≤ 20 else s.
#[inline]
pub fn softplus_torch(s: f32) -> f32 {
    if s > 20.0 {
        s
    } else {
        s.exp().ln_1p()
    }
}

/// `Gate.forward` for a batch of tokens. x [t, dim] (bf16-valued f32, floated
/// exactly like `linear(x.float(), weight.float())`). Returns (weights [t,6]
/// fp32, indices [t,6]).
#[allow(clippy::too_many_arguments)]
pub fn gate_forward(
    x: &[f32],
    t: usize,
    dim: usize,
    gate_w: &[f32],          // [n_exp, dim] fp32
    bias: Option<&[f32]>,    // [n_exp] fp32, selection-only (layers ≥ 3)
    tid2eid: Option<(&[i32], &[i64])>, // (table [vocab,6] int32, token ids [t]) — hash layers 0–2
    n_exp: usize,
    topk: usize,
    route_scale: f32,
) -> (Vec<f32>, Vec<i64>) {
    assert_eq!(x.len(), t * dim);
    assert_eq!(gate_w.len(), n_exp * dim);
    let scores = gemm_f32(x, t, dim, gate_w, n_exp); // fp32 GEMM always
    let mut slots: Vec<(Vec<f32>, Vec<i64>)> = (0..t).map(|_| (vec![0.0; topk], vec![0; topk])).collect();
    par_rows(&mut slots, |i, slot| {
        let (wrow, irow) = slot;
        let mut s: Vec<f32> = scores[i * n_exp..(i + 1) * n_exp].to_vec();
        for v in s.iter_mut() {
            *v = softplus_torch(*v).sqrt();
        }
        let original = s.clone(); // weights are gathered from UN-biased scores
        let sel = match bias {
            Some(b) => {
                let mut bsel = s;
                for (e, v) in bsel.iter_mut().enumerate() {
                    *v += b[e];
                }
                bsel
            }
            None => s,
        };
        let idx: Vec<i64> = match tid2eid {
            Some((table, ids)) => {
                let id = ids[i] as usize;
                (0..topk).map(|j| table[id * topk + j] as i64).collect()
            }
            None => topk_deterministic(&sel, topk),
        };
        let mut wsum = 0.0f32;
        for (j, &e) in idx.iter().enumerate() {
            let w = original[e as usize];
            wrow[j] = w;
            wsum += w;
        }
        for j in 0..topk {
            wrow[j] = wrow[j] / wsum * route_scale;
        }
        irow.copy_from_slice(&idx);
    });
    let mut weights = vec![0.0f32; t * topk];
    let mut indices = vec![0i64; t * topk];
    for (i, (w, ix)) in slots.into_iter().enumerate() {
        weights[i * topk..(i + 1) * topk].copy_from_slice(&w);
        indices[i * topk..(i + 1) * topk].copy_from_slice(&ix);
    }
    (weights, indices)
}

/// One expert's forward on one token (§B.9 Expert): fp32 inner math,
/// asymmetric clamps (up ±limit, gate ≤ +limit), silu(gate)·up, optional
/// routing weight, w2 on the bf16-rounded intermediate. Weight matrices are the
/// exact f32 dequant; `inner_block` is 32 for FP4 routed experts / 128 for FP8
/// shared expert. Returns [dim] bf16-rounded.
pub fn expert_forward_token(
    x: &[f32], // [dim] bf16-valued f32
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    dim: usize,
    inter: usize,
    inner_block: usize,
    swiglu_limit: f32,
    routing_weight: Option<f32>,
) -> Vec<f32> {
    let mut gate = quant_gemm(x, 1, dim, w1, inter, inner_block);
    let mut up = quant_gemm(x, 1, dim, w3, inter, inner_block);
    if swiglu_limit > 0.0 {
        for v in up.iter_mut() {
            *v = v.clamp(-swiglu_limit, swiglu_limit);
        }
        for v in gate.iter_mut() {
            *v = v.min(swiglu_limit);
        }
    }
    let mut h = vec![0.0f32; inter];
    for i in 0..inter {
        let g = gate[i];
        let mut val = (g * sigmoid(g)) * up[i]; // silu(gate) * up, fp32
        if let Some(w) = routing_weight {
            val *= w;
        }
        h[i] = bf(val); // x.to(dtype) before w2
    }
    quant_gemm(&h, 1, inter, w2, dim, inner_block)
}

// ---------------------------------------------------------------------------
// Weight extraction from Lane A's Dsv4Layer (replay path; unit tests build the
// Cpu* structs directly with synthetic tensors)
// ---------------------------------------------------------------------------

pub(crate) fn take_f32(map: &mut HashMap<String, HostTensor>, key: &str, numel: usize) -> Result<Vec<f32>> {
    match map.remove(key) {
        Some(HostTensor::F32 { data, .. }) => {
            if data.len() != numel {
                bail!("{key}: expected {} elements, got {}", numel, data.len());
            }
            Ok(data)
        }
        Some(other) => Err(anyhow!("{key}: expected F32, got {:?}", other.shape())),
        None => Err(anyhow!("{key}: missing")),
    }
}

pub(crate) fn take_bf16_as_f32(map: &mut HashMap<String, HostTensor>, key: &str, numel: usize) -> Result<Vec<f32>> {
    match map.remove(key) {
        Some(HostTensor::BF16 { data, .. }) => {
            if data.len() != numel {
                bail!("{key}: expected {} elements, got {}", numel, data.len());
            }
            Ok(data.iter().map(|v| v.to_f32()).collect())
        }
        Some(HostTensor::F32 { data, .. }) => {
            // tolerate already-upcast (cast rules upcast several bf16 tensors)
            if data.len() != numel {
                bail!("{key}: expected {} elements, got {}", numel, data.len());
            }
            Ok(data)
        }
        Some(other) => Err(anyhow!("{key}: expected BF16, got {:?}", other.shape())),
        None => Err(anyhow!("{key}: missing")),
    }
}

pub(crate) fn take_i32(map: &mut HashMap<String, HostTensor>, key: &str, numel: usize) -> Result<Vec<i32>> {
    match map.remove(key) {
        Some(HostTensor::I32 { data, .. }) => {
            if data.len() != numel {
                bail!("{key}: expected {} elements, got {}", numel, data.len());
            }
            Ok(data)
        }
        Some(other) => Err(anyhow!("{key}: expected I32, got {:?}", other.shape())),
        None => Err(anyhow!("{key}: missing")),
    }
}

pub(crate) fn opt_f32(map: &mut HashMap<String, HostTensor>, key: &str, numel: usize) -> Result<Option<Vec<f32>>> {
    if !map.contains_key(key) {
        return Ok(None);
    }
    take_f32(map, key, numel).map(Some)
}

// ---------------------------------------------------------------------------
// Index-list helpers (model.py:260-282, 743-747 — pure functions, bit-exact)
// ---------------------------------------------------------------------------

/// `get_window_topk_idxs`: SWA ring index lists.
/// prefill (start_pos == 0): row i = [max(0,i−127)..i], −1-padded to min(s,128).
/// decode start_pos ≥ 127: all 128 physical slots, oldest→newest
/// (cat([sp+1..128), [0..sp])), sp = start_pos % 128. Early decode: [0..sp], −1 pad.
pub fn window_topk_idxs(window: usize, s: usize, start_pos: usize) -> Vec<Vec<i64>> {
    if start_pos >= window - 1 {
        let sp = start_pos % window;
        let mut row = Vec::with_capacity(window);
        row.extend((sp + 1..window).map(|v| v as i64));
        row.extend((0..=sp).map(|v| v as i64));
        vec![row; s]
    } else if start_pos > 0 {
        let mut row: Vec<i64> = (0..=start_pos as i64).collect();
        row.resize(window, -1);
        vec![row; s]
    } else {
        let tw = s.min(window);
        (0..s)
            .map(|i| {
                let base = i.saturating_sub(window - 1);
                (0..tw)
                    .map(|j| {
                        let v = base + j;
                        if v > i {
                            -1
                        } else {
                            v as i64
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

/// `get_compress_topk_idxs` (HCA): every completed 128-token block, no top-k.
/// prefill row i: blocks 0..(i+1)//ratio − 1 (−1-masked), + offset.
/// decode: arange(0, (start_pos+1)//ratio) + offset.
pub fn compress_topk_idxs(ratio: usize, s: usize, start_pos: usize, offset: usize) -> Vec<Vec<i64>> {
    if start_pos > 0 {
        vec![(0..(start_pos + 1) / ratio).map(|v| (v + offset) as i64).collect(); s]
    } else {
        let nb = s / ratio;
        (0..s)
            .map(|i| {
                let lim = (i + 1) / ratio;
                (0..nb)
                    .map(|j| if j >= lim { -1 } else { (j + offset) as i64 })
                    .collect()
            })
            .collect()
    }
}

/// `get_dspark_topk_idxs`: 133-entry non-causal block list —
/// cat([arange(min(128, start_pos+1)), 128+arange(block)]), identical for all draft rows.
pub fn dspark_topk_idxs(window: usize, block: usize, start_pos: usize) -> Vec<i64> {
    let mut v: Vec<i64> = (0..window.min(start_pos + 1)).map(|x| x as i64).collect();
    v.extend((0..block).map(|i| (window + i) as i64));
    v
}

