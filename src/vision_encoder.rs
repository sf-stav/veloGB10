//! CPU vision encoder — the validated visual tower forward (reference path).
//!
//! This is the exact math validated by the V2 cross-chain per-block rel-L2 oracle
//! (PLAN/W2_PREPROC_SPEC.md §10, GREEN: patch_embed+pos 3.9e-7, 27 blocks 3e-6..1.6e-5, merger
//! 1.47e-5). It consumes `pixel_values` (flatten [N, 1536], N = gh*gw, in the
//! `vision_preproc::flatten` order) and grid `(gh, gw)`, and produces the merged image
//! embeddings `[N/4, 5120]` to splice at the language embed layer.
//!
//! This is the CPU reference the GPU path must match, and is the compute engine for a first
//! (correctity-first) vision serving path. The GPU port (V2) replaces the heavy matmuls.

use base64::Engine;
use crate::vision_tower::{VisualTower};

/// Vision rotary: inv_freq length = (head_dim//2)/2 (mirrors Qwen3VLVisionRotaryEmbedding).
fn vision_inv_freq(hd: usize) -> Vec<f64> {
    let dim = (hd / 2) as f64;
    (0..hd / 4).map(|i| 10000.0f64.powf(-(2.0 * i as f64) / dim)).collect()
}

fn layernorm(x: &[f32], w: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (ri, row) in x.chunks_exact(n).enumerate() {
        let mean = row.iter().sum::<f32>() / n as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + 1e-6).sqrt();
        for (i, v) in row.iter().enumerate() {
            out[ri * n + i] = (v - mean) * inv * w[i] + b[i];
        }
    }
    out
}

fn gelu_tanh(x: f32) -> f32 {
    const K: f32 = 0.7978845608028654;
    0.5 * x * (1.0 + (K * (x + 0.044715 * x * x * x)).tanh())
}

fn gelu(x: f32) -> f32 {
    // nn.GELU() default erf form (the merger).
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    sign * y
}

// ---- S-B9-REL-VISIONFIX F2: the CPU tower's thread count -----------------------------------
// The CPU tower is the scalar reference: every output element is one left-to-right dot product
// (`acc = b; acc += x[k] * w[k]`), which is what the oracle validated and what must stay
// bit-exact. One 448 x 448 image is 3.6e11 such multiply-adds and the time grows with the image
// (the 53.8 s of the first measurement was a smaller image, not a faster path). The first
// implementation ran ONE dependent add chain at a time (about 3 minutes for 448 x 448 on a GB10
// core: latency-bound, a fraction of the core's throughput); `gemm_rows` now advances 16 tokens'
// chains side by side (see there). Numbers are in PLAN/REL_VISIONFIX_REPORT.md, not here: they go
// stale with every build.
//
// What can change without changing a single bit: WHICH thread computes a token row, and the order
// in which (token, output) pairs are visited. `--vision-cpu-threads N` splits the token rows over N
// threads (default 1). Every dot product is still summed in the same order, so the embeddings are
// identical for every N and equal to the first implementation's (unit-tested).

static CPU_TOWER_THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);

/// `--vision-cpu-threads N` (set once at boot by the servers; 1 = single-threaded, the default).
pub fn set_cpu_tower_threads(n: usize) {
    CPU_TOWER_THREADS.store(n.clamp(1, 64), std::sync::atomic::Ordering::Relaxed);
}
/// The thread count `forward_cpu` uses.
pub fn cpu_tower_threads() -> usize {
    CPU_TOWER_THREADS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Run `f(first_row, rows_out)` over `n` rows of `per_row` outputs each, split into `threads`
/// contiguous row ranges, one per thread (`threads` <= 1, or one row: inline on this thread).
/// Threads spawned here inherit the calling thread's CPU affinity (the server's worker mask).
fn par_rows(n: usize, per_row: usize, out: &mut [f32], threads: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    assert_eq!(out.len(), n * per_row);
    let t = threads.clamp(1, n.max(1));
    if t == 1 || per_row == 0 {
        f(0, out);
        return;
    }
    let chunk = n.div_ceil(t);
    std::thread::scope(|sc| {
        let f = &f;
        for (c, part) in out.chunks_mut(chunk * per_row).enumerate() {
            sc.spawn(move || f(c * chunk, part));
        }
    });
}

/// Rows `first .. first + out.len() / outn` of `[N, inn] @ [inn, outn]^T + bias`, weights
/// row-major `[outn, inn]`. Tokens are taken `TOKENS` at a time: the block is transposed once to
/// `[k][token]`, and every weight row is applied to all of its tokens in one pass over `k`, with one
/// accumulator per token. A dot product is a serial add chain (about 4 cycles per add), so one chain
/// at a time runs at a fraction of the core; the `TOKENS` independent chains advance side by side,
/// and the weight matrix (tens of MB) streams past once per block instead of once per token.
/// BITS: each token's accumulator starts at `b[o]` and adds `x[k] * w[k]` for ascending `k`, one
/// multiply and one add, exactly the order and the operations of the first implementation, so the
/// result is identical to it (unit-tested bit for bit, partial blocks included). Padded lanes of a
/// partial block hold zeros and are never stored.
fn gemm_rows(x: &[f32], wt: &[f32], b: &[f32], out: &mut [f32], first: usize, inn: usize, outn: usize) {
    const TOKENS: usize = 16;
    if outn == 0 { return; } // nothing to produce (the first implementation's loops were no-ops too)
    let rows = out.len() / outn;
    let mut xt = vec![0f32; inn * TOKENS];
    let mut i0 = 0;
    while i0 < rows {
        let nt = (rows - i0).min(TOKENS);
        // [k][token]: lane t of step k is token i0 + t's k-th input (zero past the block's end)
        for t in 0..TOKENS {
            if t < nt {
                let xr = &x[(first + i0 + t) * inn..(first + i0 + t + 1) * inn];
                for k in 0..inn { xt[k * TOKENS + t] = xr[k]; }
            } else {
                for k in 0..inn { xt[k * TOKENS + t] = 0.0; }
            }
        }
        for o in 0..outn {
            let w = &wt[o * inn..(o + 1) * inn];
            let mut acc = [b[o]; TOKENS];
            for (k, &wk) in w.iter().enumerate() {
                let xk: &[f32; TOKENS] = xt[k * TOKENS..(k + 1) * TOKENS].try_into().unwrap();
                for t in 0..TOKENS {
                    acc[t] += xk[t] * wk;
                }
            }
            for t in 0..nt {
                out[(i0 + t) * outn + o] = acc[t];
            }
        }
        i0 += nt;
    }
}

/// Generic GEMM `[N, inn] @ [inn, outn]^T + bias`, weights row-major `[outn, inn]`.
fn gemm_vec(x: &[f32], wt: &[f32], b: &[f32], out: &mut [f32], inn: usize, outn: usize, n: usize) {
    par_rows(n, outn, &mut out[..n * outn], cpu_tower_threads(), |first, part| gemm_rows(x, wt, b, part, first, inn, outn));
}

/// The first implementation of `gemm_vec`, kept as the bit-exactness reference of the tests.
#[cfg(test)]
fn gemm_vec_reference(x: &[f32], wt: &[f32], b: &[f32], out: &mut [f32], inn: usize, outn: usize, n: usize) {
    for i in 0..n {
        let xr = &x[i * inn..(i + 1) * inn];
        for o in 0..outn {
            let w = &wt[o * inn..(o + 1) * inn];
            let mut acc = b[o];
            for k in 0..inn {
                acc += xr[k] * w[k];
            }
            out[i * outn + o] = acc;
        }
    }
}

pub fn pos_embed_bilinear(w: &[f32], gh: usize, gw: usize, num_side: usize, hidden: usize, merge: usize) -> Vec<f32> {
    let n = gh * gw;
    let mut pe = vec![0.0f32; n * hidden];
    let mut hfl = vec![0usize; gh];
    let mut hfr = vec![0.0f32; gh];
    let mut wfl = vec![0usize; gw];
    let mut wfr = vec![0.0f32; gw];
    for i in 0..gh {
        let g = (num_side - 1) as f32 * i as f32 / (gh - 1).max(1) as f32;
        let f = g.floor();
        hfl[i] = f as usize;
        hfr[i] = g - f;
    }
    for i in 0..gw {
        let g = (num_side - 1) as f32 * i as f32 / (gw - 1).max(1) as f32;
        let f = g.floor();
        wfl[i] = f as usize;
        wfr[i] = g - f;
    }
    let hcel = |i: usize| (hfl[i] + 1).min(num_side - 1);
    let wcel = |i: usize| (wfl[i] + 1).min(num_side - 1);
    let mut reorder = Vec::with_capacity(n);
    for hi in 0..gh / merge {
        for wi in 0..gw / merge {
            for mh in 0..merge {
                for mw in 0..merge {
                    reorder.push((hi * merge + mh) * gw + (wi * merge + mw));
                }
            }
        }
    }
    for (tok, &pat) in reorder.iter().enumerate() {
        let (py, px) = (pat / gw, pat % gw);
        let (hy, hf) = (hfl[py], hfr[py]);
        let (wy, wf) = (wfl[px], wfr[px]);
        let (hcy, wcx) = (hcel(py), wcel(px));
        let corners = [
            (hy * num_side + wy, (1.0 - hf) * (1.0 - wf)),
            (hy * num_side + wcx, (1.0 - hf) * wf),
            (hcy * num_side + wy, hf * (1.0 - wf)),
            (hcy * num_side + wcx, hf * wf),
        ];
        for (gi, wt) in corners {
            for d in 0..hidden {
                pe[tok * hidden + d] += w[gi * hidden + d] * wt;
            }
        }
    }
    pe
}

fn vision_position_ids(gh: usize, gw: usize, merge: usize) -> Vec<[usize; 2]> {
    let mut pos = Vec::with_capacity(gh * gw);
    for hi in 0..gh / merge {
        for wi in 0..gw / merge {
            for mh in 0..merge {
                for mw in 0..merge {
                    pos.push([hi * merge + mh, wi * merge + mw]);
                }
            }
        }
    }
    pos
}

pub fn vision_cos_sin(gh: usize, gw: usize, hd: usize, merge: usize) -> (Vec<f32>, Vec<f32>) {
    let n = gh * gw;
    let inv = vision_inv_freq(hd);
    let nu = inv.len();
    let pos = vision_position_ids(gh, gw, merge);
    let mut cos = vec![0.0f32; n * hd];
    let mut sin = vec![0.0f32; n * hd];
    for (tok, [py, px]) in pos.iter().enumerate() {
        // ang = [py*inv | px*inv] -> hd/2
        let mut ang = vec![0.0f64; hd / 2];
        for j in 0..nu {
            ang[j] = *py as f64 * inv[j];
            ang[nu + j] = *px as f64 * inv[j];
        }
        // emb = [ang | ang] -> hd
        for d in 0..hd / 2 {
            let a = ang[d];
            cos[tok * hd + d] = a.cos() as f32;
            sin[tok * hd + d] = a.sin() as f32;
            cos[tok * hd + hd / 2 + d] = a.cos() as f32;
            sin[tok * hd + hd / 2 + d] = a.sin() as f32;
        }
    }
    (cos, sin)
}

fn rotate_half_hd(q: &[f32], hd: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; hd];
    for i in 0..hd / 2 {
        out[i] = -q[hd / 2 + i];
        out[hd / 2 + i] = q[i];
    }
    out
}

impl VisualTower {
    /// CPU forward. `pixel_values`: [N, wpv] f32 flatten; `(gh, gw)` patch grid.
    /// Returns merged embeddings [N/merge², out_hidden].
    pub fn forward_cpu(&self, pixel_values: &[f32], gh: usize, gw: usize) -> Vec<f32> {
        let d = self.dims;
        let (hidden, inter, merge) = (d.hidden, d.inter, d.merge);
        let hd = d.head_dim();
        let wpv = d.wpv();
        let mi = d.merge_inter();
        let n = gh * gw;
        assert_eq!(pixel_values.len(), n * wpv, "pixel_values len");

        // 1. patch_embed [N,wpv] @ [wpv,hidden]^T + b
        let mut h = vec![0.0f32; n * hidden];
        gemm_vec(pixel_values, &self.patch_embed_w, &self.patch_embed_b, &mut h, wpv, hidden, n);
        // 2. pos-embed bilinear
        let pe = pos_embed_bilinear(&self.pos_embed_w, gh, gw, d.num_side(), hidden, merge);
        for i in 0..n {
            for dd in 0..hidden {
                h[i * hidden + dd] += pe[i * hidden + dd];
            }
        }
        // 3. rotary cos/sin
        let (cos, sin) = vision_cos_sin(gh, gw, hd, merge);

        // 4. blocks
        for blk in &self.blocks {
            let norm1 = layernorm(&h, &blk.norm1_w, &blk.norm1_b, hidden);
            let attn_out = block_attn(blk, &norm1, &cos, &sin, n, hd, d.heads, hidden, inter);
            for i in 0..h.len() {
                h[i] += attn_out[i];
            }
            let norm2 = layernorm(&h, &blk.norm2_w, &blk.norm2_b, hidden);
            let mut fc1 = vec![0.0f32; n * inter];
            gemm_vec(&norm2, &blk.fc1_w, &blk.fc1_b, &mut fc1, hidden, inter, n);
            for v in fc1.iter_mut() {
                *v = gelu_tanh(*v);
            }
            let mut fc2 = vec![0.0f32; n * hidden];
            gemm_vec(&fc1, &blk.fc2_w, &blk.fc2_b, &mut fc2, inter, hidden, n);
            for i in 0..h.len() {
                h[i] += fc2[i];
            }
        }

        // 5. merger: layernorm -> view [N,hidden] -> [N/merge², mi] -> fc1 -> gelu -> fc2 -> [N/merge²,out]
        let ln = layernorm(&h, &self.merger_norm_w, &self.merger_norm_b, hidden);
        let tn = n / (merge * merge);
        let mut m = vec![0.0f32; tn * mi];
        for r in 0..tn {
            for g in 0..merge * merge {
                let src = (r * merge * merge + g) * hidden;
                for dd in 0..hidden {
                    m[r * mi + g * hidden + dd] = ln[src + dd];
                }
            }
        }
        let mut fc1 = vec![0.0f32; tn * mi];
        gemm_vec(&m, &self.merger_fc1_w, &self.merger_fc1_b, &mut fc1, mi, mi, tn);
        for v in fc1.iter_mut() {
            *v = gelu(*v);
        }
        let mut out = vec![0.0f32; tn * d.out_hidden];
        gemm_vec(&fc1, &self.merger_fc2_w, &self.merger_fc2_b, &mut out, mi, d.out_hidden, tn);
        out
    }
}

/// Vision self-attention block (norm1 applied). qkv -> reshape [N,3,heads,hd] -> rope -> full
/// attention (N tokens, not causal) -> proj.
fn block_attn(blk: &crate::vision_tower::VisualBlock, x: &[f32], cos: &[f32], sin: &[f32],
              n: usize, hd: usize, heads: usize, hidden: usize, inter: usize) -> Vec<f32> {
    let mut qkv = vec![0.0f32; n * 3 * hidden];
    gemm_vec(x, &blk.qkv_w, &blk.qkv_b, &mut qkv, hidden, 3 * hidden, n);
    let mut q = vec![0.0f32; n * heads * hd];
    let mut k = vec![0.0f32; n * heads * hd];
    let mut v = vec![0.0f32; n * heads * hd];
    for i in 0..n {
        for hh in 0..heads {
            for d in 0..hd {
                let base = i * 3 * heads * hd + hh * hd + d;
                let off = i * heads * hd + hh * hd + d;
                q[off] = qkv[base];
                k[off] = qkv[base + heads * hd];
                v[off] = qkv[base + 2 * heads * hd];
            }
        }
    }
    apply_rope_vision(&mut q, cos, sin, n, heads, hd);
    apply_rope_vision(&mut k, cos, sin, n, heads, hd);
    let scale = (hd as f32).powf(-0.5);
    let mut out = vec![0.0f32; n * heads * hd];
    // each query row is independent (it reads q, k, v and writes its own row of `out`): the rows
    // are split over the tower's threads, every sum in the order it always had
    let (q, k, v) = (&q, &k, &v);
    par_rows(n, heads * hd, &mut out, cpu_tower_threads(), |first, part| {
        let mut scores = vec![0.0f32; n];
        for (r, row) in part.chunks_mut(heads * hd).enumerate() {
            let i = first + r;
            for hh in 0..heads {
                let qi = i * heads * hd + hh * hd;
                for j in 0..n {
                    let kj = j * heads * hd + hh * hd;
                    let mut acc = 0.0f32;
                    for d in 0..hd {
                        acc += q[qi + d] * k[kj + d];
                    }
                    scores[j] = acc * scale;
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut ssum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = ((*s - mx) as f64).exp() as f32;
                    ssum += *s;
                }
                for j in 0..n {
                    let kj = j * heads * hd + hh * hd;
                    let wj = scores[j] / ssum;
                    for d in 0..hd {
                        row[hh * hd + d] += wj * v[kj + d];
                    }
                }
            }
        }
    });
    let mut proj = vec![0.0f32; n * hidden];
    gemm_vec(&out, &blk.proj_w, &blk.proj_b, &mut proj, hidden, hidden, n);
    proj
}

fn apply_rope_vision(qk: &mut [f32], cos: &[f32], sin: &[f32], n: usize, heads: usize, hd: usize) {
    for i in 0..n {
        for hh in 0..heads {
            let off = i * heads * hd + hh * hd;
            let val: Vec<f32> = qk[off..off + hd].to_vec();
            let rh = rotate_half_hd(&val, hd);
            for d in 0..hd {
                let c = cos[i * hd + d];
                let s = sin[i * hd + d];
                qk[off + d] = val[d] * c + rh[d] * s;
            }
        }
    }
}

/// A decoded + processed vision input ready to splice: merged image embeddings + grid.
/// VISCACHE: Clone so a cache hit hands out the stored rows without disturbing the entry.
#[derive(Clone)]
pub struct VisionOutput {
    /// Merged embeddings [num_tokens, OUT_HIDDEN] (the language embed-splice rows).
    pub merged: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
    /// Number of image tokens = grid_h*grid_w / merge^2.
    pub num_tokens: usize,
    /// VIS-2: PRE-tower identity of the image (hash of the preprocessed pixel_values + grid), for
    /// prefix-cache / checkpoint keys: two different images of the same size expand to identical
    /// image_pad runs, so token ids alone cannot tell them apart (the 2026-08-27 contamination class).
    pub pixel_hash: u64,
}

/// Decode a `data:` image URL (or a bare base64), preprocess it, and run the vision tower.
/// Returns the merged embeddings to splice at the language embed layer.
pub fn process_data_url(tower: &VisualTower, data_url: &str) -> anyhow::Result<VisionOutput> {
    let pre = preprocess_data_url(data_url, &tower.preproc)?;
    let merged = tower.forward_cpu(&pre.pixel_values, pre.grid_h, pre.grid_w);
    let tn = (pre.grid_h * pre.grid_w) / (tower.dims.merge * tower.dims.merge);
    Ok(VisionOutput { merged, grid_h: pre.grid_h, grid_w: pre.grid_w, num_tokens: tn, pixel_hash: pixel_hash(&pre) })
}

/// Decode a `data:` image URL (or bare base64) and run the vision tower on the GPU.
/// Same decode + preprocess as `process_data_url`; only the forward is `GpuVisualTower` instead of
/// `forward_cpu`. The GPU tower must already be built (`GpuVisualTower::new`).
pub fn process_data_url_gpu(gvt: &mut crate::vision_gpu::GpuVisualTower, data_url: &str) -> anyhow::Result<VisionOutput> {
    let pre = preprocess_data_url(data_url, &gvt.host().preproc)?;
    let (merged, _states) = gvt.forward(&pre.pixel_values, pre.grid_h, pre.grid_w, false)?;
    let tn = (pre.grid_h * pre.grid_w) / (gvt.host().dims.merge * gvt.host().dims.merge);
    Ok(VisionOutput { merged, grid_h: pre.grid_h, grid_w: pre.grid_w, num_tokens: tn, pixel_hash: pixel_hash(&pre) })
}

/// Decode + preprocess a `data:` URL into the flatten `pixel_values` + grid (the CPU and GPU paths
/// share this; only the tower forward differs).
fn preprocess_data_url(data_url: &str, cfg: &crate::vision_preproc::VisionPreprocConfig) -> anyhow::Result<crate::vision_preproc::PreprocessedImage> {
    use anyhow::anyhow;
    if data_url.starts_with("http://") || data_url.starts_with("https://") {
        // owner ruling 2026-09-29: no remote fetches (no SSRF surface) — base64 data URLs only
        return Err(anyhow!("remote image URLs are not supported; send the image as a base64 data: URL"));
    }
    let b64 = data_url
        .split("base64,")
        .nth(1)
        .ok_or_else(|| anyhow!("not a base64 data url: {}", &data_url[..data_url.len().min(32)]))?;
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64.trim())?;
    // VIS-2 raw-input safety bound (D-VIS-RESIZE keeps one, independent of --image-max-edge): read
    // the dimensions from the header and refuse to DECODE anything above MAX_INPUT_PIXELS (an RGB
    // decode costs 3 B/px before the resize brings it down).
    let (w0, h0) = image::ImageReader::new(std::io::Cursor::new(&bytes))
        .with_guessed_format()?
        .into_dimensions()?;
    if (w0 as u64) * (h0 as u64) > MAX_INPUT_PIXELS {
        return Err(anyhow!("image {w0}x{h0} exceeds the decode limit of {} MP", MAX_INPUT_PIXELS / 1_000_000));
    }
    let img = image::load_from_memory(&bytes)?.to_rgb8();
    let (w, h) = img.dimensions();
    Ok(crate::vision_preproc::preprocess_image(h as usize, w as usize, img.as_raw(), cfg))
}

/// VIS-2: decoded-pixel ceiling (64 MP, ~192 MB as RGB8) for one input image, checked from the
/// header before decoding. Everything below it is accepted and resized per --image-max-edge.
pub const MAX_INPUT_PIXELS: u64 = 64_000_000;

/// PRE-tower identity of one preprocessed image (pixel_values bits + grid).
fn pixel_hash(pre: &crate::vision_preproc::PreprocessedImage) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write_usize(pre.grid_h);
    h.write_usize(pre.grid_w);
    for v in &pre.pixel_values {
        h.write_u32(v.to_bits());
    }
    h.finish()
}

/// A region of the token stream occupied by one image's merged embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct ImageSpan {
    /// Token index (in the EXPANDED stream) where this image's embeddings begin.
    pub start: usize,
    pub num_tokens: usize,
    /// VIS-2: the merged-token grid (rows x cols = num_tokens) — the mrope h/w extents.
    pub grid_h: usize,
    pub grid_w: usize,
    /// VIS-2: PRE-tower pixel identity (see `VisionOutput::pixel_hash`).
    pub pixel_hash: u64,
}

/// VIS-2: the 3-axis (t, h, w) RoPE position of every token of an EXPANDED prompt, plus the
/// rope delta — HF `Qwen4ExpModel.get_rope_index` (transformers 5.17.0) for one sequence of text and
/// images. Text runs count up from the current position on all three axes. An image span of
/// `grid_h x grid_w` merged tokens at start position `st` gives token j (row r = j / grid_w,
/// col c = j % grid_w) the position (st, st + r, st + c), then the current position advances by
/// max(grid_h, grid_w). delta = max(position) + 1 - len: every position past the prompt is its KV
/// index + delta (delta <= 0).
pub fn mrope_positions(len: usize, spans: &[ImageSpan]) -> anyhow::Result<(Vec<[i64; 3]>, i64)> {
    let mut out = Vec::with_capacity(len);
    let mut cur: i64 = 0;
    let mut i = 0usize;
    let mut sp = spans.iter().peekable();
    while i < len {
        match sp.peek() {
            Some(s) if s.start == i => {
                anyhow::ensure!(s.grid_h * s.grid_w == s.num_tokens && s.num_tokens > 0 && s.start + s.num_tokens <= len,
                                "image span at {} ({} tokens, grid {}x{}) is inconsistent with a {len}-token prompt",
                                s.start, s.num_tokens, s.grid_h, s.grid_w);
                for j in 0..s.num_tokens {
                    let (r, c) = ((j / s.grid_w) as i64, (j % s.grid_w) as i64);
                    out.push([cur, cur + r, cur + c]);
                }
                cur += s.grid_h.max(s.grid_w) as i64;
                i += s.num_tokens;
                sp.next();
            }
            Some(s) => {
                anyhow::ensure!(s.start > i, "image spans out of order at {}", s.start);
                out.push([cur; 3]);
                cur += 1;
                i += 1;
            }
            None => {
                out.push([cur; 3]);
                cur += 1;
                i += 1;
            }
        }
    }
    anyhow::ensure!(sp.next().is_none(), "image span past the end of a {len}-token prompt");
    let max = out.iter().flat_map(|p| p.iter().copied()).max().unwrap_or(-1);
    Ok((out, max + 1 - len as i64))
}

/// What `expand_image_pads` needs per image (in prompt order).
#[derive(Debug, Clone, Copy)]
pub struct ImageInfo {
    pub num_tokens: usize,
    pub grid_h: usize,
    pub grid_w: usize,
    pub pixel_hash: u64,
}

/// Expand each `image_pad` placeholder in `tokens` into `n = grid_h*grid_w/merge²`
/// copies — the number of merged image tokens the vision tower emits — and record the span of each.
/// This is the CPU-side prerequisite for the model-path splice: the expanded token stream has the
/// same length as the merged-embedding rows, and each span tells the prefill where to overwrite.
///
/// `image_token_count` must be in the same order as the images in the rendered prompt.
pub fn expand_image_pads(tokens: &[u32], images: &[ImageInfo], image_pad: u32) -> (Vec<u32>, Vec<ImageSpan>) {
    let mut out = Vec::with_capacity(tokens.len());
    let mut spans = Vec::new();
    let mut img = 0usize;
    for &t in tokens {
        if t == image_pad && img < images.len() {
            let ii = images[img];
            let n = ii.num_tokens;
            let start = out.len();
            for _ in 0..n {
                out.push(image_pad);
            }
            spans.push(ImageSpan { start, num_tokens: n, grid_h: ii.grid_h, grid_w: ii.grid_w, pixel_hash: ii.pixel_hash });
            img += 1;
        } else {
            out.push(t);
        }
    }
    (out, spans)
}

/// The server-side vision-prep result: the expanded token stream (image_pad spans expanded to the
/// merged-token count) plus the concatenated merged embeddings and each image's span. This is what
/// gets threaded to the model prefill, which overwrites the span rows with `image_embeds`.
pub struct PreparedVision {
    pub expanded_tokens: Vec<u32>,
    /// Concatenated merged embeddings, in image order: len = sum(num_tokens) * OUT_HIDDEN.
    pub image_embeds: Vec<f32>,
    pub spans: Vec<ImageSpan>,
    /// VISCACHE: images of this request served from the embed cache (the `cache=<hits>/<N>` of
    /// the dispatch line). The tower never ran for them this request.
    pub cache_hits: usize,
}

// ---------------------------------------------------------------------------------------------
// VISCACHE: a per-image, content-keyed cache of vision-tower OUTPUTS (--vision-embed-cache-mb).
// A chat harness re-sends the same screenshots on every turn; the tower does not need to see an
// image it has already run. Keyed on the RAW PAYLOAD, before any decode: the `data:` URL string
// the prepare loop has in hand. This path never fetches (remote URLs are refused in
// `preprocess_data_url`), so the URL string IS the identity of the content bytes — the same
// string decodes to the same bytes, preprocesses to the same pixel_values, and expands to the
// same span; the stored entry keeps that string and a hit compares it BYTE FOR BYTE, so an
// FNV-1a hash collision can never serve the wrong image (one memcmp per hit against the ~2.3 s
// per image the lookup saved). The value is the whole VisionOutput (merged rows + grid +
// num_tokens + pixel_hash), so `expand_image_pads` and the span assembly run exactly as on a
// miss and the rows that flow downstream are bit-identical to the ones the tower produced.
// LRU + cap follow the spill Policy (kv_spill::Policy::put): least recently used shifts out
// first (the CkptStore tick counter of wp16.rs decides who), and an entry bigger than the whole
// cap is REFUSED, never inserted — the engine just runs the tower for it, as without the cache.
// Head-local state is consistent by construction: under TP the vision dispatch runs on the head
// only (server.rs) and the merged rows ship to the ranks over the wire.
// ---------------------------------------------------------------------------------------------

/// FNV-1a 64-bit (the cache key; the entry's stored string is the exact-match check).
fn fnv1a64(s: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in s {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// One cached image: the payload identity (exact-match check) + the tower's whole output.
#[derive(Clone)]
struct VcEntry {
    /// the `data:` URL string the entry was keyed on — compared on every hit
    key: String,
    /// FNV-1a of `key` — the fast pre-filter in front of the string compare
    hash: u64,
    /// bytes charged against the cap (the payload string + the merged f32 rows)
    bytes: usize,
    /// LRU tick (the CkptStore `used`: raised on every touch, the lowest shifts out)
    used: u64,
    out: VisionOutput,
}

/// Bytes one (key, output) pair occupies.
fn vc_entry_bytes(key: &str, out: &VisionOutput) -> usize {
    key.len() + out.merged.len() * std::mem::size_of::<f32>()
}

/// The cache itself: a byte-capped Vec of entries, LRU by tick. Pure (no locks, no metrics) so
/// the eviction policy is unit-tested without a GPU or the process-global store.
pub struct VisionEmbedCache {
    cap_bytes: usize,
    bytes: usize,
    entries: Vec<VcEntry>,
    tick: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl VisionEmbedCache {
    pub fn new(cap_bytes: usize) -> Self {
        Self { cap_bytes, bytes: 0, entries: Vec::new(), tick: 0, hits: 0, misses: 0, evictions: 0 }
    }
    /// `--vision-embed-cache-mb 0` = off: no lookups counted, nothing kept.
    pub fn enabled(&self) -> bool { self.cap_bytes > 0 }
    pub fn bytes(&self) -> usize { self.bytes }
    /// (hits, misses, evictions) since boot.
    pub fn stats(&self) -> (u64, u64, u64) { (self.hits, self.misses, self.evictions) }
    /// Is this payload key held (no touch — for tests and introspection; `get` is the hot path).
    pub fn holds(&self, key: &str) -> bool {
        let h = fnv1a64(key.as_bytes());
        self.entries.iter().any(|e| e.hash == h && e.key == key)
    }
    /// The stored output for `key`, refreshed to most-recent; None = miss. A hash match with a
    /// different string is a MISS (the exact-match check behind the FNV key).
    pub fn get(&mut self, key: &str) -> Option<VisionOutput> {
        let h = fnv1a64(key.as_bytes());
        self.get_hashed(h, key)
    }
    /// `get` with a precomputed key hash (the collision path is unit-tested through this seam).
    pub fn get_hashed(&mut self, h: u64, key: &str) -> Option<VisionOutput> {
        if self.cap_bytes == 0 { return None; }
        let i = match self.entries.iter().position(|e| e.hash == h && e.key == key) {
            Some(i) => i,
            None => { self.misses += 1; return None; }
        };
        self.tick += 1;
        self.entries[i].used = self.tick;
        self.hits += 1;
        Some(self.entries[i].out.clone())
    }
    /// Insert (or refresh) one image's output; returns how many entries shifted out. An entry
    /// bigger than the whole cap is refused (the kv_spill Policy::put precedent), and the live
    /// total is shifted down to the cap by evicting the least recently used (never the fresh
    /// one, which carries the highest tick).
    pub fn put(&mut self, key: &str, out: &VisionOutput) -> usize {
        if self.cap_bytes == 0 { return 0; }
        let bytes = vc_entry_bytes(key, out);
        if bytes > self.cap_bytes { return 0; }
        self.tick += 1;
        let e = VcEntry { key: key.to_string(), hash: fnv1a64(key.as_bytes()), bytes, used: self.tick, out: out.clone() };
        match self.entries.iter().position(|x| x.hash == e.hash && x.key == e.key) {
            // refresh: replace in place (the old bytes are part of the total, so no underflow)
            Some(i) => { let old = std::mem::replace(&mut self.entries[i], e); self.bytes = self.bytes - old.bytes + bytes; }
            None => { self.bytes += bytes; self.entries.push(e); }
        }
        let mut evicted = 0usize;
        while self.bytes > self.cap_bytes {
            // the CkptStore `acquire` rule: the least-recently-used entry of the whole store goes
            let Some((_, i)) = self.entries.iter().enumerate().map(|(i, e)| (e.used, i)).min() else { break };
            self.bytes -= self.entries.remove(i).bytes;
            evicted += 1;
        }
        self.evictions += evicted as u64;
        evicted
    }
}

/// The one cache of the process (the dispatch runs on tokio blocking workers; a lock poisoned by
/// a panic is recovered `into_inner()`, the `vision_dispatch_with_recovery` style — the entries
/// are plain data and stay sound across a panic of whoever held the lock). The cap is resolved
/// once, lazily, from `--vision-embed-cache-mb` (default 512 MiB; 0 = off).
static VISCACHE: std::sync::LazyLock<std::sync::Mutex<VisionEmbedCache>> = std::sync::LazyLock::new(|| {
    let mb = crate::opts::ranged(crate::opt!("vision-embed-cache-mb")).unwrap_or(512).max(0) as usize;
    std::sync::Mutex::new(VisionEmbedCache::new(mb * (1usize << 20)))
});

/// One image's tower run behind the VISCACHE: look the payload up BEFORE fetch/decode/preprocess/
/// tower (a hit runs none of them), run `tower` on a miss and insert its output. The guards are
/// scoped around the lock — the tower (multi-second) never runs holding it, the same reason the
/// GPU-tower dispatch scopes its guard (PR #16 review R1). Two racing misses on the same image
/// both run the tower; the later insert refreshes the entry. Returns (output, served-from-cache).
fn tower_cached(url: &str, tower: impl FnOnce() -> anyhow::Result<VisionOutput>) -> anyhow::Result<(VisionOutput, bool)> {
    let cache = &*VISCACHE;
    let hit = {
        let mut c = cache.lock().unwrap_or_else(|p| p.into_inner());
        let on = c.enabled();
        let hit = c.get(url);
        (on, hit)
    };
    if let Some(o) = hit.1 {
        crate::metrics::vision_cache_hit();
        return Ok((o, true));
    }
    if hit.0 {
        crate::metrics::vision_cache_miss();
    }
    let out = tower()?;
    let mut c = cache.lock().unwrap_or_else(|p| p.into_inner());
    if c.enabled() {
        let ev = c.put(url, &out);
        crate::metrics::vision_cache_evictions(ev as u64);
        crate::metrics::vision_cache_bytes(c.bytes() as u64);
    }
    Ok((out, false))
}

/// For a request whose messages contain images, decode+preprocess+run the tower for each image (in
/// prompt order) and expand the `image_pad` span accordingly. `image_urls` = the captured
/// `ChatMessage.images[*].url` in prompt order; `prompt_tokens` = the raw (un-expanded) prompt.
pub fn prepare_vision_request(
    tower: &VisualTower,
    image_urls: &[String],
    prompt_tokens: &[u32],
) -> anyhow::Result<PreparedVision> {
    use anyhow::Result as R;
    if image_urls.is_empty() {
        return Ok(PreparedVision { expanded_tokens: prompt_tokens.to_vec(), image_embeds: vec![], spans: vec![], cache_hits: 0 });
    }
    let mut embeds: Vec<f32> = Vec::new();
    let mut infos = Vec::with_capacity(image_urls.len());
    let mut hits = 0usize;
    for url in image_urls {
        // VISCACHE: the payload-keyed memo wraps the whole decode+preprocess+tower.
        let (o, from_cache) = tower_cached(url, || process_data_url(tower, url))?;
        hits += from_cache as usize;
        let mg = tower.dims.merge;
        infos.push(ImageInfo { num_tokens: o.num_tokens, grid_h: o.grid_h / mg, grid_w: o.grid_w / mg, pixel_hash: o.pixel_hash });
        embeds.extend_from_slice(&o.merged);
    }
    let (expanded_tokens, spans) = expand_image_pads(prompt_tokens, &infos, tower.image_pad);
    Ok(PreparedVision { expanded_tokens, image_embeds: embeds, spans, cache_hits: hits })
}

/// GPU twin of `prepare_vision_request`: decode+preprocess on the host, run the GPU tower, expand
/// the `image_pad` span. `image_embeds` comes back as the merged f32 embeddings (the scheduler
/// uploads them to bf16 for the prefill splice, exactly as the CPU path does).
pub fn prepare_vision_request_gpu(
    gvt: &mut crate::vision_gpu::GpuVisualTower,
    image_urls: &[String],
    prompt_tokens: &[u32],
) -> anyhow::Result<PreparedVision> {
    use anyhow::Result as R;
    if image_urls.is_empty() {
        return Ok(PreparedVision { expanded_tokens: prompt_tokens.to_vec(), image_embeds: vec![], spans: vec![], cache_hits: 0 });
    }
    let mut embeds: Vec<f32> = Vec::new();
    let mut infos = Vec::with_capacity(image_urls.len());
    let mut hits = 0usize;
    for url in image_urls {
        // VISCACHE: same payload-keyed memo as the CPU twin — a cached image never touches the
        // GPU tower at all (and the CPU-fallback arm of the dispatch hits the same entries).
        let (o, from_cache) = tower_cached(url, || process_data_url_gpu(gvt, url))?;
        hits += from_cache as usize;
        let mg = gvt.host().dims.merge;
        infos.push(ImageInfo { num_tokens: o.num_tokens, grid_h: o.grid_h / mg, grid_w: o.grid_w / mg, pixel_hash: o.pixel_hash });
        embeds.extend_from_slice(&o.merged);
    }
    let (expanded_tokens, spans) = expand_image_pads(prompt_tokens, &infos, gvt.host().image_pad);
    Ok(PreparedVision { expanded_tokens, image_embeds: embeds, spans, cache_hits: hits })
}

/// S-B9-REL-VISIONFIX F2: the CPU tower's thread count changes who computes a row, never a bit.
#[cfg(test)]
mod cpu_tower_thread_tests {
    use super::*;
    use crate::vision_tower::{TowerDims, VisualBlock};

    /// `forward_cpu` reads a process-global thread count: the tests that set it run one at a time.
    static THREADS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn lcg(seed: u64) -> impl FnMut(usize, f32, f32) -> Vec<f32> {
        let mut x = seed | 1;
        move |n, scale, bias| (0..n).map(|_| {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            bias + scale * (((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
        }).collect()
    }

    fn small_tower() -> VisualTower {
        let dims = TowerDims { hidden: 48, depth: 2, inter: 96, heads: 3, num_pos: 64, patch: 2, temporal: 2, merge: 2, in_ch: 3, out_hidden: 40 };
        let mut w = lcg(0xC0FFEE);
        let (h, i, mi) = (dims.hidden, dims.inter, dims.merge_inter());
        let blocks = (0..dims.depth).map(|_| VisualBlock {
            norm1_w: w(h, 0.2, 1.0), norm1_b: w(h, 0.1, 0.0), norm2_w: w(h, 0.2, 1.0), norm2_b: w(h, 0.1, 0.0),
            qkv_w: w(3 * h * h, 0.25, 0.0), qkv_b: w(3 * h, 0.1, 0.0), proj_w: w(h * h, 0.25, 0.0), proj_b: w(h, 0.1, 0.0),
            fc1_w: w(i * h, 0.25, 0.0), fc1_b: w(i, 0.1, 0.0), fc2_w: w(h * i, 0.25, 0.0), fc2_b: w(h, 0.1, 0.0),
        }).collect();
        VisualTower {
            dims,
            preproc: crate::vision_preproc::VisionPreprocConfig { patch_size: 2, merge_size: 2, temporal_patch_size: 2, in_channels: 3,
                                                                  min_pixels: 16, max_pixels: 1 << 20, max_edge: 0 },
            image_pad: 0,
            patch_embed_w: w(h * dims.wpv(), 0.5, 0.0), patch_embed_b: w(h, 0.1, 0.0), pos_embed_w: w(dims.num_pos * h, 0.5, 0.0),
            blocks,
            merger_norm_w: w(h, 0.2, 1.0), merger_norm_b: w(h, 0.1, 0.0),
            merger_fc1_w: w(mi * mi, 0.1, 0.0), merger_fc1_b: w(mi, 0.1, 0.0),
            merger_fc2_w: w(dims.out_hidden * mi, 0.1, 0.0), merger_fc2_b: w(dims.out_hidden, 0.1, 0.0),
            source: "synthetic".into(),
        }
    }

    /// The GEMM as it is now (token blocks, any thread count) against the first implementation,
    /// bit for bit: shapes that are not multiples of the block or of the thread count included.
    #[test]
    fn the_blocked_threaded_gemm_is_bit_identical_to_the_first_implementation() {
        let _g = THREADS.lock().unwrap_or_else(|e| e.into_inner());
        let mut w = lcg(7);
        for (n, inn, outn) in [(1usize, 5usize, 3usize), (16, 8, 8), (17, 33, 9), (50, 24, 70), (131, 48, 96)] {
            let (x, wt, b) = (w(n * inn, 2.0, 0.0), w(outn * inn, 2.0, 0.0), w(outn, 1.0, 0.0));
            let mut want = vec![0f32; n * outn];
            gemm_vec_reference(&x, &wt, &b, &mut want, inn, outn, n);
            for threads in [1usize, 2, 3, 8, 64] {
                set_cpu_tower_threads(threads);
                let mut got = vec![9f32; n * outn];
                gemm_vec(&x, &wt, &b, &mut got, inn, outn, n);
                assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "n={n} inn={inn} outn={outn} threads={threads}");
            }
        }
        set_cpu_tower_threads(1);
    }

    /// Instrument (ignored): the first implementation against the blocked kernel at the served MLP
    /// shape (784 tokens of a 448 x 448 image, 1152 -> 4304), one thread, bits compared.
    #[test]
    #[ignore]
    fn instrument_gemm_timing_at_a_served_shape() {
        let _g = THREADS.lock().unwrap_or_else(|e| e.into_inner());
        let mut w = lcg(11);
        let (n, inn, outn) = (784usize, 1152usize, 1152usize);
        let (x, wt, b) = (w(n * inn, 2.0, 0.0), w(outn * inn, 0.1, 0.0), w(outn, 1.0, 0.0));
        set_cpu_tower_threads(1);
        let (mut want, mut got) = (vec![0f32; n * outn], vec![0f32; n * outn]);
        let t = std::time::Instant::now();
        gemm_vec_reference(&x, &wt, &b, &mut want, inn, outn, n);
        let t_ref = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        gemm_vec(&x, &wt, &b, &mut got, inn, outn, n);
        let t_new = t.elapsed().as_secs_f64();
        assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "bits differ");
        println!("GEMM {n}x{inn}x{outn}: first implementation {t_ref:.3} s, blocked {t_new:.3} s, x{:.2}", t_ref / t_new);
    }

    /// The whole tower: every thread count gives the embeddings of one thread, bit for bit, on
    /// square and non-square grids (attention rows, patch embed, MLPs and the merger all split).
    #[test]
    fn the_cpu_tower_is_bit_identical_for_every_thread_count() {
        let _g = THREADS.lock().unwrap_or_else(|e| e.into_inner());
        let tower = small_tower();
        for (gh, gw) in [(4usize, 4usize), (6, 4), (2, 10)] {
            let pv: Vec<f32> = (0..gh * gw * tower.dims.wpv()).map(|i| ((i * 37 + gh) % 255) as f32 / 127.5 - 1.0).collect();
            set_cpu_tower_threads(1);
            let one = tower.forward_cpu(&pv, gh, gw);
            assert!(one.iter().all(|v| v.is_finite()) && one.iter().any(|v| *v != 0.0));
            for threads in [2usize, 3, 7, 16, 64] {
                set_cpu_tower_threads(threads);
                let many = tower.forward_cpu(&pv, gh, gw);
                assert!(one.len() == many.len() && one.iter().zip(&many).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "grid {gh}x{gw}: {threads} threads differ from 1");
            }
        }
        set_cpu_tower_threads(1);
    }

    /// The option: default 1, a registry option with no environment variable, validated range.
    #[test]
    fn the_thread_option_is_a_flag_with_the_old_default() {
        let o = &crate::opts::REG[crate::opts::idx("vision-cpu-threads")];
        assert_eq!((o.def, o.env), ("1", ""));
        assert!(o.help.contains("bit-identical") && o.help.contains("Default 1"));
        for (v, want) in [("1", Some(1usize)), ("8", Some(8)), (" 64 ", Some(64)), ("0", None), ("65", None), ("-1", None), ("many", None), ("", None)] {
            assert_eq!(crate::vision_tower::parse_cpu_tower_threads(v).ok(), want, "{v:?}");
        }
        // the clamp of the setter itself (a caller that bypasses the parser cannot ask for 0)
        let _g = THREADS.lock().unwrap_or_else(|e| e.into_inner());
        set_cpu_tower_threads(0);
        assert_eq!(cpu_tower_threads(), 1);
        set_cpu_tower_threads(1);
    }
}

#[cfg(test)]
mod vis2_tests {
    use super::*;
    #[test]
    fn mrope_positions_hand_case() {
        let spans = [ImageSpan { start: 3, num_tokens: 6, grid_h: 2, grid_w: 3, pixel_hash: 1 },
                     ImageSpan { start: 11, num_tokens: 2, grid_h: 1, grid_w: 2, pixel_hash: 2 }];
        let (p, d) = mrope_positions(14, &spans).unwrap();
        let want: Vec<[i64; 3]> = vec![
            [0, 0, 0], [1, 1, 1], [2, 2, 2],
            [3, 3, 3], [3, 3, 4], [3, 3, 5], [3, 4, 3], [3, 4, 4], [3, 4, 5],
            [6, 6, 6], [7, 7, 7],
            [8, 8, 8], [8, 8, 9],
            [10, 10, 10],
        ];
        assert_eq!(p, want);
        assert_eq!(d, 11 - 14);
        let (p, d) = mrope_positions(5, &[]).unwrap();
        assert_eq!(p, (0..5).map(|i| [i; 3]).collect::<Vec<_>>());
        assert_eq!(d, 0);
        assert!(mrope_positions(4, &[ImageSpan { start: 2, num_tokens: 6, grid_h: 2, grid_w: 3, pixel_hash: 0 }]).is_err());
    }
}

/// VISCACHE: the payload-keyed embed cache — eviction order, cap enforcement (fake byte sizes,
/// no GPU), and the exact-match check behind the FNV key.
#[cfg(test)]
mod viscache_tests {
    use super::*;

    /// A fake tower output: `n` merged f32 rows of one value (entry bytes = key + 4n).
    fn out(v: f32, n: usize) -> VisionOutput {
        VisionOutput { merged: vec![v; n], grid_h: 1, grid_w: n, num_tokens: n, pixel_hash: v.to_bits() as u64 }
    }

    /// LRU order + cap: the least recently USED (not the oldest inserted) shifts out, the live
    /// total never exceeds the cap, an over-cap payload is refused (the kv_spill Policy::put
    /// precedent), cap 0 keeps nothing and counts nothing.
    #[test]
    fn the_cache_shifts_out_the_least_recently_used_until_under_the_cap() {
        // entry size = key.len() + 4 * merged.len() = 1 + 8 = 9 bytes; the cap holds two.
        let mut c = VisionEmbedCache::new(20);
        c.put("a", &out(1.0, 2));
        c.put("b", &out(2.0, 2));
        assert_eq!(c.bytes(), 18);
        assert!(c.get("a").is_some()); // touch a -> b is now the least recently used
        assert_eq!(c.put("c", &out(3.0, 2)), 1); // 27 > 20: exactly one shift-out
        assert!(c.holds("a") && !c.holds("b") && c.holds("c"), "b (LRU) goes, a (touched) stays");
        assert_eq!(c.bytes(), 18);
        assert!(c.bytes() <= 20);
        assert_eq!(c.stats(), (1, 0, 1), "1 get-hit, 0 lookups-missed, 1 eviction");
        // over-cap payload: refused, the store unchanged
        assert_eq!(c.put("z", &out(9.0, 100)), 0); // 1 + 400 bytes > the whole cap
        assert!(!c.holds("z"));
        assert_eq!(c.bytes(), 18);
        // a refresh of a held key replaces it, no eviction, no duplicate
        assert_eq!(c.put("a", &out(1.5, 1)), 0);
        assert_eq!(c.bytes(), 14); // 1 + 4 + (1 + 8)
        assert_eq!(c.holds("c"), true);
        // cap 0 = off: nothing kept, nothing counted
        let mut off = VisionEmbedCache::new(0);
        off.put("a", &out(1.0, 2));
        assert!(off.get("a").is_none());
        assert_eq!(off.stats(), (0, 0, 0));
    }

    /// A hash collision must miss: same hash, different payload string, the stored key is
    /// compared byte for byte and the tower (the caller's `run`) is not short-circuited. The hit
    /// path returns the stored rows bit for bit — what a miss would have flowed downstream.
    #[test]
    fn a_same_hash_different_payload_lookup_misses_and_a_hit_is_bit_exact() {
        let mut c = VisionEmbedCache::new(1 << 20);
        let stored = out(-0.75, 3);
        c.put("data:image/png;base64,AAA", &stored);
        // forged collision: the hash of the stored key, the string of a different payload
        let h = fnv1a64(b"data:image/png;base64,AAA");
        assert!(c.get_hashed(h, "data:image/png;base64,AAB").is_none(), "same hash, different bytes: miss");
        assert_eq!(c.stats(), (0, 1, 0), "the collision counted as a miss, not a hit");
        // the real key hits, and the served rows are the stored rows bit for bit
        let got = c.get("data:image/png;base64,AAA").unwrap_or_else(|| panic!("the exact key must hit"));
        assert!(got.merged.iter().zip(&stored.merged).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert_eq!((got.grid_h, got.grid_w, got.num_tokens, got.pixel_hash),
                   (stored.grid_h, stored.grid_w, stored.num_tokens, stored.pixel_hash));
        assert_eq!(c.stats(), (1, 1, 0));
    }

    /// The registry row: `--vision-embed-cache-mb`, default 512, range from 0 (= off).
    #[test]
    fn the_cache_flag_defaults_to_512_mib_and_zero_is_off() {
        let o = &crate::opts::REG[crate::opts::idx("vision-embed-cache-mb")];
        assert_eq!((o.def, o.env), ("512", ""));
        assert!(matches!(o.ty, crate::opts::Ty::IntRange(0, _)), "the range must start at 0 (0 = off)");
        let id = crate::opt!("vision-embed-cache-mb");
        assert_eq!(crate::opts::ranged_from(id, None).unwrap_or(-1), 512, "the unset default is 512 MiB");
        assert_eq!(crate::opts::ranged_from(id, Some("0")).unwrap_or(-1), 0);
        assert!(crate::opts::ranged_from(id, Some("-1")).is_err());
        assert!(crate::opts::ranged_from(id, Some("many")).is_err());
    }
}
