//! The f32 CPU oracle of the Inkling-2 vision tower (op for op after `Inkling2VisionModel`, transformers main), f64
//! accumulation in the dot products. Gate: per-block rel-L2 vs transformers' f32 run (tools/inkling2/ref_vision.py).

use super::preproc::PreImage;
use super::{interp_axis, rope_inv_freq, Grid, VisionHost};
use crate::inkling2::cpu::matmul;

/// Taps of one forward: `embed` (patch + position, CLS), `ln_pre`, `layer{i}` outputs, `merger` ([tokens][2304]).
pub struct Taps { pub embed: Vec<f32>, pub ln_pre: Vec<f32>, pub layers: Vec<Vec<f32>>, pub merger: Vec<f32> }

/// erf in f64: the Taylor series below |x| < 3 (cancellation costs ~3 digits: |err| ~ 1e-13), Abramowitz-Stegun 7.1.26
/// above (|err| < 1.5e-7 where erf is within 2.2e-5 of 1).
pub fn erf64(x: f64) -> f64 {
    let (s, a) = (x.signum(), x.abs());
    if a < 3.0 {
        let (mut sum, mut term, a2) = (a, a, a * a);
        for n in 1..200 {
            term *= -a2 / n as f64;
            let c = term / (2 * n + 1) as f64;
            sum += c;
            if c.abs() < 1e-17 * sum.abs() { break; }
        }
        return s * sum * 2.0 / std::f64::consts::PI.sqrt();
    }
    let t = 1.0 / (1.0 + 0.3275911 * a);
    s * (1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-a * a).exp())
}

pub fn gelu(x: f32) -> f32 { let x = x as f64; (0.5 * x * (1.0 + erf64(x / std::f64::consts::SQRT_2))) as f32 }

/// LayerNorm (biased variance, eps), f64 statistics.
pub fn layernorm(x: &[f32], d: usize, w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let mut o = vec![0f32; x.len()];
    for (r, row) in x.chunks_exact(d).enumerate() {
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
        let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
        let inv = 1.0 / (var + eps as f64).sqrt();
        for i in 0..d { o[r * d + i] = (((row[i] as f64 - mean) * inv) as f32) * w[i] + b[i]; }
    }
    o
}

/// The patch matrix `[patches][1200]` of an image in the reference's sequence order (block-major, (py, px, c)).
pub fn patch_matrix(pre: &PreImage, patch: usize) -> Vec<f32> {
    let g = pre.grid;
    let tile = 2 * patch;
    let pin = 3 * patch * patch;
    let mut x = vec![0f32; g.patches() * pin];
    for p in 0..g.patches() {
        let t = p / 4;
        let (my, mx) = ((p % 4) / 2, p % 2);
        for py in 0..patch {
            for px in 0..patch {
                let src = t * tile * tile * 3 + ((my * patch + py) * tile + mx * patch + px) * 3;
                let dst = p * pin + (py * patch + px) * 3;
                x[dst..dst + 3].copy_from_slice(&pre.pixels[src..src + 3]);
            }
        }
    }
    x
}

/// The interpolated position embedding of every patch `[patches][1024]` (f32, 4 taps summed in tap order).
pub fn pos_embed(host: &VisionHost, g: Grid) -> Vec<f32> {
    let c = &host.cfg;
    let d = c.hidden;
    let (ph, pw) = (2 * g.rows, 2 * g.cols);
    let mut o = vec![0f32; g.patches() * d];
    for p in 0..g.patches() {
        let (r, col) = g.patch_rc(p);
        let (ht, hw) = interp_axis(r, ph, c.grid);
        let (wt, ww) = interp_axis(col, pw, c.grid);
        let mut taps = [(0usize, 0f32); 4];
        for a in 0..2 { for b in 0..2 { taps[a * 2 + b] = (ht[a] * c.grid + wt[b], hw[a] * ww[b]); } }
        for i in 0..d {
            let mut s = 0f32;
            for &(idx, w) in &taps { s += host.pos[idx * d + i] * w; }
            o[p * d + i] = s;
        }
    }
    o
}

/// Per sequence row: the rotation angle of each of the 32 channel pairs.
pub fn rope_angles(host: &VisionHost, g: Grid) -> Vec<f32> {
    let f = rope_inv_freq(&host.cfg);
    let nf = f.len();
    let mut a = vec![0f32; g.seq() * 2 * nf];
    for p in 0..g.patches() {
        let (r, col) = g.patch_rc(p);
        let row = &mut a[(1 + p) * 2 * nf..(2 + p) * 2 * nf];
        for j in 0..nf { row[j] = (col + 1) as f32 * f[j]; row[nf + j] = (r + 1) as f32 * f[j]; }
    }
    a
}

fn rope(x: &mut [f32], ang: &[f32], heads: usize, hd: usize) {
    let np = hd / 2;
    for (r, row) in x.chunks_exact_mut(heads * hd).enumerate() {
        let an = &ang[r * np..(r + 1) * np];
        for h in 0..heads {
            for j in 0..np {
                let (c, s) = (an[j].cos(), an[j].sin());
                let (x1, x2) = (row[h * hd + 2 * j], row[h * hd + 2 * j + 1]);
                row[h * hd + 2 * j] = x1 * c - x2 * s;
                row[h * hd + 2 * j + 1] = x2 * c + x1 * s;
            }
        }
    }
}

fn add_bias(y: &mut [f32], b: &[f32]) { let n = b.len(); for row in y.chunks_exact_mut(n) { for i in 0..n { row[i] += b[i]; } } }

/// Full non-causal attention of one image, f64 softmax.
fn attention(q: &[f32], k: &[f32], v: &[f32], s: usize, heads: usize, hd: usize) -> Vec<f32> {
    let d = heads * hd;
    let mut o = vec![0f32; s * d];
    let nt = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
    let scale = 1.0 / (hd as f64).sqrt();
    let rows_per = s.div_ceil(nt);
    std::thread::scope(|sc| {
        for (ti, ochunk) in o.chunks_mut(rows_per * d).enumerate() {
            sc.spawn(move || {
                let mut sc_ = vec![0f64; s];
                for (lr, orow) in ochunk.chunks_exact_mut(d).enumerate() {
                    let i = ti * rows_per + lr;
                    for h in 0..heads {
                        let qi = &q[i * d + h * hd..i * d + h * hd + hd];
                        let mut mx = f64::NEG_INFINITY;
                        for j in 0..s {
                            let kj = &k[j * d + h * hd..j * d + h * hd + hd];
                            let mut a = 0f64;
                            for t in 0..hd { a += qi[t] as f64 * kj[t] as f64; }
                            sc_[j] = a * scale;
                            mx = mx.max(sc_[j]);
                        }
                        let mut tot = 0f64;
                        for j in 0..s { sc_[j] = (sc_[j] - mx).exp(); tot += sc_[j]; }
                        for t in 0..hd {
                            let mut a = 0f64;
                            for j in 0..s { a += sc_[j] * v[j * d + h * hd + t] as f64; }
                            orow[h * hd + t] = (a / tot) as f32;
                        }
                    }
                }
            });
        }
    });
    o
}

/// One image through the tower; `taps` keeps every block's output.
pub fn forward(host: &VisionHost, pre: &PreImage, keep_layers: bool) -> Taps {
    let c = &host.cfg;
    let (d, g) = (c.hidden, pre.grid);
    let s = g.seq();
    let xp = patch_matrix(pre, c.patch);
    let pe = matmul(&xp, g.patches(), &host.patch_w, d, c.patch_in());
    let pos = pos_embed(host, g);
    let mut h = vec![0f32; s * d];
    h[..d].copy_from_slice(&host.cls);
    for p in 0..g.patches() { for i in 0..d { h[(1 + p) * d + i] = pe[p * d + i] + pos[p * d + i]; } }
    let embed = h.clone();
    h = layernorm(&h, d, &host.ln_pre_w, &host.ln_pre_b, c.eps);
    let ln_pre = h.clone();
    let ang = rope_angles(host, g);
    let mut layers = Vec::new();
    for l in &host.layers {
        let x = layernorm(&h, d, &l.ln1_w, &l.ln1_b, c.eps);
        let mut q = matmul(&x, s, &l.q_w, d, d); add_bias(&mut q, &l.q_b);
        let mut k = matmul(&x, s, &l.k_w, d, d); add_bias(&mut k, &l.k_b);
        let mut v = matmul(&x, s, &l.v_w, d, d); add_bias(&mut v, &l.v_b);
        rope(&mut q, &ang, c.heads, c.head_dim);
        rope(&mut k, &ang, c.heads, c.head_dim);
        let a = attention(&q, &k, &v, s, c.heads, c.head_dim);
        let mut o = matmul(&a, s, &l.o_w, d, d); add_bias(&mut o, &l.o_b);
        for r in 0..s { for i in 0..d { h[r * d + i] += l.ls1[i] * o[r * d + i]; } }
        let x = layernorm(&h, d, &l.ln2_w, &l.ln2_b, c.eps);
        let mut f = matmul(&x, s, &l.fc1_w, c.inter, d); add_bias(&mut f, &l.fc1_b);
        for v in f.iter_mut() { *v = gelu(*v); }
        let mut y = matmul(&f, s, &l.fc2_w, d, c.inter); add_bias(&mut y, &l.fc2_b);
        for r in 0..s { for i in 0..d { h[r * d + i] += l.ls2[i] * y[r * d + i]; } }
        if keep_layers { layers.push(h.clone()); }
    }
    let m = &h[d..];
    let t = g.tokens();
    let mut z = matmul(m, t, &host.proj_in, c.proj_hidden, 4 * d);
    for v in z.iter_mut() { *v = gelu(*v); }
    let merger = matmul(&z, t, &host.proj_out, c.text_hidden, c.proj_hidden);
    Taps { embed, ln_pre, layers, merger }
}
