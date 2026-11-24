//! Host references for the KM kernels (f64 where the reference is a real-number formula, faithful fp32 where the op order is
//! the contract). Test-only. The formulas are model.py 281-293, 328-365, 794-829, 940-996 and kernel.py 407-474.

pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    pub fn uni(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn normal(&mut self) -> f32 {
        let (u1, u2) = (self.uni().max(1e-12), self.uni());
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}
pub fn bf(x: f32) -> f32 { half::bf16::from_f32(x).to_f32() }
pub fn bf_bits(x: f32) -> u16 { half::bf16::from_f32(x).to_bits() }
pub fn from_bits(b: u16) -> f32 { half::bf16::from_bits(b).to_f32() }

/// The 24-value mix record (pre 4, post 4, comb 16) from the whole stream (4 x dim, bf16-valued f32), in f64.
pub fn hc_record(stream: &[f32], fnw: &[f32], k: usize, scale: &[f32; 3], base: &[f32; 24], norm_eps: f64, hc_eps: f64, iters: usize) -> [f64; 24] {
    let ss: f64 = stream.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let rs = 1.0 / (ss / k as f64 + norm_eps).sqrt();
    let mut mix = [0f64; 24];
    for m in 0..24 {
        let mut s = 0f64;
        for i in 0..k { s += fnw[m * k + i] as f64 * stream[i] as f64; }
        mix[m] = s * rs;
    }
    record_from_mixes(&mix, scale, base, hc_eps, iters)
}
pub fn record_from_mixes(mix: &[f64; 24], scale: &[f32; 3], base: &[f32; 24], eps: f64, iters: usize) -> [f64; 24] {
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    let mut o = [0f64; 24];
    for j in 0..4 { o[j] = sig(mix[j] * scale[0] as f64 + base[j] as f64) + eps; }
    for j in 0..4 { o[4 + j] = 2.0 * sig(mix[4 + j] * scale[1] as f64 + base[4 + j] as f64); }
    let mut c = [[0f64; 4]; 4];
    for j in 0..4 { for k in 0..4 { c[j][k] = mix[8 + 4 * j + k] * scale[2] as f64 + base[8 + 4 * j + k] as f64; } }
    for j in 0..4 {
        let mx = c[j].iter().cloned().fold(f64::MIN, f64::max);
        let mut s = 0.0;
        for k in 0..4 { c[j][k] = (c[j][k] - mx).exp(); s += c[j][k]; }
        for k in 0..4 { c[j][k] = c[j][k] / s + eps; }
    }
    let col = |c: &mut [[f64; 4]; 4]| { for k in 0..4 { let s: f64 = (0..4).map(|j| c[j][k]).sum::<f64>() + eps; for j in 0..4 { c[j][k] /= s; } } };
    let row = |c: &mut [[f64; 4]; 4]| { for j in 0..4 { let s: f64 = c[j].iter().sum::<f64>() + eps; for k in 0..4 { c[j][k] /= s; } } };
    col(&mut c);
    for _ in 1..iters { row(&mut c); col(&mut c); }
    for j in 0..4 { for k in 0..4 { o[8 + 4 * j + k] = c[j][k]; } }
    o
}

/// hc_pre exactly as the kernel orders it (fp32, products rounded, j ascending), bf16 result; then the RMSNorm in f64.
pub fn hc_pre_y(x: &[f32], dim: usize, pre: &[f32; 4]) -> Vec<f32> {
    (0..dim).map(|d| {
        let mut s = pre[0] * x[d];
        for j in 1..4 { s += pre[j] * x[j * dim + d]; }
        bf(s)
    }).collect()
}
pub fn rmsnorm_ref(y: &[f32], w: Option<&[f32]>, eps: f64) -> Vec<f64> {
    let n = y.len() as f64;
    let ms: f64 = y.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / n;
    let r = 1.0 / (ms + eps).sqrt();
    y.iter().enumerate().map(|(i, &v)| v as f64 * r * w.map_or(1.0, |w| w[i] as f64)).collect()
}
/// hc_post exactly as the kernel orders it.
pub fn hc_post_ref(xs: &[f32], resid: &[f32], dim: usize, post: &[f32], comb: &[f32]) -> Vec<f32> {
    let mut o = vec![0f32; 4 * dim];
    for k in 0..4 { for d in 0..dim {
        let mut s = comb[k] * resid[d];
        for j in 1..4 { s += comb[4 * j + k] * resid[j * dim + d]; }
        o[k * dim + d] = bf(post[k] * xs[d] + s);
    } }
    o
}
pub fn softplus_sqrt(x: f64) -> f64 { (if x > 20.0 { x } else { x.exp().ln_1p() }).sqrt() }
/// (indices, weights) of the router for one row of scores (f64), ties to the lower id.
pub fn route_ref(scores: &[f64], bias: &[f64], topk: usize, scale: f64) -> (Vec<usize>, Vec<f64>, f64) {
    let s: Vec<f64> = scores.iter().map(|&x| softplus_sqrt(x)).collect();
    let mut idx: Vec<usize> = (0..s.len()).collect();
    idx.sort_by(|&a, &b| (s[b] + bias[b]).partial_cmp(&(s[a] + bias[a])).unwrap().then(a.cmp(&b)));
    let pick: Vec<usize> = idx[..topk].to_vec();
    // the margin between the last pick and the first non-pick (a near-tie is not a defect)
    let margin = (s[idx[topk - 1]] + bias[idx[topk - 1]]) - (s[idx[topk]] + bias[idx[topk]]);
    let sum: f64 = pick.iter().map(|&i| s[i]).sum();
    let w = pick.iter().map(|&i| s[i] / (sum + 1e-20) * scale).collect();
    (pick, w, margin)
}
/// Engram gate for one (token, copy): returns the gate value (f64).
pub fn engram_gate_ref(h: &[f32], key: &[f32], wq: &[f32], wk: &[f32], eps: f64) -> f64 {
    let n = h.len() as f64;
    let rh = 1.0 / (h.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / n + eps).sqrt();
    let rk = 1.0 / (key.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / n + eps).sqrt();
    let dot: f64 = (0..h.len()).map(|i| h[i] as f64 * (wq[i] as f32 * wk[i]) as f64 * key[i] as f64).sum::<f64>() * rh * rk * n.powf(-0.5);
    let g = dot.abs().max(1e-6).sqrt().copysign(dot);
    1.0 / (1.0 + (-g).exp())
}
