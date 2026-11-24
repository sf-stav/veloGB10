//! DSV4.1 CPU oracle (leg OF): the numeric formats of the reference's quantised linears, in plain f32 / f64.
//! REF map §2.2 (`act_quant`), §2.4 (`fp8_gemm`), §2.5 (`fp4_gemm`). Nothing here touches a GPU.
//!
//! * `act_quant`: per row, per group of 32: `s = pow2ceil(max(amax, 1e-4) * fp32(1/448))`,
//!   `q = e4m3(clamp(x / s, -448, 448))` (round to nearest even), value `q * s`.
//! * GEMM: per 32-wide K block the 32 exact products are summed (here in f64, which is exact for e4m3 x e4m3
//!   products), rounded once to f32, scaled by the two power-of-two scales (exact), and added to an f32
//!   accumulator sequentially over the K blocks; one bf16 rounding of the total.

use half::bf16;

pub const GROUP: usize = 32;

/// bf16 round-trip (round to nearest even).
#[inline]
pub fn bf16r(x: f32) -> f32 { bf16::from_f32(x).to_f32() }

/// The 256 e4m3fn code values (NaN codes decode to NaN).
pub fn e4m3_table() -> [f32; 256] {
    let mut t = [0f32; 256];
    for c in 0..256usize {
        let s = if c & 0x80 != 0 { -1.0f32 } else { 1.0 };
        let e = (c >> 3) & 0xf;
        let m = (c & 7) as f32;
        t[c] = if e == 0xf && (c & 7) == 7 { f32::NAN }
               else if e == 0 { s * m * (2f32).powi(-9) }
               else { s * (1.0 + m / 8.0) * (2f32).powi(e as i32 - 7) };
    }
    t
}

/// Positive finite e4m3 magnitudes in code order (codes 0..=126), for the nearest-even encoder.
fn e4m3_pos() -> &'static Vec<f32> {
    static T: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    T.get_or_init(|| { let t = e4m3_table(); (0..127).map(|c| t[c]).collect() })
}

/// fp32 -> e4m3fn code with saturation to +-448 and round to nearest even (ties to the even code).
pub fn e4m3_encode(x: f32) -> u8 {
    let neg = x.is_sign_negative();
    let a = x.abs();
    let p = e4m3_pos();
    let code: usize = if !(a < 448.0) { 126 } else {
        // first index with p[i] >= a
        let i = p.partition_point(|&v| v < a);
        if i == 0 { 0 } else {
            let (lo, hi) = (p[i - 1] as f64, p[i] as f64);
            let d_lo = a as f64 - lo;
            let d_hi = hi - a as f64;
            if d_lo < d_hi { i - 1 } else if d_hi < d_lo { i } else if (i - 1) % 2 == 0 { i - 1 } else { i }
        }
    };
    (code as u8) | if neg { 0x80 } else { 0 }
}

/// e8m0 byte -> 2^(b - 127) (255 = NaN in the format; never stored by this model).
#[inline]
pub fn e8m0(b: u8) -> f32 { f32::from_bits((b as u32) << 23) }

/// e2m1 nibble -> value (sign bit 3).
pub const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// `fast_round_scale`: the smallest power of two >= v (v = amax * fp32(1/448)), by exponent bits.
#[inline]
pub fn pow2ceil_bits(v: f32) -> f32 {
    let b = v.to_bits();
    let e = ((b >> 23) & 0xff) as i32 - 127 + if b & 0x7f_ffff != 0 { 1 } else { 0 };
    f32::from_bits(((e + 127) as u32) << 23)
}

/// A quantised activation: `q` = e4m3 values (unscaled), `s` = one power-of-two scale per group of 32.
pub struct ActQ { pub m: usize, pub k: usize, pub q: Vec<f32>, pub s: Vec<f32> }

/// `act_quant(x, 32, ue8m0)` on `m` rows of `k` (values already bf16-exact f32).
pub fn act_quant(x: &[f32], m: usize, k: usize) -> ActQ {
    assert_eq!(x.len(), m * k);
    assert_eq!(k % GROUP, 0);
    let tab = e4m3_table();
    let inv = 1.0f32 / 448.0f32;
    let mut q = vec![0f32; m * k];
    let mut s = vec![0f32; m * k / GROUP];
    for r in 0..m {
        for g in 0..k / GROUP {
            let base = r * k + g * GROUP;
            let mut amax = 0f32;
            for &v in &x[base..base + GROUP] { amax = amax.max(v.abs()); }
            let amax = amax.max(1e-4);
            let sc = pow2ceil_bits(amax * inv);
            s[r * (k / GROUP) + g] = sc;
            for j in 0..GROUP {
                let v = (x[base + j] / sc).clamp(-448.0, 448.0);
                q[base + j] = tab[e4m3_encode(v) as usize];
            }
        }
    }
    ActQ { m, k, q, s }
}

/// Dequantised values of an `ActQ` (what the inplace variant would write back), bf16-rounded.
pub fn act_dequant_bf16(a: &ActQ) -> Vec<f32> {
    let g = a.k / GROUP;
    (0..a.m * a.k).map(|i| bf16r(a.q[i] * a.s[(i / a.k) * g + (i % a.k) / GROUP])).collect()
}

/// A quantised weight matrix `[n, k]` the GEMM can read a decoded (unscaled) row of.
pub trait QRows: Sync {
    fn n(&self) -> usize;
    fn k(&self) -> usize;
    /// Unscaled decoded values of output row `n` (length k).
    fn decode_row(&self, n: usize, out: &mut [f32]);
    /// Power-of-two scale of (row n, K block kb).
    fn scale(&self, n: usize, kb: usize) -> f32;
}

/// FP8 weight with 32 x 32 block scales (`fp8_gemm`): codes `[n, k]`, scales `[ceil(n/32), k/32]` as e8m0 bytes.
pub struct Fp8W { pub n: usize, pub k: usize, pub codes: Vec<u8>, pub scale: Vec<u8> }
impl QRows for Fp8W {
    fn n(&self) -> usize { self.n }
    fn k(&self) -> usize { self.k }
    fn decode_row(&self, n: usize, out: &mut [f32]) {
        let t = e4m3_table();
        for (o, &c) in out.iter_mut().zip(&self.codes[n * self.k..(n + 1) * self.k]) { *o = t[c as usize]; }
    }
    fn scale(&self, n: usize, kb: usize) -> f32 { e8m0(self.scale[(n / 32) * (self.k / GROUP) + kb]) }
}

/// MXFP4 expert weight (`fp4_gemm`): packed nibbles `[n, k/2]` (low nibble = even k), scales `[n, k/32]` e8m0 bytes.
pub struct Fp4W { pub n: usize, pub k: usize, pub packed: Vec<u8>, pub scale: Vec<u8> }
impl QRows for Fp4W {
    fn n(&self) -> usize { self.n }
    fn k(&self) -> usize { self.k }
    fn decode_row(&self, n: usize, out: &mut [f32]) {
        let row = &self.packed[n * self.k / 2..(n + 1) * self.k / 2];
        for (i, &b) in row.iter().enumerate() {
            out[2 * i] = E2M1[(b & 0xf) as usize];
            out[2 * i + 1] = E2M1[(b >> 4) as usize];
        }
    }
    fn scale(&self, n: usize, kb: usize) -> f32 { e8m0(self.scale[n * (self.k / GROUP) + kb]) }
}

/// `C[m, n] = A @ W^T` over the listed K blocks, in order, as the f32 accumulator (no bf16 rounding).
/// `kblocks = None` means all blocks ascending (the reference). Multi-threaded over output rows.
pub fn qgemm_f32(a: &ActQ, w: &dyn QRows, kblocks: Option<&[usize]>) -> Vec<f32> {
    let (m, n, k) = (a.m, w.n(), w.k());
    assert_eq!(a.k, k);
    let nb = k / GROUP;
    let all: Vec<usize>;
    let kbs: &[usize] = match kblocks { Some(l) => l, None => { all = (0..nb).collect(); &all } };
    let mut out = vec![0f32; m * n];
    let threads = std::thread::available_parallelism().map_or(4, |x| x.get()).min(16).min(n.max(1));
    let per = (n + threads - 1) / threads;
    let parts: Vec<(usize, Vec<f32>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads).map(|t| {
            let (n0, n1) = (t * per, ((t + 1) * per).min(n));
            sc.spawn(move || {
                let mut res = vec![0f32; m * (n1.saturating_sub(n0))];
                let mut row = vec![0f32; k];
                for nn in n0..n1 {
                    w.decode_row(nn, &mut row);
                    for mm in 0..m {
                        let aq = &a.q[mm * k..(mm + 1) * k];
                        let mut acc = 0f32;
                        for &kb in kbs {
                            let mut local = 0f64;
                            let b0 = kb * GROUP;
                            for j in 0..GROUP { local += aq[b0 + j] as f64 * row[b0 + j] as f64; }
                            // both scales are powers of two: the products below are exact
                            acc += local as f32 * a.s[mm * nb + kb] * w.scale(nn, kb);
                        }
                        res[(nn - n0) * m + mm] = acc;
                    }
                }
                (n0, res)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (n0, res) in parts {
        let cnt = res.len() / m.max(1);
        for j in 0..cnt { for mm in 0..m { out[mm * n + n0 + j] = res[j * m + mm]; } }
    }
    out
}

/// The same GEMM summed in f64 over all K blocks (the "true" value the f32 accumulator approximates); `[m, n]`.
pub fn qgemm_f64(a: &ActQ, w: &dyn QRows) -> Vec<f64> {
    let (m, n, k) = (a.m, w.n(), w.k());
    let nb = k / GROUP;
    let threads = std::thread::available_parallelism().map_or(4, |x| x.get()).min(16).min(n.max(1));
    let per = (n + threads - 1) / threads;
    let parts: Vec<(usize, Vec<f64>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads).map(|t| {
            let (n0, n1) = (t * per, ((t + 1) * per).min(n));
            sc.spawn(move || {
                let mut res = vec![0f64; m * n1.saturating_sub(n0)];
                let mut row = vec![0f32; k];
                for nn in n0..n1 {
                    w.decode_row(nn, &mut row);
                    for mm in 0..m {
                        let mut acc = 0f64;
                        for kb in 0..nb {
                            let mut local = 0f64;
                            for j in kb * GROUP..(kb + 1) * GROUP { local += a.q[mm * k + j] as f64 * row[j] as f64; }
                            acc += local * (a.s[mm * nb + kb] * w.scale(nn, kb)) as f64;
                        }
                        res[(nn - n0) * m + mm] = acc;
                    }
                }
                (n0, res)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out = vec![0f64; m * n];
    for (n0, res) in parts {
        let cnt = res.len() / m.max(1);
        for j in 0..cnt { for mm in 0..m { out[mm * n + n0 + j] = res[j * m + mm]; } }
    }
    out
}

/// The full linear: quantise `x` (bf16-exact f32, `m` rows) and run the reference GEMM; output bf16-rounded.
pub fn qlinear_bf16(x: &[f32], m: usize, w: &dyn QRows) -> Vec<f32> {
    let a = act_quant(x, m, w.k());
    qgemm_f32(&a, w, None).into_iter().map(bf16r).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_roundtrip_and_ties() {
        let t = e4m3_table();
        for c in 0..255usize {
            if c == 0x7f || c == 0xff { continue; }
            // negative zero decodes to -0.0 and re-encodes to its own code
            assert_eq!(e4m3_encode(t[c]) as usize, c, "code {c:#x}");
        }
        assert_eq!(e4m3_encode(1000.0), 126);
        assert_eq!(e4m3_encode(-1000.0), 0xfe);
        // 1.0625 is the midpoint between 1.0 (code 0x38, even) and 1.125 (0x39): ties to even
        assert_eq!(e4m3_encode(1.0625), 0x38);
        assert_eq!(e4m3_encode(1.1875), 0x3a); // midpoint of 1.125 (odd) and 1.25 (even)
        assert_eq!(e4m3_encode(1e-9), 0);
    }

    #[test]
    fn pow2ceil_matches_exact_powers() {
        assert_eq!(pow2ceil_bits(1.0), 1.0);
        assert_eq!(pow2ceil_bits(1.0000001), 2.0);
        assert_eq!(pow2ceil_bits(0.3), 0.5);
        assert_eq!(pow2ceil_bits(0.25), 0.25);
    }

    #[test]
    fn act_quant_floor_and_range() {
        let x = vec![0.0f32; 32];
        let a = act_quant(&x, 1, 32);
        assert!(a.q.iter().all(|&v| v == 0.0));
        // the 1e-4 floor decides the scale of an all-zero group
        assert_eq!(a.s[0], pow2ceil_bits(1e-4f32 * (1.0f32 / 448.0f32)));
        let mut y = vec![0.1f32; 32];
        y[3] = 448.0 * 8.0;
        let b = act_quant(&y, 1, 32);
        assert_eq!(b.s[0], 8.0);
        assert_eq!(b.q[3], 448.0);
    }

    #[test]
    fn fp4_decode_low_nibble_first() {
        let w = Fp4W { n: 1, k: 32, packed: (0..16u8).map(|i| (i & 7) | (((i + 1) & 7) << 4)).collect(), scale: vec![127; 1] };
        let mut o = vec![0f32; 32];
        w.decode_row(0, &mut o);
        assert_eq!(&o[0..4], &[0.0, 0.5, 0.5, 1.0]);
    }

    #[test]
    fn gemm_matches_a_direct_dequantised_dot() {
        // 2 rows x 64 K, 3 output rows; compare the blockwise accumulator with an f64 reference on dequantised values
        let k = 64;
        let x: Vec<f32> = (0..2 * k).map(|i| bf16r(((i * 37 % 101) as f32 - 50.0) / 13.0)).collect();
        let codes: Vec<u8> = (0..3 * k).map(|i| ((i * 7 + 3) % 120) as u8).collect();
        let w = Fp8W { n: 3, k, codes, scale: vec![126, 127] };
        // scales: [ceil(3/32)=1, k/32=2] = 2 bytes
        let a = act_quant(&x, 2, k);
        let c = qgemm_f32(&a, &w, None);
        let t = e4m3_table();
        for mm in 0..2 { for n in 0..3 {
            let mut r = 0f64;
            for j in 0..k { r += (a.q[mm * k + j] * a.s[mm * 2 + j / 32]) as f64 * (t[w.codes[n * k + j] as usize] * w.scale(n, j / 32)) as f64; }
            assert!((c[mm * 3 + n] as f64 - r).abs() <= r.abs() * 1e-6 + 1e-9, "{} vs {}", c[mm * 3 + n], r);
        } }
    }
}
