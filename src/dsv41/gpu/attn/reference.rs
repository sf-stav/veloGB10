//! CPU references for the sparse attention core (leg KA): the storage codecs (FP8 E4M3 + E8M0/32 rows,
//! the 288-B FP4 E2M1 + E4M3/16 records), an exact f64 attention, an f32 emulation of the reference
//! kernel's own arithmetic (kernel.py sparse_attn: blocks of 64 in the given index order, running max from
//! -1e30, bf16 P before P.V, the sink in the denominator), and the epilogue (inverse RoPE, act_quant) with
//! the device's sincos ported operation for operation (so the epilogue can be checked bit for bit).

use crate::dsv41::gpu::abi;

/// fp32 -> bf16 bits, round to nearest even (NaN kept quiet).
pub fn bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    if x.is_nan() { return ((u >> 16) | 0x40) as u16; }
    let r = 0x7FFF + ((u >> 16) & 1);
    ((u.wrapping_add(r)) >> 16) as u16
}
pub fn bf16_f32(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }
pub fn bf16_round(x: f32) -> f32 { bf16_f32(bf16_bits(x)) }

/// The 127 non-negative finite E4M3 values in code order (code 0x00..0x7E), ascending.
fn e4m3_table() -> &'static [f32; 127] {
    static T: std::sync::OnceLock<[f32; 127]> = std::sync::OnceLock::new();
    T.get_or_init(|| { let mut t = [0f32; 127]; for (c, v) in t.iter_mut().enumerate() { *v = abi::e4m3(c as u8); } t })
}

/// fp32 -> E4M3 (fn), round to nearest even, saturating to +-448 (torch `.to(float8_e4m3fn)` after the
/// reference's clamp; the device's cvt.rn.satfinite).
pub fn e4m3_encode(x: f32) -> u8 {
    let t = e4m3_table();
    let a = x.abs().min(448.0);
    let i = t.partition_point(|&v| v < a);
    let code = if i >= t.len() { 126 } else if t[i] == a || i == 0 { i } else {
        let (lo, hi) = (t[i - 1], t[i]);
        let (dl, dh) = (a - lo, hi - a);
        if dl < dh { i - 1 } else if dh < dl { i } else if (i - 1) % 2 == 0 { i - 1 } else { i }
    };
    (code as u8) | if x.is_sign_negative() { 0x80 } else { 0 }   // -0 keeps its sign, as cvt.rn.satfinite does
}

/// fp32 -> E2M1 nibble, round to nearest even on {0, .5, 1, 1.5, 2, 3, 4, 6}, saturating to +-6.
pub fn e2m1_encode(x: f32) -> u8 {
    const G: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let a = x.abs().min(6.0);
    let i = G.partition_point(|&v| v < a);
    let code = if i >= 8 { 7 } else if G[i] == a || i == 0 { i } else {
        let (dl, dh) = (a - G[i - 1], G[i] - a);
        if dl < dh { i - 1 } else if dh < dl { i } else if (i - 1) % 2 == 0 { i - 1 } else { i }
    };
    (code as u8) | if x.is_sign_negative() { 8 } else { 0 }
}

/// kernel.py fast_round_scale: the exponent of pow2ceil(max(amax, 1e-4) * fp32(1/448)).
pub fn aq_exp(amax: f32) -> i32 {
    let v = amax.max(1e-4f32) * (1.0f32 / 448.0f32);
    let b = v.to_bits();
    ((b >> 23) & 0xFF) as i32 - 127 + i32::from(b & 0x7F_FFFF != 0)
}
fn pow2(e: i32) -> f32 { f32::from_bits(((e + 127) as u32) << 23) }

/// act_quant (block 32, ue8m0) of a row: (E4M3 codes, E8M0 bytes); the values a row decodes to are
/// e4m3(code) * 2^(byte - 127), exactly the reference's in-place result.
pub fn act_quant_fp8(x: &[f32]) -> (Vec<u8>, Vec<u8>) {
    assert!(x.len() % 32 == 0);
    let mut q = vec![0u8; x.len()];
    let mut s = vec![0u8; x.len() / 32];
    for g in 0..x.len() / 32 {
        let grp = &x[32 * g..32 * g + 32];
        let am = grp.iter().fold(0f32, |m, v| m.max(v.abs()));
        let e = aq_exp(am);
        let sc = pow2(e);
        for (i, v) in grp.iter().enumerate() { q[32 * g + i] = e4m3_encode(v / sc); }
        s[g] = (e + 127) as u8;
    }
    (q, s)
}

/// Decode an FP8 plane row (codes + E8M0 per 32).
pub fn fp8_row_decode(q: &[u8], s: &[u8]) -> Vec<f32> {
    q.iter().enumerate().map(|(i, &c)| abi::e4m3(c) * abi::e8m0(s[i / 32])).collect()
}

/// fp4_act_quant(latent, 16, scale = e4m3): the 288-B record (E2M1 low nibble first, then 32 E4M3 scales).
pub fn kv_record_encode(x: &[f32]) -> [u8; 288] {
    assert_eq!(x.len(), 512);
    let mut r = [0u8; 288];
    for g in 0..32 {
        let grp = &x[16 * g..16 * g + 16];
        let am = grp.iter().fold(0f32, |m, v| m.max(v.abs())).max(6.0 * 2f32.powi(-9));
        let sc8 = e4m3_encode(am / 6.0);
        let sc = abi::e4m3(sc8);
        r[256 + g] = sc8;
        for (i, v) in grp.iter().enumerate() {
            let c = e2m1_encode((v / sc).clamp(-6.0, 6.0));
            let d = 16 * g + i;
            r[d / 2] |= (c & 15) << (4 * (d & 1));
        }
    }
    r
}

/// Decode a 288-B record.
pub fn kv_record_decode(r: &[u8]) -> Vec<f32> {
    (0..512).map(|d| abi::e2m1((r[d / 2] >> (4 * (d & 1))) & 15) * abi::e4m3(r[256 + d / 16])).collect()
}

/// Exact attention of one row (f64): q [heads][512] (bf16 values), keys = the valid keys' exact values,
/// sinks [heads] -> o [heads][512] (unrounded).
pub fn attend_f64(q: &[f32], heads: usize, keys: &[&[f32]], sink: &[f32], scale: f32) -> Vec<f64> {
    let mut out = vec![0f64; heads * 512];
    for h in 0..heads {
        let qh = &q[h * 512..h * 512 + 512];
        let s: Vec<f64> = keys.iter().map(|k| qh.iter().zip(k.iter()).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() * scale as f64).collect();
        let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(sink[h] as f64);
        let den: f64 = s.iter().map(|x| (x - m).exp()).sum::<f64>() + (sink[h] as f64 - m).exp();
        for (j, k) in keys.iter().enumerate() {
            let p = (s[j] - m).exp() / den;
            for d in 0..512 { out[h * 512 + d] += p * k[d] as f64; }
        }
    }
    out
}

/// The reference kernel's arithmetic in f32 (kernel.py sparse_attn as REFb emulates it): the index list
/// in its given order (`None` = a -1 slot), blocks of 64, running max from -1e30, corr = exp(m - m'),
/// sum = fma(sum, corr, rowsum p), acc = acc * corr + bf16(p) . v, then sum += exp(sink - m), bf16(acc / sum).
pub fn attend_ref_layout(q: &[f32], heads: usize, list: &[Option<&[f32]>], sink: &[f32], scale: f32) -> Vec<f32> {
    let mut out = vec![0f32; heads * 512];
    let nb = list.len().div_ceil(64);
    for h in 0..heads {
        let qh = &q[h * 512..h * 512 + 512];
        let (mut m, mut l) = (-1e30f32, 0f32);
        let mut acc = vec![0f32; 512];
        for b in 0..nb {
            let blk = &list[64 * b..(64 * b + 64).min(list.len())];
            let s: Vec<f32> = blk.iter().map(|k| match k {
                Some(k) => (qh.iter().zip(k.iter()).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32) * scale,
                None => f32::NEG_INFINITY,
            }).collect();
            let mn = s.iter().cloned().fold(m, f32::max);
            let corr = (m - mn).exp();
            let p: Vec<f32> = s.iter().map(|x| (x - mn).exp()).collect();
            let ps: f32 = p.iter().sum();
            l = l.mul_add(corr, ps);
            for a in acc.iter_mut() { *a *= corr; }
            for (j, k) in blk.iter().enumerate() {
                if let Some(k) = k {
                    let pb = bf16_round(p[j]);
                    for d in 0..512 { acc[d] += pb * k[d]; }
                }
            }
            m = mn;
        }
        l += (sink[h] - m).exp();
        for d in 0..512 { out[h * 512 + d] = bf16_round(acc[d] / l); }
    }
    out
}

/// The device's d41_sincos (00_abi.cuh; KA's d41a_sincos before the SL RoPE unification), operation for operation (f32 FMA = `mul_add`): libdevice's sincosf fast path
/// below |x| < 105615, a double-precision reduction above.
pub fn sincos_dev(x: f32) -> (f32, f32) {
    let (r, k): (f32, i32) = if x.abs() < 105615.0 {
        let kf = (x * f32::from_bits(0x3F22_F983)).round_ties_even();
        let mut r = kf.mul_add(f32::from_bits(0xBFC9_0FDA), x);
        r = kf.mul_add(f32::from_bits(0xB3A2_2168), r);
        r = kf.mul_add(f32::from_bits(0xA7C2_34C5), r);
        (r, kf as i32)
    } else {
        let xd = x as f64;
        let kd = (xd * 0.63661977236758138).round_ties_even();
        let mut rd = (-kd).mul_add(1.5707963267948966, xd);
        rd = (-kd).mul_add(6.123233995736766e-17, rd);
        (rd as f32, kd as i64 as i32)
    };
    let r2 = r * r;
    let mut pc = r2.mul_add(f32::from_bits(0x37CB_AC00), f32::from_bits(0xBAB6_07ED));
    pc = pc.mul_add(r2, f32::from_bits(0x3D2A_AABB));
    pc = pc.mul_add(r2, f32::from_bits(0xBEFF_FFFF));
    pc = pc.mul_add(r2, 1.0);
    let r3 = r2.mul_add(r, 0.0);
    let mut ps = r2.mul_add(f32::from_bits(0xB94D_4153), f32::from_bits(0x3C08_85E4));
    ps = ps.mul_add(r2, f32::from_bits(0xBE2A_AAA8));
    ps = ps.mul_add(r3, r);
    let (sv, cv) = if k & 1 != 0 { (pc, ps) } else { (ps, pc) };
    (if k & 2 != 0 { -sv } else { sv }, if (k + 1) & 2 != 0 { -cv } else { cv })
}

/// The epilogue on one row's bf16 o [heads][512] (as the kernel's EPI flags): the inverse RoPE on dims
/// 448..511 (rotate by -angle, angle = fp32(pos) * freq, bf16 out) then act_quant per 32 (bf16 dequantised
/// values, E4M3 codes, E8M0 bytes). `rope_on` / `aq_on` select the steps.
pub fn epilogue(o: &[f32], heads: usize, pos: i32, freq: &[f32; 32], rope_on: bool, aq_on: bool) -> (Vec<f32>, Vec<u8>, Vec<u8>) {
    let mut x = o.to_vec();
    if rope_on {
        for h in 0..heads {
            for i in 0..32 {
                let (s, c) = sincos_dev(pos as f32 * freq[i]);
                let d = h * 512 + 448 + 2 * i;
                let (x0, x1) = (x[d], x[d + 1]);
                // torch's conj complex multiply as its CUDA build contracts it (d41_rope_inv, 00_abi.cuh)
                x[d] = bf16_round(x0.mul_add(c, x1 * s));
                x[d + 1] = bf16_round(x1.mul_add(c, -(x0 * s)));
            }
        }
    }
    let mut codes = vec![0u8; x.len()];
    let mut scales = vec![0u8; x.len() / 32];
    if aq_on {
        for g in 0..x.len() / 32 {
            let grp = &mut x[32 * g..32 * g + 32];
            let am = grp.iter().fold(0f32, |m, v| m.max(v.abs()));
            let e = aq_exp(am);
            let sc = pow2(e);
            for (i, v) in grp.iter_mut().enumerate() {
                let c = e4m3_encode(*v / sc);
                codes[32 * g + i] = c;
                *v = abi::e4m3(c) * sc;
            }
            scales[g] = (e + 127) as u8;
        }
    }
    (x, codes, scales)
}

/// rel-L2 = |a - b| / |b|.
pub fn rel_l2(a: &[f32], b: &[f64]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b.iter()) { num += (*x as f64 - y) * (*x as f64 - y); den += y * y; }
    (num / den.max(1e-300)).sqrt()
}
pub fn rel_l2_f32(a: &[f32], b: &[f32]) -> f64 {
    let bb: Vec<f64> = b.iter().map(|v| *v as f64).collect();
    rel_l2(a, &bb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv41_attn_codecs_round_trip() {
        assert_eq!(e4m3_encode(448.0), 0x7E);
        assert_eq!(e4m3_encode(1000.0), 0x7E);
        assert_eq!(e4m3_encode(-1.0), 0xB8);
        assert_eq!(e4m3_encode(0.0), 0);
        assert_eq!(e4m3_encode(-0.0), 0x80);
        for c in 0u8..=0x7E { assert_eq!(e4m3_encode(abi::e4m3(c)), c, "code {c:#x}"); }
        // ties to even: 1.0 + 1/16 sits between 1.0 (0x38) and 1.125 (0x39) -> 0x38
        assert_eq!(e4m3_encode(1.0625), 0x38);
        assert_eq!(e4m3_encode(1.1875), 0x3A);
        for (x, c) in [(0.25f32, 0u8), (0.75, 2), (1.25, 2), (1.75, 4), (2.5, 4), (3.5, 6), (5.0, 6), (7.0, 7), (-0.75, 10)] {
            assert_eq!(e2m1_encode(x), c, "{x}");
        }
        let x: Vec<f32> = (0..512).map(|i| ((i as f32 * 0.37).sin() * 3.0) * if i % 41 == 0 { 20.0 } else { 1.0 }).collect();
        let r = kv_record_encode(&x);
        let y = kv_record_decode(&r);
        // re-encoding the decoded values gives the same values (bytes differ only where an E2M1 "-0" decodes to 0)
        assert_eq!(kv_record_decode(&kv_record_encode(&y)), y, "re-encoding a decoded record keeps its values");
        for g in 0..32 {
            let am = x[16 * g..16 * g + 16].iter().fold(0f32, |m, v| m.max(v.abs()));
            assert!((16 * g..16 * g + 16).all(|i| (y[i] - x[i]).abs() <= am / 5.0), "group {g}: the record approximates its input");
        }
        let (q, s) = act_quant_fp8(&x);
        let z = fp8_row_decode(&q, &s);
        assert_eq!(act_quant_fp8(&z), (q, s), "act_quant of its own output is the identity");
        assert_eq!(aq_exp(448.0), 0);
        assert_eq!(aq_exp(449.0), 1);
        assert_eq!(aq_exp(0.0), aq_exp(1e-4));
    }

    #[test]
    fn dsv41_attn_sincos_port_is_accurate() {
        for &x in &[0.0f32, 0.5, 1.0, 3.14159, 100.0, 1234.5, 99999.0, 105616.0, 524287.0, 1048575.0, -7.25] {
            let (s, c) = sincos_dev(x);
            assert!((s as f64 - (x as f64).sin()).abs() < 2e-7 && (c as f64 - (x as f64).cos()).abs() < 2e-7, "{x}: {s} {c}");
        }
    }

    #[test]
    fn dsv41_attn_reference_layouts_agree() {
        // the f32 block emulation and the f64 exact softmax agree to bf16-P noise
        let heads = 2;
        let q: Vec<f32> = (0..heads * 512).map(|i| bf16_round(((i as f32) * 0.013).cos())).collect();
        let keys: Vec<Vec<f32>> = (0..150).map(|j| (0..512).map(|d| bf16_round(((j * 7 + d) as f32 * 0.01).sin())).collect()).collect();
        let kr: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
        let list: Vec<Option<&[f32]>> = kr.iter().map(|k| Some(*k)).collect();
        let sink = [0.5f32, -1.0];
        let a = attend_ref_layout(&q, heads, &list, &sink, 0.044194173);
        let b = attend_f64(&q, heads, &kr, &sink, 0.044194173);
        assert!(rel_l2(&a, &b) < 5e-3, "{}", rel_l2(&a, &b));
    }
}
