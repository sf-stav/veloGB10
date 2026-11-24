//! CPU reference of the pack-declared expert ACTIVATION quantisers (`hf_quant_config.json`
//! `required_activation_quantization`), statement for statement after the external reference's device helpers
//! (SGLang `jit/csrc/inkling2/routed_quant.cuh`: `quantize_part`, `part_max`, `block_max_reduce`, `mxfp8_half_block`):
//!
//!   * NVFP4 (layers 0-19): `nvfp4_e2m1`, block 16, `float8_e4m3fn` block scales, per-token amax, scale selection
//!     `fouroversix_mae`: per block, quantise with the block scale for a max code of 6 AND of 4, keep the one with the
//!     smaller sum of absolute dequantisation errors (pairwise sum order, 4 strictly smaller wins).
//!     outer = amax / 1536 (the tensor-level decode scale), encoder = 1536 / amax.
//!   * MXFP8 (layers 20-23): `mxfp8_e4m3`, block 32, UE8M0 scale = the smallest power of two >= amax / 448
//!     (exponent bits rounded up), values clamped to +-448, RNE.
//!
//! The inputs are the bf16 values the reference quantises (its MoE input is bf16; the GPU path rounds our fp32 MoE
//! input to bf16 first). Every f32 operation here is the IEEE round-to-nearest-even op of the device code
//! (`__fmul_rn`, `__fdiv_rn`, `fminf`); the e4m3 / e2m1 conversions are exact RNE + satfinite models of
//! `cvt.rn.satfinite.e4m3x2.f32` / `cvt.rn.satfinite.e2m1x2.f32`. Bit-exactness vs the reference's kernels is gated
//! by `tests` below on fixtures the reference itself produced (tools/inkling2/actq_fixtures.py).

/// e2m1 magnitudes by code (bit 3 = sign).
pub const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn rne(x: f64) -> f64 { x.round_ties_even() }

/// `cvt.rn.satfinite.e4m3x2.f32` for one value: (bits, value). NaN -> 0x7F; |v| >= 448 (incl. inf) -> 448.
pub fn e4m3_rn(v: f32) -> (u8, f32) {
    if v.is_nan() { return (0x7F, f32::NAN); }
    let s: u8 = if v.is_sign_negative() { 0x80 } else { 0 };
    let a = (v as f64).abs();
    let code: u8 = if a >= 448.0 {
        0x7E
    } else if a < 0.015625 {
        // subnormal range: m * 2^-9 (m = 8 is the smallest normal 2^-6 = code 0x08)
        rne(a * 512.0) as u8
    } else {
        let mut e = a.log2().floor() as i32;
        // guard the log2 rounding at exact powers of two
        if 2f64.powi(e) > a { e -= 1; }
        if 2f64.powi(e + 1) <= a { e += 1; }
        let m = a / 2f64.powi(e);
        let mut mant = rne((m - 1.0) * 8.0) as i32;
        let mut ee = e;
        if mant == 8 { mant = 0; ee += 1; }
        let c = ((ee + 7) << 3) | mant;
        if c >= 0x7F { 0x7E } else { c as u8 }
    };
    let bits = s | code;
    (bits, e4m3_val(bits))
}

/// The value of e4m3 bits (0x7F / 0xFF = NaN).
pub fn e4m3_val(b: u8) -> f32 {
    let (s, e, m) = ((b >> 7) & 1, ((b >> 3) & 0xF) as i32, (b & 7) as i32);
    if e == 15 && m == 7 { return f32::NAN; }
    let mag = if e == 0 { m as f64 * 2f64.powi(-9) } else { (1.0 + m as f64 / 8.0) * 2f64.powi(e - 7) };
    (if s == 1 { -mag } else { mag }) as f32
}

/// `cvt.rn.satfinite.e2m1x2.f32` for one value: (4-bit code, value). |v| >= 6 -> 6.
pub fn e2m1_rn(v: f32) -> (u8, f32) {
    let s: u8 = if v.is_sign_negative() { 8 } else { 0 };
    let a = (v as f64).abs();
    let code: u8 = if a.is_nan() {
        7 // not reached for the reference's finite inputs
    } else if a < 1.0 {
        rne(a * 2.0) as u8 // 0 -> 0, 1 -> 0.5, 2 -> 1.0 (= code 2)
    } else if a >= 6.0 {
        7
    } else {
        let mut e = a.log2().floor() as i32;
        if 2f64.powi(e) > a { e -= 1; }
        if 2f64.powi(e + 1) <= a { e += 1; }
        let m = a / 2f64.powi(e);
        let mut mant = rne((m - 1.0) * 2.0) as i32;
        let mut ee = e;
        if mant == 2 { mant = 0; ee += 1; }
        (((ee + 1) << 1) | mant).min(7) as u8
    };
    let val = E2M1[code as usize];
    (s | code, if s != 0 { -val } else { val })
}

/// bf16 rounding (RNE) of an f32, as `__float2bfloat16_rn`.
pub fn bf16_rn(x: f32) -> f32 {
    let b = x.to_bits();
    if x.is_nan() { return x; }
    let r = b.wrapping_add(0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(r)
}

const FLT_MAX: f32 = f32::MAX;

/// One 16-value NVFP4 block (`quantize_part`): (8 packed bytes, low nibble = even element; e4m3 scale bits).
pub fn nv4_block(x: &[f32], amax: f32, block_max: f32) -> ([u8; 8], u8) {
    debug_assert_eq!(x.len(), 16);
    let encoder = (1536.0f32 / amax).min(FLT_MAX);
    let decoder = 1.0f32 / encoder;
    let outer = amax * ((1.0f64 / 1536.0) as f32);
    let (bits6, scale6) = e4m3_rn((block_max * (encoder * ((1.0f64 / 6.0) as f32))).min(448.0));
    let (bits4, scale4) = e4m3_rn((block_max * (encoder * 0.25f32)).min(448.0));
    let inverse6 = (1.0f32 / (scale6 * decoder)).min(FLT_MAX);
    let inverse4 = (1.0f32 / (scale4 * decoder)).min(FLT_MAX);
    let mut p6 = [0u8; 8];
    let mut p4 = [0u8; 8];
    let mut e6 = [0f32; 16];
    let mut e4 = [0f32; 16];
    for i in 0..16 {
        let (c6, r6) = e2m1_rn(x[i] * inverse6);
        let (c4, r4) = e2m1_rn(x[i] * inverse4);
        p6[i / 2] |= c6 << (4 * (i & 1));
        p4[i / 2] |= c4 << (4 * (i & 1));
        e6[i] = ((r6 * scale6) * outer - x[i]).abs();
        e4[i] = ((r4 * scale4) * outer - x[i]).abs();
    }
    // pairwise tree: ((0+1)+(2+3)) per 4-lane part, then xor-1 / xor-2 shuffles = a full pairwise tree over 16
    let tree = |e: &[f32; 16]| -> f32 {
        let mut v = *e;
        let mut w = 16;
        while w > 1 {
            for i in 0..w / 2 { v[i] = v[2 * i] + v[2 * i + 1]; }
            w /= 2;
        }
        v[0]
    };
    if tree(&e4) < tree(&e6) { (p4, bits4) } else { (p6, bits6) }
}

/// NVFP4 of one bf16 row (`x` already bf16-valued): (codes [k/2], e4m3 scale bits [k/16], amax).
pub fn nv4_row(x: &[f32]) -> (Vec<u8>, Vec<u8>, f32) {
    assert!(x.len() % 16 == 0);
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let mut codes = vec![0u8; x.len() / 2];
    let mut sc = vec![0u8; x.len() / 16];
    for b in 0..x.len() / 16 {
        let blk = &x[16 * b..16 * b + 16];
        let bm = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let (p, s) = nv4_block(blk, amax, bm);
        codes[8 * b..8 * b + 8].copy_from_slice(&p);
        sc[b] = s;
    }
    (codes, sc, amax)
}

/// Dequantised NVFP4 row value j (without the outer amax / 1536): e2m1 * e4m3.
pub fn nv4_deq(codes: &[u8], sc: &[u8], j: usize) -> f32 {
    let c = (codes[j / 2] >> (4 * (j & 1))) & 0xF;
    let v = E2M1[(c & 7) as usize];
    let v = if c & 8 != 0 { -v } else { v };
    v * e4m3_val(sc[j / 16])
}

/// One 32-value MXFP8 block (`mxfp8_half_block` / Triton `_mxfp8_block`): (32 e4m3 codes, ue8m0 exponent).
pub fn mx8_block(x: &[f32]) -> ([u8; 32], u8) {
    debug_assert_eq!(x.len(), 32);
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let min_scale = f32::from_bits(0x0040_0000);
    let raw = amax * ((1.0f64 / 448.0) as f32);
    let mut exponent = ((raw.to_bits().wrapping_add(0x007F_FFFF) & 0x7F80_0000) >> 23) as i32;
    exponent = if raw <= min_scale { 0 } else { exponent.min(254) };
    if raw.is_nan() { exponent = 255; }
    let scale = f32::from_bits((exponent as u32) << 23).max(min_scale);
    let mut inverse = if exponent == 0 { f32::from_bits(0x7F00_0000) } else { 1.0f32 / scale };
    if exponent == 255 { inverse = 0.0; }
    let mut q = [0u8; 32];
    for i in 0..32 {
        let v = (x[i] * inverse).max(-448.0).min(448.0);
        q[i] = e4m3_rn(v).0;
    }
    (q, exponent as u8)
}

/// MXFP8 of one bf16 row: (codes [k], exponents [k/32]).
pub fn mx8_row(x: &[f32]) -> (Vec<u8>, Vec<u8>) {
    assert!(x.len() % 32 == 0);
    let mut c = vec![0u8; x.len()];
    let mut e = vec![0u8; x.len() / 32];
    for b in 0..x.len() / 32 {
        let (q, ex) = mx8_block(&x[32 * b..32 * b + 32]);
        c[32 * b..32 * b + 32].copy_from_slice(&q);
        e[b] = ex;
    }
    (c, e)
}

/// Dequantised MXFP8 value j: e4m3 * 2^(e - 127).
pub fn mx8_deq(codes: &[u8], ex: &[u8], j: usize) -> f32 {
    e4m3_val(codes[j]) * f32::from_bits(((ex[j / 32] as u32) << 23).max(0x0040_0000))
}

// ------------------------------------------------------------------------------------------- test inputs
/// The fixture generator's hash (tools/inkling2/actq_fixtures.py `h32`).
pub fn h32(i: u32, seed: u32) -> u32 {
    let mut x = i.wrapping_mul(0x9E37_79B1).wrapping_add(seed.wrapping_mul(0x85EB_CA77));
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    x
}

/// bf16 bits of element (r, c) of the fixture's [R][k] matrix (`hbits`).
pub fn hbits(r: usize, c: usize, k: usize, seed: u32) -> u16 {
    let hv = h32((r * k + c) as u32, seed);
    if r == 0 { return 0; }
    if r == 1 { return if c == 7 { 0x3F80 } else { 0 }; }
    if hv % 17 == 0 { return 0; }
    let sign = hv >> 31;
    let mut e0 = 120 + (r % 12) as u32;
    if r == 2 && c == 100 { e0 += 12; }
    if r == 3 { e0 = 70; }
    if r == 4 { e0 = 150; }
    let e = e0 + ((hv >> 8) & 7) - 4;
    ((sign << 15) | (e << 7) | (hv & 0x7F)) as u16
}

pub fn bf16_val(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }

/// `h01` of the fixture generator.
pub fn h01(i: u32, seed: u32) -> f32 { ((h32(i, seed) >> 8) as f64 / (1u64 << 24) as f64) as f32 }

pub const FIX_ROWS: usize = 24;

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::SafeTensors;

    fn fixture() -> Option<Vec<u8>> {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/src/inkling2/actq/fixtures/sgl_actq.safetensors");
        std::fs::read(p).ok()
    }
    fn get<'a>(st: &'a SafeTensors<'a>, n: &str) -> &'a [u8] { st.tensor(n).unwrap_or_else(|_| panic!("{n}")).data() }

    #[test]
    fn e4m3_e2m1_rne_tables() {
        // every e4m3 code value maps to itself; midpoints round to even
        for b in 0u8..0x7F {
            let v = e4m3_val(b);
            assert_eq!(e4m3_rn(v).0, b, "e4m3 {b:#x} {v}");
        }
        assert_eq!(e4m3_rn(17.0).0, e4m3_rn(16.0).0); // 17 = midpoint 16 / 18 -> 16 (even mantissa)
        assert_eq!(e4m3_rn(1e9).0, 0x7E);
        for (i, &v) in E2M1.iter().enumerate() { assert_eq!(e2m1_rn(v).0, i as u8); }
        assert_eq!(e2m1_rn(0.25).1, 0.0);
        assert_eq!(e2m1_rn(0.75).1, 1.0);
        assert_eq!(e2m1_rn(1.25).1, 1.0);
        assert_eq!(e2m1_rn(1.75).1, 2.0);
        assert_eq!(e2m1_rn(2.5).1, 2.0);
        assert_eq!(e2m1_rn(3.5).1, 4.0);
        assert_eq!(e2m1_rn(5.0).1, 4.0);
        assert_eq!(e2m1_rn(5.01).1, 6.0);
        assert_eq!(e2m1_rn(100.0).1, 6.0);
        assert_eq!(e2m1_rn(-0.1).0, 8);
    }

    #[test]
    fn quantisers_match_external_reference_bit_exact() {
        let Some(raw) = fixture() else { return; };
        let st = SafeTensors::deserialize(&raw).unwrap();
        for (k, seed) in [(2304usize, 1u32), (1152, 2)] {
            let (mut nd_c, mut nd_s, mut nd_a, mut md_c, mut md_e) = (0, 0, 0, 0, 0);
            let (rc, rs, ra) = (get(&st, &format!("{k}.nv.codes")), get(&st, &format!("{k}.nv.scales")), get(&st, &format!("{k}.nv.amax")));
            let (mc, me) = (get(&st, &format!("{k}.mx.codes")), get(&st, &format!("{k}.mx.exp")));
            for r in 0..FIX_ROWS {
                let x: Vec<f32> = (0..k).map(|c| bf16_val(hbits(r, c, k, seed))).collect();
                let (c, s, a) = nv4_row(&x);
                nd_c += c.iter().zip(&rc[r * k / 2..(r + 1) * k / 2]).filter(|(a, b)| a != b).count();
                nd_s += s.iter().zip(&rs[r * k / 16..(r + 1) * k / 16]).filter(|(a, b)| a != b).count();
                let ra_r = f32::from_le_bytes(ra[4 * r..4 * r + 4].try_into().unwrap());
                nd_a += (ra_r.to_bits() != a.to_bits()) as usize;
                let (c, e) = mx8_row(&x);
                md_c += c.iter().zip(&mc[r * k..(r + 1) * k]).filter(|(a, b)| a != b).count();
                md_e += e.iter().zip(&me[r * k / 32..(r + 1) * k / 32]).filter(|(a, b)| a != b).count();
            }
            assert_eq!((nd_c, nd_s, nd_a), (0, 0, 0), "nvfp4 width {k}: code / scale / amax mismatches");
            assert_eq!((md_c, md_e), (0, 0), "mxfp8 width {k}: code / exponent mismatches");
        }
    }
}
