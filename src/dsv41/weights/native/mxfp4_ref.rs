//! DSV4.1 native MXFP4 experts (leg KN2): the host model of the reference arithmetic, used by the
//! unit tests and `--probe-dsv41-kernel --dsv41-kernel-set mxfp4` as the oracle the kernels must
//! equal BIT FOR BIT.
//!
//! - `act_quant` (kernel.py 22-124, REF_MAP §2.2): amax floored at 1e-4, the scale is the smallest
//!   power of two >= amax * fp32(1/448) by the exponent bits, e4m3 round-to-nearest-even satfinite.
//! - `fp4_gemm` (kernel.py 478-591, REF_MAP §2.5): per 32-wide k-block the tensor-core dot, then
//!   `acc += dot * 2^(sa + sb - 254)` in FP32, k ascending, bf16 at the end. The dot of 32 exact
//!   products is ROUNDED TOWARD ZERO to f32 when it needs more than 24 bits: measured on GB10
//!   (KN2 probe, 2026-11-18, 512K dots, 136K of them not representable) for TileLang's own
//!   e4m3 x e4m3 instruction AND the block-scaled e2m1 x e4m3 one the kernels use — the two agree
//!   on every dot. An RNE model differs from both in ~1 dot in 4 of those.
//! - the Expert's SwiGLU (model.py 845-853, REF_MAP §4 items 13-14) and the MoE sum (REF_MAP §4
//!   item 12): per-expert bf16, FP32 sum in ascending expert id.
//!
//! The one place this model can differ from the GPU is `expf` (host libm vs CUDA's): a 1-ulp
//! difference that crosses a bf16 rounding boundary of h. The probe counts it separately.

use half::bf16;

/// e4m3 value x 2^9 as an exact integer.
#[inline]
pub fn e4m3_int(c: u8) -> i64 {
    let (s, e, m) = (c >> 7, (c >> 3) & 15, (c & 7) as i64);
    let v = if e == 0 { m } else { (8 + m) << (e - 1) };
    if s == 1 { -v } else { v }
}
/// e2m1 value x 2 as an exact integer.
#[inline]
pub fn e2m1_int(c: u8) -> i64 {
    let (e, m) = ((c >> 1) & 3, (c & 1) as i64);
    let v = if e == 0 { m } else { (2 + m) << (e - 1) };
    if c & 8 != 0 { -v } else { v }
}
#[inline]
pub fn bf16r(v: f32) -> f32 { bf16::from_f32(v).to_f32() }

/// f32 -> e4m3fn, round to nearest even, saturating to +-448 (cvt.rn.satfinite.e4m3x2.f32).
pub fn e4m3_rne(x: f32) -> u8 {
    if x.is_nan() { return 0x7F; }
    let s = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs() as f64;
    if a >= 448.0 { return s | 0x7E; }
    let rne = |v: f64| -> i64 { let f = v.floor(); let d = v - f; if d > 0.5 || (d == 0.5 && (f as i64) % 2 == 1) { f as i64 + 1 } else { f as i64 } };
    if a < 2f64.powi(-6) {
        let m = rne(a / 2f64.powi(-9));   // 0..8; 8 is the smallest normal (code 0x08)
        return s | m as u8;
    }
    let mut e = a.log2().floor() as i32;
    if 2f64.powi(e) > a { e -= 1; }
    if 2f64.powi(e + 1) <= a { e += 1; }
    let mut m = rne((a / 2f64.powi(e) - 1.0) * 8.0);
    if m == 8 { m = 0; e += 1; }
    if e > 8 || (e == 8 && m > 6) { return s | 0x7E; }
    s | (((e + 7) as u8) << 3) | m as u8
}

/// act_quant's power-of-two scale (kernel.py fast_round_scale), as a biased exponent byte.
pub fn pow2_scale_e8m0(amax: f32) -> u8 {
    let v = amax.max(1e-4) * (1.0f32 / 448.0f32);
    let b = v.to_bits();
    let e = ((b >> 23) & 0xFF) as i32 - 127 + if b & 0x7F_FFFF != 0 { 1 } else { 0 };
    (e + 127) as u8
}
#[inline]
pub fn e8m0_f(e: u8) -> f32 { f32::from_bits((e as u32) << 23) }

/// act_quant of one row (bf16-valued f32s, len % 32 == 0) -> e4m3 codes and ue8m0 scales.
pub fn act_quant(x: &[f32], q: &mut [u8], s: &mut [u8]) {
    for (g, chunk) in x.chunks_exact(32).enumerate() {
        let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let e = pow2_scale_e8m0(amax);
        let sc = e8m0_f(e);
        for (i, v) in chunk.iter().enumerate() { q[g * 32 + i] = e4m3_rne((v / sc).clamp(-448.0, 448.0)); }
        s[g] = e;
    }
}

/// One k-block's dot as GB10's tensor core returns it: the exact sum (units of 2^-10) rounded
/// toward zero to f32, times 2^(sa + sb - 254) (exact).
#[inline]
pub fn dot_rz(sum: i64, sa: u8, sb: u8) -> f32 {
    let ex = sum as f64;   // exact: |sum| < 2^31
    let mut f = ex as f32;
    if (f as f64).abs() > ex.abs() { f = f32::from_bits(f.to_bits() - 1); }
    (f as f64 * 2f64.powi(sa as i32 + sb as i32 - 254 - 10)) as f32
}

/// The reference fp4_gemm for one (weight row slice, activation row): `w` = the row's packed
/// codes from k-block 0 of the slice, `ws` its scales, `xq` / `xs` the activation codes / scales.
pub fn fp4_dot(w: &[u8], ws: &[u8], xq: &[u8], xs: &[u8], nkb: usize) -> f32 {
    let mut acc = 0f32;
    for kb in 0..nkb {
        let mut sum = 0i64;
        for k in 0..32 {
            let b = w[kb * 16 + k / 2];
            let c = if k & 1 == 1 { b >> 4 } else { b & 15 };
            sum += e2m1_int(c) * e4m3_int(xq[kb * 32 + k]);
        }
        acc += dot_rz(sum, ws[kb], xs[kb]);
    }
    acc
}

/// The Expert's SwiGLU on fp4_gemm's bf16 outputs, with the routing weight, rounded to bf16.
pub fn swiglu_h(gate_bf: f32, up_bf: f32, w: f32, limit: f32) -> f32 {
    let u = if up_bf < -limit { -limit } else if up_bf > limit { limit } else { up_bf };
    let g = if gate_bf > limit { limit } else { gate_bf };
    let s = g / (1.0 + (-g).exp());
    bf16r(w * (s * u))
}

/// One raw expert (checkpoint layout, full matrices).
pub struct RawExpert<'a> { pub w1: &'a [u8], pub s1: &'a [u8], pub w3: &'a [u8], pub s3: &'a [u8], pub w2: &'a [u8], pub s2: &'a [u8] }

/// The intermediate results of one route, for a GPU-vs-host comparison step by step.
pub struct RouteRef { pub gate: Vec<f32>, pub up: Vec<f32>, pub hq: Vec<u8>, pub hs: Vec<u8> }

/// FC1 of one route on this rank's slice: gate / up (bf16 values), then h and its act_quant.
pub fn fc1(e: &RawExpert, dim: usize, i0: usize, ns: usize, xq: &[u8], xs: &[u8], w: f32, limit: f32) -> RouteRef {
    let nkb = dim / 32;
    let (mut gate, mut up, mut h) = (vec![0f32; ns], vec![0f32; ns], vec![0f32; ns]);
    for i in 0..ns {
        let r = i0 + i;
        gate[i] = bf16r(fp4_dot(&e.w1[r * dim / 2..], &e.s1[r * nkb..], xq, xs, nkb));
        up[i] = bf16r(fp4_dot(&e.w3[r * dim / 2..], &e.s3[r * nkb..], xq, xs, nkb));
        h[i] = swiglu_h(gate[i], up[i], w, limit);
    }
    let (mut hq, mut hs) = (vec![0u8; ns], vec![0u8; ns / 32]);
    act_quant(&h, &mut hq, &mut hs);
    RouteRef { gate, up, hq, hs }
}

/// FC2 of one route on this rank's slice from its h codes: output column o of w2 over k in
/// `[i0, i0 + ns)`; rounded to bf16 when the rank holds the whole expert.
pub fn fc2(e: &RawExpert, dim: usize, inter: usize, i0: usize, ns: usize, hq: &[u8], hs: &[u8], round_bf16: bool) -> Vec<f32> {
    let (nkb_full, nkb) = (inter / 32, ns / 32);
    (0..dim).map(|o| {
        let v = fp4_dot(&e.w2[o * inter / 2 + i0 / 2..], &e.s2[o * nkb_full + i0 / 32..], hq, hs, nkb);
        if round_bf16 { bf16r(v) } else { v }
    }).collect()
}

/// The ordered combine: y[o] = sum over the row's routes in ascending expert id of out[route][o].
pub fn combine(ids: &[i32], outs: &[&[f32]], dim: usize) -> Vec<f32> {
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by_key(|&k| ids[k]);
    let mut y = vec![0f32; dim];
    for k in order { for o in 0..dim { y[o] += outs[k][o]; } }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv41_mx_value_codes_and_rounding() {
        for c in 0..=255u8 {
            if c & 0x7F == 0x7F { continue; }
            let v = crate::dsv41::gpu::abi::e4m3(c);
            assert_eq!(e4m3_int(c) as f64, v as f64 * 512.0, "e4m3 {c:#x}");
            assert_eq!(e4m3_rne(v), if c == 0x80 { 0x80 } else { c }, "e4m3 round trip {c:#x}");
        }
        for n in 0..16u8 { assert_eq!(e2m1_int(n) as f32, crate::dsv41::gpu::abi::e2m1(n) * 2.0); }
        // ties to even: 1 + 1/16 sits between 0x38 (1.0) and 0x39 (1.125) -> even mantissa 0x38
        assert_eq!(e4m3_rne(1.0625), 0x38);
        assert_eq!(e4m3_rne(1.1875), 0x3A);
        assert_eq!(e4m3_rne(500.0), 0x7E);
        assert_eq!(e4m3_rne(-3.0 * 2f32.powi(-10)), 0x82, "subnormal tie 1.5 -> 2");
        // act_quant's scale: amax 448 -> 2^0; just above -> 2^1; the 1e-4 floor
        assert_eq!(pow2_scale_e8m0(448.0), 127);
        assert_eq!(pow2_scale_e8m0(448.5), 128);
        assert_eq!(pow2_scale_e8m0(0.0), pow2_scale_e8m0(1e-4));
    }

    #[test]
    fn dsv41_mx_dot_rounds_toward_zero_past_24_bits() {
        let big = (1i64 << 25) + 1;   // needs 26 bits
        assert_eq!(dot_rz(big, 127, 127 + 10), (1i64 << 25) as f32);
        assert_eq!(dot_rz(-big, 127, 127 + 10), -((1i64 << 25) as f32));
        assert_eq!(dot_rz(12345, 127, 127 + 10), 12345.0);
    }
}
