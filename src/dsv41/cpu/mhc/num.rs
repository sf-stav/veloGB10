//! DSV4.1 OM oracle: number formats and comparison statistics shared by the mHC / norm / router
//! oracles (leg OM).
//!
//! The reference keeps activations in bf16 and computes the mixers in fp32. The oracle represents a
//! bf16 tensor as `f32` values that are exactly bf16 (every producer here ends with [`bf16_round`]),
//! so a comparison against a dump is a comparison of bf16 values.

/// fp32 -> bf16 bits, round to nearest even (torch's `.to(torch.bfloat16)`); NaN stays a quiet NaN.
pub fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x0040) as u16;
    }
    let round = 0x7FFF + ((b >> 16) & 1);
    (b.wrapping_add(round) >> 16) as u16
}

/// bf16 bits -> f32 (exact).
pub fn bf16_to_f32(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }

/// Round an fp32 value to the nearest bf16 value (RNE), returned as f32.
pub fn bf16_round(x: f32) -> f32 { bf16_to_f32(bf16_bits(x)) }

/// f16 bits -> f32 (exact).
pub fn f16_to_f32(b: u16) -> f32 { half::f16::from_bits(b).to_f32() }

/// Sum of `a[i] * b[i]` with f64 accumulation (each product of two f32 is exact in f64). Used where
/// the reference sums in an order we do not know (cuBLAS GEMMs, torch block reductions): the f64
/// sum rounded once to f32 is within the reference's own fp32 rounding of the true value.
pub fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = [0f64; 4];
    let n4 = a.len() / 4 * 4;
    let mut i = 0;
    while i < n4 {
        s[0] += a[i] as f64 * b[i] as f64;
        s[1] += a[i + 1] as f64 * b[i + 1] as f64;
        s[2] += a[i + 2] as f64 * b[i + 2] as f64;
        s[3] += a[i + 3] as f64 * b[i + 3] as f64;
        i += 4;
    }
    for j in n4..a.len() {
        s[0] += a[j] as f64 * b[j] as f64;
    }
    (s[0] + s[1]) + (s[2] + s[3])
}

/// `x.square().sum()` of the reference: fp32 squares (each rounded, as torch materialises them),
/// summed with f64 accumulation, rounded once.
pub fn sum_sq_f32(x: &[f32]) -> f32 {
    let mut s = 0f64;
    for &v in x {
        s += (v * v) as f64;
    }
    s as f32
}

/// `x.square().mean(-1)` as torch computes it: the sum times the fp32 factor `1 / n`
/// (torch's `MeanOps` multiplies by `float(1) / n`; the sum order inside the GPU reduction is not
/// reproduced, so this is within about one fp32 ulp of the reference).
pub fn mean_sq_f32(x: &[f32]) -> f32 { sum_sq_f32(x) * (1.0f32 / x.len() as f32) }

/// `torch.rsqrt(v)`: computed correctly rounded here; the GPU's `rsqrtf` is within 2 ulp of it.
pub fn rsqrt_f32(v: f32) -> f32 { (1.0f64 / (v as f64).sqrt()) as f32 }

/// Monotone integer image of an f32 (for ulp distances; -0 and +0 map to the same point).
fn ordered(x: f32) -> i64 {
    let b = x.to_bits() as i32;
    if b < 0 { -((b & 0x7FFF_FFFF) as i64) } else { b as i64 }
}

/// Distance in fp32 ulps.
pub fn f32_ulps(a: f32, b: f32) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    (ordered(a) - ordered(b)).unsigned_abs()
}

/// Distance in bf16 ulps between two bf16-exact values.
pub fn bf16_ulps(a: f32, b: f32) -> u64 { f32_ulps(a, b) >> 16 }

/// Relative L2 distance |a - b| / |b| (b = the reference), in f64.
pub fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let d = *x as f64 - *y as f64;
        num += d * d;
        den += (*y as f64) * (*y as f64);
    }
    if den == 0.0 { num.sqrt() } else { (num / den).sqrt() }
}

/// Comparison statistics of one point (accumulates over rows / layers).
#[derive(Clone, Debug, Default)]
pub struct Cmp {
    /// elements compared
    pub n: u64,
    /// elements that are not bitwise equal
    pub mismatch: u64,
    /// largest distance in ulps (bf16 ulps for bf16 points, fp32 ulps for fp32 points)
    pub max_ulps: u64,
    /// largest absolute difference
    pub max_abs: f64,
    /// sum of squared differences and of squared reference values (for the rel-L2)
    pub sq_diff: f64,
    pub sq_ref: f64,
    /// the element with the largest ulp distance: (ours, reference, largest |reference| of the compared rows)
    pub worst: (f32, f32),
}

impl Cmp {
    fn add_one(&mut self, ours: f32, reference: f32, ulps: u64) {
        self.n += 1;
        if ours.to_bits() != reference.to_bits() && !(ours == 0.0 && reference == 0.0) {
            self.mismatch += 1;
        }
        if ulps > self.max_ulps {
            self.max_ulps = ulps;
            self.worst = (ours, reference);
        }
        let d = (ours as f64 - reference as f64).abs();
        if d > self.max_abs || d.is_nan() {
            self.max_abs = d;
        }
        self.sq_diff += d * d;
        self.sq_ref += (reference as f64) * (reference as f64);
    }
    /// Add a bf16 comparison (both sides bf16-exact).
    pub fn bf16(&mut self, ours: &[f32], reference: &[f32]) {
        assert_eq!(ours.len(), reference.len(), "bf16 compare: length");
        for (a, b) in ours.iter().zip(reference) {
            self.add_one(*a, *b, bf16_ulps(*a, *b));
        }
    }
    /// Add an fp32 comparison.
    pub fn f32(&mut self, ours: &[f32], reference: &[f32]) {
        assert_eq!(ours.len(), reference.len(), "f32 compare: length");
        for (a, b) in ours.iter().zip(reference) {
            self.add_one(*a, *b, f32_ulps(*a, *b));
        }
    }
    pub fn merge(&mut self, o: &Cmp) {
        self.n += o.n;
        self.mismatch += o.mismatch;
        if o.max_ulps > self.max_ulps {
            self.max_ulps = o.max_ulps;
            self.worst = o.worst;
        }
        if o.max_abs > self.max_abs || o.max_abs.is_nan() {
            self.max_abs = o.max_abs;
        }
        self.sq_diff += o.sq_diff;
        self.sq_ref += o.sq_ref;
    }
    pub fn rel_l2(&self) -> f64 { if self.sq_ref == 0.0 { self.sq_diff.sqrt() } else { (self.sq_diff / self.sq_ref).sqrt() } }
    /// Largest absolute difference over the RMS of the reference values (a 1-ulp bf16 flip of a
    /// typical element is ~4e-3; a large ulp distance on an element far below the RMS shows here
    /// as a small number).
    pub fn max_abs_over_rms(&self) -> f64 { if self.sq_ref == 0.0 || self.n == 0 { self.max_abs } else { self.max_abs / (self.sq_ref / self.n as f64).sqrt() } }
    pub fn mismatch_frac(&self) -> f64 { if self.n == 0 { 0.0 } else { self.mismatch as f64 / self.n as f64 } }
    /// One cell for a table: "bitwise" or "k/n differ (frac), max u ulp, rel-L2 r".
    pub fn cell(&self) -> String {
        if self.n == 0 {
            return "-".into();
        }
        if self.mismatch == 0 {
            return format!("bitwise ({})", self.n);
        }
        format!("{}/{} ({:.1e}), max {} ulp ({:.3e} vs {:.3e}), max abs {:.1e} = {:.1e} rms, rel {:.1e}", self.mismatch, self.n, self.mismatch_frac(), self.max_ulps, self.worst.0, self.worst.1, self.max_abs, self.max_abs_over_rms(), self.rel_l2())
    }
}
