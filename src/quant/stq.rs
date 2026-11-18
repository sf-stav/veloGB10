//! Simulated STQ1_0 / Ternary2 / Ls3 block formats (fake-quant only; the engine serves the baked values unmodified).
//! Self-contained (imports nothing from its siblings); `mod.rs` has the directory map.

// ---------------------------------------------------------------------------------------------
// SQ campaign — STQ1_0 / ternary-2bit / 3-bit-LS quality-simulation quantizers.
//
// Values-only round-trips over 256-weight blocks (STQ1_0's QK_K), after AngelSlim's PTQ encoder
// for the Hy4 STQ1_0 build (`docs/sq_refs/angelslim_stq1_0_quant_cuda.patch`): imatrix-weighted
// least-squares scale `d = Σ(w·q·x)/Σ(w·q²)`, zero placed at the lane of minimum incremental cost
// `w·(x² − (|x|−d)²)`, 3 alternating rounds, `w[j] = qw[j]·sqrt(σ² + x[j]²)` with `σ² = 2Σx²/256`.
// The formats these simulate (the probe bakes the round-trip VALUES into NVFP4 tensors via
// `--stq-bake`; the engine then serves them unmodified — only the error is real):
//   * Stq1_0   1.3125 bpw — fp16 scale/256 + 4-bit slot + 1-bit sign over stride-16 groups of
//              4 lanes with exactly one forced zero per group (3:4 structure).
//   * Ternary2 2.0625 bpw — {−d, 0, +d} with FREE zero placement, fp16 scale/256 (TQ2_0-class).
//   * Ls3Bit   3.0625 bpw — 8 uniform levels ±(2i+1)/2 · d, fp16 scale/256 (IQ3-XXS-class).
// ---------------------------------------------------------------------------------------------

pub const STQ_BLOCK: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StqKind {
    Stq1_0,
    Ternary2,
    Ls3Bit,
}

impl StqKind {
    pub fn bpw(self) -> f32 {
        match self {
            StqKind::Stq1_0 => 1.3125,
            StqKind::Ternary2 => 2.0625,
            StqKind::Ls3Bit => 3.0625,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            StqKind::Stq1_0 => "stq1_0",
            StqKind::Ternary2 => "ternary2",
            StqKind::Ls3Bit => "ls3bit",
        }
    }
}

/// fp16 round-trip: the real formats store the block scale as fp16 — simulate that loss too.
#[inline]
fn round_f16(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

/// The AngelSlim per-block weights: `qw·sqrt(σ² + x²)` (σ² = 2·mean(x²)); unweighted when
/// `qw` is absent.
#[inline]
fn stq_weight(x: f32, sigma2: f32, qw: Option<f32>) -> f32 {
    let base = (sigma2 + x * x).sqrt();
    match qw {
        Some(q) => q * base,
        None => base,
    }
}

/// STQ1_0 block encode→decode. `x`/`out` are 256-long; `qw` (optional) is the matching slice of
/// per-input-channel importance. Verbatim port of `quantize_row_stq1_0_impl` + its decode.
pub fn stq1_0_block(x: &[f32], qw: Option<&[f32]>, out: &mut [f32]) {
    debug_assert_eq!(x.len(), STQ_BLOCK);
    let (mut sumx2, mut amax) = (0.0f32, 0.0f32);
    for &v in x {
        sumx2 += v * v;
        amax = amax.max(v.abs());
    }
    if !(amax > 0.0) {
        out.fill(0.0);
        return;
    }
    let sigma2 = 2.0 * sumx2 / STQ_BLOCK as f32;
    let mut weight = [0.0f32; STQ_BLOCK];
    for j in 0..STQ_BLOCK {
        weight[j] = stq_weight(x[j], sigma2, qw.map(|q| q[j]));
    }
    let mut sel = [0i8; STQ_BLOCK];
    let mut d = amax;
    for _ in 0..3 {
        // Zero placement over the stride-16 groups: group g owns lanes chunk*64 + gloc + p*16.
        for g in 0..STQ_BLOCK / 4 {
            let chunk = g / 16;
            let gloc = g % 16;
            let base = chunk * 64 + gloc;
            let mut zero_pos = 0usize;
            let mut best = f32::INFINITY;
            for p in 0..4 {
                let j = base + p * 16;
                let ax = x[j].abs();
                let cost = weight[j] * (x[j] * x[j] - (ax - d) * (ax - d));
                if cost < best {
                    best = cost;
                    zero_pos = p;
                }
            }
            for p in 0..4 {
                let j = base + p * 16;
                sel[j] = if p == zero_pos { 0 } else if x[j] < 0.0 { -1 } else { 1 };
            }
        }
        let (mut sumqx, mut sumq2) = (0.0f32, 0.0f32);
        for j in 0..STQ_BLOCK {
            let q = sel[j] as f32;
            sumqx += weight[j] * q * x[j];
            sumq2 += weight[j] * q * q;
        }
        if !(sumq2 > 0.0) {
            break;
        }
        let dnew = sumqx / sumq2;
        if !(dnew > 0.0) {
            break;
        }
        let converged = (dnew - d).abs() <= 1e-6 * d;
        d = dnew;
        if converged {
            break;
        }
    }
    let d16 = round_f16(d);
    for j in 0..STQ_BLOCK {
        out[j] = sel[j] as f32 * d16;
    }
}

/// Ternary-2bit block: {−d, 0, +d}, free placement (zero iff |x| < d/2 — the argmin, independent
/// of w since w scales both candidate costs equally), LS scale from the weighted objective.
pub fn ternary2_block(x: &[f32], qw: Option<&[f32]>, out: &mut [f32]) {
    debug_assert_eq!(x.len(), STQ_BLOCK);
    let (mut sumx2, mut amax, mut sumabs) = (0.0f32, 0.0f32, 0.0f32);
    for &v in x {
        sumx2 += v * v;
        sumabs += v.abs();
        amax = amax.max(v.abs());
    }
    if !(amax > 0.0) {
        out.fill(0.0);
        return;
    }
    let sigma2 = 2.0 * sumx2 / STQ_BLOCK as f32;
    let mut weight = [0.0f32; STQ_BLOCK];
    for j in 0..STQ_BLOCK {
        weight[j] = stq_weight(x[j], sigma2, qw.map(|q| q[j]));
    }
    let mut sel = [0.0f32; STQ_BLOCK];
    let mut d = sumabs / STQ_BLOCK as f32; // amax is a poor ternary init; mean|x| is not
    for _ in 0..3 {
        let half_d = 0.5 * d;
        for j in 0..STQ_BLOCK {
            sel[j] = if x[j].abs() < half_d {
                0.0
            } else if x[j] < 0.0 {
                -1.0
            } else {
                1.0
            };
        }
        let (mut sumqx, mut sumq2) = (0.0f32, 0.0f32);
        for j in 0..STQ_BLOCK {
            sumqx += weight[j] * sel[j] * x[j];
            sumq2 += weight[j] * sel[j] * sel[j];
        }
        if !(sumq2 > 0.0) {
            break;
        }
        let dnew = sumqx / sumq2;
        if !(dnew > 0.0) {
            break;
        }
        let converged = (dnew - d).abs() <= 1e-6 * d;
        d = dnew;
        if converged {
            break;
        }
    }
    let d16 = round_f16(d);
    for j in 0..STQ_BLOCK {
        out[j] = sel[j] * d16;
    }
}

/// 3-bit block: 8 uniform levels ±(2i+1)/2 · d (IQ3-XXS-class granularity without its
/// nonuniform codebook). Nearest-level placement is weight-independent; the LS scale is not.
pub fn ls3_block(x: &[f32], qw: Option<&[f32]>, out: &mut [f32]) {
    debug_assert_eq!(x.len(), STQ_BLOCK);
    let (mut sumx2, mut amax) = (0.0f32, 0.0f32);
    for &v in x {
        sumx2 += v * v;
        amax = amax.max(v.abs());
    }
    if !(amax > 0.0) {
        out.fill(0.0);
        return;
    }
    let sigma2 = 2.0 * sumx2 / STQ_BLOCK as f32;
    let mut weight = [0.0f32; STQ_BLOCK];
    for j in 0..STQ_BLOCK {
        weight[j] = stq_weight(x[j], sigma2, qw.map(|q| q[j]));
    }
    let mut sel = [0.0f32; STQ_BLOCK];
    let mut d = amax / 3.5;
    for _ in 0..3 {
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        for j in 0..STQ_BLOCK {
            let t = x[j] * inv;
            let idx = (t.abs() - 0.5).round().clamp(0.0, 3.0);
            let level = (2.0 * idx + 1.0) * 0.5;
            sel[j] = if t < 0.0 { -level } else { level };
        }
        let (mut sumqx, mut sumq2) = (0.0f32, 0.0f32);
        for j in 0..STQ_BLOCK {
            sumqx += weight[j] * sel[j] * x[j];
            sumq2 += weight[j] * sel[j] * sel[j];
        }
        if !(sumq2 > 0.0) {
            break;
        }
        let dnew = sumqx / sumq2;
        if !(dnew > 0.0) {
            break;
        }
        let converged = (dnew - d).abs() <= 1e-6 * d;
        d = dnew;
        if converged {
            break;
        }
    }
    let d16 = round_f16(d);
    for j in 0..STQ_BLOCK {
        out[j] = sel[j] * d16;
    }
}

fn stq_block_dispatch(kind: StqKind, x: &[f32], qw: Option<&[f32]>, out: &mut [f32]) {
    match kind {
        StqKind::Stq1_0 => stq1_0_block(x, qw, out),
        StqKind::Ternary2 => ternary2_block(x, qw, out),
        StqKind::Ls3Bit => ls3_block(x, qw, out),
    }
}

/// In-place values-only round-trip of an [M, K] f32 row-major tensor through one SQ format.
/// K must be a multiple of 256. `qw` = optional per-input-channel importance (len K).
/// Row-parallel; the tool is expected to run under a capped+niced taskset (SQ politeness rule).
pub fn fake_quant_stq(w: &mut [f32], m: usize, k: usize, qw: Option<&[f32]>, kind: StqKind) {
    assert_eq!(w.len(), m * k, "shape mismatch");
    assert_eq!(k % STQ_BLOCK, 0, "K={k} is not a multiple of STQ_BLOCK={STQ_BLOCK}");
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).max(1);
    let rows_per = m.div_ceil(nthreads).max(1);
    std::thread::scope(|sc| {
        for chunk in w.chunks_mut(rows_per * k) {
            let qw = qw.map(|q| &q[..k]);
            sc.spawn(move || {
                let mut blk = [0.0f32; STQ_BLOCK];
                for row in chunk.chunks_mut(k) {
                    for (b, out) in row.chunks_mut(STQ_BLOCK).enumerate() {
                        blk.copy_from_slice(&out[..STQ_BLOCK]);
                        let qws = qw.map(|q| &q[b * STQ_BLOCK..][..STQ_BLOCK]);
                        stq_block_dispatch(kind, &blk, qws, out);
                    }
                }
            });
        }
    });
}

/// Weighted SSD of a round-trip vs `x` under the AngelSlim objective (for `--stq-check`).
pub fn stq_weighted_ssd(x: &[f32], y: &[f32], qw: Option<&[f32]>) -> f64 {
    let n = x.len();
    let sumx2: f32 = x.iter().map(|v| v * v).sum();
    let sigma2 = 2.0 * sumx2 / n as f32;
    let mut ssd = 0.0f64;
    for j in 0..n {
        let w = stq_weight(x[j], sigma2, qw.map(|q| q[j]));
        let e = x[j] - y[j];
        ssd += (w * e * e) as f64;
    }
    ssd
}

/// The REFERENCE STQ1_0 encoder (upstream: d = amax, zero = argmin|x| per stride-16 group) —
/// the baseline the LS+imatrix encoder is measured against in `--stq-check`.
pub fn stq1_0_block_reference(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), STQ_BLOCK);
    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }
    if !(amax > 0.0) {
        out.fill(0.0);
        return;
    }
    let d16 = round_f16(amax);
    for g in 0..STQ_BLOCK / 4 {
        let chunk = g / 16;
        let gloc = g % 16;
        let base = chunk * 64 + gloc;
        let mut zero_pos = 0usize;
        let mut smallest = f32::INFINITY;
        for p in 0..4 {
            let j = base + p * 16;
            let ax = x[j].abs();
            if ax < smallest {
                smallest = ax;
                zero_pos = p;
            }
        }
        for p in 0..4 {
            let j = base + p * 16;
            out[j] = if p == zero_pos { 0.0 } else if x[j] < 0.0 { -d16 } else { d16 };
        }
    }
}

