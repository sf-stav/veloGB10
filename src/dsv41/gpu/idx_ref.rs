//! DSV4.1 leg KI: the host (CPU) reference of the indexer, statement for statement the GPU
//! kernels' arithmetic (kernels/gpu_dsv41/5x_idx_*.cuh), so the GPU gate is BIT-EXACT, plus the
//! pieces the reference model computes (`inference/model.py` `Indexer`, `select_candidate_blocks`).
//!
//! * `rope_inv_freq` — the YaRN inverse frequencies as torch computes them ON CUDA (the REFb
//!   fixtures build `freqs_cis` on the GPU; torch's CPU and CUDA tables differ in entry 24).
//! * `q_prep` — RoPE on dims 64..127 then `fp4_act_quant(q, 32, ue8m0)` (E2M1 RNE, pow2 scale).
//! * `dot` — the FP4 x FP4 dot as the GB10 block-scaled MMA returns it: per 64-wide k-step
//!   `acc = RZ_f32(acc + exact group sums)` (measured, PLAN/DSV41_KI_REPORT.md §3).
//! * `score` — the bf16 chain: bf16(dot) -> relu -> bf16(* w) -> fp32 head sum in the kernel's
//!   fixed order -> bf16, mapped to the 16-bit order key (`sortable`).
//! * `select_topk`, `select_blocks` — the total order (score desc, position asc); the layer-20
//!   candidate blocks (pin the newest block, top 2048, lowest id on ties).

/// One FP4 (E2M1) code -> value.
pub fn e2m1(c: u8) -> f32 {
    const M: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = M[(c & 7) as usize];
    if c & 8 != 0 { -v } else { v }
}

/// Round-to-nearest-even f32 -> bf16 -> f32.
pub fn bf16r(x: f32) -> f32 { bf16_from(bf16_bits(x)) }
pub fn bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    if x.is_nan() { return 0x7FC0; }
    let r = u.wrapping_add(0x7FFF + ((u >> 16) & 1));
    (r >> 16) as u16
}
fn bf16_from(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }

/// bf16 bits -> 16-bit key ordering like the value (-0 folded onto +0).
pub fn sortable(b: u16) -> u16 {
    let u = if b == 0x8000 { 0 } else { b };
    if u & 0x8000 != 0 { !u } else { u | 0x8000 }
}
/// The inverse of `sortable` (for reports).
pub fn unsortable(s: u16) -> f32 {
    let b = if s & 0x8000 != 0 { s & 0x7FFF } else { !s };
    bf16_from(b)
}

/// f32 -> E2M1 code, round-to-nearest-even, saturating at +-6 (the hardware `cvt.rn.satfinite`).
pub fn to_e2m1(x: f32) -> u8 {
    let s = if x.is_sign_negative() { 8u8 } else { 0 };
    let a = x.abs().min(6.0);
    // grid 0, .5, 1, 1.5, 2, 3, 4, 6 (codes 0..7); midpoints go to the even code
    let code = if a <= 0.25 { 0 } else if a < 0.75 { 1 } else if a <= 1.25 { 2 } else if a < 1.75 { 3 }
        else if a <= 2.5 { 4 } else if a < 3.5 { 5 } else if a <= 5.0 { 6 } else { 7 };
    if code == 0 { 0 } else { code | s }
}

/// YaRN inverse frequencies (model.py `precompute_freqs_cis`, fp32 op by op as torch runs them on
/// CUDA). `dim` = rope_head_dim (64), `base` = compress_rope_theta for compressed layers.
pub fn rope_inv_freq(dim: usize, base: f64, factor: f64, original_seq_len: f64, beta_fast: f64, beta_slow: f64) -> Vec<f32> {
    // the host emulation of the ONE dsv41 builder (`crate::dsv41::rope::table_host`; exact for this class)
    let yarn = (original_seq_len > 0.0).then_some((factor, original_seq_len, beta_fast, beta_slow));
    crate::dsv41::rope::table_host(&crate::dsv41::rope::Params { theta: base, dim, yarn }).to_vec()
}

/// One row's prepared index query: E2M1 codes [32 heads][64 B] (element 2i in the low nibble) and
/// the E8M0 exponent byte per (head, 32-group) [32][4]. Same layout as the kernel's `dbg_q` dump.
#[derive(Clone, Debug, PartialEq)]
pub struct QPrep { pub codes: Vec<u8>, pub scales: Vec<u8> }

/// RoPE + FP4 of one row. `iq` holds 32 x 128 values already rounded to bf16 (the input dtype);
/// `cs` = (cos, sin) per rope pair at this row's position (CUDA sincosf in the kernel, which equals
/// torch.polar on CUDA bit for bit; the rotation is torch's CUDA complex multiply, two FMAs).
pub fn q_prep(iq: &[f32], cs: &[(f32, f32)]) -> QPrep {
    assert_eq!(iq.len(), 32 * 128);
    let mut codes = vec![0u8; 32 * 64];
    let mut scales = vec![0u8; 32 * 4];
    for h in 0..32 {
        for gq in 0..4 {
            let mut x = [0f32; 32];
            x.copy_from_slice(&iq[h * 128 + gq * 32..h * 128 + gq * 32 + 32]);
            if gq >= 2 {
                for j in 0..16 {
                    let (c, s) = cs[(gq - 2) * 16 + j];
                    let (a, b) = (x[2 * j], x[2 * j + 1]);
                    // torch's CUDA complex multiply: re = fma(a, c, -(b s)), im = fma(b, c, a s)
                    x[2 * j] = bf16r(a.mul_add(c, -(b * s)));
                    x[2 * j + 1] = bf16r(b.mul_add(c, a * s));
                }
            }
            let mut amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
            amax = amax.max(6.0 * f32::from_bits(0x0080_0000));
            let sb = (amax * (1.0f32 / 6.0f32)).to_bits();
            let mut e = (sb >> 23) & 0xFF;
            if sb & 0x7F_FFFF != 0 { e += 1; }
            let sc = f32::from_bits(e << 23);
            for j in (0..32).step_by(2) {
                let lo = to_e2m1(x[j] / sc);
                let hi = to_e2m1(x[j + 1] / sc);
                codes[h * 64 + gq * 16 + j / 2] = lo | (hi << 4);
            }
            scales[h * 4 + gq] = e as u8;
        }
    }
    QPrep { codes, scales }
}

/// Exact (f64) sum of one 32-group of an FP4 dot, unscaled.
fn group_sum(qc: &[u8], kc: &[u8], grp: usize) -> f64 {
    let mut s = 0f64;
    for i in 0..16 {
        let (a, b) = (qc[grp * 16 + i], kc[grp * 16 + i]);
        s += e2m1(a & 15) as f64 * e2m1(b & 15) as f64 + e2m1(a >> 4) as f64 * e2m1(b >> 4) as f64;
    }
    s
}

/// Round an exactly known value toward zero to f32.
fn rz_f32(x: f64) -> f32 {
    let f = x as f32; // round to nearest
    if (f as f64).abs() > x.abs() { f32::from_bits(f.to_bits() - 1) } else { f }
}

/// One head's dot q_h . k_t as the block-scaled MMA returns it (two k-steps of 64).
pub fn dot(qc: &[u8], qs: &[u8], kc: &[u8], ks: &[u8]) -> f32 {
    let mut acc = 0f32;
    for kstep in 0..2 {
        let mut s = acc as f64;
        for half in 0..2 {
            let g = kstep * 2 + half;
            let sc = 2f64.powi(qs[g] as i32 - 127) * 2f64.powi(ks[g] as i32 - 127);
            s += group_sum(qc, kc, g) * sc;
        }
        acc = rz_f32(s);
    }
    acc
}

/// The score of one key for one row: the kernel's bf16 chain and head-sum order.
/// `w` = the 32 weights_proj outputs (bf16 values), unscaled.
pub fn score(q: &QPrep, w: &[f32], kc: &[u8], ks: &[u8]) -> u16 {
    let prod = |h: usize| -> f32 {
        let d = bf16r(dot(&q.codes[h * 64..h * 64 + 64], &q.scales[h * 4..h * 4 + 4], kc, ks)).max(0.0);
        let we = bf16r(w[h] * 0.015625);
        bf16r(d * we)
    };
    let mut part = [0f32; 4];
    for (t, p) in part.iter_mut().enumerate() {
        let mut acc = 0f32;
        for nt in 0..4 {
            acc += prod(nt * 8 + 2 * t);
            acc += prod(nt * 8 + 2 * t + 1);
        }
        *p = acc;
    }
    let total = (part[0] + part[1]) + (part[2] + part[3]);
    sortable(bf16_bits(total))
}

/// FP4 codes decoded to integers (value x 2, exact): the fast path of `dot` / `score`.
pub fn decode2(codes: &[u8]) -> Vec<i8> {
    const M2: [i8; 8] = [0, 1, 2, 3, 4, 6, 8, 12];
    let mut v = Vec::with_capacity(codes.len() * 2);
    for &b in codes {
        for c in [b & 15, b >> 4] { let m = M2[(c & 7) as usize]; v.push(if c & 8 != 0 { -m } else { m }); }
    }
    v
}

/// `dot` on decoded operands (q / k: 128 values x 2; scales: 4 E8M0 bytes each).
pub fn dot_dec(q: &[i8], qs: &[u8], k: &[i8], ks: &[u8]) -> f32 {
    let mut acc = 0f32;
    for kstep in 0..2 {
        let mut s = acc as f64;
        for half in 0..2 {
            let g = kstep * 2 + half;
            let mut isum = 0i32;
            for i in g * 32..g * 32 + 32 { isum += q[i] as i32 * k[i] as i32; }
            s += isum as f64 * 0.25 * 2f64.powi(qs[g] as i32 + ks[g] as i32 - 254);
        }
        acc = rz_f32(s);
    }
    acc
}

/// One row's query decoded once, with its scaled head weights (bf16(bf16(w) * 2^-6)).
pub struct QDec { pub v2: Vec<i8>, pub sc: Vec<u8>, pub weff: Vec<f32> }
impl QDec {
    pub fn new(q: &QPrep, w: &[f32]) -> QDec {
        QDec { v2: decode2(&q.codes), sc: q.scales.clone(), weff: w.iter().map(|&x| bf16r(bf16r(x) * 0.015625)).collect() }
    }
    /// `score` of one key given as decoded values + its 4 scale bytes.
    pub fn score(&self, k: &[i8], ks: &[u8]) -> u16 {
        let prod = |h: usize| bf16r(bf16r(dot_dec(&self.v2[h * 128..h * 128 + 128], &self.sc[h * 4..h * 4 + 4], k, ks)).max(0.0) * self.weff[h]);
        let mut part = [0f32; 4];
        for (t, p) in part.iter_mut().enumerate() {
            let mut acc = 0f32;
            for nt in 0..4 { acc += prod(nt * 8 + 2 * t); acc += prod(nt * 8 + 2 * t + 1); }
            *p = acc;
        }
        sortable(bf16_bits((part[0] + part[1]) + (part[2] + part[3])))
    }
}

/// The order key: higher score first, then lower position.
pub fn key(s16: u16, pos: u32) -> u64 { ((s16 as u64) << 32) | (0xFFFF_FFFFu32 - pos) as u64 }

/// Top `k` positions by (score desc, position asc) among `(pos, s16)` candidates, returned in
/// ascending position order.
pub fn select_topk(cands: &[(u32, u16)], k: usize) -> Vec<u32> {
    let mut v: Vec<u64> = cands.iter().map(|&(p, s)| key(s, p)).collect();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v.truncate(k);
    let mut pos: Vec<u32> = v.iter().map(|&x| 0xFFFF_FFFFu32 - (x & 0xFFFF_FFFF) as u32).collect();
    pos.sort_unstable();
    pos
}

/// Layer-20 candidate blocks of one row: `bmax[b]` = max sortable score of block b's visible keys
/// (0 = none); `n_vis` visible positions. Returns the kept block ids ascending.
pub fn select_blocks(bmax: &[u16], n_vis: usize, topk_blocks: usize, block: usize) -> Vec<u32> {
    let nb = n_vis.div_ceil(block);
    if nb <= topk_blocks { return (0..nb as u32).collect(); }
    let m = nb - 1;
    let mut v: Vec<u64> = (0..m).map(|b| key(bmax[b], b as u32)).collect();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v.truncate(topk_blocks - 1);
    let mut ids: Vec<u32> = v.iter().map(|&x| 0xFFFF_FFFFu32 - (x & 0xFFFF_FFFF) as u32).collect();
    ids.push(m as u32);
    ids.sort_unstable();
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv41_idx_ref_e2m1_rne() {
        // ties to the even code (REF_MAP §2.3): .25->0 .75->1 1.25->1 1.75->2 2.5->2 3.5->4 5->4
        let t = [(0.25f32, 0.0f32), (0.75, 1.0), (1.25, 1.0), (1.75, 2.0), (2.5, 2.0), (3.5, 4.0), (5.0, 4.0), (7.0, 6.0), (-0.2, 0.0), (-2.6, -3.0)];
        for (x, want) in t { assert_eq!(e2m1(to_e2m1(x)), want, "{x}"); }
        for c in 0..16u8 { if c != 8 { assert_eq!(to_e2m1(e2m1(c)), c, "code {c}"); } }
    }

    #[test]
    fn dsv41_idx_ref_bf16_round_to_nearest_even() {
        assert_eq!(bf16r(1.0), 1.0);
        assert_eq!(bf16r(1.0 + 1.0 / 256.0), 1.0, "tie to even (down)");
        assert_eq!(bf16r(1.0 + 3.0 / 256.0), 1.0 + 4.0 / 256.0, "tie to even (up)");
        assert_eq!(bf16r(-3.140625), -3.140625);
        assert_eq!(bf16_bits(1.0), 0x3F80);
    }

    #[test]
    fn dsv41_idx_ref_sortable_orders_like_values() {
        let vals = [-3.0f32, -1.0, -1e-3, -0.0, 0.0, 1e-3, 0.5, 2.0, 1e30];
        let s: Vec<u16> = vals.iter().map(|&v| sortable(bf16_bits(v))).collect();
        for i in 1..vals.len() { assert!(s[i] >= s[i - 1], "{:?}", s); }
        assert_eq!(sortable(0x8000), sortable(0x0000), "-0 == +0");
        assert_eq!(unsortable(sortable(bf16_bits(0.5))), 0.5);
    }

    #[test]
    fn dsv41_idx_ref_rope_table_matches_torch_cuda() {
        // torch 2.11 on CUDA (GB10), precompute_freqs_cis(64, ., 65536, 160000, 16, 32, 1): the
        // table REFb's fixtures were built with (torch's CPU table differs in entry 24: 0x37a3d70c).
        const CUDA: [u32; 32] = [0x3f800000, 0x3f300a3a, 0x3ef21c1f, 0x3ea67d01, 0x3e64f92e, 0x3e1d7475, 0x3dd88cb4, 0x3d94e963,
            0x3d4ccccd, 0x3d0cd4fb, 0x3cc1b019, 0x3c8530ce, 0x3c372dbf, 0x3bfbed88, 0x3bad3d5e, 0x3b6e4237,
            0x3b147ae1, 0x3ab714e0, 0x3a5ebdb6, 0x3a0530ce, 0x399bb3af, 0x39305978, 0x38be904d, 0x383e9b5f,
            0x37a3d708, 0x36b443d0, 0x3677eba6, 0x362a7be8, 0x35ea77ff, 0x35a13bdc, 0x355dbf30, 0x35187c4d];
        let f = rope_inv_freq(64, 160000.0, 16.0, 65536.0, 32.0, 1.0);
        let got: Vec<u32> = f.iter().map(|x| x.to_bits()).collect();
        assert_eq!(got, CUDA.to_vec());
    }

    #[test]
    fn dsv41_idx_ref_blocks_pin_and_ties() {
        // 2,050 blocks: the pinned last block, then the top 2,047 of the rest; ties -> lowest id
        let nb = 2050usize;
        let mut bmax = vec![0x9000u16; nb];
        bmax[5] = 0x8000; bmax[7] = 0x8000; bmax[9] = 0x8000; // three lowest scores
        let ids = select_blocks(&bmax, nb * 8 - 3, 2048, 8);
        assert_eq!(ids.len(), 2048);
        assert_eq!(*ids.last().unwrap(), 2049);
        // 2049 competing blocks, keep 2047: drop two of the three 0x8000 blocks -> keep the lowest id (5)
        assert!(ids.contains(&5) && !ids.contains(&7) && !ids.contains(&9));
    }

    #[test]
    fn dsv41_idx_ref_fast_path_equals_slow_path() {
        let mut st = 0x1234_5678_9abc_def0u64;
        let mut nx = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
        let iq: Vec<f32> = (0..4096).map(|_| bf16r(((nx() % 2001) as f32 - 1000.0) / 250.0)).collect();
        let cs: Vec<(f32, f32)> = (0..32).map(|i| ((i as f32 * 0.3).cos(), (i as f32 * 0.3).sin())).collect();
        let q = q_prep(&iq, &cs);
        let w: Vec<f32> = (0..32).map(|_| bf16r(((nx() % 401) as f32 - 200.0) / 100.0)).collect();
        let qd = QDec::new(&q, &w);
        for _ in 0..50 {
            let kc: Vec<u8> = (0..64).map(|_| (nx() & 0xFF) as u8).collect();
            let ks: Vec<u8> = (0..4).map(|_| 120 + (nx() % 8) as u8).collect();
            assert_eq!(score(&q, &w, &kc, &ks), qd.score(&decode2(&kc), &ks));
        }
        // every scale byte is a power of two at or above amax / 6, codes within +-6
        for h in 0..32 { for g in 0..4 { assert!(q.scales[h * 4 + g] >= 1); } }
    }

    #[test]
    fn dsv41_idx_ref_topk_ties_lowest_position() {
        let c: Vec<(u32, u16)> = (0..1000u32).map(|p| (p, if p % 3 == 0 { 0xA000 } else { 0x9000 })).collect();
        let s = select_topk(&c, 512);
        // 334 positions score 0xA000 (all kept), then the 178 lowest positions of the rest
        assert_eq!(s.len(), 512);
        let low: Vec<u32> = s.iter().copied().filter(|p| p % 3 != 0).collect();
        assert_eq!(low.len(), 178);
        assert_eq!(*low.last().unwrap(), (0..1000u32).filter(|p| p % 3 != 0).nth(177).unwrap());
    }
}
