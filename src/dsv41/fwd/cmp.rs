//! SL: comparisons of the forward's outputs with REFb's dumps, the per-class tolerances (stated before the
//! first run, per backend), and the cache decoders (FP8 window planes, 288-B main-KV records, index-K planes)
//! that turn our quantised caches back into the values the reference stores (it keeps dequantised bf16).

use crate::dsv41::gpu::abi;

/// One comparison: rel-L2 = ||a - b|| / ||b||, the share of exactly equal values, the largest |a - b|.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cmp { pub rel: f64, pub same: f64, pub max_abs: f64, pub n: usize }

pub fn cmp(a: &[f32], b: &[f32]) -> Cmp {
    let n = a.len().min(b.len());
    if n == 0 { return Cmp::default(); }
    let (mut d2, mut b2, mut same, mut mx) = (0f64, 0f64, 0usize, 0f64);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        let d = x - y;
        d2 += d * d;
        b2 += y * y;
        if a[i] == b[i] { same += 1; }
        mx = mx.max(d.abs());
    }
    Cmp { rel: if b2 > 0.0 { (d2 / b2).sqrt() } else { d2.sqrt() }, same: same as f64 / n as f64, max_abs: mx, n }
}

/// What a point is checked against (rel-L2 at most, identical share at least).
#[derive(Clone, Copy, Debug)]
pub struct Tol { pub rel: f64, pub same: f64 }

/// The backend a pass ran on; EXL3 points that carry quantised weights get the EXL3 ceilings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass { Native, Exl3 }

/// Per-point tolerances. Native: from the oracle / kernel legs' measurements against the same dumps (OA, OM, OF,
/// KA, KN2). EXL3 (a priori, from the trellis distortion at K bits, rel-L2 ~ 2^-K x 1.1..1.3 per linear; the
/// pack's K per class is in I0's census): weight-free points (mixes, norms, hc_post, router on dumped
/// input) keep the native bound; points behind EXL3 linears get a ceiling of ~1.5-2x the composed distortion.
pub fn tol(p: Pass, point: &str, layer: usize) -> Tol {
    let k2 = (18..=22).contains(&layer);   // EAQ v1: routed experts K = 2 at layers 18-22, K = 3 elsewhere
    match (p, point) {
        (_, "attn_mix" | "ffn_mix") => Tol { rel: 1e-5, same: 0.0 },
        (_, "attn_in" | "ffn_in") => Tol { rel: 1e-3, same: 0.99 },
        (_, "x_mid" | "block_out") => Tol { rel: 1e-3, same: 0.99 },
        (_, "router") => Tol { rel: 0.0, same: 0.995 },
        (Pass::Native, "window_ring" | "compress_kv" | "index_k") => Tol { rel: 1e-2, same: 0.99 },
        (Pass::Native, "kv_state" | "score_state") => Tol { rel: 1e-5, same: 0.0 },
        (Pass::Native, "topk") => Tol { rel: 0.0, same: 0.95 },
        (Pass::Native, "attn_out") => Tol { rel: 2.5e-2, same: 0.0 },
        (Pass::Native, "ffn_out") => Tol { rel: 1e-2, same: 0.0 },
        (Pass::Native, "engram") => Tol { rel: 1e-3, same: 0.0 },
        (Pass::Exl3, "window_ring") => Tol { rel: 0.10, same: 0.0 },
        (Pass::Exl3, "compress_kv" | "index_k") => Tol { rel: 0.15, same: 0.0 },
        (Pass::Exl3, "kv_state" | "score_state") => Tol { rel: 0.10, same: 0.0 },
        (Pass::Exl3, "topk") => Tol { rel: 0.0, same: 0.0 },   // overlap gate below (mean overlap >= 0.75)
        (Pass::Exl3, "attn_out") => Tol { rel: 0.20, same: 0.0 },
        (Pass::Exl3, "ffn_out") => Tol { rel: if k2 { 0.45 } else { 0.25 }, same: 0.0 },
        (Pass::Exl3, "engram") => Tol { rel: 0.02, same: 0.0 },
        // DSpark stages against OD's oracle traces (the trace is the reference; same classes as the trunk)
        (_, "ds_attn_mix" | "ds_ffn_mix") => Tol { rel: 1e-5, same: 0.0 },
        (_, "ds_attn_x" | "ds_ffn_x" | "ds_hc_mid" | "ds_hc_out") => Tol { rel: 1e-3, same: 0.99 },
        (_, "ds_router") => Tol { rel: 0.0, same: 0.995 },
        (Pass::Native, "ds_q") => Tol { rel: 5e-3, same: 0.0 },
        (Pass::Native, "ds_kv") => Tol { rel: 1e-2, same: 0.99 },
        (Pass::Native, "ds_attn_out") => Tol { rel: 2.5e-2, same: 0.0 },
        (Pass::Native, "ds_ffn_out") => Tol { rel: 1e-2, same: 0.0 },
        (Pass::Native, "ds_main_x") => Tol { rel: 1e-3, same: 0.99 },
        (Pass::Exl3, "ds_q") => Tol { rel: 0.08, same: 0.0 },
        (Pass::Exl3, "ds_kv") => Tol { rel: 0.10, same: 0.0 },
        (Pass::Exl3, "ds_attn_out") => Tol { rel: 0.20, same: 0.0 },
        (Pass::Exl3, "ds_ffn_out") => Tol { rel: 0.20, same: 0.0 },   // DSpark experts K = 4 in the EAQ v1 pack
        (Pass::Exl3, "ds_main_x") => Tol { rel: 0.10, same: 0.0 },
        _ => Tol { rel: f64::INFINITY, same: 0.0 },   // informational points
    }
}

/// Mean top-k overlap a topk point must reach.
pub fn topk_overlap_min(p: Pass) -> f64 { match p { Pass::Native => 0.99, Pass::Exl3 => 0.75 } }

pub fn bf16_to_f32(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }
pub fn f32_to_bf16_bits(v: f32) -> u16 { (v.to_bits() >> 16) as u16 }   // exact for values that came from bf16

/// FP8 window planes (q [rows][512] E4M3, s [rows][16] E8M0) -> values.
pub fn decode_fp8_rows(q: &[u8], s: &[u8], rows: usize, dim: usize) -> Vec<f32> {
    let mut v = vec![0f32; rows * dim];
    for r in 0..rows {
        for c in 0..dim {
            v[r * dim + c] = abi::e4m3(q[r * dim + c]) * abi::e8m0(s[r * (dim / 32) + c / 32]);
        }
    }
    v
}

/// 288-B main-KV records (256 B E2M1 pairs, element 2i in the low nibble, then 32 E4M3 scales per 16) -> values.
pub fn decode_kv_records(rec: &[u8], rows: usize) -> Vec<f32> {
    let mut v = vec![0f32; rows * 512];
    for r in 0..rows {
        let b = &rec[r * 288..(r + 1) * 288];
        for c in 0..512 {
            let byte = b[c / 2];
            let code = if c % 2 == 0 { byte & 15 } else { byte >> 4 };
            v[r * 512 + c] = abi::e2m1(code) * abi::e4m3(b[256 + c / 16]);
        }
    }
    v
}

/// Index-K planes (q [rows][64 B] E2M1 pairs, s [rows][4 B] E8M0 per 32) -> values [rows][128].
pub fn decode_index_k(q: &[u8], s: &[u8], rows: usize) -> Vec<f32> {
    let mut v = vec![0f32; rows * 128];
    for r in 0..rows {
        for c in 0..128 {
            let byte = q[r * 64 + c / 2];
            let code = if c % 2 == 0 { byte & 15 } else { byte >> 4 };
            v[r * 128 + c] = abi::e2m1(code) * abi::e8m0(s[r * 4 + c / 32]);
        }
    }
    v
}

/// Top-k agreement of two selection tables (each row ascending, -1 padded): rows identical, mean overlap
/// (|ours ∩ ref| / |ref|), the worst row's overlap.
pub fn topk_agree(ours: &[i32], ours_stride: usize, refk: &[i64], ref_stride: usize, rows: usize) -> (usize, f64, f64) {
    let (mut same, mut sum, mut worst) = (0usize, 0f64, 1f64);
    for r in 0..rows {
        let a: Vec<i64> = ours[r * ours_stride..(r + 1) * ours_stride].iter().filter(|&&x| x >= 0).map(|&x| x as i64).collect();
        let b: Vec<i64> = refk[r * ref_stride..(r + 1) * ref_stride].iter().copied().filter(|&x| x >= 0).collect();
        if a == b { same += 1; }
        let bs: std::collections::HashSet<i64> = b.iter().copied().collect();
        let inter = a.iter().filter(|x| bs.contains(x)).count();
        let ov = if b.is_empty() { if a.is_empty() { 1.0 } else { 0.0 } } else { inter as f64 / b.len() as f64 };
        sum += ov;
        worst = worst.min(ov);
    }
    (same, if rows > 0 { sum / rows as f64 } else { 1.0 }, worst)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dsv41_sl_cmp_and_decoders() {
        let c = cmp(&[1.0, 2.0, 3.0], &[1.0, 2.0, 4.0]);
        assert!((c.rel - (1.0f64 / (1.0 + 4.0 + 16.0f64).sqrt())).abs() < 1e-12 && (c.same - 2.0 / 3.0).abs() < 1e-12);
        // one record: code 2 (= 1.0) in element 0, scale byte 0x38 (E4M3 1.0)
        let mut rec = vec![0u8; 288];
        rec[0] = 0x02;
        rec[256] = 0x38;
        let v = decode_kv_records(&rec, 1);
        assert_eq!(v[0], 1.0);
        let (same, ov, worst) = topk_agree(&[1, 2, 3, -1], 4, &[1, 2, 4], 3, 1);
        assert_eq!(same, 0);
        assert!((ov - 2.0 / 3.0).abs() < 1e-12 && (worst - 2.0 / 3.0).abs() < 1e-12);
    }
}
