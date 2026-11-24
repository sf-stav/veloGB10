//! DSV4.1 CPU oracle — leg OF, part 2: the Engram module (`model.py` `Engram` 328-365, `ParallelEngramEmbedding`
//! 296-325; REF map §3.7) and the proof object for T0 §3.6 (does `wkv` split exactly by Engram head shard?).
//!
//! ```text
//! e   = bf16( float(e4m3 row) * 2^e8m0 )            rows gathered by ENGa's ids, 24 columns x 256  -> [6144]
//! kv  = wkv(e)                                       fp8 linear 6144 -> 25600 (act_quant + fp8_gemm), bf16
//! key = kv[:20480] as [4, 5120], value = kv[20480:]
//! h   = stream [4, 5120] (fp32);  w = q_weight * k_weight
//! rstd = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps)         per (token, copy), eps = 1e-20
//! dot  = sum((h * w) * key) * rstd * 5120^-0.5
//! gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))
//! out  = bf16(h + gate * value)                      the same value row added to all four copies
//! ```
//! The module has NO norm over the concatenated embedding rows: `kv` is a plain linear map of `e`, and the only
//! reductions over feature dimensions (`mean(h^2)`, `mean(key^2)`, the dot) read the REDUCED `kv` / the stream.

use super::load::Store;
use super::quant::*;
use anyhow::{ensure, Context, Result};

pub const HC: usize = 4;
pub const DIM: usize = 5120;
pub const HEADS: usize = 8;
pub const NCOL: usize = 24;
pub const COLW: usize = 256;
pub const KIN: usize = NCOL * COLW; // 6144
pub const KOUT: usize = (HC + 1) * DIM; // 25600
pub const EPS: f32 = 1e-20;

pub struct EngramW { pub wkv: Fp8W, pub q: Vec<f32>, pub k: Vec<f32> }

impl EngramW {
    /// `weights` holds `layers.L.engram.{wkv,q_weight,k_weight}`.
    pub fn load(weights: &Store, layer: usize) -> Result<EngramW> {
        let b = format!("layers.{layer}.engram");
        let wkv = weights.fp8(&format!("{b}.wkv"))?;
        ensure!(wkv.n == KOUT && wkv.k == KIN, "wkv shape [{}, {}]", wkv.n, wkv.k);
        Ok(EngramW { wkv, q: weights.f32s(&format!("{b}.q_weight"))?, k: weights.f32s(&format!("{b}.k_weight"))? })
    }
}

/// Gather the bf16 rows `[n, 6144]` for hash ids `[n, 24]` (absolute row indices into the layer's table).
/// `tables` holds `layers.L.engram.embed.{weight,scale}` (the native shards 47 / 48).
pub fn embed_rows(tables: &Store, layer: usize, ids: &[i64], n: usize) -> Result<Vec<f32>> {
    ensure!(ids.len() == n * NCOL, "ids {} != {n} x {NCOL}", ids.len());
    let wn = format!("layers.{layer}.engram.embed.weight");
    let sn = format!("layers.{layer}.engram.embed.scale");
    let tab = e4m3_table();
    let mut out = vec![0f32; n * KIN];
    for (i, &id) in ids.iter().enumerate() {
        let w = tables.rows(&wn, id as u64, 1).with_context(|| format!("row {id}"))?;
        let s = tables.rows(&sn, id as u64, 1)?;
        ensure!(w.len() == COLW && s.len() == COLW / GROUP, "row sizes {} {}", w.len(), s.len());
        for j in 0..COLW { out[i * COLW + j] = bf16r(tab[w[j] as usize] * e8m0(s[j / GROUP])); }
    }
    Ok(out)
}

/// K blocks (of 32) of the K range owned by the given Engram heads: columns `c` with `c % 8 == head`, ascending.
pub fn head_kblocks(heads: &[usize]) -> Vec<usize> {
    let mut cols: Vec<usize> = (0..NCOL).filter(|c| heads.contains(&(c % HEADS))).collect();
    cols.sort();
    cols.iter().flat_map(|&c| c * COLW / GROUP..(c + 1) * COLW / GROUP).collect()
}

/// Heads held by rank `r` at tensor-parallel width `tp` (session rules: TP8 r, TP4 {2r, 2r+1}, TP2 {4r..4r+3}).
pub fn rank_heads(tp: usize, r: usize) -> Vec<usize> { let per = HEADS / tp; (r * per..(r + 1) * per).collect() }

/// `kv` as the f32 accumulator for `e` (`[n, 6144]` bf16-exact): the reference (`shards = None`, one sequential
/// pass over the 192 K blocks) or per-rank partials over the heads' K ranges, reduced in rank order in fp32.
/// The result is NOT yet rounded to bf16 (the caller rounds the reduced total once).
pub fn kv_f32(w: &EngramW, e: &[f32], n: usize, shards: Option<&[Vec<usize>]>) -> Vec<f32> {
    let a = act_quant(e, n, KIN);
    match shards {
        None => qgemm_f32(&a, &w.wkv, None),
        Some(sh) => {
            let mut tot = vec![0f32; n * KOUT];
            for heads in sh {
                let kb = head_kblocks(heads);
                let p = qgemm_f32(&a, &w.wkv, Some(&kb));
                for i in 0..tot.len() { tot[i] += p[i]; }
            }
            tot
        }
    }
}

/// The gate and the residual write: `x` = stream `[n, 4, 5120]` (bf16-exact), `kv` = bf16-rounded `[n, 25600]`.
pub fn apply(w: &EngramW, x: &[f32], kv: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * HC * DIM];
    let wgt: Vec<f32> = (0..HC * DIM).map(|i| w.q[i] * w.k[i]).collect();
    for t in 0..n {
        let kvr = &kv[t * KOUT..(t + 1) * KOUT];
        let value = &kvr[HC * DIM..];
        for c in 0..HC {
            let h = &x[(t * HC + c) * DIM..(t * HC + c + 1) * DIM];
            let key = &kvr[c * DIM..(c + 1) * DIM];
            let (mut h2, mut k2, mut dt) = (0f64, 0f64, 0f64);
            for i in 0..DIM {
                h2 += (h[i] * h[i]) as f64;
                k2 += (key[i] * key[i]) as f64;
                dt += ((h[i] * wgt[c * DIM + i]) * key[i]) as f64;
            }
            let rstd = (1.0 / ((h2 / DIM as f64) as f32 + EPS).sqrt()) * (1.0 / ((k2 / DIM as f64) as f32 + EPS).sqrt());
            let dot = dt as f32 * rstd * (DIM as f32).powf(-0.5);
            let s = dot.abs().max(1e-6).sqrt().copysign(dot);
            let gate = 1.0 / (1.0 + (-s).exp());
            for i in 0..DIM { out[(t * HC + c) * DIM + i] = bf16r(h[i] + gate * value[i]); }
        }
    }
    out
}

/// Whole module for `n` tokens: rows -> kv -> gate. `shards` as in `kv_f32`.
pub fn forward(w: &EngramW, rows: &[f32], x: &[f32], n: usize, shards: Option<&[Vec<usize>]>) -> Vec<f32> {
    let kv: Vec<f32> = kv_f32(w, rows, n, shards).into_iter().map(bf16r).collect();
    apply(w, x, &kv, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_shards_partition_the_k_range() {
        for tp in [2usize, 4, 8] {
            let mut all: Vec<usize> = (0..tp).flat_map(|r| head_kblocks(&rank_heads(tp, r))).collect();
            all.sort();
            assert_eq!(all, (0..KIN / GROUP).collect::<Vec<_>>(), "tp {tp}");
            // every shard is a union of whole 256-wide columns (8 groups of 32 each)
            for r in 0..tp { assert_eq!(head_kblocks(&rank_heads(tp, r)).len() % 8, 0); }
        }
        assert_eq!(head_kblocks(&[0]).len(), 24);
        assert_eq!(&head_kblocks(&[1])[..9], &[8, 9, 10, 11, 12, 13, 14, 15, 72]);
    }

    /// Synthetic exactness of the K split (no fixture): partial sums over disjoint K ranges, summed in fp32,
    /// equal the sequential result up to fp32 reordering, and the groups of `act_quant` never straddle a shard.
    #[test]
    fn k_split_matches_sequential_up_to_fp32_order() {
        let n = 3;
        let kout = 64usize;
        let wkv = Fp8W { n: kout, k: KIN, codes: (0..kout * KIN).map(|i| ((i * 2654435761usize >> 7) % 120) as u8).collect(), scale: vec![127; kout.div_ceil(32) * (KIN / GROUP)] };
        let e: Vec<f32> = (0..n * KIN).map(|i| bf16r((((i * 40503) % 977) as f32 - 488.0) / 97.0)).collect();
        let a = act_quant(&e, n, KIN);
        let full = qgemm_f32(&a, &wkv, None);
        for tp in [2usize, 4, 8] {
            let mut tot = vec![0f32; n * kout];
            for r in 0..tp {
                let p = qgemm_f32(&a, &wkv, Some(&head_kblocks(&rank_heads(tp, r))));
                for i in 0..tot.len() { tot[i] += p[i]; }
            }
            let scale = full.iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
            let mut maxrel = 0f64;
            for i in 0..tot.len() { maxrel = maxrel.max((tot[i] - full[i]).abs() as f64 / scale); }
            assert!(maxrel < 1e-6, "tp {tp}: {maxrel}");
        }
        // activation groups are shard-local: quantising a shard's own slice alone gives the same q and scales
        let sh = head_kblocks(&rank_heads(4, 1));
        let mut slice = Vec::new();
        for &kb in &sh { slice.extend_from_slice(&e[kb * GROUP..(kb + 1) * GROUP]); }
        let a2 = act_quant(&slice, 1, sh.len() * GROUP);
        for (j, &kb) in sh.iter().enumerate() {
            assert_eq!(a2.s[j], a.s[kb]);
            assert_eq!(&a2.q[j * GROUP..(j + 1) * GROUP], &a.q[kb * GROUP..(kb + 1) * GROUP]);
        }
    }
}
