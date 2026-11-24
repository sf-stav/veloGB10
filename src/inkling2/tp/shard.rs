//! The split of one TP rank (the design: PLAN/INKLING2_TP2_REPORT.md "Design"): what rank `r` of `W` holds, and the
//! host-side slicing of the pack's tensors at load.
//!
//! * attention: query heads [r*24/W, (r+1)*24/W), KV heads [r*8/W, ..) (group 3 kept on one rank): the fused projection
//!   rows of those heads (q, gate, r, sink, k, the dynamic-conv taps kd, v, vd), their q / r clip scales and k / v conv
//!   taps, and the matching wo columns (input width 24/W * 128); `rel_logits_proj` (per layer, not per head) replicated.
//! * MoE: every expert, an inner-width slice of whole 128-channel blocks (1,152 = 9 blocks: 5 / 4 at W = 2, 3 / 2 / 2 / 2
//!   at W = 4; the NVFP4 F128 fragments and the prefill down GEMM need K % 128, the MXFP8 blocks K % 32); router, norms,
//!   convs, AttnRes, MessageGates, embeddings replicated.
//! * LM head: vocabulary rows [v0, v0 + vn) (whole 128-row blocks except the last rank's tail).
//!
//! Two partial sums per layer visit cross the link: wo's output and the MoE combine's output, [m][2304] fp32 each.

use anyhow::{ensure, Result};

pub const QH: usize = 24;
pub const KVH: usize = 8;
pub const HD: usize = 128;
pub const DREL: usize = 16;
/// MoE inner block (Drop 3 #3b: 64-channel granule, 1,152 = 18 blocks: exact 576/576 at TP2,
/// 320/320/256/256 at TP4, 192/192/128x6 at TP8 — cuts rank-0 straggler from 2.0x -> 1.5x at TP8).
pub const IBLK: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rank {
    pub rank: usize,
    pub world: usize,
    /// query heads [h0, h0 + nq)
    pub h0: usize,
    pub nq: usize,
    /// KV heads [g0, g0 + nkv)
    pub g0: usize,
    pub nkv: usize,
    /// MoE inner channels [c0, c1)
    pub c0: usize,
    pub c1: usize,
    /// vocabulary rows [v0, v0 + vn)
    pub v0: usize,
    pub vn: usize,
}

/// The inner-channel range of `rank` of `world` over `inner` channels, in whole `IBLK` blocks (lower ranks take the
/// remainder blocks).
pub fn inner_range(inner: usize, world: usize, rank: usize) -> Result<(usize, usize)> {
    ensure!(inner % IBLK == 0, "MoE inner {inner} is not a multiple of {IBLK}");
    let nb = inner / IBLK;
    ensure!(nb >= world, "MoE inner {inner}: {nb} blocks cannot split over {world} ranks");
    let (base, rem) = (nb / world, nb % world);
    let start: usize = (0..rank).map(|r| base + (r < rem) as usize).sum();
    let n = base + (rank < rem) as usize;
    Ok((start * IBLK, (start + n) * IBLK))
}

/// The vocabulary range of `rank` of `world` over `v` rows (whole 128-row blocks; the last rank takes the tail).
pub fn vocab_range(v: usize, world: usize, rank: usize) -> (usize, usize) {
    let per = v.div_ceil(world).div_ceil(128) * 128;
    let v0 = (rank * per).min(v);
    let v1 = ((rank + 1) * per).min(v);
    (v0, v1 - v0)
}

impl Rank {
    pub fn solo(inner: usize, vocab: usize) -> Rank {
        Rank { rank: 0, world: 1, h0: 0, nq: QH, g0: 0, nkv: KVH, c0: 0, c1: inner, v0: 0, vn: vocab }
    }
    pub fn new(rank: usize, world: usize, inner: usize, vocab: usize) -> Result<Rank> {
        ensure!(matches!(world, 1 | 2 | 3 | 4 | 6 | 8) && rank < world, "inkling2 TP: rank {rank} of world {world} (TP1 / TP2 / TP3 / TP4 / TP6 / TP8)");
        // Split by whole KV heads (group-3 GQA kept intact on every rank: zero KV duplication and bit-exact per-head kernels):
        // TP2 [4,4] (12Q), TP3 [3,3,2] (9/9/6Q), TP4 [2x4] (6Q), TP6 [2,2,1,1,1,1] (6/6/3x4Q), TP8 [1x8] (3Q).
        let (base_kv, rem_kv) = (KVH / world, KVH % world);
        let g0: usize = (0..rank).map(|r| base_kv + (r < rem_kv) as usize).sum();
        let nkv = base_kv + (rank < rem_kv) as usize;
        let (h0, nq) = (g0 * (QH / KVH), nkv * (QH / KVH));
        let (c0, c1) = inner_range(inner, world, rank)?;
        let (v0, vn) = vocab_range(vocab, world, rank);
        Ok(Rank { rank, world, h0, nq, g0, nkv, c0, c1, v0, vn })
    }
    pub fn is_solo(&self) -> bool { self.world == 1 }
    /// K / V row width of this rank (cache rows, k / v conv rings).
    pub fn kvw(&self) -> usize { self.nkv * HD }
    pub fn inner(&self) -> usize { self.c1 - self.c0 }
    pub fn describe(&self) -> String {
        format!("rank {}/{}: query heads {}..{} ({}), KV heads {}..{} ({}), MoE inner channels {}..{} ({} of every expert), vocabulary rows {}..{}",
                self.rank, self.world, self.h0, self.h0 + self.nq, self.nq, self.g0, self.g0 + self.nkv, self.nkv,
                self.c0, self.c1, self.c1 - self.c0, self.v0, self.v0 + self.vn)
    }
}

/// Rows [r0, r1) of a row-major `[rows][k]` matrix.
pub fn rows(v: &[f32], k: usize, r0: usize, r1: usize) -> Vec<f32> { v[r0 * k..r1 * k].to_vec() }

/// Columns [c0, c1) of a row-major `[n][k]` matrix.
pub fn cols(v: &[f32], k: usize, c0: usize, c1: usize) -> Vec<f32> {
    let n = v.len() / k;
    let mut o = Vec::with_capacity(n * (c1 - c0));
    for r in 0..n { o.extend_from_slice(&v[r * k + c0..r * k + c1]); }
    o
}

/// Byte columns [c0, c1) of a row-major `[n][row_bytes]` byte matrix.
pub fn byte_cols(v: &[u8], row_bytes: usize, c0: usize, c1: usize) -> Vec<u8> {
    let n = v.len() / row_bytes;
    let mut o = Vec::with_capacity(n * (c1 - c0));
    for r in 0..n { o.extend_from_slice(&v[r * row_bytes + c0..r * row_bytes + c1]); }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inner_split_whole_blocks() {
        assert_eq!(inner_range(1152, 1, 0).unwrap(), (0, 1152));
        assert_eq!(inner_range(1152, 2, 0).unwrap(), (0, 576));
        assert_eq!(inner_range(1152, 2, 1).unwrap(), (576, 1152));
        let w4: Vec<_> = (0..4).map(|r| inner_range(1152, 4, r).unwrap()).collect();
        assert_eq!(w4, vec![(0, 320), (320, 640), (640, 896), (896, 1152)]);
        for (a, b) in &w4 { assert_eq!((b - a) % 64, 0); }
        let w8: Vec<_> = (0..8).map(|r| inner_range(1152, 8, r).unwrap()).collect();
        assert_eq!(w8, vec![(0, 192), (192, 384), (384, 512), (512, 640), (640, 768), (768, 896), (896, 1024), (1024, 1152)]);
        for (a, b) in &w8 { assert_eq!((b - a) % 64, 0); }
        let w3: Vec<_> = (0..3).map(|r| inner_range(1152, 3, r).unwrap()).collect();
        assert_eq!(w3, vec![(0, 384), (384, 768), (768, 1152)]);
        let w6: Vec<_> = (0..6).map(|r| inner_range(1152, 6, r).unwrap()).collect();
        assert_eq!(w6, vec![(0, 192), (192, 384), (384, 576), (576, 768), (768, 960), (960, 1152)]);
        let r3: Vec<_> = (0..3).map(|r| Rank::new(r, 3, 1152, 200_058).unwrap()).collect();
        assert_eq!(r3.iter().map(|x| (x.g0, x.nkv, x.h0, x.nq)).collect::<Vec<_>>(), vec![(0, 3, 0, 9), (3, 3, 9, 9), (6, 2, 18, 6)]);
        let r6: Vec<_> = (0..6).map(|r| Rank::new(r, 6, 1152, 200_058).unwrap()).collect();
        assert_eq!(r6.iter().map(|x| (x.g0, x.nkv, x.h0, x.nq)).collect::<Vec<_>>(), vec![(0, 2, 0, 6), (2, 2, 6, 6), (4, 1, 12, 3), (5, 1, 15, 3), (6, 1, 18, 3), (7, 1, 21, 3)]);
        assert!(inner_range(1100, 2, 0).is_err());
    }

    #[test]
    fn vocab_split_covers_once() {
        for w in [1usize, 2, 3, 4, 6, 8] {
            let mut next = 0;
            for r in 0..w {
                let (v0, n) = vocab_range(200058, w, r);
                assert_eq!(v0, next);
                if r + 1 < w { assert_eq!(n % 128, 0); }
                next = v0 + n;
            }
            assert_eq!(next, 200058);
        }
    }

    #[test]
    fn rank_geometry() {
        let r = Rank::new(1, 2, 1152, 200058).unwrap();
        assert_eq!((r.h0, r.nq, r.g0, r.nkv, r.kvw()), (12, 12, 4, 4, 512));
        // TP3: whole-KV-head split [3,3,2] KV = [9,9,6] Q, exact [384,384,384] MoE.
        let r3_0 = Rank::new(0, 3, 1152, 200058).unwrap();
        let r3_1 = Rank::new(1, 3, 1152, 200058).unwrap();
        let r3_2 = Rank::new(2, 3, 1152, 200058).unwrap();
        assert_eq!((r3_0.h0, r3_0.nq, r3_0.g0, r3_0.nkv, r3_0.kvw(), r3_0.inner()), (0, 9, 0, 3, 384, 384));
        assert_eq!((r3_1.h0, r3_1.nq, r3_1.g0, r3_1.nkv, r3_1.kvw(), r3_1.inner()), (9, 9, 3, 3, 384, 384));
        assert_eq!((r3_2.h0, r3_2.nq, r3_2.g0, r3_2.nkv, r3_2.kvw(), r3_2.inner()), (18, 6, 6, 2, 256, 384));
        let r = Rank::new(3, 4, 1152, 200058).unwrap();
        assert_eq!((r.h0, r.nq, r.g0, r.nkv, r.kvw(), r.inner()), (18, 6, 6, 2, 256, 256));
        // TP6: whole-KV-head split [2,2,1,1,1,1] KV = [6,6,3,3,3,3] Q, exact [192 x 6] MoE.
        let tp6: Vec<_> = (0..6).map(|rk| Rank::new(rk, 6, 1152, 200058).unwrap()).collect();
        assert_eq!(tp6.iter().map(|x| x.nkv).collect::<Vec<_>>(), vec![2, 2, 1, 1, 1, 1]);
        assert_eq!(tp6.iter().map(|x| x.nq).collect::<Vec<_>>(), vec![6, 6, 3, 3, 3, 3]);
        assert_eq!(tp6.iter().map(|x| x.inner()).collect::<Vec<_>>(), vec![192; 6]);
        let r = Rank::new(7, 8, 1152, 200058).unwrap();
        assert_eq!((r.h0, r.nq, r.g0, r.nkv, r.kvw(), r.inner()), (21, 3, 7, 1, 128, 128));
        assert!(Rank::new(0, 5, 1152, 200058).is_err());
        assert!(Rank::new(0, 7, 1152, 200058).is_err());
        assert_eq!(cols(&[1., 2., 3., 4., 5., 6.], 3, 1, 3), vec![2., 3., 5., 6.]);
        assert_eq!(byte_cols(&[1, 2, 3, 4], 2, 1, 2), vec![2, 4]);
    }
}
