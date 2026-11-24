//! DSV4.1 native MXFP4 experts (leg KN2): the "mx1" packed device layout of a routed / DSpark
//! expert, built on the host from the checkpoint's raw tensors (`w` [N][K/2] u8, element 2i in the
//! LOW nibble; `scale` [N][K/32] ue8m0). The kernel side and the fragment map are documented in
//! `kernels/gpu_dsv41/30_mxfp4_moe.cuh`; this file is the only producer of the layout.
//!
//! Per expert, per rank (inner slice `[i0, i0 + ns)`, `ns` a multiple of 32):
//! - **w13**: `ns / 8` m-tiles over K = dim; m-tile j holds w1 rows `i0 + 8j .. +7` in A rows 0-7 and
//!   w3 rows `i0 + 8j .. +7` in A rows 8-15 (SwiGLU pairs meet in one lane).
//! - **w2**: `dim / 16` m-tiles over K = the slice; m-tile j holds w2 rows `16j .. +15`, columns
//!   `[i0, i0 + ns)`.
//! - an m-tile = codes `[nkb][32 lanes][8 B]` then scales `[ceil(nkb / 4)][16 x u32]`.
//!
//! Bytes: codes are exactly the checkpoint's nibbles (no padding); scales are padded to a multiple of
//! 4 k-blocks per row (w2 at TP4 / TP8: 18 / 9 k-blocks -> 20 / 12 bytes per row).

use anyhow::{ensure, Result};

pub const CODES_UNIT: usize = 256;
pub const SCALES_UNIT: usize = 64;

/// Bytes of one m-tile block with `nkb` k-blocks of 32.
pub const fn mtile_bytes(nkb: usize) -> usize { nkb * CODES_UNIT + nkb.div_ceil(4) * SCALES_UNIT }

/// The geometry of one expert on one rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertGeom {
    /// model dim (5120)
    pub dim: usize,
    /// full inner width (2304)
    pub inter: usize,
    /// this rank's inner slice
    pub i0: usize,
    pub ns: usize,
}

impl ExpertGeom {
    pub fn whole(dim: usize, inter: usize) -> Self { ExpertGeom { dim, inter, i0: 0, ns: inter } }
    /// rank `r` of `n` inner-width slices (T0 §3.1: 2304 / 2, / 4, / 8; multiples of 32)
    pub fn slice(dim: usize, inter: usize, r: usize, n: usize) -> Result<Self> {
        ensure!(n > 0 && r < n && inter % n == 0 && (inter / n) % 32 == 0, "inner width {inter} does not split into {n} slices of 32-multiples");
        Ok(ExpertGeom { dim, inter, i0: r * (inter / n), ns: inter / n })
    }
    pub fn is_whole(&self) -> bool { self.i0 == 0 && self.ns == self.inter }
    pub fn validate(&self) -> Result<()> {
        ensure!(self.dim % 64 == 0 && self.inter % 32 == 0 && self.ns % 32 == 0 && self.ns > 0 && self.i0 % 32 == 0 && self.i0 + self.ns <= self.inter,
                "bad expert geometry {self:?} (dim % 64, slices of 32)");
        Ok(())
    }
    pub fn nkb1(&self) -> usize { self.dim / 32 }
    pub fn nkb2(&self) -> usize { self.ns / 32 }
    pub fn w13_mtile(&self) -> usize { mtile_bytes(self.nkb1()) }
    pub fn w13_bytes(&self) -> usize { self.w13_mtile() * (self.ns / 8) }
    pub fn w2_mtile(&self) -> usize { mtile_bytes(self.nkb2()) }
    pub fn w2_bytes(&self) -> usize { self.w2_mtile() * (self.dim / 16) }
    /// checkpoint bytes of this slice (codes + scales) — what the call streams per routed expert
    pub fn raw_bytes(&self) -> usize { 3 * self.ns * self.dim / 2 + 3 * self.ns * self.dim / 32 }
}

/// One raw checkpoint matrix: `w` [n][k/2], `s` [n][k/32].
#[derive(Clone, Copy)]
pub struct RawMat<'a> { pub w: &'a [u8], pub s: &'a [u8], pub n: usize, pub k: usize }

impl<'a> RawMat<'a> {
    pub fn new(w: &'a [u8], s: &'a [u8], n: usize, k: usize) -> Result<Self> {
        ensure!(k % 32 == 0 && w.len() == n * k / 2 && s.len() == n * k / 32, "raw MXFP4 matrix: {} code bytes / {} scales for [{n}, {k}]", w.len(), s.len());
        Ok(RawMat { w, s, n, k })
    }
    fn row(&self, r: usize) -> (&'a [u8], &'a [u8]) { (&self.w[r * self.k / 2..(r + 1) * self.k / 2], &self.s[r * self.k / 32..(r + 1) * self.k / 32]) }
}

#[inline]
fn nib(row: &[u8], k: usize) -> u64 { let b = row[k >> 1]; (if k & 1 == 1 { b >> 4 } else { b & 15 }) as u64 }

/// Pack one m-tile: 16 rows (codes + scales of the FULL row), k-range `[k0, k0 + 32 nkb)`.
fn pack_mtile(out: &mut [u8], rows: &[(&[u8], &[u8]); 16], k0: usize, nkb: usize) {
    debug_assert_eq!(out.len(), mtile_bytes(nkb));
    for kb in 0..nkb {
        for lane in 0..32 {
            let (g, t) = (lane >> 2, lane & 3);
            let mut v = 0u64;
            for n in 0..16 {
                let (r, i) = (n >> 2, n & 3);
                let row = g + 8 * (r & 1);
                let k = kb * 32 + 4 * t + 16 * (r >> 1) + i;
                v |= nib(rows[row].0, k0 + k) << (4 * n);
            }
            out[kb * CODES_UNIT + lane * 8..kb * CODES_UNIT + lane * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
    }
    let sc = &mut out[nkb * CODES_UNIT..];
    sc.fill(0);
    for q in 0..nkb.div_ceil(4) {
        for g in 0..8 {
            for h in 0..2 {
                for v in 0..4 {
                    let kb = 4 * q + v;
                    if kb < nkb { sc[q * SCALES_UNIT + (2 * g + h) * 4 + v] = rows[g + 8 * h].1[k0 / 32 + kb]; }
                }
            }
        }
    }
}

/// w1 + w3 of one expert -> `geom.w13_bytes()` bytes.
pub fn pack_w13(out: &mut [u8], w1: &RawMat, w3: &RawMat, geom: &ExpertGeom) -> Result<()> {
    geom.validate()?;
    ensure!(w1.n == geom.inter && w3.n == geom.inter && w1.k == geom.dim && w3.k == geom.dim, "w1 / w3 must be [{}, {}]", geom.inter, geom.dim);
    ensure!(out.len() == geom.w13_bytes(), "w13 buffer {} != {}", out.len(), geom.w13_bytes());
    let mb = geom.w13_mtile();
    for j in 0..geom.ns / 8 {
        let rows: [(&[u8], &[u8]); 16] = std::array::from_fn(|x| {
            let r = geom.i0 + 8 * j + (x & 7);
            if x < 8 { w1.row(r) } else { w3.row(r) }
        });
        pack_mtile(&mut out[j * mb..(j + 1) * mb], &rows, 0, geom.nkb1());
    }
    Ok(())
}

/// w2 of one expert (columns of this rank's slice) -> `geom.w2_bytes()` bytes.
pub fn pack_w2(out: &mut [u8], w2: &RawMat, geom: &ExpertGeom) -> Result<()> {
    geom.validate()?;
    ensure!(w2.n == geom.dim && w2.k == geom.inter, "w2 must be [{}, {}]", geom.dim, geom.inter);
    ensure!(out.len() == geom.w2_bytes(), "w2 buffer {} != {}", out.len(), geom.w2_bytes());
    let mb = geom.w2_mtile();
    for j in 0..geom.dim / 16 {
        let rows: [(&[u8], &[u8]); 16] = std::array::from_fn(|x| w2.row(16 * j + x));
        pack_mtile(&mut out[j * mb..(j + 1) * mb], &rows, geom.i0, geom.nkb2());
    }
    Ok(())
}

/// Unpack one weight code back from an m-tile (tests: the layout round-trips).
pub fn unpack_code(tile: &[u8], row: usize, k: usize) -> u8 {
    let kb = k / 32;
    let kk = k % 32;
    let (half16, i) = (kk / 16, kk % 4);
    let t = (kk % 16) / 4;
    let g = row % 8;
    let r = (row / 8) + 2 * half16;
    let lane = g * 4 + t;
    let n = 4 * r + i;
    let v = u64::from_le_bytes(tile[kb * CODES_UNIT + lane * 8..kb * CODES_UNIT + lane * 8 + 8].try_into().unwrap());
    ((v >> (4 * n)) & 15) as u8
}
pub fn unpack_scale(tile: &[u8], nkb: usize, row: usize, kb: usize) -> u8 {
    let (g, h) = (row % 8, row / 8);
    tile[nkb * CODES_UNIT + (kb / 4) * SCALES_UNIT + (2 * g + h) * 4 + kb % 4]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..n).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 24) as u8 }).collect()
    }

    #[test]
    fn dsv41_mx1_layout_round_trips_every_code_and_scale() {
        let (dim, inter) = (128, 96);
        let w1 = rng_bytes(inter * dim / 2, 1); let s1 = rng_bytes(inter * dim / 32, 2);
        let w3 = rng_bytes(inter * dim / 2, 3); let s3 = rng_bytes(inter * dim / 32, 4);
        let w2 = rng_bytes(dim * inter / 2, 5); let s2 = rng_bytes(dim * inter / 32, 6);
        let (m1, m3, m2) = (RawMat::new(&w1, &s1, inter, dim).unwrap(), RawMat::new(&w3, &s3, inter, dim).unwrap(), RawMat::new(&w2, &s2, dim, inter).unwrap());
        for (r, n) in [(0, 1), (1, 3), (2, 3)] {
            let g = ExpertGeom::slice(dim, inter, r, n).unwrap();
            let mut p13 = vec![0u8; g.w13_bytes()];
            let mut p2 = vec![0u8; g.w2_bytes()];
            pack_w13(&mut p13, &m1, &m3, &g).unwrap();
            pack_w2(&mut p2, &m2, &g).unwrap();
            let raw = |m: &RawMat, row: usize, k: usize| nib(m.row(row).0, k) as u8;
            for j in 0..g.ns / 8 {
                let tile = &p13[j * g.w13_mtile()..(j + 1) * g.w13_mtile()];
                for row in 0..16 {
                    let (m, rr) = if row < 8 { (&m1, g.i0 + 8 * j + row) } else { (&m3, g.i0 + 8 * j + row - 8) };
                    for k in 0..dim { assert_eq!(unpack_code(tile, row, k), raw(m, rr, k), "w13 tile {j} row {row} k {k}"); }
                    for kb in 0..dim / 32 { assert_eq!(unpack_scale(tile, g.nkb1(), row, kb), m.row(rr).1[kb]); }
                }
            }
            for j in 0..dim / 16 {
                let tile = &p2[j * g.w2_mtile()..(j + 1) * g.w2_mtile()];
                for row in 0..16 {
                    for k in 0..g.ns { assert_eq!(unpack_code(tile, row, k), raw(&m2, 16 * j + row, g.i0 + k)); }
                    for kb in 0..g.nkb2() { assert_eq!(unpack_scale(tile, g.nkb2(), row, kb), m2.row(16 * j + row).1[g.i0 / 32 + kb]); }
                }
            }
        }
    }

    #[test]
    fn dsv41_mx1_sizes_match_the_checkpoint_bytes() {
        for n in [1, 2, 4, 8] {
            let g = ExpertGeom::slice(5120, 2304, 0, n).unwrap();
            // codes are exact; only the w2 scales pad to whole quads at TP4 / TP8
            let codes = 3 * g.ns * g.dim / 2;
            let scales13 = 2 * g.ns * (g.dim / 32);
            let scales2 = g.dim * (g.nkb2().div_ceil(4) * 4);
            assert_eq!(g.w13_bytes() + g.w2_bytes(), codes + scales13 + scales2);
        }
        assert!(ExpertGeom::slice(5120, 2304, 0, 16).is_err(), "144 is not a multiple of 32");
    }
}
