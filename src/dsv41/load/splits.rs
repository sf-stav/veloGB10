//! DSV4.1 loader (leg LS): the pure slice arithmetic of the TP split (PLAN/DSV41_TP_DESIGN.md §3.1,
//! §3.3, §3.5, §3.6).
//!
//! Two things live here and nothing else (no file, no census, no GPU):
//!
//! * [`widths`] — how many units (a 128-wide Hadamard block for EXL3, a 32-wide quant group for the
//!   native formats, a head, a group) rank r of a w-way split holds, with T0's rotation by an id so
//!   that the extra block of an uneven split lands on a different rank for neighbouring experts
//!   (TP4: 18 blocks = (5,4,5,4) for even experts, (4,5,4,5) for odd; TP8: (3,2,2,2,3,2,2,2) rotated
//!   by `id mod 4`);
//! * [`SliceSpec`] — one regular strided gather out of a tensor's bytes: `outer` runs, one per
//!   position of the dimensions before the cut axis, each `len * inner` bytes long, `axis * inner`
//!   bytes apart. A cut along dim 0 is one contiguous run (`outer == 1`); a cut along dim 1 of a
//!   `[k/16][n/16][words]` trellis is one run per tile row (the shape `shard_plan::Piece` already
//!   ships for Naive's expert TP); the Engram `wkv` input cut (`[3 n-gram orders][8 heads][256]`) is
//!   `outer == 3`.

use anyhow::{bail, ensure, Result};

/// The EXL3 Hadamard block (elements): every cut of an EXL3 module's logical in / out dimension is a
/// multiple of this (`src/exl3_forward/reprime.rs`: "per 128-element row block").
pub const HAD: u64 = 128;
/// The native formats' quantization group (FP8 32 x 32 blocks, MXFP4 /32): cuts are multiples of it.
pub const QGROUP: u64 = 32;

/// Units held by each rank of a `world`-way split of `units` units. The remainder (`units % world`)
/// is spread one unit per rank, on every `world / rem`-th rank starting from `id` mod that spacing,
/// so neighbouring ids (experts) put their extra unit on different ranks. Always sums to `units`.
pub fn widths(units: u64, world: usize, id: usize) -> Vec<u64> {
    let w = world.max(1) as u64;
    let (base, rem) = (units / w, units % w);
    let mut v = vec![base; world.max(1)];
    if rem == 0 { return v; }
    if w % rem == 0 {
        let s = (w / rem) as usize;
        for (r, x) in v.iter_mut().enumerate() { if (r + id) % s == 0 { *x += 1; } }
    } else {
        // no regular spacing: the first `rem` ranks of the rotation starting at `id`
        for (r, x) in v.iter_mut().enumerate() { if ((r + world - id % world) % world) < rem as usize { *x += 1; } }
    }
    v
}

/// `(start, len)` in units of rank `rank`.
pub fn part(units: u64, world: usize, rank: usize, id: usize) -> (u64, u64) {
    let ws = widths(units, world, id);
    (ws[..rank].iter().sum(), ws[rank])
}

/// The vocabulary range `[a, b)` rank `rank` holds of the LM head and the DSpark Markov head: whole
/// 128-row blocks (the EXL3 output Hadamard needs complete blocks), `vocab / 128 = 1010` of them, so
/// TP2 is 505 + 505, TP4 (253, 253, 252, 252), TP8 (127, 127, 126 x 6). Differs from the preliminary
/// `rankmap::RankSlot::vocab` (equal rows) at TP4 / TP8; the TP leg takes THIS range.
pub fn vocab_range(vocab: usize, world: usize, rank: usize) -> std::ops::Range<usize> {
    let h = HAD as usize;
    if vocab % h != 0 { // not block-aligned (a slice pack): equal rows
        let (a, n) = part(vocab as u64, world, rank, 0);
        return a as usize..(a + n) as usize;
    }
    let (a, n) = part((vocab / h) as u64, world, rank, 0);
    a as usize * h..(a + n) as usize * h
}

/// One regular strided gather of a tensor's bytes (see the module doc).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceSpec {
    /// number of runs
    pub outer: u64,
    /// positions along the cut axis (full)
    pub axis: u64,
    /// bytes per position along the cut axis
    pub inner: u64,
    /// first held position
    pub start: u64,
    /// held positions
    pub len: u64,
}

impl SliceSpec {
    /// Bytes this slice holds.
    pub fn bytes(&self) -> u64 { self.outer * self.len * self.inner }
    /// Bytes of the whole tensor the slice is cut from.
    pub fn full_bytes(&self) -> u64 { self.outer * self.axis * self.inner }
    pub fn is_whole(&self) -> bool { self.len == self.axis }
    /// `(offset from the tensor's first byte, rows, stride, total bytes)` as `shard_plan::Piece` wants them
    /// (`rows <= 1` = one contiguous range).
    pub fn geometry(&self) -> (u64, u64, u64, u64) {
        let off = self.start * self.inner;
        if self.outer <= 1 || self.is_whole() { (off, 1, 0, self.bytes()) } else { (off, self.outer, self.axis * self.inner, self.bytes()) }
    }
}

/// Which logical dimension of a linear a cut runs along.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis { Out, In }

/// A cut of a module's logical dimension, in ELEMENTS: positions `[start, start + len)` of every one
/// of `groups` equal sub-ranges of the `total` elements (`groups == 1` is a plain range). `groups > 1`
/// is the Engram `wkv` input: `[3 n-gram orders][8 heads x 256]`, a rank keeps its heads of every order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cut {
    pub axis: Axis,
    pub total: u64,
    pub groups: u64,
    pub start: u64,
    pub len: u64,
}

impl Cut {
    pub fn range(axis: Axis, total: u64, start: u64, len: u64) -> Cut { Cut { axis, total, groups: 1, start, len } }
    /// Elements held in total.
    pub fn held(&self) -> u64 { self.groups * self.len }
    /// Elements per group.
    pub fn group_total(&self) -> u64 { self.total / self.groups }
}

/// The tile (16) and element (2 bytes) arithmetic of one EXL3 / plain / FP8 / FP4 tensor under a cut.
/// Returns the slice and the held shape; `Ok(None)` when the cut leaves this tensor whole (it is on the
/// other axis, or it is a scalar). Every shape is checked against `cut.total` — a mismatch is a bug in
/// the caller's table, never a silent wrong slice.
pub fn cut_tensor(kind: TensorKind, shape: &[u64], elt: u64, cut: &Cut) -> Result<Option<(SliceSpec, Vec<u64>)>> {
    ensure!(cut.groups >= 1 && cut.total % cut.groups == 0 && cut.start + cut.len <= cut.group_total(), "bad cut {cut:?}");
    let g = cut.groups;
    let gt = cut.group_total();
    let held_shape = |dim: usize, held_units: u64| { let mut s = shape.to_vec(); s[dim] = held_units; s };
    match kind {
        TensorKind::Exl3Trellis => {
            ensure!(shape.len() == 3, "trellis shape {shape:?}");
            let (kb, nb, words) = (shape[0], shape[1], shape[2]);
            match cut.axis {
                Axis::In => {
                    ensure!(kb * 16 == cut.total, "trellis in dim {} tiles vs cut total {}", kb, cut.total);
                    ensure!(cut.start % 16 == 0 && cut.len % 16 == 0 && gt % 16 == 0, "in cut not tile aligned {cut:?}");
                    let spec = SliceSpec { outer: g, axis: gt / 16, inner: nb * words * elt, start: cut.start / 16, len: cut.len / 16 };
                    Ok(Some((spec, held_shape(0, g * cut.len / 16))))
                }
                Axis::Out => {
                    ensure!(g == 1, "grouped out cut not supported");
                    ensure!(nb * 16 == cut.total, "trellis out dim {} tiles vs cut total {}", nb, cut.total);
                    ensure!(cut.start % 16 == 0 && cut.len % 16 == 0, "out cut not tile aligned {cut:?}");
                    let spec = SliceSpec { outer: kb, axis: nb, inner: words * elt, start: cut.start / 16, len: cut.len / 16 };
                    Ok(Some((spec, held_shape(1, cut.len / 16))))
                }
            }
        }
        // F16 vector over the module's input (suh) / output (svh)
        TensorKind::Exl3Suh | TensorKind::Exl3Svh => {
            let mine = (kind == TensorKind::Exl3Suh) == (cut.axis == Axis::In);
            if !mine { return Ok(None); }
            ensure!(shape.len() == 1 && shape[0] == cut.total, "vector shape {shape:?} vs cut total {}", cut.total);
            Ok(Some((SliceSpec { outer: g, axis: gt, inner: elt, start: cut.start, len: cut.len }, vec![g * cut.len])))
        }
        TensorKind::Whole => Ok(None),
        // a 1-D parameter along the output (attention sink per head, a bias)
        TensorKind::Vector => {
            if cut.axis != Axis::Out { return Ok(None); }
            ensure!(shape.len() == 1 && shape[0] == cut.total && g == 1, "vector shape {shape:?} vs cut total {}", cut.total);
            Ok(Some((SliceSpec { outer: 1, axis: cut.total, inner: elt, start: cut.start, len: cut.len }, vec![cut.len])))
        }
        // a 2-D matrix [out_blocks, in_blocks_stored]: `bo` output rows and `bi` input elements per stored cell
        // (plain weight bo = bi = 1; FP8 scale 32 x 32; MXFP4 weight bo = 1, bi = 1/2 (two elements per byte)
        // is expressed by `pack`), `pack` elements per stored cell along the input
        TensorKind::Matrix { bo, bi, pack } => {
            ensure!(shape.len() == 2, "matrix shape {shape:?}");
            let (rows, cols) = (shape[0], shape[1]);
            match cut.axis {
                Axis::Out => {
                    ensure!(g == 1 && rows * bo == cut.total, "matrix rows {} x {bo} vs cut total {}", rows, cut.total);
                    ensure!(cut.start % bo == 0 && cut.len % bo == 0, "out cut not block aligned {cut:?} bo {bo}");
                    Ok(Some((SliceSpec { outer: 1, axis: rows, inner: cols * elt, start: cut.start / bo, len: cut.len / bo }, held_shape(0, cut.len / bo))))
                }
                Axis::In => {
                    let cell = bi * pack;
                    ensure!(cols * cell == cut.total, "matrix cols {} x {cell} vs cut total {}", cols, cut.total);
                    ensure!(gt % cell == 0 && cut.start % cell == 0 && cut.len % cell == 0, "in cut not cell aligned {cut:?} cell {cell}");
                    let spec = SliceSpec { outer: rows * g, axis: gt / cell, inner: elt, start: cut.start / cell, len: cut.len / cell };
                    Ok(Some((spec, held_shape(1, g * cut.len / cell))))
                }
            }
        }
    }
}

/// How a tensor's bytes relate to its module's logical (out, in) dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorKind {
    Exl3Trellis,
    Exl3Suh,
    Exl3Svh,
    /// 1-D parameter indexed by output position
    Vector,
    /// 2-D `[out / bo][in / (bi * pack)]` cells
    Matrix { bo: u64, bi: u64, pack: u64 },
    /// never cut (mul1, second-level scales, norms, anything replicated)
    Whole,
}

/// Bytes per element of a safetensors dtype.
pub fn elt_bytes(dtype: &str) -> Result<u64> {
    Ok(match dtype {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" => 1,
        d => bail!("dtype {d} has no known element size"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t0_patterns_for_the_expert_inner_width() {
        // 2304 / 128 = 18 blocks
        assert_eq!(widths(18, 2, 0), vec![9, 9]);
        assert_eq!(widths(18, 4, 0), vec![5, 4, 5, 4]);
        assert_eq!(widths(18, 4, 1), vec![4, 5, 4, 5]);
        assert_eq!(widths(18, 4, 6), vec![5, 4, 5, 4]);
        assert_eq!(widths(18, 8, 0), vec![3, 2, 2, 2, 3, 2, 2, 2]);
        // rotation by id mod 4: the 3s move one rank down per id (id 1: ranks 3 and 7)
        assert_eq!(widths(18, 8, 1), vec![2, 2, 2, 3, 2, 2, 2, 3]);
        for id in 0..16 { for w in [1usize, 2, 4, 8] { assert_eq!(widths(18, w, id).iter().sum::<u64>(), 18, "w{w} id{id}"); } }
        // over four consecutive experts every TP8 rank takes the 3 exactly once: the busiest rank reads +8 % at one row
        let mut threes = [0u32; 8];
        for id in 0..4 { for (r, x) in widths(18, 8, id).iter().enumerate() { if *x == 3 { threes[r] += 1; } } }
        assert_eq!(threes, [1; 8]);
        // an irregular remainder still sums
        for u in 1..60u64 { for w in [3usize, 5, 6, 7] { for id in 0..9 { assert_eq!(widths(u, w, id).iter().sum::<u64>(), u); } } }
    }

    #[test]
    fn the_vocab_ranges_are_whole_hadamard_blocks_and_tile_the_head() {
        for w in [1usize, 2, 4, 8] {
            let mut next = 0;
            for r in 0..w {
                let v = vocab_range(129280, w, r);
                assert_eq!(v.start, next);
                assert_eq!(v.start % 128, 0, "w{w} r{r}");
                assert_eq!(v.end % 128, 0);
                next = v.end;
            }
            assert_eq!(next, 129280);
        }
        assert_eq!(vocab_range(129280, 2, 1), 64640..129280);
        assert_eq!(vocab_range(129280, 4, 0).len(), 253 * 128);
        assert_eq!(vocab_range(129280, 8, 7).len(), 126 * 128);
    }

    /// Every rank's slices of one tensor tile its bytes exactly once (the Naive test's shape, on the real V4.1 geometry).
    fn tiles_once(kind: TensorKind, shape: &[u64], elt: u64, cuts: &[Cut]) {
        let full: u64 = shape.iter().product::<u64>() * elt;
        let mut covered = vec![0u8; full as usize];
        let mut held = 0;
        for c in cuts {
            let (spec, hs) = cut_tensor(kind, shape, elt, c).unwrap().expect("cut");
            assert_eq!(hs.iter().product::<u64>() * elt, spec.bytes(), "held shape {hs:?} vs {spec:?}");
            assert_eq!(spec.full_bytes(), full);
            let (off, rows, stride, total) = spec.geometry();
            let row_len = total / rows;
            for r in 0..rows { for b in 0..row_len { covered[(off + r * stride + b) as usize] += 1; } }
            held += spec.bytes();
        }
        assert_eq!(held, full);
        assert!(covered.iter().all(|c| *c == 1), "{kind:?} {shape:?}: not every byte exactly once");
    }

    #[test]
    fn expert_cuts_tile_every_tensor_once_at_every_world() {
        // scaled-down copies of the V4.1 expert geometry: in 5120 -> 256, inner 2304 -> 18 blocks of 128 -> use real block counts
        for w in [2usize, 4, 8] {
            for id in 0..3 {
                let bl = widths(18, w, id);
                let mut start = 0;
                let (mut gate, mut down, mut svh, mut suh2) = (vec![], vec![], vec![], vec![]);
                for r in 0..w {
                    let (a, n) = (start * 128, bl[r] * 128);
                    start += bl[r];
                    gate.push(Cut::range(Axis::Out, 2304, a, n));
                    down.push(Cut::range(Axis::In, 2304, a, n));
                    svh.push(Cut::range(Axis::Out, 2304, a, n));
                    suh2.push(Cut::range(Axis::In, 2304, a, n));
                }
                // w1 / w3 trellis [320, 144, 3 x 16 words] is big: shrink the in dim to 2 tile rows for the byte-level check
                tiles_once(TensorKind::Exl3Trellis, &[2, 144, 6], 2, &gate.iter().map(|c| Cut { total: 2304, ..*c }).collect::<Vec<_>>());
                let _ = svh;
                tiles_once(TensorKind::Exl3Trellis, &[144, 2, 6], 2, &down);
                tiles_once(TensorKind::Exl3Svh, &[2304], 2, &gate);
                tiles_once(TensorKind::Exl3Suh, &[2304], 2, &suh2);
            }
        }
    }

    #[test]
    fn out_cut_of_a_trellis_is_a_strided_gather_and_an_in_cut_is_contiguous() {
        let c = Cut::range(Axis::Out, 2304, 128 * 5, 128 * 4);
        let (s, shape) = cut_tensor(TensorKind::Exl3Trellis, &[320, 144, 48], 2, &c).unwrap().unwrap();
        assert_eq!((s.outer, s.axis, s.start, s.len), (320, 144, 40, 32));
        assert_eq!(shape, vec![320, 32, 48]);
        let (off, rows, stride, total) = s.geometry();
        assert_eq!((off, rows, stride, total), (40 * 96, 320, 144 * 96, 320 * 32 * 96));
        let c = Cut::range(Axis::In, 2304, 128 * 5, 128 * 4);
        let (s, shape) = cut_tensor(TensorKind::Exl3Trellis, &[144, 320, 48], 2, &c).unwrap().unwrap();
        assert_eq!(s.geometry(), (40 * 320 * 96, 1, 0, 32 * 320 * 96));
        assert_eq!(shape, vec![32, 320, 48]);
        // the other axis's vectors are left whole
        assert!(cut_tensor(TensorKind::Exl3Suh, &[5120], 2, &Cut::range(Axis::In, 2304, 0, 128)).is_err(), "a wrong total is refused");
        assert!(cut_tensor(TensorKind::Exl3Suh, &[5120], 2, &Cut::range(Axis::Out, 2304, 0, 128)).unwrap().is_none());
        assert!(cut_tensor(TensorKind::Exl3Svh, &[2304], 2, &Cut::range(Axis::In, 2304, 0, 128)).unwrap().is_none());
    }

    #[test]
    fn the_engram_wkv_input_cut_keeps_a_ranks_heads_of_every_ngram_order() {
        // TP2 rank 1: heads 4..8 of 8, 256 elements per head, 3 orders: in = 6144 = [3][8][256]
        let cut = Cut { axis: Axis::In, total: 6144, groups: 3, start: 4 * 256, len: 4 * 256 };
        let (s, shape) = cut_tensor(TensorKind::Exl3Trellis, &[384, 1600, 80], 2, &cut).unwrap().unwrap();
        assert_eq!((s.outer, s.axis, s.start, s.len), (3, 128, 64, 64));
        assert_eq!(shape, vec![192, 1600, 80]);
        tiles_once(TensorKind::Exl3Suh, &[6144], 2, &[
            Cut { axis: Axis::In, total: 6144, groups: 3, start: 0, len: 1024 }, Cut { axis: Axis::In, total: 6144, groups: 3, start: 1024, len: 1024 }]);
    }

    #[test]
    fn native_matrices_cut_rows_contiguously_and_columns_strided() {
        // FP8 32 x 32 weight [2304, 5120] cut on rows; MXFP4 expert w2 [5120, 2304 / 2] bytes cut on the input
        let (s, sh) = cut_tensor(TensorKind::Matrix { bo: 1, bi: 1, pack: 1 }, &[2304, 5120], 1, &Cut::range(Axis::Out, 2304, 576, 576)).unwrap().unwrap();
        assert_eq!(s.geometry(), (576 * 5120, 1, 0, 576 * 5120));
        assert_eq!(sh, vec![576, 5120]);
        let w2 = TensorKind::Matrix { bo: 1, bi: 1, pack: 2 };
        let (s, sh) = cut_tensor(w2, &[5120, 1152], 1, &Cut::range(Axis::In, 2304, 576, 576)).unwrap().unwrap();
        assert_eq!((s.outer, s.axis, s.start, s.len), (5120, 1152, 288, 288));
        assert_eq!(sh, vec![5120, 288]);
        // its scale [5120, 72] (one E8M0 per 32 inputs) cut by the same input range
        let sc = TensorKind::Matrix { bo: 1, bi: 32, pack: 1 };
        let (s, _) = cut_tensor(sc, &[5120, 72], 1, &Cut::range(Axis::In, 2304, 576, 576)).unwrap().unwrap();
        assert_eq!((s.axis, s.start, s.len), (72, 18, 18));
        // an FP8 block scale [160, 72] (32 x 32) on the input; and wq_b's rows by head
        let sc = TensorKind::Matrix { bo: 32, bi: 32, pack: 1 };
        let (s, _) = cut_tensor(sc, &[160, 72], 1, &Cut::range(Axis::In, 2304, 576, 576)).unwrap().unwrap();
        assert_eq!((s.outer, s.start, s.len), (160, 18, 18));
        assert!(cut_tensor(TensorKind::Matrix { bo: 32, bi: 32, pack: 1 }, &[1024, 40], 1, &Cut::range(Axis::Out, 32768, 16 * 512, 16 * 512)).unwrap().is_some());
    }
}
