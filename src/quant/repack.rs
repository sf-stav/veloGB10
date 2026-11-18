//! MMA weight repack (the Marlin permutation for `mma.sync.m16n8k16`) and weight fusion / row subsets.
//! Siblings: `mod.rs` has the NVFP4 / FP8 tensor layouts these permute; `mod.rs` holds the directory map.

use super::*;

// ============================ MMA weight repack (the Marlin permutation) ============================
//
// The decode/verify GEMM runs on `mma.sync.m16n8k16` tensor cores, which demand that each lane of a
// warp already hold *specific* elements of the 16x16 A-tile in *specific* registers. Storing weights
// row-major forces every lane to gather its 4 elements from 2 rows K/2 bytes apart — 8 scattered
// sectors per load instruction. So we permute ONCE, at load, into exactly the order the fragment
// wants: lane L's whole A-fragment becomes ONE contiguous aligned load, and a warp's tile is one
// contiguous run of bytes. This is the single change that turns the load from a gather into a stream.
//
// The mma A-fragment for m16n8k16 (PTX ISA), with g = lane>>2 and t = lane&3, is 8 elements in 4
// 32-bit registers holding bf16 pairs:
//
//     ra[0] = { A[g  ][2t  ], A[g  ][2t+1] }        ra[2] = { A[g  ][2t+8], A[g  ][2t+9] }
//     ra[1] = { A[g+8][2t  ], A[g+8][2t+1] }        ra[3] = { A[g+8][2t+8], A[g+8][2t+9] }
//
// so lane L wants two rows (g, g+8) x two column-pairs (2t, 2t+8) of the tile — and nothing else.
//
// The K-tile is 16 wide, which is EXACTLY the NVFP4 scale-block size. One mma step therefore consumes
// exactly one scale block per row, so the block scale is a constant over the step and can be folded
// into the A-fragment before the mma — a per-*weight* cost that does not scale with N. That is the
// whole reason the kernel is flat in N.

/// M rows and K columns per MMA tile. Both fixed by the `m16n8k16` fragment shape.
pub const MMA_M: usize = 16;
pub const MMA_K: usize = 16;
/// Bytes per repacked tile: 16x16 nibbles = 128 B for NVFP4, 16x16 bytes = 256 B for FP8.
pub const MMA_TILE_FP4: usize = MMA_M * MMA_K / 2;
pub const MMA_TILE_FP8: usize = MMA_M * MMA_K;

/// Where element (row r, col c) of a tile lives inside a repacked NVFP4 tile.
/// Returns (byte offset within the 128-byte tile, true if the value is in the HIGH nibble).
#[inline]
pub(super) fn fp4_tile_slot(r: usize, c: usize) -> (usize, bool) {
    let (g, hi_row) = (r & 7, r >> 3);         // lane group, and which of the fragment's two rows
    let (t, hi_col) = ((c & 7) >> 1, c >> 3);  // lane within group, and which column-pair
    let lane = g * 4 + t;
    let j = hi_row | (hi_col << 1);            // ra[j]: bit0 = row+8, bit1 = col+8
    (lane * 4 + j, (c & 1) == 1)               // odd column = high nibble (packing convention)
}

/// Where element (row r, col c) of a tile lives inside a repacked FP8 tile (byte offset).
#[inline]
pub(super) fn fp8_tile_slot(r: usize, c: usize) -> usize {
    let (g, hi_row) = (r & 7, r >> 3);
    let (t, hi_col) = ((c & 7) >> 1, c >> 3);
    let lane = g * 4 + t;
    let j = (c & 1) | (hi_row << 1) | (hi_col << 2);
    lane * 8 + j
}

/// Permute a row-major NVFP4 tensor into MMA tile order.
///
/// In:  `qw` [M, K/2] packed nibbles (even col = low nibble), `sc` [M, K/16] E4M3 block scales.
/// Out: `wt` [M/16 * K/16 * 128] tiles, `st` [M/16 * K/16 * 16] scales (one per row of the tile).
///
/// Panics on M%16 or K%16 — every reduction and output dim in this model family is a multiple of 16,
/// and silently falling back to a slow path is how a "fast" engine ends up not being one.
pub fn repack_nvfp4_mma(qw: &[u8], sc: &[u8], m: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    assert!(m % MMA_M == 0 && k % MMA_K == 0, "MMA repack needs M,K % 16 == 0 (got {}x{})", m, k);
    // The GEMM walks k-blocks in ADJACENT PAIRS (so a warp's two scale reads land in one 32-byte DRAM
    // sector instead of wasting half of two). An odd k-block count would make it silently skip the last
    // block -- a wrong answer that still looks like a model. Every K in this family is a multiple of 32.
    assert!(k % 32 == 0, "the paired-k GEMM needs K % 32 == 0 (got K={})", k);
    let (ntm, nblk) = (m / MMA_M, k / MMA_K);
    let mut wt = vec![0u8; ntm * nblk * MMA_TILE_FP4];
    let mut st = vec![0u8; ntm * nblk * MMA_M];
    repack_driver(ntm * nblk, |t, wtc, stc| {
        let (mt, kb) = (t / nblk, t % nblk);
        repack_fp4_tile(qw, sc, k, nblk, mt, kb, wtc, stc);
    }, &mut wt, &mut st);
    (wt, st)
}

/// One (mt, kb) tile of the NVFP4 repack: 128 B of packed weights + 16 B of scales. Factored out so
/// the sequential and threaded drivers emit the same bytes through the SAME code path.
#[inline]
fn repack_fp4_tile(qw: &[u8], sc: &[u8], k: usize, nblk: usize, mt: usize, kb: usize,
                   wt_tile: &mut [u8], st_tile: &mut [u8]) {
    for r in 0..MMA_M {
        let row = mt * MMA_M + r;
        st_tile[r] = sc[row * nblk + kb];
        // Copy a byte at a time: source and destination pack the same (even,odd) column pair
        // into the same nibble positions, so whole bytes move without re-nibbling.
        for cp in 0..(MMA_K / 2) {
            let c = cp * 2;
            let (off, _) = fp4_tile_slot(r, c);
            wt_tile[off] = qw[row * (k / 2) + (kb * MMA_K + c) / 2];
        }
    }
}

/// Permute a row-major FP8 tensor [M, K] into MMA tile order. Row scales are unchanged: FP8 scales
/// are per output row, constant over K, so they fold into the f32 accumulator once at the end.
pub fn repack_fp8_mma(qw: &[u8], m: usize, k: usize) -> Vec<u8> {    assert!(m % MMA_M == 0 && k % MMA_K == 0, "MMA repack needs M,K % 16 == 0 (got {}x{})", m, k);
    let (ntm, nblk) = (m / MMA_M, k / MMA_K);
    let mut wt = vec![0u8; ntm * nblk * MMA_TILE_FP8];
    repack_driver(ntm * nblk, |t, wtc, _| {
        let (mt, kb) = (t / nblk, t % nblk);
        repack_fp8_tile(qw, k, mt, kb, wtc);
    }, &mut wt, &mut []);
    wt
}

/// One (mt, kb) tile of the FP8 repack: 256 B of weights (no per-tile scales — see above).
#[inline]
fn repack_fp8_tile(qw: &[u8], k: usize, mt: usize, kb: usize, wt_tile: &mut [u8]) {
    for r in 0..MMA_M {
        let row = mt * MMA_M + r;
        for c in 0..MMA_K {
            wt_tile[fp8_tile_slot(r, c)] = qw[row * k + kb * MMA_K + c];
        }
    }
}

/// Quantize a bf16 weight [M, K] into the `gemm_dsv4_fp8_bsb` device format (the engine's
/// `Fp8Weight`): MMA-repacked e4m3 codes + UE8M0 block scales [M/128, K/128] (row-major,
/// matching the kernel's `Sb[(mt>>3)*nkb + kb]`). Per 128-row × 128-K block: amax (floored
/// at 1e-4) → s = 2^ceil(log2(amax/448)) (UE8M0, always up — the §C.1 convention) →
/// codes = e4m3(x/s). One-time load-path quantizer (the DSpark fp8 draft heads behind
/// --dspark-fp8-logits) — NOT the serving GEMM's weights (those ship quantized).
/// Returns (repacked codes, sb bytes).
pub fn quantize_fp8_bsb(w: &[bf16], m: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(w.len(), m * k, "shape mismatch");
    assert!(m % 128 == 0 && k % 128 == 0, "fp8_bsb quantize needs M,K % 128 (got {m}x{k})");
    let nkb = k / 128;
    let nbm = m / 128;
    let mut codes = vec![0u8; m * k];
    let mut sb = vec![0u8; nbm * nkb];
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).max(1);
    let bms_per = nbm.div_ceil(nthreads).max(1);
    std::thread::scope(|sc| {
        for (t, (cp, sp)) in codes
            .chunks_mut(bms_per * 128 * k)
            .zip(sb.chunks_mut(bms_per * nkb))
            .enumerate()
        {
            sc.spawn(move || {
                let bm0 = t * bms_per;
                for bm in bm0..(bm0 + bms_per).min(nbm) {
                    let cl = &mut cp[(bm - bm0) * 128 * k..][..128 * k];
                    let sl = &mut sp[(bm - bm0) * nkb..][..nkb];
                    for bk in 0..nkb {
                        let mut amax = 0.0f32;
                        for r in 0..128 {
                            let row = &w[(bm * 128 + r) * k + bk * 128..][..128];
                            for x in row {
                                amax = amax.max(x.to_f32().abs());
                            }
                        }
                        amax = amax.max(1e-4);
                        let s = crate::dsv4_cpu::fast_round_scale(amax, 1.0 / 448.0);
                        sl[bk] = (s.to_bits() >> 23) as u8; // UE8M0 byte (s is a normal pow2)
                        let inv = 1.0 / s;
                        for r in 0..128 {
                            let src = &w[(bm * 128 + r) * k + bk * 128..][..128];
                            let dst = &mut cl[r * k + bk * 128..][..128];
                            for (d, x) in dst.iter_mut().zip(src) {
                                *d = f32_to_e4m3(x.to_f32() * inv);
                            }
                        }
                    }
                }
            });
        }
    });
    (repack_fp8_mma(&codes, m, k), sb)
}


/// Don't spawn a thread for less work than this: a thread costs tens of µs, 4096 tiles (~0.5 MB)
/// costs ~2-3 ms single-core — below that the spawn overhead is the optimization.
const REPACK_MIN_TILES_PER_THREAD: usize = 4096;

thread_local! {
    static REPACK_THREADING: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Kill switch for the per-call threading, per thread. The load pipeline's WORKERS set this off:
/// the worker pool IS the parallelism there — 8 workers each spawning 16 inner threads is a
/// 6x-oversubscribed thrash on 20 cores (measured: pipeline crawls). The shard-loop inline repack
/// and any serial caller keep the default (threaded, GPU idle during load).
pub fn set_repack_threading(on: bool) { REPACK_THREADING.with(|c| c.set(on)); }

/// Drive a per-tile repack body over `total_tiles` tiles. Small inputs run sequentially on the
/// caller; big inputs partition the tiles into disjoint contiguous ranges across `std::thread::scope`
/// workers. Every output byte is a pure function of the input bytes and the tile index, so the
/// threaded result is BYTE-IDENTICAL to the sequential one — that is the load-correctness contract.
fn repack_driver<F: Fn(usize, &mut [u8], &mut [u8]) + Sync>(
    total_tiles: usize, body: F, wt: &mut [u8], st: &mut [u8],
) {
    let wt_tile = wt.len() / total_tiles.max(1);
    let st_tile = st.len() / total_tiles.max(1);
    let max_t = if REPACK_THREADING.with(|c| c.get()) {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(16)
    } else { 1 };
    let nthreads = max_t.min(total_tiles.div_ceil(REPACK_MIN_TILES_PER_THREAD)).max(1);
    if nthreads <= 1 {
        for t in 0..total_tiles {
            body(t, &mut wt[t * wt_tile..(t + 1) * wt_tile],
                 if st_tile > 0 { &mut st[t * st_tile..(t + 1) * st_tile] } else { &mut [] });
        }
        return;
    }
    let per = total_tiles.div_ceil(nthreads);
    std::thread::scope(|s| {
        let mut wt_rest: &mut [u8] = wt;
        let mut st_rest: &mut [u8] = st;
        let mut begin = 0usize;
        while begin < total_tiles {
            let n = per.min(total_tiles - begin);
            let (wtc, wtl) = wt_rest.split_at_mut(n * wt_tile);
            let (stc, stl) = st_rest.split_at_mut(n * st_tile);
            wt_rest = wtl;
            st_rest = stl;
            let body = &body;
            s.spawn(move || {
                for t in begin..begin + n {
                    let i = t - begin;
                    body(t, &mut wtc[i * wt_tile..(i + 1) * wt_tile],
                         if st_tile > 0 { &mut stc[i * st_tile..(i + 1) * st_tile] } else { &mut [] });
                }
            });
            begin += n;
        }
    });
}

// ================================ Weight FUSION ================================
//
// Several projections in this architecture read the SAME activation and are separate tensors only
// because the checkpoint stores them that way:
//
//   GDN:       in_proj_qkv + in_proj_z + in_proj_b + in_proj_a   (all read the normed hidden)
//   Attention: q_proj + k_proj + v_proj                          (likewise)
//
// Running them as separate GEMMs is a disaster at the small end. `in_proj_b`/`in_proj_a` have
// M = num_value_heads (32 on 9B), so `grid = M/16 = 2` — TWO blocks on a 48-SM GPU. Measured: 26 us
// to move 74 KB (2.8 GB/s against a 234 GB/s machine), and across 24 GDN layers that is **4.7% of all
// GEMM time to move 0.03% of the bytes**. Concatenating the four along M turns four launches into one,
// lets the tiny tensors ride inside a big efficient kernel, and lengthens the surviving kernel (which
// also helps the ramp/tail problem). vLLM and SGLang fuse QKV and gate/up at load for the same reason.
//
// The one real constraint is quantization metadata. NVFP4's `global_scale` is PER TENSOR, so a fused
// weight has several. But every segment boundary here is a multiple of 16 (conv_dim, value_dim and
// num_heads all are), so each 16-row MMA tile lies entirely within ONE source tensor — and the scale
// can be a per-TILE lookup the kernel reads once per block. No requantization, no loss of precision.

/// Concatenate NVFP4 tensors along M (rows). All must share K.
///
/// Returns the row-major concatenation plus `gs_tile[M/16]`: the reciprocal global scale that applies
/// to each 16-row tile. Feed the first two through `repack_nvfp4_mma`.
pub fn fuse_nvfp4(parts: &[(&[u8], &[u8], f32, usize)], k: usize) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    assert!(k % BLOCK == 0);
    let (mut qw, mut sc, mut gs) = (Vec::new(), Vec::new(), Vec::new());
    for &(pq, ps, inv_gs, m) in parts {
        assert!(m % MMA_M == 0, "fused segment M={} must be a multiple of {}", m, MMA_M);
        assert_eq!(pq.len(), m * k / 2, "fused segment qweight size");
        assert_eq!(ps.len(), m * k / BLOCK, "fused segment scales size");
        qw.extend_from_slice(pq);
        sc.extend_from_slice(ps);
        gs.extend(std::iter::repeat(inv_gs).take(m / MMA_M));   // one entry per 16-row tile
    }
    (qw, sc, gs)
}

/// Concatenate FP8 tensors along M. FP8 scales are already per output row, so they just concatenate.
/// Concatenate FP8 tensors along M. `nb_k` selects the scheme and the per-part scale length:
/// 0 → legacy per-row (part scales are [part_m]); >0 → Qwen block-128 (part scales are
/// [(part_m/128)*nb_k]). Band scales concatenate exactly like rows do — segments are 128-row
/// aligned (asserted here and by the sharder), so fused band boundaries stay block-aligned.
pub fn fuse_fp8_len(parts: &[(&[u8], &[f32], usize)], k: usize, nb_k: usize) -> (Vec<u8>, Vec<f32>) {
    let (mut qw, mut rs) = (Vec::new(), Vec::new());
    for &(pq, prs, m) in parts {
        assert!(m % MMA_M == 0, "fused segment M={} must be a multiple of {}", m, MMA_M);
        assert_eq!(pq.len(), m * k, "fused segment fp8 size");
        let want = if nb_k > 0 { (m / 128) * nb_k } else { m };
        assert_eq!(prs.len(), want, "fused segment fp8 scale size (nb_k={nb_k})");
        qw.extend_from_slice(pq);
        rs.extend_from_slice(prs);
    }
    (qw, rs)
}

/// Legacy per-row entry point (scale length == m).
pub fn fuse_fp8(parts: &[(&[u8], &[f32], usize)], k: usize) -> (Vec<u8>, Vec<f32>) {
    fuse_fp8_len(parts, k, 0)
}

