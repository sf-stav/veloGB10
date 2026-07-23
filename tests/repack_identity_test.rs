//! Byte-identity of the threaded repack (load-speed campaign) vs the original sequential
//! byte-loop, kept here verbatim as an independent reference. A load-corruption in the MMA
//! permutation changes GEMM outputs immediately; this pins the permutation itself.

use gb10_inference::quant;

// ---- reference: the pre-threading implementation, duplicated on purpose ----

const REF_MMA_M: usize = 16;
const REF_MMA_K: usize = 16;

fn ref_fp4_tile_slot(r: usize, c: usize) -> (usize, bool) {
    let (g, hi_row) = (r & 7, r >> 3);
    let (t, hi_col) = ((c & 7) >> 1, c >> 3);
    let lane = g * 4 + t;
    let j = hi_row | (hi_col << 1);
    (lane * 4 + j, (c & 1) == 1)
}

fn ref_fp8_tile_slot(r: usize, c: usize) -> usize {
    let (g, hi_row) = (r & 7, r >> 3);
    let (t, hi_col) = ((c & 7) >> 1, c >> 3);
    let lane = g * 4 + t;
    let j = (c & 1) | (hi_row << 1) | (hi_col << 2);
    lane * 8 + j
}

fn ref_repack_nvfp4(qw: &[u8], sc: &[u8], m: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let (ntm, nblk) = (m / REF_MMA_M, k / REF_MMA_K);
    let mut wt = vec![0u8; ntm * nblk * (REF_MMA_M * REF_MMA_K / 2)];
    let mut st = vec![0u8; ntm * nblk * REF_MMA_M];
    for mt in 0..ntm {
        for kb in 0..nblk {
            let base = (mt * nblk + kb) * (REF_MMA_M * REF_MMA_K / 2);
            for r in 0..REF_MMA_M {
                let row = mt * REF_MMA_M + r;
                st[(mt * nblk + kb) * REF_MMA_M + r] = sc[row * nblk + kb];
                for cp in 0..(REF_MMA_K / 2) {
                    let c = cp * 2;
                    let (off, _) = ref_fp4_tile_slot(r, c);
                    wt[base + off] = qw[row * (k / 2) + (kb * REF_MMA_K + c) / 2];
                }
            }
        }
    }
    (wt, st)
}

fn ref_repack_fp8(qw: &[u8], m: usize, k: usize) -> Vec<u8> {
    let (ntm, nblk) = (m / REF_MMA_M, k / REF_MMA_K);
    let mut wt = vec![0u8; ntm * nblk * REF_MMA_M * REF_MMA_K];
    for mt in 0..ntm {
        for kb in 0..nblk {
            let base = (mt * nblk + kb) * REF_MMA_M * REF_MMA_K;
            for r in 0..REF_MMA_M {
                let row = mt * REF_MMA_M + r;
                for c in 0..REF_MMA_K {
                    wt[base + ref_fp8_tile_slot(r, c)] = qw[row * k + kb * REF_MMA_K + c];
                }
            }
        }
    }
    wt
}

// ---------------------------------------------------------------------------

#[test]
fn threaded_repack_byte_identical_nvfp4() {
    // Sizes: sequential path (<4096 tiles), just over the threading threshold, a real GDN
    // fused shape, and a big LM-head-class shape (16 threads, 100K+ tiles).
    for (m, k) in [(16usize, 32usize), (32, 4096), (256, 4096), (12352, 4096), (8192, 8192)] {
        let qw: Vec<u8> = (0..m * k / 2).map(|i| ((i * 131 + 7) ^ (i >> 8)) as u8).collect();
        let sc: Vec<u8> = (0..m * k / 16).map(|i| (i * 57 + 11) as u8).collect();
        let (w0, s0) = ref_repack_nvfp4(&qw, &sc, m, k);
        let (w1, s1) = quant::repack_nvfp4_mma(&qw, &sc, m, k);
        assert_eq!(w0, w1, "wt mismatch at {m}x{k}");
        assert_eq!(s0, s1, "st mismatch at {m}x{k}");
    }
}

#[test]
fn threaded_repack_byte_identical_fp8() {
    for (m, k) in [(16usize, 16usize), (32, 4096), (4096, 4096)] {
        let qw: Vec<u8> = (0..m * k).map(|i| ((i * 31 + 5) ^ (i >> 7)) as u8).collect();
        assert_eq!(ref_repack_fp8(&qw, m, k), quant::repack_fp8_mma(&qw, m, k), "fp8 mismatch at {m}x{k}");
    }
}
