//! Inline unit tests of the quant module, moved out of `quant.rs` verbatim (module `tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::*;

    #[test]
    fn e4m3_roundtrip_is_a_float_not_an_int() {
        // The bug that motivates this test: treating the E4M3 byte as an integer.
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert!((e4m3_to_f32(f32_to_e4m3(1.0)) - 1.0).abs() < 1e-6);
        assert!((e4m3_to_f32(f32_to_e4m3(448.0)) - 448.0).abs() < 1e-3);
        // A code whose integer value bears no relation to its float value.
        let v = e4m3_to_f32(0x38); // exp=7, man=0 -> 1.0
        assert!((v - 1.0).abs() < 1e-6, "0x38 should decode to 1.0, got {}", v);
    }

    #[test]
    fn e2m1_grid_is_exact() {
        for (i, &v) in E2M1.iter().enumerate() {
            assert_eq!(e2m1_to_f32(i as u8), v);
            assert!((e2m1_to_f32(f32_to_e2m1(v)) - v).abs() < 1e-6);
            assert!((e2m1_to_f32(f32_to_e2m1(-v)) + v).abs() < 1e-6);
        }
        // Clamps at the top of the grid.
        assert_eq!(e2m1_to_f32(f32_to_e2m1(100.0)), 6.0);
    }

    #[test]
    fn exactly_representable_block_survives_roundtrip() {
        // A block whose values all sit on the E2M1 grid times one scale must round-trip exactly.
        let k = 16;
        let vals: Vec<bf16> = E2M1.iter().chain(E2M1.iter())
            .map(|&v| bf16::from_f32(v * 2.0)).collect();
        assert_eq!(vals.len(), k);
        let q = quantize_nvfp4(&vals, 1, k);
        let d = dequantize_nvfp4(&q);
        for (a, b) in vals.iter().zip(d.iter()) {
            assert!((a.to_f32() - b.to_f32()).abs() < 1e-3, "{} vs {}", a, b);
        }
    }

    /// The tile permutation must be a BIJECTION. If two (r,c) map to one slot, weights are silently
    /// overwritten and the model is quietly wrong — it still loads, still generates, just worse.
    #[test]
    fn mma_tile_slots_are_bijective() {
        let mut seen4 = [false; MMA_TILE_FP4 * 2];   // 128 bytes x 2 nibbles
        let mut seen8 = [false; MMA_TILE_FP8];
        for r in 0..MMA_M {
            for c in 0..MMA_K {
                let (off, hi) = fp4_tile_slot(r, c);
                let idx = off * 2 + hi as usize;
                assert!(!seen4[idx], "fp4 slot collision at ({},{})", r, c);
                seen4[idx] = true;
                let o8 = fp8_tile_slot(r, c);
                assert!(!seen8[o8], "fp8 slot collision at ({},{})", r, c);
                seen8[o8] = true;
            }
        }
        assert!(seen4.iter().all(|&b| b) && seen8.iter().all(|&b| b), "tile not fully covered");
    }

    /// Repack, then walk the inverse map the CUDA `*_tiled` kernels use, and demand the original back.
    #[test]
    fn mma_repack_roundtrips() {
        let (m, k) = (32usize, 64usize);
        let mut s = 99u64;
        let w: Vec<bf16> = (0..m * k).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            bf16::from_f32(((s >> 33) as f32 / 2f32.powi(31) - 0.5) * 0.2)
        }).collect();

        let q4 = quantize_nvfp4(&w, m, k);
        let (wt, st) = repack_nvfp4_mma(&q4.qweight, &q4.scales, m, k);
        let nblk = k / MMA_K;
        for row in 0..m {
            for c in 0..k {
                let (mt, kb) = (row / MMA_M, c / MMA_K);
                let (off, hi) = fp4_tile_slot(row % MMA_M, c % MMA_K);
                let byte = wt[(mt * nblk + kb) * MMA_TILE_FP4 + off];
                let got = if hi { byte >> 4 } else { byte & 0x0F };
                let want = { let b = q4.qweight[row * (k / 2) + c / 2];
                             if c % 2 == 1 { b >> 4 } else { b & 0x0F } };
                assert_eq!(got, want, "fp4 nibble ({},{})", row, c);
                assert_eq!(st[(mt * nblk + kb) * MMA_M + row % MMA_M],
                           q4.scales[row * nblk + kb], "fp4 scale ({},{})", row, c);
            }
        }

        let q8 = quantize_fp8(&w, m, k);
        let wt8 = repack_fp8_mma(&q8.qweight, m, k);
        for row in 0..m {
            for c in 0..k {
                let (mt, kb) = (row / MMA_M, c / MMA_K);
                let off = fp8_tile_slot(row % MMA_M, c % MMA_K);
                assert_eq!(wt8[(mt * nblk + kb) * MMA_TILE_FP8 + off], q8.qweight[row * k + c],
                           "fp8 byte ({},{})", row, c);
            }
        }
    }

    #[test]
    fn roundtrip_error_is_bounded() {
        // Gaussian weights: relative error should land in the few-percent range, not blow up.
        let (m, k) = (8usize, 256usize);
        let mut s = 12345u64;
        let w: Vec<bf16> = (0..m * k).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / (2f32.powi(31));
            bf16::from_f32((u - 0.5) * 0.1)
        }).collect();
        let q = quantize_nvfp4(&w, m, k);
        let d = dequantize_nvfp4(&q);
        let (rel, _) = roundtrip_error(&w, &d);
        assert!(rel < 0.15, "relative error {} too large", rel);
        assert_eq!(q.qweight.len(), m * k / 2);
        assert_eq!(q.scales.len(), m * k / BLOCK);
    }
