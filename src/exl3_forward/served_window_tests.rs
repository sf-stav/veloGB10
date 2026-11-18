//! Unit tests (module `served_window_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    use super::*;

    /// Highest `scores[]` index the scorers write for `m` rows when the last row sits at `pos`:
    /// `scores[(r0 + r) * nblk_stride + j]` for `j < (pos + 1) / ratio` (gpu_batch.cu).
    fn last_score_index(m: usize, pos: usize, max_pos: usize, ratio: usize) -> Option<usize> {
        let nb = (pos + 1) / ratio;
        (nb > 0).then(|| (m - 1) * qsa_nblk_stride(max_pos, ratio) + nb - 1)
    }

    /// The scratch holds everything the kernels touch, for every served window: the scorer's last
    /// write of the last row at the last position, and the top-k kernels' whole-quad read of that
    /// row's last blocks (`xq_tk3_ld4`: blocks 4q..4q+3 whenever 4q < nb) -- also when the pitch is
    /// not a multiple of 4, which the old fixed 65536-block rows used to hide.
    #[test]
    fn the_score_plane_holds_every_write_and_every_quad_read() {
        for max_pos in [2usize, 4097, 32_769, 65_537, 100_011, 262_145, 262_148, 300_001, 524_289, MAX_POS_PACKED - 1] {
            for ratio in [1usize, 4, 64] {
                for w in [1usize, 2, 8, 9, 13, 16] {
                    let len = qsa_scores_len(true, ratio, w, max_pos);
                    let pitch = (max_pos / ratio).max(1);
                    for m in 1..=w {
                        let Some(top) = last_score_index(m, max_pos - 1, max_pos, ratio) else { continue };
                        assert!(top < len, "write: max_pos {max_pos} ratio {ratio} w {w} m {m}: index {top} >= {len}");
                        // the quad holding the row's last block, read whole
                        let nb = max_pos / ratio;
                        let quad_end = (m - 1) * pitch + 4 * ((nb - 1) / 4) + 3;
                        assert!(quad_end < len, "quad read: max_pos {max_pos} ratio {ratio} w {w} m {m}: index {quad_end} >= {len}");
                    }
                }
            }
        }
        // without the pad the last row's quad read leaves the plane whenever the pitch is not a multiple of 4
        let (max_pos, ratio, w) = (100_011usize, 4usize, 9usize);
        let pitch = max_pos / ratio;
        assert_eq!(pitch % 4, 2);
        assert!((w - 1) * pitch + 4 * ((pitch - 1) / 4) + 3 >= w * pitch, "the unpadded plane is 2 f32 short here");
        assert_eq!(qsa_scores_len(false, 4, 8, 262_145), 0, "no indexer: no score plane");
    }

    /// The bug public PR #21 reported, on the shipped shape (Flash-Next: ratio 4, verify width 8): the
    /// old size came from the trained window capped at 262144. It agrees with the scorer up to
    /// max_pos 262147 (floor division) and is overrun by every wider served window.
    #[test]
    fn the_trained_window_sizing_was_overrun_by_a_wider_served_window() {
        let (ratio, w) = (4usize, 8usize);
        let old_len = w * (262_144usize.clamp(1, 262_144) / ratio).max(1);
        for max_pos in [4097usize, 32_769, 262_145, 262_147] {
            assert!(last_score_index(w, max_pos - 1, max_pos, ratio).unwrap() < old_len, "max_pos {max_pos} fitted");
            assert!(w * qsa_nblk_stride(max_pos, ratio) <= old_len);
        }
        // one block past: the last verify row overruns near the end of the window
        assert!(last_score_index(w, 262_147, 262_148, ratio).unwrap() >= old_len);
        // --max-seq-len 524288: row 4's very first block is already past the end, at any position
        assert_eq!(last_score_index(5, 3, 524_289, ratio).unwrap(), old_len);
        // the new size holds both
        assert!(last_score_index(w, 262_147, 262_148, ratio).unwrap() < qsa_scores_len(true, ratio, w, 262_148));
        assert!(last_score_index(w, 524_288, 524_289, ratio).unwrap() < qsa_scores_len(true, ratio, w, 524_289));
    }

    /// The loader refuses a window the 20-bit packings cannot carry, and names the flag to lower.
    #[test]
    fn the_loader_guard_stops_at_the_packing_width() {
        assert!(check_max_pos_packable(1).is_ok());
        assert!(check_max_pos_packable(262_145).is_ok());
        assert!(check_max_pos_packable(MAX_POS_PACKED - 1).is_ok());
        for bad in [MAX_POS_PACKED, MAX_POS_PACKED + 1, usize::MAX] {
            let e = check_max_pos_packable(bad).unwrap_err().to_string();
            assert!(e.contains("2^20") && e.contains("--max-seq-len"), "{e}");
        }
        // the guard and the geometry word agree on the last window that fits
        let top = MAX_POS_PACKED - 1;
        assert_eq!(crate::gpu::pack_qsa_geom(top, 640, qsa_nblk_stride(top, 4)) & 0xFFFFF, top as u64);
    }

    /// Wiring pin (the unit suite cannot build a `Scratch`): the score plane is allocated from the
    /// served-window argument, the decode scorer pitches by the same function and checks the plane,
    /// and the loader runs the guard. The served path itself is covered by the GPU gate.
    #[test]
    fn the_scratch_is_sized_from_the_served_window() {
        let src = src_text!("exl3_forward");
        let code: String = src.lines().filter(|l| !l.trim_start().starts_with("//")).collect::<Vec<_>>().join("\n");
        let once = |needle: &str| assert_eq!(code.matches(needle).count(), 1, "expected exactly once: {needle}");
        // (needles are split so this test's own text never matches)
        let once = |a: &str, b: &str| once(&[a, b].concat());
        once("qsa_scores: dev.alloc_zeros::<f32>(", "qsa_scores_len(cfg.has_indexer(), cfg.indexer_compress_ratio, w, max_pos))?,");
        once("pub fn new(dev: &Arc<CudaDevice>, cfg: &crate::qwen::Config, ", "w: usize, rdim: usize, max_pos: usize) -> Result<Self> {");
        once("anyhow::ensure!(sc.qsa_scores.len() ", ">= m * nblk_stride,");
        once("check_max_pos_packable(", "max_pos)?;");
        let trained = ["max_position_embeddings", ".clamp(1, 262144)"].concat();
        assert!(!code.contains(&trained), "no buffer may be sized from the trained window");
    }
