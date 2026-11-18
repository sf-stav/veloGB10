//! Unit tests (module `spill_region_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    use super::*;

    #[test]
    fn slot_regions_are_byte_strided_for_every_element_size() {
        for (size, name) in [(1usize, "u8 KV"), (2, "u16 QSA / PLE"), (4, "f32 GDN state / carry")] {
            for slot in 0..4 {
                let (at, len) = spill_slot_region(0x1000, 400, size, 4, slot);
                assert_eq!((at, len), (0x1000 + (slot * 100 * size) as u64, 100 * size), "{name} slot {slot}");
            }
            let (last, len) = spill_slot_region(0x1000, 400, size, 4, 3);
            assert_eq!(last + len as u64, 0x1000 + (400 * size) as u64, "{name}: the four slots tile the plane exactly");
        }
    }

    /// ISSUES 123: a snapshot of `tokens` positions carries rows through the end of the block holding position `tokens`.
    #[test]
    fn spill_rows_cover_the_resume_block_and_stop_at_max_pos() {
        assert_eq!(spill_rows(4327, 4, 262_144), 4328); // resume at 4327: its block is rows 4324..4328
        assert_eq!(spill_rows(4328, 4, 262_144), 4332);
        assert_eq!(spill_rows(1, 4, 262_144), 4);
        assert_eq!(spill_rows(100, 1, 262_144), 101); // no indexer: rows 0..=tokens
        assert_eq!(spill_rows(262_143, 4, 262_144), 262_144);
    }

    /// ISSUES 123: one region per segment, each starting at the segment base of THIS slot and `rows` rows long.
    #[test]
    fn spill_row_regions_take_the_slot_prefix_of_every_segment() {
        // [width 4][2 segments][rows_cap 10][row 3 B] = 240 B; slot 2, 7 rows
        let r = spill_row_regions(0x100, 240, 4, 2, 10, 3, 7).unwrap();
        assert_eq!(r, vec![(0x100 + 120, 21), (0x100 + 150, 21)]);
        assert!(spill_row_regions(0x100, 241, 4, 2, 10, 3, 7).is_err(), "a plane of another shape is refused");
        assert!(spill_row_regions(0x100, 240, 4, 4, 10, 3, 7).is_err(), "slot out of range");
        assert!(spill_row_regions(0x100, 240, 4, 2, 10, 3, 11).is_err(), "more rows than the plane holds");
        // the whole-lane case reproduces the version-2 region (one segment, every row)
        assert_eq!(spill_row_regions(0x0, 400, 4, 1, 100, 1, 100).unwrap(), vec![(100, 100)]);
    }

    /// Issue sf-stav/veloGB10#41 (reported by @AbsolomBane): the spill / prefix-pool copy of a pooled QSA key plane must
    /// address every lane where the kernels do (`slot * nblk_stride * hd` bf16, nblk_stride = floor(max_pos / ratio):
    /// `qsa_pool_range` / `qsa_pool_update_b`). The plane used to be allocated with ceil(max_pos / ratio) blocks per
    /// lane, so the copy (which derives the lane stride from the plane size) stepped lanes 1 block further than the
    /// kernels whenever max_pos % ratio != 0 — every --max-seq-len that is a multiple of 4 (max_pos = window + 1).
    #[test]
    fn pooled_plane_copy_addresses_every_lane_where_the_kernels_do() {
        let (ratio, hd, width) = (4usize, 128usize, 4usize);
        for max_pos in [262_145usize, 262_144, 524_289, 1_048_575, 1_048_572, 4_097, 4] {
            let nblk = qsa_nblk_stride(max_pos, ratio);
            let elems = qsa_pool_plane_elems(width, max_pos, ratio, hd);
            assert_eq!(elems, width * (max_pos / ratio) * hd, "max_pos {max_pos}: [width][floor(max_pos / ratio)][hd]");
            let blocks = spill_rows(max_pos / 2, ratio, max_pos).div_ceil(ratio).min(nblk);
            for slot in 0..width {
                let r = spill_row_regions(0, elems * 2, width, slot, nblk, hd * 2, blocks).unwrap();
                let kernel_lane_base = (slot * nblk * hd * 2) as u64; // qsa_pool_range: pool_p + slot * nblk_stride * hdx * 2
                assert_eq!(r, vec![(kernel_lane_base, blocks * hd * 2)], "max_pos {max_pos} slot {slot}");
                // the copied blocks stay inside the lane the kernels give this slot
                assert!(blocks <= nblk && r[0].0 + r[0].1 as u64 <= ((slot + 1) * nblk * hd * 2) as u64);
            }
        }
        // the issue's worked example: max_pos 1048575, ratio 4, hd 128 -> lane 1 starts at 67,108,608 bytes (not 67,108,864)
        let elems = qsa_pool_plane_elems(4, 1_048_575, 4, 128);
        assert_eq!(spill_row_regions(0, elems * 2, 4, 1, qsa_nblk_stride(1_048_575, 4), 256, 1).unwrap()[0].0, 67_108_608);
        assert_eq!(spill_row_regions(0, elems * 2, 4, 2, qsa_nblk_stride(1_048_575, 4), 256, 1).unwrap()[0].0, 134_217_216);
    }
