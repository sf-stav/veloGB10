//! Unit tests for the geometry-word packers (former inline test module `geom_pack_tests` of `gpu.rs`; file map in `mod.rs`).

    //! The unpack expressions below are the kernels' own (gpu_batch.cu: gqa_attn_splitk for nh_packed
    //! and bs_packed, rmsnorm_rope_kvwrite_b for stride_nkv, qsa_score_b for the geometry word).
    use super::*;

    fn unpack_nh(w: i32) -> (i32, i32, i32) { (w >> 20, (w >> 10) & 0x3FF, w & 0x3FF) }
    fn unpack_bs(w: u64) -> (u64, u64, u64, u64) { (w & 0x7FFFF, (w >> 19) & 0x3F, (w >> 25) & 0x3F, (w >> 31) & 0x7FFFF) }
    fn unpack_stride_nkv(w: u32) -> (u32, u32) { (w & 0x7FFFF, w >> 19) }
    fn unpack_geom(w: u64) -> (u64, u64, u64) { (w & 0xFFFFF, (w >> 20) & 0xFFFFF, (w >> 40) & 0xFFFFF) }

    /// Every field survives the round trip at its widest value, and at the shipped shapes.
    #[test]
    fn fields_round_trip_at_their_widest_values() {
        assert_eq!(unpack_nh(pack_nh(2047, 1023, 1023)), (2047, 1023, 1023));
        assert_eq!(unpack_nh(pack_nh(32, 256, 4)), (32, 256, 4));
        assert_eq!(unpack_bs(pack_bs(524_287, 63, 63, 524_287)), (524_287, 63, 63, 524_287));
        assert_eq!(unpack_bs(pack_bs(262_144, 32, 8, 10_240)), (262_144, 32, 8, 10_240));
        assert_eq!(unpack_stride_nkv(pack_stride_nkv(8191, 524_287)), (524_287, 8191));
        assert_eq!(unpack_stride_nkv(pack_stride_nkv(4, 262_144)), (262_144, 4));
        assert_eq!(unpack_geom(pack_qsa_geom(1_048_575, 1_048_575, 1_048_575)), (1_048_575, 1_048_575, 1_048_575));
        assert_eq!(unpack_geom(pack_qsa_geom(262_145, 640, 65_536)), (262_145, 640, 65_536));
    }

    /// What the old release build did with a window one past the field: the unchecked expression puts
    /// the stride's top bit into ns_grid and hands the kernel stride 0 (every slot aliases slot 0).
    #[test]
    fn the_unchecked_expression_spills_into_the_neighbour() {
        let (stride, ns_grid, batch, q_pitch) = (1usize << 19, 2usize, 1usize, 0usize);
        let old = ((q_pitch as u64) << 31) | ((batch as u64) << 25) | ((ns_grid as u64) << 19) | (stride as u64);
        assert_eq!(unpack_bs(old), (0, 3, 1, 0), "stride 2^19 reads back as stride 0 with ns_grid 2 -> 3");
        let old_geom = ((1u64 << 20)) | (640u64 << 20) | (262_144u64 << 40);
        assert_eq!(unpack_geom(old_geom), (0, 641, 262_144), "max_pos 2^20 reads back as 0 with qk_dim 640 -> 641");
    }

    #[test]
    #[should_panic(expected = "nh_packed field overflow")]
    fn nh_packed_refuses_an_oversized_head_dim() { pack_nh(32, 1024, 4); }

    #[test]
    #[should_panic(expected = "bs_packed field overflow")]
    fn bs_packed_refuses_an_oversized_stride() { pack_bs(1 << 19, 1, 1, 1); }

    #[test]
    #[should_panic(expected = "bs_packed field overflow")]
    fn bs_packed_refuses_an_oversized_batch() { pack_bs(4096, 1, 64, 1); }

    #[test]
    #[should_panic(expected = "stride_nkv field overflow")]
    fn stride_nkv_refuses_an_oversized_window() { pack_stride_nkv(4, 1 << 19); }

    /// `max_pos` is this word's first field: its 20 bits are what caps the EXL3 window
    /// (`exl3_forward::MAX_POS_PACKED`).
    #[test]
    #[should_panic(expected = "qsa geom field overflow")]
    fn qsa_geom_refuses_an_oversized_window() { pack_qsa_geom(crate::exl3_forward::MAX_POS_PACKED, 640, 1); }

    #[test]
    #[should_panic(expected = "qsa geom field overflow")]
    fn qsa_geom_refuses_an_oversized_block_count() { pack_qsa_geom(4096, 640, 1 << 20); }
