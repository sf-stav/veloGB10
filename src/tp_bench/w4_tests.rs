    use super::*;

    #[test]
    fn sweep_payloads_are_whole_rows() {
        let want_rows = [(2u64, 195usize), (8, 781), (21, 2050), (42, 4101)];
        for (mb, rows) in want_rows {
            let n = w4_sweep_floats(mb).unwrap();
            assert_eq!(n, rows * W4_ROW_FLOATS, "{mb} MB");
            assert!(crate::tp_xport::pipe_ok(n));
            assert!(n * 4 <= mb as usize * 1_000_000 && n * 4 + W4_ROW_FLOATS * 4 > mb as usize * 1_000_000, "{mb} MB rounds DOWN by under one row");
        }
        assert!(w4_sweep_floats(0).is_err() && w4_sweep_floats(513).is_err());
        assert_eq!(w4_parse_sizes("2, 8,21,42").unwrap(), vec![2, 8, 21, 42]);
        for bad in ["", " , ", "2,x", "0", "600"] { assert!(w4_parse_sizes(bad).is_err(), "{bad:?}"); }
    }

    #[test]
    fn wire_bytes_count_only_the_first_round_as_f16() {
        let n = 1_000_000usize;
        assert_eq!(w4_wire_bytes(n, 2, false), 8_000_000, "world 4 fp32: two full exchanges");
        assert_eq!(w4_wire_bytes(n, 2, true), 6_000_000, "world 4 f16 first round: 0.5 + 1.0 payloads");
        assert_eq!(w4_wire_bytes(n, 1, false), 4_000_000, "world 2 fp32");
        assert_eq!(w4_wire_bytes(n, 1, true), 2_000_000, "world 2 f16 wire: the TP-F dual-rail f16 payload");
        assert_eq!(w4_wire_bytes(n, 3, true), 10_000_000, "world 8");
    }

    #[test]
    fn combos_refuse_what_the_serial_pair_cannot_carry() {
        for io in ["f32", "f32to16", "f16", "f16w"] {
            assert_eq!(w4_check_combo(1, io).unwrap(), 1, "{io}");
            assert_eq!(w4_check_combo(2, io).unwrap(), 2, "{io}");
        }
        assert_eq!(w4_check_combo(0, "f32").unwrap(), 1);
        for io in ["f32to16", "f16", "f16w"] {
            let e = w4_check_combo(0, io).unwrap_err().to_string();
            assert!(e.contains("--xport 0") && e.contains("pipelined"), "{io}: {e}");
        }
        assert!(w4_check_combo(3, "f32").is_err());
        assert!(w4_check_combo(1, "bf16").is_err());
    }

    /// The reference is not vacuous: it is rank-identical, equals the closed forms at both phases, the data
    /// tells the two world-4 associations apart, and the f16 IOs differ from each other on real data.
    #[test]
    fn reference_tree_is_rank_identical_and_phase_sensitive() {
        let n = 4000;
        let mut differ = 0;
        for i in 0..n {
            let mut v = [0f32; 8];
            for r in 0..4 { v[r] = w4_fill(r as i32, i, 3); }
            let all = w4_tree_all(&v, 4, 1);
            assert!((1..4).all(|r| all[r].to_bits() == all[0].to_bits()), "elem {i}: ranks disagree");
            let a02 = (v[0] + v[2]) + (v[1] + v[3]);
            let a01 = (v[0] + v[1]) + (v[2] + v[3]);
            assert_eq!(all[0].to_bits(), a02.to_bits(), "shift 1 is (p0+p2)+(p1+p3)");
            assert_eq!(w4_tree_all(&v, 4, 0)[0].to_bits(), a01.to_bits(), "shift 0 is (p0+p1)+(p2+p3)");
            if a02.to_bits() != a01.to_bits() { differ += 1; }
            // world 2 has one round: the phase shift cannot matter
            assert_eq!(w4_tree_all(&v, 2, 0)[0].to_bits(), w4_tree_all(&v, 2, 1)[0].to_bits());
            assert_eq!(w4_tree_all(&v, 2, 0)[0].to_bits(), (v[0] + v[1]).to_bits());
        }
        assert!(differ > 50, "the fill data must distinguish the two associations (got {differ})");
        // world 8: three rounds, every rank agrees
        let v = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let all = w4_tree_all(&v, 8, 1);
        assert!((1..8).all(|r| all[r].to_bits() == all[0].to_bits()));
    }

    #[test]
    fn expected_bits_follow_the_io_rounding_points() {
        let (mut f32to16_vs_f16, mut f32_vs_f16) = (0, 0);
        for i in 0..4000 {
            let mut f = [0f32; 8];
            let mut h = [0f32; 8];
            for r in 0..4 { f[r] = w4_fill(r as i32, i, 1); h[r] = half::f16::from_f32(f[r]).to_f32(); }
            let s32 = w4_tree_all(&f, 4, 1)[0];
            let s16 = w4_tree_all(&h, 4, 1)[0];
            assert_eq!(w4_expected_bits("f32", 4, i, 1, 1), s32.to_bits());
            assert_eq!(w4_expected_bits("f32to16", 4, i, 1, 1), half::f16::from_f32(s32).to_bits() as u32);
            assert_eq!(w4_expected_bits("f16", 4, i, 1, 1), half::f16::from_f32(s16).to_bits() as u32);
            assert_eq!(w4_expected_bits("f16w", 4, i, 1, 1), w4_expected_bits("f16", 4, i, 1, 1), "the f16 wire is bitwise the same sum");
            if w4_expected_bits("f32to16", 4, i, 1, 1) != w4_expected_bits("f16", 4, i, 1, 1) { f32to16_vs_f16 += 1; }
            if half::f16::from_f32(s32).to_f32().to_bits() != s32.to_bits() { f32_vs_f16 += 1; }
        }
        assert!(f32to16_vs_f16 > 0, "f16-rounded inputs and fp32 inputs must be distinguishable by the reference");
        assert!(f32_vs_f16 > 2000, "the f16 output rounding must be visible in the reference");
    }

    /// The checker detects one flipped bit, reports it, and names the other association exactly.
    #[test]
    fn checker_detects_a_flipped_bit_and_a_phase_shift() {
        let (world, it, n) = (4usize, 2usize, 3000usize);
        for io in ["f32", "f16w"] {
            let exact: Vec<u32> = (0..n).map(|i| w4_expected_bits(io, world, i, it, 1)).collect();
            let (bad, first) = w4_count_bad(io, world, it, 1, n, &|i| exact[i]);
            assert_eq!((bad, first), (0, None), "{io}: exact input must pass");
            let mut flipped = exact.clone();
            flipped[1234] ^= 1;
            let (bad, first) = w4_count_bad(io, world, it, 1, n, &|i| flipped[i]);
            assert_eq!(bad, 1, "{io}");
            assert_eq!(first, Some((1234, flipped[1234], exact[1234])));
            let other: Vec<u32> = (0..n).map(|i| w4_expected_bits(io, world, i, it, 0)).collect();
            let (bad_vs_mine, _) = w4_count_bad(io, world, it, 1, n, &|i| other[i]);
            let (bad_vs_other, _) = w4_count_bad(io, world, it, 0, n, &|i| other[i]);
            assert!(bad_vs_mine >= n / 16, "{io}: the shifted association must fail the aligned reference on every cancellation-quad element (got {bad_vs_mine}; else the check is vacuous)");
            assert_eq!(bad_vs_other, 0, "{io}: and pass its own");
        }
    }
