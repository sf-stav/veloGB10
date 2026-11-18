//! Unit tests (module `wp24_key_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    use super::*;

    /// Default = today's keys (the boot precapture's); the WP24 bits never collide with the key
    /// fields below them (class << 30, qsa bucket << 21, qsa bit 20, slot << 8, k/i/m) or with the
    /// WP15 / DHEADP bits 40 / 41, and the modes are pairwise distinct.
    #[test]
    fn rq_key_bits_are_inert_by_default_and_disjoint() {
        assert_eq!(rq_draft_bits(0), 0);
        assert_eq!(rq_verify_bits(false), 0);
        let low = (3usize << 30) | (1 << 20) | (((1usize << 9) - 1) << 21) | (255 << 8) | 255;
        let pen = (1usize << 40) | (1 << 41);
        // TP-C: the scratch-id field (bits 48..55) sits above every other key field
        assert_eq!((0x7Fusize << 48) & (low | pen | rq_draft_bits(3) | rq_verify_bits(true) | (6usize << 30)), 0);
        let bits = [rq_draft_bits(1), rq_draft_bits(2), rq_verify_bits(true)];
        for (i, &b) in bits.iter().enumerate() {
            assert!(b != 0 && b & (low | pen) == 0, "WP24 bit {b:#x} collides");
            for &c in &bits[i + 1..] { assert_ne!(b, c); }
        }
        let (d1, d2) = (dkey_of(1, 7, true, 3) | rq_draft_bits(1), dkey_of(1, 7, true, 3) | rq_draft_bits(2));
        assert_ne!(d1, d2);
        assert_ne!(d1, dkey_of(1, 7, true, 3));
        assert_eq!(vkey_of(2, 8, false, 0) | rq_verify_bits(false), vkey_of(2, 8, false, 0));
    }

    /// PACK4S: the packed family key gains the rq bit and nothing else; with the bit clear it is
    /// PACK4's key exactly, and the bit touches no field of the packed key or the scratch id.
    #[test]
    fn packed_key_rq_family_bit() {
        for &(r, qsa, qb) in &[(4usize, false, 0usize), (16, false, 0), (10, true, 0), (16, true, 5), (15, true, 511)] {
            let (g, q) = (pack_key(r, qsa, qb, false), pack_key(r, qsa, qb, true));
            assert_eq!(g, pkeyv_of(r, qsa, qb), "greedy and match-mode pairs keep the PACK4 family");
            assert_eq!(q, g | (1usize << 44));
            assert_ne!(q, g);
            assert_eq!(g & (1usize << 44), 0);
            assert_eq!((q | g) & (0x7Fusize << 48), 0, "the scratch id field stays free");
            assert_eq!(q >> 30 & 7, 4, "still the packed class");
        }
        // 13 widths x 2 families are 26 distinct keys per regime
        let keys: std::collections::BTreeSet<usize> = crate::lane_pack::PACK_WIDTHS
            .flat_map(|r| [pack_key(r, false, 0, false), pack_key(r, false, 0, true)]).collect();
        assert_eq!(keys.len(), 26);
    }

    /// SampParams' packing is unchanged by the flag parameter (flag 1 = today's rows).
    #[test]
    fn samp_pack_flag_one_is_the_old_row() {
        let p = SampParams::sampled(1.0, 0.95, 20, 0.0, 0x1234_5678_9ABC_DEF0, 7).unwrap();
        assert_eq!(p.pack(), p.pack_flag(1));
        assert_eq!(p.pack()[0], 1);
        let q = p.pack_flag(2);
        assert_eq!(q[0], 2);
        assert_eq!(q[1..], p.pack()[1..]);
    }

    /// The lane rule: greedy lanes never stage real-q; ratio stages mode 2 with the request's
    /// truncation and (default) temperature; match with no dump stages nothing (today).
    #[test]
    fn rq_stage_lane_rule() {
        assert_eq!(RqStage::for_lane(true, None, 0.0, 0.95, 20, 0.0, 9, 0), None);
        assert_eq!(RqStage::for_lane(true, None, 1.0, 0.95, 1, 0.0, 9, 0), None);
        if !crate::wp24::dump_on() {
            assert_eq!(RqStage::for_lane(false, None, 1.0, 0.95, 20, 0.0, 9, 0), None);
        }
        let r = RqStage::for_lane(true, None, 0.7, 0.8, 20, 0.05, 9, 42).unwrap();
        assert_eq!((r.mode, r.tau, r.top_p, r.top_k, r.seed, r.dctr), (2, 0.7, 0.8, 20, 9, 42));
        let r = RqStage::for_lane(true, Some(1.2), 0.7, 1.5, 0, 0.0, 9, 0).unwrap();
        assert_eq!((r.tau, r.top_p, r.top_k), (1.2, 1.0, 0)); // top_p > 1 -> 1 (SampParams rule)
    }
