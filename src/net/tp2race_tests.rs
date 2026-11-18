    use super::probe_gen_delta;

    /// The alignment invariant: after the bump, the NEXT exchange lands on the wanted slot.
    #[test]
    fn gen_delta_lands_next_gen_on_wanted_slot() {
        for gen in [0u64, 1, 7, 8, 9, 4095, 3052830, u64::MAX - 16] {
            for want in 0u64..8 {
                let d = probe_gen_delta(gen, want);
                assert!(d < 8, "delta must stay in 0..=7");
                assert_eq!((gen.wrapping_add(d).wrapping_add(1)) & 7, want,
                           "gen {gen} want {want} delta {d}");
            }
        }
    }

    /// Zero delta exactly when the next generation is already on the wanted slot.
    #[test]
    fn gen_delta_is_minimal() {
        assert_eq!(probe_gen_delta(7, 0), 0);   // next gen 8 -> slot 0 already
        assert_eq!(probe_gen_delta(6, 0), 1);   // next gen 7 -> slot 7; want 0 -> +1
        assert_eq!(probe_gen_delta(0, 0), 7);   // next gen 1 -> slot 1; want 0 -> +7
        assert_eq!(probe_gen_delta(0, 1), 0);   // next gen 1 -> slot 1
        assert_eq!(probe_gen_delta(0, 2), 1);
    }
