//! Unit tests (module `df2_carry`) moved out of the old inline block; text unchanged.

    use super::df2_carry_ok;

    const ON: bool = true;
    const DF2: bool = true;
    const SLOT_A: usize = 0;
    const SLOT_B: usize = 1;
    /// The real guard (frontier term active). `len_only` is the diagnostics instrument.
    const FULL: bool = false;

    /// The natural multi-turn flow: turn 1 ran to frontier N = 2240; turn 2's prompt is that
    /// whole sequence plus the client's next message (plen = 2440).
    #[test]
    fn multi_turn_carries() {
        assert!(df2_carry_ok(ON, DF2, 2048, 2440, SLOT_A, Some(SLOT_A), 2240, FULL));
    }

    /// Turn 2's prompt is SHORTER than turn 1's frontier: the rows the draft needs are exactly
    /// the ones the generation has already overwritten. Must refuse. THIS is the case the
    /// frontier term exists for -- and the case `len_only` wrongly allows.
    #[test]
    fn frontier_past_prompt_refused() {
        assert!(!df2_carry_ok(ON, DF2, 2048, 2200, SLOT_A, Some(SLOT_A), 2240, FULL));
    }

    /// Negative-control instrument check (SPEC §7 gate 7): with the frontier term dropped the
    /// SAME state is accepted, which is what lets the gate force a known-bad carry.
    #[test]
    fn len_only_instrument_admits_the_bad_carry() {
        assert!(df2_carry_ok(ON, DF2, 2048, 2200, SLOT_A, Some(SLOT_A), 2240, true));
    }

    /// §5 hazard: slot A once ran the drafter at 3000, then slot B clobbered the ring. The
    /// per-slot length for A still says 3000, so a LENGTH-ONLY check (DSpark's shape) would carry
    /// here and feed the draft slot B's rows. The identity guard refuses.
    #[test]
    fn stale_ring_from_another_slot_refused() {
        assert!(!df2_carry_ok(ON, DF2, 2048, 2600, SLOT_A, Some(SLOT_B), 3000, FULL));
    }

    /// Cold start: `reuse == 0` never carries (the lane primes from 0).
    #[test]
    fn cold_never_carries() {
        assert!(!df2_carry_ok(ON, DF2, 0, 2600, SLOT_A, Some(SLOT_A), 3000, FULL));
    }

    /// Unproven ring identity (reset / failed prime) refuses even when the length looks right.
    #[test]
    fn unproven_identity_refused() {
        assert!(!df2_carry_ok(ON, DF2, 2048, 2600, SLOT_A, None, 3000, FULL));
    }

    /// The ring must have covered the reused prefix.
    #[test]
    fn reuse_past_frontier_refused() {
        assert!(!df2_carry_ok(ON, DF2, 2500, 2600, SLOT_A, Some(SLOT_A), 2400, FULL));
    }

    /// Flag OFF is today's behaviour exactly: never carry, whatever the state says.
    #[test]
    fn flag_off_never_carries() {
        assert!(!df2_carry_ok(false, DF2, 2048, 2440, SLOT_A, Some(SLOT_A), 2240, FULL));
    }

    /// Non-DFlash2 source never carries.
    #[test]
    fn non_df2_source_never_carries() {
        assert!(!df2_carry_ok(ON, false, 2048, 2440, SLOT_A, Some(SLOT_A), 2240, FULL));
    }
