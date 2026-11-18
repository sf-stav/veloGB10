//! Unit tests (module `ple_headroom_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    //! PR #17 (opt-in port): the `--ple-ram auto` headroom arithmetic.
    //! `ple_auto_headroom_gib(per_lane, width)` = 16 GiB base + per_lane per ADDITIONAL lane.
    //! Default per-lane 0 must reproduce the pre-port fixed 16 GiB exactly (same decision).
    use super::ple_auto_headroom_gib;

    const GIB: u64 = 1 << 30;

    #[test]
    fn default_zero_is_the_old_fixed_16gib_exactly() {
        for width in [1usize, 2, 4, 10, 32] {
            assert_eq!(ple_auto_headroom_gib(0.0, width), 16 * GIB, "width {width}");
        }
    }

    #[test]
    fn two_gib_per_lane_matches_the_pr_arithmetic() {
        // the PR's measured case: --max-batch 10 -> 16 + 2*(10-1) = 34 GiB
        assert_eq!(ple_auto_headroom_gib(2.0, 10), 34 * GIB);
        assert_eq!(ple_auto_headroom_gib(2.0, 4), 16 * GIB + 6 * GIB);
    }

    #[test]
    fn single_lane_never_gains_headroom() {
        assert_eq!(ple_auto_headroom_gib(8.0, 1), 16 * GIB);
        assert_eq!(ple_auto_headroom_gib(0.0, 1), 16 * GIB);
    }

    #[test]
    fn inf_per_lane_with_width_one_is_still_16gib_not_nan_zeroed() {
        // inf * 0 lanes = NaN -> `as u64` = 0: the guard must keep the 16 GiB base.
        assert_eq!(ple_auto_headroom_gib(f64::INFINITY, 1), 16 * GIB);
        assert_eq!(ple_auto_headroom_gib(f64::NAN, 1), 16 * GIB);
    }

    #[test]
    fn saturates_instead_of_overflowing() {
        assert_eq!(ple_auto_headroom_gib(f64::INFINITY, 64), u64::MAX);
        assert_eq!(ple_auto_headroom_gib(-5.0, 8), 16 * GIB, "negative per-lane clamps to 0");
    }
