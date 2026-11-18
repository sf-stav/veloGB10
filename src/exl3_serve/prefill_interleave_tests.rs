//! Unit tests (module `prefill_interleave_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::*;

    #[test]
    fn the_flag_parses_to_zero_when_absent_and_refuses_garbage() {
        assert_eq!(parse_prefill_interleave(None).unwrap(), 0);
        assert_eq!(parse_prefill_interleave(Some("")).unwrap(), 0);
        assert_eq!(parse_prefill_interleave(Some(" 2 ")).unwrap(), 2);
        assert_eq!(parse_prefill_interleave(Some("0")).unwrap(), 0);
        for bad in ["on", "-1", "1.5"] {
            let e = parse_prefill_interleave(Some(bad)).unwrap_err().to_string();
            assert!(e.starts_with("--prefill-interleave:"), "{e}");
        }
        let a: Vec<String> = ["--prefill-interleave=3", "--port", "9000"].iter().map(|s| s.to_string()).collect();
        assert_eq!(parse_prefill_interleave(arg(&a, "--prefill-interleave")).unwrap(), 3);
    }

    /// Off is silent (default boot logs unchanged); on says so, at TP=1 and (PERFFIX1) under TP, where every
    /// rank runs the same steps at the same chunk boundaries.
    #[test]
    fn the_resolved_value_is_the_flag_at_every_topology_and_the_boot_line_says_so() {
        assert_eq!(resolve_prefill_interleave(0, false, 4095), (0, None));
        assert_eq!(resolve_prefill_interleave(0, true, 4095), (0, None));
        let (n, line) = resolve_prefill_interleave(2, false, 1024);
        assert_eq!(n, 2);
        let line = line.unwrap();
        assert!(line.starts_with("[exl3-serve] prefill interleave ON (--prefill-interleave 2)") && line.contains("chunk 1024 tokens"), "{line}");
        let (n, line) = resolve_prefill_interleave(2, true, 1024);
        assert_eq!(n, 2, "PERFFIX1: every rank runs the flag's value under TP");
        let line = line.unwrap();
        assert!(line.contains("prefill interleave ON under TP (--prefill-interleave 2)") && line.contains("in lockstep"), "{line}");
    }

    /// The hook never runs after the final chunk (the seam step follows it), for a non-cancellable
    /// internal prefill, or with the flag off (at every topology since PERFFIX1).
    #[test]
    fn the_hook_runs_only_between_chunks_of_a_cancellable_prefill() {
        assert!(interleave_point(1, true, 1024, 20_000));
        assert!(interleave_point(4, true, 19_999, 20_000));
        assert!(!interleave_point(1, true, 20_000, 20_000), "final chunk");
        assert!(!interleave_point(0, true, 1024, 20_000), "flag off");
        assert!(!interleave_point(1, false, 1024, 20_000), "WP16 XCHECK reference prefill");
    }

    /// Every served grid ends exactly at `n`, whatever its options (realign, tail split, absorb-tail),
    /// so no hook follows the last chunk and every earlier boundary is a hook point. (A property of
    /// `plan_grid` that the hook predicate relies on; the hook itself is covered by the GPU gate.)
    #[test]
    fn interleave_points_are_the_grid_minus_its_last_chunk() {
        use crate::exl3_forward::plan_grid;
        for (from, n, c, realign, tail, absorb) in [
            (0usize, 20_000usize, 1024usize, false, None, None),
            (0, 20_000, 4095, true, Some(19_900), Some(16)),
            (3_000, 20_000, 1024, true, Some(19_000), Some(16)),
            (0, 900, 1024, false, None, None),
            (0, 4095, 4095, true, None, Some(16)),
        ] {
            let grid = plan_grid(from, n, c, realign, tail, psc_rows(c, false), absorb, true);
            assert_eq!(grid.last().copied(), Some(n), "the grid ends at n");
            let points = grid.iter().filter(|&&e| interleave_point(1, true, e, n)).count();
            assert_eq!(points, grid.len() - 1, "grid {grid:?}");
        }
    }

    /// The registry row: a plain integer flag, no environment variable, default off.
    #[test]
    fn the_flag_is_registered_without_an_environment_variable() {
        let o = crate::opt!("prefill-interleave").def();
        assert_eq!((o.flag, o.env, o.def), ("prefill-interleave", "", "1"), "owner 2026-10-24: default 1");
        // the server's unset fallback is the registry default, not a literal
        let serve: String = src_text!("exl3_serve").lines().filter(|l| !l.trim_start().starts_with("//")).collect::<Vec<_>>().join("\n");
        assert!(serve.contains(".unwrap_or_else(|| crate::opt!(\"prefill-interleave\").def().def.to_string())"), "unset -> registry default");
        assert!(matches!(o.ty, crate::opts::Ty::Int));
        assert_eq!(o.scope, crate::opts::Scope::Spmd);
    }
