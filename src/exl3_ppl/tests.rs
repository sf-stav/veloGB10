//! Unit tests of `exl3_ppl` (the former inline `mod tests`; same module name). See `mod.rs` for the code under test.

    use super::*;

    fn h(vals: &[f32]) -> Vec<u16> { vals.iter().map(|&x| f16::from_f32(x).to_bits()).collect() }

    #[test]
    fn nll_of_a_hand_computed_row() {
        // logits [1, 2, 0, -1], target 1 -> ln(e + e^2 + 1 + e^-1) - 2 = 0.4401896985611953
        let lut = f16_lut();
        let (nll, rs) = nll_row(&h(&[1.0, 2.0, 0.0, -1.0]), 1, &lut);
        assert!((nll - 0.4401896985611953).abs() < 1e-9, "{nll}");
        assert_eq!(rs.argmax, 1);
        assert!(!rs.nonfinite);
        // uniform row, any target -> ln 4
        let (nll, _) = nll_row(&h(&[0.0; 4]), 3, &lut);
        assert!((nll - 1.3862943611198906).abs() < 1e-12, "{nll}");
        // large logits: the max-shift must keep it finite and exact
        let (nll, _) = nll_row(&h(&[100.0, 99.0, -100.0, 100.0]), 1, &lut);
        assert!((nll - 1.8619948040582557).abs() < 1e-9, "{nll}");
    }

    #[test]
    fn accumulation_matches_the_hand_computed_perplexity() {
        let lut = f16_lut();
        let rows = [(h(&[1.0, 2.0, 0.0, -1.0]), 1usize), (h(&[0.0; 4]), 0usize)];
        let mut acc = PplAcc::default();
        for (row, t) in &rows {
            let (nll, rs) = nll_row(row, *t, &lut);
            acc.add(nll, rs.argmax, *t as u32, rs.nonfinite);
        }
        assert_eq!(acc.n, 2);
        // row 0: argmax = index 1 = target; row 1: a 4-way tie resolves to the lowest index (0) = target
        assert_eq!(acc.top1, 2);
        assert!((acc.mean_nll() - 0.9132420298405429).abs() < 1e-9);
        assert!((acc.ppl() - 2.4923898513482348).abs() < 1e-8);
    }

    #[test]
    fn head_groups_never_exceed_the_decode_width_or_the_scratch() {
        let v = 248_320;
        assert_eq!(head_group_rows(2048 * 12288, v, 2048), 16);          // the served chunk: capped at 16
        assert_eq!(head_group_rows(4095 * 12288, v, 4095), 16);
        assert_eq!(head_group_rows(3 * v + 5, v, 2048), 3);               // scratch holds only 3 rows
        assert_eq!(head_group_rows(v - 1, v, 2048), 0);                   // cannot hold one row (refused upstream)
        assert_eq!(head_group_rows(2048 * 12288, v, 5), 5);               // a 5-row chunk
        assert!(PPL_GROUP_ROWS <= 16, "above 16 rows the lm_head leaves the m<=16 body");
    }

    #[test]
    fn nonfinite_terms_are_counted_not_summed() {
        let mut acc = PplAcc::default();
        acc.add(1.0, 0, 0, false);
        acc.add(f64::NAN, 0, 1, true);
        assert_eq!(acc.n, 2);
        assert_eq!(acc.nonfinite_nll, 1);
        assert_eq!(acc.nonfinite_rows, 1);
        assert_eq!(acc.sum, 1.0);
    }

    #[test]
    fn windows_follow_the_legacy_chunks_rule() {
        assert_eq!(plan_windows(10, 4, 8), vec![(0, 4), (4, 4), (8, 2)]);   // short tail >= 2 scored
        assert_eq!(plan_windows(9, 4, 8), vec![(0, 4), (4, 4)]);           // 1-token tail dropped
        assert_eq!(plan_windows(100, 4, 2), vec![(0, 4), (4, 4)]);         // max_windows
        assert!(plan_windows(1, 4, 8).is_empty());
        assert!(plan_windows(0, 4, 8).is_empty());
    }

    #[test]
    fn row_targets_shift_by_one_and_end_unscored() {
        let win = [10u32, 11, 12, 13];
        assert_eq!(row_targets(&win, 0, 4), vec![11, 12, 13, NO_TARGET]);
        // a chunk boundary inside the window: the last row of chunk 0 still has its target
        assert_eq!(row_targets(&win, 0, 2), vec![11, 12]);
        assert_eq!(row_targets(&win, 2, 4), vec![13, NO_TARGET]);
        // n-1 scored positions over any chunking
        let scored: usize = [(0usize, 3usize), (3, 4)].iter()
            .map(|&(a, b)| row_targets(&win, a, b).iter().filter(|&&t| t != NO_TARGET).count()).sum();
        assert_eq!(scored, win.len() - 1);
    }

    fn argv(s: &[&str]) -> Vec<String> { s.iter().map(|x| x.to_string()).collect() }

    #[test]
    fn flag_parsing_defaults_and_overrides() {
        let a = PplArgs::from_args(&argv(&["--perplexity", "--model-dir", "/m", "--text", "t.txt"])).unwrap();
        assert_eq!((a.window, a.max_windows, a.max_pos), (1024, 8, 4096));
        assert_eq!(a.chunk, crate::exl3_serve::DEFAULT_PREFILL_CHUNK);
        assert_eq!(a.ref_windows, 0);
        let a = PplArgs::from_args(&argv(&["--perplexity", "--model-dir=/m", "--text", "t.txt", "--window", "2048",
            "--max-windows=5", "--max-seq-len", "4096", "--prefill-chunk", "512", "--ppl-ref-windows", "1",
            "--ppl-csv", "o.csv", "--ppl-dump-ids", "i.json", "--ppl-label", "x"])).unwrap();
        assert_eq!((a.window, a.max_windows, a.max_pos, a.chunk, a.ref_windows), (2048, 5, 4096, 512, 1));
        assert_eq!((a.csv.as_deref(), a.dump_ids.as_deref(), a.label.as_str()), (Some("o.csv"), Some("i.json"), "x"));
    }

    // ---- S-B9-REL-PPL2 W4: the registered options and what the tool refuses ----------------------

    const TOOL_FLAGS: [&str; 12] = ["perplexity", "text", "window", "max-windows", "ppl-ids", "ppl-csv",
        "ppl-dump-ids", "ppl-ref-windows", "ppl-label", "ppl-min-top1", "ppl-min-ppl", "ppl-max-ppl"];

    #[test]
    fn every_perplexity_option_is_registered_typed_and_listed_once() {
        let help = crate::opts::help_advanced();
        let diag = crate::opts::help_text(true);
        for f in TOOL_FLAGS {
            let id = crate::opts::lookup(f).unwrap_or_else(|| panic!("--{f} is not registered"));
            let o = id.def();
            assert_eq!(o.env, "", "--{f}: a new option has no environment variable");
            assert_eq!(o.cat, crate::opts::Cat::Ppl, "--{f}");
            assert_eq!(o.scope, crate::opts::Scope::Local, "--{f}: one process, never shipped to a node");
            assert!(!matches!(o.ty, crate::opts::Ty::Internal(_)), "--{f} must be a command-line option");
            // listed exactly once in --help, and not in --help-diag
            let lines = help.lines().filter(|l| l.trim_start().starts_with(&format!("--{f} ")) || l.trim() == format!("--{f}")).count();
            assert_eq!(lines, 1, "--{f} must be listed once in --help ({lines})");
            assert!(!diag.contains(&format!("    --{f} ")), "--{f} must not be a --help-diag option");
        }
        // typed with ranges: integers are ranged, paths and the label are text
        use crate::opts::Ty;
        assert_eq!(crate::opts::lookup("window").unwrap().def().ty, Ty::IntRange(1, i64::MAX));
        assert_eq!(crate::opts::lookup("max-windows").unwrap().def().ty, Ty::IntRange(1, i64::MAX));
        assert_eq!(crate::opts::lookup("ppl-ref-windows").unwrap().def().ty, Ty::IntRange(0, i64::MAX));
        assert_eq!(crate::opts::lookup("perplexity").unwrap().def().ty, Ty::Flag);
        for f in ["text", "ppl-ids", "ppl-csv", "ppl-dump-ids"] {
            assert_eq!(crate::opts::lookup(f).unwrap().def().ty, Ty::Path, "--{f}");
        }
        // every registered --ppl-* name is one of ours (the refusal's list is the registry's)
        let mut reg: Vec<&str> = ppl_flag_names();
        reg.sort();
        let mut want: Vec<&str> = TOOL_FLAGS.iter().copied().filter(|f| f.starts_with("ppl-")).collect();
        want.sort();
        assert_eq!(reg, want);
    }

    #[test]
    fn registry_ranges_refuse_through_parse_cli() {
        // The first bad option ends the parse before anything is written to the process-global store.
        let bad = |a: &[&str]| crate::opts::parse_cli(&argv(a));
        assert_eq!(bad(&["x", "--window", "0"]).unwrap_err(), "--window takes an integer >= 1 (got '0')");
        assert_eq!(bad(&["x", "--max-windows", "0"]).unwrap_err(), "--max-windows takes an integer >= 1 (got '0')");
        assert_eq!(bad(&["x", "--ppl-ref-windows", "-1"]).unwrap_err(), "--ppl-ref-windows takes an integer >= 0 (got '-1')");
        assert!(bad(&["x", "--window", "abc"]).is_err());
        assert!(bad(&["x", "--window=1.5"]).is_err());
        assert!(bad(&["x", "--ppl-csv"]).is_err(), "a path option needs a value");
        assert!(bad(&["x", "--ppl-min-top1", "high"]).is_err(), "a float option takes a number");
        // the pure parser agrees with parse_cli on every one of them (it is the same walk)
        assert_eq!(crate::opts::parse_args(&argv(&["x", "--window", "0"])).unwrap_err(),
                   "--window takes an integer >= 1 (got '0')");
        // and a good line is typed, canonical and last-wins
        let v = crate::opts::parse_args(&argv(&["x", "--window", "512", "--window=+1024", "--max-windows", "3"])).unwrap();
        let id = crate::opts::lookup("window").unwrap();
        assert_eq!(v.iter().rev().find(|(i, _)| *i == id).unwrap().1.as_deref(), Some("1024"));
    }

    #[test]
    fn an_unknown_ppl_name_is_refused_not_dropped() {
        // parse_cli alone skips a name it does not know: that is the defect (the reference was dropped)
        assert!(crate::opts::parse_cli(&argv(&["x", "--ppl-ref-windowz", "3"])).is_ok());
        let base = ["--perplexity", "--model-dir", "/m", "--text", "t"];
        let with = |extra: &[&str]| { let mut v: Vec<&str> = base.to_vec(); v.extend_from_slice(extra); PplArgs::from_args(&argv(&v)) };
        let e = format!("{:#}", with(&["--ppl-ref-windowz", "3"]).unwrap_err());
        assert!(e.contains("unknown option --ppl-ref-windowz") && e.contains("Did you mean --ppl-ref-windows?"), "{e}");
        let e = format!("{:#}", with(&["--ppl-bogus=1"]).unwrap_err());
        assert!(e.contains("unknown option --ppl-bogus") && e.contains("--ppl-csv"), "{e}");
        assert!(!e.contains("Did you mean"), "no suggestion for a name that is not close to any: {e}");
        assert!(with(&["--ppl"]).is_err(), "the bare namespace name is not an option either");
        // the registered names are all accepted
        assert!(with(&["--ppl-ref-windows", "1", "--ppl-csv", "o.csv", "--ppl-dump-ids", "i.json", "--ppl-label", "x",
                       "--ppl-min-top1", "0.3", "--ppl-min-ppl", "1.5", "--ppl-max-ppl", "50"]).is_ok());
        // a name that merely starts like another tool's flag is not in this namespace
        assert!(with(&["--prefill-chunk", "512"]).is_ok());
    }

    #[test]
    fn kv_cache_refusals() {
        let base = ["--perplexity", "--model-dir", "/m", "--text", "t"];
        let with = |kv: &str| { let mut v: Vec<&str> = base.to_vec(); v.extend_from_slice(&["--kv-cache", kv]); PplArgs::from_args(&argv(&v)) };
        for ok in ["f32", "f16", "fp8", "q8"] { assert!(with(ok).is_ok(), "{ok}"); }
        let e = format!("{:#}", with("q4x").unwrap_err());
        assert!(e.contains("q4x") && e.contains("deliberately broken"), "{e}");
        for nv in ["bf16", "q4", "tq", "tq3", "k8v4", "k8v8"] {
            let e = format!("{:#}", with(nv).unwrap_err());
            assert!(e.contains(&format!("--kv-cache {nv} is an NVFP4 KV format")), "{nv}: {e}");
        }
        assert!(with("int4").is_err());
        assert!(PplArgs::from_args(&argv(&base)).is_ok(), "no --kv-cache = the served default");
    }

    #[test]
    fn text_and_ppl_ids_are_alternatives() {
        let both = PplArgs::from_args(&argv(&["--perplexity", "--model-dir", "/m", "--text", "t.txt", "--ppl-ids", "i.json"]));
        let e = format!("{:#}", both.unwrap_err());
        assert!(e.contains("--text and --ppl-ids are alternatives"), "{e}");
        let ids = PplArgs::from_args(&argv(&["--perplexity", "--model-dir", "/m", "--ppl-ids", "i.json"])).unwrap();
        assert_eq!((ids.text.as_deref(), ids.ids.as_deref()), (None, Some("i.json")));
        assert!(PplArgs::from_args(&argv(&["--perplexity", "--model-dir", "/m"])).is_err());
    }

    #[test]
    fn defaults_are_the_registry_rows() {
        let a = PplArgs::from_args(&argv(&["--perplexity", "--model-dir", "/m", "--text", "t"])).unwrap();
        assert_eq!((a.window, a.max_windows, a.ref_windows, a.label.as_str()), (1024, 8, 0, "ppl"));
        assert_eq!(a.bounds, RunBounds::defaults());
        assert_eq!(a.bounds, RunBounds { min_ppl: 1.05, max_ppl: 1000.0, min_top1: 0.02 });
        for (flag, bad) in [("--ppl-min-top1", "2"), ("--ppl-min-top1", "-0.1"), ("--ppl-min-ppl", "0.5"),
                            ("--ppl-max-ppl", "1.0"), ("--ppl-min-ppl", "nan"), ("--ppl-max-ppl", "inf")] {
            let r = PplArgs::from_args(&argv(&["--perplexity", "--model-dir", "/m", "--text", "t", flag, bad]));
            assert!(r.is_err(), "{flag} {bad}");
        }
    }

    // ---- S-B9-REL-PPL2 W5: each check red once on a deliberately broken input, then green -------

    const V: usize = 40; // a toy vocabulary row

    /// a distinct, zero-free toy row (f16 bit patterns 0x3c00.. upward)
    fn real(seed: u16) -> Vec<u16> { (0..V as u16).map(|i| 0x3c00 + seed * 64 + i).collect() }
    fn cat(rs: &[Vec<u16>]) -> Vec<u16> { rs.iter().flatten().copied().collect() }

    #[test]
    fn group_scan_red_on_unwritten_sparse_and_duplicated_rows_green_on_real_ones() {
        let (a, b) = (real(1), real(2));
        // GREEN: distinct rows with no zeros
        assert_eq!(group_defects(&cat(&[a.clone(), b.clone(), real(3)]), V, 3), GroupDefects::default());
        // GREEN: a legitimate row may carry a few zeros (padding columns): 2 of 40 is 5%, not "more than"
        let mut few = real(4);
        few[0] = 0; few[7] = 0;
        assert_eq!(group_defects(&cat(&[a.clone(), few]), V, 2), GroupDefects::default());
        // RED 1: a kernel that wrote nothing leaves the zero poison: an all-zero row
        let d = group_defects(&cat(&[a.clone(), vec![0; V], b.clone()]), V, 3);
        assert_eq!(d, GroupDefects { unwritten: 1, sparse: 0, duplicate: 0 });
        // RED 2: the whole group unwritten (a GEMM that no-ops at this width): every row, and each
        // zero row equals the one before it
        let d = group_defects(&cat(&[vec![0; V], vec![0; V], vec![0; V], vec![0; V]]), V, 4);
        assert_eq!(d, GroupDefects { unwritten: 4, sparse: 0, duplicate: 3 });
        // RED 3: part of the grid never ran: a long run of zeros inside a written row
        let mut half = real(5);
        for x in half.iter_mut().take(V / 2) { *x = 0; }
        assert_eq!(group_defects(&cat(&[a.clone(), half]), V, 2), GroupDefects { unwritten: 0, sparse: 1, duplicate: 0 });
        // RED 4: stale / broadcast output: two neighbours bit-identical
        let d = group_defects(&cat(&[a.clone(), a.clone(), b.clone()]), V, 3);
        assert_eq!(d, GroupDefects { unwritten: 0, sparse: 0, duplicate: 1 });
        // only the first `m` rows are looked at (the tail of the staging buffer is stale by design)
        assert_eq!(group_defects(&cat(&[a.clone(), b.clone(), vec![0; V]]), V, 2), GroupDefects::default());
        // a single row: nothing to compare with, but an unwritten row is still seen
        assert_eq!(group_defects(&vec![0; V], V, 1), GroupDefects { unwritten: 1, sparse: 0, duplicate: 0 });
    }

    #[test]
    fn run_bounds_red_on_the_v1_garbage_and_on_a_leak_green_on_the_real_cells() {
        let b = RunBounds::defaults();
        // RED: the first version's 4.05 cell (a 6-bit lm_head through a kernel that wrote nothing)
        let e = check_top1(51175, 0.0003, &b).unwrap_err();
        assert!(e.contains("top-1 accuracy 0.0003") && e.contains("NOT valid"), "{e}");
        let e = check_ppl(724390.9645, &b).unwrap_err();
        assert!(e.contains("outside 1.05 .. 1000"), "{e}");
        // RED: a perplexity near 1 = the model was told its answers (causal-mask / offset defect)
        assert!(check_ppl(1.0003, &b).is_err());
        // RED: not a number
        assert!(check_ppl(f64::NAN, &b).is_err() && check_ppl(f64::INFINITY, &b).is_err());
        assert!(check_top1(1000, f64::NAN, &b).is_err());
        // RED: the floor binds at its edge, not at zero
        assert!(check_top1(1000, 0.0199, &b).is_err());
        // GREEN: the 18 cells of the first session (PPL 2.31 .. 6.42, top-1 0.60 .. 0.77)
        for (ppl, t1) in [(3.5264, 0.7465), (6.3994, 0.6087), (2.3135, 0.7691), (3.5409, 0.7468), (6.4184, 0.6108),
                          (2.3179, 0.7698), (6.4239, 0.6104), (3.5355, 0.7466)] {
            check_ppl(ppl, &b).unwrap();
            check_top1(51175, t1, &b).unwrap();
        }
        // a handful of tokens says nothing about a model: the floor waits for TOP1_MIN_SAMPLE
        assert!(check_top1(TOP1_MIN_SAMPLE - 1, 0.0, &b).is_ok());
        assert!(check_top1(TOP1_MIN_SAMPLE, 0.0, &b).is_err());
        // tighter, class-specific bounds from a harness (code-like text)
        let code = RunBounds { min_ppl: 1.5, max_ppl: 12.0, min_top1: 0.5 };
        assert!(check_ppl(3.5264, &code).is_ok() && check_top1(51175, 0.7465, &code).is_ok());
        assert!(check_ppl(13.0, &code).is_err() && check_top1(51175, 0.45, &code).is_err());
        // --ppl-min-top1 0 switches the floor off, by the user's choice
        let off = RunBounds { min_top1: 0.0, ..b };
        assert!(check_top1(1000, 0.0, &off).is_ok());
    }

    // ---- S-B9-REL-PPL2 1b: a failing check ENDS A RUN (synthetic device / windows, no GPU) ------

    /// A synthetic lm_head: `buf` plays the device logits buffer. `stale_init` models uninitialised
    /// device memory (`alloc_zeros` does not zero): distinct non-zero rows.
    #[derive(Clone, Copy, PartialEq)]
    enum Fault { None, WritesNothing, WritesHalfTheColumns, BroadcastsRow0, WidthDiffers }

    struct FakeHead {
        v: usize,
        buf: Vec<u16>,
        fault: Fault,
        poison_calls: usize,
        groups: usize,
    }

    fn healthy_val(r: usize, j: usize) -> u16 { 0x3c00 + ((r * 13 + j * 5) % 400) as u16 }

    impl FakeHead {
        fn new(v: usize, r_max: usize, fault: Fault) -> FakeHead {
            let buf = (0..r_max * v).map(|i| 0x3d00 + (i % 200) as u16).collect(); // stale, non-zero
            FakeHead { v, buf, fault, poison_calls: 0, groups: 0 }
        }
    }

    impl HeadDevice for FakeHead {
        fn poison(&mut self, _m: usize) -> Result<()> {
            self.poison_calls += 1;
            self.buf.iter_mut().for_each(|x| *x = 0);
            Ok(())
        }
        fn run_group(&mut self, r0: usize, m: usize, host: &mut [u16]) -> Result<()> {
            self.groups += 1;
            let v = self.v;
            for r in 0..m {
                for j in 0..v {
                    let val = match self.fault {
                        Fault::WritesNothing => continue,
                        Fault::WritesHalfTheColumns if j >= v / 2 => continue,
                        Fault::BroadcastsRow0 => healthy_val(r0, j),
                        _ => healthy_val(r0 + r, j),
                    };
                    self.buf[r * v + j] = val;
                }
            }
            host[..self.buf.len()].copy_from_slice(&self.buf);
            Ok(())
        }
        fn run_single(&mut self, r0: usize, out: &mut [u16]) -> Result<()> {
            for j in 0..self.v {
                out[j] = healthy_val(r0, j) + (self.fault == Fault::WidthDiffers) as u16;
            }
            Ok(())
        }
        fn bits(&self) -> i32 { 6 }
    }

    fn head_state(v: usize, r_max: usize, targets: Vec<u32>) -> HeadState {
        let mut st = HeadState::new(r_max, 8, v);
        st.targets = targets;
        st
    }

    #[test]
    fn a_head_that_writes_nothing_ends_the_run_and_the_poison_is_what_shows_it() {
        let (v, r_max) = (64usize, 4usize);
        let targets: Vec<u32> = (0..10).map(|i| i as u32).collect();
        // GREEN: a healthy head scores every row, and the width check ran and agreed
        let mut dev = FakeHead::new(v, r_max, Fault::None);
        let mut st = head_state(v, r_max, targets.clone());
        head_chunk(&mut dev, &mut st, 10).unwrap();
        assert_eq!(st.acc.n, 10);
        assert_eq!(st.counters.groups, 3);                      // 4 + 4 + 2 rows
        assert_eq!(dev.poison_calls, 3, "every group is poisoned before its launch");
        assert_eq!(st.counters.width_check, Some((0, 4)));
        assert_eq!((st.counters.unwritten_rows, st.counters.sparse_rows, st.counters.duplicate_rows), (0, 0, 0));
        // RED: the launch writes nothing; the buffer held stale (distinct, non-zero) rows. Only the
        // poison turns that into the all-zero rows the scan refuses.
        let mut dev = FakeHead::new(v, r_max, Fault::WritesNothing);
        let mut st = head_state(v, r_max, targets.clone());
        let e = head_chunk(&mut dev, &mut st, 10).unwrap_err().to_string();
        assert!(e.contains("were never written") && e.contains("NOT valid") && e.contains("lm_head bits 6"), "{e}");
        assert_eq!(st.counters.groups, 1, "the run stops at the first group");
        assert_eq!(st.acc.n, 0, "nothing of the broken group was scored");
        // RED: only part of the grid ran
        let mut dev = FakeHead::new(v, r_max, Fault::WritesHalfTheColumns);
        let e = head_chunk(&mut dev, &mut head_state(v, r_max, targets.clone()), 10).unwrap_err().to_string();
        assert!(e.contains("mostly zero"), "{e}");
        // RED: row 0 broadcast into every row
        let mut dev = FakeHead::new(v, r_max, Fault::BroadcastsRow0);
        let e = head_chunk(&mut dev, &mut head_state(v, r_max, targets.clone()), 10).unwrap_err().to_string();
        assert!(e.contains("bit-identical"), "{e}");
        // RED: the m = 1 recomputation disagrees with the group
        let mut dev = FakeHead::new(v, r_max, Fault::WidthDiffers);
        let e = head_chunk(&mut dev, &mut head_state(v, r_max, targets), 10).unwrap_err().to_string();
        assert!(e.contains("width check FAILED"), "{e}");
        // a group with no scored target is not launched at all
        let mut dev = FakeHead::new(v, r_max, Fault::WritesNothing);
        let mut st = head_state(v, r_max, vec![NO_TARGET; 4]);
        head_chunk(&mut dev, &mut st, 4).unwrap();
        assert_eq!((dev.poison_calls, dev.groups, st.acc.n), (0, 0, 0));
    }

    #[test]
    fn the_poison_plan_covers_what_every_stage_of_every_head_class_writes() {
        // Every stage output of the chain must be zeroed before the launch, for the fused 3/4/5-bit
        // head, the unfused 6-bit head (exl3_hmma_gemm -> y_raw -> exl3_had_svh: the M1 hole of the
        // part-1 review, where only the final buffer was zeroed) and a split-K head.
        for (fused, ks) in [(true, 1u32), (false, 1), (true, 2), (false, 4)] {
            let plan = head_poison_plan(ks);
            for out in chain_stage_outputs(fused, ks) {
                assert!(plan.contains(&out), "fused={fused} ks={ks}: the chain writes {out:?} but the plan does not zero it");
            }
        }
        assert!(chain_stage_outputs(false, 1).contains(&HeadBuf::YRaw));
        assert!(!chain_stage_outputs(true, 1).contains(&HeadBuf::YRaw), "the fused path never touches y_raw");
        assert!(chain_stage_outputs(false, 2).contains(&HeadBuf::KsWs) && !head_poison_plan(1).contains(&HeadBuf::KsWs));
        assert!(head_poison_plan(1).contains(&HeadBuf::Logits));
    }

    fn acc_of(n: usize, top1: f64, mean_nll: f64) -> PplAcc {
        PplAcc { sum: mean_nll * n as f64, n, top1: (top1 * n as f64).round() as usize, ..Default::default() }
    }

    fn code_bounds() -> RunBounds { RunBounds { min_ppl: 1.5, max_ppl: 12.0, min_top1: 0.5 } }

    #[test]
    fn one_garbage_window_among_good_ones_fails_the_run() {
        let mut j = RunJudge::new(code_bounds());
        for wi in 1..=24 {
            j.after_window(wi, 2047, &acc_of(2047, 0.75, 1.26)).unwrap();      // code-like: PPL 3.5, top-1 0.75
        }
        // arithmetic of the review: the garbage window barely moves the running totals ...
        let mut cumulative = j.total.clone();
        cumulative.merge(&acc_of(2047, 0.001, 13.4));
        assert!(cumulative.top1_frac() > 0.7 && cumulative.ppl() < 6.0, "{} {}", cumulative.top1_frac(), cumulative.ppl());
        assert!(check_top1(cumulative.n, cumulative.top1_frac(), &code_bounds()).is_ok());
        assert!(check_ppl(cumulative.ppl(), &code_bounds()).is_ok(), "the mean alone passes both bounds");
        // ... so the window is judged on its own
        let e = j.after_window(25, 2047, &acc_of(2047, 0.001, 13.4)).unwrap_err();
        assert!(e.contains("window 25 (alone)") && e.contains("top-1 accuracy 0.0010"), "{e}");
        // and so is a window that is only too perplexing (top-1 fine)
        let mut j = RunJudge::new(code_bounds());
        let e = j.after_window(1, 2047, &acc_of(2047, 0.9, 7.0)).unwrap_err();     // PPL 1097 > 12
        assert!(e.contains("window 1 (alone)") && e.contains("above --ppl-max-ppl 12"), "{e}");
    }

    #[test]
    fn a_broken_head_is_known_after_its_first_window() {
        let mut j = RunJudge::new(RunBounds::defaults());
        let e = j.after_window(1, 2047, &acc_of(2047, 0.0003, 13.49)).unwrap_err();
        assert!(e.contains("window 1"), "{e}");
        // windows too short for the per-window reading: the running total takes over
        let mut j = RunJudge::new(RunBounds::defaults());
        assert!(j.after_window(1, 31, &acc_of(31, 0.0, 12.0)).is_ok(), "31 tokens: no reading yet");
        assert!(j.after_window(2, 31, &acc_of(31, 0.0, 12.0)).is_ok(), "62 tokens: no reading yet");
        let e = j.after_window(3, 31, &acc_of(31, 0.0, 12.0)).unwrap_err();
        assert!(e.contains("after window 3"), "{e}");
        // a window that scored the wrong number of positions, or non-finite terms, fails by itself
        let mut j = RunJudge::new(RunBounds::defaults());
        assert!(j.after_window(1, 2047, &acc_of(2000, 0.7, 1.0)).unwrap_err().contains("2000 positions scored"));
        let mut bad = acc_of(2047, 0.7, 1.0);
        bad.nonfinite_nll = 1;
        assert!(j.after_window(1, 2047, &bad).unwrap_err().contains("non-finite"));
    }

    #[test]
    fn the_final_bounds_end_a_run_every_window_of_which_passed() {
        // PPL ~ 1.0003 in every window (the model was told its answers), top-1 0.99: no window is
        // above the ceiling and none below the top-1 floor; only the conclusion can say no.
        let mut j = RunJudge::new(RunBounds::defaults());
        for wi in 1..=3 { j.after_window(wi, 2047, &acc_of(2047, 0.99, 0.0003)).unwrap(); }
        let head = HeadCounters { groups: 384, ..Default::default() };
        let e = j.conclude(&head).unwrap_err();
        assert!(e.contains("perplexity 1.0003") && e.contains("outside 1.05 .. 1000"), "{e}");
        // the path is cut: a run that never launched the head cannot conclude
        let mut j = RunJudge::new(RunBounds::defaults());
        j.after_window(1, 2047, &acc_of(2047, 0.75, 1.26)).unwrap();
        assert!(j.conclude(&HeadCounters::default()).unwrap_err().contains("never ran"));
        // GREEN: the first session's real cells conclude, and say whether the floor applied
        for (ppl, t1) in [(3.5264f64, 0.7465), (6.3994, 0.6087), (2.3135, 0.7691)] {
            let mut j = RunJudge::new(RunBounds::defaults());
            j.after_window(1, 2047, &acc_of(2047, t1, ppl.ln())).unwrap();
            let s = j.conclude(&head).unwrap();
            assert!((s.ppl - ppl).abs() < 1e-9 && s.top1_floor_applied, "{s:?}");
        }
        // a run under 64 tokens concludes without a top-1 floor and SAYS so
        let mut j = RunJudge::new(RunBounds::defaults());
        j.after_window(1, 9, &acc_of(9, 0.0, 3.0)).unwrap();
        let s = j.conclude(&head).unwrap();
        assert!(!s.top1_floor_applied);
    }

    struct FakeWindows {
        script: Vec<PplAcc>,
        scored: usize,
    }

    impl WindowSource for FakeWindows {
        fn score_window(&mut self, wi: usize, win: &[u32]) -> Result<WindowScore> {
            self.scored += 1;
            Ok(WindowScore { acc: self.script[wi].clone(), nll: vec![1.0; win.len()], secs: 0.0 })
        }
        fn counters(&self) -> HeadCounters { HeadCounters { groups: 10, rows_checked: 10, ..Default::default() } }
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("ppl2_run_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Drive the whole post-device path (score_all -> finish_run) over `script`, with outputs in a
    /// fresh temp dir. Returns (result, stdout text, dir).
    fn drive(tag: &str, script: Vec<PplAcc>, bounds: RunBounds) -> (Result<()>, String, std::path::PathBuf, usize) {
        let dir = temp_dir(tag);
        let n_win = script.len();
        let n = 5usize; // tokens per window; acc.n is whatever the script says
        let ids: Vec<u32> = (0..(n * n_win) as u32).collect();
        let wins: Vec<(usize, usize)> = (0..n_win).map(|i| (i * n, n)).collect();
        let a = PplArgs {
            dir: "/m".into(), text: Some("t".into()), ids: None, window: n, max_windows: n_win, max_pos: 4096,
            chunk: 4095, ref_windows: 0,
            csv: Some(dir.join("o.csv").to_string_lossy().into_owned()),
            dump_ids: Some(dir.join("ids.json").to_string_lossy().into_owned()),
            label: "t".into(), bounds,
        };
        let mut src = FakeWindows { script: script.into_iter().map(|mut a| { a.n = a.n; a }).collect(), scored: 0 };
        let mut judge = RunJudge::new(bounds);
        let mut out: Vec<u8> = Vec::new();
        // the fake windows report acc.n scored positions; the judge expects n - 1 = 4 per window
        let res = (|| {
            let (nll, failure) = score_all(&mut src, &wins, &ids, &mut judge, &mut out)?;
            let fi = FinishInputs { a: &a, kv_name: "q8", wins: &wins, ids: &ids, vocab: 1000, src: "t",
                                    per_window_nll: &nll, dec_nll: &[], head: src.counters(), elapsed_s: 0.0 };
            finish_run(&fi, &judge, failure, &mut out)
        })();
        (res, String::from_utf8(out).unwrap(), dir, src.scored)
    }

    fn w4(top1_hits: usize, nll: f64) -> PplAcc {
        // 4 scored positions per window (n = 5 tokens), floor-exempt (< 64): only the conclusion judges
        PplAcc { sum: nll * 4.0, n: 4, top1: top1_hits, ..Default::default() }
    }

    #[test]
    fn a_failed_run_prints_no_result_line_and_owns_no_output_name() {
        let b = RunBounds::defaults();
        // GREEN: three plausible windows -> RESULT once, csv and ids under their real names
        let (r, out, dir, scored) = drive("ok", vec![w4(3, 1.2), w4(3, 1.3), w4(2, 1.1)], b);
        r.unwrap();
        assert_eq!(scored, 3);
        assert_eq!(out.matches("RESULT: tokens=12").count(), 1, "{out}");
        assert!(out.contains("PPL_HEAD_CHECK groups=10") && out.contains("PPL_BOUNDS"), "{out}");
        assert!(dir.join("o.csv").exists() && dir.join("ids.json").exists());
        assert!(!dir.join("o.csv.INVALID").exists());
        let csv = std::fs::read_to_string(dir.join("o.csv")).unwrap();
        assert!(csv.starts_with("label,window,pos,target,nll,nll_decode"), "{csv}");

        // RED (final bounds): every window has PPL ~ 1.0 -> no window is out of range alone, the run is
        let (r, out, dir, scored) = drive("leak", vec![w4(4, 0.0003), w4(4, 0.0003)], b);
        let e = r.unwrap_err().to_string();
        assert!(e.contains("perplexity 1.0003") && scored == 2, "{e}");
        assert!(!out.contains("RESULT:"), "a failed run prints no RESULT line:\n{out}");
        assert!(!dir.join("o.csv").exists() && !dir.join("ids.json").exists(), "no output under the real names");
        let inv = std::fs::read_to_string(dir.join("o.csv.INVALID")).unwrap();
        assert!(inv.starts_with("# INVALID RUN - perplexity 1.0003"), "{inv}");
        assert!(out.contains("PPL_HEAD_CHECK") && out.contains("PPL_STATS"), "the diagnostics are still printed");

        // RED (a window): the second window is the model's output of an lm_head that is not this
        // model's; the run stops there (the third window is never scored)
        let bad = PplAcc { sum: 13.0 * 80.0, n: 80, top1: 0, ..Default::default() };
        let mut script = vec![acc_of(80, 0.75, 1.2), bad, acc_of(80, 0.75, 1.2)];
        // windows are judged against n - 1 = 4 positions: give the judge's expectation what the script says
        for a in script.iter_mut() { a.n = 4; }
        script[1] = PplAcc { sum: 13.0 * 4.0, n: 4, top1: 0, ..Default::default() };
        let (r, out, dir, scored) = drive("bad_window", script, b);
        // 4 tokens are below the per-window gate, so the conclusion decides: PPL e^(mean of 1.2, 13, 1.2)
        let e = r.unwrap_err().to_string();
        assert!(!out.contains("RESULT:") && scored == 3, "{e}\n{out}");
        assert!(!dir.join("o.csv").exists() && dir.join("o.csv.INVALID").exists());
    }

    #[test]
    fn a_window_judged_alone_stops_the_run_where_it_lands() {
        let b = RunBounds::defaults();
        // 65 scored positions per window so the per-window gate applies
        let big = |hits: usize, nll: f64| PplAcc { sum: nll * 65.0, n: 65, top1: hits, ..Default::default() };
        let dir = temp_dir("stop");
        let n = 66usize;
        let ids: Vec<u32> = (0..(n * 3) as u32).collect();
        let wins: Vec<(usize, usize)> = (0..3).map(|i| (i * n, n)).collect();
        let mut src = FakeWindows { script: vec![big(48, 1.2), big(0, 13.0), big(48, 1.2)], scored: 0 };
        let mut judge = RunJudge::new(b);
        let mut out: Vec<u8> = Vec::new();
        let (nll, failure) = score_all(&mut src, &wins, &ids, &mut judge, &mut out).unwrap();
        assert_eq!(src.scored, 2, "the third window is never scored");
        assert_eq!(nll.len(), 2);
        assert!(failure.as_deref().unwrap().contains("window 2 (alone)"), "{failure:?}");
        let _ = dir;
    }

    // ---- 1b item 8 / 6: the refusals added to the tool ---------------------------------------

    #[test]
    fn a_value_less_path_option_that_would_swallow_the_next_flag_is_refused() {
        let tool = |extra: &[&str]| {
            let mut v = vec!["--perplexity", "--model-dir", "/m"];
            v.extend_from_slice(extra);
            PplArgs::from_args(&argv(&v))
        };
        let e = format!("{:#}", tool(&["--text", "--mtp", "off"]).unwrap_err());
        assert!(e.contains("--text needs a value") && e.contains("--mtp") && e.contains("swallow"), "{e}");
        let e = format!("{:#}", tool(&["--text", "t", "--ppl-csv", "--window", "2048"]).unwrap_err());
        assert!(e.contains("--ppl-csv needs a value") && e.contains("--window"), "{e}");
        assert!(tool(&["--ppl-label", "--ppl-min-top1", "0.1", "--text", "t"]).is_err());
        let e = format!("{:#}", PplArgs::from_args(&argv(&["--perplexity", "--model-dir", "--kv-cache", "q8", "--text", "t"])).unwrap_err());
        assert!(e.contains("--model-dir needs a value"), "{e}");
        // an explicit `=` cannot swallow anything; a normal line passes
        assert!(tool(&["--text=--odd-name.txt"]).is_ok());
        assert!(tool(&["--text", "t.txt", "--ppl-csv", "o.csv", "--window", "512"]).is_ok());
    }

    #[test]
    fn a_repeated_option_with_different_values_is_refused() {
        let tool = |extra: &[&str]| {
            let mut v = vec!["--perplexity", "--model-dir", "/m", "--text", "t"];
            v.extend_from_slice(extra);
            PplArgs::from_args(&argv(&v))
        };
        // validated as q8, measured as q4x (the bridge reads the first): refused instead
        let e = format!("{:#}", tool(&["--kv-cache", "q4x", "--kv-cache", "q8"]).unwrap_err());
        assert!(e.contains("--kv-cache is given twice with different values ('q4x' and 'q8')"), "{e}");
        assert!(tool(&["--window", "1024", "--window=2048"]).is_err());
        assert!(tool(&["--ppl-label", "a", "--ppl-label", "b"]).is_err());
        // the same value twice is not a conflict
        assert!(tool(&["--kv-cache", "q8", "--kv-cache=q8", "--window", "512", "--window", "512"]).is_ok());
        // the mode flag itself may repeat
        assert!(tool(&["--perplexity"]).is_ok());
    }

    #[test]
    fn the_unknown_name_hint_lists_every_option_of_the_tool() {
        let e = format!("{:#}", PplArgs::from_args(&argv(&["--perplexity", "--model-dir", "/m", "--text", "t", "--ppl-zzzz-qqqq", "1"])).unwrap_err());
        for f in ["--perplexity", "--text", "--window", "--max-windows", "--ppl-ids", "--ppl-max-ppl", "--model-dir"] {
            assert!(e.contains(f), "{f} missing from: {e}");
        }
    }

    #[test]
    fn flag_parsing_rejects_nonsense() {
        let base = ["--perplexity", "--model-dir", "/m", "--text", "t"];
        let with = |extra: &[&str]| { let mut v: Vec<&str> = base.to_vec(); v.extend_from_slice(extra); PplArgs::from_args(&argv(&v)) };
        assert!(with(&["--window", "abc"]).is_err());
        assert!(with(&["--window", "1"]).is_err());
        assert!(with(&["--window", "8192"]).is_err(), "window > max-seq-len must be refused");
        assert!(with(&["--max-windows", "0"]).is_err());
        assert!(PplArgs::from_args(&argv(&["--perplexity", "--model-dir", "/m"])).is_err(), "no text");
        assert!(PplArgs::from_args(&argv(&["--perplexity", "--text", "t"])).is_err(), "no model dir");
    }
