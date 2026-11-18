//! Unit tests of the options registry: parsing, help snapshots, env-read scans of the sources. Map: `mod.rs`.

    use super::*;

    // ---- F7 item 2: the = form is accepted for EVERY row (a value the row itself accepts) ----
    #[test]
    fn f7_equals_form_accepted_for_every_row() {
        for o in REG.iter() {
            if matches!(o.ty, Ty::Internal(_)) { continue; }
            // the row's default is NOT always a parseable value (a Choice may default to a
            // meta-word like "all" outside the list; an Int may default to "unset"): pick the
            // first value canon accepts for the row's type
            let valid = match o.ty {
                Ty::Flag | Ty::Bool | Ty::IntFlag => "on".to_string(),
                Ty::Text | Ty::Path => if o.def.is_empty() { "x".to_string() } else { o.def.to_string() },
                Ty::Choice(list) => if list.contains(&o.def) { o.def.to_string() } else { list[0].to_string() },
                _ => if o.def.parse::<f64>().is_ok() { o.def.to_string() } else { "1".to_string() },
            };
            let arg = format!("--{}={valid}", o.flag);
            assert!(parse_args(&[arg.clone()]).is_ok(), "{arg}");
        }
    }

    // ---- F7b: unknown flags warn, never refuse -----------------------------------------------
    #[test]
    fn f7b_unknown_flags_warn_with_near_names() {
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(unknown_warnings(&a(&["bin", "--max-bach", "4"])),
                   vec!["unknown option --max-bach: ignored; did you mean --max-batch?"]);
        // nothing close: no suggestion at all
        assert_eq!(unknown_warnings(&a(&["bin", "--qqqqqqqqqq"])),
                   vec!["unknown option --qqqqqqqqqq: ignored"]);
        // registered rows, hand-parsed flags, value slots and the separator never warn
        assert!(unknown_warnings(&a(&["bin", "--max-batch", "4", "--bench-mtp", "--model-dir", "/m"])).is_empty());
        // the bare separator ends the options: a "--x" after it is positional, not a warning
        assert!(unknown_warnings(&a(&["bin", "--", "--x"])).is_empty());
        // each distinct name exactly once
        assert_eq!(unknown_warnings(&a(&["bin", "--max-bach", "--max-bach"])).len(), 1);
        // argv[0] (the program path) is never judged
        assert!(unknown_warnings(&a(&["/home/x/--weird-path"])).is_empty());
    }

    /// MTPAGREE1 (public v0.7.6 report, Q3): the --otel-* options are read through a closure in
    /// otel::config_from_args, so the literal scan above never saw three of them and the guard printed
    /// "unknown option --otel-include-tokens: ignored" for a value the server applied. All six are
    /// known and value-taking (the value slot after each is not judged).
    #[test]
    fn f7b_otel_options_are_known_and_take_a_value() {
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let w = unknown_warnings(&a(&["bin", "--server", "--otel-endpoint", "http://127.0.0.1:4318",
                                       "--otel-include-tokens", "on", "--otel-batch-size", "256",
                                       "--otel-batch-interval-ms", "50", "--otel-model-id", "m", "--otel-topology", "tp2"]));
        assert!(w.is_empty(), "{w:?}");
    }

    #[test]
    fn f7b_hand_flags_cover_every_argv_literal() {
        // zero false positives, by construction: every "--x" literal in the hand-dispatch files is
        // a registered option or on HAND_FLAGS, so unknown_warnings can never flag one.
        fn flag_literals(src: &str) -> Vec<String> {
            let b = src.as_bytes();
            let mut out = Vec::new();
            let mut i = 0;
            while i + 2 < b.len() {
                if b[i] == b'"' && b[i + 1] == b'-' && b[i + 2] == b'-' {
                    let mut j = i + 3;
                    while j < b.len() && (b[j].is_ascii_lowercase() || b[j].is_ascii_digit() || b[j] == b'-') { j += 1; }
                    if j > i + 3 && j < b.len() && b[j] == b'"' {
                        out.push(src[i + 3..j].to_string());
                        i = j;
                        continue;
                    }
                }
                i += 1;
            }
            out
        }
        let mut hand: Vec<&str> = HAND_FLAGS.iter().map(|(f, _, _)| *f).collect();
        let sorted = hand.is_sorted();
        hand.sort();
        assert!(sorted, "HAND_FLAGS stays sorted");
        for (file, src) in [("src/main.rs", src_text!("main")),
                            ("src/exl3_ppl.rs", src_text!("exl3_ppl")),
                            ("src/exl3_serve.rs", src_text!("exl3_serve")),
                            ("src/lib.rs", src_text!("lib"))] {
            for lit in flag_literals(src) {
                let known = lookup(&lit).is_some() || hand.binary_search(&lit.as_str()).is_ok();
                assert!(known, "{file}: the argv literal --{lit} is neither registered nor on HAND_FLAGS (a warning would fire on our own launch lines)");
            }
        }
    }

    /// S-B9-REL-MTPREPROBE2 W1.4: every decision-affecting spec flag must be Spmd so it
    /// rides TpConfig into the nodes and the boot digest (a Head-scoped spec flag would
    /// silently not ship and the head/nodes would disagree on the schedule). Pins the
    /// whole Cat::Spec class, plus the schedule-affecting Diag drills by name.
    #[test]
    fn spec_decision_flags_are_spmd() {
        for o in REG {
            if o.cat == Cat::Spec {
                assert_eq!(o.scope, Scope::Spmd, "--{} must be Spmd (rides TpConfig + boot digest)", o.flag);
            }
        }
        for f in ["mtp-reprobe", "mtp-reprobe-drill"] {
            let o = REG.iter().find(|o| o.flag == f).unwrap_or_else(|| panic!("missing --{f}"));
            assert_eq!(o.scope, Scope::Spmd, "--{f} must be Spmd (rides TpConfig + boot digest)");
        }
    }

    #[test]
    fn flags_unique_and_well_formed() {
        let mut seen = std::collections::HashSet::new();
        for o in REG {
            assert!(seen.insert(o.flag), "duplicate flag --{}", o.flag);
            assert!(!o.flag.is_empty() && o.flag.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'),
                    "bad flag name --{}", o.flag);
            assert!(!o.help.is_empty(), "--{} has no help", o.flag);
        }
        let mut envs = std::collections::HashSet::new();
        for o in REG.iter().filter(|o| !o.env.is_empty()) { assert!(envs.insert(o.env), "env {} mapped twice", o.env); }
        for (e, _) in REMOVED { assert!(!envs.contains(e), "{e} is both registered and removed"); }
    }

    /// CLI-1's standing proof (AGENTS §7): the engine source reads no environment variable of its own.
    /// Every `std::env` read / write in src/ must name a third-party variable (HOME, CARGO_MANIFEST_DIR)
    /// or a `#[test]`-only fixture path; native/*.c must not call getenv at all. The one reader of the
    /// whole environment is `opts::env_violations` (the startup refusal).
    #[test]
    fn no_engine_env_reads_remain() {
        const ALLOWED: [&str; 3] = ["HOME", "CARGO_MANIFEST_DIR", "GB10_TEST_MODEL_DIR"];
        const CALLS: [&str; 6] = ["env::var(", "env::var_os(", "env::set_var(", "env::remove_var(", "env::vars(", "env::vars_os("];
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut bad = Vec::new();
        let mut stack = vec![root.join("src")];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() { stack.push(p); continue; }
                if p.extension().map_or(true, |x| x != "rs") || p.ends_with("opts.rs") || p.starts_with(root.join("src/opts")) { continue; }
                let text = std::fs::read_to_string(&p).unwrap();
                for (i, line) in text.lines().enumerate() {
                    let code = line.split("//").next().unwrap_or("");
                    for c in CALLS {
                        let mut from = 0;
                        while let Some(k) = code[from..].find(c) {
                            let at = from + k + c.len();
                            let rest = code[at..].trim_start();
                            let lit = rest.strip_prefix('"').and_then(|r| r.split('"').next());
                            if !lit.map_or(false, |n| ALLOWED.contains(&n)) {
                                bad.push(format!("{}:{}: {}", p.display(), i + 1, line.trim()));
                            }
                            from = at;
                        }
                    }
                }
            }
        }
        for f in ["native/net_shim.c", "native/net_shim/01_helpers_proxy_nway.inc", "native/net_shim/02_init_accessors_bench.inc", "native/net_shim/03_hot_path.inc", "native/net_shim/04_startup_audit_channel.inc"] {
            let text = std::fs::read_to_string(root.join(f)).unwrap();
            for (i, line) in text.lines().enumerate() {
                if line.split("//").next().unwrap_or("").contains("getenv(") { bad.push(format!("{f}:{}: {}", i + 1, line.trim())); }
            }
        }
        assert!(bad.is_empty(), "environment reads outside the registry (AGENTS §7 / CLI-1):\n{}", bad.join("\n"));
    }

    /// rel/cutb (v0.7.3): --exit-on-fatal became a default-ON registered Bool (owner decision
    /// 2026-10-06). Pins the registry row (env slot empty, def "on") and the parse semantics the
    /// serve path relies on: bare flag = on, `on|1` = on, `off|0` = off; the reader
    /// `var(opt).map_or(true, |v| v != "0")` makes an unset value ON (the flipped default).
    #[test]
    fn exit_on_fatal_default_on() {
        let o = &REG[idx("exit-on-fatal")];
        assert_eq!(o.env, "");
        assert_eq!(o.def, "on");
        assert_eq!(canon(o, None).unwrap().as_deref(), Some("1")); // bare --exit-on-fatal
        assert_eq!(canon(o, Some("on")).unwrap().as_deref(), Some("1"));
        assert_eq!(canon(o, Some("1")).unwrap().as_deref(), Some("1"));
        assert_eq!(canon(o, Some("off")).unwrap().as_deref(), Some("0"));
        assert_eq!(canon(o, Some("0")).unwrap().as_deref(), Some("0"));
        assert!(canon(o, Some("bogus")).is_err());
        // the exact read exl3_serve.rs performs, through the real store + parser:
        let id = crate::opt!("exit-on-fatal");
        let read = || crate::opts::var(id).map_or(true, |v| v != "0");
        crate::opts::test_set(id, None);        assert!(read(),  "unset must mean ON (the flipped default)");
        crate::opts::test_set(id, Some("0"));   assert!(!read(), "--exit-on-fatal off must disable");
        crate::opts::test_set(id, Some("1"));   assert!(read(),  "--exit-on-fatal on must exit");
        crate::opts::test_set(id, None);        // leave the store as found
        // the shipped CLI surface end-to-end: bare flag and value form both parse
        let a = "--exit-on-fatal".to_string();
        assert!(parse_cli(&[a.clone()]).is_ok());
        assert!(read(), "bare --exit-on-fatal must mean ON");
        let off = "off".to_string();
        assert!(parse_cli(&[a, off]).is_ok());
        assert!(!read(), "--exit-on-fatal off must disable");
        crate::opts::test_set(id, None);        // restore
    }

    /// S-B9-REL-SURFACE2B R8: the value list `--help` shows and the values the help text explains
    /// agree. `--pq8-flash` and `--exl3-score-mr` explain `0` and `1`, so their lists show them;
    /// an option that explains only on / off keeps the short list.
    #[test]
    fn a_choice_hint_lists_the_values_its_text_explains() {
        assert_eq!(value_hint(&REG[idx("pq8-flash")]), " <0|off|1|on|causal>");
        assert_eq!(value_hint(&REG[idx("exl3-score-mr")]), " <0|off|1|on|generic>");
        assert_eq!(value_hint(&REG[idx("mtp")]), " <auto|on|off>", "the older spellings stay accepted and unlisted");
        assert_eq!(value_hint(&REG[idx("tool-quote-pairing")]), " <off|on|cost>");
        let mut choices = 0usize;
        for o in REG.iter() {
            let Ty::Choice(l) = o.ty else { continue };
            choices += 1;
            let hint = value_hint(o);
            let shown: Vec<&str> = hint.trim().trim_matches(|c| c == '<' || c == '>').split('|').collect();
            for v in l.iter().filter(|v| help_explains(o.help, v)) {
                assert!(shown.contains(v), "--{}: the help text explains '{v} = ...' but the list shows {hint}", o.flag);
            }
        }
        assert!(choices > 20, "{choices}");
    }

    /// The value of a Choice option is the set value or the row's default (`choice`): the
    /// defaults that are owner rulings (2026-10-09: `reasoning`, `off`) are stated by their rows
    /// and nowhere else.
    #[test]
    fn choice_reads_the_row_default() {
        for (flag, def) in [("unclosed-think", "reasoning"), ("tool-quote-pairing", "off"), ("tool-block-end", "after-function"), ("tool-value-trim", "one")] {
            let id = lookup(flag).unwrap();
            assert_eq!((id.def().def, id.def().env, id.def().scope), (def, "", Scope::Head), "--{flag}");
            if get(id).is_none() { assert_eq!(choice(id), def, "--{flag}"); }
            let Ty::Choice(l) = id.def().ty else { panic!("--{flag} is a Choice") };
            assert!(l.contains(&def) && l.len() >= 2, "--{flag}: {l:?}");
        }
    }

    /// S-B9-REL-BENCHFIX2 (W3): a ranged integer option is refused at startup outside its range
    /// (it used to pass `Ty::Int` and become the default silently in the reader), and what is
    /// stored is the canonical number.
    #[test]
    fn ranged_integer_options_refuse_at_startup() {
        let o = &REG[idx("sse-keepalive")];
        // the default is an owner ruling (2026-10-09: on, 10 s): this row's `def` is the one place
        // that states it, and this line and `the_keepalive_boot_line_and_its_default` test it
        assert_eq!((o.env, o.def, o.ty, o.scope), ("", "10", Ty::IntRange(0, 3600), Scope::Head));
        for bad in ["-1", "3601", "99999999999999999999", "18446744073709551615", "ten", "1.5", ""] {
            let e = canon(o, Some(bad)).unwrap_err();
            assert_eq!(e, format!("--sse-keepalive takes an integer from 0 to 3600 (got '{bad}')"));
        }
        assert_eq!(canon(o, None).unwrap_err(), "--sse-keepalive needs a value");
        for (given, stored) in [("0", "0"), ("-0", "0"), ("+5", "5"), (" 5 ", "5"), ("1", "1"), ("3600", "3600"), ("010", "10")] {
            assert_eq!(canon(o, Some(given)).unwrap().as_deref(), Some(stored), "given {given:?}");
        }
        // the whole command line, as `main` parses it: FATAL + exit 2 there on an Err
        let args = |v: &str| vec!["gb10_inference".to_string(), "--sse-keepalive".to_string(), v.to_string()];
        assert_eq!(parse_cli(&args("-1")).unwrap_err(), "--sse-keepalive takes an integer from 0 to 3600 (got '-1')");
        assert!(parse_cli(&["x".to_string(), "--sse-keepalive=86400".to_string()]).is_err());
        // the reader: the default is the row's (one source), and a value that reached the store
        // without `canon` is an Err, never the default
        let id = crate::opt!("sse-keepalive");
        assert_eq!(ranged_from(id, None), Ok(10));
        assert_eq!(ranged_from(id, Some("25")), Ok(25));
        assert_eq!(ranged_from(id, Some("-1")).unwrap_err(), "--sse-keepalive takes an integer from 0 to 3600 (value '-1')");
        assert!(ranged_from(id, Some("")).is_err() && ranged_from(id, Some("3601")).is_err());
        assert!(ranged_from(crate::opt!("mtp-reprobe"), None).is_err(), "not a ranged row");
        // the two sibling options with the same silent fallback (LR-4, LR-5): 0 = unlimited
        for (flag, def) in [("max-waiting", 256), ("stream-backlog-events", 65536)] {
            let o = &REG[idx(flag)];
            assert_eq!((o.env, o.ty), ("", Ty::IntRange(0, i64::MAX)));
            assert_eq!(canon(o, Some("-1")).unwrap_err(), format!("--{flag} takes an integer >= 0 (got '-1')"));
            assert!(canon(o, Some("many")).is_err());
            assert_eq!(canon(o, Some("0")).unwrap().as_deref(), Some("0"));
            assert_eq!(canon(o, Some("-0")).unwrap().as_deref(), Some("0"));
            let id = lookup(flag).unwrap();
            assert_eq!(ranged_from(id, None), Ok(def));
            assert_eq!(ranged_from(id, Some("7")), Ok(7));
            assert!(ranged_from(id, Some("-1")).is_err());
        }
        // every ranged row's default is a number inside its own range, or the row is off unless given
        // ("unset": the test drills; their readers take `get`, an Option, never `ranged`)
        for (i, o) in REG.iter().enumerate() {
            if let Ty::IntRange(lo, hi) = o.ty {
                assert!(lo <= hi, "--{}", o.flag);
                assert!(o.def == "unset" || ranged_from(OptId(i as u16), None).is_ok(),
                        "--{}: default '{}' outside {lo}..={hi}", o.flag, o.def);
                assert_eq!(value_hint(o), " <N>");
            }
        }
        // S-B9-CF-AGREE16B (W3): the two lockstep drills are validated at startup. `--tp-width-drill 0`
        // or a negative value used to pass the parse and arm nothing, in silence.
        let w = &REG[idx("tp-width-drill")];
        assert_eq!((w.env, w.def, w.ty, w.scope, w.cat), ("", "unset", Ty::IntRange(1, 31), Scope::Local, Cat::Diag));
        for bad in ["0", "-1", "32", "sixteen", "16.0", ""] {
            assert_eq!(canon(w, Some(bad)).unwrap_err(), format!("--tp-width-drill takes an integer from 1 to 31 (got '{bad}')"));
        }
        assert_eq!(canon(w, None).unwrap_err(), "--tp-width-drill needs a value");
        for (given, stored) in [("1", "1"), ("16", "16"), (" 16 ", "16"), ("+31", "31")] {
            assert_eq!(canon(w, Some(given)).unwrap().as_deref(), Some(stored));
        }
        assert!(parse_cli(&["x".to_string(), "--tp-width-drill".to_string(), "0".to_string()]).is_err());
        let a = &REG[idx("tp-agree-drill")];
        assert_eq!((a.def, a.ty, a.scope), ("unset", Ty::IntRange(0, i64::MAX), Scope::Local));
        assert_eq!(canon(a, Some("-1")).unwrap_err(), "--tp-agree-drill takes an integer >= 0 (got '-1')");
        assert_eq!(canon(a, Some("0")).unwrap().as_deref(), Some("0"));
    }

    /// S-B9-REL-SMALLFIXES item 3: what `--help` says. The generated sections come from the registry
    /// (`help_text`); the hand-written part is `print_help` in src/main.rs (a binary: its text is
    /// read from the source here, as the env-read test above reads the tree).
    #[test]

    /// CLIPRUNE1: `--help` is the day-to-day launch set ONLY; `--help-advanced` is everything.
    /// Drift in either fails against the committed snapshot fixtures; refresh (after a DELIBERATE
    /// content change only) with `cargo test -- --bless`.
    /// Owner 2026-10-23: the SSD spill tier is OFF unless asked for, and the flag is in the day-to-day `--help`. The
    /// server's fallback reads this registry default (exl3_serve), so the two cannot disagree.
    #[test]
    fn kv_spill_is_off_by_default_and_in_the_day_help() {
        let o = crate::opt!("kv-spill").def();
        assert_eq!((o.def, o.vis), ("off", Vis::Day));
        assert!(help_day().contains("--kv-spill"), "--kv-spill must be listed by the normal --help");
    }

    #[test]
    fn help_day_snapshot() { check_help_snapshot("help_day.txt", &help_day()); }

    #[test]
    fn help_advanced_snapshot() { check_help_snapshot("help_advanced.txt", &help_advanced()); }

    fn check_help_snapshot(name: &str, actual: &str) {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let want = std::fs::read_to_string(dir.join(name))
            .unwrap_or_else(|_| panic!("{name} fixture missing; create it with: cargo test --lib -- --ignored bless_help_snapshots"));
        assert_eq!(want, actual, "{name} drifted; if the change is intended refresh with: cargo test --lib -- --ignored bless_help_snapshots");
    }

    /// Refresh the snapshot fixtures. Ignored by default (a normal run only compares):
    /// cargo test --lib -- --ignored bless_help_snapshots
    #[test]
    #[ignore = "explicit refresh only: cargo test --lib -- --ignored bless_help_snapshots"]
    fn bless_help_snapshots() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("help_day.txt"), help_day()).unwrap();
        std::fs::write(dir.join("help_advanced.txt"), help_advanced()).unwrap();
    }

    /// The advanced view must list EVERY accepted option: every non-Internal registry row and
    /// every hand-read flag (registry rows via their help line; hand flags via the mode section).
    fn lists_flag(view: &str, flag: &str) -> bool {
        let pre = format!("    --{flag} ");
        let bare = format!("    --{flag}");
        view.lines().any(|l| l.starts_with(&pre) || l.trim_end() == bare)
    }

    #[test]
    fn help_advanced_covers_every_accepted_option() {
        let h = help_advanced();
        for o in REG.iter() {
            if !matches!(o.ty, Ty::Internal(_)) {
                assert!(lists_flag(&h, &o.flag), "--{} missing from --help-advanced", o.flag);
            }
        }
        for (f, _, _) in HAND_FLAGS {
            assert!(lists_flag(&h, f), "--{f} missing from --help-advanced");
        }
    }

    /// `--help` stays day-to-day only: no advanced registry row, no advanced hand flag leaks in.
    #[test]
    fn help_day_stays_day_to_day() {
        let h = help_day();
        for o in REG.iter() {
            if o.vis == Vis::Adv && !matches!(o.ty, Ty::Internal(_)) {
                assert!(!lists_flag(&h, &o.flag), "--{} (advanced) leaked into --help", o.flag);
            }
        }
        for (f, _, v) in HAND_FLAGS {
            if *v == Vis::Adv {
                assert!(!lists_flag(&h, f), "--{f} (advanced hand flag) leaked into --help");
            }
        }
    }

    /// N1 (FIX1): reworked onto the post-CLIPRUNE1 help API — `--help` IS `help_day()`, so
    /// "listed in --help" is asserted against it; `help_text(false)` is retired (debug_assert).
    /// The test had lost its #[test] attribute while it still called the retired fn (the ISSUES
    /// CLIPRUNE1 asterisk) — it runs again from this change.
    /// MIGRATION NOTE (N2/N3): when OUT1's boot_report uniform-stage API merges (branch feat/out1,
    /// unmerged at FIX1 time), the boot-line assertions move to it; --help stays the flag source.
    #[test]
    fn help_says_what_the_servers_do() {
        let (day, diag) = (help_day(), help_text(true));
        let adv = help_advanced();
        // every one of these is a serving option with its default shown. CLIPRUNE1 split --help
        // into tiers: Vis::Day rows are the day-to-day launch set (--help); every accepted row is
        // in --help-advanced; a serving row never lives only in --help-diag.
        for (flag, def) in [("qsa-row-regime", "[on]"), ("prefill-interleave", "[1]"), ("mtp-lanes", "[off]"),
                            ("prefill-sched", "[cursor]"), ("mtp", "[auto]"), ("lane-order", "[rr]"), ("lane-pack", "[on]")] {
            let o = REG.iter().find(|o| o.flag == flag).unwrap_or_else(|| panic!("--{flag} is not registered"));
            assert_ne!(o.cat, Cat::Diag, "--{flag} is a serving option");
            assert_eq!(o.env, "", "--{flag}: no environment variable");
            let line = help_line(o);
            assert!(line.trim_start().starts_with(&format!("--{flag} ")), "{line}");
            assert!(line.ends_with(def), "--{flag} must show its default {def}: ...{}", &line[line.len().saturating_sub(40)..]);
            if o.vis == Vis::Day {
                assert!(day.contains(&line), "--{flag} missing from --help");
            }
            assert!(adv.contains(&line), "--{flag} missing from --help-advanced");
            assert!(!diag.contains(&line), "--{flag} must not be a --help-diag option");
        }
        // the two flags main.rs parsed but --help never listed: both ride to the TP nodes
        for flag in ["mtp-lanes", "prefill-sched"] {
            let o = &REG[idx(flag)];
            assert_eq!(o.scope, Scope::Spmd, "--{flag}");
            assert!(o.help.starts_with("NVFP4 server"), "--{flag} names the server that reads it: {}", o.help);
            assert!(canon(o, None).is_err(), "--{flag} needs a value (main.rs reads the next argument)");
            assert!(canon(o, Some("bogus")).is_err());
        }
        assert_eq!(canon(&REG[idx("mtp-lanes")], Some("on")).unwrap().as_deref(), Some("on"));
        assert_eq!(canon(&REG[idx("prefill-sched")], Some("inline")).unwrap().as_deref(), Some("inline"));
        // --lane-order: the measured numbers (PLAN/CF_FCFS_REPORT.md section 4), not the design arithmetic
        let lo = REG[idx("lane-order")].help;
        assert!(lo.contains("23% lower with 2") && lo.contains("30% lower with 3"), "{lo}");
        assert!(!lo.contains("about 25%") && !lo.contains("about 33%"), "{lo}");
        // --prefill-interleave: no claim beyond what was measured (N = 1 byte identity at TP=2/TP=4, PERFFIX1)
        let il = REG[idx("prefill-interleave")].help;
        assert!(!il.contains("is removed by the per-row attention regime"), "{il}");
        assert!(il.contains("measured byte-identical to N = 0 (greedy, TP=2 and TP=4"), "{il}");
        // --mtp: says that off is honoured and names the banner
        assert!(REG[idx("mtp")].help.contains("MTP OFF (--mtp off)"));

        // the hand-written part
        let main = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/cli/help.rs")).unwrap();
        let a = main.find("fn print_help() {").expect("print_help");
        let body = &main[a..a + main[a..].find("\n}\n").expect("end of print_help")];
        // --max-tokens: a default for requests that omit it, per server; never "[8192]" for every server
        assert!(!body.contains("Default generation cap when a request omits max_tokens [8192]"));
        assert!(body.contains("--max-tokens <N>           Output length for a CHAT request that omits max_tokens. A default,"));
        assert!(body.contains("not a cap") && body.contains("[EXL3: unlimited; NVFP4: 8192]"));
        // --default-presence-penalty: the EXL3 server's default is 0.0
        assert!(!body.contains("Presence penalty       [2.0 on 2B, else 1.5]"));
        assert!(body.contains("[EXL3: 0.0; NVFP4: 2.0 on Qwen3.5 0.8B/2B, else 1.5]"));
        // unsent sampler values: the thinking-mode values of the EXL3 chat path are named
        assert!(!body.contains("defaults 0.7 / 0.8 / 20)"));
        // FLASHNEXT-DF2-B (ISSUES work[113]): the unsent values come from the pack's
        // generation_config.json; the help names the source, the Qwen3.8 triples and the fallback.
        assert!(body.contains("Unsent: the pack's") && body.contains("generation_config.json triple")
                && body.contains("1.0 / 0.95 / 20") && body.contains("0.7 / 0.8 / 20"));
        // --mtp: off is honoured by every server; the option is described ONCE, by its registry line
        // (SURFACE2 item 5: the hand-written block repeated it, and the two texts had drifted apart)
        assert!(!body.contains("on/off force it (benchmarking)."));
        let mtp = help_line(&REG[idx("mtp")]);
        assert!(mtp.trim_start().starts_with("--mtp <auto|on|off> "), "a three-value choice in --help: {}", &mtp[..60]);
        assert!(mtp.contains("--exl3-no-mtp") && mtp.contains("1 / true = on"), "the older spellings are named");
        for v in ["1", "0", "true", "false"] { assert!(canon(&REG[idx("mtp")], Some(v)).is_ok(), "--mtp {v} is still accepted"); }
        assert!(REG[idx("exl3-no-mtp")].help.contains("older, EXL3-only spelling of --mtp off"));
        // no double-encoded characters in the help (they printed as garbage: U+00C2..U+00F4 + U+0080..U+00BF)
        assert_eq!(body.lines().filter(|l| double_encoded(l)).count(), 0, "print_help has double-encoded (mojibake) lines");
    }

    /// UTF-8 text that was decoded as Latin-1 and encoded again: a lead byte U+00C2..U+00F4
    /// followed by a continuation byte U+0080..U+00BF, both as characters (an em dash reads as
    /// three junk characters).
    fn double_encoded(l: &str) -> bool {
        let c: Vec<char> = l.chars().collect();
        c.windows(2).any(|w| ('\u{c2}'..='\u{f4}').contains(&w[0]) && ('\u{80}'..='\u{bf}').contains(&w[1]))
    }

    /// SURFACE2 item 7: no double-encoded line anywhere in the sources (48 lines of src/main.rs
    /// outside --help held them: 31 comments and 17 printed strings, RESULT lines and boot lines
    /// among them). Walks src/, kernels/, scripts/ and the build files.
    #[test]
    fn no_double_encoded_text_in_the_sources() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files: Vec<std::path::PathBuf> = vec![root.join("build.rs"), root.join("Cargo.toml")];
        let mut dirs = vec![root.join("src"), root.join("kernels"), root.join("scripts")];
        while let Some(d) = dirs.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() { dirs.push(p); continue; }
                if matches!(p.extension().and_then(|x| x.to_str()), Some("rs" | "cu" | "h" | "sh" | "py")) { files.push(p); }
            }
        }
        assert!(files.len() > 60, "the walk found the tree ({} files)", files.len());
        // Lines that are ALLOWED to match. The scan flags a character in U+00C2..U+00F4 followed
        // by one in U+0080..U+00BF, which is what double-encoded text looks like, but a line may
        // hold such a pair on purpose (two real Latin-1 characters side by side, a test fixture
        // of mojibake). List it here as (path from the repository root, the line trimmed) and it
        // passes; an entry that no longer matches a line fails, so the list cannot rot. Empty:
        // no source line needs it today.
        const ALLOWED: &[(&str, &str)] = &[];
        let mut used = vec![false; ALLOWED.len()];
        let mut hits: Vec<String> = Vec::new();
        for f in &files {
            let Ok(text) = std::fs::read_to_string(f) else { continue }; // not UTF-8: not this defect
            let rel = f.strip_prefix(root).unwrap_or(f).display().to_string();
            for (i, l) in text.lines().enumerate() {
                if !double_encoded(l) { continue; }
                match ALLOWED.iter().position(|(p, line)| *p == rel && *line == l.trim()) {
                    Some(k) => used[k] = true,
                    None => hits.push(format!("{rel}:{}", i + 1)),
                }
            }
        }
        assert!(hits.is_empty(), "{} double-encoded line(s), first: {:?}", hits.len(), &hits[..hits.len().min(8)]);
        let stale: Vec<&(&str, &str)> = ALLOWED.iter().zip(&used).filter(|(_, u)| !**u).map(|(a, _)| a).collect();
        assert!(stale.is_empty(), "allow-list entries that match no line any more: {stale:?}");
        // the scan itself: mojibake is flagged, the same characters in proper text are not
        assert!(double_encoded("caf\u{c3}\u{a9}") && !double_encoded("caf\u{e9} \u{2014} na\u{ef}ve"));
    }

    /// The option entries of the hand-written part of `--help`: a line that starts, after 2 to 8
    /// spaces, with `--flag` (print_help writes them as `the exact 4-space-indented --flag help format` or inside a
    /// multi-line string whose lines begin with `\x20`).
    fn hand_written_entries(body: &str) -> std::collections::BTreeSet<String> {
        let mut out = std::collections::BTreeSet::new();
        for line in body.lines() {
            let l = line.trim_start();
            let l = l.strip_prefix("print").and_then(|r| r.strip_prefix("ln!(\u{22}")).unwrap_or(l);
            let (l, lead) = match l.strip_prefix("\\x20") { Some(r) => (r, 1), None => (l, 0) };
            let spaces = lead + l.len() - l.trim_start_matches(' ').len();
            let Some(rest) = l.trim_start_matches(' ').strip_prefix("--") else { continue };
            if !(2..=8).contains(&spaces) { continue; }
            let flag: String = rest.chars().take_while(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '-').collect();
            if !flag.is_empty() { out.insert(flag); }
        }
        out
    }

    /// SURFACE2 item 5: an option is described ONCE in `--help`. `--mtp` and `--prefill-interleave`
    /// were listed by hand AND by their registry line, and the two `--mtp` texts had drifted apart
    /// (one cited a version the code does not have). They are registry-only now. The older options
    /// still listed twice are pinned here, by name: the set may shrink (delete the hand-written
    /// entry, then its name below), it must never grow.
    #[test]
    fn help_lists_an_option_once() {
        const STILL_TWICE: [&str; 22] = [
            "cpu-sample", "draft-vocab", "exact-gemm", "fp8-prefill", "fuse-residual", "hc-fp16", "max-seq-len", "moe-variant",
            "mxfp4", "no-decode-graphs", "no-gqpack", "ple-offload", "prefill-scalar", "rdma-dev", "splitk-gemm", "tp-cache",
            "tp-fp32-partials", "tp-graph", "tp-prefill-overlap", "tp-seq-parallel", "tp-shard-mtp", "tp-vp-sampled",
        ];
        let main = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/cli/help.rs")).unwrap();
        let a = main.find("fn print_help() {").expect("print_help");
        let body = &main[a..a + main[a..].find("\n}\n").expect("end of print_help")];
        let hand = hand_written_entries(body);
        assert!(hand.len() > 80 && hand.contains("max-seq-len") && hand.contains("max-tokens"), "the parse found the entries ({})", hand.len());
        let twice: Vec<&str> = REG.iter().filter(|o| o.cat != Cat::Diag && !matches!(o.ty, Ty::Internal(_)))
            .map(|o| o.flag).filter(|f| hand.contains(*f)).collect();
        let mut want: Vec<&str> = STILL_TWICE.to_vec();
        let mut got = twice.clone();
        want.sort();
        got.sort();
        assert_eq!(got, want, "options listed by hand and by the registry");
        for (flag, day) in [("mtp", true), ("prefill-interleave", false)] {
            assert!(!hand.contains(flag), "--{flag} is described by its registry line only");
            let line = help_line(&REG[idx(flag)]);
            let view = if day { help_day() } else { help_advanced() };
            assert_eq!(view.matches(line.as_str()).count(), 1, "--{flag}: one generated line");
        }
        // no help or registry text cites a release the crate is not (the cut bumps the version, not the text)
        let ver = env!("CARGO_PKG_VERSION");
        let parts: Vec<u32> = ver.split('.').map(|x| x.parse().unwrap()).collect();
        let future = |text: &str| -> Option<String> {
            let b = text.as_bytes();
            (0..b.len()).filter(|&i| b[i] == b'v' && (i == 0 || !b[i - 1].is_ascii_alphanumeric())).find_map(|i| {
                let v: String = text[i + 1..].chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
                let v = v.trim_end_matches('.');
                let n: Vec<u32> = v.split('.').filter_map(|x| x.parse().ok()).collect();
                (n.len() == 3 && n > parts).then(|| format!("v{v}"))
            })
        };
        for o in REG.iter() {
            assert_eq!(future(o.help), None, "--{} cites a release newer than {ver}", o.flag);
        }
        assert_eq!(future(body), None, "print_help cites a release newer than {ver}");
        assert_eq!(future("since v0.7.3 and v99.0.1"), Some("v99.0.1".to_string()), "the check can fail");
    }

    #[test]
    fn canon_rules() {
        let f = &REG[idx("exl3-round-prof")];
        assert_eq!(canon(f, None).unwrap().as_deref(), Some("1"));
        assert_eq!(canon(f, Some("off")).unwrap(), None);
        assert!(canon(f, Some("bogus")).is_err());
        let b = &REG[idx("exl3-head-fill")];
        assert_eq!(canon(b, Some("off")).unwrap().as_deref(), Some("0"));
        assert_eq!(canon(b, Some("on")).unwrap().as_deref(), Some("1"));
        let i = &REG[idx("load-workers")];
        assert!(canon(i, Some("x")).is_err());
        assert_eq!(canon(i, Some("12")).unwrap().as_deref(), Some("12"));
    }

    /// V5: a raw stored value's truth does not depend on its spelling.
    #[test]
    fn on_reads_every_spelling_of_off_as_off() {
        for off in ["", " ", "0", "off", "OFF", "Off", "false", "False", "FALSE", "no", "No", "NO", " no "] {
            assert!(!is_on_value(off), "{off:?} must read as off");
        }
        for on in ["1", "on", "On", "true", "True", "yes", "2", "ssd", "attn,mlp"] {
            assert!(is_on_value(on), "{on:?} must read as on");
        }
    }

    /// V5: a Choice option whose list carries `no` stores it raw (`install_head` copies the head's stored
    /// values as they are); that value must read as off. (`install_head` itself is not called here: it
    /// rewrites the whole process-global store.)
    #[test]
    fn a_value_installed_raw_as_no_reads_off() {
        let id = lookup("mtp-lanes").expect("registered");
        assert!(matches!(id.def().ty, Ty::Choice(l) if l.contains(&"no")), "the option's list carries `no`");
        for raw in ["no", "False", "No"] {
            assert!(!is_on_value(raw));
        }
        // the pure parse of the same command line: `--mtp-lanes no` is accepted and stored as given
        let o = id.def();
        assert_eq!(canon(o, Some("no")).unwrap().as_deref(), Some("no"));
    }


    // ---- S-B9-REL-FINALFIX F7 (addendum 2026-10-11): the option-swallow fix ----

    /// The refusal sentence, one builder so the pinned text stays identical everywhere.
    fn f7_msg(flag: &str, next: &str) -> String {
        format!("--{flag} needs a value, but the next argument is {next}; to give a value that really starts with \"--\" write --{flag}={next}")
    }
    fn f7_last(flag: &str) -> String {
        format!("--{flag} needs a value, but it is the last argument; to give a value that starts with \"--\" write --{flag}=<value>")
    }

    /// Every Text/Path row (44 Text + 30 Path = 74; NIGHT1 added --probe-gpu-wait-mib; THINKBUDGET1 added --reasoning-config; WIDEVERIFY1 added the
    /// --ids-file Path row; COPYREC1 --copy-span-record-dir; LANEWIDE --probe-lane-wide-rows; PACKPF --packpf-lens): the four cases of the addendum.
    #[test]
    fn f7_text_path_rows_never_eat_a_flag_shaped_token() {
        let rows: Vec<&Opt> = REG.iter().filter(|o| matches!(o.ty, Ty::Text | Ty::Path)).collect();
        assert_eq!(rows.len(), 74, "44 Text + 30 Path (fp4l-mtp-sidecar is a Choice since 2026-11-01; WIDEVERIFY1 --ids-file; COPYREC1 --copy-span-record-dir; LANEWIDE --probe-lane-wide-rows; PACKPF --packpf-lens; NIGHT1 --probe-gpu-wait-mib)");
        let args = |v: &[&str]| -> Vec<String> { v.iter().map(|s| s.to_string()).collect() };
        for o in rows {
            let id = lookup(o.flag).unwrap();
            let f = format!("--{}", o.flag);
            let e = parse_cli(&args(&[&f, "--tp", "2"])).unwrap_err();
            assert_eq!(e, f7_msg(o.flag, "--tp"), "{f}");
            let e = parse_cli(&args(&[&f, "--mtp", "off"])).unwrap_err();
            assert_eq!(e, f7_msg(o.flag, "--mtp"), "{f}");
            assert_eq!(parse_cli(&args(&[&f])).unwrap_err(), f7_last(o.flag), "{f}");
            if o.flag == "draft-vocab" {
                // the one CHECKED Text row: a value outside off|on|N|P%|0.F is refused at boot
                // (a flag-shaped one included), a valid one is stored canonical
                let e = parse_cli(&args(&[&f, "-x"])).unwrap_err();
                assert!(e.starts_with("--draft-vocab takes"), "{e}");
                assert!(parse_cli(&args(&[&format!("{f}=--tp")])).is_err(), "{f}");
                assert!(parse_cli(&args(&[&f, "ON", "--tp", "2"])).is_ok(), "{f}");
                assert_eq!(crate::opts::var(id).as_deref(), Ok("on"), "{f}");
                crate::opts::test_set(id, None);
                continue;
            }
            assert!(parse_cli(&args(&[&f, "-x"])).is_ok(), "{f}");
            crate::opts::test_set(id, None);
            assert!(parse_cli(&args(&[&f, "v", "--tp", "2"])).is_ok(), "{f}");
            crate::opts::test_set(id, None);
            assert_eq!(parse_cli(&args(&[&f])).unwrap_err(), f7_last(o.flag), "{f}");
            assert!(parse_cli(&args(&[&format!("{f}=--tp")])).is_ok(), "{f}");
            assert_eq!(crate::opts::var(id).as_deref(), Ok("--tp"), "{f}");
            crate::opts::test_set(id, None);
        }
    }

    /// Value types already fail loudly on a flag-shaped value; switch types never eat one.
    #[test]
    fn f7_other_value_types_fail_or_never_eat_a_flag_shaped_token() {
        let args = |v: &[&str]| -> Vec<String> { v.iter().map(|s| s.to_string()).collect() };
        let iv = REG.iter().find(|o| o.ty == Ty::Int).expect("an Int row");
        let e = parse_cli(&args(&[&format!("--{}", iv.flag), "--tp"])).unwrap_err();
        assert!(e.starts_with(&format!("--{} takes", iv.flag)) && e.contains("'--tp'"), "{e}");
        let e = parse_cli(&args(&["--sse-keepalive", "--tp"])).unwrap_err();
        assert!(e.starts_with("--sse-keepalive takes an integer from") && e.contains("'--tp'"), "{e}");
        let fl = REG.iter().find(|o| matches!(o.ty, Ty::Float)).expect("a Float row");
        let e = parse_cli(&args(&[&format!("--{}", fl.flag), "--tp"])).unwrap_err();
        assert!(e.starts_with(&format!("--{} takes", fl.flag)) && e.contains("'--tp'"), "{e}");
        let e = parse_cli(&args(&["--mtp", "--tp"])).unwrap_err();
        assert!(e.starts_with("--mtp takes ") && e.contains("(got '--tp')"), "{e}");
        let id = crate::opt!("exit-on-fatal");
        assert!(parse_cli(&args(&["--exit-on-fatal", "--tp", "2"])).is_ok());
        assert_eq!(crate::opts::var(id).as_deref(), Ok("1"));
        crate::opts::test_set(id, None);
        for want in [Ty::IntFlag, Ty::Flag] {
            let o = REG.iter().find(|o| o.ty == want).expect("a row of the type");
            let oid = lookup(o.flag).unwrap();
            assert!(parse_cli(&args(&[&format!("--{}", o.flag), "--tp", "2"])).is_ok(), "{}", o.flag);
            crate::opts::test_set(oid, None);
            }
    }

    /// F7: the --help footer carries the same sentence as the refusal.
    #[test]
    fn f7_help_footer_says_the_escape() {
        for diag in [true] {
            assert!(help_text(diag).contains("must be written --option=value"), "diag={diag}");
        }
    }

    #[test]
    fn f7_differential_vs_frozen_old_parser() { f7_differential(25_000); }
    // ---- F9 (owner decision A, 2026-10-11): hand-read value flags refuse the swallow ----
    fn f9_args(v: &[&str]) -> Vec<String> { v.iter().map(|s| s.to_string()).collect() }
    // the frozen pre-F9 reader (F9 changes no read; the guard is the startup pass)
    fn frozen_read<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
        args.iter().position(|a| a == flag || a.starts_with(&format!("{flag}=")))
            .and_then(|i| {
                let arg = &args[i];
                if let Some(val) = arg.strip_prefix(&format!("{flag}=")) { Some(val) }
                else { args.get(i + 1).map(|s| s.as_str()) }
            })
    }
    #[test]
    fn f9_generated_hand_flag_rules() {
        for &(f, takes, _) in HAND_FLAGS {
            let registry_owned = lookup(f).is_some();
            let fflag = format!("--{f}");
            // (a) a swallowed flag refuses, naming BOTH flags
            let v = f9_args(&["--max-batch", "4", &fflag, "--probe-binv"]);
            let r = refuse_hand_flag_swallows(&v);
            let optional = OPTIONAL_VALUE_FLAGS.iter().any(|(vf, _)| *vf == f);
            if takes && !optional && !registry_owned {
                let m = r.expect(f);
                let w1 = format!("--{f} needs a value, but the next argument is --probe-binv");
                let w2 = format!("write --{f}=--probe-binv");
                assert!(m.contains(&w1), "{f}: {m}");
                assert!(m.contains(&w2), "{f}: {m}");
            } else {
                assert!(r.is_none(), "{f}: unexpected refusal");
            }
            // (b) the = form always works, even a value starting with --
            let v = f9_args(&[&format!("--{f}=--value"), "x"]);
            assert!(refuse_hand_flag_swallows(&v).is_none());
            if !registry_owned {
                assert_eq!(crate::arg_value(&v, &fflag), Some("--value"));
            }
            // (c) values that start with ONE dash or merely contain -- are values
            if !registry_owned {
                for val in ["-1", "a--b", "0.0.0.0"] {
                    let v = f9_args(&[&fflag, val]);
                    assert!(refuse_hand_flag_swallows(&v).is_none());
                    assert_eq!(crate::arg_value(&v, &fflag), Some(val));
                }
            }
            // (d) last on the line refuses (except optional --tp; registry rows own themselves)
            let v = f9_args(&["--probe-binv", &fflag]);
            let r = refuse_hand_flag_swallows(&v);
            if takes && !optional && !registry_owned {
                let m = r.expect(f);
                let w1 = format!("--{f} needs a value, but it is the last argument");
                assert!(m.contains(&w1), "{f}: {m}");
            } else {
                assert!(r.is_none(), "{f}: unexpected refusal");
            }
        }
    }
    #[test]
    fn f9_tp_optional_value_keeps_working() {
        assert!(refuse_hand_flag_swallows(&f9_args(&["--tp"])).is_none());
        assert!(refuse_hand_flag_swallows(&f9_args(&["--tp", "4"])).is_none());
        assert!(refuse_hand_flag_swallows(&f9_args(&["--tp=4"])).is_none());
        // F9b: bare --tp before --nodes is the DOCUMENTED form (README, cards, 36 scripts)
        assert!(refuse_hand_flag_swallows(&f9_args(&["--tp", "--nodes", "a,b"])).is_none());
        assert!(tp_bare_hint(&f9_args(&["--tp", "--nodes", "a,b"])).is_some());
        assert!(tp_bare_hint(&f9_args(&["--tp"])).is_some());
        assert!(tp_bare_hint(&f9_args(&["--tp", "4"])).is_none());
        assert!(tp_bare_hint(&f9_args(&["--tp=4"])).is_none());
        assert!(tp_bare_hint(&f9_args(&["--server"])).is_none());
    }
    #[test]
    fn f9_real_lines_parse_as_before() {
        let node = f9_args(&["--node", "--host", "0.0.0.0", "--port", "29500"]);
        assert!(refuse_hand_flag_swallows(&node).is_none());
        let owner = f9_args(&["--server", "--host", "0.0.0.0", "--port", "9000", "--model-dir", "/m",
            "--tp", "4", "--nodes", "a:1,b:2,c:3", "--max-seq-len", "262144",
            "--max-batch", "4", "--max-tokens", "65536", "--prefix-cache", "on", "--mtp=auto"]);
        assert!(refuse_hand_flag_swallows(&owner).is_none());
        assert!(refuse_hand_flag_swallows(&f9_args(&["--zzz-unknown", "--port", "9000"])).is_none());
        assert!(refuse_hand_flag_swallows(&f9_args(&["--host", "--max-batch", "4"])).is_none());
    }
    #[test]
    fn f9b_help_optionals_cover_every_help_entry() {
        for diag in [true] {
            for line in help_text(diag).lines() {
                let t = line.trim_start();
                let Some(body) = t.strip_prefix("--") else { continue };
                let Some((name, rest)) = body.split_once(" ") else { continue };
                if !rest.starts_with("[") { continue }
                assert!(
                    OPTIONAL_VALUE_FLAGS.iter().any(|(vf, _)| *vf == name) || lookup(name).is_some(),
                    "help shows --{name} with an optional [x] hint but it is not on OPTIONAL_VALUE_FLAGS");
            }
        }
    }
    #[test]
    fn f9b_named_lines_parse() {
        // the public README zero-configuration line and the three model-card lines (a real peer IP
        // stands in for the placeholder): bare --tp before --nodes is the documented form
        let readme = f9_args(&["--server", "--tp", "--nodes", "10.0.0.2:29500", "--port", "9000"]);
        assert!(refuse_hand_flag_swallows(&readme).is_none());
        for _ in 0..3 {
            let card = f9_args(&["--tp", "--nodes", "10.0.0.2:29500", "--port", "9000"]);
            assert!(refuse_hand_flag_swallows(&card).is_none());
        }
        assert!(refuse_hand_flag_swallows(&f9_args(&["--output-prompts", "--max-batch", "4"])).is_none());
        assert!(refuse_hand_flag_swallows(&f9_args(&["--splitk-gemm", "--x"])).is_none());
        assert!(refuse_hand_flag_swallows(&f9_args(&["--head", "--model-dir", "/m"])).is_none());
        // the refusal itself stays for every other value-taking hand flag
        let m = refuse_hand_flag_swallows(&f9_args(&["--max-batch", "--probe-binv"])).expect("refuses");
        assert!(m.contains("--max-batch needs a value, but the next argument is --probe-binv"));
    }

    fn f9_differential(n: u64) {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let mut pool = Vec::new();
        for &(f, _, _) in HAND_FLAGS { pool.push(format!("--{f}")); }
        for v in ["4", "29500", "on", "auto", "-1", "a--b", "0.0.0.0", "262144", "--",
                  "--max-batch", "--tp", "--probe-binv", "--unknown-flag", "--nodes=a,b"] {
            pool.push(v.to_string());
        }
        for f in ["model-dir", "port", "nodes", "max-batch", "tp"] {
            pool.push(format!("--{f}=--x"));
            pool.push(format!("--{f}=4"));
        }
        let mut value_taking = Vec::new();
        for &(f, t, _) in HAND_FLAGS {
            if t && lookup(f).is_none() && !OPTIONAL_VALUE_FLAGS.iter().any(|(vf, _)| *vf == f) {
                value_taking.push(format!("--{f}"));
            }
        }
        // the intended-refusal predicate: a value-taking hand flag (not a registry row, and not on
        // the OPTIONAL_VALUE_FLAGS list: --tp, --head, --probe-tq, ...) met a --token next, or sat
        // last on the line
        let intended = |v: &[String]| {
            for j in 0..v.len() {
                let Some(body) = v[j].strip_prefix("--") else { continue };
                if body.is_empty() { continue; }
                let (name, inline) = match body.split_once('=') {
                    Some((a, b)) => (a, Some(b)),
                    None => (body, None),
                };
                if inline.is_some() || lookup(name).is_some() { continue; }
                let mut hit = false;
                for &(hf, takes, _) in HAND_FLAGS {
                    if hf == name && !OPTIONAL_VALUE_FLAGS.iter().any(|(vf, _)| *vf == hf) {
                        if takes && v.get(j + 1).is_some_and(|x| x.starts_with("--")) { hit = true; }
                        if takes && j + 1 == v.len() { hit = true; }
                    }
                }
                if hit { return true; }
            }
            false
        };
        let mut identical = 0u64;
        let mut refused = 0u64;
        for _ in 0..n {
            let len = 1 + (next() % 8) as usize;
            let mut v = Vec::new();
            for _ in 0..len { v.push(pool[(next() % pool.len() as u64) as usize].clone()); }
            let r = refuse_hand_flag_swallows(&v);
            if let Some(m) = r {
                assert!(intended(&v), "unintended refusal {m} on {v:?}");
                refused += 1;
            } else {
                assert!(!intended(&v), "missed intended refusal on {v:?}");
                for f in &value_taking {
                    assert_eq!(frozen_read(&v, f), crate::arg_value(&v, f));
                }
                identical += 1;
            }
        }
        assert!(refused > n / 20, "the pool must exercise the refusal");
        eprintln!("f9_differential: {n} vectors: identical {identical}, intended refusals {refused}");
    }
    #[test] fn f9_differential_fast() { f9_differential(25_000); }
    #[test] #[ignore] fn f9_differential_long() { f9_differential(1_000_000); }

    #[test]
    #[ignore]
    fn f7_differential_long() { f7_differential(1_000_000); }

    /// F7 differential vs a FROZEN COPY of the pre-F7 walk: identical everywhere except a
    /// Text/Path row whose next token starts with "--" (now refused) and the last-argument
    /// sentence. lookup/canon are shared; F7 touches neither.
    fn f7_differential(n: u64) {
        fn old_bool_word(v: &str) -> Option<bool> {
            match v.trim().to_ascii_lowercase().as_str() {
                "1" | "on" | "true" | "yes" => Some(true),
                "0" | "off" | "false" | "no" => Some(false),
                _ => None,
            }
        }
        fn old_takes_next(ty: Ty, next: Option<&String>) -> bool {
            let Some(n) = next else { return false };
            match ty {
                Ty::Flag | Ty::Bool => old_bool_word(n).is_some(),
                Ty::IntFlag => old_bool_word(n).is_some() || n.trim().parse::<i64>().is_ok(),
                _ => true,
            }
        }
        fn old_parse_one(args: &[String], i: usize) -> Result<Option<(OptId, Option<String>, usize)>, String> {
            let Some(body) = args[i].strip_prefix("--") else { return Ok(None) };
            let (name, inline) = match body.split_once('=') { Some((n, v)) => (n, Some(v)), None => (body, None) };
            let Some(id) = lookup(name) else { return Ok(None) };
            let o = id.def();
            if matches!(o.ty, Ty::Internal(_)) { return Ok(None); }
            let mut next = i + 1;
            let v: Option<&str> = match inline {
                Some(v) => Some(v),
                None if old_takes_next(o.ty, args.get(next)) => { next += 1; Some(args[next - 1].as_str()) }
                None => None,
            };
            Ok(Some((id, canon(o, v)?, next)))
        }
        fn old_walk(args: &[String]) -> Result<Vec<(usize, Option<String>)>, String> {
            let mut out = Vec::new();
            let mut i = 0;
            while i < args.len() {
                match old_parse_one(args, i)? {
                    Some((id, c, next)) => { out.push((id.0 as usize, c)); i = next; }
                    None => i += 1,
                }
            }
            Ok(out)
        }
        let mut pool: Vec<String> = Vec::new();
        for o in REG.iter() {
            if matches!(o.ty, Ty::Internal(_)) { continue; }
            pool.push(format!("--{}", o.flag));
        }
        pool.push("--unknown-flag".into());
        for extra in ["--tp=2", "--rdma-dev=--tp", "--spec-pass=off", "on", "off", "1", "0",
                      "2", "262144", "-1", "3.5", "v", "-x", "--tp", "--", "auto", "x=y"] {
            pool.push(extra.into());
        }
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; s.wrapping_mul(0x2545_F491_4F6C_DD1D) };
        let mut identical = 0u64;
        let mut refused = 0u64;
        for _ in 0..n {
            let len = 1 + (next() % 8) as usize;
            let args: Vec<String> = (0..len).map(|_| pool[(next() % pool.len() as u64) as usize].clone()).collect();
            let old = old_walk(&args);
            let new = parse_args(&args);
            match (old, new) {
                (Ok(a), Ok(b)) => {
                    let b: Vec<(usize, Option<String>)> = b.into_iter().map(|(i, v)| (i.0 as usize, v)).collect();
                    assert_eq!(a, b, "parsed values differ: {args:?}");
                    identical += 1;
                }
                (Err(e1), Err(e2)) => {
                    if e1 == e2 { identical += 1; }
                    else {
                        assert!(e2.contains("needs a value, but the next argument is")
                                || e2.contains("it is the last argument"),
                            "an error F7 did not touch changed: {args:?} old={e1} new={e2}");
                        refused += 1;
                    }
                }
                (Ok(_), Err(e)) => {
                    assert!(e.contains("needs a value, but the next argument is"), "{args:?} -> {e}");
                    refused += 1;
                }
                (Err(e), Ok(_)) => panic!("F7 made a refusing line parse: {args:?} old={e}"),
            }
        }
        assert!(refused > 0, "the pool must exercise the new refusal");
        eprintln!("f7_differential: {n} vectors: identical {identical}, differing {refused}");
    }


    /// WIDEVERIFY1: `--copy-span on|off` is a registry flag (no environment name), off by default,
    /// rides the SPMD option snapshot like its two knobs, and refuses a misspelt value at startup.
    #[test]
    fn copy_span_flag_parses_on_off_and_defaults_off() {
        let o = &REG[idx("copy-span")];
        // V0710: "auto" = resolved per family at the EXL3 boot (src/family_defaults.rs); unset still reads off
        assert_eq!((o.env, o.def, o.ty, o.scope, o.cat), ("", "auto", Ty::Bool, Scope::Spmd, Cat::Spec));
        for (given, stored) in [("on", "1"), ("off", "0"), ("1", "1"), ("0", "0"), ("true", "1"), ("no", "0")] {
            assert_eq!(canon(o, Some(given)).unwrap().as_deref(), Some(stored), "given {given:?}");
        }
        assert_eq!(canon(o, None).unwrap().as_deref(), Some("1"), "a bare --copy-span is on");
        assert_eq!(canon(o, Some("maybe")).unwrap_err(), "--copy-span takes on|off (got 'maybe')");
        let a = |v: &[&str]| -> Vec<String> { v.iter().map(|s| s.to_string()).collect() };
        let got = parse_args(&a(&["x", "--copy-span", "on", "--copy-span-min-n", "8"])).unwrap();
        assert!(got.contains(&(crate::opt!("copy-span"), Some("1".to_string()))), "{got:?}");
        assert!(got.contains(&(crate::opt!("copy-span-min-n"), Some("8".to_string()))), "{got:?}");
        let got = parse_args(&a(&["x", "--copy-span=off"])).unwrap();
        assert!(got.contains(&(crate::opt!("copy-span"), Some("0".to_string()))), "{got:?}");
        assert!(parse_cli(&a(&["x", "--copy-span=maybe"])).is_err());
        // the stored "0" reads as off
        assert!(!is_on_value("0") && is_on_value("1"));
    }
