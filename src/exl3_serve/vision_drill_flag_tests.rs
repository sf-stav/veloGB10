//! Unit tests (module `vision_drill_flag_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::*;

    /// The registry store is process-global: one test at a time writes the drill's row.
    static STORE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn argv(tokens: &[&str]) -> Vec<String> { tokens.iter().map(|s| s.to_string()).collect() }

    /// Parse `tokens` as the process would and ask the serve path's question.
    fn armed_after_parsing(tokens: &[&str]) -> bool {
        let drill = crate::opt!("vision-fail-gemm");
        // hermetic: the command line may name no OTHER registered option (parse_cli would write
        // its row into the store every other test shares)
        for t in tokens.iter().filter_map(|t| t.strip_prefix("--")) {
            let name = t.split('=').next().unwrap();
            assert!(name == "vision-fail-gemm" || crate::opts::lookup(name).is_none(), "--{name} is a registry option: keep it out of this test");
        }
        crate::opts::test_set(drill, None);
        crate::opts::parse_cli(&argv(tokens)).expect("parse_cli");
        let armed = vision_fail_gemm_armed();
        crate::opts::test_set(drill, None);
        armed
    }

    #[test]
    fn the_drill_arms_wherever_the_flag_stands() {
        let _g = STORE.lock().unwrap_or_else(|e| e.into_inner());
        // LAST argument: the case that was silently not armed
        assert!(armed_after_parsing(&["--server", "--model-dir", "/m", "--vision-fail-gemm"]), "the flag as the last argument");
        // in the middle, and first
        assert!(armed_after_parsing(&["--server", "--vision-fail-gemm", "--model-dir", "/m"]), "the flag in the middle");
        assert!(armed_after_parsing(&["--vision-fail-gemm", "--server", "--model-dir", "/m"]), "the flag first");
        // alone
        assert!(armed_after_parsing(&["--vision-fail-gemm"]));
        // absent: not armed (the default of a server)
        assert!(!armed_after_parsing(&["--server", "--model-dir", "/m"]), "absent");
        assert!(!armed_after_parsing(&[]));
        // an explicit off is off (the old reader ARMED the drill on `--vision-fail-gemm=0`)
        assert!(!armed_after_parsing(&["--server", "--vision-fail-gemm=0", "--model-dir", "/m"]));
        assert!(!armed_after_parsing(&["--server", "--vision-fail-gemm", "off"]));
        assert!(armed_after_parsing(&["--server", "--vision-fail-gemm=1"]));
        // a look-alike is not the flag
        assert!(!armed_after_parsing(&["--server", "--vision-fail-gemm-x"]));
    }

    /// What the old reader answered on the same command lines: the bug, pinned as the reason.
    #[test]
    fn the_value_reader_misses_a_trailing_bare_flag() {
        let last = argv(&["--server", "--model-dir", "/m", "--vision-fail-gemm"]);
        assert!(arg(&last, "--vision-fail-gemm").is_none(), "a value reader sees no value after a trailing flag");
        let mid = argv(&["--server", "--vision-fail-gemm", "--model-dir", "/m"]);
        assert_eq!(arg(&mid, "--vision-fail-gemm"), Some("--model-dir"), "and takes the NEXT FLAG for the value in the middle");
        let zero = argv(&["--vision-fail-gemm=0"]);
        assert!(arg(&zero, "--vision-fail-gemm").is_some(), "and counts `=0` as given");
    }

    /// S-B9-REL-VISIONFIX2 V5: the other two members of H2's class, now registry flags: `--vision-cpu`
    /// (NVFP4 server; a CPU tower is a multi-minute path, and `=0` used to FORCE it) and `--parity`
    /// (a bench protocol flag; as the last argument it silently ran the non-parity recipe). Through
    /// the real argument path: last, middle, first, alone, absent, `=0`, `=off`, `=1`, look-alike.
    #[test]
    fn vision_cpu_and_parity_are_read_by_presence_wherever_they_stand() {
        let _g = STORE.lock().unwrap_or_else(|e| e.into_inner());
        fn on_after(flag: &str, tokens: &[&str]) -> bool {
            let id = crate::opts::lookup(flag).expect("a registry flag");
            assert!(matches!(id.def().ty, crate::opts::Ty::Flag), "--{flag} is a presence flag");
            for t in tokens.iter().filter_map(|t| t.strip_prefix("--")) {
                let name = t.split('=').next().unwrap();
                assert!(name == flag || crate::opts::lookup(name).is_none(), "--{name} is a registry option: keep it out of this test");
            }
            crate::opts::test_set(id, None);
            crate::opts::parse_cli(&argv(tokens)).expect("parse_cli");
            let r = crate::opts::on(id);
            crate::opts::test_set(id, None);
            r
        }
        for flag in ["vision-cpu", "parity"] {
            let f = format!("--{flag}");
            assert!(on_after(flag, &["--server", "--model-dir", "/m", &f]), "{f} as the LAST argument");
            assert!(on_after(flag, &["--server", &f, "--model-dir", "/m"]), "{f} in the middle");
            assert!(on_after(flag, &[&f, "--server", "--model-dir", "/m"]), "{f} first");
            assert!(on_after(flag, &[&f]), "{f} alone");
            assert!(!on_after(flag, &["--server", "--model-dir", "/m"]), "{f} absent");
            assert!(!on_after(flag, &[]), "{f} absent, no arguments");
            assert!(!on_after(flag, &["--server", &format!("{f}=0"), "--model-dir", "/m"]), "{f}=0 is off");
            assert!(!on_after(flag, &["--server", &format!("{f}=off")]), "{f}=off is off");
            assert!(!on_after(flag, &["--server", &f, "off"]), "{f} off is off");
            assert!(on_after(flag, &["--server", &format!("{f}=1")]), "{f}=1 is on");
            assert!(!on_after(flag, &["--server", &format!("{f}-x")]), "a look-alike is not the flag");
        }
        // the options exist in --help (Serve / Diag), each with a help text
        for flag in ["vision-cpu", "parity"] {
            let id = crate::opts::lookup(flag).unwrap();
            assert!(id.def().help.len() > 40 && id.def().env.is_empty(), "--{flag}: a help text and no environment variable");
        }
    }

    /// V4a: `--vision-cpu-threads` is range-checked when the command line is parsed (the process
    /// exits 2 there): a bad value used to fail INSIDE the tower load, which booted the server
    /// text-only with image requests rejected.
    #[test]
    fn a_bad_vision_cpu_threads_value_is_refused_at_startup() {
        let _g = STORE.lock().unwrap_or_else(|e| e.into_inner());
        let id = crate::opt!("vision-cpu-threads");
        for bad in ["0", "65", "-1", "abc", "1.5", ""] {
            crate::opts::test_set(id, None);
            let r = crate::opts::parse_cli(&argv(&["--server", "--vision-cpu-threads", bad]));
            assert!(r.is_err(), "--vision-cpu-threads {bad:?} must be refused, got {r:?}");
            let r = crate::opts::parse_cli(&argv(&[&format!("--vision-cpu-threads={bad}")]));
            assert!(r.is_err(), "--vision-cpu-threads={bad:?} must be refused");
            assert!(crate::opts::get(id).is_none(), "a refused value is not stored");
        }
        // The store is process-global and `VisualTower::load` (other test modules) reads this row: only the
        // default value is ever written to it here; the other values go through the pure accessor.
        crate::opts::test_set(id, None);
        crate::opts::parse_cli(&argv(&["--vision-cpu-threads", "1"])).expect("a good value");
        assert_eq!(crate::opts::ranged(id).unwrap(), 1);
        crate::opts::test_set(id, None);
        for (good, n) in [("1", 1i64), ("8", 8), ("64", 64), (" 4 ", 4)] {
            assert_eq!(crate::opts::ranged_from(id, Some(good)).unwrap(), n);
        }
        assert_eq!(crate::opts::ranged(id).unwrap(), 1, "unset = the default 1");
        // a value that reached the store another way is an Err for the accessor, never a silent default
        for bad in ["0", "65", "-1", "many", ""] {
            assert!(crate::opts::ranged_from(id, Some(bad)).is_err(), "{bad:?}");
        }
    }

    /// The serve path asks the registry, and nothing in the tree reads a registry flag with a value
    /// reader's `is_some()` again. The value readers stay right for options that TAKE a value. The
    /// two bare flags that were still misread (`--vision-cpu`, `--parity`) are registry flags since
    /// S-B9-REL-VISIONFIX2 V5 (the list of bare-and-still-misread names is empty and may not grow).
    #[test]
    fn no_registry_flag_is_read_with_a_value_readers_is_some() {
        const TAKES_A_VALUE: [&str; 5] = ["server-port", "bench-file", "model-dir", "model", "chat-template"];
        const BARE_AND_STILL_MISREAD: [&str; 0] = [];
        let serve = src_text!("exl3_serve");
        assert_eq!(serve.matches(&["vision_boot(model_dir, ", "vision_fail_gemm_armed())"].concat()).count(), 1, "the serve path asks the registry");
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        let mut dirs = vec![root.clone()];
        while let Some(d) = dirs.pop() {
            for e in std::fs::read_dir(&d).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() { dirs.push(p); } else if p.extension().map_or(false, |x| x == "rs") { files.push(p); }
            }
        }
        assert!(files.len() > 40);
        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for f in &files {
            let text = std::fs::read_to_string(f).unwrap();
            for reader in ["arg(args, \"--", "parse_arg(args, \"--", "arg_value(args, \"--", "parse_arg(&args, \"--"] {
                for (at, _) in text.match_indices(reader) {
                    let rest = &text[at + reader.len()..];
                    let Some(q) = rest.find('"') else { continue };
                    let after = rest[q + 1..].trim_start();
                    let Some(after) = after.strip_prefix(')') else { continue };
                    let after = after.trim_start();
                    if !(after.starts_with(".is_some()") || after.starts_with(".is_none()")) { continue; }
                    let name = rest[..q].to_string();
                    if let Some(id) = crate::opts::lookup(&name) {
                        assert!(!matches!(id.def().ty, crate::opts::Ty::Flag | crate::opts::Ty::Bool),
                                "{}: --{name} is a registry flag read with a value reader's is_some(): use opts::on", f.display());
                    }
                    seen.insert(name);
                }
            }
        }
        let allowed: std::collections::BTreeSet<String> = TAKES_A_VALUE.iter().chain(BARE_AND_STILL_MISREAD.iter()).map(|s| s.to_string()).collect();
        assert_eq!(seen, allowed, "options read with <value reader>(..).is_some(): a new one must take a value, or be read by presence");
    }
