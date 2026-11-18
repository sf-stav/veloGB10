//! Unit tests (module `mtp_switch_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::*;
    use crate::exl3_forward::{mtp_disabled_by_opt, mtp_off_by_opt, mtp_off_of, MtpOff};
    use crate::opts::{Cat, Scope, Ty};

    /// The registry tests share one process-global store: serialize the ones that write `mtp`.
    static STORE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn the_decision_function() {
        // not passed / auto / on (and the NVFP4 server's older spellings of on) = speculation ON
        for v in [None, Some("auto"), Some("on"), Some("1"), Some("true")] {
            assert_eq!(mtp_off_of(v, false), None, "--mtp {v:?} must leave the head on");
        }
        // off (and its older spellings) = OFF, attributed to --mtp
        for v in ["off", "0", "false"] {
            assert_eq!(mtp_off_of(Some(v), false), Some(MtpOff::MtpFlag), "--mtp {v}");
        }
        // the hatch keeps working, alone and next to any --mtp value (it names itself)
        for v in [None, Some("auto"), Some("on"), Some("off")] {
            assert_eq!(mtp_off_of(v, true), Some(MtpOff::Hatch), "--exl3-no-mtp with --mtp {v:?}");
        }
        assert_eq!(MtpOff::MtpFlag.flag(), "--mtp off");
        assert_eq!(MtpOff::Hatch.flag(), "--exl3-no-mtp");
    }

    #[test]
    fn mtp_is_a_registered_spmd_option() {
        let id = crate::opts::lookup("mtp").expect("--mtp must be in the registry");
        let o = id.def();
        assert_eq!(o.env, "", "a new option has no environment variable");
        assert_eq!(o.scope, Scope::Spmd, "--mtp must ride TpConfig to the nodes and enter the boot digest");
        assert_eq!(o.cat, Cat::Spec);
        assert_eq!(o.def, "auto");
        match o.ty {
            Ty::Choice(l) => for v in ["auto", "on", "off"] { assert!(l.contains(&v), "--mtp {v}"); },
            other => panic!("--mtp must be a Choice (validated at startup), got {other:?}"),
        }
        assert!(o.help.contains("MTP OFF (--mtp off)"), "the help names the banner: {}", o.help);
        // the hatch stays registered and Spmd
        let h = crate::opts::lookup("exl3-no-mtp").expect("--exl3-no-mtp stays");
        assert_eq!(h.def().scope, Scope::Spmd);
    }

    /// The real parser into the real store, read back by the gate every EXL3 site uses.
    #[test]
    fn the_command_line_reaches_the_gate() {
        let _g = STORE.lock().unwrap_or_else(|e| e.into_inner());
        let (mtp, hatch) = (crate::opt!("mtp"), crate::opt!("exl3-no-mtp"));
        let args = |a: &[&str]| -> Vec<String> { a.iter().map(|s| s.to_string()).collect() };
        let reset = || { crate::opts::test_set(mtp, None); crate::opts::test_set(hatch, None); };
        reset();
        assert!(!mtp_disabled_by_opt(), "a default boot speculates");
        for off in [&["--mtp", "off"][..], &["--mtp=off"][..]] {
            reset();
            crate::opts::parse_cli(&args(off)).unwrap();
            assert_eq!(mtp_off_by_opt(), Some(MtpOff::MtpFlag), "{off:?}");
            assert!(mtp_disabled_by_opt(), "{off:?} must turn the head off");
        }
        for on in [&["--mtp", "on"][..], &["--mtp=auto"][..], &["--mtp", "auto"][..]] {
            reset();
            crate::opts::parse_cli(&args(on)).unwrap();
            assert_eq!(mtp_off_by_opt(), None, "{on:?} is unchanged: the head runs");
        }
        reset();
        crate::opts::parse_cli(&args(&["--exl3-no-mtp"])).unwrap();
        assert_eq!(mtp_off_by_opt(), Some(MtpOff::Hatch));
        // an invalid value is refused by the startup parse, with the accepted spellings in the message
        reset();
        let e = crate::opts::parse_cli(&args(&["--mtp", "maybe"])).unwrap_err();
        assert!(e.contains("--mtp takes auto|on|off") && e.contains("'maybe'"), "{e}");
        assert!(crate::opts::parse_cli(&args(&["--mtp"])).is_err(), "a bare --mtp needs a value");
        // TP: the head's value is what a node installs (and an unset head clears a node's own)
        reset();
        crate::opts::parse_cli(&args(&["--mtp", "off"])).unwrap();
        let shipped = crate::opts::snapshot();
        assert!(shipped.contains(&("mtp".to_string(), "off".to_string())), "--mtp off must be in the TpConfig snapshot: {shipped:?}");
        assert!(crate::opts::spmd_digest().is_some(), "and in the boot agree digest");
        reset();
        let mine: Vec<(String, String)> = shipped.into_iter().filter(|(k, _)| k == "mtp").collect();
        // (install only this row: a full install_head would clear options other tests hold)
        crate::opts::test_set(mtp, mine.first().map(|(_, v)| v.as_str()));
        assert!(mtp_disabled_by_opt(), "a node with the head's --mtp off has its head off");
        reset();
    }

    /// SURFACE2 item 4: with speculation off, no boot line describes speculative rounds (the WP23
    /// DDS line still printed under `--mtp off`), and the head line says the head is not used.
    #[test]
    fn speculation_boot_lines_follow_the_switch() {
        assert!(spec_live(true, None));
        assert!(!spec_live(true, Some(MtpOff::MtpFlag)) && !spec_live(true, Some(MtpOff::Hatch)));
        assert!(!spec_live(false, None), "a pack without a draft head never speculates");
        assert_eq!(head_loaded_line(false, None), None);
        assert_eq!(head_loaded_line(true, None).unwrap(),
                   "[exl3-serve] M1 pack head loaded (MTP draft head for chain-verify serving)", "the default line is unchanged");
        let off = head_loaded_line(true, Some(MtpOff::MtpFlag)).unwrap();
        assert!(off.starts_with("[exl3-serve] M1 pack head loaded") && off.contains("NOT used on this boot: --mtp off"), "{off}");
        assert!(head_loaded_line(true, Some(MtpOff::Hatch)).unwrap().contains("NOT used on this boot: --exl3-no-mtp"));
        // source pin (the boot needs a GPU): the line before each of these prints is its guard
        let src = src_text!("exl3_serve");
        for (line, guard) in [("WP23 DDS: depth ceiling", "if spec_live && draft_conf > 0.0 {"),
                              ("load-adaptive batching (--spec-lanes-max)", "if width > 1 && spec_live {"),
                              ("serial lane order (--lane-order)", "if width > 1 && spec_live {"),
                              ("spec sampling: {} (draft temperature", "&& !crate::exl3_forward::mtp_disabled_by_opt() {")] {
            let print = ["crate::dprintln!(\"[exl3-serve] ", line].concat();
            assert_eq!(src.matches(print.as_str()).count(), 1, "one `{line}` boot line");
            let before = &src[..src.find(print.as_str()).unwrap()];
            let guard_line = before.lines().rev().nth(1).unwrap();
            assert!(guard_line.trim_end().ends_with(guard), "`{line}` must sit behind `{guard}`, found `{}`", guard_line.trim());
        }
    }

    #[test]
    fn the_boot_banner() {
        assert_eq!(mtp_banner(false, None, 5, 0.4), None, "no MTP head in the pack: no line");
        assert_eq!(mtp_banner(false, Some(MtpOff::MtpFlag), 5, 0.4), None);
        let on = mtp_banner(true, None, 5, 0.4).unwrap();
        assert!(on.starts_with("[exl3-serve] MTP chain-verify ON (k=5,") && on.contains("dynamic draft stop at 0.4"), "{on}");
        assert!(on.contains("--mtp off"), "the ON line names the switch that turns it off: {on}");
        let off = mtp_banner(true, Some(MtpOff::MtpFlag), 5, 0.4).unwrap();
        assert!(off.contains("MTP OFF (--mtp off)"), "{off}");
        assert!(!off.contains("chain-verify ON"), "{off}");
        let hatch = mtp_banner(true, Some(MtpOff::Hatch), 5, 0.4).unwrap();
        assert_eq!(hatch, "[exl3-serve] MTP chain-verify OFF (--exl3-no-mtp set — diagnostics hatch)", "unchanged line");
        // both OFF lines keep the prefix the gate scripts match (scripts/i18d_chain.sh MOFF)
        for l in [&off, &hatch] { assert!(l.starts_with("[exl3-serve] MTP chain-verify OFF"), "{l}"); }
    }
