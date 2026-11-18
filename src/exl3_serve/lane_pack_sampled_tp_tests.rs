//! Unit tests (module `lane_pack_sampled_tp_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::*;

    /// The registry store is process-global: one test at a time writes this option's row.
    static STORE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn argv(tokens: &[&str]) -> Vec<String> { tokens.iter().map(|s| s.to_string()).collect() }

    #[test]
    fn the_option_reaches_the_boot_and_rides_to_the_nodes() {
        let _g = STORE.lock().unwrap_or_else(|e| e.into_inner());
        let id = crate::opt!("lane-pack-sampled-tp");
        crate::opts::test_set(id, None);
        // default: ON (sampled requests join packed rounds under TP)
        assert!(pack_sampled_under_tp(), "the default is on");
        assert_eq!(id.def().def, "on");
        // the head ships it: an SPMD option (the node installs the head's value; the boot agree digest covers it)
        assert_eq!(id.def().scope, crate::opts::Scope::Spmd);
        crate::opts::parse_cli(&argv(&["--lane-pack-sampled-tp", "off"])).unwrap();
        assert!(!pack_sampled_under_tp(), "--lane-pack-sampled-tp off");
        let shipped = crate::opts::snapshot();
        assert!(shipped.contains(&("lane-pack-sampled-tp".to_string(), "off".to_string())), "in the TpConfig snapshot: {shipped:?}");
        assert!(crate::opts::spmd_digest().is_some(), "and in the boot-agree digest");
        crate::opts::test_set(id, None);
        crate::opts::parse_cli(&argv(&["--lane-pack-sampled-tp=on"])).unwrap();
        assert!(pack_sampled_under_tp(), "--lane-pack-sampled-tp=on");
        // a value that is not on|off is refused when the command line is parsed, with the accepted spellings
        crate::opts::test_set(id, None);
        let e = crate::opts::parse_cli(&argv(&["--lane-pack-sampled-tp", "maybe"])).unwrap_err();
        assert!(e.contains("--lane-pack-sampled-tp") && e.contains("'maybe'"), "{e}");
        assert!(crate::opts::parse_cli(&argv(&["--lane-pack-sampled-tp"])).is_err(), "a bare flag needs a value");
        assert!(crate::opts::get(id).is_none(), "a refused value is not stored");
        // `off` keeps the graph set of the builds before the option: the greedy family alone; `on` adds the
        // family the boot's sampling rule asks for (real-q under --spec-sampling ratio, the default)
        use crate::lane_pack::{fams, PackFam};
        assert_eq!(fams(true, false), vec![PackFam::Greedy]);
        assert_eq!(fams(false, false), vec![PackFam::Greedy]);
        assert_eq!(fams(true, true), vec![PackFam::Greedy, PackFam::Rq]);
        assert_eq!(fams(false, true), vec![PackFam::Greedy, PackFam::Match]);
        // the boot's decision: TP=1 always packs sampled requests; under TP the option decides (and a skipped
        // sampled lane is counted as `sampled`: `sampled_lone = dump_on || !pack_sampled`)
        let src = src_text!("exl3_serve");
        let code = &src[..src.find("mod lane_pack_sampled_tp_tests").unwrap()];
        assert!(code.contains("let sampled_ok = tp_rank.is_none() || pack_sampled_under_tp();"));
        assert!(code.contains("sampled_lone: crate::wp24::dump_on() || !self.pack_sampled,"));
        crate::opts::test_set(id, None);
    }
