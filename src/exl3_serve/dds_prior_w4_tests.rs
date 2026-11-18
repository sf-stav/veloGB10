//! Unit tests (module `dds_prior_w4_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::*;

    /// The pre-TP-4F `dds_prior(true)` body, verbatim (only the option read is replaced by `raw`): the frozen
    /// reference that world 2 must equal bit for bit.
    fn old_world2(raw: Option<&str>) -> f64 {
        match raw.map(|v| v.trim().to_ascii_lowercase()) {
            Some(v) if v == "0" || v == "off" || v == "false" => CostFit::PRIOR_SLOPE,
            Some(v) if v == "1" || v == "on" || v.is_empty() => TP2_DDS_PRIOR,
            Some(v) => v.parse::<f64>().ok().filter(|x| *x > 0.0).unwrap_or(TP2_DDS_PRIOR),
            None => TP2_DDS_PRIOR,
        }
    }

    const RAWS: [Option<&str>; 16] = [None, Some(""), Some(" "), Some("on"), Some("ON"), Some("1"), Some("off"), Some("OFF"),
        Some("0"), Some("false"), Some("2.7"), Some("3.5"), Some("0.0"), Some("-1"), Some("garbage"), Some(" 1.25 ")];

    #[test]
    fn world_1_and_2_are_bit_identical_to_the_pre_world_aware_prior() {
        assert_eq!(dds_prior(false).to_bits(), CostFit::PRIOR_SLOPE.to_bits());
        assert_eq!(dds_prior_w(1).to_bits(), 6.0f64.to_bits());
        assert_eq!(dds_prior_w(0).to_bits(), 6.0f64.to_bits());
        for raw in RAWS {
            assert_eq!(resolve_dds_prior(raw, 2).to_bits(), old_world2(raw).to_bits(), "raw {raw:?}");
        }
        // unset (the test process has no --tp-dds-prior): the runtime entry points are the old constants
        assert_eq!(dds_prior(true).to_bits(), 2.7f64.to_bits());
        assert_eq!(dds_prior_tp().to_bits(), 2.7f64.to_bits());
        assert_eq!(dds_prior_w(2).to_bits(), 2.7f64.to_bits());
    }

    #[test]
    fn world_2_boot_hash_word_and_hash_are_unchanged() {
        let sample = [128u32, 2048, 7, 1, 512, 0, 262144, 1, 0, 0x3F00_0000, 0, 0, 1, 2, 3];
        let hash = |word: u32| fnv32(0xB007, sample.iter().copied().chain([word]).chain([9, 9, 9]));
        assert_eq!(boot_prior_word(2), (2.7f64 as f32).to_bits(), "the world-2 word");
        assert_eq!(boot_prior_word(2), (dds_prior(true) as f32).to_bits(), "== the pre-change `(dds_prior(true) as f32).to_bits()`");
        assert_eq!(hash(boot_prior_word(2)), hash((dds_prior(true) as f32).to_bits()));
        // every world-2 value the flag can produce hashes exactly as the old code did
        for raw in RAWS {
            assert_eq!(hash((resolve_dds_prior(raw, 2) as f32).to_bits()), hash((old_world2(raw) as f32).to_bits()), "raw {raw:?}");
        }
        // world 4's word differs (its prior is 2.1): ranks must agree among themselves, and do, both resolve the same flag
        assert_eq!(boot_prior_word(4), (2.1f64 as f32).to_bits());
        assert_ne!(hash(boot_prior_word(4)), hash(boot_prior_word(2)));
    }

    #[test]
    fn world_4_default_is_the_derived_value_and_an_override_still_overrides() {
        assert_eq!(TP4_DDS_PRIOR, 2.1);
        assert_eq!(dds_prior_w(4).to_bits(), TP4_DDS_PRIOR.to_bits());
        assert_eq!(dds_prior_tp4().to_bits(), TP4_DDS_PRIOR.to_bits());
        // unset / on / unusable = the world's default (the TP=2 arms' "unusable falls back to default" rule)
        for raw in [None, Some(""), Some("on"), Some("1"), Some("garbage"), Some("-1"), Some("0.0")] {
            assert_eq!(resolve_dds_prior(raw, 4), TP4_DDS_PRIOR, "raw {raw:?}");
        }
        for raw in [Some("off"), Some("0"), Some("false")] { assert_eq!(resolve_dds_prior(raw, 4), 6.0, "raw {raw:?}"); }
        assert_eq!(resolve_dds_prior(Some("2.7"), 4), 2.7, "a numeric override overrides at W=4 (even to the W=2 value)");
        assert_eq!(resolve_dds_prior(Some("3.5"), 4), 3.5);
        assert_eq!(resolve_dds_prior(Some(" 1.25 "), 4), 1.25);
        // the override is world-independent: the same number at world 2 and 4
        for n in ["0.5", "1.9", "4.4"] { assert_eq!(resolve_dds_prior(Some(n), 2), resolve_dds_prior(Some(n), 4)); }
        // any other TP world keeps the TP=2 constant
        assert_eq!(resolve_dds_prior(None, 3), 2.7);
    }
