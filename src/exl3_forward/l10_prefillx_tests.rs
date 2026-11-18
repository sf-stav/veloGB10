//! Unit tests (module `l10_prefillx_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    use super::*;

    #[test]
    fn md5_matches_rfc1321_vectors() {
        assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(md5_hex(b"The quick brown fox jumps over the lazy dog"), "9e107d9d372bb6826bd81d3542a419d6");
        // a multi-block message (> 64 bytes, padding crosses a block)
        assert_eq!(md5_hex(b"12345678901234567890123456789012345678901234567890123456789012345678901234567890"),
                   "57edf4a22be3c955ac49da2e2107b67a");
    }

    #[test]
    fn tier_passes_q1_default_and_rejects_the_flash256_defect() {
        // Q1 ids16 default vs its bit-0 floor: S 2.21e-2 / 1.03e-2, KV 2.86e-2 / 1.60e-2, rank 1
        assert!(prefillx_tier_fails(2.21e-2, Some(2.86e-2), 1.03e-2, Some(1.60e-2), 1, 0.0).is_empty());
        // Q1 p2k default: rank 4 gap 0.95 sits inside the near-tie
        assert!(prefillx_tier_fails(4.7e-2, Some(1.04e-1), 2.83e-2, Some(6.2e-2), 4, 0.95).is_empty());
        // the flash256 staging defect (--pq8-flash=1): 5.3x / 4.8x the floor
        let f = prefillx_tier_fails(1.50e-1, Some(2.95e-1), 2.83e-2, Some(6.2e-2), 1, 0.0);
        assert_eq!(f.len(), 2, "{f:?}");
        // first token outside the near-tie, NaN relL2 (report missing), KV on one side only
        assert_eq!(prefillx_tier_fails(1e-2, None, 1e-2, None, 6, 0.5).len(), 1);
        assert_eq!(prefillx_tier_fails(1e-2, None, 1e-2, None, 2, 1.5).len(), 1);
        assert_eq!(prefillx_tier_fails(f64::NAN, None, 1e-2, None, 1, 0.0).len(), 1);
        assert_eq!(prefillx_tier_fails(1e-2, Some(1e-2), 1e-2, None, 1, 0.0).len(), 1);
    }
