//! Unit tests (module `rel_gates2_reprime_words_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    use super::*;

    /// GATES2 finding: the REPRIME readback must fit the pinned buffer it is sized with, and cover every word
    /// xq_accept writes (out[0..=k+2]); accept_out itself is wider (K4) and is NOT the copy length.
    #[test]
    fn reprime_readback_is_the_pin_width_not_the_accept_out_width() {
        assert!(RP_ACCEPT_WORDS >= MTP_MAX_K + 3, "must cover out[0..=k+2] at the max depth");
        assert!(RP_ACCEPT_WORDS <= accept_out_words(), "must fit inside accept_out");
        assert!(accept_out_words() > RP_ACCEPT_WORDS, "K4: accept_out is deliberately wider (packed tail to 16 rows)");
        assert_eq!(RP_ACCEPT_WORDS, MTP_MAX_K + 3, "readback width follows the max depth (k=7 -> 10 words/40 B; k=10 -> 13 words/52 B)");
    }

    /// WIDEVERIFY1 refinement 2: a default scratch (vrows = MTP_MAX_K + 1) keeps every pin, readback,
    /// accept width and re-prime width of before; a wide-copy scratch (COPY_VROWS) covers k = COPY_MAX_K.
    #[test]
    fn wide_copy_verify_pins_follow_the_scratch_rows() {
        let d = MTP_MAX_K + 1;
        assert_eq!(rp_accept_words(d), RP_ACCEPT_WORDS, "default scratch: the 13-word pin, unchanged");
        assert_eq!(accept_out_words_for(d), accept_out_words(), "default scratch: accept_out unchanged (16 words)");
        assert_eq!(accept_out_words(), 16);
        for w in [MTP_MAX_K + 1, 12, 16, 64] {
            for k in 0..=MTP_MAX_K {
                assert_eq!(rp_rows_of(w, d, k), MTP_MAX_K.min(w - 1), "every round of <= MTP_MAX_K drafts keeps the old re-prime width");
                assert_eq!(rp_rows_of(w, COPY_VROWS, k), MTP_MAX_K.min(w - 1), "also on a wide scratch");
            }
        }
        // the wide copy verify: k <= COPY_MAX_K drafts, xq_accept writes out[0..=k+2]
        assert_eq!(COPY_VROWS, COPY_MAX_K + 1);
        assert!(COPY_MAX_K > MTP_MAX_K && COPY_VROWS <= SAVE_PLANE_ROWS && COPY_VROWS <= GDN_RING_CMAX);
        assert!(rp_accept_words(COPY_VROWS) >= COPY_MAX_K + 3, "the wide pin covers out[0..=k+2] at k = COPY_MAX_K");
        assert!(rp_accept_words(COPY_VROWS) <= accept_out_words_for(COPY_VROWS), "the readback fits accept_out");
        assert!(accept_out_words_for(COPY_VROWS) >= COPY_MAX_K + 3 && accept_out_words_for(COPY_VROWS) >= 16);
        for k in MTP_MAX_K + 1..=COPY_MAX_K {
            assert_eq!(rp_rows_of(COPY_VROWS, COPY_VROWS, k), COPY_MAX_K, "a wide copy round re-primes every accepted draft");
            assert!(rp_rows_of(COPY_VROWS, COPY_VROWS, k) >= k && rp_rows_of(COPY_VROWS, COPY_VROWS, k) < COPY_VROWS,
                    "head_reprime_fixed_launches needs rows < the scratch width");
        }
    }

    /// The ring-commit extent is mirrored in the kernel source (static smem kr_sh / g_sh + the device trap).
    #[test]
    fn gdn_ring_cmax_mirrors_the_kernel() {
        let cu = src_text!("kernels/exl3_bench");
        let line = cu.lines().find(|l| l.starts_with("#define XQ_GDN_RING_CMAX ")).expect("XQ_GDN_RING_CMAX define");
        let v: usize = line.split_whitespace().nth(2).unwrap().parse().unwrap();
        assert_eq!(v, GDN_RING_CMAX);
        assert_eq!(cu.matches("__shared__ float kr_sh[XQ_GDN_RING_CMAX][128];").count(), 3, "the three ring-commit bodies");
    }
