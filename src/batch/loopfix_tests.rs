//! Unit tests (module `loopfix_tests`) moved out of the old inline block; text unchanged.

    use super::{hist_trim_cut, pen_draft_extras};

    #[test]
    fn trim_matches_the_old_rule_when_nothing_is_pending() {
        assert_eq!(hist_trim_cut(256, 0), 0);
        assert_eq!(hist_trim_cut(257, 0), 128);   // was: len > 256 => drain(0..128)
        assert_eq!(hist_trim_cut(400, 3), 128);
    }

    #[test]
    fn trim_never_drops_uncounted_tokens() {
        for len in 0..600usize {
            for pending in 0..=len {
                let cut = hist_trim_cut(len, pending);
                assert!(len - cut >= pending, "len {len} pending {pending} cut {cut}");
            }
        }
    }

    #[test]
    fn draft_extras_follow_plain_decode() {
        let (cap, pe) = (16usize, 16usize);
        let d = [11u32, 22, 33];
        let e = pen_draft_extras(&d, cap, pe);
        let col = |c: usize| -> Vec<i32> { e[c * pe..c * pe + pe].iter().copied().filter(|&t| t >= 0).collect() };
        assert_eq!(col(0), Vec::<i32>::new());           // the committed column: output only
        assert_eq!(col(1), vec![11]);
        assert_eq!(col(2), vec![11, 22]);
        assert_eq!(col(3), vec![11, 22, 33]);           // width n = 4: last real column
        assert_eq!(col(15), vec![11, 22, 33]);          // unused columns repeat the deepest prefix
        assert_eq!(pen_draft_extras(&[], cap, pe), vec![-1i32; cap * pe]);
    }
