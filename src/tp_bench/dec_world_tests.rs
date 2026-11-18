    use super::*;

    /// The key generator the W=4 key-merge bench validates against must actually exercise the tie-break
    /// (equal values on several ranks, lowest id wins) and never hand two ranks the same key.
    #[test]
    fn dec_key_ties_are_frequent_and_keys_distinct() {
        let (mut ties, mut rows) = (0usize, 0usize);
        for it in 0..50 {
            for row in 0..64 {
                let ks: Vec<u64> = (0..4).map(|r| dec_key(r, row, it)).collect();
                for i in 0..4 { for j in i + 1..4 { assert_ne!(ks[i], ks[j], "ranks {i},{j} row {row} it {it}"); } }
                let top = ks.iter().map(|k| k >> 32).max().unwrap();
                if ks.iter().filter(|k| **k >> 32 == top).count() > 1 { ties += 1; }
                // the decoded id of the max key is a valid global id (< 4 * 62080)
                let mx = *ks.iter().max().unwrap();
                assert!(0xFFFF_FFFFu32 - (mx & 0xFFFF_FFFF) as u32 <= 4 * 62_080);
                rows += 1;
            }
        }
        assert!(ties * 4 >= rows, "only {ties}/{rows} rows had a top-value tie: the lowest-id tie-break is not exercised");
        assert_eq!(dec_key(1, 7, 3), dec_key(1, 7, 3), "deterministic");
    }

    /// The max of the keys is the lowest-id holder of the top value (the total order the engine relies on).
    #[test]
    fn dec_key_max_is_value_desc_then_lowest_id() {
        for it in 0..20 {
            for row in 0..64 {
                let ks: Vec<(u64, u32)> = (0..4).map(|r| {
                    let k = dec_key(r, row, it);
                    (k >> 32, 0xFFFF_FFFFu32 - (k & 0xFFFF_FFFF) as u32)
                }).collect();
                let top = ks.iter().map(|k| k.0).max().unwrap();
                let lowest = ks.iter().filter(|k| k.0 == top).map(|k| k.1).min().unwrap();
                let mx = (0..4).map(|r| dec_key(r, row, it)).max().unwrap();
                assert_eq!(0xFFFF_FFFFu32 - (mx & 0xFFFF_FFFF) as u32, lowest, "row {row} it {it}");
            }
        }
    }
