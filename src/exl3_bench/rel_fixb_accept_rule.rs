//! Host-side proof tests (module `rel_fixb_accept_rule`), moved out of `exl3_bench.rs`; no GPU unless a test says so.
//! Part of the `exl3_bench` probe module (see `mod.rs` for the map); test bodies are unchanged.
    /// The lone kernel: d has k drafts, e has k+1 argmaxes; a = longest match prefix.
    fn accept_lone(d: &[i32], e: &[i32], k: usize) -> usize {
        let mut a = 0;
        while a < k && d[a] == e[a] { a += 1; }
        a
    }

    /// xq_accept_seg AFTER the K2 fix: kj = k + 1 rows; the loop bound is the DRAFTS
    /// (kj - 1 = k), so a never reaches kj — the commit replays at most k + 1 rows.
    fn accept_seg_fixed(d: &[i32], e: &[i32], kj: usize) -> usize {
        let mut a = 0;
        while a < kj - 1 && d[a] == e[a] { a += 1; }
        a
    }

    /// The PRE-FIX loop (REL v0.7.3): bounded on the ROW count — a matching zero/stale
    /// tail draft slot let a reach kj, the commit replayed kj + 1 rows and at k = 7
    /// C > XQ_GDN_RING_CMAX trapped in xq_gdn_commit_ring_seg.
    fn accept_seg_pre_fix(d: &[i32], e: &[i32], kj: usize) -> usize {
        let mut a = 0;
        while a < kj && d[a] == e[a] { a += 1; }
        a
    }

    #[test]
    fn accept_rule_model_k1_to_k7_zero_tail() {
        for k in 1..=7usize {
            let kj = k + 1;
            // All drafts match, argmaxes equal the drafts, and the TAIL draft slot
            // (a zero or stale value past the last draft) equals the last argmax.
            let d: Vec<i32> = (0..kj).map(|i| if i < k { 100 + i as i32 } else { 0 }).collect();
            let e: Vec<i32> = (0..kj).map(|i| if i < k { 100 + i as i32 } else { 0 }).collect();
            // The fixed rule accepts all k drafts (bonus = e[k]) and NEVER more.
            assert_eq!(accept_seg_fixed(&d, &e, kj), k, "k={k}");
            assert_eq!(accept_seg_fixed(&d, &e, kj), accept_lone(&d, &e, k), "k={k} == lone");
            // The pre-fix rule ran one past the drafts — the exact K2 bug.
            assert_eq!(accept_seg_pre_fix(&d, &e, kj), kj, "k={k}: pre-fix overrun (proof the model can fail)");
            // Partial matches: a stops at the first mismatch, identical to the lone rule.
            for stop in 0..k {
                let mut d2 = d.clone();
                d2[stop] = -1; // mismatch at `stop`
                assert_eq!(accept_seg_fixed(&d2, &e, kj), stop, "k={k} stop={stop}");
                assert_eq!(accept_seg_fixed(&d2, &e, kj), accept_lone(&d2, &e, k), "k={k} stop={stop}");
            }
            // The commit row count (a + 1) stays within the segment at every k.
            assert!(accept_seg_fixed(&d, &e, kj) + 1 <= kj, "k={k}");
        }
    }
