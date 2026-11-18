//! Unit tests (module `wp23_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::*;

    fn cal(depth: bool) -> DraftCal {
        DraftCal {
            bins: Default::default(),
            total: 0.0,
            dbins: if depth { Some(vec![Default::default(); WP23_DEPTH_BUCKETS]) } else { None },
        }
    }

    /// deterministic label stream: (pass index, score, accepted), deeper passes less likely
    fn stream(n: usize) -> Vec<(usize, f32, bool)> {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        (0..n).map(|_| {
            let i = (next() % 7) as usize;
            let score = 8.0 + (next() % 2400) as f32 / 100.0;
            let acc = ((next() % 100) as f32 / 100.0) < (score / 32.0 - 0.06 * i as f32);
            (i, score, acc)
        }).collect()
    }

    fn feed(c: &mut DraftCal, labels: &[(usize, f32, bool)]) {
        for (j, &(i, sc, a)) in labels.iter().enumerate() {
            c.add_label(i, sc, a);
            if j % 3 == 2 { c.decay_step(); }
        }
    }

    #[test]
    fn depth_bins_off_is_the_legacy_calibrator() {
        let mut c = cal(false);
        feed(&mut c, &stream(4000));
        for i in 0..8 {
            for q in 0..400 {
                let sc = 4.0 + q as f32 * 0.1;
                assert_eq!(c.estimate_at(i, sc).to_bits(), c.estimate(sc).to_bits(), "i {i} score {sc}");
            }
        }
    }

    #[test]
    fn depth_bins_keep_the_pooled_map_identical() {
        let (mut a, mut b) = (cal(false), cal(true));
        let l = stream(4000);
        feed(&mut a, &l);
        feed(&mut b, &l);
        assert_eq!(a.total.to_bits(), b.total.to_bits());
        assert_eq!(a.bins.len(), b.bins.len());
        for ((ka, va), (kb, vb)) in a.bins.iter().zip(b.bins.iter()) {
            assert_eq!(ka, kb);
            assert_eq!((va.0.to_bits(), va.1.to_bits()), (vb.0.to_bits(), vb.1.to_bits()));
        }
    }

    #[test]
    fn depth_estimate_uses_its_bucket_else_the_pool() {
        let mut c = cal(true);
        // pass 0: logit 20 always accepted; pass 5 (bucket 3): logit 20 accepted 1 in 4
        for _ in 0..40 { c.add_label(0, 20.5, true); }
        for j in 0..40 { c.add_label(5, 20.5, j % 4 == 0); }
        assert!(c.total >= DraftCal::BURN_IN);
        let pooled = c.estimate(20.5);
        assert!((pooled - 50.0 / 80.0).abs() < 1e-12);
        assert!((c.estimate_at(0, 20.5) - 1.0).abs() < 1e-12);
        assert!((c.estimate_at(6, 20.5) - 0.25).abs() < 1e-12); // pass 6 shares bucket 3
        // bucket 2 (passes 3, 4) has no labels: the pooled estimate
        assert_eq!(c.estimate_at(3, 20.5).to_bits(), pooled.to_bits());
        // below every populated bin of the bucket: the pooled fallback (its lowest-bin rule)
        assert_eq!(c.estimate_at(5, 12.0).to_bits(), c.estimate(12.0).to_bits());
        // burn-in: optimistic 1.0
        let mut d = cal(true);
        for _ in 0..10 { d.add_label(5, 20.5, false); }
        assert_eq!(d.estimate_at(5, 20.5), 1.0);
    }

    #[test]
    fn cost_fit_prior_and_slope() {
        assert_eq!(CostFit::default().slope(), 6.0);
        // constant drafts/round: no spread, the prior stays
        let mut g = CostFit::default();
        for _ in 0..200 { g.add(7.0, 56.0); }
        assert!((g.slope() - 6.0).abs() < 1e-9, "{}", g.slope());
        // spread drafts with a true 9 ms/draft slope: identified (shrunk slightly toward 6)
        let mut h = CostFit::default();
        for r in 0..200 { let x = (r % 7 + 1) as f64; h.add(x, 30.0 + 9.0 * x); }
        let s = h.slope();
        assert!(s > 8.7 && s < 9.0, "{s}");
        // never negative
        let mut n = CostFit::default();
        for r in 0..200 { let x = (r % 7 + 1) as f64; n.add(x, 90.0 - 30.0 * x); }
        assert!(n.slope() >= 0.0);
    }

    #[test]
    fn guard_target_rules() {
        // off, too early, or no EMAs: the base target
        assert_eq!(wp23_target(0.4, false, 100, 6.0, 5.6, 56.0), 0.4);
        assert_eq!(wp23_target(0.4, true, 15, 6.0, 5.6, 56.0), 0.4);
        assert_eq!(wp23_target(0.4, true, 100, 6.0, 0.0, 0.0), 0.4);
        // AnsiC-like: 6 ms/draft at 0.1 tok/ms -> 0.6
        assert!((wp23_target(0.4, true, 16, 6.0, 5.6, 56.0) - 0.6).abs() < 1e-12);
        // prose-like: 6 * 2.7 / 45 = 0.36 < base -> base (the guard only raises)
        assert_eq!(wp23_target(0.4, true, 100, 6.0, 2.7, 45.0), 0.4);
        // long context, expensive rows: clamped at 0.8
        assert_eq!(wp23_target(0.4, true, 100, 20.0, 5.0, 60.0), 0.8);
        // an operator base above the guard wins
        assert_eq!(wp23_target(0.9, true, 100, 20.0, 5.0, 60.0), 0.9);
    }

    #[test]
    fn depth_buckets_cover_the_window() {
        let b: Vec<usize> = (0..crate::exl3_forward::MTP_MAX_K).map(wp23_depth_bucket).collect();
        assert_eq!(b, vec![0, 1, 1, 2, 2, 3, 3, 3, 3, 3]); // MTP26/DEPTH: k window 7 -> 10 (buckets unchanged, coarse bins)
        assert!(b.iter().all(|&x| x < WP23_DEPTH_BUCKETS));
    }
