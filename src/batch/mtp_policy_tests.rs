//! Unit tests (module `mtp_policy_tests`) moved out of the old inline block; text unchanged.

    use super::{MtpPolicy, MTP_EVAL_FIRST, MTP_EVAL_WINDOW};

    /// An r(d) table shaped like the MEASURED TP=4 serving calibration at 2k ctx
    /// (PLAN/03 §results: r(2/3/4/5) = 1.13/1.26/1.39/1.48, fresh calibration) — the regime
    /// where the d3 sweet spot lives.
    fn r_tp4_2k() -> Vec<(usize, Vec<(usize, f32)>)> {
        vec![(2048, vec![(2, 1.13), (3, 1.26), (4, 1.39), (5, 1.48), (6, 1.6), (8, 1.9)])]
    }

    /// Simulate `steps` MTP steps at the policy's current depth with a fixed hazard curve
    /// `hz` (conditional acceptance per position), then tick. Repeats until `windows`
    /// evaluations have happened. Returns the depth history.
    fn run(policy: &mut MtpPolicy, hz: &[f64], steps_per_window: usize, windows: usize) -> Vec<usize> {
        let mut hist = vec![policy.depth()];
        for _ in 0..windows {
            for _ in 0..steps_per_window {
                // draw the accepted prefix from the hazard curve
                let mut acc = 0usize;
                while acc < policy.depth() - 1 && rand_step(hz.get(acc).copied().unwrap_or(0.0)) { acc += 1; }
                // record_step is private but the test module is a child of this module — visible.
                let emitted = (acc + 1) as u64;
                policy.record_step((policy.depth() - 1) as u64, acc as u64, emitted);
            }
            policy.tick(2048);
            hist.push(policy.depth());
        }
        hist
    }

    /// Deterministic stand-in for a Bernoulli(h) draw: an LCG whose STATE is the mixed word
    /// (the first version forgot to store it back and drew from a linear sequence — correlated
    /// draws that measured 77% acceptance from a 44% hazard and invalidated the fixture).
    fn rand_step(h: f64) -> bool {
        use std::cell::Cell;
        thread_local! { static I: Cell<u64> = Cell::new(0x853c49e6748fea9b); }
        I.with(|i| {
            let s = i.get().wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            i.set(s);
            let x = (s >> 33) as f64 / (1u64 << 31) as f64;
            x < h
        })
    }

    /// PLAN/10 #12 regression: the falling-conditional weak-head shape (the documented hazard
    /// decay — yield_at's own doc: "hazards decay because each draft is conditioned on a chain
    /// of its own guesses") with the MEASURED TP=4 r(d) at 2k. At hz = [.44, .50, .45]:
    ///   yield(3)/r(3) = 1.660/1.26 = 1.317    yield(4)/r(4) = 1.759/1.39 = 1.266
    /// d3 wins by 4.1% — MORE than 0, LESS than the 5% up-margin. The OLD symmetric margin
    /// blocked exactly this switch (the P2 carry-forward: "policy never tried d3"); with the
    /// asymmetric down-margin (1.0) the descent must happen.
    /// COPY27B: with the copy arm's exploration on, a policy that stepped down still visits the
    /// deepest depth every `explore_every`-th window (so stale deep hazards are refreshed); off
    /// (the default, every non-copy boot) the low-alpha descent is unchanged and never probes.
    #[test]
    fn copy_open_deep_only_unpinned() {
        use crate::gpu::MAX_AUTO_DEPTH;
        let mut p = MtpPolicy::new(true, None, None, r_tp4_2k());
        assert_eq!(p.depth(), 4);
        p.open_deep();
        assert_eq!(p.depth(), MAX_AUTO_DEPTH);
        let mut q = MtpPolicy::new(true, None, Some(3), r_tp4_2k());
        q.open_deep();
        assert_eq!(q.depth(), 3, "a pinned depth stays pinned");
    }

    #[test]
    fn copy_explore_probes_max_depth_only_when_armed() {
        use crate::gpu::MAX_AUTO_DEPTH;
        let hz = [0.44, 0.50, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45];
        let mut off = MtpPolicy::new(true, None, None, r_tp4_2k());
        let h_off = run(&mut off, &hz, MTP_EVAL_WINDOW as usize, 12);
        let mut on = MtpPolicy::new(true, None, None, r_tp4_2k());
        on.set_explore(4);
        let h_on = run(&mut on, &hz, MTP_EVAL_WINDOW as usize, 12);
        let maxd = r_tp4_2k()[0].1.iter().map(|&(d, _)| d).max().unwrap();
        assert!(!h_off.contains(&MAX_AUTO_DEPTH) || maxd < MAX_AUTO_DEPTH, "default policy probed: {h_off:?}");
        if maxd >= MAX_AUTO_DEPTH {
            assert!(h_on.contains(&MAX_AUTO_DEPTH), "armed policy never probed d{MAX_AUTO_DEPTH}: {h_on:?}");
        }
        assert!(h_on.iter().filter(|&&d| d == MAX_AUTO_DEPTH).count() <= 4, "probes are one window each: {h_on:?}");
    }

    #[test]
    fn low_alpha_descends_to_d3() {
        let mut p = MtpPolicy::new(true, None, None, r_tp4_2k());
        assert_eq!(p.depth(), 4, "auto policy opens at 4");
        let hz = [0.44, 0.50, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45];
        let hist = run(&mut p, &hz, MTP_EVAL_FIRST as usize + 32, 4);
        assert!(hist.contains(&3), "policy never tried d3 at a low-alpha curve; history {hist:?}");
    }

    /// Control: a HIGH-α curve (@1:83%) keeps deep depths — the asymmetry must not collapse
    /// the policy to d2/d3 when deep drafts pay (the 9B prose regime the symmetric margin was
    /// built for: p1≈0.83 predicted 3.94 tok/step at d6 vs 2.64 actual; the hazard model
    /// handles that, the margin must not fight it).
    #[test]
    fn high_alpha_stays_deep() {
        let mut p = MtpPolicy::new(true, None, None, r_tp4_2k());
        let hz = [0.83, 0.85, 0.86, 0.87, 0.88, 0.88, 0.88, 0.88];
        let hist = run(&mut p, &hz, MTP_EVAL_WINDOW as usize + 8, 4);
        assert!(hist.iter().all(|&d| d >= 4), "policy bailed shallow on a high-alpha curve; history {hist:?}");
    }

    /// Up-switches keep the margin: a d3 policy with a curve that favors d5 by only 1% must
    /// NOT switch (that's the flapping the up-margin exists to prevent).
    #[test]
    fn up_switch_keeps_margin() {
        let mut p = MtpPolicy::new(true, None, Some(3), r_tp4_2k());
        // At d3 the marginal positions 3/4 are unobserved (prior 0.5 carried) — the model
        // extrapolates pessimistically, so a d5 that's really only ~1% better on paper never
        // clears the 1.05 up-margin. Pin at 3, feed a flat hazard where deeper is barely better.
        let hz = [0.80, 0.95, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99];
        let hist = run(&mut p, &hz, MTP_EVAL_WINDOW as usize + 8, 4);
        assert!(hist.iter().all(|&d| d == 3), "up-switched without clearing the margin; history {hist:?}");
    }
