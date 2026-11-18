//! Unit tests (module `rel_fixb_router_selection`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    use super::*;

    /// The pre-R16 (v0.7.2) decision, copied from
    /// `git show baseline/cf-2026-10-03:src/exl3_forward.rs`:
    ///   let fused = <m-free gates> && m <= 8;
    ///   let fold  = fused && wp10_fold_on() && ne <= 512 && m * topk <= 128;
    ///   ... fname = ep_fold ? "xq_router_fold_c_ep" : coal ? "xq_router_fold_c"
    ///               : w4r ? "xq_router_fold_w4" : remap ? "xq_router_fold" : "_m0";
    /// with ep_fold = coal && snap.is_none() && ep && route_fold_on().
    fn baseline_router_entry(m: usize, ep: bool, coal: bool, route_fold: bool, fused_ok: bool,
                             fold_ok: bool, topk_cap: bool, w4r: bool, gemv_remap: bool,
                             xcheck: bool) -> RouterEntry {
        let fused = fused_ok && m <= 8;
        let fold = fused && fold_ok && topk_cap;
        if !fold {
            return if fused { RouterEntry::FusedPair } else { RouterEntry::Legacy };
        }
        let ep_fold = coal && !xcheck && ep && route_fold;
        if ep_fold { RouterEntry::FoldEp }
        else if coal { RouterEntry::FoldCoal }
        else if w4r { RouterEntry::FoldW4 }
        else if gemv_remap { RouterEntry::Fold }
        else { RouterEntry::FoldM0 }
    }

    /// The PRE-FIX (master, REL v0.7.3) dispatch copied from the same file — the K1 bug
    /// this session fixes: r16 without coal/EP and ep_fold without m <= 8. Kept here so
    /// `router_selection_pre_fix_predicate_fails` proves the table test can detect the
    /// regression class (if this copy stops failing the asserts, master changed — re-copy).
    fn pre_fix_router_entry(m: usize, ep: bool, coal: bool, route_fold: bool, r16_on: bool,
                            fused_ok: bool, fold_ok: bool, topk_cap: bool, w4r: bool,
                            gemv_remap: bool, xcheck: bool) -> RouterEntry {
        let r16 = m > 8 && m <= 16 && r16_on; // no coal, no !ep  <-- K1
        let fold = fused_ok && fold_ok && topk_cap && (m <= 8 || r16);
        if !fold {
            return if fused_ok && m <= 8 { RouterEntry::FusedPair } else { RouterEntry::Legacy };
        }
        let ep_fold = coal && !xcheck && ep && route_fold; // no m <= 8  <-- K1
        if ep_fold { RouterEntry::FoldEp }
        else if coal && r16 { RouterEntry::FoldR16 }
        else if coal { RouterEntry::FoldCoal }
        else if w4r { RouterEntry::FoldW4 }
        else if gemv_remap { RouterEntry::Fold }
        else { RouterEntry::FoldM0 }
    }

    /// RouterEntry restricted to the baseline's variant set (no FoldR16).
    fn baseline_view(e: RouterEntry) -> Option<RouterEntry> {
        match e {
            RouterEntry::FoldR16 => None,
            other => Some(other),
        }
    }

    #[test]
    fn router_selection_table_exhaustive() {
        let mut combos = 0usize;
        for m in 1..=16usize {
            for &ep in &[false, true] {
            for &coal in &[false, true] {
            for &route_fold in &[false, true] {
            for &r16_hw in &[false, true] {
            for &fused_ok in &[false, true] {
            for &fold_ok in &[false, true] {
            for &topk_cap in &[false, true] {
            for &w4r in &[false, true] {
            for &gemv_remap in &[false, true] {
            for &xcheck in &[false, true] {
                combos += 1;
                let e = router_entry(m, ep, coal, route_fold, r16_hw, fused_ok, fold_ok,
                                     topk_cap, w4r, gemv_remap, xcheck);
                let ctx = format!("m={} ep={} coal={} rf={} r16={} fu={} fo={} tk={} w4r={} gr={} xc={} -> {:?}",
                                  m, ep, coal, route_fold, r16_hw, fused_ok, fold_ok,
                                  topk_cap, w4r, gemv_remap, xcheck, e);
                // (a) no M <= 8 entry ever serves m > 8.
                if m > 8 {
                    assert!(
                        matches!(e, RouterEntry::FoldR16 | RouterEntry::Legacy),
                        "K1 invariant violated: {}", ctx
                    );
                }
                // (b) m <= 8 selection == the pre-R16 baseline decision.
                if m <= 8 {
                    let b = baseline_router_entry(m, ep, coal, route_fold, fused_ok, fold_ok,
                                                  topk_cap, w4r, gemv_remap, xcheck);
                    assert_eq!(baseline_view(e), Some(b), "m<=8 differs from baseline: {}", ctx);
                }
                // (c) with r16 off, the selection equals the baseline for EVERY m.
                if !r16_hw {
                    let b = baseline_router_entry(m, ep, coal, route_fold, fused_ok, fold_ok,
                                                  topk_cap, w4r, gemv_remap, xcheck);
                    assert_eq!(baseline_view(e), Some(b), "r16-off differs from baseline: {}", ctx);
                }
                // (d) at 9..=16 rows a non-EP coal rank takes the r16 fold; an EP rank
                // (or any r16-ineligible width) keeps the legacy chain.
                if (9..=16).contains(&m) && fused_ok && fold_ok && topk_cap {
                    if r16_hw && coal && !ep {
                        assert_eq!(e, RouterEntry::FoldR16, "{}", ctx);
                    }
                    if ep {
                        assert_eq!(e, RouterEntry::Legacy, "{}", ctx);
                    }
                }
            }}}}}}}}}}
        }
        assert_eq!(combos, 16 * 1024);
    }

    #[test]
    fn router_selection_pre_fix_predicate_fails() {
        // Proof the table test can detect the K1 class: run the PRE-FIX (master)
        // predicate through invariant (a)'s check and confirm it VIOLATES it.
        let mut violations = 0usize;
        for m in 9..=16usize {
            for &ep in &[false, true] {
            for &coal in &[false, true] {
            for &route_fold in &[false, true] {
            for &fused_ok in &[false, true] {
            for &fold_ok in &[false, true] {
            for &topk_cap in &[false, true] {
            for &w4r in &[false, true] {
            for &gemv_remap in &[false, true] {
            for &xcheck in &[false, true] {
                let e = pre_fix_router_entry(m, ep, coal, route_fold, true, fused_ok, fold_ok,
                                             topk_cap, w4r, gemv_remap, xcheck);
                if !matches!(e, RouterEntry::FoldR16 | RouterEntry::Legacy) {
                    violations += 1; // an M <= 8 body selected above 8 rows
                }
            }}}}}}}}}
        }
        assert!(violations > 0, "pre-fix predicate no longer violates the invariant —                                  master's dispatch changed; re-copy pre_fix_router_entry");
        // The two review counterexamples, explicitly:
        // EP rank (TP >= 2, route-fold on, coal): xq_router_fold_c_ep (M <= 8) at m = 12.
        assert_eq!(pre_fix_router_entry(12, true, true, true, true, true, true, true, false, true, false),
                   RouterEntry::FoldEp);
        // --router-coal 0: falls to xq_router_fold / _m0 (M <= 8) at m = 12.
        assert_eq!(pre_fix_router_entry(12, false, false, true, true, true, true, true, false, true, false),
                   RouterEntry::Fold);
        // ...and the FIXED selection for both is the legacy chain.
        assert_eq!(router_entry(12, true, true, true, true, true, true, true, false, true, false),
                   RouterEntry::Legacy);
        assert_eq!(router_entry(12, false, false, true, true, true, true, true, false, true, false),
                   RouterEntry::Legacy);
        // The r16 fold is still reachable on a non-EP coal rank at 9..16 rows.
        assert_eq!(router_entry(12, false, true, true, true, true, true, true, false, true, false),
                   RouterEntry::FoldR16);
    }
