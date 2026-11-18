//! Unit tests (module `wp18_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    use super::*;

    /// wp18_plan is the SPEC the walk implements: row groups of `rg` in ascending order, each over
    /// the ascending 8,192-block tile ladder 0..ceil(nb_max / T); score_grid sizes the
    /// xq_wp18_score CTA grid ((ceil(tb / TN), ceil(gn / TM)); (0, 0) only when the group's widest
    /// row has nb == 0 — the merge still runs to emit tail-only lists). The merge's carried-list
    /// contract (a row's tile kt may read the list only after tile kt - 1 wrote it) needs the FULL
    /// ladder per group. (The GPU side is checked by --probe-exl3-binv / --wp18-xcheck on the box.)
    #[test]
    fn wp18_plan_groups_and_tiles_cover() {
        for &(pos0, rows, ratio, rg) in &[
            (8_189usize, 77usize, 4usize, 256usize), // odd tail chunk past the dense limit
            (0, 2048, 4, 256),                       // whole-chunk sparse rows, default groups
            (126_976, 2048, 4, 2048),                // 128K window end, one wide group (4 tiles)
            (3, 5, 8, 32),                           // rg floor, ratio > rows
            (0, 1, 4, 2048),                         // single row, nb == 0 (merge-only step)
        ] {
            let v = wp18_plan(pos0, rows, ratio, rg);
            let (mut next, mut i) = (0usize, 0usize);
            while i < v.len() {
                let g0 = v[i].g0;
                let gn = rg.min(rows - g0);
                assert_eq!(g0, next, "pos0={pos0} rows={rows} rg={rg}: group starts at {g0}, want {next}");
                let pos_first = pos0 + g0;
                let nb_max = (pos_first + gn) / ratio;
                let nt = nb_max.div_ceil(WP18_T).max(1);
                let kts: Vec<usize> = v[i..i + nt].iter().map(|s| s.kt).collect();
                assert_eq!(kts, (0..nt).collect::<Vec<_>>(),
                    "pos0={pos0} rows={rows} rg={rg}: tiles of group {g0} not the ascending ladder 0..{nt}");
                for k in 0..nt {
                    let s = &v[i + k];
                    assert_eq!((s.g0, s.gn, s.pos_first), (g0, gn, pos_first));
                    let tb = nb_max.saturating_sub(k * WP18_T).min(WP18_T);
                    let gx = tb.div_ceil(WP18_TN) as u32;
                    let gy = if gx > 0 { gn.div_ceil(WP18_TM) as u32 } else { 0 };
                    assert_eq!(s.score_grid, (gx, gy), "pos0={pos0} rows={rows} rg={rg} kt={k}");
                    assert!(gx > 0 || nb_max == 0, "pos0={pos0} rows={rows} rg={rg} kt={k}: gx == 0 with visible blocks");
                }
                i += nt;
                next = g0 + gn;
            }
            assert_eq!(next, rows, "pos0={pos0} rows={rows} rg={rg}: groups cover {next} rows");
        }
    }
