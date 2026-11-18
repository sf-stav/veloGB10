//! Unit tests (module `reprime_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    use super::*;

    /// WP22 rung 1: slabs are consecutive, <= 16 rows, start at 0 and cover [0, m) exactly — so
    /// every fc row goes through exactly one split-K call of <= 16 rows.
    #[test]
    fn wp22r1_slabs_cover() {
        for m in 1..=64usize {
            let s = wp22r1_slabs(m);
            let mut next = 0usize;
            for &(r0, ms) in &s {
                assert_eq!(r0, next, "m={m}: slab starts at {r0}, expected {next}");
                assert!(ms >= 1 && ms <= 16, "m={m}: slab of {ms} rows");
                next = r0 + ms;
            }
            assert_eq!(next, m, "m={m}: slabs cover {next} rows");
            assert_eq!(s.len(), m.div_ceil(16));
        }
    }

    /// Host mirror of xq_reprime_stage's row rule (tap row, token, position) — the SPEC the
    /// kernel implements (the GPU side is checked by --reprime-xcheck on the box).
    fn stage_row(a: usize, r: usize, p: i32, b: i32, emit: &[i32]) -> (usize, i32, i32) {
        let src = if r < a { r } else if a > 0 { a - 1 } else { 0 };
        let q: i32 = if r < a { r as i32 } else { a as i32 - 1 };
        let tok = if q >= 0 { emit[q as usize] } else { b };
        (src, tok, p + 1 + q)
    }

    /// Rows r < a are the eager re-prime's rows exactly; every masked row sits at position p+a
    /// with the token entering there (a >= 1: the last accepted draft; a = 0: the bonus b at p),
    /// so its pooled-block refresh targets block(p+a): a block an accepted row refreshes (a >= 1)
    /// or block(p) (a = 0), and every masked position lies inside the verify window [p, p+k].
    #[test]
    fn reprime_stage_row_rule() {
        let (p, b) = (1000i32, 77i32);
        for rows in 1..=7usize {
            let emit: Vec<i32> = (0..rows).map(|i| 500 + i as i32).collect();
            for a in 0..=rows {
                for r in 0..rows {
                    let (src, tok, pos) = stage_row(a, r, p, b, &emit);
                    if r < a {
                        assert_eq!((src, tok, pos), (r, emit[r], p + 1 + r as i32), "valid row a={a} r={r}");
                    } else {
                        assert_eq!(pos, p + a as i32, "masked row a={a} r={r}: position");
                        assert_eq!(tok, if a > 0 { emit[a - 1] } else { b }, "masked row a={a} r={r}: token");
                        assert!(src < rows.max(1), "masked row a={a} r={r}: tap row {src}");
                    }
                    assert!(pos >= p && pos <= p + rows as i32, "row a={a} r={r}: position {pos} outside [p, p+R]");
                }
            }
        }
    }
