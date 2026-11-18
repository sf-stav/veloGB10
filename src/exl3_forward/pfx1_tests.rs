//! Unit tests (module `pfx1_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    use super::*;

    // Mirrors of kernels/exl3_bench.cu's XQ_R8_* (xq_gemm_f16_f32_rows8).
    const R8_ROWS: usize = 8;
    const R8_COLS: usize = 64;
    const R8_KT: usize = 256;
    const R8_PF: usize = 4;
    const R8_G: usize = 8 * R8_ROWS + 8;

    /// Host mirror of rows8's x staging: thread tid stores uint4 j (row tid&7, 8-k group
    /// (tid>>3)+8j, halves e) at xs[group*G + e*8 + row]; the compute loop reads k's 8 row values
    /// at xs[(k>>3)*G + (k&7)*8 .. +8]. Every (k, row) of a tile is stored exactly once, where it
    /// is read, in bounds, 16-B aligned for the float4 reads, and each STS (fixed j, e) of a
    /// warp hits 32 distinct banks.
    #[test]
    fn rows8_smem_map() {
        let xj = (R8_KT / 8) * R8_ROWS / R8_COLS;
        assert_eq!(xj, 4);
        let size = (R8_KT / 8) * R8_G;
        let mut owner = vec![None::<(usize, usize)>; size];
        for tid in 0..R8_COLS {
            let (xr, xq0) = (tid & 7, tid >> 3);
            for j in 0..xj {
                for e in 0..8 {
                    let addr = (xq0 + 8 * j) * R8_G + e * R8_ROWS + xr;
                    let k = (xq0 + 8 * j) * 8 + e;
                    assert!(addr < size);
                    assert!(owner[addr].is_none(), "slot {addr} stored twice");
                    owner[addr] = Some((k, xr));
                    // the read side's address for (k, row xr)
                    assert_eq!((k >> 3) * R8_G + (k & 7) * R8_ROWS + xr, addr);
                }
            }
        }
        for k in 0..R8_KT {
            let base = (k >> 3) * R8_G + (k & 7) * R8_ROWS;
            assert_eq!(base % 4, 0, "float4 read of k={k} misaligned");
            for r in 0..R8_ROWS { assert_eq!(owner[base + r], Some((k, r)), "k={k} r={r}"); }
        }
        for warp in 0..R8_COLS / 32 {
            for j in 0..xj {
                for e in 0..8 {
                    let mut banks: Vec<usize> = (0..32).map(|lane| {
                        let tid = warp * 32 + lane;
                        (((tid >> 3) + 8 * j) * R8_G + e * R8_ROWS + (tid & 7)) % 32
                    }).collect();
                    banks.sort_unstable();
                    banks.dedup();
                    assert_eq!(banks.len(), 32, "warp {warp} j {j} e {e}: bank conflict");
                }
            }
        }
    }

    /// Host mirror of rows8's weight pipeline: wA holds steps 0..PF-1 (prologue); each pass at
    /// gs loads wB <- gs+PF.., consumes wA (gs..gs+PF-1), loads wA <- gs+2PF.., consumes wB
    /// (clamped to the last step). Every 16-k step is consumed exactly once, ascending (the
    /// ascending-k chain), and the step each xs group index names is the step consumed, for
    /// every K the host dispatch admits (K % 256 == 0). The L2 run-ahead prefetches every line
    /// from step 2PF on exactly once, never past the row.
    #[test]
    fn rows8_w_ring() {
        const L2A: usize = 16;
        let wload = |s: usize, nsteps: usize| -> Vec<usize> { (0..R8_PF).map(|p| (s + p).min(nsteps - 1)).collect() };
        for k in (256..=8192usize).step_by(256) {
            let nsteps = k / 16;
            let mut wa = wload(0, nsteps);
            let mut consumed = Vec::new();
            let mut lines = Vec::new();
            for s in (2 * R8_PF..2 * R8_PF + L2A).step_by(4) { if s < nsteps { lines.push(s); } }
            let mut gs = 0usize;
            for k0 in (0..k).step_by(R8_KT) {
                for s0 in (0..R8_KT / 16).step_by(2 * R8_PF) {
                    let sp = gs + 2 * R8_PF + L2A;
                    if sp < nsteps { lines.push(sp); }
                    if sp + 4 < nsteps { lines.push(sp + 4); }
                    let wb = wload(gs + R8_PF, nsteps);
                    for p in 0..R8_PF {
                        assert_eq!(k0 / 16 + s0 + p, wa[p], "K={k}: xs group vs w step");
                        consumed.push(wa[p]);
                    }
                    wa = wload(gs + 2 * R8_PF, nsteps);
                    for p in 0..R8_PF {
                        assert_eq!(k0 / 16 + s0 + R8_PF + p, wb[p], "K={k}: xs group vs w step");
                        consumed.push(wb[p]);
                    }
                    gs += 2 * R8_PF;
                }
            }
            assert_eq!(consumed, (0..nsteps).collect::<Vec<_>>(), "K={k}");
            let want: Vec<usize> = (2 * R8_PF..nsteps).step_by(4).collect();
            assert_eq!(lines, want, "K={k}: L2 prefetch lines");
        }
    }

    /// (a) the MT rule (no --exl3-wide-mt / --pfx1-off in the test env): B3's cells.
    #[test]
    fn pfx1_mt_rule() {
        if crate::opts::var(crate::opt!("exl3-wide-mt")).is_ok() || crate::opts::var(crate::opt!("pfx1-off")).is_ok() { return; }
        for (c, mt) in [(1usize, 1usize), (14, 1), (26, 1), (72, 1), (112, 1), (113, 2), (160, 2),
                        (383, 2), (384, 4), (512, 4), (2048, 4), (4096, 4)] {
            assert_eq!(pfx1_moe_mt(c), mt, "c={c}");
        }
    }

    /// (c) chain_yraw = max(c*12288, vocab) holds every prefill chain: m = c at N <= 12288,
    /// the head fc chains at 4c x 2560, and the m = 1 lm_head at N = vocab.
    #[test]
    fn pfx1_chain_yraw_cover() {
        let v = 248_320usize;
        for c in 1..=4096usize {
            let cap = (c * PF_CHAIN_NMAX).max(v);
            assert!(c * PF_CHAIN_NMAX <= cap && 4 * c * 2560 <= cap && v <= cap, "c={c}");
        }
        // the diet at the served C = 2048: 1.017 GB -> 50.3 MB
        assert!(2048 * v * 2 - 2048 * PF_CHAIN_NMAX * 2 > 960_000_000);
    }

    /// (f) width 16 joins the warm-up whenever the chunk is wider; widths never exceed C.
    #[test]
    fn pfx1_warmup_widths() {
        if crate::opts::var(crate::opt!("pfx1-off")).is_ok() { return; }
        assert_eq!(FwdModel::prefill_warmup_widths(2048), vec![2048, 512, 160, 32, 16]);
        assert_eq!(FwdModel::prefill_warmup_widths(100), vec![100, 32, 16]);
        assert_eq!(FwdModel::prefill_warmup_widths(16), vec![16]);
    }

    /// TP-4S2: the ragged split widths that join the warm-up where SP is live. The served C = 2048 list gets
    /// exactly [2018, 1021, 299, 131]; short chunks (no split range) get none; for EVERY chunk width the
    /// extras are few, ragged (not a multiple of 4), inside the split range, distinct from the base list and
    /// from each other, and cost at most ~2 base-width chunks of rows. The row splits they produce are
    /// uneven at world 4 (the served 2018 -> 505/505/505/503) and odd at world 2 where the residue is odd.
    #[test]
    fn tp4s2_sp_warmup_extra() {
        let base = FwdModel::prefill_warmup_widths(2048);
        assert_eq!(FwdModel::sp_warmup_extra(&base), vec![2018, 1021, 299, 131]);
        assert_eq!(FwdModel::sp_warmup_extra(&FwdModel::prefill_warmup_widths(100)), Vec::<usize>::new());
        assert_eq!(FwdModel::sp_warmup_extra(&[]), Vec::<usize>::new());
        // a mid width (C = 256) keeps the ones that fit the split range
        assert_eq!(FwdModel::sp_warmup_extra(&FwdModel::prefill_warmup_widths(256)), vec![226, 131]);
        for c in 1..=xtp::SP_MAX_ROWS {
            let base = FwdModel::prefill_warmup_widths(c);
            let ex = FwdModel::sp_warmup_extra(&base);
            assert!(ex.len() <= 4, "c={c}: {ex:?}");
            assert!(ex.iter().sum::<usize>() <= 2 * c, "c={c}: boot cost {ex:?}");
            for (i, &w) in ex.iter().enumerate() {
                assert!((xtp::SP_MIN_ROWS..=c).contains(&w) && w <= xtp::SP_MAX_ROWS, "c={c}: {w} outside the split range");
                assert!(w % 4 != 0, "c={c}: {w} is a clean multiple of 4");
                assert!(!base.contains(&w) && !ex[..i].contains(&w), "c={c}: {w} duplicated");
            }
        }
        // the split of the served-class width is uneven at world 4 and the odd widths are odd halves at world 2
        let sp = xtp::SpRows::new(2018, 3, 4);
        assert_eq!((sp.blocks[0], sp.blocks[3]), ((0, 505), (1515, 2018)));
        let (h0, h1) = (xtp::SpRows::new(299, 0, 2).own, xtp::SpRows::new(299, 1, 2).own);
        assert_eq!((h0.1 - h0.0, h1.1 - h1.0), (150, 149));
    }
