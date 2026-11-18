//! Inline unit tests, moved out of `model.rs` verbatim (module `t0c_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::{a1_smem, DEFAULT_SMEM_CAP};

    /// T0-c: at ks == 1 the window IS the whole of k, so the staged request must be byte-identical
    /// to the pre-T0-c number. This is the property that makes the change safe on the ks == 1 paths
    /// (every single-pass `nv_gemm` / `nv_gemm_grouped_xh` launch, which are the majority).
    #[test]
    fn ks1_is_byte_identical_to_the_pre_t0c_arithmetic() {
        for &k in &[2048usize, 4096, 8192, 16384] {
            for m in 1..=16usize {
                let got = a1_smem(m, k, 1);
                let pre_t0c = m * (k + 8) * 2;
                if pre_t0c <= DEFAULT_SMEM_CAP {
                    assert_eq!(got as usize, pre_t0c, "ks=1 m={m} k={k} must match pre-T0-c");
                } else {
                    assert_eq!(got, 0, "ks=1 m={m} k={k} over the cap must fall back, not over-allocate");
                }
            }
        }
    }

    /// T0-c: the whole point — shapes the 48KB gate rejected now come back, and they come back
    /// because the request shrank by ks, not because the cap was raised.
    ///
    /// The exact identity, so a drift in either direction fails here:
    ///   pre-T0-c  = m*(k + 8)*2          (the whole of K)
    ///   T0-c      = m*(k/ks + 8)*2       (this CTA's window)
    /// so T0-c == pre/ks + 16m*(1 - 1/ks) — the `+8` pad is PER WINDOW and scales with m, which is
    /// why the request shrinks by slightly less than ks. It is exactly `pre/ks` only in the limit.
    #[test]
    fn windowing_readmits_the_shapes_the_48kb_gate_rejected() {
        // mlp.down: k = 16384. Pre-T0-c at m = 8 was 8 * (16384 + 8) * 2 = 262272 B -> never staged.
        let pre_t0c_m8 = 8usize * (16384 + 8) * 2;
        assert!(pre_t0c_m8 > DEFAULT_SMEM_CAP, "precondition: m=8/k=16384 really was rejected");
        for ks in [8usize, 16, 32] {
            let got = a1_smem(8, 16384, ks) as usize;
            assert!(got > 0, "m=8 k=16384 ks={ks} should be stageable now");
            assert!(got <= DEFAULT_SMEM_CAP);
            assert_eq!(got, pre_t0c_m8 / ks + 16 * 8 * (ks - 1) / ks,
                       "the windowed request must equal pre/ks + 16m(1-1/ks) (m=8 k=16384 ks={ks})");
        }
    }

    /// T0-c: the arithmetic must equal the kernel's `SK` exactly. The kernel computes
    /// `SK = ((kb1 - kb0) << 4) + 8` halves and holds `m` rows, i.e. `m * SK * 2` bytes. This is the
    /// host/device twin check the doc comment demands (the host and kernel must never drift).
    #[test]
    fn host_bytes_match_the_kernel_sk_formula() {
        for ks in [1usize, 2, 4, 8, 16] {
            for &k in &[4096usize, 16384] {
                let m = 8usize;
                let kb0 = (k / 16) / ks;
                let kb1 = kb0 + (k / 16) / ks;          // the kernel's chunk
                let kernel_sk: usize = ((kb1 - kb0) << 4) + 8;
                let kernel_bytes = m * kernel_sk * 2;
                let host = a1_smem(m, k, ks) as usize;
                if host > 0 {
                    assert_eq!(host, kernel_bytes, "host/kernel smem must agree (m={m} k={k} ks={ks})");
                }
            }
        }
    }

    /// Guards the invariants that make the window legal: ks must divide the k-tile count, and the
    /// window must stay a whole multiple of 128 columns so the fused `suh` path's nv_had128 never
    /// straddles a butterfly group.
    #[test]
    fn rejects_shapes_the_window_cannot_serve() {
        assert_eq!(a1_smem(0, 4096, 1), 0, "no rows");
        assert_eq!(a1_smem(17, 4096, 1), 0, "over MAX_ROWS");
        assert_eq!(a1_smem(8, 4096, 0), 0, "ks == 0");
        assert_eq!(a1_smem(8, 4104, 1), 0, "k not a multiple of 16");
        assert_eq!(a1_smem(8, 4096, 3), 0, "ks does not divide the k-tiles");
        // k = 4096 (256 tiles): ks = 16 leaves 16 tiles = 256 columns = 2 whole 128-blocks -> LEGAL;
        // ks = 64 leaves 4 tiles = 64 columns, straddling one -> refused.
        assert!(a1_smem(8, 4096, 16) > 0, "a 2-block window is legal");
        assert_eq!(a1_smem(8, 4096, 64), 0, "window narrower than 128 columns");
    }

    /// T0-c engagement: at the REAL shapes, the shipped dispatcher (`chain_plan`) must actually
    /// route the staged path. This is the "identical hashes is not a liveness receipt" guard — the
    /// smem arithmetic can be perfect while nothing ever asks for it.
    ///
    /// Every projection in the campaign (p90 ITL -25%) rests on this: if chain_plan hands back
    /// `a1 == false` or `ks == 1` for these shapes, T0-c never engages in the serving path.
    #[test]
    fn t0c_engages_on_the_real_shapes_through_the_real_dispatcher() {
        // (name, k, n) from src/naive/config.rs tests: hidden 4096, ffn 16384, moe_inter 2048,
        // vocab 152576, index_head_dim 128, head_dim = hidden/heads.
        let shapes: &[(&str, usize, usize)] = &[
            ("mlp.down", 16384, 4096),
            ("mlp.gate_up", 4096, 4096),
            ("q_proj", 4096, 4096),
            ("o_proj", 4096, 4096),
            ("lm_head", 4096, 152576),
        ];
        let mut engaged = 0;
        for &(name, k, n) in shapes {
            let (ks, a1) = super::super::gpu::chain_plan(k, n);
            let smem = a1_smem(8, k, ks);
            // The staged body can only be used when the plan says so AND the window fits.
            let staged = a1 && smem > 0;
            println!("T0cEngage {name:<12} k={k:<6} n={n:<7} ks={ks} a1={a1} smem={smem} staged={staged}");
            if staged { engaged += 1; }
            // Whatever it decides, the two must agree — a plan that asks for a staged body with no
            // smem (or vice versa) would be the silent-dispatch bug this whole item is about.
            if a1 && ks > 1 { assert!(smem > 0, "{name}: staged split-K with no smem is unreachable"); }
        }
        assert!(engaged > 0, "T0-c engaged on NO real shape — the change would be dead code");
    }
