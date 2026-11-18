//! Unit tests (module `wp20_b5_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    use super::*;

    #[test]
    fn k5_expert_selection_picks_the_b5_entries() {
        // A-once fits in the 48 KiB cap: the grouped a1b5 entry serves the a1 rung.
        assert!(a1_smem(16, 512) > 0);
        assert_eq!(grouped_fn(16, 512, 5), "xq_gemm_grouped_a1b5");
        // A-once exceeds the cap: grouped falls back to xh and the gu path takes the K-chunk.
        assert_eq!(a1_smem(16, 2048), 0);
        assert_eq!(grouped_fn(16, 2048, 5), "xq_gemm_grouped_xh");
        assert_eq!(gu_chunk(16, 2048, false, 5).map(|(n, _)| n), Some("xq_moe_gu_epi_b5_c2"));
        assert_eq!(gu_chunk(16, 2048, true, 5).map(|(n, _)| n), Some("xq_moe_gu_epi_b5_c2"));
        // b4 keeps requiring the diet twin; b5 is SHFL either way (same answer, diet or not).
        assert_eq!(gu_chunk(16, 2048, false, 4), None);
        assert_eq!(gu_chunk(16, 2048, true, 4).map(|(n, _)| n), Some("xq_moe_gu_epi_b4_c2"));
    }

    #[test]
    fn the_cu_file_instantiates_the_b5_set() {
        let cu = src_text!("kernels/exl3_bench");
        for n in ["WP20_GU_ENTRY_B5(xq_moe_gu_epi_b5, 1)",
                  "WP20_GU_ENTRY_B5(xq_moe_gu_epi_b5_c2, 2)",
                  "WP20_GU_ENTRY_B5(xq_moe_gu_epi_b5_c4, 4)",
                  "xq_moe_dn_epi_b5_k10",
                  "xq_gemm_grouped_a1b5",
                  // P7 B3 leg 2: the K = 2/6/7 twins
                  "WP20_GU_ENTRY_BW(xq_moe_gu_epi_b6, 1, 6)",
                  "WP20_GU_ENTRY_BW(xq_moe_gu_epi_b7_c2, 2, 7)",
                  "xq_moe_dn_epi_b2_k10",
                  "xq_moe_dn_epi_b6_k10",
                  "xq_moe_dn_epi_b7_k10",
                  "xq_gemm_grouped_a1b2",
                  "xq_gemm_grouped_a1b6",
                  "xq_gemm_grouped_a1b7"] {
            assert!(cu.contains(n), "{n} missing from exl3_bench.cu");
        }
    }

    #[test]
    fn the_b5_arm_is_top10_only_and_admitted_by_the_gate() {
        let src: String = src_text!("exl3_forward")
            .lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("\n");
        assert!(src.contains("matches!(bits, 2 | 3 | 4 | 5 | 6 | 7)"), "the WP20 gate must admit K = 2..=7");
        let i = src.find("Some(\"xq_moe_dn_epi_b5_k10\")").expect("the b5 down arm");
        let seg = &src[i.saturating_sub(160)..i + 80];
        assert!(seg.contains("topk == 10"), "the b5 dn entry is top-10 only: {seg}");
        assert!(seg.contains("xq_moe_gu_epi_b5"), "the b5 gu entry sits in the same arm: {seg}");
    }
