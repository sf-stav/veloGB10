//! Unit tests (module `wide_bits_guard_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    use super::*;

    #[test]
    fn the_predicates_match_the_kernels_switches() {
        for b in [3usize, 4, 5] { assert!(wide_trellis_supports_bits(b), "{b}"); }
        for b in [0usize, 1, 2] { assert!(!wide_trellis_supports_bits(b), "{b}"); } // PR#27-D2: 8 now serves
        for b in [6usize, 7] { assert!(wide_trellis_supports_bits(b), "B5: {b}"); }
        // P7-B4 (+leg 2): the pipelined prefill MoE kernels decode 2..7-bit trellises.
        for b in [2usize, 3, 4, 5, 6, 7] { assert!(moe_pf_kernel_supports_bits(b), "{b}"); }
        // P7-B1: the expert set and the dense set are DIFFERENT kernel families — each width is admitted
        // only where its body exists (the asymmetry is the loader's loud-refuse trigger, not a bug).
        // K35: k2 3 / 5 / 7 (K = 1.5 / 2.5 / 3.5) are expert widths (HALF twins of the grouped / wide
        // bodies); the dense family has K = 1.5 only. K8EXP: K = 7 / 8 (k2 14 / 16) are expert widths too —
        // K = 7 on the integer family (a1b7 / xh case 7 / WP20 b7 / pf 7), K = 8 on its own grouped entry
        // (xq_gemm_grouped_b8, every width and every A/B flag) + the wide-tiles prefill case 8; coop stays 2..=6.
        for k2 in [3usize, 4, 5, 6, 7, 8, 10, 12, 14, 16] { assert!(expert_k2_supported(k2), "k2 {k2}"); }
        for k2 in [2usize, 9, 11, 13, 15] { assert!(!expert_k2_supported(k2), "k2 {k2}"); }
        for m in [1usize, 8, 9, 10, 16] { for k in [512usize, 2560, 768] { assert_eq!(grouped_fn(m, k, 8), "xq_gemm_grouped_b8", "m {m} k {k}"); } }
        assert_eq!(grouped_fn(4, 2560, 7), "xq_gemm_grouped_a1b7");
        assert!(wide_tiles_supports_bits(8) && !moe_pf_kernel_supports_bits(8), "K=8 prefill rides the wide tiles");
        assert!(gu_chunk(12, 2560, true, 8).is_none(), "no K=8 WP20 chunk entry");
        for b in [7usize, 8, 17] { assert!(!moe_coop_supports_bits(b), "coop {b}"); }
        for b in 2usize..=6 { assert!(moe_coop_supports_bits(b), "coop {b}"); }
        assert!(src_text!("kernels/exl3_bench").contains("xq_gemm_grouped_b8("), "the K=8 entry exists");
        assert!(crate::exl3_bench::MODULE_FNS.contains(&"xq_gemm_grouped_b8"), "the K=8 entry is loaded");
        for k2 in [3usize, 4, 6, 8, 10, 12, 14, 16] { assert!(quad_k2_supported(k2), "dense k2 {k2}"); }
        for k2 in [2usize, 5, 7, 9, 11, 13, 15] { assert!(!quad_k2_supported(k2), "dense k2 {k2}"); }
        // the width CODE the kernels switch on: integer K as K, fractional K = KA + 0.5 as 16 + KA (never an alias)
        assert_eq!(exl3_kernel_bits(6), 3); assert_eq!(exl3_kernel_bits(8), 4); assert_eq!(exl3_kernel_bits(16), 8);
        assert_eq!(exl3_kernel_bits(3), 17); assert_eq!(exl3_kernel_bits(5), 18); assert_eq!(exl3_kernel_bits(7), 19);
        assert!(exl3_is_frac_bits(17) && exl3_is_frac_bits(19) && !exl3_is_frac_bits(3) && !exl3_is_frac_bits(16) && !exl3_is_frac_bits(20));
        assert_eq!(exl3_width_name(19), "3.5"); assert_eq!(exl3_width_name(18), "2.5"); assert_eq!(exl3_width_name(4), "4");
        // a fractional code never takes an integer-width arm (the alias hazard the code exists for); PFHALF: the pipelined
        // prefill kernel serves it on its OWN HALF arms (xq_moe_pf_entry's EXL3_FRAC_BITS cases), never an integer one
        for code in [17usize, 18, 19] {
            assert!(moe_pf_kernel_supports_bits(code) && !wide_trellis_supports_bits(code) && wide_tiles_supports_bits(code), "{code}");
            assert_eq!(grouped_fn(8, 512, code), "xq_gemm_grouped_xh");
            assert!(gu_chunk(12, 2560, true, code).is_none());
        }
        let t = width_support_text();
        assert!(t.contains("experts serve K = 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8") && t.contains("no expert kernel for K = 1)")
                && t.contains("dense modules serve K = 1.5, 2, 3, 4, 5, 6, 7, 8") && t.contains("no dense kernel for K = 1, 2.5, 3.5"), "{t}");
        for b in [0usize, 1, 8] { assert!(!moe_pf_kernel_supports_bits(b), "{b}"); }
        // the kernel source agrees (B5 widened the wide family to 3..=7; PR#27-D2 added 8; K35 added the fractional
        // HALF cases to the TILES kernel only; the pins below track it)
        let cu = src_text!("kernels/exl3_bench");
        assert!(cu.contains("#define EXL3_FRAC_BITS(KA) (16 + (KA))"), "the width code macro");
        assert!(cu.contains("switch (bits) { XQ_GXH_FRAC(1) XQ_GXH_FRAC(2) XQ_GXH_FRAC(3) default: break; }"), "grouped_xh fractional cases");
        for sw in ["switch (bits) { EXL3_WT_CASE(3) EXL3_WT_CASE(4) EXL3_WT_CASE(5) EXL3_WT_CASE(6) EXL3_WT_CASE(7) EXL3_WT_CASE(8) EXL3_WT_CASE_H(1) EXL3_WT_CASE_H(2) EXL3_WT_CASE_H(3) default: break; }",
                   "switch (bits) { EXL3_WG_CASE(3) EXL3_WG_CASE(4) EXL3_WG_CASE(5) EXL3_WG_CASE(6) EXL3_WG_CASE(7) EXL3_WG_CASE(8) default: break; }",
                   "EXL3_WIDE_CASE(3);\n        EXL3_WIDE_CASE(4);\n        EXL3_WIDE_CASE(5);\n        EXL3_WIDE_CASE(6);\n        EXL3_WIDE_CASE(7);\n        EXL3_WIDE_CASE(8); // PR#27-D2: K=8 dense/wide serve\n        default: break;"] {
            assert!(cu.contains(sw), "the wide kernel's bit switch changed (a 6-bit case?): update wide_trellis_supports_bits\n{sw}");
        }
    }

    #[test]
    fn the_refusal_names_the_width_the_option_and_the_arm() {
        let t = wide_arm_bits_refusal(6, 2048, 2048, 4096, 0, true);
        for needle in ["6-bit module", "2048 rows", "--exl3-recon-min", "reconstruct + cuBLASLt", "garbage"] {
            assert!(t.contains(needle), "{needle:?} missing: {t}");
        }
        // F3 (owner 2026-10-10): the text names the REAL remaining cause (a call site without
        // reconstruct buffers) and blames neither --exl3-recon-min (the dispatch now overrides it
        // for these widths) nor --max-batch (the serve caps lanes at 16) — it may MENTION
        // --max-batch only to say the cap removed it as a cause.
        assert!(!t.contains("taken because --max-batch"), "no --max-batch blame: {t}");
        assert!(t.contains("the serve caps --max-batch at 16 lanes"), "the cap is named as the removed cause: {t}");
        // buffers ARE present here, so the cause must read as the routing-invariant bug branch:
        assert!(t.contains("routing invariant"), "buffers present but unused reads as a bug: {t}");
        // and the no-buffers branch names the only thing that can still reach the refusal:
        let t3 = wide_arm_bits_refusal(6, 40, 2048, 4096, 17, false);
        assert!(t3.contains("no reconstruct buffers"), "the remaining cause is named: {t3}");
    }

    /// F3: the 4.05 pack's 6-bit modules never reach the wide arm when reconstruct buffers exist —
    /// the dispatch forces the reconstruct arm whatever --exl3-recon-min says (owner 2026-10-10:
    /// "either support things properly or gracefully fall back").
    #[test]
    fn wide_bits_forced_to_the_reconstruct_arm() {
        let src: String = src_text!("exl3_forward").lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("\n");
        let f = &src[src.find("fn chain_wide_gemm(").expect("chain_wide_gemm")..];
        let f = &f[..f.find("\n}\n").expect("fn end")];
        let forced = f.find("Some(r) if !wide_trellis_supports_bits(q.bits as usize) => {").expect("the forced-recon arm");
        let option = f.find("rmin > 0 && m >= rmin").expect("the option's arm");
        assert!(forced < option, "the forced arm must be matched BEFORE the option's arm");
        assert!(f.contains("WIDE_FALLBACK_NOTE.set(())"), "the once-per-boot line");
        assert!(f.contains("exl3_recon_forced(q.bits)"), "the /metrics counter");
    }

    /// The guards sit IN FRONT of the launches they protect (a refusal after the launch protects nothing).
    #[test]
    fn the_guards_precede_the_launches() {
        let src: String = src_text!("exl3_forward").lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("\n");
        let src = &src[..src.find("mod wide_bits_guard_tests").unwrap()];
        let f = &src[src.find("fn chain_wide_gemm(").unwrap()..];
        let f = &f[..f.find("\n}\n").unwrap()];
        let (g, l) = (f.find("wide_trellis_supports_bits(q.bits").expect("guard"), f.find("\"exl3_hmma_gemm_wide\"").expect("launch"));
        assert!(g < l, "the bit-width guard must come before the wide launch");
        let m = &src[src.find("let pk = tune::get(&T_PF_MOE_KERNEL").expect("pk")..];
        let (g1, g2, l) = (m.find("moe_pf_kernel_supports_bits(moe.bits)").expect("pf guard"),
                           m.find("wide_tiles_supports_bits(moe.bits)").expect("tiles guard"),
                           m.find("\"xq_moe_pf_count").expect("first launch"));
        assert!(g1 < l && g2 < l, "the MoE guards must come before the first MoE launch");
    }
