//! Unit tests (module `tp4g_dv3_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    use super::*;

    /// `#define NAME value` of the kernel source, parsed (the host mirrors must equal them).
    fn kdef(name: &str) -> usize {
        let src = src_text!("kernels/exl3_bench");
        let key = format!("#define {name} ");
        let line = src.lines().find(|l| l.starts_with(&key))
            .unwrap_or_else(|| panic!("#define {name} missing"));
        line[key.len()..].split_whitespace().next().unwrap().parse().unwrap()
    }

    #[test]
    fn dv3_geometry_mirrors_kernel_defines() {
        assert_eq!(kdef("XQ_DV3_TS"), DV3_TS);
        assert_eq!(kdef("XQ_DV3_G"), DV3_G);
        assert_eq!(kdef("XQ_DV3_G6"), DV3_G6);
        assert_eq!(kdef("XQ_DV3_CH"), 64);
        assert_eq!(DV3_TS / 64, DV3_NCH);
    }

    #[test]
    fn dv3_group_gate_truth_table() {
        // G = 12: TP=1 (24/2), TP=2 (12/1), and a wider multiple
        for (nh, nkv) in [(24, 2), (12, 1), (36, 3)] { assert_eq!(dv3_group(nh, nkv), Some(12), "{nh}/{nkv}"); }
        // G = 6: TP=4 (6/1), and the 12/2 shape
        for (nh, nkv) in [(6, 1), (12, 2), (24, 4), (18, 3)] { assert_eq!(dv3_group(nh, nkv), Some(6), "{nh}/{nkv}"); }
        // everything else stays on v2
        for (nh, nkv) in [(8, 1), (16, 2), (24, 1), (20, 2), (10, 1), (7, 1), (9, 1), (3, 1), (1, 1), (0, 2), (12, 0), (0, 0), (13, 2), (5, 1)] {
            assert_eq!(dv3_group(nh, nkv), None, "{nh}/{nkv}");
        }
    }

    #[test]
    fn dv3_plan_names_blocks_and_coverage() {
        // G = 12 is the pre-TP-4G plan, byte for byte
        assert_eq!(dv3_plan(12, 1), ("xq_attn_dense_dots", "xq_attn_dense_acc1", 16, 192));
        assert_eq!(dv3_plan(12, 2), ("xq_attn_dense_dots", "xq_attn_dense_acc4", 64, 192));
        assert_eq!(dv3_plan(12, 16), ("xq_attn_dense_dots", "xq_attn_dense_acc4", 64, 192));
        // G = 6: the _g6 twins, 16 * 6 = 96 threads
        assert_eq!(dv3_plan(6, 1), ("xq_attn_dense_dots_g6", "xq_attn_dense_acc1_g6", 16, 96));
        assert_eq!(dv3_plan(6, 4), ("xq_attn_dense_dots_g6", "xq_attn_dense_acc4_g6", 64, 96));
        for g in [6usize, 12] {
            for m in [1usize, 2, 3, 4, 8, 16] {
                let (_, acc, sd, block) = dv3_plan(g, m);
                // one thread per (head, dim-group): g heads x 16 dgl; the grid's x covers hd = 256
                assert_eq!(block as usize, g * 16);
                assert!(block <= 1024 && block % 32 == 0, "block {block}");
                assert_eq!(256 / sd * sd, 256);
                // the W4S twin always exists and keeps the group suffix
                let tw = dv3_w4_twin(acc);
                assert_ne!(tw, acc);
                assert_eq!(tw.ends_with("_g6"), g == 6, "{tw}");
            }
        }
    }

    #[test]
    fn dv3_every_planned_kernel_is_declared_and_in_the_manifest() {
        let src = src_text!("kernels/exl3_bench");
        for g in [6usize, 12] {
            for m in [1usize, 4] {
                let (dots, acc, _, _) = dv3_plan(g, m);
                for name in [dots, acc, dv3_w4_twin(acc)] {
                    assert!(src.contains(&format!("\n{name}(")), "kernel {name} not defined in exl3_bench.cu");
                    assert!(crate::exl3_bench::MODULE_FNS.contains(&name), "kernel {name} missing from MODULE_FNS");
                }
            }
        }
    }

    /// Static shared memory per CTA as the kernel source declares it (+ the 16 B ptxas reserves);
    /// the quoted numbers are the `-Xptxas -v` smem of the compiled kernels (TP-4G report).
    fn dots_smem(g: usize) -> usize { g * 32 * 8 * 4 + 8 * 32 * 33 * 4 + 8 * g * 4 + 16 }
    fn acc_smem(g: usize, dpt: usize) -> usize { 2 * 64 * (16 * dpt) * 4 + 2 * 64 * 4 * 4 + g * (64 + 4) * 4 + 16 }
    fn w4a_smem(g: usize, dpt: usize) -> usize { 5 * 32 * (16 * dpt) * 4 + 5 * 32 * 4 * 4 + g * (32 + 4) * 4 + 16 }

    #[test]
    fn dv3_static_smem_arithmetic() {
        const STATIC_CAP: usize = 48 * 1024;
        // G = 12 must equal the pre-TP-4G ptxas numbers exactly
        assert_eq!((dots_smem(12), acc_smem(12, 4), acc_smem(12, 1)), (46480, 38096, 13520));
        assert_eq!((w4a_smem(12, 4), w4a_smem(12, 1)), (45264, 14544));
        // G = 6 (measured by ptxas -v)
        assert_eq!((dots_smem(6), acc_smem(6, 4), acc_smem(6, 1)), (40144, 36464, 11888));
        assert_eq!((w4a_smem(6, 4), w4a_smem(6, 1)), (44400, 13680));
        for g in [6usize, 12] {
            for v in [dots_smem(g), acc_smem(g, 4), acc_smem(g, 1), w4a_smem(g, 4), w4a_smem(g, 1)] {
                assert!(v <= STATIC_CAP, "static smem {v} over 48 KB");
            }
            // 2 CTAs/SM of the dots kernel (launch bounds 256,2) stay inside the 99 KB opt-in cap
            assert!(2 * dots_smem(g) <= 101376);
        }
    }

    /// CPU model of the index arithmetic the G-templated kernels use (the part TP-4G changed):
    /// dots — per chunk a warp walks ceil(G/4) head groups; lane L finishes dot (position L>>2,
    /// head hg*4 + (L&3)), stored / maxed only when that head is < G (PARTIAL guard, G % 4 != 0);
    /// acc — thread tid owns head tid>>4 of the kv group, dim-group tid&15, block = 16 * G.
    /// Every (position, head) must be produced exactly once and no head >= G ever stored.
    #[test]
    fn dv3_lane_maps_cover_each_dot_once() {
        for g in [6usize, 12] {
            let nhg = (g + 3) / 4;
            let partial = g % 4 != 0;
            let mut stores = vec![0u32; 8 * g];          // [position-in-warp][head]
            let mut wmx_writes = vec![0u32; g];          // wmx[warp][head] (one warp)
            let mut tile_rows_computed = 0usize;         // (hg, hh) q-row computations per warp-chunk
            for hg in 0..nhg {
                for hh in 0..4 {
                    if partial && hg * 4 + hh >= g { continue; }
                    tile_rows_computed += 1;
                }
                for lane in 0..32usize {
                    let (i, h) = (lane >> 2, hg * 4 + (lane & 3));
                    if !partial || h < g { stores[i * g + h] += 1; }
                    if lane < 4 && (!partial || hg * 4 + lane < g) { wmx_writes[hg * 4 + lane] += 1; }
                }
            }
            assert!(stores.iter().all(|&c| c == 1), "G={g}: every (position, head) dot stored exactly once");
            assert!(wmx_writes.iter().all(|&c| c == 1), "G={g}: every head's warp max written exactly once");
            assert_eq!(tile_rows_computed, g, "G={g}: one q row per head, none for the padding rows");
            if !partial { assert_eq!(nhg * 4, g); }      // G=12: no guard can fire
            // acc / w4a: 16 * g threads = g heads x 16 dim groups, each (head, dgl) once
            let mut owners = vec![0u32; g * 16];
            for tid in 0..16 * g { owners[(tid >> 4) * 16 + (tid & 15)] += 1; }
            assert!(owners.iter().all(|&c| c == 1));
            // pmx publish: threads 0..G of the dots CTA
            assert!(g <= 256);
        }
    }

    #[test]
    fn dv3_dots_grid_at_w4() {
        // ~96 CTAs whatever m is; nkv = 1 at W=4 (kv replicated), clamp to the 36-chunk plane
        let nchg = |nkv: usize, m: usize| (96 / (nkv * m)).clamp(1, DV3_NCH);
        assert_eq!([nchg(1, 1), nchg(1, 2), nchg(1, 4), nchg(1, 8), nchg(1, 16)], [36, 36, 24, 12, 6]);
        assert_eq!([nchg(2, 1), nchg(2, 4), nchg(2, 16)], [36, 12, 3]);
    }
