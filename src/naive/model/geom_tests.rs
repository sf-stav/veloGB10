//! Inline unit tests, moved out of `model.rs` verbatim (module `geom_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::{pack_geom, GEOM_FIELDS};

    /// The compiled-from source, so a mask edited in the kernel but not in `pack_geom` fails here.
    const CU: &str = src_text!("kernels/gpu_naive");

    /// The kernel's hand-written unpack must agree with `pack_geom`, field for field. If the two
    /// expressions ever drift, this fails instead of the drift reaching a serving box.
    #[test]
    fn geom_fields_match_cuda() {
        for &(name, shift, width) in GEOM_FIELDS.iter() {
            let mask = (1u64 << width) - 1;
            let expr = if shift == 0 {
                format!("geom & 0x{:X}", mask)
            } else {
                format!("(geom >> {}) & 0x{:X}", shift, mask)
            };
            assert!(
                CU.contains(&expr),
                "kernels/gpu_naive.cu has no `{expr}` for field {name} (shift {shift}, {width} bits): \
                 the CUDA unpack and pack_geom have drifted apart"
            );
        }
        // Both DSA split kernels must still unpack S this way (2 sites today).
        assert_eq!(
            CU.matches("geom >> 40").count(),
            2,
            "expected exactly nv_attn_dsa_split and nv_attn_dsa_split_grp to unpack S at bit 40"
        );
    }

    /// S is `l.slots`, which is `max_pos` for a DSA layer and is used as the stride `g*S+t`.
    /// It must survive the shipped posture's 262144 -- which is why it gets 24 bits.
    #[test]
    fn geom_survives_the_shipped_max_pos() {
        for &s in &[4096usize, 65536, 131072, 262144] {
            let got = (pack_geom(2048, 8, 256, s) >> 40) & 0xFFFFFF;
            assert_eq!(got as usize, s, "S={s} truncated to {got} in the geom word");
        }
    }

    /// No field bleeds into a neighbour, at the shapes the serving path actually launches.
    #[test]
    fn geom_fields_round_trip_without_bleed() {
        for &k in &[2048usize, 4096] {
            for &ns in &[1usize, 8] {
                for &sl in &[64usize, 256, 2048] {
                    for &s in &[4096usize, 262144] {
                        let g = pack_geom(k, ns, sl, s);
                        let f = |sh: u32, w: u32| ((g >> sh) & ((1i64 << w) - 1)) as usize;
                        assert_eq!(
                            (f(0, 16), f(16, 8), f(24, 16), f(40, 24)),
                            (k, ns, sl, s),
                            "field bleed at k={k} nsplits={ns} split_len={sl} S={s}"
                        );
                    }
                }
            }
        }
    }

    /// The historical failure, stated so a future "simplification" back to the old layout fails
    /// loudly rather than at 262144 on a serving box.
    #[test]
    fn the_pre_fix_layout_would_have_truncated_s() {
        let s = 262144usize;
        let pre_fix = (((s as i64) & 0xFFFF) << 48) as i64; // exactly what 2acaa750 packed
        assert_eq!((pre_fix >> 48) & 0xFFFF, 0, "precondition: the old layout really did truncate S to 0");
        let now = (pack_geom(2048, 8, 256, s) >> 40) & 0xFFFFFF;
        assert_eq!(now as usize, s, "the current layout must carry S intact");
    }
