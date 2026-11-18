//! Unit tests (module `rope_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    use super::*;

    /// The spec's ramp, computed HERE: HF/vLLM YaRN blends each dim's inverse frequency from
    /// pure interpolation (inv/factor) at the low-frequency end to pure extrapolation (inv,
    /// unchanged) at the high-frequency end; the python-f64 correction range (beta_fast = 32
    /// rotations, beta_slow = 1) locates the ramp. Returns (inv, ramp) per dim.
    fn reference_yarn(rdim: usize, theta: f32, factor: f32, native_max: usize) -> Vec<(f32, f32)> {
        let find = |rotations: f64| -> f64 {
            (native_max as f64 / (rotations * 2.0 * std::f64::consts::PI)).ln()
                * (rdim as f64) / (2.0 * (theta as f64).ln())
        };
        let low = find(32.0).floor().max(0.0);
        let high = find(1.0).ceil().min(rdim as f64 - 1.0);
        let (min_r, max_r) = if (high - low).abs() < 1e-12 {
            (low as f32, low as f32 + 0.001)
        } else { (low as f32, high as f32) };
        (0..rdim / 2).map(|i| {
            // HF-canonical inv_freq = 1 / base^(2i/rdim) - the ramp spec's own form; the EXL3
            // legacy loop's theta.powf(-e) is an equivalent expression that differs by 1 ulp.
            let inv = 1.0f32 / theta.powf((2 * i) as f32 / rdim as f32);
            let ramp = ((i as f32 - min_r) / (max_r - min_r)).clamp(0.0, 1.0);
            let yarn = inv / factor * ramp + inv * (1.0 - ramp);
            (yarn, ramp)
        }).collect()
    }

    /// The tables' YaRN freqs match the independent reference: the ramp's extremes are
    /// bit-exact (ramp 0 = pure extrapolation, ramp 1 = pure interpolation), the blend
    /// region within 1e-6 relative (a different but equivalent op order).
    #[test]
    fn yarn_freqs_match_an_independently_derived_ramp() {
        let (rdim, theta, native, factor) = (256usize, 1e7f32, 262_144usize, 4.0f32);
        let fr = RopeFreqs::yarn(rdim, theta, factor, native);
        let want = reference_yarn(rdim, theta, factor, native);
        assert_eq!(fr.half(), want.len());
        let mut worst = 0f32;
        let mut extremes = 0usize;
        for (i, (w, ramp)) in want.iter().enumerate() {
            let g = fr.inv(i);
            if *ramp == 0.0 || *ramp == 1.0 {
                assert_eq!(g.to_bits(), w.to_bits(), "dim {i}: the ramp extreme must be exact");
                extremes += 1;
            } else {
                worst = worst.max((g - w).abs() / g.abs().max(1e-30));
            }
        }
        assert!(extremes >= 2, "the reference must pin both ramp extremes");
        assert!(worst < 1e-6, "the blend region deviates by {worst} relative");
    }

    /// Factor 1.0 IS native: same bits as the legacy load-time loop, mscale exactly 1.0
    /// (so the cos_sin multiply is the f32 identity and no table byte changes).
    #[test]
    fn factor_one_is_the_native_table_bit_for_bit() {
        let (rdim, theta) = (64usize, 1e7f32);
        let fr = RopeFreqs::native(rdim, theta);
        for p in [0usize, 1, 7, 2048, 262_143] {
            for i in 0..rdim / 2 {
                let inv = theta.powf(-(2.0 * i as f32) / rdim as f32);
                let f = p as f32 * inv;
                assert_eq!(fr.cos_sin(p, i).0.to_bits(), f.cos().to_bits(), "p {p} dim {i} cos");
                assert_eq!(fr.cos_sin(p, i).1.to_bits(), f.sin().to_bits(), "p {p} dim {i} sin");
            }
        }
        assert_eq!(fr.mscale().to_bits(), 1.0f32.to_bits());
    }

    /// mscale: the vLLM formula 0.1·ln(f)+1, derived here; exactly 1.0 at f <= 1.
    #[test]
    fn mscale_is_the_vllm_formula() {
        for f in [2.0f32, 4.0, 8.0, 16.0] {
            let want = 0.1f32 * f.ln() + 1.0;
            assert_eq!(crate::dspark::oracle::yaarn_mscale(f).to_bits(), want.to_bits(), "factor {f}");
        }
        for f in [1.0f32, 0.5] {
            assert_eq!(crate::dspark::oracle::yaarn_mscale(f).to_bits(), 1.0f32.to_bits(), "factor {f}");
        }
    }

    /// The ramp's shape: the highest-frequency dim keeps its native frequency (pure
    /// extrapolation at ramp 0), the lowest-frequency dim interpolates by 1/factor
    /// (ramp 1) — the PR's own consistency check, restated against the independent math.
    #[test]
    fn yarn_interpolates_low_freq_dims_and_keeps_high_freq_dims() {
        let (rdim, theta, native) = (64usize, 1e7f32, 262_144usize);
        let native_fr = RopeFreqs::native(rdim, theta);
        let yarn_fr = RopeFreqs::yarn(rdim, theta, 4.0, native);
        assert_eq!(yarn_fr.inv(0).to_bits(), native_fr.inv(0).to_bits(), "dim 0 keeps its native frequency");
        let last = rdim / 2 - 1;
        assert!((yarn_fr.inv(last) - native_fr.inv(last) / 4.0).abs()
                <= native_fr.inv(last) / 4.0 * 1e-5,
                "the lowest-frequency dim interpolates by 1/factor");
    }
