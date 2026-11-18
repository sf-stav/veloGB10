//! Inline unit tests, moved out of `gptq.rs` verbatim (module `tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::*;

    #[test]
    fn igs_headroom_uses_block_distribution_not_literal_outlier() {
        let mut stats = IgsHistogram {
            bins: vec![0; IGS_HIST_BINS],
            running_max: 100.0,
            zero_blocks: 0,
            invalid_blocks: 0,
        };
        let common_bin = igs_hist_bin_index(1.0);
        stats.bins[common_bin] = 10_000;
        stats.bins[igs_hist_bin_index(100.0)] = 1;
        let diag = igs_scale_from_hist(&stats, IgsCalibConfig::default()).unwrap();
        let anchor = igs_hist_bin_center(common_bin);
        assert_eq!(diag.anchor, Some(anchor));
        assert!((diag.selected_amax - 16384.0 * anchor).abs() < diag.selected_amax * 1e-6);
        assert!(diag.has_headroom);
        assert!(!diag.range_exceeds_e4m3);
    }

    #[test]
    fn igs_max_mode_preserves_legacy_reciprocal_scale() {
        let stats = IgsHistogram {
            bins: vec![0; IGS_HIST_BINS],
            running_max: 42.0,
            zero_blocks: 3,
            invalid_blocks: 0,
        };
        let cfg = IgsCalibConfig { method: IgsMethod::Max, ..IgsCalibConfig::default() };
        let diag = igs_scale_from_hist(&stats, cfg).unwrap();
        assert_eq!(diag.selected_amax, 42.0);
        assert!((diag.input_global_scale - 2688.0 / 42.0).abs() < 1e-6);
    }

    #[test]
    fn igs_headroom_flags_ranges_no_single_e4m3_global_scale_can_cover() {
        let mut stats = IgsHistogram {
            bins: vec![0; IGS_HIST_BINS],
            running_max: 2.0f32.powi(20),
            zero_blocks: 0,
            invalid_blocks: 0,
        };
        // Keep the anchor above the deliberate upper/1e6 noise-floor cutoff while still
        // exceeding the E4M3 normal-range ratio.
        stats.bins[igs_hist_bin_index(2.0)] = 10_000;
        stats.bins[igs_hist_bin_index(2.0f32.powi(20))] = 10_000;
        let diag = igs_scale_from_hist(&stats, IgsCalibConfig::default()).unwrap();
        assert!(diag.range_exceeds_e4m3);
        assert!(!diag.has_headroom);
    }

    #[test]
    fn igs_rejects_non_finite_activation_blocks() {
        let stats = IgsHistogram {
            bins: vec![1; IGS_HIST_BINS],
            running_max: 1.0,
            zero_blocks: 0,
            invalid_blocks: 1,
        };
        assert!(igs_scale_from_hist(&stats, IgsCalibConfig::default()).is_none());
    }

    #[test]
    fn pretokenized_calibration_preserves_sample_boundaries() {
        let path = std::env::temp_dir().join(format!("gptq-pretokenized-test-{}.jsonl", std::process::id()));
        std::fs::write(&path,
            "{\"input_ids\":[1,2,3,4],\"text\":\"ignored\"}\n{\"input_ids\":[5,6,7,8],\"text\":\"ignored too\"}\n").unwrap();
        let samples = calib_tokens(Path::new("/tokenizer-is-deliberately-absent"), &path, 2, 4, 16).unwrap();
        assert_eq!(samples, vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn maca_accepts_variable_rows_but_fixed_loader_rejects_them() {
        let path =
            std::env::temp_dir().join(format!("gptq-maca-test-{}.jsonl", std::process::id()));
        std::fs::write(&path,
            "{\"input_ids\":[1,2],\"text\":\"short\"}\n{\"input_ids\":[3,4,5,6],\"text\":\"long\"}\n").unwrap();
        assert!(calib_tokens(
            Path::new("/tokenizer-is-deliberately-absent"),
            &path,
            2,
            4,
            16
        )
        .is_err());
        let samples = calib_tokens_mode(
            Path::new("/tokenizer-is-deliberately-absent"),
            &path,
            2,
            4,
            16,
            true,
        )
        .unwrap();
        assert_eq!(samples, vec![vec![1, 2], vec![3, 4, 5, 6]]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn dflash2_checkpoint_roundtrip_is_exact() {
        let path = std::env::temp_dir().join(format!("gptq-df2-checkpoint-{}.bin", std::process::id()));
        let r = Rec { qw: vec![1, 2, 3, 4], sc: vec![9, 8], m: 16, k: 32,
                      gs: 123.25, igs: Some(77.5) };
        write_df2_checkpoint(&path, &[("layers.0.self_attn.q_proj.weight".into(), &r)]).unwrap();
        let mut got = read_df2_checkpoint(&path).unwrap();
        assert_eq!(got.len(), 1);
        let (name, q) = got.pop().unwrap();
        assert_eq!(name, "layers.0.self_attn.q_proj.weight");
        assert_eq!((q.qw, q.sc, q.m, q.k, q.gs, q.igs),
                   (r.qw, r.sc, r.m, r.k, r.gs, r.igs));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn output_dir_must_be_distinct_and_empty() {
        let root = std::env::temp_dir().join(format!("gptq-output-test-{}", std::process::id()));
        let input = root.join("input");
        let out = root.join("out");
        std::fs::create_dir_all(&input).unwrap();
        let err = prepare_output_dir(&input, &[&input]).unwrap_err().to_string();
        assert!(err.contains("must differ"));
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(out.join("partial.bin"), b"x").unwrap();
        let err = prepare_output_dir(&out, &[&input]).unwrap_err().to_string();
        assert!(err.contains("not empty"));
        std::fs::remove_file(out.join("partial.bin")).unwrap();
        prepare_output_dir(&out, &[&input]).unwrap();
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Families pushed around forced shard boundaries (tiny shard size) must never be split.
    #[test]
    fn writer_keeps_packed_families_in_one_shard() {
        let dir = std::env::temp_dir().join(format!("gptq-writer-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut w = Writer::new(&dir, 1000);   // 1 KB shards
        for i in 0..7 {
            let stem = format!("layers.{i}.w");
            // a GPTQ triple (~700 B) …
            w.push_nvfp4(&stem, vec![1u8; 512], vec![2u8; 64], 16, 64, 1.0, None);
            // … a verbatim triple in index (name) order, as the base-artifact copy emits it
            let vs = format!("mtp.{i}.w");
            w.push(Out { name: format!("{vs}.weight_global_scale"), dtype: safetensors::Dtype::F32, shape: vec![1], data: vec![0u8; 4] });
            w.push(Out { name: format!("{vs}.weight_packed"), dtype: safetensors::Dtype::U8, shape: vec![16, 32], data: vec![3u8; 512] });
            w.push(Out { name: format!("{vs}.weight_scale"), dtype: safetensors::Dtype::F8_E4M3, shape: vec![16, 4], data: vec![4u8; 64] });
            // and a plain bf16 tensor
            w.push(Out { name: format!("norm.{i}.weight"), dtype: safetensors::Dtype::BF16, shape: vec![64], data: vec![0u8; 128] });
        }
        w.finish().expect("no split families");
        let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap()).unwrap();
        let wm = idx["weight_map"].as_object().unwrap();
        assert!(wm.len() == 7 * 7);
        let shards: std::collections::BTreeSet<&str> = wm.values().map(|v| v.as_str().unwrap()).collect();
        assert!(shards.len() > 5, "the tiny shard size must have produced many shards ({})", shards.len());
        std::fs::remove_dir_all(&dir).unwrap();
    }
    fn host_scale_stats(w: &[f32], s_tensor: f32, nclip: usize) -> (f64, f64, f64) {
        const RATIOS: [f32; 7] = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7];
        let mut out = (0.0, 0.0, 0.0);
        for x in w.chunks_exact(16) {
            let amax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let mut best = (u8::default(), f64::INFINITY);
            for &ratio in RATIOS.iter().take(nclip) {
                let raw = if amax > 0.0 { amax * ratio / quant::E2M1_MAX / s_tensor } else { 0.0 };
                let scode = quant::f32_to_e4m3(raw);
                let scale = quant::e4m3_to_f32(scode) * s_tensor;
                let mse = x.iter().map(|&v| {
                    let q = if scale > 0.0 { quant::e2m1_to_f32(quant::f32_to_e2m1(v / scale)) * scale } else { 0.0 };
                    let d = (q - v) as f64;
                    d * d
                }).sum::<f64>();
                if mse < best.1 { best = (scode, mse); }
            }
            let es = quant::e4m3_to_f32(best.0);
            let scale = es * s_tensor;
            for &v in x {
                let code = if scale > 0.0 { quant::f32_to_e2m1(v / scale) } else { 0 };
                let z = es * quant::e2m1_to_f32(code);
                out.0 += v as f64 * z as f64;
                out.1 += z as f64 * z as f64;
            }
            out.2 += best.1;
        }
        out
    }

    fn host_hessian_loss(w: &[f32; 16], h: &[f32; 256], s_tensor: f32, code: u8) -> f64 {
        let scale = quant::e4m3_to_f32(code) * s_tensor;
        let mut dw = [0.0f64; 16];
        for i in 0..16 {
            let q = quant::e2m1_to_f32(quant::f32_to_e2m1(w[i] / scale)) * scale;
            dw[i] = (w[i] - q) as f64;
        }
        let mut loss = 0.0f64;
        for i in 0..16 {
            for j in 0..16 {
                loss += dw[i] * h[i * 16 + j] as f64 * dw[j];
            }
        }
        loss
    }

    fn host_local_hessian_code(w: &[f32; 16], h: &[f32; 256], s_tensor: f32) -> u8 {
        (1u8..=126).min_by(|&a, &b| {
            host_hessian_loss(w, h, s_tensor, a)
                .total_cmp(&host_hessian_loss(w, h, s_tensor, b))
                .then_with(|| a.cmp(&b))
        }).unwrap()
    }

    fn host_clip_code(w: &[f32; 16], s_tensor: f32) -> u8 {
        const RATIOS: [f32; 7] = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7];
        let amax = w.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        RATIOS.into_iter().map(|ratio| {
            quant::f32_to_e4m3(amax * ratio / quant::E2M1_MAX / s_tensor)
        }).min_by(|&a, &b| {
            let mse = |code| {
                let scale = quant::e4m3_to_f32(code) * s_tensor;
                w.iter().map(|&v| {
                    let q = quant::e2m1_to_f32(quant::f32_to_e2m1(v / scale)) * scale;
                    let d = (v - q) as f64;
                    d * d
                }).sum::<f64>()
            };
            mse(a).total_cmp(&mse(b)).then_with(|| a.cmp(&b))
        }).unwrap()
    }

    #[test]
    fn static_activation_order_is_descending_stable_and_finite_first() {
        let order = static_activation_order(&[2.0, 9.0, 9.0, f32::NAN, 1.0, f32::INFINITY]);
        assert_eq!(order, vec![1, 2, 0, 4, 3, 5]);
    }

    #[test]
    fn alternating_scale_fit_never_increases_nvfp4_error() {
        let w: Vec<f32> = (0..256).map(|i| {
            let x = i as f32 - 127.5;
            (x * 0.071).sin() * (0.2 + (i % 29) as f32 * 0.037) + x * 0.0009
        }).collect();
        let amax = w.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let initial = e4m3_scale_of(amax) * 1.8;
        let fit = alternating_scale_fit(initial, 6, |s| host_scale_stats(&w, s, 7));
        assert!(fit.final_mse <= fit.initial_mse);
        assert!(fit.accepted > 0, "synthetic bad initial scale should admit an improving step");
        assert!((fit.scale - initial).abs() > initial * 1e-5);
    }

    #[test]
    fn alternating_scale_fit_keeps_zero_tensor_scale() {
        let w = [0.0f32; 32];
        let fit = alternating_scale_fit(1.0, 4, |s| host_scale_stats(&w, s, 7));
        assert_eq!(fit.scale, 1.0);
        assert_eq!(fit.final_mse, 0.0);
        assert_eq!(fit.accepted, 0);
    }

    #[test]
    fn local_hessian_fp8_sweep_beats_clip_mse_on_its_objective() {
        let mut w = [0.0f32; 16];
        w[0] = 0.37;
        w[1] = -0.43;
        for (i, v) in w[2..].iter_mut().enumerate() {
            *v = if i & 1 == 0 { 4.0 + i as f32 * 0.14 } else { -4.3 - i as f32 * 0.11 };
        }
        let mut h = [0.0f32; 256];
        for i in 0..16 { h[i * 16 + i] = if i < 2 { 1.0e6 } else { 1.0 }; }
        let amax = w.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let s_tensor = e4m3_scale_of(amax);
        let local = host_local_hessian_code(&w, &h, s_tensor);
        let clip = host_clip_code(&w, s_tensor);
        let local_loss = host_hessian_loss(&w, &h, s_tensor, local);
        let clip_loss = host_hessian_loss(&w, &h, s_tensor, clip);
        assert_ne!(local, clip, "the weighted fixture must distinguish the two objectives");
        assert!(local_loss < clip_loss, "local={local_loss:e}, clip={clip_loss:e}");
    }
