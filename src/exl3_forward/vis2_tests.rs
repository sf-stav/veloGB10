//! Unit tests (module `vis2_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    use super::*;

    /// VIS-2 position oracle (AGENTS §3: class-different reference): the reference
    /// Qwen4ExpModel.get_rope_index + Qwen4ExpTextRotaryEmbedding (transformers 5.17.0) on a
    /// two-image sequence (tests/fixtures/vis2_mrope_hf.json, scripts/vis2/mrope_fixture.py) vs the
    /// engine's mrope_positions + mrope_cos_row: positions and delta exact, cos/sin within f32 noise.
    #[test]
    fn vis2_mrope_matches_hf() {
        let raw = std::fs::read_to_string("tests/fixtures/vis2_mrope_hf.json").expect("fixture");
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let spans: Vec<crate::vision_encoder::ImageSpan> = v["spans"].as_array().unwrap().iter().map(|s| {
            crate::vision_encoder::ImageSpan {
                start: s["start"].as_u64().unwrap() as usize, num_tokens: s["num_tokens"].as_u64().unwrap() as usize,
                grid_h: s["grid_h"].as_u64().unwrap() as usize, grid_w: s["grid_w"].as_u64().unwrap() as usize,
                pixel_hash: 0,
            }
        }).collect();
        let len = v["len"].as_u64().unwrap() as usize;
        let (pos, delta) = crate::vision_encoder::mrope_positions(len, &spans).unwrap();
        let want: Vec<[i64; 3]> = v["positions"].as_array().unwrap().iter()
            .map(|p| { let a = p.as_array().unwrap(); [a[0].as_i64().unwrap(), a[1].as_i64().unwrap(), a[2].as_i64().unwrap()] })
            .collect();
        assert_eq!(pos, want, "3-axis positions differ from HF get_rope_index");
        assert_eq!(delta, v["delta"].as_i64().unwrap(), "rope delta differs from HF");
        let rdim = v["rotary_dim"].as_u64().unwrap() as usize;
        let theta = v["rope_theta"].as_f64().unwrap() as f32;
        let s: Vec<usize> = v["mrope_section"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let fr_t = RopeFreqs::native(rdim, theta);
        let mut worst = 0f32;
        for (t, p) in pos.iter().enumerate() {
            let row = mrope_cos_row(&fr_t, *p, [s[0], s[1], s[2]]);
            for i in 0..rdim / 2 {
                let c = v["cos"][t][i].as_f64().unwrap() as f32;
                let sn = v["sin"][t][i].as_f64().unwrap() as f32;
                worst = worst.max((row[i] - c).abs()).max((row[rdim + i] - sn).abs());
            }
        }
        assert!(worst < 2e-5, "cos/sin differ from HF by {worst}");
        // a text position's row is cos_tab's row exactly (t == h == w)
        assert_eq!(mrope_cos_row(&fr_t, [7, 7, 7], [11, 11, 10]), mrope_cos_row(&fr_t, [7, 7, 7], [32, 0, 0]));
    }
