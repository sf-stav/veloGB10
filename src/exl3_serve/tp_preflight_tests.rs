//! Unit tests (module `tp_preflight_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    #[test]
    fn tp_preflight_passes_an_fp4_local_pack() {
        let d = std::env::temp_dir().join(format!("v0710_fp4l_pf_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("config.json"), serde_json::json!({"model_type": "qwen4_exp", "quantization_config": {
            "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": {
                "a.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16, "weight_scale_encoding": "csf"},
                "a.self_attn.q_proj": {"quant_algo": "MXFP8", "group_size": 32}}}}).to_string()).unwrap();
        std::fs::write(d.join("model.safetensors.index.json"), "{}").unwrap();
        let r = super::tp_pack_width_preflight(d.to_str().unwrap());
        std::fs::remove_dir_all(&d).ok();
        r.expect("an FP4-Local pack passes the TP head preflight (v0.7.9 refused it: no quantization_config.json)");
    }

    #[test]
    fn tp_pack_width_preflight_refusal_names_the_ship() {
        let e = super::tp_pack_width_preflight("/nonexistent-pack-dir").unwrap_err().to_string();
        assert!(e.contains("pack width preflight (before the node sync; nothing was shipped)"),
            "{e}");
    }
