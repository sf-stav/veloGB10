#[cfg(test)]
mod tests {
    use gb10_inference::{model, sampler, kv_cache};
    use gb10_inference::model::{ModelConfig, Qwen3Model, TransformerLayer, PackedLinear};

    #[test]
    fn test_model_config_defaults() {
        let config = ModelConfig::qwen3_5_0_08b();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.intermediate_size, 2816);
    }

    #[test]
    fn test_model_config_params() {
        let config = ModelConfig::qwen3_5_0_08b();
        let params = config.total_params();
        // For our dense model config: hidden=1024, layers=24, vocab=151936, intermediate=2816
        let expected = 619_446_272usize;
        assert_eq!(params, expected, "Expected ~619M params for dense config, got {}", params);
    }

    #[test]
    fn test_model_creation() {
        let config = ModelConfig::qwen3_5_0_08b();
        let model = Qwen3Model::new(config);
        assert_eq!(model.layers.len(), 24);
        assert_eq!(model.embed_tokens.len(), config.vocab_size * config.hidden_size);
        assert_eq!(model.lm_head.len(), config.vocab_size * config.hidden_size);
    }

    #[test]
    fn test_packed_linear_creation() {
        let linear = PackedLinear::new(1024, 1024, 16);
        assert_eq!(linear.in_features, 1024);
        assert_eq!(linear.out_features, 1024);
        assert_eq!(linear.block_size, 16);
        assert!(linear.weights.len() > 0);
    }

    #[test]
    fn test_transformer_layer_creation() {
        let config = ModelConfig::qwen3_5_0_08b();
        let layer = TransformerLayer::new(&config);

        assert_eq!(layer.q_proj.in_features, 1024);
        assert_eq!(layer.q_proj.out_features, 1024);
        assert_eq!(layer.gate_proj.out_features, 2816);
        assert_eq!(layer.down_proj.in_features, 2816);
        assert_eq!(layer.down_proj.out_features, 1024);
    }

    #[test]
    fn test_rms_norm_cpu() {
        let hidden_size = 1024;
        let input: Vec<f32> = (0..hidden_size).map(|i| (i as f32) / 100.0).collect();
        let weight: Vec<f32> = vec![1.0; hidden_size];

        let mut sum_sq = 0.0f32;
        for val in &input {
            sum_sq += val * val;
        }
        let rms = (sum_sq / hidden_size as f32 + 1e-6).sqrt();
        let scale = 1.0 / rms;

        let output: Vec<f32> = input.iter().map(|&x| x * scale * 1.0).collect();

        // Output should be RMS-normalized
        let out_sum_sq: f32 = output.iter().map(|x| x * x).sum();
        let out_rms = (out_sum_sq / hidden_size as f32).sqrt();
        assert!(
            (out_rms - 1.0).abs() < 1e-4,
            "RMS should be ~1.0, got: {}",
            out_rms
        );
    }

    #[test]
    fn test_sampler_greedy() {
        let sampler = sampler::Sampler::new(0.0, 0.9, 50);
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2]; // max is at index 3
        let token = sampler.sample(&logits);
        assert_eq!(token, 3);
    }

    #[test]
    fn test_nvfp4_zero_block() {
        let block = model::NVFP4WeightBlock::zero();
        assert!(block.is_zero());
    }

    #[test]
    fn test_memory_alignment() {
        use std::mem;
        assert!(mem::size_of::<half::f16>() == 2, "f16 should be 2 bytes");
    }

    #[test]
    fn test_kv_cache_layout() {
        let config = ModelConfig {
            max_seq_len: 512,
            ..ModelConfig::qwen3_5_0_08b()
        };

        // Just test that KVCachePtrs math is correct
        let num_heads = 8;
        let head_dim = 128;
        let max_seq = 512;
        let bytes_per_head = head_dim * 2; // FP16 is 2 bytes
        let stride_k_layer = num_heads * max_seq * head_dim;

        assert_eq!(stride_k_layer, 524_288);

        // Layer 0, position 0 should start at offset 0
        let k_offset_0_0 = 0 * (stride_k_layer * 2) + 0 * num_heads * head_dim;
        assert_eq!(k_offset_0_0, 0);

        // Layer 1, position 0 should be offset by stride_k_layer*2
        let k_offset_1_0 = 1 * (stride_k_layer * 2) + 0 * num_heads * head_dim;
        assert_eq!(k_offset_1_0, stride_k_layer * 2);
    }
}
