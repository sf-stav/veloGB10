//! Unit tests for the shard-factor selection (former inline test module `shard_factor_tests` of `gpu.rs`; file map in `mod.rs`).

    use super::*;

    fn cfg(intermediate_size: usize, num_heads: usize, num_kv_heads: usize,
           lin_k: usize, lin_v: usize, vocab: usize, num_experts: usize, si: usize) -> Config {
        Config {
            family: crate::qwen::Family::Qwen35,
            hc_count: 1, hc_lowrank: 0, gdn_gate_sigmoid: false, ple_layer: None, ple_ngram_size: 0,
            ple_heads_per_ngram: 0, ple_vocab_base: 0, ple_embed_dim: 0, ple_conv_kernel: 0,
            ple_vocab_divisor: 128, ple_seed: 1234, indexer_n_heads: 0, indexer_head_dim: 0,
            indexer_budget: 0, indexer_compress_ratio: 1,
            hidden_size: 5120,
            intermediate_size,
            num_layers: 64,
            num_heads,
            num_kv_heads,
            head_dim: 128,
            lin_num_k_heads: lin_k,
            lin_num_v_heads: lin_v,
            lin_k_dim: 128,
            lin_v_dim: 128,
            conv_kernel: 4,
            vocab_size: vocab,
            rms_eps: 1e-6,
            rope_theta: 1e7,
            // Pre-existing breakage (not Phase-2): 7e188f9 added Config::rope_yarn_factor and
            // did not update this test initializer, so `cargo test --lib` stopped compiling.
            rope_yarn_factor: 1.0,
            yarn_native_window: 262144,
            rotary_dim: 64,
            max_position_embeddings: 262144,
            eos_token_id: 0,
            layer_types: Vec::new(),
            tie_word_embeddings: true,
            is_moe: num_experts > 0,
            num_experts,
            num_experts_per_tok: 8,
            moe_intermediate_size: 1024,
            shared_expert_intermediate_size: si,
            mlp_only_layers: Vec::new(),
            qk_norm: false,
            router_sigmoid: false,
            router_expert_bias: false,
            route_norm: true,
            router_scaling: 1.0,
            lm_head_fp32: false,
        }
    }

    #[test]
    fn factors_replicate_when_not_divisible() {
        // kv_heads=2 → attention factor must be 1 at world>2 (the binding constraint).
        let c = cfg(17408, 32, 2, 16, 48, 248320, 256, 1536);
        for world in [4usize, 8, 16] {
            assert_eq!(attn_shard_factor_for(&c, world, true), 1, "attn world {world}");
        }
        assert_eq!(attn_shard_factor_for(&c, 2, true), 2, "attn world 2");

        // kv_heads=4 → attention divides up to 4, then replicates.
        let c4 = cfg(17408, 32, 4, 16, 48, 248320, 256, 1536);
        assert_eq!(attn_shard_factor_for(&c4, 4, true), 4, "attn kv=4 world 4");
        assert_eq!(attn_shard_factor_for(&c4, 8, true), 1, "attn kv=4 world 8");

        // GDN heads 16/48 → divide up to 16; a non-divisible pair replicates.
        assert_eq!(gdn_shard_factor_for(&c, 16, true), 16, "gdn world 16");
        let c_gdn = cfg(17408, 32, 2, 15, 48, 248320, 256, 1536);
        assert_eq!(gdn_shard_factor_for(&c_gdn, 2, true), 1, "gdn lin_k=15 world 2");

        // FFN intermediate must divide 16·world for the col shard (dense only, mixers-independent).
        let c_ffn = cfg(48, 32, 2, 16, 48, 248320, 256, 1536);
        assert_eq!(ffn_shard_factor_for(&c_ffn, 2, false), 1, "ffn 48 world 2 (not 32-divisible)");
        assert_eq!(ffn_shard_factor_for(&c, 8, false), 8, "ffn 17408 world 8");

        // Experts band: 256 divides through 16.
        assert_eq!(experts_shard_factor_for(&c, 16, true), 16, "experts world 16");
        let c_exp = cfg(17408, 32, 2, 16, 48, 248320, 255, 1536);
        assert_eq!(experts_shard_factor_for(&c_exp, 2, true), 1, "experts 255 world 2");

        // Shared expert uses si, not intermediate_size.
        assert_eq!(shared_expert_shard_factor_for(&c, 16, true), 16, "shared si=1536 world 16");
        let c_si = cfg(17408, 32, 2, 16, 48, 248320, 256, 100);
        assert_eq!(shared_expert_shard_factor_for(&c_si, 2, true), 1, "shared si=100 world 2");

        // Vocab: 248320 divides 16·world through 16.
        assert_eq!(vocab_shard_factor_for(&c, 16, true), 16, "vocab world 16");
        let c_voc = cfg(17408, 32, 2, 16, 48, 248321, 256, 1536);
        assert_eq!(vocab_shard_factor_for(&c_voc, 2, true), 1, "vocab 248321 world 2");

        // mixers=false gates attn/gdn/experts/shared/vocab to 1, but NOT dense FFN.
        assert_eq!(attn_shard_factor_for(&c4, 4, false), 1, "attn mixers off");
        assert_eq!(gdn_shard_factor_for(&c, 16, false), 1, "gdn mixers off");
        assert_eq!(experts_shard_factor_for(&c, 16, false), 1, "experts mixers off");
        assert_eq!(vocab_shard_factor_for(&c, 16, false), 1, "vocab mixers off");
        assert_eq!(ffn_shard_factor_for(&c, 8, false), 8, "ffn independent of mixers");
    }
