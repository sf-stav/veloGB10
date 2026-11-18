//! `impl GpuModel`: `new`, `load_from_dir*` entry points, PTX init and the kernel-build-id handshake, batch kernel table,
//! KV mode, RoPE / TQ table builders and the MTP head loader; plus the `ReleaseWorkers` guard. `load_from_dir_impl` is in `load.rs`. Map: `mod.rs`.

use super::*;

impl GpuModel {
    pub fn new(host: &crate::qwen::Model) -> anyhow::Result<Self> {
        let dev = CudaDevice::new(0)?;
        let stream = fork_blocking_stream(&dev);
        let blas = CudaBlas::new(dev.clone())?;
        unsafe { blas.set_stream(Some(&stream))?; }
        Self::init_ptx(&dev)?;
        let (k, bk) = Self::load_batch_kernels(&dev)?;
        // E5/YARN1: trunk YaRN — the flag wins, then the shipped TpConfig (node), then the
        // pack config's rope_scaling (one precedence rule: qwen::pick_yarn).
        let mut cfg_host = host.config.clone();
        let (yarn, yarn_src) = crate::qwen::resolved_yarn(&cfg_host);
        cfg_host.rope_yarn_factor = yarn;
        let (cos_table, sin_table) = Self::build_rope_tables(&dev, &cfg_host, yarn_src)?;
        let sc_pos = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_rope = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_slot = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_winsrc = dev.alloc_zeros::<i32>(MAX_VERIFY * 8).unwrap();
        let sc_parent = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_path = dev.alloc_zeros::<u8>(MAX_VERIFY * MAX_VERIFY).unwrap();
        let sc_tok = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_pstart = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let moe_ids = dev.alloc_zeros::<i32>(16 * MAX_VERIFY).unwrap();
        let moe_wts = dev.alloc_zeros::<f32>(16 * MAX_VERIFY).unwrap();
        let sc_i1a = dev.alloc_zeros::<i32>(1).unwrap();
        let sc_i1b = dev.alloc_zeros::<i32>(1).unwrap();
        let sv_pf = dev.alloc_zeros::<f32>(3 * MAX_VERIFY).unwrap();
        let sv_ki = dev.alloc_zeros::<i32>(2 * MAX_VERIFY).unwrap();
        let sv_cand = dev.alloc_zeros::<u64>(MAX_VERIFY * 16).unwrap();
        let sv_sd = dev.alloc_zeros::<u32>(MAX_VERIFY).unwrap();
        let sv_p  = dev.alloc_zeros::<f32>(MAX_VERIFY).unwrap();
        let sv_r  = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let mr_tok = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let mr_pos = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        // alloc_zeros is cuMemAllocAsync and does NOT zero. This buffer MUST be all zeros.
        let mut mtp_sids = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        dev.memset_zeros(&mut mtp_sids).unwrap();
        let deq_scratch = std::sync::Mutex::new(None);
        let fp8e_map: std::sync::Mutex<std::collections::HashMap<u64, cudarc::driver::CudaSlice<u8>>> = std::sync::Mutex::new(std::collections::HashMap::new());
        let fp8e_scratch: std::sync::Mutex<Option<(cudarc::driver::CudaSlice<u8>, cudarc::driver::CudaSlice<f32>)>> = std::sync::Mutex::new(None);
        let fp8e_map: std::sync::Mutex<std::collections::HashMap<u64, cudarc::driver::CudaSlice<u8>>> = std::sync::Mutex::new(std::collections::HashMap::new());
        let fp8e_scratch: std::sync::Mutex<Option<(cudarc::driver::CudaSlice<u8>, cudarc::driver::CudaSlice<f32>)>> = std::sync::Mutex::new(None);
        let splitk_partials = dev.alloc_zeros::<f32>(SPLITK_MAX_TILES * SPLITK_MAX_S * 256).unwrap();
        // R9 (world>2): fp32 upcast scratch for the no-rounding all-reduce. Sized for the widest
        // single-slot reduction (TP_SLOT_BYTES/4 fp32 elems = the whole ring slot in fp32).
        let tp_f32_scratch = dev.alloc_zeros::<f32>(crate::tp::TP_SLOT_BYTES / 2).unwrap();
        dev.synchronize().unwrap();
        let cfg = host.config.clone();
        let moe_g = new_moe_grouped_scratch(&dev, &cfg, MAX_VERIFY * cfg.num_experts_per_tok);
        let moe_g_pf = new_moe_grouped_scratch(&dev, &cfg, crate::batch::PREFILL_CHUNK * cfg.num_experts_per_tok);
        let verify_params = dev.alloc_zeros::<i32>(4).unwrap();
        let verify_resid = dev.alloc_zeros::<half::bf16>(cfg.resid_width() * MAX_VERIFY).unwrap();
        let verify_graphs = std::sync::Mutex::new(std::collections::HashMap::new());
        let gdn_slot_tables = std::sync::Mutex::new(std::collections::HashMap::new());
        let _h = cfg.hidden_size;
        let _pref = "model.language_model";

        // Upload weights directly from the Model's f32 fields (small models only)
        let up_f = |v: &[f32]| -> S { dev.htod_sync_copy(v).unwrap() };
        let up_b = |v: &[f32]| -> B {
            let bv: Vec<half::bf16> = v.iter().map(|&x| half::bf16::from_f32(x)).collect();
            dev.htod_sync_copy(&bv).unwrap()
        };
        let embed = W::Bf16(up_b(&host.embed_tokens));
        let final_norm = up_f(&host.norm);
        let lm_head = host.lm_head.as_ref().map(|lh| W::Bf16(up_b(lh)));

        let layers: Vec<GpuLayer> = host.layers.iter().map(|l| {
            let la = l.linear_attn.as_ref().map(|la| GpuLinearAttn {
                in_proj: GdnIn::Split {
                    qkv: W::Bf16(up_b(&la.in_proj_qkv)), z: W::Bf16(up_b(&la.in_proj_z)),
                    b: W::Bf16(up_b(&la.in_proj_b)), a: W::Bf16(up_b(&la.in_proj_a)) },
                conv1d: up_f(&la.conv1d), a_log: up_f(&la.a_log), dt_bias: up_f(&la.dt_bias),
                norm: up_f(&la.norm), out_proj: W::Bf16(up_b(&la.out_proj)),
            });
            let fa = l.full_attn.as_ref().map(|fa| GpuFullAttn {
                qkv: AttnIn::Split { q: W::Bf16(up_b(&fa.q_proj)), k: W::Bf16(up_b(&fa.k_proj)),
                                     v: W::Bf16(up_b(&fa.v_proj)) },
                o_proj: W::Bf16(up_b(&fa.o_proj)), q_norm: up_f(&fa.q_norm), k_norm: up_f(&fa.k_norm), indexer: None,
            });
            GpuLayer {
                layer_type: l.layer_type, la, fa,
                mlp: Ffn::Dense(GpuMlp { gate: W::Bf16(up_b(&l.mlp.gate_proj)), up: W::Bf16(up_b(&l.mlp.up_proj)), down: W::Bf16(up_b(&l.mlp.down_proj)) }),
                input_ln: up_f(&l.input_layernorm), post_ln: up_f(&l.post_attention_layernorm),
                hc: None, ple: None,
            }
        }).collect();

        // Load MTP head if present
        let mtp = Self::load_mtp_gpu(&host, &dev)?;
        let (kv_quant, kv_tq, kv_tq_b3, kv_k8v4, kv_k8v8) = Self::kv_modes_from_opts();
        let tq_tables = Self::build_tq_tables(&dev)?;
        dev.synchronize()?;
        let vmask_words = dev.htod_sync_copy(&vec![0u32; mask_words_for(cfg.vocab_size) * MAX_VERIFY]).unwrap();
        let vmask_flags = dev.htod_sync_copy(&vec![0i32; MAX_VERIFY]).unwrap();
        Ok(Self { dev, blas, stream, cfg, embed, lm_head, final_norm, layers, mtp, k, bk, cos_table, sin_table, sc_pos, sc_rope, sc_slot, sc_winsrc, sc_parent, sc_path, sc_tok, sc_pstart, moe_ids, moe_wts, moe_g, moe_g_pf, sc_i1a, sc_i1b, sv_pf, sv_ki, sv_sd, sv_cand, sv_p, sv_r, mr_tok, mr_pos, verify_params, verify_resid, verify_graphs, gdn_slot_tables, sv_t20: std::sync::OnceLock::new(), sv_t20_scratch: std::sync::OnceLock::new(), vmask_words, vmask_flags, vmask_words_ptr: std::sync::atomic::AtomicU64::new(0), vmask_flags_ptr: std::sync::atomic::AtomicU64::new(0), deq_scratch, fp8e_map, fp8e_scratch, splitk_partials, tp_f32_scratch, mtp_sids, draft_head: None, draft_ids: Vec::new(), lm_head_sharded: false, tp_sharded_at_load: false, gdn_chunk_raw_fn: gdn_chunk_load_raw_fn(), gdn_chunk_tc_raw_fn: gdn_chunk_tc_load_raw_fn(), kv_quant, kv_tq, kv_tq_b3, kv_k8v4, kv_k8v8, tq_tables, mxfp4: None, hc_mixer: None, tp_rank: 0, tp_world: 1, tp_ctx_dptr: 0, head_visits: None, dflash_tap: None, df2_capture: None, df2_capture_tree: None, df2_prime: None, dspark_capture: None, dspark_capture_tree: None, dspark_prime: None, dflash_lm_head: None, df2_full_head: None, df2_q_head: std::sync::OnceLock::new(), gptq_tap: std::sync::Mutex::new(None), rotated: Default::default(), rot_small: parking_lot::Mutex::new(None), w4a4: None, igs_by_name: Default::default(), igs_tap: std::sync::Mutex::new(None) })
    }

    /// Stream-load from safetensors directly as bf16 — no f32 intermediate.
    /// Required for 27B+ models where the f32 intermediate would exceed 128 GB.
    pub fn load_from_dir(model_dir: &str) -> anyhow::Result<(Self, crate::qwen::Config)> {
        let (mut m, cfg) = Self::load_from_dir_impl(model_dir, None, 1)?;
        m.apply_transform_config(model_dir)?;
        m.init_w4a4(model_dir)?;
        Ok((m, cfg))
    }

    /// TP load: same as `load_from_dir`, but when the family REQUIRES shard-at-load (hy_v3 —
    /// the full 169.6 GB does not fit one node even transiently, so the post-load in-place shard
    /// in `attach_tp` can never run) and mixer sharding is on, every sharded tensor is sliced to
    /// `rank` HOST-side before upload. Peak GPU memory is then ~half the model, not 1.5x it.
    /// `world` rides into the returned model's `tp_world` so the attach/guard semantics agree.
    pub fn load_from_dir_tp(model_dir: &str, rank: i32, world: i32) -> anyhow::Result<(Self, crate::qwen::Config)> {
        let (mut m, cfg) = Self::load_from_dir_impl(model_dir, Some(rank), world)?;
        m.apply_transform_config(model_dir)?;
        m.init_w4a4(model_dir)?;
        Ok((m, cfg))
    }
}

impl GpuModel {
    pub(super) fn init_ptx(dev: &Arc<CudaDevice>) -> anyhow::Result<()> {
        let ptx = Ptx::from_src(crate::ptx_embed::GPU_KERNELS);
        let fnames = ["rmsnorm_qwen","rmsnorm_gated","rmsnorm_perhead","silu_mul","add_residual",
            "sigmoid_gate","rope_rot_half","gqa_attention","write_kv","split_qgate","conv1d_step","delta_step",
            "f32tobf16","bf16tof32","fused_residual_rmsnorm","argmax_pass1","argmax_pass2",
            // FLASHNEXT-DF2-B: the hc-mean DFlash2 tap for hyper-connection trunks (every
            // gpu_kernels loader lists it — the last load wins cudarc's module slot).
            "df2_tap_hcmean_b"];
        dev.load_ptx(ptx, "gpu_kernels", &fnames)?;
        Ok(())
    }

    /// Refuse to run a binary whose PTX came from different kernel sources.
    ///
    /// The PTX is loaded from disk at runtime, so a deploy that copies the binary but not the kernels
    /// leaves a fresh binary driving OLD kernels. When launch parameters have changed (shared-memory
    /// size, grid shape, an argument list) that is not a slowdown — it is CUDA_ERROR_ILLEGAL_ADDRESS
    /// out of code that is perfectly correct. It shipped once. `scripts/deploy.sh` checks the three
    /// files agree, but a procedural check only guards the paths it is actually run on.
    ///
    /// So build.rs hashes the .cu bytes and bakes the hash into BOTH the PTX (`-DKERNEL_BUILD_ID`) and
    /// the binary (`env!("KERNEL_BUILD_ID")`). If they disagree the pair is mismatched, and we say so
    /// at startup — loudly, on any box, however it was assembled.
    pub(crate) fn assert_kernel_build_id(dev: &Arc<CudaDevice>, module: &str) -> anyhow::Result<()> {
        let expect = u64::from_str_radix(env!("KERNEL_BUILD_ID"), 16).unwrap_or(0);
        let Some(f) = dev.get_func(module, "kernel_build_id") else {
            anyhow::bail!("src/ptx/{module}.ptx has no kernel_build_id — it predates the build-ID stamp \
                           and cannot be verified against this binary. Run `cargo build --release`.");
        };
        let out = dev.alloc_zeros::<u64>(1)?;
        unsafe {
            use cudarc::driver::LaunchAsync;
            f.launch(cudarc::driver::LaunchConfig {
                grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0
            }, (&out,))?;
        }
        dev.synchronize()?;
        let got = dev.dtoh_sync_copy(&out)?[0];
        if got != expect {
            anyhow::bail!(
                "STALE KERNELS: src/ptx/{module}.ptx was built from different kernel sources than this \
                 binary (ptx={got:016x}, binary={expect:016x}). A deploy is THREE files — the binary \
                 AND both src/ptx/*.ptx. Run `cargo build --release && ./scripts/deploy.sh`.");
        }
        Ok(())
    }

    pub(super) fn load_batch_kernels(dev: &Arc<CudaDevice>) -> anyhow::Result<(KernelTable, HashMap<String, CudaFunction>)> {
        // TurboQuant b=3 K ([kv-tq]=3) loads the -DTQ_B3 build (src/ptx/gpu_batch_b3.ptx) —
        // same kernel names, 68-B TQ rows. The b=2 build (gpu_batch.ptx) stays untouched for
        // [kv-tq]=1. Both carry the same KERNEL_BUILD_ID (same source bytes).
        let tq_b3 = crate::opts::var(crate::opt!("kv-tq")).ok().as_deref() == Some("3");
        let (ptx_src, module) = if tq_b3 { (crate::ptx_embed::GPU_BATCH_B3, "gpu_batch_b3") } else { (crate::ptx_embed::GPU_BATCH, "gpu_batch") };
        let bptx = Ptx::from_src(ptx_src);
        let bfnames = ["rmsnorm_b","fused_res_rmsnorm_b","fused_res_rmsnorm_q_b","add_residual_b","silu_mul_b",
            "rmsnorm_perhead_b","rmsnorm_rope_b","rmsnorm_rope_kvwrite_b","rmsnorm_rope_kvwrite_q4_b","rmsnorm_gated_b","rope_b","conv1d_b","delta_step_b",
            "write_kv_b","compact_kv_b","sigmoid_gate_b","split_qgate_b",
            "gather_rope_b","embed_gather_b","argmax_b","gemm_binv_argmax_b","argmax_part_b",
            "conv1d_prefill","conv1d_prefill_state","delta_step_prefill","gdn_chunk_prefill_b","gdn_prep_b","attn_softmax_tile","attn_finalize","attn_tile_init","attn_prefill_fa_b","attn_prefill_fa_sc_b","write_kv_prefill","gqa_attn_prefill",
            "rep_penalty_b","rep_penalty_f32_b","pen_apply_b","pen_apply_f32_b","pen_count_add_b","pen_slot_reset_b","pen_prompt_set_b","logit_mask_b","logit_mask_f32_b","gqa_attn_splitk","gqa_attn_reduce","sample_b","sample_f32_b","concat_b",
            "gemm_binv_b","gemm_binv_f32_b","sample_prob_b","spec_verify_b","spec_verify_f32_b","spec_verify_realq_b","df2_topk20_dump_b","nll_b","argmax_f32_b","verify_chain_params_b",
            "gemm_mma_fp4_b","gemm_mma_fp8_b","gemm_mma_fp8_blk_b","fp8_repack_e_b","fp8_blk_scales_b","fp8_pack_b","fp8_quant_pack_b","gemm_fp8_prefill_e_b","gemm_mma_fp4_splitk_b","gemm_mma_splitk_reduce_b","gemm_dsv4_fp8_bsb","dequant_fp4_tiled_b","dequant_fp8_tiled_b","dequant_fp8_blk_tiled_b",
            "embed_gather_fp4_tiled_b","embed_gather_fp8_tiled_b","bw_read_b",
            "split_gdn_b","split_qkv_b",
            "moe_router_topk_b","moe_router_topk_sigmoid_b","moe_router_topk_sigmoid_f32_b","moe_experts_b","moe_experts_fp4_b","moe_shared_combine_b",
            "moe_gate_up_fp4_warp","moe_silu_b","moe_down_fp4_warp",
            "gemm_moe_mma_fp4","gemm_moe_mma_fp4_x2","gemm_moe_mma_fp4_u4","moe_combine_experts_b","moe_silu_bf16_b","moe_silu_fold_b","add_f32_bf16_b",
            "gemm_moe_mma_fp4_fold","moe_combine_experts_fold_b","moe_shared_slot_combine_b",
            "gemm_moe_mma_fp4_fold_u4","gemm_moe_mma_fp4_fold_x2","gemm_moe_mma_fp4_fold_rast",
            "gemm_moe_mma_fp4_fold_lb5","gemm_moe_mma_fp4_fold_lb4","gemm_moe_mma_fp4_fold_pdl",
            "e11_fill_x_b",
            "moe_count_b","moe_offsets_b","moe_scatter_b","moe_tilemap_b","moe_gather_x_b",
            "moe_offsets_fold_b","moe_tilemap_fold_b","moe_shared_scatter_b",
            "gemm_moe_grouped_mma_fp4","gemm_moe_grouped_mma_fp4_x2","gemm_moe_grouped_mma_fp4_u4","gemm_moe_grouped_mma_fp4_x4","moe_combine_grouped_b",
            "gemm_moe_grouped_mma_fp4_fold","gemm_moe_grouped_mma_fp4_x4_fold","moe_combine_grouped_fold_b",
            "write_kv_b_q4","write_kv_prefill_q4","compact_kv_q4","dequant_kv_q4","gqa_attn_splitk_q4",
            "gqa_attn_splitk_q4_gq","gqa_attn_splitk_gq","gqa_attn_verify_e","gqa_attn_verify_e_p3","gqa_attn_reduce_e",
            "write_kv_b_k8v4","write_kv_prefill_k8v4","compact_kv_k8v4","dequant_kv_k8v4",
            "gqa_attn_splitk_k8v4","gqa_attn_splitk_k8v4_gq",
            "write_kv_b_k8v8","write_kv_prefill_k8v8","compact_kv_k8v8","gqa_attn_verify_e_k8v8",
            "gqa_attn_splitk_k8v8","gqa_attn_splitk_k8v8_gq",
            "write_kv_b_tq","write_kv_prefill_tq","rotate_q_tq","compact_kv_tq","dequant_kv_tq",
            "dequant_kv_tq_full","gqa_attn_splitk_tq","gqa_attn_splitk_tq_gq","gqa_attn_splitk_tq_dbg_scores",
            "tp_mask_rows","tp_gate_copy_signal","tp_wait_add","tp_wait_add_g","tp_wait_add_4way","tp_reduce_resnorm_b",
            "tp_wait_maxloc_b","tp_wait_maxloc_g",
            "ids_advance_b","penalty_ring_push_b","penalty_window_b","seed_advance_b",
            "tp_bench_fill","tp_bench_validate","tp_bench_stall",
            "gdn_rollback_b",
            "hc_expand_b",
            "hc_norm_b",
            "silu_div_b",
            "hc_mix_b",
            "hc_inject_b",
            "rmsnorm_gated_sig_b",
            "ple_hash_b",
            "ple_ring_commit_b",
            "ple_gather_rows_b",
            "ple_dequant_rows_b",
            "ple_gate_b",
            "ple_dconv_decode_b",
            "ple_dconv_prefill_b",
            "ple_dconv_state_b",
            "ple_slot_copy_b",
            "hc_add_bcast_b",
            "qsa_key_write_b",
            "qsa_score_b",
            "qsa_block_keys_b",
            "qsa_score_prefill_b",
            "qsa_topk_b",
            "gqa_attn_sel_splitk",
            "gqa_attn_sel_splitk_k8v4",
            "gqa_attn_sel_prefill",
            "gqa_attn_sel_prefill2",
            "qsa_compact_b",
            "qsa_score_combine_b",
            "gptq_bf16_to_f32_b",
            "calib_profile_b",
            "gptq_absmax_b",
            "igs_hist_b",
            "gptq_absmax_f32_b",
            "gptq_hadamard16_b",
            "gptq_sweep_b",
            "gptq_gather_rows_b",
            "gptq_silu_mul_gu_b",
            "gptq_rotate_act_b",
            "gptq_scale_stats_f32_b",
            "gptq_scale_stats_bf16_b",
            "gptq_static_scales_b",
            "gptq_static_scales_hessian_b",
            "gptq_permute_w_b",
            "gptq_permute_h_b",
            "gptq_sweep_static_b",
            "kernel_build_id",
        ];
        dev.load_ptx(bptx, module, &bfnames)?;
        Self::assert_kernel_build_id(dev, module)?;
        let mut bk = HashMap::new();
        for n in &bfnames {
            if let Some(f) = dev.get_func(module, n) { bk.insert(n.to_string(), f); }
        }
        let gf = |n: &str| dev.get_func("gpu_kernels", n);
        let k = KernelTable {
            rmsnorm_qwen: gf("rmsnorm_qwen"), rmsnorm_gated: gf("rmsnorm_gated"),
            rmsnorm_perhead: gf("rmsnorm_perhead"), silu_mul: gf("silu_mul"),
            add_residual: gf("add_residual"), sigmoid_gate: gf("sigmoid_gate"),
            rope_rot_half: gf("rope_rot_half"), gqa_attention: gf("gqa_attention"),
            write_kv: gf("write_kv"), split_qgate: gf("split_qgate"),
            conv1d_step: gf("conv1d_step"), delta_step: gf("delta_step"),
            f32tobf16: gf("f32tobf16"), bf16tof32: gf("bf16tof32"),
            fused_residual_rmsnorm: gf("fused_residual_rmsnorm"),
            argmax_pass1: gf("argmax_pass1"), argmax_pass2: gf("argmax_pass2"),
        };
        Ok((k, bk))
    }

    /// (kv_quant, kv_tq, kv_tq_b3, kv_k8v4) from the env with the mutual-exclusion assert (the q4,
    /// TQ and k8v4 cache layouts are alternatives — mixing them would silently double-book the KV
    /// allocations).
    /// NOTE: kv_quant is presence-based (the pre-existing convention), but kv_tq and kv_k8v4 are
    /// VALUE-based — only `[kv-tq]=1` (b=2 K, the golden-anchored E4 layout) or `[kv-tq]=3`
    /// (b=3 K, the quality fix) enable the TQ path, and only `[kv-k8v4]=1` enables the k8v4
    /// path; `=0` (or unset) restores the current path byte-for-byte (the task's escape-hatch
    /// acceptance).
    pub(super) fn kv_modes_from_opts() -> (bool, bool, bool, bool, bool) {
        let q4 = crate::opts::var(crate::opt!("kv-quant")).is_ok();
        let (tq, b3) = match crate::opts::var(crate::opt!("kv-tq")).ok().as_deref() {
            Some("3") => (true, true),
            Some("1") => (true, false),
            _ => (false, false),
        };
        let k8v4 = crate::opts::var(crate::opt!("kv-k8v4")).ok().as_deref() == Some("1");
        let k8v8 = crate::opts::var(crate::opt!("kv-k8v8")).ok().as_deref() == Some("1");
        assert!(!(k8v8 && (q4 || tq || k8v4)),
                "[kv-k8v8] is mutually exclusive with the q4/TQ/k8v4 cache layouts (one KV \
                 layout per process; mixing would double-book the cache buffers)");
        assert!(!(q4 && tq) && !(q4 && k8v4) && !(tq && k8v4),
                "[kv-quant], [kv-tq] and [kv-k8v4] are mutually exclusive — the \
                 TurboQuant cache is the 3.5-bit successor of the q4 layout and k8v4 is the \
                 int8-K successor (EXPERT_KV_K8V4_RESPONSE.md); pick one");
        (q4, tq, b3, k8v4, k8v8)
    }

    pub(super) fn build_rope_tables(dev: &Arc<CudaDevice>, cfg: &crate::qwen::Config,
                         yarn_src: Option<crate::qwen::YarnSource>) -> anyhow::Result<(S, S)> {
        let rdim = cfg.rotary_dim;
        let half = rdim / 2;
        let theta = cfg.rope_theta;
        // E5 (512K mandate): `rope_yarn_factor > 1` switches the TRUNK tables to YaRN — the
        // same per-dim ramp math the dspark oracle ships (yaarn_freqs, validated against
        // vLLM's yarn_scaling_rope) — and extends the tables to factor × native rows (the
        // kernels index rows by ABSOLUTE position; a 512K serve on native 256K tables reads
        // past the end). mscale = 0.1·ln(factor)+1 folds into cos/sin exactly like the draft
        // and vLLM. factor 1.0 (default) leaves the legacy tables byte-identical.
        let factor = cfg.rope_yarn_factor;
        let max_pos = if factor > 1.0 {
            let rows = (cfg.yarn_native_window as f64 * factor as f64).ceil() as usize;
            // YARN1: this trunk's ONE boot line names the factor and where it came from.
            let src = match yarn_src {
                Some(crate::qwen::YarnSource::Flag) => "--rope-yarn-factor",
                Some(crate::qwen::YarnSource::Tp) => "shipped TpConfig",
                Some(crate::qwen::YarnSource::Config) | None => "config rope_scaling",
            };
            eprintln!("[rope] YaRN factor {factor} (from {src}): {rows} rows (native {}), mscale {:.4}",
                      cfg.yarn_native_window, crate::dspark::oracle::yaarn_mscale(factor));
            rows
        } else {
            cfg.max_position_embeddings
        };
        let mut cos_t = vec![0.0f32; max_pos * rdim];
        let mut sin_t = vec![0.0f32; max_pos * rdim];
        let (mscale, freqs) = if factor > 1.0 {
            // vLLM defaults beta_fast=32, beta_slow=1 (the draft artifact's values); the ramp
            // anchors at the NATIVE window and mscale has ONE formula in the tree (YARN1).
            (crate::dspark::oracle::yaarn_mscale(factor),
             crate::dspark::oracle::yaarn_freqs(rdim, theta, factor,
                                                cfg.yarn_native_window, 32, 1))
        } else {
            (1.0f32, Vec::new())
        };
        for p in 0..max_pos {
            let pf = p as f32;
            for i in 0..half {
                let f = if factor > 1.0 {
                    pf * freqs[i]
                } else {
                    pf * theta.powf(-(2.0 * i as f32) / rdim as f32)
                };
                let (mut c, mut s) = (f.cos(), f.sin());
                if factor > 1.0 { c *= mscale; s *= mscale; }
                cos_t[p * rdim + i] = c; sin_t[p * rdim + i] = s;
                cos_t[p * rdim + i + half] = c; sin_t[p * rdim + i + half] = s;
            }
        }
        let cos_table = dev.htod_sync_copy(&cos_t)?;
        let sin_table = dev.htod_sync_copy(&sin_t)?;
        Ok((cos_table, sin_table))
    }

    /// Upload the fixed TurboQuant tables once (the cos/sin-table pattern): one f32 buffer
    /// [Pi(128x128) | PiT | S | ST | cb2(4) | cb3(8) | qjl_scale] indexed by the TQ_TAB_* offsets
    /// in gpu_batch.cu. The bytes come from the reference goldens (src/tq_consts/, generated by
    /// /tmp/tq_ref2/golden.py: Π seed 1234, S seed 5678, deterministic Lloyd–Max codebooks);
    /// ST is the transpose of S computed here. `--probe-tq` verifies the embedded bytes are
    /// identical to the golden files (sha256) so the engine and the reference share constants.
    pub fn build_tq_tables(dev: &Arc<CudaDevice>) -> anyhow::Result<S> {
        const D: usize = 128;
        let as_f32 = |b: &[u8]| -> Vec<f32> {
            b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        let pi = as_f32(include_bytes!("../tq_consts/Pi.bin"));
        let pit = as_f32(include_bytes!("../tq_consts/Pit.bin"));
        let s = as_f32(include_bytes!("../tq_consts/S.bin"));
        let cb = as_f32(include_bytes!("../tq_consts/codebooks.bin"));   // cb1[0,2) cb2[2,6) cb3[6,14) cb4[14,30)
        let scale = as_f32(include_bytes!("../tq_consts/scale.bin"));
        assert_eq!(pi.len(), D * D, "Pi.bin must be 128x128 f32");
        assert_eq!(pit.len(), D * D, "Pit.bin must be 128x128 f32");
        assert_eq!(s.len(), D * D, "S.bin must be 128x128 f32");
        assert_eq!(cb.len(), 30, "codebooks.bin must hold 30 f32 centroids");
        assert_eq!(scale.len(), 1, "scale.bin must be one f32");
        let mut st = vec![0.0f32; D * D];
        for j in 0..D { for i in 0..D { st[j * D + i] = s[i * D + j]; } }
        let mut t: Vec<f32> = Vec::with_capacity(4 * D * D + 13);
        t.extend_from_slice(&pi);
        t.extend_from_slice(&pit);
        t.extend_from_slice(&s);
        t.extend_from_slice(&st);
        t.extend_from_slice(&cb[2..6]);   // cb2: the K channel (2-bit)
        t.extend_from_slice(&cb[6..14]);  // cb3: the V channel (3-bit)
        t.push(scale[0]);                 // sqrt(pi/2)/128
        assert_eq!(t.len(), 4 * D * D + 13, "TQ table layout: 4 matrices + 4 + 8 + 1");
        Ok(dev.htod_sync_copy(&t)?)
    }

    fn load_mtp_gpu(host: &crate::qwen::Model, dev: &Arc<CudaDevice>) -> anyhow::Result<Option<GpuMtpLayer>> {
        let mtp = match &host.mtp {
            Some(m) => {
                let up_f = |v: &[f32]| -> S { dev.htod_sync_copy(v).unwrap() };
                let up_b = |v: &[f32]| -> B {
                    let bv: Vec<half::bf16> = v.iter().map(|&x| half::bf16::from_f32(x)).collect();
                    dev.htod_sync_copy(&bv).unwrap()
                };
                if crate::boot_report::debug() {
                    println!("Loading MTP head...");
                }
                Some(GpuMtpLayer {
                    fc: W::Bf16(up_b(&m.fc)),
                    pre_fc_norm_hidden: up_f(&m.pre_fc_norm_hidden),
                    pre_fc_norm_embedding: up_f(&m.pre_fc_norm_embedding),
                    input_ln: up_f(&m.input_ln), post_ln: up_f(&m.post_ln),
                    fa: GpuFullAttn {
                        qkv: AttnIn::Split { q: W::Bf16(up_b(&m.q_proj)), k: W::Bf16(up_b(&m.k_proj)),
                                             v: W::Bf16(up_b(&m.v_proj)) },
                        o_proj: W::Bf16(up_b(&m.o_proj)), q_norm: up_f(&m.q_norm), k_norm: up_f(&m.k_norm), indexer: None,
                    },
                    mlp: Ffn::Dense(GpuMlp { gate: W::Bf16(up_b(&m.gate_proj)), up: W::Bf16(up_b(&m.up_proj)), down: W::Bf16(up_b(&m.down_proj)) }),
                    final_norm: up_f(&m.final_norm),
                    fc_sharded: false, attn_sharded: false, mlp_sharded: false, q4: None,
                })
            }
            None => { println!("No MTP head."); None }
        };
        Ok(mtp)
    }
}

/// S-B9-REL-PPL: runs `F` when dropped — on return, `?` AND unwind. The generic loader's shard
/// pipeline wraps its worker release in this so a panic in the shard loop cannot leave the scoped
/// workers parked on their condvar (`std::thread::scope` joins before it resumes an unwind: that
/// was the v0.7.3 `--perplexity` hang).
pub(crate) struct ReleaseWorkers<F: FnMut()>(pub F);
impl<F: FnMut()> Drop for ReleaseWorkers<F> {
    fn drop(&mut self) { (self.0)() }
}
