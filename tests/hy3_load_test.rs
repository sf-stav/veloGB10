//! Hy3 (hy_v3) bring-up gate: load a QUANTIZED tiny hy_v3 fixture and run a complete forward.
//!
//! Fixture (make it with the repo tooling, see below): 3 layers (layer 0 dense, layers 1-2 MoE
//! with the sigmoid+bias router, renorm x 2.826, UNGATED shared expert), 2 Q / 1 KV heads at
//! head_dim 128, qk_norm, vocab 256, plus an MTP block at model.layers.3.* that the loader must
//! IGNORE without failing. Quantized with the serving recipe `all,-router,-embed,-lmhead`.
//!
//!   python3 scripts/gen_hy3_fixture.py /tmp/hy3_p2a
//!   ./target/release/gb10_inference --quantize --model-dir /tmp/hy3_p2a --out /tmp/hy3_p2a_q \
//!       --recipe all,-router,-embed,-lmhead
//!   GB10_HY3_FIXTURE=/tmp/hy3_p2a_q cargo test --release --test hy3_load_test
//!
//! The MoE router math itself is gated separately (scripts/verify_hy3_moe.py over --probe-moe,
//! and tests/hy3_router_test.rs for the hand-computed formula).

use gb10_inference::gpu::{GpuModel, Pool};
use gb10_inference::qwen::Family;

fn fixture() -> String {
    std::env::var("GB10_HY3_FIXTURE")
        .expect("GB10_HY3_FIXTURE must point at the QUANTIZED tiny hy_v3 fixture \
                 (run scripts/gen_hy3_fixture.py + --quantize; see test header)")
}

/// Prefill + greedy decode, driven twice from scratch; must complete and be deterministic.
#[test]
fn hy3_loads_and_forwards() {
    let dir = fixture();
    let (gpu, cfg) = GpuModel::load_from_dir(&dir).expect("gpu load");

    // Loader shape: family, layers, MTP skipped, fp32-logits flag honored.
    assert_eq!(cfg.family, Family::HyV3);
    assert_eq!(cfg.num_layers, 3);
    assert_eq!(cfg.head_dim, 128);
    assert!(cfg.qk_norm && cfg.router_sigmoid && cfg.router_expert_bias && cfg.lm_head_fp32);
    assert!(!cfg.is_moe_layer(0) && cfg.is_moe_layer(1) && cfg.is_moe_layer(2));
    assert!(!gpu.mtp_present(), "the model.layers.3.* MTP block must NOT be loaded");

    let kv_stride = 64;
    let mut pool = Pool::new(gpu.dev().clone());
    let mut state = gpu.new_batch_state(1, 1, kv_stride);
    let mut bufs = gpu.new_decode_buffers(1);
    let h = cfg.hidden_size;
    let prompt: Vec<u32> = vec![1, 7, 42, 100, 200, 255];

    let run_once = |gpu: &GpuModel, pool: &mut Pool, bufs: &mut gb10_inference::gpu::DecodeBuffers,
                    state: &mut gb10_inference::gpu::BatchGpuState| -> Vec<u32> {
        gpu.zero_slot_state(state, 0, kv_stride);
        let (first, h0) = gpu.prefill_batch(pool, &prompt, state, 0, kv_stride, 0);
        pool.release_bf16(h0, h * prompt.len());
        assert!(first < cfg.vocab_size as u32, "prefill first token out of vocab");
        let mut out = vec![first];
        let mut cur = first;
        for step in 0..4 {
            let pos = prompt.len() + step;
            gpu.dev().htod_sync_copy_into(&[cur as i32], &mut bufs.tokens_dev).unwrap();
            gpu.dev().htod_sync_copy_into(&[pos as i32], &mut bufs.pos_dev).unwrap();
            gpu.dev().synchronize().unwrap();
            let next = gpu.forward_decode(pool, bufs, state, kv_stride, pos + 1, 1);
            cur = next[0];
            assert!(cur < cfg.vocab_size as u32, "decode token out of vocab");
            out.push(cur);
        }
        out
    };

    let a = run_once(&gpu, &mut pool, &mut bufs, &mut state);
    let b = run_once(&gpu, &mut pool, &mut bufs, &mut state);
    assert_eq!(a, b, "greedy prefill+decode must be deterministic across identical runs");
    println!("HY3_FORWARD_OK tokens={:?}", a);
}

/// Long-prompt prefill (n >= PF_MIN = 128): routes attention through the TILED (cuBLAS)
/// prefill path — the scalar gqa_attn_prefill is only used for short chunks. Must complete
/// and be deterministic at head_dim 128.
#[test]
fn hy3_tiled_prefill_forward() {
    let dir = fixture();
    let (gpu, cfg) = GpuModel::load_from_dir(&dir).expect("gpu load");
    let kv_stride = 512;
    let mut pool = Pool::new(gpu.dev().clone());
    let mut state = gpu.new_batch_state(1, 1, kv_stride);
    // 160 deterministic tokens in-vocab (vocab 256).
    let toks: Vec<u32> = (0..160u32).map(|i| (i * 37 + 11) % 256).collect();
    let a = gpu.window_argmax(&mut pool, &mut state, &toks, kv_stride);
    let b = gpu.window_argmax(&mut pool, &mut state, &toks, kv_stride);
    assert_eq!(a.len(), toks.len());
    assert_eq!(a, b, "tiled prefill must be deterministic");
    assert!(a.iter().all(|&t| t < cfg.vocab_size as u32));
    println!("HY3_TILED_PREFILL_OK n={}", toks.len());
}
