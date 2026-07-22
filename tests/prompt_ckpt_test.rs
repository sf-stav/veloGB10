//! The prompt-checkpoint round trip must be EXACT.
//!
//! `admit()` snapshots the GDN recurrent state at the message boundary, lets decode move it, and later
//! winds it back to serve the next turn of the conversation. If that round trip loses anything, the
//! model quietly answers from a corrupted state and no gate we have would notice — the output would
//! still be fluent, just wrong.
//!
//! Chunking is held CONSTANT here on purpose. Reusing a prefix necessarily re-chunks the prefill, and
//! our prefill runs on cuBLAS, which picks a different kernel per shape (AGENTS.md §2.4) — so a cached
//! turn is not bit-identical to a cold one, and that is inherent to prefix caching, not to this
//! snapshot. This test isolates the ONE thing that must be exact: snapshot -> mutate -> restore is a
//! no-op. Both paths below prefill the identical two chunks; the only difference is that path B moves
//! the state in between and restores it.
use gb10_inference::gpu::{GpuModel, Pool};
use gb10_inference::qwen::Model;

/// PANICS if no model is found. A test that silently skips is a test that passes while proving
/// nothing — the first cut of this looked for `model.safetensors`, the bf16 models are sharded, so it
/// "passed" in 0.00s without loading anything at all.
fn model_dir() -> String {
    let base = std::env::var("GB10_TEST_BF16_DIR")
        .expect("GB10_TEST_BF16_DIR must point at the dir holding {2b,0.8b}-bf16 (test fixture)");
    for d in [format!("{base}/2b-bf16"), format!("{base}/0.8b-bf16")] {
        let p = std::path::Path::new(&d);
        let has_weights = p.is_dir() && std::fs::read_dir(p).map(|mut it|
            it.any(|e| e.map(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))
                        .unwrap_or(false))).unwrap_or(false);
        if has_weights { return d; }
    }
    panic!("no bf16 model found under $GB10_TEST_BF16_DIR — this test must not silently skip");
}

#[test]
fn prompt_checkpoint_round_trip_is_bit_exact() {
    let dir = model_dir();
    let model = Model::load(&dir).expect("load model");
    let gpu = GpuModel::new(&model).expect("gpu load");
    let mut pool = Pool::new(gpu.dev().clone());

    let kv_stride = 2048usize;
    // slot 0 = the lane; slot 1 = the checkpoint. (No KV for slot 1 — the snapshot never needs it.)
    let mut state = gpu.new_batch_state(1, 2, kv_stride);
    gpu.dev().synchronize().unwrap();

    // An arbitrary but deterministic token sequence, split at C exactly as admit() splits at the
    // message boundary.
    let n = 384usize;
    let c = 256usize;
    let toks: Vec<u32> = (0..n).map(|i| ((i * 7919 + 13) % 30000) as u32).collect();

    // --- Path A: prefill [0,C), snapshot, prefill [C,N). No mutation in between.
    gpu.zero_slot_state(&mut state, 0, kv_stride);
    let (_, ha) = gpu.prefill_batch(&mut pool, &toks[..c], &mut state, 0, kv_stride, 0);
    pool.release_bf16(ha, model.config.hidden_size * c);
    gpu.copy_gdn_slot(&state, 0, 1);
    let (tok_a, _) = gpu.prefill_batch(&mut pool, &toks[c..], &mut state, 0, kv_stride, c);

    // --- Path B: identical, but MOVE the state before restoring it — this is what decode does to a
    // lane between the snapshot and the next turn.
    gpu.zero_slot_state(&mut state, 0, kv_stride);
    let (_, hb) = gpu.prefill_batch(&mut pool, &toks[..c], &mut state, 0, kv_stride, 0);
    pool.release_bf16(hb, model.config.hidden_size * c);
    gpu.copy_gdn_slot(&state, 0, 1);                       // snapshot at the boundary
    let (_, hm) = gpu.prefill_batch(&mut pool, &toks[c..], &mut state, 0, kv_stride, c);  // move it
    pool.release_bf16(hm, model.config.hidden_size * (n - c));
    gpu.copy_gdn_slot(&state, 1, 0);                       // wind it back
    let (tok_b, _) = gpu.prefill_batch(&mut pool, &toks[c..], &mut state, 0, kv_stride, c);

    assert_eq!(tok_a, tok_b,
               "prompt-checkpoint round trip is LOSSY: the same chunk, prefilled from a restored \
                state, predicted a different token ({tok_a} vs {tok_b}). snapshot -> mutate -> \
                restore must be a no-op.");
}
