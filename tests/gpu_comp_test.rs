//! Isolate GPU kernel bugs: compare each GPU op against the validated CPU qwen.rs op.
use gb10_inference::gpu::GpuModel;
use gb10_inference::qwen::Model;

#[test]
fn test_gpu_rmsnorm_vs_cpu() {
    let model_path = std::env::var("GB10_TEST_MODEL")
        .expect("GB10_TEST_MODEL must point at model.safetensors (test fixture)");
    let host = Model::load(&model_path).expect("load");
    let gpu = GpuModel::new(&host).expect("gpu");

    let h = host.config.hidden_size;
    // token 0 embed
    let tok = 760u32;
    let emb_host = host.embed(tok);
    let emb_dev = gpu.dev().htod_sync_copy(&emb_host).unwrap();

    // GPU rmsnorm with layer0 input_ln
    let ln = &host.layers[0].input_layernorm;
    let ln_dev = gpu.dev().htod_sync_copy(ln).unwrap();
    let mut out_dev = gpu.dev().alloc_zeros::<f32>(h).unwrap();
    gpu.rmsnorm_qwen(&mut out_dev, &emb_dev, &ln_dev, h, host.config.rms_eps);
    let gpu_out = gpu.dev().dtoh_sync_copy(&out_dev).unwrap();

    // CPU rmsnorm
    let mut cpu_out = vec![0.0f32; h];
    Model::rmsnorm(&mut cpu_out, &emb_host, ln, host.config.rms_eps);

    let mut mx = 0.0f32;
    for i in 0..h { mx = mx.max((gpu_out[i] - cpu_out[i]).abs()); }
    println!("rmsnorm gpu-vs-cpu max abs diff: {:.6}", mx);
}
