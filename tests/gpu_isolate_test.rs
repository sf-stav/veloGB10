//! Isolate conv1d / delta_step / rmsnorm_gated kernels with controlled inputs.
use gb10_inference::gpu::GpuModel;
use gb10_inference::qwen::Model;

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

#[test]
fn test_conv1d_and_gated() {
    let model_path = std::env::var("GB10_TEST_MODEL")
        .expect("GB10_TEST_MODEL must point at model.safetensors (test fixture)");
    let host = Model::load(&model_path).unwrap();
    let gpu = GpuModel::new(&host).unwrap();
    let la = host.layers[0].linear_attn.as_ref().unwrap();
    let conv_dim = host.config.key_dim()*2 + host.config.value_dim();
    let ck = host.config.conv_kernel;

    // conv1d: x arbitrary, state zeros
    let x_host: Vec<f32> = (0..conv_dim).map(|i| 0.1*(i as f32)-3.0).collect();
    let state_host = vec![0.0f32; conv_dim*ck];
    let mut x_dev = gpu.dev().htod_sync_copy(&x_host).unwrap();
    let mut state_dev = gpu.dev().htod_sync_copy(&state_host).unwrap();
    let w_dev = gpu.dev().htod_sync_copy(&la.conv1d).unwrap();
    gpu.test_conv1d(&mut x_dev, &mut state_dev, &w_dev, conv_dim, ck);
    let x_gpu = gpu.dev().dtoh_sync_copy(&x_dev).unwrap();
    // CPU expected
    let mut st = vec![0.0f32; conv_dim*ck];
    let mut xexp = x_host.clone();
    for c in 0..conv_dim {
        for j in 1..ck { st[c*ck+j-1]=st[c*ck+j]; }
        st[c*ck+ck-1]=x_host[c];
        let mut acc=0.0; for j in 0..ck { acc += la.conv1d[c*ck+j]*st[c*ck+j]; }
        xexp[c]=silu(acc);
    }
    let m = x_gpu.iter().zip(&xexp).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
    println!("conv1d gpu-vs-cpu max abs diff: {:.6}", m);

    // rmsnorm_gated: synthetic core, z, weight
    let vd = host.config.lin_v_dim;
    let core: Vec<f32> = (0..vd).map(|i| (i as f32)*0.01-0.5).collect();
    let z: Vec<f32> = (0..vd).map(|i| 0.2*((i as f32)-30.0)).collect();
    let core_dev = gpu.dev().htod_sync_copy(&core).unwrap();
    let z_dev = gpu.dev().htod_sync_copy(&z).unwrap();
    let norm_dev = gpu.dev().htod_sync_copy(&la.norm).unwrap();
    let mut out_dev = gpu.dev().alloc_zeros::<f32>(vd).unwrap();
    gpu.rmsnorm_gated(&mut out_dev, &core_dev, &z_dev, &norm_dev, vd, host.config.rms_eps);
    let out_gpu = gpu.dev().dtoh_sync_copy(&out_dev).unwrap();
    let mut cpu=vec![0.0f32;vd];
    Model::rmsnorm_gated(&mut cpu, &core, &z, &la.norm, host.config.rms_eps);
    let m2 = out_gpu.iter().zip(&cpu).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
    println!("rmsnorm_gated gpu-vs-cpu max abs diff: {:.6}", m2);
}
