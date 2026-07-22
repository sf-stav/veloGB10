//! Isolate delta_step kernel against CPU recurrence with controlled inputs.
use gb10_inference::gpu::GpuModel;
use gb10_inference::qwen::Model;

fn l2norm(x: &mut [f32], eps: f32) {
    let s: f32 = x.iter().map(|v| v*v).sum();
    let inv = 1.0/(s+eps).sqrt();
    for v in x.iter_mut() { *v *= inv; }
}
fn softplus(x: f32) -> f32 { if x > 20.0 { x } else { (1.0+x.exp()).ln() } }
fn sigmoid(x: f32) -> f32 { 1.0/(1.0+(-x).exp()) }

#[test]
fn test_delta_step_isolated() {
    let model_path = std::env::var("GB10_TEST_MODEL")
        .expect("GB10_TEST_MODEL must point at model.safetensors (test fixture)");
    let host = Model::load(&model_path).unwrap();
    let gpu = GpuModel::new(&host).unwrap();
    let la = host.layers[0].linear_attn.as_ref().unwrap();
    let nh = host.config.lin_num_v_heads;
    let kd = host.config.lin_k_dim;
    let vd = host.config.lin_v_dim;

    // controlled q,k,v [nh*kd], b,a [nh]; real a_log,dt_bias
    let q: Vec<f32> = (0..nh*kd).map(|i| 0.01*((i as f32)-100.0)).collect();
    let k: Vec<f32> = (0..nh*kd).map(|i| 0.02*((i as f32)-200.0)).collect();
    let v: Vec<f32> = (0..nh*vd).map(|i| 0.005*((i as f32)-50.0)).collect();
    let b: Vec<f32> = (0..nh).map(|i| -0.3).collect();
    let a: Vec<f32> = (0..nh).map(|i| 0.4).collect();

    let qd = gpu.dev().htod_sync_copy(&q).unwrap();
    let kd_ = gpu.dev().htod_sync_copy(&k).unwrap();
    let vd_ = gpu.dev().htod_sync_copy(&v).unwrap();
    let bd = gpu.dev().htod_sync_copy(&b).unwrap();
    let ad = gpu.dev().htod_sync_copy(&a).unwrap();
    let alogd = gpu.dev().htod_sync_copy(&la.a_log).unwrap();
    let dtbd = gpu.dev().htod_sync_copy(&la.dt_bias).unwrap();
    let mut stated = gpu.dev().alloc_zeros::<f32>(nh*kd*vd).unwrap();
    let mut outd = gpu.dev().alloc_zeros::<f32>(nh*vd).unwrap();
    gpu.test_delta_step(&mut outd, &qd, &kd_, &vd_, &bd, &ad, &mut stated, nh, kd, vd, &alogd, &dtbd);
    let gpu_out = gpu.dev().dtoh_sync_copy(&outd).unwrap();

    // CPU recurrence
    let mut cpu_out = vec![0.0f32; nh*vd];
    for head in 0..nh {
        let mut qh = q[head*kd..head*kd+kd].to_vec();
        let mut kh = k[head*kd..head*kd+kd].to_vec();
        l2norm(&mut qh, 1e-6); l2norm(&mut kh, 1e-6);
        let scale = 1.0/(kd as f32).sqrt();
        for x in qh.iter_mut() { *x *= scale; }
        let beta = sigmoid(b[head]);
        let g = -la.a_log[head].exp() * softplus(a[head] + la.dt_bias[head]);
        let gt = g.exp();
        let mut S = vec![0.0f32; kd*vd]; // zero init
        for s in S.iter_mut() { *s *= gt; }
        let vh = &v[head*vd..head*vd+vd];
        let mut kv_mem = vec![0.0f32; vd];
        for bb in 0..vd { let mut m=0.0; for aa in 0..kd { m += S[aa*vd+bb]*kh[aa]; } kv_mem[bb]=m; }
        let mut delta = vec![0.0f32; vd];
        for bb in 0..vd { delta[bb] = (vh[bb]-kv_mem[bb])*beta; }
        for aa in 0..kd { for bb in 0..vd { S[aa*vd+bb] += kh[aa]*delta[bb]; } }
        for bb in 0..vd { let mut o=0.0; for aa in 0..kd { o += S[aa*vd+bb]*qh[aa]; } cpu_out[head*vd+bb]=o; }
    }
    let m = gpu_out.iter().zip(&cpu_out).map(|(x,y)|(x-y).abs()).fold(0.0f32,f32::max);
    println!("delta_step gpu-vs-cpu max abs diff: {:.6}", m);
    // show a few values
    println!("gpu[:4] = {:?}", &gpu_out[..4]);
    println!("cpu[:4] = {:?}", &cpu_out[..4]);
}
