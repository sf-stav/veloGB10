//! Compare GPU mixers vs validated CPU qwen.rs mixers for layer 0 (linear) and layer 3 (full).
use gb10_inference::gpu::{GpuModel, Pool};
use gb10_inference::qwen::{Model, ModelState, LayerType};

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
fn test_gpu_mixers_vs_cpu() {
    let model_path = std::env::var("GB10_TEST_MODEL")
        .expect("GB10_TEST_MODEL must point at model.safetensors (test fixture)");
    let host = Model::load(&model_path).expect("load");
    let gpu = GpuModel::new(&host).expect("gpu");
    let h = host.config.hidden_size;
    let mut cstate = ModelState::new(&host);
    let mut pool = Pool::new(gpu.dev().clone());

    // token 0 embed + input layernorm
    let tok = 760u32;
    let emb = host.embed(tok);
    let mut normed = vec![0.0f32; h];
    Model::rmsnorm(&mut normed, &emb, &host.layers[0].input_layernorm, host.config.rms_eps);
    let normed_dev = gpu.dev().htod_sync_copy(&normed).unwrap();

    // ---- layer 0 linear_attn ----
    if host.layers[0].layer_type == LayerType::LinearAttention {
        // CPU
        let mut mixer_cpu = vec![0.0f32; h];
        let la = host.layers[0].linear_attn.as_ref().unwrap();
        let conv = &mut cstate.conv_states[0].clone();
        let ss = &mut cstate.s_states[0].clone();
        host.linear_attn_forward(&mut mixer_cpu, &normed, la, conv, ss);
        // GPU
        let conv0 = gpu.dev().htod_sync_copy(&vec![0.0f32; conv.len()]).unwrap();
        let s0 = gpu.dev().htod_sync_copy(&vec![0.0f32; ss.len()]).unwrap();
        let mut convd = conv0; let mut sd = s0;
        let core_dev = gpu.linear_attn_core(&mut pool, &normed_dev, gpu.layer(0).la.as_ref().unwrap(), &mut convd, &mut sd);
        let core_gpu = gpu.dev().dtoh_sync_copy(&core_dev).unwrap();
        // CPU core: replicate up to delta (CPU linear_attn_forward computes core internally; recompute)
        let mut core_cpu = vec![0.0f32; host.config.lin_num_v_heads * host.config.lin_v_dim];
        {
            // reuse host helper by calling a CPU core path: compute via linear_attn_forward is opaque,
            // so compute core manually mirroring qwen.rs delta (single step, zero state).
            let cfg = &host.config;
            let nh = cfg.lin_num_v_heads; let kd = cfg.lin_k_dim; let vd = cfg.lin_v_dim;
            let key_dim = cfg.key_dim(); let value_dim = cfg.value_dim();
            let conv_dim = key_dim*2 + value_dim;
            // qkv = in_proj_qkv @ normed ; conv (state zeros)
            let mut qkv = vec![0.0f32; conv_dim];
            Model::matvec(&mut qkv, &la.in_proj_qkv, &normed, conv_dim, h);
            // conv (zero state): for each c, out=silu(w[k-1]*x) since state zero
            let ck = cfg.conv_kernel;
            for c in 0..conv_dim { let acc = la.conv1d[c*ck+ck-1]*qkv[c]; qkv[c] = acc/(1.0+(-acc).exp()); }
            // b,a
            let mut b = vec![0.0f32; nh]; let mut a = vec![0.0f32; nh];
            Model::matvec(&mut b, &la.in_proj_b, &normed, nh, h);
            Model::matvec(&mut a, &la.in_proj_a, &normed, nh, h);
            // per-head delta
            for head in 0..nh {
                let mut qh = qkv[head*kd..head*kd+kd].to_vec();
                let mut kh = qkv[key_dim+head*kd..key_dim+head*kd+kd].to_vec();
                let vh = &qkv[2*key_dim+head*vd..2*key_dim+head*vd+vd];
                let mut sq=0.0f32; for x in &qh {sq+=x*x;} let qn=1.0/(sq+1e-6).sqrt(); for x in qh.iter_mut(){*x*=qn;}
                let mut sk=0.0f32; for x in &kh {sk+=x*x;} let kn=1.0/(sk+1e-6).sqrt(); for x in kh.iter_mut(){*x*=kn;}
                let scale=1.0/(kd as f32).sqrt(); for x in qh.iter_mut(){*x*=scale;}
                let beta=1.0/(1.0+(-b[head]).exp());
                let sp=if a[head]+la.dt_bias[head]>20.0 {a[head]+la.dt_bias[head]} else {(1.0+(a[head]+la.dt_bias[head]).exp()).ln()};
                let gt=(-la.a_log[head].exp()*sp).exp();
                let mut S=vec![0.0f32;kd*vd]; for s in S.iter_mut(){*s*=gt;}
                let mut km=vec![0.0f32;vd];
                for bb in 0..vd { let mut m=0.0; for aa in 0..kd {m+=S[aa*vd+bb]*kh[aa];} km[bb]=m; }
                let mut delta=vec![0.0f32;vd];
                for bb in 0..vd { delta[bb]=(vh[bb]-km[bb])*beta; }
                for aa in 0..kd { for bb in 0..vd { S[aa*vd+bb]+=kh[aa]*delta[bb]; } }
                for bb in 0..vd { let mut o=0.0; for aa in 0..kd {o+=S[aa*vd+bb]*qh[aa];} core_cpu[head*vd+bb]=o; }
            }
        }
        println!("linear_attn CORE gpu-vs-cpu max abs diff: {:.6}", maxdiff(&core_gpu, &core_cpu));
        let mixer_dev = gpu.linear_attn(&mut pool, &normed_dev, gpu.layer(0).la.as_ref().unwrap(), &mut convd, &mut sd);
        let mixer_gpu = gpu.dev().dtoh_sync_copy(&mixer_dev).unwrap();
        println!("linear_attn mixer gpu-vs-cpu max abs diff: {:.6}", maxdiff(&mixer_gpu, &mixer_cpu));
        println!("  gpu[:6] = {:?}", &mixer_gpu[..6]);
        println!("  cpu[:6] = {:?}", &mixer_cpu[..6]);
        let ratio: Vec<f32> = mixer_gpu.iter().zip(&mixer_cpu).take(6).map(|(g,c)| if c.abs()>1e-6 {g/c} else {0.0}).collect();
        println!("  ratio[:6] = {:?}", ratio);
    }

    // ---- layer 3 full_attn (pos 0) ----
    if host.layers[3].layer_type == LayerType::FullAttention {
        let mut normed3 = vec![0.0f32; h];
        // reuse a normed input (doesn't need to match layer3 real input for component check)
        Model::rmsnorm(&mut normed3, &emb, &host.layers[3].input_layernorm, host.config.rms_eps);
        let normed3_dev = gpu.dev().htod_sync_copy(&normed3).unwrap();
        let fa = host.layers[3].full_attn.as_ref().unwrap();
        let rdim = host.config.rotary_dim;
        let (cos, sin) = rope(&host.config, 0);
        let cosd = gpu.dev().htod_sync_copy(&cos).unwrap();
        let sind = gpu.dev().htod_sync_copy(&sin).unwrap();
        // CPU: need k/v caches; build fresh empty then call full_attn_forward at pos 0
        let mut kc: Vec<Vec<f32>> = (0..host.config.num_kv_heads).map(|_| vec![]).collect();
        let mut vc: Vec<Vec<f32>> = (0..host.config.num_kv_heads).map(|_| vec![]).collect();
        let mut mixer_cpu = vec![0.0f32; h];
        host.full_attn_forward(&mut mixer_cpu, &normed3, fa, 0, &mut kc, &mut vc, &cos, &sin);
        // GPU: allocate empty caches sized to stride
        let stride = host.config.max_position_embeddings;
        let bytes = host.config.num_kv_heads * host.config.head_dim * stride;
        let mut kcd = gpu.dev().alloc_zeros::<f32>(bytes).unwrap();
        let mut vcd = gpu.dev().alloc_zeros::<f32>(bytes).unwrap();
        let mixer_dev = gpu.full_attn(&mut pool, &normed3_dev, gpu.layer(3).fa.as_ref().unwrap(), 0, &mut kcd, &mut vcd, &cosd, &sind);
        let mixer_gpu = gpu.dev().dtoh_sync_copy(&mixer_dev).unwrap();
        println!("full_attn mixer gpu-vs-cpu max abs diff: {:.6}", maxdiff(&mixer_gpu, &mixer_cpu));
    }

    // ---- mlp layer 0 ----
    {
        let mut mlp_cpu = host.mlp_forward(&normed, &host.layers[0].mlp);
        let mlp_dev = gpu.mlp(&mut pool, &normed_dev, &gpu.layer(0).mlp);
        let mlp_gpu = gpu.dev().dtoh_sync_copy(&mlp_dev).unwrap();
        println!("mlp gpu-vs-cpu max abs diff: {:.6}", maxdiff(&mlp_gpu, &mlp_cpu));
        let _ = &mut mlp_cpu;
    }
}

fn rope(cfg: &gb10_inference::qwen::Config, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let rdim = cfg.rotary_dim; let half = rdim/2; let theta = cfg.rope_theta;
    let mut cos = vec![0.0f32; rdim]; let mut sin = vec![0.0f32; rdim];
    for i in 0..half {
        let f = (pos as f32)*theta.powf(-(2.0*i as f32)/rdim as f32);
        cos[i] = f.cos(); sin[i] = f.sin(); cos[i+half] = cos[i]; sin[i+half] = sin[i];
    }
    (cos, sin)
}
