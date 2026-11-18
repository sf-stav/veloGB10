//! `impl GpuModel`: accessors and the original f32 single-token forward path (cuBLAS gemv + elementwise kernels),
//! `forward_token*`, `full_attn`, `linear_attn`, `logits`, `argmax_gpu`, `new_state*`. Map: `mod.rs`.

use super::*;

impl GpuModel {
    pub fn dev(&self) -> &Arc<CudaDevice> { &self.dev }
    pub fn cfg(&self) -> &Config { &self.cfg }
    /// Layer count (PP-prefill and probes need the range bounds without touching `layers`).
    pub fn nlayers(&self) -> usize { self.layers.len() }
    pub fn layer(&self, li: usize) -> &GpuLayer { &self.layers[li] }

    /// bf16 GEMM: out[outn] = W_bf16[outn,inn] @ x_f32[inn], compute in f32, output f32.
    fn gemm_bf16(&self, pool: &mut Pool, w: &B, x: &S, out: &mut S, inn: usize, outn: usize) {
        let xb = pool.get_bf16(inn);
        let mut yb = pool.get_bf16(outn);
        unsafe { self.k.f32tobf16.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(inn),block_dim:(256,1,1),shared_mem_bytes:0},
            (*xb.device_ptr() as u64, d(x), inn as i32)).unwrap(); }
        let cfg = cudarc::cublas::GemmConfig::<half::bf16> {
            transa: OP::CUBLAS_OP_T, transb: OP::CUBLAS_OP_N,
            m: outn as i32, n: 1, k: inn as i32,
            alpha: half::bf16::from_f32(1.0), lda: inn as i32,
            ldb: inn as i32, beta: half::bf16::from_f32(0.0), ldc: outn as i32,
        };
        unsafe { self.blas.gemm(cfg, w, &xb, &mut yb).expect("gemm bf16"); }
        unsafe { self.k.bf16tof32.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(outn),block_dim:(256,1,1),shared_mem_bytes:0},
            (d(out), *yb.device_ptr() as u64, outn as i32)).unwrap(); }
        pool.release_bf16(xb, inn);
        pool.release_bf16(yb, outn);
    }

    pub fn rmsnorm_qwen(&self, out: &S, x: &S, w: &S, n: usize, eps: f32) {
        unsafe { self.k.rmsnorm_qwen.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(1,1,1),block_dim:(n as u32,1,1),shared_mem_bytes:(n*4) as u32},
            (d(out), d(x), d(w), n as i32, fbits(eps))).unwrap(); }
    }
    pub fn rmsnorm_gated(&self, out: &S, x: &S, z: &S, w: &S, m: usize, eps: f32) {
        unsafe { self.k.rmsnorm_gated.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(1,1,1),block_dim:(m as u32,1,1),shared_mem_bytes:(m*4) as u32},
            (d(out), d(x), d(z), d(w), m as i32, fbits(eps))).unwrap(); }
    }
    pub fn test_conv1d(&self, x: &S, state: &mut S, w: &S, conv_dim: usize, k: usize) {
        unsafe { self.k.conv1d_step.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(conv_dim),block_dim:(256,1,1),shared_mem_bytes:0},
            (d(x), d(state), d(w), conv_dim as i32, k as i32)).unwrap(); }
    }
    pub fn test_delta_step(&self, out: &mut S, q: &S, k: &S, v: &S, b: &S, a: &S,
                           state: &mut S, nh: usize, kd: usize, vd: usize, a_log: &S, dt_bias: &S) {
        let smem = ((kd + 4*vd + kd) as u32)*4;
        unsafe { self.k.delta_step.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(nh as u32,1,1),block_dim:(kd as u32,1,1),shared_mem_bytes:smem},
            (d(out), d(q), d(k), d(v), d(b), d(a), d(state),
             nh as i32, kd as i32, vd as i32, d(a_log), d(dt_bias))).unwrap(); }
    }

    /// forward one token; `hidden` becomes the residual (consumed). Returns final-normed hidden.
    pub fn forward_token(&self, pool: &mut Pool, hidden: S, pos: usize, state: &mut GpuState, cos: &S, sin: &S) -> S {
        self.forward_token_inner(pool, hidden, pos, state, cos, sin, None)
    }
    pub fn forward_token_captured(&self, pool: &mut Pool, hidden: S, pos: usize, state: &mut GpuState, cos: &S, sin: &S,
                                  caps: &mut Vec<Vec<f32>>) -> S {
        self.forward_token_inner(pool, hidden, pos, state, cos, sin, Some(caps))
    }
    fn forward_token_inner(&self, pool: &mut Pool, hidden: S, pos: usize, state: &mut GpuState, cos: &S, sin: &S,
                           mut caps: Option<&mut Vec<Vec<f32>>>) -> S {
        let h = self.cfg.hidden_size;
        let residual = hidden;
        let normed = pool.get(h);
        for (li, layer) in self.layers.iter().enumerate() {
            self.rmsnorm_qwen(&normed, &residual, &layer.input_ln, h, self.cfg.rms_eps);
            let mixer = match layer.layer_type {
                LayerType::LinearAttention => self.linear_attn(pool, &normed, layer.la.as_ref().unwrap(),
                    state.conv_state[li].as_mut().unwrap(), state.s_state[li].as_mut().unwrap()),
                LayerType::FullAttention => self.full_attn(pool, &normed, layer.fa.as_ref().unwrap(), pos,
                    state.k_cache[li].as_mut().unwrap(), state.v_cache[li].as_mut().unwrap(), cos, sin),
            };
            // fused: residual += mixer; then normed = rmsnorm(residual, post_ln)
            unsafe { self.k.fused_residual_rmsnorm.clone().unwrap().launch_on_stream(&self.stream,
                LaunchConfig{grid_dim:(1,1,1),block_dim:(h as u32,1,1),shared_mem_bytes:(h*4) as u32},
                (d(&normed), d(&residual), d(&mixer), d(&layer.post_ln), h as i32, fbits(self.cfg.rms_eps))).unwrap(); }
            let mlp_out = match &layer.mlp {
                Ffn::Dense(m) => self.mlp(pool, &normed, m),
                Ffn::Moe(_) => panic!("legacy single-token forward does not support MoE (use the batched path)"),
            };
            unsafe { self.k.add_residual.clone().unwrap().launch_on_stream(&self.stream,
                LaunchConfig{grid_dim:grid(h),block_dim:(256,1,1),shared_mem_bytes:0},
                (d(&residual), d(&residual), d(&mlp_out), h as i32)).unwrap(); }
            if let Some(c) = caps.as_mut() {
                // TRUNCATE to h. Pool buffers are power-of-two BUCKETS now, so a dtoh of the whole
                // slice can hand back more elements than were asked for (h=5120 on 27B lands in an
                // 8192 bucket). Anything comparing these captures by length would silently disagree.
                c.push({
                    self.sync_stream();
                    let mut v = self.dev.dtoh_sync_copy(&residual).unwrap();
                    v.truncate(h);
                    v
                });
            }
            pool.release(mixer, h); pool.release(mlp_out, h);
        }
        let out = pool.get(h);
        self.rmsnorm_qwen(&out, &residual, &self.final_norm, h, self.cfg.rms_eps);
        pool.release(residual, h); pool.release(normed, h);
        out
    }

    pub fn mlp(&self, pool: &mut Pool, x: &S, mlp: &GpuMlp) -> S {
        let (im, h) = (self.cfg.intermediate_size, self.cfg.hidden_size);
        let mut gate = pool.get(im);
        let mut upb = pool.get(im);
        self.gemm_bf16(pool, mlp.gate.bf16(), x, &mut gate, h, im);
        self.gemm_bf16(pool, mlp.up.bf16(), x, &mut upb, h, im);
        unsafe { self.k.silu_mul.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(im),block_dim:(256,1,1),shared_mem_bytes:0},
            (d(&gate), d(&gate), d(&upb), im as i32)).unwrap(); }
        let mut out = pool.get(h);
        self.gemm_bf16(pool, mlp.down.bf16(), &gate, &mut out, im, h);
        pool.release(gate, im); pool.release(upb, im);
        out
    }

    pub fn full_attn(&self, pool: &mut Pool, hidden: &S, fa: &GpuFullAttn, pos: usize,
                 kc: &mut S, vc: &mut S, cos: &S, sin: &S) -> S {
        let cfg = &self.cfg;
        let (h, nh, nkv, hd, rdim) = (cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim);
        let stride = state_stride(cfg);
        let mut qg = pool.get(nh*hd*2);
        let mut k = pool.get(nkv*hd);
        let mut v = pool.get(nkv*hd);
        let q = pool.get(nh*hd);
        let gate = pool.get(nh*hd);
        let (qp, kp, vp) = match &fa.qkv {
            AttnIn::Split { q, k, v } => (q, k, v),
            AttnIn::Fused(_) => panic!("legacy single-stream path cannot use a fused (quantized) model"),
        };
        self.gemm_bf16(pool, qp.bf16(), hidden, &mut qg, h, nh*hd*2);
        self.gemm_bf16(pool, kp.bf16(), hidden, &mut k, h, nkv*hd);
        self.gemm_bf16(pool, vp.bf16(), hidden, &mut v, h, nkv*hd);
        unsafe { self.k.split_qgate.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(nh*hd),block_dim:(256,1,1),shared_mem_bytes:0},
            (d(&q), d(&gate), d(&qg), nh as i32, hd as i32)).unwrap(); }
        unsafe { self.k.rmsnorm_perhead.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(nh as u32,1,1),block_dim:(hd as u32,1,1),shared_mem_bytes:(hd*4) as u32},
            (d(&q), d(&q), d(&fa.q_norm), nh as i32, hd as i32, fbits(cfg.rms_eps))).unwrap(); }
        unsafe { self.k.rmsnorm_perhead.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(nkv as u32,1,1),block_dim:(hd as u32,1,1),shared_mem_bytes:(hd*4) as u32},
            (d(&k), d(&k), d(&fa.k_norm), nkv as i32, hd as i32, fbits(cfg.rms_eps))).unwrap(); }
        unsafe { self.k.rope_rot_half.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(nh*(rdim/2)),block_dim:(256,1,1),shared_mem_bytes:0},
            (d(&q), d(cos), d(sin), nh as i32, hd as i32, rdim as i32)).unwrap(); }
        unsafe { self.k.rope_rot_half.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(nkv*(rdim/2)),block_dim:(256,1,1),shared_mem_bytes:0},
            (d(&k), d(cos), d(sin), nkv as i32, hd as i32, rdim as i32)).unwrap(); }
        unsafe { self.k.write_kv.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(nkv*hd),block_dim:(256,1,1),shared_mem_bytes:0},
            (d(kc), d(vc), d(&k), d(&v), pos as i32, stride as i32, nkv as i32, hd as i32)).unwrap(); }
        let attn = pool.get(nh*hd);
        let scale = 1.0f32/(hd as f32).sqrt();
        let smem = ((pos+1) as u32 + hd as u32)*4;
        unsafe { self.k.gqa_attention.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(nh as u32,1,1),block_dim:(hd as u32,1,1),shared_mem_bytes:smem},
            (d(&attn), d(&q), d(kc), d(vc), (pos+1) as i32, stride as i32, nh as i32, nkv as i32, hd as i32, fbits(scale))).unwrap(); }
        unsafe { self.k.sigmoid_gate.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(nh*hd),block_dim:(256,1,1),shared_mem_bytes:0},
            (d(&attn), d(&gate), (nh*hd) as i32)).unwrap(); }
        let mut out = pool.get(h);
        self.gemm_bf16(pool, fa.o_proj.bf16(), &attn, &mut out, nh*hd, h);
        pool.release(qg, nh*hd*2); pool.release(k, nkv*hd); pool.release(v, nkv*hd);
        pool.release(q, nh*hd); pool.release(gate, nh*hd); pool.release(attn, nh*hd);
        out
    }

    pub fn linear_attn(&self, pool: &mut Pool, hidden: &S, la: &GpuLinearAttn, conv_state: &mut S, s_state: &mut S) -> S {
        let cfg = &self.cfg;
        let (h, vd, nh) = (cfg.hidden_size, cfg.lin_v_dim, cfg.lin_num_v_heads);
        let value_dim = cfg.value_dim();
        let core = self.linear_attn_core(pool, hidden, la, conv_state, s_state);
        let mut z = pool.get(value_dim);
        let pz = match &la.in_proj {
            GdnIn::Split { z, .. } => z,
            GdnIn::Fused(_) => panic!("legacy single-stream path cannot use a fused (quantized) model"),
        };
        self.gemm_bf16(pool, pz.bf16(), hidden, &mut z, h, value_dim);
        let normed = pool.get(value_dim);
        unsafe { self.k.rmsnorm_gated.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(nh as u32,1,1),block_dim:(vd as u32,1,1),shared_mem_bytes:(vd*4) as u32},
            (d(&normed), d(&core), d(&z), d(&la.norm), vd as i32, fbits(cfg.rms_eps))).unwrap(); }
        let mut out = pool.get(h);
        self.gemm_bf16(pool, la.out_proj.bf16(), &normed, &mut out, value_dim, h);
        pool.release(core, nh*vd); pool.release(z, value_dim); pool.release(normed, value_dim);
        out
    }
    pub fn linear_attn_core(&self, pool: &mut Pool, hidden: &S, la: &GpuLinearAttn, conv_state: &mut S, s_state: &mut S) -> S {
        let cfg = &self.cfg;
        let (h, nh, kd, vd) = (cfg.hidden_size, cfg.lin_num_v_heads, cfg.lin_k_dim, cfg.lin_v_dim);
        let key_dim = cfg.key_dim(); let value_dim = cfg.value_dim();
        let conv_dim = key_dim*2 + value_dim; let ck = cfg.conv_kernel;
        let (pqkv, pb, pa) = match &la.in_proj {
            GdnIn::Split { qkv, b, a, .. } => (qkv, b, a),
            GdnIn::Fused(_) => panic!("legacy single-stream path cannot use a fused (quantized) model"),
        };
        let mut qkv = pool.get(conv_dim);
        self.gemm_bf16(pool, pqkv.bf16(), hidden, &mut qkv, h, conv_dim);
        unsafe { self.k.conv1d_step.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:grid(conv_dim),block_dim:(256,1,1),shared_mem_bytes:0},
            (d(&qkv), d(conv_state), d(&la.conv1d), conv_dim as i32, ck as i32)).unwrap(); }
        let qkv_base = d(&qkv);
        let k_ptr = qkv_base + u(key_dim)*4;
        let v_ptr = qkv_base + u(2*key_dim)*4;
        let mut b = pool.get(nh);
        let mut a = pool.get(nh);
        self.gemm_bf16(pool, pb.bf16(), hidden, &mut b, h, nh);
        self.gemm_bf16(pool, pa.bf16(), hidden, &mut a, h, nh);
        let core = pool.get(nh*vd);
        let smem = ((kd + 4*vd + kd) as u32)*4;
        unsafe { self.k.delta_step.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(nh as u32,1,1),block_dim:(kd as u32,1,1),shared_mem_bytes:smem},
            (d(&core), qkv_base, k_ptr, v_ptr, d(&b), d(&a), d(s_state),
             nh as i32, kd as i32, vd as i32, d(&la.a_log), d(&la.dt_bias))).unwrap(); }
        pool.release(qkv, conv_dim); pool.release(b, nh); pool.release(a, nh);
        core
    }

    pub fn logits(&self, pool: &mut Pool, hidden: &S) -> S {
        let (v, h) = (self.cfg.vocab_size, self.cfg.hidden_size);
        let mut logits = pool.get(v);
        self.gemm_bf16(pool, self.embed.bf16(), hidden, &mut logits, h, v);
        logits
    }

    /// Greedy argmax of logits, computed on device (two-pass). Returns the token id.
    pub fn argmax_gpu(&self, pool: &mut Pool, logits: &S) -> u32 {
        let v = self.cfg.vocab_size;
        let block = 1024usize;
        let nblocks = (v + block - 1) / block;
        let idxs = pool.get(nblocks);   // i32 reinterpreted
        let vals = pool.get(nblocks);   // f32
        unsafe { self.k.argmax_pass1.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(nblocks as u32,1,1),block_dim:(block as u32,1,1),shared_mem_bytes:(block*4*2) as u32},
            (d(&idxs), d(&vals), d(logits), v as i32)).unwrap(); }
        let token_dev = self.dev.alloc_zeros::<i32>(1).unwrap();
        unsafe { self.k.argmax_pass2.clone().unwrap().launch_on_stream(&self.stream,
            LaunchConfig{grid_dim:(1,1,1),block_dim:(nblocks as u32,1,1),shared_mem_bytes:(nblocks*4*2) as u32},
            (*token_dev.device_ptr() as u64, d(&idxs), d(&vals), nblocks as i32)).unwrap(); }
        self.sync_stream();
        let token_host = self.dev.dtoh_sync_copy(&token_dev).unwrap();
        pool.release(idxs, nblocks);
        pool.release(vals, nblocks);
        token_host[0] as u32
    }

    pub fn embed_row(&self, _token: u32) -> S {
        // Deprecated — use embed_gather_b via embed_batch instead
        self.dev.alloc_zeros::<f32>(self.cfg.hidden_size).unwrap()
    }

    /// Allocate per-layer inference state (KV caches for full-attn, conv+recurrent for linear-attn).
    pub fn new_state(&self) -> GpuState { self.new_state_stride(self.cfg.max_position_embeddings) }
    pub fn new_state_stride(&self, stride: usize) -> GpuState {
        let cfg = &self.cfg;
        let mut k_cache = vec![];
        let mut v_cache = vec![];
        let mut conv_state = vec![];
        let mut s_state = vec![];
        for lt in &cfg.layer_types {
            match lt {
                LayerType::FullAttention => {
                    let bytes = cfg.num_kv_heads * cfg.head_dim * stride;
                    k_cache.push(Some(self.dev.alloc_zeros::<f32>(bytes).unwrap()));
                    v_cache.push(Some(self.dev.alloc_zeros::<f32>(bytes).unwrap()));
                    conv_state.push(None);
                    s_state.push(None);
                }
                LayerType::LinearAttention => {
                    let conv_dim = cfg.key_dim() * 2 + cfg.value_dim();
                    conv_state.push(Some(self.dev.alloc_zeros::<f32>(conv_dim * cfg.conv_kernel).unwrap()));
                    s_state.push(Some(self.dev.alloc_zeros::<f32>(cfg.lin_num_v_heads * cfg.lin_k_dim * cfg.lin_v_dim).unwrap()));
                    k_cache.push(None);
                    v_cache.push(None);
                }
            }
        }
        GpuState { k_cache, v_cache, conv_state, s_state, pos: 0, max_seq_len: stride }
    }

    // ===================== BATCHED (continuous batching) =====================
}
