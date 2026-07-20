use half::f16;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::ptr::NonNull;

#[derive(Debug, Clone, Copy)]
pub struct DevicePtr(pub NonNull<f16>);

unsafe impl Send for DevicePtr {}
unsafe impl Sync for DevicePtr {}

impl DevicePtr {
    pub fn as_ptr(&self) -> *mut f16 {
        self.0.as_ptr()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub num_experts: usize,
    pub experts_per_token: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 128,
            num_layers: 24,
            vocab_size: 151936,
            intermediate_size: 2816,
            max_seq_len: 4096,
            rms_norm_eps: 1e-6,
            num_experts: 0,
            experts_per_token: 0,
        }
    }
}

impl ModelConfig {
    pub fn qwen3_5_0_08b() -> Self { Self::default() }

    pub fn total_params(&self) -> usize {
        let embedding = self.vocab_size * self.hidden_size;
        let lm_head = self.vocab_size * self.hidden_size;
        let qkv_proj = 3 * self.hidden_size * self.hidden_size;
        let o_proj = self.hidden_size * self.hidden_size;
        let gate_up_proj = 2 * self.hidden_size * self.intermediate_size;
        let down_proj = self.hidden_size * self.intermediate_size;
        let per_layer = qkv_proj + o_proj + gate_up_proj + down_proj;
        embedding + lm_head + self.num_layers * per_layer
    }
}

#[derive(Debug, Clone)]
pub struct NVFP4WeightBlock {
    pub scale: f16,
    pub packed_values: u64,
}

impl NVFP4WeightBlock {
    pub fn zero() -> Self {
        Self { scale: f16::from_f32(0.0), packed_values: 0 }
    }
    pub fn is_zero(&self) -> bool { self.scale == f16::from_f32(0.0) }
}

#[derive(Debug, Clone)]
pub struct PackedLinear {
    pub weights: Vec<NVFP4WeightBlock>,
    pub scales: Vec<f16>,
    pub in_features: usize,
    pub out_features: usize,
    pub block_size: usize,
}

impl PackedLinear {
    pub fn new(in_features: usize, out_features: usize, block_size: usize) -> Self {
        let num_blocks = ((in_features + block_size - 1) / block_size) * out_features;
        Self {
            weights: vec![NVFP4WeightBlock::zero(); num_blocks],
            scales: vec![f16::from_f32(0.0); num_blocks],
            in_features,
            out_features,
            block_size,
        }
    }
    pub fn num_blocks(&self) -> usize {
        let rows = (self.out_features + self.block_size - 1) / self.block_size;
        let cols = (self.in_features + self.block_size - 1) / self.block_size;
        rows * cols
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RoPEParams {
    pub dim: usize,
    pub max_seq_len: usize,
    pub base: f32,
}

impl Default for RoPEParams {
    fn default() -> Self {
        Self { dim: 128, max_seq_len: 4096, base: 10000.0 }
    }
}

#[derive(Debug, Clone)]
pub struct DenseLinear {
    pub weight: Vec<f16>,
    pub bias: Vec<f16>,
    pub weight_ptr: Option<DevicePtr>,
    pub bias_ptr: Option<DevicePtr>,
}

impl DenseLinear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weight: vec![f16::from_f32(0.0); out_features * in_features],
            bias: vec![f16::from_f32(0.0); out_features],
            weight_ptr: None,
            bias_ptr: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransformerLayer {
    pub q_proj: PackedLinear,
    pub k_proj: PackedLinear,
    pub v_proj: PackedLinear,
    pub o_proj: PackedLinear,
    pub gate_proj: PackedLinear,
    pub up_proj: PackedLinear,
    pub down_proj: PackedLinear,
    pub dense_q_proj: DenseLinear,
    pub dense_k_proj: DenseLinear,
    pub dense_v_proj: DenseLinear,
    pub dense_o_proj: DenseLinear,
    pub dense_gate_proj: DenseLinear,
    pub dense_up_proj: DenseLinear,
    pub dense_down_proj: DenseLinear,
    pub input_layernorm_weight: Vec<f16>,
    pub post_attention_layernorm_weight: Vec<f16>,
    pub input_layernorm_weight_ptr: Option<DevicePtr>,
    pub post_attention_layernorm_weight_ptr: Option<DevicePtr>,
}

impl TransformerLayer {
    pub fn new(config: &ModelConfig) -> Self {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let kv_heads = config.num_kv_heads * config.head_dim;
        Self {
            q_proj: PackedLinear::new(hidden, hidden, 16),
            k_proj: PackedLinear::new(hidden, kv_heads, 16),
            v_proj: PackedLinear::new(hidden, kv_heads, 16),
            o_proj: PackedLinear::new(hidden, hidden, 16),
            gate_proj: PackedLinear::new(hidden, intermediate, 16),
            up_proj: PackedLinear::new(hidden, intermediate, 16),
            down_proj: PackedLinear::new(intermediate, hidden, 16),
            dense_q_proj: DenseLinear::new(hidden, hidden),
            dense_k_proj: DenseLinear::new(hidden, kv_heads),
            dense_v_proj: DenseLinear::new(hidden, kv_heads),
            dense_o_proj: DenseLinear::new(hidden, hidden),
            dense_gate_proj: DenseLinear::new(hidden, intermediate),
            dense_up_proj: DenseLinear::new(hidden, intermediate),
            dense_down_proj: DenseLinear::new(intermediate, hidden),
            input_layernorm_weight: vec![f16::from_f32(1.0); hidden],
            post_attention_layernorm_weight: vec![f16::from_f32(1.0); hidden],
            input_layernorm_weight_ptr: None,
            post_attention_layernorm_weight_ptr: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3Model {
    pub config: ModelConfig,
    pub embed_tokens: Vec<f16>,
    pub lm_head: Vec<f16>,
    pub lm_head_packed: PackedLinear,
    pub layers: Vec<TransformerLayer>,
    pub embed_tokens_ptr: Option<DevicePtr>,
    pub lm_head_ptr: Option<DevicePtr>,
}

impl Qwen3Model {
    pub fn new(config: ModelConfig) -> Self {
        let layers = (0..config.num_layers).map(|_| TransformerLayer::new(&config)).collect();
        Self {
            embed_tokens: vec![f16::from_f32(0.0); config.vocab_size * config.hidden_size],
            lm_head: vec![f16::from_f32(0.0); config.vocab_size * config.hidden_size],
            lm_head_packed: PackedLinear::new(config.hidden_size, config.vocab_size, 16),
            config,
            layers,
            embed_tokens_ptr: None,
            lm_head_ptr: None,
        }
    }

    /// Upload weights from host buffers to unified memory device pointers
    pub fn upload_to_device(&mut self, memory: &crate::memory::UnifiedMemoryPool) -> anyhow::Result<()> {
        let embed_ptr = memory.allocate::<f16>(self.embed_tokens.len())?;
        memory.memcpy_h2d(embed_ptr as *mut u8, self.embed_tokens.as_ptr() as *const u8, self.embed_tokens.len() * std::mem::size_of::<f16>())?;
        self.embed_tokens_ptr = Some(DevicePtr(NonNull::new(embed_ptr).unwrap()));

        let lm_head_ptr = memory.allocate::<f16>(self.lm_head.len())?;
        memory.memcpy_h2d(lm_head_ptr as *mut u8, self.lm_head.as_ptr() as *const u8, self.lm_head.len() * std::mem::size_of::<f16>())?;
        self.lm_head_ptr = Some(DevicePtr(NonNull::new(lm_head_ptr).unwrap()));

        // Upload layer weights
        for layer in &mut self.layers {
            // Layer norms
            let ln1_ptr = memory.allocate::<f16>(layer.input_layernorm_weight.len())?;
            memory.memcpy_h2d(ln1_ptr as *mut u8, layer.input_layernorm_weight.as_ptr() as *const u8, layer.input_layernorm_weight.len() * std::mem::size_of::<f16>())?;
            layer.input_layernorm_weight_ptr = Some(DevicePtr(NonNull::new(ln1_ptr).unwrap()));

            let ln2_ptr = memory.allocate::<f16>(layer.post_attention_layernorm_weight.len())?;
            memory.memcpy_h2d(ln2_ptr as *mut u8, layer.post_attention_layernorm_weight.as_ptr() as *const u8, layer.post_attention_layernorm_weight.len() * std::mem::size_of::<f16>())?;
            layer.post_attention_layernorm_weight_ptr = Some(DevicePtr(NonNull::new(ln2_ptr).unwrap()));

            // Dense projections
            let dense_projs = [
                (&layer.dense_q_proj.weight, &mut layer.dense_q_proj.weight_ptr),
                (&layer.dense_k_proj.weight, &mut layer.dense_k_proj.weight_ptr),
                (&layer.dense_v_proj.weight, &mut layer.dense_v_proj.weight_ptr),
                (&layer.dense_o_proj.weight, &mut layer.dense_o_proj.weight_ptr),
                (&layer.dense_gate_proj.weight, &mut layer.dense_gate_proj.weight_ptr),
                (&layer.dense_up_proj.weight, &mut layer.dense_up_proj.weight_ptr),
                (&layer.dense_down_proj.weight, &mut layer.dense_down_proj.weight_ptr),
            ];

            for (weight, weight_ptr) in dense_projs {
                let ptr = memory.allocate::<f16>(weight.len())?;
                memory.memcpy_h2d(ptr as *mut u8, weight.as_ptr() as *const u8, weight.len() * std::mem::size_of::<f16>())?;
                *weight_ptr = Some(DevicePtr(NonNull::new(ptr).unwrap()));
            }
        }

        Ok(())
    }

    /// Load model weights from safetensors file into host buffers
    pub fn load_from_safetensors(path: &str, config: ModelConfig) -> anyhow::Result<Self> {
        eprintln!("Loading model from: {}", path);
        eprintln!("File exists: {}", std::path::Path::new(path).exists());
        let data = std::fs::read(path)?;
        eprintln!("Read {} bytes", data.len());
        let tensors = SafeTensors::deserialize(&data)?;
        let mut model = Self::new(config);

        // Map of tensor names to their data
        let tensor_map: HashMap<String, &[u8]> = tensors.tensors()
            .into_iter()
            .map(|(name, view)| (name, view.data()))
            .collect();

        // Helper to copy tensor data
        fn copy_tensor<T: bytemuck::Pod>(dst: &mut [T], src: &[u8], name: &str) -> anyhow::Result<()> {
            let src_t = bytemuck::try_cast_slice(src)
                .map_err(|e| anyhow::anyhow!("Failed to cast tensor {}: {}", name, e))?;
            if dst.len() != src_t.len() {
                anyhow::bail!("Size mismatch for {}: expected {}, got {}", name, dst.len(), src_t.len());
            }
            dst.copy_from_slice(src_t);
            Ok(())
        }

        // Embedding
        if let Some(data) = tensor_map.get("model.embed_tokens.weight") {
            copy_tensor(&mut model.embed_tokens, data, "model.embed_tokens.weight")?;
        }

        // LM head
        if let Some(data) = tensor_map.get("lm_head.weight") {
            copy_tensor(&mut model.lm_head, data, "lm_head.weight")?;
        }

        // Layer weights
        for layer_idx in 0..model.config.num_layers {
            let layer = &mut model.layers[layer_idx];
            let prefix = format!("model.layers.{}", layer_idx);

            // Layer norms
            if let Some(data) = tensor_map.get(&format!("{}.input_layernorm.weight", prefix)) {
                copy_tensor(&mut layer.input_layernorm_weight, data, &format!("{}.input_layernorm.weight", prefix))?;
            }
            if let Some(data) = tensor_map.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
                copy_tensor(&mut layer.post_attention_layernorm_weight, data, &format!("{}.post_attention_layernorm.weight", prefix))?;
            }

            // QKV projections
            if let Some(data) = tensor_map.get(&format!("{}.self_attn.q_proj.weight", prefix)) {
                copy_tensor(&mut layer.dense_q_proj.weight, data, &format!("{}.self_attn.q_proj.weight", prefix))?;
            }
            if let Some(data) = tensor_map.get(&format!("{}.self_attn.k_proj.weight", prefix)) {
                copy_tensor(&mut layer.dense_k_proj.weight, data, &format!("{}.self_attn.k_proj.weight", prefix))?;
            }
            if let Some(data) = tensor_map.get(&format!("{}.self_attn.v_proj.weight", prefix)) {
                copy_tensor(&mut layer.dense_v_proj.weight, data, &format!("{}.self_attn.v_proj.weight", prefix))?;
            }
            if let Some(data) = tensor_map.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
                copy_tensor(&mut layer.dense_o_proj.weight, data, &format!("{}.self_attn.o_proj.weight", prefix))?;
            }

            // MLP projections
            if let Some(data) = tensor_map.get(&format!("{}.mlp.gate_proj.weight", prefix)) {
                copy_tensor(&mut layer.dense_gate_proj.weight, data, &format!("{}.mlp.gate_proj.weight", prefix))?;
            }
            if let Some(data) = tensor_map.get(&format!("{}.mlp.up_proj.weight", prefix)) {
                copy_tensor(&mut layer.dense_up_proj.weight, data, &format!("{}.mlp.up_proj.weight", prefix))?;
            }
            if let Some(data) = tensor_map.get(&format!("{}.mlp.down_proj.weight", prefix)) {
                copy_tensor(&mut layer.dense_down_proj.weight, data, &format!("{}.mlp.down_proj.weight", prefix))?;
            }
        }

        Ok(model)
    }
}
