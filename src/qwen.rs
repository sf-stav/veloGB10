//! Qwen3.5-0.8B hybrid architecture (Qwen3_5ForConditionalGeneration text path).
//!
//! Pure-CPU f32 forward pass, validated against the transformers reference.
//! Two layer types per `layer_types`:
//!   - linear_attention: GatedDeltaNet (conv1d + gated delta-rule recurrence)
//!   - full_attention:   GQA with output gate, mrope (text-only = rotate_half RoPE)
//! Tied lm_head (logits = embed_tokens^T @ h).

use half::bf16;
use safetensors::SafeTensors;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Family { Qwen35, HyV3 }

#[derive(Clone, Debug)]
pub struct Config {
    pub family: Family,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub lin_num_k_heads: usize,
    pub lin_num_v_heads: usize,
    pub lin_k_dim: usize,
    pub lin_v_dim: usize,
    pub conv_kernel: usize,
    pub vocab_size: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub rotary_dim: usize,
    pub max_position_embeddings: usize,
    pub eos_token_id: u32,
    pub layer_types: Vec<LayerType>,
    pub tie_word_embeddings: bool,
    // MoE (qwen3_5_moe). The dense-FFN family (qwen3_5) leaves `is_moe=false` and the rest zeroed.
    // Layout (from the 35B checkpoint): experts are STACKED 3D with gate+up FUSED —
    //   `layers.N.mlp.experts.gate_up_proj` [num_experts, 2*moe_inter, hidden],
    //   `layers.N.mlp.experts.down_proj`    [num_experts, hidden, moe_inter],
    //   `layers.N.mlp.gate.weight`          [num_experts, hidden]   (router),
    //   `layers.N.mlp.shared_expert.{gate,up,down}_proj.weight` (standard 2D MLP),
    //   `layers.N.mlp.shared_expert_gate.weight` [1, hidden]        (sigmoid gate).
    // Router (from the reference): softmax over all experts (fp32) → top-k → RENORMALIZE the top-k →
    // those are the combine weights; shared expert added as sigmoid(shared_gate·h) * shared_mlp(h).
    pub is_moe: bool,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,       // top-k
    pub moe_intermediate_size: usize,     // per-expert FFN width
    pub shared_expert_intermediate_size: usize,
    pub mlp_only_layers: Vec<usize>,      // layers that are DENSE despite is_moe (empty on the 35B)
    // hy_v3 (Hy3). Pure-GQA MoE — no GDN at all. Config lives at the ROOT of config.json (no
    // text_config nesting). Router is NOT the qwen3_5_moe one: sigmoid scores + a learned per-expert
    // bias used for SELECTION ONLY (noaux_tc), top-k renormalized (route_norm), output × router_scaling
    // (2.826). Tensor names differ too: mlp.router.gate.weight, mlp.expert_bias [E],
    // mlp.shared_mlp.{gate,up,down}_proj (NOT shared_expert.*). Per-head qk_norm on q,k pre-RoPE.
    // LM head accumulation must be fp32 (enable_lm_head_fp32).
    pub qk_norm: bool,
    pub router_sigmoid: bool,
    pub router_expert_bias: bool,
    pub route_norm: bool,
    pub router_scaling: f32,
    pub lm_head_fp32: bool,
}

impl Config {
    /// Parse config from the model's config.json file.
    pub fn from_config_json(path: &str) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let root: serde_json::Value = serde_json::from_str(&raw)?;

        // Family detection. hy_v3 keeps everything at the ROOT of config.json (no text_config);
        // the qwen3.5 family nests under text_config.
        let model_type = root["model_type"].as_str().unwrap_or("");
        let family = if model_type == "hy_v3" { Family::HyV3 } else { Family::Qwen35 };
        let tc_owned;
        let tc: &serde_json::Value = if family == Family::HyV3 {
            tc_owned = root.clone(); &tc_owned
        } else {
            &root["text_config"]
        };

        let hidden_size = tc["hidden_size"].as_u64().unwrap_or(2048) as usize;

        // MoE detection: qwen3_5_moe and hy_v3 carry `num_experts` (the dense qwen3_5 family does not).
        let num_experts = tc["num_experts"].as_u64().unwrap_or(0) as usize;
        let is_moe = num_experts > 0;
        let num_experts_per_tok = tc["num_experts_per_tok"].as_u64().unwrap_or(0) as usize;
        let moe_intermediate_size = tc["moe_intermediate_size"].as_u64().unwrap_or(0) as usize;
        // hy_v3's config.json OMITS `shared_expert_intermediate_size`; the HF reference derives it
        // as moe_intermediate_size * num_shared_experts (Hy3: 1536 * 1). Defaulting to 0 there sized
        // the shared expert to NOTHING — a silently-dead shared MLP on every MoE layer.
        let num_shared_experts = tc["num_shared_experts"].as_u64().unwrap_or(1) as usize;
        let shared_expert_intermediate_size =
            tc["shared_expert_intermediate_size"].as_u64().map(|x| x as usize)
            .unwrap_or(if is_moe { moe_intermediate_size * num_shared_experts } else { 0 });
        let mlp_only_layers: Vec<usize> = if family == Family::HyV3 {
            // first_k_dense_replace=N → the first N layers are DENSE (Hy3: layer 0, inter 13312).
            let k = tc["first_k_dense_replace"].as_u64().unwrap_or(0) as usize;
            (0..k).collect()
        } else {
            tc["mlp_only_layers"].as_array()
                .map(|a| a.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect())
                .unwrap_or_default()
        };

        // Dense-FFN width. MoE checkpoints OMIT `intermediate_size` (they use moe_intermediate_size);
        // fall back to the MoE width, not the 6144 default, so nothing sizes a buffer to a wrong value.
        let intermediate_size = tc["intermediate_size"].as_u64().map(|x| x as usize)
            .unwrap_or(if moe_intermediate_size > 0 { moe_intermediate_size } else { 6144 });
        let num_layers = tc["num_hidden_layers"].as_u64().unwrap_or(24) as usize;
        let num_heads = tc["num_attention_heads"].as_u64().unwrap_or(8) as usize;
        let num_kv_heads = tc["num_key_value_heads"].as_u64().unwrap_or(2) as usize;
        let head_dim = tc["head_dim"].as_u64().unwrap_or(256) as usize;
        let lin_num_k_heads = tc["linear_num_key_heads"].as_u64().unwrap_or(16) as usize;
        let lin_num_v_heads = tc["linear_num_value_heads"].as_u64().unwrap_or(16) as usize;
        let lin_k_dim = tc["linear_key_head_dim"].as_u64().unwrap_or(128) as usize;
        let lin_v_dim = tc["linear_value_head_dim"].as_u64().unwrap_or(128) as usize;
        let conv_kernel = tc["linear_conv_kernel_dim"].as_u64().unwrap_or(4) as usize;
        let vocab_size = tc["vocab_size"].as_u64().unwrap_or(248320) as usize;
        let rms_eps = tc["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let max_position_embeddings = tc["max_position_embeddings"].as_u64().unwrap_or(262144) as usize;
        let eos_token_id = tc["eos_token_id"].as_u64().unwrap_or(248044) as u32;

        // Parse RoPE
        let rope_theta = tc["rope_parameters"]["rope_theta"]
            .as_f64().unwrap_or(1e7) as f32;
        let rotary_dim = if family == Family::HyV3 {
            head_dim   // rope_type "default" = full-dim rotary (no partial_rotary_factor in hy_v3)
        } else {
            let partial_rotary = tc["rope_parameters"]["partial_rotary_factor"]
                .as_f64().unwrap_or(0.25) as f32;
            (head_dim as f32 * partial_rotary) as usize
        };

        // Parse layer types. hy_v3 is PURE GQA (no GDN anywhere) — every layer is full_attention.
        let layer_types: Vec<LayerType> = if family == Family::HyV3 {
            vec![LayerType::FullAttention; num_layers]
        } else {
            tc["layer_types"].as_array()
                .map(|arr| arr.iter().map(|v| {
                    let s = v.as_str().unwrap_or("");
                    if s.contains("full") { LayerType::FullAttention }
                    else { LayerType::LinearAttention }
                }).collect())
                .unwrap_or_else(|| {
                    // Fallback: derive from full_attention_interval
                    let interval = tc["full_attention_interval"].as_u64().unwrap_or(4) as usize;
                    (0..num_layers).map(|i| {
                        if i % interval == interval - 1 { LayerType::FullAttention }
                        else { LayerType::LinearAttention }
                    }).collect()
                })
        };

        // hy_v3 router / qk_norm / lm-head fields (defaults reproduce the qwen3_5_moe behavior).
        let qk_norm = tc["qk_norm"].as_bool().unwrap_or(false);
        let router_sigmoid = tc["moe_router_use_sigmoid"].as_bool().unwrap_or(false);
        let router_expert_bias = tc["moe_router_enable_expert_bias"].as_bool().unwrap_or(false);
        let route_norm = tc["route_norm"].as_bool().unwrap_or(true);
        let router_scaling = tc["router_scaling_factor"].as_f64().unwrap_or(1.0) as f32;
        let lm_head_fp32 = tc["enable_lm_head_fp32"].as_bool().unwrap_or(false);

        // tie_word_embeddings
        let tie = tc["tie_word_embeddings"].as_bool()
            .or_else(|| root["tie_word_embeddings"].as_bool())
            .unwrap_or(true);

        Ok(Self {
            family,
            hidden_size, intermediate_size, num_layers, num_heads, num_kv_heads,
            head_dim, lin_num_k_heads, lin_num_v_heads, lin_k_dim, lin_v_dim,
            conv_kernel, vocab_size, rms_eps, rope_theta, rotary_dim,
            max_position_embeddings, eos_token_id, layer_types,
            tie_word_embeddings: tie,
            is_moe, num_experts, num_experts_per_tok, moe_intermediate_size,
            shared_expert_intermediate_size, mlp_only_layers,
            qk_norm, router_sigmoid, router_expert_bias, route_norm, router_scaling, lm_head_fp32,
        })
    }

    pub fn key_dim(&self) -> usize { self.lin_k_dim * self.lin_num_k_heads }
    pub fn value_dim(&self) -> usize { self.lin_v_dim * self.lin_num_v_heads }

    /// Is layer `i` an MoE FFN layer? (MoE model AND not one of the dense `mlp_only_layers`.)
    pub fn is_moe_layer(&self, i: usize) -> bool {
        self.is_moe && !self.mlp_only_layers.contains(&i)
    }

    /// Does full-attention fuse an output sigmoid-gate into q_proj (output rows [q|gate] per head)?
    /// qwen3_5 full-attn does; hy_v3's q_proj is bare [num_heads*head_dim, hidden] (plain GQA).
    /// Drives the qkv output width everywhere: nh*hd*(1 + gate) + 2*nkv*hd.
    pub fn attn_out_gate(&self) -> bool { self.family != Family::HyV3 }

    /// Placeholder for auto-detect fallback (no config.json available).
    fn from_config_json_placeholder() -> Self {
        let mut lt = Vec::new();
        for i in 0..24 {
            lt.push(if i % 4 == 3 { LayerType::FullAttention } else { LayerType::LinearAttention });
        }
        Self {
            family: Family::Qwen35,
            hidden_size: 0, intermediate_size: 0, num_layers: 24,
            num_heads: 8, num_kv_heads: 2, head_dim: 256,
            lin_num_k_heads: 16, lin_num_v_heads: 16, lin_k_dim: 128, lin_v_dim: 128,
            conv_kernel: 4, vocab_size: 248320, rms_eps: 1e-6, rope_theta: 1e7,
            rotary_dim: 64, max_position_embeddings: 262144, eos_token_id: 248044,
            layer_types: lt, tie_word_embeddings: true,
            is_moe: false, num_experts: 0, num_experts_per_tok: 0, moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0, mlp_only_layers: Vec::new(),
            qk_norm: false, router_sigmoid: false, router_expert_bias: false,
            route_norm: true, router_scaling: 1.0, lm_head_fp32: false,
        }
    }
}

#[derive(Clone)]
pub struct LinearAttn {
    pub in_proj_qkv: Vec<f32>, // [key_dim*2 + value_dim, hidden]
    pub in_proj_z: Vec<f32>,   // [value_dim, hidden]
    pub in_proj_b: Vec<f32>,   // [num_v_heads, hidden]
    pub in_proj_a: Vec<f32>,   // [num_v_heads, hidden]
    pub conv1d: Vec<f32>,      // [conv_dim, conv_kernel]
    pub a_log: Vec<f32>,       // [num_v_heads]
    pub dt_bias: Vec<f32>,     // [num_v_heads]
    pub norm: Vec<f32>,        // [lin_v_dim]
    pub out_proj: Vec<f32>,    // [hidden, value_dim]
}

#[derive(Clone)]
pub struct FullAttn {
    pub q_proj: Vec<f32>, // [num_heads*head_dim*2, hidden]
    pub k_proj: Vec<f32>, // [num_kv_heads*head_dim, hidden]
    pub v_proj: Vec<f32>, // [num_kv_heads*head_dim, hidden]
    pub o_proj: Vec<f32>, // [hidden, num_heads*head_dim]
    pub q_norm: Vec<f32>, // [head_dim]
    pub k_norm: Vec<f32>, // [head_dim]
}

#[derive(Clone)]
pub struct Mlp {
    pub gate_proj: Vec<f32>, // [intermediate, hidden]
    pub up_proj: Vec<f32>,   // [intermediate, hidden]
    pub down_proj: Vec<f32>, // [hidden, intermediate]
}

#[derive(Clone)]
pub struct Layer {
    pub layer_type: LayerType,
    pub linear_attn: Option<LinearAttn>,
    pub full_attn: Option<FullAttn>,
    pub mlp: Mlp,
    pub input_layernorm: Vec<f32>,        // [hidden]
    pub post_attention_layernorm: Vec<f32>, // [hidden]
}

#[derive(Clone)]
pub struct Model {
    pub config: Config,
    pub embed_tokens: Vec<f32>,
    pub layers: Vec<Layer>,
    pub norm: Vec<f32>,
    pub mtp: Option<MtpLayer>,
    pub lm_head: Option<Vec<f32>>,  // None = tied (use embed_tokens)
}

#[derive(Clone)]
pub struct MtpLayer {
    pub fc: Vec<f32>,                    // [hidden, 2*hidden]
    pub pre_fc_norm_hidden: Vec<f32>,    // [hidden]
    pub pre_fc_norm_embedding: Vec<f32>, // [hidden]
    pub input_ln: Vec<f32>,             // [hidden]
    pub post_ln: Vec<f32>,              // [hidden]
    pub q_proj: Vec<f32>,               // [nh*hd*2, hidden]
    pub k_proj: Vec<f32>,               // [nkv*hd, hidden]
    pub v_proj: Vec<f32>,               // [nkv*hd, hidden]
    pub o_proj: Vec<f32>,               // [hidden, nh*hd]
    pub q_norm: Vec<f32>,               // [hd]
    pub k_norm: Vec<f32>,               // [hd]
    pub gate_proj: Vec<f32>,            // [intermediate, hidden]
    pub up_proj: Vec<f32>,              // [intermediate, hidden]
    pub down_proj: Vec<f32>,            // [hidden, intermediate]
    pub final_norm: Vec<f32>,           // [hidden]
}

fn bf16_slice_to_f32(data: &[u8]) -> Vec<f32> {
    let n = data.len() / 2;
    let mut out = Vec::with_capacity(n);
    let bytes: &[u16] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, n) };
    for &b in bytes {
        out.push(bf16::from_bits(b).to_f32());
    }
    out
}

fn f32_slice_to_f32(data: &[u8]) -> Vec<f32> {
    let n = data.len() / 4;
    let mut out = Vec::with_capacity(n);
    let bytes: &[f32] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
    out.extend_from_slice(bytes);
    out
}

fn load_w(
    map: &std::collections::HashMap<String, (&str, &[u8])>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let (dt, data) = map.get(name).unwrap_or_else(|| panic!("missing tensor: {}", name));
    let v = if *dt == "BF16" || *dt == "F16" {
        bf16_slice_to_f32(data)
    } else if *dt == "F32" {
        f32_slice_to_f32(data)
    } else {
        panic!("unsupported dtype {} for {}", dt, name);
    };
    assert_eq!(v.len(), rows * cols, "shape mismatch {}: got {} expect {}", name, v.len(), rows*cols);
    v
}

impl Model {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        // Determine if path is a directory or a file
        let (config_path, safetensors_files, _tokenizer_path) = if std::path::Path::new(path).is_dir() {
            // Model directory mode: find config.json and safetensors files
            let dir = std::path::Path::new(path);
            let cfg = dir.join("config.json").to_string_lossy().to_string();

            // Find safetensors files: check for index.json first
            let index_path = dir.join("model.safetensors.index.json");
            let files: Vec<String> = if index_path.exists() {
                // Multi-file: read index to get shard list
                let index_raw = std::fs::read_to_string(&index_path)?;
                let index: serde_json::Value = serde_json::from_str(&index_raw)?;
                let weight_map = index["weight_map"].as_object().unwrap();
                let mut shards: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
                for (_, v) in weight_map {
                    if let Some(s) = v.as_str() { shards.insert(s.to_string()); }
                }
                shards.into_iter().map(|s| dir.join(s).to_string_lossy().to_string()).collect()
            } else {
                // Single file or glob
                let mut found = vec![];
                for entry in std::fs::read_dir(dir)? {
                    let entry = entry?;
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.ends_with(".safetensors") { found.push(entry.path().to_string_lossy().to_string()); }
                }
                found.sort();
                found
            };
            (cfg, files, String::new())
        } else {
            // Single file mode
            (String::new(), vec![path.to_string()], String::new())
        };

        // Load config
        let cfg = if config_path.is_empty() {
            // Fallback: auto-detect from embedding shape (backward compat with single-file mode)
            let raw = std::fs::read(&safetensors_files[0])?;
            let st = SafeTensors::deserialize(&raw)?;
            let embed_view = st.tensor(&format!("{}.embed_tokens.weight", "model.language_model"))
                .unwrap_or_else(|_| panic!("missing embed_tokens"));
            let hidden = embed_view.shape()[1];
            let gate_key = format!("{}.layers.0.mlp.gate_proj.weight", "model.language_model");
            let gate_view = st.tensor(&gate_key).unwrap_or_else(|_| panic!("missing gate_proj"));
            let intermediate = gate_view.shape()[0];
            let mut c = Config::from_config_json_placeholder();
            c.hidden_size = hidden;
            c.intermediate_size = intermediate;
            println!("Auto-detected (no config.json): hidden={}, intermediate={}", hidden, intermediate);
            c
        } else {
            println!("Loading config from {}...", config_path);
            Config::from_config_json(&config_path)?
        };

        println!("Model config: hidden={} layers={} inter={} heads={} kv={} gdn_v={} GQA={}:1",
                 cfg.hidden_size, cfg.num_layers, cfg.intermediate_size,
                 cfg.num_heads, cfg.num_kv_heads, cfg.lin_num_v_heads,
                 cfg.num_heads / cfg.num_kv_heads);

        let h = cfg.hidden_size;
        let pref = "model.language_model";

        // Load all safetensors files. Keep raw bytes alive in a Vec.
        let mut all_raw: Vec<Vec<u8>> = Vec::new();
        for (i, sf_path) in safetensors_files.iter().enumerate() {
            if safetensors_files.len() > 1 {
                println!("  Loading shard {}/{}: {}", i + 1, safetensors_files.len(),
                         std::path::Path::new(sf_path).file_name().unwrap_or_default().to_string_lossy());
            }
            all_raw.push(std::fs::read(sf_path)?);
        }

        // Build merged tensor map from all shards
        let mut map: std::collections::HashMap<String, (&str, &[u8])> = std::collections::HashMap::new();
        for raw in &all_raw {
            let st = SafeTensors::deserialize(raw)?;
            use safetensors::Dtype;
            for (name, view) in st.tensors() {
                let dt = match view.dtype() {
                    Dtype::BF16 => "BF16",
                    Dtype::F16 => "F16",
                    Dtype::F32 => "F32",
                    _ => "OTHER",
                };
                map.insert(name, (dt, view.data()));
            }
        }

        let embed = load_w(&map, &format!("{}.embed_tokens.weight", pref), cfg.vocab_size, h);
        let norm = load_w(&map, &format!("{}.norm.weight", pref), h, 1);

        // Load lm_head (tied or untied)
        let lm_head = if !cfg.tie_word_embeddings {
            println!("Loading untied lm_head...");
            Some(load_w(&map, "lm_head.weight", cfg.vocab_size, h))
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let lpref = format!("{}.layers.{}", pref, i);
            let lt = cfg.layer_types[i];
            let input_ln = load_w(&map, &format!("{}.input_layernorm.weight", lpref), h, 1);
            let post_ln = load_w(&map, &format!("{}.post_attention_layernorm.weight", lpref), h, 1);
            let mlp = Mlp {
                gate_proj: load_w(&map, &format!("{}.mlp.gate_proj.weight", lpref), cfg.intermediate_size, h),
                up_proj: load_w(&map, &format!("{}.mlp.up_proj.weight", lpref), cfg.intermediate_size, h),
                down_proj: load_w(&map, &format!("{}.mlp.down_proj.weight", lpref), h, cfg.intermediate_size),
            };
            let (linear_attn, full_attn) = match lt {
                LayerType::LinearAttention => {
                    let key_dim = cfg.key_dim();
                    let value_dim = cfg.value_dim();
                    let conv_dim = key_dim * 2 + value_dim;
                    let la = LinearAttn {
                        in_proj_qkv: load_w(&map, &format!("{}.linear_attn.in_proj_qkv.weight", lpref), conv_dim, h),
                        in_proj_z: load_w(&map, &format!("{}.linear_attn.in_proj_z.weight", lpref), value_dim, h),
                        in_proj_b: load_w(&map, &format!("{}.linear_attn.in_proj_b.weight", lpref), cfg.lin_num_v_heads, h),
                        in_proj_a: load_w(&map, &format!("{}.linear_attn.in_proj_a.weight", lpref), cfg.lin_num_v_heads, h),
                        conv1d: load_w(&map, &format!("{}.linear_attn.conv1d.weight", lpref), conv_dim, cfg.conv_kernel),
                        a_log: load_w(&map, &format!("{}.linear_attn.A_log", lpref), cfg.lin_num_v_heads, 1),
                        dt_bias: load_w(&map, &format!("{}.linear_attn.dt_bias", lpref), cfg.lin_num_v_heads, 1),
                        norm: load_w(&map, &format!("{}.linear_attn.norm.weight", lpref), cfg.lin_v_dim, 1),
                        out_proj: load_w(&map, &format!("{}.linear_attn.out_proj.weight", lpref), h, value_dim),
                    };
                    (Some(la), None)
                }
                LayerType::FullAttention => {
                    let fa = FullAttn {
                        q_proj: load_w(&map, &format!("{}.self_attn.q_proj.weight", lpref), cfg.num_heads * cfg.head_dim * 2, h),
                        k_proj: load_w(&map, &format!("{}.self_attn.k_proj.weight", lpref), cfg.num_kv_heads * cfg.head_dim, h),
                        v_proj: load_w(&map, &format!("{}.self_attn.v_proj.weight", lpref), cfg.num_kv_heads * cfg.head_dim, h),
                        o_proj: load_w(&map, &format!("{}.self_attn.o_proj.weight", lpref), h, cfg.num_heads * cfg.head_dim),
                        q_norm: load_w(&map, &format!("{}.self_attn.q_norm.weight", lpref), cfg.head_dim, 1),
                        k_norm: load_w(&map, &format!("{}.self_attn.k_norm.weight", lpref), cfg.head_dim, 1),
                    };
                    (None, Some(fa))
                }
            };
            layers.push(Layer { layer_type: lt, linear_attn, full_attn, mlp, input_layernorm: input_ln, post_attention_layernorm: post_ln });
        }

        // Load MTP head if present
        let mtp = if map.contains_key("mtp.fc.weight") {
            println!("Loading MTP head...");
            Some(MtpLayer {
                fc: load_w(&map, "mtp.fc.weight", h, h * 2),
                pre_fc_norm_hidden: load_w(&map, "mtp.pre_fc_norm_hidden.weight", h, 1),
                pre_fc_norm_embedding: load_w(&map, "mtp.pre_fc_norm_embedding.weight", h, 1),
                input_ln: load_w(&map, "mtp.layers.0.input_layernorm.weight", h, 1),
                post_ln: load_w(&map, "mtp.layers.0.post_attention_layernorm.weight", h, 1),
                q_proj: load_w(&map, "mtp.layers.0.self_attn.q_proj.weight", cfg.num_heads * cfg.head_dim * 2, h),
                k_proj: load_w(&map, "mtp.layers.0.self_attn.k_proj.weight", cfg.num_kv_heads * cfg.head_dim, h),
                v_proj: load_w(&map, "mtp.layers.0.self_attn.v_proj.weight", cfg.num_kv_heads * cfg.head_dim, h),
                o_proj: load_w(&map, "mtp.layers.0.self_attn.o_proj.weight", h, cfg.num_heads * cfg.head_dim),
                q_norm: load_w(&map, "mtp.layers.0.self_attn.q_norm.weight", cfg.head_dim, 1),
                k_norm: load_w(&map, "mtp.layers.0.self_attn.k_norm.weight", cfg.head_dim, 1),
                gate_proj: load_w(&map, "mtp.layers.0.mlp.gate_proj.weight", cfg.intermediate_size, h),
                up_proj: load_w(&map, "mtp.layers.0.mlp.up_proj.weight", cfg.intermediate_size, h),
                down_proj: load_w(&map, "mtp.layers.0.mlp.down_proj.weight", h, cfg.intermediate_size),
                final_norm: load_w(&map, "mtp.norm.weight", h, 1),
            })
        } else {
            println!("No MTP head found (model has no multi-token prediction).");
            None
        };

        Ok(Self { config: cfg, embed_tokens: embed, layers, norm, mtp, lm_head })
    }

    // ---- math helpers ----

    /// Standard Qwen3.5 RMSNorm: out = (x * rsqrt(mean(x^2)+eps)) * (1 + w)
    pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
        let n = x.len();
        let mut sum_sq = 0.0f32;
        for &v in x {
            sum_sq += v * v;
        }
        let inv = 1.0 / (sum_sq / n as f32 + eps).sqrt();
        for i in 0..n {
            out[i] = x[i] * inv * (1.0 + w[i]);
        }
    }

    /// Gated RMSNorm (linear attn): out = rmsnorm(x)*w*silu(z)
    pub fn rmsnorm_gated(out: &mut [f32], x: &[f32], z: &[f32], w: &[f32], eps: f32) {
        let n = x.len();
        let mut sum_sq = 0.0f32;
        for &v in x {
            sum_sq += v * v;
        }
        let inv = 1.0 / (sum_sq / n as f32 + eps).sqrt();
        for i in 0..n {
            let zn = z[i] / (1.0 + (-z[i]).exp()); // silu
            out[i] = x[i] * inv * w[i] * zn;
        }
    }

    /// o[out] = sum_i w[out*in + i] * x[i]
    pub fn matvec(o: &mut [f32], w: &[f32], x: &[f32], out: usize, inn: usize) {
        for r in 0..out {
            let row = &w[r * inn..r * inn + inn];
            let mut s = 0.0f32;
            for c in 0..inn {
                s += row[c] * x[c];
            }
            o[r] = s;
        }
    }

    fn l2norm(x: &mut [f32], eps: f32) {
        let mut s = 0.0f32;
        for &v in x.iter() {
            s += v * v;
        }
        let inv = 1.0 / (s + eps).sqrt();
        for v in x.iter_mut() {
            *v *= inv;
        }
    }

    /// Apply rotate_half RoPE to one head's vector (head_dim). rotary on first rotary_dim.
    /// cos,sin length = rotary_dim.
    fn apply_rope_head(x: &mut [f32], cos: &[f32], sin: &[f32], rotary_dim: usize) {
        let half = rotary_dim / 2;
        for i in 0..half {
            let x1 = x[i];
            let x2 = x[i + half];
            // rotate_half: (-x2, x1)
            x[i] = x1 * cos[i] - x2 * sin[i];
            x[i + half] = x2 * cos[i] + x1 * sin[i];
        }
    }

    /// Full-attention forward for ONE token (decode). seq_len = current number of cached KV + this token (1-indexed pos).
    /// kv_cache: [num_kv_heads][max_pos][head_dim] for k and v (filled up to pos-1).
    /// pos is the 0-indexed position of THIS token.
    pub fn full_attn_forward(
        &self,
        out: &mut [f32],            // [hidden]
        hidden: &[f32],             // [hidden], already layernormed
        fa: &FullAttn,
        pos: usize,
        k_cache: &mut [Vec<f32>],   // [num_kv_heads], each len grows
        v_cache: &mut [Vec<f32>],
        cos: &[f32],
        sin: &[f32],
    ) {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let nh = cfg.num_heads;
        let nkv = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let rdim = cfg.rotary_dim;

        // qg = q_proj(h) -> [nh*hd*2]; split q[nh,hd], gate[nh,hd]
        let mut qg = vec![0.0f32; nh * hd * 2];
        Self::matvec(&mut qg, &fa.q_proj, hidden, nh * hd * 2, h);
        // q_norm applied per-head, then rope, on query; gate used later.
        let scale = 1.0f32 / (hd as f32).sqrt();

        // k = k_proj(h) -> [nkv,hd]; v -> [nkv,hd]
        let mut k = vec![0.0f32; nkv * hd];
        let mut v = vec![0.0f32; nkv * hd];
        Self::matvec(&mut k, &fa.k_proj, hidden, nkv * hd, h);
        Self::matvec(&mut v, &fa.v_proj, hidden, nkv * hd, h);

        // per-head q_norm / k_norm (shared weight across heads), then rope.
        // q_proj output is laid out [head0: q(256),gate(256), head1: q(256),gate(256), ...] (view [nh, hd*2], chunk)
        let qstride = hd * 2;
        let mut q = vec![0.0f32; nh * hd];
        for head in 0..nh {
            let qsrc = &qg[head * qstride..head * qstride + hd];
            let qdst = &mut q[head * hd..head * hd + hd];
            Self::rmsnorm(qdst, qsrc, &fa.q_norm, cfg.rms_eps);
            Self::apply_rope_head(qdst, cos, sin, rdim);
        }
        for head in 0..nkv {
            let kdst = &mut k[head * hd..head * hd + hd];
            // k_norm in place (source==dest ok since rmsnorm reads all before writing? we wrote temp)
            let mut tmp = kdst.to_vec();
            Self::rmsnorm(kdst, &mut tmp, &fa.k_norm, cfg.rms_eps);
            Self::apply_rope_head(kdst, cos, sin, rdim);
        }

        // write k,v to cache at pos
        for head in 0..nkv {
            // cache stores per head, contiguous per position
            let base = pos * hd;
            let needed = base + hd;
            if k_cache[head].len() < needed {
                k_cache[head].resize(needed, 0.0);
                v_cache[head].resize(needed, 0.0);
            }
            k_cache[head][base..base + hd].copy_from_slice(&k[head * hd..head * hd + hd]);
            v_cache[head][base..base + hd].copy_from_slice(&v[head * hd..head * hd + hd]);
        }

        // attention: for each query head, GQA group = nh/nkv
        let group = nh / nkv;
        let attn_len = pos + 1;
        let mut attn_out = vec![0.0f32; nh * hd];
        for qh in 0..nh {
            let kvh = qh / group;
            // scores over 0..attn_len
            let mut scores = vec![0.0f32; attn_len];
            let qv = &q[qh * hd..qh * hd + hd];
            let mut maxs = f32::NEG_INFINITY;
            for t in 0..attn_len {
                let kv = &k_cache[kvh][t * hd..t * hd + hd];
                let mut s = 0.0f32;
                for d in 0..hd {
                    s += qv[d] * kv[d];
                }
                s *= scale;
                scores[t] = s;
                if s > maxs {
                    maxs = s;
                }
            }
            let mut sumexp = 0.0f32;
            for t in 0..attn_len {
                scores[t] = (scores[t] - maxs).exp();
                sumexp += scores[t];
            }
            let inv = 1.0 / sumexp;
            let ao = &mut attn_out[qh * hd..qh * hd + hd];
            for d in 0..hd {
                ao[d] = 0.0;
            }
            for t in 0..attn_len {
                let w = scores[t] * inv;
                let kv = &v_cache[kvh][t * hd..t * hd + hd];
                for d in 0..hd {
                    ao[d] += w * kv[d];
                }
            }
        }
        // apply gate: attn_out[head,hd] *= sigmoid(gate[head,hd]); gate is the 2nd half of each head block in qg
        for head in 0..nh {
            for d in 0..hd {
                let g = qg[head * qstride + hd + d];
                let sig = 1.0 / (1.0 + (-g).exp());
                attn_out[head * hd + d] *= sig;
            }
        }
        // o_proj
        Self::matvec(out, &fa.o_proj, &attn_out, h, nh * hd);
    }

    /// Linear-attention (GatedDeltaNet) forward for ONE token (decode step).
    /// conv_state: [conv_dim][conv_kernel]; recurrent S: [num_v_heads][k_dim][v_dim]
    pub fn linear_attn_forward(
        &self,
        out: &mut [f32],          // [hidden]
        hidden: &[f32],           // [hidden], already layernormed
        la: &LinearAttn,
        conv_state: &mut [f32],   // flat [conv_dim * conv_kernel]
        s_state: &mut [f32],      // flat [num_v_heads * k_dim * v_dim]
    ) {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let nh = cfg.lin_num_v_heads; // == k heads (ratio 1)
        let kd = cfg.lin_k_dim;
        let vd = cfg.lin_v_dim;
        let key_dim = cfg.key_dim();   // nh_k * kd
        let value_dim = cfg.value_dim();
        let conv_dim = key_dim * 2 + value_dim;
        let ck = cfg.conv_kernel;

        // projections
        let mut qkv = vec![0.0f32; conv_dim];
        Self::matvec(&mut qkv, &la.in_proj_qkv, hidden, conv_dim, h);
        let mut z = vec![0.0f32; value_dim];
        Self::matvec(&mut z, &la.in_proj_z, hidden, value_dim, h);
        let mut b = vec![0.0f32; nh];
        Self::matvec(&mut b, &la.in_proj_b, hidden, nh, h);
        let mut a = vec![0.0f32; nh];
        Self::matvec(&mut a, &la.in_proj_a, hidden, nh, h);

        // depthwise causal conv1d update (shift state, conv, silu)
        for c in 0..conv_dim {
            // shift left, place new sample at end
            let st = &mut conv_state[c * ck..c * ck + ck];
            for j in 1..ck {
                st[j - 1] = st[j];
            }
            st[ck - 1] = qkv[c];
            let mut acc = 0.0f32;
            for j in 0..ck {
                acc += la.conv1d[c * ck + j] * st[j];
            }
            qkv[c] = acc / (1.0 + (-acc).exp()); // silu
        }

        // split q,k,v
        let mut q = vec![0.0f32; nh * kd];
        let mut k = vec![0.0f32; nh * kd];
        let mut v = vec![0.0f32; nh * vd];
        // q = qkv[0..key_dim], k = qkv[key_dim..2*key_dim], v = qkv[2*key_dim..conv_dim]
        for i in 0..nh * kd {
            q[i] = qkv[i];
            k[i] = qkv[key_dim + i];
        }
        for i in 0..nh * vd {
            v[i] = qkv[2 * key_dim + i];
        }

        let scale = 1.0f32 / (kd as f32).sqrt();
        let mut beta = vec![0.0f32; nh];
        let mut g = vec![0.0f32; nh];
        for i in 0..nh {
            beta[i] = 1.0 / (1.0 + (-b[i]).exp());
            // g = -exp(a_log) * softplus(a + dt_bias)
            let sp = if a[i] + la.dt_bias[i] > 20.0 {
                a[i] + la.dt_bias[i]
            } else {
                (1.0 + (a[i] + la.dt_bias[i]).exp()).ln()
            };
            g[i] = -la.a_log[i].exp() * sp;
        }

        let mut core = vec![0.0f32; nh * vd]; // output [nh, vd]
        for head in 0..nh {
            // l2norm q,k per head
            let qh = &mut q[head * kd..head * kd + kd];
            let kh = &mut k[head * kd..head * kd + kd];
            Self::l2norm(qh, 1e-6);
            Self::l2norm(kh, 1e-6);
            for d in 0..kd {
                qh[d] *= scale;
            }
            let gt = g[head].exp();
            let st = &mut s_state[head * kd * vd..head * kd * vd + kd * vd]; // [kd, vd]
            // st *= exp(g)
            for v in st.iter_mut() {
                *v *= gt;
            }
            // kv_mem[b] = sum_a st[a,b]*k[a]; delta = (v-kv_mem)*beta; st[a,b]+=k[a]*delta[b]
            let vh = &v[head * vd..head * vd + vd];
            let mut delta = vec![0.0f32; vd];
            for bb in 0..vd {
                let mut km = 0.0f32;
                for aa in 0..kd {
                    km += st[aa * vd + bb] * kh[aa];
                }
                delta[bb] = (vh[bb] - km) * beta[head];
            }
            for aa in 0..kd {
                let kv = kh[aa];
                let base = aa * vd;
                for bb in 0..vd {
                    st[base + bb] += kv * delta[bb];
                }
            }
            // out[b] = sum_a st[a,b]*q[a]
            let oh = &mut core[head * vd..head * vd + vd];
            for bb in 0..vd {
                let mut s = 0.0f32;
                for aa in 0..kd {
                    s += st[aa * vd + bb] * qh[aa];
                }
                oh[bb] = s;
            }
        }

        // gated norm: rmsnorm(core, head_v_dim) * w * silu(z)
        let mut normed = vec![0.0f32; nh * vd];
        for head in 0..nh {
            let x = &core[head * vd..head * vd + vd];
            let zz = &z[head * vd..head * vd + vd];
            let o = &mut normed[head * vd..head * vd + vd];
            Self::rmsnorm_gated(o, x, zz, &la.norm, cfg.rms_eps);
        }
        // out_proj
        Self::matvec(out, &la.out_proj, &normed, h, value_dim);
    }

    /// Run the full stack for ONE token at position `pos` (decode). Updates caches in `state`.
    /// `hidden_in` is the residual stream for this token.
    pub fn forward_token(
        &self,
        hidden_in: &[f32],
        pos: usize,
        state: &mut ModelState,
        cos: &[f32],
        sin: &[f32],
    ) -> Vec<f32> {
        self.forward_token_inner(hidden_in, pos, state, cos, sin, None)
    }

    fn forward_token_inner(
        &self,
        hidden_in: &[f32],
        pos: usize,
        state: &mut ModelState,
        cos: &[f32],
        sin: &[f32],
        mut layer_outs: Option<&mut Vec<Vec<f32>>>,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let mut residual = hidden_in.to_vec();
        let mut normed = vec![0.0f32; h];

        for (li, layer) in self.layers.iter().enumerate() {
            Self::rmsnorm(&mut normed, &residual, &layer.input_layernorm, cfg.rms_eps);
            let mut mixer_out = vec![0.0f32; h];
            match layer.layer_type {
                LayerType::LinearAttention => {
                    let la = layer.linear_attn.as_ref().unwrap();
                    let conv = &mut state.conv_states[li];
                    let sst = &mut state.s_states[li];
                    self.linear_attn_forward(&mut mixer_out, &normed, la, conv, sst);
                }
                LayerType::FullAttention => {
                    let fa = layer.full_attn.as_ref().unwrap();
                    let kc = &mut state.k_caches[li];
                    let vc = &mut state.v_caches[li];
                    self.full_attn_forward(&mut mixer_out, &normed, fa, pos, kc, vc, cos, sin);
                }
            }
            for i in 0..h {
                residual[i] += mixer_out[i];
            }
            // mlp
            Self::rmsnorm(&mut normed, &residual, &layer.post_attention_layernorm, cfg.rms_eps);
            let mut mlp_out = self.mlp_forward(&normed, &layer.mlp);
            for i in 0..h {
                residual[i] += mlp_out[i];
            }
            if let Some(cap) = layer_outs.as_mut() {
                cap.push(residual.clone());
            }
            // free
            let _ = &mut mlp_out;
        }
        // final norm
        let mut normed2 = vec![0.0f32; h];
        Self::rmsnorm(&mut normed2, &residual, &self.norm, cfg.rms_eps);
        normed2
    }

    /// Like forward_token but also returns residual after each layer.
    pub fn forward_token_captured(
        &self,
        hidden_in: &[f32],
        pos: usize,
        state: &mut ModelState,
        cos: &[f32],
        sin: &[f32],
        layer_outs: &mut Vec<Vec<f32>>,
    ) -> Vec<f32> {
        self.forward_token_inner(hidden_in, pos, state, cos, sin, Some(layer_outs))
    }

    pub fn mlp_forward(&self, x: &[f32], mlp: &Mlp) -> Vec<f32> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let im = cfg.intermediate_size;
        let mut gate = vec![0.0f32; im];
        let mut up = vec![0.0f32; im];
        Self::matvec(&mut gate, &mlp.gate_proj, x, im, h);
        Self::matvec(&mut up, &mlp.up_proj, x, im, h);
        for i in 0..im {
            gate[i] = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i]; // silu(gate)*up
        }
        let mut out = vec![0.0f32; h];
        Self::matvec(&mut out, &mlp.down_proj, &gate, h, im);
        out
    }

    /// logits = embed_tokens^T @ hidden  (tied head)
    pub fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let v = self.config.vocab_size;
        let h = self.config.hidden_size;
        let mut logits = vec![0.0f32; v];
        Self::matvec(&mut logits, &self.embed_tokens, hidden, v, h);
        logits
    }

    /// embedding lookup
    pub fn embed(&self, token: u32) -> Vec<f32> {
        let h = self.config.hidden_size;
        let base = (token as usize) * h;
        self.embed_tokens[base..base + h].to_vec()
    }

    /// Precompute cos/sin for a position (rotate_half, rotary_dim).
    pub fn rope_tables(&self, pos: usize) -> (Vec<f32>, Vec<f32>) {
        let rdim = self.config.rotary_dim;
        let half = rdim / 2;
        let theta = self.config.rope_theta;
        let mut cos = vec![0.0f32; rdim];
        let mut sin = vec![0.0f32; rdim];
        // inv_freq[i] = theta^(-2i/rdim), i in 0..half ; freq = pos*inv_freq
        // cos/sin length rdim = cat(freq,freq) but for rotate_half we index i and i+half
        // with cos[i]==cos[i+half]==cos(freq_i) (since emb=cat(freq,freq))
        for i in 0..half {
            let inv = theta.powf(-(2.0 * i as f32) / rdim as f32);
            let f = (pos as f32) * inv;
            cos[i] = f.cos();
            sin[i] = f.sin();
            cos[i + half] = cos[i];
            sin[i + half] = sin[i];
        }
        (cos, sin)
    }
}

/// Mutable per-layer inference state (KV for full-attn, conv+recurrent for linear-attn).
pub struct ModelState {
    pub k_caches: Vec<Vec<Vec<f32>>>, // [layer][kv_head] flat [pos*hd..]
    pub v_caches: Vec<Vec<Vec<f32>>>,
    pub conv_states: Vec<Vec<f32>>,   // [layer] flat [conv_dim*ck]
    pub s_states: Vec<Vec<f32>>,      // [layer] flat [num_v_heads*kd*vd]
}

impl ModelState {
    pub fn new(model: &Model) -> Self {
        let cfg = &model.config;
        let mut k_caches = Vec::new();
        let mut v_caches = Vec::new();
        let mut conv_states = Vec::new();
        let mut s_states = Vec::new();
        for lt in &cfg.layer_types {
            match lt {
                LayerType::FullAttention => {
                    let mut kc = Vec::new();
                    let mut vc = Vec::new();
                    for _ in 0..cfg.num_kv_heads {
                        kc.push(Vec::new());
                        vc.push(Vec::new());
                    }
                    k_caches.push(kc);
                    v_caches.push(vc);
                    conv_states.push(Vec::new());
                    s_states.push(Vec::new());
                }
                LayerType::LinearAttention => {
                    let conv_dim = cfg.key_dim() * 2 + cfg.value_dim();
                    conv_states.push(vec![0.0f32; conv_dim * cfg.conv_kernel]);
                    s_states.push(vec![0.0f32; cfg.lin_num_v_heads * cfg.lin_k_dim * cfg.lin_v_dim]);
                    k_caches.push(Vec::new());
                    v_caches.push(Vec::new());
                }
            }
        }
        Self { k_caches, v_caches, conv_states, s_states }
    }
}
