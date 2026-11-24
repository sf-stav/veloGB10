//! The Inkling-2 looped text architecture as the engine needs it, parsed from the pack's
//! `config.json` (`model_type: "inkling2_loop"`, the text part under `text_config`, the bundled
//! drafter under `mtp_config`) and cross-checked against `hf_quant_config.json`.
//!
//! Reference: transformers `models/inkling2_loop/configuration_inkling2_loop.py` (pinned commit in
//! PLAN/INKLING2_S01_REPORT.md). Two places where the checkpoint config and the reference differ
//! and we follow the CHECKPOINT, failing loudly when they disagree:
//!   * layer kinds: the reference derives `layer_types` as `hybrid` iff `(i + 1) % 6 == 0` (a
//!     hard-coded 6) and ignores `local_layer_ids`; we read `local_layer_ids` (sliding = listed)
//!     and, for a config whose `n_layers_per_attn_res_block` is 6, assert the two agree.
//!   * `intermediate_size` in the checkpoint is the MoE expert width (`moe_intermediate_size`).

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct MtpConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub eps: f32,
    pub rope_theta: f64,
    pub local_layer_ids: Vec<usize>,
    pub sliding_window: usize,
    pub block_size: usize,
    pub markov_rank: usize,
    pub markov_head_type: String,
    pub target_layer_ids: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Inkling2Config {
    pub hidden: usize,
    pub layers: usize,
    pub vocab: usize,
    pub unpadded_vocab: usize,
    /// full-attention layers
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// sliding-window layers
    pub swa_heads: usize,
    pub swa_kv_heads: usize,
    pub swa_head_dim: usize,
    pub sliding_window: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub log_floor: f32,
    pub log_alpha: f32,
    pub eps: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub block: usize,
    pub n_experts: usize,
    pub topk: usize,
    pub inter: usize,
    pub route_scale: f32,
    pub gate_softcap: f32,
    pub up_softcap: f32,
    pub logit_softcap: f32,
    pub conv_k: usize,
    pub loop_start: usize,
    pub loop_end: usize,
    pub passes: usize,
    pub gate_rank: usize,
    /// `is_sliding[l]` for the physical layers
    pub is_sliding: Vec<bool>,
    pub eos: Vec<u32>,
    pub image_token_id: u32,
    pub mtp: Option<MtpConfig>,
}

fn u(v: &Value, k: &str) -> Result<usize> {
    v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).with_context(|| format!("config: missing integer `{k}`"))
}
fn uo(v: &Value, k: &str, d: usize) -> usize { v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(d) }
fn fo(v: &Value, k: &str, d: f64) -> f64 { v.get(k).and_then(|x| x.as_f64()).unwrap_or(d) }
fn ids(v: &Value, k: &str) -> Option<Vec<usize>> {
    v.get(k).and_then(|x| x.as_array()).map(|a| a.iter().filter_map(|y| y.as_u64().map(|z| z as usize)).collect())
}

impl Inkling2Config {
    /// True when `dir/config.json` names the looped Inkling-2 family.
    pub fn is_inkling2_dir(dir: &Path) -> bool {
        let Ok(b) = std::fs::read(dir.join("config.json")) else { return false };
        let Ok(v) = serde_json::from_slice::<Value>(&b) else { return false };
        let mt = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
        mt == "inkling2_loop" || mt == "inkling2_loop_text"
    }

    pub fn load(dir: &Path) -> Result<Inkling2Config> {
        let raw = std::fs::read(dir.join("config.json")).with_context(|| format!("{}/config.json", dir.display()))?;
        let root: Value = serde_json::from_slice(&raw)?;
        Self::from_json(&root)
    }

    pub fn from_json(root: &Value) -> Result<Inkling2Config> {
        let mt = root.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
        ensure!(mt == "inkling2_loop" || mt == "inkling2_loop_text", "config: model_type {mt:?} is not inkling2_loop");
        let t = root.get("text_config").unwrap_or(root);
        let layers = u(t, "num_hidden_layers")?;
        let block = uo(t, "n_layers_per_attn_res_block", 6);
        let is_sliding: Vec<bool> = match ids(t, "local_layer_ids") {
            Some(l) => (0..layers).map(|i| l.contains(&i)).collect(),
            None => match t.get("layer_types").and_then(|x| x.as_array()) {
                Some(a) => a.iter().map(|x| x.as_str() == Some("hybrid_sliding")).collect(),
                None => (0..layers).map(|i| (i + 1) % 6 != 0).collect(),
            },
        };
        ensure!(is_sliding.len() == layers, "config: {} layer kinds for {layers} layers", is_sliding.len());
        if block == 6 && t.get("layer_types").is_none() {
            for (i, &s) in is_sliding.iter().enumerate() {
                ensure!(s == ((i + 1) % 6 != 0), "config: local_layer_ids disagree with the reference's (i+1)%6 rule at layer {i}");
            }
        }
        let (mut ls, mut le, mut np, mut gr) = (uo(t, "loop_layer_index_start", 6), uo(t, "loop_layer_index_end", 18),
                                                 uo(t, "loop_num_passes", 3), uo(t, "loop_gate_rank", 64));
        if let Some(lc) = t.get("loop_config") {
            ls = uo(lc, "core_start", ls); le = uo(lc, "core_end", le); np = uo(lc, "num_passes", np); gr = uo(lc, "gate_rank", gr);
            ensure!(!lc.get("is_varloop").and_then(|x| x.as_bool()).unwrap_or(false), "config: is_varloop=true is not supported");
        }
        ensure!(np >= 1 && ls > 0 && ls % block == 0 && le % block == 0 && le <= layers && ls < le,
                "config: loop [{ls}, {le}) x{np} invalid for block {block} / {layers} layers");
        ensure!(uo(t, "n_shared_experts", 0) == 0, "config: n_shared_experts > 0 is not in the reference");
        let sf = t.get("scoring_func").and_then(|x| x.as_str()).unwrap_or("sigmoid");
        ensure!(sf == "sigmoid", "config: scoring_func {sf} (only sigmoid)");
        if let Some(b) = t.get("use_value_norm_no_gamma").and_then(|x| x.as_bool()) { ensure!(b, "config: use_value_norm_no_gamma=false unsupported"); }
        if let Some(b) = t.get("attn_res_normalize_values").and_then(|x| x.as_bool()) { ensure!(b, "config: attn_res_normalize_values=false unsupported"); }
        for k in ["q_bias", "o_bias"] {
            if let Some(b) = t.get(k).and_then(|x| x.as_bool()) { ensure!(!b, "config: {k}=true unsupported"); }
        }
        let vocab = u(t, "vocab_size")?;
        let eos: Vec<u32> = match root.get("eos_token_id").or_else(|| t.get("eos_token_id")) {
            Some(Value::Number(n)) => vec![n.as_u64().unwrap_or(200006) as u32],
            Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_u64().map(|y| y as u32)).collect(),
            _ => vec![200006],
        };
        let heads = uo(t, "num_attention_heads", 24);
        let kv_heads = uo(t, "num_key_value_heads", 8);
        let head_dim = uo(t, "head_dim", 128);
        let cfg = Inkling2Config {
            hidden: u(t, "hidden_size")?,
            layers,
            vocab,
            unpadded_vocab: uo(t, "unpadded_vocab_size", vocab),
            heads, kv_heads, head_dim,
            swa_heads: uo(t, "swa_num_attention_heads", heads),
            swa_kv_heads: uo(t, "swa_num_key_value_heads", kv_heads),
            swa_head_dim: uo(t, "swa_head_dim", head_dim),
            sliding_window: uo(t, "sliding_window_size", 1024),
            d_rel: uo(t, "d_rel", 16),
            rel_extent: uo(t, "rel_extent", 1024),
            log_floor: fo(t, "log_scaling_n_floor", 8192.0) as f32,
            log_alpha: fo(t, "log_scaling_alpha", 0.1) as f32,
            eps: fo(t, "rms_norm_eps", 1e-6) as f32,
            qk_clip: fo(t, "qk_l2_clip_threshold", 22.627416997969522) as f32,
            rel_clip: fo(t, "rel_l2_clip_threshold", 8.0) as f32,
            block,
            n_experts: t.get("n_routed_experts").or_else(|| t.get("num_local_experts")).and_then(|x| x.as_u64()).unwrap_or(128) as usize,
            topk: uo(t, "num_experts_per_tok", 8),
            inter: t.get("moe_intermediate_size").or_else(|| t.get("intermediate_size")).and_then(|x| x.as_u64()).unwrap_or(1152) as usize,
            route_scale: t.get("route_scale").or_else(|| t.get("routed_scaling_factor")).and_then(|x| x.as_f64()).unwrap_or(8.0) as f32,
            gate_softcap: fo(t, "mlp_gate_softcap", 4.0) as f32,
            up_softcap: fo(t, "mlp_up_softcap", 25.0) as f32,
            logit_softcap: fo(t, "final_logit_softcapping", 30.0) as f32,
            conv_k: t.get("sconv_kernel_size").or_else(|| t.get("conv_kernel_size")).and_then(|x| x.as_u64()).unwrap_or(4) as usize,
            loop_start: ls, loop_end: le, passes: np, gate_rank: gr,
            is_sliding,
            eos,
            image_token_id: root.get("image_token_id").and_then(|x| x.as_u64()).unwrap_or(200054) as u32,
            mtp: match root.get("mtp_config") { Some(m) => Some(MtpConfig::from_json(m)?), None => None },
        };
        ensure!(cfg.unpadded_vocab <= cfg.vocab, "config: unpadded_vocab_size > vocab_size");
        ensure!(cfg.heads % cfg.kv_heads == 0 && cfg.swa_heads % cfg.swa_kv_heads == 0, "config: heads not a multiple of kv heads");
        ensure!(cfg.hidden % 32 == 0 && cfg.inter % 32 == 0, "config: hidden/inter must be multiples of 32 (MXFP8 blocks)");
        if cfg.conv_k == 0 { bail!("config: conv kernel size 0"); }
        Ok(cfg)
    }

    pub fn is_loop(&self, l: usize) -> bool { l >= self.loop_start && l < self.loop_end }
    pub fn is_boundary(&self, l: usize) -> bool { l % self.block == 0 }
    pub fn loop_len(&self) -> usize { self.loop_end - self.loop_start }
    /// Logical layer visits per token (48 on the stock config).
    pub fn visits(&self) -> usize { self.layers + (self.passes - 1) * self.loop_len() }
    /// (heads, kv_heads, head_dim) of physical layer `l`.
    pub fn attn_geom(&self, l: usize) -> (usize, usize, usize) {
        if self.is_sliding[l] { (self.swa_heads, self.swa_kv_heads, self.swa_head_dim) } else { (self.heads, self.kv_heads, self.head_dim) }
    }
    /// Relative-bias extent of layer `l` (the sliding window on sliding layers, `rel_extent` on full ones).
    pub fn rel_ext(&self, l: usize) -> usize { if self.is_sliding[l] { self.sliding_window } else { self.rel_extent } }
    /// The logical cache slot of physical layer `l` in loop pass `p` (pass 0 and non-loop layers: `l`).
    pub fn cache_slot(&self, l: usize, pass: usize) -> usize {
        if pass == 0 { l } else { self.layers + (pass - 1) * self.loop_len() + (l - self.loop_start) }
    }
}

impl MtpConfig {
    fn from_json(m: &Value) -> Result<MtpConfig> {
        Ok(MtpConfig {
            hidden: u(m, "hidden_size")?,
            layers: u(m, "num_hidden_layers")?,
            heads: u(m, "num_attention_heads")?,
            kv_heads: u(m, "num_key_value_heads")?,
            head_dim: u(m, "head_dim")?,
            inter: u(m, "intermediate_size")?,
            eps: fo(m, "rms_norm_eps", 1e-6) as f32,
            rope_theta: fo(m, "rope_theta", 1e6),
            local_layer_ids: ids(m, "local_layer_ids").unwrap_or_default(),
            sliding_window: uo(m, "sliding_window_size", 1024),
            block_size: uo(m, "block_size", 8),
            markov_rank: uo(m, "markov_rank", 256),
            markov_head_type: m.get("markov_head_type").and_then(|x| x.as_str()).unwrap_or("").to_string(),
            target_layer_ids: ids(m, "target_layer_ids").unwrap_or_default(),
        })
    }
}
