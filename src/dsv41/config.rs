//! DSV4.1 (leg I0): the model configuration, parsed from a pack's `config.json` (`text_config`,
//! `vision_config`, `quantization_config`) plus the EXL3 pack's `quantization_config.json` when present.
//!
//! Every field of `text_config` is parsed into a typed field (no defaults are invented: a missing
//! required key is an error naming it), every key the parser does not know is listed in
//! `Dsv41Config::unknown_keys` (a newer pack's new field is noticed, never silently ignored), and
//! [`Dsv41Config::validate`] checks the cross-field invariants the forward relies on (43 compress
//! ratios, the KV / index / candidate source wiring, the Engram prime layout sum, the shape products).
//! [`diff_dirs`] lists every difference between two packs' configs.

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::path::Path;

/// `rope_scaling` (YaRN).
#[derive(Clone, Debug, PartialEq)]
pub struct Rope {
    pub theta: f64,
    pub kind: String,
    pub factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub original_max_position_embeddings: u64,
    /// compressed KV rotates at its own base (`compress_rope_theta`)
    pub compress_theta: f64,
}

/// `vision_config` (presence = the vision path exists; M4 uses the fields).
#[derive(Clone, Debug, PartialEq)]
pub struct Vision {
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub inter: usize,
    pub patch: usize,
    pub rope_theta: f64,
    pub downsample: usize,
    pub max_image_tokens: usize,
    pub min_pixels: u64,
    pub max_wh_ratio: Option<f64>,
}

/// The pack's quantization: config.json's block, plus `quantization_config.json` when the pack has one.
#[derive(Clone, Debug, PartialEq)]
pub struct Quant {
    /// "fp8" (DeepSeek native: FP8 32x32 ue8m0 blocks + FP4 experts), "exl3", or whatever the pack says
    pub method: String,
    /// the raw block from config.json
    pub block: Value,
    /// `quantization_config.json` was found (EXL3 packs: per-tensor `tensor_storage`)
    pub has_file: bool,
    /// number of `tensor_storage` entries in the file (0 without one)
    pub tensor_storage_entries: usize,
}

/// DeepSeek-V4.1-Flash text model configuration (`text_config`), typed.
#[derive(Clone, Debug, PartialEq)]
pub struct Dsv41Config {
    pub model_type: String,
    pub text_model_type: String,
    pub bos: u32,
    pub eos: u32,
    pub pad: u32,
    pub image_token_id: u32,
    pub vocab: usize,
    pub hidden: usize,
    pub moe_inter: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub hidden_act: String,
    pub swiglu_limit: f64,
    pub rms_eps: f64,
    pub max_pos: usize,
    pub rope: Rope,
    pub tie_embeddings: bool,
    pub n_routed: usize,
    pub n_shared: usize,
    pub top_k: usize,
    pub scoring_func: String,
    pub topk_method: String,
    pub norm_topk_prob: bool,
    pub routed_scale: f64,
    pub window: usize,
    /// one entry per layer, DSpark stages included (0 = SWA only, r = compressed r:1)
    pub compress_ratios: Vec<u32>,
    pub kv_sources: Vec<usize>,
    pub index_sources: Vec<usize>,
    pub index_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    /// None = candidate pre-filtering off (`candidate_source_layer_id` < 0)
    pub candidate_source: Option<usize>,
    pub candidate_topk_blocks: usize,
    pub candidate_block: usize,
    pub hc: usize,
    pub sinkhorn_iters: usize,
    pub hc_eps: f64,
    pub engram_layers: Vec<usize>,
    pub engram_rows: Vec<u64>,
    pub engram_max_ngram: usize,
    pub engram_vocab: u64,
    pub engram_heads: usize,
    pub engram_head_dim: usize,
    pub engram_pad: u32,
    pub engram_compressed_vocab: usize,
    pub dspark_stages: usize,
    pub dspark_block: usize,
    pub dspark_noise_token: u32,
    pub dspark_targets: Vec<usize>,
    pub dspark_markov_rank: usize,
    pub dspark_routed: usize,
    pub dspark_top_k: usize,
    pub vision: Option<Vision>,
    pub quant: Quant,
    /// `engram_table_dir` (top level; the EXL3 pack carries it, null)
    pub engram_table_dir: Option<String>,
    /// keys present in the file that this parser does not know (path form, e.g. `text_config.foo`)
    pub unknown_keys: Vec<String>,
    /// a slice pack written by `--dsv41-slice-pack`: the real layer ids it keeps
    pub slice_layers: Option<Vec<usize>>,
}

/// A key reader that remembers what it consumed, so unknown keys can be listed.
struct Rd<'a> { v: &'a Value, path: &'static str, seen: Vec<&'static str> }

impl<'a> Rd<'a> {
    fn new(v: &'a Value, path: &'static str) -> Self { Rd { v, path, seen: Vec::new() } }
    fn raw(&mut self, k: &'static str) -> Option<&'a Value> { self.seen.push(k); self.v.get(k).filter(|x| !x.is_null()) }
    fn req(&mut self, k: &'static str) -> Result<&'a Value> { let p = self.path; self.raw(k).with_context(|| format!("{p}.{k} missing")) }
    fn u(&mut self, k: &'static str) -> Result<usize> { let p = self.path; self.req(k)?.as_u64().map(|x| x as usize).with_context(|| format!("{p}.{k} is not an unsigned integer")) }
    fn i(&mut self, k: &'static str) -> Result<i64> { let p = self.path; self.req(k)?.as_i64().with_context(|| format!("{p}.{k} is not an integer")) }
    fn f(&mut self, k: &'static str) -> Result<f64> { let p = self.path; self.req(k)?.as_f64().with_context(|| format!("{p}.{k} is not a number")) }
    fn s(&mut self, k: &'static str) -> Result<String> { let p = self.path; Ok(self.req(k)?.as_str().with_context(|| format!("{p}.{k} is not a string"))?.to_string()) }
    fn b(&mut self, k: &'static str) -> Result<bool> { let p = self.path; self.req(k)?.as_bool().with_context(|| format!("{p}.{k} is not a bool")) }
    fn ul(&mut self, k: &'static str) -> Result<Vec<u64>> {
        let p = self.path;
        self.req(k)?.as_array().with_context(|| format!("{p}.{k} is not a list"))?
            .iter().map(|x| x.as_u64().with_context(|| format!("{p}.{k} has a non-integer entry"))).collect()
    }
    fn unknown(&self, out: &mut Vec<String>) {
        if let Some(o) = self.v.as_object() {
            for k in o.keys() { if !self.seen.contains(&k.as_str()) { out.push(format!("{}.{k}", self.path)); } }
        }
    }
}

fn read_json(p: &Path) -> Result<Value> {
    let t = std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?;
    serde_json::from_str(&t).with_context(|| format!("parse {}", p.display()))
}

/// Is `dir` a DeepSeek-V4.1 pack (config.json `model_type` = "deepseek_v41", any weight format)?
pub fn is_dsv41_dir(dir: &str) -> bool {
    std::fs::read_to_string(Path::new(dir).join("config.json")).ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v.get("model_type").and_then(Value::as_str).map(|m| m == "deepseek_v41"))
        .unwrap_or(false)
}

impl Dsv41Config {
    /// Parse `dir/config.json` (+ `dir/quantization_config.json` presence). Does not validate.
    pub fn load(dir: &Path) -> Result<Self> {
        let v = read_json(&dir.join("config.json"))?;
        let qfile = dir.join("quantization_config.json");
        let qf = if qfile.exists() { Some(read_json(&qfile)?) } else { None };
        Self::from_value(&v, qf.as_ref())
    }

    /// Parse from the JSON values (the unit tests build synthetic configs this way).
    pub fn from_value(v: &Value, qfile: Option<&Value>) -> Result<Self> {
        let mut unknown = Vec::new();
        let mut top = Rd::new(v, "config");
        let model_type = top.s("model_type")?;
        let bos = top.u("bos_token_id")? as u32;
        let eos = top.u("eos_token_id")? as u32;
        let pad = top.u("pad_token_id")? as u32;
        let image_token_id = top.u("image_token_id")? as u32;
        for k in ["architectures", "dtype", "transformers_version"] { top.raw(k); }
        let qblock = top.req("quantization_config")?.clone();
        let engram_table_dir = top.raw("engram_table_dir").and_then(Value::as_str).map(str::to_string);
        let slice_layers = top.raw("dsv41_slice").and_then(|s| s.get("layers")).and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect());
        let tv = top.req("text_config")?;
        let vv = top.raw("vision_config");
        top.unknown(&mut unknown);

        let mut t = Rd::new(tv, "text_config");
        let text_model_type = t.s("model_type")?;
        let vocab = t.u("vocab_size")?;
        let hidden = t.u("hidden_size")?;
        let moe_inter = t.u("moe_intermediate_size")?;
        let layers = t.u("num_hidden_layers")?;
        let heads = t.u("num_attention_heads")?;
        let kv_heads = t.u("num_key_value_heads")?;
        let head_dim = t.u("head_dim")?;
        let rope_dim = t.u("qk_rope_head_dim")?;
        let q_lora_rank = t.u("q_lora_rank")?;
        let o_lora_rank = t.u("o_lora_rank")?;
        let o_groups = t.u("o_groups")?;
        let hidden_act = t.s("hidden_act")?;
        let swiglu_limit = t.f("swiglu_limit")?;
        let rms_eps = t.f("rms_norm_eps")?;
        ensure!(!t.b("attention_bias")?, "text_config.attention_bias = true is not a V4.1 layout");
        for k in ["attention_dropout", "initializer_range", "use_cache"] { t.raw(k); }
        let tie_embeddings = t.b("tie_word_embeddings")?;
        let max_pos = t.u("max_position_embeddings")?;
        let theta = t.f("rope_theta")?;
        let rs = t.req("rope_scaling")?;
        let mut r = Rd::new(rs, "text_config.rope_scaling");
        let rope = Rope {
            theta,
            kind: r.s("rope_type")?,
            factor: r.f("factor")?,
            beta_fast: r.f("beta_fast")?,
            beta_slow: r.f("beta_slow")?,
            original_max_position_embeddings: r.u("original_max_position_embeddings")? as u64,
            compress_theta: t.f("compress_rope_theta")?,
        };
        r.unknown(&mut unknown);
        let n_routed = t.u("n_routed_experts")?;
        let n_shared = t.u("n_shared_experts")?;
        let top_k = t.u("num_experts_per_tok")?;
        let scoring_func = t.s("scoring_func")?;
        let topk_method = t.s("topk_method")?;
        let norm_topk_prob = t.b("norm_topk_prob")?;
        let routed_scale = t.f("routed_scaling_factor")?;
        let window = t.u("sliding_window")?;
        let compress_ratios = t.ul("compress_ratios")?.into_iter().map(|x| x as u32).collect();
        let kv_sources = t.ul("kv_source_layer_ids")?.into_iter().map(|x| x as usize).collect();
        let index_sources = t.ul("index_source_layer_ids")?.into_iter().map(|x| x as usize).collect();
        let index_heads = t.u("index_n_heads")?;
        let index_head_dim = t.u("index_head_dim")?;
        let index_topk = t.u("index_topk")?;
        let cs = t.i("candidate_source_layer_id")?;
        let candidate_source = if cs < 0 { None } else { Some(cs as usize) };
        let candidate_topk_blocks = t.u("candidate_topk_blocks")?;
        let candidate_block = t.u("candidate_block_size")?;
        let hc = t.u("hc_mult")?;
        let sinkhorn_iters = t.u("hc_sinkhorn_iters")?;
        let hc_eps = t.f("hc_eps")?;
        let engram_layers = t.ul("engram_layer_ids")?.into_iter().map(|x| x as usize).collect();
        let engram_rows = t.ul("engram_num_embeddings")?;
        let engram_max_ngram = t.u("engram_max_ngram_size")?;
        let engram_vocab = t.u("engram_vocab_size")? as u64;
        let engram_heads = t.u("engram_n_heads")?;
        let engram_head_dim = t.u("engram_head_dim")?;
        let engram_pad = t.u("engram_pad_token_id")? as u32;
        let engram_compressed_vocab = t.u("engram_compressed_vocab_size")?;
        let dspark_stages = t.u("num_nextn_predict_layers")?;
        let dspark_block = t.u("dspark_block_size")?;
        let dspark_noise_token = t.u("dspark_noise_token_id")? as u32;
        let dspark_targets = t.ul("dspark_target_layer_ids")?.into_iter().map(|x| x as usize).collect();
        let dspark_markov_rank = t.u("dspark_markov_rank")?;
        let dspark_routed = t.u("dspark_n_routed_experts")?;
        let dspark_top_k = t.u("dspark_num_experts_per_tok")?;
        t.unknown(&mut unknown);

        let vision = match vv {
            None => None,
            Some(vv) => {
                let mut w = Rd::new(vv, "vision_config");
                w.raw("model_type");
                let vis = Vision {
                    layers: w.u("num_hidden_layers")?,
                    hidden: w.u("hidden_size")?,
                    heads: w.u("num_attention_heads")?,
                    inter: w.u("intermediate_size")?,
                    patch: w.u("patch_size")?,
                    rope_theta: w.f("rope_theta")?,
                    downsample: w.u("downsample_ratio")?,
                    max_image_tokens: w.u("max_image_tokens")?,
                    min_pixels: w.u("min_pixels")? as u64,
                    max_wh_ratio: w.raw("max_wh_ratio").and_then(Value::as_f64),
                };
                w.unknown(&mut unknown);
                Some(vis)
            }
        };
        let method = qfile.and_then(|q| q.get("quant_method")).or_else(|| qblock.get("quant_method"))
            .and_then(Value::as_str).unwrap_or("unknown").to_string();
        let quant = Quant {
            method,
            block: qblock,
            has_file: qfile.is_some(),
            tensor_storage_entries: qfile.and_then(|q| q.get("tensor_storage")).and_then(Value::as_object).map_or(0, |o| o.len()),
        };
        Ok(Dsv41Config {
            model_type, text_model_type, bos, eos, pad, image_token_id, vocab, hidden, moe_inter, layers, heads, kv_heads,
            head_dim, rope_dim, q_lora_rank, o_lora_rank, o_groups, hidden_act, swiglu_limit, rms_eps, max_pos, rope,
            tie_embeddings, n_routed, n_shared, top_k, scoring_func, topk_method, norm_topk_prob, routed_scale, window,
            compress_ratios, kv_sources, index_sources, index_heads, index_head_dim, index_topk, candidate_source,
            candidate_topk_blocks, candidate_block, hc, sinkhorn_iters, hc_eps, engram_layers, engram_rows,
            engram_max_ngram, engram_vocab, engram_heads, engram_head_dim, engram_pad, engram_compressed_vocab,
            dspark_stages, dspark_block, dspark_noise_token, dspark_targets, dspark_markov_rank, dspark_routed,
            dspark_top_k, vision, quant, engram_table_dir, unknown_keys: unknown, slice_layers,
        })
    }

    // ---- derived quantities (the forward and the census use these, never literals) ----

    /// Total layer slots: the trunk plus the DSpark stages (ids `layers..layers+dspark_stages`).
    pub fn total_layers(&self) -> usize { self.layers + self.dspark_stages }
    pub fn is_dspark(&self, layer: usize) -> bool { layer >= self.layers && layer < self.total_layers() }
    pub fn ratio(&self, layer: usize) -> u32 { self.compress_ratios.get(layer).copied().unwrap_or(0) }
    /// `n_hash_cols` = (max_ngram - 1) x heads (24): rows fetched per token per Engram layer.
    pub fn engram_cols(&self) -> usize { (self.engram_max_ngram - 1) * self.engram_heads }
    /// The hc mixing projection rows: (2 + hc) x hc (24).
    pub fn hc_mix(&self) -> usize { (2 + self.hc) * self.hc }
    /// Routed experts and top-k of a layer slot (DSpark stages have their own).
    pub fn moe_of(&self, layer: usize) -> (usize, usize) {
        if self.is_dspark(layer) { (self.dspark_routed, self.dspark_top_k) } else { (self.n_routed, self.top_k) }
    }
    /// The KV source a compressing layer reads (`kv_source_layers`: the last source at or below it).
    pub fn kv_source_of(&self, layer: usize) -> Option<usize> {
        if self.ratio(layer) == 0 || layer >= self.layers { return None; }
        self.kv_sources.iter().copied().filter(|&s| s <= layer).max()
    }
    /// The index source whose top-k a compressing layer uses.
    pub fn index_source_of(&self, layer: usize) -> Option<usize> {
        if self.ratio(layer) == 0 || layer >= self.layers { return None; }
        self.index_sources.iter().copied().filter(|&s| s <= layer).max()
    }
    /// Does this index source mask its scores with the candidate blocks of the candidate source?
    pub fn uses_candidates(&self, layer: usize) -> bool {
        self.index_sources.contains(&layer) && self.candidate_source.map_or(false, |c| c < layer)
    }
    /// The attention class of a layer slot (what the forward dispatches on).
    pub fn attn_class(&self, layer: usize) -> AttnClass {
        if self.is_dspark(layer) { return AttnClass::DsparkStage; }
        let r = self.ratio(layer);
        if r == 0 { return AttnClass::SwaOnly; }
        let kv_src = self.kv_sources.contains(&layer);
        let idx_src = self.index_sources.contains(&layer);
        AttnClass::Compressed { ratio: r, kv_source: kv_src, index_source: idx_src,
                                candidate_source: self.candidate_source == Some(layer),
                                uses_candidates: self.uses_candidates(layer) }
    }

    /// The cross-field invariants. Errors name the field; Ok carries the list of facts checked.
    pub fn validate(&self) -> Result<Vec<String>> {
        let mut ok = Vec::new();
        let mut chk = |cond: bool, what: String| -> Result<()> { if cond { ok.push(what); Ok(()) } else { bail!("config: {what} FAILED") } };
        chk(self.model_type == "deepseek_v41", format!("model_type {} = deepseek_v41", self.model_type))?;
        chk(self.compress_ratios.len() == self.total_layers(),
            format!("compress_ratios has {} entries = {} layers + {} DSpark stages", self.compress_ratios.len(), self.layers, self.dspark_stages))?;
        chk((self.layers..self.total_layers()).all(|l| self.ratio(l) == 0), "DSpark stages are sliding-window only (ratio 0)".into())?;
        chk(self.kv_heads == 1, format!("one shared KV head (MQA), num_key_value_heads {}", self.kv_heads))?;
        chk(self.rope_dim < self.head_dim && self.rope_dim % 2 == 0, format!("rope dim {} inside head_dim {}", self.rope_dim, self.head_dim))?;
        chk(self.heads % self.o_groups == 0, format!("o_groups {} divides heads {}", self.o_groups, self.heads))?;
        for &s in &self.kv_sources {
            chk(s < self.layers && self.ratio(s) > 0, format!("KV source {s} compresses (ratio {})", self.ratio(s)))?;
        }
        for &s in &self.index_sources {
            chk(s < self.layers && self.ratio(s) > 0, format!("index source {s} compresses (ratio {})", self.ratio(s)))?;
        }
        chk(self.kv_sources.iter().all(|s| self.index_sources.contains(s)), "every KV source is an index source (it owns the index-K cache)".into())?;
        for l in 0..self.layers {
            if self.ratio(l) == 0 { continue; }
            let kv = self.kv_source_of(l);
            let ix = self.index_source_of(l);
            chk(kv.map_or(false, |s| self.ratio(s) == self.ratio(l)), format!("layer {l} (ratio {}) reads KV source {kv:?} of the same ratio", self.ratio(l)))?;
            chk(ix.map_or(false, |s| self.ratio(s) == self.ratio(l)), format!("layer {l} uses index source {ix:?} of the same ratio"))?;
        }
        if let Some(c) = self.candidate_source {
            chk(self.index_sources.contains(&c) && self.kv_sources.contains(&c), format!("candidate source {c} is a KV and index source"))?;
            chk(self.candidate_topk_blocks > 0 && self.candidate_block > 0, format!("candidate pool {} blocks x {}", self.candidate_topk_blocks, self.candidate_block))?;
        }
        chk(self.engram_layers.len() == self.engram_rows.len(), format!("{} Engram layers have {} table sizes", self.engram_layers.len(), self.engram_rows.len()))?;
        chk(self.engram_layers.iter().all(|&l| l < self.layers), "Engram layers are trunk layers".into())?;
        chk(self.engram_max_ngram >= 2 && self.engram_heads > 0, format!("Engram {} heads x n-gram orders 2..={}", self.engram_heads, self.engram_max_ngram))?;
        let lay = crate::dsv41::rankmap::engram_layout(self);
        for (i, rows) in self.engram_rows.iter().enumerate() {
            let sum: u64 = lay.primes[i].iter().flatten().sum();
            chk(sum == *rows, format!("Engram layer {} table rows {rows} = sum of its {} prime buckets ({sum})", self.engram_layers[i], self.engram_cols()))?;
        }
        if let Some(d) = self.first_decoder_layer() {
            chk(self.kv_sources.contains(&d), format!("CED: the first decoder layer {d} is a KV source"))?;
            chk((d..self.layers).all(|l| self.kv_source_of(l) == Some(d)), format!("CED: every decoder layer {d}..{} reads layer {d}'s compressed KV", self.layers - 1))?;
            chk(self.engram_layers.iter().all(|&l| l < d), format!("CED: the Engram layers {:?} are encoder layers (< {d})", self.engram_layers))?;
        }
        chk(self.swa_slots() == self.window + self.dspark_block + 1, format!("SWA ring {} = window {} + backtracking {}", self.swa_slots(), self.window, self.backtrack()))?;
        chk(self.dspark_targets.iter().all(|&l| l < self.layers), format!("DSpark targets {:?} are trunk layers", self.dspark_targets))?;
        chk(self.dspark_stages == 0 || (self.dspark_block > 0 && !self.dspark_targets.is_empty()), "DSpark block and targets set".into())?;
        chk(self.n_shared == 1, format!("one shared expert (n_shared_experts {})", self.n_shared))?;
        chk(self.top_k <= self.n_routed && self.dspark_top_k <= self.dspark_routed, "top-k <= experts (trunk and DSpark)".into())?;
        chk(self.scoring_func == "sqrtsoftplus" && self.topk_method == "noaux_tc", format!("router {} / {}", self.scoring_func, self.topk_method))?;
        chk(self.rope.kind == "yarn", format!("rope_scaling {} (YaRN)", self.rope.kind))?;
        chk(self.rms_eps > 0.0 && self.rms_eps < 1e-5, format!("rms eps {:e}", self.rms_eps))?;
        chk(self.index_topk > 0 && self.index_heads * self.index_head_dim > 0, format!("indexer {} heads x {} top-{}", self.index_heads, self.index_head_dim, self.index_topk))?;
        chk(!self.tie_embeddings, "untied embedding and head".into())?;
        Ok(ok)
    }

    /// The attention wiring of the trunk as runs of consecutive layers that share it:
    /// (first, last, ratio, KV source, index source, own indexer?, uses candidates?).
    pub fn wiring(&self) -> Vec<(usize, usize, u32, Option<usize>, Option<usize>, bool, bool)> {
        let mut out: Vec<(usize, usize, u32, Option<usize>, Option<usize>, bool, bool)> = Vec::new();
        for l in 0..self.layers {
            let k = (self.ratio(l), self.kv_source_of(l), self.index_source_of(l), self.index_sources.contains(&l), self.uses_candidates(l));
            match out.last_mut() {
                Some(r) if (r.2, r.3, r.4, r.5, r.6) == k && !k.3 && r.1 + 1 == l => r.1 = l,
                _ => out.push((l, l, k.0, k.1, k.2, k.3, k.4)),
            }
        }
        out
    }

    /// Speculative backtracking rows every ring keeps beyond its window: the draft block + 1 (DeepSeek's
    /// own engine: NEXTN + 1 = 6 for block 5; PLAN/DSV41_XREF_DEEPINFER.md §4.1).
    pub fn backtrack(&self) -> usize { self.dspark_block + 1 }
    /// SWA ring slots per layer slot (trunk and DSpark stages): window + backtracking = 134.
    pub fn swa_slots(&self) -> usize { self.window + self.backtrack() }
    /// Compressor frontier ring rows of a ratio-r KV source: r + backtracking - 1 (7 at ratio 2).
    pub fn frontier_rows(&self, ratio: u32) -> usize { ratio as usize + self.backtrack() - 1 }
    /// The CED split: the first trunk layer with compress ratio 1 is the first DECODER layer (20);
    /// every layer before it is an encoder layer. Derived from `compress_ratios`, never hard-coded.
    pub fn first_decoder_layer(&self) -> Option<usize> { (0..self.layers).find(|&l| self.ratio(l) == 1) }
    /// The CED role of a layer slot.
    pub fn layer_role(&self, layer: usize) -> LayerRole {
        if self.is_dspark(layer) { return LayerRole::DsparkStage; }
        match self.first_decoder_layer() {
            Some(d) if layer >= d => LayerRole::Decoder { kv_on_all_rows: layer == d && self.kv_sources.contains(&layer) },
            _ => LayerRole::Encoder,
        }
    }

    /// A one-line summary (boot rows, census header).
    pub fn summary(&self) -> String {
        format!("{} layers + {} DSpark stages, hidden {}, {} heads x {} (rope {}), {} routed experts top-{} + {} shared (DSpark {} top-{}), vocab {}, context {}",
                self.layers, self.dspark_stages, self.hidden, self.heads, self.head_dim, self.rope_dim, self.n_routed, self.top_k,
                self.n_shared, self.dspark_routed, self.dspark_top_k, self.vocab, self.max_pos)
    }
}

/// The CED (compressed-encoder prefill) role of a layer slot. CED runs the ENCODER layers on every
/// prompt row; the first decoder layer computes its KV-source work (compressor -> main KV, index K)
/// for every prompt row from its attention-norm input (exact), then the DECODER layers run only the
/// kept tail rows (queries, candidates, SWA, MoE). `exact` prefill runs every layer on every row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerRole {
    Encoder,
    /// `kv_on_all_rows`: this decoder layer's compressor / index-K run on ALL prompt rows (layer 20)
    Decoder { kv_on_all_rows: bool },
    DsparkStage,
}

/// The attention dispatch class of a layer slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnClass {
    /// ratio 0: sliding window only (layers 0, 1)
    SwaOnly,
    Compressed { ratio: u32, kv_source: bool, index_source: bool, candidate_source: bool, uses_candidates: bool },
    DsparkStage,
}

/// Every differing JSON path between two packs' `config.json` (`a` vs `b`), as `path: a | b` lines.
pub fn diff_dirs(a: &Path, b: &Path) -> Result<Vec<String>> {
    let va = read_json(&a.join("config.json"))?;
    let vb = read_json(&b.join("config.json"))?;
    let mut out = Vec::new();
    diff_values("", &va, &vb, &mut out);
    Ok(out)
}

fn diff_values(path: &str, a: &Value, b: &Value, out: &mut Vec<String>) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let mut keys: Vec<&String> = x.keys().chain(y.keys()).collect();
            keys.sort();
            keys.dedup();
            for k in keys {
                let p = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                match (x.get(k), y.get(k)) {
                    (Some(u), Some(v)) => diff_values(&p, u, v, out),
                    (Some(u), None) => out.push(format!("{p}: {} | (absent)", short(u))),
                    (None, Some(v)) => out.push(format!("{p}: (absent) | {}", short(v))),
                    (None, None) => {}
                }
            }
        }
        _ if a == b => {}
        _ => out.push(format!("{path}: {} | {}", short(a), short(b))),
    }
}

fn short(v: &Value) -> String {
    let s = v.to_string();
    if s.len() > 120 { format!("{}...", &s[..117]) } else { s }
}

/// The reference's `inference/config.json` (ModelArgs names) checked field by field against the
/// HF `text_config` this parser read: every mapped key must agree. Returns the mismatches.
pub fn check_reference_args(cfg: &Dsv41Config, args_json: &Path) -> Result<Vec<String>> {
    let a = read_json(args_json)?;
    let mut bad = Vec::new();
    let num = |k: &str| a.get(k).and_then(Value::as_f64);
    let mut eq = |k: &str, want: f64| {
        match num(k) { Some(g) if (g - want).abs() <= want.abs() * 1e-12 => {}, g => bad.push(format!("{k}: reference {g:?} vs config {want}")) }
    };
    eq("vocab_size", cfg.vocab as f64); eq("dim", cfg.hidden as f64); eq("moe_inter_dim", cfg.moe_inter as f64);
    eq("n_layers", cfg.layers as f64); eq("n_mtp_layers", cfg.dspark_stages as f64); eq("dspark_block_size", cfg.dspark_block as f64);
    eq("dspark_noise_token_id", cfg.dspark_noise_token as f64); eq("dspark_markov_rank", cfg.dspark_markov_rank as f64);
    eq("dspark_n_routed_experts", cfg.dspark_routed as f64); eq("dspark_n_activated_experts", cfg.dspark_top_k as f64);
    eq("n_heads", cfg.heads as f64); eq("n_routed_experts", cfg.n_routed as f64); eq("n_shared_experts", cfg.n_shared as f64);
    eq("n_activated_experts", cfg.top_k as f64); eq("route_scale", cfg.routed_scale); eq("swiglu_limit", cfg.swiglu_limit);
    eq("q_lora_rank", cfg.q_lora_rank as f64); eq("head_dim", cfg.head_dim as f64); eq("rope_head_dim", cfg.rope_dim as f64);
    eq("norm_eps", cfg.rms_eps); eq("o_groups", cfg.o_groups as f64); eq("o_lora_rank", cfg.o_lora_rank as f64);
    eq("window_size", cfg.window as f64); eq("original_seq_len", cfg.rope.original_max_position_embeddings as f64);
    eq("rope_theta", cfg.rope.theta); eq("rope_factor", cfg.rope.factor); eq("beta_fast", cfg.rope.beta_fast); eq("beta_slow", cfg.rope.beta_slow);
    eq("index_n_heads", cfg.index_heads as f64); eq("index_head_dim", cfg.index_head_dim as f64); eq("index_topk", cfg.index_topk as f64);
    eq("candidate_source_layer", cfg.candidate_source.map_or(-1.0, |c| c as f64));
    eq("candidate_topk_blocks", cfg.candidate_topk_blocks as f64); eq("candidate_block_size", cfg.candidate_block as f64);
    eq("hc_mult", cfg.hc as f64); eq("hc_sinkhorn_iters", cfg.sinkhorn_iters as f64); eq("hc_eps", cfg.hc_eps);
    eq("engram_vocab_size", cfg.engram_vocab as f64); eq("engram_max_ngram_size", cfg.engram_max_ngram as f64);
    eq("engram_pad_id", cfg.engram_pad as f64); eq("engram_compressed_vocab_size", cfg.engram_compressed_vocab as f64);
    eq("engram_n_heads", cfg.engram_heads as f64); eq("engram_head_dim", cfg.engram_head_dim as f64);
    eq("compress_rope_theta", cfg.rope.compress_theta); eq("image_token_id", cfg.image_token_id as f64);
    let list = |k: &str| a.get(k).and_then(Value::as_array).map(|v| v.iter().filter_map(Value::as_u64).collect::<Vec<u64>>());
    let mut leq = |k: &str, want: Vec<u64>| if list(k).as_ref() != Some(&want) { bad.push(format!("{k}: reference {:?} vs config {want:?}", list(k))) };
    leq("compress_ratios", cfg.compress_ratios.iter().map(|&x| x as u64).collect());
    leq("kv_source_layers", cfg.kv_sources.iter().map(|&x| x as u64).collect());
    leq("index_source_layers", cfg.index_sources.iter().map(|&x| x as u64).collect());
    leq("engram_layer_ids", cfg.engram_layers.iter().map(|&x| x as u64).collect());
    leq("engram_num_embeddings", cfg.engram_rows.clone());
    leq("dspark_target_layer_ids", cfg.dspark_targets.iter().map(|&x| x as u64).collect());
    if let Some(v) = &cfg.vision {
        let mut veq = |k: &str, want: f64| match a.get(k).and_then(Value::as_f64) { Some(g) if g == want => {}, g => bad.push(format!("{k}: reference {g:?} vs vision_config {want}")) };
        veq("vision_n_layers", v.layers as f64); veq("vision_dim", v.hidden as f64); veq("vision_n_heads", v.heads as f64);
        veq("vision_inter_dim", v.inter as f64); veq("vision_patch_size", v.patch as f64); veq("vision_downsample_ratio", v.downsample as f64);
        veq("vision_max_n_token", v.max_image_tokens as f64); veq("vision_min_pixels", v.min_pixels as f64); veq("vision_rope_theta", v.rope_theta);
    }
    Ok(bad)
}

/// The real V4.1-Flash config.json text, for unit tests that must not depend on a model dir.
#[cfg(test)]
pub(crate) fn test_config_json() -> Value {
    serde_json::from_str(r#"{
      "architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "dtype": "bfloat16",
      "transformers_version": "5.6.0", "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2, "image_token_id": 129264,
      "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32],
                              "scale_fmt": "ue8m0", "expert_dtype": "fp4"},
      "text_config": {
        "model_type": "deepseek_v41_text", "vocab_size": 129280, "hidden_size": 5120, "moe_intermediate_size": 2304,
        "num_hidden_layers": 40, "num_attention_heads": 64, "num_key_value_heads": 1, "head_dim": 512, "qk_rope_head_dim": 64,
        "q_lora_rank": 1280, "o_lora_rank": 1024, "o_groups": 8, "hidden_act": "silu", "swiglu_limit": 10.0,
        "rms_norm_eps": 1e-20, "attention_bias": false, "attention_dropout": 0.0, "initializer_range": 0.02, "use_cache": true,
        "tie_word_embeddings": false, "max_position_embeddings": 1048576, "rope_theta": 10000,
        "rope_scaling": {"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 65536},
        "n_routed_experts": 384, "n_shared_experts": 1, "num_experts_per_tok": 6, "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc", "norm_topk_prob": true, "routed_scaling_factor": 1.5, "sliding_window": 128,
        "compress_ratios": [0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
        "compress_rope_theta": 160000, "kv_source_layer_ids": [2, 8, 14, 20], "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "index_n_heads": 32, "index_head_dim": 128, "index_topk": 512, "candidate_source_layer_id": 20,
        "candidate_topk_blocks": 2048, "candidate_block_size": 8, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-06,
        "engram_layer_ids": [1, 14], "engram_num_embeddings": [384006168, 384016682], "engram_max_ngram_size": 4,
        "engram_vocab_size": 16000000, "engram_n_heads": 8, "engram_head_dim": 256, "engram_pad_token_id": 2,
        "engram_compressed_vocab_size": 99092, "num_nextn_predict_layers": 3, "dspark_block_size": 5,
        "dspark_noise_token_id": 128799, "dspark_target_layer_ids": [37, 38, 39], "dspark_markov_rank": 256,
        "dspark_n_routed_experts": 128, "dspark_num_experts_per_tok": 3
      },
      "vision_config": {"model_type": "deepseek_v41_vision", "num_hidden_layers": 32, "hidden_size": 1024, "num_attention_heads": 16,
        "intermediate_size": 2816, "patch_size": 14, "rope_theta": 10000, "downsample_ratio": 3, "max_image_tokens": 1024,
        "min_pixels": 295936, "max_wh_ratio": null}
    }"#).expect("test config JSON")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv41_config_parses_and_validates_the_real_layout() {
        let c = Dsv41Config::from_value(&test_config_json(), None).unwrap();
        assert!(c.unknown_keys.is_empty(), "{:?}", c.unknown_keys);
        let facts = c.validate().unwrap();
        assert!(facts.len() > 40, "{}", facts.len());
        assert_eq!(c.total_layers(), 43);
        assert_eq!(c.engram_cols(), 24);
        assert_eq!(c.hc_mix(), 24);
        assert_eq!(c.kv_source_of(7), Some(2));
        assert_eq!(c.kv_source_of(19), Some(14));
        assert_eq!(c.kv_source_of(39), Some(20));
        assert_eq!(c.index_source_of(27), Some(24));
        assert_eq!(c.kv_source_of(1), None);
        assert!(c.uses_candidates(24) && !c.uses_candidates(20) && !c.uses_candidates(25));
        assert_eq!(c.attn_class(0), AttnClass::SwaOnly);
        assert_eq!(c.attn_class(41), AttnClass::DsparkStage);
        assert_eq!(c.attn_class(20), AttnClass::Compressed { ratio: 1, kv_source: true, index_source: true, candidate_source: true, uses_candidates: false });
        assert_eq!(c.attn_class(3), AttnClass::Compressed { ratio: 2, kv_source: false, index_source: false, candidate_source: false, uses_candidates: false });
        assert_eq!(c.moe_of(40), (128, 3));
        assert_eq!(c.rms_eps, 1e-20);
        assert_eq!((c.backtrack(), c.swa_slots(), c.frontier_rows(2)), (6, 134, 7));
        assert_eq!(c.first_decoder_layer(), Some(20));
        assert_eq!(c.layer_role(19), LayerRole::Encoder);
        assert_eq!(c.layer_role(20), LayerRole::Decoder { kv_on_all_rows: true });
        assert_eq!(c.layer_role(39), LayerRole::Decoder { kv_on_all_rows: false });
        assert_eq!(c.layer_role(40), LayerRole::DsparkStage);
    }

    #[test]
    fn dsv41_config_refuses_broken_wiring_and_reports_unknown_keys() {
        let mut v = test_config_json();
        v["text_config"]["compress_ratios"] = serde_json::json!([0, 2]);
        assert!(Dsv41Config::from_value(&v, None).unwrap().validate().unwrap_err().to_string().contains("compress_ratios"));
        let mut v = test_config_json();
        v["text_config"]["engram_num_embeddings"] = serde_json::json!([384006168u64, 384016683u64]);
        assert!(Dsv41Config::from_value(&v, None).unwrap().validate().unwrap_err().to_string().contains("prime buckets"));
        let mut v = test_config_json();
        v["text_config"]["new_field"] = serde_json::json!(1);
        assert_eq!(Dsv41Config::from_value(&v, None).unwrap().unknown_keys, vec!["text_config.new_field".to_string()]);
        let mut v = test_config_json();
        v["text_config"].as_object_mut().unwrap().remove("index_topk");
        assert!(Dsv41Config::from_value(&v, None).unwrap_err().to_string().contains("text_config.index_topk missing"));
    }

    #[test]
    fn dsv41_config_diff_lists_paths() {
        let a = test_config_json();
        let mut b = a.clone();
        b["quantization_config"] = serde_json::json!({"quant_method": "exl3"});
        b["engram_table_dir"] = Value::Null;
        let mut out = Vec::new();
        diff_values("", &a, &b, &mut out);
        assert!(out.iter().any(|l| l.starts_with("engram_table_dir: (absent) | null")), "{out:?}");
        assert!(out.iter().any(|l| l.starts_with("quantization_config.quant_method: \"fp8\" | \"exl3\"")), "{out:?}");
        assert!(out.iter().any(|l| l.starts_with("quantization_config.scale_fmt")), "{out:?}");
        assert!(!out.iter().any(|l| l.starts_with("text_config")), "{out:?}");
    }
}
