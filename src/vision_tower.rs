//! Visual tower of the Qwen3.5-27B vision model: weight structures + strict loader.
//!
//! The 333 `model.visual.*` tensors (all BF16) form: patch_embed (Conv3d), pos_embed (learned
//! bilinear table), 27 ViT blocks, and the merger. Shapes are pinned in PLAN/W2_PREPROC_SPEC.md;
//! the weights are proven correct by the V2 cross-chain per-block rel-L2 oracle. This module loads
//! them strictly: a missing or unexpected `model.visual.*` tensor is an ERROR, never a warning.

use anyhow::{anyhow, Result};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

pub const PATCH: usize = 16;
pub const TEMPORAL: usize = 2;
pub const MERGE: usize = 2;
pub const IN_CH: usize = 3;
pub const HIDDEN: usize = 1152;
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = HIDDEN / HEADS; // 72
pub const INTER: usize = 4304;
pub const NUM_BLOCKS: usize = 27;
pub const NUM_POS: usize = 2304;
/// Merger output width of the historical Qwen3.5-27B tower (= text hidden 5120). Other models,
/// including qwen4_exp, use the geometry-driven `TowerDims::out_hidden` value.
pub const OUT_HIDDEN: usize = 5120;
pub const MERGE_INTER: usize = HIDDEN * MERGE * MERGE; // 4608

/// Vision-tower geometry as advertised by the model's `config.json` `vision_config` block.
///
/// The engine's vision path (strict loader + CPU/GPU tower + encoder + splice) is now GEOMETRY-
/// DRIVEN: `TowerDims` comes from `vision_config` and every shape in the checkpoint is loaded
/// against it, so the whole Qwen3.5/3.8 VL family serves images — 0.8b (768/12), 2b+4b
/// (1024/24), 9b (1152/27→4096 out), 3.6-35b (1152/27→2048 out), 3.5-122b (1152/27→3072 out)
/// and 27B (1152/27→5120 out). The `nvfp4-*` quantized dirs keep the tower's attention+norm
/// weights as raw BF16 and pack only the MLP weights (`weight_packed/_scale/_global_scale`);
/// those are dequantized to f32 at load (engine NVFP4 convention, GPU-kernel semantics).
///
/// Before this (V1.2 strict loader, 2026-08-24) the shapes were hardcoded to the 27B tower and
/// ANY other `model.visual.*` checkpoint PANICKED the server at startup (2026-08-29 report:
/// "shape mismatch model.visual.blocks.0.norm1.weight: got 1024 expect 1152"). Loading is now
/// total: any missing/unexpected/mismatched tensor is an `Err` (never a panic) and the serve
/// path degrades to text-only with a visible notice.
#[derive(Clone, Debug, Default)]
pub struct VisionGeometry {
    pub hidden_size: Option<usize>,
    pub depth: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub num_heads: Option<usize>,
    pub num_position_embeddings: Option<usize>,
    pub patch_size: Option<usize>,
    pub spatial_merge_size: Option<usize>,
    pub temporal_patch_size: Option<usize>,
    pub in_channels: Option<usize>,
    pub out_hidden_size: Option<usize>,
}

impl VisionGeometry {
    /// Every field must be present; the checkpoint is then loaded against these dims.
    pub fn to_dims(&self) -> Result<TowerDims> {
        let need = |v: Option<usize>, name: &str| -> Result<usize> {
            v.ok_or_else(|| anyhow!("vision_config missing {}", name))
        };
        Ok(TowerDims {
            hidden: need(self.hidden_size, "hidden_size")?,
            depth: need(self.depth, "depth")?,
            inter: need(self.intermediate_size, "intermediate_size")?,
            heads: need(self.num_heads, "num_heads")?,
            num_pos: need(self.num_position_embeddings, "num_position_embeddings")?,
            patch: need(self.patch_size, "patch_size")?,
            temporal: need(self.temporal_patch_size, "temporal_patch_size")?,
            merge: need(self.spatial_merge_size, "spatial_merge_size")?,
            in_ch: need(self.in_channels, "in_channels")?,
            out_hidden: need(self.out_hidden_size, "out_hidden_size")?,
        })
    }
}

/// Fully-resolved tower dimensions built from `vision_config` after validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TowerDims {
    pub hidden: usize,
    pub depth: usize,
    pub inter: usize,
    pub heads: usize,
    pub num_pos: usize,
    pub patch: usize,
    pub temporal: usize,
    pub merge: usize,
    pub in_ch: usize,
    pub out_hidden: usize,
}

impl TowerDims {
    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }
    /// Merger hidden width: hidden * merge^2 (4608 for the 27B tower).
    pub fn merge_inter(&self) -> usize {
        self.merge * self.merge * self.hidden
    }
    /// Elements per patch row: in_ch * temporal * patch^2.
    pub fn wpv(&self) -> usize {
        self.in_ch * self.temporal * self.patch * self.patch
    }
    /// Bilinear pos-embed table side: sqrt(num_position_embeddings).
    pub fn num_side(&self) -> usize {
        (self.num_pos as f64).sqrt() as usize
    }
}

impl std::fmt::Display for VisionGeometry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn o(v: Option<usize>) -> String {
            v.map(|x| x.to_string()).unwrap_or_else(|| "?".into())
        }
        write!(
            f,
            "hidden={} depth={} inter={} heads={} pos={} patch={} merge={} temporal={} in_ch={} out={}",
            o(self.hidden_size), o(self.depth), o(self.intermediate_size), o(self.num_heads),
            o(self.num_position_embeddings), o(self.patch_size), o(self.spatial_merge_size),
            o(self.temporal_patch_size), o(self.in_channels), o(self.out_hidden_size),
        )
    }
}

/// Probe the model dir's `config.json` for a declared vision tower.
/// `Ok(Some(g))` = the config declares `vision_config`; `Ok(None)` = no vision tower (text-only).
/// The serve path then calls [`VisualTower::load`] for `Some` models — success means the full
/// vision path (CPU + GPU + splice) is enabled for that geometry.
pub fn vision_geometry(model_dir: &str) -> Result<Option<VisionGeometry>> {
    let raw = std::fs::read_to_string(Path::new(model_dir).join("config.json"))?;
    geometry_from_config_json(&raw)
}

/// Parse `config.json`'s `vision_config` block (pure; unit-tested). Missing block → `Ok(None)`.
fn geometry_from_config_json(raw: &str) -> Result<Option<VisionGeometry>> {
    let v: serde_json::Value =
        serde_json::from_str(raw).map_err(|e| anyhow!("config.json: {e}"))?;
    let vc = match v.get("vision_config") {
        Some(x) => x,
        None => return Ok(None),
    };
    if !vc.is_object() {
        return Err(anyhow!("vision_config is not an object"));
    }
    let g = |k: &str| vc.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
    Ok(Some(VisionGeometry {
        hidden_size: g("hidden_size"),
        depth: g("depth"),
        intermediate_size: g("intermediate_size"),
        num_heads: g("num_heads"),
        num_position_embeddings: g("num_position_embeddings"),
        patch_size: g("patch_size"),
        spatial_merge_size: g("spatial_merge_size"),
        temporal_patch_size: g("temporal_patch_size"),
        in_channels: g("in_channels"),
        out_hidden_size: g("out_hidden_size"),
    }))
}

/// Per-ViT-block weights (12 tensors). Stored as f32 (weights are BF16 on disk).
#[derive(Clone, Debug)]
pub struct VisualBlock {
    pub norm1_w: Vec<f32>,
    pub norm1_b: Vec<f32>,
    pub norm2_w: Vec<f32>,
    pub norm2_b: Vec<f32>,
    pub qkv_w: Vec<f32>,   // [3*HIDDEN, HIDDEN] = [3456, 1152]
    pub qkv_b: Vec<f32>,   // [3*HIDDEN]
    pub proj_w: Vec<f32>,  // [HIDDEN, HIDDEN]
    pub proj_b: Vec<f32>,  // [HIDDEN]
    pub fc1_w: Vec<f32>,   // [INTER, HIDDEN] = [4304, 1152]
    pub fc1_b: Vec<f32>,   // [INTER]
    pub fc2_w: Vec<f32>,   // [HIDDEN, INTER]
    pub fc2_b: Vec<f32>,   // [HIDDEN]
}

/// The complete vision tower weights, plus the resolved geometry that produced them.
#[derive(Clone, Debug)]
pub struct VisualTower {
    /// Resolved per-model dimensions (from `vision_config`).
    pub dims: TowerDims,
    /// The model's own image preprocessing settings (from `preprocessor_config.json`).
    pub preproc: crate::vision_preproc::VisionPreprocConfig,
    /// The `<|image_pad|>` token id (from `config.json` `image_token_id`).
    pub image_pad: u32,
    pub patch_embed_w: Vec<f32>, // [hidden, in_ch, temporal, patch, patch]
    pub patch_embed_b: Vec<f32>, // [hidden]
    pub pos_embed_w: Vec<f32>,   // [num_pos, hidden]
    pub blocks: Vec<VisualBlock>,
    pub merger_norm_w: Vec<f32>, // [hidden]
    pub merger_norm_b: Vec<f32>, // [hidden]
    pub merger_fc1_w: Vec<f32>,  // [merge_inter, merge_inter]
    pub merger_fc1_b: Vec<f32>,  // [merge_inter]
    pub merger_fc2_w: Vec<f32>,  // [out_hidden, merge_inter]
    pub merger_fc2_b: Vec<f32>,  // [out_hidden]
}

struct Map<'a> {
    m: HashMap<String, (&'a str, &'a [u8])>,
}

impl<'a> Map<'a> {
    fn build(all_raw: &'a [Vec<u8>]) -> Result<Self> {
        let mut m = HashMap::new();
        for raw in all_raw {
            let st = SafeTensors::deserialize(raw)?;
            use safetensors::Dtype;
            for (name, view) in st.tensors() {
                let dt = match view.dtype() {
                    Dtype::BF16 => "BF16",
                    Dtype::F16 => "F16",
                    Dtype::F32 => "F32",
                    _ => "OTHER",
                };
                m.insert(name.to_string(), (dt, view.data()));
            }
        }
        Ok(Map { m })
    }

    fn get(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        let (dt, data) = self
            .m
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor: {}", name))?;
        let v = match *dt {
            "BF16" | "F16" => {
                let m = data.len() / 2;
                let mut out = Vec::with_capacity(m);
                for i in 0..m {
                    let b = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                    out.push(f32::from_bits((b as u32) << 16));
                }
                out
            }
            "F32" => {
                let m = data.len() / 4;
                let mut out = Vec::with_capacity(m);
                for i in 0..m {
                    out.push(f32::from_le_bytes([
                        data[i * 4],
                        data[i * 4 + 1],
                        data[i * 4 + 2],
                        data[i * 4 + 3],
                    ]));
                }
                out
            }
            other => return Err(anyhow!("unsupported dtype {} for {}", other, name)),
        };
        if v.len() != n {
            return Err(anyhow!("shape mismatch {}: got {} expect {}", name, v.len(), n));
        }
        Ok(v)
    }

    /// Load one row-major weight `[m, k]`, accepting the engine's NVFP4 pack convention:
    /// `{stem}.weight` (raw BF16/F16/F32) OR `{stem}.weight_packed` + `{stem}.weight_scale`
    /// + `{stem}.weight_global_scale` (the `--quantize` layout; dequantizes to f32 with the
    /// GPU kernels' exact semantics — no bf16 intermediate).
    fn get_linear(&self, stem: &str, m: usize, k: usize) -> Result<Vec<f32>> {
        let raw = format!("{stem}.weight");
        if self.m.contains_key(&raw) {
            return self.get(&raw, m * k);
        }
        let pname = format!("{stem}.weight_packed");
        let (_, pdata) = self.m.get(&pname).ok_or_else(|| {
            anyhow!("missing tensor: {} (or {})", raw, pname)
        })?;
        let (_, sdata) = self.m.get(&format!("{stem}.weight_scale"))
            .ok_or_else(|| anyhow!("missing tensor: {}.weight_scale", stem))?;
        let (_, gdata) = self.m.get(&format!("{stem}.weight_global_scale"))
            .ok_or_else(|| anyhow!("missing tensor: {}.weight_global_scale", stem))?;
        let gs = f32::from_le_bytes(gdata[..4].try_into().unwrap());
        let q = crate::quant::Nvfp4Tensor {
            qweight: pdata.to_vec(),
            scales: sdata.to_vec(),
            global_scale: gs,
            m,
            k,
        };
        let v = crate::quant::dequantize_nvfp4_f32(&q);
        debug_assert_eq!(v.len(), m * k, "packed dequant size {} vs {}x{}", v.len(), m, k);
        Ok(v)
    }
}

impl VisualTower {
    /// Strict-load every `model.visual.*` tensor, geometry-driven.
    ///
    /// Dimensions come from `config.json`'s `vision_config` (`TowerDims`); the text model's
    /// `hidden_size` must equal the tower out width (the splice writes image embeddings into the
    /// embedding rows), and MLP weights may be raw BF16 or the engine NVFP4 packed form
    /// (`*_weight_packed/_scale/_global_scale`, dequantized to f32 with GPU-kernel semantics).
    /// Errors (never panics) on missing/unexpected/mismatched tensors — the serve path then
    /// degrades to text-only. The 27B raw tower is exactly the V1.2 contract (333 tensors).
    pub fn load(model_dir: &str) -> Result<Self> {
        let dir = Path::new(model_dir);
        if !dir.is_dir() {
            return Err(anyhow!("not a directory: {}", model_dir));
        }
        // --- geometry, text width and image-pad token from config.json ---
        let raw_cfg = std::fs::read_to_string(dir.join("config.json"))?;
        let geo = geometry_from_config_json(&raw_cfg)?
            .ok_or_else(|| anyhow!("no vision_config in {}", model_dir))?;
        let dims = geo.to_dims()?;
        let cfg_v: serde_json::Value = serde_json::from_str(&raw_cfg)?;
        let text_hidden = cfg_v.get("text_config")
            .and_then(|t| t.get("hidden_size"))
            .or_else(|| cfg_v.get("hidden_size"))
            .and_then(|x| x.as_u64()).map(|x| x as usize);
        if let Some(th) = text_hidden {
            if th != dims.out_hidden {
                return Err(anyhow!(
                    "vision out_hidden {} != text hidden {} — unsupported model layout",
                    dims.out_hidden, th,
                ));
            }
        }
        let image_pad = cfg_v.get("image_token_id")
            .and_then(|x| x.as_u64()).map(|x| x as u32)
            .unwrap_or(248056); // the Qwen3.5/3.8 fleet constant (validated across all dirs)
        let preproc = load_preproc(dir, &dims)?;
        // gather safetensors shards
        let mut shards: Vec<String> = vec![];
        let index = dir.join("model.safetensors.index.json");
        if index.exists() {
            let raw = std::fs::read_to_string(&index)?;
            let j: serde_json::Value = serde_json::from_str(&raw)?;
            if let Some(wm) = j["weight_map"].as_object() {
                // Only the shards that hold `model.visual.*`. Reading every shard into host memory
                // to find them is what pushed the box into the kernel's OOM path on a 97 GB
                // artifact (2026-08-28): the tower is ~1.3 GB and lives in one 4 GB shard.
                let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
                for (k, v) in wm {
                    if !k.starts_with("model.visual.") { continue; }
                    if let Some(s) = v.as_str() {
                        set.insert(s.to_string());
                    }
                }
                for s in set {
                    shards.push(dir.join(s).to_string_lossy().to_string());
                }
            }
        } else {
            for entry in std::fs::read_dir(dir)? {
                let e = entry?;
                let nm = e.file_name().to_string_lossy().to_string();
                if nm.ends_with(".safetensors") {
                    shards.push(e.path().to_string_lossy().to_string());
                }
            }
            shards.sort();
        }
        if shards.is_empty() {
            return Err(anyhow!("no safetensors found in {}", model_dir));
        }
        let mut all_raw: Vec<Vec<u8>> = Vec::new();
        for s in &shards {
            all_raw.push(std::fs::read(s)?);
        }
        let map = Map::build(&all_raw)?;

        // Load all visual tensors strictly, geometry-driven.
        let mut block_names: Vec<String> = Vec::new();
        let hidden = dims.hidden;
        let mut blocks = Vec::with_capacity(dims.depth);
        for i in 0..dims.depth {
            let bp = format!("blocks.{i}.");
            let p = format!("model.visual.{bp}");
            let b = VisualBlock {
                norm1_w: map.get(&format!("{p}norm1.weight"), hidden)?,
                norm1_b: map.get(&format!("{p}norm1.bias"), hidden)?,
                norm2_w: map.get(&format!("{p}norm2.weight"), hidden)?,
                norm2_b: map.get(&format!("{p}norm2.bias"), hidden)?,
                qkv_w: map.get(&format!("{p}attn.qkv.weight"), 3 * hidden * hidden)?,
                qkv_b: map.get(&format!("{p}attn.qkv.bias"), 3 * hidden)?,
                proj_w: map.get(&format!("{p}attn.proj.weight"), hidden * hidden)?,
                proj_b: map.get(&format!("{p}attn.proj.bias"), hidden)?,
                fc1_w: map.get_linear(&format!("{p}mlp.linear_fc1"), dims.inter, hidden)?,
                fc1_b: map.get(&format!("{p}mlp.linear_fc1.bias"), dims.inter)?,
                fc2_w: map.get_linear(&format!("{p}mlp.linear_fc2"), hidden, dims.inter)?,
                fc2_b: map.get(&format!("{p}mlp.linear_fc2.bias"), hidden)?,
            };
            block_names.extend(block_names_consumed(&bp));
            blocks.push(b);
        }
        // aggregate the consumed names for the strict "unexpected" check (incl. NVFP4 variants)
        let mut consumed: std::collections::HashSet<String> = std::collections::HashSet::new();
        consumed.insert("model.visual.patch_embed.proj.weight".into());
        consumed.insert("model.visual.patch_embed.proj.bias".into());
        consumed.insert("model.visual.pos_embed.weight".into());
        consumed.insert("model.visual.merger.norm.weight".into());
        consumed.insert("model.visual.merger.norm.bias".into());
        consumed.insert("model.visual.merger.linear_fc1.weight".into());
        consumed.insert("model.visual.merger.linear_fc1.weight_packed".into());
        consumed.insert("model.visual.merger.linear_fc1.weight_scale".into());
        consumed.insert("model.visual.merger.linear_fc1.weight_global_scale".into());
        consumed.insert("model.visual.merger.linear_fc1.bias".into());
        consumed.insert("model.visual.merger.linear_fc2.weight".into());
        consumed.insert("model.visual.merger.linear_fc2.weight_packed".into());
        consumed.insert("model.visual.merger.linear_fc2.weight_scale".into());
        consumed.insert("model.visual.merger.linear_fc2.weight_global_scale".into());
        consumed.insert("model.visual.merger.linear_fc2.bias".into());
        for b in &block_names {
            consumed.insert(b.clone());
        }

        let mi = dims.merge_inter();
        let tower = VisualTower {
            dims,
            preproc,
            image_pad,
            patch_embed_w: map.get("model.visual.patch_embed.proj.weight", hidden * dims.wpv())?,
            patch_embed_b: map.get("model.visual.patch_embed.proj.bias", hidden)?,
            pos_embed_w: map.get("model.visual.pos_embed.weight", dims.num_pos * hidden)?,
            blocks,
            merger_norm_w: map.get("model.visual.merger.norm.weight", hidden)?,
            merger_norm_b: map.get("model.visual.merger.norm.bias", hidden)?,
            merger_fc1_w: map.get_linear("model.visual.merger.linear_fc1", mi, mi)?,
            merger_fc1_b: map.get("model.visual.merger.linear_fc1.bias", mi)?,
            merger_fc2_w: map.get_linear("model.visual.merger.linear_fc2", dims.out_hidden, mi)?,
            merger_fc2_b: map.get("model.visual.merger.linear_fc2.bias", dims.out_hidden)?,
        };

        // Strict "unexpected tensor" check: every model.visual.* key must have been consumed.
        for name in map.m.keys() {
            if name.starts_with("model.visual.") && !consumed.contains(name.as_str()) {
                return Err(anyhow!("unexpected visual tensor not consumed: {}", name));
            }
        }
        Ok(tower)
    }

    /// Number of visual tensors consumed: 3 (patch w/b + pos) + depth*12 + 6 (merger).
    pub fn tensor_count(&self) -> usize {
        3 + self.dims.depth * 12 + 6
    }
}

fn block_names_12(bp: &str) -> Vec<String> {
    [
        "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
        "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias",
        "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
        "mlp.linear_fc2.weight", "mlp.linear_fc2.bias",
    ]
    .iter()
    .map(|s| format!("model.visual.{bp}{s}"))
    .collect()
}

/// The 12 standard block names plus the NVFP4 pack variants of the two MLP weights (the
/// `nvfp4-*` quant dirs replace `*.weight` with `*_weight_packed/_scale/_global_scale`).
fn block_names_consumed(bp: &str) -> Vec<String> {
    let mut v = block_names_12(bp);
    for stem in [
        format!("model.visual.{bp}mlp.linear_fc1"),
        format!("model.visual.{bp}mlp.linear_fc2"),
    ] {
        v.push(format!("{stem}.weight_packed"));
        v.push(format!("{stem}.weight_scale"));
        v.push(format!("{stem}.weight_global_scale"));
    }
    v
}

/// The model's own image preprocessing settings from `preprocessor_config.json`; fields the
/// file does not specify fall back to the 27B defaults. Values that contradict the
/// `vision_config` geometry are a hard error (the checkpoint layout would not match).
fn load_preproc(dir: &Path, dims: &TowerDims) -> Result<crate::vision_preproc::VisionPreprocConfig> {
    let mut cfg = crate::vision_preproc::QWEN27B_PREPROC;
    let p = dir.join("preprocessor_config.json");
    if p.exists() {
        let raw = std::fs::read_to_string(&p)?;
        let v: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| anyhow!("preprocessor_config.json: {e}"))?;
        let get = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
        if let Some(x) = get("patch_size") { cfg.patch_size = x; }
        if let Some(x) = get("merge_size") { cfg.merge_size = x; }
        if let Some(x) = get("temporal_patch_size") { cfg.temporal_patch_size = x; }
        if let Some(x) = get("in_channels") { cfg.in_channels = x; }
        if let Some(sz) = v.get("size").and_then(|o| o.as_object()) {
            if let Some(x) = sz.get("shortest_edge").and_then(|x| x.as_u64()) { cfg.min_pixels = x as usize; }
            if let Some(x) = sz.get("longest_edge").and_then(|x| x.as_u64()) { cfg.max_pixels = x as usize; }
        }
    }
    if cfg.patch_size != dims.patch || cfg.merge_size != dims.merge
        || cfg.temporal_patch_size != dims.temporal || cfg.in_channels != dims.in_ch
    {
        return Err(anyhow!(
            "preprocessor_config (patch={} merge={} temporal={} in_ch={}) contradicts vision_config \
             (patch={} merge={} temporal={} in_ch={})",
            cfg.patch_size, cfg.merge_size, cfg.temporal_patch_size, cfg.in_channels,
            dims.patch, dims.merge, dims.temporal, dims.in_ch,
        ));
    }
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_load_333() {
        let dir = std::env::var("GB10_TEST_MODEL_DIR")
            .unwrap_or_else(|_| "models/3.8-27b-nvfp4-full-all".to_string());
        let t = VisualTower::load(&dir).unwrap();
        assert_eq!(t.tensor_count(), 333, "visual tensor count");
        assert_eq!(t.blocks.len(), 27);
        assert_eq!(t.patch_embed_w.len(), 1152 * 3 * 2 * 16 * 16);
        assert_eq!(t.pos_embed_w.len(), 2304 * 1152);
        assert_eq!(t.merger_fc2_w.len(), t.dims.out_hidden * t.dims.merge_inter());
    }

    /// The geometry probe: every Qwen3.5/3.8 VL geometry parses; `to_dims` requires all fields.
    /// (Before generalization, `is_supported_by_build` limited vision to the 27B tower — the
    /// 0.8b/2b/4b/9b/35b towers are now loaded and served, not excluded.)
    #[test]
    fn geometry_gate() {
        let s = r#"{"vision_config":{"hidden_size":1024,"depth":24,"hidden_act":"gelu_pytorch_tanh","intermediate_size":4096,"num_heads":16,"num_position_embeddings":2304,"patch_size":16,"spatial_merge_size":2,"temporal_patch_size":2,"in_channels":3,"out_hidden_size":2560}}"#;
        let g = geometry_from_config_json(s).unwrap().unwrap();
        let d = g.to_dims().unwrap();
        assert_eq!((d.hidden, d.depth, d.inter, d.heads, d.out_hidden), (1024, 24, 4096, 16, 2560));
        assert_eq!(d.head_dim(), 64);
        assert_eq!(d.merge_inter(), 4096);
        assert_eq!(d.num_side(), 48);
        let s = r#"{"vision_config":{"hidden_size":1152,"depth":27,"intermediate_size":4304,"num_heads":16,"num_position_embeddings":2304,"patch_size":16,"spatial_merge_size":2,"temporal_patch_size":2,"in_channels":3,"out_hidden_size":5120}}"#;
        let d = geometry_from_config_json(s).unwrap().unwrap().to_dims().unwrap();
        assert_eq!(d.head_dim(), 72);
        assert!(geometry_from_config_json(r#"{"hidden_size":2048}"#).unwrap().is_none(),
            "no vision_config → text-only");
        assert!(geometry_from_config_json(r#"{"vision_config":{"hidden_size":1152}}"#)
            .unwrap().unwrap().to_dims().is_err(),
            "partial geometry is rejected");
    }
}
