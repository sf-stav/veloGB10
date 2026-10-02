//! Visual tower of the Qwen3.5-27B vision model: weight structures + strict loader.
//!
//! The 333 `model.visual.*` tensors (all BF16) form: patch_embed (Conv3d), pos_embed (learned
//! bilinear table), 27 ViT blocks, and the merger. Shapes are pinned in PLAN/W2_PREPROC_SPEC.md;
//! the weights are proven correct by the V2 cross-chain per-block rel-L2 oracle. This module loads
//! them strictly: a missing or unexpected `model.visual.*` tensor is an ERROR, never a warning.

use anyhow::{anyhow, Result};
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
    /// Where the weights came from (boot/probe line): the model's shards, or `VISION_BF16_FILE`.
    pub source: String,
}

/// VIS (owner 2026-10-01): an EXL3 pack may carry the ORIGINAL bf16 tower next to its quantized one in
/// this file (the 333 `model.visual.*` tensors); the loader then takes the tower ONLY from it.
pub const VISION_BF16_FILE: &str = "vision_tower_bf16.safetensors";

struct Map {
    /// name -> (safetensors dtype, raw little-endian bytes, shape)
    m: HashMap<String, (String, Vec<u8>, Vec<usize>)>,
}

impl Map {
    /// B13 fix (S-B14): SLICED read. The old path `std::fs::read` every whole shard into
    /// `all_raw` — 105.7 GB of anonymous host RAM on this artifact to extract a ~0.5 GB
    /// tower — which was the serve-prep memory sink that killed every `--server` boot at
    /// TP=1 and TP=2 (memwatch floor-hit between "binding HTTP" and listen; the 1 Hz
    /// trace showed a monotonic ~1 GB/s drain exactly here). Now: parse each shard's
    /// safetensors HEADER only (8-byte LE length + JSON), then seek+read just the
    /// `model.visual.*` byte ranges. dtype strings match the safetensors spellings the
    /// getters already switch on ("BF16"/"F16"/"F32"; packed tensors are consumed raw).
    fn build_sliced(shards: &[String]) -> Result<Self> {
        use std::io::{Read, Seek, SeekFrom};
        let mut m = HashMap::new();
        for s in shards {
            let mut f = std::fs::File::open(s)
                .map_err(|e| anyhow!("{}: {e}", s))?;
            let mut u64buf = [0u8; 8];
            f.read_exact(&mut u64buf)?;
            let hlen = u64::from_le_bytes(u64buf) as usize;
            let mut hjson = vec![0u8; hlen];
            f.read_exact(&mut hjson)?;
            let hdr: serde_json::Map<String, serde_json::Value> =
                serde_json::from_slice(&hjson)?;
            for (name, meta) in hdr {
                if name == "__metadata__" || !name.starts_with("model.visual.") {
                    continue;
                }
                let dt = meta.get("dtype").and_then(|x| x.as_str())
                    .ok_or_else(|| anyhow!("{}: header has no dtype", name))?.to_string();
                let offs = meta.get("data_offsets").and_then(|x| x.as_array())
                    .ok_or_else(|| anyhow!("{}: header has no data_offsets", name))?;
                let start = offs.first().and_then(|x| x.as_u64())
                    .ok_or_else(|| anyhow!("{}: bad data_offsets start", name))? as usize;
                let end = offs.get(1).and_then(|x| x.as_u64())
                    .ok_or_else(|| anyhow!("{}: bad data_offsets end", name))? as usize;
                let shape: Vec<usize> = meta.get("shape").and_then(|x| x.as_array())
                    .map(|a| a.iter().filter_map(|d| d.as_u64()).map(|d| d as usize).collect())
                    .unwrap_or_default();
                let mut data = vec![0u8; end - start];
                f.seek(SeekFrom::Start((8 + hlen + start) as u64))?;
                f.read_exact(&mut data)?;
                m.insert(name, (dt, data, shape));
            }
        }
        if m.is_empty() {
            return Err(anyhow!("no model.visual.* tensors in any shard"));
        }
        Ok(Map { m })
    }

    fn get(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        let (dt, data, _) = self
            .m
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor: {}", name))?;
        let v = match dt.as_str() {
            "BF16" => {
                let m = data.len() / 2;
                let mut out = Vec::with_capacity(m);
                for i in 0..m {
                    let b = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                    out.push(f32::from_bits((b as u32) << 16));
                }
                out
            }
            // IEEE half. (Until VIS-1 this arm shared the BF16 bit-shift, which is wrong for F16;
            // no earlier tower stored F16. The EXL3 pack stores norms, biases and patch_embed as F16.)
            "F16" => data.chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
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
        if self.m.contains_key(&format!("{stem}.trellis")) {
            return self.get_exl3(stem, m, k);
        }
        let (_, pdata, _) = self.m.get(&pname).ok_or_else(|| {
            anyhow!("missing tensor: {} (or {})", raw, pname)
        })?;
        let (_, sdata, _) = self.m.get(&format!("{stem}.weight_scale"))
            .ok_or_else(|| anyhow!("missing tensor: {}.weight_scale", stem))?;
        let (_, gdata, _) = self.m.get(&format!("{stem}.weight_global_scale"))
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

    /// VIS-1: one EXL3 trellis linear (`{stem}.trellis/.suh/.svh/.mul1`, the Flash-Next pack's
    /// 5-bit tower) reconstructed to a dense row-major `[m, k]` f32 weight (out = m, in = k):
    /// W[in][out] = suh[in] * (H · decode(trellis) · H)[in][out] * svh[out], H the orthonormal
    /// 128-block Hadamard — the reference's `LinearEXL3.get_weight_tensor` (exl3.py:227-237),
    /// kept in f32 here (the reference rounds to fp16 between steps). The decode is the
    /// oracle-proven bit-exact `exl3::decode_trellis_host`.
    ///
    /// The pack pads a width that is not a multiple of 128 (MLP 4304 -> 4352). Padded OUTPUT
    /// columns must carry svh = 0 (so they are exactly 0) and are dropped; padded INPUT rows are
    /// dropped, which is exact only because their producer's padded outputs are exactly 0 — the
    /// caller checks that on the producer (`check_zero_pad`).
    fn get_exl3(&self, stem: &str, m: usize, k: usize) -> Result<Vec<f32>> {
        let fetch = |suf: &str| self.m.get(&format!("{stem}.{suf}"))
            .ok_or_else(|| anyhow!("missing tensor: {stem}.{suf}"));
        let (tdt, tdata, tshape) = fetch("trellis")?;
        if tdt != "I16" || tshape.len() != 3 || tshape[2] % 16 != 0 {
            return Err(anyhow!("{stem}.trellis: expected I16 [K/16, N/16, 16*bits], got {tdt} {tshape:?}"));
        }
        let (kb, nb, bits) = (tshape[0], tshape[1], tshape[2] / 16);
        let (kp, np) = (kb * 16, nb * 16);
        if kp < k || np < m || kp % 128 != 0 || np % 128 != 0 {
            return Err(anyhow!("{stem}: padded shape {kp}x{np} cannot hold in={k} out={m} (or is not 128-aligned)"));
        }
        let (_, mul1, _) = fetch("mul1")?;
        let mul1 = u32::from_le_bytes(mul1.get(..4).ok_or_else(|| anyhow!("{stem}.mul1 empty"))?.try_into().unwrap());
        if mul1 != crate::exl3::MUL1 {
            return Err(anyhow!("{stem}: mul1 {mul1:#x} != the mul1 codebook constant {:#x}", crate::exl3::MUL1));
        }
        let half_vec = |suf: &str, n: usize| -> Result<Vec<f32>> {
            let (dt, data, _) = fetch(suf)?;
            if dt != "F16" || data.len() != 2 * n {
                return Err(anyhow!("{stem}.{suf}: expected F16 [{n}], got {dt} {} bytes", data.len()));
            }
            Ok(data.chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect())
        };
        let (suh, svh) = (half_vec("suh", kp)?, half_vec("svh", np)?);
        if svh[m..].iter().any(|&v| v != 0.0) {
            return Err(anyhow!("{stem}: padded output columns {m}..{np} have non-zero svh — cannot drop them"));
        }
        let trellis: Vec<i16> = tdata.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        let dec = crate::exl3::decode_trellis_host(&trellis, kb, nb, bits);
        let mut w: Vec<f32> = dec.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect(); // [kp][np]
        let s = 1.0f32 / (128f32).sqrt();
        // left: H over each 128-row block of every column, then scale row i by suh[i]
        let mut col = [0f32; 128];
        for blk in 0..kp / 128 {
            for n in 0..np {
                for j in 0..128 { col[j] = w[(blk * 128 + j) * np + n]; }
                had128(&mut col);
                for j in 0..128 { w[(blk * 128 + j) * np + n] = col[j] * s * suh[blk * 128 + j]; }
            }
        }
        // right: H over each 128-column block of every row, then scale column n by svh[n]
        for i in 0..kp {
            let row = &mut w[i * np..(i + 1) * np];
            for blk in 0..np / 128 {
                let r = &mut row[blk * 128..(blk + 1) * 128];
                had128(r);
                for j in 0..128 { r[j] *= s * svh[blk * 128 + j]; }
            }
        }
        // transpose [in][out] -> row-major [out][in], dropping padding
        let mut out = vec![0f32; m * k];
        for o in 0..m {
            for i in 0..k { out[o * k + i] = w[i * np + o]; }
        }
        Ok(out)
    }

    /// The padded tail `[n..]` of a bias must be exactly zero (EXL3 width padding).
    fn check_zero_pad(&self, name: &str, n: usize) -> Result<()> {
        let (dt, data, _) = self.m.get(name).ok_or_else(|| anyhow!("missing tensor: {}", name))?;
        let esz = if dt == "F32" { 4 } else { 2 };
        if data.len() / esz > n && data[n * esz..].iter().any(|&b| b != 0) {
            return Err(anyhow!("{name}: padded tail beyond {n} is not zero"));
        }
        Ok(())
    }

    /// `get` that accepts a zero-padded tensor longer than `n` (EXL3 pads widths to 128) and
    /// returns its first `n` values.
    fn get_padded(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        let len = self.m.get(name).map(|(dt, d, _)| d.len() / if dt == "F32" { 4 } else { 2 }).unwrap_or(n);
        if len > n {
            self.check_zero_pad(name, n)?;
            let mut v = self.get(name, len)?;
            v.truncate(n);
            return Ok(v);
        }
        self.get(name, n)
    }
}

/// In-place unnormalized Walsh-Hadamard butterfly on 128 values (natural / Sylvester order — the
/// same op order as `exl3_bench::had128_ref` and the `exl3_had128` kernel).
fn had128(v: &mut [f32]) {
    let mut len = 1usize;
    while len < 128 {
        for i in 0..128 {
            if i & len == 0 {
                let (a, b) = (v[i], v[i + len]);
                v[i] = a + b;
                v[i + len] = a - b;
            }
        }
        len <<= 1;
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
        let bf16_file = dir.join(VISION_BF16_FILE);
        if bf16_file.is_file() {
            shards.push(bf16_file.to_string_lossy().to_string());
        } else if index.exists() {
            let raw = std::fs::read_to_string(&index)?;
            let j: serde_json::Value = serde_json::from_str(&raw)?;
            if let Some(wm) = j["weight_map"].as_object() {
                let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
                for (_, v) in wm {
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
        // B13 fix: sliced read — headers only, then just the model.visual.* ranges
        // (the old all_raw path pulled every whole shard into host RAM: 105.7 GB).
        let map = Map::build_sliced(&shards)?;

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
                proj_w: map.get_linear(&format!("{p}attn.proj"), hidden, hidden)?,
                proj_b: map.get(&format!("{p}attn.proj.bias"), hidden)?,
                fc1_w: map.get_linear(&format!("{p}mlp.linear_fc1"), dims.inter, hidden)?,
                fc1_b: map.get_padded(&format!("{p}mlp.linear_fc1.bias"), dims.inter)?,
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
        for lin in ["linear_fc1", "linear_fc2"] {
            for suf in crate::exl3::EXL3_SUFFIXES {
                consumed.insert(format!("model.visual.merger.{lin}.{suf}"));
            }
        }
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
            source: if bf16_file.is_file() { format!("{VISION_BF16_FILE} (original bf16 tower)") }
                    else { "the model shards".to_string() },
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
/// `nvfp4-*` quant dirs replace `*.weight` with `*_weight_packed/_scale/_global_scale`) and the
/// EXL3 pack variants (VIS-1: `attn.proj` and both MLP weights as trellis quadruples; the pack
/// also stores quantized `attn.{q,k,v}_proj` copies of the EXACT bf16 fused `attn.qkv.weight`,
/// which the tower uses instead — the copies are accepted and ignored).
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
    for lin in ["attn.proj", "mlp.linear_fc1", "mlp.linear_fc2", "attn.q_proj", "attn.k_proj", "attn.v_proj"] {
        for suf in crate::exl3::EXL3_SUFFIXES {
            v.push(format!("model.visual.{bp}{lin}.{suf}"));
        }
    }
    for lin in ["attn.q_proj", "attn.k_proj", "attn.v_proj"] {
        v.push(format!("model.visual.{bp}{lin}.bias"));
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
    // D-VIS-RESIZE: --image-max-edge N (default 1024; 0 = no cap)
    if let Ok(v) = crate::opts::var(crate::opt!("image-max-edge")) {
        cfg.max_edge = v.trim().parse().map_err(|_| anyhow!("--image-max-edge must be a pixel count (0 = no cap), got {v:?}"))?;
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
        assert_eq!(t.merger_fc2_w.len(), 5120 * 4608);
    }

    /// VIS-1: F16 tensors decode as IEEE half (they used to share the BF16 bit-shift), BF16 unchanged.
    #[test]
    fn f16_and_bf16_decode() {
        let vals = [1.0f32, -2.5, 0.000_123_4, 65504.0];
        let f16: Vec<u8> = vals.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect();
        let bf16: Vec<u8> = vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes()).collect();
        let mut m = HashMap::new();
        m.insert("a".to_string(), ("F16".to_string(), f16, vec![4]));
        m.insert("b".to_string(), ("BF16".to_string(), bf16, vec![4]));
        let map = Map { m };
        let a = map.get("a", 4).unwrap();
        let b = map.get("b", 4).unwrap();
        for (i, v) in vals.iter().enumerate() {
            assert_eq!(a[i], half::f16::from_f32(*v).to_f32());
            assert_eq!(b[i], half::bf16::from_f32(*v).to_f32());
        }
    }

    /// VIS-1: the unnormalized 128-point butterfly is a Hadamard transform: applying it twice gives 128 x.
    #[test]
    fn had128_is_an_involution_up_to_128() {
        let x: Vec<f32> = (0..128).map(|i| ((i * 37 % 11) as f32) - 5.0).collect();
        let mut y = x.clone();
        had128(&mut y);
        had128(&mut y);
        for i in 0..128 {
            assert_eq!(y[i], 128.0 * x[i]);
        }
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
