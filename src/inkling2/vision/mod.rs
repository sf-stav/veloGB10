//! S5: native image input for Inkling-2 (`model.visual.*`, 422 bf16 tensors, 0.637 GB): preprocessing, the 23-layer ViT,
//! the 2x2 merger and the splice of the image features at the `<|unused_200054|>` placeholder rows of the prompt.
//!
//! Reference (read line by line, not the plan): transformers `models/inkling2/{image_processing,modeling}_inkling2.py`
//! (`Inkling2ImageProcessor`, `Inkling2VisionModel`) + `vision_utils.py` (`get_vision_position_ids`,
//! `get_vision_interpolation_indices_and_weights`) + the `inkling2` conversion-mapping names. Facts the code relies on:
//!   * no resize: the image is padded (bottom / right) to multiples of 40 with -1 on the 0..255 scale, then
//!     `(x - 127.5) / 127.5` (the fused rescale + normalize, fp32); RGBA is composited onto an 8x8 checkerboard 204 / 153
//!     with `(rgb * a + bg * (255 - a) + 127) // 255`.
//!   * one LM token per 40x40 tile; a tile is 2x2 patches of 20x20x3, each flattened (py, px, c) = 1200; the patch
//!     sequence of an image is block-major (tile row, tile col, in-tile row, in-tile col), CLS first.
//!   * patch embed `Linear(1200 -> 1024, no bias)` + the 32x32 learned position table resampled to the image's patch
//!     grid (bilinear, align_corners=False, border clamp); CLS = class_embedding + cls_position_embedding; `ln_pre`.
//!   * axial 2D RoPE over the whole head (64): 16 inverse frequencies `10000^(-2i/32)`, channel pair j < 16 rotates
//!     by (col + 1) * f[j], pair j >= 16 by (row + 1) * f[j - 16] (patches 1-indexed, CLS at (0, 0)); pairs interleaved.
//!   * 23 pre-LN layers: `h += ls_1 * out_proj(attn(ln_1 h))`, `h += ls_2 * c_proj(gelu(c_fc(ln_2 h)))` (biases on all
//!     four attention linears and both MLP linears, exact erf GELU, non-causal attention within one image, scale 1/8).
//!   * merger: drop CLS, 4 consecutive patch rows (one tile) concatenated -> 4096, `proj_in` (4096 -> 4096) -> GELU ->
//!     `proj_out` (4096 -> 2304), no biases. The result REPLACES the post-`embed_norm` embedding of the placeholder row
//!     (`masked_scatter` on `inputs_embeds`): it is not normed.
//!
//! Map: [`preproc`] data URL -> pixels -> tiles (+ the per-image token cap), [`cpu`] the f32 oracle, [`gpu`] the
//! tower on the `gpu_inkling2` module (bands b0..bf), [`probe`] `--probe-inkling2-vision`, [`serve`] the request side
//! (decode, expand, cache) used by the server.

pub mod preproc;
pub mod cpu;
pub mod gpu;
pub mod probe;
pub mod serve;

use anyhow::{ensure, Context, Result};
use std::path::Path;

pub const VP: &str = "model.visual.";
/// The LM image placeholder token.
pub const IMAGE_TOKEN: u32 = 200054;

/// The tower's shape (`config.json` `vision_config`; every field read, not assumed).
#[derive(Clone, Debug)]
pub struct VisionCfg {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub patch: usize,
    pub merge: usize,
    pub grid: usize,
    pub proj_hidden: usize,
    pub text_hidden: usize,
    pub eps: f32,
    pub rope_theta: f32,
}

impl VisionCfg {
    pub fn load(dir: &Path) -> Result<VisionCfg> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).context("config.json")?)?;
        let vc = v.get("vision_config").context("config.json has no vision_config")?;
        let u = |k: &str, d: usize| vc.get(k).and_then(|x| x.as_u64()).map_or(d, |x| x as usize);
        let hidden = u("hidden_size", 1024);
        let heads = u("num_attention_heads", 16);
        let mlp_ratio = vc.get("mlp_ratio").and_then(|x| x.as_f64()).unwrap_or(4.0);
        let c = VisionCfg {
            hidden, layers: u("num_hidden_layers", 23), heads, head_dim: hidden / heads,
            inter: vc.get("intermediate_size").and_then(|x| x.as_u64()).map_or((hidden as f64 * mlp_ratio) as usize, |x| x as usize),
            patch: u("native_patch_size", u("patch_size", 20)), merge: u("adapter_merge_size", u("spatial_merge_size", 2)),
            grid: u("posemb_grid_size", 32), proj_hidden: u("projection_hidden_dim", 4096), text_hidden: u("text_hidden_size", 4608),
            eps: vc.get("layer_norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-5) as f32,
            rope_theta: vc.pointer("/rope_parameters/rope_theta").and_then(|x| x.as_f64()).unwrap_or(10000.0) as f32,
        };
        ensure!(!vc.get("projector_bias").and_then(|x| x.as_bool()).unwrap_or(false), "vision projector_bias=true is not served");
        ensure!(c.head_dim == 64 && c.hidden == 1024 && c.patch == 20 && c.merge == 2,
                "vision tower shape {c:?}: this build serves hidden 1024 / head_dim 64 / patch 20 / merge 2 only");
        ensure!(c.text_hidden == 2304 && c.proj_hidden == 4096, "vision merger {} -> {} (expected 4096 -> 2304)", c.proj_hidden, c.text_hidden);
        Ok(c)
    }
    /// Pixels per side of one LM token tile (40).
    pub fn tile(&self) -> usize { self.patch * self.merge }
    /// Inputs of one patch (1200).
    pub fn patch_in(&self) -> usize { 3 * self.patch * self.patch }
}

/// Host f32 copies of one encoder layer (bf16 values).
pub struct LayerHost {
    pub ln1_w: Vec<f32>, pub ln1_b: Vec<f32>, pub ln2_w: Vec<f32>, pub ln2_b: Vec<f32>,
    pub ls1: Vec<f32>, pub ls2: Vec<f32>,
    pub q_w: Vec<f32>, pub q_b: Vec<f32>, pub k_w: Vec<f32>, pub k_b: Vec<f32>, pub v_w: Vec<f32>, pub v_b: Vec<f32>,
    pub o_w: Vec<f32>, pub o_b: Vec<f32>,
    pub fc1_w: Vec<f32>, pub fc1_b: Vec<f32>, pub fc2_w: Vec<f32>, pub fc2_b: Vec<f32>,
}

/// Every tower tensor, f32 on the host.
pub struct VisionHost {
    pub cfg: VisionCfg,
    pub patch_w: Vec<f32>,
    pub pos: Vec<f32>,
    pub cls: Vec<f32>,
    pub ln_pre_w: Vec<f32>,
    pub ln_pre_b: Vec<f32>,
    pub layers: Vec<LayerHost>,
    pub proj_in: Vec<f32>,
    pub proj_out: Vec<f32>,
}

impl VisionHost {
    pub fn load(dir: &Path) -> Result<VisionHost> {
        let cfg = VisionCfg::load(dir)?;
        let st = crate::fp4l::pack::StDir::open(dir)?;
        let g = |n: &str, want: usize| -> Result<Vec<f32>> {
            let name = format!("{VP}{n}");
            let v = st.plain_f32(&name)?;
            ensure!(v.len() == want, "{name}: {} values, expected {want}", v.len());
            Ok(v)
        };
        let (d, i) = (cfg.hidden, cfg.inter);
        let b = "backbone.";
        let ce = g(&format!("{b}embeddings.class_embedding"), d)?;
        let cp = g(&format!("{b}embeddings.cls_position_embedding"), d)?;
        // the reference adds the two in the parameter dtype (bf16 + bf16 in a bf16 load, f32 in an f32 load): f32 here
        let cls: Vec<f32> = ce.iter().zip(&cp).map(|(a, b)| a + b).collect();
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = format!("{b}layers.{l}.");
            layers.push(LayerHost {
                ln1_w: g(&format!("{p}ln_1.weight"), d)?, ln1_b: g(&format!("{p}ln_1.bias"), d)?,
                ln2_w: g(&format!("{p}ln_2.weight"), d)?, ln2_b: g(&format!("{p}ln_2.bias"), d)?,
                ls1: g(&format!("{p}ls_1.gamma"), d)?, ls2: g(&format!("{p}ls_2.gamma"), d)?,
                q_w: g(&format!("{p}self_attn.q_proj.weight"), d * d)?, q_b: g(&format!("{p}self_attn.q_proj.bias"), d)?,
                k_w: g(&format!("{p}self_attn.k_proj.weight"), d * d)?, k_b: g(&format!("{p}self_attn.k_proj.bias"), d)?,
                v_w: g(&format!("{p}self_attn.v_proj.weight"), d * d)?, v_b: g(&format!("{p}self_attn.v_proj.bias"), d)?,
                o_w: g(&format!("{p}self_attn.out_proj.weight"), d * d)?, o_b: g(&format!("{p}self_attn.out_proj.bias"), d)?,
                fc1_w: g(&format!("{p}mlp.c_fc.weight"), i * d)?, fc1_b: g(&format!("{p}mlp.c_fc.bias"), i)?,
                fc2_w: g(&format!("{p}mlp.c_proj.weight"), d * i)?, fc2_b: g(&format!("{p}mlp.c_proj.bias"), d)?,
            });
        }
        Ok(VisionHost {
            patch_w: g(&format!("{b}embeddings.patch_embedding.weight"), d * cfg.patch_in())?,
            pos: g(&format!("{b}embeddings.position_embedding"), cfg.grid * cfg.grid * d)?,
            cls,
            ln_pre_w: g(&format!("{b}ln_pre.weight"), d)?, ln_pre_b: g(&format!("{b}ln_pre.bias"), d)?,
            layers,
            proj_in: g("proj_in.weight", cfg.proj_hidden * cfg.merge * cfg.merge * d)?,
            proj_out: g("proj_out.weight", cfg.text_hidden * cfg.proj_hidden)?,
            cfg,
        })
    }
}

/// Patch-sequence geometry of one image (`rows x cols` LM tokens).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid { pub rows: usize, pub cols: usize }
impl Grid {
    pub fn tokens(&self) -> usize { self.rows * self.cols }
    /// patches (no CLS)
    pub fn patches(&self) -> usize { 4 * self.tokens() }
    /// sequence rows of the tower (CLS + patches)
    pub fn seq(&self) -> usize { 1 + self.patches() }
    /// (row, col) on the patch grid of patch `p` (block-major order, merge 2)
    pub fn patch_rc(&self, p: usize) -> (usize, usize) {
        let t = p / 4;
        let (tr, tc) = (t / self.cols, t % self.cols);
        let (iy, ix) = ((p % 4) / 2, p % 2);
        (2 * tr + iy, 2 * tc + ix)
    }
}

/// Bilinear taps (align_corners=False, border clamp) of one axis: index `i` of `size` into a `side` table, f32 as torch.
pub fn interp_axis(i: usize, size: usize, side: usize) -> ([usize; 2], [f32; 2]) {
    let src = ((i as f32 + 0.5) * side as f32) / size as f32 - 0.5;
    let fl = src.floor();
    let mut t = [0usize; 2];
    let mut w = [0f32; 2];
    for o in 0..2 {
        let raw = fl as i64 + o as i64;
        t[o] = raw.clamp(0, side as i64 - 1) as usize;
        w[o] = (1.0 - (src - fl - o as f32).abs()).max(0.0);
    }
    (t, w)
}

/// The 16 RoPE inverse frequencies (f32 as torch: `1 / base^(arange(0, 32, 2) / 32)`).
pub fn rope_inv_freq(cfg: &VisionCfg) -> Vec<f32> {
    let sd = cfg.head_dim / 2;
    (0..sd / 2).map(|i| 1.0f32 / cfg.rope_theta.powf((2 * i) as f32 / sd as f32)).collect()
}
