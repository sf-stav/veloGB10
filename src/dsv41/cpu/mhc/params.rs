//! DSV4.1 OM oracle: the mHC / norm / router parameters of any pack (native, any EXL3 pack, a
//! slice), found through the census (class + layer + role, never per-pack names) and widened to f32.
//! BF16 (native), F16 (the router gates of some EXL3 packs) and F32 all convert exactly.

use super::router::{GateParams, ScoreFunc};
use super::{BlockParams, HcMixParams, HcShape};
use crate::dsv41::census::classify::{Role, TensorClass};
use crate::dsv41::census::{Census, Entry, LinearFormat};
use anyhow::{bail, ensure, Context, Result};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// A pack opened for parameter reads.
pub struct PackParams {
    pub census: Census,
    pub shape: HcShape,
}

/// The value bytes of a tensor as f32.
pub fn to_f32(dtype: &str, b: &[u8]) -> Result<Vec<f32>> {
    Ok(match dtype {
        "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "BF16" => b.chunks_exact(2).map(|c| super::num::bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        "F16" => b.chunks_exact(2).map(|c| super::num::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        d => bail!("parameter dtype {d} is not a plain float (BF16 / F16 / F32)"),
    })
}

impl PackParams {
    /// Census the pack (headers only, about a second on a full pack).
    pub fn open(dir: &Path) -> Result<PackParams> {
        let census = Census::build(dir).with_context(|| format!("census of {}", dir.display()))?;
        let shape = HcShape::of(&census.cfg);
        Ok(PackParams { census, shape })
    }

    /// The one tensor of (class, layer, role); `None` when the pack has none (e.g. no `bias_vl`).
    pub fn find(&self, class: TensorClass, layer: Option<usize>, role: Role) -> Option<&Entry> {
        let l = layer.map(|v| v as u16);
        self.census.entries.iter().find(|e| e.class == class && e.layer == l && e.role == role && e.sub.is_none())
    }

    fn need(&self, class: TensorClass, layer: Option<usize>, role: Role) -> Result<&Entry> {
        self.find(class, layer, role).with_context(|| format!("{}: no {} {:?} tensor for layer {layer:?}", self.census.dir.display(), class.name(), role))
    }

    /// The tensor's values as f32 (plain dtypes only).
    pub fn read(&self, e: &Entry) -> Result<Vec<f32>> {
        let path = self.census.dir.join(&self.census.files[e.file as usize]);
        let mut f = std::fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
        f.seek(SeekFrom::Start(e.offset))?;
        let mut b = vec![0u8; e.bytes as usize];
        f.read_exact(&mut b).with_context(|| format!("read {} from {}", e.name, path.display()))?;
        to_f32(&e.dtype, &b).with_context(|| e.name.clone())
    }

    /// A row range `[r0, r1)` of a 2-D plain tensor (the head's vocabulary rows, without reading 1.3 GB).
    pub fn read_rows(&self, e: &Entry, r0: usize, r1: usize) -> Result<Vec<f32>> {
        ensure!(e.shape.len() == 2 && r1 as u64 <= e.shape[0] && r0 <= r1, "{}: rows {r0}..{r1} of {:?}", e.name, e.shape);
        let es = crate::dsv41::st::dtype_size(&e.dtype).context("dtype")?;
        let row = e.shape[1] * es;
        let path = self.census.dir.join(&self.census.files[e.file as usize]);
        let mut f = std::fs::File::open(&path)?;
        f.seek(SeekFrom::Start(e.offset + r0 as u64 * row))?;
        let mut b = vec![0u8; ((r1 - r0) as u64 * row) as usize];
        f.read_exact(&mut b)?;
        to_f32(&e.dtype, &b)
    }

    fn mix(&self, layer: usize, attn: bool) -> Result<HcMixParams> {
        let (cf, cs, cb) = if attn {
            (TensorClass::HcAttnFn, TensorClass::HcAttnScale, TensorClass::HcAttnBase)
        } else {
            (TensorClass::HcFfnFn, TensorClass::HcFfnScale, TensorClass::HcFfnBase)
        };
        let s = self.shape;
        let fn_ = self.read(self.need(cf, Some(layer), Role::Param)?)?;
        let sc = self.read(self.need(cs, Some(layer), Role::Param)?)?;
        let base = self.read(self.need(cb, Some(layer), Role::Param)?)?;
        ensure!(fn_.len() == s.mix() * s.width() && sc.len() == 3 && base.len() == s.mix(), "layer {layer}: hc mix parameter sizes");
        Ok(HcMixParams { fn_, scale: [sc[0], sc[1], sc[2]], base })
    }

    /// A block's mHC mixers and its two norms (backbone layers and DSpark stages alike).
    pub fn block(&self, layer: usize) -> Result<BlockParams> {
        let attn_norm = self.read(self.need(TensorClass::AttnNorm, Some(layer), Role::Weight)?)?;
        let ffn_norm = self.read(self.need(TensorClass::FfnNorm, Some(layer), Role::Weight)?)?;
        ensure!(attn_norm.len() == self.shape.dim && ffn_norm.len() == self.shape.dim, "layer {layer}: norm sizes");
        Ok(BlockParams { attn_mix: self.mix(layer, true)?, ffn_mix: self.mix(layer, false)?, attn_norm, ffn_norm })
    }

    /// The router of a layer slot (DSpark stages have their own expert count and top-k).
    pub fn gate(&self, layer: usize) -> Result<GateParams> {
        let cfg = &self.census.cfg;
        let (n_experts, top_k) = cfg.moe_of(layer);
        let we = self.need(TensorClass::GateWeight, Some(layer), Role::Weight)?;
        let m = self.census.module_of(we);
        ensure!(matches!(m.format, LinearFormat::Plain(_)), "layer {layer}: router gate is {} (the oracle reads plain gates only)", m.format.label());
        let weight = self.read(we)?;
        let bias = self.read(self.need(TensorClass::GateBias, Some(layer), Role::Bias)?)?;
        let bias_vl = match self.find(TensorClass::GateBiasVl, Some(layer), Role::Param) {
            Some(e) => Some(self.read(e)?),
            None => None,
        };
        ensure!(weight.len() == n_experts * cfg.hidden && bias.len() == n_experts, "layer {layer}: gate sizes");
        Ok(GateParams {
            weight,
            bias,
            bias_vl,
            n_experts,
            dim: cfg.hidden,
            top_k,
            route_scale: cfg.routed_scale as f32,
            score: ScoreFunc::parse(&cfg.scoring_func)?,
            norm_topk_prob: cfg.norm_topk_prob,
            gate_temp: 1.0, // ModelArgs.gate_temp default; no config key sets it for V4.1
        })
    }

    /// The dtype the pack stores a layer's router gate in (BF16 native, F16 in some EXL3 packs).
    pub fn gate_dtype(&self, layer: usize) -> Option<String> { self.find(TensorClass::GateWeight, Some(layer), Role::Weight).map(|e| e.dtype.clone()) }

    /// The final norm weight (`norm.weight`).
    pub fn final_norm(&self) -> Result<Vec<f32>> { self.read(self.need(TensorClass::FinalNorm, None, Role::Weight)?) }

    /// The LM head entry if the pack stores it plain (native: BF16; EXL3 packs: a trellis, `None`).
    pub fn plain_head(&self) -> Option<&Entry> {
        let e = self.find(TensorClass::Head, None, Role::Weight)?;
        matches!(self.census.module_of(e).format, LinearFormat::Plain(_)).then_some(e)
    }
}
