//! DSV4.1 attention oracle (leg OA): the attention weights of one layer, read from DeepSeek's native
//! checkpoint (or a native slice: any directory with `model.safetensors.index.json` and the native
//! tensor names `layers.{L}.attn.*`). Conversions follow `inference/convert.py`:
//! * FP8 linears (`wq_a`, `wq_b`, `wkv`, `wo_b`, `indexer.wq_b`) stay E4M3 + 32 x 32 E8M0 blocks
//!   ([`Fp8Lin`]);
//! * `wo_a` is dequantised to bf16 at load (convert.py 157-172: `float(w) * scale` over 32 x 32
//!   blocks, `.bfloat16()`; exact) and runs as a bf16 einsum per o_group;
//! * the ratio-2 compressor `wkv` / `wgate` are bf16 in the file and fp32 at run time (exact
//!   widening); the ratio-1 compressor `wkv`, `indexer.wk`, `indexer.weights_proj` stay bf16;
//! * norms bf16, `attn_sink` fp32.

use super::num::{bf16, bf16_val, e4m3_decode, e8m0_decode, DenseLin, Fp8Lin};
use crate::dsv41::config::Dsv41Config;
use crate::dsv41::st::{read_header, read_tensor, StHeader};
use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// A native checkpoint: tensor name -> shard, headers cached.
pub struct Ckpt {
    pub dir: PathBuf,
    map: HashMap<String, String>,
    headers: Mutex<HashMap<String, StHeader>>,
}

impl Ckpt {
    pub fn open(dir: &Path) -> Result<Ckpt> {
        let p = dir.join("model.safetensors.index.json");
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?)
            .with_context(|| format!("parse {}", p.display()))?;
        let wm = v.get("weight_map").and_then(|m| m.as_object()).with_context(|| format!("{}: no weight_map", p.display()))?;
        let map = wm.iter().filter_map(|(k, f)| f.as_str().map(|f| (k.clone(), f.to_string()))).collect();
        Ok(Ckpt { dir: dir.to_path_buf(), map, headers: Mutex::new(HashMap::new()) })
    }

    pub fn has(&self, name: &str) -> bool { self.map.contains_key(name) }

    /// (dtype, shape, raw bytes) of one tensor.
    pub fn raw(&self, name: &str) -> Result<(String, Vec<usize>, Vec<u8>)> {
        let file = self.map.get(name).with_context(|| format!("{}: no tensor {name} in the index", self.dir.display()))?;
        let mut hs = self.headers.lock().map_err(|_| anyhow::anyhow!("header cache poisoned"))?;
        if !hs.contains_key(file) {
            hs.insert(file.clone(), read_header(&self.dir.join(file))?);
        }
        let h = &hs[file];
        let t = h.tensors.iter().find(|t| t.name == name).with_context(|| format!("{file}: no tensor {name}"))?;
        let (dtype, shape) = (t.dtype.clone(), t.shape.iter().map(|&x| x as usize).collect());
        Ok((dtype, shape, read_tensor(h, name)?))
    }

    /// A BF16 / F16 / F32 tensor as f32 values (exact widening).
    pub fn f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let (dt, shape, b) = self.raw(name)?;
        let v = match dt.as_str() {
            "BF16" => b.chunks_exact(2).map(|c| bf16_val(u16::from_le_bytes([c[0], c[1]]))).collect(),
            "F16" => b.chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
            "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            d => bail!("{name}: dtype {d} is not a float tensor"),
        };
        Ok((shape, v))
    }

    /// A 1-D float vector of the expected length.
    pub fn vec(&self, name: &str, len: usize) -> Result<Vec<f32>> {
        let (s, v) = self.f32(name)?;
        ensure!(v.len() == len, "{name}: shape {s:?}, expected {len} elements");
        Ok(v)
    }

    /// A dense [n, k] matrix (bf16 / fp32) as a [`DenseLin`].
    pub fn dense(&self, name: &str, n: usize, k: usize) -> Result<DenseLin> {
        let (s, w) = self.f32(name)?;
        ensure!(s == [n, k], "{name}: shape {s:?}, expected [{n}, {k}]");
        Ok(DenseLin { n, k, w })
    }

    /// An FP8 linear `{prefix}.weight` (F8_E4M3 [n, k]) + `{prefix}.scale` (F8_E8M0 32 x 32 blocks).
    pub fn fp8(&self, prefix: &str, n: usize, k: usize) -> Result<Fp8Lin> {
        let (dw, sw, w) = self.raw(&format!("{prefix}.weight"))?;
        let (ds, ss, s) = self.raw(&format!("{prefix}.scale"))?;
        ensure!(dw == "F8_E4M3" && sw == [n, k], "{prefix}.weight: {dw} {sw:?}, expected F8_E4M3 [{n}, {k}]");
        ensure!(ds == "F8_E8M0" && ss == [n.div_ceil(32), k.div_ceil(32)], "{prefix}.scale: {ds} {ss:?}, expected F8_E8M0 32 x 32 blocks");
        Ok(Fp8Lin::from_raw(n, k, &w, &s))
    }

    /// An FP8 32 x 32 block matrix dequantised to bf16 (convert.py's `wo_a` rule), [n][k].
    pub fn fp8_to_bf16(&self, prefix: &str, n: usize, k: usize) -> Result<Vec<f32>> {
        let (dw, sw, w) = self.raw(&format!("{prefix}.weight"))?;
        let (ds, ss, s) = self.raw(&format!("{prefix}.scale"))?;
        ensure!(dw == "F8_E4M3" && sw == [n, k], "{prefix}.weight: {dw} {sw:?}");
        ensure!(ds == "F8_E8M0" && ss.len() == 2 && n % ss[0] == 0 && k % ss[1] == 0, "{prefix}.scale: {ds} {ss:?}");
        let (bo, bi) = (n / ss[0], k / ss[1]);
        ensure!((bo, bi) == (32, 32) || (bo, bi) == (128, 128), "{prefix}: block {bo} x {bi}");
        let mut out = vec![0f32; n * k];
        for r in 0..n {
            for c in 0..k {
                out[r * k + c] = bf16(e4m3_decode(w[r * k + c]) * e8m0_decode(s[(r / bo) * ss[1] + c / bi]));
            }
        }
        Ok(out)
    }
}

/// The compressor of a KV-source layer.
pub struct CompressorW {
    pub ratio: usize,
    /// [head_dim, dim]: fp32 at ratio > 1 (bf16 values), bf16 at ratio 1
    pub wkv: DenseLin,
    /// ratio > 1 only
    pub wgate: Option<DenseLin>,
    pub norm: Vec<f32>,
}

/// The indexer of an index-source layer.
pub struct IndexerW {
    /// q_lora -> index_heads x index_head_dim (FP8)
    pub wq_b: Fp8Lin,
    /// dim -> index_heads (bf16)
    pub weights_proj: DenseLin,
    /// owners (the KV sources) only: head_dim -> index_head_dim (bf16) and its RMSNorm
    pub wk: Option<DenseLin>,
    pub k_norm: Option<Vec<f32>>,
}

/// Every attention weight of one backbone layer.
pub struct AttnWeights {
    pub layer: usize,
    pub wq_a: Fp8Lin,
    pub q_norm: Vec<f32>,
    pub wq_b: Fp8Lin,
    pub wkv: Fp8Lin,
    pub kv_norm: Vec<f32>,
    /// [o_groups * o_lora, heads * head_dim / o_groups] bf16 values
    pub wo_a: Vec<f32>,
    pub wo_b: Fp8Lin,
    pub sink: Vec<f32>,
    pub comp: Option<CompressorW>,
    pub idx: Option<IndexerW>,
}

impl AttnWeights {
    /// Load layer `l` (a backbone layer: 0..cfg.layers). Only the pieces the config says the
    /// layer has are read (a compressor at KV sources, an indexer at index sources, `wk` at owners).
    pub fn load(ck: &Ckpt, cfg: &Dsv41Config, l: usize) -> Result<AttnWeights> {
        ensure!(l < cfg.layers, "layer {l}: the attention oracle covers the backbone (0..{})", cfg.layers);
        let p = format!("layers.{l}.attn");
        let (d, hd, ql) = (cfg.hidden, cfg.head_dim, cfg.q_lora_rank);
        let (g, ol) = (cfg.o_groups, cfg.o_lora_rank);
        let gin = cfg.heads * hd / g;
        let ratio = cfg.ratio(l) as usize;
        let comp = if ratio > 0 && cfg.kv_sources.contains(&l) {
            Some(CompressorW {
                ratio,
                wkv: ck.dense(&format!("{p}.compressor.wkv.weight"), hd, d)?,
                wgate: if ratio > 1 { Some(ck.dense(&format!("{p}.compressor.wgate.weight"), hd, d)?) } else { None },
                norm: ck.vec(&format!("{p}.compressor.norm.weight"), hd)?,
            })
        } else {
            None
        };
        let idx = if ratio > 0 && cfg.index_sources.contains(&l) {
            let (ih, ihd) = (cfg.index_heads, cfg.index_head_dim);
            let owner = cfg.kv_sources.contains(&l);
            Some(IndexerW {
                wq_b: ck.fp8(&format!("{p}.indexer.wq_b"), ih * ihd, ql)?,
                weights_proj: ck.dense(&format!("{p}.indexer.weights_proj.weight"), ih, d)?,
                wk: if owner { Some(ck.dense(&format!("{p}.indexer.wk.weight"), ihd, hd)?) } else { None },
                k_norm: if owner { Some(ck.vec(&format!("{p}.indexer.k_norm.weight"), ihd)?) } else { None },
            })
        } else {
            None
        };
        Ok(AttnWeights {
            layer: l,
            wq_a: ck.fp8(&format!("{p}.wq_a"), ql, d)?,
            q_norm: ck.vec(&format!("{p}.q_norm.weight"), ql)?,
            wq_b: ck.fp8(&format!("{p}.wq_b"), cfg.heads * hd, ql)?,
            wkv: ck.fp8(&format!("{p}.wkv"), hd, d)?,
            kv_norm: ck.vec(&format!("{p}.kv_norm.weight"), hd)?,
            wo_a: ck.fp8_to_bf16(&format!("{p}.wo_a"), g * ol, gin)?,
            wo_b: ck.fp8(&format!("{p}.wo_b"), d, g * ol)?,
            sink: ck.vec(&format!("{p}.attn_sink"), cfg.heads)?,
            comp,
            idx,
        })
    }
}
