//! DSV4.1 leg OD: the DSpark oracle's weights, read from a NATIVE checkpoint (or a native slice that
//! keeps the `mtp.*` stages plus `embed` / `head`), as model.py + convert.py hold them at run time:
//!
//! | tensor (checkpoint) | stored | oracle form |
//! |---|---|---|
//! | `mtp.0.main_proj` | e4m3 [5120, 15360] + e8m0 32x32 | `QLin::Fp8` |
//! | `mtp.s.attn.{wq_a, wq_b, wkv, wo_b}`, `ffn.shared_experts.w1/w2/w3` | e4m3 + e8m0 32x32 | `QLin::Fp8` |
//! | `mtp.s.attn.wo_a` | e4m3 [8192, 4096] + e8m0 32x32 | dequantised to bf16 (convert.py 192-205: exact) |
//! | `mtp.s.ffn.experts.E.w1/w2/w3` | int8 = e2m1 x2 + e8m0 [n, k/32] | `QLin::Fp4`, read on first use |
//! | `mtp.s.ffn.gate.weight` | bf16 [128, 5120] | bf16 (the reference widens to fp32 exactly) |
//! | `mtp.s.ffn.gate.bias`, `attn.attn_sink`, `hc_*` | f32 | f32 |
//! | norms, `markov_head.{embed,head}`, `confidence_head.proj` | bf16 | bf16 (head / proj widened exactly) |
//! | `embed`, `head` (tied to the backbone) | bf16 | embed rows read on demand; head resident |
//!
//! Nothing here is per-pack: names and shapes come from the checkpoint index and the config. An EXL3
//! pack would need a host trellis decode (leg KE) behind the same `QLin`-shaped interface; see the
//! OD report's handoff.

use super::dspark_num::{Bf16Mat, F32Mat, QLin};
use crate::dsv41::config::Dsv41Config;
use crate::dsv41::st::{self, StHeader};
use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// A safetensors checkpoint addressed by tensor name (`model.safetensors.index.json`, or the one
/// `*.safetensors` file of a single-file dir).
pub struct Ckpt {
    pub dir: PathBuf,
    map: HashMap<String, String>,
    headers: Mutex<HashMap<String, Arc<StHeader>>>,
}

impl Ckpt {
    pub fn open(dir: &Path) -> Result<Ckpt> {
        let idx = dir.join("model.safetensors.index.json");
        let mut map = HashMap::new();
        if idx.exists() {
            let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&idx)?).with_context(|| format!("{}", idx.display()))?;
            let wm = v.get("weight_map").and_then(|m| m.as_object()).context("index has no weight_map")?;
            for (k, f) in wm { map.insert(k.clone(), f.as_str().unwrap_or_default().to_string()); }
        } else {
            for e in std::fs::read_dir(dir)? {
                let p = e?.path();
                if p.extension().is_some_and(|x| x == "safetensors") {
                    let h = st::read_header(&p)?;
                    let f = p.file_name().unwrap().to_string_lossy().to_string();
                    for t in &h.tensors { map.insert(t.name.clone(), f.clone()); }
                }
            }
        }
        ensure!(!map.is_empty(), "{}: no tensors", dir.display());
        Ok(Ckpt { dir: dir.to_path_buf(), map, headers: Mutex::new(HashMap::new()) })
    }

    pub fn has(&self, name: &str) -> bool { self.map.contains_key(name) }

    fn header(&self, name: &str) -> Result<Arc<StHeader>> {
        let f = self.map.get(name).with_context(|| format!("{}: no tensor {name}", self.dir.display()))?;
        let mut hs = self.headers.lock().unwrap();
        if let Some(h) = hs.get(f) { return Ok(h.clone()); }
        let h = Arc::new(st::read_header(&self.dir.join(f))?);
        hs.insert(f.clone(), h.clone());
        Ok(h)
    }

    /// (dtype, shape) of a tensor.
    pub fn meta(&self, name: &str) -> Result<(String, Vec<usize>)> {
        let h = self.header(name)?;
        let t = h.tensors.iter().find(|t| t.name == name).with_context(|| format!("{name} not in its shard header"))?;
        Ok((t.dtype.clone(), t.shape.iter().map(|&x| x as usize).collect()))
    }

    /// A tensor's raw bytes, after checking its dtype and shape.
    pub fn raw(&self, name: &str, dtype: &str, shape: &[usize]) -> Result<Vec<u8>> {
        let (dt, sh) = self.meta(name)?;
        ensure!(dt == dtype && sh == shape, "{name}: checkpoint has {dt} {sh:?}, the oracle expects {dtype} {shape:?}");
        let h = self.header(name)?;
        st::read_tensor(&h, name)
    }

    /// Rows `[r0, r1)` of a 2-D tensor (the embedding).
    pub fn rows(&self, name: &str, r0: usize, r1: usize) -> Result<Vec<u8>> {
        let h = self.header(name)?;
        let t = h.tensors.iter().find(|t| t.name == name).context("tensor")?;
        ensure!(t.shape.len() == 2 && (r1 as u64) <= t.shape[0] && r0 < r1, "{name}: rows {r0}..{r1} of {:?}", t.shape);
        let es = st::dtype_size(&t.dtype).context("dtype")?;
        let row = t.shape[1] * es;
        let mut f = std::fs::File::open(&h.path)?;
        f.seek(SeekFrom::Start(h.data_start + t.begin + r0 as u64 * row))?;
        let mut v = vec![0u8; (r1 - r0) * row as usize];
        f.read_exact(&mut v)?;
        Ok(v)
    }

    pub fn bf16(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
        Ok(self.raw(name, "BF16", shape)?.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect())
    }

    /// A bf16 vector as f32 values.
    pub fn bf16_f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        Ok(self.bf16(name, shape)?.into_iter().map(super::dspark_num::bf16_bits).collect())
    }

    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        Ok(self.raw(name, "F32", shape)?.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    }

    /// An fp8 linear `[n, k]` with its 32 x 32 e8m0 block scales.
    pub fn fp8(&self, base: &str, n: usize, k: usize) -> Result<QLin> {
        let w = self.raw(&format!("{base}.weight"), "F8_E4M3", &[n, k])?;
        let s = self.raw(&format!("{base}.scale"), "F8_E8M0", &[n.div_ceil(32), k.div_ceil(32)])?;
        Ok(QLin::Fp8 { n, k, w, s })
    }

    /// An MXFP4 expert linear `[n, k]` (int8 `[n, k/2]` holding two e2m1 codes, e8m0 `[n, k/32]`).
    pub fn fp4(&self, base: &str, n: usize, k: usize) -> Result<QLin> {
        let w = self.raw(&format!("{base}.weight"), "I8", &[n, k / 2])?;
        let s = self.raw(&format!("{base}.scale"), "F8_E8M0", &[n, k / 32])?;
        Ok(QLin::Fp4 { n, k, w, s })
    }

    /// convert.py 192-205: an fp8 weight with 32 x 32 (or 128 x 128) e8m0 blocks dequantised to bf16
    /// (`wo_a`; exact: an e4m3 value times a power of two fits bf16).
    pub fn fp8_to_bf16(&self, base: &str, n: usize, k: usize) -> Result<Bf16Mat> {
        let w = self.raw(&format!("{base}.weight"), "F8_E4M3", &[n, k])?;
        let (_, ssh) = self.meta(&format!("{base}.scale"))?;
        ensure!(ssh.len() == 2 && n % ssh[0] == 0 && k % ssh[1] == 0, "{base}.scale shape {ssh:?}");
        let (ob, ib) = (n / ssh[0], k / ssh[1]);
        ensure!((ob, ib) == (32, 32) || (ob, ib) == (128, 128), "{base}: block {ob}x{ib}");
        let s = self.raw(&format!("{base}.scale"), "F8_E8M0", &ssh)?;
        let lut = super::dspark_num::e4m3_lut();
        let mut out = Vec::with_capacity(n * k);
        for r in 0..n {
            for c in 0..k {
                let v = lut[w[r * k + c] as usize] * super::dspark_num::e8m0_decode(s[(r / ob) * ssh[1] + c / ib]);
                let b = super::dspark_num::bf16r(v);
                ensure!(b == v, "{base}[{r},{c}]: {v} is not exact in bf16");
                out.push((b.to_bits() >> 16) as u16);
            }
        }
        Ok(Bf16Mat { n, k, w: out })
    }
}

/// One routed expert of a DSpark stage (w1 gate, w3 up: [inter, dim]; w2 down: [dim, inter]).
pub struct ExpertW { pub w1: QLin, pub w2: QLin, pub w3: QLin }

/// One DSpark stage (`mtp.{s}`), every resident tensor.
pub struct StageW {
    pub stage: usize,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub hc_attn_fn: F32Mat,
    pub hc_attn_base: Vec<f32>,
    pub hc_attn_scale: Vec<f32>,
    pub hc_ffn_fn: F32Mat,
    pub hc_ffn_base: Vec<f32>,
    pub hc_ffn_scale: Vec<f32>,
    pub wq_a: QLin,
    pub q_norm: Vec<f32>,
    pub wq_b: QLin,
    pub wkv: QLin,
    pub kv_norm: Vec<f32>,
    pub wo_a: Bf16Mat,
    pub wo_b: QLin,
    pub sink: Vec<f32>,
    pub gate_w: Bf16Mat,
    pub gate_bias: Vec<f32>,
    pub shared: ExpertW,
    experts: Mutex<HashMap<usize, Arc<ExpertW>>>,
}

/// The whole drafter: three stages, the stage-0 main projection, the stage-2 head pieces, and the
/// tied backbone embedding / head.
pub struct DsparkW {
    pub cfg: Dsv41Config,
    pub ckpt: Ckpt,
    pub main_proj: QLin,
    pub main_norm: Vec<f32>,
    pub stages: Vec<StageW>,
    pub norm: Vec<f32>,
    pub head: Bf16Mat,
    pub markov_embed: Bf16Mat,
    pub markov_head: Bf16Mat,
    pub confidence: Bf16Mat,
}

impl StageW {
    fn load(ck: &Ckpt, cfg: &Dsv41Config, stage: usize) -> Result<StageW> {
        let p = format!("mtp.{stage}");
        let d = cfg.hidden;
        let hd = cfg.head_dim;
        let mix = cfg.hc_mix();
        let inter = cfg.moe_inter;
        let h = cfg.heads;
        let og = cfg.o_groups;
        let ol = cfg.o_lora_rank;
        let ql = cfg.q_lora_rank;
        let hcd = cfg.hc * d;
        let shared = ExpertW {
            w1: ck.fp8(&format!("{p}.ffn.shared_experts.w1"), inter, d)?,
            w2: ck.fp8(&format!("{p}.ffn.shared_experts.w2"), d, inter)?,
            w3: ck.fp8(&format!("{p}.ffn.shared_experts.w3"), inter, d)?,
        };
        Ok(StageW {
            stage,
            attn_norm: ck.bf16_f32(&format!("{p}.attn_norm.weight"), &[d])?,
            ffn_norm: ck.bf16_f32(&format!("{p}.ffn_norm.weight"), &[d])?,
            hc_attn_fn: F32Mat { n: mix, k: hcd, w: ck.f32(&format!("{p}.hc_attn_fn"), &[mix, hcd])? },
            hc_attn_base: ck.f32(&format!("{p}.hc_attn_base"), &[mix])?,
            hc_attn_scale: ck.f32(&format!("{p}.hc_attn_scale"), &[3])?,
            hc_ffn_fn: F32Mat { n: mix, k: hcd, w: ck.f32(&format!("{p}.hc_ffn_fn"), &[mix, hcd])? },
            hc_ffn_base: ck.f32(&format!("{p}.hc_ffn_base"), &[mix])?,
            hc_ffn_scale: ck.f32(&format!("{p}.hc_ffn_scale"), &[3])?,
            wq_a: ck.fp8(&format!("{p}.attn.wq_a"), ql, d)?,
            q_norm: ck.bf16_f32(&format!("{p}.attn.q_norm.weight"), &[ql])?,
            wq_b: ck.fp8(&format!("{p}.attn.wq_b"), h * hd, ql)?,
            wkv: ck.fp8(&format!("{p}.attn.wkv"), hd, d)?,
            kv_norm: ck.bf16_f32(&format!("{p}.attn.kv_norm.weight"), &[hd])?,
            wo_a: ck.fp8_to_bf16(&format!("{p}.attn.wo_a"), og * ol, h * hd / og)?,
            wo_b: ck.fp8(&format!("{p}.attn.wo_b"), d, og * ol)?,
            sink: ck.f32(&format!("{p}.attn.attn_sink"), &[h])?,
            gate_w: Bf16Mat { n: cfg.dspark_routed, k: d, w: ck.bf16(&format!("{p}.ffn.gate.weight"), &[cfg.dspark_routed, d])? },
            gate_bias: ck.f32(&format!("{p}.ffn.gate.bias"), &[cfg.dspark_routed])?,
            shared,
            experts: Mutex::new(HashMap::new()),
        })
    }

    /// Routed expert `e`, read from the checkpoint on first use and kept.
    pub fn expert(&self, ck: &Ckpt, cfg: &Dsv41Config, e: usize) -> Result<Arc<ExpertW>> {
        if let Some(x) = self.experts.lock().unwrap().get(&e) { return Ok(x.clone()); }
        ensure!(e < cfg.dspark_routed, "expert {e} >= {}", cfg.dspark_routed);
        let p = format!("mtp.{}.ffn.experts.{e}", self.stage);
        let (d, inter) = (cfg.hidden, cfg.moe_inter);
        let x = Arc::new(ExpertW { w1: ck.fp4(&format!("{p}.w1"), inter, d)?, w2: ck.fp4(&format!("{p}.w2"), d, inter)?, w3: ck.fp4(&format!("{p}.w3"), inter, d)? });
        self.experts.lock().unwrap().insert(e, x.clone());
        Ok(x)
    }

    pub fn experts_loaded(&self) -> usize { self.experts.lock().unwrap().len() }
}

impl DsparkW {
    /// Load every resident DSpark tensor from a native checkpoint (or slice) directory.
    pub fn load(dir: &Path) -> Result<DsparkW> {
        let cfg = Dsv41Config::load(dir)?;
        ensure!(cfg.dspark_stages > 0 && cfg.dspark_block > 0, "{}: no DSpark stages in the config", dir.display());
        ensure!(cfg.quant.method != "exl3", "{}: an EXL3 pack; the DSpark oracle reads the native checkpoint (FP8 / MXFP4) only", dir.display());
        let ck = Ckpt::open(dir)?;
        let (d, v) = (cfg.hidden, cfg.vocab);
        let last = cfg.dspark_stages - 1;
        if !ck.has("mtp.0.main_proj.weight") { bail!("{}: no mtp.0.main_proj (not a checkpoint with the DSpark stages)", dir.display()); }
        let stages = (0..cfg.dspark_stages).map(|s| StageW::load(&ck, &cfg, s)).collect::<Result<Vec<_>>>()?;
        let nt = cfg.dspark_targets.len();
        let r = cfg.dspark_markov_rank;
        let main_proj = ck.fp8("mtp.0.main_proj", d, nt * d)?;
        let main_norm = ck.bf16_f32("mtp.0.main_norm.weight", &[d])?;
        let norm = ck.bf16_f32(&format!("mtp.{last}.norm.weight"), &[d])?;
        let head = Bf16Mat { n: v, k: d, w: ck.bf16("head.weight", &[v, d])? };
        let markov_embed = Bf16Mat { n: v, k: r, w: ck.bf16(&format!("mtp.{last}.markov_head.embed.weight"), &[v, r])? };
        let markov_head = Bf16Mat { n: v, k: r, w: ck.bf16(&format!("mtp.{last}.markov_head.head.weight"), &[v, r])? };
        let confidence = Bf16Mat { n: 1, k: d + r, w: ck.bf16(&format!("mtp.{last}.confidence_head.proj.weight"), &[1, d + r])? };
        ensure!(ck.has("embed.weight"), "{}: no embed.weight (the DSpark stages share the backbone embedding)", dir.display());
        Ok(DsparkW { cfg, ckpt: ck, main_proj, main_norm, stages, norm, head, markov_embed, markov_head, confidence })
    }

    /// Embedding row of token `t` (bf16 values).
    pub fn embed_row(&self, t: u32) -> Result<Vec<f32>> {
        let d = self.cfg.hidden;
        let b = self.ckpt.rows("embed.weight", t as usize, t as usize + 1)?;
        ensure!(b.len() == 2 * d, "embed row width");
        Ok(b.chunks_exact(2).map(|x| super::dspark_num::bf16_bits(u16::from_le_bytes([x[0], x[1]]))).collect())
    }
}
