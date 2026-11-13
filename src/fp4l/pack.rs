//! FP4L — reader for the NVIDIA ModelOpt mixed-precision Qwen3.8-Flash-Next layout
//! (`nvidia/Qwen3.8-Flash-Next-FP4-Local`): `config.json` with `quantization_config.quant_method =
//! "modelopt"`, `model.safetensors.index.json`, shards under `tensors/`.
//!
//! The reader classifies every LINEAR by the tensors present next to its `.weight` (the config's
//! `quantized_layers` is cross-checked, never trusted alone — it names the MTP experts
//! `mtp.layers.48.*` while the tensors are `mtp.layers.0.*`):
//!
//! | kind | tensors | dequant |
//! |---|---|---|
//! | `Mxfp8` | `weight` F8_E4M3 `[n, k]`, `weight_scale` U8 (UE8M0) `[n, k/32]` | `e4m3(w) * 2^(s - 127)` |
//! | `Nvfp4Csf` | `weight` U8 `[n, k/2]` (low nibble = even k), `weight_scale.nvfp4_csf_{fixed,exceptions}`, `weight_scale_2` F32 `[]`, `input_scale` F32 `[]` (A4 only, unused) | `e2m1(q) * e4m3(s) * weight_scale_2` |
//! | `Nvfp4` | `weight` U8 `[n, k/2]`, `weight_scale` F8_E4M3 `[n, k/16]`, `weight_scale_2` F32 `[]` | same |
//! | `Bf16` | `weight` BF16 `[n, k]` | exact |
//!
//! Every read is a positioned read of the exact byte range (no mmap: 375K tensors over 21 files).

use super::csf;
use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

/// One tensor's location and type inside the pack.
#[derive(Clone, Debug)]
pub struct TRef {
    pub file: u16,        // index into Fp4lPack::files
    pub dtype: Dt,
    pub shape: Vec<usize>,
    pub off: u64,         // absolute byte offset of the data in the file
    pub len: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dt { Bf16, F16, F32, U8, U32, I64, F8e4m3 }

impl Dt {
    fn parse(s: &str) -> Result<Dt> {
        Ok(match s {
            "BF16" => Dt::Bf16, "F16" => Dt::F16, "F32" => Dt::F32, "U8" => Dt::U8, "U32" => Dt::U32,
            "I64" => Dt::I64, "F8_E4M3" => Dt::F8e4m3,
            other => bail!("FP4L: unsupported safetensors dtype {other}"),
        })
    }
    pub fn size(self) -> usize { match self { Dt::Bf16 | Dt::F16 => 2, Dt::F32 | Dt::U32 => 4, Dt::I64 => 8, Dt::U8 | Dt::F8e4m3 => 1 } }
}

/// How a linear module is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinKind { Mxfp8, Nvfp4Csf, Nvfp4, Bf16 }

/// MXFP8 host image: OCP MX E4M3 codes `[n][k]` (k contiguous) + UE8M0 scales `[n][k/32]`.
#[derive(Clone)]
pub struct Mx8Host { pub codes: Vec<u8>, pub scales: Vec<u8>, pub n: usize, pub k: usize }

/// NVFP4 host image: nibbles `[n][k/2]` (low nibble = even k), DECODED E4M3 block scales `[n][k/16]`,
/// the per-tensor global multiplier `gs` (`weight_scale_2`; dequant MULTIPLIES by it).
#[derive(Clone)]
pub struct Nv4Host { pub packed: Vec<u8>, pub scales: Vec<u8>, pub gs: f32, pub n: usize, pub k: usize }

/// Any directory of safetensors files (an index json, or every `*.safetensors` at the top level).
pub struct StDir {
    pub dir: PathBuf,
    pub files: Vec<std::fs::File>,
    pub file_names: Vec<String>,
    pub t: HashMap<String, TRef>,
}

impl StDir {
    pub fn open(dir: &Path) -> Result<StDir> {
        let idx_p = dir.join("model.safetensors.index.json");
        let (file_names, wm) = if idx_p.is_file() {
            let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(&idx_p)?)?;
            let wm = idx.get("weight_map").and_then(|x| x.as_object()).context("index has no weight_map")?.clone();
            let mut f: Vec<String> = wm.values().filter_map(|v| v.as_str().map(String::from)).collect();
            f.sort();
            f.dedup();
            (f, Some(wm))
        } else {
            let mut f: Vec<String> = std::fs::read_dir(dir)?.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| n.ends_with(".safetensors")).collect();
            f.sort();
            (f, None)
        };
        ensure!(!file_names.is_empty(), "{}: no safetensors files", dir.display());
        let mut files = Vec::new();
        let mut t = HashMap::new();
        for (fi, f) in file_names.iter().enumerate() {
            let p = dir.join(f);
            let fh = std::fs::File::open(&p).with_context(|| format!("open {}", p.display()))?;
            let mut len8 = [0u8; 8];
            fh.read_exact_at(&mut len8, 0)?;
            let n = u64::from_le_bytes(len8);
            ensure!(n < (1 << 31), "{}: implausible header length {n}", p.display());
            let mut hb = vec![0u8; n as usize];
            fh.read_exact_at(&mut hb, 8)?;
            let h: serde_json::Value = serde_json::from_slice(&hb)?;
            for (name, e) in h.as_object().context("shard header is not an object")? {
                if name == "__metadata__" { continue; }
                let dt = Dt::parse(e["dtype"].as_str().unwrap_or("?"))?;
                let shape: Vec<usize> = e["shape"].as_array().context("shape")?.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
                let o = e["data_offsets"].as_array().context("data_offsets")?;
                let (a, b) = (o[0].as_u64().unwrap_or(0), o[1].as_u64().unwrap_or(0));
                let want = shape.iter().product::<usize>() as u64 * dt.size() as u64;
                ensure!(b - a == want, "{name}: {} bytes for shape {:?} {:?}", b - a, shape, dt);
                t.insert(name.clone(), TRef { file: fi as u16, dtype: dt, shape, off: 8 + n + a, len: b - a });
            }
            files.push(fh);
        }
        if let Some(wm) = wm {
            for (name, f) in &wm {
                let f = f.as_str().unwrap_or("");
                let tr = t.get(name).with_context(|| format!("index names {name} but no shard header has it"))?;
                ensure!(file_names[tr.file as usize] == f, "{name}: index says {f}, header found in {}", file_names[tr.file as usize]);
            }
        }
        Ok(StDir { dir: dir.to_path_buf(), files, file_names, t })
    }

    pub fn has(&self, name: &str) -> bool { self.t.contains_key(name) }
    pub fn get(&self, name: &str) -> Result<&TRef> { self.t.get(name).with_context(|| format!("missing tensor {name}")) }

    pub fn raw(&self, name: &str) -> Result<Vec<u8>> {
        let r = self.get(name)?;
        let mut v = vec![0u8; r.len as usize];
        self.files[r.file as usize].read_exact_at(&mut v, r.off).with_context(|| format!("read {name}"))?;
        Ok(v)
    }
    /// Bytes `[a, b)` of a tensor's data (row slices of large tensors, PLE rows, one expert of a stack).
    pub fn raw_range(&self, name: &str, a: u64, b: u64) -> Result<Vec<u8>> {
        let r = self.get(name)?;
        ensure!(a <= b && b <= r.len, "{name}: range [{a}, {b}) outside {} bytes", r.len);
        let mut v = vec![0u8; (b - a) as usize];
        self.files[r.file as usize].read_exact_at(&mut v, r.off + a)?;
        Ok(v)
    }
    /// A plain (BF16 / F16 / F32) tensor, or a byte range of one, as f32 values.
    pub fn plain_f32_range(&self, name: &str, a_elem: usize, b_elem: usize) -> Result<Vec<f32>> {
        let r = self.get(name)?;
        let sz = r.dtype.size() as u64;
        let b = self.raw_range(name, a_elem as u64 * sz, b_elem as u64 * sz)?;
        Ok(match r.dtype {
            Dt::Bf16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            Dt::F16 => b.chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
            Dt::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            d => bail!("{name}: dtype {d:?} is not a plain float tensor"),
        })
    }
    pub fn plain_f32(&self, name: &str) -> Result<Vec<f32>> {
        let n = self.get(name)?.shape.iter().product::<usize>();
        self.plain_f32_range(name, 0, n)
    }
    pub fn numel(&self, name: &str) -> Result<usize> { Ok(self.get(name)?.shape.iter().product()) }
}

pub struct Fp4lPack {
    pub st: StDir,
    pub cfg_json: serde_json::Value,
    /// `quantization_config.quantized_layers` (module -> quant_algo), for the cross-check.
    pub qlayers: HashMap<String, String>,
}

impl std::ops::Deref for Fp4lPack {
    type Target = StDir;
    fn deref(&self) -> &StDir { &self.st }
}

/// The quant formats this module serves, checked from `quantization_config.quantized_layers` (module -> quant_algo).
/// Every unsupported format is collected (with a module that uses it) into ONE error. 2026-11-07 (owner): NVIDIA's
/// ModelOpt NVFP4 Flash-Next pack (FP8 n-gram table, FP8_BLOCK_SCALES MTP experts) is NOT supported — its
/// FP4-Local variant is; the message says so instead of a bare "quant_algo FP8 is not supported".
pub fn check_quant_config(q: &serde_json::Value) -> Result<HashMap<String, String>> {
    let mut qlayers = HashMap::new();
    let mut unsupported: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    if let Some(m) = q.get("quantized_layers").and_then(|x| x.as_object()) {
        for (k, v) in m {
            let algo = v.get("quant_algo").and_then(|x| x.as_str()).unwrap_or("?").to_string();
            if let Some(enc) = v.get("weight_scale_encoding").and_then(|x| x.as_str()) {
                ensure!(enc == "csf", "FP4L: {k}: weight_scale_encoding {enc:?} is not supported (only \"csf\" = {})", csf::CODEC);
            }
            let gs = v.get("group_size").and_then(|x| x.as_u64()).unwrap_or(0);
            match algo.as_str() {
                "MXFP8" => ensure!(gs == 32, "FP4L: {k}: MXFP8 group_size {gs} != 32"),
                "NVFP4" | "W4A16_NVFP4" => ensure!(gs == 16, "FP4L: {k}: {algo} group_size {gs} != 16"),
                other => { unsupported.entry(other.to_string()).or_insert_with(|| k.clone()); }
            }
            qlayers.insert(k.clone(), algo);
        }
    }
    if !unsupported.is_empty() {
        let list = unsupported.iter().map(|(a, k)| format!("{a} ({k})")).collect::<Vec<_>>().join(", ");
        let hint = if unsupported.contains_key("FP8") || unsupported.contains_key("FP8_BLOCK_SCALES") {
            " This looks like NVIDIA's ModelOpt NVFP4 Flash-Next checkpoint, which this engine does not serve; its FP4-Local \
variant (nvidia/Qwen3.8-Flash-Next-FP4-Local: MXFP8 dense + NVFP4 experts) is supported."
        } else { "" };
        bail!("FP4L: this ModelOpt pack uses quant format(s) the engine does not serve: {list}. Served: MXFP8 (group 32) and \
NVFP4 (group 16).{hint}");
    }
    Ok(qlayers)
}

/// Cheap pre-flight (config.json only): the same format check as `Fp4lPack::open`, for the TP head to run BEFORE it ships
/// the pack to its nodes (2026-11-07: an unsupported pack used to ship ~98 GB per node before a node refused it).
pub fn preflight(dir: &Path) -> Result<()> {
    let cfg_json: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))
        .with_context(|| format!("FP4L: read {}/config.json", dir.display()))?)?;
    let q = cfg_json.get("quantization_config").context("FP4L: config.json has no quantization_config")?;
    check_quant_config(q).map(|_| ())
}

#[cfg(test)]
mod quant_check_tests {
    fn q(layers: serde_json::Value) -> serde_json::Value { serde_json::json!({"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": layers}) }
    #[test]
    fn fp4_local_formats_pass_and_the_nvidia_nvfp4_pack_is_refused_with_the_hint() {
        let ok = q(serde_json::json!({"a.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16, "weight_scale_encoding": "csf"},
                                      "a.self_attn.q_proj": {"quant_algo": "MXFP8", "group_size": 32}}));
        assert_eq!(super::check_quant_config(&ok).unwrap().len(), 2);
        let nv = q(serde_json::json!({"l1.ple.ple_embedding.ngram_embedding": {"quant_algo": "FP8"},
                                      "mtp.layers.0.mlp.experts": {"quant_algo": "FP8_BLOCK_SCALES", "group_size": 128},
                                      "l0.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16}}));
        let e = format!("{:#}", super::check_quant_config(&nv).unwrap_err());
        assert!(e.contains("FP8 (l1.ple.ple_embedding.ngram_embedding)") && e.contains("FP8_BLOCK_SCALES (mtp.layers.0.mlp.experts)"), "{e}");
        assert!(e.contains("FP4-Local"), "{e}");
    }
}

/// True when `dir` holds a ModelOpt mixed-precision qwen4_exp checkpoint this module serves.
pub fn is_fp4l_dir(dir: &Path) -> bool {
    let Ok(raw) = std::fs::read(dir.join("config.json")) else { return false };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&raw) else { return false };
    let mt = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
    let q = v.get("quantization_config");
    let qm = q.and_then(|q| q.get("quant_method")).and_then(|x| x.as_str()).unwrap_or("");
    mt.starts_with("qwen4_exp") && qm == "modelopt" && dir.join("model.safetensors.index.json").is_file()
}

impl Fp4lPack {
    pub fn open(dir: &Path) -> Result<Fp4lPack> {
        let cfg_json: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))
            .with_context(|| format!("FP4L: read {}/config.json", dir.display()))?)?;
        let q = cfg_json.get("quantization_config").context("FP4L: config.json has no quantization_config")?;
        ensure!(q.get("quant_method").and_then(|x| x.as_str()) == Some("modelopt"),
                "FP4L: quant_method is not modelopt");
        ensure!(q.get("quant_algo").and_then(|x| x.as_str()) == Some("MIXED_PRECISION"),
                "FP4L: quant_algo is not MIXED_PRECISION");
        let qlayers = check_quant_config(q)?;
        Ok(Fp4lPack { st: StDir::open(dir)?, cfg_json, qlayers })
    }

    /// Storage kind of the linear module `stem` (no `.weight` suffix).
    pub fn lin_kind(&self, stem: &str) -> Result<LinKind> {
        let w = self.get(&format!("{stem}.weight"))?;
        let k = match w.dtype {
            Dt::F8e4m3 => {
                let s = self.get(&format!("{stem}.weight_scale"))?;
                ensure!(s.dtype == Dt::U8, "FP4L: {stem}: MXFP8 scale dtype {:?}", s.dtype);
                LinKind::Mxfp8
            }
            Dt::U8 if self.has(&format!("{stem}.weight_scale.nvfp4_csf_fixed")) => LinKind::Nvfp4Csf,
            Dt::U8 => {
                let s = self.get(&format!("{stem}.weight_scale"))?;
                ensure!(s.dtype == Dt::F8e4m3, "FP4L: {stem}: NVFP4 scale dtype {:?}", s.dtype);
                LinKind::Nvfp4
            }
            Dt::Bf16 => LinKind::Bf16,
            d => bail!("FP4L: {stem}: weight dtype {d:?} is not a linear this reader knows"),
        };
        Ok(k)
    }

    pub fn shape2(&self, name: &str) -> Result<(usize, usize)> {
        let r = self.get(name)?;
        ensure!(r.shape.len() == 2, "FP4L: {name}: shape {:?} is not 2-D", r.shape);
        Ok((r.shape[0], r.shape[1]))
    }

    /// MXFP8 linear (`stem` without `.weight`).
    pub fn mxfp8(&self, stem: &str) -> Result<Mx8Host> {
        ensure!(self.lin_kind(stem)? == LinKind::Mxfp8, "FP4L: {stem} is not MXFP8");
        let (n, k) = self.shape2(&format!("{stem}.weight"))?;
        let (sn, sk) = self.shape2(&format!("{stem}.weight_scale"))?;
        ensure!(k % 32 == 0 && sn == n && sk == k / 32, "FP4L: {stem}: MXFP8 scales [{sn}, {sk}] for weight [{n}, {k}]");
        Ok(Mx8Host { codes: self.raw(&format!("{stem}.weight"))?, scales: self.raw(&format!("{stem}.weight_scale"))?, n, k })
    }

    /// MXFP8 linear, rows `[r0, r1)` only (TP column shards = output rows).
    pub fn mxfp8_rows(&self, stem: &str, r0: usize, r1: usize) -> Result<Mx8Host> {
        let (n, k) = self.shape2(&format!("{stem}.weight"))?;
        ensure!(r0 <= r1 && r1 <= n, "FP4L: {stem}: rows [{r0}, {r1}) outside {n}");
        Ok(Mx8Host {
            codes: self.raw_range(&format!("{stem}.weight"), (r0 * k) as u64, (r1 * k) as u64)?,
            scales: self.raw_range(&format!("{stem}.weight_scale"), (r0 * k / 32) as u64, (r1 * k / 32) as u64)?,
            n: r1 - r0, k,
        })
    }

    /// NVFP4 linear (csf or plain scales), scales decoded.
    pub fn nvfp4(&self, stem: &str) -> Result<Nv4Host> {
        let kind = self.lin_kind(stem)?;
        let (n, k2) = self.shape2(&format!("{stem}.weight"))?;
        let k = k2 * 2;
        ensure!(k % 16 == 0, "FP4L: {stem}: K {k} not a multiple of 16");
        let scales = match kind {
            LinKind::Nvfp4Csf => {
                let fname = format!("{stem}.weight_scale.nvfp4_csf_fixed");
                let fr = self.get(&fname)?;
                let s = csf::decode(&self.raw(&fname)?, &fr.shape, &self.raw(&format!("{stem}.weight_scale.nvfp4_csf_exceptions"))?)
                    .with_context(|| format!("FP4L: csf decode {stem}"))?;
                let (rows, cols) = csf::geometry(&fr.shape)?;
                ensure!(rows == n && cols == k / 16, "FP4L: {stem}: csf plane {rows}x{cols} for weight [{n}, {k}]");
                s
            }
            LinKind::Nvfp4 => {
                let (sn, sk) = self.shape2(&format!("{stem}.weight_scale"))?;
                ensure!(sn == n && sk == k / 16, "FP4L: {stem}: scales [{sn}, {sk}] for weight [{n}, {k}]");
                self.raw(&format!("{stem}.weight_scale"))?
            }
            other => bail!("FP4L: {stem} is {other:?}, not NVFP4"),
        };
        let gs = self.scalar_f32(&format!("{stem}.weight_scale_2"))?;
        ensure!(gs.is_finite() && gs > 0.0, "FP4L: {stem}: weight_scale_2 {gs} is not a positive finite multiplier");
        Ok(Nv4Host { packed: self.raw(&format!("{stem}.weight"))?, scales, gs, n, k })
    }

    pub fn scalar_f32(&self, name: &str) -> Result<f32> {
        let r = self.get(name)?;
        ensure!(r.dtype == Dt::F32 && r.len == 4, "FP4L: {name}: want one F32, got {:?} {:?}", r.dtype, r.shape);
        let b = self.raw(name)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn i64s(&self, name: &str) -> Result<Vec<i64>> {
        let r = self.get(name)?;
        ensure!(r.dtype == Dt::I64, "FP4L: {name}: want I64, got {:?}", r.dtype);
        Ok(self.raw(name)?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
    }

    /// The text config (`text_config` of the VL config) as the engine's `qwen::Config` input.
    pub fn text_config(&self) -> &serde_json::Value {
        self.cfg_json.get("text_config").unwrap_or(&self.cfg_json)
    }
}

// --------------------------------------------------------------------------------------------
// Exact host decoders (the reference side of every FP4L gate).
// --------------------------------------------------------------------------------------------

/// E2M1 code (sign in bit 3) -> value.
pub const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// OCP E4M3 ("fn": no infinities, 0x7F/0xFF = NaN) byte -> f32, exact.
pub fn e4m3(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as f32;
    if e == 15 && (b & 7) == 7 { return f32::NAN; }
    if e == 0 { s * m / 8.0 * 2f32.powi(-6) } else { s * (1.0 + m / 8.0) * 2f32.powi(e - 7) }
}

/// UE8M0 byte -> 2^(b - 127) (0xFF = NaN).
pub fn ue8m0(b: u8) -> f32 {
    if b == 0xFF { return f32::NAN; }
    f32::from_bits(if b == 0 { 1 << 22 } else { (b as u32) << 23 }) // 2^-127 is the subnormal 0x00400000
}

impl Mx8Host {
    /// Row `r` dequantized to f32 (exact: e4m3 * 2^k is representable in f32 for every k in the format).
    pub fn row_f32(&self, r: usize, out: &mut [f32]) {
        let (k, kb) = (self.k, self.k / 32);
        for c in 0..k {
            out[c] = e4m3(self.codes[r * k + c]) * ue8m0(self.scales[r * kb + c / 32]);
        }
    }
}

impl Nv4Host {
    /// Row `r` dequantized to f32: `e2m1 * e4m3 * gs` (the e2m1*e4m3 product is exact in f16/f32;
    /// the gs multiply is one f32 rounding — the kernels apply gs to the f32 accumulator instead).
    pub fn row_f32(&self, r: usize, out: &mut [f32]) {
        let (k, kh, kb) = (self.k, self.k / 2, self.k / 16);
        for c in 0..k {
            let byte = self.packed[r * kh + c / 2];
            let q = if c % 2 == 0 { byte & 0xF } else { byte >> 4 };
            out[c] = E2M1[q as usize] * e4m3(self.scales[r * kb + c / 16]) * self.gs;
        }
    }
    /// Row `r` WITHOUT the global multiplier (`e2m1 * e4m3`, exact in f16).
    pub fn row_unscaled(&self, r: usize, out: &mut [f32]) {
        let (k, kh, kb) = (self.k, self.k / 2, self.k / 16);
        for c in 0..k {
            let byte = self.packed[r * kh + c / 2];
            let q = if c % 2 == 0 { byte & 0xF } else { byte >> 4 };
            out[c] = E2M1[q as usize] * e4m3(self.scales[r * kb + c / 16]);
        }
    }
}

/// The power-of-two shift k that puts every nonzero |v| of `vals` into f16's normal range
/// [2^-14, 65504], so that f16(v * 2^k) is exact for every bf16 v (8 significant bits <= f16's 11);
/// k = 0 when the values already fit; when the range is wider than f16's, the shift that keeps the
/// largest values exact (the loader counts what still rounds).
pub fn exact_f16_shift(vals: &[f32]) -> i32 {
    // For bf16 sources (8 significant bits): v * 2^k is exact in f16 iff its binade e(v) + k lies in [-17, 15]
    // (f16 normals hold 11 bits; a subnormal's step 2^-24 still holds a bf16 value whose binade is >= -17; the
    // largest bf16 below 2^16, 65280, is an f16 value). So a tensor fits one shift iff its binades span <= 32.
    let (mut amax, mut amin) = (0f32, f32::INFINITY);
    for &v in vals { let a = v.abs(); if a > 0.0 && a.is_finite() { amax = amax.max(a); amin = amin.min(a); } }
    if amax == 0.0 { return 0; }
    let (e_max, e_min) = (amax.log2().floor() as i32, amin.log2().floor() as i32);
    let k_hi = 15 - e_max;     // the largest value stays < 2^16
    let k_lo = -17 - e_min;    // the smallest value stays exact
    if k_lo <= 0 && 0 <= k_hi { 0 } else if k_lo <= k_hi { if k_lo > 0 { k_lo } else { k_hi } } else { k_hi }
}

/// True when `v` is exactly representable as an IEEE f16 (normal or subnormal).
pub fn f16_exact(v: f32) -> bool {
    if !v.is_finite() { return false; }
    half::f16::from_f32(v).to_f32() == v
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_f16_shift_holds_every_bf16_binade_within_32() {
        // a tensor spanning binades [-40, -9] (32 binades): one shift makes every bf16 value of it exact in f16
        let mut vals = Vec::new();
        for e in -40..=-9 { for m in [0x80u32, 0xA5, 0xFF] { vals.push(f32::from_bits(((e + 127) as u32) << 23 | (m & 0x7F) << 16)); } }
        let k = exact_f16_shift(&vals);
        let sc = 2f32.powi(k);
        assert!(vals.iter().all(|&v| f16_exact(v * sc)), "shift {k}");
        // already in range: no shift
        assert_eq!(exact_f16_shift(&[1.0, 2f32.powi(-17), 65280.0]), 0);
        // a 34-binade tensor: the largest values stay exact (never overflow), only the bottom loses bits
        let wide = [2f32.powi(10), 2f32.powi(-24)];
        let k = exact_f16_shift(&wide);
        assert!(f16_exact(wide[0] * 2f32.powi(k)) && (wide[0] * 2f32.powi(k)) < 65536.0);
    }
    #[test]
    fn e4m3_table_matches_the_ocp_format() {
        assert_eq!(e4m3(0x7E), 448.0);
        assert_eq!(e4m3(0x01), 2f32.powi(-9));
        assert_eq!(e4m3(0x38), 1.0);
        assert_eq!(e4m3(0xB8), -1.0);
        assert!(e4m3(0x7F).is_nan());
        assert_eq!(e4m3(0x08), 2f32.powi(-6));
        // every finite code is exact in f16 (|v| <= 448, >= 2^-9)
        for b in 0..=255u8 { let v = e4m3(b); if v.is_finite() { assert!(f16_exact(v), "e4m3 {b:#x}"); } }
    }
    #[test]
    fn ue8m0_is_a_power_of_two() {
        assert_eq!(ue8m0(127), 1.0);
        assert_eq!(ue8m0(117), 2f32.powi(-10));
        assert_eq!(ue8m0(0), 2f32.powi(-127));
        assert!(ue8m0(255).is_nan());
    }
    #[test]
    fn e2m1_times_e4m3_is_exact_in_f16() {
        for q in 0..16 { for b in 0..=255u8 { let s = e4m3(b); if s.is_finite() { assert!(f16_exact(E2M1[q] * s)); } } }
    }
}
