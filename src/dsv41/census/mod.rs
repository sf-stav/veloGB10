//! DSV4.1 (leg I0): the pack-generic tensor census.
//!
//! For EVERY tensor of a pack (native FP8 / MXFP4, any EXL3 pack, a mixed pack, a slice) the census
//! records its name, class, role, layer, sub-index, dtype, shape, bytes and the format of the module
//! it belongs to. Formats come from the safetensors headers (dtype + shapes: a trellis' last dim is
//! 16 x K, a scale's shape gives its block) and are cross-checked against
//! `quantization_config.json` `tensor_storage` where the pack has one — never from per-pack constants.
//! A BF16 / F16 / F32 linear inside an EXL3 pack is a first-class `LinearFormat::Plain`.
//!
//! `Census::build` refuses nothing silently: every problem (a name that is no V4.1 tensor, a shape
//! that disagrees with the config, a tensor_storage entry that disagrees with the header, an index
//! key with no tensor, a missing module) is collected in `problems`; `Census::check` fails on any.

pub mod classify;
pub mod report;
#[cfg(test)]
pub(crate) mod synth;
#[cfg(test)]
mod tests;

use crate::dsv41::config::Dsv41Config;
use crate::dsv41::st;
use anyhow::{bail, Context, Result};
use classify::{classify, logical, Logical, Role, TensorClass};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// A linear's (or a parameter's) storage format, derived from its tensors.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LinearFormat {
    /// an unquantized matrix (safetensors dtype BF16 / F16 / F32): the plain-linear path
    Plain(String),
    /// F8_E4M3 values with one scale per `bo` x `bi` block (DeepSeek: 32 x 32, ue8m0)
    Fp8Block { bo: u32, bi: u32, scale: String },
    /// E2M1 values packed two per byte with one scale per `block` inputs (MXFP4: 32 / E8M0;
    /// NVFP4: 16 / E4M3 plus a per-tensor `scale2`)
    Fp4 { block: u32, scale: String, scale2: bool },
    /// EXL3 trellis codes, K bits per weight (from the trellis' last dim / 16)
    Exl3 { k: u8, mul1: bool, suh: bool, svh: bool },
    /// not a linear: a norm weight, a bias, an hc parameter, a sink (the dtype)
    Param(String),
    /// the module's tensors do not form any known format (a census problem)
    Unknown(String),
}

impl LinearFormat {
    pub fn label(&self) -> String {
        match self {
            LinearFormat::Plain(d) => format!("plain {d}"),
            LinearFormat::Fp8Block { bo, bi, scale } => format!("FP8 E4M3 {bo}x{bi} ({})", scale_name(scale)),
            LinearFormat::Fp4 { block, scale, scale2 } => {
                let kind = match (block, scale.as_str(), scale2) { (32, "F8_E8M0", false) => "MXFP4", (16, "F8_E4M3", true) => "NVFP4", _ => "FP4" };
                format!("{kind} E2M1 /{block} ({}{})", scale_name(scale), if *scale2 { " + f32 tensor scale" } else { "" })
            }
            LinearFormat::Exl3 { k, .. } => format!("EXL3 K{k}"),
            LinearFormat::Param(d) => format!("param {d}"),
            LinearFormat::Unknown(why) => format!("UNKNOWN ({why})"),
        }
    }
    /// The weight backend that holds a linear of this format.
    pub fn backend(&self) -> Option<crate::dsv41::weights::BackendKind> {
        use crate::dsv41::weights::BackendKind as B;
        match self {
            LinearFormat::Plain(_) => Some(B::Plain),
            LinearFormat::Fp8Block { .. } | LinearFormat::Fp4 { .. } => Some(B::Native),
            LinearFormat::Exl3 { .. } => Some(B::Exl3),
            _ => None,
        }
    }
}

fn scale_name(s: &str) -> &'static str {
    match s { "F8_E8M0" => "ue8m0 scales", "F8_E4M3" => "e4m3 scales", "F32" => "f32 scales", "BF16" => "bf16 scales", _ => "other scales" }
}

/// One tensor.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    /// index into `Census::files`
    pub file: u32,
    pub class: TensorClass,
    pub role: Role,
    pub layer: Option<u16>,
    pub sub: Option<u16>,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: u64,
    /// index into `Census::modules`
    pub module: u32,
    /// absolute byte offset of the data in its file (the slice extractor copies from here)
    pub offset: u64,
}

impl Entry {
    /// The per-tensor format label (the module's format plus this tensor's role).
    pub fn format_label(&self, m: &Module) -> String {
        match (&m.format, self.role) {
            (LinearFormat::Exl3 { k, .. }, Role::Trellis) => format!("EXL3 K{k} trellis"),
            (LinearFormat::Exl3 { .. }, Role::Suh) => "F16 hadamard in (suh)".into(),
            (LinearFormat::Exl3 { .. }, Role::Svh) => "F16 hadamard out (svh)".into(),
            (LinearFormat::Exl3 { .. }, Role::Mul1) => "I32 codebook mul1".into(),
            (LinearFormat::Fp8Block { bo, bi, .. }, Role::Weight) => format!("FP8 E4M3 {bo}x{bi} values"),
            (LinearFormat::Fp8Block { bo, bi, scale }, Role::Scale) => format!("{} {bo}x{bi} block scales", short_dtype(scale)),
            (LinearFormat::Fp4 { block, .. }, Role::Weight) => format!("E2M1 x2 packed (/{block})"),
            (LinearFormat::Fp4 { block, scale, .. }, Role::Scale) => format!("{} /{block} block scales", short_dtype(scale)),
            (_, Role::Bias) => format!("{} bias", self.dtype),
            (f, _) => match f { LinearFormat::Plain(d) => format!("plain {d}"), LinearFormat::Param(d) => format!("param {d}"), _ => self.dtype.clone() },
        }
    }
}

fn short_dtype(d: &str) -> &str { match d { "F8_E8M0" => "E8M0", "F8_E4M3" => "E4M3", x => x } }

/// One module (all tensors of one linear / parameter share a module).
#[derive(Clone, Debug)]
pub struct Module {
    pub key: String,
    pub class: TensorClass,
    pub layer: Option<u16>,
    pub sub: Option<u16>,
    pub format: LinearFormat,
    pub logical: Option<Logical>,
    pub tensors: Vec<u32>,
    pub bytes: u64,
}

impl Module {
    /// Is this an EXL3 per-group `wo_a.slice.N` module?
    pub fn is_wo_a_slice(&self) -> bool { self.class == TensorClass::AttnWoA && self.sub.is_some() }
}

/// The census of one pack directory.
pub struct Census {
    pub dir: PathBuf,
    pub cfg: Dsv41Config,
    pub files: Vec<String>,
    pub entries: Vec<Entry>,
    pub modules: Vec<Module>,
    /// keys of `model.safetensors.index.json` (0 = no index: a single-file pack)
    pub index_keys: usize,
    pub index_total: Option<u64>,
    /// the tensors that carry a `tensor_storage` entry in quantization_config.json
    pub storage_covered: usize,
    pub problems: Vec<String>,
    pub notes: Vec<String>,
}

fn torch_dtype(t: &str) -> Option<&'static str> {
    Some(match t {
        "torch.float16" => "F16", "torch.bfloat16" => "BF16", "torch.float32" => "F32", "torch.float64" => "F64",
        "torch.int8" => "I8", "torch.uint8" => "U8", "torch.int16" => "I16", "torch.int32" => "I32", "torch.int64" => "I64",
        "torch.float8_e4m3fn" => "F8_E4M3", "torch.float8_e5m2" => "F8_E5M2", "torch.float8_e8m0fnu" => "F8_E8M0", "torch.bool" => "BOOL",
        _ => return None,
    })
}

impl Census {
    /// Read every header of the pack at `dir` and classify, group, derive and validate.
    pub fn build(dir: &Path) -> Result<Census> {
        Self::build_with(dir, &mut |_, _| {})
    }

    /// `build`, reporting (files read, files total) to `progress` (a boot_report::Progress hook).
    pub fn build_with(dir: &Path, progress: &mut dyn FnMut(u64, u64)) -> Result<Census> {
        let cfg = Dsv41Config::load(dir).with_context(|| format!("{}: config", dir.display()))?;
        let mut problems = Vec::new();
        let mut notes = Vec::new();
        // ---- the index (or every *.safetensors when there is none) ----
        let index_path = dir.join("model.safetensors.index.json");
        let (weight_map, index_total): (BTreeMap<String, String>, Option<u64>) = if index_path.exists() {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?).context("index JSON")?;
            let wm = v.get("weight_map").and_then(Value::as_object).context("index has no weight_map")?;
            let total = v.get("metadata").and_then(|m| m.get("total_size")).and_then(Value::as_u64);
            (wm.iter().map(|(k, f)| (k.clone(), f.as_str().unwrap_or_default().to_string())).collect(), total)
        } else {
            (BTreeMap::new(), None)
        };
        let mut files: Vec<String> = if weight_map.is_empty() {
            let mut v: Vec<String> = std::fs::read_dir(dir)?.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_string()).filter(|n| n.ends_with(".safetensors")).collect();
            v.sort();
            if v.is_empty() { bail!("{}: no model.safetensors.index.json and no *.safetensors", dir.display()); }
            notes.push(format!("no index: read {} *.safetensors file(s)", v.len()));
            v
        } else {
            let mut v: Vec<String> = weight_map.values().cloned().collect();
            v.sort(); v.dedup();
            v
        };
        files.retain(|f| !f.is_empty());
        // ---- the headers ----
        let mut entries: Vec<Entry> = Vec::new();
        let mut mkeys: Vec<String> = Vec::new();
        let mut seen: HashMap<String, u32> = HashMap::new();
        let mut unknown = 0usize;
        let nf = files.len() as u64;
        for (fi, f) in files.iter().enumerate() {
            let h = st::read_header(&dir.join(f))?;
            for t in h.tensors {
                if let Some(prev) = seen.insert(t.name.clone(), fi as u32) {
                    problems.push(format!("tensor {} is in {} and {}", t.name, files[prev as usize], f));
                }
                let bytes = t.bytes();
                let Some(c) = classify(&t.name, cfg.layers) else {
                    unknown += 1;
                    if unknown <= 20 { problems.push(format!("{}: not a V4.1 tensor name ({} {:?})", t.name, t.dtype, t.shape)); }
                    continue;
                };
                entries.push(Entry { name: t.name, file: fi as u32, class: c.class, role: c.role, layer: c.layer, sub: c.sub,
                                     dtype: t.dtype, shape: t.shape, bytes, module: u32::MAX, offset: h.data_start + t.begin });
                mkeys.push(c.module);
            }
            progress(fi as u64 + 1, nf);
        }
        if unknown > 20 { problems.push(format!("... {} more names that are not V4.1 tensors", unknown - 20)); }
        // ---- index exactness ----
        if !weight_map.is_empty() {
            let mut missing = 0usize;
            for (k, f) in &weight_map {
                match seen.get(k) {
                    None => { missing += 1; if missing <= 10 { problems.push(format!("index key {k} has no tensor in {f}")); } }
                    Some(&fi) if &files[fi as usize] != f => problems.push(format!("index says {k} is in {f}, it is in {}", files[fi as usize])),
                    _ => {}
                }
            }
            if missing > 10 { problems.push(format!("... {} more index keys without a tensor", missing - 10)); }
            let extra = seen.keys().filter(|k| !weight_map.contains_key(*k)).count();
            if extra > 0 { problems.push(format!("{extra} tensors in the files are not in the index")); }
        }
        let total: u64 = entries.iter().map(|e| e.bytes).sum();
        if let Some(t) = index_total {
            let all_bytes = total; // unknown tensors are problems already; compare what was read
            if t != all_bytes && unknown == 0 { problems.push(format!("index total_size {t} != header bytes {all_bytes}")); }
        }
        // ---- modules ----
        let mut by_key: BTreeMap<String, Vec<u32>> = BTreeMap::new();
        for (i, k) in mkeys.into_iter().enumerate() { by_key.entry(k).or_default().push(i as u32); }
        let mut modules = Vec::with_capacity(by_key.len());
        for (key, idx) in by_key {
            let e0 = &entries[idx[0] as usize];
            let (class, layer, sub) = (e0.class, e0.layer, e0.sub);
            if idx.iter().any(|&i| { let e = &entries[i as usize]; e.class != class || e.layer != layer || e.sub != sub }) {
                problems.push(format!("{key}: tensors of one module disagree on class / layer"));
            }
            let mi = modules.len() as u32;
            for &i in &idx { entries[i as usize].module = mi; }
            let lg = logical(class, &cfg, layer, class == TensorClass::AttnWoA && sub.is_some());
            let (format, mut probs) = derive_format(&key, class, &idx.iter().map(|&i| &entries[i as usize]).collect::<Vec<_>>(), lg);
            problems.append(&mut probs);
            let bytes = idx.iter().map(|&i| entries[i as usize].bytes).sum();
            modules.push(Module { key, class, layer, sub, format, logical: lg, tensors: idx, bytes });
        }
        let mut c = Census { dir: dir.to_path_buf(), cfg, files, entries, modules, index_keys: weight_map.len(), index_total,
                             storage_covered: 0, problems, notes };
        c.check_tensor_storage()?;
        c.check_completeness();
        if let Err(e) = c.cfg.validate() { c.problems.push(format!("{e:#}")); }
        for k in &c.cfg.unknown_keys { c.notes.push(format!("config key not known to the parser: {k}")); }
        Ok(c)
    }

    /// Fail if anything is wrong (the census equals the index, every tensor is understood).
    pub fn check(&self) -> Result<()> {
        if self.problems.is_empty() { Ok(()) } else { bail!("{}: {} census problem(s):\n  {}", self.dir.display(), self.problems.len(), self.problems.join("\n  ")) }
    }

    pub fn total_bytes(&self) -> u64 { self.entries.iter().map(|e| e.bytes).sum() }

    pub fn module_of(&self, e: &Entry) -> &Module { &self.modules[e.module as usize] }

    /// The layers this pack carries (all slots for a full pack; a slice's kept list).
    pub fn kept_layers(&self) -> Vec<usize> {
        match &self.cfg.slice_layers { Some(v) => v.clone(), None => (0..self.cfg.total_layers()).collect() }
    }

    /// Cross-check `quantization_config.json` `tensor_storage` against the headers: every stored
    /// tensor's dtype / shape / bytes, and an EXL3 module's `bits_per_weight` against K from the trellis.
    fn check_tensor_storage(&mut self) -> Result<()> {
        let p = self.dir.join("quantization_config.json");
        if !p.exists() { return Ok(()); }
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&p)?).context("quantization_config.json")?;
        let Some(ts) = v.get("tensor_storage").and_then(Value::as_object) else {
            self.notes.push("quantization_config.json has no tensor_storage: formats come from the headers alone".into());
            return Ok(());
        };
        let by_name: HashMap<&str, usize> = self.entries.iter().enumerate().map(|(i, e)| (e.name.as_str(), i)).collect();
        let kept = self.cfg.slice_layers.is_some();
        let mut covered = 0usize;
        let mut empty = 0usize;
        let mut problems = Vec::new();
        for (mkey, ent) in ts {
            let stored = ent.get("stored_tensors").and_then(Value::as_object);
            let Some(stored) = stored.filter(|s| !s.is_empty()) else { empty += 1; continue };
            let mut module: Option<u32> = None;
            for (tname, td) in stored {
                let Some(&i) = by_name.get(tname.as_str()) else {
                    if !kept { problems.push(format!("tensor_storage {mkey}: {tname} is not in the pack")); }
                    continue;
                };
                covered += 1;
                let e = &self.entries[i];
                module = Some(e.module);
                let tdt = td.get("dtype").and_then(Value::as_str).and_then(torch_dtype);
                let tshape: Option<Vec<u64>> = td.get("shape").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).collect());
                let tbytes = td.get("n_bytes").and_then(Value::as_u64);
                if tdt != Some(e.dtype.as_str()) || tshape.as_ref() != Some(&e.shape) || tbytes != Some(e.bytes) {
                    problems.push(format!("tensor_storage {tname}: {:?} {:?} {:?} vs header {} {:?} {}", tdt, tshape, tbytes, e.dtype, e.shape, e.bytes));
                }
            }
            if let (Some(mi), Some(bits)) = (module, ent.get("bits_per_weight").and_then(Value::as_f64)) {
                let m = &self.modules[mi as usize];
                match &m.format {
                    LinearFormat::Exl3 { k, .. } if (*k as f64 - bits).abs() < 1e-9 => {}
                    f => problems.push(format!("tensor_storage {mkey}: bits_per_weight {bits} vs header-derived {}", f.label())),
                }
                if let Some(qf) = ent.get("quant_format").and_then(Value::as_str) {
                    if qf != "exl3" || !matches!(m.format, LinearFormat::Exl3 { .. }) {
                        problems.push(format!("tensor_storage {mkey}: quant_format {qf} vs header-derived {}", m.format.label()));
                    }
                }
            }
        }
        self.storage_covered = covered;
        self.notes.push(format!("tensor_storage: {} entries, {} stored tensors cross-checked against the headers ({} entries store nothing); {} tensors have no entry (format from the header alone)",
                                ts.len(), covered, empty, self.entries.len() - covered));
        self.problems.append(&mut problems);
        Ok(())
    }

    /// Every module the config says a kept layer must have (plus the globals) is present once.
    fn check_completeness(&mut self) {
        let cfg = &self.cfg;
        let mut have: HashMap<(TensorClass, Option<u16>, Option<u16>), usize> = HashMap::new();
        for m in &self.modules { *have.entry((m.class, m.layer, m.sub)).or_default() += 1; }
        let mut missing = Vec::new();
        let mut need = |c: TensorClass, l: Option<u16>, s: Option<u16>| {
            if !have.contains_key(&(c, l, s)) { missing.push(format!("{} layer {l:?} sub {s:?}", c.name())); }
        };
        use TensorClass as C;
        let vision = cfg.vision.is_some();
        for l in self.kept_layers() {
            let lu = Some(l as u16);
            for c in [C::AttnSink, C::AttnNorm, C::AttnQNorm, C::AttnKvNorm, C::AttnWqA, C::AttnWqB, C::AttnWkv, C::AttnWoB, C::FfnNorm,
                      C::HcAttnFn, C::HcAttnBase, C::HcAttnScale, C::HcFfnFn, C::HcFfnBase, C::HcFfnScale, C::GateWeight, C::GateBias,
                      C::SharedW1, C::SharedW2, C::SharedW3] { need(c, lu, None); }
            if vision { need(C::GateBiasVl, lu, None); }
            if !have.contains_key(&(C::AttnWoA, lu, None)) { for g in 0..cfg.o_groups { need(C::AttnWoA, lu, Some(g as u16)); } }
            let (ne, _) = cfg.moe_of(l);
            for e in 0..ne { for c in [C::ExpertW1, C::ExpertW2, C::ExpertW3] { need(c, lu, Some(e as u16)); } }
            if cfg.kv_sources.contains(&l) {
                for c in [C::CompWkv, C::CompNorm, C::IdxWk, C::IdxKNorm] { need(c, lu, None); }
                if cfg.ratio(l) > 1 { need(C::CompWgate, lu, None); }
            }
            if cfg.index_sources.contains(&l) { need(C::IdxWqB, lu, None); need(C::IdxWeightsProj, lu, None); }
            if cfg.engram_layers.contains(&l) { for c in [C::EngramWkv, C::EngramQ, C::EngramK] { need(c, lu, None); } }
            if cfg.is_dspark(l) {
                if l == cfg.layers { need(C::DsMainProj, lu, None); need(C::DsMainNorm, lu, None); }
                if l + 1 == cfg.total_layers() { for c in [C::DsNorm, C::DsMarkovEmbed, C::DsMarkovHead, C::DsConfidence] { need(c, lu, None); } }
            }
        }
        for c in [C::Embed, C::Head, C::FinalNorm] { need(c, None, None); }
        let slice_vision = self.cfg.slice_layers.is_none() || self.modules.iter().any(|m| m.class.group() == "vision");
        if vision && slice_vision {
            for c in [C::VisPatchEmbed, C::VisNorm, C::AlignerW1, C::AlignerW2] { need(c, None, None); }
            let nb = cfg.vision.as_ref().map_or(0, |v| v.layers);
            for b in 0..nb { for c in [C::VisNorm, C::VisQkv, C::VisWo, C::VisMlpW1, C::VisMlpW2] { need(c, None, Some(b as u16)); } }
        }
        let n_missing = missing.len();
        for m in missing.into_iter().take(20) { self.problems.push(format!("missing module: {m}")); }
        if n_missing > 20 { self.problems.push(format!("... {} more missing modules", n_missing - 20)); }
        let tables = self.modules.iter().filter(|m| m.class == C::EngramTable).count();
        let want_tables = self.kept_layers().iter().filter(|l| cfg.engram_layers.contains(l)).count();
        if tables == 0 && want_tables > 0 {
            self.notes.push("Engram tables are NOT in this pack: every pack uses DeepSeek's FP8 tables from the native checkpoint's shards (owner 2026-11-18)".into());
        } else if tables != want_tables {
            self.notes.push(format!("Engram tables: {tables} in the pack, {want_tables} Engram layers kept"));
        }
        let extra: Vec<String> = self.modules.iter().filter(|m| m.layer.map_or(false, |l| !self.kept_layers().contains(&(l as usize))))
            .map(|m| m.key.clone()).take(5).collect();
        if !extra.is_empty() { self.problems.push(format!("modules of layers this slice does not list: {extra:?}")); }
    }

    /// The per-class sums `(tensors, modules, bytes)` keyed by (class, module format label).
    pub fn by_class_format(&self) -> BTreeMap<(TensorClass, String), (usize, usize, u64)> {
        let mut out: BTreeMap<(TensorClass, String), (usize, usize, u64)> = BTreeMap::new();
        for m in &self.modules {
            let r = out.entry((m.class, m.format.label())).or_default();
            r.0 += m.tensors.len();
            r.1 += 1;
            r.2 += m.bytes;
        }
        out
    }

    /// The per-tensor-format sums `(tensors, bytes)` keyed by the tensor format label.
    pub fn by_tensor_format(&self) -> BTreeMap<String, (usize, u64)> {
        let mut out: BTreeMap<String, (usize, u64)> = BTreeMap::new();
        for e in &self.entries {
            let r = out.entry(e.format_label(self.module_of(e))).or_default();
            r.0 += 1;
            r.1 += e.bytes;
        }
        out
    }

    /// A short weights line for the boot configuration block: the formats and their share of bytes.
    pub fn weights_line(&self) -> String {
        let total = self.total_bytes() as f64;
        let mut per: BTreeMap<String, u64> = BTreeMap::new();
        let mut lin_weights = 0f64;
        let mut lin_bits = 0f64;
        for m in &self.modules {
            if !m.class.is_linear() { continue; }
            let label = match &m.format { LinearFormat::Exl3 { .. } => "EXL3".to_string(), f => f.label() };
            *per.entry(label).or_default() += m.bytes;
            if let Some(Logical::Linear { out, inn }) = m.logical {
                let n = (out * inn) as f64;
                let n = if m.class == TensorClass::EngramTable { 0.0 } else { n };
                lin_weights += n;
                lin_bits += if n > 0.0 { m.bytes as f64 * 8.0 } else { 0.0 };
            }
        }
        let mut parts: Vec<(u64, String)> = per.into_iter().map(|(k, b)| (b, k)).collect();
        parts.sort_by(|a, b| b.cmp(a));
        let list: Vec<String> = parts.iter().map(|(b, k)| format!("{k} {:.1}%", *b as f64 * 100.0 / total)).collect();
        format!("{:.1} GB, {:.2} bits per linear weight (Engram tables excluded); {}", total / 1e9,
                if lin_weights > 0.0 { lin_bits / lin_weights } else { 0.0 }, list.join(", "))
    }
}

/// Derive a module's format from its tensors and check them against the logical shape.
fn derive_format(key: &str, class: TensorClass, ts: &[&Entry], lg: Option<Logical>) -> (LinearFormat, Vec<String>) {
    let mut p = Vec::new();
    let get = |r: Role| ts.iter().find(|e| e.role == r).copied();
    let roles: Vec<Role> = ts.iter().map(|e| e.role).collect();
    for (i, r) in roles.iter().enumerate() { if roles[..i].contains(r) { p.push(format!("{key}: role {} twice", r.name())); } }
    let Some(lg) = lg else { return (LinearFormat::Unknown("no logical shape".into()), vec![format!("{key}: class {} has no logical shape for this config", class.name())]) };
    let want_bias = |p: &mut Vec<String>, out: u64| if let Some(b) = get(Role::Bias) { if b.shape != [out] { p.push(format!("{key}: bias {:?} != [{out}]", b.shape)); } };
    // non-linear parameters: one tensor of the logical shape
    if !class.is_linear() {
        let e = ts[0];
        let want: Vec<u64> = match lg { Logical::Vector(n) => vec![n], Logical::Matrix(a, b) => vec![a, b], Logical::Scalar => vec![], Logical::Linear { out, inn } => vec![out, inn] };
        if ts.len() != 1 { p.push(format!("{key}: {} tensors for a parameter", ts.len())); }
        if e.shape != want { p.push(format!("{key}: shape {:?} != config {:?}", e.shape, want)); }
        if !matches!(e.dtype.as_str(), "BF16" | "F16" | "F32") { p.push(format!("{key}: parameter dtype {}", e.dtype)); }
        return (LinearFormat::Param(e.dtype.clone()), p);
    }
    let Logical::Linear { out, inn } = lg else { return (LinearFormat::Unknown("linear without [out, in]".into()), p) };
    if let Some(tr) = get(Role::Trellis) {
        // EXL3: trellis [in/16, out/16, 16K] I16, suh [in] F16, svh [out] F16, mul1 [] I32
        let k = tr.shape.get(2).copied().unwrap_or(0) / 16;
        if tr.dtype != "I16" || tr.shape.len() != 3 || tr.shape[0] * 16 != inn || tr.shape[1] * 16 != out || tr.shape[2] % 16 != 0 || !(1..=8).contains(&k) {
            p.push(format!("{key}: trellis {} {:?} does not fit [in/16={}, out/16={}, 16K]", tr.dtype, tr.shape, inn / 16, out / 16));
        }
        let suh = get(Role::Suh);
        let svh = get(Role::Svh);
        let mul1 = get(Role::Mul1);
        if let Some(s) = suh { if s.shape != [inn] || s.dtype != "F16" { p.push(format!("{key}: suh {} {:?} != F16 [{inn}]", s.dtype, s.shape)); } }
        if let Some(s) = svh { if s.shape != [out] || s.dtype != "F16" { p.push(format!("{key}: svh {} {:?} != F16 [{out}]", s.dtype, s.shape)); } }
        if let Some(m) = mul1 { if !m.shape.is_empty() || m.dtype != "I32" { p.push(format!("{key}: mul1 {} {:?} != I32 []", m.dtype, m.shape)); } }
        for e in ts { if !matches!(e.role, Role::Trellis | Role::Suh | Role::Svh | Role::Mul1 | Role::Bias) { p.push(format!("{key}: unexpected {} next to a trellis", e.name)); } }
        want_bias(&mut p, out);
        return (LinearFormat::Exl3 { k: k as u8, mul1: mul1.is_some(), suh: suh.is_some(), svh: svh.is_some() }, p);
    }
    let Some(w) = get(Role::Weight) else { return (LinearFormat::Unknown("no weight / trellis".into()), vec![format!("{key}: no weight or trellis tensor")]) };
    want_bias(&mut p, out);
    let scale = get(Role::Scale);
    match (w.dtype.as_str(), scale) {
        ("BF16" | "F16" | "F32", None) => {
            if w.shape != [out, inn] { p.push(format!("{key}: plain weight {:?} != [{out}, {inn}]", w.shape)); }
            (LinearFormat::Plain(w.dtype.clone()), p)
        }
        ("F8_E4M3", Some(s)) => {
            let ok = w.shape == [out, inn] && s.shape.len() == 2 && s.shape[0] > 0 && s.shape[1] > 0;
            let (bo, bi) = if ok { (out.div_ceil(s.shape[0]), inn.div_ceil(s.shape[1])) } else { (0, 0) };
            if !ok || out.div_ceil(bo.max(1)) != s.shape[0] || inn.div_ceil(bi.max(1)) != s.shape[1] {
                p.push(format!("{key}: FP8 weight {:?} / scale {:?} do not form blocks of [{out}, {inn}]", w.shape, s.shape));
            }
            (LinearFormat::Fp8Block { bo: bo as u32, bi: bi as u32, scale: s.dtype.clone() }, p)
        }
        ("I8" | "U8", Some(s)) => {
            let ok = w.shape == [out, inn / 2] && s.shape.len() == 2 && s.shape[0] == out && s.shape[1] > 0 && inn % s.shape[1] == 0;
            let block = if ok { inn / s.shape[1] } else { 0 };
            if !ok { p.push(format!("{key}: FP4 weight {:?} / scale {:?} do not fit [{out}, {inn}/2]", w.shape, s.shape)); }
            let s2 = get(Role::Scale2).is_some();
            (LinearFormat::Fp4 { block: block as u32, scale: s.dtype.clone(), scale2: s2 }, p)
        }
        (d, s) => {
            let why = format!("weight {d} with{} scale", if s.is_some() { "" } else { "out" });
            p.push(format!("{key}: {why} is no known format"));
            (LinearFormat::Unknown(why), p)
        }
    }
}
