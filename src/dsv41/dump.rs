//! DSV4.1 (leg I0): the tensor-dump format shared with the Python reference
//! (`tools/dsv41_ref/dumpfmt.py` implements the same format; both sides are pinned by tests).
//!
//! A dump is a DIRECTORY:
//! * `dump.json` — the sidecar: `{"format": "dsv41-dump/1", "producer", "model_dir", "pack",
//!   "mode": "exact" | "ced", "prompt_ids": [..], "positions": [start, end), "dtype_notes": {point: text},
//!   "points": [{"file", "layer", "point", "dtype", "shape"}], "extra": {..}}`;
//! * one safetensors file per point, named `L{layer}.{point}.safetensors`, holding ONE tensor whose
//!   name is the point. `layer` is the layer slot 0..=42 (DSpark stages 40..42) or `emb` (before
//!   layer 0) or `head` (after the last layer). Rows are the dumped positions in order.
//!
//! The standard point names (`POINTS`) are what the reference forward and every oracle / GPU leg
//! use, so dumps from different producers line up by file name. Other names are allowed
//! (`[a-z0-9_]+`) and must be described in `dtype_notes`.

use crate::dsv41::st::{self, StOut, StSource};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::path::{Path, PathBuf};

pub const FORMAT: &str = "dsv41-dump/1";

/// The standard dump points: (layer kind, name, shape, meaning).
pub const POINTS: &[(&str, &str, &str, &str)] = &[
    ("emb", "x", "[S, D]", "token embedding (before the hc expansion)"),
    ("emb", "engram_ids", "[S, n_engram_layers, 24] i64", "Engram hash row ids (engram.py NgramHashState)"),
    ("L", "hc_in", "[S, hc, D]", "the block's input stream (after this layer's Engram, if any)"),
    ("L", "engram_out", "[S, hc, D]", "the stream after the Engram module (Engram layers)"),
    ("L", "attn_pre", "[S, hc]", "hc_pre mix used by the attention"),
    ("L", "attn_post", "[S, hc]", "attention hc_post coefficients"),
    ("L", "attn_comb", "[S, hc, hc]", "attention hc_post comb (Sinkhorn)"),
    ("L", "attn_x", "[S, D]", "attention input after hc_pre + attn_norm"),
    ("L", "qr", "[S, q_lora]", "q_norm(wq_a(x))"),
    ("L", "q", "[S, H, head_dim]", "queries after RoPE"),
    ("L", "kv", "[S, head_dim]", "window KV after kv_norm, RoPE and the fp8 quant-dequant"),
    ("L", "comp_kv", "[Sc, head_dim]", "compressed latents written this forward (after RoPE + fp4 quant-dequant)"),
    ("L", "index_k", "[Sc, index_head_dim]", "index keys written this forward"),
    ("L", "index_sel", "[S, index_topk] i32", "selected compressed positions (-1 = none), shifted by the window"),
    ("L", "cand_blocks", "[S, candidate_topk_blocks] i32", "the candidate source's kept blocks (layer 20)"),
    ("L", "attn_o", "[S, H, head_dim]", "sparse attention output after the inverse RoPE"),
    ("L", "attn_out", "[S, D]", "wo_b output"),
    ("L", "hc_mid", "[S, hc, D]", "the stream after the attention hc_post"),
    ("L", "ffn_x", "[S, D]", "MoE input after hc_pre + ffn_norm"),
    ("L", "router_ids", "[S, top_k] i32", "selected experts"),
    ("L", "router_w", "[S, top_k] f32", "their weights (renormalized x routed_scaling_factor)"),
    ("L", "ffn_out", "[S, D]", "MoE output (routed + shared)"),
    ("L", "hc_out", "[S, hc, D]", "the block's output stream"),
    ("head", "x", "[S, D]", "hc_pre + final norm"),
    ("head", "logits", "[S, V] f32", "FP32 logits"),
    ("head", "main_hidden", "[S, targets x D]", "DSpark conditioning: stream means at the target layers' inputs"),
    ("L", "draft_ids", "[S, block + 1] i64", "DSpark output ids (last stage)"),
    ("L", "draft_logits", "[S, block, V] f32", "DSpark draft logits (last stage)"),
    ("L", "confidence", "[S, block] f32", "DSpark confidence head (last stage)"),
];

/// The layer part of a file name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
/// `Global` = REFb's per-prompt points (`G.{point}`); I0 dumps never write it.
pub enum DumpLayer { Emb, Layer(u16), Head, Global }

impl DumpLayer {
    pub fn tag(self) -> String { match self { DumpLayer::Emb => "emb".into(), DumpLayer::Layer(l) => l.to_string(), DumpLayer::Head => "head".into(), DumpLayer::Global => "G".into() } }
    pub fn parse(s: &str) -> Option<DumpLayer> {
        match s { "emb" => Some(DumpLayer::Emb), "head" => Some(DumpLayer::Head), "G" => Some(DumpLayer::Global), x => x.parse().ok().map(DumpLayer::Layer) }
    }
    fn json(self) -> Value { match self { DumpLayer::Layer(l) => json!(l), x => json!(x.tag()) } }
}

pub fn file_name(layer: DumpLayer, point: &str) -> String { format!("L{}.{point}.safetensors", layer.tag()) }

fn point_ok(p: &str) -> bool { !p.is_empty() && p.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_') }

/// One dumped tensor, read back.
#[derive(Clone, Debug, PartialEq)]
pub struct DumpTensor { pub layer: DumpLayer, pub point: String, pub dtype: String, pub shape: Vec<u64>, pub bytes: Vec<u8> }

impl DumpTensor {
    /// The values as f32 (floating dtypes F32 / F16 / BF16 / F64 and integer dtypes).
    pub fn to_f32(&self) -> Result<Vec<f32>> { bytes_to_f32(&self.dtype, &self.bytes).with_context(|| self.point.clone()) }
}

/// Raw little-endian tensor bytes of a safetensors dtype as f32 (floating and integer dtypes).
pub fn bytes_to_f32(dtype: &str, b: &[u8]) -> Result<Vec<f32>> {
    Ok(match dtype {
        "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "F64" => b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
        "BF16" => b.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        "F16" => b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        "I32" => b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32).collect(),
        "I64" => b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
        "U8" | "BOOL" => b.iter().map(|&x| x as f32).collect(),
        d => bail!("dtype {d} has no f32 view"),
    })
}

/// Writes a dump directory: `put` each point, then `finish` writes the sidecar.
pub struct DumpWriter { dir: PathBuf, meta: Value, points: Vec<Value>, notes: BTreeMap<String, String> }

impl DumpWriter {
    /// `mode` = "exact" | "ced".
    pub fn create(dir: &Path, producer: &str, model_dir: &str, pack: &str, mode: &str, prompt_ids: &[u32], positions: (u64, u64)) -> Result<Self> {
        ensure!(mode == "exact" || mode == "ced", "dump mode must be exact or ced (got {mode})");
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        let meta = json!({"format": FORMAT, "producer": producer, "model_dir": model_dir, "pack": pack, "mode": mode,
                          "prompt_ids": prompt_ids, "positions": [positions.0, positions.1], "extra": {}});
        Ok(DumpWriter { dir: dir.to_path_buf(), meta, points: Vec::new(), notes: BTreeMap::new() })
    }
    pub fn note(&mut self, point: &str, text: &str) { self.notes.insert(point.to_string(), text.to_string()); }
    pub fn extra(&mut self, key: &str, v: Value) { self.meta["extra"][key] = v; }
    /// Write one point (raw little-endian bytes of `dtype` x `shape`).
    pub fn put(&mut self, layer: DumpLayer, point: &str, dtype: &str, shape: &[u64], bytes: &[u8]) -> Result<()> {
        ensure!(point_ok(point), "dump point name {point:?} must be [a-z0-9_]+");
        let f = file_name(layer, point);
        ensure!(!self.points.iter().any(|p| p["file"] == f.as_str()), "dump point {f} written twice");
        let t = [StOut { name: point.to_string(), dtype: dtype.to_string(), shape: shape.to_vec(), src: StSource::Bytes(bytes) }];
        st::write_file(&self.dir.join(&f), &t, &BTreeMap::new(), &mut |_| {})?;
        self.points.push(json!({"file": f, "layer": layer.json(), "point": point, "dtype": dtype, "shape": shape}));
        Ok(())
    }
    pub fn put_f32(&mut self, layer: DumpLayer, point: &str, shape: &[u64], v: &[f32]) -> Result<()> {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.put(layer, point, "F32", shape, &b)
    }
    pub fn finish(mut self) -> Result<PathBuf> {
        self.meta["points"] = Value::Array(self.points);
        self.meta["dtype_notes"] = serde_json::to_value(&self.notes)?;
        let p = self.dir.join("dump.json");
        std::fs::write(&p, serde_json::to_string_pretty(&self.meta)?)?;
        Ok(p)
    }
}

// ------------------------------------------------------------------------------------------------ REFb layout

/// One REFb tensor, read back (floats widened to f32, integers to i64; `rows` = the `.rows` companion).
#[derive(Clone, Debug)]
pub struct RefTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// float dtypes (F32 / BF16 / F16) widened to f32; empty for integer tensors
    pub f: Vec<f32>,
    /// integer dtypes (I64 / I32) widened to i64; empty for float tensors
    pub i: Vec<i64>,
    /// absolute positions of the first dimension (`.rows` companion), when dumped
    pub rows: Option<Vec<i64>>,
}

impl RefTensor {
    /// Elements per first-dimension row.
    pub fn stride(&self) -> usize { self.shape.iter().skip(1).product::<usize>().max(1) }
    pub fn n_rows(&self) -> usize { self.shape.first().copied().unwrap_or(0) }
    /// Row `r` (by index) of a float tensor.
    pub fn row_f(&self, r: usize) -> &[f32] { let s = self.stride(); &self.f[r * s..(r + 1) * s] }
    /// Row `r` (by index) of an integer tensor.
    pub fn row_i(&self, r: usize) -> &[i64] { let s = self.stride(); &self.i[r * s..(r + 1) * s] }
    /// The index of absolute position `pos` (the `.rows` companion; without one, index = position).
    pub fn index_of(&self, pos: i64) -> Option<usize> {
        match &self.rows {
            Some(r) => r.binary_search(&pos).ok().or_else(|| r.iter().position(|&p| p == pos)),
            None => (pos >= 0 && (pos as usize) < self.n_rows()).then_some(pos as usize),
        }
    }
    /// Absolute positions of every row.
    pub fn positions(&self) -> Vec<i64> { self.rows.clone().unwrap_or_else(|| (0..self.n_rows() as i64).collect()) }
}

/// The one reader of REFb's dump layout (`tools/dsv41_ref/refdump.py`; written before I0's one-file-per-point
/// format landed, so it differs): one directory per prompt and prefill variant with `meta.json` (prompt length `P`,
/// mode, row subset, and a `tensors` index name -> file / dtype / shape), `L{LL}.safetensors` per layer holding
/// `L{LL}.{point}` and its `L{LL}.{point}.rows` companion (int64 absolute positions of the first dimension), and
/// `G.safetensors` for per-prompt points; the fixture's `../outputs.safetensors` (`G.logits`, decode steps, DSpark
/// drafts) is indexed too. A dump still being written (no `meta.json` yet) is indexed from the file headers.
/// OM, OA and SL read through this type (`cpu::mhc::refdump` and `cpu::attn_fixture::RefDump` are thin wrappers);
/// [`Dump::open`] also opens a REFb directory.
pub struct RefbDump {
    pub dir: PathBuf,
    /// `meta.json` (Null while REFb is still writing the dump)
    pub meta: Value,
    /// prompt length (0 when `meta.json` is not written yet)
    pub p: usize,
    /// tensor name -> file
    index: HashMap<String, String>,
    headers: Mutex<HashMap<String, st::StHeader>>,
}

impl RefbDump {
    /// Is `dir` laid out as a REFb dump (a variant directory or a fixture directory holding `prefill_exact/`)?
    pub fn is_refb(dir: &Path) -> bool {
        let d = Self::variant_dir(dir);
        d.join("meta.json").exists() || d.join("L00.safetensors").exists()
    }

    /// A fixture directory holding `prefill_exact/` resolves to that variant; anything else is itself. The fixture
    /// directory's own `meta.json` (the REFb summary: prompt, outputs, teacher-forced stats, no `tensors` index) does
    /// not stop the descent: only a directory that itself holds layer files is already a variant directory.
    pub fn variant_dir(dir: &Path) -> PathBuf {
        if dir.join("prefill_exact").is_dir() && !dir.join("L00.safetensors").exists() {
            dir.join("prefill_exact")
        } else {
            dir.to_path_buf()
        }
    }

    /// The fixture's name for a `--dsv41-dump-ref` path: the fixture directory (`.../agent_4k`), whether the path names
    /// it or its `prefill_exact` variant (so OD's trace directories, "named like the fixture", match either way).
    pub fn fixture_name(path: &Path) -> String {
        let s = path.to_string_lossy();
        let pd = Path::new(s.trim_end_matches('/'));
        let leaf = if pd.file_name().map_or(false, |x| x == "prefill_exact") { pd.parent().unwrap_or(pd) } else { pd };
        leaf.file_name().map(|x| x.to_string_lossy().to_string()).unwrap_or_else(|| s.to_string())
    }

    /// Open a variant directory (`.../short_chat/prefill_exact`) or a fixture directory holding `prefill_exact/`.
    pub fn open(dir: &Path) -> Result<RefbDump> {
        let dir = Self::variant_dir(dir);
        let mp = dir.join("meta.json");
        let mut index = HashMap::new();
        let mut headers = HashMap::new();
        let (meta, p) = if mp.exists() {
            let meta: Value = serde_json::from_slice(&std::fs::read(&mp).with_context(|| format!("read {}", mp.display()))?)
                .with_context(|| format!("parse {}", mp.display()))?;
            let t = meta.get("tensors").and_then(Value::as_object).with_context(|| format!("{}: no tensors index (not a REFb dump?)", mp.display()))?;
            for (k, v) in t {
                if let Some(f) = v.get("file").and_then(Value::as_str) {
                    index.insert(k.clone(), f.to_string());
                }
            }
            let p = meta.get("P").and_then(Value::as_u64).context("meta.json: no P")? as usize;
            (meta, p)
        } else {
            // a dump still being written: index the layer files already flushed
            let mut files: Vec<String> = std::fs::read_dir(&dir)
                .with_context(|| format!("read_dir {}", dir.display()))?
                .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().to_string()))
                .filter(|n| n.ends_with(".safetensors"))
                .collect();
            files.sort();
            for f in files {
                let Ok(h) = st::read_header(&dir.join(&f)) else { continue }; // a file being written
                for t in &h.tensors {
                    index.insert(t.name.clone(), f.clone());
                }
                headers.insert(f, h);
            }
            ensure!(!index.is_empty(), "{}: no meta.json and no readable layer files", dir.display());
            (Value::Null, 0)
        };
        // the fixture's outputs (G.logits, G.decode_*, the DSpark drafts) sit one level up
        let outs = dir.join("..").join("outputs.safetensors");
        if outs.exists() {
            let h = st::read_header(&outs)?;
            for t in &h.tensors {
                index.entry(t.name.clone()).or_insert_with(|| "../outputs.safetensors".to_string());
            }
            headers.insert("../outputs.safetensors".to_string(), h);
        }
        Ok(RefbDump { dir, meta, p, index, headers: Mutex::new(headers) })
    }

    /// Prompt length (an error while `meta.json` is not written yet).
    pub fn prompt_len(&self) -> Result<usize> { self.meta.get("P").and_then(Value::as_u64).map(|v| v as usize).context("meta.json has no P") }
    pub fn mode(&self) -> &str { self.meta.get("mode").and_then(Value::as_str).unwrap_or("?") }
    pub fn name(&self) -> &str { self.meta.get("name").and_then(Value::as_str).unwrap_or("?") }

    /// The key of a per-layer point (`L07.x_in`).
    pub fn key(layer: usize, point: &str) -> String { format!("L{layer:02}.{point}") }

    /// Is the tensor in the dump's index?
    pub fn has(&self, key: &str) -> bool { self.index.contains_key(key) }

    /// Every indexed tensor name (including the `.rows` companions), sorted.
    pub fn keys(&self) -> Vec<String> { let mut k: Vec<String> = self.index.keys().cloned().collect(); k.sort(); k }

    fn header(&self, file: &str) -> Result<st::StHeader> {
        let mut h = self.headers.lock().map_err(|_| anyhow::anyhow!("header cache poisoned"))?;
        if let Some(x) = h.get(file) {
            return Ok(x.clone());
        }
        let x = st::read_header(&self.dir.join(file))?;
        h.insert(file.to_string(), x.clone());
        Ok(x)
    }

    /// (safetensors dtype, shape, bytes) of a tensor.
    pub fn raw(&self, key: &str) -> Result<(String, Vec<usize>, Vec<u8>)> {
        let file = self.index.get(key).with_context(|| format!("{}: no tensor {key} in the dump", self.dir.display()))?;
        let h = self.header(file)?;
        let t = h.tensors.iter().find(|t| t.name == key).with_context(|| format!("{file}: no tensor {key}"))?;
        Ok((t.dtype.clone(), t.shape.iter().map(|&x| x as usize).collect(), st::read_tensor(&h, key)?))
    }

    /// (dtype, shape) from the header only.
    pub fn describe(&self, key: &str) -> Result<(String, Vec<u64>)> {
        let file = self.index.get(key).with_context(|| format!("{}: no tensor {key} in the dump", self.dir.display()))?;
        let h = self.header(file)?;
        let t = h.tensors.iter().find(|t| t.name == key).with_context(|| format!("{file}: no tensor {key}"))?;
        Ok((t.dtype.clone(), t.shape.clone()))
    }

    /// A float tensor (F32 / BF16 / F16) as f32 values.
    pub fn f32(&self, key: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let (dt, sh, b) = self.raw(key)?;
        ensure!(matches!(dt.as_str(), "F32" | "BF16" | "F16"), "{key}: dtype {dt} is not a float tensor");
        Ok((sh, bytes_to_f32(&dt, &b)?))
    }

    /// An integer tensor (I32 / I64) as i64 values.
    pub fn ints(&self, key: &str) -> Result<(Vec<usize>, Vec<i64>)> {
        let (dt, sh, b) = self.raw(key)?;
        let v = match dt.as_str() {
            "I64" => b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
            "I32" => b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64).collect(),
            d => bail!("{key}: dtype {d} is not an integer tensor"),
        };
        Ok((sh, v))
    }

    /// The row positions of a per-row point (its `.rows` companion).
    pub fn rows(&self, key: &str) -> Result<Vec<usize>> { Ok(self.ints(&format!("{key}.rows"))?.1.into_iter().map(|v| v as usize).collect()) }

    /// Read a tensor (and its `.rows` companion, if dumped).
    pub fn get(&self, key: &str) -> Result<RefTensor> {
        let (dtype, shape, b) = self.raw(key)?;
        let (mut f, mut i) = (Vec::new(), Vec::new());
        match dtype.as_str() {
            "F32" | "BF16" | "F16" => f = bytes_to_f32(&dtype, &b)?,
            "I64" => i = b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
            "I32" => i = b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64).collect(),
            d => bail!("{key}: dtype {d} not handled"),
        }
        let rows_key = format!("{key}.rows");
        let rows = if self.has(&rows_key) { Some(self.ints(&rows_key)?.1) } else { None };
        if let Some(r) = &rows {
            ensure!(r.len() == shape.first().copied().unwrap_or(0), "{key}: {} row positions for {:?}", r.len(), shape);
        }
        Ok(RefTensor { name: key.to_string(), dtype, shape, f, i, rows })
    }

    /// A per-layer point.
    pub fn layer(&self, layer: usize, point: &str) -> Result<RefTensor> { self.get(&Self::key(layer, point)) }
    /// A per-layer point if dumped.
    pub fn layer_opt(&self, layer: usize, point: &str) -> Result<Option<RefTensor>> {
        let k = Self::key(layer, point);
        if self.has(&k) { self.get(&k).map(Some) } else { Ok(None) }
    }
    /// A per-prompt point (`G.{point}`).
    pub fn global(&self, point: &str) -> Result<RefTensor> { self.get(&format!("G.{point}")) }
    pub fn global_opt(&self, point: &str) -> Result<Option<RefTensor>> {
        let k = format!("G.{point}");
        if self.has(&k) { self.get(&k).map(Some) } else { Ok(None) }
    }
    /// The layers having `point` dumped (a dump still being written has a prefix of them).
    pub fn layers_with(&self, point: &str) -> Vec<usize> { (0..64).filter(|&l| self.has(&Self::key(l, point))).collect() }
    /// The layers with a dumped `x_in`.
    pub fn layers(&self) -> Vec<usize> { self.layers_with("x_in") }
}

/// A REFb key as an I0-style (layer, point): `L07.x_in` -> (Layer(7), "x_in"), `G.logits` -> (Global, "logits").
fn refb_split(key: &str) -> Option<(DumpLayer, String)> {
    let (head, point) = key.split_once('.')?;
    let layer = if head == "G" { DumpLayer::Global } else { DumpLayer::Layer(head.strip_prefix('L')?.parse().ok()?) };
    Some((layer, point.to_string()))
}

/// A dump directory, read back (the sidecar validated against every file).
/// Opens I0's layout (`dump.json`) and REFb's layout (`meta.json`, see [`RefbDump`]); for a REFb directory the points
/// are its per-layer and `G.` tensors (the `.rows` companions are reachable through [`Dump::refb`]).
pub struct Dump { pub dir: PathBuf, pub meta: Value, pub points: Vec<(DumpLayer, String, String, Vec<u64>)>, refb: Option<RefbDump> }

impl Dump {
    pub fn open(dir: &Path) -> Result<Dump> {
        if !dir.join("dump.json").exists() && RefbDump::is_refb(dir) { return Self::open_refb(dir); }
        let meta: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("dump.json")).with_context(|| format!("{}: no dump.json", dir.display()))?)?;
        ensure!(meta.get("format").and_then(Value::as_str) == Some(FORMAT), "{}: format is not {FORMAT}", dir.display());
        let mode = meta.get("mode").and_then(Value::as_str).unwrap_or("");
        ensure!(mode == "exact" || mode == "ced", "{}: mode {mode:?}", dir.display());
        let mut points = Vec::new();
        for p in meta.get("points").and_then(Value::as_array).context("dump.json has no points")? {
            let file = p.get("file").and_then(Value::as_str).context("point without file")?;
            let point = p.get("point").and_then(Value::as_str).context("point without name")?.to_string();
            let layer = match p.get("layer") { Some(Value::Number(n)) => DumpLayer::Layer(n.as_u64().context("layer")? as u16),
                                                Some(Value::String(s)) => DumpLayer::parse(s).context("layer tag")?, _ => bail!("{file}: layer") };
            ensure!(file == file_name(layer, &point), "{file}: name does not match layer {:?} point {point}", layer);
            let h = st::read_header(&dir.join(file))?;
            ensure!(h.tensors.len() == 1 && h.tensors[0].name == point, "{file}: must hold exactly the tensor {point}");
            let t = &h.tensors[0];
            let dtype = p.get("dtype").and_then(Value::as_str).unwrap_or("");
            let shape: Vec<u64> = p.get("shape").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
            ensure!(t.dtype == dtype && t.shape == shape, "{file}: sidecar {dtype} {shape:?} vs file {} {:?}", t.dtype, t.shape);
            points.push((layer, point, t.dtype.clone(), t.shape.clone()));
        }
        Ok(Dump { dir: dir.to_path_buf(), meta, points, refb: None })
    }
    fn open_refb(dir: &Path) -> Result<Dump> {
        let r = RefbDump::open(dir)?;
        let mut points = Vec::new();
        for k in r.keys() {
            if k.ends_with(".rows") { continue; }
            let Some((layer, point)) = refb_split(&k) else { continue };
            let (dt, shape) = r.describe(&k)?;
            points.push((layer, point, dt, shape));
        }
        Ok(Dump { dir: r.dir.clone(), meta: r.meta.clone(), points, refb: Some(r) })
    }
    /// The REFb reader behind this dump (None for an I0 dump).
    pub fn refb(&self) -> Option<&RefbDump> { self.refb.as_ref() }
    /// The name of a point as its layout spells it: the file name (I0) or the REFb key (`L07.x_in`, `G.logits`).
    pub fn label(&self, layer: DumpLayer, point: &str) -> String {
        if self.refb.is_none() { return file_name(layer, point); }
        match layer { DumpLayer::Layer(l) => RefbDump::key(l as usize, point), _ => format!("G.{point}") }
    }
    pub fn prompt_ids(&self) -> Vec<u64> { if let Some(r) = &self.refb { return r.global("prompt_ids").map(|t| t.i.iter().map(|&x| x as u64).collect()).unwrap_or_default(); } self.meta.get("prompt_ids").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default() }
    pub fn read(&self, layer: DumpLayer, point: &str) -> Result<DumpTensor> {
        if let Some(r) = &self.refb {
            let (dtype, shape, bytes) = r.raw(&self.label(layer, point))?;
            return Ok(DumpTensor { layer, point: point.into(), dtype, shape: shape.iter().map(|&x| x as u64).collect(), bytes });
        }
        let f = file_name(layer, point);
        let h = st::read_header(&self.dir.join(&f))?;
        let t = &h.tensors[0];
        Ok(DumpTensor { layer, point: point.into(), dtype: t.dtype.clone(), shape: t.shape.clone(), bytes: st::read_tensor(&h, point)? })
    }
}

/// rel-L2 ||a - b|| / ||b|| (b = the reference).
pub fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) { let d = *x as f64 - *y as f64; num += d * d; den += (*y as f64) * (*y as f64); }
    if den == 0.0 { if num == 0.0 { 0.0 } else { f64::INFINITY } } else { (num / den).sqrt() }
}

/// Per common point: (file, rel-L2 of `a` vs reference `b`, or why it cannot be compared).
pub fn compare(a: &Dump, b: &Dump) -> Vec<(String, Result<f64, String>)> {
    let mut out = Vec::new();
    for (l, p, _, shape) in &a.points {
        let f = a.label(*l, p);
        let Some((_, _, _, bs)) = b.points.iter().find(|(bl, bp, _, _)| bl == l && bp == p) else { continue };
        let r = if bs != shape { Err(format!("shape {shape:?} vs {bs:?}")) } else {
            match (a.read(*l, p).and_then(|t| t.to_f32()), b.read(*l, p).and_then(|t| t.to_f32())) {
                (Ok(x), Ok(y)) => Ok(rel_l2(&x, &y)),
                (Err(e), _) | (_, Err(e)) => Err(format!("{e:#}")),
            }
        };
        out.push((f, r));
    }
    out
}

/// A small demo dump (the Rust side of the Python cross-check: `dumpfmt.py ls` must read it).
pub fn write_demo(dir: &Path) -> Result<PathBuf> {
    let mut w = DumpWriter::create(dir, &format!("gb10_inference {} --dsv41-dump-demo", env!("CARGO_PKG_VERSION")), "-", "demo", "exact", &[0, 1, 2], (0, 3))?;
    let x: Vec<f32> = (0..3 * 8).map(|i| i as f32 * 0.25 - 1.0).collect();
    w.put_f32(DumpLayer::Emb, "x", &[3, 8], &x)?;
    let bf: Vec<u8> = x.iter().flat_map(|v| half::bf16::from_f32(*v * 2.0).to_le_bytes()).collect();
    w.put(DumpLayer::Layer(3), "attn_x", "BF16", &[3, 8], &bf)?;
    let ids: Vec<u8> = [5i32, -1, 7, 9, 11, 13].iter().flat_map(|v| v.to_le_bytes()).collect();
    w.put(DumpLayer::Layer(20), "index_sel", "I32", &[3, 2], &ids)?;
    w.note("attn_x", "bf16, = 2 x emb.x");
    w.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv41_dump_roundtrip_and_compare() {
        let d = std::env::temp_dir().join(format!("dsv41_dump_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        write_demo(&d).unwrap();
        let a = Dump::open(&d).unwrap();
        assert_eq!(a.points.len(), 3);
        assert_eq!(a.prompt_ids(), vec![0, 1, 2]);
        let t = a.read(DumpLayer::Layer(3), "attn_x").unwrap();
        let e = a.read(DumpLayer::Emb, "x").unwrap().to_f32().unwrap();
        assert_eq!(t.to_f32().unwrap(), e.iter().map(|v| v * 2.0).collect::<Vec<_>>());
        assert_eq!(a.read(DumpLayer::Layer(20), "index_sel").unwrap().to_f32().unwrap()[1], -1.0);
        let c = compare(&a, &a);
        assert_eq!(c.len(), 3);
        assert!(c.iter().all(|(_, r)| r.as_ref().map_or(false, |v| *v == 0.0)));
        assert_eq!(file_name(DumpLayer::Layer(42), "draft_ids"), "L42.draft_ids.safetensors");
        assert_eq!(file_name(DumpLayer::Head, "logits"), "Lhead.logits.safetensors");
        // a duplicate point and a bad name are refused
        let mut w = DumpWriter::create(&d.join("x"), "t", "-", "-", "ced", &[], (0, 0)).unwrap();
        w.put_f32(DumpLayer::Emb, "x", &[1], &[1.0]).unwrap();
        assert!(w.put_f32(DumpLayer::Emb, "x", &[1], &[1.0]).is_err());
        assert!(w.put_f32(DumpLayer::Emb, "Bad", &[1], &[1.0]).is_err());
        assert!(DumpWriter::create(&d.join("y"), "t", "-", "-", "fast", &[], (0, 0)).is_err());
        std::fs::remove_dir_all(&d).ok();
    }

    /// A synthetic REFb-layout dump (meta.json + L00 + G files, an `.rows` companion, and the fixture's outputs one
    /// level up) read through the one reader, the I0 `Dump` wrapper, and the OM / OA entry points.
    #[test]
    fn fixture_name_is_the_fixture_dir_for_both_path_forms() {
        for (p, want) in [("/r/agent_4k", "agent_4k"), ("/r/agent_4k/", "agent_4k"), ("/r/agent_4k/prefill_exact", "agent_4k"), ("/r/agent_4k/prefill_exact/", "agent_4k")] {
            assert_eq!(RefbDump::fixture_name(Path::new(p)), want, "{p}");
        }
    }

    #[test]
    fn dsv41_refb_layout_is_read_by_the_one_reader() {
        let root = std::env::temp_dir().join(format!("dsv41_refb_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let d = root.join("prefill_exact");
        std::fs::create_dir_all(&d).unwrap();
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let i64b = |v: &[i64]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let bf = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| half::bf16::from_f32(*x).to_le_bytes()).collect() };
        let (xin, rows, attn) = (f32b(&[1., 2., 3., 4., 5., 6.]), i64b(&[3, 7]), bf(&[0.5, -1.5, 2.0, 4.0]));
        let l0 = [
            StOut { name: "L00.x_in".into(), dtype: "F32".into(), shape: vec![2, 3], src: StSource::Bytes(&xin) },
            StOut { name: "L00.x_in.rows".into(), dtype: "I64".into(), shape: vec![2], src: StSource::Bytes(&rows) },
            StOut { name: "L00.attn_in".into(), dtype: "BF16".into(), shape: vec![2, 2], src: StSource::Bytes(&attn) },
            StOut { name: "L00.attn_in.rows".into(), dtype: "I64".into(), shape: vec![2], src: StSource::Bytes(&rows) },
        ];
        st::write_file(&d.join("L00.safetensors"), &l0, &BTreeMap::new(), &mut |_| {}).unwrap();
        let ids = i64b(&[10, 11, 12]);
        let g = [StOut { name: "G.prompt_ids".into(), dtype: "I64".into(), shape: vec![3], src: StSource::Bytes(&ids) }];
        st::write_file(&root.join("outputs.safetensors"), &g, &BTreeMap::new(), &mut |_| {}).unwrap();
        let tens: serde_json::Map<String, Value> = [("L00.x_in", "F32", vec![2, 3]), ("L00.x_in.rows", "I64", vec![2]), ("L00.attn_in", "BF16", vec![2, 2]), ("L00.attn_in.rows", "I64", vec![2])]
            .into_iter().map(|(k, dt, sh)| (k.to_string(), json!({"file": "L00.safetensors", "dtype": dt, "shape": sh}))).collect();
        std::fs::write(d.join("meta.json"), serde_json::to_string(&json!({"P": 8, "mode": "exact", "name": "syn", "tensors": tens})).unwrap()).unwrap();

        // the real fixture root carries its own summary meta.json (name, P, outputs; NO tensors index): it must not
        // stop the descent into prefill_exact/ (PK5 found every probe failing on the root with "no tensors index")
        std::fs::write(root.join("meta.json"), serde_json::to_string(&json!({"leg": "refb", "name": "syn", "P": 8, "mode": "exact", "outputs": {}})).unwrap()).unwrap();
        // a fixture root and its variant directory open the same dump
        for dir in [&root, &d] {
            let r = RefbDump::open(dir).unwrap();
            assert_eq!((r.p, r.mode(), r.name()), (8, "exact", "syn"));
            assert_eq!(r.layers(), vec![0]);
            assert_eq!(r.layers_with("attn_in"), vec![0]);
            let t = r.layer(0, "x_in").unwrap();
            assert_eq!((t.shape.clone(), t.f.clone(), t.positions()), (vec![2, 3], vec![1., 2., 3., 4., 5., 6.], vec![3, 7]));
            assert_eq!((t.index_of(7), t.index_of(4), t.row_f(1)), (Some(1), None, &[4., 5., 6.][..]));
            assert_eq!(r.f32("L00.attn_in").unwrap().1, vec![0.5, -1.5, 2.0, 4.0]);
            assert_eq!(r.rows("L00.x_in").unwrap(), vec![3, 7]);
            assert_eq!(r.global("prompt_ids").unwrap().i, vec![10, 11, 12]); // from ../outputs.safetensors
            assert!(r.f32("L00.x_in.rows").is_err() && r.ints("L00.x_in").is_err() && r.layer(5, "x_in").is_err());
        }
        // Dump::open takes the REFb layout: I0 accessors, REFb labels, no `.rows` points
        let dd = Dump::open(&root).unwrap();
        assert!(dd.refb().is_some());
        assert_eq!(dd.prompt_ids(), vec![10, 11, 12]);
        let names: Vec<String> = dd.points.iter().map(|(l, p, _, _)| dd.label(*l, p)).collect();
        assert_eq!(names, vec!["G.prompt_ids", "L00.attn_in", "L00.x_in"]);
        assert_eq!(dd.read(DumpLayer::Layer(0), "x_in").unwrap().to_f32().unwrap()[5], 6.0);
        assert_eq!(dd.read(DumpLayer::Layer(0), "attn_in").unwrap().to_f32().unwrap(), vec![0.5, -1.5, 2.0, 4.0]);
        let c = compare(&dd, &dd);
        assert_eq!(c.len(), 3);
        assert!(c.iter().all(|(_, r)| r.as_ref().map_or(false, |v| *v == 0.0)));
        assert_eq!(RefbDump::variant_dir(&root), d);
        assert_eq!(RefbDump::variant_dir(&d), d);
        // the old entry points are the same reader
        assert_eq!(crate::dsv41::cpu::mhc::refdump::RefDump::open(&d).unwrap().layers(), vec![0]);
        let oa = crate::dsv41::cpu::attn::fixture::RefDump::open(&root).unwrap();
        assert_eq!((oa.p().unwrap(), oa.layers(), oa.has("L00.x_in")), (8, vec![0], true));
        assert_eq!(oa.ints("L00.x_in.rows").unwrap().1, vec![3, 7]);
        // an I0 dump is still an I0 dump
        assert!(!RefbDump::is_refb(&root.join("none")));
        std::fs::remove_dir_all(&root).ok();
    }
}
