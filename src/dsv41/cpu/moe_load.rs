//! DSV4.1 CPU oracle (leg OF): read tensors of a native checkpoint (or a slice of it) through its
//! `model.safetensors.index.json`, whole or by leading-dimension rows, and the REFb per-layer dump files.

use super::quant::{Fp4W, Fp8W};
use crate::dsv41::st::{self, StHeader};
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub struct Store {
    pub dir: PathBuf,
    map: HashMap<String, String>,
    hdrs: Mutex<HashMap<String, std::sync::Arc<StHeader>>>,
}

impl Store {
    pub fn open(dir: &Path) -> Result<Store> {
        let idx = dir.join("model.safetensors.index.json");
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&idx).with_context(|| format!("read {}", idx.display()))?)?;
        let wm = v.get("weight_map").and_then(|m| m.as_object()).context("index has no weight_map")?;
        let map = wm.iter().filter_map(|(k, f)| f.as_str().map(|f| (k.clone(), f.to_string()))).collect();
        Ok(Store { dir: dir.to_path_buf(), map, hdrs: Mutex::new(HashMap::new()) })
    }
    pub fn has(&self, name: &str) -> bool { self.map.contains_key(name) }
    fn header(&self, file: &str) -> Result<std::sync::Arc<StHeader>> {
        let mut g = self.hdrs.lock().unwrap();
        if let Some(h) = g.get(file) { return Ok(h.clone()); }
        let h = std::sync::Arc::new(st::read_header(&self.dir.join(file))?);
        g.insert(file.to_string(), h.clone());
        Ok(h)
    }
    /// (dtype, shape, absolute file, absolute byte offset, byte length) of a tensor.
    pub fn locate(&self, name: &str) -> Result<(String, Vec<u64>, PathBuf, u64, u64)> {
        let f = self.map.get(name).with_context(|| format!("{}: no tensor {name} in the index", self.dir.display()))?;
        let h = self.header(f)?;
        let t = h.tensors.iter().find(|t| t.name == name).with_context(|| format!("{f}: header lacks {name}"))?;
        Ok((t.dtype.clone(), t.shape.clone(), h.path.clone(), h.data_start + t.begin, t.bytes()))
    }
    pub fn bytes(&self, name: &str) -> Result<(String, Vec<u64>, Vec<u8>)> {
        let (dt, sh, path, off, len) = self.locate(name)?;
        Ok((dt, sh, read_at(&path, off, len)?))
    }
    /// Rows `[r0, r0 + n)` of a tensor, by its leading dimension.
    pub fn rows(&self, name: &str, r0: u64, n: u64) -> Result<Vec<u8>> {
        let (_, sh, path, off, len) = self.locate(name)?;
        let per = len / sh[0];
        if r0 + n > sh[0] { bail!("{name}: rows {r0}+{n} past {}", sh[0]); }
        read_at(&path, off + r0 * per, n * per)
    }
    pub fn f32s(&self, name: &str) -> Result<Vec<f32>> {
        let (dt, _, b) = self.bytes(name)?;
        bytes_to_f32(&dt, &b)
    }
    pub fn fp8(&self, base: &str) -> Result<Fp8W> {
        let (dt, sh, codes) = self.bytes(&format!("{base}.weight"))?;
        if dt != "F8_E4M3" { bail!("{base}.weight is {dt}, not F8_E4M3"); }
        let (_, _, scale) = self.bytes(&format!("{base}.scale"))?;
        Ok(Fp8W { n: sh[0] as usize, k: sh[1] as usize, codes, scale })
    }
    pub fn fp4(&self, base: &str) -> Result<Fp4W> {
        let (dt, sh, packed) = self.bytes(&format!("{base}.weight"))?;
        if dt != "I8" { bail!("{base}.weight is {dt}, not the packed-fp4 I8"); }
        let (_, _, scale) = self.bytes(&format!("{base}.scale"))?;
        Ok(Fp4W { n: sh[0] as usize, k: sh[1] as usize * 2, packed, scale })
    }
}

pub fn read_at(path: &Path, off: u64, len: u64) -> Result<Vec<u8>> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    f.seek(SeekFrom::Start(off))?;
    let mut v = vec![0u8; len as usize];
    f.read_exact(&mut v)?;
    Ok(v)
}

pub fn bytes_to_f32(dt: &str, b: &[u8]) -> Result<Vec<f32>> {
    Ok(match dt {
        "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "BF16" => b.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        "F16" => b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        "I64" => b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
        d => bail!("dtype {d} has no f32 view"),
    })
}

/// One REFb per-layer dump file (`L01.safetensors` / `G.safetensors`): named tensors, read on demand.
pub struct RefFile { h: StHeader }
impl RefFile {
    pub fn open(dir: &Path, stem: &str) -> Result<RefFile> {
        Ok(RefFile { h: st::read_header(&dir.join(format!("{stem}.safetensors")))? })
    }
    pub fn has(&self, name: &str) -> bool { self.h.tensors.iter().any(|t| t.name == name) }
    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.h.tensors.iter().find(|t| t.name == name).with_context(|| format!("no {name}"))?.shape.iter().map(|&x| x as usize).collect())
    }
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.h.tensors.iter().find(|t| t.name == name).with_context(|| format!("{}: no {name}", self.h.path.display()))?;
        bytes_to_f32(&t.dtype, &st::read_tensor(&self.h, name)?)
    }
    pub fn i64s(&self, name: &str) -> Result<Vec<i64>> {
        Ok(st::read_tensor(&self.h, name)?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
    }
}
