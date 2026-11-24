//! DSV4.1 (leg I0): safetensors header reading and writing, shared by the census, the slice
//! extractor and the tensor-dump format. Header-only reads (8-byte little-endian length + JSON);
//! writes pad the JSON with spaces to an 8-byte boundary (the format's recommendation), so a
//! tensor's data offset is always 8-byte aligned.

use anyhow::{bail, ensure, Context, Result};
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Element size of a safetensors dtype string (None = unknown dtype: the census refuses it).
pub fn dtype_size(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" => 1,
        "I16" | "U16" | "F16" | "BF16" => 2,
        "I32" | "U32" | "F32" => 4,
        "I64" | "U64" | "F64" => 8,
        _ => return None,
    })
}

/// One tensor of a safetensors file, as its header describes it.
#[derive(Clone, Debug, PartialEq)]
pub struct StTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// byte range inside the data section (relative to `StHeader::data_start`)
    pub begin: u64,
    pub end: u64,
}

impl StTensor {
    pub fn bytes(&self) -> u64 { self.end - self.begin }
    pub fn numel(&self) -> u64 { self.shape.iter().product() }
}

/// A file's header: its tensors in data order, the absolute offset of the data section, metadata.
#[derive(Clone, Debug)]
pub struct StHeader {
    pub path: PathBuf,
    pub data_start: u64,
    pub file_len: u64,
    pub tensors: Vec<StTensor>,
    pub metadata: BTreeMap<String, String>,
}

/// Read only the header of `path` (no tensor data). Checks every tensor's byte range against its
/// dtype x shape and the file length, so a truncated or mislabelled file fails here.
pub fn read_header(path: &Path) -> Result<StHeader> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let file_len = f.metadata()?.len();
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8).with_context(|| format!("{}: no header length", path.display()))?;
    let n = u64::from_le_bytes(len8);
    ensure!(n > 1 && n < (1 << 31) && 8 + n <= file_len, "{}: bad header length {n}", path.display());
    let mut buf = vec![0u8; n as usize];
    f.read_exact(&mut buf)?;
    let v: serde_json::Value = serde_json::from_slice(&buf).with_context(|| format!("{}: header JSON", path.display()))?;
    let obj = v.as_object().with_context(|| format!("{}: header is not an object", path.display()))?;
    let data_start = 8 + n;
    let mut tensors = Vec::with_capacity(obj.len());
    let mut metadata = BTreeMap::new();
    for (k, e) in obj {
        if k == "__metadata__" {
            if let Some(m) = e.as_object() {
                for (mk, mv) in m { metadata.insert(mk.clone(), mv.as_str().unwrap_or_default().to_string()); }
            }
            continue;
        }
        let dtype = e.get("dtype").and_then(|d| d.as_str()).with_context(|| format!("{}: {k} has no dtype", path.display()))?.to_string();
        let shape: Vec<u64> = e.get("shape").and_then(|s| s.as_array()).with_context(|| format!("{}: {k} has no shape", path.display()))?
            .iter().map(|x| x.as_u64().unwrap_or(u64::MAX)).collect();
        ensure!(!shape.contains(&u64::MAX), "{}: {k} has a bad shape", path.display());
        let off = e.get("data_offsets").and_then(|s| s.as_array()).with_context(|| format!("{}: {k} has no data_offsets", path.display()))?;
        ensure!(off.len() == 2, "{}: {k} data_offsets", path.display());
        let (begin, end) = (off[0].as_u64().unwrap_or(u64::MAX), off[1].as_u64().unwrap_or(u64::MAX));
        ensure!(begin <= end && data_start + end <= file_len, "{}: {k} data range {begin}..{end} past the file end", path.display());
        let es = dtype_size(&dtype).with_context(|| format!("{}: {k} has an unknown dtype {dtype}", path.display()))?;
        let numel: u64 = shape.iter().product();
        ensure!(numel * es == end - begin, "{}: {k} is {} bytes, {dtype} x {:?} needs {}", path.display(), end - begin, shape, numel * es);
        tensors.push(StTensor { name: k.clone(), dtype, shape, begin, end });
    }
    tensors.sort_by_key(|t| t.begin);
    Ok(StHeader { path: path.to_path_buf(), data_start, file_len, tensors, metadata })
}

/// Where a tensor's bytes come from when a file is written.
pub enum StSource<'a> {
    Bytes(&'a [u8]),
    /// `len` bytes at absolute offset `offset` of an existing file
    File { path: &'a Path, offset: u64, len: u64 },
    /// `len` bytes that are never written (a sparse hole: header-only test packs)
    Hole(u64),
}

impl StSource<'_> {
    fn len(&self) -> u64 {
        match self { StSource::Bytes(b) => b.len() as u64, StSource::File { len, .. } => *len, StSource::Hole(n) => *n }
    }
}

/// One tensor to write.
pub struct StOut<'a> {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub src: StSource<'a>,
}

/// The JSON header for `tensors` (in order), padded with spaces to a multiple of 8 bytes.
pub fn header_bytes(tensors: &[StOut], metadata: &BTreeMap<String, String>) -> Result<Vec<u8>> {
    let mut m = serde_json::Map::new();
    if !metadata.is_empty() {
        let mut md = serde_json::Map::new();
        for (k, v) in metadata { md.insert(k.clone(), serde_json::Value::String(v.clone())); }
        m.insert("__metadata__".into(), serde_json::Value::Object(md));
    }
    let mut off = 0u64;
    for t in tensors {
        let es = dtype_size(&t.dtype).with_context(|| format!("{}: unknown dtype {}", t.name, t.dtype))?;
        let n = t.src.len();
        ensure!(t.shape.iter().product::<u64>() * es == n, "{}: {} bytes do not match {} x {:?}", t.name, n, t.dtype, t.shape);
        ensure!(!m.contains_key(&t.name), "duplicate tensor {}", t.name);
        m.insert(t.name.clone(), serde_json::json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [off, off + n]}));
        off += n;
    }
    let mut h = serde_json::to_vec(&serde_json::Value::Object(m))?;
    while h.len() % 8 != 0 { h.push(b' '); }
    Ok(h)
}

/// Write a safetensors file. File sources are copied in 64 MiB pieces; `progress` is called with
/// the bytes written so far (a boot_report::Progress can hang off it). Holes leave the file sparse.
pub fn write_file(path: &Path, tensors: &[StOut], metadata: &BTreeMap<String, String>,
                  progress: &mut dyn FnMut(u64)) -> Result<u64> {
    let h = header_bytes(tensors, metadata)?;
    let tmp = path.with_extension("safetensors.partial");
    let mut f = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?);
    f.write_all(&(h.len() as u64).to_le_bytes())?;
    f.write_all(&h)?;
    let mut written = 0u64;
    let mut buf = vec![0u8; 64 << 20];
    for t in tensors {
        match &t.src {
            StSource::Bytes(b) => { f.write_all(b)?; written += b.len() as u64; progress(written); }
            StSource::File { path: sp, offset, len } => {
                let mut s = std::fs::File::open(sp).with_context(|| format!("open {}", sp.display()))?;
                s.seek(SeekFrom::Start(*offset))?;
                let mut left = *len;
                while left > 0 {
                    let n = left.min(buf.len() as u64) as usize;
                    s.read_exact(&mut buf[..n]).with_context(|| format!("read {} from {}", t.name, sp.display()))?;
                    f.write_all(&buf[..n])?;
                    left -= n as u64;
                    written += n as u64;
                    progress(written);
                }
            }
            StSource::Hole(n) => {
                f.flush()?;
                let inner = f.get_mut();
                inner.seek(SeekFrom::Current(*n as i64))?;
                written += n;
            }
        }
    }
    f.flush()?;
    let inner = f.into_inner().map_err(|e| anyhow::anyhow!("flush {}: {e}", tmp.display()))?;
    // a trailing hole must still extend the file
    inner.set_len(8 + h.len() as u64 + written)?;
    inner.sync_all().ok();
    drop(inner);
    std::fs::rename(&tmp, path).with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(8 + h.len() as u64 + written)
}

/// Read one tensor's bytes (small tensors: dumps, tests, spot checks).
pub fn read_tensor(h: &StHeader, name: &str) -> Result<Vec<u8>> {
    let Some(t) = h.tensors.iter().find(|t| t.name == name) else { bail!("{}: no tensor {name}", h.path.display()) };
    let mut f = std::fs::File::open(&h.path)?;
    f.seek(SeekFrom::Start(h.data_start + t.begin))?;
    let mut v = vec![0u8; t.bytes() as usize];
    f.read_exact(&mut v)?;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv41_st_roundtrip_with_hole_and_padding() {
        let dir = std::env::temp_dir().join(format!("dsv41_st_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.safetensors");
        let a: Vec<u8> = (0..12u8).collect();
        let ts = vec![
            StOut { name: "x".into(), dtype: "F32".into(), shape: vec![3], src: StSource::Bytes(&a) },
            StOut { name: "big".into(), dtype: "BF16".into(), shape: vec![1024, 1024], src: StSource::Hole(2 << 20) },
            StOut { name: "y".into(), dtype: "I32".into(), shape: vec![3], src: StSource::Bytes(&a) },
        ];
        let mut md = BTreeMap::new();
        md.insert("k".to_string(), "v".to_string());
        let n = write_file(&p, &ts, &md, &mut |_| {}).unwrap();
        let h = read_header(&p).unwrap();
        assert_eq!(h.file_len, n);
        assert_eq!(h.data_start % 8, 0);
        assert_eq!(h.tensors.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(), ["x", "big", "y"]);
        assert_eq!(read_tensor(&h, "y").unwrap(), a);
        assert_eq!(h.metadata.get("k").map(String::as_str), Some("v"));
        // a size mismatch is refused at write time
        let bad = vec![StOut { name: "z".into(), dtype: "F32".into(), shape: vec![4], src: StSource::Bytes(&a) }];
        assert!(header_bytes(&bad, &BTreeMap::new()).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
