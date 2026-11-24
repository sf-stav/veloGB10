//! DSV4.1 (leg I0): the slice extractor — a valid mini-pack of chosen layers of any V4.1 pack
//! (native or EXL3, any mix), for the single-box legs (`--dsv41-slice-pack`).
//!
//! A slice keeps the REAL layer ids (tensor names unchanged, `layers.20.*` stays `layers.20.*`,
//! DSpark stage s stays `mtp.s.*`), the globals (embedding, head, final norm, image separators),
//! every tensor of the kept layers byte for byte, and writes `config.json` with a `dsv41_slice`
//! block (kept layers, source, what was left out, missing cross-layer dependencies), an index, the
//! filtered `quantization_config.json` and the tokenizer / chat-format files. Left out unless asked:
//! the Engram tables (~101 GB per Engram layer in the native checkpoint; every pack reads DeepSeek's
//! tables from the native shards or the packed head files) and the vision tower. The census re-reads
//! the slice before the command reports success.

use crate::dsv41::census::classify::TensorClass;
use crate::dsv41::census::Census;
use crate::dsv41::st::{self, StOut, StSource};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct SliceOpts {
    pub layers: Vec<usize>,
    pub engram_tables: bool,
    pub vision: bool,
}

/// What a slice run did.
#[derive(Clone, Debug)]
pub struct SliceReport {
    pub out: PathBuf,
    pub tensors: usize,
    pub bytes: u64,
    pub files: usize,
    pub missing_deps: Vec<String>,
    pub excluded: Vec<String>,
}

/// Parse `--layers 0,1,2,14` (also ranges `18-22`).
pub fn parse_layers(s: &str) -> Result<Vec<usize>> {
    let mut v = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => { let (a, b): (usize, usize) = (a.trim().parse()?, b.trim().parse()?); ensure!(a <= b, "bad range {part}"); v.extend(a..=b); }
            None => v.push(part.parse().with_context(|| format!("bad layer {part:?}"))?),
        }
    }
    v.sort();
    v.dedup();
    ensure!(!v.is_empty(), "--layers is empty");
    Ok(v)
}

/// The cross-layer reads a kept layer has that the slice does not keep.
pub fn missing_dependencies(c: &crate::dsv41::config::Dsv41Config, layers: &[usize]) -> Vec<String> {
    let mut out = Vec::new();
    for &l in layers {
        if let Some(s) = c.kv_source_of(l) { if !layers.contains(&s) { out.push(format!("layer {l} reads the compressed KV / index-K of KV source {s}")); } }
        if let Some(s) = c.index_source_of(l) { if !layers.contains(&s) { out.push(format!("layer {l} reuses the top-k of index source {s}")); } }
        if c.uses_candidates(l) { if let Some(s) = c.candidate_source { if !layers.contains(&s) { out.push(format!("layer {l} masks with the candidate blocks of layer {s}")); } } }
        if c.is_dspark(l) {
            out.push(format!("DSpark stage {} needs main_hidden from target layers {:?} (a dump, or the trunk)", l - c.layers, c.dspark_targets));
        }
    }
    out.dedup();
    out
}

fn free_bytes(dir: &Path) -> Option<u64> {
    let c = std::ffi::CString::new(dir.as_os_str().to_string_lossy().as_bytes()).ok()?;
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut s) } != 0 { return None; }
    Some(s.f_bavail as u64 * s.f_frsize as u64)
}

/// Write the slice. `progress(done, total)` gets the bytes copied (a boot_report::Progress hook).
pub fn write_slice(src: &Path, out: &Path, o: &SliceOpts, progress: &mut dyn FnMut(u64, u64)) -> Result<SliceReport> {
    let c = Census::build(src)?;
    c.check().context("the source pack's census has problems; refusing to slice it")?;
    ensure!(c.cfg.slice_layers.is_none(), "{} is a slice already", src.display());
    for &l in &o.layers { ensure!(l < c.cfg.total_layers(), "layer {l} >= {} layer slots", c.cfg.total_layers()); }
    if out.exists() && std::fs::read_dir(out)?.next().is_some() { bail!("{} exists and is not empty", out.display()); }
    std::fs::create_dir_all(out).with_context(|| format!("create {}", out.display()))?;

    // ---- select ----
    let mut excluded: BTreeMap<&'static str, (usize, u64)> = BTreeMap::new();
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, e) in c.entries.iter().enumerate() {
        let file = match e.layer {
            Some(l) if !o.layers.contains(&(l as usize)) => continue,
            Some(_) if e.class == TensorClass::EngramTable && !o.engram_tables => {
                let x = excluded.entry("engram tables").or_default(); x.0 += 1; x.1 += e.bytes; continue;
            }
            Some(l) => format!("model-L{l:02}.safetensors"),
            None if e.class.group() == "vision" => {
                if !o.vision { let x = excluded.entry("vision tower").or_default(); x.0 += 1; x.1 += e.bytes; continue; }
                "model-vision.safetensors".to_string()
            }
            None => "model-globals.safetensors".to_string(),
        };
        groups.entry(file).or_default().push(i);
    }
    let total: u64 = groups.values().flatten().map(|&i| c.entries[i].bytes).sum();
    let tensors: usize = groups.values().map(Vec::len).sum();
    if let Some(free) = free_bytes(out) {
        ensure!(free > total + total / 20 + (1 << 30), "{}: {:.1} GB free, the slice needs {:.1} GB (+5% + 1 GB margin)",
                out.display(), free as f64 / 1e9, total as f64 / 1e9);
    }
    // ---- write the shards (tensors in source order, so reads are sequential) ----
    let mut done = 0u64;
    let mut wm = serde_json::Map::new();
    for (file, idx) in &groups {
        let mut idx = idx.clone();
        idx.sort_by_key(|&i| (c.entries[i].file, c.entries[i].offset));
        let srcs: Vec<PathBuf> = idx.iter().map(|&i| c.dir.join(&c.files[c.entries[i].file as usize])).collect();
        let outs: Vec<StOut> = idx.iter().zip(&srcs).map(|(&i, p)| {
            let e = &c.entries[i];
            StOut { name: e.name.clone(), dtype: e.dtype.clone(), shape: e.shape.clone(), src: StSource::File { path: p.as_path(), offset: e.offset, len: e.bytes } }
        }).collect();
        let base = done;
        st::write_file(&out.join(file), &outs, &BTreeMap::new(), &mut |b| progress(base + b, total))?;
        done += idx.iter().map(|&i| c.entries[i].bytes).sum::<u64>();
        for &i in &idx { wm.insert(c.entries[i].name.clone(), Value::String(file.clone())); }
    }
    std::fs::write(out.join("model.safetensors.index.json"),
                   serde_json::to_string_pretty(&json!({"metadata": {"total_size": total}, "weight_map": wm}))?)?;

    // ---- config.json with the slice block ----
    let missing = missing_dependencies(&c.cfg, &o.layers);
    let mut cfg: Value = serde_json::from_str(&std::fs::read_to_string(src.join("config.json"))?)?;
    let excl: Vec<Value> = excluded.iter().map(|(k, (n, b))| json!({"what": k, "tensors": n, "bytes": b})).collect();
    cfg["dsv41_slice"] = json!({
        "layers": o.layers, "source_dir": src.display().to_string(),
        "source_tensors": c.entries.len(), "source_bytes": c.total_bytes(),
        "excluded": excl, "missing_dependencies": missing,
        "created_by": format!("gb10_inference {} --dsv41-slice-pack", env!("CARGO_PKG_VERSION")),
    });
    std::fs::write(out.join("config.json"), serde_json::to_string_pretty(&cfg)?)?;
    // ---- quantization_config.json: the kept modules' tensor_storage entries ----
    let qp = src.join("quantization_config.json");
    if qp.exists() {
        let mut q: Value = serde_json::from_str(&std::fs::read_to_string(&qp)?)?;
        if let Some(ts) = q.get_mut("tensor_storage").and_then(Value::as_object_mut) {
            ts.retain(|_, v| v.get("stored_tensors").and_then(Value::as_object)
                .map_or(false, |s| !s.is_empty() && s.keys().all(|k| wm.contains_key(k))));
        }
        std::fs::write(out.join("quantization_config.json"), serde_json::to_string(&q)?)?;
    }
    // ---- tokenizer / chat format ----
    for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"] {
        let p = src.join(f);
        if p.exists() { std::fs::copy(&p, out.join(f)).with_context(|| format!("copy {f}"))?; }
    }
    let enc = src.join("encoding/encoding.py");
    if enc.exists() {
        std::fs::create_dir_all(out.join("encoding"))?;
        std::fs::copy(&enc, out.join("encoding/encoding.py"))?;
    }
    // ---- re-read: the census of the slice must be clean and byte-identical on spot checks ----
    let s = Census::build(out)?;
    s.check().context("the written slice's census has problems")?;
    ensure!(s.entries.len() == tensors && s.total_bytes() == total, "slice census {} tensors / {} B != planned {tensors} / {total}", s.entries.len(), s.total_bytes());
    spot_check(&c, &s)?;
    Ok(SliceReport { out: out.to_path_buf(), tensors, bytes: total, files: groups.len(), missing_deps: missing,
                     excluded: excluded.iter().map(|(k, (n, b))| format!("{k}: {n} tensors, {:.1} GB", *b as f64 / 1e9)).collect() })
}

/// Compare the first and last 4 KiB of every tensor (all of a small one) between source and slice.
fn spot_check(src: &Census, sl: &Census) -> Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let by: std::collections::HashMap<&str, &crate::dsv41::census::Entry> = src.entries.iter().map(|e| (e.name.as_str(), e)).collect();
    let mut open: BTreeMap<String, std::fs::File> = BTreeMap::new();
    let mut read = |dir: &Path, f: &str, off: u64, n: usize| -> Result<Vec<u8>> {
        let key = dir.join(f).display().to_string();
        if !open.contains_key(&key) { open.insert(key.clone(), std::fs::File::open(dir.join(f))?); }
        let h = open.get_mut(&key).unwrap();
        h.seek(SeekFrom::Start(off))?;
        let mut v = vec![0u8; n];
        h.read_exact(&mut v)?;
        Ok(v)
    };
    for e in &sl.entries {
        let s = by.get(e.name.as_str()).with_context(|| format!("{} not in the source", e.name))?;
        ensure!(s.dtype == e.dtype && s.shape == e.shape && s.bytes == e.bytes, "{}: header differs from the source", e.name);
        let n = e.bytes.min(4096) as usize;
        let windows = if e.bytes <= 8192 { vec![(0, e.bytes as usize)] } else { vec![(0, n), (e.bytes - n as u64, n)] };
        for (o, len) in windows {
            if len == 0 { continue; }
            let a = read(&src.dir, &src.files[s.file as usize], s.offset + o, len)?;
            let b = read(&sl.dir, &sl.files[e.file as usize], e.offset + o, len)?;
            ensure!(a == b, "{}: bytes differ from the source at +{o}", e.name);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv41::census::synth::{small_config, tmp, write_pack, SFmt};

    #[test]
    fn dsv41_slice_parse_layers() {
        assert_eq!(parse_layers("0,1,2,14,20,21,40-42").unwrap(), vec![0, 1, 2, 14, 20, 21, 40, 41, 42]);
        assert_eq!(parse_layers(" 3 , 3,2").unwrap(), vec![2, 3]);
        assert!(parse_layers("").is_err());
        assert!(parse_layers("5-2").is_err());
    }

    #[test]
    fn dsv41_slice_of_a_synthetic_pack_is_a_clean_pack() {
        let src = tmp("slice_src");
        let all: Vec<usize> = (0..43).collect();
        let pol = |c: TensorClass, _l: Option<u16>, _s: Option<u16>| match c {
            TensorClass::Embed | TensorClass::DsMarkovEmbed | TensorClass::GateWeight | TensorClass::IdxWeightsProj
            | TensorClass::DsMarkovHead | TensorClass::DsConfidence => SFmt::Plain("BF16"),
            _ => SFmt::Exl3(3),
        };
        write_pack(&src, &small_config(2, 2), &all, true, false, &pol);
        // write real bytes into one small tensor so the spot check compares something non-zero
        let out = tmp("slice_out");
        std::fs::remove_dir_all(&out).ok();
        let o = SliceOpts { layers: vec![0, 1, 3, 21, 24, 40, 41, 42], engram_tables: false, vision: false };
        let r = write_slice(&src, &out, &o, &mut |_, _| {}).unwrap();
        assert_eq!(r.files, 9);
        assert!(r.missing_deps.iter().any(|m| m.contains("layer 3 reads the compressed KV / index-K of KV source 2")), "{:?}", r.missing_deps);
        assert!(r.missing_deps.iter().any(|m| m.contains("layer 24 masks with the candidate blocks of layer 20")), "{:?}", r.missing_deps);
        let s = Census::build(&out).unwrap();
        assert_eq!(s.kept_layers(), o.layers);
        assert!(s.modules.iter().any(|m| m.class == TensorClass::DsMainProj));
        // the slice of a slice is refused; a non-empty output dir is refused
        let out2 = tmp("slice_out2");
        assert!(write_slice(&out, &out2, &o, &mut |_, _| {}).is_err());
        assert!(write_slice(&src, &out, &o, &mut |_, _| {}).is_err());
        for d in [src, out, out2] { std::fs::remove_dir_all(&d).ok(); }
    }
}
