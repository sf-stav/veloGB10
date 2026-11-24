//! Read the Engram tables out of a native DeepSeek-V4.1 checkpoint (safetensors shards).

use super::manifest::{Geometry, SourceLayer, TensorRef, RECORD_BYTES, SCALE_BYTES, VALUE_BYTES};
use anyhow::{anyhow, bail, Context, Result};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

/// Parse a safetensors header; returns (header_len + 8, header json).
pub fn read_header(path: &Path) -> Result<(u64, serde_json::Value)> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut n = [0u8; 8];
    f.read_exact_at(&mut n, 0)?;
    let len = u64::from_le_bytes(n);
    if len > 64 << 20 {
        bail!("{}: implausible safetensors header length {len}", path.display());
    }
    let mut buf = vec![0u8; len as usize];
    f.read_exact_at(&mut buf, 8)?;
    Ok((len + 8, serde_json::from_slice(&buf)?))
}

fn tensor_ref(model_dir: &Path, index: &serde_json::Value, name: &str) -> Result<TensorRef> {
    let shard = index["weight_map"][name].as_str().ok_or_else(|| anyhow!("tensor {name} not in model.safetensors.index.json"))?.to_string();
    let (base, hdr) = read_header(&model_dir.join(&shard))?;
    let t = hdr.get(name).ok_or_else(|| anyhow!("tensor {name} missing in {shard}"))?;
    let off = t["data_offsets"][0].as_u64().ok_or_else(|| anyhow!("{name}: no data_offsets"))?;
    Ok(TensorRef {
        name: name.to_string(),
        shard,
        dtype: t["dtype"].as_str().unwrap_or("?").to_string(),
        shape: t["shape"].as_array().map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0)).collect()).unwrap_or_default(),
        abs_offset: base + off,
    })
}

/// Engram config of the checkpoint (config.json `engram_*` keys, top level or under `text_config`).
pub struct SourceConfig {
    pub geometry: Geometry,
    pub layers: Vec<SourceLayer>,
}

pub fn load(model_dir: &Path) -> Result<SourceConfig> {
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(model_dir.join("config.json")).context("config.json")?)?;
    let c = if cfg.get("engram_layer_ids").is_some() { &cfg } else { &cfg["text_config"] };
    let u = |k: &str| c[k].as_u64().ok_or_else(|| anyhow!("config.json: missing {k}"));
    let ids: Vec<u32> = c["engram_layer_ids"].as_array().ok_or_else(|| anyhow!("no engram_layer_ids"))?.iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let ne: Vec<u64> = c["engram_num_embeddings"].as_array().ok_or_else(|| anyhow!("no engram_num_embeddings"))?.iter().map(|v| v.as_u64().unwrap()).collect();
    if u("engram_head_dim")? as usize != VALUE_BYTES {
        bail!("engram_head_dim {} != {VALUE_BYTES}", c["engram_head_dim"]);
    }
    let geometry = Geometry::compute(&ids, u("engram_max_ngram_size")? as usize, u("engram_n_heads")? as usize, u("engram_vocab_size")?, &ne);
    geometry.check_tiles().map_err(|e| anyhow!(e))?;
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(model_dir.join("model.safetensors.index.json"))?)?;
    let mut layers = vec![];
    for (li, id) in ids.iter().enumerate() {
        let weight = tensor_ref(model_dir, &index, &format!("layers.{id}.engram.embed.weight"))?;
        let scale = tensor_ref(model_dir, &index, &format!("layers.{id}.engram.embed.scale"))?;
        if weight.dtype != "F8_E4M3" || scale.dtype != "F8_E8M0" || weight.shape != [ne[li], VALUE_BYTES as u64] || scale.shape != [ne[li], SCALE_BYTES as u64] {
            bail!("layer {id}: unexpected Engram table dtype/shape: {:?} {:?} {:?} {:?}", weight.dtype, weight.shape, scale.dtype, scale.shape);
        }
        layers.push(SourceLayer { layer_id: *id, weight, scale });
    }
    Ok(SourceConfig { geometry, layers })
}

/// Open shard file handles for the tables.
pub struct SourceFiles {
    pub weight: Vec<File>,
    pub scale: Vec<File>,
}
impl SourceFiles {
    pub fn open(model_dir: &Path, layers: &[SourceLayer]) -> Result<Self> {
        let o = |t: &TensorRef| -> Result<File> {
            let p: PathBuf = model_dir.join(&t.shard);
            File::open(&p).with_context(|| format!("open {}", p.display()))
        };
        Ok(SourceFiles { weight: layers.iter().map(|l| o(&l.weight)).collect::<Result<_>>()?, scale: layers.iter().map(|l| o(&l.scale)).collect::<Result<_>>()? })
    }
}

/// Read source rows [row0, row0+n) of layer `li` into `out` as 264-byte records (value then scale).
pub fn read_records(files: &SourceFiles, layers: &[SourceLayer], li: usize, row0: u64, n: usize, wbuf: &mut Vec<u8>, sbuf: &mut Vec<u8>, out: &mut Vec<u8>) -> Result<()> {
    wbuf.resize(n * VALUE_BYTES, 0);
    sbuf.resize(n * SCALE_BYTES, 0);
    files.weight[li].read_exact_at(wbuf, layers[li].weight.abs_offset + row0 * VALUE_BYTES as u64)?;
    files.scale[li].read_exact_at(sbuf, layers[li].scale.abs_offset + row0 * SCALE_BYTES as u64)?;
    out.clear();
    out.reserve(n * RECORD_BYTES);
    for i in 0..n {
        out.extend_from_slice(&wbuf[i * VALUE_BYTES..(i + 1) * VALUE_BYTES]);
        out.extend_from_slice(&sbuf[i * SCALE_BYTES..(i + 1) * SCALE_BYTES]);
    }
    Ok(())
}

pub fn fadvise_dontneed(f: &File, off: u64, len: u64) {
    use std::os::fd::AsRawFd;
    unsafe {
        libc::posix_fadvise(f.as_raw_fd(), off as i64, len as i64, libc::POSIX_FADV_DONTNEED);
    }
}
