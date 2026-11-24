//! Generic `ShardSpecPlan` -> `crate::shard_plan::RankPlan` bridge.
//!
//! Replaces bespoke per-model shard-shipping code with a single declarative adapter that reads
//! safetensors headers in `model_dir`, resolves every tensor against a [`ShardSpecPlan`], and
//! emits standard [`crate::shard_plan::Piece`] segments (contiguous row slices or strided column
//! slices) into the **exact same** content-addressed `blobs/<sha256>` cache used by `cluster`.
//!
//! Existing model-family shippers (`Exl3`, `Nvfp4`, `Naive`, `Fp4l`, `Dsv41`) stay untouched;
//! new models (e.g. Inkling-2 or standard HF Llama/Qwen/MoE packs) can use this bridge directly.

use super::placement::Placement;
use super::plan::ShardSpecPlan;
use crate::shard_plan::{Piece, PlanFile, PlanSrc, RankPlan, ShardFormat, ShardInfo, PLAN_VERSION, SHARD_FILE};
use anyhow::{bail, ensure, Context, Result};
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Parsed safetensors entry: `(dtype, shape, abs_file_offset, byte_len)`.
#[derive(Clone, Debug)]
struct StEntry {
    dtype: String,
    shape: Vec<usize>,
    off: u64,
    len: u64,
}

fn dtype_bytes(dtype: &str) -> Result<usize> {
    Ok(match dtype {
        "F32" | "I32" | "U32" => 4,
        "BF16" | "F16" | "I16" | "U16" => 2,
        "F8_E4M3" | "F8_E5M2" | "I8" | "U8" | "BOOL" => 1,
        "F64" | "I64" | "U64" => 8,
        other => bail!("dist::ship: unsupported safetensors dtype '{other}'"),
    })
}

fn read_st_header(path: &Path) -> Result<BTreeMap<String, StEntry>> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut lb = [0u8; 8];
    f.read_exact(&mut lb)?;
    let hlen = u64::from_le_bytes(lb) as usize;
    ensure!(hlen > 0 && hlen <= 64 * 1024 * 1024, "invalid safetensors header len {hlen}");
    let mut hb = vec![0u8; hlen];
    f.read_exact(&mut hb)?;
    let data_base = 8u64 + hlen as u64;
    let v: serde_json::Value = serde_json::from_slice(&hb)?;
    let obj = v.as_object().context("safetensors header not an object")?;
    let mut out = BTreeMap::new();
    for (name, val) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = val.get("dtype").and_then(|x| x.as_str()).context("missing dtype")?.to_string();
        let shape: Vec<usize> = val
            .get("shape")
            .and_then(|x| x.as_array())
            .context("missing shape")?
            .iter()
            .map(|x| x.as_u64().context("shape dim") .map(|u| u as usize))
            .collect::<Result<_>>()?;
        let offs = val.get("data_offsets").and_then(|x| x.as_array()).context("missing data_offsets")?;
        ensure!(offs.len() == 2, "{name}: data_offsets len != 2");
        let a = offs[0].as_u64().context("data_offsets[0]")?;
        let b = offs[1].as_u64().context("data_offsets[1]")?;
        ensure!(b >= a, "{name}: data_offsets [{a}, {b}]");
        out.insert(name.clone(), StEntry { dtype, shape, off: data_base + a, len: b - a });
    }
    let _ = f.seek(SeekFrom::Start(0));
    Ok(out)
}

/// Discover all `.safetensors` files in `model_dir` (sorted deterministically).
fn list_safetensors(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for ent in std::fs::read_dir(model_dir).with_context(|| format!("read_dir {}", model_dir.display()))? {
        let p = ent?.path();
        if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            files.push(p);
        }
    }
    files.sort();
    Ok(files)
}

/// Build a [`RankPlan`] for `(rank, world)` from any declarative [`ShardSpecPlan`].
///
/// - Standard non-weight files (`config.json`, tokenizers, templates, `quantization_config.json`)
///   are shipped as `PlanSrc::Whole`.
/// - Every tensor in `*.safetensors` is resolved via `spec.resolve(name, &shape, rank, world)`:
///   - `!owned` (expert belonging to another rank): omitted on this rank;
///   - `Replicated`: whole tensor piece (`rows: 1, stride: 0`);
///   - `Shard { axis: 0 }`: contiguous slice of leading dimension (`rows: 1, stride: 0`);
///   - `Shard { axis: 1 }` (2-D `[rows, cols]`): strided column gather (`rows = shape[0]`, `stride = shape[1] * elem_bytes`).
/// - Emits one or more synthesized safetensors segment files (`seg/rank<r>-000.safetensors`, ...),
///   a generated `model.safetensors.index.json`, and `gb10_shard.json` (with `presliced` names).
pub fn build_rank_plan_from_spec(
    model_dir: &Path,
    spec: &ShardSpecPlan,
    format: ShardFormat,
    world: usize,
    rank: usize,
    max_segment_bytes: u64,
) -> Result<RankPlan> {
    ensure!(world >= 1 && rank < world, "dist::ship: invalid rank {rank}/{world}");
    let st_files = list_safetensors(model_dir)?;
    ensure!(!st_files.is_empty(), "dist::ship: no *.safetensors files in {}", model_dir.display());

    let mut files: Vec<PlanFile> = Vec::new();
    let mut presliced: Vec<String> = Vec::new();

    // Ship standard small metadata / tokenizer files whole.
    for fname in [
        "config.json",
        "quantization_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ] {
        let p = model_dir.join(fname);
        if p.is_file() {
            files.push(PlanFile { logical: fname.to_string(), src: PlanSrc::Whole(p) });
        }
    }

    // Slice every tensor according to `spec` and pack into bounded-size segments.
    let seg_cap = max_segment_bytes.max(1 << 20);
    let mut segments: Vec<Vec<Piece>> = vec![Vec::new()];
    let mut cur_bytes: u64 = 0;

    for st_path in &st_files {
        let hdr = read_st_header(st_path)?;
        for (name, e) in hdr {
            let res = spec.resolve(&name, &e.shape, rank, world)
                .with_context(|| format!("dist::ship resolving {name} {:?}", e.shape))?;
            if !res.owned {
                continue;
            }
            let eb = dtype_bytes(&e.dtype)? as u64;
            let piece = match (res.placement, res.range) {
                (Placement::Replicated, _) | (_, None) => Piece {
                    name: name.clone(),
                    dtype: e.dtype.clone(),
                    shape: e.shape.clone(),
                    src: st_path.clone(),
                    off: e.off,
                    len: e.len,
                    rows: 1,
                    stride: 0,
                },
                (Placement::Shard { axis: 0 }, Some(sr)) => {
                    let tail_elems: u64 = e.shape[1..].iter().map(|&d| d as u64).product::<u64>().max(1);
                    let row_bytes = tail_elems * eb;
                    presliced.push(name.clone());
                    Piece {
                        name: name.clone(),
                        dtype: e.dtype.clone(),
                        shape: res.local_shape,
                        src: st_path.clone(),
                        off: e.off + sr.start as u64 * row_bytes,
                        len: sr.len() as u64 * row_bytes,
                        rows: 1,
                        stride: 0,
                    }
                }
                (Placement::Shard { axis: 1 }, Some(sr)) => {
                    ensure!(
                        e.shape.len() == 2,
                        "dist::ship: axis-1 strided slicing requires 2D tensor, got {name} {:?}",
                        e.shape
                    );
                    let rows = e.shape[0] as u64;
                    let cols = e.shape[1] as u64;
                    let stride = cols * eb;
                    let col_off = sr.start as u64 * eb;
                    let local_row_bytes = sr.len() as u64 * eb;
                    presliced.push(name.clone());
                    Piece {
                        name: name.clone(),
                        dtype: e.dtype.clone(),
                        shape: res.local_shape,
                        src: st_path.clone(),
                        off: e.off + col_off,
                        len: rows * local_row_bytes,
                        rows,
                        stride,
                    }
                }
                (other, _) => bail!("dist::ship: unsupported placement {other:?} for {name}"),
            };

            if cur_bytes > 0 && cur_bytes + piece.len > seg_cap {
                segments.push(Vec::new());
                cur_bytes = 0;
            }
            cur_bytes += piece.len;
            segments.last_mut().unwrap().push(piece);
        }
    }

    let mut weight_map = serde_json::Map::new();
    let mut total_weight_bytes: u64 = 0;
    for (idx, ps) in segments.into_iter().enumerate() {
        if ps.is_empty() {
            continue;
        }
        let logical = format!("seg/rank{rank}-{idx:03}.safetensors");
        for p in &ps {
            weight_map.insert(p.name.clone(), serde_json::Value::String(logical.clone()));
            total_weight_bytes += p.len;
        }
        files.push(PlanFile { logical, src: PlanSrc::Segment(ps) });
    }

    let index_json = serde_json::to_vec_pretty(&serde_json::json!({
        "metadata": { "total_size": total_weight_bytes },
        "weight_map": weight_map,
    }))?;
    files.push(PlanFile {
        logical: "model.safetensors.index.json".into(),
        src: PlanSrc::Bytes(index_json),
    });

    let info = ShardInfo {
        plan_version: PLAN_VERSION,
        format: format.name().into(),
        world,
        rank,
        deal: format!("dist:{}", spec.family),
        source_model: model_dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".into()),
        presliced: presliced.clone(),
    };
    files.push(PlanFile {
        logical: SHARD_FILE.into(),
        src: PlanSrc::Bytes(serde_json::to_vec_pretty(&info)?),
    });

    Ok(RankPlan {
        format,
        world,
        rank,
        deal: format!("dist:{}", spec.family),
        files,
        presliced,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard_plan::write_segment;

    fn write_test_safetensors(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut hdr = serde_json::Map::new();
        let mut off = 0u64;
        for (name, dt, shape, bytes) in tensors {
            hdr.insert(
                (*name).to_string(),
                serde_json::json!({"dtype": dt, "shape": shape, "data_offsets": [off, off + bytes.len() as u64]}),
            );
            off += bytes.len() as u64;
        }
        let mut j = serde_json::to_vec(&serde_json::Value::Object(hdr)).unwrap();
        while j.len() % 8 != 0 { j.push(b' '); }
        let mut out = (j.len() as u64).to_le_bytes().to_vec();
        out.extend(j);
        for (_, _, _, bytes) in tensors {
            out.extend(bytes);
        }
        std::fs::write(path, out).unwrap();
    }

    fn read_tensor_bytes(seg_bytes: &[u8], name: &str) -> (Vec<usize>, Vec<u8>) {
        let hlen = u64::from_le_bytes(seg_bytes[..8].try_into().unwrap()) as usize;
        let v: serde_json::Value = serde_json::from_slice(&seg_bytes[8..8 + hlen]).unwrap();
        let t = v.get(name).unwrap();
        let shape: Vec<usize> = t["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let a = t["data_offsets"][0].as_u64().unwrap() as usize;
        let b = t["data_offsets"][1].as_u64().unwrap() as usize;
        (shape, seg_bytes[8 + hlen + a..8 + hlen + b].to_vec())
    }

    #[test]
    fn generic_spec_ship_slices_col_row_replicated_and_reconstructs_exact() {
        let tmp = std::env::temp_dir().join(format!("dist_ship_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("config.json"), b"{\"model_type\":\"test\"}").unwrap();

        // Q [512, 256] F32 (col-sharded dim 0), O [256, 512] F32 (row-sharded dim 1), norm [256] F32 (replicated)
        let mk = |n: usize, seed: u32| -> Vec<u8> {
            (0..n).flat_map(|i| ((i as u32).wrapping_mul(2654435761) ^ seed).to_le_bytes()).collect()
        };
        let q_full = mk(512 * 256, 1);
        let o_full = mk(256 * 512, 2);
        let norm_full = mk(256, 3);
        write_test_safetensors(
            &tmp.join("model.safetensors"),
            &[
                ("model.layers.0.self_attn.q_proj.weight", "F32", vec![512, 256], q_full.clone()),
                ("model.layers.0.self_attn.o_proj.weight", "F32", vec![256, 512], o_full.clone()),
                ("model.layers.0.input_layernorm.weight", "F32", vec![256], norm_full.clone()),
            ],
        );

        let spec = ShardSpecPlan::dense_transformer(128);
        let world = 4usize;
        let mut q_recon = Vec::new();
        let mut o_rows: Vec<Vec<u8>> = vec![Vec::new(); 256];
        let mut per_rank_bytes = Vec::new();

        for r in 0..world {
            let rp = build_rank_plan_from_spec(&tmp, &spec, ShardFormat::Naive, world, r, 64 << 20).unwrap();
            let seg = rp.files.iter().find_map(|f| match &f.src {
                PlanSrc::Segment(ps) => Some(ps),
                _ => None,
            }).unwrap();
            let mut buf = Vec::new();
            write_segment(seg, &mut buf).unwrap();
            per_rank_bytes.push(buf.len());

            let (qs, qb) = read_tensor_bytes(&buf, "model.layers.0.self_attn.q_proj.weight");
            assert_eq!(qs, vec![128, 256]);
            q_recon.extend_from_slice(&qb);

            let (os, ob) = read_tensor_bytes(&buf, "model.layers.0.self_attn.o_proj.weight");
            assert_eq!(os, vec![256, 128]);
            for row in 0..256 {
                o_rows[row].extend_from_slice(&ob[row * 128 * 4..(row + 1) * 128 * 4]);
            }

            let (ns, nb) = read_tensor_bytes(&buf, "model.layers.0.input_layernorm.weight");
            assert_eq!(ns, vec![256]);
            assert_eq!(nb, norm_full);
        }

        assert_eq!(q_recon, q_full, "col-sharded Q_proj must reconstruct byte-for-byte");
        let o_recon: Vec<u8> = o_rows.into_iter().flatten().collect();
        assert_eq!(o_recon, o_full, "row-sharded O_proj (strided gather) must reconstruct byte-for-byte");
        // Each rank shipped ~1/4 of Q+O (~263 KB instead of ~1.05 MB)
        let full_w = q_full.len() + o_full.len() + norm_full.len();
        assert!(per_rank_bytes[0] < full_w / 3, "rank 0 segment {} B should be ~1/4 of {} B", per_rank_bytes[0], full_w);

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
