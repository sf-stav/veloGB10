//! The DFlash2 artifact loader — reads the 81-tensor BF16 safetensors (the REAL sha-pinned
//! checkpoint or a shape-identical synthetic), asserts the exact inventory / shapes / dtypes
//! (BF16-only) and an optional sha256 pin, and converts everything to host f32 for the oracle.
//! NO GPU upload in S2F (S3F owns device layout — do not guess it now).

use std::path::Path;

use half::bf16;
use safetensors::SafeTensors;

use crate::dflash2::oracle::{ConvWeights, Dflash2Weights, LayerWeights};
use crate::dflash2::{inventory, N_PARAMS, N_TENSORS};

/// A loaded (and validated) artifact + its host-f32 weights.
pub struct LoadedArtifact {
    pub weights: Dflash2Weights,
    pub n_tensors: usize,
    pub n_params: u64,
    pub sha256: String,
    pub file_size: u64,
    pub header_size: u64,
}

/// Read `dir/model.safetensors`, validate inventory/shapes/dtypes, optional sha256 pin, and
/// upcast every tensor to f32. Returns an error (not a soft skip) on any mismatch.
pub fn load(dir: &str, sha256_pin: Option<&str>) -> Result<LoadedArtifact, anyhow::Error> {
    let path = Path::new(dir).join("model.safetensors");
    let buf = std::fs::read(&path).map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    let file_size = buf.len() as u64;

    // Optional sha256 pin (full hex; the REAL artifact pin is `crate::dflash2::REAL_SHA256`).
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(&buf);
    let hexd: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if let Some(pin) = sha256_pin {
        anyhow::ensure!(
            hexd.eq_ignore_ascii_case(pin),
            "sha256 mismatch: file={hexd} pin={pin}"
        );
    }

    let st = SafeTensors::deserialize(&buf).map_err(|e| anyhow::anyhow!("parse safetensors: {e}"))?;

    // Exact name-set equality vs the 81-tensor inventory (sorted → deterministic comparison).
    let inv = inventory();
    let mut file_names: Vec<String> = st.names().iter().map(|s| s.to_string()).collect();
    file_names.sort();
    let mut inv_names: Vec<String> = inv.iter().map(|(n, _)| n.clone()).collect();
    inv_names.sort();
    anyhow::ensure!(
        file_names == inv_names,
        "inventory name mismatch (have {} tensors; expected {N_TENSORS})",
        file_names.len()
    );
    anyhow::ensure!(file_names.len() == N_TENSORS, "tensor count {} != {N_TENSORS}", file_names.len());

    // Per-tensor dtype (BF16-only) + exact shape, in FIXED inventory order.
    let mut weights = Dflash2Weights {
        layers: Vec::with_capacity(crate::dflash2::N_LAYERS),
        fc: Vec::new(),
        hidden_norm: Vec::new(),
        norm: Vec::new(),
        hidden_projection: Vec::new(),
        predecessor_codebook: Vec::new(),
        successor_codebook: Vec::new(),
    };

    let bf16_to_f32 = |name: &str, shape: &[usize]| -> Result<Vec<f32>, anyhow::Error> {
        let view = st.tensor(name).map_err(|e| anyhow::anyhow!("tensor {name}: {e}"))?;
        anyhow::ensure!(view.dtype() == safetensors::Dtype::BF16, "{name}: not BF16 ({:?})", view.dtype());
        anyhow::ensure!(view.shape() == shape, "{name}: shape {:?} != expected {shape:?}", view.shape());
        anyhow::ensure!(!name.contains("embed_tokens") && !name.contains("lm_head"),
            "{name}: embed/lm_head must NOT be in the checkpoint (borrowed from the target)");
        let data = view.data();
        Ok(data
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect())
    };

    let mut n_params: u64 = 0;
    for (name, shape) in &inv {
        n_params += shape.iter().map(|&d| d as u64).product::<u64>();
        let v = bf16_to_f32(name, shape)?;
        assign(&mut weights, name, v)?;
    }
    anyhow::ensure!(n_params == N_PARAMS, "param count {n_params} != {N_PARAMS}");

    Ok(LoadedArtifact {
        weights,
        n_tensors: N_TENSORS,
        n_params,
        sha256: hexd,
        file_size,
        header_size: file_size - n_params * 2,
    })
}

/// Load the logical 81-tensor DFlash2 inventory from a mixed MR-GPTQ artifact. Projection
/// families may be represented by the standard NVFP4 packed triple; they are dequantized here
/// only to build the oracle/prime mirror. The serving round separately uploads the packed bytes.
pub fn load_runtime(dir: &str) -> Result<LoadedArtifact, anyhow::Error> {
    let rd = crate::gptq::ShardReader::open(Path::new(dir))?;
    let inv = inventory();
    let mut weights = Dflash2Weights {
        layers: Vec::with_capacity(crate::dflash2::N_LAYERS),
        fc: Vec::new(), hidden_norm: Vec::new(), norm: Vec::new(),
        hidden_projection: Vec::new(), predecessor_codebook: Vec::new(), successor_codebook: Vec::new(),
    };
    let mut n_params = 0u64;
    for (name, shape) in &inv {
        n_params += shape.iter().map(|&d| d as u64).product::<u64>();
        let vals = if rd.metas.contains_key(name) {
            let (got, v) = rd.read_bf16(name)?;
            anyhow::ensure!(&got == shape, "{name}: shape {got:?} != {shape:?}");
            v
        } else {
            let stem = name.strip_suffix(".weight").ok_or_else(|| anyhow::anyhow!("missing tensor {name}"))?;
            let pn = format!("{stem}.weight_packed");
            let sn = format!("{stem}.weight_scale");
            let gn = format!("{stem}.weight_global_scale");
            let (pm, qw) = rd.read_bytes(&pn)?;
            let (sm, sc) = rd.read_bytes(&sn)?;
            let (_, gb) = rd.read_bytes(&gn)?;
            anyhow::ensure!(shape.len() == 2 && pm.shape == vec![shape[0], shape[1] / 2]
                && sm.shape == vec![shape[0], shape[1] / 16], "{stem}: malformed NVFP4 family");
            anyhow::ensure!(gb.len() == 4, "{gn}: expected one f32");
            let global_scale = f32::from_le_bytes(gb[..4].try_into().unwrap());
            crate::quant::dequantize_nvfp4(&crate::quant::Nvfp4Tensor {
                qweight: qw, scales: sc, global_scale, m: shape[0], k: shape[1],
            })
        };
        assign(&mut weights, name, vals.into_iter().map(|x| x.to_f32()).collect())?;
    }
    anyhow::ensure!(n_params == N_PARAMS, "param count {n_params} != {N_PARAMS}");
    let file_size = rd.metas.values().map(|m| m.off.1 - m.off.0).sum();
    Ok(LoadedArtifact {
        weights, n_tensors: N_TENSORS, n_params, sha256: "mixed-runtime".into(),
        file_size, header_size: file_size.saturating_sub(n_params * 2),
    })
}

/// Assign one f32 tensor into the weight struct by name (fixed-order, deterministic).
fn assign(w: &mut Dflash2Weights, name: &str, v: Vec<f32>) -> Result<(), anyhow::Error> {
    match name {
        "fc.weight" => w.fc = v,
        "hidden_norm.weight" => w.hidden_norm = v,
        "norm.weight" => w.norm = v,
        "candidate_selector.hidden_projection.weight" => w.hidden_projection = v,
        "candidate_selector.predecessor_codebook" => w.predecessor_codebook = v,
        "candidate_selector.successor_codebook" => w.successor_codebook = v,
        _ => {
            // layer tensors: `layers.{i}.{suffix}`
            let rest = name.strip_prefix("layers.").ok_or_else(|| anyhow::anyhow!("bad name {name}"))?;
            let (i_str, suffix) = rest.split_once('.').ok_or_else(|| anyhow::anyhow!("bad layer name {name}"))?;
            let i: usize = i_str.parse().map_err(|_| anyhow::anyhow!("bad layer index {name}"))?;
            while w.layers.len() <= i {
                w.layers.push(empty_layer());
            }
            let l = &mut w.layers[i];
            match suffix {
                "self_attn.q_proj.weight" => l.q_proj = v,
                "self_attn.k_proj.weight" => l.k_proj = v,
                "self_attn.v_proj.weight" => l.v_proj = v,
                "self_attn.o_proj.weight" => l.o_proj = v,
                "self_attn.q_norm.weight" => l.q_norm = v,
                "self_attn.k_norm.weight" => l.k_norm = v,
                "input_layernorm.weight" => l.input_ln = v,
                "post_attention_layernorm.weight" => l.post_ln = v,
                "mlp.gate_proj.weight" => l.gate_proj = v,
                "mlp.up_proj.weight" => l.up_proj = v,
                "mlp.down_proj.weight" => l.down_proj = v,
                "attention_conv.base_kernel" => l.attention_conv.base_kernel = v,
                "attention_conv.kernel_projection.weight" => l.attention_conv.kernel_projection = v,
                "mlp_conv.base_kernel" => l.mlp_conv.base_kernel = v,
                "mlp_conv.kernel_projection.weight" => l.mlp_conv.kernel_projection = v,
                _ => anyhow::bail!("unknown tensor {name}"),
            }
        }
    }
    Ok(())
}

fn empty_layer() -> LayerWeights {
    LayerWeights {
        q_proj: Vec::new(),
        k_proj: Vec::new(),
        v_proj: Vec::new(),
        o_proj: Vec::new(),
        q_norm: Vec::new(),
        k_norm: Vec::new(),
        input_ln: Vec::new(),
        post_ln: Vec::new(),
        gate_proj: Vec::new(),
        up_proj: Vec::new(),
        down_proj: Vec::new(),
        attention_conv: ConvWeights { base_kernel: Vec::new(), kernel_projection: Vec::new() },
        mlp_conv: ConvWeights { base_kernel: Vec::new(), kernel_projection: Vec::new() },
    }
}
