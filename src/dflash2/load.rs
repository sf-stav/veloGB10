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

// ---------------------------------------------------------------------------
// PLAN/25 §1a — the NVFP4 weight-only artifact (draft round-time cuts).
//
// A baked dir (`--df2-bake-nvfp4 <src> <dst>`) holds:
//   * `nvfp4.safetensors` — one packed triple per quantized linear, the same
//     compressed-tensors convention the trunk quantizer emits:
//     `{name}.weight_packed` U8 [M*K/2] (E2M1 nibbles), `{name}.weight_scale`
//     U8 [M,K/16] (E4M3 block scales), `{name}.weight_global_scale` F32 [1].
//   * `model.safetensors` — the tensors that stay hi-prec (selector codebooks +
//     hidden_projection, all norms, conv base_kernels) PLUS the BF16 twins the
//     prompt-prime path needs (`fc.weight`, per-layer `k_proj`/`v_proj`:
//     `gemm_mma_fp4_b` is N≤16 and prime runs M up to 8192, so prime keeps the
//     bf16 `gemm_tiled_b` path — 367 MB, the only duplicated bytes).
//   * `config.json` — the original plus `df2_quant: "nvfp4"`,
//     `df2_quant_source_sha256` (the baked-from pin) and `df2_quant_recipe`.
// Quantization is weight-only RTN; selector codebooks/hp stay BF16 per the plan
// (the walk gathers them row-wise — quantizing saves memory, not bandwidth, and
// spends draft quality to do it).
// ---------------------------------------------------------------------------

/// One packed NVFP4 tensor exactly as stored in `nvfp4.safetensors` (pre-repack;
/// `repack_nvfp4_mma` runs at load, mirroring the trunk's own flow).
#[derive(Clone)]
pub struct PackedNvfp4 {
    pub name: String,
    pub qweight: Vec<u8>,
    pub scales: Vec<u8>,
    pub global_scale: f32,
    pub m: usize,
    pub k: usize,
}

/// One FP8-E4M3 tensor exactly as stored in `fp8.safetensors` (pre-repack;
/// `repack_fp8_mma` runs at load — the trunk's own DSV4-attention flow). 8 bits +
/// one f32 scale PER ROW (multiply on dequant; `gemm_mma_fp8_b` indexes `rs[m]`).
#[derive(Clone)]
pub struct PackedFp8 {
    pub name: String,
    pub qweight: Vec<u8>,
    pub row_scale: Vec<f32>,
    pub m: usize,
    pub k: usize,
}

/// The per-tensor payload of a baked artifact — which weight-only format the
/// sidecar carries.
pub enum QuantTensor {
    Nvfp4(PackedNvfp4),
    Fp8(PackedFp8),
}

impl QuantTensor {
    pub fn name(&self) -> &str {
        match self { QuantTensor::Nvfp4(p) => &p.name, QuantTensor::Fp8(p) => &p.name }
    }
}

/// The quantized side of a baked artifact (paired with [`load`]'s hi-prec weights).
pub struct QuantArtifact {
    pub tensors: Vec<QuantTensor>,
    pub source_sha256: String,
    pub recipe: String,
}

/// The linear weights §1a quantizes (everything the per-step block pass streams).
/// Selector/codebook/norm/base_kernel tensors are NOT here (stay BF16 in
/// `model.safetensors`).
pub fn quantized_inventory() -> Vec<(String, usize, usize)> {
    let mut v = Vec::new();
    v.push(("fc.weight".to_string(), 5120, 25600));
    for i in 0..crate::dflash2::N_LAYERS {
        let p = move |s: &str| format!("layers.{i}.{s}");
        v.push((p("self_attn.q_proj.weight"), 4096, 5120));
        v.push((p("self_attn.k_proj.weight"), 1024, 5120));
        v.push((p("self_attn.v_proj.weight"), 1024, 5120));
        v.push((p("self_attn.o_proj.weight"), 5120, 4096));
        v.push((p("mlp.gate_proj.weight"), 17408, 5120));
        v.push((p("mlp.up_proj.weight"), 17408, 5120));
        v.push((p("mlp.down_proj.weight"), 5120, 17408));
        v.push((p("attention_conv.kernel_projection.weight"), 1280, 5120));
        v.push((p("mlp_conv.kernel_projection.weight"), 1280, 5120));
    }
    v
}

/// The BF16 twins prime_window needs (gemm_mma_fp4_b is N≤16; prime is not).
pub fn bf16_twin_inventory() -> Vec<(String, usize, usize)> {
    let mut v = vec![("fc.weight".to_string(), 5120, 25600)];
    for i in 0..crate::dflash2::N_LAYERS {
        v.push((format!("layers.{i}.self_attn.k_proj.weight"), 1024, 5120));
        v.push((format!("layers.{i}.self_attn.v_proj.weight"), 1024, 5120));
    }
    v
}

/// Cheap baked-dir probe: config.json carries `df2_quant == "nvfp4" | "fp8"`.
pub fn is_baked(dir: &str) -> bool {
    let Ok(cfg_raw) = std::fs::read_to_string(Path::new(dir).join("config.json")) else { return false };
    let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&cfg_raw) else { return false };
    matches!(cfg.get("df2_quant").and_then(|v| v.as_str()), Some("nvfp4") | Some("fp8"))
}

/// The baked format this dir carries (None = not baked).
pub fn baked_fmt(dir: &str) -> Option<&'static str> {
    let cfg_raw = std::fs::read_to_string(Path::new(dir).join("config.json")).ok()?;
    let cfg = serde_json::from_str::<serde_json::Value>(&cfg_raw).ok()?;
    match cfg.get("df2_quant").and_then(|v| v.as_str())? {
        "nvfp4" => Some("nvfp4"),
        "fp8" => Some("fp8"),
        _ => None,
    }
}

/// Read a baked dir (NVFP4 or FP8 sidecar — `config.json: df2_quant` picks): the
/// hi-prec weights via [`load`] (its inventory check is RELAXED to the kept set —
/// see `load_kept`) plus the packed sidecar.
pub fn load_quantized(dir: &str) -> Result<(LoadedArtifact, QuantArtifact), anyhow::Error> {
    // The hi-prec side: same reader, same asserts, kept-subset inventory.
    let weights = load_kept(dir)?;
    let fmt = baked_fmt(dir).ok_or_else(|| anyhow::anyhow!("config.json: df2_quant is neither nvfp4 nor fp8 — not a baked artifact"))?;
    // The packed side.
    let path = Path::new(dir).join(if fmt == "fp8" { "fp8.safetensors" } else { "nvfp4.safetensors" });
    let buf = std::fs::read(&path).map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    let st = SafeTensors::deserialize(&buf).map_err(|e| anyhow::anyhow!("parse safetensors: {e}"))?;
    let cfg: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(Path::new(dir).join("config.json"))
            .map_err(|e| anyhow::anyhow!("read config.json: {e}"))?)?;
    let source_sha = cfg.get("df2_quant_source_sha256").and_then(|v| v.as_str())
        .unwrap_or("").to_string();
    let recipe = cfg.get("df2_quant_recipe").and_then(|v| v.as_str())
        .unwrap_or(if fmt == "fp8" { "fp8-rtw" } else { "nvfp4-rtw" }).to_string();

    let inv = quantized_inventory();
    let mut tensors = Vec::with_capacity(inv.len());
    for (name, m, k) in &inv {
        let get = |suffix: &str| -> Result<safetensors::tensor::TensorView, anyhow::Error> {
            let full = format!("{name}.{suffix}");
            let view = st.tensor(&full)
                .map_err(|e| anyhow::anyhow!("tensor {full}: {e} — is this a baked artifact?"))?;
            Ok(view)
        };
        if fmt == "fp8" {
            let qw = get("weight_packed")?;
            let rs = get("weight_row_scale")?;
            anyhow::ensure!(qw.dtype() == safetensors::Dtype::U8 && rs.dtype() == safetensors::Dtype::F32,
                "{name}: fp8 packed dtypes must be U8/F32");
            anyhow::ensure!(qw.shape() == [*m, *k], "{name}: qweight {:?} != [{m},{k}]", qw.shape());
            anyhow::ensure!(rs.shape() == [*m], "{name}: row_scale {:?} != [{m}]", rs.shape());
            tensors.push(QuantTensor::Fp8(PackedFp8 {
                name: name.clone(),
                qweight: qw.data().to_vec(),
                row_scale: bytemuck::cast_slice(rs.data()).to_vec(),
                m: *m, k: *k,
            }));
        } else {
            let qw = get("weight_packed")?;
            let sc = get("weight_scale")?;
            let gs = get("weight_global_scale")?;
            // weight_scale stores E4M3 bytes — the --quantize emit convention types it F8_E4M3
            // (bytewise identical to U8; accept both so hand-built sidecars also load).
            anyhow::ensure!(qw.dtype() == safetensors::Dtype::U8
                            && (sc.dtype() == safetensors::Dtype::U8 || sc.dtype() == safetensors::Dtype::F8_E4M3)
                            && gs.dtype() == safetensors::Dtype::F32,
                "{name}: packed dtypes must be U8/(U8|F8_E4M3)/F32");
            anyhow::ensure!(qw.shape() == [*m, k / 2], "{name}: qweight {:?} != [{m},{}]", qw.shape(), k / 2);
            anyhow::ensure!(sc.shape() == [*m, k / 16], "{name}: scales {:?} != [{m},{}]", sc.shape(), k / 16);
            anyhow::ensure!(gs.data().len() == 4, "{name}: global_scale must be one f32");
            tensors.push(QuantTensor::Nvfp4(PackedNvfp4 {
                name: name.clone(),
                qweight: qw.data().to_vec(),
                scales: sc.data().to_vec(),
                global_scale: f32::from_le_bytes(gs.data().try_into().unwrap()),
                m: *m, k: *k,
            }));
        }
    }
    anyhow::ensure!(tensors.len() == inv.len(), "packed tensor count {} != {}", tensors.len(), inv.len());
    Ok((weights, QuantArtifact { tensors, source_sha256: source_sha, recipe }))
}

/// The BF16 reader for the KEPT subset (hi-prec tensors + the prime twins).
/// Same per-tensor asserts as [`load`], but the inventory is the kept set, not
/// all 81 names.
fn load_kept(dir: &str) -> Result<LoadedArtifact, anyhow::Error> {
    let path = Path::new(dir).join("model.safetensors");
    let buf = std::fs::read(&path).map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    let file_size = buf.len() as u64;
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(&buf);
    let hexd: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();

    let st = SafeTensors::deserialize(&buf).map_err(|e| anyhow::anyhow!("parse safetensors: {e}"))?;

    // Kept = hi-prec (norms/base_kernels/codebooks/hp — everything in the 81-tensor
    // inventory EXCEPT the quantized linears) + the BF16 prime twins.
    let quant: std::collections::HashSet<String> =
        quantized_inventory().into_iter().map(|(n, _, _)| n).collect();
    let twins = bf16_twin_inventory();
    let mut kept: Vec<(String, Vec<usize>)> = Vec::new();
    for (n, s) in inventory() {
        if !quant.contains(&n) { kept.push((n.to_string(), s.to_vec())); }
    }
    for (n, m, k) in &twins {
        kept.push((n.clone(), vec![*m, *k]));
    }
    // `fc` appears in both loops (twin) — dedupe keeps one entry.
    kept.sort();
    kept.dedup();

    let mut file_names: Vec<String> = st.names().iter().map(|s| s.to_string()).collect();
    file_names.sort();
    let mut kept_names: Vec<String> = kept.iter().map(|(n, _)| n.clone()).collect();
    kept_names.sort();
    anyhow::ensure!(file_names == kept_names,
        "baked model.safetensors inventory mismatch (have {}; expected the kept {})",
        file_names.len(), kept_names.len());

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
        let data = view.data();
        Ok(data.chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect())
    };
    for (name, shape) in &kept {
        let v = bf16_to_f32(name, shape)?;
        assign(&mut weights, name, v)?;
    }
    Ok(LoadedArtifact {
        weights,
        n_tensors: kept.len(),
        n_params: kept.iter().map(|(_, s)| s.iter().map(|&d| d as u64).product::<u64>()).sum(),
        sha256: hexd,
        file_size,
        header_size: 0,
    })
}
