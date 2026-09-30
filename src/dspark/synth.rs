//! Deterministic synthetic-weight generator (`--gen-dspark-synth <dir>`) — writes a REAL
//! safetensors artifact SHAPE-IDENTICAL to the RadixArk/Qwen3.8-27B-DSpark checkpoint (62 tensors,
//! exactly 1,359,284,737 params, all BF16), plus `config.json` and a `SYNTHETIC_README.md` marker.
//!
//! # Determinism (binding)
//!
//! Fixed seed ([`SYNTH_SEED`]), a single documented PRNG (xorshift64*), no system entropy, no
//! HashMap/map iteration in the generation order. Two runs → byte-identical file → identical
//! sha256. The safetensors crate itself sorts the header by tensor name (all BF16 → name order),
//! which is deterministic for this fixed name set.
//!
//! # Init scheme (exact — S3's activation-magnitude sanity depends on it)
//!
//! * Linear weights (`q/k/v/o/gate/up/down_proj`, `fc`, `confidence.weight`): `N(0, 1) * 1/√fan_in`
//!   where `fan_in` is the input dimension (`shape[1]`). Box–Muller normal in f64, drawn two
//!   uniforms per value from the xorshift64* stream; converted to BF16 with round-to-nearest-even.
//! * Norm weights (every `*.norm.weight`): small POSITIVE uniform `0.01 + 0.01·U[0,1)` so the
//!   qwen3 zero-centered `(1 + w)·x` form is ≈ 1 (DECISION F).
//! * `markov.W1.weight` (embedding) and `markov.W2.weight`: `N(0, 1) * 1/√256` so the rank-256
//!   bigram bias `W2 @ W1[tok]` is O(1) relative to `logits0`.
//! * `confidence.bias`: exactly `0.0`.
//!
//! The borrowed target-side embed/lm-head rows (NOT in the artifact) come from
//! [`SyntheticTables`] with an independent seed.

use std::path::Path;

use half::bf16;
use safetensors::tensor::TensorView;
use safetensors::Dtype;

use crate::dspark::{inventory, N_PARAMS, N_TENSORS, SYNTH_SEED};

/// xorshift64* (Vigna): the single documented PRNG for the whole generator. Deterministic; a
/// bijection on nonzero u64, so a nonzero seed never collapses to zero.
#[inline]
fn xorshift64s(mut x: u64) -> u64 {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// splitmix64 finalizer (used only to derive per-table/per-token row seeds for `SyntheticTables`).
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Deterministic PRNG stream over the artifact weights.
pub struct SynthRng {
    state: u64,
}

impl SynthRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = xorshift64s(self.state);
        self.state
    }

    /// Uniform `[0, 1)` from the top 24 bits.
    #[inline]
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32)
    }

    /// Standard normal via Box–Muller (f64 math, two uniforms per value).
    pub fn normal(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(f64::MIN_POSITIVE);
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

/// Deterministic on-the-fly generator for the target-borrowed embed/lm-head rows (the tensors the
/// 62-tensor checkpoint deliberately omits). Row `n` entries are uniform `[-1, 1] * scale`, drawn
/// from a per-(table, token) xorshift64* stream so any row is reproducible without materializing
/// the full `[vocab, hidden]` table. The embed/head seed is independent of the artifact seed so
/// re-rolling the artifact weights never changes the borrowed surface (DECISION O).
pub struct SyntheticTables {
    seed: u64,
}

impl SyntheticTables {
    pub const TABLE_EMBED: u64 = 0;
    pub const TABLE_HEAD: u64 = 1;

    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// One deterministic row of `n` elements, scaled by `scale`.
    pub fn row(&self, table: u64, token: u32, n: usize, scale: f32) -> Vec<f32> {
        let mut s = splitmix64(self.seed ^ table.rotate_left(17) ^ (token as u64).rotate_left(5));
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            s = xorshift64s(s);
            let u = (s >> 40) as f32 / ((1u64 << 24) as f32);
            out.push((u * 2.0 - 1.0) * scale);
        }
        out
    }
}

/// Summary of a generated artifact (for the probe/report).
#[derive(Clone, Debug)]
pub struct GenSummary {
    pub dir: String,
    pub sha256: String,
    pub n_tensors: usize,
    pub n_params: u64,
    pub file_size: u64,
    /// The safetensors header bytes: 8-byte length prefix + JSON (padded to 8).
    pub header_size: u64,
}

/// Generate one tensor's BF16 byte payload per the documented init scheme.
fn gen_tensor(rng: &mut SynthRng, name: &str, shape: &[usize], rank: usize) -> Vec<u8> {
    let n: usize = shape.iter().product();
    let mut bytes = Vec::with_capacity(n * 2);
    if name == "confidence_head.proj.bias" {
        bytes.extend_from_slice(&bf16::from_f32(0.0).to_le_bytes());
        return bytes;
    }
    if name.ends_with("norm.weight") {
        for _ in 0..n {
            let w = 0.01 + 0.01 * rng.uniform();
            bytes.extend_from_slice(&bf16::from_f32(w).to_le_bytes());
        }
        return bytes;
    }
    let scale = if name.starts_with("markov_head.") {
        1.0f32 / (rank as f32).sqrt()
    } else {
        1.0f32 / (shape[1] as f32).sqrt()
    };
    for _ in 0..n {
        let v = rng.normal() * scale;
        bytes.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
    }
    bytes
}

/// Generate the synthetic artifact into `dir` (created if absent). Deterministic for a fixed seed.
/// Asserts 62 tensors and exactly 1,359,284,737 params at generation time (FAIL LOUDLY otherwise).
pub fn generate(dir: &str, seed: u64) -> Result<GenSummary, anyhow::Error> {
    std::fs::create_dir_all(dir)
        .map_err(|e| anyhow::anyhow!("create {}: {e}", dir))?;

    let inv = inventory();
    anyhow::ensure!(inv.len() == N_TENSORS, "inventory has {} tensors, want {N_TENSORS}", inv.len());

    let rank = crate::dspark::MARKOV_RANK;
    let mut rng = SynthRng::new(seed);
    let mut bufs: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::with_capacity(inv.len());
    let mut n_params: u64 = 0;
    for (name, shape) in &inv {
        let bytes = gen_tensor(&mut rng, name, shape, rank);
        n_params += shape.iter().map(|&d| d as u64).product::<u64>();
        bufs.push((name.clone(), shape.clone(), bytes));
    }
    anyhow::ensure!(bufs.len() == N_TENSORS, "generated {} tensors, want {N_TENSORS}", bufs.len());
    anyhow::ensure!(
        n_params == N_PARAMS,
        "generated {n_params} params, want {N_PARAMS} (the reconciled count)"
    );

    let path = Path::new(dir).join("model.safetensors");
    let views: Vec<(String, TensorView)> = bufs
        .iter()
        .map(|(n, s, b)| {
            (
                n.clone(),
                TensorView::new(Dtype::BF16, s.clone(), b.as_slice())
                    .expect("TensorView (BF16, correct byte len)"),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, None, &path)
        .map_err(|e| anyhow::anyhow!("write {}: {e}", path.display()))?;

    // config.json — mirror the addendum's field names (Table A1). serde_json sorts keys → deterministic.
    let cfg = serde_json::json!({
        "architectures": ["DSparkDraftModel"],
        "auto_map": { "AutoModelForCausalLM": "dspark.DSparkDraftModel" },
        "model_type": "qwen3",
        "hidden_size": crate::dspark::HIDDEN,
        "num_hidden_layers": crate::dspark::N_LAYERS,
        "num_attention_heads": crate::dspark::NUM_HEADS,
        "num_key_value_heads": crate::dspark::NUM_KV_HEADS,
        "head_dim": crate::dspark::HEAD_DIM,
        "intermediate_size": crate::dspark::INTER,
        "vocab_size": crate::dspark::VOCAB,
        "rms_norm_eps": crate::dspark::RMS_EPS,
        "max_position_embeddings": crate::dspark::MAX_POSITIONS,
        "rope_theta": crate::dspark::ROPE_THETA,
        "rope_scaling": {
            "type": "yarn",
            "factor": crate::dspark::ROPE_FACTOR,
            "original_max_position_embeddings": crate::dspark::ORIG_CTX,
            "beta_fast": crate::dspark::BETA_FAST,
            "beta_slow": crate::dspark::BETA_SLOW,
            "mscale": 1.0,
            "mscale_all_dim": 1.0
        },
        "dspark_config": {
            "block_size": crate::dspark::BLOCK,
            "mask_token_id": crate::dspark::MASK_TOKEN_ID,
            "target_layer_ids": crate::dspark::TAP_LAYERS,
            "markov_rank": crate::dspark::MARKOV_RANK,
            "confidence_threshold": 0.5
        },
        "synthetic": true,
        "generator_seed": seed
    });
    std::fs::write(
        Path::new(dir).join("config.json"),
        serde_json::to_string_pretty(&cfg)?,
    )
    .map_err(|e| anyhow::anyhow!("write config.json: {e}"))?;

    std::fs::write(
        Path::new(dir).join("SYNTHETIC_README.md"),
        format!(
            "# SYNTHETIC DSpark artifact\n\n\
             This directory holds a **synthetic**, deterministic, license-clean weight artifact\n\
             that is SHAPE-IDENTICAL to the RadixArk/Qwen3.8-27B-DSpark checkpoint (62 tensors,\n\
             {N_PARAMS} params, all BF16). It is **NOT** the real checkpoint — generated by\n\
             `--gen-dspark-synth` (seed {seed}). No embedding table and no lm_head, exactly like\n\
             the real checkpoint: both are borrowed from the target at runtime.\n\n\
             Do not redistribute or use as a real model.\n"
        ),
    )
    .map_err(|e| anyhow::anyhow!("write SYNTHETIC_README.md: {e}"))?;

    // sha256 + sizes (streaming hash of the written file).
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    let mut f = std::fs::File::open(&path)?;
    std::io::copy(&mut f, &mut h)?;
    let hexd: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    let file_size = std::fs::metadata(&path)?.len();
    let data_size = n_params * 2; // BF16 = 2 bytes/param

    Ok(GenSummary {
        dir: dir.to_string(),
        sha256: hexd,
        n_tensors: bufs.len(),
        n_params,
        file_size,
        header_size: file_size - data_size,
    })
}

/// Convenience: the default generation target (outside the repo; never in git).
pub fn default_dir() -> String {
    crate::opts::var(crate::opt!("dspark-synth-dir")).unwrap_or_else(|_| crate::dspark::DEFAULT_SYNTH_DIR.to_string())
}

/// The default seed (documented constant).
pub fn default_seed() -> u64 {
    SYNTH_SEED
}
