//! Deterministic synthetic support for DFlash2 (`--gen-dflash2-synth <dir>`) — writes a REAL
//! safetensors artifact SHAPE-IDENTICAL to the incoai/Qwen3.8-27B-DFlash2 checkpoint (81 tensors,
//! exactly 1,924,404,480 params, all BF16), plus `config.json` and a `SYNTHETIC_README.md` marker.
//! The REAL artifact is the primary S2F target; this generator exists for wiring tests.
//!
//! # Determinism (binding)
//!
//! Fixed seed ([`SYNTH_SEED`]), a single documented PRNG (xorshift64*), no system entropy, no
//! HashMap/map iteration in the generation order. Two runs → byte-identical file → identical
//! sha256. (Same algorithm family as `src/dspark/synth.rs`, INDEPENDENT seeds — R9: no
//! cross-family reuse.)
//!
//! # Golden-harness parity (binding)
//!
//! [`SyntheticTables`] (the target-borrowed embed/head rows) and the tap-hiddens generator are
//! ported EXACTLY by `tool_probe/dflash2_golden.py` (xorshift64* / splitmix64, u64 wraps) so the
//! vendor reference and this oracle see BIT-IDENTICAL inputs.
//!
//! # Init scheme (exact)
//!
//! * Linear weights (`q/k/v/o/gate/up/down_proj`, `fc`, `kernel_projection`, `hidden_projection`):
//!   `N(0, 1) * 1/√fan_in` (fan_in = `shape[1]`). Box–Muller normal in f64, two uniforms per value.
//! * Norm weights (every `*norm.weight`): small POSITIVE uniform `0.5 + 0.5·U[0,1)` — the PLAIN
//!   `w·x` qwen3 convention (DECISION F: trained norms are O(1)-centered, NOT zero-centered).
//! * `*_conv.base_kernel`: `N(0,1) * 1/√hidden` + 1.0 on the tap-0 diagonal-class entries is NOT
//!   attempted — plain `N(0,1) * 1/√hidden` for every entry (structure checks assert mechanics,
//!   not quality).
//! * Selector codebooks: `N(0,1) * 1/√256` (rank-256 latents O(1) vs the unary logits).

use std::path::Path;

use half::bf16;
use safetensors::tensor::TensorView;
use safetensors::Dtype;

use crate::dflash2::{inventory, N_PARAMS, N_TENSORS, SYNTH_SEED};

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

/// Deterministic on-the-fly generator for the target-borrowed embed/lm-head rows AND the
/// synthetic tap hiddens (none are in the 81-tensor checkpoint). Row `n` entries are uniform
/// `[-1, 1] * scale`, drawn from a per-(table, token) xorshift64* stream so any row is
/// reproducible without materializing the full table (DECISION O).
pub struct SyntheticTables {
    seed: u64,
}

impl SyntheticTables {
    pub const TABLE_EMBED: u64 = 0;
    pub const TABLE_HEAD: u64 = 1;
    /// The synthetic TAP hiddens table (seeded with `SYNTH_TAP_SEED` at the call site — the
    /// probe/golden construct a second `SyntheticTables` for it).
    pub const TABLE_TAPS: u64 = 2;

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
fn gen_tensor(rng: &mut SynthRng, name: &str, shape: &[usize]) -> Vec<u8> {
    let n: usize = shape.iter().product();
    let mut bytes = Vec::with_capacity(n * 2);
    if name.ends_with("norm.weight") {
        // PLAIN w·x convention (DECISION F) — O(1)-centered positive gains.
        for _ in 0..n {
            let w = 0.5 + 0.5 * rng.uniform();
            bytes.extend_from_slice(&bf16::from_f32(w).to_le_bytes());
        }
        return bytes;
    }
    let scale = if name.contains("codebook") {
        1.0f32 / (crate::dflash2::SELECTOR_RANK as f32).sqrt()
    } else {
        1.0f32 / (shape[shape.len() - 1] as f32).sqrt()
    };
    for _ in 0..n {
        let v = rng.normal() * scale;
        bytes.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
    }
    bytes
}

/// Generate the synthetic artifact into `dir` (created if absent). Deterministic for a fixed
/// seed. Asserts 81 tensors and exactly 1,924,404,480 params at generation time (FAIL LOUDLY).
pub fn generate(dir: &str, seed: u64) -> Result<GenSummary, anyhow::Error> {
    std::fs::create_dir_all(dir)
        .map_err(|e| anyhow::anyhow!("create {}: {e}", dir))?;

    let inv = inventory();
    anyhow::ensure!(inv.len() == N_TENSORS, "inventory has {} tensors, want {N_TENSORS}", inv.len());

    let mut rng = SynthRng::new(seed);
    let mut bufs: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::with_capacity(inv.len());
    let mut n_params: u64 = 0;
    for (name, shape) in &inv {
        let bytes = gen_tensor(&mut rng, name, shape);
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

    // config.json — mirror the REAL artifact's field names (config.json, verified 2026-08-19).
    let cfg = serde_json::json!({
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "hidden_size": crate::dflash2::HIDDEN,
        "num_hidden_layers": crate::dflash2::N_LAYERS,
        "num_attention_heads": crate::dflash2::NUM_HEADS,
        "num_key_value_heads": crate::dflash2::NUM_KV_HEADS,
        "head_dim": crate::dflash2::HEAD_DIM,
        "intermediate_size": crate::dflash2::INTER,
        "vocab_size": crate::dflash2::VOCAB,
        "rms_norm_eps": crate::dflash2::RMS_EPS,
        "max_position_embeddings": crate::dflash2::MAX_POSITIONS,
        "num_target_layers": 64,
        "is_causal": crate::dflash2::IS_CAUSAL,
        "layer_types": vec!["sliding_attention"; 5],
        "use_sliding_window": true,
        "sliding_window": crate::dflash2::SLIDING_WINDOW,
        "max_window_layers": 5,
        "tie_word_embeddings": false,
        "rope_parameters": {
            "rope_theta": crate::dflash2::ROPE_THETA,
            "rope_type": "default"
        },
        "dflash_config": {
            "block_size": crate::dflash2::block(),
            "conv_group_size": crate::dflash2::CONV_GROUP,
            "conv_kernel_size": crate::dflash2::CONV_KERNEL,
            "mask_token_id": crate::dflash2::MASK_TOKEN_ID,
            "selector_rank": crate::dflash2::SELECTOR_RANK,
            "selector_top_k": crate::dflash2::SELECTOR_TOP_K,
            "target_layer_ids": crate::dflash2::TAP_LAYERS
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
            "# SYNTHETIC DFlash2 artifact\n\n\
             This directory holds a **synthetic**, deterministic, license-clean weight artifact\n\
             that is SHAPE-IDENTICAL to the incoai/Qwen3.8-27B-DFlash2 checkpoint (81 tensors,\n\
             {N_PARAMS} params, all BF16). It is **NOT** the real checkpoint — generated by\n\
             `--gen-dflash2-synth` (seed {seed}). No embedding table and no lm_head, exactly like\n\
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

/// Convenience: the default generation target (gitignored `tool_probe/`).
pub fn default_dir() -> String {
    crate::opts::var(crate::opt!("dflash2-synth-dir"))
        .unwrap_or_else(|_| crate::dflash2::DEFAULT_SYNTH_DIR.to_string())
}

/// The default seed (documented constant).
pub fn default_seed() -> u64 {
    SYNTH_SEED
}
