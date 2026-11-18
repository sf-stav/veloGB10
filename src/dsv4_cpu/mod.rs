//! DSV4 CPU reference model (Lane B owns this file).
//!
//! Plain-Rust reimplementation of the bundle reference forward
//! (`/mnt/models/DeepSeek-V4-Flash-DSpark/inference/model.py`) following
//! DEEPSEEK_V4_PORT.md §B exactly, feeding on `crate::dsv4_load` (Lane A's loader).
//! Replays the oracle npz inputs and emits per-piece outputs for
//! `scripts/dsv4_diff.py` (Lane C).
//!
//! # Numerics policy (what is bit-faithful vs tolerance-level)
//!
//! Bit-faithful by construction (integer/bit-trick math, §C.1–2): the QAT-sim
//! round-trips (`act_quant_sim` / `fp4_act_quant_sim`), the UE8M0 pow2 scale
//! computation, the E4M3/E2M1 RNE casts (incl. sign-of-zero and ties-to-even),
//! the fp8/fp4 code extraction for GEMM activations, all index-list helpers
//! (pure functions), the deterministic top-k (§12.B.2), and the Sinkhorn
//! iteration *order* (§B.8).
//!
//! Tolerance-level (vs the torch/tilelang oracle): anything involving GEMM or
//! torch reduction accumulation *order* (cuBLAS/tilelang reduction trees vs our
//! sequential f32 dots) and transcendental ulps (exp/sigmoid/pow/cos/sin between
//! CUDA libm and the host libm). The Hadamard rotation is the oracle's fp32
//! matmul emulation (single rounding) — same structure, order-tolerance only.
//!
//! Activations are stored as `Vec<f32>` holding **bf16-rounded values** at every
//! point the reference materializes a bf16 tensor (`bf16::from_f32` RNE
//! round-trip, `bf()`). This is value-identical to the reference's bf16 storage
//! and keeps the code in one dtype; the two places the reference does *not*
//! float first (the weight-free per-head q rescale) are reproduced with
//! per-op bf16 roundings exactly.
//!
//! Map of this directory (every public item keeps its `crate::dsv4_cpu::` path via `pub use self::<sub>::*`):
//! - `mod.rs` (this file): the shared imports.
//! - `numerics.rs`: bf16 rounding, E4M3 / E2M1 RNE casts, UE8M0 scale tricks, QAT-sim round-trips, parallel row helpers, GEMMs.
//! - `ops.rs`: RoPE, Hadamard rotation, sparse attention, RMSNorm, hyper-connection math, top-k, router / expert, weight-take helpers, index lists.
//! - `compressor.rs`: the compressor state machine (B.5) and the indexer (B.6).
//! - `attn_moe.rs`: the attention (B.1-B.4) and MoE (B.9) forwards.
//! - `block.rs`: the block forward (B.8), the DSpark stages (B.10) and the replay-driver piece outputs.
//! - `load.rs`: Dsv4Layer -> CpuLayer extraction and the packed-cache dequant oracle helpers.

use anyhow::{anyhow, bail, Result};
use half::bf16;
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::dsv4_load::{Dsv4Config, Dsv4Layer, HostTensor, LayerKind};
use crate::quant::{e2m1_to_f32, e4m3_to_f32, Nvfp4Tensor};

mod numerics;
mod ops;
mod compressor;
mod attn_moe;
mod block;
mod load;

pub use self::numerics::*;
pub use self::ops::*;
pub use self::compressor::*;
pub use self::attn_moe::*;
pub use self::block::*;
pub use self::load::*;
