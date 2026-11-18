//! Dsv4GpuModel — the full DeepSeek-V4-Flash-DSpark trunk on GPU (G3 integration, §6 step 2).
//!
//! Wraps the lane-3-proven `Dsv4AttnRuntime` (SWA/CSA/HCA kind-generic — one runtime dispatches
//! all 43 layers) into the complete trunk:
//!   `embed → [layer 0..n-1: hc_pre → norm → attn → hc_post → hc_pre → norm → MoE → hc_post]
//!    → hc_head (sigmoid-only collapse) → final RMSNorm → LM head (bf16→fp32 logits)`
//! and exposes a per-step forward returning the last-row fp32 logits (the greedy target).
//!
//! DSpark stages (§B.10) are SKIPPED here — Phase 5. TP=2 sharding (§5) lands next; this is the
//! single-process correctness path (a slice of N layers fits one node; the full 43-layer / 167 GB
//! trunk needs TP=2 and is gated on the shard-at-load constructor, added in a follow-up commit).
//!
//! ## The two G3 gates this model backs
//! - **Head gate** (`forward_head`): load trunk-top only, run hc_head+norm+lm_head on the oracle's
//!   `dsv4_head.npz` `x` [s,4,4096], diff `collapsed`/`logits` vs the npz. Validates the NEW
//!   trunk-top code (`dsv4_hc_head_b` + `gemm_binv_f32_b` LM head) in isolation.
//! - **Trunk gate** (`forward`): embed → N layers → head on a short prompt, diff vs `dsv4_cpu`'s
//!   full-trunk chain. Proves the integration glue (embed, kind dispatch, position/state tracking,
//!   stream replication) before TP=2 sharding.
//!
//! Stream invariant (AGENTS §2.1): every launch rides `rt.stream` (the runtime's blocking compute
//! stream). The G2 MoE runtime still sync-brackets default-stream launches — promoted to the
//! blocking stream in the next commit (G3.3).
//!
//! Map of this directory (the `Dsv4GpuModel` impl is cut by phase; every item keeps its `crate::dsv4_model::` path):
//! - `mod.rs` (this file): the struct, `PREFILL_CHUNK` and the shared imports.
//! - `load.rs`: trunk-top / layer loaders, converted-artifact loading, `attach_tp`, `extract_trunk_top`.
//! - `forward.rs`: head, embed, streams / prefill / decode / graphed / capture / verify forwards, state snapshots.
//! - `prefix_cache.rs`: prefix-cached forwards, `PrefixCache` and its entries.

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use half::bf16;
use std::path::Path;
use std::sync::Arc;

use crate::dsv4_attn::{
    CompLoad, Dsv4AttnRuntime, Dsv4AttnState, Dsv4GpuLayer, Fp8Weight, IndexerLoad, B,
};
pub use crate::dsv4_attn::LayerVerifySnap;
use crate::dsv4_comp::CompSpec;
use crate::dsv4_convert::ArtFile;
use crate::dsv4_cpu::{self, CompressorWeights};
use crate::dsv4_gpu::S;
use crate::dsv4_load::{self, Dsv4Config, HostTensor};
use crate::dsv4_moe::Dsv4MoeHost;
use crate::gpu::{self, Dsv4MoeGpu, MoeGroupedScratch};

/// Prefill chunk size (§12.B.5 — multiple of 128 so chunk boundaries land on compressor-block
/// boundaries; also a multiple of the grouped-MoE N=16 tile). Prompts longer than this are
/// processed in chunks by [`Dsv4GpuModel::forward`], carrying the compressor frontier
/// (kv_state/score_state), indexer cache tail, and window ring across chunks in `states` —
/// bitwise-identical to one-shot prefill (the 3B compressor gate + the model-level
/// `--probe-dsv4 --chunked` gate). Serving caps `s_max` at this value, so ALL prefill
/// scratch is chunk-sized, not prompt-sized (the 200K OOM fix — DSV4_LONG_CONTEXT_1M §3).
pub const PREFILL_CHUNK: usize = 4096;

/// Full DSV4 trunk model (single-process; TP=2 lands in a follow-up).
pub struct Dsv4GpuModel {
    pub dev: Arc<CudaDevice>,
    pub cfg: Dsv4Config,
    pub rt: Dsv4AttnRuntime,
    /// Trunk layers 0..n_layers-1 (a slice for the probe; the full trunk is n_layers = cfg.n_layers).
    pub layers: Vec<Dsv4GpuLayer>,
    /// Per-layer attention state (KV ring + compressor/indexer caches), parallel to `layers`.
    pub states: Vec<Dsv4AttnState>,
    /// G2 grouped-MoE scratch (E16 metadata pipeline). p_max = 16 * topk (the ≤16-row chunk cap).
    pub scratch: MoeGroupedScratch,
    // ---- trunk top (§A.1) ----
    /// `embed.weight` [vocab, dim] bf16 (`ParallelEmbedding`).
    pub embed: B,
    /// `norm.weight` [dim] f32 — the final trunk RMSNorm (before hc_head? no — see forward_head).
    pub norm: S,
    /// `head.weight` [vocab, dim] bf16 (fp32-valued; the GEMM reads bf16 → fp32 logits, bit-identical
    /// to the reference's fp32 param read at half the bytes — §12.A.4).
    pub head: B,
    /// `hc_head_fn` [hc, hc*dim] = [4, 16384] f32 (final sigmoid-only collapse, §B.8).
    pub hc_head_fn: S,
    /// `hc_head_base` [hc] = [4] f32.
    pub hc_head_base: S,
    /// `hc_head_scale` [1] f32.
    pub hc_head_scale: S,
    /// TP=2 rank (set by attach_tp; -1 = single-process). Drives the R3.2 vocab-parallel
    /// maxloc head (`forward_next`): rank r computes logits rows [r*vocab/2, (r+1)*vocab/2).
    pub tp_rank: i32,
    /// DSpark selective rollback (R4): per-layer verify activations [layers][verify_width, dim],
    /// captured during `forward_verify_capture_main` so a rejection re-applies compressor and
    /// KV state for the committed prefix WITHOUT a full re-forward (ring writes from the
    /// verify are valid for committed positions; only compressor/indexer state and the
    /// committed ring rows need re-application). Lazily allocated on first verify.
    pub x_cap: Option<Vec<B>>,
    /// CUDA-graph decode (R3A.1/E2, `--graph=1`): per-fire-variant whole-forward graphs,
    /// lazily captured on the first decode token of each variant. None = eager (default).
    pub graphs: Option<crate::dsv4_graph::DecodeGraphs>,
}

mod load;
mod forward;
mod prefix_cache;

pub use self::prefix_cache::*;
