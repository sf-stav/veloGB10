//! DSV4 Phase-3 lane 3A — one SWA trunk layer (layer 0) end-to-end on GPU.
//!
//! Assembly target: the G1-proven CPU reference (`dsv4_cpu::block_forward` with
//! `LayerKind::Swa`), diffed against oracle `/mnt/models/dsv4-oracle-v2/dsv4_swa.npz`
//! (historical — the oracle-v2 path was deleted with the obsolete model; 0731
//! regeneration lives under /tmp/dsv4-0731-ref or fresh exports from dsv4_ref.py).
//! Per DSV4_PHASE3_HANDOFF.md §6 lane 3A, the trunk `Block.forward` per sublayer is
//! `hc_pre → rms_norm → sublayer → hc_post`, twice (attn + ffn):
//!
//! ```text
//! attn: hc_pre → rmsnorm(attn_norm) →
//!         fp8_bsb(wq_a) → rmsnorm(q_norm) → fp8_bsb(wq_b) → per-head rescale (bit-exact)
//!         → rope_last(q) ∥ fp8_bsb(wkv) → rmsnorm(kv_norm) → rope_last(kv)
//!         → kv_sim_g64_strided(kv[..., :448]) → ring-write (write-before-attention)
//!         → gather_attn(window list, sink denominator-only, 512^-0.5) → rope_last(o, inverse)
//!         → olo_proj (grouped-LoRA O, bf16 wo_a) → fp8_bsb(wo_b)
//!       → hc_post
//! ffn:  hc_pre → rmsnorm(ffn_norm) → router (score → tid2eid hash [layer 0] → weights;
//!         runs on the UN-simmed x) → shared expert (fp8_bsb(w1;w3) → swiglu_clamp_shared
//!         → fp8_bsb(w2)) + routed experts (G2 runtime, 16-row chunks — the N=1/grouped
//!         bitwise-invariant regime) → add_residual_b → hc_post
//! ```
//!
//! Stream invariant (AGENTS §2.1): every spine launch runs on the blocking compute stream;
//! the G2 MoE runtime launches on the device default (legacy NULL) stream, which the blocking
//! stream synchronizes with in both directions. `dev.synchronize()` brackets the MoE runtime
//! calls in this validation-path assembly (belt and braces; integration re-wires streams).
//!
//! Batch-invariance: every per-row kernel computes row-local math independent of `s`;
//! the fp8_bsb GEMM is G2-proven N-invariant (col-0 bitwise at N=1..16 + full-prefix), so the
//! ≤16-row chunking used here is bitwise-identical to any width-≤16 decomposition; the MoE
//! runtime is G2-proven N=1-vs-grouped bitwise. Decode row ≡ same row inside a 16-wide verify.
//!
//! Map of this directory (`Dsv4AttnRuntime`'s impl is cut by phase; every item keeps its `crate::dsv4_attn::` path):
//! - `mod.rs` (this file): kernel-name tables, weight / layer / state / snapshot types and the runtime struct.
//! - `setup.rs`: runtime construction, layer / MTP-stage upload and attention-state allocation.
//! - `ops.rs`: building-block ops: activation quant, FP8 GEMMs, TP all-reduce, rmsnorm, hc_pre/post/head, LM head.
//! - `attn.rs`: `attn_forward` (SWA / CSA / HCA trunk attention sublayer, one long function kept whole).
//! - `dspark.rs`: the DSpark stage attention (warm, forward, capture, device variants).
//! - `moe.rs`: MoE sublayer pieces and the `block_forward*` entry points.

use anyhow::{anyhow, Context, Result};
use cudarc::driver::result;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig,
};
use cudarc::nvrtc::Ptx;
use half::bf16;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::dsv4_comp::{CompKernels, CompSpec, DevRope, GpuCompressor, GpuIndexer};
use crate::dsv4_gpu::{self, Dsv4Kernels};
use crate::dsv4_launch;
use crate::gpu::{self, Dsv4MoeGpu, MoeGroupedScratch};
use crate::{dsv4_cpu, dsv4_load, dsv4_moe, quant};

/// bf16 device buffer (mirrors dsv4_gpu::B).
pub type B = CudaSlice<bf16>;
/// f32 device buffer (mirrors dsv4_gpu::S).
pub type S = CudaSlice<f32>;

const GATHER_SMEM: u32 = 88320;
/// Fused hc_pre (T6): 25 stride-halving tree arrays × 768 threads × 4 B.
const HC_PRE_FUSED_SMEM: u32 = 25 * 768 * 4;

/// Spine kernels used from gpu_dsv4.ptx (raw driver path, blocking stream).
const SPINE_FUNCS: &[&'static str] = &[
    "dsv4_rmsnorm_b",
    "dsv4_rmsnorm_pair_b",
    "dsv4_rope_last_b",
    "dsv4_rope_pair_b",
    "dsv4_rope_q_inline_b",
    "dsv4_iota_b",
    "dsv4_act_quant_g128",
    "dsv4_gather_attn",
    "dsv4_fused_gather_b",
    "dsv4_hc_pre_rsqrt_b",
    "dsv4_hc_mixes_b",
    "dsv4_hc_split_sinkhorn",
    "dsv4_hc_collapse_b",
    "dsv4_hc_pre_fused_b",
    "dsv4_hc_post_b",
    "dsv4_hc_head_b",
    "dsv4_embed_b",
    "dsv4_router_score_b",
    "dsv4_router_tid2eid_b",
    "dsv4_router_weights_b",
    "dsv4_router_bias_add_b",
    "dsv4_topk",
    "dsv4_swiglu_clamp_shared",
    "dsv4_main_hidden_b",
    "dsv4_markov_gather_b",
];
/// Lane-3A kernels from gpu_dsv4_attn.ptx. Lane-3C adds the strided/compress/placement
/// kernels (CSA/HCA index-list construction — see kernels/gpu_dsv4_attn.cu §6-8).
const ATTN_FUNCS: &[&'static str] = &[
    "dsv4_attn_rescale_b",
    "dsv4_kv_sim_g64_strided",
    "dsv4_rescale_rope_sim_b",
    "dsv4_ring_write_b",
    "dsv4_window_idxs_b",
    "dsv4_olo_proj_b",
    "dsv4_window_idxs_strided_b",
    "dsv4_window_idxs_verify_b",
    "dsv4_dspark_draft_idxs_b",
    "dsv4_compress_idxs_b",
    "dsv4_idxs_place_b",
    "dsv4_olo_proj_tc_b",
    "dsv4_olo_proj_tc4_b",
];
/// gpu_batch.ptx functions (cudarc path): the FP8 GEMM, the G2 MoE kernels, residual add,
/// the bf16→fp32 LM-head GEMM (gemm_binv_f32_b, the hy_v3 enable_lm_head_fp32 twin —
/// DSV4's head.weight is bf16, logits fp32; batch-invariant by the same strided-k tree), AND the
/// TP=2 doorbell all-reduce pair (`tp_gate_copy_signal`/`tp_wait_add` — same two-kernel handshake
/// the Hy3 path uses, `gpu.rs:3195`; the DSV4 runtime launches them on its blocking compute stream).
const BATCH_FUNCS: &[&'static str] = &[
    "gemm_dsv4_fp8_bsb",
    "gemm_dsv4_fp8_bsb2",
    "gemm_dsv4_fp8_bsb2q",
    "gemm_dsv4_fp8_bsb1q",
    "gemm_dsv4_fp8_bsb_pf",
    "gemm_dsv4_fp8_bsb_pf2",
    "dsv4_olo_einsum_fp8_b",
    "gemm_binv_f32_b",
    "gemm_moe_mma_fp4",
    "gemm_moe_mma_fp4_x2",
    "gemm_moe_mma_fp4_u4",
    "gemm_moe_grouped_mma_fp4",
    "gemm_moe_grouped_mma_fp4_x2",
    "gemm_moe_grouped_mma_fp4_u4",
    "moe_count_b",
    "moe_offsets_b",
    "moe_scatter_b",
    "moe_tilemap_b",
    "moe_gather_x_b",
    "moe_combine_experts_b",
    "moe_combine_grouped_b",
    "add_residual_b",
    "tp_gate_copy_signal",
    "tp_wait_add",
    "dsv4_argmax_pair_b",
    "tp_wait_maxloc_b",
    "tp_wait_maxloc_g",
];
/// gpu_dsv4.ptx functions the G2 MoE runtime drives via cudarc (test-path contract).
const MOE_DFUNCS: &[&'static str] = &["dsv4_act_quant_sim_g128", "dsv4_swiglu_clamp"];

fn ceil256(n: usize) -> u32 {
    ((n + 255) / 256) as u32
}

/// Item 2.5 / §6-a resolution: the tolerance-class fast paths (wo_a fp8 einsum, compressor
/// pair) are the DEFAULT; `--exact-gemm` (CLI → --exact-gemm transport, shipped to the
/// node via TpConfig) selects the locked bit-exact kernels. Env-first, then the shipped
/// config, then the fast-path default — the node resolves the same value as the head.
pub fn exact_gemm_enabled() -> bool {
    crate::opts::var(crate::opt!("exact-gemm")).is_ok()
        || crate::tp::tp_config().map(|c| c.exact_gemm).unwrap_or(false)
}

/// Item 2.5 load-time quant of wo_a: fp8 e4m3 + UE8M0 128-block scales via the session-2
/// `quantize_fp8_bsb` pattern, groups as contiguous R-row bands (the einsum kernel's
/// per-head-group tiles). Asserts the fp8_bsb geometry contract (M,K % 128).
fn quantize_wo_a(dev: &Arc<CudaDevice>, wo_a: &[bf16], g: usize, r: usize, k: usize) -> Result<Fp8Weight> {
    let (wt, sb) = crate::quant::quantize_fp8_bsb(wo_a, g * r, k);
    Ok(Fp8Weight {
        wt: dev.htod_sync_copy(&wt)?,
        sb: dev.htod_sync_copy(&sb)?,
        m: g * r,
        k,
    })
}

/// Item 2.5 wo_a fast path: fp8-quantize the de-rotated attention output `o` ([s, G*K]
/// contiguous) and run the per-group fp8 einsum (`dsv4_olo_einsum_fp8_b`, pf2-class
/// schedule) into `oflat` [s_pad, G*R]. Tolerance-class: the fp8 operands differ from the
/// exact bf16 path's bits by construction — gated by rel-L2 bounds, never bitwise.
#[allow(clippy::too_many_arguments)]
fn olo_einsum_fast<X: dsv4_gpu::Dsv4Buf<bf16>, C: dsv4_gpu::Dsv4Buf<u8>>(
    rt: &Dsv4AttnRuntime,
    oflat: &mut X,
    o: &X,
    wq: &Fp8Weight,
    s: usize,
    g: usize,
    r: usize,
    gd: usize,
) -> Result<()> {
    let (oc, osa) = rt.quant_g128::<X, C>(o, s, g * gd)?;
    let nchunks = s.div_ceil(16).max(1);
    let gy8 = nchunks.div_ceil(8).max(1);
    let gx = (r / 32) * g;
    let gy = gy8.max((288usize.div_ceil(gx)).min(nchunks)) as u32; // pf2's 2-wave fill floor
    let f = &rt.bk["dsv4_olo_einsum_fp8_b"];
    let cfg = LaunchConfig {
        grid_dim: (gx as u32, gy, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (r_i, gd_i, s_i, g_i) = (r as i32, gd as i32, s as i32, g as i32);
    let ofv = oflat.view(0, s * g * r);
    let ov = o.view(0, s * g * gd);
    let ocv = oc.view(0, s * g * gd);
    let osav = osa.view(0, s * g * (gd / 128));
    unsafe {
        f.clone()
            .launch_on_stream(&rt.stream, cfg, (&ofv, &wq.wt, &wq.sb, &ocv, &osav, r_i, gd_i, s_i, g_i, 0u64))
            .map_err(|e| anyhow!("dsv4_olo_einsum_fp8_b s={s}: {e:?}"))?;
    }
    Ok(())
}

/// One FP8 block-scale weight on device (MMA-repacked codes + UE8M0 block scales).
pub struct Fp8Weight {
    pub wt: CudaSlice<u8>, // MMA-repacked [M, K]
    pub sb: CudaSlice<u8>, // UE8M0 [M/128, K/128]
    pub m: usize,
    pub k: usize,
}

/// CPU-side attention-compressor load weights (CSA/HCA). Held on the layer; uploaded
/// into a `GpuCompressor` at state construction. Mirrors `dsv4_cpu::CompressorWeights`.
pub struct CompLoad {
    pub spec: CompSpec,
    pub w: dsv4_cpu::CompressorWeights,
}

/// CPU-side indexer load weights (CSA only). The indexer's wq_b ships as fp8_e4m3 +
/// UE8M0 scales; we MMA-repack at load (matching the attention wq_b path) and upload
/// into a `GpuIndexer` at state construction.
pub struct IndexerLoad {
    pub comp: CompLoad,        // the indexer's own rotate compressor (ratio 4, hd 128)
    pub wq_b_wt: Vec<u8>,      // MMA-repacked fp8 codes [nh*ihd, qlr]
    pub wq_b_sb: Vec<u8>,      // UE8M0 [nh*ihd/128, qlr/128]
    pub weights_proj: Vec<f32>,// [nh, dim] bf16-valued f32 (§A.2)
}

/// Device-resident weights for one trunk layer (SWA/CSA/HCA — see module doc).
/// CSA/HCA additionally hold CPU-side compressor (+ indexer) load weights, uploaded
/// into `GpuCompressor` / `GpuIndexer` at `Dsv4AttnState::new_state` time.
pub struct Dsv4GpuLayer {
    pub kind: dsv4_load::LayerKind,
    pub wq_a: Fp8Weight,   // [1024, 4096]
    pub wq_b: Fp8Weight,   // [32768, 1024]
    pub wkv: Fp8Weight,    // [512, 4096]
    pub wo_b: Fp8Weight,   // [4096, 8192]
    pub sh_gu: Fp8Weight,  // shared expert [w1; w3] fused: [4096, 4096]
    pub sh_w2: Fp8Weight,  // shared expert w2: [4096, 2048]
    pub wo_a: B,           // bf16 [8*1024, 4096] (§F.2 load-time dequant)
    /// Item 2.5 fast-path copy of wo_a: fp8 e4m3 + UE8M0 128-block scales, MMA-repacked
    /// (quant::quantize_fp8_bsb at load), groups as contiguous R-row bands — the
    /// `dsv4_olo_einsum_fp8_b` operand. Always loaded; the --exact-gemm path ignores it.
    pub wo_a_q: Option<Fp8Weight>,
    pub q_norm: S,         // f32 [1024]
    pub kv_norm: S,        // f32 [512]
    pub attn_norm: S,      // f32 [4096]
    pub ffn_norm: S,       // f32 [4096]
    pub sink: S,           // f32 [64]
    pub hc_attn_fn: S,     // f32 [24, 16384]
    pub hc_attn_base: S,   // f32 [24]
    pub hc_attn_scale: S,  // f32 [3]
    pub hc_ffn_fn: S,
    pub hc_ffn_base: S,
    pub hc_ffn_scale: S,
    pub gate_w: S,                        // f32 [256, 4096] (bf16 upcast)
    pub tid2eid: Option<CudaSlice<i32>>,  // [vocab, 6] — hash layers 0..2
    pub gate_bias: Option<S>,             // [256] — layers >= 3
    pub moe: Dsv4MoeGpu,                  // 256 routed experts, NVFP4 MMA-repacked
    /// Attention compressor load weights (None for SWA). Uploaded into the state's
    /// `attn_compressor` at `new_state`.
    pub comp_load: Option<CompLoad>,
    /// Indexer load weights (CSA only). Uploaded into the state's `indexer`.
    pub idx_load: Option<IndexerLoad>,
}

/// KV + compressor state for one trunk layer (SWA/CSA/HCA).
///
/// Layout (mirrors `dsv4_cpu::AttnState`):
/// - **SWA**: `kv_cache` is `[window, head_dim]` (the 128-row ring); no compressor,
///   no indexer. Decode gathers over the ring.
/// - **CSA**: `kv_cache` is `[window + max_seq/4, head_dim]` — rows `[0..window]` are
///   the ring, rows `[window..]` are the attention-compressor cache tail (copied from
///   `attn_compressor.cache` after each compression — see `attn_forward` doc). Decode
///   gathers over the full `kv_cache`. Prefill gathers over `kv_attn_scratch`
///   (`[kv, compressor_cache]` concatenated, matching the CPU reference's temporary).
///   `indexer` owns its own rotate compressor + `[max_seq/4, 128]` cache (private to
///   the score path; never gathered).
/// - **HCA**: same as CSA minus the indexer; compressor ratio is 128 (non-overlapping).
pub struct Dsv4AttnState {
    /// Unified decode gather source. SWA: `[window, hd]`. CSA/HCA: `[window + max_comp_rows, hd]`
    /// (ring ++ compressor-cache tail). The compressor tail is refreshed each step via d2d
    /// from `attn_compressor.cache` (the compressor owns its own buffer — see module doc).
    pub kv_cache: B,
    /// Prefill gather scratch `[max_s + max_comp_rows, hd]` (CSA/HCA only). Holds
    /// `[current_kv, compressor_cache]` for the prefill gather (the CPU reference's
    /// temporary `kv_attn = kv.clone(); kv_attn.extend(kvc)`). SWA: unused (prefill
    /// gathers over the just-projected `kv` directly).
    pub kv_attn_scratch: Option<B>,
    /// Attention compressor (CSA/HCA). Owns its cache + frontier state.
    pub attn_compressor: Option<GpuCompressor>,
    /// CSA indexer. Owns its rotate compressor + indexer kv_cache.
    pub indexer: Option<GpuIndexer>,
}

/// Per-layer DSpark verify rollback snapshot (the full persistent attention state: the KV ring +
/// compressor tail, the attention-compressor frontier+cache, the CSA indexer frontier+cache).
/// D2D-copied on the compute stream before the verify forward; [`restore`](Self::restore) rewinds
/// to the pre-verify position so the committed prefix can be re-advanced (or the state fully
/// rewound for the forced-mismatch gate).
pub struct LayerVerifySnap {
    pub kv_cache: B,
    pub comp: Option<crate::dsv4_comp::CompFullSnapshot>,
    pub idx: Option<crate::dsv4_comp::CompFullSnapshot>,
}

impl Dsv4AttnState {
    /// Deep-copy the full persistent state (kv_cache + compressor + indexer) for DSpark verify
    /// rollback. The verify forward mutates all three; this snapshots the pre-verify state.
    pub fn snapshot_verify(
        &self,
        dev: &Arc<CudaDevice>,
        stream: &CudaStream,
    ) -> Result<LayerVerifySnap> {
        use cudarc::driver::result;
        let n = self.kv_cache.len();
        let kv = dev.alloc_zeros::<bf16>(n).map_err(|e| anyhow!("snap kv_cache alloc: {e}"))?;
        unsafe {
            result::memcpy_dtod_async(*kv.device_ptr(), *self.kv_cache.device_ptr(), n * 2, stream.stream)
                .map_err(|e| anyhow!("snap kv_cache dtod: {e}"))?;
        }
        let comp = match &self.attn_compressor {
            Some(c) => Some(c.snapshot_full(dev, stream)?),
            None => None,
        };
        let idx = match &self.indexer {
            Some(ix) => Some(ix.snapshot_full(dev, stream)?),
            None => None,
        };
        Ok(LayerVerifySnap { kv_cache: kv, comp, idx })
    }

    /// Rewind the persistent state to a [`LayerVerifySnap`] (D2D restore on the compute stream).
    pub fn restore_verify(
        &self,
        snap: &LayerVerifySnap,
        dev: &Arc<CudaDevice>,
        stream: &CudaStream,
    ) -> Result<()> {
        use cudarc::driver::result;
        let n = self.kv_cache.len();
        unsafe {
            result::memcpy_dtod_async(*self.kv_cache.device_ptr(), *snap.kv_cache.device_ptr(), n * 2, stream.stream)
                .map_err(|e| anyhow!("restore kv_cache dtod: {e}"))?;
        }
        if let (Some(c), Some(s)) = (&self.attn_compressor, &snap.comp) {
            c.restore_full(s, stream)?;
        }
        if let (Some(ix), Some(s)) = (&self.indexer, &snap.idx) {
            ix.restore_full(s, stream)?;
        }
        let _ = dev; // (dev reserved for a future host-fallback path)
        Ok(())
    }
}

/// Observable outputs of one `block_forward` (device-resident; the replay driver copies out).
/// Generic over the buffer family (GSlice for the graphed decode); the defaults keep the bare
/// `BlockOut` name working for the eager/prefill/verify instantiation.
pub struct BlockOut<X = B, F = S, I = CudaSlice<i32>> {
    pub y: X,                          // [s, hc*dim] new streams
    pub attn_out: X,                   // [s, dim]
    pub ffn_out: X,                    // [s, dim]
    pub router_w: F,                   // [s, topk]
    pub router_idx: I,                 // [s, topk]
    pub topk_idx: I,                   // [s, topk_t]
    pub topk_t: usize,
}

/// The lane runtime: streams, kernel modules, RoPE table. Kind-generic (one runtime
/// per layer — SWA loads the plain-θ RoPE table; CSA/HCA load the YaRN table). Lane 3C
/// adds the `CompKernels` module (gpu_dsv4_comp.ptx + the spine subset the compressor
/// and indexer use) for CSA/HCA.
pub struct Dsv4AttnRuntime {
    pub dev: Arc<CudaDevice>,
    pub stream: CudaStream,
    pub spine: Dsv4Kernels,
    pub attn: Dsv4Kernels,
    /// Lane-3B comp kernels (None for SWA — compressor/indexer aren't constructed).
    pub comp: Option<CompKernels>,
    pub bk: HashMap<String, CudaFunction>,
    pub df: HashMap<String, CudaFunction>,
    /// RoPE table for THIS runtime's primary kind (the lane-3A/3C single-kind path).
    /// `attn_forward` uses [`rope_for`](Self::rope_for) instead, which prefers `ropes`.
    pub rope: DevRope,
    /// All-kind RoPE tables (populated by [`new_multikind`]; empty for the lane `new`).
    /// The full-trunk `Dsv4GpuModel` uses this to dispatch SWA/CSA/HCA layers through one
    /// runtime — §B.1.3 (SWA plain θ vs CSA/HCA YaRN, force-disabled on SWA).
    pub ropes: HashMap<dsv4_load::LayerKind, DevRope>,
    pub window: usize,
    /// TP=2 doorbell context device pointer (0 = single-process / no all-reduce). Set by
    /// `Dsv4GpuModel::attach_tp`; when non-zero, `block_forward` takes the TP path (router →
    /// routed rank-local partial → `tp_all_reduce_bf16` → +shared → hc_post).
    pub tp_ctx_dptr: u64,
}

mod setup;
mod ops;
mod attn;
mod dspark;
mod moe;

