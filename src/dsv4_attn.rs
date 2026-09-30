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

impl Dsv4AttnRuntime {
    /// Build the runtime: load all modules + the per-kind RoPE table (`positions` rows).
    /// `kind` selects the RoPE table (YaRN for CSA/HCA, plain for SWA) and whether the
    /// comp module is loaded (CSA/HCA only — SWA skips the load to stay 3A-byte-identical).
    pub fn new(dev: &Arc<CudaDevice>, kind: dsv4_load::LayerKind, positions: usize, cfg: &dsv4_load::Dsv4Config) -> Result<Self> {
        let stream = dsv4_gpu::blocking_compute_stream(dev);
        let spine = Dsv4Kernels::load(dev, SPINE_FUNCS).context("load gpu_dsv4.ptx (spine)")?;
        spine
            .set_dynamic_smem("dsv4_gather_attn", GATHER_SMEM)
            .context("set_dynamic_smem gather_attn")?;
        spine
            .set_dynamic_smem("dsv4_fused_gather_b", GATHER_SMEM)
            .context("set_dynamic_smem fused_gather_b")?;
        spine
            .set_dynamic_smem("dsv4_hc_pre_fused_b", HC_PRE_FUSED_SMEM)
            .context("set_dynamic_smem hc_pre_fused_b")?;
        let attn = Dsv4Kernels::load_module(dev, "src/ptx/gpu_dsv4_attn.ptx", ATTN_FUNCS)
            .context("load gpu_dsv4_attn.ptx (lane 3A/3C)")?;
        // CSA/HCA need the comp module (compressor + indexer kernels). SWA skips it
        // (no compressor/indexer constructed — stays 3A-byte-identical).
        let comp = if matches!(kind, dsv4_load::LayerKind::Csa | dsv4_load::LayerKind::Hca) {
            Some(CompKernels::load(dev).context("CompKernels::load (lane 3C)")?)
        } else {
            None
        };

        let bptx = Ptx::from_src(
            std::fs::read_to_string("src/ptx/gpu_batch.ptx").context("read gpu_batch.ptx")?,
        );
        dev.load_ptx(bptx, "gpu_batch", BATCH_FUNCS).context("load_ptx gpu_batch")?;
        let dptx = Ptx::from_src(
            std::fs::read_to_string("src/ptx/gpu_dsv4.ptx").context("read gpu_dsv4.ptx")?,
        );
        dev.load_ptx(dptx, "gpu_dsv4", MOE_DFUNCS).context("load_ptx gpu_dsv4 (moe)")?;
        let collect = |m: &str, names: &[&'static str]| -> HashMap<String, CudaFunction> {
            names
                .iter()
                .map(|n| (n.to_string(), dev.get_func(m, n).unwrap_or_else(|| panic!("missing {m}::{n}"))))
                .collect()
        };
        let bk = collect("gpu_batch", BATCH_FUNCS);
        let df = collect("gpu_dsv4", MOE_DFUNCS);

        // RoPE table for this layer kind (dsv4_cpu::layer_rope_table semantics).
        let table = dsv4_cpu::layer_rope_table(cfg, kind, positions);
        let rope = DevRope::from_cpu(dev, &table).context("upload rope table")?;
        Ok(Self {
            dev: dev.clone(),
            stream,
            spine,
            attn,
            comp,
            bk,
            df,
            rope,
            ropes: HashMap::new(),
            window: cfg.window_size,
            tp_ctx_dptr: 0,
        })
    }

    /// Full-trunk constructor: one runtime serving ALL three layer kinds (SWA + CSA + HCA),
    /// used by `Dsv4GpuModel`. Loads the comp module unconditionally (CSA/HCA need it; SWA
    /// doesn't construct a compressor/indexer, so the extra module is harmless) and builds
    /// all three RoPE tables (§B.1.3 — SWA plain θ vs CSA/HCA YaRN). `attn_forward` selects
    /// the rope via [`rope_for`](Self::rope_for); the lane `new` path stays unaffected.
    pub fn new_multikind(
        dev: &Arc<CudaDevice>,
        positions: usize,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<Self> {
        let stream = dsv4_gpu::blocking_compute_stream(dev);
        let spine = Dsv4Kernels::load(dev, SPINE_FUNCS).context("load gpu_dsv4.ptx (spine)")?;
        spine
            .set_dynamic_smem("dsv4_gather_attn", GATHER_SMEM)
            .context("set_dynamic_smem gather_attn")?;
        spine
            .set_dynamic_smem("dsv4_fused_gather_b", GATHER_SMEM)
            .context("set_dynamic_smem fused_gather_b")?;
        spine
            .set_dynamic_smem("dsv4_hc_pre_fused_b", HC_PRE_FUSED_SMEM)
            .context("set_dynamic_smem hc_pre_fused_b")?;
        let attn = Dsv4Kernels::load_module(dev, "src/ptx/gpu_dsv4_attn.ptx", ATTN_FUNCS)
            .context("load gpu_dsv4_attn.ptx (lane 3A/3C)")?;
        // Multikind always needs the comp module (CSA + HCA layers construct compressors).
        let comp = Some(CompKernels::load(dev).context("CompKernels::load (multikind)")?);

        let bptx = Ptx::from_src(
            std::fs::read_to_string("src/ptx/gpu_batch.ptx").context("read gpu_batch.ptx")?,
        );
        dev.load_ptx(bptx, "gpu_batch", BATCH_FUNCS).context("load_ptx gpu_batch")?;
        let dptx = Ptx::from_src(
            std::fs::read_to_string("src/ptx/gpu_dsv4.ptx").context("read gpu_dsv4.ptx")?,
        );
        dev.load_ptx(dptx, "gpu_dsv4", MOE_DFUNCS).context("load_ptx gpu_dsv4 (moe)")?;
        let collect = |m: &str, names: &[&'static str]| -> HashMap<String, CudaFunction> {
            names
                .iter()
                .map(|n| (n.to_string(), dev.get_func(m, n).unwrap_or_else(|| panic!("missing {m}::{n}"))))
                .collect()
        };
        let bk = collect("gpu_batch", BATCH_FUNCS);
        let df = collect("gpu_dsv4", MOE_DFUNCS);

        // All three RoPE tables (§B.1.3). SWA forces YaRN off (plain θ=10000); CSA/HCA YaRN.
        // DevRope owns a CudaSlice (not Clone), so SWA lives in the `rope` fallback field and
        // `ropes` holds only CSA/HCA — `rope_for(Swa)` falls through to `rope`.
        let rope = DevRope::from_cpu(dev, &dsv4_cpu::layer_rope_table(cfg, dsv4_load::LayerKind::Swa, positions))
            .context("upload rope table (multikind SWA)")?;
        let mut ropes = HashMap::new();
        for kind in [dsv4_load::LayerKind::Csa, dsv4_load::LayerKind::Hca] {
            let table = dsv4_cpu::layer_rope_table(cfg, kind, positions);
            ropes.insert(kind, DevRope::from_cpu(dev, &table).context("upload rope table (multikind)")?);
        }
        Ok(Self {
            dev: dev.clone(),
            stream,
            spine,
            attn,
            comp,
            bk,
            df,
            rope,
            ropes,
            window: cfg.window_size,
            tp_ctx_dptr: 0,
        })
    }

    /// RoPE table for `kind`: the multikind table if present (full trunk), else the lane's
    /// single-kind `rope` (the 3A/3C replay path).
    pub fn rope_for(&self, kind: dsv4_load::LayerKind) -> &DevRope {
        self.ropes.get(&kind).unwrap_or(&self.rope)
    }

    /// Synchronize the compute stream (surfaces any prior async kernel error).
    pub fn synchronize(&self) -> Result<()> {
        self.dev.synchronize().map_err(|e| anyhow!("stream sync: {e:?}"))
    }

    /// Input embedding gather + replicate to hc streams (§1.1, model.py:916 forward_embed):
    /// out [s, hc*dim] bf16 = embed[ids[t], :] copied into every stream. `embed` is [vocab, dim].
    pub fn embed_tokens<X: dsv4_gpu::Dsv4Buf<bf16>>(&self, embed: &B, ids: &CudaSlice<i32>, s: usize, cfg: &dsv4_load::Dsv4Config) -> Result<X> {
        let (hc, dim) = (cfg.hc_mult, cfg.dim);
        let out = X::alloc_zeros(&self.dev, self.stream.stream, s * hc * dim)?;
        let (s_i, dim_i, hc_i) = (s as i32, dim as i32, hc as i32);
        dsv4_launch!(self.spine, "dsv4_embed_b", self.stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
            (&out, embed, ids, &s_i, &dim_i, &hc_i))?;
        Ok(out)
    }

    /// DSpark trunk interface (§B.10): mean-over-streams of 3 layer outputs (layers 40/41/42)
    /// concatenated → main_hidden [s, 3*dim]. `dsv4_main_hidden_b` fuses mean + concat.
    pub fn compute_main_hidden(&self, x40: &B, x41: &B, x42: &B, s: usize, cfg: &dsv4_load::Dsv4Config) -> Result<B> {
        let (hc, dim) = (cfg.hc_mult, cfg.dim);
        let out = self.dev.alloc_zeros::<bf16>(s * 3 * dim)?;
        let total = (s * 3 * dim) as u32;
        let (s_i, hc_i, dim_i) = (s as i32, hc as i32, dim as i32);
        dsv4_launch!(self.spine, "dsv4_main_hidden_b", self.stream.stream,
            (ceil256(total as usize), 1, 1), (256, 1, 1), 0,
            (&out, x40, x41, x42, &s_i, &hc_i, &dim_i))?;
        Ok(out)
    }

    /// Load + upload trunk layer `layer_id` (SWA for this lane; the loader is kind-generic).
    /// `rank`/`world` TP-shard the routed-expert bank when `world > 1` (§5: each rank uploads its
    /// `[rank·ne/world, (rank+1)·ne/world)` expert slice via `Dsv4MoeGpu::upload_sharded`;
    /// everything else — attn, mHC, router, shared, norms — is replicated). `world == 1` is the
    /// unsharded single-process path (`Dsv4MoeGpu::upload`, e_base=0/e_span=ne).
    pub fn upload_layer(
        &self,
        bundle: &Path,
        cfg: &dsv4_load::Dsv4Config,
        layer_id: usize,
        rank: usize,
        world: usize,
    ) -> Result<Dsv4GpuLayer> {
        let dev = &self.dev;
        let fp8 = |key: &str| -> Result<Fp8Weight> {
            let name = format!("layers.{layer_id}.{key}");
            let (shape, codes, sb) = dsv4_load::read_raw_fp8(bundle, &name)
                .with_context(|| format!("read_raw_fp8 {name}"))?;
            let (m, k) = (shape[0], shape[1]);
            let wt = quant::repack_fp8_mma(&codes, m, k);
            Ok(Fp8Weight {
                wt: dev.htod_sync_copy(&wt).with_context(|| format!("upload {name} wt"))?,
                sb: dev.htod_sync_copy(&sb).with_context(|| format!("upload {name} sb"))?,
                m,
                k,
            })
        };
        let wq_a = fp8("attn.wq_a.weight")?;
        let wq_b = fp8("attn.wq_b.weight")?;
        let wkv = fp8("attn.wkv.weight")?;
        let wo_b = fp8("attn.wo_b.weight")?;
        // Shared expert: fuse [w1; w3] into one [4096, 4096] gate_up GEMM (per-16-row-tile
        // repack makes concat == fused; scale blocks stack at the 128-row boundary).
        let sh_w1 = fp8("ffn.shared_experts.w1.weight")?;
        let sh_w3 = fp8("ffn.shared_experts.w3.weight")?;
        let sh_w2 = fp8("ffn.shared_experts.w2.weight")?;
        let sh_gu = {
            let (m1, m3, k) = (sh_w1.m, sh_w3.m, sh_w1.k);
            anyhow::ensure!(k == sh_w3.k && m1 == m3 && (m1 + m3) % 128 == 0, "shared gu geometry");
            let name = |base: &str| format!("layers.{layer_id}.{base}");
            let (_, c1, s1) = dsv4_load::read_raw_fp8(bundle, &name("ffn.shared_experts.w1.weight"))?;
            let (_, c3, s3) = dsv4_load::read_raw_fp8(bundle, &name("ffn.shared_experts.w3.weight"))?;
            let mut codes = c1;
            codes.extend_from_slice(&c3);
            let mut sb = s1;
            sb.extend_from_slice(&s3);
            let wt = quant::repack_fp8_mma(&codes, m1 + m3, k);
            Fp8Weight {
                wt: dev.htod_sync_copy(&wt)?,
                sb: dev.htod_sync_copy(&sb)?,
                m: m1 + m3,
                k,
            }
        };

        // Strict-loaded layer for everything else (wo_a bf16 dequant, norms, hc, gate, experts).
        let layer = dsv4_load::load_layer(bundle, cfg, layer_id).context("load_layer")?;
        let mut map = layer.tensors;
        let wo_a: Vec<bf16> = match map.remove("attn.wo_a.weight") {
            Some(dsv4_load::HostTensor::BF16 { data, shape }) => {
                anyhow::ensure!(shape == vec![cfg.o_groups * cfg.o_lora_rank, cfg.dim], "wo_a shape {shape:?}");
                data
            }
            other => return Err(anyhow!("attn.wo_a.weight: expected BF16 dequant, got {:?}", other.map(|t| t.shape().to_vec()))),
        };
        let gate_w_bf: Vec<bf16> = match map.remove("ffn.gate.weight") {
            Some(dsv4_load::HostTensor::BF16 { data, .. }) => data,
            other => return Err(anyhow!("ffn.gate.weight: expected BF16, got {:?}", other.map(|t| t.shape().to_vec()))),
        };
        let gate_w: Vec<f32> = gate_w_bf.iter().map(|v| v.to_f32()).collect();
        let tid2eid = if cfg.is_hash_layer(layer_id) {
            let t = dsv4_cpu::take_i32(&mut map, "ffn.gate.tid2eid", cfg.vocab_size * cfg.n_activated_experts)?;
            Some(dev.htod_sync_copy(&t)?)
        } else {
            None
        };
        let gate_bias = if cfg.is_hash_layer(layer_id) {
            None
        } else {
            let b = dsv4_cpu::take_f32(&mut map, "ffn.gate.bias", cfg.n_routed_experts)?;
            Some(dev.htod_sync_copy(&b)?)
        };
        let q_norm = dsv4_cpu::take_f32(&mut map, "attn.q_norm.weight", cfg.q_lora_rank)?;
        let kv_norm = dsv4_cpu::take_f32(&mut map, "attn.kv_norm.weight", cfg.head_dim)?;
        let attn_norm = dsv4_cpu::take_f32(&mut map, "attn_norm.weight", cfg.dim)?;
        let ffn_norm = dsv4_cpu::take_f32(&mut map, "ffn_norm.weight", cfg.dim)?;
        let sink = dsv4_cpu::take_f32(&mut map, "attn.attn_sink", cfg.n_heads)?;
        let hc_attn_fn = dsv4_cpu::take_f32(&mut map, "hc_attn_fn", 24 * cfg.hc_mult * cfg.dim)?;
        let hc_attn_base = dsv4_cpu::take_f32(&mut map, "hc_attn_base", 24)?;
        let hc_attn_scale = dsv4_cpu::take_f32(&mut map, "hc_attn_scale", 3)?;
        let hc_ffn_fn = dsv4_cpu::take_f32(&mut map, "hc_ffn_fn", 24 * cfg.hc_mult * cfg.dim)?;
        let hc_ffn_base = dsv4_cpu::take_f32(&mut map, "hc_ffn_base", 24)?;
        let hc_ffn_scale = dsv4_cpu::take_f32(&mut map, "hc_ffn_scale", 3)?;

        // Lane 3C: attention compressor + indexer weights (CSA/HCA only). Held on the
        // host; uploaded into GpuCompressor/GpuIndexer at `new_state`. The keys and
        // geometry mirror dsv4_cpu::cpu_layer_core's CSA/Hca arms exactly (§A.2).
        let kind = cfg.layer_kind(layer_id);
        let (hd, qlr, ihd, inh) = (cfg.head_dim, cfg.q_lora_rank, cfg.index_head_dim, cfg.index_n_heads);
        let comp_load = match kind {
            dsv4_load::LayerKind::Swa => None,
            dsv4_load::LayerKind::Csa => Some(CompLoad {
                spec: CompSpec::csa_attn(cfg.dim, cfg.rope_head_dim),
                w: dsv4_cpu::CompressorWeights {
                    wkv: dsv4_cpu::take_f32(&mut map, "attn.compressor.wkv.weight", 2 * hd * cfg.dim)?,
                    wgate: dsv4_cpu::take_f32(&mut map, "attn.compressor.wgate.weight", 2 * hd * cfg.dim)?,
                    norm: dsv4_cpu::take_f32(&mut map, "attn.compressor.norm.weight", hd)?,
                    ape: dsv4_cpu::take_f32(&mut map, "attn.compressor.ape", 4 * 2 * hd)?,
                    ratio: 4, head_dim: hd, rope_dim: cfg.rope_head_dim,
                    overlap: true, rotate: false, sim_group: 64, dim: cfg.dim,
                },
            }),
            dsv4_load::LayerKind::Hca => Some(CompLoad {
                spec: CompSpec::hca_attn(cfg.dim, cfg.rope_head_dim),
                w: dsv4_cpu::CompressorWeights {
                    wkv: dsv4_cpu::take_f32(&mut map, "attn.compressor.wkv.weight", hd * cfg.dim)?,
                    wgate: dsv4_cpu::take_f32(&mut map, "attn.compressor.wgate.weight", hd * cfg.dim)?,
                    norm: dsv4_cpu::take_f32(&mut map, "attn.compressor.norm.weight", hd)?,
                    ape: dsv4_cpu::take_f32(&mut map, "attn.compressor.ape", 128 * hd)?,
                    ratio: 128, head_dim: hd, rope_dim: cfg.rope_head_dim,
                    overlap: false, rotate: false, sim_group: 64, dim: cfg.dim,
                },
            }),
        };
        // CSA indexer: wq_b (raw fp8, MMA-repacked — NOT the dequant in `map`),
        // weights_proj (bf16-valued f32), and its own rotate compressor (ratio 4, hd 128).
        let idx_load = if matches!(kind, dsv4_load::LayerKind::Csa) {
            let idx_name = format!("layers.{layer_id}.attn.indexer.wq_b.weight");
            let (shape, codes, sb) = dsv4_load::read_raw_fp8(bundle, &idx_name)
                .with_context(|| format!("read_raw_fp8 {idx_name}"))?;
            anyhow::ensure!(shape[0] == inh * ihd && shape[1] == qlr, "indexer wq_b shape {shape:?}");
            let wq_b_wt = quant::repack_fp8_mma(&codes, inh * ihd, qlr);
            let weights_proj = dsv4_cpu::take_bf16_as_f32(
                &mut map, "attn.indexer.weights_proj.weight", inh * cfg.dim)?;
            Some(IndexerLoad {
                comp: CompLoad {
                    spec: CompSpec::indexer(cfg.dim, cfg.rope_head_dim),
                    w: dsv4_cpu::CompressorWeights {
                        wkv: dsv4_cpu::take_f32(&mut map, "attn.indexer.compressor.wkv.weight", 2 * ihd * cfg.dim)?,
                        wgate: dsv4_cpu::take_f32(&mut map, "attn.indexer.compressor.wgate.weight", 2 * ihd * cfg.dim)?,
                        norm: dsv4_cpu::take_f32(&mut map, "attn.indexer.compressor.norm.weight", ihd)?,
                        ape: dsv4_cpu::take_f32(&mut map, "attn.indexer.compressor.ape", 4 * 2 * ihd)?,
                        ratio: 4, head_dim: ihd, rope_dim: cfg.rope_head_dim,
                        overlap: true, rotate: true, sim_group: 32, dim: cfg.dim,
                    },
                },
                wq_b_wt,
                wq_b_sb: sb,
                weights_proj,
            })
        } else {
            None
        };

        let host = dsv4_moe::pack_moe_layer(
            &dsv4_load::Dsv4Layer {
                tensors: std::mem::take(&mut map),
                experts_w1: layer.experts_w1,
                experts_w2: layer.experts_w2,
                experts_w3: layer.experts_w3,
            },
            cfg,
        )
        .context("pack_moe_layer")?;
        let moe = if world > 1 {
            let e_span = cfg.n_routed_experts / world;
            anyhow::ensure!(e_span * world == cfg.n_routed_experts, "ne {} not divisible by world {world}", cfg.n_routed_experts);
            Dsv4MoeGpu::upload_sharded(dev, &host, rank * e_span, e_span).context("Dsv4MoeGpu::upload_sharded")?
        } else {
            Dsv4MoeGpu::upload(dev, &host).context("Dsv4MoeGpu::upload")?
        };

        let up = |v: &[f32]| -> Result<S> { Ok(dev.htod_sync_copy(v)?) };
        Ok(Dsv4GpuLayer {
            kind,
            wq_a,
            wq_b,
            wkv,
            wo_b,
            sh_gu,
            sh_w2,
            wo_a: dev.htod_sync_copy(&wo_a)?,
            wo_a_q: Some(quantize_wo_a(dev, &wo_a, cfg.o_groups, cfg.o_lora_rank, cfg.dim)?),
            q_norm: up(&q_norm)?,
            kv_norm: up(&kv_norm)?,
            attn_norm: up(&attn_norm)?,
            ffn_norm: up(&ffn_norm)?,
            sink: up(&sink)?,
            hc_attn_fn: up(&hc_attn_fn)?,
            hc_attn_base: up(&hc_attn_base)?,
            hc_attn_scale: up(&hc_attn_scale)?,
            hc_ffn_fn: up(&hc_ffn_fn)?,
            hc_ffn_base: up(&hc_ffn_base)?,
            hc_ffn_scale: up(&hc_ffn_scale)?,
            gate_w: up(&gate_w)?,
            tid2eid,
            gate_bias,
            moe,
            comp_load,
            idx_load,
        })
    }

    /// Load + upload a DSpark stage (`mtp.{stage}.*`) — a full SWA-kind MoE block (no
    /// compressor/indexer; bias-routed, never hash). Mirrors [`upload_layer`](Self::upload_layer)
    /// but with the `mtp.{stage}.` key prefix and `load_mtp_stage` for the strict load. Returns
    /// the uploaded layer AND the leftover tensor map (carrying the stage extras: main_proj/
    /// main_norm for stage 0; norm/hc_head/markov/confidence for stage 2). DSpark runs
    /// single-process on the head (full 256 experts, world=1) — no TP sharding here.
    pub fn upload_mtp_stage(
        &self,
        bundle: &Path,
        cfg: &dsv4_load::Dsv4Config,
        stage: usize,
    ) -> Result<(Dsv4GpuLayer, std::collections::HashMap<String, dsv4_load::HostTensor>)> {
        let dev = &self.dev;
        let pfx = format!("mtp.{stage}.");
        let fp8 = |key: &str| -> Result<Fp8Weight> {
            let name = format!("{pfx}{key}");
            let (shape, codes, sb) = dsv4_load::read_raw_fp8(bundle, &name)
                .with_context(|| format!("read_raw_fp8 {name}"))?;
            let (m, k) = (shape[0], shape[1]);
            let wt = quant::repack_fp8_mma(&codes, m, k);
            Ok(Fp8Weight {
                wt: dev.htod_sync_copy(&wt).with_context(|| format!("upload {name} wt"))?,
                sb: dev.htod_sync_copy(&sb).with_context(|| format!("upload {name} sb"))?,
                m, k,
            })
        };
        let wq_a = fp8("attn.wq_a.weight")?;
        let wq_b = fp8("attn.wq_b.weight")?;
        let wkv = fp8("attn.wkv.weight")?;
        let wo_b = fp8("attn.wo_b.weight")?;
        // Shared expert fuse [w1; w3].
        let sh_w1 = fp8("ffn.shared_experts.w1.weight")?;
        let sh_w3 = fp8("ffn.shared_experts.w3.weight")?;
        let sh_w2 = fp8("ffn.shared_experts.w2.weight")?;
        let sh_gu = {
            let (m1, m3, k) = (sh_w1.m, sh_w3.m, sh_w1.k);
            anyhow::ensure!(k == sh_w3.k && m1 == m3 && (m1 + m3) % 128 == 0, "dspark shared gu geometry");
            let nm = |base: &str| format!("{pfx}{base}");
            let (_, c1, s1) = dsv4_load::read_raw_fp8(bundle, &nm("ffn.shared_experts.w1.weight"))?;
            let (_, c3, s3) = dsv4_load::read_raw_fp8(bundle, &nm("ffn.shared_experts.w3.weight"))?;
            let mut codes = c1; codes.extend_from_slice(&c3);
            let mut sb = s1; sb.extend_from_slice(&s3);
            let wt = quant::repack_fp8_mma(&codes, m1 + m3, k);
            Fp8Weight { wt: dev.htod_sync_copy(&wt)?, sb: dev.htod_sync_copy(&sb)?, m: m1 + m3, k }
        };
        // Strict load (mtp.{stage}.* — embed/head skipped as tied).
        let layer = dsv4_load::load_mtp_stage(bundle, cfg, stage).context("load_mtp_stage")?;
        let mut map = layer.tensors;
        let wo_a: Vec<bf16> = match map.remove("attn.wo_a.weight") {
            Some(dsv4_load::HostTensor::BF16 { data, shape }) => {
                anyhow::ensure!(shape == vec![cfg.o_groups * cfg.o_lora_rank, cfg.dim], "dspark wo_a shape {shape:?}");
                data
            }
            other => return Err(anyhow!("dspark attn.wo_a.weight: expected BF16, got {:?}", other.map(|t| t.shape().to_vec()))),
        };
        let gate_w_bf: Vec<bf16> = match map.remove("ffn.gate.weight") {
            Some(dsv4_load::HostTensor::BF16 { data, .. }) => data,
            other => return Err(anyhow!("dspark ffn.gate.weight: expected BF16, got {:?}", other.map(|t| t.shape().to_vec()))),
        };
        let gate_w: Vec<f32> = gate_w_bf.iter().map(|v| v.to_f32()).collect();
        // DSpark stages are always bias-routed (gate.bias present, never tid2eid).
        let gate_bias = {
            let b = dsv4_cpu::take_f32(&mut map, "ffn.gate.bias", cfg.n_routed_experts)?;
            Some(dev.htod_sync_copy(&b)?)
        };
        let q_norm = dsv4_cpu::take_f32(&mut map, "attn.q_norm.weight", cfg.q_lora_rank)?;
        let kv_norm = dsv4_cpu::take_f32(&mut map, "attn.kv_norm.weight", cfg.head_dim)?;
        let attn_norm = dsv4_cpu::take_f32(&mut map, "attn_norm.weight", cfg.dim)?;
        let ffn_norm = dsv4_cpu::take_f32(&mut map, "ffn_norm.weight", cfg.dim)?;
        let sink = dsv4_cpu::take_f32(&mut map, "attn.attn_sink", cfg.n_heads)?;
        let hc_attn_fn = dsv4_cpu::take_f32(&mut map, "hc_attn_fn", 24 * cfg.hc_mult * cfg.dim)?;
        let hc_attn_base = dsv4_cpu::take_f32(&mut map, "hc_attn_base", 24)?;
        let hc_attn_scale = dsv4_cpu::take_f32(&mut map, "hc_attn_scale", 3)?;
        let hc_ffn_fn = dsv4_cpu::take_f32(&mut map, "hc_ffn_fn", 24 * cfg.hc_mult * cfg.dim)?;
        let hc_ffn_base = dsv4_cpu::take_f32(&mut map, "hc_ffn_base", 24)?;
        let hc_ffn_scale = dsv4_cpu::take_f32(&mut map, "hc_ffn_scale", 3)?;
        // Pull the DSpark stage extras out of the map BEFORE pack_moe_layer takes it. These
        // are the stage-0 main_proj/main_norm and the stage-2 head/markov/confidence tensors
        // (absent on the other stages — `map.remove` leaves them None there).
        let mut extras: std::collections::HashMap<String, dsv4_load::HostTensor> = std::collections::HashMap::new();
        for k in ["main_norm.weight", "main_proj.weight", "main_proj.scale",
                  "norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale",
                  "markov_head.markov_w1.weight", "markov_head.markov_w2.weight",
                  "confidence_head.proj.weight"] {
            if let Some(t) = map.remove(k) { extras.insert(k.to_string(), t); }
        }
        let host = dsv4_moe::pack_moe_layer(
            &dsv4_load::Dsv4Layer {
                tensors: std::mem::take(&mut map),
                experts_w1: layer.experts_w1,
                experts_w2: layer.experts_w2,
                experts_w3: layer.experts_w3,
            }, cfg,
        ).context("pack_moe_layer (dspark stage)")?;
        let moe = Dsv4MoeGpu::upload(dev, &host).context("Dsv4MoeGpu::upload (dspark stage)")?;
        let up = |v: &[f32]| -> Result<S> { Ok(dev.htod_sync_copy(v)?) };
        let gl = Dsv4GpuLayer {
            kind: dsv4_load::LayerKind::Swa,
            wq_a, wq_b, wkv, wo_b, sh_gu, sh_w2,
            wo_a: dev.htod_sync_copy(&wo_a)?,
            wo_a_q: Some(quantize_wo_a(dev, &wo_a, cfg.o_groups, cfg.o_lora_rank, cfg.dim)?),
            q_norm: up(&q_norm)?, kv_norm: up(&kv_norm)?,
            attn_norm: up(&attn_norm)?, ffn_norm: up(&ffn_norm)?,
            sink: up(&sink)?,
            hc_attn_fn: up(&hc_attn_fn)?, hc_attn_base: up(&hc_attn_base)?, hc_attn_scale: up(&hc_attn_scale)?,
            hc_ffn_fn: up(&hc_ffn_fn)?, hc_ffn_base: up(&hc_ffn_base)?, hc_ffn_scale: up(&hc_ffn_scale)?,
            gate_w: up(&gate_w)?,
            tid2eid: None, gate_bias, moe,
            comp_load: None, idx_load: None,
        };
        Ok((gl, extras))
    }

    /// Fresh per-layer attention state. SWA: window ring only (3A path). CSA/HCA:
    /// unified `kv_cache[window + max_comp_rows, hd]` (ring ++ compressor-cache tail,
    /// zero-init — unwritten compressor rows are never indexed), prefill scratch
    /// `[max_s + max_comp_rows, hd]`, and the attention compressor (+ CSA indexer)
    /// constructed from the layer's load weights.
    ///
    /// `max_seq_len` sizes the compressor caches (the oracle's `kv_cache.shape[0]`
    /// gives the right value per piece: e.g. CSA@8K → max_seq=16384 → cache 128+4096).
    /// `s_max` sizes the compressor GEMM scratch (the largest prefill chunk).
    #[allow(clippy::too_many_arguments)]
    pub fn new_state(
        &self,
        cfg: &dsv4_load::Dsv4Config,
        layer: &Dsv4GpuLayer,
        max_seq_len: usize,
        s_max: usize,
    ) -> Result<Dsv4AttnState> {
        let dev = &self.dev;
        let win = cfg.window_size;
        let hd = cfg.head_dim;
        let comp_ks = self.comp.as_ref().expect("CSA/HCA state needs comp kernels (runtime built without)");
        let comp_load = layer.comp_load.as_ref().expect("CSA/HCA layer has no comp_load");
        let max_comp_rows = match layer.kind {
            dsv4_load::LayerKind::Swa => 0,
            dsv4_load::LayerKind::Csa => max_seq_len / 4,
            dsv4_load::LayerKind::Hca => max_seq_len / 128,
        };
        // The kind's compressor ratio (SWA never reaches the scratch sizing below).
        let comp_ratio = match layer.kind {
            dsv4_load::LayerKind::Swa => 1,
            dsv4_load::LayerKind::Csa => 4,
            dsv4_load::LayerKind::Hca => 128,
        };
        // Unified kv_cache: [window + max_comp_rows, hd], zero-init (AGENTS §2.2 —
        // alloc_zeros doesn't zero, so explicit htod of a zero vec).
        let kv_rows = win + max_comp_rows;
        let zeros = vec![bf16::from_f32(0.0); kv_rows * hd];
        let kv_cache = dev.htod_sync_copy(&zeros).context("kv_cache alloc")?;

        // Prefill scratch: R2.1 resized for the batched-continuation path. The start_pos==0
        // path uses [s + nb_committed] rows; the R2.1 continuation path (start_pos>0) uses
        // [win(prefix) + s(new) + nb_committed(comp)] rows. The worst case (last chunk at
        // max_seq_len) needs win + s_max + max_comp_rows. The delta vs the old chunk-only
        // sizing is win + max_comp_rows rows/layer — ~50 MB at SEQ=8192, ~320 MB at 32K
        // (the 1M case is R5's budget concern; the comp_cache part is the max_comp_rows tail
        // the old comment warned about — but the batched attention needs it in the scratch
        // for a single-buffer gather, and the sequential alternative is ~36% slower per chunk).
        let scratch_rows = win + s_max + max_comp_rows;
        let zeros_sc = vec![bf16::from_f32(0.0); scratch_rows * hd];
        let kv_attn_scratch = Some(dev.htod_sync_copy(&zeros_sc).context("kv_attn_scratch alloc")?);

        // Attention compressor: owns its cache (3B) + frontier state. The compressor's
        // cache is ALIASED to kv_cache[win..] (Item 3: the epilogue writes directly to the
        // kv_cache tail, eliminating the per-step d2d mirror and recovering ~6 GB at 1M).
        // The gather reads from the unified kv_cache (ring + compressor-cache tail) contiguously.
        let mut attn_compressor = GpuCompressor::new(
            dev, comp_ks, &self.stream, comp_load.spec, &comp_load.w,
            cfg.norm_eps, max_comp_rows, s_max,
        ).context("GpuCompressor::new (attention)")?;
        let tail_dptr = *kv_cache.device_ptr() + (win * hd * 2) as u64;
        attn_compressor.set_cache_alias(tail_dptr);
        // Reclaim the full-size cache allocation — the alias replaces it. Keep a 1-element
        // dummy so the CudaSlice field is valid (snapshot/restore size from cache_rows).
        attn_compressor.cache = dev.alloc_zeros::<bf16>(1)?;
        let attn_compressor = Some(attn_compressor);

        // CSA indexer: own rotate compressor + indexer kv_cache [max_seq/4, 128].
        let indexer = if let Some(idx_load) = layer.idx_load.as_ref() {
            Some(GpuIndexer::new(
                dev, comp_ks, &self.stream, cfg.dim, cfg.rope_head_dim, cfg.q_lora_rank,
                cfg.index_n_heads, cfg.index_head_dim, cfg.index_topk,
                &idx_load.comp.w, cfg.norm_eps, &idx_load.wq_b_wt, &idx_load.wq_b_sb,
                &idx_load.weights_proj, max_seq_len, s_max,
            ).context("GpuIndexer::new")?)
        } else {
            None
        };

        Ok(Dsv4AttnState { kv_cache, kv_attn_scratch, attn_compressor, indexer })
    }

    /// Fresh SWA ring state (window × 512 bf16, explicitly zeroed — AGENTS §2.2).
    /// R2.1: SWA continuation chunks need a unified scratch [win + s_max] for the
    /// batched gather (prefix window + new kv). The start_pos==0 path gathers from
    /// `kv` directly (no scratch); the continuation path gathers from the scratch.
    pub fn new_state_swa(&self, cfg: &dsv4_load::Dsv4Config, s_max: usize) -> Result<Dsv4AttnState> {
        let win = cfg.window_size;
        let hd = cfg.head_dim;
        let zeros = vec![bf16::from_f32(0.0); win * hd];
        // R2.1: [win + s_max, hd] — prefix window + one chunk of new kv.
        let scratch_rows = win + s_max;
        let zeros_sc = vec![bf16::from_f32(0.0); scratch_rows * hd];
        Ok(Dsv4AttnState {
            kv_cache: self.dev.htod_sync_copy(&zeros)?,
            kv_attn_scratch: Some(self.dev.htod_sync_copy(&zeros_sc)?),
            attn_compressor: None,
            indexer: None,
        })
    }

    // ---------------------------------------------------------------------------------------------
    // Primitive wrappers (all launches on the blocking compute stream unless noted)
    // ---------------------------------------------------------------------------------------------

    /// §C.1 FP8 act-quant (non-inplace): bf16 x [rows, n] → codes [rows, n] + UE8M0 [rows, n/128].
    pub fn quant_g128<X: dsv4_gpu::Dsv4Buf<bf16>, C: dsv4_gpu::Dsv4Buf<u8>>(&self, x: &X, rows: usize, n: usize) -> Result<(C, C)> {
        let mut y = C::alloc_zeros(&self.dev, self.stream.stream, rows * n)?;
        let mut s = C::alloc_zeros(&self.dev, self.stream.stream, rows * (n / 128))?;
        self.quant_g128_into(&mut y, &mut s, x, rows, n)?;
        Ok((y, s))
    }

    /// R3.1: `quant_g128` into caller buffers (the MoE workspace — zero per-call allocs).
    pub fn quant_g128_into<X: dsv4_gpu::Dsv4Buf<bf16>, C: dsv4_gpu::Dsv4Buf<u8>>(&self, y: &mut C, s: &mut C, x: &X, rows: usize, n: usize) -> Result<()> {
        let blocks = ceil256(rows * (n / 128) * 32);
        let (r_i, n_i) = (rows as i32, n as i32);
        dsv4_launch!(self.spine, "dsv4_act_quant_g128", self.stream.stream, (blocks, 1, 1), (256, 1, 1), 0,
            (x, &mut *y, &mut *s, &r_i, &n_i))?;
        Ok(())
    }

    /// §C.3 FP8 block-scale GEMM at activation width s (chunked ≤16 — the G2-proven
    /// N-invariant regime; chunking is bitwise-identical to any other ≤16 decomposition).
    /// `codes`/`sa` are the full [s, k] / [s, k/128] activation quant. Output [s, m] bf16.
    pub fn fp8_bsb_rows<X: dsv4_gpu::Dsv4Buf<bf16>, C: dsv4_gpu::Dsv4Buf<u8>>(&self, w: &Fp8Weight, codes: &C, sa: &C, s: usize) -> Result<X> {
        let (m, _k) = (w.m, w.k);
        let mut c = X::alloc_zeros(&self.dev, self.stream.stream, s * m)?;
        self.fp8_bsb_rows_into(&mut c, w, codes, sa, s)?;
        Ok(c)
    }

    /// R3A.1 E2 (first rung): ONE fused launch for two independent fp8 projections sharing
    /// the activation (production: wq_a + wkv per layer). Tier-1 item 1.4 (RUN 16): the
    /// launch is `gemm_dsv4_fp8_bsb1q` — ONE 16-row tile per CTA (96 CTAs at the decode
    /// shapes) instead of the two-tile bsb2q (48 CTAs, ramp-bound at ~1 CTA/SM): 28.4 µs vs
    /// 42.5 µs isolated (221.7 vs 148.2 GB/s). Per-element chains identical to the two
    /// separate bsb launches — gated bitwise at the production shapes, N ∈ {1,6,16}.
    /// Only for s ≤ 16 (decode/verify widths; prefill keeps the pf path per projection).
    pub fn fp8_bsb2q_rows<X: dsv4_gpu::Dsv4Buf<bf16>, C: dsv4_gpu::Dsv4Buf<u8>>(&self, w0: &Fp8Weight, w1: &Fp8Weight, codes: &C, sa: &C, s: usize) -> Result<(X, X)> {
        anyhow::ensure!(s >= 1 && s <= 16, "fp8_bsb2q is a ≤16-row launch (s={s})");
        let (m0, m1, k) = (w0.m, w1.m, w0.k);
        anyhow::ensure!(w1.k == k && m0 % 128 == 0 && m1 % 128 == 0 && k % 128 == 0,
            "fp8_bsb2q geometry {m0}x{m1}x{k}");
        let c0 = X::alloc_zeros(&self.dev, self.stream.stream, s * m0)?;
        let c1 = X::alloc_zeros(&self.dev, self.stream.stream, s * m1)?;
        let f = &self.bk["gemm_dsv4_fp8_bsb1q"];
        let cfg = LaunchConfig {
            grid_dim: ((m0 / 16 + m1 / 16) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let m01: u64 = (m0 as u64) | ((m1 as u64) << 32);
        let kn: u64 = (k as u64) | ((s as u64) << 32);
        let (c0v, c1v) = (c0.view(0, s * m0), c1.view(0, s * m1));
        let (cv, sav) = (codes.view(0, s * k), sa.view(0, s * (k / 128)));
        unsafe {
            f.clone()
                .launch_on_stream(&self.stream, cfg,
                    (&c0v, &w0.wt, &w0.sb, &c1v, &w1.wt, &w1.sb, &cv, &sav, m01, kn, 0u64))
                .map_err(|e| anyhow!("gemm_dsv4_fp8_bsb1q s={s}: {e:?}"))?;
        }
        Ok((c0, c1))
    }

    /// R3.1: `fp8_bsb_rows` into a caller buffer (the MoE workspace — zero per-call allocs).
    /// R3A.1: launches the two-tiles-per-CTA `gemm_dsv4_fp8_bsb2` (identical per-element
    /// chains, two independent weight streams per warp — E1b; grid halves to (m+31)/32).
    /// R3A.4 (P1): at s>16 ONE weight-stationary launch (chunk loop inside; each weight
    /// tile read once per launch, L2-hot across chunks) instead of ceil(s/16) full-weight
    /// re-read launches. Tier-2 2.2 (session 6): the launch is `gemm_dsv4_fp8_bsb_pf2` at
    /// K<=4096 — TWO 16-row weight tiles per CTA sharing each chunk's X fragments (the
    /// bsb2 decode trick at width), with a grid.y fill floor — and stays
    /// `gemm_dsv4_fp8_bsb_pf` at K>4096 (wo_b: pf2's dual weight stream measures 4-6%
    /// slower there). Per-element chains identical to the <=16-row decomposition — gated
    /// bitwise at s in {17, 63, 64, 65, 130, 2048}.
    pub fn fp8_bsb_rows_into<OB: dsv4_gpu::Dsv4Buf<bf16>, C: dsv4_gpu::Dsv4Buf<u8>>(&self, c: &mut OB, w: &Fp8Weight, codes: &C, sa: &C, s: usize) -> Result<()> {
        let (m, k) = (w.m, w.k);
        anyhow::ensure!(m % 128 == 0 && k % 128 == 0, "fp8_bsb geometry {m}x{k}");
        anyhow::ensure!(c.len() >= s * m, "fp8_bsb_rows_into: out {} < {s}*{m}", c.len());
        if s > 16 {
            // grid.y = chunk groups: ~48k CTAs of parallelism without re-serializing
            // chunks (weight tiles re-read <= gridDim.y times, L2-hot within).
            let nchunks = s.div_ceil(16);
            let gy8 = nchunks.div_ceil(8).max(1); // ~8 chunks per CTA (pf grouping)
            if k > 4096 {
                // wo_b-class (long K): pf2's dual weight stream per CTA measures 4-6%
                // SLOWER at K=8192 (session-6 bench) — keep the single-tile pf here.
                let f = &self.bk["gemm_dsv4_fp8_bsb_pf"];
                let cfg = LaunchConfig {
                    grid_dim: ((m / 16) as u32, gy8 as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let cv = c.view(0, s * m);
                let (xv, sav) = (codes.view(0, s * k), sa.view(0, s * (k / 128)));
                unsafe {
                    f.clone()
                        .launch_on_stream(&self.stream, cfg, (&cv, &w.wt, &w.sb, &xv, &sav, m as i32, k as i32, s as i32, 0u64))
                        .map_err(|e| anyhow!("gemm_dsv4_fp8_bsb_pf s={s}: {e:?}"))?;
                }
                return Ok(());
            }
            // pf2 (two 16-row tiles per CTA sharing each chunk's X fragments; session-6
            // bench 1.10-1.30x on every K<=4096 projection). grid.x = tile pairs; the
            // fill floor (2 waves of the 3 CTA/SM residency) keeps small-M shapes from
            // underfilling the machine (wkv [512,4096] @512: 0.88x unfilled -> 1.30x).
            let gx = (m + 31) / 32;
            let gy = gy8.max((288usize.div_ceil(gx)).min(nchunks)) as u32;
            let f = &self.bk["gemm_dsv4_fp8_bsb_pf2"];
            let cfg = LaunchConfig {
                grid_dim: (gx as u32, gy, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let cv = c.view(0, s * m);
            let (xv, sav) = (codes.view(0, s * k), sa.view(0, s * (k / 128)));
            unsafe {
                f.clone()
                    .launch_on_stream(&self.stream, cfg, (&cv, &w.wt, &w.sb, &xv, &sav, m as i32, k as i32, s as i32, 0u64))
                    .map_err(|e| anyhow!("gemm_dsv4_fp8_bsb_pf2 s={s}: {e:?}"))?;
            }
            return Ok(());
        }
        let f = &self.bk["gemm_dsv4_fp8_bsb2"];
        let mut r0 = 0usize;
        while r0 < s {
            let n = (s - r0).min(16);
            let xv = codes.view(r0 * k, n * k);
            let sav = sa.view(r0 * (k / 128), n * (k / 128));
            let cv = c.view(r0 * m, n * m);
            let cfg = LaunchConfig {
                grid_dim: (((m + 31) / 32) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                f.clone()
                    .launch_on_stream(&self.stream, cfg, (&cv, &w.wt, &w.sb, &xv, &sav, m as i32, k as i32, n as i32, 0u64))
                    .map_err(|e| anyhow!("gemm_dsv4_fp8_bsb2 chunk r0={r0} n={n}: {e:?}"))?;
            }
            r0 += n;
        }
        Ok(())
    }

    /// TP=2 sum-all-reduce of a bf16 device buffer across the two boxes — the doorbell two-kernel
    /// handshake (`tp_gate_copy_signal` → `tp_wait_add`), launched on THIS runtime's blocking compute
    /// stream. Mirrors `gpu.rs::tp_all_reduce_bf16` (3183) but on the DSV4 runtime's stream/bk.
    /// SPIN-WAIT comm (no main-thread sync); the persistent RDMA proxy (spawned at `attach_tp`)
    /// watches the device-side epoch and ships the payload. Chunked to the ring slot. No-op when
    /// `tp_ctx_dptr == 0` (single-process). `buf` is the rank-local PARTIAL; on return it holds the
    /// full Σ (bf16: ship bf16, sum in fp32 in K2, one bf16 round — the standard TP-partials class).
    pub fn tp_all_reduce_bf16(&self, buf: &mut B, n: usize) -> Result<()> {
        if self.tp_ctx_dptr == 0 || n == 0 {
            return Ok(());
        }
        let buf_ptr = *buf.device_ptr();
        let chunk = ((crate::tp::TP_SLOT_BYTES - 64) / 2) & !7; // bf16 elems per ring slot
        let mut off = 0usize;
        while off < n {
            let c = (n - off).min(chunk);
            let p = buf_ptr + (off * 2) as u64;
            let nbytes = (c * 2) as u32;
            let c_i = c as i32;
            {
                let f = &self.bk["tp_gate_copy_signal"];
                let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: 0 };
                unsafe {
                    f.clone().launch_on_stream(&self.stream, cfg, (self.tp_ctx_dptr, p, nbytes))
                        .map_err(|e| anyhow!("tp_gate_copy_signal: {e:?}"))?;
                }
            }
            {
                let f = &self.bk["tp_wait_add"];
                let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: 0 };
                unsafe {
                    f.clone().launch_on_stream(&self.stream, cfg, (self.tp_ctx_dptr, p, p, c_i, 0i32))
                        .map_err(|e| anyhow!("tp_wait_add: {e:?}"))?;
                }
            }
            off += c;
        }
        Ok(())
    }

    /// fp32 RMSNorm (bf16 in/out): y = rmsnorm(x, w) row-wise, dim columns.
    pub fn rmsnorm<X: dsv4_gpu::Dsv4Buf<half::bf16>>(&self, x: &X, w: &S, rows: usize, dim: usize, eps: f32) -> Result<X> {
        let y = X::alloc_zeros(&self.dev, self.stream.stream, rows * dim)?;
        let (r_i, d_i) = (rows as i32, dim as i32);
        dsv4_launch!(self.spine, "dsv4_rmsnorm_b", self.stream.stream, (rows as u32, 1, 1), (256, 1, 1), 0,
            (&y, x, w, &r_i, &d_i, &eps))?;
        Ok(y)
    }

    /// mHC hc_pre (§B.8): returns (y [s,dim] bf16, post [s,4] f32, comb [s,16] f32).
    ///
    /// T6: at PREFILL width (s ≥ 256) ONE fused launch (`dsv4_hc_pre_fused_b`) replaces the
    /// 4-launch chain (rsqrt + mixes + sinkhorn + collapse) — bitwise-identical per element
    /// (same per-thread fmaf chains + stride-halving trees), 3 tokens/768-thread block so the
    /// hc_fn rows are read once per 3 tokens instead of once per (m, t) block. At small s
    /// (decode s=1, verify s=6) the chain stays: the 24 dots need 24 blocks of parallelism
    /// there, and a single fused block serializes them (the measured negative precedent from
    /// the qwen era — parallelism-bound at decode, byte-redundant at width — matches).
    pub fn hc_pre<X: dsv4_gpu::Dsv4Buf<bf16>, F: dsv4_gpu::Dsv4Buf<f32>>(&self, x: &X, s: usize, hc_fn: &S, hc_base: &S, hc_scale: &S, cfg: &dsv4_load::Dsv4Config) -> Result<(X, F, F)> {
        let (hc, dim) = (cfg.hc_mult, cfg.dim);
        let hcdim = hc * dim;
        let rsqrt = F::alloc_zeros(&self.dev, self.stream.stream, s)?;
        let mixes = F::alloc_zeros(&self.dev, self.stream.stream, s * 24)?;
        let pre = F::alloc_zeros(&self.dev, self.stream.stream, s * hc)?;
        let post = F::alloc_zeros(&self.dev, self.stream.stream, s * hc)?;
        let comb = F::alloc_zeros(&self.dev, self.stream.stream, s * hc * hc)?;
        let y = X::alloc_zeros(&self.dev, self.stream.stream, s * dim)?;
        let (s_i, hcd_i, dim_i, hc_i) = (s as i32, hcdim as i32, dim as i32, hc as i32);
        let eps = cfg.norm_eps;
        if s >= 256 {
            let grid = ((s + 2) / 3) as u32;
            dsv4_launch!(self.spine, "dsv4_hc_pre_fused_b", self.stream.stream, (grid, 1, 1), (768, 1, 1), HC_PRE_FUSED_SMEM,
                (&rsqrt, &mixes, &pre, &post, &comb, &y, x, hc_fn, hc_scale, hc_base, &s_i, &hcd_i, &dim_i, &hc_i, &eps))?;
        } else {
            dsv4_launch!(self.spine, "dsv4_hc_pre_rsqrt_b", self.stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
                (&rsqrt, x, &s_i, &hcd_i, &eps))?;
            dsv4_launch!(self.spine, "dsv4_hc_mixes_b", self.stream.stream, (24u32, s as u32, 1), (256, 1, 1), 0,
                (&mixes, hc_fn, x, &rsqrt, &s_i, &hcd_i))?;
            dsv4_launch!(self.spine, "dsv4_hc_split_sinkhorn", self.stream.stream, (ceil256(s), 1, 1), (256, 1, 1), 0,
                (&mixes, hc_scale, hc_base, &pre, &post, &comb, &s_i))?;
            dsv4_launch!(self.spine, "dsv4_hc_collapse_b", self.stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
                (&y, x, &pre, &s_i, &dim_i, &hc_i))?;
        }
        Ok((y, post, comb))
    }

    /// mHC hc_post (§B.8): out [s, hc*dim] = post[k]·sub_out + Σ_j comb[j,k]·resid[j].
    pub fn hc_post<X: dsv4_gpu::Dsv4Buf<bf16>, F: dsv4_gpu::Dsv4Buf<f32>>(&self, sub_out: &X, resid: &X, post: &F, comb: &F, s: usize, cfg: &dsv4_load::Dsv4Config) -> Result<X> {
        let (hc, dim) = (cfg.hc_mult, cfg.dim);
        let out = X::alloc_zeros(&self.dev, self.stream.stream, s * hc * dim)?;
        let (s_i, dim_i, hc_i) = (s as i32, dim as i32, hc as i32);
        dsv4_launch!(self.spine, "dsv4_hc_post_b", self.stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
            (&out, sub_out, resid, post, comb, &s_i, &dim_i, &hc_i))?;
        Ok(out)
    }

    /// `hc_head` (§B.8, model.py:709-716): final trunk collapse, sigmoid-only (no post/comb/
    /// Sinkhorn). `x` [s, hc*dim] bf16 streams → `collapsed` [s, dim] bf16. One block/token;
    /// reduction trees match hc_pre (tolerance-level vs the CPU pairwise tree).
    pub fn hc_head<X: dsv4_gpu::Dsv4Buf<bf16>>(
        &self,
        x: &X,
        hc_fn: &S,
        hc_base: &S,
        hc_scale: &S,
        s: usize,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<X> {
        let (hc, dim) = (cfg.hc_mult, cfg.dim);
        let y = X::alloc_zeros(&self.dev, self.stream.stream, s * dim)?;
        let (s_i, dim_i, hc_i) = (s as i32, dim as i32, hc as i32);
        dsv4_launch!(self.spine, "dsv4_hc_head_b", self.stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
            (&y, x, hc_fn, hc_base, hc_scale, &s_i, &dim_i, &hc_i, &cfg.norm_eps, &cfg.hc_eps))?;
        Ok(y)
    }

    /// LM head (§A.1): bf16 `head` [vocab, dim] (fp32-valued) @ `x` [dim, n] bf16 → fp32 logits
    /// [vocab, n] via `gemm_binv_f32_b` (the hy_v3 enable_lm_head_fp32 twin — batch-invariant
    /// strided-k tree, f32 epilogue, no bf16 round). n ≤ 8 uses the templated fast path.
    pub fn lm_head<X: dsv4_gpu::Dsv4Buf<bf16>, F: dsv4_gpu::Dsv4Buf<f32>>(&self, head: &B, x: &X, n: usize, dim: usize, vocab: usize) -> Result<F> {
        let logits = F::alloc_zeros(&self.dev, self.stream.stream, vocab * n)?;
        let f = &self.bk["gemm_binv_f32_b"];
        // gemm_binv_impl<N> uses `extern __shared__ float sh[N][256]` for the tree reduce —
        // size the smem to N·256 floats (the trunk's n=1 path uses 256·4 = 1024 B; multi-row
        // DSpark verify / markov need the full N·256·4).
        let cfg = LaunchConfig {
            grid_dim: (vocab as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (n * 256 * 4) as u32,
        };
        let logits_v = logits.view(0, vocab * n);
        let xv = x.view(0, n * dim);
        unsafe {
            f.clone()
                .launch_on_stream(&self.stream, cfg, (&logits_v, head, &xv, vocab as i32, dim as i32, n as i32))
                .map_err(|e| anyhow!("gemm_binv_f32_b lm_head: {e:?}"))?;
        }
        Ok(logits)
    }

    /// R3.2 (L2): vocab-parallel LM head + the 8 B maxloc exchange under TP=2. This rank computes
    /// ONLY its vocab half's fp32 logits (the `gemm_binv_f32_b` kernel is row-independent — the
    /// half rows are per-element bitwise-identical to the unsharded head), runs the single-block
    /// argmax (val desc, idx asc — the same total order as the host `dsv4_argmax`), ships the 8 B
    /// (val, global idx) on the doorbell (K1) and combines with the peer's (K2 maxloc, same total
    /// order). The winner is bitwise the unsharded full-vocab argmax — SPMD lockstep holds with
    /// no token broadcast. Requires the TP doorbell attached. Greedy decode only — sampler/
    /// confidence-head paths keep the full-logits [`lm_head`](Self::lm_head).
    pub fn lm_head_maxloc_tp(&self, head: &B, x: &B, dim: usize, vocab: usize, rank: usize) -> Result<u32> {
        anyhow::ensure!(self.tp_ctx_dptr != 0, "lm_head_maxloc_tp: TP not attached");
        anyhow::ensure!(vocab % 2 == 0, "lm_head_maxloc_tp: odd vocab {vocab}");
        let rows = vocab / 2;
        let row0 = rank * rows;
        let mut half = self.dev.alloc_zeros::<f32>(rows)?;
        let hv = head.slice(row0 * dim..(row0 + rows) * dim);
        {
            let f = &self.bk["gemm_binv_f32_b"];
            let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: (256 * 4) as u32 };
            unsafe {
                f.clone()
                    .launch_on_stream(&self.stream, cfg, (&mut half, &hv, x, rows as i32, dim as i32, 1i32))
                    .map_err(|e| anyhow!("gemm_binv_f32_b lm_head_rows: {e:?}"))?;
            }
        }
        let mut pair = self.dev.alloc_zeros::<u32>(2)?;
        {
            let f = &self.bk["dsv4_argmax_pair_b"];
            let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                f.clone()
                    .launch_on_stream(&self.stream, cfg, (&half, rows as i32, &mut pair, row0 as i32))
                    .map_err(|e| anyhow!("dsv4_argmax_pair_b: {e:?}"))?;
            }
        }
        let mut out_idx = self.dev.alloc_zeros::<i32>(1)?;
        let pair_ptr = *pair.device_ptr();
        let out_ptr = *out_idx.device_ptr();
        {
            let f = &self.bk["tp_gate_copy_signal"];
            let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                f.clone()
                    .launch_on_stream(&self.stream, cfg, (self.tp_ctx_dptr, pair_ptr, 8u32))
                    .map_err(|e| anyhow!("tp_gate_copy_signal maxloc: {e:?}"))?;
            }
        }
        {
            // v2 receive mode (EXPERT_GPU_ALLREDUCE §3.2): the maxloc barrier inherits the
            // GPU-direct two-stage gate automatically.
            let k2 = if crate::gpu::GpuModel::tp_gpu_recv_on_pub() { "tp_wait_maxloc_g" } else { "tp_wait_maxloc_b" };
            let f = &self.bk[k2];
            let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                f.clone()
                    .launch_on_stream(&self.stream, cfg, (self.tp_ctx_dptr, pair_ptr, out_ptr))
                    .map_err(|e| anyhow!("tp_wait_maxloc: {e:?}"))?;
            }
        }
        let v: Vec<i32> = self.dev.dtoh_sync_copy(&out_idx)?;
        Ok(v[0] as u32)
    }

    /// Trunk attention sublayer (§B.1–B.4) on the attn_norm output `x` [s, dim] bf16.
    /// Kind-dispatched: SWA (3A path — window-only ring gather); CSA (window ++ indexer
    /// top-512 blocks, ratio-4 overlap compressor); HCA (window ++ ALL compressed blocks,
    /// ratio-128 non-overlap compressor). Returns (attn_out [s, dim] bf16, topk_idx
    /// [s, t] i32, t).
    ///
    /// CSA/HCA index space (decode): window indices address `kv_cache[0..win]` (ring
    /// physical slots 0..win-1); compress indices address `kv_cache[win..win+nb]` (the
    /// compressor cache tail, copied from `attn_compressor.cache` after each step).
    /// Prefill gathers over `kv_attn_scratch[0..s+nb]` (kv ++ compressor cache, matching
    /// the CPU reference's `kv_attn = kv.clone(); kv_attn.extend(kvc)`).
    pub fn attn_forward<X: dsv4_gpu::Dsv4Buf<bf16>, F: dsv4_gpu::Dsv4Buf<f32>, I: dsv4_gpu::Dsv4Buf<i32>, C: dsv4_gpu::Dsv4Buf<u8>, U: dsv4_gpu::Dsv4Buf<u32>>(
        &self,
        layer: &Dsv4GpuLayer,
        st: &mut Dsv4AttnState,
        x: &X,
        s: usize,
        start_pos: usize,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<(X, I, usize)> {
        let (dim, qlr, nh, hd, rd, win) =
            (cfg.dim, cfg.q_lora_rank, cfg.n_heads, cfg.head_dim, cfg.rope_head_dim, cfg.window_size);
        let eps = cfg.norm_eps;
        let kind = layer.kind;
        // Per-kind RoPE table (§B.1.3): SWA plain θ vs CSA/HCA YaRN. `rope_for` picks the
        // multikind table when present (full trunk), else the lane's single-kind `rope`.
        let rope = self.rope_for(kind);

        // Q path: quantize x once (K=4096 shared by wq_a and wkv — same codes, bit-identical).
        // R3A.1 E2: wq_a + wkv in ONE fused launch at decode/verify widths (independent ops,
        // same activation — per-element chains identical to the two separate launches, gated).
        let (xc, xsa) = self.quant_g128::<X, C>(x, s, dim)?;
        let (qr_pre, kv) = if s <= 16 {
            self.fp8_bsb2q_rows(&layer.wq_a, &layer.wkv, &xc, &xsa, s)?
        } else {
            (self.fp8_bsb_rows(&layer.wq_a, &xc, &xsa, s)?, self.fp8_bsb_rows(&layer.wkv, &xc, &xsa, s)?)
        };
        let pair_seq = crate::dsv4_gpu::env_flag_once(crate::opt!("pair-seq"));
        let s_i = s as i32;
        let (sp_i, nh_i, rd_i) = (start_pos as i32, nh as i32, rd as i32);
        let rows_q = s * nh;
        let (rows_i, hd_i) = (rows_q as i32, hd as i32);
        // Seq arm only: the q-position iota array (pair arms compute positions inline).
        // Lives to the de-rotation site.
        let pos_q_dev = if pair_seq {
            Some(dsv4_gpu::iota_positions::<I>(&self.dev, &self.spine, &self.stream, start_pos as i32, 1, nh as i32, rows_q)?)
        } else { None };
        // R3A.1 E2 rung 2: the q/kv RMSNorms are independent (same-stage outputs of the fused
        // wq_a/wkv GEMM) — ONE pair launch, bitwise == two singles (the spine pair gate).
        let (qr, kv) = if !pair_seq {
            let qr = X::alloc_zeros(&self.dev, self.stream.stream, s * qlr)?;
            let kv_n = X::alloc_zeros(&self.dev, self.stream.stream, s * hd)?;
            let (qlr_i, hd_i) = (qlr as i32, hd as i32);
            dsv4_launch!(self.spine, "dsv4_rmsnorm_pair_b", self.stream.stream, (s as u32, 2, 1), (256, 1, 1), 0,
                (&qr, &qr_pre, &layer.q_norm, &s_i, &qlr_i,
                 &kv_n, &kv, &layer.kv_norm, &s_i, &hd_i, &eps))?;
            (qr, kv_n)
        } else {
            (self.rmsnorm(&qr_pre, &layer.q_norm, s, qlr, eps)?,
             self.rmsnorm(&kv, &layer.kv_norm, s, hd, eps)?)
        };
        let (qrc, qrsa) = self.quant_g128::<X, C>(&qr, s, qlr)?;
        let q = self.fp8_bsb_rows::<X, C>(&layer.wq_b, &qrc, &qrsa, s)?;
        if !pair_seq && s <= 16 {
            // E2 rung 5 (Tier 1.2): rescale(q) + rope(q,kv) + KV-sim in ONE launch
            // (3 -> 1; bitwise == the separate kernels — the spine fused-tail gate).
            dsv4_launch!(self.attn, "dsv4_rescale_rope_sim_b", self.stream.stream,
                ((rows_q + s) as u32, 1, 1), (256, 1, 1), 0,
                (&q, &kv, &rope.cos, &rope.sin, &sp_i, &nh_i, &rows_i, &s_i, &hd_i, &rd_i, &eps))?;
        } else {
        // §B.1.1 per-head weight-free RMS rescale (bit-exact vs dsv4_cpu::attn_qkv).
        dsv4_launch!(self.attn, "dsv4_attn_rescale_b", self.stream.stream, (rows_q as u32, 1, 1), (256, 1, 1), 0,
            (&q, &rows_i, &hd_i, &eps))?;
        if !pair_seq {
            // R3A.1 E2 rung 2: ONE rope launch for q+kv with INLINE positions (identical
            // integers to the iota arrays it replaces; per-element math unchanged). Replaces
            // 2 iota + 2 rope launches. q: p = start_pos + row/nh; kv: p = start_pos + row.
            dsv4_launch!(self.spine, "dsv4_rope_pair_b", self.stream.stream,
                ((((rows_q + 7) / 8) as u32), 2, 1), (256, 1, 1), 0,
                (&q, &kv, &rope.cos, &rope.sin, &sp_i, &nh_i, &rows_i, &s_i, &hd_i, &rd_i))?;
        } else {
            // RoPE on q's last 64 dims (every head of token t at position start_pos + t).
            dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(rows_q * 32), 1, 1), (256, 1, 1), 0,
                (&q, &rope.cos, &rope.sin, pos_q_dev.as_ref().unwrap(), &rows_i, &hd_i, &rd_i, &0i32))?;
            let pos_kv_dev = dsv4_gpu::iota_positions::<I>(&self.dev, &self.spine, &self.stream, start_pos as i32, 1, 1, s)?;
            dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(s * 32), 1, 1), (256, 1, 1), 0,
                (&kv, &rope.cos, &rope.sin, &pos_kv_dev, &s_i, &hd_i, &rd_i, &0i32))?;
        }
        let stride_i = hd as i32;
        let nope_i = (hd - rd) as i32;
        dsv4_launch!(self.attn, "dsv4_kv_sim_g64_strided", self.stream.stream,
            (ceil256(s * ((hd - rd) / 64) * 32), 1, 1), (256, 1, 1), 0,
            (&kv, &s_i, &stride_i, &nope_i))?;
        }

        // ---- ring write + index list + compressor + gather (prefill batched; start_pos>0 sequential) ----
        let scale = (hd as f64).powf(-0.5) as f32;
        // Pad s up to a multiple of 16 for dsv4_olo_proj_tc_b's [16,8] WMMA tile grid: the kernel
        // launches ceil(s/16) M-tiles and each reads/writes a full 16-row slab of o/oflat, so an
        // unpadded s%16!=0 makes the last tile stride OOB (CUDA_ERROR_ILLEGAL_ADDRESS — silent
        // single-process where the OOB lands in mapped adjacent memory, fatal under TP=2's layout
        // where the 128-expert heap is smaller). Phantom rows are never read by downstream ops
        // (de-rotation, quant_g128, fp8_bsb all use the real `s`/`rows_q` row counts).
        let s_pad = ((s + 15) / 16) * 16;
        let rows_q_pad = s_pad * nh;
        let o = X::alloc_zeros(&self.dev, self.stream.stream, rows_q_pad * hd)?;
        let win_i = win as i32;

        // R2.0 instrument: record which attention path each (layer, start_pos, s) takes —
        // the audit's missing "is the batched chunk-prefill path engaged for chunks 2+?" probe.
        // Gated by --prefill-trace so it is free in production. The sequential per-token loop
        // (start_pos>0) is the structural suspect #1 (§3b #1): chunks 2+ ride it, not the batched
        // prefill path, which would explain the ~27–45 prefill rate sagging with depth.
        let _pf_trace = crate::opts::var(crate::opt!("prefill-trace")).ok().is_some();
        let (idxs, t): (I, usize) = if start_pos == 0 {
            // ===================== PREFILL (start_pos == 0): batched — UNCHANGED (lane 3A/3C + trunk gate) =====================
            // §B.2 ring write BEFORE attention (current token attends to itself), rotated for S>win.
            let lo = if s > win { s - win } else { 0 };
            let sp_zero = 0i32;
            dsv4_launch!(self.attn, "dsv4_ring_write_b", self.stream.stream, ((s - lo) as u32, 1, 1), (256, 1, 1), 0,
                (&st.kv_cache, &kv, &s_i, &sp_zero, &win_i, &hd_i))?;
            let nh16 = (nh / 16) as u32;
            match kind {
                dsv4_load::LayerKind::Swa => {
                    let t_win = s.min(win);
                    let idxs = I::alloc_zeros(&self.dev, self.stream.stream, s * t_win)?;
                    let t_win_i = t_win as i32;
                    dsv4_launch!(self.attn, "dsv4_window_idxs_b", self.stream.stream,
                        (ceil256(s * t_win), 1, 1), (256, 1, 1), 0,
                        (&idxs, &s_i, &sp_zero, &win_i, &t_win_i))?;
                    let (t_i, n_i) = (t_win as i32, s as i32);
                    dsv4_launch!(self.spine, "dsv4_gather_attn", self.stream.stream,
                        (s as u32, 1u32, nh16), (256, 1, 1), GATHER_SMEM,
                        (&q, &kv, &o, &layer.sink, &idxs, &t_i, &n_i, &scale))?;
                    (idxs, t_win)
                }
                dsv4_load::LayerKind::Csa | dsv4_load::LayerKind::Hca => {
                    let comp_ks = self.comp.as_ref().expect("CSA/HCA needs comp kernels");
                    let compressor = st.attn_compressor.as_mut().expect("CSA/HCA needs attn_compressor");
                    let ratio = compressor.spec.ratio;
                    let nb_committed = compressor.prefill::<X, I>(&self.dev, comp_ks, &self.stream, x, s, rope)?;
                    let t_win = s.min(win);
                    let t_comp = match kind {
                        dsv4_load::LayerKind::Csa => {
                            let indexer = st.indexer.as_mut().expect("CSA needs indexer");
                            let f_bsb = &self.bk["gemm_dsv4_fp8_bsb2"];
                            indexer.forward::<X, F, I, U>(&self.dev, comp_ks, &self.stream, f_bsb, x, &qr, s, 0, s, rope)?
                        }
                        dsv4_load::LayerKind::Hca => s / ratio,
                        _ => unreachable!(),
                    };
                    let t_total = t_win + t_comp;
                    let idxs = I::alloc_zeros(&self.dev, self.stream.stream, s * t_total)?;
                    let (t_win_i, t_total_i) = (t_win as i32, t_total as i32);
                    dsv4_launch!(self.attn, "dsv4_window_idxs_strided_b", self.stream.stream,
                        (ceil256(s * t_win), 1, 1), (256, 1, 1), 0,
                        (&idxs, &s_i, &sp_zero, &win_i, &t_win_i, &t_total_i))?;
                    if t_comp > 0 {
                        match kind {
                            dsv4_load::LayerKind::Hca => {
                                let (offset_i, ratio_i, t_comp_i, col_off_i) =
                                    (s as i32, ratio as i32, t_comp as i32, t_win as i32);
                                dsv4_launch!(self.attn, "dsv4_compress_idxs_b", self.stream.stream,
                                    (ceil256(s * t_comp), 1, 1), (256, 1, 1), 0,
                                    (&idxs, &s_i, &sp_zero, &ratio_i, &offset_i, &t_comp_i, &t_total_i, &col_off_i))?;
                            }
                            dsv4_load::LayerKind::Csa => {
                                let (k_i, col_off_i) = (t_comp as i32, t_win as i32);
                                let indexer = st.indexer.as_ref().expect("CSA needs indexer");
                                dsv4_launch!(self.attn, "dsv4_idxs_place_b", self.stream.stream,
                                    (ceil256(s * t_comp), 1, 1), (256, 1, 1), 0,
                                    (&idxs, indexer.idx_dev(), &s_i, &k_i, &t_total_i, &col_off_i))?;
                            }
                            _ => unreachable!(),
                        }
                    }
                    let scratch = st.kv_attn_scratch.as_ref().unwrap();
                    let comp_bytes = nb_committed * hd * 2;
                    if comp_bytes > 0 {
                        // Item 3: the compressor's cache is aliased to kv_cache[win..]. The
                        // epilogue wrote there; read the comp_cache tail from the alias.
                        let comp_src = if compressor.cache_alias_ptr != 0 {
                            compressor.cache_alias_ptr
                        } else {
                            *compressor.cache.device_ptr()
                        };
                        unsafe {
                            result::memcpy_dtod_async(*scratch.device_ptr(), kv.dptr(), s * hd * 2, self.stream.stream)
                                .map_err(|e| anyhow!("prefill kv→scratch dtod: {e}"))?;
                            result::memcpy_dtod_async((*scratch.device_ptr()) + (s * hd * 2) as u64, comp_src, comp_bytes, self.stream.stream)
                                .map_err(|e| anyhow!("prefill comp→scratch dtod: {e}"))?;
                        }
                    }
                    let (t_i, n_i) = (t_total as i32, (s + nb_committed) as i32);
                    dsv4_launch!(self.spine, "dsv4_gather_attn", self.stream.stream,
                        (s as u32, 1u32, nh16), (256, 1, 1), GATHER_SMEM,
                        (&q, scratch, &o, &layer.sink, &idxs, &t_i, &n_i, &scale))?;
                    (idxs, t_total)
                }
            }
        } else if start_pos > 0 && s >= win && start_pos % win == 0 && s % 128 == 0 {
            // ===================== R2.1 BATCHED CONTINUATION (start_pos > 0, prefill chunk 2+) =====================
            // The opt-ladder-1 batched the COMPRESSOR for chunks 2+; this extends the batching
            // to the ATTENTION (window ring-write + gather) — eliminating the sequential per-token
            // loop that ran for every chunk after the first (the root cause of the ~27–45 prefill
            // rate sagging with depth). Gate: start_pos > 0 && s >= win && start_pos % win == 0
            // (prefill chunks have s=PREFILL_CHUNK=4096 >> win=128, start_pos is a multiple of
            // 4096; verify has s ≤ ~6 < 128, so this branch is never reached for verify/decode).
            //
            // Build a UNIFIED scratch = [prefix window (win rows from ring) | new kv (s rows) |
            // comp_cache (nb rows from the aliased kv_cache tail)] and gather from it with logical
            // indices. The window for row r (absolute position start_pos+r) is the contiguous
            // scratch rows [r+1 .. r+win] — prefix fills the early entries (no masking needed).
            // This is bitwise-identical to one-shot prefill (same positions, same order: the
            // gather sums over the same KV vectors in the same oldest→newest order). The ring is
            // written AFTER the gather (for future chunks/decodes); the prefix is saved to scratch
            // BEFORE the ring write clobbers it.
            let nh16 = (nh / 16) as u32;
            let sp_i = start_pos as i32;
            let win_plus_s = win + s;

            // 1. Compressor prefill_at (batched — already proven by opt-ladder-1). For CSA the
            //    indexer.forward below also calls its own compressor; the ATTENTION compressor
            //    here writes the comp_cache to the aliased kv_cache[win..] tail.
            let (nb_committed, t_comp, idxs): (usize, usize, I) = match kind {
                dsv4_load::LayerKind::Swa => {
                    // SWA: no compressor, no comp_cache. scratch = [prefix(win) | new(s)].
                    (0, 0, I::alloc_zeros(&self.dev, self.stream.stream, s * win)?)
                }
                dsv4_load::LayerKind::Csa | dsv4_load::LayerKind::Hca => {
                    let comp_ks = self.comp.as_ref().expect("CSA/HCA needs comp kernels");
                    let compressor = st.attn_compressor.as_mut().expect("CSA/HCA needs attn_compressor");
                    let ratio = compressor.spec.ratio;
                    let nb = compressor.prefill_at::<X, I>(&self.dev, comp_ks, &self.stream, x, s, start_pos, rope)?;
                    let nb_committed = (start_pos + s) / ratio;
                    let t_comp = match kind {
                        dsv4_load::LayerKind::Csa => {
                            let indexer = st.indexer.as_mut().expect("CSA needs indexer");
                            let f_bsb = &self.bk["gemm_dsv4_fp8_bsb2"];
                            // offset = win + s (comp_cache starts at scratch row win+s in the
                            // unified buffer; the remask kernel adds offset to block indices).
                            indexer.forward::<X, F, I, U>(&self.dev, comp_ks, &self.stream, f_bsb, x, &qr, s, start_pos, win + s, rope)?
                        }
                        dsv4_load::LayerKind::Hca => nb_committed,
                        _ => unreachable!(),
                    };
                    let t_total = win + t_comp;
                    (nb_committed, t_comp, I::alloc_zeros(&self.dev, self.stream.stream, s * t_total)?)
                }
            };
            let t_total = win + t_comp;

            // 2. Build the unified scratch: [prefix(win) | new(s) | comp(nb)].
            let scratch = st.kv_attn_scratch.as_ref().unwrap();
            // 2a. Copy prefix window (win rows) from the ring (kv_cache[0..win-1]).
            unsafe {
                result::memcpy_dtod_async(
                    *scratch.device_ptr(),
                    *st.kv_cache.device_ptr(),
                    win * hd * 2,
                    self.stream.stream,
                ).map_err(|e| anyhow!("cont prefill ring→scratch dtod: {e}"))?;
            }
            // 2b. Copy new kv (s rows) into scratch[win..win+s-1].
            unsafe {
                result::memcpy_dtod_async(
                    (*scratch.device_ptr()) + (win * hd * 2) as u64,
                    kv.dptr(),
                    s * hd * 2,
                    self.stream.stream,
                ).map_err(|e| anyhow!("cont prefill kv→scratch dtod: {e}"))?;
            }
            // 2c. Copy comp_cache (nb_committed rows) from the aliased kv_cache[win..] tail
            //     into scratch[win+s..win+s+nb-1]. (CSA/HCA only; SWA skips this.)
            if nb_committed > 0 {
                let comp_src = if let Some(ac) = st.attn_compressor.as_ref() {
                    if ac.cache_alias_ptr != 0 { ac.cache_alias_ptr } else { *ac.cache.device_ptr() }
                } else { unreachable!() };
                let comp_bytes = nb_committed * hd * 2;
                unsafe {
                    result::memcpy_dtod_async(
                        (*scratch.device_ptr()) + (win_plus_s * hd * 2) as u64,
                        comp_src,
                        comp_bytes,
                        self.stream.stream,
                    ).map_err(|e| anyhow!("cont prefill comp→scratch dtod: {e}"))?;
                }
            }

            // 3. Build window idxs (batched, continuation regime: v = row + 1 + j).
            let (t_win_i, t_total_i) = (win as i32, t_total as i32);
            dsv4_launch!(self.attn, "dsv4_window_idxs_strided_b", self.stream.stream,
                (ceil256(s * win), 1, 1), (256, 1, 1), 0,
                (&idxs, &s_i, &sp_i, &win_i, &t_win_i, &t_total_i))?;

            // 4. Build compress idxs (HCA only — CSA already placed by the indexer above).
            if t_comp > 0 {
                match kind {
                    dsv4_load::LayerKind::Hca => {
                        let ratio = st.attn_compressor.as_ref().expect("HCA").spec.ratio;
                        let (offset_i, ratio_i, t_comp_i, col_off_i) =
                            (win_plus_s as i32, ratio as i32, t_comp as i32, win as i32);
                        dsv4_launch!(self.attn, "dsv4_compress_idxs_b", self.stream.stream,
                            (ceil256(s * t_comp), 1, 1), (256, 1, 1), 0,
                            (&idxs, &s_i, &sp_i, &ratio_i, &offset_i, &t_comp_i, &t_total_i, &col_off_i))?;
                    }
                    dsv4_load::LayerKind::Csa => {
                        // The indexer's remask already wrote block_index + (win+s) into idxs.
                        // Place them into the unified idxs buffer at column win.
                        let (k_i, col_off_i) = (t_comp as i32, win as i32);
                        let indexer = st.indexer.as_ref().expect("CSA needs indexer");
                        dsv4_launch!(self.attn, "dsv4_idxs_place_b", self.stream.stream,
                            (ceil256(s * t_comp), 1, 1), (256, 1, 1), 0,
                            (&idxs, indexer.idx_dev(), &s_i, &k_i, &t_total_i, &col_off_i))?;
                    }
                    _ => unreachable!(),
                }
            }

            // 5. Gather from the unified scratch (batched over all s rows).
            let (t_i, n_i) = (t_total as i32, (win_plus_s + nb_committed) as i32);
            dsv4_launch!(self.spine, "dsv4_gather_attn", self.stream.stream,
                (s as u32, 1u32, nh16), (256, 1, 1), GATHER_SMEM,
                (&q, scratch, &o, &layer.sink, &idxs, &t_i, &n_i, &scale))?;

            // 6. Ring write: write the new kv rows into the ring (for future chunks/decodes).
            //    The gather read from scratch (not the ring), so the ring write can happen here
            //    without clobbering anything the gather needed. dsv4_ring_write_b with
            //    start_pos > 0 writes all s rows; the final ring state has the last win rows at
            //    the correct physical slots (wrapping ensures the most recent write wins).
            dsv4_launch!(self.attn, "dsv4_ring_write_b", self.stream.stream,
                (s as u32, 1, 1), (256, 1, 1), 0,
                (&st.kv_cache, &kv, &s_i, &sp_i, &win_i, &hd_i))?;

            (idxs, t_total)
        } else if start_pos > 0 && s <= win
            && !crate::dsv4_gpu::env_flag_once(crate::opt!("verify-seq"))
            && !crate::dsv4_gpu::env_flag_once(crate::opt!("graph"))
        {
            // ===================== SESSION-10 FUSED GATHER (verify/decode, ONE launch/layer) =====================
            // Queue #5: `dsv4_fused_gather_b` replaces, for the verify (s=6) and decode (s=1)
            // shapes at any start_pos, the assembly [window idxs + compress idxs/place + the
            // unified-scratch d2d copies + dsv4_gather_attn] — the index lists are computed
            // in-kernel and the KV is read directly from the ring / the layer's new kv / the
            // comp tail. Per-(row, head) chains are bitwise == the SEQ/R4 assemblies (same
            // key order, tile boundaries, online-softmax and P·V chains — the masked entries
            // keep their tile positions as p=0 / fma(0,0) no-ops). Compressor + indexer stay
            // batched (identical to the assembly arms; the indexer remask uses the batched
            // id space win+s — uniform with comp_off below). --verify-seq (the documented
            // A/B arm) and --graph (the graph policies capture the assembly kernels) keep
            // the old paths. The returned idx buffer is a placeholder (topk_idx is consumed
            // only by the debug replay tool, which runs the assembly path).
            let mut nb_max = 0usize;
            let mut t_comp_max = 0usize;
            let mut idx_dev: u64 = 0;
            let mut ratio_use = 0usize;
            if matches!(kind, dsv4_load::LayerKind::Csa | dsv4_load::LayerKind::Hca) {
                let comp_ks = self.comp.as_ref().expect("CSA/HCA needs comp kernels");
                let compressor = st.attn_compressor.as_mut().expect("CSA/HCA needs attn_compressor");
                let ratio = compressor.spec.ratio;
                // kernel dispatch: ratio > 0 selects the HCA arange path; CSA (indexer
                // idx_dev) needs ratio == 0 (the kernel branches idx_csa != 0 second).
                ratio_use = if kind == dsv4_load::LayerKind::Hca { ratio } else { 0 };
                if s >= ratio && start_pos % ratio == 0 {
                    compressor.prefill_at::<X, I>(&self.dev, comp_ks, &self.stream, x, s, start_pos, rope)?;
                } else {
                    compressor.forward_tokens::<X, F, I, U>(&self.dev, comp_ks, &self.stream, x, s, start_pos, rope)?;
                }
                nb_max = (start_pos + s) / ratio;
                t_comp_max = match kind {
                    dsv4_load::LayerKind::Csa => {
                        let indexer = st.indexer.as_mut().expect("CSA needs indexer");
                        let f_bsb = &self.bk["gemm_dsv4_fp8_bsb2"];
                        let k = indexer.forward::<X, F, I, U>(&self.dev, comp_ks, &self.stream, f_bsb, x, &qr, s, start_pos, win + s, rope)?;
                        idx_dev = *indexer.idx_dev().device_ptr();
                        k
                    }
                    _ => nb_max,
                };
            }
            // ONE fused gather over [ring | new kv | comp tail]; index lists + scratch
            // copies computed in-kernel (bitwise == the assembly arms).
            let nh16 = (nh / 16) as u32;
            let ring = &st.kv_cache;
            let comp_tail = match st.attn_compressor.as_ref() {
                Some(ac) if ac.cache_alias_ptr != 0 => ac.cache_alias_ptr,
                Some(ac) => *ac.cache.device_ptr(),
                None => *ring.device_ptr(), // SWA: never dereferenced (t_comp_max == 0)
            };
            let (s_i, sp_i, win_i, tcm_i, ratio_i, off_i, k_i) = (
                s as i32, start_pos as i32, win as i32,
                t_comp_max as i32, ratio_use as i32, (win + s) as i32, t_comp_max as i32,
            );
            dsv4_launch!(self.spine, "dsv4_fused_gather_b", self.stream.stream,
                (s as u32, 1u32, nh16), (256, 1, 1), GATHER_SMEM,
                (&q, ring, &kv, &comp_tail, &idx_dev, &o, &layer.sink,
                 &s_i, &sp_i, &win_i, &ratio_i, &off_i, &tcm_i, &k_i, &scale))?;
            // Ring write AFTER the gather (batched last-wins — identical to the R4 arm).
            dsv4_launch!(self.attn, "dsv4_ring_write_b", self.stream.stream,
                (s as u32, 1, 1), (256, 1, 1), 0,
                (ring, &kv, &s_i, &sp_i, &win_i, &hd_i))?;
            let t_total = win + t_comp_max;
            let idxs = I::alloc_zeros(&self.dev, self.stream.stream, s * t_total)?;
            (idxs, t_total)
        } else if start_pos > 0 && s > 1 && start_pos >= win
            && !crate::dsv4_gpu::env_flag_once(crate::opt!("verify-seq"))
        {
            // ===================== R4 BATCHED VERIFY (s>1, start_pos >= win) =====================
            // The R2.1 batched-continuation machinery generalized to arbitrary alignment by a
            // rotated, position-ordered prefix copy. The compressor/indexer state work stays
            // batched (identical to the SEQ arm's front part); the window ring's per-token
            // sequencing (3-4 launches × s rows) becomes ONE s-wide gather over a scratch the
            // ring writes cannot clobber, then ONE batched ring write (same last-wins
            // semantics as the prefill/R2.1 arms). Per-row attended positions/KV/order are
            // identical to the SEQ arm — the equivalence gate compares the arms bitwise.
            let nh16 = (nh / 16) as u32;
            let sp_i = start_pos as i32;
            let win_plus_s = win + s;

            // 1. Compressor + (CSA) indexer — batched, identical to the SEQ arm.
            let mut nb_max = 0usize;
            if matches!(kind, dsv4_load::LayerKind::Csa | dsv4_load::LayerKind::Hca) {
                let comp_ks = self.comp.as_ref().expect("CSA/HCA needs comp kernels");
                let compressor = st.attn_compressor.as_mut().expect("CSA/HCA needs attn_compressor");
                let ratio = compressor.spec.ratio;
                if s >= ratio && start_pos % ratio == 0 {
                    compressor.prefill_at::<X, I>(&self.dev, comp_ks, &self.stream, x, s, start_pos, rope)?;
                } else {
                    compressor.forward_tokens::<X, F, I, U>(&self.dev, comp_ks, &self.stream, x, s, start_pos, rope)?;
                }
                nb_max = (start_pos + s) / ratio;
            }

            // 2. Unified scratch [prefix(win, position-ordered) | new(s) | comp(nb)].
            let t_comp = match kind {
                dsv4_load::LayerKind::Swa => 0usize,
                dsv4_load::LayerKind::Csa => {
                    let comp_ks = self.comp.as_ref().expect("CSA needs comp kernels");
                    let indexer = st.indexer.as_mut().expect("CSA needs indexer");
                    let f_bsb = &self.bk["gemm_dsv4_fp8_bsb2"];
                    indexer.forward::<X, F, I, U>(&self.dev, comp_ks, &self.stream, f_bsb, x, &qr, s, start_pos, win + s, rope)?
                }
                dsv4_load::LayerKind::Hca => nb_max,
            };
            let t_total = win + t_comp;
            let idxs = I::alloc_zeros(&self.dev, self.stream.stream, s * t_total)?;
            let scratch = match st.kv_attn_scratch.as_ref() {
                Some(sc) => sc.clone(),
                None => self.dev.alloc_zeros::<bf16>((win + s + nb_max) * hd)?,
            };
            let sp = start_pos % win;
            unsafe {
                // 2a. rotated prefix copy (position-ordered: scratch row m = position
                // start_pos-win+m for every alignment).
                result::memcpy_dtod_async(
                    *scratch.device_ptr(),
                    *st.kv_cache.device_ptr() + (sp * hd * 2) as u64,
                    (win - sp) * hd * 2,
                    self.stream.stream,
                ).map_err(|e| anyhow!("verify prefix rot-a: {e}"))?;
                result::memcpy_dtod_async(
                    (*scratch.device_ptr()) + ((win - sp) * hd * 2) as u64,
                    *st.kv_cache.device_ptr(),
                    sp * hd * 2,
                    self.stream.stream,
                ).map_err(|e| anyhow!("verify prefix rot-b: {e}"))?;
                // 2b. the new kv rows (batched projection from the front of attn_forward).
                result::memcpy_dtod_async(
                    (*scratch.device_ptr()) + (win * hd * 2) as u64,
                    kv.dptr(),
                    s * hd * 2,
                    self.stream.stream,
                ).map_err(|e| anyhow!("verify kv→scratch: {e}"))?;
                // 2c. the committed compressor tail (CSA/HCA).
                if nb_max > 0 {
                    let ac = st.attn_compressor.as_ref().unwrap();
                    let comp_src = if ac.cache_alias_ptr != 0 { ac.cache_alias_ptr } else { *ac.cache.device_ptr() };
                    result::memcpy_dtod_async(
                        (*scratch.device_ptr()) + (win_plus_s * hd * 2) as u64,
                        comp_src,
                        nb_max * hd * 2,
                        self.stream.stream,
                    ).map_err(|e| anyhow!("verify comp→scratch: {e}"))?;
                }
            }

            // 3. window idxs (all rows at once) + compress part (per-row causal by the
            //    existing kernels' own semantics).
            let (s_i, win_i, t_total_i) = (s as i32, win as i32, t_total as i32);
            dsv4_launch!(self.attn, "dsv4_window_idxs_verify_b", self.stream.stream,
                (ceil256(s * win), 1, 1), (256, 1, 1), 0,
                (&idxs, &s_i, &win_i, &win_i, &t_total_i))?;
            match kind {
                dsv4_load::LayerKind::Swa => {}
                dsv4_load::LayerKind::Hca => {
                    if t_comp > 0 {
                        let ratio = st.attn_compressor.as_ref().expect("HCA").spec.ratio;
                        let (offset_i, ratio_i, t_comp_i, col_off_i) =
                            (win_plus_s as i32, ratio as i32, t_comp as i32, win as i32);
                        dsv4_launch!(self.attn, "dsv4_compress_idxs_b", self.stream.stream,
                            (ceil256(s * t_comp), 1, 1), (256, 1, 1), 0,
                            (&idxs, &s_i, &sp_i, &ratio_i, &offset_i, &t_comp_i, &t_total_i, &col_off_i))?;
                    }
                }
                dsv4_load::LayerKind::Csa => {
                    if t_comp > 0 {
                        let (k_i, col_off_i) = (t_comp as i32, win as i32);
                        let indexer = st.indexer.as_ref().expect("CSA needs indexer");
                        dsv4_launch!(self.attn, "dsv4_idxs_place_b", self.stream.stream,
                            (ceil256(s * t_comp), 1, 1), (256, 1, 1), 0,
                            (&idxs, indexer.idx_dev(), &s_i, &k_i, &t_total_i, &col_off_i))?;
                    }
                }
            }

            // 4. ONE s-wide gather over the scratch.
            let (t_i, n_i) = (t_total as i32, (win_plus_s + nb_max) as i32);
            dsv4_launch!(self.spine, "dsv4_gather_attn", self.stream.stream,
                (s as u32, 1u32, nh16), (256, 1, 1), GATHER_SMEM,
                (&q, scratch, &o, &layer.sink, &idxs, &t_i, &n_i, &scale))?;

            // 5. Ring write AFTER the gather (last-wins semantics, identical to the SEQ
            //    arm's sequential per-token writes — the prefill/R2.1 arms prove the batch form).
            dsv4_launch!(self.attn, "dsv4_ring_write_b", self.stream.stream,
                (s as u32, 1, 1), (256, 1, 1), 0,
                (&st.kv_cache, &kv, &s_i, &sp_i, &win_i, &hd_i))?;

            (idxs, t_total)
        } else {
            // ===================== DECODE / VERIFY (start_pos > 0): sequential per-token loop =====================
            // Reference-faithful: the reference's start_pos>0 path is single-token decode
            // (model.py:535 `kv.squeeze(1)` asserts seqlen==1). A batched ring-write of s>1 verify
            // tokens would clobber prefix slots that earlier rows still need (slots [start_pos%win ..
            // (start_pos+s-1)%win] hold prefix positions inside row 0's 128-window). Writing token r →
            // attending row r → writing token r+1 matches N sequential decodes exactly; s==1 decode
            // is one iteration (bit-identical to the former batched s==1 path). Each row also gets its
            // own (start_pos+r) for the window ring order and the HCA compress count — the batched
            // path used a single start_pos (wrong for verify rows >0). The attention compressor cache
            // and the CSA indexer are append-only (row r reads only prefix + rows <r), so they run
            // batched ONCE before the loop; only the window ring needs per-token sequencing.
            let mut nb_max = 0usize;
            if matches!(kind, dsv4_load::LayerKind::Csa | dsv4_load::LayerKind::Hca) {
                let comp_ks = self.comp.as_ref().expect("CSA/HCA needs comp kernels");
                let compressor = st.attn_compressor.as_mut().expect("CSA/HCA needs attn_compressor");
                let ratio = compressor.spec.ratio;
                // Batched prefill pool when the chunk boundary aligns (start_pos%ratio==0 AND
                // s>=ratio) — the parallel chunk-prefill speedup (one batched pool launch
                // instead of s sequential decode+sync round-trips). Sequential decode otherwise
                // (verify/decode with small s — the pool can't run the per-token state machine).
                // Bitwise-identical to one-shot (§12.B.5 — the carry flag handles block 0's
                // overlap from the frontier).
                if s >= ratio && start_pos % ratio == 0 {
                    compressor.prefill_at::<X, I>(&self.dev, comp_ks, &self.stream, x, s, start_pos, rope)?;
                } else {
                    compressor.forward_tokens::<X, F, I, U>(&self.dev, comp_ks, &self.stream, x, s, start_pos, rope)?;
                }
                nb_max = (start_pos + s) / ratio;
                // Item 3 alias: the compressor's epilogue already wrote its cache rows
                // directly to kv_cache[win..] (the alias). No d2d mirror needed — the
                // gather reads from the unified kv_cache (ring + tail) contiguously.
            }
            let (t_total, idxs_out): (usize, I) = match kind {
                dsv4_load::LayerKind::Swa => (win, I::alloc_zeros(&self.dev, self.stream.stream, s * win)?),
                dsv4_load::LayerKind::Csa => {
                    let comp_ks = self.comp.as_ref().expect("CSA needs comp kernels");
                    let indexer = st.indexer.as_mut().expect("CSA needs indexer");
                    let f_bsb = &self.bk["gemm_dsv4_fp8_bsb2"];
                    let k = indexer.forward::<X, F, I, U>(&self.dev, comp_ks, &self.stream, f_bsb, x, &qr, s, start_pos, win, rope)?;
                    let t_total = win + k;
                    // CUDA-graph mode: alloc for the CAP (k ≤ index_topk) so replay never
                    // outgrows the baked buffer; the layout args below stay at the live t.
                    let cap = if crate::dsv4_gpu::env_flag_once(crate::opt!("graph")) {
                        s * (win + indexer.index_topk)
                    } else {
                        s * t_total
                    };
                    let idxs = I::alloc_zeros(&self.dev, self.stream.stream, cap)?;
                    // Place the batched indexer [s, k] output at col win for ALL rows (per-row causal
                    // already enforced by score/remask — selections are a pure function of the prefix).
                    let (k_i, t_total_i, col_off_i) = (k as i32, t_total as i32, win as i32);
                    dsv4_launch!(self.attn, "dsv4_idxs_place_b", self.stream.stream,
                        (ceil256(s * k), 1, 1), (256, 1, 1), 0,
                        (&idxs, indexer.idx_dev(), &s_i, &k_i, &t_total_i, &col_off_i))?;
                    (t_total, idxs)
                }
                dsv4_load::LayerKind::Hca => {
                    let ratio = st.attn_compressor.as_ref().expect("HCA needs compressor").spec.ratio;
                    let t_comp_max = (start_pos + s) / ratio;
                    let t_total = win + t_comp_max;
                    // CUDA-graph mode: alloc for the compressor cache CAPACITY (see CSA note).
                    let cap = if crate::dsv4_gpu::env_flag_once(crate::opt!("graph")) {
                        s * (win + st.attn_compressor.as_ref().expect("HCA").cache_rows)
                    } else {
                        s * t_total
                    };
                    (t_total, I::alloc_zeros(&self.dev, self.stream.stream, cap)?)
                }
            };
            let gather_n = (win + nb_max) as i32;
            let t_total_i = t_total as i32;
            let nh16 = (nh / 16) as u32;
            let one_i = 1i32;
            let t_win_i = win as i32;
            for r in 0..s {
                let sp_r = start_pos + r;
                let sp_r_i = sp_r as i32;
                // (1) ring write kv[r] → slot sp_r % win (write BEFORE attention — self-attend).
                let kv_row = kv.view(r * hd, hd);
                dsv4_launch!(self.attn, "dsv4_ring_write_b", self.stream.stream, (1u32, 1, 1), (256, 1, 1), 0,
                    (&st.kv_cache, &kv_row, &one_i, &sp_r_i, &win_i, &hd_i))?;
                // (2) window idxs for row r at sp=start_pos+r (into idxs_out[r, 0..win]); HCA compress
                //     idxs at idxs_out[r, win..win+t_comp_r] (count grows with r). CSA compress slots
                //     were placed batched above (k uniform; per-row masking carried as −1).
                let idxs_row = idxs_out.view(r * t_total, t_total);
                dsv4_launch!(self.attn, "dsv4_window_idxs_b", self.stream.stream,
                    (ceil256(win), 1, 1), (256, 1, 1), 0,
                    (&idxs_row, &one_i, &sp_r_i, &win_i, &t_win_i))?;
                let t_r = match kind {
                    dsv4_load::LayerKind::Hca => {
                        let ratio = st.attn_compressor.as_ref().expect("HCA").spec.ratio;
                        let t_comp_r = (sp_r + 1) / ratio;
                        if t_comp_r > 0 {
                            let (offset_i, ratio_i, t_comp_r_i, col_off_i) =
                                (win as i32, ratio as i32, t_comp_r as i32, win as i32);
                            dsv4_launch!(self.attn, "dsv4_compress_idxs_b", self.stream.stream,
                                (ceil256(t_comp_r), 1, 1), (256, 1, 1), 0,
                                (&idxs_row, &one_i, &sp_r_i, &ratio_i, &offset_i, &t_comp_r_i, &t_total_i, &col_off_i))?;
                        }
                        win + t_comp_r
                    }
                    _ => t_total, // SWA: win; CSA: win+k (indexer placed batched).
                };
                // (3) gather row r over [ring + comp tail] (kv_cache). Per-row (s=1) — the ring now
                //     holds exactly prefix + tokens 0..r, as a sequential decode would.
                let q_row = q.view(r * nh * hd, nh * hd);
                let o_row = o.view(r * nh * hd, nh * hd);
                let t_r_i = t_r as i32;
                dsv4_launch!(self.spine, "dsv4_gather_attn", self.stream.stream,
                    (1u32, 1u32, nh16), (256, 1, 1), GATHER_SMEM,
                    (&q_row, &st.kv_cache, &o_row, &layer.sink, &idxs_row, &t_r_i, &gather_n, &scale))?;
            }
            (idxs_out, t_total)
        };
        if _pf_trace {
            let path = if start_pos == 0 { "batched" }
                else if start_pos > 0 && s >= win && start_pos % win == 0 { "batched-cont" }
                else { "SEQ-per-token" };
            eprintln!("[dsv4-pf] attn kind={kind:?} sp={start_pos} s={s} path={path}");
        }

        // De-rotation (inverse RoPE on the attention output, compensating K≡V leak).
        if !pair_seq {
            // R3A.1 E2 rung 2: inline q positions (identical integers), inverse=1.
            dsv4_launch!(self.spine, "dsv4_rope_q_inline_b", self.stream.stream, (ceil256(rows_q * 32), 1, 1), (256, 1, 1), 0,
                (&o, &rope.cos, &rope.sin, &sp_i, &nh_i, &rows_i, &hd_i, &rd_i, &1i32))?;
        } else {
            dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(rows_q * 32), 1, 1), (256, 1, 1), 0,
                (&o, &rope.cos, &rope.sin, pos_q_dev.as_ref().unwrap(), &rows_i, &hd_i, &rd_i, &1i32))?;
        }

        // §B.1.4 grouped-LoRA O: bf16 einsum with wo_a.view(8,1024,4096), then fp8_bsb(wo_b).
        // Uses the WMMA tensor-core kernel (dsv4_olo_proj_tc_b) for production prefill speed
        // (~30% of the long8k replay was the scalar variant). One warp per [16,16] output tile.
        // R3A.1: tiles_n matches the kernel's internal (r+15)/16 — the old (r+7)/8 launched
        // 2x the CTAs, half exiting immediately (512 dead CTAs per launch at decode).
        // NOTE (R3.3): a decode-width GEMV replacement was tried and REVERTED — the trunk is
        // width-bitwise end-to-end (a row at s≤16 must be bit-identical to the same row at
        // prefill width; the non-128-aligned chunked-prefill tail rides the sequential s=1 path
        // and is gate-compared against one-shot). Only reduction-order-preserving changes are
        // admissible at decode width.
        let (g, r, gd) = (cfg.o_groups, cfg.o_lora_rank, nh * hd / cfg.o_groups);
        let mut oflat = X::alloc_zeros(&self.dev, self.stream.stream, s_pad * g * r)?;
        anyhow::ensure!(r % 64 == 0, "olo tc4 pack needs r % 64 == 0 (r={r})");
        let tiles_m = (s + 15) / 16;
        let (s_i, g_i, r_i, gd_i, ors_i) = (s as i32, g as i32, r as i32, gd as i32, (nh * hd) as i32);
        if !exact_gemm_enabled() {
            // Item 2.5 fast path (default): fp8 einsum-class per-group BMM — the weights are
            // fp8-quantized at load (per-head-group tiles), o is quantized here, and the
            // reduction order is scheduler-chosen (pf2-class schedule). Tolerance-gated, not
            // bitwise (AGENTS §3). The --exact-gemm path below keeps the locked chains.
            let wq = layer.wo_a_q.as_ref().expect("wo_a fp8 quant must be loaded");
            olo_einsum_fast::<X, C>(self, &mut oflat, &o, wq, s, g, r, gd)?;
        } else if s > 16 {
            // R3A.4 P5: C=4-packed kernel at prefill widths — 4 n-tiles per CTA share one A
            // load (the [16,gd] slab re-read drops 4x); identical per-element chains (gated
            // bitwise vs the C=1 kernel). Decode/verify keeps C=1 (ILP-4 register pressure
            // loses at 1-warp decode geometry — E0c measured).
            let packs_n = r / 64;
            dsv4_launch!(self.attn, "dsv4_olo_proj_tc4_b", self.stream.stream,
                ((packs_n * g) as u32, tiles_m as u32, 1), (32, 1, 1), 0,
                (&oflat, &o, &layer.wo_a, &s_i, &g_i, &r_i, &gd_i, &ors_i))?;
        } else {
            let tiles_n = (r + 15) / 16;
            dsv4_launch!(self.attn, "dsv4_olo_proj_tc_b", self.stream.stream,
                ((tiles_n * g) as u32, tiles_m as u32, 1), (32, 1, 1), 0,
                (&oflat, &o, &layer.wo_a, &s_i, &g_i, &r_i, &gd_i, &ors_i))?;
        }
        let (oc, osa) = self.quant_g128::<X, C>(&oflat, s, g * r)?;
        let attn_out = self.fp8_bsb_rows(&layer.wo_b, &oc, &osa, s)?;
        Ok((attn_out, idxs, t))
    }

    /// DSpark warm (prefill) ring write (§B.10 DSparkAttention, start_pos==0 branch): compute
    /// main_kv = kv_norm(wkv(main_x)) for ALL `s` trunk positions, RoPE 0..s−1, FP8-sim the
    /// nope dims, write into the stage's 128-ring (rotated last-128 for s>win). NO attention,
    /// NO FFN — the warm pass only primes the ring. Each of the 3 stages warms with the SAME
    /// main_x (computed once from main_hidden by the caller).
    pub fn dspark_attn_warm(
        &self,
        layer: &Dsv4GpuLayer,
        st: &mut Dsv4AttnState,
        main_x: &B,
        s: usize,
        rope: &DevRope,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<()> {
        self.dspark_attn_warm_range(layer, st, main_x, s, 0, rope, cfg)
    }

    /// Range-position warm: like [`dspark_attn_warm`] but RoPE + ring-write at positions
    /// `start_pos .. start_pos+s-1` (the DSpark re-prime after a verify — writes main_kv for the
    /// committed verify positions so the draft ring stays contiguous, AGENTS §2.6).
    pub fn dspark_attn_warm_range(
        &self,
        layer: &Dsv4GpuLayer,
        st: &mut Dsv4AttnState,
        main_x: &B,
        s: usize,
        start_pos: usize,
        rope: &DevRope,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<()> {
        let (dim, hd, rd, win, eps) =
            (cfg.dim, cfg.head_dim, cfg.rope_head_dim, cfg.window_size, cfg.norm_eps);
        let (mxc, mxsa) = self.quant_g128(main_x, s, dim)?;
        let mkv = self.fp8_bsb_rows::<B, CudaSlice<u8>>(&layer.wkv, &mxc, &mxsa, s)?;
        let mkv = self.rmsnorm(&mkv, &layer.kv_norm, s, hd, eps)?;
        let pos_mk: Vec<i32> = (0..s).map(|i| (start_pos + i) as i32).collect();
        let pos_mk_dev = self.dev.htod_sync_copy(&pos_mk)?;
        let (s_i, hd_i, rd_i, sp_i) = (s as i32, hd as i32, rd as i32, start_pos as i32);
        dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(s * 32), 1, 1), (256, 1, 1), 0,
            (&mkv, &rope.cos, &rope.sin, &pos_mk_dev, &s_i, &hd_i, &rd_i, &0i32))?;
        let (stride_i, nope_i) = (hd as i32, (hd - rd) as i32);
        dsv4_launch!(self.attn, "dsv4_kv_sim_g64_strided", self.stream.stream,
            (ceil256(s * ((hd - rd) / 64) * 32), 1, 1), (256, 1, 1), 0,
            (&mkv, &s_i, &stride_i, &nope_i))?;
        // ring_write_b writes rows [lo, s) of mkv to cache[(start_pos+r)%win]; lo is 0 unless this
        // is a start_pos==0 prefill with s>win (the rotated last-128 case). Re-prime (start_pos>0)
        // always has lo=0 → writes all s rows at (start_pos+r)%win.
        let lo = if start_pos == 0 && s > win { s - win } else { 0 };
        let win_i = win as i32;
        dsv4_launch!(self.attn, "dsv4_ring_write_b", self.stream.stream, ((s - lo) as u32, 1, 1), (256, 1, 1), 0,
            (&st.kv_cache, &mkv, &s_i, &sp_i, &win_i, &hd_i))?;
        Ok(())
    }

    /// DSpark draft attention (§B.10 DSparkAttention, start_pos>0 branch). Writes main_kv at
    /// slot start_pos%win, then attends the `block` draft rows over [ring ++ draft_kv] using the
    /// 133-entry non-causal index list (window ++ 5 draft slots — every row sees all 5 drafts).
    /// `x` [block, dim] is the attn_norm output; `main_x` [1, dim] the projected trunk hidden.
    /// Returns attn_out [block, dim]. NON-CAUSAL within the block (no sequential per-row loop).
    #[allow(clippy::too_many_arguments)]
    pub fn dspark_attn_forward(
        &self,
        layer: &Dsv4GpuLayer,
        st: &mut Dsv4AttnState,
        x: &B,
        block: usize,
        start_pos: usize,
        main_x: &B,
        rope: &DevRope,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<B> {
        self.dspark_attn_forward_impl(layer, st, x, block, start_pos, main_x, rope, cfg, false).map(|r| r.0)
    }

    /// Debug (R4 audit): capture the attention's internal `q`/`kv` (draft q/kv) and `o`
    /// (gather+de-rotation output) and `oflat` (grouped-LoRA wo_a output) alongside wo_b.
    pub fn dspark_attn_forward_capture(
        &self,
        layer: &Dsv4GpuLayer,
        st: &mut Dsv4AttnState,
        x: &B,
        block: usize,
        start_pos: usize,
        main_x: &B,
        rope: &DevRope,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<(B, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        self.dspark_attn_forward_impl(layer, st, x, block, start_pos, main_x, rope, cfg, true)
    }

    fn dspark_attn_forward_impl(
        &self,
        layer: &Dsv4GpuLayer,
        st: &mut Dsv4AttnState,
        x: &B,
        block: usize,
        start_pos: usize,
        main_x: &B,
        rope: &DevRope,
        cfg: &dsv4_load::Dsv4Config,
        capture: bool,
    ) -> Result<(B, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        use cudarc::driver::result;
        let (dim, qlr, nh, hd, rd, win, eps) =
            (cfg.dim, cfg.q_lora_rank, cfg.n_heads, cfg.head_dim, cfg.rope_head_dim, cfg.window_size, cfg.norm_eps);

        // ---- main_kv at start_pos (written BEFORE draft attention) ----
        let (mxc, mxsa) = self.quant_g128(main_x, 1, dim)?;
        let mkv = self.fp8_bsb_rows::<B, CudaSlice<u8>>(&layer.wkv, &mxc, &mxsa, 1)?;
        let mkv = self.rmsnorm(&mkv, &layer.kv_norm, 1, hd, eps)?;
        let (sp_i, one_i, hd_i, rd_i) = (start_pos as i32, 1i32, hd as i32, rd as i32);
        let sp_dev = self.dev.htod_sync_copy(&[start_pos as i32])?;
        dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(32), 1, 1), (256, 1, 1), 0,
            (&mkv, &rope.cos, &rope.sin, &sp_dev, &one_i, &hd_i, &rd_i, &0i32))?;
        let (stride_i, nope_i) = (hd as i32, (hd - rd) as i32);
        dsv4_launch!(self.attn, "dsv4_kv_sim_g64_strided", self.stream.stream,
            (ceil256(((hd - rd) / 64) * 32), 1, 1), (256, 1, 1), 0,
            (&mkv, &one_i, &stride_i, &nope_i))?;
        let win_i = win as i32;
        dsv4_launch!(self.attn, "dsv4_ring_write_b", self.stream.stream, (1u32, 1, 1), (256, 1, 1), 0,
            (&st.kv_cache, &mkv, &one_i, &sp_i, &win_i, &hd_i))?;

        // ---- draft q/kv at positions start_pos+1 .. start_pos+block ----
        let pos0 = start_pos + 1;
        let (xc, xsa) = self.quant_g128::<B, CudaSlice<u8>>(x, block, dim)?;
        let qr_pre = self.fp8_bsb_rows(&layer.wq_a, &xc, &xsa, block)?;
        let qr = self.rmsnorm(&qr_pre, &layer.q_norm, block, qlr, eps)?;
        let (qrc, qrsa) = self.quant_g128::<B, CudaSlice<u8>>(&qr, block, qlr)?;
        let q = self.fp8_bsb_rows(&layer.wq_b, &qrc, &qrsa, block)?;
        let q_pre_host: Vec<f32> = if capture {
            self.dev.dtoh_sync_copy(&q)?.iter().map(|v| v.to_f32()).collect()
        } else { Vec::new() };
        let rows_q = block * nh;
        let (rows_i, hd_i2) = (rows_q as i32, hd as i32);
        // §B.1.1 per-head rescale: ONE BLOCK PER ROW (kernel maps blockIdx.x -> row; the old
        // ceil256 grid rescaled only rows 0..1 — every head-row >= 2 went UNRESCALED, which
        // was the DSpark draft-quality bug: draft q systematically miscalibrated past the
        // first two head rows, acceptance collapsing at d3).
        dsv4_launch!(self.attn, "dsv4_attn_rescale_b", self.stream.stream, (rows_q as u32, 1, 1), (256, 1, 1), 0,
            (&q, &rows_i, &hd_i2, &eps))?;
        let q_rsc_host: Vec<f32> = if capture {
            self.dev.dtoh_sync_copy(&q)?.iter().map(|v| v.to_f32()).collect()
        } else { Vec::new() };
        let pos_q_dev = dsv4_gpu::iota_positions::<CudaSlice<i32>>(&self.dev, &self.spine, &self.stream, pos0 as i32, 1, nh as i32, rows_q)?;
        dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(rows_q * 32), 1, 1), (256, 1, 1), 0,
            (&q, &rope.cos, &rope.sin, &pos_q_dev, &rows_i, &hd_i2, &rd_i, &0i32))?;
        let q_host: Vec<f32> = if capture {
            self.dev.dtoh_sync_copy(&q)?.iter().map(|v| v.to_f32()).collect()
        } else { Vec::new() };
        let kv = self.fp8_bsb_rows(&layer.wkv, &xc, &xsa, block)?;
        let kv = self.rmsnorm(&kv, &layer.kv_norm, block, hd, eps)?;
        let pos_kv_dev = dsv4_gpu::iota_positions::<CudaSlice<i32>>(&self.dev, &self.spine, &self.stream, pos0 as i32, 1, 1, block)?;
        let blk_i = block as i32;
        dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(block * 32), 1, 1), (256, 1, 1), 0,
            (&kv, &rope.cos, &rope.sin, &pos_kv_dev, &blk_i, &hd_i, &rd_i, &0i32))?;
        dsv4_launch!(self.attn, "dsv4_kv_sim_g64_strided", self.stream.stream,
            (ceil256(block * ((hd - rd) / 64) * 32), 1, 1), (256, 1, 1), 0,
            (&kv, &blk_i, &stride_i, &nope_i))?;
        let kv_host: Vec<f32> = if capture {
            self.dev.dtoh_sync_copy(&kv)?.iter().map(|v| v.to_f32()).collect()
        } else { Vec::new() };

        // ---- kv_cat = [ring(128) ++ draft_kv(block)] ----
        let total_kv = win + block;
        let mut kv_cat = self.dev.alloc_zeros::<bf16>(total_kv * hd)?;
        unsafe {
            result::memcpy_dtod_async(*kv_cat.device_ptr(), *st.kv_cache.device_ptr(), win * hd * 2, self.stream.stream)
                .map_err(|e| anyhow!("dspark ring→kv_cat dtod: {e}"))?;
            result::memcpy_dtod_async(
                (*kv_cat.device_ptr()) + (win * hd * 2) as u64,
                *kv.device_ptr(), block * hd * 2, self.stream.stream,
            ).map_err(|e| anyhow!("dspark draft_kv→kv_cat dtod: {e}"))?;
        }

        // ---- 133-entry non-causal index list [0..min(128,start_pos+1) ++ 128..128+block] × block ----
        // R4/drafter-graph prep: device-side (was a host Vec build + htod = a full sync per
        // draft step). Row-uniform integer math — bit-exact vs the host build.
        let t_win = win.min(start_pos + 1);
        let t = t_win + block;
        let idxs_dev = self.dev.alloc_zeros::<i32>(block * t)?;
        let (blk_i2, t_win_i, t_i0, win_i2) = (block as i32, t_win as i32, t as i32, win as i32);
        dsv4_launch!(self.attn, "dsv4_dspark_draft_idxs_b", self.stream.stream,
            (ceil256(block * t), 1, 1), (256, 1, 1), 0,
            (&idxs_dev, &blk_i2, &t_win_i, &t_i0, &win_i2))?;

        // ---- batched gather over block rows × t keys (non-causal) ----
        let scale = (hd as f64).powf(-0.5) as f32;
        let s_pad = ((block + 15) / 16) * 16;
        let rows_q_pad = s_pad * nh;
        let mut o = self.dev.alloc_zeros::<bf16>(rows_q_pad * hd)?;
        let nh16 = (nh / 16) as u32;
        let (t_i, n_i) = (t as i32, total_kv as i32);
        dsv4_launch!(self.spine, "dsv4_gather_attn", self.stream.stream,
            (block as u32, 1u32, nh16), (256, 1, 1), GATHER_SMEM,
            (&q, &kv_cat, &o, &layer.sink, &idxs_dev, &t_i, &n_i, &scale))?;

        // ---- de-rotation with the DRAFT freqs (inverse RoPE) ----
        dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(rows_q * 32), 1, 1), (256, 1, 1), 0,
            (&o, &rope.cos, &rope.sin, &pos_q_dev, &rows_i, &hd_i2, &rd_i, &1i32))?;
        let o_host: Vec<f32> = if capture {
            self.dev.dtoh_sync_copy(&o)?.iter().map(|v| v.to_f32()).collect()
        } else { Vec::new() };

        // ---- grouped-LoRA O projection (wo_a einsum + wo_b fp8_bsb) — same as trunk ----
        let (g, r, gd) = (cfg.o_groups, cfg.o_lora_rank, nh * hd / cfg.o_groups);
        let mut oflat = self.dev.alloc_zeros::<bf16>(s_pad * g * r)?;
        let (blk_i2, g_i, r_i, gd_i, ors_i) = (block as i32, g as i32, r as i32, gd as i32, (nh * hd) as i32);
        if !exact_gemm_enabled() {
            let wq = layer.wo_a_q.as_ref().expect("wo_a fp8 quant must be loaded");
            olo_einsum_fast::<B, CudaSlice<u8>>(self, &mut oflat, &o, wq, block, g, r, gd)?;
        } else {
            let tiles_n = (r + 15) / 16;
            dsv4_launch!(self.attn, "dsv4_olo_proj_tc_b", self.stream.stream,
                ((tiles_n * g) as u32, s_pad as u32 / 16, 1), (32, 1, 1), 0,
                (&oflat, &o, &layer.wo_a, &blk_i2, &g_i, &r_i, &gd_i, &ors_i))?;
        }
        let oflat_host: Vec<f32> = if capture {
            self.dev.dtoh_sync_copy(&oflat)?.iter().map(|v| v.to_f32()).collect()
        } else { Vec::new() };
        let (oc, osa) = self.quant_g128::<B, CudaSlice<u8>>(&oflat, block, g * r)?;
        let attn_out = self.fp8_bsb_rows(&layer.wo_b, &oc, &osa, block)?;
        Ok((attn_out, q_pre_host, q_rsc_host, kv_host, o_host, oflat_host))
    }

    /// CUDA-graph (--dspark-graph) variant of [`dspark_attn_forward`](Self::dspark_attn_forward):
    /// the SAME kernel sequence and argument values, but capture-legal — every transient is
    /// a `Dsv4Buf` (GSlice bump slices of the workspace slab under capture) and the three
    /// position vectors (main_kv sp, draft q positions, draft kv positions) arrive as
    /// PERSISTENT device buffers refreshed outside the graph per step (the eager impl
    /// htod's/iota-allocs them per call — illegal inside capture). No debug captures.
    /// Bitwise-identical to the eager impl by construction (same kernels, same order).
    #[allow(clippy::too_many_arguments)]
    pub fn dspark_attn_forward_dev<X: dsv4_gpu::Dsv4Buf<bf16>, C: dsv4_gpu::Dsv4Buf<u8>, I: dsv4_gpu::Dsv4Buf<i32>>(
        &self,
        layer: &Dsv4GpuLayer,
        st: &mut Dsv4AttnState,
        x: &X,
        block: usize,
        start_pos: usize,
        main_x: &X,
        rope: &DevRope,
        cfg: &dsv4_load::Dsv4Config,
        pos_sp: &CudaSlice<i32>,
        pos_q: &CudaSlice<i32>,
        pos_kv: &CudaSlice<i32>,
    ) -> Result<X> {
        use cudarc::driver::result;
        let (dim, qlr, nh, hd, rd, win, eps) =
            (cfg.dim, cfg.q_lora_rank, cfg.n_heads, cfg.head_dim, cfg.rope_head_dim, cfg.window_size, cfg.norm_eps);

        // ---- main_kv at start_pos (written BEFORE draft attention) ----
        let (mxc, mxsa) = self.quant_g128::<X, C>(main_x, 1, dim)?;
        let mkv = self.fp8_bsb_rows::<X, C>(&layer.wkv, &mxc, &mxsa, 1)?;
        let mkv = self.rmsnorm(&mkv, &layer.kv_norm, 1, hd, eps)?;
        let (sp_i, one_i, hd_i, rd_i) = (start_pos as i32, 1i32, hd as i32, rd as i32);
        dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(32), 1, 1), (256, 1, 1), 0,
            (&mkv, &rope.cos, &rope.sin, pos_sp, &one_i, &hd_i, &rd_i, &0i32))?;
        let (stride_i, nope_i) = (hd as i32, (hd - rd) as i32);
        dsv4_launch!(self.attn, "dsv4_kv_sim_g64_strided", self.stream.stream,
            (ceil256(((hd - rd) / 64) * 32), 1, 1), (256, 1, 1), 0,
            (&mkv, &one_i, &stride_i, &nope_i))?;
        let win_i = win as i32;
        dsv4_launch!(self.attn, "dsv4_ring_write_b", self.stream.stream, (1u32, 1, 1), (256, 1, 1), 0,
            (&st.kv_cache, &mkv, &one_i, &sp_i, &win_i, &hd_i))?;

        // ---- draft q/kv at positions start_pos+1 .. start_pos+block ----
        let (xc, xsa) = self.quant_g128::<X, C>(x, block, dim)?;
        let qr_pre = self.fp8_bsb_rows::<X, C>(&layer.wq_a, &xc, &xsa, block)?;
        let qr = self.rmsnorm(&qr_pre, &layer.q_norm, block, qlr, eps)?;
        let (qrc, qrsa) = self.quant_g128::<X, C>(&qr, block, qlr)?;
        let q = self.fp8_bsb_rows::<X, C>(&layer.wq_b, &qrc, &qrsa, block)?;
        let rows_q = block * nh;
        let (rows_i, hd_i2) = (rows_q as i32, hd as i32);
        // ONE BLOCK PER ROW (the acceptance-cliff fix — see dspark_attn_forward_impl).
        dsv4_launch!(self.attn, "dsv4_attn_rescale_b", self.stream.stream, (rows_q as u32, 1, 1), (256, 1, 1), 0,
            (&q, &rows_i, &hd_i2, &eps))?;
        dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(rows_q * 32), 1, 1), (256, 1, 1), 0,
            (&q, &rope.cos, &rope.sin, pos_q, &rows_i, &hd_i2, &rd_i, &0i32))?;
        let kv = self.fp8_bsb_rows::<X, C>(&layer.wkv, &xc, &xsa, block)?;
        let kv = self.rmsnorm(&kv, &layer.kv_norm, block, hd, eps)?;
        let blk_i = block as i32;
        dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(block * 32), 1, 1), (256, 1, 1), 0,
            (&kv, &rope.cos, &rope.sin, pos_kv, &blk_i, &hd_i, &rd_i, &0i32))?;
        dsv4_launch!(self.attn, "dsv4_kv_sim_g64_strided", self.stream.stream,
            (ceil256(block * ((hd - rd) / 64) * 32), 1, 1), (256, 1, 1), 0,
            (&kv, &blk_i, &stride_i, &nope_i))?;

        // ---- kv_cat = [ring(128) ++ draft_kv(block)] ----
        let total_kv = win + block;
        let kv_cat = X::alloc_zeros(&self.dev, self.stream.stream, total_kv * hd)?;
        unsafe {
            result::memcpy_dtod_async(kv_cat.dptr(), *st.kv_cache.device_ptr(), win * hd * 2, self.stream.stream)
                .map_err(|e| anyhow!("dspark-dev ring→kv_cat dtod: {e}"))?;
            result::memcpy_dtod_async(
                kv_cat.dptr() + (win * hd * 2) as u64,
                kv.dptr(), block * hd * 2, self.stream.stream,
            ).map_err(|e| anyhow!("dspark-dev draft_kv→kv_cat dtod: {e}"))?;
        }

        // ---- non-causal index list [0..min(128,sp+1) ++ 128..128+block] × block (device-side) ----
        let t_win = win.min(start_pos + 1);
        let t = t_win + block;
        let idxs_dev = I::alloc_zeros(&self.dev, self.stream.stream, block * t)?;
        let (blk_i2, t_win_i, t_i0, win_i2) = (block as i32, t_win as i32, t as i32, win as i32);
        dsv4_launch!(self.attn, "dsv4_dspark_draft_idxs_b", self.stream.stream,
            (ceil256(block * t), 1, 1), (256, 1, 1), 0,
            (&idxs_dev, &blk_i2, &t_win_i, &t_i0, &win_i2))?;

        // ---- batched gather over block rows × t keys (non-causal) ----
        let scale = (hd as f64).powf(-0.5) as f32;
        let s_pad = ((block + 15) / 16) * 16;
        let rows_q_pad = s_pad * nh;
        let o = X::alloc_zeros(&self.dev, self.stream.stream, rows_q_pad * hd)?;
        let nh16 = (nh / 16) as u32;
        let (t_i, n_i) = (t as i32, total_kv as i32);
        dsv4_launch!(self.spine, "dsv4_gather_attn", self.stream.stream,
            (block as u32, 1u32, nh16), (256, 1, 1), GATHER_SMEM,
            (&q, &kv_cat, &o, &layer.sink, &idxs_dev, &t_i, &n_i, &scale))?;

        // ---- de-rotation with the DRAFT freqs (inverse RoPE) ----
        dsv4_launch!(self.spine, "dsv4_rope_last_b", self.stream.stream, (ceil256(rows_q * 32), 1, 1), (256, 1, 1), 0,
            (&o, &rope.cos, &rope.sin, pos_q, &rows_i, &hd_i2, &rd_i, &1i32))?;

        // ---- grouped-LoRA O projection (wo_a einsum + wo_b fp8_bsb) — same as trunk ----
        let (g, r, gd) = (cfg.o_groups, cfg.o_lora_rank, nh * hd / cfg.o_groups);
        let mut oflat = X::alloc_zeros(&self.dev, self.stream.stream, s_pad * g * r)?;
        let (blk_i2, g_i, r_i, gd_i, ors_i) = (block as i32, g as i32, r as i32, gd as i32, (nh * hd) as i32);
        if !exact_gemm_enabled() {
            let wq = layer.wo_a_q.as_ref().expect("wo_a fp8 quant must be loaded");
            olo_einsum_fast::<X, C>(self, &mut oflat, &o, wq, block, g, r, gd)?;
        } else {
            let tiles_n = (r + 15) / 16;
            dsv4_launch!(self.attn, "dsv4_olo_proj_tc_b", self.stream.stream,
                ((tiles_n * g) as u32, s_pad as u32 / 16, 1), (32, 1, 1), 0,
                (&oflat, &o, &layer.wo_a, &blk_i2, &g_i, &r_i, &gd_i, &ors_i))?;
        }
        let (oc, osa) = self.quant_g128::<X, C>(&oflat, block, g * r)?;
        self.fp8_bsb_rows::<X, C>(&layer.wo_b, &oc, &osa, block)
    }

    /// Router on the UN-simmed x (fp32 purity, §9.4/§12.A.5). Returns (sel [s,topk] global expert
    /// ids, router_w [s,topk] fp32). Replicated+identical on both TP ranks (gate_w/bias/tid2eid are
    /// replicated; the input x is identical post-attention) — so the routed-expert SELECTION is the
    /// same on every rank; only the routed-expert COMPUTATION differs (each rank's band).
    /// R3.1: scores/biased ride the persistent workspace (sel/router_w stay per-call — they are
    /// returned across dispatch boundaries, e.g. block_forward_tp_sim shares them across ranks).
    pub fn moe_router<X: dsv4_gpu::Dsv4Buf<bf16>, F: dsv4_gpu::Dsv4Buf<f32>, I: dsv4_gpu::Dsv4Buf<i32>>(
        &self,
        layer: &Dsv4GpuLayer,
        x: &X,
        s: usize,
        ids: &CudaSlice<i32>,
        cfg: &dsv4_load::Dsv4Config,
        scratch: &mut MoeGroupedScratch,
    ) -> Result<(I, F)> {
        let (dim, ne, topk) = (cfg.dim, cfg.n_routed_experts, cfg.n_activated_experts);
        let s_i = s as i32;
        let ws = scratch.dsv4_ws_mut();
        anyhow::ensure!(s <= ws.s_cap, "moe_router: s {s} > workspace s_cap {}", ws.s_cap);
        let scores = &mut ws.scores;
        let (dim_i, ne_i, tk_i) = (dim as i32, ne as i32, topk as i32);
        dsv4_launch!(self.spine, "dsv4_router_score_b", self.stream.stream, (ne as u32, s as u32, 1), (256, 1, 1), 0,
            (scores, &layer.gate_w, x, &s_i, &dim_i, &ne_i))?;
        let sel = I::alloc_zeros(&self.dev, self.stream.stream, s * topk)?;
        match (&layer.tid2eid, &layer.gate_bias) {
            (Some(tbl), _) => {
                dsv4_launch!(self.spine, "dsv4_router_tid2eid_b", self.stream.stream, (ceil256(s * topk), 1, 1), (256, 1, 1), 0,
                    (&sel, tbl, ids, &s_i, &tk_i))?;
            }
            (None, Some(bias)) => {
                let biased = &mut ws.biased;
                dsv4_launch!(self.spine, "dsv4_router_bias_add_b", self.stream.stream, (ceil256(s * ne), 1, 1), (256, 1, 1), 0,
                    (biased, &*scores, bias, &s_i, &ne_i))?;
                dsv4_launch!(self.spine, "dsv4_topk", self.stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
                    (&*biased, &sel, &s_i, &ne_i, &tk_i))?;
            }
            (None, None) => {
                dsv4_launch!(self.spine, "dsv4_topk", self.stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
                    (&*scores, &sel, &s_i, &ne_i, &tk_i))?;
            }
        }
        let router_w = F::alloc_zeros(&self.dev, self.stream.stream, s * topk)?;
        let rs = cfg.route_scale;
        dsv4_launch!(self.spine, "dsv4_router_weights_b", self.stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
            (&router_w, &*scores, &sel, &s_i, &ne_i, &tk_i, &rs))?;
        Ok((sel, router_w))
    }

    /// Shared expert (replicated) on the pristine x (§B.9: shared input is the original bf16 x)
    /// into `ws.sh_out`. Identical on both TP ranks — it is added AFTER the routed all-reduce
    /// (reducing after the add would double it; see `gpu.rs:2577` for the proven Hy3 reference).
    /// R3.1: all intermediates ride the workspace (zero per-call allocs on the serving path).
    pub fn moe_shared_into<X: dsv4_gpu::Dsv4Buf<bf16>>(&self, layer: &Dsv4GpuLayer, x: &X, s: usize, cfg: &dsv4_load::Dsv4Config, ws: &mut crate::gpu::Dsv4MoeWorkspace) -> Result<()> {
        let (dim, inter) = (cfg.dim, cfg.moe_inter_dim);
        anyhow::ensure!(s <= ws.s_cap, "moe_shared_into: s {s} > workspace s_cap {}", ws.s_cap);
        self.quant_g128_into(&mut ws.sh_q, &mut ws.sh_qs, x, s, dim)?;
        self.fp8_bsb_rows_into(&mut ws.sh_gu, &layer.sh_gu, &ws.sh_q, &ws.sh_qs, s)?; // [s, 2*inter] gate|up
        let (inter_i, limit, s_i) = (inter as i32, cfg.swiglu_limit, s as i32);
        dsv4_launch!(self.spine, "dsv4_swiglu_clamp_shared", self.stream.stream, (ceil256(s * inter), 1, 1), (256, 1, 1), 0,
            (&mut ws.sh_h, &ws.sh_gu, &limit, &inter_i, &s_i))?;
        self.quant_g128_into(&mut ws.sh_q2, &mut ws.sh_qs2, &ws.sh_h, s, inter)?;
        self.fp8_bsb_rows_into(&mut ws.sh_out, &layer.sh_w2, &ws.sh_q2, &ws.sh_qs2, s)?; // [s, dim]
        Ok(())
    }

    /// Shared expert (replicated), allocating variant — probes/one-shots. Production uses
    /// [`moe_shared_into`](Self::moe_shared_into) on the workspace.
    pub fn moe_shared(&self, layer: &Dsv4GpuLayer, x: &B, s: usize, cfg: &dsv4_load::Dsv4Config) -> Result<B> {
        let (dim, inter) = (cfg.dim, cfg.moe_inter_dim);
        let (shc, shsa) = self.quant_g128::<B, CudaSlice<u8>>(x, s, dim)?;
        let gu = self.fp8_bsb_rows::<B, CudaSlice<u8>>(&layer.sh_gu, &shc, &shsa, s)?; // [s, 2*inter] gate|up
        let h = self.dev.alloc_zeros::<bf16>(s * inter)?;
        let (inter_i, limit, s_i) = (inter as i32, cfg.swiglu_limit, s as i32);
        dsv4_launch!(self.spine, "dsv4_swiglu_clamp_shared", self.stream.stream, (ceil256(s * inter), 1, 1), (256, 1, 1), 0,
            (&h, &gu, &limit, &inter_i, &s_i))?;
        let (hc, hsa) = self.quant_g128::<B, CudaSlice<u8>>(&h, s, inter)?;
        Ok(self.fp8_bsb_rows(&layer.sh_w2, &hc, &hsa, s)?) // [s, dim]
    }

    /// Routed experts (the TP=2 rank-LOCAL partial) into the workspace's `routed_out`. `sel`/
    /// `router_w` from [`moe_router`]; the runtime sims its input in place → staged chunk copies
    /// (x stays pristine). Under TP=2 each rank's `Dsv4MoeGpu` holds only its `[e_base,
    /// e_base+e_span)` band, so remote (token,slot) pairs contribute exact zeros — this output is
    /// a PARTIAL summed by the all-reduce. R3.1: chunk staging + the expert helpers run entirely
    /// on the compute stream against the persistent workspace — NO host syncs, NO per-call allocs
    /// (the old path did 3 `synchronize()` + ~11 allocs + an htod per dispatch per layer).
    pub fn moe_routed<X: dsv4_gpu::Dsv4Buf<bf16>, F: dsv4_gpu::Dsv4Buf<f32>, I: dsv4_gpu::Dsv4Buf<i32>>(
        &self,
        layer: &Dsv4GpuLayer,
        scratch: &mut MoeGroupedScratch,
        x: &X,
        s: usize,
        sel: &I,
        router_w: &F,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<()> {
        let (dim, topk) = (cfg.dim, cfg.n_activated_experts);
        anyhow::ensure!(s <= scratch.dsv4_ws_ref().s_cap, "moe_routed: s {s} > workspace s_cap");
        // R3A.4: full-width dispatches of up to d_cap rows instead of 128 sequential 16-row
        // dispatches — each touched expert's weights are read ceil(rows_e/16) times per
        // dispatch instead of once per 16-row dispatch (~21x less expert-weight traffic; the
        // grouped GEMM's per-element K-order is ppad-independent, so outputs are
        // bitwise-identical to the 16-row decomposition — gated). s > d_cap (one-shot
        // probes, e.g. the 32k §12.B.5 reference) loops d_cap chunks of the same form.
        let d_cap = gpu::MOE_D_CAP.min(scratch.p_max() / topk);
        anyhow::ensure!(d_cap >= 1, "moe_routed: scratch p_max {} < topk", scratch.p_max());
        let mut r0 = 0usize;
        while r0 < s {
            let n = (s - r0).min(d_cap);
            {
                let ws = scratch.dsv4_ws_mut();
                unsafe {
                    result::memcpy_dtod_async(*ws.xc.device_ptr(), x.dptr() + (r0 * dim * 2) as u64, n * dim * 2, self.stream.stream)
                        .map_err(|e| anyhow!("d2d xc stage: {e}"))?;
                    result::memcpy_dtod_async(*ws.idc.device_ptr(), sel.dptr() + (r0 * topk * 4) as u64, n * topk * 4, self.stream.stream)
                        .map_err(|e| anyhow!("d2d idc stage: {e}"))?;
                    result::memcpy_dtod_async(*ws.wtc.device_ptr(), router_w.dptr() + (r0 * topk * 4) as u64, n * topk * 4, self.stream.stream)
                        .map_err(|e| anyhow!("d2d wtc stage: {e}"))?;
                }
            }
            if n == 1 {
                gpu::dsv4_moe_experts_n1_ws(&self.dev, &self.stream, &self.bk, &self.df, &layer.moe,
                    scratch.dsv4_ws_mut(), 1, topk, cfg.swiglu_limit)?;
            } else {
                gpu::dsv4_moe_experts_grouped_ws(&self.dev, &self.stream, &self.bk, &self.df, &layer.moe,
                    scratch, n, topk, cfg.swiglu_limit)?;
            }
            {
                let ws = scratch.dsv4_ws_mut();
                unsafe {
                    result::memcpy_dtod_async(*ws.routed_out.device_ptr() + (r0 * dim * 2) as u64, *ws.outc.device_ptr(), n * dim * 2, self.stream.stream)
                        .map_err(|e| anyhow!("d2d out fold: {e}"))?;
                }
            }
            r0 += n;
        }
        Ok(())
    }

    /// `ffn_out = bf16(fp32(routed) + fp32(shared))` — the post-all-reduce combine (one bf16 RNE).
    pub fn ffn_combine<X: dsv4_gpu::Dsv4Buf<bf16>>(&self, routed: &B, shared: &B, s: usize, dim: usize) -> Result<X> {
        let ffn_out = X::alloc_zeros(&self.dev, self.stream.stream, s * dim)?;
        let total_i = (s * dim) as i32;
        let cfg_launch = LaunchConfig { grid_dim: (ceil256(s * dim), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let out_v = ffn_out.view(0, s * dim);
        unsafe {
            self.bk["add_residual_b"]
                .clone()
                .launch_on_stream(&self.stream, cfg_launch, (&out_v, routed, shared, total_i))
                .map_err(|e| anyhow!("add_residual_b: {e:?}"))?;
        }
        Ok(ffn_out)
    }

    /// FFN sublayer (§B.9): router (on the un-simmed x) → shared expert + routed experts
    /// (G2 runtime, ≤16-row chunks) → residual add. `x` [s, dim] bf16 is never mutated
    /// (routed chunks sim private copies). Returns (ffn_out, router_w [s,6], router_idx [s,6]).
    /// Single-process path (no TP). Under TP=2 use [`moe_router`]+[`moe_routed`]→all-reduce→
    /// [`moe_shared`]+[`ffn_combine`] (see `block_forward_tp`).
    pub fn moe_forward<X: dsv4_gpu::Dsv4Buf<bf16>, F: dsv4_gpu::Dsv4Buf<f32>, I: dsv4_gpu::Dsv4Buf<i32>>(
        &self,
        layer: &Dsv4GpuLayer,
        scratch: &mut MoeGroupedScratch,
        x: &X,
        s: usize,
        ids: &CudaSlice<i32>,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<(X, F, I)> {
        let (sel, router_w) = self.moe_router(layer, x, s, ids, cfg, scratch)?;
        self.moe_shared_into(layer, x, s, cfg, scratch.dsv4_ws_mut())?;
        self.moe_routed(layer, scratch, x, s, &sel, &router_w, cfg)?;
        let ffn_out = {
            let ws = scratch.dsv4_ws_ref();
            self.ffn_combine(&ws.routed_out, &ws.sh_out, s, cfg.dim)?
        };
        Ok((ffn_out, router_w, sel))
    }

    /// Single-node TP=2 simulation block forward (the model-side TP validator, no transport).
    /// Mirrors [`block_forward`](Self::block_forward) but splits the FFN at the routed|shared
    /// boundary: rank_a's attention runs (advancing `state_a` — attention is replicated, so rank_a
    /// drives it; rank_b's state is untouched), then BOTH ranks' routed-expert partials are computed
    /// on the identical `y2n` and SUMMED (the doorbell all-reduce, simulated here as a bf16 add),
    /// then the replicated shared expert is added, then `hc_post`. Returns the TP block output `x3`.
    /// `layer_b`/`scratch_b` are rank_b's; only rank_b's MoE band is read (its attn/mHC/shared are
    /// identical to rank_a's and unused). Compare the result to a full-256 forward to validate the
    /// §5 sharding + the all-reduce-before-shared boundary compounding across layers.
    #[allow(clippy::too_many_arguments)]
    pub fn block_forward_tp_sim(
        &self,
        layer_a: &Dsv4GpuLayer,
        state_a: &mut Dsv4AttnState,
        scratch_a: &mut MoeGroupedScratch,
        layer_b: &Dsv4GpuLayer,
        scratch_b: &mut MoeGroupedScratch,
        x: &B,
        s: usize,
        start_pos: usize,
        ids: &CudaSlice<i32>,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<B> {
        // --- attention sublayer (rank_a; replicated, so rank_a drives it) ---
        let (y, posts, combs) = self.hc_pre::<B, S>(x, s, &layer_a.hc_attn_fn, &layer_a.hc_attn_base, &layer_a.hc_attn_scale, cfg)?;
        let yn = self.rmsnorm(&y, &layer_a.attn_norm, s, cfg.dim, cfg.norm_eps)?;
        let (attn_out, _topk_idx, _topk_t) = self.attn_forward::<B, S, CudaSlice<i32>, CudaSlice<u8>, CudaSlice<u32>>(layer_a, state_a, &yn, s, start_pos, cfg)?;
        let x2 = self.hc_post(&attn_out, x, &posts, &combs, s, cfg)?;
        // --- ffn sublayer: router (replicated) → routed partials (rank-local) → ALL-REDUCE → +shared → hc_post ---
        let (y2, posts2, combs2) = self.hc_pre::<B, S>(&x2, s, &layer_a.hc_ffn_fn, &layer_a.hc_ffn_base, &layer_a.hc_ffn_scale, cfg)?;
        let y2n = self.rmsnorm(&y2, &layer_a.ffn_norm, s, cfg.dim, cfg.norm_eps)?;
        let (sel, router_w) = self.moe_router::<B, S, CudaSlice<i32>>(layer_a, &y2n, s, ids, cfg, scratch_a)?;
        self.moe_routed(layer_a, scratch_a, &y2n, s, &sel, &router_w, cfg)?;
        self.moe_routed(layer_b, scratch_b, &y2n, s, &sel, &router_w, cfg)?;
        // ALL-REDUCE (simulated): routed_sum = bf16(routed_a + routed_b). The doorbell's tp_wait_add
        // does this same fp32-add-then-bf16-round in-kernel; here it's add_residual_b on the host stream.
        let routed_sum = self.ffn_combine(&scratch_a.dsv4_ws_ref().routed_out, &scratch_b.dsv4_ws_ref().routed_out, s, cfg.dim)?;
        self.moe_shared_into(layer_a, &y2n, s, cfg, scratch_a.dsv4_ws_mut())?;
        let ffn_out = self.ffn_combine(&routed_sum, &scratch_a.dsv4_ws_ref().sh_out, s, cfg.dim)?;
        let x3 = self.hc_post(&ffn_out, &x2, &posts2, &combs2, s, cfg)?;
        Ok(x3)
    }

    /// Full trunk `Block.forward` (§B.8 ordering): hc_pre → norm → sublayer → hc_post, twice.
    /// `x` [s, hc*dim] bf16 streams; `ids` [s] i32 (hash-router table lookups).
    pub fn block_forward<X: dsv4_gpu::Dsv4Buf<bf16>, F: dsv4_gpu::Dsv4Buf<f32>, I: dsv4_gpu::Dsv4Buf<i32>, C: dsv4_gpu::Dsv4Buf<u8>, U: dsv4_gpu::Dsv4Buf<u32>>(
        &self,
        layer: &Dsv4GpuLayer,
        st: &mut Dsv4AttnState,
        scratch: &mut MoeGroupedScratch,
        x: &X,
        s: usize,
        start_pos: usize,
        ids: &CudaSlice<i32>,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<BlockOut<X, F, I>> {
        // --- attention sublayer (replicated under TP — no cross-rank reduction) ---
        let (y, posts, combs) = self.hc_pre::<X, F>(x, s, &layer.hc_attn_fn, &layer.hc_attn_base, &layer.hc_attn_scale, cfg)?;
        let yn = self.rmsnorm(&y, &layer.attn_norm, s, cfg.dim, cfg.norm_eps)?;
        let (attn_out, topk_idx, topk_t) = self.attn_forward::<X, F, I, C, U>(layer, st, &yn, s, start_pos, cfg)?;
        let x2 = self.hc_post(&attn_out, x, &posts, &combs, s, cfg)?;
        // --- ffn sublayer ---
        let (y2, posts2, combs2) = self.hc_pre::<X, F>(&x2, s, &layer.hc_ffn_fn, &layer.hc_ffn_base, &layer.hc_ffn_scale, cfg)?;
        let y2n = self.rmsnorm(&y2, &layer.ffn_norm, s, cfg.dim, cfg.norm_eps)?;
        let (ffn_out, router_w, router_idx) = if self.tp_ctx_dptr != 0 {
            // TP=2: router (replicated) → routed PARTIAL (rank-local) → ALL-REDUCE → +shared (replicated).
            // Boundary = routed combine BEFORE the shared add (shared replicated; reducing after
            // double-counts it — the proven Hy3 pattern, gpu.rs:2577).
            let (sel, rw) = self.moe_router(layer, &y2n, s, ids, cfg, scratch)?;
            self.moe_routed(layer, scratch, &y2n, s, &sel, &rw, cfg)?;
            self.tp_all_reduce_bf16(&mut scratch.dsv4_ws_mut().routed_out, s * cfg.dim)?;
            self.moe_shared_into(layer, &y2n, s, cfg, scratch.dsv4_ws_mut())?;
            let ffn = {
                let ws = scratch.dsv4_ws_ref();
                self.ffn_combine(&ws.routed_out, &ws.sh_out, s, cfg.dim)?
            };
            (ffn, rw, sel)
        } else {
            self.moe_forward(layer, scratch, &y2n, s, ids, cfg)?
        };
        let x3 = self.hc_post(&ffn_out, &x2, &posts2, &combs2, s, cfg)?;
        Ok(BlockOut {
            y: x3,
            attn_out,
            ffn_out,
            router_w,
            router_idx,
            topk_idx,
            topk_t,
        })
    }

    /// Traced `Block.forward` for the batch-invariance localization probe: identical math to
    /// [`block_forward`](Self::block_forward), but also returns the POST-ATTENTION mid-block
    /// streams (`x2` after the attention hc_post) so the caller can tell whether a row-0
    /// divergence enters at the attention sublayer or the MoE/ffn sublayer. Returns
    /// `(mid_post_attn, final_post_ffn)`, both `[s, hc*dim]` bf16.
    pub fn block_forward_traced(
        &self,
        layer: &Dsv4GpuLayer,
        st: &mut Dsv4AttnState,
        scratch: &mut MoeGroupedScratch,
        x: &B,
        s: usize,
        start_pos: usize,
        ids: &CudaSlice<i32>,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<(B, B)> {
        let (y, posts, combs) = self.hc_pre::<B, S>(x, s, &layer.hc_attn_fn, &layer.hc_attn_base, &layer.hc_attn_scale, cfg)?;
        let yn = self.rmsnorm(&y, &layer.attn_norm, s, cfg.dim, cfg.norm_eps)?;
        let (attn_out, _topk_idx, _topk_t) = self.attn_forward::<B, S, CudaSlice<i32>, CudaSlice<u8>, CudaSlice<u32>>(layer, st, &yn, s, start_pos, cfg)?;
        let mid = self.hc_post(&attn_out, x, &posts, &combs, s, cfg)?;
        let (y2, posts2, combs2) = self.hc_pre::<B, S>(&mid, s, &layer.hc_ffn_fn, &layer.hc_ffn_base, &layer.hc_ffn_scale, cfg)?;
        let y2n = self.rmsnorm(&y2, &layer.ffn_norm, s, cfg.dim, cfg.norm_eps)?;
        let (ffn_out, _router_w, _router_idx) = self.moe_forward::<B, S, CudaSlice<i32>>(layer, scratch, &y2n, s, ids, cfg)?;
        let out = self.hc_post(&ffn_out, &mid, &posts2, &combs2, s, cfg)?;
        Ok((mid, out))
    }
}
