//! Full-GPU forward for Qwen3.5-0.8B (f32). cuBLAS gemv for linears + custom CUDA kernels
//! for elementwise/recurrent ops. Validated against the same reference as qwen.rs.
//!
//! File map (this is the former single `gpu.rs`; every item keeps its `crate::gpu::` path, moved public items are
//! re-exported below). `GpuModel` itself, the weight/layer types, constants and the launch macros live here; the
//! `impl GpuModel` methods are cut by topic into several files (each file is another `impl GpuModel` block):
//! - `construct.rs`, `load.rs`: construction, kernel-module loading; `load_from_dir_impl` (the loader) alone.
//! - `launch_args.rs`, `gemm_policy.rs`, `e9.rs`: kernel-argument helpers, GEMM/shard policy predicates, E9 raw launches.
//! - `shard_ops.rs`, `tp_shard.rs`, `eff.rs`, `tp_comm.rs`: TP shard-at-load ops, shard/attach methods, effective head
//!   counts, TP all-reduce plumbing.
//! - `forward_token.rs`, `forward_batch.rs`, `attn_batch.rs`, `gemm_prefill.rs`, `launch_gemm.rs`, `gates.rs`, `moe_batch.rs`:
//!   the f32 single-token path and the batched forward (GEMM launches, feature gates, attention, MoE).
//! - `prefill.rs`, `decode.rs`, `verify.rs`, `verify_sample.rs`, `graphs.rs`, `embed.rs`, `state.rs`: prefill, decode,
//!   verify (speculative), CUDA graphs, embedding / vision splice, per-slot state and KV management.
//! - `mtp.rs`, `generate.rs`, `df2_capture.rs`: MTP head, TP / DFlash generation loops, DFlash2 / DSpark taps.
//! - `batch_types.rs`, `pool.rs`, `dsv4_moe.rs`, `xchain.rs`: batch-state types, buffer pool + tripwire, DSV4 MoE kernels,
//!   cross-chain capture / dump.
//! - `qwen4_exp.rs`, `qsa.rs`, `gptq_gpu.rs`, `gptq_rot.rs`, `w4a4.rs`, `bandwidth.rs`: qwen4_exp HC/PLE, QSA indexer, GPTQ
//!   taps and rotation, W4A4 prefill + IGS, the bandwidth probe.
//! - `probes_*.rs`, `bench_*.rs`, `debug.rs`: probes, benches and debug dumps.
//! - `geom_pack_tests.rs` ... `release_workers_tests.rs`: the former inline unit-test modules.
//! Macro note: `blaunch!` / `blaunch_e9!` are textually scoped, so they are defined below BEFORE every `mod` line.

use std::sync::Arc;
use std::collections::HashMap;

use cudarc::cublas::{sys::cublasOperation_t as OP, CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaDevice, CudaFunction, DevicePtr, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

use crate::qwen::{Config, LayerType};

/// Launch a batched kernel by name with a typed tuple of DeviceRepr args.
macro_rules! blaunch {
    ($s:expr, $name:expr, $g:expr, $b:expr, $smem:expr, ($($a:expr),+ $(,)?)) => {
        unsafe {
            let (g0,g1,g2) = $g;
            let (b0,b1,b2) = $b;
            // Lazy panic closures, not expect(concat!(..)): the kernel name is a runtime `&str` on the
            // paths that pick a layout-specific variant, and an eager format! would allocate on every
            // launch of the hot path.
            let name: &str = $name;
            $s.bk.get(name).cloned().unwrap_or_else(|| panic!("batch kernel {}", name)).launch_on_stream(
                &$s.stream,
                LaunchConfig { grid_dim: (g0,g1,g2), block_dim: (b0,b1,b2), shared_mem_bytes: $smem },
                ($($a),+)
            ).unwrap_or_else(|e| panic!("batch launch {}: {:?}", name, e));
        }
    };
}

/// E9 launch: same tuple-args convention as `blaunch!`, but on the E9 module's raw handle with the
/// programmatic stream-serialization attribute (the kernel may start its preamble while the previous
/// kernel still runs; it must griddepcontrol.wait before consuming the previous kernel's output —
/// every E9-marked kernel does, in its prologue). Falls back to the plain `blaunch!` when the
/// programmatic path is off (single-node, the --e9-no-fold escape, or a module-load failure) —
/// the launch is then exactly the pre-E9 one.
macro_rules! blaunch_e9 {
    ($s:expr, $name:expr, $g:expr, $b:expr, $smem:expr, ($($a:expr),+ $(,)?)) => {
        if let Some(f) = $s.e9_handle($name) {
            let (g0,g1,g2) = $g;
            let (b0,b1,b2) = $b;
            // The params array MUST be constructed inline in the launch statement: as_kernel_param()
            // yields a pointer INTO its argument expression, and a `let`-bound array would drop those
            // temporaries at the end of the `let` statement — dangling pointees for cuLaunchKernelEx.
            // (cudarc's own LaunchAsync is safe because it takes the args tuple as a named parameter.)
            unsafe { $s.launch_pss(f, (g0,g1,g2), (b0,b1,b2), $smem, &mut [$($a.as_kernel_param()),+]); }
        } else {
            blaunch!($s, $name, $g, $b, $smem, ($($a),+));
        }
    };
}

mod attn_batch;
mod bandwidth;
mod batch_types;
mod bench_mtp;
mod bench_tree;
mod construct;
mod debug;
mod decode;
mod df2_capture;
mod dsv4_moe;
mod e9;
mod eff;
mod embed;
mod forward_batch;
mod forward_token;
mod gates;
mod gemm_policy;
mod gemm_prefill;
mod generate;
mod gptq_gpu;
mod gptq_rot;
mod graphs;
mod launch_args;
mod launch_gemm;
mod load;
mod moe_batch;
mod mtp;
mod pool;
mod prefill;
mod probes_gemm;
mod probes_mxfp4;
mod probes_verify;
mod qsa;
mod qwen4_exp;
mod shard_ops;
mod state;
mod tp_comm;
mod tp_shard;
mod verify;
mod verify_sample;
mod w4a4;
mod xchain;
#[cfg(test)]
mod geom_pack_tests;
#[cfg(test)]
mod release_workers_tests;
#[cfg(test)]
mod shard_factor_tests;
#[cfg(test)]
mod shard_raw_fp8_blk_tests;
#[cfg(test)]
mod shard_raw_real_tests;
#[cfg(test)]
mod shard_raw_tests;
#[cfg(test)]
mod xchain2_tests;

pub use self::bandwidth::*;
pub use self::batch_types::*;
pub(crate) use self::construct::*;
pub use self::dsv4_moe::*;
use self::e9::*;
use self::gemm_policy::*;
pub use self::gptq_gpu::*;
pub use self::graphs::*;
pub use self::launch_args::*;
pub use self::pool::*;
pub use self::qsa::*;
pub(crate) use self::shard_ops::*;
pub use self::xchain::*;

pub type S = cudarc::driver::CudaSlice<f32>;
pub type B = cudarc::driver::CudaSlice<half::bf16>; // big matmul weights in bf16

// -----------------------------------------------------------------------------------------------
// AR landing 2 (EXPERT_GPU_ALLREDUCE §5/§9): fused reduce+residual+norm epilogue plumbing.
// The mixer/FFN epilogue sites run `tp_reduce_resnorm_b` (one kernel: K1-published local partial
// + the peer slot -> the K2 rounding -> the fused_res_rmsnorm_b(/q) body) in place of the
// two-launch K2 + norm chain when --tp-reduce-fuse is on and the reduction is a single barrier.
// The reduce sites stop after K1 and hand the LOCAL partial upward via these types; the layer loop
// launches the fused kernel and releases the carried buffers.
// -----------------------------------------------------------------------------------------------

/// The local partial a K1-only reduction left behind (AR landing 2).
pub(crate) enum FusedPartial {
    /// mode 0 — bf16 local partial: it IS the caller's `out` buffer (no extra allocation).
    Bf16,
    /// mode 1 — fp32 local partial in a pool buffer the caller must release after the fused launch.
    F32(S),
    /// R10 (world>2) mode 1 — rounds-1 preliminary exchanges accumulated into the PERSISTENT fp32
    /// scratch and the final K1 shipped it; tp_reduce_resnorm_b reads the scratch as its fp32
    /// local. No buffer to release (the scratch is engine-owned).
    F32Scratch,
}

/// What a K1-only helper reports back to its caller (ownership of pool buffers stays with the
/// caller, so this is a pure mode marker).
pub(crate) enum FusedLocal {
    /// world==2 bf16 site: the local partial is the caller's bf16 `out` buffer.
    Bf16Buf,
    /// world==2 fp32 site: the local partial is the caller's fp32 pool buffer.
    F32Buf,
    /// world>2 (R10): the fp32 accumulator lives in the persistent scratch (already K1'd).
    F32Scratch,
}

/// What a reduction at an AR-landing-2 epilogue site produced.
pub(crate) struct FusedReduce {
    /// The bf16 buffer the reduce site was asked to fill. mode 0: holds the LOCAL bf16 partial
    /// (the fused kernel reads it); mode 1: unused sink (K2 would have written it).
    pub sink: B,
    /// fp32 local partial for mode 1 (a pool buffer); None for mode 0.
    pub partial: Option<S>,
    /// tp_reduce_resnorm_b `fp32_mode`: 0 = bf16 partials, 1 = fp32 partials.
    pub mode: i32,
}

/// Outcome of a row-parallel reduction at an epilogue site (mixer or FFN).
pub(crate) enum ReduceOut {
    /// Full-rank bf16 result — launch the plain norm kernel, then release the buffer.
    Full(B),
    /// AR landing 2: K1-only ran; launch `tp_reduce_resnorm_b` with the FusedReduce, then release
    /// its buffers (sink, and `partial` when present).
    Fused(FusedReduce),
}

/// Logits buffer provenance for the verify path. hy_v3's `enable_lm_head_fp32` contract requires
/// the GREEDY verify's argmax to come from unrounded fp32 logits (a bf16 round there flips
/// near-ties vs the fp32 decode — a silent losslessness hole); qwen stays bf16 end-to-end.
pub enum VerifyLogits {
    B16(B),
    F32(S),
}

impl VerifyLogits {
    /// Unwrap the bf16 arm (callers that asked for the default/bf16 mode).
    pub fn into_b16(self) -> B {
        match self {
            VerifyLogits::B16(b) => b,
            VerifyLogits::F32(_) => panic!("fp32 logits reached a bf16-only consumer"),
        }
    }
}

/// E16/E3b scratch for the device-only grouped-MoE paths (see moe_batch). All buffers are
/// written before they are read within one dispatch (count is memset per dispatch — it
/// accumulates via atomicAdd; perm_tok gets the -1 padding fill per dispatch).
pub struct MoeGroupedScratch {
    count: cudarc::driver::CudaSlice<i32>,     // [ne] per-expert pair counts (atomicAdd target)
    poff: cudarc::driver::CudaSlice<i32>,      // [ne+2] padded prefix offsets; poff[ne] = routed total,
                                               // poff[ne+1] = + the E12 shared region (fold path only)
    cursor: cudarc::driver::CudaSlice<i32>,    // [ne] scatter cursors (= poff at dispatch start)
    perm_tok: cudarc::driver::CudaSlice<i32>,  // [ppad_max] token per padded row (-1 = padding)
    perm_wt: cudarc::driver::CudaSlice<f32>,   // [ppad_max] combine weight per padded row
    inv_pos: cudarc::driver::CudaSlice<i32>,   // [p_max] padded row per pair (-1 = remote expert)
    tile_e: cudarc::driver::CudaSlice<i32>,    // [ppad_max/8+1] expert id per 8-token n-tile
    p_max: usize,                              // pair bound this instance was sized for
    ppad_max: usize,                           // static upper bound: p_max + ne*16 (full padding)
    /// DSV4 R3.1 (L1) persistent MoE workspace — Some only for the DSV4 builder (qwen leaves
    /// it None and its path is untouched). See `Dsv4MoeWorkspace`.
    pub dsv4_ws: Option<Box<Dsv4MoeWorkspace>>,
}

impl MoeGroupedScratch {
    /// The DSV4 persistent workspace (present iff built by `new_moe_grouped_scratch_raw`).
    pub fn dsv4_ws_mut(&mut self) -> &mut Dsv4MoeWorkspace {
        self.dsv4_ws.as_mut().expect("dsv4 moe workspace (built by new_moe_grouped_scratch_raw)")
    }
    pub fn dsv4_ws_ref(&self) -> &Dsv4MoeWorkspace {
        self.dsv4_ws.as_ref().expect("dsv4 moe workspace (built by new_moe_grouped_scratch_raw)")
    }
    /// Pair bound this instance was sized for (batch·topk per dispatch).
    pub fn p_max(&self) -> usize {
        self.p_max
    }
}

fn new_moe_grouped_scratch(dev: &Arc<CudaDevice>, cfg: &Config, p_max: usize) -> MoeGroupedScratch {
    let ne = cfg.num_experts;
    // E23: 32-token grouped prefill tiles need 32-padded experts. E12's shared-expert region
    // (moe_offsets_fold_b appends it after the routed band; poff grows to ne+2) holds pad32(ns)
    // rows for the whole dispatch's tokens — pad32(p_max/k), the worst case (verify: 32).
    let ppad_max = if ne > 0 {
        p_max + ne * 32 + ((p_max / cfg.num_experts_per_tok.max(1) + 31) / 32) * 32
    } else {
        p_max   // dense models: num_experts_per_tok == 0 — the old allocation, no E12 region
    };
    MoeGroupedScratch {
        count: dev.alloc_zeros::<i32>(ne).unwrap(),
        poff: dev.alloc_zeros::<i32>(ne + 2).unwrap(),
        cursor: dev.alloc_zeros::<i32>(ne).unwrap(),
        perm_tok: dev.alloc_zeros::<i32>(ppad_max).unwrap(),
        perm_wt: dev.alloc_zeros::<f32>(ppad_max).unwrap(),
        inv_pos: dev.alloc_zeros::<i32>(p_max).unwrap(),
        tile_e: dev.alloc_zeros::<i32>(ppad_max / 8 + 1).unwrap(),
        p_max,
        ppad_max,
        dsv4_ws: None,
    }
}

pub const IGS_HIST_BINS: usize = 512;
pub const IGS_HIST_LOG2_MIN: f32 = -40.0;
pub const IGS_HIST_LOG2_MAX: f32 = 40.0;

pub(crate) struct IgsTapBuffers {
    stats: cudarc::driver::CudaSlice<u64>,
    running_max: cudarc::driver::CudaSlice<u32>,
}

/// Mergeable activation statistics for one physical NVFP4 GEMM input.
#[derive(Clone, Debug)]
pub struct IgsHistogram {
    pub bins: Vec<u64>,
    pub running_max: f32,
    pub zero_blocks: u64,
    pub invalid_blocks: u64,
}
pub struct GpuModel {
    dev: Arc<CudaDevice>,
    blas: CudaBlas,
    stream: cudarc::driver::CudaStream,
    cfg: Config,
    embed: W,
    lm_head: Option<W>,  // None = tied (use embed)
    /// TP=2 vocab-parallel: set by `tp_shard_weights` when `lm_head` was col-sharded to this rank's
    /// half of the vocab rows [rank*v/2, +v/2). `logits_batch` then computes only this half and
    /// gathers the peer's via one zero-fill + sum all-reduce (bitwise-identical to the full head).
    lm_head_sharded: bool,
    /// TP=2 shard-at-load (hy_v3): the weights were sharded to this rank INSIDE the loader
    /// (`load_from_dir_tp`), because the full 169.6 GB model does not fit one node for the
    /// post-load in-place shard. `attach_tp` must skip `tp_shard_weights` (and verifies the
    /// mixer-shard flag agrees with what the loader did).
    tp_sharded_at_load: bool,
    /// 4-bit KV cache ([kv-quant]=1): K/V are stored per-16-element affine-packed blocks
    /// (9 B per 16 elems = 3.56x fewer KV bytes). Deterministic per position — decode/verify/
    /// prefill all dequantize identically, so the lossless-MTP contract is preserved. The KV
    /// read at long context (~25 ms/token at 16K bf16) shrinks ~3.5x.
    kv_quant: bool,
    /// TurboQuant KV (E4, [kv-tq]=1): 3.5-bit pack-at-write rotated-domain cache (52-B K/V
    /// rows, hd=128) with the dual-dot QJL score (see HY3_TURBOQUANT_KV_PLAN.md). Mutually
    /// exclusive with kv_quant (asserted at load). Deterministic per position — the q4 lossless
    /// contract carries over by construction (one block per row, fixed reductions).
    kv_tq: bool,
    /// TurboQuant 3-bit-K variant ([kv-tq]=3): K codes at 3 bits (the b=3 Lloyd–Max
    /// codebook, 48 B) -> 68-B K rows; QJL signs + fp16 norms retained; V rows unchanged (50
    /// meaningful B, padded to the shared row size). Loads the -DTQ_B3 kernel build
    /// (src/ptx/gpu_batch_b3.ptx). The b=2 layout ([kv-tq]=1) stays golden-anchored.
    kv_tq_b3: bool,
    /// k8v4 KV ([kv-k8v4]=1): int8 K per-16 affine blocks with an fp16 scale (20 B/16 stride,
    /// 160 B per (head, pos) at hd 128) + the q4 V layout reused byte-for-byte (96 B). Mutually
    /// exclusive with kv_quant and kv_tq (asserted at load). K and V caches diverge in row size
    /// for the first time — every size site uses the per-channel row bytes (kv_k_row_bytes /
    /// kvq_row_bytes). The K readers dequantize via the fp32(fp16 scale) path (u32 int8 loads);
    /// the V channel is byte-for-byte the q4 signed-nibble layout (EXPERT_KV_K8V4_RESPONSE.md §3).
    kv_k8v4: bool,
    /// int8 K+V per-16-block cache ([kv-k8v8]=1 / --kv-cache k8v8): both channels the k8v4
    /// K layout (16 B codes + 2 B fp16 scale, 20 B/16). The decode/verify path reads it
    /// DIRECTLY (gqa_attn_verify_e_k8v8 — the p8b kernel); prefill dequants to bf16 scratch
    /// (dequant_kv_k8v4, channel-agnostic) like the other packed modes.
    kv_k8v8: bool,
    /// TQ host tables (uploaded once, the cos/sin-table pattern): a single f32 buffer
    /// [Pi | PiT | S | ST | cb2(4) | cb3(8) | qjl_scale] — see build_tq_tables and the TQ_TAB_*
    /// offsets in gpu_batch.cu.
    tq_tables: S,
    /// MXFP4-native serving mode (`--mxfp4=on`, QWEN_MXFP4_NATIVE_DESIGN.md): the fp4 GEMMs at
    /// batch <= MAX_VERIFY run on the sm_121a OMMA path instead of the dequantize-to-bf16 chain.
    /// None = default chain, byte-identical contract (the mode is opt-in, off by default).
    mxfp4: Option<crate::mxfp4::Mxfp4State>,
    final_norm: S,
    /// qwen4_exp: the trunk's final hyper-connection mixer (replaces the final norm: streams → hidden).
    hc_mixer: Option<GpuHc>,
    layers: Vec<GpuLayer>,
    /// `--gptq`: armed Hessian accumulators (see `GptqTap`); None in every other mode.
    pub(crate) gptq_tap: std::sync::Mutex<Option<GptqTap>>,
    /// MR-GPTQ artifacts (`quantization_config.transform = hadamard16`): the NVFP4 weights whose
    /// input activations are micro-rotated (H16/4 per 16-block) before the GEMM, by qweight pointer.
    pub(crate) rotated: std::collections::HashSet<u64>,
    /// Rotation scratch for the decode-sized GEMMs (graph-capturable: fixed address).
    pub(crate) rot_small: parking_lot::Mutex<Option<B>>,
    /// NVFP4 W4A4 prefill (src/w4a4.rs, --w4a4-prefill): None unless requested.
    pub(crate) w4a4: Option<crate::w4a4::W4a4State>,
    /// `{stem}.input_global_scale` values read from the artifact (compressed-tensors), by stem.
    pub(crate) igs_by_name: std::collections::HashMap<String, f32>,
    /// `--calib-igs`: armed per-block activation histograms by NVFP4 qweight pointer (prefill only).
    pub(crate) igs_tap: std::sync::Mutex<Option<std::collections::HashMap<u64, IgsTapBuffers>>>,
    mtp: Option<GpuMtpLayer>,
    k: KernelTable,
    bk: HashMap<String, CudaFunction>,
    /// P4 B3: raw CUfunction for gdn_chunk_prefill_b (86KB smem opt-in; cuLaunchKernel path).
    gdn_chunk_raw_fn: Option<cudarc::driver::sys::CUfunction>,
    gdn_chunk_tc_raw_fn: Option<cudarc::driver::sys::CUfunction>,
    cos_table: S,
    sin_table: S,
    // Persistent staging buffers for the MTP hot path (verify_forward). Preallocated once so the
    // steady-state loop does ZERO cuMemAllocAsync / CudaSlice-drop — eliminates the buffer-lifetime
    // race where a drop (cuMemFreeAsync on the device stream) was unordered w.r.t. in-flight compute
    // kernels (see EXPERT_MTP_NONDETERMINISM.md).
    sc_pos: cudarc::driver::CudaSlice<i32>,      // [MAX_VERIFY] verify KV-cache positions (write offset + causal bound)
    sc_rope: cudarc::driver::CudaSlice<i32>,     // [MAX_VERIFY] verify ROPE positions. == sc_pos for a
                                                 // chain; a TREE gives siblings the same rope pos (tree
                                                 // depth) but DISTINCT sc_pos (KV slot). See tree drafting.
    sc_slot: cudarc::driver::CudaSlice<i32>,     // [MAX_VERIFY] verify slot_ids (n = depth)
    sc_winsrc: cudarc::driver::CudaSlice<i32>,   // [MAX_VERIFY*8] conv window sources per verify column
                                                 // (tree drafting; chain-identity today). See conv1d_prefill.
    sc_parent: cudarc::driver::CudaSlice<i32>,   // [MAX_VERIFY] GDN scan parent per verify column (tree; chain today)
    sc_path: cudarc::driver::CudaSlice<u8>,      // [MAX_VERIFY*MAX_VERIFY] rank->slot map per verify column (tree; chain=identity)
    sc_tok: cudarc::driver::CudaSlice<i32>,      // [MAX_VERIFY] verify/argmax token ids
    sc_pstart: cudarc::driver::CudaSlice<i32>,   // [MAX_VERIFY] FOREST per-column prefix boundary (lane committed len)
    moe_ids: cudarc::driver::CudaSlice<i32>,      // [16*MAX_VERIFY] MoE router top-k expert ids (capture-safe scratch)
    moe_wts: cudarc::driver::CudaSlice<f32>,      // [16*MAX_VERIFY] MoE router renormalized top-k weights
    /// E16: device-only grouped-MoE scratch for the verify-width regime (2..=MAX_VERIFY tokens).
    /// Preallocated at load (capture-safe) — the host NEVER reads these back, so the dispatch has no
    /// dtoh sync anywhere: the grouped GEMM's grid is the static upper bound and each block
    /// early-exits against poff[ne] read ON DEVICE. Sized for MAX_VERIFY·k pairs + full padding.
    moe_g: MoeGroupedScratch,
    /// E3b: same device-only scratch at PREFILL scale (PREFILL_CHUNK·k pairs + full padding) — the
    /// grouped prefill path (batch >= 128) runs the identical zero-sync sequence, which removes the
    /// per-MoE-layer per-window full pipeline drain (sync_stream + dtoh of poff) the old host-sized
    /// dispatch paid. Metadata only (~1 MB); the big bf16 workspaces stay pool-allocated per window.
    moe_g_pf: MoeGroupedScratch,
    sc_i1a: cudarc::driver::CudaSlice<i32>,      // [1] mtp_draft_step token
    sc_i1b: cudarc::driver::CudaSlice<i32>,      // [1] mtp_draft_step pos
    // Same discipline for the STOCHASTIC verify (spec_verify_b). This path was allocating and
    // dropping six small buffers per decode step and uploading six arrays with synchronous
    // NULL-stream copies; against a BLOCKING compute stream each of those is a full serialization
    // barrier, and together they cost ~10 ms/step — more than the GDN rollback, the draft and the
    // re-prime combined. Packed and preallocated, the steady-state step now does zero alloc/free.
    sv_pf: cudarc::driver::CudaSlice<f32>,  // [3*MAX_VERIFY] temps | top_ps | draft_qprobs
    sv_ki: cudarc::driver::CudaSlice<i32>,  // [2*MAX_VERIFY] draft_tokens | top_ks
    sv_sd: cudarc::driver::CudaSlice<u32>,  // [MAX_VERIFY] per-column seeds
    // S5F2 real-q verify: the per-column candidate (token, q) table for spec_verify_realq_b's
    // exact relu(p - q) residual (the SGLang speculative_sampling_classic_kernel semantics).
    // Packed as u64 pairs ((q bits << 32) | tok) — ONE pointer keeps the kernel launch at the
    // 12-arg cudarc cap.
    sv_cand: cudarc::driver::CudaSlice<u64>, // [MAX_VERIFY*16] packed (q<<32)|tok pairs
    sv_p:  cudarc::driver::CudaSlice<f32>,  // [MAX_VERIFY] out: p_of_draft
    sv_r:  cudarc::driver::CudaSlice<i32>,  // [MAX_VERIFY] out: resid_tok (bonus at depth-1)
    // S5F3 draft-parity dump: the top-20 (token, renorm-p) table out + the per-column logits
    // scratch for `df2_topk20_dump_b` (dump-only; allocated on first use when the step dump is
    // armed — OnceLock keeps the two constructor init-lists untouched).
    sv_t20: std::sync::OnceLock<cudarc::driver::CudaSlice<u64>>,   // [MAX_VERIFY * 20] packed (p<<32)|tok
    sv_t20_scratch: std::sync::OnceLock<cudarc::driver::CudaSlice<half::bf16>>, // [MAX_VERIFY * vocab]
    /// W2 (Phase 13): per-POSITION schema-mask scratch for the verify paths
    /// ([MAX_VERIFY][mask_words_for(vocab)] u32 + [MAX_VERIFY] i32 flags). The two AtomicU64s are
    /// the ACTIVE-mask pointers (0 = none) so the struct stays Sync; the buffers themselves are
    /// allocated with the model (a few hundred KB) and only written when a schema is armed.
    vmask_words: cudarc::driver::CudaSlice<u32>,
    vmask_flags: cudarc::driver::CudaSlice<i32>,
    vmask_words_ptr: std::sync::atomic::AtomicU64,
    vmask_flags_ptr: std::sync::atomic::AtomicU64,
    mr_tok: cudarc::driver::CudaSlice<i32>, // [MAX_VERIFY] batched re-prime tokens
    mr_pos: cudarc::driver::CudaSlice<i32>, // [MAX_VERIFY] batched re-prime positions
    /// E13 verify-graph: persistent per-replay inputs/outputs for the captured chain verify.
    /// `verify_params` = [pos_start, slot, n, ck] consumed by verify_chain_params_b (the on-device
    /// expansion of the six chain-identity arrays — no htod inside the graph); `verify_resid` is the
    /// backbone hidden the captured core builds in place (the residual the caller re-primes from).
    verify_params: cudarc::driver::CudaSlice<i32>,  // [4]
    verify_resid: B,                                // [resid_width() * MAX_VERIFY] — rw-wide on qwen4_exp
    /// Captured chain-verify graphs, keyed by (width n, ckpt_slot, has_penalty, state graph_epoch).
    /// Capture happens once per key (the eager call the caller already made serves as warmup; the
    /// capture pass itself only RECORDS). Replay writes params+tokens to the persistent bufs and
    /// launches. The graph_epoch term ties each capture to the `BatchGpuState` it was captured
    /// against: the captured kernel args embed that state's device pointers, so replaying across a
    /// dropped state is a use-after-free (B8 blocker B).
    verify_graphs: std::sync::Mutex<std::collections::HashMap<(usize, usize, bool, u64), CudaGraph>>,
    /// E0: per-state GDN slot-pointer tables feeding the fused `gdn_rollback_b` rollback kernel —
    /// [n_gdn] conv_state base pointers then [n_gdn] s_state base pointers, rank-local. Keyed by
    /// `graph_epoch` (the same lifetime discipline as `verify_graphs`: the table embeds the state's
    /// device pointers, so it must never outlive-select a dropped state).
    gdn_slot_tables: std::sync::Mutex<std::collections::HashMap<u64, (cudarc::driver::CudaSlice<u64>, i32)>>,
    /// bf16 scratch for dequantize-then-cuBLAS on the PREFILL path. Grown to the largest tensor.
    deq_scratch: std::sync::Mutex<Option<(B, usize)>>,
    /// F8 parity lane (2026-09-07): W8A8 e4m3 prefill GEMM state.
    /// `fp8e_map`: W::Fp8Blk packed-storage ptr -> e4m3 k32 A-frag panels (one-time on-device
    /// repack by fp8_repack_e_b; the decode kernel keeps reading the ORIGINAL k16 layout).
    /// `fp8e_scratch`: (Bq e4m3 panels, Xs f32 scales) grown to the largest chunk.
    fp8e_map: std::sync::Mutex<std::collections::HashMap<u64, cudarc::driver::CudaSlice<u8>>>,
    fp8e_scratch: std::sync::Mutex<Option<(cudarc::driver::CudaSlice<u8>, cudarc::driver::CudaSlice<f32>)>>,
    /// fp32 split-K partials for `gemm_mma_fp4_splitk_b` (SPLITK_MAX_TILES x SPLITK_MAX_S x 256).
    /// Preallocated at load: the decode/verify path is CUDA-graph captured, where a fresh alloc is
    /// illegal. Fixed address, reused by every split GEMM — the partial+reduce pair is always
    /// back-to-back on one stream, so the scratch is never aliased.
    splitk_partials: S,
    /// R9 (world>2): persistent fp32 upcast scratch for the no-rounding all-reduce in
    /// `tp_all_reduce_bf16`. Allocated at init (before any graph capture) so the mode-2 path is
    /// capture-safe — a per-call Pool alloc inside a captured stream is capture-illegal. Sized for
    /// the widest single-slot reduction (hidden * batch_probe, fp32).
    tp_f32_scratch: S,
    /// A SMALLER LM head, used ONLY to pick draft tokens (FR-Spec). Picking a draft needs an argmax
    /// over the vocabulary -- a second full read of the 572 MB LM head, 11% of a decode step, paid
    /// (depth-1) times per speculative step. That is essentially all of r(d)'s slope. This head holds
    /// only the most frequent ~26% of the vocabulary (plus every special token), so the draft chain
    /// reads a quarter of the bytes.
    ///
    /// The VERIFY keeps the full head, so greedy stays exactly lossless: a token outside the subset is
    /// simply never proposed, which costs acceptance and never correctness.
    draft_head: Option<W>,
    /// Row -> real token id for `draft_head`. Both draft paths already copy the chosen index back to
    /// the host, so the remap is a host-side lookup and costs nothing.
    draft_ids: Vec<u32>,
    /// All-zeros slot ids for every MTP-head forward. The MTP head owns a SINGLE-slot KV cache, so
    /// its `write_kv_b` slot is always 0 — but the kernel takes slot_ids like any other, and the
    /// callers had to hand it one. `new_decode_buffers` hands out the IDENTITY map [0,1,2,...], so a
    /// caller that reached for the nearest slot_ids buffer (as both MTP probes did) made the head
    /// write a whole slot-stride past the end of its cache. That is a silent heap corruption that
    /// only faults when the address happens to be unmapped — which is exactly the "intermittent
    /// CUDA_ERROR_ILLEGAL_ADDRESS" that went unexplained for so long. So the head no longer accepts
    /// a slot_ids pointer at all: it uses this one.
    mtp_sids: cudarc::driver::CudaSlice<i32>,
    // ---- TP=2 (Stage 3). Default single-node: rank 0, world 1. `attach_tp` shards the weights, spawns
    // the RDMA proxy thread, and caches the device pointer to the doorbell ctx here. The forward's
    // `tp_all_reduce_bf16` drives the two-kernel handshake (gate/copy/signal → wait/add) entirely on the
    // GPU stream — no main-thread sync — with the proxy thread doing the verbs.
    //
    // There is deliberately NO host-side epoch counter: the barrier epoch lives in the device ctx and is
    // incremented by K1, so it is never a kernel argument. Capture would freeze an argument; a device
    // counter survives replay, which is what makes graph capture a no-op (round-3 capture hygiene).
    tp_rank: i32,
    tp_world: i32,
    tp_ctx_dptr: u64,
    /// Proof D per-head visit counters (--tp-head-proof only; None otherwise).
    head_visits: Option<cudarc::driver::CudaSlice<u64>>,
    /// E29-B3 DFlash tap staging (rank 0 only, --tp-dflash=1): `forward_batch_dev` and
    /// `verify_forward_core_topo` copy the post-FFN-add residuals of layers {1,20,39,58,77} into
    /// this sink per forward (device-side only, compute stream). None on the node and in every
    /// non-DFlash run — the capture is a strict no-op there (zero kernel or barrier cost).
    dflash_tap: Option<crate::dflash::DflashTapSink>,
    /// S4F DFlash2 tap capture sink (DEFAULT-OFF; `--df2-capture` rides TpConfig, single-process
    /// probes set it via `set_df2_capture`): the trunk copies hidden_states[l+1] for
    /// `dflash2::TAP_LAYERS` = [5,19,33,47,61] into this drafter-owned staging at draft time
    /// only. `None` = the dead path — zero launches, zero host work (the R1 timing-free proof).
    df2_capture: Option<std::sync::Arc<crate::dflash2::capture::Df2TapSink>>,
    /// PLAN/25 Phase 1: WIDE (MAX_VERIFY-col) tap sink for the tree-verify path — the topo
    /// verify captures n > BLOCK columns here; the accepted path is gathered into the round's
    /// 8-col staging by `sync_staging_from_wide`. None unless serving `dflash2-tree`.
    df2_capture_tree: Option<std::sync::Arc<crate::dflash2::capture::Df2TapSink>>,
    /// S5F DFlash2 PROMPT-PRIME sink (DEFAULT-OFF; set via `set_df2_prime_sink`): the PREFILL
    /// path copies the same tap residuals into this wider `[25600, cols]` buffer per window, so
    /// the round's `prime_window` can seed the ctx ring with the PROMPT's taps (the 8-column
    /// staging only ever holds the last verify's span). `None` = dead path.
    df2_prime: Option<std::sync::Arc<crate::dflash2::capture::Df2PrimeSink>>,
    /// WI1 DSpark tap sinks — the DF2 trio, byte-for-byte in structure, at the DSpark tap
    /// layers `dspark::TAP_LAYERS` = [4,16,28,40,52]. All three default to None (dead
    /// branches); the server arms exactly the set the chosen spec-source needs, so DSpark and
    /// DFlash2 taps can coexist (independent buffers, different layer sets) — the acceptance
    /// probe runs both arms back-to-back in one process.
    dspark_capture: Option<std::sync::Arc<crate::dflash2::capture::Df2TapSink>>,
    dspark_capture_tree: Option<std::sync::Arc<crate::dflash2::capture::Df2TapSink>>,
    dspark_prime: Option<std::sync::Arc<crate::dflash2::capture::Df2PrimeSink>>,
    /// E29-B3: the TARGET's FULL-vocab bf16 lm_head [v, h] (rank 0 only, --tp-dflash=1),
    /// captured at load BEFORE the vocab-parallel shard slices it away. The draft path needs the
    /// full head (the rank-local half is vocab-sharded); None on the node (it never drafts).
    dflash_lm_head: Option<B>,
    /// S9F (the TP-DF2 leg): the TARGET's FULL lm_head in the serving dtype (NVFP4/FP8),
    /// kept on EVERY rank when [df2-tp]=1 — captured in `tp_shard_weights` before the
    /// vocab-parallel shard slices the rank-local copy. The DFlash2 round borrows it (the
    /// round's draft head GEMM needs all v rows; the rank-local shard keeps only v/world).
    /// None = the normal (possibly sharded) `lm_head` is used — single-node and non-DF2 TP.
    df2_full_head: Option<W>,
    /// F8-CATCHUP: in-memory NVFP4-MMA copy of a BF16 borrowed head (built lazily on the first
    /// round borrow; see `df2_q_head_borrow`). The verify path NEVER touches this — the serving
    /// `lm_head` stays exactly as loaded, so greedy losslessness is structural. Escape hatch for
    /// A/B: GB10_NO_DF2_QHEAD=1 keeps the bf16 `gemm_binv_b` path.
    df2_q_head: std::sync::OnceLock<Option<W>>,
}

/// Widest verify the persistent stochastic scratch is sized for (matches GEMM_BINV_NMAX).
pub const MAX_VERIFY: usize = 16;

/// W2 one-shot diagnostics (armed-mask shape, printed once each per process).
static MASK_DIAG_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static VMASK_DIAG_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// W2: u32 words needed for one vocabulary bitset (bit t = token t allowed).
#[inline]
pub const fn mask_words_for(vocab: usize) -> usize { (vocab + 31) / 32 }
/// TurboQuant KV row stride (bytes) — the d=128 layout: K row = 52 B, V row = 50 B padded to 52
/// so both caches share one row size (see HY3_TURBOQUANT_KV_PLAN.md + /tmp/tq_ref2/REPORT.md).
/// MUST match `TQ_ROW_BYTES` in kernels/gpu_batch.cu.
pub const TQ_ROW_BYTES: usize = 52;
/// TurboQuant 3-bit-K row stride ([kv-tq]=3): K row = 68 B = codes[0,48) 3-bit |
/// signs[48,64) | rn[64,66) | kn[66,68); V rows stay 50 meaningful B, padded to the shared
/// stride. MUST match `TQ_ROW_BYTES` under -DTQ_B3 in kernels/gpu_batch.cu.
pub const TQ_ROW_BYTES_B3: usize = 68;
/// tq_encode_rows dynamic smem (bytes) — MUST match TQ_ENCODE_SMEM_BYTES in gpu_batch.cu.
pub const TQ_ENCODE_SMEM_BYTES: u32 = 4160;
/// tq_encode_rows dynamic smem for the b=3 build (the K row staging grows 52 -> 68 B).
pub const TQ_ENCODE_SMEM_BYTES_B3: u32 = 4176;
/// dequant_kv_tq dynamic smem (bytes): 3 x hd floats.
pub const TQ_DEQ_SMEM_BYTES: u32 = 3 * 128 * 4;
/// rotate_q_tq dynamic smem (bytes): hd floats.
pub const TQ_ROTATE_SMEM_BYTES: u32 = 128 * 4;

/// KV cache format a given call site uses. TQ (TurboQuant, [kv-tq]=1), q4 ([kv-quant]=1)
/// and k8v4 ([kv-k8v4]=1) are mutually exclusive (asserted at load); the MTP draft cache is
/// always Bf16 (the draft KV stays bf16 by decree — HY3_TURBOQUANT_KV_PLAN.md §6).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KVCacheMode { Bf16, Q4, Tq, K8v4, K8v8 }
/// SM count of the GB10 (DGX Spark) — the "48-SM GPU" of the launch-geometry notes throughout.
/// The persistent-grid serving GEMM (`gemm_mma_fp4_b`) sizes its grid from it.
pub const GB10_SMS: usize = 48;
/// Resident blocks per SM for `gemm_mma_fp4_b`, asserted by its `__launch_bounds__(256, 6)`.
/// The persistent grid launches exactly GB10_SMS * this many blocks (capped at the tile count)
/// so every launch is a single full wave with no partial-wave stranding (AGENTS.md §7, lever 2).
const MMA_FP4_BLOCKS_PER_SM: usize = 6;
/// Split-K bounds for the NVFP4 mma GEMM (gemm_mma_fp4_splitk_b). A shape splits only when its tile
/// count fits the partials buffer: SPLITK_MAX_TILES tiles x SPLITK_MAX_S splits x 256 fp32
/// (the 16x16 fragment slots) = 4.7 MB, preallocated at load so the decode-graph capture path
/// never allocates. Both bounds are functions of the weight shape only — never of N (AGENTS.md §2.4).
const SPLITK_MAX_TILES: usize = 2 * GB10_SMS * MMA_FP4_BLOCKS_PER_SM;
const SPLITK_MAX_S: usize = 8;
/// Query positions per gqa_attn_prefill block == warps per block == hd/32 (the kernel derives QT from
/// blockDim). The warps sweep the SAME keys, so one K/V fetch from L2 serves hd/32 queries (8 at
/// hd=256, 4 at hd=128). Kept for reference; the launch sites compute `hd / 32` directly.
#[allow(dead_code)]
pub const GQA_PF_QT: usize = 8;
/// Query / key tile for the tiled (cuBLAS tensor-core) prefill attention. Sized so S stays a few MB
/// per kv head and the GEMMs stay big enough that launch overhead is noise.
/// Force the scalar `gqa_attn_prefill` instead of the tiled tensor-core path.
///
/// The scalar kernel is the ORACLE. It is slow but it is the reference that the tiled path — and any
/// future mma kernel — must agree with, and it stays reachable on purpose: this kernel has already
/// produced two confident wrong diagnoses, so "run it the simple way and diff" must always be one env
/// var away. `--prefill-scalar=1`.
fn prefill_scalar() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("prefill-scalar")).is_ok())
}

/// --zero-kv=1 — restore the (default-off) full-KV-cache memset on cold admits. Off is the
/// production behavior: attention never reads KV beyond pos, so zeroing it was dead TTFT work.
fn zero_kv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("zero-kv")).is_ok())
}

pub const PF_BR: usize = 1024;
pub const PF_BC: usize = 1024;
/// Below this many columns the tiled path's fixed costs (tile init, per-tile GEMM launches) dominate,
/// so the scalar kernel wins. Prompt prefill and the MTP prompt-prime chunks are far above it.
pub const PF_MIN: usize = 128;

/// Widest N that `profile_mtp` probes. Callers must allocate `2 + PROFILE_MAX_N` state slots: the
/// GDN kernels write one checkpoint slot per verify column, so an N-column probe with a checkpoint
/// at slot 2 touches slots 2..=(2 + N - 2).
pub const PROFILE_MAX_N: usize = 8;

/// How many of the most-frequent token ids the DRAFT head keeps under `--draft-vocab on` (FR-Spec).
/// Measured on prose+code, the top 65536 ids of this BPE vocabulary (26%) cover 97.5% of emitted
/// tokens -- and the tokens they miss are rare AND hard to predict, i.e. exactly the ones the drafter
/// was already getting wrong. Default ON for the Qwen3.5/3.6/3.8 family (quant::DraftVocab::default_for);
/// `--draft-vocab off` builds no draft head.
pub const DRAFT_VOCAB_TOP: usize = crate::quant::DRAFT_VOCAB_ON_ROWS;

/// Depths the auto-policy calibrates and may choose between. Capped at PROFILE_MAX_N because the
/// calibration probe runs a real verify at each of these widths against a `2 + PROFILE_MAX_N`-slot
/// state. Acceptance decays geometrically while a flat verify makes deeper drafts nearly free, so the
/// optimum sits in the middle of this range — it is not a corner.
pub const AUTO_DEPTHS: &[usize] = &[2, 3, 4, 5, 6, 8];
/// Deepest depth the scheduler must be able to serve; sizes the per-lane checkpoint slots.
pub const MAX_AUTO_DEPTH: usize = 8;

/// Process-global generation counter for `BatchGpuState.graph_epoch`. The verify CUDA graph is
/// captured against ONE state's device pointers; folding this generation into the graph key means a
/// graph captured against a dropped state (the τ harness creates a fresh state per rep) can never be
/// selected for replay — stale pointers, use-after-free, OOB (B8 blocker B).
static GRAPH_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Padded verify bucket for a chain-verify width (B8/G1, DSpark K-DSP5). The M<=8 verify runs as a
/// small family of FIXED graph shapes {2,4,6,8} so the confidence-truncated width (k_verify) selects
/// a bucket instead of minting a new shape. Invariant: padding must not change the per-column
/// reduction order -- columns >= the true width are never read (acceptance only consults preds[..n]),
/// and the launch is batch-invariant (probe_binv), so a padded column computes values that nothing
/// observes. `verify_bucket_bitexact` is the gate that proves it on real activations.
#[inline]
pub fn verify_bucket_width(n: usize) -> usize {
    match n {
        0 => 0,
        1..=2 => 2,
        3..=4 => 4,
        5..=6 => 6,
        _ => 8,
    }
}

/// The padded verify buckets, widest first for logging.
pub const VERIFY_BUCKETS: &[usize] = &[8, 6, 4, 2];

/// E17: context lengths the MTP cost calibration measures r(d) at. The verify's KV attention bytes
/// grow ∝ context, so one short-context r(d) table mis-picks the depth at ≥16K. The points are the
/// geometric means of power-of-4 buckets: bucket i covers (point[i-1]×2, point[i]×2] —
/// 2048 → ≤4K, 8192 → 4–16K, 32768 → 16–64K, 131072 → >64K.
pub const MTP_CALIB_CTX: &[usize] = &[2048, 8192, 32768, 131072];

/// The ctx points to actually measure given the server's max_seq_len: the measurement needs a KV
/// cache of that length, so points are capped at max_seq_len; the top measured point doubles as the
/// asymptotic table for everything above it. Never empty.
pub fn mtp_calib_ctx_points(max_seq_len: usize) -> Vec<usize> {
    let mut v: Vec<usize> = MTP_CALIB_CTX.iter().copied().filter(|&p| p <= max_seq_len).collect();
    if v.is_empty() { v.push(max_seq_len.max(1)); }
    v
}

/// vd columns of the GDN recurrent state per block — must match GDN_C in gpu_batch.cu.
///
/// The scan holds S in shared memory for the whole token loop instead of re-reading it from global
/// six times per token. Chunking `vd` is what turns nh (=32) blocks into nh*4 (=128), i.e. a full
/// wave on 48 SMs instead of two thirds of them idle.
const GDN_C: usize = 32;

/// (blocks per (sequence, head), shared bytes) for the GDN scan kernels.
pub fn gdn_launch(kd: usize, vd: usize) -> (usize, u32) {
    assert!(vd % GDN_C == 0, "GDN vd={} must be a multiple of GDN_C={}", vd, GDN_C);
    // S_sh[kd][GDN_C+1] + Srow[kd] + kv_mem/vbuf/delta[GDN_C] + qrow[kd] + krow[kd]
    let floats = kd * (GDN_C + 1) + kd + 3 * GDN_C + 2 * kd;
    (vd / GDN_C, (floats * 4) as u32)
}

// Safe: GpuModel is only used from the single scheduler task (tokio::spawn).
unsafe impl Send for GpuModel {}

/// A weight tensor, in whatever format it was stored. Quantized weights stay PACKED in VRAM — that
/// is the entire point: decode is weight-bandwidth-bound, so fewer bytes per weight IS the speedup.
/// Dequantization happens in registers inside the GEMV.
///
/// Mixed precision per tensor is deliberate. Measured on 9B, the GDN projections are ~3x more
/// perplexity-sensitive per parameter than anything else, while the LM head is the LEAST sensitive —
/// the opposite of the folklore everyone ships. So the default recipe spends 8 bits on GDN and 4 on
/// everything else, including the LM head.
pub enum W {
    Bf16(B),
    Fp8 { data: cudarc::driver::CudaSlice<u8>, row_scale: S, m: usize, k: usize },
    /// Qwen fine-grained FP8 (quant_method "fp8", weight_block_size [128,128]): E4M3 codes
    /// MMA-repacked like `Fp8`, but scales are per-128x128-block `weight_scale_inv` upcast to
    /// f32, stored flat [(m/128) * nb_k], nb_k = k/128. Decode runs `gemm_mma_fp8_blk_b`.
    Fp8Blk { data: cudarc::driver::CudaSlice<u8>, blk_scale: S, nb_k: usize, m: usize, k: usize },
    /// `gs` is ONE reciprocal tensor-scale per 16-row MMA tile, not a scalar. A fused weight stacks
    /// several source tensors along M and each brought its own NVFP4 global scale; every segment
    /// boundary is 16-aligned, so a tile lies wholly inside one segment and the kernel reads its
    /// scale once per block.
    Nvfp4 { qweight: cudarc::driver::CudaSlice<u8>, scales: cudarc::driver::CudaSlice<u8>,
            gs: S, m: usize, k: usize },
    /// RAW (non-MMA-repacked) NVFP4, used ONLY for MoE stacked experts: `qweight` [m, k/2], `scales`
    /// [m, k/16] E4M3, `gs` = 1/global_scale. The scalar `moe_experts_fp4_b` kernel reads this directly
    /// (the MMA repack is for the tensor-core GEMV, which the per-expert MoE path doesn't use).
    Nvfp4Raw { qweight: cudarc::driver::CudaSlice<u8>, scales: cudarc::driver::CudaSlice<u8>,
               gs: f32, m: usize, k: usize },
}

impl W {
    pub fn is_quantized(&self) -> bool { !matches!(self, W::Bf16(_)) }
    /// The packed-storage device pointer (NVFP4 qweight / FP8 data; 0 for bf16) — the key the
    /// mxfp4-native allowlist lookups and the shared-slot descriptor use.
    pub fn packed_ptr(&self) -> u64 {
        match self {
            W::Bf16(_) => 0,
            W::Fp8 { data, .. } => *data.device_ptr() as u64,
            W::Fp8Blk { data, .. } => *data.device_ptr() as u64,
            W::Nvfp4 { qweight, .. } | W::Nvfp4Raw { qweight, .. } => *qweight.device_ptr() as u64,
        }
    }
    /// The legacy single-stream path (`gemm_bf16` / `forward_token`) predates quantization and is not
    /// the serving path. Rather than silently mis-handling a packed weight there, it asserts.
    pub fn bf16(&self) -> &B {
        match self {
            W::Bf16(b) => b,
            _ => panic!("legacy single-stream path cannot use quantized weights — use the batched path"),
        }
    }
    /// Bytes this weight streams per token — the number that sets decode speed.
    pub fn bytes(&self) -> usize {
        match self {
            W::Bf16(_) => 0, // unknown here; only used for reporting on quantized paths
            W::Fp8 { m, k, .. } => m * k + m * 4,
            W::Fp8Blk { m, k, nb_k, .. } => m * k + (m / 128) * nb_k * 4,
            W::Nvfp4 { m, k, .. } => m * k / 2 + m * k / 16 + m / 4,
            W::Nvfp4Raw { m, k, .. } => m * k / 2 + m * k / 16,
        }
    }
}

/// M (output rows) of a packed weight — 0 for bf16, whose shape the slice length can't recover.
fn w_m(w: &W) -> usize {
    match w { W::Nvfp4 { m, .. } | W::Fp8 { m, .. } | W::Fp8Blk { m, .. } | W::Nvfp4Raw { m, .. } => *m, W::Bf16(_) => 0 }
}
/// K (reduction dim) of a packed weight — the dim a ROW-parallel shard shrinks.
fn w_k(w: &W) -> usize {
    match w { W::Nvfp4 { k, .. } | W::Fp8 { k, .. } | W::Fp8Blk { k, .. } | W::Nvfp4Raw { k, .. } => *k, W::Bf16(_) => 0 }
}

pub struct GpuMtpLayer {
    pub fc: W,                         // [hidden, 2*hidden]
    pub pre_fc_norm_hidden: S,         // [hidden]
    pub pre_fc_norm_embedding: S,      // [hidden]
    pub input_ln: S,                   // [hidden]
    pub post_ln: S,                   // [hidden]
    pub fa: GpuFullAttn,               // same structure as main model's full-attn
    pub mlp: Ffn,                      // dense on qwen3_5; MoE on qwen3_5_moe (the MTP FFN is MoE too)
    pub final_norm: S,                 // [hidden]
    /// TP MTP-block sharding (--tp-shard-mtp): per-tensor flags, a property of the TENSOR (the
    /// same rule as `ffn_batch`'s sharded param and `GpuMoe::experts_sharded` — never inferred
    /// from tp_world alone, so a bf16/odd-shape MTP tensor can stay replicated while its siblings
    /// shard). fc shards COLUMN-parallel (an m-slice; the gather in mtp_forward is the same
    /// zero+2D-copy+all-reduce trick logits_batch uses — bitwise-identical to the full GEMM);
    /// attention shards head-parallel + o_proj row-parallel; the dense MLP shards col/row like a
    /// main layer; a MoE MLP rides `experts_sharded` on its own GpuMoe. When all three are false
    /// the head is the classic REPLICATED barrier-free draft (the pre-TP4-campaign default).
    pub fc_sharded: bool,
    pub attn_sharded: bool,
    pub mlp_sharded: bool,             // dense MLP only; the MoE arm rides moe.experts_sharded
    /// qwen4_exp MTP head (Qwen4ExpForCausalLMMTP): the input fusion is per-stream —
    /// streams = fc_hidden(rmsnorm(hidden_streams)) + fc_embedding(rmsnorm(embed)) broadcast —
    /// then ONE hyper-connected decoder layer (attention + MoE) and the head's own mixer. `fc`,
    /// `input_ln`, `post_ln`, `final_norm` are placeholders on this arm.
    pub q4: Option<GpuMtpQ4>,
}

pub struct GpuMtpQ4 {
    pub fc_embedding: W,     // [h, h]
    pub fc_hidden: W,        // [h, h] (applied per stream)
    pub hc: (GpuHc, GpuHc),  // the layer's attn / mlp hyper-connections
    pub mixer: GpuHc,        // streams → hidden for the LM head (no inject)
}

pub struct GpuMlp { pub gate: W, pub up: W, pub down: W }

/// MoE FFN (qwen3_5_moe, hy_v3). Experts are STACKED, gate+up FUSED, exactly as the checkpoint stores them:
///   `gate_up` = `experts.gate_up_proj` [num_experts, 2*moe_inter, hidden]  (flattened to one W)
///   `down`    = `experts.down_proj`    [num_experts, hidden, moe_inter]    (flattened to one W)
///   `router`  = `mlp.gate.weight`      [num_experts, hidden]  (hy_v3: `mlp.router.gate.weight`)
///   `shared`  = a standard MLP (hy_v3: `shared_mlp`).
/// qwen3_5_moe: `shared_gate` = Some([1, hidden]) sigmoid gate, `expert_bias` = None. Forward:
///   softmax→top-k→renorm the k weights, sum the k experts, + sigmoid(shared_gate·h)·shared(h).
/// hy_v3: `shared_gate` = None (the shared expert is UNGATED — a plain add), `expert_bias` = the
///   router's learned per-expert selection bias. Forward: sigmoid(router·h) fp32, top-k selected by
///   score+bias, combine weights the UN-biased sigmoid scores renormalized × cfg.router_scaling,
///   + shared(h). The router kernel (`moe_router_topk*_b`) is the single shared decision point for
///   both expert-GEMM paths (bf16 `moe_experts_b` and NVFP4 `gemm_moe_mma_fp4`).
pub struct GpuMoe {
    pub router: W,
    pub gate_up: W,
    pub down: W,
    pub shared: GpuMlp,
    pub shared_gate: Option<W>,
    /// Optional FUSED `cat(gate_proj, up_proj)` [2*si, hidden] for the shared expert: one GEMM at
    /// [2I, H] instead of two at [I, H] — at Hy3's si=1536 the separate gate/up GEMMs run at
    /// ~115 GB/s (small-M latency); fused they share the experts' fast path. Numerically
    /// identical (same per-row math; `moe_silu_bf16_b` == `silu_mul_b` bit for bit).
    pub shared_gate_up: Option<W>,
    /// hy_v3 only: fp32 [num_experts], added to the sigmoid scores for top-k SELECTION only
    /// (aux-free balancing) — never enters the combine weights.
    pub expert_bias: Option<S>,
    /// TP=2 expert-parallel: set by `tp_shard_weights` when THIS layer's `gate_up`/`down` were halved
    /// along the stacked-expert dim (this rank owns experts [rank·ne/2, rank·ne/2 + ne/2)). Read by
    /// `moe_batch` to (a) remap the router's global expert ids to local ones and zero remote slots,
    /// and (b) all-reduce the expert-combined output before the (replicated) shared-expert add.
    /// A property of the TENSOR, not of the process (see `ffn_batch`): the MTP head's MoE keeps
    /// `false` even under TP=2 — the draft head must stay barrier-free.
    pub experts_sharded: bool,
    /// E8 (revised, shared expert only): TP=2 shared-expert sharding — the fused gate+up
    /// col-parallel (paired segments: each rank owns gate[i]·up[i] for its i-half) + down
    /// row-parallel (each rank computes half of the shared MLP's output). Set at load (hy3) /
    /// attach (qwen, down-only) when the shared weights were actually sliced. When true,
    /// `moe_batch` folds the shared PARTIAL into the routed experts' all-reduce (ONE reduce
    /// restores both sums) instead of reducing the routed output and adding the replicated
    /// shared expert afterwards. The fold reassociates the fp32 sum (the shared add moves
    /// pre-reduce), so it is NOT byte-identical to the replicated path — gated by the
    /// lossless/acceptance battery, revertible by flag (--e8-no-shard=1).
    pub shared_sharded: bool,
    /// E12: the shared slot's weight geometry for the fold kernels, as a device i64[16] written
    /// once at load: [0..4) = gate_up { qweight, scales, gs, (SK<<32)|sk_off } (SK = h, sk_off = 0),
    /// [4..8) = down { qweight, scales, gs, (sdn_k<<32)|rank·sdn_k }, [8] = the shared slot's gu M
    /// (2·si, or 2·si_local = si under the E8 shard), [12] = the down's M (h — uniform no-op).
    /// The fold GEMMs take the descriptor POINTER as their single extra arg (cudarc launch tuples
    /// cap at 12 elements) and read the shared base + per-slot M/K/K-offset from it. None for
    /// models that never fold (qwen's split shared gate/up, bf16 shared tensors, the DSV4
    /// builders).
    pub shared_slot_desc: Option<cudarc::driver::CudaSlice<i64>>,
}

/// A layer's FFN is either the dense MLP (qwen3_5) or an MoE block (qwen3_5_moe). One enum so the four
/// forward sites dispatch through `ffn_batch` and the dense path is untouched.
pub enum Ffn { Dense(GpuMlp), Moe(GpuMoe) }

/// The projections that read the same activation, either FUSED into one weight (the quantized serving
/// path: one GEMM instead of three or four, and the pathological M=32 GDN beta/alpha launches stop
/// existing) or left Split.
///
/// bf16 stays Split. It is not the serving path, and the legacy single-stream forward — which the
/// PyTorch reference tests anchor the whole model on — indexes these tensors individually. Fusing
/// bf16 would mean either duplicating the weights (a 27B bf16 model would not fit) or rewriting that
/// path; neither is worth it for a path that is 3.5x slower than the one we ship.
pub enum AttnIn { Fused(W), Split { q: W, k: W, v: W } }
pub enum GdnIn  { Fused(W), Split { qkv: W, z: W, b: W, a: W } }

pub struct GpuFullAttn { pub qkv: AttnIn, pub o_proj: W, pub q_norm: S, pub k_norm: S,
                         /// qwen4_exp QSA sparse-attention indexer (None = dense attention).
                         pub indexer: Option<GpuIndexer> }
/// qwen4_exp QSA indexer of one full-attention layer (Qwen4ExpTextQSAIndexer): index_qk_proj
/// [(heads+1)*hd_idx, h] → q heads | one raw key, q/k layernorms [hd_idx], and the device `QsaParams`
/// block the kernels read (rope tables, norm weights, eps, geometry — see kernels/gpu_batch.cu).
pub struct GpuIndexer {
    pub qk_proj: W,
    pub q_norm: S,
    pub k_norm: S,
    pub params: cudarc::driver::CudaSlice<u8>,
}
pub struct GpuLinearAttn {
    pub in_proj: GdnIn, pub conv1d: S,
    pub a_log: S, pub dt_bias: S, pub norm: S, pub out_proj: W,
}

impl AttnIn {
    /// Rows of the fused attention weight: [q(|gate, qwen3_5 only) ; k ; v]. hy_v3's q_proj carries
    /// no output gate, so its fused width is nh*hd + 2*nkv*hd.
    pub fn fused_m(cfg: &Config) -> usize {
        let gate = if cfg.attn_out_gate() { 2 } else { 1 };
        cfg.num_heads * cfg.head_dim * gate + 2 * cfg.num_kv_heads * cfg.head_dim
    }
}
impl GdnIn {
    /// Rows of the fused GDN weight: [qkv ; z ; b ; a].
    pub fn fused_m(cfg: &Config) -> usize {
        cfg.key_dim() * 2 + cfg.value_dim() + cfg.value_dim() + 2 * cfg.lin_num_v_heads
    }
}
pub struct GpuLayer {
    pub layer_type: LayerType,
    pub la: Option<GpuLinearAttn>,
    pub fa: Option<GpuFullAttn>,
    pub mlp: Ffn,
    pub input_ln: S,
    pub post_ln: S,
    /// qwen4_exp: the (attention, mlp) hyper-connection gated residuals. None on every other family
    /// (whose `input_ln`/`post_ln` play the pre-norm role; qwen4_exp has NO layer norms at all —
    /// its `input_ln`/`post_ln` are 1-element placeholders never read).
    pub hc: Option<(GpuHc, GpuHc)>,
    /// qwen4_exp: the PLE n-gram injection, on exactly one trunk layer (`cfg.ple_layer`).
    pub ple: Option<GpuPle>,
}

/// qwen4_exp hyper-connection ("gated residual", Qwen4ExpTextGatedResidual). Given the multi-stream
/// residual r [hc*h]:  hn = grouped_rmsnorm(r)·(1+w);  d = silu(down·hn / hc);  u = sigmoid(up·d);
/// x = mean_s(u[s] ⊙ hn[s])  (the sublayer input);  inj[s] = 2·sigmoid(winj[s]·hn / hc)  (the per-stream
/// injection weight: r[s] += inj[s]·sublayer(x)). The final mixer has no `inject`.
pub struct GpuHc {
    pub norm: S,            // [hc*h] f32
    pub down: W,            // [lowrank, hc*h]
    pub up: W,              // [hc*h, lowrank]
    pub inject: Option<B>,  // [hc, hc*h] bf16 (block_inject_weight; M=4 stays bf16)
}

/// Where the PLE n-gram table lives: resident on the device (30.7 GB of 96-B records) or on SSD,
/// read per forward with `pread` (`--ple-offload ssd`).
pub enum PleTable { Device(cudarc::driver::CudaSlice<u8>), Ssd(crate::ple::PleSsd) }

/// qwen4_exp PLE layer (Qwen4ExpTextPLELayer): hashed n-gram rows → key/value projections → a
/// per-stream sigmoid gate against the normed residual → dilated depthwise conv → added to every
/// stream. `stage`/`ids` are the per-forward record staging (sized for the prefill chunk).
pub struct GpuPle {
    pub key_proj: W,        // [hc*h, ple_dim]
    pub value_proj: W,      // [h, ple_dim]
    pub norm_key: S,        // [hc*h]
    pub norm_query: S,      // [hc*h]
    pub norm_conv: S,       // [hc*h]
    pub conv1d: S,          // [hc*h * K] f32 (checkpoint [hc*h, 1, K])
    pub hash: crate::ple::PleHash,
    pub hash_tab: cudarc::driver::CudaSlice<i64>,
    pub table: PleTable,
    pub gs: S,              // [num_shards] reciprocal global scales
    pub rows_per_shard: usize,
    pub stage_rows: usize,  // capacity in tokens
    pub stage: cudarc::driver::CudaSlice<u8>,     // [stage_rows * heads * 96]
    pub ids: cudarc::driver::CudaSlice<i64>,      // [stage_rows * heads]
}
/// Tokens of staging the PLE keeps: the scheduler's PREFILL_CHUNK (8192) — the largest forward.
pub const PLE_STAGE_ROWS: usize = 8192;
/// Red-zone fill for the GDN head-execution proof. A quiet NaN with a recognisable payload, compared
/// BITWISE so that any write — including a NaN write — is detected.
const TP_REDZONE_SENTINEL: u32 = 0x7FBA_DBAD;

#[derive(Default)]
struct KernelTable {
    rmsnorm_qwen: Option<CudaFunction>,
    rmsnorm_gated: Option<CudaFunction>,
    rmsnorm_perhead: Option<CudaFunction>,
    silu_mul: Option<CudaFunction>,
    add_residual: Option<CudaFunction>,
    sigmoid_gate: Option<CudaFunction>,
    rope_rot_half: Option<CudaFunction>,
    gqa_attention: Option<CudaFunction>,
    write_kv: Option<CudaFunction>,
    split_qgate: Option<CudaFunction>,
    conv1d_step: Option<CudaFunction>,
    delta_step: Option<CudaFunction>,
    f32tobf16: Option<CudaFunction>,
    bf16tof32: Option<CudaFunction>,
    fused_residual_rmsnorm: Option<CudaFunction>,
    argmax_pass1: Option<CudaFunction>,
    argmax_pass2: Option<CudaFunction>,
}

pub struct GpuState {
    pub k_cache: Vec<Option<S>>, // full-attn layers
    pub v_cache: Vec<Option<S>>,
    pub conv_state: Vec<Option<S>>,
    pub s_state: Vec<Option<S>>,
    pub pos: usize,
    pub max_seq_len: usize,
}

// F9 xhash call-site tags (thread-local): kept in `mod.rs` because the decode, verify, attention and bench files
// all read them (a private item here is visible to every submodule, so no visibility change was needed).
thread_local! { static F9X_CUR: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
thread_local! { static F9X_ATTN: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

// Used by the MoE forward (`moe_batch.rs`) and the IGS tap (`w4a4.rs`); same reasoning as above.
impl GpuModel {
    #[inline] fn igs_armed(&self) -> bool { self.igs_tap.lock().unwrap().is_some() }
}

// Capture-suppression guard: lives here because `xchain.rs` (defines the sink) and `graphs.rs` (captures) both use it.
/// RAII guard: suppress xchain for the duration of a CUDA-graph capture function (warmup included),
/// restoring on drop. The dtoh the capture sink performs is capture-illegal AND capture-poisoning
/// (the enqueue attempt invalidates the stream even when it early-returns), so suppression must
/// cover the whole function, not just the BeginCapture..EndCapture window.
struct XchainCaptureGuard;
impl XchainCaptureGuard { fn new() -> Self { xchain_set_suppressed(true); Self } }
impl Drop for XchainCaptureGuard { fn drop(&mut self) { xchain_set_suppressed(false); } }
